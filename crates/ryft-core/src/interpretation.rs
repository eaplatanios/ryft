//! Contains machinery for _interpreting_ (i.e., _replaying_) staged [`Program`]s through chosen value semantics.
//!
//! Interpretation walks a [`Program`] in [`Instruction`] order, lifts its stored constants into the active value
//! domain, and sends every operation through [`Context::bind`]. The supplied context determines what replay means:
//! an [`EagerContext`] computes concrete values, a [`StagingContext`](crate::StagingContext) records the replayed
//! work in another program, and a transform context applies batching, differentiation, or partial-evaluation rules.
//!
//! # Entry Points
//!
//! [`Program::interpret`] is the ordinary eager entry point. It instantiates replay with the program's
//! [`EagerContext`], checks the structured input contract, evaluates every instruction, and reconstructs the
//! declared output structure.
//!
//! [`Program::interpret_in_context`] replays the same program through an explicitly supplied [`Context`]. Use it to
//! inline a program into an active trace or to run it through a transform context. [`Program::interpret_with`] is the
//! lower-level fold interface for callers that need custom input feeding or result handling around the same
//! instruction walk.
//!
//! # Operation Interpretation Rules
//!
//! [`InterpretableOperation`] defines eager semantics for one [`Operation`] over a chosen value type and interpretation
//! context. Operand-driven operations may ignore the context. Nullary construction and captured constants request only
//! the narrow context capabilities they actually use. The context parameter `C` is deliberately unbounded on the trait
//! itself. Adding `C: Context` there would make [`EagerContext`]'s `Context` implementation recursively require the
//! very [`InterpretableOperation`] implementation being proven, which can overflow the trait solver. Implementations
//! must therefore place only their actual capability requirements on `C`, such as zero construction or constant
//! materialization, and must not add a blanket [`Context`] bound.
//!
//! # Division of Responsibilities
//!
//! Interpretation owns the program walk, [`Context`] owns each bind's semantics, and operation payloads own their
//! primitive or higher-order execution rules. This division is what makes replay reusable. Replaying through an
//! [`EagerContext`] evaluates, replaying through a [`StagingContext`](crate::StagingContext) records equivalent
//! instructions in another builder, and replaying through a transform context invokes that transform's per-operation
//! rules, which may evaluate, stage, or residualize different portions of the program.
//!
//! # Extending Interpretation
//!
//! Implement [`InterpretableOperation`] on operation payloads using the narrowest context capabilities required by the
//! body. Higher-order operations request nested replay through the instruction-scoped [`InterpretationDriver`]. Wrapper
//! operation enums should dispatch to payload implementations rather than duplicate their execution loops. New replay
//! modes normally do not require a second interpreter. Implement a new [`Context`] whose `bind` method supplies the
//! desired semantics and call [`Program::interpret_in_context`].

use std::fmt::Debug;

use crate::contexts::{Context, Domain, EagerContext};
use crate::macros::check_count;
use crate::parameters::{ParameterError, Parameterized, ParameterizedFamily};
use crate::programs::ProgramError;
use crate::programs::atoms::{Atom, AtomId};
use crate::programs::instructions::Instruction;
use crate::programs::operations::Operation;
use crate::programs::programs::Program;
use crate::programs::regions::{EmptyRegionDriver, RegionDriver, RegionRef, RegionReplayMappings, ReplayRegionDriver};
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::Value;

/// Provides access to attached [`Region`](crate::Region)s during interpretation, scoped to one [`Operation`]
/// application. During [`Program`] replay that application is exactly one [`Instruction`]. Direct rule invocation
/// supplies an equivalent [`RegionDriver`] for that one call. Region-free applications supply a driver with no regions,
/// while region-carrying applications expose their borrowed regions through the same contract. Implementations re-enter
/// the active interpreter directly over borrowed regions, without materializing standalone programs.
pub trait InterpretationDriver<C: Domain>: RegionDriver<C::Value, C::Operation> {
    /// Interprets the [`Region`](crate::Region) at `index` over the provided input values, re-entering the active
    /// program interpreter, and returns the region's output values.
    fn interpret_region(&self, context: &C, index: usize, inputs: Vec<C::Value>)
    -> Result<Vec<C::Value>, ProgramError>;
}

impl<C: Domain> InterpretationDriver<C> for EmptyRegionDriver {
    #[inline]
    fn interpret_region(
        &self,
        _context: &C,
        _index: usize,
        _inputs: Vec<C::Value>,
    ) -> Result<Vec<C::Value>, ProgramError> {
        Err(ProgramError::MalformedProgram("empty region driver cannot interpret a region".to_string()))
    }
}

/// Adapts the [`BindingRegionDriver`](crate::BindingRegionDriver) supplied to one eager [`Context::bind`] operation
/// application to the interpretation-specific recursion provided by [`InterpretationDriver`]. Its regions are exactly
/// the nested computations supplied to that application, and [`EagerInterpretationDriver::interpret_region`] replays a
/// selected region through the [`EagerContext`] without taking ownership of it.
pub(crate) struct EagerInterpretationDriver<'r, D> {
    /// Binding [`RegionDriver`] supplied to the active [`Operation`] application.
    driver: &'r D,
}

impl<'r, D> EagerInterpretationDriver<'r, D> {
    /// Creates a new [`EagerInterpretationDriver`].
    #[inline]
    pub(crate) fn new(driver: &'r D) -> Self {
        Self { driver }
    }
}

impl<V: Value, O: Operation<V::Type>, D: RegionDriver<V, O>> RegionDriver<V, O> for EagerInterpretationDriver<'_, D> {
    #[inline]
    fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, V, O>>
    where
        V: 'r,
        O: 'r,
    {
        self.driver.regions()
    }
}

impl<V: Value, O: Operation<V::Type> + InterpretableOperation<EagerContext<V, O>>, D: RegionDriver<V, O>>
    InterpretationDriver<EagerContext<V, O>> for EagerInterpretationDriver<'_, D>
{
    #[inline]
    fn interpret_region(
        &self,
        context: &EagerContext<V, O>,
        index: usize,
        inputs: Vec<V>,
    ) -> Result<Vec<V>, ProgramError> {
        self.region(index)?.interpret_in_context(context, inputs)
    }
}

/// Represents [`Operation`]s that can be interpreted (i.e., executed) over a chosen value semantics. The interpretation
/// [`Domain`] `C` is the single source of truth for the type, value, and operation families participating in
/// interpretation. The contract deliberately requires only [`Domain`] and not [`Context`] as [`EagerContext`]'s
/// [`Context`] implementation requires `O: InterpretableOperation<Self>`, and making [`Context`] itself reachable from
/// this trait would make that obligation self-referential and overflow the trait solver. Implementations therefore add
/// only the context capabilities they actually consume (e.g., `C: Zero<C::Value>` for nullary construction).
pub trait InterpretableOperation<C: Domain>: Operation<C::Type> {
    /// Interprets this [`Operation`] given the provided input values and returns the resulting output values.
    ///
    /// # Parameters
    ///
    ///   - `context`: [`Context`] providing the value-construction capabilities used by this rule.
    ///   - `driver`: [`InterpretationDriver`] providing [`Instruction`]-scoped access to the application
    ///     [`Region`](crate::Region)s.
    ///   - `inputs`: Input values to interpret this [`Operation`] on.
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError>;
}

impl<
    T: Type,
    V: Value<Type = T>,
    O: Operation<T>,
    Input: Parameterized<V, ParameterStructure: Debug + PartialEq>,
    Output: Parameterized<V>,
> Program<V, O, Input, Output>
{
    /// Interprets/executes this [`Program`] with the provided input. This is the main replay entry point for staged
    /// [`Program`]s. It checks that the provided input value matches the program's expected input structure and type,
    /// evaluates the [`Instruction`]s in order, and finally builds a structured output value from the computed output
    /// values. This is the eager instantiation of [`Self::interpret_in_context`] using this program's own
    /// [`EagerContext`], whose [`bind`](Context::bind) interprets each operation immediately through its
    /// [`InterpretableOperation`] rule and whose [`lift`](Context::lift) is the identity.
    #[inline]
    pub fn interpret(&self, input: Input) -> Result<Output, ProgramError>
    where
        O: InterpretableOperation<EagerContext<V, O>>,
        Input: Parameterized<V, To<V> = Input>,
        Output: Parameterized<V, To<V> = Output>,
    {
        self.interpret_in_context(&EagerContext::<V, O>::new(), input)
    }

    /// Interprets/executes this [`Program`] with the provided input by replaying it through the supplied [`Context`].
    /// Constants are lifted with [`Context::lift`] and each [`Instruction`] is bound with [`Context::bind`]. The
    /// program stays over the context's staged `Constant` representation `V` while values of the context's `Value`
    /// type flow through the replay, so the input and output are the program's `Input` and `Output` reparameterized
    /// at `C::Value`. Because the context owns the semantics of each bind, this gives the eager/staging duality for
    /// free. An eager context (for which `C::Value = V` and this function is [`Self::interpret`]) computes each
    /// operation immediately through its [`InterpretableOperation`] implementation, a staging context records the
    /// replayed operations into the active trace, and a transform context runs its per-operation rules. It checks that
    /// the provided input matches the program's expected input structure and types before replaying.
    ///
    /// This is the plain-program sibling of [`PartialEvaluation::interpret`](crate::PartialEvaluation::interpret),
    /// which additionally wires residual-input feeders, and the transform-aware counterpart of structural relocation
    /// through [`ProgramBuilder::splice_program`](crate::ProgramBuilder::splice_program), which records
    /// [`Instruction`]s directly into a builder without routing through `bind`'s transform interception. Nested program
    /// interpretation (e.g., control flow branches, custom derivative programs, etc.) routes back through here via the
    /// driver behind the rule's [`InterpretationDriver`].
    pub fn interpret_in_context<C: Context<Type = T, Constant = V, Operation = O>>(
        &self,
        context: &C,
        input: Input::To<C::Value>,
    ) -> Result<Output::To<C::Value>, ProgramError>
    where
        O: Clone,
        Input::Family: ParameterizedFamily<C::Value>,
        Output::Family: ParameterizedFamily<C::Value>,
    {
        // Validate that the caller supplied an input with the expected parameter structure.
        let input_structure = input.parameter_structure();
        if input_structure != self.input_structure {
            return Err(ParameterError::MismatchedParameterStructures {
                left_structure: format!("{:?}", self.input_structure),
                right_structure: format!("{input_structure:?}"),
            }
            .into());
        }

        // Flatten the structured input and validate each input value's type.
        let inputs = input.into_parameters().collect::<Vec<_>>();
        for (input, input_id) in inputs.iter().zip(self.input_ids().iter()) {
            let Some(declared) = self.atoms().get(input_id.index()) else {
                return Err(ProgramError::UnboundAtomId { id: *input_id });
            };
            let declared = declared.r#type();
            let actual = input.r#type();
            if !declared.is_refined_by(actual.as_ref()) {
                return Err(TypeError {
                    message: format!(
                        "encountered input type {actual} which is incompatible with the program's \
                        declared type {declared}",
                    ),
                }
                .into());
            }
        }

        // Replay through the context's lift/bind protocol and reshape the flat outputs back into the expected
        // structured output form of this program, reparameterized at the context's value type. All instructions
        // share one mapping scope so that a staging destination imports each unchanged source region at most once.
        let source = self.entry_region_ref();
        let region_mappings = RegionReplayMappings::new();
        let outputs = self.interpret_with(
            inputs,
            |_, constant| context.lift(constant.clone()),
            |instruction, inputs| {
                context.bind(
                    instruction.operation().clone(),
                    ReplayRegionDriver::new(source, instruction.regions(), &region_mappings)?,
                    inputs,
                )
            },
        )?;

        Ok(Output::To::<C::Value>::from_parameters(self.output_structure.clone(), outputs)?)
    }
}

impl<V: Value, O: Operation<V::Type>, Input: Parameterized<V>, Output: Parameterized<V>> Program<V, O, Input, Output> {
    /// Interprets/executes this [`Program`]'s [`Instruction`]s using the caller-supplied value and error semantics.
    /// Transforms and backends specialize this interpretation by choosing a runtime value type `V`, an error type `E`,
    /// a constant-lifting closure `lift_fn`, and an instruction-interpretation closure `interpret_fn`. Inputs and
    /// outputs are flat [`Vec`]s aligned with the program's [`Self::input_ids`] and [`Self::output_ids`]. Structured
    /// input/output handling stays at the call site so that callers can use any parameter family of their choice.
    ///
    /// The `E` type parameter mirrors `V`: a [`Program`] is not tied to a single interpretation, and each
    /// interpretation has its own natural error type. Eager execution interprets instructions into concrete values and
    /// fails with [`ProgramError`], while a backend that lowers each instruction into a compiler IR interprets them
    /// into IR value handles and fails with that backend's own error (e.g., the XLA backend lowers a program into MLIR
    /// values, failing with an MLIR or sharding lowering error). The `E: From<ProgramError>` bound lets one signature
    /// serve both: callers choose the error their closures fail with, and this function's own structural errors (e.g.,
    /// [`ProgramError::UnboundAtomId`] or an input/output count mismatch) fold into that type. A backend error could
    /// instead be boxed into [`ProgramError::Custom`] and recovered by downcasting, but because this function is
    /// already generic over `V`, carrying the matching `E` keeps each interpreter's error statically typed rather than
    /// erasing it to a runtime downcast.
    ///
    /// # Parameters
    ///
    ///   - `inputs`: Flat input values aligned with [`Self::input_ids`].
    ///   - `lift_fn`: Closure that lifts an [`Atom::Constant`]'s carried `V` into the runtime leaf type `Value`. This
    ///     closure receives the constant's [`AtomId`] for callers that surface diagnostics or maintain parallel atom
    ///     tables and is invoked at most once per live constant atom, in atom-index order.
    ///   - `interpret_fn`: Closure that interprets one [`Instruction`]'s [`Operation`] to its already-lifted inputs and
    ///     returns the instruction's outputs. The full [`Instruction`] is provided so that the closure can inspect the
    ///     operation's expected output [`Atom`] IDs when needed (e.g., to look up output [`Type`]s).
    #[inline]
    pub fn interpret_with<
        RuntimeValue: Clone,
        Error: From<ProgramError>,
        LiftFn: FnMut(AtomId, &V) -> Result<RuntimeValue, Error>,
        InterpretFn: FnMut(&Instruction<O>, &[RuntimeValue]) -> Result<Vec<RuntimeValue>, Error>,
    >(
        &self,
        inputs: Vec<RuntimeValue>,
        lift_fn: LiftFn,
        interpret_fn: InterpretFn,
    ) -> Result<Vec<RuntimeValue>, Error> {
        self.entry_region_ref().interpret_with(inputs, lift_fn, interpret_fn)
    }
}

impl<V: Value, O: Operation<V::Type>> RegionRef<'_, V, O> {
    /// Interprets this borrowed [`Region`](crate::Region) through the provided [`Context`] using flat input and output
    /// values. The region and every nested region attached to its [`Instruction`]s are replayed directly from the
    /// source arena. When the provided context stages an unchanged nested region, one replay-scoped mapping preserves
    /// repeated roots and shared descendants in the destination arena.
    pub fn interpret_in_context<C: Context<Type = V::Type, Constant = V, Operation = O>>(
        self,
        context: &C,
        inputs: Vec<C::Value>,
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, self.input_ids().len(), ProgramError);
        for (input, input_id) in inputs.iter().zip(self.input_ids()) {
            let declared =
                self.atoms().get(input_id.index()).ok_or(ProgramError::UnboundAtomId { id: *input_id })?.r#type();
            let actual = input.r#type();
            if !declared.is_refined_by(actual.as_ref()) {
                return Err(TypeError {
                    message: format!(
                        "encountered input type {actual} which is incompatible with the region's declared type \
                         {declared}",
                    ),
                }
                .into());
            }
        }
        let region_mappings = RegionReplayMappings::new();
        self.interpret_with(
            inputs,
            |_, constant| context.lift(constant.clone()),
            |instruction, inputs| {
                context.bind(
                    instruction.operation().clone(),
                    ReplayRegionDriver::new(self, instruction.regions(), &region_mappings)?,
                    inputs,
                )
            },
        )
    }

    /// Interprets/executes this borrowed [`RegionRef`]'s [`Instruction`]s using the caller-supplied value and error
    /// semantics. This is the borrowed-[`Region`](crate::Region) counterpart of [`Program::interpret_with`]. It replays
    /// the region directly from its source arena without first materializing a standalone [`Program`], while preserving
    /// the same flat input, constant-lifting, instruction-dispatch, and output-gathering behavior.
    pub fn interpret_with<
        RuntimeValue: Clone,
        Error: From<ProgramError>,
        LiftFn: FnMut(AtomId, &V) -> Result<RuntimeValue, Error>,
        InterpretFn: FnMut(&Instruction<O>, &[RuntimeValue]) -> Result<Vec<RuntimeValue>, Error>,
    >(
        self,
        inputs: Vec<RuntimeValue>,
        mut lift_fn: LiftFn,
        mut interpret_fn: InterpretFn,
    ) -> Result<Vec<RuntimeValue>, Error> {
        let atoms = self.atoms();
        let input_ids = self.input_ids();
        let instructions = self.instructions();
        let output_ids = self.output_ids();
        check_count!("input", inputs, input_ids.len(), ProgramError);

        // Count every future consumer of each atom, including final region outputs. These counts let us move each
        // value out on its last use and clone it only when a later consumer still needs it.
        let mut remaining_uses = vec![0usize; atoms.len()];
        for instruction in instructions {
            for input_id in instruction.inputs().iter().copied() {
                let Some(remaining_uses) = remaining_uses.get_mut(input_id.index()) else {
                    return Err(ProgramError::UnboundAtomId { id: input_id }.into());
                };
                *remaining_uses += 1;
            }
        }
        for output_id in output_ids.iter().copied() {
            let Some(remaining_uses) = remaining_uses.get_mut(output_id.index()) else {
                return Err(ProgramError::UnboundAtomId { id: output_id }.into());
            };
            *remaining_uses += 1;
        }

        // Store concrete input values in a sparse value table indexed by `AtomId`.
        let mut values = vec![None; atoms.len()];
        for (input_id, input) in input_ids.iter().copied().zip(inputs) {
            let Some(slot) = values.get_mut(input_id.index()) else {
                return Err(ProgramError::UnboundAtomId { id: input_id }.into());
            };
            *slot = Some(input);
        }

        // Materialize literal constants that are live. Dead constants can remain unset because no instruction or
        // region output will read them.
        for (atom_index, atom) in atoms.iter().enumerate() {
            if remaining_uses[atom_index] == 0 {
                continue;
            }
            if let Atom::Constant(value) = atom {
                values[atom_index] = Some(lift_fn(AtomId::new(atom_index), value)?);
            }
        }

        // Replay instructions in region order, reusing one scratch input buffer to avoid per-instruction allocation.
        let max_input_count = instructions.iter().map(|instruction| instruction.inputs().len()).max().unwrap_or(0);
        let mut instruction_inputs = Vec::with_capacity(max_input_count);
        for instruction in instructions {
            instruction_inputs.clear();
            for input_id in instruction.inputs().iter().copied() {
                // Consume the appropriate input value for the current instruction. If this is the last consumer,
                // move the value out of the table. Otherwise, clone it so later consumers can still read it.
                let remaining_uses = remaining_uses.get_mut(input_id.index()).unwrap();
                debug_assert!(*remaining_uses > 0);
                *remaining_uses -= 1;
                let value = values.get_mut(input_id.index()).unwrap();
                let value = if *remaining_uses == 0 { value.take().unwrap() } else { value.as_ref().unwrap().clone() };
                instruction_inputs.push(value);
            }

            // Apply the operation using the supplied dispatcher and ensure it produces the expected number of outputs.
            let outputs = interpret_fn(instruction, instruction_inputs.as_slice())?;
            check_count!("output", outputs, instruction.outputs().len(), ProgramError);

            for (output_id, output) in instruction.outputs().iter().copied().zip(outputs) {
                let Some(value) = values.get_mut(output_id.index()) else {
                    return Err(ProgramError::UnboundAtomId { id: output_id }.into());
                };

                // Keep only outputs with a future consumer. Dead instruction results do not need to occupy the table.
                if remaining_uses[output_id.index()] != 0 {
                    *value = Some(output);
                }
            }
        }

        // Gather the region outputs using the same last-use transfer logic that we used for instruction inputs.
        let mut outputs = Vec::with_capacity(output_ids.len());
        for output_id in output_ids.iter().copied() {
            let remaining_uses = remaining_uses.get_mut(output_id.index()).unwrap();
            debug_assert!(*remaining_uses > 0);
            *remaining_uses -= 1;
            let value = values.get_mut(output_id.index()).unwrap();
            let value = if *remaining_uses == 0 { value.take().unwrap() } else { value.as_ref().unwrap().clone() };
            outputs.push(value);
        }

        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::scalars::{Scalar, ScalarOperation};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::operations::math::{AddOperation, NegOperation};
    use crate::parameters::{ParameterError, Parameterized, Placeholder};
    use crate::programs::types::TypeError;
    use crate::programs::{AtomId, ProgramBuilder, ProgramError};
    use crate::tests::{TestArray, TestRegionOperation};
    use crate::tracing::TracingContext;
    use crate::types::{ArrayType, DataType, Shape, Size};

    use super::*;

    #[test]
    fn test_empty_region_driver_interpret_region() {
        let context = EagerContext::<Scalar>::new();
        let expected = ProgramError::MalformedProgram("empty region driver cannot interpret a region".to_string());
        let expected_region = ProgramError::MalformedProgram("region index 0 is out of range".to_string());
        assert_eq!(RegionDriver::<Scalar, ScalarOperation<Scalar>>::regions(&EmptyRegionDriver).count(), 0);
        assert_eq!(RegionDriver::<Scalar, ScalarOperation<Scalar>>::region_count(&EmptyRegionDriver), 0);
        assert!(matches!(
            RegionDriver::<Scalar, ScalarOperation<Scalar>>::region(&EmptyRegionDriver, 0),
            Err(error) if error == expected_region,
        ));
        assert_eq!(EmptyRegionDriver.interpret_region(&context, 0, Vec::<Scalar>::new()), Err(expected));
    }

    #[test]
    fn test_program_interpret_materializes_duplicate_outputs() {
        // A program whose two outputs are the same atom materializes that value into both output positions.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let i0 = builder.add_input(DataType::F32);
        let o0 = builder.add_instruction(AddOperation, Vec::new(), vec![i0, i0]).unwrap()[0];
        let program = builder
            .build::<Scalar, (Scalar, Scalar)>(vec![o0, o0], Placeholder, (Placeholder, Placeholder))
            .unwrap();
        assert_eq!(program.interpret(Scalar::from(2.0f32)), Ok((Scalar::from(4.0f32), Scalar::from(4.0f32))));
    }

    #[test]
    fn test_program_interpret_in_context_preserves_replayed_region_sharing() {
        let mut nested_builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let nested_input = nested_builder.add_input(DataType::F64);
        let nested = nested_builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![nested_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut source_builder = ProgramBuilder::<Scalar, TestRegionOperation>::new();
        let shared = source_builder.import_program(nested);
        let source_input = source_builder.add_input(DataType::F64);
        let first = source_builder
            .add_instruction(TestRegionOperation::WithRegions(&["body"]), vec![shared], vec![source_input])
            .unwrap()[0];
        let second = source_builder
            .add_instruction(TestRegionOperation::WithRegions(&["body"]), vec![shared], vec![first])
            .unwrap()[0];
        let source = source_builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![second], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let context = TracingContext::<Scalar, TestRegionOperation>::new();
        let input = context.input(DataType::F64);
        let outputs = source.interpret_in_context(&context, vec![input]).unwrap();
        let destination = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Scalar>, Vec<Scalar>>(
                vec![outputs[0].atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(destination.regions().len(), 2);
        assert_eq!(destination.instructions().len(), 2);
        assert_eq!(destination.instructions()[0].regions(), destination.instructions()[1].regions());
        assert_eq!(destination.instructions()[0].regions(), &[crate::RegionId::new(0)]);
    }

    #[test]
    fn test_program_interpret_lifts_live_constants_once() {
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let i0 = builder.add_input(DataType::F64);
        let c0 = builder.add_constant(Scalar::from(7.0f64));
        let c1 = builder.add_constant(Scalar::from(3.0f64));
        let o0 = builder.add_instruction(AddOperation, Vec::new(), vec![i0, c1]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![o0], Placeholder, Placeholder).unwrap();
        let mut lifted_constants = Vec::new();
        assert_eq!(
            program.interpret_with(
                vec![Scalar::from(2.0f64)],
                |atom_id, value| {
                    lifted_constants.push((atom_id, *value));
                    Ok(*value)
                },
                |instruction, inputs| instruction.operation().interpret(
                    &EagerContext::<Scalar>::new(),
                    &EmptyRegionDriver,
                    inputs,
                ),
            ),
            Ok(vec![Scalar::from(5.0f64)]),
        );
        assert_eq!(lifted_constants, vec![(c1, Scalar::from(3.0f64))]);
        assert_eq!(c0, AtomId::new(1));
    }

    #[test]
    fn test_program_interpret_with_mismatched_parameter_structures() {
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let i0 = builder.add_input(DataType::F64);
        let program = builder.build::<Vec<Scalar>, Scalar>(vec![i0], vec![Placeholder], Placeholder).unwrap();
        assert!(matches!(
            program.interpret(vec![Scalar::from(1.0f64), Scalar::from(2.0f64)]),
            Err(ProgramError::Parameter(ParameterError::MismatchedParameterStructures {
                left_structure,
                right_structure,
            })) if left_structure == format!("{:?}", vec![Placeholder])
                && right_structure == format!("{:?}", vec![1.0f64, 2.0f64].parameter_structure())
        ));
    }

    #[test]
    fn test_program_interpret_input_type_checking() {
        // A statically typed program input rejects values whose concrete types do not match it exactly.
        let mut builder = ProgramBuilder::<TestArray, AddOperation>::new();
        let i0 = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)])));
        let o0 = builder.add_instruction(AddOperation, Vec::new(), vec![i0, i0]).unwrap()[0];
        let program = builder.build::<TestArray, TestArray>(vec![o0], Placeholder, Placeholder).unwrap();
        assert_eq!(program.interpret(TestArray::vector(vec![1.0, 2.0])).unwrap().values, vec![2.0, 4.0]);
        assert!(matches!(
            program.interpret(TestArray::vector(vec![1.0, 2.0, 3.0])),
            Err(ProgramError::Type(TypeError { message })) if message
                == "encountered input type f64[3] which is incompatible with the program's declared type f64[2]",
        ));

        // An unbounded dynamically sized program input accepts concrete values of any size, so one staged program
        // replays at several concrete sizes. Rank mismatches are still rejected.
        let mut builder = ProgramBuilder::<TestArray, AddOperation>::new();
        let i0 = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None)])));
        let o0 = builder.add_instruction(AddOperation, Vec::new(), vec![i0, i0]).unwrap()[0];
        let program = builder.build::<TestArray, TestArray>(vec![o0], Placeholder, Placeholder).unwrap();
        assert_eq!(program.interpret(TestArray::vector(vec![1.0, 2.0])).unwrap().values, vec![2.0, 4.0]);
        assert_eq!(program.interpret(TestArray::vector(vec![1.0, 2.0, 3.0])).unwrap().values, vec![2.0, 4.0, 6.0]);
        assert!(matches!(
            program.interpret(TestArray::scalar(1.0)),
            Err(ProgramError::Type(TypeError { message })) if message
                == "encountered input type f64[] which is incompatible with the program's declared type f64[*]",
        ));

        // A bounded dynamically sized program input enforces its exclusive upper bound on concrete sizes.
        let mut builder = ProgramBuilder::<TestArray, AddOperation>::new();
        let i0 = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(Some(3))])));
        let o0 = builder.add_instruction(AddOperation, Vec::new(), vec![i0, i0]).unwrap()[0];
        let program = builder.build::<TestArray, TestArray>(vec![o0], Placeholder, Placeholder).unwrap();
        assert_eq!(program.interpret(TestArray::vector(vec![1.0, 2.0])).unwrap().values, vec![2.0, 4.0]);
        assert!(matches!(
            program.interpret(TestArray::vector(vec![1.0, 2.0, 3.0])),
            Err(ProgramError::Type(TypeError { message })) if message
                == "encountered input type f64[3] which is incompatible with the program's declared type f64[<3]",
        ));
    }

    #[test]
    fn test_program_interpret_with_wrong_number_of_operation_inputs() {
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let i0 = builder.add_input(DataType::F64);
        let program = builder.build::<Scalar, Scalar>(vec![i0], Placeholder, Placeholder).unwrap();
        assert!(matches!(
            program.interpret_with(
                Vec::<Scalar>::new(),
                |_, value| Ok(*value),
                |instruction, inputs| instruction.operation().interpret(
                    &EagerContext::<Scalar>::new(),
                    &EmptyRegionDriver,
                    inputs,
                ),
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        ));
    }

    #[test]
    fn test_program_interpret_with_wrong_number_of_operation_outputs() {
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let i0 = builder.add_input(DataType::F64);
        let o0 = builder.add_instruction(NegOperation, Vec::new(), vec![i0]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![o0], Placeholder, Placeholder).unwrap();
        assert!(matches!(
            program.interpret_with(
                vec![Scalar::from(2.0f64)],
                |_, value| Ok(*value),
                |_, _| Ok::<Vec<Scalar>, ProgramError>(Vec::new()),
            ),
            Err(ProgramError::InvalidOutputCount { expected: 1, actual: 0 }),
        ));
    }
}
