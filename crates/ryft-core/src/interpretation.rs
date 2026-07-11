//! Contains machinery for _interpreting_ (i.e., _replaying_) staged [`Program`]s through chosen value semantics.
//!
//! Interpretation walks a [`Program`] in [`Instruction`] order, lifts its stored constants into the active value
//! domain, and sends every operation through [`Context::bind`]. The supplied context determines what replay means:
//! an eager context computes concrete values, a staging context splices the work into another program, and a transform
//! context applies batching, differentiation, or partial-evaluation rules.
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
//! [`InterpretableProgramOperation`] is the recursive fixed point for operation families containing nested flat
//! programs. It lets a higher-order operation replay its body without requiring the full wrapper operation enum's
//! interpretation implementation while that implementation is still being established. Its separate `Constant`
//! parameter supports nested programs whose stored capture representation differs from the flowing runtime value.
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
//! body. An operation family containing nested programs should implement [`InterpretableProgramOperation`] and keep the
//! recursive replay logic with the higher-order operation that owns the program. Wrapper operation enums should
//! dispatch to payload implementations rather than duplicate their execution loops. New replay modes normally do not
//! require a second interpreter. Implement a new [`Context`] whose `bind` method supplies the desired semantics and
//! call [`Program::interpret_in_context`].

use std::fmt::Debug;

use crate::contexts::{Context, EagerContext};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::parameters::{ParameterError, Parameterized, ParameterizedFamily};
use crate::programs::{Atom, AtomId, Instruction, Program, ProgramError, Value};
use crate::types::{Type, TypeError, Typed};

/// Represents [`Operation`]s that can be interpreted (i.e., executed) over a chosen value semantics. The interpretation
/// [`Context`] `C` is deliberately *unbounded* at the trait level. [`EagerContext`]'s [`Context`] implementation
/// requires `O: InterpretableOperation<V, Self>`, and so any [`Context`] bound reachable from an interpretation
/// implementation would make that obligation self-referential and overflow the trait solver. Implementations therefore
/// bound `C` only by the context capabilities they actually use (e.g., `C: Zero<V>` for nullary construction or
/// `C: Constant<V, Stored, Payload>` for captured-constant materialization), and never by [`Context`] itself.
pub trait InterpretableOperation<V: Value, C>: Operation<V::Type> {
    /// Interprets this [`Operation`] given the provided input values and returns the resulting output values.
    ///
    /// # Parameters
    ///
    ///   - `context`: Interpretation context. Nullary and captured-payload implementations construct values through
    ///     its context capabilities, while operand-driven value semantics ignore it.
    ///   - `inputs`: Input values to interpret this [`Operation`] on.
    fn interpret(&self, context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError>;
}

/// Represents closed [`Operation`] families that can recursively interpret nested flat [`Program`]s. This trait names
/// the recursive fixed point needed by higher-order interpretation helpers without requiring the full operation enum's
/// [`InterpretableOperation`] implementation while proving that implementation. Operation families implement it by
/// replaying nested flat [`Program`]s through their operation-owned interpretation rules.
///
/// The `Constant` parameter is the nested program's constant value type. It defaults to `V`, which covers direct
/// linear programs whose constants are already runtime values. Captured higher-order programs and custom derivative
/// bodies can set `Constant` to their captured constant type and let the implementation decide how to lift those
/// constants into `V`.
pub trait InterpretableProgramOperation<V: Value, C, Constant: Value<Type = V::Type> = V>:
    Operation<V::Type> + Sized
{
    /// Interprets a nested flat [`Program`].
    ///
    /// # Parameters
    ///
    ///   - `context`: Interpretation context to use.
    ///   - `program`: Nested [`Program`] to interpret.
    ///   - `input`: Input values to use for interpreting the provided [`Program`].
    fn interpret_program(
        context: &C,
        program: &Program<Constant, Self, Vec<Constant>, Vec<Constant>>,
        input: Vec<V>,
    ) -> Result<Vec<V>, ProgramError>;
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
        O: Clone + InterpretableOperation<V, EagerContext<V, O>>,
        Input: Parameterized<V, To<V> = Input>,
        Output: Parameterized<V, To<V> = Output>,
    {
        self.interpret_in_context(&EagerContext::<V, O>::new(), input)
    }

    /// Interprets/executes this [`Program`] with the provided input by replaying it through the supplied [`Context`].
    /// Constants are lifted with [`Context::lift`] and each [`Instruction`] is bound with [`Context::bind`]. The
    /// program stays over the context's staged [`Constant`](crate::Domain::Constant) representation `V` while values
    /// of the context's [`Value`](crate::Domain::Value) type flow through the replay, so the input and output are the
    /// program's `Input` and `Output` reparameterized at `C::Value`. Because the context owns the semantics of each
    /// bind, this gives the eager/staging duality for free. An eager context (for which `C::Value = V` and this
    /// function is [`Self::interpret`]) computes each operation immediately through its [`InterpretableOperation`]
    /// implementation, a staging context splices the program into the active trace, and a transform context runs its
    /// per-operation rules. It checks that the provided input matches the program's expected input structure and types
    /// before replaying.
    ///
    /// This is the plain-program sibling of [`PartialEvaluation::interpret`](crate::PartialEvaluation::interpret),
    /// which additionally wires residual-input feeders, and the transform-aware counterpart of
    /// [`StagingContext::stage_program`](crate::StagingContext::stage_program), which records instructions directly
    /// into a builder without routing through `bind`'s transform interception. Nested program interpretation (e.g.,
    /// control flow branches, custom derivative programs, etc.) does *not* route through here. Instead, it goes
    /// through the [`InterpretableProgramOperation`] witness, whose interpretation context is deliberately not
    /// [`Context`]-bounded.
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
        for (input, input_id) in inputs.iter().zip(self.input_ids.iter()) {
            let Some(declared) = self.atoms.get(input_id.index()) else {
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
        // structured output form of this program, reparameterized at the context's value type.
        let outputs = self.interpret_with(
            inputs,
            |_, constant| context.lift(constant.clone()),
            |instruction, inputs| context.bind(instruction.operation().clone(), inputs),
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
    pub fn interpret_with<
        Value: Clone,
        Error: From<ProgramError>,
        LiftFn: FnMut(AtomId, &V) -> Result<Value, Error>,
        InterpretFn: FnMut(&Instruction<O>, &[Value]) -> Result<Vec<Value>, Error>,
    >(
        &self,
        inputs: Vec<Value>,
        mut lift_fn: LiftFn,
        mut interpret_fn: InterpretFn,
    ) -> Result<Vec<Value>, Error> {
        check_count!("input", inputs, self.input_ids.len(), ProgramError);

        // Count every future consumer of each atom, including final program outputs. These counts let us move each
        // value out on its last use and clone it only when a later consumer still needs it.
        let mut remaining_uses = vec![0usize; self.atoms.len()];
        for instruction in self.instructions.iter() {
            for input_id in instruction.inputs().iter().copied() {
                let Some(remaining_uses) = remaining_uses.get_mut(input_id.index()) else {
                    return Err(ProgramError::UnboundAtomId { id: input_id }.into());
                };
                *remaining_uses += 1;
            }
        }
        for output_id in self.output_ids.iter().copied() {
            let Some(remaining_uses) = remaining_uses.get_mut(output_id.index()) else {
                return Err(ProgramError::UnboundAtomId { id: output_id }.into());
            };
            *remaining_uses += 1;
        }

        // Store concrete input values in a sparse value table indexed by [`AtomId`].
        let mut values = vec![None; self.atoms.len()];
        for (input_id, input) in self.input_ids.iter().copied().zip(inputs) {
            let Some(slot) = values.get_mut(input_id.index()) else {
                return Err(ProgramError::UnboundAtomId { id: input_id }.into());
            };
            *slot = Some(input);
        }

        // Materialize literal constants that are live. Dead constants can remain unset because no instruction or
        // program output will read them.
        for (atom_index, atom) in self.atoms.iter().enumerate() {
            if remaining_uses[atom_index] == 0 {
                continue;
            }
            if let Atom::Constant(value) = atom {
                values[atom_index] = Some(lift_fn(AtomId::new(atom_index), value)?);
            }
        }

        // Replay instructions in program order, reusing one scratch input buffer to avoid per-instruction allocation.
        let max_input_count = self.instructions.iter().map(|instruction| instruction.inputs().len()).max().unwrap_or(0);
        let mut instruction_inputs = Vec::with_capacity(max_input_count);
        for instruction in self.instructions.iter() {
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

        // Gather the program outputs using the same last-use transfer logic that we used for the instruction inputs.
        let mut outputs = Vec::with_capacity(self.output_ids.len());
        for output_id in self.output_ids.iter().copied() {
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
    use crate::contexts::EagerContext;
    use crate::operations::arithmetic::{AddOperation, NegOperation};
    use crate::parameters::{ParameterError, Parameterized, Placeholder};
    use crate::programs::{AtomId, ProgramBuilder, ProgramError};
    use crate::tests::TestArray;
    use crate::types::{ArrayType, DataType, Shape, Size, TypeError};

    use super::*;

    #[test]
    fn test_program_interpret_materializes_duplicate_outputs() {
        // A program whose two outputs are the same atom materializes that value into both output positions.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let i0 = builder.add_input(DataType::F32);
        let o0 = builder.add_instruction(AddOperation, vec![i0, i0]).unwrap()[0];
        let program = builder
            .build::<Scalar, (Scalar, Scalar)>(vec![o0, o0], Placeholder, (Placeholder, Placeholder))
            .unwrap();
        assert_eq!(program.interpret(Scalar::from(2.0f32)), Ok((Scalar::from(4.0f32), Scalar::from(4.0f32))));
    }

    #[test]
    fn test_program_interpret_lifts_live_constants_once() {
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let i0 = builder.add_input(DataType::F64);
        let c0 = builder.add_constant(Scalar::from(7.0f64));
        let c1 = builder.add_constant(Scalar::from(3.0f64));
        let o0 = builder.add_instruction(AddOperation, vec![i0, c1]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![o0], Placeholder, Placeholder).unwrap();
        let mut lifted_constants = Vec::new();
        assert_eq!(
            program.interpret_with(
                vec![Scalar::from(2.0f64)],
                |atom_id, value| {
                    lifted_constants.push((atom_id, *value));
                    Ok(*value)
                },
                |instruction, inputs| instruction.operation().interpret(&EagerContext::<Scalar>::new(), inputs),
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
        let o0 = builder.add_instruction(AddOperation, vec![i0, i0]).unwrap()[0];
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
        let o0 = builder.add_instruction(AddOperation, vec![i0, i0]).unwrap()[0];
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
        let o0 = builder.add_instruction(AddOperation, vec![i0, i0]).unwrap()[0];
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
                |instruction, inputs| instruction.operation().interpret(&EagerContext::<Scalar>::new(), inputs),
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        ));
    }

    #[test]
    fn test_program_interpret_with_wrong_number_of_operation_outputs() {
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let i0 = builder.add_input(DataType::F64);
        let o0 = builder.add_instruction(NegOperation, vec![i0]).unwrap()[0];
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
