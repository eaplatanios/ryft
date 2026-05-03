use std::cell::RefCell;
use std::rc::Rc;

use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::SupportsAdd;
use crate::operations::constants::SupportsZero;
use crate::parameters::Parameterized;
use crate::tracing::engines::{Tracer, TracingContext, TracingEngine};
use crate::tracing::{AtomId, Instruction, Program, ProgramBuilder, Traceable, TracingError};
use crate::types::{Type, Typed};

/// Context that is used while _transposing_ [`Program`](crate::Program)s. This context is threaded through the
/// transposition transformation using [`LinearOperation::transpose`]. It owns the active [`ProgramBuilder`] that is
/// used for building the transposed [`Program`](crate::Program).
pub struct TranspositionContext<T: Type, V: Traceable<T>, O: Clone + Operation<T>> {
    /// [`ProgramBuilder`] that owns the reverse linear [`Program`](crate::tracing::Program) currently being staged.
    pub builder: Rc<RefCell<ProgramBuilder<T, V, O>>>,
}

impl<T: Type, V: Traceable<T>, O: Clone + Operation<T>> TranspositionContext<T, V, O> {
    /// Creates a new [`TranspositionContext`] that stages into the provided [`ProgramBuilder`].
    ///
    /// # Parameters
    ///
    ///   - `builder`: Shared builder that will own the staged reverse linear program.
    pub fn new(builder: Rc<RefCell<ProgramBuilder<T, V, O>>>) -> Self {
        Self { builder }
    }

    /// Stages `operation` in the active transpose builder and returns its output atoms.
    ///
    /// Output types are inferred with [`Operation::infer_output_types`] from the current types of
    /// `inputs`. New variable atoms are allocated before the instruction is recorded, and the
    /// returned atom ids are ordered like the operation outputs.
    ///
    /// # Parameters
    ///
    ///   - `operation`: Operation to append to the active transpose builder.
    ///   - `inputs`: Atom ids in the active transpose builder that feed `operation`.
    pub fn stage(&self, operation: O, inputs: &[AtomId]) -> Result<Vec<AtomId>, TracingError> {
        let mut builder_borrow = self.builder.borrow_mut();
        let input_types =
            inputs.iter().map(|atom| builder_borrow.atoms[atom.index].r#type().into_owned()).collect::<Vec<_>>();
        let output_types = operation.infer_output_types(&input_types)?;
        let outputs = output_types.into_iter().map(|r#type| builder_borrow.add_variable(r#type)).collect::<Vec<_>>();
        builder_borrow
            .instructions
            .push(Instruction { operation, inputs: inputs.to_vec(), outputs: outputs.clone() });
        Ok(outputs)
    }
}

/// Operation-level contract for staged linear maps that can be transposed.
///
/// A [`LinearOperation`] is the capability an operation carrier provides after a primal program has
/// been linearized. Implementors describe how one staged linear instruction contributes to the
/// reverse linear program used by VJP and reverse-mode gradient transforms. The trait is
/// implemented by primitive operation types, such as [`AddOperation`](crate::AddOperation), and by carrier enums,
/// such as [`LinearArrayOperation`](crate::tracing_v2::LinearArrayOperation), that delegate to primitive rules.
///
/// For a linear instruction `y = L(x)`, [`transpose`](Self::transpose) receives symbolic cotangent
/// atoms for `y` and returns symbolic cotangent contributions for `x`. Rules may reuse existing
/// cotangent atoms, return `None` for structural zeros, or stage additional linear operations in
/// the active [`TranspositionContext`]. The rule does not receive concrete primal values; any
/// required metadata must be encoded in the operation itself or in staged atom types.
///
/// Structural validation happens when the linear program is built and when transpose rules stage
/// additional operations in the transpose builder.
pub trait LinearOperation<T: Type, V: Traceable<T>, O: Clone + Operation<T>>: Operation<T> {
    /// Applies this operation's transpose rule to symbolic output cotangents.
    ///
    /// The returned vector must contain one entry per operation input. Each `Some(atom)` is a
    /// staged cotangent contribution in the active transpose builder, and each `None` means the
    /// corresponding input receives a structural zero from this operation.
    ///
    /// # Parameters
    ///
    ///   - `context`: Active transpose context used to stage any new linear operations required by
    ///     the rule.
    ///   - `output_cotangents`: Cotangent atoms aligned with this operation's outputs. `None`
    ///     entries represent structural zeros.
    fn transpose(
        &self,
        context: &mut TranspositionContext<T, V, O>,
        output_cotangents: &[Option<AtomId>],
    ) -> Result<Vec<Option<AtomId>>, TracingError>;
}

impl<T: Type, V: Traceable<T>, O: Clone + LinearOperation<T, V, O> + SupportsAdd<T, V> + SupportsZero<T, V>>
    TranspositionContext<T, V, O>
{
    /// Transposes a complete linear [`Program`] using this context's current builder.
    ///
    /// This method is the program-level transposition entry point for callers that have created a
    /// dedicated [`TranspositionContext`] for one complete pullback. It treats
    /// [`builder`](Self::builder) as the destination for the transposed program, records cotangent
    /// inputs for the primal outputs, walks `program` in reverse instruction order, and then builds
    /// the resulting pullback program.
    ///
    /// The active builder is consumed when the pullback is built. On success, this context is left
    /// with a fresh empty builder. If a transpose rule needs to transpose a nested program while
    /// preserving the surrounding builder, it should call [`transpose_nested`](Self::transpose_nested)
    /// instead.
    ///
    /// # Parameters
    ///
    ///   - `program`: Linear program whose output-cotangent-to-input-cotangent pullback should be
    ///     staged into this context's current builder.
    pub fn transpose<Input: Parameterized<V>, Output: Parameterized<V>>(
        &mut self,
        program: &Program<T, V, O, Input, Output>,
    ) -> Result<Program<T, V, O, Output, Input>, TracingError> {
        fn accumulate<T: Type, V: Traceable<T>, O: SupportsAdd<T, V> + Operation<T> + Clone>(
            builder: &Rc<RefCell<ProgramBuilder<T, V, O>>>,
            adjoints: &mut [Option<AtomId>],
            atom: AtomId,
            contribution: AtomId,
        ) -> Result<(), TracingError> {
            adjoints[atom.index] = Some(match adjoints[atom.index] {
                Some(existing) => {
                    let mut builder_borrow = builder.borrow_mut();
                    let abstract_value = builder_borrow.atoms[existing.index].r#type().into_owned();
                    let output = builder_borrow.add_variable(abstract_value);
                    builder_borrow.instructions.push(Instruction {
                        operation: O::add_operation(),
                        inputs: vec![existing, contribution],
                        outputs: vec![output],
                    });
                    output
                }
                None => contribution,
            });
            Ok(())
        }

        fn stage_zero<T: Type, V: Traceable<T>, O: SupportsZero<T, V> + Operation<T> + Clone>(
            builder: &Rc<RefCell<ProgramBuilder<T, V, O>>>,
            r#type: T,
        ) -> AtomId {
            let mut builder_borrow = builder.borrow_mut();
            let output = builder_borrow.add_variable(r#type.clone());
            builder_borrow.instructions.push(Instruction {
                operation: <O as SupportsZero<T, V>>::zero_operation(r#type),
                inputs: vec![],
                outputs: vec![output],
            });
            output
        }

        let builder = self.builder.clone();
        let mut output_cotangent_inputs = Vec::with_capacity(program.output_ids.len());
        for output in program.output_ids.iter() {
            let output_atom = program.atoms.get(output.index).ok_or(TracingError::UnboundAtomId { id: *output })?;
            let cotangent_input = builder.borrow_mut().add_input(output_atom.r#type().into_owned());
            output_cotangent_inputs.push(cotangent_input);
        }

        let mut adjoints = vec![None; program.atoms.len()];
        for (cotangent, output) in output_cotangent_inputs.into_iter().zip(program.output_ids.iter().copied()) {
            accumulate::<T, V, O>(&builder, adjoints.as_mut_slice(), output, cotangent)?;
        }

        for instruction in program.instructions.iter().rev() {
            if instruction.outputs.iter().all(|output| adjoints[output.index].is_none()) {
                continue;
            }
            let instruction_output_cotangents =
                instruction.outputs.iter().map(|output| adjoints[output.index]).collect::<Vec<_>>();
            let input_cotangents = instruction.operation.transpose(self, instruction_output_cotangents.as_slice())?;
            for (input, contribution) in instruction.inputs.iter().copied().zip(input_cotangents) {
                if let Some(contribution) = contribution {
                    accumulate::<T, V, O>(&builder, adjoints.as_mut_slice(), input, contribution)?;
                }
            }
        }

        let outputs = program
            .input_ids
            .iter()
            .copied()
            .map(|input| match adjoints[input.index] {
                Some(adjoint) => adjoint,
                None => stage_zero::<T, V, O>(&builder, program.atoms[input.index].r#type().into_owned()),
            })
            .collect::<Vec<_>>();
        drop(builder);
        let builder = self.builder.clone();
        self.builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        let builder = match Rc::try_unwrap(builder) {
            Ok(builder) => builder.into_inner(),
            Err(_) => return Err(TracingError::EscapedProgramBuilder),
        };
        builder
            .build(outputs, program.output_structure.clone(), program.input_structure.clone())?
            .simplified()
    }

    /// Transposes a nested linear [`Program`] without consuming this context's current builder.
    ///
    /// This method is for transpose rules that carry linear subprograms as operation metadata, such
    /// as captured control-flow branches. It temporarily replaces [`builder`](Self::builder) with a
    /// fresh sibling builder, calls [`transpose`](Self::transpose) for the nested `program`, and
    /// then restores the original builder before returning the nested pullback result. This keeps
    /// nested transposition from appending instructions to the surrounding pullback or consuming the
    /// builder that the surrounding rule still needs. The original builder is restored whether the
    /// nested transposition succeeds or returns an error.
    ///
    /// # Parameters
    ///
    ///   - `program`: Nested linear program to transpose in an isolated temporary builder.
    pub fn transpose_nested<Input: Parameterized<V>, Output: Parameterized<V>>(
        &mut self,
        program: &Program<T, V, O, Input, Output>,
    ) -> Result<Program<T, V, O, Output, Input>, TracingError> {
        let parent_builder = self.builder.clone();
        self.builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        let result = self.transpose(program);
        self.builder = parent_builder;
        result
    }
}

impl<T: Type, V: Traceable<T>, O: Clone + Operation<T>, Input: Parameterized<V>, Output: Parameterized<V>>
    Program<T, V, O, Input, Output>
{
    /// Transposes this linear pushforward [`Program`] into its reverse-mode pullback.
    ///
    /// `output_examples` carries one representative value per primal output and is only used to
    /// validate the output count; cotangent input types come from this program's own atom metadata.
    /// Disconnected primal inputs are emitted as zero operations, which the value type's [`Zero`](crate::Zero)
    /// implementation evaluates at interpretation time.
    ///
    /// # Parameters
    ///
    ///   - `output_examples`: Representative output values aligned with this program's output atoms.
    pub fn transpose(&self, output_examples: &[V]) -> Result<Program<T, V, O, Output, Input>, TracingError>
    where
        O: LinearOperation<T, V, O> + SupportsAdd<T, V> + SupportsZero<T, V>,
    {
        let expected_output_count = self.output_ids.len();
        check_count!("output", output_examples, expected_output_count, TracingError);
        let builder = Rc::new(RefCell::new(ProgramBuilder::<T, V, O>::new()));
        let mut context = TranspositionContext::new(builder);
        context.transpose(self)
    }
}

impl<'engine, E: TracingEngine + ?Sized> TracingContext<'engine, E> {
    /// Transposes a traced linear [`Program`] using this [`TracingContext`] for zero materialization.
    ///
    /// The transpose program itself is staged in a fresh linear-program builder. Any zero operation
    /// produced for a disconnected primal input is then replaced with a traced constant whose
    /// underlying outer-trace value is synthesized through [`Engine::zero`](crate::tracing::engines::Engine::zero).
    pub fn transpose<
        Input: Parameterized<Tracer<'engine, E>>,
        Output: Parameterized<Tracer<'engine, E>>,
        O: Clone
            + LinearOperation<E::Type, Tracer<'engine, E>, O>
            + SupportsAdd<E::Type, Tracer<'engine, E>>
            + SupportsZero<E::Type, Tracer<'engine, E>>,
    >(
        &self,
        program: &Program<E::Type, Tracer<'engine, E>, O, Input, Output>,
    ) -> Result<Program<E::Type, Tracer<'engine, E>, O, Output, Input>, TracingError>
    where
        E: 'static,
    {
        let builder = Rc::new(RefCell::new(ProgramBuilder::<E::Type, Tracer<'engine, E>, O>::new()));
        let mut context = TranspositionContext::new(builder);
        let pullback = context.transpose(program)?;

        let mut builder = ProgramBuilder::<E::Type, Tracer<'engine, E>, O>::new();
        builder.atoms = pullback.atoms.clone();
        builder.input_ids = pullback.input_ids.clone();
        let mut atom_remapping: Vec<Option<AtomId>> = vec![None; builder.atoms.len()];
        let mut rewritten_instructions = Vec::with_capacity(pullback.instructions.len());
        for instruction in &pullback.instructions {
            if let Some(zero_operation) = instruction.operation.as_zero_operation()
                && instruction.outputs.len() == 1
                && instruction.inputs.is_empty()
            {
                // Zero ops in traced pullbacks have no inputs from which interpretation can recover a tracing
                // context, so materialize each one as a constant in this outer trace and remap its uses.
                let zero_tracer = self.constant(self.engine.zero(&zero_operation.r#type)?);
                let constant_atom = builder.add_constant(zero_tracer);
                atom_remapping[instruction.outputs[0].index] = Some(constant_atom);
            } else {
                let inputs = instruction
                    .inputs
                    .iter()
                    .map(|atom| atom_remapping[atom.index].unwrap_or(*atom))
                    .collect::<Vec<_>>();
                rewritten_instructions.push(Instruction {
                    operation: instruction.operation.clone(),
                    inputs,
                    outputs: instruction.outputs.clone(),
                });
            }
        }
        builder.instructions = rewritten_instructions;
        let outputs = pullback
            .output_ids
            .iter()
            .map(|atom| atom_remapping[atom.index].unwrap_or(*atom))
            .collect::<Vec<_>>();
        builder
            .build(outputs, pullback.input_structure.clone(), pullback.output_structure.clone())?
            .simplified()
    }
}
