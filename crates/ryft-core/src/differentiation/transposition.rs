use std::cell::RefCell;
use std::rc::Rc;

use crate::differentiation::Cotangent;
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::SupportsAdd;
use crate::operations::constants::SupportsZero;
use crate::parameters::{Parameter, Parameterized};
use crate::tracing::domains::{
    ProgramTracingContext, ProgramTracingDomain, RuntimeDomain, Tracer, TracingContext, TracingDomain,
};
use crate::tracing::{AtomId, Instruction, Program, ProgramBuilder, Traceable, TracingError};
use crate::types::{Type, Typed};

/// Represents [`Operation`]s that are _linear_, meaning that they can be _transposed_. For a linear [`Instruction`]
/// `y = L(x)`, [`transpose`](Self::transpose) receives symbolic cotangent [`Tracer`]s for `y` and returns symbolic
/// cotangent contributions for `x`, representing the transposed cotangent. Rules may reuse existing cotangents,
/// return [`Cotangent::Zero`] for structural zeros, or stage additional linear operations in the active
/// [`ProgramTracingContext`]. The rule does not receive concrete primal values; any required metadata must be encoded
/// in the operation itself or in staged atom types.
///
/// Refer to the documentation of [`Program::transpose`] for more information what _transposition_ means here and how
/// it relates to the algebraic notion of transposition.
pub trait LinearOperation<T: Type + Parameter, V: Traceable<T>, O: Operation<T>>: Operation<T> {
    /// Applies this operation's transpose rule to the provided symbolic output cotangents. The returned vector must
    /// contain one entry per operation input. Each [`Cotangent::Staged`] value is a staged cotangent contribution in
    /// the active transpose builder, and each [`Cotangent::Zero`] means that the corresponding input receives a
    /// structural zero from this operation.
    fn transpose<'transpose>(
        &self,
        context: &mut ProgramTracingContext<'transpose, T, V, O>,
        output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, TracingError>;
}

impl<
    T: Type + Parameter,
    V: Traceable<T>,
    O: LinearOperation<T, V, O> + SupportsZero<T, V> + SupportsAdd<T, V>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
> Program<T, V, O, Input, Output>
{
    /// Transposes this linear pushforward [`Program`] into its reverse-mode pullback. This is the main entrypoint for
    /// transposing linear [`Program`]s. In the algebraic sense, _transposing_ a linear map `L: X -> Y` gives a map on
    /// _dual_ spaces `L^T: Y* -> X*`. In finite dimensions this is the same operation represented by a matrix
    /// transpose. Here the linear map is not stored as a matrix. It is a staged [`Program`] that maps input tangents
    /// to output tangents, and transposition builds the dual program that maps output cotangents back to input
    /// cotangents. Operationally, transposition creates cotangent inputs for this program's outputs, walks the
    /// instructions in reverse order, and applies each primitive operation's [`LinearOperation::transpose`] rule
    /// to accumulate cotangent contributions for the original inputs. This is the same decomposition of reverse-mode
    /// automatic differentiation as in [this paper](https://arxiv.org/abs/2204.10923).
    ///
    /// Disconnected primal inputs are emitted as [`ZeroOperation`](crate::operations::ZeroOperation)s, which the value
    /// type's [`Zero`](crate::Zero) implementation evaluates at interpretation time. For linear programs whose values
    /// are [`Tracer`]s from an outer trace, use [`TracingContext::transpose`] instead so that those disconnected-input
    /// zeros can be materialized in the surrounding tracing context.
    #[inline]
    pub fn transpose(&self) -> Result<Program<T, V, O, Output, Input>, TracingError> {
        let builder = Rc::new(RefCell::new(ProgramBuilder::<T, V, O>::new()));
        let domain = ProgramTracingDomain::new();
        let mut context = ProgramTracingContext::new(&domain, builder);
        context.transpose(self)
    }
}

impl<'domain, D: RuntimeDomain + TracingDomain> TracingContext<'domain, D> {
    /// Transposes the provided traced linear [`Program`] and materializes standalone zero [`Cotangent`]s. This is a
    /// wrapper around the builder-level transposition implementation on [`ProgramTracingContext::transpose`]. It first
    /// stages the ordinary pullback program in a fresh [`ProgramTracingContext`], then performs the extra rewrite
    /// needed when the linear program's values are [`Tracer`]s belonging to this outer [`TracingContext`].
    ///
    /// Use this method when transposing a linear program inside an outer trace, such as when staging a traced
    /// reverse-mode pullback. Use [`Program::transpose`] for ordinary complete linear program transposition, and use
    /// [`ProgramTracingContext::transpose`] only when you already own the destination [`ProgramBuilder`] and want to
    /// run the lower-level transposition algorithm directly.
    ///
    /// The transposed program is first staged as an ordinary linear [`Program`]. When a primal input is disconnected
    /// from the outputs, transposition represents its cotangent as a [`ZeroOperation`](crate::ZeroOperation). Such an
    /// input-free operation cannot recover the surrounding [`TracingContext`] during later interpretation, and so this
    /// method replaces each standalone zero operation with a constant [`Tracer`] created in this [`TracingContext`].
    /// The concrete zero value stored in that tracer is synthesized through [`RuntimeDomain::zero`], while the final
    /// pullback still receives and returns traced [`Cotangent`]s.
    pub fn transpose<
        Input: Parameterized<Tracer<'domain, D>>,
        Output: Parameterized<Tracer<'domain, D>>,
        O: Clone
            + LinearOperation<D::Type, Tracer<'domain, D>, O>
            + SupportsZero<D::Type, Tracer<'domain, D>>
            + SupportsAdd<D::Type, Tracer<'domain, D>>,
    >(
        &self,
        program: &Program<D::Type, Tracer<'domain, D>, O, Input, Output>,
    ) -> Result<Program<D::Type, Tracer<'domain, D>, O, Output, Input>, TracingError> {
        // First build the ordinary transposed program. At this point disconnected inputs are still represented
        // as input-free zero operations in the transposed program.
        let builder = Rc::new(RefCell::new(ProgramBuilder::<D::Type, Tracer<'domain, D>, O>::new()));
        let domain = ProgramTracingDomain::new();
        let mut context = ProgramTracingContext::new(&domain, builder);
        let transposed_program = context.transpose(program)?;

        // Rewrite the transposed program into a sibling builder. We preserve the existing atom table and inputs,
        // then use `atom_remapping` only for atoms that need to point at replacement constants.
        let mut builder = ProgramBuilder::<D::Type, Tracer<'domain, D>, O>::new();
        builder.atoms = transposed_program.atoms.clone();
        builder.input_ids = transposed_program.input_ids.clone();
        let mut atom_remapping = vec![None; builder.atoms.len()];
        let mut rewritten_instructions = Vec::with_capacity(transposed_program.instructions.len());
        for instruction in &transposed_program.instructions {
            if let Some(zero_operation) = instruction.operation().as_zero_operation()
                && instruction.outputs().len() == 1
                && instruction.inputs().is_empty()
            {
                // Zero operations in traced pullbacks have no inputs from which interpretation can recover a tracing
                // context, and so we materialize each one as a constant in this tracing context and remap its uses.
                let zero = builder.add_constant(self.constant(self.domain.zero(&zero_operation.r#type())?));
                atom_remapping[instruction.outputs()[0].index()] = Some(zero);
            } else {
                // Preserve non-zero instructions, rewriting only the inputs that consumed a zero operation
                // we replaced with a traced constant above.
                let inputs = instruction
                    .inputs()
                    .iter()
                    .map(|atom| atom_remapping[atom.index()].unwrap_or(*atom))
                    .collect::<Vec<_>>();
                rewritten_instructions.push(Instruction::new(
                    instruction.operation().clone(),
                    inputs,
                    instruction.outputs().to_vec(),
                ));
            }
        }
        builder.instructions = rewritten_instructions;

        // Outputs can also refer directly to replaced zero-operation atoms, and so we apply the same remapping before
        // building. The subsequent simplification removes the skipped zero instructions and their old output atoms.
        let outputs = transposed_program
            .output_ids
            .iter()
            .map(|atom| atom_remapping[atom.index()].unwrap_or(*atom))
            .collect::<Vec<_>>();
        builder
            .build(outputs, transposed_program.input_structure.clone(), transposed_program.output_structure.clone())?
            .into_simplified()
    }
}

impl<
    'domain,
    T: 'domain + Type + Parameter,
    V: 'domain + Traceable<T>,
    O: 'domain + LinearOperation<T, V, O> + SupportsZero<T, V> + SupportsAdd<T, V>,
> TracingContext<'domain, ProgramTracingDomain<T, V, O>>
{
    /// Transposes the provided linear [`Program`] using this [`TracingContext`]'s [`ProgramBuilder`]. This is the
    /// builder-level implementation behind [`Program::transpose`]. Refer to the documentation of [`Program::transpose`]
    /// for the conceptual relationship between program transposition, algebraic transposition, pushforward functions,
    /// and pullback functions. This function is for callers that already own a [`ProgramTracingContext`] and need the
    /// transposed program to be staged through that context's [`ProgramBuilder`].
    ///
    /// This function treats [`builder`](Self::builder) as the destination for the transposed program, records cotangent
    /// inputs for the primal outputs, walks `program` in reverse instruction order, and transposes each [`Instruction`]
    /// using [`LinearOperation::transpose`]. The active [`ProgramBuilder`] is consumed when the pullback is built. On
    /// success, this context is left with a fresh empty builder. If a transpose rule needs to transpose a nested
    /// program while preserving the surrounding builder, it should call [`transpose_nested`](Self::transpose_nested)
    /// instead.
    pub fn transpose<Input: Parameterized<V>, Output: Parameterized<V>>(
        &mut self,
        program: &Program<T, V, O, Input, Output>,
    ) -> Result<Program<T, V, O, Output, Input>, TracingError> {
        /// Accumulates one staged cotangent contribution for `atom` into the reverse-pass adjoint table. The first
        /// contribution is stored directly, while later contributions are summed by staging an add instruction in the
        /// transpose builder.
        ///
        /// # Parameters
        ///
        ///   - `builder`: Destination builder for the transposed program.
        ///   - `adjoints`: Per-primal-atom table storing the currently accumulated cotangent atom, if any.
        ///   - `atom`: Primal atom whose cotangent is being accumulated.
        ///   - `contribution`: Staged cotangent atom to add into `atom`'s adjoint slot.
        fn accumulate<T: Type, V: Traceable<T>, O: Operation<T>>(
            builder: &Rc<RefCell<ProgramBuilder<T, V, O>>>,
            adjoints: &mut [Option<AtomId>],
            atom: AtomId,
            contribution: AtomId,
        ) -> Result<(), TracingError>
        where
            O: SupportsAdd<T, V>,
        {
            // Contributions must already be atoms in the transpose builder. Otherwise the `AtomId` could alias an
            // unrelated atom index and corrupt the pullback graph.
            if builder.borrow().atoms().get(contribution.index()).is_none() {
                return Err(TracingError::UnboundAtomId { id: contribution });
            }

            // Locate the primal atom's adjoint slot. An out-of-range slot means the input program is malformed.
            let adjoint = adjoints.get_mut(atom.index()).ok_or(TracingError::UnboundAtomId { id: atom })?;

            // If this atom already has a cotangent, stage an add so both contributions flow into one accumulated
            // adjoint. Otherwise, keep the first contribution directly and avoid emitting an unnecessary add.
            *adjoint = Some(match *adjoint {
                Some(existing) => {
                    let mut builder_borrow = builder.borrow_mut();
                    let outputs = builder_borrow.add_instruction(O::add_operation(), vec![existing, contribution])?;
                    check_count!("output", outputs, 1, TracingError);
                    outputs[0]
                }
                None => contribution,
            });
            Ok(())
        }

        // Reuse this context's current builder as the destination for the pullback program, and reserve the main
        // structural vectors up front. These are conservative lower bounds that cover cotangent inputs, one instruction
        // per reversed primal instruction, and possible zero outputs for disconnected primal inputs.
        let builder = self.builder().clone();
        {
            let mut builder_borrow = builder.borrow_mut();
            builder_borrow
                .atoms
                .reserve(program.output_ids.len() + program.instructions.len() + program.input_ids.len());
            builder_borrow.input_ids.reserve(program.output_ids.len());
            builder_borrow.instructions.reserve(program.instructions.len() + program.input_ids.len());
        }

        // Seed the reverse pass with one cotangent input for each primal output. The adjoint table is indexed by atoms
        // from the original program, and each slot stores the staged pullback atom that currently represents the
        // accumulated cotangent for that primal atom.
        let mut adjoints = vec![None; program.atoms().len()];
        for output in program.output_ids().iter().copied() {
            let output_atom = program.atoms().get(output.index()).ok_or(TracingError::UnboundAtomId { id: output })?;
            let cotangent_input = builder.borrow_mut().add_input(output_atom.r#type().into_owned());
            accumulate::<T, V, O>(&builder, adjoints.as_mut_slice(), output, cotangent_input)?;
        }

        // Walk the primal program backward, applying each operation's transpose rule only when at least one of its
        // outputs has a non-zero accumulated cotangent. The scratch vector avoids allocating a fresh cotangent vector
        // for every live instruction.
        let max_instruction_output_count =
            program.instructions().iter().map(|instruction| instruction.outputs().len()).max().unwrap_or(0);
        let mut instruction_output_cotangents = Vec::with_capacity(max_instruction_output_count);
        for instruction in program.instructions().iter().rev() {
            // Skip dead reverse edges early: if none of an instruction's outputs carries an adjoint, the instruction
            // cannot contribute to any input cotangent.
            let mut has_output_adjoint = false;
            for output in instruction.outputs().iter().copied() {
                if adjoints.get(output.index()).ok_or(TracingError::UnboundAtomId { id: output })?.is_some() {
                    has_output_adjoint = true;
                    break;
                }
            }
            if !has_output_adjoint {
                continue;
            }

            // Materialize the instruction's output cotangents in operation-result order. Missing adjoint slots become
            // structural zeros so transpose rules can distinguish unused outputs without staging zero operations.
            instruction_output_cotangents.clear();
            for output in instruction.outputs().iter().copied() {
                instruction_output_cotangents.push(
                    match adjoints.get(output.index()).copied().ok_or(TracingError::UnboundAtomId { id: output })? {
                        Some(atom) => Cotangent::staged(self.tracer(atom, None)),
                        None => Cotangent::Zero,
                    },
                );
            }

            // Apply the primitive transpose rule and require exactly one cotangent contribution per primal input. This
            // prevents malformed rules from silently dropping or inventing cotangents through iterator truncation.
            let input_cotangents = instruction.operation().transpose(self, instruction_output_cotangents.as_slice())?;
            check_count!("input", input_cotangents, instruction.inputs().len(), TracingError);
            for (input, contribution) in instruction.inputs().iter().copied().zip(input_cotangents) {
                if let Some(contribution) = contribution.as_staged() {
                    // Staged contributions must belong to this pullback builder before their atom IDs can be accumulated.
                    if !Rc::ptr_eq(&builder, contribution.builder()) {
                        return Err(TracingError::MismatchedProgramBuilders);
                    }
                    accumulate::<T, V, O>(&builder, adjoints.as_mut_slice(), input, contribution.atom_id()?)?;
                }
            }
        }
        instruction_output_cotangents.clear();

        // The pullback outputs are the accumulated cotangents for the primal inputs. Disconnected primal inputs receive
        // explicit typed zero operations, staged through the builder so normal type inference validates the result.
        let outputs = program
            .input_ids()
            .iter()
            .copied()
            .map(|input| {
                match adjoints.get(input.index()).copied().ok_or(TracingError::UnboundAtomId { id: input })? {
                    Some(adjoint) => Ok::<AtomId, TracingError>(adjoint),
                    None => {
                        let input_atom =
                            program.atoms().get(input.index()).ok_or(TracingError::UnboundAtomId { id: input })?;
                        let mut builder_borrow = builder.borrow_mut();
                        let outputs = builder_borrow
                            .add_instruction(O::zero_operation(input_atom.r#type().into_owned()), Vec::new())?;
                        check_count!("output", outputs, 1, TracingError);
                        Ok(outputs[0])
                    }
                }
            })
            .collect::<Result<Vec<_>, TracingError>>()?;

        // Consume the active builder to build the pullback, leaving this context with a fresh empty builder for any
        // subsequent tracing work.
        drop(builder);
        let builder = std::mem::replace(&mut self.builder, Rc::new(RefCell::new(ProgramBuilder::new())));
        let builder = match Rc::try_unwrap(builder) {
            Ok(builder) => builder.into_inner(),
            Err(_) => return Err(TracingError::EscapedProgramBuilder),
        };
        builder.build(outputs, program.output_structure().clone(), program.input_structure().clone())
    }

    /// Transposes the provided linear [`Program`] without consuming this [`TracingContext`]'s [`ProgramBuilder`].
    /// This is the nested-program variant of [`ProgramTracingContext::transpose`]. Refer to the documentation of
    /// [`Program::transpose`] for the conceptual relationship between program transposition, algebraic transposition,
    /// pushforward functions, and pullback functions. This function is for transposition rules that carry linear
    /// sub-programs as operation metadata, such as captured control-flow branches.
    ///
    /// This function temporarily replaces [`builder`](Self::builder) with a fresh sibling builder, calls
    /// [`transpose`](Self::transpose) for the provided [`Program`], and then restores the original [`ProgramBuilder`]
    /// before returning the nested pullback result. This keeps nested transposition from appending [`Instruction`]s to
    /// the surrounding pullback or consuming the builder that the surrounding rule still needs. The original builder
    /// is restored whether the nested transposition succeeds or fails.
    #[inline]
    pub fn transpose_nested<Input: Parameterized<V>, Output: Parameterized<V>>(
        &mut self,
        program: &Program<T, V, O, Input, Output>,
    ) -> Result<Program<T, V, O, Output, Input>, TracingError> {
        let parent_builder = std::mem::replace(&mut self.builder, Rc::new(RefCell::new(ProgramBuilder::new())));
        let result = self.transpose(program);
        self.builder = parent_builder;
        result
    }
}

// TODO(eaplatanios): Review the unit tests.
#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::marker::PhantomData;
    use std::rc::Rc;

    use pretty_assertions::assert_eq;

    use crate::macros::check_count;
    use crate::operations::Operation;
    use crate::operations::constants::ZeroOperation;
    use crate::parameters::Placeholder;
    use crate::tracing::{Atom, AtomId, Instruction, Program, ProgramBuilder, ProgramTracingContext};
    use crate::types::{DataType, TypeError};

    use super::*;

    #[derive(Clone, Debug)]
    enum TestLinearOperation {
        Identity,
        BadArity,
        ForeignContribution,
        Add,
        Zero(ZeroOperation<DataType>),
    }

    impl Operation<DataType> for TestLinearOperation {
        #[inline]
        fn name(&self) -> &'static str {
            match self {
                Self::Identity => "identity",
                Self::BadArity => "bad_arity",
                Self::ForeignContribution => "foreign_contribution",
                Self::Add => "add",
                Self::Zero(_) => "zero",
            }
        }

        fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
            match self {
                Self::Identity | Self::BadArity | Self::ForeignContribution => {
                    check_count!("input", input_types, 1, TypeError);
                    Ok(vec![input_types[0].clone()])
                }
                Self::Add => {
                    check_count!("input", input_types, 2, TypeError);
                    Ok(vec![input_types[0].clone()])
                }
                Self::Zero(zero) => zero.infer_output_types(input_types),
            }
        }
    }

    impl SupportsAdd<DataType, f64> for TestLinearOperation {
        fn add_operation() -> Self {
            Self::Add
        }
    }

    impl SupportsZero<DataType, f64> for TestLinearOperation {
        fn zero_operation(r#type: DataType) -> Self {
            Self::Zero(ZeroOperation::new(r#type))
        }
    }

    impl LinearOperation<DataType, f64, TestLinearOperation> for TestLinearOperation {
        fn transpose<'transpose>(
            &self,
            context: &mut ProgramTracingContext<'transpose, DataType, f64, TestLinearOperation>,
            output_cotangents: &[Cotangent<'transpose, DataType, f64, TestLinearOperation>],
        ) -> Result<Vec<Cotangent<'transpose, DataType, f64, TestLinearOperation>>, TracingError> {
            check_count!("output", output_cotangents, 1, TracingError);
            match self {
                Self::Identity => Ok(vec![output_cotangents[0].clone()]),
                Self::BadArity => Ok(Vec::new()),
                Self::ForeignContribution => {
                    let foreign_builder = Rc::new(RefCell::new(ProgramBuilder::new()));
                    let foreign_context = ProgramTracingContext::new(context.domain(), foreign_builder);
                    Ok(vec![Cotangent::Staged(foreign_context.input(DataType::F64))])
                }
                Self::Add => Ok(vec![output_cotangents[0].clone(), output_cotangents[0].clone()]),
                Self::Zero(_) => Ok(Vec::new()),
            }
        }
    }

    fn unary_program(operation: TestLinearOperation) -> Program<DataType, f64, TestLinearOperation, f64, f64> {
        let mut builder = ProgramBuilder::<DataType, f64, TestLinearOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, vec![input]).unwrap()[0];
        builder.build(vec![output], Placeholder, Placeholder).unwrap()
    }

    #[test]
    fn test_transpose_rejects_invalid_rule_input_cotangent_count() {
        assert_eq!(
            unary_program(TestLinearOperation::BadArity).transpose().unwrap_err(),
            TracingError::InvalidInputCount { expected: 1, got: 0 },
        );
    }

    #[test]
    fn test_transpose_rejects_foreign_builder_contribution() {
        assert_eq!(
            unary_program(TestLinearOperation::ForeignContribution).transpose().unwrap_err(),
            TracingError::MismatchedProgramBuilders,
        );
    }

    #[test]
    fn test_transpose_reports_unbound_input_atom() {
        let program = Program::<DataType, f64, TestLinearOperation, f64, ()> {
            atoms: Vec::new(),
            input_ids: vec![AtomId::new(0)],
            output_ids: Vec::new(),
            instructions: Vec::new(),
            input_structure: Placeholder,
            output_structure: (),
            marker: PhantomData,
        };
        assert_eq!(program.transpose().unwrap_err(), TracingError::UnboundAtomId { id: AtomId::new(0) },);
    }

    #[test]
    fn test_transpose_reports_unbound_instruction_output_atom() {
        let input = AtomId::new(0);
        let missing_output = AtomId::new(1);
        let program = Program::<DataType, f64, TestLinearOperation, f64, f64> {
            atoms: vec![Atom::Variable(DataType::F64)],
            input_ids: vec![input],
            output_ids: vec![input],
            instructions: vec![Instruction::new(TestLinearOperation::Identity, vec![input], vec![missing_output])],
            input_structure: Placeholder,
            output_structure: Placeholder,
            marker: PhantomData,
        };
        assert_eq!(program.transpose().unwrap_err(), TracingError::UnboundAtomId { id: missing_output },);
    }

    #[test]
    fn test_transpose_stages_disconnected_input_zero_through_builder_inference() {
        let program = Program::<DataType, f64, TestLinearOperation, f64, ()> {
            atoms: vec![Atom::Variable(DataType::F64)],
            input_ids: vec![AtomId::new(0)],
            output_ids: Vec::new(),
            instructions: Vec::new(),
            input_structure: Placeholder,
            output_structure: (),
            marker: PhantomData,
        };

        let transposed = program.transpose().unwrap();

        assert_eq!(transposed.output_ids(), &[AtomId::new(0)]);
        assert!(matches!(transposed.instructions()[0].operation(), TestLinearOperation::Zero(_)));
    }
}
