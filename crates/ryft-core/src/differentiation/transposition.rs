use std::cell::RefCell;
use std::rc::Rc;

use crate::differentiation::Cotangent;
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::SupportsAdd;
use crate::operations::constants::SupportsZero;
use crate::parameters::{Parameter, Parameterized};
use crate::tracing::contexts::{Context, ProgramTracingContext, TracingContext};
use crate::tracing::domains::{DomainTracer, ProgramTracingDomain, RuntimeDomain, TracingDomain};
use crate::tracing::{AtomId, Instruction, Program, ProgramBuilder, Traceable, TracingError};
use crate::types::{Type, Typed};

/// Represents [`Operation`]s that provide a transpose rule for linear [`Program`]s. For a linear [`Instruction`]
/// `y = L(x)`, [`transpose`](Self::transpose) receives symbolic [`Cotangent`]s for `y` and returns symbolic cotangent
/// contributions for `x`, representing the transposed cotangent. Rules may reuse existing cotangents, return
/// [`Cotangent::Zero`] for structural zeros, or stage additional linear operations in the active
/// [`ProgramTracingContext`]. The rule does not receive concrete primal values; any required metadata must be encoded
/// in the operation itself or in staged atom types.
///
/// Refer to the documentation of [`Program::transpose`] for more information what _transposition_ means here and how
/// it relates to the algebraic notion of transposition.
pub trait TransposableOperation<T: Type + Parameter, V: Traceable<T>, O: Operation<T>>: Operation<T> {
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
    O: TransposableOperation<T, V, O> + SupportsZero<T, V> + SupportsAdd<T, V>,
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
    /// instructions in reverse order, and applies each primitive operation's [`TransposableOperation::transpose`] rule
    /// to accumulate cotangent contributions for the original inputs. This is the same decomposition of reverse-mode
    /// automatic differentiation as in [this paper](https://arxiv.org/abs/2204.10923).
    ///
    /// Disconnected primal inputs are emitted as [`ZeroOperation`](crate::operations::ZeroOperation)s, which the value
    /// type's [`Zero`](crate::Zero) implementation evaluates at interpretation time. For linear programs whose values
    /// are [`Tracer`](crate::tracing::Tracer)s from an outer trace, use [`TracingContext::transpose`] instead so that
    /// those disconnected-input zeros can be materialized in the surrounding tracing context.
    #[inline]
    pub fn transpose(&self) -> Result<Program<T, V, O, Output, Input>, TracingError> {
        let builder = Rc::new(RefCell::new(ProgramBuilder::<T, V, O>::new()));
        let domain = ProgramTracingDomain::new();
        let mut context = ProgramTracingContext::new(&domain, builder);
        context.transpose(self)
    }
}

impl<'domain, D: RuntimeDomain + TracingDomain<Operation: SupportsZero<D::Type, D::Constant>>>
    TracingContext<'domain, D>
{
    /// Transposes the provided traced linear [`Program`] and materializes standalone zero [`Cotangent`]s. This is a
    /// wrapper around the transposition implementation on [`ProgramTracingContext::transpose_with_zero_fn`]. It uses
    /// the same reverse-walk implementation as [`Program::transpose`], but changes how standalone zero cotangents are
    /// represented when the linear program's values are [`Tracer`](crate::tracing::Tracer)s belonging to this outer
    /// [`TracingContext`].
    ///
    /// Use this method when transposing a linear program inside an outer trace, such as when staging a traced
    /// reverse-mode pullback. Use [`Program::transpose`] for ordinary complete linear program transposition, and use
    /// [`ProgramTracingContext::transpose`] only when you already own the destination [`ProgramBuilder`] and want to
    /// run the lower-level transposition algorithm directly.
    ///
    /// When a primal input is disconnected from the outputs, or when a transpose rule must materialize a structural
    /// zero cotangent, an input-free [`ZeroOperation`](crate::ZeroOperation) cannot recover the surrounding
    /// [`TracingContext`] during later interpretation. This method materializes each standalone zero as a constant
    /// [`Tracer`](crate::tracing::Tracer) created in this [`TracingContext`]. The zero is staged through the domain's
    /// ordinary zero operation, so domains whose traced constants are abstract metadata do not need to materialize a
    /// runtime value just to transpose an enclosing traced program.
    #[inline]
    pub fn transpose<
        Input: Parameterized<DomainTracer<'domain, D>>,
        Output: Parameterized<DomainTracer<'domain, D>>,
        O: TransposableOperation<D::Type, DomainTracer<'domain, D>, O>
            + SupportsZero<D::Type, DomainTracer<'domain, D>>
            + SupportsAdd<D::Type, DomainTracer<'domain, D>>,
    >(
        &self,
        program: &Program<D::Type, DomainTracer<'domain, D>, O, Input, Output>,
    ) -> Result<Program<D::Type, DomainTracer<'domain, D>, O, Output, Input>, TracingError> {
        let builder = Rc::new(RefCell::new(ProgramBuilder::<D::Type, DomainTracer<'domain, D>, O>::new()));
        let domain = ProgramTracingDomain::new();
        let mut context = ProgramTracingContext::new(&domain, builder);
        context.transpose_with_zero_fn(
            program,
            Some(|builder: &mut ProgramBuilder<D::Type, DomainTracer<'domain, D>, O>, r#type: &D::Type| {
                let operation = D::Operation::zero_operation(r#type.clone());
                let outputs = self.stage_operation(operation, &[] as &[DomainTracer<'domain, D>])?;
                check_count!("output", outputs, 1, TracingError);
                Ok(builder.add_constant(outputs.into_iter().next().expect("checked above")))
            }),
        )
    }
}

impl<
    'domain,
    T: 'domain + Type + Parameter,
    V: 'domain + Traceable<T>,
    O: 'domain + TransposableOperation<T, V, O> + SupportsZero<T, V> + SupportsAdd<T, V>,
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
    /// using [`TransposableOperation::transpose`]. The active [`ProgramBuilder`] is consumed when the pullback is
    /// built. On success, this context is left with a fresh empty builder. If a transpose rule needs to transpose a
    /// nested program while preserving the surrounding builder, it should call
    /// [`transpose_nested`](Self::transpose_nested) instead.
    #[inline]
    pub fn transpose<Input: Parameterized<V>, Output: Parameterized<V>>(
        &mut self,
        program: &Program<T, V, O, Input, Output>,
    ) -> Result<Program<T, V, O, Output, Input>, TracingError> {
        self.transpose_with_zero_fn(
            program,
            None::<fn(&mut ProgramBuilder<T, V, O>, &T) -> Result<AtomId, TracingError>>,
        )
    }

    /// Transposes the provided linear [`Program`] using this [`TracingContext`]'s [`ProgramBuilder`] and the supplied
    /// optional zero function. Active-context callers provide `zero_fn` so zero operations staged by higher-order
    /// transpose rules are replaced with constants before the pullback is built.
    pub fn transpose_with_zero_fn<
        Input: Parameterized<V>,
        Output: Parameterized<V>,
        ZeroFn: FnMut(&mut ProgramBuilder<T, V, O>, &T) -> Result<AtomId, TracingError>,
    >(
        &mut self,
        program: &Program<T, V, O, Input, Output>,
        mut zero_fn: Option<ZeroFn>,
    ) -> Result<Program<T, V, O, Output, Input>, TracingError> {
        /// Materializes a standalone "zero" into `builder`, either through the active-context `zero_fn` or as an
        /// ordinary typed [`ZeroOperation`](crate::ZeroOperation).
        fn zero<
            T: Type + Parameter,
            V: Traceable<T>,
            O: Operation<T> + SupportsZero<T, V>,
            ZeroFn: FnMut(&mut ProgramBuilder<T, V, O>, &T) -> Result<AtomId, TracingError>,
        >(
            builder: &mut ProgramBuilder<T, V, O>,
            zero_fn: &mut Option<ZeroFn>,
            r#type: &T,
        ) -> Result<AtomId, TracingError> {
            if let Some(zero_fn) = zero_fn {
                return zero_fn(builder, r#type);
            }
            let outputs = builder.add_instruction(O::zero_operation(r#type.clone()), Vec::new())?;
            check_count!("output", outputs, 1, TracingError);
            Ok(outputs[0])
        }

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
        fn accumulate<T: Type, V: Traceable<T>, O: Operation<T> + SupportsAdd<T, V>>(
            builder: &Rc<RefCell<ProgramBuilder<T, V, O>>>,
            adjoints: &mut [Option<AtomId>],
            atom: AtomId,
            contribution: AtomId,
        ) -> Result<(), TracingError> {
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

        // The pullback outputs are the accumulated cotangents for the primal inputs. Disconnected primal inputs are
        // handled by the caller-provided zero materializer so ordinary and active-context transposition share the
        // reverse walk while choosing different representations for standalone zeros.
        let mut outputs = program
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
                        zero(&mut builder_borrow, &mut zero_fn, input_atom.r#type().as_ref())
                    }
                }
            })
            .collect::<Result<Vec<_>, TracingError>>()?;

        if zero_fn.is_some() {
            let mut builder_borrow = builder.borrow_mut();
            let mut atom_remapping = vec![None; builder_borrow.atoms.len()];
            let instructions = std::mem::take(&mut builder_borrow.instructions);
            let mut rewritten_instructions = Vec::with_capacity(instructions.len());
            for instruction in instructions {
                let (operation, instruction_inputs, instruction_outputs) = instruction.into_parts();
                if let Some(zero_operation) = operation.as_zero_operation()
                    && instruction_outputs.len() == 1
                    && instruction_inputs.is_empty()
                {
                    // Zero operations in traced pullbacks have no inputs from which interpretation can recover a
                    // tracing context, and so active-context callers materialize each one as a constant and remap
                    // its uses before the pullback program is built.
                    let zero = zero(&mut builder_borrow, &mut zero_fn, zero_operation.r#type())?;
                    let remapped_output = atom_remapping
                        .get_mut(instruction_outputs[0].index())
                        .ok_or(TracingError::UnboundAtomId { id: instruction_outputs[0] })?;
                    *remapped_output = Some(zero);
                } else {
                    let inputs = instruction_inputs
                        .into_iter()
                        .map(|atom| {
                            Ok(atom_remapping
                                .get(atom.index())
                                .ok_or(TracingError::UnboundAtomId { id: atom })?
                                .unwrap_or(atom))
                        })
                        .collect::<Result<Vec<_>, TracingError>>()?;
                    rewritten_instructions.push(Instruction::new(operation, inputs, instruction_outputs));
                }
            }
            builder_borrow.instructions = rewritten_instructions;
            for output in &mut outputs {
                if let Some(remapped) = atom_remapping
                    .get(output.index())
                    .ok_or(TracingError::UnboundAtomId { id: *output })?
                    .as_ref()
                    .copied()
                {
                    *output = remapped;
                }
            }
        }

        // Consume the active builder to build the pullback, leaving this context with a fresh empty builder
        // for any subsequent tracing work.
        drop(builder);
        let builder = self.replace_builder(Rc::new(RefCell::new(ProgramBuilder::new())));
        let builder = match Rc::try_unwrap(builder) {
            Ok(builder) => builder.into_inner(),
            Err(_) => return Err(TracingError::EscapedProgramBuilder),
        };
        let program = builder.build(outputs, program.output_structure().clone(), program.input_structure().clone())?;
        if zero_fn.is_some() { program.into_simplified() } else { Ok(program) }
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
        let parent_builder = self.replace_builder(Rc::new(RefCell::new(ProgramBuilder::new())));
        let result = self.transpose(program);
        self.replace_builder(parent_builder);
        result
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::marker::PhantomData;
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::differentiation::Cotangent;
    use crate::macros::check_count;
    use crate::operations::Operation;
    use crate::operations::arithmetic::SupportsAdd;
    use crate::operations::constants::{SupportsZero, ZeroOperation};
    use crate::operations::scalars::ScalarOperation;
    use crate::parameters::Placeholder;
    use crate::tracing::contexts::{Context, TracingContext};
    use crate::tracing::domains::{DomainTracer, ProgramTracingDomain, ScalarDomain};
    use crate::tracing::{
        Atom, AtomId, Instruction, Program, ProgramBuilder, ProgramTracingContext, Traceable, TracingError,
    };
    use crate::types::{DataType, TypeError};

    use super::TransposableOperation;

    type TestTracingValue<'domain> = DomainTracer<'domain, ScalarDomain<f64>>;

    /// Test-only linear operation type used to exercise transposition validation paths. Most variants model tiny scalar
    /// primitives so the generated programs stay readable. The sentinel variants intentionally violate transpose rule
    /// contracts or builder ownership rules. Built-in scalar operations cannot represent those failures because their
    /// transpose implementations are valid by construction.
    #[derive(Clone, Debug)]
    enum TestLinearOperation {
        /// Single-input passthrough used when a test needs a live instruction whose transpose forwards its cotangent.
        Identity,

        /// Two-input addition used to verify cotangent accumulation through repeated primal inputs.
        Add,

        /// Single-input, two-output operation used to verify that unused operation results are passed to transpose
        /// rules as structural zero cotangents.
        TwoOutputs,

        /// Single-input operation whose transpose stages a zero operation as the input cotangent contribution.
        /// Active-context transposition has a rewrite path that materializes staged zero operations as constants in
        /// the surrounding trace. Built-in scalar operations do not stage that exact failure-mode-shaped zero
        /// contribution, so this sentinel keeps that path directly covered.
        StagedZeroContribution,

        /// Single-input operation whose transpose deliberately returns no input cotangent contributions. This violates
        /// the transpose-rule arity contract and verifies that the transposition pass rejects malformed custom rules
        /// instead of silently dropping cotangents.
        BadArity,

        /// Single-input operation whose transpose deliberately returns a cotangent staged in another builder. This
        /// verifies that the transposition pass rejects contributions from foreign builders before their atom IDs can
        /// alias unrelated atoms in the destination pullback.
        ForeignContribution,

        /// Real zero operation wrapper used by the generic `SupportsZero` implementation for this test operation enum.
        Zero(ZeroOperation<DataType>),
    }

    impl Operation<DataType> for TestLinearOperation {
        #[inline]
        fn name(&self) -> &'static str {
            match self {
                Self::Identity => "identity",
                Self::Add => "add",
                Self::TwoOutputs => "two_outputs",
                Self::StagedZeroContribution => "staged_zero_contribution",
                Self::BadArity => "bad_arity",
                Self::ForeignContribution => "foreign_contribution",
                Self::Zero(_) => "zero",
            }
        }

        fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
            match self {
                Self::Identity | Self::StagedZeroContribution | Self::BadArity | Self::ForeignContribution => {
                    check_count!("input", input_types, 1, TypeError);
                    Ok(vec![input_types[0].clone()])
                }
                Self::Add => {
                    check_count!("input", input_types, 2, TypeError);
                    Ok(vec![input_types[0].clone()])
                }
                Self::TwoOutputs => {
                    check_count!("input", input_types, 1, TypeError);
                    Ok(vec![input_types[0].clone(), input_types[0].clone()])
                }
                Self::Zero(zero) => zero.infer_output_types(input_types),
            }
        }

        fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
            match self {
                Self::Zero(zero) => zero.render(formatter, indentation),
                _ => formatter.write_str(self.name()),
            }
        }
    }

    impl<V: Traceable<DataType>> SupportsAdd<DataType, V> for TestLinearOperation {
        #[inline]
        fn add_operation() -> Self {
            Self::Add
        }
    }

    impl<V: Traceable<DataType>> SupportsZero<DataType, V> for TestLinearOperation {
        #[inline]
        fn zero_operation(r#type: DataType) -> Self {
            Self::Zero(ZeroOperation::new(r#type))
        }

        #[inline]
        fn as_zero_operation(&self) -> Option<&ZeroOperation<DataType>> {
            match self {
                Self::Zero(zero) => Some(zero),
                _ => None,
            }
        }
    }

    impl<V: Traceable<DataType>> TransposableOperation<DataType, V, TestLinearOperation> for TestLinearOperation {
        fn transpose<'transpose>(
            &self,
            context: &mut ProgramTracingContext<'transpose, DataType, V, TestLinearOperation>,
            output_cotangents: &[Cotangent<'transpose, DataType, V, TestLinearOperation>],
        ) -> Result<Vec<Cotangent<'transpose, DataType, V, TestLinearOperation>>, TracingError> {
            match self {
                Self::Identity => {
                    check_count!("output", output_cotangents, 1, TracingError);
                    Ok(vec![output_cotangents[0].clone()])
                }
                Self::Add => {
                    check_count!("output", output_cotangents, 1, TracingError);
                    Ok(vec![output_cotangents[0].clone(), output_cotangents[0].clone()])
                }
                Self::TwoOutputs => {
                    check_count!("output", output_cotangents, 2, TracingError);
                    assert!(output_cotangents[1].is_zero());
                    Ok(vec![output_cotangents[0].clone()])
                }
                Self::StagedZeroContribution => {
                    check_count!("output", output_cotangents, 1, TracingError);
                    let zero = {
                        let mut builder = context.builder().borrow_mut();
                        let outputs =
                            builder.add_instruction(Self::Zero(ZeroOperation::new(DataType::F64)), Vec::new())?;
                        check_count!("output", outputs, 1, TracingError);
                        outputs[0]
                    };
                    Ok(vec![Cotangent::staged(context.tracer(zero, None))])
                }
                Self::BadArity => {
                    check_count!("output", output_cotangents, 1, TracingError);
                    Ok(Vec::new())
                }
                Self::ForeignContribution => {
                    check_count!("output", output_cotangents, 1, TracingError);
                    let foreign_builder = Rc::new(RefCell::new(ProgramBuilder::new()));
                    let foreign_context = ProgramTracingContext::new(context.domain(), foreign_builder);
                    Ok(vec![Cotangent::staged(foreign_context.input(DataType::F64))])
                }
                Self::Zero(_) => {
                    check_count!("output", output_cotangents, 1, TracingError);
                    Ok(Vec::new())
                }
            }
        }
    }

    #[test]
    fn test_program_transpose_identity() {
        let mut builder = ProgramBuilder::<DataType, f64, TestLinearOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(TestLinearOperation::Identity, vec![input]).unwrap()[0];
        let program = builder.build::<f64, f64>(vec![output], Placeholder, Placeholder).unwrap();
        let transposed = program.transpose().unwrap();
        assert_eq!(transposed.input_ids(), &[AtomId::new(0)]);
        assert_eq!(transposed.output_ids(), &[AtomId::new(0)]);
        assert!(transposed.instructions().is_empty());
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64 .
                in (%0)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_program_transpose_accumulates_contributions_to_repeated_input() {
        let mut builder = ProgramBuilder::<DataType, f64, TestLinearOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(TestLinearOperation::Add, vec![input, input]).unwrap()[0];
        let program = builder.build::<f64, f64>(vec![output], Placeholder, Placeholder).unwrap();
        let transposed = program.transpose().unwrap();
        assert_eq!(transposed.input_ids(), &[AtomId::new(0)]);
        assert_eq!(transposed.output_ids(), &[AtomId::new(1)]);
        assert_eq!(transposed.instructions().len(), 1);
        assert!(matches!(transposed.instructions()[0].operation(), TestLinearOperation::Add));
        assert_eq!(transposed.instructions()[0].inputs(), &[AtomId::new(0), AtomId::new(0)]);
        assert_eq!(transposed.instructions()[0].outputs(), &[AtomId::new(1)]);
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = add %0 %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_program_transpose_passes_zero_for_unused_instruction_output() {
        let mut builder = ProgramBuilder::<DataType, f64, TestLinearOperation>::new();
        let input = builder.add_input(DataType::F64);
        let outputs = builder.add_instruction(TestLinearOperation::TwoOutputs, vec![input]).unwrap().to_vec();
        let program = builder.build::<f64, f64>(vec![outputs[0]], Placeholder, Placeholder).unwrap();
        let transposed = program.transpose().unwrap();
        assert_eq!(outputs, &[AtomId::new(1), AtomId::new(2)]);
        assert_eq!(transposed.input_ids(), &[AtomId::new(0)]);
        assert_eq!(transposed.output_ids(), &[AtomId::new(0)]);
        assert!(transposed.instructions().is_empty());
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64 .
                in (%0)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_program_transpose_rejects_invalid_rule_input_cotangent_count() {
        let mut builder = ProgramBuilder::<DataType, f64, TestLinearOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(TestLinearOperation::BadArity, vec![input]).unwrap()[0];
        let program = builder.build::<f64, f64>(vec![output], Placeholder, Placeholder).unwrap();
        assert!(matches!(program.transpose(), Err(TracingError::InvalidInputCount { expected: 1, got: 0 }),));
    }

    #[test]
    fn test_program_transpose_rejects_foreign_builder_contribution() {
        let mut builder = ProgramBuilder::<DataType, f64, TestLinearOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(TestLinearOperation::ForeignContribution, vec![input]).unwrap()[0];
        let program = builder.build::<f64, f64>(vec![output], Placeholder, Placeholder).unwrap();
        assert!(matches!(program.transpose(), Err(TracingError::MismatchedProgramBuilders),));
    }

    #[test]
    fn test_program_transpose_materializes_disconnected_input_zero() {
        let mut builder = ProgramBuilder::<DataType, f64, TestLinearOperation>::new();
        builder.add_input(DataType::F64);
        let program = builder.build::<f64, ()>(Vec::new(), Placeholder, ()).unwrap();
        let transposed = program.transpose().unwrap();
        assert!(transposed.input_ids().is_empty());
        assert_eq!(transposed.output_ids(), &[AtomId::new(0)]);
        assert_eq!(transposed.instructions().len(), 1);
        assert!(transposed.instructions()[0].inputs().is_empty());
        assert_eq!(transposed.instructions()[0].outputs(), &[AtomId::new(0)]);
        assert!(matches!(
            transposed.instructions()[0].operation(),
            TestLinearOperation::Zero(zero) if zero.r#type() == &DataType::F64,
        ));
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda  .
                let %0:f64 = zero [type=f64]
                in (%0)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_program_transpose_skips_dead_instruction() {
        let mut builder = ProgramBuilder::<DataType, f64, TestLinearOperation>::new();
        let dead_input = builder.add_input(DataType::F64);
        let live_input = builder.add_input(DataType::F64);
        let dead_output = builder.add_instruction(TestLinearOperation::BadArity, vec![dead_input]).unwrap()[0];
        let output = builder.add_instruction(TestLinearOperation::Identity, vec![live_input]).unwrap()[0];
        let program = builder.build::<(f64, f64), f64>(vec![output], (Placeholder, Placeholder), Placeholder).unwrap();
        let transposed = program.transpose().unwrap();
        assert_eq!(dead_output, AtomId::new(2));
        assert_eq!(transposed.input_ids(), &[AtomId::new(0)]);
        assert_eq!(transposed.output_ids(), &[AtomId::new(1), AtomId::new(0)]);
        assert_eq!(transposed.instructions().len(), 1);
        assert!(transposed.instructions()[0].inputs().is_empty());
        assert_eq!(transposed.instructions()[0].outputs(), &[AtomId::new(1)]);
        assert!(matches!(
            transposed.instructions()[0].operation(),
            TestLinearOperation::Zero(zero) if zero.r#type() == &DataType::F64,
        ));
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = zero [type=f64]
                in (%1, %0)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_program_tracing_context_transpose_nested_restores_parent_builder_on_success() {
        let domain = ProgramTracingDomain::<DataType, f64, TestLinearOperation>::new();
        let parent_builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, TestLinearOperation>::new()));
        let mut context = ProgramTracingContext::new(&domain, parent_builder.clone());
        let parent_input = context.input(DataType::F64);
        let mut nested_builder = ProgramBuilder::<DataType, f64, TestLinearOperation>::new();
        let nested_input = nested_builder.add_input(DataType::F64);
        let nested_output =
            nested_builder.add_instruction(TestLinearOperation::Identity, vec![nested_input]).unwrap()[0];
        let nested_program = nested_builder.build::<f64, f64>(vec![nested_output], Placeholder, Placeholder).unwrap();
        let transposed = context.transpose_nested(&nested_program).unwrap();
        assert_eq!(parent_input.atom_id(), Ok(AtomId::new(0)));
        assert!(Rc::ptr_eq(context.builder(), &parent_builder));
        let parent_builder = parent_builder.borrow();
        assert_eq!(parent_builder.atoms().len(), 1);
        assert_eq!(parent_builder.input_ids(), &[AtomId::new(0)]);
        assert!(parent_builder.instructions().is_empty());
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64 .
                in (%0)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_program_tracing_context_transpose_nested_restores_parent_builder_on_failure() {
        let domain = ProgramTracingDomain::<DataType, f64, TestLinearOperation>::new();
        let parent_builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, TestLinearOperation>::new()));
        let mut context = ProgramTracingContext::new(&domain, parent_builder.clone());
        let parent_input = context.input(DataType::F64);
        let mut nested_builder = ProgramBuilder::<DataType, f64, TestLinearOperation>::new();
        let nested_input = nested_builder.add_input(DataType::F64);
        let nested_output =
            nested_builder.add_instruction(TestLinearOperation::BadArity, vec![nested_input]).unwrap()[0];
        let nested_program = nested_builder.build::<f64, f64>(vec![nested_output], Placeholder, Placeholder).unwrap();
        assert!(matches!(
            context.transpose_nested(&nested_program),
            Err(TracingError::InvalidInputCount { expected: 1, got: 0 }),
        ));
        assert_eq!(parent_input.atom_id(), Ok(AtomId::new(0)));
        assert!(Rc::ptr_eq(context.builder(), &parent_builder));
        let parent_builder = parent_builder.borrow();
        assert_eq!(parent_builder.atoms().len(), 1);
        assert_eq!(parent_builder.input_ids(), &[AtomId::new(0)]);
        assert!(parent_builder.instructions().is_empty());
    }

    #[test]
    fn test_program_transpose_reports_unbound_input_atom() {
        let input = AtomId::new(0);
        let program = Program::<DataType, f64, TestLinearOperation, f64, ()> {
            atoms: Vec::new(),
            input_ids: vec![input],
            output_ids: Vec::new(),
            instructions: Vec::new(),
            input_structure: Placeholder,
            output_structure: (),
            marker: PhantomData,
        };
        assert!(matches!(program.transpose(), Err(TracingError::UnboundAtomId { id }) if id == input));
    }

    #[test]
    fn test_program_transpose_reports_unbound_instruction_output_atom() {
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
        assert!(matches!(
            program.transpose(),
            Err(TracingError::UnboundAtomId { id }) if id == missing_output,
        ));
    }

    #[test]
    fn test_tracing_context_transpose_materializes_disconnected_input_zero_as_constant() {
        let domain = ScalarDomain::<f64>::new();
        let outer_builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let tracing_context = TracingContext::new(&domain, outer_builder.clone());
        let mut builder = ProgramBuilder::<DataType, TestTracingValue<'_>, TestLinearOperation>::new();
        let connected_input = builder.add_input(DataType::F64);
        let disconnected_input = builder.add_input(DataType::F64);
        let program = builder
            .build::<Vec<TestTracingValue<'_>>, TestTracingValue<'_>>(
                vec![connected_input],
                vec![Placeholder, Placeholder],
                Placeholder,
            )
            .unwrap();
        let pullback = tracing_context.transpose(&program).unwrap();
        assert_eq!(disconnected_input, AtomId::new(1));
        assert_eq!(pullback.input_ids(), &[AtomId::new(0)]);
        assert_eq!(pullback.output_ids(), &[AtomId::new(0), AtomId::new(1)]);
        assert!(pullback.instructions().is_empty());
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = const
                in (%0, %1)
            "}
            .trim_end(),
        );
        let zero_output = pullback.output_ids()[1];
        let Atom::Constant(zero) = &pullback.atoms()[zero_output.index()] else {
            panic!("disconnected input cotangent was not materialized as a pullback constant");
        };
        assert_eq!(zero.atom_id(), Ok(AtomId::new(0)));
        assert!(Rc::ptr_eq(zero.builder(), tracing_context.builder()));
        let outer_builder = outer_builder.borrow();
        assert_eq!(outer_builder.atoms().len(), 1);
        assert_eq!(outer_builder.instructions().len(), 1);
        assert!(outer_builder.instructions()[0].inputs().is_empty());
        assert_eq!(outer_builder.instructions()[0].outputs(), &[AtomId::new(0)]);
        assert!(matches!(
            outer_builder.instructions()[0].operation(),
            ScalarOperation::Zero(zero) if zero.r#type() == &DataType::F64,
        ));
    }

    #[test]
    fn test_tracing_context_transpose_materializes_staged_zero_contribution_as_constant() {
        let domain = ScalarDomain::<f64>::new();
        let outer_builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let tracing_context = TracingContext::new(&domain, outer_builder.clone());
        let mut builder = ProgramBuilder::<DataType, TestTracingValue<'_>, TestLinearOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(TestLinearOperation::StagedZeroContribution, vec![input]).unwrap()[0];
        let program = builder
            .build::<TestTracingValue<'_>, TestTracingValue<'_>>(vec![output], Placeholder, Placeholder)
            .unwrap();
        let pullback = tracing_context.transpose(&program).unwrap();
        assert_eq!(pullback.input_ids(), &[AtomId::new(0)]);
        assert_eq!(pullback.output_ids(), &[AtomId::new(1)]);
        assert!(pullback.instructions().is_empty());
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = const
                in (%1)
            "}
            .trim_end(),
        );
        let zero_output = pullback.output_ids()[0];
        let Atom::Constant(zero) = &pullback.atoms()[zero_output.index()] else {
            panic!("staged zero contribution was not materialized as a pullback constant");
        };
        assert_eq!(zero.atom_id(), Ok(AtomId::new(0)));
        assert!(Rc::ptr_eq(zero.builder(), tracing_context.builder()));
        let outer_builder = outer_builder.borrow();
        assert_eq!(outer_builder.atoms().len(), 1);
        assert_eq!(outer_builder.instructions().len(), 1);
        assert!(outer_builder.instructions()[0].inputs().is_empty());
        assert_eq!(outer_builder.instructions()[0].outputs(), &[AtomId::new(0)]);
        assert!(matches!(
            outer_builder.instructions()[0].operation(),
            ScalarOperation::Zero(zero) if zero.r#type() == &DataType::F64,
        ));
    }
}
