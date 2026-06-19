use std::cell::RefCell;
use std::rc::Rc;

use crate::contexts::{Context, StagingContext};
use crate::differentiation::{Cotangent, DifferentiableType};
use crate::domains::AbstractDomain;
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::AddOperation;
use crate::operations::constants::ZeroOperation;
use crate::parameters::Parameterized;
use crate::programs::{AtomId, Program, ProgramBuilder, ProgramError, Value};
use crate::tracing::{AbstractTracingContext, DomainTracer, TracingContext};
use crate::types::{Type, Typed};

// TODO(eaplatanios): Review this module again.

/// Represents [`Operation`]s that provide a transpose rule for linear [`Program`]s. For a linear [`Instruction`]
/// `y = L(x)`, [`transpose`](Self::transpose) receives symbolic [`Cotangent`]s for `y` and returns symbolic
/// cotangent contributions for `x`, representing the transposed cotangent. Rules may reuse existing cotangents,
/// return [`Cotangent::Zero`] for structural zeros, or stage additional linear operations in the active
/// [`AbstractTracingContext`]. The rule does not receive concrete primal values. Instead, it receives the staged
/// types of the instruction's inputs, and any further metadata must be encoded in the operation itself.
///
/// Refer to the documentation of [`Program::transpose`] for more information what _transposition_ means here and how
/// it relates to the algebraic notion of transposition.
pub trait TransposableOperation<T: Type, V: Value<T>, O: Operation<T>>: Operation<T> {
    /// Applies this operation's transpose rule to the provided symbolic output cotangents. The returned vector must
    /// contain one entry per operation input. Each [`Cotangent::Staged`] value is a staged cotangent contribution in
    /// the active [`AbstractTracingContext`], and each [`Cotangent::Zero`] means that the corresponding input receives
    /// a structural zero from this operation.
    ///
    /// # Parameters
    ///
    ///   - `context`: Active [`AbstractTracingContext`] in which rules may stage additional linear operations.
    ///   - `input_types`: Staged types of the instruction's inputs, in operation-input order. Rules whose cotangent
    ///     shapes are not recoverable from the operation payload alone (e.g., a broadcast operation's pre-broadcast
    ///     shape) read them from here.
    ///   - `output_cotangents`: Symbolic cotangents for the instruction's outputs, in operation-output order.
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<'transpose, T, V, O>,
        input_types: &[&T],
        output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, ProgramError>;
}

impl<
    T: DifferentiableType,
    V: Value<T>,
    O: TransposableOperation<T, V, O> + From<ZeroOperation<T>> + From<AddOperation>,
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
    /// Disconnected primal inputs are emitted as [`ZeroOperation`]s, which the value type's [`Zero`](crate::Zero)
    /// implementation evaluates at interpretation time. For linear programs whose values are [`Tracer`](crate::Tracer)s
    /// from an outer trace, use [`TracingContext::transpose`] instead so that those disconnected-input zeros can be
    /// materialized in the surrounding tracing context.
    #[inline]
    pub fn transpose(&self) -> Result<Program<T, V, O, Output, Input>, ProgramError> {
        let builder = Rc::new(RefCell::new(ProgramBuilder::<T, V, O>::new()));
        let domain = AbstractDomain::new();
        let mut context = AbstractTracingContext::new(&domain, builder);
        context.transpose(self)
    }
}

impl<'context, C: Context<Type: DifferentiableType, Operation: From<ZeroOperation<C::Type>>>>
    TracingContext<'context, C>
{
    /// Transposes the provided traced linear [`Program`] whose values are [`Tracer`](crate::Tracer)s belonging to this
    /// outer [`TracingContext`]. This uses the same reverse-walk implementation as [`Program::transpose`] in a fresh
    /// [`AbstractTracingContext`].
    ///
    /// Use this method when transposing a linear program inside an outer trace, such as when staging a traced
    /// reverse-mode pullback. Use [`Program::transpose`] for ordinary complete linear program transposition, and use
    /// [`AbstractTracingContext::transpose`] only when you already own the destination [`ProgramBuilder`] and want to
    /// run the lower-level transposition algorithm directly.
    ///
    /// Disconnected primal inputs and transpose-rule-staged structural zeros are emitted as input-free
    /// [`ZeroOperation`] instructions in the pullback. These are materialized at interpretation time: a pullback
    /// [`ZeroOperation`] interpreted over outer-trace [`Tracer`](crate::Tracer)s stages a typed zero into the
    /// surrounding [`TracingContext`] through the threaded interpretation context, and so backends whose traced
    /// constants are abstract metadata do not need to materialize a runtime value just to transpose an enclosing
    /// traced program.
    #[inline]
    pub fn transpose<
        Input: Parameterized<DomainTracer<'context, C>>,
        Output: Parameterized<DomainTracer<'context, C>>,
        O: TransposableOperation<C::Type, DomainTracer<'context, C>, O>
            + From<ZeroOperation<C::Type>>
            + From<AddOperation>,
    >(
        &self,
        program: &Program<C::Type, DomainTracer<'context, C>, O, Input, Output>,
    ) -> Result<Program<C::Type, DomainTracer<'context, C>, O, Output, Input>, ProgramError> {
        let builder = Rc::new(RefCell::new(ProgramBuilder::<C::Type, DomainTracer<'context, C>, O>::new()));
        let domain = AbstractDomain::new();
        let mut context = AbstractTracingContext::new(&domain, builder);
        context.transpose(program)
    }
}

impl<
    'domain,
    T: 'domain + Type + DifferentiableType,
    V: 'domain + Value<T>,
    O: 'domain + TransposableOperation<T, V, O> + From<ZeroOperation<T>> + From<AddOperation>,
> TracingContext<'domain, AbstractDomain<T, V, O>>
{
    /// Transposes the provided linear [`Program`] using this [`TracingContext`]'s [`ProgramBuilder`]. This is the
    /// builder-level implementation behind [`Program::transpose`]. Refer to the documentation of [`Program::transpose`]
    /// for the conceptual relationship between program transposition, algebraic transposition, pushforward functions,
    /// and pullback functions. This function is for callers that already own a [`AbstractTracingContext`] and need the
    /// transposed program to be staged through that context's [`ProgramBuilder`].
    ///
    /// This function treats [`builder`](Self::builder) as the destination for the transposed program, records cotangent
    /// inputs for the primal outputs, walks `program` in reverse instruction order, and transposes each [`Instruction`]
    /// using [`TransposableOperation::transpose`]. The active [`ProgramBuilder`] is consumed when the pullback is
    /// built. On success, this context is left with a fresh empty builder. A transpose rule that needs to transpose a
    /// nested subprogram (e.g., a captured control-flow branch) should instead call [`Program::transpose`], which
    /// transposes it in its own fresh context without touching the surrounding builder.
    #[inline]
    pub fn transpose<Input: Parameterized<V>, Output: Parameterized<V>>(
        &mut self,
        program: &Program<T, V, O, Input, Output>,
    ) -> Result<Program<T, V, O, Output, Input>, ProgramError> {
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
        fn accumulate<T: Type, V: Value<T>, O: Operation<T> + From<AddOperation>>(
            builder: &Rc<RefCell<ProgramBuilder<T, V, O>>>,
            adjoints: &mut [Option<AtomId>],
            atom: AtomId,
            contribution: AtomId,
        ) -> Result<(), ProgramError> {
            // Contributions must already be atoms in the transpose builder. Otherwise the `AtomId` could alias an
            // unrelated atom index and corrupt the pullback graph.
            if builder.borrow().atoms().get(contribution.index()).is_none() {
                return Err(ProgramError::UnboundAtomId { id: contribution }.into());
            }

            // Locate the primal atom's adjoint slot. An out-of-range slot means the input program is malformed.
            let adjoint = adjoints.get_mut(atom.index()).ok_or(ProgramError::UnboundAtomId { id: atom })?;

            // If this atom already has a cotangent, stage an add so both contributions flow into one accumulated
            // adjoint. Otherwise, keep the first contribution directly and avoid emitting an unnecessary add.
            *adjoint = Some(match *adjoint {
                Some(existing) => {
                    let mut builder_borrow = builder.borrow_mut();
                    let outputs = builder_borrow.add_instruction(AddOperation, vec![existing, contribution])?;
                    check_count!("output", outputs, 1, ProgramError);
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

        // Seed the reverse pass with one cotangent input for each primal output, typed with that output's cotangent
        // type (e.g., swapping unreduced and reduced sharding axes for arrays). The adjoint table is indexed by atoms
        // from the original program, and each slot stores the staged pullback atom that currently represents the
        // accumulated cotangent for that primal atom.
        let mut adjoints = vec![None; program.atoms().len()];
        for output in program.output_ids().iter().copied() {
            let output_atom = program.atoms().get(output.index()).ok_or(ProgramError::UnboundAtomId { id: output })?;
            let cotangent_input = builder.borrow_mut().add_input(output_atom.r#type().cotangent());
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
                if adjoints.get(output.index()).ok_or(ProgramError::UnboundAtomId { id: output })?.is_some() {
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
                    match adjoints.get(output.index()).copied().ok_or(ProgramError::UnboundAtomId { id: output })? {
                        Some(atom) => Cotangent::staged(self.tracer(atom, None)),
                        None => Cotangent::Zero,
                    },
                );
            }

            // Apply the primitive transpose rule and require exactly one cotangent contribution per primal input. This
            // prevents malformed rules from silently dropping or inventing cotangents through iterator truncation.
            let input_types = instruction
                .inputs()
                .iter()
                .copied()
                .map(|input| {
                    program
                        .atoms()
                        .get(input.index())
                        .map(Typed::r#type)
                        .ok_or(ProgramError::UnboundAtomId { id: input })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let input_types = input_types.iter().map(|r#type| r#type.as_ref()).collect::<Vec<_>>();
            let input_cotangents = instruction.operation().transpose(
                self,
                input_types.as_slice(),
                instruction_output_cotangents.as_slice(),
            )?;
            check_count!("input", input_cotangents, instruction.inputs().len(), ProgramError);
            for (input, contribution) in instruction.inputs().iter().copied().zip(input_cotangents) {
                if let Some(contribution) = contribution.as_staged() {
                    // Staged contributions must belong to this pullback builder before their atom IDs can be accumulated.
                    if !Rc::ptr_eq(&builder, contribution.builder()) {
                        return Err(ProgramError::MismatchedProgramBuilders);
                    }
                    accumulate::<T, V, O>(&builder, adjoints.as_mut_slice(), input, contribution.atom_id()?)?;
                }
            }
        }
        instruction_output_cotangents.clear();

        // The pullback outputs are the accumulated cotangents for the primal inputs. Disconnected primal inputs are
        // emitted as input-free [`ZeroOperation`] instructions, which the value type's [`Zero`](crate::Zero)
        // implementation evaluates at interpretation time.
        let outputs = program
            .input_ids()
            .iter()
            .copied()
            .map(|input| {
                match adjoints.get(input.index()).copied().ok_or(ProgramError::UnboundAtomId { id: input })? {
                    Some(adjoint) => Ok::<AtomId, ProgramError>(adjoint),
                    None => {
                        let input_atom =
                            program.atoms().get(input.index()).ok_or(ProgramError::UnboundAtomId { id: input })?;
                        let mut builder_borrow = builder.borrow_mut();
                        let outputs = builder_borrow
                            .add_instruction(ZeroOperation::new(input_atom.r#type().cotangent()), Vec::new())?;
                        check_count!("output", outputs, 1, ProgramError);
                        Ok(outputs[0])
                    }
                }
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;

        // Consume the active builder to build the pullback, leaving this context with a fresh empty builder
        // for any subsequent tracing work.
        drop(builder);
        let builder = self.replace_builder(Rc::new(RefCell::new(ProgramBuilder::new())));
        let builder = match Rc::try_unwrap(builder) {
            Ok(builder) => builder.into_inner(),
            Err(_) => return Err(ProgramError::EscapedProgramBuilder),
        };
        builder.build(outputs, program.output_structure().clone(), program.input_structure().clone())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::marker::PhantomData;
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::contexts::StagingContext;
    use crate::differentiation::Cotangent;
    use crate::macros::check_count;
    use crate::operations::Operation;
    use crate::operations::arithmetic::AddOperation;
    use crate::operations::constants::ZeroOperation;
    use crate::operations::scalars::ScalarOperation;
    use crate::parameters::Placeholder;
    use crate::programs::{Atom, AtomId, Instruction, Program, ProgramBuilder, ProgramError, Value};
    use crate::scalars::ScalarDomain;
    use crate::tracing::{AbstractTracingContext, DomainTracer, TracingContext};
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

        /// Single-input operation whose transpose stages a [`ZeroOperation`] as the input cotangent contribution. The
        /// staged zero remains an input-free [`ZeroOperation`] instruction in the pullback and is materialized at
        /// interpretation time. Built-in scalar operations do not stage that exact structural-zero contribution, so
        /// this sentinel keeps that path directly covered.
        StagedZeroContribution,

        /// Single-input operation whose transpose deliberately returns no input cotangent contributions. This violates
        /// the transpose-rule arity contract and verifies that the transposition pass rejects malformed custom rules
        /// instead of silently dropping cotangents.
        BadArity,

        /// Single-input operation whose transpose deliberately returns a cotangent staged in another builder. This
        /// verifies that the transposition pass rejects contributions from foreign builders before their atom IDs can
        /// alias unrelated atoms in the destination pullback.
        ForeignContribution,

        /// Real zero operation wrapper used by the `From<ZeroOperation>`/`TryFrom<ZeroOperation>` conversions for this
        /// test operation enum.
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

    impl From<AddOperation> for TestLinearOperation {
        #[inline]
        fn from(_operation: AddOperation) -> Self {
            Self::Add
        }
    }

    impl From<ZeroOperation<DataType>> for TestLinearOperation {
        #[inline]
        fn from(operation: ZeroOperation<DataType>) -> Self {
            Self::Zero(operation)
        }
    }

    impl<'operation> TryFrom<&'operation TestLinearOperation> for &'operation ZeroOperation<DataType> {
        type Error = ();

        #[inline]
        fn try_from(value: &'operation TestLinearOperation) -> Result<Self, ()> {
            match value {
                TestLinearOperation::Zero(zero) => Ok(zero),
                _ => Err(()),
            }
        }
    }

    impl<V: Value<DataType>> TransposableOperation<DataType, V, TestLinearOperation> for TestLinearOperation {
        fn transpose<'transpose>(
            &self,
            context: &mut AbstractTracingContext<'transpose, DataType, V, TestLinearOperation>,
            _input_types: &[&DataType],
            output_cotangents: &[Cotangent<'transpose, DataType, V, TestLinearOperation>],
        ) -> Result<Vec<Cotangent<'transpose, DataType, V, TestLinearOperation>>, ProgramError> {
            match self {
                Self::Identity => {
                    check_count!("output", output_cotangents, 1, ProgramError);
                    Ok(vec![output_cotangents[0].clone()])
                }
                Self::Add => {
                    check_count!("output", output_cotangents, 1, ProgramError);
                    Ok(vec![output_cotangents[0].clone(), output_cotangents[0].clone()])
                }
                Self::TwoOutputs => {
                    check_count!("output", output_cotangents, 2, ProgramError);
                    assert!(output_cotangents[1].is_zero());
                    Ok(vec![output_cotangents[0].clone()])
                }
                Self::StagedZeroContribution => {
                    check_count!("output", output_cotangents, 1, ProgramError);
                    let zero = {
                        let mut builder = context.builder().borrow_mut();
                        let outputs =
                            builder.add_instruction(Self::Zero(ZeroOperation::new(DataType::F64)), Vec::new())?;
                        check_count!("output", outputs, 1, ProgramError);
                        outputs[0]
                    };
                    Ok(vec![Cotangent::staged(context.tracer(zero, None))])
                }
                Self::BadArity => {
                    check_count!("output", output_cotangents, 1, ProgramError);
                    Ok(Vec::new())
                }
                Self::ForeignContribution => {
                    check_count!("output", output_cotangents, 1, ProgramError);
                    let foreign_builder = Rc::new(RefCell::new(ProgramBuilder::new()));
                    let foreign_context = AbstractTracingContext::new(context.domain(), foreign_builder);
                    Ok(vec![Cotangent::staged(foreign_context.input(DataType::F64))])
                }
                Self::Zero(_) => {
                    check_count!("output", output_cotangents, 1, ProgramError);
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
        assert!(matches!(program.transpose(), Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),));
    }

    #[test]
    fn test_program_transpose_rejects_foreign_builder_contribution() {
        let mut builder = ProgramBuilder::<DataType, f64, TestLinearOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(TestLinearOperation::ForeignContribution, vec![input]).unwrap()[0];
        let program = builder.build::<f64, f64>(vec![output], Placeholder, Placeholder).unwrap();
        assert!(matches!(program.transpose(), Err(ProgramError::MismatchedProgramBuilders),));
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
        assert!(matches!(program.transpose(), Err(ProgramError::UnboundAtomId { id }) if id == input));
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
            Err(ProgramError::UnboundAtomId { id }) if id == missing_output,
        ));
    }

    #[test]
    fn test_tracing_context_transpose_materializes_disconnected_input_zero_as_zero_instruction() {
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
        
        // The disconnected input's cotangent is emitted as an input-free `ZeroOperation` instruction in the pullback,
        // which is materialized at interpretation time rather than as a pullback constant staged at transpose time.
        assert_eq!(pullback.instructions().len(), 1);
        assert!(pullback.instructions()[0].inputs().is_empty());
        assert_eq!(pullback.instructions()[0].outputs(), &[AtomId::new(1)]);
        assert!(matches!(
            pullback.instructions()[0].operation(),
            TestLinearOperation::Zero(zero) if zero.r#type() == &DataType::F64,
        ));
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = zero [type=f64]
                in (%0, %1)
            "}
            .trim_end(),
        );
        
        // The outer tracing context is left untouched: transposition stages no zero into it.
        let outer_builder = outer_builder.borrow();
        assert!(outer_builder.atoms().is_empty());
        assert!(outer_builder.instructions().is_empty());
    }

    #[test]
    fn test_tracing_context_transpose_materializes_staged_zero_contribution_as_zero_instruction() {
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
        
        // The transpose-rule-staged structural zero stays an input-free `ZeroOperation` instruction in the pullback,
        // materialized at interpretation time rather than as a pullback constant staged at transpose time.
        assert_eq!(pullback.instructions().len(), 1);
        assert!(pullback.instructions()[0].inputs().is_empty());
        assert_eq!(pullback.instructions()[0].outputs(), &[AtomId::new(1)]);
        assert!(matches!(
            pullback.instructions()[0].operation(),
            TestLinearOperation::Zero(zero) if zero.r#type() == &DataType::F64,
        ));
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = zero [type=f64]
                in (%1)
            "}
            .trim_end(),
        );
        
        // The outer tracing context is left untouched: transposition stages no zero into it.
        let outer_builder = outer_builder.borrow();
        assert!(outer_builder.atoms().is_empty());
        assert!(outer_builder.instructions().is_empty());
    }
}
