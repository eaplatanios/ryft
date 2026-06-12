use crate::batching::BatchingError;
use crate::contexts::StagingContext;
use crate::differentiation::{Cotangent, Tangent, TransposableOperation};
use crate::domains::Domain;
use crate::macros::check_count;
use crate::operations::constants::SupportsZero;
use crate::operations::control_flow::{ConditionOperation, WhileOperation};
use crate::operations::{BooleanLike, Operation};
use crate::parameters::Placeholder;
use crate::programs::{Atom, AtomId, Instruction, Program, ProgramBuilder, ProgramError, Value};
use crate::tracing::{AbstractTracer, AbstractTracingContext, Tracer};
use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation, BatchingContext};
use crate::tracing_v2::differentiation::{NestedLinearization, SupportsNestedLinearization};
use crate::tracing_v2::{
    DifferentiableOperation, DifferentiationContext, FactorParameterizedOperation, JvpTracer, LinearOperationOf,
    ResidualFactor, ResidualizedOperation, TangentContext,
};
use crate::types::{ArrayType, Type, TypeError, Typed};

impl<'domain, E> BooleanLike for JvpTracer<'domain, E>
where
    E: DifferentiationContext<Type = ArrayType>,
    E::Value: BooleanLike,
{
    /// Returns this [`JvpTracer`] unchanged. A JVP tracer pairs a primal with a tangent, and reinterpreting only the
    /// primal payload as Boolean would silently sever that pairing, so a Boolean reinterpretation must be expressed
    /// through explicitly staged operations instead.
    #[inline]
    fn as_boolean(&self) -> Self {
        self.clone()
    }

    #[inline]
    fn boolean(&self) -> Result<bool, ProgramError> {
        self.primal().boolean()
    }
}

impl<V: Value<ArrayType> + BooleanLike> BooleanLike for ArrayBatch<V> {
    /// Returns an [`ArrayBatch`] that wraps the Boolean reinterpretation of the carried value (via the value's own
    /// [`BooleanLike::as_boolean`]) under the same batch axis.
    fn as_boolean(&self) -> Self {
        match self.batch_axis() {
            // This unwrap is safe because `as_boolean` preserves structural metadata, so the batch axis that was
            // valid for this batch remains in bounds for the reinterpreted value.
            Some(axis) => Self::mapped(self.value().as_boolean(), axis).unwrap(),
            None => Self::unbatched(self.value().as_boolean()),
        }
    }

    fn boolean(&self) -> Result<bool, ProgramError> {
        if let Some(axis) = self.batch_axis() {
            return Err(ProgramError::Concretization {
                message: format!("cannot extract a concrete boolean from a value batched along axis {axis}"),
            });
        }
        self.value().boolean()
    }
}

/// Returns a concrete cotangent atom for `cotangent`, staging a typed `Zero` op when the cotangent
/// is structurally zero. Higher-order linear rules use this when they must consume all output
/// cotangents jointly.
pub(crate) fn stage_cotangent<'transpose, T: Type, V: Value<T>, O>(
    context: &AbstractTracingContext<'transpose, T, V, O>,
    cotangent: &Cotangent<'transpose, T, V, O>,
    output_type: &T,
) -> AbstractTracer<'transpose, T, V, O>
where
    O: Operation<T> + crate::operations::constants::SupportsZero<T>,
{
    match cotangent {
        Cotangent::Staged(cotangent) => return cotangent.clone(),
        Cotangent::Zero => {}
    }
    let builder = context.builder();
    let mut builder_borrow = builder.borrow_mut();
    let output = builder_borrow.add_variable(output_type.clone());
    builder_borrow
        .instructions
        .push(Instruction::new(O::zero_operation(output_type.clone()), vec![], vec![output]));
    drop(builder_borrow);
    context.tracer(output, None)
}

/// Trait that represents linear [`Operation`] types that support/include a captured-predicate condition. The
/// captured-predicate condition is the linear-program counterpart of [`ConditionOperation`]: the Boolean predicate is
/// a primal value captured at linearization time as a residual factor, the operation inputs are exactly the branch
/// operand tangents (or cotangents), and the staged map runs the linear branch program selected by the predicate,
/// which is linear in the branch operands (the predicate itself has no tangent space). Linear operation enums
/// implement this trait so that the JVP rule of [`ConditionOperation`] can stage the captured-predicate condition
/// without knowing which linear operation type is in use.
pub trait SupportsLinearCondition<T: Type, V: Value<T>, F>: Sized {
    /// Constructs the linear-operation representation of the captured-predicate condition.
    ///
    /// # Parameters
    ///
    ///   - `predicate`: Captured Boolean predicate factor that selects the branch program to run.
    ///   - `true_branch`: Linear branch [`Program`] evaluated when the predicate is true.
    ///   - `false_branch`: Linear branch [`Program`] evaluated when the predicate is false.
    fn linear_condition_operation(
        predicate: F,
        true_branch: Program<T, V, Self, Vec<V>, Vec<V>>,
        false_branch: Program<T, V, Self, Vec<V>, Vec<V>>,
    ) -> Self;
}

/// Trait that represents linear [`Operation`] types that support/include the staged doubled-state while loop. The
/// staged while loop is the linear-program counterpart of [`WhileOperation`]: its state is the primal state followed
/// by the tangent state, its body interleaves recomputed primal operations with the body pushforward (whose
/// loop-varying residual references are rewritten into operand form), and its condition recomputes the original loop
/// predicate from the primal half of the state. Linear operation enums implement this trait so that the JVP rule of
/// [`WhileOperation`] can stage the fused loop without knowing which linear operation type is in use.
pub trait SupportsLinearWhile<T: Type, V: Value<T>, F: Value<T>, O>: Clone + Sized {
    /// Wraps `operation` as a recomputed primal operation embedded in a linear program. Fused while bodies use
    /// recomputed operations to rebuild the primal state (and the loop-varying residuals derived from it) inside the
    /// loop instead of capturing residuals once at staging time.
    fn recompute_operation(operation: O) -> Self;

    /// Constructs the nullary linear operation that materializes the captured `factor` as a program value. The while
    /// JVP rule uses residual injections to feed the loop-entry primal state into the staged linear loop, and fused
    /// programs use them to materialize nested primal program constants.
    fn residual_operation(factor: F) -> Self;

    /// Rewrites this operation's loop-varying [`ResidualFactor::Reference`] factors into operand form against
    /// `residual_atoms`, where `residual_atoms[i]` is the fused-body atom carrying residual `i`.
    ///
    /// Captured-factor linear maps whose factor is recomputed in-loop become recomputed multi-operand primal
    /// operations (for example, a scale by a referenced residual becomes a recomputed elementwise product), with the
    /// residual atom spliced into `inputs`. Every rewritten operation is wrapped in the recomputed-primal form
    /// produced by [`Self::recompute_operation`] so fused bodies carry uniform provenance. Operations carrying only
    /// closed [`ResidualFactor::Constant`] factors pass through unchanged, and operations whose residual references
    /// cannot be rewritten into operand form are rejected.
    ///
    /// # Parameters
    ///
    ///   - `residual_atoms`: Fused-body atoms carrying the recomputed residual values, indexed by residual index.
    ///   - `inputs`: Already-remapped operand atoms of this operation inside the fused body.
    fn defactorize(
        &self,
        residual_atoms: &[AtomId],
        inputs: Vec<AtomId>,
    ) -> Result<DefactorizedOperation<Self>, ProgramError>;

    /// Constructs the linear-operation representation of the fused doubled-state while loop.
    ///
    /// # Parameters
    ///
    ///   - `condition`: Extended condition [`Program`] recomputing the loop predicate from the primal half of the
    ///     doubled state.
    ///   - `body`: Fused body [`Program`] mapping `[primal_state..., tangent_state...]` to the next doubled state.
    fn linear_while_operation(
        condition: Program<T, V, Self, Vec<V>, Vec<V>>,
        body: Program<T, V, Self, Vec<V>, Vec<V>>,
    ) -> Result<Self, TypeError>;
}

/// Result of rewriting one pushforward operation's residual references into operand form (see
/// [`SupportsLinearWhile::defactorize`]).
pub enum DefactorizedOperation<O> {
    /// Defactorized operation to stage over `inputs`.
    Operation {
        /// Operation with loop-varying residual references rewritten into operand form.
        operation: O,

        /// Operand atoms of the defactorized operation, including any spliced-in residual atoms.
        inputs: Vec<AtomId>,
    },

    /// The operation reduces to forwarding an existing fused-body atom, so no instruction is staged.
    Forward {
        /// Atom carrying the operation's single output.
        atom: AtomId,
    },
}

/// Extends a residual-extended condition branch program to the joined output signature
/// `[original_outputs..., true_branch_residuals..., false_branch_residuals...]`.
///
/// `program` must already produce `[original_outputs..., own_residuals...]` (the shape produced by
/// [`linearize_nested_program`]). This helper appends one typed nullary zero instruction per peer-branch residual
/// slot and reorders the program outputs into the joined signature: the true branch (`own_residuals_first`) keeps its
/// own residuals before the peer zeros, while the false branch moves its own residuals after them. This is the
/// analog of JAX's `_join_cond_outputs`: both joined branches produce identical output signatures, with each branch
/// emitting typed zeros in the residual slots owned by the other branch.
///
/// The appended zero instructions are nullary and the reordered outputs reference existing atoms, so the direct
/// program-field extension preserves every [`Program`] invariant that [`crate::programs::ProgramBuilder`] would have
/// established.
fn join_condition_branch_outputs<V, O>(
    mut program: Program<ArrayType, V, O, Vec<V>, Vec<V>>,
    original_output_count: usize,
    peer_residual_types: &[ArrayType],
    own_residuals_first: bool,
) -> Program<ArrayType, V, O, Vec<V>, Vec<V>>
where
    V: Value<ArrayType>,
    O: Operation<ArrayType> + SupportsZero<ArrayType>,
{
    let mut zero_ids = Vec::with_capacity(peer_residual_types.len());
    for residual_type in peer_residual_types {
        let zero_id = AtomId::new(program.atoms.len());
        program.atoms.push(Atom::Variable(residual_type.clone()));
        program
            .instructions
            .push(Instruction::new(O::zero_operation(residual_type.clone()), vec![], vec![zero_id]));
        zero_ids.push(zero_id);
    }
    let own_residual_ids = program.output_ids.split_off(original_output_count);
    if own_residuals_first {
        program.output_ids.extend(own_residual_ids);
        program.output_ids.extend(zero_ids);
    } else {
        program.output_ids.extend(zero_ids);
        program.output_ids.extend(own_residual_ids);
    }
    program.output_structure = vec![Placeholder; program.output_ids.len()];
    program
}

/// Rewrites a branch pushforward's local [`ResidualFactor::Reference`]s onto the enclosing linearization residual
/// environment using `factors`, where `factors[i]` is the enclosing factor registered for the branch's residual `i`.
/// Closed [`ResidualFactor::Constant`] factors are carried over unchanged.
fn remap_branch_residual_factors<D>(
    program: &Program<ArrayType, D::Tangent, LinearOperationOf<D>, Vec<D::Tangent>, Vec<D::Tangent>>,
    factors: &[ResidualFactor<ArrayType, D::Value>],
) -> Result<Program<ArrayType, D::Tangent, LinearOperationOf<D>, Vec<D::Tangent>, Vec<D::Tangent>>, ProgramError>
where
    D: DifferentiationContext<Type = ArrayType>,
    LinearOperationOf<D>: ResidualizedOperation<D>,
{
    program.map_operations(|operation| {
        operation.try_map_factors(&mut |factor| match factor {
            ResidualFactor::Reference { index, .. } => factors.get(*index).cloned().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "condition branch pushforward references residual {index} but only {} residuals were captured",
                    factors.len(),
                ))
            }),
            ResidualFactor::Constant(value) => Ok(ResidualFactor::Constant(value.clone())),
        })
    })
}

/// Inlines a nested primal `program` into the linear `builder` as recomputed primal operations, mapping the
/// program's input atoms onto `input_atoms` and returning the builder atoms carrying the program outputs.
///
/// Each instruction is wrapped through [`SupportsLinearWhile::recompute_operation`] and each program constant is
/// materialized through a nullary residual injection whose closed factor is the constant lifted into the enclosing
/// context, since the linear program's value type generally differs from the primal constant type.
fn inline_recomputed_primal_program<E, O>(
    differentiable: &E,
    builder: &mut ProgramBuilder<<E as Domain>::Type, E::Tangent, LinearOperationOf<E>>,
    program: &Program<
        <E as Domain>::Type,
        <E as Domain>::Constant,
        O,
        Vec<<E as Domain>::Constant>,
        Vec<<E as Domain>::Constant>,
    >,
    input_atoms: &[AtomId],
) -> Result<Vec<AtomId>, ProgramError>
where
    E: DifferentiationContext,
    O: Clone + Operation<<E as Domain>::Type>,
    LinearOperationOf<E>:
        SupportsLinearWhile<<E as Domain>::Type, E::Tangent, ResidualFactor<<E as Domain>::Type, E::Value>, O>,
{
    check_count!("input", input_atoms, program.input_ids().len(), ProgramError);
    let mut atom_map: Vec<Option<AtomId>> = vec![None; program.atoms().len()];
    for (program_atom, builder_atom) in program.input_ids().iter().zip(input_atoms.iter()) {
        atom_map[program_atom.index()] = Some(*builder_atom);
    }
    for (atom_index, atom) in program.atoms().iter().enumerate() {
        if let Atom::Constant(constant) = atom {
            let factor = ResidualFactor::Constant(differentiable.lift(constant.clone())?);
            let outputs = builder.add_instruction(LinearOperationOf::<E>::residual_operation(factor), vec![])?;
            check_count!("output", outputs, 1, ProgramError);
            atom_map[atom_index] = Some(outputs[0]);
        }
    }
    let map_atom = |atom_map: &[Option<AtomId>], atom: AtomId| {
        atom_map.get(atom.index()).copied().flatten().ok_or(ProgramError::UnboundAtomId { id: atom })
    };
    for instruction in program.instructions() {
        let inputs = instruction
            .inputs()
            .iter()
            .map(|input| map_atom(atom_map.as_slice(), *input))
            .collect::<Result<Vec<_>, _>>()?;
        let operation = LinearOperationOf::<E>::recompute_operation(instruction.operation().clone());
        let outputs = builder.add_instruction(operation, inputs)?.to_vec();
        check_count!("output", outputs, instruction.outputs().len(), ProgramError);
        for (program_atom, builder_atom) in instruction.outputs().iter().zip(outputs.into_iter()) {
            atom_map[program_atom.index()] = Some(builder_atom);
        }
    }
    program.output_ids().iter().map(|output| map_atom(atom_map.as_slice(), *output)).collect()
}

/// Inlines a body pushforward `program` into the linear `builder`, mapping the program's tangent input atoms onto
/// `input_atoms` and rewriting loop-varying [`ResidualFactor::Reference`] factors into operand form against
/// `residual_atoms` through [`SupportsLinearWhile::defactorize`]. Returns the builder atoms carrying the program
/// outputs.
fn inline_defactorized_pushforward_program<E>(
    builder: &mut ProgramBuilder<<E as Domain>::Type, E::Tangent, LinearOperationOf<E>>,
    program: &Program<<E as Domain>::Type, E::Tangent, LinearOperationOf<E>, Vec<E::Tangent>, Vec<E::Tangent>>,
    input_atoms: &[AtomId],
    residual_atoms: &[AtomId],
) -> Result<Vec<AtomId>, ProgramError>
where
    E: DifferentiationContext,
    LinearOperationOf<E>: SupportsLinearWhile<
            <E as Domain>::Type,
            E::Tangent,
            ResidualFactor<<E as Domain>::Type, E::Value>,
            <E as Domain>::Operation,
        >,
{
    check_count!("input", input_atoms, program.input_ids().len(), ProgramError);
    let mut atom_map: Vec<Option<AtomId>> = vec![None; program.atoms().len()];
    for (program_atom, builder_atom) in program.input_ids().iter().zip(input_atoms.iter()) {
        atom_map[program_atom.index()] = Some(*builder_atom);
    }
    for (atom_index, atom) in program.atoms().iter().enumerate() {
        if let Atom::Constant(constant) = atom {
            atom_map[atom_index] = Some(builder.add_constant(constant.clone()));
        }
    }
    let map_atom = |atom_map: &[Option<AtomId>], atom: AtomId| {
        atom_map.get(atom.index()).copied().flatten().ok_or(ProgramError::UnboundAtomId { id: atom })
    };
    for instruction in program.instructions() {
        let inputs = instruction
            .inputs()
            .iter()
            .map(|input| map_atom(atom_map.as_slice(), *input))
            .collect::<Result<Vec<_>, _>>()?;
        match instruction.operation().defactorize(residual_atoms, inputs)? {
            DefactorizedOperation::Operation { operation, inputs } => {
                let outputs = builder.add_instruction(operation, inputs)?.to_vec();
                check_count!("output", outputs, instruction.outputs().len(), ProgramError);
                for (program_atom, builder_atom) in instruction.outputs().iter().zip(outputs.into_iter()) {
                    atom_map[program_atom.index()] = Some(builder_atom);
                }
            }
            DefactorizedOperation::Forward { atom } => {
                check_count!("output", instruction.outputs(), 1, ProgramError);
                atom_map[instruction.outputs()[0].index()] = Some(atom);
            }
        }
    }
    program.output_ids().iter().map(|output| map_atom(atom_map.as_slice(), *output)).collect()
}

/// Builds the extended condition and fused body programs of the staged doubled-state while loop from one nested
/// symbolic linearization of the loop body (see the [`WhileOperation`] JVP rule below).
///
/// Both programs consume the doubled state `[primal_state..., tangent_state...]`. The fused body inlines the
/// residual-extended primal body program as recomputed primal operations over the primal half, then inlines the body
/// pushforward over the tangent half with its loop-varying residual references defactorized against the recomputed
/// residual atoms, and outputs `[next_primal_state..., next_tangent_state...]`. The extended condition recomputes
/// the original loop predicate from the primal half and ignores the tangent half.
fn build_fused_while_programs<E, O>(
    differentiable: &E,
    condition: &Program<
        <E as Domain>::Type,
        <E as Domain>::Constant,
        O,
        Vec<<E as Domain>::Constant>,
        Vec<<E as Domain>::Constant>,
    >,
    linearization: &NestedLinearization<E, O>,
) -> Result<
    (
        Program<<E as Domain>::Type, E::Tangent, LinearOperationOf<E>, Vec<E::Tangent>, Vec<E::Tangent>>,
        Program<<E as Domain>::Type, E::Tangent, LinearOperationOf<E>, Vec<E::Tangent>, Vec<E::Tangent>>,
    ),
    ProgramError,
>
where
    E: DifferentiationContext + Domain<Operation = O>,
    O: Clone + Operation<<E as Domain>::Type>,
    LinearOperationOf<E>:
        SupportsLinearWhile<<E as Domain>::Type, E::Tangent, ResidualFactor<<E as Domain>::Type, E::Value>, O>,
{
    let state_types = condition.input_types();
    let state_count = state_types.len();
    let residual_count = linearization.residual_types.len();

    let mut body_builder = ProgramBuilder::<<E as Domain>::Type, E::Tangent, LinearOperationOf<E>>::new();
    let primal_inputs =
        state_types.iter().map(|state_type| body_builder.add_input(state_type.clone())).collect::<Vec<_>>();
    let tangent_inputs =
        state_types.iter().map(|state_type| body_builder.add_input(state_type.clone())).collect::<Vec<_>>();
    let primal_outputs = inline_recomputed_primal_program(
        differentiable,
        &mut body_builder,
        &linearization.primal_program,
        primal_inputs.as_slice(),
    )?;
    check_count!("output", primal_outputs, state_count + residual_count, ProgramError);
    let tangent_outputs = inline_defactorized_pushforward_program::<E>(
        &mut body_builder,
        &linearization.pushforward_program,
        tangent_inputs.as_slice(),
        &primal_outputs[state_count..],
    )?;
    check_count!("output", tangent_outputs, state_count, ProgramError);
    let mut body_outputs = primal_outputs[..state_count].to_vec();
    body_outputs.extend(tangent_outputs);
    let fused_body =
        body_builder.build(body_outputs, vec![Placeholder; 2 * state_count], vec![Placeholder; 2 * state_count])?;

    let mut condition_builder = ProgramBuilder::<<E as Domain>::Type, E::Tangent, LinearOperationOf<E>>::new();
    let condition_primal_inputs = state_types
        .iter()
        .map(|state_type| condition_builder.add_input(state_type.clone()))
        .collect::<Vec<_>>();
    for state_type in &state_types {
        condition_builder.add_input(state_type.clone());
    }
    let condition_outputs = inline_recomputed_primal_program(
        differentiable,
        &mut condition_builder,
        condition,
        condition_primal_inputs.as_slice(),
    )?;
    check_count!("output", condition_outputs, 1, ProgramError);
    let extended_condition =
        condition_builder.build(condition_outputs, vec![Placeholder; 2 * state_count], vec![Placeholder])?;
    Ok((extended_condition, fused_body))
}

/// JVP rule for [`ConditionOperation`] with full JAX
/// [`cond`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html) parity: the rule never concretizes the
/// predicate, so forward-mode differentiation of a runtime-predicate condition composes under abstract tracing
/// (tracer-valued differentiation contexts) by staging condition structure instead.
///
/// The rule mirrors JAX's `cond` JVP plus partial evaluation:
///
///   1. Both branches are linearized *symbolically* at the branch input types via [`linearize_nested_program`] — no
///      primal operand values are involved, so no branch computation is evaluated here.
///   2. The residual-extended primal branches are joined to a common output signature
///      `[outputs..., true_residuals..., false_residuals...]`, with each branch emitting typed zeros in the other
///      branch's residual slots (JAX's `_join_cond_outputs` analog).
///   3. One primal [`ConditionOperation`] over the joined branches is bound in the primal domain. Eager domains
///      *interpret* that condition and therefore still evaluate only the branch selected by the runtime predicate;
///      staging domains record it, so the primal trace gains one `condition` operation with residual-extended
///      branches.
///   4. The primal condition's residual outputs are registered in the active linearization residual environment, and
///      each branch pushforward's local residual references are remapped onto the resulting factors.
///   5. One linear condition ([`SupportsLinearCondition`]) is staged over the operand tangents, capturing the
///      predicate primal as a residual factor (exactly like [`SelectOperation`](
///      crate::operations::control_flow::SelectOperation)'s rule captures its condition) and carrying both
///      residualized branch pushforwards. Replaying the resulting pushforward with fresh tangents instantiates the
///      captured predicate factor and runs the branch pushforward selected at the original primal point.
///
/// The predicate is the first operand and its tangent is ignored (Boolean predicates have no tangent space).
/// Reverse mode composes through the total linear-condition transpose rule, which transposes both branch programs
/// and carries the predicate factor verbatim.
impl<V, D, O> DifferentiableOperation<D> for ConditionOperation<V, O, ArrayType>
where
    V: Value<ArrayType>,
    D: DifferentiationContext<Type = ArrayType, Constant = V> + Domain<Operation = O>,
    O: Clone
        + Operation<ArrayType>
        + SupportsZero<ArrayType>
        + From<ConditionOperation<V, O, ArrayType>>
        + SupportsNestedLinearization<D>,
    LinearOperationOf<D>:
        ResidualizedOperation<D> + SupportsLinearCondition<ArrayType, D::Tangent, ResidualFactor<ArrayType, D::Value>>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, self.input_types().len(), ProgramError);
        let predicate = &inputs[0];
        let operands = &inputs[1..];

        // Linearize both branches symbolically at the branch input types and join their residual signatures.
        let NestedLinearization {
            primal_program: true_primal_program,
            pushforward_program: true_pushforward_program,
            residual_types: true_residual_types,
        } = O::linearize_nested_program(context.differentiable(), self.true_branch())?;
        let NestedLinearization {
            primal_program: false_primal_program,
            pushforward_program: false_pushforward_program,
            residual_types: false_residual_types,
        } = O::linearize_nested_program(context.differentiable(), self.false_branch())?;
        let output_count = self.true_branch().output_ids().len();
        let joined_true_branch =
            join_condition_branch_outputs(true_primal_program, output_count, false_residual_types.as_slice(), true);
        let joined_false_branch =
            join_condition_branch_outputs(false_primal_program, output_count, true_residual_types.as_slice(), false);

        // Bind one primal condition over the joined branches. `ConditionOperation::new` validates that the joined
        // branches agree on their input and output signatures; eager domains interpret the staged condition and so
        // still evaluate only the branch selected by the runtime predicate.
        let primal_condition =
            ConditionOperation::new(self.predicate_type().clone(), joined_true_branch, joined_false_branch)?;
        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let mut bound_outputs = context.bind_primal(O::from(primal_condition), primal_inputs.as_slice())?;
        check_count!(
            "output",
            bound_outputs,
            output_count + true_residual_types.len() + false_residual_types.len(),
            ProgramError,
        );

        // Register the primal condition's residual outputs in the enclosing residual environment and remap each
        // branch pushforward's local residual references onto the resulting factors.
        let residual_values = bound_outputs.split_off(output_count);
        let primal_outputs = bound_outputs;
        let residual_factors = residual_values.into_iter().map(|value| context.factor(value)).collect::<Vec<_>>();
        let (true_residual_factors, false_residual_factors) = residual_factors.split_at(true_residual_types.len());
        let true_branch = remap_branch_residual_factors::<D>(&true_pushforward_program, true_residual_factors)?;
        let false_branch = remap_branch_residual_factors::<D>(&false_pushforward_program, false_residual_factors)?;

        // Stage one linear condition over the operand tangents, capturing the predicate primal as a residual factor.
        let predicate_factor = predicate.factor(context);
        let tangent_operands = operands
            .iter()
            .map(|input| context.materialize_tangent(input.tangent().clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let tangent_outputs = context.stage_operation(
            LinearOperationOf::<D>::linear_condition_operation(predicate_factor, true_branch, false_branch),
            tangent_operands.as_slice(),
        )?;
        check_count!("output", tangent_outputs, output_count, ProgramError);
        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| JvpTracer::from_value(primal, tangent))
            .collect())
    }
}

impl<V: Value<ArrayType>, O> TransposableOperation<ArrayType, V, O> for WhileOperation<V, O, ArrayType>
where
    O: Operation<ArrayType>,
{
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, ArrayType, V, O>,
        _input_types: &[&ArrayType],
        _output_cotangents: &[Cotangent<'transpose, ArrayType, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, O>>, ProgramError> {
        Err(ProgramError::UnsupportedOperation {
            message: "while does not support transposition (reverse-mode differentiation through while loops is not \
                      supported)"
                .to_string(),
        })
    }
}

/// JVP rule for [`WhileOperation`] with full JAX
/// [`while_loop`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html) parity: the rule never
/// concretizes the loop predicate and never unrolls iterations, so forward-mode differentiation of a while loop
/// composes under abstract tracing (tracer-valued differentiation contexts) by staging loop structure instead.
///
/// The rule mirrors JAX's `while_loop` JVP:
///
///   1. The body is linearized *symbolically* once at the loop state types via [`linearize_nested_program`](
///      crate::tracing_v2::linearize_nested_program) — no primal state values are involved and no iteration runs
///      here.
///   2. The primal [`WhileOperation`] is bound *unchanged* (original condition and body) in the primal domain. Eager
///      domains interpret it and drive the loop on concrete state; staging domains record one `while` operation.
///   3. One linear while loop ([`SupportsLinearWhile`]) is staged over the doubled state
///      `[primal_state..., tangent_state...]`. Its fused body interleaves primal recomputation with tangent
///      propagation: the residual-extended primal body program is inlined as recomputed primal operations over the
///      primal half, and the body pushforward is inlined over the tangent half with its loop-varying
///      [`ResidualFactor::Reference`] factors *defactorized* into operand form against the recomputed residual
///      atoms (a scale by a referenced residual becomes an elementwise product, the captured dot maps become
///      operand-form dots). Its condition recomputes the original loop predicate from the primal half and ignores
///      the tangent half.
///   4. The loop-entry primal state enters the linear program through nullary residual injections
///      ([`SupportsLinearWhile::residual_operation`]), one per state element. Replaying the pushforward at the same
///      primal point therefore genuinely re-runs the loop — the trip count comes from the captured primal point —
///      and fresh tangents propagate through the same iterations.
///
/// Eager domains consequently also stage-and-replay: the staged linear while is interpreted when the tangent
/// program runs instead of being unrolled into per-iteration linear instructions at rule time, and the rule's
/// tangent outputs are the tangent half of the staged loop's outputs (the primal half is dead in the linear
/// program, since the rule's primal outputs come from the bound primal while — the same redundancy JAX accepts).
///
/// Reverse-mode differentiation through a while loop keeps erroring in [`WhileOperation`]'s transpose rule, exactly
/// like JAX: the fused linear loop recomputes primal state *forward* through the iterations, so transposing it would
/// have to run that recomputation backwards, which a while loop cannot express.
impl<V, D, O> DifferentiableOperation<D> for WhileOperation<V, O, ArrayType>
where
    V: Value<ArrayType>,
    D: DifferentiationContext<Type = ArrayType, Constant = V> + Domain<Operation = O>,
    O: Clone + Operation<ArrayType> + From<WhileOperation<V, O, ArrayType>> + SupportsNestedLinearization<D>,
    LinearOperationOf<D>:
        ResidualizedOperation<D> + SupportsLinearWhile<ArrayType, D::Tangent, ResidualFactor<ArrayType, D::Value>, O>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        let state_count = self.state_types().len();
        check_count!("input", inputs, state_count, ProgramError);

        // Linearize the body symbolically once at the loop state types and build the doubled-state programs.
        let linearization = O::linearize_nested_program(context.differentiable(), self.body())?;
        let (extended_condition, fused_body) =
            build_fused_while_programs(context.differentiable(), self.condition(), &linearization)?;

        // Bind the primal while unchanged: eager domains drive the loop on concrete state here, while staging
        // domains record one `while` operation with the original condition and body.
        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal_outputs = context.bind_primal(O::from(self.clone()), primal_inputs.as_slice())?;
        check_count!("output", primal_outputs, state_count, ProgramError);

        // Inject the loop-entry primal state into the linear program through nullary residual injections and stage
        // one linear while over the doubled state `[primal_state..., tangent_state...]`.
        let mut linear_inputs = Vec::with_capacity(2 * state_count);
        for input in inputs {
            let factor = input.factor(context);
            let mut outputs = context.stage_operation(
                LinearOperationOf::<D>::residual_operation(factor),
                &[] as &[Tracer<TangentContext<'jvp, D>>],
            )?;
            check_count!("output", outputs, 1, ProgramError);
            linear_inputs.push(outputs.remove(0));
        }
        for input in inputs {
            linear_inputs.push(context.materialize_tangent(input.tangent().clone())?);
        }
        let linear_while = LinearOperationOf::<D>::linear_while_operation(extended_condition, fused_body)?;
        let linear_outputs = context.stage_operation(linear_while, linear_inputs.as_slice())?;
        check_count!("output", linear_outputs, 2 * state_count, ProgramError);

        // The rule's tangent outputs are the tangent half of the staged loop's outputs.
        Ok(primal_outputs
            .into_iter()
            .zip(linear_outputs.into_iter().skip(state_count))
            .map(|(primal, tangent)| JvpTracer::from_value(primal, tangent))
            .collect())
    }
}

/// Batches a condition over `true_branch` and `false_branch` by reading the predicate from the first input.
///
/// A lane-uniform predicate is concretized via [`BooleanLike::boolean`] and selects one branch to interpret over the
/// remaining operand inputs. A lane-varying predicate interprets both branches over the operand inputs and merges
/// their outputs per lane via [`Select`](crate::operations::control_flow::Select).
pub(crate) fn batch_condition_with_interpreter<VOperation, V, O, F>(
    true_branch: &Program<ArrayType, VOperation, O, Vec<VOperation>, Vec<VOperation>>,
    false_branch: &Program<ArrayType, VOperation, O, Vec<VOperation>, Vec<VOperation>>,
    inputs: &[ArrayBatch<V>],
    mut interpret_program: F,
) -> Result<Vec<ArrayBatch<V>>, ProgramError>
where
    VOperation: Value<ArrayType>,
    V: Value<ArrayType> + BooleanLike + crate::operations::control_flow::Select<Condition = V>,
    O: Operation<ArrayType>,
    F: FnMut(
        &Program<ArrayType, VOperation, O, Vec<VOperation>, Vec<VOperation>>,
        Vec<ArrayBatch<V>>,
    ) -> Result<Vec<ArrayBatch<V>>, ProgramError>,
{
    let Some((predicate_batch, operand_inputs)) = inputs.split_first() else {
        return Err(BatchingError::UnsupportedOperation {
            message: "cannot batch a condition operation with no predicate input".to_string(),
        }
        .into());
    };
    match predicate_batch.batch_axis() {
        None => {
            let predicate = predicate_batch.value().boolean()?;
            let branch = if predicate { true_branch } else { false_branch };
            interpret_program(branch, operand_inputs.to_vec())
        }
        Some(predicate_axis) => {
            let true_outputs = interpret_program(true_branch, operand_inputs.to_vec())?;
            let false_outputs = interpret_program(false_branch, operand_inputs.to_vec())?;
            check_count!("output", true_outputs, false_outputs.len(), ProgramError);
            true_outputs
                .into_iter()
                .zip(false_outputs)
                .map(|(true_output, false_output)| -> Result<ArrayBatch<V>, ProgramError> {
                    let output_axis = match (true_output.batch_axis(), false_output.batch_axis()) {
                        (Some(left), Some(right)) if left != right => {
                            return Err(BatchingError::MisalignedBatchAxes {
                                message: format!(
                                    "condition branches produced lane-varying outputs at mismatched axes \
                                    ({left} vs {right})",
                                ),
                            }
                            .into());
                        }
                        (Some(axis), _) | (_, Some(axis)) => axis,
                        (None, None) => predicate_axis,
                    };
                    let selected = V::select(
                        predicate_batch.value().clone(),
                        true_output.value().clone(),
                        false_output.value().clone(),
                    )?;
                    let output_type = selected.r#type().into_owned();
                    ArrayBatch::new(output_type, selected, Some(output_axis))
                })
                .collect()
        }
    }
}

impl<V, O> BatchableOperation<V, ()> for ConditionOperation<V, O, ArrayType>
where
    V: Value<ArrayType> + BooleanLike + crate::operations::control_flow::Select<Condition = V>,
    O: BatchableOperation<V, ()>,
{
    fn batch(&self, _context: &(), inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        batch_condition_with_interpreter(self.true_branch(), self.false_branch(), inputs, |program, program_inputs| {
            program.interpret_with(
                program_inputs,
                |_, constant| Ok(ArrayBatch::unbatched(constant.clone())),
                |instruction, instruction_inputs| instruction.operation().batch(&(), instruction_inputs),
            )
        })
    }
}

impl<C, O> BatchableOperation<Tracer<C>, BatchingContext<C>> for ConditionOperation<C::Constant, O, ArrayType>
where
    C: StagingContext<Type = ArrayType>,
    C::Constant: Value<ArrayType> + BooleanLike,
    Tracer<C>: crate::operations::control_flow::Select<Condition = Tracer<C>>,
    O: BatchableOperation<Tracer<C>, BatchingContext<C>>,
{
    fn batch(
        &self,
        context: &BatchingContext<C>,
        inputs: &[ArrayBatch<Tracer<C>>],
    ) -> Result<Vec<ArrayBatch<Tracer<C>>>, ProgramError> {
        batch_condition_with_interpreter(self.true_branch(), self.false_branch(), inputs, |program, program_inputs| {
            context.interpret_program(program, program_inputs)
        })
    }
}

/// `Tangent`-specific batching for [`ConditionOperation`]. The generic impl above doesn't apply
/// because [`Tangent`] does not implement [`BooleanLike`] or
/// [`Select`](crate::operations::control_flow::Select) (those would require materializing
/// the inner symbolic-zero tangent at the tangent layer). We materialize each input's
/// [`Tangent::Zero`] to the matching `V::zero(t)` via the default [`BatchableOperation`] rule,
/// dispatch to the V-level [`ConditionOperation`] batching rule (which itself handles
/// lane-uniform vs lane-varying predicates by selecting per lane), and re-wrap each output as
/// `Tangent::Value`. This is the same materialize-then-dispatch pattern used by
/// [`LinearArrayOperation`](crate::tracing_v2::operations::primitive::LinearArrayOperation)'s tangent batching rule.
impl<V, O> BatchableOperation<Tangent<ArrayType, V>, ()> for ConditionOperation<V, O, ArrayType>
where
    Self: BatchableOperation<V>,
    V: Value<ArrayType>
        + crate::operations::constants::Zero<ArrayType>
        + BooleanLike
        + crate::operations::control_flow::Select<Condition = V>,
    O: BatchableOperation<V>,
{
    fn batch(
        &self,
        _context: &(),
        inputs: &[ArrayBatch<Tangent<ArrayType, V>>],
    ) -> Result<Vec<ArrayBatch<Tangent<ArrayType, V>>>, ProgramError> {
        let materialized: Vec<ArrayBatch<V>> = inputs
            .iter()
            .map(|input| -> Result<ArrayBatch<V>, ProgramError> {
                let value = match input.value() {
                    Tangent::Zero(t) => V::zero(t)?,
                    Tangent::Value(v) => v.clone(),
                };
                ArrayBatch::new(input.r#type().into_owned(), value, input.batch_axis())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let v_outputs = <Self as BatchableOperation<V>>::batch(self, &(), materialized.as_slice())?;
        v_outputs
            .into_iter()
            .map(|out| -> Result<ArrayBatch<Tangent<ArrayType, V>>, ProgramError> {
                let output_type = out.r#type().into_owned();
                let output_axis = out.batch_axis();
                ArrayBatch::new(output_type, Tangent::Value(out.into_value()), output_axis)
            })
            .collect()
    }
}

fn batch_while_with_interpreter<VOperation, V, O, F>(
    while_operation: &WhileOperation<VOperation, O, ArrayType>,
    inputs: &[ArrayBatch<V>],
    mut interpret_program: F,
) -> Result<Vec<ArrayBatch<V>>, ProgramError>
where
    VOperation: Value<ArrayType>,
    V: Value<ArrayType>
        + BooleanLike
        + crate::tracing_v2::operations::reduce::Reduce
        + std::ops::BitAnd<Output = V>
        + crate::operations::control_flow::Select<Condition = V>
        + crate::operations::manipulation::Broadcast<Output = V>,
    O: Operation<ArrayType>,
    F: FnMut(
        &Program<ArrayType, VOperation, O, Vec<VOperation>, Vec<VOperation>>,
        Vec<ArrayBatch<V>>,
    ) -> Result<Vec<ArrayBatch<V>>, ProgramError>,
{
    // Run the condition once on the initial state to discover whether the predicate is
    // lane-uniform or lane-varying. The two cases diverge from here: lane-uniform takes the
    // original eager-loop path; lane-varying threads a per-lane mask through every iteration
    // and runs the body until no lane is still active.
    let mut state = inputs.to_vec();
    let initial_condition_outputs = interpret_program(while_operation.condition(), state.clone())?;
    check_count!("output", initial_condition_outputs, 1, ProgramError);
    let initial_predicate = initial_condition_outputs.into_iter().next().unwrap();
    if initial_predicate.batch_axis().is_none() {
        if !initial_predicate.value().boolean()? {
            return Ok(state);
        }
        state = interpret_program(while_operation.body(), state)?;
        return run_lane_uniform_while_loop::<VOperation, V, O, F>(
            while_operation.condition(),
            while_operation.body(),
            state,
            &mut interpret_program,
        );
    }
    // Lane-varying path: the predicate carries a batch axis. Track a per-lane mask, mask
    // state updates per lane via `Select`, and exit once `any(mask)` is false.
    run_lane_varying_while_loop::<VOperation, V, O, F>(
        while_operation.condition(),
        while_operation.body(),
        state,
        initial_predicate,
        &mut interpret_program,
    )
}

impl<V, O> BatchableOperation<V, ()> for WhileOperation<V, O, ArrayType>
where
    V: Value<ArrayType>
        + BooleanLike
        + crate::tracing_v2::operations::reduce::Reduce
        + std::ops::BitAnd<Output = V>
        + crate::operations::control_flow::Select<Condition = V>
        + crate::operations::manipulation::Broadcast<Output = V>,
    O: BatchableOperation<V, ()>,
{
    fn batch(&self, _context: &(), inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        batch_while_with_interpreter(self, inputs, |program, program_inputs| {
            program.interpret_with(
                program_inputs,
                |_, constant| Ok(ArrayBatch::unbatched(constant.clone())),
                |instruction, instruction_inputs| instruction.operation().batch(&(), instruction_inputs),
            )
        })
    }
}

impl<C, O> BatchableOperation<Tracer<C>, BatchingContext<C>> for WhileOperation<C::Constant, O, ArrayType>
where
    C: StagingContext<Type = ArrayType>,
    C::Constant: Value<ArrayType> + BooleanLike,
    Tracer<C>: crate::tracing_v2::operations::reduce::Reduce
        + std::ops::BitAnd<Output = Tracer<C>>
        + crate::operations::control_flow::Select<Condition = Tracer<C>>
        + crate::operations::manipulation::Broadcast<Output = Tracer<C>>,
    O: BatchableOperation<Tracer<C>, BatchingContext<C>>,
{
    fn batch(
        &self,
        context: &BatchingContext<C>,
        inputs: &[ArrayBatch<Tracer<C>>],
    ) -> Result<Vec<ArrayBatch<Tracer<C>>>, ProgramError> {
        batch_while_with_interpreter(self, inputs, |program, program_inputs| {
            context.interpret_program(program, program_inputs)
        })
    }
}

/// Eager loop that drives a [`WhileOperation`] whose condition program produces a lane-uniform
/// scalar Boolean predicate. Each iteration runs the body when the predicate is `true` and exits
/// when it becomes `false`. This is the original simple loop preserved for the lane-uniform case.
fn run_lane_uniform_while_loop<VOperation, V, O, F>(
    condition: &Program<ArrayType, VOperation, O, Vec<VOperation>, Vec<VOperation>>,
    body: &Program<ArrayType, VOperation, O, Vec<VOperation>, Vec<VOperation>>,
    mut state: Vec<ArrayBatch<V>>,
    interpret_program: &mut F,
) -> Result<Vec<ArrayBatch<V>>, ProgramError>
where
    VOperation: Value<ArrayType>,
    V: Value<ArrayType> + BooleanLike,
    F: FnMut(
        &Program<ArrayType, VOperation, O, Vec<VOperation>, Vec<VOperation>>,
        Vec<ArrayBatch<V>>,
    ) -> Result<Vec<ArrayBatch<V>>, ProgramError>,
{
    loop {
        let condition_outputs = interpret_program(condition, state.clone())?;
        check_count!("output", condition_outputs, 1, ProgramError);
        let predicate_batch = &condition_outputs[0];
        if predicate_batch.batch_axis().is_some() {
            return Err(BatchingError::UnsupportedOperation {
                message: "while loop condition produced a lane-varying predicate mid-iteration after starting \
                    lane-uniform; this is not yet supported"
                    .to_string(),
            }
            .into());
        }
        if !predicate_batch.value().boolean()? {
            return Ok(state);
        }
        state = interpret_program(body, state)?;
    }
}

/// Eager loop that drives a [`WhileOperation`] whose condition program produces a lane-varying
/// predicate (one Boolean per mapped lane). Each iteration:
///
///   1. Updates the per-lane active mask by AND-ing with the current per-lane predicate.
///   2. Stops when no lane is still active (`any(mask) == false`).
///   3. Runs the body to produce candidate updated state.
///   4. Masks state updates per lane via [`Select`](crate::operations::control_flow::Select)
///      so inactive lanes retain their prior state forever.
///
/// This implementation requires a value type that supports [`Reduce`](
/// crate::tracing_v2::operations::reduce::Reduce) (for the `any` aggregation),
/// [`BitAnd`](std::ops::BitAnd) (for `mask & current`),
/// [`Select`](crate::operations::control_flow::Select), and
/// [`Broadcast`](crate::operations::manipulation::Broadcast) — the same
/// primitives every staged value type already needs for the rest of the operation enum.
fn run_lane_varying_while_loop<VOperation, V, O, F>(
    condition: &Program<ArrayType, VOperation, O, Vec<VOperation>, Vec<VOperation>>,
    body: &Program<ArrayType, VOperation, O, Vec<VOperation>, Vec<VOperation>>,
    mut state: Vec<ArrayBatch<V>>,
    initial_predicate: ArrayBatch<V>,
    interpret_program: &mut F,
) -> Result<Vec<ArrayBatch<V>>, ProgramError>
where
    VOperation: Value<ArrayType>,
    V: Value<ArrayType>
        + BooleanLike
        + crate::tracing_v2::operations::reduce::Reduce
        + std::ops::BitAnd<Output = V>
        + crate::operations::control_flow::Select<Condition = V>
        + crate::operations::manipulation::Broadcast<Output = V>,
    F: FnMut(
        &Program<ArrayType, VOperation, O, Vec<VOperation>, Vec<VOperation>>,
        Vec<ArrayBatch<V>>,
    ) -> Result<Vec<ArrayBatch<V>>, ProgramError>,
{
    let predicate_axis = initial_predicate.batch_axis().ok_or_else(|| BatchingError::MisalignedBatchAxes {
        message: "lane-varying while batching requires a batched initial predicate".to_string(),
    })?;
    let mut active_mask = initial_predicate;
    loop {
        if !lane_varying_any_active(&active_mask, predicate_axis)? {
            return Ok(state);
        }
        let body_outputs = interpret_program(body, state.clone())?;
        check_count!("output", body_outputs, state.len(), ProgramError);
        state = state
            .into_iter()
            .zip(body_outputs)
            .map(|(prior, candidate)| mask_state_element(&active_mask, predicate_axis, candidate, prior))
            .collect::<Result<Vec<_>, _>>()?;
        let next_condition_outputs = interpret_program(condition, state.clone())?;
        check_count!("output", next_condition_outputs, 1, ProgramError);
        let next_predicate = next_condition_outputs.into_iter().next().unwrap();
        if next_predicate.batch_axis().is_none() {
            return Err(BatchingError::UnsupportedOperation {
                message: "while loop predicate became lane-uniform mid-iteration after starting lane-varying; \
                    this is not yet supported"
                    .to_string(),
            }
            .into());
        }
        active_mask = combine_active_mask(active_mask, next_predicate)?;
    }
}

/// Returns `true` when at least one lane of `mask` is active by reducing along `predicate_axis`
/// and extracting the resulting scalar Boolean.
fn lane_varying_any_active<V: Value<ArrayType> + BooleanLike + crate::tracing_v2::operations::reduce::Reduce>(
    mask: &ArrayBatch<V>,
    predicate_axis: usize,
) -> Result<bool, ProgramError> {
    let reduced = mask
        .value()
        .clone()
        .reduce(&[predicate_axis], crate::tracing_v2::operations::reduce::ReductionKind::Any);
    reduced.boolean()
}

/// Combines the prior `active_mask` with the current `next_predicate` via logical AND. Both must
/// be batched on the same physical axis; the result inherits that axis.
fn combine_active_mask<V: Value<ArrayType> + std::ops::BitAnd<Output = V>>(
    active_mask: ArrayBatch<V>,
    next_predicate: ArrayBatch<V>,
) -> Result<ArrayBatch<V>, ProgramError> {
    let axis = active_mask.batch_axis();
    let combined = active_mask.into_value() & next_predicate.into_value();
    let combined_type = combined.r#type().into_owned();
    ArrayBatch::new(combined_type, combined, axis)
}

/// Builds the masked update for one state element by broadcasting the per-lane mask to the
/// element's physical shape and selecting between the candidate body output and the prior state
/// per lane.
fn mask_state_element<V>(
    active_mask: &ArrayBatch<V>,
    predicate_axis: usize,
    candidate: ArrayBatch<V>,
    prior: ArrayBatch<V>,
) -> Result<ArrayBatch<V>, ProgramError>
where
    V: Value<ArrayType>
        + crate::operations::control_flow::Select<Condition = V>
        + crate::operations::manipulation::Broadcast<Output = V>,
{
    let candidate_axis =
        candidate.batch_axis().or(prior.batch_axis()).ok_or_else(|| BatchingError::UnsupportedOperation {
            message: "lane-varying while body produced a lane-uniform state element; this is not yet supported"
                .to_string(),
        })?;
    let candidate_type = candidate.r#type().into_owned();
    let mask_type = active_mask.r#type().into_owned();
    let mask_output_axes: Vec<usize> = (0..mask_type.rank())
        .map(|i| {
            if i == predicate_axis {
                candidate_axis
            } else if i < predicate_axis {
                // mask axes left of the predicate axis carry over to the candidate left of `candidate_axis`.
                i
            } else {
                // mask axes right of the predicate axis carry over to the candidate right of `candidate_axis`.
                i + (candidate_type.rank() - mask_type.rank())
            }
        })
        .collect();
    let mask_output_type = ArrayType::new(mask_type.data_type(), candidate_type.shape().clone());
    let broadcasted_mask = active_mask.value().clone().broadcast(mask_output_type, mask_output_axes.as_slice())?;
    let selected = V::select(broadcasted_mask, candidate.into_value(), prior.into_value())?;
    let selected_type = selected.r#type().into_owned();
    ArrayBatch::new(selected_type, selected, Some(candidate_axis))
}

/// `Tangent`-specific batching for [`WhileOperation`]. Like the `ConditionOperation` Tangent impl
/// above, this exists because [`Tangent`] does not implement [`BooleanLike`]. The staged linear while emitted by
/// the [`WhileOperation`] JVP rule batches through
/// [`LinearArrayOperation`](crate::tracing_v2::operations::primitive::LinearArrayOperation)'s tangent batching rule
/// (materialize-then-dispatch onto the value-level loop), not through this impl, so this path is only reachable
/// when a caller batches a tangent-valued `While` over an ordinary operation enum directly; it returns
/// [`BatchingError::UnsupportedOperation`] in that case.
impl<V, O> BatchableOperation<Tangent<ArrayType, V>, ()> for WhileOperation<V, O, ArrayType>
where
    V: Value<ArrayType> + BooleanLike,
    O: BatchableOperation<Tangent<ArrayType, V>, ()>,
{
    fn batch(
        &self,
        _context: &(),
        _inputs: &[ArrayBatch<Tangent<ArrayType, V>>],
    ) -> Result<Vec<ArrayBatch<Tangent<ArrayType, V>>>, ProgramError> {
        Err(BatchingError::UnsupportedOperation {
            message: "missing batching rule for while over tangent runtime values".to_string(),
        }
        .into())
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use crate::macros::check_types;
    use std::cell::RefCell;
    use std::fmt::Display;
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_macros::Parameter;

    use crate::contexts::Context;
    use crate::domains::Domain;
    use crate::operations::InterpretableOperation;
    use crate::operations::arithmetic::{
        ADD_OPERATION_NAME, SUB_OPERATION_NAME, Scale, SupportsAdd, SupportsNeg, SupportsScale,
    };
    use crate::operations::constants::{One, OneLike, SupportsZero, Zero, ZeroLike};
    use crate::parameters::{Parameter, Placeholder};
    use crate::programs::{Program, ProgramBuilder, Value};
    use crate::tracing_v2::{ArrayOperation, FactorParameterizedOperation};
    use crate::types::DataType;
    use crate::types::TypeError;

    use super::*;

    #[derive(Clone, Debug, Parameter, PartialEq)]
    enum TestValue {
        Bool(bool),
        Number(f64),
    }

    impl Display for TestValue {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Bool(value) => Display::fmt(value, formatter),
                Self::Number(value) => Display::fmt(value, formatter),
            }
        }
    }

    impl Typed<ArrayType> for TestValue {
        fn r#type(&self) -> Cow<'_, ArrayType> {
            match self {
                Self::Bool(_) => Cow::Owned(ArrayType::scalar(DataType::Boolean)),
                Self::Number(_) => Cow::Owned(ArrayType::scalar(DataType::F64)),
            }
        }
    }

    impl Value<ArrayType> for TestValue {}

    impl ZeroLike for TestValue {
        fn zero_like(&self) -> Self {
            match self {
                Self::Bool(_) => Self::Bool(false),
                Self::Number(_) => Self::Number(0.0),
            }
        }
    }

    impl OneLike for TestValue {
        fn one_like(&self) -> Self {
            match self {
                Self::Bool(_) => Self::Bool(true),
                Self::Number(_) => Self::Number(1.0),
            }
        }
    }

    impl Zero<ArrayType> for TestValue {
        fn zero(value_type: &ArrayType) -> Result<Self, ProgramError> {
            match value_type.data_type() {
                DataType::Boolean => Ok(Self::Bool(false)),
                DataType::F64 => Ok(Self::Number(0.0)),
                _ => Err(crate::types::TypeError {
                    message: format!("test value cannot synthesize zero for {value_type}"),
                }
                .into()),
            }
        }
    }

    impl One<ArrayType> for TestValue {
        fn one(value_type: &ArrayType) -> Result<Self, ProgramError> {
            match value_type.data_type() {
                DataType::Boolean => Ok(Self::Bool(true)),
                DataType::F64 => Ok(Self::Number(1.0)),
                _ => Err(crate::types::TypeError {
                    message: format!("test value cannot synthesize one for {value_type}"),
                }
                .into()),
            }
        }
    }

    impl BooleanLike for TestValue {
        fn as_boolean(&self) -> Self {
            match self {
                Self::Bool(value) => Self::Bool(*value),
                Self::Number(value) => Self::Bool(*value != 0.0),
            }
        }

        fn boolean(&self) -> Result<bool, ProgramError> {
            match self {
                Self::Bool(value) => Ok(*value),
                value => Err(ProgramError::Concretization {
                    message: format!(
                        "cannot extract a concrete boolean from a value of type {}; expected bool[]",
                        value.r#type(),
                    ),
                }),
            }
        }
    }

    #[derive(Clone, Debug)]
    enum TestOperation {
        Add,
        Sub,
        IsPositive,
        Condition(Box<ConditionOperation<TestValue, TestOperation, ArrayType>>),
        While(Box<WhileOperation<TestValue, TestOperation, ArrayType>>),
    }

    impl Display for TestOperation {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "{}", self.name())
        }
    }

    impl Operation<ArrayType> for TestOperation {
        #[inline]
        fn name(&self) -> &'static str {
            match self {
                Self::Add => ADD_OPERATION_NAME,
                Self::Sub => SUB_OPERATION_NAME,
                Self::IsPositive => "is_positive",
                Self::Condition(condition) => condition.name(),
                Self::While(while_operation) => while_operation.name(),
            }
        }

        fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
            match self {
                Self::Add | Self::Sub => {
                    check_count!("input", input_types, 2, TypeError);
                    check_types!(self.name(), &input_types[..1], &input_types[1..]);
                    Ok(vec![input_types[0].clone()])
                }
                Self::IsPositive => {
                    check_count!("input", input_types, 1, TypeError);
                    Ok(vec![ArrayType::scalar(DataType::Boolean)])
                }
                Self::Condition(condition) => condition.infer_output_types(input_types),
                Self::While(while_operation) => while_operation.infer_output_types(input_types),
            }
        }

        fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
            match self {
                Self::Condition(condition) => condition.render(formatter, indentation),
                Self::While(while_operation) => while_operation.render(formatter, indentation),
                _ => Display::fmt(self, formatter),
            }
        }
    }

    impl InterpretableOperation<ArrayType, TestValue> for TestOperation {
        fn interpret(&self, inputs: &[TestValue]) -> Result<Vec<TestValue>, ProgramError> {
            match self {
                Self::Add => match (&inputs[0], &inputs[1]) {
                    (TestValue::Number(left), TestValue::Number(right)) => Ok(vec![TestValue::Number(left + right)]),
                    _ => Err(TypeError { message: ("add expected numeric inputs").into() }.into()),
                },
                Self::Sub => match (&inputs[0], &inputs[1]) {
                    (TestValue::Number(left), TestValue::Number(right)) => Ok(vec![TestValue::Number(left - right)]),
                    _ => Err(TypeError { message: ("sub expected numeric inputs").into() }.into()),
                },
                Self::IsPositive => match &inputs[0] {
                    TestValue::Number(value) => Ok(vec![TestValue::Bool(*value > 0.0)]),
                    _ => Err(TypeError { message: ("is_positive expected a numeric input").into() }.into()),
                },
                Self::Condition(condition) => condition.interpret(inputs),
                Self::While(while_operation) => while_operation.interpret(inputs),
            }
        }
    }

    #[derive(Clone, Debug)]
    enum TestLinearOperation {
        Add,
        Neg,
        Scale {
            factor: TestValue,
        },
        /// Captured-predicate condition staged by the [`ConditionOperation`] JVP rule. The predicate is stored as a
        /// [`ResidualFactor`] but interpretation only supports closed [`ResidualFactor::Constant`] predicates: this
        /// test enum is factor-invariant (its `WithFactor` is `Self`), so residual references cannot be rebound.
        Condition {
            predicate: ResidualFactor<ArrayType, TestValue>,
            true_branch: Box<Program<ArrayType, TestValue, TestLinearOperation, Vec<TestValue>, Vec<TestValue>>>,
            false_branch: Box<Program<ArrayType, TestValue, TestLinearOperation, Vec<TestValue>, Vec<TestValue>>>,
        },

        /// Captured-factor residual injection staged by the [`WhileOperation`] JVP rule. Like `Condition`,
        /// interpretation only supports closed [`ResidualFactor::Constant`] factors because this test enum is
        /// factor-invariant.
        Residual {
            factor: ResidualFactor<ArrayType, TestValue>,
        },

        /// Recomputed primal operation embedded in the fused linear while body.
        Recompute(TestDifferentiableOperation),

        /// Fused doubled-state while loop staged by the [`WhileOperation`] JVP rule.
        While(Box<WhileOperation<TestValue, TestLinearOperation, ArrayType>>),
    }

    impl Display for TestLinearOperation {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "{}", self.name())
        }
    }

    impl Operation<ArrayType> for TestLinearOperation {
        #[inline]
        fn name(&self) -> &'static str {
            match self {
                Self::Add => "linear_add",
                Self::Neg => "linear_neg",
                Self::Scale { .. } => "linear_scale",
                Self::Condition { .. } => "linear_condition",
                Self::Residual { .. } => "residual",
                Self::Recompute(operation) => operation.name(),
                Self::While(_) => "while",
            }
        }

        fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
            match self {
                Self::Add => {
                    check_count!("input", input_types, 2, TypeError);
                    check_types!(self.name(), &input_types[..1], &input_types[1..]);
                    Ok(vec![input_types[0].clone()])
                }
                Self::Neg | Self::Scale { .. } => {
                    check_count!("input", input_types, 1, TypeError);
                    Ok(vec![input_types[0].clone()])
                }
                Self::Condition { true_branch, .. } => {
                    check_types!("condition operand", &true_branch.input_types(), input_types);
                    Ok(true_branch.output_types())
                }
                Self::Residual { factor } => {
                    check_count!("input", input_types, 0, TypeError);
                    Ok(vec![factor.r#type().into_owned()])
                }
                Self::Recompute(operation) => operation.infer_output_types(input_types),
                Self::While(while_operation) => while_operation.infer_output_types(input_types),
            }
        }
    }

    impl InterpretableOperation<ArrayType, TestValue> for TestLinearOperation {
        fn interpret(&self, inputs: &[TestValue]) -> Result<Vec<TestValue>, ProgramError> {
            match self {
                Self::Add => match (&inputs[0], &inputs[1]) {
                    (TestValue::Number(left), TestValue::Number(right)) => Ok(vec![TestValue::Number(left + right)]),
                    _ => Err(TypeError { message: ("linear add expected numeric inputs").into() }.into()),
                },
                Self::Neg => match &inputs[0] {
                    TestValue::Number(value) => Ok(vec![TestValue::Number(-value)]),
                    _ => Err(TypeError { message: ("linear neg expected a numeric input").into() }.into()),
                },
                Self::Scale { factor } => match (factor, &inputs[0]) {
                    (TestValue::Number(factor), TestValue::Number(value)) => {
                        Ok(vec![TestValue::Number(factor * value)])
                    }
                    _ => Err(TypeError { message: ("linear scale expected numeric inputs").into() }.into()),
                },
                Self::Condition { predicate, true_branch, false_branch } => {
                    let ResidualFactor::Constant(predicate) = predicate else {
                        return Err(ProgramError::UnsupportedOperation {
                            message: "the test linear condition only interprets closed constant predicates".to_string(),
                        });
                    };
                    let branch = if predicate.boolean()? { true_branch } else { false_branch };
                    branch.interpret(inputs.to_vec())
                }
                Self::Residual { factor } => {
                    let ResidualFactor::Constant(value) = factor else {
                        return Err(ProgramError::UnsupportedOperation {
                            message: "the test linear residual only interprets closed constant factors".to_string(),
                        });
                    };
                    Ok(vec![value.clone()])
                }
                Self::Recompute(operation) => operation.interpret(inputs),
                Self::While(while_operation) => while_operation.interpret(inputs),
            }
        }
    }

    impl TransposableOperation<ArrayType, TestValue, TestLinearOperation> for TestLinearOperation {
        fn transpose<'transpose>(
            &self,
            _context: &mut AbstractTracingContext<'transpose, ArrayType, TestValue, TestLinearOperation>,
            _input_types: &[&ArrayType],
            output_cotangents: &[Cotangent<'transpose, ArrayType, TestValue, TestLinearOperation>],
        ) -> Result<Vec<Cotangent<'transpose, ArrayType, TestValue, TestLinearOperation>>, ProgramError> {
            match self {
                Self::Add => {
                    check_count!("output", output_cotangents, 1, ProgramError);
                    Ok(vec![output_cotangents[0].clone(), output_cotangents[0].clone()])
                }
                Self::Neg => {
                    check_count!("output", output_cotangents, 1, ProgramError);
                    match &output_cotangents[0] {
                        Cotangent::Staged(cotangent) => Ok(vec![Cotangent::Staged(-cotangent.clone())]),
                        Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                    }
                }
                Self::Scale { factor } => {
                    check_count!("output", output_cotangents, 1, ProgramError);
                    match &output_cotangents[0] {
                        Cotangent::Staged(cotangent) => {
                            Ok(vec![Cotangent::Staged(cotangent.clone().scale(factor.clone()))])
                        }
                        Cotangent::Zero => Ok(vec![Cotangent::Zero]),
                    }
                }
                Self::Condition { .. } | Self::Residual { .. } | Self::Recompute(_) | Self::While(_) => {
                    Err(ProgramError::UnsupportedOperation {
                        message: format!("the test linear operation {} does not support transposition", self.name()),
                    })
                }
            }
        }
    }

    impl<Factor: Value<ArrayType>> FactorParameterizedOperation<ArrayType, Factor> for TestLinearOperation {
        type WithFactor<MappedFactor: Value<ArrayType>> = Self;

        fn try_map_factors<MappedFactor: Value<ArrayType>, MapFactorFn>(
            &self,
            _map_factor: &mut MapFactorFn,
        ) -> Result<Self::WithFactor<MappedFactor>, ProgramError>
        where
            MapFactorFn: FnMut(&Factor) -> Result<MappedFactor, ProgramError>,
        {
            Ok(self.clone())
        }
    }

    impl SupportsAdd<ArrayType> for TestLinearOperation {
        fn add_operation() -> Self {
            Self::Add
        }
    }

    impl crate::operations::constants::SupportsZero<ArrayType> for TestLinearOperation {
        fn zero_operation(_type: ArrayType) -> Self {
            // The test linear operation enum doesn't include a Zero variant; the tests below never disconnect
            // primal inputs, so this constructor is unreachable in practice.
            Self::Scale { factor: TestValue::Number(0.0) }
        }
    }

    impl SupportsNeg<ArrayType> for TestLinearOperation {
        fn neg_operation() -> Self {
            Self::Neg
        }
    }

    impl SupportsScale<ArrayType, TestValue> for TestLinearOperation {
        fn scale_operation(factor: TestValue) -> Self {
            Self::Scale { factor }
        }
    }

    impl SupportsLinearCondition<ArrayType, TestValue, ResidualFactor<ArrayType, TestValue>> for TestLinearOperation {
        fn linear_condition_operation(
            predicate: ResidualFactor<ArrayType, TestValue>,
            true_branch: Program<ArrayType, TestValue, Self, Vec<TestValue>, Vec<TestValue>>,
            false_branch: Program<ArrayType, TestValue, Self, Vec<TestValue>, Vec<TestValue>>,
        ) -> Self {
            Self::Condition { predicate, true_branch: Box::new(true_branch), false_branch: Box::new(false_branch) }
        }
    }

    impl SupportsLinearWhile<ArrayType, TestValue, ResidualFactor<ArrayType, TestValue>, TestDifferentiableOperation>
        for TestLinearOperation
    {
        fn recompute_operation(operation: TestDifferentiableOperation) -> Self {
            Self::Recompute(operation)
        }

        fn residual_operation(factor: ResidualFactor<ArrayType, TestValue>) -> Self {
            Self::Residual { factor }
        }

        fn defactorize(
            &self,
            residual_atoms: &[AtomId],
            inputs: Vec<AtomId>,
        ) -> Result<DefactorizedOperation<Self>, ProgramError> {
            // The test linear enum is factor-invariant, so the only residual references it can carry are the ones in
            // its own nullary residual injections; everything else passes through unchanged.
            match self {
                Self::Residual { factor: ResidualFactor::Reference { index, .. } } => {
                    let atom = residual_atoms
                        .get(*index)
                        .copied()
                        .ok_or(ProgramError::UnboundAtomId { id: AtomId::new(*index) })?;
                    Ok(DefactorizedOperation::Forward { atom })
                }
                operation => Ok(DefactorizedOperation::Operation { operation: operation.clone(), inputs }),
            }
        }

        fn linear_while_operation(
            condition: Program<ArrayType, TestValue, Self, Vec<TestValue>, Vec<TestValue>>,
            body: Program<ArrayType, TestValue, Self, Vec<TestValue>, Vec<TestValue>>,
        ) -> Result<Self, TypeError> {
            Ok(Self::While(Box::new(WhileOperation::new(condition, body)?)))
        }
    }

    #[derive(Clone, Debug)]
    enum TestDifferentiableOperation {
        Zero(ArrayType),
        IsPositive,
        SubtractOne,
        Scale { factor: TestValue },
        Condition(Box<ConditionOperation<TestValue, TestDifferentiableOperation, ArrayType>>),
        While(Box<WhileOperation<TestValue, TestDifferentiableOperation, ArrayType>>),
    }

    impl From<ConditionOperation<TestValue, TestDifferentiableOperation, ArrayType>> for TestDifferentiableOperation {
        fn from(operation: ConditionOperation<TestValue, TestDifferentiableOperation, ArrayType>) -> Self {
            Self::Condition(Box::new(operation))
        }
    }

    impl From<WhileOperation<TestValue, TestDifferentiableOperation, ArrayType>> for TestDifferentiableOperation {
        fn from(operation: WhileOperation<TestValue, TestDifferentiableOperation, ArrayType>) -> Self {
            Self::While(Box::new(operation))
        }
    }

    impl Display for TestDifferentiableOperation {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "{}", self.name())
        }
    }

    impl Operation<ArrayType> for TestDifferentiableOperation {
        #[inline]
        fn name(&self) -> &'static str {
            match self {
                Self::Zero(_) => "zero",
                Self::IsPositive => "is_positive",
                Self::SubtractOne => "subtract_one",
                Self::Scale { .. } => "scale",
                Self::Condition(condition) => condition.name(),
                Self::While(while_operation) => while_operation.name(),
            }
        }

        fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
            match self {
                Self::Zero(value_type) => {
                    check_count!("input", input_types, 0, TypeError);
                    Ok(vec![value_type.clone()])
                }
                Self::IsPositive => {
                    check_count!("input", input_types, 1, TypeError);
                    Ok(vec![ArrayType::scalar(DataType::Boolean)])
                }
                Self::SubtractOne | Self::Scale { .. } => {
                    check_count!("input", input_types, 1, TypeError);
                    Ok(vec![input_types[0].clone()])
                }
                Self::Condition(condition) => condition.infer_output_types(input_types),
                Self::While(while_operation) => while_operation.infer_output_types(input_types),
            }
        }

        fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
            match self {
                Self::Condition(condition) => condition.render(formatter, indentation),
                Self::While(while_operation) => while_operation.render(formatter, indentation),
                _ => Display::fmt(self, formatter),
            }
        }
    }

    impl InterpretableOperation<ArrayType, TestValue> for TestDifferentiableOperation {
        fn interpret(&self, inputs: &[TestValue]) -> Result<Vec<TestValue>, ProgramError> {
            match self {
                Self::Zero(value_type) => {
                    check_count!("input", inputs, 0, ProgramError);
                    Ok(vec![TestValue::zero(value_type)?])
                }
                Self::IsPositive => match &inputs[0] {
                    TestValue::Number(value) => Ok(vec![TestValue::Bool(*value > 0.0)]),
                    _ => Err(TypeError { message: ("is_positive expected a numeric input").into() }.into()),
                },
                Self::SubtractOne => match &inputs[0] {
                    TestValue::Number(value) => Ok(vec![TestValue::Number(value - 1.0)]),
                    _ => Err(TypeError { message: ("subtract_one expected a numeric input").into() }.into()),
                },
                Self::Scale { factor } => match (factor, &inputs[0]) {
                    (TestValue::Number(factor), TestValue::Number(value)) => {
                        Ok(vec![TestValue::Number(factor * value)])
                    }
                    _ => Err(TypeError { message: ("scale expected numeric inputs").into() }.into()),
                },
                Self::Condition(condition) => condition.interpret(inputs),
                Self::While(while_operation) => while_operation.interpret(inputs),
            }
        }
    }

    impl SupportsZero<ArrayType> for TestDifferentiableOperation {
        fn zero_operation(r#type: ArrayType) -> Self {
            Self::Zero(r#type)
        }
    }

    #[derive(Copy, Clone, Debug)]
    struct TestDomain;

    impl Domain for TestDomain {
        type Type = ArrayType;
        type Value = TestValue;
        type Constant = TestValue;
        type Operation = TestDifferentiableOperation;
    }

    impl Context for TestDomain {
        fn lift(&self, constant: TestValue) -> Result<TestValue, ProgramError> {
            Ok(constant)
        }

        fn bind(&self, operation: Self::Operation, inputs: &[Self::Value]) -> Result<Vec<Self::Value>, ProgramError> {
            operation.interpret(inputs)
        }
    }

    impl DifferentiationContext for TestDomain {
        type Tangent = TestValue;
        type LinearOperation<V: Value<ArrayType>, F: Value<ArrayType>> = TestLinearOperation;

        fn zero_tangent(&self, type_: &Self::Type) -> Result<Self::Tangent, ProgramError> {
            let mut outputs =
                self.bind(<Self::Operation as SupportsZero<Self::Type>>::zero_operation(type_.clone()), &[])?;
            check_count!("output", outputs, 1, ProgramError);
            Ok(outputs.pop().expect("zero operation produces exactly one output"))
        }
    }

    /// Generic JVP dispatch for the test operation enum, mirroring the shape of the [`ArrayOperation`] dispatch so
    /// that the custom operations also differentiate against derived contexts such as
    /// [`NestedLinearizationContextOf`](crate::tracing_v2::differentiation::NestedLinearizationContextOf) (whose
    /// primal values are tracers). Primal results are produced through [`TangentContext::bind_primal`] so that they
    /// are interpreted eagerly or staged depending on the context. The `Condition` variant intentionally has no rule
    /// here: the tests below exercise [`ConditionOperation`]'s rule directly.
    impl<D> DifferentiableOperation<D> for TestDifferentiableOperation
    where
        D: DifferentiationContext<Type = ArrayType, Constant = TestValue>
            + Domain<Operation = TestDifferentiableOperation>,
        LinearOperationOf<D>: SupportsScale<ArrayType, TestValue>,
    {
        fn jvp<'jvp>(
            &self,
            context: &mut TangentContext<'jvp, D>,
            inputs: &[JvpTracer<'jvp, D>],
        ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
        where
            D: 'jvp,
        {
            match self {
                Self::Zero(value_type) => {
                    check_count!("input", inputs, 0, ProgramError);
                    let mut primals = context.bind_primal(Self::Zero(value_type.clone()), &[])?;
                    check_count!("output", primals, 1, ProgramError);
                    Ok(vec![JvpTracer::from_zero_tangent(primals.pop().unwrap(), value_type.clone())])
                }
                Self::IsPositive | Self::Condition(_) | Self::While(_) => {
                    Err(ProgramError::UnsupportedOperation { message: format!("missing jvp rule for {}", self.name()) })
                }
                Self::SubtractOne => {
                    check_count!("input", inputs, 1, ProgramError);
                    let mut primals = context.bind_primal(self.clone(), &[inputs[0].primal().clone()])?;
                    check_count!("output", primals, 1, ProgramError);
                    Ok(vec![JvpTracer::new(primals.pop().unwrap(), inputs[0].tangent().clone())])
                }
                Self::Scale { factor } => {
                    check_count!("input", inputs, 1, ProgramError);
                    let mut primals = context.bind_primal(self.clone(), &[inputs[0].primal().clone()])?;
                    check_count!("output", primals, 1, ProgramError);
                    let materialized_tangent = context.materialize_tangent(inputs[0].tangent().clone())?;
                    let tangent_outputs = context.stage_operation(
                        LinearOperationOf::<D>::scale_operation(factor.clone()),
                        &[materialized_tangent],
                    )?;
                    check_count!("output", tangent_outputs, 1, ProgramError);
                    Ok(vec![JvpTracer::from_value(primals.pop().unwrap(), tangent_outputs[0].clone())])
                }
            }
        }
    }

    impl SupportsNestedLinearization<TestDomain> for TestDifferentiableOperation {
        fn linearize_nested_program(
            differentiable: &TestDomain,
            program: &Program<ArrayType, TestValue, Self, Vec<TestValue>, Vec<TestValue>>,
        ) -> Result<NestedLinearization<TestDomain, Self>, ProgramError> {
            crate::tracing_v2::differentiation::linearize_nested_program(differentiable, program)
        }
    }

    fn add_one_branch() -> Program<ArrayType, TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>> {
        let mut builder = ProgramBuilder::<ArrayType, TestValue, TestOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let one = builder.add_constant(TestValue::Number(1.0));
        let output = builder.add_instruction(TestOperation::Add, vec![input, one]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    fn subtract_one_branch() -> Program<ArrayType, TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>> {
        let mut builder = ProgramBuilder::<ArrayType, TestValue, TestOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let one = builder.add_constant(TestValue::Number(1.0));
        let output = builder.add_instruction(TestOperation::Sub, vec![input, one]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    fn identity_array_branch()
    -> Program<ArrayType, TestValue, ArrayOperation<TestValue, ArrayType>, Vec<TestValue>, Vec<TestValue>> {
        let mut builder = ProgramBuilder::<ArrayType, TestValue, ArrayOperation<TestValue, ArrayType>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    fn custom_scale_branch(
        factor: f64,
    ) -> Program<ArrayType, TestValue, TestDifferentiableOperation, Vec<TestValue>, Vec<TestValue>> {
        let mut builder = ProgramBuilder::<ArrayType, TestValue, TestDifferentiableOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(TestDifferentiableOperation::Scale { factor: TestValue::Number(factor) }, vec![input])
            .unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    fn custom_while_condition_branch()
    -> Program<ArrayType, TestValue, TestDifferentiableOperation, Vec<TestValue>, Vec<TestValue>> {
        let mut builder = ProgramBuilder::<ArrayType, TestValue, TestDifferentiableOperation>::new();
        let counter = builder.add_input(ArrayType::scalar(DataType::F64));
        let _value = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(TestDifferentiableOperation::IsPositive, vec![counter]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder, Placeholder], vec![Placeholder]).unwrap()
    }

    fn custom_while_body_branch()
    -> Program<ArrayType, TestValue, TestDifferentiableOperation, Vec<TestValue>, Vec<TestValue>> {
        let mut builder = ProgramBuilder::<ArrayType, TestValue, TestDifferentiableOperation>::new();
        let counter = builder.add_input(ArrayType::scalar(DataType::F64));
        let value = builder.add_input(ArrayType::scalar(DataType::F64));
        let next_counter = builder.add_instruction(TestDifferentiableOperation::SubtractOne, vec![counter]).unwrap()[0];
        let next_value = builder
            .add_instruction(TestDifferentiableOperation::Scale { factor: TestValue::Number(2.0) }, vec![value])
            .unwrap()[0];
        builder
            .build(vec![next_counter, next_value], vec![Placeholder, Placeholder], vec![Placeholder, Placeholder])
            .unwrap()
    }

    #[test]
    fn test_condition_interprets_true_and_false_branches() {
        let condition =
            ConditionOperation::new(ArrayType::scalar(DataType::Boolean), add_one_branch(), subtract_one_branch())
                .unwrap();

        assert_eq!(
            condition.interpret(&[TestValue::Bool(true), TestValue::Number(3.0)]),
            Ok(vec![TestValue::Number(4.0)]),
        );
        assert_eq!(
            condition.interpret(&[TestValue::Bool(false), TestValue::Number(3.0)]),
            Ok(vec![TestValue::Number(2.0)]),
        );
    }

    #[test]
    fn test_condition_program_rendering_includes_nested_branches() {
        let condition =
            ConditionOperation::new(ArrayType::scalar(DataType::Boolean), add_one_branch(), subtract_one_branch())
                .unwrap();
        let mut builder = ProgramBuilder::<ArrayType, TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(TestOperation::Condition(Box::new(condition)), vec![predicate, input])
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();

        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[] .
                let %2:f64[] = condition [
                    true_branch={
                        lambda %0:f64[] .
                        let %1:f64[] = const
                            %2:f64[] = add %0 %1
                        in (%2)
                    },
                    false_branch={
                        lambda %0:f64[] .
                        let %1:f64[] = const
                            %2:f64[] = sub %0 %1
                        in (%2)
                    },
                ] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_condition_rejects_branch_output_mismatch() {
        let mut builder = ProgramBuilder::<ArrayType, TestValue, TestOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(TestOperation::IsPositive, vec![input]).unwrap()[0];
        let bool_branch = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        assert!(ConditionOperation::new(ArrayType::scalar(DataType::Boolean), add_one_branch(), bool_branch).is_err());
    }

    #[test]
    fn test_while_interprets_until_condition_is_false() {
        let mut condition_builder = ProgramBuilder::<ArrayType, TestValue, TestOperation>::new();
        let condition_input = condition_builder.add_input(ArrayType::scalar(DataType::F64));
        let condition_output =
            condition_builder.add_instruction(TestOperation::IsPositive, vec![condition_input]).unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![condition_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let while_operation = WhileOperation::new(condition, subtract_one_branch()).unwrap();

        assert_eq!(while_operation.interpret(&[TestValue::Number(3.0)]), Ok(vec![TestValue::Number(0.0)]),);
    }

    #[test]
    fn test_while_program_rendering_includes_condition_and_body() {
        let mut condition_builder = ProgramBuilder::<ArrayType, TestValue, TestOperation>::new();
        let condition_input = condition_builder.add_input(ArrayType::scalar(DataType::F64));
        let condition_output =
            condition_builder.add_instruction(TestOperation::IsPositive, vec![condition_input]).unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![condition_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let while_operation = WhileOperation::new(condition, subtract_one_branch()).unwrap();
        let mut builder = ProgramBuilder::<ArrayType, TestValue, TestOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(TestOperation::While(Box::new(while_operation)), vec![input]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = while [
                    condition={
                        lambda %0:f64[] .
                        let %1:bool[] = is_positive %0
                        in (%1)
                    },
                    body={
                        lambda %0:f64[] .
                        let %1:f64[] = const
                            %2:f64[] = sub %0 %1
                        in (%2)
                    },
                ] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_array_operation_condition_infers_output_types() {
        let condition = ConditionOperation::new(
            ArrayType::scalar(DataType::Boolean),
            identity_array_branch(),
            identity_array_branch(),
        )
        .unwrap();
        let operation = ArrayOperation::Condition(Box::new(condition));

        assert_eq!(
            operation.infer_output_types(&[ArrayType::scalar(DataType::Boolean), ArrayType::scalar(DataType::F64)]),
            Ok(vec![ArrayType::scalar(DataType::F64)]),
        );
    }

    fn expect_tangent_value<'jvp, T: crate::types::Type, V: crate::programs::Value<T>>(tangent: &Tangent<T, V>) -> V {
        match tangent {
            Tangent::Value(value) => value.clone(),
            Tangent::Zero(_) => {
                panic!("expected a concrete tangent value, not a symbolic zero")
            }
        }
    }

    #[test]
    fn test_generic_condition_jvp_uses_custom_operations() {
        let condition = ConditionOperation::new(
            ArrayType::scalar(DataType::Boolean),
            custom_scale_branch(2.0),
            custom_scale_branch(3.0),
        )
        .unwrap();
        let domain = TestDomain;
        let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, TestValue, TestLinearOperation>::new()));
        let mut context = TangentContext::new(&domain, builder.clone());
        let tangent_input = context.input(ArrayType::scalar(DataType::F64));
        let predicate = JvpTracer::from_zero_tangent(TestValue::Bool(true), ArrayType::scalar(DataType::Boolean));
        let outputs = condition
            .jvp(&mut context, &[predicate, JvpTracer::from_value(TestValue::Number(4.0), tangent_input)])
            .unwrap();

        assert_eq!(outputs[0].primal(), &TestValue::Number(8.0));
        let tangent_output = expect_tangent_value(outputs[0].tangent()).atom_id().unwrap();
        drop(outputs);
        drop(context);
        let builder = Rc::try_unwrap(builder).unwrap().into_inner();
        let tangent_program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![tangent_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(tangent_program.interpret(vec![TestValue::Number(10.0)]), Ok(vec![TestValue::Number(20.0)]));
    }

    #[test]
    fn test_condition_jvp_stages_condition_under_abstract_tracing() {
        // The headline capability of the staged-condition JVP rule: differentiating a runtime-predicate condition
        // under abstract tracing (a tracer-valued differentiation context) succeeds by staging condition structure
        // instead of concretizing the predicate, which previously failed with `ProgramError::Concretization`. The
        // true branch is sin(x), whose pushforward captures cos(x) as a residual, so the staged primal condition
        // must carry residual-extended joined branches; the false branch is 3 * x, which captures no residuals.
        use std::convert::Infallible;

        use crate::tests::{TestArray, TestArrayDomain};
        use crate::tracing::{DomainTracer, TracingContext};
        use crate::tracing_v2::LinearArrayOperation;
        use crate::tracing_v2::test_util::scalar_scale_branch;

        let mut sin_builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray, ArrayType>>::new();
        let sin_input = sin_builder.add_input(ArrayType::scalar(DataType::F64));
        let sin_output = sin_builder.add_instruction(ArrayOperation::Sin, vec![sin_input]).unwrap()[0];
        let sin_branch = sin_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![sin_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let condition =
            ConditionOperation::new(ArrayType::scalar(DataType::Boolean), sin_branch, scalar_scale_branch(3.0))
                .unwrap();

        let outer_builder =
            Rc::new(RefCell::new(ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray, ArrayType>>::new()));
        let predicate_input = outer_builder.borrow_mut().add_input(ArrayType::scalar(DataType::Boolean));
        let operand_input = outer_builder.borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let outer_context = TracingContext::new(&TestArrayDomain, outer_builder.clone());
        let primal_predicate = outer_context.tracer(predicate_input, None);
        let primal_operand = outer_context.tracer(operand_input, None);

        let linear_builder = Rc::new(RefCell::new(ProgramBuilder::<
            ArrayType,
            DomainTracer<TestArrayDomain>,
            LinearArrayOperation<
                DomainTracer<TestArrayDomain>,
                TestArray,
                ArrayType,
                Infallible,
                ResidualFactor<ArrayType, DomainTracer<TestArrayDomain>>,
            >,
        >::new()));
        let mut context = TangentContext::new(&outer_context, linear_builder.clone());
        let tangent_operand = context.input(ArrayType::scalar(DataType::F64));

        let outputs = condition
            .jvp(
                &mut context,
                &[
                    JvpTracer::from_zero_tangent(primal_predicate, ArrayType::scalar(DataType::Boolean)),
                    JvpTracer::from_value(primal_operand, tangent_operand),
                ],
            )
            .expect("the condition JVP rule should stage condition structure instead of concretizing the predicate");
        assert_eq!(outputs.len(), 1);

        // The primal trace gained exactly one condition over the residual-extended joined branches: the original
        // output plus the true branch's cos(x) residual, with the false branch padding that slot with a typed zero.
        let outer_builder = outer_builder.borrow();
        assert_eq!(outer_builder.instructions().len(), 1);
        let staged_primal = outer_builder.instructions()[0].operation();
        assert_eq!(staged_primal.name(), "condition");
        let ArrayOperation::Condition(staged_condition) = staged_primal else {
            panic!("expected the staged primal operation to be a condition");
        };
        assert_eq!(staged_condition.true_branch().output_ids().len(), 2);
        assert_eq!(staged_condition.false_branch().output_ids().len(), 2);

        // The linear trace gained exactly one captured-predicate linear condition over the operand tangent.
        let linear_builder = linear_builder.borrow();
        assert_eq!(linear_builder.instructions().len(), 1);
        let staged_linear = linear_builder.instructions()[0].operation();
        assert_eq!(staged_linear.name(), "condition");
        assert!(matches!(staged_linear, LinearArrayOperation::Condition { .. }));
    }

    #[test]
    fn test_generic_while_jvp_propagates_tangents_through_iterations() {
        let while_operation = WhileOperation::new(custom_while_condition_branch(), custom_while_body_branch()).unwrap();
        let domain = TestDomain;
        let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, TestValue, TestLinearOperation>::new()));
        let mut context = TangentContext::new(&domain, builder.clone());
        let counter_tangent_input = context.input(ArrayType::scalar(DataType::F64));
        let value_tangent_input = context.input(ArrayType::scalar(DataType::F64));
        let outputs = while_operation
            .jvp(
                &mut context,
                &[
                    JvpTracer::from_value(TestValue::Number(3.0), counter_tangent_input),
                    JvpTracer::from_value(TestValue::Number(5.0), value_tangent_input),
                ],
            )
            .unwrap();

        assert_eq!(
            outputs.iter().map(|output| output.primal().clone()).collect::<Vec<_>>(),
            vec![TestValue::Number(0.0), TestValue::Number(40.0)],
        );
        let tangent_outputs = outputs
            .iter()
            .map(|output| expect_tangent_value(output.tangent()).atom_id().unwrap())
            .collect::<Vec<_>>();
        drop(outputs);
        drop(context);
        let builder = Rc::try_unwrap(builder).unwrap().into_inner();
        let tangent_program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                tangent_outputs,
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        assert_eq!(
            tangent_program.interpret(vec![TestValue::Number(0.0), TestValue::Number(1.0)]),
            Ok(vec![TestValue::Number(0.0), TestValue::Number(8.0)]),
        );
    }

    #[test]
    fn test_while_jvp_stages_doubled_state_loop_under_abstract_tracing() {
        // The headline capability of the staged-while JVP rule: differentiating a while loop under abstract tracing
        // (a tracer-valued differentiation context) succeeds by staging loop structure instead of concretizing the
        // loop predicate, which previously failed with `ProgramError::Concretization`. The loop doubles its state
        // until it reaches 8: the primal trace must gain exactly one `while` over the original condition and body,
        // and the linear trace must gain one nullary residual injection (the loop-entry primal state) plus one
        // doubled-state linear `while` whose fused programs consume `[primal_state, tangent_state]`.
        use std::convert::Infallible;

        use crate::operations::compare::ComparisonDirection;
        use crate::tests::{TestArray, TestArrayDomain};
        use crate::tracing::{DomainTracer, TracingContext};
        use crate::tracing_v2::LinearArrayOperation;

        type TestArrayOp = ArrayOperation<TestArray, ArrayType>;

        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<ArrayType, TestArray, TestArrayOp>::new();
        let condition_state = condition_builder.add_input(scalar_f64.clone());
        let threshold = condition_builder.add_constant(TestArray::scalar(8.0));
        let predicate = condition_builder
            .add_instruction(
                TestArrayOp::Compare { direction: ComparisonDirection::LessThan },
                vec![condition_state, threshold],
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<ArrayType, TestArray, TestArrayOp>::new();
        let body_state = body_builder.add_input(scalar_f64.clone());
        let doubled = body_builder.add_instruction(TestArrayOp::Add, vec![body_state, body_state]).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![doubled], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let while_operation = WhileOperation::<TestArray, TestArrayOp, ArrayType>::new(condition, body).unwrap();

        let outer_builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, TestArray, TestArrayOp>::new()));
        let state_input = outer_builder.borrow_mut().add_input(scalar_f64.clone());
        let outer_context = TracingContext::new(&TestArrayDomain, outer_builder.clone());
        let primal_state = outer_context.tracer(state_input, None);

        let linear_builder = Rc::new(RefCell::new(ProgramBuilder::<
            ArrayType,
            DomainTracer<TestArrayDomain>,
            LinearArrayOperation<
                DomainTracer<TestArrayDomain>,
                TestArray,
                ArrayType,
                Infallible,
                ResidualFactor<ArrayType, DomainTracer<TestArrayDomain>>,
            >,
        >::new()));
        let mut context = TangentContext::new(&outer_context, linear_builder.clone());
        let tangent_state = context.input(scalar_f64.clone());

        let outputs = while_operation
            .jvp(&mut context, &[JvpTracer::from_value(primal_state, tangent_state)])
            .expect("the while JVP rule should stage loop structure instead of concretizing the predicate");
        assert_eq!(outputs.len(), 1);

        // The primal trace gained exactly one while over the original condition and body.
        let outer_builder = outer_builder.borrow();
        assert_eq!(outer_builder.instructions().len(), 1);
        let staged_primal = outer_builder.instructions()[0].operation();
        assert_eq!(staged_primal.name(), "while");
        let ArrayOperation::While(staged_while) = staged_primal else {
            panic!("expected the staged primal operation to be a while loop");
        };
        assert_eq!(staged_while.condition().input_types(), vec![scalar_f64.clone()]);
        assert_eq!(staged_while.body().input_types(), vec![scalar_f64.clone()]);

        // The linear trace gained one residual injection for the loop-entry primal state and one doubled-state
        // linear while whose fused condition and body consume `[primal_state, tangent_state]`.
        let linear_builder = linear_builder.borrow();
        assert_eq!(linear_builder.instructions().len(), 2);
        assert_eq!(linear_builder.instructions()[0].operation().name(), "residual");
        let staged_linear = linear_builder.instructions()[1].operation();
        assert_eq!(staged_linear.name(), "while");
        let LinearArrayOperation::While(staged_linear_while) = staged_linear else {
            panic!("expected the staged linear operation to be a while loop");
        };
        assert_eq!(staged_linear_while.condition().input_types(), vec![scalar_f64.clone(), scalar_f64.clone()]);
        assert_eq!(staged_linear_while.body().input_types(), vec![scalar_f64.clone(), scalar_f64.clone()]);
        assert_eq!(staged_linear_while.body().output_types(), vec![scalar_f64.clone(), scalar_f64]);
    }
}
