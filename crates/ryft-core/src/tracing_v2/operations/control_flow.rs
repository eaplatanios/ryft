use std::fmt::Debug;

use crate::batching::BatchingError;
use crate::contexts::{Context, StagingContext};
use crate::differentiation::{Cotangent, TransposableOperation};
use crate::domains::Domain;
use crate::macros::check_count;
use crate::operations::arithmetic::SupportsAdd;
use crate::operations::constants::{SupportsOne, SupportsZero};
use crate::operations::control_flow::scan::stacked_scan_type;
use crate::operations::control_flow::{ConditionOperation, SupportsSelect, WhileOperation};
use crate::operations::logical::SupportsAnd;
use crate::operations::manipulation::{
    Broadcast, SupportsBroadcast, SupportsDynamicUpdateSlice, SupportsTranspose, Transpose,
};
use crate::operations::{BooleanLike, Operation};
use crate::parameters::{Parameterized, ParameterizedFamily, Placeholder};
use crate::programs::{Atom, AtomId, Instruction, Program, ProgramBuilder, ProgramError, Value};
use crate::tracing::{AbstractTracer, AbstractTracingContext, Tracer};
use crate::tracing_v2::batching::{
    ArrayBatch, BatchableOperation, BatchingContext, ProgramBatchableOperation, ProgramBatchingOutputAxes,
    align_batch_axis, broadcast_to_batched, move_axis_permutation,
};
use crate::tracing_v2::differentiation::{NestedLinearization, ProgramLinearizableOperation};
use crate::tracing_v2::operations::reduce::{ReductionKind, SupportsReduce};
use crate::tracing_v2::operations::scan::SupportsLinearScan;
use crate::tracing_v2::operations::select::SupportsLinearSelect;
use crate::tracing_v2::{
    DifferentiableOperation, DifferentiationContext, FactorParameterizedOperation, JvpTracer, LinearOperationOf,
    ResidualFactor, ResidualizedOperation, TangentContext,
};
use crate::types::{ArrayType, DataType, Size, Type, TypeError, Typed};

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
    /// operations (for example, a scale by a referenced residual becomes a recomputed elementwise product and a
    /// select over a referenced condition becomes a recomputed operand-form select), with the residual atom spliced
    /// into `inputs`. Every such rewrite is wrapped in the recomputed-primal form produced by
    /// [`Self::recompute_operation`] so fused bodies carry uniform provenance. Higher-order payloads rewrite
    /// recursively: a condition over a referenced predicate becomes an operand-form condition whose branches receive
    /// the union of their referenced residuals as forwarded trailing operands, and a linear scan whose residual
    /// stacks reference loop-varying residuals moves those stacks into extra scanned operands with its body
    /// rewritten against the new trailing lane inputs. Operations carrying only closed [`ResidualFactor::Constant`]
    /// factors pass through unchanged, and the remaining unsupported shapes (for example, custom VJP call residual
    /// references or a constant-predicate condition whose branches reference loop-varying residuals) are rejected
    /// with precise errors.
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

/// Appends one fresh variable atom to a built `program` by direct program-field extension (every appended atom is a
/// fresh variable, so the [`Program`] invariants that [`ProgramBuilder`] would have established are preserved) and
/// returns its id.
fn append_program_variable<V: Value<ArrayType>, O>(
    program: &mut Program<ArrayType, V, O, Vec<V>, Vec<V>>,
    r#type: ArrayType,
) -> AtomId {
    let id = AtomId::new(program.atoms.len());
    program.atoms.push(Atom::Variable(r#type));
    id
}

/// Appends one instruction with a single fresh output atom to a built `program` by direct program-field extension
/// (the appended instruction reads existing atoms and writes a fresh variable, so the [`Program`] invariants that
/// [`ProgramBuilder`] would have established are preserved) and returns the output id.
fn append_program_instruction<V: Value<ArrayType>, O>(
    program: &mut Program<ArrayType, V, O, Vec<V>, Vec<V>>,
    operation: O,
    inputs: Vec<AtomId>,
    output_type: ArrayType,
) -> AtomId {
    let output = append_program_variable(program, output_type);
    program.instructions.push(Instruction::new(operation, inputs, vec![output]));
    output
}

/// Normalizes output `output_index` of a naturally batched program (see [`batch_program`](
/// crate::tracing_v2::batching::batch_program)) to carry its mapped lane axis at `target_axis` by appending a
/// staged axis-moving operation at the program tail: a transpose when the output is batched at a different axis, and
/// a broadcast that inserts the lane axis when the output is lane-uniform. The staged `condition` and `while`
/// batching rules use this to make the batched programs they capture agree on one output-axis layout (matching
/// branch signatures for `condition` and loop-invariant state types for `while`).
fn normalize_batched_program_output_axis<V, O>(
    program: &mut Program<ArrayType, V, O, Vec<V>, Vec<V>>,
    output_index: usize,
    current_axis: Option<usize>,
    target_axis: usize,
    axis_size: usize,
) -> Result<(), ProgramError>
where
    V: Value<ArrayType>,
    O: Operation<ArrayType> + SupportsTranspose<ArrayType> + SupportsBroadcast<ArrayType>,
{
    let output_id = program.output_ids[output_index];
    let output_type = program.output_types()[output_index].clone();
    match current_axis {
        Some(axis) if axis == target_axis => {}
        Some(axis) => {
            let (logical_type, lane_dimension) = output_type.without_dimension(axis)?;
            let permuted_type = logical_type.with_inserted_dimension(target_axis, lane_dimension)?;
            let permutation = move_axis_permutation(output_type.rank(), axis, target_axis);
            program.output_ids[output_index] = append_program_instruction(
                program,
                O::transpose_operation(permutation),
                vec![output_id],
                permuted_type,
            );
        }
        None => {
            let physical_type = output_type.with_inserted_dimension(target_axis, Size::Static(axis_size))?;
            let output_axes = (0..output_type.rank())
                .map(|axis| if axis < target_axis { axis } else { axis + 1 })
                .collect::<Vec<_>>();
            program.output_ids[output_index] = append_program_instruction(
                program,
                O::broadcast_operation(physical_type.clone(), output_axes),
                vec![output_id],
                physical_type,
            );
        }
    }
    Ok(())
}

/// Inlines `program` into `builder`, mapping the program's input atoms onto `input_atoms`, copying its constants,
/// and staging its instructions verbatim. Returns the builder atoms carrying the program outputs. The staged masked
/// `while` batching rule uses this to compose batched body and condition programs into one masked loop body.
fn inline_program_into_builder<V, O>(
    builder: &mut ProgramBuilder<ArrayType, V, O>,
    program: &Program<ArrayType, V, O, Vec<V>, Vec<V>>,
    input_atoms: &[AtomId],
) -> Result<Vec<AtomId>, ProgramError>
where
    V: Value<ArrayType>,
    O: Clone + Operation<ArrayType>,
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
        let outputs = builder.add_instruction(instruction.operation().clone(), inputs)?.to_vec();
        check_count!("output", outputs, instruction.outputs().len(), ProgramError);
        for (program_atom, builder_atom) in instruction.outputs().iter().zip(outputs.into_iter()) {
            atom_map[program_atom.index()] = Some(builder_atom);
        }
    }
    program.output_ids().iter().map(|output| map_atom(atom_map.as_slice(), *output)).collect()
}

/// Extends a residual-extended condition branch program to the joined output signature
/// `[original_outputs..., true_branch_residuals..., false_branch_residuals...]`.
///
/// `program` must already produce `[original_outputs..., own_residuals...]` (the shape produced by
/// [`linearize_program`]). This helper appends one typed nullary zero instruction per peer-branch residual
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

/// Builds the augmented condition and body programs of the bounded staged while loop (see the [`WhileOperation`] JVP
/// rule below) by direct program-field extension, the same precedent [`join_condition_branch_outputs`] uses: appended
/// input atoms and instructions reference existing atoms or fresh variables, so every [`Program`] invariant that
/// [`ProgramBuilder`] would have established is preserved.
///
/// The augmented loop state is `[original_state..., counter (i64 scalar), residual_stacks..., mask_stack]`:
///
///   - The body runs the residual-extended primal body (which outputs `[next_state..., residuals...]`) on the
///     original state slots, then *stores* instead of returning each per-iteration residual: residual `k` is
///     broadcast to `[1, …]` and written into stack `k` at lane `counter` via `dynamic_update_slice`, a scalar
///     Boolean `one` (true) is written into the Boolean `[bound]` mask stack at lane `counter`, and the counter
///     advances by an i64 `one`. Because the enclosing while keeps `iteration_bound = bound`, the counter is always
///     strictly below `bound` whenever the body runs, so the writes can never clamp.
///   - The condition is the original loop condition extended with ignored extra-state inputs.
///
/// Returns the extended condition, the augmented body, and the `[bound, …]` residual stack types.
fn build_bounded_while_programs<V, O>(
    condition: &Program<ArrayType, V, O, Vec<V>, Vec<V>>,
    primal_body: &Program<ArrayType, V, O, Vec<V>, Vec<V>>,
    residual_types: &[ArrayType],
    bound: usize,
) -> Result<
    (Program<ArrayType, V, O, Vec<V>, Vec<V>>, Program<ArrayType, V, O, Vec<V>, Vec<V>>, Vec<ArrayType>),
    ProgramError,
>
where
    V: Value<ArrayType>,
    O: Clone
        + Operation<ArrayType>
        + SupportsZero<ArrayType>
        + SupportsOne<ArrayType>
        + SupportsAdd<ArrayType>
        + SupportsBroadcast<ArrayType>
        + SupportsDynamicUpdateSlice<ArrayType>,
{
    let state_count = condition.input_types().len();
    let counter_type = ArrayType::scalar(DataType::I64);
    let boolean_scalar_type = ArrayType::scalar(DataType::Boolean);
    let mask_stack_type = stacked_scan_type(&boolean_scalar_type, bound);
    for residual_type in residual_types {
        if residual_type.static_shape().is_none() {
            return Err(TypeError {
                message: format!(
                    "jvp of a bounded while loop requires statically shaped body residuals but got {residual_type}",
                ),
            }
            .into());
        }
    }
    let stack_types = residual_types
        .iter()
        .map(|residual_type| stacked_scan_type(residual_type, bound))
        .collect::<Vec<_>>();

    // Body: append the extra loop-state inputs, store each residual into its stack at lane `counter`, mark lane
    // `counter` valid in the mask stack, and advance the counter.
    let mut body = primal_body.clone();
    let counter_input = append_program_variable(&mut body, counter_type.clone());
    body.input_ids.push(counter_input);
    let stack_inputs = stack_types
        .iter()
        .map(|stack_type| {
            let stack_input = append_program_variable(&mut body, stack_type.clone());
            body.input_ids.push(stack_input);
            stack_input
        })
        .collect::<Vec<_>>();
    let mask_input = append_program_variable(&mut body, mask_stack_type.clone());
    body.input_ids.push(mask_input);
    body.input_structure = vec![Placeholder; body.input_ids.len()];
    let residual_outputs = body.output_ids.split_off(state_count);
    check_count!("output", residual_outputs, residual_types.len(), ProgramError);
    let zero_index = residual_types.iter().any(|residual_type| residual_type.rank() > 0).then(|| {
        append_program_instruction(&mut body, O::zero_operation(counter_type.clone()), vec![], counter_type.clone())
    });
    let mut next_stacks = Vec::with_capacity(stack_types.len());
    for ((residual_output, residual_type), (stack_input, stack_type)) in
        residual_outputs.iter().zip(residual_types).zip(stack_inputs.iter().zip(stack_types.iter()))
    {
        let lane_type = stacked_scan_type(residual_type, 1);
        let output_axes = (1..=residual_type.rank()).collect::<Vec<_>>();
        let expanded = append_program_instruction(
            &mut body,
            O::broadcast_operation(lane_type.clone(), output_axes),
            vec![*residual_output],
            lane_type,
        );
        let mut write_inputs = vec![*stack_input, expanded, counter_input];
        // These unwraps are safe because `zero_index` is staged whenever some residual has rank at least one.
        write_inputs.extend((0..residual_type.rank()).map(|_| zero_index.unwrap()));
        next_stacks.push(append_program_instruction(
            &mut body,
            O::dynamic_update_slice_operation(),
            write_inputs,
            stack_type.clone(),
        ));
    }
    let true_scalar = append_program_instruction(
        &mut body,
        O::one_operation(boolean_scalar_type.clone()),
        vec![],
        boolean_scalar_type.clone(),
    );
    let true_lane_type = stacked_scan_type(&boolean_scalar_type, 1);
    let true_lane = append_program_instruction(
        &mut body,
        O::broadcast_operation(true_lane_type.clone(), vec![]),
        vec![true_scalar],
        true_lane_type,
    );
    let next_mask = append_program_instruction(
        &mut body,
        O::dynamic_update_slice_operation(),
        vec![mask_input, true_lane, counter_input],
        mask_stack_type.clone(),
    );
    let one_i64 =
        append_program_instruction(&mut body, O::one_operation(counter_type.clone()), vec![], counter_type.clone());
    let next_counter =
        append_program_instruction(&mut body, O::add_operation(), vec![counter_input, one_i64], counter_type.clone());
    body.output_ids.push(next_counter);
    body.output_ids.extend(next_stacks);
    body.output_ids.push(next_mask);
    body.output_structure = vec![Placeholder; body.output_ids.len()];

    // Condition: the original loop condition extended with ignored extra-state inputs.
    let mut extended_condition = condition.clone();
    let extra_state_types = std::iter::once(counter_type)
        .chain(stack_types.iter().cloned())
        .chain(std::iter::once(mask_stack_type));
    for extra_state_type in extra_state_types {
        let extra_input = append_program_variable(&mut extended_condition, extra_state_type);
        extended_condition.input_ids.push(extra_input);
    }
    extended_condition.input_structure = vec![Placeholder; extended_condition.input_ids.len()];
    Ok((extended_condition, body, stack_types))
}

/// JVP rule for [`ConditionOperation`] with full JAX
/// [`cond`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html) parity: the rule never concretizes the
/// predicate, so forward-mode differentiation of a runtime-predicate condition composes under abstract tracing
/// (tracer-valued differentiation contexts) by staging condition structure instead.
///
/// The rule mirrors JAX's `cond` JVP plus partial evaluation:
///
///   1. Both branches are linearized *symbolically* at the branch input types via [`linearize_program`] — no
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
        + ProgramLinearizableOperation<D>,
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
        check_count!("input", inputs, self.true_branch().input_types().len() + 1, ProgramError);
        let predicate = &inputs[0];
        let operands = &inputs[1..];

        // Linearize both branches symbolically at the branch input types and join their residual signatures.
        let NestedLinearization {
            primal_program: true_primal_program,
            pushforward_program: true_pushforward_program,
            residual_types: true_residual_types,
        } = self.true_branch().linearize(context.differentiable())?;
        let NestedLinearization {
            primal_program: false_primal_program,
            pushforward_program: false_pushforward_program,
            residual_types: false_residual_types,
        } = self.false_branch().linearize(context.differentiable())?;
        let output_count = self.true_branch().output_ids().len();
        let joined_true_branch =
            join_condition_branch_outputs(true_primal_program, output_count, false_residual_types.as_slice(), true);
        let joined_false_branch =
            join_condition_branch_outputs(false_primal_program, output_count, true_residual_types.as_slice(), false);

        // Bind one primal condition over the joined branches. `ConditionOperation::new` validates that the joined
        // branches agree on their input and output signatures; eager domains interpret the staged condition and so
        // still evaluate only the branch selected by the runtime predicate.
        let primal_condition = ConditionOperation::new(joined_true_branch, joined_false_branch)?;
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
    /// Rejects transposition. This rule is only reachable for *unbounded* staged while loops — the doubled-state
    /// fused linear loop staged by the [`WhileOperation`] JVP rule, which recomputes primal state *forward* through
    /// the iterations, so transposing it would have to run that recomputation backwards, which a while loop cannot
    /// express. Two paths avoid it entirely: concretizing domains unroll the loop into a straight-line pushforward
    /// that transposes (so eager reverse-mode differentiation through unbounded while loops works), and bounded
    /// loops ([`WhileOperation::with_iteration_bound`]) never stage a linear `while` — their tangent side is a
    /// masked linear scan whose transpose is total, so reverse mode through staged bounded loops flows through the
    /// scan transpose without reaching this rule.
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, ArrayType, V, O>,
        _input_types: &[&ArrayType],
        _output_cotangents: &[Cotangent<'transpose, ArrayType, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, O>>, ProgramError> {
        Err(ProgramError::UnsupportedOperation {
            message: "while does not support transposition (reverse-mode differentiation through staged unbounded \
                      while loops is not supported; eager differentiation unrolls the loop instead, and loops built \
                      with `with_iteration_bound` stage a transposable masked scan)"
                .to_string(),
        })
    }
}

/// JVP rule for [`WhileOperation`]: a hybrid that picks one of three strategies at rule time through
/// [`DifferentiationContext::supports_primal_concretization`] and [`WhileOperation::iteration_bound`].
///
/// **Concretizing domains unroll the loop.** When primal values are concrete, the trip count is decidable at rule
/// time: the rule drives the loop on the carried primal state, evaluating the condition through
/// [`Context::bind`](crate::contexts::Context::bind), extracting the concrete predicate via [`BooleanLike::boolean`],
/// and staging each iteration's body pushforward (one [`Linearization::at`] per iteration, with residual references
/// rebound onto the enclosing environment) into the active linear builder over the tangent state. A semantic
/// iteration bound caps the unrolling, so the rule differentiates exactly the truncated loop. The staged pushforward
/// is a straight-line linear program: replaying it with fresh tangents re-runs the captured per-iteration linear
/// maps, and — because it contains no loop structure — it *transposes*, so eager reverse-mode differentiation
/// (`vjp` / `value_and_grad`) through unbounded, data-dependent while loops works. This exceeds JAX, which traces
/// `while_loop` even under eager execution and therefore cannot reverse-differentiate through it at all. The staged
/// paths below are reached only when [`supports_primal_concretization`](
/// DifferentiationContext::supports_primal_concretization) is `false` (tracing/abstract domains).
///
/// **Loops with an iteration bound store residual stacks and stage one masked linear scan.** When the loop carries a
/// semantic iteration bound `B`, it has a static iteration budget, so the rule stores per-iteration residuals
/// instead of recomputing them — which makes the staged pushforward *transposable* and reverse mode
/// (`vjp` / `value_and_grad`) through bounded loops total:
///
///   1. The body is linearized *symbolically* once at the loop state types via [`linearize_program`](
///      crate::tracing_v2::linearize_program), exactly like the unbounded staged path.
///   2. An *augmented* primal while is bound over the state `[original_state..., counter (i64 scalar, starting at
///      zero), residual_stacks (one zero-initialized `[B, …]` stack per residual), mask_stack (a false-initialized
///      Boolean `[B]` stack)]` (see [`build_bounded_while_programs`]): each iteration runs the residual-extended
///      body, writes its residuals into the stacks at lane `counter`, marks lane `counter` valid in the mask stack,
///      and increments the counter. The augmented while keeps `iteration_bound = B`, so the writes can never clamp,
///      and lanes at or beyond the actual trip count keep their initial zero/false values.
///   3. The bound stacks (and the mask stack) are registered in the enclosing linearization residual environment,
///      and one linear scan ([`SupportsLinearScan`]) of length `B` is staged over the materialized state tangents:
///      its body applies the residualized body pushforward (whose scan-local residual references resolve to the
///      per-lane stack slices) and wraps each tangent output in a captured-condition select
///      ([`SupportsLinearSelect`]) over the mask lane, choosing the pushforward output on valid lanes and the
///      *carried tangent input* on lanes beyond the actual trip count — those lanes pass tangents through unchanged.
///   4. The scan transposes totally (transposed body + flipped direction + the same stacks), and the transposed
///      selects route cotangents correctly for inactive lanes too: the masked pushforward side receives a zero
///      cotangent (which the linear pushforward transpose maps to zero) while the carried side receives the full
///      cotangent, so cotangents also pass through inactive lanes unchanged. Reverse mode through a staged bounded
///      while therefore composes with no while-specific transpose code.
///
/// **Unbounded loops stage one doubled-state fused linear loop.** Without a bound no statically shaped residual
/// stack exists, so the rule stages loop structure that recomputes instead, with full JAX
/// [`while_loop`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html) parity:
///
///   1. The body is linearized *symbolically* once at the loop state types via [`linearize_program`](
///      crate::tracing_v2::linearize_program) — no primal state values are involved and no iteration runs
///      here.
///   2. The primal [`WhileOperation`] is bound *unchanged* (original condition and body) in the primal domain,
///      recording one `while` operation.
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
/// Transposition of the unbounded *staged* loop is not supported: [`WhileOperation`]'s transpose rule errors,
/// because the fused linear loop recomputes primal state *forward* through the iterations, and transposing it would
/// have to run that recomputation backwards, which a while loop cannot express. Reverse mode through a *staged* while
/// loop therefore requires an iteration bound (the masked-scan pushforward above transposes totally). Concretizing
/// domains are unaffected: their pushforwards are straight-line and contain no loop to transpose, so eager reverse
/// mode works even for unbounded loops.
impl<V, D, O> DifferentiableOperation<D> for WhileOperation<V, O, ArrayType>
where
    V: Value<ArrayType>,
    D: DifferentiationContext<Type = ArrayType, Constant = V> + Domain<Operation = O>,
    D::Value: BooleanLike,
    O: Clone
        + Operation<ArrayType>
        + From<WhileOperation<V, O, ArrayType>>
        + DifferentiableOperation<D>
        + ProgramLinearizableOperation<D>
        + SupportsZero<ArrayType>
        + SupportsOne<ArrayType>
        + SupportsAdd<ArrayType>
        + SupportsBroadcast<ArrayType>
        + SupportsDynamicUpdateSlice<ArrayType>,
    LinearOperationOf<D>: ResidualizedOperation<D>
        + SupportsLinearWhile<ArrayType, D::Tangent, ResidualFactor<ArrayType, D::Value>, O>
        + SupportsLinearScan<ArrayType, D::Tangent, ResidualFactor<ArrayType, D::Value>>
        + SupportsLinearSelect<ArrayType, ResidualFactor<ArrayType, D::Value>>,
    Vec<V>: Parameterized<
            V,
            Family: ParameterizedFamily<D::Value> + ParameterizedFamily<D::Tangent>,
            To<V> = Vec<V>,
            To<D::Value> = Vec<D::Value>,
            To<D::Tangent> = Vec<D::Tangent>,
            ParameterStructure: Debug + PartialEq,
        >,
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

        // Linearize the body symbolically once at the loop state types. The eager-unroll path below evaluates this
        // linearization's primal side at each iteration's concrete state and stages the per-iteration pushforward,
        // while the staged paths embed its programs into linear loop structure.
        let linearization = self.body().linearize(context.differentiable())?;

        // Eager-unroll path (concretizing domains): drive the loop on the concrete primal state, evaluating the
        // condition and the body's primal side per iteration, and stage each iteration's body pushforward into the
        // active linear builder so the accumulated tangent program stays straight-line (and therefore transposable).
        if context.differentiable().supports_primal_concretization() {
            let mut state_primals = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
            // Materialize symbolic-zero state tangents into concrete tangent atoms at loop entry so each body
            // pushforward can be staged over them.
            let mut state_tangents = inputs
                .iter()
                .map(|input| context.materialize_tangent(input.tangent().clone()))
                .collect::<Result<Vec<_>, _>>()?;
            let mut completed_iterations = 0;
            loop {
                // A semantic iteration bound truncates the unrolled loop even while the condition still produces
                // true, so the staged pushforward differentiates exactly the truncated loop.
                let truncated = self.iteration_bound().is_some_and(|bound| completed_iterations >= bound);
                let exhausted = truncated || {
                    let condition_outputs = self.condition().interpret_with(
                        state_primals.clone(),
                        |_, constant| context.differentiable().lift(constant.clone()),
                        |instruction, instruction_inputs| {
                            context.differentiable().bind(instruction.operation().clone(), instruction_inputs)
                        },
                    )?;
                    check_count!("output", condition_outputs, 1, ProgramError);
                    !condition_outputs[0].boolean()?
                };
                if exhausted {
                    return Ok(state_primals
                        .into_iter()
                        .zip(state_tangents)
                        .map(|(primal, tangent)| JvpTracer::from_value(primal, tangent))
                        .collect());
                }

                // Evaluate the body's primal side at the current concrete state, recovering the next primal state and
                // a value-bound pushforward whose residual references are closed over this iteration's residuals.
                let (next_primals, pushforward) = linearization.at(context.differentiable(), state_primals.clone())?;
                check_count!("output", next_primals, state_count, ProgramError);

                // Register this iteration's residual values in the active linearization residual environment, mapping
                // each pushforward-local residual index onto the enclosing factor it resolves to.
                let residual_factors =
                    pushforward.residuals().iter().map(|residual| context.factor(residual.clone())).collect::<Vec<_>>();

                // Stage the iteration's pushforward into the active tangent builder, threading the carried tangent
                // atoms in and out. This mirrors `Pushforward::apply`, but stages each instruction (with its residual
                // references rebound onto the enclosing environment) instead of interpreting it, so the accumulated
                // tangent program is the unrolled straight-line pushforward of the loop.
                let next_tangents = pushforward.program().interpret_with(
                    state_tangents,
                    |_, tangent| context.lift(tangent.clone()),
                    |instruction, tangent_inputs| {
                        let operation = instruction.operation().try_map_factors(&mut |factor| match factor {
                            ResidualFactor::Reference { index, .. } => {
                                residual_factors.get(*index).cloned().ok_or_else(|| {
                                    ProgramError::MalformedProgram(format!(
                                        "while body pushforward references residual {index} but only {} residuals \
                                         were captured",
                                        residual_factors.len(),
                                    ))
                                })
                            }
                            ResidualFactor::Constant(value) => Ok(ResidualFactor::Constant(value.clone())),
                        })?;
                        context.stage_operation(operation, tangent_inputs)
                    },
                )?;
                check_count!("output", next_tangents, state_count, ProgramError);
                state_primals = next_primals;
                state_tangents = next_tangents;
                completed_iterations += 1;
            }
        }

        // Bounded staged path: store instead of recompute. The augmented primal while stores every per-iteration
        // residual into a preallocated `[bound, ...]` stack (plus a Boolean lane-validity mask), and the tangent
        // side becomes one masked linear scan of length `bound`, which transposes totally.
        if let Some(bound) = self.iteration_bound() {
            let state_types = self.state_types();
            let counter_type = ArrayType::scalar(DataType::I64);
            let boolean_scalar_type = ArrayType::scalar(DataType::Boolean);
            let mask_stack_type = stacked_scan_type(&boolean_scalar_type, bound);

            // Bind the augmented primal while over `[state..., counter, residual_stacks..., mask_stack]`, with the
            // counter starting at zero and the stacks (including the Boolean mask, whose zero is false) starting at
            // typed zeros staged in the primal domain.
            let (extended_condition, augmented_body, stack_types) = build_bounded_while_programs(
                self.condition(),
                &linearization.primal_program,
                linearization.residual_types.as_slice(),
                bound,
            )?;
            let augmented_while =
                WhileOperation::new(extended_condition, augmented_body)?.with_iteration_bound(bound)?;
            let mut primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
            let zero_state_types =
                std::iter::once(&counter_type).chain(stack_types.iter()).chain(std::iter::once(&mask_stack_type));
            for zero_state_type in zero_state_types {
                let mut zeros = context.bind_primal(O::zero_operation(zero_state_type.clone()), &[])?;
                check_count!("output", zeros, 1, ProgramError);
                primal_inputs.push(zeros.remove(0));
            }
            let mut bound_outputs = context.bind_primal(O::from(augmented_while), primal_inputs.as_slice())?;
            check_count!("output", bound_outputs, state_count + 2 + stack_types.len(), ProgramError);
            let mask_value = bound_outputs.pop().unwrap();
            let stack_values = bound_outputs.split_off(state_count + 1);
            // Drop the internal iteration counter output; the rule's primal outputs are the original state.
            bound_outputs.truncate(state_count);
            let primal_outputs = bound_outputs;

            // Register the bound residual stacks in the enclosing residual environment (scan-local indices align
            // with the pushforward's references by construction) and rewrite closed constant factors into
            // lane-uniform stacks, exactly like the scan JVP rule.
            let mut stack_factors = stack_values.into_iter().map(|value| context.factor(value)).collect::<Vec<_>>();
            let pushforward_body = linearization.pushforward_program.map_operations(|operation| {
                operation.try_map_factors(&mut |factor| match factor {
                    ResidualFactor::Reference { index, r#type } => {
                        Ok(ResidualFactor::Reference { index: *index, r#type: r#type.clone() })
                    }
                    ResidualFactor::Constant(value) => {
                        let value_type = value.r#type().into_owned();
                        if value_type.static_shape().is_none() {
                            return Err(TypeError {
                                message: format!(
                                    "while body pushforwards cannot capture a constant factor of dynamically sized \
                                     type {value_type}",
                                ),
                            }
                            .into());
                        }
                        let stacked_type = stacked_scan_type(&value_type, bound);
                        let output_axes = (1..=value_type.rank()).collect::<Vec<_>>();
                        let mut broadcasted = context.bind_primal(
                            O::broadcast_operation(stacked_type, output_axes),
                            std::slice::from_ref(value),
                        )?;
                        check_count!("output", broadcasted, 1, ProgramError);
                        let scan_local_index = stack_factors.len();
                        stack_factors.push(context.factor(broadcasted.remove(0)));
                        Ok(ResidualFactor::Reference { index: scan_local_index, r#type: value_type })
                    }
                })
            })?;

            // Register the mask stack and derive the per-state-element select conditions: scalar state elements
            // reference the Boolean `[bound]` mask stack directly (its lane type is `bool[]`), while non-scalar
            // elements reference a broadcast of the mask stack to `[bound, ...state_shape]` bound in the primal
            // domain outside the loop, because select requires shape-congruent conditions.
            let mask_index = stack_factors.len();
            stack_factors.push(context.factor(mask_value.clone()));
            let select_conditions = state_types
                .iter()
                .map(|state_type| -> Result<ResidualFactor<ArrayType, D::Value>, ProgramError> {
                    if state_type.rank() == 0 {
                        return Ok(ResidualFactor::Reference {
                            index: mask_index,
                            r#type: boolean_scalar_type.clone(),
                        });
                    }
                    let condition_type = ArrayType::new(DataType::Boolean, state_type.shape().clone());
                    let stacked_condition_type = stacked_scan_type(&condition_type, bound);
                    let mut broadcasted = context.bind_primal(
                        O::broadcast_operation(stacked_condition_type, vec![0]),
                        std::slice::from_ref(&mask_value),
                    )?;
                    check_count!("output", broadcasted, 1, ProgramError);
                    let index = stack_factors.len();
                    stack_factors.push(context.factor(broadcasted.remove(0)));
                    Ok(ResidualFactor::Reference { index, r#type: condition_type })
                })
                .collect::<Result<Vec<_>, _>>()?;

            // Wrap each pushforward output in a captured-condition select over the mask lane choosing the
            // pushforward result on valid lanes and the carried tangent input on lanes beyond the actual trip
            // count, then stage one linear scan of length `bound` over the materialized state tangents.
            let mut scan_body = pushforward_body;
            check_count!("input", scan_body.input_ids, state_count, ProgramError);
            check_count!("output", scan_body.output_ids, state_count, ProgramError);
            let masked_outputs = scan_body
                .output_ids
                .clone()
                .into_iter()
                .zip(scan_body.input_ids.clone())
                .zip(select_conditions.into_iter().zip(state_types.iter()))
                .map(|((pushforward_output, carried_input), (condition, state_type))| {
                    let select_output = AtomId::new(scan_body.atoms.len());
                    scan_body.atoms.push(Atom::Variable(state_type.clone()));
                    scan_body.instructions.push(Instruction::new(
                        LinearOperationOf::<D>::linear_select_operation(condition),
                        vec![pushforward_output, carried_input],
                        vec![select_output],
                    ));
                    select_output
                })
                .collect::<Vec<_>>();
            scan_body.output_ids = masked_outputs;
            scan_body.output_structure = vec![Placeholder; state_count];
            let tangent_operands = inputs
                .iter()
                .map(|input| context.materialize_tangent(input.tangent().clone()))
                .collect::<Result<Vec<_>, _>>()?;
            let linear_scan =
                LinearOperationOf::<D>::linear_scan_operation(scan_body, stack_factors, state_count, bound, false, 1)?;
            let tangent_outputs = context.stage_operation(linear_scan, tangent_operands.as_slice())?;
            check_count!("output", tangent_outputs, state_count, ProgramError);
            return Ok(primal_outputs
                .into_iter()
                .zip(tangent_outputs)
                .map(|(primal, tangent)| JvpTracer::from_value(primal, tangent))
                .collect());
        }

        // Unbounded staged path: build the doubled-state fused recompute programs that drive forward-mode
        // interpretation and lowering.
        let (extended_condition, fused_body) =
            build_fused_while_programs(context.differentiable(), self.condition(), &linearization)?;

        // Bind the primal while unchanged, recording one `while` operation with the original condition and body.
        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal_outputs = context.bind_primal(O::from(self.clone()), primal_inputs.as_slice())?;
        check_count!("output", primal_outputs, state_count, ProgramError);

        // Inject the loop-entry primal state into the linear program through nullary residual injections and stage
        // one linear while over the doubled state `[primal_state..., tangent_state...]`. Replaying the pushforward at
        // the same primal point therefore genuinely re-runs the loop — the trip count comes from the captured primal
        // point — and fresh tangents propagate through the same iterations.
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

/// Staged batching for [`ConditionOperation`] under tracing contexts. Primal values in a [`BatchingContext`] over a
/// staging context are always tracers, so a lane-uniform predicate can never be concretized to pick one branch the
/// way the value-level rule above does. Instead of erroring, this rule *stages batched condition structure*:
///
///   - **Lane-uniform predicate.** Both branch programs are batched at the operand lane axes via
///     [`batch_program`](crate::tracing_v2::batching::batch_program) (the batching analog of
///     [`linearize_program`](crate::tracing_v2::linearize_program)), their per-output lane axes are
///     normalized to a common layout by appending staged axis-moving operations at the branch tails when they
///     disagree (a transpose for a mismatched axis, a broadcast for a lane-uniform output paired with a batched
///     one), and one [`ConditionOperation`] over the batched branches is staged into the parent context with the
///     unbatched predicate passed through as its scalar Boolean operand. The staged trace therefore keeps one
///     `condition` operation whose branches run whole batches per lane.
///   - **Lane-varying predicate.** Both branches are interpreted over the operand inputs and merged per lane via
///     [`Select`](crate::operations::control_flow::Select), exactly like the value-level rule: every per-lane
///     primitive stages through the tracers, so the multi-operation rewrite composes under tracing already.
impl<C, O> BatchableOperation<Tracer<C>, BatchingContext<C>> for ConditionOperation<C::Constant, O, ArrayType>
where
    C: StagingContext<Type = ArrayType, Operation = O>,
    C::Constant: Value<ArrayType> + BooleanLike,
    Tracer<C>: crate::operations::control_flow::Select<Condition = Tracer<C>>,
    O: BatchableOperation<Tracer<C>, BatchingContext<C>>
        + ProgramBatchableOperation<C::Constant>
        + SupportsTranspose<ArrayType>
        + SupportsBroadcast<ArrayType>
        + From<ConditionOperation<C::Constant, O, ArrayType>>,
{
    fn batch(
        &self,
        context: &BatchingContext<C>,
        inputs: &[ArrayBatch<Tracer<C>>],
    ) -> Result<Vec<ArrayBatch<Tracer<C>>>, ProgramError> {
        let Some((predicate_batch, operand_inputs)) = inputs.split_first() else {
            return Err(BatchingError::UnsupportedOperation {
                message: "cannot batch a condition operation with no predicate input".to_string(),
            }
            .into());
        };
        if predicate_batch.batch_axis().is_some() {
            // Lane-varying predicate: interpret both branches and merge their outputs per lane via `Select`.
            return batch_condition_with_interpreter(
                self.true_branch(),
                self.false_branch(),
                inputs,
                |program, program_inputs| context.interpret_program(program, program_inputs),
            );
        }

        // Lane-uniform (abstract) predicate: batch both branches at the operand lane axes and normalize their
        // output axes to a common layout, preferring the true branch's natural axis when both are batched.
        let axis_size = context.axis_size();
        let operand_axes = operand_inputs.iter().map(|input| input.batch_axis()).collect::<Vec<_>>();
        let (mut batched_true_branch, true_axes) =
            self.true_branch().batched(axis_size, operand_axes.as_slice(), ProgramBatchingOutputAxes::Natural)?;
        let (mut batched_false_branch, false_axes) =
            self.false_branch()
                .batched(axis_size, operand_axes.as_slice(), ProgramBatchingOutputAxes::Natural)?;
        check_count!("output", false_axes, true_axes.len(), ProgramError);
        let mut output_axes = Vec::with_capacity(true_axes.len());
        for (output_index, (true_axis, false_axis)) in true_axes.iter().zip(false_axes.iter()).enumerate() {
            let target_axis = match (true_axis, false_axis) {
                (None, None) => {
                    output_axes.push(None);
                    continue;
                }
                (Some(axis), _) | (None, Some(axis)) => *axis,
            };
            normalize_batched_program_output_axis(
                &mut batched_true_branch,
                output_index,
                *true_axis,
                target_axis,
                axis_size,
            )?;
            normalize_batched_program_output_axis(
                &mut batched_false_branch,
                output_index,
                *false_axis,
                target_axis,
                axis_size,
            )?;
            output_axes.push(Some(target_axis));
        }

        // Stage one condition over the batched branches with the unbatched predicate passed through.
        let batched_condition = ConditionOperation::new(batched_true_branch, batched_false_branch)?;
        let mut staged_inputs = Vec::with_capacity(inputs.len());
        staged_inputs.push(predicate_batch.value().clone());
        staged_inputs.extend(operand_inputs.iter().map(|input| input.value().clone()));
        let outputs = context.parent_context().stage_operation(O::from(batched_condition), staged_inputs.as_slice())?;
        check_count!("output", outputs, output_axes.len(), ProgramError);
        outputs
            .into_iter()
            .zip(output_axes)
            .map(|(output, axis)| {
                let physical_type = output.r#type().into_owned();
                ArrayBatch::new(physical_type, output, axis)
            })
            .collect()
    }
}

pub(crate) fn batch_while_with_interpreter<VOperation, V, O, F>(
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
    // and runs the body until no lane is still active. Both loops respect the semantic
    // iteration bound: a bounded while runs at most `bound` body applications by definition.
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
            // One body application already ran above, so the loop helper receives the remaining budget.
            while_operation.iteration_bound().map(|bound| bound - 1),
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
        while_operation.iteration_bound(),
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

/// Staged batching for [`WhileOperation`] under tracing contexts. Primal values in a [`BatchingContext`] over a
/// staging context are always tracers, so the loop cannot be driven operationally the way the value-level rule above
/// drives it (per-iteration predicate extraction would concretize tracers). Instead, this rule *stages batched loop
/// structure*:
///
///   1. Every batched state input is realigned to lane axis `0` in the parent context, and the body is batched at
///      the state lane axes via [`batch_program`](crate::tracing_v2::batching::batch_program),
///      iterating the axes to a fixed point: a while loop's state types are loop-invariant, so a lane-uniform state
///      element whose update depends on a batched element *becomes* batched, and the rule widens that element's
///      input axis and re-batches until the body is axis-invariant (the iteration count is bounded by the state
///      count because every non-final pass widens at least one element). Body outputs that remain lane-uniform or
///      land on a different axis than their loop-invariant input are normalized by staged axis-moving operations at
///      the batched body's tail, and widened parent inputs gain their lane axis through staged broadcasts.
///   2. The condition is batched at the stabilized axes. When its predicate output stays *lane-uniform*, one
///      [`WhileOperation`] over the batched condition and body is staged directly, preserving any semantic
///      [`iteration_bound`](WhileOperation::with_iteration_bound) (so bounded loops stay reverse-capable under
///      `batch`).
///   3. When the predicate output is *batched* (per-lane termination), every state element is widened to a batched
///      element and the masked loop the value-level rule runs operationally is built as program data over the
///      augmented state `[state..., active_mask]`: the staged condition reduces the mask with a lane-axis `any`,
///      and the staged body applies the batched body, selects per state element between the candidate update and
///      the carried state under the (broadcast) mask, recomputes the per-lane predicate on the new state, and ANDs
///      it into the mask. The initial mask is the batched condition staged once over the initial state in the
///      parent context, and the iteration bound is preserved (lanes share masked iterations, so capping the staged
///      loop matches per-lane truncation exactly, like the operational rule).
impl<C, O> BatchableOperation<Tracer<C>, BatchingContext<C>> for WhileOperation<C::Constant, O, ArrayType>
where
    C: StagingContext<Type = ArrayType, Operation = O>,
    C::Constant: Value<ArrayType> + BooleanLike,
    Tracer<C>: Broadcast<Output = Tracer<C>> + Transpose,
    O: Clone
        + ProgramBatchableOperation<C::Constant>
        + SupportsTranspose<ArrayType>
        + SupportsBroadcast<ArrayType>
        + SupportsReduce<ArrayType>
        + SupportsSelect<ArrayType>
        + SupportsAnd<ArrayType>
        + From<WhileOperation<C::Constant, O, ArrayType>>,
{
    fn batch(
        &self,
        context: &BatchingContext<C>,
        inputs: &[ArrayBatch<Tracer<C>>],
    ) -> Result<Vec<ArrayBatch<Tracer<C>>>, ProgramError> {
        let state_count = self.state_types().len();
        check_count!("input", inputs, state_count, ProgramError);
        let axis_size = context.axis_size();

        // Realign every batched state input to lane axis 0 in the parent context, so the loop-invariance fixed
        // point below only ever distinguishes lane-uniform (`None`) from batched-at-0 (`Some(0)`) state elements.
        let mut state = inputs.iter().map(|input| align_batch_axis(input, 0)).collect::<Result<Vec<_>, _>>()?;
        let mut state_axes = state.iter().map(|input| input.batch_axis()).collect::<Vec<_>>();

        // Iterate the body's batch axes to a fixed point: a lane-uniform state element whose update is batched
        // becomes batched. Every non-final pass widens at least one of the `state_count` elements, so the loop
        // stabilizes within `state_count + 1` passes by construction; the trailing error guards the contract that
        // separately implemented batching rules report widening monotonically.
        let mut batched_body = None;
        for _ in 0..=state_count {
            let (candidate_body, body_axes) =
                self.body().batched(axis_size, state_axes.as_slice(), ProgramBatchingOutputAxes::Natural)?;
            check_count!("output", body_axes, state_count, ProgramError);
            let mut widened = false;
            for (state_axis, body_axis) in state_axes.iter_mut().zip(body_axes.iter()) {
                if state_axis.is_none() && body_axis.is_some() {
                    *state_axis = Some(0);
                    widened = true;
                }
            }
            if !widened {
                batched_body = Some((candidate_body, body_axes));
                break;
            }
        }
        let Some((mut batched_body, mut body_axes)) = batched_body else {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "while loop batching failed to stabilize the loop state batch axes within {state_count} \
                     widening passes",
                ),
            }
            .into());
        };

        // Batch the condition at the stabilized axes; a batched predicate output means per-lane termination, in
        // which case every state element participates in per-lane masking and is therefore widened to a batched
        // element before the masked loop structure is built.
        let (mut batched_condition, mut condition_axes) =
            self.condition().batched(axis_size, state_axes.as_slice(), ProgramBatchingOutputAxes::Natural)?;
        check_count!("output", condition_axes, 1, ProgramError);
        let lane_varying = condition_axes[0].is_some();
        if lane_varying && state_axes.iter().any(Option::is_none) {
            state_axes = vec![Some(0); state_count];
            (batched_body, body_axes) =
                self.body().batched(axis_size, state_axes.as_slice(), ProgramBatchingOutputAxes::Natural)?;
            check_count!("output", body_axes, state_count, ProgramError);
            (batched_condition, condition_axes) =
                self.condition().batched(axis_size, state_axes.as_slice(), ProgramBatchingOutputAxes::Natural)?;
            check_count!("output", condition_axes, 1, ProgramError);
        }

        // Normalize the batched body's output axes to the loop-invariant input axes and widen the parent state
        // values whose elements became batched (their lane axis is materialized through a staged broadcast).
        for (output_index, (state_axis, body_axis)) in state_axes.iter().zip(body_axes.iter()).enumerate() {
            if let Some(target_axis) = state_axis {
                normalize_batched_program_output_axis(
                    &mut batched_body,
                    output_index,
                    *body_axis,
                    *target_axis,
                    axis_size,
                )?;
            }
        }
        for (element, state_axis) in state.iter_mut().zip(state_axes.iter()) {
            if state_axis.is_some() && element.batch_axis().is_none() {
                *element = broadcast_to_batched(element, 0, axis_size)?;
            }
        }
        let state_values = state.iter().map(|element| element.value().clone()).collect::<Vec<_>>();

        // Lane-uniform predicate: stage one while over the batched condition and body directly.
        if !lane_varying {
            let batched_while =
                WhileOperation::new(batched_condition, batched_body)?.with_iteration_bound(self.iteration_bound())?;
            let outputs = context.parent_context().stage_operation(O::from(batched_while), state_values.as_slice())?;
            check_count!("output", outputs, state_count, ProgramError);
            return outputs
                .into_iter()
                .zip(state_axes)
                .map(|(output, axis)| {
                    let physical_type = output.r#type().into_owned();
                    ArrayBatch::new(physical_type, output, axis)
                })
                .collect();
        }

        // Lane-varying predicate: build the masked loop as program data over `[state..., active_mask]`.
        let Some(predicate_axis) = condition_axes[0] else {
            return Err(BatchingError::UnsupportedOperation {
                message: "while loop condition produced a lane-uniform predicate after widening the loop state to \
                          batched elements; this is not supported"
                    .to_string(),
            }
            .into());
        };
        let batched_state_types = batched_body.input_types();
        let mask_type = batched_condition.output_types()[0].clone();

        // Staged condition: `any(active_mask)` along the lane axis, ignoring the state inputs.
        let mut condition_builder = ProgramBuilder::<ArrayType, C::Constant, O>::new();
        for state_type in &batched_state_types {
            condition_builder.add_input(state_type.clone());
        }
        let condition_mask_input = condition_builder.add_input(mask_type.clone());
        let any_active = condition_builder.add_instruction(
            O::reduce_operation(vec![predicate_axis], ReductionKind::Any, None),
            vec![condition_mask_input],
        )?[0];
        let masked_condition =
            condition_builder.build(vec![any_active], vec![Placeholder; state_count + 1], vec![Placeholder])?;

        // Staged body: candidate updates from the batched body, per-element masked selection, the per-lane
        // predicate recomputed on the new state, and the mask narrowed via AND.
        let mut body_builder = ProgramBuilder::<ArrayType, C::Constant, O>::new();
        let body_state_inputs = batched_state_types
            .iter()
            .map(|state_type| body_builder.add_input(state_type.clone()))
            .collect::<Vec<_>>();
        let body_mask_input = body_builder.add_input(mask_type.clone());
        let candidates = inline_program_into_builder(&mut body_builder, &batched_body, body_state_inputs.as_slice())?;
        check_count!("output", candidates, state_count, ProgramError);
        let mut next_state = Vec::with_capacity(state_count);
        for ((candidate, carried_input), state_type) in
            candidates.into_iter().zip(body_state_inputs.iter()).zip(batched_state_types.iter())
        {
            let element_mask_type = ArrayType::new(DataType::Boolean, state_type.shape().clone());
            let element_mask = if element_mask_type == mask_type {
                body_mask_input
            } else {
                body_builder.add_instruction(
                    O::broadcast_operation(element_mask_type, vec![predicate_axis]),
                    vec![body_mask_input],
                )?[0]
            };
            let selected =
                body_builder.add_instruction(O::select_operation(), vec![element_mask, candidate, *carried_input])?[0];
            next_state.push(selected);
        }
        let next_predicate = inline_program_into_builder(&mut body_builder, &batched_condition, next_state.as_slice())?;
        check_count!("output", next_predicate, 1, ProgramError);
        let next_mask = body_builder.add_instruction(O::and_operation(), vec![body_mask_input, next_predicate[0]])?[0];
        let mut body_outputs = next_state;
        body_outputs.push(next_mask);
        let masked_body =
            body_builder.build(body_outputs, vec![Placeholder; state_count + 1], vec![Placeholder; state_count + 1])?;

        // The initial mask is the batched condition applied to the initial state, staged in the parent context.
        let initial_mask_outputs = context.parent_context().stage_program(&batched_condition, state_values.clone())?;
        check_count!("output", initial_mask_outputs, 1, ProgramError);
        let mut staged_inputs = state_values;
        staged_inputs.extend(initial_mask_outputs);

        let masked_while =
            WhileOperation::new(masked_condition, masked_body)?.with_iteration_bound(self.iteration_bound())?;
        let mut outputs = context.parent_context().stage_operation(O::from(masked_while), staged_inputs.as_slice())?;
        check_count!("output", outputs, state_count + 1, ProgramError);
        // Drop the internal mask output; the rule's outputs are the original state elements, all batched at 0.
        outputs.truncate(state_count);
        outputs
            .into_iter()
            .map(|output| {
                let physical_type = output.r#type().into_owned();
                ArrayBatch::new(physical_type, output, Some(0))
            })
            .collect()
    }
}

/// Eager loop that drives a [`WhileOperation`] whose condition program produces a lane-uniform
/// scalar Boolean predicate. Each iteration runs the body when the predicate is `true` and exits
/// when it becomes `false` or once the remaining iteration budget (the semantic iteration bound
/// minus any body applications the caller already performed) is exhausted. This is the original
/// simple loop preserved for the lane-uniform case.
fn run_lane_uniform_while_loop<VOperation, V, O, F>(
    condition: &Program<ArrayType, VOperation, O, Vec<VOperation>, Vec<VOperation>>,
    body: &Program<ArrayType, VOperation, O, Vec<VOperation>, Vec<VOperation>>,
    mut state: Vec<ArrayBatch<V>>,
    mut remaining_iterations: Option<usize>,
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
        if remaining_iterations == Some(0) {
            return Ok(state);
        }
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
        remaining_iterations = remaining_iterations.map(|remaining| remaining - 1);
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
    iteration_bound: Option<usize>,
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
    let mut remaining_iterations = iteration_bound;
    loop {
        // The semantic iteration bound applies per lane, and every lane shares the same masked iterations, so
        // capping the shared loop at `bound` body applications matches the per-lane truncation semantics exactly.
        if remaining_iterations == Some(0) || !lane_varying_any_active(&active_mask, predicate_axis)? {
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
        remaining_iterations = remaining_iterations.map(|remaining| remaining - 1);
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
    use crate::differentiation::Tangent;
    use crate::domains::Domain;
    use crate::operations::InterpretableOperation;
    use crate::operations::arithmetic::{
        ADD_OPERATION_NAME, SUB_OPERATION_NAME, Scale, SupportsAdd, SupportsNeg, SupportsScale,
    };
    use crate::operations::constants::{One, OneLike, SupportsZero, Zero, ZeroLike};
    use crate::parameters::{Parameter, Placeholder};
    use crate::programs::{Program, ProgramBuilder, Value};
    use crate::tracing_v2::{ArrayOperation, FactorParameterizedOperation};
    use crate::types::TypeError;
    use crate::types::{DataType, Shape, Size};

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

        /// Captured-condition select required by the bounded staged while path's trait bounds. Like `Condition`,
        /// interpretation only supports closed [`ResidualFactor::Constant`] conditions because this test enum is
        /// factor-invariant; the `TestValue`-based tests only build unbounded loops, so the bounded path never
        /// stages it.
        Select {
            condition: ResidualFactor<ArrayType, TestValue>,
        },
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
                Self::Select { .. } => "linear_select",
            }
        }

        fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
            match self {
                Self::Add | Self::Select { .. } => {
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
                Self::Select { condition } => {
                    let ResidualFactor::Constant(condition) = condition else {
                        return Err(ProgramError::UnsupportedOperation {
                            message: "the test linear select only interprets closed constant conditions".to_string(),
                        });
                    };
                    Ok(vec![if condition.boolean()? { inputs[0].clone() } else { inputs[1].clone() }])
                }
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
                Self::Condition { .. }
                | Self::Residual { .. }
                | Self::Recompute(_)
                | Self::While(_)
                | Self::Select { .. } => Err(ProgramError::UnsupportedOperation {
                    message: format!("the test linear operation {} does not support transposition", self.name()),
                }),
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

    impl SupportsLinearSelect<ArrayType, ResidualFactor<ArrayType, TestValue>> for TestLinearOperation {
        fn linear_select_operation(condition: ResidualFactor<ArrayType, TestValue>) -> Self {
            Self::Select { condition }
        }
    }

    impl SupportsLinearScan<ArrayType, TestValue, ResidualFactor<ArrayType, TestValue>> for TestLinearOperation {
        /// Rejects linear-scan construction with a precise error. The bounded staged while path is the only caller,
        /// and scalar [`TestValue`]s cannot represent the stacked `[bound, …]` residuals it requires, so the tests
        /// below only build unbounded loops.
        fn linear_scan_operation(
            _body: Program<ArrayType, TestValue, Self, Vec<TestValue>, Vec<TestValue>>,
            _residual_stacks: Vec<ResidualFactor<ArrayType, TestValue>>,
            _carry_count: usize,
            _length: usize,
            _reverse: bool,
            _unroll: usize,
        ) -> Result<Self, ProgramError> {
            Err(ProgramError::UnsupportedOperation {
                message: "the test linear operation does not support linear scans".to_string(),
            })
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

    /// Test primal operation enum. The `One`, `Add`, `Broadcast`, and `DynamicUpdateSlice` variants exist only to
    /// satisfy the bounded staged while path's trait bounds on the [`WhileOperation`] JVP rule; the tests below only
    /// build unbounded loops over scalar [`TestValue`]s, so the array-shaped variants are never staged and reject
    /// interpretation precisely.
    #[derive(Clone, Debug)]
    enum TestDifferentiableOperation {
        Zero(ArrayType),
        One(ArrayType),
        Add,
        IsPositive,
        SubtractOne,
        Scale { factor: TestValue },
        Broadcast { output_type: ArrayType },
        DynamicUpdateSlice,
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
                Self::One(_) => "one",
                Self::Add => ADD_OPERATION_NAME,
                Self::IsPositive => "is_positive",
                Self::SubtractOne => "subtract_one",
                Self::Scale { .. } => "scale",
                Self::Broadcast { .. } => "broadcast",
                Self::DynamicUpdateSlice => "dynamic_update_slice",
                Self::Condition(condition) => condition.name(),
                Self::While(while_operation) => while_operation.name(),
            }
        }

        fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
            match self {
                Self::Zero(value_type) | Self::One(value_type) => {
                    check_count!("input", input_types, 0, TypeError);
                    Ok(vec![value_type.clone()])
                }
                Self::Add => {
                    check_count!("input", input_types, 2, TypeError);
                    check_types!(self.name(), &input_types[..1], &input_types[1..]);
                    Ok(vec![input_types[0].clone()])
                }
                Self::IsPositive => {
                    check_count!("input", input_types, 1, TypeError);
                    Ok(vec![ArrayType::scalar(DataType::Boolean)])
                }
                Self::SubtractOne | Self::Scale { .. } => {
                    check_count!("input", input_types, 1, TypeError);
                    Ok(vec![input_types[0].clone()])
                }
                Self::Broadcast { output_type } => {
                    check_count!("input", input_types, 1, TypeError);
                    Ok(vec![output_type.clone()])
                }
                Self::DynamicUpdateSlice => {
                    if input_types.len() < 2 {
                        return Err(TypeError {
                            message: format!(
                                "dynamic_update_slice expected at least 2 inputs but got {}",
                                input_types.len(),
                            ),
                        });
                    }
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
                Self::One(value_type) => {
                    check_count!("input", inputs, 0, ProgramError);
                    Ok(vec![TestValue::one(value_type)?])
                }
                Self::Add => match (&inputs[0], &inputs[1]) {
                    (TestValue::Number(left), TestValue::Number(right)) => Ok(vec![TestValue::Number(left + right)]),
                    _ => Err(TypeError { message: ("add expected numeric inputs").into() }.into()),
                },
                Self::Broadcast { .. } | Self::DynamicUpdateSlice => Err(ProgramError::UnsupportedOperation {
                    message: format!("the scalar test value cannot interpret {}", self.name()),
                }),
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

    impl SupportsOne<ArrayType> for TestDifferentiableOperation {
        fn one_operation(r#type: ArrayType) -> Self {
            Self::One(r#type)
        }
    }

    impl SupportsAdd<ArrayType> for TestDifferentiableOperation {
        fn add_operation() -> Self {
            Self::Add
        }
    }

    impl SupportsBroadcast<ArrayType> for TestDifferentiableOperation {
        fn broadcast_operation(output_type: ArrayType, _output_axes: Vec<usize>) -> Self {
            Self::Broadcast { output_type }
        }
    }

    impl SupportsDynamicUpdateSlice<ArrayType> for TestDifferentiableOperation {
        fn dynamic_update_slice_operation() -> Self {
            Self::DynamicUpdateSlice
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
    /// [`LinearizationContextOf`](crate::tracing_v2::differentiation::LinearizationContextOf) (whose
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
            LinearOperationOf<D>: SupportsZero<ArrayType>,
        {
            match self {
                Self::Zero(value_type) => {
                    check_count!("input", inputs, 0, ProgramError);
                    let mut primals = context.bind_primal(Self::Zero(value_type.clone()), &[])?;
                    check_count!("output", primals, 1, ProgramError);
                    Ok(vec![JvpTracer::from_zero_tangent(primals.pop().unwrap(), value_type.clone())])
                }
                Self::One(_)
                | Self::Add
                | Self::IsPositive
                | Self::Broadcast { .. }
                | Self::DynamicUpdateSlice
                | Self::Condition(_)
                | Self::While(_) => {
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

    impl ProgramLinearizableOperation<TestDomain> for TestDifferentiableOperation {
        fn linearize_program(
            differentiable: &TestDomain,
            program: &Program<ArrayType, TestValue, Self, Vec<TestValue>, Vec<TestValue>>,
        ) -> Result<NestedLinearization<TestDomain, Self>, ProgramError> {
            crate::tracing_v2::differentiation::linearize_program(differentiable, program)
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
        let condition = ConditionOperation::new(add_one_branch(), subtract_one_branch()).unwrap();

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
        let condition = ConditionOperation::new(add_one_branch(), subtract_one_branch()).unwrap();
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

        assert!(ConditionOperation::new(add_one_branch(), bool_branch).is_err());
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
        let condition = ConditionOperation::new(identity_array_branch(), identity_array_branch()).unwrap();
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
        let condition = ConditionOperation::new(custom_scale_branch(2.0), custom_scale_branch(3.0)).unwrap();
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
        let condition = ConditionOperation::new(sin_branch, scalar_scale_branch(3.0)).unwrap();

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

    #[test]
    fn test_while_jvp_defactorizes_state_dependent_products_under_abstract_tracing() {
        // Under abstract tracing the while JVP rule takes the staged path, and a body that squares its state
        // captures that state as a loop-varying residual on both sides of the product rule. The fused doubled-state
        // loop must rewrite those references into operand form (`defactorize`), which shows up as recomputed `mul`
        // instructions inside the staged linear while body.
        use std::convert::Infallible;

        use crate::operations::compare::ComparisonDirection;
        use crate::tests::{TestArray, TestArrayDomain};
        use crate::tracing::{DomainTracer, TracingContext};
        use crate::tracing_v2::LinearArrayOperation;

        type TestArrayOp = ArrayOperation<TestArray, ArrayType>;

        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<ArrayType, TestArray, TestArrayOp>::new();
        let condition_counter = condition_builder.add_input(scalar_f64.clone());
        let _condition_value = condition_builder.add_input(scalar_f64.clone());
        let condition_zero =
            condition_builder.add_instruction(TestArrayOp::ZeroLike, vec![condition_counter]).unwrap()[0];
        let predicate = condition_builder
            .add_instruction(
                TestArrayOp::Compare { direction: ComparisonDirection::GreaterThan },
                vec![condition_counter, condition_zero],
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![predicate], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<ArrayType, TestArray, TestArrayOp>::new();
        let body_counter = body_builder.add_input(scalar_f64.clone());
        let body_value = body_builder.add_input(scalar_f64.clone());
        let one = body_builder.add_instruction(TestArrayOp::OneLike, vec![body_counter]).unwrap()[0];
        let next_counter = body_builder.add_instruction(TestArrayOp::Sub, vec![body_counter, one]).unwrap()[0];
        let squared = body_builder.add_instruction(TestArrayOp::Mul, vec![body_value, body_value]).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(
                vec![next_counter, squared],
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let while_operation = WhileOperation::<TestArray, TestArrayOp, ArrayType>::new(condition, body).unwrap();

        let outer_builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, TestArray, TestArrayOp>::new()));
        let counter_input = outer_builder.borrow_mut().add_input(scalar_f64.clone());
        let value_input = outer_builder.borrow_mut().add_input(scalar_f64.clone());
        let outer_context = TracingContext::new(&TestArrayDomain, outer_builder.clone());
        let primal_counter = outer_context.tracer(counter_input, None);
        let primal_value = outer_context.tracer(value_input, None);

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
        let counter_tangent = context.input(scalar_f64.clone());
        let value_tangent = context.input(scalar_f64);

        let outputs = while_operation
            .jvp(
                &mut context,
                &[
                    JvpTracer::from_value(primal_counter, counter_tangent),
                    JvpTracer::from_value(primal_value, value_tangent),
                ],
            )
            .expect("the while JVP rule should stage loop structure instead of concretizing the predicate");
        assert_eq!(outputs.len(), 2);

        // The linear trace gained two residual injections (the loop-entry primal state) and one doubled-state
        // linear while whose fused body carries the defactorized product rule as recomputed `mul` instructions.
        let linear_builder = linear_builder.borrow();
        assert_eq!(linear_builder.instructions().len(), 3);
        assert_eq!(linear_builder.instructions()[0].operation().name(), "residual");
        assert_eq!(linear_builder.instructions()[1].operation().name(), "residual");
        let staged_linear = linear_builder.instructions()[2].operation();
        let LinearArrayOperation::While(staged_linear_while) = staged_linear else {
            panic!("expected the staged linear operation to be a while loop");
        };
        assert!(staged_linear_while.body().to_string().contains("mul"));
    }

    #[test]
    fn test_while_transposition_rejects_staged_linear_loops() {
        // The staged doubled-state linear while recomputes primal state forward through the iterations, so its
        // transpose rule keeps rejecting reverse mode for unbounded loops; bounded loops never stage a linear
        // `while` (their tangent side is a transposable masked scan) and so never reach this rule.
        use crate::domains::AbstractDomain;

        let mut condition_builder = ProgramBuilder::<ArrayType, TestValue, TestOperation>::new();
        let condition_input = condition_builder.add_input(ArrayType::scalar(DataType::F64));
        let condition_output =
            condition_builder.add_instruction(TestOperation::IsPositive, vec![condition_input]).unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![condition_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let while_operation = WhileOperation::new(condition, subtract_one_branch()).unwrap();

        let domain = AbstractDomain::new();
        let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, TestValue, TestOperation>::new()));
        let mut context = AbstractTracingContext::new(&domain, builder);
        let result = while_operation
            .transpose(&mut context, &[&ArrayType::scalar(DataType::F64)], &[Cotangent::Zero])
            .map(|_| ());
        assert_eq!(
            result,
            Err(ProgramError::UnsupportedOperation {
                message: "while does not support transposition (reverse-mode differentiation through staged \
                          unbounded while loops is not supported; eager differentiation unrolls the loop instead, \
                          and loops built with `with_iteration_bound` stage a transposable masked scan)"
                    .to_string(),
            }),
        );
    }

    use std::collections::HashMap;
    use std::convert::Infallible;

    use crate::operations::compare::ComparisonDirection;
    use crate::operations::control_flow::ScanOperation;
    use crate::tests::{TestArray, TestArrayDomain};
    use crate::tracing::{DomainTracer, TracingContext};
    use crate::tracing_v2::{LinearArrayOperation, ResidualizedOperation};

    /// Test array operation enum used by the defactorization tests below.
    type TestArrayOperation = ArrayOperation<TestArray, ArrayType>;

    /// Abstract tangent tracer produced by jvp under abstract tracing over [`TestArrayDomain`] (the `&TestArrayDomain`
    /// borrow is promoted to `'static` because the domain is a unit struct).
    type AbstractTangentTracer = DomainTracer<'static, TestArrayDomain>;

    /// Linear operation enum staged by jvp under abstract tracing over [`TestArrayDomain`].
    type AbstractLinearOperation = LinearArrayOperation<
        AbstractTangentTracer,
        TestArray,
        ArrayType,
        Infallible,
        ResidualFactor<ArrayType, AbstractTangentTracer>,
    >;

    /// Eager interpreting domain over [`TestArray`] values that reports no support for primal concretization. Hybrid
    /// rules (in particular the while JVP rule) therefore take their staged, non-concretizing paths while every
    /// primal bind still computes concrete values, which lets the tests below interpret fused while bodies
    /// numerically without abstract tracers.
    #[derive(Copy, Clone, Debug)]
    struct StagedDispatchTestArrayDomain;

    impl Domain for StagedDispatchTestArrayDomain {
        type Type = ArrayType;
        type Value = TestArray;
        type Constant = TestArray;
        type Operation = TestArrayOperation;
    }

    impl Context for StagedDispatchTestArrayDomain {
        fn lift(&self, constant: TestArray) -> Result<TestArray, ProgramError> {
            Ok(constant)
        }

        fn bind(&self, operation: Self::Operation, inputs: &[Self::Value]) -> Result<Vec<Self::Value>, ProgramError> {
            operation.interpret(inputs)
        }
    }

    impl DifferentiationContext for StagedDispatchTestArrayDomain {
        type Tangent = TestArray;
        type LinearOperation<V: Value<ArrayType>, F: Value<ArrayType>> =
            LinearArrayOperation<V, TestArray, ArrayType, Infallible, F>;

        fn zero_tangent(&self, type_: &ArrayType) -> Result<Self::Tangent, ProgramError> {
            TestArray::zero(type_)
        }

        fn supports_primal_concretization(&self) -> bool {
            false
        }
    }

    /// Builds the `state < threshold` while condition program over one scalar state element.
    fn scalar_threshold_condition(
        threshold: f64,
    ) -> Program<ArrayType, TestArray, TestArrayOperation, Vec<TestArray>, Vec<TestArray>> {
        let mut builder = ProgramBuilder::<ArrayType, TestArray, TestArrayOperation>::new();
        let state = builder.add_input(ArrayType::scalar(DataType::F64));
        let threshold = builder.add_constant(TestArray::scalar(threshold));
        let predicate = builder
            .add_instruction(
                TestArrayOperation::Compare { direction: ComparisonDirection::LessThan },
                vec![state, threshold],
            )
            .unwrap()[0];
        builder.build(vec![predicate], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds the `while (x < 10) { x = select(x < 4, 3 * x, 2 * x) }` loop whose select condition is loop-varying.
    fn select_while_operation() -> WhileOperation<TestArray, TestArrayOperation, ArrayType> {
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut builder = ProgramBuilder::<ArrayType, TestArray, TestArrayOperation>::new();
        let state = builder.add_input(scalar_f64);
        let four = builder.add_constant(TestArray::scalar(4.0));
        let predicate = builder
            .add_instruction(
                TestArrayOperation::Compare { direction: ComparisonDirection::LessThan },
                vec![state, four],
            )
            .unwrap()[0];
        let three = builder.add_constant(TestArray::scalar(3.0));
        let tripled = builder.add_instruction(TestArrayOperation::Mul, vec![state, three]).unwrap()[0];
        let two = builder.add_constant(TestArray::scalar(2.0));
        let doubled = builder.add_instruction(TestArrayOperation::Mul, vec![state, two]).unwrap()[0];
        let next = builder.add_instruction(TestArrayOperation::Select, vec![predicate, tripled, doubled]).unwrap()[0];
        let body = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![next], vec![Placeholder], vec![Placeholder])
            .unwrap();
        WhileOperation::new(scalar_threshold_condition(10.0), body).unwrap()
    }

    /// Builds the `while (x < 100) { x = if (x < 10) { x * x } else { 2 * x } }` loop whose nested condition has a
    /// loop-varying predicate and whose true branch references the loop-varying residual `x`.
    fn condition_while_operation() -> WhileOperation<TestArray, TestArrayOperation, ArrayType> {
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut square_builder = ProgramBuilder::<ArrayType, TestArray, TestArrayOperation>::new();
        let square_input = square_builder.add_input(scalar_f64.clone());
        let squared =
            square_builder.add_instruction(TestArrayOperation::Mul, vec![square_input, square_input]).unwrap()[0];
        let square_branch = square_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![squared], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut double_builder = ProgramBuilder::<ArrayType, TestArray, TestArrayOperation>::new();
        let double_input = double_builder.add_input(scalar_f64.clone());
        let two = double_builder.add_constant(TestArray::scalar(2.0));
        let doubled = double_builder.add_instruction(TestArrayOperation::Mul, vec![double_input, two]).unwrap()[0];
        let double_branch = double_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![doubled], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let nested_condition = ConditionOperation::new(square_branch, double_branch).unwrap();

        let mut builder = ProgramBuilder::<ArrayType, TestArray, TestArrayOperation>::new();
        let state = builder.add_input(scalar_f64);
        let ten = builder.add_constant(TestArray::scalar(10.0));
        let predicate = builder
            .add_instruction(TestArrayOperation::Compare { direction: ComparisonDirection::LessThan }, vec![state, ten])
            .unwrap()[0];
        let next = builder
            .add_instruction(TestArrayOperation::Condition(Box::new(nested_condition)), vec![predicate, state])
            .unwrap()[0];
        let body = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![next], vec![Placeholder], vec![Placeholder])
            .unwrap();
        WhileOperation::new(scalar_threshold_condition(100.0), body).unwrap()
    }

    /// Builds the `while (x < threshold) { x = 2 * x }` loop with the provided semantic iteration bound.
    fn bounded_doubling_while_operation(
        threshold: f64,
        bound: usize,
    ) -> WhileOperation<TestArray, TestArrayOperation, ArrayType> {
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut builder = ProgramBuilder::<ArrayType, TestArray, TestArrayOperation>::new();
        let state = builder.add_input(scalar_f64);
        let two = builder.add_constant(TestArray::scalar(2.0));
        let doubled = builder.add_instruction(TestArrayOperation::Mul, vec![state, two]).unwrap()[0];
        let body = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![doubled], vec![Placeholder], vec![Placeholder])
            .unwrap();
        WhileOperation::new(scalar_threshold_condition(threshold), body)
            .unwrap()
            .with_iteration_bound(bound)
            .unwrap()
    }

    /// Builds the `while (x < threshold) { x = x * x }` loop with the provided semantic iteration bound. Squaring
    /// captures the loop state itself as a loop-varying residual, so differentiating this loop exercises the
    /// per-iteration residual stacks of the bounded staged path.
    fn bounded_squaring_while_operation(
        threshold: f64,
        bound: usize,
    ) -> WhileOperation<TestArray, TestArrayOperation, ArrayType> {
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut builder = ProgramBuilder::<ArrayType, TestArray, TestArrayOperation>::new();
        let state = builder.add_input(scalar_f64);
        let squared = builder.add_instruction(TestArrayOperation::Mul, vec![state, state]).unwrap()[0];
        let body = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![squared], vec![Placeholder], vec![Placeholder])
            .unwrap();
        WhileOperation::new(scalar_threshold_condition(threshold), body)
            .unwrap()
            .with_iteration_bound(bound)
            .unwrap()
    }

    /// Builds the `while (x < 50) { x = scan(cumulative product over xs = [2, 3], init = x) }` loop whose body
    /// stages a scan; the scan JVP's residual stacks reference the while-body residual environment.
    fn scan_while_operation() -> WhileOperation<TestArray, TestArrayOperation, ArrayType> {
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut scan_body_builder = ProgramBuilder::<ArrayType, TestArray, TestArrayOperation>::new();
        let carry = scan_body_builder.add_input(scalar_f64.clone());
        let x_slice = scan_body_builder.add_input(scalar_f64.clone());
        let product = scan_body_builder.add_instruction(TestArrayOperation::Mul, vec![carry, x_slice]).unwrap()[0];
        let scan_body = scan_body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![product], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        let scan = ScanOperation::new(scan_body, 1, 2).unwrap();

        let mut builder = ProgramBuilder::<ArrayType, TestArray, TestArrayOperation>::new();
        let state = builder.add_input(scalar_f64);
        let stacked_inputs = builder.add_constant(TestArray::vector(vec![2.0, 3.0]));
        let next = builder
            .add_instruction(TestArrayOperation::Scan(Box::new(scan)), vec![state, stacked_inputs])
            .unwrap()[0];
        let body = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![next], vec![Placeholder], vec![Placeholder])
            .unwrap();
        WhileOperation::new(scalar_threshold_condition(50.0), body).unwrap()
    }

    /// Stages the while JVP rule under abstract tracing (tracer-valued primals) for one scalar state element and
    /// returns the staged doubled-state linear while for structural assertions.
    fn staged_linear_while_under_abstract_tracing(
        while_operation: WhileOperation<TestArray, TestArrayOperation, ArrayType>,
    ) -> WhileOperation<AbstractTangentTracer, AbstractLinearOperation, ArrayType> {
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let outer_builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, TestArray, TestArrayOperation>::new()));
        let state_input = outer_builder.borrow_mut().add_input(scalar_f64.clone());
        let outer_context = TracingContext::new(&TestArrayDomain, outer_builder.clone());
        let primal_state = outer_context.tracer(state_input, None);
        let linear_builder =
            Rc::new(RefCell::new(ProgramBuilder::<ArrayType, AbstractTangentTracer, AbstractLinearOperation>::new()));
        let mut context = TangentContext::new(&outer_context, linear_builder.clone());
        let tangent_state = context.input(scalar_f64);
        let outputs = while_operation
            .jvp(&mut context, &[JvpTracer::from_value(primal_state, tangent_state)])
            .expect("the while JVP rule should defactorize the staged fused loop instead of rejecting it");
        assert_eq!(outputs.len(), 1);
        drop(outputs);
        let linear_builder = linear_builder.borrow();
        linear_builder
            .instructions()
            .iter()
            .find_map(|instruction| match instruction.operation() {
                AbstractLinearOperation::While(operation) => Some(operation.as_ref().clone()),
                _ => None,
            })
            .expect("the linear trace should contain one staged linear while")
    }

    /// Runs the while JVP rule's staged path over concrete [`TestArray`] state through
    /// [`StagedDispatchTestArrayDomain`] and returns the concrete primal output together with the directly
    /// interpretable pushforward program over one scalar tangent input.
    fn staged_while_pushforward(
        while_operation: WhileOperation<TestArray, TestArrayOperation, ArrayType>,
        state: f64,
    ) -> (
        TestArray,
        Program<
            ArrayType,
            TestArray,
            LinearArrayOperation<TestArray, TestArray, ArrayType>,
            Vec<TestArray>,
            Vec<TestArray>,
        >,
    ) {
        let domain = StagedDispatchTestArrayDomain;
        let builder = Rc::new(RefCell::new(ProgramBuilder::<
            ArrayType,
            TestArray,
            LinearArrayOperation<TestArray, TestArray, ArrayType, Infallible, ResidualFactor<ArrayType, TestArray>>,
        >::new()));
        let residuals = Rc::new(RefCell::new(Vec::new()));
        let residual_atoms = Rc::new(RefCell::new(HashMap::new()));
        let mut context =
            TangentContext::new_with_residuals(&domain, builder.clone(), residuals.clone(), residual_atoms);
        let tangent_input = context.input(ArrayType::scalar(DataType::F64));
        let outputs = while_operation
            .jvp(&mut context, &[JvpTracer::from_value(TestArray::scalar(state), tangent_input)])
            .expect("the while JVP rule should defactorize the staged fused loop instead of rejecting it");
        assert_eq!(outputs.len(), 1);
        let primal = outputs[0].primal().clone();
        let tangent_output = expect_tangent_value(outputs[0].tangent()).atom_id().unwrap();
        drop(outputs);
        drop(context);
        let builder = Rc::try_unwrap(builder).unwrap().into_inner();
        let tangent_program = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![tangent_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let residuals = residuals.borrow();
        let tangent_program = tangent_program
            .map_operations(|operation| {
                ResidualizedOperation::<StagedDispatchTestArrayDomain>::instantiate_residuals(
                    operation,
                    residuals.as_slice(),
                )
            })
            .unwrap();
        (primal, tangent_program)
    }

    /// Runs the eager-domain jvp of `while_operation` over one scalar state element at `(state, tangent)`,
    /// interpreting the staged fused linear loop immediately.
    fn eager_while_jvp(
        while_operation: WhileOperation<TestArray, TestArrayOperation, ArrayType>,
        state: f64,
        tangent: f64,
    ) -> (TestArray, TestArray) {
        TestArrayDomain
            .jvp(
                move |x| {
                    let mut outputs = x
                        .context()
                        .stage_operation(TestArrayOperation::While(Box::new(while_operation)), &[&x])
                        .unwrap();
                    outputs.remove(0)
                },
                TestArray::scalar(state),
                TestArray::scalar(tangent),
            )
            .unwrap()
    }

    #[test]
    fn test_while_jvp_defactorizes_loop_varying_select_conditions() {
        // `while (x < 10) { x = select(x < 4, 3 * x, 2 * x) }` captures the select condition `x < 4` as a
        // loop-varying residual. Under abstract tracing the staged path must rewrite it into a recomputed
        // operand-form select inside the fused body (previously rejected with "operand-form linear select is not
        // implemented").
        let staged_while = staged_linear_while_under_abstract_tracing(select_while_operation());
        let body_rendering = staged_while.body().to_string();
        assert!(body_rendering.contains("select"), "{body_rendering}");

        // Replaying the staged pushforward at `x = 1` runs three iterations whose select predicates are true, true,
        // and false (`x` visits 1, 3, 9), so the primal output is 18 and the tangent map is `3 * 3 * 2 = 18`.
        let (primal, tangent_program) = staged_while_pushforward(select_while_operation(), 1.0);
        assert_eq!(primal.values, vec![18.0]);
        assert_eq!(tangent_program.interpret(vec![TestArray::scalar(1.0)]).unwrap()[0].values, vec![18.0]);
        assert_eq!(tangent_program.interpret(vec![TestArray::scalar(2.0)]).unwrap()[0].values, vec![36.0]);

        // The eager-domain jvp entry point produces the same value and tangent numbers.
        let (primal, tangent) = eager_while_jvp(select_while_operation(), 1.0, 1.0);
        assert_eq!(primal.values, vec![18.0]);
        assert_eq!(tangent.values, vec![18.0]);
    }

    #[test]
    fn test_while_jvp_defactorizes_loop_varying_condition_predicates() {
        // `while (x < 100) { x = if (x < 10) { x * x } else { 2 * x } }` captures the nested condition's predicate
        // as a loop-varying residual, and the true branch's product rule references the loop-varying residual `x`
        // itself. Under abstract tracing the staged path must rewrite the linear condition into operand form with
        // the predicate as operand 0 and the branch-referenced residual forwarded as a trailing operand (previously
        // rejected with "operand-form linear condition is not implemented").
        let staged_while = staged_linear_while_under_abstract_tracing(condition_while_operation());
        let contains_operand_condition = staged_while
            .body()
            .instructions()
            .iter()
            .any(|instruction| matches!(instruction.operation(), AbstractLinearOperation::OperandCondition { .. }));
        assert!(contains_operand_condition, "{}", staged_while.body());

        // Replaying the staged pushforward at `x = 2` runs five iterations whose nested predicates are true, true,
        // false, false, false (`x` visits 2, 4, 16, 32, 64), exercising both branch paths and the
        // forwarded-residual mechanics: the tangent map multiplies `2 x` per squaring iteration and `2` per
        // doubling iteration, so the primal output is 128 and `d x_5 / d x_0 = 4 * 8 * 2 * 2 * 2 = 256`.
        let (primal, tangent_program) = staged_while_pushforward(condition_while_operation(), 2.0);
        assert_eq!(primal.values, vec![128.0]);
        assert_eq!(tangent_program.interpret(vec![TestArray::scalar(1.0)]).unwrap()[0].values, vec![256.0]);
        assert_eq!(tangent_program.interpret(vec![TestArray::scalar(0.5)]).unwrap()[0].values, vec![128.0]);

        // The eager-domain jvp entry point produces the same value and tangent numbers.
        let (primal, tangent) = eager_while_jvp(condition_while_operation(), 2.0, 1.0);
        assert_eq!(primal.values, vec![128.0]);
        assert_eq!(tangent.values, vec![256.0]);
    }

    #[test]
    fn test_bounded_while_jvp_stages_augmented_while_and_masked_scan_under_abstract_tracing() {
        // Under abstract tracing a *bounded* while takes the store-instead-of-recompute path: the primal trace gains
        // one augmented `while` whose state appends an i64 iteration counter, one `[bound, ...]` stack per body
        // residual, and one Boolean `[bound]` mask stack — and whose iteration bound is preserved — while the linear
        // trace stages one masked linear `scan` of length `bound` and *no* linear `while`.
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let while_operation = bounded_doubling_while_operation(8.0, 5);
        let outer_builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, TestArray, TestArrayOperation>::new()));
        let state_input = outer_builder.borrow_mut().add_input(scalar_f64.clone());
        let outer_context = TracingContext::new(&TestArrayDomain, outer_builder.clone());
        let primal_state = outer_context.tracer(state_input, None);
        let linear_builder =
            Rc::new(RefCell::new(ProgramBuilder::<ArrayType, AbstractTangentTracer, AbstractLinearOperation>::new()));
        let mut context = TangentContext::new(&outer_context, linear_builder.clone());
        let tangent_state = context.input(scalar_f64.clone());
        let outputs = while_operation
            .jvp(&mut context, &[JvpTracer::from_value(primal_state, tangent_state)])
            .expect("the bounded while JVP rule should stage the augmented loop and the masked scan");
        assert_eq!(outputs.len(), 1);
        drop(outputs);

        // The primal trace gained one augmented while over `[state, counter, mask_stack]`: the product rule's only
        // surviving factor is the lifted constant 2 (a closed constant whose tangent side is symbolic zero needs no
        // loop-varying residual), so no residual stack joins the state and the constant is broadcast into a
        // lane-uniform stack *outside* the loop. The mask stack still threads through the loop, marking each lane
        // that actually ran.
        let outer_builder = outer_builder.borrow();
        let staged_while = outer_builder
            .instructions()
            .iter()
            .find_map(|instruction| match instruction.operation() {
                TestArrayOperation::While(staged_while) => Some(staged_while.as_ref().clone()),
                _ => None,
            })
            .expect("the primal trace should contain the augmented while");
        assert_eq!(staged_while.iteration_bound(), Some(5));
        let mask_stack_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(5)]));
        assert_eq!(
            staged_while.state_types(),
            vec![scalar_f64.clone(), ArrayType::scalar(DataType::I64), mask_stack_type],
        );
        assert!(staged_while.body().to_string().contains("dynamic_update_slice"), "{}", staged_while.body());

        // The linear trace stages exactly one masked scan (whose body wraps the pushforward in a per-lane select
        // over the mask stack) and no linear while. The scan carries two factor stacks: the broadcast lane-uniform
        // constant stack and the mask stack.
        let linear_builder = linear_builder.borrow();
        assert!(
            linear_builder
                .instructions()
                .iter()
                .all(|instruction| !matches!(instruction.operation(), AbstractLinearOperation::While(_))),
        );
        let staged_scan = linear_builder
            .instructions()
            .iter()
            .find_map(|instruction| match instruction.operation() {
                scan @ AbstractLinearOperation::Scan { .. } => Some(scan.clone()),
                _ => None,
            })
            .expect("the linear trace should contain one masked linear scan");
        let AbstractLinearOperation::Scan { body, residual_stacks, carry_count, length, reverse, unroll } =
            &staged_scan
        else {
            unreachable!("the staged operation was matched as a scan above");
        };
        assert_eq!(residual_stacks.len(), 2);
        assert_eq!(*carry_count, 1);
        assert_eq!(*length, 5);
        assert!(!reverse);
        assert_eq!(*unroll, 1);
        assert!(body.to_string().contains("select"), "{body}");
    }

    #[test]
    fn test_bounded_while_value_and_grad_computes_gradient_through_staged_masked_scan() {
        // The headline bounded-while capability: end-to-end reverse mode through a *staged* while loop.
        // `f(x) = while (x < 8, iteration_bound = 5) { x = 2 * x }` at `x = 1` runs three iterations (`x` visits 1,
        // 2, 4), so the actual trip count 3 is strictly below the bound 5 and the two trailing lanes matter: their
        // mask entries are false, so they must pass tangents through unchanged in the forward scan and cotangents
        // through unchanged in the transposed scan. Locally `f(x) = 8 x`: value 8, gradient 8.
        let while_operation = bounded_doubling_while_operation(8.0, 5);
        let (output, pullback) = StagedDispatchTestArrayDomain
            .vjp(
                move |x| {
                    let mut outputs =
                        x.context().stage_operation(TestArrayOperation::While(Box::new(while_operation)), &[&x])?;
                    Ok(outputs.remove(0))
                },
                TestArray::scalar(1.0),
            )
            .unwrap();
        assert_eq!(output.values, vec![8.0]);

        // The pullback contains the transposed (reversed) linear scan and no while loop, and every cotangent seed
        // scales the hand-computed gradient 8.
        let rendered_pullback = pullback.to_string();
        assert!(rendered_pullback.contains("scan"), "{rendered_pullback}");
        assert!(rendered_pullback.contains("reverse=true"), "{rendered_pullback}");
        assert!(!rendered_pullback.contains("while"), "{rendered_pullback}");
        assert_eq!(pullback.interpret(TestArray::scalar(1.0)).map(|cotangent| cotangent.values), Ok(vec![8.0]));
        assert_eq!(pullback.interpret(TestArray::scalar(2.0)).map(|cotangent| cotangent.values), Ok(vec![16.0]));

        // `value_and_grad` composes the same machinery end to end.
        let while_operation = bounded_doubling_while_operation(8.0, 5);
        let (value, gradient) = StagedDispatchTestArrayDomain
            .value_and_grad(
                move |x| {
                    let mut outputs = x
                        .context()
                        .stage_operation(TestArrayOperation::While(Box::new(while_operation)), &[&x])
                        .unwrap();
                    outputs.remove(0)
                },
                TestArray::scalar(1.0),
            )
            .unwrap();
        assert_eq!(value.values, vec![8.0]);
        assert_eq!(gradient.values, vec![8.0]);
    }

    #[test]
    fn test_bounded_while_value_and_grad_stores_loop_varying_residual_stacks() {
        // The store-instead-of-recompute proof: `while (x < 100, iteration_bound = 4) { x = x * x }` at `x = 2`
        // squares three times (`x` visits 2, 4, 16 → 256, trip count 3 < bound 4), and the product rule references
        // the *per-iteration* state as a loop-varying residual, so the gradient depends on the stored stack lanes
        // `[2, 4, 16, 0]` — including the zero lane beyond the trip count, which the mask must keep inert in both
        // directions. Locally `f(x) = x⁸`: value 256 and gradient `8 x⁷ = 1024`.
        let while_operation = bounded_squaring_while_operation(100.0, 4);
        let (output, pullback) = StagedDispatchTestArrayDomain
            .vjp(
                move |x| {
                    let mut outputs =
                        x.context().stage_operation(TestArrayOperation::While(Box::new(while_operation)), &[&x])?;
                    Ok(outputs.remove(0))
                },
                TestArray::scalar(2.0),
            )
            .unwrap();
        assert_eq!(output.values, vec![256.0]);
        let rendered_pullback = pullback.to_string();
        assert!(rendered_pullback.contains("reverse=true"), "{rendered_pullback}");
        assert_eq!(pullback.interpret(TestArray::scalar(1.0)).map(|cotangent| cotangent.values), Ok(vec![1024.0]));

        // The eager-domain reverse-mode entry point produces the same value and gradient numbers.
        let while_operation = bounded_squaring_while_operation(100.0, 4);
        let (value, gradient) = TestArrayDomain
            .value_and_grad(
                move |x| {
                    let mut outputs = x
                        .context()
                        .stage_operation(TestArrayOperation::While(Box::new(while_operation)), &[&x])
                        .unwrap();
                    outputs.remove(0)
                },
                TestArray::scalar(2.0),
            )
            .unwrap();
        assert_eq!(value.values, vec![256.0]);
        assert_eq!(gradient.values, vec![1024.0]);
    }

    #[test]
    fn test_bounded_while_value_and_grad_supports_vector_state() {
        // Vector-state coverage for the bounded staged path: the residual stacks gain trailing axes (written at
        // `[counter, 0]` through the staged zero index) and the per-lane select conditions come from a broadcast of
        // the Boolean `[bound]` mask stack to `[bound, 2]`, staged outside the loop. The loop
        // `while (sum(x) < 20, iteration_bound = 4) { x = x * x }` at `x = [1.5, 2]` squares twice (sums visit 3.5
        // and 6.25 before reaching 21.0625), so `f(x) = sum(x⁴)` locally: value `1.5⁴ + 2⁴ = 21.0625` and gradient
        // `4 x³ = [13.5, 32]`, with trip count 2 strictly below the bound 4.
        use crate::tracing_v2::operations::reduce::ReductionKind;

        let vector_f64 = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]));
        let mut condition_builder = ProgramBuilder::<ArrayType, TestArray, TestArrayOperation>::new();
        let condition_state = condition_builder.add_input(vector_f64.clone());
        let summed = condition_builder
            .add_instruction(
                TestArrayOperation::Reduce { axes: vec![0], kind: ReductionKind::Sum, output_sharding: None },
                vec![condition_state],
            )
            .unwrap()[0];
        let threshold = condition_builder.add_constant(TestArray::scalar(20.0));
        let predicate = condition_builder
            .add_instruction(
                TestArrayOperation::Compare { direction: ComparisonDirection::LessThan },
                vec![summed, threshold],
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<ArrayType, TestArray, TestArrayOperation>::new();
        let body_state = body_builder.add_input(vector_f64.clone());
        let squared = body_builder.add_instruction(TestArrayOperation::Mul, vec![body_state, body_state]).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![squared], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let while_operation = WhileOperation::new(condition, body).unwrap().with_iteration_bound(4).unwrap();

        let (value, gradient) = StagedDispatchTestArrayDomain
            .value_and_grad(
                move |x| {
                    let mut outputs = x
                        .context()
                        .stage_operation(TestArrayOperation::While(Box::new(while_operation)), &[&x])
                        .unwrap();
                    let state = outputs.remove(0);
                    let mut outputs = state
                        .context()
                        .stage_operation(
                            TestArrayOperation::Reduce {
                                axes: vec![0],
                                kind: ReductionKind::Sum,
                                output_sharding: None,
                            },
                            &[&state],
                        )
                        .unwrap();
                    outputs.remove(0)
                },
                TestArray::vector(vec![1.5, 2.0]),
            )
            .unwrap();
        assert_eq!(value.values, vec![21.0625]);
        assert_eq!(gradient.values, vec![13.5, 32.0]);
    }

    #[test]
    fn test_bounded_while_eager_value_and_grad_matches_staged_numbers() {
        // The eager-domain entry point differentiates the same bounded loop to identical numbers: the loop exits
        // through its condition after three iterations, well below the bound of five.
        let while_operation = bounded_doubling_while_operation(8.0, 5);
        let (value, gradient) = TestArrayDomain
            .value_and_grad(
                move |x| {
                    let mut outputs = x
                        .context()
                        .stage_operation(TestArrayOperation::While(Box::new(while_operation)), &[&x])
                        .unwrap();
                    outputs.remove(0)
                },
                TestArray::scalar(1.0),
            )
            .unwrap();
        assert_eq!(value.values, vec![8.0]);
        assert_eq!(gradient.values, vec![8.0]);
    }

    #[test]
    fn test_bounded_while_staged_pushforward_replays_with_fresh_tangents() {
        // The bounded staged pushforward replays with fresh tangents: the masked scan multiplies the carried
        // tangent by 2 on each of the three valid lanes and passes it through the two lanes beyond the trip count,
        // so the tangent map is exactly `t ↦ 8 t`.
        let (primal, tangent_program) = staged_while_pushforward(bounded_doubling_while_operation(8.0, 5), 1.0);
        assert_eq!(primal.values, vec![8.0]);
        assert!(tangent_program.to_string().contains("scan"), "{tangent_program}");
        assert_eq!(tangent_program.interpret(vec![TestArray::scalar(1.0)]).unwrap()[0].values, vec![8.0]);
        assert_eq!(tangent_program.interpret(vec![TestArray::scalar(2.0)]).unwrap()[0].values, vec![16.0]);
    }

    #[test]
    fn test_bounded_while_truncation_differentiates_consistently_across_paths() {
        // A loop whose condition never turns false truncates at the bound by definition: with bound 3 the doubling
        // loop computes `f(x) = 8 x`, so at `x = 2` the value is 16 and the gradient is 8 — identical between plain
        // interpretation, the eager-domain entry point, and the staged dispatch domain (where every mask lane is
        // true).
        let while_operation = bounded_doubling_while_operation(f64::INFINITY, 3);
        let outputs = while_operation.interpret(&[TestArray::scalar(2.0)]).unwrap();
        assert_eq!(outputs[0].values, vec![16.0]);

        let while_operation = bounded_doubling_while_operation(f64::INFINITY, 3);
        let (value, gradient) = TestArrayDomain
            .value_and_grad(
                move |x| {
                    let mut outputs = x
                        .context()
                        .stage_operation(TestArrayOperation::While(Box::new(while_operation)), &[&x])
                        .unwrap();
                    outputs.remove(0)
                },
                TestArray::scalar(2.0),
            )
            .unwrap();
        assert_eq!(value.values, vec![16.0]);
        assert_eq!(gradient.values, vec![8.0]);

        let while_operation = bounded_doubling_while_operation(f64::INFINITY, 3);
        let (value, gradient) = StagedDispatchTestArrayDomain
            .value_and_grad(
                move |x| {
                    let mut outputs = x
                        .context()
                        .stage_operation(TestArrayOperation::While(Box::new(while_operation)), &[&x])
                        .unwrap();
                    outputs.remove(0)
                },
                TestArray::scalar(2.0),
            )
            .unwrap();
        assert_eq!(value.values, vec![16.0]);
        assert_eq!(gradient.values, vec![8.0]);
    }

    #[test]
    fn test_while_jvp_defactorizes_scan_residual_stacks() {
        // `while (x < 50) { x = scan(cumulative product over xs = [2, 3], init = x) }`: the scan JVP stores its
        // per-iteration residuals as stacked outputs and stages a linear scan whose residual stacks reference the
        // enclosing while-body residual environment. Under abstract tracing the staged path must move those stacks
        // into operand position as extra scanned inputs, leaving the rewritten scan with no factor stacks
        // (previously rejected by the defactorization catch-all).
        let staged_while = staged_linear_while_under_abstract_tracing(scan_while_operation());
        let contains_operand_form_scan = staged_while.body().instructions().iter().any(|instruction| {
            matches!(
                instruction.operation(),
                AbstractLinearOperation::Scan { residual_stacks, .. } if residual_stacks.is_empty(),
            )
        });
        assert!(contains_operand_form_scan, "{}", staged_while.body());

        // Replaying the staged pushforward at `x = 1` runs three iterations (`x` visits 1, 6, 36), each multiplying
        // the tangent by `2 * 3 = 6`, so the primal output is 216 and the tangent map is `6^3 = 216`.
        let (primal, tangent_program) = staged_while_pushforward(scan_while_operation(), 1.0);
        assert_eq!(primal.values, vec![216.0]);
        assert_eq!(tangent_program.interpret(vec![TestArray::scalar(1.0)]).unwrap()[0].values, vec![216.0]);
        assert_eq!(tangent_program.interpret(vec![TestArray::scalar(2.0)]).unwrap()[0].values, vec![432.0]);

        // The eager-domain jvp entry point produces the same value and tangent numbers.
        let (primal, tangent) = eager_while_jvp(scan_while_operation(), 1.0, 1.0);
        assert_eq!(primal.values, vec![216.0]);
        assert_eq!(tangent.values, vec![216.0]);
    }

    /// Builds the per-lane countdown loop `while (x > 0) { x = x - 1 }` over one scalar state element.
    fn countdown_while_operation() -> WhileOperation<TestArray, TestArrayOperation, ArrayType> {
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<ArrayType, TestArray, TestArrayOperation>::new();
        let condition_state = condition_builder.add_input(scalar_f64.clone());
        let zero = condition_builder.add_instruction(TestArrayOperation::ZeroLike, vec![condition_state]).unwrap()[0];
        let predicate = condition_builder
            .add_instruction(
                TestArrayOperation::Compare { direction: ComparisonDirection::GreaterThan },
                vec![condition_state, zero],
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<ArrayType, TestArray, TestArrayOperation>::new();
        let body_state = body_builder.add_input(scalar_f64);
        let one = body_builder.add_instruction(TestArrayOperation::OneLike, vec![body_state]).unwrap()[0];
        let next = body_builder.add_instruction(TestArrayOperation::Sub, vec![body_state, one]).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![next], vec![Placeholder], vec![Placeholder])
            .unwrap();
        WhileOperation::new(condition, body).unwrap()
    }

    /// Stages `while_operation` over one batched lane (mapped at axis 0 with `lane_count` lanes) under tracing and
    /// returns the staged batched program for structural and numeric assertions.
    fn batch_while_under_tracing(
        while_operation: WhileOperation<TestArray, TestArrayOperation, ArrayType>,
        lane_count: usize,
    ) -> Program<ArrayType, TestArray, TestArrayOperation, TestArray, TestArray> {
        use crate::tracing_v2::batching::BatchContext;
        let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        let parent_context = TracingContext::new(&TestArrayDomain, builder.clone());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(lane_count)]));
        let input_atom = builder.borrow_mut().add_input(input_type);
        let input_tracer = parent_context.tracer(input_atom, None);
        let output = BatchContext::batch(
            &parent_context,
            |lane| {
                let mut outputs =
                    lane.context().stage_operation(TestArrayOperation::While(Box::new(while_operation)), &[&lane])?;
                Ok(outputs.remove(0))
            },
            input_tracer,
            Some(0),
            Some(0),
            None,
        )
        .unwrap();
        let output_atom = output.atom_id().unwrap();
        builder
            .borrow()
            .clone()
            .build::<TestArray, TestArray>(vec![output_atom], Placeholder, Placeholder)
            .unwrap()
    }

    #[test]
    fn test_batch_stages_masked_while_for_lane_varying_predicates_under_tracing() {
        // vmap-under-tracing of the per-lane countdown loop: the predicate `x > 0` is per lane, so the staged
        // batching rule builds the masked loop as program data — exactly one staged `while` over the augmented
        // state `[state, active_mask]` whose condition reduces the mask with a lane-axis `any` and whose body masks
        // state updates per lane — instead of unrolling (the body's single `sub` appears exactly once in the staged
        // trace). Lanes [3, 1, 2] terminate after 3, 1, and 2 iterations, and inactive lanes carry their final
        // state, matching the eager operational path lane for lane.
        let program = batch_while_under_tracing(countdown_while_operation(), 3);
        let rendered = program.to_string();
        assert_eq!(rendered.matches("= while").count(), 1, "{rendered}");
        assert!(rendered.contains("reduce_any"), "{rendered}");
        assert_eq!(rendered.matches("sub").count(), 1, "{rendered}");
        let output = program.interpret(TestArray::vector(vec![3.0, 1.0, 2.0])).unwrap();
        assert_eq!(output.values, vec![0.0, 0.0, 0.0]);

        // The semantic iteration bound is preserved on the staged masked while: every lane performs at most two
        // body applications, so lane 0 truncates at 1.0 — the numbers of the eager operational bounded path.
        let program = batch_while_under_tracing(countdown_while_operation().with_iteration_bound(2).unwrap(), 3);
        let rendered = program.to_string();
        assert!(rendered.contains("iteration_bound=2"), "{rendered}");
        let output = program.interpret(TestArray::vector(vec![3.0, 1.0, 2.0])).unwrap();
        assert_eq!(output.values, vec![1.0, 0.0, 0.0]);
    }

    /// Builds the `while (counter > 0) { (counter, value) = (counter - 1, value + value) }` loop whose predicate
    /// depends only on the counter state element.
    fn counter_doubling_while_operation() -> WhileOperation<TestArray, TestArrayOperation, ArrayType> {
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<ArrayType, TestArray, TestArrayOperation>::new();
        let condition_counter = condition_builder.add_input(scalar_f64.clone());
        condition_builder.add_input(scalar_f64.clone());
        let zero = condition_builder.add_instruction(TestArrayOperation::ZeroLike, vec![condition_counter]).unwrap()[0];
        let predicate = condition_builder
            .add_instruction(
                TestArrayOperation::Compare { direction: ComparisonDirection::GreaterThan },
                vec![condition_counter, zero],
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<ArrayType, TestArray, TestArrayOperation>::new();
        let body_counter = body_builder.add_input(scalar_f64.clone());
        let body_value = body_builder.add_input(scalar_f64);
        let one = body_builder.add_instruction(TestArrayOperation::OneLike, vec![body_counter]).unwrap()[0];
        let next_counter = body_builder.add_instruction(TestArrayOperation::Sub, vec![body_counter, one]).unwrap()[0];
        let doubled = body_builder.add_instruction(TestArrayOperation::Add, vec![body_value, body_value]).unwrap()[0];
        let body = body_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(
                vec![next_counter, doubled],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        WhileOperation::new(condition, body).unwrap()
    }

    #[test]
    fn test_batch_stages_plain_while_for_lane_uniform_predicates_under_tracing() {
        use crate::tracing_v2::batching::BatchContext;

        // vmap-under-tracing of a loop whose predicate depends only on a lane-uniform counter: the staged batching
        // rule batches the condition and body at the state lane axes and stages one plain `while` — no mask
        // machinery (`reduce_any` / per-element `select`) appears in the staged program. Two iterations double the
        // batched value twice: [1, 2, 3] -> [4, 8, 12], with the lane-uniform counter ending at 0.
        let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        let parent_context = TracingContext::new(&TestArrayDomain, builder.clone());
        let counter_atom = builder.borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let value_atom =
            builder.borrow_mut().add_input(ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)])));
        let counter_tracer = parent_context.tracer(counter_atom, None);
        let value_tracer = parent_context.tracer(value_atom, None);
        let (counter_output, value_output) = BatchContext::batch(
            &parent_context,
            |(counter, value)| {
                let while_operation = counter_doubling_while_operation();
                let mut outputs = counter
                    .context()
                    .stage_operation(TestArrayOperation::While(Box::new(while_operation)), &[&counter, &value])?;
                let value_output = outputs.remove(1);
                Ok((outputs.remove(0), value_output))
            },
            (counter_tracer, value_tracer),
            (None, Some(0)),
            (None, Some(0)),
            None,
        )
        .unwrap();
        let output_atoms = vec![counter_output.atom_id().unwrap(), value_output.atom_id().unwrap()];
        let program = builder
            .borrow()
            .clone()
            .build::<(TestArray, TestArray), (TestArray, TestArray)>(
                output_atoms,
                (Placeholder, Placeholder),
                (Placeholder, Placeholder),
            )
            .unwrap();
        let rendered = program.to_string();
        assert_eq!(rendered.matches("= while").count(), 1, "{rendered}");
        assert!(!rendered.contains("reduce_any"), "{rendered}");
        assert!(!rendered.contains("select"), "{rendered}");
        let (counter_output, value_output) =
            program.interpret((TestArray::scalar(2.0), TestArray::vector(vec![1.0, 2.0, 3.0]))).unwrap();
        assert_eq!(counter_output.values, vec![0.0]);
        assert_eq!(value_output.values, vec![4.0, 8.0, 12.0]);
    }

    #[test]
    fn test_normalize_batched_program_output_axis_appends_axis_moving_operations() {
        // Transpose arm: a batched program output carrying its lane axis at position 1 is normalized to position 0
        // by an appended staged transpose at the program tail.
        let mut builder = ProgramBuilder::<ArrayType, TestArray, TestArrayOperation>::new();
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let input = builder.add_input(input_type);
        let mut program = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        normalize_batched_program_output_axis(&mut program, 0, Some(1), 0, 3).unwrap();
        assert_eq!(
            program.output_types(),
            vec![ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3), Size::Static(2)]))],
        );
        let outputs = program.interpret(vec![TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]).unwrap();
        assert_eq!(outputs[0].values, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // Broadcast arm: a lane-uniform output gains the lane axis at position 0 through an appended broadcast.
        let mut builder = ProgramBuilder::<ArrayType, TestArray, TestArrayOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let mut program = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        normalize_batched_program_output_axis(&mut program, 0, None, 0, 4).unwrap();
        let outputs = program.interpret(vec![TestArray::scalar(7.0)]).unwrap();
        assert_eq!(outputs[0].values, vec![7.0; 4]);
    }

    #[test]
    fn test_jvp_of_batched_bounded_while_under_tracing_composes_with_masked_scan() {
        use crate::tracing_v2::batching::BatchContext;

        // F5 x F6 composition: jvp of a *vmapped bounded* while under the non-concretizing staged dispatch domain.
        // Batching stages one masked bounded while (the predicate `x < 8` is per lane and the iteration bound 5
        // survives the staged rewrite), so the while JVP rule takes the bounded staged path: stored residual
        // stacks plus a masked linear scan on the tangent side. Lanes [1, 5, 9] double 3, 1, and 0 times, so the
        // primal is [8, 10, 9] and the per-lane tangent scale is 2^iterations = [8, 2, 1].
        fn batched_bounded_while<C>(x: crate::tracing::Tracer<C>) -> crate::tracing::Tracer<C>
        where
            C: crate::contexts::StagingContext<Type = ArrayType, Constant = TestArray, Operation = TestArrayOperation>,
        {
            let context = x.context().clone();
            let mapped: crate::tracing::Tracer<C> = BatchContext::batch(
                &context,
                |lane| {
                    let while_operation = bounded_doubling_while_operation(8.0, 5);
                    let mut outputs = lane
                        .context()
                        .stage_operation(TestArrayOperation::While(Box::new(while_operation)), &[&lane])?;
                    Ok(outputs.remove(0))
                },
                x,
                Some(0),
                Some(0),
                None,
            )
            .unwrap();
            mapped
        }
        let (primal, tangent) = StagedDispatchTestArrayDomain
            .jvp(batched_bounded_while, TestArray::vector(vec![1.0, 5.0, 9.0]), TestArray::vector(vec![1.0, 1.0, 1.0]))
            .unwrap();
        assert_eq!(primal.values, vec![8.0, 10.0, 9.0]);
        assert_eq!(tangent.values, vec![8.0, 2.0, 1.0]);

        // The plain eager domain produces the same numbers...
        let (primal, tangent) = TestArrayDomain
            .jvp(batched_bounded_while, TestArray::vector(vec![1.0, 5.0, 9.0]), TestArray::vector(vec![1.0, 1.0, 1.0]))
            .unwrap();
        assert_eq!(primal.values, vec![8.0, 10.0, 9.0]);
        assert_eq!(tangent.values, vec![8.0, 2.0, 1.0]);

        // ... and reverse mode composes through the masked linear scan: the pullback contains the reversed scan
        // and no while loop, and the per-lane gradients match the tangent scales.
        let (output, pullback) = StagedDispatchTestArrayDomain
            .vjp(|x| Ok(batched_bounded_while(x)), TestArray::vector(vec![1.0, 5.0, 9.0]))
            .unwrap();
        assert_eq!(output.values, vec![8.0, 10.0, 9.0]);
        let rendered_pullback = pullback.to_string();
        assert!(rendered_pullback.contains("scan"), "{rendered_pullback}");
        assert!(rendered_pullback.contains("reverse=true"), "{rendered_pullback}");
        assert!(!rendered_pullback.contains("while"), "{rendered_pullback}");
        let cotangent = pullback.interpret(TestArray::vector(vec![1.0, 1.0, 1.0])).unwrap();
        assert_eq!(cotangent.values, vec![8.0, 2.0, 1.0]);
    }
}
