use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;

use ryft_macros::Parameter;

use crate::ElementwiseOperation;
use crate::batching::{ArrayBatch, BatchingError};
use crate::broadcasting::Broadcastable;
use crate::contexts::{Context, StagingContext, ValueResolution};
use crate::domains::Domain;
use crate::interpretation::InterpretableOperation;
use crate::macros::{check_builders, check_count};
use crate::operations::Operation;
use crate::operations::manipulation::{Broadcast, Transpose, TransposeOperation};
use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily, Placeholder};
use crate::programs::{AtomId, Program, ProgramBuilder, ProgramError, Value};
use crate::sharding::ShardingDimension;
use crate::tracing::{DomainTracer, DomainTracingContext, Tracer, TracerState, TracingContext};
use crate::tracing_v2::differentiation::{DifferentiationContext, replay_via_bind};
use crate::types::{ArrayType, Size, Typed};

// TODO(eaplatanios): Review this module.

/// Batching rule for one staged operation.
///
/// `BatchableOperation::batch` takes batched physical inputs paired with mapped-axis metadata and returns batched
/// physical outputs with their lane axes — the same shape as JAX's per-primitive batching rules (`fn(batched_args,
/// batch_dims, **params) -> (result_value, result_dim)`). Most primitive rules are context-free and use the default
/// `C = ()`. Active `batch` supplies [`BatchingContext`] for rules that must replay captured nested programs or
/// stage backend extension operations through the enclosing transform.
///
/// # Contract
///
///   - **Axis alignment.** If two or more inputs carry a mapped axis (`batch_axis.is_some()`),
///     elementwise operations require them to agree on the axis position. When they disagree,
///     return [`BatchingError::MisalignedBatchAxes`] with an error message that names
///     the misaligned axes and suggests the user repositions one of them with `Transpose` (the
///     N-D axis permutation primitive) before invoking the operation. Operations with explicit
///     axis arguments (`Dot`, `Transpose`, `Reshape`, …) rewrite those arguments to thread the
///     mapped axis through correctly.
///   - **Output axes.** For elementwise operations, output `ArrayBatch::batch_axis` matches the
///     common input axis. For axis-arg operations, the output axis follows from the lifted
///     axis arguments (see the per-op helpers in `tracing_v2::operations` — `lift_dot_dimensions`,
///     `lift_permutation`, `lift_reshape_shapes`).
///   - **Zero propagation.** Linear batching rules preserve zero tangent payloads through their operation-specific
///     semantics; canonical staged zeros are
///     handled before batching reaches concrete value-level interpretation.
///   - **Missing rule.** Variants without a defined batching rule (for example, a while-loop
///     whose loop predicate varies across lanes) return [`BatchingError::UnsupportedOperation`]
///     with a human-readable message that points at a likely fix.
///
/// The internal elementwise lifting helper computes the lifted op plus per-output axes for any pure elementwise op;
/// the matching value-level applicator composes the rule's axis arithmetic with [`InterpretableOperation::interpret`].
pub trait BatchableOperation<V: Value<ArrayType>, C>: Operation<ArrayType> {
    /// Applies this operation to packed batched inputs, returning batched outputs with the
    /// resulting lane axes, using `context` for rules that need active transform state.
    fn batch(&self, context: &C, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, ProgramError>;
}

/// Blanket [`BatchableOperation`] impl for any [`ElementwiseOperation`].
///
/// Mirrors the existing
/// [`impl<O: ElementwiseOperation> Operation<ArrayType> for O`](crate::operations::Operation):
/// every elementwise primitive automatically gets the standard elementwise batching rule, so per-op
/// `BatchableOperation` impls do not have to be written for elementwise primitives (`Add`, `Sub`, `Mul`, `Div`,
/// `Neg`, `Sin`, `Cos`, …). Ops with non-trivial axis arithmetic (`Dot`, `Transpose`, `Reshape`, …) and the
/// [`ArrayOperation`](crate::tracing_v2::ArrayOperation) operation enum (whose impls live with the enum in
/// [`operations::primitive`](crate::tracing_v2::operations::primitive)) keep their explicit impls; coherence is
/// preserved because none of those types implement [`ElementwiseOperation`].
impl<
    O: Clone + InterpretableOperation<ArrayType, V, C> + ElementwiseOperation,
    V: Value<ArrayType> + Broadcast + Transpose,
    C,
> BatchableOperation<V, C> for O
{
    #[inline]
    fn batch(&self, context: &C, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        apply_elementwise_batch(context, self, inputs)
    }
}

/// Walks `inputs` to compute the per-lane input types, the per-input axis metadata, and the
/// common lane size — the three quantities every per-op batching rule consumes before
/// dispatching to its axis-arithmetic helper. Returns `MismatchedBatchSizes` when two mapped
/// inputs disagree on their lane size and `DynamicBatchAxis` when any mapped input's axis is
/// non-static. When no inputs are mapped, the returned `axis_size` is `0` (no rule that needs a
/// lane size is ever invoked in that situation).
pub fn batch_input_metadata<V: Typed<ArrayType>>(
    inputs: &[ArrayBatch<V>],
) -> Result<(Vec<ArrayType>, Vec<Option<usize>>, usize), ProgramError> {
    let input_axes: Vec<Option<usize>> = inputs.iter().map(|input| input.batch_axis()).collect();
    let per_lane_input_types: Vec<ArrayType> =
        inputs.iter().map(|input| input.logical_type()).collect::<Result<Vec<_>, _>>()?;
    let mut axis_size: Option<usize> = None;
    for input in inputs {
        if let Some(size) = input.axis_size()? {
            match axis_size {
                Some(existing) if existing != size => {
                    return Err(BatchingError::MismatchedBatchSizes { expected: existing, actual: size }.into());
                }
                Some(_) => {}
                None => axis_size = Some(size),
            }
        }
    }
    Ok((per_lane_input_types, input_axes, axis_size.unwrap_or(0)))
}

/// Returns the [`ShardingDimension`] to insert at a newly introduced output batch axis, derived from the batched
/// inputs' mapped-dimension shardings.
///
/// This mirrors JAX deriving the batched dimension's sharding from the inputs' mapped-dimension specs
/// (`_mapped_axis_spec` feeding `get_sharding_for_vmap`): each genuinely batched input contributes the
/// [`ShardingDimension`] of its mapped axis, batched inputs that disagree are a
/// [`BatchingError::MisalignedBatchAxes`], and lane-uniform inputs (or inputs without a
/// [`Sharding`](crate::sharding::Sharding)) contribute nothing. When no batched input pins the axis the result is
/// [`ShardingDimension::Replicated`], so the new batch dimension is replicated. Deriving from the original
/// (pre-alignment) inputs avoids spuriously disagreeing with a lane-uniform operand that batching later broadcasts to
/// gain a singleton batch axis.
pub fn batch_dimension_sharding<V: Typed<ArrayType>>(
    inputs: &[ArrayBatch<V>],
) -> Result<ShardingDimension, ProgramError> {
    let mut result: Option<ShardingDimension> = None;
    for input in inputs {
        let Some(axis) = input.batch_axis() else {
            continue;
        };
        let physical_type = input.r#type();
        let Some(sharding) = physical_type.sharding() else {
            continue;
        };
        let dimension = sharding.dimensions()[axis].clone();
        match &result {
            Some(existing) if *existing != dimension => {
                return Err(BatchingError::MisalignedBatchAxes {
                    message: format!(
                        "batched inputs disagree on the sharding of their mapped axis: {existing} and {dimension}"
                    ),
                }
                .into());
            }
            _ => result = Some(dimension),
        }
    }
    Ok(result.unwrap_or(ShardingDimension::Replicated))
}

/// Applies a lifted operation to `inputs` via [`InterpretableOperation::interpret`] and packages
/// each output value with the corresponding entry of `output_axes`.
///
/// `output_axes` must have one entry per output produced by `lifted_op` on these inputs. This function is public so
/// that backend-owned operation enums (e.g., in `ryft-xla`) can implement [`BatchableOperation::batch`] for their
/// extension operations using the same application path as the built-in rules.
///
/// `context` supplies the value interpretation context directly. Active batching callers pass
/// [`BatchingContext::parent_context`] instead of recovering the context from input operands. This keeps lifted
/// interpretation well-defined when every operand is a symbolic zero and therefore carries no payload context.
pub fn apply_with_axes<V: Value<ArrayType>, C, O: InterpretableOperation<ArrayType, V, C>>(
    context: &C,
    lifted_op: &O,
    inputs: &[ArrayBatch<V>],
    output_axes: &[Option<usize>],
) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
    if inputs.is_empty() {
        return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 });
    }
    let input_values: Vec<V> = inputs.iter().map(|input| input.value().clone()).collect();
    let output_values = lifted_op.interpret(context, input_values.as_slice())?;
    check_count!("output", output_values, output_axes.len(), ProgramError);
    output_values
        .into_iter()
        .zip(output_axes.iter().copied())
        .map(|(value, axis)| {
            let output_type = value.r#type().into_owned();
            ArrayBatch::new(output_type, value, axis)
        })
        .collect()
}

/// Generic value-level batching helper for pure elementwise operations. Matches JAX's
/// `defbroadcasting` behavior: lane-uniform inputs are broadcast to the common batched physical
/// shape before applying the operation, so each value-level primitive only ever sees inputs that
/// agree on shape at the boundary. This is the canonical implementation of
/// [`BatchableOperation::batch`] for elementwise primitives.
///
/// Inputs whose mapped lane axis is at a different physical position from the first batched
/// input are realigned with an inserted [`TransposeOperation`] before broadcasting, matching
/// JAX's `matchaxis` policy. The canonical axis position is the first batched input's axis.
pub(crate) fn apply_elementwise_batch<
    V: Value<ArrayType> + Broadcast + Transpose,
    C,
    O: Clone + InterpretableOperation<ArrayType, V, C>,
>(
    context: &C,
    operation: &O,
    inputs: &[ArrayBatch<V>],
) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
    let (per_lane_types, original_input_axes, axis_size) = batch_input_metadata(inputs)?;
    let common_axis = original_input_axes.iter().copied().flatten().next();
    let aligned_inputs: Vec<ArrayBatch<V>> = match common_axis {
        None => inputs.to_vec(),
        Some(target) => inputs.iter().map(|input| align_batch_axis(input, target)).collect::<Result<_, _>>()?,
    };
    // Realignment only moves each mapped axis to `common_axis`; the per-lane types are unchanged.
    let input_axes: Vec<Option<usize>> = original_input_axes.iter().map(|axis| axis.and(common_axis)).collect();
    let (lifted_op, output_axes) = lift_elementwise(operation, &per_lane_types, &input_axes, axis_size)?;
    let broadcasted_inputs = match common_axis {
        None => aligned_inputs,
        Some(batch_axis) => {
            // Mirroring JAX's `defbroadcasting` policy, every operand whose per-lane shape is narrower than the
            // common per-lane shape of all operands (trailing-aligned) is broadcast to that common shape with the lane
            // axis at `batch_axis`. When the operands' per-lane shapes are not broadcast-compatible, the operands are
            // left at their lane-axis-inserted physical shapes so the operation surfaces its own shape error against
            // the original shapes. Realignment preserves the per-lane types, so the metadata ones are reused here.
            let common_per_lane = Broadcastable::broadcasted(per_lane_types.as_slice()).ok();
            let broadcasted_physical_type = |per_lane_type: &ArrayType| -> Result<ArrayType, ProgramError> {
                let common = common_per_lane.as_ref().unwrap_or(per_lane_type);
                elementwise_broadcast_target(per_lane_type, common, batch_axis, axis_size)
            };
            // Maps the operand's per-lane dimension `index` (trailing-aligned within the common per-lane shape) to its
            // position in the broadcast target, accounting for the lane axis insertion.
            let target_position = |per_lane_rank: usize, target_rank: usize, index: usize| {
                let position = (target_rank - per_lane_rank) + index;
                if position < batch_axis { position } else { position + 1 }
            };
            aligned_inputs
                .iter()
                .zip(input_axes.iter())
                .zip(per_lane_types.iter())
                .map(|((input, axis), per_lane_type)| {
                    let physical_type = broadcasted_physical_type(per_lane_type)?;
                    if physical_type == *input.r#type() {
                        return Ok(input.clone());
                    }
                    let target_rank = physical_type.rank() - 1;
                    let output_axes: Vec<usize> = match axis {
                        // Mapped operand with a narrower per-lane shape: keep the lane axis fixed and trailing-align
                        // the remaining dimensions.
                        Some(_) => (0..input.r#type().rank())
                            .map(|dimension| match dimension.cmp(&batch_axis) {
                                std::cmp::Ordering::Equal => batch_axis,
                                std::cmp::Ordering::Less => {
                                    target_position(per_lane_type.rank(), target_rank, dimension)
                                }
                                std::cmp::Ordering::Greater => {
                                    target_position(per_lane_type.rank(), target_rank, dimension - 1)
                                }
                            })
                            .collect(),
                        None => (0..per_lane_type.rank())
                            .map(|dimension| target_position(per_lane_type.rank(), target_rank, dimension))
                            .collect(),
                    };
                    let broadcasted = input.value().clone().broadcast(physical_type.clone(), output_axes.as_slice())?;
                    ArrayBatch::new(physical_type, broadcasted, Some(batch_axis))
                })
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    apply_with_axes(context, &lifted_op, &broadcasted_inputs, &output_axes)
}

/// Realigns a batched input by moving its mapped lane axis to `target_axis`.
///
/// Identity case (already at `target_axis`, or unbatched) returns the input unchanged. Otherwise
/// stages a [`TransposeOperation`] via the receiver's [`Transpose`] impl and returns a new
/// [`ArrayBatch`] whose physical type and value reflect the realigned axis.
///
/// # Parameters
///
///   - `input`: Batched input to realign.
///   - `target_axis`: Desired position of the mapped lane axis in the output.
pub(crate) fn align_batch_axis<V: Value<ArrayType> + Transpose>(
    input: &ArrayBatch<V>,
    target_axis: usize,
) -> Result<ArrayBatch<V>, ProgramError> {
    let Some(current_axis) = input.batch_axis() else {
        return Ok(input.clone());
    };
    if current_axis == target_axis {
        return Ok(input.clone());
    }
    let rank = input.r#type().rank();
    let permutation = move_axis_permutation(rank, current_axis, target_axis);
    let permuted_value = input.value().clone().transpose(permutation)?;
    let permuted_type = permuted_value.r#type().into_owned();
    ArrayBatch::new(permuted_type, permuted_value, Some(target_axis))
}

/// Broadcasts a lane-uniform `operand` to gain a singleton batch axis at `target_axis`.
///
/// This is the canonical building block for mixed batched/unbatched primitive rules (e.g.,
/// [`DotOperation::batch`](crate::tracing_v2::operations::dot::DotOperation)) and for lifting
/// lane-uniform residuals during linearization: it inserts a new axis at `target_axis` in the
/// operand's type, broadcasts the value to that shape, and returns the result as a batched
/// [`ArrayBatch`]. Elementwise rules instead broadcast lane-uniform operands to the full common
/// batched shape inside [`apply_elementwise_batch`].
///
/// Returns an error when called on an already-batched input — callers are expected to dispatch
/// the lane-uniform case explicitly.
///
/// # Parameters
///
///   - `operand`: Lane-uniform input to lift.
///   - `target_axis`: Position of the inserted batch axis in the output.
///   - `axis_size`: Size of the inserted batch axis.
pub(crate) fn broadcast_to_batched<V: Value<ArrayType> + Broadcast>(
    operand: &ArrayBatch<V>,
    target_axis: usize,
    axis_size: usize,
) -> Result<ArrayBatch<V>, ProgramError> {
    if operand.batch_axis().is_some() {
        return Err(BatchingError::MisalignedBatchAxes {
            message: "broadcast_to_batched expects a lane-uniform operand but received a batched value".to_string(),
        }
        .into());
    }
    let per_lane_type = operand.logical_type()?;
    let physical_type = per_lane_type.with_inserted_dimension(target_axis, Size::Static(axis_size))?;
    let output_axes: Vec<usize> = (0..per_lane_type.rank()).map(|i| if i < target_axis { i } else { i + 1 }).collect();
    let broadcasted = operand.value().clone().broadcast(physical_type.clone(), output_axes.as_slice())?;
    ArrayBatch::new(physical_type, broadcasted, Some(target_axis))
}

/// Returns the broadcast target for one elementwise operand under the JAX `defbroadcasting` policy: the common
/// per-lane shape with the mapped lane axis inserted at `batch_axis`. The common per-lane target only contributes
/// its shape — each operand keeps its own data type (e.g., a Boolean select condition broadcast against numeric
/// branches stays Boolean).
fn elementwise_broadcast_target(
    per_lane_type: &ArrayType,
    common_per_lane: &ArrayType,
    batch_axis: usize,
    axis_size: usize,
) -> Result<ArrayType, ProgramError> {
    let mut target = common_per_lane.clone();
    target.data_type = per_lane_type.data_type();
    Ok(target.with_inserted_dimension(batch_axis, Size::Static(axis_size))?)
}

/// Generic lifting rule for pure elementwise operations.
///
/// All inputs that carry the mapped axis must place it at the same position; lane-uniform inputs
/// (with `input_axes[i] == None`) are accepted and pass through unchanged. The lifted operation
/// is `operation.clone()` since elementwise semantics are preserved by adding the leading batch
/// dimension. Output types are inferred from the parent-physical input types, and every output
/// is given the common batch axis.
pub(crate) fn lift_elementwise<O: Clone + Operation<ArrayType>>(
    operation: &O,
    input_types: &[ArrayType],
    input_axes: &[Option<usize>],
    axis_size: usize,
) -> Result<(O, Vec<Option<usize>>), ProgramError> {
    check_count!("input", input_axes, input_types.len(), ProgramError);

    let mut common_axis: Option<usize> = None;
    for axis in input_axes.iter().copied().flatten() {
        match common_axis {
            Some(existing) if existing != axis => {
                return Err(BatchingError::MisalignedBatchAxes {
                    message: format!(
                        "elementwise lift for '{}' cannot align batch axis {axis} with existing batch axis {existing}: \
                        the operands' batched lane dimensions are at different positions. Stage a Transpose to move \
                        one operand's batch axis to the other's position, or use `in_axes` to align them at the \
                        outer batch boundary",
                        operation.name(),
                    ),
                }
                .into());
            }
            Some(_) => {}
            None => common_axis = Some(axis),
        }
    }

    // For ops whose `infer_output_types` requires all inputs to share a shape (e.g.,
    // `SelectOperation`), infer against the common per-lane shape with the lane axis inserted.
    // Ops with built-in broadcasting semantics (e.g., `AddOperation`) accept the broadcasted
    // shapes equally. When the per-lane shapes are not broadcast-compatible, fall back to the
    // lane-axis-inserted physical types so the operation surfaces its own shape error.
    let broadcasted_input_types: Vec<ArrayType> = match (common_axis, Broadcastable::broadcasted(input_types)) {
        (Some(axis), Ok(common)) => input_types
            .iter()
            .map(|per_lane_type| elementwise_broadcast_target(per_lane_type, &common, axis, axis_size))
            .collect::<Result<Vec<_>, _>>()?,
        _ => input_types
            .iter()
            .zip(input_axes.iter())
            .map(|(per_lane_type, axis)| -> Result<ArrayType, ProgramError> {
                match axis {
                    Some(k) => Ok(per_lane_type.with_inserted_dimension(*k, Size::Static(axis_size))?),
                    None => Ok(per_lane_type.clone()),
                }
            })
            .collect::<Result<Vec<_>, _>>()?,
    };
    let output_count = operation.infer_output_types(broadcasted_input_types.as_slice())?.len();
    Ok((operation.clone(), vec![common_axis; output_count]))
}

/// Staging context used by [`batch_program`] to capture a batched program replay: an ordinary trace over the
/// program's `(ArrayType, V, O)` type universe.
///
/// The capture parameter is pinned to `V` explicitly (rather than left at its default) so that bounds written against
/// this alias match their obligations syntactically.
pub type ProgramBatchingContext<V, O> = TracingContext<ArrayType, V, O, V>;

/// Policy for choosing a batched program's output axes.
///
/// Program batching always replays the program over physical values whose mapped lane axes are specified by the
/// caller. This policy controls how the replayed output tracers are packaged into the resulting program:
///
///   - [`Natural`](Self::Natural) keeps the mapped axes produced by the per-operation batching rules. Lane-uniform
///     outputs remain lane-uniform and are reported as `None`.
///   - [`AlignAllTo`](Self::AlignAllTo) normalizes every output to the requested mapped axis, moving already-batched
///     outputs with [`Transpose`] and broadcasting lane-uniform outputs across the lane.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ProgramBatchingOutputAxes {
    /// Keep the output axes naturally produced by the batched replay.
    Natural,

    /// Align every output to the specified mapped axis.
    AlignAllTo(usize),
}

/// Operation types whose captured flat programs can be batched into standalone lane-carrying programs.
///
/// This is the program-level counterpart of [`BatchableOperation`], implemented by closed operation enums (via
/// [`batch_program`]) so that higher-order operations can batch the programs they capture. The re-wrapping
/// `batch` rules of [`CustomJvpOperation`](crate::tracing_v2::operations::custom_derivatives::CustomJvpOperation)
/// and [`CustomVjpOperation`](crate::tracing_v2::operations::custom_derivatives::CustomVjpOperation) bound their
/// captured-program operation type by this trait. Routing program-level batching through a dedicated, lifetime-free
/// trait keeps the trait solver's recursion finite: the closed enum impl discharges the derived batching-context
/// obligations once, against the single [`ProgramBatchingContext`] type, instead of every batching rule re-deriving
/// them with fresh higher-ranked lifetimes (which defeats the solver's cycle detection and overflows).
pub trait BatchableProgramOperation<V: Value<ArrayType>>: Operation<ArrayType> + Sized {
    /// Batches `program` into a standalone program over lane-carrying physical types; refer to
    /// [`batch_program`] for the input/output axis conventions.
    fn batch_program(
        program: &Program<ArrayType, V, Self, Vec<V>, Vec<V>>,
        axis_size: usize,
        input_batch_axes: &[Option<usize>],
        output_batch_axes: ProgramBatchingOutputAxes,
    ) -> Result<(Program<ArrayType, V, Self, Vec<V>, Vec<V>>, Vec<Option<usize>>), ProgramError>;
}

impl<V: Value<ArrayType>, O: BatchableProgramOperation<V>> Program<ArrayType, V, O, Vec<V>, Vec<V>> {
    /// Batches this flat [`Program`] into a standalone program over lane-carrying physical types.
    ///
    /// Refer to [`batch_program`] for the precise replay semantics. This method requires
    /// [`BatchableProgramOperation`] rather than a direct [`BatchableOperation`] bound on the operation type because
    /// the latter is self-referential through derived batching contexts and would overflow the trait solver at the
    /// recursive call sites of higher-order batching rules.
    pub fn batched(
        &self,
        axis_size: usize,
        input_batch_axes: &[Option<usize>],
        output_batch_axes: ProgramBatchingOutputAxes,
    ) -> Result<(Self, Vec<Option<usize>>), ProgramError> {
        O::batch_program(self, axis_size, input_batch_axes, output_batch_axes)
    }
}

/// Batches a captured program into a standalone program over lane-carrying physical types.
///
/// This is the batching analog of symbolic program linearization: staged higher-order
/// batching rules use it to batch captured programs *without* concretizing any lane values, so that batched
/// control-flow and custom-derivative structure can be staged back into the enclosing trace. Unlike linearization,
/// batching does not split value spaces — the batched replay stays in one tracer space — so the packaging is one
/// fresh replay: the program is replayed through a [`BatchingContext`] over a fresh [`ProgramBatchingContext`] trace,
/// lifting every instruction through its [`BatchableOperation`] rule, and the resulting staged program is extracted
/// together with the requested output-axis policy.
///
/// Inputs whose `input_batch_axes[i]` is `Some(k)` consume the original logical input type with a mapped lane axis of
/// size `axis_size` inserted at position `k`, while inputs with `None` enter lane-uniform at their original logical
/// types. [`ProgramBatchingOutputAxes::Natural`] keeps the mapped axes produced by the batching rules. This is what
/// staged control-flow batching needs, because branch/body outputs are normalized to the surrounding operation's
/// signature afterward. [`ProgramBatchingOutputAxes::AlignAllTo`] imposes a canonical output axis, which is what
/// custom-derivative re-wrapping needs so independently batched primal/JVP/forward/backward programs have mutually
/// consistent signatures.
///
/// # Parameters
///
///   - `program`: Captured flat program over per-lane (logical) input and output types.
///   - `axis_size`: Size of the mapped lane axis.
///   - `input_batch_axes`: Mapped lane-axis position per program input, or `None` for a lane-uniform input.
///   - `output_batch_axes`: Policy for packaging the batched program outputs.
pub fn batch_program<V, O>(
    program: &Program<ArrayType, V, O, Vec<V>, Vec<V>>,
    axis_size: usize,
    input_batch_axes: &[Option<usize>],
    output_batch_axes: ProgramBatchingOutputAxes,
) -> Result<(Program<ArrayType, V, O, Vec<V>, Vec<V>>, Vec<Option<usize>>), ProgramError>
where
    V: Value<ArrayType> + 'static,
    O: Clone + Operation<ArrayType> + 'static,
    O: BatchableOperation<Tracer<ProgramBatchingContext<V, O>>, BatchingContext<ProgramBatchingContext<V, O>>>,
    Tracer<ProgramBatchingContext<V, O>>: Broadcast + Transpose,
{
    let logical_input_types = program.input_types();
    let input_count = logical_input_types.len();
    check_count!("input", input_batch_axes, input_count, ProgramError);
    let parent_context: ProgramBatchingContext<V, O> = TracingContext::new();
    let builder = parent_context.builder().clone();
    // Keep every tracer and context that holds a clone of `builder` inside this scope so that recovering the
    // builder below is a real ownership check.
    let (output_atom_ids, output_axes) = {
        let batching_context = BatchingContext::new(parent_context, axis_size);
        let mut input_values = Vec::with_capacity(input_count);
        for (logical_type, axis) in logical_input_types.iter().zip(input_batch_axes.iter()) {
            let physical_type = match axis {
                Some(position) => logical_type.with_inserted_dimension(*position, Size::Static(axis_size))?,
                None => logical_type.clone(),
            };
            let atom = builder.borrow_mut().add_input(physical_type);
            input_values.push(batching_context.batched_value(atom, logical_type.clone(), *axis));
        }
        let output_values = batching_context.stage_program(program, input_values)?;
        let mut output_atom_ids = Vec::with_capacity(output_values.len());
        let mut output_axes = Vec::with_capacity(output_values.len());
        for output_value in output_values {
            match output_batch_axes {
                ProgramBatchingOutputAxes::Natural => {
                    let atom = output_value.atom_id()?;
                    output_axes.push(output_value.meta().0.axis());
                    output_atom_ids.push(atom);
                }
                ProgramBatchingOutputAxes::AlignAllTo(target_axis) => {
                    let atom = output_value.atom_id()?;
                    let axis = output_value.meta().0.axis();
                    let logical_type = output_value.r#type().into_owned();
                    let physical_type = match axis {
                        Some(k) => logical_type.with_inserted_dimension(k, Size::Static(axis_size))?,
                        None => logical_type,
                    };
                    let parent_batch = ArrayBatch::new(
                        physical_type.clone(),
                        batching_context.parent_context().tracer(atom, Some(physical_type)),
                        axis,
                    )?;
                    let aligned_batch = match axis {
                        Some(axis) if axis == target_axis => parent_batch,
                        Some(_) => align_batch_axis(&parent_batch, target_axis)?,
                        None => broadcast_to_batched(&parent_batch, target_axis, axis_size)?,
                    };
                    output_atom_ids.push(aligned_batch.into_value().atom_id()?);
                    output_axes.push(Some(target_axis));
                }
            }
        }
        Ok::<_, ProgramError>((output_atom_ids, output_axes))
    }?;
    let output_count = output_atom_ids.len();
    let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
    let batched_program = builder
        .build(output_atom_ids, vec![Placeholder; input_count], vec![Placeholder; output_count])?
        .into_simplified()?;
    Ok((batched_program, output_axes))
}

/// Trace context that introduces exactly one batched lane at a chosen axis.
///
/// [`BatchingContext`] is the active context for one level of `batch`: it runs the user's function
/// against logical per-lane [`ArrayType`]s while leaving the runtime value type of the staged
/// program equal to the parent context's value type. Operations staged through this context are
/// lifted through their [`BatchableOperation`] rules at bind time. The lifted operation
/// is then staged into the parent context, so nested transforms compose by wrapping contexts
/// rather than by making each active transform pretend to be a backend domain.
///
/// Nested `batch` composes by repeated context wrapping:
/// `BatchingContext<BatchingContext<C>>` is a two-level batching trace, and the staged program's
/// value type remains `C::Value` regardless of the nesting depth. Each level owns its own
/// `axis_size` and optional `axis_name`, while primitive binds recursively pass through every
/// parent context in order.
#[derive(Debug)]
pub struct BatchingContext<C: Context<Type = ArrayType>> {
    /// Parent trace context wrapped by this batching level.
    parent_context: C,

    /// Size of the batched lane this level introduces.
    axis_size: usize,

    /// Optional human-readable name for this batched axis. Collectives such as `psum`, `pmean`, and
    /// `pmax` can address this axis by name from inside the batched function body.
    axis_name: Option<String>,
}

impl<C: StagingContext<Type = ArrayType>> BatchingContext<C> {
    /// Creates a new anonymous [`BatchingContext`] that wraps `parent_context` with the supplied lane size.
    #[inline]
    pub fn new(parent_context: C, axis_size: usize) -> Self {
        Self::with_axis_name(parent_context, axis_size, None)
    }

    /// Creates a new [`BatchingContext`] with an optionally named batched axis. Collectives such as `psum`,
    /// `pmean`, and `pmax` can address a named axis from inside the batched function body.
    #[inline]
    pub fn with_axis_name(parent_context: C, axis_size: usize, axis_name: Option<String>) -> Self {
        Self { parent_context, axis_size, axis_name }
    }
}

impl<C> BatchingContext<C>
where
    C: StagingContext<Type = ArrayType>,
    C::Operation: BatchableOperation<Tracer<C, C::Meta>, Self>,
{
    /// Creates a live [`BatchingTracer`] referring to `atom` in the parent builder, carrying the given logical
    /// (per-lane) type and mapped lane axis at this batching level.
    ///
    /// This is the axis-carrying counterpart of [`StagingContext::tracer`]: callers that have already staged an atom
    /// at its physical type and know where its mapped lane axis sits use this to attach that axis to the flowing
    /// value as the head of its [`Meta`](StagingContext::Meta) stack. The tail (the parent context's per-level axes)
    /// is left lane-uniform here, which is correct for a fresh program input that has no enclosing batched value;
    /// an enclosing nested-`batch` level instead prepends its axis onto the *incoming* value's existing stack
    /// directly (see [`BatchContext::batch`]).
    ///
    /// # Parameters
    ///
    ///   - `atom`: Staged atom in the parent builder.
    ///   - `logical_type`: Per-lane (unbatched) type the value reports inside the batched body.
    ///   - `batch_axis`: Mapped lane axis carried by the value ([`BatchAxis::uniform`] when lane-uniform).
    #[inline]
    pub fn batched_value(
        &self,
        atom: AtomId,
        logical_type: ArrayType,
        batch_axis: impl Into<BatchAxis>,
    ) -> BatchingTracer<C> {
        Tracer::new_with_meta(
            self.clone(),
            TracerState::Live(atom),
            logical_type,
            (batch_axis.into(), C::Meta::default()),
        )
    }
}

impl<C: StagingContext<Type = ArrayType>> BatchingContext<C> {
    /// Interprets a captured flat program while staging batched primitive calls into the parent context.
    ///
    /// This only requires the program's own operation family `O` to be batchable; it deliberately does not require the
    /// enclosing context's [`Operation`](Domain::Operation) to be batchable, so higher-order batching rules can replay
    /// a captured sub-program through a [`BatchingContext`] whose [`Context`] impl is not yet in scope.
    pub(crate) fn interpret_program<O>(
        &self,
        program: &Program<ArrayType, C::Constant, O, Vec<C::Constant>, Vec<C::Constant>>,
        inputs: Vec<ArrayBatch<C::Value>>,
    ) -> Result<Vec<ArrayBatch<C::Value>>, ProgramError>
    where
        O: BatchableOperation<Tracer<C, C::Meta>, Self>,
    {
        program.interpret_with(
            inputs,
            |_, constant| Ok(ArrayBatch::unbatched(self.parent_context.constant(constant.clone()))),
            |instruction, instruction_inputs| instruction.operation().batch(self, instruction_inputs),
        )
    }
}

impl<C: Context<Type = ArrayType>> BatchingContext<C> {
    /// Returns the parent [`Context`] this batching context wraps. Batching rules use this to stage operations
    /// directly at the parent level — for example, [`forward_collective_to_parent`](
    /// crate::tracing_v2::operations::collective::forward_collective_to_parent) re-stages a collective that targets
    /// an outer named axis.
    #[inline]
    pub fn parent_context(&self) -> &C {
        &self.parent_context
    }

    /// Returns this batch level's named axis, if the enclosing `batch` call named one. Batching rules for
    /// collective-like operations match their own axis name against this to decide whether to consume the mapped
    /// lane axis at this level or forward the operation to [`BatchingContext::parent_context`].
    #[inline]
    pub fn axis_name(&self) -> Option<&str> {
        self.axis_name.as_deref()
    }

    /// Returns this batch level's lane count.
    #[inline]
    pub fn axis_size(&self) -> usize {
        self.axis_size
    }
}

impl<C: Context<Type = ArrayType>> Clone for BatchingContext<C> {
    fn clone(&self) -> Self {
        Self {
            parent_context: self.parent_context.clone(),
            axis_size: self.axis_size,
            axis_name: self.axis_name.clone(),
        }
    }
}

impl<C> Domain for BatchingContext<C>
where
    C: StagingContext<Type = ArrayType>,
    C::Operation: BatchableOperation<Tracer<C, C::Meta>, Self>,
{
    type Type = ArrayType;
    type Value = BatchingTracer<C>;
    type Constant = C::Constant;
    type Operation = C::Operation;
}

impl<C> Context for BatchingContext<C>
where
    C: StagingContext<Type = ArrayType>,
    C::Operation: BatchableOperation<Tracer<C, C::Meta>, Self>,
{
    /// Lifts a constant payload into this batching context by recording it as a lane-uniform [`BatchingTracer`].
    #[inline]
    fn lift(&self, constant: C::Constant) -> Result<BatchingTracer<C>, ProgramError> {
        Ok(self.constant(constant))
    }

    /// Binding in a batching context routes through [`StagingContext::stage_operation`], which lifts the operation over
    /// each input's mapped batch axis through the operation's [`BatchableOperation`] rule.
    #[inline]
    fn bind<P: Into<Self::Operation>>(
        &self,
        operation: P,
        inputs: &[BatchingTracer<C>],
    ) -> Result<Vec<BatchingTracer<C>>, ProgramError> {
        let operation = operation.into();
        self.stage_operation(operation, inputs)
    }

    #[inline]
    fn resolve(&self, value: &BatchingTracer<C>) -> ValueResolution<C::Constant> {
        if !Rc::ptr_eq(self.builder(), value.context().builder()) {
            return ValueResolution::Opaque;
        }
        let Ok(atom_id) = value.atom_id() else {
            return ValueResolution::Opaque;
        };
        match self.builder().borrow().atoms().get(atom_id.index()).and_then(|atom| atom.as_constant()) {
            Some(constant) => ValueResolution::Concrete(constant.clone()),
            None => ValueResolution::Staged(atom_id),
        }
    }
}

impl<C> StagingContext for BatchingContext<C>
where
    C: StagingContext<Type = ArrayType>,
    C::Operation: BatchableOperation<Tracer<C, C::Meta>, Self>,
{
    type Meta = (BatchAxis, C::Meta);

    #[inline]
    fn builder(&self) -> &Rc<RefCell<ProgramBuilder<Self::Type, Self::Constant, Self::Operation>>> {
        self.parent_context.builder()
    }

    fn stage_operation<P: Into<Self::Operation>, I: std::borrow::Borrow<BatchingTracer<C>>>(
        &self,
        operation: P,
        inputs: &[I],
    ) -> Result<Vec<BatchingTracer<C>>, ProgramError> {
        let operation = operation.into();
        check_builders!(self.builder(), [inputs.iter().map(|input| input.borrow().context().builder())])
            .map_err(|error| self.error(error))?;
        if self.builder().borrow().error.is_some() {
            let input_types = inputs.iter().map(|input| input.borrow().r#type().into_owned()).collect::<Vec<_>>();
            let output_types = operation.infer_output_types(input_types.as_slice())?;
            return Ok(output_types
                .into_iter()
                .map(|r#type| Tracer::new(self.clone(), TracerState::Poison, r#type))
                .collect());
        }

        // Zero-input operations (e.g., `ZeroOperation`, `OneOperation`) are lane-uniform by
        // construction: every batch lane receives the same constant value, and there is no input
        // batch axis to lift through. Stage them directly into the parent's builder with an empty
        // input list and surface the resulting parent atoms as lane-uniform values (a default
        // [`BatchAxis::uniform`] meta, via the default `tracer` path).
        if inputs.is_empty() {
            let parent_outputs = self.parent_context.stage_nullary_operation(operation)?;
            return Ok(parent_outputs
                .into_iter()
                .map(|parent_value| -> Result<BatchingTracer<C>, ProgramError> {
                    let parent_physical_type = parent_value.r#type().into_owned();
                    let atom = parent_value.atom_id()?;
                    Ok(self.tracer(atom, Some(parent_physical_type)))
                })
                .collect::<Result<Vec<_>, _>>()?);
        }

        // Build parent-level input batches. Each `ArrayBatch` wraps the same atom as a *parent* trace value at the
        // parent-physical (= this level's physical) type, with this level's mapped batch axis. This level's axis for
        // each input is the head of the value's `Meta` cons-stack (`input.meta().0`), and the parent value the rule
        // dispatches through carries the *tail* of that stack (`input.meta().1`), which is exactly the parent context's
        // own per-level axes — so when the parent is itself a `BatchingContext` (nested `batch`), that parent's
        // `stage_operation` reads *its* axis straight off the value in hand, with no side table. The rule's body
        // (`operation.batch(...)`) then dispatches through the parent value's primitive impls, staging directly into
        // the parent context; multi-op staging (e.g., lane-varying `Condition` lowering to two branches + a per-lane
        // `Select`) emerges automatically.
        let mut parent_input_batches: Vec<ArrayBatch<C::Value>> = Vec::with_capacity(inputs.len());
        for input in inputs {
            let input = input.borrow();
            let atom = match input.atom_id() {
                Ok(atom) => atom,
                Err(error) => return Err(self.error(error)),
            };
            let logical_type = input.r#type().into_owned();
            let axis = input.meta().0.axis();
            let parent_physical_type = match axis {
                Some(k) => logical_type.with_inserted_dimension(k, Size::Static(self.axis_size))?,
                None => logical_type,
            };
            let parent_value = Tracer::new_with_meta(
                self.parent_context.clone(),
                TracerState::Live(atom),
                parent_physical_type.clone(),
                input.meta().1.clone(),
            );
            parent_input_batches.push(ArrayBatch::new(parent_physical_type, parent_value, axis)?);
        }
        let output_batches = operation.batch(self, parent_input_batches.as_slice())?;

        let mut output_values = Vec::with_capacity(output_batches.len());
        for output_batch in output_batches {
            let axis = output_batch.batch_axis();
            let parent_value = output_batch.into_value();
            let parent_physical_type = parent_value.r#type().into_owned();
            let atom = parent_value.atom_id()?;
            let logical_type = match axis {
                Some(k) => parent_physical_type.without_dimension(k)?.0,
                None => parent_physical_type,
            };
            // The output value's `Meta` stack is this level's output axis (head) on top of the rule's parent output
            // value's stack (tail), so the outer levels' axes for this freshly staged atom are carried through.
            output_values.push(Tracer::new_with_meta(
                self.clone(),
                TracerState::Live(atom),
                logical_type,
                (BatchAxis::from(axis), parent_value.meta().clone()),
            ));
        }
        Ok(output_values)
    }
}

impl<C> DifferentiationContext for BatchingContext<C>
where
    C: StagingContext<Type = ArrayType> + DifferentiationContext + Domain<Type = ArrayType>,
    C: DifferentiationContext<Tangent = <C as Domain>::Value>,
    BatchingContext<C>:
        StagingContext<Type = ArrayType, Constant = <C as Domain>::Constant, Operation = <C as Domain>::Operation>,
{
    type Tangent = BatchingTracer<C>;

    #[inline]
    fn validate_primal(&self, primal: &Self::Value) -> Result<(), ProgramError> {
        check_builders!(self.builder(), primal.context().builder()).map_err(|error| self.error(error))
    }

    /// Differentiation through a batching context is only available when the parent context is itself a staging
    /// context whose tangent is its own staged value, so primal values are always tracers and concretizing
    /// extractions on them cannot succeed.
    #[inline]
    fn supports_primal_concretization(&self) -> bool {
        false
    }
}

/// Value flowing through a [`BatchingContext<C>`]: the unified [`Tracer`] specialized to carry a [`BatchAxis`] as its
/// metadata. The batch axis rides on the value itself, so the per-operation [`BatchableOperation`] rules route the
/// mapped lane through [`StagingContext::stage_operation`] from the value in hand. Its capability impls (arithmetic,
/// `Broadcast`, `Dot`, `Reduce`, `Select`, …) are the shared `Tracer<C, C::Meta>` impls, so batching needs no bespoke
/// value-level operation impls of its own.
///
/// The carried [`ArrayType`] is the *logical* (per-lane, unbatched) type, matching what the staged value reports to
/// the batched function body; the physical type with the mapped axis inserted is reconstructed from the value's
/// [`BatchAxis`] when it is handed to a batching rule.
///
/// The [`Meta`](StagingContext::Meta) is a recursive per-level cons-stack `(BatchAxis, C::Meta)` mirroring the context
/// nesting: the head [`BatchAxis`] is *this* batching level's mapped lane axis, and the tail `C::Meta` is the parent
/// context's metadata (itself another `(BatchAxis, …)` stack for a nested `vmap`). This is what lets every level of a
/// nested `batch` recover its own lane axis for a value without any side table — `batch` over `batch` simply prepends
/// one more axis onto the incoming value's stack.
pub type BatchingTracer<C> = Tracer<BatchingContext<C>, (BatchAxis, <C as StagingContext>::Meta)>;

/// Lane-carrying batching value selected by an ordinary backend [`Domain`]. This is the [`BatchingTracer`] flowing
/// through the [`BatchingContext`] that wraps a fresh trace over `D`'s constant and operation families.
///
/// The parent trace is a plain [`TracingContext`] whose [`StagingContext::Meta`] is `()`, so the per-level cons-stack
/// bottoms out at `(BatchAxis, ())`. This is spelled concretely (rather than via the [`BatchingTracer`] alias) so the
/// `()` tail is written directly instead of as a `<TracingContext<…> as StagingContext>::Meta` projection, which the
/// trait solver does not always normalize to `()` when the same alias appears in several positions of a generic
/// signature.
pub type DomainBatchingValue<D> = Tracer<
    BatchingContext<TracingContext<ArrayType, <D as Domain>::Constant, <D as Domain>::Operation>>,
    (BatchAxis, ()),
>;

/// Per-value lane dimension index carried as the [`Tracer`] metadata of a [`BatchingTracer`]. `Some(k)` means the
/// value's mapped batch lane sits at physical axis `k`; [`BatchAxis::uniform`] (the [`Default`]) means the value is
/// lane-uniform — it carries no physical dimension for the current batch lanes and is the same value for every lane.
///
/// This is the per-value counterpart of the per-`batch()`-call [`BatchAxisSpecification`]; it rides on the staged
/// value itself — as the [`Tracer`] metadata of a batching value — so the per-operation
/// batching rules route the mapped lane straight from the value in hand.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Parameter)]
pub struct BatchAxis(Option<usize>);

impl BatchAxis {
    /// Creates a mapped lane axis at physical position `axis`.
    #[inline]
    pub fn mapped(axis: usize) -> Self {
        Self(Some(axis))
    }

    /// Creates a lane-uniform axis (the value is shared across every batch lane). Equivalent to [`BatchAxis::default`].
    #[inline]
    pub fn uniform() -> Self {
        Self(None)
    }

    /// Returns the mapped lane axis position, or `None` when this value is lane-uniform.
    #[inline]
    pub fn axis(&self) -> Option<usize> {
        self.0
    }

    /// Returns `true` when this value is lane-uniform (carries no mapped lane axis).
    #[inline]
    pub fn is_uniform(&self) -> bool {
        self.0.is_none()
    }
}

impl From<Option<usize>> for BatchAxis {
    fn from(axis: Option<usize>) -> Self {
        Self(axis)
    }
}

impl From<usize> for BatchAxis {
    fn from(axis: usize) -> Self {
        Self(Some(axis))
    }
}

/// Specification of the mapped axis introduced by one [`Batch::batch`] / [`BatchContext::batch`] call: an optional
/// explicit lane size and an optional axis name.
///
/// The lane size is normally inferred from the mapped inputs; provide an explicit size to either pin it or to drive
/// a fully-broadcast `batch` whose lane count would otherwise be unobservable. The axis name makes the mapped axis
/// addressable by name from collectives (`psum`, `pmean`, `pmax`) inside the batched function body, mirroring JAX's
/// `vmap(..., axis_name=...)`.
///
/// [`BatchAxisSpecification`] converts from the plain size forms, so call sites that do not need a name can pass
/// `None`, `Some(size)`, or `size` directly:
///
/// ```ignore
/// domain.batch(f, input, in_axes, out_axes, None)?;                                     // Inferred size, anonymous.
/// domain.batch(f, input, in_axes, out_axes, 8)?;                                        // Explicit size, anonymous.
/// domain.batch(f, input, in_axes, out_axes, BatchAxisSpecification::named("devices"))?; // Inferred size, named.
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BatchAxisSpecification {
    /// Explicit lane size, or `None` to infer it from the mapped inputs.
    size: Option<usize>,

    /// Name that collectives can use to address this axis, or `None` for an anonymous axis.
    name: Option<String>,
}

impl BatchAxisSpecification {
    /// Creates an axis specification with an explicit lane size.
    pub fn sized(size: usize) -> Self {
        Self { size: Some(size), name: None }
    }

    /// Creates a named axis specification whose lane size is inferred from the mapped inputs.
    pub fn named(name: impl Into<String>) -> Self {
        Self { size: None, name: Some(name.into()) }
    }

    /// Creates a named axis specification with an explicit lane size.
    pub fn sized_and_named(size: usize, name: impl Into<String>) -> Self {
        Self { size: Some(size), name: Some(name.into()) }
    }
}

impl From<Option<usize>> for BatchAxisSpecification {
    fn from(size: Option<usize>) -> Self {
        Self { size, name: None }
    }
}

impl From<usize> for BatchAxisSpecification {
    fn from(size: usize) -> Self {
        Self::sized(size)
    }
}

/// Per-leaf axis specification for the `in_axes` / `out_axes` arguments of [`Batch::batch`] and
/// [`BatchContext::batch`].
///
/// [`BatchAxesSpecification::PerLeaf`] carries one [`BatchAxis`] per leaf of the matching parameter structure (the
/// fully-explicit form), while [`BatchAxesSpecification::Uniform`] applies one [`BatchAxis`] to every leaf — the typed
/// counterpart of JAX's `in_axes=0` shorthand. Both entry points accept anything convertible into
/// [`BatchAxesSpecification`]. Plain per-leaf values convert automatically, and a single leaf additionally converts
/// from the bare `Option<usize>` / `usize` forms, so call sites can pass `Some(0)` for a single leaf,
/// `(BatchAxis::mapped(0), BatchAxis::uniform())` for a pair, or `BatchAxesSpecification::Uniform(0.into())` to map
/// axis 0 of every leaf without spelling out the structure.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BatchAxesSpecification<A> {
    /// One [`BatchAxis`] applied to every leaf of the matching parameter structure.
    Uniform(BatchAxis),

    /// Explicit per-leaf [`BatchAxis`] specifications matching the parameter structure exactly.
    PerLeaf(A),
}

impl<A> From<A> for BatchAxesSpecification<A> {
    fn from(axes: A) -> Self {
        Self::PerLeaf(axes)
    }
}

/// Single-leaf ergonomic conversion: a bare `Option<usize>` maps the one leaf at that axis (or lane-uniform).
impl From<Option<usize>> for BatchAxesSpecification<BatchAxis> {
    fn from(axis: Option<usize>) -> Self {
        Self::PerLeaf(BatchAxis::from(axis))
    }
}

/// Single-leaf ergonomic conversion: a bare `usize` maps the one leaf at that axis.
impl From<usize> for BatchAxesSpecification<BatchAxis> {
    fn from(axis: usize) -> Self {
        Self::PerLeaf(BatchAxis::from(axis))
    }
}

/// Extension trait that exposes [`Batch::batch`] as a method on any [`Domain`] whose `Type` is
/// [`ArrayType`].
///
/// `domain.batch(f, input, in_axes, out_axes, axis)` is the concrete-value batching entry point; it mirrors how
/// `jvp` sits on [`DifferentiationContext`]. Already-traced values use
/// the active context's [`BatchContext::batch`] path so nested transforms compose through context wrapping.
pub trait Batch: Domain<Type = ArrayType> {
    /// Maps a traced function over array axes selected per leaf by `in_axes` and places each output's
    /// mapped axis at the position requested by `out_axes`.
    ///
    /// Each `in_axes` leaf is either `Some(k)` (the input is mapped on axis `k` of its physical type)
    /// or `None` (the input is lane-uniform / broadcast across the batched lanes). When at least one
    /// input is mapped, the lane size is inferred from those inputs; the `axis` parameter accepts
    /// anything convertible to a [`BatchAxis`] and can supply an explicit lane size to either pin the
    /// inferred size or drive a fully-broadcast `batch` whose lane count would otherwise be
    /// unobservable, as well as an axis name that collectives (`psum`, `pmean`, `pmax`) inside the
    /// batched function can address. The per-leaf `out_axes` selects where the mapped axis lands
    /// in each output: `Some(k)` requests position `k` (an explicit transpose is staged when the
    /// natural output axis differs), and `None` declares the corresponding output to be lane-uniform
    /// (e.g., a value produced from broadcast inputs without staging any per-lane work).
    ///
    /// This is the concrete-value entry point. Already-traced values use [`BatchContext::batch`] on
    /// their active context.
    fn batch<F, I, O>(
        &self,
        function: F,
        input: I,
        in_axes: impl Into<BatchAxesSpecification<I::To<BatchAxis>>>,
        out_axes: impl Into<BatchAxesSpecification<O::To<BatchAxis>>>,
        axis: impl Into<BatchAxisSpecification>,
    ) -> Result<O::To<Self::Value>, BatchingError>
    where
        Self: Context,
        I: Parameterized<
                Self::Value,
                ParameterStructure: Debug + PartialEq,
                Family: ParameterizedFamily<ArrayType>
                            + ParameterizedFamily<Self::Constant>
                            + ParameterizedFamily<BatchAxis>
                            + ParameterizedFamily<DomainTracer<Self>>
                            + ParameterizedFamily<DomainBatchingValue<Self>>,
            >,
        O: Parameterized<
                DomainBatchingValue<Self>,
                ParameterStructure: Debug + PartialEq,
                Family: ParameterizedFamily<ArrayType>
                            + ParameterizedFamily<Self::Value>
                            + ParameterizedFamily<Self::Constant>
                            + ParameterizedFamily<BatchAxis>
                            + ParameterizedFamily<DomainTracer<Self>>
                            + ParameterizedFamily<DomainBatchingValue<Self>>,
            >,
        I::To<BatchAxis>: Parameterized<BatchAxis, ParameterStructure = I::ParameterStructure>,
        O::To<BatchAxis>: Parameterized<BatchAxis, ParameterStructure = O::ParameterStructure>,
        I::To<DomainTracer<Self>>: Parameterized<
                DomainTracer<Self>,
                ParameterStructure = I::ParameterStructure,
                To<ArrayType> = I::To<ArrayType>,
                To<Self::Value> = I,
                To<Self::Constant> = I::To<Self::Constant>,
                To<BatchAxis> = I::To<BatchAxis>,
                To<DomainBatchingValue<Self>> = I::To<DomainBatchingValue<Self>>,
            >,
        O::To<DomainTracer<Self>>: Parameterized<
                DomainTracer<Self>,
                ParameterStructure = O::ParameterStructure,
                To<ArrayType> = O::To<ArrayType>,
                To<Self::Value> = O::To<Self::Value>,
                To<Self::Constant> = O::To<Self::Constant>,
                To<BatchAxis> = O::To<BatchAxis>,
                To<DomainBatchingValue<Self>> = O,
            >,
        I::To<ArrayType>: Parameterized<
                ArrayType,
                To<Self::Value> = I,
                To<Self::Constant> = I::To<Self::Constant>,
                To<DomainTracer<Self>> = I::To<DomainTracer<Self>>,
                To<DomainBatchingValue<Self>> = I::To<DomainBatchingValue<Self>>,
            >,
        O::To<ArrayType>: Parameterized<
                ArrayType,
                To<Self::Value> = O::To<Self::Value>,
                To<Self::Constant> = O::To<Self::Constant>,
                To<DomainTracer<Self>> = O::To<DomainTracer<Self>>,
                To<DomainBatchingValue<Self>> = O,
            >,
        Self::Operation: Clone
            + From<TransposeOperation>
            + BatchableOperation<DomainTracer<Self>, BatchingContext<DomainTracingContext<Self>>>,
        F: FnOnce(I::To<DomainBatchingValue<Self>>) -> Result<O, ProgramError>,
    {
        let structure = input.parameter_structure();
        let input_values = input.into_parameters().collect::<Vec<_>>();
        let parent_context: DomainTracingContext<Self> = TracingContext::new();
        let builder = parent_context.builder().clone();
        let mut input_tracers = Vec::with_capacity(input_values.len());
        for value in input_values.iter() {
            let physical_type = value.r#type().into_owned();
            let atom = builder.borrow_mut().add_input(physical_type.clone());
            input_tracers.push(parent_context.tracer(atom, Some(physical_type)));
        }
        let traced_input = I::To::<DomainTracer<Self>>::from_parameters(structure.clone(), input_tracers)?;
        // Batching rules ride up the `ProgramError`-typed staging kernel as `ProgramError::Custom` payloads; the
        // `From<ProgramError>` conversions behind the `?` operators below re-type them so the public `batch` surfaces
        // a transform-owned `BatchingError`, mirroring how `value_and_grad` surfaces a `DifferentiationError`.
        let traced_output: O::To<DomainTracer<Self>> =
            BatchContext::batch(&parent_context, function, traced_input, in_axes.into(), out_axes.into(), axis)?;
        if let Some(error) = builder.borrow_mut().error.take() {
            return Err(error.into());
        }
        let output_structure = traced_output.parameter_structure();
        let output_atom_ids = traced_output.parameters().map(Tracer::atom_id).collect::<Result<Vec<_>, _>>()?;
        drop(traced_output);
        drop(parent_context);

        let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        let program: Program<ArrayType, Self::Constant, Self::Operation, I::To<Self::Constant>, O::To<Self::Constant>> =
            builder.build(output_atom_ids, structure, output_structure.clone())?;
        // The replay folds through `self` directly: an eager domain interprets each instruction immediately,
        // while a staging context stages it into its enclosing trace.
        let output_values = replay_via_bind(self, &program, input_values)?;
        Ok(O::To::<Self::Value>::from_parameters(output_structure, output_values)?)
    }
}

impl<D: Domain<Type = ArrayType>> Batch for D {}

/// Extension trait that exposes batching as a method on active array contexts.
///
/// This is the already-traced counterpart of [`Batch`]. It wraps the receiver in a [`BatchingContext`] and routes all
/// primitive binds through the current transform stack, so `batch` composes with tracing, JVP, VJP, and other context
/// wrappers through the same [`StagingContext::stage_operation`] path.
///
/// The receiver flows its own [`Tracer`] metadata (its [`StagingContext::Meta`]), so this trait is implemented for
/// plain tracing contexts ([`Meta`](StagingContext::Meta)` = ()`), for batching contexts
/// ([`Meta`](StagingContext::Meta)` = `[`BatchAxis`], which is what makes nested `vmap` work), and for any other
/// staging context.
pub trait BatchContext: StagingContext<Type = ArrayType> {
    /// Maps a traced function over per-leaf array axes inside this active context. The `in_axes` and `out_axes`
    /// parameters accept anything convertible to [`BatchAxesSpecification`] (explicit per-leaf [`BatchAxis`] axes or
    /// one uniform leaf specification), and the `axis` parameter accepts anything convertible to a
    /// [`BatchAxisSpecification`] (an optional explicit lane size and an optional axis name).
    fn batch<F, I, O>(
        &self,
        function: F,
        input: I,
        in_axes: impl Into<BatchAxesSpecification<I::To<BatchAxis>>>,
        out_axes: impl Into<BatchAxesSpecification<O::To<BatchAxis>>>,
        axis: impl Into<BatchAxisSpecification>,
    ) -> Result<O::To<Tracer<Self, Self::Meta>>, ProgramError>
    where
        Self::Operation:
            Clone + From<TransposeOperation> + BatchableOperation<Tracer<Self, Self::Meta>, BatchingContext<Self>>,
        I: Parameterized<
                Tracer<Self, Self::Meta>,
                ParameterStructure: Debug + PartialEq,
                Family: ParameterizedFamily<ArrayType>
                            + ParameterizedFamily<Self::Constant>
                            + ParameterizedFamily<BatchAxis>
                            + ParameterizedFamily<Tracer<Self, Self::Meta>>
                            + ParameterizedFamily<BatchingTracer<Self>>,
            >,
        O: Parameterized<
                BatchingTracer<Self>,
                ParameterStructure: Debug + PartialEq,
                Family: ParameterizedFamily<ArrayType>
                            + ParameterizedFamily<Self::Constant>
                            + ParameterizedFamily<BatchAxis>
                            + ParameterizedFamily<Tracer<Self, Self::Meta>>
                            + ParameterizedFamily<BatchingTracer<Self>>,
            >,
        I::To<BatchAxis>: Parameterized<BatchAxis, ParameterStructure = I::ParameterStructure>,
        O::To<BatchAxis>: Parameterized<BatchAxis, ParameterStructure = O::ParameterStructure>,
        I::To<ArrayType>: Parameterized<
                ArrayType,
                To<Self::Constant> = I::To<Self::Constant>,
                To<BatchingTracer<Self>> = I::To<BatchingTracer<Self>>,
            >,
        O::To<ArrayType>: Parameterized<
                ArrayType,
                To<Tracer<Self, Self::Meta>> = O::To<Tracer<Self, Self::Meta>>,
                To<BatchingTracer<Self>> = O,
            >,
        F: FnOnce(I::To<BatchingTracer<Self>>) -> Result<O, ProgramError>,
    {
        let axis = axis.into();
        let parent_context = self.clone();
        let input_structure = input.parameter_structure();
        let input_tracers = input.into_parameters().collect::<Vec<_>>();
        let in_axes_values = match in_axes.into() {
            BatchAxesSpecification::Uniform(leaf_axis) => vec![leaf_axis; input_tracers.len()],
            BatchAxesSpecification::PerLeaf(axes) => {
                let in_axes_structure = axes.parameter_structure();
                if in_axes_structure != input_structure {
                    return Err(ParameterError::MismatchedParameterStructures {
                        left_structure: format!("{input_structure:?}"),
                        right_structure: format!("{in_axes_structure:?}"),
                    }
                    .into());
                }
                axes.into_parameters().collect::<Vec<_>>()
            }
        };
        if input_tracers.is_empty() && axis.size.is_none() {
            return Err(BatchingError::EmptyBatch.into());
        }

        let mut resolved_axis_size = axis.size;
        let mut inputs_with_axes = Vec::with_capacity(input_tracers.len());
        for (tracer, axis) in input_tracers.into_iter().zip(in_axes_values.iter().copied()) {
            let parent_physical_type = tracer.r#type().into_owned();
            match axis.axis() {
                Some(batch_axis) => {
                    let (per_lane_type, dimension) = parent_physical_type.without_dimension(batch_axis)?;
                    let Some(size) = dimension.value() else {
                        return Err(
                            BatchingError::DynamicBatchAxis { r#type: parent_physical_type, axis: batch_axis }.into()
                        );
                    };
                    match resolved_axis_size {
                        Some(existing_size) if existing_size != size => {
                            return Err(
                                BatchingError::MismatchedBatchSizes { expected: existing_size, actual: size }.into()
                            );
                        }
                        Some(_) => {}
                        None => resolved_axis_size = Some(size),
                    }
                    inputs_with_axes.push((tracer, BatchAxis::mapped(batch_axis), per_lane_type));
                }
                None => {
                    inputs_with_axes.push((tracer, BatchAxis::uniform(), parent_physical_type));
                }
            }
        }
        let resolved_axis_size = resolved_axis_size.ok_or(BatchingError::EmptyBatch)?;

        let batching_context = BatchingContext::with_axis_name(parent_context.clone(), resolved_axis_size, axis.name);
        let parent_builder = parent_context.builder().clone();

        let mut batched_input_values = Vec::with_capacity(inputs_with_axes.len());
        for (parent_tracer, axis, logical_type) in inputs_with_axes.iter() {
            let atom = parent_tracer.atom_id()?;
            // Prepend this level's mapped axis onto the *incoming* value's existing `Meta` stack: the head is this
            // level's axis and the tail is the outer value's stack, so a value already mapped by an enclosing `batch`
            // keeps every outer level's axis. A fresh program input simply rides the parent context's default stack.
            batched_input_values.push(Tracer::new_with_meta(
                batching_context.clone(),
                TracerState::Live(atom),
                logical_type.clone(),
                (*axis, parent_tracer.meta().clone()),
            ));
        }
        let batched_input = I::To::<BatchingTracer<Self>>::from_parameters(input_structure, batched_input_values)?;
        let batched_output =
            function(batched_input).map_err(|error| parent_builder.borrow_mut().error.take().unwrap_or(error))?;
        parent_builder.borrow_mut().error.take().map_or(Ok(()), Err)?;

        let output_structure = batched_output.parameter_structure();
        // Each output's mapped lane axis at this level is the head of its `Meta` stack (`value.meta().0`), and the
        // tail (`value.meta().1`) is the parent context's per-level axes for that staged atom. Carrying that tail into
        // the re-wrapped parent value is what threads an enclosing `batch`'s axis through this one for nested `vmap`:
        // the outer driver then reads its own axis straight off this value's stack head with no side table.
        let outputs = batched_output
            .parameters()
            .map(|value| Ok((value.atom_id()?, value.meta().0.axis(), value.meta().1.clone())))
            .collect::<Result<Vec<_>, ProgramError>>()?;
        let out_axes_values = match out_axes.into() {
            BatchAxesSpecification::Uniform(leaf_axis) => vec![leaf_axis; outputs.len()],
            BatchAxesSpecification::PerLeaf(axes) => {
                let out_axes_structure = axes.parameter_structure();
                if out_axes_structure != output_structure {
                    return Err(ParameterError::MismatchedParameterStructures {
                        left_structure: format!("{output_structure:?}"),
                        right_structure: format!("{out_axes_structure:?}"),
                    }
                    .into());
                }
                axes.into_parameters().collect::<Vec<_>>()
            }
        };
        drop(batched_output);
        drop(batching_context);

        let parent_outputs = outputs
            .into_iter()
            .zip(out_axes_values.iter().map(|axis| axis.axis()))
            .map(
                |((atom, current_axis, parent_meta), expected_axis)| -> Result<Tracer<Self, Self::Meta>, ProgramError> {
                    let physical_type = parent_context.builder().borrow().atoms()[atom.index()].r#type().into_owned();
                    let parent_tracer = Tracer::new_with_meta(
                        parent_context.clone(),
                        TracerState::Live(atom),
                        physical_type,
                        parent_meta,
                    );
                    match (current_axis, expected_axis) {
                        (None, None) => Ok(parent_tracer),
                        // The output's mapped-axis presence disagrees with the caller's `out_axes` declaration.
                        // Collapsing a mapped output requires an explicit reduction inside the batched function, and
                        // materializing a missing axis requires an explicit broadcast; position-only disagreements are
                        // instead repaired with the staged transpose in the arm below.
                        (None, Some(_)) | (Some(_), None) => {
                            Err(BatchingError::MismatchedOutputAxes { expected: expected_axis, actual: current_axis }
                                .into())
                        }
                        (Some(current), Some(expected)) if current == expected => Ok(parent_tracer),
                        (Some(current), Some(expected)) => {
                            let rank = parent_tracer.r#type().as_ref().rank();
                            let permutation = move_axis_permutation(rank, current, expected);
                            parent_tracer.transpose(permutation)
                        }
                    }
                },
            )
            .collect::<Result<Vec<_>, ProgramError>>()?;

        Ok(O::To::<Tracer<Self, Self::Meta>>::from_parameters(output_structure, parent_outputs)?)
    }
}

impl<C> BatchContext for C where C: StagingContext<Type = ArrayType> {}

/// Returns the axis permutation that moves dimension `from` to position `to`, shifting the other
/// dimensions to preserve their relative order. Returns the identity permutation when
/// `from == to`.
pub(crate) fn move_axis_permutation(rank: usize, from: usize, to: usize) -> Vec<usize> {
    let mut permutation: Vec<usize> = (0..rank).collect();
    let axis = permutation.remove(from);
    permutation.insert(to, axis);
    permutation
}
