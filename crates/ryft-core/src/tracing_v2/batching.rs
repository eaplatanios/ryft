use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::convert::Infallible;
use std::fmt::{Debug, Display};
use std::rc::Rc;

use crate::ElementwiseOperation;
use crate::batching::BatchingError;
use crate::broadcasting::Broadcastable;
use crate::contexts::{Context, StagingContext};
use crate::domains::{AbstractDomain, Domain};
use crate::macros::check_count;
use crate::operations::manipulation::{Broadcast, SupportsTranspose, Transpose};
use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily};
use crate::programs::{AtomId, Program, ProgramBuilder, ProgramError, Value};
use crate::tracing::{DomainTracer, Tracer, TracerState, TracingContext};
use crate::tracing_v2::differentiation::DifferentiationContext;
use crate::types::{ArrayType, Size, Typed};

/// Maps a traced `function` over array axes selected per leaf by `in_axes` and places each output's mapped axis at
/// the position requested by `out_axes`. This is the module-level equivalent of [`Batch::batch`]; refer to its
/// documentation for the full semantics.
///
/// # Parameters
///
///   - `domain`: [`Domain`] that provides the traced operation, type, and constant representations.
///   - `function`: Function/closure to map over the batched lanes.
///   - `input`: Input value whose mapped leaves drive the batched lanes.
///   - `in_axes`: Per-leaf mapped-axis selection for `input`, or one uniform leaf specification.
///   - `out_axes`: Per-leaf mapped-axis placement for the output, or one uniform leaf specification.
///   - `axis`: Mapped-axis specification carrying an optional explicit lane size and an optional axis name.
#[inline]
pub fn batch<'domain, D, F, I, O>(
    domain: &'domain D,
    function: F,
    input: I,
    in_axes: impl Into<BatchAxes<I::To<Option<usize>>>>,
    out_axes: impl Into<BatchAxes<O::To<Option<usize>>>>,
    axis: impl Into<BatchAxis>,
) -> Result<O::To<D::Value>, BatchingError>
where
    D: Context<Type = ArrayType> + 'domain,
    D::Value: 'domain,
    D::Constant: 'domain,
    I: Parameterized<
            D::Value,
            ParameterStructure: Debug + PartialEq,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<D::Constant>
                        + ParameterizedFamily<Option<usize>>
                        + ParameterizedFamily<DomainTracer<'domain, D>>
                        + ParameterizedFamily<BatchingTracer<'domain, D>>,
        >,
    O: Parameterized<
            BatchingTracer<'domain, D>,
            ParameterStructure: Debug + PartialEq,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<D::Value>
                        + ParameterizedFamily<D::Constant>
                        + ParameterizedFamily<Option<usize>>
                        + ParameterizedFamily<DomainTracer<'domain, D>>
                        + ParameterizedFamily<BatchingTracer<'domain, D>>,
        >,
    I::To<Option<usize>>: Parameterized<Option<usize>, ParameterStructure = I::ParameterStructure>,
    O::To<Option<usize>>: Parameterized<Option<usize>, ParameterStructure = O::ParameterStructure>,
    I::To<DomainTracer<'domain, D>>: Parameterized<
            DomainTracer<'domain, D>,
            ParameterStructure = I::ParameterStructure,
            To<ArrayType> = I::To<ArrayType>,
            To<D::Value> = I,
            To<D::Constant> = I::To<D::Constant>,
            To<Option<usize>> = I::To<Option<usize>>,
            To<BatchingTracer<'domain, D>> = I::To<BatchingTracer<'domain, D>>,
        >,
    O::To<DomainTracer<'domain, D>>: Parameterized<
            DomainTracer<'domain, D>,
            ParameterStructure = O::ParameterStructure,
            To<ArrayType> = O::To<ArrayType>,
            To<D::Value> = O::To<D::Value>,
            To<D::Constant> = O::To<D::Constant>,
            To<Option<usize>> = O::To<Option<usize>>,
            To<BatchingTracer<'domain, D>> = O,
        >,
    I::To<ArrayType>: Parameterized<
            ArrayType,
            To<D::Value> = I,
            To<D::Constant> = I::To<D::Constant>,
            To<DomainTracer<'domain, D>> = I::To<DomainTracer<'domain, D>>,
            To<BatchingTracer<'domain, D>> = I::To<BatchingTracer<'domain, D>>,
        >,
    O::To<ArrayType>: Parameterized<
            ArrayType,
            To<D::Value> = O::To<D::Value>,
            To<D::Constant> = O::To<D::Constant>,
            To<DomainTracer<'domain, D>> = O::To<DomainTracer<'domain, D>>,
            To<BatchingTracer<'domain, D>> = O,
        >,
    D::Operation: Clone
        + InterpretableOperation<ArrayType, D::Value>
        + SupportsTranspose<ArrayType>
        + for<'context> BatchableOperation<DomainTracer<'context, D>, BatchingContext<TracingContext<'context, D>>>,
    F: FnOnce(I::To<BatchingTracer<'domain, D>>) -> Result<O, ProgramError>,
{
    domain.batch(function, input, in_axes, out_axes, axis)
}

/// Packed array value carrying lane metadata for one batching transform.
///
/// [`ArrayBatch`] is the production batching representation for `tracing_v2`: its [`ArrayType`] is the
/// physical type of `value`, so it includes the mapped lane dimension when [`ArrayBatch::batch_axis`]
/// is `Some`. The logical per-lane type is derived by removing that dimension.
///
/// A `None` batch axis is an explicit lane-uniform state. It means the value does not contain a
/// physical dimension for the current batch lanes and should be interpreted as the same value for
/// every lane. For example, a traced constant in `batch(|x| x + 1)` is represented with
/// `batch_axis == None`, while `x` carries the mapped input axis. Runtime control-flow predicates
/// also require `None` today because a single predicate can select one branch for all lanes, while
/// a lane-varying predicate would need a dedicated batching rule. `None` is not limited to
/// rank-0 values: any shaped constant or operand can be lane-uniform when none of its physical
/// dimensions indexes the current lanes.
#[derive(Clone, Debug, PartialEq)]
pub struct ArrayBatch<V: Typed<ArrayType>> {
    /// Physical array type of `value`.
    r#type: ArrayType,

    /// Packed array value.
    value: V,

    /// Axis in `r#type` and `value` that represents the mapped batch dimension, or `None` when
    /// `value` is uniform across the current batch lanes.
    batch_axis: Option<usize>,
}

impl<V: Typed<ArrayType>> ArrayBatch<V> {
    /// Creates a packed array batch from explicit physical metadata.
    ///
    /// # Parameters
    ///
    ///   - `r#type`: Physical type of `value`. This type includes `batch_axis` when present.
    ///   - `value`: Physical array value.
    ///   - `batch_axis`: Mapped axis in `r#type` and `value`, or `None` when `value` is shared
    ///     uniformly across lanes.
    pub fn new(r#type: ArrayType, value: V, batch_axis: Option<usize>) -> Result<Self, ProgramError> {
        if let Some(axis) = batch_axis
            && axis >= r#type.rank()
        {
            return Err(BatchingError::BatchAxisOutOfBounds { r#type, axis }.into());
        }
        Ok(Self { r#type, value, batch_axis })
    }

    /// Returns the mapped axis, if the physical value carries one.
    #[inline]
    pub fn batch_axis(&self) -> Option<usize> {
        self.batch_axis
    }

    /// Returns the physical value.
    #[inline]
    pub fn value(&self) -> &V {
        &self.value
    }

    /// Consumes `self` and returns the physical value.
    #[inline]
    pub fn into_value(self) -> V {
        self.value
    }

    /// Returns the static mapped axis size, if this value is batched.
    pub fn axis_size(&self) -> Result<Option<usize>, ProgramError> {
        let Some(axis) = self.batch_axis else {
            return Ok(None);
        };
        let Some(size) = self.r#type.dimension(axis as isize).value() else {
            return Err(BatchingError::DynamicBatchAxis { r#type: self.r#type.clone(), axis }.into());
        };
        Ok(Some(size))
    }

    /// Returns the scalar-body type obtained by removing the mapped axis.
    pub fn logical_type(&self) -> Result<ArrayType, ProgramError> {
        let Some(axis) = self.batch_axis else {
            return Ok(self.r#type.clone());
        };
        Ok(self.r#type.without_dimension(axis)?.0)
    }

    /// Wraps a value that already contains a mapped axis.
    ///
    /// # Parameters
    ///
    ///   - `value`: Packed array value.
    ///   - `batch_axis`: Mapped axis in `value`.
    pub fn mapped(value: V, batch_axis: usize) -> Result<Self, ProgramError> {
        Self::new(value.r#type().into_owned(), value, Some(batch_axis))
    }

    /// Wraps a value that is uniform across the current batch lanes.
    pub fn unbatched(value: V) -> Self {
        Self { r#type: value.r#type().into_owned(), value, batch_axis: None }
    }
}

impl<V: Typed<ArrayType> + Parameter> Parameter for ArrayBatch<V> {}

impl<V: Typed<ArrayType>> Typed<ArrayType> for ArrayBatch<V> {
    #[inline]
    fn r#type(&self) -> Cow<'_, ArrayType> {
        Cow::Borrowed(&self.r#type)
    }
}

impl<V: Display + Typed<ArrayType>> Display for ArrayBatch<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.batch_axis {
            Some(axis) => write!(formatter, "batch[{}, axis={axis}]({})", self.r#type, self.value),
            None => write!(formatter, "batch[{}, lane-uniform]({})", self.r#type, self.value),
        }
    }
}

impl<V: Value<ArrayType>> Value<ArrayType> for ArrayBatch<V> {}

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
///   - **Symbolic-zero short-circuit.** When `V` is
///     [`Tangent<ArrayType, V'>`](crate::differentiation::Tangent), the rule on
///     [`LinearArrayOperation`](crate::tracing_v2::LinearArrayOperation) materializes `Tangent::Zero` once via
///     `V::zero` and dispatches to
///     the underlying V-level rule. Tangent's V-trait impls (`Add`, `Sub`, `Scale`, `Neg`,
///     `LeftDot`, `RightDot`, `Reshape`, `Transpose`) propagate `Zero` correctly, so the
///     short-circuit happens automatically.
///   - **Missing rule.** Variants without a defined batching rule (for example, a while-loop
///     whose loop predicate varies across lanes) return [`BatchingError::UnsupportedOperation`]
///     with a human-readable message that points at a likely fix.
///
/// The internal elementwise lifting helper computes the lifted op plus per-output axes for any pure elementwise op;
/// the matching value-level applicator composes the rule's axis arithmetic with [`InterpretableOperation::interpret`].
pub trait BatchableOperation<V: Value<ArrayType>, C = ()>: Operation<ArrayType> {
    /// Applies this operation to packed batched inputs, returning batched outputs with the
    /// resulting lane axes, using `context` for rules that need active transform state.
    fn batch(&self, context: &C, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, ProgramError>;
}

impl<V: Value<ArrayType>, C> BatchableOperation<V, C> for Infallible {
    fn batch(&self, _context: &C, _inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        match *self {}
    }
}

/// Blanket [`BatchableOperation`] impl for any [`ElementwiseOperation`].
///
/// Mirrors the existing
/// [`impl<O: ElementwiseOperation> Operation<ArrayType> for O`](crate::operations::Operation):
/// every elementwise primitive automatically gets the standard elementwise batching rule, so per-op
/// `BatchableOperation` impls do not have to be written for elementwise primitives (`Add`, `Sub`, `Mul`, `Div`,
/// `Neg`, `Sin`, `Cos`, `Scale`, …). Ops with non-trivial axis arithmetic (`Dot`, `Transpose`, `Reshape`, …) and the
/// operation enums ([`ArrayOperation`](crate::tracing_v2::ArrayOperation) and
/// [`LinearArrayOperation`](crate::tracing_v2::LinearArrayOperation), whose impls live with the enums in
/// [`operations::primitive`](crate::tracing_v2::operations::primitive)) keep their explicit impls; coherence is
/// preserved because none of those types implement [`ElementwiseOperation`].
impl<O, V, C> BatchableOperation<V, C> for O
where
    O: ElementwiseOperation + Clone + InterpretableOperation<ArrayType, V>,
    V: Value<ArrayType> + Broadcast<Output = V> + Transpose,
{
    #[inline]
    fn batch(&self, _context: &C, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        apply_elementwise_batch(self, inputs)
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

/// Applies a lifted operation to `inputs` via [`InterpretableOperation::interpret`] and packages
/// each output value with the corresponding entry of `output_axes`.
///
/// `output_axes` must have one entry per output produced by `lifted_op` on these inputs.
pub(crate) fn apply_with_axes<V: Value<ArrayType>, O>(
    lifted_op: &O,
    inputs: &[ArrayBatch<V>],
    output_axes: &[Option<usize>],
) -> Result<Vec<ArrayBatch<V>>, ProgramError>
where
    O: InterpretableOperation<ArrayType, V>,
{
    let input_values: Vec<V> = inputs.iter().map(|input| input.value().clone()).collect();
    let output_values = lifted_op.interpret(input_values.as_slice())?;
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
/// See [`broadcast_inputs_to_common`] for the exact broadcasting policy.
pub(crate) fn apply_elementwise_batch<V: Value<ArrayType> + Broadcast<Output = V> + Transpose, O>(
    operation: &O,
    inputs: &[ArrayBatch<V>],
) -> Result<Vec<ArrayBatch<V>>, ProgramError>
where
    O: Clone + InterpretableOperation<ArrayType, V>,
{
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
        Some(batch_axis) => broadcast_inputs_to_common(&aligned_inputs, &input_axes, batch_axis, axis_size)?,
    };
    apply_with_axes(&lifted_op, &broadcasted_inputs, &output_axes)
}

/// Broadcasts the operands of an elementwise application to the common batched physical shape.
///
/// Mirroring JAX's `defbroadcasting` policy, every operand whose per-lane shape is narrower than
/// the common per-lane shape of all operands (trailing-aligned) is broadcast to that common shape
/// with the lane axis at `batch_axis`. This covers both lane-uniform operands (a rank-0 bias
/// batched against rank-2 lanes becomes a full `[lanes, ...]` operand rather than a bare `[lanes]`
/// vector) and mapped operands with differing per-lane ranks (a mapped per-lane scalar batched
/// against mapped per-lane vectors gains the vector dimensions). When the operands' per-lane
/// shapes are not broadcast-compatible, the operands are left at their lane-axis-inserted
/// physical shapes so the operation surfaces its own shape error against the original shapes.
///
/// Narrower operands are always materialized with an explicit [`BroadcastOperation`], even
/// when the operation's own type inference would accept the unbroadcast shapes: staged programs
/// must remain lowerable by every backend, and backends such as XLA lower elementwise operations
/// to shape-congruent primitives (e.g., [`stablehlo.add`](
/// https://openxla.org/stablehlo/spec#add)) with no implicit broadcasting.
fn broadcast_inputs_to_common<V: Value<ArrayType> + Broadcast<Output = V>>(
    inputs: &[ArrayBatch<V>],
    input_axes: &[Option<usize>],
    batch_axis: usize,
    axis_size: usize,
) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
    // Common per-lane broadcast target shared by all operands (mapped inputs drop their lane axis;
    // lane-uniform inputs contribute their full type). `None` when the per-lane shapes are not
    // broadcast-compatible, in which case the operation reports its own error downstream.
    let per_lane_types = inputs.iter().map(|input| input.logical_type()).collect::<Result<Vec<_>, _>>()?;
    let common_per_lane = Broadcastable::broadcasted(per_lane_types.as_slice()).ok();
    let broadcasted_physical_type = |per_lane_type: &ArrayType| -> Result<ArrayType, ProgramError> {
        let target = common_per_lane.as_ref().unwrap_or(per_lane_type);
        Ok(target.with_inserted_dimension(batch_axis, Size::Static(axis_size))?)
    };
    // Maps the operand's per-lane dimension `index` (trailing-aligned within the common per-lane
    // shape) to its position in the broadcast target, accounting for the lane axis insertion.
    let target_position = |per_lane_rank: usize, target_rank: usize, index: usize| {
        let position = (target_rank - per_lane_rank) + index;
        if position < batch_axis { position } else { position + 1 }
    };
    inputs
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
                // Mapped operand with a narrower per-lane shape: keep the lane axis fixed and
                // trailing-align the remaining dimensions.
                Some(_) => (0..input.r#type().rank())
                    .map(|dimension| match dimension.cmp(&batch_axis) {
                        std::cmp::Ordering::Equal => batch_axis,
                        std::cmp::Ordering::Less => target_position(per_lane_type.rank(), target_rank, dimension),
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
        .collect()
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
    let permuted_value = input.value().clone().transpose(permutation);
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
/// batched shape via [`broadcast_lane_uniform_inputs`].
///
/// Returns an error when called on an already-batched input — callers are expected to dispatch
/// the lane-uniform case explicitly.
///
/// # Parameters
///
///   - `operand`: Lane-uniform input to lift.
///   - `target_axis`: Position of the inserted batch axis in the output.
///   - `axis_size`: Size of the inserted batch axis.
pub(crate) fn broadcast_to_batched<V: Value<ArrayType> + Broadcast<Output = V>>(
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
    // `SelectOperation`), infer against the common per-lane shape with the lane axis inserted,
    // matching the operand shapes that [`broadcast_inputs_to_common`] produces. Ops with built-in
    // broadcasting semantics (e.g., `AddOperation`) accept the broadcasted shapes equally. When
    // the per-lane shapes are not broadcast-compatible, fall back to the lane-axis-inserted
    // physical types so the operation surfaces its own shape error.
    let broadcasted_input_types: Vec<ArrayType> = match (common_axis, Broadcastable::broadcasted(input_types)) {
        (Some(axis), Ok(common)) => {
            vec![common.with_inserted_dimension(axis, Size::Static(axis_size))?; input_types.len()]
        }
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

/// Staging context used by [`batch_flat_program`] to capture a batched program replay: an ordinary trace over the
/// zero-sized [`AbstractDomain`] token for the program's `(V, O)` universe.
///
/// The `'static` lifetime keeps every trait bound mentioning this context lifetime-free, which is what lets the
/// trait solver close the recursive cycle between the custom-derivative re-wrapping `batch` rules and the closed
/// operation enums' [`SupportsProgramBatching`] impls (a higher-ranked lifetime here would re-instantiate the
/// cycle's goals in fresh placeholder universes and overflow). It is honest, not a hack: [`AbstractDomain`] is a
/// zero-sized behavior-free token, so a `&'static` borrow of it is materialized for free. The capture parameter
/// is pinned to `V` explicitly (rather than left at its `D::Value` projection default) so that bounds written
/// against this alias match their obligations syntactically.
pub type ProgramBatchingContext<V, O> = TracingContext<'static, AbstractDomain<ArrayType, V, O>, V>;

/// Operation types whose captured flat programs can be batched into standalone lane-carrying programs.
///
/// This is the program-level counterpart of [`BatchableOperation`], implemented by closed operation enums (via
/// [`batch_flat_program`]) so that higher-order operations can batch the programs they capture. The re-wrapping
/// `batch` rules of [`CustomJvpOperation`](crate::tracing_v2::operations::custom_derivatives::CustomJvpOperation)
/// and [`CustomVjpOperation`](crate::tracing_v2::operations::custom_derivatives::CustomVjpOperation) bound their
/// captured-program operation type by this trait. Routing program-level batching through a dedicated, lifetime-free
/// trait keeps the trait solver's recursion finite: the closed enum impl discharges the derived batching-context
/// obligations once, against the single [`ProgramBatchingContext`] type, instead of every batching rule re-deriving
/// them with fresh higher-ranked lifetimes (which defeats the solver's cycle detection and overflows).
pub trait SupportsProgramBatching<V: Value<ArrayType>>: Operation<ArrayType> + Sized {
    /// Batches `program` into a standalone program over lane-carrying physical types; refer to the documentation
    /// of [`batch_flat_program`] for the input/output axis conventions.
    fn batch_flat_program(
        program: &crate::tracing_v2::operations::control_flow::FlatProgram<V, Self>,
        axis_size: usize,
    ) -> Result<crate::tracing_v2::operations::control_flow::FlatProgram<V, Self>, ProgramError>;
}

/// Batches a captured flat program into a standalone program over lane-carrying physical types.
///
/// The returned program consumes every original input with a mapped lane axis of size `axis_size` inserted at
/// position `0` and produces every original output with the lane axis at position `0` (lane-uniform results are
/// broadcast across the lane). Instructions are lifted through their [`BatchableOperation`] rules by replaying the
/// program through a fresh [`BatchingContext`] over a [`ProgramBatchingContext`] trace, so multi-operation rewrites
/// (for example, lane-varying conditionals) compose exactly as they do under [`batch`].
///
/// This is the program-level batching primitive behind [`SupportsProgramBatching`]: batching every captured
/// program against a uniform all-inputs-mapped-at-`0` convention keeps the batched programs' signatures mutually
/// consistent, so custom-derivative operations can be re-wrapped instead of inlined and the custom derivative
/// survives `batch`.
pub fn batch_flat_program<V, O>(
    program: &crate::tracing_v2::operations::control_flow::FlatProgram<V, O>,
    axis_size: usize,
) -> Result<crate::tracing_v2::operations::control_flow::FlatProgram<V, O>, ProgramError>
where
    V: Value<ArrayType> + 'static,
    O: Clone + Operation<ArrayType> + 'static,
    O: BatchableOperation<Tracer<ProgramBatchingContext<V, O>>, BatchingContext<ProgramBatchingContext<V, O>>>,
    Tracer<ProgramBatchingContext<V, O>>: Broadcast<Output = Tracer<ProgramBatchingContext<V, O>>> + Transpose,
{
    use crate::parameters::Placeholder;
    use crate::tracing_v2::operations::control_flow::flat_program_input_types;

    let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
    // `AbstractDomain` is a zero-sized token, so leaking one boxed instance materializes the `'static` borrow that
    // `ProgramBatchingContext` requires without allocating.
    let domain: &'static AbstractDomain<ArrayType, V, O> = Box::leak(Box::new(AbstractDomain::new()));
    let parent_context = TracingContext::new(domain, builder.clone());
    let logical_input_types = flat_program_input_types(program);
    let input_count = logical_input_types.len();
    // Keep every tracer and context that holds a clone of `builder` inside this scope so that recovering the
    // builder below is a real ownership check.
    let (output_atom_ids, output_count) = {
        let batching_context = BatchingContext::new(parent_context, axis_size);
        let mut input_tracers = Vec::with_capacity(input_count);
        for logical_type in &logical_input_types {
            let physical_type = logical_type.with_inserted_dimension(0, Size::Static(axis_size))?;
            let atom = builder.borrow_mut().add_input(physical_type);
            batching_context.register_axis(atom, Some(0));
            input_tracers.push(batching_context.tracer(atom, Some(logical_type.clone())));
        }
        let output_tracers = batching_context.stage_program(program, input_tracers)?;
        let output_count = output_tracers.len();
        let mut output_atom_ids = Vec::with_capacity(output_count);
        for output_tracer in output_tracers {
            let atom = output_tracer.atom_id()?;
            let axis = batching_context.axis_for(atom);
            let logical_type = output_tracer.r#type().into_owned();
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
                Some(0) => parent_batch,
                Some(_) => align_batch_axis(&parent_batch, 0)?,
                None => broadcast_to_batched(&parent_batch, 0, axis_size)?,
            };
            output_atom_ids.push(aligned_batch.into_value().atom_id()?);
        }
        (output_atom_ids, output_count)
    };
    let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
    builder
        .build(output_atom_ids, vec![Placeholder; input_count], vec![Placeholder; output_count])?
        .into_simplified()
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

    /// Per-atom batch-axis annotations for atoms staged through this batching level. Missing keys
    /// are treated as lane-uniform (axis = `None`). Lives in interior mutability because the
    /// [`StagingContext::stage_operation`] hook takes `&self` but needs to record output axes as it
    /// stages instructions into the parent [`ProgramBuilder`].
    axis_table: Rc<RefCell<HashMap<AtomId, usize>>>,
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
        Self { parent_context, axis_size, axis_name, axis_table: Rc::new(RefCell::new(HashMap::new())) }
    }

    /// Registers an explicit batch axis annotation for the given [`AtomId`]. Passing `None`
    /// removes any existing annotation (the atom is then treated as lane-uniform).
    pub fn register_axis(&self, atom: AtomId, axis: Option<usize>) {
        let mut table = self.axis_table.borrow_mut();
        match axis {
            Some(k) => {
                table.insert(atom, k);
            }
            None => {
                table.remove(&atom);
            }
        }
    }

    /// Returns the batch axis annotation for the given [`AtomId`], or `None` if the atom is
    /// lane-uniform (no entry in the table).
    pub fn axis_for(&self, atom: AtomId) -> Option<usize> {
        self.axis_table.borrow().get(&atom).copied()
    }

    /// Interprets a captured flat program while staging batched primitive calls into the parent context.
    pub(crate) fn interpret_program<O>(
        &self,
        program: &Program<ArrayType, C::Constant, O, Vec<C::Constant>, Vec<C::Constant>>,
        inputs: Vec<ArrayBatch<Tracer<C>>>,
    ) -> Result<Vec<ArrayBatch<Tracer<C>>>, ProgramError>
    where
        O: BatchableOperation<Tracer<C>, Self>,
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
            axis_table: self.axis_table.clone(),
        }
    }
}

impl<C> Domain for BatchingContext<C>
where
    C: StagingContext<Type = ArrayType>,
    C::Operation: BatchableOperation<Tracer<C>, Self>,
{
    type Type = ArrayType;
    type Value = Tracer<Self>;
    type Constant = C::Constant;
    type Operation = C::Operation;
}

impl<C> Context for BatchingContext<C>
where
    C: StagingContext<Type = ArrayType>,
    C::Operation: BatchableOperation<Tracer<C>, Self>,
{
    /// Lifts a constant payload into this batching context by recording it as a constant [`Tracer`].
    #[inline]
    fn lift(&self, constant: C::Constant) -> Result<Tracer<Self>, ProgramError> {
        Ok(self.constant(constant))
    }

    /// Binding in a batching context routes through [`StagingContext::stage_operation`], which lifts the operation over
    /// each input's recorded batch axis through the operation's [`BatchableOperation`] rule.
    #[inline]
    fn bind(&self, operation: C::Operation, inputs: &[Tracer<Self>]) -> Result<Vec<Tracer<Self>>, ProgramError> {
        self.stage_operation(operation, inputs)
    }
}

impl<C> StagingContext for BatchingContext<C>
where
    C: StagingContext<Type = ArrayType>,
    C::Operation: BatchableOperation<Tracer<C>, Self>,
{
    #[inline]
    fn builder(&self) -> &Rc<RefCell<ProgramBuilder<Self::Type, Self::Constant, Self::Operation>>> {
        self.parent_context.builder()
    }

    fn stage_operation<I: std::borrow::Borrow<Tracer<Self>>>(
        &self,
        operation: Self::Operation,
        inputs: &[I],
    ) -> Result<Vec<Tracer<Self>>, ProgramError> {
        if inputs.iter().any(|input| !Rc::ptr_eq(self.builder(), input.borrow().context().builder())) {
            return Err(self.error(ProgramError::MismatchedProgramBuilders));
        }
        if self.builder().borrow().error.is_some() {
            let input_types = inputs.iter().map(|input| input.borrow().r#type().into_owned()).collect::<Vec<_>>();
            let output_types = operation.infer_output_types(input_types.as_slice())?;
            return Ok(output_types
                .into_iter()
                .map(|r#type| Tracer::new(TracerState::Poison, r#type, self.clone()))
                .collect());
        }

        // Zero-input operations (e.g., `ZeroOperation`, `OneOperation`) are lane-uniform by
        // construction: every batch lane receives the same constant value, and there is no input
        // batch axis to lift through. Stage them directly into the parent's builder with an empty
        // input list and surface the resulting parent atoms as lane-uniform tracers (no entry in
        // `axis_table`).
        if inputs.is_empty() {
            let parent_outputs = self.parent_context.stage_operation::<&Tracer<C>>(operation, &[])?;
            return Ok(parent_outputs
                .into_iter()
                .map(|parent_tracer| -> Result<Tracer<Self>, ProgramError> {
                    let parent_physical_type = parent_tracer.r#type().into_owned();
                    let atom = parent_tracer.atom_id()?;
                    Ok(self.tracer(atom, Some(parent_physical_type)))
                })
                .collect::<Result<Vec<_>, _>>()?);
        }

        let input_atom_ids: Vec<AtomId> =
            match inputs.iter().map(|input| input.borrow().atom_id()).collect::<Result<_, _>>() {
                Ok(ids) => ids,
                Err(error) => return Err(self.error(error)),
            };
        let logical_input_types: Vec<ArrayType> =
            inputs.iter().map(|input| input.borrow().r#type().into_owned()).collect();
        let input_axes: Vec<Option<usize>> = input_atom_ids.iter().map(|atom| self.axis_for(*atom)).collect();

        // Build parent-level input batches. Each ArrayBatch wraps the same atom as a parent trace
        // value at the parent-physical (= this level's physical) type, with its recorded batch
        // axis. The rule's body (`operation.batch(...)`) then dispatches through that parent
        // value's primitive impls, so each primitive call inside the rule stages directly into the
        // parent context. Multi-op staging (e.g., lane-varying Condition lowering to two branches
        // + a per-lane Select) emerges automatically.
        let mut parent_input_batches: Vec<ArrayBatch<Tracer<C>>> = Vec::with_capacity(inputs.len());
        for ((atom, logical_type), axis) in input_atom_ids.iter().zip(logical_input_types.iter()).zip(input_axes.iter())
        {
            let parent_physical_type = match axis {
                Some(k) => logical_type.with_inserted_dimension(*k, Size::Static(self.axis_size))?,
                None => logical_type.clone(),
            };
            let parent_tracer = self.parent_context.tracer(*atom, Some(parent_physical_type.clone()));
            parent_input_batches.push(ArrayBatch::new(parent_physical_type, parent_tracer, *axis)?);
        }
        let output_batches = operation.batch(self, parent_input_batches.as_slice())?;

        let mut output_tracers = Vec::with_capacity(output_batches.len());
        for output_batch in output_batches {
            let axis = output_batch.batch_axis();
            let parent_tracer = output_batch.into_value();
            let parent_physical_type = parent_tracer.r#type().into_owned();
            let atom = parent_tracer.atom_id()?;
            self.register_axis(atom, axis);
            let logical_type = match axis {
                Some(k) => parent_physical_type.without_dimension(k)?.0,
                None => parent_physical_type,
            };
            output_tracers.push(self.tracer(atom, Some(logical_type)));
        }
        Ok(output_tracers)
    }
}

impl<C> DifferentiationContext for BatchingContext<C>
where
    C: StagingContext<Type = ArrayType> + DifferentiationContext + Domain<Type = ArrayType, Value = Tracer<C>>,
    C: DifferentiationContext<Tangent = Tracer<C>>,
    BatchingContext<C>: StagingContext<Type = ArrayType, Constant = <C as Domain>::Constant>,
{
    type Tangent = Tracer<BatchingContext<C>>;
    type LinearOperation<V: Value<ArrayType>, F: Value<ArrayType>> = C::LinearOperation<V, F>;

    #[inline]
    fn zero_tangent(&self, type_: &ArrayType) -> Result<Tracer<BatchingContext<C>>, ProgramError> {
        let value = self.parent_context.zero_tangent(type_)?;
        let atom = value.atom_id()?;
        Ok(self.tracer(atom, Some(type_.clone())))
    }

    #[inline]
    fn validate_primal(&self, primal: &Self::Value) -> Result<(), ProgramError> {
        if std::rc::Rc::ptr_eq(self.builder(), primal.context().builder()) {
            Ok(())
        } else {
            Err(self.error(ProgramError::MismatchedProgramBuilders))
        }
    }
}

/// Batching tracer selected by an ordinary backend [`Domain`].
pub type BatchingTracer<'domain, D> = Tracer<BatchingContext<TracingContext<'domain, D>>>;

/// Specification of the mapped axis introduced by one [`Batch::batch`] / [`BatchContext::batch`] call: an optional
/// explicit lane size and an optional axis name.
///
/// The lane size is normally inferred from the mapped inputs; provide an explicit size to either pin it or to drive
/// a fully-broadcast `batch` whose lane count would otherwise be unobservable. The axis name makes the mapped axis
/// addressable by name from collectives (`psum`, `pmean`, `pmax`) inside the batched function body, mirroring JAX's
/// `vmap(..., axis_name=...)`.
///
/// [`BatchAxis`] converts from the plain size forms, so call sites that do not need a name can pass `None`,
/// `Some(size)`, or `size` directly:
///
/// ```ignore
/// domain.batch(f, input, in_axes, out_axes, None)?;                        // Inferred size, anonymous.
/// domain.batch(f, input, in_axes, out_axes, 8)?;                           // Explicit size, anonymous.
/// domain.batch(f, input, in_axes, out_axes, BatchAxis::named("devices"))?; // Inferred size, named.
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BatchAxis {
    /// Explicit lane size, or `None` to infer it from the mapped inputs.
    size: Option<usize>,

    /// Name that collectives can use to address this axis, or `None` for an anonymous axis.
    name: Option<String>,
}

impl BatchAxis {
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

impl From<Option<usize>> for BatchAxis {
    fn from(size: Option<usize>) -> Self {
        Self { size, name: None }
    }
}

impl From<usize> for BatchAxis {
    fn from(size: usize) -> Self {
        Self::sized(size)
    }
}

/// Per-leaf axis specification for the `in_axes` / `out_axes` arguments of [`Batch::batch`] and
/// [`BatchContext::batch`].
///
/// [`BatchAxes::PerLeaf`] carries one `Option<usize>` per leaf of the matching parameter structure (the
/// fully-explicit form), while [`BatchAxes::Uniform`] applies one axis specification to every leaf — the typed
/// counterpart of JAX's `in_axes=0` shorthand. Both entry points accept anything convertible into [`BatchAxes`],
/// and plain per-leaf values convert automatically, so call sites can pass `Some(0)` for a single leaf,
/// `(Some(0), None)` for a pair, or `BatchAxes::Uniform(Some(0))` to map axis 0 of every leaf without spelling
/// out the structure.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BatchAxes<A> {
    /// One axis specification applied to every leaf of the matching parameter structure.
    Uniform(Option<usize>),

    /// Explicit per-leaf axis specifications matching the parameter structure exactly.
    PerLeaf(A),
}

impl<A> From<A> for BatchAxes<A> {
    fn from(axes: A) -> Self {
        Self::PerLeaf(axes)
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
    fn batch<'domain, F, I, O>(
        &'domain self,
        function: F,
        input: I,
        in_axes: impl Into<BatchAxes<I::To<Option<usize>>>>,
        out_axes: impl Into<BatchAxes<O::To<Option<usize>>>>,
        axis: impl Into<BatchAxis>,
    ) -> Result<O::To<Self::Value>, BatchingError>
    where
        Self: Context,
        Self::Value: 'domain,
        Self::Constant: 'domain,
        I: Parameterized<
                Self::Value,
                ParameterStructure: Debug + PartialEq,
                Family: ParameterizedFamily<ArrayType>
                            + ParameterizedFamily<Self::Constant>
                            + ParameterizedFamily<Option<usize>>
                            + ParameterizedFamily<DomainTracer<'domain, Self>>
                            + ParameterizedFamily<BatchingTracer<'domain, Self>>,
            >,
        O: Parameterized<
                BatchingTracer<'domain, Self>,
                ParameterStructure: Debug + PartialEq,
                Family: ParameterizedFamily<ArrayType>
                            + ParameterizedFamily<Self::Value>
                            + ParameterizedFamily<Self::Constant>
                            + ParameterizedFamily<Option<usize>>
                            + ParameterizedFamily<DomainTracer<'domain, Self>>
                            + ParameterizedFamily<BatchingTracer<'domain, Self>>,
            >,
        I::To<Option<usize>>: Parameterized<Option<usize>, ParameterStructure = I::ParameterStructure>,
        O::To<Option<usize>>: Parameterized<Option<usize>, ParameterStructure = O::ParameterStructure>,
        I::To<DomainTracer<'domain, Self>>: Parameterized<
                DomainTracer<'domain, Self>,
                ParameterStructure = I::ParameterStructure,
                To<ArrayType> = I::To<ArrayType>,
                To<Self::Value> = I,
                To<Self::Constant> = I::To<Self::Constant>,
                To<Option<usize>> = I::To<Option<usize>>,
                To<BatchingTracer<'domain, Self>> = I::To<BatchingTracer<'domain, Self>>,
            >,
        O::To<DomainTracer<'domain, Self>>: Parameterized<
                DomainTracer<'domain, Self>,
                ParameterStructure = O::ParameterStructure,
                To<ArrayType> = O::To<ArrayType>,
                To<Self::Value> = O::To<Self::Value>,
                To<Self::Constant> = O::To<Self::Constant>,
                To<Option<usize>> = O::To<Option<usize>>,
                To<BatchingTracer<'domain, Self>> = O,
            >,
        I::To<ArrayType>: Parameterized<
                ArrayType,
                To<Self::Value> = I,
                To<Self::Constant> = I::To<Self::Constant>,
                To<DomainTracer<'domain, Self>> = I::To<DomainTracer<'domain, Self>>,
                To<BatchingTracer<'domain, Self>> = I::To<BatchingTracer<'domain, Self>>,
            >,
        O::To<ArrayType>: Parameterized<
                ArrayType,
                To<Self::Value> = O::To<Self::Value>,
                To<Self::Constant> = O::To<Self::Constant>,
                To<DomainTracer<'domain, Self>> = O::To<DomainTracer<'domain, Self>>,
                To<BatchingTracer<'domain, Self>> = O,
            >,
        Self::Operation: Clone
            + InterpretableOperation<ArrayType, Self::Value>
            + SupportsTranspose<ArrayType>
            + for<'context> BatchableOperation<
                DomainTracer<'context, Self>,
                BatchingContext<TracingContext<'context, Self>>,
            >,
        F: FnOnce(I::To<BatchingTracer<'domain, Self>>) -> Result<O, ProgramError>,
    {
        let structure = input.parameter_structure();
        let input_values = input.into_parameters().collect::<Vec<_>>();
        let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        let parent_context = TracingContext::new(self, builder.clone());
        let mut input_tracers = Vec::with_capacity(input_values.len());
        for value in input_values.iter() {
            let physical_type = value.r#type().into_owned();
            let atom = builder.borrow_mut().add_input(physical_type.clone());
            input_tracers.push(parent_context.tracer(atom, Some(physical_type)));
        }
        let traced_input = I::To::<DomainTracer<'domain, Self>>::from_parameters(structure.clone(), input_tracers)?;
        // Batching rules ride up the `ProgramError`-typed staging kernel as `ProgramError::Custom` payloads; the
        // `From<ProgramError>` conversions behind the `?` operators below re-type them so the public `batch` surfaces
        // a transform-owned `BatchingError`, mirroring how `value_and_grad` surfaces a `DifferentiationError`.
        let traced_output: O::To<DomainTracer<'domain, Self>> =
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
        let output_values = program.interpret_with(
            input_values,
            |_, constant| self.lift(constant.clone()),
            |instruction, inputs| instruction.operation().interpret(inputs),
        )?;
        Ok(O::To::<Self::Value>::from_parameters(output_structure, output_values)?)
    }
}

impl<D: Domain<Type = ArrayType>> Batch for D {}

/// Extension trait that exposes batching as a method on active array contexts.
///
/// This is the already-traced counterpart of [`Batch`]. It wraps the receiver in a [`BatchingContext`] and routes all
/// primitive binds through the current transform stack, so `batch` composes with tracing, JVP, VJP, and other context
/// wrappers through the same [`StagingContext::stage_operation`] path.
pub trait BatchContext: StagingContext<Type = ArrayType> {
    /// Maps a traced function over per-leaf array axes inside this active context. The `in_axes` and `out_axes`
    /// parameters accept anything convertible to [`BatchAxes`] (explicit per-leaf axes or one uniform leaf
    /// specification), and the `axis` parameter accepts anything convertible to a [`BatchAxis`] (an optional
    /// explicit lane size and an optional axis name).
    #[allow(private_bounds)]
    fn batch<F, I, O>(
        &self,
        function: F,
        input: I,
        in_axes: impl Into<BatchAxes<I::To<Option<usize>>>>,
        out_axes: impl Into<BatchAxes<O::To<Option<usize>>>>,
        axis: impl Into<BatchAxis>,
    ) -> Result<O::To<Tracer<Self>>, ProgramError>
    where
        Self::Operation: Clone + SupportsTranspose<ArrayType> + BatchableOperation<Tracer<Self>, BatchingContext<Self>>,
        I: Parameterized<
                Tracer<Self>,
                ParameterStructure: Debug + PartialEq,
                Family: ParameterizedFamily<ArrayType>
                            + ParameterizedFamily<Self::Constant>
                            + ParameterizedFamily<Option<usize>>
                            + ParameterizedFamily<Tracer<Self>>
                            + ParameterizedFamily<Tracer<BatchingContext<Self>>>,
            >,
        O: Parameterized<
                Tracer<BatchingContext<Self>>,
                ParameterStructure: Debug + PartialEq,
                Family: ParameterizedFamily<ArrayType>
                            + ParameterizedFamily<Self::Constant>
                            + ParameterizedFamily<Option<usize>>
                            + ParameterizedFamily<Tracer<Self>>
                            + ParameterizedFamily<Tracer<BatchingContext<Self>>>,
            >,
        I::To<Option<usize>>: Parameterized<Option<usize>, ParameterStructure = I::ParameterStructure>,
        O::To<Option<usize>>: Parameterized<Option<usize>, ParameterStructure = O::ParameterStructure>,
        I::To<ArrayType>: Parameterized<
                ArrayType,
                To<Self::Constant> = I::To<Self::Constant>,
                To<Tracer<BatchingContext<Self>>> = I::To<Tracer<BatchingContext<Self>>>,
            >,
        O::To<ArrayType>:
            Parameterized<ArrayType, To<Tracer<Self>> = O::To<Tracer<Self>>, To<Tracer<BatchingContext<Self>>> = O>,
        F: FnOnce(I::To<Tracer<BatchingContext<Self>>>) -> Result<O, ProgramError>,
    {
        let axis = axis.into();
        let parent_context = self.clone();
        let input_structure = input.parameter_structure();
        let input_tracers = input.into_parameters().collect::<Vec<_>>();
        let in_axes_values = match in_axes.into() {
            BatchAxes::Uniform(leaf_axis) => vec![leaf_axis; input_tracers.len()],
            BatchAxes::PerLeaf(axes) => {
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
            match axis {
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
                    inputs_with_axes.push((tracer, Some(batch_axis), per_lane_type));
                }
                None => {
                    inputs_with_axes.push((tracer, None, parent_physical_type));
                }
            }
        }
        let resolved_axis_size = resolved_axis_size.ok_or(BatchingError::EmptyBatch)?;

        let batching_context = BatchingContext::with_axis_name(parent_context.clone(), resolved_axis_size, axis.name);
        let parent_builder = parent_context.builder().clone();

        let mut batched_input_tracers = Vec::with_capacity(inputs_with_axes.len());
        for (parent_tracer, axis, logical_type) in inputs_with_axes.iter() {
            let atom = parent_tracer.atom_id()?;
            batching_context.register_axis(atom, *axis);
            batched_input_tracers.push(batching_context.tracer(atom, Some(logical_type.clone())));
        }
        let batched_input =
            I::To::<Tracer<BatchingContext<Self>>>::from_parameters(input_structure, batched_input_tracers)?;
        let batched_output =
            function(batched_input).map_err(|error| parent_builder.borrow_mut().error.take().unwrap_or(error))?;
        parent_builder.borrow_mut().error.take().map_or(Ok(()), Err)?;

        let output_structure = batched_output.parameter_structure();
        let output_atom_ids =
            batched_output.parameters().map(|tracer| tracer.atom_id()).collect::<Result<Vec<_>, _>>()?;
        let out_axes_values = match out_axes.into() {
            BatchAxes::Uniform(leaf_axis) => vec![leaf_axis; output_atom_ids.len()],
            BatchAxes::PerLeaf(axes) => {
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
        let output_axes = output_atom_ids.iter().map(|atom| batching_context.axis_for(*atom)).collect::<Vec<_>>();
        drop(batched_output);
        drop(batching_context);

        let parent_outputs = output_atom_ids
            .into_iter()
            .zip(output_axes)
            .zip(out_axes_values.iter().copied())
            .map(|((atom, current_axis), expected_axis)| -> Result<Tracer<Self>, ProgramError> {
                let parent_tracer = parent_context.tracer(atom, None);
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
                        Ok(parent_tracer.transpose(permutation))
                    }
                }
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;

        Ok(O::To::<Tracer<Self>>::from_parameters(output_structure, parent_outputs)?)
    }
}

impl<C: StagingContext<Type = ArrayType>> BatchContext for C {}

/// Returns the axis permutation that moves dimension `from` to position `to`, shifting the other
/// dimensions to preserve their relative order. Returns the identity permutation when
/// `from == to`.
fn move_axis_permutation(rank: usize, from: usize, to: usize) -> Vec<usize> {
    let mut permutation: Vec<usize> = (0..rank).collect();
    let axis = permutation.remove(from);
    permutation.insert(to, axis);
    permutation
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::operations::constants::OneLike;
    use crate::operations::manipulation::Transpose;
    use crate::operations::trigonometric::Sin;
    use crate::parameters::Placeholder;
    use crate::tests::{TestArray, TestArrayDomain};
    use crate::tracing_v2::LinearizationTracer;
    use crate::tracing_v2::operations::control_flow::ConditionOperation;
    use crate::tracing_v2::operations::primitive::ArrayOperation;
    use crate::tracing_v2::operations::{Collective, CollectiveKind};
    use crate::tracing_v2::test_util::{assert_close, scalar_scale_branch};
    use crate::types::{DataType, Shape};

    use super::*;

    #[test]
    fn test_batching_error_conversions_normalize_round_trips() {
        // A batching error that crossed into the kernel as a custom payload converts back to itself, and a
        // `BatchingError::Program` converts back to the program error it carries, so round trips never nest.
        let batching = BatchingError::MismatchedBatchSizes { expected: 4, actual: 5 };
        let program = ProgramError::from(batching.clone());
        assert!(matches!(
            program.downcast_custom::<BatchingError>(),
            Some(BatchingError::MismatchedBatchSizes { expected: 4, actual: 5 }),
        ));
        assert_eq!(BatchingError::from(program), batching);

        let program = ProgramError::EscapedProgramBuilder;
        let batching = BatchingError::from(program.clone());
        assert_eq!(batching, BatchingError::Program(ProgramError::EscapedProgramBuilder));
        assert_eq!(ProgramError::from(batching), program);
    }

    #[test]
    fn test_array_batch_derives_logical_type_from_batch_axis() {
        let batch = ArrayBatch::mapped(TestArray::vector(vec![1.0, 2.0, 3.0]), 0).unwrap();

        assert_eq!(batch.axis_size(), Ok(Some(3)));
        assert_eq!(batch.logical_type(), Ok(ArrayType::scalar(DataType::F64)));
    }

    #[test]
    fn test_batch_uses_one_packed_array_value() {
        let output: TestArray = TestArrayDomain
            .batch(
                |x| Ok(x.clone() * x.clone() + x.sin()),
                TestArray::vector(vec![0.0, 1.0, 2.0]),
                Some(0),
                Some(0),
                None,
            )
            .unwrap();

        assert_eq!(output.r#type, ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)])),);
        for (actual, expected) in output.values.iter().zip([0.0, 1.0 + 1.0f64.sin(), 4.0 + 2.0f64.sin()]) {
            assert_close(*actual, expected);
        }
    }

    #[test]
    fn test_batch_broadcasts_scalar_constants_inside_packed_operations() {
        let output: TestArray = TestArrayDomain
            .batch(|x| Ok(x.clone() + x.one_like()), TestArray::vector(vec![2.0, 4.0, 6.0]), Some(0), Some(0), None)
            .unwrap();

        assert_eq!(output.values, vec![3.0, 5.0, 7.0]);
    }

    #[test]
    fn test_batch_maps_structured_packed_inputs_and_outputs() {
        let output: (TestArray, TestArray) = TestArrayDomain
            .batch(
                |(left, right)| Ok((left.clone() + right.clone(), left * right)),
                (TestArray::vector(vec![1.0, 3.0]), TestArray::vector(vec![2.0, 4.0])),
                (Some(0), Some(0)),
                (Some(0), Some(0)),
                None,
            )
            .unwrap();

        assert_eq!(output.0.values, vec![3.0, 7.0]);
        assert_eq!(output.1.values, vec![2.0, 12.0]);
    }

    #[test]
    fn test_batch_named_axis_psum_reduces_over_lanes() {
        let output: TestArray = TestArrayDomain
            .batch(
                |x| Ok(x.collective("i", CollectiveKind::PSum)),
                TestArray::vector(vec![1.0, 2.0, 3.0]),
                Some(0),
                None,
                BatchAxis::named("i"),
            )
            .unwrap();

        assert_eq!(output.r#type, ArrayType::scalar(DataType::F64));
        assert_eq!(output.values, vec![6.0]);
    }

    #[test]
    fn test_nested_batch_named_axes_route_collectives_to_matching_level() {
        // The inner `psum` targets the *outer* named axis, so each inner lane must reduce over the
        // outer lanes: column sums of [[1, 2], [3, 4]].
        let x = TestArray::new(
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2)])),
            vec![1.0, 2.0, 3.0, 4.0],
        );
        let output: TestArray = TestArrayDomain
            .batch(
                |row| {
                    let context = row.context().clone();
                    BatchContext::batch(
                        &context,
                        |scalar| Ok(scalar.collective("outer", CollectiveKind::PSum)),
                        row,
                        Some(0),
                        Some(0),
                        BatchAxis::named("inner"),
                    )
                },
                x,
                Some(0),
                None,
                BatchAxis::named("outer"),
            )
            .unwrap();

        assert_eq!(output.r#type, ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)])),);
        assert_eq!(output.values, vec![4.0, 6.0]);
    }

    #[test]
    fn test_value_and_grad_flows_through_batch_staged_broadcast() {
        use crate::tracing_v2::operations::reduce::{Reduce, ReductionKind};

        // The scalar input is lane-uniform inside the batch, so the elementwise batching rule
        // stages a `Broadcast` on the differentiated value; the gradient must flow back
        // through the broadcast's transpose rule (a sum-reduction over the lane axis).
        let (value, gradient) = crate::tracing_v2::value_and_grad(
            &TestArrayDomain,
            |x| {
                let context = x.context().clone();
                let y = context.constant(TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]));
                let mapped: LinearizationTracer<'_, TestArrayDomain> = BatchContext::batch(
                    &context,
                    |(lane, shift)| Ok(lane * shift),
                    (y, x),
                    (Some(0), None),
                    Some(0),
                    None,
                )
                .unwrap();
                mapped.reduce(&[0], ReductionKind::Sum)
            },
            TestArray::scalar(2.0),
        )
        .unwrap();
        assert_close(value.values[0], 20.0);
        assert_eq!(gradient.values, vec![10.0]);
    }

    #[test]
    fn test_batch_composes_with_context_jvp() {
        let output: (TestArray, TestArray) = TestArrayDomain
            .batch(
                |x| {
                    let context = x.context().clone();
                    DifferentiationContext::jvp(&context, |y| y.clone() * y, x.clone(), x.one_like())
                },
                TestArray::vector(vec![2.0, 3.0]),
                Some(0),
                (Some(0), Some(0)),
                None,
            )
            .unwrap();

        assert_eq!(output.0.values, vec![4.0, 9.0]);
        assert_eq!(output.1.values, vec![4.0, 6.0]);
    }

    #[test]
    fn test_batch_composes_with_context_value_and_grad() {
        let output: (TestArray, TestArray) = TestArrayDomain
            .batch(
                |x| {
                    let context = x.context().clone();
                    Ok(context.value_and_grad(|y| y.clone() * y, x).expect("scalar value_and_grad should succeed"))
                },
                TestArray::vector(vec![2.0, 3.0]),
                Some(0),
                (Some(0), Some(0)),
                None,
            )
            .unwrap();

        assert_eq!(output.0.values, vec![4.0, 9.0]);
        assert_eq!(output.1.values, vec![4.0, 6.0]);
    }

    #[test]
    fn test_context_batch_composes_inside_jvp() {
        let (primal, tangent): (TestArray, TestArray) = TestArrayDomain
            .jvp(
                |x| {
                    let context = x.context().clone();
                    let output: LinearizationTracer<'_, TestArrayDomain> =
                        BatchContext::batch(&context, |lane| Ok(lane.clone() * lane), x, Some(0), Some(0), None)
                            .unwrap();
                    output
                },
                TestArray::vector(vec![2.0, 3.0]),
                TestArray::vector(vec![1.0, 1.0]),
            )
            .unwrap();

        assert_eq!(primal.values, vec![4.0, 9.0]);
        assert_eq!(tangent.values, vec![4.0, 6.0]);
    }

    #[test]
    fn test_context_batch_composes_inside_value_and_grad() {
        use crate::tracing_v2::operations::reduce::{Reduce, ReductionKind};

        let (value, gradient): (TestArray, TestArray) = crate::tracing_v2::value_and_grad(
            &TestArrayDomain,
            |x| {
                let context = x.context().clone();
                let mapped: crate::tracing_v2::LinearizationTracer<'_, TestArrayDomain> =
                    BatchContext::batch(&context, |lane| Ok(lane.clone() * lane), x, Some(0), Some(0), None).unwrap();
                mapped.reduce(&[0], ReductionKind::Sum)
            },
            TestArray::vector(vec![2.0, 3.0]),
        )
        .unwrap();

        assert_eq!(value.values, vec![13.0]);
        assert_eq!(gradient.values, vec![4.0, 6.0]);
    }

    #[test]
    fn test_batching_rule_auto_aligns_unaligned_batch_axes() {
        // Both square so the lane sizes agree (4), but they sit on different batch axes.
        // `apply_elementwise_batch` realigns the second operand to match the first batched
        // input's canonical axis (JAX's matchaxis policy), then computes elementwise add.
        //
        // Left is identity-like along axis 0; right is transposed (axis 1). Using row 0 of each
        // lane: left[lane=k, j] == 1.0; right[lane=k, j] == 1.0 (since right is symmetric here),
        // so the sum is `2.0` for every element after realignment.
        let left = ArrayBatch::mapped(TestArray::matrix(4, 4, vec![1.0; 16]), 0).unwrap();
        let right = ArrayBatch::mapped(TestArray::matrix(4, 4, vec![1.0; 16]), 1).unwrap();

        let outputs = ArrayOperation::<TestArray, ArrayType>::Add.batch(&(), &[left, right]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert!(outputs[0].value().values().iter().all(|value| (value - 2.0).abs() < 1e-12));
    }

    #[test]
    fn test_lift_elementwise_binary_op() {
        let scalar = ArrayType::scalar(DataType::F64);
        let op = ArrayOperation::<TestArray, ArrayType>::Add;
        let (lifted_op, output_axes) =
            lift_elementwise(&op, &[scalar.clone(), scalar], &[Some(0), Some(0)], 5).unwrap();

        assert!(matches!(lifted_op, ArrayOperation::Add));
        assert_eq!(output_axes, vec![Some(0)]);
    }

    #[test]
    fn test_lift_elementwise_unary_op() {
        let scalar = ArrayType::scalar(DataType::F64);
        let op = ArrayOperation::<TestArray, ArrayType>::Sin;
        let (lifted_op, output_axes) = lift_elementwise(&op, &[scalar], &[Some(0)], 7).unwrap();

        assert!(matches!(lifted_op, ArrayOperation::Sin));
        assert_eq!(output_axes, vec![Some(0)]);
    }

    #[test]
    fn test_lift_elementwise_rejects_misaligned_input_axes() {
        let scalar = ArrayType::scalar(DataType::F64);
        let op = ArrayOperation::<TestArray, ArrayType>::Add;
        let err = lift_elementwise(&op, &[scalar.clone(), scalar], &[Some(0), Some(1)], 5).unwrap_err();
        // `lift_elementwise` is an operation-level batching helper, so its `BatchingError` rides up as a
        // `ProgramError::Custom` payload; recover the concrete error with `downcast_custom`.
        assert!(matches!(err.downcast_custom::<BatchingError>(), Some(BatchingError::MisalignedBatchAxes { .. }),));
    }

    #[test]
    fn test_lift_elementwise_passes_through_lane_uniform_inputs() {
        let scalar = ArrayType::scalar(DataType::F64);
        let op = ArrayOperation::<TestArray, ArrayType>::Add;
        let (lifted_op, output_axes) = lift_elementwise(&op, &[scalar.clone(), scalar], &[Some(0), None], 5).unwrap();

        assert!(matches!(lifted_op, ArrayOperation::Add));
        assert_eq!(output_axes, vec![Some(0)]);
    }

    #[test]
    fn test_nested_batch_squares_every_element() {
        // x has shape [3, 4]; outer batch maps axis 0 (size 3), inner batch maps axis 0 of the
        // per-outer-lane shape [4]. Each element should be squared.
        let x_data: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let x = TestArray::matrix(3, 4, x_data.clone());

        let output: TestArray = TestArrayDomain
            .batch(
                |row| {
                    let context = row.context().clone();
                    BatchContext::batch(&context, |scalar| Ok(scalar.clone() * scalar), row, Some(0), Some(0), None)
                },
                x,
                Some(0),
                Some(0),
                None,
            )
            .unwrap();

        assert_eq!(output.r#type, ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3), Size::Static(4)])),);
        let expected: Vec<f64> = x_data.iter().map(|value| value * value).collect();
        for (actual, expected) in output.values.iter().zip(expected.iter()) {
            assert_close(*actual, *expected);
        }
    }

    #[test]
    fn test_nested_batch_over_dot_lifts_dimension_numbers() {
        use crate::tracing_v2::operations::dot::{Dot, DotDimensionNumbers};

        // x has shape [3, 4]; outer batch over axis 0 produces per-lane rank-1 vectors. Inside,
        // we want every per-lane vector dotted with itself, giving a per-lane scalar; batch
        // over the leading axis then yields a length-3 vector of dot products.
        let x_data: Vec<f64> = (1..=12).map(|value| value as f64).collect();
        let x = TestArray::matrix(3, 4, x_data);

        let output: TestArray = TestArrayDomain
            .batch(|row| Ok(row.clone().dot(row, &DotDimensionNumbers::inner_product())), x, Some(0), Some(0), None)
            .unwrap();

        assert_eq!(output.r#type, ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)])),);
        // Lane 0: [1,2,3,4]·[1,2,3,4] = 30. Lane 1: [5,6,7,8]·[5,6,7,8] = 174. Lane 2: 446.
        for (actual, expected) in output.values.iter().zip([30.0_f64, 174.0, 446.0].iter()) {
            assert_close(*actual, *expected);
        }
    }

    #[test]
    fn test_nested_batch_over_transpose_lifts_permutation() {
        // x has shape [2, 3, 4]; outer batch over axis 0 yields per-lane rank-2 matrices,
        // which we transpose. The combined effect is to permute axes 1 and 2 of the original
        // tensor, leaving the batch axis (originally axis 0) in place.
        let x_data: Vec<f64> = (0..24).map(|value| value as f64).collect();
        let x = TestArray {
            r#type: ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3), Size::Static(4)])),
            values: x_data,
        };

        let output: TestArray =
            TestArrayDomain.batch(|row| Ok(row.transpose(vec![1, 0])), x, Some(0), Some(0), None).unwrap();

        assert_eq!(
            output.r#type,
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(4), Size::Static(3)])),
        );
        // Spot-check: original [0, 0, 0] = 0 → output[0, 0, 0] = 0. Original [0, 0, 1] = 1 → output[0, 1, 0] = 1.
        assert_eq!(output.values[0], 0.0);
        assert_eq!(output.values[1 * 3], 1.0);
    }

    #[test]
    fn test_batch_broadcasts_lane_uniform_input_with_in_axes_none() {
        // x is a [4]-vector mapped on axis 0 (lanes), y is a lane-uniform scalar that should be
        // added to every lane. The output should be element-wise `x + y` over the 4 lanes.
        let x = TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]);
        let y = TestArray::scalar(10.0);
        let output: TestArray = TestArrayDomain
            .batch(|(left, right)| Ok(left + right), (x, y), (Some(0), None), Some(0), None)
            .unwrap();
        assert_eq!(output.values, vec![11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_batch_with_axis_size_validates_mapped_lane_count() {
        // With explicit axis_size = Some(4), the lane count is pinned. A mapped input of size 4
        // must agree, and the lane count flows through to subsequent operations.
        let x = TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]);
        let output: TestArray = TestArrayDomain.batch(|x| Ok(x.clone() + x), x, Some(0), Some(0), Some(4)).unwrap();
        assert_eq!(output.values, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_batch_with_out_axes_none_rejects_mapped_output() {
        // Function produces a per-lane output (mapped on axis 0), but `out_axes = None` declares
        // the output as lane-uniform — matching JAX's semantics. The batch rejects because the
        // computed output is genuinely per-lane; users wanting to collapse the lane axis must
        // apply an explicit reduction inside the function.
        let x = TestArray::vector(vec![1.0, 2.0, 3.0]);
        let result: Result<TestArray, BatchingError> =
            TestArrayDomain.batch(|x| Ok(x.clone() + x), x, Some(0), None, None);
        assert!(matches!(result, Err(BatchingError::MismatchedOutputAxes { expected: None, actual: Some(0) })));
    }

    #[test]
    fn test_batch_with_out_axes_position_rejects_lane_uniform_output() {
        // No input is mapped, so the output never picks up the lane axis, but `out_axes = Some(0)`
        // requests a mapped output; `batch` refuses to materialize the axis with an implicit broadcast.
        let x = TestArray::vector(vec![1.0, 2.0, 3.0]);
        let result: Result<TestArray, BatchingError> =
            TestArrayDomain.batch(|x| Ok(x.clone() + x), x, None, Some(0), Some(3));
        assert!(matches!(result, Err(BatchingError::MismatchedOutputAxes { expected: Some(0), actual: None })));
    }

    #[test]
    fn test_batch_rejects_dynamic_batch_axis() {
        // A mapped input whose batch dimension is `Size::Dynamic` cannot be batched: batch has no
        // way to determine the lane count.
        let dynamic_input = TestArray {
            r#type: ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None)])),
            values: vec![1.0, 2.0, 3.0],
        };
        let result: Result<TestArray, BatchingError> =
            TestArrayDomain.batch(|x| Ok(x.clone() + x), dynamic_input, Some(0), Some(0), None);
        assert!(matches!(result, Err(BatchingError::DynamicBatchAxis { axis: 0, .. })));
    }

    #[test]
    fn test_batch_with_mismatched_axis_size_rejects_mapped_input() {
        // axis_size=Some(5) conflicts with the mapped input of length 4; this should be detected.
        let x = TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]);
        let result: Result<TestArray, BatchingError> =
            TestArrayDomain.batch(|x| Ok(x.clone() + x), x, Some(0), Some(0), Some(5));
        assert!(matches!(result, Err(BatchingError::MismatchedBatchSizes { expected: 5, actual: 4 })));
    }

    #[test]
    fn test_batch_repositions_output_with_out_axes() {
        // Outer batch over axis 0 of a [3, 4] matrix: each lane returns its row unchanged.
        // out_axes=Some(1) requests that the batch axis end up at position 1 of the rank-2
        // output, which forces a transpose to swap the axes.
        let x_data: Vec<f64> = (0..12).map(|value| value as f64).collect();
        let x = TestArray::matrix(3, 4, x_data.clone());
        let output: TestArray = TestArrayDomain.batch(|row| Ok(row), x, Some(0), Some(1), None).unwrap();
        assert_eq!(output.r#type, ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(4), Size::Static(3)])),);
        // Transpose of [3, 4]: output[i, j] = x[j, i]. Row-major flat indexing:
        // x[j, i] = x_data[j*4 + i]; output[i, j] = output_values[i*3 + j].
        for j in 0..3 {
            for i in 0..4 {
                assert_eq!(output.values[i * 3 + j], x_data[j * 4 + i]);
            }
        }
    }

    #[test]
    fn test_nested_batch_with_mixed_in_axes_propagates_broadcast() {
        // Outer batch over axis 0 of `x: [3, 4]` exposes a rank-1 row to the closure; inside, a
        // second inner batch maps that row's lane axis 0 while broadcasting a captured `bias`
        // scalar to every inner lane. The combined output is x + bias broadcasted.
        let x_data: Vec<f64> = (0..12).map(|value| value as f64).collect();
        let x = TestArray::matrix(3, 4, x_data.clone());
        let bias = TestArray::scalar(0.5);

        let output: TestArray = TestArrayDomain
            .batch(
                |(row, bias_inner)| {
                    let context = row.context().clone();
                    BatchContext::batch(
                        &context,
                        |(scalar, bias_inner)| Ok(scalar + bias_inner),
                        (row, bias_inner),
                        (Some(0), None),
                        Some(0),
                        None,
                    )
                },
                (x, bias),
                (Some(0), None),
                Some(0),
                None,
            )
            .unwrap();

        assert_eq!(output.r#type, ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3), Size::Static(4)])),);
        let expected: Vec<f64> = x_data.iter().map(|value| value + 0.5).collect();
        for (actual, expected) in output.values.iter().zip(expected.iter()) {
            assert_close(*actual, *expected);
        }
    }

    #[test]
    fn test_nested_batch_over_reshape_lifts_input_and_output_shapes() {
        use crate::tracing_v2::operations::reshape::Reshape;

        // x has shape [2, 6]; outer batch over axis 0 yields per-lane rank-1 vectors of size 6,
        // which we reshape to per-lane [2, 3]. The combined effect should be a [2, 2, 3] tensor
        // whose leading axis is the original batch dimension.
        let x_data: Vec<f64> = (0..12).map(|value| value as f64).collect();
        let x = TestArray::matrix(2, 6, x_data.clone());

        let output: TestArray = TestArrayDomain
            .batch(|row| row.reshape(Shape::new(vec![Size::Static(2), Size::Static(3)])), x, Some(0), Some(0), None)
            .unwrap();

        assert_eq!(
            output.r#type,
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2), Size::Static(3)])),
        );
        // Row-major reshape preserves payload ordering; the lifted op only repositions strides.
        assert_eq!(output.values, x_data);
    }

    #[test]
    fn test_batch_lifts_captured_condition_at_trace_time() {
        // A captured-true Condition inside batch: each lane scaled by 2.0. The
        // `ConditionOperation::lift` trace-time path re-traces the picked branch through a
        // fresh BatchingContext and stages the lifted ConditionOperation directly into the
        // outer trace.
        let output: TestArray = TestArrayDomain
            .batch(
                |x| {
                    let condition = ConditionOperation::with_captured_predicate(
                        true,
                        scalar_scale_branch(2.0),
                        scalar_scale_branch(3.0),
                    )
                    .unwrap();
                    let op = ArrayOperation::Condition(Box::new(condition));
                    let outputs = x.context().stage_operation(op, &[&x])?;
                    Ok(outputs.into_iter().next().unwrap())
                },
                TestArray::vector(vec![1.0, 4.0, 9.0]),
                Some(0),
                Some(0),
                None,
            )
            .unwrap();
        assert_eq!(output.values, vec![2.0, 8.0, 18.0]);
    }

    #[test]
    fn test_batch_lifts_lane_varying_condition_via_select() {
        // A runtime-predicate Condition inside batch with a lane-varying predicate: each lane
        // independently chooses between `on_true` (scale by 2.0) and `on_false` (scale by 3.0).
        // The trace-time `BatchingContext` dispatches the rule's `batch`, whose
        // lane-varying branch evaluates both branches over the operand axes and combines per lane
        // via `Select`. Multi-op staging emerges automatically through `Tracer`'s value-level traits.
        let predicate = TestArray::vector(vec![1.0, 0.0, 1.0, 0.0]);
        let operand = TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]);

        let output: TestArray = TestArrayDomain
            .batch(
                |(pred, operand)| {
                    let condition = ConditionOperation::new(
                        ArrayType::scalar(DataType::Boolean),
                        scalar_scale_branch(2.0),
                        scalar_scale_branch(3.0),
                    )
                    .unwrap();
                    let op = ArrayOperation::Condition(Box::new(condition));
                    let outputs = pred.context().stage_operation(op, &[&pred, &operand])?;
                    Ok(outputs.into_iter().next().unwrap())
                },
                (predicate, operand),
                (Some(0), Some(0)),
                Some(0),
                None,
            )
            .unwrap();
        // Expected per-lane: [1*2, 2*3, 3*2, 4*3] = [2, 6, 6, 12].
        assert_eq!(output.values, vec![2.0, 6.0, 6.0, 12.0]);
    }

    #[test]
    fn test_batch_over_zero_operation_yields_lane_uniform_output() {
        // End-to-end: a batched function that stages `ZeroOperation` produces a lane-uniform zero
        // value at the per-lane scalar type. Verifies that the trace-time stage hook accepts a
        // zero-input operation and that the post-trace replay materializes the same zero for
        // every lane through the lane-uniform broadcast path.
        let output: TestArray = TestArrayDomain
            .batch(
                |x| {
                    let zero_op = ArrayOperation::<TestArray, ArrayType>::Zero(
                        crate::operations::constants::ZeroOperation::new(ArrayType::scalar(DataType::F64)),
                    );
                    let no_inputs: &[&BatchingTracer<'_, _>] = &[];
                    let zero = x.context().stage_operation(zero_op, no_inputs)?.into_iter().next().unwrap();
                    Ok(x + zero)
                },
                TestArray::vector(vec![1.0, 2.0, 3.0]),
                Some(0),
                Some(0),
                None,
            )
            .unwrap();

        assert_eq!(output.values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_batch() {
        let output: TestArray = batch(
            &TestArrayDomain,
            |x| Ok(x.clone() * x),
            TestArray::vector(vec![1.0, 2.0, 3.0]),
            Some(0),
            Some(0),
            None,
        )
        .unwrap();
        assert_eq!(output.r#type, ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)])));
        assert_eq!(output.values, vec![1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_batch_axis_constructors_and_conversions() {
        assert_eq!(BatchAxis::from(None), BatchAxis::default());
        assert_eq!(BatchAxis::from(Some(4)), BatchAxis::sized(4));
        assert_eq!(BatchAxis::from(4), BatchAxis::sized(4));
        assert_eq!(BatchAxis::sized_and_named(4, "i"), BatchAxis::sized_and_named(4, "i").clone());
        assert_ne!(BatchAxis::sized(4), BatchAxis::sized(5));
        assert_ne!(BatchAxis::named("i"), BatchAxis::named("j"));
        assert_ne!(BatchAxis::named("i"), BatchAxis::default());
        assert_eq!(format!("{:?}", BatchAxis::named("i")), "BatchAxis { size: None, name: Some(\"i\") }");
    }

    #[test]
    fn test_batch_axes_uniform_applies_one_spec_to_every_leaf() {
        let x = TestArray::vector(vec![1.0, 3.0]);
        let y = TestArray::vector(vec![2.0, 4.0]);
        let output: (TestArray, TestArray) = TestArrayDomain
            .batch(
                |(left, right)| Ok((left.clone() + right.clone(), left * right)),
                (x, y),
                BatchAxes::Uniform(Some(0)),
                BatchAxes::Uniform(Some(0)),
                None,
            )
            .unwrap();
        assert_eq!(output.0.values, vec![3.0, 7.0]);
        assert_eq!(output.1.values, vec![2.0, 12.0]);

        // Plain per-leaf values convert to `BatchAxes::PerLeaf`.
        assert_eq!(BatchAxes::from((Some(0), None::<usize>)), BatchAxes::PerLeaf((Some(0), None::<usize>)));
        assert_eq!(format!("{:?}", BatchAxes::<Option<usize>>::Uniform(Some(1))), "Uniform(Some(1))");
    }

    #[test]
    fn test_batch_broadcasts_mapped_inputs_with_mixed_per_lane_ranks() {
        // x is mapped with per-lane shape [3]; y is mapped with a per-lane scalar shape. The
        // elementwise rule broadcasts y's per-lane scalar across the common per-lane shape, so
        // each lane computes `row + shift` with its own shift.
        let x = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = TestArray::vector(vec![10.0, 20.0]);
        let output: TestArray = TestArrayDomain
            .batch(|(row, shift)| Ok(row + shift), (x, y), BatchAxes::Uniform(Some(0)), Some(0), None)
            .unwrap();
        assert_eq!(output.r#type, ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)])),);
        assert_eq!(output.values, vec![11.0, 12.0, 13.0, 24.0, 25.0, 26.0]);
    }

    #[test]
    fn test_batch_broadcasts_scalar_lane_uniform_operands_to_full_shape() {
        // A lane-uniform scalar constant added to a mapped [3, 4] input: the elementwise rule
        // materializes a `BroadcastOperation` to the full common batched shape so the staged
        // add receives shape-congruent operands — required for backends such as XLA whose
        // elementwise lowerings (e.g., `stablehlo.add`) have no implicit broadcasting.
        let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        let parent_context = TracingContext::new(&TestArrayDomain, builder.clone());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3), Size::Static(4)]));
        let input_atom = builder.borrow_mut().add_input(input_type);
        let input_tracer = parent_context.tracer(input_atom, None);
        let output = BatchContext::batch(
            &parent_context,
            |x| {
                let bias = x.context().constant(TestArray::scalar(1.0));
                Ok(x + bias)
            },
            input_tracer,
            Some(0),
            Some(0),
            None,
        )
        .unwrap();
        let output_atom = output.atom_id().unwrap();
        let program = builder
            .borrow()
            .clone()
            .build::<TestArray, TestArray>(vec![output_atom], Placeholder, Placeholder)
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[3, 4] .
                let %1:f64[] = const
                    %2:f64[3, 4] = broadcast [target_type=f64[3, 4], output_axes=[]] %1
                    %3:f64[3, 4] = add %0 %2
                in (%3)
            "}
            .trim_end(),
        );
        let input = TestArray::matrix(3, 4, (0..12).map(|value| value as f64).collect());
        let output = program.interpret(input).unwrap();
        assert_eq!(output.values, (0..12).map(|value| value as f64 + 1.0).collect::<Vec<_>>());
    }
}
