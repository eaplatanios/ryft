use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::rc::Rc;

use crate::ElementwiseOperation;
use crate::batching::BatchingError;
use crate::broadcasting::Broadcastable;
use crate::contexts::{Context, ProvidesContext, StagingContext};
use crate::domains::Domain;
use crate::macros::{check_builders, check_count};
use crate::operations::manipulation::{Broadcast, Transpose, TransposeOperation};
use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily};
use crate::programs::{AtomId, Program, ProgramBuilder, ProgramError, Value};
use crate::sharding::ShardingDimension;
use crate::tracing::{DomainTracer, DomainTracingContext, Tracer, TracerState, TracingContext};
use crate::tracing_v2::differentiation::DifferentiationContext;
use crate::types::{ArrayType, Size, Typed};

/// Maps a traced `function` over array axes selected per leaf by `in_axes` and places each output's mapped axis at
/// the position requested by `out_axes`. This is the module-level equivalent of [`Batch::batch`]; refer to its
/// documentation for the full semantics.
///
/// # Parameters
///
///   - `context`: [`Context`] that provides the traced operation, type, and constant representations.
///   - `function`: Function/closure to map over the batched lanes.
///   - `input`: Input value whose mapped leaves drive the batched lanes.
///   - `in_axes`: Per-leaf mapped-axis selection for `input`, or one uniform leaf specification.
///   - `out_axes`: Per-leaf mapped-axis placement for the output, or one uniform leaf specification.
///   - `axis`: Mapped-axis specification carrying an optional explicit lane size and an optional axis name.
#[inline]
pub fn batch<C, F, I, O>(
    context: &C,
    function: F,
    input: I,
    in_axes: impl Into<BatchAxes<I::To<Option<usize>>>>,
    out_axes: impl Into<BatchAxes<O::To<Option<usize>>>>,
    axis: impl Into<BatchAxis>,
) -> Result<O::To<C::Value>, BatchingError>
where
    C: Context<Type = ArrayType>,
    I: Parameterized<
            C::Value,
            ParameterStructure: Debug + PartialEq,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<C::Constant>
                        + ParameterizedFamily<Option<usize>>
                        + ParameterizedFamily<DomainTracer<C>>
                        + ParameterizedFamily<BatchingTracer<C>>,
        >,
    O: Parameterized<
            BatchingTracer<C>,
            ParameterStructure: Debug + PartialEq,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<C::Value>
                        + ParameterizedFamily<C::Constant>
                        + ParameterizedFamily<Option<usize>>
                        + ParameterizedFamily<DomainTracer<C>>
                        + ParameterizedFamily<BatchingTracer<C>>,
        >,
    I::To<Option<usize>>: Parameterized<Option<usize>, ParameterStructure = I::ParameterStructure>,
    O::To<Option<usize>>: Parameterized<Option<usize>, ParameterStructure = O::ParameterStructure>,
    I::To<DomainTracer<C>>: Parameterized<
            DomainTracer<C>,
            ParameterStructure = I::ParameterStructure,
            To<ArrayType> = I::To<ArrayType>,
            To<C::Value> = I,
            To<C::Constant> = I::To<C::Constant>,
            To<Option<usize>> = I::To<Option<usize>>,
            To<BatchingTracer<C>> = I::To<BatchingTracer<C>>,
        >,
    O::To<DomainTracer<C>>: Parameterized<
            DomainTracer<C>,
            ParameterStructure = O::ParameterStructure,
            To<ArrayType> = O::To<ArrayType>,
            To<C::Value> = O::To<C::Value>,
            To<C::Constant> = O::To<C::Constant>,
            To<Option<usize>> = O::To<Option<usize>>,
            To<BatchingTracer<C>> = O,
        >,
    I::To<ArrayType>: Parameterized<
            ArrayType,
            To<C::Value> = I,
            To<C::Constant> = I::To<C::Constant>,
            To<DomainTracer<C>> = I::To<DomainTracer<C>>,
            To<BatchingTracer<C>> = I::To<BatchingTracer<C>>,
        >,
    O::To<ArrayType>: Parameterized<
            ArrayType,
            To<C::Value> = O::To<C::Value>,
            To<C::Constant> = O::To<C::Constant>,
            To<DomainTracer<C>> = O::To<DomainTracer<C>>,
            To<BatchingTracer<C>> = O,
        >,
    C::Operation: Clone
        + InterpretableOperation<ArrayType, C::Value>
        + From<TransposeOperation>
        + BatchableOperation<DomainTracer<C>, BatchingContext<DomainTracingContext<C>>>,
    F: FnOnce(I::To<BatchingTracer<C>>) -> Result<O, ProgramError>,
{
    context.batch(function, input, in_axes, out_axes, axis)
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

impl<V: Value<ArrayType>> Value<ArrayType> for ArrayBatch<V> {
    type InterpretationContext = V::InterpretationContext;

    #[inline]
    fn interpretation_context(&self) -> Option<V::InterpretationContext> {
        self.value().interpretation_context()
    }
}

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
/// `Neg`, `Sin`, `Cos`, `Scale`, …). Ops with non-trivial axis arithmetic (`Dot`, `Transpose`, `Reshape`, …) and the
/// [`ArrayOperation`](crate::tracing_v2::ArrayOperation) operation enum (whose impls live with the enum in
/// [`operations::primitive`](crate::tracing_v2::operations::primitive)) keep their explicit impls; coherence is
/// preserved because none of those types implement [`ElementwiseOperation`].
impl<
    O: Clone + InterpretableOperation<ArrayType, V> + ElementwiseOperation,
    V: Value<ArrayType> + Broadcast + Transpose,
> BatchableOperation<V, V::InterpretationContext> for O
{
    #[inline]
    fn batch(
        &self,
        context: &V::InterpretationContext,
        inputs: &[ArrayBatch<V>],
    ) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
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

// TODO(eaplatanios): Review this function carefully.
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
pub fn apply_with_axes<V: Value<ArrayType>, O: InterpretableOperation<ArrayType, V>>(
    context: &V::InterpretationContext,
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
    O: Clone + InterpretableOperation<ArrayType, V>,
>(
    context: &V::InterpretationContext,
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
            // the original shapes.
            let per_lane_types =
                aligned_inputs.iter().map(|input| input.logical_type()).collect::<Result<Vec<_>, _>>()?;
            let common_per_lane = Broadcastable::broadcasted(per_lane_types.as_slice()).ok();
            let broadcasted_physical_type = |per_lane_type: &ArrayType| -> Result<ArrayType, ProgramError> {
                // The common per-lane target only contributes its shape: each operand keeps its own data type (e.g., a
                // Boolean select condition broadcast against numeric branches stays Boolean).
                let mut target = common_per_lane.as_ref().unwrap_or(per_lane_type).clone();
                target.data_type = per_lane_type.data_type();
                Ok(target.with_inserted_dimension(batch_axis, Size::Static(axis_size))?)
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
        (Some(axis), Ok(common)) => {
            let common = common.with_inserted_dimension(axis, Size::Static(axis_size))?;
            input_types
                .iter()
                .map(|per_lane_type| {
                    // The common per-lane target only contributes its shape: each operand keeps its own data
                    // type (e.g., a Boolean select condition broadcast against numeric branches stays Boolean).
                    let mut target = common.clone();
                    target.data_type = per_lane_type.data_type();
                    target
                })
                .collect()
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
    use crate::parameters::Placeholder;

    let logical_input_types = program.input_types();
    let input_count = logical_input_types.len();
    check_count!("input", input_batch_axes, input_count, ProgramError);
    let parent_context: ProgramBatchingContext<V, O> = TracingContext::new();
    let builder = parent_context.builder().clone();
    // Keep every tracer and context that holds a clone of `builder` inside this scope so that recovering the
    // builder below is a real ownership check.
    let (output_atom_ids, output_axes) = {
        let batching_context = BatchingContext::new(parent_context, axis_size);
        let mut input_tracers = Vec::with_capacity(input_count);
        for (logical_type, axis) in logical_input_types.iter().zip(input_batch_axes.iter()) {
            let physical_type = match axis {
                Some(position) => logical_type.with_inserted_dimension(*position, Size::Static(axis_size))?,
                None => logical_type.clone(),
            };
            let atom = builder.borrow_mut().add_input(physical_type);
            batching_context.register_axis(atom, *axis);
            input_tracers.push(batching_context.tracer(atom, Some(logical_type.clone())));
        }
        let output_tracers = batching_context.stage_program(program, input_tracers)?;
        let mut output_atom_ids = Vec::with_capacity(output_tracers.len());
        let mut output_axes = Vec::with_capacity(output_tracers.len());
        for output_tracer in output_tracers {
            match output_batch_axes {
                ProgramBatchingOutputAxes::Natural => {
                    let atom = output_tracer.atom_id()?;
                    output_axes.push(batching_context.axis_for(atom));
                    output_atom_ids.push(atom);
                }
                ProgramBatchingOutputAxes::AlignAllTo(target_axis) => {
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

    /// Per-atom batch-axis annotations for atoms staged through this batching level. Missing keys
    /// are treated as lane-uniform (axis = `None`). Lives in interior mutability because the
    /// [`StagingContext::stage_operation`] hook takes `&self` but needs to record output axes as it
    /// stages instructions into the parent [`ProgramBuilder`].
    axis_table: Rc<RefCell<HashMap<AtomId, usize>>>,
}

impl<C: StagingContext<Type = ArrayType, Value = Tracer<C>>> BatchingContext<C> {
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
    C: StagingContext<Type = ArrayType, Value = Tracer<C>>,
    C::Operation: BatchableOperation<Tracer<C>, Self>,
{
    type Type = ArrayType;
    type Value = Tracer<Self>;
    type Constant = C::Constant;
    type Operation = C::Operation;
}

impl<C> Context for BatchingContext<C>
where
    C: StagingContext<Type = ArrayType, Value = Tracer<C>>,
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
    fn bind<P: Into<Self::Operation>>(
        &self,
        operation: P,
        inputs: &[Tracer<Self>],
    ) -> Result<Vec<Tracer<Self>>, ProgramError> {
        let operation = operation.into();
        self.stage_operation(operation, inputs)
    }
}

impl<C> StagingContext for BatchingContext<C>
where
    C: StagingContext<Type = ArrayType, Value = Tracer<C>>,
    C::Operation: BatchableOperation<Tracer<C>, Self>,
{
    #[inline]
    fn builder(&self) -> &Rc<RefCell<ProgramBuilder<Self::Type, Self::Constant, Self::Operation>>> {
        self.parent_context.builder()
    }

    fn stage_operation<P: Into<Self::Operation>, I: std::borrow::Borrow<Tracer<Self>>>(
        &self,
        operation: P,
        inputs: &[I],
    ) -> Result<Vec<Tracer<Self>>, ProgramError> {
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
        // input list and surface the resulting parent atoms as lane-uniform tracers (no entry in
        // `axis_table`).
        if inputs.is_empty() {
            let parent_outputs = self.parent_context.stage_nullary_operation(operation)?;
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
    C: StagingContext<Type = ArrayType, Value = Tracer<C>> + DifferentiationContext + Domain<Type = ArrayType>,
    C: DifferentiationContext<Tangent = Tracer<C>>,
    BatchingContext<C>: StagingContext<
            Type = ArrayType,
            Value = Tracer<BatchingContext<C>>,
            Constant = <C as Domain>::Constant,
            Operation = <C as Domain>::Operation,
        >,
{
    type Tangent = Tracer<BatchingContext<C>>;

    #[inline]
    fn validate_primal(&self, primal: &Self::Value) -> Result<(), ProgramError> {
        check_builders!(self.builder(), primal.context().builder()).map_err(|error| self.error(error))
    }

    /// Differentiation through a batching context is only available when the parent context is itself a staging
    /// context (this impl requires `C: Domain<Value = Tracer<C>>`), so primal values are always tracers and
    /// concretizing extractions on them cannot succeed.
    #[inline]
    fn supports_primal_concretization(&self) -> bool {
        false
    }
}

impl<C: Context<Type = ArrayType>> ProvidesContext<BatchingContext<C>> for BatchingContext<C>
where
    BatchingContext<C>: Context<Type = ArrayType>,
{
    #[inline]
    fn context(&self) -> BatchingContext<C> {
        self.clone()
    }
}

/// Batching tracer selected by an ordinary backend [`Domain`].
pub type BatchingTracer<D> =
    Tracer<BatchingContext<TracingContext<ArrayType, <D as Domain>::Constant, <D as Domain>::Operation>>>;

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
    // TODO(eaplatanios): Should this be taking a context as its first argument instead of extracting it from the
    //  inputs? Is `self` the right context anyway to replace the one that is currently being extracted?
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
        in_axes: impl Into<BatchAxes<I::To<Option<usize>>>>,
        out_axes: impl Into<BatchAxes<O::To<Option<usize>>>>,
        axis: impl Into<BatchAxis>,
    ) -> Result<O::To<Self::Value>, BatchingError>
    where
        Self: Context,
        I: Parameterized<
                Self::Value,
                ParameterStructure: Debug + PartialEq,
                Family: ParameterizedFamily<ArrayType>
                            + ParameterizedFamily<Self::Constant>
                            + ParameterizedFamily<Option<usize>>
                            + ParameterizedFamily<DomainTracer<Self>>
                            + ParameterizedFamily<BatchingTracer<Self>>,
            >,
        O: Parameterized<
                BatchingTracer<Self>,
                ParameterStructure: Debug + PartialEq,
                Family: ParameterizedFamily<ArrayType>
                            + ParameterizedFamily<Self::Value>
                            + ParameterizedFamily<Self::Constant>
                            + ParameterizedFamily<Option<usize>>
                            + ParameterizedFamily<DomainTracer<Self>>
                            + ParameterizedFamily<BatchingTracer<Self>>,
            >,
        I::To<Option<usize>>: Parameterized<Option<usize>, ParameterStructure = I::ParameterStructure>,
        O::To<Option<usize>>: Parameterized<Option<usize>, ParameterStructure = O::ParameterStructure>,
        I::To<DomainTracer<Self>>: Parameterized<
                DomainTracer<Self>,
                ParameterStructure = I::ParameterStructure,
                To<ArrayType> = I::To<ArrayType>,
                To<Self::Value> = I,
                To<Self::Constant> = I::To<Self::Constant>,
                To<Option<usize>> = I::To<Option<usize>>,
                To<BatchingTracer<Self>> = I::To<BatchingTracer<Self>>,
            >,
        O::To<DomainTracer<Self>>: Parameterized<
                DomainTracer<Self>,
                ParameterStructure = O::ParameterStructure,
                To<ArrayType> = O::To<ArrayType>,
                To<Self::Value> = O::To<Self::Value>,
                To<Self::Constant> = O::To<Self::Constant>,
                To<Option<usize>> = O::To<Option<usize>>,
                To<BatchingTracer<Self>> = O,
            >,
        I::To<ArrayType>: Parameterized<
                ArrayType,
                To<Self::Value> = I,
                To<Self::Constant> = I::To<Self::Constant>,
                To<DomainTracer<Self>> = I::To<DomainTracer<Self>>,
                To<BatchingTracer<Self>> = I::To<BatchingTracer<Self>>,
            >,
        O::To<ArrayType>: Parameterized<
                ArrayType,
                To<Self::Value> = O::To<Self::Value>,
                To<Self::Constant> = O::To<Self::Constant>,
                To<DomainTracer<Self>> = O::To<DomainTracer<Self>>,
                To<BatchingTracer<Self>> = O,
            >,
        Self::Operation: Clone
            + InterpretableOperation<ArrayType, Self::Value>
            + From<TransposeOperation>
            + BatchableOperation<DomainTracer<Self>, BatchingContext<DomainTracingContext<Self>>>,
        F: FnOnce(I::To<BatchingTracer<Self>>) -> Result<O, ProgramError>,
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
        // TODO(eaplatanios): Review this and figure out if there is a way to avoid having to do it this way.
        //  For example, could and should the context be passed as the first argument of this function like we do for
        //  other transforms?
        // Recover the interpretation context once from the program inputs (works for eager `()` and traced contexts
        // alike) and thread that single context through every instruction, since nested trace contexts are not
        // `Default`-constructible per instruction.
        let interpretation_context =
            input_values.iter().find_map(|value| value.interpretation_context()).ok_or_else(|| {
                ProgramError::from(crate::types::TypeError {
                    message: "cannot recover an interpretation context from the interpreted program inputs".to_string(),
                })
            })?;
        let output_values = program.interpret_with(
            input_values,
            |_, constant| self.lift(constant.clone()),
            |instruction, inputs| instruction.operation().interpret(&interpretation_context, inputs),
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
pub trait BatchContext: StagingContext<Type = ArrayType, Value = Tracer<Self>> {
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
        Self::Operation: Clone + From<TransposeOperation> + BatchableOperation<Tracer<Self>, BatchingContext<Self>>,
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
                        parent_tracer.transpose(permutation)
                    }
                }
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;

        Ok(O::To::<Tracer<Self>>::from_parameters(output_structure, parent_outputs)?)
    }
}

impl<C: StagingContext<Type = ArrayType, Value = Tracer<C>>> BatchContext for C {}

// TODO(eaplatanios): Review this function.
/// Returns the axis permutation that moves dimension `from` to position `to`, shifting the other
/// dimensions to preserve their relative order. Returns the identity permutation when
/// `from == to`.
pub(crate) fn move_axis_permutation(rank: usize, from: usize, to: usize) -> Vec<usize> {
    let mut permutation: Vec<usize> = (0..rank).collect();
    let axis = permutation.remove(from);
    permutation.insert(to, axis);
    permutation
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::operations::arithmetic::AddOperation;
    use crate::operations::constants::OneLike;
    use crate::operations::control_flow::ConditionOperation;
    use crate::operations::manipulation::Transpose;
    use crate::operations::trigonometric::{Sin, SinOperation};
    use crate::parameters::Placeholder;
    use crate::tests::{TestArray, TestArrayDomain};
    use crate::tracing_v2::LinearizationTracer;
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
                    let output: crate::tracing_v2::LinearizationTracer<'_, TestArrayDomain> =
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
        let context = EagerContext::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
        let outputs = ArrayOperation::<TestArray>::Add(AddOperation).batch(&context, &[left, right]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), Some(0));
        assert!(outputs[0].value().values().iter().all(|value| (value - 2.0).abs() < 1e-12));
    }

    #[test]
    fn test_lift_elementwise_binary_op() {
        let scalar = ArrayType::scalar(DataType::F64);
        let op = ArrayOperation::<TestArray>::Add(AddOperation);
        let (lifted_op, output_axes) =
            lift_elementwise(&op, &[scalar.clone(), scalar], &[Some(0), Some(0)], 5).unwrap();
        assert!(matches!(lifted_op, ArrayOperation::Add(_)));
        assert_eq!(output_axes, vec![Some(0)]);
    }

    #[test]
    fn test_lift_elementwise_unary_op() {
        let scalar = ArrayType::scalar(DataType::F64);
        let op = ArrayOperation::<TestArray>::Sin(SinOperation);
        let (lifted_op, output_axes) = lift_elementwise(&op, &[scalar], &[Some(0)], 7).unwrap();
        assert!(matches!(lifted_op, ArrayOperation::Sin(_)));
        assert_eq!(output_axes, vec![Some(0)]);
    }

    #[test]
    fn test_lift_elementwise_rejects_misaligned_input_axes() {
        let scalar = ArrayType::scalar(DataType::F64);
        let op = ArrayOperation::<TestArray>::Add(AddOperation);
        let err = lift_elementwise(&op, &[scalar.clone(), scalar], &[Some(0), Some(1)], 5).unwrap_err();
        // `lift_elementwise` is an operation-level batching helper, so its `BatchingError` rides up as a
        // `ProgramError::Custom` payload; recover the concrete error with `downcast_custom`.
        assert!(matches!(err.downcast_custom::<BatchingError>(), Some(BatchingError::MisalignedBatchAxes { .. }),));
    }

    #[test]
    fn test_lift_elementwise_passes_through_lane_uniform_inputs() {
        let scalar = ArrayType::scalar(DataType::F64);
        let op = ArrayOperation::<TestArray>::Add(AddOperation);
        let (lifted_op, output_axes) = lift_elementwise(&op, &[scalar.clone(), scalar], &[Some(0), None], 5).unwrap();
        assert!(matches!(lifted_op, ArrayOperation::Add(_)));
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
            .batch(|row| Ok(row.dot(&row, &DotDimensionNumbers::inner_product())), x, Some(0), Some(0), None)
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
            TestArrayDomain.batch(|row| row.transpose(vec![1, 0]), x, Some(0), Some(0), None).unwrap();

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
        use crate::operations::manipulation::Reshape;

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
    fn test_batch_stages_lane_uniform_condition_predicates() {
        // A lane-uniform *abstract* condition predicate under trace-time batching cannot be concretized to pick one
        // branch (previously this surfaced a `Concretization` error), so the staged batching rule batches both
        // branch programs at the operand lane axes and stages exactly one `condition` operation over them, with the
        // unbatched predicate passed through. Interpreting the staged batched program with both concrete predicate
        // values matches the eager operational path lane for lane (scale by 2 when true and by 3 when false).
        let parent_context = DomainTracingContext::<TestArrayDomain>::new();
        let builder = parent_context.builder().clone();
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let operand_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        let predicate_atom = builder.borrow_mut().add_input(predicate_type.clone());
        let operand_atom = builder.borrow_mut().add_input(operand_type);
        let predicate_tracer = parent_context.tracer(predicate_atom, None);
        let operand_tracer = parent_context.tracer(operand_atom, None);
        let output = BatchContext::batch(
            &parent_context,
            |(predicate, x)| {
                let condition = ConditionOperation::new(scalar_scale_branch(2.0), scalar_scale_branch(3.0)).unwrap();
                let op = ArrayOperation::Condition(Box::new(condition));
                let outputs = x.context().stage_operation(op, &[&predicate, &x])?;
                Ok(outputs.into_iter().next().unwrap())
            },
            (predicate_tracer, operand_tracer),
            (None, Some(0)),
            Some(0),
            None,
        )
        .unwrap();
        let output_atom = output.atom_id().unwrap();
        let program = builder
            .borrow()
            .clone()
            .build::<(TestArray, TestArray), TestArray>(vec![output_atom], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        let condition_count = program
            .instructions()
            .iter()
            .filter(|instruction| instruction.operation().name() == "condition")
            .count();
        assert_eq!(condition_count, 1, "{program}");
        let truthy = TestArray::new(ArrayType::scalar(DataType::Boolean), vec![1.0]);
        let falsy = TestArray::new(ArrayType::scalar(DataType::Boolean), vec![0.0]);
        let operand = TestArray::vector(vec![1.0, 4.0, 9.0]);
        assert_eq!(program.interpret((truthy, operand.clone())).unwrap().values, vec![2.0, 8.0, 18.0]);
        assert_eq!(program.interpret((falsy, operand)).unwrap().values, vec![3.0, 12.0, 27.0]);
    }

    #[test]
    fn test_batch_normalizes_lane_uniform_condition_branch_output_axes() {
        // The two branches of a staged batched condition may disagree on their natural output lane axes: here the
        // true branch scales the batched operand per lane (axis 0) while the false branch returns a lane-uniform
        // constant (no lane axis). The staged rule normalizes the false branch by appending a broadcast at its
        // tail, so the staged condition stays well-typed and both predicate values interpret correctly per lane.
        let mut constant_builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
        constant_builder.add_input(ArrayType::scalar(DataType::F64));
        let constant_output = constant_builder.add_constant(TestArray::scalar(7.0));
        let constant_branch = constant_builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![constant_output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let parent_context = DomainTracingContext::<TestArrayDomain>::new();
        let builder = parent_context.builder().clone();
        let predicate_atom = builder.borrow_mut().add_input(ArrayType::scalar(DataType::Boolean));
        let operand_atom =
            builder.borrow_mut().add_input(ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)])));
        let predicate_tracer = parent_context.tracer(predicate_atom, None);
        let operand_tracer = parent_context.tracer(operand_atom, None);
        let output = BatchContext::batch(
            &parent_context,
            |(predicate, x)| {
                let condition = ConditionOperation::new(scalar_scale_branch(2.0), constant_branch).unwrap();
                let op = ArrayOperation::Condition(Box::new(condition));
                let outputs = x.context().stage_operation(op, &[&predicate, &x])?;
                Ok(outputs.into_iter().next().unwrap())
            },
            (predicate_tracer, operand_tracer),
            (None, Some(0)),
            Some(0),
            None,
        )
        .unwrap();
        let output_atom = output.atom_id().unwrap();
        let program = builder
            .borrow()
            .clone()
            .build::<(TestArray, TestArray), TestArray>(vec![output_atom], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        let rendered = program.to_string();
        assert!(rendered.contains("broadcast"), "{rendered}");
        let truthy = TestArray::new(ArrayType::scalar(DataType::Boolean), vec![1.0]);
        let falsy = TestArray::new(ArrayType::scalar(DataType::Boolean), vec![0.0]);
        let operand = TestArray::vector(vec![1.0, 4.0, 9.0]);
        assert_eq!(program.interpret((truthy, operand.clone())).unwrap().values, vec![2.0, 8.0, 18.0]);
        assert_eq!(program.interpret((falsy, operand)).unwrap().values, vec![7.0, 7.0, 7.0]);
    }

    #[test]
    fn test_batch_lifts_lane_varying_condition_via_select() {
        // A runtime-predicate Condition inside batch with a lane-varying predicate: each lane
        // independently chooses between `on_true` (scale by 2.0) and `on_false` (scale by 3.0).
        // The trace-time `BatchingContext` dispatches the rule's `batch`, whose
        // lane-varying branch evaluates both branches over the operand axes and combines per lane
        // via `Select`. Multi-op staging emerges automatically through `Tracer`'s value-level traits.
        let predicate = TestArray::new(
            ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(4)])),
            vec![1.0, 0.0, 1.0, 0.0],
        );
        let operand = TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]);

        let output: TestArray = TestArrayDomain
            .batch(
                |(pred, operand)| {
                    let condition =
                        ConditionOperation::new(scalar_scale_branch(2.0), scalar_scale_branch(3.0)).unwrap();
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
                    let zero_op = ArrayOperation::<TestArray>::Zero(crate::operations::constants::ZeroOperation::new(
                        ArrayType::scalar(DataType::F64),
                    ));
                    let zero = x.context().stage_nullary_operation(zero_op)?.into_iter().next().unwrap();
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
    fn test_batch_dimension_sharding_derives_from_the_mapped_axis() {
        use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
        use crate::types::Size;

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();

        // A batched input whose mapped axis is sharded contributes that `ShardingDimension`.
        let batched = ArrayBatch::mapped(sharded_type.clone(), 0).unwrap();
        assert_eq!(
            batch_dimension_sharding(std::slice::from_ref(&batched)).unwrap(),
            ShardingDimension::sharded(["x"])
        );

        // A lane-uniform input contributes nothing, so the axis defaults to replicated.
        let lane_uniform = ArrayBatch::unbatched(sharded_type);
        assert_eq!(
            batch_dimension_sharding(std::slice::from_ref(&lane_uniform)).unwrap(),
            ShardingDimension::replicated()
        );

        // Batched inputs that disagree on their mapped-axis sharding are rejected.
        let replicated_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]))
            .with_sharding(Sharding::replicated(mesh, 2))
            .unwrap();
        let other = ArrayBatch::mapped(replicated_type, 0).unwrap();
        let error = batch_dimension_sharding(&[batched, other]).unwrap_err();
        assert!(matches!(
            error.downcast_custom::<BatchingError>(),
            Some(BatchingError::MisalignedBatchAxes { message })
                if message.contains("disagree on the sharding of their mapped axis"),
        ));
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
        let parent_context = DomainTracingContext::<TestArrayDomain>::new();
        let builder = parent_context.builder().clone();
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
                    %2:f64[3, 4] = broadcast [output_type=f64[3, 4], output_axes=[]] %1
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
