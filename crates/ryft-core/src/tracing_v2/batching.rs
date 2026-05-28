use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

use thiserror::Error;

use crate::differentiation::Tangent;
use crate::macros::check_count;
use crate::operations::arithmetic::Scale;
use crate::operations::constants::{One, OneLike, Zero, ZeroLike};
use crate::operations::trigonometric::{Cos, Sin};
use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily};
use crate::tracing::contexts::{Context, TracingContext};
use crate::tracing::domains::{DomainTracer, Tracer, TracerState, TracingDomain};
use crate::tracing::{AtomId, Program, ProgramBuilder, Traceable, TracingError, Value};
use crate::tracing_v2::operations::reshape::ReshapeOps;
use crate::tracing_v2::operations::{BroadcastInDim, SupportsReduce};
use crate::tracing_v2::{
    ArrayOperation, ConditionOperation, ControlFlowError, ControlFlowValue, Differentiable, LinearArrayOperation,
    MaybeCollective, NoOperationExtension, SupportsCollective, WhileOperation,
};
use crate::types::{ArrayType, Size, Typed};
use crate::{AddOperation, ElementwiseOperation, MulOperation, SubOperation, SupportsConstantLike};

/// Errors emitted by explicit batching and `vmap` helpers.
#[derive(Clone, Debug, Error, PartialEq, Eq, Hash)]
pub enum BatchingError {
    /// No mapped array leaves were provided and no explicit axis size is available.
    #[error("encountered an empty batch")]
    EmptyBatch,

    /// Different batched leaves disagreed on the mapped axis size.
    #[error("mismatched batch sizes across batched leaves")]
    MismatchedBatchSize,

    /// A primitive has no packed-array batching rule.
    #[error("missing batching rule for operation '{operation}'")]
    MissingBatchingRule {
        /// Name of the operation that could not be batched.
        operation: String,
    },

    /// A batching rule encountered batch axes it does not yet know how to align.
    #[error("{message}")]
    UnsupportedBatchAxisAlignment {
        /// Human-readable explanation of the unsupported axis placement.
        message: String,
    },

    /// A public `vmap` output did not carry the mapped axis.
    #[error("{message}")]
    UnbatchedOutput {
        /// Human-readable explanation of the output mismatch.
        message: String,
    },

    /// A mapped axis has dynamic size and no explicit axis size was provided.
    #[error("batch axis {axis} of array type {type_} has dynamic size")]
    DynamicBatchAxis {
        /// Physical array type containing the mapped axis.
        type_: ArrayType,

        /// Mapped axis.
        axis: usize,
    },

    /// A mapped axis is outside the rank of its array type.
    #[error("batch axis {axis} is out of bounds for array type {type_}")]
    InvalidBatchAxis {
        /// Physical array type.
        type_: ArrayType,

        /// Invalid axis.
        axis: usize,
    },

    /// Wrapper around parameter-lifting failures from the [`Parameterized`] infrastructure.
    #[error(transparent)]
    Parameter(#[from] ParameterError),
}

/// Packed array value carrying lane metadata for one batching transform.
///
/// [`ArrayBatch`] is the production batching carrier for `tracing_v2`: its [`ArrayType`] is the
/// physical type of `value`, so it includes the mapped lane dimension when [`ArrayBatch::batch_axis`]
/// is `Some`. The logical per-lane type is derived by removing that dimension.
///
/// A `None` batch axis is an explicit lane-uniform state. It means the value does not contain a
/// physical dimension for the current batch lanes and should be interpreted as the same value for
/// every lane. For example, a traced constant in `vmap(|x| x + 1)` is represented with
/// `batch_axis == None`, while `x` carries the mapped input axis. Runtime control-flow predicates
/// also require `None` today because a single predicate can select one branch for all lanes, while
/// a lane-varying predicate would need a dedicated batching rule. `None` is not limited to
/// rank-0 values: any shaped constant or operand can be lane-uniform when none of its physical
/// dimensions indexes the current lanes.
#[derive(Clone, Debug, PartialEq)]
pub struct ArrayBatch<V> {
    /// Physical array type of `value`.
    r#type: ArrayType,

    /// Packed array value.
    value: V,

    /// Axis in `type_` and `value` that represents the mapped batch dimension, or `None` when
    /// `value` is uniform across the current batch lanes.
    batch_axis: Option<usize>,
}

impl<V> ArrayBatch<V> {
    /// Creates a packed array batch from explicit physical metadata.
    ///
    /// # Parameters
    ///
    ///   - `type_`: Physical type of `value`. This type includes `batch_axis` when present.
    ///   - `value`: Physical array value.
    ///   - `batch_axis`: Mapped axis in `type_` and `value`, or `None` when `value` is shared
    ///     uniformly across lanes.
    pub fn new(type_: ArrayType, value: V, batch_axis: Option<usize>) -> Result<Self, TracingError> {
        if let Some(axis) = batch_axis
            && axis >= type_.rank()
        {
            return Err(BatchingError::InvalidBatchAxis { type_, axis }.into());
        }
        Ok(Self { r#type: type_, value, batch_axis })
    }

    /// Wraps a value that already contains a mapped axis.
    ///
    /// # Parameters
    ///
    ///   - `value`: Packed array value.
    ///   - `batch_axis`: Mapped axis in `value`.
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
    pub fn axis_size(&self) -> Result<Option<usize>, TracingError> {
        let Some(axis) = self.batch_axis else {
            return Ok(None);
        };
        let Some(size) = self.r#type.dimension(axis as isize).value() else {
            return Err(BatchingError::DynamicBatchAxis { type_: self.r#type.clone(), axis }.into());
        };
        Ok(Some(size))
    }

    /// Returns the scalar-body type obtained by removing the mapped axis.
    pub fn logical_type(&self) -> Result<ArrayType, TracingError> {
        let Some(axis) = self.batch_axis else {
            return Ok(self.r#type.clone());
        };
        Ok(self.r#type.without_dimension(axis)?.0)
    }
}

impl<V: Typed<ArrayType>> ArrayBatch<V> {
    /// Wraps a value that already contains a mapped axis.
    ///
    /// # Parameters
    ///
    ///   - `value`: Packed array value.
    ///   - `batch_axis`: Mapped axis in `value`.
    pub fn mapped(value: V, batch_axis: usize) -> Result<Self, TracingError> {
        Self::new(value.r#type().into_owned(), value, Some(batch_axis))
    }

    /// Wraps a value that is uniform across the current batch lanes.
    pub fn unbatched(value: V) -> Self {
        Self { r#type: value.r#type().into_owned(), value, batch_axis: None }
    }
}

impl<V> Parameter for ArrayBatch<V> where V: Parameter {}

impl<V> Typed<ArrayType> for ArrayBatch<V> {
    #[inline]
    fn r#type(&self) -> Cow<'_, ArrayType> {
        Cow::Borrowed(&self.r#type)
    }
}

impl<V: Display> Display for ArrayBatch<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.batch_axis {
            Some(axis) => write!(formatter, "batch[{}, axis={axis}]({})", self.r#type, self.value),
            None => write!(formatter, "batch[{}, lane-uniform]({})", self.r#type, self.value),
        }
    }
}

impl<V: Traceable<ArrayType>> Traceable<ArrayType> for ArrayBatch<V> {}

impl<V: Value<ArrayType>> Value<ArrayType> for ArrayBatch<V> {}

impl<V: ControlFlowValue> ControlFlowValue for ArrayBatch<V> {
    fn control_flow_predicate(&self) -> Result<bool, TracingError> {
        if self.batch_axis.is_some() {
            return Err(ControlFlowError::MissingTransformRule { transform: "batched predicate control flow" }.into());
        }
        self.value.control_flow_predicate()
    }
}

/// Batching rule for one staged operation.
///
/// `BatchableOperation::batch` takes batched physical inputs paired with mapped-axis metadata and returns batched
/// physical outputs with their lane axes — the same shape as JAX's per-primitive batching rules (`fn(batched_args,
/// batch_dims, **params) -> (result_value, result_dim)`). Most primitive rules are context-free and use the default
/// `Context = ()`. Active `vmap` supplies [`BatchingContext`] for rules that must replay captured nested programs or
/// stage backend extension operations through the enclosing transform.
///
/// # Contract
///
///   - **Axis alignment.** If two or more inputs carry a mapped axis (`batch_axis.is_some()`),
///     elementwise operations require them to agree on the axis position. When they disagree,
///     return [`BatchingError::UnsupportedBatchAxisAlignment`] with an error message that names
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
///     [`LinearArrayOperation`] materializes `Tangent::Zero` once via `V::zero` and dispatches to
///     the underlying V-level rule. Tangent's V-trait impls (`Add`, `Sub`, `Scale`, `Neg`,
///     `LeftDot`, `RightDot`, `Reshape`, `Transpose`) propagate `Zero` correctly, so the
///     short-circuit happens automatically.
///   - **Missing rule.** Variants without a defined batching rule (for example, a while-loop
///     whose loop predicate varies across lanes) return [`BatchingError::MissingBatchingRule`]
///     with an operation string that is human-readable and points at a likely fix.
///
/// The internal elementwise lifting helper computes the lifted op plus per-output axes for any pure elementwise op;
/// the matching value-level applicator composes the rule's axis arithmetic with [`InterpretableOperation::interpret`].
pub trait BatchableOperation<V, Context = ()>: Operation<ArrayType> + Sized
where
    V: Traceable<ArrayType>,
{
    /// Applies this operation to packed batched inputs, returning batched outputs with the
    /// resulting lane axes, using `context` for rules that need active transform state.
    fn batch(&self, context: &Context, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError>;
}

impl<V: Traceable<ArrayType>, Context> BatchableOperation<V, Context> for NoOperationExtension {
    fn batch(&self, _context: &Context, _inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError> {
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
/// carrier enums ([`ArrayOperation`], [`LinearArrayOperation`]) keep their explicit impls; coherence is preserved
/// because none of those types implement [`ElementwiseOperation`].
impl<O, V, Context> BatchableOperation<V, Context> for O
where
    O: ElementwiseOperation + Clone + InterpretableOperation<ArrayType, V>,
    V: Traceable<ArrayType> + BroadcastInDim + crate::tracing_v2::operations::transpose::Transpose,
{
    #[inline]
    fn batch(&self, _context: &Context, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError> {
        apply_elementwise_batch(self, inputs)
    }
}

/// Walks `inputs` to compute the per-lane input types, the per-input axis metadata, and the
/// common lane size — the three quantities every per-op batching rule consumes before
/// dispatching to its axis-arithmetic helper. Returns `MismatchedBatchSize` when two mapped
/// inputs disagree on their lane size and `DynamicBatchAxis` when any mapped input's axis is
/// non-static. When no inputs are mapped, the returned `axis_size` is `0` (no rule that needs a
/// lane size is ever invoked in that situation).
pub fn batch_input_metadata<V>(
    inputs: &[ArrayBatch<V>],
) -> Result<(Vec<ArrayType>, Vec<Option<usize>>, usize), TracingError> {
    let input_axes: Vec<Option<usize>> = inputs.iter().map(|input| input.batch_axis()).collect();
    let per_lane_input_types: Vec<ArrayType> =
        inputs.iter().map(|input| input.logical_type()).collect::<Result<Vec<_>, _>>()?;
    let mut axis_size: Option<usize> = None;
    for input in inputs {
        if let Some(size) = input.axis_size()? {
            match axis_size {
                Some(existing) if existing != size => return Err(BatchingError::MismatchedBatchSize.into()),
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
pub(crate) fn apply_with_axes<V, O>(
    lifted_op: &O,
    inputs: &[ArrayBatch<V>],
    output_axes: &[Option<usize>],
) -> Result<Vec<ArrayBatch<V>>, TracingError>
where
    V: Traceable<ArrayType>,
    O: InterpretableOperation<ArrayType, V>,
{
    let input_values: Vec<V> = inputs.iter().map(|input| input.value().clone()).collect();
    let output_values = lifted_op.interpret(input_values.as_slice())?;
    check_count!("output", output_values, output_axes.len(), TracingError);
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
pub(crate) fn apply_elementwise_batch<V, O>(
    operation: &O,
    inputs: &[ArrayBatch<V>],
) -> Result<Vec<ArrayBatch<V>>, TracingError>
where
    V: Traceable<ArrayType>
        + crate::tracing_v2::operations::broadcast::BroadcastInDim
        + crate::tracing_v2::operations::transpose::Transpose,
    O: Clone + InterpretableOperation<ArrayType, V>,
{
    let (_, original_input_axes, axis_size) = batch_input_metadata(inputs)?;
    let canonical_axis = original_input_axes.iter().copied().flatten().next();
    let aligned_inputs: Vec<ArrayBatch<V>> = match canonical_axis {
        None => inputs.to_vec(),
        Some(target) => inputs.iter().map(|input| align_batch_axis(input, target)).collect::<Result<_, _>>()?,
    };
    let (per_lane_types, input_axes, axis_size_after_alignment) = batch_input_metadata(&aligned_inputs)?;
    debug_assert_eq!(axis_size, axis_size_after_alignment);
    let (lifted_op, output_axes) = lift_elementwise(operation, &per_lane_types, &input_axes, axis_size)?;
    let common_axis = input_axes.iter().copied().flatten().next();
    let broadcasted_inputs: Vec<ArrayBatch<V>> = match common_axis {
        None => aligned_inputs,
        Some(batch_axis) => aligned_inputs
            .iter()
            .zip(input_axes.iter())
            .map(|(input, axis)| -> Result<ArrayBatch<V>, TracingError> {
                match axis {
                    Some(_) => Ok(input.clone()),
                    None => broadcast_to_batched(input, batch_axis, axis_size),
                }
            })
            .collect::<Result<_, _>>()?,
    };
    apply_with_axes(&lifted_op, &broadcasted_inputs, &output_axes)
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
pub(crate) fn align_batch_axis<V>(input: &ArrayBatch<V>, target_axis: usize) -> Result<ArrayBatch<V>, TracingError>
where
    V: Traceable<ArrayType> + crate::tracing_v2::operations::transpose::Transpose,
{
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
/// This is the canonical building block shared by [`apply_elementwise_batch`] and by mixed
/// batched/unbatched primitive rules (e.g., [`DotOperation::batch`](
/// crate::tracing_v2::operations::dot::DotOperation)): it inserts a new axis at `target_axis`
/// in the operand's type, broadcasts the value to that shape, and returns the result as a
/// batched [`ArrayBatch`].
///
/// Returns an error when called on an already-batched input — callers are expected to dispatch
/// the lane-uniform case explicitly.
///
/// # Parameters
///
///   - `operand`: Lane-uniform input to lift.
///   - `target_axis`: Position of the inserted batch axis in the output.
///   - `axis_size`: Size of the inserted batch axis.
pub(crate) fn broadcast_to_batched<V>(
    operand: &ArrayBatch<V>,
    target_axis: usize,
    axis_size: usize,
) -> Result<ArrayBatch<V>, TracingError>
where
    V: Traceable<ArrayType> + crate::tracing_v2::operations::broadcast::BroadcastInDim,
{
    if operand.batch_axis().is_some() {
        return Err(BatchingError::UnsupportedBatchAxisAlignment {
            message: "broadcast_to_batched expects a lane-uniform operand but received a batched value".to_string(),
        }
        .into());
    }
    let per_lane_type = operand.logical_type()?;
    let physical_type = per_lane_type.with_inserted_dimension(target_axis, Size::Static(axis_size))?;
    let broadcast_dimensions: Vec<usize> =
        (0..per_lane_type.rank()).map(|i| if i < target_axis { i } else { i + 1 }).collect();
    let broadcasted = operand.value().clone().broadcast_in_dim(physical_type.clone(), broadcast_dimensions);
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
) -> Result<(O, Vec<Option<usize>>), TracingError> {
    if input_types.len() != input_axes.len() {
        return Err(BatchingError::UnsupportedBatchAxisAlignment {
            message: format!(
                "elementwise lift for '{}' received {} input type(s) but {} axis annotation(s)",
                operation.name(),
                input_types.len(),
                input_axes.len(),
            ),
        }
        .into());
    }

    let mut common_axis: Option<usize> = None;
    for axis in input_axes.iter().copied().flatten() {
        match common_axis {
            Some(existing) if existing != axis => {
                return Err(BatchingError::UnsupportedBatchAxisAlignment {
                    message: format!(
                        "elementwise lift for '{}' cannot align batch axis {axis} with existing batch axis {existing}: \
                        the operands' batched lane dimensions are at different positions. Stage a Transpose to move \
                        one operand's batch axis to the other's position, or use `in_axes` to align them at the \
                        outer vmap boundary",
                        operation.name(),
                    ),
                }
                .into());
            }
            Some(_) => {}
            None => common_axis = Some(axis),
        }
    }

    let parent_physical_input_types = input_types
        .iter()
        .zip(input_axes.iter())
        .map(|(per_lane_type, axis)| -> Result<ArrayType, TracingError> {
            match axis {
                Some(k) => Ok(per_lane_type.with_inserted_dimension(*k, Size::Static(axis_size))?),
                None => Ok(per_lane_type.clone()),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    // For ops whose `infer_output_types` requires all inputs to share a shape (e.g.,
    // `SelectOperation`), broadcast the lane-uniform inputs to the common batched physical
    // shape via [`Broadcastable::broadcasted`] before inference. Ops with built-in
    // broadcasting semantics (e.g., `AddOperation`) accept the broadcasted shapes equally.
    let broadcasted_input_types: Vec<ArrayType> = if common_axis.is_some() {
        match crate::broadcasting::Broadcastable::broadcasted(parent_physical_input_types.as_slice()) {
            Ok(common) => parent_physical_input_types.iter().map(|_| common.clone()).collect(),
            Err(_) => parent_physical_input_types.clone(),
        }
    } else {
        parent_physical_input_types.clone()
    };
    let output_count = operation.infer_output_types(broadcasted_input_types.as_slice())?.len();
    Ok((operation.clone(), vec![common_axis; output_count]))
}

/// Builds the common error for zero-input carrier operations that must be handled by the staging path.
fn missing_zero_input_batch_rule(carrier: &str, kind: &str) -> TracingError {
    BatchingError::MissingBatchingRule {
        operation: format!(
            "{carrier}::{kind}: zero-input operations are lane-uniform by construction — stage them through the \
             active context, which handles the lane-uniform short-circuit, instead of invoking `batch` directly",
        ),
    }
    .into()
}

/// Dispatches non-control-flow [`ArrayOperation`] variants to their primitive batching rules.
///
/// Higher-order variants are intentionally returned as `None` so concrete impls can handle them with their specialized
/// recursive bounds instead of forcing the trait solver through one fully generic recursive carrier impl.
fn batch_array_non_control_operation<VCarrier, VRule, Extension>(
    operation: &ArrayOperation<VCarrier, ArrayType, Extension>,
    inputs: &[ArrayBatch<VRule>],
) -> Result<Option<Vec<ArrayBatch<VRule>>>, TracingError>
where
    VCarrier: Traceable<ArrayType>,
    VRule: Traceable<ArrayType>
        + Add<Output = VRule>
        + Sub<Output = VRule>
        + Mul<Output = VRule>
        + Div<Output = VRule>
        + Neg<Output = VRule>
        + Scale<VCarrier, Output = VRule>
        + crate::operations::constants::ConstantLike<f64>
        + Sin
        + Cos
        + ZeroLike
        + OneLike
        + crate::tracing_v2::operations::matrix::DotOps
        + ReshapeOps
        + BroadcastInDim
        + crate::tracing_v2::operations::reduce::Reduce
        + crate::tracing_v2::operations::compare::Compare<Output = VRule>
        + crate::tracing_v2::operations::logical::LogicalBinary
        + crate::tracing_v2::operations::logical::LogicalNot
        + crate::tracing_v2::operations::select::Select
        + crate::tracing_v2::operations::transpose::Transpose
        + ControlFlowValue,
    Vec<VRule>: Parameterized<VRule, To<VRule> = Vec<VRule>, ParameterStructure: Debug + PartialEq>,
{
    let outputs = match operation {
        ArrayOperation::Add => AddOperation.batch(&(), inputs)?,
        ArrayOperation::Sub => SubOperation.batch(&(), inputs)?,
        ArrayOperation::Mul => MulOperation.batch(&(), inputs)?,
        ArrayOperation::Div => crate::operations::arithmetic::DivOperation.batch(&(), inputs)?,
        ArrayOperation::Neg => crate::operations::arithmetic::NegOperation.batch(&(), inputs)?,
        ArrayOperation::Sin => crate::operations::trigonometric::SinOperation.batch(&(), inputs)?,
        ArrayOperation::Cos => crate::operations::trigonometric::CosOperation.batch(&(), inputs)?,
        ArrayOperation::Select => crate::tracing_v2::operations::select::SelectOperation.batch(&(), inputs)?,
        ArrayOperation::ZeroLike => crate::operations::constants::ZeroLikeOperation.batch(&(), inputs)?,
        ArrayOperation::OneLike => crate::operations::constants::OneLikeOperation.batch(&(), inputs)?,
        ArrayOperation::ConstantLike { value } => {
            crate::operations::constants::ConstantLikeOperation::<ArrayType, f64>::new(*value).batch(&(), inputs)?
        }
        ArrayOperation::Scale { factor } => {
            crate::operations::arithmetic::ScaleOperation::new(factor.clone()).batch(&(), inputs)?
        }
        ArrayOperation::Dot { dimensions } => {
            crate::tracing_v2::operations::dot::DotOperation::new(dimensions.clone()).batch(&(), inputs)?
        }
        ArrayOperation::Transpose { permutation } => {
            crate::tracing_v2::operations::transpose::TransposeOperation::new(permutation.clone()).batch(&(), inputs)?
        }
        ArrayOperation::Reshape { input_shape, output_shape } => {
            crate::tracing_v2::operations::reshape::ReshapeOperation::new(input_shape.clone(), output_shape.clone())
                .batch(&(), inputs)?
        }
        ArrayOperation::BroadcastInDim { target_type, broadcast_dimensions } => {
            crate::tracing_v2::operations::broadcast::BroadcastInDimOperation::new(
                target_type.clone(),
                broadcast_dimensions.clone(),
            )
            .batch(&(), inputs)?
        }
        ArrayOperation::Reduce { input_shape, axes, kind } => {
            crate::tracing_v2::operations::reduce::ReduceOperation::new(input_shape.clone(), axes.clone(), *kind)
                .batch(&(), inputs)?
        }
        ArrayOperation::Compare { kind } => {
            crate::tracing_v2::operations::compare::CompareOperation::new(*kind).batch(&(), inputs)?
        }
        ArrayOperation::Logical { kind } => {
            crate::tracing_v2::operations::logical::LogicalOperation::new(*kind).batch(&(), inputs)?
        }
        ArrayOperation::Collective { axis_name, kind } => {
            crate::tracing_v2::operations::collective::CollectiveOperation::new(axis_name.clone(), *kind)
                .batch(&(), inputs)?
        }
        ArrayOperation::Condition(_) | ArrayOperation::While(_) | ArrayOperation::Extension(_) => return Ok(None),
        ArrayOperation::Zero(_) => return Err(missing_zero_input_batch_rule("ArrayOperation", "Zero")),
        ArrayOperation::One(_) => return Err(missing_zero_input_batch_rule("ArrayOperation", "One")),
    };
    Ok(Some(outputs))
}

/// Blanket value-level batching impl for the [`ArrayOperation`] sum type.
impl<V, Extension> BatchableOperation<V, ()> for ArrayOperation<V, ArrayType, Extension>
where
    V: Value<ArrayType>
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Div<Output = V>
        + Neg<Output = V>
        + Scale<V, Output = V>
        + crate::operations::constants::ConstantLike<f64>
        + Sin
        + Cos
        + ZeroLike
        + OneLike
        + crate::tracing_v2::operations::matrix::DotOps
        + ReshapeOps
        + BroadcastInDim
        + crate::tracing_v2::operations::reduce::Reduce
        + crate::tracing_v2::operations::compare::Compare<Output = V>
        + crate::tracing_v2::operations::logical::LogicalBinary
        + crate::tracing_v2::operations::logical::LogicalNot
        + crate::tracing_v2::operations::select::Select
        + crate::tracing_v2::operations::transpose::Transpose
        + ControlFlowValue,
    Extension: BatchableOperation<V>,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: Debug + PartialEq>,
{
    fn batch(&self, context: &(), inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError> {
        if let Some(outputs) = batch_array_non_control_operation(self, inputs)? {
            return Ok(outputs);
        }
        match self {
            Self::Condition(condition) => condition.batch(context, inputs),
            Self::While(while_operation) => while_operation.batch(context, inputs),
            Self::Extension(extension) => extension.batch(&(), inputs),
            _ => unreachable!("non-control-flow ArrayOperation variants are handled above"),
        }
    }
}

/// Blanket active batching impl for the [`ArrayOperation`] sum type.
impl<C, Extension> BatchableOperation<Tracer<C>, BatchingContext<C>> for ArrayOperation<C::Value, ArrayType, Extension>
where
    C: Context<Type = ArrayType>,
    C::Value: Value<ArrayType> + ControlFlowValue,
    Tracer<C>: Add<Output = Tracer<C>>
        + Sub<Output = Tracer<C>>
        + Mul<Output = Tracer<C>>
        + Div<Output = Tracer<C>>
        + Neg<Output = Tracer<C>>
        + Scale<C::Value, Output = Tracer<C>>
        + crate::operations::constants::ConstantLike<f64>
        + Sin
        + Cos
        + ZeroLike
        + OneLike
        + crate::tracing_v2::operations::matrix::DotOps
        + ReshapeOps
        + BroadcastInDim
        + crate::tracing_v2::operations::reduce::Reduce
        + crate::tracing_v2::operations::compare::Compare<Output = Tracer<C>>
        + crate::tracing_v2::operations::logical::LogicalBinary
        + crate::tracing_v2::operations::logical::LogicalNot
        + crate::tracing_v2::operations::select::Select
        + crate::tracing_v2::operations::transpose::Transpose
        + ControlFlowValue,
    Extension: BatchableOperation<Tracer<C>, BatchingContext<C>>,
    Vec<Tracer<C>>: Parameterized<Tracer<C>, To<Tracer<C>> = Vec<Tracer<C>>, ParameterStructure: Debug + PartialEq>,
{
    fn batch(
        &self,
        context: &BatchingContext<C>,
        inputs: &[ArrayBatch<Tracer<C>>],
    ) -> Result<Vec<ArrayBatch<Tracer<C>>>, TracingError> {
        if let Some(outputs) = batch_array_non_control_operation(self, inputs)? {
            return Ok(outputs);
        }
        match self {
            Self::Condition(condition) => condition.batch(context, inputs),
            Self::While(while_operation) => while_operation.batch(context, inputs),
            Self::Extension(extension) => extension.batch(context, inputs),
            _ => unreachable!("non-control-flow ArrayOperation variants are handled above"),
        }
    }
}

/// Dispatches non-control-flow [`LinearArrayOperation`] variants to their primitive batching rules.
fn batch_linear_non_control_operation<VCarrier, VRule, Extension>(
    operation: &LinearArrayOperation<VCarrier, ArrayType, Extension>,
    inputs: &[ArrayBatch<VRule>],
) -> Result<Option<Vec<ArrayBatch<VRule>>>, TracingError>
where
    VCarrier: Traceable<ArrayType>,
    VRule: Traceable<ArrayType>
        + Add<Output = VRule>
        + Sub<Output = VRule>
        + Mul<Output = VRule>
        + Neg<Output = VRule>
        + Scale<VCarrier, Output = VRule>
        + ZeroLike
        + OneLike
        + crate::operations::constants::ConstantLike<f64>
        + crate::tracing_v2::operations::matrix::DotOps
        + crate::tracing_v2::operations::dot::LeftDot<VCarrier>
        + crate::tracing_v2::operations::dot::RightDot<VCarrier>
        + ReshapeOps
        + crate::tracing_v2::operations::broadcast::BroadcastInDim
        + crate::tracing_v2::operations::reduce::Reduce
        + crate::tracing_v2::operations::logical::LogicalBinary
        + crate::tracing_v2::operations::select::Select
        + crate::tracing_v2::operations::transpose::Transpose,
    Vec<VRule>: Parameterized<VRule, To<VRule> = Vec<VRule>, ParameterStructure: Debug + PartialEq>,
{
    let outputs = match operation {
        LinearArrayOperation::Add => crate::operations::arithmetic::AddOperation.batch(&(), inputs)?,
        LinearArrayOperation::Sub => crate::operations::arithmetic::SubOperation.batch(&(), inputs)?,
        LinearArrayOperation::Mul => crate::operations::arithmetic::MulOperation.batch(&(), inputs)?,
        LinearArrayOperation::Neg => crate::operations::arithmetic::NegOperation.batch(&(), inputs)?,
        LinearArrayOperation::ZeroLike => crate::operations::constants::ZeroLikeOperation.batch(&(), inputs)?,
        LinearArrayOperation::OneLike => crate::operations::constants::OneLikeOperation.batch(&(), inputs)?,
        LinearArrayOperation::ConstantLike { value } => {
            crate::operations::constants::ConstantLikeOperation::<ArrayType, f64>::new(*value).batch(&(), inputs)?
        }
        LinearArrayOperation::Scale { factor } => {
            crate::operations::arithmetic::ScaleOperation::new(factor.clone()).batch(&(), inputs)?
        }
        LinearArrayOperation::Transpose { permutation } => {
            crate::tracing_v2::operations::transpose::TransposeOperation::new(permutation.clone()).batch(&(), inputs)?
        }
        LinearArrayOperation::LeftDot { factor, dimensions } => {
            crate::tracing_v2::operations::dot::LeftDotOperation::new(factor.clone(), dimensions.clone())
                .batch(&(), inputs)?
        }
        LinearArrayOperation::RightDot { factor, dimensions } => {
            crate::tracing_v2::operations::dot::RightDotOperation::new(factor.clone(), dimensions.clone())
                .batch(&(), inputs)?
        }
        LinearArrayOperation::Reshape { input_shape, output_shape } => {
            crate::tracing_v2::operations::reshape::ReshapeOperation::new(input_shape.clone(), output_shape.clone())
                .batch(&(), inputs)?
        }
        LinearArrayOperation::BroadcastInDim { target_type, broadcast_dimensions } => {
            crate::tracing_v2::operations::broadcast::BroadcastInDimOperation::new(
                target_type.clone(),
                broadcast_dimensions.clone(),
            )
            .batch(&(), inputs)?
        }
        LinearArrayOperation::Reduce { input_shape, axes, kind } => {
            crate::tracing_v2::operations::reduce::ReduceOperation::new(input_shape.clone(), axes.clone(), *kind)
                .batch(&(), inputs)?
        }
        LinearArrayOperation::Condition(_) | LinearArrayOperation::While(_) | LinearArrayOperation::Extension(_) => {
            return Ok(None);
        }
        LinearArrayOperation::Zero(_) => return Err(missing_zero_input_batch_rule("LinearArrayOperation", "Zero")),
        LinearArrayOperation::One(_) => return Err(missing_zero_input_batch_rule("LinearArrayOperation", "One")),
    };
    Ok(Some(outputs))
}

/// Blanket value-level batching impl for the [`LinearArrayOperation`] sum type.
impl<V, Extension> BatchableOperation<V, ()> for LinearArrayOperation<V, ArrayType, Extension>
where
    V: Value<ArrayType>
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + Scale<V, Output = V>
        + ZeroLike
        + OneLike
        + crate::operations::constants::ConstantLike<f64>
        + crate::tracing_v2::operations::matrix::DotOps
        + crate::tracing_v2::operations::dot::LeftDot<V>
        + crate::tracing_v2::operations::dot::RightDot<V>
        + ReshapeOps
        + crate::tracing_v2::operations::broadcast::BroadcastInDim
        + crate::tracing_v2::operations::reduce::Reduce
        + crate::tracing_v2::operations::logical::LogicalBinary
        + crate::tracing_v2::operations::select::Select
        + crate::tracing_v2::operations::transpose::Transpose
        + ControlFlowValue,
    Extension: BatchableOperation<V>,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: Debug + PartialEq>,
{
    fn batch(&self, context: &(), inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError> {
        if let Some(outputs) = batch_linear_non_control_operation(self, inputs)? {
            return Ok(outputs);
        }
        match self {
            Self::Condition(condition) => condition.batch(context, inputs),
            Self::While(while_operation) => while_operation.batch(context, inputs),
            Self::Extension(extension) => extension.batch(&(), inputs),
            _ => unreachable!("non-control-flow LinearArrayOperation variants are handled above"),
        }
    }
}

/// Blanket active batching impl for the [`LinearArrayOperation`] sum type.
impl<C, Extension> BatchableOperation<Tracer<C>, BatchingContext<C>>
    for LinearArrayOperation<C::Value, ArrayType, Extension>
where
    C: Context<Type = ArrayType>,
    C::Value: Value<ArrayType> + ControlFlowValue,
    Tracer<C>: Add<Output = Tracer<C>>
        + Sub<Output = Tracer<C>>
        + Mul<Output = Tracer<C>>
        + Neg<Output = Tracer<C>>
        + Scale<C::Value, Output = Tracer<C>>
        + ZeroLike
        + OneLike
        + crate::operations::constants::ConstantLike<f64>
        + crate::tracing_v2::operations::matrix::DotOps
        + crate::tracing_v2::operations::dot::LeftDot<C::Value>
        + crate::tracing_v2::operations::dot::RightDot<C::Value>
        + ReshapeOps
        + crate::tracing_v2::operations::broadcast::BroadcastInDim
        + crate::tracing_v2::operations::reduce::Reduce
        + crate::tracing_v2::operations::logical::LogicalBinary
        + crate::tracing_v2::operations::select::Select
        + crate::tracing_v2::operations::transpose::Transpose
        + ControlFlowValue,
    Extension: BatchableOperation<Tracer<C>, BatchingContext<C>>,
    Vec<Tracer<C>>: Parameterized<Tracer<C>, To<Tracer<C>> = Vec<Tracer<C>>, ParameterStructure: Debug + PartialEq>,
{
    fn batch(
        &self,
        context: &BatchingContext<C>,
        inputs: &[ArrayBatch<Tracer<C>>],
    ) -> Result<Vec<ArrayBatch<Tracer<C>>>, TracingError> {
        if let Some(outputs) = batch_linear_non_control_operation(self, inputs)? {
            return Ok(outputs);
        }
        match self {
            Self::Condition(condition) => condition.batch(context, inputs),
            Self::While(while_operation) => while_operation.batch(context, inputs),
            Self::Extension(extension) => extension.batch(context, inputs),
            _ => unreachable!("non-control-flow LinearArrayOperation variants are handled above"),
        }
    }
}

impl<V, Extension> BatchableOperation<Tangent<ArrayType, V>, ()> for LinearArrayOperation<V, ArrayType, Extension>
where
    V: Value<ArrayType>
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + Scale<Output = V>
        + Zero<ArrayType>
        + One<ArrayType>
        + ZeroLike
        + OneLike
        + crate::operations::constants::ConstantLike<f64>
        + crate::tracing_v2::operations::matrix::DotOps
        + crate::tracing_v2::operations::dot::LeftDot<V>
        + crate::tracing_v2::operations::dot::RightDot<V>
        + ReshapeOps
        + crate::tracing_v2::operations::broadcast::BroadcastInDim
        + crate::tracing_v2::operations::reduce::Reduce
        + crate::tracing_v2::operations::logical::LogicalBinary
        + crate::tracing_v2::operations::select::Select
        + crate::tracing_v2::operations::transpose::Transpose
        + ControlFlowValue,
    Extension: BatchableOperation<V> + BatchableOperation<Tangent<ArrayType, V>, ()>,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: Debug + PartialEq>,
{
    fn batch(
        &self,
        context: &(),
        inputs: &[ArrayBatch<Tangent<ArrayType, V>>],
    ) -> Result<Vec<ArrayBatch<Tangent<ArrayType, V>>>, TracingError> {
        match self {
            Self::Condition(condition) => {
                return <ConditionOperation<V, LinearArrayOperation<V, ArrayType, Extension>, ArrayType>
                    as BatchableOperation<Tangent<ArrayType, V>, ()>>::batch(
                    condition, context, inputs,
                );
            }
            Self::While(while_op) => {
                return <WhileOperation<V, LinearArrayOperation<V, ArrayType, Extension>, ArrayType>
                    as BatchableOperation<Tangent<ArrayType, V>, ()>>::batch(
                    while_op, context, inputs,
                );
            }
            _ => {}
        }

        // First-order linear ops over tangent values: materialize `Tangent::Zero` to `V::zero`
        // once, dispatch to the V-level batching rule, and re-wrap as `Tangent::Value`. Symbolic
        // zero propagates through every Tangent V-trait impl (`Add`, `Sub`, `Neg`, `Scale`,
        // `LeftDot`, `RightDot`, `Reshape`, `Transpose`), so dispatching through `apply_with_axes`
        // on `lifted_op.interpret(tangent_values)` would also work — but the materialize-then-
        // dispatch path lets us reuse the V-level rule unchanged, which keeps the rule defined in
        // exactly one place.
        let always_materialize = matches!(self, LinearArrayOperation::ZeroLike | LinearArrayOperation::OneLike);
        if !always_materialize && inputs.iter().all(|input| input.value().is_zero()) {
            // Use the V-level rule purely for the lifted output types/axes; the value-level
            // interpret would have nothing to do for symbolic zeros.
            let materialized_zero_inputs = inputs
                .iter()
                .map(|input| -> Result<ArrayBatch<V>, TracingError> {
                    ArrayBatch::new(input.r#type().into_owned(), V::zero(&input.r#type())?, input.batch_axis())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let v_outputs = <LinearArrayOperation<V, ArrayType, Extension> as BatchableOperation<V>>::batch(
                self,
                &(),
                materialized_zero_inputs.as_slice(),
            )?;
            return v_outputs
                .into_iter()
                .map(|v_batch| -> Result<ArrayBatch<Tangent<ArrayType, V>>, TracingError> {
                    let output_type = v_batch.r#type().into_owned();
                    let output_axis = v_batch.batch_axis();
                    ArrayBatch::new(output_type.clone(), Tangent::zero(output_type), output_axis)
                })
                .collect();
        }

        let materialized = inputs
            .iter()
            .map(|input| -> Result<ArrayBatch<V>, TracingError> {
                let materialized_value = match input.value() {
                    Tangent::Zero(zero_type) => V::zero(zero_type)?,
                    Tangent::Value(value) => value.clone(),
                };
                ArrayBatch::new(input.r#type().into_owned(), materialized_value, input.batch_axis())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let v_outputs = <LinearArrayOperation<V, ArrayType, Extension> as BatchableOperation<V>>::batch(
            self,
            &(),
            materialized.as_slice(),
        )?;
        v_outputs
            .into_iter()
            .map(|v_batch| -> Result<ArrayBatch<Tangent<ArrayType, V>>, TracingError> {
                let output_type = v_batch.r#type().into_owned();
                let output_batch_axis = v_batch.batch_axis();
                let output_value = v_batch.into_value();
                ArrayBatch::new(output_type, Tangent::Value(output_value), output_batch_axis)
            })
            .collect()
    }
}

/// Trace context that introduces exactly one batched lane at a chosen axis.
///
/// [`BatchingContext`] is the active carrier for one level of `vmap`: it runs the user's function
/// against logical per-lane [`ArrayType`]s while leaving the runtime value type of the staged
/// program equal to the parent context's value type. Operations staged through this context are
/// lifted through their [`BatchableOperation`] rules at bind time. The lifted operation
/// is then staged into the parent context, so nested transforms compose by wrapping contexts
/// rather than by making each active transform pretend to be a backend domain.
///
/// Nested `vmap` composes by repeated context wrapping:
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
    /// [`Context::stage_operation`] hook takes `&self` but needs to record output axes as it
    /// stages instructions into the parent [`ProgramBuilder`].
    axis_table: Rc<RefCell<HashMap<AtomId, usize>>>,
}

impl<C: Context<Type = ArrayType>> BatchingContext<C> {
    /// Creates a new anonymous [`BatchingContext`] that wraps `parent_context` with the supplied lane size.
    #[inline]
    pub fn new(parent_context: C, axis_size: usize) -> Self {
        Self { parent_context, axis_size, axis_name: None, axis_table: Rc::new(RefCell::new(HashMap::new())) }
    }

    /// Creates a new [`BatchingContext`] with a named batched axis.
    #[inline]
    pub fn with_axis_name(parent_context: C, axis_size: usize, axis_name: impl Into<String>) -> Self {
        Self {
            parent_context,
            axis_size,
            axis_name: Some(axis_name.into()),
            axis_table: Rc::new(RefCell::new(HashMap::new())),
        }
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
        program: &Program<ArrayType, C::Value, O, Vec<C::Value>, Vec<C::Value>>,
        inputs: Vec<ArrayBatch<Tracer<C>>>,
    ) -> Result<Vec<ArrayBatch<Tracer<C>>>, TracingError>
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

impl<C> Context for BatchingContext<C>
where
    C: Context<Type = ArrayType>,
    C::Operation: BatchableOperation<Tracer<C>, Self>
        + SupportsConstantLike<ArrayType, C::Value, f64>
        + MaybeCollective
        + SupportsCollective<ArrayType, C::Value>
        + SupportsReduce<ArrayType, C::Value>,
    Tracer<C>: Mul<Output = Tracer<C>>,
{
    type Type = ArrayType;
    type Value = C::Value;
    type Operation = C::Operation;

    #[inline]
    fn builder(&self) -> &Rc<RefCell<ProgramBuilder<Self::Type, Self::Value, Self::Operation>>> {
        self.parent_context.builder()
    }

    fn stage_operation<I: std::borrow::Borrow<Tracer<Self>>>(
        &self,
        operation: Self::Operation,
        inputs: &[I],
    ) -> Result<Vec<Tracer<Self>>, TracingError> {
        if inputs.iter().any(|input| !Rc::ptr_eq(self.builder(), input.borrow().context().builder())) {
            return Err(self.error(TracingError::MismatchedProgramBuilders));
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
                .map(|parent_tracer| -> Result<Tracer<Self>, TracingError> {
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
        // Named-axis collective interception. If the staged operation is a collective targeting
        // this level's named axis, consume the lane axis here via `CollectiveOperation::batch`.
        // If it targets a different axis name, re-stage the same collective at the parent
        // context — which may be another `BatchingContext` (whose `stage_operation` re-runs this
        // same interception) or an ordinary tracing context.
        let output_batches = if let Some((collective_name, collective_kind)) =
            MaybeCollective::as_collective(&operation)
        {
            use crate::tracing_v2::operations::collective::{CollectiveOperation, SupportsCollective};
            let collective_name = collective_name.to_string();
            if self.axis_name.as_deref() == Some(collective_name.as_str()) {
                CollectiveOperation::new(collective_name, collective_kind)
                    .batch(self, parent_input_batches.as_slice())?
            } else {
                let parent_operation = <C::Operation as SupportsCollective<ArrayType, C::Value>>::collective_operation(
                    collective_name,
                    collective_kind,
                );
                let parent_input_tracers: Vec<&Tracer<C>> =
                    parent_input_batches.iter().map(|batch| batch.value()).collect();
                let parent_outputs =
                    self.parent_context.stage_operation(parent_operation, parent_input_tracers.as_slice())?;
                check_count!("output", parent_outputs, parent_input_batches.len(), TracingError);
                parent_outputs
                    .into_iter()
                    .zip(parent_input_batches.iter())
                    .map(|(parent_tracer, input_batch)| {
                        let physical_type = parent_tracer.r#type().into_owned();
                        ArrayBatch::new(physical_type, parent_tracer, input_batch.batch_axis())
                    })
                    .collect::<Result<Vec<_>, _>>()?
            }
        } else {
            operation.batch(self, parent_input_batches.as_slice())?
        };

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

impl<C> Differentiable for BatchingContext<C>
where
    C: Context<Type = ArrayType> + Differentiable<Type = ArrayType, Value = Tracer<C>, Tangent = Tracer<C>>,
    BatchingContext<C>: Context<Type = ArrayType, Value = <C as Context>::Value>,
{
    type Type = ArrayType;
    type Value = Tracer<BatchingContext<C>>;
    type Tangent = Tracer<BatchingContext<C>>;
    type CapturedValue = <C as Context>::Value;
    type LinearOperationCarrier<V>
        = C::LinearOperationCarrier<V>
    where
        V: Traceable<ArrayType>;

    #[inline]
    fn zero_primal(&self, type_: &ArrayType) -> Result<Tracer<BatchingContext<C>>, TracingError> {
        let value = Differentiable::zero_primal(&self.parent_context, type_)?;
        let atom = value.atom_id()?;
        Ok(self.tracer(atom, Some(type_.clone())))
    }

    #[inline]
    fn one_primal(&self, type_: &ArrayType) -> Result<Tracer<BatchingContext<C>>, TracingError> {
        let value = Differentiable::one_primal(&self.parent_context, type_)?;
        let atom = value.atom_id()?;
        Ok(self.tracer(atom, Some(type_.clone())))
    }

    #[inline]
    fn zero_tangent(&self, type_: &ArrayType) -> Result<Tracer<BatchingContext<C>>, TracingError> {
        let value = Differentiable::zero_tangent(&self.parent_context, type_)?;
        let atom = value.atom_id()?;
        Ok(self.tracer(atom, Some(type_.clone())))
    }

    #[inline]
    fn lift_captured_primal(&self, value: Self::CapturedValue) -> Result<Self::Value, TracingError> {
        Ok(self.constant(value))
    }
}

/// Batching tracer selected by an ordinary backend [`TracingDomain`].
pub type BatchingTracer<'domain, D> = Tracer<BatchingContext<TracingContext<'domain, D>>>;

/// Extension trait that exposes [`vmap`] as a method on any [`TracingDomain`] whose `Type` is
/// [`ArrayType`].
///
/// `domain.vmap(f, input, in_axes, out_axes, axis_size)` is the receiver-style entry point to
/// [`vmap`]; it mirrors how `jvp` sits on [`DifferentiableDomain`](crate::tracing_v2::DifferentiableDomain).
/// This domain-level API maps concrete values. Already-traced values use the active context's
/// [`BatchingContext::vmap`] path so nested transforms compose through context wrapping.
pub trait Vmap: TracingDomain<Type = ArrayType> {
    /// Maps a traced function over per-leaf array axes. Equivalent to the free function
    /// [`vmap`](crate::tracing_v2::batching::vmap); see that function for the full semantics.
    #[inline]
    fn vmap<'domain, F, Input, Output>(
        &'domain self,
        function: F,
        input: Input,
        in_axes: Input::To<Option<usize>>,
        out_axes: Output::To<Option<usize>>,
        axis_size: Option<usize>,
    ) -> Result<Output, TracingError>
    where
        Self: Sized,
        Self::Value: 'domain,
        Self::Constant: 'domain,
        Input: Parameterized<
                Self::Value,
                ParameterStructure: Debug + PartialEq,
                Family: ParameterizedFamily<ArrayType>
                            + ParameterizedFamily<Self::Constant>
                            + ParameterizedFamily<Option<usize>>
                            + ParameterizedFamily<DomainTracer<'domain, Self>>
                            + ParameterizedFamily<BatchingTracer<'domain, Self>>,
            >,
        Output: Parameterized<
                Self::Value,
                ParameterStructure: Debug + PartialEq,
                Family: ParameterizedFamily<ArrayType>
                            + ParameterizedFamily<Self::Constant>
                            + ParameterizedFamily<Option<usize>>
                            + ParameterizedFamily<DomainTracer<'domain, Self>>
                            + ParameterizedFamily<BatchingTracer<'domain, Self>>,
            >,
        Input::To<Option<usize>>: Parameterized<Option<usize>, ParameterStructure = Input::ParameterStructure>,
        Output::To<Option<usize>>: Parameterized<Option<usize>, ParameterStructure = Output::ParameterStructure>,
        Input::To<DomainTracer<'domain, Self>>: Parameterized<
                DomainTracer<'domain, Self>,
                ParameterStructure = Input::ParameterStructure,
                To<ArrayType> = Input::To<ArrayType>,
                To<Self::Value> = Input,
                To<Self::Constant> = Input::To<Self::Constant>,
                To<Option<usize>> = Input::To<Option<usize>>,
                To<BatchingTracer<'domain, Self>> = Input::To<BatchingTracer<'domain, Self>>,
                Family: ParameterizedFamily<ArrayType>
                            + ParameterizedFamily<Self::Value>
                            + ParameterizedFamily<Self::Constant>
                            + ParameterizedFamily<Option<usize>>
                            + ParameterizedFamily<DomainTracer<'domain, Self>>
                            + ParameterizedFamily<BatchingTracer<'domain, Self>>,
            >,
        Output::To<DomainTracer<'domain, Self>>: Parameterized<
                DomainTracer<'domain, Self>,
                ParameterStructure = Output::ParameterStructure,
                To<ArrayType> = Output::To<ArrayType>,
                To<Self::Value> = Output,
                To<Self::Constant> = Output::To<Self::Constant>,
                To<Option<usize>> = Output::To<Option<usize>>,
                To<BatchingTracer<'domain, Self>> = Output::To<BatchingTracer<'domain, Self>>,
                Family: ParameterizedFamily<ArrayType>
                            + ParameterizedFamily<Self::Value>
                            + ParameterizedFamily<Self::Constant>
                            + ParameterizedFamily<Option<usize>>
                            + ParameterizedFamily<DomainTracer<'domain, Self>>
                            + ParameterizedFamily<BatchingTracer<'domain, Self>>,
            >,
        Input::To<ArrayType>: Parameterized<
                ArrayType,
                To<Self::Value> = Input,
                To<Self::Constant> = Input::To<Self::Constant>,
                To<DomainTracer<'domain, Self>> = Input::To<DomainTracer<'domain, Self>>,
                To<BatchingTracer<'domain, Self>> = Input::To<BatchingTracer<'domain, Self>>,
            >,
        Output::To<ArrayType>: Parameterized<
                ArrayType,
                To<Self::Value> = Output,
                To<Self::Constant> = Output::To<Self::Constant>,
                To<DomainTracer<'domain, Self>> = Output::To<DomainTracer<'domain, Self>>,
                To<BatchingTracer<'domain, Self>> = Output::To<BatchingTracer<'domain, Self>>,
            >,
        Output::To<BatchingTracer<'domain, Self>>: Parameterized<
                BatchingTracer<'domain, Self>,
                To<ArrayType> = Output::To<ArrayType>,
                To<Self::Value> = Output,
                To<Self::Constant> = Output::To<Self::Constant>,
            >,
        Self::Operation: Clone
            + InterpretableOperation<ArrayType, Self::Value>
            + crate::tracing_v2::operations::transpose::SupportsTranspose<ArrayType, Self::Constant>
            + crate::tracing_v2::operations::collective::MaybeCollective
            + crate::tracing_v2::operations::collective::SupportsCollective<ArrayType, Self::Constant>
            + crate::tracing_v2::operations::reduce::SupportsReduce<ArrayType, Self::Constant>
            + crate::operations::constants::SupportsConstantLike<ArrayType, Self::Constant, f64>
            + for<'context> BatchableOperation<
                DomainTracer<'context, Self>,
                BatchingContext<TracingContext<'context, Self>>,
            >,
        for<'context> DomainTracer<'context, Self>: std::ops::Mul<Output = DomainTracer<'context, Self>>,
        F: FnOnce(
            Input::To<BatchingTracer<'domain, Self>>,
        ) -> Result<Output::To<BatchingTracer<'domain, Self>>, TracingError>,
    {
        vmap(self, function, input, in_axes, out_axes, axis_size)
    }
}

impl<D: TracingDomain<Type = ArrayType>> Vmap for D {}

/// Extension trait that exposes [`vmap`] as a method on active array contexts.
///
/// This is the already-traced counterpart of [`Vmap`]. It wraps the receiver in a [`BatchingContext`] and routes all
/// primitive binds through the current transform stack, so `vmap` composes with tracing, JVP, VJP, and other context
/// wrappers through the same [`Context::stage_operation`] path.
pub trait VmapContext: Context<Type = ArrayType> {
    /// Maps a traced function over per-leaf array axes inside this active context.
    #[allow(private_bounds)]
    fn vmap<F, Input, Output>(
        &self,
        function: F,
        input: Input,
        in_axes: Input::To<Option<usize>>,
        out_axes: Output::To<Option<usize>>,
        axis_size: Option<usize>,
    ) -> Result<Output, TracingError>
    where
        Self::Operation: Clone
            + crate::tracing_v2::operations::transpose::SupportsTranspose<ArrayType, Self::Value>
            + BatchableOperation<Tracer<Self>, BatchingContext<Self>>
            + MaybeCollective
            + SupportsCollective<ArrayType, Self::Value>
            + SupportsReduce<ArrayType, Self::Value>
            + SupportsConstantLike<ArrayType, Self::Value, f64>,
        Tracer<Self>: Mul<Output = Tracer<Self>>,
        Input: Parameterized<
                Tracer<Self>,
                ParameterStructure: Debug + PartialEq,
                Family: ParameterizedFamily<ArrayType>
                            + ParameterizedFamily<Self::Value>
                            + ParameterizedFamily<Option<usize>>
                            + ParameterizedFamily<Tracer<Self>>
                            + ParameterizedFamily<Tracer<BatchingContext<Self>>>,
            >,
        Output: Parameterized<
                Tracer<Self>,
                ParameterStructure: Debug + PartialEq,
                Family: ParameterizedFamily<ArrayType>
                            + ParameterizedFamily<Self::Value>
                            + ParameterizedFamily<Option<usize>>
                            + ParameterizedFamily<Tracer<Self>>
                            + ParameterizedFamily<Tracer<BatchingContext<Self>>>,
            >,
        Input::To<Option<usize>>: Parameterized<Option<usize>, ParameterStructure = Input::ParameterStructure>,
        Output::To<Option<usize>>: Parameterized<Option<usize>, ParameterStructure = Output::ParameterStructure>,
        Input::To<Self::Value>: Parameterized<Self::Value>,
        Output::To<Self::Value>: Parameterized<Self::Value>,
        Input::To<ArrayType>: Parameterized<
                ArrayType,
                To<Self::Value> = Input::To<Self::Value>,
                To<Tracer<BatchingContext<Self>>> = Input::To<Tracer<BatchingContext<Self>>>,
            >,
        Output::To<ArrayType>: Parameterized<
                ArrayType,
                To<Self::Value> = Output::To<Self::Value>,
                To<Tracer<BatchingContext<Self>>> = Output::To<Tracer<BatchingContext<Self>>>,
            >,
        Output::To<Tracer<BatchingContext<Self>>>: Parameterized<
                Tracer<BatchingContext<Self>>,
                To<ArrayType> = Output::To<ArrayType>,
                To<Self::Value> = Output::To<Self::Value>,
            >,
        F: FnOnce(
            Input::To<Tracer<BatchingContext<Self>>>,
        ) -> Result<Output::To<Tracer<BatchingContext<Self>>>, TracingError>,
    {
        let parent_context = self.clone();
        let input_structure = input.parameter_structure();
        let in_axes_structure = in_axes.parameter_structure();
        if in_axes_structure != input_structure {
            return Err(ParameterError::MismatchedParameterStructures {
                left_structure: format!("{input_structure:?}"),
                right_structure: format!("{in_axes_structure:?}"),
            }
            .into());
        }
        let input_tracers = input.into_parameters().collect::<Vec<_>>();
        let in_axes_values = in_axes.into_parameters().collect::<Vec<_>>();
        if input_tracers.is_empty() && axis_size.is_none() {
            return Err(BatchingError::EmptyBatch.into());
        }

        let mut resolved_axis_size = axis_size;
        let mut inputs_with_axes = Vec::with_capacity(input_tracers.len());
        for (tracer, axis) in input_tracers.into_iter().zip(in_axes_values.iter().copied()) {
            let parent_physical_type = tracer.r#type().into_owned();
            match axis {
                Some(batch_axis) => {
                    let (per_lane_type, dimension) = parent_physical_type.without_dimension(batch_axis)?;
                    let Some(size) = dimension.value() else {
                        return Err(
                            BatchingError::DynamicBatchAxis { type_: parent_physical_type, axis: batch_axis }.into()
                        );
                    };
                    match resolved_axis_size {
                        Some(existing_size) if existing_size != size => {
                            return Err(BatchingError::MismatchedBatchSize.into());
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

        let batching_context = BatchingContext::new(parent_context.clone(), resolved_axis_size);
        let parent_builder = parent_context.builder().clone();

        let mut batched_input_tracers = Vec::with_capacity(inputs_with_axes.len());
        for (parent_tracer, axis, logical_type) in inputs_with_axes.iter() {
            let atom = parent_tracer.atom_id()?;
            batching_context.register_axis(atom, *axis);
            batched_input_tracers.push(batching_context.tracer(atom, Some(logical_type.clone())));
        }
        let batched_input =
            Input::To::<Tracer<BatchingContext<Self>>>::from_parameters(input_structure, batched_input_tracers)?;
        let batched_output =
            function(batched_input).map_err(|error| parent_builder.borrow_mut().error.take().unwrap_or(error))?;
        parent_builder.borrow_mut().error.take().map_or(Ok(()), Err)?;

        let output_structure = batched_output.parameter_structure();
        let out_axes_structure = out_axes.parameter_structure();
        if out_axes_structure != output_structure {
            return Err(ParameterError::MismatchedParameterStructures {
                left_structure: format!("{output_structure:?}"),
                right_structure: format!("{out_axes_structure:?}"),
            }
            .into());
        }
        let output_atom_ids =
            batched_output.parameters().map(|tracer| tracer.atom_id()).collect::<Result<Vec<_>, _>>()?;
        let output_axes = output_atom_ids.iter().map(|atom| batching_context.axis_for(*atom)).collect::<Vec<_>>();
        drop(batched_output);
        drop(batching_context);

        let out_axes_values = out_axes.into_parameters().collect::<Vec<_>>();
        let parent_outputs = output_atom_ids
            .into_iter()
            .zip(output_axes)
            .zip(out_axes_values.iter().copied())
            .map(|((atom, current_axis), expected_axis)| -> Result<Tracer<Self>, TracingError> {
                let parent_tracer = parent_context.tracer(atom, None);
                match (current_axis, expected_axis) {
                    (None, None) => Ok(parent_tracer),
                    (None, Some(expected)) => Err(BatchingError::UnbatchedOutput {
                        message: format!("vmap output is lane-uniform but out_axes requested position {expected}"),
                    }
                    .into()),
                    (Some(current), None) => Err(BatchingError::UnbatchedOutput {
                        message: format!(
                            "vmap output is mapped on axis {current} but out_axes requested None: \
                            `out_axes = None` declares the output as lane-uniform (matching JAX's \
                            semantics) and requires the function not to produce a mapped output. \
                            To collapse the lane axis, apply an explicit reduction (e.g., \
                            `ReductionKind::Sum` over axis {current}) inside the function before \
                            returning",
                        ),
                    }
                    .into()),
                    (Some(current), Some(expected)) if current == expected => Ok(parent_tracer),
                    (Some(current), Some(expected)) => {
                        use crate::tracing_v2::operations::transpose::Transpose;
                        let rank = parent_tracer.r#type().as_ref().rank();
                        let permutation = move_axis_permutation(rank, current, expected);
                        Ok(parent_tracer.transpose(permutation))
                    }
                }
            })
            .collect::<Result<Vec<_>, TracingError>>()?;

        Ok(Output::from_parameters(output_structure, parent_outputs)?)
    }
}

impl<C: Context<Type = ArrayType>> VmapContext for C {}

/// Returns the axis permutation that moves dimension `from` to position `to`, shifting the other
/// dimensions to preserve their relative order. Returns the identity permutation when
/// `from == to`.
fn move_axis_permutation(rank: usize, from: usize, to: usize) -> Vec<usize> {
    let mut permutation: Vec<usize> = (0..rank).collect();
    let axis = permutation.remove(from);
    permutation.insert(to, axis);
    permutation
}

/// Maps a traced function over array axes selected per leaf by `in_axes` and places each output's
/// mapped axis at the position requested by `out_axes`.
///
/// Each `in_axes` leaf is either `Some(k)` (the input is mapped on axis `k` of its physical type)
/// or `None` (the input is lane-uniform / broadcast across the batched lanes). When at least one
/// input is mapped, the lane size is inferred from those inputs; `axis_size` can be supplied to
/// either pin the lane size explicitly or to drive a fully-broadcast `vmap` whose lane count
/// would otherwise be unobservable. The per-leaf `out_axes` selects where the mapped axis lands
/// in each output: `Some(k)` requests position `k` (an explicit transpose is staged when the
/// natural output axis differs), and `None` declares the corresponding output to be lane-uniform
/// (e.g., a value produced from broadcast inputs without staging any per-lane work).
///
/// This is the concrete-value entry point. Already-traced values use [`BatchingContext::vmap`] on their active
/// context.
pub fn vmap<'domain, D, F, Input, Output>(
    domain: &'domain D,
    function: F,
    input: Input,
    in_axes: Input::To<Option<usize>>,
    out_axes: Output::To<Option<usize>>,
    axis_size: Option<usize>,
) -> Result<Output, TracingError>
where
    D: TracingDomain<Type = ArrayType> + 'domain,
    D::Value: 'domain,
    D::Constant: 'domain,
    Input: Parameterized<
            D::Value,
            ParameterStructure: Debug + PartialEq,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<D::Constant>
                        + ParameterizedFamily<Option<usize>>
                        + ParameterizedFamily<DomainTracer<'domain, D>>
                        + ParameterizedFamily<BatchingTracer<'domain, D>>,
        >,
    Output: Parameterized<
            D::Value,
            ParameterStructure: Debug + PartialEq,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<D::Constant>
                        + ParameterizedFamily<Option<usize>>
                        + ParameterizedFamily<DomainTracer<'domain, D>>
                        + ParameterizedFamily<BatchingTracer<'domain, D>>,
        >,
    Input::To<Option<usize>>: Parameterized<Option<usize>, ParameterStructure = Input::ParameterStructure>,
    Output::To<Option<usize>>: Parameterized<Option<usize>, ParameterStructure = Output::ParameterStructure>,
    Input::To<DomainTracer<'domain, D>>: Parameterized<
            DomainTracer<'domain, D>,
            ParameterStructure = Input::ParameterStructure,
            To<ArrayType> = Input::To<ArrayType>,
            To<D::Value> = Input,
            To<D::Constant> = Input::To<D::Constant>,
            To<Option<usize>> = Input::To<Option<usize>>,
            To<BatchingTracer<'domain, D>> = Input::To<BatchingTracer<'domain, D>>,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<D::Value>
                        + ParameterizedFamily<D::Constant>
                        + ParameterizedFamily<Option<usize>>
                        + ParameterizedFamily<DomainTracer<'domain, D>>
                        + ParameterizedFamily<BatchingTracer<'domain, D>>,
        >,
    Output::To<DomainTracer<'domain, D>>: Parameterized<
            DomainTracer<'domain, D>,
            ParameterStructure = Output::ParameterStructure,
            To<ArrayType> = Output::To<ArrayType>,
            To<D::Value> = Output,
            To<D::Constant> = Output::To<D::Constant>,
            To<Option<usize>> = Output::To<Option<usize>>,
            To<BatchingTracer<'domain, D>> = Output::To<BatchingTracer<'domain, D>>,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<D::Value>
                        + ParameterizedFamily<D::Constant>
                        + ParameterizedFamily<Option<usize>>
                        + ParameterizedFamily<DomainTracer<'domain, D>>
                        + ParameterizedFamily<BatchingTracer<'domain, D>>,
        >,
    Input::To<ArrayType>: Parameterized<
            ArrayType,
            To<D::Value> = Input,
            To<D::Constant> = Input::To<D::Constant>,
            To<DomainTracer<'domain, D>> = Input::To<DomainTracer<'domain, D>>,
            To<BatchingTracer<'domain, D>> = Input::To<BatchingTracer<'domain, D>>,
        >,
    Output::To<ArrayType>: Parameterized<
            ArrayType,
            To<D::Value> = Output,
            To<D::Constant> = Output::To<D::Constant>,
            To<DomainTracer<'domain, D>> = Output::To<DomainTracer<'domain, D>>,
            To<BatchingTracer<'domain, D>> = Output::To<BatchingTracer<'domain, D>>,
        >,
    Output::To<BatchingTracer<'domain, D>>: Parameterized<
            BatchingTracer<'domain, D>,
            To<ArrayType> = Output::To<ArrayType>,
            To<D::Value> = Output,
            To<D::Constant> = Output::To<D::Constant>,
        >,
    D::Operation: Clone
        + InterpretableOperation<ArrayType, D::Value>
        + crate::tracing_v2::operations::transpose::SupportsTranspose<ArrayType, D::Constant>
        + MaybeCollective
        + SupportsCollective<ArrayType, D::Constant>
        + SupportsReduce<ArrayType, D::Constant>
        + SupportsConstantLike<ArrayType, D::Constant, f64>
        + for<'context> BatchableOperation<DomainTracer<'context, D>, BatchingContext<TracingContext<'context, D>>>,
    for<'context> DomainTracer<'context, D>: Mul<Output = DomainTracer<'context, D>>,
    F: FnOnce(Input::To<BatchingTracer<'domain, D>>) -> Result<Output::To<BatchingTracer<'domain, D>>, TracingError>,
{
    let structure = input.parameter_structure();
    let input_values = input.into_parameters().collect::<Vec<_>>();
    let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
    let parent_context = TracingContext::new(domain, builder.clone());
    let mut input_tracers = Vec::with_capacity(input_values.len());
    for value in input_values.iter() {
        let physical_type = value.r#type().into_owned();
        let atom = builder.borrow_mut().add_input(physical_type.clone());
        input_tracers.push(parent_context.tracer(atom, Some(physical_type)));
    }
    let traced_input = Input::To::<DomainTracer<'domain, D>>::from_parameters(structure.clone(), input_tracers)?;
    let traced_output: Output::To<DomainTracer<'domain, D>> =
        parent_context.vmap(function, traced_input, in_axes, out_axes, axis_size)?;
    builder.borrow_mut().error.take().map_or(Ok(()), Err)?;
    let output_structure = traced_output.parameter_structure();
    let output_atom_ids = traced_output.parameters().map(Tracer::atom_id).collect::<Result<Vec<_>, _>>()?;
    drop(traced_output);
    drop(parent_context);

    let builder = Rc::try_unwrap(builder).map_err(|_| TracingError::EscapedProgramBuilder)?.into_inner();
    let program: Program<ArrayType, D::Constant, D::Operation, Input::To<D::Constant>, Output::To<D::Constant>> =
        builder.build(output_atom_ids, structure, output_structure.clone())?;
    let output_values = program.interpret_with(
        input_values,
        |_, constant| domain.lift_constant(constant.clone()),
        |instruction, inputs| instruction.operation().interpret(inputs),
    )?;
    Ok(Output::from_parameters(output_structure, output_values)?)
}
