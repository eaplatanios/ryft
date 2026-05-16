use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

use ryft_macros::Parameter;
use thiserror::Error;

use crate::ElementwiseOperation;
use crate::differentiation::Tangent;
use crate::macros::check_count;
use crate::operations::arithmetic::Scale;
use crate::operations::constants::{One, OneLike, Zero, ZeroLike};
use crate::operations::trigonometric::{Cos, Sin};
use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily};
use crate::tracing::domains::{Domain, Tracer, TracerState, TracingContext, TracingDomain};
use crate::tracing::{AtomId, Program, ProgramBuilder, Traceable, TracingError, Value};
use crate::tracing_v2::operations::BroadcastInDim;
use crate::tracing_v2::operations::reshape::ReshapeOps;
use crate::tracing_v2::{
    ArrayOperation, ConditionOperation, ControlFlowError, ControlFlowValue, LinearArrayOperation, NoOperationExtension,
    WhileOperation,
};
use crate::types::{ArrayType, Size, Typed};

/// Errors emitted by explicit batching and `vmap` helpers.
#[derive(Clone, Debug, Error, PartialEq, Eq, Hash)]
pub enum BatchingError {
    /// No mapped array leaves were provided and no explicit axis size is available.
    #[error("encountered an empty batch")]
    EmptyBatch,

    /// Structured lanes did not share the same [`Parameterized`] shape in the reference fallback.
    #[error("mismatched parameter structures across batch lanes")]
    MismatchedParameterStructures,

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
#[derive(Clone, Debug, Parameter, PartialEq)]
pub struct ArrayBatch<V: Typed<ArrayType> + Parameter> {
    /// Physical array type of `value`.
    r#type: ArrayType,

    /// Packed array value.
    value: V,

    /// Axis in `type_` and `value` that represents the mapped batch dimension, or `None` when
    /// `value` is uniform across the current batch lanes.
    batch_axis: Option<usize>,
}

impl<V: Typed<ArrayType> + Parameter> ArrayBatch<V> {
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
    pub fn mapped(value: V, batch_axis: usize) -> Result<Self, TracingError>
    where
        V: Traceable<ArrayType>,
    {
        Self::new(value.r#type().into_owned(), value, Some(batch_axis))
    }

    /// Wraps a value that is uniform across the current batch lanes.
    pub fn unbatched(value: V) -> Self
    where
        V: Traceable<ArrayType>,
    {
        Self { r#type: value.r#type().into_owned(), value, batch_axis: None }
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

impl<V: Typed<ArrayType> + Parameter> Typed<ArrayType> for ArrayBatch<V> {
    #[inline]
    fn r#type(&self) -> Cow<'_, ArrayType> {
        Cow::Borrowed(&self.r#type)
    }
}

impl<V: Display + Typed<ArrayType> + Parameter> Display for ArrayBatch<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.batch_axis {
            Some(axis) => write!(formatter, "batch[{}, axis={axis}]({})", self.r#type, self.value),
            None => write!(formatter, "batch[{}, lane-uniform]({})", self.r#type, self.value),
        }
    }
}

impl<V: Traceable<ArrayType>> Traceable<ArrayType> for ArrayBatch<V> {}

impl<V: Value<ArrayType>> Value<ArrayType> for ArrayBatch<V> {}

impl<V: Traceable<ArrayType> + ControlFlowValue> ControlFlowValue for ArrayBatch<V> {
    fn control_flow_predicate(&self) -> Result<bool, TracingError> {
        if self.batch_axis.is_some() {
            return Err(ControlFlowError::MissingTransformRule { transform: "batched predicate control flow" }.into());
        }
        self.value.control_flow_predicate()
    }
}

/// Value-level capability to "batch" a carrier-level constant into the receiver type. Used by
/// higher-order batching rules (`ConditionOperation`, `WhileOperation`) when walking captured
/// inner programs whose constants are at `Self::CarrierValue` while the rule operates at `Self`.
///
/// The JAX equivalent is implicit: `lax.constant(value)` inside a batching rule dispatches
/// through whichever trace is currently active (a base trace evaluates, a `BatchTrace` lifts,
/// a `JitTrace` stages, etc.). In Rust we make this capability explicit as a trait on `Self`
/// (the rule's value type) whose associated `CarrierValue` names the carrier-level value type
/// that this `Self` knows how to lift constants from.
///
/// `template` is any existing [`ArrayBatch<Self>`] — its `value()` carries any context the
/// `Self` needs to construct a new value (e.g., a `Tracer` carries its `TracingContext`).
/// Most rules already have at least one input batch to use as the template.
pub trait Batchable: Sized + Traceable<ArrayType> {
    /// Carrier-level value type — the value type captured by the operation carrier whose batching
    /// rule is firing. For value-level rules over concrete `V`, this is `V` itself; for
    /// trace-time rules over `Tracer<'_, Parent>`, this is `Parent::Value`; for autodiff rules
    /// over `Tangent<ArrayType, V>`, this is the inner `V`.
    type CarrierValue: Traceable<ArrayType>;

    /// Lifts a `CarrierValue` constant into a lane-uniform [`ArrayBatch<Self>`] using `template`
    /// to supply any context needed to construct a `Self` value.
    fn batch(template: &ArrayBatch<Self>, value: Self::CarrierValue) -> Result<ArrayBatch<Self>, TracingError>;
}

/// Tangent lifting: a `V` constant becomes a non-zero `Tangent::Value` wrapping it. Used by
/// autodiff value-level batching paths (`jacrev` walking a linearized program over batched
/// `Tangent` inputs).
impl<V> Batchable for Tangent<ArrayType, V>
where
    V: Traceable<ArrayType>,
{
    type CarrierValue = V;

    #[inline]
    fn batch(_template: &ArrayBatch<Self>, value: V) -> Result<ArrayBatch<Self>, TracingError> {
        Ok(ArrayBatch::unbatched(Tangent::Value(value)))
    }
}

/// Value-level capability bundle satisfied by every type used as a `VRule` in the
/// [`BatchableOperation`] impl for [`ArrayOperation`].
///
/// `Vmappable<VCarrier>` is a single supertrait that aggregates the union of value-level
/// capabilities the ordinary-op carrier needs (arithmetic, broadcasting, transposing, reducing,
/// comparing, logical, selecting, control-flow predicate extraction, and the constant-lifting
/// [`Batchable`] facility). Implementing it for a concrete value type unlocks the full
/// [`vmap`](crate::tracing_v2::batching::vmap) surface for that type without spelling out the
/// dozen-plus individual bounds at every call site. Mirrors JAX's "this is a tracer-aware value
/// type" duck-typed contract.
///
/// A blanket implementation is provided for every type that already satisfies the union of
/// bounds, so end users do not normally implement `Vmappable` directly.
pub trait Vmappable<VCarrier>:
    Traceable<ArrayType>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Scale<VCarrier, Output = Self>
    + Sin
    + Cos
    + ZeroLike
    + OneLike
    + crate::tracing_v2::operations::matrix::DotOps
    + ReshapeOps
    + crate::tracing_v2::operations::broadcast::BroadcastInDim
    + crate::tracing_v2::operations::reduce::Reduce
    + crate::tracing_v2::operations::compare::Compare
    + crate::tracing_v2::operations::logical::LogicalBinary
    + crate::tracing_v2::operations::logical::LogicalNot
    + crate::tracing_v2::operations::select::Select
    + crate::tracing_v2::operations::transpose::Transpose
    + ControlFlowValue
    + Batchable<CarrierValue = VCarrier>
where
    VCarrier: Traceable<ArrayType>,
{
}

impl<V, VCarrier> Vmappable<VCarrier> for V
where
    VCarrier: Traceable<ArrayType>,
    V: Traceable<ArrayType>
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Div<Output = V>
        + Neg<Output = V>
        + Scale<VCarrier, Output = V>
        + Sin
        + Cos
        + ZeroLike
        + OneLike
        + crate::tracing_v2::operations::matrix::DotOps
        + ReshapeOps
        + crate::tracing_v2::operations::broadcast::BroadcastInDim
        + crate::tracing_v2::operations::reduce::Reduce
        + crate::tracing_v2::operations::compare::Compare
        + crate::tracing_v2::operations::logical::LogicalBinary
        + crate::tracing_v2::operations::logical::LogicalNot
        + crate::tracing_v2::operations::select::Select
        + crate::tracing_v2::operations::transpose::Transpose
        + ControlFlowValue
        + Batchable<CarrierValue = VCarrier>,
{
}

/// Trace-time lifting for a [`Tracer`]: stage a `constant` instruction in the parent context
/// (extracted from `template`'s tracer) and wrap the resulting tracer as an unbatched batch.
/// Used by [`BatchingDomain::stage`] when dispatching a rule whose `V == Tracer<Parent>` —
/// the rule's body invokes `Tracer::batch(template, parent_value)` and the constant lands
/// in the parent's [`TracingContext`].
impl<'parent, Parent> Batchable for Tracer<'parent, Parent>
where
    Parent: TracingDomain<Type = ArrayType>,
{
    type CarrierValue = Parent::Value;

    #[inline]
    fn batch(template: &ArrayBatch<Self>, value: Parent::Value) -> Result<ArrayBatch<Self>, TracingError> {
        Ok(ArrayBatch::unbatched(template.value().context().constant(value)))
    }
}

/// Batching rule for one staged operation.
///
/// `BatchableOperation::batch` is the value-level entry point: it takes batched physical inputs
/// paired with their mapped-axis metadata and returns batched physical outputs with their lane
/// axes — the same shape as JAX's per-primitive batching rules (`fn(batched_args, batch_dims,
/// **params) -> (result_value, result_dim)`).
///
/// The type-level [`lift`](Self::lift) sibling factors out just the axis-rewriting part of the
/// rule: it takes per-input type-and-axis metadata, returns the lifted operation to stage at the
/// parent level, and is used by nested-vmap splicing (where the splice has tracer inputs, not
/// concrete values, and wants to stage the lifted op directly into the outer trace rather than
/// run interpret on the inputs). In a Python framework like JAX, this distinction collapses
/// because the rule's calls into other primitives dispatch dynamically to either value-level or
/// trace-staging based on whether the argument is a `Tracer`. In Rust, the trait's `V` is bound
/// at impl time and higher-order ops (`Condition`, `While`) capture inner programs over a
/// specific value type, so the rule can't be uniformly polymorphic over "concrete value vs
/// tracer". Splitting `batch` from `lift` is the cleanest factoring.
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
/// The [`lift_elementwise`] helper computes the lifted op + per-output axes for any pure
/// elementwise op; [`apply_with_axes`] is the matching value-level applicator used by per-op
/// impls to compose the rule's axis arithmetic with [`InterpretableOperation::interpret`].
pub trait BatchableOperation<V: Traceable<ArrayType>>: Operation<ArrayType> + Sized {
    /// Applies this operation to packed batched inputs, returning batched outputs with the
    /// resulting lane axes — the per-primitive value-level batching rule.
    ///
    /// Higher-order rules that walk captured inner programs lift their carrier-level constants
    /// to `V` via the [`Batchable`] trait — the impl on `V` carries any context needed to
    /// construct a `V` from a [`Batchable::CarrierValue`]. The bound on those rules' impls
    /// names the `Batchable` capability explicitly.
    fn batch(&self, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError>;
}

impl<V: Traceable<ArrayType>> BatchableOperation<V> for NoOperationExtension {
    fn batch(&self, _inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError> {
        match *self {}
    }
}

/// Blanket [`BatchableOperation`] impl for any [`ElementwiseOperation`].
///
/// Mirrors the existing
/// [`impl<O: ElementwiseOperation> Operation<ArrayType> for O`](crate::operations::Operation):
/// every elementwise primitive automatically gets the canonical
/// [`apply_elementwise_batch`] rule, so per-op `BatchableOperation` impls do not have to be
/// written for elementwise primitives (`Add`, `Sub`, `Mul`, `Div`, `Neg`, `Sin`, `Cos`,
/// `Scale`, …). Ops with non-trivial axis arithmetic (`Dot`, `Transpose`, `Reshape`, …) and
/// the carrier enums ([`ArrayOperation`], [`LinearArrayOperation`]) keep their explicit
/// impls; coherence is preserved because none of those types implement
/// [`ElementwiseOperation`].
impl<O, V> BatchableOperation<V> for O
where
    O: ElementwiseOperation + Clone + InterpretableOperation<ArrayType, V>,
    V: Traceable<ArrayType> + BroadcastInDim + crate::tracing_v2::operations::transpose::Transpose,
{
    #[inline]
    fn batch(&self, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError> {
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
) -> Result<(Vec<ArrayType>, Vec<Option<usize>>, usize), TracingError>
where
    V: Typed<ArrayType> + Parameter,
{
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
pub fn apply_with_axes<V, O>(
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
pub fn apply_elementwise_batch<V, O>(
    operation: &O,
    inputs: &[ArrayBatch<V>],
) -> Result<Vec<ArrayBatch<V>>, TracingError>
where
    V: Traceable<ArrayType>
        + crate::tracing_v2::operations::broadcast::BroadcastInDim
        + crate::tracing_v2::operations::transpose::Transpose,
    O: Clone + Operation<ArrayType> + InterpretableOperation<ArrayType, V>,
{
    let (_, original_input_axes, axis_size) = batch_input_metadata(inputs)?;
    let canonical_axis = original_input_axes.iter().copied().flatten().next();
    let aligned_inputs: Vec<ArrayBatch<V>> = match canonical_axis {
        None => inputs.to_vec(),
        Some(target) => inputs
            .iter()
            .map(|input| align_batch_axis(input, target))
            .collect::<Result<_, _>>()?,
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
pub fn align_batch_axis<V>(input: &ArrayBatch<V>, target_axis: usize) -> Result<ArrayBatch<V>, TracingError>
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
pub fn broadcast_to_batched<V>(
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
pub fn lift_elementwise<O: Clone + Operation<ArrayType>>(
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
        let refs: Vec<&ArrayType> = parent_physical_input_types.iter().collect();
        match crate::broadcasting::Broadcastable::broadcasted(refs.as_slice()) {
            Ok(common) => parent_physical_input_types.iter().map(|_| common.clone()).collect(),
            Err(_) => parent_physical_input_types.clone(),
        }
    } else {
        parent_physical_input_types.clone()
    };
    let output_count = operation.infer_output_types(broadcasted_input_types.as_slice())?.len();
    Ok((operation.clone(), vec![common_axis; output_count]))
}

impl<VCarrier, VRule, Extension> BatchableOperation<VRule> for ArrayOperation<VCarrier, ArrayType, Extension>
where
    VCarrier: Value<ArrayType> + ControlFlowValue + Batchable<CarrierValue = VCarrier>,
    VRule: Vmappable<VCarrier>,
    Extension:
        Clone + BatchableOperation<VRule> + BatchableOperation<VCarrier> + InterpretableOperation<ArrayType, VRule>,
    Vec<VRule>: Parameterized<VRule, To<VRule> = Vec<VRule>, ParameterStructure: Debug + PartialEq>,
{
    fn batch(&self, inputs: &[ArrayBatch<VRule>]) -> Result<Vec<ArrayBatch<VRule>>, TracingError> {
        let missing = |kind: &str| -> TracingError {
            BatchingError::MissingBatchingRule {
                operation: format!(
                    "ArrayOperation::{kind}: zero-input operations are lane-uniform by \
                     construction — stage them through `TracingContext::stage`, which handles \
                     the lane-uniform short-circuit, instead of invoking `batch` directly",
                ),
            }
            .into()
        };
        match self {
            Self::Add => crate::operations::arithmetic::AddOperation.batch(inputs),
            Self::Sub => crate::operations::arithmetic::SubOperation.batch(inputs),
            Self::Mul => crate::operations::arithmetic::MulOperation.batch(inputs),
            Self::Div => crate::operations::arithmetic::DivOperation.batch(inputs),
            Self::Neg => crate::operations::arithmetic::NegOperation.batch(inputs),
            Self::Sin => crate::operations::trigonometric::SinOperation.batch(inputs),
            Self::Cos => crate::operations::trigonometric::CosOperation.batch(inputs),
            Self::Select => crate::tracing_v2::operations::select::SelectOperation.batch(inputs),
            Self::ZeroLike => crate::operations::constants::ZeroLikeOperation.batch(inputs),
            Self::OneLike => crate::operations::constants::OneLikeOperation.batch(inputs),
            Self::Scale { factor } => crate::operations::arithmetic::ScaleOperation::new(factor.clone()).batch(inputs),
            Self::Dot { dimensions } => {
                crate::tracing_v2::operations::dot::DotOperation::new(dimensions.clone()).batch(inputs)
            }
            Self::Transpose { permutation } => {
                crate::tracing_v2::operations::transpose::TransposeOperation::new(permutation.clone()).batch(inputs)
            }
            Self::Reshape { input_shape, output_shape } => {
                crate::tracing_v2::operations::reshape::ReshapeOperation::new(input_shape.clone(), output_shape.clone())
                    .batch(inputs)
            }
            Self::BroadcastInDim { target_type, broadcast_dimensions } => {
                crate::tracing_v2::operations::broadcast::BroadcastInDimOperation::new(
                    target_type.clone(),
                    broadcast_dimensions.clone(),
                )
                .batch(inputs)
            }
            Self::Reduce { input_shape, axes, kind } => {
                crate::tracing_v2::operations::reduce::ReduceOperation::new(
                    input_shape.clone(),
                    axes.clone(),
                    *kind,
                )
                .batch(inputs)
            }
            Self::Compare { kind } => crate::tracing_v2::operations::compare::CompareOperation::new(*kind).batch(inputs),
            Self::Logical { kind } => crate::tracing_v2::operations::logical::LogicalOperation::new(*kind).batch(inputs),
            Self::Collective { axis_name, kind } => {
                crate::tracing_v2::operations::collective::CollectiveOperation::new(axis_name.clone(), *kind)
                    .batch(inputs)
            }
            Self::Condition(condition) => condition.batch(inputs),
            Self::While(while_op) => while_op.batch(inputs),
            Self::Extension(extension) => extension.batch(inputs),
            Self::Zero(_) => Err(missing("Zero")),
            Self::One(_) => Err(missing("One")),
        }
    }
}

impl<VCarrier, VRule, Extension> BatchableOperation<VRule> for LinearArrayOperation<VCarrier, ArrayType, Extension>
where
    VCarrier: Value<ArrayType> + ControlFlowValue + Batchable<CarrierValue = VCarrier>,
    VRule: Traceable<ArrayType>
        + Add<Output = VRule>
        + Sub<Output = VRule>
        + Neg<Output = VRule>
        + Scale<VCarrier, Output = VRule>
        + ZeroLike
        + OneLike
        + crate::tracing_v2::operations::matrix::DotOps
        + crate::tracing_v2::operations::dot::LeftDot<VCarrier>
        + crate::tracing_v2::operations::dot::RightDot<VCarrier>
        + ReshapeOps
        + crate::tracing_v2::operations::broadcast::BroadcastInDim
        + crate::tracing_v2::operations::reduce::Reduce
        + crate::tracing_v2::operations::logical::LogicalBinary
        + crate::tracing_v2::operations::select::Select
        + crate::tracing_v2::operations::transpose::Transpose
        + ControlFlowValue
        + Batchable<CarrierValue = VCarrier>,
    Extension:
        Clone + BatchableOperation<VRule> + BatchableOperation<VCarrier> + InterpretableOperation<ArrayType, VRule>,
    Vec<VRule>: Parameterized<VRule, To<VRule> = Vec<VRule>, ParameterStructure: Debug + PartialEq>,
{
    fn batch(&self, inputs: &[ArrayBatch<VRule>]) -> Result<Vec<ArrayBatch<VRule>>, TracingError> {
        let missing = |kind: &str| -> TracingError {
            BatchingError::MissingBatchingRule {
                operation: format!(
                    "LinearArrayOperation::{kind}: zero-input operations are lane-uniform by \
                     construction — stage them through `TracingContext::stage`, which handles \
                     the lane-uniform short-circuit, instead of invoking `batch` directly",
                ),
            }
            .into()
        };
        match self {
            Self::Add => crate::operations::arithmetic::AddOperation.batch(inputs),
            Self::Sub => crate::operations::arithmetic::SubOperation.batch(inputs),
            Self::Neg => crate::operations::arithmetic::NegOperation.batch(inputs),
            Self::ZeroLike => crate::operations::constants::ZeroLikeOperation.batch(inputs),
            Self::OneLike => crate::operations::constants::OneLikeOperation.batch(inputs),
            Self::Scale { factor } => crate::operations::arithmetic::ScaleOperation::new(factor.clone()).batch(inputs),
            Self::Transpose { permutation } => {
                crate::tracing_v2::operations::transpose::TransposeOperation::new(permutation.clone()).batch(inputs)
            }
            Self::LeftDot { factor, dimensions } => {
                crate::tracing_v2::operations::dot::LeftDotOperation::new(factor.clone(), dimensions.clone())
                    .batch(inputs)
            }
            Self::RightDot { factor, dimensions } => {
                crate::tracing_v2::operations::dot::RightDotOperation::new(factor.clone(), dimensions.clone())
                    .batch(inputs)
            }
            Self::Reshape { input_shape, output_shape } => {
                crate::tracing_v2::operations::reshape::ReshapeOperation::new(input_shape.clone(), output_shape.clone())
                    .batch(inputs)
            }
            Self::BroadcastInDim { target_type, broadcast_dimensions } => {
                crate::tracing_v2::operations::broadcast::BroadcastInDimOperation::new(
                    target_type.clone(),
                    broadcast_dimensions.clone(),
                )
                .batch(inputs)
            }
            Self::Reduce { input_shape, axes, kind } => {
                crate::tracing_v2::operations::reduce::ReduceOperation::new(
                    input_shape.clone(),
                    axes.clone(),
                    *kind,
                )
                .batch(inputs)
            }
            Self::Condition(condition) => condition.batch(inputs),
            Self::While(while_op) => while_op.batch(inputs),
            Self::Extension(extension) => extension.batch(inputs),
            Self::Zero(_) => Err(missing("Zero")),
            Self::One(_) => Err(missing("One")),
        }
    }
}

impl<V, Extension> BatchableOperation<Tangent<ArrayType, V>> for LinearArrayOperation<V, ArrayType, Extension>
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
        + crate::tracing_v2::operations::matrix::DotOps
        + crate::tracing_v2::operations::dot::LeftDot<V>
        + crate::tracing_v2::operations::dot::RightDot<V>
        + ReshapeOps
        + crate::tracing_v2::operations::broadcast::BroadcastInDim
        + crate::tracing_v2::operations::reduce::Reduce
        + crate::tracing_v2::operations::logical::LogicalBinary
        + crate::tracing_v2::operations::select::Select
        + crate::tracing_v2::operations::transpose::Transpose
        + ControlFlowValue
        + Batchable<CarrierValue = V>,
    Extension: Clone
        + BatchableOperation<V>
        + BatchableOperation<Tangent<ArrayType, V>>
        + InterpretableOperation<ArrayType, V>,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: Debug + PartialEq>,
{
    fn batch(
        &self,
        inputs: &[ArrayBatch<Tangent<ArrayType, V>>],
    ) -> Result<Vec<ArrayBatch<Tangent<ArrayType, V>>>, TracingError> {
        match self {
            Self::Condition(condition) => {
                return <ConditionOperation<V, LinearArrayOperation<V, ArrayType, Extension>, ArrayType>
                    as BatchableOperation<Tangent<ArrayType, V>>>::batch(condition, inputs);
            }
            Self::While(while_op) => {
                return <WhileOperation<V, LinearArrayOperation<V, ArrayType, Extension>, ArrayType>
                    as BatchableOperation<Tangent<ArrayType, V>>>::batch(while_op, inputs);
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

/// Wrapping [`TracingDomain`] that lifts a parent domain `Parent` into a tracing context that
/// introduces exactly one batched lane at a chosen axis.
///
/// [`BatchingDomain`] is the carrier for one level of `vmap`: it traces the user's function against
/// logical per-lane [`ArrayType`]s while leaving the runtime value type of the staged program equal
/// to the parent's value type. Operations staged through this domain are lifted through their
/// [`BatchableOperation`] rules at trace time — the [`TracingDomain::stage`] override on
/// [`BatchingDomain`] applies the per-primitive lift and forwards the lifted operation to the
/// parent's staging path, so the resulting [`Program`] in the parent already contains the
/// physical batched instructions. Nested vmaps compose by repeated wrapping; the parent's stage
/// hook is invoked recursively, so the outermost trace sees an instruction that has been lifted
/// through every batching level in order.
///
/// Nested `vmap` composes by repeated wrapping at the type level:
/// `BatchingDomain<'_, BatchingDomain<'_, Parent>>` is a two-level batching context, and the
/// staged program's value type remains `Parent::Value` regardless of the nesting depth. Each level
/// of wrapping owns its own `axis_size` (and optionally `axis_name`) so the inner and outer lanes
/// can be different.
#[derive(Debug)]
pub struct BatchingDomain<'parent, Parent: TracingDomain<Type = ArrayType>> {
    /// Parent [`TracingDomain`] borrowed by this batching level.
    parent: &'parent Parent,

    /// Size of the batched lane this level introduces.
    axis_size: usize,

    /// Optional human-readable name for this batched axis. When supplied, future collective
    /// operations such as `psum`/`pmean`/`all_gather` will be able to address this axis by name
    /// from inside the batched function body. Today the name is metadata-only; collectives are a
    /// future extension.
    axis_name: Option<String>,

    /// Per-atom batch-axis annotations for atoms staged through this batching level. Missing keys
    /// are treated as lane-uniform (axis = `None`). Lives in interior mutability because the
    /// [`TracingDomain::stage`] override takes `&self` but needs to record output axes as it
    /// stages instructions into the underlying [`ProgramBuilder`].
    axis_table: RefCell<HashMap<AtomId, usize>>,
}

impl<'parent, Parent: TracingDomain<Type = ArrayType>> BatchingDomain<'parent, Parent> {
    /// Creates a new anonymous [`BatchingDomain`] that wraps `parent` with the supplied lane size.
    #[inline]
    pub fn new(parent: &'parent Parent, axis_size: usize) -> Self {
        Self { parent, axis_size, axis_name: None, axis_table: RefCell::new(HashMap::new()) }
    }

    /// Creates a new [`BatchingDomain`] with a named batched axis.
    #[inline]
    pub fn with_axis_name(parent: &'parent Parent, axis_size: usize, axis_name: impl Into<String>) -> Self {
        Self { parent, axis_size, axis_name: Some(axis_name.into()), axis_table: RefCell::new(HashMap::new()) }
    }

    /// Returns the parent tracing domain wrapped by this batching level.
    #[inline]
    pub fn parent(&self) -> &'parent Parent {
        self.parent
    }

    /// Returns the size of the batched lane introduced by this level.
    #[inline]
    pub fn axis_size(&self) -> usize {
        self.axis_size
    }

    /// Returns the optional name of this batched axis.
    #[inline]
    pub fn axis_name(&self) -> Option<&str> {
        self.axis_name.as_deref()
    }

    /// Returns the lane size of the named axis introduced by *this* batching level, if its name
    /// matches `axis_name`.
    ///
    /// This currently only inspects this level — full nested-vmap name resolution would walk the
    /// parent chain, which requires a `NamedAxisLookup` trait abstraction over arbitrary parents
    /// and is a separate follow-up. Today's collective ops only support single-level vmap by
    /// design, and this helper exists so future interception logic in
    /// [`BatchingDomain::stage`] (matched against [`MaybeCollective::as_collective`](
    /// crate::tracing_v2::operations::collective::MaybeCollective::as_collective)) can do
    /// efficient single-level matching without changing the dispatch path.
    #[inline]
    pub fn axis_size_for_name(&self, axis_name: &str) -> Option<usize> {
        if self.axis_name.as_deref() == Some(axis_name) {
            Some(self.axis_size)
        } else {
            None
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
}

impl<'parent, Parent: TracingDomain<Type = ArrayType>> Domain for BatchingDomain<'parent, Parent> {
    type Type = ArrayType;
    type Value = Parent::Value;
}

impl<'parent, Parent> TracingDomain for BatchingDomain<'parent, Parent>
where
    Parent: TracingDomain<Type = ArrayType>,
    Parent::OperationCarrier: for<'d> BatchableOperation<Tracer<'d, Parent>>,
{
    type OperationCarrier = Parent::OperationCarrier;

    fn stage<'domain>(
        &'domain self,
        context: &TracingContext<'domain, Self>,
        operation: Self::OperationCarrier,
        inputs: &[&Tracer<'domain, Self>],
    ) -> Result<Vec<Tracer<'domain, Self>>, TracingError>
    where
        Self: 'domain,
    {
        if inputs.iter().any(|input| !Rc::ptr_eq(&context.builder, &input.context().builder)) {
            return Err(context.error(TracingError::MismatchedProgramBuilders));
        }
        if context.builder.borrow().error.is_some() {
            let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
            let output_types = operation.infer_output_types(input_types.as_slice())?;
            return Ok(output_types
                .into_iter()
                .map(|r#type| Tracer::new(TracerState::Poison, r#type, context.clone()))
                .collect());
        }

        // Zero-input operations (e.g., `ZeroOperation`, `OneOperation`) are lane-uniform by
        // construction: every batch lane receives the same constant value, and there is no input
        // batch axis to lift through. Stage them directly into the parent's builder with an empty
        // input list and surface the resulting parent atoms as lane-uniform tracers (no entry in
        // `axis_table`). This sidesteps `BatchableOperation::batch`, which cannot construct
        // parent-level tracers without an input from which to extract a tracing context.
        if inputs.is_empty() {
            let parent_context = TracingContext::new(self.parent, context.builder.clone());
            let parent_outputs = parent_context.stage(operation, &[])?;
            return Ok(parent_outputs
                .into_iter()
                .map(|parent_tracer| -> Result<Tracer<'domain, Self>, TracingError> {
                    let parent_physical_type = parent_tracer.r#type().into_owned();
                    let atom = parent_tracer.atom_id()?;
                    Ok(context.tracer(atom, Some(parent_physical_type)))
                })
                .collect::<Result<Vec<_>, _>>()?);
        }

        let input_atom_ids: Vec<AtomId> = match inputs.iter().map(|input| input.atom_id()).collect::<Result<_, _>>() {
            Ok(ids) => ids,
            Err(error) => return Err(context.error(error)),
        };
        let logical_input_types: Vec<ArrayType> = inputs.iter().map(|input| input.r#type().into_owned()).collect();
        let input_axes: Vec<Option<usize>> = input_atom_ids.iter().map(|atom| self.axis_for(*atom)).collect();

        // Build parent-level input batches. Each ArrayBatch wraps the same atom as a parent-level
        // Tracer at the parent-physical (= this level's physical) type, with its recorded batch
        // axis. The rule's body (`operation.batch(...)`) then dispatches through Tracer's
        // value-level trait impls (Add, Scale, Dot, Select, etc.), so each primitive call inside
        // the rule stages directly into the parent's builder. Multi-op staging (e.g., lane-
        // varying Condition lowering to two branches + a per-lane Select) emerges automatically.
        let parent_context = TracingContext::new(self.parent, context.builder.clone());
        let mut parent_input_batches: Vec<ArrayBatch<Tracer<'domain, Parent>>> = Vec::with_capacity(inputs.len());
        for ((atom, logical_type), axis) in input_atom_ids.iter().zip(logical_input_types.iter()).zip(input_axes.iter())
        {
            let parent_physical_type = match axis {
                Some(k) => logical_type.with_inserted_dimension(*k, Size::Static(self.axis_size))?,
                None => logical_type.clone(),
            };
            let parent_tracer = parent_context.tracer(*atom, Some(parent_physical_type.clone()));
            parent_input_batches.push(ArrayBatch::new(parent_physical_type, parent_tracer, *axis)?);
        }
        let output_batches = operation.batch(parent_input_batches.as_slice())?;

        let mut output_tracers = Vec::with_capacity(output_batches.len());
        for output_batch in output_batches {
            let axis = output_batch.batch_axis();
            let parent_tracer = output_batch.into_value();
            let parent_physical_type = parent_tracer.r#type().into_owned();
            let atom = parent_tracer.atom_id()?;
            if let Some(k) = axis {
                self.axis_table.borrow_mut().insert(atom, k);
            }
            let logical_type = match axis {
                Some(k) => parent_physical_type.without_dimension(k)?.0,
                None => parent_physical_type,
            };
            output_tracers.push(context.tracer(atom, Some(logical_type)));
        }
        Ok(output_tracers)
    }
}

/// Extension trait that exposes [`vmap`] as a method on any [`TracingDomain`] whose `Type` is
/// [`ArrayType`].
///
/// `domain.vmap(f, input, in_axes, out_axes, axis_size)` is the receiver-style entry point to
/// [`vmap`]; it mirrors how `jvp` sits on [`DifferentiableDomain`](crate::tracing_v2::DifferentiableDomain).
/// Dispatch to the value-input vs. tracer-input strategy is handled by [`VmapDispatch`].
pub trait Vmap: TracingDomain<Type = ArrayType> {
    /// Maps a traced function over per-leaf array axes. Equivalent to the free function
    /// [`vmap`](crate::tracing_v2::batching::vmap); see that function for the full semantics.
    #[inline]
    #[allow(private_bounds)]
    fn vmap<'domain, F, Input, Output, Leaf, Marker>(
        &'domain self,
        function: F,
        input: Input,
        in_axes: Input::To<Option<usize>>,
        out_axes: Output::To<Option<usize>>,
        axis_size: Option<usize>,
    ) -> Result<Output, TracingError>
    where
        Self: Sized,
        Leaf: VmapDispatch<'domain, Self, Input, Output, Marker>,
        Input: Parameterized<Leaf, ParameterStructure: Debug + PartialEq, Family: ParameterizedFamily<Option<usize>>>,
        Output: Parameterized<Leaf, Family: ParameterizedFamily<Option<usize>>>,
        F: FnOnce(Leaf::FunctionInput) -> Result<Leaf::FunctionOutput, TracingError>,
    {
        vmap(self, function, input, in_axes, out_axes, axis_size)
    }
}

impl<D: TracingDomain<Type = ArrayType>> Vmap for D {}

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
/// The function dispatches via [`VmapDispatch`] so the same entry point covers both concrete
/// value inputs (which wrap `domain` in a single-level [`BatchingDomain`], trace `function`, and
/// replay the program over physical values) and already-traced tracer inputs (which lift the
/// per-primitive [`BatchableOperation::batch`] rule directly into the outer trace at staging
/// time). Mirrors the [`JvpDispatch`](crate::tracing_v2::forward) pattern used for `jvp`.
#[allow(private_bounds)]
pub fn vmap<'domain, D, F, Input, Output, Leaf, Marker>(
    domain: &'domain D,
    function: F,
    input: Input,
    in_axes: Input::To<Option<usize>>,
    out_axes: Output::To<Option<usize>>,
    axis_size: Option<usize>,
) -> Result<Output, TracingError>
where
    D: TracingDomain<Type = ArrayType>,
    Leaf: VmapDispatch<'domain, D, Input, Output, Marker>,
    Input: Parameterized<Leaf, ParameterStructure: Debug + PartialEq, Family: ParameterizedFamily<Option<usize>>>,
    Output: Parameterized<Leaf, Family: ParameterizedFamily<Option<usize>>>,
    F: FnOnce(Leaf::FunctionInput) -> Result<Leaf::FunctionOutput, TracingError>,
{
    Leaf::invoke(domain, function, input, in_axes, out_axes, axis_size)
}

/// Marker selecting concrete-value [`vmap`] dispatch.
#[doc(hidden)]
pub struct VmapDispatchValueMarker;

/// Marker selecting already-traced [`vmap`] dispatch.
#[doc(hidden)]
pub struct VmapDispatchTracerMarker;

/// Dispatch trait used by [`vmap`] so it can operate both on concrete values and on already
/// traced values.
///
/// The public transform is intentionally small; this trait is where the concrete and traced
/// execution strategies branch apart. Mirrors
/// [`JvpDispatch`](crate::tracing_v2::forward::JvpDispatch).
#[doc(hidden)]
pub trait VmapDispatch<'domain, D, Input, Output, Marker>: Traceable<ArrayType> + Parameter + Sized
where
    D: TracingDomain<Type = ArrayType>,
    Input: Parameterized<Self, ParameterStructure: Debug + PartialEq, Family: ParameterizedFamily<Option<usize>>>,
    Output: Parameterized<Self, Family: ParameterizedFamily<Option<usize>>>,
{
    /// Input type expected by the user-provided function.
    type FunctionInput;

    /// Output type produced by the user-provided function.
    type FunctionOutput;

    /// Invokes [`vmap`] for one leaf regime.
    fn invoke<F>(
        domain: &'domain D,
        function: F,
        input: Input,
        in_axes: Input::To<Option<usize>>,
        out_axes: Output::To<Option<usize>>,
        axis_size: Option<usize>,
    ) -> Result<Output, TracingError>
    where
        F: FnOnce(Self::FunctionInput) -> Result<Self::FunctionOutput, TracingError>;
}

/// Value-input dispatch for [`vmap`]: wraps `domain` in a single-level [`BatchingDomain`], traces
/// `function` against the per-lane logical types, and replays the staged program through
/// primitive batching rules over the original physical values.
impl<'domain, D, V, Input, Output> VmapDispatch<'domain, D, Input, Output, VmapDispatchValueMarker> for V
where
    D: TracingDomain<Type = ArrayType, Value = V> + 'domain,
    V: Traceable<ArrayType> + crate::tracing_v2::operations::transpose::Transpose + 'domain,
    Input: Parameterized<
            V,
            ParameterStructure: Debug + PartialEq,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<ArrayBatch<V>>
                        + ParameterizedFamily<Option<usize>>
                        + ParameterizedFamily<Tracer<'domain, BatchingDomain<'domain, D>>>,
        >,
    Output: Parameterized<
            V,
            ParameterStructure: Debug + PartialEq,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<ArrayBatch<V>>
                        + ParameterizedFamily<Option<usize>>
                        + ParameterizedFamily<Tracer<'domain, BatchingDomain<'domain, D>>>,
        >,
    Input::To<Option<usize>>: Parameterized<Option<usize>, ParameterStructure = Input::ParameterStructure>,
    Output::To<Option<usize>>: Parameterized<Option<usize>, ParameterStructure = Output::ParameterStructure>,
    Input::To<ArrayType>: Parameterized<
            ArrayType,
            To<V> = Input,
            To<Tracer<'domain, BatchingDomain<'domain, D>>> = Input::To<Tracer<'domain, BatchingDomain<'domain, D>>>,
        >,
    Output::To<ArrayType>: Parameterized<
            ArrayType,
            To<V> = Output,
            To<Tracer<'domain, BatchingDomain<'domain, D>>> = Output::To<Tracer<'domain, BatchingDomain<'domain, D>>>,
        >,
    Output::To<Tracer<'domain, BatchingDomain<'domain, D>>>: Parameterized<
            Tracer<'domain, BatchingDomain<'domain, D>>,
            To<ArrayType> = Output::To<ArrayType>,
            To<V> = Output,
        >,
    D::OperationCarrier: Clone + InterpretableOperation<ArrayType, V> + for<'d> BatchableOperation<Tracer<'d, D>>,
{
    type FunctionInput = Input::To<Tracer<'domain, BatchingDomain<'domain, D>>>;
    type FunctionOutput = Output::To<Tracer<'domain, BatchingDomain<'domain, D>>>;

    fn invoke<F>(
        domain: &'domain D,
        function: F,
        input: Input,
        in_axes: Input::To<Option<usize>>,
        out_axes: Output::To<Option<usize>>,
        axis_size: Option<usize>,
    ) -> Result<Output, TracingError>
    where
        F: FnOnce(Self::FunctionInput) -> Result<Self::FunctionOutput, TracingError>,
    {
        let structure = input.parameter_structure();
        let in_axes_structure = in_axes.parameter_structure();
        if in_axes_structure != structure {
            return Err(ParameterError::MismatchedParameterStructures {
                left_structure: format!("{:?}", structure),
                right_structure: format!("{in_axes_structure:?}"),
            }
            .into());
        }
        let input_values = input.into_parameters().collect::<Vec<_>>();
        let in_axes_values = in_axes.into_parameters().collect::<Vec<_>>();
        if input_values.is_empty() && axis_size.is_none() {
            return Err(BatchingError::EmptyBatch.into());
        }

        let mut resolved_axis_size = axis_size;
        let mut logical_types = Vec::with_capacity(input_values.len());
        let mut batched_inputs = Vec::with_capacity(input_values.len());
        for (value, axis) in input_values.into_iter().zip(in_axes_values.iter().copied()) {
            let physical_type = value.r#type().into_owned();
            match axis {
                Some(batch_axis) => {
                    let (logical_type, dimension) = physical_type.without_dimension(batch_axis)?;
                    let Some(size) = dimension.value() else {
                        return Err(BatchingError::DynamicBatchAxis { type_: physical_type, axis: batch_axis }.into());
                    };
                    match resolved_axis_size {
                        Some(existing) if existing != size => return Err(BatchingError::MismatchedBatchSize.into()),
                        Some(_) => {}
                        None => resolved_axis_size = Some(size),
                    }
                    logical_types.push(logical_type);
                    batched_inputs.push(ArrayBatch::new(physical_type, value, Some(batch_axis))?);
                }
                None => {
                    logical_types.push(physical_type.clone());
                    batched_inputs.push(ArrayBatch::new(physical_type, value, None)?);
                }
            }
        }

        let resolved_axis_size = resolved_axis_size.ok_or(BatchingError::EmptyBatch)?;
        let batching_domain = BatchingDomain::new(domain, resolved_axis_size);
        // SAFETY: `batching_domain` is owned by this function frame and outlives every reference
        // extended below. The borrow's lifetime is artificially extended to `'domain` only to
        // satisfy the closure's type-level lifetime constraint; no reference produced via this
        // extension escapes the enclosing function. Removing this transmute requires HRTB on `F`
        // plus GAT-style `FunctionInput`/`FunctionOutput`; deferred as a follow-up.
        let batching_domain_ref: &'domain BatchingDomain<'domain, D> = unsafe {
            std::mem::transmute::<&BatchingDomain<'domain, D>, &'domain BatchingDomain<'domain, D>>(&batching_domain)
        };

        // Trace `function` against the batching domain. Inputs are added to the builder with
        // their PHYSICAL types so the `stage` override can stage already-lifted instructions
        // whose input types match. We register the per-input batch axis on the batching domain
        // before running `function`, then collect output atoms + axes from the resulting trace.
        let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        let physical_input_types_vec: Vec<ArrayType> = batched_inputs.iter().map(|b| b.r#type().into_owned()).collect();
        let physical_input_axes_vec: Vec<Option<usize>> = in_axes_values.clone();
        let mut input_tracers_vec: Vec<Tracer<'domain, BatchingDomain<'domain, D>>> =
            Vec::with_capacity(physical_input_types_vec.len());
        {
            let context = TracingContext::new(batching_domain_ref, builder.clone());
            for (physical_type, axis) in physical_input_types_vec.iter().zip(physical_input_axes_vec.iter().copied()) {
                let atom = builder.borrow_mut().add_input(physical_type.clone());
                batching_domain_ref.register_axis(atom, axis);
                let logical_type = match axis {
                    Some(k) => physical_type.without_dimension(k)?.0,
                    None => physical_type.clone(),
                };
                input_tracers_vec.push(context.tracer(atom, Some(logical_type)));
            }
        }
        let input_structure = structure.clone();
        let physical_input_types =
            Input::To::<ArrayType>::from_parameters(input_structure.clone(), physical_input_types_vec)?;
        let input_tracers = Input::To::<Tracer<'domain, BatchingDomain<'domain, D>>>::from_parameters(
            input_structure.clone(),
            input_tracers_vec,
        )?;
        let output_tracers =
            function(input_tracers).map_err(|error| builder.borrow_mut().error.take().unwrap_or(error))?;
        let _ = builder.borrow_mut().error.take().map_or(Ok(()), Err)?;

        let output_structure = output_tracers.parameter_structure();
        let out_axes_structure = out_axes.parameter_structure();
        if out_axes_structure != output_structure {
            return Err(ParameterError::MismatchedParameterStructures {
                left_structure: format!("{output_structure:?}"),
                right_structure: format!("{out_axes_structure:?}"),
            }
            .into());
        }
        let output_atom_ids: Vec<AtomId> =
            output_tracers.parameters().map(|tracer| tracer.atom_id()).collect::<Result<Vec<_>, _>>()?;
        let output_axes_vec: Vec<Option<usize>> =
            output_atom_ids.iter().map(|atom| batching_domain_ref.axis_for(*atom)).collect();
        drop(output_tracers);

        let builder = Rc::try_unwrap(builder).map_err(|_| TracingError::EscapedProgramBuilder)?.into_inner();
        let program: Program<ArrayType, V, D::OperationCarrier, Input, Output> =
            builder.build(output_atom_ids, input_structure, output_structure.clone())?;
        let _ = physical_input_types;

        let physical_input_values: Vec<V> = batched_inputs.into_iter().map(|b| b.into_value()).collect();
        let physical_output_values: Vec<V> = program.interpret_with(
            physical_input_values,
            |_, constant: &V| Ok::<_, TracingError>(constant.clone()),
            |instruction, inputs| instruction.operation().interpret(inputs),
        )?;

        let out_axes_values = out_axes.into_parameters().collect::<Vec<_>>();
        let output_values = physical_output_values
            .into_iter()
            .zip(output_axes_vec.into_iter())
            .zip(out_axes_values.iter().copied())
            .map(|((value, current_axis), expected_axis)| -> Result<V, TracingError> {
                match (current_axis, expected_axis) {
                    (None, None) => Ok(value),
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
                    (Some(current), Some(expected)) if current == expected => Ok(value),
                    (Some(current), Some(expected)) => {
                        let rank = value.r#type().as_ref().rank();
                        let permutation = move_axis_permutation(rank, current, expected);
                        Ok(value.transpose(permutation))
                    }
                }
            })
            .collect::<Result<Vec<_>, TracingError>>()?;
        Ok(Output::from_parameters(output_structure, output_values)?)
    }
}

/// Tracer-input dispatch for [`vmap`]: replays `function` symbolically inside an enclosing
/// [`Tracer`] scope, building an inner [`BatchingDomain`] that shares the outer builder so the
/// per-primitive [`BatchableOperation::batch`] rule stages lifted instructions directly into the
/// outer trace.
impl<'domain, OuterDomain, V, Input, Output> VmapDispatch<'domain, OuterDomain, Input, Output, VmapDispatchTracerMarker>
    for Tracer<'domain, OuterDomain>
where
    OuterDomain: TracingDomain<Type = ArrayType, Value = V> + 'domain,
    OuterDomain::OperationCarrier: Clone
        + crate::tracing_v2::operations::transpose::SupportsTranspose<ArrayType, V>
        + for<'d> BatchableOperation<Tracer<'d, OuterDomain>>,
    V: Traceable<ArrayType> + Clone + 'domain,
    Input: Parameterized<
            Tracer<'domain, OuterDomain>,
            ParameterStructure: Debug + PartialEq,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<V>
                        + ParameterizedFamily<Option<usize>>
                        + ParameterizedFamily<Tracer<'domain, OuterDomain>>
                        + ParameterizedFamily<Tracer<'domain, BatchingDomain<'domain, OuterDomain>>>,
        >,
    Output: Parameterized<
            Tracer<'domain, OuterDomain>,
            ParameterStructure: Debug + PartialEq,
            Family: ParameterizedFamily<ArrayType>
                        + ParameterizedFamily<V>
                        + ParameterizedFamily<Option<usize>>
                        + ParameterizedFamily<Tracer<'domain, OuterDomain>>
                        + ParameterizedFamily<Tracer<'domain, BatchingDomain<'domain, OuterDomain>>>,
        >,
    Input::To<Option<usize>>: Parameterized<Option<usize>, ParameterStructure = Input::ParameterStructure>,
    Output::To<Option<usize>>: Parameterized<Option<usize>, ParameterStructure = Output::ParameterStructure>,
    Input::To<V>: Parameterized<V>,
    Output::To<V>: Parameterized<V>,
    Input::To<ArrayType>: Parameterized<
            ArrayType,
            To<V> = Input::To<V>,
            To<Tracer<'domain, BatchingDomain<'domain, OuterDomain>>> = Input::To<
                Tracer<'domain, BatchingDomain<'domain, OuterDomain>>,
            >,
        >,
    Output::To<ArrayType>: Parameterized<
            ArrayType,
            To<V> = Output::To<V>,
            To<Tracer<'domain, BatchingDomain<'domain, OuterDomain>>> = Output::To<
                Tracer<'domain, BatchingDomain<'domain, OuterDomain>>,
            >,
        >,
    Output::To<Tracer<'domain, BatchingDomain<'domain, OuterDomain>>>: Parameterized<
            Tracer<'domain, BatchingDomain<'domain, OuterDomain>>,
            To<ArrayType> = Output::To<ArrayType>,
            To<V> = Output::To<V>,
        >,
{
    type FunctionInput = Input::To<Tracer<'domain, BatchingDomain<'domain, OuterDomain>>>;
    type FunctionOutput = Output::To<Tracer<'domain, BatchingDomain<'domain, OuterDomain>>>;

    fn invoke<F>(
        domain: &'domain OuterDomain,
        function: F,
        input: Input,
        in_axes: Input::To<Option<usize>>,
        out_axes: Output::To<Option<usize>>,
        axis_size: Option<usize>,
    ) -> Result<Output, TracingError>
    where
        F: FnOnce(Self::FunctionInput) -> Result<Self::FunctionOutput, TracingError>,
    {
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
        let outer_context =
            input_tracers.first().map(|tracer| tracer.context().clone()).ok_or(BatchingError::EmptyBatch)?;

        let mut resolved_axis_size = axis_size;
        let mut logical_types = Vec::with_capacity(input_tracers.len());
        let mut inputs_with_axes: Vec<(Tracer<'domain, OuterDomain>, Option<usize>)> =
            Vec::with_capacity(input_tracers.len());
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
                    logical_types.push(per_lane_type);
                    inputs_with_axes.push((tracer, Some(batch_axis)));
                }
                None => {
                    logical_types.push(parent_physical_type);
                    inputs_with_axes.push((tracer, None));
                }
            }
        }
        let resolved_axis_size = resolved_axis_size.ok_or(BatchingError::EmptyBatch)?;

        let batching_domain = BatchingDomain::new(domain, resolved_axis_size);
        // SAFETY: see the matching SAFETY comment on the value-marker impl. `batching_domain` is
        // owned by this function frame and outlives the `function` call below.
        let batching_domain_ref: &'domain BatchingDomain<'domain, OuterDomain> = unsafe {
            std::mem::transmute::<&BatchingDomain<'domain, OuterDomain>, &'domain BatchingDomain<'domain, OuterDomain>>(
                &batching_domain,
            )
        };

        // Build a BatchingDomain context that shares the outer's builder; staging via the
        // batching-domain override appends lifted instructions directly into the outer trace.
        let outer_builder = outer_context.builder().clone();
        let inner_context = TracingContext::new(batching_domain_ref, outer_builder.clone());

        // Register input axes on the BatchingDomain for each existing outer atom, and wrap each
        // outer tracer's atom as a BatchingDomain tracer carrying the per-lane logical type.
        let mut inner_input_tracers_vec: Vec<Tracer<'domain, BatchingDomain<'domain, OuterDomain>>> =
            Vec::with_capacity(inputs_with_axes.len());
        for ((outer_tracer, axis), logical_type) in inputs_with_axes.iter().zip(logical_types.iter()) {
            let atom = outer_tracer.atom_id()?;
            batching_domain_ref.register_axis(atom, *axis);
            inner_input_tracers_vec.push(inner_context.tracer(atom, Some(logical_type.clone())));
        }
        let inner_input_tracers = Input::To::<Tracer<'domain, BatchingDomain<'domain, OuterDomain>>>::from_parameters(
            input_structure.clone(),
            inner_input_tracers_vec,
        )?;
        let inner_output_tracers =
            function(inner_input_tracers).map_err(|error| outer_builder.borrow_mut().error.take().unwrap_or(error))?;
        let _ = outer_builder.borrow_mut().error.take().map_or(Ok(()), Err)?;

        let output_structure = inner_output_tracers.parameter_structure();
        let out_axes_structure = out_axes.parameter_structure();
        if out_axes_structure != output_structure {
            return Err(ParameterError::MismatchedParameterStructures {
                left_structure: format!("{output_structure:?}"),
                right_structure: format!("{out_axes_structure:?}"),
            }
            .into());
        }
        let output_atom_ids: Vec<AtomId> =
            inner_output_tracers.parameters().map(|tracer| tracer.atom_id()).collect::<Result<Vec<_>, _>>()?;
        let output_axes_vec: Vec<Option<usize>> =
            output_atom_ids.iter().map(|atom| batching_domain_ref.axis_for(*atom)).collect();
        drop(inner_output_tracers);

        let out_axes_values = out_axes.into_parameters().collect::<Vec<_>>();
        let outer_output_tracers = output_atom_ids
            .into_iter()
            .zip(output_axes_vec.into_iter())
            .zip(out_axes_values.iter().copied())
            .map(|((atom, current_axis), expected_axis)| -> Result<Tracer<'domain, OuterDomain>, TracingError> {
                let outer_tracer = outer_context.tracer(atom, None);
                match (current_axis, expected_axis) {
                    (None, None) => Ok(outer_tracer),
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
                    (Some(current), Some(expected)) if current == expected => Ok(outer_tracer),
                    (Some(current), Some(expected)) => {
                        use crate::tracing_v2::operations::transpose::Transpose;
                        let rank = outer_tracer.r#type().as_ref().rank();
                        let permutation = move_axis_permutation(rank, current, expected);
                        Ok(outer_tracer.transpose(permutation))
                    }
                }
            })
            .collect::<Result<Vec<_>, TracingError>>()?;

        Ok(Output::from_parameters(output_structure, outer_output_tracers)?)
    }
}

/// Interprets a [`crate::tracing_v2::FlatProgram`] (a `Program` over `Vec<V_carrier>` input and
/// output) through batching rules at a different value type `V_rule`. Used by the batching
/// implementations of [`ConditionOperation`] and [`WhileOperation`] to recurse into their
/// captured branch / condition / body programs. The [`Batchable`] impl on `V_rule` is used to
/// lift captured carrier-level constants into `V_rule` (e.g., wrapping as `Tangent::Value` for
/// the tangent value-level case, or staging an outer constant for the trace-time case).
pub fn interpret_batched_flat_program<VCarrier, VRule, O>(
    program: &Program<ArrayType, VCarrier, O, Vec<VCarrier>, Vec<VCarrier>>,
    inputs: Vec<ArrayBatch<VRule>>,
) -> Result<Vec<ArrayBatch<VRule>>, TracingError>
where
    VCarrier: Traceable<ArrayType>,
    VRule: Traceable<ArrayType> + Batchable<CarrierValue = VCarrier>,
    O: Clone + BatchableOperation<VRule>,
{
    let template_input = inputs.first().cloned();
    program.interpret_with(
        inputs,
        |_, constant: &VCarrier| {
            let template = template_input.as_ref().ok_or_else(|| BatchingError::MissingBatchingRule {
                operation: "interpret_batched_flat_program cannot lift a constant when the inner program has no inputs"
                    .to_string(),
            })?;
            VRule::batch(template, constant.clone())
        },
        |instruction, instruction_inputs| instruction.operation().batch(instruction_inputs),
    )
}

/// Drives a lane-varying [`ConditionOperation`] by evaluating both branches over the same lane
/// configuration and combining the results via per-lane [`select::Select`](crate::tracing_v2::operations::select::Select).
pub fn batch_condition_with_lane_varying_predicate<VCarrier, VRule, O>(
    true_branch: &Program<ArrayType, VCarrier, O, Vec<VCarrier>, Vec<VCarrier>>,
    false_branch: &Program<ArrayType, VCarrier, O, Vec<VCarrier>, Vec<VCarrier>>,
    predicate_batch: &ArrayBatch<VRule>,
    predicate_axis: usize,
    operand_inputs: &[ArrayBatch<VRule>],
) -> Result<Vec<ArrayBatch<VRule>>, TracingError>
where
    VCarrier: Traceable<ArrayType>,
    VRule: Traceable<ArrayType>
        + ControlFlowValue
        + crate::tracing_v2::operations::select::Select
        + Batchable<CarrierValue = VCarrier>,
    O: Clone + BatchableOperation<VRule>,
{
    let true_outputs = interpret_batched_flat_program(true_branch, operand_inputs.to_vec())?;
    let false_outputs = interpret_batched_flat_program(false_branch, operand_inputs.to_vec())?;
    check_count!("output", true_outputs, false_outputs.len(), TracingError);
    true_outputs
        .into_iter()
        .zip(false_outputs)
        .map(|(true_output, false_output)| -> Result<ArrayBatch<VRule>, TracingError> {
            let output_axis = match (true_output.batch_axis(), false_output.batch_axis()) {
                (Some(left), Some(right)) if left != right => {
                    return Err(BatchingError::UnsupportedBatchAxisAlignment {
                        message: format!(
                            "condition branches produced lane-varying outputs at mismatched axes ({left} vs {right})",
                        ),
                    }
                    .into());
                }
                (Some(axis), _) | (_, Some(axis)) => axis,
                (None, None) => predicate_axis,
            };
            let selected = VRule::select(
                predicate_batch.value().clone(),
                true_output.value().clone(),
                false_output.value().clone(),
            )?;
            let output_type = selected.r#type().into_owned();
            ArrayBatch::new(output_type, selected, Some(output_axis))
        })
        .collect()
}
