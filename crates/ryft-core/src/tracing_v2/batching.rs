use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Neg, Sub};

use ryft_macros::Parameter;
use thiserror::Error;

use crate::differentiation::Tangent;
use crate::macros::check_count;
use crate::operations::arithmetic::Scale;
use crate::operations::constants::{One, OneLike, Zero, ZeroLike};
use crate::operations::trigonometric::{Cos, Sin};
use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily};
use crate::tracing::domains::{Domain, Tracer, TracingContext, TracingDomain};
use crate::tracing::{Program, Traceable, TracingError, Value};
use crate::tracing_v2::operations::reshape::ReshapeOps;
use crate::tracing_v2::{
    ArrayOperation, ConditionOperation, ControlFlowError, ControlFlowValue, LinearArrayOperation,
    NoOperationExtension, WhileOperation,
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

/// Output of a [`BatchableOperation::lift`] call: the lifted operation to stage at the parent
/// level, the per-output parent-physical types, and the per-output axis metadata.
#[derive(Clone, Debug)]
pub struct BatchingOutput<O> {
    /// Operation to stage at the parent (outer) level. May differ from the input operation when
    /// axis arguments need to be bumped past the introduced batch dimension.
    operation: O,

    /// Parent-physical output types, with the introduced batch dimension included.
    output_types: Vec<ArrayType>,

    /// Mapped-axis position within each output's parent-physical type, or `None` if the output
    /// is lane-uniform.
    output_axes: Vec<Option<usize>>,
}

impl<O> BatchingOutput<O> {
    /// Creates a new [`BatchingOutput`].
    #[inline]
    pub fn new(operation: O, output_types: Vec<ArrayType>, output_axes: Vec<Option<usize>>) -> Self {
        Self { operation, output_types, output_axes }
    }

    /// Returns the operation to stage at the parent level.
    #[inline]
    pub fn operation(&self) -> &O {
        &self.operation
    }

    /// Returns the parent-physical output types produced by this batching rule.
    #[inline]
    pub fn output_types(&self) -> &[ArrayType] {
        &self.output_types
    }

    /// Returns the mapped output axes produced by this batching rule.
    #[inline]
    pub fn output_axes(&self) -> &[Option<usize>] {
        &self.output_axes
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
/// impls to compose the lift with [`InterpretableOperation::interpret`].
pub trait BatchableOperation<V: Traceable<ArrayType>>: Operation<ArrayType> + Sized {
    /// Applies this operation to packed batched inputs, returning batched outputs with the
    /// resulting lane axes.
    ///
    /// # Parameters
    ///
    ///   - `inputs`: Physical input values paired with their mapped-axis metadata.
    fn batch(&self, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError>;

    /// Type-level lift: rewrites this operation's axis arguments through one batching level.
    /// Returns the lifted operation to stage at the parent level along with per-output axis
    /// metadata. Used by nested-vmap splicing, which has only types (no concrete values) to
    /// work with.
    ///
    /// For ops whose batching can't be expressed as a single lifted op (such as lane-varying
    /// [`ConditionOperation`]s, which need to evaluate both branches and combine via per-lane
    /// `select`), this surfaces [`BatchingError::MissingBatchingRule`].
    ///
    /// # Parameters
    ///
    ///   - `input_types`: Per-lane logical types of each input (rank one less than the
    ///     parent-physical type when the corresponding `input_axes` entry is `Some(_)`).
    ///   - `input_axes`: Mapped-axis position within each input's parent-physical type, or
    ///     `None` if lane-uniform.
    ///   - `axis_size`: Size of the batched lane this level introduces.
    fn lift(
        &self,
        input_types: &[ArrayType],
        input_axes: &[Option<usize>],
        axis_size: usize,
    ) -> Result<BatchingOutput<Self>, TracingError>;
}

impl<V: Traceable<ArrayType>> BatchableOperation<V> for NoOperationExtension {
    fn batch(&self, _inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError> {
        match *self {}
    }

    fn lift(
        &self,
        _input_types: &[ArrayType],
        _input_axes: &[Option<usize>],
        _axis_size: usize,
    ) -> Result<BatchingOutput<Self>, TracingError> {
        match *self {}
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

/// Reconstructs the parent-physical input types from per-lane logical types and input axes,
/// inserting `axis_size` at each mapped position. Used by per-op `lift` impls to feed
/// `infer_output_types` on the parent (post-lift) operation.
pub fn parent_physical_input_types(
    input_types: &[ArrayType],
    input_axes: &[Option<usize>],
    axis_size: usize,
) -> Result<Vec<ArrayType>, TracingError> {
    input_types
        .iter()
        .zip(input_axes.iter())
        .map(|(per_lane_type, axis)| -> Result<ArrayType, TracingError> {
            match axis {
                Some(k) => Ok(per_lane_type.with_inserted_dimension(*k, Size::Static(axis_size))?),
                None => Ok(per_lane_type.clone()),
            }
        })
        .collect()
}

/// Generic value-level batching for a pure elementwise op: extracts per-input metadata,
/// invokes [`lift_elementwise`], and applies the result via [`apply_with_axes`].
pub fn batch_elementwise<V, O>(operation: &O, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError>
where
    V: Traceable<ArrayType>,
    O: Clone + Operation<ArrayType> + InterpretableOperation<ArrayType, V>,
{
    let (per_lane_types, input_axes, axis_size) = batch_input_metadata(inputs)?;
    let (lifted_op, output_axes) = lift_elementwise(operation, &per_lane_types, &input_axes, axis_size)?;
    apply_with_axes(&lifted_op, inputs, output_axes.as_slice())
}

/// Type-level lift for a pure elementwise op: same as [`lift_elementwise`] but also infers the
/// parent-physical output types via [`Operation::infer_output_types`], returning a complete
/// [`BatchingOutput`].
pub fn lift_elementwise_output<O: Clone + Operation<ArrayType>>(
    operation: &O,
    input_types: &[ArrayType],
    input_axes: &[Option<usize>],
    axis_size: usize,
) -> Result<BatchingOutput<O>, TracingError> {
    let (lifted_op, output_axes) = lift_elementwise(operation, input_types, input_axes, axis_size)?;
    let parent_inputs = parent_physical_input_types(input_types, input_axes, axis_size)?;
    let output_types = lifted_op.infer_output_types(parent_inputs.as_slice())?;
    Ok(BatchingOutput::new(lifted_op, output_types, output_axes))
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
    let output_count = operation.infer_output_types(parent_physical_input_types.as_slice())?.len();
    Ok((operation.clone(), vec![common_axis; output_count]))
}


impl<V, Extension> BatchableOperation<V> for ArrayOperation<V, ArrayType, Extension>
where
    V: Value<ArrayType>
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Div<Output = V>
        + Neg<Output = V>
        + Scale<Output = V>
        + Sin
        + Cos
        + Zero<ArrayType>
        + One<ArrayType>
        + ZeroLike
        + OneLike
        + crate::tracing_v2::operations::matrix::DotOps
        + ReshapeOps
        + crate::tracing_v2::operations::select::Select
        + ControlFlowValue,
    Extension: Clone + BatchableOperation<V> + InterpretableOperation<ArrayType, V>,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: Debug + PartialEq>,
{
    fn batch(&self, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError> {
        let missing = |kind: &str| -> TracingError {
            BatchingError::MissingBatchingRule { operation: format!("ArrayOperation::{kind} (no batching rule yet)") }
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
            Self::Condition(condition) => condition.batch(inputs),
            Self::While(while_op) => while_op.batch(inputs),
            Self::Extension(extension) => extension.batch(inputs),
            Self::Zero(_) => Err(missing("Zero")),
            Self::One(_) => Err(missing("One")),
        }
    }

    fn lift(
        &self,
        input_types: &[ArrayType],
        input_axes: &[Option<usize>],
        axis_size: usize,
    ) -> Result<BatchingOutput<Self>, TracingError> {
        let missing = |kind: &str| -> TracingError {
            BatchingError::MissingBatchingRule { operation: format!("ArrayOperation::{kind} (no batching rule yet)") }
                .into()
        };
        match self {
            Self::Add => {
                let output = <crate::operations::arithmetic::AddOperation as BatchableOperation<V>>::lift(
                    &crate::operations::arithmetic::AddOperation,
                    input_types,
                    input_axes,
                    axis_size,
                )?;
                Ok(BatchingOutput::new(Self::Add, output.output_types, output.output_axes))
            }
            Self::Sub => {
                let output = <crate::operations::arithmetic::SubOperation as BatchableOperation<V>>::lift(
                    &crate::operations::arithmetic::SubOperation,
                    input_types,
                    input_axes,
                    axis_size,
                )?;
                Ok(BatchingOutput::new(Self::Sub, output.output_types, output.output_axes))
            }
            Self::Mul => {
                let output = <crate::operations::arithmetic::MulOperation as BatchableOperation<V>>::lift(
                    &crate::operations::arithmetic::MulOperation,
                    input_types,
                    input_axes,
                    axis_size,
                )?;
                Ok(BatchingOutput::new(Self::Mul, output.output_types, output.output_axes))
            }
            Self::Div => {
                let output = <crate::operations::arithmetic::DivOperation as BatchableOperation<V>>::lift(
                    &crate::operations::arithmetic::DivOperation,
                    input_types,
                    input_axes,
                    axis_size,
                )?;
                Ok(BatchingOutput::new(Self::Div, output.output_types, output.output_axes))
            }
            Self::Neg => {
                let output = <crate::operations::arithmetic::NegOperation as BatchableOperation<V>>::lift(
                    &crate::operations::arithmetic::NegOperation,
                    input_types,
                    input_axes,
                    axis_size,
                )?;
                Ok(BatchingOutput::new(Self::Neg, output.output_types, output.output_axes))
            }
            Self::Sin => {
                let output = <crate::operations::trigonometric::SinOperation as BatchableOperation<V>>::lift(
                    &crate::operations::trigonometric::SinOperation,
                    input_types,
                    input_axes,
                    axis_size,
                )?;
                Ok(BatchingOutput::new(Self::Sin, output.output_types, output.output_axes))
            }
            Self::Cos => {
                let output = <crate::operations::trigonometric::CosOperation as BatchableOperation<V>>::lift(
                    &crate::operations::trigonometric::CosOperation,
                    input_types,
                    input_axes,
                    axis_size,
                )?;
                Ok(BatchingOutput::new(Self::Cos, output.output_types, output.output_axes))
            }
            Self::Select => {
                let output = <crate::tracing_v2::operations::select::SelectOperation as BatchableOperation<V>>::lift(
                    &crate::tracing_v2::operations::select::SelectOperation,
                    input_types,
                    input_axes,
                    axis_size,
                )?;
                Ok(BatchingOutput::new(Self::Select, output.output_types, output.output_axes))
            }
            Self::ZeroLike => {
                let output = <crate::operations::constants::ZeroLikeOperation as BatchableOperation<V>>::lift(
                    &crate::operations::constants::ZeroLikeOperation,
                    input_types,
                    input_axes,
                    axis_size,
                )?;
                Ok(BatchingOutput::new(Self::ZeroLike, output.output_types, output.output_axes))
            }
            Self::OneLike => {
                let output = <crate::operations::constants::OneLikeOperation as BatchableOperation<V>>::lift(
                    &crate::operations::constants::OneLikeOperation,
                    input_types,
                    input_axes,
                    axis_size,
                )?;
                Ok(BatchingOutput::new(Self::OneLike, output.output_types, output.output_axes))
            }
            Self::Scale { factor } => {
                let scale_op = crate::operations::arithmetic::ScaleOperation::new(factor.clone());
                let output = <crate::operations::arithmetic::ScaleOperation<ArrayType, V> as BatchableOperation<V>>::lift(
                    &scale_op,
                    input_types,
                    input_axes,
                    axis_size,
                )?;
                Ok(BatchingOutput::new(Self::Scale { factor: output.operation.factor().clone() }, output.output_types, output.output_axes))
            }
            Self::Dot { dimensions } => {
                let dot_op = crate::tracing_v2::operations::dot::DotOperation::new(dimensions.clone());
                let output = <crate::tracing_v2::operations::dot::DotOperation as BatchableOperation<V>>::lift(
                    &dot_op,
                    input_types,
                    input_axes,
                    axis_size,
                )?;
                Ok(BatchingOutput::new(
                    Self::Dot { dimensions: output.operation.dimensions().clone() },
                    output.output_types,
                    output.output_axes,
                ))
            }
            Self::Transpose { permutation } => {
                let transpose_op =
                    crate::tracing_v2::operations::transpose::TransposeOperation::new(permutation.clone());
                let output =
                    <crate::tracing_v2::operations::transpose::TransposeOperation as BatchableOperation<V>>::lift(
                        &transpose_op,
                        input_types,
                        input_axes,
                        axis_size,
                    )?;
                Ok(BatchingOutput::new(
                    Self::Transpose { permutation: output.operation.permutation().to_vec() },
                    output.output_types,
                    output.output_axes,
                ))
            }
            Self::Reshape { input_shape, output_shape } => {
                let reshape_op = crate::tracing_v2::operations::reshape::ReshapeOperation::new(
                    input_shape.clone(),
                    output_shape.clone(),
                );
                let output = <crate::tracing_v2::operations::reshape::ReshapeOperation as BatchableOperation<V>>::lift(
                    &reshape_op,
                    input_types,
                    input_axes,
                    axis_size,
                )?;
                Ok(BatchingOutput::new(
                    Self::Reshape {
                        input_shape: output.operation.input_shape().clone(),
                        output_shape: output.operation.output_shape().clone(),
                    },
                    output.output_types,
                    output.output_axes,
                ))
            }
            Self::Extension(extension) => {
                let BatchingOutput { operation, output_types, output_axes } =
                    extension.lift(input_types, input_axes, axis_size)?;
                Ok(BatchingOutput::new(Self::Extension(operation), output_types, output_axes))
            }
            Self::Condition(_) => Err(missing("Condition (use the value-level batch path)")),
            Self::While(_) => Err(missing("While (use the value-level batch path)")),
            Self::Zero(_) => Err(missing("Zero")),
            Self::One(_) => Err(missing("One")),
        }
    }
}

/// Reconstructs the parent-physical input types from per-lane logical types, input axes, and the
/// lane size. Used by the lift helpers to feed `infer_output_types`.
impl<V, Extension> BatchableOperation<V> for LinearArrayOperation<V, ArrayType, Extension>
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
        + ReshapeOps
        + crate::tracing_v2::operations::select::Select
        + ControlFlowValue,
    Extension: Clone + BatchableOperation<V> + InterpretableOperation<ArrayType, V>,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: Debug + PartialEq>,
{
    fn batch(&self, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError> {
        let missing = |kind: &str| -> TracingError {
            BatchingError::MissingBatchingRule {
                operation: format!("LinearArrayOperation::{kind} (no batching rule yet)"),
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
            Self::Condition(condition) => condition.batch(inputs),
            Self::While(while_op) => while_op.batch(inputs),
            Self::Extension(extension) => extension.batch(inputs),
            Self::Zero(_) => Err(missing("Zero")),
            Self::One(_) => Err(missing("One")),
        }
    }

    fn lift(
        &self,
        input_types: &[ArrayType],
        input_axes: &[Option<usize>],
        axis_size: usize,
    ) -> Result<BatchingOutput<Self>, TracingError> {
        let missing = |kind: &str| -> TracingError {
            BatchingError::MissingBatchingRule {
                operation: format!("LinearArrayOperation::{kind} (no batching rule yet)"),
            }
            .into()
        };
        match self {
            Self::Add => {
                let output = <crate::operations::arithmetic::AddOperation as BatchableOperation<V>>::lift(
                    &crate::operations::arithmetic::AddOperation,
                    input_types,
                    input_axes,
                    axis_size,
                )?;
                Ok(BatchingOutput::new(Self::Add, output.output_types, output.output_axes))
            }
            Self::Sub => {
                let output = <crate::operations::arithmetic::SubOperation as BatchableOperation<V>>::lift(
                    &crate::operations::arithmetic::SubOperation,
                    input_types,
                    input_axes,
                    axis_size,
                )?;
                Ok(BatchingOutput::new(Self::Sub, output.output_types, output.output_axes))
            }
            Self::Neg => {
                let output = <crate::operations::arithmetic::NegOperation as BatchableOperation<V>>::lift(
                    &crate::operations::arithmetic::NegOperation,
                    input_types,
                    input_axes,
                    axis_size,
                )?;
                Ok(BatchingOutput::new(Self::Neg, output.output_types, output.output_axes))
            }
            Self::ZeroLike => {
                let output = <crate::operations::constants::ZeroLikeOperation as BatchableOperation<V>>::lift(
                    &crate::operations::constants::ZeroLikeOperation,
                    input_types,
                    input_axes,
                    axis_size,
                )?;
                Ok(BatchingOutput::new(Self::ZeroLike, output.output_types, output.output_axes))
            }
            Self::OneLike => {
                let output = <crate::operations::constants::OneLikeOperation as BatchableOperation<V>>::lift(
                    &crate::operations::constants::OneLikeOperation,
                    input_types,
                    input_axes,
                    axis_size,
                )?;
                Ok(BatchingOutput::new(Self::OneLike, output.output_types, output.output_axes))
            }
            Self::Scale { factor } => {
                let scale_op = crate::operations::arithmetic::ScaleOperation::new(factor.clone());
                let output = <crate::operations::arithmetic::ScaleOperation<ArrayType, V> as BatchableOperation<V>>::lift(
                    &scale_op,
                    input_types,
                    input_axes,
                    axis_size,
                )?;
                Ok(BatchingOutput::new(
                    Self::Scale { factor: output.operation.factor().clone() },
                    output.output_types,
                    output.output_axes,
                ))
            }
            Self::Transpose { permutation } => {
                let transpose_op =
                    crate::tracing_v2::operations::transpose::TransposeOperation::new(permutation.clone());
                let output =
                    <crate::tracing_v2::operations::transpose::TransposeOperation as BatchableOperation<V>>::lift(
                        &transpose_op,
                        input_types,
                        input_axes,
                        axis_size,
                    )?;
                Ok(BatchingOutput::new(
                    Self::Transpose { permutation: output.operation.permutation().to_vec() },
                    output.output_types,
                    output.output_axes,
                ))
            }
            Self::LeftDot { factor, dimensions } => {
                let left_dot_op =
                    crate::tracing_v2::operations::dot::LeftDotOperation::new(factor.clone(), dimensions.clone());
                let output = <crate::tracing_v2::operations::dot::LeftDotOperation<V> as BatchableOperation<V>>::lift(
                    &left_dot_op,
                    input_types,
                    input_axes,
                    axis_size,
                )?;
                Ok(BatchingOutput::new(
                    Self::LeftDot {
                        factor: output.operation.factor().clone(),
                        dimensions: output.operation.dimensions().clone(),
                    },
                    output.output_types,
                    output.output_axes,
                ))
            }
            Self::RightDot { factor, dimensions } => {
                let right_dot_op =
                    crate::tracing_v2::operations::dot::RightDotOperation::new(factor.clone(), dimensions.clone());
                let output = <crate::tracing_v2::operations::dot::RightDotOperation<V> as BatchableOperation<V>>::lift(
                    &right_dot_op,
                    input_types,
                    input_axes,
                    axis_size,
                )?;
                Ok(BatchingOutput::new(
                    Self::RightDot {
                        factor: output.operation.factor().clone(),
                        dimensions: output.operation.dimensions().clone(),
                    },
                    output.output_types,
                    output.output_axes,
                ))
            }
            Self::Reshape { input_shape, output_shape } => {
                let reshape_op = crate::tracing_v2::operations::reshape::ReshapeOperation::new(
                    input_shape.clone(),
                    output_shape.clone(),
                );
                let output = <crate::tracing_v2::operations::reshape::ReshapeOperation as BatchableOperation<V>>::lift(
                    &reshape_op,
                    input_types,
                    input_axes,
                    axis_size,
                )?;
                Ok(BatchingOutput::new(
                    Self::Reshape {
                        input_shape: output.operation.input_shape().clone(),
                        output_shape: output.operation.output_shape().clone(),
                    },
                    output.output_types,
                    output.output_axes,
                ))
            }
            Self::Extension(extension) => {
                let BatchingOutput { operation, output_types, output_axes } =
                    extension.lift(input_types, input_axes, axis_size)?;
                Ok(BatchingOutput::new(Self::Extension(operation), output_types, output_axes))
            }
            Self::Condition(_) => Err(missing("Condition (use the value-level batch path)")),
            Self::While(_) => Err(missing("While (use the value-level batch path)")),
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
        + ReshapeOps
        + crate::tracing_v2::operations::select::Select
        + ControlFlowValue,
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
        let v_outputs =
            <LinearArrayOperation<V, ArrayType, Extension> as BatchableOperation<V>>::batch(self, materialized.as_slice())?;
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

    fn lift(
        &self,
        input_types: &[ArrayType],
        input_axes: &[Option<usize>],
        axis_size: usize,
    ) -> Result<BatchingOutput<Self>, TracingError> {
        <Self as BatchableOperation<V>>::lift(self, input_types, input_axes, axis_size)
    }
}

/// Wrapping [`TracingDomain`] that lifts a parent domain `Parent` into a tracing context that
/// introduces exactly one batched lane at a chosen axis.
///
/// [`BatchingDomain`] is the carrier for one level of `vmap`: it traces the user's function against
/// logical per-lane [`ArrayType`]s while leaving the runtime value type of the staged program equal
/// to the parent's value type. The actual lifting of staged operations through their
/// [`BatchableOperation`] rules happens after tracing completes, when the resulting program is
/// lowered with [`interpret_batched_program`] or spliced into an outer trace.
///
/// Nested `vmap` composes by repeated wrapping at the type level:
/// `BatchingDomain<'_, BatchingDomain<'_, Parent>>` is a two-level batching context, and the
/// staged program's value type remains `Parent::Value` regardless of the nesting depth. Each level
/// of wrapping owns its own `axis_size` (and optionally `axis_name`) so the inner and outer lanes
/// can be different.
#[derive(Clone, Debug)]
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
}

impl<'parent, Parent: TracingDomain<Type = ArrayType>> BatchingDomain<'parent, Parent> {
    /// Creates a new anonymous [`BatchingDomain`] that wraps `parent` with the supplied lane size.
    #[inline]
    pub fn new(parent: &'parent Parent, axis_size: usize) -> Self {
        Self { parent, axis_size, axis_name: None }
    }

    /// Creates a new [`BatchingDomain`] with a named batched axis.
    #[inline]
    pub fn with_axis_name(parent: &'parent Parent, axis_size: usize, axis_name: impl Into<String>) -> Self {
        Self { parent, axis_size, axis_name: Some(axis_name.into()) }
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
}

impl<'parent, Parent: TracingDomain<Type = ArrayType>> Domain for BatchingDomain<'parent, Parent> {
    type Type = ArrayType;
    type Value = Parent::Value;
}

impl<'parent, Parent: TracingDomain<Type = ArrayType>> TracingDomain for BatchingDomain<'parent, Parent> {
    type OperationCarrier = Parent::OperationCarrier;
}

/// Extension trait that exposes [`vmap`] as a method on any [`TracingDomain`] whose `Type` is
/// [`ArrayType`].
///
/// `domain.vmap(f, input, in_axes, out_axes, axis_size)` is the receiver-style entry point to
/// [`vmap`]; it mirrors how [`jacfwd`](crate::tracing_v2::DifferentiableDomain) /
/// [`jacrev`](crate::tracing_v2::jacrev) sit on their respective domain traits.
pub trait Vmap: TracingDomain<Type = ArrayType> {
    /// Maps a traced function over per-leaf array axes. Equivalent to the free function
    /// [`vmap`](crate::tracing_v2::batching::vmap); see that function for the full bounds and
    /// semantics.
    #[inline]
    #[allow(private_bounds)]
    fn vmap<'domain, F, Input, Output, V>(
        &'domain self,
        function: F,
        input: Input,
        in_axes: Input::To<Option<usize>>,
        out_axes: Output::To<Option<usize>>,
        axis_size: Option<usize>,
    ) -> Result<Output, TracingError>
    where
        Self: TracingDomain<Type = ArrayType, Value = V> + Sized,
        V: Traceable<ArrayType> + crate::tracing_v2::operations::transpose::Transpose + 'domain,
        Input: Parameterized<
                V,
                ParameterStructure: Debug + PartialEq,
                Family: ParameterizedFamily<ArrayType>
                            + ParameterizedFamily<ArrayBatch<V>>
                            + ParameterizedFamily<Option<usize>>
                            + ParameterizedFamily<Tracer<'domain, BatchingDomain<'domain, Self>>>,
            >,
        Output: Parameterized<
                V,
                ParameterStructure: Debug + PartialEq,
                Family: ParameterizedFamily<ArrayType>
                            + ParameterizedFamily<ArrayBatch<V>>
                            + ParameterizedFamily<Option<usize>>
                            + ParameterizedFamily<Tracer<'domain, BatchingDomain<'domain, Self>>>,
            >,
        Input::To<Option<usize>>: Parameterized<Option<usize>, ParameterStructure = Input::ParameterStructure>,
        Output::To<Option<usize>>: Parameterized<Option<usize>, ParameterStructure = Output::ParameterStructure>,
        Input::To<ArrayType>: Parameterized<
                ArrayType,
                To<V> = Input,
                To<Tracer<'domain, BatchingDomain<'domain, Self>>> = Input::To<
                    Tracer<'domain, BatchingDomain<'domain, Self>>,
                >,
            >,
        Output::To<ArrayType>: Parameterized<
                ArrayType,
                To<V> = Output,
                To<Tracer<'domain, BatchingDomain<'domain, Self>>> = Output::To<
                    Tracer<'domain, BatchingDomain<'domain, Self>>,
                >,
            >,
        Output::To<Tracer<'domain, BatchingDomain<'domain, Self>>>: Parameterized<
                Tracer<'domain, BatchingDomain<'domain, Self>>,
                To<ArrayType> = Output::To<ArrayType>,
                To<V> = Output,
            >,
        F: FnOnce(
            Input::To<Tracer<'domain, BatchingDomain<'domain, Self>>>,
        ) -> Result<Output::To<Tracer<'domain, BatchingDomain<'domain, Self>>>, TracingError>,
        Self::OperationCarrier: Clone + BatchableOperation<V>,
    {
        vmap(self, function, input, in_axes, out_axes, axis_size)
    }
}

impl<D: TracingDomain<Type = ArrayType>> Vmap for D {}

/// Splices a program traced through a [`BatchingDomain`] into an outer tracing context, using
/// each instruction's [`BatchableOperation::batch`] rule to determine the lifted operation to
/// stage at the outer level.
///
/// This is the kernel used by nested `vmap` (where the outer context is itself a tracing scope,
/// and the inner program records the per-inner-lane computation). For each inner constant, an
/// outer constant is added to the parent builder. For each inner instruction, the rule's `lift`
/// is consulted to produce the parent-level operation, which is then staged through the outer
/// tracer's normal staging path.
///
/// # Parameters
///
///   - `outer_context`: Tracing context that owns the parent builder receiving the spliced ops.
///   - `inner_program`: Program produced by tracing the inner function through a [`BatchingDomain`].
///   - `inputs`: Outer tracers paired with their batch-axis position (or `None` if lane-uniform)
///     at the outer level. Order matches `inner_program.input_ids`.
///   - `axis_size`: Size of the batched lane the inner trace introduced.
#[allow(private_bounds)]
pub fn splice_batched_program_into_trace<'domain, OuterDomain, V, Input, Output>(
    outer_context: &TracingContext<'domain, OuterDomain>,
    inner_program: &Program<ArrayType, V, OuterDomain::OperationCarrier, Input, Output>,
    inputs: Vec<(Tracer<'domain, OuterDomain>, Option<usize>)>,
    axis_size: usize,
) -> Result<Vec<(Tracer<'domain, OuterDomain>, Option<usize>)>, TracingError>
where
    OuterDomain: TracingDomain<Type = ArrayType, Value = V>,
    V: Traceable<ArrayType> + Clone + 'domain,
    OuterDomain::OperationCarrier: Clone + BatchableOperation<V>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    inner_program.interpret_with(
        inputs,
        |_, constant: &V| -> Result<(Tracer<'domain, OuterDomain>, Option<usize>), TracingError> {
            // Inner constants are lane-uniform at the outer level (they appeared as captured
            // values inside the inner trace and do not vary across the inner batch axis).
            Ok((outer_context.constant(constant.clone()), None))
        },
        |instruction,
         instruction_inputs: &[(Tracer<'domain, OuterDomain>, Option<usize>)]|
         -> Result<Vec<(Tracer<'domain, OuterDomain>, Option<usize>)>, TracingError> {
            let parent_physical_input_types =
                instruction_inputs.iter().map(|(tracer, _)| tracer.r#type().into_owned()).collect::<Vec<_>>();
            let input_axes = instruction_inputs.iter().map(|(_, axis)| *axis).collect::<Vec<_>>();
            let per_lane_input_types = parent_physical_input_types
                .iter()
                .zip(input_axes.iter())
                .map(|(physical_type, axis)| -> Result<ArrayType, TracingError> {
                    match axis {
                        Some(k) => Ok(physical_type.without_dimension(*k)?.0),
                        None => Ok(physical_type.clone()),
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            let lifted =
                instruction.operation().lift(per_lane_input_types.as_slice(), input_axes.as_slice(), axis_size)?;
            let input_tracers: Vec<&Tracer<'domain, OuterDomain>> =
                instruction_inputs.iter().map(|(tracer, _)| tracer).collect();
            let output_tracers = outer_context.stage(lifted.operation().clone(), input_tracers.as_slice())?;
            let output_axes = lifted.output_axes().to_vec();
            check_count!("output", output_tracers, output_axes.len(), TracingError);
            Ok(output_tracers.into_iter().zip(output_axes).collect())
        },
    )
}

/// Interprets a staged program once through packed-array batching rules.
pub fn interpret_batched_program<V, O, Input, Output>(
    program: &Program<ArrayType, V, O, Input, Output>,
    input: Input::To<ArrayBatch<V>>,
) -> Result<Output::To<ArrayBatch<V>>, TracingError>
where
    V: Traceable<ArrayType>,
    O: Clone + BatchableOperation<V>,
    Input: Parameterized<V, ParameterStructure: Debug + PartialEq, Family: ParameterizedFamily<ArrayBatch<V>>>,
    Output: Parameterized<V, Family: ParameterizedFamily<ArrayBatch<V>>>,
{
    let input_structure = input.parameter_structure();
    if &input_structure != program.input_structure() {
        return Err(ParameterError::MismatchedParameterStructures {
            left_structure: format!("{:?}", program.input_structure()),
            right_structure: format!("{input_structure:?}"),
        }
        .into());
    }

    let outputs = program.interpret_with(
        input.into_parameters().collect(),
        |_, constant| Ok(ArrayBatch::unbatched(constant.clone())),
        |instruction, inputs| instruction.operation().batch(inputs),
    )?;
    Ok(Output::To::<ArrayBatch<V>>::from_parameters(program.output_structure().clone(), outputs)?)
}

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
/// `vmap` wraps `domain` in a single-level [`BatchingDomain`], traces `function` against the
/// per-lane logical types, and replays the staged program through primitive batching rules over
/// the original physical values.
#[allow(private_bounds)]
pub fn vmap<'domain, D, F, Input, Output, V>(
    domain: &'domain D,
    function: F,
    input: Input,
    in_axes: Input::To<Option<usize>>,
    out_axes: Output::To<Option<usize>>,
    axis_size: Option<usize>,
) -> Result<Output, TracingError>
where
    D: TracingDomain<Type = ArrayType, Value = V>,
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
    F: FnOnce(
        Input::To<Tracer<'domain, BatchingDomain<'domain, D>>>,
    ) -> Result<Output::To<Tracer<'domain, BatchingDomain<'domain, D>>>, TracingError>,
    D::OperationCarrier: Clone + BatchableOperation<V>,
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
    // SAFETY: `batching_domain` is owned by this function frame and outlives the `trace` call below
    // (which returns before this function returns). The borrow's lifetime is artificially extended
    // to `'domain` only to satisfy the closure's type-level lifetime constraint; no reference
    // produced via this extension escapes the enclosing function.
    let batching_domain_ref: &'domain BatchingDomain<'domain, D> = unsafe {
        std::mem::transmute::<&BatchingDomain<'domain, D>, &'domain BatchingDomain<'domain, D>>(&batching_domain)
    };
    let input_types = Input::To::<ArrayType>::from_parameters(structure.clone(), logical_types)?;
    let (_, program): (Output::To<ArrayType>, Program<ArrayType, V, D::OperationCarrier, Input, Output>) =
        batching_domain_ref.trace(function, input_types)?;
    let batched_input = Input::To::<ArrayBatch<V>>::from_parameters(structure, batched_inputs)?;
    let batched_output = interpret_batched_program(&program, batched_input)?;

    let output_structure = batched_output.parameter_structure();
    let out_axes_structure = out_axes.parameter_structure();
    if out_axes_structure != output_structure {
        return Err(ParameterError::MismatchedParameterStructures {
            left_structure: format!("{output_structure:?}"),
            right_structure: format!("{out_axes_structure:?}"),
        }
        .into());
    }
    let out_axes_values = out_axes.into_parameters().collect::<Vec<_>>();
    let output_values = batched_output
        .into_parameters()
        .zip(out_axes_values.iter().copied())
        .map(|(batch, expected_axis)| -> Result<V, TracingError> {
            let current_axis = batch.batch_axis();
            let value = batch.into_value();
            match (current_axis, expected_axis) {
                (None, None) => Ok(value),
                (None, Some(expected)) => Err(BatchingError::UnbatchedOutput {
                    message: format!("vmap output is lane-uniform but out_axes requested position {expected}"),
                }
                .into()),
                (Some(current), None) => Err(BatchingError::UnbatchedOutput {
                    message: format!(
                        "vmap output is mapped on axis {current} but out_axes requested None \
                        (lane-collapsing reductions are not yet supported)",
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

/// Maps a traced function over array axes selected per leaf from inside an enclosing trace, with
/// the per-output mapped axis placed at the position requested by `out_axes`.
///
/// `vmap_inside` is the trace-level counterpart of [`vmap`]: it accepts outer tracers paired with
/// per-leaf `in_axes`, traces `function` through a fresh inner [`BatchingDomain`] wrapping the
/// outer context's domain, then splices the resulting inner program into the outer trace via
/// [`splice_batched_program_into_trace`]. Each inner instruction is lifted through its
/// [`BatchableOperation::batch`] rule; lane-uniform inputs (with `in_axes[i] == None`) flow
/// through unchanged at the parent level.
///
/// `out_axes` follows the same semantics as in [`vmap`]: `Some(k)` requests output position `k`
/// (a [`TransposeOperation`](crate::tracing_v2::operations::transpose::TransposeOperation) is
/// staged in the outer trace when the natural axis differs), and `None` declares the output as
/// lane-uniform. `axis_size` can pin the lane count when no inputs are mapped.
///
/// # Parameters
///
///   - `function`: Per-lane function operating on tracers at the inner batching level.
///   - `input`: Outer tracers carrying the parent-physical leaves.
///   - `in_axes`: Per-leaf mapped axis or `None` for lane-uniform.
///   - `out_axes`: Per-leaf target mapped axis or `None` for lane-uniform.
///   - `axis_size`: Optional explicit lane count, required when no inputs are mapped.
#[allow(private_bounds)]
pub fn vmap_inside<'domain, OuterDomain, F, Input, Output, V>(
    function: F,
    input: Input,
    in_axes: Input::To<Option<usize>>,
    out_axes: Output::To<Option<usize>>,
    axis_size: Option<usize>,
) -> Result<Output, TracingError>
where
    OuterDomain: TracingDomain<Type = ArrayType, Value = V> + 'domain,
    OuterDomain::OperationCarrier:
        Clone + BatchableOperation<V> + crate::tracing_v2::operations::transpose::SupportsTranspose<ArrayType, V>,
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
    F: FnOnce(
        Input::To<Tracer<'domain, BatchingDomain<'domain, OuterDomain>>>,
    ) -> Result<Output::To<Tracer<'domain, BatchingDomain<'domain, OuterDomain>>>, TracingError>,
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

    let batching_domain = BatchingDomain::new(outer_context.domain(), resolved_axis_size);
    // SAFETY: `batching_domain` is owned by this function frame and outlives the `trace` call
    // below (which returns before this function returns). The borrow's lifetime is artificially
    // extended to `'domain` only to satisfy the inner closure's type-level lifetime constraint;
    // no reference produced via this extension escapes the enclosing function.
    let batching_domain_ref: &'domain BatchingDomain<'domain, OuterDomain> = unsafe {
        std::mem::transmute::<&BatchingDomain<'domain, OuterDomain>, &'domain BatchingDomain<'domain, OuterDomain>>(
            &batching_domain,
        )
    };
    let inner_input_types = Input::To::<ArrayType>::from_parameters(input_structure.clone(), logical_types)?;
    let (_, inner_program): (
        Output::To<ArrayType>,
        Program<ArrayType, V, OuterDomain::OperationCarrier, Input::To<V>, Output::To<V>>,
    ) = batching_domain_ref.trace(function, inner_input_types)?;

    let spliced_outputs =
        splice_batched_program_into_trace(&outer_context, &inner_program, inputs_with_axes, resolved_axis_size)?;

    let output_structure = inner_program.output_structure().clone();
    let out_axes_structure = out_axes.parameter_structure();
    if out_axes_structure != output_structure {
        return Err(ParameterError::MismatchedParameterStructures {
            left_structure: format!("{output_structure:?}"),
            right_structure: format!("{out_axes_structure:?}"),
        }
        .into());
    }
    let out_axes_values = out_axes.into_parameters().collect::<Vec<_>>();
    let outer_output_tracers = spliced_outputs
        .into_iter()
        .zip(out_axes_values.iter().copied())
        .map(|((tracer, axis), expected_axis)| -> Result<Tracer<'domain, OuterDomain>, TracingError> {
            match (axis, expected_axis) {
                (None, None) => Ok(tracer),
                (None, Some(expected)) => Err(BatchingError::UnbatchedOutput {
                    message: format!("vmap_inside output is lane-uniform but out_axes requested position {expected}"),
                }
                .into()),
                (Some(current), None) => Err(BatchingError::UnbatchedOutput {
                    message: format!(
                        "vmap_inside output is mapped on axis {current} but out_axes requested None \
                        (lane-collapsing reductions are not yet supported)",
                    ),
                }
                .into()),
                (Some(current), Some(expected)) if current == expected => Ok(tracer),
                (Some(current), Some(expected)) => {
                    use crate::tracing_v2::operations::transpose::Transpose;
                    let rank = tracer.r#type().as_ref().rank();
                    let permutation = move_axis_permutation(rank, current, expected);
                    Ok(tracer.transpose(permutation))
                }
            }
        })
        .collect::<Result<Vec<_>, TracingError>>()?;

    Ok(Output::from_parameters(output_structure, outer_output_tracers)?)
}

/// Interprets a [`crate::tracing_v2::FlatProgram`] (a `Program` over `Vec<V>` input and output)
/// through batching rules, taking and returning packed [`ArrayBatch`]es. Used by the batching
/// implementations of [`ConditionOperation`] and [`WhileOperation`] to recurse into their nested
/// branch / condition / body programs over the same lane configuration.
pub fn interpret_batched_flat_program<V, O>(
    program: &Program<ArrayType, V, O, Vec<V>, Vec<V>>,
    inputs: Vec<ArrayBatch<V>>,
) -> Result<Vec<ArrayBatch<V>>, TracingError>
where
    V: Traceable<ArrayType>,
    O: Clone + BatchableOperation<V>,
{
    program.interpret_with(
        inputs,
        |_, constant| Ok::<_, TracingError>(ArrayBatch::unbatched(constant.clone())),
        |instruction, instruction_inputs| instruction.operation().batch(instruction_inputs),
    )
}

/// Drives a lane-varying [`ConditionOperation`] by evaluating both branches over the same lane
/// configuration and combining the results via per-lane [`select::Select`](crate::tracing_v2::operations::select::Select).
pub fn batch_condition_with_lane_varying_predicate<V, O>(
    true_branch: &Program<ArrayType, V, O, Vec<V>, Vec<V>>,
    false_branch: &Program<ArrayType, V, O, Vec<V>, Vec<V>>,
    predicate_batch: &ArrayBatch<V>,
    predicate_axis: usize,
    operand_inputs: &[ArrayBatch<V>],
) -> Result<Vec<ArrayBatch<V>>, TracingError>
where
    V: Value<ArrayType> + ControlFlowValue + crate::tracing_v2::operations::select::Select,
    O: Clone + BatchableOperation<V>,
{
    let true_outputs = interpret_batched_flat_program(true_branch, operand_inputs.to_vec())?;
    let false_outputs = interpret_batched_flat_program(false_branch, operand_inputs.to_vec())?;
    check_count!("output", true_outputs, false_outputs.len(), TracingError);
    true_outputs
        .into_iter()
        .zip(false_outputs)
        .map(|(true_output, false_output)| -> Result<ArrayBatch<V>, TracingError> {
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
            let selected =
                V::select(predicate_batch.value().clone(), true_output.value().clone(), false_output.value().clone())?;
            let output_type = selected.r#type().into_owned();
            ArrayBatch::new(output_type, selected, Some(output_axis))
        })
        .collect()
}

/// Tangent-runtime counterpart of [`interpret_batched_flat_program`]: lifts each constant to
/// [`Tangent::Value`] and dispatches per-instruction batching through `BatchableOperation<Tangent<…>>`.
pub fn interpret_batched_flat_program_tangent<V, O>(
    program: &Program<ArrayType, V, O, Vec<V>, Vec<V>>,
    inputs: Vec<ArrayBatch<Tangent<ArrayType, V>>>,
) -> Result<Vec<ArrayBatch<Tangent<ArrayType, V>>>, TracingError>
where
    V: Traceable<ArrayType>,
    O: Clone + BatchableOperation<Tangent<ArrayType, V>>,
{
    program.interpret_with(
        inputs,
        |_, constant| Ok::<_, TracingError>(ArrayBatch::unbatched(Tangent::Value(constant.clone()))),
        |instruction, instruction_inputs| {
            BatchableOperation::<Tangent<ArrayType, V>>::batch(instruction.operation(), instruction_inputs)
        },
    )
}

