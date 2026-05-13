use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Neg, Sub};

use ryft_macros::Parameter;
use thiserror::Error;

use crate::differentiation::Tangent;
use crate::macros::check_count;
use crate::operations::arithmetic::{
    AddOperation, DivOperation, MulOperation, NegOperation, Scale, ScaleOperation, SubOperation,
};
use crate::operations::constants::{One, OneLike, Zero, ZeroLike};
use crate::operations::trigonometric::{Cos, CosOperation, Sin, SinOperation};
use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily};
use crate::tracing::domains::{Domain, Tracer, TracingContext, TracingDomain};
use crate::tracing::{Program, Traceable, TracingError, Value};
use crate::tracing_v2::operations::reshape::ReshapeOps;
use crate::tracing_v2::{
    ArrayOperation, ConditionOperation, ConditionPredicate, ControlFlowError, ControlFlowValue, LinearArrayOperation,
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

/// Packed-array batching rule for one staged operation.
///
/// Implementations receive physical array values paired with mapped-axis metadata
/// ([`ArrayBatch`]) and must return physical array values whose output axis metadata reflects
/// where the mapped lanes end up. The trait is the value-level counterpart of [`BatchingRule`]:
/// `BatchableOperation::batch` runs during single-level `vmap` interpretation when concrete
/// values are available, while `BatchingRule::lift` runs during nested-`vmap` splicing when only
/// types are known.
///
/// # Contract
///
///   - **Axis alignment.** If two or more inputs carry a mapped axis (`batch_axis.is_some()`),
///     they must agree on the axis position. When they disagree, return
///     [`BatchingError::UnsupportedBatchAxisAlignment`] with an error message that names the
///     misaligned axes and suggests the user repositions one of them with `Transpose` (the N-D
///     axis permutation primitive) before invoking the operation. Lane-uniform inputs
///     (`batch_axis.is_none()`) may appear in any combination with mapped inputs and pass through
///     unchanged.
///   - **Axis size.** Every mapped input must have a static dimension at its batch axis with the
///     same size. Use [`common_batch_axis_and_size`] to validate this; surface
///     [`BatchingError::MismatchedBatchSize`] or [`BatchingError::DynamicBatchAxis`] as needed.
///   - **Output axes.** When the operation is elementwise (the typical case), the output mapped
///     axis matches the inputs'. Operations with explicit axis arguments must rewrite those
///     arguments to skip the mapped axis (for example, [`TransposeOperation`]'s permutation is
///     lifted via
///     [`lift_permutation`](crate::tracing_v2::operations::transpose::lift_permutation)).
///   - **Symbolic-zero short-circuit.** When the implementation operates over
///     [`Tangent<ArrayType, V>`](crate::differentiation::Tangent), check whether every input is
///     [`Tangent::Zero`] and return zero outputs without staging the underlying op (see
///     [`batch_linear_with_symbolic_zero`] for the reference pattern). `ZeroLike` and `OneLike`
///     are exceptions because the output's shape depends on the exemplar input's runtime type.
///   - **Missing rule.** Variants that have no defined batching semantics (for example, a
///     while-loop predicate that varies across lanes) must return
///     [`BatchingError::MissingBatchingRule`] with an operation string that is human-readable and
///     points at a likely fix.
///
/// See [`lift_elementwise`] and the per-op helpers in this module
/// ([`batch_dot_operation`], [`batch_transpose_operation`], [`batch_reshape_operation`]) for
/// reference implementations that satisfy this contract.
pub trait BatchableOperation<V: Traceable<ArrayType>>: Operation<ArrayType> {
    /// Applies this operation to packed batched inputs.
    ///
    /// # Parameters
    ///
    ///   - `inputs`: Physical input values and their mapped-axis metadata.
    fn batch(&self, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError>;
}

impl<V: Traceable<ArrayType>> BatchableOperation<V> for NoOperationExtension {
    fn batch(&self, _inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError> {
        match *self {}
    }
}

/// Output of a [`BatchingRule::lift`] call.
#[derive(Clone, Debug)]
pub struct BatchingRuleOutput<O> {
    /// Operation to stage at the parent (outer) level. May differ from the input operation when
    /// axis arguments need to be bumped past the introduced batch dimension.
    pub operation: O,

    /// Parent-physical output types, with the introduced batch dimension included.
    pub output_types: Vec<ArrayType>,

    /// Mapped-axis position within each output's parent-physical type, or `None` if the output
    /// is lane-uniform.
    pub output_axes: Vec<Option<usize>>,
}

/// Type-level batching rule for one staged operation.
///
/// Unlike [`BatchableOperation::batch`] which operates on concrete `ArrayBatch<V>` values, this
/// rule transforms an operation symbolically: given per-input type-and-axis metadata, it returns
/// the lifted operation that should be staged at the parent level along with the per-output
/// type-and-axis metadata. This is the rule used by nested-vmap splicing, which has no concrete
/// values to interpret on.
///
/// Most elementwise operations have a trivial lift (same op, same axis), and can use
/// [`lift_elementwise`] as their implementation body. Operations with explicit axis arguments
/// must provide their own rule that rewrites those arguments to account for the introduced batch
/// dimension.
pub trait BatchingRule: Operation<ArrayType> + Sized {
    /// Lifts this operation through one batching level.
    ///
    /// # Parameters
    ///
    ///   - `input_types`: Per-lane logical types of each input (rank one less than the
    ///     parent-physical type when the corresponding `input_axes` entry is `Some(_)`, and equal
    ///     to the parent-physical type when it is `None`).
    ///   - `input_axes`: Mapped-axis position within each input's parent-physical type, or `None`
    ///     if the input is lane-uniform.
    ///   - `axis_size`: Size of the batched lane this level introduces.
    fn lift(
        &self,
        input_types: &[ArrayType],
        input_axes: &[Option<usize>],
        axis_size: usize,
    ) -> Result<BatchingRuleOutput<Self>, TracingError>;
}

impl BatchingRule for NoOperationExtension {
    fn lift(
        &self,
        _input_types: &[ArrayType],
        _input_axes: &[Option<usize>],
        _axis_size: usize,
    ) -> Result<BatchingRuleOutput<Self>, TracingError> {
        match *self {}
    }
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
) -> Result<BatchingRuleOutput<O>, TracingError> {
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

    let parent_physical_output_types = operation.infer_output_types(parent_physical_input_types.as_slice())?;
    let output_axes = vec![common_axis; parent_physical_output_types.len()];

    Ok(BatchingRuleOutput { operation: operation.clone(), output_types: parent_physical_output_types, output_axes })
}

fn validate_axis_size<V: Typed<ArrayType> + Parameter>(
    expected_axis: &mut Option<usize>,
    expected_size: &mut Option<usize>,
    value: &ArrayBatch<V>,
) -> Result<(), TracingError> {
    let Some(axis) = value.batch_axis else {
        return Ok(());
    };
    match expected_axis {
        Some(existing_axis) if *existing_axis != axis => {
            return Err(BatchingError::UnsupportedBatchAxisAlignment {
                message: format!(
                    "cannot align batch axis {axis} with existing batch axis {existing_axis}: \
                    a Transpose on one of the operands can bring the batch axes into alignment, \
                    or pass matching `in_axes` to the enclosing vmap",
                ),
            }
            .into());
        }
        Some(_) => {}
        None => *expected_axis = Some(axis),
    }

    let size = value.axis_size()?.ok_or(BatchingError::EmptyBatch)?;
    match expected_size {
        Some(existing_size) if *existing_size != size => Err(BatchingError::MismatchedBatchSize.into()),
        Some(_) => Ok(()),
        None => {
            *expected_size = Some(size);
            Ok(())
        }
    }
}

fn common_batch_axis_and_size<V: Typed<ArrayType> + Parameter>(
    inputs: &[ArrayBatch<V>],
) -> Result<(Option<usize>, Option<usize>), TracingError> {
    let mut batch_axis = None;
    let mut axis_size = None;
    for input in inputs {
        validate_axis_size(&mut batch_axis, &mut axis_size, input)?;
    }
    Ok((batch_axis, axis_size))
}

fn validate_output_batch_axis<O: Operation<ArrayType>>(
    operation: &O,
    output_type: &ArrayType,
    batch_axis: Option<usize>,
    axis_size: Option<usize>,
) -> Result<(), TracingError> {
    if let (Some(axis), Some(size)) = (batch_axis, axis_size) {
        if axis >= output_type.rank() {
            return Err(BatchingError::UnsupportedBatchAxisAlignment {
                message: format!("operation '{}' removed batch axis {axis}", operation.name()),
            }
            .into());
        }
        if output_type.dimension(axis as isize) != Size::Static(size) {
            return Err(BatchingError::MismatchedBatchSize.into());
        }
    }
    Ok(())
}

/// Batches a `Dot` instruction by computing the lifted dimension numbers via
/// [`lift_dot_dimensions`](crate::tracing_v2::operations::lift_dot_dimensions) and then
/// invoking the value-level [`Dot`](crate::tracing_v2::operations::dot::Dot) trait on the
/// physical operands with those lifted dimensions.
fn batch_dot_operation<V>(
    dimensions: &crate::tracing_v2::operations::DotDimensionNumbers,
    inputs: &[ArrayBatch<V>],
) -> Result<Vec<ArrayBatch<V>>, TracingError>
where
    V: Traceable<ArrayType> + crate::tracing_v2::operations::dot::Dot,
{
    check_count!("input", inputs, 2, TracingError);
    let axes: Vec<Option<usize>> = inputs.iter().map(|input| input.batch_axis()).collect();
    let mut expected_size: Option<usize> = None;
    for input in inputs.iter() {
        if let Some(size) = input.axis_size()? {
            match expected_size {
                Some(existing) if existing != size => return Err(BatchingError::MismatchedBatchSize.into()),
                Some(_) => {}
                None => expected_size = Some(size),
            }
        }
    }
    let Some((lifted_dimensions, output_axis)) =
        crate::tracing_v2::operations::lift_dot_dimensions(dimensions, axes[0], axes[1])
    else {
        return Err(BatchingError::MissingBatchingRule {
            operation: "Dot with mixed batched/unbatched inputs".to_string(),
        }
        .into());
    };
    let lifted_value = inputs[0].value().clone().dot(inputs[1].value().clone(), &lifted_dimensions);
    let output_type = lifted_value.r#type().into_owned();
    Ok(vec![ArrayBatch::new(output_type, lifted_value, output_axis)?])
}

/// Batches a `LeftDot` instruction by lifting its dimension numbers through
/// [`lift_left_dot_dimensions`](crate::tracing_v2::operations::lift_left_dot_dimensions) and
/// invoking the value-level [`Dot`](crate::tracing_v2::operations::dot::Dot) trait with the
/// captured factor on the LHS.
fn batch_left_dot_operation<V>(
    factor: &V,
    dimensions: &crate::tracing_v2::operations::DotDimensionNumbers,
    inputs: &[ArrayBatch<V>],
) -> Result<Vec<ArrayBatch<V>>, TracingError>
where
    V: Traceable<ArrayType> + crate::tracing_v2::operations::dot::Dot,
{
    check_count!("input", inputs, 1, TracingError);
    let factor_rank = factor.r#type().as_ref().rank();
    let (lifted_dimensions, output_axis) =
        crate::tracing_v2::operations::lift_left_dot_dimensions(dimensions, factor_rank, inputs[0].batch_axis());
    let lifted_value = factor.clone().dot(inputs[0].value().clone(), &lifted_dimensions);
    let output_type = lifted_value.r#type().into_owned();
    Ok(vec![ArrayBatch::new(output_type, lifted_value, output_axis)?])
}

/// Batches a `RightDot` instruction by lifting its dimension numbers through
/// [`lift_right_dot_dimensions`](crate::tracing_v2::operations::lift_right_dot_dimensions) and
/// invoking the value-level [`Dot`](crate::tracing_v2::operations::dot::Dot) trait with the
/// captured factor on the RHS.
fn batch_right_dot_operation<V>(
    factor: &V,
    dimensions: &crate::tracing_v2::operations::DotDimensionNumbers,
    inputs: &[ArrayBatch<V>],
) -> Result<Vec<ArrayBatch<V>>, TracingError>
where
    V: Traceable<ArrayType> + crate::tracing_v2::operations::dot::Dot,
{
    check_count!("input", inputs, 1, TracingError);
    let (lifted_dimensions, output_axis) =
        crate::tracing_v2::operations::lift_right_dot_dimensions(dimensions, inputs[0].batch_axis());
    let lifted_value = inputs[0].value().clone().dot(factor.clone(), &lifted_dimensions);
    let output_type = lifted_value.r#type().into_owned();
    Ok(vec![ArrayBatch::new(output_type, lifted_value, output_axis)?])
}

/// Batches a `Reshape` instruction by lifting its per-lane input and output shapes through
/// [`lift_reshape_shapes`](crate::tracing_v2::operations::lift_reshape_shapes) and applying the
/// lifted output shape to the physical operand via the value-level
/// [`Reshape`](crate::tracing_v2::operations::reshape::Reshape) trait.
fn batch_reshape_operation<V>(
    input_shape: &crate::types::Shape,
    output_shape: &crate::types::Shape,
    inputs: &[ArrayBatch<V>],
) -> Result<Vec<ArrayBatch<V>>, TracingError>
where
    V: Traceable<ArrayType> + ReshapeOps,
{
    check_count!("input", inputs, 1, TracingError);
    let Some(k_in) = inputs[0].batch_axis() else {
        let lifted = inputs[0].value().clone().reshape(output_shape.clone())?;
        let output_type = lifted.r#type().into_owned();
        return Ok(vec![ArrayBatch::new(output_type, lifted, None)?]);
    };
    let axis_size = inputs[0].axis_size()?.ok_or(BatchingError::EmptyBatch)?;
    let Some((_, lifted_output_shape, k_out)) =
        crate::tracing_v2::operations::lift_reshape_shapes(input_shape, output_shape, k_in, axis_size)
    else {
        return Err(BatchingError::MissingBatchingRule {
            operation: format!(
                "Reshape with batch axis {k_in} crossing reshape group boundaries in {input_shape} -> {output_shape}",
            ),
        }
        .into());
    };
    let lifted_value = inputs[0].value().clone().reshape(lifted_output_shape)?;
    let output_type = lifted_value.r#type().into_owned();
    Ok(vec![ArrayBatch::new(output_type, lifted_value, Some(k_out))?])
}

/// Batches a `Transpose` instruction by lifting the permutation through
/// [`lift_permutation`](crate::tracing_v2::operations::lift_permutation) and applying it to the
/// physical operand.
fn batch_transpose_operation<V>(
    permutation: &[usize],
    inputs: &[ArrayBatch<V>],
) -> Result<Vec<ArrayBatch<V>>, TracingError>
where
    V: Traceable<ArrayType> + crate::tracing_v2::operations::transpose::Transpose,
{
    check_count!("input", inputs, 1, TracingError);
    let (lifted_permutation, output_axis) = match inputs[0].batch_axis() {
        Some(batch_axis) => {
            (crate::tracing_v2::operations::lift_permutation(permutation, batch_axis), Some(batch_axis))
        }
        None => (permutation.to_vec(), None),
    };
    let lifted_value = inputs[0].value().clone().transpose(lifted_permutation);
    let output_type = lifted_value.r#type().into_owned();
    Ok(vec![ArrayBatch::new(output_type, lifted_value, output_axis)?])
}

fn batch_by_interpreting_physical_operation<V, O>(
    operation: &O,
    inputs: &[ArrayBatch<V>],
) -> Result<Vec<ArrayBatch<V>>, TracingError>
where
    V: Traceable<ArrayType>,
    O: Operation<ArrayType> + InterpretableOperation<ArrayType, V>,
{
    let (batch_axis, axis_size) = common_batch_axis_and_size(inputs)?;

    let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
    let output_types = operation.infer_output_types(input_types.as_slice())?;
    let input_values = inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
    let output_values = operation.interpret(input_values.as_slice())?;
    check_count!("output", output_values, output_types.len(), TracingError);

    output_types
        .into_iter()
        .zip(output_values)
        .map(|(type_, value)| {
            validate_output_batch_axis(operation, &type_, batch_axis, axis_size)?;
            ArrayBatch::new(type_, value, batch_axis)
        })
        .collect()
}

impl<
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
    Extension: Clone + BatchableOperation<V>,
> BatchableOperation<V> for ArrayOperation<V, ArrayType, Extension>
where
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: Debug + PartialEq>,
{
    fn batch(&self, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError> {
        match self {
            Self::Zero(zero) => batch_by_interpreting_physical_operation(zero, inputs),
            Self::One(one) => batch_by_interpreting_physical_operation(one, inputs),
            Self::Add => batch_by_interpreting_physical_operation(&AddOperation, inputs),
            Self::Sub => batch_by_interpreting_physical_operation(&SubOperation, inputs),
            Self::Mul => batch_by_interpreting_physical_operation(&MulOperation, inputs),
            Self::Div => batch_by_interpreting_physical_operation(&DivOperation, inputs),
            Self::Neg => batch_by_interpreting_physical_operation(&NegOperation, inputs),
            Self::Sin => batch_by_interpreting_physical_operation(&SinOperation, inputs),
            Self::Cos => batch_by_interpreting_physical_operation(&CosOperation, inputs),
            Self::ZeroLike => {
                batch_by_interpreting_physical_operation(&crate::operations::constants::ZeroLikeOperation, inputs)
            }
            Self::OneLike => {
                batch_by_interpreting_physical_operation(&crate::operations::constants::OneLikeOperation, inputs)
            }
            Self::Dot { dimensions } => batch_dot_operation(dimensions, inputs),
            Self::Transpose { permutation } => batch_transpose_operation(permutation, inputs),
            Self::Scale { factor } => {
                batch_by_interpreting_physical_operation(&ScaleOperation::new(factor.clone()), inputs)
            }
            Self::Reshape { input_shape, output_shape } => batch_reshape_operation(input_shape, output_shape, inputs),
            Self::Select => batch_by_interpreting_physical_operation(
                &crate::tracing_v2::operations::select::SelectOperation,
                inputs,
            ),
            Self::Condition(condition) => condition.batch(inputs),
            Self::While(while_op) => while_op.batch(inputs),
            Self::Extension(extension) => extension.batch(inputs),
        }
    }
}

impl<V, Extension> BatchingRule for ArrayOperation<V, ArrayType, Extension>
where
    V: Clone + Debug + PartialEq + Traceable<ArrayType>,
    Extension: Clone + BatchingRule,
{
    fn lift(
        &self,
        input_types: &[ArrayType],
        input_axes: &[Option<usize>],
        axis_size: usize,
    ) -> Result<BatchingRuleOutput<Self>, TracingError> {
        let missing = |kind: &str| -> TracingError {
            BatchingError::MissingBatchingRule {
                operation: format!("ArrayOperation::{kind} (type-level lift not yet implemented)"),
            }
            .into()
        };
        match self {
            // Elementwise variants — same op, common batch axis, parent-physical output types
            // inferred by the op carrier itself.
            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::Div
            | Self::Neg
            | Self::Sin
            | Self::Cos
            | Self::Select
            | Self::ZeroLike
            | Self::OneLike => lift_elementwise(self, input_types, input_axes, axis_size),
            Self::Scale { .. } => lift_elementwise(self, input_types, input_axes, axis_size),
            Self::Extension(extension) => {
                let BatchingRuleOutput { operation, output_types, output_axes } =
                    extension.lift(input_types, input_axes, axis_size)?;
                Ok(BatchingRuleOutput { operation: Self::Extension(operation), output_types, output_axes })
            }
            Self::Dot { dimensions } => {
                check_count!("input", input_types, 2, TracingError);
                if input_axes.len() != 2 {
                    return Err(TracingError::InvalidInputCount { expected: 2, got: input_axes.len() });
                }
                let Some((lifted_dimensions, output_axis)) =
                    crate::tracing_v2::operations::lift_dot_dimensions(dimensions, input_axes[0], input_axes[1])
                else {
                    return Err(missing("Dot with mixed batched/unbatched inputs"));
                };
                let parent_inputs: Vec<ArrayType> = input_types
                    .iter()
                    .zip(input_axes.iter())
                    .map(|(t, ax)| match ax {
                        Some(k) => t.with_inserted_dimension(*k, Size::Static(axis_size)),
                        None => Ok(t.clone()),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let lifted_op = Self::Dot { dimensions: lifted_dimensions };
                let output_types = lifted_op.infer_output_types(parent_inputs.as_slice())?;
                let output_axes = vec![output_axis; output_types.len()];
                Ok(BatchingRuleOutput { operation: lifted_op, output_types, output_axes })
            }
            Self::Transpose { permutation } => {
                check_count!("input", input_types, 1, TracingError);
                if input_axes.len() != 1 {
                    return Err(TracingError::InvalidInputCount { expected: 1, got: input_axes.len() });
                }
                let (lifted_permutation, output_axis) = match input_axes[0] {
                    Some(batch_axis) => {
                        (crate::tracing_v2::operations::lift_permutation(permutation, batch_axis), Some(batch_axis))
                    }
                    None => (permutation.clone(), None),
                };
                let parent_input = match input_axes[0] {
                    Some(k) => input_types[0].with_inserted_dimension(k, Size::Static(axis_size))?,
                    None => input_types[0].clone(),
                };
                let lifted_op = Self::Transpose { permutation: lifted_permutation };
                let output_types = lifted_op.infer_output_types(&[parent_input])?;
                Ok(BatchingRuleOutput { operation: lifted_op, output_types, output_axes: vec![output_axis] })
            }
            Self::Reshape { input_shape, output_shape } => {
                check_count!("input", input_types, 1, TracingError);
                if input_axes.len() != 1 {
                    return Err(TracingError::InvalidInputCount { expected: 1, got: input_axes.len() });
                }
                let Some(k_in) = input_axes[0] else {
                    return lift_elementwise(self, input_types, input_axes, axis_size);
                };
                let Some((lifted_input_shape, lifted_output_shape, k_out)) =
                    crate::tracing_v2::operations::lift_reshape_shapes(input_shape, output_shape, k_in, axis_size)
                else {
                    return Err(missing(&format!(
                        "Reshape with batch axis {k_in} crossing reshape group boundaries in \
                        {input_shape} -> {output_shape}"
                    )));
                };
                let parent_input = input_types[0].with_inserted_dimension(k_in, Size::Static(axis_size))?;
                let lifted_op = Self::Reshape { input_shape: lifted_input_shape, output_shape: lifted_output_shape };
                let output_types = lifted_op.infer_output_types(&[parent_input])?;
                Ok(BatchingRuleOutput { operation: lifted_op, output_types, output_axes: vec![Some(k_out)] })
            }
            // Variants whose lifted forms still need dedicated rules.
            Self::Zero(_) => Err(missing("Zero")),
            Self::One(_) => Err(missing("One")),
            Self::Condition(_) => Err(missing("Condition")),
            Self::While(_) => Err(missing("While")),
        }
    }
}

impl<
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
> BatchableOperation<V> for LinearArrayOperation<V, ArrayType, Extension>
where
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: Debug + PartialEq>,
{
    fn batch(&self, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError> {
        match self {
            Self::Condition(condition) => condition.batch(inputs),
            Self::While(while_op) => while_op.batch(inputs),
            Self::Extension(extension) => extension.batch(inputs),
            Self::Transpose { permutation } => batch_transpose_operation(permutation, inputs),
            Self::LeftDot { factor, dimensions } => batch_left_dot_operation(factor, dimensions, inputs),
            Self::RightDot { factor, dimensions } => batch_right_dot_operation(factor, dimensions, inputs),
            Self::Reshape { input_shape, output_shape } => batch_reshape_operation(input_shape, output_shape, inputs),
            _ => batch_by_interpreting_physical_operation(self, inputs),
        }
    }
}

impl<V, Extension> BatchingRule for LinearArrayOperation<V, ArrayType, Extension>
where
    V: Clone + Debug + PartialEq + Traceable<ArrayType>,
    Extension: Clone + BatchingRule,
{
    fn lift(
        &self,
        input_types: &[ArrayType],
        input_axes: &[Option<usize>],
        axis_size: usize,
    ) -> Result<BatchingRuleOutput<Self>, TracingError> {
        let missing = |kind: &str| -> TracingError {
            BatchingError::MissingBatchingRule {
                operation: format!("LinearArrayOperation::{kind} (type-level lift not yet implemented)"),
            }
            .into()
        };
        match self {
            // Elementwise linear variants: same op, same axis, parent-physical output types
            // inferred by the carrier.
            Self::Add | Self::Sub | Self::Neg | Self::ZeroLike | Self::OneLike => {
                lift_elementwise(self, input_types, input_axes, axis_size)
            }
            Self::Scale { .. } => lift_elementwise(self, input_types, input_axes, axis_size),
            Self::Extension(extension) => {
                let BatchingRuleOutput { operation, output_types, output_axes } =
                    extension.lift(input_types, input_axes, axis_size)?;
                Ok(BatchingRuleOutput { operation: Self::Extension(operation), output_types, output_axes })
            }
            Self::Transpose { permutation } => {
                check_count!("input", input_types, 1, TracingError);
                if input_axes.len() != 1 {
                    return Err(TracingError::InvalidInputCount { expected: 1, got: input_axes.len() });
                }
                let (lifted_permutation, output_axis) = match input_axes[0] {
                    Some(batch_axis) => {
                        (crate::tracing_v2::operations::lift_permutation(permutation, batch_axis), Some(batch_axis))
                    }
                    None => (permutation.clone(), None),
                };
                let parent_input = match input_axes[0] {
                    Some(k) => input_types[0].with_inserted_dimension(k, Size::Static(axis_size))?,
                    None => input_types[0].clone(),
                };
                let lifted_op = Self::Transpose { permutation: lifted_permutation };
                let output_types = lifted_op.infer_output_types(&[parent_input])?;
                Ok(BatchingRuleOutput { operation: lifted_op, output_types, output_axes: vec![output_axis] })
            }
            Self::LeftDot { factor, dimensions } => {
                check_count!("input", input_types, 1, TracingError);
                if input_axes.len() != 1 {
                    return Err(TracingError::InvalidInputCount { expected: 1, got: input_axes.len() });
                }
                let factor_rank = factor.r#type().as_ref().rank();
                let (lifted_dimensions, output_axis) =
                    crate::tracing_v2::operations::lift_left_dot_dimensions(dimensions, factor_rank, input_axes[0]);
                let parent_input = match input_axes[0] {
                    Some(k) => input_types[0].with_inserted_dimension(k, Size::Static(axis_size))?,
                    None => input_types[0].clone(),
                };
                let lifted_op = Self::LeftDot { factor: factor.clone(), dimensions: lifted_dimensions };
                let output_types = lifted_op.infer_output_types(&[parent_input])?;
                Ok(BatchingRuleOutput { operation: lifted_op, output_types, output_axes: vec![output_axis] })
            }
            Self::RightDot { factor, dimensions } => {
                check_count!("input", input_types, 1, TracingError);
                if input_axes.len() != 1 {
                    return Err(TracingError::InvalidInputCount { expected: 1, got: input_axes.len() });
                }
                let (lifted_dimensions, output_axis) =
                    crate::tracing_v2::operations::lift_right_dot_dimensions(dimensions, input_axes[0]);
                let parent_input = match input_axes[0] {
                    Some(k) => input_types[0].with_inserted_dimension(k, Size::Static(axis_size))?,
                    None => input_types[0].clone(),
                };
                let lifted_op = Self::RightDot { factor: factor.clone(), dimensions: lifted_dimensions };
                let output_types = lifted_op.infer_output_types(&[parent_input])?;
                Ok(BatchingRuleOutput { operation: lifted_op, output_types, output_axes: vec![output_axis] })
            }
            Self::Reshape { input_shape, output_shape } => {
                check_count!("input", input_types, 1, TracingError);
                if input_axes.len() != 1 {
                    return Err(TracingError::InvalidInputCount { expected: 1, got: input_axes.len() });
                }
                let Some(k_in) = input_axes[0] else {
                    return lift_elementwise(self, input_types, input_axes, axis_size);
                };
                let Some((lifted_input_shape, lifted_output_shape, k_out)) =
                    crate::tracing_v2::operations::lift_reshape_shapes(input_shape, output_shape, k_in, axis_size)
                else {
                    return Err(missing(&format!(
                        "Reshape with batch axis {k_in} crossing reshape group boundaries in \
                        {input_shape} -> {output_shape}"
                    )));
                };
                let parent_input = input_types[0].with_inserted_dimension(k_in, Size::Static(axis_size))?;
                let lifted_op = Self::Reshape { input_shape: lifted_input_shape, output_shape: lifted_output_shape };
                let output_types = lifted_op.infer_output_types(&[parent_input])?;
                Ok(BatchingRuleOutput { operation: lifted_op, output_types, output_axes: vec![Some(k_out)] })
            }
            // Variants still needing dedicated rules.
            Self::Zero(_) => Err(missing("Zero")),
            Self::One(_) => Err(missing("One")),
            Self::Condition(_) => Err(missing("Condition")),
            Self::While(_) => Err(missing("While")),
        }
    }
}

/// Symbolic-zero-aware batched interpretation of a [`LinearArrayOperation`].
///
/// When every input batch is structurally zero ([`Tangent::Zero`]), the operation produces
/// structurally zero outputs whose [`ArrayType`]s are derived from the operation's
/// [`Operation::infer_output_types`] without touching the leaf type's arithmetic. Otherwise,
/// each lane is materialized via [`Zero::zero`] for [`Tangent::Zero`] inputs and forwarded to
/// the existing [`BatchableOperation`] implementation over `V`.
///
/// `ZeroLike` and `OneLike` always need their exemplar input materialized to derive the output
/// value, so the short-circuit does not apply to those two variants.
fn batch_linear_with_symbolic_zero<V, Extension>(
    operation: &LinearArrayOperation<V, ArrayType, Extension>,
    inputs: &[ArrayBatch<Tangent<ArrayType, V>>],
) -> Result<Vec<ArrayBatch<Tangent<ArrayType, V>>>, TracingError>
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
    let (batch_axis, axis_size) = common_batch_axis_and_size(inputs)?;
    let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
    let output_types = operation.infer_output_types(input_types.as_slice())?;

    let always_materialize = matches!(operation, LinearArrayOperation::ZeroLike | LinearArrayOperation::OneLike);
    if !always_materialize && inputs.iter().all(|input| input.value().is_zero()) {
        return output_types
            .into_iter()
            .map(|output_type| {
                validate_output_batch_axis(operation, &output_type, batch_axis, axis_size)?;
                let value = Tangent::zero(output_type.clone());
                ArrayBatch::new(output_type, value, batch_axis)
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

    let v_outputs = BatchableOperation::<V>::batch(operation, materialized.as_slice())?;

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

impl<
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
    Extension: Clone + BatchableOperation<V> + BatchableOperation<Tangent<ArrayType, V>> + InterpretableOperation<ArrayType, V>,
> BatchableOperation<Tangent<ArrayType, V>> for LinearArrayOperation<V, ArrayType, Extension>
where
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: Debug + PartialEq>,
{
    fn batch(
        &self,
        inputs: &[ArrayBatch<Tangent<ArrayType, V>>],
    ) -> Result<Vec<ArrayBatch<Tangent<ArrayType, V>>>, TracingError> {
        match self {
            Self::Condition(condition) => condition.batch(inputs),
            Self::While(while_op) => while_op.batch(inputs),
            Self::Extension(extension) => extension.batch(inputs),
            _ => batch_linear_with_symbolic_zero(self, inputs),
        }
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
    pub parent: &'parent Parent,

    /// Size of the batched lane this level introduces.
    pub axis_size: usize,

    /// Optional human-readable name for this batched axis. When supplied, future collective
    /// operations such as `psum`/`pmean`/`all_gather` will be able to address this axis by name
    /// from inside the batched function body. Today the name is metadata-only; collectives are a
    /// future extension.
    pub axis_name: Option<String>,
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
/// each instruction's [`BatchingRule`] to determine the lifted operation to stage at the outer
/// level.
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
    OuterDomain::OperationCarrier: Clone + BatchingRule,
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
            let BatchingRuleOutput { operation, output_types: _, output_axes } =
                instruction.operation.lift(per_lane_input_types.as_slice(), input_axes.as_slice(), axis_size)?;
            let input_tracers: Vec<&Tracer<'domain, OuterDomain>> =
                instruction_inputs.iter().map(|(tracer, _)| tracer).collect();
            let output_tracers = outer_context.stage(operation, input_tracers.as_slice())?;
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
    if input_structure != program.input_structure {
        return Err(ParameterError::MismatchedParameterStructures {
            left_structure: format!("{:?}", program.input_structure),
            right_structure: format!("{input_structure:?}"),
        }
        .into());
    }

    let outputs = program.interpret_with(
        input.into_parameters().collect(),
        |_, constant| Ok(ArrayBatch::unbatched(constant.clone())),
        |instruction, inputs| instruction.operation.batch(inputs),
    )?;
    Ok(Output::To::<ArrayBatch<V>>::from_parameters(program.output_structure.clone(), outputs)?)
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
/// [`BatchingRule`]; lane-uniform inputs (with `in_axes[i] == None`) flow through unchanged at
/// the parent level.
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
        Clone + BatchingRule + crate::tracing_v2::operations::transpose::SupportsTranspose<ArrayType, V>,
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
    let outer_context = input_tracers.first().map(|tracer| tracer.context.clone()).ok_or(BatchingError::EmptyBatch)?;

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

    let batching_domain = BatchingDomain::new(outer_context.domain, resolved_axis_size);
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

    let output_structure = inner_program.output_structure.clone();
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
fn interpret_batched_flat_program<V, O>(
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
        |instruction, instruction_inputs| instruction.operation.batch(instruction_inputs),
    )
}

impl<V, O> BatchableOperation<V> for ConditionOperation<V, O, ArrayType>
where
    Self: Operation<ArrayType>,
    V: Value<ArrayType> + ControlFlowValue + crate::tracing_v2::operations::select::Select,
    O: Clone + BatchableOperation<V>,
{
    fn batch(&self, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError> {
        match &self.predicate {
            ConditionPredicate::Captured(predicate) => {
                let branch = if *predicate { &self.true_branch } else { &self.false_branch };
                interpret_batched_flat_program(branch, inputs.to_vec())
            }
            ConditionPredicate::RuntimeInput(_) => {
                let Some((predicate_batch, operand_inputs)) = inputs.split_first() else {
                    return Err(BatchingError::MissingBatchingRule {
                        operation: "condition with no predicate input".to_string(),
                    }
                    .into());
                };
                match predicate_batch.batch_axis() {
                    None => {
                        let predicate = predicate_batch.value().control_flow_predicate()?;
                        let branch = if predicate { &self.true_branch } else { &self.false_branch };
                        interpret_batched_flat_program(branch, operand_inputs.to_vec())
                    }
                    Some(predicate_axis) => batch_condition_with_lane_varying_predicate(
                        &self.true_branch,
                        &self.false_branch,
                        predicate_batch,
                        predicate_axis,
                        operand_inputs,
                    ),
                }
            }
        }
    }
}

/// Drives a lane-varying [`ConditionOperation`] by evaluating both branches over the same lane
/// configuration and combining the results via per-lane [`Select`].
fn batch_condition_with_lane_varying_predicate<V, O>(
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

impl<V, O> BatchableOperation<V> for WhileOperation<V, O, ArrayType>
where
    Self: Operation<ArrayType>,
    V: Value<ArrayType> + ControlFlowValue,
    O: Clone + BatchableOperation<V>,
{
    fn batch(&self, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, TracingError> {
        let mut state = inputs.to_vec();
        loop {
            let condition_outputs = interpret_batched_flat_program(&self.condition, state.clone())?;
            check_count!("output", condition_outputs, 1, TracingError);
            let predicate_batch = &condition_outputs[0];
            if predicate_batch.batch_axis().is_some() {
                return Err(BatchingError::MissingBatchingRule {
                    operation: "while with lane-varying loop predicate".to_string(),
                }
                .into());
            }
            if !predicate_batch.value().control_flow_predicate()? {
                return Ok(state);
            }
            state = interpret_batched_flat_program(&self.body, state)?;
        }
    }
}

impl<V, O> BatchingRule for ConditionOperation<V, O, ArrayType>
where
    Self: Operation<ArrayType>,
    V: Clone + Debug + PartialEq + Traceable<ArrayType>,
    O: Clone,
{
    fn lift(
        &self,
        _input_types: &[ArrayType],
        _input_axes: &[Option<usize>],
        _axis_size: usize,
    ) -> Result<BatchingRuleOutput<Self>, TracingError> {
        Err(BatchingError::MissingBatchingRule {
            operation: "ConditionOperation (type-level lift not yet implemented)".to_string(),
        }
        .into())
    }
}

impl<V, O> BatchingRule for WhileOperation<V, O, ArrayType>
where
    Self: Operation<ArrayType>,
    V: Clone + Debug + PartialEq + Traceable<ArrayType>,
    O: Clone,
{
    fn lift(
        &self,
        _input_types: &[ArrayType],
        _input_axes: &[Option<usize>],
        _axis_size: usize,
    ) -> Result<BatchingRuleOutput<Self>, TracingError> {
        Err(BatchingError::MissingBatchingRule {
            operation: "WhileOperation (type-level lift not yet implemented)".to_string(),
        }
        .into())
    }
}

/// Tangent-runtime counterpart of [`interpret_batched_flat_program`]: lifts each constant to
/// [`Tangent::Value`] and dispatches per-instruction batching through `BatchableOperation<Tangent<…>>`.
fn interpret_batched_flat_program_tangent<V, O>(
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
            BatchableOperation::<Tangent<ArrayType, V>>::batch(&instruction.operation, instruction_inputs)
        },
    )
}

impl<V, O> BatchableOperation<Tangent<ArrayType, V>> for ConditionOperation<V, O, ArrayType>
where
    Self: Operation<ArrayType>,
    V: Value<ArrayType> + ControlFlowValue,
    O: Clone + BatchableOperation<Tangent<ArrayType, V>>,
{
    fn batch(
        &self,
        inputs: &[ArrayBatch<Tangent<ArrayType, V>>],
    ) -> Result<Vec<ArrayBatch<Tangent<ArrayType, V>>>, TracingError> {
        let predicate = match &self.predicate {
            ConditionPredicate::Captured(predicate) => *predicate,
            ConditionPredicate::RuntimeInput(_) => {
                return Err(BatchingError::MissingBatchingRule {
                    operation: "condition with runtime predicate over tangent runtime values".to_string(),
                }
                .into());
            }
        };
        let branch = if predicate { &self.true_branch } else { &self.false_branch };
        interpret_batched_flat_program_tangent(branch, inputs.to_vec())
    }
}

impl<V, O> BatchableOperation<Tangent<ArrayType, V>> for WhileOperation<V, O, ArrayType>
where
    Self: Operation<ArrayType>,
    V: Value<ArrayType> + ControlFlowValue,
    O: Clone + BatchableOperation<Tangent<ArrayType, V>>,
{
    fn batch(
        &self,
        _inputs: &[ArrayBatch<Tangent<ArrayType, V>>],
    ) -> Result<Vec<ArrayBatch<Tangent<ArrayType, V>>>, TracingError> {
        // While loops over tangent runtime values cannot make a loop-control decision: the
        // condition program requires a primal value to derive its scalar boolean output, and
        // a `Tangent` lane carries only zero/value tangent metadata, not a primal predicate.
        // Pushforward/pullback programs do not emit `While` today (the JVP rule unrolls loops at
        // trace time and `WhileOperation::transpose` errors), so this path is unreachable from
        // `jacfwd` / `jacrev`; callers manually constructing a tangent `While` get a clear error.
        Err(BatchingError::MissingBatchingRule { operation: "while over tangent runtime values".to_string() }.into())
    }
}
