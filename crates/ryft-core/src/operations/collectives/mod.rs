//! Contains the named-axis collective operations, which exchange or reduce values across a named axis, together with
//! their interpretation, partial-evaluation, batching, forward-mode differentiation, and transposition rules. These
//! are the analogues of [JAX's parallel operators](https://docs.jax.dev/en/latest/jax.lax.html#parallel-operators).
//!
//! This module owns the vocabulary that every collective shares — [`CollectiveMode`], [`CollectiveOptions`], named
//! axis resolution, and the shape-changing collective engine with its two operation-generating macros — while each
//! operation family lives in its own submodule: [`parallel_reduce`], [`all_gather`], [`parallel_sum_scatter`],
//! [`parallel_permute`], and [`all_to_all`].
//!
//! Collectives reference an enclosing named-axis binder by name, validated against the active
//! [`NamedAxes`] environment at staging time. A name bound by an enclosing `batch` level is
//! resolved at trace time by the operations' batching rules, which collapse or materialize the mapped batch axis at
//! the binding level, while a name bound to a device mesh axis by a `shard_map` manual region stays in the staged
//! body and lowers to cross-device collectives over that mesh axis.

// TODO(eaplatanios): Review this module.

use std::fmt::Debug;

use crate::arrays::batching::{DynamicArrayBatchingPolicy, broadcast_array};
use crate::arrays::{
    ArrayBatch, ArrayBatching, ArrayBatchingPolicy, ArrayIrBatch, ArrayIrBatching, ArrayIrType, ArrayType, Dimension,
    DimensionType, DimensionValue, LinearResiduals, Shape, Sharding, StaticArrayBatchingPolicy,
};
use crate::axes::{AxisError, NamedAxes, NamedAxis};
use crate::batching::{BatchAxis, BatchingContext, BatchingError};
use crate::contexts::{Context, Domain, ProjectedContext};
use crate::differentiation::{DifferentiableType, DifferentiationDual, DifferentiationError, LinearCallOperation};
use crate::macros::check_count;
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::dimensions::dimension_requirement::DimensionRequirement;
use crate::operations::dimensions::dimension_size::{DimensionSize, DimensionSizeOperation};
use crate::operations::manipulation::broadcasting::{Broadcast, DynamicBroadcastOperation};
use crate::operations::manipulation::reshaping::{DynamicReshapeOperation, Reshape, ReshapeParameters};
use crate::operations::manipulation::slicing::resized_output_sharding;
use crate::operations::manipulation::transposition::Transpose;
use crate::operations::math::div::Div;
use crate::operations::math::mul::Mul;
use crate::partial::PartialValue;
use crate::programs::{
    MaybeZero, Operation, OperationProjection, ProgramError, TypeError, Typed, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

pub mod all_gather;
pub mod all_to_all;
pub mod parallel_permute;
pub mod parallel_reduce;
pub mod parallel_sum_scatter;

pub use all_gather::{ALL_GATHER_OPERATION_NAME, AllGather, AllGatherOperation, AllGatherOutputVariance};
pub use all_to_all::{ALL_TO_ALL_OPERATION_NAME, AllToAll, AllToAllOperation, ParallelSwapAxes};
pub use parallel_permute::{
    PARALLEL_PERMUTE_OPERATION_NAME, ParallelPermute, ParallelPermuteOperation, ParallelShuffle,
};
pub use parallel_reduce::{ParallelReduce, ParallelReduceOperation, ParallelReductionKind};
pub use parallel_sum_scatter::{PARALLEL_SUM_SCATTER_OPERATION_NAME, ParallelSumScatter, ParallelSumScatterOperation};

/// Shape semantics used by collectives that can either materialize a named axis or tile an existing array axis.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum CollectiveMode {
    /// Materializes the named axis as a new ranked array dimension, or consumes one ranked dimension when scattering.
    #[default]
    Untiled,

    /// Preserves array rank by multiplying or dividing an existing ranked array dimension.
    Tiled,
}

/// Shared shape and grouping options for all-gather, sum-scatter, and all-to-all.
#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub struct CollectiveOptions {
    /// Rank-changing or rank-preserving shape semantics.
    mode: CollectiveMode,

    /// Optional ordered partition of logical participant indices.
    axis_index_groups: Option<Vec<Vec<usize>>>,
}

impl Debug for CollectiveOptions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.axis_index_groups {
            None => Debug::fmt(&self.mode, formatter),
            Some(axis_index_groups) => formatter
                .debug_struct("CollectiveOptions")
                .field("mode", &self.mode)
                .field("axis_index_groups", axis_index_groups)
                .finish(),
        }
    }
}

impl CollectiveOptions {
    /// Creates collective options for `mode` with no participant subgroups.
    #[inline]
    pub fn new(mode: CollectiveMode) -> Self {
        Self { mode, axis_index_groups: None }
    }

    /// Creates rank-preserving tiled collective options with no participant subgroups.
    #[inline]
    pub fn tiled() -> Self {
        Self::new(CollectiveMode::Tiled)
    }

    /// Returns these options with the provided ordered participant groups.
    #[inline]
    pub fn with_axis_index_groups(mut self, axis_index_groups: Vec<Vec<usize>>) -> Self {
        self.axis_index_groups = Some(axis_index_groups);
        self
    }

    /// Returns the selected shape mode.
    #[inline]
    pub fn mode(&self) -> CollectiveMode {
        self.mode
    }

    /// Returns the ordered participant groups, if any.
    #[inline]
    pub fn axis_index_groups(&self) -> Option<&[Vec<usize>]> {
        self.axis_index_groups.as_deref()
    }

    /// Validates these options against the full named-axis size and returns the effective group size used for shape
    /// arithmetic.
    pub(super) fn effective_axis_size(&self, operation_name: &str, axis_size: usize) -> Result<usize, TypeError> {
        effective_collective_axis_size(operation_name, axis_size, self.axis_index_groups())
    }
}

/// Validates an optional ordered participant partition and returns its effective group size without copying it.
pub(super) fn effective_collective_axis_size(
    operation_name: &str,
    axis_size: usize,
    groups: Option<&[Vec<usize>]>,
) -> Result<usize, TypeError> {
    validate_collective_axis_size(operation_name, axis_size)?;
    let Some(groups) = groups else {
        return Ok(axis_size);
    };
    let Some(first_group) = groups.first() else {
        return Err(TypeError::invalid(format!("`{operation_name}` axis index groups must not be empty")));
    };
    if first_group.is_empty() {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` axis index groups must contain at least one participant",
        )));
    }
    let group_size = first_group.len();
    let mut seen = vec![false; axis_size];
    for (group_index, group) in groups.iter().enumerate() {
        if group.len() != group_size {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` axis index group {group_index} has size {} but every group must have size \
                     {group_size}",
                group.len(),
            )));
        }
        for &participant in group {
            let Some(participant_seen) = seen.get_mut(participant) else {
                return Err(TypeError::invalid(format!(
                    "`{operation_name}` axis index {participant} is out of bounds for axis size {axis_size}",
                )));
            };
            if *participant_seen {
                return Err(TypeError::invalid(format!(
                    "`{operation_name}` axis index groups contain participant {participant} more than once",
                )));
            }
            *participant_seen = true;
        }
    }
    if let Some(missing) = seen.iter().position(|seen| !seen) {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` axis index groups do not contain participant {missing}",
        )));
    }
    Ok(group_size)
}

/// Rejects ragged collective operands before any parent binding can stage or execute collective work.
pub(super) fn reject_ragged_collective_inputs<V: Value<Type = ArrayType>>(
    operation_name: &str,
    inputs: &[ArrayBatch<V>],
) -> Result<(), BatchingError> {
    if let Some((index, ragged_axis)) = inputs
        .iter()
        .enumerate()
        .find_map(|(index, input)| input.ragged_axes().first().map(|ragged_axis| (index, ragged_axis)))
    {
        return Err(BatchingError::UnsupportedOperation {
            message: format!(
                "`{}` does not support bounded ragged dimension `{}` on operand {}",
                operation_name,
                ragged_axis.dimension(),
                index,
            ),
        });
    }
    Ok(())
}

/// Re-stages a collective that targets a different (outer) named axis into the batching context's parent.
///
/// Under nested `batch` levels, a collective is consumed by the level whose
/// [`axis_name`](crate::batching::BatchingContext::axis_name) matches its axis name and must pass through
/// every inner level untouched: each inner batch item participates in the outer collective independently, so the
/// operands' mapped axes are preserved as-is on the forwarded outputs. The parent may itself be another
/// [`BatchingContext`] — whose own rule dispatch repeats this name
/// resolution at the next level — or an ordinary tracing context. Batching rules for custom collective-like
/// operations should use this helper for their "not my axis" arm.
pub fn forward_collective_to_parent<C, P: ArrayBatchingPolicy<C>>(
    context: &BatchingContext<C, ArrayBatching<P>>,
    parent_operation: C::Operation,
    inputs: &[ArrayBatch<<C as Domain>::Value>],
) -> Result<Vec<ArrayBatch<<C as Domain>::Value>>, BatchingError>
where
    C: Context<Type = ArrayType>,
{
    reject_ragged_collective_inputs(parent_operation.name(), inputs)?;
    let parent_input_values: Vec<<C as Domain>::Value> = inputs.iter().map(|batch| batch.value().clone()).collect();
    let parent_outputs = context.parent().bind(parent_operation, Vec::new(), &parent_input_values)?;
    check_count!("output", parent_outputs, inputs.len(), ProgramError);
    parent_outputs
        .into_iter()
        .zip(inputs.iter())
        .map(|(parent_value, input_batch)| ArrayBatch::new(parent_value, input_batch.batch_axis()))
        .collect()
}

/// Resolves the size of the named axis bound by the active [`NamedAxes`] environment, failing fast with
/// [`AxisError::UnboundAxisName`] when no enclosing binder binds `axis_name`. The shape-changing collective
/// capabilities bake the resolved size into their operation payloads at staging time, because their output shapes
/// depend on it while [`Operation::infer_output_types`] only sees input types.
pub(super) fn resolve_named_axis_size<C: NamedAxes>(context: &C, axis_name: &str) -> Result<usize, ProgramError> {
    match context.named_axis(axis_name) {
        Some(NamedAxis::Batched { size: Some(size) } | NamedAxis::Mesh { size, .. }) if size > 0 => Ok(size),
        Some(NamedAxis::Batched { size: Some(_) } | NamedAxis::Mesh { .. }) => {
            Err(TypeError::invalid(format!("collective axis `{axis_name}` must contain at least one participant",))
                .into())
        }
        Some(NamedAxis::Batched { size: None }) => Err(BatchingError::UnsupportedOperation {
            message: format!(
                "collective axis `{axis_name}` has a dynamic extent that must remain a first-class operand"
            ),
        }
        .into()),
        None => Err(BatchingError::Axis(AxisError::UnboundAxisName { name: axis_name.to_string() }).into()),
    }
}

/// Rejects an invalid zero-participant collective before any multiplication, division, or remainder operation.
pub(crate) fn validate_collective_axis_size(operation_name: &str, axis_size: usize) -> Result<(), TypeError> {
    if axis_size == 0 {
        Err(TypeError::invalid(format!("`{operation_name}` axis size must be greater than zero")))
    } else {
        Ok(())
    }
}

/// Validates the shared operand contract of the shape-changing collectives (exactly one operand with no unreduced
/// axes) and returns the operand's static dimensions.
pub(super) fn shape_changing_collective_dimensions(
    operation_name: &str,
    input_types: &[ArrayType],
) -> Result<Vec<usize>, TypeError> {
    check_count!("input", input_types, 1, TypeError);
    if operation_name != PARALLEL_SUM_SCATTER_OPERATION_NAME && !input_types[0].unreduced_axes().is_empty() {
        return Err(TypeError::invalid(format!("`{operation_name}` does not support unreduced operands")));
    }
    let Some(shape) = input_types[0].static_shape() else {
        return Err(TypeError::invalid(format!("`{operation_name}` does not support dynamically shaped operands")));
    };
    Ok(shape.dimensions().to_vec())
}

/// Builds a shape-changing collective's output type from its operand and resized dimensions, carrying the operand
/// sharding through with the same per-dimension placement (the dimension count never changes).
pub(super) fn shape_changing_collective_output_type(
    operation_name: &'static str,
    input_type: &ArrayType,
    output_dimensions: Vec<usize>,
) -> Result<ArrayType, TypeError> {
    let output_sizes = output_dimensions.into_iter().map(Dimension::Static).collect::<Vec<_>>();
    let sharding = resized_output_sharding(input_type, output_sizes.as_slice(), operation_name)?;
    let mut output_type =
        ArrayType::new(input_type.data_type(), Shape::new(output_sizes)).with_memory(input_type.memory());
    output_type.sharding = sharding;
    Ok(output_type)
}

/// Infers one canonical mixed collective result from an array operand followed by one explicit extent per output
/// axis.
pub(super) fn infer_explicit_shape_changing_collective_output_type(
    operation_name: &'static str,
    input_types: &[ArrayIrType],
    base_output_type: ArrayType,
    unchanged_input_axes: &[Option<usize>],
    validate_exact_extents: impl FnOnce(&ArrayType, &[Dimension]) -> Result<(), TypeError>,
) -> Result<Vec<ArrayIrType>, TypeError> {
    let expected = 1 + base_output_type.rank();
    check_count!("input", input_types, expected, TypeError);
    let input_type = <&ArrayType>::try_from(&input_types[0])?;
    if operation_name != PARALLEL_SUM_SCATTER_OPERATION_NAME && !input_type.unreduced_axes().is_empty() {
        return Err(TypeError::invalid(format!("`{operation_name}` does not support unreduced operands")));
    }
    let output_extents = ArrayIrType::extents(&input_types[1..])?;
    if unchanged_input_axes.len() != output_extents.len() {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` internal output-axis mapping has length {} but the result rank is {}",
            unchanged_input_axes.len(),
            output_extents.len(),
        )));
    }
    for (output_axis, (&input_axis, output_extent)) in unchanged_input_axes.iter().zip(&output_extents).enumerate() {
        let Some(input_axis) = input_axis else { continue };
        let input_extent = input_type.shape().dimensions().get(input_axis).ok_or_else(|| {
            TypeError::invalid(format!(
                "`{operation_name}` unchanged output axis {output_axis} references input axis {input_axis}, which is \
                 out of bounds for rank {}",
                input_type.rank(),
            ))
        })?;
        if output_extent != input_extent {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` output axis {output_axis} extent {output_extent} must equal unchanged input axis \
                 {input_axis} extent {input_extent}",
            )));
        }
    }
    validate_exact_extents(input_type, output_extents.as_slice())?;
    Ok(vec![base_output_type.with_shape(Shape::new(output_extents)).into()])
}

/// Interprets a shape-changing collective outside any binder: only the degenerate single-participant axis
/// (`axis_size == 1`) has defined per-item semantics (the identity), and any larger axis reports an error because
/// the other participants do not exist per item.
pub(super) fn interpret_degenerate_collective<V: Clone>(
    operation_name: &str,
    axis_name: &str,
    axis_size: usize,
    inputs: &[V],
) -> Result<Vec<V>, ProgramError> {
    check_count!("input", inputs, 1, ProgramError);
    if axis_size != 1 {
        return Err(ProgramError::UnsupportedOperation {
            message: format!(
                "cannot interpret `{operation_name}` over axis `{axis_name}` of size {axis_size} without an \
                 enclosing binder",
            ),
        });
    }
    Ok(vec![inputs[0].clone()])
}

/// Representation boundary used only by shape-changing collective batching rules.
///
/// The collective kernels own every formula. This trait exposes only the extent representation and the alignment and
/// reshape encodings that differ between homogeneous arrays and composite array/dimension programs.
pub(crate) trait CollectiveBatchingPolicy<C: Context<Type = ArrayType>>: ArrayBatchingPolicy<C> {
    /// Extent representation consumed by the shared collective kernels.
    type ShapeExtent: Clone + Debug + Div + Mul;

    /// Returns and validates the active mapped-axis extent in the kernel's representation.
    fn collective_axis_extent(
        context: &BatchingContext<C, ArrayBatching<Self>>,
        operation_name: &str,
        axis_name: &str,
        axis_size: usize,
    ) -> Result<Self::ShapeExtent, BatchingError>;

    /// Materializes a statically known extent in the kernel's representation.
    fn collective_extent_constant(
        context: &BatchingContext<C, ArrayBatching<Self>>,
        extent: usize,
    ) -> Result<Self::ShapeExtent, BatchingError>;

    /// Materializes a statically known type-level dimension in the kernel's representation.
    fn collective_extent_from_dimension(
        context: &BatchingContext<C, ArrayBatching<Self>>,
        dimension: &Dimension,
    ) -> Result<Self::ShapeExtent, BatchingError> {
        let extent = dimension.value().ok_or_else(|| BatchingError::UnsupportedOperation {
            message: "shape-changing collective batching requires statically shaped operands".to_string(),
        })?;
        Self::collective_extent_constant(context, extent)
    }

    /// Enforces exact divisibility when it is not statically decidable.
    fn require_divisible_collective_extents(
        left: &Self::ShapeExtent,
        right: &Self::ShapeExtent,
    ) -> Result<(), BatchingError>;

    /// Aligns `batch` to the leading mapped axis using its complete logical input extents.
    fn match_collective_axis(
        context: &BatchingContext<C, ArrayBatching<Self>>,
        batch: &ArrayBatch<C::Value>,
        input_extents: &[Self::ShapeExtent],
    ) -> Result<ArrayBatch<C::Value>, BatchingError>;

    /// Reshapes `value` using a complete extent list in this policy's representation.
    fn reshape_collective(
        context: &BatchingContext<C, ArrayBatching<Self>>,
        value: C::Value,
        output_extents: &[Self::ShapeExtent],
        output_sharding: Option<Sharding>,
    ) -> Result<C::Value, BatchingError>;
}

impl<C> CollectiveBatchingPolicy<C> for StaticArrayBatchingPolicy
where
    C: Context<Type = ArrayType, Value: Broadcast + Reshape + Transpose>,
{
    type ShapeExtent = usize;

    fn collective_axis_extent(
        context: &BatchingContext<C, ArrayBatching<Self>>,
        operation_name: &str,
        axis_name: &str,
        axis_size: usize,
    ) -> Result<Self::ShapeExtent, BatchingError> {
        let batch_size = *context.axis_extent();
        if batch_size != axis_size {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`{operation_name}` over axis `{axis_name}` resolved axis size {axis_size} but the mapped batch \
                     axis has size {batch_size}",
                ),
            });
        }
        Ok(batch_size)
    }

    fn collective_extent_constant(
        _context: &BatchingContext<C, ArrayBatching<Self>>,
        extent: usize,
    ) -> Result<Self::ShapeExtent, BatchingError> {
        Ok(extent)
    }

    fn require_divisible_collective_extents(
        left: &Self::ShapeExtent,
        right: &Self::ShapeExtent,
    ) -> Result<(), BatchingError> {
        if *right == 0 || left % right != 0 {
            return Err(BatchingError::UnsupportedOperation {
                message: format!("extent {left} must be divisible by extent {right}"),
            });
        }
        Ok(())
    }

    fn match_collective_axis(
        context: &BatchingContext<C, ArrayBatching<Self>>,
        batch: &ArrayBatch<C::Value>,
        _input_extents: &[Self::ShapeExtent],
    ) -> Result<ArrayBatch<C::Value>, BatchingError> {
        Self::match_axis(context, batch, 0.into())
    }

    fn reshape_collective(
        _context: &BatchingContext<C, ArrayBatching<Self>>,
        value: C::Value,
        output_extents: &[Self::ShapeExtent],
        output_sharding: Option<Sharding>,
    ) -> Result<C::Value, BatchingError> {
        let output_shape = Shape::new(output_extents.iter().copied().map(Dimension::Static).collect());
        if value.r#type().shape() == &output_shape && value.r#type().sharding() == output_sharding.as_ref() {
            return Ok(value);
        }
        Ok(value.reshape(ReshapeParameters::new(output_shape).with_output_sharding(output_sharding))?)
    }
}

impl<C> CollectiveBatchingPolicy<ProjectedContext<C, ArrayType>> for DynamicArrayBatchingPolicy
where
    C: Context<
            Type = ArrayIrType,
            Operation: From<DynamicBroadcastOperation>
                           + From<ConstantOperation<DimensionValue>>
                           + From<DimensionSizeOperation>
                           + From<DynamicReshapeOperation>
                           + OperationProjection<ArrayType>,
        >,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value:
        ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>> + ValueProjection<DimensionType>,
    <C::Value as ValueProjection<DimensionType>>::Projected:
        DimensionRequirement + Div + Mul + Value<Type = DimensionType>,
{
    type ShapeExtent = <C::Value as ValueProjection<DimensionType>>::Projected;

    fn collective_axis_extent(
        context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatching<Self>>,
        _operation_name: &str,
        _axis_name: &str,
        axis_size: usize,
    ) -> Result<Self::ShapeExtent, BatchingError> {
        let axis_extent = <C::Value as ValueProjection<DimensionType>>::into_projected(context.axis_extent().clone())?;
        let axis_size = Self::collective_extent_constant(context, axis_size)?;
        axis_extent.require_equal(&axis_size)?;
        Ok(axis_extent)
    }

    fn collective_extent_constant(
        context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatching<Self>>,
        extent: usize,
    ) -> Result<Self::ShapeExtent, BatchingError> {
        let value = DimensionValue::constant(extent).map_err(ProgramError::from)?;
        let mut outputs = context.parent().parent().bind(ConstantOperation::new(value), Vec::new(), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(<C::Value as ValueProjection<DimensionType>>::into_projected(outputs.remove(0))?)
    }

    fn require_divisible_collective_extents(
        left: &Self::ShapeExtent,
        right: &Self::ShapeExtent,
    ) -> Result<(), BatchingError> {
        left.require_divisible_by(right).map_err(Into::into)
    }

    fn match_collective_axis(
        context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatching<Self>>,
        batch: &ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>,
        input_extents: &[Self::ShapeExtent],
    ) -> Result<ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>, BatchingError> {
        if !batch.batch_axis().is_replicated() {
            return batch.move_axis(0);
        }

        let input_type = batch.unbatched_type();
        let input_extent_dimensions =
            input_extents.iter().map(|extent| extent.r#type().to_dimension()).collect::<Vec<_>>();
        let value = if input_type.shape().dimensions() == input_extent_dimensions {
            batch.value().clone()
        } else {
            Self::reshape_collective(context, batch.value().clone(), input_extents, input_type.sharding().cloned())?
        };
        let output_axes = (1..=input_type.rank()).collect::<Vec<_>>();
        let output_sharding = input_type
            .sharding()
            .map(|sharding| {
                sharding
                    .with_inserted_dimension(0, context.axis_sharding().clone())
                    .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })
            })
            .transpose()?;
        let mut output_extents = Vec::with_capacity(input_extents.len() + 1);
        output_extents.push(context.axis_extent().clone());
        output_extents
            .extend(input_extents.iter().cloned().map(<C::Value as ValueProjection<DimensionType>>::from_projected));
        let value = broadcast_array(
            context.parent().parent(),
            <C::Value as ValueProjection<ArrayType>>::from_projected(value),
            output_extents,
            output_axes,
            output_sharding,
        )?;
        ArrayBatch::new(<C::Value as ValueProjection<ArrayType>>::into_projected(value)?, BatchAxis::from_position(0))
    }

    fn reshape_collective(
        context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatching<Self>>,
        value: <C::Value as ValueProjection<ArrayType>>::Projected,
        output_extents: &[Self::ShapeExtent],
        output_sharding: Option<Sharding>,
    ) -> Result<<C::Value as ValueProjection<ArrayType>>::Projected, BatchingError> {
        let operation = DynamicReshapeOperation::new().with_output_sharding(output_sharding);
        let inputs = std::iter::once(<C::Value as ValueProjection<ArrayType>>::from_projected(value))
            .chain(output_extents.iter().cloned().map(<C::Value as ValueProjection<DimensionType>>::from_projected))
            .collect::<Vec<_>>();
        let mut outputs = context.parent().parent().bind(operation, Vec::new(), inputs.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(<C::Value as ValueProjection<ArrayType>>::into_projected(outputs.remove(0))?)
    }
}

/// Forwards one shape-changing collective while updating its mapped result axis.
pub(super) fn forward_shape_changing_collective<C, P>(
    context: &BatchingContext<C, ArrayBatching<P>>,
    operation: C::Operation,
    input: &ArrayBatch<C::Value>,
    output_batch_axis: Option<usize>,
) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError>
where
    C: Context<Type = ArrayType>,
    P: ArrayBatchingPolicy<C>,
{
    let mut outputs = context.parent().bind(operation, Vec::new(), std::slice::from_ref(input.value()))?;
    check_count!("output", outputs, 1, ProgramError);
    let output = outputs.remove(0);
    let output_batch_axis = output_batch_axis.map_or_else(BatchAxis::replicated, BatchAxis::from_position);
    Ok(vec![ArrayBatch::new(output, output_batch_axis)?])
}

/// Implements the shared structure of the shape-changing collectives: the operation struct with its accessors, the
/// `Display`/`Operation` implementations (with payload-dependent output-shape inference provided as a closure over
/// the operand dimensions), degenerate interpretation, default partial evaluation, the linear forward-mode rule
/// (the tangent rides the same collective), and, for the collectives that request it, the homogeneous value-level
/// staging capability that resolves the named axis size from the active [`NamedAxes`] environment. The batching rules
/// are hand-written below the macro invocations because each collective materializes the mapped batch axis
/// differently.
macro_rules! shape_changing_collective {
    // Public form: the operation is always generated, while the `capability = Trait::method` section is optional and
    // only requested by collectives that still expose a homogeneous staging capability. The field list is forwarded
    // as an opaque token tree so that the optional capability section stays independent of the field repetition.
    (
        $(#[$operation_documentation:meta])*
        operation = $operation:ident,
        name = $operation_name:ident = $name_literal:literal,
        $(
            $(#[$capability_documentation:meta])*
            capability = $capability:ident::$method:ident,
        )?
        fields = $fields:tt,
        infer = |$infer_self:ident, $input_type:ident, $dimensions:ident| $infer:block $(,)?
    ) => {
        shape_changing_collective! {
            @operation
            $(#[$operation_documentation])*
            operation = $operation,
            name = $operation_name = $name_literal,
            fields = $fields,
            infer = |$infer_self, $input_type, $dimensions| $infer,
        }

        $(
            shape_changing_collective! {
                @capability
                operation = $operation,
                $(#[$capability_documentation])*
                capability = $capability::$method,
                fields = $fields,
            }
        )?
    };

    // Internal branch: generates the operation struct with its accessors, the `Display`/`Operation` implementations,
    // degenerate interpretation, default partial evaluation, and the linear forward-mode rule.
    (
        @operation
        $(#[$operation_documentation:meta])*
        operation = $operation:ident,
        name = $operation_name:ident = $name_literal:literal,
        fields = { $($(#[$field_documentation:meta])* $field:ident: $field_type:ty),* $(,)? },
        infer = |$infer_self:ident, $input_type:ident, $dimensions:ident| $infer:block $(,)?
    ) => {
        /// Canonical operation name for the operation.
        pub const $operation_name: &str = $name_literal;

        $(#[$operation_documentation])*
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        pub struct $operation {
            /// Axis name referenced by this collective.
            axis_name: String,

            /// Number of participants along the named axis, resolved from the active [`NamedAxes`] environment when
            /// the operation is staged.
            axis_size: usize,

            $($(#[$field_documentation])* $field: $field_type,)*
        }

        impl $operation {
            /// Creates a new operation over the named axis with the provided resolved axis size.
            #[inline]
            pub fn new(axis_name: String, axis_size: usize, $($field: $field_type),*) -> Self {
                Self { axis_name, axis_size, $($field),* }
            }

            /// Returns the axis name referenced by this collective.
            #[inline]
            pub fn axis_name(&self) -> &str {
                &self.axis_name
            }

            /// Returns the number of participants along the named axis.
            #[inline]
            pub fn axis_size(&self) -> usize {
                self.axis_size
            }
        }

        impl Display for $operation {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.render(formatter, 0)
            }
        }

        impl Operation for $operation {
            type Type = ArrayType;

            #[inline]
            fn name(&self) -> &'static str {
                $operation_name
            }

            fn infer_output_types(
                &self,
                input_types: &[ArrayType],
                region_interfaces: &[RegionInterface<ArrayType>],
            ) -> Result<Vec<ArrayType>, TypeError> {
                check_count!("region", region_interfaces, 0, TypeError);
                validate_collective_axis_size($name_literal, self.axis_size)?;
                let $dimensions = shape_changing_collective_dimensions($name_literal, input_types)?;
                let $infer_self = self;
                let $input_type = &input_types[0];
                Ok(vec![$infer?])
            }

            fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
                OperationFormatter::new(formatter, indentation, $operation_name)?.bracketed(|operation| {
                    operation.field("axis_name", format_args!("{:?}", self.axis_name))?;
                    operation.field("axis_size", &self.axis_size)?;
                    $(operation.field(stringify!($field), format_args!("{:?}", &self.$field))?;)*
                    Ok(())
                })
            }
        }

        impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for $operation {
            fn interpret<D: InterpretationDriver<C>>(
                &self,
                _context: &C,
                _driver: &D,
                inputs: &[C::Value],
            ) -> Result<Vec<C::Value>, ProgramError> {
                interpret_degenerate_collective($name_literal, &self.axis_name, self.axis_size, inputs)
            }
        }

        // Partial evaluation defers to the default fold-or-residualize behavior of
        // `Program::partially_evaluate`.
        impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for $operation where
            C::Operation: From<$operation>
        {
        }

        // Forward-mode rule: the collective is linear, so the tangent rides the same collective. Structural-zero
        // tangents stay symbolic, retyped to the output tangent type (the collective changes shapes).
        impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for $operation
        where
            C::Operation: From<$operation>,
        {
            fn jvp<D: DifferentiationDriver<C>>(
                &self,
                context: &C,
                _driver: &D,
                inputs: &[DifferentiationDual<C::Value>],
            ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
                check_count!("input", inputs, 1, ProgramError);
                let mut primal_outputs =
                    context.bind(self.clone(), Vec::new(), std::slice::from_ref(inputs[0].primal()))?;
                check_count!("output", primal_outputs, 1, ProgramError);
                let primal = primal_outputs.remove(0);
                let tangent = match inputs[0].tangent() {
                    MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
                    MaybeZero::Value(tangent) => {
                        let mut tangent_outputs =
                            context.bind(self.clone(), Vec::new(), std::slice::from_ref(tangent))?;
                        check_count!("output", tangent_outputs, 1, ProgramError);
                        MaybeZero::Value(tangent_outputs.remove(0))
                    }
                };
                Ok(vec![DifferentiationDual::new(primal, tangent)?])
            }
        }
    };

    // Internal branch: generates the homogeneous value-level staging capability that resolves the named axis size
    // from the active [`NamedAxes`] environment.
    (
        @capability
        operation = $operation:ident,
        $(#[$capability_documentation:meta])*
        capability = $capability:ident::$method:ident,
        fields = { $($(#[$field_documentation:meta])* $field:ident: $field_type:ty),* $(,)? } $(,)?
    ) => {
        $(#[$capability_documentation])*
        pub trait $capability: Sized {
            /// Stages this collective over axis `axis_name`, resolving the axis size from the active
            /// [`NamedAxes`](crate::axes::NamedAxes) environment and returning an
            /// [`AxisError::UnboundAxisName`](crate::axes::AxisError::UnboundAxisName) error when no enclosing binder
            /// binds the name.
            fn $method(&self, axis_name: &str, $($field: $field_type),*) -> Result<Self, ProgramError>;
        }

        impl<V: Value<Type = ArrayType>> $capability for V
        where
            V::DispatchDomain: Context + NamedAxes,
            <V::DispatchDomain as Domain>::Operation: From<$operation>,
        {
            fn $method(&self, axis_name: &str, $($field: $field_type),*) -> Result<Self, ProgramError> {
                let context = self.dispatch_domain();
                let axis_size = resolve_named_axis_size(&context, axis_name)?;
                let mut outputs = context.bind(
                    $operation::new(axis_name.to_string(), axis_size, $($field),*),
                    Vec::new(),
                    std::slice::from_ref(self),
                )?;
                check_count!("output", outputs, 1, ProgramError);
                Ok(outputs.remove(0))
            }
        }
    };
}

pub(super) use shape_changing_collective;

macro_rules! impl_shape_changing_collective_member_operation {
    // Implements the explicit array IR boundary shared by the three shape-changing collective payloads.
    ($operation:ty, $infer_output_types:ident) => {
        impl MemberOperation<ArrayIrType> for $operation {
            fn infer_parent_region_input_types(
                &self,
                _input_types: &[ArrayIrType],
                region_interfaces: &[RegionInterface<ArrayIrType>],
            ) -> Result<Vec<Option<Vec<ArrayIrType>>>, TypeError> {
                Ok(vec![None; region_interfaces.len()])
            }

            fn infer_parent_output_types(
                &self,
                input_types: &[ArrayIrType],
                region_interfaces: &[RegionInterface<ArrayIrType>],
            ) -> Result<Vec<ArrayIrType>, TypeError> {
                check_count!("region", region_interfaces, 0, TypeError);
                $infer_output_types(self, input_types)
            }

            fn rename_parent_type_identities(
                &self,
                renaming: &TypeIdentityRenaming<DimensionVariable>,
            ) -> Result<Self, TypeError> {
                self.rename_type_identities(renaming)
            }
        }

        impl<C> MemberInterpretableOperation<C> for $operation
        where
            C: Domain<Type = ArrayIrType>,
            C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType> + DimensionSize<usize> + Reshape>
                + ValueProjection<DimensionType, Projected = DimensionValue>,
        {
            fn interpret_in_parent<D: InterpretationDriver<C>>(
                &self,
                _context: &C,
                _driver: &D,
                inputs: &[C::Value],
            ) -> Result<Vec<C::Value>, ProgramError> {
                let Some((input, output_extents)) = inputs.split_first() else {
                    return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 });
                };
                let input = <C::Value as ValueProjection<ArrayType>>::into_projected(input.clone())?;
                let concrete_input_type = input.r#type().as_ref().clone().with_shape(Shape::new(
                    (0..input.r#type().rank())
                        .map(|axis| input.dimension_size(axis).map(Dimension::Static))
                        .collect::<Result<Vec<_>, _>>()?,
                ));
                let mut output_types = self.infer_output_types(std::slice::from_ref(&concrete_input_type), &[])?;
                check_count!("output", output_types, 1, ProgramError);
                let output_type = output_types.remove(0);
                let expected_extents = output_type.static_shape().ok_or_else(|| {
                    TypeError::invalid(format!("`{}` could not resolve its concrete output shape", self.name()))
                })?;
                if output_extents.len() != expected_extents.rank() {
                    return Err(ProgramError::InvalidInputCount {
                        expected: 1 + expected_extents.rank(),
                        actual: inputs.len(),
                    });
                }
                for (axis, (extent, expected)) in output_extents.iter().zip(expected_extents.dimensions()).enumerate() {
                    let actual = <C::Value as ValueProjection<DimensionType>>::into_projected(extent.clone())?.extent();
                    if actual != *expected {
                        return Err(ProgramError::InvalidArgument {
                            message: format!(
                                "`{}` output axis {axis} extent must equal observed result extent {expected} but got \
                                 {actual}",
                                self.name(),
                            ),
                        });
                    }
                }
                let effective_axis_size = self.effective_axis_size()?;
                if effective_axis_size != 1 {
                    return Err(ProgramError::UnsupportedOperation {
                        message: format!(
                            "cannot interpret `{}` over axis `{}` of size {} without an enclosing binder",
                            self.name(),
                            self.axis_name(),
                            effective_axis_size,
                        ),
                    });
                }
                let output = match self.options().mode() {
                    CollectiveMode::Tiled => input,
                    CollectiveMode::Untiled => input.reshape(Shape::from(expected_extents))?,
                };
                Ok(vec![<C::Value as ValueProjection<ArrayType>>::from_projected(output)])
            }
        }
    };
}

pub(super) use impl_shape_changing_collective_member_operation;

/// Returns an exact first-class collective extent constant.
pub(super) fn collective_extent_constant<V>(context: &V::DispatchDomain, extent: usize) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayIrType>,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Constant: From<DimensionValue>,
{
    context.lift(DimensionValue::constant(extent)?.into())
}

/// Returns one first-class dimension for every input array axis, using exact constants for static axes and explicit
/// [`DimensionSize`] gateways for dynamic axes.
pub(super) fn collective_input_extents<V>(context: &V::DispatchDomain, value: &V) -> Result<Vec<V>, ProgramError>
where
    V: Value<Type = ArrayIrType> + DimensionSize<V>,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Constant: From<DimensionValue>,
{
    let r#type = value.r#type();
    let input_type = <&ArrayType>::try_from(r#type.as_ref())?;
    input_type
        .shape()
        .dimensions()
        .iter()
        .enumerate()
        .map(|(axis, dimension)| match dimension {
            Dimension::Static(extent) => collective_extent_constant(context, *extent),
            Dimension::Dynamic(_) => value.dimension_size(axis),
        })
        .collect()
}

/// Computes one tiled collective result extent by multiplying an input-axis extent by the effective participant count.
pub(super) fn multiplied_collective_extent<V>(
    context: &V::DispatchDomain,
    input_extent: &V,
    effective_axis_size: usize,
) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayIrType> + ValueProjection<DimensionType>,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Constant: From<DimensionValue>,
    <V as ValueProjection<DimensionType>>::Projected: Mul,
{
    let input_extent = <V as ValueProjection<DimensionType>>::into_projected(input_extent.clone())?;
    let effective_axis_size = collective_extent_constant(context, effective_axis_size)?;
    let effective_axis_size = <V as ValueProjection<DimensionType>>::into_projected(effective_axis_size)?;
    Ok(<V as ValueProjection<DimensionType>>::from_projected(input_extent.mul(&effective_axis_size)?))
}

/// Computes one tiled collective result extent by requiring exact divisibility and dividing an input-axis extent by
/// the effective participant count.
pub(super) fn divided_collective_extent<V>(
    context: &V::DispatchDomain,
    input_extent: &V,
    effective_axis_size: usize,
) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayIrType> + ValueProjection<DimensionType>,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Constant: From<DimensionValue>,
    <V as ValueProjection<DimensionType>>::Projected: DimensionRequirement + Div,
{
    let input_extent = <V as ValueProjection<DimensionType>>::into_projected(input_extent.clone())?;
    let effective_axis_size = collective_extent_constant(context, effective_axis_size)?;
    let effective_axis_size = <V as ValueProjection<DimensionType>>::into_projected(effective_axis_size)?;
    input_extent.require_divisible_by(&effective_axis_size)?;
    Ok(<V as ValueProjection<DimensionType>>::from_projected(input_extent.div(&effective_axis_size)?))
}

/// Requires an input axis extent to equal the effective participant count used by an untiled collective.
pub(super) fn require_collective_axis_extent<V>(
    context: &V::DispatchDomain,
    input_extent: &V,
    effective_axis_size: usize,
) -> Result<(), ProgramError>
where
    V: Value<Type = ArrayIrType> + ValueProjection<DimensionType>,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Constant: From<DimensionValue>,
    <V as ValueProjection<DimensionType>>::Projected: DimensionRequirement,
{
    let input_extent = <V as ValueProjection<DimensionType>>::into_projected(input_extent.clone())?;
    let effective_axis_size = collective_extent_constant(context, effective_axis_size)?;
    let effective_axis_size = <V as ValueProjection<DimensionType>>::into_projected(effective_axis_size)?;
    input_extent.require_equal(&effective_axis_size)
}

/// Requires an input axis extent to be exactly divisible by the effective participant count.
pub(super) fn require_collective_axis_divisible<V>(
    context: &V::DispatchDomain,
    input_extent: &V,
    effective_axis_size: usize,
) -> Result<(), ProgramError>
where
    V: Value<Type = ArrayIrType> + ValueProjection<DimensionType>,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Constant: From<DimensionValue>,
    <V as ValueProjection<DimensionType>>::Projected: DimensionRequirement,
{
    let input_extent = <V as ValueProjection<DimensionType>>::into_projected(input_extent.clone())?;
    let effective_axis_size = collective_extent_constant(context, effective_axis_size)?;
    let effective_axis_size = <V as ValueProjection<DimensionType>>::into_projected(effective_axis_size)?;
    input_extent.require_divisible_by(&effective_axis_size)
}

/// Applies the mixed array IR JVP shared by shape-changing collectives whose transpose is another collective.
/// Explicit output extents and the exact input shape become ordinary residuals of one linear call.
pub(super) fn jvp_shape_changing_collective_with_adjoint<C, Forward, Adjoint>(
    operation: &Forward,
    adjoint: Adjoint,
    context: &C,
    inputs: &[DifferentiationDual<C::Value>],
) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError>
where
    C: Context<Type = ArrayIrType>,
    C::Operation: From<Forward>
        + From<Adjoint>
        + From<DimensionSizeOperation>
        + From<LinearCallOperation<ArrayIrType>>
        + From<ConstantOperation<DimensionValue>>,
    Forward: Clone + Operation<Type = ArrayType>,
    Adjoint: Operation<Type = ArrayType>,
{
    let Some((array, output_extents)) = inputs.split_first() else {
        return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
    };
    let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
    let primal = context.bind(operation.clone(), Vec::new(), primal_inputs.as_slice())?.remove(0);
    let tangent = match array.tangent() {
        MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
        MaybeZero::Value(array_tangent) => {
            let mut residuals = LinearResiduals::new();
            let output_extents = residuals.retain_all(output_extents.iter().map(|extent| extent.primal().clone()));
            let input_shape = residuals.retain_shape(context, array.primal())?;
            let forward_operation = operation.clone();
            let forward_output_extents = output_extents.clone();
            let tangent = LinearCallOperation::stage(
                context,
                residuals.into_values(),
                vec![array_tangent.clone()],
                move |residuals, linear_inputs| {
                    let mut collective_inputs = Vec::with_capacity(1 + forward_output_extents.len());
                    collective_inputs.push(linear_inputs[0].clone());
                    collective_inputs.extend(forward_output_extents.iter().map(|index| residuals[*index].clone()));
                    linear_inputs[0].dispatch_domain().bind(forward_operation, Vec::new(), collective_inputs.as_slice())
                },
                move |residuals, output_cotangents| {
                    let transpose_context = output_cotangents[0].dispatch_domain();
                    let input_dimensions = input_shape.dimensions(&transpose_context, residuals)?;
                    let mut adjoint_inputs = Vec::with_capacity(1 + input_dimensions.len());
                    adjoint_inputs.push(output_cotangents[0].clone());
                    adjoint_inputs.extend(input_dimensions);
                    transpose_context.bind(adjoint, Vec::new(), adjoint_inputs.as_slice())
                },
            )?
            .remove(0);
            MaybeZero::Value(tangent)
        }
    };
    Ok(vec![DifferentiationDual::new(primal, tangent)?])
}

/// Validates a mixed collective's array operand and replicated result extents.
pub(super) fn explicit_collective_inputs<'a, V: Value<Type = ArrayIrType>>(
    operation_name: &str,
    inputs: &'a [ArrayIrBatch<V>],
) -> Result<(&'a ArrayIrBatch<V>, &'a [ArrayIrBatch<V>]), BatchingError> {
    let Some((array, output_extents)) = inputs.split_first() else {
        return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
    };
    <&ArrayType>::try_from(&array.unbatched_type())?;
    if let Some(ragged_axis) = array.ragged_axes().first() {
        return Err(BatchingError::UnsupportedOperation {
            message: format!(
                "`{}` does not support bounded ragged dimension `{}` on operand 0",
                operation_name,
                ragged_axis.dimension(),
            ),
        });
    }
    for output_extent in output_extents {
        output_extent.validate_replicated_dimension()?;
    }
    Ok((array, output_extents))
}

/// Binds a mixed collective over a non-matching named axis after lifting the mapped axis into its explicit result
/// extents. Replicated arrays require no lifting and remain replicated.
pub(super) fn forward_explicit_collective<C, O>(
    operation: O,
    context: &BatchingContext<C, ArrayIrBatching>,
    array: &ArrayIrBatch<C::Value>,
    output_extents: &[ArrayIrBatch<C::Value>],
    output_batch_axis: Option<usize>,
) -> Result<Vec<ArrayIrBatch<C::Value>>, BatchingError>
where
    C: Context<Type = ArrayIrType, Operation: From<O>>,
{
    let mut physical_output_extents = output_extents.iter().map(|extent| extent.value().clone()).collect::<Vec<_>>();
    if let Some(output_batch_axis) = output_batch_axis {
        physical_output_extents.insert(output_batch_axis, context.axis_extent().clone());
    }
    let physical_inputs = std::iter::once(array.value().clone()).chain(physical_output_extents).collect::<Vec<_>>();
    context
        .parent()
        .bind(operation, Vec::new(), physical_inputs.as_slice())?
        .into_iter()
        .map(|output| match output_batch_axis {
            Some(output_batch_axis) => ArrayIrBatch::new(output, BatchAxis::from_position(output_batch_axis)),
            None => Ok(ArrayIrBatch::replicated(output)),
        })
        .collect()
}

/// Stages the adjoint collective of a linear shape-changing collective on the output cotangent: a known operand
/// receives a structural zero, a zero output cotangent stays symbolic, and a live cotangent rides the provided
/// adjoint operation.
pub(super) fn transpose_shape_changing_collective<V, O, A>(
    context: &mut TracingContext<V, O>,
    inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
    outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    adjoint: A,
) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError>
where
    V: Value<Type = ArrayType>,
    O: Operation<Type = ArrayType> + From<A>,
    A: Operation<Type = ArrayType>,
{
    check_count!("input", inputs, 1, ProgramError);
    check_count!("output", outputs, 1, ProgramError);
    if inputs[0].is_known() {
        return Ok(vec![MaybeZero::Zero(inputs[0].r#type().cotangent()?)]);
    }
    match &outputs[0] {
        MaybeZero::Value(cotangent) => {
            let mut contributions = context.bind(O::from(adjoint), Vec::new(), std::slice::from_ref(cotangent))?;
            check_count!("output", contributions, 1, ProgramError);
            Ok(vec![MaybeZero::Value(contributions.remove(0))])
        }
        MaybeZero::Zero(_) => Ok(vec![MaybeZero::Zero(inputs[0].r#type().cotangent()?)]),
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, DataType, DimensionBounds, DimensionVariable,
    };
    use crate::batching::BatchableOperation;
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{MemberDifferentiableOperation, transpose_mixed_operation};
    use crate::operations::collectives::all_gather::infer_explicit_all_gather_output_types;
    use crate::operations::collectives::all_to_all::infer_explicit_all_to_all_output_types;
    use crate::operations::collectives::parallel_sum_scatter::infer_explicit_parallel_sum_scatter_output_types;

    use super::*;

    /// Returns the static `f32` vector type of the provided length used by the shape-changing collective tests.
    pub(super) fn f32_vector(length: usize) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(length)]))
    }

    #[test]
    fn test_collective_options_validate_axis_index_groups() {
        let options = CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]);
        assert_eq!(options.mode(), CollectiveMode::Tiled);
        assert_eq!(options.axis_index_groups(), Some([vec![0, 2], vec![3, 1]].as_slice()));
        assert_eq!(options.effective_axis_size("all_gather", 4), Ok(2));

        assert_eq!(
            CollectiveOptions::default().with_axis_index_groups(Vec::new()).effective_axis_size("all_gather", 4),
            Err(TypeError::invalid("`all_gather` axis index groups must not be empty")),
        );
        assert_eq!(
            CollectiveOptions::default()
                .with_axis_index_groups(vec![vec![0, 1], vec![2]])
                .effective_axis_size("all_gather", 3),
            Err(TypeError::invalid("`all_gather` axis index group 1 has size 1 but every group must have size 2",)),
        );
        assert_eq!(
            CollectiveOptions::default()
                .with_axis_index_groups(vec![vec![0, 1], vec![1, 2]])
                .effective_axis_size("all_gather", 4),
            Err(TypeError::invalid("`all_gather` axis index groups contain participant 1 more than once",)),
        );
        assert_eq!(
            CollectiveOptions::default()
                .with_axis_index_groups(vec![vec![0, 1], vec![2, 4]])
                .effective_axis_size("all_gather", 4),
            Err(TypeError::invalid("`all_gather` axis index 4 is out of bounds for axis size 4")),
        );
        assert_eq!(
            CollectiveOptions::default()
                .with_axis_index_groups(vec![vec![0, 1]])
                .effective_axis_size("all_gather", 3),
            Err(TypeError::invalid("`all_gather` axis index groups do not contain participant 2")),
        );
    }

    #[test]
    fn test_untiled_collective_type_inference() {
        let shape = |dimensions| ArrayType::new(DataType::F32, Shape::new(dimensions));

        assert_eq!(
            infer_explicit_all_gather_output_types(
                &AllGatherOperation::new(
                    "x".to_string(),
                    4,
                    1,
                    CollectiveOptions::default(),
                    AllGatherOutputVariance::Varying,
                ),
                &[
                    shape(vec![Dimension::Static(2), Dimension::Static(3)]).into(),
                    DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(4).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                ],
            ),
            Ok(vec![shape(vec![Dimension::Static(2), Dimension::Static(4), Dimension::Static(3)]).into()]),
        );
        assert_eq!(
            infer_explicit_parallel_sum_scatter_output_types(
                &ParallelSumScatterOperation::new("x".to_string(), 4, 1, CollectiveOptions::default()),
                &[
                    shape(vec![Dimension::Static(2), Dimension::Static(4), Dimension::Static(3)]).into(),
                    DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                ],
            ),
            Ok(vec![shape(vec![Dimension::Static(2), Dimension::Static(3)]).into()]),
        );
        assert_eq!(
            infer_explicit_all_to_all_output_types(
                &AllToAllOperation::new("x".to_string(), 4, 1, 0, CollectiveOptions::default()),
                &[
                    shape(vec![Dimension::Static(2), Dimension::Static(4), Dimension::Static(3)]).into(),
                    DimensionValue::constant(4).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                ],
            ),
            Ok(vec![shape(vec![Dimension::Static(4), Dimension::Static(2), Dimension::Static(3)]).into()]),
        );
        assert_eq!(
            infer_explicit_all_to_all_output_types(
                &AllToAllOperation::new("x".to_string(), 4, 1, 1, CollectiveOptions::default()),
                &[
                    shape(vec![Dimension::Static(2), Dimension::Static(4), Dimension::Static(3)]).into(),
                    DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(4).unwrap().r#type().into_owned().into(),
                    DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                ],
            ),
            Ok(vec![shape(vec![Dimension::Static(2), Dimension::Static(4), Dimension::Static(3)]).into()]),
        );
        assert_eq!(
            infer_explicit_parallel_sum_scatter_output_types(
                &ParallelSumScatterOperation::new("x".to_string(), 4, 1, CollectiveOptions::default()),
                &[
                    shape(vec![Dimension::Static(2), Dimension::Static(5)]).into(),
                    DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
                ],
            ),
            Err(TypeError::invalid("`parallel_sum_scatter` untiled scatter axis 1 size 5 must equal group size 4",)),
        );
    }

    #[test]
    fn test_grouped_collective_shape_arithmetic_uses_group_size() {
        let grouped = CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]);
        let result_extent = DimensionValue::constant(6).unwrap().r#type().into_owned();
        assert_eq!(
            infer_explicit_all_gather_output_types(
                &AllGatherOperation::new("x".to_string(), 4, 0, grouped.clone(), AllGatherOutputVariance::Varying,),
                &[f32_vector(3).into(), result_extent.into(),],
            ),
            Ok(vec![f32_vector(6).into()]),
        );
        assert_eq!(
            infer_explicit_parallel_sum_scatter_output_types(
                &ParallelSumScatterOperation::new("x".to_string(), 4, 0, grouped),
                &[f32_vector(6).into(), DimensionValue::constant(3).unwrap().r#type().into_owned().into()],
            ),
            Ok(vec![f32_vector(3).into()]),
        );
    }

    #[test]
    fn test_explicit_shape_changing_collective_type_inference() {
        let input_axis = DimensionVariable::new("input", DimensionBounds::new(1, Some(17)).unwrap());
        let split_result = DimensionVariable::new("split", DimensionBounds::new(1, Some(9)).unwrap());
        let concat_result = DimensionVariable::new("concat", DimensionBounds::new(2, Some(33)).unwrap());
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(input_axis.clone()), Dimension::Static(3)]),
        );

        assert_eq!(
            infer_explicit_all_gather_output_types(
                &AllGatherOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::tiled(),
                    AllGatherOutputVariance::Varying
                ),
                &[
                    input_type.clone().into(),
                    ArrayIrType::Dimension(DimensionType::new(concat_result.clone())),
                    DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                ],
            ),
            Ok(vec![
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(concat_result.clone()), Dimension::Static(3)]),
                )
                .into()
            ]),
        );
        assert_eq!(
            infer_explicit_parallel_sum_scatter_output_types(
                &ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled()),
                &[
                    input_type.clone().into(),
                    ArrayIrType::Dimension(DimensionType::new(split_result.clone())),
                    DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                ],
            ),
            Ok(vec![
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(split_result.clone()), Dimension::Static(3)]),
                )
                .into()
            ]),
        );
        assert_eq!(
            infer_explicit_all_to_all_output_types(
                &AllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled()),
                &[
                    input_type.clone().into(),
                    ArrayIrType::Dimension(DimensionType::new(split_result.clone())),
                    ArrayIrType::Dimension(DimensionType::new(concat_result.clone())),
                ],
            ),
            Ok(vec![
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(split_result), Dimension::Dynamic(concat_result),]),
                )
                .into()
            ]),
        );
        assert_eq!(
            infer_explicit_all_to_all_output_types(
                &AllToAllOperation::new("x".to_string(), 2, 0, 0, CollectiveOptions::tiled()),
                &[
                    ArrayIrType::Array(input_type.clone()),
                    ArrayIrType::Dimension(DimensionType::new(input_axis)),
                    DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                ],
            ),
            Ok(vec![input_type.into()]),
        );

        let exact_six = DimensionValue::constant(6).unwrap().r#type().into_owned();
        assert_eq!(
            infer_explicit_all_gather_output_types(
                &AllGatherOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::tiled(),
                    AllGatherOutputVariance::Varying
                ),
                &[f32_vector(3).into(), exact_six.into()],
            ),
            Ok(vec![f32_vector(6).into()]),
        );
        let exact_five = DimensionValue::constant(5).unwrap().r#type().into_owned();
        assert_eq!(
            infer_explicit_all_gather_output_types(
                &AllGatherOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::tiled(),
                    AllGatherOutputVariance::Varying
                ),
                &[f32_vector(3).into(), exact_five.into()],
            ),
            Err(TypeError::invalid(
                "`all_gather` result extent must equal input axis 0 extent 3 multiplied by axis group size 2; \
                 expected 6 \
                 but got 5"
                    .to_string(),
            )),
        );
        assert_eq!(
            infer_explicit_parallel_sum_scatter_output_types(
                &ParallelSumScatterOperation::new("empty".to_string(), 0, 0, CollectiveOptions::tiled()),
                &[f32_vector(3).into(), DimensionValue::constant(3).unwrap().r#type().into_owned().into()],
            ),
            Err(TypeError::invalid("`parallel_sum_scatter` axis size must be greater than zero")),
        );
    }

    #[test]
    fn test_untiled_collectives_over_batched_axis_materialize_rank_changes() {
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2)
            .with_axis_name("x".to_string());
        let mapped_matrix = || ArrayBatch::new(Array::matrix(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0]), Some(0)).unwrap();

        let gathered = AllGatherOperation::new(
            "x".to_string(),
            2,
            1,
            CollectiveOptions::default(),
            AllGatherOutputVariance::Varying,
        )
        .batch(&context, &crate::EmptyRegionDriver, &[mapped_matrix()])
        .unwrap()
        .into_parts()
        .0;
        assert_eq!(gathered[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(gathered[0].value(), &Array::matrix(2, 2, vec![1.0_f32, 3.0, 2.0, 4.0]),);

        let scattered = ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::default())
            .batch(&context, &crate::EmptyRegionDriver, &[mapped_matrix()])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(scattered[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(scattered[0].value(), &Array::vector(vec![4.0_f32, 6.0]));
        assert_eq!(scattered[0].unbatched_type(), ArrayType::scalar(DataType::F32));

        let exchanged = AllToAllOperation::new("x".to_string(), 2, 0, 0, CollectiveOptions::default())
            .batch(&context, &crate::EmptyRegionDriver, &[mapped_matrix()])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(exchanged[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(exchanged[0].value(), &Array::matrix(2, 2, vec![1.0_f32, 3.0, 2.0, 4.0]),);
    }

    #[test]
    fn test_shape_changing_collective_transposes_are_involutive() {
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;

        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(f32_vector(8));
        let output = builder
            .add_instruction(
                ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled()),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        let transposed_twice =
            program.transpose_with_respect_to(&[0]).unwrap().transpose_with_respect_to(&[0]).unwrap();
        assert!(matches!(transposed_twice.instructions()[0].operation(), ArrayOperation::ParallelSumScatter(_)));
        assert_eq!(transposed_twice.input_types(), program.input_types());
        assert_eq!(transposed_twice.output_types(), program.output_types());

        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder
            .add_input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4), Dimension::Static(3)])));
        let output = builder
            .add_instruction(
                AllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled()),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        let transposed_twice =
            program.transpose_with_respect_to(&[0]).unwrap().transpose_with_respect_to(&[0]).unwrap();
        assert!(matches!(transposed_twice.instructions()[0].operation(), ArrayOperation::AllToAll(_)));
        assert_eq!(transposed_twice.input_types(), program.input_types());
        assert_eq!(transposed_twice.output_types(), program.output_types());
    }

    #[test]
    fn test_explicit_shape_changing_collective_member_transforms() -> Result<(), ProgramError> {
        type Context = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        // A live tangent through a dynamically shaped mixed collective stages one residual-aware linear call directly
        // through the payload's member JVP rule.
        let variable = DimensionVariable::new("items", DimensionBounds::new(1, Some(9))?);
        let dimension_type = DimensionType::new(variable.clone());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable)]));
        let context = Context::new();
        let primal = context.input(array_type.clone().into());
        let tangent = context.input(array_type.into());
        let extent = context.input(dimension_type.into());
        let extent_tangent_type = extent.r#type().tangent()?;
        let outputs = AllGatherOperation::new(
            "x".to_string(),
            1,
            0,
            CollectiveOptions::tiled(),
            AllGatherOutputVariance::Varying,
        )
        .jvp_in_parent(
            &context,
            &crate::EmptyRegionDriver,
            &[
                DifferentiationDual::new(primal, MaybeZero::Value(tangent))?,
                DifferentiationDual::new(extent, MaybeZero::Zero(extent_tangent_type))?,
            ],
        )?;
        assert!(matches!(outputs[0].tangent(), MaybeZero::Value(_)));
        assert!(
            context
                .builder()
                .borrow()
                .instructions()
                .iter()
                .any(|instruction| matches!(instruction.operation(), ArrayIrOperation::LinearCall(_)))
        );

        // Direct mixed transposition delegates the array contribution through the homogeneous projection and gives
        // the explicit extent operand a structural-zero cotangent.
        let mut context = Context::new();
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]));
        let output_cotangent = context.input(array_type.clone().into());
        let extent_type = DimensionValue::constant(3)?.r#type().into_owned();
        let cotangents = transpose_mixed_operation(
            &mut context,
            &ParallelSumScatterOperation::new("x".to_string(), 1, 0, CollectiveOptions::tiled()),
            &[PartialValue::Unknown(array_type.into()), PartialValue::Unknown(extent_type.into())],
            &[MaybeZero::Value(output_cotangent)],
        )?;
        assert!(matches!(cotangents.as_slice(), [MaybeZero::Value(_), MaybeZero::Zero(_)]));
        assert!(matches!(
            context.builder().borrow().instructions()[0].operation(),
            ArrayIrOperation::Array(ArrayOperation::AllGather(_)),
        ));

        Ok(())
    }
}
