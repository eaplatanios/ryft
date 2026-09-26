//! Shared machinery of the collectives that resize an array axis: [`AllGatherOperation`],
//! [`ParallelSumScatterOperation`], and [`AllToAllOperation`]. Their output shapes depend on the participant count and
//! on whether the named axis is materialized as a new array axis or tiled into an existing one, so, beyond the linear
//! collective structure of the sibling `linear` module, they share:
//!
//!   - [`CollectiveArrayExtentBatchingPolicy`], the representation boundary that lets one batching kernel per
//!     collective handle both homogeneous arrays with static extents and composite array/dimension programs with
//!     first-class extents ([`RaggedAllToAllOperation`] reuses it as well),
//!   - the first-class extent arithmetic that computes and validates result extents at staging time, and
//!   - the explicit [`ArrayIrType`] boundary, where the result extents are passed as additional dimension inputs, with
//!     its type inference, interpretation, batching, and forward-mode differentiation rules.
//!
//! [`AllGatherOperation`]: super::AllGatherOperation
//! [`ParallelSumScatterOperation`]: super::ParallelSumScatterOperation
//! [`AllToAllOperation`]: super::AllToAllOperation
//! [`RaggedAllToAllOperation`]: super::RaggedAllToAllOperation

// TODO(eaplatanios): Review this module.

use std::fmt::Debug;

use crate::arrays::batching::DynamicArrayExtentBatchingPolicy;
use crate::arrays::{
    ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrType,
    ArrayType, Dimension, DimensionType, DimensionValue, LinearResiduals, Shape, Sharding,
    StaticArrayExtentBatchingPolicy,
};
use crate::batching::{BatchAxis, BatchingContext, BatchingError};
use crate::contexts::{Context, ProjectedContext};
use crate::differentiation::{
    DifferentiableType, DifferentiationContext, DifferentiationDual, DifferentiationError, DifferentiationPolicy,
};
use crate::macros::check_count;
use crate::operations::arithmetic::{Div, Mul, Rem};
use crate::operations::assertions::Assert;
use crate::operations::comparisons::{Compare, ComparisonDirection};
use crate::operations::constants::constant::{ConstantOperation, DimensionConstant};
use crate::operations::differentiation::linear_call::LinearCallOperation;
use crate::operations::dimensions::dimension_max::DimensionMax;
use crate::operations::dimensions::dimension_size::{DimensionSize, DimensionSizeOperation};
use crate::operations::manipulation::broadcasting::{Broadcast, DynamicBroadcast, DynamicBroadcastOperation};
use crate::operations::manipulation::reshaping::{DynamicReshapeOperation, Reshape};
use crate::operations::manipulation::transposition::Transpose;
use crate::programs::{
    MaybeZero, Operation, OperationProjection, ProgramError, TypeError, Typed, Value, ValueProjection,
};

/// Infers one canonical mixed collective result from an array operand followed by one explicit extent per output
/// axis.
///
/// # Parameters
///
///   - `operation_name`: Name of the collective, used in diagnostics.
///   - `accepts_unreduced`: Whether the collective accepts array operands with unreduced axes (refer to
///     [`linear_collective_dimensions`](super::linear::linear_collective_dimensions) for more
///     information).
///   - `input_types`: Array operand type followed by one explicit extent type per output axis.
///   - `base_output_type`: Output type whose shape is replaced by the explicit extents.
///   - `unchanged_input_axes`: For every output axis, the input axis whose extent it must preserve, if any.
///   - `validate_exact_extents`: Collective-specific validation of the explicit extents against the array operand.
pub(super) fn infer_explicit_shape_changing_collective_output_type(
    operation_name: &'static str,
    accepts_unreduced: bool,
    input_types: &[ArrayIrType],
    base_output_type: ArrayType,
    unchanged_input_axes: &[Option<usize>],
    validate_exact_extents: impl FnOnce(&ArrayType, &[Dimension]) -> Result<(), TypeError>,
) -> Result<Vec<ArrayIrType>, TypeError> {
    let expected = 1 + base_output_type.rank();
    check_count!("input", input_types, expected, TypeError);
    let input_type = <&ArrayType>::try_from(&input_types[0])?;
    if !accepts_unreduced && !input_type.unreduced_axes().is_empty() {
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

/// Representation boundary used only by shape-changing collective batching rules.
///
/// The collective kernels own every formula. This trait exposes only the extent representation and the alignment and
/// reshape encodings that differ between homogeneous arrays and composite array/dimension programs.
pub(crate) trait CollectiveArrayExtentBatchingPolicy<C: Context<Type = ArrayType>>:
    ArrayExtentBatchingPolicy<C>
{
    /// Extent representation consumed by the shared collective kernels.
    type ShapeExtent: Clone + Debug + Div + Mul;

    /// Returns and validates the active mapped-axis extent in the kernel's representation.
    fn collective_axis_extent(
        context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
        operation_name: &str,
        axis_name: &str,
        axis_size: usize,
    ) -> Result<Self::ShapeExtent, BatchingError>;

    /// Materializes a statically known extent in the kernel's representation.
    fn collective_extent_constant(
        context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
        extent: usize,
    ) -> Result<Self::ShapeExtent, BatchingError>;

    /// Materializes a statically known type-level dimension in the kernel's representation.
    fn collective_extent_from_dimension(
        context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
        dimension: &Dimension,
    ) -> Result<Self::ShapeExtent, BatchingError> {
        let extent = dimension.value().ok_or_else(|| BatchingError::UnsupportedOperation {
            message: "shape-changing collective batching requires statically shaped operands".to_string(),
        })?;
        Self::collective_extent_constant(context, extent)
    }

    /// Enforces exact divisibility and returns a positive divisor safe for subsequent arithmetic.
    fn require_divisible_collective_extents(
        context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
        left: &Self::ShapeExtent,
        right: &Self::ShapeExtent,
    ) -> Result<Self::ShapeExtent, BatchingError>;

    /// Aligns `batch` to the leading mapped axis using its complete logical input extents.
    fn match_collective_axis(
        context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
        batch: &ArrayBatch<C::Value>,
        input_extents: &[Self::ShapeExtent],
    ) -> Result<ArrayBatch<C::Value>, BatchingError>;

    /// Reshapes `value` using a complete extent list in this policy's representation.
    fn reshape_collective(
        context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
        value: C::Value,
        output_extents: &[Self::ShapeExtent],
        output_sharding: Option<Sharding>,
    ) -> Result<C::Value, BatchingError>;
}

impl<C> CollectiveArrayExtentBatchingPolicy<C> for StaticArrayExtentBatchingPolicy
where
    C: Context<Type = ArrayType, Value: Broadcast + Reshape + Transpose>,
{
    type ShapeExtent = usize;

    fn collective_axis_extent(
        context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
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
        _context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
        extent: usize,
    ) -> Result<Self::ShapeExtent, BatchingError> {
        Ok(extent)
    }

    fn require_divisible_collective_extents(
        _context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
        left: &Self::ShapeExtent,
        right: &Self::ShapeExtent,
    ) -> Result<Self::ShapeExtent, BatchingError> {
        if *right == 0 || left % right != 0 {
            return Err(BatchingError::UnsupportedOperation {
                message: format!("extent {left} must be divisible by extent {right}"),
            });
        }
        Ok(*right)
    }

    fn match_collective_axis(
        context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
        batch: &ArrayBatch<C::Value>,
        _input_extents: &[Self::ShapeExtent],
    ) -> Result<ArrayBatch<C::Value>, BatchingError> {
        Self::match_axis(context, batch, 0.into())
    }

    fn reshape_collective(
        _context: &BatchingContext<C, ArrayBatchingPolicy<Self>>,
        value: C::Value,
        output_extents: &[Self::ShapeExtent],
        output_sharding: Option<Sharding>,
    ) -> Result<C::Value, BatchingError> {
        let output_shape = Shape::new(output_extents.iter().copied().map(Dimension::Static).collect());
        if value.r#type().shape() == &output_shape && value.r#type().sharding() == output_sharding.as_ref() {
            return Ok(value);
        }
        Ok(value.reshape_with_output_sharding(output_shape, output_sharding)?)
    }
}

impl<C> CollectiveArrayExtentBatchingPolicy<ProjectedContext<C, ArrayType>> for DynamicArrayExtentBatchingPolicy
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
    C::Value: Assert
        + DimensionSize
        + DynamicBroadcast
        + ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>
        + ValueProjection<DimensionType>,
    <C::Value as ValueProjection<DimensionType>>::Projected:
        Compare<C::Value> + DimensionMax + Rem + Div + Mul + Value<Type = DimensionType>,
{
    type ShapeExtent = <C::Value as ValueProjection<DimensionType>>::Projected;

    fn collective_axis_extent(
        context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatchingPolicy<Self>>,
        _operation_name: &str,
        _axis_name: &str,
        axis_size: usize,
    ) -> Result<Self::ShapeExtent, BatchingError> {
        let axis_extent = ValueProjection::<DimensionType>::into_projected(context.axis_extent().clone())?;
        let axis_size = Self::collective_extent_constant(context, axis_size)?;
        axis_extent.compare(&axis_size, ComparisonDirection::Equal)?.assert(
            "collective axis extent must match the participant count",
            &[
                ("extent", ValueProjection::<DimensionType>::from_projected(axis_extent.clone())),
                ("participants", ValueProjection::<DimensionType>::from_projected(axis_size)),
            ],
        )?;
        Ok(axis_extent)
    }

    fn collective_extent_constant(
        context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatchingPolicy<Self>>,
        extent: usize,
    ) -> Result<Self::ShapeExtent, BatchingError> {
        let value = DimensionValue::constant(extent).map_err(ProgramError::from)?;
        let mut outputs = context.parent().parent().bind(ConstantOperation::new(value), Vec::new(), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(ValueProjection::<DimensionType>::into_projected(outputs.remove(0))?)
    }

    fn require_divisible_collective_extents(
        context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatchingPolicy<Self>>,
        left: &Self::ShapeExtent,
        right: &Self::ShapeExtent,
    ) -> Result<Self::ShapeExtent, BatchingError> {
        let zero = Self::collective_extent_constant(context, 0)?;
        let one = Self::collective_extent_constant(context, 1)?;
        right.compare(&zero, ComparisonDirection::GreaterThan)?.assert(
            "collective divisor must be positive",
            &[("divisor", ValueProjection::<DimensionType>::from_projected(right.clone()))],
        )?;
        let divisor = right.dimension_max(&one)?;
        left.rem(&divisor)?.compare(&zero, ComparisonDirection::Equal)?.assert(
            "collective extent must be divisible by the participant count",
            &[
                ("extent", ValueProjection::<DimensionType>::from_projected(left.clone())),
                ("divisor", ValueProjection::<DimensionType>::from_projected(right.clone())),
            ],
        )?;
        Ok(divisor)
    }

    fn match_collective_axis(
        context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatchingPolicy<Self>>,
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
            .map(|sharding| sharding.batched(0, context.axis_sharding().clone()))
            .transpose()?;
        let mut output_extents = Vec::with_capacity(input_extents.len() + 1);
        output_extents.push(context.axis_extent().clone());
        output_extents
            .extend(input_extents.iter().cloned().map(<C::Value as ValueProjection<DimensionType>>::from_projected));
        let value = <C::Value as ValueProjection<ArrayType>>::from_projected(value)
            .dynamic_broadcast_with_output_sharding(&output_extents, &output_axes, output_sharding)?;
        ArrayBatch::new(<C::Value as ValueProjection<ArrayType>>::into_projected(value)?, BatchAxis::from_position(0))
    }

    fn reshape_collective(
        context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatchingPolicy<Self>>,
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
    context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
    operation: C::Operation,
    input: &ArrayBatch<C::Value>,
    output_batch_axis: Option<usize>,
) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError>
where
    C: Context<Type = ArrayType>,
    P: ArrayExtentBatchingPolicy<C>,
{
    let mut outputs = context.parent().bind(operation, Vec::new(), std::slice::from_ref(input.value()))?;
    check_count!("output", outputs, 1, ProgramError);
    let output = outputs.remove(0);
    let output_batch_axis = output_batch_axis.map_or_else(BatchAxis::replicated, BatchAxis::from_position);
    Ok(vec![ArrayBatch::new(output, output_batch_axis)?])
}

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
                    let actual = ValueProjection::<DimensionType>::into_projected(extent.clone())?.extent();
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
    V::DispatchDomain: DimensionConstant<Value = V>,
{
    context.dimension_constant(extent)
}

/// Returns one first-class dimension for every input array axis, using exact constants for static axes and explicit
/// [`DimensionSize`] gateways for dynamic axes.
pub(super) fn collective_input_extents<V>(context: &V::DispatchDomain, value: &V) -> Result<Vec<V>, ProgramError>
where
    V: Value<Type = ArrayIrType> + DimensionSize<V>,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    V::DispatchDomain: DimensionConstant,
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
    V::DispatchDomain: DimensionConstant,
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
    V: Value<Type = ArrayIrType> + Assert + ValueProjection<DimensionType>,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    V::DispatchDomain: DimensionConstant,
    <V as ValueProjection<DimensionType>>::Projected:
        Value<Type = DimensionType> + Compare<V> + DimensionMax + Rem + Div,
{
    let input_extent = <V as ValueProjection<DimensionType>>::into_projected(input_extent.clone())?;
    let effective_axis_size = collective_extent_constant(context, effective_axis_size)?;
    let effective_axis_size = <V as ValueProjection<DimensionType>>::into_projected(effective_axis_size)?;
    let zero = ValueProjection::<DimensionType>::into_projected(context.dimension_constant(0)?)?;
    let one = ValueProjection::<DimensionType>::into_projected(context.dimension_constant(1)?)?;
    effective_axis_size.compare(&zero, ComparisonDirection::GreaterThan)?.assert(
        "collective divisor must be positive",
        &[("divisor", ValueProjection::<DimensionType>::from_projected(effective_axis_size.clone()))],
    )?;
    let divisor = effective_axis_size.dimension_max(&one)?;
    input_extent.rem(&divisor)?.compare(&zero, ComparisonDirection::Equal)?.assert(
        "collective extent must be divisible by the participant count",
        &[
            ("extent", ValueProjection::<DimensionType>::from_projected(input_extent.clone())),
            ("divisor", ValueProjection::<DimensionType>::from_projected(effective_axis_size)),
        ],
    )?;
    Ok(ValueProjection::<DimensionType>::from_projected(input_extent.div(&divisor)?))
}

/// Requires an input axis extent to equal the effective participant count used by an untiled collective.
pub(super) fn require_collective_axis_extent<V>(
    context: &V::DispatchDomain,
    input_extent: &V,
    effective_axis_size: usize,
) -> Result<(), ProgramError>
where
    V: Value<Type = ArrayIrType> + Assert + ValueProjection<DimensionType>,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    V::DispatchDomain: DimensionConstant,
    <V as ValueProjection<DimensionType>>::Projected: Value<Type = DimensionType> + Compare<V>,
{
    let input_extent = <V as ValueProjection<DimensionType>>::into_projected(input_extent.clone())?;
    let effective_axis_size = collective_extent_constant(context, effective_axis_size)?;
    let effective_axis_size = <V as ValueProjection<DimensionType>>::into_projected(effective_axis_size)?;
    input_extent.compare(&effective_axis_size, ComparisonDirection::Equal)?.assert(
        "collective axis extent must match the participant count",
        &[
            ("extent", ValueProjection::<DimensionType>::from_projected(input_extent)),
            ("participants", ValueProjection::<DimensionType>::from_projected(effective_axis_size)),
        ],
    )
}

/// Requires an input axis extent to be exactly divisible by the effective participant count.
pub(super) fn require_collective_axis_divisible<V>(
    context: &V::DispatchDomain,
    input_extent: &V,
    effective_axis_size: usize,
) -> Result<(), ProgramError>
where
    V: Value<Type = ArrayIrType> + Assert + ValueProjection<DimensionType>,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    V::DispatchDomain: DimensionConstant,
    <V as ValueProjection<DimensionType>>::Projected: Value<Type = DimensionType> + Compare<V> + DimensionMax + Rem,
{
    let input_extent = <V as ValueProjection<DimensionType>>::into_projected(input_extent.clone())?;
    let effective_axis_size = collective_extent_constant(context, effective_axis_size)?;
    let effective_axis_size = <V as ValueProjection<DimensionType>>::into_projected(effective_axis_size)?;
    let zero = ValueProjection::<DimensionType>::into_projected(context.dimension_constant(0)?)?;
    let one = ValueProjection::<DimensionType>::into_projected(context.dimension_constant(1)?)?;
    effective_axis_size.compare(&zero, ComparisonDirection::GreaterThan)?.assert(
        "collective divisor must be positive",
        &[("divisor", ValueProjection::<DimensionType>::from_projected(effective_axis_size.clone()))],
    )?;
    input_extent
        .rem(&effective_axis_size.dimension_max(&one)?)?
        .compare(&zero, ComparisonDirection::Equal)?
        .assert(
            "collective extent must be divisible by the participant count",
            &[
                ("extent", ValueProjection::<DimensionType>::from_projected(input_extent)),
                ("divisor", ValueProjection::<DimensionType>::from_projected(effective_axis_size)),
            ],
        )
}

/// Applies the mixed array IR JVP shared by shape-changing collectives whose transpose is another collective.
/// Explicit output extents and the exact input shape become ordinary residuals of one linear call.
pub(super) fn jvp_shape_changing_collective_with_adjoint<C, Forward, Adjoint, P: DifferentiationPolicy<C>>(
    operation: &Forward,
    adjoint: Adjoint,
    context: &DifferentiationContext<C, P>,
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
    let Some((array, _)) = inputs.split_first() else {
        return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
    };
    let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
    let primal = context.primal().bind(operation.clone(), Vec::new(), primal_inputs.as_slice())?.remove(0);
    let tangent = match array.tangent() {
        MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
        MaybeZero::Value(array_tangent) => {
            let tangent_inputs = context.dual_primal_to_tangent(inputs)?;
            let (array, output_extents) = tangent_inputs.split_first().unwrap();
            let context = context.tangent();
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

/// Splits a mixed collective's inputs into its validated array operand and unchecked explicit result extents.
pub(super) fn explicit_collective_inputs<'a, V: Value<Type = ArrayIrType>>(
    inputs: &'a [ArrayIrBatch<V>],
) -> Result<(&'a ArrayIrBatch<V>, &'a [ArrayIrBatch<V>]), BatchingError> {
    let Some((array, output_extents)) = inputs.split_first() else {
        return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
    };
    <&ArrayType>::try_from(&array.unbatched_type())?;
    Ok((array, output_extents))
}

/// Validates that every explicit result extent of a mixed collective is replicated.
pub(super) fn validate_explicit_collective_output_extents<V: Value<Type = ArrayIrType>>(
    output_extents: &[ArrayIrBatch<V>],
) -> Result<(), BatchingError> {
    for output_extent in output_extents {
        output_extent.validate_replicated_dimension()?;
    }
    Ok(())
}

/// Binds a mixed collective over a non-matching named axis after lifting the mapped axis into its explicit result
/// extents. Replicated arrays require no lifting and remain replicated.
pub(super) fn forward_explicit_collective<C, O>(
    operation: O,
    context: &BatchingContext<C, ArrayIrBatchingPolicy>,
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

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, DataType, DimensionBounds, DimensionVariable,
    };
    use crate::batching::BatchableOperation;
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{MemberDifferentiableOperation, transpose_mixed_operation};
    use crate::macros::check_operation_partial_evaluation;
    use crate::operations::collectives::CollectiveOptions;
    use crate::operations::collectives::all_gather::{
        AllGatherOperation, AllGatherOutputVariance, infer_explicit_all_gather_output_types,
    };
    use crate::operations::collectives::all_to_all::{AllToAllOperation, infer_explicit_all_to_all_output_types};
    use crate::operations::collectives::parallel_sum_scatter::{
        ParallelSumScatterOperation, infer_explicit_parallel_sum_scatter_output_types,
    };
    use crate::operations::collectives::tests::f32_vector;
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::{EmptyRegionDriver, ProgramBuilder};
    use crate::tracing::TracingContext;

    use super::*;

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
    fn test_explicit_shape_changing_collective_member_transforms() -> Result<(), ProgramError> {
        type Context = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        // A live tangent through a dynamically shaped mixed collective stages one residual-aware linear call directly
        // through the payload's member JVP rule.
        let variable = DimensionVariable::new("items", DimensionBounds::new(1, Some(9))?);
        let dimension_type = DimensionType::from(variable.clone());
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
            &DifferentiationContext::fused(context.clone()),
            &EmptyRegionDriver,
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
        let context = Context::new();
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]));
        let output_cotangent = context.input(array_type.clone().into());
        let extent_type = DimensionValue::constant(3)?.r#type().into_owned();
        let mut context = crate::differentiation::TranspositionContext::new(context);
        let inputs = [PartialValue::Unknown(array_type.into()), PartialValue::Unknown(extent_type.into())];
        let accumulators = context.cotangent_accumulators(&inputs, &[])?;
        transpose_mixed_operation(
            &mut context,
            &ParallelSumScatterOperation::new("x".to_string(), 1, 0, CollectiveOptions::tiled()),
            &inputs,
            &[MaybeZero::Value(output_cotangent)],
            &accumulators,
        )?;
        let cotangents = context.take_cotangents(&accumulators)?;
        assert!(matches!(cotangents.as_slice(), [MaybeZero::Value(_), MaybeZero::Zero(_)]));
        assert!(matches!(
            context.builder().borrow().instructions()[0].operation(),
            ArrayIrOperation::Array(ArrayOperation::AllGather(_)),
        ));

        Ok(())
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
                    ArrayIrType::Dimension(DimensionType::from(concat_result.clone())),
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
                    ArrayIrType::Dimension(DimensionType::from(split_result.clone())),
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
                    ArrayIrType::Dimension(DimensionType::from(split_result.clone())),
                    ArrayIrType::Dimension(DimensionType::from(concat_result.clone())),
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
                    ArrayIrType::Dimension(DimensionType::from(input_axis)),
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
        let mapped_matrix =
            || ArrayBatch::new(Array::matrix(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap(), Some(0)).unwrap();

        let gathered = AllGatherOperation::new(
            "x".to_string(),
            2,
            1,
            CollectiveOptions::default(),
            AllGatherOutputVariance::Varying,
        )
        .batch(&context, &EmptyRegionDriver, &[mapped_matrix()])
        .unwrap()
        .into_parts()
        .0;
        assert_eq!(gathered[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(gathered[0].value(), &Array::matrix(2, 2, vec![1.0_f32, 3.0, 2.0, 4.0]).unwrap(),);

        let scattered = ParallelSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::default())
            .batch(&context, &EmptyRegionDriver, &[mapped_matrix()])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(scattered[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(scattered[0].value(), &Array::vector(vec![4.0_f32, 6.0]).unwrap());
        assert_eq!(scattered[0].unbatched_type(), ArrayType::scalar(DataType::F32));

        let exchanged = AllToAllOperation::new("x".to_string(), 2, 0, 0, CollectiveOptions::default())
            .batch(&context, &EmptyRegionDriver, &[mapped_matrix()])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(exchanged[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(exchanged[0].value(), &Array::matrix(2, 2, vec![1.0_f32, 3.0, 2.0, 4.0]).unwrap(),);
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
            program.transpose_with_respect_to(&[0], &[]).unwrap().transpose_with_respect_to(&[0], &[]).unwrap();
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
            program.transpose_with_respect_to(&[0], &[]).unwrap().transpose_with_respect_to(&[0], &[]).unwrap();
        assert!(matches!(transposed_twice.instructions()[0].operation(), ArrayOperation::AllToAll(_)));
        assert_eq!(transposed_twice.input_types(), program.input_types());
        assert_eq!(transposed_twice.output_types(), program.output_types());
    }

    #[test]
    fn test_array_ir_explicit_collective_eager_contracts() {
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap());
        let extent = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());

        assert_eq!(
            context.bind(
                AllGatherOperation::new(
                    "x".to_string(),
                    1,
                    0,
                    CollectiveOptions::tiled(),
                    AllGatherOutputVariance::Varying
                ),
                Vec::new(),
                &[input.clone(), extent.clone()],
            ),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            context.bind(
                ParallelSumScatterOperation::new("x".to_string(), 1, 0, CollectiveOptions::tiled()),
                Vec::new(),
                &[input.clone(), extent.clone()],
            ),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            context.bind(
                AllToAllOperation::new("x".to_string(), 1, 0, 0, CollectiveOptions::tiled()),
                Vec::new(),
                &[input.clone(), extent.clone()],
            ),
            Ok(vec![input.clone()]),
        );
        assert_eq!(
            context.bind(
                AllToAllOperation::new("x".to_string(), 1, 0, 1, CollectiveOptions::tiled()),
                Vec::new(),
                &[
                    ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],).unwrap()),
                    ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
                    ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()),
                ],
            ),
            Ok(vec![ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],).unwrap())]),
        );

        assert_eq!(
            context
                .bind(
                    AllGatherOperation::new(
                        "x".to_string(),
                        1,
                        0,
                        CollectiveOptions::tiled(),
                        AllGatherOutputVariance::Varying
                    ),
                    Vec::new(),
                    &[input.clone(), ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap()),],
                )
                .unwrap_err()
                .to_string(),
            "`all_gather` output axis 0 extent must equal observed result extent 3 but got 4",
        );
        assert_eq!(
            context
                .bind(
                    AllGatherOperation::new(
                        "x".to_string(),
                        2,
                        0,
                        CollectiveOptions::tiled(),
                        AllGatherOutputVariance::Varying
                    ),
                    Vec::new(),
                    &[input.clone(), ArrayIrValue::Dimension(DimensionValue::constant(6).unwrap()),],
                )
                .unwrap_err(),
            ProgramError::UnsupportedOperation {
                message: "cannot interpret `all_gather` over axis `x` of size 2 without an enclosing binder"
                    .to_string(),
            },
        );
        assert_eq!(
            context
                .bind(
                    ParallelSumScatterOperation::new("empty".to_string(), 0, 0, CollectiveOptions::tiled()),
                    Vec::new(),
                    &[input.clone(), extent.clone()],
                )
                .unwrap_err(),
            ProgramError::Type(TypeError::invalid("`parallel_sum_scatter` axis size must be greater than zero")),
        );

        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = AllGatherOperation::new("x".to_string(), 1, 0, CollectiveOptions::tiled(), AllGatherOutputVariance::Varying),
            cases = [
                {
                    inputs = [(@known, input.clone()), (@known, extent.clone())],
                    outputs = [(@known, input.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = input.r#type().into_owned(), replay = input.clone())),
                        (@known, extent.clone()),
                    ],
                    outputs = [(@residual, input.clone())],
                    residual_instructions = 1,
                },
            ],
        );

        let variable = DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap());
        let dimension_type = DimensionType::from(variable.clone());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = builder.add_input(array_type.into());
        let result_extent = builder.add_input(dimension_type.clone().into());
        let output = builder
            .add_instruction(
                AllGatherOperation::new(
                    "x".to_string(),
                    1,
                    0,
                    CollectiveOptions::tiled(),
                    AllGatherOutputVariance::Varying,
                ),
                Vec::new(),
                vec![array, result_extent],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let primal = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap());
        let tangent = ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]).unwrap());
        let result_extent = ArrayIrValue::Dimension(DimensionValue::new(dimension_type.clone(), 3).unwrap());
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.interpret(vec![primal.clone(), result_extent.clone(), tangent.clone(),]),
            Ok(vec![primal, tangent]),
        );
        assert!(
            jvp.instructions()
                .iter()
                .any(|instruction| { matches!(instruction.operation(), ArrayIrOperation::LinearCall(_)) })
        );
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap()), result_extent])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        let cotangent = ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]).unwrap());
        let mut pullback_inputs = vec![cotangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![cotangent]));
        let zero_extent = ArrayIrValue::Dimension(DimensionValue::new(dimension_type, 0).unwrap());
        let zero_array = || {
            ArrayIrValue::Array(
                Array::from_elements::<f32>(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(0)])), &[])
                    .unwrap(),
            )
        };
        let mut primal_outputs = linearization.primal().interpret(vec![zero_array(), zero_extent]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let zero_cotangent = zero_array();
        let mut pullback_inputs = vec![zero_cotangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![zero_cotangent]));
        assert!(matches!(
            program.transpose_with_respect_to(&[0], &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "direct `all_gather` transposition with runtime-dependent type metadata requires \
                    linearization so that the relevant primal information can be retained as residuals",
        ));
    }

    #[test]
    fn test_array_ir_shape_changing_collective_linearization() {
        let variable = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let dimension_type = DimensionType::from(variable.clone());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = builder.add_input(array_type.into());
        let result_extent = builder.add_input(dimension_type.clone().into());
        let output = builder
            .add_instruction(
                ParallelSumScatterOperation::new("x".to_string(), 1, 0, CollectiveOptions::tiled()),
                Vec::new(),
                vec![array, result_extent],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        assert!(linearization.tangent().to_string().contains("linear_call [residual_count=1]"));
        let input = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap());
        let extent = ArrayIrValue::Dimension(DimensionValue::new(dimension_type, 3).unwrap());
        let mut primal_outputs = linearization.primal().interpret(vec![input, extent]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let cotangent = ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]).unwrap());
        let mut pullback_inputs = vec![cotangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![cotangent]));

        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = builder.add_input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])).into());
        let extent = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()));
        let output = builder
            .add_instruction(
                AllToAllOperation::new("x".to_string(), 1, 0, 0, CollectiveOptions::tiled()),
                Vec::new(),
                vec![array, extent],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();
        assert!(linearization.tangent().to_string().contains("linear_call"));
        let input = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap());
        let mut primal_outputs = linearization.primal().interpret(vec![input]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let cotangent = ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]).unwrap());
        let mut pullback_inputs = vec![cotangent.clone()];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![cotangent]));
    }
}
