use std::fmt::Display;

use crate::arrays::batching::{RaggedAxis, align_array_batch, dimension_constant};
use crate::arrays::{
    ArrayBatch, ArrayBatching, ArrayBatchingPolicy, ArrayIrBatch, ArrayIrBatching, ArrayIrType, ArrayType, Dimension,
    DimensionType, DimensionValue, LinearResiduals, Shape, Sharding, ShardingDimension,
};
use crate::axes::Axis;
use crate::batching::{BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    BroadcastDerivativeAlignment, DifferentiableOperation, DifferentiableType, DifferentiationDriver,
    DifferentiationDual, DifferentiationError, LinearCallOperation, TransposableOperation, TranspositionContext,
    TranspositionDriver, transpose_projected_operation,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation, impl_reference_free_dischargeable_operation};
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::constants::zero::ZeroOperation;
use crate::operations::constants::zero_like::ZeroLikeOperation;
use crate::operations::dimensions::dimension_size::{DimensionSize, DimensionSizeOperation};
use crate::operations::manipulation::conversion::ConvertElementTypeOperation;
use crate::operations::manipulation::reshaping::{
    DynamicReshapeOperation, ReshapeOperation, lift_output_sharding_for_leading_batch_axis,
};
use crate::operations::manipulation::transposition::{Transpose, TransposeOperation};
use crate::operations::math::add::AddOperation;
use crate::operations::math::reduce::{Reduce, ReduceOperation, ReductionKind};
use crate::operations::sharding::ReshardOperation;
use crate::partial::{
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationValue, PartialValue,
    PartiallyEvaluatableOperation,
};
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, OperationProjection, ProgramError, RegionInterface, Type, TypeError,
    TypeIdentityPosition, TypeIdentityRenaming, Typed, Value, ValueProjection,
};
use crate::tracing::{NestedTracingContext, Tracer, TracingContext};

// TODO(eaplatanios): Review this module.

/// Canonical operation name shared by [`BroadcastOperation`] and [`DynamicBroadcastOperation`].
pub const BROADCAST_OPERATION_NAME: &str = "broadcast";

/// Mixed [`Operation`] that broadcasts one array using one explicit first-class dimension operand per output axis.
///
/// Operand zero is the array. Every remaining operand describes the corresponding output-axis extent, in order.
/// Exact dimension types produce static axes while non-exact dimension types retain their variables as dynamic axes.
/// `output_axes[input_axis]` names the output axis to which that input axis maps.
/// Refer to [`DynamicBroadcast`] for the underlying array broadcasting semantics.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DynamicBroadcastOperation {
    /// Vector that contains, for each input axis, the output axis to which it maps.
    output_axes: Vec<usize>,

    /// Optional requested output [`Sharding`].
    output_sharding: Option<Sharding>,
}

impl DynamicBroadcastOperation {
    /// Creates a broadcast with the supplied input-to-output axis mapping.
    #[inline]
    pub fn new(output_axes: Vec<usize>) -> Self {
        Self { output_axes, output_sharding: None }
    }

    /// Returns this operation with the requested output `sharding`.
    #[inline]
    pub fn with_output_sharding(mut self, sharding: impl Into<Option<Sharding>>) -> Self {
        self.output_sharding = sharding.into();
        self
    }

    /// Returns the output axes. The resulting slice contains, for each input axis, the output axis to which it maps.
    #[inline]
    pub fn output_axes(&self) -> &[usize] {
        self.output_axes.as_slice()
    }

    /// Returns the requested output sharding, if any.
    #[inline]
    pub fn output_sharding(&self) -> Option<&Sharding> {
        self.output_sharding.as_ref()
    }
}

impl Display for DynamicBroadcastOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for DynamicBroadcastOperation {
    type Type = ArrayIrType;

    #[inline]
    fn name(&self) -> &'static str {
        BROADCAST_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("region", region_interfaces, 0, TypeError);
        let Some((input_type, output_extent_types)) = input_types.split_first() else {
            return Err(TypeError::invalid(format!(
                "`{BROADCAST_OPERATION_NAME}` expects an array followed by its output extents"
            )));
        };
        let input_type = <&ArrayType>::try_from(input_type)?;
        let output_shape = Shape::new(ArrayIrType::extents(output_extent_types)?);
        Ok(vec![infer_explicit_broadcast_output_type(input_type, output_shape, self)?.into()])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, BROADCAST_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("output_axes", format_args!("{:?}", self.output_axes))?;
            if let Some(output_sharding) = &self.output_sharding {
                operation.field("output_sharding", output_sharding)?;
            }
            Ok(())
        })
    }
}

impl<C> InterpretableOperation<C> for DynamicBroadcastOperation
where
    C: Domain<Type = ArrayIrType>,
    C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType> + Broadcast>
        + ValueProjection<DimensionType, Projected = DimensionValue>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        let Some((input, output_extents)) = inputs.split_first() else {
            return Err(TypeError::invalid(format!(
                "`{BROADCAST_OPERATION_NAME}` expects an array followed by its output extents"
            ))
            .into());
        };
        let input = <C::Value as ValueProjection<ArrayType>>::into_projected(input.clone())?;
        let output_shape = Shape::new(
            output_extents
                .iter()
                .cloned()
                .map(<C::Value as ValueProjection<DimensionType>>::into_projected)
                .map(|result| result.map(|extent| Dimension::Static(extent.extent())))
                .collect::<Result<Vec<_>, _>>()?,
        );
        let output_type = infer_explicit_broadcast_output_type(input.r#type().as_ref(), output_shape, self)?;
        Ok(vec![<C::Value as ValueProjection<ArrayType>>::from_projected(
            input.broadcast(output_type, self.output_axes())?,
        )])
    }
}

impl<C: Context<Type = ArrayIrType, Operation: From<DynamicBroadcastOperation>>> PartiallyEvaluatableOperation<C>
    for DynamicBroadcastOperation
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        if self.output_sharding().is_none()
            && driver.region_count() == 0
            && let Some(input) = inputs.first()
            && let Ok(input_type) = <&ArrayType>::try_from(input.r#type().as_ref())
            && input_type.static_shape().is_some()
            && self.output_axes().iter().copied().eq(0..input_type.rank())
            && self
                .infer_output_types(&inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(), &[])?
                == vec![input.r#type().into_owned()]
        {
            // A static identity broadcast cannot observe its exact dimension operands. Preserve the input directly
            // so an unknown array does not leave a redundant broadcast in the residual program.
            return Ok(vec![input.clone()]);
        }
        context.fold_or_residualize(self.clone(), driver.regions().map(|region| region.to_program()).collect(), inputs)
    }
}

impl_reference_free_dischargeable_operation!(DynamicBroadcastOperation);

/// Batching rule for [`DynamicBroadcastOperation`]. A mapped input is canonicalized to a leading batch axis, which is
/// represented in both the lifted output extents and the input-to-output axis mapping. A mapped output extent uses its
/// declared finite bound as physical packed storage and records its per-item extent vector as transform-owned ragged
/// metadata; replicated extents retain their existing first-class representation.
impl<C> BatchableOperation<C, ArrayIrBatching> for DynamicBroadcastOperation
where
    C: Context<Type = ArrayIrType>,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>,
    C::Operation: From<DynamicBroadcastOperation>
        + From<ConstantOperation<DimensionValue>>
        + From<DimensionSizeOperation>
        + OperationProjection<ArrayType>,
    <C::Operation as OperationProjection<ArrayType>>::Projected: From<TransposeOperation>,
{
    fn batch<D: BatchingDriver<C, ArrayIrBatching>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatching>,
        _driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayIrBatching>, BatchingError> {
        let Some((input, output_extents)) = inputs.split_first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };
        <&ArrayType>::try_from(&input.unbatched_type())?;
        let ragged_extents = output_extents
            .iter()
            .enumerate()
            .filter_map(|(axis, extent)| extent.mapped_dimension_extents().map(|extents| (axis, extent, extents)))
            .collect::<Vec<_>>();
        for extent in output_extents {
            if extent.mapped_dimension_extents().is_none() {
                extent.validate_replicated_dimension()?;
            }
        }

        if input.batch_axis().is_replicated() && ragged_extents.is_empty() {
            return Ok(context
                .parent()
                .bind(self.clone(), Vec::new(), &inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>())?
                .into_iter()
                .map(ArrayIrBatch::replicated)
                .collect::<Vec<_>>()
                .into());
        }

        let moved_input = align_array_batch(context, input.clone(), Axis::from(0))?;

        let mut lifted_output_axes = Vec::with_capacity(self.output_axes().len() + 1);
        lifted_output_axes.push(0);
        lifted_output_axes.extend(self.output_axes().iter().map(|axis| axis + 1));
        let mut operation = Self::new(lifted_output_axes);
        if let Some(output_sharding) = self.output_sharding() {
            operation = operation.with_output_sharding(lift_output_sharding_for_leading_batch_axis(
                output_sharding,
                context.axis_sharding().clone(),
            )?);
        }

        let mut lifted_inputs = Vec::with_capacity(inputs.len() + 1);
        lifted_inputs.push(moved_input.value().clone());
        lifted_inputs.push(context.axis_extent().clone());
        for extent in output_extents {
            if extent.mapped_dimension_extents().is_some() {
                let unbatched_type = extent.unbatched_type();
                let extent_type = <&DimensionType>::try_from(&unbatched_type)?;
                let physical_extent =
                    extent_type.bounds().upper().and_then(|upper| upper.checked_sub(1)).ok_or_else(|| {
                        BatchingError::InvalidBatchMetadata {
                            message: format!(
                                "ragged broadcast dimension {} requires a finite, nonempty declared upper bound",
                                extent_type.variable(),
                            ),
                        }
                    })?;
                lifted_inputs.push(dimension_constant(context.parent(), physical_extent)?);
            } else {
                lifted_inputs.push(extent.value().clone());
            }
        }
        let mut outputs = context.parent().bind(operation, Vec::new(), lifted_inputs.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);

        let mut ragged_axes = moved_input
            .ragged_axes()
            .iter()
            .cloned()
            .map(|ragged_axis| {
                let axis = if ragged_axis.axis() == 0 { 0 } else { self.output_axes()[ragged_axis.axis() - 1] + 1 };
                let extent_axes = ragged_axis
                    .extent_axes()
                    .iter()
                    .map(|axis| if *axis == 0 { 0 } else { self.output_axes()[*axis - 1] + 1 })
                    .collect();
                RaggedAxis::new(axis, ragged_axis.extents().clone(), ragged_axis.dimension().clone(), extent_axes)
            })
            .collect::<Vec<_>>();
        ragged_axes.extend(
            ragged_extents
                .into_iter()
                .map(|(axis, extent, extents)| -> Result<_, BatchingError> {
                    let unbatched_type = extent.unbatched_type();
                    let extent_type = <&DimensionType>::try_from(&unbatched_type)?;
                    Ok(RaggedAxis::new(axis + 1, extents.clone(), extent_type.variable().clone(), vec![0]))
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
        let output = ArrayIrBatch::new(outputs.remove(0), BatchAxis::from_position(0))?;
        Ok(vec![output.with_ragged_axes(ragged_axes)?].into())
    }
}

/// [`Operation`] that performs general N-dimensional broadcasting over the homogeneous array language.
///
/// This is the member-family broadcast primitive: complete output geometry is carried by the [`ArrayType`] metadata
/// stored in the operation payload, so the operation has exactly one operand and no explicit extent edges. It and
/// [`ReshapeOperation`] form the homogeneous baseline that [`ProjectedContext`](crate::contexts::ProjectedContext)
/// serves, which is why transform rules for mixed operations can delegate to them once operand geometry is resolved.
/// Refer to the documentation of [`Broadcast`] for the underlying resolved-geometry contract.
///
/// Programs that need first-class dynamic extents stage [`DynamicBroadcastOperation`] instead, which takes one
/// explicit first-class dimension operand per output axis.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BroadcastOperation {
    /// Output [`ArrayType`].
    output_type: ArrayType,

    /// Vector that contains, for each input axis `i`, the output axis that it maps to.
    output_axes: Vec<usize>,
}

impl BroadcastOperation {
    /// Creates a new [`BroadcastOperation`] with the supplied output type and output axes.
    #[inline]
    pub fn new(output_type: ArrayType, output_axes: Vec<usize>) -> Self {
        Self { output_type, output_axes }
    }

    /// Returns the output [`ArrayType`] of this [`BroadcastOperation`].
    #[inline]
    pub fn output_type(&self) -> &ArrayType {
        &self.output_type
    }

    /// Returns the output axes of this [`BroadcastOperation`]. The resulting slice contains, for each input axis,
    /// the output axis that it maps to.
    #[inline]
    pub fn output_axes(&self) -> &[usize] {
        self.output_axes.as_slice()
    }
}

impl Display for BroadcastOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for BroadcastOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        BROADCAST_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        match input_types[0].broadcast(self.output_type.clone(), self.output_axes.as_slice()) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError::invalid(error.to_string())),
        }
    }

    #[inline]
    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<ArrayType as crate::Type>::Identity>,
    ) -> Result<Self, TypeError> {
        Ok(Self { output_type: self.output_type.rename_identities(renaming)?, output_axes: self.output_axes.clone() })
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("output_type", &self.output_type)?;
            operation.field("output_axes", format_args!("{:?}", self.output_axes))
        })
    }
}

impl<C: Domain<Type = ArrayType, Value: Broadcast>> InterpretableOperation<C> for BroadcastOperation {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].broadcast(self.output_type.clone(), self.output_axes())?])
    }
}

impl<C: Context<Type = ArrayType, Operation: From<BroadcastOperation>>> PartiallyEvaluatableOperation<C>
    for BroadcastOperation
{
}

impl_differentiable_operation! {
    BroadcastOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType, Value: Broadcast, Operation: From<BroadcastOperation>>,
    {
        |operation, _context, _driver, inputs| {
            // Forward-mode differentiation rule for `BroadcastOperation`. Broadcasting is structural-linear, so
            // tangent follows the same axis mapping as the primal. A structural-zero input tangent remains structural
            // and acquires the primal output's tangent type.
            check_count!("input", inputs, 1, ProgramError);
            let primal =
                inputs[0].primal().broadcast(operation.output_type().clone(), operation.output_axes())?;
            let tangent_type = primal.r#type().tangent()?;
            let tangent = match inputs[0].tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(tangent_type),
                MaybeZero::Value(tangent) => {
                    MaybeZero::Value(tangent.broadcast(tangent_type, operation.output_axes())?)
                }
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType>
            + From<AddOperation<ArrayType>>
            + From<BroadcastOperation>
            + From<ConvertElementTypeOperation<ArrayType>>
            + From<ReduceOperation>
            + From<TransposeOperation>
            + From<ReshapeOperation>
            + From<ReshardOperation>
            + From<ZeroLikeOperation<ArrayType>>,
    {
        |operation, _context, _driver, inputs, outputs| {
            // Transposition rule for `BroadcastOperation`. The pullback of a broadcast is a sum-reduction over
            // output axis the input was replicated along (i.e., the axes of the target type that are not named in
            // `output_axes`, plus the mapped axes whose input extent is `1` stretched to a larger target extent).
            // After the reduction, the surviving axes are reordered into input-axis order when `output_axes` is not
            // monotonically increasing, and stretched unit axes are restored with a reshape so the cotangent matches
            // the input type exactly. Symbolic-zero cotangents propagate unchanged, and an input with no cotangent
            // space receives the structural zero of that space.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            let input_cotangent_type = inputs[0].r#type().cotangent()?;
            if input_cotangent_type.is_zero_space() {
                return Ok(vec![MaybeZero::Zero(input_cotangent_type)]);
            }
            let MaybeZero::Value(cotangent) = &outputs[0] else {
                return Ok(vec![MaybeZero::Zero(input_cotangent_type)]);
            };
            Ok(vec![MaybeZero::Value(
                cotangent.unalign_cotangent_along(&input_cotangent_type, operation.output_axes())?,
            )])
        }
    },
}

/// Forward-mode rule for mixed broadcast. The explicit output extents are ordinary non-differentiated shape values.
/// Exact input geometry replays the mixed broadcast directly; dynamic input geometry retains its exact extents so the
/// linear transpose can reduce, reorder, and rebind the input cotangent using first-class dimension residuals.
impl<C> DifferentiableOperation<C> for DynamicBroadcastOperation
where
    C: Context<Type = ArrayIrType>,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Operation: From<DynamicBroadcastOperation>
        + From<DimensionSizeOperation>
        + From<LinearCallOperation<ArrayIrType>>
        + From<DynamicReshapeOperation>
        + From<ZeroOperation<ArrayType>>
        + From<ConstantOperation<DimensionValue>>
        + OperationProjection<ArrayType, Projected: From<ReduceOperation> + From<TransposeOperation>>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let Some((array, output_extents)) = inputs.split_first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };
        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal = context.bind(self.clone(), Vec::new(), primal_inputs.as_slice())?.remove(0);
        let tangent = match array.tangent() {
            MaybeZero::Zero(_) => {
                let tangent_type = primal.r#type().tangent()?;
                if tangent_type.identities().any(|(position, _)| position == TypeIdentityPosition::Reference) {
                    let array_tangent_type = <&ArrayType>::try_from(&tangent_type)?.clone();
                    let dynamic_extents = array_tangent_type
                        .shape()
                        .dimensions()
                        .iter()
                        .zip(output_extents)
                        .filter_map(|(dimension, extent)| {
                            matches!(dimension, Dimension::Dynamic(_)).then(|| extent.primal().clone())
                        })
                        .collect::<Vec<_>>();
                    MaybeZero::Value(
                        context
                            .bind(ZeroOperation::new(array_tangent_type), Vec::new(), dynamic_extents.as_slice())?
                            .remove(0),
                    )
                } else {
                    MaybeZero::Zero(tangent_type)
                }
            }
            MaybeZero::Value(array_tangent) => {
                let input_cotangent_type = <&ArrayType>::try_from(array.primal().r#type().as_ref())?.cotangent()?;
                if input_cotangent_type
                    .shape()
                    .dimensions()
                    .iter()
                    .all(|dimension| matches!(dimension, Dimension::Static(_)))
                {
                    let mut tangent_inputs = Vec::with_capacity(inputs.len());
                    tangent_inputs.push(array_tangent.clone());
                    tangent_inputs.extend(output_extents.iter().map(|extent| extent.primal().clone()));
                    MaybeZero::Value(context.bind(self.clone(), Vec::new(), tangent_inputs.as_slice())?.remove(0))
                } else {
                    let mut residuals = LinearResiduals::new();
                    let output_extents =
                        residuals.retain_all(output_extents.iter().map(|extent| extent.primal().clone()));
                    let input_shape = residuals.retain_shape(context, array.primal())?;
                    let forward_operation = self.clone();
                    let forward_output_extents = output_extents.clone();
                    let transpose_output_axes = self.output_axes().to_vec();
                    let transpose_target_type = input_cotangent_type.clone();
                    let tangent = LinearCallOperation::stage(
                        context,
                        residuals.into_values(),
                        vec![array_tangent.clone()],
                        move |residuals, linear_inputs| {
                            let mut broadcast_inputs = Vec::with_capacity(1 + forward_output_extents.len());
                            broadcast_inputs.push(linear_inputs[0].clone());
                            broadcast_inputs
                                .extend(forward_output_extents.iter().map(|index| residuals[*index].clone()));
                            linear_inputs[0].dispatch_domain().bind(
                                forward_operation,
                                Vec::new(),
                                broadcast_inputs.as_slice(),
                            )
                        },
                        move |residuals, output_cotangents| {
                            let transpose_context = output_cotangents[0].dispatch_domain();
                            let mut contribution =
                                <Tracer<NestedTracingContext<C>> as ValueProjection<ArrayType>>::into_projected(
                                    output_cotangents[0].clone(),
                                )?;
                            let contribution_type = contribution.r#type().into_owned();
                            if transpose_output_axes.len() != transpose_target_type.rank()
                                || transpose_output_axes.iter().any(|axis| *axis >= contribution_type.rank())
                            {
                                return Err(TypeError::invalid(format!(
                                    "cannot unalign cotangent type {contribution_type} to input cotangent type \
                                     {transpose_target_type} using output axes {transpose_output_axes:?}",
                                ))
                                .into());
                            }
                            let mut kept_axes = Vec::with_capacity(transpose_target_type.rank());
                            for (target_axis, output_axis) in transpose_output_axes.iter().copied().enumerate() {
                                let target_dimension = transpose_target_type.dimension(target_axis);
                                let output_dimension = contribution_type.dimension(output_axis);
                                if target_dimension == output_dimension {
                                    kept_axes.push((target_axis, output_axis));
                                } else if target_dimension != Dimension::Static(1) {
                                    return Err(TypeError::invalid(format!(
                                        "cannot unalign cotangent axis {output_axis} of size {output_dimension} to \
                                         input axis {target_axis} of size {target_dimension}",
                                    ))
                                    .into());
                                }
                            }
                            let reduce_axes = (0..contribution_type.rank())
                                .filter(|axis| kept_axes.iter().all(|(_, output_axis)| output_axis != axis))
                                .collect::<Vec<_>>();
                            if !reduce_axes.is_empty() {
                                contribution = contribution.reduce(reduce_axes.as_slice(), ReductionKind::Sum);
                            }
                            let mut kept_axes_by_output = kept_axes.clone();
                            kept_axes_by_output.sort_by_key(|(_, output_axis)| *output_axis);
                            let permutation = kept_axes
                                .iter()
                                .map(|kept| kept_axes_by_output.iter().position(|candidate| candidate == kept).unwrap())
                                .collect::<Vec<_>>();
                            if permutation.iter().enumerate().any(|(axis, position)| axis != *position) {
                                contribution = Transpose::transpose(&contribution, permutation)?;
                            }
                            let contribution = contribution.into_value();

                            // Rebind exact input geometry through ordinary dimension operands. This prevents a
                            // metadata-only identity from standing in for runtime geometry at the linear boundary.
                            let contribution_type = contribution.r#type().into_owned();
                            let mut exact_inputs = Vec::with_capacity(transpose_target_type.rank() + 1);
                            exact_inputs.push(contribution);
                            exact_inputs.extend(input_shape.dimensions(&transpose_context, residuals)?);
                            let contribution_type = <&ArrayType>::try_from(&contribution_type)?;
                            if contribution_type.shape() == transpose_target_type.shape() {
                                transpose_context.bind(
                                    DynamicBroadcastOperation::new((0..transpose_target_type.rank()).collect())
                                        .with_output_sharding(transpose_target_type.sharding().cloned()),
                                    Vec::new(),
                                    exact_inputs.as_slice(),
                                )
                            } else {
                                transpose_context.bind(
                                    DynamicReshapeOperation::new()
                                        .with_output_sharding(transpose_target_type.sharding().cloned()),
                                    Vec::new(),
                                    exact_inputs.as_slice(),
                                )
                            }
                        },
                    )?
                    .remove(0);
                    MaybeZero::Value(tangent)
                }
            }
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

/// Direct transposition rule for mixed broadcast. Static input geometry delegates to the homogeneous array pullback,
/// while every explicit output extent receives a structural-zero cotangent. Dynamic input geometry requires
/// linearization so [`DifferentiableOperation::jvp`] can retain its exact extents as residuals.
impl<V, O> TransposableOperation<V, O> for DynamicBroadcastOperation
where
    V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    O: Operation<Type = ArrayIrType> + OperationProjection<ArrayType>,
    <O as OperationProjection<ArrayType>>::Projected: From<BroadcastOperation>
        + TransposableOperation<
            <V as ValueProjection<ArrayType>>::Projected,
            <O as OperationProjection<ArrayType>>::Projected,
        >,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<'_, V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        let Some((input, output_extents)) = inputs.split_first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };
        let input_cotangent_type = <&ArrayType>::try_from(input.r#type().as_ref())?.cotangent()?;
        if input_cotangent_type
            .shape()
            .dimensions()
            .iter()
            .any(|dimension| matches!(dimension, Dimension::Dynamic(_)))
        {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "direct transposition of a dynamic `{BROADCAST_OPERATION_NAME}` requires linearization so its \
                     input extents can be retained as residuals",
                ),
            }
            .into());
        }

        let output_type = match outputs {
            [MaybeZero::Zero(r#type)] => <&ArrayType>::try_from(r#type)?.clone(),
            [MaybeZero::Value(value)] => <&ArrayType>::try_from(value.r#type().as_ref())?.clone(),
            _ => return Err(ProgramError::InvalidOutputCount { expected: 1, actual: outputs.len() }.into()),
        };
        let operation = <O as OperationProjection<ArrayType>>::Projected::from(BroadcastOperation::new(
            output_type,
            self.output_axes().to_vec(),
        ));
        let mut cotangents = transpose_projected_operation(context, &operation, std::slice::from_ref(input), outputs)?;
        cotangents.extend(
            output_extents
                .iter()
                .map(|extent| Ok(MaybeZero::Zero(extent.r#type().cotangent()?)))
                .collect::<Result<Vec<_>, DifferentiationError>>()?,
        );
        Ok(cotangents)
    }
}

impl<C: Context<Type = ArrayType, Value: Broadcast>, P: ArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>>
    for BroadcastOperation
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        _context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatching<P>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        match inputs[0].batch_axis_position() {
            None => {
                // A replicated input has no mapped axis to lift, so the original broadcast remains replicated.
                let output_value = inputs[0].value().broadcast(self.output_type().clone(), self.output_axes())?;
                Ok(vec![ArrayBatch::replicated(output_value)].into())
            }
            Some(batch_axis) => {
                // Insert the mapped axis at the same physical output position and shift every existing broadcast-axis
                // mapping at or after that position around it.
                let axis_size = ArrayBatch::common_batch_size(inputs)?.expect("a mapped input pins the batch size");
                let mut output_type =
                    self.output_type().with_inserted_dimension(batch_axis, Dimension::Static(axis_size))?;
                let mut output_axes = self
                    .output_axes()
                    .iter()
                    .map(|&output_axis| if output_axis >= batch_axis { output_axis + 1 } else { output_axis })
                    .collect::<Vec<_>>();
                output_axes.insert(batch_axis, batch_axis);
                let axis_sharding = ArrayBatch::sharding_for_inputs(inputs)?;
                let output_sharding = self.output_type().sharding().cloned();
                let input_mesh = inputs.iter().find_map(|input| {
                    input.batch_axis_position()?;
                    input.r#type().sharding().map(|sharding| sharding.mesh().clone())
                });
                let output_sharding = match (output_sharding, input_mesh) {
                    (Some(sharding), _) => Some(sharding),
                    (None, Some(mesh)) if !matches!(&axis_sharding, ShardingDimension::Replicated) => {
                        Some(Sharding::replicated(mesh, self.output_type().rank()))
                    }
                    (None, None) => None,
                    (None, Some(_)) => None,
                };
                if let Some(sharding) = output_sharding {
                    output_type.sharding = Some(
                        sharding
                            .with_inserted_dimension(batch_axis, axis_sharding)
                            .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })?,
                    );
                }
                let output_value = inputs[0].value().broadcast(output_type.clone(), output_axes.as_slice())?;
                Ok(vec![ArrayBatch::new(output_value, BatchAxis::from_position(batch_axis))?].into())
            }
        }
    }
}

/// Represents the ability to broadcast an array to a fully resolved output geometry.
///
/// `value.broadcast(output_type, output_axes)` replicates `value` into an output whose complete geometry is already
/// known, carried by the [`ArrayType`] argument rather than by first-class extent operands. It is the layer that both
/// interpretation and the composite eager path delegate to once operand extents are concrete, mirroring the plain
/// [`Reshape`](crate::operations::manipulation::Reshape) capability, and it is what [`BroadcastOperation`] binds when
/// staged.
///
/// Callers that need first-class dynamic extents use [`DynamicBroadcast`] instead, whose output extents are explicit
/// Static Single Assignment (SSA) operands and which therefore stays shape-polymorphic.
pub trait Broadcast: Sized {
    /// Broadcasts `self` to `output_type` using `output_axes`.
    ///
    /// # Parameters
    ///
    ///   - `output_type`: [`ArrayType`] of the output array.
    ///   - `output_axes`: Slice that contains, for each axis `i` of the input, the output axis that it maps to. This
    ///     slice must have length equal to the input's rank and contain distinct values in `0..output_type.rank()`.
    fn broadcast(&self, output_type: ArrayType, output_axes: &[usize]) -> Result<Self, ProgramError>;
}

impl Broadcast for ArrayType {
    fn broadcast(&self, output_type: ArrayType, output_axes: &[usize]) -> Result<ArrayType, ProgramError> {
        if self.data_type() != output_type.data_type() {
            return Err(TypeError::invalid(format!(
                "broadcasting input data type {} does not match output data type {}",
                self.data_type(),
                output_type.data_type(),
            ))
            .into());
        }

        if self.memory() != output_type.memory() {
            return Err(TypeError::invalid(format!(
                "broadcasting input memory {} does not match output memory {}",
                self.memory(),
                output_type.memory(),
            ))
            .into());
        }

        let input_rank = self.rank();
        let output_rank = output_type.rank();
        if output_axes.len() != input_rank {
            return Err(TypeError::invalid(format!(
                "broadcasting output axes has length {} but input has rank {}",
                output_axes.len(),
                input_rank,
            ))
            .into());
        }

        let mut seen = vec![false; output_rank];
        for (input_axis, &output_axis) in output_axes.iter().enumerate() {
            if output_axis >= output_rank {
                return Err(TypeError::invalid(format!(
                    "broadcasting `output_axes[{}] = {}` is out of bounds for output rank {}",
                    input_axis, output_axis, output_rank,
                ))
                .into());
            }
            if seen[output_axis] {
                return Err(TypeError::invalid(format!(
                    "broadcasting output axes map two input axes to output axis {output_axis}",
                ))
                .into());
            }
            seen[output_axis] = true;

            let input_dimension = self.dimension(input_axis);
            let output_dimension = output_type.dimension(output_axis);
            match (input_dimension, output_dimension.clone()) {
                // Identical sizes always map through, including identical dynamic sizes.
                (input_dimension, output_dimension) if input_dimension == output_dimension => {}
                // A static size-1 input dimension is replicated to match any static output extent. Expanding it
                // into a dynamic output dimension is unsupported because the replication count is unknown.
                (Dimension::Static(1), Dimension::Static(_)) => {}
                (Dimension::Static(1), Dimension::Dynamic(_)) => {
                    return Err(TypeError::invalid(format!(
                        "broadcasting cannot expand input axis {} of size 1 into dynamic output size {}",
                        input_axis, output_dimension,
                    ))
                    .into());
                }
                (Dimension::Static(input_size), Dimension::Static(output_size)) => {
                    return Err(TypeError::invalid(format!(
                        "broadcasting input axis {} has size {}, which is neither {} nor 1",
                        input_axis, input_size, output_size,
                    ))
                    .into());
                }
                // All remaining combinations pair a dynamic size with a mismatched size on the other side.
                (input_dimension, output_dimension) => {
                    return Err(TypeError::invalid(format!(
                        "broadcasting input axis {input_axis} has size {input_dimension} but the output has size \
                            {output_dimension}; a dynamic dimension only broadcasts to an identical dynamic dimension",
                    ))
                    .into());
                }
            }
        }

        // Output axes that no input axis maps to replicate the input along that axis, which requires a known
        // replication count and is therefore unsupported for dynamic output dimensions.
        for (output_axis, mapped) in seen.iter().enumerate() {
            let output_dimension = output_type.dimension(output_axis);
            if !mapped && matches!(output_dimension, Dimension::Dynamic(_)) {
                return Err(TypeError::invalid(format!(
                    "broadcasting cannot replicate the input into unmapped dynamic output axis {} of size {}",
                    output_axis, output_dimension,
                ))
                .into());
            }
        }
        Ok(output_type)
    }
}

/// Infers the result of the canonical mixed broadcast from its explicit output extent types.
pub(crate) fn infer_explicit_broadcast_output_type(
    input: &ArrayType,
    output_shape: Shape,
    operation: &DynamicBroadcastOperation,
) -> Result<ArrayType, TypeError> {
    let output_rank = output_shape.rank();
    if operation.output_axes().len() != input.rank() {
        return Err(TypeError::invalid(format!(
            "broadcasting output axes has length {} but input has rank {}",
            operation.output_axes().len(),
            input.rank(),
        )));
    }

    let mut mapped_output_axes = vec![false; output_rank];
    for (input_axis, &output_axis) in operation.output_axes().iter().enumerate() {
        if output_axis >= output_rank {
            return Err(TypeError::invalid(format!(
                "broadcasting `output_axes[{}] = {}` is out of bounds for output rank {}",
                input_axis, output_axis, output_rank,
            )));
        }
        if mapped_output_axes[output_axis] {
            return Err(TypeError::invalid(format!(
                "broadcasting output axes map two input axes to output axis {output_axis}",
            )));
        }
        mapped_output_axes[output_axis] = true;

        let input_dimension = input.dimension(input_axis);
        let output_dimension = output_shape.dimensions()[output_axis].clone();
        match (input_dimension, output_dimension) {
            // Equal axes preserve their extent, including the identity of an equal dynamic extent.
            (input_dimension, output_dimension) if input_dimension == output_dimension => {}
            // The explicit output extent operand supplies the runtime replication count that the metadata-only
            // homogeneous operation lacks, so a unit input may expand to either a static or dynamic extent.
            (Dimension::Static(1), _) => {}
            (Dimension::Static(input_extent), Dimension::Static(output_extent)) => {
                return Err(TypeError::invalid(format!(
                    "broadcasting input axis {} has size {}, which is neither {} nor 1",
                    input_axis, input_extent, output_extent,
                )));
            }
            (input_dimension, output_dimension) => {
                return Err(TypeError::invalid(format!(
                    "broadcasting input axis {input_axis} has size {input_dimension} but the output has size \
                        {output_dimension}; a dynamic dimension only broadcasts to an identical dynamic dimension",
                )));
            }
        }
    }

    let output_sharding = match operation.output_sharding() {
        Some(sharding) => Some(sharding.clone()),
        None => input
            .sharding()
            .map(|sharding| sharding.with_broadcasted_dimensions(output_rank, operation.output_axes()))
            .transpose()
            .map_err(|error| TypeError::invalid(error.to_string()))?,
    };

    if input.shape() == &output_shape && operation.output_axes().iter().copied().eq(0..input.rank()) {
        return input.clone().with_sharding(output_sharding).map_err(|error| TypeError::invalid(error.to_string()));
    }

    ArrayType::new(input.data_type(), output_shape)
        .with_memory(input.memory())
        .with_sharding(output_sharding)
        .map_err(|error| TypeError::invalid(error.to_string()))
}

impl<V: Value<Type = ArrayType, DispatchDomain: Context<Type = ArrayType, Operation: From<BroadcastOperation>>>>
    Broadcast for V
{
    #[inline]
    fn broadcast(&self, output_type: ArrayType, output_axes: &[usize]) -> Result<Self, ProgramError> {
        let output_type = self.r#type().broadcast(output_type, output_axes)?;
        if self.r#type().as_ref() == &output_type && output_axes.iter().copied().eq(0..output_type.rank()) {
            return Ok(self.clone());
        }
        Ok(self
            .dispatch_domain()
            .bind(BroadcastOperation::new(output_type, output_axes.to_vec()), Vec::new(), std::slice::from_ref(self))?
            .remove(0))
    }
}

/// Broadcasts an array using one explicit first-class dimension value per output axis.
///
/// Exact dimension values describe static axes and computed dimension values describe dynamic axes. Both forms bind
/// the same [`DynamicBroadcastOperation`]; backend lowering chooses the appropriate static, bounded, or dynamic
/// representation from the inferred result type.
///
/// Exact host sizes can use [`DynamicBroadcast::dynamic_broadcast_to_sizes`]:
///
/// ```rust
/// use ryft_core::operations::manipulation::DynamicBroadcast;
/// use ryft_core::{Array, ArrayIrValue};
///
/// let input = ArrayIrValue::Array(Array::vector(vec![1.0, 2.0, 3.0]));
/// let output = input.dynamic_broadcast_to_sizes(&[2, 3]).unwrap();
/// assert_eq!(
///     output,
///     ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0])),
/// );
/// ```
///
/// Computed or input dimensions remain ordinary SSA operands:
///
/// ```rust
/// use ryft_core::operations::manipulation::DynamicBroadcast;
/// use ryft_core::arrays::{
///     ArrayIrType, ArrayType, DataType, DimensionBounds, DimensionType, DimensionVariable,
/// };
/// use ryft_core::{Array, ArrayIrOperation, ArrayIrValue, StagingContext, TracingContext, Typed};
///
/// type C = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
///
/// let context = C::new();
/// let scalar = context.input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
/// let extent = context.input(ArrayIrType::Dimension(DimensionType::new(DimensionVariable::new(
///     "extent",
///     DimensionBounds::new(1, Some(9)).unwrap(),
/// ))));
/// let output = scalar.dynamic_broadcast_to(&[extent]).unwrap();
/// assert_eq!(output.r#type().to_string(), "f32[extent]");
/// ```
pub trait DynamicBroadcast: Value<Type = ArrayIrType> + DimensionSize + Sized {
    /// Broadcasts `self` using an explicit input-to-output axis mapping and one first-class value per output extent.
    ///
    /// # Parameters
    ///
    ///   - `output_dimensions`: Ordered output extents, one per result axis.
    ///   - `output_axes`: Output axis corresponding to each input axis.
    fn dynamic_broadcast(&self, output_dimensions: &[Self], output_axes: &[usize]) -> Result<Self, ProgramError> {
        self.dynamic_broadcast_with_output_sharding(output_dimensions, output_axes, None)
    }

    /// Broadcasts `self` with an explicit requested output sharding.
    fn dynamic_broadcast_with_output_sharding(
        &self,
        output_dimensions: &[Self],
        output_axes: &[usize],
        output_sharding: Option<Sharding>,
    ) -> Result<Self, ProgramError>;

    /// Broadcasts `self` to `output_dimensions` using NumPy-style right alignment.
    fn dynamic_broadcast_to(&self, output_dimensions: &[Self]) -> Result<Self, ProgramError> {
        let r#type = self.r#type();
        let input_type = <&ArrayType>::try_from(r#type.as_ref())?;
        let offset = output_dimensions.len().checked_sub(input_type.rank()).ok_or_else(|| {
            TypeError::invalid(format!(
                "cannot broadcast rank-{} input to {} output dimensions",
                input_type.rank(),
                output_dimensions.len(),
            ))
        })?;
        let output_axes = (0..input_type.rank()).map(|axis| axis + offset).collect::<Vec<_>>();
        self.dynamic_broadcast(output_dimensions, output_axes.as_slice())
    }

    /// Broadcasts `self` by prepending the supplied first-class leading dimensions.
    fn dynamic_broadcast_leading(&self, leading_dimensions: &[Self]) -> Result<Self, ProgramError>
    where
        Self::DispatchDomain: Context<Type = ArrayIrType>,
        <Self::DispatchDomain as Domain>::Constant: From<DimensionValue>,
    {
        let r#type = self.r#type();
        let input_type = <&ArrayType>::try_from(r#type.as_ref())?;
        let mut output_dimensions = Vec::with_capacity(leading_dimensions.len() + input_type.rank());
        output_dimensions.extend_from_slice(leading_dimensions);
        for (axis, dimension) in input_type.shape().dimensions().iter().enumerate() {
            output_dimensions.push(match dimension {
                Dimension::Static(extent) => self.dispatch_domain().lift(DimensionValue::constant(*extent)?.into())?,
                Dimension::Dynamic(_) => self.dimension_size(axis)?,
            });
        }
        let output_axes = (0..input_type.rank()).map(|axis| axis + leading_dimensions.len()).collect::<Vec<_>>();
        self.dynamic_broadcast(output_dimensions.as_slice(), output_axes.as_slice())
    }

    /// Broadcasts `self` to an exact static shape.
    fn dynamic_broadcast_to_sizes(&self, output_sizes: &[usize]) -> Result<Self, ProgramError>
    where
        Self::DispatchDomain: Context<Type = ArrayIrType>,
        <Self::DispatchDomain as Domain>::Constant: From<DimensionValue>,
    {
        let output_dimensions = output_sizes
            .iter()
            .map(|extent| self.dispatch_domain().lift(DimensionValue::constant(*extent)?.into()))
            .collect::<Result<Vec<_>, _>>()?;
        self.dynamic_broadcast_to(output_dimensions.as_slice())
    }

    /// Broadcasts `self` by prepending exact static leading dimensions.
    fn dynamic_broadcast_leading_sizes(&self, leading_sizes: &[usize]) -> Result<Self, ProgramError>
    where
        Self::DispatchDomain: Context<Type = ArrayIrType>,
        <Self::DispatchDomain as Domain>::Constant: From<DimensionValue>,
    {
        let context = self.dispatch_domain();
        let leading_dimensions = leading_sizes
            .iter()
            .map(|extent| context.lift(DimensionValue::constant(*extent)?.into()))
            .collect::<Result<Vec<_>, _>>()?;
        self.dynamic_broadcast_leading(leading_dimensions.as_slice())
    }
}

impl<
    V: Value<
            Type = ArrayIrType,
            DispatchDomain: Context<
                Type = ArrayIrType,
                Operation: From<DynamicBroadcastOperation> + From<DimensionSizeOperation>,
            >,
        >,
> DynamicBroadcast for V
where
    <V::DispatchDomain as Domain>::Constant: From<DimensionValue>,
{
    #[inline]
    fn dynamic_broadcast_with_output_sharding(
        &self,
        output_dimensions: &[Self],
        output_axes: &[usize],
        output_sharding: Option<Sharding>,
    ) -> Result<Self, ProgramError> {
        let input_type = self.r#type();
        let input_type = <&ArrayType>::try_from(input_type.as_ref())?;
        let output_shape =
            Shape::new(ArrayIrType::extents(output_dimensions.iter().map(|dimension| dimension.r#type()))?);
        let operation = DynamicBroadcastOperation::new(output_axes.to_vec()).with_output_sharding(output_sharding);
        let output_type = infer_explicit_broadcast_output_type(input_type, output_shape, &operation)?;
        if input_type == &output_type && output_axes.iter().copied().eq(0..input_type.rank()) {
            return Ok(self.clone());
        }

        let mut inputs = Vec::with_capacity(output_dimensions.len() + 1);
        inputs.push(self.clone());
        inputs.extend_from_slice(output_dimensions);
        Ok(self.dispatch_domain().bind(operation, Vec::new(), inputs.as_slice())?.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, DataType, DimensionBounds, DimensionValue,
        DimensionVariable, Layout, LogicalMesh, Memory, MeshAxis, MeshAxisType, Sharding, ShardingDimension,
        StridedLayout,
    };
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{TransposableOperation, differentiate_at};
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::{EmptyRegionDriver, ProgramBuilder, ProgramError, Typed};
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_explicit_broadcast() {
        let operation = DynamicBroadcastOperation::new(vec![1]);
        assert_eq!(operation.name(), BROADCAST_OPERATION_NAME);
        assert_eq!(operation.to_string(), "broadcast [output_axes=[1]]");

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![4])))
            .with_memory(Memory::Host { pinned: true });
        let two = DimensionValue::constant(2).unwrap();
        let dynamic_extent =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap()));
        assert_eq!(
            operation.infer_output_types(
                &[input_type.into(), two.r#type().into_owned().into(), dynamic_extent.clone().into(),],
                &[],
            ),
            Ok(vec![
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Static(2), Dimension::Dynamic(dynamic_extent.variable().clone()),]),
                )
                .with_memory(Memory::Host { pinned: true })
                .into()
            ]),
        );

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![4])))
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap()
            .with_memory(Memory::Host { pinned: true });
        let three = DimensionValue::constant(3).unwrap();
        assert_eq!(
            operation.infer_output_types(
                &[input_type.into(), two.r#type().into_owned().into(), three.r#type().into_owned().into()],
                &[],
            ),
            Ok(vec![
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
                    .with_sharding(
                        Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],)
                            .unwrap(),
                    )
                    .unwrap()
                    .with_memory(Memory::Host { pinned: true })
                    .into(),
            ]),
        );

        assert_eq!(
            DynamicBroadcastOperation::new(vec![0, 0]).infer_output_types(
                &[
                    ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1), Dimension::Static(1)]),).into(),
                    two.r#type().into_owned().into(),
                ],
                &[],
            ),
            Err(TypeError::invalid("broadcasting output axes map two input axes to output axis 0")),
        );
        assert_eq!(
            operation.infer_output_types(&[two.r#type().into_owned().into()], &[]),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
        assert_eq!(
            operation.infer_output_types(
                &[ArrayType::scalar(DataType::F32).into(), ArrayType::scalar(DataType::I64).into()],
                &[],
            ),
            Err(TypeError::invalid("expected dimension type but got array type")),
        );
    }

    #[test]
    fn test_explicit_broadcast_batching_preserves_a_declared_dynamic_extent() {
        // A mapped array input canonicalizes to a leading batch axis, and the declared output extents cross the
        // batching rule untouched: the lifted broadcast consumes the transform's own batch extent followed by exactly
        // the original extent operands. A declared *dynamic* extent makes that forwarding observable, because a rule
        // that reconstructed output geometry from the operand type would have to stage its own `dimension_size` read.
        type TraceContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let trace = TraceContext::new();
        let columns = DimensionVariable::new("columns", DimensionBounds::new(1, Some(9)).unwrap());
        let declared_extent = trace.input(DimensionType::new(columns.clone()).into());
        let input = trace
            .input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(1)])).into());
        let declared_extent_id = declared_extent.atom_id().unwrap();
        let input_id = input.atom_id().unwrap();
        let axis_extent = trace.constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
        let axis_extent_id = axis_extent.atom_id().unwrap();
        let context = BatchingContext::<_, ArrayIrBatching>::new(trace.clone(), axis_extent);

        let [output] = DynamicBroadcastOperation::new(vec![0])
            .batch(
                &context,
                &EmptyRegionDriver,
                &[ArrayIrBatch::new(input, BatchAxis::new(0)).unwrap(), ArrayIrBatch::replicated(declared_extent)],
            )
            .unwrap()
            .into_parts()
            .0
            .try_into()
            .unwrap();
        assert_eq!(output.batch_axis(), BatchAxis::new(0));
        assert_eq!(
            output.value().r#type().as_ref(),
            &ArrayIrType::Array(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Static(2), Dimension::Dynamic(columns)]),
            )),
        );
        let output_id = output.into_value().atom_id().unwrap();

        // The mapped axis already sits at position zero, so no axis move is staged and the single lifted broadcast
        // forwards the declared extent operand by identity.
        let builder = trace.builder().borrow();
        let [instruction] = builder.instructions() else {
            panic!("expected exactly one lifted broadcast instruction");
        };
        assert_eq!(instruction.inputs(), &[input_id, axis_extent_id, declared_extent_id]);
        drop(builder);

        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output_id],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<columns ∈ [1, 9)>, %1:f32[2, 1] .
                let %2:dimension<2> = const 2
                    %3:f32[2, columns] = broadcast [output_axes=[0, 1]] %1 %2 %0
                in (%3)
            "}
            .trim_end(),
        );
        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()),
                ArrayIrValue::Array(Array::from_f64s(
                    ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(1)])),
                    vec![1.0, 2.0],
                )),
            ]),
            Ok(vec![ArrayIrValue::Array(Array::from_f64s(
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)])),
                vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            ))]),
        );
    }

    #[test]
    fn test_broadcast() {
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let operation = BroadcastOperation::new(output_type.clone(), vec![1]);

        // Operation identity and accessors.
        assert_eq!(operation.name(), BROADCAST_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "broadcast [output_type=f64[2, 3], output_axes=[1]]");
        assert_eq!(*operation.output_type(), output_type);
        assert_eq!(operation.output_axes(), &[1]);

        // Type inference validates the axis mapping and returns the target type.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [
                {
                    input_types = [input_type.clone()],
                    output_types = [output_type.clone()],
                },
                {
                    input_types = [],
                    error = "expected 1 input but got 0",
                },
                {
                    input_types = [ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]))],
                    error = "broadcasting input data type f32 does not match output data type f64",
                },
                {
                    input_types = [output_type.clone()],
                    error = "broadcasting output axes has length 1 but input has rank 2",
                },
                {
                    input_types = [ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]))],
                    error = "broadcasting input axis 0 has size 2, which is neither 3 nor 1",
                },
            ],
        );

        // Interpretation replicates the payload along the added axis.
        let input = Array::vector(vec![1.0, 2.0, 3.0]);
        let output = operation
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(*output[0].r#type(), output_type);
        assert_eq!(output[0].to_f64s(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

        // Invalid interpreter arity reports the exact program error.
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        // Program rendering uses the canonical operation name and includes the captured metadata.
        let mut builder = ProgramBuilder::<Array, BroadcastOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_output =
            builder.add_instruction(operation.clone(), Vec::new(), vec![program_input], None).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![program_output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[3] .
                let %1:f64[2, 3] = broadcast [output_type=f64[2, 3], output_axes=[1]] %0
                in (%1)
            "}
            .trim_end(),
        );

        // Check standard partial evaluation with known and residual operands.
        let input = Array::vector(vec![1.0, 2.0, 3.0]);
        let expected = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        check_operation_partial_evaluation!(
            backend = (Array, ArrayOperation<Array>),
            operation = operation.clone(),
            cases = [
                {
                    inputs = [(@known, input.clone())],
                    outputs = [(@known, expected.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [(@unknown(type = input.r#type().into_owned(), replay = input.clone()))],
                    outputs = [(@residual, expected)],
                    residual_instructions = 1,
                },
            ],
        );

        // Batching preserves replicated inputs and lifts mapped inputs through the explicit output-axis mapping.
        check_operation_batching!(
            @exact,
            operation = BroadcastOperation::new(
                ArrayType::new(DataType::F64, Shape::new(vec![3.into(), 4.into()])),
                vec![0],
            ),
            axis_size = 2,
            cases = [
                {
                    inputs = [(@replicated, Array::vector(vec![1.0, 2.0, 3.0]))],
                    outputs = [(@replicated, Array::matrix(
                        3,
                        4,
                        vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0],
                    ))],
                },
                {
                    inputs = [(@mapped(axis = 0), Array::matrix(
                        2,
                        3,
                        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    ))],
                    outputs = [(@mapped(axis = 0), Array::from_f64s(
                        ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into(), 4.into()])),
                        vec![
                            1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0,
                            4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0,
                        ],
                    ))],
                },
            ],
        );

        // Broadcasting is structural-linear meaning that its JVP broadcasts both the primal and its tangent.
        check_operation_differentiation!(
            @approx(step = 0.125, epsilon = 1e-9),
            operation = BroadcastOperation::new(
                ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 2.into()])),
                vec![1],
            ),
            cases = [{
                primals = [Array::vector(vec![1.0, 2.0])],
                tangents = [Array::vector(vec![3.0, 4.0])],
                primal_outputs = [Array::matrix(2, 2, vec![1.0, 2.0, 1.0, 2.0])],
                tangent_outputs = [Array::matrix(2, 2, vec![3.0, 4.0, 3.0, 4.0])],
                jvp = indoc! {"
                    lambda %0:f64[2], %1:f64[2] .
                    let %2:f64[2, 2] = broadcast [output_type=f64[2, 2], output_axes=[1]] %0
                        %3:f64[2, 2] = broadcast [output_type=f64[2, 2], output_axes=[1]] %1
                    in (%2, %3)
                "},
            }],
        );

        // The pullback sums the output cotangent over every newly replicated output axis.
        check_operation_transposition!(
            @exact,
            operation = operation,
            cases = [{
                inputs = [(@linear(type = ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![Dimension::Static(3)]),
                )))],
                output_cotangents = [Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])],
                input_cotangents = [Array::vector(vec![5.0, 7.0, 9.0])],
                pullback = indoc! {"
                    lambda %0:f64[2, 3] .
                    let %1:f64[3] = reduce_sum [axes=[0]] %0
                    in (%1)
                "},
            }],
        );
    }

    #[test]
    fn test_broadcast_with_sharding() {
        // Explicit mapped-axis sharding remains attached to the lifted batch dimension.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        let logical_output_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(4)]))
                .with_sharding(Sharding::replicated(mesh.clone(), 2))
                .unwrap();
        let expected_output_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(3), Dimension::Static(4)]),
        )
        .with_sharding(
            Sharding::new(
                mesh,
                vec![
                    ShardingDimension::sharded(["x"]),
                    ShardingDimension::replicated(),
                    ShardingDimension::replicated(),
                ],
            )
            .unwrap(),
        )
        .unwrap();

        check_operation_batching!(
            @exact,
            operation = BroadcastOperation::new(logical_output_type, vec![0]),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::from_f64s(
                    input_type,
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                ))],
                outputs = [(@mapped(axis = 0), Array::from_f64s(
                    expected_output_type,
                    vec![
                        1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0,
                        4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0,
                    ],
                ))],
            }],
        );

        // A mapped physical axis retains its mesh placement even when the logical operation result has no sharding.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        let expected_output_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(3), Dimension::Static(4)]),
        )
        .with_sharding(
            Sharding::new(
                mesh,
                vec![
                    ShardingDimension::sharded(["x"]),
                    ShardingDimension::replicated(),
                    ShardingDimension::replicated(),
                ],
            )
            .unwrap(),
        )
        .unwrap();
        check_operation_batching!(
            @exact,
            operation = BroadcastOperation::new(
                ArrayType::new(DataType::F64, Shape::new(vec![3.into(), 4.into()])),
                vec![0],
            ),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::from_f64s(
                    input_type,
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                ))],
                outputs = [(@mapped(axis = 0), Array::from_f64s(
                    expected_output_type,
                    vec![
                        1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0,
                        4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0,
                    ],
                ))],
            }],
        );

        // A non-monotonic axis mapping keeps an explicitly sharded mapped axis at the beginning, middle, and end of
        // the physical result, even though the logical output itself has no sharding annotation.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);
        for batch_axis in 0..3 {
            let mut input_dimensions = vec![Dimension::Static(2), Dimension::Static(3)];
            input_dimensions.insert(batch_axis, Dimension::Static(2));
            let mut input_sharding = vec![ShardingDimension::replicated(); 3];
            input_sharding[batch_axis] = ShardingDimension::sharded(["x"]);
            let input_type = ArrayType::new(DataType::F64, Shape::new(input_dimensions))
                .with_sharding(Sharding::new(mesh.clone(), input_sharding).unwrap())
                .unwrap();
            let input =
                ArrayBatch::new(Array::from_f64s(input_type, vec![0.0; 12]), BatchAxis::from_position(batch_axis))
                    .unwrap();

            let output = BroadcastOperation::new(
                ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)])),
                vec![1, 0],
            )
            .batch(&context, &EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts()
            .0
            .remove(0);
            let mut output_dimensions = vec![Dimension::Static(3), Dimension::Static(2)];
            output_dimensions.insert(batch_axis, Dimension::Static(2));
            let mut output_sharding = vec![ShardingDimension::replicated(); 3];
            output_sharding[batch_axis] = ShardingDimension::sharded(["x"]);
            assert_eq!(
                output.r#type().as_ref(),
                &ArrayType::new(DataType::F64, Shape::new(output_dimensions))
                    .with_sharding(Sharding::new(mesh.clone(), output_sharding).unwrap())
                    .unwrap(),
            );
            assert_eq!(output.batch_axis(), BatchAxis::from_position(batch_axis));
        }

        // Differentiation derives the tangent target from the primal output, preserving promoted tangent types.
        let primal_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(2)]));
        let tangent_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]));
        let primal_output_type =
            ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let tangent_output_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let (primal, tangent) = differentiate_at(Array::from_f64s(primal_type, vec![2.0, 4.0]))
            .jvp(Array::from_f64s(tangent_type, vec![1.0, 3.0]), |value| {
                value.broadcast(primal_output_type.clone(), &[1])
            })
            .unwrap();
        assert_eq!(primal.r#type().as_ref(), &primal_output_type);
        assert_eq!(tangent.r#type().as_ref(), &tangent_output_type);
        assert_eq!(tangent.to_f64s(), vec![1.0, 3.0, 1.0, 3.0]);

        // Input axis 0 (size 2) maps to output axis 2 and input axis 1 (size 3) maps to output axis 0, so the pullback
        // must sum over output axis 1 and swap the surviving axes back into input order.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let output_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(3), Dimension::Static(4), Dimension::Static(2)]),
        );
        check_operation_transposition!(
            @exact,
            operation = BroadcastOperation::new(output_type.clone(), vec![2, 0]),
            cases = [{
                inputs = [(@linear(type = input_type))],
                output_cotangents = [Array::from_f64s(
                    output_type,
                    (0..24).map(|value| value as f64).collect(),
                )],
                input_cotangents = [Array::matrix(2, 3, vec![12.0, 44.0, 76.0, 16.0, 48.0, 80.0])],
            }],
        );

        // Input axis 0 has extent 1 stretched to 2 in the target, so the pullback sums over it and restores the unit
        // axis with a reshape.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1), Dimension::Static(3)]));
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        check_operation_transposition!(
            @exact,
            operation = BroadcastOperation::new(output_type, vec![0, 1]),
            cases = [{
                inputs = [(@linear(type = input_type))],
                output_cotangents = [Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])],
                input_cotangents = [Array::matrix(1, 3, vec![5.0, 7.0, 9.0])],
            }],
        );

        // Symbolic-zero cotangents remain symbolic and acquire the input's promoted cotangent type.
        let input_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(3)]));
        let output_type =
            ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let operation = BroadcastOperation::new(output_type.clone(), vec![1]);
        let input_cotangent_type = input_type.cotangent().unwrap();
        let output_cotangent_type = output_type.cotangent().unwrap();

        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let contributions = operation
            .transpose(
                &mut TranspositionContext::new(context.clone()),
                &EmptyRegionDriver,
                &[PartialValue::Unknown(input_type)],
                &[MaybeZero::Zero(output_cotangent_type)],
            )
            .unwrap();
        assert_eq!(contributions.len(), 1);
        assert!(contributions[0].is_zero());
        assert_eq!(contributions[0].r#type().as_ref(), &input_cotangent_type);

        // Inputs with no cotangent space receive the structural zero of that space rather than being rejected.
        let input_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(3)]));
        let output_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let operation = BroadcastOperation::new(output_type.clone(), vec![1]);
        let contributions = operation
            .transpose(
                &mut TranspositionContext::new(context.clone()),
                &EmptyRegionDriver,
                &[PartialValue::Unknown(input_type.clone())],
                &[MaybeZero::Zero(output_type.cotangent().unwrap())],
            )
            .unwrap();
        assert_eq!(contributions.len(), 1);
        assert!(contributions[0].is_zero());
        assert_eq!(contributions[0].r#type().as_ref(), &input_type.cotangent().unwrap());
    }

    #[test]
    fn test_array_type_broadcast() {
        // Type-level broadcasting validates arbitrary mappings without consuming the input type.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        assert_eq!(input_type.broadcast(output_type.clone(), &[1]), Ok(output_type.clone()));
        assert_eq!(
            input_type.broadcast(output_type.clone().with_memory(Memory::Host { pinned: true }), &[1]),
            Err(ProgramError::Type(TypeError::invalid(
                "broadcasting input memory Device does not match output memory Host[Pinned]".to_string(),
            ))),
        );
        assert_eq!(
            input_type.broadcast(output_type.clone(), &[2]),
            Err(ProgramError::Type(TypeError::invalid(
                "broadcasting `output_axes[0] = 2` is out of bounds for output rank 2".to_string(),
            ))),
        );
        assert_eq!(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(3)]))
                .broadcast(output_type, &[1, 1]),
            Err(ProgramError::Type(TypeError::invalid(
                "broadcasting output axes map two input axes to output axis 1".to_string(),
            ))),
        );

        // A dynamic input dimension maps only to the identical dynamic dimension, while replicated output axes must
        // remain static because their replication counts need to be known.
        let batch = DimensionVariable::new("batch", DimensionBounds::unbounded());
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Static(3)]));
        let output_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(2), Dimension::Static(3)]),
        );
        assert_eq!(input_type.broadcast(output_type.clone(), &[0, 2]), Ok(output_type));

        let unbounded = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("input", DimensionBounds::unbounded()))]),
        );
        assert_eq!(
            unbounded.broadcast(
                ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                        "output",
                        DimensionBounds::non_negative(Some(4)).unwrap(),
                    ))]),
                ),
                &[0],
            ),
            Err(ProgramError::Type(TypeError::invalid(
                "broadcasting input axis 0 has size input but the output has size output; a dynamic dimension \
                    only broadcasts to an identical dynamic dimension"
                    .to_string(),
            ))),
        );
        assert_eq!(
            unbounded.broadcast(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])), &[0],),
            Err(ProgramError::Type(TypeError::invalid(
                "broadcasting input axis 0 has size input but the output has size 3; a dynamic dimension \
                    only broadcasts to an identical dynamic dimension"
                    .to_string(),
            ))),
        );
        assert_eq!(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1)])).broadcast(unbounded.clone(), &[0]),
            Err(ProgramError::Type(TypeError::invalid(
                "broadcasting cannot expand input axis 0 of size 1 into dynamic output size input".to_string(),
            ))),
        );
        assert_eq!(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])).broadcast(unbounded, &[0]),
            Err(ProgramError::Type(TypeError::invalid(
                "broadcasting input axis 0 has size 3 but the output has size input; a dynamic dimension \
                    only broadcasts to an identical dynamic dimension"
                    .to_string(),
            ))),
        );
        assert_eq!(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])).broadcast(
                ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![
                        Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
                        Dimension::Static(3),
                    ])
                ),
                &[1],
            ),
            Err(ProgramError::Type(TypeError::invalid(
                "broadcasting cannot replicate the input into unmapped dynamic output axis 0 of size dynamic"
                    .to_string(),
            ))),
        );
    }

    #[test]
    fn test_array_broadcast() {
        // Arbitrary axis mappings replicate the payload along every unmapped target axis.
        let target = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(3), Dimension::Static(2)]),
        );
        let output = Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).broadcast(target, &[0, 2]).unwrap();
        assert_eq!(output.to_f64s(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]);

        // Static unit axes stretch to the target extent, and empty target dimensions produce empty payloads.
        let target = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let output = Array::matrix(1, 3, vec![1.0, 2.0, 3.0]).broadcast(target, &[0, 1]).unwrap();
        assert_eq!(output.to_f64s(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let target = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(0), Dimension::Static(2)]));
        let output = Array::vector(vec![1.0, 2.0]).broadcast(target, &[1]).unwrap();
        assert_eq!(output.to_f64s(), Vec::<f64>::new());

        // The eager backend primitive preserves leading and right-aligned axis mappings.
        let output = Array::vector(vec![1.0, 2.0, 3.0])
            .broadcast(ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into()])), &[1])
            .unwrap();
        assert_eq!(*output.r#type(), ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into()])));
        assert_eq!(output.to_f64s(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let output = Array::scalar(7.0)
            .broadcast(ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into()])), &[])
            .unwrap();
        assert_eq!(output.to_f64s(), vec![7.0; 6]);
        let output = Array::vector(vec![10.0, 20.0, 30.0])
            .broadcast(ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into()])), &[1])
            .unwrap();
        assert_eq!(output.to_f64s(), vec![10.0, 20.0, 30.0, 10.0, 20.0, 30.0]);
        assert_eq!(
            Array::scalar(1.0).broadcast(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(0)])), &[],),
            Ok(Array::from_f64s(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(0)])), Vec::new(),)),
        );

        // Convenience broadcasts preserve placement metadata, project dimension shardings, and clear a physical
        // layout only when the shape changes. An exact identity preserves the complete type.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![8])))
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap()
            .with_memory(Memory::Host { pinned: true });
        let input = Array::from_f64s(input_type.clone(), vec![1.0, 2.0, 3.0]);
        let identity = input.broadcast(input_type.clone(), &[0]).unwrap();
        assert_eq!(*identity.r#type(), input_type);
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into()]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
                    .unwrap(),
            )
            .unwrap()
            .with_memory(Memory::Host { pinned: true });
        let output = input.broadcast(output_type, &[1]).unwrap();
        assert_eq!(output.r#type().memory(), Memory::Host { pinned: true });
        assert_eq!(output.r#type().layout(), None);
        assert_eq!(
            output.r#type().sharding(),
            Some(
                &Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],)
                    .unwrap(),
            ),
        );

        // Oversized target shapes fail through checked element-count arithmetic instead of panicking or wrapping.
        assert_eq!(
            Array::scalar(1.0).broadcast(
                ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(usize::MAX), Dimension::Static(2)]),),
                &[],
            ),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "shape [{}, 2] element count does not fit in usize",
                usize::MAX,
            )))),
        );
    }
}
