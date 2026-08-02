//! Differentiation and transposition rules for composite array-program operations.
//!
//! Homogeneous array rules remain with their owning operations. This module owns only the composite boundary:
//! extent-sensitive rules whose linear programs consume first-class dimensions, direct static delegation to the
//! homogeneous rules, and structural-zero cotangents for non-differentiable dimension operands. Keeping that boundary
//! in one capability-specific module avoids duplicating the composite context bounds across operation payloads while
//! removing the mathematics from the central operation-family module.

use crate::differentiation::forward::jvp_projected_operation;
use crate::differentiation::reverse::transpose_projected_operation;
use crate::operations::control_flow::{
    TemporalResidualOperation, TemporalResidualType, WhileResidualStackOperation, WhileResidualStackType,
    jvp_array_program_while,
};
use crate::operations::dimensions::RUNTIME_DIMENSION_DATA_TYPE;
use crate::operations::logical::AndOperation;

use super::*;

impl TemporalResidualType for ArrayProgramType {
    #[inline]
    fn temporal_storage_type(&self) -> Result<Self, TypeError> {
        Ok(match self {
            Self::Array(r#type) => Self::Array(r#type.clone()),
            Self::Dimension(_) => Self::Array(ArrayType::scalar(RUNTIME_DIMENSION_DATA_TYPE)),
        })
    }
}

impl<O> TemporalResidualOperation<ArrayProgramType> for O
where
    O: Operation<ArrayProgramType> + From<DimensionFromScalarOperation> + From<DimensionToScalarOperation>,
{
    fn residual_to_storage(residual_type: &ArrayProgramType) -> Result<Option<Self>, TypeError> {
        Ok(match residual_type {
            ArrayProgramType::Array(_) => None,
            ArrayProgramType::Dimension(_) => Some(Self::from(DimensionToScalarOperation)),
        })
    }

    fn residual_from_storage(residual_type: &ArrayProgramType) -> Result<Option<Self>, TypeError> {
        Ok(match residual_type {
            ArrayProgramType::Array(_) => None,
            ArrayProgramType::Dimension(r#type) => {
                Some(Self::from(DimensionFromScalarOperation::new(r#type.variable().clone())))
            }
        })
    }
}

impl WhileResidualStackType for ArrayProgramType {
    #[inline]
    fn from_array_type(r#type: ArrayType) -> Self {
        Self::Array(r#type)
    }

    fn array_type(&self) -> Result<&ArrayType, TypeError> {
        match self {
            Self::Array(r#type) => Ok(r#type),
            Self::Dimension(r#type) => {
                Err(TypeError::invalid(format!("expected an array-backed bounded-while state type but got {}", r#type)))
            }
        }
    }
}

impl<A: Value<Type = ArrayType>, O> WhileResidualStackOperation<ArrayProgramType, A> for O
where
    O: Operation<ArrayProgramType> + From<ArrayProgramOperation<A>> + TemporalResidualOperation<ArrayProgramType>,
{
    fn residual_stack_zero(r#type: ArrayProgramType) -> Self {
        let ArrayProgramType::Array(r#type) = r#type else {
            unreachable!("bounded-while stack zeros are always arrays")
        };
        Self::from(ArrayProgramOperation::<A>::from(ZeroOperation::new(r#type)))
    }

    fn residual_stack_one(r#type: ArrayProgramType) -> Self {
        let ArrayProgramType::Array(r#type) = r#type else {
            unreachable!("bounded-while stack ones are always arrays")
        };
        Self::from(ArrayProgramOperation::<A>::from(OneOperation::new(r#type)))
    }

    #[inline]
    fn residual_stack_broadcast(output_type: ArrayType, output_axes: Vec<usize>) -> Self {
        Self::from(ArrayProgramOperation::<A>::Array(ArrayOperation::Broadcast(LegacyBroadcastOperation::new(
            output_type,
            output_axes,
        ))))
    }

    #[inline]
    fn residual_stack_update() -> Self {
        Self::from(ArrayProgramOperation::<A>::Array(ArrayOperation::DynamicUpdateSlice(DynamicUpdateSliceOperation)))
    }

    #[inline]
    fn residual_stack_add() -> Self {
        Self::from(ArrayProgramOperation::<A>::from(AddOperation))
    }

    #[inline]
    fn residual_stack_select() -> Self {
        Self::from(ArrayProgramOperation::<A>::Array(ArrayOperation::Select(SelectOperation)))
    }

    #[inline]
    fn mask_reduce_any(axes: Vec<usize>) -> Self {
        Self::from(ArrayProgramOperation::<A>::Array(ArrayOperation::Reduce(ReduceOperation::new(
            axes,
            ReductionKind::Any,
        ))))
    }

    #[inline]
    fn mask_and() -> Self {
        Self::from(ArrayProgramOperation::<A>::Array(ArrayOperation::And(AndOperation)))
    }
}

impl<
    A: Value<Type = ArrayType>,
    C: Context<Type = ArrayProgramType, Constant: ValueProjection<ArrayType, Projected = A>> + Zero<C::Value>,
> DifferentiableOperation<C> for ArrayProgramOperation<A>
where
    C::Value: Concretizable<bool> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Operation: From<ArrayProgramOperation<A>>
        + From<ConditionOperation<C::Constant>>
        + From<DimensionFromScalarOperation>
        + From<DimensionToScalarOperation>
        + From<LinearCallOperation<ArrayProgramType>>
        + From<ScanOperation<C::Constant>>
        + From<WhileOperation>
        + From<ZeroOperation<ArrayProgramType>>
        + OperationProjection<ArrayType, Projected = ArrayOperation<A>>,
    ArrayOperation<A>: DifferentiableOperation<ProjectedContext<C, ArrayType>>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        if let Self::LinearCall(operation) = self {
            return operation.jvp(context, driver, inputs);
        }
        if let Self::Pad(_) = self {
            if inputs.len() < 2 {
                return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
            }
            let (array_inputs, output_extents) = inputs.split_at(2);
            let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
            let primal = context.bind(self.clone(), Vec::new(), primal_inputs.as_slice())?.remove(0);
            let tangent = if array_inputs.iter().all(|input| input.tangent().is_zero()) {
                MaybeZero::Zero(primal.r#type().tangent())
            } else {
                let projected_context = ProjectedContext::<C, ArrayType>::new(context.clone());
                let mut tangent_inputs = array_inputs
                    .iter()
                    .map(|input| -> Result<C::Value, DifferentiationError> {
                        let tangent = match input.tangent() {
                            MaybeZero::Zero(r#type) => MaybeZero::Zero(<&ArrayType>::try_from(r#type)?.clone()),
                            MaybeZero::Value(value) => MaybeZero::Value(
                                <C::Value as ValueProjection<ArrayType>>::into_projected(value.clone())?,
                            ),
                        };
                        Ok(<C::Value as ValueProjection<ArrayType>>::from_projected(
                            tangent.materialize(&projected_context)?,
                        ))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let operand_cotangent_type =
                    <&ArrayType>::try_from(array_inputs[0].primal().r#type().as_ref())?.cotangent();
                if operand_cotangent_type
                    .shape()
                    .dimensions()
                    .iter()
                    .all(|dimension| matches!(dimension, Dimension::Static(_)))
                {
                    tangent_inputs.extend(output_extents.iter().map(|extent| extent.primal().clone()));
                    MaybeZero::Value(context.bind(self.clone(), Vec::new(), tangent_inputs.as_slice())?.remove(0))
                } else {
                    let Self::Pad(operation) = self else {
                        unreachable!();
                    };
                    let mut residuals = LinearResiduals::new();
                    let output_extents =
                        residuals.retain_all(output_extents.iter().map(|extent| extent.primal().clone()));
                    let operand_shape = residuals.retain_shape::<A, _>(context, array_inputs[0].primal())?;
                    let forward_operation = operation.clone();
                    let forward_output_extents = output_extents.clone();
                    let transpose_operation = operation.clone();
                    let transpose_operand_type = operand_cotangent_type.clone();
                    let transpose_padding_type =
                        <&ArrayType>::try_from(array_inputs[1].primal().r#type().as_ref())?.cotangent();
                    let transpose_output_type = <&ArrayType>::try_from(primal.r#type().as_ref())?.cotangent();
                    let tangent = LinearCallOperation::stage(
                        context,
                        residuals.into_values(),
                        tangent_inputs,
                        move |residuals, linear_inputs| {
                            let mut pad_inputs = linear_inputs.to_vec();
                            pad_inputs.extend(forward_output_extents.iter().map(|index| residuals[*index].clone()));
                            linear_inputs[0].dispatch_domain().bind(
                                ArrayProgramOperation::<A>::from(forward_operation),
                                Vec::new(),
                                pad_inputs.as_slice(),
                            )
                        },
                        move |residuals, output_cotangents| {
                            let transpose_context = output_cotangents[0].dispatch_domain();
                            let output_cotangent = output_cotangents[0].clone();
                            let dimension_constant = |extent| dimension_constant::<A, _>(&transpose_context, extent);
                            let input_extents = operand_shape.dimensions::<A, _>(&transpose_context, residuals)?;

                            // Inverse edge padding first recovers the dilated input. Its exact result extents are
                            // `n + max(n - 1, 0) * interior`, derived from the retained input geometry.
                            let mut dilated_extents = Vec::with_capacity(transpose_operand_type.rank());
                            for (axis, input_extent) in input_extents.iter().enumerate() {
                                let interior = usize::try_from(transpose_operation.interior_padding()[axis])
                                    .map_err(|_| TypeError::invalid("'pad' interior padding must be nonnegative"))?;
                                if interior == 0 {
                                    dilated_extents.push(input_extent.clone());
                                    continue;
                                }
                                let one = dimension_constant(1)?;
                                let input_type = <&DimensionType>::try_from(input_extent.r#type().as_ref())?.clone();
                                let one_type = <&DimensionType>::try_from(one.r#type().as_ref())?.clone();
                                let less_one = transpose_context
                                    .bind(
                                        ArrayProgramOperation::<A>::from(DimensionOperation::SaturatingSub(
                                            DimensionSaturatingSubOperation::new(&input_type, &one_type)?,
                                        )),
                                        Vec::new(),
                                        &[input_extent.clone(), one],
                                    )?
                                    .remove(0);
                                let interior = dimension_constant(interior)?;
                                let less_one_type = <&DimensionType>::try_from(less_one.r#type().as_ref())?.clone();
                                let interior_type = <&DimensionType>::try_from(interior.r#type().as_ref())?.clone();
                                let gaps = transpose_context
                                    .bind(
                                        ArrayProgramOperation::<A>::from(DimensionOperation::Mul(
                                            DimensionMulOperation::new(&less_one_type, &interior_type)?,
                                        )),
                                        Vec::new(),
                                        &[less_one, interior],
                                    )?
                                    .remove(0);
                                let gaps_type = <&DimensionType>::try_from(gaps.r#type().as_ref())?.clone();
                                dilated_extents.push(
                                    transpose_context
                                        .bind(
                                            ArrayProgramOperation::<A>::from(DimensionOperation::Add(
                                                DimensionAddOperation::new(&input_type, &gaps_type)?,
                                            )),
                                            Vec::new(),
                                            &[input_extent.clone(), gaps],
                                        )?
                                        .remove(0),
                                );
                            }

                            let inverse_low = transpose_operation
                                .edge_padding_low()
                                .iter()
                                .enumerate()
                                .map(|(axis, padding)| {
                                    padding.checked_neg().ok_or_else(|| {
                                        TypeError::invalid(format!(
                                            "'pad' transpose cannot negate edge_padding_low at axis {axis} with \
                                             value {padding}",
                                        ))
                                    })
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            let inverse_high = transpose_operation
                                .edge_padding_high()
                                .iter()
                                .enumerate()
                                .map(|(axis, padding)| {
                                    padding.checked_neg().ok_or_else(|| {
                                        TypeError::invalid(format!(
                                            "'pad' transpose cannot negate edge_padding_high at axis {axis} with \
                                             value {padding}",
                                        ))
                                    })
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            let zero = transpose_context
                                .bind(
                                    ArrayProgramOperation::<A>::from(ZeroOperation::new(
                                        transpose_padding_type.clone(),
                                    )),
                                    Vec::new(),
                                    &[],
                                )?
                                .remove(0);
                            let mut inverse_inputs = vec![output_cotangent.clone(), zero];
                            inverse_inputs.extend(dilated_extents);
                            let unpadded = transpose_context
                                .bind(
                                    ArrayProgramOperation::<A>::from(PadOperation::new(
                                        inverse_low,
                                        inverse_high,
                                        vec![0; transpose_operand_type.rank()],
                                    )?),
                                    Vec::new(),
                                    inverse_inputs.as_slice(),
                                )?
                                .remove(0);
                            let starts = (0..transpose_operand_type.rank())
                                .map(|_| dimension_constant(0))
                                .collect::<Result<Vec<_>, _>>()?;
                            let mut slice_inputs = Vec::with_capacity(1 + 2 * transpose_operand_type.rank());
                            slice_inputs.push(unpadded);
                            slice_inputs.extend(starts);
                            slice_inputs.extend(input_extents.iter().cloned());
                            let strides = transpose_operation
                                .interior_padding()
                                .iter()
                                .enumerate()
                                .map(|(axis, padding)| {
                                    usize::try_from(*padding)
                                        .ok()
                                        .and_then(|padding| padding.checked_add(1))
                                        .ok_or_else(|| {
                                            TypeError::invalid(format!(
                                                "'pad' transpose stride overflows usize on axis {axis}",
                                            ))
                                        })
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            let input_cotangent = transpose_context
                                .bind(
                                    ArrayProgramOperation::<A>::from(
                                        DynamicShapeSliceOperation::new(transpose_operand_type.rank())
                                            .with_strides(strides)?,
                                    ),
                                    Vec::new(),
                                    slice_inputs.as_slice(),
                                )?
                                .remove(0);

                            // Select padding positions before summing so non-finite cotangents at operand positions
                            // cannot contaminate the padding-value contribution.
                            let mask_input_type =
                                transpose_operand_type.clone().with_data_type(DataType::Boolean).with_layout(None);
                            let mask_input_extents = mask_input_type
                                .shape()
                                .dimensions()
                                .iter()
                                .enumerate()
                                .filter_map(|(axis, dimension)| {
                                    matches!(dimension, Dimension::Dynamic(_)).then(|| input_extents[axis].clone())
                                })
                                .collect::<Vec<_>>();
                            let mask_input = transpose_context
                                .bind(
                                    ArrayProgramOperation::<A>::from(ZeroOperation::new(mask_input_type)),
                                    Vec::new(),
                                    mask_input_extents.as_slice(),
                                )?
                                .remove(0);
                            let mask_padding = transpose_context
                                .bind(
                                    ArrayProgramOperation::<A>::from(OneOperation::new(
                                        transpose_padding_type
                                            .clone()
                                            .with_data_type(DataType::Boolean)
                                            .with_layout(None),
                                    )),
                                    Vec::new(),
                                    &[],
                                )?
                                .remove(0);
                            let mut mask_inputs = vec![mask_input, mask_padding];
                            mask_inputs.extend(output_extents.iter().map(|index| residuals[*index].clone()));
                            let mask = transpose_context
                                .bind(
                                    ArrayProgramOperation::<A>::from(transpose_operation.clone()),
                                    Vec::new(),
                                    mask_inputs.as_slice(),
                                )?
                                .remove(0);
                            let output_zero_extents = transpose_output_type
                                .shape()
                                .dimensions()
                                .iter()
                                .enumerate()
                                .filter_map(|(axis, dimension)| {
                                    matches!(dimension, Dimension::Dynamic(_))
                                        .then(|| residuals[output_extents[axis]].clone())
                                })
                                .collect::<Vec<_>>();
                            let output_zero = transpose_context
                                .bind(
                                    ArrayProgramOperation::<A>::from(ZeroOperation::new(transpose_output_type.clone())),
                                    Vec::new(),
                                    output_zero_extents.as_slice(),
                                )?
                                .remove(0);
                            let selected = transpose_context
                                .bind(
                                    ArrayProgramOperation::<A>::Array(ArrayOperation::from(SelectOperation)),
                                    Vec::new(),
                                    &[mask, output_cotangent, output_zero],
                                )?
                                .remove(0);
                            let padding_cotangent = transpose_context
                                .bind(
                                    ArrayProgramOperation::<A>::Array(ArrayOperation::from(ReduceOperation::new(
                                        (0..transpose_output_type.rank()).collect(),
                                        ReductionKind::Sum,
                                    ))),
                                    Vec::new(),
                                    &[selected],
                                )?
                                .remove(0);
                            Ok(vec![input_cotangent, padding_cotangent])
                        },
                    )?
                    .remove(0);
                    MaybeZero::Value(tangent)
                }
            };
            return Ok(vec![DifferentiationDual::new(primal, tangent)?]);
        }
        if let Self::CustomCall(operation) = self {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "custom call '{}' has no differentiation rule; wrap it with `custom_jvp` or `custom_vjp` to \
                     provide one",
                    operation.target_name(),
                ),
            }
            .into());
        }
        if matches!(self, Self::Condition(_)) {
            return ConditionOperation::<C::Constant>::new().jvp(context, driver, inputs);
        }
        if let Self::Scan(operation) = self {
            let scan = ScanOperation::<C::Constant>::new(operation.carry_count(), operation.length())
                .with_reverse(operation.reverse())
                .with_unroll(operation.unroll())?;
            return scan.jvp(context, driver, inputs);
        }
        if let Self::While(operation) = self {
            return jvp_array_program_while(operation, context, driver, inputs);
        }
        if let Self::Array(ArrayOperation::Slice(operation)) = self {
            let [operand] = inputs else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() }.into());
            };
            let operand_type = <&ArrayType>::try_from(operand.primal().r#type().as_ref())?.clone();
            if operand_type.shape().dimensions().iter().any(|dimension| matches!(dimension, Dimension::Dynamic(_))) {
                let primal = context.bind(self.clone(), Vec::new(), std::slice::from_ref(operand.primal()))?.remove(0);
                let tangent = match operand.tangent() {
                    MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()),
                    MaybeZero::Value(operand_tangent) => {
                        let mut residuals = LinearResiduals::new();
                        let operand_shape = residuals.retain_shape::<A, _>(context, operand.primal())?;
                        let forward_operation = operation.clone();
                        let transpose_shape = operand_shape.clone();
                        let transpose_operand_type = operand_type.cotangent();
                        let transpose_starts = operation.start_indices().to_vec();
                        let transpose_strides = operation.strides().to_vec();
                        let tangent = LinearCallOperation::stage(
                            context,
                            residuals.into_values(),
                            vec![operand_tangent.clone()],
                            move |_, linear_inputs| {
                                linear_inputs[0].dispatch_domain().bind(
                                    ArrayProgramOperation::<A>::Array(ArrayOperation::Slice(forward_operation)),
                                    Vec::new(),
                                    std::slice::from_ref(&linear_inputs[0]),
                                )
                            },
                            move |residuals, output_cotangents| {
                                let transpose_context = output_cotangents[0].dispatch_domain();
                                let mut output_cotangent = output_cotangents[0].clone();
                                let zero_extents = transpose_shape.dynamic_dimensions(residuals);
                                let zeros = transpose_context
                                    .bind(
                                        ArrayProgramOperation::<A>::from(ZeroOperation::new(
                                            transpose_operand_type.clone(),
                                        )),
                                        Vec::new(),
                                        zero_extents.as_slice(),
                                    )?
                                    .remove(0);
                                if transpose_strides.iter().any(|stride| *stride != 1) {
                                    let padding_value = transpose_context
                                        .bind(
                                            ArrayProgramOperation::<A>::from(ZeroOperation::new(ArrayType::scalar(
                                                transpose_operand_type.data_type(),
                                            ))),
                                            Vec::new(),
                                            &[],
                                        )?
                                        .remove(0);
                                    output_cotangent = transpose_context
                                        .bind(
                                            ArrayProgramOperation::<A>::Array(ArrayOperation::Pad(PadOperation::new(
                                                vec![0; transpose_operand_type.rank()],
                                                vec![0; transpose_operand_type.rank()],
                                                transpose_strides.iter().map(|stride| stride - 1).collect(),
                                            )?)),
                                            Vec::new(),
                                            &[output_cotangent, padding_value],
                                        )?
                                        .remove(0);
                                }
                                transpose_context.bind(
                                    ArrayProgramOperation::<A>::Array(ArrayOperation::UpdateSlice(
                                        UpdateSliceOperation::new(transpose_starts),
                                    )),
                                    Vec::new(),
                                    &[zeros, output_cotangent],
                                )
                            },
                        )?
                        .remove(0);
                        MaybeZero::Value(tangent)
                    }
                };
                return Ok(vec![DifferentiationDual::new(primal, tangent)?]);
            }
        }
        if let Self::Array(ArrayOperation::DynamicSlice(operation)) = self {
            let (operand, start_indices) =
                inputs.split_first().ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?;
            let operand_type = <&ArrayType>::try_from(operand.primal().r#type().as_ref())?.clone();
            if operand_type.shape().dimensions().iter().any(|dimension| matches!(dimension, Dimension::Dynamic(_))) {
                let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
                let primal = context.bind(self.clone(), Vec::new(), primal_inputs.as_slice())?.remove(0);
                let tangent = match operand.tangent() {
                    MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()),
                    MaybeZero::Value(operand_tangent) => {
                        // Start indices have zero differential spaces but remain ordinary residual SSA values because
                        // both the forward slice and its transpose need their concrete runtime values.
                        let mut residuals = LinearResiduals::new();
                        let start_indices =
                            residuals.retain_all(start_indices.iter().map(|index| index.primal().clone()));
                        let operand_shape = residuals.retain_shape::<A, _>(context, operand.primal())?;
                        let forward_operation = operation.clone();
                        let forward_start_indices = start_indices.clone();
                        let transpose_shape = operand_shape.clone();
                        let transpose_operand_type = operand_type.cotangent();
                        let tangent = LinearCallOperation::stage(
                            context,
                            residuals.into_values(),
                            vec![operand_tangent.clone()],
                            move |residuals, linear_inputs| {
                                let mut slice_inputs = Vec::with_capacity(1 + forward_start_indices.len());
                                slice_inputs.push(linear_inputs[0].clone());
                                slice_inputs
                                    .extend(forward_start_indices.iter().map(|index| residuals[*index].clone()));
                                linear_inputs[0].dispatch_domain().bind(
                                    ArrayProgramOperation::<A>::Array(ArrayOperation::DynamicSlice(forward_operation)),
                                    Vec::new(),
                                    slice_inputs.as_slice(),
                                )
                            },
                            move |residuals, output_cotangents| {
                                let transpose_context = output_cotangents[0].dispatch_domain();
                                let zero_extents = transpose_shape.dynamic_dimensions(residuals);
                                let zeros = transpose_context
                                    .bind(
                                        ArrayProgramOperation::<A>::from(ZeroOperation::new(
                                            transpose_operand_type.clone(),
                                        )),
                                        Vec::new(),
                                        zero_extents.as_slice(),
                                    )?
                                    .remove(0);
                                let mut update_inputs = Vec::with_capacity(2 + start_indices.len());
                                update_inputs.push(zeros);
                                update_inputs.push(output_cotangents[0].clone());
                                update_inputs.extend(start_indices.iter().map(|index| residuals[*index].clone()));
                                transpose_context.bind(
                                    ArrayProgramOperation::<A>::Array(ArrayOperation::DynamicUpdateSlice(
                                        DynamicUpdateSliceOperation,
                                    )),
                                    Vec::new(),
                                    update_inputs.as_slice(),
                                )
                            },
                        )?
                        .remove(0);
                        MaybeZero::Value(tangent)
                    }
                };
                return Ok(vec![DifferentiationDual::new(primal, tangent)?]);
            }
        }
        if let Self::Array(ArrayOperation::DynamicUpdateSlice(_)) = self {
            if inputs.len() < 2 {
                return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
            }
            let operand = &inputs[0];
            let update = &inputs[1];
            let start_indices = &inputs[2..];
            let operand_type = <&ArrayType>::try_from(operand.primal().r#type().as_ref())?.clone();
            if operand_type.shape().dimensions().iter().any(|dimension| matches!(dimension, Dimension::Dynamic(_))) {
                let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
                let primal = context.bind(self.clone(), Vec::new(), primal_inputs.as_slice())?.remove(0);
                if operand.tangent().is_zero() && update.tangent().is_zero() {
                    return Ok(vec![DifferentiationDual::new(
                        primal.clone(),
                        MaybeZero::Zero(primal.r#type().tangent()),
                    )?]);
                }

                // The integer starts are the ordinary primal residuals shared by the forward update and its two
                // transpose branches. Input extents are retained only when a missing operand tangent must be
                // materialized inside the forward region; otherwise the output cotangent itself supplies the base
                // geometry to the transpose without an extra residual.
                let mut residuals = LinearResiduals::new();
                let start_indices = residuals.retain_all(start_indices.iter().map(|index| index.primal().clone()));
                let operand_is_live = !operand.tangent().is_zero();
                let update_is_live = !update.tangent().is_zero();
                let operand_shape = (!operand_is_live)
                    .then(|| residuals.retain_shape::<A, _>(context, operand.primal()))
                    .transpose()?;
                let update_type = <&ArrayType>::try_from(update.primal().r#type().as_ref())?.clone();
                let update_shape = (operand_is_live || !update_is_live)
                    .then(|| residuals.retain_shape::<A, _>(context, update.primal()))
                    .transpose()?;
                let mut linear_values = Vec::with_capacity(usize::from(operand_is_live) + usize::from(update_is_live));
                if let MaybeZero::Value(tangent) = operand.tangent() {
                    linear_values.push(tangent.clone());
                }
                if let MaybeZero::Value(tangent) = update.tangent() {
                    linear_values.push(tangent.clone());
                }
                let forward_operand_type = operand_type.tangent();
                let forward_update_type = update_type.tangent();
                let forward_start_indices = start_indices.clone();
                let forward_operand_shape = operand_shape.clone();
                let forward_update_shape = update_shape.clone();
                let transpose_start_indices = start_indices.clone();
                let transpose_update_shape = update_shape.clone();
                let transpose_update_type = update_type.cotangent();
                let update_sizes = if update_is_live {
                    transpose_update_type
                        .shape()
                        .dimensions()
                        .iter()
                        .enumerate()
                        .map(|(axis, dimension)| {
                            dimension.value().ok_or_else(|| {
                                TypeError::invalid(format!(
                                    "'dynamic_update_slice' transpose requires a static update extent but axis \
                                     {axis} has size {dimension}",
                                ))
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?
                } else {
                    Vec::new()
                };
                let tangent = LinearCallOperation::stage(
                    context,
                    residuals.into_values(),
                    linear_values,
                    move |residuals, linear_inputs| {
                        let forward_context = linear_inputs[0].dispatch_domain();
                        let mut linear_index = 0;
                        let operand_tangent = if operand_is_live {
                            let tangent = linear_inputs[linear_index].clone();
                            linear_index += 1;
                            tangent
                        } else {
                            let extents = forward_operand_shape.as_ref().unwrap().dynamic_dimensions(residuals);
                            forward_context
                                .bind(
                                    ArrayProgramOperation::<A>::from(ZeroOperation::new(forward_operand_type.clone())),
                                    Vec::new(),
                                    extents.as_slice(),
                                )?
                                .remove(0)
                        };
                        let update_tangent = if update_is_live {
                            linear_inputs[linear_index].clone()
                        } else {
                            let extents = forward_update_shape.as_ref().unwrap().dynamic_dimensions(residuals);
                            forward_context
                                .bind(
                                    ArrayProgramOperation::<A>::from(ZeroOperation::new(forward_update_type.clone())),
                                    Vec::new(),
                                    extents.as_slice(),
                                )?
                                .remove(0)
                        };
                        let mut update_inputs = Vec::with_capacity(2 + forward_start_indices.len());
                        update_inputs.extend([operand_tangent, update_tangent]);
                        update_inputs.extend(forward_start_indices.iter().map(|index| residuals[*index].clone()));
                        forward_context.bind(
                            ArrayProgramOperation::<A>::Array(ArrayOperation::DynamicUpdateSlice(
                                DynamicUpdateSliceOperation,
                            )),
                            Vec::new(),
                            update_inputs.as_slice(),
                        )
                    },
                    move |residuals, output_cotangents| {
                        let transpose_context = output_cotangents[0].dispatch_domain();
                        let mut cotangents =
                            Vec::with_capacity(usize::from(operand_is_live) + usize::from(update_is_live));
                        if operand_is_live {
                            let extents = transpose_update_shape.as_ref().unwrap().dynamic_dimensions(residuals);
                            let update_zero = transpose_context
                                .bind(
                                    ArrayProgramOperation::<A>::from(ZeroOperation::new(transpose_update_type.clone())),
                                    Vec::new(),
                                    extents.as_slice(),
                                )?
                                .remove(0);
                            let mut input_cotangent_inputs = vec![output_cotangents[0].clone(), update_zero];
                            input_cotangent_inputs
                                .extend(transpose_start_indices.iter().map(|index| residuals[*index].clone()));
                            cotangents.push(
                                transpose_context
                                    .bind(
                                        ArrayProgramOperation::<A>::Array(ArrayOperation::DynamicUpdateSlice(
                                            DynamicUpdateSliceOperation,
                                        )),
                                        Vec::new(),
                                        input_cotangent_inputs.as_slice(),
                                    )?
                                    .remove(0),
                            );
                        }
                        if update_is_live {
                            let mut update_cotangent_inputs = vec![output_cotangents[0].clone()];
                            update_cotangent_inputs
                                .extend(transpose_start_indices.iter().map(|index| residuals[*index].clone()));
                            cotangents.push(
                                transpose_context
                                    .bind(
                                        ArrayProgramOperation::<A>::Array(ArrayOperation::DynamicSlice(
                                            DynamicSliceOperation::new(update_sizes),
                                        )),
                                        Vec::new(),
                                        update_cotangent_inputs.as_slice(),
                                    )?
                                    .remove(0),
                            );
                        }
                        Ok(cotangents)
                    },
                )?
                .remove(0);
                return Ok(vec![DifferentiationDual::new(primal, MaybeZero::Value(tangent))?]);
            }
        }
        if let Self::Array(ArrayOperation::Gather(operation)) = self {
            let [operand, indices] = inputs else {
                return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
            };
            let operand_type = <&ArrayType>::try_from(operand.primal().r#type().as_ref())?.clone();
            if operand_type.shape().dimensions().iter().any(|dimension| matches!(dimension, Dimension::Dynamic(_))) {
                let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
                let primal = context.bind(self.clone(), Vec::new(), primal_inputs.as_slice())?.remove(0);
                let tangent = match operand.tangent() {
                    MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()),
                    MaybeZero::Value(operand_tangent) => {
                        let mut residuals = LinearResiduals::new();
                        let indices_index = residuals.retain(indices.primal().clone());
                        let operand_shape = residuals.retain_shape::<A, _>(context, operand.primal())?;
                        let forward_operation = operation.clone();
                        let transpose_operand_type = operand_type.cotangent();
                        let dimensions = operation.dimensions();
                        let transpose_operation = ScatterOperation::new(
                            ScatterDimensionNumbers::new(
                                dimensions.offset_dimensions().to_vec(),
                                dimensions.collapsed_slice_dimensions().to_vec(),
                                dimensions.start_index_map().to_vec(),
                            )
                            .with_batching_dimensions(
                                dimensions.operand_batching_dimensions().to_vec(),
                                dimensions.start_indices_batching_dimensions().to_vec(),
                            ),
                            ScatterReductionKind::Add,
                        )
                        .with_mode(operation.mode())
                        .with_indices_are_sorted(operation.indices_are_sorted())
                        .with_unique_indices(operation.unique_indices());
                        let tangent = LinearCallOperation::stage(
                            context,
                            residuals.into_values(),
                            vec![operand_tangent.clone()],
                            move |residuals, linear_inputs| {
                                linear_inputs[0].dispatch_domain().bind(
                                    ArrayProgramOperation::<A>::Array(ArrayOperation::Gather(forward_operation)),
                                    Vec::new(),
                                    &[linear_inputs[0].clone(), residuals[indices_index].clone()],
                                )
                            },
                            move |residuals, output_cotangents| {
                                let transpose_context = output_cotangents[0].dispatch_domain();
                                let zeros = transpose_context
                                    .bind(
                                        ArrayProgramOperation::<A>::from(ZeroOperation::new(
                                            transpose_operand_type.clone(),
                                        )),
                                        Vec::new(),
                                        operand_shape.dynamic_dimensions(residuals).as_slice(),
                                    )?
                                    .remove(0);
                                transpose_context.bind(
                                    ArrayProgramOperation::<A>::Array(ArrayOperation::Scatter(transpose_operation)),
                                    Vec::new(),
                                    &[zeros, residuals[indices_index].clone(), output_cotangents[0].clone()],
                                )
                            },
                        )?
                        .remove(0);
                        MaybeZero::Value(tangent)
                    }
                };
                return Ok(vec![DifferentiationDual::new(primal, tangent)?]);
            }
        }
        if let Self::Array(ArrayOperation::Reduce(operation)) = self
            && matches!(operation.kind(), ReductionKind::Max | ReductionKind::Min)
        {
            let [operand] = inputs else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() }.into());
            };
            let operand_type = <&ArrayType>::try_from(operand.primal().r#type().as_ref())?.clone();
            if operand_type.shape().dimensions().iter().any(|dimension| matches!(dimension, Dimension::Dynamic(_))) {
                let primal = context.bind(self.clone(), Vec::new(), std::slice::from_ref(operand.primal()))?.remove(0);
                let tangent = match operand.tangent() {
                    MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()),
                    MaybeZero::Value(operand_tangent) => {
                        let mut residuals = LinearResiduals::new();
                        let operand_shape = residuals.retain_shape::<A, _>(context, operand.primal())?;
                        let input_extents = operand_shape.dimensions::<A, _>(context, residuals.values())?;
                        let output_axes = (0..operand_type.rank())
                            .filter(|axis| !operation.axes().contains(axis))
                            .collect::<Vec<_>>();
                        let mut broadcast_inputs = Vec::with_capacity(1 + input_extents.len());
                        broadcast_inputs.push(primal.clone());
                        broadcast_inputs.extend(input_extents.iter().cloned());
                        let broadcast_primal = context
                            .bind(
                                ArrayProgramOperation::<A>::from(BroadcastOperation::new(output_axes.clone())),
                                Vec::new(),
                                broadcast_inputs.as_slice(),
                            )?
                            .remove(0);
                        let mask = context
                            .bind(
                                ArrayProgramOperation::<A>::Array(ArrayOperation::Compare(CompareOperation::new(
                                    ComparisonDirection::Equal,
                                ))),
                                Vec::new(),
                                &[operand.primal().clone(), broadcast_primal],
                            )?
                            .remove(0);
                        let numeric_mask = context
                            .bind(
                                ArrayProgramOperation::<A>::Array(ArrayOperation::ConvertElementType(
                                    ConvertElementTypeOperation::new(operand_type.tangent().data_type()),
                                )),
                                Vec::new(),
                                &[mask],
                            )?
                            .remove(0);
                        let tie_count = context
                            .bind(
                                ArrayProgramOperation::<A>::Array(ArrayOperation::Reduce(ReduceOperation::new(
                                    operation.axes().to_vec(),
                                    ReductionKind::Sum,
                                ))),
                                Vec::new(),
                                std::slice::from_ref(&numeric_mask),
                            )?
                            .remove(0);
                        let mut tie_broadcast_inputs = Vec::with_capacity(1 + input_extents.len());
                        tie_broadcast_inputs.push(tie_count);
                        tie_broadcast_inputs.extend(input_extents);
                        let broadcast_tie_count = context
                            .bind(
                                ArrayProgramOperation::<A>::from(BroadcastOperation::new(output_axes.clone())),
                                Vec::new(),
                                tie_broadcast_inputs.as_slice(),
                            )?
                            .remove(0);
                        let normalized_mask = context
                            .bind(
                                ArrayProgramOperation::<A>::Array(ArrayOperation::Div(DivOperation)),
                                Vec::new(),
                                &[numeric_mask, broadcast_tie_count],
                            )?
                            .remove(0);
                        let mask_index = residuals.retain(normalized_mask);
                        let forward_axes = operation.axes().to_vec();
                        let transpose_shape = operand_shape.clone();
                        let transpose_output_axes = output_axes.clone();
                        let transpose_target_type = operand_type.cotangent();
                        let tangent = LinearCallOperation::stage(
                            context,
                            residuals.into_values(),
                            vec![operand_tangent.clone()],
                            move |residuals, linear_inputs| {
                                let forward_context = linear_inputs[0].dispatch_domain();
                                let masked_tangent = forward_context
                                    .bind(
                                        ArrayProgramOperation::<A>::Array(ArrayOperation::Mul(MulOperation)),
                                        Vec::new(),
                                        &[residuals[mask_index].clone(), linear_inputs[0].clone()],
                                    )?
                                    .remove(0);
                                forward_context.bind(
                                    ArrayProgramOperation::<A>::Array(ArrayOperation::Reduce(ReduceOperation::new(
                                        forward_axes.clone(),
                                        ReductionKind::Sum,
                                    ))),
                                    Vec::new(),
                                    &[masked_tangent],
                                )
                            },
                            move |residuals, output_cotangents| {
                                let transpose_context = output_cotangents[0].dispatch_domain();
                                let input_extents =
                                    transpose_shape.dimensions::<A, _>(&transpose_context, residuals)?;
                                let mut broadcast_inputs = Vec::with_capacity(1 + input_extents.len());
                                broadcast_inputs.push(output_cotangents[0].clone());
                                broadcast_inputs.extend(input_extents);
                                let broadcasted = transpose_context
                                    .bind(
                                        ArrayProgramOperation::<A>::from(
                                            BroadcastOperation::new(transpose_output_axes.clone())
                                                .with_output_sharding(transpose_target_type.sharding().cloned()),
                                        ),
                                        Vec::new(),
                                        broadcast_inputs.as_slice(),
                                    )?
                                    .remove(0);
                                transpose_context.bind(
                                    ArrayProgramOperation::<A>::Array(ArrayOperation::Mul(MulOperation)),
                                    Vec::new(),
                                    &[residuals[mask_index].clone(), broadcasted],
                                )
                            },
                        )?
                        .remove(0);
                        MaybeZero::Value(tangent)
                    }
                };
                return Ok(vec![DifferentiationDual::new(primal, tangent)?]);
            }
        }
        if let Self::Array(ArrayOperation::Reduce(operation)) = self
            && matches!(operation.kind(), ReductionKind::Sum | ReductionKind::Mean)
        {
            let [operand] = inputs else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() }.into());
            };
            let operand_type = <&ArrayType>::try_from(operand.primal().r#type().as_ref())?.clone();
            if operand_type.shape().dimensions().iter().any(|dimension| matches!(dimension, Dimension::Dynamic(_))) {
                let primal = context.bind(self.clone(), Vec::new(), std::slice::from_ref(operand.primal()))?.remove(0);
                let tangent = match operand.tangent() {
                    MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()),
                    MaybeZero::Value(operand_tangent) => {
                        let mut residuals = LinearResiduals::new();
                        let operand_shape = residuals.retain_shape::<A, _>(context, operand.primal())?;
                        let forward_operation = operation.clone();
                        let transpose_operand_type = operand_type.cotangent();
                        let transpose_axes = operation.axes().to_vec();
                        let transpose_kind = operation.kind();
                        let transpose_output_axes = (0..transpose_operand_type.rank())
                            .filter(|axis| !transpose_axes.contains(axis))
                            .collect::<Vec<_>>();
                        let tangent = LinearCallOperation::stage(
                            context,
                            residuals.into_values(),
                            vec![operand_tangent.clone()],
                            move |_, linear_inputs| {
                                linear_inputs[0].dispatch_domain().bind(
                                    ArrayProgramOperation::<A>::Array(ArrayOperation::Reduce(forward_operation)),
                                    Vec::new(),
                                    std::slice::from_ref(&linear_inputs[0]),
                                )
                            },
                            move |residuals, output_cotangents| {
                                let transpose_context = output_cotangents[0].dispatch_domain();
                                let input_extents = operand_shape.dimensions::<A, _>(&transpose_context, residuals)?;
                                let mut broadcast_inputs = Vec::with_capacity(1 + input_extents.len());
                                broadcast_inputs.push(output_cotangents[0].clone());
                                broadcast_inputs.extend(input_extents.iter().cloned());
                                let broadcasted = transpose_context
                                    .bind(
                                        ArrayProgramOperation::<A>::from(
                                            BroadcastOperation::new(transpose_output_axes.clone())
                                                .with_output_sharding(transpose_operand_type.sharding().cloned()),
                                        ),
                                        Vec::new(),
                                        broadcast_inputs.as_slice(),
                                    )?
                                    .remove(0);
                                if transpose_kind == ReductionKind::Sum {
                                    return Ok(vec![broadcasted]);
                                }

                                let mut element_count = dimension_constant::<A, _>(&transpose_context, 1)?;
                                for axis in &transpose_axes {
                                    let left_type =
                                        <&DimensionType>::try_from(element_count.r#type().as_ref())?.clone();
                                    let right_type =
                                        <&DimensionType>::try_from(input_extents[*axis].r#type().as_ref())?.clone();
                                    element_count = transpose_context
                                        .bind(
                                            ArrayProgramOperation::<A>::from(DimensionOperation::Mul(
                                                DimensionMulOperation::new(&left_type, &right_type)?,
                                            )),
                                            Vec::new(),
                                            &[element_count, input_extents[*axis].clone()],
                                        )?
                                        .remove(0);
                                }
                                let element_count = transpose_context
                                    .bind(
                                        ArrayProgramOperation::<A>::from(DimensionToScalarOperation),
                                        Vec::new(),
                                        &[element_count],
                                    )?
                                    .remove(0);
                                let element_count = transpose_context
                                    .bind(
                                        ArrayProgramOperation::<A>::Array(ArrayOperation::ConvertElementType(
                                            ConvertElementTypeOperation::new(transpose_operand_type.data_type()),
                                        )),
                                        Vec::new(),
                                        &[element_count],
                                    )?
                                    .remove(0);
                                transpose_context.bind(
                                    ArrayProgramOperation::<A>::Array(ArrayOperation::Div(DivOperation)),
                                    Vec::new(),
                                    &[broadcasted, element_count],
                                )
                            },
                        )?
                        .remove(0);
                        MaybeZero::Value(tangent)
                    }
                };
                return Ok(vec![DifferentiationDual::new(primal, tangent)?]);
            }
        }
        if let Self::Concatenate(_) = self {
            let Some((result_extent, array_inputs)) = inputs.split_last() else {
                return Err(TypeError::invalid(format!(
                    "'{}' differentiation expects at least one array followed by its result extent",
                    CONCATENATE_OPERATION_NAME,
                ))
                .into());
            };
            if array_inputs.is_empty() {
                return match result_extent.primal().r#type().as_ref() {
                    ArrayProgramType::Array(_) => Err(TypeError::invalid(format!(
                        "'{}' differentiation expects a trailing result-extent dimension",
                        CONCATENATE_OPERATION_NAME,
                    ))
                    .into()),
                    ArrayProgramType::Dimension(_) => Err(TypeError::invalid(format!(
                        "'{}' differentiation expects at least one array before its result extent",
                        CONCATENATE_OPERATION_NAME,
                    ))
                    .into()),
                };
            }

            let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
            let primal = context.bind(self.clone(), Vec::new(), primal_inputs.as_slice())?.remove(0);
            let tangent = if array_inputs.iter().all(|input| input.tangent().is_zero()) {
                MaybeZero::Zero(primal.r#type().tangent())
            } else {
                // Concatenation is linear in its array operands. Materialize only the structural zero array tangents
                // needed beside live tangents, and replay the primal result extent as an unchanged shape input.
                let projected_context = ProjectedContext::<C, ArrayType>::new(context.clone());
                let mut tangent_inputs = array_inputs
                    .iter()
                    .map(|input| -> Result<C::Value, DifferentiationError> {
                        let tangent = match input.tangent() {
                            MaybeZero::Zero(r#type) => MaybeZero::Zero(<&ArrayType>::try_from(r#type)?.clone()),
                            MaybeZero::Value(value) => MaybeZero::Value(
                                <C::Value as ValueProjection<ArrayType>>::into_projected(value.clone())?,
                            ),
                        };
                        Ok(<C::Value as ValueProjection<ArrayType>>::from_projected(
                            tangent.materialize(&projected_context)?,
                        ))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let input_cotangent_types = array_inputs
                    .iter()
                    .map(|input| <&ArrayType>::try_from(input.primal().r#type().as_ref()).map(ArrayType::cotangent))
                    .collect::<Result<Vec<_>, _>>()?;
                if input_cotangent_types
                    .iter()
                    .flat_map(|r#type| r#type.shape().dimensions())
                    .all(|dimension| matches!(dimension, Dimension::Static(_)))
                {
                    tangent_inputs.push(result_extent.primal().clone());
                    MaybeZero::Value(context.bind(self.clone(), Vec::new(), tangent_inputs.as_slice())?.remove(0))
                } else {
                    let Self::Concatenate(operation) = self else {
                        unreachable!();
                    };
                    let mut residuals = LinearResiduals::new();
                    let result_extent_index = residuals.retain(result_extent.primal().clone());
                    let mut input_shapes = Vec::with_capacity(array_inputs.len());
                    for input in array_inputs {
                        input_shapes.push(residuals.retain_shape::<A, _>(context, input.primal())?);
                    }
                    let forward_operation = operation.clone();
                    let transpose_axis = operation.axis();
                    let tangent = LinearCallOperation::stage(
                        context,
                        residuals.into_values(),
                        tangent_inputs,
                        move |residuals, linear_inputs| {
                            let mut concatenate_inputs = linear_inputs.to_vec();
                            concatenate_inputs.push(residuals[result_extent_index].clone());
                            linear_inputs[0].dispatch_domain().bind(
                                ArrayProgramOperation::<A>::from(forward_operation),
                                Vec::new(),
                                concatenate_inputs.as_slice(),
                            )
                        },
                        move |residuals, output_cotangents| {
                            let transpose_context = output_cotangents[0].dispatch_domain();
                            let zero = dimension_constant::<A, _>(&transpose_context, 0)?;
                            let mut offset = zero.clone();
                            let mut cotangents = Vec::with_capacity(input_shapes.len());
                            for (input_index, input_shape) in input_shapes.iter().enumerate() {
                                let sizes = input_shape.dimensions::<A, _>(&transpose_context, residuals)?;
                                let mut starts = vec![zero.clone(); sizes.len()];
                                starts[transpose_axis] = offset.clone();
                                let mut slice_inputs = Vec::with_capacity(1 + 2 * sizes.len());
                                slice_inputs.push(output_cotangents[0].clone());
                                slice_inputs.extend(starts);
                                slice_inputs.extend(sizes.iter().cloned());
                                cotangents.push(
                                    transpose_context
                                        .bind(
                                            ArrayProgramOperation::<A>::from(DynamicShapeSliceOperation::new(
                                                sizes.len(),
                                            )),
                                            Vec::new(),
                                            slice_inputs.as_slice(),
                                        )?
                                        .remove(0),
                                );
                                if input_index + 1 < input_shapes.len() {
                                    let offset_type = <&DimensionType>::try_from(offset.r#type().as_ref())?.clone();
                                    let size_type =
                                        <&DimensionType>::try_from(sizes[transpose_axis].r#type().as_ref())?.clone();
                                    offset = transpose_context
                                        .bind(
                                            ArrayProgramOperation::<A>::from(DimensionOperation::Add(
                                                DimensionAddOperation::new(&offset_type, &size_type)?,
                                            )),
                                            Vec::new(),
                                            &[offset, sizes[transpose_axis].clone()],
                                        )?
                                        .remove(0);
                                }
                            }
                            Ok(cotangents)
                        },
                    )?
                    .remove(0);
                    MaybeZero::Value(tangent)
                }
            };
            return Ok(vec![DifferentiationDual::new(primal, tangent)?]);
        }
        if let Self::Reshape(operation) = self {
            let Some((array, output_extents)) = inputs.split_first() else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
            };
            let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
            let primal = context.bind(self.clone(), Vec::new(), primal_inputs.as_slice())?.remove(0);
            let tangent = match array.tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()),
                MaybeZero::Value(array_tangent) => {
                    let input_type = <&ArrayType>::try_from(array.primal().r#type().as_ref())?.clone();
                    let input_cotangent_type = input_type.cotangent();
                    let permuted_input_cotangent_type = match operation.dimensions() {
                        Some(dimensions) => input_cotangent_type.transpose(dimensions)?,
                        None => input_cotangent_type.clone(),
                    };
                    if permuted_input_cotangent_type
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
                        // Record each distinct dynamic input extent while the source array is still available. The
                        // values become ordinary trailing residual operands of the linear call below; repeated type
                        // identities reuse the same SSA value in first-use order.
                        let source_axes = operation.dimensions().map_or_else(
                            || (0..input_type.rank()).collect::<Vec<_>>(),
                            |dimensions| dimensions.to_vec(),
                        );
                        let mut residuals = LinearResiduals::new();
                        let output_extents =
                            residuals.retain_all(output_extents.iter().map(|extent| extent.primal().clone()));
                        let input_shape = residuals.retain_shape::<A, _>(context, array.primal())?;
                        let permuted_input_shape =
                            ExactShape(source_axes.iter().map(|axis| input_shape.0[*axis]).collect());

                        // The forward region is the ordinary tangent reshape over the canonical residuals-first
                        // boundary `[residuals..., tangent]`. The input-extent residuals are unused here but share
                        // one deterministic residual boundary with the transpose region.
                        let forward_operation = operation.clone();
                        let forward_output_extents = output_extents.clone();
                        let transpose_operation = operation.clone();
                        let transpose_target_type = input_cotangent_type.clone();
                        let transpose_permuted_type = permuted_input_cotangent_type.clone();
                        let tangent = LinearCallOperation::stage(
                            context,
                            residuals.into_values(),
                            vec![array_tangent.clone()],
                            move |residuals, linear_inputs| {
                                let mut reshape_inputs = Vec::with_capacity(1 + forward_output_extents.len());
                                reshape_inputs.push(linear_inputs[0].clone());
                                reshape_inputs
                                    .extend(forward_output_extents.iter().map(|index| residuals[*index].clone()));
                                linear_inputs[0].dispatch_domain().bind(
                                    ArrayProgramOperation::<A>::from(forward_operation),
                                    Vec::new(),
                                    reshape_inputs.as_slice(),
                                )
                            },
                            move |residuals, output_cotangents| {
                                let transpose_context = output_cotangents[0].dispatch_domain();
                                let bridge_sharding = match (
                                    transpose_permuted_type.sharding(),
                                    <&ArrayType>::try_from(output_cotangents[0].r#type().as_ref())?.sharding(),
                                ) {
                                    (Some(sharding), _) => Some(sharding.clone()),
                                    (None, Some(sharding)) => Some(Sharding::replicated(
                                        sharding.mesh().clone(),
                                        transpose_permuted_type.rank(),
                                    )),
                                    (None, None) => None,
                                };
                                let mut inverse_operation = ReshapeOperation::new();
                                if let Some(bridge_sharding) = bridge_sharding {
                                    inverse_operation = inverse_operation.with_output_sharding(bridge_sharding);
                                }
                                let mut inverse_inputs = Vec::with_capacity(transpose_permuted_type.rank() + 1);
                                inverse_inputs.push(output_cotangents[0].clone());
                                inverse_inputs
                                    .extend(permuted_input_shape.dimensions::<A, _>(&transpose_context, residuals)?);
                                let cotangent = transpose_context
                                    .bind(
                                        ArrayProgramOperation::<A>::from(inverse_operation),
                                        Vec::new(),
                                        inverse_inputs.as_slice(),
                                    )?
                                    .remove(0);
                                let cotangent = if let Some(dimensions) = transpose_operation.dimensions() {
                                    transpose_context
                                        .bind(
                                            ArrayProgramOperation::<A>::Array(ArrayOperation::Transpose(
                                                TransposeOperation::new(dimensions.inverse()?),
                                            )),
                                            Vec::new(),
                                            std::slice::from_ref(&cotangent),
                                        )?
                                        .remove(0)
                                } else {
                                    cotangent
                                };
                                let cotangent_type = cotangent.r#type();
                                let actual_type = <&ArrayType>::try_from(cotangent_type.as_ref())?;
                                if actual_type != &transpose_target_type {
                                    return Err(TypeError::invalid(format!(
                                        "inverse reshape cotangent type {actual_type} does not match input cotangent \
                                         type {transpose_target_type}",
                                    ))
                                    .into());
                                }
                                Ok(vec![cotangent])
                            },
                        )?
                        .remove(0);
                        MaybeZero::Value(tangent)
                    }
                }
            };
            return Ok(vec![DifferentiationDual::new(primal, tangent)?]);
        }
        if let Self::Broadcast(_) = self {
            let Some((array, output_extents)) = inputs.split_first() else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
            };
            let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
            let primal = context.bind(self.clone(), Vec::new(), primal_inputs.as_slice())?.remove(0);
            let tangent = match array.tangent() {
                MaybeZero::Zero(_) => {
                    let tangent_type = primal.r#type().tangent();
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
                        let operation = ArrayProgramOperation::<A>::from(ZeroOperation::new(array_tangent_type));
                        MaybeZero::Value(context.bind(operation, Vec::new(), dynamic_extents.as_slice())?.remove(0))
                    } else {
                        MaybeZero::Zero(tangent_type)
                    }
                }
                MaybeZero::Value(array_tangent) => {
                    let input_cotangent_type = <&ArrayType>::try_from(array.primal().r#type().as_ref())?.cotangent();
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
                        let Self::Broadcast(operation) = self else {
                            unreachable!();
                        };
                        let mut residuals = LinearResiduals::new();
                        let output_extents =
                            residuals.retain_all(output_extents.iter().map(|extent| extent.primal().clone()));
                        let input_shape = residuals.retain_shape::<A, _>(context, array.primal())?;
                        let forward_operation = operation.clone();
                        let forward_output_extents = output_extents.clone();
                        let transpose_output_axes = operation.output_axes().to_vec();
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
                                    ArrayProgramOperation::<A>::from(forward_operation),
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
                                            "cannot unalign cotangent axis {output_axis} of size {output_dimension} \
                                             to input axis {target_axis} of size {target_dimension}",
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
                                    .map(|kept| {
                                        kept_axes_by_output.iter().position(|candidate| candidate == kept).unwrap()
                                    })
                                    .collect::<Vec<_>>();
                                if permutation.iter().enumerate().any(|(axis, position)| axis != *position) {
                                    contribution = Transpose::transpose(&contribution, permutation)?;
                                }
                                let contribution = contribution.into_value();

                                // Rebind the exact input geometry through ordinary dimension operands. Besides making
                                // every dynamic extent an explicit data dependency, this prevents a metadata-only
                                // identity from standing in for the runtime shape at the linear-call boundary.
                                let contribution_type = contribution.r#type().into_owned();
                                let mut exact_inputs = Vec::with_capacity(transpose_target_type.rank() + 1);
                                exact_inputs.push(contribution);
                                exact_inputs.extend(input_shape.dimensions::<A, _>(&transpose_context, residuals)?);
                                let contribution_type = <&ArrayType>::try_from(&contribution_type)?;
                                if contribution_type.shape() == transpose_target_type.shape() {
                                    transpose_context.bind(
                                        ArrayProgramOperation::<A>::from(
                                            BroadcastOperation::new((0..transpose_target_type.rank()).collect())
                                                .with_output_sharding(transpose_target_type.sharding().cloned()),
                                        ),
                                        Vec::new(),
                                        exact_inputs.as_slice(),
                                    )
                                } else {
                                    transpose_context.bind(
                                        ArrayProgramOperation::<A>::from(
                                            ReshapeOperation::new()
                                                .with_output_sharding(transpose_target_type.sharding().cloned()),
                                        ),
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
            return Ok(vec![DifferentiationDual::new(primal, tangent)?]);
        }
        if matches!(self, Self::AllGather(_) | Self::PSumScatter(_) | Self::AllToAll(_)) {
            let Some((array, output_extents)) = inputs.split_first() else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
            };
            let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
            let primal = context.bind(self.clone(), Vec::new(), primal_inputs.as_slice())?.remove(0);
            let tangent = match array.tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()),
                MaybeZero::Value(array_tangent) => {
                    let mut residuals = LinearResiduals::new();
                    let output_extents =
                        residuals.retain_all(output_extents.iter().map(|extent| extent.primal().clone()));
                    let input_shape = residuals.retain_shape::<A, _>(context, array.primal())?;
                    let forward_operation = self.clone();
                    let forward_output_extents = output_extents.clone();
                    let transpose_operation = self.clone();
                    let transpose_target_type = <&ArrayType>::try_from(array.primal().r#type().as_ref())?.cotangent();
                    let tangent = LinearCallOperation::stage(
                        context,
                        residuals.into_values(),
                        vec![array_tangent.clone()],
                        move |residuals, linear_inputs| {
                            let mut collective_inputs = Vec::with_capacity(1 + forward_output_extents.len());
                            collective_inputs.push(linear_inputs[0].clone());
                            collective_inputs
                                .extend(forward_output_extents.iter().map(|index| residuals[*index].clone()));
                            linear_inputs[0].dispatch_domain().bind(
                                forward_operation,
                                Vec::new(),
                                collective_inputs.as_slice(),
                            )
                        },
                        move |residuals, output_cotangents| {
                            let transpose_context = output_cotangents[0].dispatch_domain();
                            let input_dimensions = input_shape.dimensions::<A, _>(&transpose_context, residuals)?;
                            let adjoint_operation = match &transpose_operation {
                                ArrayProgramOperation::AllGather(operation)
                                    if operation.output_variance() == AllGatherOutputVariance::Varying
                                        || operation.output_variance() == AllGatherOutputVariance::Reduced =>
                                {
                                    ArrayProgramOperation::<A>::from(PSumScatterOperation::new(
                                        operation.axis_name().to_string(),
                                        operation.axis_size(),
                                        operation.concat_axis(),
                                        operation.options().clone(),
                                    ))
                                }
                                ArrayProgramOperation::PSumScatter(operation) => {
                                    ArrayProgramOperation::<A>::from(AllGatherOperation::new(
                                        operation.axis_name().to_string(),
                                        operation.axis_size(),
                                        operation.scatter_axis(),
                                        operation.options().clone(),
                                        AllGatherOutputVariance::Varying,
                                    ))
                                }
                                ArrayProgramOperation::AllToAll(operation) => {
                                    ArrayProgramOperation::<A>::from(AllToAllOperation::new(
                                        operation.axis_name().to_string(),
                                        operation.axis_size(),
                                        operation.concat_axis(),
                                        operation.split_axis(),
                                        operation.options().clone(),
                                    ))
                                }
                                ArrayProgramOperation::AllGather(operation) => {
                                    let output_cotangent_type = output_cotangents[0].r#type();
                                    let output_cotangent_type = <&ArrayType>::try_from(output_cotangent_type.as_ref())?;
                                    let output_rank = output_cotangent_type.rank();
                                    let zero = dimension_constant::<A, _>(&transpose_context, 0)?;
                                    let chunk_extent = match operation.options().mode() {
                                        CollectiveMode::Tiled => input_dimensions[operation.concat_axis()].clone(),
                                        CollectiveMode::Untiled => dimension_constant::<A, _>(&transpose_context, 1)?,
                                    };
                                    let start = if operation.axis_size() == 1 {
                                        zero.clone()
                                    } else {
                                        let axis_index = transpose_context
                                            .bind(
                                                ArrayProgramOperation::<A>::Array(ArrayOperation::AxisIndex(
                                                    AxisIndexOperation::new(operation.axis_name().to_string()),
                                                )),
                                                Vec::new(),
                                                &[],
                                            )?
                                            .remove(0);
                                        let axis_index_variable = DimensionVariable::new(
                                            format!("{}_index", operation.axis_name()),
                                            DimensionBounds::non_negative(Some(operation.axis_size()))?,
                                        );
                                        let axis_index = transpose_context
                                            .bind(
                                                ArrayProgramOperation::<A>::from(DimensionFromScalarOperation::new(
                                                    axis_index_variable,
                                                )),
                                                Vec::new(),
                                                std::slice::from_ref(&axis_index),
                                            )?
                                            .remove(0);
                                        let axis_index_type =
                                            <&DimensionType>::try_from(axis_index.r#type().as_ref())?.clone();
                                        let chunk_extent_type =
                                            <&DimensionType>::try_from(chunk_extent.r#type().as_ref())?.clone();
                                        transpose_context
                                            .bind(
                                                ArrayProgramOperation::<A>::from(DimensionOperation::Mul(
                                                    DimensionMulOperation::new(&axis_index_type, &chunk_extent_type)?,
                                                )),
                                                Vec::new(),
                                                &[axis_index, chunk_extent.clone()],
                                            )?
                                            .remove(0)
                                    };
                                    let mut starts = vec![zero; output_rank];
                                    starts[operation.concat_axis()] = start;
                                    let mut slice_sizes = input_dimensions.clone();
                                    if operation.options().mode() == CollectiveMode::Untiled {
                                        slice_sizes.insert(operation.concat_axis(), chunk_extent);
                                    }
                                    let mut slice_inputs = Vec::with_capacity(1 + 2 * output_rank);
                                    slice_inputs.push(output_cotangents[0].clone());
                                    slice_inputs.extend(starts);
                                    slice_inputs.extend(slice_sizes);
                                    let selected = transpose_context
                                        .bind(
                                            ArrayProgramOperation::<A>::from(DynamicShapeSliceOperation::new(
                                                output_rank,
                                            )),
                                            Vec::new(),
                                            slice_inputs.as_slice(),
                                        )?
                                        .remove(0);
                                    let mut reshape_inputs = Vec::with_capacity(1 + input_dimensions.len());
                                    reshape_inputs.push(selected);
                                    reshape_inputs.extend(input_dimensions);
                                    return transpose_context.bind(
                                        ArrayProgramOperation::<A>::from(
                                            ReshapeOperation::new()
                                                .with_output_sharding(transpose_target_type.sharding().cloned()),
                                        ),
                                        Vec::new(),
                                        reshape_inputs.as_slice(),
                                    );
                                }
                                _ => unreachable!(),
                            };
                            let mut adjoint_inputs = Vec::with_capacity(1 + input_dimensions.len());
                            adjoint_inputs.push(output_cotangents[0].clone());
                            adjoint_inputs.extend(input_dimensions);
                            transpose_context.bind(adjoint_operation, Vec::new(), adjoint_inputs.as_slice())
                        },
                    )?
                    .remove(0);
                    MaybeZero::Value(tangent)
                }
            };
            return Ok(vec![DifferentiationDual::new(primal, tangent)?]);
        }
        let dynamic_constant_type = match self {
            Self::DynamicZero(operation) => Some(operation.r#type()),
            Self::DynamicOne(operation) => Some(operation.r#type()),
            Self::DynamicIota(operation) => Some(operation.r#type()),
            _ => None,
        };
        if let Some(output_type) = dynamic_constant_type {
            // Dynamic zero, one, and iota are constant with respect to their extent operands, but their zero tangents
            // still need those runtime extents for materialization. Stage dynamic zero while the operands remain
            // available instead of leaving a type-only structural zero for the generic output boundary.
            let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
            let primal = context.bind(self.clone(), Vec::new(), primal_inputs.as_slice())?.remove(0);
            let tangent_operation = ArrayProgramOperation::<A>::from(ZeroOperation::new(output_type.tangent()));
            let tangent = context.bind(tangent_operation, Vec::new(), primal_inputs.as_slice())?.remove(0);
            return Ok(vec![DifferentiationDual::new(primal, MaybeZero::Value(tangent))?]);
        }

        let Self::Array(operation) = self else {
            // Dimension-only and mixed shape-observation operations carry no differential dependence. Replaying the
            // primal through the composite context preserves their explicit SSA dependencies while structural zeros
            // prevent first-class dimensions from entering the tangent program.
            return Ok(context
                .bind(self.clone(), Vec::new(), &inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>())?
                .into_iter()
                .map(DifferentiationDual::new_with_zero_tangent)
                .collect());
        };

        jvp_projected_operation(context, operation, inputs)
    }
}

impl<
    A: Value<Type = ArrayType>,
    V: Value<Type = ArrayProgramType> + ValueProjection<ArrayType, Projected = A>,
    O: Operation<ArrayProgramType>
        + OperationProjection<ArrayType, Projected = ArrayOperation<A>>
        + From<ArrayProgramOperation<A>>
        + From<ConditionOperation<V>>
        + From<LinearCallOperation<ArrayProgramType>>
        + From<ScanOperation<V>>
        + From<ZeroOperation<ArrayProgramType>>,
> TransposableOperation<V, O> for ArrayProgramOperation<A>
where
    ArrayOperation<A>: TransposableOperation<A, ArrayOperation<A>>,
    ProjectedValue<ArrayType, Tracer<TracingContext<V, O>>>:
        BroadcastDerivativeAlignment + ElementwiseDerivativeAlignment<ArrayType> + Transpose,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        if let Self::LinearCall(operation) = self {
            return operation.transpose(context, driver, inputs, outputs);
        }
        if let Self::CustomCall(operation) = self {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "custom call '{}' cannot be transposed because foreign kernels are opaque",
                    operation.target_name(),
                ),
            }
            .into());
        }
        if matches!(self, Self::RngBitGenerator(_)) {
            return Err(ProgramError::UnsupportedOperation {
                message: "'rng_bit_generator' cannot be transposed because random bits are discrete".to_string(),
            }
            .into());
        }
        if matches!(self, Self::Condition(_)) {
            return ConditionOperation::<V>::new().transpose(context, driver, inputs, outputs);
        }
        if let Self::Scan(operation) = self {
            let scan = ScanOperation::<V>::new(operation.carry_count(), operation.length())
                .with_reverse(operation.reverse())
                .with_unroll(operation.unroll())?
                .with_captures(
                    operation
                        .captures()
                        .iter()
                        .map(|capture| match capture {
                            ArrayProgramValue::Array(value) => V::from_projected(value.clone()),
                            ArrayProgramValue::Dimension(_) => {
                                unreachable!("validated scan captures are always arrays")
                            }
                        })
                        .collect(),
                );
            return scan.transpose(context, driver, inputs, outputs);
        }
        if matches!(self, Self::DynamicZero(_) | Self::DynamicOne(_) | Self::DynamicIota(_)) {
            check_count!("output", outputs, 1, ProgramError);
            // A shaped constructor depends on its extent operands only as non-differentiable shape inputs. Its
            // array value is constant with respect to those operands, so every extent receives a structural-zero
            // cotangent regardless of the array output cotangent.
            return Ok(inputs.iter().map(|input| MaybeZero::Zero(input.r#type().cotangent())).collect());
        }
        if matches!(self, Self::AllGather(_) | Self::PSumScatter(_) | Self::AllToAll(_)) {
            let Some((array_input, output_extents)) = inputs.split_first() else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
            };
            if matches!(
                self,
                Self::AllGather(operation) if operation.output_variance() == AllGatherOutputVariance::Invariant
            ) {
                return Err(ProgramError::UnsupportedOperation {
                    message: "direct invariant 'all_gather' transposition requires linearization so that the current \
                              participant can select its gathered chunk"
                        .to_string(),
                }
                .into());
            }
            if array_input.r#type().identities().any(|(position, _)| position == TypeIdentityPosition::Reference)
                || output_extents.iter().any(|extent| {
                    extent.r#type().identities().any(|(position, _)| position == TypeIdentityPosition::Reference)
                })
            {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "direct '{}' transposition with dynamic extents requires linearization so that the primal \
                         geometry can be retained as residuals",
                        self.name(),
                    ),
                }
                .into());
            }
            let operation = match self {
                Self::AllGather(operation) => ArrayOperation::AllGather(operation.clone()),
                Self::PSumScatter(operation) => ArrayOperation::PSumScatter(operation.clone()),
                Self::AllToAll(operation) => ArrayOperation::AllToAll(operation.clone()),
                _ => unreachable!(),
            };
            let mut cotangents =
                Self::Array(operation).transpose(context, driver, std::slice::from_ref(array_input), outputs)?;
            cotangents.extend(output_extents.iter().map(|extent| MaybeZero::Zero(extent.r#type().cotangent())));
            return Ok(cotangents);
        }
        if let Self::Pad(operation) = self {
            if inputs.len() < 2 {
                return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
            }
            let (array_inputs, output_extents) = inputs.split_at(2);
            if array_inputs.iter().any(|input| {
                <&ArrayType>::try_from(input.r#type().as_ref()).is_ok_and(|r#type| {
                    r#type.shape().dimensions().iter().any(|dimension| matches!(dimension, Dimension::Dynamic(_)))
                })
            }) || output_extents.iter().any(|extent| {
                <&DimensionType>::try_from(extent.r#type().as_ref())
                    .is_ok_and(|r#type| matches!(r#type.to_dimension(), Dimension::Dynamic(_)))
            }) {
                return Err(ProgramError::UnsupportedOperation {
                    message: "direct 'pad' transposition with dynamic extents requires linearization so that the \
                              primal geometry can be retained as residuals"
                        .to_string(),
                }
                .into());
            }

            // Exact extents make the mixed instruction identical to the established homogeneous pad map. Delegate
            // that pullback for the two differentiable array operands and assign structural-zero cotangents to the
            // trailing extent values.
            let array_operation = Self::Array(ArrayOperation::from(operation.clone()));
            let mut cotangents = array_operation.transpose(context, driver, array_inputs, outputs)?;
            cotangents.extend(output_extents.iter().map(|extent| MaybeZero::Zero(extent.r#type().cotangent())));
            return Ok(cotangents);
        }
        if let Self::Concatenate(operation) = self {
            let Some((result_extent, array_inputs)) = inputs.split_last() else {
                return Err(TypeError::invalid(format!(
                    "'{}' transpose expects at least one array followed by its result extent",
                    CONCATENATE_OPERATION_NAME,
                ))
                .into());
            };
            if array_inputs.is_empty() {
                return match result_extent.r#type().as_ref() {
                    ArrayProgramType::Array(_) => Err(TypeError::invalid(format!(
                        "'{}' transpose expects a trailing result-extent dimension",
                        CONCATENATE_OPERATION_NAME,
                    ))
                    .into()),
                    ArrayProgramType::Dimension(_) => Err(TypeError::invalid(format!(
                        "'{}' transpose expects at least one array before its result extent",
                        CONCATENATE_OPERATION_NAME,
                    ))
                    .into()),
                };
            }
            for input in array_inputs {
                let input_type = input.r#type();
                let input_type = <&ArrayType>::try_from(input_type.as_ref())?;
                if matches!(input_type.dimension(operation.axis()), Dimension::Dynamic(_)) {
                    return Err(ProgramError::UnsupportedOperation {
                        message: format!(
                            "direct transposition of a dynamic '{}' requires linearization so its input extents can \
                             be retained as residuals",
                            CONCATENATE_OPERATION_NAME,
                        ),
                    }
                    .into());
                }
            }

            // Static concatenation uses the established homogeneous pullback, which slices the output cotangent at
            // cumulative input offsets. The explicit result extent is a non-differentiable shape input and has a
            // structural-zero cotangent.
            let array_operation = Self::Array(ArrayOperation::from(operation.clone()));
            let mut cotangents = array_operation.transpose(context, driver, array_inputs, outputs)?;
            cotangents.push(MaybeZero::Zero(result_extent.r#type().cotangent()));
            return Ok(cotangents);
        }
        if let Self::Broadcast(operation) = self {
            let Some((input, output_extents)) = inputs.split_first() else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
            };
            let input_cotangent_type = <&ArrayType>::try_from(input.r#type().as_ref())?.cotangent();
            if input_cotangent_type
                .shape()
                .dimensions()
                .iter()
                .any(|dimension| matches!(dimension, Dimension::Dynamic(_)))
            {
                return Err(ProgramError::UnsupportedOperation {
                    message: "direct transposition of a dynamic 'broadcast' requires linearization so its input \
                              extents can be retained as residuals"
                        .to_string(),
                }
                .into());
            }

            // Exact extents let the canonical homogeneous broadcast pullback own the reduction and axis alignment.
            let output_type = match outputs {
                [MaybeZero::Zero(r#type)] => <&ArrayType>::try_from(r#type)?.clone(),
                [MaybeZero::Value(value)] => <&ArrayType>::try_from(value.r#type().as_ref())?.clone(),
                _ => return Err(ProgramError::InvalidOutputCount { expected: 1, actual: outputs.len() }.into()),
            };
            let array_operation = Self::Array(ArrayOperation::from(LegacyBroadcastOperation::new(
                output_type,
                operation.output_axes().to_vec(),
            )));
            let mut cotangents = array_operation.transpose(context, driver, std::slice::from_ref(input), outputs)?;
            cotangents.extend(output_extents.iter().map(|extent| MaybeZero::Zero(extent.r#type().cotangent())));
            return Ok(cotangents);
        }

        if let Self::Reshape(operation) = self {
            let Some((input, output_extents)) = inputs.split_first() else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
            };
            let input_cotangent_type = <&ArrayType>::try_from(input.r#type().as_ref())?.cotangent();
            let permuted_input_cotangent_type = match operation.dimensions() {
                Some(dimensions) => input_cotangent_type.transpose(dimensions)?,
                None => input_cotangent_type.clone(),
            };
            if permuted_input_cotangent_type
                .shape()
                .dimensions()
                .iter()
                .any(|dimension| matches!(dimension, Dimension::Dynamic(_)))
            {
                return Err(ProgramError::UnsupportedOperation {
                    message: "direct transposition of a dynamic 'reshape' requires linearization so its input extents \
                              are available as explicit residuals"
                        .to_string(),
                }
                .into());
            }

            // Exact extents allow the homogeneous reshape rule to own inverse geometry, permutation, sharding, and
            // cotangent alignment. The result cotangent has the primal reshape's output shape.
            let output_type = match outputs {
                [MaybeZero::Zero(r#type)] => <&ArrayType>::try_from(r#type)?.clone(),
                [MaybeZero::Value(value)] => <&ArrayType>::try_from(value.r#type().as_ref())?.clone(),
                _ => return Err(ProgramError::InvalidOutputCount { expected: 1, actual: outputs.len() }.into()),
            };
            let mut parameters = ReshapeParameters::new(output_type.shape().clone());
            if let Some(dimensions) = operation.dimensions() {
                parameters = parameters.with_dimensions(dimensions.clone());
            }
            if let Some(output_sharding) = operation.output_sharding() {
                parameters = parameters.with_output_sharding(output_sharding.clone());
            }
            let array_operation = Self::Array(ArrayOperation::from(LegacyReshapeOperation::new(parameters)));
            let mut cotangents = array_operation.transpose(context, driver, std::slice::from_ref(input), outputs)?;
            cotangents.extend(output_extents.iter().map(|extent| MaybeZero::Zero(extent.r#type().cotangent())));
            return Ok(cotangents);
        }

        let Self::Array(operation) = self else {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("operation `{}` is not transposable", self.name()),
            }
            .into());
        };

        transpose_projected_operation(context, operation, inputs, outputs)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::array_programs::{ArrayProgramOperation, ArrayProgramValue};
    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::backends::dimensions::DimensionValue;
    use crate::operations::compare::{CompareOperation, ComparisonDirection};
    use crate::operations::control_flow::{ConditionOperation, ScanOperation, WhileOperation};
    use crate::operations::dimensions::DimensionFromScalarOperation;
    use crate::operations::manipulation::{BroadcastOperation, ReshapeOperation};
    use crate::operations::math::{AddOperation, MulOperation, ReduceOperation, ReductionKind};
    use crate::parameters::Placeholder;
    use crate::programs::{Program, ProgramBuilder};
    use crate::types::{
        ArrayProgramType, ArrayType, DataType, Dimension, DimensionBounds, DimensionType, DimensionVariable, Shape,
    };

    type TestValue = ArrayProgramValue<Array>;
    type TestOperation = ArrayProgramOperation<Array>;

    fn array(value: Array) -> TestValue {
        TestValue::Array(value)
    }

    fn dimension(r#type: &DimensionType, extent: usize) -> TestValue {
        TestValue::Dimension(DimensionValue::new(r#type.clone(), extent).unwrap())
    }

    fn scale_branch(
        dimension_type: DimensionType,
        factor: f64,
    ) -> Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>> {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let extent = builder.add_input(ArrayProgramType::Dimension(dimension_type));
        let operand = builder.add_input(ArrayProgramType::Array(ArrayType::scalar(DataType::F64)));
        let factor = builder.add_constant(array(Array::scalar(factor)));
        let output = builder
            .add_instruction(
                TestOperation::Array(ArrayOperation::from(MulOperation)),
                Vec::new(),
                vec![operand, factor],
            )
            .unwrap()[0];
        builder.build(vec![extent, output], vec![Placeholder; 2], vec![Placeholder; 2]).unwrap()
    }

    #[test]
    fn test_composite_condition_jvp_preserves_dimension_outputs_without_tangent_slots() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap()));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayProgramType::Array(ArrayType::scalar(DataType::Boolean)));
        let extent = builder.add_input(ArrayProgramType::Dimension(extent_type.clone()));
        let operand = builder.add_input(ArrayProgramType::Array(ArrayType::scalar(DataType::F64)));
        let true_branch = scale_branch(extent_type.clone(), 2.0);
        let false_branch = scale_branch(extent_type.clone(), 3.0);
        let regions = vec![
            builder.import_region(true_branch.entry_region_ref()),
            builder.import_region(false_branch.entry_region_ref()),
        ];
        let outputs = builder
            .add_instruction(
                TestOperation::Condition(ConditionOperation::new()),
                regions,
                vec![predicate, extent, operand],
            )
            .unwrap()
            .to_vec();
        let program = builder.build(outputs, vec![Placeholder; 3], vec![Placeholder; 2]).unwrap();

        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_count(), 4);
        assert_eq!(jvp.output_count(), 3);
        let outputs = jvp
            .interpret(vec![
                array(Array::from_f64s(ArrayType::scalar(DataType::Boolean), vec![1.0])),
                dimension(&extent_type, 4),
                array(Array::scalar(5.0)),
                array(Array::scalar(7.0)),
            ])
            .unwrap();
        assert!(matches!(&outputs[0], TestValue::Dimension(value) if value.extent() == 4));
        assert!(matches!(&outputs[1], TestValue::Array(value) if value.to_f64s() == vec![10.0]));
        assert!(matches!(&outputs[2], TestValue::Array(value) if value.to_f64s() == vec![14.0]));

        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                array(Array::from_f64s(ArrayType::scalar(DataType::Boolean), vec![1.0])),
                dimension(&extent_type, 4),
                array(Array::scalar(5.0)),
            ])
            .unwrap();
        let residuals = primal_outputs.split_off(2);
        let mut pullback_inputs = vec![array(Array::scalar(1.0))];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![array(Array::scalar(2.0))]),);
    }

    fn product_scan_body(
        extent_type: DimensionType,
    ) -> Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>> {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let extent = builder.add_input(ArrayProgramType::Dimension(extent_type));
        let carry = builder.add_input(ArrayProgramType::Array(ArrayType::scalar(DataType::F64)));
        let item = builder.add_input(ArrayProgramType::Array(ArrayType::scalar(DataType::F64)));
        let product = builder
            .add_instruction(TestOperation::Array(ArrayOperation::from(MulOperation)), Vec::new(), vec![carry, item])
            .unwrap()[0];
        builder.build(vec![extent, product, product], vec![Placeholder; 3], vec![Placeholder; 3]).unwrap()
    }

    #[test]
    fn test_composite_scan_jvp_forwards_a_dynamic_length_and_dimension_carry() {
        let carry_extent_type =
            DimensionType::new(DimensionVariable::new("carry_extent", DimensionBounds::positive(Some(8)).unwrap()));
        let length_variable = DimensionVariable::new("length", DimensionBounds::positive(Some(8)).unwrap());
        let length_type = DimensionType::new(length_variable.clone());
        let length = Dimension::Dynamic(length_variable);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let extent = builder.add_input(ArrayProgramType::Dimension(carry_extent_type.clone()));
        let carry = builder.add_input(ArrayProgramType::Array(ArrayType::scalar(DataType::F64)));
        let values =
            builder.add_input(ArrayProgramType::Array(ArrayType::new(DataType::F64, Shape::new(vec![length.clone()]))));
        let runtime_length = builder.add_input(ArrayProgramType::Dimension(length_type.clone()));
        let body = product_scan_body(carry_extent_type.clone());
        let region = builder.import_region(body.entry_region_ref());
        let outputs = builder
            .add_instruction(
                TestOperation::Scan(ScanOperation::new(2, length)),
                vec![region],
                vec![extent, carry, values, runtime_length],
            )
            .unwrap()
            .to_vec();
        let program = builder.build(outputs, vec![Placeholder; 4], vec![Placeholder; 3]).unwrap();

        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_count(), 6);
        assert_eq!(jvp.output_count(), 5);
        let outputs = jvp
            .interpret(vec![
                dimension(&carry_extent_type, 4),
                array(Array::scalar(1.0)),
                array(Array::vector(vec![2.0, 3.0, 4.0])),
                dimension(&length_type, 3),
                array(Array::scalar(5.0)),
                array(Array::vector(vec![0.5, 1.0, 1.5])),
            ])
            .unwrap();
        assert!(matches!(&outputs[0], TestValue::Dimension(value) if value.extent() == 4));
        assert!(matches!(&outputs[1], TestValue::Array(value) if value.to_f64s() == vec![24.0]));
        assert!(matches!(&outputs[3], TestValue::Array(value) if value.to_f64s() == vec![143.0]));

        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 3);
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                dimension(&carry_extent_type, 4),
                array(Array::scalar(1.0)),
                array(Array::vector(vec![2.0, 3.0, 4.0])),
                dimension(&length_type, 3),
            ])
            .unwrap();
        let residuals = primal_outputs.split_off(3);
        let mut pullback_inputs = vec![array(Array::scalar(1.0)), array(Array::vector(vec![0.0, 0.0, 0.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![array(Array::scalar(24.0)), array(Array::vector(vec![12.0, 8.0, 6.0]))]),
        );
    }

    #[test]
    fn test_composite_scan_pullback_stacks_varying_dimension_residuals_through_scalar_gateways() {
        let iteration_variable = DimensionVariable::new("iteration", DimensionBounds::positive(Some(4)).unwrap());
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let scalar_u64 = ArrayType::scalar(DataType::U64);

        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let state = body_builder.add_input(ArrayProgramType::Array(scalar_f64.clone()));
        let counter = body_builder.add_input(ArrayProgramType::Array(scalar_u64.clone()));
        let iteration = body_builder
            .add_instruction(
                TestOperation::from(DimensionFromScalarOperation::new(iteration_variable)),
                Vec::new(),
                vec![counter],
            )
            .unwrap()[0];
        let repeated = body_builder
            .add_instruction(
                TestOperation::from(BroadcastOperation::new(Vec::new())),
                Vec::new(),
                vec![state, iteration],
            )
            .unwrap()[0];
        let next_state = body_builder
            .add_instruction(
                TestOperation::Array(ArrayOperation::from(ReduceOperation::new(vec![0], ReductionKind::Sum))),
                Vec::new(),
                vec![repeated],
            )
            .unwrap()[0];
        let one = body_builder.add_constant(array(Array::scalar(1_u64)));
        let next_counter = body_builder
            .add_instruction(TestOperation::Array(ArrayOperation::from(AddOperation)), Vec::new(), vec![counter, one])
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![next_state, next_counter, next_state],
                vec![Placeholder; 2],
                vec![Placeholder; 3],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let state = builder.add_input(ArrayProgramType::Array(scalar_f64));
        let counter = builder.add_input(ArrayProgramType::Array(scalar_u64));
        let region = builder.import_region(body.entry_region_ref());
        let outputs = builder
            .add_instruction(TestOperation::Scan(ScanOperation::new(2, 2)), vec![region], vec![state, counter])
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder; 2], vec![Placeholder; 3])
            .unwrap();

        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 2);
        let rendered_primal = linearization.primal().to_string();
        let rendered_tangent = linearization.tangent().to_string();
        assert!(rendered_primal.contains("dimension_to_scalar"), "{rendered_primal}");
        assert!(rendered_tangent.contains("dimension_from_scalar"), "{rendered_tangent}");
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![array(Array::scalar(2.0)), array(Array::scalar(1_u64))])
            .unwrap();
        let residuals = primal_outputs.split_off(3);
        let mut pullback_inputs = vec![array(Array::scalar(1.0)), array(Array::vector(vec![0.0, 0.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![array(Array::scalar(2.0))]));
    }

    fn doubling_while_regions(
        extent_type: DimensionType,
    ) -> Vec<Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>> {
        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        condition_builder.add_input(ArrayProgramType::Dimension(extent_type.clone()));
        let state = condition_builder.add_input(ArrayProgramType::Array(ArrayType::scalar(DataType::F64)));
        let limit = condition_builder.add_constant(array(Array::scalar(8.0)));
        let predicate = condition_builder
            .add_instruction(
                TestOperation::Array(ArrayOperation::from(CompareOperation::new(ComparisonDirection::LessThan))),
                Vec::new(),
                vec![state, limit],
            )
            .unwrap()[0];
        let condition = condition_builder.build(vec![predicate], vec![Placeholder; 2], vec![Placeholder]).unwrap();

        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let extent = body_builder.add_input(ArrayProgramType::Dimension(extent_type));
        let state = body_builder.add_input(ArrayProgramType::Array(ArrayType::scalar(DataType::F64)));
        let doubled = body_builder
            .add_instruction(TestOperation::Array(ArrayOperation::from(AddOperation)), Vec::new(), vec![state, state])
            .unwrap()[0];
        let body = body_builder.build(vec![extent, doubled], vec![Placeholder; 2], vec![Placeholder; 2]).unwrap();
        vec![condition, body]
    }

    #[test]
    fn test_composite_while_jvp_omits_the_dimension_state_tangent() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap()));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let extent = builder.add_input(ArrayProgramType::Dimension(extent_type.clone()));
        let state = builder.add_input(ArrayProgramType::Array(ArrayType::scalar(DataType::F64)));
        let regions = doubling_while_regions(extent_type.clone());
        let regions = regions.iter().map(|region| builder.import_region(region.entry_region_ref())).collect();
        let outputs = builder
            .add_instruction(
                TestOperation::While(WhileOperation::new().with_iteration_bound(4).unwrap()),
                regions,
                vec![extent, state],
            )
            .unwrap()
            .to_vec();
        let program = builder.build(outputs, vec![Placeholder; 2], vec![Placeholder; 2]).unwrap();

        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_count(), 3);
        assert_eq!(jvp.output_count(), 3);
        let outputs = jvp
            .interpret(vec![dimension(&extent_type, 4), array(Array::scalar(1.0)), array(Array::scalar(3.0))])
            .unwrap();
        assert!(matches!(&outputs[0], TestValue::Dimension(value) if value.extent() == 4));
        assert!(matches!(&outputs[1], TestValue::Array(value) if value.to_f64s() == vec![8.0]));
        assert!(matches!(&outputs[2], TestValue::Array(value) if value.to_f64s() == vec![24.0]));

        let linearization = program.linearize().unwrap();
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![dimension(&extent_type, 4), array(Array::scalar(1.0))])
            .unwrap();
        let residuals = primal_outputs.split_off(2);
        let mut pullback_inputs = vec![array(Array::scalar(1.0))];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![array(Array::scalar(8.0))]),);
    }

    #[test]
    fn test_composite_bounded_while_pullback_supports_batched_predicates() {
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));

        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let state = condition_builder.add_input(ArrayProgramType::Array(vector_type.clone()));
        let limits = condition_builder.add_constant(array(Array::vector(vec![2.0, 4.0, 8.0])));
        let predicate = condition_builder
            .add_instruction(
                TestOperation::Array(ArrayOperation::from(CompareOperation::new(ComparisonDirection::LessThan))),
                Vec::new(),
                vec![state, limits],
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let state = body_builder.add_input(ArrayProgramType::Array(vector_type.clone()));
        let doubled = body_builder
            .add_instruction(TestOperation::Array(ArrayOperation::from(AddOperation)), Vec::new(), vec![state, state])
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![doubled], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let state = builder.add_input(ArrayProgramType::Array(vector_type));
        let regions =
            vec![builder.import_region(condition.entry_region_ref()), builder.import_region(body.entry_region_ref())];
        let outputs = builder
            .add_instruction(
                TestOperation::While(WhileOperation::new().with_iteration_bound(4).unwrap()),
                regions,
                vec![state],
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder], vec![Placeholder])
            .unwrap();

        let linearization = program.linearize().unwrap();
        let mut primal_outputs =
            linearization.primal().interpret(vec![array(Array::vector(vec![1.0, 1.0, 1.0]))]).unwrap();
        assert_eq!(primal_outputs[0], array(Array::vector(vec![2.0, 4.0, 8.0])));
        let residuals = primal_outputs.split_off(1);
        let mut pullback_inputs = vec![array(Array::vector(vec![1.0, 1.0, 1.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![array(Array::vector(vec![2.0, 4.0, 8.0]))]),
        );
    }

    #[test]
    fn test_composite_bounded_while_pullback_threads_invariant_dimension_residuals_as_scan_carries() {
        let extent_variable = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let extent_type = DimensionType::new(extent_variable.clone());
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent_variable)]));

        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        condition_builder.add_input(ArrayProgramType::Dimension(extent_type.clone()));
        condition_builder.add_input(ArrayProgramType::Array(vector_type.clone()));
        let counter = condition_builder.add_input(ArrayProgramType::Array(ArrayType::scalar(DataType::I64)));
        let limit = condition_builder.add_constant(array(Array::scalar(2_i64)));
        let predicate = condition_builder
            .add_instruction(
                TestOperation::Array(ArrayOperation::from(CompareOperation::new(ComparisonDirection::LessThan))),
                Vec::new(),
                vec![counter, limit],
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let extent = body_builder.add_input(ArrayProgramType::Dimension(extent_type.clone()));
        let vector = body_builder.add_input(ArrayProgramType::Array(vector_type.clone()));
        let counter = body_builder.add_input(ArrayProgramType::Array(ArrayType::scalar(DataType::I64)));
        let reshaped = body_builder
            .add_instruction(TestOperation::from(ReshapeOperation::new()), Vec::new(), vec![vector, extent])
            .unwrap()[0];
        let one = body_builder.add_constant(array(Array::scalar(1_i64)));
        let next_counter = body_builder
            .add_instruction(TestOperation::Array(ArrayOperation::from(AddOperation)), Vec::new(), vec![counter, one])
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![extent, reshaped, next_counter],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let extent = builder.add_input(ArrayProgramType::Dimension(extent_type.clone()));
        let vector = builder.add_input(ArrayProgramType::Array(vector_type));
        let counter = builder.add_input(ArrayProgramType::Array(ArrayType::scalar(DataType::I64)));
        let regions =
            vec![builder.import_region(condition.entry_region_ref()), builder.import_region(body.entry_region_ref())];
        let outputs = builder
            .add_instruction(
                TestOperation::While(WhileOperation::new().with_iteration_bound(4).unwrap()),
                regions,
                vec![extent, vector, counter],
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder; 3], vec![Placeholder; 3])
            .unwrap();

        let linearization = program.linearize().unwrap();
        let rendered_primal = linearization.primal().to_string();
        let rendered_tangent = linearization.tangent().to_string();
        assert!(rendered_primal.contains("scan [carry_count=1"), "{rendered_primal}");
        assert!(!rendered_primal.contains("dimension_to_scalar"), "{rendered_primal}");
        assert!(!rendered_tangent.contains("dimension_from_scalar"), "{rendered_tangent}");
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                dimension(&extent_type, 3),
                array(Array::vector(vec![1.0, 2.0, 3.0])),
                array(Array::scalar(0_i64)),
            ])
            .unwrap();
        let residuals = primal_outputs.split_off(3);
        let mut pullback_inputs = vec![array(Array::vector(vec![1.0, 1.0, 1.0]))];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![array(Array::vector(vec![1.0, 1.0, 1.0]))]),
        );
    }

    #[test]
    fn test_composite_bounded_while_pullback_stacks_varying_dimension_residuals_through_scalar_gateways() {
        let iteration_variable = DimensionVariable::new("iteration", DimensionBounds::positive(Some(4)).unwrap());
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let scalar_u64 = ArrayType::scalar(DataType::U64);

        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        condition_builder.add_input(ArrayProgramType::Array(scalar_f64.clone()));
        let counter = condition_builder.add_input(ArrayProgramType::Array(scalar_u64.clone()));
        let limit = condition_builder.add_constant(array(Array::scalar(3_u64)));
        let predicate = condition_builder
            .add_instruction(
                TestOperation::Array(ArrayOperation::from(CompareOperation::new(ComparisonDirection::LessThan))),
                Vec::new(),
                vec![counter, limit],
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let state = body_builder.add_input(ArrayProgramType::Array(scalar_f64.clone()));
        let counter = body_builder.add_input(ArrayProgramType::Array(scalar_u64.clone()));
        let iteration = body_builder
            .add_instruction(
                TestOperation::from(DimensionFromScalarOperation::new(iteration_variable)),
                Vec::new(),
                vec![counter],
            )
            .unwrap()[0];
        let repeated = body_builder
            .add_instruction(
                TestOperation::from(BroadcastOperation::new(Vec::new())),
                Vec::new(),
                vec![state, iteration],
            )
            .unwrap()[0];
        let next_state = body_builder
            .add_instruction(
                TestOperation::Array(ArrayOperation::from(ReduceOperation::new(vec![0], ReductionKind::Sum))),
                Vec::new(),
                vec![repeated],
            )
            .unwrap()[0];
        let one = body_builder.add_constant(array(Array::scalar(1_u64)));
        let next_counter = body_builder
            .add_instruction(TestOperation::Array(ArrayOperation::from(AddOperation)), Vec::new(), vec![counter, one])
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![next_state, next_counter],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let state = builder.add_input(ArrayProgramType::Array(scalar_f64));
        let counter = builder.add_input(ArrayProgramType::Array(scalar_u64));
        let regions =
            vec![builder.import_region(condition.entry_region_ref()), builder.import_region(body.entry_region_ref())];
        let outputs = builder
            .add_instruction(
                TestOperation::While(WhileOperation::new().with_iteration_bound(4).unwrap()),
                regions,
                vec![state, counter],
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let linearization = program.linearize().unwrap();
        let rendered_primal = linearization.primal().to_string();
        let rendered_tangent = linearization.tangent().to_string();
        assert!(rendered_primal.contains("dimension_to_scalar"), "{rendered_primal}");
        assert!(rendered_tangent.contains("dimension_from_scalar"), "{rendered_tangent}");
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![array(Array::scalar(2.0)), array(Array::scalar(1_u64))])
            .unwrap();
        let residuals = primal_outputs.split_off(2);
        let mut pullback_inputs = vec![array(Array::scalar(1.0))];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![array(Array::scalar(2.0))]));
    }
}
