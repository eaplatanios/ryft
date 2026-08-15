use crate::arrays::ir::ArrayIrValue;
use crate::arrays::{ArrayIrType, DimensionType, DimensionValue};
use crate::differentiation::DifferentiableType;
use crate::operations::constants::iota::IotaOperation;
use crate::operations::differentiation::stop_gradient::StopGradient;
use crate::operations::dimensions::DimensionSize;
use crate::operations::manipulation::{DynamicBroadcast, DynamicReshape};
use crate::programs::ValueProjection;

use super::*;

/// Array-member projection used throughout the mixed attention composition.
type ArrayProjection<V> = <V as ValueProjection<ArrayType>>::Projected;

/// Private capability for constructing an iota with one mixed value's runtime geometry.
pub trait AttentionIota: Value<Type = ArrayIrType> {
    /// Constructs an `i32` iota matching `target` and varying along `axis`.
    fn attention_iota(target: &Self, axis: usize) -> Result<Self, ProgramError>;
}

impl<
    V: Value<
            Type = ArrayIrType,
            DispatchDomain: Context<Type = ArrayIrType, Operation: From<IotaOperation<ArrayType>>>,
        > + DimensionSize
        + ValueProjection<ArrayType>,
> AttentionIota for V
{
    fn attention_iota(target: &Self, axis: usize) -> Result<Self, ProgramError> {
        let target_type = target.r#type();
        let target_type = <&ArrayType>::try_from(target_type.as_ref())?;
        let output_type = target_type.clone().with_data_type(DataType::I32);
        let dimensions = array_dimensions(target)?;
        let dynamic_dimensions = output_type
            .shape()
            .dimensions()
            .iter()
            .enumerate()
            .filter_map(|(axis, dimension)| {
                matches!(dimension, Dimension::Dynamic(_)).then(|| dimensions[axis].clone())
            })
            .collect::<Vec<_>>();
        let mut outputs = target.dispatch_domain().bind(
            IotaOperation::new(output_type, axis)?,
            Vec::new(),
            dynamic_dimensions.as_slice(),
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<A> AttentionIota for ArrayIrValue<A>
where
    A: Value<Type = ArrayType> + DimensionSize<usize>,
    A::DispatchDomain: Iota<A>,
{
    fn attention_iota(target: &Self, axis: usize) -> Result<Self, ProgramError> {
        let target = <ArrayIrValue<A> as ValueProjection<ArrayType>>::projected(target)?;
        let output_type = ArrayType::new(
            DataType::I32,
            Shape::new(
                (0..target.r#type().rank())
                    .map(|axis| target.dimension_size(axis).map(Dimension::Static))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
        );
        Ok(Self::Array(target.dispatch_domain().iota(&output_type, axis)?))
    }
}

/// Projects the array member of one mixed array-IR value.
fn project_array<V: ValueProjection<ArrayType>>(value: V) -> Result<ArrayProjection<V>, ProgramError> {
    value.into_projected().map_err(Into::into)
}

/// Returns one first-class extent value for every axis of `value`.
fn array_dimensions<V: Value<Type = ArrayIrType> + DimensionSize>(value: &V) -> Result<Vec<V>, ProgramError> {
    let r#type = value.r#type();
    let r#type = <&ArrayType>::try_from(r#type.as_ref())?;
    (0..r#type.rank()).map(|axis| value.dimension_size(axis)).collect()
}

/// Broadcasts one projected array to the exact runtime geometry of `target`.
fn broadcast_like<V>(
    value: <V as ValueProjection<ArrayType>>::Projected,
    target: &V,
    axes: &[usize],
) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayIrType> + DimensionSize + DynamicBroadcast + ValueProjection<ArrayType>,
{
    <V as ValueProjection<ArrayType>>::from_projected(value)
        .dynamic_broadcast(array_dimensions(target)?.as_slice(), axes)
}

/// Materializes a scalar literal and broadcasts it to `target` using first-class extents.
fn fill_like<V>(target: &V, value: f64) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayIrType> + DimensionSize + DynamicBroadcast + ValueProjection<ArrayType>,
    ArrayProjection<V>: Value<Type = ArrayType>,
    <ArrayProjection<V> as Value>::DispatchDomain: Fill<f64, ArrayProjection<V>>,
{
    let target_type = target.r#type();
    let target_type = <&ArrayType>::try_from(target_type.as_ref())?;
    let scalar = project_array::<V>(target.clone())?
        .dispatch_domain()
        .fill(&ArrayType::scalar(target_type.data_type()), value)?;
    broadcast_like::<V>(scalar, target, &[])
}

/// Expands grouped key/value heads through first-class broadcast and reshape extents.
fn expand_key_value_heads_ir<V>(operand: &V, dimensions: &AttentionDimensions) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayIrType> + DimensionSize + DynamicBroadcast + DynamicReshape + ValueProjection<ArrayType>,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Constant: From<DimensionValue>,
{
    if dimensions.key_value_heads == dimensions.query_heads {
        return Ok(operand.clone());
    }
    let context = operand.dispatch_domain();
    let group = dimensions.query_heads / dimensions.key_value_heads;
    let operand_dimensions = array_dimensions(operand)?;
    let key_value_heads = context.lift(DimensionValue::constant(dimensions.key_value_heads)?.into())?;
    let group = context.lift(DimensionValue::constant(group)?.into())?;
    let query_heads = context.lift(DimensionValue::constant(dimensions.query_heads)?.into())?;
    let expanded = operand.dynamic_broadcast(
        &[
            operand_dimensions[0].clone(),
            operand_dimensions[1].clone(),
            key_value_heads,
            group,
            operand_dimensions[3].clone(),
        ],
        &[0, 1, 2, 4],
    )?;
    expanded.dynamic_reshape(&[
        operand_dimensions[0].clone(),
        operand_dimensions[1].clone(),
        query_heads,
        operand_dimensions[3].clone(),
    ])
}

/// Adds the implicit batch axis used to evaluate an unbatched attention operand.
fn normalize_attention_operand_ir<V>(operand: &V) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayIrType> + DimensionSize + DynamicReshape + ValueProjection<ArrayType>,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Constant: From<DimensionValue>,
{
    let r#type = operand.r#type();
    let r#type = <&ArrayType>::try_from(r#type.as_ref())?;
    if r#type.rank() == 4 {
        return Ok(operand.clone());
    }
    let batch = operand.dispatch_domain().lift(DimensionValue::constant(1)?.into())?;
    let mut dimensions = array_dimensions(operand)?;
    dimensions.insert(0, batch);
    operand.clone().dynamic_reshape(dimensions.as_slice())
}

/// Prepends singleton axes until one bias or mask has the normalized `BNTS` score rank.
fn normalize_attention_score_operand_ir<V>(operand: &V) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayIrType> + DimensionSize + DynamicReshape + ValueProjection<ArrayType>,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Constant: From<DimensionValue>,
{
    let r#type = operand.r#type();
    let r#type = <&ArrayType>::try_from(r#type.as_ref())?;
    if r#type.rank() == 4 {
        return Ok(operand.clone());
    }
    let context = operand.dispatch_domain();
    let singleton = context.lift(DimensionValue::constant(1)?.into())?;
    let mut dimensions = vec![singleton; 4 - r#type.rank()];
    dimensions.extend(array_dimensions(operand)?);
    operand.clone().dynamic_reshape(dimensions.as_slice())
}

/// Removes the implicit batch axis after evaluating unbatched attention.
fn denormalize_attention_output_ir<V>(
    output: ArrayProjection<V>,
    rank: usize,
) -> Result<ArrayProjection<V>, ProgramError>
where
    V: Value<Type = ArrayIrType> + DimensionSize + DynamicReshape + ValueProjection<ArrayType>,
{
    if rank == 4 {
        return Ok(output);
    }
    let output = <V as ValueProjection<ArrayType>>::from_projected(output);
    let dimensions = array_dimensions(&output)?;
    let output = output.dynamic_reshape(&dimensions[1..])?;
    project_array::<V>(output)
}

/// Adds the implicit batch axis to the `TN` residual of an unbatched attention operation.
fn normalize_attention_residual_ir<V>(residual: &V) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayIrType> + DimensionSize + DynamicReshape + ValueProjection<ArrayType>,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Constant: From<DimensionValue>,
{
    let r#type = residual.r#type();
    let r#type = <&ArrayType>::try_from(r#type.as_ref())?;
    if r#type.rank() == 3 {
        return Ok(residual.clone());
    }
    let batch = residual.dispatch_domain().lift(DimensionValue::constant(1)?.into())?;
    let mut dimensions = array_dimensions(residual)?;
    dimensions.insert(0, batch);
    residual.clone().dynamic_reshape(dimensions.as_slice())
}

/// Applies all built-in and sequence-length masks to mixed-IR logits.
fn apply_masks_ir<V>(
    scores: V,
    mask: Option<&V>,
    configuration: AttentionConfiguration,
    query_sequence_lengths: Option<&V>,
    key_value_sequence_lengths: Option<&V>,
) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayIrType>
        + AttentionIota
        + DimensionSize
        + DynamicBroadcast
        + DynamicReshape
        + ValueProjection<ArrayType, Projected: Value<Type = ArrayType> + Add + And + Compare + Select + Sub>,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Constant: From<DimensionValue>,
    <<V as ValueProjection<ArrayType>>::Projected as Value>::DispatchDomain:
        Fill<f64, <V as ValueProjection<ArrayType>>::Projected>,
{
    if mask.is_none()
        && !configuration.causal()
        && configuration.local_window().is_none()
        && query_sequence_lengths.is_none()
        && key_value_sequence_lengths.is_none()
    {
        return Ok(scores);
    }
    let columns = V::attention_iota(&scores, 3)?;
    let mut visible = match mask {
        None => None,
        Some(mask) => {
            let mask = normalize_attention_score_operand_ir(mask)?;
            let mask = project_array::<V>(mask)?;
            Some(project_array::<V>(broadcast_like::<V>(mask, &scores, &[0, 1, 2, 3])?)?)
        }
    };
    if configuration.causal() {
        let rows = V::attention_iota(&scores, 2)?;
        let columns_array = project_array::<V>(columns.clone())?;
        let rows_array = project_array::<V>(rows.clone())?;
        let causal = columns_array.compare(&rows_array, ComparisonDirection::LessThanOrEqual)?;
        visible = Some(match visible {
            None => causal,
            Some(visible) => visible.and(&causal)?,
        });
    }
    if let Some((left, right)) = configuration.local_window() {
        let rows = V::attention_iota(&scores, 2)?;
        let columns_array = project_array::<V>(columns.clone())?;
        let rows_array = project_array::<V>(rows.clone())?;
        let left = fill_like(&rows, left as f64)?;
        let right = fill_like(&rows, right as f64)?;
        let lower = rows_array.sub(&project_array::<V>(left)?)?;
        let upper = rows_array.add(&project_array::<V>(right)?)?;
        let in_window = columns_array
            .compare(&lower, ComparisonDirection::GreaterThanOrEqual)?
            .and(&columns_array.compare(&upper, ComparisonDirection::LessThanOrEqual)?)?;
        visible = Some(match visible {
            None => in_window,
            Some(visible) => visible.and(&in_window)?,
        });
    }
    if let Some(lengths) = query_sequence_lengths {
        let rows = V::attention_iota(&scores, 2)?;
        let bounds = lengths.clone().dynamic_broadcast(array_dimensions(&rows)?.as_slice(), &[0])?;
        let in_range =
            project_array::<V>(rows)?.compare(&project_array::<V>(bounds)?, ComparisonDirection::LessThan)?;
        visible = Some(match visible {
            None => in_range,
            Some(visible) => visible.and(&in_range)?,
        });
    }
    if let Some(lengths) = key_value_sequence_lengths {
        let bounds = lengths.clone().dynamic_broadcast(array_dimensions(&columns)?.as_slice(), &[0])?;
        let in_range =
            project_array::<V>(columns)?.compare(&project_array::<V>(bounds)?, ComparisonDirection::LessThan)?;
        visible = Some(match visible {
            None => in_range,
            Some(visible) => visible.and(&in_range)?,
        });
    }
    let scores_array = project_array::<V>(scores.clone())?;
    let score_type = scores.r#type();
    let score_type = <&ArrayType>::try_from(score_type.as_ref())?;
    let large_negative = if score_type.data_type() == DataType::F64 { -0.7 * f64::MAX } else { -0.7 * f32::MAX as f64 };
    let masked = fill_like(&scores, large_negative)?;
    Ok(<V as ValueProjection<ArrayType>>::from_projected(<ArrayProjection<V> as Select>::select(
        &visible.unwrap(),
        &scores_array,
        &project_array::<V>(masked)?,
    )?))
}

/// Shared normalized operands and masked logits consumed by the forward and backward attention compositions.
struct PreparedAttention<V>
where
    V: Value<Type = ArrayIrType> + ValueProjection<ArrayType>,
{
    /// Normalized query operand.
    query: V,

    /// Normalized key operand.
    key: V,

    /// Key operand expanded from key/value heads to query heads.
    expanded_key: V,

    /// Value operand expanded from key/value heads to query heads.
    expanded_value: V,

    /// Masked logits converted to the data type used by softmax.
    logits: ArrayProjection<V>,

    /// Shared query/key/value data type.
    data_type: DataType,

    /// Data type used by softmax and its residual statistic.
    softmax_type: DataType,

    /// Explicit or default score scale.
    scale: f64,
}

/// Normalizes attention operands and constructs the masked score logits shared by forward and backward.
fn prepare_attention_ir<V>(
    inputs: &AttentionInputs<V>,
    configuration: AttentionConfiguration,
    dimensions: &AttentionDimensions,
) -> Result<PreparedAttention<V>, ProgramError>
where
    V: Value<Type = ArrayIrType>
        + AttentionIota
        + DimensionSize
        + DynamicBroadcast
        + DynamicReshape
        + ValueProjection<ArrayType>
        + ValueProjection<DimensionType>,
    ArrayProjection<V>: Value<Type = ArrayType>
        + Add
        + And
        + Broadcast
        + Compare
        + ConvertElementType
        + Dot
        + Mul
        + Reshape
        + Select
        + Sub,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Constant: From<DimensionValue>,
    <ArrayProjection<V> as Value>::DispatchDomain: Fill<f64, ArrayProjection<V>>,
{
    let query = normalize_attention_operand_ir(&inputs.query)?;
    let key = normalize_attention_operand_ir(&inputs.key)?;
    let value = normalize_attention_operand_ir(&inputs.value)?;
    let expanded_key = expand_key_value_heads_ir(&key, dimensions)?;
    let expanded_value = expand_key_value_heads_ir(&value, dimensions)?;
    let data_type = dimensions.data_type;
    // Dot products, scaling, bias, and masking promote low-precision inputs to `f32`, while `f64` remains `f64`.
    // The stable softmax itself always runs in `f32`, matching the public attention contract.
    let logits_type = if data_type == DataType::F64 { DataType::F64 } else { DataType::F32 };
    let softmax_type = DataType::F32;
    let scores = project_array::<V>(query.clone())?.dot(
        &project_array::<V>(expanded_key.clone())?,
        &DotDimensionNumbers::new(vec![3], vec![3], vec![0, 2], vec![0, 2]),
    );
    let scores = if data_type == logits_type { scores } else { scores.convert_element_type(logits_type)? };
    let mut scores = <V as ValueProjection<ArrayType>>::from_projected(scores);
    let scale = configuration.scale().unwrap_or(1.0 / (dimensions.head_dimension as f64).sqrt());
    let scale_value = fill_like(&scores, scale)?;
    scores = <V as ValueProjection<ArrayType>>::from_projected(
        project_array::<V>(scores)?.mul(&project_array::<V>(scale_value)?)?,
    );
    if let Some(bias) = &inputs.bias {
        let bias = project_array::<V>(normalize_attention_score_operand_ir(bias)?)?;
        let bias =
            if bias.r#type().data_type() == logits_type { bias } else { bias.convert_element_type(logits_type)? };
        let bias = broadcast_like::<V>(bias, &scores, &[0, 1, 2, 3])?;
        scores = <V as ValueProjection<ArrayType>>::from_projected(
            project_array::<V>(scores)?.add(&project_array::<V>(bias)?)?,
        );
    }
    let logits = apply_masks_ir(
        scores,
        inputs.mask.as_ref(),
        configuration,
        inputs.query_sequence_lengths.as_ref(),
        inputs.key_value_sequence_lengths.as_ref(),
    )?;
    let logits = project_array::<V>(logits)?;
    let logits = if logits_type == softmax_type { logits } else { logits.convert_element_type(softmax_type)? };
    Ok(PreparedAttention { query, key, expanded_key, expanded_value, logits, data_type, softmax_type, scale })
}

/// Evaluates portable attention in the mixed array IR using ordinary array operations and first-class extents.
///
/// This is the authoritative staged composition used by portable XLA lowering. Every shape-dependent constructor,
/// broadcast, and reshape takes its extents from ordinary dimension SSA values, so the same program supports static
/// and bounded-dynamic batch and sequence dimensions without backend-side attention mathematics.
pub fn dot_product_attention_ir_composition<V>(
    inputs: &AttentionInputs<V>,
    configuration: AttentionConfiguration,
) -> Result<(ArrayProjection<V>, Option<ArrayProjection<V>>), ProgramError>
where
    V: Value<Type = ArrayIrType>
        + AttentionIota
        + DimensionSize
        + DynamicBroadcast
        + DynamicReshape
        + ValueProjection<ArrayType>
        + ValueProjection<DimensionType>,
    ArrayProjection<V>: Value<Type = ArrayType>
        + Add
        + And
        + Broadcast
        + Compare
        + ConvertElementType
        + Div
        + Dot
        + Exp
        + Log
        + Mul
        + Reduce
        + Reshape
        + Select
        + StopGradient
        + Sub
        + Transpose,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Constant: From<DimensionValue>,
    <ArrayProjection<V> as Value>::DispatchDomain: Fill<f64, ArrayProjection<V>>,
{
    if configuration.dropout().is_some() {
        return Err(ProgramError::UnsupportedOperation {
            message: format!(
                "`{DOT_PRODUCT_ATTENTION_OPERATION_NAME}` dropout is only supported by the fused CUDA lowering"
            ),
        });
    }
    let operation = DotProductAttentionOperation::new(configuration, inputs.signature());
    let mut input_types = vec![
        <&ArrayType>::try_from(inputs.query.r#type().as_ref())?.clone(),
        <&ArrayType>::try_from(inputs.key.r#type().as_ref())?.clone(),
        <&ArrayType>::try_from(inputs.value.r#type().as_ref())?.clone(),
    ];
    input_types.extend(
        [
            inputs.bias.as_ref(),
            inputs.mask.as_ref(),
            inputs.query_sequence_lengths.as_ref(),
            inputs.key_value_sequence_lengths.as_ref(),
        ]
        .into_iter()
        .flatten()
        .map(|value| <&ArrayType>::try_from(value.r#type().as_ref()).cloned())
        .collect::<Result<Vec<_>, _>>()?,
    );
    operation.infer_output_types(input_types.as_slice(), &[])?;
    let query_rank = input_types[0].rank();
    let dimensions = validated_attention_operands(
        DOT_PRODUCT_ATTENTION_OPERATION_NAME,
        &input_types[0],
        &input_types[1],
        &input_types[2],
        inputs.bias.as_ref().map(|_| &input_types[3]),
        inputs.mask.as_ref().map(|_| &input_types[3 + usize::from(inputs.bias.is_some())]),
    )?;

    let prepared = prepare_attention_ir(inputs, configuration, &dimensions)?;
    let logits = <V as ValueProjection<ArrayType>>::from_projected(prepared.logits.clone());
    let maxima = prepared.logits.reduce(&[3], ReductionKind::Max);
    let maxima_broadcast = broadcast_like::<V>(maxima.clone(), &logits, &[0, 1, 2])?;
    let shifted = prepared.logits.sub(&project_array::<V>(maxima_broadcast)?)?;
    let exponentials = shifted.exp()?;
    let exponentials_ir = <V as ValueProjection<ArrayType>>::from_projected(exponentials.clone());
    let sums = exponentials.reduce(&[3], ReductionKind::Sum);
    let sums_broadcast = broadcast_like::<V>(sums.clone(), &exponentials_ir, &[0, 1, 2])?;
    let weights = exponentials.div(&project_array::<V>(sums_broadcast)?)?;
    let weights = if prepared.data_type == prepared.softmax_type {
        weights
    } else {
        weights.convert_element_type(prepared.data_type)?
    };
    let attended = weights.dot(
        &project_array::<V>(prepared.expanded_value)?,
        &DotDimensionNumbers::new(vec![3], vec![1], vec![0, 1], vec![0, 2]),
    );
    let mut output = attended.transpose([0, 2, 1, 3])?;
    if let Some(query_lengths) = &inputs.query_sequence_lengths {
        output = zero_query_rows_ir::<V>(output, query_lengths, 1)?;
    }
    let output = denormalize_attention_output_ir::<V>(output, query_rank)?;
    let activation_output = if configuration.return_residual() {
        let statistic = maxima.add(&sums.log()?)?;
        let statistic = if prepared.softmax_type == prepared.data_type {
            statistic
        } else {
            statistic.convert_element_type(prepared.data_type)?
        };
        let statistic = statistic.transpose([0, 2, 1])?;
        Some(denormalize_attention_output_ir::<V>(statistic, query_rank)?.stop_gradient())
    } else {
        None
    };
    Ok((output, activation_output))
}

/// Zeros query rows outside each batch item's logical sequence length.
fn zero_query_rows_ir<V>(
    value: ArrayProjection<V>,
    query_lengths: &V,
    row_axis: usize,
) -> Result<ArrayProjection<V>, ProgramError>
where
    V: Value<Type = ArrayIrType>
        + AttentionIota
        + DimensionSize
        + DynamicBroadcast
        + ValueProjection<ArrayType, Projected: Value<Type = ArrayType> + Compare + Select>,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <ArrayProjection<V> as Value>::DispatchDomain: Fill<f64, ArrayProjection<V>>,
{
    let value = <V as ValueProjection<ArrayType>>::from_projected(value);
    let rows = V::attention_iota(&value, row_axis)?;
    let lengths = query_lengths.clone().dynamic_broadcast(array_dimensions(&value)?.as_slice(), &[0])?;
    let in_range = project_array::<V>(rows)?.compare(&project_array::<V>(lengths)?, ComparisonDirection::LessThan)?;
    let zero = fill_like(&value, 0.0)?;
    <ArrayProjection<V> as Select>::select(&in_range, &project_array::<V>(value)?, &project_array::<V>(zero)?)
}

/// Evaluates the portable attention backward pass in the mixed array IR using the same first-class geometry and mask
/// construction as [`dot_product_attention_ir_composition`].
pub fn dot_product_attention_backward_ir_composition<V>(
    inputs: &AttentionInputs<V>,
    output: &V,
    activation: &V,
    output_cotangent: &V,
    configuration: AttentionConfiguration,
) -> Result<Vec<ArrayProjection<V>>, ProgramError>
where
    V: Value<Type = ArrayIrType>
        + AttentionIota
        + DimensionSize
        + DynamicBroadcast
        + DynamicReshape
        + ValueProjection<ArrayType>
        + ValueProjection<DimensionType>,
    ArrayProjection<V>: Value<Type = ArrayType>
        + Add
        + And
        + Broadcast
        + Compare
        + ConvertElementType
        + Dot
        + Exp
        + Mul
        + Reduce
        + Reshape
        + Select
        + Sub
        + Transpose,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Constant: From<DimensionValue>,
    <ArrayProjection<V> as Value>::DispatchDomain: Fill<f64, ArrayProjection<V>>,
{
    if configuration.dropout().is_some() {
        return Err(ProgramError::UnsupportedOperation {
            message: format!(
                "`{DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME}` dropout is only supported by the fused CUDA \
                 lowering"
            ),
        });
    }
    let operation = DotProductAttentionBackwardOperation::new(configuration, inputs.signature());
    let mut input_types = vec![
        <&ArrayType>::try_from(inputs.query.r#type().as_ref())?.clone(),
        <&ArrayType>::try_from(inputs.key.r#type().as_ref())?.clone(),
        <&ArrayType>::try_from(inputs.value.r#type().as_ref())?.clone(),
    ];
    input_types.extend(
        [
            inputs.bias.as_ref(),
            inputs.mask.as_ref(),
            inputs.query_sequence_lengths.as_ref(),
            inputs.key_value_sequence_lengths.as_ref(),
        ]
        .into_iter()
        .flatten()
        .map(|value| <&ArrayType>::try_from(value.r#type().as_ref()).cloned())
        .collect::<Result<Vec<_>, _>>()?,
    );
    input_types.extend([
        <&ArrayType>::try_from(output.r#type().as_ref())?.clone(),
        <&ArrayType>::try_from(activation.r#type().as_ref())?.clone(),
        <&ArrayType>::try_from(output_cotangent.r#type().as_ref())?.clone(),
    ]);
    operation.infer_output_types(input_types.as_slice(), &[])?;
    let input_ranks = [input_types[0].rank(), input_types[1].rank(), input_types[2].rank()];
    let dimensions = validated_attention_operands(
        DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME,
        &input_types[0],
        &input_types[1],
        &input_types[2],
        inputs.bias.as_ref().map(|_| &input_types[3]),
        inputs.mask.as_ref().map(|_| &input_types[3 + usize::from(inputs.bias.is_some())]),
    )?;
    let prepared = prepare_attention_ir(inputs, configuration, &dimensions)?;
    let output = normalize_attention_operand_ir(output)?;
    let output_cotangent = normalize_attention_operand_ir(output_cotangent)?;
    let logits = <V as ValueProjection<ArrayType>>::from_projected(prepared.logits.clone());
    let activation = normalize_attention_residual_ir(activation)?;
    let statistic = project_array::<V>(activation)?.transpose([0, 2, 1])?;
    let statistic = if prepared.softmax_type == prepared.data_type {
        statistic
    } else {
        statistic.convert_element_type(prepared.softmax_type)?
    };
    let statistic = broadcast_like::<V>(statistic, &logits, &[0, 1, 2])?;
    let weights = prepared.logits.sub(&project_array::<V>(statistic)?)?.exp()?;

    let output_cotangent = match &inputs.query_sequence_lengths {
        None => project_array::<V>(output_cotangent)?,
        Some(query_lengths) => zero_query_rows_ir::<V>(project_array::<V>(output_cotangent)?, query_lengths, 1)?,
    };
    let convert = |operand: ArrayProjection<V>| -> Result<ArrayProjection<V>, ProgramError> {
        if prepared.data_type == prepared.softmax_type {
            Ok(operand)
        } else {
            Ok(operand.convert_element_type(prepared.softmax_type)?)
        }
    };
    let softmax_query = convert(project_array::<V>(prepared.query.clone())?)?;
    let softmax_key = convert(project_array::<V>(prepared.expanded_key.clone())?)?;
    let softmax_value = convert(project_array::<V>(prepared.expanded_value)?)?;
    let softmax_output = convert(project_array::<V>(output)?)?;
    let softmax_output_cotangent = convert(output_cotangent)?;
    let weight_cotangents = softmax_output_cotangent
        .dot(&softmax_value, &DotDimensionNumbers::new(vec![3], vec![3], vec![0, 2], vec![0, 2]));
    let delta = softmax_output_cotangent
        .mul(&softmax_output)?
        .reduce(&[3], ReductionKind::Sum)
        .transpose([0, 2, 1])?;
    let weights_ir = <V as ValueProjection<ArrayType>>::from_projected(weights.clone());
    let delta = broadcast_like::<V>(delta, &weights_ir, &[0, 1, 2])?;
    let logit_cotangents = weights.mul(&weight_cotangents.sub(&project_array::<V>(delta)?)?)?;
    let logit_cotangents_ir = <V as ValueProjection<ArrayType>>::from_projected(logit_cotangents.clone());
    let scale_value = fill_like(&logit_cotangents_ir, prepared.scale)?;
    let scaled_logit_cotangents = logit_cotangents.mul(&project_array::<V>(scale_value)?)?;
    let mut query_cotangent = scaled_logit_cotangents
        .dot(&softmax_key, &DotDimensionNumbers::new(vec![3], vec![1], vec![0, 1], vec![0, 2]))
        .transpose([0, 2, 1, 3])?;
    if let Some(query_lengths) = &inputs.query_sequence_lengths {
        query_cotangent = zero_query_rows_ir::<V>(query_cotangent, query_lengths, 1)?;
    }
    let key_cotangent = scaled_logit_cotangents
        .dot(&softmax_query, &DotDimensionNumbers::new(vec![2], vec![1], vec![0, 1], vec![0, 2]))
        .transpose([0, 2, 1, 3])?;
    let value_cotangent = weights
        .dot(&softmax_output_cotangent, &DotDimensionNumbers::new(vec![2], vec![1], vec![0, 1], vec![0, 2]))
        .transpose([0, 2, 1, 3])?;

    let (key_cotangent, value_cotangent) = if dimensions.key_value_heads == dimensions.query_heads {
        (key_cotangent, value_cotangent)
    } else {
        let context = prepared.query.dispatch_domain();
        let group = dimensions.query_heads / dimensions.key_value_heads;
        let key_dimensions = array_dimensions(&prepared.key)?;
        let key_value_heads = context.lift(DimensionValue::constant(dimensions.key_value_heads)?.into())?;
        let group = context.lift(DimensionValue::constant(group)?.into())?;
        let head_dimension = context.lift(DimensionValue::constant(dimensions.head_dimension)?.into())?;
        let grouped_dimensions =
            [key_dimensions[0].clone(), key_dimensions[1].clone(), key_value_heads, group, head_dimension];
        let grouped_key =
            <V as ValueProjection<ArrayType>>::from_projected(key_cotangent).dynamic_reshape(&grouped_dimensions)?;
        let grouped_value =
            <V as ValueProjection<ArrayType>>::from_projected(value_cotangent).dynamic_reshape(&grouped_dimensions)?;
        (
            project_array::<V>(grouped_key)?.reduce(&[3], ReductionKind::Sum),
            project_array::<V>(grouped_value)?.reduce(&[3], ReductionKind::Sum),
        )
    };
    let cotangents = [query_cotangent, key_cotangent, value_cotangent]
        .into_iter()
        .zip(input_types[..3].iter().zip(input_ranks))
        .map(|(cotangent, (input_type, rank))| {
            let cotangent_data_type = input_type.cotangent().data_type();
            let cotangent = if cotangent.r#type().data_type() == cotangent_data_type {
                cotangent
            } else {
                cotangent.convert_element_type(cotangent_data_type)?
            };
            denormalize_attention_output_ir::<V>(cotangent, rank)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut cotangents = cotangents;
    if let Some(bias) = &inputs.bias {
        let bias_type = bias.r#type();
        let bias_type = <&ArrayType>::try_from(bias_type.as_ref())?;
        let bias_cotangent_type = bias_type.cotangent();
        if bias_cotangent_type.is_zero_space() {
            return Ok(cotangents);
        }
        let bias_dimensions = std::iter::repeat_n(Dimension::Static(1), 4 - bias_type.rank())
            .chain(bias_type.shape().dimensions().iter().cloned())
            .collect::<Vec<_>>();
        let logit_type = logit_cotangents.r#type();
        let reduce_axes = bias_dimensions
            .iter()
            .zip(logit_type.shape().dimensions())
            .enumerate()
            .filter_map(|(axis, (bias, logit))| (bias == &Dimension::Static(1) && logit != bias).then_some(axis))
            .collect::<Vec<_>>();
        let bias_cotangent = if reduce_axes.is_empty() {
            logit_cotangents
        } else {
            logit_cotangents.reduce(reduce_axes.as_slice(), ReductionKind::Sum)
        };
        let bias_dimensions = array_dimensions(bias)?;
        let bias_cotangent = <V as ValueProjection<ArrayType>>::from_projected(bias_cotangent)
            .dynamic_reshape(bias_dimensions.as_slice())?;
        let bias_cotangent = project_array::<V>(bias_cotangent)?;
        let bias_cotangent = if bias_cotangent_type.data_type() == prepared.softmax_type {
            bias_cotangent
        } else {
            bias_cotangent.convert_element_type(bias_cotangent_type.data_type())?
        };
        cotangents.push(bias_cotangent);
    }
    Ok(cotangents)
}
