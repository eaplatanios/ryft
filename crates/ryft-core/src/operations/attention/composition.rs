use crate::arrays::ir::ArrayIrValue;
use crate::arrays::{ArrayIrType, DimensionType};
use crate::batching::BatchingLevelExtent;
use crate::differentiation::DifferentiableType;
use crate::operations::constants::iota::IotaOperation;
use crate::operations::differentiation::stop_gradient::StopGradient;
use crate::operations::manipulation::{DynamicBroadcast, DynamicReshape};
use crate::programs::{Type, ValueDomainDispatch};

use super::*;

/// Type universe in which the portable attention bodies run, which determines how they obtain and construct the extents
/// of the arrays they shape (i.e., for broadcasts, reshapes, iotas, and fills). The bodies are written once over a
/// value `V` whose [`Type`](crate::Type) implements this trait, and they compute on its array view [`Self::ArrayValue`]:
///
///   - [`ArrayType`] implements it for homogeneous array values, whose extents are static host numbers. The array
///     view is the value itself, and every extent must be static, so attention over dynamically shaped arrays in a
///     homogeneous array family fails with an exact error. Such programs stage attention in an array IR family
///     instead, which is how backends support bounded-dynamic shapes (e.g., XLA tracers).
///   - [`ArrayIrType`] implements it for array IR values, whose extents are first-class dimension values. Every
///     shape-dependent constructor, broadcast, and reshape takes its extents from ordinary dimension values, so the
///     same body supports static and dynamic batch and sequence extents.
///
/// The implementations are on the type universes rather than on the values, which keeps them coherent for every value
/// type, and so a value participates exactly when its operations satisfy the bounds of its universe's implementation.
pub trait AttentionUniverse<V>: Type {
    /// Array view of `V` on which the attention mathematics computes.
    type ArrayValue: Value<Type = ArrayType>
        + Add
        + And
        + Broadcast
        + Compare
        + ConvertElementType
        + Div
        + Dot
        + Dropout
        + Exp
        + Log
        + Mul
        + Reduce
        + Reshape
        + Select
        + StopGradient
        + Sub
        + Transpose;

    /// Extent of one axis (e.g., `usize` for static extents or a dimension value for first-class extents).
    type Extent: Clone;

    /// Returns the array view of `value`.
    fn into_array(value: V) -> Result<Self::ArrayValue, ProgramError>;

    /// Returns the value whose array view is `array`.
    fn from_array(array: Self::ArrayValue) -> V;

    /// Returns the array type of `value`.
    fn array_type(value: &V) -> Result<ArrayType, ProgramError>;

    /// Returns the array type that `r#type` describes.
    fn to_array_type(r#type: &Self) -> Result<ArrayType, TypeError>;

    /// Returns the type of this universe that describes the array type `r#type`.
    fn from_array_type(r#type: ArrayType) -> Self;

    /// Returns the extent of the batching level whose extent is `extent`, constructed next to `value` (i.e., in its
    /// context).
    fn batch_extent(value: &V, extent: &BatchingLevelExtent<V>) -> Result<Self::Extent, ProgramError>;

    /// Returns the extent of `value` along `axis`.
    fn extent(value: &V, axis: usize) -> Result<Self::Extent, ProgramError>;

    /// Returns the constant extent `extent`, constructed next to `value` (i.e., in its context).
    fn constant_extent(value: &V, extent: usize) -> Result<Self::Extent, ProgramError>;

    /// Returns the product of two extents, constructed next to `value` (i.e., in its context).
    fn multiply_extents(value: &V, left: &Self::Extent, right: &Self::Extent) -> Result<Self::Extent, ProgramError>;

    /// Broadcasts `value` to the provided output extents, where `axes` names the output axis of every input axis.
    fn broadcast(value: &V, extents: &[Self::Extent], axes: &[usize]) -> Result<V, ProgramError>;

    /// Reshapes `value` to the provided output extents.
    fn reshape(value: &V, extents: &[Self::Extent]) -> Result<V, ProgramError>;

    /// Returns an `i32` iota with the shape of `target` that counts along `axis`.
    fn iota(target: &V, axis: usize) -> Result<V, ProgramError>;

    /// Returns `value` at the data type of `target`, broadcast to the shape of `target`.
    fn fill(target: &V, value: f64) -> Result<V, ProgramError>;
}

impl<A> AttentionUniverse<A> for ArrayType
where
    A: Value<Type = ArrayType>
        + Add
        + And
        + Broadcast
        + Compare
        + ConvertElementType
        + Div
        + Dot
        + Dropout
        + Exp
        + Log
        + Mul
        + Reduce
        + Reshape
        + Select
        + StopGradient
        + Sub
        + Transpose,
    A::Domain: Fill<f64, A> + Iota<A>,
{
    type ArrayValue = A;
    type Extent = usize;

    #[inline]
    fn into_array(value: A) -> Result<A, ProgramError> {
        Ok(value)
    }

    #[inline]
    fn from_array(array: A) -> A {
        array
    }

    #[inline]
    fn array_type(value: &A) -> Result<ArrayType, ProgramError> {
        Ok(value.r#type().into_owned())
    }

    #[inline]
    fn to_array_type(r#type: &ArrayType) -> Result<ArrayType, TypeError> {
        Ok(r#type.clone())
    }

    #[inline]
    fn from_array_type(r#type: ArrayType) -> ArrayType {
        r#type
    }

    fn batch_extent(_value: &A, extent: &BatchingLevelExtent<A>) -> Result<usize, ProgramError> {
        match extent {
            BatchingLevelExtent::Static(extent) => Ok(*extent),
            BatchingLevelExtent::Dynamic(_) => Err(TypeError::invalid(
                "attention over homogeneous arrays cannot be batched with a dynamic extent; use array IR values for \
                 dynamic extents",
            )
            .into()),
        }
    }

    fn extent(value: &A, axis: usize) -> Result<usize, ProgramError> {
        let r#type = value.r#type();
        let dimension = r#type.dimension(axis);
        dimension.value().ok_or_else(|| {
            TypeError::invalid(format!(
                "attention over homogeneous arrays requires static extents, but axis {axis} of `{}` has extent \
                 `{dimension}`; use array IR values for dynamic extents",
                r#type.as_ref(),
            ))
            .into()
        })
    }

    #[inline]
    fn constant_extent(_value: &A, extent: usize) -> Result<usize, ProgramError> {
        Ok(extent)
    }

    fn multiply_extents(_value: &A, left: &usize, right: &usize) -> Result<usize, ProgramError> {
        left.checked_mul(*right).ok_or_else(|| {
            TypeError::invalid(format!("attention extent product {left} × {right} overflows `usize`")).into()
        })
    }

    fn broadcast(value: &A, extents: &[usize], axes: &[usize]) -> Result<A, ProgramError> {
        // The output keeps the memory space and the placement of the input along the mapped axes, and its new axes are
        // replicated, because a broadcast cannot change manual variation.
        let r#type = value.r#type();
        let sharding = r#type
            .sharding()
            .map(|sharding| sharding.with_broadcasted_dimensions(extents.len(), axes))
            .transpose()
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        let output_type = ArrayType::new(r#type.data_type(), static_shape(extents))
            .with_memory(r#type.memory())
            .with_sharding(sharding)
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        Broadcast::broadcast(value, output_type, axes)
    }

    #[inline]
    fn reshape(value: &A, extents: &[usize]) -> Result<A, ProgramError> {
        value.reshape(static_shape(extents))
    }

    fn iota(target: &A, axis: usize) -> Result<A, ProgramError> {
        let output_type = target.r#type().into_owned().with_data_type(DataType::I32).with_layout(None);
        target.domain().iota(&output_type, axis)
    }

    fn fill(target: &A, value: f64) -> Result<A, ProgramError> {
        let target_type = target.r#type();
        let scalar = target.domain().fill(&ArrayType::scalar(target_type.data_type()), value)?;
        let extents = (0..target_type.rank())
            .map(|axis| <Self as AttentionUniverse<A>>::extent(target, axis))
            .collect::<Result<Vec<_>, _>>()?;
        <Self as AttentionUniverse<A>>::broadcast(&scalar, extents.as_slice(), &[])
    }
}

impl<V> AttentionUniverse<V> for ArrayIrType
where
    V: Value<Type = ArrayIrType>
        + AttentionIota
        + DimensionSize
        + DynamicBroadcast
        + DynamicReshape
        + ValueProjection<ArrayType>
        + ValueProjection<DimensionType>,
    <V as ValueProjection<ArrayType>>::Projected: Value<Type = ArrayType>
        + Add
        + And
        + Broadcast
        + Compare
        + ConvertElementType
        + Div
        + Dot
        + Dropout
        + Exp
        + Log
        + Mul
        + Reduce
        + Reshape
        + Select
        + StopGradient
        + Sub
        + Transpose,
    <<V as ValueProjection<ArrayType>>::Projected as Value>::Domain:
        Fill<f64, <V as ValueProjection<ArrayType>>::Projected>,
    V::Domain: Context<Type = ArrayIrType, Operation: From<DimensionMulOperation>> + DimensionConstant,
{
    type ArrayValue = <V as ValueProjection<ArrayType>>::Projected;
    type Extent = V;

    #[inline]
    fn into_array(value: V) -> Result<Self::ArrayValue, ProgramError> {
        ValueProjection::<ArrayType>::into_projected(value).map_err(Into::into)
    }

    #[inline]
    fn from_array(array: Self::ArrayValue) -> V {
        <V as ValueProjection<ArrayType>>::from_projected(array)
    }

    #[inline]
    fn array_type(value: &V) -> Result<ArrayType, ProgramError> {
        Ok(<&ArrayType>::try_from(value.r#type().as_ref())?.clone())
    }

    #[inline]
    fn to_array_type(r#type: &ArrayIrType) -> Result<ArrayType, TypeError> {
        Ok(<&ArrayType>::try_from(r#type)?.clone())
    }

    #[inline]
    fn from_array_type(r#type: ArrayType) -> ArrayIrType {
        r#type.into()
    }

    fn batch_extent(value: &V, extent: &BatchingLevelExtent<V>) -> Result<V, ProgramError> {
        match extent {
            BatchingLevelExtent::Static(extent) => value.domain().dimension_constant(*extent),
            BatchingLevelExtent::Dynamic(extent) => Ok(extent.clone()),
        }
    }

    #[inline]
    fn extent(value: &V, axis: usize) -> Result<V, ProgramError> {
        value.dimension_size(axis)
    }

    #[inline]
    fn constant_extent(value: &V, extent: usize) -> Result<V, ProgramError> {
        value.domain().dimension_constant(extent)
    }

    fn multiply_extents(value: &V, left: &V, right: &V) -> Result<V, ProgramError> {
        let left_type = left.r#type();
        let left_type = <&DimensionType>::try_from(left_type.as_ref())?;
        let right_type = right.r#type();
        let right_type = <&DimensionType>::try_from(right_type.as_ref())?;
        let mut outputs = value.domain().bind(
            DimensionMulOperation::new(left_type, right_type)?,
            Vec::new(),
            &[left.clone(), right.clone()],
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }

    #[inline]
    fn broadcast(value: &V, extents: &[V], axes: &[usize]) -> Result<V, ProgramError> {
        value.dynamic_broadcast(extents, axes)
    }

    #[inline]
    fn reshape(value: &V, extents: &[V]) -> Result<V, ProgramError> {
        value.clone().dynamic_reshape(extents)
    }

    #[inline]
    fn iota(target: &V, axis: usize) -> Result<V, ProgramError> {
        V::attention_iota(target, axis)
    }

    fn fill(target: &V, value: f64) -> Result<V, ProgramError> {
        let target_type = target.r#type();
        let target_type = <&ArrayType>::try_from(target_type.as_ref())?;
        let scalar = <Self as AttentionUniverse<V>>::into_array(target.clone())?
            .domain()
            .fill(&ArrayType::scalar(target_type.data_type()), value)?;
        let extents = (0..target_type.rank()).map(|axis| target.dimension_size(axis)).collect::<Result<Vec<_>, _>>()?;
        <Self as AttentionUniverse<V>>::from_array(scalar).dynamic_broadcast(extents.as_slice(), &[])
    }
}

/// Array view of a value of an [`AttentionUniverse`].
pub(super) type ArrayOf<V> = <<V as Typed>::Type as AttentionUniverse<V>>::ArrayValue;

/// Extent of a value of an [`AttentionUniverse`].
pub(super) type ExtentOf<V> = <<V as Typed>::Type as AttentionUniverse<V>>::Extent;

/// Private capability for constructing an iota with one mixed value's runtime shape.
pub trait AttentionIota: Value<Type = ArrayIrType> {
    /// Constructs an `i32` iota matching `target` and varying along `axis`.
    fn attention_iota(target: &Self, axis: usize) -> Result<Self, ProgramError>;
}

impl<
    V: Value<
            Dispatch = ValueDomainDispatch,
            Type = ArrayIrType,
            Domain: Context<Type = ArrayIrType, Operation: From<IotaOperation<ArrayType>>>,
        > + DimensionSize
        + ValueProjection<ArrayType>,
> AttentionIota for V
{
    fn attention_iota(target: &Self, axis: usize) -> Result<Self, ProgramError> {
        let target_type = target.r#type();
        let target_type = <&ArrayType>::try_from(target_type.as_ref())?;
        let output_type = target_type.clone().with_data_type(DataType::I32);
        let dimensions =
            (0..target_type.rank()).map(|axis| target.dimension_size(axis)).collect::<Result<Vec<_>, _>>()?;
        let dynamic_dimensions = output_type
            .shape()
            .dimensions()
            .iter()
            .enumerate()
            .filter_map(|(axis, dimension)| {
                matches!(dimension, Dimension::Dynamic(_)).then(|| dimensions[axis].clone())
            })
            .collect::<Vec<_>>();
        let mut outputs =
            target
                .domain()
                .bind(IotaOperation::new(output_type, axis)?, Vec::new(), dynamic_dimensions.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<A> AttentionIota for ArrayIrValue<A>
where
    A: Value<Type = ArrayType> + DimensionSize<usize>,
    A::Domain: Iota<A>,
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
        Ok(Self::Array(target.domain().iota(&output_type, axis)?))
    }
}

/// Returns the array view of `value`.
fn array<V: Value<Type: AttentionUniverse<V>>>(value: V) -> Result<ArrayOf<V>, ProgramError> {
    V::Type::into_array(value)
}

/// Returns the value whose array view is `array`.
fn value<V: Value<Type: AttentionUniverse<V>>>(array: ArrayOf<V>) -> V {
    V::Type::from_array(array)
}

/// Returns the extent of every axis of `value`.
pub(super) fn extents<V: Value<Type: AttentionUniverse<V>>>(value: &V) -> Result<Vec<ExtentOf<V>>, ProgramError> {
    (0..V::Type::array_type(value)?.rank()).map(|axis| V::Type::extent(value, axis)).collect()
}

/// Broadcasts one array to the exact shape of `target`, where `axes` names the output axis of every input axis.
fn broadcast_like<V: Value<Type: AttentionUniverse<V>>>(
    array: ArrayOf<V>,
    target: &V,
    axes: &[usize],
) -> Result<V, ProgramError> {
    V::Type::broadcast(&value::<V>(array), extents(target)?.as_slice(), axes)
}

/// Expands grouped key/value heads to the query heads through broadcast and reshape extents.
fn expand_key_value_heads<V: Value<Type: AttentionUniverse<V>>>(
    key_or_value: &V,
    dimensions: &AttentionDimensions,
) -> Result<V, ProgramError> {
    if dimensions.key_value_heads == dimensions.query_heads {
        return Ok(key_or_value.clone());
    }
    let group = dimensions.query_heads / dimensions.key_value_heads;
    let key_or_value_extents = extents(key_or_value)?;
    let key_value_heads = V::Type::constant_extent(key_or_value, dimensions.key_value_heads)?;
    let group = V::Type::constant_extent(key_or_value, group)?;
    let query_heads = V::Type::constant_extent(key_or_value, dimensions.query_heads)?;
    let expanded = V::Type::broadcast(
        key_or_value,
        &[
            key_or_value_extents[0].clone(),
            key_or_value_extents[1].clone(),
            key_value_heads,
            group,
            key_or_value_extents[3].clone(),
        ],
        &[0, 1, 2, 4],
    )?;
    V::Type::reshape(
        &expanded,
        &[
            key_or_value_extents[0].clone(),
            key_or_value_extents[1].clone(),
            query_heads,
            key_or_value_extents[3].clone(),
        ],
    )
}

/// Adds the implicit batch axis used to evaluate one unbatched `TNH` attention input (i.e., a query, key, value,
/// forward output, or output cotangent).
fn normalize_attention_input<V: Value<Type: AttentionUniverse<V>>>(input: &V) -> Result<V, ProgramError> {
    if V::Type::array_type(input)?.rank() == 4 {
        return Ok(input.clone());
    }
    let mut input_extents = extents(input)?;
    input_extents.insert(0, V::Type::constant_extent(input, 1)?);
    V::Type::reshape(input, input_extents.as_slice())
}

/// Prepends singleton axes until one bias or mask input has the normalized `BNTS` score rank.
fn normalize_attention_score_input<V: Value<Type: AttentionUniverse<V>>>(input: &V) -> Result<V, ProgramError> {
    let rank = V::Type::array_type(input)?.rank();
    if rank == 4 {
        return Ok(input.clone());
    }
    let singleton = V::Type::constant_extent(input, 1)?;
    let mut input_extents = vec![singleton; 4 - rank];
    input_extents.extend(extents(input)?);
    V::Type::reshape(input, input_extents.as_slice())
}

/// Removes the implicit batch axis after evaluating unbatched attention.
fn denormalize_attention_output<V: Value<Type: AttentionUniverse<V>>>(
    output: ArrayOf<V>,
    rank: usize,
) -> Result<ArrayOf<V>, ProgramError> {
    if rank == 4 {
        return Ok(output);
    }
    let output = value::<V>(output);
    let output_extents = extents(&output)?;
    array(V::Type::reshape(&output, &output_extents[1..])?)
}

/// Applies all built-in and sequence-length masks to the attention logits.
fn apply_masks<V: Value<Type: AttentionUniverse<V>>>(
    scores: V,
    mask: Option<&V>,
    configuration: AttentionConfiguration,
    query_sequence_lengths: Option<&V>,
    key_value_sequence_lengths: Option<&V>,
) -> Result<V, ProgramError> {
    if mask.is_none()
        && !configuration.causal()
        && configuration.local_window().is_none()
        && query_sequence_lengths.is_none()
        && key_value_sequence_lengths.is_none()
    {
        return Ok(scores);
    }
    let columns = V::Type::iota(&scores, 3)?;
    let mut visible = match mask {
        None => None,
        Some(mask) => {
            let mask = array(normalize_attention_score_input(mask)?)?;
            Some(array(broadcast_like::<V>(mask, &scores, &[0, 1, 2, 3])?)?)
        }
    };
    if configuration.causal() {
        let rows = array(V::Type::iota(&scores, 2)?)?;
        let causal = array(columns.clone())?.compare(&rows, ComparisonDirection::LessThanOrEqual)?;
        visible = Some(match visible {
            None => causal,
            Some(visible) => visible.and(&causal)?,
        });
    }
    if let Some((left, right)) = configuration.local_window() {
        let rows = V::Type::iota(&scores, 2)?;
        let columns = array(columns.clone())?;
        let left = array(V::Type::fill(&rows, left as f64)?)?;
        let right = array(V::Type::fill(&rows, right as f64)?)?;
        let rows = array(rows)?;
        let lower = rows.sub(&left)?;
        let upper = rows.add(&right)?;
        let in_window = columns
            .compare(&lower, ComparisonDirection::GreaterThanOrEqual)?
            .and(&columns.compare(&upper, ComparisonDirection::LessThanOrEqual)?)?;
        visible = Some(match visible {
            None => in_window,
            Some(visible) => visible.and(&in_window)?,
        });
    }
    if let Some(lengths) = query_sequence_lengths {
        let rows = V::Type::iota(&scores, 2)?;
        let bounds = V::Type::broadcast(lengths, extents(&rows)?.as_slice(), &[0])?;
        let in_range = array(rows)?.compare(&array(bounds)?, ComparisonDirection::LessThan)?;
        visible = Some(match visible {
            None => in_range,
            Some(visible) => visible.and(&in_range)?,
        });
    }
    if let Some(lengths) = key_value_sequence_lengths {
        let bounds = V::Type::broadcast(lengths, extents(&columns)?.as_slice(), &[0])?;
        let in_range = array(columns)?.compare(&array(bounds)?, ComparisonDirection::LessThan)?;
        visible = Some(match visible {
            None => in_range,
            Some(visible) => visible.and(&in_range)?,
        });
    }
    let score_type = array(scores.clone())?.r#type().into_owned();
    let large_negative = if score_type.data_type() == DataType::F64 { -0.7 * f64::MAX } else { -0.7 * f32::MAX as f64 };
    let masked = array(V::Type::fill(&scores, large_negative)?)?;
    Ok(value::<V>(visible.unwrap().select(&array(scores)?, &masked)?))
}

/// Shared normalized inputs and masked logits consumed by the forward and backward attention bodies.
struct PreparedAttention<V: Value<Type: AttentionUniverse<V>>> {
    /// Normalized query.
    query: V,

    /// Normalized key.
    key: V,

    /// Normalized key expanded from key/value heads to query heads.
    expanded_key: V,

    /// Normalized value expanded from key/value heads to query heads.
    expanded_value: V,

    /// Masked logits converted to the data type used by softmax.
    logits: ArrayOf<V>,

    /// Shared query/key/value data type.
    data_type: DataType,

    /// Data type used by softmax and its residual statistic.
    softmax_type: DataType,

    /// Explicit or default score scale.
    scale: f64,
}

/// Normalizes attention inputs and constructs the masked score logits shared by the forward and backward bodies.
fn prepare_attention<V: Value<Type: AttentionUniverse<V>>>(
    inputs: &AttentionInputs<V>,
    configuration: AttentionConfiguration,
    dimensions: &AttentionDimensions,
) -> Result<PreparedAttention<V>, ProgramError> {
    let query = normalize_attention_input(&inputs.query)?;
    let key = normalize_attention_input(&inputs.key)?;
    let value_input = normalize_attention_input(&inputs.value)?;
    let expanded_key = expand_key_value_heads(&key, dimensions)?;
    let expanded_value = expand_key_value_heads(&value_input, dimensions)?;
    let data_type = dimensions.data_type;
    // Dot products, scaling, bias, and masking promote low-precision inputs to `f32`, while `f64` remains `f64`.
    // The stable softmax itself always runs in `f32`, matching the public attention contract.
    let logits_type = if data_type == DataType::F64 { DataType::F64 } else { DataType::F32 };
    let softmax_type = DataType::F32;
    let scores = array(query.clone())?
        .dot(&array(expanded_key.clone())?, &DotDimensionNumbers::new(vec![3], vec![3], vec![0, 2], vec![0, 2]))?;
    let scores = if data_type == logits_type { scores } else { scores.convert_element_type(logits_type)? };
    let mut scores = value::<V>(scores);
    let scale = configuration.scale().unwrap_or(1.0 / (dimensions.head_dimension as f64).sqrt());
    let scale_value = array(V::Type::fill(&scores, scale)?)?;
    scores = value::<V>(array(scores)?.mul(&scale_value)?);
    if let Some(bias) = &inputs.bias {
        let bias = array(normalize_attention_score_input(bias)?)?;
        let bias =
            if bias.r#type().data_type() == logits_type { bias } else { bias.convert_element_type(logits_type)? };
        let bias = array(broadcast_like::<V>(bias, &scores, &[0, 1, 2, 3])?)?;
        scores = value::<V>(array(scores)?.add(&bias)?);
    }
    let logits = apply_masks(
        scores,
        inputs.mask.as_ref(),
        configuration,
        inputs.query_sequence_lengths.as_ref(),
        inputs.key_value_sequence_lengths.as_ref(),
    )?;
    let logits = array(logits)?;
    let logits = if logits_type == softmax_type { logits } else { logits.convert_element_type(softmax_type)? };
    Ok(PreparedAttention { query, key, expanded_key, expanded_value, logits, data_type, softmax_type, scale })
}

/// Applies the configured dropout to `probabilities`, whose shape is the normalized `BNTS` score shape, using the
/// dropout random state of `inputs` (refer to [`AttentionInputs::dropout_state`]). The forward and backward bodies call
/// this with arrays of the same shape and the same state, so they apply the same mask.
fn apply_dropout<V: Value<Type: AttentionUniverse<V>>>(
    probabilities: ArrayOf<V>,
    inputs: &AttentionInputs<V>,
    configuration: AttentionConfiguration,
) -> Result<ArrayOf<V>, ProgramError> {
    let Some(rate) = configuration.dropout() else {
        return Ok(probabilities);
    };
    let state = inputs.dropout_state.clone().map(array).transpose()?;
    Ok(probabilities.dropout(rate, state.as_ref())?.0)
}

/// Returns the canonical input types of `inputs`.
fn attention_input_types<V: Value<Type: AttentionUniverse<V>>>(
    inputs: &AttentionInputs<V>,
) -> Result<Vec<ArrayType>, ProgramError> {
    inputs.clone().into_values().iter().map(V::Type::array_type).collect()
}

/// Evaluates portable scaled dot-product attention using ordinary array operations, returning the attended output
/// together with the log-sum-exp residual when `configuration` requests it.
///
/// This is the body of the forward named attention definition (refer to [`AttentionKey`]). Every shape-dependent
/// constructor, broadcast, and reshape takes its extents from the value's [`AttentionUniverse`], so the same body
/// supports static extents in homogeneous array families and static and dynamic extents in array IR families without
/// backend-side attention mathematics.
pub fn dot_product_attention_ir_composition<V: Value<Type: AttentionUniverse<V>>>(
    inputs: &AttentionInputs<V>,
    configuration: AttentionConfiguration,
) -> Result<(ArrayOf<V>, Option<ArrayOf<V>>), ProgramError> {
    let input_types = attention_input_types(inputs)?;
    attention_forward_output_types(
        DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
        configuration,
        inputs.signature(),
        input_types.as_slice(),
    )?;
    let query_rank = input_types[0].rank();
    let dimensions = validated_attention_inputs(
        DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
        &input_types[0],
        &input_types[1],
        &input_types[2],
        inputs.bias.as_ref().map(|_| &input_types[3]),
        inputs.mask.as_ref().map(|_| &input_types[3 + usize::from(inputs.bias.is_some())]),
    )?;

    let prepared = prepare_attention(inputs, configuration, &dimensions)?;
    let logits = value::<V>(prepared.logits.clone());
    let maxima = prepared.logits.reduce(&[3], ReductionKind::Max)?;
    let maxima_broadcast = array(broadcast_like::<V>(maxima.clone(), &logits, &[0, 1, 2])?)?;
    let shifted = prepared.logits.sub(&maxima_broadcast)?;
    let exponentials = shifted.exp()?;
    let exponentials_value = value::<V>(exponentials.clone());
    let sums = exponentials.reduce(&[3], ReductionKind::Sum)?;
    let sums_broadcast = array(broadcast_like::<V>(sums.clone(), &exponentials_value, &[0, 1, 2])?)?;
    let weights = apply_dropout(exponentials.div(&sums_broadcast)?, inputs, configuration)?;
    let weights = if prepared.data_type == prepared.softmax_type {
        weights
    } else {
        weights.convert_element_type(prepared.data_type)?
    };
    let attended = weights
        .dot(&array(prepared.expanded_value)?, &DotDimensionNumbers::new(vec![3], vec![1], vec![0, 1], vec![0, 2]))?;
    let mut output = attended.transpose([0, 2, 1, 3])?;
    if let Some(query_lengths) = &inputs.query_sequence_lengths {
        output = zero_query_rows::<V>(output, query_lengths, 1)?;
    }
    let output = denormalize_attention_output::<V>(output, query_rank)?;
    let activation_output = if configuration.return_residual() {
        let statistic = maxima.add(&sums.log()?)?;
        let statistic = if prepared.softmax_type == prepared.data_type {
            statistic
        } else {
            statistic.convert_element_type(prepared.data_type)?
        };
        let statistic = statistic.transpose([0, 2, 1])?;
        Some(denormalize_attention_output::<V>(statistic, query_rank)?.stop_gradient()?)
    } else {
        None
    };
    Ok((output, activation_output))
}

/// Zeros query rows outside each batch item's logical sequence length.
fn zero_query_rows<V: Value<Type: AttentionUniverse<V>>>(
    rows_value: ArrayOf<V>,
    query_lengths: &V,
    row_axis: usize,
) -> Result<ArrayOf<V>, ProgramError> {
    let rows_value = value::<V>(rows_value);
    let rows = V::Type::iota(&rows_value, row_axis)?;
    let lengths = V::Type::broadcast(query_lengths, extents(&rows_value)?.as_slice(), &[0])?;
    let in_range = array(rows)?.compare(&array(lengths)?, ComparisonDirection::LessThan)?;
    let zero = array(V::Type::fill(&rows_value, 0.0)?)?;
    in_range.select(&array(rows_value)?, &zero)
}

/// Evaluates the portable attention backward pass using the same extents and mask construction as
/// [`dot_product_attention_ir_composition`], returning the cotangents of the query, key, and value, followed by that of
/// a present bias whose cotangent space is not zero-dimensional. This is the body of the backward named attention
/// definition (refer to [`AttentionKey`]), whose inputs are the forward inputs followed by the attended output, its
/// `TN` or `BTN` log-sum-exp residual, and the output cotangent. Boolean masks, integer sequence lengths, dropout
/// states, and a bias whose cotangent space is zero-dimensional have no cotangent outputs.
///
/// The body recomputes the normalized probabilities `P` from the masked logits in `f32`, which avoids the precision
/// that rounding the output or the residual to a narrower query data type loses, and so the output and the residual
/// are validated inputs for backend implementations whose numerical values this body does not use. It computes
///
/// ```text
/// dV = transpose(P) * dO
/// dW = dO * transpose(V)
/// C = dW - dW[pivot]
/// dS = P * (C - sum(P * C))
/// dQ = scale * dS * K
/// dK = scale * transpose(dS) * Q,
/// ```
///
/// where the pivot is the first maximum logit of each row, whose `dS` entry is replaced by the negative sum of all
/// other entries, which preserves the dominant derivative when its probability rounds to one. Grouped query heads are
/// reduced back to each key/value head, a broadcast bias cotangent is reduced to the shape of its bias, and padded
/// query rows are zeroed. With dropout, `P` is the dropped probabilities in `dV`, and the cotangent of the dropped
/// probabilities is mapped back through the same dropout mask, which the same random state and probability shape
/// reproduce. Higher-order derivatives differentiate this body directly.
pub fn dot_product_attention_backward_ir_composition<V: Value<Type: AttentionUniverse<V>>>(
    inputs: &AttentionInputs<V>,
    output: &V,
    activation: &V,
    output_cotangent: &V,
    configuration: AttentionConfiguration,
) -> Result<Vec<ArrayOf<V>>, ProgramError> {
    let mut input_types = attention_input_types(inputs)?;
    input_types.extend([
        V::Type::array_type(output)?,
        V::Type::array_type(activation)?,
        V::Type::array_type(output_cotangent)?,
    ]);
    attention_backward_output_types(
        DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME,
        configuration,
        inputs.signature(),
        input_types.as_slice(),
    )?;
    let input_ranks = [input_types[0].rank(), input_types[1].rank(), input_types[2].rank()];
    let dimensions = validated_attention_inputs(
        DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME,
        &input_types[0],
        &input_types[1],
        &input_types[2],
        inputs.bias.as_ref().map(|_| &input_types[3]),
        inputs.mask.as_ref().map(|_| &input_types[3 + usize::from(inputs.bias.is_some())]),
    )?;
    let prepared = prepare_attention(inputs, configuration, &dimensions)?;
    let output_cotangent = normalize_attention_input(output_cotangent)?;
    let logits = value::<V>(prepared.logits.clone());
    // The public residual and output have the query data type, which can lose the normalization correction in
    // narrow formats. Recompute probabilities in the softmax type instead of recovering them from those values.
    let maxima = prepared.logits.reduce(&[3], ReductionKind::Max)?;
    let maxima = array(broadcast_like::<V>(maxima, &logits, &[0, 1, 2])?)?;
    let exponentials = prepared.logits.sub(&maxima)?.exp()?;
    let sums = exponentials.reduce(&[3], ReductionKind::Sum)?;
    let sums = array(broadcast_like::<V>(sums, &logits, &[0, 1, 2])?)?;
    let weights = exponentials.div(&sums)?;

    // Pick exactly one maximum, including when logits tie. Keeping the pivot out of the final sum avoids
    // subtracting two nearly equal dominant contributions. The integer constants cannot carry tangents.
    let columns = V::Type::iota(&logits, 3)?;
    let column_array = array(columns.clone())?;
    let sentinel = array(V::Type::fill(&columns, i32::MAX as f64)?)?;
    let is_maximum = prepared.logits.compare(&maxima, ComparisonDirection::Equal)?;
    let pivot = is_maximum.select(&column_array, &sentinel)?.reduce(&[3], ReductionKind::Min)?;
    let pivot = array(broadcast_like::<V>(pivot, &logits, &[0, 1, 2])?)?;
    let is_pivot = column_array.compare(&pivot, ComparisonDirection::Equal)?;

    let output_cotangent = match &inputs.query_sequence_lengths {
        None => array(output_cotangent)?,
        Some(query_lengths) => zero_query_rows::<V>(array(output_cotangent)?, query_lengths, 1)?,
    };
    let convert = |array: ArrayOf<V>| -> Result<ArrayOf<V>, ProgramError> {
        if prepared.data_type == prepared.softmax_type {
            Ok(array)
        } else {
            Ok(array.convert_element_type(prepared.softmax_type)?)
        }
    };
    let softmax_query = convert(array(prepared.query.clone())?)?;
    let softmax_key = convert(array(prepared.expanded_key.clone())?)?;
    let softmax_value = convert(array(prepared.expanded_value)?)?;
    let softmax_output_cotangent = convert(output_cotangent)?;

    // With dropout, the output uses the dropped probabilities `D = dropout(P)`, so the value cotangent uses `D` and the
    // cotangent of `P` is the cotangent of `D` through the same mask.
    let dropped_weights = apply_dropout(weights.clone(), inputs, configuration)?;
    let weight_cotangents = apply_dropout(
        softmax_output_cotangent
            .dot(&softmax_value, &DotDimensionNumbers::new(vec![3], vec![3], vec![0, 2], vec![0, 2]))?,
        inputs,
        configuration,
    )?;
    let zero = array(V::Type::fill(&logits, 0.0)?)?;
    let pivot_cotangent = is_pivot.select(&weight_cotangents, &zero)?.reduce(&[3], ReductionKind::Sum)?;
    let pivot_cotangent = array(broadcast_like::<V>(pivot_cotangent, &logits, &[0, 1, 2])?)?;
    // Center the incoming weight cotangents before their weighted sum. This preserves a tiny dominant derivative
    // and gives exact zero when every weight cotangent is equal, even if the rounded probabilities do not sum to one.
    let centered = weight_cotangents.sub(&pivot_cotangent)?;
    let delta = weights.mul(&centered)?.reduce(&[3], ReductionKind::Sum)?;
    let delta = array(broadcast_like::<V>(delta, &logits, &[0, 1, 2])?)?;
    let logit_cotangents = weights.mul(&centered.sub(&delta)?)?;
    let other_cotangents = is_pivot.select(&zero, &logit_cotangents)?;
    let pivot_cotangent = other_cotangents.reduce(&[3], ReductionKind::Sum)?;
    let pivot_cotangent = array(broadcast_like::<V>(pivot_cotangent, &logits, &[0, 1, 2])?)?;
    let logit_cotangents = is_pivot.select(&zero.sub(&pivot_cotangent)?, &other_cotangents)?;
    let logit_cotangents_value = value::<V>(logit_cotangents.clone());
    let scale_value = array(V::Type::fill(&logit_cotangents_value, prepared.scale)?)?;
    let scaled_logit_cotangents = logit_cotangents.mul(&scale_value)?;
    let mut query_cotangent = scaled_logit_cotangents
        .dot(&softmax_key, &DotDimensionNumbers::new(vec![3], vec![1], vec![0, 1], vec![0, 2]))?
        .transpose([0, 2, 1, 3])?;
    if let Some(query_lengths) = &inputs.query_sequence_lengths {
        query_cotangent = zero_query_rows::<V>(query_cotangent, query_lengths, 1)?;
    }
    let key_cotangent = scaled_logit_cotangents
        .dot(&softmax_query, &DotDimensionNumbers::new(vec![2], vec![1], vec![0, 1], vec![0, 2]))?
        .transpose([0, 2, 1, 3])?;
    let value_cotangent = dropped_weights
        .dot(&softmax_output_cotangent, &DotDimensionNumbers::new(vec![2], vec![1], vec![0, 1], vec![0, 2]))?
        .transpose([0, 2, 1, 3])?;

    let (key_cotangent, value_cotangent) = if dimensions.key_value_heads == dimensions.query_heads {
        (key_cotangent, value_cotangent)
    } else {
        let group = dimensions.query_heads / dimensions.key_value_heads;
        let key_extents = extents(&prepared.key)?;
        let key_value_heads = V::Type::constant_extent(&prepared.key, dimensions.key_value_heads)?;
        let group = V::Type::constant_extent(&prepared.key, group)?;
        let head_dimension = V::Type::constant_extent(&prepared.key, dimensions.head_dimension)?;
        let grouped_extents = [key_extents[0].clone(), key_extents[1].clone(), key_value_heads, group, head_dimension];
        let grouped_key = V::Type::reshape(&value::<V>(key_cotangent), &grouped_extents)?;
        let grouped_value = V::Type::reshape(&value::<V>(value_cotangent), &grouped_extents)?;
        (array(grouped_key)?.reduce(&[3], ReductionKind::Sum)?, array(grouped_value)?.reduce(&[3], ReductionKind::Sum)?)
    };
    let mut cotangents = [query_cotangent, key_cotangent, value_cotangent]
        .into_iter()
        .zip(input_types[..3].iter().zip(input_ranks))
        .map(|(cotangent, (input_type, rank))| {
            let cotangent_data_type = input_type.cotangent()?.data_type();
            let cotangent = if cotangent.r#type().data_type() == cotangent_data_type {
                cotangent
            } else {
                cotangent.convert_element_type(cotangent_data_type)?
            };
            denormalize_attention_output::<V>(cotangent, rank)
        })
        .collect::<Result<Vec<_>, ProgramError>>()?;
    if let Some(bias) = &inputs.bias {
        let bias_type = V::Type::array_type(bias)?;
        let bias_cotangent_type = bias_type.cotangent()?;
        if bias_cotangent_type.is_zero_space() {
            return Ok(cotangents);
        }
        let bias_dimensions = std::iter::repeat_n(Dimension::Static(1), 4 - bias_type.rank())
            .chain(bias_type.shape().dimensions().iter().cloned())
            .collect::<Vec<_>>();
        let logit_type = logit_cotangents.r#type().into_owned();
        let reduce_axes = bias_dimensions
            .iter()
            .zip(logit_type.shape().dimensions())
            .enumerate()
            .filter_map(|(axis, (bias, logit))| (bias == &Dimension::Static(1) && logit != bias).then_some(axis))
            .collect::<Vec<_>>();
        let bias_cotangent = if reduce_axes.is_empty() {
            logit_cotangents
        } else {
            logit_cotangents.reduce(reduce_axes.as_slice(), ReductionKind::Sum)?
        };
        let bias_cotangent = array(V::Type::reshape(&value::<V>(bias_cotangent), extents(bias)?.as_slice())?)?;
        let bias_cotangent = if bias_cotangent_type.data_type() == prepared.softmax_type {
            bias_cotangent
        } else {
            bias_cotangent.convert_element_type(bias_cotangent_type.data_type())?
        };
        cotangents.push(bias_cotangent);
    }
    Ok(cotangents)
}
