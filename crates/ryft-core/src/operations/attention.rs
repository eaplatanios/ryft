use std::fmt::Display;

use crate::backends::scalars::Scalar;
use crate::batching::{
    ArrayBatch, BatchAxis, BatchableOperation, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    DifferentiableOperation, DifferentiationDriver, DifferentiationDual, DifferentiationError,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_non_transposable_operation};
use crate::operations::compare::{Compare, ComparisonDirection};
use crate::operations::constants::{Fill, Iota};
use crate::operations::control_flow::Select;
use crate::operations::manipulation::{Broadcast, ConvertElementType, Reshape, Transpose};
use crate::operations::math::{Div, Dot, DotDimensionNumbers, Exp, Mul, Reduce, ReductionKind, Sub};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{TypeError, Typed};
use crate::programs::{ProgramError, Value};
use crate::types::{ArrayType, DataType, Shape, Size};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`DotProductAttentionOperation`].
pub const DOT_PRODUCT_ATTENTION_OPERATION_NAME: &str = "dot_product_attention";

/// Built-in attention mask applied to the score matrix of a [`DotProductAttentionOperation`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AttentionMask {
    /// No masking: every query position attends to every key/value position.
    None,

    /// Causal masking: query position `i` attends only to key/value positions `j <= i`, the autoregressive
    /// decoder convention.
    Causal,
}

impl Display for AttentionMask {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => formatter.write_str("none"),
            Self::Causal => formatter.write_str("causal"),
        }
    }
}

/// Primitive representing scaled dot-product attention — the analogue of [JAX's
/// `jax.nn.dot_product_attention`](https://docs.jax.dev/en/latest/_autosummary/jax.nn.dot_product_attention.html).
/// The three operands use the `BTNH` logical layout: `query [batch, q_seq, heads, head_dim]` and `key`/`value`
/// `[batch, kv_seq, heads, head_dim]`, all carrying one shared floating-point data type, and the output is
/// `[batch, q_seq, heads, head_dim]` at that same type. Semantically the operation computes
/// `softmax(scale · query · keyᵀ + mask) · value` per batch item and head, with the softmax running at `f32` for
/// every operand type narrower than `f32` (`f64` operands keep an `f64` softmax) and an optional built-in causal
/// [`AttentionMask`] — which is exactly how the reference array backend and the portable XLA lowering evaluate it
/// (see [`dot_product_attention_composition`]). On CUDA targets, the XLA lowering instead emits the
/// `__cudnn$fmhaSoftmax` custom call, reaching cuDNN's fused flash-attention kernels.
///
/// Grouped-query attention (fewer key/value heads than query heads) is not supported yet, and neither are bias
/// operands, dropout, or variable-sequence-length masks. The operation is inference-oriented: differentiating
/// reports an error directing users to differentiate an explicit attention composition instead. Batching folds one
/// mapped axis into the batch dimension and reuses the same operation (attention is batch-parallel) — refer to the
/// batching rule documentation on this operation.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DotProductAttentionOperation {
    /// Multiplier applied to the attention scores before masking and softmax (typically `1 / sqrt(head_dim)`).
    scale: f64,

    /// Built-in mask applied to the attention scores before the softmax.
    mask: AttentionMask,
}

impl DotProductAttentionOperation {
    /// Creates a new [`DotProductAttentionOperation`] with the provided score scale and mask.
    #[inline]
    pub fn new(scale: f64, mask: AttentionMask) -> Self {
        Self { scale, mask }
    }

    /// Returns the multiplier applied to the attention scores before masking and softmax.
    #[inline]
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Returns the built-in mask applied to the attention scores before the softmax.
    #[inline]
    pub fn mask(&self) -> AttentionMask {
        self.mask
    }
}

impl Display for DotProductAttentionOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Operation::<ArrayType>::render(self, formatter, 0)
    }
}

impl Operation<ArrayType> for DotProductAttentionOperation {
    #[inline]
    fn name(&self) -> &'static str {
        DOT_PRODUCT_ATTENTION_OPERATION_NAME
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, DOT_PRODUCT_ATTENTION_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("scale", &self.scale)?;
            operation.field("mask", &self.mask)
        })
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 3, TypeError);
        let query = static_attention_dimensions("query", &input_types[0])?;
        let key = static_attention_dimensions("key", &input_types[1])?;
        let value = static_attention_dimensions("value", &input_types[2])?;
        let data_type = input_types[0].data_type();
        if !data_type_is_float(data_type) {
            return Err(TypeError {
                message: format!(
                    "'{DOT_PRODUCT_ATTENTION_OPERATION_NAME}' requires floating-point operands but got data type \
                     {data_type}"
                ),
            });
        }
        for (descriptor, dimensions, input_type) in [("key", &key, &input_types[1]), ("value", &value, &input_types[2])]
        {
            if input_type.data_type() != data_type {
                return Err(TypeError {
                    message: format!(
                        "'{DOT_PRODUCT_ATTENTION_OPERATION_NAME}' {descriptor} data type {} does not match the query \
                         data type {data_type}",
                        input_type.data_type(),
                    ),
                });
            }
            if dimensions[0] != query[0] {
                return Err(TypeError {
                    message: format!(
                        "'{DOT_PRODUCT_ATTENTION_OPERATION_NAME}' {descriptor} batch dimension ({}) does not match \
                         the query batch dimension ({})",
                        dimensions[0], query[0],
                    ),
                });
            }
            if dimensions[2] != query[2] {
                return Err(TypeError {
                    message: format!(
                        "'{DOT_PRODUCT_ATTENTION_OPERATION_NAME}' {descriptor} heads dimension ({}) does not match \
                         the query heads dimension ({}); grouped-query attention is not supported yet",
                        dimensions[2], query[2],
                    ),
                });
            }
            if dimensions[3] != query[3] {
                return Err(TypeError {
                    message: format!(
                        "'{DOT_PRODUCT_ATTENTION_OPERATION_NAME}' {descriptor} head dimension ({}) does not match \
                         the query head dimension ({})",
                        dimensions[3], query[3],
                    ),
                });
            }
        }
        if value[1] != key[1] {
            return Err(TypeError {
                message: format!(
                    "'{DOT_PRODUCT_ATTENTION_OPERATION_NAME}' value sequence dimension ({}) does not match the key \
                     sequence dimension ({})",
                    value[1], key[1],
                ),
            });
        }
        for input_type in input_types {
            if !input_type.unreduced_axes().is_empty() {
                return Err(TypeError {
                    message: format!("'{DOT_PRODUCT_ATTENTION_OPERATION_NAME}' does not support unreduced operands"),
                });
            }
        }
        Ok(vec![ArrayType::new(data_type, Shape::new(query.iter().map(|&size| Size::Static(size)).collect()))])
    }
}

/// Returns whether `data_type` is a (real) floating-point data type.
fn data_type_is_float(data_type: DataType) -> bool {
    matches!(
        data_type,
        DataType::F4E2M1FN
            | DataType::F6E2M3FN
            | DataType::F6E3M2FN
            | DataType::F8E3M4
            | DataType::F8E4M3
            | DataType::F8E4M3FN
            | DataType::F8E4M3FNUZ
            | DataType::F8E4M3B11FNUZ
            | DataType::F8E5M2
            | DataType::F8E5M2FNUZ
            | DataType::F8E8M0FNU
            | DataType::BF16
            | DataType::F16
            | DataType::F32
            | DataType::F64
    )
}

/// Returns the static `[batch, sequence, heads, head_dim]` dimensions of a [`DotProductAttentionOperation`]
/// operand type, rejecting dynamic shapes and any rank other than 4.
fn static_attention_dimensions(descriptor: &str, value_type: &ArrayType) -> Result<[usize; 4], TypeError> {
    let Some(shape) = value_type.static_shape() else {
        return Err(TypeError {
            message: format!("'{DOT_PRODUCT_ATTENTION_OPERATION_NAME}' {descriptor} must have a static shape"),
        });
    };
    match *shape.dimensions() {
        [batch, sequence, heads, head_dimension] => Ok([batch, sequence, heads, head_dimension]),
        ref dimensions => Err(TypeError {
            message: format!(
                "'{DOT_PRODUCT_ATTENTION_OPERATION_NAME}' {descriptor} must have rank 4 but got rank {}",
                dimensions.len(),
            ),
        }),
    }
}

impl<C: Domain<Type = ArrayType, Value: DotProductAttention>> InterpretableOperation<C>
    for DotProductAttentionOperation
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 3, ProgramError);
        Ok(vec![inputs[0].dot_product_attention(&inputs[1], &inputs[2], self.scale, self.mask)?])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for DotProductAttentionOperation where
    C::Operation: From<DotProductAttentionOperation>
{
}

/// The operation is the inference fast path, so there is no differentiation rule: differentiating reports an error
/// directing users to differentiate an explicit attention composition instead (the fused backward custom call is a
/// recorded follow-up).
impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for DotProductAttentionOperation
where
    C::Operation: From<DotProductAttentionOperation>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        _inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "'{DOT_PRODUCT_ATTENTION_OPERATION_NAME}' does not support differentiation; differentiate an \
                 explicit attention composition instead"
            ),
        }
        .into())
    }
}

impl_non_transposable_operation!(DotProductAttentionOperation);

/// Batching rule for [`DotProductAttentionOperation`]: attention is batch-parallel, so one mapped batch level folds
/// into the operation's own batch dimension. Every operand is aligned to a physical batch axis at position 0 (mapped
/// operands are realigned, replicated operands are broadcast into `axis_size` copies), the resulting
/// `[v, batch, seq, heads, head_dim]` operands are reshaped to `[v * batch, seq, heads, head_dim]`, the same
/// operation runs over the merged batch, and the output splits the mapped axis back out to
/// `[v, batch, q_seq, heads, head_dim]`. When every operand is replicated, the lifted operation is the unbatched
/// operation itself with a replicated output.
impl<C: Context<Type = ArrayType, Value: Broadcast + Reshape + Transpose>> BatchableOperation<C>
    for DotProductAttentionOperation
where
    DotProductAttentionOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        check_count!("input", inputs, 3, ProgramError);
        let Some(axis_size) = ArrayBatch::common_batch_size(inputs)? else {
            // Every operand is replicated: the lifted operation is the unbatched operation itself.
            return self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()]);
        };
        let axis_sharding = ArrayBatch::sharding_for_inputs(inputs)?;
        let static_dimensions = |value_type: &ArrayType| -> Result<Vec<usize>, BatchingError> {
            match value_type.static_shape() {
                Some(shape) => Ok(shape.dimensions().to_vec()),
                None => Err(ProgramError::from(TypeError {
                    message: format!(
                        "'{DOT_PRODUCT_ATTENTION_OPERATION_NAME}' batching requires statically shaped operands"
                    ),
                })
                .into()),
            }
        };
        // Fold the mapped axis into the attention batch dimension: `[v, batch, seq, heads, head_dim]` reshapes to
        // `[v * batch, seq, heads, head_dim]` and the merged operands run through the same (fused) operation.
        let merged_inputs = inputs
            .iter()
            .map(|input| {
                let aligned = input.match_axis(0, axis_size, axis_sharding.clone())?;
                let dimensions = static_dimensions(&aligned.r#type())?;
                let merged_dimensions = std::iter::once(Size::Static(dimensions[0] * dimensions[1]))
                    .chain(dimensions[2..].iter().map(|&size| Size::Static(size)))
                    .collect();
                let merged_value = aligned.value().reshape(Shape::new(merged_dimensions))?;
                Ok(ArrayBatch::replicated(merged_value))
            })
            .collect::<Result<Vec<_>, BatchingError>>()?;
        let mut outputs =
            self.interpret_with_batch_axes(context, merged_inputs.as_slice(), &[BatchAxis::replicated()])?;
        // Split the mapped axis back out of the merged output batch dimension and map the result at axis 0.
        let output = outputs.remove(0);
        let output_dimensions = static_dimensions(&output.r#type())?;
        let split_dimensions = [axis_size, output_dimensions[0] / axis_size]
            .into_iter()
            .chain(output_dimensions[1..].iter().copied())
            .map(Size::Static)
            .collect();
        let split_value = output.value().reshape(Shape::new(split_dimensions))?;
        let split_type = split_value.r#type().into_owned();
        Ok(vec![ArrayBatch::new(split_type, split_value, BatchAxis::new(0))?])
    }
}

/// Value-level scaled dot-product attention capability. Refer to the documentation of
/// [`DotProductAttentionOperation`] for the `BTNH` operand convention, the exact semantics, and the transform rules.
pub trait DotProductAttention: Sized {
    /// Computes scaled dot-product attention with `self` as the query (shape `[batch, q_seq, heads, head_dim]`)
    /// over `key`/`value` (shape `[batch, kv_seq, heads, head_dim]`), returning the attended
    /// `[batch, q_seq, heads, head_dim]` output at the operand data type, and a [`ProgramError`] if something goes
    /// wrong.
    ///
    /// # Parameters
    ///
    ///   - `key`: Key operand aligned with `value` along the key/value sequence dimension.
    ///   - `value`: Value operand whose rows are mixed by the attention weights.
    ///   - `scale`: Multiplier applied to the attention scores before masking and softmax.
    ///   - `mask`: Built-in [`AttentionMask`] applied to the attention scores before the softmax.
    fn dot_product_attention(
        &self,
        key: &Self,
        value: &Self,
        scale: f64,
        mask: AttentionMask,
    ) -> Result<Self, ProgramError>;
}

/// Any context-carrying value computes attention by binding a [`DotProductAttentionOperation`] through its own
/// context. The `From<DotProductAttentionOperation>` bound makes this disjoint from the eager reference value types
/// (whose context operation is [`ConstantOperation`](crate::operations::constants::ConstantOperation)), so it covers
/// the transform tracers and backend-owned values without conflicting with concrete implementations.
impl<V: Value<Type = ArrayType>> DotProductAttention for V
where
    V::DispatchDomain: Context<Operation: From<DotProductAttentionOperation>>,
{
    fn dot_product_attention(
        &self,
        key: &Self,
        value: &Self,
        scale: f64,
        mask: AttentionMask,
    ) -> Result<Self, ProgramError> {
        let mut outputs = self.dispatch_domain().bind(
            DotProductAttentionOperation::new(scale, mask),
            Vec::new(),
            &[self.clone(), key.clone(), value.clone()],
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Evaluates scaled dot-product attention as the portable composition: `scores = query · keyᵀ` per batch item and
/// head (`[batch, heads, q_seq, kv_seq]`), converted to the softmax data type (`f32` for operand types narrower
/// than `f32`, matching the XLA attention path and keeping low-precision softmaxes stable; `f64` stays `f64`),
/// multiplied by `scale`, optionally masked causally (score positions with column index greater than the row index
/// are replaced by `-1e30`), passed through a max-stabilized softmax over the last axis, converted back to the
/// operand data type, contracted with `value`, and transposed back to the `BTNH` output layout. This is the shared
/// semantics behind the concrete [`DotProductAttention`] implementations and the portable XLA lowering.
pub(crate) fn dot_product_attention_composition<C, V>(
    context: &C,
    query: &V,
    key: &V,
    value: &V,
    scale: f64,
    mask: AttentionMask,
) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayType>
        + Broadcast
        + Compare<Output = V>
        + ConvertElementType
        + Div
        + Dot
        + Exp
        + Mul
        + Reduce
        + Select<Condition = V>
        + Sub
        + Transpose,
    C: Fill<Scalar, V> + Iota<V>,
{
    // Validate the operand contract up front through the operation's own type inference, so the eager reference
    // route reports the same precise type errors as staged binding.
    let operation = DotProductAttentionOperation::new(scale, mask);
    operation.infer_output_types(
        &[query.r#type().into_owned(), key.r#type().into_owned(), value.r#type().into_owned()],
        &[],
    )?;
    let data_type = query.r#type().data_type();
    // Scores over `[batch, heads]`: `query [b, qs, n, d] · key [b, ks, n, d]` contracting `d` -> `[b, n, qs, ks]`.
    let scores = query.dot(key, &DotDimensionNumbers::new(vec![3], vec![3], vec![0, 2], vec![0, 2]));
    let softmax_type = if data_type == DataType::F64 { DataType::F64 } else { DataType::F32 };
    let scores = if data_type == softmax_type { scores } else { scores.convert_element_type(softmax_type)? };
    let scores_type = scores.r#type().into_owned();
    let fill = |value: f64| -> Result<V, ProgramError> {
        let scalar = if softmax_type == DataType::F32 { Scalar::from(value as f32) } else { Scalar::from(value) };
        context.fill(&scores_type, scalar)
    };
    let scores = scores.mul(&fill(scale)?)?;
    let scores = match mask {
        AttentionMask::None => scores,
        AttentionMask::Causal => {
            // A score position is visible when its column (key/value) index does not exceed its row (query) index.
            let index_type = ArrayType::new(DataType::I32, scores_type.shape().clone());
            let rows = context.iota(&index_type, 2)?;
            let columns = context.iota(&index_type, 3)?;
            let visible = columns.compare(&rows, ComparisonDirection::LessThanOrEqual)?;
            V::select(&visible, &scores, &fill(-1.0e30)?)?
        }
    };
    // Max-stabilized softmax over the key/value sequence (last) axis.
    let score_axes = &[0, 1, 2];
    let maxima = scores.reduce(&[3], ReductionKind::Max).broadcast(scores_type.clone(), score_axes)?;
    let exponentials = scores.sub(&maxima)?.exp()?;
    let sums = exponentials.reduce(&[3], ReductionKind::Sum).broadcast(scores_type, score_axes)?;
    let weights = exponentials.div(&sums)?;
    let weights = if data_type == softmax_type { weights } else { weights.convert_element_type(data_type)? };
    // Context values: `weights [b, n, qs, ks] · value [b, ks, n, d]` contracting `ks` -> `[b, n, qs, d]`, then
    // transposed back to the `BTNH` output layout `[b, qs, n, d]`.
    let attended = weights.dot(value, &DotDimensionNumbers::new(vec![3], vec![1], vec![0, 1], vec![0, 2]));
    attended.transpose([0, 2, 1, 3])
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::batching::ProgramBatchingOutputAxesPolicy;
    use crate::contexts::EagerContext;
    use crate::macros::{check_operation_transposition, check_operation_type_inference};
    use crate::parameters::Placeholder;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::sharding::ShardingDimension;

    use super::*;

    /// Returns the static `BTNH` [`ArrayType`] with the provided data type and dimensions used throughout these
    /// tests.
    fn attention_type(data_type: DataType, dimensions: [usize; 4]) -> ArrayType {
        ArrayType::new(data_type, Shape::new(dimensions.iter().map(|&size| Size::Static(size)).collect()))
    }

    /// Plain-Rust host reference for `BTNH` scaled dot-product attention with `f64` accumulation.
    fn host_attention(
        query: &[f64],
        key: &[f64],
        value: &[f64],
        [batch, q_seq, heads, head_dim]: [usize; 4],
        kv_seq: usize,
        scale: f64,
        causal: bool,
    ) -> Vec<f64> {
        let query_at = |b: usize, s: usize, n: usize, d: usize| query[((b * q_seq + s) * heads + n) * head_dim + d];
        let key_at = |b: usize, s: usize, n: usize, d: usize| key[((b * kv_seq + s) * heads + n) * head_dim + d];
        let value_at = |b: usize, s: usize, n: usize, d: usize| value[((b * kv_seq + s) * heads + n) * head_dim + d];
        let mut output = vec![0.0; batch * q_seq * heads * head_dim];
        for b in 0..batch {
            for n in 0..heads {
                for i in 0..q_seq {
                    let mut scores = vec![0.0; kv_seq];
                    for (j, score) in scores.iter_mut().enumerate() {
                        let mut product = 0.0;
                        for d in 0..head_dim {
                            product += query_at(b, i, n, d) * key_at(b, j, n, d);
                        }
                        *score = if causal && j > i { f64::NEG_INFINITY } else { product * scale };
                    }
                    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let exponentials: Vec<f64> = scores.iter().map(|score| (score - maximum).exp()).collect();
                    let sum: f64 = exponentials.iter().sum();
                    for d in 0..head_dim {
                        let mut attended = 0.0;
                        for (j, exponential) in exponentials.iter().enumerate() {
                            attended += exponential / sum * value_at(b, j, n, d);
                        }
                        output[((b * q_seq + i) * heads + n) * head_dim + d] = attended;
                    }
                }
            }
        }
        output
    }

    #[test]
    fn test_dot_product_attention() {
        // Small-shape correctness on the reference array backend against a plain-Rust host reference computed with
        // `f64` accumulation: `b = 1`, `s = 4`, `h = 2`, `d = 3` in `f32`.
        let dimensions = [1, 4, 2, 3];
        let operand_type = attention_type(DataType::F32, dimensions);
        let query_values: Vec<f64> = (0..24).map(|i| ((i * 7 % 11) as f64 - 5.0) * 0.25).collect();
        let key_values: Vec<f64> = (0..24).map(|i| ((i * 5 % 13) as f64 - 6.0) * 0.25).collect();
        let value_values: Vec<f64> = (0..24).map(|i| ((i * 3 % 7) as f64 - 3.0) * 0.5).collect();
        let query = Array::from_f64s(operand_type.clone(), query_values.clone());
        let key = Array::from_f64s(operand_type.clone(), key_values.clone());
        let value = Array::from_f64s(operand_type.clone(), value_values.clone());
        let scale = 0.5;

        let unmasked = query.dot_product_attention(&key, &value, scale, AttentionMask::None).unwrap();
        assert_eq!(unmasked.r#type().as_ref(), &operand_type);
        let expected_unmasked = host_attention(&query_values, &key_values, &value_values, dimensions, 4, scale, false);
        for (actual, expected) in unmasked.to_f64s().into_iter().zip(expected_unmasked.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-5);
        }

        // The causal mask excludes later key/value positions for early query rows, so the masked output both
        // matches the causal host reference and differs from the unmasked output.
        let causal = query.dot_product_attention(&key, &value, scale, AttentionMask::Causal).unwrap();
        let expected_causal = host_attention(&query_values, &key_values, &value_values, dimensions, 4, scale, true);
        for (actual, expected) in causal.to_f64s().into_iter().zip(expected_causal.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-5);
        }
        assert_ne!(causal.to_f64s(), unmasked.to_f64s());

        // A narrow-precision operand type runs its softmax at `f32` and converts back, so a small `bf16` case stays
        // within the `bf16` grid's tolerance of the host reference.
        let bf16_dimensions = [1, 2, 1, 2];
        let bf16_type = attention_type(DataType::BF16, bf16_dimensions);
        let bf16_query_values = vec![0.5, -0.25, 1.0, 0.75];
        let bf16_key_values = vec![0.25, 0.5, -0.5, 1.0];
        let bf16_value_values = vec![1.0, 2.0, -1.0, 0.5];
        let bf16_query = Array::from_f64s(bf16_type.clone(), bf16_query_values.clone());
        let bf16_key = Array::from_f64s(bf16_type.clone(), bf16_key_values.clone());
        let bf16_value = Array::from_f64s(bf16_type.clone(), bf16_value_values.clone());
        let bf16_output = bf16_query.dot_product_attention(&bf16_key, &bf16_value, 1.0, AttentionMask::None).unwrap();
        assert_eq!(bf16_output.r#type().as_ref(), &bf16_type);
        let expected_bf16 =
            host_attention(&bf16_query_values, &bf16_key_values, &bf16_value_values, bf16_dimensions, 2, 1.0, false);
        for (actual, expected) in bf16_output.to_f64s().into_iter().zip(expected_bf16.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 2e-2);
        }

        // The staged operation renders its payload.
        let operation = DotProductAttentionOperation::new(scale, AttentionMask::Causal);
        assert_eq!(operation.scale(), scale);
        assert_eq!(operation.mask(), AttentionMask::Causal);
        assert_eq!(operation.name(), DOT_PRODUCT_ATTENTION_OPERATION_NAME);
        assert_eq!(operation.to_string(), "dot_product_attention [scale=0.5, mask=causal]");
        assert_eq!(
            DotProductAttentionOperation::new(0.125, AttentionMask::None).to_string(),
            "dot_product_attention [scale=0.125, mask=none]",
        );
    }

    #[test]
    fn test_dot_product_attention_type_inference() {
        let operation = DotProductAttentionOperation::new(0.5, AttentionMask::None);
        let query = attention_type(DataType::F32, [2, 4, 2, 3]);
        let key_value = attention_type(DataType::F32, [2, 5, 2, 3]);
        check_operation_type_inference!(
            operation = operation,
            cases = [
                {
                    type = ArrayType,
                    input_types = [query.clone(), key_value.clone(), key_value.clone()],
                    output_types = [query.clone()],
                },
                {
                    input_types = [
                        attention_type(DataType::BF16, [2, 4, 2, 3]),
                        attention_type(DataType::BF16, [2, 5, 2, 3]),
                        attention_type(DataType::BF16, [2, 5, 2, 3]),
                    ],
                    output_types = [attention_type(DataType::BF16, [2, 4, 2, 3])],
                },
                {
                    input_types = [query.clone(), key_value.clone()],
                    error = "expected 3 inputs but got 2",
                },
                {
                    input_types = [
                        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2), Size::Static(4)])),
                        key_value.clone(),
                        key_value.clone(),
                    ],
                    error = "'dot_product_attention' query must have rank 4 but got rank 2",
                },
                {
                    input_types = [
                        query.clone(),
                        attention_type(DataType::F32, [2, 5, 2, 3]),
                        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)])),
                    ],
                    error = "'dot_product_attention' value must have rank 4 but got rank 1",
                },
                {
                    input_types = [
                        attention_type(DataType::I32, [2, 4, 2, 3]),
                        attention_type(DataType::I32, [2, 5, 2, 3]),
                        attention_type(DataType::I32, [2, 5, 2, 3]),
                    ],
                    error = "'dot_product_attention' requires floating-point operands but got data type i32",
                },
                {
                    input_types = [
                        attention_type(DataType::C64, [2, 4, 2, 3]),
                        attention_type(DataType::C64, [2, 5, 2, 3]),
                        attention_type(DataType::C64, [2, 5, 2, 3]),
                    ],
                    error = "'dot_product_attention' requires floating-point operands but got data type c64",
                },
                {
                    input_types = [
                        query.clone(),
                        attention_type(DataType::F16, [2, 5, 2, 3]),
                        attention_type(DataType::F16, [2, 5, 2, 3]),
                    ],
                    error = "'dot_product_attention' key data type f16 does not match the query data type f32",
                },
                {
                    input_types = [query.clone(), attention_type(DataType::F32, [3, 5, 2, 3]), key_value.clone()],
                    error = "'dot_product_attention' key batch dimension (3) does not match the query batch \
                             dimension (2)",
                },
                {
                    input_types = [query.clone(), attention_type(DataType::F32, [2, 5, 1, 3]), key_value.clone()],
                    error = "'dot_product_attention' key heads dimension (1) does not match the query heads \
                             dimension (2); grouped-query attention is not supported yet",
                },
                {
                    input_types = [query.clone(), key_value.clone(), attention_type(DataType::F32, [2, 5, 2, 4])],
                    error = "'dot_product_attention' value head dimension (4) does not match the query head \
                             dimension (3)",
                },
                {
                    input_types = [query.clone(), key_value.clone(), attention_type(DataType::F32, [2, 6, 2, 3])],
                    error = "'dot_product_attention' value sequence dimension (6) does not match the key sequence \
                             dimension (5)",
                },
                {
                    input_types = [
                        ArrayType::new(
                            DataType::F32,
                            Shape::new(vec![Size::Static(2), Size::Dynamic(None), Size::Static(2), Size::Static(3)]),
                        ),
                        key_value.clone(),
                        key_value,
                    ],
                    error = "'dot_product_attention' query must have a static shape",
                },
            ],
        );
    }

    #[test]
    fn test_dot_product_attention_batching() {
        // Two batch items with per-item shape `[1, 2, 1, 2]`, so the batching rule folds the mapped axis into the
        // attention batch dimension and the per-item expectations come from unbatched calls.
        let dimensions = [1, 2, 1, 2];
        let operand_type = attention_type(DataType::F32, dimensions);
        let item_0_query = vec![0.5, -0.25, 1.0, 0.75];
        let item_0_key = vec![0.25, 0.5, -0.5, 1.0];
        let item_0_value = vec![1.0, 2.0, -1.0, 0.5];
        let item_1_query = vec![-0.5, 0.75, 0.25, -1.0];
        let item_1_key = vec![1.0, -0.25, 0.5, 0.5];
        let item_1_value = vec![0.5, -1.5, 2.0, 1.0];
        let item = |values: &[f64]| Array::from_f64s(operand_type.clone(), values.to_vec());
        let scale = 0.5;
        let operation = DotProductAttentionOperation::new(scale, AttentionMask::Causal);
        let attend = |query: &[f64], key: &[f64], value: &[f64]| {
            item(query).dot_product_attention(&item(key), &item(value), scale, AttentionMask::Causal).unwrap()
        };

        let stacked_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(2), Size::Static(1), Size::Static(2), Size::Static(1), Size::Static(2)]),
        );
        let stack = |first: &[f64], second: &[f64]| {
            let values = first.iter().chain(second.iter()).copied().collect::<Vec<_>>();
            let value = Array::from_f64s(stacked_type.clone(), values);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0)).unwrap()
        };
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);

        // All three operands mapped at axis 0: the output is mapped at axis 0 and matches the per-item results.
        let outputs = operation
            .batch(
                &context,
                &EmptyRegionDriver,
                &[
                    stack(&item_0_query, &item_1_query),
                    stack(&item_0_key, &item_1_key),
                    stack(&item_0_value, &item_1_value),
                ],
            )
            .unwrap();
        let expected: Vec<f64> = attend(&item_0_query, &item_0_key, &item_0_value)
            .to_f64s()
            .into_iter()
            .chain(attend(&item_1_query, &item_1_key, &item_1_value).to_f64s())
            .collect();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().to_f64s(), expected);

        // Mixed mapped/replicated operands: the replicated key/value pair is broadcast into per-item copies, so
        // every batch item attends over the same key/value cache.
        let replicated = |values: &[f64]| ArrayBatch::replicated(item(values));
        let outputs = operation
            .batch(
                &context,
                &EmptyRegionDriver,
                &[stack(&item_0_query, &item_1_query), replicated(&item_0_key), replicated(&item_0_value)],
            )
            .unwrap();
        let expected_shared: Vec<f64> = attend(&item_0_query, &item_0_key, &item_0_value)
            .to_f64s()
            .into_iter()
            .chain(attend(&item_1_query, &item_0_key, &item_0_value).to_f64s())
            .collect();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().to_f64s(), expected_shared);

        // All-replicated operands stay replicated.
        let outputs = operation
            .batch(
                &context,
                &EmptyRegionDriver,
                &[replicated(&item_0_query), replicated(&item_0_key), replicated(&item_0_value)],
            )
            .unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value().to_f64s(), attend(&item_0_query, &item_0_key, &item_0_value).to_f64s());

        // The staged batched program folds the mapped axis in with reshapes around the same fused operation.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let inputs = vec![
            builder.add_input(operand_type.clone()),
            builder.add_input(operand_type.clone()),
            builder.add_input(operand_type),
        ];
        let output = builder.add_instruction(operation, Vec::new(), inputs).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let (batched, output_axes) = program
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0); 3],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f32[2, 1, 2, 1, 2], %1:f32[2, 1, 2, 1, 2], %2:f32[2, 1, 2, 1, 2] .
                let %3:f32[2, 2, 1, 2] = reshape [shape=[2, 2, 1, 2]] %0
                    %4:f32[2, 2, 1, 2] = reshape [shape=[2, 2, 1, 2]] %1
                    %5:f32[2, 2, 1, 2] = reshape [shape=[2, 2, 1, 2]] %2
                    %6:f32[2, 2, 1, 2] = dot_product_attention [scale=0.5, mask=causal] %3 %4 %5
                    %7:f32[2, 1, 2, 1, 2] = reshape [shape=[2, 1, 2, 1, 2]] %6
                in (%7)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_dot_product_attention_differentiation_rejection() {
        // Differentiation is rejected with an error directing to an explicit attention composition, and
        // transposition reports the standard non-transposable-operation error.
        let operand_type = attention_type(DataType::F32, [1, 2, 1, 2]);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let inputs = vec![
            builder.add_input(operand_type.clone()),
            builder.add_input(operand_type.clone()),
            builder.add_input(operand_type.clone()),
        ];
        let operation = DotProductAttentionOperation::new(0.5, AttentionMask::None);
        let output = builder.add_instruction(operation, Vec::new(), inputs).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            program.jvp(),
            Err(error) if error.to_string().contains(
                "'dot_product_attention' does not support differentiation; differentiate an explicit attention \
                 composition instead",
            ),
        ));
        check_operation_transposition!(
            @rejected,
            operation = DotProductAttentionOperation::new(0.5, AttentionMask::None),
            input_types = [operand_type.clone(), operand_type.clone(), operand_type],
        );
    }

    #[test]
    fn test_dot_product_attention_rendering() {
        // The staged program pins the operation's rendering, including the payload fields.
        let query_type = attention_type(DataType::BF16, [2, 4, 2, 3]);
        let key_value_type = attention_type(DataType::BF16, [2, 5, 2, 3]);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let inputs = vec![
            builder.add_input(query_type),
            builder.add_input(key_value_type.clone()),
            builder.add_input(key_value_type),
        ];
        let operation = DotProductAttentionOperation::new(0.125, AttentionMask::Causal);
        let output = builder.add_instruction(operation, Vec::new(), inputs).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bf16[2, 4, 2, 3], %1:bf16[2, 5, 2, 3], %2:bf16[2, 5, 2, 3] .
                let %3:bf16[2, 4, 2, 3] = dot_product_attention [scale=0.125, mask=causal] %0 %1 %2
                in (%3)
            "}
            .trim_end(),
        );
    }
}
