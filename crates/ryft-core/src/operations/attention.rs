use std::fmt::Display;

use crate::backends::scalars::Scalar;
use crate::batching::{
    ArrayBatch, ArrayBatching, ArrayBatchingPolicy, BatchAxis, BatchableOperation, BatchingContext, BatchingDriver,
    BatchingError, InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::DifferentiableType;
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::compare::{Compare, ComparisonDirection};
use crate::operations::constants::{Fill, Iota, Zero, ZeroOperation};
use crate::operations::control_flow::Select;
use crate::operations::logical::And;
use crate::operations::manipulation::{ConvertElementType, LegacyBroadcast, Reshape, Transpose};
use crate::operations::math::{Add, Div, Dot, DotDimensionNumbers, Exp, Log, Mul, Reduce, ReductionKind, Sub};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{TypeError, Typed};
use crate::programs::{ProgramError, Value};
use crate::sharding::Sharding;
use crate::tracing::DomainTracer;
use crate::tracing_v2::{CustomVjp, custom_vjp};
use crate::types::{ArrayType, DataType, Dimension, Shape};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`DotProductAttentionOperation`].
pub const DOT_PRODUCT_ATTENTION_OPERATION_NAME: &str = "dot_product_attention";

/// Canonical operation name for [`DotProductAttentionBackwardOperation`].
pub const DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME: &str = "dot_product_attention_backward";

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
/// The operands use the `BTNH` logical layout: `query [batch, q_seq, heads, head_dim]` and `key`/`value`
/// `[batch, kv_seq, kv_heads, head_dim]`, all carrying one shared floating-point data type, and the first output is
/// `[batch, q_seq, heads, head_dim]` at that same type. The key/value heads count must divide the query heads count:
/// when it is smaller the operation computes grouped-query attention, with query head `i` attending key/value head
/// `i / (heads / kv_heads)`. Semantically the operation computes
/// `softmax(scale · query · keyᵀ + bias + mask) · value` per batch item and head, with the softmax running at
/// `f32` for every operand type narrower than `f32` (`f64` operands keep an `f64` softmax) — which is exactly how the
/// reference array backend and the portable XLA lowering evaluate it (see [`dot_product_attention_composition`]).
/// On CUDA targets, the XLA lowering instead emits the `__cudnn$fmhaSoftmax` custom call, reaching cuDNN's fused
/// flash-attention kernels.
///
/// An optional fourth `bias` operand of shape `[batch | 1, heads | 1, q_seq, kv_seq]` (its leading two dimensions
/// broadcast) is added to the already scaled scores after the softmax-type upcast and before masking; its data type
/// must equal the query data type, and it is converted to the softmax data type together with the scores. The causal
/// [`AttentionMask`] can be tightened with [`with_sliding_window`](Self::with_sliding_window) so query row `r`
/// attends only key/value positions `[max(0, r + 1 - window), r]`. Training forwards request a second output with
/// [`with_activation_output`](Self::with_activation_output): the `f32[batch, heads, q_seq]` natural-log log-sum-exp
/// statistic of the masked logits over the key/value axis, which is exactly the residual
/// [`DotProductAttentionBackwardOperation`] consumes.
///
/// Variable sequence lengths (padding) are supported through an optional trailing operand pair
/// `(query_sequence_lengths, key_value_sequence_lengths)`, both `i32[batch]`, appended after the bias when one is
/// present, so the operand counts decode uniquely: 3 plain, 4 with a bias, 5 with sequence lengths, and 6 with both.
/// The semantics match the fused cuDNN kernels exactly: key/value columns at or beyond
/// `key_value_sequence_lengths[b]` are fully excluded from the softmax (their scores are masked to `-1e30` before
/// the softmax, composing with the causal mask as `column <= row AND column < key_value_sequence_lengths[b]`), and
/// out-of-range query rows (`row >= query_sequence_lengths[b]`) are exact zeros in both the attended output and the
/// activation statistic, mirroring XLA's memzeroed fMHA outputs.
///
/// Dropout on the attention weights is carried as an optional `(rate, seed)` attribute (refer to
/// [`with_dropout`](Self::with_dropout)); type inference accepts it because it changes no output shapes, but only
/// the fused cuDNN kernels implement it — the portable composition, and therefore the reference array backend and
/// every non-CUDA lowering, rejects it, exactly like JAX where only the cuDNN implementation of
/// `jax.nn.dot_product_attention` supports dropout. The fused kernels' determinism contract (validated on hardware):
/// the kernel installs `seed` into a per-target-name dropout state and advances an internal offset by 16 on every
/// execution, so same-seed reproducibility holds for the *first* call of a freshly compiled executable (repeated
/// calls of one executable draw fresh masks), and the forward and backward thunks each keep their own state, so the
/// forward and backward dropout masks agree per matching call index.
///
/// The operation itself is the inference fast path and rejects differentiation; the training path is
/// [`differentiable_dot_product_attention`], which pairs the activation-producing forward with
/// [`DotProductAttentionBackwardOperation`] through [`custom_vjp`]. Batching folds one mapped axis into the batch
/// dimension and reuses the same operation (attention is batch-parallel) — refer to the batching rule documentation
/// on this operation.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DotProductAttentionOperation {
    /// Multiplier applied to the attention scores before the bias, masking, and softmax (typically
    /// `1 / sqrt(head_dim)`).
    scale: f64,

    /// Built-in mask applied to the attention scores before the softmax.
    mask: AttentionMask,

    /// Optional sliding-window width tightening the causal mask; refer to
    /// [`with_sliding_window`](Self::with_sliding_window).
    sliding_window: Option<usize>,

    /// Optional `(rate, seed)` dropout applied to the attention weights by the fused cuDNN kernels; refer to
    /// [`with_dropout`](Self::with_dropout).
    dropout: Option<(f64, u64)>,

    /// Whether the operation produces the `f32[batch, heads, q_seq]` log-sum-exp activation statistic as a second
    /// output; refer to [`with_activation_output`](Self::with_activation_output).
    activation_output: bool,
}

impl DotProductAttentionOperation {
    /// Creates a new [`DotProductAttentionOperation`] with the provided score scale and mask, no sliding window,
    /// no dropout, and no activation output.
    #[inline]
    pub fn new(scale: f64, mask: AttentionMask) -> Self {
        Self { scale, mask, sliding_window: None, dropout: None, activation_output: false }
    }

    /// Sets the sliding-window width: query row `r` attends only key/value positions `[max(0, r + 1 - window), r]`
    /// (the causal upper bound plus a window lower bound, matching cuDNN's `sliding_window_length`). The window is
    /// only meaningful combined with [`AttentionMask::Causal`] and must be positive; type inference rejects a zero
    /// window and a sliding window combined with [`AttentionMask::None`].
    #[inline]
    pub fn with_sliding_window<W: Into<Option<usize>>>(mut self, sliding_window: W) -> Self {
        self.sliding_window = sliding_window.into();
        self
    }

    /// Sets the `(rate, seed)` dropout applied to the attention weights. The rate must lie in the open interval
    /// `(0, 1)` (type inference rejects other rates; a zero rate is expressed by omitting dropout altogether). Only
    /// the fused cuDNN kernels implement dropout — refer to the operation documentation for the exact support and
    /// determinism contract.
    #[inline]
    pub fn with_dropout<P: Into<Option<(f64, u64)>>>(mut self, dropout: P) -> Self {
        self.dropout = dropout.into();
        self
    }

    /// Requests the training-forward activation output: the operation produces a second `f32[batch, heads, q_seq]`
    /// output carrying the natural-log log-sum-exp statistic of the post-scale, post-bias, post-mask logits over the
    /// key/value axis, which [`DotProductAttentionBackwardOperation`] consumes to recover the attention weights
    /// without re-running the softmax reductions.
    #[inline]
    pub fn with_activation_output(mut self) -> Self {
        self.activation_output = true;
        self
    }

    /// Returns the multiplier applied to the attention scores before the bias, masking, and softmax.
    #[inline]
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Returns the built-in mask applied to the attention scores before the softmax.
    #[inline]
    pub fn mask(&self) -> AttentionMask {
        self.mask
    }

    /// Returns the optional sliding-window width tightening the causal mask.
    #[inline]
    pub fn sliding_window(&self) -> Option<usize> {
        self.sliding_window
    }

    /// Returns the optional `(rate, seed)` dropout applied to the attention weights.
    #[inline]
    pub fn dropout(&self) -> Option<(f64, u64)> {
        self.dropout
    }

    /// Returns whether the operation produces the log-sum-exp activation statistic as a second output.
    #[inline]
    pub fn activation_output(&self) -> bool {
        self.activation_output
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

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        if !matches!(input_types.len(), 3..=6) {
            return Err(TypeError::invalid(format!(
                "'{DOT_PRODUCT_ATTENTION_OPERATION_NAME}' expects 3 (query, key, value), 4 (query, key, value, \
                     bias), 5 (query, key, value, query sequence lengths, key/value sequence lengths), or 6 (query, \
                     key, value, bias, query sequence lengths, key/value sequence lengths) inputs but got {}",
                input_types.len(),
            )));
        }
        let bias_type = matches!(input_types.len(), 4 | 6).then(|| &input_types[3]);
        let dimensions = validated_attention_operands(
            DOT_PRODUCT_ATTENTION_OPERATION_NAME,
            &input_types[0],
            &input_types[1],
            &input_types[2],
            bias_type,
            self.mask,
            self.sliding_window,
        )?;
        if matches!(input_types.len(), 5 | 6) {
            validated_sequence_length_operands(
                DOT_PRODUCT_ATTENTION_OPERATION_NAME,
                &input_types[input_types.len() - 2],
                &input_types[input_types.len() - 1],
                dimensions.batch,
            )?;
        }
        validated_dropout(DOT_PRODUCT_ATTENTION_OPERATION_NAME, self.dropout)?;
        for input_type in input_types {
            if !input_type.unreduced_axes().is_empty() {
                return Err(TypeError::invalid(format!(
                    "'{DOT_PRODUCT_ATTENTION_OPERATION_NAME}' does not support unreduced operands"
                )));
            }
        }
        // The attended output is query-shaped at the query data type, so the inferred output type is the query type
        // itself, propagating operand-level metadata such as sharding.
        let mut output_types = vec![input_types[0].clone()];
        if self.activation_output {
            output_types.push(attention_activation_type(&dimensions, &input_types[0])?);
        }
        Ok(output_types)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, DOT_PRODUCT_ATTENTION_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("scale", &self.scale)?;
            operation.field("mask", &self.mask)?;
            if let Some(sliding_window) = self.sliding_window {
                operation.field("sliding_window", &sliding_window)?;
            }
            if let Some((rate, seed)) = self.dropout {
                operation.field("dropout_rate", &rate)?;
                operation.field("dropout_seed", &seed)?;
            }
            if self.activation_output {
                operation.field("activation", &self.activation_output)?;
            }
            Ok(())
        })
    }
}

/// Primitive representing the backward (gradient) pass of scaled dot-product attention — the portable analogue of
/// the `__cudnn$fmhaSoftmaxBackward` fused kernel. Its operands are the forward operands followed by the forward
/// outputs, the incoming output cotangent, and the forward's optional sequence lengths:
/// `(query, key, value[, bias], output, activation, output_cotangent[, query_sequence_lengths,
/// key_value_sequence_lengths])` — 6 operands plain, 7 with a bias, 8 with sequence lengths, and 9 with both —
/// where `output` and `output_cotangent` share the query type and `activation` is the `f32[batch, heads, q_seq]`
/// log-sum-exp statistic produced by a [`DotProductAttentionOperation`] with [an activation
/// output](DotProductAttentionOperation::with_activation_output). The outputs are the operand cotangents
/// `(query_cotangent, key_cotangent, value_cotangent[, bias_cotangent])`, shaped like the corresponding operands
/// (the bias cotangent sums over the bias's broadcast leading dimensions).
///
/// The `scale`, `mask`, `sliding_window`, and `dropout` attributes must match the forward operation; the backward
/// recomputes the masked logits from them, recovers the attention weights as `exp(logits - activation)`, and applies
/// the standard attention backward at the softmax data type (`f32` for operand types narrower than `f32`; `f64`
/// operands keep an `f64` computation with the `f32` statistic widened) — refer to
/// [`dot_product_attention_backward_composition`] for the exact formulas. Masked score positions carry `-1e30`
/// logits and therefore recover exactly zero weight, contributing no gradient. Under variable sequence lengths the
/// backward zeroes the out-of-range query rows of the incoming output cotangent before use (so the key/value
/// cotangents receive no contribution from them), forces the corresponding query-cotangent rows to exact zeros, and
/// leaves the key/value-cotangent columns at or beyond `key_value_sequence_lengths[b]` exactly zero through the
/// recovered zero weights — matching the fused kernel's memzeroed gradients. Dropout follows the forward contract:
/// the attribute is accepted by type inference but implemented only by the fused cuDNN kernels, whose forward and
/// backward thunks keep separate dropout states that agree per matching call index (refer to
/// [`DotProductAttentionOperation`]).
///
/// The operation rejects differentiation (higher-order derivatives go through an explicit attention composition) and
/// batches with the same merge-reshape rule as the forward operation.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DotProductAttentionBackwardOperation {
    /// Multiplier applied to the attention scores of the forward operation this backward pass differentiates.
    scale: f64,

    /// Built-in mask of the forward operation this backward pass differentiates.
    mask: AttentionMask,

    /// Optional sliding-window width of the forward operation this backward pass differentiates.
    sliding_window: Option<usize>,

    /// Optional `(rate, seed)` dropout of the forward operation this backward pass differentiates.
    dropout: Option<(f64, u64)>,
}

impl DotProductAttentionBackwardOperation {
    /// Creates a new [`DotProductAttentionBackwardOperation`] with the provided score scale and mask, no sliding
    /// window, and no dropout. The attributes must match the forward [`DotProductAttentionOperation`] being
    /// differentiated.
    #[inline]
    pub fn new(scale: f64, mask: AttentionMask) -> Self {
        Self { scale, mask, sliding_window: None, dropout: None }
    }

    /// Sets the sliding-window width of the forward operation this backward pass differentiates. Refer to the
    /// documentation of [`DotProductAttentionOperation::with_sliding_window`] for the window semantics.
    #[inline]
    pub fn with_sliding_window<W: Into<Option<usize>>>(mut self, sliding_window: W) -> Self {
        self.sliding_window = sliding_window.into();
        self
    }

    /// Sets the `(rate, seed)` dropout of the forward operation this backward pass differentiates. Refer to the
    /// documentation of [`DotProductAttentionOperation::with_dropout`] for the dropout contract.
    #[inline]
    pub fn with_dropout<P: Into<Option<(f64, u64)>>>(mut self, dropout: P) -> Self {
        self.dropout = dropout.into();
        self
    }

    /// Returns the multiplier applied to the attention scores of the forward operation.
    #[inline]
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Returns the built-in mask of the forward operation.
    #[inline]
    pub fn mask(&self) -> AttentionMask {
        self.mask
    }

    /// Returns the optional sliding-window width of the forward operation.
    #[inline]
    pub fn sliding_window(&self) -> Option<usize> {
        self.sliding_window
    }

    /// Returns the optional `(rate, seed)` dropout of the forward operation.
    #[inline]
    pub fn dropout(&self) -> Option<(f64, u64)> {
        self.dropout
    }
}

impl Display for DotProductAttentionBackwardOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Operation::<ArrayType>::render(self, formatter, 0)
    }
}

impl Operation<ArrayType> for DotProductAttentionBackwardOperation {
    #[inline]
    fn name(&self) -> &'static str {
        DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        if !matches!(input_types.len(), 6..=9) {
            return Err(TypeError::invalid(format!(
                "'{DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME}' expects 6 (query, key, value, output, \
                     activation, output cotangent), 7 (adding a bias after the value), 8 (adding trailing query and \
                     key/value sequence lengths), or 9 (adding both) inputs but got {}",
                input_types.len(),
            )));
        }
        let bias_type = matches!(input_types.len(), 7 | 9).then(|| &input_types[3]);
        let dimensions = validated_attention_operands(
            DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME,
            &input_types[0],
            &input_types[1],
            &input_types[2],
            bias_type,
            self.mask,
            self.sliding_window,
        )?;
        if matches!(input_types.len(), 8 | 9) {
            validated_sequence_length_operands(
                DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME,
                &input_types[input_types.len() - 2],
                &input_types[input_types.len() - 1],
                dimensions.batch,
            )?;
        }
        validated_dropout(DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME, self.dropout)?;
        let offset = if bias_type.is_some() { 4 } else { 3 };
        // The forward-output-shaped operands compare by data type and shape only, so operand-level metadata such as
        // sharding never fails the structural contract.
        let matches_expected = |actual: &ArrayType, expected: &ArrayType| -> bool {
            actual.data_type() == expected.data_type() && actual.shape() == expected.shape()
        };
        let expected_output_type = attention_output_type(&dimensions);
        for (descriptor, index) in [("output", offset), ("output cotangent", offset + 2)] {
            if !matches_expected(&input_types[index], &expected_output_type) {
                return Err(TypeError::invalid(format!(
                    "'{DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME}' {descriptor} type {} does not match the \
                         expected forward output type {expected_output_type}",
                    input_types[index],
                )));
            }
        }
        let expected_activation_type = attention_activation_type(&dimensions, &input_types[0])?;
        if !matches_expected(&input_types[offset + 1], &expected_activation_type) {
            return Err(TypeError::invalid(format!(
                "'{DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME}' activation type {} does not match the \
                     expected activation type {expected_activation_type}",
                input_types[offset + 1],
            )));
        }
        for input_type in input_types {
            if !input_type.unreduced_axes().is_empty() {
                return Err(TypeError::invalid(format!(
                    "'{DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME}' does not support unreduced operands"
                )));
            }
        }
        let mut output_types = vec![input_types[0].clone(), input_types[1].clone(), input_types[2].clone()];
        if bias_type.is_some() {
            output_types.push(input_types[3].clone());
        }
        Ok(output_types)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME)?.bracketed(
            |operation| {
                operation.field("scale", &self.scale)?;
                operation.field("mask", &self.mask)?;
                if let Some(sliding_window) = self.sliding_window {
                    operation.field("sliding_window", &sliding_window)?;
                }
                if let Some((rate, seed)) = self.dropout {
                    operation.field("dropout_rate", &rate)?;
                    operation.field("dropout_seed", &seed)?;
                }
                Ok(())
            },
        )
    }
}

/// Returns the static `[batch, sequence, heads, head_dim]` dimensions of an attention operand type, rejecting
/// dynamic shapes and any rank other than 4.
fn static_attention_dimensions(
    operation_name: &str,
    descriptor: &str,
    value_type: &ArrayType,
) -> Result<[usize; 4], TypeError> {
    let Some(shape) = value_type.static_shape() else {
        return Err(TypeError::invalid(format!("'{operation_name}' {descriptor} must have a static shape")));
    };
    match *shape.dimensions() {
        [batch, sequence, heads, head_dimension] => Ok([batch, sequence, heads, head_dimension]),
        ref dimensions => Err(TypeError::invalid(format!(
            "'{operation_name}' {descriptor} must have rank 4 but got rank {}",
            dimensions.len(),
        ))),
    }
}

/// Validated static dimensions shared by the attention operations' operand contracts.
struct AttentionDimensions {
    /// Shared batch dimension of every operand.
    batch: usize,

    /// Query sequence length.
    query_sequence: usize,

    /// Number of query heads.
    query_heads: usize,

    /// Key/value sequence length.
    key_value_sequence: usize,

    /// Number of key/value heads; divides `query_heads`, with grouped-query attention when strictly smaller.
    key_value_heads: usize,

    /// Head (feature) dimension of every operand.
    head_dimension: usize,

    /// Shared floating-point operand data type.
    data_type: DataType,
}

/// Validates the shared operand contract of the attention operations — the `BTNH` query/key/value shapes and data
/// types (including the grouped-query heads divisibility), the optional broadcastable bias, and the sliding-window
/// attribute — and returns the validated [`AttentionDimensions`]. Refer to the documentation of
/// [`DotProductAttentionOperation`] for the contract itself.
fn validated_attention_operands(
    operation_name: &str,
    query_type: &ArrayType,
    key_type: &ArrayType,
    value_type: &ArrayType,
    bias_type: Option<&ArrayType>,
    mask: AttentionMask,
    sliding_window: Option<usize>,
) -> Result<AttentionDimensions, TypeError> {
    let query = static_attention_dimensions(operation_name, "query", query_type)?;
    let key = static_attention_dimensions(operation_name, "key", key_type)?;
    let value = static_attention_dimensions(operation_name, "value", value_type)?;
    let data_type = query_type.data_type();
    if !data_type.is_floating_point() {
        return Err(TypeError::invalid(format!(
            "'{operation_name}' requires floating-point operands but got data type {data_type}"
        )));
    }
    for (descriptor, dimensions, input_type) in [("key", &key, key_type), ("value", &value, value_type)] {
        if input_type.data_type() != data_type {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' {descriptor} data type {} does not match the query data type {data_type}",
                input_type.data_type(),
            )));
        }
        if dimensions[0] != query[0] {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' {descriptor} batch dimension ({}) does not match the query batch dimension \
                     ({})",
                dimensions[0], query[0],
            )));
        }
        if dimensions[3] != query[3] {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' {descriptor} head dimension ({}) does not match the query head dimension \
                     ({})",
                dimensions[3], query[3],
            )));
        }
    }
    if value[1] != key[1] {
        return Err(TypeError::invalid(format!(
            "'{operation_name}' value sequence dimension ({}) does not match the key sequence dimension ({})",
            value[1], key[1],
        )));
    }
    if value[2] != key[2] {
        return Err(TypeError::invalid(format!(
            "'{operation_name}' value heads dimension ({}) does not match the key heads dimension ({})",
            value[2], key[2],
        )));
    }
    if key[2] == 0 || query[2] % key[2] != 0 {
        return Err(TypeError::invalid(format!(
            "'{operation_name}' key/value heads dimension ({}) must divide the query heads dimension ({})",
            key[2], query[2],
        )));
    }
    if let Some(bias_type) = bias_type {
        let Some(bias_shape) = bias_type.static_shape() else {
            return Err(TypeError::invalid(format!("'{operation_name}' bias must have a static shape")));
        };
        let [bias_batch, bias_heads, bias_rows, bias_columns] = *bias_shape.dimensions() else {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' bias must have rank 4 but got rank {}",
                bias_shape.dimensions().len(),
            )));
        };
        if bias_type.data_type() != data_type {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' bias data type {} does not match the query data type {data_type}",
                bias_type.data_type(),
            )));
        }
        if bias_batch != 1 && bias_batch != query[0] {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' bias batch dimension ({bias_batch}) must be 1 or match the query batch \
                     dimension ({})",
                query[0],
            )));
        }
        if bias_heads != 1 && bias_heads != query[2] {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' bias heads dimension ({bias_heads}) must be 1 or match the query heads \
                     dimension ({})",
                query[2],
            )));
        }
        if bias_rows != query[1] {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' bias query-sequence dimension ({bias_rows}) does not match the query \
                     sequence dimension ({})",
                query[1],
            )));
        }
        if bias_columns != key[1] {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' bias key/value-sequence dimension ({bias_columns}) does not match the key \
                     sequence dimension ({})",
                key[1],
            )));
        }
    }
    match (sliding_window, mask) {
        (Some(0), _) => {
            return Err(TypeError::invalid(format!("'{operation_name}' sliding window must be positive")));
        }
        (Some(_), AttentionMask::None) => {
            return Err(TypeError::invalid(format!("'{operation_name}' sliding window requires the causal mask")));
        }
        _ => {}
    }
    Ok(AttentionDimensions {
        batch: query[0],
        query_sequence: query[1],
        query_heads: query[2],
        key_value_sequence: key[1],
        key_value_heads: key[2],
        head_dimension: query[3],
        data_type,
    })
}

/// Validates the optional trailing pair of `i32[batch]` sequence-length operands shared by the attention
/// operations: each operand must be a statically shaped rank-1 `i32` vector whose size matches the shared batch
/// dimension. Refer to the documentation of [`DotProductAttentionOperation`] for the padding semantics.
fn validated_sequence_length_operands(
    operation_name: &str,
    query_lengths_type: &ArrayType,
    key_value_lengths_type: &ArrayType,
    batch: usize,
) -> Result<(), TypeError> {
    for (descriptor, value_type) in
        [("query sequence lengths", query_lengths_type), ("key/value sequence lengths", key_value_lengths_type)]
    {
        if value_type.data_type() != DataType::I32 {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' {descriptor} must have data type i32 but got {}",
                value_type.data_type(),
            )));
        }
        let Some(shape) = value_type.static_shape() else {
            return Err(TypeError::invalid(format!("'{operation_name}' {descriptor} must have a static shape")));
        };
        match *shape.dimensions() {
            [size] if size == batch => {}
            [size] => {
                return Err(TypeError::invalid(format!(
                    "'{operation_name}' {descriptor} size ({size}) does not match the batch dimension ({batch})",
                )));
            }
            ref dimensions => {
                return Err(TypeError::invalid(format!(
                    "'{operation_name}' {descriptor} must have rank 1 but got rank {}",
                    dimensions.len(),
                )));
            }
        }
    }
    Ok(())
}

/// Validates the optional `(rate, seed)` dropout attribute shared by the attention operations: the rate must lie in
/// the open interval `(0, 1)` (a zero rate is expressed by omitting dropout altogether).
fn validated_dropout(operation_name: &str, dropout: Option<(f64, u64)>) -> Result<(), TypeError> {
    if let Some((rate, _)) = dropout {
        if !(rate > 0.0 && rate < 1.0) {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' dropout rate must lie in the open interval (0, 1) but got {rate}",
            )));
        }
    }
    Ok(())
}

/// Returns the static [`Shape`] with the provided dimensions.
fn static_shape(dimensions: &[usize]) -> Shape {
    Shape::new(dimensions.iter().map(|&size| Dimension::Static(size)).collect())
}

/// Returns the `[batch, q_seq, heads, head_dim]` attention output type at the operand data type.
fn attention_output_type(dimensions: &AttentionDimensions) -> ArrayType {
    ArrayType::new(
        dimensions.data_type,
        static_shape(&[dimensions.batch, dimensions.query_sequence, dimensions.query_heads, dimensions.head_dimension]),
    )
}

/// Returns the `f32[batch, heads, q_seq]` activation (log-sum-exp statistic) type of the training forward. The
/// activation inherits the query's sharding with its dimensions permuted into the `[batch, heads, q_seq]` order (and
/// the head dimension dropped), so sharded operand types keep a coherent inferred output signature.
fn attention_activation_type(dimensions: &AttentionDimensions, query_type: &ArrayType) -> Result<ArrayType, TypeError> {
    let activation_type = ArrayType::new(
        DataType::F32,
        static_shape(&[dimensions.batch, dimensions.query_heads, dimensions.query_sequence]),
    );
    let Some(query_sharding) = query_type.sharding() else {
        return Ok(activation_type);
    };
    let query_dimensions = query_sharding.dimensions();
    let sharding = Sharding::new(
        query_sharding.mesh().clone(),
        vec![query_dimensions[0].clone(), query_dimensions[2].clone(), query_dimensions[1].clone()],
    )
    .and_then(|sharding| activation_type.with_sharding(sharding))
    .map_err(|error| TypeError::invalid(error.to_string()))?;
    Ok(sharding)
}

/// Returns the data type the attention softmax runs at for the provided operand data type: `f32` for every operand
/// type narrower than `f32`, and `f64` for `f64` operands.
fn attention_softmax_data_type(data_type: DataType) -> DataType {
    if data_type == DataType::F64 { DataType::F64 } else { DataType::F32 }
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
        let (query, key, value, bias, sequence_lengths) = match inputs {
            [query, key, value] => (query, key, value, None, None),
            [query, key, value, bias] => (query, key, value, Some(bias), None),
            [query, key, value, query_lengths, key_value_lengths] => {
                (query, key, value, None, Some((query_lengths, key_value_lengths)))
            }
            [query, key, value, bias, query_lengths, key_value_lengths] => {
                (query, key, value, Some(bias), Some((query_lengths, key_value_lengths)))
            }
            _ => {
                return Err(TypeError::invalid(format!(
                    "'{DOT_PRODUCT_ATTENTION_OPERATION_NAME}' expects 3 to 6 inputs but got {}",
                    inputs.len(),
                ))
                .into());
            }
        };
        if self.activation_output {
            let (output, activation) = query.dot_product_attention_with_activation(
                key,
                value,
                bias,
                sequence_lengths,
                self.scale,
                self.mask,
                self.sliding_window,
                self.dropout,
            )?;
            Ok(vec![output, activation])
        } else {
            let output = query.dot_product_attention_with_options(
                key,
                value,
                bias,
                sequence_lengths,
                self.scale,
                self.mask,
                self.sliding_window,
                self.dropout,
            )?;
            Ok(vec![output])
        }
    }
}

impl<C: Domain<Type = ArrayType, Value: DotProductAttentionBackward>> InterpretableOperation<C>
    for DotProductAttentionBackwardOperation
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        let (query, key, value, bias, output, activation, output_cotangent, sequence_lengths) = match inputs {
            [query, key, value, output, activation, output_cotangent] => {
                (query, key, value, None, output, activation, output_cotangent, None)
            }
            [query, key, value, bias, output, activation, output_cotangent] => {
                (query, key, value, Some(bias), output, activation, output_cotangent, None)
            }
            [query, key, value, output, activation, output_cotangent, query_lengths, key_value_lengths] => (
                query,
                key,
                value,
                None,
                output,
                activation,
                output_cotangent,
                Some((query_lengths, key_value_lengths)),
            ),
            [query, key, value, bias, output, activation, output_cotangent, query_lengths, key_value_lengths] => (
                query,
                key,
                value,
                Some(bias),
                output,
                activation,
                output_cotangent,
                Some((query_lengths, key_value_lengths)),
            ),
            _ => {
                return Err(TypeError::invalid(format!(
                    "'{DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME}' expects 6 to 9 inputs but got {}",
                    inputs.len(),
                ))
                .into());
            }
        };
        let (query_cotangent, key_cotangent, value_cotangent, bias_cotangent) = query
            .dot_product_attention_backward_with_options(
                key,
                value,
                bias,
                sequence_lengths,
                output,
                activation,
                output_cotangent,
                self.scale,
                self.mask,
                self.sliding_window,
                self.dropout,
            )?;
        let mut outputs = vec![query_cotangent, key_cotangent, value_cotangent];
        if let Some(bias_cotangent) = bias_cotangent {
            outputs.push(bias_cotangent);
        }
        Ok(outputs)
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for DotProductAttentionOperation where
    C::Operation: From<DotProductAttentionOperation>
{
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for DotProductAttentionBackwardOperation where
    C::Operation: From<DotProductAttentionBackwardOperation>
{
}

impl_differentiable_operation! {
    DotProductAttentionOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Operation: From<DotProductAttentionOperation>,
    {
        |_operation, _context, _driver, _inputs| {
            // The operation is the inference fast path, so there is no differentiation rule: differentiating reports an
            // error directing users to the [`differentiable_dot_product_attention`] training entry point, which pairs
            // the activation-producing forward with [`DotProductAttentionBackwardOperation`] through [`custom_vjp`].
            Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "'{DOT_PRODUCT_ATTENTION_OPERATION_NAME}' does not support differentiation; use \
                     'differentiable_dot_product_attention' for the training path"
                ),
            }
            .into())
        }
    },
    transpose = @nonlinear,
}

impl_differentiable_operation! {
    DotProductAttentionBackwardOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Operation: From<DotProductAttentionBackwardOperation>,
    {
        |_operation, _context, _driver, _inputs| {
            // The backward operation rejects differentiation: second-order derivatives go through an explicit attention
            // composition instead of the fused backward pass.
            Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "'{DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME}' does not support differentiation; \
                     differentiate an explicit attention composition for higher-order derivatives"
                ),
            }
            .into())
        }
    },
    transpose = @nonlinear,
}

/// Shared merge-reshape batching rule for [`DotProductAttentionOperation`] and
/// [`DotProductAttentionBackwardOperation`]: attention is batch-parallel, so one mapped batch level folds into the
/// operations' own batch dimension. Every operand is aligned to a physical batch axis at position 0 (mapped operands
/// are realigned, replicated operands are broadcast into `axis_size` copies), the resulting `[v, batch, ...]`
/// operands are reshaped to `[v * batch, ...]`, the same operation runs over the merged batch, and every output
/// splits the mapped axis back out to `[v, batch, ...]` mapped at axis 0. When every operand is replicated, the
/// lifted operation is the unbatched operation itself with replicated outputs.
///
/// The optional bias operand needs special handling because its batch dimension may be a broadcast `1`: a merged
/// `[v * 1, ...]` bias could not broadcast against the merged `[v * batch, ...]` scores, so the bias batch dimension
/// is materialized to the per-item batch before merging (bias broadcasting is pointwise, so the results are
/// unchanged). The matching bias-cotangent output of the backward operation is then summed back over the
/// materialized per-item batch after splitting, restoring the broadcast `1` of the unbatched contract. The optional
/// rank-1 `i32[batch]` sequence-length operands ride the general rule unchanged: aligned `[v, batch]` lengths merge
/// to `[v * batch]`, concatenating the per-item lengths along the folded batch axis.
fn batch_attention_merge_reshape<C, O, P: ArrayBatchingPolicy<C>>(
    operation: &O,
    context: &BatchingContext<C, ArrayBatching<P>>,
    inputs: &[ArrayBatch<C::Value>],
    bias_index: Option<usize>,
    bias_cotangent_index: Option<usize>,
) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError>
where
    C: Context<Type = ArrayType, Value: LegacyBroadcast + Reduce + Reshape + Transpose>,
    O: Operation<ArrayType> + InterpretableBatchableOperation<C, ArrayBatching<P>>,
{
    let output_count = |input_types: &[ArrayType]| -> Result<usize, BatchingError> {
        Ok(operation.infer_output_types(input_types, &[]).map_err(ProgramError::from)?.len())
    };
    let Some(axis_size) = ArrayBatch::common_batch_size(inputs)? else {
        // Every operand is replicated: the lifted operation is the unbatched operation itself.
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let axes = vec![BatchAxis::replicated(); output_count(input_types.as_slice())?];
        return operation.interpret_with_batch_axes(context, inputs, axes.as_slice());
    };
    let axis_sharding = ArrayBatch::sharding_for_inputs(inputs)?;
    let static_dimensions = |value_type: &ArrayType| -> Result<Vec<usize>, BatchingError> {
        match value_type.static_shape() {
            Some(shape) => Ok(shape.dimensions().to_vec()),
            None => Err(ProgramError::from(TypeError::invalid(format!(
                "'{}' batching requires statically shaped operands",
                operation.name()
            )))
            .into()),
        }
    };
    // Fold the mapped axis into the attention batch dimension: `[v, batch, ...]` reshapes to `[v * batch, ...]` and
    // the merged operands run through the same (fused) operation.
    let aligned_inputs = inputs
        .iter()
        .map(|input| input.match_axis(0, axis_size, axis_sharding.clone()))
        .collect::<Result<Vec<_>, _>>()?;
    let per_item_batch = static_dimensions(&aligned_inputs[0].r#type())?[1];
    let mut materialized_bias_batch = false;
    let merged_inputs = aligned_inputs
        .iter()
        .enumerate()
        .map(|(index, aligned)| {
            let mut dimensions = static_dimensions(&aligned.r#type())?;
            let mut value = aligned.value().clone();
            if bias_index == Some(index) && dimensions[1] != per_item_batch {
                // A broadcast bias batch dimension is materialized to the per-item batch before merging.
                let mut materialized_dimensions = dimensions.clone();
                materialized_dimensions[1] = per_item_batch;
                let materialized_type =
                    ArrayType::new(aligned.r#type().data_type(), static_shape(materialized_dimensions.as_slice()));
                let identity_axes = (0..dimensions.len()).collect::<Vec<_>>();
                value = value.legacy_broadcast(materialized_type, identity_axes.as_slice())?;
                dimensions = materialized_dimensions;
                materialized_bias_batch = true;
            }
            let merged_dimensions = std::iter::once(Dimension::Static(dimensions[0] * dimensions[1]))
                .chain(dimensions[2..].iter().map(|&size| Dimension::Static(size)))
                .collect();
            let merged_value = value.reshape(Shape::new(merged_dimensions))?;
            Ok(ArrayBatch::replicated(merged_value))
        })
        .collect::<Result<Vec<_>, BatchingError>>()?;
    let merged_types = merged_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
    let axes = vec![BatchAxis::replicated(); output_count(merged_types.as_slice())?];
    let outputs = operation.interpret_with_batch_axes(context, merged_inputs.as_slice(), axes.as_slice())?;
    // Split the mapped axis back out of each merged output batch dimension and map the result at axis 0.
    outputs
        .into_iter()
        .enumerate()
        .map(|(index, output)| {
            let output_dimensions = static_dimensions(&output.r#type())?;
            let split_dimensions = [axis_size, output_dimensions[0] / axis_size]
                .into_iter()
                .chain(output_dimensions[1..].iter().copied())
                .map(Dimension::Static)
                .collect();
            let mut split_value = output.value().reshape(Shape::new(split_dimensions))?;
            if materialized_bias_batch && bias_cotangent_index == Some(index) {
                // The bias cotangent sums back over the materialized per-item batch, restoring the broadcast `1`.
                let summed = split_value.reduce(&[1], ReductionKind::Sum);
                let summed_dimensions = static_dimensions(&summed.r#type())?;
                let mut restored_dimensions = summed_dimensions.clone();
                restored_dimensions.insert(1, 1);
                split_value = summed.reshape(static_shape(restored_dimensions.as_slice()))?;
            }
            let split_type = split_value.r#type().into_owned();
            Ok(ArrayBatch::new(split_type, split_value, BatchAxis::new(0))?)
        })
        .collect()
}

/// Batching rule for [`DotProductAttentionOperation`]: one mapped batch level folds into the operation's own batch
/// dimension via the shared merge-reshape rule; refer to [`batch_attention_merge_reshape`].
impl<C: Context<Type = ArrayType, Value: LegacyBroadcast + Reduce + Reshape + Transpose>, P: ArrayBatchingPolicy<C>>
    BatchableOperation<C, ArrayBatching<P>> for DotProductAttentionOperation
where
    DotProductAttentionOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        let bias_index = matches!(inputs.len(), 4 | 6).then_some(3);
        batch_attention_merge_reshape(self, context, inputs, bias_index, None)
    }
}

/// Batching rule for [`DotProductAttentionBackwardOperation`]: the same merge-reshape rule as the forward
/// operation, additionally restoring a broadcast bias-cotangent batch dimension; refer to
/// [`batch_attention_merge_reshape`].
impl<C: Context<Type = ArrayType, Value: LegacyBroadcast + Reduce + Reshape + Transpose>, P: ArrayBatchingPolicy<C>>
    BatchableOperation<C, ArrayBatching<P>> for DotProductAttentionBackwardOperation
where
    DotProductAttentionBackwardOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        let bias_index = matches!(inputs.len(), 7 | 9).then_some(3);
        batch_attention_merge_reshape(self, context, inputs, bias_index, bias_index)
    }
}

/// Value-level scaled dot-product attention capability. Refer to the documentation of
/// [`DotProductAttentionOperation`] for the `BTNH` operand convention, the exact semantics, and the transform rules.
pub trait DotProductAttention: Sized {
    /// Computes scaled dot-product attention with `self` as the query (shape `[batch, q_seq, heads, head_dim]`)
    /// over `key`/`value` (shape `[batch, kv_seq, kv_heads, head_dim]` with `kv_heads` dividing `heads`), returning
    /// the attended `[batch, q_seq, heads, head_dim]` output at the operand data type, and a [`ProgramError`] if
    /// something goes wrong.
    ///
    /// # Parameters
    ///
    ///   - `key`: Key operand aligned with `value` along the key/value sequence dimension.
    ///   - `value`: Value operand whose rows are mixed by the attention weights.
    ///   - `scale`: Multiplier applied to the attention scores before masking and softmax.
    ///   - `mask`: Built-in [`AttentionMask`] applied to the attention scores before the softmax.
    ///   - `sliding_window`: Optional sliding-window width tightening the causal mask; refer to
    ///     [`DotProductAttentionOperation::with_sliding_window`].
    fn dot_product_attention(
        &self,
        key: &Self,
        value: &Self,
        scale: f64,
        mask: AttentionMask,
        sliding_window: Option<usize>,
    ) -> Result<Self, ProgramError> {
        self.dot_product_attention_with_options(key, value, None, None, scale, mask, sliding_window, None)
    }

    /// Computes scaled dot-product attention like [`dot_product_attention`](Self::dot_product_attention) with an
    /// additional `bias` of shape `[batch | 1, heads | 1, q_seq, kv_seq]` at the operand data type, added to the
    /// already scaled scores before masking and softmax.
    ///
    /// # Parameters
    ///
    ///   - `key`: Key operand aligned with `value` along the key/value sequence dimension.
    ///   - `value`: Value operand whose rows are mixed by the attention weights.
    ///   - `bias`: Bias added to the scaled attention scores (its leading two dimensions broadcast).
    ///   - `scale`: Multiplier applied to the attention scores before the bias, masking, and softmax.
    ///   - `mask`: Built-in [`AttentionMask`] applied to the attention scores before the softmax.
    ///   - `sliding_window`: Optional sliding-window width tightening the causal mask.
    fn dot_product_attention_with_bias(
        &self,
        key: &Self,
        value: &Self,
        bias: &Self,
        scale: f64,
        mask: AttentionMask,
        sliding_window: Option<usize>,
    ) -> Result<Self, ProgramError> {
        self.dot_product_attention_with_options(key, value, Some(bias), None, scale, mask, sliding_window, None)
    }

    /// Computes scaled dot-product attention in its full inference form: an optional bias, optional variable
    /// sequence lengths (out-of-range key/value columns fully excluded and out-of-range query rows exact zeros —
    /// refer to [`DotProductAttentionOperation`]), and optional dropout (implemented only by the fused cuDNN
    /// kernels).
    ///
    /// # Parameters
    ///
    ///   - `key`: Key operand aligned with `value` along the key/value sequence dimension.
    ///   - `value`: Value operand whose rows are mixed by the attention weights.
    ///   - `bias`: Optional bias added to the scaled attention scores (its leading two dimensions broadcast).
    ///   - `sequence_lengths`: Optional `(query, key/value)` per-batch-item sequence lengths, both `i32[batch]`.
    ///   - `scale`: Multiplier applied to the attention scores before the bias, masking, and softmax.
    ///   - `mask`: Built-in [`AttentionMask`] applied to the attention scores before the softmax.
    ///   - `sliding_window`: Optional sliding-window width tightening the causal mask.
    ///   - `dropout`: Optional `(rate, seed)` dropout applied to the attention weights.
    #[allow(clippy::too_many_arguments)]
    fn dot_product_attention_with_options(
        &self,
        key: &Self,
        value: &Self,
        bias: Option<&Self>,
        sequence_lengths: Option<(&Self, &Self)>,
        scale: f64,
        mask: AttentionMask,
        sliding_window: Option<usize>,
        dropout: Option<(f64, u64)>,
    ) -> Result<Self, ProgramError>;

    /// Computes the training forward of scaled dot-product attention: the attended output together with the
    /// `f32[batch, heads, q_seq]` natural-log log-sum-exp activation statistic of the masked logits over the
    /// key/value axis, which [`DotProductAttentionBackward`] consumes. The optional bias, sequence lengths, and
    /// dropout follow [`dot_product_attention_with_options`](Self::dot_product_attention_with_options); under
    /// variable sequence lengths the out-of-range query rows of both outputs are exact zeros.
    ///
    /// # Parameters
    ///
    ///   - `key`: Key operand aligned with `value` along the key/value sequence dimension.
    ///   - `value`: Value operand whose rows are mixed by the attention weights.
    ///   - `bias`: Optional bias added to the scaled attention scores (its leading two dimensions broadcast).
    ///   - `sequence_lengths`: Optional `(query, key/value)` per-batch-item sequence lengths, both `i32[batch]`.
    ///   - `scale`: Multiplier applied to the attention scores before the bias, masking, and softmax.
    ///   - `mask`: Built-in [`AttentionMask`] applied to the attention scores before the softmax.
    ///   - `sliding_window`: Optional sliding-window width tightening the causal mask.
    ///   - `dropout`: Optional `(rate, seed)` dropout applied to the attention weights.
    #[allow(clippy::too_many_arguments)]
    fn dot_product_attention_with_activation(
        &self,
        key: &Self,
        value: &Self,
        bias: Option<&Self>,
        sequence_lengths: Option<(&Self, &Self)>,
        scale: f64,
        mask: AttentionMask,
        sliding_window: Option<usize>,
        dropout: Option<(f64, u64)>,
    ) -> Result<(Self, Self), ProgramError>;
}

/// Value-level backward (gradient) pass of scaled dot-product attention. Refer to the documentation of
/// [`DotProductAttentionBackwardOperation`] for the operand convention and the exact semantics.
pub trait DotProductAttentionBackward: Sized {
    /// Computes the query/key/value cotangents of scaled dot-product attention with `self` as the forward query.
    ///
    /// # Parameters
    ///
    ///   - `key`: Key operand of the forward pass.
    ///   - `value`: Value operand of the forward pass.
    ///   - `output`: Attended output produced by the forward pass.
    ///   - `activation`: `f32[batch, heads, q_seq]` log-sum-exp statistic produced by the forward pass.
    ///   - `output_cotangent`: Incoming cotangent of the forward output.
    ///   - `scale`: Score multiplier of the forward pass.
    ///   - `mask`: Built-in [`AttentionMask`] of the forward pass.
    ///   - `sliding_window`: Optional sliding-window width of the forward pass.
    #[allow(clippy::too_many_arguments)]
    fn dot_product_attention_backward(
        &self,
        key: &Self,
        value: &Self,
        output: &Self,
        activation: &Self,
        output_cotangent: &Self,
        scale: f64,
        mask: AttentionMask,
        sliding_window: Option<usize>,
    ) -> Result<(Self, Self, Self), ProgramError> {
        let (query_cotangent, key_cotangent, value_cotangent, _) = self.dot_product_attention_backward_with_options(
            key,
            value,
            None,
            None,
            output,
            activation,
            output_cotangent,
            scale,
            mask,
            sliding_window,
            None,
        )?;
        Ok((query_cotangent, key_cotangent, value_cotangent))
    }

    /// Computes the query/key/value/bias cotangents of scaled dot-product attention with a bias operand, like
    /// [`dot_product_attention_backward`](Self::dot_product_attention_backward). The bias cotangent is shaped like
    /// the bias operand, summing over its broadcast leading dimensions.
    ///
    /// # Parameters
    ///
    ///   - `key`: Key operand of the forward pass.
    ///   - `value`: Value operand of the forward pass.
    ///   - `bias`: Bias operand of the forward pass (its leading two dimensions broadcast).
    ///   - `output`: Attended output produced by the forward pass.
    ///   - `activation`: `f32[batch, heads, q_seq]` log-sum-exp statistic produced by the forward pass.
    ///   - `output_cotangent`: Incoming cotangent of the forward output.
    ///   - `scale`: Score multiplier of the forward pass.
    ///   - `mask`: Built-in [`AttentionMask`] of the forward pass.
    ///   - `sliding_window`: Optional sliding-window width of the forward pass.
    #[allow(clippy::too_many_arguments)]
    fn dot_product_attention_backward_with_bias(
        &self,
        key: &Self,
        value: &Self,
        bias: &Self,
        output: &Self,
        activation: &Self,
        output_cotangent: &Self,
        scale: f64,
        mask: AttentionMask,
        sliding_window: Option<usize>,
    ) -> Result<(Self, Self, Self, Self), ProgramError> {
        let (query_cotangent, key_cotangent, value_cotangent, bias_cotangent) = self
            .dot_product_attention_backward_with_options(
                key,
                value,
                Some(bias),
                None,
                output,
                activation,
                output_cotangent,
                scale,
                mask,
                sliding_window,
                None,
            )?;
        // The full form returns a bias cotangent whenever a bias operand is provided.
        Ok((query_cotangent, key_cotangent, value_cotangent, bias_cotangent.unwrap()))
    }

    /// Computes the operand cotangents of scaled dot-product attention in its full form: an optional bias (whose
    /// cotangent is returned as the fourth value exactly when the bias is provided), optional variable sequence
    /// lengths, and optional dropout, all matching the forward pass — refer to
    /// [`DotProductAttentionBackwardOperation`] for the exact semantics.
    ///
    /// # Parameters
    ///
    ///   - `key`: Key operand of the forward pass.
    ///   - `value`: Value operand of the forward pass.
    ///   - `bias`: Optional bias operand of the forward pass (its leading two dimensions broadcast).
    ///   - `sequence_lengths`: Optional `(query, key/value)` per-batch-item sequence lengths of the forward pass.
    ///   - `output`: Attended output produced by the forward pass.
    ///   - `activation`: `f32[batch, heads, q_seq]` log-sum-exp statistic produced by the forward pass.
    ///   - `output_cotangent`: Incoming cotangent of the forward output.
    ///   - `scale`: Score multiplier of the forward pass.
    ///   - `mask`: Built-in [`AttentionMask`] of the forward pass.
    ///   - `sliding_window`: Optional sliding-window width of the forward pass.
    ///   - `dropout`: Optional `(rate, seed)` dropout of the forward pass.
    #[allow(clippy::too_many_arguments)]
    fn dot_product_attention_backward_with_options(
        &self,
        key: &Self,
        value: &Self,
        bias: Option<&Self>,
        sequence_lengths: Option<(&Self, &Self)>,
        output: &Self,
        activation: &Self,
        output_cotangent: &Self,
        scale: f64,
        mask: AttentionMask,
        sliding_window: Option<usize>,
        dropout: Option<(f64, u64)>,
    ) -> Result<(Self, Self, Self, Option<Self>), ProgramError>;
}

/// Any context-carrying value computes attention by binding a [`DotProductAttentionOperation`] through its own
/// context. The `From<DotProductAttentionOperation>` bound makes this disjoint from the eager reference value types
/// (whose context operation is [`ConstantOperation`](crate::operations::constants::ConstantOperation)), so it covers
/// the transform tracers and backend-owned values without conflicting with concrete implementations.
impl<V: Value<Type = ArrayType>> DotProductAttention for V
where
    V::DispatchDomain: Context<Operation: From<DotProductAttentionOperation>>,
{
    fn dot_product_attention_with_options(
        &self,
        key: &Self,
        value: &Self,
        bias: Option<&Self>,
        sequence_lengths: Option<(&Self, &Self)>,
        scale: f64,
        mask: AttentionMask,
        sliding_window: Option<usize>,
        dropout: Option<(f64, u64)>,
    ) -> Result<Self, ProgramError> {
        let operation = DotProductAttentionOperation::new(scale, mask)
            .with_sliding_window(sliding_window)
            .with_dropout(dropout);
        let operands = attention_operands(self, key, value, bias, sequence_lengths);
        let mut outputs = self.dispatch_domain().bind(operation, Vec::new(), operands.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }

    fn dot_product_attention_with_activation(
        &self,
        key: &Self,
        value: &Self,
        bias: Option<&Self>,
        sequence_lengths: Option<(&Self, &Self)>,
        scale: f64,
        mask: AttentionMask,
        sliding_window: Option<usize>,
        dropout: Option<(f64, u64)>,
    ) -> Result<(Self, Self), ProgramError> {
        let operation = DotProductAttentionOperation::new(scale, mask)
            .with_sliding_window(sliding_window)
            .with_dropout(dropout)
            .with_activation_output();
        let operands = attention_operands(self, key, value, bias, sequence_lengths);
        let mut outputs = self.dispatch_domain().bind(operation, Vec::new(), operands.as_slice())?;
        check_count!("output", outputs, 2, ProgramError);
        let activation = outputs.remove(1);
        Ok((outputs.remove(0), activation))
    }
}

/// Collects the forward attention operands in the operation's operand order: `query`, `key`, and `value`, followed
/// by the optional bias and the optional trailing sequence-length pair.
fn attention_operands<V: Clone>(
    query: &V,
    key: &V,
    value: &V,
    bias: Option<&V>,
    sequence_lengths: Option<(&V, &V)>,
) -> Vec<V> {
    let mut operands = vec![query.clone(), key.clone(), value.clone()];
    if let Some(bias) = bias {
        operands.push(bias.clone());
    }
    if let Some((query_lengths, key_value_lengths)) = sequence_lengths {
        operands.push(query_lengths.clone());
        operands.push(key_value_lengths.clone());
    }
    operands
}

/// Any context-carrying value computes the attention backward pass by binding a
/// [`DotProductAttentionBackwardOperation`] through its own context; refer to the [`DotProductAttention`] blanket
/// implementation for the disjointness argument.
impl<V: Value<Type = ArrayType>> DotProductAttentionBackward for V
where
    V::DispatchDomain: Context<Operation: From<DotProductAttentionBackwardOperation>>,
{
    fn dot_product_attention_backward_with_options(
        &self,
        key: &Self,
        value: &Self,
        bias: Option<&Self>,
        sequence_lengths: Option<(&Self, &Self)>,
        output: &Self,
        activation: &Self,
        output_cotangent: &Self,
        scale: f64,
        mask: AttentionMask,
        sliding_window: Option<usize>,
        dropout: Option<(f64, u64)>,
    ) -> Result<(Self, Self, Self, Option<Self>), ProgramError> {
        let operation = DotProductAttentionBackwardOperation::new(scale, mask)
            .with_sliding_window(sliding_window)
            .with_dropout(dropout);
        let mut operands = vec![self.clone(), key.clone(), value.clone()];
        if let Some(bias) = bias {
            operands.push(bias.clone());
        }
        operands.extend([output.clone(), activation.clone(), output_cotangent.clone()]);
        if let Some((query_lengths, key_value_lengths)) = sequence_lengths {
            operands.push(query_lengths.clone());
            operands.push(key_value_lengths.clone());
        }
        let expected_output_count = if bias.is_some() { 4 } else { 3 };
        let mut outputs = self.dispatch_domain().bind(operation, Vec::new(), operands.as_slice())?;
        check_count!("output", outputs, expected_output_count, ProgramError);
        let bias_cotangent = bias.is_some().then(|| outputs.remove(3));
        let value_cotangent = outputs.remove(2);
        let key_cotangent = outputs.remove(1);
        Ok((outputs.remove(0), key_cotangent, value_cotangent, bias_cotangent))
    }
}

/// Fills a constant of `type`'s (floating-point) data type from an `f64` value, converting the scalar to the exact
/// element data type so the fill matches the target type (the softmax data type for score constants, and the operand
/// data type for the exact zeros written into out-of-range padded rows).
fn attention_fill<C, V>(context: &C, r#type: &ArrayType, value: f64) -> Result<V, ProgramError>
where
    V: Typed<Type = ArrayType>,
    C: Fill<Scalar, V>,
{
    context.fill(r#type, Scalar::from(value).convert_element_type(r#type.data_type())?)
}

/// Expands grouped key/value heads to one head per query head for the attention compositions: a
/// `[batch, kv_seq, kv_heads, head_dim]` operand broadcasts to `[batch, kv_seq, kv_heads, group, head_dim]` (with
/// `group = heads / kv_heads`) and reshapes to `[batch, kv_seq, heads, head_dim]`, so each key/value head is
/// repeated `group` times consecutively and query head `i` attends key/value head `i / group`. Operands that
/// already carry one key/value head per query head are returned unchanged.
fn expand_key_value_heads<V>(operand: &V, dimensions: &AttentionDimensions) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayType> + LegacyBroadcast + Reshape,
{
    if dimensions.key_value_heads == dimensions.query_heads {
        return Ok(operand.clone());
    }
    let group = dimensions.query_heads / dimensions.key_value_heads;
    let expanded_type = ArrayType::new(
        dimensions.data_type,
        static_shape(&[
            dimensions.batch,
            dimensions.key_value_sequence,
            dimensions.key_value_heads,
            group,
            dimensions.head_dimension,
        ]),
    );
    operand.legacy_broadcast(expanded_type, &[0, 1, 2, 4])?.reshape(static_shape(&[
        dimensions.batch,
        dimensions.key_value_sequence,
        dimensions.query_heads,
        dimensions.head_dimension,
    ]))
}

/// Applies the built-in attention masks to `scores` (shape `[batch, heads, q_seq, kv_seq]` at the softmax data
/// type), replacing every masked position with `masked_fill` (`-1e30` in the compositions): the causal mask keeps a
/// score position when its column (key/value) index does not exceed its row (query) index — the top-left alignment
/// validated against the cuDNN kernels, where row `r` sees keys `0..=r` — and a sliding window additionally
/// requires `column > row - window`, so row `r` attends keys `[max(0, r + 1 - window), r]`. Optional
/// `key_value_sequence_lengths` (`i32[batch]`, broadcast per batch item) additionally require
/// `column < key_value_sequence_lengths[b]`, fully excluding out-of-range key/value columns — the padding
/// composition validated against the cuDNN `PADDING`/`PADDING_CAUSAL` kernels.
fn apply_attention_masks<C, V>(
    context: &C,
    scores: V,
    mask: AttentionMask,
    sliding_window: Option<usize>,
    key_value_sequence_lengths: Option<&V>,
    masked_fill: &V,
) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayType> + And + LegacyBroadcast + Compare<V> + Select + Sub,
    C: Fill<Scalar, V> + Iota<V>,
{
    if mask == AttentionMask::None && key_value_sequence_lengths.is_none() {
        return Ok(scores);
    }
    let index_type = ArrayType::new(DataType::I32, scores.r#type().shape().clone());
    let columns = context.iota(&index_type, 3)?;
    let mut visible = None;
    if mask == AttentionMask::Causal {
        let rows = context.iota(&index_type, 2)?;
        let mut causal_visible = columns.compare(&rows, ComparisonDirection::LessThanOrEqual)?;
        if let Some(window) = sliding_window {
            let window = i32::try_from(window)
                .map_err(|_| TypeError::invalid("sliding window must fit in a 32-bit integer".to_string()))?;
            let lower_bound = rows.sub(&context.fill(&index_type, Scalar::from(window))?)?;
            causal_visible = causal_visible.and(&columns.compare(&lower_bound, ComparisonDirection::GreaterThan)?)?;
        }
        visible = Some(causal_visible);
    }
    if let Some(lengths) = key_value_sequence_lengths {
        // The `[batch]` lengths broadcast against the `[batch, heads, q_seq, kv_seq]` column indices.
        let bounds = lengths.legacy_broadcast(index_type, &[0])?;
        let in_range = columns.compare(&bounds, ComparisonDirection::LessThan)?;
        visible = Some(match visible {
            None => in_range,
            Some(visible) => visible.and(&in_range)?,
        });
    }
    // At least one mask contributed a visibility condition given the early return above.
    V::select(&visible.unwrap(), &scores, masked_fill)
}

/// Computes the masked attention logits shared by the forward and backward compositions: the
/// `query · expanded-keyᵀ` scores per batch item and head (`[batch, heads, q_seq, kv_seq]`, contracting the head
/// dimension with batch dimensions `[0, 2]` on both sides), converted to the softmax data type, multiplied by
/// `scale`, shifted by the optional broadcast `bias` (converted to the softmax data type alongside the scores), and
/// masked via [`apply_attention_masks`] (including the optional key/value sequence-length column exclusion).
fn attention_logits<C, V>(
    context: &C,
    query: &V,
    expanded_key: &V,
    bias: Option<&V>,
    key_value_sequence_lengths: Option<&V>,
    scale: f64,
    mask: AttentionMask,
    sliding_window: Option<usize>,
) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayType>
        + Add
        + And
        + LegacyBroadcast
        + Compare<V>
        + ConvertElementType
        + Dot
        + Mul
        + Select
        + Sub,
    C: Fill<Scalar, V> + Iota<V>,
{
    let data_type = query.r#type().data_type();
    // Scores over `[batch, heads]`: `query [b, qs, n, d] · key [b, ks, n, d]` contracting `d` -> `[b, n, qs, ks]`.
    let scores = query.dot(expanded_key, &DotDimensionNumbers::new(vec![3], vec![3], vec![0, 2], vec![0, 2]));
    let softmax_type = attention_softmax_data_type(data_type);
    let scores = if data_type == softmax_type { scores } else { scores.convert_element_type(softmax_type)? };
    let scores_type = scores.r#type().into_owned();
    let scores = scores.mul(&attention_fill(context, &scores_type, scale)?)?;
    let scores = match bias {
        None => scores,
        Some(bias) => {
            let bias = if bias.r#type().data_type() == softmax_type {
                bias.clone()
            } else {
                bias.convert_element_type(softmax_type)?
            };
            scores.add(&bias.legacy_broadcast(scores_type.clone(), &[0, 1, 2, 3])?)?
        }
    };
    apply_attention_masks(
        context,
        scores,
        mask,
        sliding_window,
        key_value_sequence_lengths,
        &attention_fill(context, &scores_type, -1.0e30)?,
    )
}

/// Replaces the out-of-range query rows of `value` (`row >= query_sequence_lengths[b]` along `row_axis`, with the
/// `i32[batch]` lengths broadcast per batch item along axis 0) with exact zeros at `value`'s own data type. This is
/// the composition counterpart of XLA memzeroing every fMHA output: a fully padded query row's softmax is otherwise
/// garbage (all its logits are finite or uniformly masked), so the explicit zeroing select is mandatory for the
/// attended output, the activation statistic, and the query cotangent, and it likewise sanitizes the incoming output
/// cotangent before the backward contractions.
fn zero_out_of_range_query_rows<C, V>(
    context: &C,
    value: V,
    query_sequence_lengths: &V,
    row_axis: usize,
) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayType> + LegacyBroadcast + Compare<V> + Select,
    C: Fill<Scalar, V> + Iota<V>,
{
    let value_type = value.r#type().into_owned();
    let index_type = ArrayType::new(DataType::I32, value_type.shape().clone());
    let rows = context.iota(&index_type, row_axis)?;
    let bounds = query_sequence_lengths.legacy_broadcast(index_type, &[0])?;
    let in_range = rows.compare(&bounds, ComparisonDirection::LessThan)?;
    V::select(&in_range, &value, &attention_fill(context, &value_type, 0.0)?)
}

/// Evaluates scaled dot-product attention as the portable composition: grouped key/value heads are expanded to one
/// head per query head (see [`expand_key_value_heads`]), the masked logits are computed via [`attention_logits`]
/// (scores at the operand data type, converted to the softmax data type — `f32` for operand types narrower than
/// `f32`, matching the XLA attention path and keeping low-precision softmaxes stable, while `f64` stays `f64` —
/// scaled, shifted by the optional broadcast bias, and masked to `-1e30`, including the optional key/value
/// sequence-length column exclusion), passed through a max-stabilized softmax over the last axis, converted back to
/// the operand data type, contracted with the expanded `value`, and transposed back to the `BTNH` output layout.
/// When `activation` is requested, the second returned value is the `f32[batch, heads, q_seq]` natural-log
/// log-sum-exp statistic of the masked logits over the key/value axis, computed as `max + ln(sum)` from the
/// softmax's own reductions. Under variable sequence lengths the out-of-range query rows of the attended output and
/// the activation statistic are forced to exact zeros via [`zero_out_of_range_query_rows`]. Dropout is rejected here
/// because only the fused CUDA lowering implements it. This is the shared semantics behind the concrete
/// [`DotProductAttention`] implementations and the portable XLA lowering.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dot_product_attention_composition<C, V>(
    context: &C,
    query: &V,
    key: &V,
    value: &V,
    bias: Option<&V>,
    sequence_lengths: Option<(&V, &V)>,
    scale: f64,
    mask: AttentionMask,
    sliding_window: Option<usize>,
    dropout: Option<(f64, u64)>,
    activation: bool,
) -> Result<(V, Option<V>), ProgramError>
where
    V: Value<Type = ArrayType>
        + Add
        + And
        + LegacyBroadcast
        + Compare<V>
        + ConvertElementType
        + Div
        + Dot
        + Exp
        + Log
        + Mul
        + Reduce
        + Reshape
        + Select
        + Sub
        + Transpose,
    C: Fill<Scalar, V> + Iota<V>,
{
    if dropout.is_some() {
        return Err(ProgramError::UnsupportedOperation {
            message: format!(
                "'{DOT_PRODUCT_ATTENTION_OPERATION_NAME}' dropout is only supported by the fused CUDA lowering"
            ),
        });
    }
    // Validate the operand contract up front through the operation's own type inference, so the eager reference
    // route reports the same precise type errors as staged binding.
    let mut operation = DotProductAttentionOperation::new(scale, mask).with_sliding_window(sliding_window);
    if activation {
        operation = operation.with_activation_output();
    }
    let mut input_types = vec![query.r#type().into_owned(), key.r#type().into_owned(), value.r#type().into_owned()];
    if let Some(bias) = bias {
        input_types.push(bias.r#type().into_owned());
    }
    let bias_type = bias.map(|_| &input_types[3]).cloned();
    if let Some((query_lengths, key_value_lengths)) = sequence_lengths {
        input_types.push(query_lengths.r#type().into_owned());
        input_types.push(key_value_lengths.r#type().into_owned());
    }
    operation.infer_output_types(input_types.as_slice(), &[])?;
    let dimensions = validated_attention_operands(
        DOT_PRODUCT_ATTENTION_OPERATION_NAME,
        &input_types[0],
        &input_types[1],
        &input_types[2],
        bias_type.as_ref(),
        mask,
        sliding_window,
    )?;
    let data_type = dimensions.data_type;
    let expanded_key = expand_key_value_heads(key, &dimensions)?;
    let expanded_value = expand_key_value_heads(value, &dimensions)?;
    let key_value_lengths = sequence_lengths.map(|(_, key_value_lengths)| key_value_lengths);
    let logits = attention_logits(context, query, &expanded_key, bias, key_value_lengths, scale, mask, sliding_window)?;
    let logits_type = logits.r#type().into_owned();
    let softmax_type = logits_type.data_type();
    // Max-stabilized softmax over the key/value sequence (last) axis.
    let logit_axes = &[0, 1, 2];
    let maxima = logits.reduce(&[3], ReductionKind::Max);
    let exponentials = logits.sub(&maxima.legacy_broadcast(logits_type.clone(), logit_axes)?)?.exp()?;
    let sums = exponentials.reduce(&[3], ReductionKind::Sum);
    let weights = exponentials.div(&sums.legacy_broadcast(logits_type, logit_axes)?)?;
    let weights = if data_type == softmax_type { weights } else { weights.convert_element_type(data_type)? };
    // Context values: `weights [b, n, qs, ks] · value [b, ks, n, d]` contracting `ks` -> `[b, n, qs, d]`, then
    // transposed back to the `BTNH` output layout `[b, qs, n, d]`.
    let attended = weights.dot(&expanded_value, &DotDimensionNumbers::new(vec![3], vec![1], vec![0, 1], vec![0, 2]));
    let mut output = attended.transpose([0, 2, 1, 3])?;
    if let Some((query_lengths, _)) = sequence_lengths {
        output = zero_out_of_range_query_rows(context, output, query_lengths, 1)?;
    }
    let activation_output = if activation {
        // The log-sum-exp statistic reuses the softmax reductions: `stat = max + ln(sum)` rowwise over the kv axis.
        let statistic = maxima.add(&sums.log()?)?;
        let mut statistic =
            if softmax_type == DataType::F32 { statistic } else { statistic.convert_element_type(DataType::F32)? };
        if let Some((query_lengths, _)) = sequence_lengths {
            statistic = zero_out_of_range_query_rows(context, statistic, query_lengths, 2)?;
        }
        Some(statistic)
    } else {
        None
    };
    Ok((output, activation_output))
}

/// Evaluates the backward (gradient) pass of scaled dot-product attention as the portable composition — the
/// standard attention backward, computed at the softmax data type (`f32` for operand types narrower than `f32`;
/// `f64` operands keep an `f64` computation) with the cotangents converted back to the operand data type. With
/// `Q [b, t, n, h]`, expanded `K`/`V [b, s, n, h]` (see [`expand_key_value_heads`]), output `O` and its cotangent
/// `dO [b, t, n, h]`, and the forward's log-sum-exp statistic `stat [b, n, t]`:
///
///   1. The masked logits `S [b, n, t, s]` are recomputed via [`attention_logits`], and the attention weights are
///      recovered as `P = exp(S - stat)` (masked positions carry `-1e30` logits, so they recover exactly zero
///      weight and contribute no gradient).
///   2. `dP[b, n, t, s] = Σ_h dO[b, t, n, h] · V[b, s, n, h]` (contracting the head axis with batch dimensions
///      `[0, 2]` on both sides).
///   3. `delta[b, n, t] = Σ_h dO[b, t, n, h] · O[b, t, n, h]` (elementwise product reduced over the head axis and
///      transposed from `[b, t, n]`), and `dS = P ∘ (dP - delta)` with `delta` broadcast over the kv axis.
///   4. `dQ[b, t, n, h] = scale · Σ_s dS[b, n, t, s] · K[b, s, n, h]` (contract axes `3/1`, batch `[0, 1]/[0, 2]`,
///      result `[b, n, t, h]` transposed to `BTNH`); `dK[b, s, n, h] = scale · Σ_t dS[b, n, t, s] · Q[b, t, n, h]`
///      and `dV[b, s, n, h] = Σ_t P[b, n, t, s] · dO[b, t, n, h]` (both contract axes `2/1`, batch `[0, 1]/[0, 2]`,
///      result `[b, n, s, h]` transposed to `[b, s, n, h]`). The `scale` factor enters `dQ`/`dK` because the logits
///      are `scale · (Q·Kᵀ) + bias`, while the bias cotangent reads `dS` unscaled.
///   5. Grouped-query attention sums `dK`/`dV` over the per-head group axis (reshaping `[b, s, n, h]` to
///      `[b, s, kv_heads, group, h]` and reducing the group axis), and the bias cotangent sums `dS` over the bias's
///      broadcast leading dimensions.
///
/// Under variable sequence lengths (the optional `i32[batch]` pair, exactly as in the forward composition) the
/// logits recomputation excludes the out-of-range key/value columns, the out-of-range query rows of `dO` are zeroed
/// before any contraction (so `dK`/`dV` receive no contribution from them), and the corresponding `dQ` rows are
/// forced to exact zeros; the `dK`/`dV` columns at or beyond `key_value_sequence_lengths[b]` end exactly zero
/// through the zero recovered weights. Dropout is rejected here because only the fused CUDA lowering implements it.
///
/// The returned cotangents are `[dQ, dK, dV]`, plus `dBias` when `bias` is present. This is the shared semantics
/// behind the concrete [`DotProductAttentionBackward`] implementations.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dot_product_attention_backward_composition<C, V>(
    context: &C,
    query: &V,
    key: &V,
    value: &V,
    bias: Option<&V>,
    sequence_lengths: Option<(&V, &V)>,
    output: &V,
    activation: &V,
    output_cotangent: &V,
    scale: f64,
    mask: AttentionMask,
    sliding_window: Option<usize>,
    dropout: Option<(f64, u64)>,
) -> Result<Vec<V>, ProgramError>
where
    V: Value<Type = ArrayType>
        + Add
        + And
        + LegacyBroadcast
        + Compare<V>
        + ConvertElementType
        + Dot
        + Exp
        + Mul
        + Reduce
        + Reshape
        + Select
        + Sub
        + Transpose,
    C: Fill<Scalar, V> + Iota<V>,
{
    if dropout.is_some() {
        return Err(ProgramError::UnsupportedOperation {
            message: format!(
                "'{DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME}' dropout is only supported by the fused CUDA \
                 lowering"
            ),
        });
    }
    // Validate the operand contract up front through the operation's own type inference, so the eager reference
    // route reports the same precise type errors as staged binding.
    let operation = DotProductAttentionBackwardOperation::new(scale, mask).with_sliding_window(sliding_window);
    let mut input_types = vec![query.r#type().into_owned(), key.r#type().into_owned(), value.r#type().into_owned()];
    if let Some(bias) = bias {
        input_types.push(bias.r#type().into_owned());
    }
    let bias_type = bias.map(|_| &input_types[3]).cloned();
    input_types.push(output.r#type().into_owned());
    input_types.push(activation.r#type().into_owned());
    input_types.push(output_cotangent.r#type().into_owned());
    if let Some((query_lengths, key_value_lengths)) = sequence_lengths {
        input_types.push(query_lengths.r#type().into_owned());
        input_types.push(key_value_lengths.r#type().into_owned());
    }
    operation.infer_output_types(input_types.as_slice(), &[])?;
    let dimensions = validated_attention_operands(
        DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME,
        &input_types[0],
        &input_types[1],
        &input_types[2],
        bias_type.as_ref(),
        mask,
        sliding_window,
    )?;
    let data_type = dimensions.data_type;
    let softmax_type = attention_softmax_data_type(data_type);
    let expanded_key = expand_key_value_heads(key, &dimensions)?;
    let expanded_value = expand_key_value_heads(value, &dimensions)?;
    // Recompute the masked logits exactly as the forward does and recover the attention weights from the stashed
    // log-sum-exp statistic: `P = exp(S - stat)`.
    let key_value_lengths = sequence_lengths.map(|(_, key_value_lengths)| key_value_lengths);
    let logits = attention_logits(context, query, &expanded_key, bias, key_value_lengths, scale, mask, sliding_window)?;
    let logits_type = logits.r#type().into_owned();
    let logit_axes = &[0, 1, 2];
    let statistic = if softmax_type == DataType::F32 {
        activation.clone()
    } else {
        activation.convert_element_type(softmax_type)?
    };
    let weights = logits.sub(&statistic.legacy_broadcast(logits_type.clone(), logit_axes)?)?.exp()?;
    // Out-of-range query rows of the incoming output cotangent are zeroed before any contraction so the key/value
    // cotangents receive no contribution from them (the forward memzeroes those output rows, so their recovered
    // weights are unreliable).
    let output_cotangent = match sequence_lengths {
        None => output_cotangent.clone(),
        Some((query_lengths, _)) => zero_out_of_range_query_rows(context, output_cotangent.clone(), query_lengths, 1)?,
    };
    // The gradient contractions all run at the softmax data type, like the forward softmax.
    let convert = |operand: &V| -> Result<V, ProgramError> {
        if data_type == softmax_type { Ok(operand.clone()) } else { operand.convert_element_type(softmax_type) }
    };
    let softmax_query = convert(query)?;
    let softmax_key = convert(&expanded_key)?;
    let softmax_value = convert(&expanded_value)?;
    let softmax_output = convert(output)?;
    let softmax_output_cotangent = convert(&output_cotangent)?;
    // `dP[b, n, t, s] = Σ_h dO[b, t, n, h] · V[b, s, n, h]`: batch `[0, 2]/[0, 2]` (batch and heads), contract the
    // head axis `3/3`; the result is `[batch dims..., lhs result..., rhs result...] = [b, n, t, s]`.
    let weight_cotangents = softmax_output_cotangent
        .dot(&softmax_value, &DotDimensionNumbers::new(vec![3], vec![3], vec![0, 2], vec![0, 2]));
    // `delta[b, n, t] = Σ_h dO[b, t, n, h] · O[b, t, n, h]`, transposed from `[b, t, n]` to `[b, n, t]`.
    let delta = softmax_output_cotangent
        .mul(&softmax_output)?
        .reduce(&[3], ReductionKind::Sum)
        .transpose([0, 2, 1])?;
    // `dS = P ∘ (dP - delta)` with `delta` broadcast over the kv axis.
    let logit_cotangents =
        weights.mul(&weight_cotangents.sub(&delta.legacy_broadcast(logits_type.clone(), logit_axes)?)?)?;
    // The logits are `scale · (Q·Kᵀ) + bias`, so the query/key cotangents carry one extra `scale` factor while the
    // bias cotangent reads `dS` unscaled.
    let scaled_logit_cotangents = logit_cotangents.mul(&attention_fill(context, &logits_type, scale)?)?;
    // `dQ[b, t, n, h] = scale · Σ_s dS[b, n, t, s] · K[b, s, n, h]`: batch `[0, 1]/[0, 2]`, contract the kv-sequence
    // axis `3/1`; the result `[b, n, t, h]` transposes to the `BTNH` layout. Out-of-range query rows are forced to
    // exact zeros, mirroring the fused kernel's memzeroed gradient.
    let mut query_cotangent = scaled_logit_cotangents
        .dot(&softmax_key, &DotDimensionNumbers::new(vec![3], vec![1], vec![0, 1], vec![0, 2]))
        .transpose([0, 2, 1, 3])?;
    if let Some((query_lengths, _)) = sequence_lengths {
        query_cotangent = zero_out_of_range_query_rows(context, query_cotangent, query_lengths, 1)?;
    }
    // `dK[b, s, n, h] = scale · Σ_t dS[b, n, t, s] · Q[b, t, n, h]`: batch `[0, 1]/[0, 2]`, contract the
    // query-sequence axis `2/1`; the result `[b, n, s, h]` transposes to `[b, s, n, h]`.
    let key_cotangent = scaled_logit_cotangents
        .dot(&softmax_query, &DotDimensionNumbers::new(vec![2], vec![1], vec![0, 1], vec![0, 2]))
        .transpose([0, 2, 1, 3])?;
    // `dV[b, s, n, h] = Σ_t P[b, n, t, s] · dO[b, t, n, h]`: the same dimension numbers as `dK`.
    let value_cotangent = weights
        .dot(&softmax_output_cotangent, &DotDimensionNumbers::new(vec![2], vec![1], vec![0, 1], vec![0, 2]))
        .transpose([0, 2, 1, 3])?;
    // Grouped-query attention: each key/value head serves `group` consecutive query heads, so its cotangent sums
    // over the per-head group axis.
    let (key_cotangent, value_cotangent) = if dimensions.key_value_heads == dimensions.query_heads {
        (key_cotangent, value_cotangent)
    } else {
        let group = dimensions.query_heads / dimensions.key_value_heads;
        let grouped_shape = static_shape(&[
            dimensions.batch,
            dimensions.key_value_sequence,
            dimensions.key_value_heads,
            group,
            dimensions.head_dimension,
        ]);
        (
            key_cotangent.reshape(grouped_shape.clone())?.reduce(&[3], ReductionKind::Sum),
            value_cotangent.reshape(grouped_shape)?.reduce(&[3], ReductionKind::Sum),
        )
    };
    let convert_back = |cotangent: V| -> Result<V, ProgramError> {
        if data_type == softmax_type { Ok(cotangent) } else { cotangent.convert_element_type(data_type) }
    };
    let mut cotangents =
        vec![convert_back(query_cotangent)?, convert_back(key_cotangent)?, convert_back(value_cotangent)?];
    if let Some(bias) = bias {
        // The bias enters the logits unscaled, so its cotangent is `dS` summed over the bias's broadcast leading
        // dimensions and reshaped back to the bias shape.
        let bias_type = bias.r#type().into_owned();
        let bias_dimensions =
            static_attention_dimensions(DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME, "bias", &bias_type)?;
        let logit_dimensions = [dimensions.batch, dimensions.query_heads];
        let reduce_axes =
            (0..2).filter(|&axis| bias_dimensions[axis] == 1 && logit_dimensions[axis] != 1).collect::<Vec<_>>();
        let bias_cotangent = if reduce_axes.is_empty() {
            logit_cotangents
        } else {
            logit_cotangents.reduce(reduce_axes.as_slice(), ReductionKind::Sum)
        };
        let bias_cotangent = bias_cotangent.reshape(bias_type.shape().clone())?;
        cotangents.push(convert_back(bias_cotangent)?);
    }
    Ok(cotangents)
}

/// Query/key/value input tree of [`differentiable_dot_product_attention`].
pub type DotProductAttentionInputs<D> = (DomainTracer<D>, DomainTracer<D>, DomainTracer<D>);

/// Residual tree of [`differentiable_dot_product_attention`]: `(query, key, value, output, activation)`.
pub type DotProductAttentionResiduals<D> =
    (DomainTracer<D>, DomainTracer<D>, DomainTracer<D>, DomainTracer<D>, DomainTracer<D>);

/// Query/key/value/bias input tree of [`differentiable_dot_product_attention_with_bias`].
pub type DotProductAttentionInputsWithBias<D> = (DomainTracer<D>, DomainTracer<D>, DomainTracer<D>, DomainTracer<D>);

/// Residual tree of [`differentiable_dot_product_attention_with_bias`]:
/// `(query, key, value, bias, output, activation)`.
pub type DotProductAttentionResidualsWithBias<D> =
    (DomainTracer<D>, DomainTracer<D>, DomainTracer<D>, DomainTracer<D>, DomainTracer<D>, DomainTracer<D>);

/// Creates the differentiable (training-path) entry point for scaled dot-product attention over `(query, key,
/// value)` trees — the analogue of how JAX's `jax.nn.dot_product_attention(implementation="cudnn")` wires its fused
/// kernels through `jax.custom_vjp` in `jax/_src/cudnn/fused_attention_stablehlo.py`. The returned [`CustomVjp`]
/// function is called inside a trace as `function.call((query, key, value))`:
///
///   - its primal binds the plain (inference) [`DotProductAttentionOperation`];
///   - under reverse-mode differentiation, its forward binds the same operation with [an activation
///     output](DotProductAttentionOperation::with_activation_output) and stashes
///     `(query, key, value, output, activation)` as residuals; and
///   - its backward binds one [`DotProductAttentionBackwardOperation`] producing the query/key/value cotangents.
///
/// As with every `custom_vjp` function, forward-mode differentiation of a staged call is rejected; the plain
/// [`DotProductAttention`] capability remains the inference path. Use
/// [`differentiable_dot_product_attention_with_bias`] when a bias operand also needs a gradient, and
/// [`differentiable_dot_product_attention_with_sequence_lengths`] for variable-sequence-length (padded) training.
///
/// # Parameters
///
///   - `scale`: Multiplier applied to the attention scores before masking and softmax.
///   - `mask`: Built-in [`AttentionMask`] applied to the attention scores before the softmax.
///   - `sliding_window`: Optional sliding-window width tightening the causal mask; refer to
///     [`DotProductAttentionOperation::with_sliding_window`].
///   - `dropout`: Optional `(rate, seed)` dropout threaded identically through the forward and backward operations
///     (the fused kernels' forward/backward dropout states agree per matching call index — refer to
///     [`DotProductAttentionOperation::with_dropout`]).
pub fn differentiable_dot_product_attention<D>(
    scale: f64,
    mask: AttentionMask,
    sliding_window: Option<usize>,
    dropout: Option<(f64, u64)>,
) -> CustomVjp<
    impl Fn(DotProductAttentionInputs<D>) -> Result<DomainTracer<D>, ProgramError>,
    impl Fn(DotProductAttentionInputs<D>) -> Result<(DomainTracer<D>, DotProductAttentionResiduals<D>), ProgramError>,
    impl Fn(DotProductAttentionResiduals<D>, DomainTracer<D>) -> Result<DotProductAttentionInputs<D>, ProgramError>,
    DotProductAttentionInputs<D>,
    DomainTracer<D>,
    DotProductAttentionResiduals<D>,
>
where
    D: Domain<Type = ArrayType>,
    DomainTracer<D>: DotProductAttention + DotProductAttentionBackward,
{
    custom_vjp(
        move |(query, key, value): DotProductAttentionInputs<D>| {
            query.dot_product_attention_with_options(&key, &value, None, None, scale, mask, sliding_window, dropout)
        },
        move |(query, key, value): DotProductAttentionInputs<D>| {
            let (output, activation) = query.dot_product_attention_with_activation(
                &key,
                &value,
                None,
                None,
                scale,
                mask,
                sliding_window,
                dropout,
            )?;
            Ok((output.clone(), (query, key, value, output, activation)))
        },
        move |(query, key, value, output, activation): DotProductAttentionResiduals<D>, output_cotangent| {
            let (query_cotangent, key_cotangent, value_cotangent, _) = query
                .dot_product_attention_backward_with_options(
                    &key,
                    &value,
                    None,
                    None,
                    &output,
                    &activation,
                    &output_cotangent,
                    scale,
                    mask,
                    sliding_window,
                    dropout,
                )?;
            Ok((query_cotangent, key_cotangent, value_cotangent))
        },
    )
}

/// Creates the differentiable (training-path) entry point for scaled dot-product attention over `(query, key,
/// value, bias)` trees, producing bias cotangents alongside the query/key/value cotangents. Refer to the
/// documentation of [`differentiable_dot_product_attention`] for the wiring and the reverse-mode-only contract.
///
/// # Parameters
///
///   - `scale`: Multiplier applied to the attention scores before the bias, masking, and softmax.
///   - `mask`: Built-in [`AttentionMask`] applied to the attention scores before the softmax.
///   - `sliding_window`: Optional sliding-window width tightening the causal mask.
///   - `dropout`: Optional `(rate, seed)` dropout threaded identically through the forward and backward operations.
pub fn differentiable_dot_product_attention_with_bias<D>(
    scale: f64,
    mask: AttentionMask,
    sliding_window: Option<usize>,
    dropout: Option<(f64, u64)>,
) -> CustomVjp<
    impl Fn(DotProductAttentionInputsWithBias<D>) -> Result<DomainTracer<D>, ProgramError>,
    impl Fn(
        DotProductAttentionInputsWithBias<D>,
    ) -> Result<(DomainTracer<D>, DotProductAttentionResidualsWithBias<D>), ProgramError>,
    impl Fn(
        DotProductAttentionResidualsWithBias<D>,
        DomainTracer<D>,
    ) -> Result<DotProductAttentionInputsWithBias<D>, ProgramError>,
    DotProductAttentionInputsWithBias<D>,
    DomainTracer<D>,
    DotProductAttentionResidualsWithBias<D>,
>
where
    D: Domain<Type = ArrayType>,
    DomainTracer<D>: DotProductAttention + DotProductAttentionBackward,
{
    custom_vjp(
        move |(query, key, value, bias): DotProductAttentionInputsWithBias<D>| {
            query.dot_product_attention_with_options(
                &key,
                &value,
                Some(&bias),
                None,
                scale,
                mask,
                sliding_window,
                dropout,
            )
        },
        move |(query, key, value, bias): DotProductAttentionInputsWithBias<D>| {
            let (output, activation) = query.dot_product_attention_with_activation(
                &key,
                &value,
                Some(&bias),
                None,
                scale,
                mask,
                sliding_window,
                dropout,
            )?;
            Ok((output.clone(), (query, key, value, bias, output, activation)))
        },
        move |(query, key, value, bias, output, activation): DotProductAttentionResidualsWithBias<D>,
              output_cotangent| {
            let (query_cotangent, key_cotangent, value_cotangent, bias_cotangent) = query
                .dot_product_attention_backward_with_options(
                    &key,
                    &value,
                    Some(&bias),
                    None,
                    &output,
                    &activation,
                    &output_cotangent,
                    scale,
                    mask,
                    sliding_window,
                    dropout,
                )?;
            // The full backward form returns a bias cotangent whenever a bias operand is provided.
            Ok((query_cotangent, key_cotangent, value_cotangent, bias_cotangent.unwrap()))
        },
    )
}

/// Query/key/value/sequence-lengths input tree of [`differentiable_dot_product_attention_with_sequence_lengths`]:
/// `(query, key, value, query_sequence_lengths, key_value_sequence_lengths)`.
pub type DotProductAttentionInputsWithSequenceLengths<D> =
    (DomainTracer<D>, DomainTracer<D>, DomainTracer<D>, DomainTracer<D>, DomainTracer<D>);

/// Residual tree of [`differentiable_dot_product_attention_with_sequence_lengths`]:
/// `(query, key, value, query_sequence_lengths, key_value_sequence_lengths, output, activation)`.
pub type DotProductAttentionResidualsWithSequenceLengths<D> = (
    DomainTracer<D>,
    DomainTracer<D>,
    DomainTracer<D>,
    DomainTracer<D>,
    DomainTracer<D>,
    DomainTracer<D>,
    DomainTracer<D>,
);

/// Creates the differentiable (training-path) entry point for variable-sequence-length (padded) scaled dot-product
/// attention over `(query, key, value, query_sequence_lengths, key_value_sequence_lengths)` trees. Refer to the
/// documentation of [`differentiable_dot_product_attention`] for the wiring and the reverse-mode-only contract, and
/// to [`DotProductAttentionOperation`] for the padding semantics.
///
/// The `i32[batch]` sequence lengths are non-differentiated inputs: they ride into the residuals so the backward
/// operation can reapply the padding masks, and their input cotangents are structural zeros of the first-class
/// zero-space cotangent type that non-differentiable types carry (reverse mode accumulates no live adjoint for
/// them).
///
/// # Parameters
///
///   - `scale`: Multiplier applied to the attention scores before masking and softmax.
///   - `mask`: Built-in [`AttentionMask`] applied to the attention scores before the softmax.
///   - `sliding_window`: Optional sliding-window width tightening the causal mask.
///   - `dropout`: Optional `(rate, seed)` dropout threaded identically through the forward and backward operations.
pub fn differentiable_dot_product_attention_with_sequence_lengths<D>(
    scale: f64,
    mask: AttentionMask,
    sliding_window: Option<usize>,
    dropout: Option<(f64, u64)>,
) -> CustomVjp<
    impl Fn(DotProductAttentionInputsWithSequenceLengths<D>) -> Result<DomainTracer<D>, ProgramError>,
    impl Fn(
        DotProductAttentionInputsWithSequenceLengths<D>,
    ) -> Result<(DomainTracer<D>, DotProductAttentionResidualsWithSequenceLengths<D>), ProgramError>,
    impl Fn(
        DotProductAttentionResidualsWithSequenceLengths<D>,
        DomainTracer<D>,
    ) -> Result<DotProductAttentionInputsWithSequenceLengths<D>, ProgramError>,
    DotProductAttentionInputsWithSequenceLengths<D>,
    DomainTracer<D>,
    DotProductAttentionResidualsWithSequenceLengths<D>,
>
where
    D: Domain<Type = ArrayType>,
    D::Operation: From<ZeroOperation<ArrayType>>,
    DomainTracer<D>: DotProductAttention + DotProductAttentionBackward,
{
    custom_vjp(
        move |(query, key, value, query_lengths, key_value_lengths): DotProductAttentionInputsWithSequenceLengths<
            D,
        >| {
            query.dot_product_attention_with_options(
                &key,
                &value,
                None,
                Some((&query_lengths, &key_value_lengths)),
                scale,
                mask,
                sliding_window,
                dropout,
            )
        },
        move |(query, key, value, query_lengths, key_value_lengths): DotProductAttentionInputsWithSequenceLengths<
            D,
        >| {
            let (output, activation) = query.dot_product_attention_with_activation(
                &key,
                &value,
                None,
                Some((&query_lengths, &key_value_lengths)),
                scale,
                mask,
                sliding_window,
                dropout,
            )?;
            Ok((output.clone(), (query, key, value, query_lengths, key_value_lengths, output, activation)))
        },
        move |(query, key, value, query_lengths, key_value_lengths, output, activation), output_cotangent| {
            let (query_cotangent, key_cotangent, value_cotangent, _) = query
                .dot_product_attention_backward_with_options(
                    &key,
                    &value,
                    None,
                    Some((&query_lengths, &key_value_lengths)),
                    &output,
                    &activation,
                    &output_cotangent,
                    scale,
                    mask,
                    sliding_window,
                    dropout,
                )?;
            // The sequence lengths are non-differentiated `i32` inputs, so their cotangents are structural zeros of
            // the first-class zero-space cotangent type that non-differentiable types carry.
            let zero_cotangent = |lengths: &DomainTracer<D>| lengths.context().zero(&lengths.r#type().cotangent());
            let query_lengths_cotangent = zero_cotangent(&query_lengths)?;
            let key_value_lengths_cotangent = zero_cotangent(&key_value_lengths)?;
            Ok((query_cotangent, key_cotangent, value_cotangent, query_lengths_cotangent, key_value_lengths_cotangent))
        },
    )
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::batching::ProgramBatchingOutputAxesPolicy;
    use crate::contexts::EagerContext;
    use crate::differentiation::value_and_gradient;
    use crate::macros::{check_operation_transposition, check_operation_type_inference};
    use crate::parameters::Placeholder;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::sharding::ShardingDimension;
    use crate::types::{DimensionBounds, DimensionVariable};

    use super::*;

    /// Returns the static [`ArrayType`] with the provided data type and dimensions used throughout these tests.
    fn attention_type(data_type: DataType, dimensions: &[usize]) -> ArrayType {
        ArrayType::new(data_type, Shape::new(dimensions.iter().map(|&size| Dimension::Static(size)).collect()))
    }

    /// Plain-Rust host reference for `BTNH` scaled dot-product attention with `f64` accumulation, covering grouped
    /// key/value heads (each key/value head repeated `heads / kv_heads` times explicitly through the head-index
    /// mapping), an optional broadcastable bias added to the scaled scores, top-left causal masking, a sliding
    /// window under which row `i` attends keys `[max(0, i + 1 - window), i]`, and optional per-batch-item
    /// `(query, key/value)` sequence lengths under which key columns `j >= kv_lengths[b]` are fully excluded and
    /// query rows `i >= q_lengths[b]` are exact zeros in both outputs. Returns the attended output and the
    /// natural-log log-sum-exp statistic of the masked logits (`[batch, heads, q_seq]`, row-major).
    #[allow(clippy::too_many_arguments)]
    fn host_attention(
        query: &[f64],
        key: &[f64],
        value: &[f64],
        bias: Option<(&[f64], [usize; 4])>,
        [batch, q_seq, heads, head_dim]: [usize; 4],
        kv_seq: usize,
        kv_heads: usize,
        scale: f64,
        causal: bool,
        window: Option<usize>,
        sequence_lengths: Option<(&[usize], &[usize])>,
    ) -> (Vec<f64>, Vec<f64>) {
        let group = heads / kv_heads;
        let query_at = |b: usize, s: usize, n: usize, d: usize| query[((b * q_seq + s) * heads + n) * head_dim + d];
        let key_at =
            |b: usize, s: usize, n: usize, d: usize| key[((b * kv_seq + s) * kv_heads + n / group) * head_dim + d];
        let value_at =
            |b: usize, s: usize, n: usize, d: usize| value[((b * kv_seq + s) * kv_heads + n / group) * head_dim + d];
        let bias_at = |b: usize, n: usize, i: usize, j: usize| match bias {
            None => 0.0,
            Some((values, [bias_batch, bias_heads, _, _])) => {
                let b = if bias_batch == 1 { 0 } else { b };
                let n = if bias_heads == 1 { 0 } else { n };
                values[((b * bias_heads + n) * q_seq + i) * kv_seq + j]
            }
        };
        let mut output = vec![0.0; batch * q_seq * heads * head_dim];
        let mut statistic = vec![0.0; batch * heads * q_seq];
        for b in 0..batch {
            for n in 0..heads {
                for i in 0..q_seq {
                    // Out-of-range query rows stay exact zeros in both outputs.
                    if sequence_lengths.is_some_and(|(query_lengths, _)| i >= query_lengths[b]) {
                        continue;
                    }
                    let mut scores = vec![0.0; kv_seq];
                    for (j, score) in scores.iter_mut().enumerate() {
                        let mut product = 0.0;
                        for d in 0..head_dim {
                            product += query_at(b, i, n, d) * key_at(b, j, n, d);
                        }
                        let masked = (causal && (j > i || window.is_some_and(|window| j + window <= i)))
                            || sequence_lengths.is_some_and(|(_, key_value_lengths)| j >= key_value_lengths[b]);
                        *score = if masked { f64::NEG_INFINITY } else { product * scale + bias_at(b, n, i, j) };
                    }
                    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let exponentials: Vec<f64> = scores.iter().map(|score| (score - maximum).exp()).collect();
                    let sum: f64 = exponentials.iter().sum();
                    statistic[(b * heads + n) * q_seq + i] = maximum + sum.ln();
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
        (output, statistic)
    }

    /// Returns a deterministic pseudo-random value sequence for test operands.
    fn test_values(count: usize, multiplier: usize, modulus: usize, shift: f64, step: f64) -> Vec<f64> {
        (0..count).map(|i| ((i * multiplier % modulus) as f64 - shift) * step).collect()
    }

    /// Computes the central finite difference of `loss` with respect to every entry of `values`.
    fn central_difference(loss: impl Fn(&[f64]) -> f64, values: &[f64]) -> Vec<f64> {
        let epsilon = 1e-5;
        (0..values.len())
            .map(|index| {
                let mut plus = values.to_vec();
                plus[index] += epsilon;
                let mut minus = values.to_vec();
                minus[index] -= epsilon;
                (loss(plus.as_slice()) - loss(minus.as_slice())) / (2.0 * epsilon)
            })
            .collect()
    }

    #[test]
    fn test_dot_product_attention() {
        // Small-shape correctness on the reference array backend against a plain-Rust host reference computed with
        // `f64` accumulation: `b = 1`, `s = 4`, `n = 2`, `h = 3` in `f32`.
        let dimensions = [1, 4, 2, 3];
        let operand_type = attention_type(DataType::F32, &dimensions);
        let query_values = test_values(24, 7, 11, 5.0, 0.25);
        let key_values = test_values(24, 5, 13, 6.0, 0.25);
        let value_values = test_values(24, 3, 7, 3.0, 0.5);
        let query = Array::from_f64s(operand_type.clone(), query_values.clone());
        let key = Array::from_f64s(operand_type.clone(), key_values.clone());
        let value = Array::from_f64s(operand_type.clone(), value_values.clone());
        let scale = 0.5;

        let unmasked = query.dot_product_attention(&key, &value, scale, AttentionMask::None, None).unwrap();
        assert_eq!(unmasked.r#type().as_ref(), &operand_type);
        let (expected_unmasked, _) =
            host_attention(&query_values, &key_values, &value_values, None, dimensions, 4, 2, scale, false, None, None);
        for (actual, expected) in unmasked.to_f64s().into_iter().zip(expected_unmasked.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-5);
        }

        // The causal mask excludes later key/value positions for early query rows, so the masked output both
        // matches the causal host reference and differs from the unmasked output.
        let causal = query.dot_product_attention(&key, &value, scale, AttentionMask::Causal, None).unwrap();
        let (expected_causal, _) =
            host_attention(&query_values, &key_values, &value_values, None, dimensions, 4, 2, scale, true, None, None);
        for (actual, expected) in causal.to_f64s().into_iter().zip(expected_causal.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-5);
        }
        assert_ne!(causal.to_f64s(), unmasked.to_f64s());

        // A narrow-precision operand type runs its softmax at `f32` and converts back, so a small `bf16` case stays
        // within the `bf16` grid's tolerance of the host reference.
        let bf16_dimensions = [1, 2, 1, 2];
        let bf16_type = attention_type(DataType::BF16, &bf16_dimensions);
        let bf16_query_values = vec![0.5, -0.25, 1.0, 0.75];
        let bf16_key_values = vec![0.25, 0.5, -0.5, 1.0];
        let bf16_value_values = vec![1.0, 2.0, -1.0, 0.5];
        let bf16_query = Array::from_f64s(bf16_type.clone(), bf16_query_values.clone());
        let bf16_key = Array::from_f64s(bf16_type.clone(), bf16_key_values.clone());
        let bf16_value = Array::from_f64s(bf16_type.clone(), bf16_value_values.clone());
        let bf16_output =
            bf16_query.dot_product_attention(&bf16_key, &bf16_value, 1.0, AttentionMask::None, None).unwrap();
        assert_eq!(bf16_output.r#type().as_ref(), &bf16_type);
        let (expected_bf16, _) = host_attention(
            &bf16_query_values,
            &bf16_key_values,
            &bf16_value_values,
            None,
            bf16_dimensions,
            2,
            1,
            1.0,
            false,
            None,
            None,
        );
        for (actual, expected) in bf16_output.to_f64s().into_iter().zip(expected_bf16.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 2e-2);
        }

        // The staged operation renders its payload, including the optional fields only when they are set.
        let operation = DotProductAttentionOperation::new(scale, AttentionMask::Causal);
        assert_eq!(operation.scale(), scale);
        assert_eq!(operation.mask(), AttentionMask::Causal);
        assert_eq!(operation.sliding_window(), None);
        assert!(!operation.activation_output());
        assert_eq!(operation.name(), DOT_PRODUCT_ATTENTION_OPERATION_NAME);
        assert_eq!(operation.to_string(), "dot_product_attention [scale=0.5, mask=causal]");
        assert_eq!(
            DotProductAttentionOperation::new(0.125, AttentionMask::None).to_string(),
            "dot_product_attention [scale=0.125, mask=none]",
        );
        let windowed = operation.with_sliding_window(3);
        assert_eq!(windowed.sliding_window(), Some(3));
        assert_eq!(windowed.to_string(), "dot_product_attention [scale=0.5, mask=causal, sliding_window=3]");
        let training = windowed.with_activation_output();
        assert!(training.activation_output());
        assert_eq!(
            training.to_string(),
            "dot_product_attention [scale=0.5, mask=causal, sliding_window=3, activation=true]",
        );
        let backward = DotProductAttentionBackwardOperation::new(scale, AttentionMask::Causal);
        assert_eq!(backward.scale(), scale);
        assert_eq!(backward.mask(), AttentionMask::Causal);
        assert_eq!(backward.sliding_window(), None);
        assert_eq!(backward.name(), DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME);
        assert_eq!(backward.to_string(), "dot_product_attention_backward [scale=0.5, mask=causal]");
        assert_eq!(
            backward.with_sliding_window(2).to_string(),
            "dot_product_attention_backward [scale=0.5, mask=causal, sliding_window=2]",
        );
    }

    #[test]
    fn test_dot_product_attention_grouped_query() {
        // Grouped-query attention with `n = 4` query heads over `kv = 2` key/value heads: the composition repeats
        // each key/value head twice (query head `i` attends key/value head `i / 2`), so the output matches a host
        // reference that repeats the key/value heads explicitly.
        let query_dimensions = [1, 3, 4, 2];
        let query_type = attention_type(DataType::F32, &query_dimensions);
        let key_value_type = attention_type(DataType::F32, &[1, 4, 2, 2]);
        let query_values = test_values(24, 7, 11, 5.0, 0.25);
        let key_values = test_values(16, 5, 13, 6.0, 0.25);
        let value_values = test_values(16, 3, 7, 3.0, 0.5);
        let query = Array::from_f64s(query_type, query_values.clone());
        let key = Array::from_f64s(key_value_type.clone(), key_values.clone());
        let value = Array::from_f64s(key_value_type, value_values.clone());
        let scale = 0.5;

        for causal in [false, true] {
            let mask = if causal { AttentionMask::Causal } else { AttentionMask::None };
            let output = query.dot_product_attention(&key, &value, scale, mask, None).unwrap();
            let (expected, _) = host_attention(
                &query_values,
                &key_values,
                &value_values,
                None,
                query_dimensions,
                4,
                2,
                scale,
                causal,
                None,
                None,
            );
            for (actual, expected) in output.to_f64s().into_iter().zip(expected.iter()) {
                assert_abs_diff_eq!(actual, expected, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn test_dot_product_attention_bias() {
        // The bias is added to the already scaled scores before masking and softmax. A full-shape bias and a
        // broadcast `[1, 1, t, s]` bias both match the host reference, and a combined causal + grouped-query + bias
        // case exercises every feature at once.
        let dimensions = [2, 3, 2, 2];
        let operand_type = attention_type(DataType::F32, &dimensions);
        let key_value_type = attention_type(DataType::F32, &[2, 4, 2, 2]);
        let query_values = test_values(24, 7, 11, 5.0, 0.25);
        let key_values = test_values(32, 5, 13, 6.0, 0.25);
        let value_values = test_values(32, 3, 7, 3.0, 0.5);
        let query = Array::from_f64s(operand_type, query_values.clone());
        let key = Array::from_f64s(key_value_type.clone(), key_values.clone());
        let value = Array::from_f64s(key_value_type, value_values.clone());
        let scale = 0.5;

        for bias_dimensions in [[2, 2, 3, 4], [1, 1, 3, 4]] {
            let bias_count = bias_dimensions.iter().product();
            let bias_values = test_values(bias_count, 11, 17, 8.0, 0.125);
            let bias = Array::from_f64s(attention_type(DataType::F32, &bias_dimensions), bias_values.clone());
            let output = query
                .dot_product_attention_with_bias(&key, &value, &bias, scale, AttentionMask::None, None)
                .unwrap();
            let (expected, _) = host_attention(
                &query_values,
                &key_values,
                &value_values,
                Some((&bias_values, bias_dimensions)),
                dimensions,
                4,
                2,
                scale,
                false,
                None,
                None,
            );
            for (actual, expected) in output.to_f64s().into_iter().zip(expected.iter()) {
                assert_abs_diff_eq!(actual, expected, epsilon = 1e-5);
            }
        }

        // Causal + grouped-query + bias combined.
        let query_dimensions = [1, 3, 4, 2];
        let query =
            Array::from_f64s(attention_type(DataType::F32, &query_dimensions), test_values(24, 7, 11, 5.0, 0.25));
        let key_value_type = attention_type(DataType::F32, &[1, 4, 2, 2]);
        let key = Array::from_f64s(key_value_type.clone(), test_values(16, 5, 13, 6.0, 0.25));
        let value = Array::from_f64s(key_value_type, test_values(16, 3, 7, 3.0, 0.5));
        let bias_dimensions = [1, 4, 3, 4];
        let bias_values = test_values(48, 11, 17, 8.0, 0.125);
        let bias = Array::from_f64s(attention_type(DataType::F32, &bias_dimensions), bias_values.clone());
        let output = query
            .dot_product_attention_with_bias(&key, &value, &bias, scale, AttentionMask::Causal, None)
            .unwrap();
        let (expected, _) = host_attention(
            &query.to_f64s(),
            &key.to_f64s(),
            &value.to_f64s(),
            Some((&bias_values, bias_dimensions)),
            query_dimensions,
            4,
            2,
            scale,
            true,
            None,
            None,
        );
        for (actual, expected) in output.to_f64s().into_iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_dot_product_attention_sliding_window() {
        // The sliding-window semantics: with `window = w`, query row `r` attends key/value positions
        // `[max(0, r + 1 - w), r]` (the causal upper bound plus a window lower bound). With `w = 1` each row
        // attends exactly its own key position, so the softmax collapses to a single weight of one and the output
        // equals the value operand row for row.
        let dimensions = [1, 4, 1, 2];
        let operand_type = attention_type(DataType::F64, &dimensions);
        let query_values = test_values(8, 7, 11, 5.0, 0.25);
        let key_values = test_values(8, 5, 13, 6.0, 0.25);
        let value_values = test_values(8, 3, 7, 3.0, 0.5);
        let query = Array::from_f64s(operand_type.clone(), query_values.clone());
        let key = Array::from_f64s(operand_type.clone(), key_values.clone());
        let value = Array::from_f64s(operand_type, value_values.clone());
        let scale = 0.5;

        let window_one = query.dot_product_attention(&key, &value, scale, AttentionMask::Causal, Some(1)).unwrap();
        for (actual, expected) in window_one.to_f64s().into_iter().zip(value_values.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-9);
        }

        // A wider window matches the host reference and differs from the plain causal output (which it approaches
        // as the window covers the whole sequence).
        let window_two = query.dot_product_attention(&key, &value, scale, AttentionMask::Causal, Some(2)).unwrap();
        let (expected_window_two, _) = host_attention(
            &query_values,
            &key_values,
            &value_values,
            None,
            dimensions,
            4,
            1,
            scale,
            true,
            Some(2),
            None,
        );
        for (actual, expected) in window_two.to_f64s().into_iter().zip(expected_window_two.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-9);
        }
        let causal = query.dot_product_attention(&key, &value, scale, AttentionMask::Causal, None).unwrap();
        assert_ne!(window_two.to_f64s(), causal.to_f64s());
        let window_four = query.dot_product_attention(&key, &value, scale, AttentionMask::Causal, Some(4)).unwrap();
        assert_eq!(window_four.to_f64s(), causal.to_f64s());
    }

    #[test]
    fn test_dot_product_attention_activation_output() {
        // The activation output is the natural-log log-sum-exp statistic of the post-scale, post-bias, post-mask
        // logits over the key/value axis, always at `f32` and shaped `[batch, heads, q_seq]`. The attended output
        // is unchanged by requesting the statistic.
        let dimensions = [1, 3, 2, 2];
        let operand_type = attention_type(DataType::F32, &dimensions);
        let key_value_type = attention_type(DataType::F32, &[1, 4, 2, 2]);
        let query_values = test_values(12, 7, 11, 5.0, 0.25);
        let key_values = test_values(16, 5, 13, 6.0, 0.25);
        let value_values = test_values(16, 3, 7, 3.0, 0.5);
        let query = Array::from_f64s(operand_type, query_values.clone());
        let key = Array::from_f64s(key_value_type.clone(), key_values.clone());
        let value = Array::from_f64s(key_value_type, value_values.clone());
        let bias_dimensions = [1, 1, 3, 4];
        let bias_values = test_values(12, 11, 17, 8.0, 0.125);
        let bias = Array::from_f64s(attention_type(DataType::F32, &bias_dimensions), bias_values.clone());
        let scale = 0.5;

        let (output, activation) = query
            .dot_product_attention_with_activation(
                &key,
                &value,
                Some(&bias),
                None,
                scale,
                AttentionMask::Causal,
                None,
                None,
            )
            .unwrap();
        assert_eq!(activation.r#type().as_ref(), &attention_type(DataType::F32, &[1, 2, 3]));
        let plain = query
            .dot_product_attention_with_bias(&key, &value, &bias, scale, AttentionMask::Causal, None)
            .unwrap();
        assert_eq!(output.to_f64s(), plain.to_f64s());
        let (expected_output, expected_statistic) = host_attention(
            &query_values,
            &key_values,
            &value_values,
            Some((&bias_values, bias_dimensions)),
            dimensions,
            4,
            2,
            scale,
            true,
            None,
            None,
        );
        for (actual, expected) in output.to_f64s().into_iter().zip(expected_output.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-5);
        }
        for (actual, expected) in activation.to_f64s().into_iter().zip(expected_statistic.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-5);
        }

        // `f64` operands keep an `f64` softmax but still produce the statistic at `f32`.
        let f64_type = attention_type(DataType::F64, &[1, 2, 1, 2]);
        let f64_query = Array::from_f64s(f64_type.clone(), vec![0.5, -0.25, 1.0, 0.75]);
        let f64_key = Array::from_f64s(f64_type.clone(), vec![0.25, 0.5, -0.5, 1.0]);
        let f64_value = Array::from_f64s(f64_type, vec![1.0, 2.0, -1.0, 0.5]);
        let (_, f64_activation) = f64_query
            .dot_product_attention_with_activation(
                &f64_key,
                &f64_value,
                None,
                None,
                1.0,
                AttentionMask::None,
                None,
                None,
            )
            .unwrap();
        assert_eq!(f64_activation.r#type().as_ref(), &attention_type(DataType::F32, &[1, 1, 2]));
    }

    #[test]
    fn test_dot_product_attention_type_inference() {
        let operation = DotProductAttentionOperation::new(0.5, AttentionMask::None);
        let query = attention_type(DataType::F32, &[2, 4, 2, 3]);
        let key_value = attention_type(DataType::F32, &[2, 5, 2, 3]);
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
                        attention_type(DataType::BF16, &[2, 4, 2, 3]),
                        attention_type(DataType::BF16, &[2, 5, 2, 3]),
                        attention_type(DataType::BF16, &[2, 5, 2, 3]),
                    ],
                    output_types = [attention_type(DataType::BF16, &[2, 4, 2, 3])],
                },
                {
                    // Grouped-query attention: one key/value head serving both query heads.
                    input_types = [
                        query.clone(),
                        attention_type(DataType::F32, &[2, 5, 1, 3]),
                        attention_type(DataType::F32, &[2, 5, 1, 3]),
                    ],
                    output_types = [query.clone()],
                },
                {
                    // A full-shape bias and a broadcast bias are both accepted.
                    input_types = [
                        query.clone(),
                        key_value.clone(),
                        key_value.clone(),
                        attention_type(DataType::F32, &[2, 2, 4, 5]),
                    ],
                    output_types = [query.clone()],
                },
                {
                    input_types = [
                        query.clone(),
                        key_value.clone(),
                        key_value.clone(),
                        attention_type(DataType::F32, &[1, 1, 4, 5]),
                    ],
                    output_types = [query.clone()],
                },
                {
                    input_types = [query.clone(), key_value.clone()],
                    error = "'dot_product_attention' expects 3 (query, key, value), 4 (query, key, value, bias), \
                             5 (query, key, value, query sequence lengths, key/value sequence lengths), or 6 (query, \
                             key, value, bias, query sequence lengths, key/value sequence lengths) inputs but got 2",
                },
                {
                    input_types = [
                        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(4)])),
                        key_value.clone(),
                        key_value.clone(),
                    ],
                    error = "'dot_product_attention' query must have rank 4 but got rank 2",
                },
                {
                    input_types = [
                        query.clone(),
                        attention_type(DataType::F32, &[2, 5, 2, 3]),
                        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)])),
                    ],
                    error = "'dot_product_attention' value must have rank 4 but got rank 1",
                },
                {
                    input_types = [
                        attention_type(DataType::I32, &[2, 4, 2, 3]),
                        attention_type(DataType::I32, &[2, 5, 2, 3]),
                        attention_type(DataType::I32, &[2, 5, 2, 3]),
                    ],
                    error = "'dot_product_attention' requires floating-point operands but got data type i32",
                },
                {
                    input_types = [
                        attention_type(DataType::C64, &[2, 4, 2, 3]),
                        attention_type(DataType::C64, &[2, 5, 2, 3]),
                        attention_type(DataType::C64, &[2, 5, 2, 3]),
                    ],
                    error = "'dot_product_attention' requires floating-point operands but got data type c64",
                },
                {
                    input_types = [
                        query.clone(),
                        attention_type(DataType::F16, &[2, 5, 2, 3]),
                        attention_type(DataType::F16, &[2, 5, 2, 3]),
                    ],
                    error = "'dot_product_attention' key data type f16 does not match the query data type f32",
                },
                {
                    input_types = [query.clone(), attention_type(DataType::F32, &[3, 5, 2, 3]), key_value.clone()],
                    error = "'dot_product_attention' key batch dimension (3) does not match the query batch \
                             dimension (2)",
                },
                {
                    // The key/value heads must agree with each other before the grouped-query divisibility check.
                    input_types = [
                        query.clone(),
                        attention_type(DataType::F32, &[2, 5, 1, 3]),
                        attention_type(DataType::F32, &[2, 5, 2, 3]),
                    ],
                    error = "'dot_product_attention' value heads dimension (2) does not match the key heads \
                             dimension (1)",
                },
                {
                    // Three key/value heads cannot serve two query heads.
                    input_types = [
                        query.clone(),
                        attention_type(DataType::F32, &[2, 5, 3, 3]),
                        attention_type(DataType::F32, &[2, 5, 3, 3]),
                    ],
                    error = "'dot_product_attention' key/value heads dimension (3) must divide the query heads \
                             dimension (2)",
                },
                {
                    input_types = [query.clone(), key_value.clone(), attention_type(DataType::F32, &[2, 5, 2, 4])],
                    error = "'dot_product_attention' value head dimension (4) does not match the query head \
                             dimension (3)",
                },
                {
                    input_types = [query.clone(), key_value.clone(), attention_type(DataType::F32, &[2, 6, 2, 3])],
                    error = "'dot_product_attention' value sequence dimension (6) does not match the key sequence \
                             dimension (5)",
                },
                {
                    input_types = [
                        query.clone(),
                        key_value.clone(),
                        key_value.clone(),
                        attention_type(DataType::F32, &[2, 2, 4]),
                    ],
                    error = "'dot_product_attention' bias must have rank 4 but got rank 3",
                },
                {
                    input_types = [
                        query.clone(),
                        key_value.clone(),
                        key_value.clone(),
                        attention_type(DataType::BF16, &[2, 2, 4, 5]),
                    ],
                    error = "'dot_product_attention' bias data type bf16 does not match the query data type f32",
                },
                {
                    input_types = [
                        query.clone(),
                        key_value.clone(),
                        key_value.clone(),
                        attention_type(DataType::F32, &[3, 2, 4, 5]),
                    ],
                    error = "'dot_product_attention' bias batch dimension (3) must be 1 or match the query batch \
                             dimension (2)",
                },
                {
                    input_types = [
                        query.clone(),
                        key_value.clone(),
                        key_value.clone(),
                        attention_type(DataType::F32, &[2, 3, 4, 5]),
                    ],
                    error = "'dot_product_attention' bias heads dimension (3) must be 1 or match the query heads \
                             dimension (2)",
                },
                {
                    input_types = [
                        query.clone(),
                        key_value.clone(),
                        key_value.clone(),
                        attention_type(DataType::F32, &[2, 2, 5, 5]),
                    ],
                    error = "'dot_product_attention' bias query-sequence dimension (5) does not match the query \
                             sequence dimension (4)",
                },
                {
                    input_types = [
                        query.clone(),
                        key_value.clone(),
                        key_value.clone(),
                        attention_type(DataType::F32, &[2, 2, 4, 4]),
                    ],
                    error = "'dot_product_attention' bias key/value-sequence dimension (4) does not match the key \
                             sequence dimension (5)",
                },
                {
                    input_types = [
                        ArrayType::new(
                            DataType::F32,
                            Shape::new(vec![
                                Dimension::Static(2),
                                Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
                                Dimension::Static(2),
                                Dimension::Static(3),
                            ]),
                        ),
                        key_value.clone(),
                        key_value.clone(),
                    ],
                    error = "'dot_product_attention' query must have a static shape",
                },
            ],
        );

        // The activation output adds the `f32[batch, heads, q_seq]` statistic as a second output.
        check_operation_type_inference!(
            operation = operation.with_activation_output(),
            cases = [
                {
                    type = ArrayType,
                    input_types = [query.clone(), key_value.clone(), key_value.clone()],
                    output_types = [query.clone(), attention_type(DataType::F32, &[2, 2, 4])],
                },
            ],
        );

        // Sequence lengths ride as a trailing `i32[batch]` pair, with or without a bias, and each operand is
        // validated with its own exact error.
        let lengths = attention_type(DataType::I32, &[2]);
        check_operation_type_inference!(
            operation = operation,
            cases = [
                {
                    type = ArrayType,
                    input_types = [
                        query.clone(),
                        key_value.clone(),
                        key_value.clone(),
                        lengths.clone(),
                        lengths.clone(),
                    ],
                    output_types = [query.clone()],
                },
                {
                    input_types = [
                        query.clone(),
                        key_value.clone(),
                        key_value.clone(),
                        attention_type(DataType::F32, &[2, 2, 4, 5]),
                        lengths.clone(),
                        lengths.clone(),
                    ],
                    output_types = [query.clone()],
                },
                {
                    input_types = [
                        query.clone(),
                        key_value.clone(),
                        key_value.clone(),
                        attention_type(DataType::F32, &[2]),
                        lengths.clone(),
                    ],
                    error = "'dot_product_attention' query sequence lengths must have data type i32 but got f32",
                },
                {
                    input_types = [
                        query.clone(),
                        key_value.clone(),
                        key_value.clone(),
                        lengths.clone(),
                        attention_type(DataType::I32, &[3]),
                    ],
                    error = "'dot_product_attention' key/value sequence lengths size (3) does not match the batch \
                             dimension (2)",
                },
                {
                    input_types = [
                        query.clone(),
                        key_value.clone(),
                        key_value.clone(),
                        attention_type(DataType::I32, &[2, 1]),
                        lengths.clone(),
                    ],
                    error = "'dot_product_attention' query sequence lengths must have rank 1 but got rank 2",
                },
            ],
        );

        // A sliding window requires the causal mask and a positive width.
        let causal = DotProductAttentionOperation::new(0.5, AttentionMask::Causal);
        check_operation_type_inference!(
            operation = causal.with_sliding_window(2),
            cases = [
                {
                    type = ArrayType,
                    input_types = [query.clone(), key_value.clone(), key_value.clone()],
                    output_types = [query.clone()],
                },
            ],
        );
        check_operation_type_inference!(
            operation = causal.with_sliding_window(0),
            cases = [
                {
                    type = ArrayType,
                    input_types = [query.clone(), key_value.clone(), key_value.clone()],
                    error = "'dot_product_attention' sliding window must be positive",
                },
            ],
        );
        check_operation_type_inference!(
            operation = operation.with_sliding_window(2),
            cases = [
                {
                    type = ArrayType,
                    input_types = [query.clone(), key_value.clone(), key_value.clone()],
                    error = "'dot_product_attention' sliding window requires the causal mask",
                },
            ],
        );
    }

    #[test]
    fn test_dot_product_attention_backward_type_inference() {
        let operation = DotProductAttentionBackwardOperation::new(0.5, AttentionMask::Causal);
        let query = attention_type(DataType::F32, &[2, 4, 2, 3]);
        let key_value = attention_type(DataType::F32, &[2, 5, 1, 3]);
        let bias = attention_type(DataType::F32, &[1, 2, 4, 5]);
        let activation = attention_type(DataType::F32, &[2, 2, 4]);
        check_operation_type_inference!(
            operation = operation,
            cases = [
                {
                    type = ArrayType,
                    input_types = [
                        query.clone(),
                        key_value.clone(),
                        key_value.clone(),
                        query.clone(),
                        activation.clone(),
                        query.clone(),
                    ],
                    output_types = [query.clone(), key_value.clone(), key_value.clone()],
                },
                {
                    // With a bias operand, the bias cotangent is produced as a fourth output.
                    input_types = [
                        query.clone(),
                        key_value.clone(),
                        key_value.clone(),
                        bias.clone(),
                        query.clone(),
                        activation.clone(),
                        query.clone(),
                    ],
                    output_types = [query.clone(), key_value.clone(), key_value.clone(), bias.clone()],
                },
                {
                    input_types = [
                        query.clone(),
                        key_value.clone(),
                        key_value.clone(),
                        query.clone(),
                        activation.clone(),
                    ],
                    error = "'dot_product_attention_backward' expects 6 (query, key, value, output, activation, \
                             output cotangent), 7 (adding a bias after the value), 8 (adding trailing query and \
                             key/value sequence lengths), or 9 (adding both) inputs but got 5",
                },
                {
                    input_types = [
                        query.clone(),
                        key_value.clone(),
                        key_value.clone(),
                        attention_type(DataType::F32, &[2, 5, 2, 3]),
                        activation.clone(),
                        query.clone(),
                    ],
                    error = "'dot_product_attention_backward' output type f32[2, 5, 2, 3] does not match the \
                             expected forward output type f32[2, 4, 2, 3]",
                },
                {
                    input_types = [
                        query.clone(),
                        key_value.clone(),
                        key_value.clone(),
                        query.clone(),
                        attention_type(DataType::F32, &[2, 2, 5]),
                        query.clone(),
                    ],
                    error = "'dot_product_attention_backward' activation type f32[2, 2, 5] does not match the \
                             expected activation type f32[2, 2, 4]",
                },
                {
                    input_types = [
                        query.clone(),
                        key_value.clone(),
                        key_value.clone(),
                        query.clone(),
                        activation.clone(),
                        attention_type(DataType::F32, &[2, 4, 2, 4]),
                    ],
                    error = "'dot_product_attention_backward' output cotangent type f32[2, 4, 2, 4] does not match \
                             the expected forward output type f32[2, 4, 2, 3]",
                },
                {
                    // The trailing sequence-length pair is accepted with and without a bias.
                    input_types = [
                        query.clone(),
                        key_value.clone(),
                        key_value.clone(),
                        query.clone(),
                        activation.clone(),
                        query.clone(),
                        attention_type(DataType::I32, &[2]),
                        attention_type(DataType::I32, &[2]),
                    ],
                    output_types = [query.clone(), key_value.clone(), key_value.clone()],
                },
                {
                    input_types = [
                        query.clone(),
                        key_value.clone(),
                        key_value.clone(),
                        bias.clone(),
                        query.clone(),
                        activation.clone(),
                        query.clone(),
                        attention_type(DataType::I32, &[2]),
                        attention_type(DataType::I32, &[2]),
                    ],
                    output_types = [query.clone(), key_value.clone(), key_value.clone(), bias.clone()],
                },
                {
                    input_types = [
                        query.clone(),
                        key_value.clone(),
                        key_value.clone(),
                        query.clone(),
                        activation.clone(),
                        query.clone(),
                        attention_type(DataType::I32, &[2]),
                        attention_type(DataType::I64, &[2]),
                    ],
                    error = "'dot_product_attention_backward' key/value sequence lengths must have data type i32 \
                             but got i64",
                },
            ],
        );
    }

    #[test]
    fn test_dot_product_attention_batching() {
        // Two batch items with per-item shape `[1, 2, 1, 2]`, so the batching rule folds the mapped axis into the
        // attention batch dimension and the per-item expectations come from unbatched calls.
        let dimensions = [1, 2, 1, 2];
        let operand_type = attention_type(DataType::F32, &dimensions);
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
            item(query)
                .dot_product_attention(&item(key), &item(value), scale, AttentionMask::Causal, None)
                .unwrap()
        };

        let stacked_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![
                Dimension::Static(2),
                Dimension::Static(1),
                Dimension::Static(2),
                Dimension::Static(1),
                Dimension::Static(2),
            ]),
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

        // The two-output training forward maps both the attended output and the statistic at axis 0.
        let training = operation.with_activation_output();
        let outputs = training
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
        let attend_with_activation = |query: &[f64], key: &[f64], value: &[f64]| {
            item(query)
                .dot_product_attention_with_activation(
                    &item(key),
                    &item(value),
                    None,
                    None,
                    scale,
                    AttentionMask::Causal,
                    None,
                    None,
                )
                .unwrap()
        };
        let (item_0_output, item_0_statistic) = attend_with_activation(&item_0_query, &item_0_key, &item_0_value);
        let (item_1_output, item_1_statistic) = attend_with_activation(&item_1_query, &item_1_key, &item_1_value);
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[1].batch_axis(), BatchAxis::new(0));
        let expected_outputs: Vec<f64> = item_0_output.to_f64s().into_iter().chain(item_1_output.to_f64s()).collect();
        let expected_statistics: Vec<f64> =
            item_0_statistic.to_f64s().into_iter().chain(item_1_statistic.to_f64s()).collect();
        assert_eq!(outputs[0].value().to_f64s(), expected_outputs);
        assert_eq!(outputs[1].value().to_f64s(), expected_statistics);
        assert_eq!(outputs[1].value().r#type().as_ref(), &attention_type(DataType::F32, &[2, 1, 1, 2]),);

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
            .unwrap()
            .into_parts();
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
    fn test_dot_product_attention_backward_batching() {
        // The backward operation batches with the same merge-reshape rule as the forward: two batch items with
        // per-item shape `[1, 2, 1, 2]` fold into the operation's batch dimension and match per-item backward calls.
        let dimensions = [1, 2, 1, 2];
        let operand_type = attention_type(DataType::F64, &dimensions);
        let item_0_query = vec![0.5, -0.25, 1.0, 0.75];
        let item_0_key = vec![0.25, 0.5, -0.5, 1.0];
        let item_0_value = vec![1.0, 2.0, -1.0, 0.5];
        let item_1_query = vec![-0.5, 0.75, 0.25, -1.0];
        let item_1_key = vec![1.0, -0.25, 0.5, 0.5];
        let item_1_value = vec![0.5, -1.5, 2.0, 1.0];
        let item_0_cotangent = vec![1.0, -0.5, 0.25, 2.0];
        let item_1_cotangent = vec![-1.0, 0.5, 1.5, -0.25];
        let item = |values: &[f64]| Array::from_f64s(operand_type.clone(), values.to_vec());
        let scale = 0.5;
        let mask = AttentionMask::Causal;
        let forward = |query: &[f64], key: &[f64], value: &[f64]| {
            item(query)
                .dot_product_attention_with_activation(&item(key), &item(value), None, None, scale, mask, None, None)
                .unwrap()
        };
        let backward = |query: &[f64], key: &[f64], value: &[f64], cotangent: &[f64]| {
            let (output, activation) = forward(query, key, value);
            item(query)
                .dot_product_attention_backward(
                    &item(key),
                    &item(value),
                    &output,
                    &activation,
                    &item(cotangent),
                    scale,
                    mask,
                    None,
                )
                .unwrap()
        };
        let (item_0_output, item_0_activation) = forward(&item_0_query, &item_0_key, &item_0_value);
        let (item_1_output, item_1_activation) = forward(&item_1_query, &item_1_key, &item_1_value);

        let stack = |first: &Array, second: &Array| {
            let mut stacked_dimensions = vec![2];
            stacked_dimensions.extend(first.r#type().static_shape().unwrap().dimensions().iter().copied());
            let stacked_type = attention_type(first.r#type().data_type(), &stacked_dimensions);
            let values = first.to_f64s().into_iter().chain(second.to_f64s()).collect::<Vec<_>>();
            let value = Array::from_f64s(stacked_type, values);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0)).unwrap()
        };
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);
        let operation = DotProductAttentionBackwardOperation::new(scale, mask);
        let outputs = operation
            .batch(
                &context,
                &EmptyRegionDriver,
                &[
                    stack(&item(&item_0_query), &item(&item_1_query)),
                    stack(&item(&item_0_key), &item(&item_1_key)),
                    stack(&item(&item_0_value), &item(&item_1_value)),
                    stack(&item_0_output, &item_1_output),
                    stack(&item_0_activation, &item_1_activation),
                    stack(&item(&item_0_cotangent), &item(&item_1_cotangent)),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 3);
        let (item_0_dq, item_0_dk, item_0_dv) = backward(&item_0_query, &item_0_key, &item_0_value, &item_0_cotangent);
        let (item_1_dq, item_1_dk, item_1_dv) = backward(&item_1_query, &item_1_key, &item_1_value, &item_1_cotangent);
        for (output, (first, second)) in
            outputs.iter().zip([(item_0_dq, item_1_dq), (item_0_dk, item_1_dk), (item_0_dv, item_1_dv)])
        {
            assert_eq!(output.batch_axis(), BatchAxis::new(0));
            let expected: Vec<f64> = first.to_f64s().into_iter().chain(second.to_f64s()).collect();
            for (actual, expected) in output.value().to_f64s().into_iter().zip(expected) {
                assert_abs_diff_eq!(actual, expected, epsilon = 1e-9);
            }
        }

        // A replicated broadcast-batch bias (per-item batch of 2, bias batch of 1) is materialized before merging
        // and its cotangent is summed back over the per-item batch, matching the unbatched bias cotangent.
        let wide_dimensions = [2, 2, 1, 2];
        let wide_type = attention_type(DataType::F64, &wide_dimensions);
        let wide = |values: &[f64]| Array::from_f64s(wide_type.clone(), values.to_vec());
        let item_0_wide_query: Vec<f64> = item_0_query.iter().chain(item_1_query.iter()).copied().collect();
        let item_1_wide_query: Vec<f64> = item_1_query.iter().chain(item_0_query.iter()).copied().collect();
        let item_0_wide_key: Vec<f64> = item_0_key.iter().chain(item_1_key.iter()).copied().collect();
        let item_1_wide_key: Vec<f64> = item_1_key.iter().chain(item_0_key.iter()).copied().collect();
        let item_0_wide_value: Vec<f64> = item_0_value.iter().chain(item_1_value.iter()).copied().collect();
        let item_1_wide_value: Vec<f64> = item_1_value.iter().chain(item_0_value.iter()).copied().collect();
        let item_0_wide_cotangent: Vec<f64> = item_0_cotangent.iter().chain(item_1_cotangent.iter()).copied().collect();
        let item_1_wide_cotangent: Vec<f64> = item_1_cotangent.iter().chain(item_0_cotangent.iter()).copied().collect();
        let bias_values = vec![0.25, -0.5, 0.75, 0.125];
        let bias = Array::from_f64s(attention_type(DataType::F64, &[1, 1, 2, 2]), bias_values.clone());
        let wide_forward = |query: &[f64], key: &[f64], value: &[f64]| {
            wide(query)
                .dot_product_attention_with_activation(
                    &wide(key),
                    &wide(value),
                    Some(&bias),
                    None,
                    scale,
                    mask,
                    None,
                    None,
                )
                .unwrap()
        };
        let wide_backward = |query: &[f64], key: &[f64], value: &[f64], cotangent: &[f64]| {
            let (output, activation) = wide_forward(query, key, value);
            wide(query)
                .dot_product_attention_backward_with_bias(
                    &wide(key),
                    &wide(value),
                    &bias,
                    &output,
                    &activation,
                    &wide(cotangent),
                    scale,
                    mask,
                    None,
                )
                .unwrap()
        };
        let (item_0_wide_output, item_0_wide_activation) =
            wide_forward(&item_0_wide_query, &item_0_wide_key, &item_0_wide_value);
        let (item_1_wide_output, item_1_wide_activation) =
            wide_forward(&item_1_wide_query, &item_1_wide_key, &item_1_wide_value);
        let outputs = operation
            .batch(
                &context,
                &EmptyRegionDriver,
                &[
                    stack(&wide(&item_0_wide_query), &wide(&item_1_wide_query)),
                    stack(&wide(&item_0_wide_key), &wide(&item_1_wide_key)),
                    stack(&wide(&item_0_wide_value), &wide(&item_1_wide_value)),
                    ArrayBatch::replicated(bias.clone()),
                    stack(&item_0_wide_output, &item_1_wide_output),
                    stack(&item_0_wide_activation, &item_1_wide_activation),
                    stack(&wide(&item_0_wide_cotangent), &wide(&item_1_wide_cotangent)),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 4);
        let item_0_expected =
            wide_backward(&item_0_wide_query, &item_0_wide_key, &item_0_wide_value, &item_0_wide_cotangent);
        let item_1_expected =
            wide_backward(&item_1_wide_query, &item_1_wide_key, &item_1_wide_value, &item_1_wide_cotangent);
        let expected = [
            (item_0_expected.0, item_1_expected.0),
            (item_0_expected.1, item_1_expected.1),
            (item_0_expected.2, item_1_expected.2),
            (item_0_expected.3, item_1_expected.3),
        ];
        for (output, (first, second)) in outputs.iter().zip(expected) {
            assert_eq!(output.batch_axis(), BatchAxis::new(0));
            let expected: Vec<f64> = first.to_f64s().into_iter().chain(second.to_f64s()).collect();
            for (actual, expected) in output.value().to_f64s().into_iter().zip(expected) {
                assert_abs_diff_eq!(actual, expected, epsilon = 1e-9);
            }
        }
        // The bias cotangent restored its broadcast batch dimension: `[v, 1, 1, t, s]`.
        assert_eq!(outputs[3].value().r#type().as_ref(), &attention_type(DataType::F64, &[2, 1, 1, 2, 2]),);
    }

    #[test]
    fn test_dot_product_attention_backward() {
        // Numeric check of the backward cotangents against central finite differences of the forward composition on
        // the reference backend, at `f64` for tight tolerances. The loss is `Σ output ∘ seed` for a fixed seed, so
        // the backward operation is fed `seed` as the output cotangent.
        let scale = 0.5;

        // Case A: plain causal attention, no grouped heads, no bias, no window.
        {
            let query_dimensions = [1, 3, 2, 2];
            let query_type = attention_type(DataType::F64, &query_dimensions);
            let key_value_type = attention_type(DataType::F64, &[1, 4, 2, 2]);
            let query_values = test_values(12, 7, 11, 5.0, 0.25);
            let key_values = test_values(16, 5, 13, 6.0, 0.25);
            let value_values = test_values(16, 3, 7, 3.0, 0.5);
            let seed = test_values(12, 13, 19, 9.0, 0.125);
            let mask = AttentionMask::Causal;
            let query = Array::from_f64s(query_type.clone(), query_values.clone());
            let key = Array::from_f64s(key_value_type.clone(), key_values.clone());
            let value = Array::from_f64s(key_value_type.clone(), value_values.clone());
            let (output, activation) = query
                .dot_product_attention_with_activation(&key, &value, None, None, scale, mask, None, None)
                .unwrap();
            let output_cotangent = Array::from_f64s(query_type.clone(), seed.clone());
            let (query_cotangent, key_cotangent, value_cotangent) = query
                .dot_product_attention_backward(
                    &key,
                    &value,
                    &output,
                    &activation,
                    &output_cotangent,
                    scale,
                    mask,
                    None,
                )
                .unwrap();

            let loss = |query_values: &[f64], key_values: &[f64], value_values: &[f64]| -> f64 {
                let query = Array::from_f64s(query_type.clone(), query_values.to_vec());
                let key = Array::from_f64s(key_value_type.clone(), key_values.to_vec());
                let value = Array::from_f64s(key_value_type.clone(), value_values.to_vec());
                let output = query.dot_product_attention(&key, &value, scale, mask, None).unwrap();
                output.to_f64s().iter().zip(seed.iter()).map(|(output, seed)| output * seed).sum()
            };
            let expected_query_cotangent =
                central_difference(|values| loss(values, &key_values, &value_values), &query_values);
            let expected_key_cotangent =
                central_difference(|values| loss(&query_values, values, &value_values), &key_values);
            let expected_value_cotangent =
                central_difference(|values| loss(&query_values, &key_values, values), &value_values);
            for (actual, expected) in query_cotangent.to_f64s().into_iter().zip(expected_query_cotangent) {
                assert_abs_diff_eq!(actual, expected, epsilon = 1e-6);
            }
            for (actual, expected) in key_cotangent.to_f64s().into_iter().zip(expected_key_cotangent) {
                assert_abs_diff_eq!(actual, expected, epsilon = 1e-6);
            }
            for (actual, expected) in value_cotangent.to_f64s().into_iter().zip(expected_value_cotangent) {
                assert_abs_diff_eq!(actual, expected, epsilon = 1e-6);
            }

            // Key/value position 3 is causally masked for every query row (`t = 3`, `s = 4`), so its recovered
            // attention weight is exactly zero and it contributes exactly zero gradient.
            let key_cotangents = key_cotangent.to_f64s();
            let value_cotangents = value_cotangent.to_f64s();
            for head in 0..2 {
                for dimension in 0..2 {
                    assert_eq!(key_cotangents[(3 * 2 + head) * 2 + dimension], 0.0);
                    assert_eq!(value_cotangents[(3 * 2 + head) * 2 + dimension], 0.0);
                }
            }
        }

        // Case B: causal + grouped-query heads + bias + sliding window combined.
        {
            let query_dimensions = [1, 3, 2, 2];
            let query_type = attention_type(DataType::F64, &query_dimensions);
            let key_value_type = attention_type(DataType::F64, &[1, 4, 1, 2]);
            let bias_type = attention_type(DataType::F64, &[1, 2, 3, 4]);
            let query_values = test_values(12, 7, 11, 5.0, 0.25);
            let key_values = test_values(8, 5, 13, 6.0, 0.25);
            let value_values = test_values(8, 3, 7, 3.0, 0.5);
            let bias_values = test_values(24, 11, 17, 8.0, 0.125);
            let seed = test_values(12, 13, 19, 9.0, 0.125);
            let mask = AttentionMask::Causal;
            let window = Some(2);
            let query = Array::from_f64s(query_type.clone(), query_values.clone());
            let key = Array::from_f64s(key_value_type.clone(), key_values.clone());
            let value = Array::from_f64s(key_value_type.clone(), value_values.clone());
            let bias = Array::from_f64s(bias_type.clone(), bias_values.clone());
            let (output, activation) = query
                .dot_product_attention_with_activation(&key, &value, Some(&bias), None, scale, mask, window, None)
                .unwrap();
            let output_cotangent = Array::from_f64s(query_type.clone(), seed.clone());
            let (query_cotangent, key_cotangent, value_cotangent, bias_cotangent) = query
                .dot_product_attention_backward_with_bias(
                    &key,
                    &value,
                    &bias,
                    &output,
                    &activation,
                    &output_cotangent,
                    scale,
                    mask,
                    window,
                )
                .unwrap();
            assert_eq!(bias_cotangent.r#type().as_ref(), &bias_type);

            let loss = |query_values: &[f64], key_values: &[f64], value_values: &[f64], bias_values: &[f64]| -> f64 {
                let query = Array::from_f64s(query_type.clone(), query_values.to_vec());
                let key = Array::from_f64s(key_value_type.clone(), key_values.to_vec());
                let value = Array::from_f64s(key_value_type.clone(), value_values.to_vec());
                let bias = Array::from_f64s(bias_type.clone(), bias_values.to_vec());
                let output = query.dot_product_attention_with_bias(&key, &value, &bias, scale, mask, window).unwrap();
                output.to_f64s().iter().zip(seed.iter()).map(|(output, seed)| output * seed).sum()
            };
            let cases: [(&Array, &[f64], Box<dyn Fn(&[f64]) -> f64>); 4] = [
                (
                    &query_cotangent,
                    &query_values,
                    Box::new(|values: &[f64]| loss(values, &key_values, &value_values, &bias_values)),
                ),
                (
                    &key_cotangent,
                    &key_values,
                    Box::new(|values: &[f64]| loss(&query_values, values, &value_values, &bias_values)),
                ),
                (
                    &value_cotangent,
                    &value_values,
                    Box::new(|values: &[f64]| loss(&query_values, &key_values, values, &bias_values)),
                ),
                (
                    &bias_cotangent,
                    &bias_values,
                    Box::new(|values: &[f64]| loss(&query_values, &key_values, &value_values, values)),
                ),
            ];
            for (cotangent, values, loss) in cases {
                let expected = central_difference(loss, values);
                for (actual, expected) in cotangent.to_f64s().into_iter().zip(expected) {
                    assert_abs_diff_eq!(actual, expected, epsilon = 1e-6);
                }
            }

            // Key/value position 3 lies outside every row's window (`t = 3`, `window = 2`), so it contributes
            // exactly zero gradient even with a bias present.
            let key_cotangents = key_cotangent.to_f64s();
            let value_cotangents = value_cotangent.to_f64s();
            for dimension in 0..2 {
                assert_eq!(key_cotangents[3 * 2 + dimension], 0.0);
                assert_eq!(value_cotangents[3 * 2 + dimension], 0.0);
            }
        }
    }

    #[test]
    fn test_dot_product_attention_padding_batching() {
        // The rank-1 `i32[batch]` sequence-length operands ride the merge-reshape batching rule: mapped `[v, batch]`
        // lengths merge to `[v * batch]`, concatenating the per-item lengths along the folded axis, and replicated
        // lengths broadcast-materialize into per-item copies.
        let dimensions = [1, 2, 1, 2];
        let operand_type = attention_type(DataType::F32, &dimensions);
        let lengths_type = attention_type(DataType::I32, &[1]);
        let item_0_query = vec![0.5, -0.25, 1.0, 0.75];
        let item_0_key = vec![0.25, 0.5, -0.5, 1.0];
        let item_0_value = vec![1.0, 2.0, -1.0, 0.5];
        let item_1_query = vec![-0.5, 0.75, 0.25, -1.0];
        let item_1_key = vec![1.0, -0.25, 0.5, 0.5];
        let item_1_value = vec![0.5, -1.5, 2.0, 1.0];
        let item = |values: &[f64]| Array::from_f64s(operand_type.clone(), values.to_vec());
        let lengths = |values: &[f64]| Array::from_f64s(lengths_type.clone(), values.to_vec());
        let scale = 0.5;
        let mask = AttentionMask::None;
        let operation = DotProductAttentionOperation::new(scale, mask);
        let attend = |query: &[f64], key: &[f64], value: &[f64], query_length: f64, key_value_length: f64| {
            item(query)
                .dot_product_attention_with_options(
                    &item(key),
                    &item(value),
                    None,
                    Some((&lengths(&[query_length]), &lengths(&[key_value_length]))),
                    scale,
                    mask,
                    None,
                    None,
                )
                .unwrap()
        };
        let stack = |first: &Array, second: &Array| {
            let mut stacked_dimensions = vec![2];
            stacked_dimensions.extend(first.r#type().static_shape().unwrap().dimensions().iter().copied());
            let stacked_type = attention_type(first.r#type().data_type(), &stacked_dimensions);
            let values = first.to_f64s().into_iter().chain(second.to_f64s()).collect::<Vec<_>>();
            let value = Array::from_f64s(stacked_type, values);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0)).unwrap()
        };
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);

        // Distinct mapped per-item lengths concatenate along the folded batch axis.
        let outputs = operation
            .batch(
                &context,
                &EmptyRegionDriver,
                &[
                    stack(&item(&item_0_query), &item(&item_1_query)),
                    stack(&item(&item_0_key), &item(&item_1_key)),
                    stack(&item(&item_0_value), &item(&item_1_value)),
                    stack(&lengths(&[2.0]), &lengths(&[1.0])),
                    stack(&lengths(&[1.0]), &lengths(&[2.0])),
                ],
            )
            .unwrap();
        let expected: Vec<f64> = attend(&item_0_query, &item_0_key, &item_0_value, 2.0, 1.0)
            .to_f64s()
            .into_iter()
            .chain(attend(&item_1_query, &item_1_key, &item_1_value, 1.0, 2.0).to_f64s())
            .collect();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().to_f64s(), expected);

        // Replicated lengths broadcast-materialize into per-item copies, like the replicated bias.
        let outputs = operation
            .batch(
                &context,
                &EmptyRegionDriver,
                &[
                    stack(&item(&item_0_query), &item(&item_1_query)),
                    stack(&item(&item_0_key), &item(&item_1_key)),
                    stack(&item(&item_0_value), &item(&item_1_value)),
                    ArrayBatch::replicated(lengths(&[1.0])),
                    ArrayBatch::replicated(lengths(&[2.0])),
                ],
            )
            .unwrap();
        let expected: Vec<f64> = attend(&item_0_query, &item_0_key, &item_0_value, 1.0, 2.0)
            .to_f64s()
            .into_iter()
            .chain(attend(&item_1_query, &item_1_key, &item_1_value, 1.0, 2.0).to_f64s())
            .collect();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().to_f64s(), expected);
    }

    #[test]
    fn test_dot_product_attention_padding() {
        // Variable sequence lengths on the reference backend against the host reference, with distinct per-item
        // lengths: key/value columns `j >= kv_lengths[b]` are fully excluded and query rows `i >= q_lengths[b]` are
        // exact zeros in both the attended output and the activation statistic (the cuDNN `PADDING` semantics), with
        // the causal variant composing as `PADDING_CAUSAL`.
        let dimensions = [3, 4, 2, 2];
        let kv_seq = 5;
        let operand_type = attention_type(DataType::F64, &dimensions);
        let key_value_type = attention_type(DataType::F64, &[3, kv_seq, 2, 2]);
        let query_values = test_values(48, 7, 11, 5.0, 0.25);
        let key_values = test_values(60, 5, 13, 6.0, 0.25);
        let value_values = test_values(60, 3, 7, 3.0, 0.5);
        let query = Array::from_f64s(operand_type.clone(), query_values.clone());
        let key = Array::from_f64s(key_value_type.clone(), key_values.clone());
        let value = Array::from_f64s(key_value_type.clone(), value_values.clone());
        let query_lengths = [4usize, 2, 1];
        let key_value_lengths = [5usize, 3, 1];
        let lengths_type = attention_type(DataType::I32, &[3]);
        let query_lengths_array =
            Array::from_f64s(lengths_type.clone(), query_lengths.iter().map(|&length| length as f64).collect());
        let key_value_lengths_array =
            Array::from_f64s(lengths_type, key_value_lengths.iter().map(|&length| length as f64).collect());
        let sequence_lengths = Some((&query_lengths_array, &key_value_lengths_array));
        let scale = 0.5;

        for causal in [false, true] {
            let mask = if causal { AttentionMask::Causal } else { AttentionMask::None };
            let (output, activation) = query
                .dot_product_attention_with_activation(&key, &value, None, sequence_lengths, scale, mask, None, None)
                .unwrap();
            let (expected_output, expected_statistic) = host_attention(
                &query_values,
                &key_values,
                &value_values,
                None,
                dimensions,
                kv_seq,
                2,
                scale,
                causal,
                None,
                Some((&query_lengths, &key_value_lengths)),
            );
            let output_values = output.to_f64s();
            for (actual, expected) in output_values.iter().zip(expected_output.iter()) {
                assert_abs_diff_eq!(actual, expected, epsilon = 1e-9);
            }
            // The activation statistic runs at `f32`, so its comparison carries the `f32` rounding envelope.
            let activation_values = activation.to_f64s();
            for (actual, expected) in activation_values.iter().zip(expected_statistic.iter()) {
                assert_abs_diff_eq!(actual, expected, epsilon = 1e-5);
            }
            // Out-of-range query rows are exact zeros in both outputs, not merely small values.
            for b in 0..3 {
                for i in query_lengths[b]..4 {
                    for n in 0..2 {
                        for d in 0..2 {
                            assert_eq!(output_values[((b * 4 + i) * 2 + n) * 2 + d], 0.0);
                        }
                        assert_eq!(activation_values[(b * 2 + n) * 4 + i], 0.0);
                    }
                }
            }
        }

        // Backward cotangents match central finite differences of the padded forward (the loss is `Σ output ∘ seed`
        // and out-of-range rows are exactly zero, so they contribute no dependence), and the out-of-range gradient
        // regions are exact zeros: query-cotangent rows `i >= q_lengths[b]` (forced by the select) and
        // key/value-cotangent positions `s >= kv_lengths[b]` (through the zero recovered weights).
        let mask = AttentionMask::Causal;
        let (output, activation) = query
            .dot_product_attention_with_activation(&key, &value, None, sequence_lengths, scale, mask, None, None)
            .unwrap();
        let seed = test_values(48, 13, 19, 9.0, 0.125);
        let output_cotangent = Array::from_f64s(operand_type.clone(), seed.clone());
        let (query_cotangent, key_cotangent, value_cotangent, bias_cotangent) = query
            .dot_product_attention_backward_with_options(
                &key,
                &value,
                None,
                sequence_lengths,
                &output,
                &activation,
                &output_cotangent,
                scale,
                mask,
                None,
                None,
            )
            .unwrap();
        assert!(bias_cotangent.is_none());
        let loss = |query_values: &[f64], key_values: &[f64], value_values: &[f64]| -> f64 {
            let query = Array::from_f64s(operand_type.clone(), query_values.to_vec());
            let key = Array::from_f64s(key_value_type.clone(), key_values.to_vec());
            let value = Array::from_f64s(key_value_type.clone(), value_values.to_vec());
            let output = query
                .dot_product_attention_with_options(&key, &value, None, sequence_lengths, scale, mask, None, None)
                .unwrap();
            output.to_f64s().iter().zip(seed.iter()).map(|(output, seed)| output * seed).sum()
        };
        let expected_query_cotangent =
            central_difference(|values| loss(values, &key_values, &value_values), &query_values);
        let expected_key_cotangent =
            central_difference(|values| loss(&query_values, values, &value_values), &key_values);
        let expected_value_cotangent =
            central_difference(|values| loss(&query_values, &key_values, values), &value_values);
        for (cotangent, expected) in [
            (&query_cotangent, expected_query_cotangent),
            (&key_cotangent, expected_key_cotangent),
            (&value_cotangent, expected_value_cotangent),
        ] {
            for (actual, expected) in cotangent.to_f64s().into_iter().zip(expected) {
                assert_abs_diff_eq!(actual, expected, epsilon = 1e-6);
            }
        }
        let query_cotangents = query_cotangent.to_f64s();
        let key_cotangents = key_cotangent.to_f64s();
        let value_cotangents = value_cotangent.to_f64s();
        for b in 0..3 {
            for i in query_lengths[b]..4 {
                for n in 0..2 {
                    for d in 0..2 {
                        assert_eq!(query_cotangents[((b * 4 + i) * 2 + n) * 2 + d], 0.0);
                    }
                }
            }
            for s in key_value_lengths[b]..kv_seq {
                for n in 0..2 {
                    for d in 0..2 {
                        assert_eq!(key_cotangents[((b * kv_seq + s) * 2 + n) * 2 + d], 0.0);
                        assert_eq!(value_cotangents[((b * kv_seq + s) * 2 + n) * 2 + d], 0.0);
                    }
                }
            }
        }
    }

    #[test]
    fn test_dot_product_attention_dropout() {
        // Dropout rides the operations as an optional `(rate, seed)` attribute: it renders only when set, type
        // inference accepts it (it changes no shapes) while rejecting rates outside the open interval `(0, 1)`, and
        // the portable composition (the reference backend) rejects it because only the fused CUDA lowering
        // implements it.
        let operation = DotProductAttentionOperation::new(0.5, AttentionMask::Causal).with_dropout((0.25, 7));
        assert_eq!(operation.dropout(), Some((0.25, 7)));
        assert_eq!(
            operation.to_string(),
            "dot_product_attention [scale=0.5, mask=causal, dropout_rate=0.25, dropout_seed=7]",
        );
        let backward = DotProductAttentionBackwardOperation::new(0.5, AttentionMask::Causal).with_dropout((0.25, 7));
        assert_eq!(backward.dropout(), Some((0.25, 7)));
        assert_eq!(
            backward.to_string(),
            "dot_product_attention_backward [scale=0.5, mask=causal, dropout_rate=0.25, dropout_seed=7]",
        );

        let query = attention_type(DataType::F32, &[2, 4, 2, 3]);
        let key_value = attention_type(DataType::F32, &[2, 5, 2, 3]);
        check_operation_type_inference!(
            operation = operation,
            cases = [
                {
                    type = ArrayType,
                    input_types = [query.clone(), key_value.clone(), key_value.clone()],
                    output_types = [query.clone()],
                },
            ],
        );
        for rate in [0.0, 1.0, 1.5] {
            check_operation_type_inference!(
                operation = DotProductAttentionOperation::new(0.5, AttentionMask::Causal).with_dropout((rate, 7)),
                cases = [
                    {
                        type = ArrayType,
                        input_types = [query.clone(), key_value.clone(), key_value.clone()],
                        error = format!(
                            "'dot_product_attention' dropout rate must lie in the open interval (0, 1) but got {rate}",
                        ),
                    },
                ],
            );
        }

        let operand_type = attention_type(DataType::F32, &[1, 2, 1, 2]);
        let operand = |values: &[f64]| Array::from_f64s(operand_type.clone(), values.to_vec());
        let query = operand(&[0.5, -0.25, 1.0, 0.75]);
        let key = operand(&[0.25, 0.5, -0.5, 1.0]);
        let value = operand(&[1.0, 2.0, -1.0, 0.5]);
        let forward_error = query
            .dot_product_attention_with_options(
                &key,
                &value,
                None,
                None,
                0.5,
                AttentionMask::Causal,
                None,
                Some((0.25, 7)),
            )
            .unwrap_err();
        assert!(
            forward_error
                .to_string()
                .contains("'dot_product_attention' dropout is only supported by the fused CUDA lowering"),
        );
        let (output, activation) = query
            .dot_product_attention_with_activation(&key, &value, None, None, 0.5, AttentionMask::Causal, None, None)
            .unwrap();
        let backward_error = query
            .dot_product_attention_backward_with_options(
                &key,
                &value,
                None,
                None,
                &output,
                &activation,
                &output,
                0.5,
                AttentionMask::Causal,
                None,
                Some((0.25, 7)),
            )
            .unwrap_err();
        assert!(
            backward_error
                .to_string()
                .contains("'dot_product_attention_backward' dropout is only supported by the fused CUDA lowering"),
        );
    }

    #[test]
    fn test_differentiable_dot_product_attention_gradient() {
        // Reverse-mode differentiation through the `custom_vjp` training entry point on the reference backend
        // matches central finite differences of the forward composition, with the loss `Σ output`.
        let scale = 0.5;
        let mask = AttentionMask::Causal;
        let query_dimensions = [1, 2, 2, 2];
        let query_type = attention_type(DataType::F64, &query_dimensions);
        let key_value_type = attention_type(DataType::F64, &[1, 3, 1, 2]);
        let query_values = test_values(8, 7, 11, 5.0, 0.25);
        let key_values = test_values(6, 5, 13, 6.0, 0.25);
        let value_values = test_values(6, 3, 7, 3.0, 0.5);
        let query = Array::from_f64s(query_type.clone(), query_values.clone());
        let key = Array::from_f64s(key_value_type.clone(), key_values.clone());
        let value = Array::from_f64s(key_value_type.clone(), value_values.clone());

        let function =
            differentiable_dot_product_attention::<EagerContext<Array, ArrayOperation<Array>>>(scale, mask, None, None);
        let (loss_value, (query_gradient, key_gradient, value_gradient)) = value_and_gradient(
            |(query, key, value)| function.call((query, key, value)).unwrap().reduce(&[0, 1, 2, 3], ReductionKind::Sum),
            (query.clone(), key.clone(), value.clone()),
        )
        .unwrap();
        let loss = |query_values: &[f64], key_values: &[f64], value_values: &[f64]| -> f64 {
            let query = Array::from_f64s(query_type.clone(), query_values.to_vec());
            let key = Array::from_f64s(key_value_type.clone(), key_values.to_vec());
            let value = Array::from_f64s(key_value_type.clone(), value_values.to_vec());
            query.dot_product_attention(&key, &value, scale, mask, None).unwrap().to_f64s().iter().sum()
        };
        assert_abs_diff_eq!(loss_value.to_f64s()[0], loss(&query_values, &key_values, &value_values), epsilon = 1e-9);
        let expected_query_gradient =
            central_difference(|values| loss(values, &key_values, &value_values), &query_values);
        let expected_key_gradient =
            central_difference(|values| loss(&query_values, values, &value_values), &key_values);
        let expected_value_gradient =
            central_difference(|values| loss(&query_values, &key_values, values), &value_values);
        for (gradient, expected) in [
            (query_gradient, expected_query_gradient),
            (key_gradient, expected_key_gradient),
            (value_gradient, expected_value_gradient),
        ] {
            for (actual, expected) in gradient.to_f64s().into_iter().zip(expected) {
                assert_abs_diff_eq!(actual, expected, epsilon = 1e-6);
            }
        }

        // The bias-carrying entry point additionally produces the bias gradient.
        let bias_type = attention_type(DataType::F64, &[1, 1, 2, 3]);
        let bias_values = test_values(6, 11, 17, 8.0, 0.125);
        let bias = Array::from_f64s(bias_type.clone(), bias_values.clone());
        let function = differentiable_dot_product_attention_with_bias::<EagerContext<Array, ArrayOperation<Array>>>(
            scale, mask, None, None,
        );
        let (_, (_, _, _, bias_gradient)) = value_and_gradient(
            |(query, key, value, bias)| {
                function.call((query, key, value, bias)).unwrap().reduce(&[0, 1, 2, 3], ReductionKind::Sum)
            },
            (query, key, value, bias),
        )
        .unwrap();
        let bias_loss = |bias_values: &[f64]| -> f64 {
            let query = Array::from_f64s(query_type.clone(), query_values.to_vec());
            let key = Array::from_f64s(key_value_type.clone(), key_values.to_vec());
            let value = Array::from_f64s(key_value_type.clone(), value_values.to_vec());
            let bias = Array::from_f64s(bias_type.clone(), bias_values.to_vec());
            query
                .dot_product_attention_with_bias(&key, &value, &bias, scale, mask, None)
                .unwrap()
                .to_f64s()
                .iter()
                .sum()
        };
        let expected_bias_gradient = central_difference(bias_loss, &bias_values);
        for (actual, expected) in bias_gradient.to_f64s().into_iter().zip(expected_bias_gradient) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_differentiable_dot_product_attention_with_sequence_lengths_gradient() {
        // The sequence-length-carrying `custom_vjp` entry point differentiates through the padded forward on the
        // reference backend: the query/key/value gradients match central finite differences of the padded forward,
        // and the non-differentiated `i32` sequence lengths receive first-class zero-space gradients.
        let scale = 0.5;
        let mask = AttentionMask::Causal;
        let query_dimensions = [2, 3, 1, 2];
        let query_type = attention_type(DataType::F64, &query_dimensions);
        let key_value_type = attention_type(DataType::F64, &[2, 4, 1, 2]);
        let lengths_type = attention_type(DataType::I32, &[2]);
        let query_values = test_values(12, 7, 11, 5.0, 0.25);
        let key_values = test_values(16, 5, 13, 6.0, 0.25);
        let value_values = test_values(16, 3, 7, 3.0, 0.5);
        let query = Array::from_f64s(query_type.clone(), query_values.clone());
        let key = Array::from_f64s(key_value_type.clone(), key_values.clone());
        let value = Array::from_f64s(key_value_type.clone(), value_values.clone());
        let query_lengths = Array::from_f64s(lengths_type.clone(), vec![3.0, 2.0]);
        let key_value_lengths = Array::from_f64s(lengths_type, vec![4.0, 2.0]);

        let function = differentiable_dot_product_attention_with_sequence_lengths::<
            EagerContext<Array, ArrayOperation<Array>>,
        >(scale, mask, None, None);
        let (loss_value, (query_gradient, key_gradient, value_gradient, query_lengths_gradient, _)) =
            value_and_gradient(
                |(query, key, value, query_lengths, key_value_lengths)| {
                    function
                        .call((query, key, value, query_lengths, key_value_lengths))
                        .unwrap()
                        .reduce(&[0, 1, 2, 3], ReductionKind::Sum)
                },
                (query.clone(), key.clone(), value.clone(), query_lengths.clone(), key_value_lengths.clone()),
            )
            .unwrap();
        let loss = |query_values: &[f64], key_values: &[f64], value_values: &[f64]| -> f64 {
            let query = Array::from_f64s(query_type.clone(), query_values.to_vec());
            let key = Array::from_f64s(key_value_type.clone(), key_values.to_vec());
            let value = Array::from_f64s(key_value_type.clone(), value_values.to_vec());
            query
                .dot_product_attention_with_options(
                    &key,
                    &value,
                    None,
                    Some((&query_lengths, &key_value_lengths)),
                    scale,
                    mask,
                    None,
                    None,
                )
                .unwrap()
                .to_f64s()
                .iter()
                .sum()
        };
        assert_abs_diff_eq!(loss_value.to_f64s()[0], loss(&query_values, &key_values, &value_values), epsilon = 1e-9);
        let expected_query_gradient =
            central_difference(|values| loss(values, &key_values, &value_values), &query_values);
        let expected_key_gradient =
            central_difference(|values| loss(&query_values, values, &value_values), &key_values);
        let expected_value_gradient =
            central_difference(|values| loss(&query_values, &key_values, values), &value_values);
        for (gradient, expected) in [
            (query_gradient, expected_query_gradient),
            (key_gradient, expected_key_gradient),
            (value_gradient, expected_value_gradient),
        ] {
            for (actual, expected) in gradient.to_f64s().into_iter().zip(expected) {
                assert_abs_diff_eq!(actual, expected, epsilon = 1e-6);
            }
        }
        // The non-differentiable `i32` sequence lengths carry first-class zero-space gradients.
        assert_eq!(query_lengths_gradient.r#type().data_type(), DataType::Zero);
    }

    #[test]
    fn test_dot_product_attention_differentiation_rejection() {
        // The plain (inference) operation rejects differentiation with an error directing to the `custom_vjp`
        // training entry point, and transposition reports the standard non-transposable-operation error. The
        // backward operation likewise rejects second-order differentiation.
        let operand_type = attention_type(DataType::F32, &[1, 2, 1, 2]);
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
                "'dot_product_attention' does not support differentiation; use \
                 'differentiable_dot_product_attention' for the training path",
            ),
        ));
        check_operation_transposition!(
            @rejected,
            operation = DotProductAttentionOperation::new(0.5, AttentionMask::None),
            input_types = [operand_type.clone(), operand_type.clone(), operand_type.clone()],
        );

        let activation_type = attention_type(DataType::F32, &[1, 1, 2]);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let inputs = vec![
            builder.add_input(operand_type.clone()),
            builder.add_input(operand_type.clone()),
            builder.add_input(operand_type.clone()),
            builder.add_input(operand_type.clone()),
            builder.add_input(activation_type.clone()),
            builder.add_input(operand_type.clone()),
        ];
        let backward_operation = DotProductAttentionBackwardOperation::new(0.5, AttentionMask::None);
        let outputs = builder.add_instruction(backward_operation, Vec::new(), inputs).unwrap().to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 6], vec![Placeholder; 3])
            .unwrap();
        assert!(matches!(
            program.jvp(),
            Err(error) if error.to_string().contains(
                "'dot_product_attention_backward' does not support differentiation; differentiate an explicit \
                 attention composition for higher-order derivatives",
            ),
        ));
        check_operation_transposition!(
            @rejected,
            operation = DotProductAttentionBackwardOperation::new(0.5, AttentionMask::None),
            input_types = [
                operand_type.clone(),
                operand_type.clone(),
                operand_type.clone(),
                operand_type.clone(),
                activation_type,
                operand_type,
            ],
        );
    }

    #[test]
    fn test_dot_product_attention_rendering() {
        // The staged programs pin the operations' renderings, including the optional payload fields.
        let query_type = attention_type(DataType::BF16, &[2, 4, 2, 3]);
        let key_value_type = attention_type(DataType::BF16, &[2, 5, 2, 3]);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let inputs = vec![
            builder.add_input(query_type.clone()),
            builder.add_input(key_value_type.clone()),
            builder.add_input(key_value_type.clone()),
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

        // The training forward renders its sliding window and activation flag, and produces two outputs.
        let bias_type = attention_type(DataType::BF16, &[1, 1, 4, 5]);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let inputs = vec![
            builder.add_input(query_type.clone()),
            builder.add_input(key_value_type.clone()),
            builder.add_input(key_value_type.clone()),
            builder.add_input(bias_type.clone()),
        ];
        let training = DotProductAttentionOperation::new(0.125, AttentionMask::Causal)
            .with_sliding_window(3)
            .with_activation_output();
        let outputs = builder.add_instruction(training, Vec::new(), inputs).unwrap().to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 4], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bf16[2, 4, 2, 3], %1:bf16[2, 5, 2, 3], %2:bf16[2, 5, 2, 3], %3:bf16[1, 1, 4, 5] .
                let %4:bf16[2, 4, 2, 3], %5:f32[2, 2, 4] = dot_product_attention [scale=0.125, mask=causal, \
                sliding_window=3, activation=true] %0 %1 %2 %3
                in (%4, %5)
            "}
            .trim_end(),
        );

        // The padded (sequence-length) form appends the trailing `i32[batch]` pair, and dropout renders its rate
        // and seed fields only when set.
        let lengths_type = attention_type(DataType::I32, &[2]);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let inputs = vec![
            builder.add_input(query_type.clone()),
            builder.add_input(key_value_type.clone()),
            builder.add_input(key_value_type.clone()),
            builder.add_input(lengths_type.clone()),
            builder.add_input(lengths_type.clone()),
        ];
        let padded = DotProductAttentionOperation::new(0.125, AttentionMask::Causal).with_dropout((0.5, 42));
        let output = builder.add_instruction(padded, Vec::new(), inputs).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 5], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bf16[2, 4, 2, 3], %1:bf16[2, 5, 2, 3], %2:bf16[2, 5, 2, 3], %3:i32[2], %4:i32[2] .
                let %5:bf16[2, 4, 2, 3] = dot_product_attention [scale=0.125, mask=causal, dropout_rate=0.5, \
                dropout_seed=42] %0 %1 %2 %3 %4
                in (%5)
            "}
            .trim_end(),
        );

        // The backward operation renders its payload and produces one cotangent per differentiated operand.
        let activation_type = attention_type(DataType::F32, &[2, 2, 4]);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let inputs = vec![
            builder.add_input(query_type.clone()),
            builder.add_input(key_value_type.clone()),
            builder.add_input(key_value_type.clone()),
            builder.add_input(bias_type),
            builder.add_input(query_type.clone()),
            builder.add_input(activation_type),
            builder.add_input(query_type),
        ];
        let backward = DotProductAttentionBackwardOperation::new(0.125, AttentionMask::Causal).with_sliding_window(3);
        let outputs = builder.add_instruction(backward, Vec::new(), inputs).unwrap().to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 7], vec![Placeholder; 4])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bf16[2, 4, 2, 3], %1:bf16[2, 5, 2, 3], %2:bf16[2, 5, 2, 3], %3:bf16[1, 1, 4, 5], \
                %4:bf16[2, 4, 2, 3], %5:f32[2, 2, 4], %6:bf16[2, 4, 2, 3] .
                let %7:bf16[2, 4, 2, 3], %8:bf16[2, 5, 2, 3], %9:bf16[2, 5, 2, 3], %10:bf16[1, 1, 4, 5] = \
                dot_product_attention_backward [scale=0.125, mask=causal, sliding_window=3] %0 %1 %2 %3 %4 %5 %6
                in (%7, %8, %9, %10)
            "}
            .trim_end(),
        );
    }
}
