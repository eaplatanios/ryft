use std::fmt::Display;

use ryft_macros::Parameterized;

use crate::arrays::batching::{
    DynamicArrayBatchingPolicy, broadcast_array, dimension_constant, folded_array_dimension,
};
use crate::arrays::{
    ArrayBatch, ArrayBatching, ArrayBatchingPolicy, ArrayIrType, ArrayType, DataType, Dimension, DimensionType,
    DimensionValue, Shape, Sharding, StaticArrayBatchingPolicy,
};
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain, ProjectedContext};
use crate::differentiation::DifferentiableType;
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::compare::{Compare, ComparisonDirection};
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::constants::fill::Fill;
use crate::operations::constants::iota::Iota;
use crate::operations::control_flow::select::Select;
use crate::operations::dimensions::dimension_mul::DimensionMulOperation;
use crate::operations::dimensions::dimension_size::DimensionSizeOperation;
use crate::operations::logical::and::And;
use crate::operations::manipulation::broadcasting::{Broadcast, DynamicBroadcastOperation};
use crate::operations::manipulation::conversion::ConvertElementType;
use crate::operations::manipulation::reshaping::{DynamicReshapeOperation, Reshape};
use crate::operations::manipulation::transposition::Transpose;
use crate::operations::math::add::Add;
use crate::operations::math::div::Div;
use crate::operations::math::dot::{Dot, DotDimensionNumbers};
use crate::operations::math::exp::Exp;
use crate::operations::math::log::Log;
use crate::operations::math::mul::Mul;
use crate::operations::math::reduce::{Reduce, ReduceOperation, ReductionKind};
use crate::operations::math::sub::Sub;
use crate::parameters::Parameter;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    Operation, OperationFormatter, OperationProjection, ProgramError, RegionInterface, TypeError, Typed, Value,
    ValueProjection,
};

mod composition;
mod differentiation;

pub use composition::{dot_product_attention_backward_ir_composition, dot_product_attention_ir_composition};
pub use differentiation::differentiable_dot_product_attention;

/// Canonical operation name for [`DotProductAttentionOperation`].
pub const DOT_PRODUCT_ATTENTION_OPERATION_NAME: &str = "dot_product_attention";

/// Canonical operation name for [`DotProductAttentionBackwardOperation`].
pub const DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME: &str = "dot_product_attention_backward";

/// Backend implementation requested for an attention operation.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum AttentionImplementation {
    /// Uses a fused implementation when the active backend supports the complete configuration and otherwise uses
    /// the portable semantic composition.
    #[default]
    Automatic,

    /// Always uses the portable semantic composition.
    Portable,

    /// Requires a fused backend implementation and reports an error when the configuration is not supported.
    Fused,
}

impl Display for AttentionImplementation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Automatic => formatter.write_str("automatic"),
            Self::Portable => formatter.write_str("portable"),
            Self::Fused => formatter.write_str("fused"),
        }
    }
}

/// Value-independent semantics of scaled dot-product attention.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct AttentionConfiguration {
    /// Explicit score scale, or [`None`] to use `1 / sqrt(head_dimension)`.
    scale: Option<f64>,

    /// Whether query position `i` may attend only to key/value positions `j <= i`.
    causal: bool,

    /// Optional inclusive `(left, right)` local-window radii.
    local_window: Option<(usize, usize)>,

    /// Requested backend implementation.
    implementation: AttentionImplementation,

    /// Whether the forward operation also returns its log-sum-exp residual.
    return_residual: bool,

    /// Optional fused-only `(rate, seed)` dropout extension.
    dropout: Option<(f64, u64)>,
}

impl AttentionConfiguration {
    /// Creates the default configuration: inferred scale, non-causal global attention, automatic implementation
    /// selection, no residual output, and no dropout.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets an explicit score scale. Passing [`None`] restores `1 / sqrt(head_dimension)`.
    #[inline]
    pub fn with_scale<S: Into<Option<f64>>>(mut self, scale: S) -> Self {
        self.scale = scale.into();
        self
    }

    /// Enables or disables causal masking.
    #[inline]
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    /// Sets the inclusive `(left, right)` local-window radii. The local window is independent of causal masking.
    #[inline]
    pub fn with_local_window<W: Into<Option<(usize, usize)>>>(mut self, local_window: W) -> Self {
        self.local_window = local_window.into();
        self
    }

    /// Sets a symmetric local-window radius.
    #[inline]
    pub fn with_symmetric_local_window<W: Into<Option<usize>>>(mut self, local_window: W) -> Self {
        self.local_window = local_window.into().map(|window| (window, window));
        self
    }

    /// Sets the requested backend implementation.
    #[inline]
    pub fn with_implementation(mut self, implementation: AttentionImplementation) -> Self {
        self.implementation = implementation;
        self
    }

    /// Requests or suppresses the log-sum-exp residual output.
    #[inline]
    pub fn with_residual(mut self, return_residual: bool) -> Self {
        self.return_residual = return_residual;
        self
    }

    /// Sets the fused-only `(rate, seed)` dropout extension.
    #[inline]
    pub fn with_dropout<D: Into<Option<(f64, u64)>>>(mut self, dropout: D) -> Self {
        self.dropout = dropout.into();
        self
    }

    /// Returns the explicit score scale, if one was configured.
    #[inline]
    pub fn scale(&self) -> Option<f64> {
        self.scale
    }

    /// Returns whether causal masking is enabled.
    #[inline]
    pub fn causal(&self) -> bool {
        self.causal
    }

    /// Returns the optional inclusive `(left, right)` local-window radii.
    #[inline]
    pub fn local_window(&self) -> Option<(usize, usize)> {
        self.local_window
    }

    /// Returns the requested backend implementation.
    #[inline]
    pub fn implementation(&self) -> AttentionImplementation {
        self.implementation
    }

    /// Returns whether the forward operation produces its log-sum-exp residual.
    #[inline]
    pub fn return_residual(&self) -> bool {
        self.return_residual
    }

    /// Returns the optional fused-only `(rate, seed)` dropout extension.
    #[inline]
    pub fn dropout(&self) -> Option<(f64, u64)> {
        self.dropout
    }
}

/// Presence metadata for the optional attention operands in their canonical order.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct AttentionOperandSignature {
    /// Whether an additive bias follows query, key, and value.
    bias: bool,

    /// Whether an arbitrary Boolean visibility mask follows the bias.
    mask: bool,

    /// Whether per-batch query sequence lengths follow the mask.
    query_sequence_lengths: bool,

    /// Whether per-batch key/value sequence lengths follow query sequence lengths.
    key_value_sequence_lengths: bool,
}

impl AttentionOperandSignature {
    /// Creates an optional-operand signature.
    #[inline]
    pub fn new(bias: bool, mask: bool, query_sequence_lengths: bool, key_value_sequence_lengths: bool) -> Self {
        Self { bias, mask, query_sequence_lengths, key_value_sequence_lengths }
    }

    /// Returns whether an additive bias is present.
    #[inline]
    pub fn has_bias(&self) -> bool {
        self.bias
    }

    /// Returns whether an arbitrary Boolean visibility mask is present.
    #[inline]
    pub fn has_mask(&self) -> bool {
        self.mask
    }

    /// Returns whether per-batch query sequence lengths are present.
    #[inline]
    pub fn has_query_sequence_lengths(&self) -> bool {
        self.query_sequence_lengths
    }

    /// Returns whether per-batch key/value sequence lengths are present.
    #[inline]
    pub fn has_key_value_sequence_lengths(&self) -> bool {
        self.key_value_sequence_lengths
    }

    /// Returns the number of optional operands described by this signature.
    #[inline]
    pub fn count(&self) -> usize {
        usize::from(self.bias)
            + usize::from(self.mask)
            + usize::from(self.query_sequence_lengths)
            + usize::from(self.key_value_sequence_lengths)
    }
}

/// Query, key, value, and optional operands supplied to scaled dot-product attention.
#[derive(Clone, Debug, PartialEq, Parameterized)]
pub struct AttentionInputs<P: Parameter> {
    /// Query array in `TNH` or `BTNH` layout.
    pub query: P,

    /// Key array in `SKH` or `BSKH` layout.
    pub key: P,

    /// Value array with the same logical shape as `key`.
    pub value: P,

    /// Optional broadcastable additive bias.
    pub bias: Option<P>,

    /// Optional broadcastable Boolean visibility mask.
    pub mask: Option<P>,

    /// Optional per-batch query sequence lengths.
    pub query_sequence_lengths: Option<P>,

    /// Optional per-batch key/value sequence lengths.
    pub key_value_sequence_lengths: Option<P>,
}

impl<P: Parameter> AttentionInputs<P> {
    /// Creates attention inputs with no optional operands.
    #[inline]
    pub fn new(query: P, key: P, value: P) -> Self {
        Self {
            query,
            key,
            value,
            bias: None,
            mask: None,
            query_sequence_lengths: None,
            key_value_sequence_lengths: None,
        }
    }

    /// Returns the optional-operand signature of these inputs.
    #[inline]
    pub fn signature(&self) -> AttentionOperandSignature {
        AttentionOperandSignature::new(
            self.bias.is_some(),
            self.mask.is_some(),
            self.query_sequence_lengths.is_some(),
            self.key_value_sequence_lengths.is_some(),
        )
    }

    /// Parses values in canonical attention operand order according to `signature`.
    ///
    /// # Parameters
    ///
    ///   - `signature`: Presence metadata for the optional operands.
    ///   - `values`: Query, key, value, and the present optional operands in canonical order.
    pub fn from_values(signature: AttentionOperandSignature, values: &[P]) -> Result<Self, TypeError>
    where
        P: Clone,
    {
        let expected_count = 3 + signature.count();
        if values.len() != expected_count {
            return Err(TypeError::invalid(format!(
                "attention input signature expects {expected_count} values but got {}",
                values.len(),
            )));
        }
        let mut values = values.iter().cloned();
        Ok(Self {
            query: values.next().unwrap(),
            key: values.next().unwrap(),
            value: values.next().unwrap(),
            bias: signature.has_bias().then(|| values.next().unwrap()),
            mask: signature.has_mask().then(|| values.next().unwrap()),
            query_sequence_lengths: signature.has_query_sequence_lengths().then(|| values.next().unwrap()),
            key_value_sequence_lengths: signature.has_key_value_sequence_lengths().then(|| values.next().unwrap()),
        })
    }

    /// Returns the values in canonical attention operand order.
    pub fn into_values(self) -> Vec<P> {
        vec![self.query, self.key, self.value]
            .into_iter()
            .chain(self.bias)
            .chain(self.mask)
            .chain(self.query_sequence_lengths)
            .chain(self.key_value_sequence_lengths)
            .collect()
    }
}

/// Scaled dot-product attention boundary. For each batch item and query head, this operation computes
///
/// ```text
/// softmax(scale * Q * transpose(K) + bias, visibility) * V.
/// ```
///
/// Query accepts `TNH` or `BTNH`; key and value accept `SKH` or `BSKH`. `T` and `S` are the query and key/value
/// sequence lengths, `N` is the query-head count, `K` is the key/value-head count, and `H` is the head dimension.
/// `K` must divide `N`; `K == N` is multi-head attention, `K == 1` is multi-query attention, and the remaining cases
/// are grouped-query attention. The attended output has the query type and shape.
///
/// [`AttentionInputs`] defines the complete operand boundary: query, key, value, then an optional broadcastable
/// additive bias, Boolean visibility mask, per-batch query lengths, and per-batch key/value lengths. Dot products,
/// scaling, bias addition, and masking use the query type promoted to at least `f32`; the masked logits are then
/// converted to `f32` for the numerically stable softmax. Visibility is the conjunction of the explicit mask, causal
/// ordering, the inclusive `(left, right)` local window, and both sequence-length predicates. Invisible scores use a
/// finite large negative value in the logits type. Query rows beyond their supplied lengths are exact zeros. When no
/// scale is configured, the operation uses `1 / sqrt(H)`.
///
/// When [`AttentionConfiguration::return_residual`] is enabled, a second `TN` or `BTN` result contains the natural-log
/// log-sum-exp statistic at the query data type. [`AttentionImplementation::Portable`] always evaluates the canonical
/// typed composition. [`AttentionImplementation::Fused`] requires a backend adapter that supports the complete
/// configuration. [`AttentionImplementation::Automatic`] may use that adapter when eligible and otherwise evaluates
/// the same portable composition. Dropout is an explicitly fused-only extension and therefore requires `Fused`.
///
/// Eager execution and portable lowering both evaluate [`dot_product_attention_ir_composition`]. The primitive itself
/// is a backend-recognizable inference boundary and is intentionally not directly differentiable; use
/// [`differentiable_dot_product_attention`] for reverse-mode differentiation. Its batching rule folds a mapped leading
/// axis into the logical batch axis while retaining this boundary for fused lowering.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DotProductAttentionOperation {
    /// Value-independent attention semantics.
    configuration: AttentionConfiguration,

    /// Presence metadata for optional operands.
    signature: AttentionOperandSignature,
}

impl DotProductAttentionOperation {
    /// Creates a new [`DotProductAttentionOperation`] from its complete semantic configuration and operand signature.
    #[inline]
    pub fn new(configuration: AttentionConfiguration, signature: AttentionOperandSignature) -> Self {
        Self { configuration, signature }
    }

    /// Returns the attention configuration.
    #[inline]
    pub fn configuration(&self) -> AttentionConfiguration {
        self.configuration
    }

    /// Returns the optional-operand signature.
    #[inline]
    pub fn signature(&self) -> AttentionOperandSignature {
        self.signature
    }
}

impl Display for DotProductAttentionOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for DotProductAttentionOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        DOT_PRODUCT_ATTENTION_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        let signature = self.signature;
        if input_types.len() != 3 + signature.count() {
            return Err(TypeError::invalid(format!(
                "'{DOT_PRODUCT_ATTENTION_OPERATION_NAME}' expects {} inputs for its optional-operand signature but \
                 got {}",
                3 + signature.count(),
                input_types.len(),
            )));
        }
        let operands = AttentionOperandTypes::forward(signature, input_types)?;
        let dimensions = validated_attention_operands(
            DOT_PRODUCT_ATTENTION_OPERATION_NAME,
            operands.query,
            operands.key,
            operands.value,
            operands.bias,
            operands.mask,
        )?;
        validated_sequence_length_operands(
            DOT_PRODUCT_ATTENTION_OPERATION_NAME,
            operands.query_sequence_lengths,
            operands.key_value_sequence_lengths,
            &dimensions.batch,
        )?;
        validated_dropout(DOT_PRODUCT_ATTENTION_OPERATION_NAME, self.configuration.dropout())?;
        if self.configuration.dropout().is_some()
            && self.configuration.implementation() != AttentionImplementation::Fused
        {
            return Err(TypeError::invalid(format!(
                "'{DOT_PRODUCT_ATTENTION_OPERATION_NAME}' dropout requires the fused implementation"
            )));
        }
        for input_type in input_types {
            if !input_type.unreduced_axes().is_empty() {
                return Err(TypeError::invalid(format!(
                    "'{DOT_PRODUCT_ATTENTION_OPERATION_NAME}' does not support unreduced operands"
                )));
            }
        }
        // The attended output is query-shaped at the query data type, so the inferred output type is the query type
        // itself, propagating operand-level metadata such as sharding.
        let mut output_types = vec![operands.query.clone()];
        if self.configuration.return_residual() {
            output_types.push(attention_activation_type(&dimensions, operands.query)?);
        }
        Ok(output_types)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, DOT_PRODUCT_ATTENTION_OPERATION_NAME)?.bracketed(|operation| {
            if let Some(scale) = self.configuration.scale() {
                operation.field("scale", &scale)?;
            }
            if self.configuration.causal() {
                operation.field("causal", &true)?;
            }
            if let Some(local_window) = self.configuration.local_window() {
                operation.field("local_window", &format_args!("({}, {})", local_window.0, local_window.1))?;
            }
            if self.configuration.implementation() != AttentionImplementation::Automatic {
                operation.field("implementation", &self.configuration.implementation())?;
            }
            if let Some((rate, seed)) = self.configuration.dropout() {
                operation.field("dropout_rate", &rate)?;
                operation.field("dropout_seed", &seed)?;
            }
            if self.configuration.return_residual() {
                operation.field("residual", &true)?;
            }
            operation.field("signature", &format_args!("{:?}", self.signature))?;
            Ok(())
        })
    }
}

/// Fused-kernel boundary for the analytical backward pass of [`DotProductAttentionOperation`].
///
/// The input prefix is the same [`AttentionInputs`] boundary used by the forward operation. It is followed by the
/// attended output, the `TN` or `BTN` log-sum-exp residual, and the incoming output cotangent. Results are cotangents
/// for query, key, value, and a present differentiable bias, in that order. Boolean masks, integer sequence lengths,
/// and a non-differentiable bias have zero-space cotangents outside this operation.
///
/// The portable composition recovers the attention probabilities as `P = exp(logits - residual)` and applies
///
/// ```text
/// dV = transpose(P) * dO
/// dS = P * (dO * transpose(V) - sum(dO * O))
/// dQ = scale * dS * K
/// dK = scale * transpose(dS) * Q.
/// ```
///
/// It uses the forward configuration and visibility predicate, reduces grouped query heads back to each key/value
/// head, reduces a broadcast bias cotangent to its original shape, and zeroes padded query rows. Portable evaluation is
/// owned by [`dot_product_attention_backward_ir_composition`]; this operation remains only so a fused forward's
/// transpose can select the matching backend backward ABI. Higher-order derivatives should differentiate the ordinary
/// portable composition instead of differentiating this boundary.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DotProductAttentionBackwardOperation {
    /// Value-independent semantics of the differentiated forward operation.
    configuration: AttentionConfiguration,

    /// Presence metadata for optional forward operands.
    signature: AttentionOperandSignature,
}

impl DotProductAttentionBackwardOperation {
    /// Creates a new backward boundary from its complete semantic configuration and operand signature.
    #[inline]
    pub fn new(configuration: AttentionConfiguration, signature: AttentionOperandSignature) -> Self {
        Self { configuration, signature }
    }

    /// Returns the attention configuration.
    #[inline]
    pub fn configuration(&self) -> AttentionConfiguration {
        self.configuration
    }

    /// Returns the optional-operand signature.
    #[inline]
    pub fn signature(&self) -> AttentionOperandSignature {
        self.signature
    }
}

impl Display for DotProductAttentionBackwardOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for DotProductAttentionBackwardOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        let signature = self.signature;
        if input_types.len() != 6 + signature.count() {
            return Err(TypeError::invalid(format!(
                "'{DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME}' expects {} inputs for its optional-operand \
                 signature but got {}",
                6 + signature.count(),
                input_types.len(),
            )));
        }
        let operands = AttentionOperandTypes::backward(signature, input_types)?;
        let dimensions = validated_attention_operands(
            DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME,
            operands.query,
            operands.key,
            operands.value,
            operands.bias,
            operands.mask,
        )?;
        validated_sequence_length_operands(
            DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME,
            operands.query_sequence_lengths,
            operands.key_value_sequence_lengths,
            &dimensions.batch,
        )?;
        validated_dropout(DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME, self.configuration.dropout())?;
        if self.configuration.dropout().is_some()
            && self.configuration.implementation() != AttentionImplementation::Fused
        {
            return Err(TypeError::invalid(format!(
                "'{DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME}' dropout requires the fused implementation"
            )));
        }
        // The forward-output-shaped operands compare by data type and shape only, so operand-level metadata such as
        // sharding never fails the structural contract.
        let matches_expected = |actual: &ArrayType, expected: &ArrayType| -> bool {
            actual.data_type() == expected.data_type() && actual.shape() == expected.shape()
        };
        if !matches_expected(operands.output.unwrap(), operands.query) {
            return Err(TypeError::invalid(format!(
                "'{DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME}' output type {} does not match the expected \
                 forward output type {}",
                operands.output.unwrap(),
                operands.query,
            )));
        }
        let expected_output_cotangent_type = operands.query.cotangent();
        if !matches_expected(operands.output_cotangent.unwrap(), &expected_output_cotangent_type) {
            return Err(TypeError::invalid(format!(
                "'{DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME}' output cotangent type {} does not match the \
                 expected cotangent type {expected_output_cotangent_type}",
                operands.output_cotangent.unwrap(),
            )));
        }
        let expected_activation_type = attention_activation_type(&dimensions, &input_types[0])?;
        if !matches_expected(operands.activation.unwrap(), &expected_activation_type) {
            return Err(TypeError::invalid(format!(
                "'{DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME}' activation type {} does not match the \
                     expected activation type {expected_activation_type}",
                operands.activation.unwrap(),
            )));
        }
        for input_type in input_types {
            if !input_type.unreduced_axes().is_empty() {
                return Err(TypeError::invalid(format!(
                    "'{DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME}' does not support unreduced operands"
                )));
            }
        }
        let mut output_types = vec![operands.query.cotangent(), operands.key.cotangent(), operands.value.cotangent()];
        if let Some(bias) = operands.bias {
            let bias_cotangent = bias.cotangent();
            if !bias_cotangent.is_zero_space() {
                output_types.push(bias_cotangent);
            }
        }
        Ok(output_types)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME)?.bracketed(
            |operation| {
                if let Some(scale) = self.configuration.scale() {
                    operation.field("scale", &scale)?;
                }
                if self.configuration.causal() {
                    operation.field("causal", &true)?;
                }
                if let Some(local_window) = self.configuration.local_window() {
                    operation.field("local_window", &format_args!("({}, {})", local_window.0, local_window.1))?;
                }
                if self.configuration.implementation() != AttentionImplementation::Automatic {
                    operation.field("implementation", &self.configuration.implementation())?;
                }
                if let Some((rate, seed)) = self.configuration.dropout() {
                    operation.field("dropout_rate", &rate)?;
                    operation.field("dropout_seed", &seed)?;
                }
                operation.field("signature", &format_args!("{:?}", self.signature))?;
                Ok(())
            },
        )
    }
}

/// Borrowed canonical view of attention operand types.
struct AttentionOperandTypes<'a> {
    /// Query type.
    query: &'a ArrayType,

    /// Key type.
    key: &'a ArrayType,

    /// Value type.
    value: &'a ArrayType,

    /// Optional bias type.
    bias: Option<&'a ArrayType>,

    /// Optional arbitrary Boolean mask type.
    mask: Option<&'a ArrayType>,

    /// Optional query-length type.
    query_sequence_lengths: Option<&'a ArrayType>,

    /// Optional key/value-length type.
    key_value_sequence_lengths: Option<&'a ArrayType>,

    /// Forward output type in a backward boundary.
    output: Option<&'a ArrayType>,

    /// Forward residual type in a backward boundary.
    activation: Option<&'a ArrayType>,

    /// Incoming output cotangent type in a backward boundary.
    output_cotangent: Option<&'a ArrayType>,
}

impl<'a> AttentionOperandTypes<'a> {
    /// Parses one forward operation boundary.
    fn forward(signature: AttentionOperandSignature, input_types: &'a [ArrayType]) -> Result<Self, TypeError> {
        let [query, key, value, optional @ ..] = input_types else {
            return Err(TypeError::invalid("attention requires query, key, and value operands"));
        };
        let mut index = 0;
        let bias = signature.has_bias().then(|| {
            let value = &optional[index];
            index += 1;
            value
        });
        let mask = signature.has_mask().then(|| {
            let value = &optional[index];
            index += 1;
            value
        });
        let query_sequence_lengths = signature.has_query_sequence_lengths().then(|| {
            let value = &optional[index];
            index += 1;
            value
        });
        let key_value_sequence_lengths = signature.has_key_value_sequence_lengths().then(|| &optional[index]);
        Ok(Self {
            query,
            key,
            value,
            bias,
            mask,
            query_sequence_lengths,
            key_value_sequence_lengths,
            output: None,
            activation: None,
            output_cotangent: None,
        })
    }

    /// Parses one backward operation boundary.
    fn backward(signature: AttentionOperandSignature, input_types: &'a [ArrayType]) -> Result<Self, TypeError> {
        let optional_count = signature.count();
        let mut operands = Self::forward(signature, &input_types[..3 + optional_count])?;
        operands.output = Some(&input_types[3 + optional_count]);
        operands.activation = Some(&input_types[4 + optional_count]);
        operands.output_cotangent = Some(&input_types[5 + optional_count]);
        Ok(operands)
    }
}

/// Returns canonical `[batch, sequence, heads, head_dimension]` dimensions for a `TNH` or `BTNH` operand.
fn attention_dimensions(
    operation_name: &str,
    descriptor: &str,
    value_type: &ArrayType,
) -> Result<[Dimension; 4], TypeError> {
    match value_type.shape().dimensions() {
        [sequence, heads, head_dimension] => {
            Ok([Dimension::Static(1), sequence.clone(), heads.clone(), head_dimension.clone()])
        }
        [batch, sequence, heads, head_dimension] => {
            Ok([batch.clone(), sequence.clone(), heads.clone(), head_dimension.clone()])
        }
        dimensions => Err(TypeError::invalid(format!(
            "'{operation_name}' {descriptor} must have rank 3 or 4 but got rank {}",
            dimensions.len(),
        ))),
    }
}

/// Validated dimensions shared by the attention operations' operand contracts.
struct AttentionDimensions {
    /// Shared batch dimension of every operand.
    batch: Dimension,

    /// Query sequence length.
    query_sequence: Dimension,

    /// Number of query heads.
    query_heads: usize,

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
    mask_type: Option<&ArrayType>,
) -> Result<AttentionDimensions, TypeError> {
    let query = attention_dimensions(operation_name, "query", query_type)?;
    let key = attention_dimensions(operation_name, "key", key_type)?;
    let value = attention_dimensions(operation_name, "value", value_type)?;
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
    let Dimension::Static(query_heads) = &query[2] else {
        return Err(TypeError::invalid(format!(
            "'{operation_name}' query heads dimension must be static but got {}",
            query[2],
        )));
    };
    let Dimension::Static(key_value_heads) = &key[2] else {
        return Err(TypeError::invalid(format!(
            "'{operation_name}' key/value heads dimension must be static but got {}",
            key[2],
        )));
    };
    let Dimension::Static(head_dimension) = &query[3] else {
        return Err(TypeError::invalid(format!(
            "'{operation_name}' head dimension must be static but got {}",
            query[3],
        )));
    };
    if *key_value_heads == 0 || *query_heads % *key_value_heads != 0 {
        return Err(TypeError::invalid(format!(
            "'{operation_name}' key/value heads dimension ({}) must divide the query heads dimension ({})",
            key_value_heads, query_heads,
        )));
    }
    for (descriptor, operand_type, expected_data_type) in
        [("bias", bias_type, None), ("mask", mask_type, Some(DataType::Boolean))]
    {
        let Some(operand_type) = operand_type else {
            continue;
        };
        let operand_dimensions = operand_type.shape().dimensions();
        if operand_dimensions.len() > 4 {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' {descriptor} must have rank at most 4 but got rank {}",
                operand_dimensions.len(),
            )));
        }
        let dimensions = std::iter::repeat_n(Dimension::Static(1), 4 - operand_dimensions.len())
            .chain(operand_dimensions.iter().cloned())
            .collect::<Vec<_>>();
        let [operand_batch, operand_heads, operand_rows, operand_columns] = dimensions.as_slice() else {
            unreachable!("attention score operands are normalized to exactly four dimensions")
        };
        if let Some(expected_data_type) = expected_data_type {
            if operand_type.data_type() != expected_data_type {
                return Err(TypeError::invalid(format!(
                    "'{operation_name}' {descriptor} must have data type {expected_data_type} but got {}",
                    operand_type.data_type(),
                )));
            }
        } else if !operand_type.data_type().is_numeric() && !operand_type.data_type().is_boolean() {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' bias must have a numeric or Boolean data type but got {}",
                operand_type.data_type(),
            )));
        }
        if operand_batch != &Dimension::Static(1) && operand_batch != &query[0] {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' {descriptor} batch dimension ({operand_batch}) must be 1 or match the query \
                 batch dimension ({})",
                query[0],
            )));
        }
        if operand_heads != &Dimension::Static(1) && operand_heads != &query[2] {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' {descriptor} heads dimension ({operand_heads}) must be 1 or match the query \
                 heads dimension ({})",
                query[2],
            )));
        }
        if operand_rows != &Dimension::Static(1) && operand_rows != &query[1] {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' {descriptor} query-sequence dimension ({operand_rows}) must be 1 or match the \
                 query sequence dimension ({})",
                query[1],
            )));
        }
        if operand_columns != &Dimension::Static(1) && operand_columns != &key[1] {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' {descriptor} key/value-sequence dimension ({operand_columns}) must be 1 or \
                 match the key sequence dimension ({})",
                key[1],
            )));
        }
    }
    Ok(AttentionDimensions {
        batch: query[0].clone(),
        query_sequence: query[1].clone(),
        query_heads: *query_heads,
        key_value_heads: *key_value_heads,
        head_dimension: *head_dimension,
        data_type,
    })
}

/// Validates the optional trailing pair of `i32[batch]` sequence-length operands shared by the attention
/// operations: each operand must be a rank-1 `i32` vector whose dimension exactly matches the shared batch dimension.
/// Refer to the documentation of [`DotProductAttentionOperation`] for the padding semantics.
fn validated_sequence_length_operands(
    operation_name: &str,
    query_lengths_type: Option<&ArrayType>,
    key_value_lengths_type: Option<&ArrayType>,
    batch: &Dimension,
) -> Result<(), TypeError> {
    for (descriptor, value_type) in
        [("query sequence lengths", query_lengths_type), ("key/value sequence lengths", key_value_lengths_type)]
    {
        let Some(value_type) = value_type else {
            continue;
        };
        if value_type.data_type() != DataType::I32 {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' {descriptor} must have data type i32 but got {}",
                value_type.data_type(),
            )));
        }
        match value_type.shape().dimensions() {
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

/// Returns the query-data-type `[batch, query_sequence, heads]` or `[query_sequence, heads]` log-sum-exp residual
/// type. The residual inherits the query's sharding with only the trailing head dimension removed.
fn attention_activation_type(dimensions: &AttentionDimensions, query_type: &ArrayType) -> Result<ArrayType, TypeError> {
    let dimensions = if query_type.rank() == 3 {
        vec![dimensions.query_sequence.clone(), Dimension::Static(dimensions.query_heads)]
    } else {
        vec![dimensions.batch.clone(), dimensions.query_sequence.clone(), Dimension::Static(dimensions.query_heads)]
    };
    let activation_type = ArrayType::new(query_type.data_type(), Shape::new(dimensions));
    let Some(query_sharding) = query_type.sharding() else {
        return Ok(activation_type);
    };
    let sharding =
        Sharding::new(query_sharding.mesh().clone(), query_sharding.dimensions()[..query_type.rank() - 1].to_vec())
            .and_then(|sharding| activation_type.with_sharding(sharding))
            .map_err(|error| TypeError::invalid(error.to_string()))?;
    Ok(sharding)
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
        let (output, residual) =
            C::Value::dot_product_attention(AttentionInputs::from_values(self.signature, inputs)?, self.configuration)?;
        Ok(std::iter::once(output).chain(residual).collect())
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
        let forward_input_count = 3 + self.signature.count();
        let (forward_inputs, suffix) = inputs.split_at(forward_input_count.min(inputs.len()));
        let [output, residual, output_cotangent] = suffix else {
            return Err(TypeError::invalid(format!(
                "'{DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME}' expects {} inputs but got {}",
                forward_input_count + 3,
                inputs.len(),
            ))
            .into());
        };
        C::Value::dot_product_attention_backward(
            AttentionInputs::from_values(self.signature, forward_inputs)?,
            output.clone(),
            residual.clone(),
            output_cotangent.clone(),
            self.configuration,
        )
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

/// Semantic role of one attention batching operand.
#[derive(Copy, Clone, Debug)]
enum AttentionBatchInput {
    /// Query, key, value, output, or output cotangent in `TNH`/`BTNH` form.
    Tensor,

    /// Broadcastable bias or Boolean mask, right-aligned to `B N T S`.
    Score,

    /// Per-batch sequence lengths.
    Length,

    /// Log-sum-exp statistic in `TN`/`BTN` form.
    Statistic,
}

/// Semantic role of one attention batching result.
#[derive(Copy, Clone, Debug)]
enum AttentionBatchOutput {
    /// Tensor whose logical rank follows the input at this index.
    Tensor(usize),

    /// Log-sum-exp statistic whose logical rank follows the query rank.
    Statistic,

    /// Bias cotangent restored to the logical bias shape at this input index.
    Score(usize),
}

/// Returns the canonical forward operand and result roles for `signature` and `configuration`.
fn attention_forward_batch_roles(
    signature: AttentionOperandSignature,
    configuration: AttentionConfiguration,
) -> (Vec<AttentionBatchInput>, Vec<AttentionBatchOutput>) {
    let mut inputs = vec![AttentionBatchInput::Tensor; 3];
    inputs.extend(signature.has_bias().then_some(AttentionBatchInput::Score));
    inputs.extend(signature.has_mask().then_some(AttentionBatchInput::Score));
    inputs.extend(signature.has_query_sequence_lengths().then_some(AttentionBatchInput::Length));
    inputs.extend(signature.has_key_value_sequence_lengths().then_some(AttentionBatchInput::Length));
    let mut outputs = vec![AttentionBatchOutput::Tensor(0)];
    outputs.extend(configuration.return_residual().then_some(AttentionBatchOutput::Statistic));
    (inputs, outputs)
}

/// Returns the canonical backward operand and result roles for `signature`.
fn attention_backward_batch_roles(
    signature: AttentionOperandSignature,
) -> (Vec<AttentionBatchInput>, Vec<AttentionBatchOutput>, Option<usize>) {
    let mut inputs = vec![AttentionBatchInput::Tensor; 3];
    let bias_index = signature.has_bias().then_some(inputs.len());
    inputs.extend(signature.has_bias().then_some(AttentionBatchInput::Score));
    inputs.extend(signature.has_mask().then_some(AttentionBatchInput::Score));
    inputs.extend(signature.has_query_sequence_lengths().then_some(AttentionBatchInput::Length));
    inputs.extend(signature.has_key_value_sequence_lengths().then_some(AttentionBatchInput::Length));
    inputs.extend([AttentionBatchInput::Tensor, AttentionBatchInput::Statistic, AttentionBatchInput::Tensor]);
    let outputs =
        vec![AttentionBatchOutput::Tensor(0), AttentionBatchOutput::Tensor(1), AttentionBatchOutput::Tensor(2)];
    (inputs, outputs, bias_index)
}

/// Static-extent normalization adapter shared by the forward and backward fused attention boundaries.
///
/// A mapped level is normalized to a leading prefix and folded into attention's batch axis. Rank-three attention has
/// an implicit logical batch of one, so `[v, T, N, H]` is already the primitive's canonical rank-four form. Rank-four
/// attention instead folds `[v, B, T, N, H]` to `[v * B, T, N, H]`. Biases and masks are first materialized to
/// `[v, B, N, T, S]`, which handles every broadcastable rank and preserves mapped score operands. Results reverse the
/// normalization, and a bias cotangent is reduced over precisely the axes broadcast by its logical operand.
fn batch_attention_static<C, O>(
    operation: &O,
    context: &BatchingContext<C, ArrayBatching<StaticArrayBatchingPolicy>>,
    inputs: &[ArrayBatch<C::Value>],
    input_roles: &[AttentionBatchInput],
    output_roles: &[AttentionBatchOutput],
) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError>
where
    C: Context<Type = ArrayType, Value: Broadcast + Reduce + Reshape + Transpose>,
    O: Operation<Type = ArrayType> + InterpretableBatchableOperation<C, ArrayBatching<StaticArrayBatchingPolicy>>,
{
    check_count!("input", input_roles, inputs.len(), ProgramError);
    let Some(axis_size) = ArrayBatch::common_batch_size(inputs)? else {
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let output_count = operation.infer_output_types(input_types.as_slice(), &[])?.len();
        let axes = vec![BatchAxis::replicated(); output_count];
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
    let aligned_inputs = inputs
        .iter()
        .map(|input| input.match_axis(0, axis_size, axis_sharding.clone()))
        .collect::<Result<Vec<_>, _>>()?;
    let query_type = inputs[0].unbatched_type();
    let query_rank = query_type.rank();
    let key_type = inputs[1].unbatched_type();
    let batch_size = if query_rank == 4 { query_type.shape()[0].value().unwrap() } else { 1 };
    let query_sequence_size = query_type.shape()[query_rank - 3].value().unwrap();
    let head_count = query_type.shape()[query_rank - 2].value().unwrap();
    let key_sequence_size = key_type.shape()[key_type.rank() - 3].value().unwrap();
    let merged_inputs = aligned_inputs
        .iter()
        .zip(input_roles)
        .map(|(aligned, role)| {
            let logical_type = aligned.unbatched_type();
            let aligned_dimensions = static_dimensions(aligned.r#type().as_ref())?;
            let value = match role {
                AttentionBatchInput::Tensor => {
                    if logical_type.rank() == 4 {
                        aligned.value().reshape(static_shape(
                            std::iter::once(axis_size * aligned_dimensions[1])
                                .chain(aligned_dimensions[2..].iter().copied())
                                .collect::<Vec<_>>()
                                .as_slice(),
                        ))?
                    } else {
                        aligned.value().clone()
                    }
                }
                AttentionBatchInput::Score => {
                    let target_dimensions = [axis_size, batch_size, head_count, query_sequence_size, key_sequence_size];
                    let rank = logical_type.rank();
                    let output_axes =
                        std::iter::once(0).chain((0..rank).map(|axis| 5 - rank + axis)).collect::<Vec<_>>();
                    aligned
                        .value()
                        .clone()
                        .broadcast(
                            ArrayType::new(logical_type.data_type(), static_shape(&target_dimensions)),
                            output_axes.as_slice(),
                        )?
                        .reshape(static_shape(&[
                            axis_size * batch_size,
                            head_count,
                            query_sequence_size,
                            key_sequence_size,
                        ]))?
                }
                AttentionBatchInput::Length => aligned.value().reshape(static_shape(&[axis_size * batch_size]))?,
                AttentionBatchInput::Statistic => {
                    if query_rank == 4 {
                        aligned.value().reshape(static_shape(&[
                            axis_size * aligned_dimensions[1],
                            aligned_dimensions[2],
                            aligned_dimensions[3],
                        ]))?
                    } else {
                        aligned.value().clone()
                    }
                }
            };
            Ok(ArrayBatch::replicated(value))
        })
        .collect::<Result<Vec<_>, BatchingError>>()?;
    let merged_types = merged_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
    let output_count = operation.infer_output_types(merged_types.as_slice(), &[])?.len();
    check_count!("output", output_roles, output_count, ProgramError);
    let axes = vec![BatchAxis::replicated(); output_count];
    let outputs = operation.interpret_with_batch_axes(context, merged_inputs.as_slice(), axes.as_slice())?;
    outputs
        .into_iter()
        .zip(output_roles)
        .map(|(output, role)| {
            let output_dimensions = static_dimensions(&output.r#type())?;
            let value = match role {
                AttentionBatchOutput::Tensor(input_index) => {
                    let logical_type = inputs[*input_index].unbatched_type();
                    if logical_type.rank() == 4 {
                        output.value().reshape(static_shape(
                            std::iter::once(axis_size)
                                .chain(std::iter::once(logical_type.shape()[0].value().unwrap()))
                                .chain(output_dimensions[1..].iter().copied())
                                .collect::<Vec<_>>()
                                .as_slice(),
                        ))?
                    } else {
                        output.value().clone()
                    }
                }
                AttentionBatchOutput::Statistic => {
                    if query_rank == 4 {
                        output.value().reshape(static_shape(&[
                            axis_size,
                            query_type.shape()[0].value().unwrap(),
                            output_dimensions[1],
                            output_dimensions[2],
                        ]))?
                    } else {
                        output.value().clone()
                    }
                }
                AttentionBatchOutput::Score(input_index) => {
                    let logical_type = inputs[*input_index].unbatched_type();
                    let normalized = output.value().reshape(static_shape(&[
                        axis_size,
                        batch_size,
                        head_count,
                        query_sequence_size,
                        key_sequence_size,
                    ]))?;
                    let offset = 4 - logical_type.rank();
                    let reduction_axes = (0..4)
                        .filter(|&axis| {
                            axis < offset || matches!(logical_type.shape()[axis - offset], Dimension::Static(1))
                        })
                        .map(|axis| axis + 1)
                        .collect::<Vec<_>>();
                    let target = inputs[*input_index].match_axis(0, axis_size, axis_sharding.clone())?;
                    normalized
                        .reduce(reduction_axes.as_slice(), ReductionKind::Sum)
                        .reshape(target.r#type().shape().clone())?
                }
            };
            ArrayBatch::new(value, BatchAxis::new(0))
        })
        .collect()
}

/// Binds a dynamic reshape in the enclosing array IR context.
fn reshape_attention_array<C>(
    context: &C,
    value: C::Value,
    dimensions: Vec<C::Value>,
) -> Result<C::Value, BatchingError>
where
    C: Context<Type = ArrayIrType, Operation: From<DynamicReshapeOperation>>,
{
    let inputs = std::iter::once(value).chain(dimensions).collect::<Vec<_>>();
    Ok(context.bind(DynamicReshapeOperation::new(), Vec::new(), inputs.as_slice())?.remove(0))
}

/// Multiplies two first-class dimensions in the enclosing array IR context.
fn multiply_attention_dimensions<C>(context: &C, left: &C::Value, right: &C::Value) -> Result<C::Value, BatchingError>
where
    C: Context<Type = ArrayIrType, Operation: From<DimensionMulOperation>>,
{
    let left_type = left.r#type();
    let left_type = <&DimensionType>::try_from(left_type.as_ref())?;
    let right_type = right.r#type();
    let right_type = <&DimensionType>::try_from(right_type.as_ref())?;
    Ok(context
        .bind(
            DimensionMulOperation::new(left_type, right_type).map_err(ProgramError::from)?,
            Vec::new(),
            &[left.clone(), right.clone()],
        )?
        .remove(0))
}

/// First-class-extent normalization adapter shared by the forward and backward fused attention boundaries.
///
/// This is the dynamic counterpart of [`batch_attention_static`]. It stages the same prefix normalization using
/// mixed `broadcast` and `reshape` operations whose result dimensions are ordinary SSA operands. Consequently a
/// dynamic mapped extent, logical batch, or sequence length never becomes host metadata or a specialization key.
fn batch_attention_dynamic<C, O>(
    operation: &O,
    context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatching<DynamicArrayBatchingPolicy>>,
    inputs: &[ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>],
    input_roles: &[AttentionBatchInput],
    output_roles: &[AttentionBatchOutput],
) -> Result<Vec<ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>>, BatchingError>
where
    C: Context<
            Type = ArrayIrType,
            Operation: From<ConstantOperation<DimensionValue>>
                           + From<DimensionMulOperation>
                           + From<DimensionSizeOperation>
                           + From<DynamicBroadcastOperation>
                           + From<DynamicReshapeOperation>
                           + OperationProjection<ArrayType>,
        >,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>,
    <C::Operation as OperationProjection<ArrayType>>::Projected: From<O> + From<ReduceOperation>,
    O: Operation<Type = ArrayType> + Clone,
{
    check_count!("input", input_roles, inputs.len(), ProgramError);
    let outer_context = context.parent().parent();
    if inputs.iter().all(|input| input.batch_axis().is_replicated()) {
        let values = inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
        return context
            .parent()
            .bind(operation.clone(), Vec::new(), values.as_slice())?
            .into_iter()
            .map(|output| Ok(ArrayBatch::replicated(output)))
            .collect();
    }

    let aligned_inputs = inputs
        .iter()
        .map(|input| DynamicArrayBatchingPolicy::match_axis(context, input, 0.into()))
        .collect::<Result<Vec<_>, _>>()?;
    let query_type = inputs[0].unbatched_type();
    let query_rank = query_type.rank();
    let key_type = inputs[1].unbatched_type();
    let query = C::Value::from_projected(aligned_inputs[0].value().clone());
    let key = C::Value::from_projected(aligned_inputs[1].value().clone());
    let mapped_extent = context.axis_extent().clone();
    let batch_extent = if query_rank == 4 {
        folded_array_dimension(outer_context, &query, 1)?
    } else {
        dimension_constant(outer_context, 1)?
    };
    let query_sequence_extent = folded_array_dimension(outer_context, &query, 1 + query_rank - 3)?;
    let head_extent = folded_array_dimension(outer_context, &query, 1 + query_rank - 2)?;
    let key_sequence_extent = folded_array_dimension(outer_context, &key, 1 + key_type.rank() - 3)?;
    let merged_batch_extent = multiply_attention_dimensions(outer_context, &mapped_extent, &batch_extent)?;

    let merged_values = aligned_inputs
        .iter()
        .zip(input_roles)
        .map(|(aligned, role)| {
            let logical_type = aligned.unbatched_type();
            let value = C::Value::from_projected(aligned.value().clone());
            match role {
                AttentionBatchInput::Tensor => {
                    if logical_type.rank() == 4 {
                        let dimensions = std::iter::once(merged_batch_extent.clone())
                            .chain(
                                (2..aligned.r#type().rank())
                                    .map(|axis| folded_array_dimension(outer_context, &value, axis))
                                    .collect::<Result<Vec<_>, _>>()?,
                            )
                            .collect();
                        reshape_attention_array(outer_context, value, dimensions)
                    } else {
                        Ok(value)
                    }
                }
                AttentionBatchInput::Score => {
                    let rank = logical_type.rank();
                    let output_axes =
                        std::iter::once(0).chain((0..rank).map(|axis| 5 - rank + axis)).collect::<Vec<_>>();
                    let normalized = broadcast_array(
                        outer_context,
                        value,
                        vec![
                            mapped_extent.clone(),
                            batch_extent.clone(),
                            head_extent.clone(),
                            query_sequence_extent.clone(),
                            key_sequence_extent.clone(),
                        ],
                        output_axes,
                        None,
                    )?;
                    reshape_attention_array(
                        outer_context,
                        normalized,
                        vec![
                            merged_batch_extent.clone(),
                            head_extent.clone(),
                            query_sequence_extent.clone(),
                            key_sequence_extent.clone(),
                        ],
                    )
                }
                AttentionBatchInput::Length => {
                    reshape_attention_array(outer_context, value, vec![merged_batch_extent.clone()])
                }
                AttentionBatchInput::Statistic => {
                    if query_rank == 4 {
                        let dimensions = std::iter::once(merged_batch_extent.clone())
                            .chain(
                                (2..aligned.r#type().rank())
                                    .map(|axis| folded_array_dimension(outer_context, &value, axis))
                                    .collect::<Result<Vec<_>, _>>()?,
                            )
                            .collect();
                        reshape_attention_array(outer_context, value, dimensions)
                    } else {
                        Ok(value)
                    }
                }
            }
        })
        .collect::<Result<Vec<_>, BatchingError>>()?;
    let merged_values = merged_values.into_iter().map(C::Value::into_projected).collect::<Result<Vec<_>, _>>()?;
    let mut outputs = context.parent().bind(operation.clone(), Vec::new(), merged_values.as_slice())?;
    check_count!("output", output_roles, outputs.len(), ProgramError);
    outputs
        .drain(..)
        .zip(output_roles)
        .map(|(output, role)| {
            let output = C::Value::from_projected(output);
            let value = match role {
                AttentionBatchOutput::Tensor(input_index) => {
                    if inputs[*input_index].unbatched_type().rank() == 4 {
                        let aligned = &aligned_inputs[*input_index];
                        let aligned_value = C::Value::from_projected(aligned.value().clone());
                        let dimensions = (0..aligned.r#type().rank())
                            .map(|axis| folded_array_dimension(outer_context, &aligned_value, axis))
                            .collect::<Result<Vec<_>, _>>()?;
                        reshape_attention_array(outer_context, output, dimensions)?
                    } else {
                        output
                    }
                }
                AttentionBatchOutput::Statistic => {
                    if query_rank == 4 {
                        let dimensions = (0..aligned_inputs[0].r#type().rank() - 1)
                            .map(|axis| folded_array_dimension(outer_context, &query, axis))
                            .collect::<Result<Vec<_>, _>>()?;
                        reshape_attention_array(outer_context, output, dimensions)?
                    } else {
                        output
                    }
                }
                AttentionBatchOutput::Score(input_index) => {
                    let normalized = reshape_attention_array(
                        outer_context,
                        output,
                        vec![
                            mapped_extent.clone(),
                            batch_extent.clone(),
                            head_extent.clone(),
                            query_sequence_extent.clone(),
                            key_sequence_extent.clone(),
                        ],
                    )?;
                    let logical_type = inputs[*input_index].unbatched_type();
                    let offset = 4 - logical_type.rank();
                    let reduction_axes = (0..4)
                        .filter(|&axis| {
                            axis < offset || matches!(logical_type.shape()[axis - offset], Dimension::Static(1))
                        })
                        .map(|axis| axis + 1)
                        .collect::<Vec<_>>();
                    let reduced = if reduction_axes.is_empty() {
                        normalized
                    } else {
                        let normalized = C::Value::into_projected(normalized)?;
                        let reduced = context
                            .parent()
                            .bind(ReduceOperation::new(reduction_axes, ReductionKind::Sum), Vec::new(), &[normalized])?
                            .remove(0);
                        C::Value::from_projected(reduced)
                    };
                    let aligned = &aligned_inputs[*input_index];
                    let aligned_value = C::Value::from_projected(aligned.value().clone());
                    let dimensions = (0..aligned.r#type().rank())
                        .map(|axis| folded_array_dimension(outer_context, &aligned_value, axis))
                        .collect::<Result<Vec<_>, _>>()?;
                    reshape_attention_array(outer_context, reduced, dimensions)?
                }
            };
            ArrayBatch::new(C::Value::into_projected(value)?, BatchAxis::new(0))
        })
        .collect()
}

/// Batching rule for [`DotProductAttentionOperation`]: one mapped batch level folds into the operation's own batch
/// dimension through the shared static-extent normalization adapter.
impl<C: Context<Type = ArrayType, Value: Broadcast + Reduce + Reshape + Transpose>>
    BatchableOperation<C, ArrayBatching<StaticArrayBatchingPolicy>> for DotProductAttentionOperation
where
    DotProductAttentionOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<StaticArrayBatchingPolicy>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<StaticArrayBatchingPolicy>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatching<StaticArrayBatchingPolicy>>, BatchingError> {
        let (input_roles, output_roles) = attention_forward_batch_roles(self.signature, self.configuration);
        Ok(batch_attention_static(self, context, inputs, input_roles.as_slice(), output_roles.as_slice())?.into())
    }
}

/// Batching rule for [`DotProductAttentionBackwardOperation`]: the same static-extent normalization as the forward
/// operation, additionally restoring a broadcast bias-cotangent batch dimension.
impl<C: Context<Type = ArrayType, Value: Broadcast + Reduce + Reshape + Transpose>>
    BatchableOperation<C, ArrayBatching<StaticArrayBatchingPolicy>> for DotProductAttentionBackwardOperation
where
    DotProductAttentionBackwardOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<StaticArrayBatchingPolicy>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<StaticArrayBatchingPolicy>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatching<StaticArrayBatchingPolicy>>, BatchingError> {
        let (input_roles, mut output_roles, bias_index) = attention_backward_batch_roles(self.signature);
        output_roles.extend(
            bias_index
                .filter(|&index| !inputs[index].unbatched_type().cotangent().is_zero_space())
                .map(AttentionBatchOutput::Score),
        );
        Ok(batch_attention_static(self, context, inputs, input_roles.as_slice(), output_roles.as_slice())?.into())
    }
}

/// First-class-extent batching rule for [`DotProductAttentionOperation`].
impl<C> BatchableOperation<ProjectedContext<C, ArrayType>, ArrayBatching<DynamicArrayBatchingPolicy>>
    for DotProductAttentionOperation
where
    C: Context<
            Type = ArrayIrType,
            Operation: From<ConstantOperation<DimensionValue>>
                           + From<DimensionMulOperation>
                           + From<DimensionSizeOperation>
                           + From<DynamicBroadcastOperation>
                           + From<DynamicReshapeOperation>
                           + OperationProjection<ArrayType>,
        >,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>,
    <C::Operation as OperationProjection<ArrayType>>::Projected:
        From<DotProductAttentionOperation> + From<ReduceOperation>,
{
    fn batch<D: BatchingDriver<ProjectedContext<C, ArrayType>, ArrayBatching<DynamicArrayBatchingPolicy>>>(
        &self,
        context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatching<DynamicArrayBatchingPolicy>>,
        _driver: &D,
        inputs: &[ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>],
    ) -> Result<BatchedOutputs<ProjectedContext<C, ArrayType>, ArrayBatching<DynamicArrayBatchingPolicy>>, BatchingError>
    {
        let (input_roles, output_roles) = attention_forward_batch_roles(self.signature, self.configuration);
        Ok(batch_attention_dynamic(self, context, inputs, input_roles.as_slice(), output_roles.as_slice())?.into())
    }
}

/// First-class-extent batching rule for [`DotProductAttentionBackwardOperation`].
impl<C> BatchableOperation<ProjectedContext<C, ArrayType>, ArrayBatching<DynamicArrayBatchingPolicy>>
    for DotProductAttentionBackwardOperation
where
    C: Context<
            Type = ArrayIrType,
            Operation: From<ConstantOperation<DimensionValue>>
                           + From<DimensionMulOperation>
                           + From<DimensionSizeOperation>
                           + From<DynamicBroadcastOperation>
                           + From<DynamicReshapeOperation>
                           + OperationProjection<ArrayType>,
        >,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>,
    <C::Operation as OperationProjection<ArrayType>>::Projected:
        From<DotProductAttentionBackwardOperation> + From<ReduceOperation>,
{
    fn batch<D: BatchingDriver<ProjectedContext<C, ArrayType>, ArrayBatching<DynamicArrayBatchingPolicy>>>(
        &self,
        context: &BatchingContext<ProjectedContext<C, ArrayType>, ArrayBatching<DynamicArrayBatchingPolicy>>,
        _driver: &D,
        inputs: &[ArrayBatch<<C::Value as ValueProjection<ArrayType>>::Projected>],
    ) -> Result<BatchedOutputs<ProjectedContext<C, ArrayType>, ArrayBatching<DynamicArrayBatchingPolicy>>, BatchingError>
    {
        let (input_roles, mut output_roles, bias_index) = attention_backward_batch_roles(self.signature);
        output_roles.extend(
            bias_index
                .filter(|&index| !inputs[index].unbatched_type().cotangent().is_zero_space())
                .map(AttentionBatchOutput::Score),
        );
        Ok(batch_attention_dynamic(self, context, inputs, input_roles.as_slice(), output_roles.as_slice())?.into())
    }
}

/// Value-level scaled dot-product attention capability. Refer to the documentation of
/// [`DotProductAttentionOperation`] for the `BTNH` operand convention, the exact semantics, and the transform rules.
pub trait DotProductAttention: Parameter + Sized {
    /// Computes attention from its canonical input structure and semantic configuration.
    ///
    /// # Parameters
    ///
    ///   - `inputs`: Query, key, value, and any optional array operands in their canonical structure.
    ///   - `configuration`: Value-independent attention semantics and implementation selection.
    fn dot_product_attention(
        inputs: AttentionInputs<Self>,
        configuration: AttentionConfiguration,
    ) -> Result<(Self, Option<Self>), ProgramError>;
}

/// Value-level backward (gradient) pass of scaled dot-product attention. Refer to the documentation of
/// [`DotProductAttentionBackwardOperation`] for the operand convention and the exact semantics.
pub(crate) trait DotProductAttentionBackward: Parameter + Sized {
    /// Computes cotangents for the differentiable attention operands.
    ///
    /// # Parameters
    ///
    ///   - `inputs`: Query, key, value, and optional operands from the forward pass.
    ///   - `output`: Attended output produced by the forward pass.
    ///   - `residual`: Log-sum-exp statistic produced by the forward pass.
    ///   - `output_cotangent`: Incoming cotangent of the forward output.
    fn dot_product_attention_backward(
        inputs: AttentionInputs<Self>,
        output: Self,
        residual: Self,
        output_cotangent: Self,
        configuration: AttentionConfiguration,
    ) -> Result<Vec<Self>, ProgramError>;
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
        inputs: AttentionInputs<Self>,
        configuration: AttentionConfiguration,
    ) -> Result<(Self, Option<Self>), ProgramError> {
        let signature = inputs.signature();
        let context = inputs.query.dispatch_domain();
        let operands = inputs.into_values();
        let mut outputs = context.bind(
            DotProductAttentionOperation::new(configuration, signature),
            Vec::new(),
            operands.as_slice(),
        )?;
        let expected_output_count = if configuration.return_residual() { 2 } else { 1 };
        check_count!("output", outputs, expected_output_count, ProgramError);
        let residual = configuration.return_residual().then(|| outputs.remove(1));
        Ok((outputs.remove(0), residual))
    }
}

/// Any context-carrying value computes the attention backward pass by binding a
/// [`DotProductAttentionBackwardOperation`] through its own context; refer to the [`DotProductAttention`] blanket
/// implementation for the disjointness argument.
impl<V: Value<Type = ArrayType>> DotProductAttentionBackward for V
where
    V::DispatchDomain: Context<Operation: From<DotProductAttentionBackwardOperation>>,
{
    fn dot_product_attention_backward(
        inputs: AttentionInputs<Self>,
        output: Self,
        residual: Self,
        output_cotangent: Self,
        configuration: AttentionConfiguration,
    ) -> Result<Vec<Self>, ProgramError> {
        let signature = inputs.signature();
        let context = inputs.query.dispatch_domain();
        let mut operands = inputs.into_values();
        operands.extend([output, residual, output_cotangent]);
        context.bind(
            DotProductAttentionBackwardOperation::new(configuration, signature),
            Vec::new(),
            operands.as_slice(),
        )
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayIrOperation, ArrayIrValue, DimensionBounds, DimensionVariable};
    use crate::batching::batch;
    use crate::contexts::StagingContext;
    use crate::parameters::Placeholder;
    use crate::tracing::TracingContext;

    use super::*;

    fn array_type(data_type: DataType, dimensions: &[usize]) -> ArrayType {
        ArrayType::new(data_type, static_shape(dimensions))
    }

    #[test]
    fn test_attention_inputs() {
        let signature = AttentionOperandSignature::new(true, true, true, true);
        let values = (0..7).map(|size| array_type(DataType::F32, &[size])).collect::<Vec<_>>();
        let inputs = AttentionInputs::from_values(signature, values.as_slice()).unwrap();

        assert_eq!(inputs.signature(), signature);
        assert_eq!(inputs.into_values(), values);
        assert!(matches!(
            AttentionInputs::from_values(signature, &values[..6]),
            Err(TypeError::Invalid { message, .. })
                if message == "attention input signature expects 7 values but got 6",
        ));
    }

    #[test]
    fn test_attention_configuration() {
        let configuration = AttentionConfiguration::new()
            .with_scale(0.25)
            .with_causal(true)
            .with_local_window((2, 1))
            .with_implementation(AttentionImplementation::Fused)
            .with_residual(true)
            .with_dropout((0.1, 7));

        assert_eq!(configuration.scale(), Some(0.25));
        assert!(configuration.causal());
        assert_eq!(configuration.local_window(), Some((2, 1)));
        assert_eq!(configuration.implementation(), AttentionImplementation::Fused);
        assert!(configuration.return_residual());
        assert_eq!(configuration.dropout(), Some((0.1, 7)));
        assert_eq!(AttentionConfiguration::new().with_symmetric_local_window(3).local_window(), Some((3, 3)));
    }

    #[test]
    fn test_dot_product_attention_type_inference() {
        let query = array_type(DataType::F32, &[2, 2, 2]);
        let key_value = array_type(DataType::F32, &[3, 1, 2]);
        let bias = array_type(DataType::F64, &[3]);
        let mask = array_type(DataType::Boolean, &[2, 3]);
        let lengths = array_type(DataType::I32, &[1]);
        let residual = array_type(DataType::F32, &[2, 2]);
        let signature = AttentionOperandSignature::new(true, true, true, true);
        let configuration = AttentionConfiguration::new().with_local_window((0, 0)).with_residual(true);
        let operation = DotProductAttentionOperation::new(configuration, signature);
        let input_types =
            vec![query.clone(), key_value.clone(), key_value.clone(), bias.clone(), mask, lengths.clone(), lengths];

        assert_eq!(
            operation.infer_output_types(input_types.as_slice(), &[]),
            Ok(vec![query.clone(), residual.clone()])
        );
        let mut backward_input_types = input_types;
        backward_input_types.extend([query.clone(), residual, query.clone()]);
        assert_eq!(
            DotProductAttentionBackwardOperation::new(configuration, signature)
                .infer_output_types(backward_input_types.as_slice(), &[]),
            Ok(vec![query, key_value.clone(), key_value, bias]),
        );

        // A non-differentiable bias remains a legal forward input but contributes no live backward result.
        let query = array_type(DataType::F32, &[2, 2, 2]);
        let key_value = array_type(DataType::F32, &[3, 1, 2]);
        let bias = array_type(DataType::I32, &[3]);
        let residual = array_type(DataType::F32, &[2, 2]);
        let structural_bias_signature = AttentionOperandSignature::new(true, false, false, false);
        let mut backward_input_types = vec![query.clone(), key_value.clone(), key_value.clone(), bias];
        backward_input_types.extend([query.clone(), residual, query.clone()]);
        assert_eq!(
            DotProductAttentionBackwardOperation::new(configuration, structural_bias_signature)
                .infer_output_types(backward_input_types.as_slice(), &[]),
            Ok(vec![query, key_value.clone(), key_value]),
        );

        let dropout = AttentionConfiguration::new().with_dropout((0.25, 7));
        assert!(matches!(
            DotProductAttentionOperation::new(dropout, AttentionOperandSignature::default())
                .infer_output_types(&backward_input_types[..3], &[]),
            Err(TypeError::Invalid { message, .. })
                if message == "'dot_product_attention' dropout requires the fused implementation",
        ));
    }

    #[test]
    fn test_dot_product_attention() {
        // Rank-three operands normalize through an implicit batch. The arbitrary mask, asymmetric local window, and
        // query lengths compose independently, while the omitted scale defaults to `1 / sqrt(head_dimension)`.
        let query = Array::from_f64s(array_type(DataType::F32, &[2, 1, 2]), vec![1.0, 0.0, 0.0, 1.0]);
        let key = Array::from_f64s(array_type(DataType::F32, &[3, 1, 2]), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let value = Array::from_f64s(array_type(DataType::F32, &[3, 1, 2]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mask =
            Array::from_elements(array_type(DataType::Boolean, &[2, 3]), &[true, false, true, true, true, false])
                .unwrap();
        let bias = Array::from_elements(array_type(DataType::I32, &[]), &[0_i32]).unwrap();
        let query_lengths = Array::from_elements(array_type(DataType::I32, &[1]), &[1_i32]).unwrap();
        let inputs = AttentionInputs {
            query,
            key,
            value,
            bias: Some(bias),
            mask: Some(mask),
            query_sequence_lengths: Some(query_lengths),
            key_value_sequence_lengths: None,
        };
        let configuration = AttentionConfiguration::new().with_local_window((1, 0)).with_residual(true);
        let (output, residual) = Array::dot_product_attention(inputs.clone(), configuration).unwrap();
        let (explicit, _) =
            Array::dot_product_attention(inputs, configuration.with_scale(Some(1.0 / 2.0_f64.sqrt()))).unwrap();

        assert_eq!(output.r#type().shape(), &static_shape(&[2, 1, 2]));
        let residual = residual.unwrap();
        assert_eq!(residual.r#type().shape(), &static_shape(&[2, 1]));
        assert_eq!(output.to_f64s(), explicit.to_f64s());
        assert_eq!(&output.to_f64s()[2..], &[0.0, 0.0]);
        assert!(residual.to_f64s()[1] < -1.0e30);

        // Float64 dot products, scaling, and bias addition retain float64 precision until the logits reach the
        // explicitly float32 softmax. Subtracting the large bias before that conversion preserves the unit gap.
        let query = Array::from_f64s(array_type(DataType::F64, &[1, 1, 1]), vec![1.0e8]);
        let key = Array::from_f64s(array_type(DataType::F64, &[2, 1, 1]), vec![1.0, 1.0 + 1.0e-8]);
        let value = Array::from_f64s(array_type(DataType::F64, &[2, 1, 1]), vec![0.0, 1.0]);
        let bias = Array::from_f64s(array_type(DataType::F64, &[2]), vec![-1.0e8, -1.0e8]);
        let output = Array::dot_product_attention(
            AttentionInputs { bias: Some(bias), ..AttentionInputs::new(query, key, value) },
            AttentionConfiguration::new().with_scale(1.0),
        )
        .unwrap()
        .0;

        assert!((output.to_f64s()[0] - 0.731_058_6).abs() < 1.0e-6);

        // MQA and its explicitly repeated MHA representation are semantically identical.
        let query = Array::from_f64s(array_type(DataType::F32, &[1, 1, 2, 1]), vec![1.0, 1.0]);
        let grouped_key = Array::from_f64s(array_type(DataType::F32, &[1, 2, 1, 1]), vec![1.0, 2.0]);
        let grouped_value = Array::from_f64s(array_type(DataType::F32, &[1, 2, 1, 1]), vec![10.0, 20.0]);
        let repeated_key = Array::from_f64s(array_type(DataType::F32, &[1, 2, 2, 1]), vec![1.0, 1.0, 2.0, 2.0]);
        let repeated_value = Array::from_f64s(array_type(DataType::F32, &[1, 2, 2, 1]), vec![10.0, 10.0, 20.0, 20.0]);
        let grouped = Array::dot_product_attention(
            AttentionInputs::new(query.clone(), grouped_key, grouped_value),
            AttentionConfiguration::new(),
        )
        .unwrap()
        .0;
        let repeated = Array::dot_product_attention(
            AttentionInputs::new(query, repeated_key, repeated_value),
            AttentionConfiguration::new(),
        )
        .unwrap()
        .0;
        grouped
            .to_f64s()
            .iter()
            .zip(repeated.to_f64s())
            .for_each(|(actual, expected)| assert_abs_diff_eq!(actual, &expected, epsilon = 1e-6));
    }

    #[test]
    fn test_dot_product_attention_rendering() {
        let operation = DotProductAttentionOperation::new(
            AttentionConfiguration::new()
                .with_scale(0.125)
                .with_causal(true)
                .with_local_window((2, 0))
                .with_implementation(AttentionImplementation::Portable)
                .with_residual(true),
            AttentionOperandSignature::new(true, false, true, true),
        );

        assert_eq!(
            operation.to_string(),
            indoc! {"
                dot_product_attention [
                    scale=0.125,
                    causal=true,
                    local_window=(2, 0),
                    implementation=portable,
                    residual=true,
                    signature=AttentionOperandSignature { bias: true, mask: false, query_sequence_lengths: true, key_value_sequence_lengths: true },
                ]"},
        );
    }

    #[test]
    fn test_dot_product_attention_batching() {
        // Each mapped example carries one complete rank-four attention problem. The boundary folds that mapped axis
        // into its logical batch axis and restores it on the output.
        let r#type = array_type(DataType::F32, &[2, 1, 1, 1, 1]);
        let query = ArrayBatch::new(Array::from_f64s(r#type.clone(), vec![1.0, 1.0]), BatchAxis::new(0)).unwrap();
        let key = ArrayBatch::new(Array::from_f64s(r#type.clone(), vec![1.0, 1.0]), BatchAxis::new(0)).unwrap();
        let value = ArrayBatch::new(Array::from_f64s(r#type, vec![5.0, 7.0]), BatchAxis::new(0)).unwrap();

        let outputs =
            DotProductAttentionOperation::new(AttentionConfiguration::new(), AttentionOperandSignature::default())
                .batch(
                    &BatchingContext::new(crate::contexts::EagerContext::<Array>::new(), 2),
                    &crate::EmptyRegionDriver,
                    &[query, key, value],
                )
                .unwrap()
                .into_parts()
                .0;

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().to_f64s(), vec![5.0, 7.0]);

        // Rank-three attention has an implicit logical batch of one. A mapped rank-two mask is normalized alongside
        // the operands rather than being mistaken for a tensor batch prefix.
        let query = ArrayBatch::new(
            Array::from_f64s(array_type(DataType::F32, &[2, 1, 1, 1]), vec![1.0, 1.0]),
            BatchAxis::new(0),
        )
        .unwrap();
        let key = ArrayBatch::new(
            Array::from_f64s(array_type(DataType::F32, &[2, 2, 1, 1]), vec![1.0, 2.0, 1.0, 2.0]),
            BatchAxis::new(0),
        )
        .unwrap();
        let value = ArrayBatch::new(
            Array::from_f64s(array_type(DataType::F32, &[2, 2, 1, 1]), vec![3.0, 9.0, 4.0, 10.0]),
            BatchAxis::new(0),
        )
        .unwrap();
        let mask = ArrayBatch::new(
            Array::from_elements(array_type(DataType::Boolean, &[2, 1, 2]), &[true, false, false, true]).unwrap(),
            BatchAxis::new(0),
        )
        .unwrap();
        let outputs = DotProductAttentionOperation::new(
            AttentionConfiguration::new().with_residual(true),
            AttentionOperandSignature::new(false, true, false, false),
        )
        .batch(
            &BatchingContext::new(crate::contexts::EagerContext::<Array>::new(), 2),
            &crate::EmptyRegionDriver,
            &[query, key, value, mask],
        )
        .unwrap()
        .into_parts()
        .0;

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].unbatched_type().shape(), &static_shape(&[1, 1, 1]));
        assert_eq!(outputs[0].value().to_f64s(), vec![3.0, 10.0]);
        assert_eq!(outputs[1].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[1].unbatched_type().shape(), &static_shape(&[1, 1]));
    }

    #[test]
    fn test_dot_product_attention_batching_with_dynamic_extent() -> Result<(), ProgramError> {
        type TraceContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let trace = TraceContext::new();
        let batch_variable = DimensionVariable::new("batch", DimensionBounds::new(1, Some(5))?);
        let r#type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![
                Dimension::Dynamic(batch_variable),
                Dimension::Static(1),
                Dimension::Static(1),
                Dimension::Static(1),
                Dimension::Static(1),
            ]),
        );
        let query = trace.input(r#type.clone().into());
        let key = trace.input(r#type.clone().into());
        let value = trace.input(r#type.clone().into());
        let output = batch(
            |(query, key, value)| {
                let query = ValueProjection::<ArrayType>::into_projected(query)?;
                let key = ValueProjection::<ArrayType>::into_projected(key)?;
                let value = ValueProjection::<ArrayType>::into_projected(value)?;
                let (output, _) = <_ as DotProductAttention>::dot_product_attention(
                    AttentionInputs::new(query.clone(), key, value),
                    AttentionConfiguration::new(),
                )?;
                Ok(output.into_value())
            },
            (query, key, value),
            (BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        )?;
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, ArrayIrValue<Array>>(
            vec![output.atom_id()?],
            vec![Placeholder, Placeholder, Placeholder],
            Placeholder,
        )?;

        assert_eq!(output.r#type().as_ref(), &ArrayIrType::Array(r#type));
        let operation_counts = program.statistics().total_operation_counts();
        assert_eq!(operation_counts.get("dimension_mul"), Some(&1));
        assert_eq!(operation_counts.get(DOT_PRODUCT_ATTENTION_OPERATION_NAME), Some(&1));
        assert_eq!(operation_counts.get("reshape"), Some(&4));

        Ok(())
    }
}
