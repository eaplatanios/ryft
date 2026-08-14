use super::*;

/// Canonical operation name for [`DotProductAttentionOperation`].
pub const DOT_PRODUCT_ATTENTION_OPERATION_NAME: &str = "dot_product_attention";

/// Canonical operation name for [`DotProductAttentionBackwardOperation`].
pub const DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME: &str = "dot_product_attention_backward";

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
