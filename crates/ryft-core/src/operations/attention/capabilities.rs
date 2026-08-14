use super::*;

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
