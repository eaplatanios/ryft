//! Reference [`Array`] kernels for the attention operation family contracts.
//!
//! Scaled dot-product attention has no dedicated reference kernel: both the forward and the backward contracts are
//! answered by evaluating the shared universe-neutral compositions eagerly over concrete arrays.

use crate::arrays::arrays::Array;
use crate::arrays::ir::ArrayIrValue;
use crate::operations::attention::{
    AttentionConfiguration, AttentionImplementation, AttentionInputs, DotProductAttention, DotProductAttentionBackward,
    dot_product_attention_backward_ir_composition, dot_product_attention_ir_composition,
};
use crate::programs::ProgramError;

impl DotProductAttention for Array {
    fn dot_product_attention(
        inputs: AttentionInputs<Self>,
        configuration: AttentionConfiguration,
    ) -> Result<(Self, Option<Self>), ProgramError> {
        if configuration.implementation() == AttentionImplementation::Fused {
            return Err(ProgramError::UnsupportedOperation {
                message: "the eager array backend does not provide a fused attention implementation".to_string(),
            });
        }
        let inputs = AttentionInputs {
            query: ArrayIrValue::Array(inputs.query),
            key: ArrayIrValue::Array(inputs.key),
            value: ArrayIrValue::Array(inputs.value),
            bias: inputs.bias.map(ArrayIrValue::Array),
            mask: inputs.mask.map(ArrayIrValue::Array),
            query_sequence_lengths: inputs.query_sequence_lengths.map(ArrayIrValue::Array),
            key_value_sequence_lengths: inputs.key_value_sequence_lengths.map(ArrayIrValue::Array),
        };
        dot_product_attention_ir_composition(&inputs, configuration)
    }
}

impl DotProductAttentionBackward for Array {
    fn dot_product_attention_backward(
        inputs: AttentionInputs<Self>,
        output: Self,
        residual: Self,
        output_cotangent: Self,
        configuration: AttentionConfiguration,
    ) -> Result<Vec<Self>, ProgramError> {
        if configuration.implementation() == AttentionImplementation::Fused {
            return Err(ProgramError::UnsupportedOperation {
                message: "the eager array backend does not provide a fused attention implementation".to_string(),
            });
        }
        let inputs = AttentionInputs {
            query: ArrayIrValue::Array(inputs.query),
            key: ArrayIrValue::Array(inputs.key),
            value: ArrayIrValue::Array(inputs.value),
            bias: inputs.bias.map(ArrayIrValue::Array),
            mask: inputs.mask.map(ArrayIrValue::Array),
            query_sequence_lengths: inputs.query_sequence_lengths.map(ArrayIrValue::Array),
            key_value_sequence_lengths: inputs.key_value_sequence_lengths.map(ArrayIrValue::Array),
        };
        dot_product_attention_backward_ir_composition(
            &inputs,
            &ArrayIrValue::Array(output),
            &ArrayIrValue::Array(residual),
            &ArrayIrValue::Array(output_cotangent),
            configuration,
        )
    }
}
