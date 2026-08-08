//! Reference [`Array`] kernels for the attention operation family contracts.
//!
//! Scaled dot-product attention has no dedicated reference kernel: both the forward and the backward contracts are
//! answered by evaluating the shared universe-neutral compositions eagerly over concrete arrays.

use crate::arrays::arrays::Array;
use crate::macros::check_count;
use crate::operations::attention::{
    AttentionMask, DotProductAttention, DotProductAttentionBackward, dot_product_attention_backward_composition,
    dot_product_attention_composition,
};
use crate::programs::{ProgramError, Value};

// TODO(eaplatanios): Review this.

impl DotProductAttention for Array {
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
        let (output, _) = dot_product_attention_composition(
            &self.dispatch_domain(),
            self,
            key,
            value,
            bias,
            sequence_lengths,
            scale,
            mask,
            sliding_window,
            dropout,
            false,
        )?;
        Ok(output)
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
        let (output, activation) = dot_product_attention_composition(
            &self.dispatch_domain(),
            self,
            key,
            value,
            bias,
            sequence_lengths,
            scale,
            mask,
            sliding_window,
            dropout,
            true,
        )?;
        // The composition returns the activation statistic whenever it is requested.
        Ok((output, activation.unwrap()))
    }
}

impl DotProductAttentionBackward for Array {
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
        let mut cotangents = dot_product_attention_backward_composition(
            &self.dispatch_domain(),
            self,
            key,
            value,
            bias,
            sequence_lengths,
            output,
            activation,
            output_cotangent,
            scale,
            mask,
            sliding_window,
            dropout,
        )?;
        let bias_cotangent = bias.is_some().then(|| cotangents.remove(3));
        check_count!("output", cotangents, 3, ProgramError);
        let value_cotangent = cotangents.remove(2);
        let key_cotangent = cotangents.remove(1);
        Ok((cotangents.remove(0), key_cotangent, value_cotangent, bias_cotangent))
    }
}
