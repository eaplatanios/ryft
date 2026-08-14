//! Differentiable scaled dot-product attention entry point.

use ryft_macros::Parameterized;

use crate::arrays::ArrayType;
use crate::contexts::{Context, Domain};
use crate::differentiation::DifferentiableType;
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::attention::{
    AttentionConfiguration, AttentionInputs, DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME,
    DOT_PRODUCT_ATTENTION_OPERATION_NAME, DotProductAttention, DotProductAttentionBackwardOperation,
    DotProductAttentionOperation,
};
use crate::operations::constants::zero::{Zero, ZeroOperationProvider};
use crate::parameters::Parameter;
use crate::programs::{ProgramError, Typed, Value};
use crate::tracing::DomainTracer;
use crate::tracing_v2::{CustomVjp, custom_vjp};

/// Residuals retained by the fused attention reverse rule.
#[derive(Clone, Debug, Parameterized)]
pub struct AttentionResiduals<P: Parameter> {
    /// Forward operands, including the present optional leaves.
    inputs: AttentionInputs<P>,

    /// Attended forward output.
    output: P,

    /// Log-sum-exp statistic consumed by the fused backward boundary.
    statistic: P,
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

/// Stages the fused attention backward boundary over one canonical input structure.
fn bind_attention_backward<D: Domain<Type = ArrayType>>(
    inputs: AttentionInputs<DomainTracer<D>>,
    output: DomainTracer<D>,
    statistic: DomainTracer<D>,
    output_cotangent: DomainTracer<D>,
    configuration: AttentionConfiguration,
) -> Result<AttentionInputs<DomainTracer<D>>, ProgramError>
where
    D::Operation: From<DotProductAttentionBackwardOperation> + ZeroOperationProvider<ArrayType>,
{
    let signature = inputs.signature();
    let context = inputs.query.dispatch_domain();
    let bias_cotangent_type = inputs.bias.as_ref().map(|bias| bias.r#type().cotangent());
    let mut operands = inputs.into_values();
    operands.extend([output, statistic, output_cotangent]);
    let mut outputs = context.bind(
        DotProductAttentionBackwardOperation::new(configuration, signature),
        Vec::new(),
        operands.as_slice(),
    )?;
    let has_live_bias_cotangent = bias_cotangent_type.as_ref().is_some_and(|r#type| !r#type.is_zero_space());
    check_count!("output", outputs, 3 + usize::from(has_live_bias_cotangent), ProgramError);
    let bias = match bias_cotangent_type {
        None => None,
        Some(_) if has_live_bias_cotangent => Some(outputs.remove(3)),
        Some(r#type) => Some(context.zero(&r#type)?),
    };
    Ok(AttentionInputs {
        query: outputs.remove(0),
        key: outputs.remove(0),
        value: outputs.remove(0),
        bias,
        mask: None,
        query_sequence_lengths: None,
        key_value_sequence_lengths: None,
    })
}

/// Creates the differentiable scaled dot-product attention entry point for one canonical [`AttentionInputs`] tree.
///
/// The primal path returns only the attended output. Reverse-mode differentiation runs a residual-producing forward
/// with the same configuration and then applies the narrow fused-backward boundary. Optional Boolean masks and integer
/// sequence lengths remain ordinary structural leaves of `inputs`; their cotangents are produced by Ryft's existing
/// zero-space machinery, while a present floating-point bias receives the cotangent returned by the backward program.
/// This one structured entry point replaces separate query/key/value, bias, and sequence-length tuple families.
///
/// # Parameters
///
///   - `configuration`: Value-independent attention semantics and implementation selection.
pub fn differentiable_dot_product_attention<D>(
    configuration: AttentionConfiguration,
) -> CustomVjp<
    impl Fn(AttentionInputs<DomainTracer<D>>) -> Result<DomainTracer<D>, ProgramError>,
    impl Fn(
        AttentionInputs<DomainTracer<D>>,
    ) -> Result<(DomainTracer<D>, AttentionResiduals<DomainTracer<D>>), ProgramError>,
    impl Fn(AttentionResiduals<DomainTracer<D>>, DomainTracer<D>) -> Result<AttentionInputs<DomainTracer<D>>, ProgramError>,
    AttentionInputs<DomainTracer<D>>,
    DomainTracer<D>,
    AttentionResiduals<DomainTracer<D>>,
>
where
    D: Domain<Type = ArrayType>,
    D::Operation: From<DotProductAttentionBackwardOperation> + ZeroOperationProvider<ArrayType>,
    DomainTracer<D>: DotProductAttention,
{
    custom_vjp(
        move |inputs: AttentionInputs<DomainTracer<D>>| {
            let (output, _) = DomainTracer::<D>::dot_product_attention(inputs, configuration.with_residual(false))?;
            Ok(output)
        },
        move |inputs: AttentionInputs<DomainTracer<D>>| {
            let (output, statistic) =
                DomainTracer::<D>::dot_product_attention(inputs.clone(), configuration.with_residual(true))?;
            let statistic = statistic.unwrap();
            Ok((output.clone(), AttentionResiduals { inputs, output, statistic }))
        },
        move |residuals: AttentionResiduals<DomainTracer<D>>, output_cotangent| {
            let structural_inputs = residuals.inputs.clone();
            let mut cotangents = bind_attention_backward::<D>(
                residuals.inputs,
                residuals.output,
                residuals.statistic,
                output_cotangent,
                configuration.with_residual(true),
            )?;
            let zero = |value: &DomainTracer<D>| value.context().zero(&value.r#type().cotangent());
            cotangents.mask = structural_inputs.mask.as_ref().map(zero).transpose()?;
            cotangents.query_sequence_lengths =
                structural_inputs.query_sequence_lengths.as_ref().map(zero).transpose()?;
            cotangents.key_value_sequence_lengths =
                structural_inputs.key_value_sequence_lengths.as_ref().map(zero).transpose()?;
            Ok(cotangents)
        },
    )
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, DataType, Dimension, Shape,
    };
    use crate::contexts::EagerContext;
    use crate::differentiation::Differentiate;
    use crate::operations::attention::dot_product_attention_ir_composition;
    use crate::operations::math::reduce::{Reduce, ReductionKind};
    use crate::tracing::Trace;

    use super::*;

    #[test]
    fn test_differentiable_dot_product_attention() {
        let r#type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(1), Dimension::Static(1), Dimension::Static(1), Dimension::Static(1)]),
        );
        let inputs = AttentionInputs {
            query: Array::from_f64s(r#type.clone(), vec![2.0]),
            key: Array::from_f64s(r#type.clone(), vec![3.0]),
            value: Array::from_f64s(r#type, vec![5.0]),
            bias: Some(Array::from_f64s(ArrayType::scalar(DataType::F64), vec![1.0])),
            mask: Some(Array::from_elements(ArrayType::scalar(DataType::Boolean), &[true]).unwrap()),
            query_sequence_lengths: Some(
                Array::from_elements(ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(1)])), &[1_i32])
                    .unwrap(),
            ),
            key_value_sequence_lengths: Some(
                Array::from_elements(ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(1)])), &[1_i32])
                    .unwrap(),
            ),
        };
        let function = differentiable_dot_product_attention::<EagerContext<Array, ArrayOperation<Array>>>(
            AttentionConfiguration::new().with_scale(0.5),
        );

        // A one-element attention distribution is identically one: the output is `value`, so query and key have
        // zero gradients while value has unit gradient.
        let (output, gradients) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .differentiate_at(inputs)
            .value_and_gradient(|inputs| function.call(inputs).unwrap().reduce(&[0, 1, 2, 3], ReductionKind::Sum))
            .unwrap();

        assert_eq!(output.to_f64s(), vec![5.0]);
        assert_eq!(gradients.query.to_f64s(), vec![0.0]);
        assert_eq!(gradients.key.to_f64s(), vec![0.0]);
        assert_eq!(gradients.value.to_f64s(), vec![1.0]);
        assert_eq!(gradients.bias.unwrap().to_f64s(), vec![0.0]);
        assert_eq!(gradients.mask.unwrap().r#type().data_type(), DataType::Zero);
        assert_eq!(gradients.query_sequence_lengths.unwrap().r#type().data_type(), DataType::Zero);
        assert_eq!(gradients.key_value_sequence_lengths.unwrap().r#type().data_type(), DataType::Zero);

        // Integer bias is accepted by the forward but has a zero-dimensional cotangent space.
        let r#type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(1), Dimension::Static(1), Dimension::Static(1), Dimension::Static(1)]),
        );
        let integer_bias = Array::from_elements(ArrayType::scalar(DataType::I32), &[0_i32]).unwrap();
        let inputs = AttentionInputs {
            query: Array::from_f64s(r#type.clone(), vec![2.0]),
            key: Array::from_f64s(r#type.clone(), vec![3.0]),
            value: Array::from_f64s(r#type, vec![5.0]),
            bias: Some(integer_bias),
            mask: None,
            query_sequence_lengths: None,
            key_value_sequence_lengths: None,
        };
        let (_, gradients) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .differentiate_at(inputs)
            .value_and_gradient(|inputs| function.call(inputs).unwrap().reduce(&[0, 1, 2, 3], ReductionKind::Sum))
            .unwrap();

        assert_eq!(gradients.bias.unwrap().r#type().data_type(), DataType::Zero);
    }

    #[test]
    fn test_dot_product_attention_composition_jvp() {
        let r#type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(1), Dimension::Static(1), Dimension::Static(1), Dimension::Static(1)]),
        );
        let scalar_type = ArrayType::scalar(DataType::F64);
        let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |inputs: AttentionInputs<_>| {
                let (output, residual) = dot_product_attention_ir_composition(
                    &inputs,
                    AttentionConfiguration::new().with_scale(0.5).with_residual(true),
                )?;
                Ok((output.into_value(), residual.unwrap().into_value()))
            },
            AttentionInputs {
                query: ArrayIrType::Array(r#type.clone()),
                key: ArrayIrType::Array(r#type.clone()),
                value: ArrayIrType::Array(r#type.clone()),
                bias: Some(ArrayIrType::Array(scalar_type.clone())),
                mask: None,
                query_sequence_lengths: None,
                key_value_sequence_lengths: None,
            },
        )
        .unwrap();
        let jvp = program.into_flat_program().jvp().unwrap();
        let outputs = jvp
            .interpret(vec![
                ArrayIrValue::Array(Array::from_f64s(r#type.clone(), vec![2.0])),
                ArrayIrValue::Array(Array::from_f64s(r#type.clone(), vec![3.0])),
                ArrayIrValue::Array(Array::from_f64s(r#type.clone(), vec![5.0])),
                ArrayIrValue::Array(Array::from_f64s(scalar_type.clone(), vec![1.0])),
                ArrayIrValue::Array(Array::from_f64s(r#type.clone(), vec![7.0])),
                ArrayIrValue::Array(Array::from_f64s(r#type.clone(), vec![11.0])),
                ArrayIrValue::Array(Array::from_f64s(r#type, vec![13.0])),
                ArrayIrValue::Array(Array::from_f64s(scalar_type, vec![17.0])),
            ])
            .unwrap();
        let [
            ArrayIrValue::Array(output),
            ArrayIrValue::Array(_residual),
            ArrayIrValue::Array(tangent),
            ArrayIrValue::Array(residual_tangent),
        ] = outputs.as_slice()
        else {
            panic!("attention JVP produces array primals and tangents")
        };

        // With one visible key, the softmax is identically one. The output is therefore the value itself: query,
        // key, and bias perturbations vanish while the value perturbation passes through unchanged.
        assert_eq!(output.to_f64s(), vec![5.0]);
        assert_eq!(tangent.to_f64s(), vec![13.0]);
        assert_eq!(residual_tangent.to_f64s(), vec![0.0]);
    }
}
