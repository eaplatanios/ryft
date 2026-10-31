//! Differentiable scaled dot-product attention entry point.

use ryft_macros::Parameterized;

use crate::arrays::ArrayType;
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::DifferentiableType;
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::attention::{
    AttentionConfiguration, AttentionImplementation, AttentionInputs, DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME,
    DOT_PRODUCT_ATTENTION_OPERATION_NAME, DotProductAttention, DotProductAttentionBackwardOperation,
    DotProductAttentionOperation,
};
use crate::operations::collectives::parallel_vary::ManualVariationAlignment;
use crate::operations::composites::rules::CustomRuleDefinition;
use crate::operations::composites::{
    Composite, CompositeCallContext, CompositeVjp, DefaultJvp, SymbolicZeroVjp, composite,
};
use crate::parameters::{Parameter, Parameterized};
use crate::programs::{MaybeZero, ProgramError, Typed, Value};
use crate::tracing::{DomainTracer, DomainTracingContext};

/// Residuals retained by the attention reverse rule.
#[derive(Clone, Debug, PartialEq, Parameterized)]
pub struct AttentionResiduals<P: Parameter> {
    /// Forward inputs, including the present optional leaves.
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
            // the activation-producing forward with [`DotProductAttentionBackwardOperation`] through a custom VJP.
            Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "`{DOT_PRODUCT_ATTENTION_OPERATION_NAME}` does not support differentiation; use \
                     `differentiable_dot_product_attention` for the training path"
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
                    "`{DOT_PRODUCT_ATTENTION_BACKWARD_OPERATION_NAME}` does not support differentiation; \
                     differentiate an explicit attention composition for higher-order derivatives"
                ),
            }
            .into())
        }
    },
    transpose = @nonlinear,
}

/// Returns the structural-zero cotangent of `value`.
fn zero_cotangent<D: Domain<Type = ArrayType>>(
    value: &DomainTracer<D>,
) -> Result<MaybeZero<DomainTracer<D>>, ProgramError>
where
    DomainTracingContext<D>: StagingContext<Type = ArrayType, Constant = D::Constant, Operation = D::Operation>,
{
    Ok(MaybeZero::Zero(value.r#type().cotangent()?))
}

/// Stages the attention backward boundary over one canonical input structure. The kernel returns the cotangents of the
/// query, key, and value, and of a bias whose cotangent space is not zero-dimensional, so the cotangents of the other
/// inputs (i.e., of the masks and sequence lengths, and of a bias with a zero-dimensional cotangent space) are
/// structural zeros, which contribute nothing to their inputs' cotangents.
fn bind_attention_backward<D: Domain<Type = ArrayType>>(
    inputs: AttentionInputs<DomainTracer<D>>,
    output: DomainTracer<D>,
    statistic: DomainTracer<D>,
    output_cotangent: DomainTracer<D>,
    configuration: AttentionConfiguration,
) -> Result<AttentionInputs<MaybeZero<DomainTracer<D>>>, ProgramError>
where
    D::Operation: From<DotProductAttentionBackwardOperation>,
    DomainTracingContext<D>: StagingContext<Type = ArrayType, Constant = D::Constant, Operation = D::Operation>,
{
    let signature = inputs.signature();
    let context = inputs.query.domain();
    let bias_cotangent_type = inputs.bias.as_ref().map(|bias| bias.r#type().cotangent()).transpose()?;
    let mask = inputs.mask.as_ref().map(zero_cotangent::<D>).transpose()?;
    let query_sequence_lengths = inputs.query_sequence_lengths.as_ref().map(zero_cotangent::<D>).transpose()?;
    let key_value_sequence_lengths = inputs.key_value_sequence_lengths.as_ref().map(zero_cotangent::<D>).transpose()?;
    let mut backward_inputs = inputs.into_values();
    backward_inputs.extend([output, statistic, output_cotangent]);
    let mut outputs = context.bind(
        DotProductAttentionBackwardOperation::new(configuration, signature),
        Vec::new(),
        backward_inputs.as_slice(),
    )?;
    let has_live_bias_cotangent = bias_cotangent_type.as_ref().is_some_and(|r#type| !r#type.is_zero_space());
    check_count!("output", outputs, 3 + usize::from(has_live_bias_cotangent), ProgramError);
    let bias = match bias_cotangent_type {
        None => None,
        Some(_) if has_live_bias_cotangent => Some(MaybeZero::Value(outputs.remove(3))),
        Some(r#type) => Some(MaybeZero::Zero(r#type)),
    };
    Ok(AttentionInputs {
        query: MaybeZero::Value(outputs.remove(0)),
        key: MaybeZero::Value(outputs.remove(0)),
        value: MaybeZero::Value(outputs.remove(0)),
        bias,
        mask,
        query_sequence_lengths,
        key_value_sequence_lengths,
    })
}

/// Differentiable scaled dot-product attention training function, built by [`differentiable_dot_product_attention`].
/// It is a [`Composite`](crate::Composite) with reverse mode rules whose input variation transitions precede its
/// composite boundary, which lets the transitions' transposes accumulate invariant-input contributions exactly once.
/// Its forward and backward rules are retained: an ordinary call traces only the attention primal, and the rules are
/// traced on the first reverse mode request of each specialization and cached for as long as the function lives.
pub struct DifferentiableDotProductAttention<D: Domain<Type = ArrayType>>
where
    DomainTracingContext<D>: StagingContext<Type = ArrayType, Constant = D::Constant, Operation = D::Operation>,
{
    /// Composite that stages the attention training call.
    function: Composite<
        DomainTracer<D>,
        AttentionInputs<DomainTracer<D>>,
        DomainTracer<D>,
        DefaultJvp,
        SymbolicZeroVjp<
            DomainTracer<D>,
            AttentionInputs<DomainTracer<D>>,
            DomainTracer<D>,
            AttentionResiduals<DomainTracer<D>>,
        >,
    >,
}

impl<D: Domain<Type = ArrayType>> DifferentiableDotProductAttention<D>
where
    D::Constant: 'static,
    D::Operation: 'static,
    DomainTracingContext<D>: StagingContext<Type = ArrayType, Constant = D::Constant, Operation = D::Operation>,
{
    /// Aligns the inputs' manual variation and stages the attention training call.
    pub fn call<C, V>(&self, inputs: AttentionInputs<V>) -> Result<V, ProgramError>
    where
        C: CompositeCallContext<Type = ArrayType, Value = V, Constant = D::Constant, Operation = D::Operation>,
        V: Value<Type = ArrayType, Domain = C> + ManualVariationAlignment<ArrayType>,
        CustomRuleDefinition<C::Constant, C::Operation>: CompositeVjp<
                SymbolicZeroVjp<
                    DomainTracer<D>,
                    AttentionInputs<DomainTracer<D>>,
                    DomainTracer<D>,
                    AttentionResiduals<DomainTracer<D>>,
                >,
                DomainTracer<D>,
                AttentionInputs<DomainTracer<D>>,
                DomainTracer<D>,
            >,
    {
        let signature = inputs.signature();
        let inputs = inputs.into_values();
        let inputs = ManualVariationAlignment::align_manual_variation(&inputs)?;
        self.function.call(AttentionInputs::from_values(signature, &inputs)?)
    }
}

/// Creates the differentiable scaled dot-product attention entry point for one canonical [`AttentionInputs`] tree.
///
/// The primal path returns only the attended output. Reverse mode differentiation runs a residual-producing forward
/// with the same configuration and then applies the narrow backward boundary. [`AttentionImplementation::Automatic`]
/// selects [`AttentionImplementation::Portable`] for this training entry point, preserving small dominant-logit
/// gradients with its stabilized backward composition. Explicitly requesting [`AttentionImplementation::Fused`] opts
/// into the backend's native forward and backward kernels, including their numerical behavior; the portable gradient
/// guarantees do not apply to those kernels. Dropout requires an explicit fused configuration. Inference through
/// [`DotProductAttention`] retains automatic fusion.
///
/// Optional Boolean masks and integer sequence lengths remain ordinary structural leaves of `inputs`; the backward rule
/// returns structural zeros for their cotangents (refer to [`Composite::with_symbolic_zero_vjp`]), which Ryft's
/// existing zero-space machinery materializes only where a consumer needs a value, while a present floating-point bias
/// receives the cotangent returned by the backward program. A structurally zero output cotangent (e.g., when the output
/// is unused) skips the backward kernel altogether. This one structured entry point replaces separate query/key/value,
/// bias, and sequence-length tuple families.
///
/// The returned function retains its rules (refer to [`DifferentiableDotProductAttention`]), so it should be created
/// once and called many times, which shares the traced rule specializations across its calls. To embed the resulting
/// programs in a heterogeneous value universe, use
/// [`Program::into_unprojected`](crate::programs::Program::into_unprojected), which converts the staged calls together
/// with their rules.
///
/// # Parameters
///
///   - `configuration`: Value-independent attention semantics and implementation selection.
pub fn differentiable_dot_product_attention<D>(
    configuration: AttentionConfiguration,
) -> DifferentiableDotProductAttention<D>
where
    D: Domain<Type = ArrayType>,
    D::Constant: 'static,
    D::Operation: 'static + From<DotProductAttentionBackwardOperation>,
    DomainTracer<D>: DotProductAttention,
    DomainTracingContext<D>: StagingContext<Type = ArrayType, Constant = D::Constant, Operation = D::Operation>,
{
    let configuration = match configuration.implementation() {
        AttentionImplementation::Automatic => configuration.with_implementation(AttentionImplementation::Portable),
        AttentionImplementation::Portable | AttentionImplementation::Fused => configuration,
    };
    let primal = move |inputs: AttentionInputs<DomainTracer<D>>| -> Result<_, ProgramError> {
        let (output, _) = DomainTracer::<D>::dot_product_attention(inputs, configuration.with_residual(false))?;
        Ok(output)
    };
    let forward = move |inputs: AttentionInputs<DomainTracer<D>>| -> Result<_, ProgramError> {
        let (output, statistic) =
            DomainTracer::<D>::dot_product_attention(inputs.clone(), configuration.with_residual(true))?;
        let statistic = statistic.unwrap();
        Ok((output.clone(), AttentionResiduals { inputs, output, statistic }))
    };
    let backward = move |residuals: AttentionResiduals<DomainTracer<D>>,
                         output_cotangent: MaybeZero<DomainTracer<D>>|
          -> Result<_, ProgramError> {
        match output_cotangent {
            MaybeZero::Value(output_cotangent) => bind_attention_backward::<D>(
                residuals.inputs,
                residuals.output,
                residuals.statistic,
                output_cotangent,
                configuration.with_residual(true),
            ),
            MaybeZero::Zero(_) => residuals.inputs.try_map_parameters(|input| zero_cotangent::<D>(&input)),
        }
    };
    DifferentiableDotProductAttention {
        function: composite(primal)
            .with_symbolic_zero_vjp(forward, backward)
            .with_name(DOT_PRODUCT_ATTENTION_OPERATION_NAME),
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use half::bf16;
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, DataType, Dimension, Shape,
    };
    use crate::contexts::EagerContext;
    use crate::differentiation::Differentiate;
    use crate::operations::attention::dot_product_attention_ir_composition;
    use crate::operations::reductions::{Reduce, ReductionKind};
    use crate::tracing::Trace;

    use super::*;

    /// Differentiates the sum of one portable attention output at unit score scale.
    fn attention_sum_gradient(inputs: AttentionInputs<Array>) -> (Array, AttentionInputs<Array>) {
        let function = differentiable_dot_product_attention::<EagerContext<Array, ArrayOperation<Array>>>(
            AttentionConfiguration::new().with_scale(1.0),
        );
        EagerContext::<Array, ArrayOperation<Array>>::new()
            .differentiate_at(inputs)
            .value_and_gradient(|inputs| {
                function.call(inputs).unwrap().reduce(&[0, 1, 2, 3], ReductionKind::Sum).unwrap()
            })
            .unwrap()
    }

    #[test]
    fn test_differentiable_dot_product_attention_call() {
        use crate::arrays::{LogicalMesh, MeshAxis, MeshAxisType, Sharding};
        use crate::axes::NamedAxis;
        use crate::programs::Operation;
        use crate::tracing::DomainTracingContext;

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let invariant = ArrayType::new_static(DataType::F32, [1, 1, 1, 1])
            .with_sharding(Sharding::replicated(mesh.clone(), 4))
            .unwrap();
        let varying = invariant
            .clone()
            .with_sharding(Sharding::replicated(mesh.clone(), 4).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let function = differentiable_dot_product_attention::<EagerContext<Array, ArrayOperation<Array>>>(
            AttentionConfiguration::new(),
        );
        let (output, program) =
            DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::trace_with_named_axes(
                |inputs| function.call(inputs),
                AttentionInputs::new(invariant.clone(), varying.clone(), invariant),
                vec![("x".to_string(), NamedAxis::Mesh { mesh, axis: 0, size: 2 })],
            )
            .unwrap();
        assert_eq!(output, varying);
        assert_eq!(
            program.instructions().iter().map(|instruction| instruction.operation().name()).collect::<Vec<_>>(),
            vec!["parallel_vary", "parallel_vary", "composite"],
        );
    }

    #[test]
    fn test_differentiable_dot_product_attention_implementation_selection() {
        let r#type = ArrayType::new_static(DataType::F32, [1, 1, 1, 1]);
        for (requested, selected) in [
            (AttentionImplementation::Automatic, "portable"),
            (AttentionImplementation::Portable, "portable"),
            (AttentionImplementation::Fused, "fused"),
        ] {
            let function = differentiable_dot_product_attention::<EagerContext<Array, ArrayOperation<Array>>>(
                AttentionConfiguration::new().with_implementation(requested),
            );
            let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
                |inputs| function.call(inputs),
                AttentionInputs::new(r#type.clone(), r#type.clone(), r#type.clone()),
            )
            .unwrap();
            assert_eq!(
                program.to_string(),
                indoc! {"
                    lambda %0:f32[1, 1, 1, 1], %1:f32[1, 1, 1, 1], %2:f32[1, 1, 1, 1] .
                    let %3:f32[1, 1, 1, 1] = composite [name=\"dot_product_attention\"] %0 %1 %2 [
                        primal={
                            lambda %0:f32[1, 1, 1, 1], %1:f32[1, 1, 1, 1], %2:f32[1, 1, 1, 1] .
                            let %3:f32[1, 1, 1, 1] = dot_product_attention [
                                implementation=SELECTED,
                                signature=AttentionInputSignature { bias: false, mask: false, query_sequence_lengths: false, key_value_sequence_lengths: false },
                            ] %0 %1 %2
                            in (%3)
                        },
                    ]
                    in (%3)
                "}
                .trim_end()
                .replace("SELECTED", selected),
            );
        }
    }

    #[test]
    fn test_differentiable_dot_product_attention() {
        let r#type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(1), Dimension::Static(1), Dimension::Static(1), Dimension::Static(1)]),
        );
        let inputs = AttentionInputs {
            query: Array::from_elements::<f64>(r#type.clone(), &[2.0]).unwrap(),
            key: Array::from_elements::<f64>(r#type.clone(), &[3.0]).unwrap(),
            value: Array::from_elements::<f64>(r#type, &[5.0]).unwrap(),
            bias: Some(Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[1.0]).unwrap()),
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
            .value_and_gradient(|inputs| {
                function.call(inputs).unwrap().reduce(&[0, 1, 2, 3], ReductionKind::Sum).unwrap()
            })
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
            query: Array::from_elements::<f64>(r#type.clone(), &[2.0]).unwrap(),
            key: Array::from_elements::<f64>(r#type.clone(), &[3.0]).unwrap(),
            value: Array::from_elements::<f64>(r#type, &[5.0]).unwrap(),
            bias: Some(integer_bias),
            mask: None,
            query_sequence_lengths: None,
            key_value_sequence_lengths: None,
        };
        let (_, gradients) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .differentiate_at(inputs)
            .value_and_gradient(|inputs| {
                function.call(inputs).unwrap().reduce(&[0, 1, 2, 3], ReductionKind::Sum).unwrap()
            })
            .unwrap();

        assert_eq!(gradients.bias.unwrap().r#type().data_type(), DataType::Zero);
    }

    #[test]
    fn test_differentiable_dot_product_attention_saturated_logits() {
        let inputs = AttentionInputs::new(
            Array::from_elements(ArrayType::new_static(DataType::F32, [1, 1, 1, 1]), &[1.0f32]).unwrap(),
            Array::from_elements(ArrayType::new_static(DataType::F32, [1, 2, 1, 1]), &[0.0f32, -20.0]).unwrap(),
            Array::from_elements(ArrayType::new_static(DataType::F32, [1, 2, 1, 1]), &[1.0f32, 0.0]).unwrap(),
        );
        let (output, gradients) = attention_sum_gradient(inputs);
        assert_eq!(output.to_f64s(), vec![1.0]);
        let key_gradient = gradients.key.to_f64s();
        let tail = (-20.0f64).exp();
        let expected = tail / (1.0 + tail).powi(2);
        assert_abs_diff_eq!(key_gradient[0], expected, epsilon = 1e-15);
        assert_eq!(key_gradient[0], -key_gradient[1]);
        assert_abs_diff_eq!(gradients.query.to_f64s()[0], 20.0 * expected, epsilon = 1e-14);
    }

    #[test]
    fn test_differentiable_dot_product_attention_ties_and_mask() {
        let inputs = AttentionInputs {
            mask: Some(
                Array::from_elements(ArrayType::new_static(DataType::Boolean, [1, 1, 1, 3]), &[true, true, false])
                    .unwrap(),
            ),
            ..AttentionInputs::new(
                Array::from_elements(ArrayType::new_static(DataType::F32, [1, 1, 1, 1]), &[1.0f32]).unwrap(),
                Array::from_elements(ArrayType::new_static(DataType::F32, [1, 3, 1, 1]), &[0.0f32, 0.0, 100.0])
                    .unwrap(),
                Array::from_elements(ArrayType::new_static(DataType::F32, [1, 3, 1, 1]), &[1.0f32, 0.0, 12.0]).unwrap(),
            )
        };
        let (output, gradients) = attention_sum_gradient(inputs);
        assert_eq!(output.to_f64s(), vec![0.5]);
        assert_eq!(gradients.key.to_f64s(), vec![0.25, -0.25, 0.0]);
        assert_eq!(gradients.query.to_f64s(), vec![0.0]);
        assert_eq!(gradients.value.to_f64s(), vec![0.5, 0.5, 0.0]);
    }

    #[test]
    fn test_differentiable_dot_product_attention_narrow_residual() {
        // The BF16 log-sum-exp residual rounds 1000 + log(2) to 1000. Reconstructing probabilities from that
        // statistic would double the gradient; recomputing normalization retains the correct quarter gradients.
        let inputs = AttentionInputs::new(
            Array::from_elements(ArrayType::new_static(DataType::BF16, [1, 1, 1, 1]), &[bf16::from_f32(1.0)]).unwrap(),
            Array::from_elements(
                ArrayType::new_static(DataType::BF16, [1, 2, 1, 1]),
                &[bf16::from_f32(1000.0), bf16::from_f32(1000.0)],
            )
            .unwrap(),
            Array::from_elements(
                ArrayType::new_static(DataType::BF16, [1, 2, 1, 1]),
                &[bf16::from_f32(1.0), bf16::from_f32(0.0)],
            )
            .unwrap(),
        );
        let (output, gradients) = attention_sum_gradient(inputs);
        assert_eq!(output.to_f64s(), vec![0.5]);
        assert_eq!(gradients.key.to_f64s(), vec![0.25, -0.25]);
        assert_eq!(gradients.query.to_f64s(), vec![0.0]);
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
                ArrayIrValue::Array(Array::from_elements::<f64>(r#type.clone(), &[2.0]).unwrap()),
                ArrayIrValue::Array(Array::from_elements::<f64>(r#type.clone(), &[3.0]).unwrap()),
                ArrayIrValue::Array(Array::from_elements::<f64>(r#type.clone(), &[5.0]).unwrap()),
                ArrayIrValue::Array(Array::from_elements::<f64>(scalar_type.clone(), &[1.0]).unwrap()),
                ArrayIrValue::Array(Array::from_elements::<f64>(r#type.clone(), &[7.0]).unwrap()),
                ArrayIrValue::Array(Array::from_elements::<f64>(r#type.clone(), &[11.0]).unwrap()),
                ArrayIrValue::Array(Array::from_elements::<f64>(r#type, &[13.0]).unwrap()),
                ArrayIrValue::Array(Array::from_elements::<f64>(scalar_type, &[17.0]).unwrap()),
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
