//! Differentiable scaled dot-product attention entry point.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ryft_macros::Parameterized;

use crate::batching::BatchableType;
use crate::contexts::{Domain, StagingContext};
use crate::differentiation::DifferentiableType;
use crate::macros::check_count;
use crate::operations::attention::composition::AttentionUniverse;
use crate::operations::attention::definitions::{
    AttentionDefinitions, align_attention_inputs, attention_composite_key,
};
use crate::operations::attention::{
    AttentionBackwardInputs, AttentionConfiguration, AttentionImplementation, AttentionInputSignature, AttentionInputs,
    AttentionRules, DOT_PRODUCT_ATTENTION_COMPOSITE_NAME, DOT_PRODUCT_ATTENTION_TRAINING_COMPOSITE_NAME,
    attention_forward_output_types,
};
use crate::operations::collectives::parallel_vary::ManualVariationAlignment;
use crate::operations::composites::{
    Composite, CompositeCallContext, CompositeVjp, CustomRuleDefinition, CustomRuleTracer, DefaultJvp, SymbolicZeroVjp,
    composite,
};
use crate::parameters::{Parameter, Parameterized};
use crate::programs::{MaybeZero, Operation, ProgramError, Typed, Value};
use crate::tracing::{DomainTracer, DomainTracingContext, TracingContext};

/// Residuals retained by the reverse mode rules of differentiable attention.
#[derive(Clone, Debug, PartialEq, Parameterized)]
pub struct AttentionResiduals<P: Parameter> {
    /// Forward inputs, including the present optional inputs.
    inputs: AttentionInputs<P>,

    /// Attended forward output.
    output: P,

    /// Log-sum-exp residual of the forward output, which the backward definition consumes.
    residual: P,
}

/// [`CustomRuleTracer`] of the operation family `(V, O)`.
type RuleTracer<V, O> = CustomRuleTracer<V, O>;

/// Reverse mode configuration of the named composites that stage differentiable attention.
pub type AttentionVjp<V, O> =
    SymbolicZeroVjp<V, O, AttentionInputs<RuleTracer<V, O>>, RuleTracer<V, O>, AttentionResiduals<RuleTracer<V, O>>>;

/// Named composite that stages differentiable attention for one optional-input signature.
type TrainingFunction<V, O> =
    Composite<V, O, AttentionInputs<RuleTracer<V, O>>, RuleTracer<V, O>, DefaultJvp, AttentionVjp<V, O>>;

/// Returns the structural-zero cotangent of `value`.
fn zero_cotangent<X: Value<Type: DifferentiableType>>(value: &X) -> Result<MaybeZero<X>, ProgramError> {
    Ok(MaybeZero::Zero(value.r#type().cotangent()?))
}

/// Differentiable scaled dot-product attention training function, built by [`differentiable_dot_product_attention`].
/// Each call stages one call of a named composite (refer to [`DOT_PRODUCT_ATTENTION_TRAINING_COMPOSITE_NAME`]) with
/// reverse mode rules, whose primal calls the forward attention definition and whose rules call the forward
/// definition with its residual and the backward definition (refer to [`AttentionKey`](super::AttentionKey)). The
/// manual variation transitions of the inputs precede the composite boundary, which lets their transposes accumulate
/// the cotangent contributions of invariant inputs exactly once.
///
/// The function owns its named composites, one per optional-input signature, together with their definitions, so
/// it should be created once and called many times. Its primal and its definitions' bodies are then traced once per
/// input types, and its rules once per specialization, for as long as the function lives.
pub struct DifferentiableDotProductAttention<D: Domain>
where
    DomainTracingContext<D>: StagingContext<Type = D::Type, Constant = D::Constant, Operation = D::Operation>,
{
    /// Named attention definitions called by the rules of this function.
    definitions: Arc<AttentionDefinitions<D::Constant, D::Operation>>,

    /// Named composites that stage the calls of this function, by optional-input signature.
    functions: Mutex<HashMap<AttentionInputSignature, Arc<TrainingFunction<D::Constant, D::Operation>>>>,
}

impl<D: Domain> DifferentiableDotProductAttention<D>
where
    D::Constant: 'static,
    D::Operation: 'static,
    D::Type: AttentionUniverse<DomainTracer<D>> + BatchableType + DifferentiableType,
    CustomRuleDefinition<D::Constant, D::Operation>: AttentionRules<D::Constant, D::Operation>
        + CompositeVjp<
            AttentionVjp<D::Constant, D::Operation>,
            DomainTracer<D>,
            AttentionInputs<DomainTracer<D>>,
            DomainTracer<D>,
        >,
    DomainTracingContext<D>: StagingContext<Type = D::Type, Constant = D::Constant, Operation = D::Operation>
        + CompositeCallContext<Type = D::Type, Value = DomainTracer<D>, Constant = D::Constant, Operation = D::Operation>,
{
    /// Returns the configuration of this function.
    #[inline]
    pub fn configuration(&self) -> AttentionConfiguration {
        self.definitions.configuration()
    }

    /// Aligns the manual variation of `inputs` and stages one differentiable attention call, returning the attended
    /// output.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] when `inputs` are invalid for the configuration (e.g., because their shapes are
    /// incompatible) or when staging the call fails.
    pub fn call<C, X>(&self, inputs: AttentionInputs<X>) -> Result<X, ProgramError>
    where
        C: CompositeCallContext<Type = D::Type, Value = X, Constant = D::Constant, Operation = D::Operation>,
        X: Value<Type = D::Type, Domain = C> + ManualVariationAlignment<D::Type>,
    {
        let signature = inputs.signature();
        let inputs = align_attention_inputs::<D::Type, DomainTracer<D>, X>(inputs)?;
        let function = self
            .functions
            .lock()
            .expect("attention functions mutex is poisoned")
            .entry(signature)
            .or_insert_with(|| Arc::new(training_function(self.definitions.clone(), signature)))
            .clone();
        function.call(inputs)
    }
}

/// Creates the differentiable scaled dot-product attention training function with the provided configuration for the
/// values of the domain `D`, for one canonical [`AttentionInputs`] tree.
///
/// The primal path returns only the attended output. Reverse mode differentiation runs the forward definition with its
/// log-sum-exp residual and then applies the backward definition, whose portable body recomputes the normalization
/// with a stabilized composition. [`AttentionImplementation::Automatic`] selects [`AttentionImplementation::Portable`]
/// for this training entry point, which preserves small dominant-logit gradients. Explicitly requesting
/// [`AttentionImplementation::Fused`] opts into the backend's native forward and backward kernels where the backend
/// provides them (e.g., cuDNN's fused attention on the XLA backend), including their numerical behavior. The portable
/// gradient guarantees do not apply to those kernels. A configured dropout applies the same mask in the forward and
/// backward definitions, which a present dropout state (refer to [`AttentionInputs::dropout_state`]) determines.
/// Inference through [`dot_product_attention`](super::dot_product_attention) retains automatic fusion.
///
/// Optional Boolean masks, integer sequence lengths, and dropout states remain ordinary structural leaves of `inputs`.
/// The backward rule returns structural zeros for their cotangents (refer to [`Composite::with_symbolic_zero_vjp`]),
/// which Ryft's zero-space machinery materializes only where a consumer needs a value, while a present floating-point
/// bias receives the cotangent returned by the backward definition. A structurally zero output cotangent (e.g., when
/// the output is unused) skips the backward definition altogether.
///
/// The returned function owns its named composites and their rules (refer to [`DifferentiableDotProductAttention`]),
/// so it should be created once and called many times. To embed the resulting programs in a heterogeneous value
/// universe, use [`Program::into_unprojected`](crate::programs::Program::into_unprojected), which converts the staged
/// calls together with their rules.
///
/// # Parameters
///
///   - `configuration`: Value-independent attention semantics and implementation selection.
pub fn differentiable_dot_product_attention<D: Domain>(
    configuration: AttentionConfiguration,
) -> DifferentiableDotProductAttention<D>
where
    D::Constant: 'static,
    D::Operation: 'static,
    D::Type: AttentionUniverse<DomainTracer<D>> + BatchableType + DifferentiableType,
    CustomRuleDefinition<D::Constant, D::Operation>: AttentionRules<D::Constant, D::Operation>
        + CompositeVjp<
            AttentionVjp<D::Constant, D::Operation>,
            DomainTracer<D>,
            AttentionInputs<DomainTracer<D>>,
            DomainTracer<D>,
        >,
    DomainTracingContext<D>: StagingContext<Type = D::Type, Constant = D::Constant, Operation = D::Operation>
        + CompositeCallContext<Type = D::Type, Value = DomainTracer<D>, Constant = D::Constant, Operation = D::Operation>,
{
    let configuration = match configuration.implementation() {
        AttentionImplementation::Automatic => configuration.with_implementation(AttentionImplementation::Portable),
        AttentionImplementation::Portable | AttentionImplementation::Fused => configuration,
    };
    DifferentiableDotProductAttention {
        definitions: Arc::new(AttentionDefinitions::new(configuration.with_residual(false))),
        functions: Mutex::new(HashMap::new()),
    }
}

/// Returns the named composite that stages differentiable attention for the provided optional-input signature, whose
/// primal and rules call the definitions in `definitions`.
fn training_function<V: Value, O: Operation<Type = V::Type>>(
    definitions: Arc<AttentionDefinitions<V, O>>,
    signature: AttentionInputSignature,
) -> TrainingFunction<V, O>
where
    V: 'static,
    O: 'static,
    V::Type: AttentionUniverse<RuleTracer<V, O>> + BatchableType + DifferentiableType,
    CustomRuleDefinition<V, O>: AttentionRules<V, O>,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>
        + CompositeCallContext<Type = V::Type, Value = RuleTracer<V, O>, Constant = V, Operation = O>,
{
    let configuration = definitions.configuration();
    let output_types = move |input_types: AttentionInputs<V::Type>| -> Result<V::Type, ProgramError> {
        let input_types = input_types
            .into_values()
            .iter()
            .map(<V::Type as AttentionUniverse<RuleTracer<V, O>>>::to_array_type)
            .collect::<Result<Vec<_>, _>>()?;
        let mut output_types = attention_forward_output_types(
            DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
            configuration,
            signature,
            input_types.as_slice(),
        )?;
        check_count!("output", output_types, 1, ProgramError);
        Ok(<V::Type as AttentionUniverse<RuleTracer<V, O>>>::from_array_type(output_types.remove(0)))
    };
    let primal = {
        let definitions = definitions.clone();
        move |inputs: AttentionInputs<RuleTracer<V, O>>| -> Result<RuleTracer<V, O>, ProgramError> {
            let mut outputs = definitions.forward(inputs, false)?;
            check_count!("output", outputs, 1, ProgramError);
            Ok(outputs.remove(0))
        }
    };
    let forward = {
        let definitions = definitions.clone();
        move |inputs: AttentionInputs<RuleTracer<V, O>>| -> Result<_, ProgramError> {
            let mut outputs = definitions.forward(inputs.clone(), true)?;
            check_count!("output", outputs, 2, ProgramError);
            let residual = outputs.remove(1);
            let output = outputs.remove(0);
            Ok((output.clone(), AttentionResiduals { inputs, output, residual }))
        }
    };
    let backward = move |residuals: AttentionResiduals<RuleTracer<V, O>>,
                         output_cotangent: MaybeZero<RuleTracer<V, O>>|
          -> Result<_, ProgramError> {
        let AttentionResiduals { inputs, output, residual } = residuals;
        let MaybeZero::Value(output_cotangent) = output_cotangent else {
            return inputs.try_map_parameters(|input| zero_cotangent(&input));
        };

        // The backward definition returns the cotangents of the query, key, and value, and of a bias whose cotangent
        // space is not zero-dimensional, so the cotangents of the other inputs (i.e., of the masks, sequence lengths,
        // and dropout states, and of a bias with a zero-dimensional cotangent space) are structural zeros, which
        // contribute nothing to their inputs' cotangents.
        let zero = |input: &Option<RuleTracer<V, O>>| input.as_ref().map(zero_cotangent).transpose();
        let mask = zero(&inputs.mask)?;
        let query_sequence_lengths = zero(&inputs.query_sequence_lengths)?;
        let key_value_sequence_lengths = zero(&inputs.key_value_sequence_lengths)?;
        let dropout_state = zero(&inputs.dropout_state)?;
        let bias_cotangent_type = inputs.bias.as_ref().map(|bias| bias.r#type().cotangent()).transpose()?;
        let mut outputs =
            definitions.backward(AttentionBackwardInputs { inputs, output, residual, output_cotangent })?;
        let bias = match bias_cotangent_type {
            Some(_) if outputs.len() == 4 => Some(MaybeZero::Value(outputs.remove(3))),
            Some(r#type) => Some(MaybeZero::Zero(r#type)),
            None => None,
        };
        check_count!("output", outputs, 3, ProgramError);
        let value = MaybeZero::Value(outputs.remove(2));
        let key = MaybeZero::Value(outputs.remove(1));
        let query = MaybeZero::Value(outputs.remove(0));
        Ok(AttentionInputs {
            query,
            key,
            value,
            bias,
            mask,
            query_sequence_lengths,
            key_value_sequence_lengths,
            dropout_state,
        })
    };
    composite(primal)
        .with_symbolic_zero_vjp(forward, backward)
        .with_name(DOT_PRODUCT_ATTENTION_TRAINING_COMPOSITE_NAME)
        .with_named_composite(
            attention_composite_key(DOT_PRODUCT_ATTENTION_TRAINING_COMPOSITE_NAME, configuration, signature, None),
            output_types,
        )
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use half::bf16;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, DataType, Dimension, Shape,
    };
    use crate::batching::{BatchAxis, batch};
    use crate::contexts::EagerContext;
    use crate::differentiation::Differentiate;
    use crate::operations::attention::definitions::attention_composite_key;
    use crate::operations::attention::dot_product_attention_ir_composition;
    use crate::operations::reductions::{Reduce, ReductionKind};
    use crate::tracing::Trace;

    use super::*;

    /// Eager array context in which the tests stage and differentiate attention.
    type ArrayContext = EagerContext<Array, ArrayOperation<Array>>;

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
                vec![("x".to_string(), NamedAxis::Mesh { mesh, axis: 0, size: 2, checks_manual_variation: true })],
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
        // The training function selects the portable implementation for automatic requests, which its named
        // composite's key records together with the configuration and the optional-input signature.
        let r#type = ArrayType::new_static(DataType::F32, [1, 1, 1, 1]);
        for (requested, selected) in [
            (AttentionImplementation::Automatic, AttentionImplementation::Portable),
            (AttentionImplementation::Portable, AttentionImplementation::Portable),
            (AttentionImplementation::Fused, AttentionImplementation::Fused),
        ] {
            let configuration = AttentionConfiguration::new().with_causal(true).with_implementation(requested);
            let function = differentiable_dot_product_attention::<ArrayContext>(configuration);
            assert_eq!(function.configuration().implementation(), selected);
            let (_, program) = ArrayContext::trace(
                |inputs| function.call(inputs),
                AttentionInputs::new(r#type.clone(), r#type.clone(), r#type.clone()),
            )
            .unwrap();
            let keys = program
                .instructions()
                .iter()
                .filter_map(|instruction| match instruction.operation() {
                    ArrayOperation::Composite(operation) => operation.key().cloned(),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(
                keys,
                vec![attention_composite_key(
                    DOT_PRODUCT_ATTENTION_TRAINING_COMPOSITE_NAME,
                    configuration.with_implementation(selected),
                    AttentionInputSignature::default(),
                    None,
                )],
            );
        }
    }

    #[test]
    fn test_differentiable_dot_product_attention_dropout() {
        // The backward definition applies the dropout mask of the forward definition, so the analytical gradient
        // agrees with the gradient of the portable forward body, whose dropout is differentiated automatically, up to
        // the rounding of their `f32` softmax computations.
        let state = Array::from_elements(ArrayType::new_static(DataType::U64, [2]), &[0_u64, 3]).unwrap();
        let inputs = AttentionInputs {
            dropout_state: Some(state),
            ..AttentionInputs::new(
                Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [1, 2, 1, 2]), &[0.1, -0.2, 0.3, 0.4])
                    .unwrap(),
                Array::from_elements::<f64>(
                    ArrayType::new_static(DataType::F64, [1, 3, 1, 2]),
                    &[0.5, 0.2, -0.3, 0.7, 0.1, -0.6],
                )
                .unwrap(),
                Array::from_elements::<f64>(
                    ArrayType::new_static(DataType::F64, [1, 3, 1, 2]),
                    &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                )
                .unwrap(),
            )
        };
        let configuration = AttentionConfiguration::new().with_dropout(0.5);
        let function = differentiable_dot_product_attention::<ArrayContext>(configuration);
        let (output, gradients) = ArrayContext::new()
            .differentiate_at(inputs.clone())
            .value_and_gradient(|inputs| function.call(inputs)?.reduce(&[0, 1, 2, 3], ReductionKind::Sum))
            .unwrap();
        let (expected_output, expected_gradients) = ArrayContext::new()
            .differentiate_at(inputs)
            .value_and_gradient(|inputs| {
                let (output, _) = dot_product_attention_ir_composition(&inputs, configuration)?;
                output.reduce(&[0, 1, 2, 3], ReductionKind::Sum)
            })
            .unwrap();
        assert_abs_diff_eq!(output.to_f64s()[0], expected_output.to_f64s()[0], epsilon = 1e-6);
        for (actual, expected) in [
            (gradients.query, expected_gradients.query),
            (gradients.key, expected_gradients.key),
            (gradients.value, expected_gradients.value),
        ] {
            actual
                .to_f64s()
                .iter()
                .zip(expected.to_f64s())
                .for_each(|(actual, expected)| assert_abs_diff_eq!(*actual, expected, epsilon = 1e-6));
        }
        assert_eq!(gradients.dropout_state.unwrap().r#type().data_type(), DataType::Zero);
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
            dropout_state: None,
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
            dropout_state: None,
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
    fn test_differentiable_dot_product_attention_batching() {
        // Batched differentiation folds the mapped axis into the forward and backward definitions, including the
        // reduction of the bias cotangent over the axes that its bias broadcasts, so the batched gradients equal the
        // gradients of every batch item.
        let query = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.2, 0.1];
        let key = [0.5, 0.2, -0.3, 0.7, 0.1, -0.6, 0.4, -0.1, 0.3, 0.2, -0.4, 0.5];
        let value = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, 0.5, 2.5, -3.0, 1.5, 0.0];
        let bias = [0.25, -0.5, 1.0, -0.75, 0.5, 0.0];
        let array = |dimensions: &[usize], elements: &[f64]| {
            Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, dimensions.to_vec()), elements).unwrap()
        };
        let function = differentiable_dot_product_attention::<ArrayContext>(AttentionConfiguration::new());
        let gradient = |inputs: AttentionInputs<Array>| {
            ArrayContext::new()
                .differentiate_at(inputs)
                .gradient(|inputs| function.call(inputs)?.reduce(&[0, 1, 2, 3], ReductionKind::Sum))
                .unwrap()
        };
        let inputs = AttentionInputs {
            bias: Some(array(&[2, 3], &bias)),
            ..AttentionInputs::new(
                array(&[2, 1, 2, 1, 2], &query),
                array(&[2, 1, 3, 1, 2], &key),
                array(&[2, 1, 3, 1, 2], &value),
            )
        };
        let axes = AttentionInputs {
            bias: Some(BatchAxis::new(0)),
            ..AttentionInputs::new(BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(0))
        };
        let batched = ArrayContext::new()
            .differentiate_at(inputs)
            .gradient(|inputs| {
                let outputs = batch(|inputs| function.call(inputs), inputs, axes.clone(), BatchAxis::new(0), None)
                    .map_err(ProgramError::from)?;
                outputs.reduce(&[0, 1, 2, 3, 4], ReductionKind::Sum)
            })
            .unwrap();
        for item in 0..2 {
            let expected = gradient(AttentionInputs {
                bias: Some(array(&[3], &bias[3 * item..3 * item + 3])),
                ..AttentionInputs::new(
                    array(&[1, 2, 1, 2], &query[4 * item..4 * item + 4]),
                    array(&[1, 3, 1, 2], &key[6 * item..6 * item + 6]),
                    array(&[1, 3, 1, 2], &value[6 * item..6 * item + 6]),
                )
            });
            for (actual, expected, size) in [
                (&batched.query, &expected.query, 4),
                (&batched.key, &expected.key, 6),
                (&batched.value, &expected.value, 6),
                (batched.bias.as_ref().unwrap(), expected.bias.as_ref().unwrap(), 3),
            ]
            .map(|(actual, expected, size)| (actual.to_f64s(), expected.to_f64s(), size))
            {
                actual[size * item..size * (item + 1)]
                    .iter()
                    .zip(expected)
                    .for_each(|(actual, expected)| assert_abs_diff_eq!(*actual, expected, epsilon = 1e-6));
            }
        }
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
                dropout_state: None,
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
