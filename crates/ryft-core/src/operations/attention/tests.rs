use approx::assert_abs_diff_eq;
use indoc::indoc;
use pretty_assertions::assert_eq;

use crate::arrays::{Array, ArrayIrOperation, ArrayIrValue, DimensionBounds, DimensionVariable};
use crate::batching::batch;
use crate::contexts::StagingContext;
use crate::parameters::Placeholder;
use crate::tracing::TracingContext;

use super::*;

#[test]
fn test_attention_inputs() {
    let signature = AttentionOperandSignature::new(true, true, true, true);
    let values = (0..7).map(|size| ArrayType::new_static(DataType::F32, [size])).collect::<Vec<_>>();
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
    let query = ArrayType::new_static(DataType::F32, [2, 2, 2]);
    let key_value = ArrayType::new_static(DataType::F32, [3, 1, 2]);
    let bias = ArrayType::new_static(DataType::F64, [3]);
    let mask = ArrayType::new_static(DataType::Boolean, [2, 3]);
    let lengths = ArrayType::new_static(DataType::I32, [1]);
    let residual = ArrayType::new_static(DataType::F32, [2, 2]);
    let signature = AttentionOperandSignature::new(true, true, true, true);
    let configuration = AttentionConfiguration::new().with_local_window((0, 0)).with_residual(true);
    let operation = DotProductAttentionOperation::new(configuration, signature);
    let input_types =
        vec![query.clone(), key_value.clone(), key_value.clone(), bias.clone(), mask, lengths.clone(), lengths];

    assert_eq!(operation.infer_output_types(input_types.as_slice(), &[]), Ok(vec![query.clone(), residual.clone()]));
    let mut backward_input_types = input_types;
    backward_input_types.extend([query.clone(), residual, query.clone()]);
    assert_eq!(
        DotProductAttentionBackwardOperation::new(configuration, signature)
            .infer_output_types(backward_input_types.as_slice(), &[]),
        Ok(vec![query, key_value.clone(), key_value, bias]),
    );

    // A non-differentiable bias remains a legal forward input but contributes no live backward result.
    let query = ArrayType::new_static(DataType::F32, [2, 2, 2]);
    let key_value = ArrayType::new_static(DataType::F32, [3, 1, 2]);
    let bias = ArrayType::new_static(DataType::I32, [3]);
    let residual = ArrayType::new_static(DataType::F32, [2, 2]);
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
            if message == "`dot_product_attention` dropout requires the fused implementation",
    ));
}

#[test]
fn test_dot_product_attention() {
    // Rank-three operands normalize through an implicit batch. The arbitrary mask, asymmetric local window, and
    // query lengths compose independently, while the omitted scale defaults to `1 / sqrt(head_dimension)`.
    let query = Array::from_f64s(ArrayType::new_static(DataType::F32, [2, 1, 2]), vec![1.0, 0.0, 0.0, 1.0]);
    let key = Array::from_f64s(ArrayType::new_static(DataType::F32, [3, 1, 2]), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let value = Array::from_f64s(ArrayType::new_static(DataType::F32, [3, 1, 2]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mask =
        Array::from_elements(ArrayType::new_static(DataType::Boolean, [2, 3]), &[true, false, true, true, true, false])
            .unwrap();
    let bias = Array::from_elements(ArrayType::new_static(DataType::I32, []), &[0_i32]).unwrap();
    let query_lengths = Array::from_elements(ArrayType::new_static(DataType::I32, [1]), &[1_i32]).unwrap();
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
    let query = Array::from_f64s(ArrayType::new_static(DataType::F64, [1, 1, 1]), vec![1.0e8]);
    let key = Array::from_f64s(ArrayType::new_static(DataType::F64, [2, 1, 1]), vec![1.0, 1.0 + 1.0e-8]);
    let value = Array::from_f64s(ArrayType::new_static(DataType::F64, [2, 1, 1]), vec![0.0, 1.0]);
    let bias = Array::from_f64s(ArrayType::new_static(DataType::F64, [2]), vec![-1.0e8, -1.0e8]);
    let output = Array::dot_product_attention(
        AttentionInputs { bias: Some(bias), ..AttentionInputs::new(query, key, value) },
        AttentionConfiguration::new().with_scale(1.0),
    )
    .unwrap()
    .0;

    assert!((output.to_f64s()[0] - 0.731_058_6).abs() < 1.0e-6);

    // MQA and its explicitly repeated MHA representation are semantically identical.
    let query = Array::from_f64s(ArrayType::new_static(DataType::F32, [1, 1, 2, 1]), vec![1.0, 1.0]);
    let grouped_key = Array::from_f64s(ArrayType::new_static(DataType::F32, [1, 2, 1, 1]), vec![1.0, 2.0]);
    let grouped_value = Array::from_f64s(ArrayType::new_static(DataType::F32, [1, 2, 1, 1]), vec![10.0, 20.0]);
    let repeated_key = Array::from_f64s(ArrayType::new_static(DataType::F32, [1, 2, 2, 1]), vec![1.0, 1.0, 2.0, 2.0]);
    let repeated_value =
        Array::from_f64s(ArrayType::new_static(DataType::F32, [1, 2, 2, 1]), vec![10.0, 10.0, 20.0, 20.0]);
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
    let r#type = ArrayType::new_static(DataType::F32, [2, 1, 1, 1, 1]);
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
        Array::from_f64s(ArrayType::new_static(DataType::F32, [2, 1, 1, 1]), vec![1.0, 1.0]),
        BatchAxis::new(0),
    )
    .unwrap();
    let key = ArrayBatch::new(
        Array::from_f64s(ArrayType::new_static(DataType::F32, [2, 2, 1, 1]), vec![1.0, 2.0, 1.0, 2.0]),
        BatchAxis::new(0),
    )
    .unwrap();
    let value = ArrayBatch::new(
        Array::from_f64s(ArrayType::new_static(DataType::F32, [2, 2, 1, 1]), vec![3.0, 9.0, 4.0, 10.0]),
        BatchAxis::new(0),
    )
    .unwrap();
    let mask = ArrayBatch::new(
        Array::from_elements(ArrayType::new_static(DataType::Boolean, [2, 1, 2]), &[true, false, false, true]).unwrap(),
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
