use std::collections::HashMap;

use approx::assert_abs_diff_eq;
use indoc::indoc;
use pretty_assertions::assert_eq;

use crate::arrays::{
    Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, DimensionBounds, DimensionVariable,
    LogicalMesh, MeshAxis, MeshAxisType, Sharding,
};
use crate::axes::NamedAxis;
use crate::batching::batch;
use crate::contexts::{EagerContext, StagingContext};
use crate::differentiation::{Differentiate, DifferentiationError};
use crate::operations::composites::{CompositeAttribute, CompositeKey};
use crate::parameters::Placeholder;
use crate::tests::hash_of;
use crate::tracing::{DomainTracingContext, Trace, TracingContext};

use super::*;

/// Eager array context in which the tests stage and evaluate attention.
type ArrayContext = EagerContext<Array, ArrayOperation<Array>>;

/// Eager array IR context in which the tests stage and evaluate attention with first-class extents.
type IrContext = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

/// Returns an `f32` array with the provided dimensions and elements.
fn f32_array(dimensions: &[usize], elements: &[f32]) -> Array {
    Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, dimensions.to_vec()), elements).unwrap()
}

/// Returns the attended output of one eager attention call with the provided configuration.
fn attend(inputs: AttentionInputs<Array>, configuration: AttentionConfiguration) -> (Array, Option<Array>) {
    dot_product_attention::<ArrayContext>(configuration).call(inputs).unwrap()
}

#[test]
fn test_attention_configuration() {
    let configuration = AttentionConfiguration::new()
        .with_scale(0.25)
        .with_causal(true)
        .with_local_window((2, 1))
        .with_implementation(AttentionImplementation::Fused)
        .with_residual(true)
        .with_dropout(0.1);

    assert_eq!(configuration.scale(), Some(0.25));
    assert!(configuration.causal());
    assert_eq!(configuration.local_window(), Some((2, 1)));
    assert_eq!(configuration.implementation(), AttentionImplementation::Fused);
    assert!(configuration.return_residual());
    assert_eq!(configuration.dropout(), Some(0.1));
    assert_eq!(configuration.with_dropout(None).dropout(), None);
    assert_eq!(AttentionConfiguration::new().with_symmetric_local_window(3).local_window(), Some((3, 3)));
}

#[test]
fn test_attention_configuration_identity() {
    // Equal configurations are equal and hash identically, so they work as map keys.
    let configuration = AttentionConfiguration::new().with_scale(0.25).with_causal(true).with_dropout(0.1);
    let same_configuration = AttentionConfiguration::new().with_scale(0.25).with_causal(true).with_dropout(0.1);
    assert_eq!(configuration, same_configuration);
    assert_eq!(hash_of(&configuration), hash_of(&same_configuration));
    let configurations = HashMap::from([(configuration, "configuration")]);
    assert_eq!(configurations.get(&same_configuration), Some(&"configuration"));
    assert_ne!(configuration, configuration.with_causal(false));

    // The scale and the dropout rate compare bitwise, so signed zeros are distinct even though `-0.0 == 0.0`.
    assert_ne!(AttentionConfiguration::new().with_scale(-0.0), AttentionConfiguration::new().with_scale(0.0));
    assert_ne!(AttentionConfiguration::new().with_scale(0.0), AttentionConfiguration::new());
    assert_ne!(configuration, configuration.with_dropout(0.2));
    assert_ne!(configuration, configuration.with_dropout(None));
    assert_ne!(AttentionConfiguration::new().with_dropout(-0.0), AttentionConfiguration::new().with_dropout(0.0));
    assert_eq!(configurations.get(&configuration.with_dropout(0.2)), None);
}

#[test]
fn test_attention_inputs() {
    let signature = AttentionInputSignature::new(true, true, true, true).with_dropout_state(true);
    let values = (0..8).map(|size| ArrayType::new_static(DataType::F32, [size])).collect::<Vec<_>>();
    let inputs = AttentionInputs::from_values(signature, values.as_slice()).unwrap();

    assert_eq!(inputs.signature(), signature);
    assert!(signature.has_dropout_state());
    assert_eq!(signature.count(), 5);
    assert_eq!(inputs.dropout_state, Some(values[7].clone()));
    assert_eq!(inputs.into_values(), values);
    assert!(matches!(
        AttentionInputs::from_values(signature, &values[..7]),
        Err(TypeError::Invalid { message, .. })
            if message == "attention input signature expects 8 values but got 7",
    ));

    // Backward inputs append the forward output, its residual, and the output cotangent to the forward inputs.
    let signature = AttentionInputSignature::new(false, true, false, false);
    let values = (0..7).map(|size| ArrayType::new_static(DataType::F32, [size])).collect::<Vec<_>>();
    let inputs = AttentionBackwardInputs::from_values(signature, values.as_slice()).unwrap();
    assert_eq!(inputs.inputs.signature(), signature);
    assert_eq!(inputs.output, values[4]);
    assert_eq!(inputs.residual, values[5]);
    assert_eq!(inputs.output_cotangent, values[6]);
    assert_eq!(inputs.into_values(), values);
    assert!(matches!(
        AttentionBackwardInputs::from_values(signature, &values[..6]),
        Err(TypeError::Invalid { message, .. })
            if message == "attention backward inputs for signature AttentionInputSignature { bias: false, mask: true, \
                           query_sequence_lengths: false, key_value_sequence_lengths: false, dropout_state: false } \
                           have 7 values but got 6",
    ));
}

#[test]
fn test_attention_key() {
    let configuration = AttentionConfiguration::new()
        .with_scale(0.125)
        .with_causal(true)
        .with_local_window((2, 0))
        .with_implementation(AttentionImplementation::Fused)
        .with_dropout(0.25)
        .with_residual(true);
    let signature = AttentionInputSignature::new(true, false, true, false).with_dropout_state(true);

    // Forward keys encode every configuration attribute, including the residual request.
    let key = AttentionKey::new(AttentionDefinitionKind::Forward, configuration, signature);
    let composite_key = key.to_composite_key();
    assert_eq!(composite_key.name(), DOT_PRODUCT_ATTENTION_COMPOSITE_NAME);
    assert_eq!(composite_key.version(), DOT_PRODUCT_ATTENTION_COMPOSITE_VERSION);
    assert_eq!(
        composite_key.to_string(),
        "ryft.dot_product_attention.v1 {scale = some(0.125), causal = true, local_window = some([2, 0]), \
         implementation = 2, dropout_rate = some(0.25), residual = true, bias = true, mask = false, \
         query_sequence_lengths = true, key_value_sequence_lengths = false, dropout_state = true}",
    );
    assert_eq!(AttentionKey::from_composite_key(&composite_key), Ok(Some(key)));

    // Backward keys ignore the residual request, so they omit its attribute.
    let key = AttentionKey::new(AttentionDefinitionKind::Backward, configuration, signature);
    assert!(!key.configuration().return_residual());
    assert_eq!(key.configuration(), configuration.with_residual(false));
    let composite_key = key.to_composite_key();
    assert_eq!(composite_key.name(), DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME);
    assert!(composite_key.attributes().iter().all(|(name, _)| name != "residual"));
    assert_eq!(AttentionKey::from_composite_key(&composite_key), Ok(Some(key)));

    // Every default configuration round-trips, including the absent optional attributes.
    for kind in [AttentionDefinitionKind::Forward, AttentionDefinitionKind::Backward] {
        for implementation in
            [AttentionImplementation::Automatic, AttentionImplementation::Portable, AttentionImplementation::Fused]
        {
            let key = AttentionKey::new(
                kind,
                AttentionConfiguration::new().with_implementation(implementation),
                AttentionInputSignature::default(),
            );
            assert_eq!(AttentionKey::from_composite_key(&key.to_composite_key()), Ok(Some(key)));
        }
    }

    // Keys of other composites are not attention keys, while malformed attention keys are rejected.
    assert_eq!(AttentionKey::from_composite_key(&CompositeKey::new("ryft.other", 1)), Ok(None));
    let forward_key = AttentionKey::new(
        AttentionDefinitionKind::Forward,
        AttentionConfiguration::new(),
        AttentionInputSignature::default(),
    )
    .to_composite_key();
    let wrong_version = CompositeKey::new(DOT_PRODUCT_ATTENTION_COMPOSITE_NAME, 2);
    assert_eq!(
        AttentionKey::from_composite_key(&wrong_version),
        Err(TypeError::invalid(format!(
            "named composite `{wrong_version}` has attention definition version 2, but only version 1 is supported",
        ))),
    );
    let extended = forward_key.clone().with_attribute("extra", true);
    assert!(matches!(
        AttentionKey::from_composite_key(&extended),
        Err(TypeError::Invalid { message, .. }) if message.ends_with("is not a valid attention definition"),
    ));
    let malformed = forward_key.attributes().iter().fold(
        CompositeKey::new(DOT_PRODUCT_ATTENTION_COMPOSITE_NAME, 1),
        |key, (name, value)| {
            let value = if name == "implementation" { CompositeAttribute::Integer(3) } else { value.clone() };
            key.with_attribute(name.clone(), value)
        },
    );
    assert!(matches!(
        AttentionKey::from_composite_key(&malformed),
        Err(TypeError::Invalid { message, .. }) if message.ends_with("is not a valid attention definition"),
    ));
    let missing = CompositeKey::new(DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME, 1).with_attribute("causal", false);
    assert!(matches!(
        AttentionKey::from_composite_key(&missing),
        Err(TypeError::Invalid { message, .. }) if message.ends_with("is not a valid attention definition"),
    ));
}

#[test]
fn test_attention_output_types() {
    let query = ArrayType::new_static(DataType::F32, [2, 2, 2]);
    let key_value = ArrayType::new_static(DataType::F32, [3, 1, 2]);
    let bias = ArrayType::new_static(DataType::F64, [3]);
    let mask = ArrayType::new_static(DataType::Boolean, [2, 3]);
    let lengths = ArrayType::new_static(DataType::I32, [1]);
    let residual = ArrayType::new_static(DataType::F32, [2, 2]);
    let signature = AttentionInputSignature::new(true, true, true, true);
    let configuration = AttentionConfiguration::new().with_local_window((0, 0)).with_residual(true);
    let input_types =
        vec![query.clone(), key_value.clone(), key_value.clone(), bias.clone(), mask, lengths.clone(), lengths];
    let name = DOT_PRODUCT_ATTENTION_COMPOSITE_NAME;
    let backward_name = DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME;

    assert_eq!(
        attention_forward_output_types(name, configuration, signature, input_types.as_slice()),
        Ok(vec![query.clone(), residual.clone()]),
    );
    assert_eq!(
        attention_forward_output_types(name, configuration.with_residual(false), signature, input_types.as_slice()),
        Ok(vec![query.clone()]),
    );
    let mut backward_input_types = input_types.clone();
    backward_input_types.extend([query.clone(), residual.clone(), query.clone()]);
    assert_eq!(
        attention_backward_output_types(backward_name, configuration, signature, backward_input_types.as_slice()),
        Ok(vec![query.clone(), key_value.clone(), key_value.clone(), bias]),
    );

    // A non-differentiable bias remains a legal forward input but contributes no backward output.
    let structural_bias_signature = AttentionInputSignature::new(true, false, false, false);
    let integer_bias = ArrayType::new_static(DataType::I32, [3]);
    let backward_input_types = vec![
        query.clone(),
        key_value.clone(),
        key_value.clone(),
        integer_bias,
        query.clone(),
        residual.clone(),
        query.clone(),
    ];
    assert_eq!(
        attention_backward_output_types(
            backward_name,
            configuration,
            structural_bias_signature,
            backward_input_types.as_slice(),
        ),
        Ok(vec![query.clone(), key_value.clone(), key_value.clone()]),
    );

    // Dropout requires a rate in the open unit interval and a valid random state when a state is present.
    let plain_types = vec![query.clone(), key_value.clone(), key_value.clone()];
    let state = ArrayType::new_static(DataType::U64, [2]);
    let stateful = AttentionInputSignature::default().with_dropout_state(true);
    let mut stateful_types = plain_types.clone();
    stateful_types.push(state.clone());
    assert_eq!(
        attention_forward_output_types(
            name,
            AttentionConfiguration::new().with_dropout(0.5),
            stateful,
            &stateful_types
        ),
        Ok(vec![query.clone()]),
    );
    assert_eq!(
        attention_forward_output_types(
            name,
            AttentionConfiguration::new().with_dropout(0.5),
            AttentionInputSignature::default(),
            &plain_types,
        ),
        Ok(vec![query.clone()]),
    );
    for rate in [0.0, 1.0, -0.5, f64::NAN] {
        assert!(matches!(
            attention_forward_output_types(name, AttentionConfiguration::new().with_dropout(rate), stateful, &stateful_types),
            Err(TypeError::Invalid { message, .. })
                if message.starts_with("`ryft.dot_product_attention` dropout rate must lie in the open interval (0, 1)"),
        ));
    }
    assert!(matches!(
        attention_forward_output_types(name, AttentionConfiguration::new(), stateful, &stateful_types),
        Err(TypeError::Invalid { message, .. })
            if message == "`ryft.dot_product_attention` received a dropout random state without a dropout rate",
    ));
    let mut invalid_state_types = plain_types;
    invalid_state_types.push(ArrayType::new_static(DataType::F32, [2]));
    assert!(matches!(
        attention_forward_output_types(
            name,
            AttentionConfiguration::new().with_dropout(0.5),
            stateful,
            &invalid_state_types,
        ),
        Err(TypeError::Invalid { .. }),
    ));

    // Inputs inside manual regions must have aligned manual variation, and reduced inputs are rejected.
    let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
    let invariant = ArrayType::new_static(DataType::F32, [1, 1, 1, 1])
        .with_sharding(Sharding::replicated(mesh.clone(), 4))
        .unwrap();
    let varying = invariant
        .clone()
        .with_sharding(Sharding::replicated(mesh.clone(), 4).with_varying_manual_axes(["x"]).unwrap())
        .unwrap();
    let reduced = invariant
        .clone()
        .with_sharding(Sharding::replicated(mesh, 4).with_reduced_axes(["x"]).unwrap())
        .unwrap();
    assert!(matches!(
        attention_forward_output_types(
            name,
            AttentionConfiguration::new(),
            AttentionInputSignature::default(),
            &[varying, invariant.clone(), invariant.clone()],
        ),
        Err(TypeError::Invalid { message, .. })
            if message == "`ryft.dot_product_attention` inputs must have matching varying manual axes; insert \
                           `parallel_vary` on the inputs that lack an axis, as `align_manual_variation` does",
    ));
    assert!(matches!(
        attention_forward_output_types(
            name,
            AttentionConfiguration::new(),
            AttentionInputSignature::default(),
            &[reduced, invariant.clone(), invariant],
        ),
        Err(TypeError::Invalid { message, .. })
            if message == "`ryft.dot_product_attention` does not support reduced inputs",
    ));
}

#[test]
fn test_dot_product_attention() {
    let r#type = ArrayType::new_static(DataType::F32, [1, 2, 1, 1]);
    let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new().with_causal(true));
    assert!(function.configuration().causal());

    // Calls with equal input types share one traced body, which the program imports once.
    let (_, program) = ArrayContext::trace(
        |inputs: AttentionInputs<_>| {
            let (first, _) = function.call(inputs.clone())?;
            let (second, _) = function.call(AttentionInputs::new(first, inputs.key, inputs.value))?;
            Ok(second)
        },
        AttentionInputs::new(r#type.clone(), r#type.clone(), r#type.clone()),
    )
    .unwrap();
    assert_eq!(
        program.to_string(),
        indoc! {"
            lambda %0:f32[1, 2, 1, 1], %1:f32[1, 2, 1, 1], %2:f32[1, 2, 1, 1] .
            let %3:f32[1, 2, 1, 1] = composite [
                key=\"ryft.dot_product_attention.v1 {scale = none, causal = true, local_window = none, implementation = 0, dropout_rate = none, residual = false, bias = false, mask = false, query_sequence_lengths = false, key_value_sequence_lengths = false, dropout_state = false}\",
            ] %0 %1 %2 [
                primal=^0={
                    lambda %0:f32[1, 2, 1, 1], %1:f32[1, 2, 1, 1], %2:f32[1, 2, 1, 1] .
                    let %3:f32[1, 1, 2, 2] = dot [
                        dimensions=(lhs_contracting=[3], rhs_contracting=[3], lhs_batching=[0, 2], rhs_batching=[0, 2]),
                    ] %0 %1
                        %4:f32[] = constant [value=1.0]
                        %5:f32[1, 1, 2, 2] = broadcast [output_type=f32[1, 1, 2, 2], output_axes=[]] %4
                        %6:f32[1, 1, 2, 2] = mul %3 %5
                        %7:i32[1, 1, 2, 2] = iota [type=i32[1, 1, 2, 2], dimension=3]
                        %8:i32[1, 1, 2, 2] = iota [type=i32[1, 1, 2, 2], dimension=2]
                        %9:bool[1, 1, 2, 2] = compare [direction=LessThanOrEqual] %7 %8
                        %10:f32[] = constant [value=-2.3819763e38]
                        %11:f32[1, 1, 2, 2] = broadcast [output_type=f32[1, 1, 2, 2], output_axes=[]] %10
                        %12:f32[1, 1, 2, 2] = select %9 %6 %11
                        %13:f32[1, 1, 2] = reduce [kind=max, axes=[3]] %12
                        %14:f32[1, 1, 2, 2] = broadcast [output_type=f32[1, 1, 2, 2], output_axes=[0, 1, 2]] %13
                        %15:f32[1, 1, 2, 2] = sub %12 %14
                        %16:f32[1, 1, 2, 2] = exp %15
                        %17:f32[1, 1, 2] = reduce [kind=sum, axes=[3]] %16
                        %18:f32[1, 1, 2, 2] = broadcast [output_type=f32[1, 1, 2, 2], output_axes=[0, 1, 2]] %17
                        %19:f32[1, 1, 2, 2] = div %16 %18
                        %20:f32[1, 1, 2, 1] = dot [
                            dimensions=(lhs_contracting=[3], rhs_contracting=[1], lhs_batching=[0, 1], rhs_batching=[0, 2]),
                        ] %19 %2
                        %21:f32[1, 2, 1, 1] = transpose [permutation=[0, 2, 1, 3]] %20
                    in (%21)
                },
            ]
                %4:f32[1, 2, 1, 1] = composite [
                    key=\"ryft.dot_product_attention.v1 {scale = none, causal = true, local_window = none, implementation = 0, dropout_rate = none, residual = false, bias = false, mask = false, query_sequence_lengths = false, key_value_sequence_lengths = false, dropout_state = false}\",
                ] %3 %1 %2 [
                    primal=^0,
                ]
            in (%4)
        "}
        .trim_end(),
    );
}

#[test]
fn test_dot_product_attention_interpretation() {
    // Rank-three query, key, and value inputs normalize through an implicit batch. The arbitrary mask, asymmetric
    // local window, and query lengths compose independently, while the omitted scale defaults to
    // `1 / sqrt(head_dimension)`.
    let inputs = AttentionInputs {
        bias: Some(Array::from_elements(ArrayType::new_static(DataType::I32, []), &[0_i32]).unwrap()),
        mask: Some(
            Array::from_elements(
                ArrayType::new_static(DataType::Boolean, [2, 3]),
                &[true, false, true, true, true, false],
            )
            .unwrap(),
        ),
        query_sequence_lengths: Some(
            Array::from_elements(ArrayType::new_static(DataType::I32, [1]), &[1_i32]).unwrap(),
        ),
        ..AttentionInputs::new(
            f32_array(&[2, 1, 2], &[1.0, 0.0, 0.0, 1.0]),
            f32_array(&[3, 1, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            f32_array(&[3, 1, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )
    };
    let configuration = AttentionConfiguration::new().with_local_window((1, 0)).with_residual(true);
    let (output, residual) = attend(inputs.clone(), configuration);
    let (explicit, _) = attend(inputs, configuration.with_scale(Some(1.0 / 2.0_f64.sqrt())));

    assert_eq!(output.r#type().shape(), &static_shape(&[2, 1, 2]));
    let residual = residual.unwrap();
    assert_eq!(residual.r#type().shape(), &static_shape(&[2, 1]));
    assert_eq!(output.to_f64s(), explicit.to_f64s());
    assert_eq!(&output.to_f64s()[2..], &[0.0, 0.0]);
    assert!(residual.to_f64s()[1] < -1.0e30);

    // Float64 dot products, scaling, and bias addition retain float64 precision until the logits reach the
    // explicitly float32 softmax. Subtracting the large bias before that conversion preserves the unit gap.
    let query = Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [1, 1, 1]), &[1.0e8]).unwrap();
    let key =
        Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2, 1, 1]), &[1.0, 1.0 + 1.0e-8]).unwrap();
    let value = Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2, 1, 1]), &[0.0, 1.0]).unwrap();
    let bias = Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2]), &[-1.0e8, -1.0e8]).unwrap();
    let (output, _) = attend(
        AttentionInputs { bias: Some(bias), ..AttentionInputs::new(query, key, value) },
        AttentionConfiguration::new().with_scale(1.0),
    );
    assert!((output.to_f64s()[0] - 0.731_058_6).abs() < 1.0e-6);

    // MQA and its explicitly repeated MHA representation are semantically identical.
    let query = f32_array(&[1, 1, 2, 1], &[1.0, 1.0]);
    let (grouped, _) = attend(
        AttentionInputs::new(
            query.clone(),
            f32_array(&[1, 2, 1, 1], &[1.0, 2.0]),
            f32_array(&[1, 2, 1, 1], &[10.0, 20.0]),
        ),
        AttentionConfiguration::new(),
    );
    let (repeated, _) = attend(
        AttentionInputs::new(
            query,
            f32_array(&[1, 2, 2, 1], &[1.0, 1.0, 2.0, 2.0]),
            f32_array(&[1, 2, 2, 1], &[10.0, 10.0, 20.0, 20.0]),
        ),
        AttentionConfiguration::new(),
    );
    grouped
        .to_f64s()
        .iter()
        .zip(repeated.to_f64s())
        .for_each(|(actual, expected)| assert_abs_diff_eq!(actual, &expected, epsilon = 1e-6));

    // Invalid inputs are rejected by the definition's output type rule before its body is traced.
    let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new());
    assert!(matches!(
        function.call(AttentionInputs::new(
            f32_array(&[1, 1, 2], &[1.0, 1.0]),
            f32_array(&[1, 1, 1], &[1.0]),
            f32_array(&[1, 1, 1], &[1.0]),
        )),
        Err(ProgramError::Type(TypeError::Invalid { message, .. })) if message.starts_with("`ryft.dot_product_attention`"),
    ));
}

#[test]
fn test_dot_product_attention_array_ir() {
    // Array IR values compute the same attention as their array members.
    let query = f32_array(&[2, 1, 2], &[1.0, 0.0, 0.0, 1.0]);
    let key = f32_array(&[3, 1, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let value = f32_array(&[3, 1, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let configuration = AttentionConfiguration::new().with_residual(true);
    let (output, residual) = attend(AttentionInputs::new(query.clone(), key.clone(), value.clone()), configuration);
    let function = dot_product_attention::<IrContext>(configuration);
    assert_eq!(
        function.call(AttentionInputs::new(
            ArrayIrValue::Array(query),
            ArrayIrValue::Array(key),
            ArrayIrValue::Array(value),
        )),
        Ok((ArrayIrValue::Array(output), residual.map(ArrayIrValue::Array))),
    );
}

#[test]
fn test_dot_product_attention_manual_variation() {
    let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
    let invariant = ArrayType::new_static(DataType::F32, [1, 1, 1, 1])
        .with_sharding(Sharding::replicated(mesh.clone(), 4))
        .unwrap();
    let varying = invariant
        .clone()
        .with_sharding(Sharding::replicated(mesh.clone(), 4).with_varying_manual_axes(["x"]).unwrap())
        .unwrap();
    let lengths = ArrayType::new_static(DataType::I32, [1])
        .with_sharding(Sharding::replicated(mesh.clone(), 1))
        .unwrap();
    let mask = ArrayType::new_static(DataType::Boolean, [1, 1, 1, 1])
        .with_sharding(Sharding::replicated(mesh.clone(), 4))
        .unwrap();
    let signature = AttentionInputSignature::new(true, true, true, true);
    let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new().with_residual(true));

    // The function aligns the manual variation of the inputs before its call, so invariant inputs become varying.
    let (outputs, program) = DomainTracingContext::<ArrayContext>::trace_with_named_axes(
        |inputs: Vec<_>| {
            let (output, residual) = function.call(AttentionInputs::from_values(signature, &inputs)?)?;
            Ok(vec![output, residual.unwrap()])
        },
        vec![varying.clone(), invariant.clone(), invariant.clone(), invariant, mask, lengths.clone(), lengths],
        vec![(
            "x".to_string(),
            NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2, checks_manual_variation: true },
        )],
    )
    .unwrap();
    let residual = ArrayType::new_static(DataType::F32, [1, 1, 1])
        .with_sharding(Sharding::replicated(mesh, 3).with_varying_manual_axes(["x"]).unwrap())
        .unwrap();
    assert_eq!(outputs, vec![varying, residual]);
    assert_eq!(
        program.instructions().iter().map(|instruction| instruction.operation().name()).collect::<Vec<_>>(),
        vec![
            "parallel_vary",
            "parallel_vary",
            "parallel_vary",
            "parallel_vary",
            "parallel_vary",
            "parallel_vary",
            "composite",
        ],
    );
}

#[test]
fn test_dot_product_attention_differentiation() {
    // The named definitions have no derivative rules, so the inference function rejects differentiation.
    let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new());
    let r#type = ArrayType::new_static(DataType::F32, [1, 1, 1, 1]);
    let inputs = AttentionInputs::new(
        Array::from_elements::<f32>(r#type.clone(), &[1.0]).unwrap(),
        Array::from_elements::<f32>(r#type.clone(), &[1.0]).unwrap(),
        Array::from_elements::<f32>(r#type, &[1.0]).unwrap(),
    );
    let result = ArrayContext::new()
        .differentiate_at(inputs)
        .gradient(|inputs| function.call(inputs)?.0.reduce(&[0, 1, 2, 3], ReductionKind::Sum));
    assert!(matches!(
        result,
        Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
            if message == "cannot differentiate a `composite` call of `ryft.dot_product_attention` that has no \
                           derivative rule",
    ));
}

#[test]
fn test_dot_product_attention_batching() {
    let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new());

    // Each mapped example carries one complete rank-four attention problem. The batching rule folds the mapped axis
    // into the logical batch axis and restores it on the output.
    let ones = f32_array(&[2, 1, 1, 1, 1], &[1.0, 1.0]);
    let values = f32_array(&[2, 1, 1, 1, 1], &[5.0, 7.0]);
    let output = batch(
        |inputs: AttentionInputs<_>| Ok(function.call(inputs)?.0),
        AttentionInputs::new(ones.clone(), ones, values),
        AttentionInputs::new(BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(0)),
        BatchAxis::new(0),
        None,
    )
    .unwrap();
    assert_eq!(output.to_f64s(), vec![5.0, 7.0]);

    // Rank-three attention has an implicit logical batch of one. A mapped rank-two mask is normalized alongside the
    // query, key, and value rather than being mistaken for a tensor batch prefix, and a replicated value is
    // broadcast along the mapped axis.
    let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new().with_residual(true));
    let inputs = AttentionInputs {
        mask: Some(
            Array::from_elements(ArrayType::new_static(DataType::Boolean, [2, 1, 2]), &[true, false, false, true])
                .unwrap(),
        ),
        ..AttentionInputs::new(
            f32_array(&[2, 1, 1, 1], &[1.0, 1.0]),
            f32_array(&[2, 2, 1, 1], &[1.0, 2.0, 1.0, 2.0]),
            f32_array(&[2, 1, 1], &[3.0, 10.0]),
        )
    };
    let axes = AttentionInputs {
        mask: Some(BatchAxis::new(0)),
        ..AttentionInputs::new(BatchAxis::new(0), BatchAxis::new(0), BatchAxis::replicated())
    };
    let (output, residual) = batch(
        |inputs: AttentionInputs<_>| {
            let (output, residual) = function.call(inputs)?;
            Ok((output, residual.unwrap()))
        },
        inputs,
        axes,
        (BatchAxis::new(0), BatchAxis::new(0)),
        None,
    )
    .unwrap();
    assert_eq!(output.r#type().shape(), &static_shape(&[2, 1, 1, 1]));
    assert_eq!(output.to_f64s(), vec![3.0, 10.0]);
    assert_eq!(residual.r#type().shape(), &static_shape(&[2, 1, 1]));

    // Batching inside a trace stages the folded call of the same definition inside the batched call's rule program.
    let r#type = ArrayType::new_static(DataType::F32, [3, 1, 2, 1, 1]);
    let (_, program) = ArrayContext::trace(
        |inputs: AttentionInputs<_>| {
            batch(
                |inputs: AttentionInputs<_>| Ok(function.call(inputs)?.0),
                inputs,
                AttentionInputs::new(BatchAxis::new(0), BatchAxis::new(1), BatchAxis::new(1)),
                BatchAxis::new(0),
                None,
            )
            .map_err(ProgramError::from)
        },
        AttentionInputs::new(
            r#type.clone(),
            ArrayType::new_static(DataType::F32, [1, 3, 2, 1, 1]),
            ArrayType::new_static(DataType::F32, [1, 3, 2, 1, 1]),
        ),
    )
    .unwrap();
    assert_eq!(
        program.to_string(),
        indoc! {"
            lambda %0:f32[3, 1, 2, 1, 1], %1:f32[1, 3, 2, 1, 1], %2:f32[1, 3, 2, 1, 1] .
            let %3:f32[3, 1, 2, 1, 1], %4:f32[3, 1, 2, 1] = composite [
                key=\"ryft.dot_product_attention.v1 {scale = none, causal = false, local_window = none, implementation = 0, dropout_rate = none, residual = true, bias = false, mask = false, query_sequence_lengths = false, key_value_sequence_lengths = false, dropout_state = false}\",
                batching=[(extent=3, input_axes=[axis 0, axis 1, axis 1], output_axes=[axis 0, axis 0])],
            ] %0 %1 %2 [
                primal={
                    lambda %0:f32[3, 1, 2, 1, 1], %1:f32[1, 3, 2, 1, 1], %2:f32[1, 3, 2, 1, 1] .
                    let %3:f32[3, 1, 2, 1, 1] = transpose [permutation=[1, 0, 2, 3, 4]] %1
                        %4:f32[3, 1, 2, 1, 1] = transpose [permutation=[1, 0, 2, 3, 4]] %2
                        %5:f32[3, 2, 1, 1] = reshape [shape=[3, 2, 1, 1]] %0
                        %6:f32[3, 2, 1, 1] = reshape [shape=[3, 2, 1, 1]] %3
                        %7:f32[3, 2, 1, 1] = reshape [shape=[3, 2, 1, 1]] %4
                        %8:f32[3, 2, 1, 1], %9:f32[3, 2, 1] = composite [
                            key=\"ryft.dot_product_attention.v1 {scale = none, causal = false, local_window = none, implementation = 0, dropout_rate = none, residual = true, bias = false, mask = false, query_sequence_lengths = false, key_value_sequence_lengths = false, dropout_state = false}\",
                        ] %5 %6 %7 [
                            primal={
                                lambda %0:f32[3, 2, 1, 1], %1:f32[3, 2, 1, 1], %2:f32[3, 2, 1, 1] .
                                let %3:f32[3, 1, 2, 2] = dot [
                                    dimensions=(lhs_contracting=[3], rhs_contracting=[3], lhs_batching=[0, 2], rhs_batching=[0, 2]),
                                ] %0 %1
                                    %4:f32[] = constant [value=1.0]
                                    %5:f32[3, 1, 2, 2] = broadcast [output_type=f32[3, 1, 2, 2], output_axes=[]] %4
                                    %6:f32[3, 1, 2, 2] = mul %3 %5
                                    %7:f32[3, 1, 2] = reduce [kind=max, axes=[3]] %6
                                    %8:f32[3, 1, 2, 2] = broadcast [output_type=f32[3, 1, 2, 2], output_axes=[0, 1, 2]] %7
                                    %9:f32[3, 1, 2, 2] = sub %6 %8
                                    %10:f32[3, 1, 2, 2] = exp %9
                                    %11:f32[3, 1, 2] = reduce [kind=sum, axes=[3]] %10
                                    %12:f32[3, 1, 2, 2] = broadcast [output_type=f32[3, 1, 2, 2], output_axes=[0, 1, 2]] %11
                                    %13:f32[3, 1, 2, 2] = div %10 %12
                                    %14:f32[3, 1, 2, 1] = dot [
                                        dimensions=(lhs_contracting=[3], rhs_contracting=[1], lhs_batching=[0, 1], rhs_batching=[0, 2]),
                                    ] %13 %2
                                    %15:f32[3, 2, 1, 1] = transpose [permutation=[0, 2, 1, 3]] %14
                                    %16:f32[3, 1, 2] = log %11
                                    %17:f32[3, 1, 2] = add %7 %16
                                    %18:f32[3, 2, 1] = transpose [permutation=[0, 2, 1]] %17
                                    %19:f32[3, 2, 1] = stop_gradient %18
                                in (%15, %19)
                            },
                        ]
                        %10:f32[3, 1, 2, 1, 1] = reshape [shape=[3, 1, 2, 1, 1]] %8
                        %11:f32[3, 1, 2, 1] = reshape [shape=[3, 1, 2, 1]] %9
                    in (%10, %11)
                },
            ]
            in (%3)
        "}
        .trim_end(),
    );
}

#[test]
fn test_dot_product_attention_batching_static_extents() {
    // Homogeneous array families support static extents only, so batching over dynamic extents fails precisely.
    let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new());
    let length = DimensionVariable::new("length", DimensionBounds::new(1, Some(3)).unwrap());
    let r#type = ArrayType::new(DataType::F32, Shape::new(vec![2.into(), length.into(), 1.into(), 1.into()]));
    let result = ArrayContext::trace(
        |inputs: AttentionInputs<_>| {
            batch(
                |inputs: AttentionInputs<_>| Ok(function.call(inputs)?.0),
                inputs,
                AttentionInputs::new(BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(0)),
                BatchAxis::new(0),
                None,
            )
            .map_err(ProgramError::from)
        },
        AttentionInputs::new(r#type.clone(), r#type.clone(), r#type),
    );
    assert!(matches!(
        result,
        Err(ProgramError::Type(TypeError::Invalid { message, .. }))
            if message.starts_with("attention over homogeneous arrays requires static extents"),
    ));
}

#[test]
fn test_dot_product_attention_batching_manual_variation() {
    // Inside a manual region, a mapped bias is normalized to the canonical score shape with its own manual variation,
    // so that it still matches the varying query, key, and value of the folded call.
    let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
    let varying = |dimensions: Vec<usize>| {
        let rank = dimensions.len();
        ArrayType::new_static(DataType::F32, dimensions)
            .with_sharding(Sharding::replicated(mesh.clone(), rank).with_varying_manual_axes(["x"]).unwrap())
            .unwrap()
    };
    let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new());
    let signature = AttentionInputSignature::new(true, false, false, false);
    let (output_type, _) = DomainTracingContext::<ArrayContext>::trace_with_named_axes(
        |inputs: Vec<_>| {
            let axes = vec![BatchAxis::new(0); 4];
            batch(
                |inputs: Vec<_>| Ok(function.call(AttentionInputs::from_values(signature, &inputs)?)?.0),
                inputs,
                axes,
                BatchAxis::new(0),
                None,
            )
            .map_err(ProgramError::from)
        },
        vec![varying(vec![2, 1, 1, 1]), varying(vec![2, 2, 1, 1]), varying(vec![2, 2, 1, 1]), varying(vec![2, 1, 2])],
        vec![(
            "x".to_string(),
            NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2, checks_manual_variation: true },
        )],
    )
    .unwrap();
    assert_eq!(output_type, varying(vec![2, 1, 1, 1]));
}

#[test]
fn test_dot_product_attention_batching_with_dynamic_extent() -> Result<(), ProgramError> {
    type TraceContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

    // Array IR families fold dynamic mapped extents with first-class dimension values.
    let function = dot_product_attention::<IrContext>(AttentionConfiguration::new());
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
        |inputs: AttentionInputs<_>| Ok(function.call(inputs)?.0),
        AttentionInputs::new(query, key, value),
        AttentionInputs::new(BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(0)),
        BatchAxis::new(0),
        None,
    )?;
    let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, ArrayIrValue<Array>>(
        vec![output.atom_id()?],
        vec![Placeholder, Placeholder, Placeholder],
        Placeholder,
    )?;
    assert_eq!(output.r#type().as_ref(), &ArrayIrType::Array(r#type));
    assert_eq!(
        program.to_string(),
        indoc! {"
            lambda %0:f32[batch, 1, 1, 1, 1], %1:f32[batch, 1, 1, 1, 1], %2:f32[batch, 1, 1, 1, 1] .
            let %3:dimension<batch ∈ [1, 5)> = dimension_size [axis=0] %0
                %4:dimension<batch ∈ [1, 5)> = dimension_size [axis=0] %1
                %5:bool[] = const true
                %6:dimension<batch ∈ [1, 5)> = dimension_size [axis=0] %2
                %7:bool[] = const true
                %8:f32[batch, 1, 1, 1, 1] = composite [
                    key=\"ryft.dot_product_attention.v1 {scale = none, causal = false, local_window = none, implementation = 0, dropout_rate = none, residual = false, bias = false, mask = false, query_sequence_lengths = false, key_value_sequence_lengths = false, dropout_state = false}\",
                    non_differentiated_count=1,
                    batching=[(extent=dimension<batch ∈ [1, 5)>, input_axes=[axis 0, axis 0, axis 0], output_axes=[axis 0])],
                ] %3 %0 %1 %2 [
                    primal={
                        lambda %0:dimension<batch ∈ [1, 5)>, %1:f32[batch, 1, 1, 1, 1], %2:f32[batch, 1, 1, 1, 1], %3:f32[batch, 1, 1, 1, 1] .
                        let %4:dimension<1> = constant [value=1]
                            %5:dimension<1> = constant [value=1]
                            %6:dimension<1> = constant [value=1]
                            %7:dimension<1> = constant [value=1]
                            %8:dimension<1> = constant [value=1]
                            %9:f32[batch, 1, 1, 1] = reshape [requires_runtime_assertion=false] %1 %0 %6 %7 %8
                            %10:dimension<1> = constant [value=1]
                            %11:dimension<1> = constant [value=1]
                            %12:dimension<1> = constant [value=1]
                            %13:dimension<1> = constant [value=1]
                            %14:f32[batch, 1, 1, 1] = reshape [requires_runtime_assertion=false] %2 %0 %11 %12 %13
                            %15:dimension<1> = constant [value=1]
                            %16:dimension<1> = constant [value=1]
                            %17:dimension<1> = constant [value=1]
                            %18:dimension<1> = constant [value=1]
                            %19:f32[batch, 1, 1, 1] = reshape [requires_runtime_assertion=false] %3 %0 %16 %17 %18
                            %20:f32[batch, 1, 1, 1] = composite [
                                key=\"ryft.dot_product_attention.v1 {scale = none, causal = false, local_window = none, implementation = 0, dropout_rate = none, residual = false, bias = false, mask = false, query_sequence_lengths = false, key_value_sequence_lengths = false, dropout_state = false}\",
                            ] %9 %14 %19 [
                                primal={
                                    lambda %0:f32[batch, 1, 1, 1], %1:f32[batch, 1, 1, 1], %2:f32[batch, 1, 1, 1] .
                                    let %3:f32[batch, 1, 1, 1] = dot [
                                        dimensions=(lhs_contracting=[3], rhs_contracting=[3], lhs_batching=[0, 2], rhs_batching=[0, 2]),
                                    ] %0 %1
                                        %4:f32[] = constant [value=1.0]
                                        %5:dimension<batch ∈ [1, 5)> = dimension_size [axis=0] %3
                                        %6:dimension<1> = constant [value=1]
                                        %7:dimension<1> = constant [value=1]
                                        %8:dimension<1> = constant [value=1]
                                        %9:f32[batch, 1, 1, 1] = broadcast [output_axes=[]] %4 %5 %6 %7 %8
                                        %10:f32[batch, 1, 1, 1] = mul %3 %9
                                        %11:f32[batch, 1, 1] = reduce [kind=max, axes=[3]] %10
                                        %12:dimension<batch ∈ [1, 5)> = dimension_size [axis=0] %10
                                        %13:dimension<1> = constant [value=1]
                                        %14:dimension<1> = constant [value=1]
                                        %15:dimension<1> = constant [value=1]
                                        %16:f32[batch, 1, 1, 1] = broadcast [output_axes=[0, 1, 2]] %11 %12 %13 %14 %15
                                        %17:f32[batch, 1, 1, 1] = sub %10 %16
                                        %18:f32[batch, 1, 1, 1] = exp %17
                                        %19:f32[batch, 1, 1] = reduce [kind=sum, axes=[3]] %18
                                        %20:dimension<batch ∈ [1, 5)> = dimension_size [axis=0] %18
                                        %21:dimension<1> = constant [value=1]
                                        %22:dimension<1> = constant [value=1]
                                        %23:dimension<1> = constant [value=1]
                                        %24:f32[batch, 1, 1, 1] = broadcast [output_axes=[0, 1, 2]] %19 %20 %21 %22 %23
                                        %25:f32[batch, 1, 1, 1] = div %18 %24
                                        %26:f32[batch, 1, 1, 1] = dot [
                                            dimensions=(lhs_contracting=[3], rhs_contracting=[1], lhs_batching=[0, 1], rhs_batching=[0, 2]),
                                        ] %25 %2
                                        %27:f32[batch, 1, 1, 1] = transpose [permutation=[0, 2, 1, 3]] %26
                                    in (%27)
                                },
                            ]
                            %21:dimension<1> = constant [value=1]
                            %22:dimension<1> = constant [value=1]
                            %23:dimension<1> = constant [value=1]
                            %24:dimension<1> = constant [value=1]
                            %25:f32[batch, 1, 1, 1, 1] = reshape [requires_runtime_assertion=false] %20 %0 %21 %22 %23 %24
                        in (%25)
                    },
                ]
            in (%8)
        "}
        .trim_end(),
    );
    Ok(())
}

#[test]
fn test_dot_product_attention_dropout() {
    let r#type = ArrayType::new_static(DataType::F32, [1, 4, 1, 1]);
    let elements = [1.0, 2.0, 3.0, 4.0];
    let inputs = AttentionInputs::new(
        f32_array(&[1, 4, 1, 1], &[0.0; 4]),
        f32_array(&[1, 4, 1, 1], &[0.0; 4]),
        Array::from_elements::<f32>(r#type, &elements).unwrap(),
    );
    let state = |seed: u64| Array::from_elements(ArrayType::new_static(DataType::U64, [2]), &[0_u64, seed]).unwrap();
    let configuration = AttentionConfiguration::new().with_dropout(0.5);
    let function = dot_product_attention::<ArrayContext>(configuration);

    // Dropout scales the kept uniform attention weights by `1 / (1 - rate)`, so each output row is the sum of a
    // subset of the values scaled by `2 / 4`. A state determines the mask, and the absent state uses a fixed state.
    let with_state = |seed| AttentionInputs { dropout_state: Some(state(seed)), ..inputs.clone() };
    let (first, _) = function.call(with_state(1)).unwrap();
    let (again, _) = function.call(with_state(1)).unwrap();
    assert_eq!(first.to_f64s(), again.to_f64s());
    let (stateless, _) = function.call(inputs.clone()).unwrap();
    let (stateless_again, _) = function.call(inputs.clone()).unwrap();
    assert_eq!(stateless.to_f64s(), stateless_again.to_f64s());
    let outputs = (0..8).map(|seed| function.call(with_state(seed)).unwrap().0.to_f64s()).collect::<Vec<_>>();
    assert!(outputs.iter().any(|output| output != &outputs[0]));
    for output in outputs {
        let subset_sum = output[0] / 0.5;
        assert!((0.0..=10.0).contains(&subset_sum));
        assert_eq!(subset_sum.fract(), 0.0);
    }

    // Definitions with dropout batch structurally, so each batch item applies the mask of its unbatched call.
    let batched_inputs = AttentionInputs::new(
        f32_array(&[2, 1, 4, 1, 1], &[0.0; 8]),
        f32_array(&[2, 1, 4, 1, 1], &[0.0; 8]),
        f32_array(&[2, 1, 4, 1, 1], &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]),
    );
    let batched_inputs = AttentionInputs { dropout_state: Some(state(1)), ..batched_inputs };
    let axes = AttentionInputs {
        dropout_state: Some(BatchAxis::replicated()),
        ..AttentionInputs::new(BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(0))
    };
    let output =
        batch(|inputs: AttentionInputs<_>| Ok(function.call(inputs)?.0), batched_inputs, axes, BatchAxis::new(0), None)
            .unwrap();
    let expected = first.to_f64s();
    assert_eq!(output.to_f64s(), [expected.clone(), expected].concat());
}

#[test]
fn test_dot_product_attention_batching_after_drop() {
    // A staged call keeps its definition after the function that staged it is dropped, and batching it later stages
    // an equivalent folded definition.
    let r#type = ArrayType::new_static(DataType::F32, [1, 1, 1, 1]);
    let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new());
    let (_, program) = ArrayContext::trace(
        |inputs: AttentionInputs<_>| Ok(function.call(inputs)?.0),
        AttentionInputs::new(r#type.clone(), r#type.clone(), r#type),
    )
    .unwrap();
    drop(function);
    let program = program.into_flat_program();
    let ones = f32_array(&[2, 1, 1, 1, 1], &[1.0, 1.0]);
    let output = batch(
        |inputs: Vec<_>| {
            let context = inputs[0].domain();
            Ok(program.entry_region_ref().interpret_in_context(&context, inputs)?.remove(0))
        },
        vec![ones.clone(), ones, f32_array(&[2, 1, 1, 1, 1], &[5.0, 7.0])],
        vec![BatchAxis::new(0); 3],
        BatchAxis::new(0),
        None,
    )
    .unwrap();
    assert_eq!(output.to_f64s(), vec![5.0, 7.0]);
}

#[test]
fn test_dot_product_attention_fused_dropout_batching() {
    // Batching a call with dropout and the fused implementation would silently replace the fused kernels and their
    // dropout mask with the portable body, so it is rejected.
    let configuration =
        AttentionConfiguration::new().with_dropout(0.5).with_implementation(AttentionImplementation::Fused);
    let function = dot_product_attention::<ArrayContext>(configuration);
    let ones = f32_array(&[2, 1, 1, 1, 1], &[1.0, 1.0]);
    let result = batch(
        |inputs: AttentionInputs<_>| Ok(function.call(inputs)?.0),
        AttentionInputs::new(ones.clone(), ones.clone(), ones),
        AttentionInputs::new(BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(0)),
        BatchAxis::new(0),
        None,
    );
    let error = result.map(|_| ()).unwrap_err().to_string();
    assert!(
        error.contains(
            "named composite `ryft.dot_product_attention` with dropout cannot be batched with the fused \
             implementation",
        ),
        "{error}",
    );
}
