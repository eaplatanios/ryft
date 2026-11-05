//! Cross-phase XLA assertions coverage.

use pretty_assertions::assert_eq;

use super::*;

#[test]
fn test_compiled_dimension_from_scalar_reports_observed_bounds_failure_on_cpu() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin
        .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
        .unwrap();
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    let input_type = replicated_scalar_type(&mesh, DataType::I64);
    let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());

    let mut builder = ProgramBuilder::new();
    let input = builder.add_input(input_type.clone().into());
    let dimension = builder
        .add_instruction(DimensionFromScalarOperation::new(extent.into()), Vec::new(), vec![input], None)
        .unwrap()[0];
    let output = builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![dimension], None).unwrap()[0];
    let program = builder
        .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
        .unwrap();
    let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
    assert_eq!(lowered.stable_hlo().matches("@ryft.assert").count(), 1, "{}", lowered.stable_hlo());
    let compiled = domain.compile_program(&lowered).unwrap();
    assert!(compiled.requires_assertion_handler);
    // Some PJRT plugins do not serialize executables containing host callbacks. When serialization is available,
    // exercise the cache-hit path and prove its runtime-feature metadata survives restoration.
    let compiled = match domain.serialize_program(&compiled).unwrap() {
        Some(bytes) => domain.deserialize_program(bytes.as_slice()).unwrap().unwrap(),
        None => compiled,
    };
    assert!(compiled.requires_assertion_handler);

    let valid =
        Array::from_host_buffer(&domain, input_type.clone(), mesh.clone(), 4_i64.to_ne_bytes().as_slice()).unwrap();
    let valid = domain.execute_program(&compiled, vec![valid]).unwrap();
    assert_eq!(read_i64s(&client, &valid[0]), vec![4]);

    let invalid = Array::from_host_buffer(&domain, input_type, mesh, 9_i64.to_ne_bytes().as_slice()).unwrap();
    let error = domain.execute_program(&compiled, vec![invalid]).unwrap_err();
    assert!(
        error.to_string().contains(
            "`dimension_from_scalar` failed: input dimension `extent` = 9 is outside its declared bounds [1, 9)"
        ),
        "{error}",
    );
}

#[test]
fn test_compiled_assertions_report_the_first_same_class_failure_on_cpu() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin
        .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
        .unwrap();
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    let input_type = replicated_scalar_type(&mesh, DataType::I64);
    let bounds = DimensionBounds::new(0, Some(10)).unwrap();
    let first_type = DimensionType::new("first", bounds);
    let second_type = DimensionType::new("second", bounds);
    let third_type = DimensionType::new("third", bounds);

    let mut builder = ProgramBuilder::new();
    let first_input = builder.add_input(input_type.clone().into());
    let second_input = builder.add_input(input_type.clone().into());
    let third_input = builder.add_input(input_type.clone().into());
    let first = builder
        .add_instruction(DimensionFromScalarOperation::new(first_type.clone()), Vec::new(), vec![first_input], None)
        .unwrap()[0];
    let second = builder
        .add_instruction(DimensionFromScalarOperation::new(second_type.clone()), Vec::new(), vec![second_input], None)
        .unwrap()[0];
    let third = builder
        .add_instruction(DimensionFromScalarOperation::new(third_type.clone()), Vec::new(), vec![third_input], None)
        .unwrap()[0];
    let predicate = builder
        .add_instruction(
            CompareOperation::<ArrayIrType>::new(ComparisonDirection::LessThanOrEqual),
            Vec::new(),
            vec![first, second],
            None,
        )
        .unwrap()[0];
    builder
        .add_instruction(
            AssertOperation::<ArrayIrType>::new("`first` <= `second`")
                .with_labels(vec!["first".to_owned(), "second".to_owned()]),
            Vec::new(),
            vec![predicate, first, second],
            None,
        )
        .unwrap();
    let predicate = builder
        .add_instruction(
            CompareOperation::<ArrayIrType>::new(ComparisonDirection::LessThanOrEqual),
            Vec::new(),
            vec![second, third],
            None,
        )
        .unwrap()[0];
    builder
        .add_instruction(
            AssertOperation::<ArrayIrType>::new("`second` <= `third`")
                .with_labels(vec!["second".to_owned(), "third".to_owned()]),
            Vec::new(),
            vec![predicate, second, third],
            None,
        )
        .unwrap();
    let output = builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![third], None).unwrap()[0];
    let program = builder
        .build::<Vec<Constant>, Vec<Constant>>(
            vec![output],
            vec![Placeholder, Placeholder, Placeholder],
            vec![Placeholder],
        )
        .unwrap();
    let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
    assert_eq!(lowered.stable_hlo().matches("@ryft.assert").count(), 5, "{}", lowered.stable_hlo());
    let compiled = domain.compile_program(&lowered).unwrap();
    let input = |value: i64| {
        Array::from_host_buffer(&domain, input_type.clone(), mesh.clone(), value.to_ne_bytes().as_slice()).unwrap()
    };

    let error = domain.execute_program(&compiled, vec![input(4), input(3), input(2)]).unwrap_err();
    assert!(error.to_string().contains("assertion failed: `first` <= `second`; first=4, second=3"), "{error}",);
    assert!(!error.to_string().contains("`second` <= `third`"), "{error}");
}

#[test]
fn test_compiled_assertions_preserve_diagnostics_on_cpu() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin
        .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
        .unwrap();
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    let input_type = replicated_scalar_type(&mesh, DataType::I64);
    let bounds = DimensionBounds::new(0, Some(20)).unwrap();
    let left_type = DimensionType::new("left", bounds);
    let right_type = DimensionType::new("right", bounds);
    for (left_value, right_value, divisible) in [(3_i64, 4_i64, false), (7, 3, true), (7, 0, true)] {
        let mut builder = ProgramBuilder::new();
        let left_input = builder.add_input(input_type.clone().into());
        let right_input = builder.add_input(input_type.clone().into());
        let left = builder
            .add_instruction(DimensionFromScalarOperation::new(left_type.clone()), Vec::new(), vec![left_input], None)
            .unwrap()[0];
        let right = builder
            .add_instruction(DimensionFromScalarOperation::new(right_type.clone()), Vec::new(), vec![right_input], None)
            .unwrap()[0];
        let mut output = right;
        let (predicate, message) = if divisible {
            let zero = builder.add_constant(Constant::Dimension(DimensionValue::constant(0).unwrap()));
            let one = builder.add_constant(Constant::Dimension(DimensionValue::constant(1).unwrap()));
            let positive = builder
                .add_instruction(
                    CompareOperation::<ArrayIrType>::new(ComparisonDirection::GreaterThan),
                    Vec::new(),
                    vec![right, zero],
                    None,
                )
                .unwrap()[0];
            builder
                .add_instruction(
                    AssertOperation::<ArrayIrType>::new("divisor must be positive")
                        .with_labels(vec!["right".to_owned()]),
                    Vec::new(),
                    vec![positive, right],
                    None,
                )
                .unwrap();
            let one_type = DimensionValue::constant(1).unwrap().r#type().into_owned();
            let maximum = ryft_core::DimensionMaxOperation::new(&right_type, &one_type).unwrap();
            let safe_type = maximum.infer_output_types(&[right_type.clone(), one_type], &[]).unwrap().remove(0);
            let safe = builder
                .add_instruction(ryft_core::DimensionOperation::Max(maximum), Vec::new(), vec![right, one], None)
                .unwrap()[0];
            let remainder = builder
                .add_instruction(
                    ryft_core::DimensionOperation::Rem(DimensionRemOperation::new(&left_type, &safe_type).unwrap()),
                    Vec::new(),
                    vec![left, safe],
                    None,
                )
                .unwrap()[0];
            output = builder
                .add_instruction(
                    ryft_core::DimensionOperation::Div(DimensionDivOperation::new(&left_type, &safe_type).unwrap()),
                    Vec::new(),
                    vec![left, safe],
                    None,
                )
                .unwrap()[0];
            (
                builder
                    .add_instruction(
                        CompareOperation::<ArrayIrType>::new(ComparisonDirection::Equal),
                        Vec::new(),
                        vec![remainder, zero],
                        None,
                    )
                    .unwrap()[0],
                "`left` must be divisible by `right`",
            )
        } else {
            (
                builder
                    .add_instruction(
                        CompareOperation::<ArrayIrType>::new(ComparisonDirection::Equal),
                        Vec::new(),
                        vec![left, right],
                        None,
                    )
                    .unwrap()[0],
                "dimensions must agree",
            )
        };
        builder
            .add_instruction(
                AssertOperation::<ArrayIrType>::new(message).with_labels(vec!["left".to_owned(), "right".to_owned()]),
                Vec::new(),
                vec![predicate, left, right],
                None,
            )
            .unwrap();
        let output = builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![output], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = domain.compile_program(&lowered).unwrap();
        let input = |value: i64| {
            Array::from_host_buffer(&domain, input_type.clone(), mesh.clone(), value.to_ne_bytes().as_slice()).unwrap()
        };
        let error = domain.execute_program(&compiled, vec![input(left_value), input(right_value)]).unwrap_err();
        let expected = if right_value == 0 {
            ryft_core::AssertionError::Failed {
                message: "divisor must be positive".to_owned(),
                observations: vec![("right".to_owned(), "0".to_owned())],
            }
        } else {
            ryft_core::AssertionError::Failed {
                message: message.to_owned(),
                observations: vec![
                    ("left".to_owned(), left_value.to_string()),
                    ("right".to_owned(), right_value.to_string()),
                ],
            }
        };
        assert!(error.to_string().contains(&expected.to_string()), "{error}");
    }
}

#[test]
fn test_compiled_per_element_validation_on_cpu() {
    use ryft_core::Compare;

    let plugin = load_cpu_plugin().unwrap();
    let client = plugin
        .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
        .unwrap();
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    let values = f32_vector(&domain, &mesh, &[1., -2., 3., -4.]);
    let limits = f32_vector(&domain, &mesh, &[0., 0., 0., 0.]);
    let staged = crate::jit::stage::<_, Vec<ArrayType>, Vec<ArrayType>>(
        |inputs| vec![inputs[0].greater_than(&inputs[1]).unwrap()],
        vec![values.r#type().into_owned(), limits.r#type().into_owned()],
        Options::new(mesh.clone()),
    )
    .unwrap()
    .into_inner();
    let lowered = domain.lower_program(staged.source_program().program(), 0, &Options::new(mesh.clone())).unwrap();
    let compiled = domain.compile_program(&lowered).unwrap();
    assert!(!compiled.requires_assertion_handler);
    let outputs = domain.execute_program(&compiled, vec![values, limits]).unwrap();
    assert_eq!(
        outputs[0]
            .addressable_shards()
            .next()
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap(),
        vec![1, 0, 1, 0]
    );
}

#[test]
fn test_compiled_assertion_bounded_elements_on_cpu() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin
        .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
        .unwrap();
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    for mode in [0, 1, 2] {
        let predicates = boolean_vector(&domain, &mesh, &[true, false, false, false]);
        let observations = f32_vector(&domain, &mesh, &[10., 20., 30., 40.]);
        let staged = crate::jit::stage::<_, Vec<ArrayType>, Vec<ArrayType>>(
            |inputs| {
                if mode == 1 {
                    batch(
                        |(predicate, observation)| {
                            predicate.assert_with_limit(
                                "bounded check",
                                &[("value", observation)],
                                std::num::NonZeroUsize::new(2).unwrap(),
                            )
                        },
                        (inputs[0].clone(), inputs[1].clone()),
                        (BatchAxis::new(0), BatchAxis::new(0)),
                        (),
                        None,
                    )
                    .unwrap();
                } else {
                    let predicate = if mode == 2 { inputs[0].reshape([2, 2]).unwrap() } else { inputs[0].clone() };
                    let observation = if mode == 2 { inputs[1].reshape([2, 2]).unwrap() } else { inputs[1].clone() };
                    predicate
                        .assert_with_limit(
                            "bounded check",
                            &[("value", observation)],
                            std::num::NonZeroUsize::new(2).unwrap(),
                        )
                        .unwrap();
                }
                vec![]
            },
            vec![predicates.r#type().into_owned(), observations.r#type().into_owned()],
            Options::new(mesh.clone()),
        )
        .unwrap()
        .into_inner();
        let lowered = domain.lower_program(staged.source_program().program(), 0, &Options::new(mesh.clone())).unwrap();
        let compiled = domain.compile_program(&lowered).unwrap();
        let error = domain.execute_program(&compiled, vec![predicates, observations.clone()]).unwrap_err();
        let expected = ryft_core::AssertionError::FailedElements {
            message: "bounded check".to_owned(),
            failures: vec![
                ryft_core::AssertionFailure {
                    index: if mode == 2 { vec![0, 1] } else { vec![1] },
                    observations: vec![("value".to_owned(), "20".to_owned())],
                },
                ryft_core::AssertionFailure {
                    index: if mode == 2 { vec![1, 0] } else { vec![2] },
                    observations: vec![("value".to_owned(), "30".to_owned())],
                },
            ],
            omitted: 1,
        };
        assert!(error.to_string().contains(&expected.to_string()), "{error}");
        let predicates = boolean_vector(&domain, &mesh, &[true; 4]);
        assert!(domain.execute_program(&compiled, vec![predicates, observations]).unwrap().is_empty());
    }
}

#[test]
fn test_compiled_assertion_bounded_sub_byte_observations_on_cpu() {
    use ryft_core::ConvertElementType;

    let plugin = load_cpu_plugin().unwrap();
    let client = plugin
        .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
        .unwrap();
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    for (data_type, values) in [
        (DataType::I1, [0., -1., 0., -1.]),
        (DataType::U1, [0., 1., 0., 1.]),
        (DataType::I2, [0., -2., -1., 1.]),
        (DataType::U2, [0., 1., 2., 3.]),
        (DataType::I4, [0., -8., -1., 7.]),
        (DataType::U4, [0., 1., 8., 15.]),
    ] {
        let predicates = boolean_vector(&domain, &mesh, &[false; 4]);
        let observations = f32_vector(&domain, &mesh, &values);
        let staged = crate::jit::stage::<_, Vec<ArrayType>, Vec<ArrayType>>(
            |inputs| {
                let observation = inputs[1].convert_element_type(data_type).unwrap();
                inputs[0]
                    .assert_with_limit(
                        "narrow values",
                        &[("value", observation)],
                        std::num::NonZeroUsize::new(3).unwrap(),
                    )
                    .unwrap();
                vec![]
            },
            vec![predicates.r#type().into_owned(), observations.r#type().into_owned()],
            Options::new(mesh.clone()),
        )
        .unwrap()
        .into_inner();
        let lowered = domain.lower_program(staged.source_program().program(), 0, &Options::new(mesh.clone())).unwrap();
        let compiled = domain.compile_program(&lowered).unwrap();
        let error = domain.execute_program(&compiled, vec![predicates, observations]).unwrap_err();
        let expected = ryft_core::AssertionError::FailedElements {
            message: "narrow values".to_owned(),
            failures: values
                .iter()
                .take(3)
                .enumerate()
                .map(|(index, value)| ryft_core::AssertionFailure {
                    index: vec![index],
                    observations: vec![("value".to_owned(), value.to_string())],
                })
                .collect(),
            omitted: 1,
        };
        assert!(error.to_string().contains(&expected.to_string()), "{data_type}: {error}");
    }
}

#[test]
fn test_compiled_assertion_scalar_observations_on_cpu() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin
        .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
        .unwrap();
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    let predicate_type = replicated_scalar_type(&mesh, DataType::Boolean);
    for observations in [
        vec![],
        vec![
            ("unsigned", DataType::U64, u64::MAX.to_ne_bytes().to_vec(), u64::MAX.to_string()),
            ("signed", DataType::I64, i64::MIN.to_ne_bytes().to_vec(), i64::MIN.to_string()),
            ("float", DataType::F32, (-0.5_f32).to_ne_bytes().to_vec(), "-0.5".to_owned()),
            ("boolean", DataType::Boolean, vec![1], "true".to_owned()),
        ],
    ] {
        let mut builder = ProgramBuilder::new();
        let predicate = builder.add_input(predicate_type.clone().into());
        let predicate =
            builder.add_instruction(AndOperation::new(), vec![], vec![predicate, predicate], None).unwrap()[0];
        let mut inputs = vec![predicate];
        for (_, data_type, _, _) in &observations {
            inputs.push(builder.add_input(replicated_scalar_type(&mesh, *data_type).into()));
        }
        builder
            .add_instruction(
                AssertOperation::<ArrayIrType>::new("scalar check")
                    .with_labels(observations.iter().map(|(label, _, _, _)| (*label).to_owned()).collect()),
                Vec::new(),
                inputs,
                None,
            )
            .unwrap();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![], vec![Placeholder; observations.len() + 1], vec![])
            .unwrap();
        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = domain.compile_program(&lowered).unwrap();
        for passed in [true, false] {
            let mut inputs = vec![
                Array::from_host_buffer(&domain, predicate_type.clone(), mesh.clone(), &[u8::from(passed)]).unwrap(),
            ];
            for (_, data_type, bytes, _) in &observations {
                inputs.push(
                    Array::from_host_buffer(&domain, replicated_scalar_type(&mesh, *data_type), mesh.clone(), bytes)
                        .unwrap(),
                );
            }
            let output = domain.execute_program(&compiled, inputs);
            if passed {
                assert!(output.unwrap().is_empty());
            } else {
                let expected = ryft_core::AssertionError::Failed {
                    message: "scalar check".to_owned(),
                    observations: observations
                        .iter()
                        .map(|(label, _, _, rendered)| ((*label).to_owned(), rendered.clone()))
                        .collect(),
                };
                let error = output.unwrap_err();
                assert!(error.to_string().contains(&expected.to_string()), "{error}");
            }
        }
    }

    let predicates = boolean_vector(&domain, &mesh, &[true, false, false]);
    let observations = f32_vector(&domain, &mesh, &[10., 20., 30.]);
    let staged = crate::jit::stage::<_, Vec<ArrayType>, Vec<ArrayType>>(
        |inputs| {
            batch(
                |(predicate, observation)| predicate.assert("mapped check", &[("value", observation)]),
                (inputs[0].clone(), inputs[1].clone()),
                (BatchAxis::new(0), BatchAxis::new(0)),
                (),
                None,
            )
            .unwrap();
            vec![]
        },
        vec![predicates.r#type().into_owned(), observations.r#type().into_owned()],
        Options::new(mesh.clone()),
    )
    .unwrap()
    .into_inner();
    let program = staged.source_program().program();
    let lowered = domain.lower_program(program, 0, &Options::new(mesh.clone())).unwrap();
    let compiled = domain.compile_program(&lowered).unwrap();
    let error = domain.execute_program(&compiled, vec![predicates, observations]).unwrap_err();
    let expected = ryft_core::AssertionError::Failed {
        message: "mapped check".to_owned(),
        observations: vec![("value".to_owned(), "20".to_owned()), ("batch_index".to_owned(), "1".to_owned())],
    };
    assert!(error.to_string().contains(&expected.to_string()), "{error}");
}

#[test]
fn test_compiled_assertion_dynamic_batching_on_cpu() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin
        .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
        .unwrap();
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    let input_type = replicated_scalar_type(&mesh, DataType::I64);
    for bounded in [false, true] {
        let extent = DimensionVariable::new("batch", DimensionBounds::new(0, Some(5)).unwrap());
        let context = TracingContext::<Constant, Operation>::new();
        let size = context.input(input_type.clone().into());
        let limit = context.input(input_type.clone().into());
        let dimension = context
            .bind(DimensionFromScalarOperation::new(extent.clone().into()), Vec::new(), &[size])
            .unwrap()
            .remove(0);
        let indices = context
            .bind(
                IotaOperation::new(ArrayType::new(DataType::I64, Shape::new(vec![extent.into()])), 0).unwrap(),
                Vec::new(),
                std::slice::from_ref(&dimension),
            )
            .unwrap()
            .remove(0);
        let limits = context
            .bind(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), &[limit, dimension.clone()])
            .unwrap()
            .remove(0);
        let predicates = context
            .bind(
                Operation::Array(ryft_core::ArrayOperation::Compare(CompareOperation::new(
                    ComparisonDirection::LessThan,
                ))),
                Vec::new(),
                &[indices.clone(), limits],
            )
            .unwrap()
            .remove(0);
        let batching = BatchingContext::new(context.clone(), dimension);
        let operation = AssertOperation::new("dynamic mapped check").with_labels(vec!["value".to_owned()]);
        let operation =
            if bounded { operation.with_failure_limit(std::num::NonZeroUsize::new(2).unwrap()) } else { operation };
        operation
            .batch(
                &batching,
                &EmptyRegionDriver,
                &[ArrayIrBatch::new(predicates, 0).unwrap(), ArrayIrBatch::new(indices, 0).unwrap()],
            )
            .unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Constant>, Vec<Constant>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = domain.compile_program(&lowered).unwrap();
        for (size, limit) in [(0_i64, 0_i64), (1, 1), (3, 3)] {
            let inputs = [size, limit]
                .into_iter()
                .map(|value| {
                    Array::from_host_buffer(&domain, input_type.clone(), mesh.clone(), &value.to_ne_bytes()).unwrap()
                })
                .collect::<Vec<_>>();
            assert!(domain.execute_program(&compiled, inputs).unwrap().is_empty());
        }
        let inputs = [3_i64, 1_i64]
            .into_iter()
            .map(|value| {
                Array::from_host_buffer(&domain, input_type.clone(), mesh.clone(), &value.to_ne_bytes()).unwrap()
            })
            .collect::<Vec<_>>();
        let error = domain.execute_program(&compiled, inputs).unwrap_err();
        let expected = if bounded {
            ryft_core::AssertionError::FailedElements {
                message: "dynamic mapped check".to_owned(),
                failures: vec![
                    ryft_core::AssertionFailure {
                        index: vec![1],
                        observations: vec![("value".to_owned(), "1".to_owned())],
                    },
                    ryft_core::AssertionFailure {
                        index: vec![2],
                        observations: vec![("value".to_owned(), "2".to_owned())],
                    },
                ],
                omitted: 0,
            }
        } else {
            ryft_core::AssertionError::Failed {
                message: "dynamic mapped check".to_owned(),
                observations: vec![("value".to_owned(), "1".to_owned()), ("batch_index".to_owned(), "1".to_owned())],
            }
        };
        assert!(error.to_string().contains(&expected.to_string()), "{error}");
    }
}

#[test]
fn test_compiled_assertion_dynamic_replicated_batching_on_cpu() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin
        .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
        .unwrap();
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    let input_type = replicated_scalar_type(&mesh, DataType::I64);
    for bounded in [false, true] {
        let context = TracingContext::<Constant, Operation>::new();
        let size = context.input(input_type.clone().into());
        let dimension = context
            .bind(
                DimensionFromScalarOperation::new(
                    DimensionVariable::new("batch", DimensionBounds::new(0, Some(5)).unwrap()).into(),
                ),
                Vec::new(),
                std::slice::from_ref(&size),
            )
            .unwrap()
            .remove(0);
        let condition = context.lift(Constant::Boolean(false)).unwrap();
        let batching = BatchingContext::new(context.clone(), dimension);
        let operation = AssertOperation::new("dynamic replicated check").with_labels(vec!["size".to_owned()]);
        let operation =
            if bounded { operation.with_failure_limit(std::num::NonZeroUsize::new(2).unwrap()) } else { operation };
        operation
            .batch(
                &batching,
                &EmptyRegionDriver,
                &[ArrayIrBatch::replicated(condition), ArrayIrBatch::replicated(size)],
            )
            .unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Constant>, Vec<Constant>>(vec![], vec![Placeholder], vec![])
            .unwrap();
        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = domain.compile_program(&lowered).unwrap();
        let empty = Array::from_host_buffer(&domain, input_type.clone(), mesh.clone(), &0_i64.to_ne_bytes()).unwrap();
        assert!(domain.execute_program(&compiled, vec![empty]).unwrap().is_empty());
        let nonempty =
            Array::from_host_buffer(&domain, input_type.clone(), mesh.clone(), &3_i64.to_ne_bytes()).unwrap();
        let error = domain.execute_program(&compiled, vec![nonempty]).unwrap_err();
        let expected = if bounded {
            ryft_core::AssertionError::FailedElements {
                message: "dynamic replicated check".to_owned(),
                failures: vec![
                    ryft_core::AssertionFailure {
                        index: vec![0],
                        observations: vec![("size".to_owned(), "3".to_owned())],
                    },
                    ryft_core::AssertionFailure {
                        index: vec![1],
                        observations: vec![("size".to_owned(), "3".to_owned())],
                    },
                ],
                omitted: 1,
            }
        } else {
            ryft_core::AssertionError::Failed {
                message: "dynamic replicated check".to_owned(),
                observations: vec![("size".to_owned(), "3".to_owned())],
            }
        };
        assert!(error.to_string().contains(&expected.to_string()), "{error}");
    }
}

#[test]
fn test_compiled_dimension_arithmetic_preserves_checked_host_diagnostics_on_cpu() {
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin
        .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
        .unwrap();
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    let input_type = replicated_scalar_type(&mesh, DataType::I64);
    let bounds = DimensionBounds::new(0, Some(10)).unwrap();
    let left_type = DimensionType::new("left", bounds);
    let right_type = DimensionType::new("right", bounds);
    let check = |operation: DimensionOperation<DimensionValue>,
                 valid_operands: (i64, i64),
                 expected_output: i64,
                 invalid_operands: (i64, i64),
                 expected_error: &str| {
        let mut builder = ProgramBuilder::new();
        let left_input = builder.add_input(input_type.clone().into());
        let right_input = builder.add_input(input_type.clone().into());
        let left = builder
            .add_instruction(DimensionFromScalarOperation::new(left_type.clone()), Vec::new(), vec![left_input], None)
            .unwrap()[0];
        let right = builder
            .add_instruction(DimensionFromScalarOperation::new(right_type.clone()), Vec::new(), vec![right_input], None)
            .unwrap()[0];
        let result = builder.add_instruction(operation, Vec::new(), vec![left, right], None).unwrap()[0];
        let output = builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![result], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        assert_eq!(lowered.stable_hlo().matches("@ryft.assert").count(), 3, "{}", lowered.stable_hlo());
        let compiled = domain.compile_program(&lowered).unwrap();
        let input = |value: i64| {
            Array::from_host_buffer(&domain, input_type.clone(), mesh.clone(), value.to_ne_bytes().as_slice()).unwrap()
        };

        let output = domain.execute_program(&compiled, vec![input(valid_operands.0), input(valid_operands.1)]).unwrap();
        assert_eq!(read_i64s(&client, &output[0]), vec![expected_output]);
        let error = domain
            .execute_program(&compiled, vec![input(invalid_operands.0), input(invalid_operands.1)])
            .unwrap_err();
        assert!(error.to_string().contains(expected_error), "{error}");
    };

    check(
        DimensionOperation::Sub(DimensionSubOperation::new(&left_type, &right_type).unwrap()),
        (5, 2),
        3,
        (2, 5),
        "`dimension_sub` requires left >= right; got left=2, right=5",
    );
    check(
        DimensionOperation::Div(DimensionDivOperation::new(&left_type, &right_type).unwrap()),
        (7, 3),
        2,
        (7, 0),
        "`dimension_div` requires right > 0; got left=7, right=0",
    );
    check(
        DimensionOperation::Rem(DimensionRemOperation::new(&left_type, &right_type).unwrap()),
        (7, 3),
        1,
        (7, 0),
        "`dimension_rem` requires right > 0; got left=7, right=0",
    );
}

#[test]
fn test_compiled_dimension_assertion_does_not_merge_ordered_io_chains_on_cpu() {
    use crate::operations::debugging::{ensure_print_handler_registered, with_captured_prints};

    let plugin = load_cpu_plugin().unwrap();
    let client = plugin
        .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
        .unwrap();
    ensure_print_handler_registered(&client).unwrap();
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    let input_type = replicated_scalar_type(&mesh, DataType::I64);
    let bounds = DimensionBounds::new(0, Some(10)).unwrap();
    let left_type = DimensionType::new("left", bounds);
    let right_type = DimensionType::new("right", bounds);

    let mut builder = ProgramBuilder::new();
    let left_input = builder.add_input(input_type.clone().into());
    let right_input = builder.add_input(input_type.clone().into());
    builder
        .add_instruction(PrintOperation::<ArrayType>::new("left"), Vec::new(), vec![left_input], None)
        .unwrap();
    let left = builder
        .add_instruction(DimensionFromScalarOperation::new(left_type.clone()), Vec::new(), vec![left_input], None)
        .unwrap()[0];
    let right = builder
        .add_instruction(DimensionFromScalarOperation::new(right_type.clone()), Vec::new(), vec![right_input], None)
        .unwrap()[0];
    let predicate = builder
        .add_instruction(
            CompareOperation::<ArrayIrType>::new(ComparisonDirection::LessThanOrEqual),
            Vec::new(),
            vec![left, right],
            None,
        )
        .unwrap()[0];
    builder
        .add_instruction(
            AssertOperation::<ArrayIrType>::new("`left` <= `right`")
                .with_labels(vec!["left".to_owned(), "right".to_owned()]),
            Vec::new(),
            vec![predicate, left, right],
            None,
        )
        .unwrap();
    let output = builder
        .add_instruction(PrintOperation::<ArrayType>::new("right"), Vec::new(), vec![right_input], None)
        .unwrap()[0];
    let program = builder
        .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
        .unwrap();
    let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
    // Only assertions create a local chain; ordered I/O is seeded from the hidden executable argument.
    assert!(lowered.signature.has_ordered_io());
    assert!(lowered.signature.requires_assertion_handler());
    assert_eq!(lowered.stable_hlo().matches("stablehlo.after_all").count(), 1, "{}", lowered.stable_hlo());
    let compiled = domain.compile_program(&lowered).unwrap();
    let input = |value: i64| {
        Array::from_host_buffer(&domain, input_type.clone(), mesh.clone(), value.to_ne_bytes().as_slice()).unwrap()
    };

    let (output, lines) = with_captured_prints(|| {
        let output = domain.execute_program(&compiled, vec![input(3), input(4)]).unwrap();
        assert_eq!(read_i64s(&client, &output[0]), vec![4]);
        output
    });
    assert_eq!(lines.len(), 2);
    assert!(lines[0].starts_with("left: "), "{:?}", lines);
    assert!(lines[1].starts_with("right: "), "{:?}", lines);
    assert_eq!(read_i64s(&client, &output[0]), vec![4]);
}
