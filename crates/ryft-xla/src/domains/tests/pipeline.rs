//! Cross-phase XLA pipeline coverage.

use pretty_assertions::assert_eq;

use super::*;

#[test]
fn test_compiled_symbolic_manipulation_primal_and_pullback() {
    let client = execution_client();
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    // Build each symbolic graph and its pullback once, then replay both at the two concrete signatures in
    // the pinned JAX oracle, plus empty rows for the conveniences that admit them. Keep dimension residuals
    // internal to a fused primal/pullback; this verifies retained
    // graph specialization, without asserting reuse of one bounded executable across physical shapes.
    for name in [
        "reshape",
        "broadcast",
        "concatenate",
        "gather",
        "slice",
        "gather_axis",
        "scatter_axis",
        "batched_slice",
        "mapped_source_slice",
        "expand_dimensions",
        "flatten",
        "column_take",
        "dynamic_gather_query",
        "dynamic_scatter_query",
        "column_scatter_query",
        "slice_axis",
        "index_axis",
    ] {
        let allows_empty_rows =
            matches!(name, "expand_dimensions" | "flatten" | "column_take" | "slice_axis" | "index_axis");
        let input_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![
                CoreDimension::Dynamic(DimensionVariable::new(
                    "n",
                    DimensionBounds::new(if allows_empty_rows { 0 } else { 4 }, Some(6)).unwrap(),
                )),
                CoreDimension::Static(4),
            ]),
        );
        let index_count = if name == "mapped_source_slice" { 4 } else { 3 };
        let dynamic_query = matches!(name, "dynamic_gather_query" | "dynamic_scatter_query" | "column_scatter_query");
        let query_dimension = if dynamic_query {
            CoreDimension::Dynamic(DimensionVariable::new("queries", DimensionBounds::new(2, Some(4)).unwrap()))
        } else {
            CoreDimension::Static(index_count)
        };
        let indices_type =
            ArrayType::new(DataType::I32, Shape::new(vec![query_dimension.clone(), CoreDimension::Static(1)]));
        let updates_type = if name == "column_scatter_query" {
            ArrayType::new(DataType::F64, Shape::new(vec![input_type.dimension(0), query_dimension]))
        } else {
            ArrayType::new(DataType::F64, Shape::new(vec![query_dimension, CoreDimension::Static(4)]))
        };
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(input_type.clone().into());
        let indices = builder.add_input(indices_type.clone().into());
        let start = builder.add_input(ArrayType::scalar(DataType::I32).into());
        let zero = builder.add_input(ArrayType::scalar(DataType::I32).into());
        builder.add_input(updates_type.into());
        let extent_operation = DimensionSizeOperation::new(&input_type, 0).unwrap();
        let extent_type = extent_operation.output_type().clone();
        let extent = builder.add_instruction(extent_operation, vec![], vec![input], None).unwrap()[0];
        let two_value = DimensionValue::constant(2).unwrap();
        let two_type = two_value.r#type().into_owned();
        let two = builder.add_constant(Constant::Dimension(two_value));
        let four = builder.add_constant(Constant::Dimension(DimensionValue::constant(4).unwrap()));
        let output = match name {
            "reshape" => {
                let doubled = builder
                    .add_instruction(
                        DimensionOperation::Mul(DimensionMulOperation::new(&extent_type, &two_type).unwrap()),
                        vec![],
                        vec![extent, two],
                        None,
                    )
                    .unwrap()[0];
                builder
                    .add_instruction(DynamicReshapeOperation::new(), vec![], vec![input, two, doubled], None)
                    .unwrap()[0]
            }
            "broadcast" => builder
                .add_instruction(
                    DynamicBroadcastOperation::new(vec![1, 2]),
                    vec![],
                    vec![input, two, extent, four],
                    None,
                )
                .unwrap()[0],
            "concatenate" => {
                let addition = DimensionAddOperation::new(&extent_type, &extent_type).unwrap();
                let doubled = builder
                    .add_instruction(DimensionOperation::Add(addition), vec![], vec![extent, extent], None)
                    .unwrap()[0];
                let operation = ConcatenateOperation::<ArrayIrType>::new(
                    0,
                    &[
                        input_type.clone().into(),
                        input_type.clone().into(),
                        builder.atoms()[doubled.index()].r#type().into_owned(),
                    ],
                )
                .unwrap();
                builder.add_instruction(operation, vec![], vec![input, input, doubled], None).unwrap()[0]
            }
            "gather" => {
                let operation =
                    GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 4])
                        .with_mode(GatherMode::Clip);
                builder
                    .add_instruction(
                        Operation::Array(ArrayOperation::Gather(operation)),
                        vec![],
                        vec![input, indices],
                        None,
                    )
                    .unwrap()[0]
            }
            "slice" => builder
                .add_instruction(
                    Operation::Array(ArrayOperation::DynamicSlice(DynamicSliceOperation::new(vec![2, 4]))),
                    vec![],
                    vec![input, start, zero],
                    None,
                )
                .unwrap()[0],
            "gather_axis"
            | "scatter_axis"
            | "batched_slice"
            | "mapped_source_slice"
            | "expand_dimensions"
            | "flatten"
            | "column_take"
            | "dynamic_gather_query"
            | "dynamic_scatter_query"
            | "column_scatter_query"
            | "slice_axis"
            | "index_axis" => input,
            _ => unreachable!(),
        };
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 5], vec![Placeholder])
            .unwrap();
        let program = if matches!(name, "gather_axis" | "scatter_axis" | "batched_slice" | "mapped_source_slice") {
            // The axis conveniences and batching trace use the same symbolic boundary as the mixed graphs.
            let staged = crate::jit::stage::<_, Vec<ArrayType>, Vec<ArrayType>>(
                |inputs| {
                    let indices = inputs[1].reshape(Shape::new(vec![CoreDimension::Static(index_count)])).unwrap();
                    vec![if name == "mapped_source_slice" {
                        batch(
                            |(input, start)| input.dynamic_slice(&[start], &[2]),
                            (inputs[0].clone(), indices),
                            (BatchAxis::new(1), BatchAxis::new(0)),
                            BatchAxis::new(0),
                            None,
                        )
                        .unwrap()
                    } else if name == "batched_slice" {
                        batch(
                            |(input, start, zero)| input.dynamic_slice(&[start, zero], &[2, 4]),
                            (inputs[0].clone(), indices, inputs[3].clone()),
                            (BatchAxis::replicated(), BatchAxis::new(0), BatchAxis::replicated()),
                            BatchAxis::new(0),
                            None,
                        )
                        .unwrap()
                    } else if name == "gather_axis" {
                        inputs[0].gather_axis(&indices, 0, GatherMode::Clip).unwrap()
                    } else {
                        inputs[0]
                            .scatter_axis(
                                &indices,
                                &inputs[4],
                                usize::from(name == "column_scatter_query"),
                                ScatterReductionKind::Add,
                                ScatterMode::Clip,
                            )
                            .unwrap()
                    }]
                },
                vec![
                    input_type.clone(),
                    ArrayType::new_static(DataType::I32, [index_count, 1]),
                    ArrayType::scalar(DataType::I32),
                    ArrayType::scalar(DataType::I32),
                    ArrayType::new_static(DataType::F64, [index_count, 4]),
                ],
                Options::new(mesh.clone()),
            )
            .unwrap()
            .into_inner();
            staged.source_program().program().clone()
        } else {
            program
        };
        let program = if matches!(
            name,
            "expand_dimensions"
                | "flatten"
                | "column_take"
                | "dynamic_gather_query"
                | "dynamic_scatter_query"
                | "column_scatter_query"
                | "slice_axis"
                | "index_axis"
        ) {
            TracingContext::<Constant, Operation>::trace(
                |inputs| {
                    Ok(vec![if name == "expand_dimensions" {
                        inputs[0].dynamic_expand_dimensions(0)?
                    } else if name == "slice_axis" {
                        inputs[0].dynamic_slice_axis(1, 1, 3, 1)?
                    } else if name == "index_axis" {
                        inputs[0].dynamic_index_axis(1, 1, false)?
                    } else if name == "flatten" {
                        inputs[0].dynamic_flatten()?
                    } else {
                        let query_size = inputs[1].dimension_size(0)?;
                        let indices = inputs[1].dynamic_reshape(&[query_size])?;
                        if matches!(name, "dynamic_scatter_query" | "column_scatter_query") {
                            inputs[0].dynamic_scatter_axis(
                                &indices,
                                &inputs[4],
                                usize::from(name == "column_scatter_query"),
                                ScatterReductionKind::Add,
                                ScatterMode::Clip,
                            )?
                        } else {
                            inputs[0].dynamic_gather_axis(
                                &indices,
                                usize::from(name == "column_take"),
                                GatherMode::Clip,
                            )?
                        }
                    }])
                },
                program.input_types().into_owned(),
            )
            .unwrap()
            .1
        } else {
            program
        };
        let linearization = program.linearize_with_respect_to(&[0]).unwrap();
        let pullback = linearization.pullback().unwrap().into_program();
        let sizes: &[usize] = if allows_empty_rows { &[0, 4, 5] } else { &[4, 5] };
        for &size in sizes {
            let index_count = if dynamic_query { size - 2 } else { index_count };
            let values = (0..size * 4).map(|value| value as f64).collect::<Vec<_>>();
            let (shape, expected) = match name {
                "reshape" => (vec![2, 2 * size], values.clone()),
                "expand_dimensions" => (vec![1, size, 4], values.clone()),
                "flatten" => (vec![size * 4], values.clone()),
                "slice_axis" => (vec![size, 2], values.chunks_exact(4).flat_map(|row| [row[1], row[2]]).collect()),
                "index_axis" => (vec![size], values.chunks_exact(4).map(|row| row[1]).collect()),
                "broadcast" => (vec![2, size, 4], [values.clone(), values.clone()].concat()),
                "concatenate" => (vec![2 * size, 4], [values.clone(), values.clone()].concat()),
                "gather" | "gather_axis" => (vec![3, 4], [&values[4..8], &values[4..8], &values[12..16]].concat()),
                "slice" => (vec![2, 4], values[4..12].to_vec()),
                "column_take" => {
                    (vec![size, 3], values.chunks_exact(4).flat_map(|row| [row[1], row[1], row[3]]).collect())
                }
                "dynamic_gather_query" => {
                    let mut expected = [&values[4..8], &values[4..8]].concat();
                    if index_count == 3 {
                        expected.extend_from_slice(&values[12..16]);
                    }
                    (vec![index_count, 4], expected)
                }
                "column_scatter_query" => {
                    let mut expected = values.clone();
                    for row in expected.chunks_exact_mut(4) {
                        row[1] += 2.;
                        if index_count == 3 {
                            row[3] += 1.;
                        }
                    }
                    (vec![size, 4], expected)
                }
                "dynamic_scatter_query" => {
                    let mut expected = values.clone();
                    expected[4..8].iter_mut().for_each(|value| *value += 2.);
                    if index_count == 3 {
                        expected[12..16].iter_mut().for_each(|value| *value += 1.);
                    }
                    (vec![size, 4], expected)
                }
                "batched_slice" => {
                    let start = 3.min(size - 2) * 4;
                    (vec![3, 2, 4], [&values[4..12], &values[4..12], &values[start..start + 8]].concat())
                }
                "mapped_source_slice" => {
                    let expected = [1_usize, 1, 3, 0]
                        .into_iter()
                        .enumerate()
                        .flat_map(|(column, start)| {
                            let start = start.min(size - 2);
                            [values[start * 4 + column], values[(start + 1) * 4 + column]]
                        })
                        .collect();
                    (vec![4, 2], expected)
                }
                "scatter_axis" => {
                    let mut expected = values.clone();
                    expected[4..8].iter_mut().for_each(|value| *value += 2.);
                    expected[12..16].iter_mut().for_each(|value| *value += 1.);
                    (vec![size, 4], expected)
                }
                _ => unreachable!(),
            };
            let input_type = ArrayType::new_static(DataType::F64, [size, 4]);
            let output_type = ArrayType::new_static(DataType::F64, shape);
            let input_types = vec![
                input_type.clone(),
                ArrayType::new_static(DataType::I32, [index_count, 1]),
                ArrayType::scalar(DataType::I32),
                ArrayType::scalar(DataType::I32),
                if name == "column_scatter_query" {
                    ArrayType::new_static(DataType::F64, [size, index_count])
                } else {
                    ArrayType::new_static(DataType::F64, [index_count, 4])
                },
                output_type.clone(),
            ];
            let specialized = program
                .clone()
                .specialize(&input_types[..5].iter().cloned().map(ArrayIrType::from).collect::<Vec<_>>())
                .unwrap();
            assert_eq!(specialized.output_types(), vec![ArrayIrType::from(output_type.clone())], "{name}, n={size}");
            let compiled = crate::jit::compile::<_, Vec<ArrayType>, Vec<ArrayType>>(
                |inputs| {
                    let mut inputs = inputs.into_iter().map(|input| input.into_value()).collect::<Vec<_>>();
                    let cotangent = inputs.pop().unwrap();
                    let context = inputs[0].context().clone();
                    let mut outputs = linearization
                        .primal()
                        .interpret_in_context(&context, inputs)
                        .unwrap_or_else(|error| panic!("{name}, n={size} primal: {error}\n{}", linearization.primal()));
                    let mut pullback_inputs = vec![cotangent];
                    pullback_inputs.extend(outputs.split_off(1));
                    outputs.extend(
                        pullback
                            .interpret_in_context(&context, pullback_inputs)
                            .unwrap_or_else(|error| panic!("{name}, n={size} pullback: {error}\n{pullback}")),
                    );
                    outputs
                        .into_iter()
                        .map(|output| ValueProjection::<ArrayType>::into_projected(output).unwrap())
                        .collect()
                },
                input_types.clone(),
                &domain,
                mesh.clone(),
            )
            .unwrap_or_else(|error| panic!("{name}, n={size} native trace: {error}"));
            let bytes = [
                values_to_bytes(&values),
                if name == "mapped_source_slice" {
                    values_to_bytes(&[1_i32, 1, 3, 0])
                } else {
                    values_to_bytes(&[1_i32, 1, 3][..index_count])
                },
                values_to_bytes(&[1_i32]),
                values_to_bytes(&[0_i32]),
                values_to_bytes(&vec![1_f64; index_count * if name == "column_scatter_query" { size } else { 4 }]),
                values_to_bytes(&vec![1_f64; expected.len()]),
            ];
            let inputs = input_types
                .into_iter()
                .zip(bytes)
                .map(|(r#type, bytes)| Array::from_host_buffer(&domain, r#type, mesh.clone(), bytes).unwrap())
                .collect();
            let outputs = domain.interpret(&compiled.executable_function(), inputs).unwrap();
            assert_eq!(outputs.len(), 2);
            for output in &outputs {
                assert_eq!(output.data_type(), DataType::F64);
                assert_eq!(output.mesh(), &mesh);
            }
            assert_eq!(outputs[0].shape().as_slice(), output_type.static_shape().unwrap().as_slice());
            assert_eq!(read_f64s(&client, &outputs[0]), expected, "{name}, n={size}");
            let mut gradient = vec![if name == "broadcast" || name == "concatenate" { 2_f64 } else { 1_f64 }; size * 4];
            if name == "gather" || name == "gather_axis" {
                gradient.fill(0.);
                gradient[4..8].fill(2.);
                gradient[12..16].fill(1.);
            }
            if name == "slice_axis" {
                gradient = (0..size).flat_map(|_| [0., 1., 1., 0.]).collect();
            }
            if name == "index_axis" {
                gradient = (0..size).flat_map(|_| [0., 1., 0., 0.]).collect();
            }
            if name == "column_take" {
                gradient = (0..size).flat_map(|_| [0., 2., 0., 1.]).collect();
            }
            if name == "dynamic_gather_query" {
                gradient.fill(0.);
                gradient[4..8].fill(2.);
                if index_count == 3 {
                    gradient[12..16].fill(1.);
                }
            }
            if name == "slice" {
                gradient.fill(0.);
                gradient[4..12].fill(1.);
            }
            if name == "batched_slice" {
                // Overlapping windows accumulate rather than overwrite their input cotangents.
                let rows = if size == 4 { vec![0., 2., 3., 1.] } else { vec![0., 2., 2., 1., 1.] };
                gradient = rows.into_iter().flat_map(|value| [value; 4]).collect();
            }
            if name == "mapped_source_slice" {
                // Each mapped column contributes only to its own clipped window, including the nonleading map.
                gradient.fill(0.);
                for (column, start) in [1_usize, 1, 3, 0].into_iter().enumerate() {
                    let start = start.min(size - 2);
                    gradient[start * 4 + column] = 1.;
                    gradient[(start + 1) * 4 + column] = 1.;
                }
            }
            assert_eq!(outputs[1].shape().as_slice(), &[size, 4]);
            assert_eq!(read_f64s(&client, &outputs[1]), gradient, "{name} pullback, n={size}");
        }
    }
}

#[test]
fn test_compiled_indexing() {
    assert_compiled_indexing(&execution_client());
}

#[cfg(feature = "cuda-13")]
#[test]
fn test_compiled_indexing_on_cuda() {
    let plugin = load_cuda_13_plugin().unwrap();
    let client = plugin
        .client(ClientOptions::GPU(GpuClientOptions {
            platform: Some(GpuPlatform::CUDA),
            allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
            ..Default::default()
        }))
        .unwrap();
    assert_compiled_indexing(&client);
    assert_compiled_indexing_symbolic_extent(&client);
}

#[test]
fn test_compiled_indexing_symbolic_extent() {
    assert_compiled_indexing_symbolic_extent(&execution_client());
}

#[test]
fn test_compiled_jointly_mapped_gather_over_possibly_empty_extent() {
    // Jointly mapping the input and the indices of a gather over a dynamic extent whose lower bound is zero
    // stages one gather with paired batching axes and a zero batching window. This runs the transformed program
    // on the CPU PJRT client used by the other tests (the `cutile` feature only enables compilation of the kernel
    // module) at empty and nonempty mapped extents, through both the primal and its pullback.
    let client = execution_client();
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    let extent = CoreDimension::Dynamic(DimensionVariable::new("n", DimensionBounds::new(0, Some(6)).unwrap()));
    let input_type = ArrayType::new(DataType::F64, Shape::new(vec![extent.clone(), CoreDimension::Static(4)]));
    let indices_type = ArrayType::new(DataType::I32, Shape::new(vec![extent, CoreDimension::Static(1)]));
    let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![], vec![0], vec![0]), vec![1])
        .with_mode(GatherMode::Clip);
    // The public `batch` entry runs under the static extent policy, which rejects a dynamic mapped axis before any
    // operation rule runs, so the joint rule is driven through the dynamic extent policy exactly as the ryft-core
    // reproduction does, with the mapped extent read from the staged input.
    let staged = crate::jit::stage::<_, Vec<ArrayType>, Vec<ArrayType>>(
        |inputs| {
            let extent = inputs[0].dimension_size(0).unwrap();
            let context = BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
                ProjectedContext::new(extent.context().clone()),
                extent,
            );
            let batches = [
                ArrayBatch::new(inputs[0].clone(), BatchAxis::new(0)).unwrap(),
                ArrayBatch::new(inputs[1].clone(), BatchAxis::new(0)).unwrap(),
            ];
            let outputs = operation.batch(&context, &EmptyRegionDriver, &batches).unwrap();
            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
            vec![outputs.into_iter().next().unwrap().into_value()]
        },
        vec![input_type, indices_type],
        Options::new(mesh.clone()),
    )
    .unwrap()
    .into_inner();
    let program = staged.source_program().program().clone();
    let rendered = program.to_string();
    assert_eq!(
        rendered,
        indoc! {"
            lambda %0:f64[n, 4], %1:i32[n, 1] .
            let %2:f64[n] = gather [
                dimensions=(offset=[], collapsed_slice=[1], start_index_map=[1], batching=[(0, 0)]),
                slice_sizes=[0, 1],
                mode=clip,
            ] %0 %1
            in (%2)
        "}
        .trim_end(),
    );
    assert!(rendered.contains("slice_sizes=[0, 1]"));
    assert!(rendered.contains("batching=[(0, 0)]"));
    let linearization = program.linearize_with_respect_to(&[0]).unwrap();
    let pullback = linearization.pullback().unwrap().into_program();
    let all_indices = [1_i32, 3, 0, 2, 3];
    for size in [0_usize, 4, 5] {
        let values = (0..size * 4).map(|value| value as f64).collect::<Vec<_>>();
        let indices = &all_indices[..size];
        let expected =
            indices.iter().enumerate().map(|(row, &index)| values[row * 4 + index as usize]).collect::<Vec<_>>();
        let mut gradient = vec![0_f64; size * 4];
        for (row, &index) in indices.iter().enumerate() {
            gradient[row * 4 + index as usize] = 1.;
        }
        let input_type = ArrayType::new_static(DataType::F64, [size, 4]);
        let output_type = ArrayType::new_static(DataType::F64, [size]);
        let input_types =
            vec![input_type.clone(), ArrayType::new_static(DataType::I32, [size, 1]), output_type.clone()];
        let specialized = program
            .clone()
            .specialize(&input_types[..2].iter().cloned().map(ArrayIrType::from).collect::<Vec<_>>())
            .unwrap();
        assert_eq!(specialized.output_types(), vec![ArrayIrType::from(output_type.clone())], "n={size}");
        let compiled = crate::jit::compile::<_, Vec<ArrayType>, Vec<ArrayType>>(
            |inputs| {
                let mut inputs = inputs.into_iter().map(|input| input.into_value()).collect::<Vec<_>>();
                let cotangent = inputs.pop().unwrap();
                let context = inputs[0].context().clone();
                let mut outputs = linearization
                    .primal()
                    .interpret_in_context(&context, inputs)
                    .unwrap_or_else(|error| panic!("n={size} primal: {error}\n{}", linearization.primal()));
                let mut pullback_inputs = vec![cotangent];
                pullback_inputs.extend(outputs.split_off(1));
                outputs.extend(
                    pullback
                        .interpret_in_context(&context, pullback_inputs)
                        .unwrap_or_else(|error| panic!("n={size} pullback: {error}\n{pullback}")),
                );
                outputs
                    .into_iter()
                    .map(|output| ValueProjection::<ArrayType>::into_projected(output).unwrap())
                    .collect()
            },
            input_types.clone(),
            &domain,
            mesh.clone(),
        )
        .unwrap_or_else(|error| panic!("n={size} native trace: {error}"));
        let bytes = [values_to_bytes(&values), values_to_bytes(indices), values_to_bytes(&vec![1_f64; size])];
        let inputs = input_types
            .into_iter()
            .zip(bytes)
            .map(|(r#type, bytes)| Array::from_host_buffer(&domain, r#type, mesh.clone(), bytes).unwrap())
            .collect();
        let outputs = domain.interpret(&compiled.executable_function(), inputs).unwrap();
        assert_eq!(outputs.len(), 2);
        for output in &outputs {
            assert_eq!(output.data_type(), DataType::F64);
            assert_eq!(output.mesh(), &mesh);
        }
        assert_eq!(outputs[0].shape().as_slice(), output_type.static_shape().unwrap().as_slice());
        assert_eq!(read_f64s(&client, &outputs[0]), expected, "n={size}");
        assert_eq!(outputs[1].shape().as_slice(), &[size, 4]);
        assert_eq!(read_f64s(&client, &outputs[1]), gradient, "pullback, n={size}");
    }
}

#[test]
fn test_compiled_jointly_mapped_scatter_over_possibly_empty_extent() {
    // Jointly mapping the input, the indices, and the updates of a scatter over a dynamic extent whose lower bound
    // is zero stages one scatter with paired batching axes. Its dual gather, built by the scatter-add transpose
    // and by the extremal JVP coefficient graph, then takes a zero batching window over that extent. This runs the
    // transformed programs on the CPU PJRT client used by the other tests (the `cutile` feature only enables
    // compilation of the kernel module) at empty and nonempty mapped extents, through both the primal and its
    // pullback.
    type DynamicPolicy = ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>;
    let client = execution_client();
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    let extent = CoreDimension::Dynamic(DimensionVariable::new("n", DimensionBounds::new(0, Some(6)).unwrap()));
    let staged_types = vec![
        ArrayType::new(DataType::F64, Shape::new(vec![extent.clone(), CoreDimension::Static(4)])),
        ArrayType::new(
            DataType::I32,
            Shape::new(vec![extent.clone(), CoreDimension::Static(1), CoreDimension::Static(1)]),
        ),
        ArrayType::new(DataType::F64, Shape::new(vec![extent, CoreDimension::Static(1)])),
    ];
    let dimensions = ScatterDimensionNumbers::new(vec![], vec![0], vec![0]);
    // The public `batch` entry runs under the static extent policy, which rejects a dynamic mapped axis before any
    // operation rule runs, so the joint rule is driven through the dynamic extent policy exactly as the ryft-core
    // reproduction does, with the mapped extent read from the staged input.
    let stage = |operation: ScatterOperation| -> FlatProgram {
        crate::jit::stage::<_, Vec<ArrayType>, Vec<ArrayType>>(
            |inputs| {
                let extent = inputs[0].dimension_size(0).unwrap();
                let context = BatchingContext::<_, DynamicPolicy>::with_policy(
                    ProjectedContext::new(extent.context().clone()),
                    extent,
                );
                let batches = inputs
                    .iter()
                    .map(|input| ArrayBatch::new(input.clone(), BatchAxis::new(0)).unwrap())
                    .collect::<Vec<_>>();
                let outputs = operation.batch(&context, &EmptyRegionDriver, &batches).unwrap();
                assert_eq!(outputs.len(), 1);
                assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
                vec![outputs.into_iter().next().unwrap().into_value()]
            },
            staged_types.clone(),
            Options::new(mesh.clone()),
        )
        .unwrap()
        .into_inner()
        .source_program()
        .program()
        .clone()
    };
    // Specializes the primal at the concrete extent `size`, compiles the primal followed by its pullback at an
    // all-ones cotangent, runs it on the client, and returns the primal output, the input gradient, and the
    // updates gradient.
    let execute = |linearization: &Linearization<Constant, Operation>,
                   pullback: &FlatProgram,
                   size: usize,
                   values: &[f64],
                   indices: &[i32],
                   updates: &[f64]| {
        let input_types = vec![
            ArrayType::new_static(DataType::F64, [size, 4]),
            ArrayType::new_static(DataType::I32, [size, 1, 1]),
            ArrayType::new_static(DataType::F64, [size, 1]),
            ArrayType::new_static(DataType::F64, [size, 4]),
        ];
        let specialized = linearization
            .primal()
            .as_ref()
            .clone()
            .specialize(&input_types[..3].iter().cloned().map(ArrayIrType::from).collect::<Vec<_>>())
            .unwrap_or_else(|error| panic!("n={size} specialization: {error}\n{}", linearization.primal()));
        assert_eq!(specialized.output_types()[0], ArrayIrType::from(input_types[0].clone()), "n={size}");
        let compiled = crate::jit::compile::<_, Vec<ArrayType>, Vec<ArrayType>>(
            |inputs| {
                let mut inputs = inputs.into_iter().map(|input| input.into_value()).collect::<Vec<_>>();
                let cotangent = inputs.pop().unwrap();
                let context = inputs[0].context().clone();
                let mut outputs = linearization
                    .primal()
                    .interpret_in_context(&context, inputs)
                    .unwrap_or_else(|error| panic!("n={size} primal: {error}\n{}", linearization.primal()));
                let mut pullback_inputs = vec![cotangent];
                pullback_inputs.extend(outputs.split_off(1));
                outputs.extend(
                    pullback
                        .interpret_in_context(&context, pullback_inputs)
                        .unwrap_or_else(|error| panic!("n={size} pullback: {error}\n{pullback}")),
                );
                outputs
                    .into_iter()
                    .map(|output| ValueProjection::<ArrayType>::into_projected(output).unwrap())
                    .collect()
            },
            input_types.clone(),
            &domain,
            mesh.clone(),
        )
        .unwrap_or_else(|error| panic!("n={size} native trace: {error}"));
        let bytes = [
            values_to_bytes(values),
            values_to_bytes(indices),
            values_to_bytes(updates),
            values_to_bytes(&vec![1_f64; size * 4]),
        ];
        let inputs = input_types
            .into_iter()
            .zip(bytes)
            .map(|(r#type, bytes)| Array::from_host_buffer(&domain, r#type, mesh.clone(), bytes).unwrap())
            .collect();
        let outputs = domain.interpret(&compiled.executable_function(), inputs).unwrap();
        assert_eq!(outputs.len(), 3, "n={size}");
        for output in &outputs {
            assert_eq!(output.data_type(), DataType::F64);
            assert_eq!(output.mesh(), &mesh);
        }
        assert_eq!(outputs[0].shape().as_slice(), &[size, 4], "n={size}");
        assert_eq!(outputs[1].shape().as_slice(), &[size, 4], "n={size}");
        assert_eq!(outputs[2].shape().as_slice(), &[size, 1], "n={size}");
        outputs.iter().map(|output| read_f64s(&client, output)).collect::<Vec<_>>()
    };
    let all_indices = [1_i32, 3, 0, 2, 3];

    // The additive scatter is linear, so its pullback is the dual gather over the paired batching axes with the
    // zero window, applied to the cotangent directly.
    let add = ScatterOperation::new(dimensions.clone(), ScatterReductionKind::Add).with_mode(ScatterMode::Clip);
    let program = stage(add);
    assert_eq!(
        program.to_string(),
        indoc! {"
            lambda %0:f64[n, 4], %1:i32[n, 1, 1], %2:f64[n, 1] .
            let %3:f64[n, 4] = scatter [
                kind=add,
                dimensions=(update_window=[], inserted_window=[1], scatter_to_operand=[1], operand_batching=[0], \
                    scatter_indices_batching=[0]),
                mode=clip,
            ] %0 %1 %2
            in (%3)
        "}
        .trim_end(),
    );
    let linearization = program.linearize_with_respect_to(&[0, 2]).unwrap();
    let pullback = linearization.pullback().unwrap().into_program();
    assert_eq!(
        pullback.to_string(),
        indoc! {"
            lambda %0:f64[n, 4], %1:i32[n, 1, 1] .
            let %2:f64[n, 1] = gather [
                dimensions=(offset=[], collapsed_slice=[1], start_index_map=[1], batching=[(0, 0)]),
                slice_sizes=[0, 1],
                mode=clip,
            ] %0 %1
            in (%0, %2)
        "}
        .trim_end(),
    );
    let all_add_updates = [10_f64, 20., 30., 40., 50.];
    for size in [0_usize, 4, 5] {
        let values = (0..size * 4).map(|value| value as f64).collect::<Vec<_>>();
        let indices = &all_indices[..size];
        let updates = &all_add_updates[..size];
        let mut expected = values.clone();
        for (row, &index) in indices.iter().enumerate() {
            expected[row * 4 + index as usize] += updates[row];
        }
        let outputs = execute(&linearization, &pullback, size, &values, indices, updates);
        assert_eq!(outputs[0], expected, "add primal, n={size}");
        assert_eq!(outputs[1], vec![1_f64; size * 4], "add input gradient, n={size}");
        assert_eq!(outputs[2], vec![1_f64; size], "add updates gradient, n={size}");
    }

    // The minimum scatter is nonlinear: its JVP coefficient graph reads each update's target back through the
    // shared dual gather, so the pullback routes the cotangent to whichever of the input and the update won.
    let minimum = ScatterOperation::new(dimensions, ScatterReductionKind::Min).with_mode(ScatterMode::Clip);
    let program = stage(minimum);
    assert_eq!(
        program.to_string(),
        indoc! {"
            lambda %0:f64[n, 4], %1:i32[n, 1, 1], %2:f64[n, 1] .
            let %3:f64[n, 4] = scatter [
                kind=min,
                dimensions=(update_window=[], inserted_window=[1], scatter_to_operand=[1], operand_batching=[0], \
                    scatter_indices_batching=[0]),
                mode=clip,
            ] %0 %1 %2
            in (%3)
        "}
        .trim_end(),
    );
    let linearization = program.linearize_with_respect_to(&[0, 2]).unwrap();
    let pullback = linearization.pullback().unwrap().into_program();
    assert!(pullback.to_string().contains("slice_sizes=[0, 1]"));
    let all_min_updates = [-1_f64, 100., -1., 100., -1.];
    for size in [0_usize, 4, 5] {
        let values = (0..size * 4).map(|value| value as f64).collect::<Vec<_>>();
        let indices = &all_indices[..size];
        let updates = &all_min_updates[..size];
        let mut expected = values.clone();
        let mut input_gradient = vec![1_f64; size * 4];
        let mut updates_gradient = vec![0_f64; size];
        for (row, &index) in indices.iter().enumerate() {
            let position = row * 4 + index as usize;
            if updates[row] < values[position] {
                expected[position] = updates[row];
                input_gradient[position] = 0.;
                updates_gradient[row] = 1.;
            }
        }
        let outputs = execute(&linearization, &pullback, size, &values, indices, updates);
        assert_eq!(outputs[0], expected, "min primal, n={size}");
        assert_eq!(outputs[1], input_gradient, "min input gradient, n={size}");
        assert_eq!(outputs[2], updates_gradient, "min updates gradient, n={size}");
    }
}

#[test]
fn test_compiled_dynamic_update_slice_partial_pullbacks() {
    // `DynamicUpdateSliceOperation`'s transpose stages only the contributions whose accumulators are needed: with
    // only the operand cotangent requested it stages one `dynamic_update_slice` of the cotangent with a zero
    // update and no `dynamic_slice`, and with only the update cotangent requested it stages one `dynamic_slice`
    // of the cotangent and no zero. The shared symbolic manipulation harness always differentiates with respect
    // to every array input, so this runs both pruned pullbacks on the CPU PJRT client used by the other tests
    // (the `cutile` feature only enables compilation of the kernel module) at symbolic operand extents, with an
    // in-bounds start and a start that clamps to the last rows.
    let client = execution_client();
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    let extent = CoreDimension::Dynamic(DimensionVariable::new("n", DimensionBounds::new(4, Some(6)).unwrap()));
    let staged = crate::jit::stage::<_, Vec<ArrayType>, Vec<ArrayType>>(
        |inputs| vec![inputs[0].dynamic_update_slice(&inputs[1], &inputs[2..]).unwrap()],
        vec![
            ArrayType::new(DataType::F64, Shape::new(vec![extent, CoreDimension::Static(4)])),
            ArrayType::new_static(DataType::F64, [2, 4]),
            ArrayType::scalar(DataType::I32),
            ArrayType::scalar(DataType::I32),
        ],
        Options::new(mesh.clone()),
    )
    .unwrap()
    .into_inner();
    let program = staged.source_program().program().clone();
    assert_eq!(
        program.to_string(),
        indoc! {"
            lambda %0:f64[n, 4], %1:f64[2, 4], %2:i32[], %3:i32[] .
            let %4:f64[n, 4] = dynamic_update_slice %0 %1 %2 %3
            in (%4)
        "}
        .trim_end(),
    );
    // Specializes the primal at the concrete extent `size`, compiles the primal followed by its pullback at an
    // all-ones cotangent with an all-100 update and the given row start, runs it on the client, and returns the
    // primal output and the single requested gradient, whose shape is checked against `gradient_shape`.
    let execute = |linearization: &Linearization<Constant, Operation>,
                   pullback: &FlatProgram,
                   size: usize,
                   values: &[f64],
                   row: i32,
                   gradient_shape: &[usize]| {
        let input_types = vec![
            ArrayType::new_static(DataType::F64, [size, 4]),
            ArrayType::new_static(DataType::F64, [2, 4]),
            ArrayType::scalar(DataType::I32),
            ArrayType::scalar(DataType::I32),
            ArrayType::new_static(DataType::F64, [size, 4]),
        ];
        let specialized = linearization
            .primal()
            .as_ref()
            .clone()
            .specialize(&input_types[..4].iter().cloned().map(ArrayIrType::from).collect::<Vec<_>>())
            .unwrap_or_else(|error| panic!("n={size} specialization: {error}\n{}", linearization.primal()));
        assert_eq!(specialized.output_types()[0], ArrayIrType::from(input_types[0].clone()), "n={size}");
        let compiled = crate::jit::compile::<_, Vec<ArrayType>, Vec<ArrayType>>(
            |inputs| {
                let mut inputs = inputs.into_iter().map(|input| input.into_value()).collect::<Vec<_>>();
                let cotangent = inputs.pop().unwrap();
                let context = inputs[0].context().clone();
                let mut outputs = linearization
                    .primal()
                    .interpret_in_context(&context, inputs)
                    .unwrap_or_else(|error| panic!("n={size} primal: {error}\n{}", linearization.primal()));
                let mut pullback_inputs = vec![cotangent];
                pullback_inputs.extend(outputs.split_off(1));
                outputs.extend(
                    pullback
                        .interpret_in_context(&context, pullback_inputs)
                        .unwrap_or_else(|error| panic!("n={size} pullback: {error}\n{pullback}")),
                );
                outputs
                    .into_iter()
                    .map(|output| ValueProjection::<ArrayType>::into_projected(output).unwrap())
                    .collect()
            },
            input_types.clone(),
            &domain,
            mesh.clone(),
        )
        .unwrap_or_else(|error| panic!("n={size} native trace: {error}"));
        let bytes = [
            values_to_bytes(values),
            values_to_bytes(&[100_f64; 8]),
            values_to_bytes(&[row]),
            values_to_bytes(&[0_i32]),
            values_to_bytes(&vec![1_f64; size * 4]),
        ];
        let inputs = input_types
            .into_iter()
            .zip(bytes)
            .map(|(r#type, bytes)| Array::from_host_buffer(&domain, r#type, mesh.clone(), bytes).unwrap())
            .collect();
        let outputs = domain.interpret(&compiled.executable_function(), inputs).unwrap();
        assert_eq!(outputs.len(), 2, "n={size}, row={row}");
        for output in &outputs {
            assert_eq!(output.data_type(), DataType::F64);
            assert_eq!(output.mesh(), &mesh);
        }
        assert_eq!(outputs[0].shape().as_slice(), &[size, 4], "n={size}, row={row}");
        assert_eq!(outputs[1].shape().as_slice(), gradient_shape, "n={size}, row={row}");
        outputs.iter().map(|output| read_f64s(&client, output)).collect::<Vec<_>>()
    };
    // Row start 1 updates rows 1 and 2; row start 9 clamps to the last two rows of the input.
    let four_rows = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.];
    let five_rows = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.];
    let four_rows_in_bounds = [0., 1., 2., 3., 100., 100., 100., 100., 100., 100., 100., 100., 12., 13., 14., 15.];
    let five_rows_in_bounds =
        [0., 1., 2., 3., 100., 100., 100., 100., 100., 100., 100., 100., 12., 13., 14., 15., 16., 17., 18., 19.];
    let four_rows_clamped = [0., 1., 2., 3., 4., 5., 6., 7., 100., 100., 100., 100., 100., 100., 100., 100.];
    let five_rows_clamped =
        [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 100., 100., 100., 100., 100., 100., 100., 100.];

    // The symbolic input extent keeps the linearization inside a `linear_call`, so each pullback is the
    // transposed call: its forward region runs the transpose rule, while its transpose region carries the
    // original forward for re-transposition. Pinning both regions checks which contributions were pruned.

    // Only the operand cotangent: the update accumulator is unneeded, so the transpose zeroes the updated window
    // of the cotangent without slicing it.
    let linearization = program.linearize_with_respect_to(&[0]).unwrap();
    let pullback = linearization.pullback().unwrap().into_program();
    assert_eq!(
        pullback.to_string(),
        indoc! {"
            lambda %0:f64[n, 4], %1:i32[], %2:i32[] .
            let %3:f64[n, 4] = linear_call [residual_count=2] %1 %2 %0 [
                forward={
                    lambda %0:i32[], %1:i32[], %2:f64[n, 4] .
                    let %3:f64[2, 4] = zero [type=f64[2, 4]]
                        %4:f64[n, 4] = dynamic_update_slice %2 %3 %0 %1
                    in (%4)
                },
                transpose={
                    lambda %0:i32[], %1:i32[], %2:f64[n, 4] .
                    let %3:f64[2, 4] = zero [type=f64[2, 4]]
                        %4:f64[n, 4] = dynamic_update_slice %2 %3 %0 %1
                    in (%4)
                },
            ]
            in (%3)
        "}
        .trim_end(),
    );
    assert_eq!(
        execute(&linearization, &pullback, 4, &four_rows, 1, &[4, 4]),
        vec![four_rows_in_bounds.to_vec(), vec![1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.]],
    );
    assert_eq!(
        execute(&linearization, &pullback, 5, &five_rows, 1, &[5, 4]),
        vec![
            five_rows_in_bounds.to_vec(),
            vec![1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
        ],
    );
    assert_eq!(
        execute(&linearization, &pullback, 4, &four_rows, 9, &[4, 4]),
        vec![four_rows_clamped.to_vec(), vec![1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.]],
    );
    assert_eq!(
        execute(&linearization, &pullback, 5, &five_rows, 9, &[5, 4]),
        vec![
            five_rows_clamped.to_vec(),
            vec![1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        ],
    );

    // Only the update cotangent: the operand accumulator is unneeded, so the transpose slices the updated window
    // out of the cotangent without materializing a zero update. The retained operand extent and the zero operand
    // tangent belong to the original forward region, which materializes the missing operand tangent.
    let linearization = program.linearize_with_respect_to(&[1]).unwrap();
    let pullback = linearization.pullback().unwrap().into_program();
    assert_eq!(
        pullback.to_string(),
        indoc! {"
            lambda %0:f64[n, 4], %1:i32[], %2:i32[], %3:dimension<n ∈ [4, 6)> .
            let %4:f64[2, 4] = linear_call [residual_count=3] %1 %2 %3 %0 [
                forward={
                    lambda %0:i32[], %1:i32[], %2:dimension<n ∈ [4, 6)>, %3:f64[n, 4] .
                    let %4:f64[2, 4] = dynamic_slice [sizes=[2, 4]] %3 %0 %1
                    in (%4)
                },
                transpose={
                    lambda %0:i32[], %1:i32[], %2:dimension<n ∈ [4, 6)>, %3:f64[2, 4] .
                    let %4:f64[n, 4] = zero [type=f64[n, 4]] %2
                        %5:f64[n, 4] = dynamic_update_slice %4 %3 %0 %1
                    in (%5)
                },
            ]
            in (%4)
        "}
        .trim_end(),
    );
    assert_eq!(
        execute(&linearization, &pullback, 4, &four_rows, 1, &[2, 4]),
        vec![four_rows_in_bounds.to_vec(), vec![1.; 8]],
    );
    assert_eq!(
        execute(&linearization, &pullback, 5, &five_rows, 1, &[2, 4]),
        vec![five_rows_in_bounds.to_vec(), vec![1.; 8]],
    );
    assert_eq!(
        execute(&linearization, &pullback, 4, &four_rows, 9, &[2, 4]),
        vec![four_rows_clamped.to_vec(), vec![1.; 8]],
    );
    assert_eq!(
        execute(&linearization, &pullback, 5, &five_rows, 9, &[2, 4]),
        vec![five_rows_clamped.to_vec(), vec![1.; 8]],
    );
}

#[test]
fn test_compiled_dynamic_slice_runtime_size_pullback() {
    // The size comes from device data and changes without recompilation. Seed the pullback with the sliced
    // values so this fixture tests both the strided forward and point-scatter adjoint, including an empty result.
    let client = execution_client();
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    let input_type = replicated_vector_type(&mesh, 6);
    let size_type = replicated_scalar_type(&mesh, DataType::I64);
    let mut builder = ProgramBuilder::new();
    let input = builder.add_input(input_type.clone().into());
    let size = builder.add_input(size_type.clone().into());
    let size = builder
        .add_instruction(
            DimensionFromScalarOperation::new(
                DimensionVariable::new("size", DimensionBounds::new(0, Some(4)).unwrap()).into(),
            ),
            Vec::new(),
            vec![size],
            None,
        )
        .unwrap()[0];
    let start = builder.add_constant(Constant::Dimension(DimensionValue::constant(9).unwrap()));
    let output = builder
        .add_instruction(
            DynamicSliceOperation::<ArrayIrType>::from_rank(1)
                .with_strides(vec![2])
                .unwrap()
                .with_bounds(ryft_core::DynamicSliceBounds::Clamp),
            Vec::new(),
            vec![input, start, size],
            None,
        )
        .unwrap()[0];
    let program = builder
        .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
        .unwrap();
    let linearization = program.linearize_with_respect_to(&[0]).unwrap();
    let pullback = linearization.pullback().unwrap().into_program();
    let context = TracingContext::<Constant, Operation>::new();
    let inputs = vec![context.input(input_type.clone().into()), context.input(size_type.clone().into())];
    let mut outputs = linearization.primal().interpret_in_context(&context, inputs).unwrap();
    let mut arguments = vec![outputs[0].clone()];
    arguments.extend(outputs.split_off(1));
    outputs.extend(pullback.interpret_in_context(&context, arguments).unwrap());
    let combined = context
        .builder()
        .borrow()
        .clone()
        .build::<Vec<Constant>, Vec<Constant>>(
            outputs.iter().map(|output| output.atom_id().unwrap()).collect(),
            vec![Placeholder; 2],
            vec![Placeholder; 2],
        )
        .unwrap();
    let lowered = domain.lower_program(&combined, 0, &Options::new(mesh.clone())).unwrap();
    let compiled = domain.compile_program(&lowered).unwrap();
    let input =
        Array::from_host_buffer(&domain, input_type, mesh.clone(), values_to_bytes(&[0_f32, 1., 2., 3., 4., 5.]))
            .unwrap();
    let size = Array::from_host_buffer(&domain, size_type.clone(), mesh.clone(), values_to_bytes(&[2_i64])).unwrap();
    let outputs = domain.execute_program(&compiled, vec![input.clone(), size]).unwrap();
    assert_eq!(outputs[0].shape().as_slice(), &[2]);
    assert_eq!(read_f32s(&client, &outputs[0]), vec![3., 5.]);
    assert_eq!(read_f32s(&client, &outputs[1]), vec![0., 0., 0., 3., 0., 5.]);
    let size = Array::from_host_buffer(&domain, size_type, mesh, values_to_bytes(&[0_i64])).unwrap();
    let outputs = domain.execute_program(&compiled, vec![input, size]).unwrap();
    assert_eq!(outputs[0].shape().as_slice(), &[0]);
    assert_eq!(read_f32s(&client, &outputs[1]), vec![0_f32; 6]);
}

#[test]
fn test_compiled_dynamic_slice_runtime_input_pullback() {
    use ryft_core::{Add, DimensionConstant, Div};

    let client = execution_client();
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    let input_type = ArrayType::new_static(DataType::F32, [4, 5]);
    let scalar_type = ArrayType::scalar(DataType::I64);
    let context = TracingContext::<Constant, Operation>::new();
    let input = context.input(input_type.clone().into());
    let rows = context
        .input(scalar_type.clone().into())
        .to_dimension(DimensionVariable::new("rows", DimensionBounds::new(0, Some(5)).unwrap()))
        .unwrap();
    let columns = context
        .input(scalar_type.clone().into())
        .to_dimension(DimensionVariable::new("columns", DimensionBounds::new(0, Some(6)).unwrap()))
        .unwrap();
    let zero = context.dimension_constant(0).unwrap();
    let one = ValueProjection::<DimensionType>::into_projected(context.dimension_constant(1).unwrap()).unwrap();
    let two = ValueProjection::<DimensionType>::into_projected(context.dimension_constant(2).unwrap()).unwrap();
    let height = ValueProjection::<DimensionType>::into_projected(rows.clone()).unwrap();
    let height = height.add(&one).unwrap().div(&two).unwrap().into_value();
    let width = ValueProjection::<DimensionType>::into_projected(columns.clone()).unwrap();
    let width = width.add(&one).unwrap().div(&two).unwrap().into_value();
    // The inner slice receives a genuinely runtime-shaped array. Its pullback must allocate that exact shape,
    // before the outer prefix slice embeds it back into the original static input shape.
    let prefix = input
        .dynamic_slice_with_dimensions(&[zero.clone(), zero.clone()], &[rows, columns], &[1, 1])
        .unwrap();
    let output = prefix.dynamic_slice_with_dimensions(&[zero.clone(), zero], &[height, width], &[2, 2]).unwrap();
    let program = context
        .builder()
        .borrow()
        .clone()
        .build::<Vec<Constant>, Vec<Constant>>(vec![output.atom_id().unwrap()], vec![Placeholder; 3], vec![Placeholder])
        .unwrap();
    let linearization = program.linearize_with_respect_to(&[0]).unwrap();
    let pullback = linearization.pullback().unwrap().into_program();
    let context = TracingContext::<Constant, Operation>::new();
    let inputs = vec![
        context.input(input_type.clone().into()),
        context.input(scalar_type.clone().into()),
        context.input(scalar_type.clone().into()),
    ];
    let mut outputs = linearization.primal().interpret_in_context(&context, inputs).unwrap();
    let mut arguments = vec![outputs[0].clone()];
    arguments.extend(outputs.split_off(1));
    // Replaying arithmetic dimensions creates fresh identities. Instantiate this standalone pullback's
    // formal dimension identities for the replayed residual signature before importing its operations.
    let input_types = arguments.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
    let pullback = pullback.with_instantiated_type_identities(&input_types).unwrap();
    outputs.extend(pullback.interpret_in_context(&context, arguments).unwrap());
    let combined = context
        .builder()
        .borrow()
        .clone()
        .build::<Vec<Constant>, Vec<Constant>>(
            outputs.iter().map(|output| output.atom_id().unwrap()).collect(),
            vec![Placeholder; 3],
            vec![Placeholder; 2],
        )
        .unwrap();
    let lowered = domain.lower_program(&combined, 0, &Options::new(mesh.clone())).unwrap();
    let compiled = domain.compile_program(&lowered).unwrap();
    let values = [
        0_f32,
        1.,
        2.,
        3.,
        f32::NAN,
        5.,
        6.,
        7.,
        8.,
        f32::NAN,
        10.,
        11.,
        12.,
        13.,
        f32::NAN,
        f32::NAN,
        f32::NAN,
        f32::NAN,
        f32::NAN,
        f32::NAN,
    ];
    let input = Array::from_host_buffer(&domain, input_type, mesh.clone(), values_to_bytes(&values)).unwrap();
    let rows = Array::from_host_buffer(&domain, scalar_type.clone(), mesh.clone(), values_to_bytes(&[3_i64])).unwrap();
    let columns =
        Array::from_host_buffer(&domain, scalar_type.clone(), mesh.clone(), values_to_bytes(&[4_i64])).unwrap();
    let outputs = domain.execute_program(&compiled, vec![input.clone(), rows, columns.clone()]).unwrap();
    assert_eq!(outputs[0].shape().as_slice(), &[2, 2]);
    assert_eq!(read_f32s(&client, &outputs[0]), vec![0., 2., 10., 12.]);
    assert_eq!(
        read_f32s(&client, &outputs[1]),
        vec![0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 10., 0., 12., 0., 0., 0., 0., 0., 0., 0.]
    );
    let rows = Array::from_host_buffer(&domain, scalar_type, mesh, values_to_bytes(&[0_i64])).unwrap();
    let outputs = domain.execute_program(&compiled, vec![input, rows, columns]).unwrap();
    assert_eq!(outputs[0].shape().as_slice(), &[0, 2]);
    assert_eq!(read_f32s(&client, &outputs[1]), vec![0_f32; 20]);
}

#[test]
fn test_compiled_associative_scan_over_dynamic_unscanned_axis() {
    use ryft_core::{Cumulative, Max, associative_scan};

    // `associative_scan` keeps every unscanned axis whole, so a bounded dynamic row axis survives its slices, pads,
    // and interleaving, and one compiled program computes the prefix maxima of every row at each logical row
    // count, exactly like the reference backend's cumulative maximum.
    let client = execution_client();
    let mesh = domain_mesh(&client, "x", 1);
    let domain = Session::new(&client).domain().with_mesh(mesh.clone());
    let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 2);
    let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(5)).unwrap());
    let dynamic_type =
        ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Dynamic(rows), CoreDimension::Static(5)]))
            .with_sharding(sharding.clone())
            .unwrap();
    let staged = crate::jit::stage::<_, ArrayType, ArrayType>(
        |input| associative_scan(&input, 1, false, &|left, right| left.max(right)).unwrap(),
        dynamic_type,
        Options::new(mesh.clone()),
    )
    .unwrap()
    .into_inner();
    let compiled: ryft_core::compilation::CompiledFunction<Domain<'_>, ArrayIrType, ArrayIrType> =
        staged.lower(&domain).unwrap().compile(&domain).unwrap();
    for rows in [2usize, 4] {
        let values = (0..rows * 5).map(|value| ((value * 7) % 11) as f32).collect::<Vec<_>>();
        let input = Array::from_host_buffer(
            &domain,
            ArrayType::new_static(DataType::F32, [rows, 5]).with_sharding(sharding.clone()).unwrap(),
            mesh.clone(),
            values_to_bytes(values.as_slice()).as_slice(),
        )
        .unwrap();
        let output = compiled.executable().execute(&domain, Value::Array(input)).unwrap();
        let Value::Array(output) = output else {
            panic!("array-only compiled function returned a first-class dimension");
        };
        output.block_until_ready().unwrap();
        assert_eq!(output.shape().as_slice(), &[rows, 5]);
        let expected =
            CoreArray::matrix(rows, 5, values).unwrap().cumulative_max(1).unwrap().elements::<f32>().unwrap();
        assert_eq!(read_f32s(&client, &output), expected);
    }
}

#[test]
fn test_shard_map_emulation_matches_execution_on_cpu() {
    // The reference backend's lockstep emulation of `shard_map` and XLA's manual computation agree exactly on
    // boundary splitting and assembly, on every collective over the manual axis, and on reference accesses and
    // write-back (including reads of a replicated reference at device-varying indices and invariant mutations of a
    // replicated reference).
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin
        .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
        .unwrap();
    assert_shard_map_emulation_matches_execution(&client, ShardMapDifferentialCase::IdentityAndSum);
    assert_shard_map_emulation_matches_execution(&client, ShardMapDifferentialCase::AxisIndex);
    assert_shard_map_emulation_matches_execution(&client, ShardMapDifferentialCase::AllGather);
    assert_shard_map_emulation_matches_execution(&client, ShardMapDifferentialCase::AllToAll);
    assert_shard_map_emulation_matches_execution(&client, ShardMapDifferentialCase::Permute);
    assert_shard_map_emulation_matches_execution(&client, ShardMapDifferentialCase::Reference);
    assert_shard_map_emulation_matches_execution(&client, ShardMapDifferentialCase::VaryingIndexReference);
    assert_shard_map_emulation_matches_execution(&client, ShardMapDifferentialCase::ReplicatedReference);
}

#[test]
fn test_shard_map_pullback_emulation_matches_execution_on_cpu() {
    // The pullback of a read of a replicated reference at device-varying indices scatters each device's cotangent
    // into a local buffer that varies along `x`, sums the buffers across `x`, and adds the sum into the replicated
    // cotangent reference, both in the emulation and in XLA's manual computation. The pullbacks of an update of a
    // sharded reference and of an invariant mutation of a replicated reference agree as well.
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin
        .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
        .unwrap();
    assert_shard_map_pullback_emulation_matches_execution(&client, ShardMapDifferentialCase::Reference);
    let case = ShardMapDifferentialCase::VaryingIndexReference;
    assert_shard_map_pullback_emulation_matches_execution(&client, case);
    let case = ShardMapDifferentialCase::ReplicatedReference;
    assert_shard_map_pullback_emulation_matches_execution(&client, case);
}

#[test]
fn test_nested_shard_map_emulation_matches_execution_on_cpu() {
    // A nested map whose inner body sums across the enclosing manual axis runs on all four devices of a 2×2 mesh,
    // both in the emulation and in XLA's nested manual computations.
    let plugin = load_cpu_plugin().unwrap();
    let client = plugin
        .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
        .unwrap();
    assert_shard_map_emulation_matches_execution(&client, ShardMapDifferentialCase::Nested);
}
