/// StableHLO lowering of broadcasting operations.
pub(crate) mod broadcasting;
/// StableHLO lowering of concatenation operations. Concatenation result refinements resize bounded storage, so
/// reconciling a different result extent or allocation bound requires finite input and output allocation bounds.
pub(crate) mod concatenation;
/// StableHLO lowering of element type conversion operations.
pub(crate) mod conversions;
/// StableHLO lowering of gathering operations.
pub(crate) mod gathering;
/// StableHLO lowering of memory placement operations.
pub(crate) mod memory;
/// StableHLO lowering of padding operations. Padding result refinements resize bounded storage, so reconciling a
/// different result extent or allocation bound requires finite input and output allocation bounds.
pub(crate) mod padding;
/// StableHLO lowering of reshaping operations.
pub(crate) mod reshaping;
/// StableHLO lowering of reversing operations.
pub(crate) mod reversing;
/// StableHLO lowering of scattering operations.
pub(crate) mod scattering;
/// StableHLO lowering of slicing operations.
pub(crate) mod slicing;
/// StableHLO lowering of transposition operations.
pub(crate) mod transposition;

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::operations::manipulation::indexing::index;
    use ryft_core::{
        ArrayIrType, ArrayOperation, ArrayType, BatchAxis, ConcatenateOperation, DataType, Dimension as CoreDimension,
        DimensionAddOperation, DimensionBounds, DimensionMulOperation, DimensionOperation, DimensionSize,
        DimensionSizeOperation, DimensionValue, DimensionVariable, DynamicBroadcastOperation, DynamicGather,
        DynamicReshape, DynamicReshapeOperation, DynamicScatter, DynamicSlice, DynamicSliceOperation,
        DynamicSliceWithDimensions, Gather, GatherDimensionNumbers, GatherMode, GatherOperation, GatherOptions,
        Indexing, Placeholder, Reshape, Scatter, ScatterMode, ScatterOptions, ScatterReductionKind, Shape,
        TracingContext, Typed, TypedProgram, ValueProjection, batch,
    };
    use ryft_pjrt::Client;
    #[cfg(feature = "cuda-13")]
    use ryft_pjrt::{ClientOptions, GpuClientOptions, GpuMemoryAllocator, GpuPlatform, load_cuda_13_plugin};

    use crate::domains::tests::{domain_mesh, read_f64s};
    use crate::experimental::ops::{Constant, Operation, ProgramBuilder as ArrayIrProgramBuilder};
    use crate::tests::{execution_client, values_to_bytes};
    use crate::{Array, Options, Session};

    /// Checks compiled basic and broadcast advanced reads and repeated indexed updates on the provided backend.
    fn assert_compiled_indexing(client: &Client<'_>) {
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let input_types = vec![
            ArrayType::new_static(DataType::F64, [3, 4]),
            ArrayType::new_static(DataType::I64, [2, 1]),
            ArrayType::new_static(DataType::I64, [3]),
            ArrayType::new_static(DataType::F64, [2, 3]),
        ];
        let compiled = crate::compile::<_, Vec<ArrayType>, Vec<ArrayType>>(
            |inputs| {
                let selection = index![&inputs[1], &inputs[2]];
                vec![
                    inputs[0].at(&index![.. by -1, 1..4 by 2]).get(&GatherOptions::new()).unwrap(),
                    inputs[0].at(&selection).get(&GatherOptions::new()).unwrap(),
                    inputs[0].at(&selection).add(&inputs[3], &ScatterOptions::new()).unwrap(),
                    inputs[0].at(&index![.. by i128::MIN, ...]).get(&GatherOptions::new()).unwrap(),
                ]
            },
            input_types.clone(),
            &domain,
            mesh.clone(),
        )
        .unwrap();
        let bytes = [
            values_to_bytes(&[0_f64, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.]),
            values_to_bytes(&[2_i64, 0]),
            values_to_bytes(&[1_i64, 1, 3]),
            values_to_bytes(&[1_f64; 6]),
        ];
        let inputs = input_types
            .into_iter()
            .zip(bytes)
            .map(|(r#type, bytes)| Array::from_host_buffer(&domain, r#type, mesh.clone(), bytes).unwrap())
            .collect::<Vec<_>>();
        let outputs = compiled.executable().execute(&domain, inputs).unwrap();
        assert_eq!(outputs[0].shape().as_slice(), &[3, 2]);
        assert_eq!(read_f64s(&client, &outputs[0]), vec![9., 11., 5., 7., 1., 3.]);
        assert_eq!(outputs[1].shape().as_slice(), &[2, 3]);
        assert_eq!(read_f64s(&client, &outputs[1]), vec![9., 9., 11., 1., 1., 3.]);
        // A huge negative stride selects only the last row and must not wrap when lowered to signed attributes.
        assert_eq!(outputs[3].shape().as_slice(), &[1, 4]);
        assert_eq!(read_f64s(&client, &outputs[3]), vec![8., 9., 10., 11.]);
        // Duplicate advanced coordinates contribute twice; the original input remains a value, not a mutable view.
        assert_eq!(read_f64s(&client, &outputs[2]), vec![0., 3., 2., 4., 4., 5., 6., 7., 8., 11., 10., 12.]);
    }

    /// Replays one symbolic indexing graph at distinct concrete extents on the provided backend.
    fn assert_compiled_indexing_symbolic_extent(client: &Client<'_>) {
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let extent = CoreDimension::Dynamic(DimensionVariable::new("n", DimensionBounds::new(4, Some(6)).unwrap()));
        let staged = crate::stage::<_, Vec<ArrayType>, Vec<ArrayType>>(
            |inputs| {
                // Runtime dimension operations live in the mixed IR family. Explicitly retain that family while
                // indexing symbolic extents, and project the array results back to the public JIT signature.
                let inputs = inputs.into_iter().map(|input| input.into_value()).collect::<Vec<_>>();
                let selection = index![&inputs[1]];
                vec![
                    inputs[0].at(&selection).get(&GatherOptions::new()).unwrap().into_projected().unwrap(),
                    inputs[0].at(&selection).add(&inputs[2], &ScatterOptions::new()).unwrap().into_projected().unwrap(),
                ]
            },
            vec![
                ArrayType::new(DataType::F64, Shape::new(vec![extent])),
                ArrayType::new_static(DataType::I64, [3]),
                ArrayType::new_static(DataType::F64, [3]),
            ],
            Options::new(mesh.clone()),
        )
        .unwrap()
        .into_inner();
        let program = staged.source_program().program().clone();
        let linearization = program.linearize_with_respect_to(&[2]).unwrap();
        let pullback = linearization.pullback().unwrap().into_program();
        // Reuse one symbolic source graph at distinct concrete extents, including negative index normalization.
        let execute = |values: &[f64]| {
            let input_types = vec![
                ArrayType::new_static(DataType::F64, [values.len()]),
                ArrayType::new_static(DataType::I64, [3]),
                ArrayType::new_static(DataType::F64, [3]),
                ArrayType::new_static(DataType::F64, [3]),
                ArrayType::new_static(DataType::F64, [values.len()]),
            ];
            let compiled = crate::compile::<_, Vec<ArrayType>, Vec<ArrayType>>(
                |inputs| {
                    let mut inputs = inputs.into_iter().map(|input| input.into_value()).collect::<Vec<_>>();
                    let mut cotangents = inputs.split_off(3);
                    let context = inputs[0].context().clone();
                    let mut outputs = linearization.primal().interpret_in_context(&context, inputs).unwrap();
                    cotangents.extend(outputs.split_off(2));
                    outputs.extend(pullback.interpret_in_context(&context, cotangents).unwrap());
                    outputs
                        .into_iter()
                        .map(|output| ValueProjection::<ArrayType>::into_projected(output).unwrap())
                        .collect()
                },
                input_types.clone(),
                &domain,
                mesh.clone(),
            )
            .unwrap();
            let bytes = [
                values_to_bytes(values),
                values_to_bytes(&[1_i64, 1, -1]),
                values_to_bytes(&[10_f64, 20., 30.]),
                values_to_bytes(&[0_f64; 3]),
                values_to_bytes(values),
            ];
            let inputs = input_types
                .into_iter()
                .zip(bytes)
                .map(|(r#type, bytes)| Array::from_host_buffer(&domain, r#type, mesh.clone(), bytes).unwrap())
                .collect::<Vec<_>>();
            compiled
                .executable()
                .execute(&domain, inputs)
                .unwrap()
                .iter()
                .map(|output| read_f64s(&client, output))
                .collect::<Vec<_>>()
        };
        assert_eq!(execute(&[1., 2., 3., 4.]), vec![vec![2., 2., 4.], vec![1., 32., 3., 34.], vec![2., 2., 4.]]);
        assert_eq!(
            execute(&[1., 2., 3., 4., 5.]),
            vec![vec![2., 2., 5.], vec![1., 32., 3., 4., 35.], vec![2., 2., 5.]]
        );
    }

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
            let dynamic_query =
                matches!(name, "dynamic_gather_query" | "dynamic_scatter_query" | "column_scatter_query");
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
            let mut builder = ArrayIrProgramBuilder::new();
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
                let staged = crate::stage::<_, Vec<ArrayType>, Vec<ArrayType>>(
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
                assert_eq!(
                    specialized.output_types(),
                    vec![ArrayIrType::from(output_type.clone())],
                    "{name}, n={size}"
                );
                let compiled = crate::compile::<_, Vec<ArrayType>, Vec<ArrayType>>(
                    |inputs| {
                        let mut inputs = inputs.into_iter().map(|input| input.into_value()).collect::<Vec<_>>();
                        let cotangent = inputs.pop().unwrap();
                        let context = inputs[0].context().clone();
                        let mut outputs =
                            linearization.primal().interpret_in_context(&context, inputs).unwrap_or_else(|error| {
                                panic!("{name}, n={size} primal: {error}\n{}", linearization.primal())
                            });
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
                let outputs = compiled.executable().execute(&domain, inputs).unwrap();
                assert_eq!(outputs.len(), 2);
                for output in &outputs {
                    assert_eq!(output.data_type(), DataType::F64);
                    assert_eq!(output.mesh(), &mesh);
                }
                assert_eq!(outputs[0].shape().as_slice(), output_type.static_shape().unwrap().as_slice());
                assert_eq!(read_f64s(&client, &outputs[0]), expected, "{name}, n={size}");
                let mut gradient =
                    vec![if name == "broadcast" || name == "concatenate" { 2_f64 } else { 1_f64 }; size * 4];
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
}
