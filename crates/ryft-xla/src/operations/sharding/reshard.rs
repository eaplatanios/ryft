use ryft_core::{ArrayType, ReshardOperation};
use ryft_mlir::ValueRef;

use crate::operations::sharding::constrain_sharding::lower_sharding_constraint;
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer};

impl LowerableOperation for ReshardOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (input_types, output_types);
        lower_sharding_constraint(
            input_values,
            self.sharding(),
            lowerer.collective_state().bound_manual_axes(),
            &mut lowerer.block(),
            lowerer.location(),
        )
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        Add, ArrayType, Cos, DataType, Device, DeviceMesh, Dimension, DomainTracer, LogicalMesh, MeshAxis,
        MeshAxisType, Mul, ProjectedValue, Reduce, Reshard, Shape, ShardMapOptions, Sharding, ShardingDimension, Sin,
        Typed, ValueProjection, auto_axes, differentiate_at, explicit_axes, shard_map,
    };
    use ryft_pjrt::Client;

    use crate::operations::sharding::shard_map::tests::read_device_f32s;
    use crate::sessions::tests::{read_first_shard, two_device_mesh};
    use crate::tests::{cpu_client, test_compilation_options, values_from_bytes, values_to_bytes};
    use crate::{
        Array, ArrayTracer, CompileTracer, CompiledFunction, FromPjrt, Options, Session, TracedProgram, compile,
        compile_with_options, trace,
    };

    /// Returns `logical_mesh` over the addressable devices of `client`.
    fn device_mesh(client: &Client<'_>, logical_mesh: LogicalMesh) -> DeviceMesh {
        let devices = client.addressable_devices().unwrap();
        DeviceMesh::new(logical_mesh, devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect())
            .unwrap()
    }

    /// Four-device mesh whose axes `x` and `y` are both explicit.
    fn explicit_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap()
    }

    /// Sharding over `mesh` that places each dimension along the axis that `dimensions` names for it, or replicates it
    /// where `dimensions` has [`None`].
    fn sharding(mesh: &LogicalMesh, dimensions: &[Option<&str>]) -> Sharding {
        let dimensions = dimensions
            .iter()
            .map(|axis| axis.map_or_else(ShardingDimension::replicated, |axis| ShardingDimension::sharded([axis])))
            .collect();
        Sharding::new(mesh.clone(), dimensions).unwrap()
    }

    #[test]
    fn test_reshard_operation_lower() {
        let mesh = explicit_mesh();
        let input_type = ArrayType::new_static(DataType::F32, [4, 2])
            .with_sharding(sharding(&mesh, &[Some("x"), Some("y")]))
            .unwrap();
        let output_sharding = sharding(&mesh, &[Some("x")]);
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            move |input: ArrayTracer| {
                auto_axes(
                    |input: ArrayTracer| {
                        let sine = input.sin()?;
                        sine.add(&sine)?.reduce_sum(&[1], None)
                    },
                    input,
                    Some(&["x"]),
                    output_sharding,
                )
                .unwrap()
            },
            input_type.clone(),
        )
        .unwrap();
        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2, "y"=2]>
                  func.func @main(%arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}], replicated={"y"}>}) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{?}, {"y", ?}]> : tensor<4x2xf32>
                    %1 = stablehlo.sine %0 : tensor<4x2xf32>
                    %2 = stablehlo.add %1, %1 : tensor<4x2xf32>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x2xf32>, tensor<f32>) -> tensor<4xf32>
                    %4 = sdy.sharding_constraint %3 <@mesh, [{"x"}], replicated={"y"}> : tensor<4xf32>
                    return %4 : tensor<4xf32>
                  }
                }
            "#},
        );
        let auto_mesh = mesh.with_axis_types(["x", "y"], MeshAxisType::Auto).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [4, 2])
            .with_sharding(Sharding::replicated(auto_mesh.clone(), 2))
            .unwrap();
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            move |input: ArrayTracer| {
                explicit_axes(
                    |input: ArrayTracer| input.cos()?.reduce_sum(&[1], None),
                    input,
                    Some(&["x"]),
                    sharding(&auto_mesh, &[Some("x"), None]),
                )
                .unwrap()
            },
            input_type,
        )
        .unwrap();
        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2, "y"=2]>
                  func.func @main(%arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{"x", ?}, {?}]> : tensor<4x2xf32>
                    %1 = stablehlo.cosine %0 : tensor<4x2xf32>
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x2xf32>, tensor<f32>) -> tensor<4xf32>
                    %3 = sdy.sharding_constraint %2 <@mesh, [{?}]> : tensor<4xf32>
                    return %3 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_reshard_operation_lower_chains_into_one_program() {
        // Multiple staged reshards inside one `compile` body compile into a single MLIR program with chained
        // `sdy.sharding_constraint` operations, which yields exactly one cache entry and exactly one PJRT execution per
        // call. In this asynchronously pipelined regime, PJRT runs the whole compiled program at once without a host
        // synchronization per reshard. The mesh axis is explicit, so each reshard is a tracked transition and the final
        // replicated sharding governs the output buffer.
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let engine = Session::builder()
            .client(client)
            .compilation_options(test_compilation_options(client, None))
            .build()
            .unwrap();

        let shape = Shape::new(vec![Dimension::Static(4)]);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let input_type = ArrayType::new(DataType::F32, shape.clone()).with_sharding(sharded.clone()).unwrap();
        let constraint_a = replicated.clone();
        let constraint_b = sharded.clone();
        let constraint_c = replicated;

        // Three staged reshards compose inside one closure. Each emits a `sdy.sharding_constraint` op into
        // the same MLIR program. After staging and compilation, the executable runs all three in one PJRT dispatch.
        let compiled: CompiledFunction<ArrayType, ArrayType> = compile(
            move |x| {
                let a = x.reshard(&constraint_a).unwrap();
                let b = a.sin().unwrap().reshard(&constraint_b).unwrap();
                b.sin().unwrap().reshard(&constraint_c).unwrap()
            },
            input_type.clone(),
            &engine,
            mesh.clone(),
        )
        .unwrap();

        // One compile means one cache entry for the whole pipeline.
        assert_eq!(engine.cached_program_count(), 1, "three staged reshards should compile into one program");

        let values = [0.1f32, 0.2, 0.3, 0.4];
        let source = Array::from_host_buffer(
            &engine,
            input_type,
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();
        let output = compiled.executable().execute(&engine, source).unwrap();

        // The last constraint replicates the final output, so every device sees the full vector.
        let observed = read_first_shard::<f32>(client, &output);
        for (got, &input) in observed.iter().zip(values.iter()) {
            let expected = input.sin().sin();
            assert!((got - expected).abs() < 1e-5, "got {got}, expected ~{expected}");
        }
    }

    #[test]
    fn test_auto_axes_executes_on_cpu() {
        let client = cpu_client(4);
        let session = Session::builder()
            .client(client)
            .compilation_options(test_compilation_options(client, None))
            .build()
            .unwrap();
        let logical_mesh = explicit_mesh();
        let mesh = device_mesh(client, logical_mesh.clone());
        let input_type = ArrayType::new_static(DataType::F32, [4, 2])
            .with_sharding(sharding(&logical_mesh, &[Some("x"), Some("y")]))
            .unwrap();
        let output_sharding = sharding(&logical_mesh, &[Some("x")]);
        // Each device holds the rows of its position along `x`.
        let expected = vec![
            vec![1.682_941_9_f32, 2.100_834_8],
            vec![1.682_941_9, 2.100_834_8],
            vec![-3.431_453_7, 0.755_142_2],
            vec![-3.431_453_7, 0.755_142_2],
        ];
        let input = Array::from_host_buffer(
            &session,
            input_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes(&[0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        )
        .unwrap();

        // The switch compiles into one program whose values use meshes that differ only in axis types, and executes
        // eagerly as well, both with the values of the JAX oracle `auto_axes(x)`.
        let compiled: CompiledFunction<ArrayType, ArrayType> = compile(
            {
                let output_sharding = output_sharding.clone();
                move |input: CompileTracer| {
                    auto_axes(
                        |input: CompileTracer| {
                            let sine = input.sin()?;
                            sine.add(&sine)?.reduce_sum(&[1], None)
                        },
                        input,
                        Some(&["x"]),
                        output_sharding,
                    )
                    .unwrap()
                }
            },
            input_type,
            &session,
            mesh,
        )
        .unwrap();
        let output = compiled.executable().execute(&session, input.clone()).unwrap();
        assert_eq!(output.r#type().sharding(), Some(&output_sharding));
        assert_eq!(read_device_f32s(&output), expected);

        let output = auto_axes(
            |input: Array| {
                let sine = input.sin()?;
                sine.add(&sine)?.reduce_sum(&[1], None)
            },
            input,
            Some(&["x"]),
            output_sharding.clone(),
        )
        .unwrap();
        assert_eq!(output.r#type().sharding(), Some(&output_sharding));
        assert_eq!(read_device_f32s(&output), expected);
    }

    #[test]
    fn test_explicit_axes_executes_on_cpu() {
        let client = cpu_client(4);
        let session = Session::builder()
            .client(client)
            .compilation_options(test_compilation_options(client, None))
            .build()
            .unwrap();
        let logical_mesh = explicit_mesh().with_axis_types(["x", "y"], MeshAxisType::Auto).unwrap();
        let mesh = device_mesh(client, logical_mesh.clone());
        let input_type = ArrayType::new_static(DataType::F32, [4, 2])
            .with_sharding(Sharding::replicated(logical_mesh.clone(), 2))
            .unwrap();
        let input = Array::from_host_buffer(
            &session,
            input_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes(&[0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        )
        .unwrap();
        let input_sharding = sharding(&logical_mesh, &[Some("x"), None]);
        let output_sharding = Sharding::replicated(logical_mesh, 1);

        // The output has the values of the JAX oracle `explicit_axes(x)`, whether the switch is compiled as one program
        // or executed eagerly. Its type is replicated, because types record no placement over auto mesh axes, while
        // the compiler chooses its runtime placement over them.
        let expected = vec![1.540_302_3_f32, -1.406_139_4, -0.369_981_4, 1.714_072_5];
        let read = |array: &Array| array.copy_to_host().unwrap().elements::<f32>().unwrap();
        let compiled: CompiledFunction<ArrayType, ArrayType> = compile(
            {
                let input_sharding = input_sharding.clone();
                move |input: CompileTracer| {
                    explicit_axes(
                        |input: CompileTracer| input.cos()?.reduce_sum(&[1], None),
                        input,
                        Some(&["x"]),
                        input_sharding,
                    )
                    .unwrap()
                }
            },
            input_type,
            &session,
            mesh,
        )
        .unwrap();
        let output = compiled.executable().execute(&session, input.clone()).unwrap();
        assert_eq!(output.r#type().sharding(), Some(&output_sharding));
        assert_eq!(read(&output), expected);
        let output =
            explicit_axes(|input: Array| input.cos()?.reduce_sum(&[1], None), input, Some(&["x"]), input_sharding)
                .unwrap();
        assert_eq!(output.r#type().sharding(), Some(&output_sharding));
        assert_eq!(read(&output), expected);
    }

    #[test]
    fn test_auto_axes_of_shard_map_outputs_executes_on_cpu() {
        // An eager `shard_map` over the mesh with `x` manual returns its output typed over the explicit mesh of its
        // input, as JAX's `shard_map` types its outputs over its context mesh, so `auto_axes` switches it directly.
        let client = cpu_client(4);
        let session = Session::builder()
            .client(client)
            .compilation_options(test_compilation_options(client, None))
            .build()
            .unwrap();
        let logical_mesh = explicit_mesh();
        let map_mesh = logical_mesh.with_axis_types(["x"], MeshAxisType::Manual).unwrap();
        let mesh = device_mesh(client, logical_mesh.clone());
        let input_sharding = sharding(&logical_mesh, &[Some("x"), Some("y")]);
        let map_sharding = sharding(&map_mesh, &[Some("x"), Some("y")]);
        let input_type = ArrayType::new_static(DataType::F32, [4, 2]).with_sharding(input_sharding.clone()).unwrap();
        let input = Array::from_host_buffer(
            &session,
            input_type.clone(),
            mesh,
            None,
            values_to_bytes(&[0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        )
        .unwrap();
        let doubled = shard_map(
            |input: ProjectedValue<ArrayType, DomainTracer<Session>>| Ok(input.clone() + input),
            input,
            map_mesh,
            map_sharding.clone(),
            map_sharding,
            ShardMapOptions::new(),
        )
        .unwrap();
        assert_eq!(doubled.r#type().as_ref(), &input_type);
        let output = auto_axes(|doubled: Array| doubled.sin(), doubled, Some(&["y"]), input_sharding.clone()).unwrap();
        assert_eq!(output.r#type().sharding(), Some(&input_sharding));

        // Each device holds the sines of its `2x1` block of the doubled matrix, whose rows its position along `x` and
        // whose column its position along `y` selects.
        let expected =
            [[0.0f32, -0.7568025], [0.9092974, -0.2794155], [0.98935825, -0.53657292], [-0.544_021_1, 0.990_607_4]];
        let actual = read_device_f32s(&output);
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-5, "got {actual}, expected {expected}");
            }
        }
    }

    #[test]
    fn test_explicit_axes_gradient_of_inputs_placed_along_auto_axes_executes_on_cpu() {
        // A concrete input is placed along the auto axes `x` and `y`, which its type does not record, so the compiled
        // function takes that placement as an input sharding override. The gradient through `explicit_axes` keeps the
        // input's placement, with the values of the JAX oracle `grad explicit_axes`.
        let client = cpu_client(4);
        let session = Session::builder()
            .client(client)
            .compilation_options(test_compilation_options(client, None))
            .build()
            .unwrap();
        let logical_mesh = explicit_mesh().with_axis_types(["x", "y"], MeshAxisType::Auto).unwrap();
        let mesh = device_mesh(client, logical_mesh.clone());
        let input_placement = sharding(&logical_mesh, &[Some("x"), Some("y")]);
        let input_type = ArrayType::new_static(DataType::F32, [4, 2]);
        let input = Array::from_host_buffer(
            &session,
            input_type.clone(),
            mesh.clone(),
            Some(input_placement.clone()),
            values_to_bytes(&[0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        )
        .unwrap();
        let input_sharding = sharding(&logical_mesh, &[Some("x"), None]);
        let compiled: CompiledFunction<ArrayType, ArrayType> = compile_with_options(
            move |input: CompileTracer| {
                let gradient = differentiate_at(input.into_value())
                    .gradient(|input| {
                        let input = ValueProjection::<ArrayType>::into_projected(input)?;
                        let output = explicit_axes(
                            |input| input.cos()?.mul(&input)?.reduce_sum(&[0, 1], None),
                            input,
                            Some(&["x"]),
                            input_sharding.clone(),
                        )?;
                        Ok(output.into_value())
                    })
                    .unwrap();
                ValueProjection::<ArrayType>::into_projected(gradient).unwrap()
            },
            input_type,
            &session,
            Options::new(mesh).with_input_shardings(vec![input_placement.clone()]),
        )
        .unwrap();
        let gradient = compiled.executable().execute(&session, input.clone()).unwrap();
        assert_eq!(gradient.r#type(), input.r#type());
        assert_eq!(gradient.sharding(), &input_placement);
        let gradient = gradient.copy_to_host_bytes().unwrap();
        let expected = [1.0f32, -0.30116868, -2.2347417, -1.4133525, 2.3735664, 5.078284, 2.6366632, -3.8450038];
        for (actual, expected) in values_from_bytes::<f32>(gradient.as_slice()).iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5, "expected {expected} but got {actual}");
        }
    }
}
