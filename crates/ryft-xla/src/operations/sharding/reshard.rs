use ryft_core::{ArrayType, ReshardOperation};
use ryft_mlir::ValueRef;

use crate::operations::sharding::constrain_sharding::lower_sharding_constraint;
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer};

impl LowerableOperation for ReshardOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        _output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        lower_sharding_constraint(
            input_values,
            self.sharding(),
            &lowerer.collective_state.bound_manual_axes,
            &mut lowerer.block,
            lowerer.location,
        )
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        Add, ArrayType, Cos, DataType, Device, DeviceMesh, DomainTracer, LogicalMesh, MeshAxis, MeshAxisType, Mul,
        ProjectedValue, Reduce, ShardMapOptions, Sharding, ShardingDimension, Sin, Typed, ValueProjection, auto_axes,
        differentiate_at, explicit_axes, shard_map,
    };
    use ryft_pjrt::{ClientOptions, CpuClientOptions, load_cpu_plugin};

    use crate::compilation::{CompileTracer, CompiledFunction, compile, compile_with_options};
    use crate::experimental::ArrayTracer;
    use crate::experimental::tracing::{TracedProgram, trace};
    use crate::tests::{values_from_bytes, values_to_bytes};
    use crate::{Array, Domain, FromPjrt, Options, Session};

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

    /// Returns the elements of the buffer that every addressable device holds for `array`, in device order.
    fn read_device_f32s(array: &Array<'_>) -> Vec<Vec<f32>> {
        array
            .addressable_shards()
            .map(|shard| {
                let bytes = shard.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
                values_from_bytes::<f32>(bytes.as_slice())
            })
            .collect()
    }

    #[test]
    fn test_reshard_lowering_opens_dimensions_over_auto_axes() {
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
    fn test_auto_axes_executes_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
        let logical_mesh = explicit_mesh();
        let devices = client.addressable_devices().unwrap();
        let mesh = DeviceMesh::new(
            logical_mesh.clone(),
            devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [4, 2])
            .with_sharding(sharding(&logical_mesh, &[Some("x"), Some("y")]))
            .unwrap();
        let output_sharding = sharding(&logical_mesh, &[Some("x")]);
        // Each device holds the rows of its position along `x`.
        let expected = vec![
            vec![1.6829419136047363f32, 2.100834846496582],
            vec![1.6829419136047363, 2.100834846496582],
            vec![-3.4314537048339844, 0.7551422119140625],
            vec![-3.4314537048339844, 0.7551422119140625],
        ];
        let input = Array::from_host_buffer(
            &domain,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes(&[0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        )
        .unwrap();

        // The switch compiles into one program whose values use meshes that differ only in axis types, and executes
        // eagerly as well, both with the values of the JAX oracle `auto_axes(x)`.
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> = compile(
            {
                let output_sharding = output_sharding.clone();
                move |input: CompileTracer<'_>| {
                    auto_axes(
                        |input: CompileTracer<'_>| {
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
            &domain,
            mesh,
        )
        .unwrap();
        let output = compiled.executable().execute(&domain, input.clone()).unwrap();
        assert_eq!(output.r#type().sharding(), Some(&output_sharding));
        assert_eq!(read_device_f32s(&output), expected);

        let output = auto_axes(
            |input: Array<'_>| {
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
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
        let logical_mesh = explicit_mesh().with_axis_types(["x", "y"], MeshAxisType::Auto).unwrap();
        let devices = client.addressable_devices().unwrap();
        let mesh = DeviceMesh::new(
            logical_mesh.clone(),
            devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [4, 2])
            .with_sharding(Sharding::replicated(logical_mesh.clone(), 2))
            .unwrap();
        let input = Array::from_host_buffer(
            &domain,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes(&[0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        )
        .unwrap();
        let input_sharding = sharding(&logical_mesh, &[Some("x"), None]);
        let output_sharding = Sharding::replicated(logical_mesh, 1);

        // The output has the values of the JAX oracle `explicit_axes(x)`, whether the switch is compiled as one program
        // or executed eagerly. Its type is replicated, because types record no placement over auto mesh axes, while
        // the compiler chooses its runtime placement over them.
        let expected = vec![1.5403022766113281f32, -1.4061393737792969, -0.36998140811920166, 1.7140724658966064];
        let read = |array: &Array<'_>| {
            values_from_bytes::<f32>(&crate::arrays_v0::host::materialize_dense_array_bytes(array).unwrap())
        };
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> = compile(
            {
                let input_sharding = input_sharding.clone();
                move |input: CompileTracer<'_>| {
                    explicit_axes(
                        |input: CompileTracer<'_>| input.cos()?.reduce_sum(&[1], None),
                        input,
                        Some(&["x"]),
                        input_sharding,
                    )
                    .unwrap()
                }
            },
            input_type,
            &domain,
            mesh,
        )
        .unwrap();
        let output = compiled.executable().execute(&domain, input.clone()).unwrap();
        assert_eq!(output.r#type().sharding(), Some(&output_sharding));
        assert_eq!(read(&output), expected);
        let output =
            explicit_axes(|input: Array<'_>| input.cos()?.reduce_sum(&[1], None), input, Some(&["x"]), input_sharding)
                .unwrap();
        assert_eq!(output.r#type().sharding(), Some(&output_sharding));
        assert_eq!(read(&output), expected);
    }

    #[test]
    fn test_auto_axes_of_shard_map_outputs_executes_on_cpu() {
        // An eager `shard_map` over the mesh with `x` manual returns its output typed over the explicit mesh of its
        // input, as JAX's `shard_map` types its outputs over its context mesh, so `auto_axes` switches it directly.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
        let logical_mesh = explicit_mesh();
        let map_mesh = logical_mesh.with_axis_types(["x"], MeshAxisType::Manual).unwrap();
        let devices = client.addressable_devices().unwrap();
        let mesh = DeviceMesh::new(
            logical_mesh.clone(),
            devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let input_sharding = sharding(&logical_mesh, &[Some("x"), Some("y")]);
        let map_sharding = sharding(&map_mesh, &[Some("x"), Some("y")]);
        let input_type = ArrayType::new_static(DataType::F32, [4, 2]).with_sharding(input_sharding.clone()).unwrap();
        let input = Array::from_host_buffer(
            &domain,
            input_type.clone(),
            mesh,
            values_to_bytes(&[0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        )
        .unwrap();
        let doubled = shard_map(
            |input: ProjectedValue<ArrayType, DomainTracer<Domain<'_>>>| Ok(input.clone() + input),
            input,
            map_mesh,
            map_sharding.clone(),
            map_sharding,
            ShardMapOptions::new(),
        )
        .unwrap();
        assert_eq!(doubled.r#type().as_ref(), &input_type);
        let output =
            auto_axes(|doubled: Array<'_>| doubled.sin(), doubled, Some(&["y"]), input_sharding.clone()).unwrap();
        assert_eq!(output.r#type().sharding(), Some(&input_sharding));

        // Each device holds the sines of its `2x1` block of the doubled matrix, whose rows its position along `x` and
        // whose column its position along `y` selects.
        let expected =
            [[0.0f32, -0.7568025], [0.9092974, -0.2794155], [0.98935825, -0.53657292], [-0.54402111, 0.99060736]];
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
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();
        let logical_mesh = explicit_mesh().with_axis_types(["x", "y"], MeshAxisType::Auto).unwrap();
        let devices = client.addressable_devices().unwrap();
        let mesh = DeviceMesh::new(
            logical_mesh.clone(),
            devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let input_placement = sharding(&logical_mesh, &[Some("x"), Some("y")]);
        let input_type = ArrayType::new_static(DataType::F32, [4, 2]);
        let input = Array::from_host_buffer_with_sharding(
            &domain,
            input_type.clone(),
            input_placement.clone(),
            mesh.clone(),
            values_to_bytes(&[0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        )
        .unwrap();
        let input_sharding = sharding(&logical_mesh, &[Some("x"), None]);
        let compiled: CompiledFunction<'_, ArrayType, ArrayType> = compile_with_options(
            move |input: CompileTracer<'_>| {
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
            &domain,
            Options::new(mesh).with_input_shardings(vec![input_placement.clone()]),
        )
        .unwrap();
        let gradient = compiled.executable().execute(&domain, input.clone()).unwrap();
        assert_eq!(gradient.r#type(), input.r#type());
        assert_eq!(gradient.sharding(), &input_placement);
        let gradient = crate::arrays_v0::host::materialize_dense_array_bytes(&gradient).unwrap();
        let expected = [1.0f32, -0.30116868, -2.2347417, -1.4133525, 2.3735664, 5.078284, 2.6366632, -3.8450038];
        for (actual, expected) in values_from_bytes::<f32>(gradient.as_slice()).iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5, "expected {expected} but got {actual}");
        }
    }
}
