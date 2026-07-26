#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use ryft_core::operations::manipulation::Reshape;
    use ryft_core::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use ryft_core::types::{ArrayType, DataType, Dimension, Shape};

    use crate::experimental::shard_map::{ShardMapTracer, TracedShardMap, TracedXlaProgram, reshard, shard_map};
    use crate::experimental::trace;

    fn manual_mesh(axis_size: usize) -> LogicalMesh {
        LogicalMesh::new(vec![MeshAxis::new("x", axis_size, MeshAxisType::Manual).unwrap()]).unwrap()
    }

    #[test]
    fn test_trace_reshape_lowers_to_stablehlo_reshape() {
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            |x: ShardMapTracer| x.reshape(Shape::new(vec![Dimension::Static(3), Dimension::Static(2)])).unwrap(),
            input_type.clone(),
        )
        .unwrap();

        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
                    %0 = stablehlo.reshape %arg0 : (tensor<2x3xf32>) -> tensor<3x2xf32>
                    return %0 : tensor<3x2xf32>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_trace_reshape_and_reshard_render_stablehlo_and_shardy() {
        let mesh = manual_mesh(4);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let sharding = sharding.clone();
                move |x: ShardMapTracer| {
                    reshard(x, sharding.clone())
                        .expect("reshard should stage before reshape")
                        .reshape(Shape::new(vec![Dimension::Static(1), Dimension::Static(8), Dimension::Static(1)]))
                        .unwrap()
                }
            },
            input_type,
        )
        .unwrap();

        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32>) -> tensor<1x8x1xf32> {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{"x"}]> : tensor<8xf32>
                    %1 = stablehlo.reshape %0 : (tensor<8xf32>) -> tensor<1x8x1xf32>
                    return %1 : tensor<1x8x1xf32>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_shard_map_reshape_renders_singleton_axis_sharding_propagation() {
        let mesh = manual_mesh(4);
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));
        let input_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let output_sharding = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();

        let traced: TracedShardMap<ArrayType, ArrayType> = shard_map(
            |x: ShardMapTracer| {
                x.reshape(Shape::new(vec![Dimension::Static(1), Dimension::Static(2), Dimension::Static(1)]))
                    .unwrap()
            },
            global_input_type,
            mesh.clone(),
            input_sharding.clone(),
            output_sharding.clone(),
        )
        .unwrap();

        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<1x8x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{}, {"x"}, {}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = stablehlo.reshape %arg1 : (tensor<2xf32>) -> tensor<1x2x1xf32>
                      sdy.return %1 : tensor<1x2x1xf32>
                    } : (tensor<8xf32>) -> tensor<1x8x1xf32>
                    return %0 : tensor<1x8x1xf32>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_shard_map_reshape_renders_replicated_merge_sharding_propagation() {
        let mesh = manual_mesh(4);
        let global_input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Static(8), Dimension::Static(2), Dimension::Static(3)]),
        );
        let input_sharding = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated(), ShardingDimension::replicated()],
        )
        .unwrap();
        let output_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap();

        let traced: TracedShardMap<ArrayType, ArrayType> = shard_map(
            |x: ShardMapTracer| x.reshape(Shape::new(vec![Dimension::Static(2), Dimension::Static(6)])).unwrap(),
            global_input_type,
            mesh.clone(),
            input_sharding.clone(),
            output_sharding.clone(),
        )
        .unwrap();

        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8x2x3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) -> (tensor<8x6xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}, {}, {}]>] out_shardings=[<@mesh, [{"x"}, {}]>] manual_axes={"x"} (%arg1: tensor<2x2x3xf32>) {
                      %1 = stablehlo.reshape %arg1 : (tensor<2x2x3xf32>) -> tensor<2x6xf32>
                      sdy.return %1 : tensor<2x6xf32>
                    } : (tensor<8x2x3xf32>) -> tensor<8x6xf32>
                    return %0 : tensor<8x6xf32>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_shard_map_reshape_renders_replicated_split_sharding_propagation() {
        let mesh = manual_mesh(4);
        let global_input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8), Dimension::Static(6)]));
        let input_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap();
        let output_sharding = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated(), ShardingDimension::replicated()],
        )
        .unwrap();

        let traced: TracedShardMap<ArrayType, ArrayType> = shard_map(
            |x: ShardMapTracer| {
                x.reshape(Shape::new(vec![Dimension::Static(2), Dimension::Static(2), Dimension::Static(3)]))
                    .unwrap()
            },
            global_input_type,
            mesh.clone(),
            input_sharding.clone(),
            output_sharding.clone(),
        )
        .unwrap();

        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8x6xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x2x3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}, {}]>] out_shardings=[<@mesh, [{"x"}, {}, {}]>] manual_axes={"x"} (%arg1: tensor<2x6xf32>) {
                      %1 = stablehlo.reshape %arg1 : (tensor<2x6xf32>) -> tensor<2x2x3xf32>
                      sdy.return %1 : tensor<2x2x3xf32>
                    } : (tensor<8x6xf32>) -> tensor<8x2x3xf32>
                    return %0 : tensor<8x2x3xf32>
                  }
                }
            "#}
        );
    }
}
