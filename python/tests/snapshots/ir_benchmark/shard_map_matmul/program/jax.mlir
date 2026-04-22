module @jit__lambda attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=4]>
  func.func public @main(%arg0: tensor<8x4xf32>, %arg1: tensor<4x2xf32>) -> (tensor<8x2xf32> {jax.result_info = "result"}) {
    %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {}]>] out_shardings=[<@mesh, [{"x"}, {}]>] manual_axes={"x"} (%arg2: tensor<2x4xf32>, %arg3: tensor<4x2xf32>) {
      %1 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf32>, tensor<4x2xf32>) -> tensor<2x2xf32>
      sdy.return %1 : tensor<2x2xf32>
    } : (tensor<8x4xf32>, tensor<4x2xf32>) -> tensor<8x2xf32>
    return %0 : tensor<8x2xf32>
  }
}
