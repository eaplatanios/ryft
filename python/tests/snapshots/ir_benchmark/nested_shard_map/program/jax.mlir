module @jit__lambda attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=2, "y"=2]>
  func.func public @main(%arg0: tensor<8xf32>) -> (tensor<8xf32> {jax.result_info = "result"}) {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", ?}]>] out_shardings=[<@mesh, [{"x", ?}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
      %1 = sdy.manual_computation(%arg1) in_shardings=[<@mesh, [{"y", ?}]>] out_shardings=[<@mesh, [{"y", ?}]>] manual_axes={"y"} (%arg2: tensor<2xf32>) {
        %3 = stablehlo.add %arg2, %arg2 : tensor<2xf32>
        sdy.return %3 : tensor<2xf32>
      } : (tensor<4xf32>) -> tensor<4xf32>
      %2 = stablehlo.add %1, %arg1 : tensor<4xf32>
      sdy.return %2 : tensor<4xf32>
    } : (tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
