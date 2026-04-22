module @jit__lambda attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=4]>
  func.func public @main(%arg0: tensor<8xf32>) -> (tensor<8xf32> {jax.result_info = "result"}) {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
      %1 = stablehlo.sine %arg1 : tensor<2xf32>
      sdy.return %1 : tensor<2xf32>
    } : (tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
