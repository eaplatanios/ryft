module @jit__lambda attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=4]>
  func.func public @main(%arg0: tensor<8xf32>) -> (tensor<8xf32> {jax.result_info = "result"}) {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
      %1 = stablehlo.cosine %arg1 : tensor<2xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2xf32>
      %3 = stablehlo.multiply %2, %1 : tensor<2xf32>
      sdy.return %3 : tensor<2xf32>
    } : (tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
