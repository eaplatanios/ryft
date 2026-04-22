module @jit_bilinear_sin attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<f64>, %arg1: tensor<f64>) -> (tensor<f64> {jax.result_info = "result"}) {
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<f64>
    %1 = stablehlo.sine %arg0 : tensor<f64>
    %2 = stablehlo.add %0, %1 : tensor<f64>
    return %2 : tensor<f64>
  }
}
