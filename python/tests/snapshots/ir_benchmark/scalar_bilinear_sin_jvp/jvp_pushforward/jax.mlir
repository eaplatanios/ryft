module @jit__unnamed_function attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<f64>, %arg1: tensor<f64>) -> (tensor<f64> {jax.result_info = "result"}) {
    %cst = stablehlo.constant dense<-0.41614683654714241> : tensor<f64>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<f64>
    %0 = stablehlo.multiply %arg0, %cst_0 : tensor<f64>
    %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %1 = stablehlo.multiply %cst_1, %arg1 : tensor<f64>
    %2 = stablehlo.add %0, %1 : tensor<f64>
    %3 = stablehlo.multiply %arg0, %cst : tensor<f64>
    %4 = stablehlo.add %2, %3 : tensor<f64>
    return %4 : tensor<f64>
  }
}
