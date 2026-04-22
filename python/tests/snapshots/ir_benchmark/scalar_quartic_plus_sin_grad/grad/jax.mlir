module @jit_quartic_plus_sin attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<f64>) -> (tensor<f64> {jax.result_info = "result"}) {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
    %1 = stablehlo.multiply %0, %arg0 : tensor<f64>
    %2 = stablehlo.cosine %arg0 : tensor<f64>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %3 = stablehlo.multiply %cst, %2 : tensor<f64>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %4 = stablehlo.multiply %1, %cst_0 : tensor<f64>
    %5 = stablehlo.add %3, %4 : tensor<f64>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %6 = stablehlo.multiply %cst_1, %arg0 : tensor<f64>
    %7 = stablehlo.multiply %0, %6 : tensor<f64>
    %8 = stablehlo.add %5, %7 : tensor<f64>
    %9 = stablehlo.multiply %6, %arg0 : tensor<f64>
    %10 = stablehlo.multiply %arg0, %9 : tensor<f64>
    %11 = stablehlo.add %8, %10 : tensor<f64>
    %12 = stablehlo.multiply %9, %arg0 : tensor<f64>
    %13 = stablehlo.add %11, %12 : tensor<f64>
    return %13 : tensor<f64>
  }
}
