module @jit_scalar_hessian_style_second_derivative attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<f64>) -> (tensor<f64> {jax.result_info = "result"}) {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %1 = stablehlo.multiply %cst, %arg0 : tensor<f64>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %2 = stablehlo.multiply %arg0, %cst_0 : tensor<f64>
    %3 = stablehlo.add %1, %2 : tensor<f64>
    %4 = stablehlo.multiply %3, %arg0 : tensor<f64>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %5 = stablehlo.multiply %0, %cst_1 : tensor<f64>
    %6 = stablehlo.add %4, %5 : tensor<f64>
    %7 = stablehlo.sine %arg0 : tensor<f64>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %8 = stablehlo.multiply %cst_2, %7 : tensor<f64>
    %9 = stablehlo.negate %8 : tensor<f64>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %10 = stablehlo.multiply %cst_3, %9 : tensor<f64>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %11 = stablehlo.multiply %6, %cst_4 : tensor<f64>
    %12 = stablehlo.add %10, %11 : tensor<f64>
    %cst_5 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %13 = stablehlo.multiply %cst_5, %arg0 : tensor<f64>
    %cst_6 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %cst_7 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %14 = stablehlo.multiply %cst_6, %cst_7 : tensor<f64>
    %15 = stablehlo.multiply %3, %13 : tensor<f64>
    %16 = stablehlo.multiply %0, %14 : tensor<f64>
    %17 = stablehlo.add %15, %16 : tensor<f64>
    %18 = stablehlo.add %12, %17 : tensor<f64>
    %19 = stablehlo.multiply %13, %arg0 : tensor<f64>
    %20 = stablehlo.multiply %14, %arg0 : tensor<f64>
    %cst_8 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %21 = stablehlo.multiply %13, %cst_8 : tensor<f64>
    %22 = stablehlo.add %20, %21 : tensor<f64>
    %cst_9 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %23 = stablehlo.multiply %cst_9, %19 : tensor<f64>
    %24 = stablehlo.multiply %arg0, %22 : tensor<f64>
    %25 = stablehlo.add %23, %24 : tensor<f64>
    %26 = stablehlo.add %18, %25 : tensor<f64>
    %27 = stablehlo.multiply %22, %arg0 : tensor<f64>
    %cst_10 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %28 = stablehlo.multiply %19, %cst_10 : tensor<f64>
    %29 = stablehlo.add %27, %28 : tensor<f64>
    %30 = stablehlo.add %26, %29 : tensor<f64>
    return %30 : tensor<f64>
  }
}
