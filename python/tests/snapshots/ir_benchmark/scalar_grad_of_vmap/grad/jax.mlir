module @jit_mapped_sum attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<f64>) -> (tensor<f64> {jax.result_info = "result"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
    %3 = stablehlo.cosine %2 : tensor<2xf64>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %5 = stablehlo.pad %4, %cst_0, low = [1], high = [0], interior = [0] : (tensor<1xf64>, tensor<f64>) -> tensor<2xf64>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %7 = stablehlo.pad %6, %cst_2, low = [0], high = [1], interior = [0] : (tensor<1xf64>, tensor<f64>) -> tensor<2xf64>
    %8 = stablehlo.add %5, %7 : tensor<2xf64>
    %9 = stablehlo.multiply %8, %3 : tensor<2xf64>
    %10 = stablehlo.multiply %2, %8 : tensor<2xf64>
    %11 = stablehlo.add %9, %10 : tensor<2xf64>
    %12 = stablehlo.multiply %8, %2 : tensor<2xf64>
    %13 = stablehlo.add %11, %12 : tensor<2xf64>
    %14 = stablehlo.slice %13 [0:1] : (tensor<2xf64>) -> tensor<1xf64>
    %15 = stablehlo.slice %13 [1:2] : (tensor<2xf64>) -> tensor<1xf64>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %16 = stablehlo.reduce(%15 init: %cst_3) applies stablehlo.add across dimensions = [0] : (tensor<1xf64>, tensor<f64>) -> tensor<f64>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %17 = stablehlo.reduce(%14 init: %cst_4) applies stablehlo.add across dimensions = [0] : (tensor<1xf64>, tensor<f64>) -> tensor<f64>
    %18 = stablehlo.add %16, %17 : tensor<f64>
    return %18 : tensor<f64>
  }
}
