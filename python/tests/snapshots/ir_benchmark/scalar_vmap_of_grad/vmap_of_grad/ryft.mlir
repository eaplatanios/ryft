module {
  func.func @main(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
    %3 = stablehlo.cosine %2 : tensor<2xf64>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2xf64>
    %5 = stablehlo.multiply %3, %4 : tensor<2xf64>
    %6 = stablehlo.multiply %2, %2 : tensor<2xf64>
    %7 = stablehlo.multiply %6, %2 : tensor<2xf64>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<2xf64>
    %9 = stablehlo.multiply %7, %8 : tensor<2xf64>
    %10 = stablehlo.add %5, %9 : tensor<2xf64>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<2xf64>
    %12 = stablehlo.multiply %2, %11 : tensor<2xf64>
    %13 = stablehlo.multiply %6, %12 : tensor<2xf64>
    %14 = stablehlo.add %10, %13 : tensor<2xf64>
    %15 = stablehlo.multiply %2, %12 : tensor<2xf64>
    %16 = stablehlo.multiply %2, %15 : tensor<2xf64>
    %17 = stablehlo.add %14, %16 : tensor<2xf64>
    %18 = stablehlo.multiply %2, %15 : tensor<2xf64>
    %19 = stablehlo.add %17, %18 : tensor<2xf64>
    %20 = stablehlo.slice %19 [0:1] : (tensor<2xf64>) -> tensor<1xf64>
    %21 = stablehlo.reshape %20 : (tensor<1xf64>) -> tensor<f64>
    %22 = stablehlo.slice %19 [1:2] : (tensor<2xf64>) -> tensor<1xf64>
    %23 = stablehlo.reshape %22 : (tensor<1xf64>) -> tensor<f64>
    %24 = stablehlo.add %21, %23 : tensor<f64>
    return %24 : tensor<f64>
  }
}
