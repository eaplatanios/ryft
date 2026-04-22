module {
  func.func @main(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.cosine %arg0 : tensor<f64>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %1 = stablehlo.multiply %0, %cst : tensor<f64>
    %2 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
    %3 = stablehlo.multiply %2, %arg0 : tensor<f64>
    %4 = stablehlo.multiply %3, %cst : tensor<f64>
    %5 = stablehlo.add %1, %4 : tensor<f64>
    %6 = stablehlo.multiply %arg0, %cst : tensor<f64>
    %7 = stablehlo.multiply %2, %6 : tensor<f64>
    %8 = stablehlo.add %5, %7 : tensor<f64>
    %9 = stablehlo.multiply %arg0, %6 : tensor<f64>
    %10 = stablehlo.multiply %arg0, %9 : tensor<f64>
    %11 = stablehlo.add %8, %10 : tensor<f64>
    %12 = stablehlo.multiply %arg0, %9 : tensor<f64>
    %13 = stablehlo.add %11, %12 : tensor<f64>
    return %13 : tensor<f64>
  }
}
