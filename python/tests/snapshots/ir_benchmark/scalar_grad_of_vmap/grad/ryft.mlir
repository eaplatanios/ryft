module {
  func.func @main(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.cosine %arg0 : tensor<f64>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %1 = stablehlo.multiply %0, %cst : tensor<f64>
    %2 = stablehlo.multiply %arg0, %cst : tensor<f64>
    %3 = stablehlo.add %1, %2 : tensor<f64>
    %4 = stablehlo.multiply %arg0, %cst : tensor<f64>
    %5 = stablehlo.add %3, %4 : tensor<f64>
    %6 = stablehlo.cosine %arg0 : tensor<f64>
    %7 = stablehlo.multiply %6, %cst : tensor<f64>
    %8 = stablehlo.add %5, %7 : tensor<f64>
    %9 = stablehlo.multiply %arg0, %cst : tensor<f64>
    %10 = stablehlo.add %8, %9 : tensor<f64>
    %11 = stablehlo.multiply %arg0, %cst : tensor<f64>
    %12 = stablehlo.add %10, %11 : tensor<f64>
    return %12 : tensor<f64>
  }
}
