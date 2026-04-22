module {
  func.func @main(%arg0: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
    %1 = stablehlo.multiply %0, %arg0 : tensor<f64>
    %2 = stablehlo.multiply %1, %arg0 : tensor<f64>
    %3 = stablehlo.sine %arg0 : tensor<f64>
    %4 = stablehlo.add %2, %3 : tensor<f64>
    %5 = stablehlo.cosine %arg0 : tensor<f64>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %6 = stablehlo.multiply %5, %cst : tensor<f64>
    %7 = stablehlo.multiply %1, %cst : tensor<f64>
    %8 = stablehlo.add %6, %7 : tensor<f64>
    %9 = stablehlo.multiply %arg0, %cst : tensor<f64>
    %10 = stablehlo.multiply %0, %9 : tensor<f64>
    %11 = stablehlo.add %8, %10 : tensor<f64>
    %12 = stablehlo.multiply %arg0, %9 : tensor<f64>
    %13 = stablehlo.multiply %arg0, %12 : tensor<f64>
    %14 = stablehlo.add %11, %13 : tensor<f64>
    %15 = stablehlo.multiply %arg0, %12 : tensor<f64>
    %16 = stablehlo.add %14, %15 : tensor<f64>
    return %4, %16 : tensor<f64>, tensor<f64>
  }
}
