module {
  func.func @main(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %0 = stablehlo.multiply %cst, %arg0 : tensor<f64>
    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %1 = stablehlo.multiply %cst_0, %arg0 : tensor<f64>
    %2 = stablehlo.add %0, %1 : tensor<f64>
    %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %3 = stablehlo.multiply %cst_1, %2 : tensor<f64>
    %cst_2 = stablehlo.constant dense<4.000000e+00> : tensor<f64>
    %4 = stablehlo.multiply %cst_2, %arg0 : tensor<f64>
    %5 = stablehlo.add %3, %4 : tensor<f64>
    %cst_3 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %6 = stablehlo.multiply %cst_3, %5 : tensor<f64>
    %cst_4 = stablehlo.constant dense<8.000000e+00> : tensor<f64>
    %7 = stablehlo.multiply %cst_4, %arg0 : tensor<f64>
    %8 = stablehlo.add %6, %7 : tensor<f64>
    %cst_5 = stablehlo.constant dense<-0.41614683654714241> : tensor<f64>
    %9 = stablehlo.multiply %cst_5, %arg0 : tensor<f64>
    %10 = stablehlo.add %8, %9 : tensor<f64>
    return %10 : tensor<f64>
  }
}
