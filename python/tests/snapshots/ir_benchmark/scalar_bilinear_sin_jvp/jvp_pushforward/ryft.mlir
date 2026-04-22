module {
  func.func @main(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<3.000000e+00> : tensor<f64>
    %0 = stablehlo.multiply %cst, %arg0 : tensor<f64>
    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %1 = stablehlo.multiply %cst_0, %arg1 : tensor<f64>
    %2 = stablehlo.add %0, %1 : tensor<f64>
    %cst_1 = stablehlo.constant dense<-0.41614683654714241> : tensor<f64>
    %3 = stablehlo.multiply %cst_1, %arg0 : tensor<f64>
    %4 = stablehlo.add %2, %3 : tensor<f64>
    return %4 : tensor<f64>
  }
}
