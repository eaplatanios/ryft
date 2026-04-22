module {
  func.func @main(%arg0: tensor<2x2xf64>) -> (tensor<2x2xf64>, tensor<2x2xf64>) {
    %cst = stablehlo.constant dense<[[5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]]> : tensor<2x2xf64>
    %0 = stablehlo.dot_general %arg0, %cst, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
    %cst_0 = stablehlo.constant dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>
    %1 = stablehlo.dot_general %arg0, %cst_0, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<2x2xf64>) -> tensor<2x2xf64>
    return %0, %2 : tensor<2x2xf64>, tensor<2x2xf64>
  }
}
