module {
  func.func @main(%arg0: tensor<1x1xf64>, %arg1: tensor<1x1xf64>, %arg2: tensor<1x1xf64>, %arg3: tensor<1x1xf64>) -> tensor<1x1xf64> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<1x1xf64>
    %1 = stablehlo.dot_general %0, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x1xf64>
    %2 = stablehlo.dot_general %1, %arg2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x1xf64>
    %3 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x1xf64>
    %4 = stablehlo.sine %3 : tensor<1x1xf64>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<1x1xf64>
    %6 = stablehlo.dot_general %5, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x1xf64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %7 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<1x1xf64>
    %8 = stablehlo.dot_general %arg0, %7, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x1xf64>
    %9 = stablehlo.add %6, %8 : tensor<1x1xf64>
    %10 = stablehlo.multiply %4, %9 : tensor<1x1xf64>
    %11 = stablehlo.negate %10 : tensor<1x1xf64>
    %12 = stablehlo.multiply %2, %11 : tensor<1x1xf64>
    %13 = stablehlo.cosine %3 : tensor<1x1xf64>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %14 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<1x1xf64>
    %15 = stablehlo.dot_general %14, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x1xf64>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %16 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f64>) -> tensor<1x1xf64>
    %17 = stablehlo.dot_general %0, %16, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x1xf64>
    %18 = stablehlo.add %15, %17 : tensor<1x1xf64>
    %19 = stablehlo.dot_general %18, %arg2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x1xf64>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %20 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f64>) -> tensor<1x1xf64>
    %21 = stablehlo.dot_general %1, %20, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x1xf64>
    %22 = stablehlo.add %19, %21 : tensor<1x1xf64>
    %23 = stablehlo.multiply %13, %22 : tensor<1x1xf64>
    %24 = stablehlo.add %12, %23 : tensor<1x1xf64>
    %25 = stablehlo.dot_general %24, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x1xf64>
    %26 = stablehlo.multiply %13, %2 : tensor<1x1xf64>
    %27 = stablehlo.dot_general %26, %7, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x1xf64>
    %28 = stablehlo.add %25, %27 : tensor<1x1xf64>
    return %28 : tensor<1x1xf64>
  }
}
