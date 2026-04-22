module @jit_matrix_hessian_style_second_derivative attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1xf64>, %arg1: tensor<1x1xf64>, %arg2: tensor<1x1xf64>, %arg3: tensor<1x1xf64>) -> (tensor<1x1xf64> {jax.result_info = "result"}) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<1x1xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<1x1xf64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %2 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<1x1xf64>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %3 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<1x1xf64>
    %4 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x1xf64>
    %5 = stablehlo.dot_general %0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x1xf64>
    %6 = stablehlo.dot_general %arg0, %1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x1xf64>
    %7 = stablehlo.add %5, %6 : tensor<1x1xf64>
    %8 = stablehlo.cosine %4 : tensor<1x1xf64>
    %9 = stablehlo.sine %4 : tensor<1x1xf64>
    %10 = stablehlo.multiply %7, %9 : tensor<1x1xf64>
    %11 = stablehlo.negate %10 : tensor<1x1xf64>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %12 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f64>) -> tensor<1x1xf64>
    %13 = stablehlo.dot_general %12, %arg3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x1xf64>
    %14 = stablehlo.dot_general %12, %3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x1xf64>
    %15 = stablehlo.dot_general %13, %arg2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x1xf64>
    %16 = stablehlo.dot_general %14, %arg2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x1xf64>
    %17 = stablehlo.dot_general %13, %2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x1xf64>
    %18 = stablehlo.add %16, %17 : tensor<1x1xf64>
    %19 = stablehlo.multiply %15, %8 : tensor<1x1xf64>
    %20 = stablehlo.multiply %18, %8 : tensor<1x1xf64>
    %21 = stablehlo.multiply %15, %11 : tensor<1x1xf64>
    %22 = stablehlo.add %20, %21 : tensor<1x1xf64>
    %23 = stablehlo.dot_general %22, %arg1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x1xf64>
    %24 = stablehlo.dot_general %19, %1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x1xf64>
    %25 = stablehlo.add %23, %24 : tensor<1x1xf64>
    return %25 : tensor<1x1xf64>
  }
}
