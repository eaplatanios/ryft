module @jit__unnamed_function attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x2xf64>) -> (tensor<2x2xf64> {jax.result_info = "result[0]"}, tensor<2x2xf64> {jax.result_info = "result[1]"}) {
    %cst = stablehlo.constant dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>
    %cst_0 = stablehlo.constant dense<[[5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]]> : tensor<2x2xf64>
    %0 = stablehlo.dot_general %arg0, %cst, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
    %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x2xf64>) -> tensor<2x2xf64>
    %2 = stablehlo.dot_general %arg0, %cst_0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
    return %2, %1 : tensor<2x2xf64>, tensor<2x2xf64>
  }
}
