module {
  sdy.mesh @mesh = <["x"=4]>
  func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
      %1 = stablehlo.cosine %arg1 : tensor<2xf32>
      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2xf32>
      %3 = stablehlo.multiply %1, %2 : tensor<2xf32>
      %4 = stablehlo.broadcast_in_dim %3, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
      %5 = stablehlo.broadcast_in_dim %3, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
      %6 = stablehlo.concatenate %4, %5, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<2xf32>
      %8 = stablehlo.broadcast_in_dim %7, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
      %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<1x2xf32>) -> tensor<2x2xf32>
      %10 = stablehlo.add %6, %9 : tensor<2x2xf32>
      %11 = stablehlo.slice %10 [0:1, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
      %12 = stablehlo.reshape %11 : (tensor<1x2xf32>) -> tensor<2xf32>
      %13 = stablehlo.slice %10 [1:2, 0:2] : (tensor<2x2xf32>) -> tensor<1x2xf32>
      %14 = stablehlo.reshape %13 : (tensor<1x2xf32>) -> tensor<2xf32>
      %15 = stablehlo.add %12, %14 : tensor<2xf32>
      sdy.return %15 : tensor<2xf32>
    } : (tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
