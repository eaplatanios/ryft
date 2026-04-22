module {
  sdy.mesh @mesh = <["x"=4]>
  func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
      %1 = stablehlo.sine %arg1 : tensor<2xf32>
      sdy.return %1 : tensor<2xf32>
    } : (tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
