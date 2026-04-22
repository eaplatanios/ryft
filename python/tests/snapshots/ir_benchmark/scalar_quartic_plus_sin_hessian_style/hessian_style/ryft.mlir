module {
  func.func @main(%arg0: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %0 = stablehlo.sine %arg0 : tensor<f64>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %1 = stablehlo.multiply %0, %cst_0 : tensor<f64>
    %2 = stablehlo.negate %1 : tensor<f64>
    %3 = stablehlo.multiply %cst, %2 : tensor<f64>
    %4 = stablehlo.cosine %arg0 : tensor<f64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %5 = stablehlo.multiply %4, %cst_1 : tensor<f64>
    %6 = stablehlo.add %3, %5 : tensor<f64>
    %7 = stablehlo.multiply %arg0, %cst_0 : tensor<f64>
    %8 = stablehlo.multiply %arg0, %cst_0 : tensor<f64>
    %9 = stablehlo.add %7, %8 : tensor<f64>
    %10 = stablehlo.multiply %arg0, %9 : tensor<f64>
    %11 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
    %12 = stablehlo.multiply %11, %cst_0 : tensor<f64>
    %13 = stablehlo.add %10, %12 : tensor<f64>
    %14 = stablehlo.multiply %cst, %13 : tensor<f64>
    %15 = stablehlo.multiply %11, %arg0 : tensor<f64>
    %16 = stablehlo.multiply %15, %cst_1 : tensor<f64>
    %17 = stablehlo.add %14, %16 : tensor<f64>
    %18 = stablehlo.add %6, %17 : tensor<f64>
    %19 = stablehlo.multiply %arg0, %cst : tensor<f64>
    %20 = stablehlo.multiply %19, %9 : tensor<f64>
    %21 = stablehlo.multiply %cst, %cst_0 : tensor<f64>
    %22 = stablehlo.multiply %arg0, %cst_1 : tensor<f64>
    %23 = stablehlo.add %21, %22 : tensor<f64>
    %24 = stablehlo.multiply %11, %23 : tensor<f64>
    %25 = stablehlo.add %20, %24 : tensor<f64>
    %26 = stablehlo.add %18, %25 : tensor<f64>
    %27 = stablehlo.multiply %arg0, %19 : tensor<f64>
    %28 = stablehlo.multiply %27, %cst_0 : tensor<f64>
    %29 = stablehlo.multiply %19, %cst_0 : tensor<f64>
    %30 = stablehlo.multiply %arg0, %23 : tensor<f64>
    %31 = stablehlo.add %29, %30 : tensor<f64>
    %32 = stablehlo.multiply %arg0, %31 : tensor<f64>
    %33 = stablehlo.add %28, %32 : tensor<f64>
    %34 = stablehlo.add %26, %33 : tensor<f64>
    %35 = stablehlo.multiply %27, %cst_0 : tensor<f64>
    %36 = stablehlo.multiply %arg0, %31 : tensor<f64>
    %37 = stablehlo.add %35, %36 : tensor<f64>
    %38 = stablehlo.add %34, %37 : tensor<f64>
    return %38 : tensor<f64>
  }
}
