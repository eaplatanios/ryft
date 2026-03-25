use std::ops::{Add, Mul, Neg};

use indoc::indoc;

use crate::{parameters::Placeholder, tracing_v2::*};

pub(crate) fn assert_reference_scalar_sine_jit_rendering() {
    let (_, compiled): (f64, CompiledFunction<f64, f64, f64>) = jit(|x: JitTracer<f64>| x.sin(), 2.0f64).unwrap();

    assert_eq!(
        compiled.to_string(),
        indoc! {"
            lambda %0:f64[] .
            let %1:f64[] = sin %0
            in (%1)
        "}
        .trim_end(),
    );
}

pub(crate) fn assert_reference_graph_rendering() {
    let mut builder = GraphBuilder::<std::sync::Arc<dyn Op<f64>>, f64>::new();
    let x = builder.add_input(&1.0f64);
    let three = builder.add_constant(3.0f64);
    let sum = builder.add_equation(std::sync::Arc::new(AddOp), vec![x, three]).unwrap()[0];
    let graph = builder.build::<f64, f64>(vec![sum], Placeholder, Placeholder);

    assert_eq!(
        graph.to_string(),
        indoc! {"
            lambda %0:f64[] .
            let %1:f64[] = const
                %2:f64[] = add %0 %1
            in (%2)
        "}
        .trim_end(),
    );
}

fn bilinear_sin<T>(inputs: (T, T)) -> T
where
    T: Clone + FloatExt + Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
{
    inputs.0.clone() * inputs.1 + inputs.0.sin()
}

fn quadratic_plus_sin<T>(x: T) -> T
where
    T: Clone + FloatExt + Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
{
    x.clone() * x.clone() + x.sin()
}

fn quartic_plus_sin<T>(x: T) -> T
where
    T: Clone + FloatExt + Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
{
    x.clone() * x.clone() * x.clone() * x.clone() + x.sin()
}

trait HigherOrderScalarValue:
    TraceValue
    + Clone
    + FloatExt
    + ZeroLike
    + OneLike
    + MatrixOps
    + Add<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
{
    fn first_derivative(self) -> Self;

    fn second_derivative(self) -> Self;

    fn third_derivative(self) -> Self;

    fn fourth_derivative(self) -> Self;
}

impl HigherOrderScalarValue for f64 {
    fn first_derivative(self) -> Self {
        grad(quartic_plus_sin, self).expect("first derivative should be computable")
    }

    fn second_derivative(self) -> Self {
        grad(first_derivative, self).expect("second derivative should be computable")
    }

    fn third_derivative(self) -> Self {
        grad(second_derivative, self).expect("third derivative should be computable")
    }

    fn fourth_derivative(self) -> Self {
        grad(third_derivative, self).expect("fourth derivative should be computable")
    }
}

impl<V> HigherOrderScalarValue for JitTracer<V>
where
    V: HigherOrderScalarValue,
{
    fn first_derivative(self) -> Self {
        grad(quartic_plus_sin, self).expect("first derivative should be computable")
    }

    fn second_derivative(self) -> Self {
        grad(first_derivative, self).expect("second derivative should be computable")
    }

    fn third_derivative(self) -> Self {
        grad(second_derivative, self).expect("third derivative should be computable")
    }

    fn fourth_derivative(self) -> Self {
        grad(third_derivative, self).expect("fourth derivative should be computable")
    }
}

fn first_derivative<V>(x: V) -> V
where
    V: HigherOrderScalarValue,
{
    V::first_derivative(x)
}

fn second_derivative<V>(x: V) -> V
where
    V: HigherOrderScalarValue,
{
    V::second_derivative(x)
}

fn third_derivative<V>(x: V) -> V
where
    V: HigherOrderScalarValue,
{
    V::third_derivative(x)
}

fn fourth_derivative<V>(x: V) -> V
where
    V: HigherOrderScalarValue,
{
    V::fourth_derivative(x)
}

fn hessian_style_second_derivative<V>(x: V) -> V
where
    V: HigherOrderScalarValue + TraceValue,
{
    let (_, second_derivative) =
        jvp(first_derivative, x.clone(), x.one_like()).expect("forward-over-reverse Hessian should succeed");
    second_derivative
}

pub(crate) fn assert_bilinear_pushforward_rendering() {
    let (_, pushforward): (f64, LinearProgram<f64, (f64, f64), f64>) =
        linearize(bilinear_sin, (2.0f64, 3.0f64)).unwrap();

    assert_eq!(
        pushforward.to_string(),
        indoc! {"
            lambda %0:f64[], %1:f64[] .
            let %2:f64[] = scale %0
                %3:f64[] = scale %1
                %4:f64[] = add %2 %3
                %5:f64[] = scale %0
                %6:f64[] = add %4 %5
            in (%6)
        "}
        .trim_end(),
    );
}

pub(crate) fn assert_bilinear_pullback_rendering() {
    let (_, pullback): (f64, LinearProgram<f64, f64, (f64, f64)>) = vjp(bilinear_sin, (2.0f64, 3.0f64)).unwrap();

    assert_eq!(
        pullback.to_string(),
        indoc! {"
            lambda %0:f64[] .
            let %1:f64[] = scale %0
                %2:f64[] = scale %0
                %3:f64[] = add %1 %2
                %4:f64[] = scale %0
            in (%3, %4)
        "}
        .trim_end(),
    );
}

pub(crate) fn assert_bilinear_jit_rendering() {
    let (_, compiled): (f64, CompiledFunction<f64, (f64, f64), f64>) = jit(bilinear_sin, (2.0f64, 3.0f64)).unwrap();

    assert_eq!(
        compiled.to_string(),
        indoc! {"
            lambda %0:f64[], %1:f64[] .
            let %2:f64[] = mul %0 %1
                %3:f64[] = sin %0
                %4:f64[] = add %2 %3
            in (%4)
        "}
        .trim_end(),
    );
}

pub(crate) fn assert_quadratic_pushforward_rendering() {
    let (_, pushforward): (f64, LinearProgram<f64, f64, f64>) = linearize(quadratic_plus_sin, 2.0f64).unwrap();

    assert_eq!(
        pushforward.to_string(),
        indoc! {"
            lambda %0:f64[] .
            let %1:f64[] = scale %0
                %2:f64[] = scale %0
                %3:f64[] = add %1 %2
                %4:f64[] = scale %0
                %5:f64[] = add %3 %4
            in (%5)
        "}
        .trim_end(),
    );
}
pub(crate) fn assert_hessian_style_second_derivative_jit_rendering() {
    let (_, compiled): (f64, CompiledFunction<f64, f64, f64>) = jit(hessian_style_second_derivative, 2.0f64).unwrap();

    assert_eq!(
        compiled.to_string(),
        indoc! {"
            lambda %0:f64[] .
            let %1:f64[] = const
                %2:f64[] = sin %0
                %3:f64[] = const
                %4:f64[] = mul %2 %3
                %5:f64[] = neg %4
                %6:f64[] = mul %1 %5
                %7:f64[] = cos %0
                %8:f64[] = const
                %9:f64[] = mul %7 %8
                %10:f64[] = add %6 %9
                %11:f64[] = mul %0 %3
                %12:f64[] = mul %0 %3
                %13:f64[] = add %11 %12
                %14:f64[] = mul %0 %13
                %15:f64[] = mul %0 %0
                %16:f64[] = mul %15 %3
                %17:f64[] = add %14 %16
                %18:f64[] = mul %1 %17
                %19:f64[] = mul %15 %0
                %20:f64[] = mul %19 %8
                %21:f64[] = add %18 %20
                %22:f64[] = add %10 %21
                %23:f64[] = mul %0 %1
                %24:f64[] = mul %23 %13
                %25:f64[] = mul %1 %3
                %26:f64[] = mul %0 %8
                %27:f64[] = add %25 %26
                %28:f64[] = mul %15 %27
                %29:f64[] = add %24 %28
                %30:f64[] = add %22 %29
                %31:f64[] = mul %0 %23
                %32:f64[] = mul %31 %3
                %33:f64[] = mul %23 %3
                %34:f64[] = mul %0 %27
                %35:f64[] = add %33 %34
                %36:f64[] = mul %0 %35
                %37:f64[] = add %32 %36
                %38:f64[] = add %30 %37
                %39:f64[] = mul %31 %3
                %40:f64[] = mul %0 %35
                %41:f64[] = add %39 %40
                %42:f64[] = add %38 %41
            in (%42)
        "}
        .trim_end(),
    );
}

pub(crate) fn assert_fourth_derivative_jit_rendering() {
    let (_, compiled): (f64, CompiledFunction<f64, f64, f64>) = jit(fourth_derivative, 2.0f64).unwrap();

    assert_eq!(
        compiled.to_string(),
        indoc! {"
            lambda %0:f64[] .
            let %1:f64[] = const
                %2:f64[] = const
                %3:f64[] = mul %1 %2
                %4:f64[] = mul %1 %2
                %5:f64[] = add %3 %4
                %6:f64[] = const
                %7:f64[] = const
                %8:f64[] = mul %6 %7
                %9:f64[] = mul %5 %8
                %10:f64[] = mul %6 %2
                %11:f64[] = mul %1 %7
                %12:f64[] = mul %1 %7
                %13:f64[] = add %11 %12
                %14:f64[] = mul %10 %13
                %15:f64[] = add %9 %14
                %16:f64[] = mul %2 %8
                %17:f64[] = mul %10 %7
                %18:f64[] = add %16 %17
                %19:f64[] = mul %1 %18
                %20:f64[] = add %15 %19
                %21:f64[] = mul %1 %18
                %22:f64[] = add %20 %21
                %23:f64[] = mul %6 %1
                %24:f64[] = mul %23 %2
                %25:f64[] = mul %1 %10
                %26:f64[] = add %24 %25
                %27:f64[] = mul %26 %7
                %28:f64[] = add %22 %27
                %29:f64[] = mul %26 %7
                %30:f64[] = add %28 %29
                %31:f64[] = mul %1 %8
                %32:f64[] = mul %23 %7
                %33:f64[] = add %31 %32
                %34:f64[] = mul %2 %33
                %35:f64[] = add %30 %34
                %36:f64[] = mul %2 %33
                %37:f64[] = add %35 %36
                %38:f64[] = mul %2 %7
                %39:f64[] = mul %2 %7
                %40:f64[] = add %38 %39
                %41:f64[] = mul %23 %40
                %42:f64[] = add %37 %41
                %43:f64[] = mul %5 %7
                %44:f64[] = mul %2 %13
                %45:f64[] = add %43 %44
                %46:f64[] = mul %1 %40
                %47:f64[] = add %45 %46
                %48:f64[] = mul %6 %47
                %49:f64[] = add %42 %48
                %50:f64[] = sin %0
                %51:f64[] = mul %6 %1
                %52:f64[] = neg %51
                %53:f64[] = mul %52 %2
                %54:f64[] = mul %53 %7
                %55:f64[] = neg %54
                %56:f64[] = mul %50 %55
                %57:f64[] = add %49 %56
            in (%57)
        "}
        .trim_end(),
    );
}
pub(crate) fn assert_inline_fourth_derivative_jit_rendering() {
    let (_, compiled): (f64, CompiledFunction<f64, f64, f64>) = jit(
        |x| {
            grad(
                |x| {
                    grad(|x| grad(quartic_plus_sin, x).expect("innermost grad should succeed"), x)
                        .expect("third derivative should succeed")
                },
                x,
            )
            .expect("second derivative should succeed")
        },
        2.0f64,
    )
    .unwrap();

    assert_eq!(
        compiled.to_string(),
        indoc! {"
            lambda %0:f64[] .
            let %1:f64[] = cos %0
                %2:f64[] = const
                %3:f64[] = const
                %4:f64[] = mul %2 %3
                %5:f64[] = neg %4
                %6:f64[] = const
                %7:f64[] = mul %5 %6
                %8:f64[] = mul %1 %7
                %9:f64[] = mul %0 %2
                %10:f64[] = mul %9 %3
                %11:f64[] = mul %2 %3
                %12:f64[] = mul %0 %11
                %13:f64[] = add %10 %12
                %14:f64[] = mul %13 %6
                %15:f64[] = add %8 %14
                %16:f64[] = mul %0 %6
                %17:f64[] = mul %0 %6
                %18:f64[] = add %16 %17
                %19:f64[] = mul %11 %18
                %20:f64[] = add %15 %19
                %21:f64[] = mul %13 %6
                %22:f64[] = add %20 %21
                %23:f64[] = mul %11 %6
                %24:f64[] = mul %2 %6
                %25:f64[] = mul %3 %24
                %26:f64[] = add %23 %25
                %27:f64[] = mul %0 %26
                %28:f64[] = add %22 %27
                %29:f64[] = mul %0 %26
                %30:f64[] = add %28 %29
                %31:f64[] = mul %0 %3
                %32:f64[] = mul %0 %3
                %33:f64[] = add %31 %32
                %34:f64[] = mul %33 %24
                %35:f64[] = add %30 %34
                %36:f64[] = mul %0 %24
                %37:f64[] = mul %9 %6
                %38:f64[] = add %36 %37
                %39:f64[] = mul %3 %38
                %40:f64[] = add %35 %39
                %41:f64[] = mul %3 %38
                %42:f64[] = add %40 %41
                %43:f64[] = mul %3 %18
                %44:f64[] = mul %33 %6
                %45:f64[] = add %43 %44
                %46:f64[] = mul %3 %6
                %47:f64[] = mul %3 %6
                %48:f64[] = add %46 %47
                %49:f64[] = mul %0 %48
                %50:f64[] = add %45 %49
                %51:f64[] = mul %2 %50
                %52:f64[] = add %42 %51
                %53:f64[] = mul %9 %48
                %54:f64[] = add %52 %53
            in (%54)
        "}
        .trim_end(),
    );
}

#[cfg(any(feature = "ndarray", test))]
use ndarray::{Array2, arr2};

#[cfg(any(feature = "ndarray", test))]
fn bilinear_matmul<M>(inputs: (M, M)) -> M
where
    M: MatrixOps,
{
    inputs.0.matmul(inputs.1)
}

#[cfg(any(feature = "ndarray", test))]
fn three_matmul_sine<M>(inputs: (M, M, M, M)) -> M
where
    M: MatrixOps + FloatExt,
{
    let (x, a, b, c) = inputs;
    x.matmul(a).sin().matmul(b).matmul(c)
}

#[cfg(any(feature = "ndarray", test))]
trait HigherOrderMatrixValue: MatrixValue + FloatExt + ZeroLike + OneLike + Sized {
    fn full_matrix_gradient(inputs: (Self, Self, Self, Self)) -> (Self, Self, Self, Self);
}

#[cfg(any(feature = "ndarray", test))]
impl HigherOrderMatrixValue for Array2<f64> {
    fn full_matrix_gradient(inputs: (Self, Self, Self, Self)) -> (Self, Self, Self, Self) {
        grad(three_matmul_sine, inputs).expect("matrix gradient should succeed")
    }
}

#[cfg(any(feature = "ndarray", test))]
impl<V> HigherOrderMatrixValue for JitTracer<V>
where
    V: HigherOrderMatrixValue,
{
    fn full_matrix_gradient(inputs: (Self, Self, Self, Self)) -> (Self, Self, Self, Self) {
        grad(three_matmul_sine, inputs).expect("matrix gradient should succeed")
    }
}

#[cfg(any(feature = "ndarray", test))]
fn first_matrix_gradient<V>(inputs: (V, V, V, V)) -> V
where
    V: HigherOrderMatrixValue,
{
    let (x_bar, _, _, _) = V::full_matrix_gradient(inputs);
    x_bar
}

#[cfg(any(feature = "ndarray", test))]
pub(crate) fn assert_matrix_jit_rendering() {
    let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let b = arr2(&[[2.0, 0.0], [1.0, 2.0]]);
    let (_, compiled): (Array2<f64>, CompiledFunction<Array2<f64>, (Array2<f64>, Array2<f64>), Array2<f64>>) =
        jit(bilinear_matmul, (a, b)).unwrap();

    assert_eq!(
        compiled.to_string(),
        indoc! {"
            lambda %0:f64[2, 2], %1:f64[2, 2] .
            let %2:f64[2, 2] = matmul %0 %1
            in (%2)
        "}
        .trim_end(),
    );
}

#[cfg(any(feature = "ndarray", test))]
pub(crate) fn assert_matrix_pushforward_rendering() {
    let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let b = arr2(&[[2.0, 0.0], [1.0, 2.0]]);
    let (_, pushforward): (Array2<f64>, LinearProgram<Array2<f64>, (Array2<f64>, Array2<f64>), Array2<f64>>) =
        linearize(bilinear_matmul, (a, b)).unwrap();

    assert_eq!(
        pushforward.to_string(),
        indoc! {"
            lambda %0:f64[2, 2], %1:f64[2, 2] .
            let %2:f64[2, 2] = right_matmul %0
                %3:f64[2, 2] = left_matmul %1
                %4:f64[2, 2] = add %2 %3
            in (%4)
        "}
        .trim_end(),
    );
}

#[cfg(any(feature = "ndarray", test))]
pub(crate) fn assert_matrix_pullback_rendering() {
    let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let b = arr2(&[[2.0, 0.0], [1.0, 2.0]]);
    let (_, pullback): (Array2<f64>, LinearProgram<Array2<f64>, Array2<f64>, (Array2<f64>, Array2<f64>)>) =
        vjp(bilinear_matmul, (a, b)).unwrap();

    assert_eq!(
        pullback.to_string(),
        indoc! {"
            lambda %0:f64[2, 2] .
            let %1:f64[2, 2] = right_matmul %0
                %2:f64[2, 2] = left_matmul %0
            in (%1, %2)
        "}
        .trim_end(),
    );
}
#[cfg(any(feature = "ndarray", test))]
pub(crate) fn assert_matrix_hessian_style_jit_rendering() {
    let x = arr2(&[[0.7f64]]);
    let a = arr2(&[[2.0f64]]);
    let b = arr2(&[[-1.5f64]]);
    let c = arr2(&[[4.0f64]]);
    let (_, compiled): (
        Array2<f64>,
        CompiledFunction<Array2<f64>, (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>), Array2<f64>>,
    ) = jit(
        |inputs| {
            let seeds = (inputs.0.one_like(), inputs.1.zero_like(), inputs.2.zero_like(), inputs.3.zero_like());
            jvp(first_matrix_gradient, inputs, seeds).expect("matrix Hessian should succeed").1
        },
        (x, a, b, c),
    )
    .unwrap();

    assert_eq!(
        compiled.to_string(),
        indoc! {"
            lambda %0:f64[1, 1], %1:f64[1, 1], %2:f64[1, 1], %3:f64[1, 1] .
            let %4:f64[1, 1] = const
                %5:f64[1, 1] = matrix_transpose %3
                %6:f64[1, 1] = matmul %4 %5
                %7:f64[1, 1] = matrix_transpose %2
                %8:f64[1, 1] = matmul %6 %7
                %9:f64[1, 1] = matmul %0 %1
                %10:f64[1, 1] = sin %9
                %11:f64[1, 1] = const
                %12:f64[1, 1] = matmul %11 %1
                %13:f64[1, 1] = const
                %14:f64[1, 1] = matmul %0 %13
                %15:f64[1, 1] = add %12 %14
                %16:f64[1, 1] = mul %10 %15
                %17:f64[1, 1] = neg %16
                %18:f64[1, 1] = mul %8 %17
                %19:f64[1, 1] = cos %9
                %20:f64[1, 1] = const
                %21:f64[1, 1] = matmul %20 %5
                %22:f64[1, 1] = const
                %23:f64[1, 1] = matrix_transpose %22
                %24:f64[1, 1] = matmul %4 %23
                %25:f64[1, 1] = add %21 %24
                %26:f64[1, 1] = matmul %25 %7
                %27:f64[1, 1] = const
                %28:f64[1, 1] = matrix_transpose %27
                %29:f64[1, 1] = matmul %6 %28
                %30:f64[1, 1] = add %26 %29
                %31:f64[1, 1] = mul %19 %30
                %32:f64[1, 1] = add %18 %31
                %33:f64[1, 1] = matrix_transpose %1
                %34:f64[1, 1] = matmul %32 %33
                %35:f64[1, 1] = mul %19 %8
                %36:f64[1, 1] = matrix_transpose %13
                %37:f64[1, 1] = matmul %35 %36
                %38:f64[1, 1] = add %34 %37
            in (%38)
        "}
        .trim_end(),
    );
}
