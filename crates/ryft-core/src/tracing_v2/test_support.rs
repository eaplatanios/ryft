use std::ops::{Add, Mul, Neg};

use indoc::indoc;

use crate::{
    parameters::Placeholder,
    tracing_v2::{engine::ArrayScalarEngine, *},
    types::ArrayType,
};

pub(crate) fn assert_reference_scalar_sine_jit_rendering() {
    let engine = ArrayScalarEngine::<f64>::new();
    let (_, compiled): (f64, Program<ArrayType, f64, f64, f64>) =
        interpret_and_trace(&engine, |x| Ok(x.sin()), 2.0f64).unwrap();

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

pub(crate) fn assert_reference_program_rendering() {
    let mut builder = ProgramBuilder::<PrimitiveOp<ArrayType, f64>, ArrayType, f64>::new();
    let x = builder.add_input(&1.0f64);
    let three = builder.add_constant(3.0f64);
    let sum = builder.add_equation(PrimitiveOp::Add, vec![x, three]).unwrap()[0];
    let program = builder.build::<f64, f64>(vec![sum], Placeholder, Placeholder);

    assert_eq!(
        program.to_string(),
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
    T: Clone + Sin + Cos + Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
{
    inputs.0.clone() * inputs.1 + inputs.0.sin()
}

fn quadratic_plus_sin<T>(x: T) -> T
where
    T: Clone + Sin + Cos + Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
{
    x.clone() * x.clone() + x.sin()
}

pub(crate) fn assert_bilinear_pushforward_rendering() {
    let engine = ArrayScalarEngine::<f64>::new();
    let (_, pushforward): (f64, LinearProgram<ArrayType, f64, (f64, f64), f64>) =
        jvp_program(&engine, |inputs| Ok(bilinear_sin(inputs)), (2.0f64, 3.0f64)).unwrap();

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

pub(crate) fn assert_bilinear_jit_rendering() {
    let engine = ArrayScalarEngine::<f64>::new();
    let (_, compiled): (f64, Program<ArrayType, f64, (f64, f64), f64>) =
        interpret_and_trace(&engine, |inputs| Ok(bilinear_sin(inputs)), (2.0f64, 3.0f64)).unwrap();

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
    let engine = ArrayScalarEngine::<f64>::new();
    let (_, pushforward): (f64, LinearProgram<ArrayType, f64, f64, f64>) =
        jvp_program(&engine, |x| Ok(quadratic_plus_sin(x)), 2.0f64).unwrap();

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
