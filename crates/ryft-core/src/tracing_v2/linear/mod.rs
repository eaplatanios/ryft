use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;

use crate::operations::arithmetic::{AddOperation, SupportsAdd};
use crate::operations::constants::{One, OneLike, SupportsZeroLike, Zero, ZeroLike};
use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::{Parameterized, ParameterizedFamily, Placeholder};
use crate::tracing::engines::{Engine, Tracer, TracingContext, TracingEngine};
use crate::tracing::{Program, ProgramBuilder, Traceable, TracingError, Value};
use crate::tracing_v2::differentiation::JvpTracer;
use crate::tracing_v2::{
    Differentiable, DifferentiableEngine, DifferentiableOperation, DifferentiableOperationTracingEngine,
    DifferentiableTracingEngine,
};
use crate::types::{ArrayType, Typed};

/// Dense Jacobian and Hessian materialization helpers.
mod dense;
/// Traced-program linearization helpers.
mod replay;
/// Public reverse-mode APIs built from traced programs and staged pullbacks.
mod reverse;

pub use dense::{CoordinateValue, DenseJacobian, jacrev};
pub(crate) use dense::{Hessian, JacFwd};
pub(crate) use reverse::Grad;
pub use reverse::{
    ConcreteValueAndGrad, TracedValueAndGrad, ValueAndGradDispatch, grad_with_aux, linearize, value_and_grad,
    value_and_grad_with_aux, vjp,
};

#[cfg(test)]
mod tests {
    use crate::operations::scalars::LinearScalarOperation;
    use crate::operations::trigonometric::Sin;
    use crate::tracing::Program;
    use crate::tracing::engines::ScalarEngine;
    use crate::tracing_v2::linearize;
    use crate::types::DataType;
    use indoc::indoc;

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    #[test]
    fn test_linearize_returns_the_primal_output_and_pushforward() {
        let engine = ScalarEngine::<f64>::new();
        let (primal, pushforward) = linearize(&engine, |x| Ok(x.clone() * x.clone() + x.sin()), 2.0f64).unwrap();

        approx_eq(primal, 2.0f64.powi(2) + 2.0f64.sin());
        approx_eq(pushforward.interpret(1.5f64).unwrap(), (4.0 + 2.0f64.cos()) * 1.5);
        assert_eq!(
            pushforward.to_string(),
            indoc! {"
            lambda %0:f64 .
            let %1:f64 = scale [factor=2] %0
                %2:f64 = scale [factor=2] %0
                %3:f64 = add %1 %2
                %4:f64 = scale [factor=-0.4161468365471424] %0
                %5:f64 = add %3 %4
            in (%5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_transposed_linear_program_matches_the_reverse_mode_pullback() {
        let engine = ScalarEngine::<f64>::new();
        let (primal, pushforward) =
            linearize(&engine, |inputs| Ok(inputs.0.clone() * inputs.1 + inputs.0.sin()), (2.0f64, 3.0f64)).unwrap();
        let pullback = pushforward.transpose(&[primal]).unwrap();
        let cotangent = pullback.interpret(1.0f64).unwrap();

        approx_eq(primal, 2.0 * 3.0 + 2.0f64.sin());
        approx_eq(cotangent.0, 3.0 + 2.0f64.cos());
        approx_eq(cotangent.1, 2.0);
        assert_eq!(
            pullback.to_string(),
            indoc! {"
            lambda %0:f64 .
            let %1:f64 = scale [factor=-0.4161468365471424] %0
                %2:f64 = scale [factor=3] %0
                %3:f64 = add %1 %2
                %4:f64 = scale [factor=2] %0
            in (%3, %4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn linear_program_display_delegates_to_the_underlying_program() {
        let engine = ScalarEngine::<f64>::new();
        let (_, pushforward): (f64, Program<DataType, f64, LinearScalarOperation<f64>, f64, f64>) =
            linearize(&engine, |x| Ok(x.clone() * x.clone() + x.sin()), 2.0f64).unwrap();

        assert_eq!(
            pushforward.to_string(),
            indoc! {"
            lambda %0:f64 .
            let %1:f64 = scale [factor=2] %0
                %2:f64 = scale [factor=2] %0
                %3:f64 = add %1 %2
                %4:f64 = scale [factor=-0.4161468365471424] %0
                %5:f64 = add %3 %4
            in (%5)
            "}
            .trim_end(),
        );
    }
}
