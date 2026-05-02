use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;

use crate::operations::arithmetic::{AddOperation, SupportsAdd};
use crate::operations::constants::{One, SupportsZero, Zero};
use crate::operations::constants::{OneLike, SupportsZeroLike, ZeroLike};
use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::{Parameterized, ParameterizedFamily, Placeholder};
use crate::tracing::engines::{Engine, Tracer, TracingContext, TracingEngine};
use crate::tracing::transposition::LinearOperation;
use crate::tracing::{Atom, AtomId, Instruction, Program, ProgramBuilder, Traceable, TracingError, Value};
use crate::tracing_v2::differentiation::JvpTracer;
use crate::tracing_v2::operations::SupportsRematerialize;
use crate::tracing_v2::operations::rematerialize::{FlatTracedRematerialize, RematerializeOperation};
use crate::tracing_v2::{
    Differentiable, DifferentiableEngine, DifferentiableOperation, DifferentiableOperationTracingEngine,
    DifferentiableTracingEngine,
};
use crate::types::{ArrayType, Typed};

/// Dense Jacobian and Hessian materialization helpers.
mod dense;
/// Rematerialization-aware compiled-gradient helpers.
mod rematerialization;
/// Traced-program linearization helpers.
mod replay;
/// Public reverse-mode APIs built from traced programs and staged pullbacks.
mod reverse;

pub use dense::{CoordinateValue, DenseJacobian, hessian, jacfwd, jacrev};
pub use rematerialization::{RematerializationPolicy, compile_grad, compile_grad_with_policy};
pub use reverse::{grad, grad_with_aux, linearize, value_and_grad, value_and_grad_with_aux, vjp};

#[cfg(test)]
mod tests {
    use crate::tracing::Program;
    use crate::tracing::engines::ScalarEngine;
    use crate::tracing_v2::{LinearScalarOperation, Sin, compile_grad, linearize};
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

    #[test]
    fn compile_grad_produces_reusable_gradient_program() {
        let engine = ScalarEngine::<f64>::new();
        let compiled = compile_grad(&engine, |x| x.clone() * x.clone() + x.sin(), 2.0f64).unwrap();

        // d/dx(x^2 + sin(x)) = 2x + cos(x)

        // Verify at the original primal point.
        let grad_at_2 = compiled.interpret(2.0f64).unwrap();
        approx_eq(grad_at_2, 2.0 * 2.0 + 2.0f64.cos());

        // Verify at a different primal point.
        let grad_at_half = compiled.interpret(0.5f64).unwrap();
        approx_eq(grad_at_half, 2.0 * 0.5 + 0.5f64.cos());

        let grad_at_pi = compiled.interpret(std::f64::consts::PI).unwrap();
        approx_eq(grad_at_pi, 2.0 * std::f64::consts::PI + std::f64::consts::PI.cos());

        // The program should contain cos (from sin's derivative), not baked constants.
        let ir = compiled.to_string();
        assert!(ir.contains("cos"), "compiled grad should compute cos symbolically, not bake constants");
    }

    #[test]
    fn compile_grad_bilinear_returns_both_partial_derivatives() {
        let engine = ScalarEngine::<f64>::new();
        let compiled =
            compile_grad(&engine, |inputs| inputs.0.clone() * inputs.1 + inputs.0.sin(), (2.0f64, 3.0f64)).unwrap();

        // df/dx = y + cos(x), df/dy = x
        let (grad_x, grad_y) = compiled.interpret((2.0f64, 3.0f64)).unwrap();
        approx_eq(grad_x, 3.0 + 2.0f64.cos());
        approx_eq(grad_y, 2.0);

        // At a different primal point:
        let (grad_x2, grad_y2) = compiled.interpret((1.0f64, 5.0f64)).unwrap();
        approx_eq(grad_x2, 5.0 + 1.0f64.cos());
        approx_eq(grad_y2, 1.0);
    }
}
