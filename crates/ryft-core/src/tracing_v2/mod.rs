//! Prototype tracing design for `ryft-core`.
//!
//! The key idea in this version is that staged computation is represented using a shared graph over
//! an open set of operation types:
//!
//! - `Parameterized<P>` lifts and lowers structured inputs and outputs.
//! - `Graph<O, V, In, Out>` is the common staging form.
//! - Each equation stores an operation object, not a tag enum.
//! - Primitive ops carry their own `eval`, `jvp`, `batch`, and transpose rules.
//! - Transform-specific context values thread both top-level runtime state and local staging state.

use thiserror::Error;

use crate::parameters::ParameterError;

mod batch;
mod context;
mod forward;
mod graph;
mod jit;
mod linear;
mod matmul;
mod ops;
#[cfg(test)]
pub(crate) mod test_support;
mod value;

pub use batch::{Batch, stack, unstack, vmap};
pub use context::{BatchingContext, JitContext, JvpContext};
pub use forward::{Dual, JvpTracer, TangentSpace, jvp};
pub use graph::{Atom, AtomId, AtomSource, Equation, Graph, GraphBuilder};
pub use jit::{CompiledFunction, JitTracer, jit};
pub use linear::{
    CoordinateValue, DenseJacobian, LinearProgram, LinearTerm, Linearized, grad, hessian, jacfwd, jacrev, jvp_program,
    linearize, value_and_grad, vjp,
};
pub use matmul::{MatMulOp, MatrixAbstract, MatrixOps, MatrixTangentSpace, MatrixTransposeOp, MatrixValue};
pub use ops::{
    AddOp, BatchOp, CosOp, JvpOp, LinearOp, LinearOpRef, MulOp, NegOp, Op, ScaleOp, SinOp, StagedOp, StagedOpRef,
};
pub use value::{FloatExt, OneLike, ScalarAbstract, TraceLeaf, TraceValue, ZeroLike};

/// Error type shared by the prototype tracing transforms.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum TraceError {
    /// Structured inputs or outputs did not have the same `Parameterized` shape.
    #[error("mismatched parameter structures")]
    MismatchedParameterStructure,

    /// A batching transform encountered zero lanes and therefore could not infer a batch size.
    #[error("encountered an empty batch")]
    EmptyBatch,

    /// A transform needed a seed value but the parameterized value contained no leaves.
    #[error("encountered an empty parameterized value while a seed value was required")]
    EmptyParameterizedValue,

    /// Different batched leaves disagreed on the number of lanes they carried.
    #[error("mismatched batch sizes across batched leaves")]
    MismatchedBatchSize,

    /// A primitive or staged graph received the wrong number of inputs.
    #[error("invalid number of inputs; expected {expected} but got {got}")]
    InvalidInputCount { expected: usize, got: usize },

    /// A primitive or staged graph produced the wrong number of outputs.
    #[error("invalid number of outputs; expected {expected} but got {got}")]
    InvalidOutputCount { expected: usize, got: usize },

    /// A staged graph referenced an atom that was never defined.
    #[error("unbound atom ID: {id}")]
    UnboundAtomId { id: usize },

    /// Abstract evaluation detected incompatible operand metadata for a primitive application.
    #[error("incompatible abstract values while tracing operation '{op}'")]
    IncompatibleAbstractValues { op: &'static str },

    /// An internal tracing invariant was violated while constructing or replaying a program.
    #[error("{0}")]
    InternalInvariantViolation(&'static str),

    /// Wrapper around parameter-lifting failures from the `Parameterized` infrastructure.
    #[error(transparent)]
    Parameter(#[from] ParameterError),
}

#[cfg(test)]
mod tests {
    use std::ops::{Add, Mul, Neg};

    use crate::{
        parameters::{Parameterized, ParameterizedFamily},
        tracing_v2::test_support,
    };

    use super::*;

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    fn bilinear_sin<Context, T>(_: &mut Context, inputs: (T, T)) -> T
    where
        T: Clone + FloatExt + Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
    {
        inputs.0.clone() * inputs.1 + inputs.0.sin()
    }

    fn quadratic_plus_sin<Context, T>(_: &mut Context, x: T) -> T
    where
        T: Clone + FloatExt + Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
    {
        x.clone() * x.clone() + x.sin()
    }

    fn quartic_plus_sin<Context, T>(_: &mut Context, x: T) -> T
    where
        T: Clone + FloatExt + Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
    {
        x.clone() * x.clone() * x.clone() * x.clone() + x.sin()
    }

    fn first_derivative<Context, V>(context: &mut Context, x: V) -> V
    where
        V: TraceValue
            + FloatExt
            + ZeroLike
            + OneLike
            + Parameterized<V, To<Linearized<V>> = Linearized<V>, ParameterStructure: Clone + PartialEq>,
        V::Family: ParameterizedFamily<Linearized<V>>,
    {
        grad(context, quartic_plus_sin, x).expect("first derivative should be computable")
    }

    fn second_derivative<Context, V>(context: &mut Context, x: V) -> V
    where
        V: TraceValue
            + FloatExt
            + ZeroLike
            + OneLike
            + Parameterized<V, To<Linearized<V>> = Linearized<V>, ParameterStructure: Clone + PartialEq>,
        V::Family: ParameterizedFamily<Linearized<V>>,
    {
        grad(context, first_derivative, x).expect("second derivative should be computable")
    }

    fn third_derivative<Context, V>(context: &mut Context, x: V) -> V
    where
        V: TraceValue
            + FloatExt
            + ZeroLike
            + OneLike
            + Parameterized<V, To<Linearized<V>> = Linearized<V>, ParameterStructure: Clone + PartialEq>,
        V::Family: ParameterizedFamily<Linearized<V>>,
    {
        grad(context, second_derivative, x).expect("third derivative should be computable")
    }

    fn fourth_derivative<Context, V>(context: &mut Context, x: V) -> V
    where
        V: TraceValue
            + FloatExt
            + ZeroLike
            + OneLike
            + Parameterized<V, To<Linearized<V>> = Linearized<V>, ParameterStructure: Clone + PartialEq>,
        V::Family: ParameterizedFamily<Linearized<V>>,
    {
        grad(context, third_derivative, x).expect("fourth derivative should be computable")
    }

    fn hessian_style_second_derivative<Context, V>(context: &mut Context, x: V) -> V
    where
        V: TraceValue
            + FloatExt
            + ZeroLike
            + OneLike
            + Parameterized<V, To<Linearized<V>> = Linearized<V>, ParameterStructure: Clone + PartialEq>,
        V::Family: ParameterizedFamily<Linearized<V>>,
    {
        let (_, second_derivative) = jvp(context, first_derivative, x.clone(), x.one_like())
            .expect("forward-over-reverse Hessian should succeed");
        second_derivative
    }

    #[test]
    fn forward_mode_uses_parameterized_structure() {
        let mut context = ();
        let (primal, tangent) = jvp(&mut context, bilinear_sin, (2.0f64, 3.0f64), (1.0f64, -1.0f64)).unwrap();
        approx_eq(primal, 2.0 * 3.0 + 2.0f64.sin());
        approx_eq(tangent, 3.0 - 2.0 + 2.0f64.cos());
        test_support::assert_bilinear_pushforward_rendering();
    }

    #[test]
    fn transposition_drives_reverse_mode() {
        let mut context = ();
        let (output, pullback) = vjp(&mut context, bilinear_sin, (2.0f64, 3.0f64)).unwrap();
        approx_eq(output, 2.0 * 3.0 + 2.0f64.sin());

        let input_cotangent = pullback.call(1.0f64).unwrap();
        approx_eq(input_cotangent.0, 3.0 + 2.0f64.cos());
        approx_eq(input_cotangent.1, 2.0);

        let gradient = grad(&mut context, quadratic_plus_sin, 2.0f64).unwrap();
        approx_eq(gradient, 4.0 + 2.0f64.cos());
        test_support::assert_bilinear_pullback_rendering();
    }

    #[test]
    fn value_and_grad_returns_both_outputs() {
        let mut context = ();
        let (value, gradient) = value_and_grad(&mut context, quadratic_plus_sin, 2.0f64).unwrap();

        approx_eq(value, 2.0f64.powi(2) + 2.0f64.sin());
        approx_eq(gradient, 4.0 + 2.0f64.cos());
        test_support::assert_quadratic_pushforward_rendering();
    }

    #[test]
    fn jacfwd_materializes_a_dense_jacobian() {
        let mut context = ();
        let jacobian = jacfwd::<(), _, (f64, f64), f64, f64>(&mut context, bilinear_sin, (2.0f64, 3.0f64)).unwrap();

        assert_eq!(jacobian.rows(), 1);
        assert_eq!(jacobian.cols(), 2);
        approx_eq(*jacobian.get(0, 0).unwrap(), 3.0 + 2.0f64.cos());
        approx_eq(*jacobian.get(0, 1).unwrap(), 2.0);
        test_support::assert_bilinear_pushforward_rendering();
    }

    #[test]
    fn jacrev_materializes_the_same_dense_jacobian() {
        let mut context = ();
        let jacobian = jacrev::<(), _, (f64, f64), f64, f64>(&mut context, bilinear_sin, (2.0f64, 3.0f64)).unwrap();

        assert_eq!(jacobian.rows(), 1);
        assert_eq!(jacobian.cols(), 2);
        approx_eq(*jacobian.get(0, 0).unwrap(), 3.0 + 2.0f64.cos());
        approx_eq(*jacobian.get(0, 1).unwrap(), 2.0);
        test_support::assert_bilinear_pullback_rendering();
    }

    #[test]
    fn hessian_materializes_a_dense_second_derivative_from_a_gradient_function() {
        let mut context = ();
        let dense_hessian = hessian(&mut context, first_derivative, 2.0f64).unwrap();

        assert_eq!(dense_hessian.rows(), 1);
        assert_eq!(dense_hessian.cols(), 1);
        approx_eq(*dense_hessian.get(0, 0).unwrap(), 12.0 * 2.0f64.powi(2) - 2.0f64.sin());
        test_support::assert_hessian_style_second_derivative_jit_rendering();
    }

    #[test]
    fn jit_captures_and_replays_a_program() {
        let mut context = ();
        let (output, compiled) = jit(&mut context, bilinear_sin, (2.0f64, 3.0f64)).unwrap();
        approx_eq(output, 2.0 * 3.0 + 2.0f64.sin());
        let replayed = compiled.call(&mut context, (5.0f64, -4.0f64)).unwrap();
        approx_eq(replayed, 5.0 * -4.0 + 5.0f64.sin());
        test_support::assert_bilinear_jit_rendering();
    }

    #[test]
    fn vectorization_stacks_and_unstacks_parameterized_inputs() {
        let mut context = ();
        let outputs =
            vmap(&mut context, bilinear_sin, vec![(1.0f64, 2.0f64), (3.0f64, 4.0f64), (5.0f64, 6.0f64)]).unwrap();

        let expected = vec![1.0 * 2.0 + 1.0f64.sin(), 3.0 * 4.0 + 3.0f64.sin(), 5.0 * 6.0 + 5.0f64.sin()];
        assert_eq!(outputs.len(), expected.len());
        for (left, right) in outputs.into_iter().zip(expected) {
            approx_eq(left, right);
        }
        test_support::assert_reference_scalar_sine_jit_rendering();
    }

    #[test]
    fn forward_over_reverse_computes_a_hessian_style_second_derivative() {
        let mut context = ();
        let (first_derivative_value, second_derivative_value) =
            jvp(&mut context, first_derivative, 2.0f64, 1.0f64).unwrap();

        approx_eq(first_derivative_value, 4.0 * 2.0f64.powi(3) + 2.0f64.cos());
        approx_eq(second_derivative_value, 12.0 * 2.0f64.powi(2) - 2.0f64.sin());
        test_support::assert_hessian_style_second_derivative_jit_rendering();
    }

    #[test]
    fn higher_order_reverse_mode_computes_a_fourth_derivative() {
        let mut context = ();
        let fourth_derivative_value = fourth_derivative(&mut context, 2.0f64);

        approx_eq(fourth_derivative_value, 24.0 + 2.0f64.sin());
        test_support::assert_fourth_derivative_jit_rendering();
    }

    #[test]
    fn inline_nested_grad_calls_compute_a_fourth_derivative() {
        let mut context = ();
        let fourth_derivative_value = grad(
            &mut context,
            |context, x| {
                grad(
                    context,
                    |context, x| {
                        grad(
                            context,
                            |context, x| grad(context, quartic_plus_sin, x).expect("innermost grad should succeed"),
                            x,
                        )
                        .expect("third derivative should succeed")
                    },
                    x,
                )
                .expect("second derivative should succeed")
            },
            2.0f64,
        )
        .expect("fourth derivative should succeed");

        approx_eq(fourth_derivative_value, 24.0 + 2.0f64.sin());
        test_support::assert_inline_fourth_derivative_jit_rendering();
    }

    #[test]
    fn jit_can_trace_a_hessian_style_second_derivative() {
        let mut context = ();
        let (second_derivative_value, compiled) = jit(&mut context, hessian_style_second_derivative, 2.0f64).unwrap();

        approx_eq(second_derivative_value, 12.0 * 2.0f64.powi(2) - 2.0f64.sin());
        let replayed = compiled.call(&mut context, 1.5f64).unwrap();
        approx_eq(replayed, 12.0 * 1.5f64.powi(2) - 1.5f64.sin());
        test_support::assert_hessian_style_second_derivative_jit_rendering();
    }

    #[test]
    fn jit_can_trace_a_fourth_derivative() {
        let mut context = ();
        let (fourth_derivative_value, compiled) = jit(&mut context, fourth_derivative, 2.0f64).unwrap();

        approx_eq(fourth_derivative_value, 24.0 + 2.0f64.sin());
        let replayed = compiled.call(&mut context, 0.5f64).unwrap();
        approx_eq(replayed, 24.0 + 0.5f64.sin());
        test_support::assert_fourth_derivative_jit_rendering();
    }
}
