//! Linearization, transposition, and higher-order differentiation utilities.
//!
//! This module is the "middle layer" between raw tracing and user-facing reverse-mode APIs. Its
//! job is to take an ordinary staged primal program and turn it into the linear objects that power
//! the rest of autodiff:
//!
//! - [`LinearProgram`] for reusable pushforwards and pullbacks,
//! - reverse-mode transforms such as [`vjp`], [`grad`], and [`value_and_grad`],
//! - dense Jacobian and Hessian materialization helpers, and
//! - rematerialization-aware compiled gradients.
//!
//! In the larger architecture, this is where the system stops thinking in terms of "run the
//! original function again" and starts reasoning about the linear maps induced by that function.

use std::{
    cell::RefCell,
    fmt::{Debug, Display},
    marker::PhantomData,
    rc::Rc,
};

use ryft_macros::Parameter;

use crate::{
    parameters::{Parameter, Parameterized, ParameterizedFamily, Placeholder},
    tracing_v2::{
        Atom, AtomId, Instruction, LinearPrimitiveOp, OneLike, Program, ProgramBuilder, Traceable, TracingError, Value,
        ZeroLike,
        batch::{Batch, stack, unstack},
        engine::Engine,
        forward::{JvpTracer, TangentSpace},
        jit::{Tracer, interpret_and_trace},
        operations::{
            CoreLinearProgramOp, CoreLinearReplayOp, DifferentiableOp, InterpretableOp, LinearAddOperation,
            LinearNegOperation, LinearScaleOperation, Op, RematerializeTracingOperation,
            rematerialize::{FlatTracedRematerialize, RematerializeOp},
        },
    },
    types::{ArrayType, Type, Typed},
};

mod dense;
mod program;
mod rematerialization;
mod replay;
mod reverse;
mod term;

pub use dense::{CoordinateValue, DenseJacobian, hessian, jacfwd, jacrev};
pub use program::LinearProgram;
pub use program::transpose_linear_program_with_output_examples;
pub use rematerialization::{RematerializationPolicy, compile_grad, compile_grad_with_policy};
pub use reverse::{grad, jvp_program, value_and_grad, vjp};
pub use term::{LinearTerm, Linearized};

pub(crate) use program::linearize_program;
pub(crate) use replay::{linearize_traced_program, replay_program_linearized_jit};
pub(crate) use reverse::jvp_traced;

type LinearizedTracedValue<'engine, E> =
    Linearized<Tracer<'engine, E>, LinearPrimitiveOp<ArrayType, Tracer<'engine, E>>>;

type TracedLinearProgram<'engine, E> = LinearProgram<
    ArrayType,
    Tracer<'engine, E>,
    Vec<Tracer<'engine, E>>,
    Vec<Tracer<'engine, E>>,
    LinearPrimitiveOp<ArrayType, Tracer<'engine, E>>,
>;

#[inline]
fn flat_leaf_parameter_structure(count: usize) -> Vec<Placeholder> {
    vec![Placeholder; count]
}

/// Traces one type-directed body and normalizes the captured program to flat leaf vectors.
///
/// Many linearization helpers want a uniform "flat vector of leaves" view even when the caller's
/// original function uses tuples, structs, or other parameterized shapes. This helper is the bridge
/// between those worlds: it traces the structured function once, then retags the captured program
/// so downstream reverse-mode code can operate on a canonical `Vec<V>` representation.
pub(crate) fn trace_flat_program_from_input_types<'engine, Input, Output, V, O, L, E, F>(
    function: F,
    traced_inputs: &[Tracer<'engine, E>],
    input_types: Input,
) -> Result<(Output, Program<ArrayType, V, O, Vec<V>, Vec<V>>), TracingError>
where
    V: Traceable<ArrayType> + Parameterized<V, ParameterStructure = Placeholder>,
    Input: Parameterized<ArrayType, ParameterStructure: Clone>,
    Output: Parameterized<ArrayType, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<V> + ParameterizedFamily<Tracer<'engine, E>>,
    Output::Family: ParameterizedFamily<V> + ParameterizedFamily<Tracer<'engine, E>>,
    O: Clone + Op<ArrayType> + 'static,
    L: Clone + 'static,
    E: Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized + 'static,
    F: FnOnce(Input::To<Tracer<'engine, E>>) -> Result<Output::To<Tracer<'engine, E>>, TracingError>,
{
    let exemplar_engine = traced_inputs.first().ok_or(TracingError::EmptyParameterizedValue)?.engine();
    let (output_types, traced_program): (Output, Program<ArrayType, V, O, Input::To<V>, Output::To<V>>) =
        crate::tracing_v2::jit::trace(exemplar_engine, function, input_types)?;
    let output_leaf_count = output_types.parameter_structure().parameter_count();
    let traced_program = Program {
        atoms: traced_program.atoms.clone(),
        input_ids: traced_program.input_ids.clone(),
        output_ids: traced_program.output_ids.clone(),
        instructions: traced_program.instructions.clone(),
        input_structure: flat_leaf_parameter_structure(traced_inputs.len()),
        output_structure: flat_leaf_parameter_structure(output_leaf_count),
        marker: std::marker::PhantomData,
    }
    .simplify()?;
    Ok((output_types, traced_program))
}

/// Linearizes one flat scalar traced program and stages its pullback with a unit cotangent seed.
///
/// This is the internal core of traced reverse-mode for scalar-output functions. Given a staged
/// primal body and symbolic primals from an enclosing trace, it builds the pushforward, transposes
/// it into a pullback, seeds that pullback with a symbolic one, and returns both the traced scalar
/// output and the traced gradient leaves.
fn reverse_mode_scalar_traced_program<'engine, V, O, L, E>(
    traced_program: &Program<ArrayType, V, O, Vec<V>, Vec<V>>,
    traced_primals: Vec<Tracer<'engine, E>>,
) -> Result<(Tracer<'engine, E>, Vec<Tracer<'engine, E>>), TracingError>
where
    V: Traceable<ArrayType> + ZeroLike + OneLike,
    O: Clone + Op<ArrayType> + 'static,
    L: Clone + 'static,
    E: Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized + 'static,
    O: InterpretableOp<ArrayType, Linearized<Tracer<'engine, E>, LinearPrimitiveOp<ArrayType, Tracer<'engine, E>>>>,
    LinearPrimitiveOp<ArrayType, Tracer<'engine, E>>: CoreLinearProgramOp<Tracer<'engine, E>>,
{
    let (outputs, pushforward) = linearize_traced_program::<V, O, L, E>(traced_program, traced_primals)?;
    if outputs.len() != 1 {
        return Err(TracingError::InvalidOutputCount { expected: 1, got: outputs.len() });
    }
    let traced_output = outputs[0].clone();
    let pullback =
        transpose_linear_program_with_output_examples::<Tracer<'engine, E>, _, _, _>(&pushforward, outputs.as_slice())?;
    let traced_gradient = pullback.call(vec![traced_output.one_like()])?;
    Ok((traced_output, traced_gradient))
}

#[cfg(test)]
mod tests {
    use std::ops::{Add, Mul, Neg};
    use std::{
        fmt::{Debug, Display},
        sync::Arc,
    };

    use indoc::indoc;

    use crate::{
        parameters::Placeholder,
        tracing_v2::{
            CustomPrimitive, DifferentiableOp, InterpretableOp, LinearOperation, LinearPrimitiveOp, Op, PrimitiveOp,
            ProgramBuilder, Sin, engine::ArrayScalarEngine, test_support,
        },
        types::{ArrayType, DataType},
    };

    use super::*;

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    fn quadratic_plus_sin<T>(x: T) -> T
    where
        T: Clone + Sin + Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
    {
        x.clone() * x.clone() + x.sin()
    }

    fn bilinear_sin<T>(inputs: (T, T)) -> T
    where
        T: Clone + Sin + Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
    {
        inputs.0.clone() * inputs.1 + inputs.0.sin()
    }

    #[derive(Clone, Default)]
    struct PanicReplayOp;

    impl Debug for PanicReplayOp {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "PanicReplay")
        }
    }

    impl Display for PanicReplayOp {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "panic_replay")
        }
    }

    impl Op for PanicReplayOp {
        fn name(&self) -> &'static str {
            "panic_replay"
        }

        fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TracingError> {
            if inputs.len() != 1 {
                return Err(TracingError::InvalidInputCount { expected: 1, got: inputs.len() });
            }
            Ok(vec![inputs[0].clone()])
        }
    }

    impl InterpretableOp<ArrayType, f64> for PanicReplayOp {
        fn interpret(&self, _inputs: &[f64]) -> Result<Vec<f64>, TracingError> {
            panic!("panic_replay interpret should not run during this transform")
        }
    }

    impl LinearOperation<ArrayType, f64> for PanicReplayOp {
        fn transpose(
            &self,
            output_cotangents: &[LinearTerm<ArrayType, f64>],
        ) -> Result<Vec<Option<LinearTerm<ArrayType, f64>>>, TracingError> {
            if output_cotangents.len() != 1 {
                return Err(TracingError::InvalidInputCount { expected: 1, got: output_cotangents.len() });
            }
            Ok(vec![Some(output_cotangents[0].clone())])
        }
    }

    impl
        DifferentiableOp<
            ArrayType,
            f64,
            LinearTerm<ArrayType, f64>,
            PrimitiveOp<ArrayType, f64>,
            LinearPrimitiveOp<ArrayType, f64>,
        > for PanicReplayOp
    {
        fn jvp(
            &self,
            _engine: &dyn Engine<
                Type = ArrayType,
                Value = f64,
                TracingOperation = PrimitiveOp<ArrayType, f64>,
                LinearOperation = LinearPrimitiveOp<ArrayType, f64>,
            >,
            inputs: &[JvpTracer<f64, LinearTerm<ArrayType, f64>>],
        ) -> Result<Vec<JvpTracer<f64, LinearTerm<ArrayType, f64>>>, TracingError> {
            if inputs.len() != 1 {
                return Err(TracingError::InvalidInputCount { expected: 1, got: inputs.len() });
            }
            Ok(vec![inputs[0].clone()])
        }
    }

    #[test]
    fn test_jvp_program_returns_the_primal_output_and_pushforward() {
        let engine = ArrayScalarEngine::<f64>::new();
        let (primal, pushforward) = jvp_program(&engine, |x| Ok(quadratic_plus_sin(x)), 2.0f64).unwrap();

        approx_eq(primal, 2.0f64.powi(2) + 2.0f64.sin());
        approx_eq(pushforward.call(1.5f64).unwrap(), (4.0 + 2.0f64.cos()) * 1.5);
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

    #[test]
    fn test_transposed_linear_program_matches_the_reverse_mode_pullback() {
        let engine = ArrayScalarEngine::<f64>::new();
        let (primal, pushforward) = jvp_program(&engine, |inputs| Ok(bilinear_sin(inputs)), (2.0f64, 3.0f64)).unwrap();
        let pullback = pushforward.transpose().unwrap();
        let cotangent = pullback.call(1.0f64).unwrap();

        approx_eq(primal, 2.0 * 3.0 + 2.0f64.sin());
        approx_eq(cotangent.0, 3.0 + 2.0f64.cos());
        approx_eq(cotangent.1, 2.0);
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

    #[test]
    fn linearize_program_does_not_replay_the_forward_program_to_recover_representatives() {
        let primitive = CustomPrimitive::<ArrayType, f64>::new(PanicReplayOp).with_jvp_rule(PanicReplayOp);
        let mut builder = ProgramBuilder::<PrimitiveOp<ArrayType, f64>, ArrayType, f64>::new();
        let input = builder.add_input(&3.0f64);
        let output = builder.add_instruction_prevalidated(
            PrimitiveOp::Custom(Arc::new(primitive)),
            vec![input],
            vec![ArrayType::scalar(DataType::F64)],
        );
        let program = builder.build::<f64, f64>(output, Placeholder, Placeholder);

        let engine = ArrayScalarEngine::<f64>::new();
        let pushforward = linearize_program(&engine, &program, vec![3.0f64]).unwrap();
        approx_eq(pushforward.call(2.5f64).unwrap(), 2.5);
    }

    #[test]
    fn transpose_linear_program_does_not_replay_the_forward_linear_program_to_recover_representatives() {
        let primitive = LinearPrimitiveOp::custom(
            CustomPrimitive::<ArrayType, f64>::new(PanicReplayOp).with_transpose_rule(PanicReplayOp),
        )
        .unwrap();
        let mut builder = ProgramBuilder::<LinearPrimitiveOp<ArrayType, f64>, ArrayType, f64>::new();
        let input = builder.add_input(&0.0f64);
        let output =
            builder.add_instruction_prevalidated(primitive, vec![input], vec![ArrayType::scalar(DataType::F64)]);
        let program = builder.build::<f64, f64>(output, Placeholder, Placeholder);
        let pushforward = LinearProgram::from_program(program, 0.0f64);

        let pullback = super::program::transpose_linear_program(&pushforward).unwrap();
        approx_eq(pullback.call(4.0f64).unwrap(), 4.0);
    }

    #[test]
    fn linear_program_display_delegates_to_the_underlying_program() {
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
        assert_eq!(pushforward.to_string(), pushforward.program().to_string());
        test_support::assert_quadratic_pushforward_rendering();
    }

    #[test]
    fn compile_grad_produces_reusable_gradient_program() {
        let engine = ArrayScalarEngine::<f64>::new();
        let compiled = compile_grad(&engine, quadratic_plus_sin, 2.0f64).unwrap();

        // d/dx(x^2 + sin(x)) = 2x + cos(x)

        // Verify at the original primal point.
        let grad_at_2 = compiled.interpret(2.0f64).unwrap();
        approx_eq(grad_at_2, 2.0 * 2.0 + 2.0f64.cos());

        // Verify at a DIFFERENT primal point Ã¢â‚¬â€ this is the key test.
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
        let engine = ArrayScalarEngine::<f64>::new();
        let compiled = compile_grad(&engine, bilinear_sin, (2.0f64, 3.0f64)).unwrap();

        // df/dx = y + cos(x), df/dy = x
        let (grad_x, grad_y) = compiled.interpret((2.0f64, 3.0f64)).unwrap();
        approx_eq(grad_x, 3.0 + 2.0f64.cos());
        approx_eq(grad_y, 2.0);

        // At a different primal point:
        let (grad_x2, grad_y2) = compiled.interpret((1.0f64, 5.0f64)).unwrap();
        approx_eq(grad_x2, 5.0 + 1.0f64.cos());
        approx_eq(grad_y2, 1.0);
    }

    // -----------------------------------------------------------------------
    // RematerializationPolicy tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compile_grad_save_all_matches_compile_grad() {
        // SaveAll should produce the same gradient as the plain compile_grad.
        let engine = ArrayScalarEngine::<f64>::new();
        let compiled_plain = compile_grad(&engine, quadratic_plus_sin, 2.0f64).unwrap();
        let compiled_save_all =
            compile_grad_with_policy(&engine, quadratic_plus_sin, 2.0f64, RematerializationPolicy::SaveAll).unwrap();

        let grad_plain = compiled_plain.interpret(2.0f64).unwrap();
        let grad_save_all = compiled_save_all.interpret(2.0f64).unwrap();
        approx_eq(grad_plain, grad_save_all);

        // Also verify at a different primal point.
        let grad_plain_2 = compiled_plain.interpret(0.5f64).unwrap();
        let grad_save_all_2 = compiled_save_all.interpret(0.5f64).unwrap();
        approx_eq(grad_plain_2, grad_save_all_2);
    }

    #[test]
    fn test_compile_grad_recompute_all_gives_correct_gradient() {
        // RecomputeAll should give d/dx(x^2 + sin(x)) = 2x + cos(x).
        let engine = ArrayScalarEngine::<f64>::new();
        let compiled =
            compile_grad_with_policy(&engine, quadratic_plus_sin, 2.0f64, RematerializationPolicy::RecomputeAll)
                .unwrap();

        approx_eq(compiled.interpret(2.0f64).unwrap(), 2.0 * 2.0 + 2.0f64.cos());
        approx_eq(compiled.interpret(0.5f64).unwrap(), 2.0 * 0.5 + 0.5f64.cos());
        approx_eq(
            compiled.interpret(std::f64::consts::PI).unwrap(),
            2.0 * std::f64::consts::PI + std::f64::consts::PI.cos(),
        );
    }

    #[test]
    fn test_compile_grad_recompute_all_matches_compile_grad() {
        // RecomputeAll should give the same numerical gradient as compile_grad.
        let engine = ArrayScalarEngine::<f64>::new();
        let compiled_plain = compile_grad(&engine, quadratic_plus_sin, 2.0f64).unwrap();
        let compiled_recompute =
            compile_grad_with_policy(&engine, quadratic_plus_sin, 2.0f64, RematerializationPolicy::RecomputeAll)
                .unwrap();

        for x in [0.0, 0.5, 1.0, 2.0, 3.0, std::f64::consts::PI] {
            let grad_plain = compiled_plain.interpret(x).unwrap();
            let grad_recompute = compiled_recompute.interpret(x).unwrap();
            approx_eq(grad_plain, grad_recompute);
        }
    }

    #[test]
    fn test_compile_grad_checkpoint_gives_correct_gradient() {
        // Checkpoint with segment_size=2 should give the correct gradient for a function with
        // ~4 instructions: x*x, sin(x), x*x + sin(x).
        let engine = ArrayScalarEngine::<f64>::new();
        let compiled = compile_grad_with_policy(
            &engine,
            quadratic_plus_sin,
            2.0f64,
            RematerializationPolicy::Checkpoint { segment_size: 2 },
        )
        .unwrap();

        approx_eq(compiled.interpret(2.0f64).unwrap(), 2.0 * 2.0 + 2.0f64.cos());
        approx_eq(compiled.interpret(0.5f64).unwrap(), 2.0 * 0.5 + 0.5f64.cos());
    }

    #[test]
    fn test_compile_grad_checkpoint_is_reusable_at_different_primals() {
        // The compiled gradient with Checkpoint can be called at multiple primal points.
        let engine = ArrayScalarEngine::<f64>::new();
        let compiled = compile_grad_with_policy(
            &engine,
            quadratic_plus_sin,
            1.0f64,
            RematerializationPolicy::Checkpoint { segment_size: 2 },
        )
        .unwrap();

        for x in [0.0, 0.5, 1.0, 2.0, 3.0, std::f64::consts::PI] {
            let expected = 2.0 * x + x.cos();
            approx_eq(compiled.interpret(x).unwrap(), expected);
        }
    }

    #[test]
    fn test_compile_grad_checkpoint_matches_compile_grad() {
        // Checkpoint should give the same numerical gradient as compile_grad.
        let engine = ArrayScalarEngine::<f64>::new();
        let compiled_plain = compile_grad(&engine, quadratic_plus_sin, 2.0f64).unwrap();
        let compiled_checkpoint = compile_grad_with_policy(
            &engine,
            quadratic_plus_sin,
            2.0f64,
            RematerializationPolicy::Checkpoint { segment_size: 2 },
        )
        .unwrap();

        for x in [0.0, 0.5, 1.0, 2.0, 3.0, std::f64::consts::PI] {
            let grad_plain = compiled_plain.interpret(x).unwrap();
            let grad_checkpoint = compiled_checkpoint.interpret(x).unwrap();
            approx_eq(grad_plain, grad_checkpoint);
        }
    }

    #[test]
    fn test_compile_grad_checkpoint_segment_size_one_matches_save_all() {
        // Checkpoint with segment_size=1 should degenerate to SaveAll.
        let engine = ArrayScalarEngine::<f64>::new();
        let compiled_save_all =
            compile_grad_with_policy(&engine, quadratic_plus_sin, 2.0f64, RematerializationPolicy::SaveAll).unwrap();
        let compiled_checkpoint = compile_grad_with_policy(
            &engine,
            quadratic_plus_sin,
            2.0f64,
            RematerializationPolicy::Checkpoint { segment_size: 1 },
        )
        .unwrap();

        for x in [0.0, 1.0, 2.0] {
            approx_eq(compiled_save_all.interpret(x).unwrap(), compiled_checkpoint.interpret(x).unwrap());
        }
    }

    #[test]
    fn test_compile_grad_checkpoint_large_segment_wraps_whole_program() {
        // Checkpoint with a segment_size larger than the number of instructions should wrap
        // the entire program in a single RematerializeOp, equivalent to RecomputeAll.
        let engine = ArrayScalarEngine::<f64>::new();
        let compiled = compile_grad_with_policy(
            &engine,
            quadratic_plus_sin,
            2.0f64,
            RematerializationPolicy::Checkpoint { segment_size: 100 },
        )
        .unwrap();

        for x in [0.0, 1.0, 2.0, std::f64::consts::PI] {
            approx_eq(compiled.interpret(x).unwrap(), 2.0 * x + x.cos());
        }
    }
}
