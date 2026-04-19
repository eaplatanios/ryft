//! Linearization, transposition, and higher-order differentiation utilities.
//!
//! This module turns forward-mode traces into staged linear programs, transposes those programs for reverse-mode
//! differentiation, and materializes dense Jacobians/Hessians for coordinate-based leaf types.

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
        OneLike, TraceError, Traceable, Value, ZeroLike,
        batch::{Batch, stack, unstack},
        engine::Engine,
        forward::{JvpTracer, TangentSpace},
        graph::{Atom, AtomId, Equation, Graph, GraphBuilder},
        jit::{CompiledFunction, JitTracer, jit, jit_for_operation, trace_program},
        operations::{
            CoreLinearProgramOp, CoreLinearReplayOp, DifferentiableOp, InterpretableOp, LinearAddOperation,
            LinearNegOperation, LinearScaleOperation, Op, RematerializeTracingOperation,
            rematerialize::{FlatTracedRematerialize, RematerializeOp},
        },
        program::{LinearProgramBuilder, LinearProgramOpRef, Program, ProgramBuilder},
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
pub(crate) use replay::{linearize_traced_program, replay_program_graph_linearized_jit};
pub(crate) use reverse::jvp_traced;

type LinearizedTracedValue<V, O, L> =
    Linearized<JitTracer<ArrayType, V, O, L>, LinearProgramOpRef<JitTracer<ArrayType, V, O, L>>>;

type TracedLinearProgram<V, O, L> = LinearProgram<
    ArrayType,
    JitTracer<ArrayType, V, O, L>,
    Vec<JitTracer<ArrayType, V, O, L>>,
    Vec<JitTracer<ArrayType, V, O, L>>,
    LinearProgramOpRef<JitTracer<ArrayType, V, O, L>>,
>;

#[inline]
fn flat_leaf_parameter_structure(count: usize) -> Vec<Placeholder> {
    vec![Placeholder; count]
}

/// Traces one type-directed body and normalizes the captured program to flat leaf vectors.
///
/// The caller supplies explicit staged input types plus the closure that should run on the
/// corresponding traced family. The captured program still uses the caller-selected staged op
/// carrier `O`, but its inputs and outputs are rewritten to `Vec<V>` so later linearization and
/// transposition code can share one flat reverse-mode path regardless of the original structure.
pub(crate) fn trace_flat_program_from_input_types<Input, Output, V, O, L, F>(
    function: F,
    traced_inputs: &[JitTracer<ArrayType, V, O, L>],
    input_types: Input,
) -> Result<(Output, Program<ArrayType, V, Vec<V>, Vec<V>, O>), TraceError>
where
    V: Traceable<ArrayType> + Parameterized<V, ParameterStructure = Placeholder>,
    Input: Parameterized<ArrayType, ParameterStructure: Clone>,
    Output: Parameterized<ArrayType, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<V> + ParameterizedFamily<JitTracer<ArrayType, V, O, L>>,
    Output::Family: ParameterizedFamily<V> + ParameterizedFamily<JitTracer<ArrayType, V, O, L>>,
    O: Clone + Op<ArrayType> + 'static,
    L: Clone + 'static,
    F: FnOnce(
        Input::To<JitTracer<ArrayType, V, O, L>>,
    ) -> Result<Output::To<JitTracer<ArrayType, V, O, L>>, TraceError>,
{
    let exemplar_engine = traced_inputs.first().ok_or(TraceError::EmptyParameterizedValue)?.engine();
    let (output_types, traced_program): (Output, Program<ArrayType, V, Input::To<V>, Output::To<V>, O>) =
        crate::tracing_v2::jit::trace_program_from_types_for_operation::<_, Input, Output, ArrayType, V, O, L>(
            exemplar_engine,
            function,
            input_types,
        )?;
    let output_leaf_count = output_types.parameter_structure().parameter_count();
    let traced_program = Program::from_graph(traced_program.graph().clone_with_structures::<Vec<V>, Vec<V>>(
        flat_leaf_parameter_structure(traced_inputs.len()),
        flat_leaf_parameter_structure(output_leaf_count),
    ))
    .simplify()?;
    Ok((output_types, traced_program))
}

/// Linearizes one flat scalar traced program and stages its pullback with a unit cotangent seed.
fn reverse_mode_scalar_traced_program<V, O, L>(
    traced_program: &Program<ArrayType, V, Vec<V>, Vec<V>, O>,
    traced_primals: Vec<JitTracer<ArrayType, V, O, L>>,
) -> Result<(JitTracer<ArrayType, V, O, L>, Vec<JitTracer<ArrayType, V, O, L>>), TraceError>
where
    V: Traceable<ArrayType> + ZeroLike + OneLike,
    O: Clone + Op<ArrayType> + 'static,
    L: Clone + 'static,
    O: InterpretableOp<
            ArrayType,
            Linearized<JitTracer<ArrayType, V, O, L>, LinearProgramOpRef<JitTracer<ArrayType, V, O, L>>>,
        >,
    LinearProgramOpRef<JitTracer<ArrayType, V, O, L>>: CoreLinearProgramOp<JitTracer<ArrayType, V, O, L>>,
{
    let (outputs, pushforward) = linearize_traced_program::<V, O, L>(traced_program, traced_primals)?;
    if outputs.len() != 1 {
        return Err(TraceError::InvalidOutputCount { expected: 1, got: outputs.len() });
    }
    let traced_output = outputs[0].clone();
    let pullback = transpose_linear_program_with_output_examples::<JitTracer<ArrayType, V, O, L>, _, _, _>(
        &pushforward,
        outputs.as_slice(),
    )?;
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
            CustomPrimitive, DifferentiableOp, GraphBuilder, InterpretableOp, LinearOperation, LinearPrimitiveOp, Op,
            PrimitiveOp, ProgramOpRef, Sin, engine::ArrayScalarEngine, test_support,
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

        fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
            if inputs.len() != 1 {
                return Err(TraceError::InvalidInputCount { expected: 1, got: inputs.len() });
            }
            Ok(vec![inputs[0].clone()])
        }
    }

    impl InterpretableOp<ArrayType, f64> for PanicReplayOp {
        fn interpret(&self, _inputs: &[f64]) -> Result<Vec<f64>, TraceError> {
            panic!("panic_replay interpret should not run during this transform")
        }
    }

    impl LinearOperation<ArrayType, f64> for PanicReplayOp {
        fn transpose(
            &self,
            output_cotangents: &[LinearTerm<ArrayType, f64>],
        ) -> Result<Vec<Option<LinearTerm<ArrayType, f64>>>, TraceError> {
            if output_cotangents.len() != 1 {
                return Err(TraceError::InvalidInputCount { expected: 1, got: output_cotangents.len() });
            }
            Ok(vec![Some(output_cotangents[0].clone())])
        }
    }

    impl DifferentiableOp<ArrayType, f64, LinearTerm<ArrayType, f64>, ProgramOpRef<f64>, LinearProgramOpRef<f64>>
        for PanicReplayOp
    {
        fn jvp(
            &self,
            _engine: &dyn Engine<
                Type = ArrayType,
                Value = f64,
                TracingOperation = ProgramOpRef<f64>,
                LinearOperation = LinearProgramOpRef<f64>,
            >,
            inputs: &[JvpTracer<f64, LinearTerm<ArrayType, f64>>],
        ) -> Result<Vec<JvpTracer<f64, LinearTerm<ArrayType, f64>>>, TraceError> {
            if inputs.len() != 1 {
                return Err(TraceError::InvalidInputCount { expected: 1, got: inputs.len() });
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
    fn linearize_program_does_not_replay_the_forward_graph_to_recover_representatives() {
        let primitive = CustomPrimitive::<ArrayType, f64>::new(PanicReplayOp).with_jvp_rule(PanicReplayOp);
        let mut builder = GraphBuilder::<ProgramOpRef<f64>, ArrayType, f64>::new();
        let input = builder.add_input(&3.0f64);
        let output = builder.add_equation_prevalidated(
            PrimitiveOp::Custom(Arc::new(primitive)),
            vec![input],
            vec![ArrayType::scalar(DataType::F64)],
        );
        let program = Program::from_graph(builder.build::<f64, f64>(output, Placeholder, Placeholder));

        let engine = ArrayScalarEngine::<f64>::new();
        let pushforward = linearize_program(&engine, &program, vec![3.0f64]).unwrap();
        approx_eq(pushforward.call(2.5f64).unwrap(), 2.5);
    }

    #[test]
    fn transpose_linear_program_does_not_replay_the_forward_linear_graph_to_recover_representatives() {
        let primitive = LinearPrimitiveOp::custom(
            CustomPrimitive::<ArrayType, f64>::new(PanicReplayOp).with_transpose_rule(PanicReplayOp),
        )
        .unwrap();
        let mut builder = GraphBuilder::<LinearProgramOpRef<f64>, ArrayType, f64>::new();
        let input = builder.add_input(&0.0f64);
        let output = builder.add_equation_prevalidated(primitive, vec![input], vec![ArrayType::scalar(DataType::F64)]);
        let program = Program::from_graph(builder.build::<f64, f64>(output, Placeholder, Placeholder));
        let pushforward = LinearProgram::from_program(program, 0.0f64);

        let pullback = super::program::transpose_linear_program(&pushforward).unwrap();
        approx_eq(pullback.call(4.0f64).unwrap(), 4.0);
    }

    #[test]
    fn linear_program_display_delegates_to_the_underlying_graph() {
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
        assert_eq!(pushforward.to_string(), pushforward.program().graph().to_string());
        test_support::assert_quadratic_pushforward_rendering();
    }

    #[test]
    fn compile_grad_produces_reusable_gradient_program() {
        let engine = ArrayScalarEngine::<f64>::new();
        let compiled = compile_grad(&engine, quadratic_plus_sin, 2.0f64).unwrap();

        // d/dx(x^2 + sin(x)) = 2x + cos(x)

        // Verify at the original primal point.
        let grad_at_2 = compiled.call(2.0f64).unwrap();
        approx_eq(grad_at_2, 2.0 * 2.0 + 2.0f64.cos());

        // Verify at a DIFFERENT primal point — this is the key test.
        let grad_at_half = compiled.call(0.5f64).unwrap();
        approx_eq(grad_at_half, 2.0 * 0.5 + 0.5f64.cos());

        let grad_at_pi = compiled.call(std::f64::consts::PI).unwrap();
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
        let (grad_x, grad_y) = compiled.call((2.0f64, 3.0f64)).unwrap();
        approx_eq(grad_x, 3.0 + 2.0f64.cos());
        approx_eq(grad_y, 2.0);

        // At a different primal point:
        let (grad_x2, grad_y2) = compiled.call((1.0f64, 5.0f64)).unwrap();
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

        let grad_plain = compiled_plain.call(2.0f64).unwrap();
        let grad_save_all = compiled_save_all.call(2.0f64).unwrap();
        approx_eq(grad_plain, grad_save_all);

        // Also verify at a different primal point.
        let grad_plain_2 = compiled_plain.call(0.5f64).unwrap();
        let grad_save_all_2 = compiled_save_all.call(0.5f64).unwrap();
        approx_eq(grad_plain_2, grad_save_all_2);
    }

    #[test]
    fn test_compile_grad_recompute_all_gives_correct_gradient() {
        // RecomputeAll should give d/dx(x^2 + sin(x)) = 2x + cos(x).
        let engine = ArrayScalarEngine::<f64>::new();
        let compiled =
            compile_grad_with_policy(&engine, quadratic_plus_sin, 2.0f64, RematerializationPolicy::RecomputeAll)
                .unwrap();

        approx_eq(compiled.call(2.0f64).unwrap(), 2.0 * 2.0 + 2.0f64.cos());
        approx_eq(compiled.call(0.5f64).unwrap(), 2.0 * 0.5 + 0.5f64.cos());
        approx_eq(
            compiled.call(std::f64::consts::PI).unwrap(),
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
            let grad_plain = compiled_plain.call(x).unwrap();
            let grad_recompute = compiled_recompute.call(x).unwrap();
            approx_eq(grad_plain, grad_recompute);
        }
    }

    #[test]
    fn test_compile_grad_checkpoint_gives_correct_gradient() {
        // Checkpoint with segment_size=2 should give the correct gradient for a function with
        // ~4 equations: x*x, sin(x), x*x + sin(x).
        let engine = ArrayScalarEngine::<f64>::new();
        let compiled = compile_grad_with_policy(
            &engine,
            quadratic_plus_sin,
            2.0f64,
            RematerializationPolicy::Checkpoint { segment_size: 2 },
        )
        .unwrap();

        approx_eq(compiled.call(2.0f64).unwrap(), 2.0 * 2.0 + 2.0f64.cos());
        approx_eq(compiled.call(0.5f64).unwrap(), 2.0 * 0.5 + 0.5f64.cos());
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
            approx_eq(compiled.call(x).unwrap(), expected);
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
            let grad_plain = compiled_plain.call(x).unwrap();
            let grad_checkpoint = compiled_checkpoint.call(x).unwrap();
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
            approx_eq(compiled_save_all.call(x).unwrap(), compiled_checkpoint.call(x).unwrap());
        }
    }

    #[test]
    fn test_compile_grad_checkpoint_large_segment_wraps_whole_program() {
        // Checkpoint with a segment_size larger than the number of equations should wrap
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
            approx_eq(compiled.call(x).unwrap(), 2.0 * x + x.cos());
        }
    }
}
