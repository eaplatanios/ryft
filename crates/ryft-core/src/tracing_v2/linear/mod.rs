use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;

use crate::operations::constants::{One, SupportsZero, Zero};
use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::{Parameterized, ParameterizedFamily, Placeholder};
use crate::tracing::engines::{Engine, Tracer, TracingContext, TracingEngine};
use crate::tracing::transposition::LinearOperation;
use crate::tracing::{Atom, AtomId, Instruction, Program, ProgramBuilder, Traceable, TracingError, Value};
use crate::tracing_v2::differentiation::JvpTracer;
use crate::tracing_v2::operations::constants::{OneLike, SupportsZeroLike, ZeroLike};
use crate::tracing_v2::operations::rematerialize::{FlatTracedRematerialize, RematerializeOperation};
use crate::tracing_v2::operations::{AddOperation, SupportsAdd, SupportsRematerialize};
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
    use std::fmt::{Debug, Display};
    use std::sync::Arc;

    use crate::macros::check_input_count;
    use crate::operations::{InterpretableOperation, Operation};
    use crate::parameters::Placeholder;
    use crate::tracing::engines::ScalarEngine;
    use crate::tracing::transposition::TranspositionContext;
    use crate::tracing::{ProgramBuilder, TracingError};
    use crate::tracing_v2::{
        ArrayOperation, CustomPrimitive, DifferentiableOperation, LinearArrayOperation, LinearScalarOperation, Sin,
    };
    use crate::types::{ArrayType, DataType, TypeError, Typed};
    use indoc::indoc;

    use super::*;

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    struct ArrayScalarEngine;

    impl Engine for ArrayScalarEngine {
        type Type = ArrayType;
        type Value = f64;

        fn zero(&self, _type: &ArrayType) -> Result<f64, TracingError> {
            Ok(0.0)
        }

        fn one(&self, _type: &ArrayType) -> Result<f64, TracingError> {
            Ok(1.0)
        }
    }

    impl TracingEngine for ArrayScalarEngine {
        type OperationCarrier = ArrayOperation<f64, ArrayType>;
    }

    impl crate::tracing_v2::LinearizableEngine for ArrayScalarEngine {
        type LinearOperationCarrier = LinearArrayOperation<f64, ArrayType>;
    }

    impl DifferentiableEngine for ArrayScalarEngine {
        type DifferentiableOperationCarrier = ArrayOperation<f64, ArrayType>;
    }

    impl DifferentiableTracingEngine for ArrayScalarEngine {
        type LinearOperationCarrier<'engine>
            = LinearArrayOperation<Tracer<'engine, Self>, ArrayType>
        where
            Self: 'engine;
    }

    #[derive(Clone, Debug, Default)]
    struct PanicReplayOp;

    impl Display for PanicReplayOp {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str(self.name())
        }
    }

    impl Operation<ArrayType> for PanicReplayOp {
        #[inline]
        fn name(&self) -> &'static str {
            "panic_replay"
        }

        fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
            check_input_count!(input_types, 1, TypeError);
            Ok(vec![input_types[0].clone()])
        }
    }

    impl InterpretableOperation<ArrayType, f64> for PanicReplayOp {
        fn interpret(&self, _inputs: &[f64]) -> Result<Vec<f64>, TracingError> {
            panic!("panic_replay interpret should not run during this transform")
        }
    }

    impl LinearOperation<ArrayType, f64, LinearArrayOperation<f64, ArrayType>> for PanicReplayOp {
        fn transpose(
            &self,
            _context: &mut TranspositionContext<ArrayType, f64, LinearArrayOperation<f64, ArrayType>>,
            output_cotangents: &[Option<crate::tracing::AtomId>],
        ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
            check_input_count!(output_cotangents, 1, TracingError);
            Ok(vec![output_cotangents[0]])
        }
    }

    impl DifferentiableOperation<ArrayScalarEngine> for PanicReplayOp {
        fn jvp(
            &self,
            _context: &mut crate::tracing_v2::JvpContext<'_, ArrayScalarEngine>,
            inputs: &[JvpTracer<f64, crate::tracing::AtomId>],
        ) -> Result<Vec<JvpTracer<f64, crate::tracing::AtomId>>, TracingError> {
            check_input_count!(inputs, 1, TracingError);
            Ok(vec![inputs[0].clone()])
        }
    }

    #[derive(Clone, Debug)]
    struct OrdinaryAddOperation;

    impl Display for OrdinaryAddOperation {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str(self.name())
        }
    }

    impl Operation<ArrayType> for OrdinaryAddOperation {
        #[inline]
        fn name(&self) -> &'static str {
            "ordinary_add"
        }

        fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
            crate::tracing_v2::operations::AddOperation.infer_output_types(input_types)
        }
    }

    impl InterpretableOperation<ArrayType, f64> for OrdinaryAddOperation {
        fn interpret(&self, inputs: &[f64]) -> Result<Vec<f64>, TracingError> {
            <crate::tracing_v2::operations::AddOperation as InterpretableOperation<ArrayType, f64>>::interpret(
                &crate::tracing_v2::operations::AddOperation,
                inputs,
            )
        }
    }

    impl crate::tracing_v2::operations::SupportsAdd<ArrayType, f64> for OrdinaryAddOperation {
        fn add_operation() -> Self {
            Self
        }
    }

    #[derive(Clone, Debug)]
    struct DifferentiableAddOperation;

    impl Display for DifferentiableAddOperation {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str(self.name())
        }
    }

    impl Operation<ArrayType> for DifferentiableAddOperation {
        #[inline]
        fn name(&self) -> &'static str {
            "differentiable_add"
        }

        fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
            crate::tracing_v2::operations::AddOperation.infer_output_types(input_types)
        }
    }

    impl InterpretableOperation<ArrayType, f64> for DifferentiableAddOperation {
        fn interpret(&self, inputs: &[f64]) -> Result<Vec<f64>, TracingError> {
            <crate::tracing_v2::operations::AddOperation as InterpretableOperation<ArrayType, f64>>::interpret(
                &crate::tracing_v2::operations::AddOperation,
                inputs,
            )
        }
    }

    impl crate::tracing_v2::operations::SupportsAdd<ArrayType, f64> for DifferentiableAddOperation {
        fn add_operation() -> Self {
            Self
        }
    }

    struct SplitCarrierEngine;

    impl Engine for SplitCarrierEngine {
        type Type = ArrayType;
        type Value = f64;

        fn zero(&self, _type: &ArrayType) -> Result<f64, TracingError> {
            Ok(0.0)
        }

        fn one(&self, _type: &ArrayType) -> Result<f64, TracingError> {
            Ok(1.0)
        }
    }

    impl TracingEngine for SplitCarrierEngine {
        type OperationCarrier = OrdinaryAddOperation;
    }

    impl crate::tracing_v2::LinearizableEngine for SplitCarrierEngine {
        type LinearOperationCarrier = LinearArrayOperation<f64, ArrayType>;
    }

    impl DifferentiableEngine for SplitCarrierEngine {
        type DifferentiableOperationCarrier = DifferentiableAddOperation;
    }

    impl DifferentiableOperation<SplitCarrierEngine> for DifferentiableAddOperation {
        fn jvp(
            &self,
            context: &mut crate::tracing_v2::JvpContext<'_, SplitCarrierEngine>,
            inputs: &[JvpTracer<f64, crate::tracing::AtomId>],
        ) -> Result<Vec<JvpTracer<f64, crate::tracing::AtomId>>, TracingError> {
            crate::tracing_v2::operations::AddOperation.jvp(context, inputs)
        }
    }

    #[test]
    fn test_concrete_ad_uses_differentiation_operation_carrier() {
        let engine = SplitCarrierEngine;
        let (_, traced_program): (f64, Program<ArrayType, f64, OrdinaryAddOperation, f64, f64>) =
            engine.interpret_and_trace(|x: Tracer<SplitCarrierEngine>| Ok(x.clone() + x), 2.0f64).unwrap();
        assert_eq!(traced_program.instructions[0].operation.name(), "ordinary_add");

        let differentiable_tracing_engine = DifferentiableOperationTracingEngine::new(&engine);
        let (_, differentiable_program): (f64, Program<ArrayType, f64, DifferentiableAddOperation, f64, f64>) =
            differentiable_tracing_engine
                .interpret_and_trace(
                    |x: Tracer<'_, DifferentiableOperationTracingEngine<SplitCarrierEngine>>| Ok(x.clone() + x),
                    2.0f64,
                )
                .unwrap();
        assert_eq!(differentiable_program.instructions[0].operation.name(), "differentiable_add");

        let (primal, pushforward) = linearize(&engine, |x| Ok(x.clone() + x), 2.0f64).unwrap();

        approx_eq(primal, 4.0);
        approx_eq(pushforward.interpret(3.0f64).unwrap(), 6.0);
    }

    #[test]
    fn test_linearize_returns_the_primal_output_and_pushforward() {
        let engine = ArrayScalarEngine;
        let (primal, pushforward) = linearize(&engine, |x| Ok(x.clone() * x.clone() + x.sin()), 2.0f64).unwrap();

        approx_eq(primal, 2.0f64.powi(2) + 2.0f64.sin());
        approx_eq(pushforward.interpret(1.5f64).unwrap(), (4.0 + 2.0f64.cos()) * 1.5);
        assert_eq!(
            pushforward.to_string(),
            indoc! {"
            lambda %0:f64[] .
            let %1:f64[] = scale [factor=2] %0
                %2:f64[] = scale [factor=2] %0
                %3:f64[] = add %1 %2
                %4:f64[] = scale [factor=-0.4161468365471424] %0
                %5:f64[] = add %3 %4
            in (%5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_transposed_linear_program_matches_the_reverse_mode_pullback() {
        let engine = ArrayScalarEngine;
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
            lambda %0:f64[] .
            let %1:f64[] = scale [factor=-0.4161468365471424] %0
                %2:f64[] = scale [factor=3] %0
                %3:f64[] = add %1 %2
                %4:f64[] = scale [factor=2] %0
            in (%3, %4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn program_linearize_does_not_replay_the_forward_program_to_recover_representatives() {
        let primitive =
            CustomPrimitive::<ArrayType, f64>::new(PanicReplayOp).with_jvp_rule::<ArrayScalarEngine, _>(PanicReplayOp);
        let mut builder = ProgramBuilder::<ArrayType, f64, ArrayOperation<f64, ArrayType>>::new();
        let input = builder.add_input(<f64 as Typed<ArrayType>>::r#type(&3.0f64).into_owned());
        let output_atom = builder.add_variable(ArrayType::scalar(DataType::F64));
        builder.instructions.push(Instruction {
            operation: ArrayOperation::Custom(Arc::new(primitive)),
            inputs: vec![input],
            outputs: vec![output_atom],
        });
        let output = vec![output_atom];
        let program = builder.build::<f64, f64>(output, Placeholder, Placeholder).unwrap();

        let engine = ArrayScalarEngine;
        let pushforward = program.linearize(&engine, vec![3.0f64]).unwrap();
        approx_eq(pushforward.interpret(2.5f64).unwrap(), 2.5);
    }

    #[test]
    fn transpose_linear_program_does_not_replay_the_forward_linear_program_to_recover_representatives() {
        let primitive = LinearArrayOperation::custom(
            CustomPrimitive::<ArrayType, f64>::new(PanicReplayOp).with_transpose_rule(PanicReplayOp),
        )
        .unwrap();
        let mut builder = ProgramBuilder::<ArrayType, f64, LinearArrayOperation<f64, ArrayType>>::new();
        let input = builder.add_input(<f64 as Typed<ArrayType>>::r#type(&0.0f64).into_owned());
        let output_atom = builder.add_variable(ArrayType::scalar(DataType::F64));
        builder.instructions.push(Instruction {
            operation: primitive,
            inputs: vec![input],
            outputs: vec![output_atom],
        });
        let output = vec![output_atom];
        let program = builder.build::<f64, f64>(output, Placeholder, Placeholder).unwrap();
        let pushforward = program;

        let pullback = pushforward.transpose(&[0.0f64]).unwrap();
        approx_eq(pullback.interpret(4.0f64).unwrap(), 4.0);
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

    // -----------------------------------------------------------------------
    // RematerializationPolicy tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compile_grad_save_all_matches_compile_grad() {
        // SaveAll should produce the same gradient as the plain compile_grad.
        let engine = ArrayScalarEngine;
        let compiled_plain = compile_grad(&engine, |x| x.clone() * x.clone() + x.sin(), 2.0f64).unwrap();
        let compiled_save_all = compile_grad_with_policy(
            &engine,
            |x| x.clone() * x.clone() + x.sin(),
            2.0f64,
            RematerializationPolicy::SaveAll,
        )
        .unwrap();

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
        let engine = ArrayScalarEngine;
        let compiled = compile_grad_with_policy(
            &engine,
            |x| x.clone() * x.clone() + x.sin(),
            2.0f64,
            RematerializationPolicy::RecomputeAll,
        )
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
        let engine = ArrayScalarEngine;
        let compiled_plain = compile_grad(&engine, |x| x.clone() * x.clone() + x.sin(), 2.0f64).unwrap();
        let compiled_recompute = compile_grad_with_policy(
            &engine,
            |x| x.clone() * x.clone() + x.sin(),
            2.0f64,
            RematerializationPolicy::RecomputeAll,
        )
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
        let engine = ArrayScalarEngine;
        let compiled = compile_grad_with_policy(
            &engine,
            |x| x.clone() * x.clone() + x.sin(),
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
        let engine = ArrayScalarEngine;
        let compiled = compile_grad_with_policy(
            &engine,
            |x| x.clone() * x.clone() + x.sin(),
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
        let engine = ArrayScalarEngine;
        let compiled_plain = compile_grad(&engine, |x| x.clone() * x.clone() + x.sin(), 2.0f64).unwrap();
        let compiled_checkpoint = compile_grad_with_policy(
            &engine,
            |x| x.clone() * x.clone() + x.sin(),
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
        let engine = ArrayScalarEngine;
        let compiled_save_all = compile_grad_with_policy(
            &engine,
            |x| x.clone() * x.clone() + x.sin(),
            2.0f64,
            RematerializationPolicy::SaveAll,
        )
        .unwrap();
        let compiled_checkpoint = compile_grad_with_policy(
            &engine,
            |x| x.clone() * x.clone() + x.sin(),
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
        // the entire program in a single RematerializeOperation, equivalent to RecomputeAll.
        let engine = ArrayScalarEngine;
        let compiled = compile_grad_with_policy(
            &engine,
            |x| x.clone() * x.clone() + x.sin(),
            2.0f64,
            RematerializationPolicy::Checkpoint { segment_size: 100 },
        )
        .unwrap();

        for x in [0.0, 1.0, 2.0, std::f64::consts::PI] {
            approx_eq(compiled.interpret(x).unwrap(), 2.0 * x + x.cos());
        }
    }
}
