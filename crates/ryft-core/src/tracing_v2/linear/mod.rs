use std::{cell::RefCell, fmt::Debug, rc::Rc};

use crate::{
    parameters::{Parameterized, ParameterizedFamily, Placeholder},
    tracing::{
        Atom, AtomId, Instruction, InterpretableOperation, Operation, Program, ProgramBuilder, Traceable, TracingError,
        Value,
    },
    tracing_v2::{
        Differentiable, DifferentiableEngine, DifferentiableOperationStagingEngine, DifferentiableStagingEngine,
        DifferentiationError, LinearOperation,
        engines::{Engine, StagingEngine},
        forward::JvpTracer,
        jit::{Tracer, TracingEngine, interpret_and_trace},
        operations::{
            DifferentiableOperation, SupportsAdd, SupportsRematerialize,
            constants::{One, OneLike, SupportsZero, Zero, ZeroLike},
            rematerialize::{FlatTracedRematerialize, RematerializeOperation},
        },
    },
    types::{ArrayType, Type, Typed},
};

/// Dense Jacobian and Hessian materialization helpers.
mod dense;
/// Program-level linearization and transpose construction.
mod program;
/// Rematerialization-aware compiled-gradient helpers.
mod rematerialization;
/// Traced-program linearization helpers.
mod replay;
/// Public reverse-mode APIs built from traced programs and staged pullbacks.
mod reverse;

pub use dense::{CoordinateValue, DenseJacobian, hessian, jacfwd, jacrev};
#[doc(hidden)]
pub use program::linearize_program;
pub use program::transpose_linear_program_with_output_examples;
pub use program::transpose_traced_linear_program;
pub use rematerialization::{RematerializationPolicy, compile_grad, compile_grad_with_policy};
#[doc(hidden)]
pub use replay::TracedLinearizableOperation;
#[doc(hidden)]
pub use replay::linearize_traced_program;
pub(crate) use reverse::jvp_traced;
pub use reverse::{grad, grad_with_aux, jvp_program, value_and_grad, value_and_grad_with_aux, vjp};

#[inline]
fn flat_leaf_parameter_structure(count: usize) -> Vec<Placeholder> {
    vec![Placeholder; count]
}

fn ensure_single_gradient_output<T, V>(outputs: &[V]) -> Result<(), TracingError>
where
    T: Type,
    V: Differentiable<T>,
{
    if outputs.len() != 1 {
        return Err(DifferentiationError::InvalidGradientOutputLeafCount { expected: 1, got: outputs.len() }.into());
    }
    Ok(())
}

/// Traces one type-directed body and normalizes the captured program to flat leaf vectors.
///
/// Many linearization helpers want a uniform "flat vector of leaves" view even when the caller's
/// original function uses tuples, structs, or other parameterized shapes. This helper is the bridge
/// between those worlds: it traces the structured function once, then retags the captured program
/// so downstream reverse-mode code can operate on a canonical `Vec<V>` representation.
pub(crate) fn trace_flat_program_from_input_types<'engine, Input, Output, V, E, F>(
    engine: &'engine E,
    function: F,
    input_types: Input,
) -> Result<(Output, Program<ArrayType, V, E::Operation, Vec<V>, Vec<V>>), TracingError>
where
    V: Traceable<ArrayType> + Parameterized<V, ParameterStructure = Placeholder>,
    Input: Parameterized<ArrayType, ParameterStructure: Clone>,
    Output: Parameterized<ArrayType, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<V> + ParameterizedFamily<Tracer<'engine, E>>,
    Output::Family: ParameterizedFamily<V> + ParameterizedFamily<Tracer<'engine, E>>,
    E: StagingEngine<Type = ArrayType, Value = V> + ?Sized + 'static,
    F: FnOnce(Input::To<Tracer<'engine, E>>) -> Result<Output::To<Tracer<'engine, E>>, TracingError>,
{
    trace_flat_program_from_trace_result::<Input, Output, V, E::Operation>(crate::tracing_v2::jit::trace(
        engine,
        function,
        input_types,
    )?)
}

pub(crate) fn trace_flat_program_from_input_engine<'engine, Input, Output, V, E, F>(
    tracing_engine: &TracingEngine<'engine, E>,
    function: F,
    input_types: Input,
) -> Result<(Output, Program<ArrayType, V, E::Operation, Vec<V>, Vec<V>>), TracingError>
where
    V: Traceable<ArrayType> + Parameterized<V, ParameterStructure = Placeholder>,
    Input: Parameterized<ArrayType, ParameterStructure: Clone>,
    Output: Parameterized<ArrayType, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<V> + ParameterizedFamily<Tracer<'engine, E>>,
    Output::Family: ParameterizedFamily<V> + ParameterizedFamily<Tracer<'engine, E>>,
    E: StagingEngine<Type = ArrayType, Value = V> + ?Sized + 'static,
    F: FnOnce(Input::To<Tracer<'engine, E>>) -> Result<Output::To<Tracer<'engine, E>>, TracingError>,
{
    let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, V, E::Operation>::new()));
    trace_flat_program_from_trace_result::<Input, Output, V, E::Operation>(crate::tracing_v2::jit::trace_with_engine(
        tracing_engine.sibling(builder),
        function,
        input_types,
    )?)
}

fn trace_flat_program_from_trace_result<Input, Output, V, O>(
    trace_result: (Output, Program<ArrayType, V, O, Input::To<V>, Output::To<V>>),
) -> Result<(Output, Program<ArrayType, V, O, Vec<V>, Vec<V>>), TracingError>
where
    V: Traceable<ArrayType> + Parameterized<V, ParameterStructure = Placeholder>,
    Input: Parameterized<ArrayType, ParameterStructure: Clone>,
    Output: Parameterized<ArrayType, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<V>,
    Output::Family: ParameterizedFamily<V>,
    O: Clone + Operation<ArrayType>,
{
    let (output_types, traced_program) = trace_result;
    let input_leaf_count = traced_program.input_ids.len();
    let output_leaf_count = output_types.parameter_structure().parameter_count();
    let Program { atoms, input_ids, output_ids, instructions, .. } = traced_program;
    let mut builder = ProgramBuilder::<ArrayType, V, O>::new();
    builder.atoms = atoms;
    builder.input_ids = input_ids;
    builder.instructions = instructions;
    let traced_program = builder
        .build(
            output_ids,
            flat_leaf_parameter_structure(input_leaf_count),
            flat_leaf_parameter_structure(output_leaf_count),
        )?
        .simplified()?;
    Ok((output_types, traced_program))
}

/// Linearizes one flat scalar traced program and stages its pullback with a unit cotangent seed.
///
/// This is the internal core of traced reverse-mode for scalar-output functions. Given a staged
/// primal body and symbolic primals from an enclosing trace, it builds the pushforward, transposes
/// it into a pullback, seeds that pullback with a symbolic one, and returns both the traced scalar
/// output and the traced gradient leaves.
fn reverse_mode_scalar_traced_program<'engine, V, E>(
    tracing_engine: TracingEngine<'engine, E>,
    traced_program: &Program<ArrayType, V, E::Operation, Vec<V>, Vec<V>>,
    traced_primals: Vec<Tracer<'engine, E>>,
) -> Result<(Tracer<'engine, E>, Vec<Tracer<'engine, E>>), TracingError>
where
    V: Traceable<ArrayType> + Differentiable<ArrayType, Tangent = V> + One<ArrayType>,
    E: DifferentiableStagingEngine<Type = ArrayType, Value = V> + ?Sized + 'static,
    E::Operation: TracedLinearizableOperation<'engine, E> + 'static,
    <E as DifferentiableStagingEngine>::LinearOperation<'engine>: Clone
        + InterpretableOperation<ArrayType, Tracer<'engine, E>>
        + LinearOperation<ArrayType, Tracer<'engine, E>, <E as DifferentiableStagingEngine>::LinearOperation<'engine>>
        + SupportsZero<ArrayType, Tracer<'engine, E>>,
{
    let (outputs, pushforward) = linearize_traced_program(tracing_engine.clone(), traced_program, traced_primals)?;
    ensure_single_gradient_output::<ArrayType, _>(outputs.as_slice())?;
    let traced_output = outputs[0].clone();
    let tracing_builder = traced_output.builder().clone();
    let pullback = transpose_traced_linear_program(tracing_engine.clone(), &pushforward)?;
    let seed_type = traced_output.r#type().into_owned();
    let _ = <V as One<ArrayType>>::one(&seed_type)?;
    let seed_value = tracing_engine.outer_engine().one(&seed_type)?;
    let seed_atom = tracing_builder.borrow_mut().add_constant(seed_value);
    let seed = traced_output.engine.tracer_from_staged_parts(seed_atom, seed_type);
    let traced_gradient = pullback.interpret(vec![seed])?;
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
    use ndarray::arr2;

    use crate::{
        parameters::Placeholder,
        tracing::{InterpretableOperation, Operation, ProgramBuilder, TracingError},
        tracing_v2::{
            CustomPrimitive, DifferentiableOperation, DifferentiationError, LinearOperation, LinearPrimitiveOperation,
            PrimitiveOperation, Sin,
            engines::ArrayScalarEngine,
            operations::{TranspositionContext, matrix::ndarray_support::Array2Engine},
            test_support,
        },
        types::{ArrayType, DataType, TypeError},
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

    impl Operation<ArrayType> for PanicReplayOp {
        fn name(&self) -> &'static str {
            "panic_replay"
        }

        fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
            if input_types.len() != 1 {
                return Err(TypeError {
                    message: format!("panic_replay expected 1 input type but got {}", input_types.len()),
                });
            }
            Ok(vec![input_types[0].clone()])
        }
    }

    impl InterpretableOperation<ArrayType, f64> for PanicReplayOp {
        fn interpret(&self, _inputs: &[f64]) -> Result<Vec<f64>, TracingError> {
            panic!("panic_replay interpret should not run during this transform")
        }
    }

    impl LinearOperation<ArrayType, f64> for PanicReplayOp {
        fn transpose(
            &self,
            _context: &mut TranspositionContext<'_, ArrayType, f64, LinearPrimitiveOperation<f64>>,
            output_cotangents: &[Option<crate::tracing::AtomId>],
        ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
            if output_cotangents.len() != 1 {
                return Err(TracingError::InvalidInputCount { expected: 1, got: output_cotangents.len() });
            }
            Ok(vec![output_cotangents[0]])
        }
    }

    impl DifferentiableOperation<ArrayScalarEngine<f64>> for PanicReplayOp {
        fn jvp(
            &self,
            _engine: &ArrayScalarEngine<f64>,
            _context: &mut crate::tracing_v2::JvpContext<'_, f64, LinearPrimitiveOperation<f64>>,
            inputs: &[JvpTracer<f64, crate::tracing::AtomId>],
        ) -> Result<Vec<JvpTracer<f64, crate::tracing::AtomId>>, TracingError> {
            if inputs.len() != 1 {
                return Err(TracingError::InvalidInputCount { expected: 1, got: inputs.len() });
            }
            Ok(vec![inputs[0].clone()])
        }
    }

    #[derive(Clone, Debug)]
    struct OrdinaryAddOperation;

    impl Display for OrdinaryAddOperation {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("ordinary_add")
        }
    }

    impl Operation<ArrayType> for OrdinaryAddOperation {
        fn name(&self) -> &'static str {
            "ordinary_add"
        }

        fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
            crate::tracing_v2::operations::AddOperation.infer_output_types(input_types)
        }
    }

    impl InterpretableOperation<ArrayType, f64> for OrdinaryAddOperation {
        fn interpret(&self, inputs: &[f64]) -> Result<Vec<f64>, TracingError> {
            crate::tracing_v2::operations::AddOperation.interpret(inputs)
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
            formatter.write_str("differentiable_add")
        }
    }

    impl Operation<ArrayType> for DifferentiableAddOperation {
        fn name(&self) -> &'static str {
            "differentiable_add"
        }

        fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
            crate::tracing_v2::operations::AddOperation.infer_output_types(input_types)
        }
    }

    impl InterpretableOperation<ArrayType, f64> for DifferentiableAddOperation {
        fn interpret(&self, inputs: &[f64]) -> Result<Vec<f64>, TracingError> {
            crate::tracing_v2::operations::AddOperation.interpret(inputs)
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

    impl StagingEngine for SplitCarrierEngine {
        type Operation = OrdinaryAddOperation;
    }

    impl DifferentiableEngine for SplitCarrierEngine {
        type DifferentiableOperation = DifferentiableAddOperation;
        type LinearOperation = LinearPrimitiveOperation<f64>;
    }

    impl DifferentiableOperation<SplitCarrierEngine> for DifferentiableAddOperation {
        fn jvp(
            &self,
            engine: &SplitCarrierEngine,
            context: &mut crate::tracing_v2::JvpContext<'_, f64, LinearPrimitiveOperation<f64>>,
            inputs: &[JvpTracer<f64, crate::tracing::AtomId>],
        ) -> Result<Vec<JvpTracer<f64, crate::tracing::AtomId>>, TracingError> {
            crate::tracing_v2::operations::AddOperation.jvp(engine, context, inputs)
        }
    }

    #[test]
    fn test_concrete_ad_uses_differentiation_operation_carrier() {
        let engine = SplitCarrierEngine;
        let (_, traced_program): (f64, Program<ArrayType, f64, OrdinaryAddOperation, f64, f64>) =
            crate::tracing_v2::interpret_and_trace(&engine, |x: Tracer<SplitCarrierEngine>| Ok(x.clone() + x), 2.0f64)
                .unwrap();
        assert_eq!(traced_program.instructions[0].operation.name(), "ordinary_add");

        let differentiable_staging_engine = DifferentiableOperationStagingEngine::new(&engine);
        let (_, differentiable_program): (f64, Program<ArrayType, f64, DifferentiableAddOperation, f64, f64>) =
            interpret_and_trace(
                differentiable_staging_engine,
                |x: Tracer<'_, DifferentiableOperationStagingEngine<SplitCarrierEngine>>| Ok(x.clone() + x),
                2.0f64,
            )
            .unwrap();
        assert_eq!(differentiable_program.instructions[0].operation.name(), "differentiable_add");

        let (primal, pushforward) = jvp_program(&engine, |x| Ok(x.clone() + x), 2.0f64).unwrap();

        approx_eq(primal, 4.0);
        approx_eq(pushforward.interpret(3.0f64).unwrap(), 6.0);
    }

    #[test]
    fn test_jvp_program_returns_the_primal_output_and_pushforward() {
        let engine = ArrayScalarEngine::<f64>::new();
        let (primal, pushforward) = jvp_program(&engine, |x| Ok(quadratic_plus_sin(x)), 2.0f64).unwrap();

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
        let engine = ArrayScalarEngine::<f64>::new();
        let (primal, pushforward) = jvp_program(&engine, |inputs| Ok(bilinear_sin(inputs)), (2.0f64, 3.0f64)).unwrap();
        let pullback = transpose_linear_program_with_output_examples(&pushforward, &[primal]).unwrap();
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
    fn linearize_program_does_not_replay_the_forward_program_to_recover_representatives() {
        let primitive = CustomPrimitive::<ArrayType, f64>::new(PanicReplayOp)
            .with_jvp_rule::<ArrayScalarEngine<f64>, _>(PanicReplayOp);
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new();
        let input = builder.add_input(3.0f64.r#type().into_owned());
        let output_atom = builder.add_variable(ArrayType::scalar(DataType::F64));
        builder.instructions.push(Instruction {
            operation: PrimitiveOperation::Custom(Arc::new(primitive)),
            inputs: vec![input],
            outputs: vec![output_atom],
        });
        let output = vec![output_atom];
        let program = builder.build::<f64, f64>(output, Placeholder, Placeholder).unwrap();

        let engine = ArrayScalarEngine::<f64>::new();
        let pushforward = linearize_program(&engine, &program, vec![3.0f64]).unwrap();
        approx_eq(pushforward.interpret(2.5f64).unwrap(), 2.5);
    }

    #[test]
    fn transpose_linear_program_does_not_replay_the_forward_linear_program_to_recover_representatives() {
        let primitive = LinearPrimitiveOperation::custom(
            CustomPrimitive::<ArrayType, f64>::new(PanicReplayOp).with_transpose_rule(PanicReplayOp),
        )
        .unwrap();
        let mut builder = ProgramBuilder::<ArrayType, f64, LinearPrimitiveOperation<f64>>::new();
        let input = builder.add_input(0.0f64.r#type().into_owned());
        let output_atom = builder.add_variable(ArrayType::scalar(DataType::F64));
        builder.instructions.push(Instruction {
            operation: primitive,
            inputs: vec![input],
            outputs: vec![output_atom],
        });
        let output = vec![output_atom];
        let program = builder.build::<f64, f64>(output, Placeholder, Placeholder).unwrap();
        let pushforward = program;

        let pullback = super::program::transpose_linear_program_with_output_examples(&pushforward, &[0.0f64]).unwrap();
        approx_eq(pullback.interpret(4.0f64).unwrap(), 4.0);
    }

    #[test]
    fn linear_program_display_delegates_to_the_underlying_program() {
        let engine = ArrayScalarEngine::<f64>::new();
        let (_, pushforward): (f64, Program<ArrayType, f64, LinearPrimitiveOperation<f64>, f64, f64>) =
            jvp_program(&engine, |x| Ok(quadratic_plus_sin(x)), 2.0f64).unwrap();

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

    #[test]
    fn test_scalar_gradient_output_requires_single_leaf() {
        let result = ensure_single_gradient_output::<ArrayType, _>(&[1.0f64, 2.0f64]);

        assert!(matches!(
            result,
            Err(TracingError::Differentiation(DifferentiationError::InvalidGradientOutputLeafCount {
                expected: 1,
                got: 2
            }))
        ));
    }

    #[test]
    fn test_compile_grad_rejects_non_scalar_array_output() {
        let engine = Array2Engine::<f64>::new();

        let result = compile_grad(&engine, |input| input, arr2(&[[1.0, 2.0], [3.0, 4.0]]));

        assert!(matches!(
            result,
            Err(TracingError::Differentiation(DifferentiationError::NonScalarGradientOutput { output_type }))
                if output_type.rank() == 2
        ));
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
        // the entire program in a single RematerializeOperation, equivalent to RecomputeAll.
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
