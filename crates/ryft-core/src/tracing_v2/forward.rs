use std::fmt::Debug;

use crate::operations::InterpretableOperation;
use crate::operations::arithmetic::{AddOperation, SupportsAdd};
use crate::operations::constants::SupportsZeroLike;
use crate::operations::constants::Zero;
use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily, Placeholder};
use crate::tracing::engines::{Engine, Tracer, TracingContext, TracingEngine};
use crate::tracing::{Program, Traceable, TracingError, Value};
use crate::tracing_v2::differentiation::Differentiable;
use crate::tracing_v2::linear::linearize;
use crate::tracing_v2::operations::{SupportsNeg, SupportsScale};
use crate::tracing_v2::{
    DifferentiableEngine, DifferentiableOperation, DifferentiableOperationTracingEngine, DifferentiableTracingEngine,
    DifferentiationError,
};
use crate::types::Typed;

/// Evaluates `function` on `primals` and propagates the supplied tangent values forward.
///
/// The returned pair is `(primal_output, tangent_output)`. Architecturally, [`jvp`] is the most
/// direct forward-mode transform in the crate: it either traces the body once to build a staged
/// pushforward or stages the whole JVP into an outer trace if the inputs are already symbolic.
/// Primitive-specific local JVP rules live in [`crate::tracing_v2::operations`]; [`jvp`] is the
/// orchestration layer that selects the concrete or traced execution path.
#[allow(private_bounds, private_interfaces)]
pub fn jvp<
    'engine,
    E: Engine,
    F: FnOnce(D::FunctionInput) -> D::FunctionOutput,
    Input: Parameterized<D, ParameterStructure: Debug + PartialEq>,
    Output: Parameterized<D>,
    D: JvpDispatch<'engine, E, Input, Output, Marker>,
    Marker,
>(
    engine: &'engine E,
    function: F,
    primals: Input,
    tangents: Input,
) -> Result<(Output, Output), TracingError> {
    D::invoke(engine, function, primals, tangents)
}

/// Marker selecting concrete-value [`jvp`] dispatch.
pub(crate) struct JvpDispatchValueMarker;

/// Marker selecting already-traced [`jvp`] dispatch.
pub(crate) struct JvpDispatchTracerMarker;

/// Dispatch trait used by [`jvp`] so it can operate both on concrete values and on already traced values.
///
/// The public transform is intentionally small; this trait is where the concrete, traced, and
/// batched execution strategies branch apart.
pub(crate) trait JvpDispatch<
    'engine,
    E: Engine,
    Input: Parameterized<Self, ParameterStructure: Debug + PartialEq>,
    Output: Parameterized<Self>,
    Marker,
>: Parameter + Sized
{
    /// Input type expected by the user-provided function.
    type FunctionInput;

    /// Output type produced by the user-provided function.
    type FunctionOutput;

    /// Invokes [`jvp`] for one leaf regime.
    fn invoke<F: FnOnce(Self::FunctionInput) -> Self::FunctionOutput>(
        engine: &'engine E,
        function: F,
        primals: Input,
        tangents: Input,
    ) -> Result<(Output, Output), TracingError>;
}

/// Concrete-value dispatch for [`jvp`]: traces the user function with [`Tracer`] to build a staged
/// pushforward via [`linearize`] and evaluates it at the supplied tangents.
impl<
    'engine,
    E: DifferentiableEngine<
            Value = V,
            LinearOperationCarrier: InterpretableOperation<E::Type, V>
                                        + SupportsNeg<E::Type, V>
                                        + SupportsAdd<E::Type, V>
                                        + SupportsScale<E::Type, V>,
        > + 'static,
    V: Value<E::Type>
        + Differentiable<E::Type, Tangent = V>
        + Zero<E::Type>
        + Parameterized<V, ParameterStructure: PartialEq>,
    Input: Parameterized<
            V,
            Family: for<'call> ParameterizedFamily<Tracer<'call, DifferentiableOperationTracingEngine<E>>>,
            ParameterStructure: Debug + PartialEq,
        >,
    Output: for<'call> Parameterized<
            V,
            Family: ParameterizedFamily<Tracer<'call, DifferentiableOperationTracingEngine<E>>>,
            To<Tracer<'call, DifferentiableOperationTracingEngine<E>>>: Parameterized<
                Tracer<'call, DifferentiableOperationTracingEngine<E>>,
                To<V> = Output,
            >,
        >,
> JvpDispatch<'engine, E, Input, Output, JvpDispatchValueMarker> for V
{
    type FunctionInput = Input::To<Tracer<'engine, DifferentiableOperationTracingEngine<E>>>;
    type FunctionOutput = Output::To<Tracer<'engine, DifferentiableOperationTracingEngine<E>>>;

    fn invoke<F: FnOnce(Self::FunctionInput) -> Self::FunctionOutput>(
        engine: &'engine E,
        function: F,
        primals: Input,
        tangents: Input,
    ) -> Result<(Output, Output), TracingError> {
        let primal_structure = primals.parameter_structure();
        let tangent_structure = tangents.parameter_structure();
        if primal_structure != tangent_structure {
            return Err(ParameterError::MismatchedParameterStructures {
                left_structure: format!("{primal_structure:?}"),
                right_structure: format!("{tangent_structure:?}"),
            }
            .into());
        }

        let (primal_output, tangent_program): (Output, Program<E::Type, V, E::LinearOperationCarrier, Input, Output>) =
            linearize(engine, |input| Ok(function(input)), primals)?;
        let tangent_output = tangent_program.interpret(tangents)?;
        Ok((primal_output, tangent_output))
    }
}

/// Already-traced dispatch for [`jvp`]: replays the user function symbolically inside an enclosing
/// [`Tracer`] scope, staging both the primal output and tangent propagation as part of the outer
/// compiled program.
impl<
    'engine,
    E: DifferentiableTracingEngine<Value = V> + TracingEngine + 'static,
    V: Traceable<E::Type> + Differentiable<E::Type, Tangent = V> + Parameterized<V, ParameterStructure = Placeholder>,
    Input,
    Output,
> JvpDispatch<'engine, E, Input, Output, JvpDispatchTracerMarker> for Tracer<'engine, E>
where
    E::OperationCarrier:
        DifferentiableOperation<TracingContext<'engine, E>> + SupportsZeroLike<E::Type, V> + SupportsAdd<E::Type, V>,
    E::LinearOperationCarrier<'engine>: InterpretableOperation<E::Type, Tracer<'engine, E>>,
    Input: Parameterized<Tracer<'engine, E>, To<Tracer<'engine, E>> = Input>,
    Input::Family: ParameterizedFamily<Tracer<'engine, E>> + ParameterizedFamily<V> + ParameterizedFamily<E::Type>,
    Input::To<E::Type>: Parameterized<E::Type, To<Tracer<'engine, E>> = Input>,
    Input::ParameterStructure: Debug + PartialEq,
    Output: Parameterized<Tracer<'engine, E>, To<Tracer<'engine, E>> = Output>,
    Output::Family: ParameterizedFamily<Tracer<'engine, E>> + ParameterizedFamily<V> + ParameterizedFamily<E::Type>,
    Output::To<E::Type>: Parameterized<E::Type, To<Tracer<'engine, E>> = Output>,
    AddOperation: InterpretableOperation<E::Type, Tracer<'engine, E>>,
{
    type FunctionInput = Input;
    type FunctionOutput = Output;

    fn invoke<F: FnOnce(Self::FunctionInput) -> Self::FunctionOutput>(
        _engine: &'engine E,
        function: F,
        primals: Input,
        tangents: Input,
    ) -> Result<(Output, Output), TracingError> {
        let primal_structure = primals.parameter_structure();
        let tangent_structure = tangents.parameter_structure();
        if primal_structure != tangent_structure {
            return Err(ParameterError::MismatchedParameterStructures {
                left_structure: format!("{primal_structure:?}"),
                right_structure: format!("{tangent_structure:?}"),
            }
            .into());
        }

        let traced_primals = primals.into_parameters().collect::<Vec<_>>();
        let traced_tangents = tangents.into_parameters().collect::<Vec<_>>();
        let Some(tracing_context) = traced_primals.first().map(|traced_primal| traced_primal.context.clone()) else {
            return Err(DifferentiationError::MissingTracedJvpInputLeaves.into());
        };
        let staged_input_types = Input::To::<E::Type>::from_parameters(
            primal_structure,
            traced_primals.iter().map(|traced_primal| traced_primal.r#type().into_owned()).collect::<Vec<_>>(),
        )?;
        let (primal_output_types, traced_program) =
            tracing_context.engine.trace(|staged_input| Ok(function(staged_input)), staged_input_types)?;
        let output_structure = primal_output_types.parameter_structure();
        let (traced_primal_output, pushforward) = tracing_context.linearize(&traced_program, traced_primals)?;
        let traced_tangent_output = pushforward.interpret(traced_tangents)?;
        Ok((
            Output::from_parameters(output_structure.clone(), traced_primal_output)?,
            Output::from_parameters(output_structure, traced_tangent_output)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use indoc::indoc;

    use crate::parameters::{ParameterError, Parameterized};
    use crate::tracing::engines::{ScalarEngine, TracingContext};
    use crate::tracing::{Program, ProgramBuilder};
    use crate::tracing_v2::DifferentiableOperation;
    use crate::tracing_v2::differentiation::{JvpContext, JvpTracer};
    use crate::tracing_v2::{LinearScalarOperation, ScalarOperation, Sin};
    use crate::types::DataType;

    use super::*;

    /// Validates that [`TracingContext`] can host a JVP rule like [`AddOperation`] when its
    /// `Value` is `Tracer<E>`: the rule stages its primal effect through the underlying engine and
    /// its tangent effect through the context's `LinearOperation` carrier.
    #[test]
    fn tracing_context_dispatches_add_jvp_with_traced_primals() {
        let engine = ScalarEngine::<f64>::new();
        let outer_builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let outer_input_a = outer_builder.borrow_mut().add_input(crate::types::DataType::F64);
        let outer_input_b = outer_builder.borrow_mut().add_input(crate::types::DataType::F64);
        let outer_tracing_context = TracingContext::new(&engine, outer_builder.clone());
        let primal_a = outer_tracing_context.tracer(outer_input_a, None);
        let primal_b = outer_tracing_context.tracer(outer_input_b, None);

        let linear_builder = Rc::new(RefCell::new(ProgramBuilder::<
            DataType,
            Tracer<'_, ScalarEngine<f64>>,
            LinearScalarOperation<Tracer<'_, ScalarEngine<f64>>>,
        >::new()));
        let tangent_a = linear_builder.borrow_mut().add_input(crate::types::DataType::F64);
        let tangent_b = linear_builder.borrow_mut().add_input(crate::types::DataType::F64);
        let mut context = JvpContext::new(&outer_tracing_context, linear_builder.clone());

        let outputs = AddOperation
            .jvp(
                &mut context,
                &[
                    JvpTracer { primal: primal_a, tangent: tangent_a },
                    JvpTracer { primal: primal_b, tangent: tangent_b },
                ],
            )
            .expect("AddOperation::jvp should run on a TracingContext");

        assert_eq!(outputs.len(), 1);
        assert_eq!(linear_builder.borrow().instructions.len(), 1);
        assert_eq!(outer_builder.borrow().instructions.len(), 1);
    }

    #[test]
    fn jvp_rejects_mismatched_parameter_structures() {
        let engine = ScalarEngine::<f64>::new();
        let result: Result<(f64, f64), TracingError> =
            jvp(&engine, |xs| xs[0].clone(), vec![2.0f64], vec![1.0f64, 2.0f64]);
        assert!(matches!(
            result,
            Err(TracingError::Parameter(ParameterError::MismatchedParameterStructures {
                left_structure,
                right_structure,
            })) if left_structure == format!("{:?}", vec![2.0f64].parameter_structure())
                && right_structure == format!("{:?}", vec![1.0f64, 2.0f64].parameter_structure())
        ));

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
    fn traced_jvp_requires_input_leaves() {
        let engine = ScalarEngine::<f64>::new();
        let empty_primals: Vec<Tracer<'_, ScalarEngine<f64>>> = Vec::new();
        let empty_tangents: Vec<Tracer<'_, ScalarEngine<f64>>> = Vec::new();

        let result: Result<(Vec<Tracer<'_, ScalarEngine<f64>>>, Vec<Tracer<'_, ScalarEngine<f64>>>), TracingError> =
            jvp(&engine, |inputs: Vec<Tracer<'_, ScalarEngine<f64>>>| inputs, empty_primals, empty_tangents);

        assert!(matches!(
            result,
            Err(TracingError::Differentiation(DifferentiationError::MissingTracedJvpInputLeaves))
        ));
    }
}
