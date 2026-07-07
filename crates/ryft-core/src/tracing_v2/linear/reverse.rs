use std::fmt::Debug;

use crate::differentiation::{DifferentiableType, TransposableOperation};
use crate::interpretation::InterpretableOperation;
use crate::operations::BooleanLike;
use crate::operations::arithmetic::AddOperation;
use crate::operations::constants::ZeroOperation;
use crate::operations::control_flow::MaybeWhile;
use crate::partial::PartiallyEvaluatableOperation;
use crate::tracing::TracingContext;
use crate::tracing_v2::differentiation::DifferentiableOperation;
use crate::tracing_v2::{Differentiate, DifferentiationError, NestedTracer};
use crate::{Context, Domain, One, Parameterized, ParameterizedFamily, ProgramError, Type, Typed, Value, Zero};

/// Computes both the primal scalar output and its reverse-mode gradient.
///
/// This is the most direct reverse-mode API when the caller needs both the function value and the
/// gradient at the same primal point. The function must return exactly one rank-0 scalar array
/// leaf. Use [`Differentiate::vjp`] directly for vector-valued functions that need an explicit output
/// cotangent.
#[allow(private_bounds)]
pub fn value_and_grad<C, F, Input>(
    context: &C,
    function: F,
    primals: Input,
) -> Result<(<C as Domain>::Value, Input::To<<C as Domain>::Value>), DifferentiationError>
where
    C: Context,
    <C as Domain>::Constant: Value<Type = <C as Domain>::Type>,
    <C as Domain>::Value: BooleanLike,
    F: FnOnce(Input::To<NestedTracer<C>>) -> NestedTracer<C>,
    Input: Parameterized<
            <C as Domain>::Value,
            Family: ParameterizedFamily<NestedTracer<C>>,
            To<<C as Domain>::Value> = Input,
            ParameterStructure: Debug + PartialEq,
        >,
    C: One<<C as Domain>::Value>,
    <C as Domain>::Type: DifferentiableType,
    <C as Domain>::Operation: Clone
        + InterpretableOperation<<C as Domain>::Value, C>
        + TransposableOperation<<C as Domain>::Constant, <C as Domain>::Operation>
        + MaybeWhile<<C as Domain>::Constant, <C as Domain>::Operation>
        + From<ZeroOperation<<C as Domain>::Type>>
        + From<AddOperation>
        + DifferentiableOperation<TracingContext<<C as Domain>::Constant, <C as Domain>::Operation>>
        + PartiallyEvaluatableOperation<TracingContext<<C as Domain>::Constant, <C as Domain>::Operation>>,
{
    let input_structure = primals.parameter_structure();
    let (output, pullback, residuals) = context.vjp(|input| Ok(function(input)), primals)?;
    // Reverse mode only defines a gradient for scalar-output functions; reject non-scalar outputs before seeding
    // (see `DifferentiationError::NonScalarGradientOutput`).
    if !output.r#type().is_scalar() {
        return Err(DifferentiationError::NonScalarGradientOutput { output_type: output.r#type().to_string() });
    }
    // The direct-transpose pullback consumes `[output_cotangents ++ residuals]`, so the single scalar-output seed is
    // followed by the linearization-point residuals; its flat input cotangents are reshaped against the closure's
    // input structure. Both the seed and the replay go through the domain itself: an eager domain constructs and
    // interprets concrete values, while a staging domain stages into its enclosing trace.
    let mut pullback_inputs = vec![context.one(output.r#type().as_ref())?];
    pullback_inputs.extend(residuals);
    let input_cotangents = pullback.interpret_in_context(context, pullback_inputs)?;
    let gradient = Input::To::<<C as Domain>::Value>::from_parameters(input_structure, input_cotangents)
        .map_err(ProgramError::from)?;
    Ok((output, gradient))
}

/// Computes the reverse-mode gradient of a scalar-output function.
///
/// This is [`value_and_grad`] with the primal value discarded — the analogue of JAX's
/// [`grad`](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html). The function must return
/// exactly one rank-0 scalar array leaf. Use [`value_and_grad`] when the function value is also
/// needed, and [`grad_with_aux`] when the function carries auxiliary outputs.
#[allow(private_bounds)]
pub fn grad<C, F, Input>(
    context: &C,
    function: F,
    primals: Input,
) -> Result<Input::To<<C as Domain>::Value>, DifferentiationError>
where
    C: Context,
    <C as Domain>::Constant: Value<Type = <C as Domain>::Type>,
    <C as Domain>::Value: BooleanLike,
    F: FnOnce(Input::To<NestedTracer<C>>) -> NestedTracer<C>,
    Input: Parameterized<
            <C as Domain>::Value,
            Family: ParameterizedFamily<NestedTracer<C>>,
            To<<C as Domain>::Value> = Input,
            ParameterStructure: Debug + PartialEq,
        >,
    C: One<<C as Domain>::Value>,
    <C as Domain>::Type: DifferentiableType,
    <C as Domain>::Operation: Clone
        + InterpretableOperation<<C as Domain>::Value, C>
        + TransposableOperation<<C as Domain>::Constant, <C as Domain>::Operation>
        + MaybeWhile<<C as Domain>::Constant, <C as Domain>::Operation>
        + From<ZeroOperation<<C as Domain>::Type>>
        + From<AddOperation>
        + DifferentiableOperation<TracingContext<<C as Domain>::Constant, <C as Domain>::Operation>>
        + PartiallyEvaluatableOperation<TracingContext<<C as Domain>::Constant, <C as Domain>::Operation>>,
{
    value_and_grad(context, function, primals).map(|(_, gradient)| gradient)
}

/// Computes a scalar-output value, auxiliary outputs, and the reverse-mode gradient.
///
/// The differentiated value is the first element returned by `function`; it must be exactly one
/// rank-0 scalar array leaf. Auxiliary leaves are returned to the caller but seeded with zero
/// cotangents when the pullback is interpreted, so they do not contribute to the gradient.
///
/// This mirrors the semantics of a `has_aux` transform while keeping the Rust API explicit: the
/// primal value and auxiliary data are returned as `((value, aux), gradient)`.
#[allow(private_bounds)]
pub fn value_and_grad_with_aux<C, F, Input, Aux>(
    context: &C,
    function: F,
    primals: Input,
) -> Result<((<C as Domain>::Value, Aux), Input::To<<C as Domain>::Value>), DifferentiationError>
where
    C: Context,
    <C as Domain>::Constant: Value<Type = <C as Domain>::Type>,
    <C as Domain>::Value: BooleanLike,
    F: FnOnce(Input::To<NestedTracer<C>>) -> (NestedTracer<C>, Aux::To<NestedTracer<C>>),
    Input: Parameterized<
            <C as Domain>::Value,
            Family: ParameterizedFamily<NestedTracer<C>>,
            To<<C as Domain>::Value> = Input,
            ParameterStructure: Debug + PartialEq,
        >,
    Aux: Parameterized<
            <C as Domain>::Value,
            Family: ParameterizedFamily<NestedTracer<C>, To = Aux::To<NestedTracer<C>>>,
            ParameterStructure: Debug + PartialEq,
        >,
    (NestedTracer<C>, Aux::To<NestedTracer<C>>): Parameterized<
            NestedTracer<C>,
            To<<C as Domain>::Value> = (<C as Domain>::Value, Aux),
            Family: ParameterizedFamily<<C as Domain>::Value>,
        >,
    C: One<<C as Domain>::Value> + Zero<<C as Domain>::Value>,
    <C as Domain>::Type: DifferentiableType,
    <C as Domain>::Operation: Clone
        + InterpretableOperation<<C as Domain>::Value, C>
        + TransposableOperation<<C as Domain>::Constant, <C as Domain>::Operation>
        + MaybeWhile<<C as Domain>::Constant, <C as Domain>::Operation>
        + From<ZeroOperation<<C as Domain>::Type>>
        + From<AddOperation>
        + DifferentiableOperation<TracingContext<<C as Domain>::Constant, <C as Domain>::Operation>>
        + PartiallyEvaluatableOperation<TracingContext<<C as Domain>::Constant, <C as Domain>::Operation>>,
    Input::Family: ParameterizedFamily<<C as Domain>::Value>,
    Aux: Parameterized<<C as Domain>::Value, To<<C as Domain>::Value> = Aux>,
{
    let input_structure = primals.parameter_structure();
    let ((output, aux), pullback, residuals): ((<C as Domain>::Value, Aux), _, _) =
        context.vjp(|input| Ok(function(input)), primals)?;
    // Reverse mode only defines a gradient for scalar-output functions; reject non-scalar outputs before seeding
    // (see `DifferentiationError::NonScalarGradientOutput`).
    if !output.r#type().is_scalar() {
        return Err(DifferentiationError::NonScalarGradientOutput { output_type: output.r#type().to_string() });
    }
    // The flat pullback consumes `[output_cotangents ++ residuals]`. The traced output flattens as the scalar output
    // leaf followed by the auxiliary leaves, so seed the output leaf with a one cotangent and every auxiliary leaf with
    // a zero cotangent, then append the linearization-point residuals. Both the seeds and the replay go through the
    // domain itself: an eager domain constructs and interprets concrete values, while a staging domain stages into
    // its enclosing trace.
    let mut pullback_inputs = vec![context.one(output.r#type().as_ref())?];
    for value in Parameterized::<<C as Domain>::Value>::parameters(&aux) {
        pullback_inputs.push(context.zero(value.r#type().as_ref())?);
    }
    pullback_inputs.extend(residuals);
    let input_cotangents = pullback.interpret_in_context(context, pullback_inputs)?;
    let gradient = Input::To::<<C as Domain>::Value>::from_parameters(input_structure, input_cotangents)
        .map_err(ProgramError::from)?;
    Ok(((output, aux), gradient))
}

/// Computes the reverse-mode gradient and auxiliary outputs of a scalar-output function.
///
/// This is [`value_and_grad_with_aux`] with the primal scalar value discarded. The return order is
/// `(gradient, aux)`, matching the common use case where auxiliary outputs are diagnostics or
/// cached intermediates and the gradient remains the primary result.
#[allow(private_bounds)]
pub fn grad_with_aux<C, F, Input, Aux>(
    context: &C,
    function: F,
    primals: Input,
) -> Result<(Input::To<<C as Domain>::Value>, Aux), DifferentiationError>
where
    C: Context,
    <C as Domain>::Constant: Value<Type = <C as Domain>::Type>,
    <C as Domain>::Value: BooleanLike,
    F: FnOnce(Input::To<NestedTracer<C>>) -> (NestedTracer<C>, Aux::To<NestedTracer<C>>),
    Input: Parameterized<
            <C as Domain>::Value,
            Family: ParameterizedFamily<NestedTracer<C>>,
            To<<C as Domain>::Value> = Input,
            ParameterStructure: Debug + PartialEq,
        >,
    Aux: Parameterized<
            <C as Domain>::Value,
            Family: ParameterizedFamily<NestedTracer<C>, To = Aux::To<NestedTracer<C>>>,
            ParameterStructure: Debug + PartialEq,
        >,
    (NestedTracer<C>, Aux::To<NestedTracer<C>>): Parameterized<
            NestedTracer<C>,
            To<<C as Domain>::Value> = (<C as Domain>::Value, Aux),
            Family: ParameterizedFamily<<C as Domain>::Value>,
        >,
    C: One<<C as Domain>::Value> + Zero<<C as Domain>::Value>,
    <C as Domain>::Type: DifferentiableType,
    <C as Domain>::Operation: Clone
        + InterpretableOperation<<C as Domain>::Value, C>
        + TransposableOperation<<C as Domain>::Constant, <C as Domain>::Operation>
        + MaybeWhile<<C as Domain>::Constant, <C as Domain>::Operation>
        + From<ZeroOperation<<C as Domain>::Type>>
        + From<AddOperation>
        + DifferentiableOperation<TracingContext<<C as Domain>::Constant, <C as Domain>::Operation>>
        + PartiallyEvaluatableOperation<TracingContext<<C as Domain>::Constant, <C as Domain>::Operation>>,
    Input::Family: ParameterizedFamily<<C as Domain>::Value>,
    Aux: Parameterized<<C as Domain>::Value, To<<C as Domain>::Value> = Aux>,
{
    value_and_grad_with_aux(context, function, primals).map(|((_, aux), gradient)| (gradient, aux))
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use crate::contexts::StagingContext;
    use crate::programs::ProgramError;
    use crate::scalars::Scalar;
    use crate::tracing::{DomainTracer, DomainTracingContext};
    use crate::tracing_v2::{Differentiate, DifferentiationError};
    use crate::types::DataType;

    use super::*;
    use crate::contexts::EagerContext;
    use crate::operations::scalars::ScalarOperation;

    #[test]
    fn test_traced_value_and_grad_requires_input_leaves() {
        let context = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let empty_primals: Vec<DomainTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>> = Vec::new();

        let result = context
            .value_and_grad(|_inputs: Vec<_>| panic!("closure should not run without traced inputs"), empty_primals);

        assert!(matches!(
            result,
            Err(DifferentiationError::Program(ProgramError::InvalidInputCount { expected: 1, actual: 0 }))
        ));
    }

    #[test]
    fn test_traced_value_and_grad_rejects_mismatched_program_builders() {
        let context_a = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let context_b = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let primal_a = context_a.input(DataType::F64);
        let primal_b = context_b.input(DataType::F64);

        let result = context_a.value_and_grad(|inputs| inputs[0].clone() + inputs[1].clone(), vec![primal_a, primal_b]);

        assert!(matches!(result, Err(DifferentiationError::Program(ProgramError::MismatchedProgramBuilders))));
    }

    #[test]
    fn test_traced_value_and_grad_invokes_function_once() {
        let context = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let primal = context.input(DataType::F64);
        let calls = Cell::new(0);

        let (_value, gradient): (
            DomainTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>,
            Vec<DomainTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>>,
        ) = context
            .value_and_grad(
                |inputs| {
                    calls.set(calls.get() + 1);
                    inputs[0].clone() * inputs[0].clone()
                },
                vec![primal],
            )
            .unwrap();

        assert_eq!(calls.get(), 1);
        assert_eq!(gradient.len(), 1);
    }

    #[test]
    fn test_value_and_grad_with_aux_ignores_aux_cotangents() {
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();

        let ((value, aux), gradient): ((Scalar, (Scalar, Scalar)), (Scalar, Scalar)) = value_and_grad_with_aux(
            &domain,
            |(x, y)| {
                let value = x.clone() * y.clone();
                let aux = (x.clone() + y, x.clone() * x);
                (value, aux)
            },
            (Scalar::from(2.0), Scalar::from(3.0)),
        )
        .unwrap();

        assert_eq!(value, 6.0);
        assert_eq!(aux, (Scalar::from(5.0), Scalar::from(4.0)));
        assert_eq!(gradient, (Scalar::from(3.0), Scalar::from(2.0)));
    }

    #[test]
    fn test_grad_returns_only_the_gradient() {
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();

        let gradient: (Scalar, Scalar) =
            grad(&domain, |(x, y)| x.clone() * y.clone() + x, (Scalar::from(2.0), Scalar::from(3.0))).unwrap();

        assert_eq!(gradient, (Scalar::from(4.0), Scalar::from(2.0)));
    }

    #[test]
    fn test_grad_with_aux_returns_gradient_and_aux() {
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();

        let (gradient, aux): ((Scalar, Scalar), Scalar) =
            grad_with_aux(&domain, |(x, y)| (x.clone() * y.clone(), x + y), (Scalar::from(2.0), Scalar::from(3.0)))
                .unwrap();

        assert_eq!(gradient, (Scalar::from(3.0), Scalar::from(2.0)));
        assert_eq!(aux, 5.0);
    }
}
