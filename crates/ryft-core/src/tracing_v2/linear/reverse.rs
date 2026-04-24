use super::*;
use crate::parameters::ParameterError;
use crate::tracing_v2::DifferentiationError;

/// Traces `function` once and returns both its primal output and a reusable pushforward program.
///
/// [`jvp_program`] is the staged counterpart to [`crate::tracing_v2::jvp`]. Instead of immediately
/// applying a tangent input, it captures the Jacobian-vector product as a staged [`Program`] over
/// linear operations that can be replayed later on any tangent with the same parameter structure.
pub fn jvp_program<'engine, E, F, Input, Output, V>(
    engine: &'engine E,
    function: F,
    primals: Input,
) -> Result<(Output, Program<ArrayType, V, E::LinearOperation, Input, Output>), TracingError>
where
    E: Engine<Type = ArrayType, Value = V> + 'static,
    V: Traceable<ArrayType> + ZeroLike,
    Input: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<Tracer<'engine, E>>,
    Output::Family: ParameterizedFamily<Tracer<'engine, E>>,
    F: FnOnce(Input::To<Tracer<'engine, E>>) -> Result<Output::To<Tracer<'engine, E>>, TracingError>,
    E::LinearOperation: Clone
        + Operation<ArrayType>
        + LinearAddOperation<ArrayType, V>
        + LinearNegOperation<ArrayType, V>
        + LinearScaleOperation<ArrayType, V>,
    E::TracingOperation: InterpretableOperation<ArrayType, V>,
    E::TracingOperation: DifferentiableOperation<E>,
{
    let input_structure = primals.parameter_structure();
    let input_primals: Vec<V> = primals.into_parameters().collect();
    let reconstructed_primals = Input::from_parameters(input_structure, input_primals.iter().cloned())?;
    let (primal_output, program) = interpret_and_trace(engine, function, reconstructed_primals)?;
    Ok((
        primal_output,
        linearize_program::<Input, Output, V, E::TracingOperation, E::LinearOperation, E>(
            engine,
            &program,
            input_primals,
        )?,
    ))
}

/// Runs forward-mode differentiation inside an existing outer trace.
///
/// This is the traced-execution path for [`crate::tracing_v2::jvp`]. Rather than producing a
/// standalone linear program, it stages both the primal replay and the pushforward application into
/// the surrounding JIT trace so higher-order transforms can keep composing symbolically.
#[allow(private_bounds)]
pub(crate) fn jvp_traced<'engine, F, Input, Output, V, O, L, E>(
    function: F,
    primals: Input,
    tangents: Input,
) -> Result<(Output, Output), TracingError>
where
    V: Traceable<ArrayType> + ZeroLike + Parameterized<V, ParameterStructure = Placeholder>,
    Input: Parameterized<Tracer<'engine, E>, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
    Output: Parameterized<Tracer<'engine, E>, ParameterStructure: Clone>,
    O: Clone + Operation<ArrayType> + 'static,
    L: Clone + Operation<ArrayType> + 'static,
    E: Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized + 'static,
    Input::Family: ParameterizedFamily<V> + ParameterizedFamily<ArrayType>,
    Output::Family: ParameterizedFamily<V> + ParameterizedFamily<ArrayType>,
    Input::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'engine, E>> = Input>,
    Output::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'engine, E>> = Output>,
    O: InterpretableOperation<ArrayType, Linearized<Tracer<'engine, E>, LinearPrimitiveOperation<Tracer<'engine, E>>>>,
    LinearPrimitiveOperation<Tracer<'engine, E>>: CoreLinearReplayOperation<Tracer<'engine, E>>,
    F: FnOnce(Input) -> Result<Output, TracingError>,
{
    let primal_structure = primals.parameter_structure();
    let tangent_structure = tangents.parameter_structure();
    if primal_structure != tangent_structure {
        return Err(ParameterError::MismatchedParameterStructures {
            left_structure: format!("{primal_structure:?}"),
            right_structure: format!("{tangent_structure:?}"),
        }
        .into());
    }

    let input_structure = primal_structure;
    let traced_primals = primals.into_parameters().collect::<Vec<_>>();
    let traced_tangents = tangents.into_parameters().collect::<Vec<_>>();
    let Some(exemplar_traced_primal) = traced_primals.first() else {
        return Err(DifferentiationError::MissingTracedJvpInputLeaves.into());
    };
    let staged_input_types = Input::To::<ArrayType>::from_parameters(
        input_structure.clone(),
        traced_primals.iter().map(|traced_primal| traced_primal.r#type().into_owned()).collect::<Vec<_>>(),
    )?;
    let (primal_output_types, traced_program) =
        trace_flat_program_from_input_types::<Input::To<ArrayType>, Output::To<ArrayType>, V, O, L, E, _>(
            exemplar_traced_primal.engine,
            move |staged_input| function(staged_input),
            staged_input_types,
        )?;
    let output_structure = primal_output_types.parameter_structure();
    let tracing_builder = exemplar_traced_primal.builder.clone();
    let (traced_primal_output, pushforward) = linearize_traced_program::<V, O, L, E>(
        exemplar_traced_primal.engine,
        tracing_builder,
        &traced_program,
        traced_primals,
    )?;
    let traced_tangent_output = pushforward.interpret(traced_tangents)?;
    Ok((
        Output::from_parameters(output_structure.clone(), traced_primal_output)?,
        Output::from_parameters(output_structure, traced_tangent_output)?,
    ))
}

/// Returns the primal output together with a pullback produced by transposing the staged pushforward.
///
/// [`vjp`] is the reusable reverse-mode primitive in the public API. It traces the primal function,
/// builds the corresponding pushforward program, and then transposes that pushforward into a staged
/// pullback that maps output cotangents back to input cotangents.
#[allow(private_bounds)]
pub fn vjp<'engine, E, F, Input, Output, V>(
    engine: &'engine E,
    function: F,
    primals: Input,
) -> Result<(Output, Program<ArrayType, V, E::LinearOperation, Output, Input>), TracingError>
where
    E: Engine<Type = ArrayType, Value = V> + 'static,
    V: Traceable<ArrayType> + ZeroLike + OneLike,
    Input: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<Tracer<'engine, E>>,
    Output::Family: ParameterizedFamily<Tracer<'engine, E>>,
    F: FnOnce(Input::To<Tracer<'engine, E>>) -> Result<Output::To<Tracer<'engine, E>>, TracingError>,
    E::LinearOperation: Clone + Operation<ArrayType>,
    E::TracingOperation: InterpretableOperation<ArrayType, V>,
    E::TracingOperation: DifferentiableOperation<E>,
    E::LinearOperation: CoreLinearProgramOperation<V>
        + LinearAddOperation<ArrayType, V>
        + LinearNegOperation<ArrayType, V>
        + LinearScaleOperation<ArrayType, V>,
{
    let (output, pushforward) = jvp_program::<E, F, Input, Output, V>(engine, function, primals)?;
    let output_examples = output.parameters().cloned().collect::<Vec<_>>();
    let pullback = transpose_linear_program_with_output_examples(engine, &pushforward, output_examples.as_slice())?;
    Ok((output, pullback))
}

/// Dispatch trait shared by [`grad`] and [`value_and_grad`] so they can operate both on concrete
/// values and on already traced values.
///
/// The trait always produces `(value, gradient)`; [`grad`] is a thin wrapper that drops the primal
/// value, while [`value_and_grad`] exposes the full pair. This keeps the public reverse-mode API
/// compact while allowing concrete replay, traced replay, and batched replay to specialize
/// independently.
#[doc(hidden)]
pub trait ValueAndGradInvocationLeaf<E, Input>: Parameter + Sized
where
    E: Engine<Type = ArrayType>,
    Input: Parameterized<Self, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
{
    /// Primal scalar output value produced for the corresponding input regime.
    type Value;

    /// Traced input type expected by the user-provided function.
    type FunctionInput<'engine>
    where
        E: 'engine;

    /// Traced scalar output type expected from the user-provided function.
    type FunctionOutput<'engine>
    where
        E: 'engine;

    /// Invokes [`value_and_grad`] for one concrete leaf regime.
    fn invoke<'engine, F>(
        engine: &'engine E,
        function: F,
        primals: Input,
    ) -> Result<(Self::Value, Input), TracingError>
    where
        F: FnOnce(Self::FunctionInput<'engine>) -> Self::FunctionOutput<'engine>;
}

/// Concrete-value dispatch for [`value_and_grad`]: evaluates the user function via [`vjp`], checks
/// that the output is a rank-0 scalar array, and pulls back a unit seed to obtain both the primal
/// scalar output and its gradient.
impl<
    E,
    V: Value<ArrayType> + ZeroLike + OneLike + Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
    Input: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
> ValueAndGradInvocationLeaf<E, Input> for V
where
    E: Engine<Type = ArrayType, Value = V> + 'static,
    V: for<'engine> Parameterized<V, To<Tracer<'engine, E>> = Tracer<'engine, E>>,
    Input::Family: for<'engine> ParameterizedFamily<Tracer<'engine, E>>,
    V::Family: for<'engine> ParameterizedFamily<Tracer<'engine, E>>,
    E::TracingOperation: InterpretableOperation<ArrayType, V>,
    E::TracingOperation: DifferentiableOperation<E>,
    E::LinearOperation: CoreLinearProgramOperation<V>
        + LinearAddOperation<ArrayType, V>
        + LinearNegOperation<ArrayType, V>
        + LinearScaleOperation<ArrayType, V>,
{
    type Value = V;
    type FunctionInput<'engine>
        = Input::To<Tracer<'engine, E>>
    where
        E: 'engine;
    type FunctionOutput<'engine>
        = Tracer<'engine, E>
    where
        E: 'engine;

    fn invoke<'engine, F>(engine: &'engine E, function: F, primals: Input) -> Result<(Self::Value, Input), TracingError>
    where
        F: FnOnce(Self::FunctionInput<'engine>) -> Self::FunctionOutput<'engine>,
    {
        let (output, pullback): (V, Program<ArrayType, V, E::LinearOperation, V, Input>) =
            vjp(engine, |input| Ok(function(input)), primals)?;
        ensure_scalar_gradient_output_type(output.r#type().as_ref())?;
        let gradient = pullback.interpret(output.one_like())?;
        Ok((output, gradient))
    }
}

/// Already-traced dispatch for [`value_and_grad`]: replays the user function symbolically inside an
/// enclosing [`Tracer`] scope, linearizes, transposes, and stages both the forward output and the
/// backward gradient so they become part of the outer compiled program.
impl<
    'engine,
    E,
    V: Traceable<ArrayType> + ZeroLike + OneLike + Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input: Parameterized<Tracer<'engine, E>, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
> ValueAndGradInvocationLeaf<E, Input> for Tracer<'engine, E>
where
    E: Engine<Type = ArrayType, Value = V> + 'static,
    V: Parameterized<V, ParameterStructure = Placeholder>,
    V::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<Tracer<'engine, E>>,
    Input::Family: ParameterizedFamily<V> + ParameterizedFamily<ArrayType>,
    Input::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'engine, E>> = Input>,
    V::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'engine, E>> = Tracer<'engine, E>>,
    E::TracingOperation:
        InterpretableOperation<ArrayType, Linearized<Tracer<'engine, E>, LinearPrimitiveOperation<Tracer<'engine, E>>>>,
    E::LinearOperation: Clone + Operation<ArrayType> + 'static,
    LinearPrimitiveOperation<Tracer<'engine, E>>: CoreLinearProgramOperation<Tracer<'engine, E>>,
{
    type Value = Tracer<'engine, E>;
    type FunctionInput<'call>
        = Input
    where
        E: 'call;
    type FunctionOutput<'call>
        = Tracer<'engine, E>
    where
        E: 'call;

    fn invoke<'call, F>(_engine: &'call E, function: F, primals: Input) -> Result<(Self::Value, Input), TracingError>
    where
        F: FnOnce(Self::FunctionInput<'call>) -> Self::FunctionOutput<'call>,
    {
        let input_structure = primals.parameter_structure();
        let traced_primals = primals.into_parameters().collect::<Vec<_>>();
        if traced_primals.is_empty() {
            return Err(DifferentiationError::MissingTracedReverseModeInputLeaves.into());
        }
        let staged_input_types = Input::To::<ArrayType>::from_parameters(
            input_structure.clone(),
            traced_primals.iter().map(|traced_primal| traced_primal.r#type().into_owned()).collect::<Vec<_>>(),
        )?;
        let (_, traced_program) =
            trace_flat_program_from_input_types::<
                Input::To<ArrayType>,
                V::To<ArrayType>,
                V,
                E::TracingOperation,
                E::LinearOperation,
                E,
                _,
            >(traced_primals[0].engine, |staged_input| Ok(function(staged_input)), staged_input_types)?;
        let (traced_output, traced_gradient) =
            reverse_mode_scalar_traced_program::<V, E::TracingOperation, E::LinearOperation, E>(
                traced_primals[0].engine,
                traced_primals[0].builder.clone(),
                &traced_program,
                traced_primals,
            )?;
        Ok((traced_output, Input::from_parameters(input_structure, traced_gradient)?))
    }
}

/// Computes both the primal scalar output and its reverse-mode gradient.
///
/// This is the most direct reverse-mode API when the caller needs both the function value and the
/// gradient at the same primal point. The function must return exactly one rank-0 scalar array
/// leaf. Use [`vjp`] directly for vector-valued functions that need an explicit output cotangent.
#[allow(private_bounds, private_interfaces)]
pub fn value_and_grad<'engine, E, F, Input, Leaf>(
    engine: &'engine E,
    function: F,
    primals: Input,
) -> Result<(<Leaf as ValueAndGradInvocationLeaf<E, Input>>::Value, Input), TracingError>
where
    E: Engine<Type = ArrayType>,
    Leaf: ValueAndGradInvocationLeaf<E, Input>,
    Input: Parameterized<Leaf, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
    F: FnOnce(
        <Leaf as ValueAndGradInvocationLeaf<E, Input>>::FunctionInput<'engine>,
    ) -> <Leaf as ValueAndGradInvocationLeaf<E, Input>>::FunctionOutput<'engine>,
{
    Leaf::invoke(engine, function, primals)
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
pub fn value_and_grad_with_aux<'engine, E, F, Input, Aux, V>(
    engine: &'engine E,
    function: F,
    primals: Input,
) -> Result<((V, Aux), Input), TracingError>
where
    E: Engine<Type = ArrayType, Value = V> + 'static,
    V: Value<ArrayType>
        + ZeroLike
        + OneLike
        + Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
    V: for<'call> Parameterized<V, To<Tracer<'call, E>> = Tracer<'call, E>>,
    Input: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
    Aux: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
    Input::Family: for<'call> ParameterizedFamily<Tracer<'call, E>>,
    Aux::Family: ParameterizedFamily<Tracer<'engine, E>, To = Aux::To<Tracer<'engine, E>>>,
    V::Family: for<'call> ParameterizedFamily<Tracer<'call, E>, To = Tracer<'call, E>>,
    F: FnOnce(Input::To<Tracer<'engine, E>>) -> (Tracer<'engine, E>, Aux::To<Tracer<'engine, E>>),
    E::TracingOperation: InterpretableOperation<ArrayType, V>,
    E::TracingOperation: DifferentiableOperation<E>,
    E::LinearOperation: CoreLinearProgramOperation<V>
        + LinearAddOperation<ArrayType, V>
        + LinearNegOperation<ArrayType, V>
        + LinearScaleOperation<ArrayType, V>,
{
    let ((output, aux), pullback): ((V, Aux), Program<ArrayType, V, E::LinearOperation, (V, Aux), Input>) =
        vjp(engine, |input| Ok(function(input)), primals)?;
    ensure_scalar_gradient_output_type(output.r#type().as_ref())?;
    let aux_zeros = Aux::from_parameters(aux.parameter_structure(), aux.parameters().map(ZeroLike::zero_like))?;
    let gradient = pullback.interpret((output.one_like(), aux_zeros))?;
    Ok(((output, aux), gradient))
}

/// Computes the reverse-mode gradient of a scalar-output function.
///
/// [`grad`] is just [`value_and_grad`] with the primal result discarded, but it is the most common
/// user-facing reverse-mode entry point and therefore gets its own dedicated wrapper. The function
/// must return exactly one rank-0 scalar array leaf. Use [`vjp`] directly for vector-valued functions
/// that need an explicit output cotangent.
#[allow(private_bounds, private_interfaces)]
pub fn grad<'engine, E, F, Input, Leaf>(engine: &'engine E, function: F, primals: Input) -> Result<Input, TracingError>
where
    E: Engine<Type = ArrayType>,
    Leaf: ValueAndGradInvocationLeaf<E, Input>,
    Input: Parameterized<Leaf, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
    F: FnOnce(
        <Leaf as ValueAndGradInvocationLeaf<E, Input>>::FunctionInput<'engine>,
    ) -> <Leaf as ValueAndGradInvocationLeaf<E, Input>>::FunctionOutput<'engine>,
{
    Leaf::invoke(engine, function, primals).map(|(_, gradient)| gradient)
}

/// Computes the reverse-mode gradient and auxiliary outputs of a scalar-output function.
///
/// This is [`value_and_grad_with_aux`] with the primal scalar value discarded. The return order is
/// `(gradient, aux)`, matching the common use case where auxiliary outputs are diagnostics or
/// cached intermediates and the gradient remains the primary result.
#[allow(private_bounds)]
pub fn grad_with_aux<'engine, E, F, Input, Aux, V>(
    engine: &'engine E,
    function: F,
    primals: Input,
) -> Result<(Input, Aux), TracingError>
where
    E: Engine<Type = ArrayType, Value = V> + 'static,
    V: Value<ArrayType>
        + ZeroLike
        + OneLike
        + Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
    V: for<'call> Parameterized<V, To<Tracer<'call, E>> = Tracer<'call, E>>,
    Input: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
    Aux: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
    Input::Family: for<'call> ParameterizedFamily<Tracer<'call, E>>,
    Aux::Family: ParameterizedFamily<Tracer<'engine, E>, To = Aux::To<Tracer<'engine, E>>>,
    V::Family: for<'call> ParameterizedFamily<Tracer<'call, E>, To = Tracer<'call, E>>,
    F: FnOnce(Input::To<Tracer<'engine, E>>) -> (Tracer<'engine, E>, Aux::To<Tracer<'engine, E>>),
    E::TracingOperation: InterpretableOperation<ArrayType, V>,
    E::TracingOperation: DifferentiableOperation<E>,
    E::LinearOperation: CoreLinearProgramOperation<V>
        + LinearAddOperation<ArrayType, V>
        + LinearNegOperation<ArrayType, V>
        + LinearScaleOperation<ArrayType, V>,
{
    value_and_grad_with_aux(engine, function, primals).map(|((_, aux), gradient)| (gradient, aux))
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;

    use crate::tracing::TracingError;
    use crate::tracing_v2::engine::ArrayScalarEngine;
    use crate::tracing_v2::operations::matrix::ndarray_support::Array2Engine;
    use crate::tracing_v2::{DifferentiationError, Tracer};

    use super::*;

    #[test]
    fn test_jvp_traced_requires_input_leaves() {
        let empty_primals: Vec<Tracer<'_, ArrayScalarEngine<f64>>> = Vec::new();
        let empty_tangents: Vec<Tracer<'_, ArrayScalarEngine<f64>>> = Vec::new();

        let result =
            jvp_traced(|inputs: Vec<Tracer<'_, ArrayScalarEngine<f64>>>| Ok(inputs), empty_primals, empty_tangents);

        assert!(matches!(
            result,
            Err(TracingError::Differentiation(DifferentiationError::MissingTracedJvpInputLeaves))
        ));
    }

    #[test]
    fn test_traced_value_and_grad_requires_input_leaves() {
        let engine = ArrayScalarEngine::<f64>::new();
        let empty_primals: Vec<Tracer<'_, ArrayScalarEngine<f64>>> = Vec::new();

        let result = <Tracer<'_, ArrayScalarEngine<f64>> as ValueAndGradInvocationLeaf<
            ArrayScalarEngine<f64>,
            Vec<Tracer<'_, ArrayScalarEngine<f64>>>,
        >>::invoke(
            &engine, |_inputs| panic!("closure should not run without traced inputs"), empty_primals
        );

        assert!(matches!(
            result,
            Err(TracingError::Differentiation(DifferentiationError::MissingTracedReverseModeInputLeaves))
        ));
    }

    #[test]
    fn test_grad_rejects_non_scalar_array_output() {
        let engine = Array2Engine::<f64>::new();

        let result = grad(&engine, |input| input, arr2(&[[1.0, 2.0], [3.0, 4.0]]));

        assert!(matches!(
            result,
            Err(TracingError::Differentiation(DifferentiationError::NonScalarGradientOutput { output_type }))
                if output_type.rank() == 2
        ));
    }

    #[test]
    fn test_value_and_grad_with_aux_ignores_aux_cotangents() {
        let engine = ArrayScalarEngine::<f64>::new();

        let ((value, aux), gradient): ((f64, (f64, f64)), (f64, f64)) = value_and_grad_with_aux(
            &engine,
            |(x, y)| {
                let value = x.clone() * y.clone();
                let aux = (x.clone() + y, x.clone() * x);
                (value, aux)
            },
            (2.0f64, 3.0f64),
        )
        .unwrap();

        assert_eq!(value, 6.0);
        assert_eq!(aux, (5.0, 4.0));
        assert_eq!(gradient, (3.0, 2.0));
    }

    #[test]
    fn test_grad_with_aux_returns_gradient_and_aux() {
        let engine = ArrayScalarEngine::<f64>::new();

        let (gradient, aux): ((f64, f64), f64) =
            grad_with_aux(&engine, |(x, y)| (x.clone() * y.clone(), x + y), (2.0f64, 3.0f64)).unwrap();

        assert_eq!(gradient, (3.0, 2.0));
        assert_eq!(aux, 5.0);
    }
}
