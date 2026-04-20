//! Reverse-mode transforms built on top of staged linear programs.
//!
//! This module takes the IR-level building blocks from the rest of [`super`] and assembles the
//! user-facing reverse-mode APIs. The broad flow is:
//!
//! 1. trace a primal program,
//! 2. linearize it into a pushforward,
//! 3. transpose that pushforward into a pullback, and
//! 4. seed the pullback appropriately to obtain gradients or value-and-gradient pairs.
//!
//! The concrete, traced, and batched variants all share that structure; they only differ in where
//! the final replay happens.

use super::*;

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
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<Tracer<'engine, E>>,
    Output::Family: ParameterizedFamily<Tracer<'engine, E>>,
    F: FnOnce(Input::To<Tracer<'engine, E>>) -> Result<Output::To<Tracer<'engine, E>>, TracingError>,
    E::LinearOperation: Clone + Operation<ArrayType>,
    E::TracingOperation: InterpretableOperation<ArrayType, V>,
    E::TracingOperation: DifferentiableOperation<
            ArrayType,
            V,
            LinearTerm<ArrayType, V, E::LinearOperation>,
            E::TracingOperation,
            E::LinearOperation,
        >,
{
    let input_structure = primals.parameter_structure();
    let input_primals: Vec<V> = primals.into_parameters().collect();
    let reconstructed_primals = Input::from_parameters(input_structure, input_primals.iter().cloned())?;
    let (primal_output, program) = interpret_and_trace(engine, function, reconstructed_primals)?;
    Ok((
        primal_output,
        linearize_program::<Input, Output, V, E::TracingOperation, E::LinearOperation>(
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
    Input: Parameterized<Tracer<'engine, E>, ParameterStructure: Clone + PartialEq>,
    Output: Parameterized<Tracer<'engine, E>, ParameterStructure: Clone>,
    O: Clone + Operation<ArrayType> + 'static,
    L: Clone + Operation<ArrayType> + 'static,
    E: Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized + 'static,
    Input::Family: ParameterizedFamily<V> + ParameterizedFamily<ArrayType>,
    Output::Family: ParameterizedFamily<V> + ParameterizedFamily<ArrayType>,
    Input::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'engine, E>> = Input>,
    Output::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'engine, E>> = Output>,
    O: InterpretableOperation<
            ArrayType,
            Linearized<Tracer<'engine, E>, LinearPrimitiveOperation<ArrayType, Tracer<'engine, E>>>,
        >,
    LinearPrimitiveOperation<ArrayType, Tracer<'engine, E>>: CoreLinearReplayOperation<Tracer<'engine, E>>,
    F: FnOnce(Input) -> Result<Output, TracingError>,
{
    if primals.parameter_structure() != tangents.parameter_structure() {
        return Err(TracingError::MismatchedParameterStructure);
    }

    let input_structure = primals.parameter_structure();
    let traced_primals = primals.into_parameters().collect::<Vec<_>>();
    let traced_tangents = tangents.into_parameters().collect::<Vec<_>>();
    let staged_input_types = Input::To::<ArrayType>::from_parameters(
        input_structure.clone(),
        traced_primals.iter().map(|traced_primal| traced_primal.r#type().into_owned()).collect::<Vec<_>>(),
    )?;
    let (primal_output_types, traced_program) =
        trace_flat_program_from_input_types::<Input::To<ArrayType>, Output::To<ArrayType>, V, O, L, E, _>(
            traced_primals.first().ok_or(TracingError::EmptyParameterizedValue)?.engine,
            move |staged_input| function(staged_input),
            staged_input_types,
        )?;
    let output_structure = primal_output_types.parameter_structure();
    let (traced_primal_output, pushforward) = linearize_traced_program::<V, O, L, E>(&traced_program, traced_primals)?;
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
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<Tracer<'engine, E>>,
    Output::Family: ParameterizedFamily<Tracer<'engine, E>>,
    F: FnOnce(Input::To<Tracer<'engine, E>>) -> Result<Output::To<Tracer<'engine, E>>, TracingError>,
    E::LinearOperation: Clone + Operation<ArrayType>,
    E::TracingOperation: InterpretableOperation<ArrayType, V>,
    E::TracingOperation: DifferentiableOperation<
            ArrayType,
            V,
            LinearTerm<ArrayType, V, E::LinearOperation>,
            E::TracingOperation,
            E::LinearOperation,
        >,
    E::LinearOperation: CoreLinearProgramOperation<V> + LinearAddOperation<ArrayType, V>,
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
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq>,
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

/// Concrete-value dispatch for [`value_and_grad`]: evaluates the user function via [`vjp`] and
/// pulls back a unit seed to obtain both the primal scalar output and its gradient.
impl<
    E,
    V: Value<ArrayType> + ZeroLike + OneLike + Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
> ValueAndGradInvocationLeaf<E, Input> for V
where
    E: Engine<Type = ArrayType, Value = V> + 'static,
    V: for<'engine> Parameterized<V, To<Tracer<'engine, E>> = Tracer<'engine, E>>,
    Input::Family: for<'engine> ParameterizedFamily<Tracer<'engine, E>>,
    V::Family: for<'engine> ParameterizedFamily<Tracer<'engine, E>>,
    E::TracingOperation: InterpretableOperation<ArrayType, V>,
    E::TracingOperation: DifferentiableOperation<
            ArrayType,
            V,
            LinearTerm<ArrayType, V, E::LinearOperation>,
            E::TracingOperation,
            E::LinearOperation,
        >,
    E::LinearOperation: CoreLinearProgramOperation<V> + LinearAddOperation<ArrayType, V>,
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
    Input: Parameterized<Tracer<'engine, E>, ParameterStructure: Clone + PartialEq>,
> ValueAndGradInvocationLeaf<E, Input> for Tracer<'engine, E>
where
    E: Engine<Type = ArrayType, Value = V> + 'static,
    V: Parameterized<V, ParameterStructure = Placeholder>,
    V::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<Tracer<'engine, E>>,
    Input::Family: ParameterizedFamily<V> + ParameterizedFamily<ArrayType>,
    Input::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'engine, E>> = Input>,
    V::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'engine, E>> = Tracer<'engine, E>>,
    E::TracingOperation: InterpretableOperation<
            ArrayType,
            Linearized<Tracer<'engine, E>, LinearPrimitiveOperation<ArrayType, Tracer<'engine, E>>>,
        >,
    E::LinearOperation: Clone + Operation<ArrayType> + 'static,
    LinearPrimitiveOperation<ArrayType, Tracer<'engine, E>>: CoreLinearProgramOperation<Tracer<'engine, E>>,
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
        let staged_input_types = Input::To::<ArrayType>::from_parameters(
            input_structure.clone(),
            traced_primals.iter().map(|traced_primal| traced_primal.r#type().into_owned()).collect::<Vec<_>>(),
        )?;
        let (_, traced_program) = trace_flat_program_from_input_types::<
            Input::To<ArrayType>,
            V::To<ArrayType>,
            V,
            E::TracingOperation,
            E::LinearOperation,
            E,
            _,
        >(
            traced_primals.first().ok_or(TracingError::EmptyParameterizedValue)?.engine,
            |staged_input| Ok(function(staged_input)),
            staged_input_types,
        )?;
        let (traced_output, traced_gradient) =
            reverse_mode_scalar_traced_program::<V, E::TracingOperation, E::LinearOperation, E>(
                traced_primals.first().ok_or(TracingError::EmptyParameterizedValue)?.engine,
                &traced_program,
                traced_primals,
            )?;
        Ok((traced_output, Input::from_parameters(input_structure, traced_gradient)?))
    }
}

/// Batched dispatch for [`value_and_grad`], enabling standalone
/// `vmap(|x| value_and_grad(f, x), inputs)` -- computing per-element function values and gradients
/// over a batch without requiring an outer [`interpret_and_trace`] wrapper.
///
/// Uses a trace-once strategy for [`Batch`]: the user function is traced once to a [`Program`],
/// and a second [`Program`] that produces `(V, Input::To<V>)` per lane is compiled via [`interpret_and_trace`].
/// Values and gradients are collected per lane and stacked separately.
impl<
    E,
    V: Traceable<ArrayType> + ZeroLike + OneLike + 'static,
    Input: Parameterized<Batch<V>, ParameterStructure: Clone + PartialEq>,
> ValueAndGradInvocationLeaf<E, Input> for Batch<V>
where
    E: Engine<Type = ArrayType, Value = V> + 'static,
    V: for<'engine> Parameterized<
            V,
            ParameterStructure = Placeholder,
            To<
                Tracer<'engine, dyn Engine<
                            Type = ArrayType,
                            Value = V,
                            TracingOperation = E::TracingOperation,
                            LinearOperation = E::LinearOperation,
                        >>,
            > = Tracer<'engine, dyn Engine<
                        Type = ArrayType,
                        Value = V,
                        TracingOperation = E::TracingOperation,
                        LinearOperation = E::LinearOperation,
                    >>,
        >,
    V::Family: for<'engine> ParameterizedFamily<
            Tracer<'engine, dyn Engine<
                        Type = ArrayType,
                        Value = V,
                        TracingOperation = E::TracingOperation,
                        LinearOperation = E::LinearOperation,
                    >>,
        >,
    Vec<V>: for<'engine> Parameterized<
            V,
            ParameterStructure = Vec<Placeholder>,
            To<
                Tracer<'engine, dyn Engine<
                            Type = ArrayType,
                            Value = V,
                            TracingOperation = E::TracingOperation,
                            LinearOperation = E::LinearOperation,
                        >>,
            > = Vec<
                Tracer<'engine, dyn Engine<
                            Type = ArrayType,
                            Value = V,
                            TracingOperation = E::TracingOperation,
                            LinearOperation = E::LinearOperation,
                        >>,
            >,
        >,
    <Vec<V> as Parameterized<V>>::Family: for<'engine> ParameterizedFamily<
            Tracer<'engine, dyn Engine<
                        Type = ArrayType,
                        Value = V,
                        TracingOperation = E::TracingOperation,
                        LinearOperation = E::LinearOperation,
                    >>,
        >,
    Input::Family: ParameterizedFamily<V>
        + for<'engine> ParameterizedFamily<
            Tracer<'engine, dyn Engine<
                        Type = ArrayType,
                        Value = V,
                        TracingOperation = E::TracingOperation,
                        LinearOperation = E::LinearOperation,
                    >>,
        >,
    Input::To<V>: Clone
        + for<'engine> Parameterized<
            V,
            ParameterStructure: Clone + PartialEq,
            To<Batch<V>> = Input,
            To<
                Tracer<'engine, dyn Engine<
                            Type = ArrayType,
                            Value = V,
                            TracingOperation = E::TracingOperation,
                            LinearOperation = E::LinearOperation,
                        >>,
            > = Input::To<
                Tracer<'engine, dyn Engine<
                            Type = ArrayType,
                            Value = V,
                            TracingOperation = E::TracingOperation,
                            LinearOperation = E::LinearOperation,
                        >>,
            >,
        >,
    E::TracingOperation: Clone + Operation<ArrayType>,
    E::LinearOperation: Operation<ArrayType>,
    E::TracingOperation: InterpretableOperation<ArrayType, V>,
    E::TracingOperation: for<'engine> InterpretableOperation<
            ArrayType,
            Linearized<
                Tracer<'engine, dyn Engine<
                            Type = ArrayType,
                            Value = V,
                            TracingOperation = E::TracingOperation,
                            LinearOperation = E::LinearOperation,
                        >>,
                LinearPrimitiveOperation<
                    ArrayType,
                    Tracer<'engine, dyn Engine<
                                Type = ArrayType,
                                Value = V,
                                TracingOperation = E::TracingOperation,
                                LinearOperation = E::LinearOperation,
                            >>,
                >,
            >,
        >,
    for<'engine> LinearPrimitiveOperation<
        ArrayType,
        Tracer<'engine, dyn Engine<
                    Type = ArrayType,
                    Value = V,
                    TracingOperation = E::TracingOperation,
                    LinearOperation = E::LinearOperation,
                >>,
    >: CoreLinearProgramOperation<
        Tracer<'engine, dyn Engine<
                    Type = ArrayType,
                    Value = V,
                    TracingOperation = E::TracingOperation,
                    LinearOperation = E::LinearOperation,
                >>,
    >,
{
    type Value = Batch<V>;
    type FunctionInput<'engine>
        = Input::To<
        Tracer<'engine, dyn Engine<
                    Type = ArrayType,
                    Value = V,
                    TracingOperation = E::TracingOperation,
                    LinearOperation = E::LinearOperation,
                >>,
    >
    where
        E: 'engine;
    type FunctionOutput<'engine>
        = Tracer<'engine, dyn Engine<
                Type = ArrayType,
                Value = V,
                TracingOperation = E::TracingOperation,
                LinearOperation = E::LinearOperation,
            >>
    where
        E: 'engine;

    fn invoke<'engine, F>(
        engine: &'engine E,
        function: F,
        primals: Input,
    ) -> Result<(Self::Value, Input), TracingError>
    where
        F: FnOnce(Self::FunctionInput<'engine>) -> Self::FunctionOutput<'engine>,
    {
        let erased_engine: &dyn Engine<
            Type = ArrayType,
            Value = V,
            TracingOperation = E::TracingOperation,
            LinearOperation = E::LinearOperation,
        > = engine;

        let lane_primals: Vec<Input::To<V>> = unstack(primals)?;
        if lane_primals.is_empty() {
            return Err(TracingError::EmptyBatch);
        }

        let lane0 = lane_primals[0].clone();
        let input_structure = lane0.parameter_structure();
        let parameter_count = input_structure.parameter_count();
        let lane0_flat: Vec<V> = lane0.into_parameters().collect();

        // Trace the user function once at lane 0 primals, consuming the FnOnce closure.
        let (_, traced_program): (V, Program<ArrayType, V, E::TracingOperation, Input::To<V>, V>) =
            interpret_and_trace(erased_engine, |staged_input| Ok(function(staged_input)), lane_primals[0].clone())?;

        // Reshape the program to flat Vec<V> inputs and outputs for the JIT compilation step.
        let flat_program = Program {
            atoms: traced_program.atoms.clone(),
            input_ids: traced_program.input_ids.clone(),
            output_ids: traced_program.output_ids.clone(),
            instructions: traced_program.instructions.clone(),
            input_structure: flat_leaf_parameter_structure(parameter_count),
            output_structure: flat_leaf_parameter_structure(1),
            marker: std::marker::PhantomData,
        }
        .with_folded_constants()?
        .simplified()?;

        // Compile both the forward evaluation and gradient into a reusable program.
        let (_, compiled_vg): (Vec<V>, Program<ArrayType, V, E::TracingOperation, Vec<V>, Vec<V>>) =
            interpret_and_trace(
                erased_engine,
                |jit_primals: Vec<
                    Tracer<'engine, dyn Engine<
                                Type = ArrayType,
                                Value = V,
                                TracingOperation = E::TracingOperation,
                                LinearOperation = E::LinearOperation,
                            >>,
                >| {
                    let (output, gradient) = reverse_mode_scalar_traced_program::<
                        V,
                        E::TracingOperation,
                        E::LinearOperation,
                        dyn Engine<
                                Type = ArrayType,
                                Value = V,
                                TracingOperation = E::TracingOperation,
                                LinearOperation = E::LinearOperation,
                            >>(erased_engine, &flat_program, jit_primals)?;
                    let mut result = Vec::with_capacity(1 + gradient.len());
                    result.push(output);
                    result.extend(gradient);
                    Ok(result)
                },
                lane0_flat,
            )?;

        // Apply per-lane and split into (value, gradient).
        let mut lane_values = Vec::with_capacity(lane_primals.len());
        let mut lane_grads = Vec::with_capacity(lane_primals.len());
        for lane in lane_primals {
            let flat: Vec<V> = lane.into_parameters().collect();
            let flat_result = compiled_vg.interpret(flat)?;
            let (value, grad_flat) = flat_result.split_first().ok_or(TracingError::EmptyParameterizedValue)?;
            lane_values.push(value.clone());
            lane_grads.push(
                Input::To::<V>::from_parameters(input_structure.clone(), grad_flat.to_vec())
                    .map_err(TracingError::from)?,
            );
        }

        let batched_values = Batch::new(lane_values);
        let batched_grads = stack(lane_grads)?;
        Ok((batched_values, batched_grads))
    }
}

/// Computes both the primal scalar output and its reverse-mode gradient.
///
/// This is the most direct reverse-mode API when the caller needs both the function value and the
/// gradient at the same primal point.
#[allow(private_bounds, private_interfaces)]
pub fn value_and_grad<'engine, E, F, Input, Leaf>(
    engine: &'engine E,
    function: F,
    primals: Input,
) -> Result<(<Leaf as ValueAndGradInvocationLeaf<E, Input>>::Value, Input), TracingError>
where
    E: Engine<Type = ArrayType>,
    Leaf: ValueAndGradInvocationLeaf<E, Input>,
    Input: Parameterized<Leaf, ParameterStructure: Clone + PartialEq>,
    F: FnOnce(
        <Leaf as ValueAndGradInvocationLeaf<E, Input>>::FunctionInput<'engine>,
    ) -> <Leaf as ValueAndGradInvocationLeaf<E, Input>>::FunctionOutput<'engine>,
{
    Leaf::invoke(engine, function, primals)
}

/// Computes the reverse-mode gradient of a scalar-output function.
///
/// [`grad`] is just [`value_and_grad`] with the primal result discarded, but it is the most common
/// user-facing reverse-mode entry point and therefore gets its own dedicated wrapper.
#[allow(private_bounds, private_interfaces)]
pub fn grad<'engine, E, F, Input, Leaf>(engine: &'engine E, function: F, primals: Input) -> Result<Input, TracingError>
where
    E: Engine<Type = ArrayType>,
    Leaf: ValueAndGradInvocationLeaf<E, Input>,
    Input: Parameterized<Leaf, ParameterStructure: Clone + PartialEq>,
    F: FnOnce(
        <Leaf as ValueAndGradInvocationLeaf<E, Input>>::FunctionInput<'engine>,
    ) -> <Leaf as ValueAndGradInvocationLeaf<E, Input>>::FunctionOutput<'engine>,
{
    Leaf::invoke(engine, function, primals).map(|(_, gradient)| gradient)
}
