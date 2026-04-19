use super::*;

pub fn jvp_program<E, F, Input, Output, V>(
    engine: &E,
    function: F,
    primals: Input,
) -> Result<(Output, LinearProgram<ArrayType, V, Input, Output, E::LinearOperation>), TraceError>
where
    E: Engine<Type = ArrayType, Value = V>,
    V: Traceable<ArrayType> + ZeroLike,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
    Output::Family: ParameterizedFamily<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
    F: FnOnce(
        Input::To<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
    ) -> Result<Output::To<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>, TraceError>,
    E::LinearOperation: Clone + Op<ArrayType>,
    E::TracingOperation: InterpretableOp<ArrayType, V>,
    E::TracingOperation: DifferentiableOp<
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
    let (primal_output, program) = trace_program(engine, function, reconstructed_primals)?;
    Ok((
        primal_output,
        linearize_program::<Input, Output, V, E::TracingOperation, E::LinearOperation>(
            engine,
            &program,
            input_primals,
        )?,
    ))
}

#[allow(private_bounds)]
pub(crate) fn jvp_traced<F, Input, Output, V, O, L>(
    function: F,
    primals: Input,
    tangents: Input,
) -> Result<(Output, Output), TraceError>
where
    V: Traceable<ArrayType> + ZeroLike + Parameterized<V, ParameterStructure = Placeholder>,
    Input: Parameterized<JitTracer<ArrayType, V, O, L>, ParameterStructure: Clone + PartialEq>,
    Output: Parameterized<JitTracer<ArrayType, V, O, L>, ParameterStructure: Clone>,
    O: Clone + Op<ArrayType> + 'static,
    L: Clone + 'static,
    Input::Family: ParameterizedFamily<V> + ParameterizedFamily<ArrayType>,
    Output::Family: ParameterizedFamily<V> + ParameterizedFamily<ArrayType>,
    Input::To<ArrayType>: Parameterized<ArrayType, To<JitTracer<ArrayType, V, O, L>> = Input>,
    Output::To<ArrayType>: Parameterized<ArrayType, To<JitTracer<ArrayType, V, O, L>> = Output>,
    O: InterpretableOp<
            ArrayType,
            Linearized<JitTracer<ArrayType, V, O, L>, LinearProgramOpRef<JitTracer<ArrayType, V, O, L>>>,
        >,
    LinearProgramOpRef<JitTracer<ArrayType, V, O, L>>: CoreLinearReplayOp<JitTracer<ArrayType, V, O, L>>,
    F: FnOnce(Input) -> Result<Output, TraceError>,
{
    if primals.parameter_structure() != tangents.parameter_structure() {
        return Err(TraceError::MismatchedParameterStructure);
    }

    let input_structure = primals.parameter_structure();
    let traced_primals = primals.into_parameters().collect::<Vec<_>>();
    let traced_tangents = tangents.into_parameters().collect::<Vec<_>>();
    let staged_input_types = Input::To::<ArrayType>::from_parameters(
        input_structure.clone(),
        traced_primals.iter().map(|traced_primal| traced_primal.tpe().into_owned()).collect::<Vec<_>>(),
    )?;
    let (primal_output_types, traced_program) =
        trace_flat_program_from_input_types::<Input::To<ArrayType>, Output::To<ArrayType>, V, O, L, _>(
            move |staged_input| function(staged_input),
            &traced_primals,
            staged_input_types,
        )?;
    let output_structure = primal_output_types.parameter_structure();
    let (traced_primal_output, pushforward) = linearize_traced_program::<V, O, L>(&traced_program, traced_primals)?;
    let traced_tangent_output = pushforward.call(traced_tangents)?;
    Ok((
        Output::from_parameters(output_structure.clone(), traced_primal_output)?,
        Output::from_parameters(output_structure, traced_tangent_output)?,
    ))
}

/// Returns the primal output together with a pullback produced by transposing the staged pushforward.
#[allow(private_bounds)]
pub fn vjp<E, F, Input, Output, V>(
    engine: &E,
    function: F,
    primals: Input,
) -> Result<(Output, LinearProgram<ArrayType, V, Output, Input, E::LinearOperation>), TraceError>
where
    E: Engine<Type = ArrayType, Value = V>,
    V: Traceable<ArrayType> + ZeroLike + OneLike,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
    Output::Family: ParameterizedFamily<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
    F: FnOnce(
        Input::To<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
    ) -> Result<Output::To<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>, TraceError>,
    E::LinearOperation: Clone + Op<ArrayType>,
    E::TracingOperation: InterpretableOp<ArrayType, V>,
    E::TracingOperation: DifferentiableOp<
            ArrayType,
            V,
            LinearTerm<ArrayType, V, E::LinearOperation>,
            E::TracingOperation,
            E::LinearOperation,
        >,
    E::LinearOperation: CoreLinearProgramOp<V> + LinearAddOperation<ArrayType, V>,
{
    let (output, pushforward) = jvp_program::<E, F, Input, Output, V>(engine, function, primals)?;
    let output_examples = output.parameters().cloned().collect::<Vec<_>>();
    let pullback = transpose_linear_program_with_output_examples(&pushforward, output_examples.as_slice())?;
    Ok((output, pullback))
}

/// Dispatch trait shared by [`grad`] and [`value_and_grad`] so they can operate both on concrete
/// values and on already traced values.
///
/// The trait always produces `(value, gradient)`; [`grad`] is a thin wrapper that drops the primal
/// value, while [`value_and_grad`] exposes the full pair.
#[doc(hidden)]
pub trait ValueAndGradInvocationLeaf<E, Input>: Parameter + Sized
where
    E: Engine<Type = ArrayType>,
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq>,
{
    /// Primal scalar output value produced for the corresponding input regime.
    type Value;

    /// Traced input type expected by the user-provided function.
    type FunctionInput;

    /// Traced scalar output type expected from the user-provided function.
    type FunctionOutput;

    /// Invokes [`value_and_grad`] for one concrete leaf regime.
    fn invoke<F>(engine: &E, function: F, primals: Input) -> Result<(Self::Value, Input), TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> Self::FunctionOutput;
}

/// Concrete-value dispatch for [`value_and_grad`]: evaluates the user function via [`vjp`] and
/// pulls back a unit seed to obtain both the primal scalar output and its gradient.
impl<
    E,
    V: Value<ArrayType> + ZeroLike + OneLike + Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq>,
> ValueAndGradInvocationLeaf<E, Input> for V
where
    E: Engine<Type = ArrayType, Value = V>,
    V: Parameterized<
            V,
            To<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>> = JitTracer<
                ArrayType,
                V,
                E::TracingOperation,
                E::LinearOperation,
            >,
        >,
    Input::Family: ParameterizedFamily<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
    V::Family: ParameterizedFamily<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
    E::TracingOperation: InterpretableOp<ArrayType, V>,
    E::TracingOperation: DifferentiableOp<
            ArrayType,
            V,
            LinearTerm<ArrayType, V, E::LinearOperation>,
            E::TracingOperation,
            E::LinearOperation,
        >,
    E::LinearOperation: CoreLinearProgramOp<V> + LinearAddOperation<ArrayType, V>,
{
    type Value = V;
    type FunctionInput = Input::To<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>;
    type FunctionOutput = JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>;

    fn invoke<F>(engine: &E, function: F, primals: Input) -> Result<(Self::Value, Input), TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> Self::FunctionOutput,
    {
        let (output, pullback): (V, LinearProgram<ArrayType, V, V, Input, E::LinearOperation>) =
            vjp(engine, |input| Ok(function(input)), primals)?;
        let gradient = pullback.call(output.one_like())?;
        Ok((output, gradient))
    }
}

/// Already-traced dispatch for [`value_and_grad`]: replays the user function symbolically inside an
/// enclosing [`JitTracer`] scope, linearizes, transposes, and stages both the forward output and the
/// backward gradient so they become part of the outer compiled graph.
impl<
    E,
    V: Traceable<ArrayType> + ZeroLike + OneLike + Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq>,
    O: Clone + Op<ArrayType> + 'static,
    L: Clone + Op<ArrayType> + 'static,
> ValueAndGradInvocationLeaf<E, Input> for JitTracer<ArrayType, V, O, L>
where
    E: Engine<Type = ArrayType, TracingOperation = O, LinearOperation = L>,
    V: Parameterized<V, ParameterStructure = Placeholder>,
    V::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<JitTracer<ArrayType, V, O, L>>,
    Input::Family: ParameterizedFamily<V> + ParameterizedFamily<ArrayType>,
    Input::To<ArrayType>: Parameterized<ArrayType, To<JitTracer<ArrayType, V, O, L>> = Input>,
    V::To<ArrayType>: Parameterized<ArrayType, To<JitTracer<ArrayType, V, O, L>> = JitTracer<ArrayType, V, O, L>>,
    O: InterpretableOp<
            ArrayType,
            Linearized<JitTracer<ArrayType, V, O, L>, LinearProgramOpRef<JitTracer<ArrayType, V, O, L>>>,
        >,
    LinearProgramOpRef<JitTracer<ArrayType, V, O, L>>: CoreLinearProgramOp<JitTracer<ArrayType, V, O, L>>,
{
    type Value = JitTracer<ArrayType, V, O, L>;
    type FunctionInput = Input;
    type FunctionOutput = JitTracer<ArrayType, V, O, L>;

    fn invoke<F>(_engine: &E, function: F, primals: Input) -> Result<(Self::Value, Input), TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> Self::FunctionOutput,
    {
        let input_structure = primals.parameter_structure();
        let traced_primals = primals.into_parameters().collect::<Vec<_>>();
        let staged_input_types = Input::To::<ArrayType>::from_parameters(
            input_structure.clone(),
            traced_primals.iter().map(|traced_primal| traced_primal.tpe().into_owned()).collect::<Vec<_>>(),
        )?;
        let (_, traced_program) =
            trace_flat_program_from_input_types::<Input::To<ArrayType>, V::To<ArrayType>, V, O, L, _>(
                |staged_input| Ok(function(staged_input)),
                &traced_primals,
                staged_input_types,
            )?;
        let (traced_output, traced_gradient) = reverse_mode_scalar_traced_program(&traced_program, traced_primals)?;
        Ok((traced_output, Input::from_parameters(input_structure, traced_gradient)?))
    }
}

/// Batched dispatch for [`value_and_grad`], enabling standalone
/// `vmap(|x| value_and_grad(f, x), inputs)` -- computing per-element function values and gradients
/// over a batch without requiring an outer [`jit`] wrapper.
///
/// Uses a trace-once strategy for [`Batch`]: the user function is traced once to a [`Program`],
/// and a [`CompiledFunction`] that produces `(V, Input::To<V>)` per lane is compiled via [`jit`].
/// Values and gradients are collected per lane and stacked separately.
impl<
    E,
    V: Traceable<ArrayType> + ZeroLike + OneLike,
    Input: Parameterized<Batch<V>, ParameterStructure: Clone + PartialEq>,
> ValueAndGradInvocationLeaf<E, Input> for Batch<V>
where
    E: Engine<Type = ArrayType, Value = V>,
    V: Parameterized<
            V,
            ParameterStructure = Placeholder,
            To<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>> = JitTracer<
                ArrayType,
                V,
                E::TracingOperation,
                E::LinearOperation,
            >,
        >,
    V::Family: ParameterizedFamily<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
    Vec<V>: Parameterized<
            V,
            ParameterStructure = Vec<Placeholder>,
            To<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>> = Vec<
                JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
            >,
        >,
    <Vec<V> as Parameterized<V>>::Family:
        ParameterizedFamily<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
    Input::Family:
        ParameterizedFamily<V> + ParameterizedFamily<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
    Input::To<V>: Clone
        + Parameterized<
            V,
            ParameterStructure: Clone + PartialEq,
            To<Batch<V>> = Input,
            To<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>> = Input::To<
                JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
            >,
        >,
    E::TracingOperation: Clone + Op<ArrayType>,
    E::TracingOperation: InterpretableOp<ArrayType, V>,
    E::TracingOperation: InterpretableOp<
            ArrayType,
            Linearized<
                JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
                LinearProgramOpRef<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
            >,
        >,
    LinearProgramOpRef<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>:
        CoreLinearProgramOp<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
{
    type Value = Batch<V>;
    type FunctionInput = Input::To<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>;
    type FunctionOutput = JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>;

    fn invoke<F>(engine: &E, function: F, primals: Input) -> Result<(Self::Value, Input), TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> Self::FunctionOutput,
    {
        let lane_primals: Vec<Input::To<V>> = unstack(primals)?;
        if lane_primals.is_empty() {
            return Err(TraceError::EmptyBatch);
        }

        let lane0 = lane_primals[0].clone();
        let input_structure = lane0.parameter_structure();
        let parameter_count = input_structure.parameter_count();
        let lane0_flat: Vec<V> = lane0.into_parameters().collect();

        // Trace the user function once at lane 0 primals, consuming the FnOnce closure.
        let (_, traced_program): (V, Program<ArrayType, V, Input::To<V>, V, E::TracingOperation>) =
            trace_program(engine, |staged_input| Ok(function(staged_input)), lane_primals[0].clone())?;

        // Reshape the program to flat Vec<V> inputs and outputs for the JIT compilation step.
        let flat_program = Program::from_graph(traced_program.graph().clone_with_structures::<Vec<V>, Vec<V>>(
            flat_leaf_parameter_structure(parameter_count),
            flat_leaf_parameter_structure(1),
        ))
        .simplify()?;

        // Compile both the forward evaluation and gradient into a reusable program.
        let (_, compiled_vg): (Vec<V>, CompiledFunction<ArrayType, V, Vec<V>, Vec<V>, E::TracingOperation>) = jit(
            engine,
            |jit_primals: Vec<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>| {
                let (output, gradient) = reverse_mode_scalar_traced_program(&flat_program, jit_primals)?;
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
            let flat_result = compiled_vg.call(flat)?;
            let (value, grad_flat) = flat_result.split_first().ok_or(TraceError::EmptyParameterizedValue)?;
            lane_values.push(value.clone());
            lane_grads.push(
                Input::To::<V>::from_parameters(input_structure.clone(), grad_flat.to_vec())
                    .map_err(TraceError::from)?,
            );
        }

        let batched_values = Batch::new(lane_values);
        let batched_grads = stack(lane_grads)?;
        Ok((batched_values, batched_grads))
    }
}

/// Computes both the primal scalar output and its reverse-mode gradient.
#[allow(private_bounds, private_interfaces)]
pub fn value_and_grad<E, F, Input, Leaf>(
    engine: &E,
    function: F,
    primals: Input,
) -> Result<(<Leaf as ValueAndGradInvocationLeaf<E, Input>>::Value, Input), TraceError>
where
    E: Engine<Type = ArrayType>,
    Leaf: ValueAndGradInvocationLeaf<E, Input>,
    Input: Parameterized<Leaf, ParameterStructure: Clone + PartialEq>,
    F: FnOnce(
        <Leaf as ValueAndGradInvocationLeaf<E, Input>>::FunctionInput,
    ) -> <Leaf as ValueAndGradInvocationLeaf<E, Input>>::FunctionOutput,
{
    Leaf::invoke(engine, function, primals)
}

/// Computes the reverse-mode gradient of a scalar-output function.
#[allow(private_bounds, private_interfaces)]
pub fn grad<E, F, Input, Leaf>(engine: &E, function: F, primals: Input) -> Result<Input, TraceError>
where
    E: Engine<Type = ArrayType>,
    Leaf: ValueAndGradInvocationLeaf<E, Input>,
    Input: Parameterized<Leaf, ParameterStructure: Clone + PartialEq>,
    F: FnOnce(
        <Leaf as ValueAndGradInvocationLeaf<E, Input>>::FunctionInput,
    ) -> <Leaf as ValueAndGradInvocationLeaf<E, Input>>::FunctionOutput,
{
    Leaf::invoke(engine, function, primals).map(|(_, gradient)| gradient)
}
