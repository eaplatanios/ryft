use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use crate::{
    parameters::{Parameter, Parameterized, ParameterizedFamily, Placeholder},
    tracing::{Program, Traceable, TracingError, Value},
    tracing_v2::{
        Differentiable, DifferentiationError, EngineTangent, LinearPrimitiveOperation, LinearTerm, PrimitiveOperation,
        Tracer,
        engine::Engine,
        linear::{
            linearize_program, replay_program_linearized_jit, trace_flat_program_from_input_types,
            transpose_linear_program_with_output_examples,
        },
        operations::constants::ZeroLike,
    },
    types::{ArrayType, Type, TypeError, Typed},
};

use super::{CoreLinearProgramOperation, DifferentiableOperation, InterpretableOperation, LinearOperation, Operation};

/// Hidden staging trait for the `rematerialize` higher-order primitive.
#[doc(hidden)]
pub trait RematerializeTracingOperation<T: Type + Display, V: Traceable<T>, L: Clone>: Clone + Operation<T> {
    /// Constructs the carrier-specific representation of the `rematerialize` higher-order primitive
    /// with a captured traced body.
    fn rematerialize_op(op: RematerializeOperation<T, V, Self, L>) -> Self;
}

/// Hidden staging trait for the `rematerialize` higher-order primitive in linear programs.
#[doc(hidden)]
pub trait LinearRematerializeCarrierOperation<T: Type + Display, V: Traceable<T>>: Clone + Operation<T> {
    /// Constructs the carrier-specific representation of the linear `rematerialize` higher-order
    /// primitive with a captured linear traced body.
    fn linear_rematerialize_op(op: LinearRematerializeOperation<T, V, Self>) -> Self;
}

/// Erased traced body for a rematerialization boundary.
///
/// This stores a flattened traced body that higher-order op nodes can carry around independently of
/// the caller's original parameter shapes.
#[derive(Clone)]
pub struct FlatTracedRematerialize<T: Type, V: Traceable<T>, O: Clone + Operation<T> = PrimitiveOperation<V>> {
    /// Canonical input types of the body.
    input_types: Vec<T>,

    /// Canonical output types of the body.
    output_types: Vec<T>,

    /// Flat body sub-program executed by this rematerialization boundary.
    program: Program<T, V, O, Vec<V>, Vec<V>>,
}

impl<T: Type, V: Traceable<T>, O: Clone + Operation<T>> FlatTracedRematerialize<T, V, O> {
    /// Builds one erased traced rematerialize body from explicit staged parts.
    #[inline]
    pub fn from_parts(input_types: Vec<T>, output_types: Vec<T>, program: Program<T, V, O, Vec<V>, Vec<V>>) -> Self {
        Self { input_types, output_types, program }
    }

    /// Returns the canonical input types of the body.
    #[inline]
    pub fn input_types(&self) -> &[T] {
        self.input_types.as_slice()
    }

    /// Returns the canonical output types of the body.
    #[inline]
    pub fn output_types(&self) -> &[T] {
        self.output_types.as_slice()
    }

    /// Returns the flat body sub-program.
    #[inline]
    pub fn program(&self) -> &Program<T, V, O, Vec<V>, Vec<V>> {
        &self.program
    }
}

/// Higher-order operation that marks its body for rematerialization during linearization.
///
/// During forward execution the body is evaluated normally. When linearized, the body's pushforward
/// is computed and staged so that the tangent program recomputes forward intermediates from the
/// inputs rather than storing them as constants. This makes [`RematerializeOperation`] the staged IR hook
/// that powers the user-facing rematerialization policies in [`crate::tracing_v2::linear`].
#[derive(Clone)]
pub struct RematerializeOperation<
    T: Type + Display,
    V: Traceable<T> + Parameter,
    O: Clone + Operation<T> = PrimitiveOperation<V>,
    L: Clone = LinearPrimitiveOperation<V>,
> {
    /// The forward body sub-program.
    body: FlatTracedRematerialize<T, V, O>,

    /// Phantom marker tying the op to the linear carrier used when the body is linearized.
    marker: PhantomData<fn() -> L>,
}

impl<T: Type + Display, V: Traceable<T>, O: Clone + Operation<T>, L: Clone> RematerializeOperation<T, V, O, L> {
    /// Builds one ordinary (non-linear) rematerialize op wrapping the given body.
    #[inline]
    pub fn new(body: FlatTracedRematerialize<T, V, O>) -> Self {
        Self { body, marker: PhantomData }
    }

    /// Returns the forward body.
    #[inline]
    pub fn body(&self) -> &FlatTracedRematerialize<T, V, O> {
        &self.body
    }
}

impl<T: Type + Display, V: Traceable<T>, O: Clone + Operation<T>, L: Clone> Debug
    for RematerializeOperation<T, V, O, L>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Rematerialize")
    }
}

impl<T: Type + Display, V: Traceable<T>, O: Clone + Operation<T>, L: Clone> Display
    for RematerializeOperation<T, V, O, L>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "rematerialize")
    }
}

impl<V: Traceable<ArrayType>, O: Clone + Operation<ArrayType>, L: Clone> Operation<ArrayType>
    for RematerializeOperation<ArrayType, V, O, L>
{
    fn name(&self) -> &'static str {
        "rematerialize"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        if input_types.len() != self.body.input_types.len() {
            return Err(TypeError {
                message: format!(
                    "rematerialize expected {} input types but got {}",
                    self.body.input_types.len(),
                    input_types.len()
                ),
            });
        }
        if input_types != self.body.input_types.as_slice() {
            return Err(TypeError {
                message: "rematerialize input types do not match the captured body signature".to_string(),
            });
        }
        Ok(self.body.output_types.clone())
    }
}

impl<V: Traceable<ArrayType>, O: Clone + Operation<ArrayType>, L: Clone> InterpretableOperation<ArrayType, V>
    for RematerializeOperation<ArrayType, V, O, L>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
    O: InterpretableOperation<ArrayType, V>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        let abstract_inputs = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let _ = self.infer_output_types(abstract_inputs.as_slice())?;
        self.body.program().interpret(inputs.to_vec())
    }
}

impl<'engine, E, V: Value<ArrayType> + ZeroLike>
    InterpretableOperation<ArrayType, crate::tracing_v2::linear::Linearized<Tracer<'engine, E>>>
    for RematerializeOperation<ArrayType, V, E::TracingOperation, E::LinearOperation>
where
    E: Engine<Type = ArrayType, Value = V> + ?Sized + 'static,
    Vec<V>: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
    E::TracingOperation: InterpretableOperation<ArrayType, V>,
    E::TracingOperation: InterpretableOperation<ArrayType, crate::tracing_v2::linear::Linearized<Tracer<'engine, E>>>,
    E::TracingOperation: RematerializeTracingOperation<ArrayType, V, E::LinearOperation>,
    LinearPrimitiveOperation<Tracer<'engine, E>>: CoreLinearProgramOperation<Tracer<'engine, E>>,
{
    fn interpret(
        &self,
        inputs: &[crate::tracing_v2::linear::Linearized<Tracer<'engine, E>>],
    ) -> Result<Vec<crate::tracing_v2::linear::Linearized<Tracer<'engine, E>>>, TracingError> {
        if inputs.is_empty() {
            return if self.body.output_types().is_empty() {
                Ok(Vec::new())
            } else {
                Err(DifferentiationError::MissingTracedRematerializeInputLeaves.into())
            };
        }
        let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let exemplar_primal_input = primal_inputs[0].clone();
        let linear_builder = inputs[0].tangent.builder.clone();
        let primal_outputs = Tracer::apply_staged_op(
            exemplar_primal_input.engine,
            exemplar_primal_input.builder.clone(),
            primal_inputs.as_slice(),
            E::TracingOperation::rematerialize_op(self.clone()),
        )?;
        let body_program = self.body().program();
        let tangent_outputs = replay_program_linearized_jit::<_, _, _, E>(
            exemplar_primal_input.engine,
            exemplar_primal_input.builder.clone(),
            linear_builder,
            body_program,
            inputs.to_vec(),
        )?;
        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs.into_iter().map(|output| output.tangent))
            .map(|(primal, tangent)| crate::tracing_v2::JvpTracer { primal, tangent })
            .collect::<Vec<_>>())
    }
}

impl<
    V: Value<ArrayType>
        + ZeroLike
        + Differentiable<
            ArrayType,
            Tangent<LinearPrimitiveOperation<V>> = LinearTerm<ArrayType, V, LinearPrimitiveOperation<V>>,
        > + 'static,
    E: Engine<Type = ArrayType, Value = V, LinearOperation = LinearPrimitiveOperation<V>> + ?Sized + 'static,
> DifferentiableOperation<E> for RematerializeOperation<ArrayType, V, E::TracingOperation>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
    E::TracingOperation: DifferentiableOperation<E>,
    E::TracingOperation: InterpretableOperation<ArrayType, V>,
    E::TracingOperation:
        for<'engine> InterpretableOperation<ArrayType, crate::tracing_v2::linear::Linearized<Tracer<'engine, E>>>,
    LinearPrimitiveOperation<V>: CoreLinearProgramOperation<V>,
    for<'engine> LinearPrimitiveOperation<Tracer<'engine, E>>: CoreLinearProgramOperation<Tracer<'engine, E>>,
{
    fn jvp(
        &self,
        engine: &E,
        inputs: &[crate::tracing_v2::JvpTracer<V, EngineTangent<E>>],
    ) -> Result<Vec<crate::tracing_v2::JvpTracer<V, EngineTangent<E>>>, TracingError> {
        let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let tangent_inputs = inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
        let primal_outputs = <Self as InterpretableOperation<ArrayType, V>>::interpret(self, primal_inputs.as_slice())?;
        let tangent_builder = if let Some(first_tangent) = tangent_inputs.first() {
            first_tangent.builder.clone()
        } else if self.body.output_types.is_empty() {
            return Ok(Vec::new());
        } else {
            return Err(DifferentiationError::MissingLinearRematerializeReplayTangentLeaves.into());
        };
        let tangent_outputs = LinearTerm::apply_staged_op(
            tangent_builder,
            tangent_inputs.as_slice(),
            LinearPrimitiveOperation::Rematerialize(Box::new(make_linear_rematerialize(
                engine,
                &self.body,
                primal_inputs,
            )?)),
            self.body.output_types.len(),
        )?;
        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| crate::tracing_v2::JvpTracer { primal, tangent })
            .collect::<Vec<_>>())
    }
}

impl<'engine, V: Value<ArrayType>, E: Engine<Type = ArrayType, Value = V> + ?Sized>
    InterpretableOperation<ArrayType, Tracer<'engine, E>>
    for RematerializeOperation<ArrayType, V, E::TracingOperation, E::LinearOperation>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
    E::TracingOperation:
        InterpretableOperation<ArrayType, V> + RematerializeTracingOperation<ArrayType, V, E::LinearOperation>,
{
    fn interpret(&self, inputs: &[Tracer<'engine, E>]) -> Result<Vec<Tracer<'engine, E>>, TracingError> {
        if inputs.is_empty() {
            return if self.body.output_types().is_empty() {
                Ok(Vec::new())
            } else {
                Err(DifferentiationError::MissingTracedRematerializeInputLeaves.into())
            };
        }
        let exemplar_input = inputs[0].clone();
        Tracer::apply_staged_op(
            exemplar_input.engine,
            exemplar_input.builder.clone(),
            inputs,
            E::TracingOperation::rematerialize_op(self.clone()),
        )
    }
}

/// Linear-only rematerialization boundary that always carries both the linear body and its transpose body.
#[derive(Clone)]
pub struct LinearRematerializeOperation<
    T: Type + Display,
    V: Traceable<T> + Parameter,
    O: Clone + Operation<T> = LinearPrimitiveOperation<V>,
> {
    /// The forward linear body sub-program.
    body: FlatTracedRematerialize<T, V, O>,

    /// The transpose linear body.
    transpose_body: FlatTracedRematerialize<T, V, O>,
}

impl<T: Type + Display, V: Traceable<T>, O: Clone + Operation<T>> LinearRematerializeOperation<T, V, O> {
    /// Builds one linear rematerialize op with an explicit transpose body.
    #[inline]
    pub fn new(body: FlatTracedRematerialize<T, V, O>, transpose_body: FlatTracedRematerialize<T, V, O>) -> Self {
        Self { body, transpose_body }
    }

    /// Returns the forward body.
    #[inline]
    pub fn body(&self) -> &FlatTracedRematerialize<T, V, O> {
        &self.body
    }

    fn transpose_op(&self) -> Self {
        Self::new(self.transpose_body.clone(), self.body.clone())
    }
}

impl<T: Type + Display, V: Traceable<T>, O: Clone + Operation<T>> Debug for LinearRematerializeOperation<T, V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "LinearRematerialize")
    }
}

impl<T: Type + Display, V: Traceable<T>, O: Clone + Operation<T>> Display for LinearRematerializeOperation<T, V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "rematerialize")
    }
}

impl<V: Traceable<ArrayType>, O: Clone + Operation<ArrayType>> Operation<ArrayType>
    for LinearRematerializeOperation<ArrayType, V, O>
{
    fn name(&self) -> &'static str {
        "rematerialize"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        if input_types.len() != self.body.input_types.len() {
            return Err(TypeError {
                message: format!(
                    "rematerialize expected {} input types but got {}",
                    self.body.input_types.len(),
                    input_types.len()
                ),
            });
        }
        if input_types != self.body.input_types.as_slice() {
            return Err(TypeError {
                message: "rematerialize input types do not match the captured body signature".to_string(),
            });
        }
        Ok(self.body.output_types.clone())
    }
}

impl<V: Traceable<ArrayType>, O: Clone + Operation<ArrayType>> InterpretableOperation<ArrayType, V>
    for LinearRematerializeOperation<ArrayType, V, O>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
    O: InterpretableOperation<ArrayType, V>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        let abstract_inputs = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let _ = self.infer_output_types(abstract_inputs.as_slice())?;
        self.body.program().interpret(inputs.to_vec())
    }
}

impl<V: Traceable<ArrayType>> LinearOperation<ArrayType, V> for LinearRematerializeOperation<ArrayType, V> {
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, V>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, V>>>, TracingError> {
        let transpose = self.transpose_op();
        if output_cotangents.is_empty() {
            return if self.body.input_types().is_empty() {
                Ok(Vec::new())
            } else {
                Err(DifferentiationError::MissingLinearRematerializeTransposeCotangentLeaves.into())
            };
        }
        let exemplar_output_cotangent = output_cotangents[0].clone();
        Ok(LinearTerm::apply_staged_op(
            exemplar_output_cotangent.builder.clone(),
            output_cotangents,
            LinearPrimitiveOperation::Rematerialize(Box::new(transpose)),
            self.body.input_types().len(),
        )?
        .into_iter()
        .map(Some)
        .collect::<Vec<_>>())
    }
}

/// Builds a linearized rematerialize op from its primal body by computing the pushforward and
/// pullback programs at the provided primal inputs.
#[allow(private_bounds)]
pub(crate) fn make_linear_rematerialize<V, E>(
    engine: &E,
    body: &FlatTracedRematerialize<ArrayType, V, E::TracingOperation>,
    input_primals: Vec<V>,
) -> Result<LinearRematerializeOperation<ArrayType, V>, TracingError>
where
    V: Traceable<ArrayType>
        + ZeroLike
        + Differentiable<
            ArrayType,
            Tangent<LinearPrimitiveOperation<V>> = LinearTerm<ArrayType, V, LinearPrimitiveOperation<V>>,
        > + 'static,
    Vec<V>: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
    E::TracingOperation: InterpretableOperation<ArrayType, V>,
    E::TracingOperation: DifferentiableOperation<E>,
    E::TracingOperation:
        for<'engine> InterpretableOperation<ArrayType, crate::tracing_v2::linear::Linearized<Tracer<'engine, E>>>,
    LinearPrimitiveOperation<V>: CoreLinearProgramOperation<V>,
    E: Engine<Type = ArrayType, Value = V, LinearOperation = LinearPrimitiveOperation<V>> + ?Sized + 'static,
    for<'engine> LinearPrimitiveOperation<Tracer<'engine, E>>: CoreLinearProgramOperation<Tracer<'engine, E>>,
{
    let body_program = body.program();
    let output_primals = body_program.interpret(input_primals.clone())?;
    let pushforward = linearize_program(engine, body_program, input_primals)?;
    let pullback = transpose_linear_program_with_output_examples(engine, &pushforward, output_primals.as_slice())?;
    Ok(LinearRematerializeOperation::new(
        FlatTracedRematerialize::from_parts(body.input_types.clone(), body.output_types.clone(), pushforward),
        FlatTracedRematerialize::from_parts(body.output_types.clone(), body.input_types.clone(), pullback),
    ))
}

// ---------------------------------------------------------------------------
// Dispatch trait and public `rematerialize` entry point
// ---------------------------------------------------------------------------

/// Dispatch trait used by [`rematerialize`] to handle both concrete values and already traced values.
#[doc(hidden)]
pub(crate) trait RematerializeInvocationLeaf<
    Input: Parameterized<Self, ParameterStructure: Clone>,
    Output: Parameterized<Self, ParameterStructure: Clone>,
>: Parameter + Sized
{
    /// Invokes [`rematerialize`] for one concrete leaf regime.
    fn invoke<F>(function: F, input: Input) -> Result<Output, TracingError>
    where
        F: FnOnce(Input) -> Output;
}

/// Concrete-value dispatch for [`rematerialize`]: the rematerialization boundary is a no-op during
/// eager execution and simply applies the body function directly.
impl<
    V: Value<ArrayType>,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
> RematerializeInvocationLeaf<Input, Output> for V
{
    fn invoke<F>(function: F, input: Input) -> Result<Output, TracingError>
    where
        F: FnOnce(Input) -> Output,
    {
        Ok(function(input))
    }
}

/// Already-traced dispatch for [`rematerialize`]: traces the body function into a sub-program and
/// stages a [`RematerializeOperation`] in the enclosing [`Tracer`] scope. The sub-program is traced
/// once over exemplar values and captured as a [`Program`] that lowering can later handle.
impl<
    'engine,
    E,
    V: Traceable<ArrayType>,
    Input: Parameterized<Tracer<'engine, E>, ParameterStructure: Clone, To<Tracer<'engine, E>> = Input>,
    Output: Parameterized<Tracer<'engine, E>, ParameterStructure: Clone, To<Tracer<'engine, E>> = Output>,
> RematerializeInvocationLeaf<Input, Output> for Tracer<'engine, E>
where
    E: Engine<Type = ArrayType, Value = V> + ?Sized + 'static,
    Input::Family: ParameterizedFamily<V> + ParameterizedFamily<ArrayType>,
    Output::Family: ParameterizedFamily<V> + ParameterizedFamily<ArrayType>,
    Input::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'engine, E>> = Input, To<V> = Input::To<V>>,
    Output::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'engine, E>> = Output, To<V> = Output::To<V>>,
    E::TracingOperation:
        InterpretableOperation<ArrayType, V> + RematerializeTracingOperation<ArrayType, V, E::LinearOperation>,
{
    fn invoke<F>(function: F, input: Input) -> Result<Output, TracingError>
    where
        F: FnOnce(Input) -> Output,
    {
        let input_structure = input.parameter_structure();
        let traced_inputs = input.into_parameters().collect::<Vec<_>>();
        let input_leaf_count = traced_inputs.len();
        let exemplar_input_types = Input::To::<ArrayType>::from_parameters(
            input_structure.clone(),
            traced_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
        )?;
        let Some(exemplar_traced_input) = traced_inputs.first().cloned() else {
            return Err(DifferentiationError::MissingTracedRematerializeInputLeaves.into());
        };
        let (exemplar_output_types, body_program) =
            trace_flat_program_from_input_types::<Input::To<ArrayType>, Output::To<ArrayType>, V, E, _>(
                exemplar_traced_input.engine,
                |staged_input| Ok(function(staged_input)),
                exemplar_input_types,
            )?;

        let output_structure = exemplar_output_types.parameter_structure();
        let output_leaf_count = output_structure.parameter_count();
        let input_types = body_program
            .input_ids
            .iter()
            .map(|id| body_program.atoms[id.index].r#type().into_owned())
            .collect::<Vec<_>>();
        let output_types = exemplar_output_types.parameters().cloned().collect::<Vec<_>>();
        let Program { atoms, input_ids, output_ids, instructions, .. } = body_program;
        let body = FlatTracedRematerialize::from_parts(
            input_types,
            output_types,
            Program {
                atoms,
                input_ids,
                output_ids,
                instructions,
                input_structure: vec![Placeholder; input_leaf_count],
                output_structure: vec![Placeholder; output_leaf_count],
                marker: std::marker::PhantomData,
            },
        );

        let staged_outputs = Tracer::apply_staged_op(
            exemplar_traced_input.engine,
            exemplar_traced_input.builder.clone(),
            traced_inputs.as_slice(),
            E::TracingOperation::rematerialize_op(RematerializeOperation::new(body)),
        )?;
        Output::from_parameters(output_structure, staged_outputs).map_err(TracingError::from)
    }
}

/// Marks `function(input)` as a rematerialization boundary.
///
/// During forward execution this is equivalent to calling `function(input)` directly. During
/// reverse-mode differentiation the forward pass of `function` is recomputed from the inputs
/// rather than having its intermediate values saved as constants, trading compute for memory.
///
/// # Example
///
/// ```ignore
/// use ryft_core::tracing_v2::{compile_grad, rematerialize};
/// use ryft_core::tracing_v2::engine::ArrayScalarEngine;
///
/// // Without rematerialize, compile_grad saves all forward intermediates.
/// // With rematerialize, the body is recomputed during the backward pass.
/// let engine = ArrayScalarEngine::<f64>::new();
/// let (_, grad_fn) = compile_grad(&engine, |x: f64| rematerialize(|y| y.sin(), x).unwrap(), 1.0)?;
/// ```
#[allow(private_bounds)]
pub fn rematerialize<F, Input, Output, V>(function: F, input: Input) -> Result<Output, TracingError>
where
    V: RematerializeInvocationLeaf<Input, Output>,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    F: FnOnce(Input) -> Output,
{
    V::invoke(function, input)
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::tracing::{Program, ProgramBuilder};
    use crate::tracing_v2::{
        DifferentiationError, JvpTracer, Linearized, Sin, Tracer,
        engine::ArrayScalarEngine,
        interpret_and_trace,
        linear::{compile_grad, grad, value_and_grad},
    };

    use super::*;

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    fn scalar_type() -> ArrayType {
        ArrayType::scalar(crate::types::DataType::F64)
    }

    fn empty_traced_body() -> FlatTracedRematerialize<ArrayType, f64> {
        let mut builder = ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new();
        let output = builder.add_constant(0.0f64);
        let program = builder.build::<Vec<f64>, Vec<f64>>(vec![output], vec![], vec![Placeholder]);
        FlatTracedRematerialize::from_parts(vec![], vec![scalar_type()], program)
    }

    fn empty_linear_body() -> FlatTracedRematerialize<ArrayType, f64, LinearPrimitiveOperation<f64>> {
        let program = ProgramBuilder::<ArrayType, f64, LinearPrimitiveOperation<f64>>::new()
            .build::<Vec<f64>, Vec<f64>>(vec![], vec![], vec![]);
        FlatTracedRematerialize::from_parts(vec![scalar_type()], vec![], program)
    }

    #[test]
    fn test_rematerialize_concrete_is_identity() {
        // rematerialize with concrete values should just call the function.
        let result: f64 = rematerialize(|x: f64| x.sin(), 2.0f64).unwrap();
        approx_eq(result, 2.0f64.sin());
    }

    #[test]
    fn test_linearized_traced_rematerialize_requires_input_leaves() {
        let operation = RematerializeOperation::<ArrayType, f64>::new(empty_traced_body());
        let inputs: Vec<Linearized<Tracer<'_, ArrayScalarEngine<f64>>>> = Vec::new();

        assert!(matches!(
            operation.interpret(inputs.as_slice()),
            Err(TracingError::Differentiation(DifferentiationError::MissingTracedRematerializeInputLeaves))
        ));
    }

    #[test]
    fn test_linear_rematerialize_jvp_requires_tangent_leaves() {
        let engine = ArrayScalarEngine::<f64>::new();
        let operation = RematerializeOperation::<ArrayType, f64>::new(empty_traced_body());
        let inputs: Vec<JvpTracer<f64, LinearTerm<ArrayType, f64>>> = Vec::new();

        assert!(matches!(
            operation.jvp(&engine, inputs.as_slice()),
            Err(TracingError::Differentiation(DifferentiationError::MissingLinearRematerializeReplayTangentLeaves))
        ));
    }

    #[test]
    fn test_traced_rematerialize_requires_input_leaves() {
        let operation = RematerializeOperation::<ArrayType, f64>::new(empty_traced_body());
        let inputs: Vec<Tracer<'_, ArrayScalarEngine<f64>>> = Vec::new();

        assert!(matches!(
            operation.interpret(inputs.as_slice()),
            Err(TracingError::Differentiation(DifferentiationError::MissingTracedRematerializeInputLeaves))
        ));
    }

    #[test]
    fn test_linear_rematerialize_transpose_requires_output_cotangent_leaves() {
        let operation = LinearRematerializeOperation::<ArrayType, f64>::new(empty_linear_body(), empty_linear_body());
        let output_cotangents: Vec<LinearTerm<ArrayType, f64>> = Vec::new();

        assert!(matches!(
            operation.transpose(output_cotangents.as_slice()),
            Err(TracingError::Differentiation(
                DifferentiationError::MissingLinearRematerializeTransposeCotangentLeaves
            ))
        ));
    }

    #[test]
    fn test_rematerialize_invocation_requires_traced_input_leaves() {
        let input: Vec<Tracer<'_, ArrayScalarEngine<f64>>> = Vec::new();

        let result = <Tracer<'_, ArrayScalarEngine<f64>> as RematerializeInvocationLeaf<
            Vec<Tracer<'_, ArrayScalarEngine<f64>>>,
            Tracer<'_, ArrayScalarEngine<f64>>,
        >>::invoke(|_inputs| panic!("closure should not run without traced inputs"), input);

        assert!(matches!(
            result,
            Err(TracingError::Differentiation(DifferentiationError::MissingTracedRematerializeInputLeaves))
        ));
    }

    #[test]
    fn test_rematerialize_jit_produces_traced_op() {
        // When used inside jit, rematerialize should produce a "rematerialize" op in the program.
        let engine = ArrayScalarEngine::<f64>::new();
        let (output, program): (f64, Program<ArrayType, f64, PrimitiveOperation<f64>, f64, f64>) =
            interpret_and_trace(&engine, |x| Ok(rematerialize(|y| y.sin(), x).unwrap()), 2.0f64).unwrap();

        approx_eq(output, 2.0f64.sin());
        let ir = program.to_string();
        assert!(ir.contains("rematerialize"), "jit program should contain the rematerialize op: {ir}");
    }

    #[test]
    fn test_rematerialize_jit_program_rendering() {
        // Check the exact rendering of the jit-traced program containing a rematerialize op.
        let engine = ArrayScalarEngine::<f64>::new();
        let (_, program): (f64, Program<ArrayType, f64, PrimitiveOperation<f64>, f64, f64>) =
            interpret_and_trace(&engine, |x| Ok(rematerialize(|y| y.sin(), x).unwrap()), 2.0f64).unwrap();

        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = rematerialize %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_rematerialize_grad_computes_correct_gradient() {
        // grad of rematerialize(sin, x) should be cos(x).
        let engine = ArrayScalarEngine::<f64>::new();
        let gradient: f64 = grad(&engine, |x| rematerialize(|y| y.sin(), x).unwrap(), 2.0f64).unwrap();

        approx_eq(gradient, 2.0f64.cos());
    }

    #[test]
    fn test_rematerialize_value_and_grad_returns_both() {
        // value_and_grad of rematerialize(sin, x) should give (sin(x), cos(x)).
        let engine = ArrayScalarEngine::<f64>::new();
        let (value, gradient): (f64, f64) =
            value_and_grad(&engine, |x| rematerialize(|y| y.sin(), x).unwrap(), 2.0f64).unwrap();

        approx_eq(value, 2.0f64.sin());
        approx_eq(gradient, 2.0f64.cos());
    }

    #[test]
    fn test_rematerialize_compile_grad_produces_reusable_gradient() {
        // compile_grad with rematerialize should produce a symbolic gradient program.
        let engine = ArrayScalarEngine::<f64>::new();
        let compiled = compile_grad(&engine, |x| rematerialize(|y| y.sin(), x).unwrap(), 2.0f64).unwrap();

        // Verify at the original primal point: d/dx sin(x) = cos(x).
        let grad_at_2 = compiled.interpret(2.0f64).unwrap();
        approx_eq(grad_at_2, 2.0f64.cos());

        // Verify at a different primal point to confirm the gradient is symbolic.
        let grad_at_half = compiled.interpret(0.5f64).unwrap();
        approx_eq(grad_at_half, 0.5f64.cos());

        let grad_at_pi = compiled.interpret(std::f64::consts::PI).unwrap();
        approx_eq(grad_at_pi, std::f64::consts::PI.cos());
    }

    #[test]
    fn test_rematerialize_grad_of_quadratic_plus_sin() {
        // grad of rematerialize(x^2 + sin(x), x) should be 2x + cos(x).
        let engine = ArrayScalarEngine::<f64>::new();
        let gradient: f64 =
            grad(&engine, |x| rematerialize(|y| y.clone() * y.clone() + y.sin(), x).unwrap(), 2.0f64).unwrap();

        approx_eq(gradient, 2.0 * 2.0 + 2.0f64.cos());
    }

    #[test]
    fn test_rematerialize_compile_grad_quadratic_plus_sin() {
        // compile_grad with rematerialize wrapping a multi-op body.
        let engine = ArrayScalarEngine::<f64>::new();
        let compiled =
            compile_grad(&engine, |x| rematerialize(|y| y.clone() * y.clone() + y.sin(), x).unwrap(), 2.0f64).unwrap();

        // d/dx(x^2 + sin(x)) = 2x + cos(x)
        let grad_at_2 = compiled.interpret(2.0f64).unwrap();
        approx_eq(grad_at_2, 2.0 * 2.0 + 2.0f64.cos());

        let grad_at_half = compiled.interpret(0.5f64).unwrap();
        approx_eq(grad_at_half, 2.0 * 0.5 + 0.5f64.cos());
    }

    #[test]
    fn test_rematerialize_does_not_affect_forward_result() {
        // The forward result with rematerialize should match the result without it.
        let without: f64 = {
            let engine = ArrayScalarEngine::<f64>::new();
            let (output, _): (f64, Program<ArrayType, f64, PrimitiveOperation<f64>, f64, f64>) =
                interpret_and_trace(&engine, |x| Ok(x.clone() * x.clone() + x.sin()), 3.0f64).unwrap();
            output
        };
        let with: f64 = {
            let engine = ArrayScalarEngine::<f64>::new();
            let (output, _): (f64, Program<ArrayType, f64, PrimitiveOperation<f64>, f64, f64>) = interpret_and_trace(
                &engine,
                |x| Ok(rematerialize(|y| y.clone() * y.clone() + y.sin(), x).unwrap()),
                3.0f64,
            )
            .unwrap();
            output
        };

        approx_eq(without, with);
    }
}
