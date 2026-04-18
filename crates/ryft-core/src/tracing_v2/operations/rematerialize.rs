//! Higher-order `rematerialize` operation for [`crate::tracing_v2`].
//!
//! Wraps a sub-computation so that its forward pass is recomputed (rather than saved) during
//! reverse-mode differentiation. During forward execution the behavior is identical to calling the
//! body directly; the difference is visible only to the differentiation transform, which embeds the
//! body's forward equations in the tangent program instead of baking intermediate values as
//! constants.

use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use crate::{
    parameters::{Parameter, Parameterized, ParameterizedFamily, Placeholder},
    tracing_v2::{
        CompiledFunction, JitTracer, LinearTerm, Program, TraceError, TraceInput, TraceOutput, Traceable, Value,
        ZeroLike,
        engine::Engine,
        jit::try_trace_program_for_operation,
        linear::{
            linearize_program, replay_program_graph_linearized_jit, transpose_linear_program_with_output_examples,
        },
        ops::{
            CoreLinearProgramOp, DifferentiableOp, InterpretableOp, LinearOperation, LinearPrimitiveOp, Op,
            RematerializeTracingOperation,
        },
        program::{LinearProgramOpRef, ProgramOpRef},
    },
    types::{ArrayType, Type, Typed},
};

/// Erased traced body for a rematerialization boundary.
pub struct FlatTracedRematerialize<T: Type, V: Typed<T> + Parameter, O = ProgramOpRef<V>> {
    /// Canonical input types of the body.
    input_types: Vec<T>,

    /// Canonical output types of the body.
    output_types: Vec<T>,

    /// The compiled body sub-program.
    compiled: CompiledFunction<T, V, Vec<V>, Vec<V>, O>,
}

impl<T: Type, V: Traceable<T>, O: Clone> Clone for FlatTracedRematerialize<T, V, O>
where
    <Vec<V> as Parameterized<V>>::ParameterStructure: Clone,
{
    fn clone(&self) -> Self {
        Self {
            input_types: self.input_types.clone(),
            output_types: self.output_types.clone(),
            compiled: self.compiled.clone(),
        }
    }
}

impl<T: Type, V: Traceable<T>, O: Clone> FlatTracedRematerialize<T, V, O> {
    /// Builds one erased traced rematerialize body from explicit staged parts.
    #[inline]
    pub fn from_parts(
        input_types: Vec<T>,
        output_types: Vec<T>,
        compiled: CompiledFunction<T, V, Vec<V>, Vec<V>, O>,
    ) -> Self {
        Self { input_types, output_types, compiled }
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

    /// Returns the compiled body sub-program.
    #[inline]
    pub fn compiled(&self) -> &CompiledFunction<T, V, Vec<V>, Vec<V>, O> {
        &self.compiled
    }
}

/// Higher-order operation that marks its body for rematerialization during linearization.
///
/// During forward execution the body is evaluated normally. When linearized, the body's pushforward
/// is computed and staged so that the tangent program recomputes forward intermediates from the
/// inputs rather than storing them as constants.
pub struct RematerializeOp<
    T: Type + Display,
    V: Traceable<T> + Parameter,
    O: Clone = ProgramOpRef<V>,
    L: Clone = LinearProgramOpRef<V>,
> {
    /// The forward body sub-program.
    body: FlatTracedRematerialize<T, V, O>,
    marker: PhantomData<fn() -> L>,
}

impl<T: Type + Display, V: Traceable<T>, O: Clone, L: Clone> Clone for RematerializeOp<T, V, O, L> {
    fn clone(&self) -> Self {
        Self { body: self.body.clone(), marker: PhantomData }
    }
}

impl<T: Type + Display, V: Traceable<T>, O: Clone, L: Clone> RematerializeOp<T, V, O, L> {
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

impl<T: Type + Display, V: Traceable<T>, O: Clone, L: Clone> Debug for RematerializeOp<T, V, O, L> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Rematerialize")
    }
}

impl<T: Type + Display, V: Traceable<T>, O: Clone, L: Clone> Display for RematerializeOp<T, V, O, L> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "rematerialize")
    }
}

impl<V: Traceable<ArrayType>, O: Clone, L: Clone> Op for RematerializeOp<ArrayType, V, O, L> {
    fn name(&self) -> &'static str {
        "rematerialize"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        if inputs.len() != self.body.input_types.len() {
            return Err(TraceError::InvalidInputCount { expected: self.body.input_types.len(), got: inputs.len() });
        }
        if inputs != self.body.input_types.as_slice() {
            return Err(TraceError::IncompatibleAbstractValues { op: "rematerialize" });
        }
        Ok(self.body.output_types.clone())
    }
}

impl<V: Traceable<ArrayType>, O: Clone, L: Clone> InterpretableOp<ArrayType, V> for RematerializeOp<ArrayType, V, O, L>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    O: InterpretableOp<ArrayType, V>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        let abstract_inputs = inputs.iter().map(|input| input.tpe().into_owned()).collect::<Vec<_>>();
        let _ = self.abstract_eval(abstract_inputs.as_slice())?;
        self.body.compiled.call(inputs.to_vec())
    }
}

impl<V: Traceable<ArrayType> + ZeroLike, O: Clone>
    InterpretableOp<ArrayType, crate::tracing_v2::linear::Linearized<JitTracer<ArrayType, V, O>>>
    for RematerializeOp<ArrayType, V, O>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    O: Op<ArrayType>,
    O: InterpretableOp<ArrayType, V>,
    O: InterpretableOp<ArrayType, crate::tracing_v2::linear::Linearized<JitTracer<ArrayType, V, O>>>,
    O: RematerializeTracingOperation<ArrayType, V, LinearProgramOpRef<V>>,
    LinearProgramOpRef<JitTracer<ArrayType, V, O>>: CoreLinearProgramOp<JitTracer<ArrayType, V, O>>,
{
    fn interpret(
        &self,
        inputs: &[crate::tracing_v2::linear::Linearized<JitTracer<ArrayType, V, O>>],
    ) -> Result<Vec<crate::tracing_v2::linear::Linearized<JitTracer<ArrayType, V, O>>>, TraceError> {
        let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let primal_output_values = <Self as InterpretableOp<ArrayType, V>>::interpret(
            self,
            primal_inputs.iter().map(|input| input.value.clone()).collect::<Vec<_>>().as_slice(),
        )?;
        let primal_outputs = JitTracer::apply_staged_op(
            primal_inputs.as_slice(),
            O::rematerialize_op(self.clone()),
            primal_output_values,
        )?;
        let tangent_outputs = replay_program_graph_linearized_jit::<_, _, _, O, LinearProgramOpRef<V>>(
            self.body().compiled().program().graph(),
            inputs.to_vec(),
        )?;
        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs.into_iter().map(|output| output.tangent))
            .map(|(primal, tangent)| crate::tracing_v2::JvpTracer { primal, tangent })
            .collect::<Vec<_>>())
    }
}

impl<V: Traceable<ArrayType> + ZeroLike, O: Clone + 'static>
    DifferentiableOp<ArrayType, V, LinearTerm<ArrayType, V, LinearProgramOpRef<V>>, O, LinearProgramOpRef<V>>
    for RematerializeOp<ArrayType, V, O>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    O: DifferentiableOp<ArrayType, V, LinearTerm<ArrayType, V, LinearProgramOpRef<V>>, O, LinearProgramOpRef<V>>,
    O: InterpretableOp<ArrayType, V>,
    O: InterpretableOp<ArrayType, crate::tracing_v2::linear::Linearized<JitTracer<ArrayType, V, O>>>,
    LinearProgramOpRef<V>: CoreLinearProgramOp<V>,
    LinearProgramOpRef<JitTracer<ArrayType, V, O>>: CoreLinearProgramOp<JitTracer<ArrayType, V, O>>,
{
    fn jvp(
        &self,
        engine: &dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = LinearProgramOpRef<V>>,
        inputs: &[crate::tracing_v2::JvpTracer<V, LinearTerm<ArrayType, V, LinearProgramOpRef<V>>>],
    ) -> Result<Vec<crate::tracing_v2::JvpTracer<V, LinearTerm<ArrayType, V, LinearProgramOpRef<V>>>>, TraceError> {
        let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let tangent_inputs = inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
        let primal_outputs = <Self as InterpretableOp<ArrayType, V>>::interpret(self, primal_inputs.as_slice())?;
        let tangent_outputs = LinearTerm::apply_staged_op(
            tangent_inputs.as_slice(),
            LinearPrimitiveOp::Rematerialize(Box::new(make_linear_rematerialize(engine, &self.body, primal_inputs)?)),
            self.body.output_types.len(),
        )?;
        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| crate::tracing_v2::JvpTracer { primal, tangent })
            .collect::<Vec<_>>())
    }
}

impl<V: Traceable<ArrayType>, O: Clone> InterpretableOp<ArrayType, JitTracer<ArrayType, V, O>>
    for RematerializeOp<ArrayType, V, O>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    O: Op<ArrayType>
        + InterpretableOp<ArrayType, V>
        + RematerializeTracingOperation<ArrayType, V, LinearProgramOpRef<V>>,
{
    fn interpret(&self, inputs: &[JitTracer<ArrayType, V, O>]) -> Result<Vec<JitTracer<ArrayType, V, O>>, TraceError> {
        let concrete_inputs = inputs.iter().map(|input| input.value.clone()).collect::<Vec<_>>();
        let output_values = <Self as InterpretableOp<ArrayType, V>>::interpret(self, concrete_inputs.as_slice())?;
        JitTracer::apply_staged_op(inputs, O::rematerialize_op(self.clone()), output_values)
    }
}

/// Linear-only rematerialization boundary that always carries both the linear body and its transpose body.
pub struct LinearRematerializeOp<T: Type + Display, V: Traceable<T> + Parameter, O: Clone = LinearProgramOpRef<V>> {
    /// The forward linear body sub-program.
    body: FlatTracedRematerialize<T, V, O>,

    /// The transpose linear body.
    transpose_body: FlatTracedRematerialize<T, V, O>,
}

impl<T: Type + Display, V: Traceable<T>, O: Clone> Clone for LinearRematerializeOp<T, V, O> {
    fn clone(&self) -> Self {
        Self { body: self.body.clone(), transpose_body: self.transpose_body.clone() }
    }
}

impl<T: Type + Display, V: Traceable<T>, O: Clone> LinearRematerializeOp<T, V, O> {
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

impl<T: Type + Display, V: Traceable<T>, O: Clone> Debug for LinearRematerializeOp<T, V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "LinearRematerialize")
    }
}

impl<T: Type + Display, V: Traceable<T>, O: Clone> Display for LinearRematerializeOp<T, V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "rematerialize")
    }
}

impl<V: Traceable<ArrayType>, O: Clone> Op for LinearRematerializeOp<ArrayType, V, O> {
    fn name(&self) -> &'static str {
        "rematerialize"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        if inputs.len() != self.body.input_types.len() {
            return Err(TraceError::InvalidInputCount { expected: self.body.input_types.len(), got: inputs.len() });
        }
        if inputs != self.body.input_types.as_slice() {
            return Err(TraceError::IncompatibleAbstractValues { op: "rematerialize" });
        }
        Ok(self.body.output_types.clone())
    }
}

impl<V: Traceable<ArrayType>, O: Clone> InterpretableOp<ArrayType, V> for LinearRematerializeOp<ArrayType, V, O>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    O: InterpretableOp<ArrayType, V>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        let abstract_inputs = inputs.iter().map(|input| input.tpe().into_owned()).collect::<Vec<_>>();
        let _ = self.abstract_eval(abstract_inputs.as_slice())?;
        self.body.compiled.call(inputs.to_vec())
    }
}

impl<V: Traceable<ArrayType>> LinearOperation<ArrayType, V> for LinearRematerializeOp<ArrayType, V> {
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, V>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, V>>>, TraceError> {
        let transpose = self.transpose_op();
        Ok(LinearTerm::apply_staged_op(
            output_cotangents,
            LinearPrimitiveOp::Rematerialize(Box::new(transpose)),
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
pub(crate) fn make_linear_rematerialize<V, O>(
    engine: &dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = LinearProgramOpRef<V>>,
    body: &FlatTracedRematerialize<ArrayType, V, O>,
    input_primals: Vec<V>,
) -> Result<LinearRematerializeOp<ArrayType, V>, TraceError>
where
    V: Traceable<ArrayType> + ZeroLike,
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    O: Clone + Op<ArrayType> + 'static,
    O: InterpretableOp<ArrayType, V>,
    O: DifferentiableOp<ArrayType, V, LinearTerm<ArrayType, V, LinearProgramOpRef<V>>, O, LinearProgramOpRef<V>>,
    O: InterpretableOp<ArrayType, crate::tracing_v2::linear::Linearized<JitTracer<ArrayType, V, O>>>,
    LinearProgramOpRef<V>: CoreLinearProgramOp<V>,
    LinearProgramOpRef<JitTracer<ArrayType, V, O>>: CoreLinearProgramOp<JitTracer<ArrayType, V, O>>,
{
    let output_primals = body.compiled.call(input_primals.clone())?;
    let pushforward = linearize_program(engine, body.compiled.program(), input_primals)?;
    let pullback = transpose_linear_program_with_output_examples(&pushforward, output_primals.as_slice())?;
    Ok(LinearRematerializeOp::new(
        FlatTracedRematerialize::from_parts(
            body.input_types.clone(),
            body.output_types.clone(),
            CompiledFunction::from_program(pushforward.program().clone()),
        ),
        FlatTracedRematerialize::from_parts(
            body.output_types.clone(),
            body.input_types.clone(),
            CompiledFunction::from_program(pullback.program().clone()),
        ),
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
    fn invoke<F>(function: F, input: Input) -> Result<Output, TraceError>
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
    fn invoke<F>(function: F, input: Input) -> Result<Output, TraceError>
    where
        F: FnOnce(Input) -> Output,
    {
        Ok(function(input))
    }
}

/// Already-traced dispatch for [`rematerialize`]: traces the body function into a sub-program and
/// stages a [`RematerializeOp`] in the enclosing [`JitTracer`] scope. The sub-program is traced
/// once over exemplar values and compiled into a [`CompiledFunction`] that lowering can later handle.
impl<
    V: Traceable<ArrayType>,
    Input: Parameterized<Self, ParameterStructure: Clone>,
    Output: Parameterized<Self, ParameterStructure: Clone>,
    O: Clone + Op<ArrayType>,
    L: Clone,
> RematerializeInvocationLeaf<Input, Output> for JitTracer<ArrayType, V, O, L>
where
    Input::Family: ParameterizedFamily<V>,
    Output::Family: ParameterizedFamily<V>,
    Input::To<V>: TraceInput<V, O, L, Traced = Input>,
    Output::To<V>: TraceOutput<V, O, L, Traced = Output>,
    O: InterpretableOp<ArrayType, V> + RematerializeTracingOperation<ArrayType, V, L>,
{
    fn invoke<F>(function: F, input: Input) -> Result<Output, TraceError>
    where
        F: FnOnce(Input) -> Output,
    {
        let input_structure = input.parameter_structure();
        let traced_inputs = input.into_parameters().collect::<Vec<_>>();
        let input_leaf_count = traced_inputs.len();
        let exemplar_primals = Input::To::<V>::from_parameters(
            input_structure.clone(),
            traced_inputs.iter().map(|input| input.value.clone()).collect::<Vec<_>>(),
        )?;

        let (exemplar_outputs, body_program): (Output::To<V>, Program<ArrayType, V, Input::To<V>, Output::To<V>, O>) =
            try_trace_program_for_operation::<_, Input::To<V>, Output::To<V>, V, O, L>(
                move |staged_input| {
                    let adapted_input =
                        Input::from_parameters(input_structure, staged_input.into_parameters().collect::<Vec<_>>())?;
                    Ok(function(adapted_input))
                },
                exemplar_primals,
            )?;

        let output_structure = exemplar_outputs.parameter_structure();
        let output_leaf_count = output_structure.parameter_count();
        let input_types = body_program
            .graph()
            .input_atoms()
            .iter()
            .map(|id| body_program.graph().atom(*id).expect("body input atom should exist").tpe().into_owned())
            .collect::<Vec<_>>();
        let output_types = exemplar_outputs.parameters().map(|x| x.tpe().into_owned()).collect::<Vec<_>>();
        let body = FlatTracedRematerialize::from_parts(
            input_types,
            output_types,
            CompiledFunction::from_graph(body_program.graph().clone_with_structures::<Vec<V>, Vec<V>>(
                vec![Placeholder; input_leaf_count],
                vec![Placeholder; output_leaf_count],
            )),
        );

        let output_values =
            body.compiled().call(traced_inputs.iter().map(|input| input.value.clone()).collect::<Vec<_>>())?;
        let staged_outputs = JitTracer::apply_staged_op(
            traced_inputs.as_slice(),
            O::rematerialize_op(RematerializeOp::new(body)),
            output_values,
        )?;
        Output::from_parameters(output_structure, staged_outputs).map_err(TraceError::from)
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
pub fn rematerialize<F, Input, Output, V>(function: F, input: Input) -> Result<Output, TraceError>
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

    use crate::tracing_v2::{
        CompiledFunction, JitTracer, Sin,
        engine::ArrayScalarEngine,
        jit,
        linear::{compile_grad, grad, value_and_grad},
    };

    use super::*;

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    #[test]
    fn test_rematerialize_concrete_is_identity() {
        // rematerialize with concrete values should just call the function.
        let result: f64 = rematerialize(|x: f64| x.sin(), 2.0f64).unwrap();
        approx_eq(result, 2.0f64.sin());
    }

    #[test]
    fn test_rematerialize_jit_produces_traced_op() {
        // When used inside jit, rematerialize should produce a "rematerialize" op in the graph.
        let engine = ArrayScalarEngine::<f64>::new();
        let (output, compiled): (f64, CompiledFunction<ArrayType, f64, f64, f64>) = jit(
            &engine,
            |x: JitTracer<ArrayType, f64>| rematerialize(|y: JitTracer<ArrayType, f64>| y.sin(), x).unwrap(),
            2.0f64,
        )
        .unwrap();

        approx_eq(output, 2.0f64.sin());
        let ir = compiled.to_string();
        assert!(ir.contains("rematerialize"), "jit graph should contain the rematerialize op: {ir}");
    }

    #[test]
    fn test_rematerialize_jit_graph_rendering() {
        // Check the exact rendering of the jit-traced graph containing a rematerialize op.
        let engine = ArrayScalarEngine::<f64>::new();
        let (_, compiled): (f64, CompiledFunction<ArrayType, f64, f64, f64>) = jit(
            &engine,
            |x: JitTracer<ArrayType, f64>| rematerialize(|y: JitTracer<ArrayType, f64>| y.sin(), x).unwrap(),
            2.0f64,
        )
        .unwrap();

        assert_eq!(
            compiled.to_string(),
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
        let gradient: f64 = grad(
            &engine,
            |x: JitTracer<ArrayType, f64>| rematerialize(|y: JitTracer<ArrayType, f64>| y.sin(), x).unwrap(),
            2.0f64,
        )
        .unwrap();

        approx_eq(gradient, 2.0f64.cos());
    }

    #[test]
    fn test_rematerialize_value_and_grad_returns_both() {
        // value_and_grad of rematerialize(sin, x) should give (sin(x), cos(x)).
        let engine = ArrayScalarEngine::<f64>::new();
        let (value, gradient): (f64, f64) = value_and_grad(
            &engine,
            |x: JitTracer<ArrayType, f64>| rematerialize(|y: JitTracer<ArrayType, f64>| y.sin(), x).unwrap(),
            2.0f64,
        )
        .unwrap();

        approx_eq(value, 2.0f64.sin());
        approx_eq(gradient, 2.0f64.cos());
    }

    #[test]
    fn test_rematerialize_compile_grad_produces_reusable_gradient() {
        // compile_grad with rematerialize should produce a symbolic gradient program.
        let engine = ArrayScalarEngine::<f64>::new();
        let compiled = compile_grad(
            &engine,
            |x: JitTracer<ArrayType, f64>| rematerialize(|y: JitTracer<ArrayType, f64>| y.sin(), x).unwrap(),
            2.0f64,
        )
        .unwrap();

        // Verify at the original primal point: d/dx sin(x) = cos(x).
        let grad_at_2 = compiled.call(2.0f64).unwrap();
        approx_eq(grad_at_2, 2.0f64.cos());

        // Verify at a different primal point to confirm the gradient is symbolic.
        let grad_at_half = compiled.call(0.5f64).unwrap();
        approx_eq(grad_at_half, 0.5f64.cos());

        let grad_at_pi = compiled.call(std::f64::consts::PI).unwrap();
        approx_eq(grad_at_pi, std::f64::consts::PI.cos());
    }

    #[test]
    fn test_rematerialize_grad_of_quadratic_plus_sin() {
        // grad of rematerialize(x^2 + sin(x), x) should be 2x + cos(x).
        let engine = ArrayScalarEngine::<f64>::new();
        let gradient: f64 = grad(
            &engine,
            |x: JitTracer<ArrayType, f64>| {
                rematerialize(|y: JitTracer<ArrayType, f64>| y.clone() * y.clone() + y.sin(), x).unwrap()
            },
            2.0f64,
        )
        .unwrap();

        approx_eq(gradient, 2.0 * 2.0 + 2.0f64.cos());
    }

    #[test]
    fn test_rematerialize_compile_grad_quadratic_plus_sin() {
        // compile_grad with rematerialize wrapping a multi-op body.
        let engine = ArrayScalarEngine::<f64>::new();
        let compiled = compile_grad(
            &engine,
            |x: JitTracer<ArrayType, f64>| {
                rematerialize(|y: JitTracer<ArrayType, f64>| y.clone() * y.clone() + y.sin(), x).unwrap()
            },
            2.0f64,
        )
        .unwrap();

        // d/dx(x^2 + sin(x)) = 2x + cos(x)
        let grad_at_2 = compiled.call(2.0f64).unwrap();
        approx_eq(grad_at_2, 2.0 * 2.0 + 2.0f64.cos());

        let grad_at_half = compiled.call(0.5f64).unwrap();
        approx_eq(grad_at_half, 2.0 * 0.5 + 0.5f64.cos());
    }

    #[test]
    fn test_rematerialize_does_not_affect_forward_result() {
        // The forward result with rematerialize should match the result without it.
        let without: f64 = {
            let engine = ArrayScalarEngine::<f64>::new();
            let (output, _): (f64, CompiledFunction<ArrayType, f64, f64, f64>) =
                jit(&engine, |x: JitTracer<ArrayType, f64>| x.clone() * x.clone() + x.sin(), 3.0f64).unwrap();
            output
        };
        let with: f64 = {
            let engine = ArrayScalarEngine::<f64>::new();
            let (output, _): (f64, CompiledFunction<ArrayType, f64, f64, f64>) = jit(
                &engine,
                |x: JitTracer<ArrayType, f64>| {
                    rematerialize(|y: JitTracer<ArrayType, f64>| y.clone() * y.clone() + y.sin(), x).unwrap()
                },
                3.0f64,
            )
            .unwrap();
            output
        };

        approx_eq(without, with);
    }
}
