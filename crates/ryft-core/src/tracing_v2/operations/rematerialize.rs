//! Higher-order `rematerialize` operation for [`crate::tracing_v2`].
//!
//! Wraps a sub-computation so that its forward pass is recomputed (rather than saved) during
//! reverse-mode differentiation. During forward execution the behavior is identical to calling the
//! body directly; the difference is visible only to the differentiation transform, which embeds the
//! body's forward equations in the tangent program instead of baking intermediate values as
//! constants.

use std::fmt::{Debug, Display};

use crate::{
    parameters::{Parameter, Parameterized, ParameterizedFamily, Placeholder},
    tracing_v2::{
        CompiledFunction, FloatExt, JitTracer, LinearTerm, MatrixOps, One, Program, TraceError, TraceValue, Zero,
        jit::try_trace_program,
        linear::{linearize_program, replay_program_graph_linearized_jit, transpose_linear_program},
        operations::reshape::ReshapeOps,
        ops::{DifferentiableOp, InterpretableOp, LinearOp, LinearPrimitiveOp, Op, PrimitiveOp},
        program::{LinearProgramOpRef, ProgramOpRef},
    },
    types::{ArrayType, Typed},
};

/// Erased traced body for a rematerialization boundary.
#[derive(Clone)]
pub struct FlatTracedRematerialize<V: TraceValue, O: Clone = ProgramOpRef<V>> {
    /// Canonical input types of the body.
    input_types: Vec<ArrayType>,

    /// Canonical output types of the body.
    output_types: Vec<ArrayType>,

    /// The compiled body sub-program.
    compiled: CompiledFunction<V, Vec<V>, Vec<V>, O>,
}

impl<V: TraceValue, O: Clone> FlatTracedRematerialize<V, O> {
    /// Builds one erased traced rematerialize body from explicit staged parts.
    #[inline]
    pub fn from_parts(
        input_types: Vec<ArrayType>,
        output_types: Vec<ArrayType>,
        compiled: CompiledFunction<V, Vec<V>, Vec<V>, O>,
    ) -> Self {
        Self { input_types, output_types, compiled }
    }

    /// Returns the canonical input types of the body.
    #[inline]
    pub fn input_types(&self) -> &[ArrayType] {
        self.input_types.as_slice()
    }

    /// Returns the canonical output types of the body.
    #[inline]
    pub fn output_types(&self) -> &[ArrayType] {
        self.output_types.as_slice()
    }

    /// Returns the compiled body sub-program.
    #[inline]
    pub fn compiled(&self) -> &CompiledFunction<V, Vec<V>, Vec<V>, O> {
        &self.compiled
    }
}

/// Higher-order operation that marks its body for rematerialization during linearization.
///
/// During forward execution the body is evaluated normally. When linearized, the body's pushforward
/// is computed and staged so that the tangent program recomputes forward intermediates from the
/// inputs rather than storing them as constants.
#[derive(Clone)]
pub struct RematerializeOp<V: TraceValue> {
    /// The forward body sub-program.
    body: FlatTracedRematerialize<V>,
}

impl<V: TraceValue> RematerializeOp<V> {
    /// Builds one ordinary (non-linear) rematerialize op wrapping the given body.
    #[inline]
    pub fn new(body: FlatTracedRematerialize<V>) -> Self {
        Self { body }
    }

    /// Returns the forward body.
    #[inline]
    pub fn body(&self) -> &FlatTracedRematerialize<V> {
        &self.body
    }
}

impl<V: TraceValue> Debug for RematerializeOp<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Rematerialize")
    }
}

impl<V: TraceValue> Display for RematerializeOp<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "rematerialize")
    }
}

impl<V: TraceValue> Op for RematerializeOp<V> {
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

impl<V: TraceValue + FloatExt + Zero + One + MatrixOps + ReshapeOps> InterpretableOp<V> for RematerializeOp<V> {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        let abstract_inputs = inputs.iter().map(Typed::tpe).collect::<Vec<_>>();
        let _ = self.abstract_eval(abstract_inputs.as_slice())?;
        self.body.compiled.call(inputs.to_vec())
    }
}

impl<V: TraceValue + FloatExt + Zero + One + MatrixOps + ReshapeOps>
    InterpretableOp<crate::tracing_v2::linear::Linearized<JitTracer<V>>> for RematerializeOp<V>
{
    fn interpret(
        &self,
        inputs: &[crate::tracing_v2::linear::Linearized<JitTracer<V>>],
    ) -> Result<Vec<crate::tracing_v2::linear::Linearized<JitTracer<V>>>, TraceError> {
        // Replay the body sub-program with JitTracer+LinearTerm inputs. This stages the body's
        // forward equations symbolically in the primal JIT builder and produces tangent atoms that
        // reference those symbolic forward values rather than baked constants.
        let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let primal_output_values = <Self as InterpretableOp<V>>::interpret(
            self,
            primal_inputs.iter().map(|input| input.value.clone()).collect::<Vec<_>>().as_slice(),
        )?;
        let primal_outputs = JitTracer::apply_staged_op(
            primal_inputs.as_slice(),
            PrimitiveOp::Rematerialize(Box::new(self.clone())),
            primal_output_values,
        )?;
        let tangent_outputs =
            replay_program_graph_linearized_jit(self.body().compiled().program().graph(), inputs.to_vec())?;
        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs.into_iter().map(|output| output.tangent))
            .map(|(primal, tangent)| crate::tracing_v2::JvpTracer { primal, tangent })
            .collect::<Vec<_>>())
    }
}

impl<
    V: TraceValue
        + FloatExt
        + Zero
        + One
        + MatrixOps
        + ReshapeOps
        + std::ops::Add<Output = V>
        + std::ops::Mul<Output = V>
        + std::ops::Neg<Output = V>,
> DifferentiableOp<V, LinearTerm<V>> for RematerializeOp<V>
{
    fn jvp(
        &self,
        inputs: &[crate::tracing_v2::JvpTracer<V, LinearTerm<V>>],
    ) -> Result<Vec<crate::tracing_v2::JvpTracer<V, LinearTerm<V>>>, TraceError> {
        let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let tangent_inputs = inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
        let primal_outputs = <Self as InterpretableOp<V>>::interpret(self, primal_inputs.as_slice())?;
        let tangent_outputs = LinearTerm::apply_staged_op(
            tangent_inputs.as_slice(),
            LinearPrimitiveOp::Rematerialize(Box::new(make_linear_rematerialize(&self.body)?)),
            self.body.output_types.len(),
        )?;
        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| crate::tracing_v2::JvpTracer { primal, tangent })
            .collect::<Vec<_>>())
    }
}

impl<V: TraceValue + FloatExt + Zero + One + MatrixOps + ReshapeOps> InterpretableOp<JitTracer<V>>
    for RematerializeOp<V>
{
    fn interpret(&self, inputs: &[JitTracer<V>]) -> Result<Vec<JitTracer<V>>, TraceError> {
        let concrete_inputs = inputs.iter().map(|input| input.value.clone()).collect::<Vec<_>>();
        let output_values = <Self as InterpretableOp<V>>::interpret(self, concrete_inputs.as_slice())?;
        JitTracer::apply_staged_op(inputs, PrimitiveOp::Rematerialize(Box::new(self.clone())), output_values)
    }
}

/// Linear-only rematerialization boundary that always carries both the linear body and its transpose body.
#[derive(Clone)]
pub struct LinearRematerializeOp<V: TraceValue> {
    /// The forward linear body sub-program.
    body: FlatTracedRematerialize<V, LinearProgramOpRef<V>>,

    /// The transpose linear body.
    transpose_body: FlatTracedRematerialize<V, LinearProgramOpRef<V>>,
}

impl<V: TraceValue> LinearRematerializeOp<V> {
    /// Builds one linear rematerialize op with an explicit transpose body.
    #[inline]
    pub fn new(
        body: FlatTracedRematerialize<V, LinearProgramOpRef<V>>,
        transpose_body: FlatTracedRematerialize<V, LinearProgramOpRef<V>>,
    ) -> Self {
        Self { body, transpose_body }
    }

    /// Returns the forward body.
    #[inline]
    pub fn body(&self) -> &FlatTracedRematerialize<V, LinearProgramOpRef<V>> {
        &self.body
    }

    fn transpose_op(&self) -> Self {
        Self::new(self.transpose_body.clone(), self.body.clone())
    }
}

impl<V: TraceValue> Debug for LinearRematerializeOp<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "LinearRematerialize")
    }
}

impl<V: TraceValue> Display for LinearRematerializeOp<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "rematerialize")
    }
}

impl<V: TraceValue> Op for LinearRematerializeOp<V> {
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

impl<V: TraceValue + FloatExt + Zero + One + MatrixOps + ReshapeOps> InterpretableOp<V> for LinearRematerializeOp<V> {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        let abstract_inputs = inputs.iter().map(Typed::tpe).collect::<Vec<_>>();
        let _ = self.abstract_eval(abstract_inputs.as_slice())?;
        self.body.compiled.call(inputs.to_vec())
    }
}

impl<V: TraceValue + FloatExt + Zero + One + MatrixOps + ReshapeOps> LinearOp<V> for LinearRematerializeOp<V> {
    fn transpose(&self, output_cotangents: &[LinearTerm<V>]) -> Result<Vec<Option<LinearTerm<V>>>, TraceError> {
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
/// pullback programs.
pub fn make_linear_rematerialize<V>(body: &FlatTracedRematerialize<V>) -> Result<LinearRematerializeOp<V>, TraceError>
where
    V: TraceValue
        + FloatExt
        + crate::tracing_v2::Zero
        + Zero
        + One
        + MatrixOps
        + ReshapeOps
        + std::ops::Add<Output = V>
        + std::ops::Mul<Output = V>
        + std::ops::Neg<Output = V>,
{
    let pushforward = linearize_program(body.compiled.program())?;
    let pullback = transpose_linear_program(&pushforward)?;
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
    V: TraceValue + FloatExt + Zero + One + crate::tracing_v2::ConcreteTraceValue + MatrixOps + ReshapeOps,
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
    V: TraceValue + FloatExt + Zero + One + MatrixOps + ReshapeOps,
    Input: Parameterized<Self, ParameterStructure: Clone>,
    Output: Parameterized<Self, ParameterStructure: Clone>,
> RematerializeInvocationLeaf<Input, Output> for JitTracer<V>
where
    Input::Family: ParameterizedFamily<V>,
    Output::Family: ParameterizedFamily<Self> + ParameterizedFamily<V>,
    Input::To<V>: Parameterized<V, To<JitTracer<V>> = Input>,
    Output::To<V>: Parameterized<V, To<JitTracer<V>> = Output>,
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

        let (exemplar_outputs, body_program): (Output::To<V>, Program<V, Input::To<V>, Output::To<V>>) =
            try_trace_program::<_, Input::To<V>, Output::To<V>, V>(
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
            .map(|id| body_program.graph().atom(*id).expect("body input atom should exist").abstract_value.clone())
            .collect::<Vec<_>>();
        let output_types = exemplar_outputs.parameters().map(Typed::tpe).collect::<Vec<_>>();
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
            PrimitiveOp::Rematerialize(Box::new(RematerializeOp::new(body))),
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
///
/// // Without rematerialize, compile_grad saves all forward intermediates.
/// // With rematerialize, the body is recomputed during the backward pass.
/// let (_, grad_fn) = compile_grad(|x: f64| rematerialize(|y| y.sin(), x).unwrap(), 1.0)?;
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
        CompiledFunction, FloatExt, JitTracer, jit,
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
        let (output, compiled): (f64, CompiledFunction<f64, f64, f64>) =
            jit(|x: JitTracer<f64>| rematerialize(|y: JitTracer<f64>| y.sin(), x).unwrap(), 2.0f64).unwrap();

        approx_eq(output, 2.0f64.sin());
        let ir = compiled.to_string();
        assert!(ir.contains("rematerialize"), "jit graph should contain the rematerialize op: {ir}");
    }

    #[test]
    fn test_rematerialize_jit_graph_rendering() {
        // Check the exact rendering of the jit-traced graph containing a rematerialize op.
        let (_, compiled): (f64, CompiledFunction<f64, f64, f64>) =
            jit(|x: JitTracer<f64>| rematerialize(|y: JitTracer<f64>| y.sin(), x).unwrap(), 2.0f64).unwrap();

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
        let gradient: f64 =
            grad(|x: JitTracer<f64>| rematerialize(|y: JitTracer<f64>| y.sin(), x).unwrap(), 2.0f64).unwrap();

        approx_eq(gradient, 2.0f64.cos());
    }

    #[test]
    fn test_rematerialize_value_and_grad_returns_both() {
        // value_and_grad of rematerialize(sin, x) should give (sin(x), cos(x)).
        let (value, gradient): (f64, f64) =
            value_and_grad(|x: JitTracer<f64>| rematerialize(|y: JitTracer<f64>| y.sin(), x).unwrap(), 2.0f64).unwrap();

        approx_eq(value, 2.0f64.sin());
        approx_eq(gradient, 2.0f64.cos());
    }

    #[test]
    fn test_rematerialize_compile_grad_produces_reusable_gradient() {
        // compile_grad with rematerialize should produce a symbolic gradient program.
        let compiled =
            compile_grad(|x: JitTracer<f64>| rematerialize(|y: JitTracer<f64>| y.sin(), x).unwrap(), 2.0f64).unwrap();

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
        let gradient: f64 = grad(
            |x: JitTracer<f64>| rematerialize(|y: JitTracer<f64>| y.clone() * y.clone() + y.sin(), x).unwrap(),
            2.0f64,
        )
        .unwrap();

        approx_eq(gradient, 2.0 * 2.0 + 2.0f64.cos());
    }

    #[test]
    fn test_rematerialize_compile_grad_quadratic_plus_sin() {
        // compile_grad with rematerialize wrapping a multi-op body.
        let compiled = compile_grad(
            |x: JitTracer<f64>| rematerialize(|y: JitTracer<f64>| y.clone() * y.clone() + y.sin(), x).unwrap(),
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
            let (output, _): (f64, CompiledFunction<f64, f64, f64>) =
                jit(|x: JitTracer<f64>| x.clone() * x.clone() + x.sin(), 3.0f64).unwrap();
            output
        };
        let with: f64 = {
            let (output, _): (f64, CompiledFunction<f64, f64, f64>) = jit(
                |x: JitTracer<f64>| rematerialize(|y: JitTracer<f64>| y.clone() * y.clone() + y.sin(), x).unwrap(),
                3.0f64,
            )
            .unwrap();
            output
        };

        approx_eq(without, with);
    }
}
