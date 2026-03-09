//! Just-in-time staging support for `tracing_v2`.
//!
//! The current `jit` transform captures a graph of staged primitive applications and replays that graph with the
//! built-in interpreter. This keeps the API shape close to the eventual compiled-backend design while remaining easy
//! to test in pure Rust.

use std::{
    cell::RefCell,
    fmt::Display,
    marker::PhantomData,
    ops::{Add, Mul, Neg},
    rc::Rc,
    sync::Arc,
};

use ryft_macros::Parameter;

use crate::{
    parameters::{Parameter, Parameterized, ParameterizedFamily},
    tracing_v2::{
        FloatExt, JitContext, OneLike, TraceError, TraceLeaf, TraceValue, ZeroLike,
        graph::{AtomId, Graph, GraphBuilder},
        ops::{AddOp, CosOp, MulOp, NegOp, SinOp, StagedOpRef},
    },
};

/// Tracer used while staging JIT programs.
#[derive(Clone, Debug, Parameter)]
pub struct JitTracer<V>
where
    V: TraceValue,
{
    /// Concrete value obtained during eager execution of the staged computation.
    pub value: V,
    atom: AtomId,
    builder: Rc<RefCell<GraphBuilder<StagedOpRef<V>, V>>>,
}

impl<V> JitTracer<V>
where
    V: TraceValue,
{
    /// Returns the staged atom identifier corresponding to this traced value.
    #[inline]
    pub fn atom(&self) -> AtomId {
        self.atom
    }

    pub(crate) fn unary(self, op: StagedOpRef<V>, apply: impl FnOnce(V) -> V) -> Self {
        let value = apply(self.value);
        let atom = self
            .builder
            .borrow_mut()
            .add_equation(op, vec![self.atom])
            .expect("staging a unary op should succeed")[0];
        Self { value, atom, builder: self.builder }
    }

    pub(crate) fn binary(self, rhs: Self, op: StagedOpRef<V>, apply: impl FnOnce(V, V) -> V) -> Self {
        debug_assert!(Rc::ptr_eq(&self.builder, &rhs.builder));
        let value = apply(self.value, rhs.value);
        let atom = self
            .builder
            .borrow_mut()
            .add_equation(op, vec![self.atom, rhs.atom])
            .expect("staging a binary op should succeed")[0];
        Self { value, atom, builder: self.builder }
    }
}

impl<V> TraceLeaf for JitTracer<V>
where
    V: TraceValue,
{
    type Abstract = V::Abstract;

    #[inline]
    fn abstract_value(&self) -> Self::Abstract {
        self.value.abstract_value()
    }
}

impl<V> ZeroLike for JitTracer<V>
where
    V: TraceValue,
{
    #[inline]
    fn zero_like(&self) -> Self {
        let value = self.value.zero_like();
        let atom = self.builder.borrow_mut().add_constant(value.clone());
        Self { value, atom, builder: self.builder.clone() }
    }
}

impl<V> OneLike for JitTracer<V>
where
    V: TraceValue + OneLike,
{
    #[inline]
    fn one_like(&self) -> Self {
        let value = self.value.one_like();
        let atom = self.builder.borrow_mut().add_constant(value.clone());
        Self { value, atom, builder: self.builder.clone() }
    }
}

impl<V> Add for JitTracer<V>
where
    V: TraceValue,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.binary(rhs, Arc::new(AddOp), |left, right| left + right)
    }
}

impl<V> Mul for JitTracer<V>
where
    V: TraceValue,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.binary(rhs, Arc::new(MulOp), |left, right| left * right)
    }
}

impl<V> Neg for JitTracer<V>
where
    V: TraceValue,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        self.unary(Arc::new(NegOp), |value| -value)
    }
}

impl<V> FloatExt for JitTracer<V>
where
    V: TraceValue,
{
    #[inline]
    fn sin(self) -> Self {
        self.unary(Arc::new(SinOp), FloatExt::sin)
    }

    #[inline]
    fn cos(self) -> Self {
        self.unary(Arc::new(CosOp), FloatExt::cos)
    }
}

/// Staged function returned by [`jit`].
///
/// In the current prototype this type stores only the staged graph and replays it with the built-in interpreter.
/// Later, once a concrete backend exists, it can grow additional fields that hold backend-specific compiled artifacts
/// while keeping the same high-level API shape.
pub struct CompiledFunction<V, Input, Output>
where
    V: TraceValue,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    graph: Graph<StagedOpRef<V>, V, Input, Output>,
    marker: PhantomData<fn(Input) -> Output>,
}

impl<V, Input, Output> CompiledFunction<V, Input, Output>
where
    V: TraceValue,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    /// Returns the staged graph backing this compiled function.
    #[inline]
    pub fn graph(&self) -> &Graph<StagedOpRef<V>, V, Input, Output> {
        &self.graph
    }

    /// Replays the staged graph on concrete input values.
    ///
    /// The `_context` argument is currently unused by the interpreter-backed prototype. It is still threaded through
    /// the API so that future backend-backed implementations can access whatever runtime state they need at call time.
    pub fn call<Context>(&self, _context: &mut Context, input: Input) -> Result<Output, TraceError>
    where
        Input::ParameterStructure: PartialEq,
        Output::ParameterStructure: Clone,
    {
        self.graph.call(input)
    }
}

impl<V, Input, Output> Display for CompiledFunction<V, Input, Output>
where
    V: TraceValue,
    V::Abstract: Display,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.graph, f)
    }
}

/// Stages `function` as a graph and returns both the eager output and the staged program.
///
/// The returned [`CompiledFunction`] currently stores only the staged graph. Later, once a concrete backend exists,
/// this type can be extended to carry backend-specific compilation artifacts alongside that graph.
pub fn jit<'context, Context, F, Input, Output, V>(
    context: &'context mut Context,
    function: F,
    input: Input,
) -> Result<(Output, CompiledFunction<V, Input, Output>), TraceError>
where
    V: TraceValue,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<JitTracer<V>>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<JitTracer<V>>,
    F: FnOnce(&mut JitContext<'context, Context, V>, Input::To<JitTracer<V>>) -> Output::To<JitTracer<V>>,
{
    let input_structure = input.parameter_structure();
    let mut jit_context = JitContext::new(context);
    let traced_input = Input::To::<JitTracer<V>>::from_parameters(
        input_structure.clone(),
        input.into_parameters().map(|value| {
            let builder = jit_context.staged_builder();
            let atom = builder.borrow_mut().add_input(&value);
            JitTracer { value, atom, builder }
        }),
    )?;

    let (output_structure, output_value, outputs) = {
        let traced_output = function(&mut jit_context, traced_input);
        let output_structure = traced_output.parameter_structure();
        let traced_outputs = traced_output.into_parameters().collect::<Vec<_>>();
        let output_value = Output::from_parameters(
            output_structure.clone(),
            traced_outputs.iter().map(|output| output.value.clone()).collect::<Vec<_>>(),
        )?;
        let outputs = traced_outputs.into_iter().map(|output| output.atom).collect::<Vec<_>>();
        (output_structure, output_value, outputs)
    };

    let (_context, builder) = jit_context.finish();
    let builder = match Rc::try_unwrap(builder) {
        Ok(builder) => builder.into_inner(),
        Err(_) => return Err(TraceError::InternalInvariantViolation("jit builder escaped the tracing scope")),
    };
    let compiled = CompiledFunction {
        graph: builder.build::<Input, Output>(outputs, input_structure, output_structure),
        marker: PhantomData,
    };
    Ok((output_value, compiled))
}

#[cfg(test)]
mod tests {
    use indoc::indoc;

    use crate::{parameters::Placeholder, tracing_v2::test_support};

    use super::*;

    #[test]
    fn jit_tracer_zero_like_adds_constant_atoms() {
        let mut context = ();
        let jit_context = JitContext::<_, f64>::new(&mut context);
        let builder = jit_context.staged_builder();
        let atom = builder.borrow_mut().add_input(&3.0f64);
        let tracer = JitTracer { value: 3.0, atom, builder };
        let zero = tracer.zero_like();
        assert_eq!(zero.value, 0.0);
        assert!(zero.atom() > atom);

        let graph = zero.builder.borrow().clone().build::<f64, f64>(vec![zero.atom()], Placeholder, Placeholder);
        assert_eq!(
            graph.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = const
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn compiled_function_replays_staged_graphs() {
        let mut context = ();
        let (output, compiled): (f64, CompiledFunction<f64, f64, f64>) = jit(
            &mut context,
            |_, x: JitTracer<f64>| {
                let squared = x.clone() * x.clone();
                squared + x.sin()
            },
            2.0f64,
        )
        .unwrap();

        assert_eq!(output, 2.0f64 * 2.0f64 + 2.0f64.sin());
        assert_eq!(compiled.call(&mut context, 0.5f64).unwrap(), 0.5f64 * 0.5f64 + 0.5f64.sin());
        assert_eq!(compiled.graph().input_atoms().len(), 1);
        assert_eq!(
            compiled.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = mul %0 %0
                    %2:f64[] = sin %0
                    %3:f64[] = add %1 %2
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn compiled_function_display_delegates_to_the_underlying_graph() {
        let mut context = ();
        let (_, compiled): (f64, CompiledFunction<f64, f64, f64>) =
            jit(&mut context, |_, x: JitTracer<f64>| x.clone() * x.clone() + x.sin(), 2.0f64).unwrap();

        assert_eq!(
            compiled.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = mul %0 %0
                    %2:f64[] = sin %0
                    %3:f64[] = add %1 %2
                in (%3)
            "}
            .trim_end(),
        );
        assert_eq!(compiled.to_string(), compiled.graph().to_string());
        test_support::assert_bilinear_jit_rendering();
    }
}
