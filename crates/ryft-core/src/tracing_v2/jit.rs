use std::{
    cell::RefCell,
    marker::PhantomData,
    ops::{Add, Mul, Neg},
    rc::Rc,
    sync::Arc,
};

use ryft_macros::Parameter;

use crate::{
    parameters::{Parameter, Parameterized, ParameterizedFamily},
    tracing_v2::{
        CompilationContext, FloatExt, JitContext, OneLike, TraceError, TraceLeaf, TraceValue, ZeroLike,
        graph::{AtomId, Graph, GraphBuilder},
        ops::{AddOp, CosOp, MulOp, NegOp, SinOp, StagedOpRef},
    },
};

#[derive(Clone, Debug, Parameter)]
pub struct JitTracer<V>
where
    V: TraceValue,
{
    pub value: V,
    atom: AtomId,
    builder: Rc<RefCell<GraphBuilder<StagedOpRef<V>, V>>>,
}

impl<V> JitTracer<V>
where
    V: TraceValue,
{
    #[inline]
    pub fn atom(&self) -> AtomId {
        self.atom
    }

    fn unary(self, op: StagedOpRef<V>, apply: impl FnOnce(V) -> V) -> Self {
        let value = apply(self.value);
        let atom = self
            .builder
            .borrow_mut()
            .add_equation(op, vec![self.atom])
            .expect("staging a unary op should succeed")[0];
        Self { value, atom, builder: self.builder }
    }

    fn binary(self, rhs: Self, op: StagedOpRef<V>, apply: impl FnOnce(V, V) -> V) -> Self {
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

pub struct CompiledFunction<V, Input, Output>
where
    V: TraceValue,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    id: usize,
    graph: Graph<StagedOpRef<V>, V, Input, Output>,
    marker: PhantomData<fn(Input) -> Output>,
}

impl<V, Input, Output> CompiledFunction<V, Input, Output>
where
    V: TraceValue,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    #[inline]
    pub fn id(&self) -> usize {
        self.id
    }

    #[inline]
    pub fn graph(&self) -> &Graph<StagedOpRef<V>, V, Input, Output> {
        &self.graph
    }

    pub fn call<Context>(&self, _context: &mut Context, input: Input) -> Result<Output, TraceError>
    where
        Input::ParameterStructure: PartialEq,
        Output::ParameterStructure: Clone,
    {
        self.graph.call(input)
    }
}

pub fn jit<'context, Context, F, Input, Output, V>(
    context: &'context mut Context,
    function: F,
    input: Input,
) -> Result<(Output, CompiledFunction<V, Input, Output>), TraceError>
where
    Context: CompilationContext,
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

    let (context, builder) = jit_context.finish();
    let builder = match Rc::try_unwrap(builder) {
        Ok(builder) => builder.into_inner(),
        Err(_) => return Err(TraceError::InternalInvariantViolation("jit builder escaped the tracing scope")),
    };
    let compiled = CompiledFunction {
        id: context.allocate_executable_id(),
        graph: builder.build::<Input, Output>(outputs, input_structure, output_structure),
        marker: PhantomData,
    };
    Ok((output_value, compiled))
}
