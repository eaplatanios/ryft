use std::ops::{Add, Mul, Neg};

use crate::{
    parameters::{Parameter, Parameterized, ParameterizedFamily},
    tracing_v2::{
        FloatExt, OneLike, TraceError, TraceLeaf, TraceValue, ZeroLike,
        context::JvpContext,
        linear::{LinearProgram, Linearized, jvp_program},
        ops::{AddOp, CosOp, JvpOp, MulOp, NegOp, SinOp},
    },
};

pub trait TangentSpace<V>: Clone + Parameter
where
    V: TraceValue,
{
    fn add(lhs: Self, rhs: Self) -> Self;

    fn neg(value: Self) -> Self;

    fn scale(factor: V, tangent: Self) -> Self;
}

impl<V> TangentSpace<V> for V
where
    V: TraceValue,
{
    #[inline]
    fn add(lhs: Self, rhs: Self) -> Self {
        lhs + rhs
    }

    #[inline]
    fn neg(value: Self) -> Self {
        -value
    }

    #[inline]
    fn scale(factor: V, tangent: Self) -> Self {
        factor * tangent
    }
}

#[derive(Clone, Debug)]
pub struct JvpTracer<V, T>
where
    V: TraceValue,
    T: TangentSpace<V>,
{
    pub primal: V,
    pub tangent: T,
}

impl<V, T> Parameter for JvpTracer<V, T>
where
    V: TraceValue,
    T: TangentSpace<V>,
{
}

impl<V, T> TraceLeaf for JvpTracer<V, T>
where
    V: TraceValue,
    T: TangentSpace<V>,
{
    type Abstract = V::Abstract;

    #[inline]
    fn abstract_value(&self) -> Self::Abstract {
        self.primal.abstract_value()
    }
}

impl<V, T> ZeroLike for JvpTracer<V, T>
where
    V: TraceValue,
    T: TangentSpace<V>,
{
    #[inline]
    fn zero_like(&self) -> Self {
        let primal = self.primal.zero_like();
        let tangent = T::scale(primal.clone(), self.tangent.clone());
        Self { primal, tangent }
    }
}

impl<V, T> OneLike for JvpTracer<V, T>
where
    V: TraceValue + OneLike,
    T: TangentSpace<V>,
{
    #[inline]
    fn one_like(&self) -> Self {
        let primal = self.primal.one_like();
        let tangent = T::scale(self.primal.zero_like(), self.tangent.clone());
        Self { primal, tangent }
    }
}

pub type Dual<V> = JvpTracer<V, V>;

fn single_output<V, T>(mut outputs: Vec<JvpTracer<V, T>>, op: &'static str) -> JvpTracer<V, T>
where
    V: TraceValue,
    T: TangentSpace<V>,
{
    debug_assert_eq!(outputs.len(), 1, "{op} should produce a single JVP output");
    outputs.pop().expect("single-output primitive should return one JVP output")
}

impl<V, T> Add for JvpTracer<V, T>
where
    V: TraceValue,
    T: TangentSpace<V>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        single_output(AddOp.jvp(&[self, rhs]).expect("add JVP rule should succeed"), "add")
    }
}

impl<V, T> Mul for JvpTracer<V, T>
where
    V: TraceValue,
    T: TangentSpace<V>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        single_output(MulOp.jvp(&[self, rhs]).expect("mul JVP rule should succeed"), "mul")
    }
}

impl<V, T> Neg for JvpTracer<V, T>
where
    V: TraceValue,
    T: TangentSpace<V>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        single_output(NegOp.jvp(&[self]).expect("neg JVP rule should succeed"), "neg")
    }
}

impl<V, T> FloatExt for JvpTracer<V, T>
where
    V: TraceValue,
    T: TangentSpace<V>,
{
    #[inline]
    fn sin(self) -> Self {
        single_output(SinOp.jvp(&[self]).expect("sin JVP rule should succeed"), "sin")
    }

    #[inline]
    fn cos(self) -> Self {
        single_output(CosOp.jvp(&[self]).expect("cos JVP rule should succeed"), "cos")
    }
}

pub fn jvp<'context, Context, F, Input, Output, V>(
    context: &'context mut Context,
    function: F,
    primals: Input,
    tangents: Input,
) -> Result<(Output, Output), TraceError>
where
    V: TraceValue,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<Linearized<V>>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<Linearized<V>>,
    F: FnOnce(&mut JvpContext<'context, Context, V>, Input::To<Linearized<V>>) -> Output::To<Linearized<V>>,
{
    if primals.parameter_structure() != tangents.parameter_structure() {
        return Err(TraceError::MismatchedParameterStructure);
    }

    let (primal_output, tangent_program): (Output, LinearProgram<V, Input, Output>) =
        jvp_program(context, function, primals)?;
    let tangent_output = tangent_program.call(tangents)?;
    Ok((primal_output, tangent_output))
}
