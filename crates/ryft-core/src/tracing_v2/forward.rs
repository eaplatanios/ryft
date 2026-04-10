//! Forward-mode differentiation primitives.
//!
//! The core value in this module is [`JvpTracer`], which carries both a primal value and a tangent value. Primitive
//! operations implement their JVP rules in [`crate::tracing_v2::ops`], while this module provides the user-facing
//! wrapper type and the `jvp` transform itself.

use std::ops::{Add, Mul, Neg};

use crate::{
    parameters::{Parameter, Parameterized, ParameterizedFamily},
    tracing_v2::{
        FloatExt, MatrixOps, TraceError, TraceValue, TransformLeaf, ZeroLike,
        jit::JitTracer,
        linear::{LinearProgram, jvp_program, try_jvp_traced},
        operations::{AddOp, CosOp, MulOp, NegOp, SinOp},
        ops::JvpOp,
    },
    types::{ArrayType, Typed},
};

/// Tangent representation for a traced primal value.
///
/// The default implementation is the primal type itself, but transforms such as `linearize` replace tangents with a
/// staged linear representation like [`crate::tracing_v2::LinearTerm`].
pub trait TangentSpace<V: TraceValue>: Clone + Parameter {
    /// Adds two tangent values.
    fn add(lhs: Self, rhs: Self) -> Self;

    /// Negates a tangent value.
    fn neg(value: Self) -> Self;

    /// Scales a tangent by a primal value.
    fn scale(factor: V, tangent: Self) -> Self;

    /// Produces a zero tangent matching the primal shape.
    fn zero_like(primal: &V, tangent: &Self) -> Self;
}

impl<V: TraceValue + FloatExt + ZeroLike> TangentSpace<V> for V {
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

    #[inline]
    fn zero_like(primal: &V, _tangent: &Self) -> Self {
        primal.zero_like()
    }
}

/// Forward-mode tracer carrying both a primal and a tangent.
#[derive(Clone, Debug)]
pub struct JvpTracer<V: TraceValue, T: TangentSpace<V>> {
    /// The primal value.
    pub primal: V,
    /// The tangent value associated with the primal.
    pub tangent: T,
}

impl<V: TraceValue, T: TangentSpace<V>> Parameter for JvpTracer<V, T> {}

impl<V: TraceValue, T: TangentSpace<V> + 'static> Typed<ArrayType> for JvpTracer<V, T> {
    #[inline]
    fn tpe(&self) -> ArrayType {
        <V as Typed<ArrayType>>::tpe(&self.primal)
    }
}

impl<V: TraceValue, T: TangentSpace<V> + 'static> TraceValue for JvpTracer<V, T> {}

impl<V: TraceValue + ZeroLike, T: TangentSpace<V>> ZeroLike for JvpTracer<V, T> {
    #[inline]
    fn zero_like(&self) -> Self {
        Self { primal: self.primal.zero_like(), tangent: T::zero_like(&self.primal, &self.tangent) }
    }
}

impl<V: TraceValue + crate::tracing_v2::OneLike, T: TangentSpace<V>> crate::tracing_v2::OneLike for JvpTracer<V, T> {
    #[inline]
    fn one_like(&self) -> Self {
        Self { primal: self.primal.one_like(), tangent: T::zero_like(&self.primal, &self.tangent) }
    }
}

/// Standard dual number representation used for first-order forward-mode evaluation.
pub type Dual<V> = JvpTracer<V, V>;

/// Dispatch trait used by [`jvp`] so it can operate both on concrete values and on already traced values.
#[doc(hidden)]
pub trait JvpInvocationLeaf<
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq>,
    Output: Parameterized<Self, ParameterStructure: Clone>,
>: Parameter + Sized
{
    /// Base leaf value used for the staged inner program.
    type Base: TraceValue + FloatExt + ZeroLike + MatrixOps;

    /// Input type expected by the user-provided function.
    type FunctionInput;

    /// Output type produced by the user-provided function.
    type FunctionOutput;

    /// Invokes [`jvp`] for one leaf regime.
    fn invoke<F>(function: F, primals: Input, tangents: Input) -> Result<(Output, Output), TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> Self::FunctionOutput;
}

fn single_output<V, T>(mut outputs: Vec<JvpTracer<V, T>>, op: &'static str) -> JvpTracer<V, T>
where
    V: TraceValue,
    T: TangentSpace<V>,
{
    debug_assert_eq!(outputs.len(), 1, "{op} should produce a single JVP output");
    outputs.pop().expect("single-output primitive should return one JVP output")
}

impl<V: TraceValue + Add<Output = V>, T: TangentSpace<V>> Add for JvpTracer<V, T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        single_output(AddOp.jvp(&[self, rhs]).expect("add JVP rule should succeed"), "add")
    }
}

impl<V: TraceValue + Mul<Output = V>, T: TangentSpace<V>> Mul for JvpTracer<V, T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        single_output(MulOp.jvp(&[self, rhs]).expect("mul JVP rule should succeed"), "mul")
    }
}

impl<V: TraceValue + Neg<Output = V>, T: TangentSpace<V>> Neg for JvpTracer<V, T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        single_output(NegOp.jvp(&[self]).expect("neg JVP rule should succeed"), "neg")
    }
}

impl<V: TraceValue + FloatExt, T: TangentSpace<V>> FloatExt for JvpTracer<V, T> {
    #[inline]
    fn sin(self) -> Self {
        single_output(SinOp.jvp(&[self]).expect("sin JVP rule should succeed"), "sin")
    }

    #[inline]
    fn cos(self) -> Self {
        single_output(CosOp.jvp(&[self]).expect("cos JVP rule should succeed"), "cos")
    }
}

impl<
    V: TransformLeaf,
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq>,
    Output: Parameterized<Self, ParameterStructure: Clone>,
> JvpInvocationLeaf<Input, Output> for V
where
    Input::Family: ParameterizedFamily<JitTracer<V>>,
    Output::Family: ParameterizedFamily<JitTracer<V>>,
{
    type Base = V;
    type FunctionInput = Input::To<JitTracer<V>>;
    type FunctionOutput = Output::To<JitTracer<V>>;

    fn invoke<F>(function: F, primals: Input, tangents: Input) -> Result<(Output, Output), TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> Self::FunctionOutput,
    {
        if primals.parameter_structure() != tangents.parameter_structure() {
            return Err(TraceError::MismatchedParameterStructure);
        }

        let (primal_output, tangent_program): (Output, LinearProgram<V, Input, Output>) =
            jvp_program(function, primals)?;
        let tangent_output = tangent_program.call(tangents)?;
        Ok((primal_output, tangent_output))
    }
}

impl<
    V: TransformLeaf,
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq>,
    Output: Parameterized<Self, ParameterStructure: Clone>,
> JvpInvocationLeaf<Input, Output> for JitTracer<V>
where
    Input::Family: ParameterizedFamily<V>,
    Output::Family: ParameterizedFamily<V>,
    Input::To<V>: Parameterized<V, To<JitTracer<V>> = Input>,
    Output::To<V>: Parameterized<V, To<JitTracer<V>> = Output>,
{
    type Base = V;
    type FunctionInput = Input;
    type FunctionOutput = Output;

    fn invoke<F>(function: F, primals: Input, tangents: Input) -> Result<(Output, Output), TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> Self::FunctionOutput,
    {
        try_jvp_traced(|input| Ok(function(input)), primals, tangents)
    }
}

/// Evaluates `function` on `primals` and propagates the supplied tangent values forward.
///
/// The returned pair is `(primal_output, tangent_output)`.
#[allow(private_bounds, private_interfaces)]
pub fn jvp<F, Input, Output, Leaf>(function: F, primals: Input, tangents: Input) -> Result<(Output, Output), TraceError>
where
    Leaf: JvpInvocationLeaf<Input, Output>,
    Input: Parameterized<Leaf, ParameterStructure: Clone + PartialEq>,
    Output: Parameterized<Leaf, ParameterStructure: Clone>,
    F: FnOnce(
        <Leaf as JvpInvocationLeaf<Input, Output>>::FunctionInput,
    ) -> <Leaf as JvpInvocationLeaf<Input, Output>>::FunctionOutput,
{
    Leaf::invoke(function, primals, tangents)
}

#[cfg(test)]
mod tests {
    use crate::tracing_v2::{OneLike, test_support};

    use super::*;

    #[test]
    fn dual_zero_like_zeros_the_tangent_component() {
        let dual = JvpTracer { primal: 3.0f64, tangent: 4.0f64 };
        let zero = dual.zero_like();
        assert_eq!(zero.primal, 0.0);
        assert_eq!(zero.tangent, 0.0);

        let ones = dual.one_like();
        assert_eq!(ones.primal, 1.0);
        assert_eq!(ones.tangent, 0.0);
        test_support::assert_quadratic_pushforward_rendering();
    }

    #[test]
    fn jvp_rejects_mismatched_parameter_structures() {
        let result: Result<(f64, f64), TraceError> =
            jvp(|xs: Vec<JitTracer<f64>>| xs[0].clone(), vec![2.0f64], vec![1.0f64, 2.0f64]);
        assert!(matches!(result, Err(TraceError::MismatchedParameterStructure)));
        test_support::assert_quadratic_pushforward_rendering();
    }
}
