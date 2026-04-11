//! Forward-mode differentiation primitives.
//!
//! The core value in this module is [`JvpTracer`], which carries both a primal value and a tangent value. Primitive
//! operations implement their JVP rules in [`crate::tracing_v2::ops`], while this module provides the user-facing
//! wrapper type and the `jvp` transform itself.

use std::ops::{Add, Mul, Neg};

use crate::{
    parameters::{Parameter, Parameterized, ParameterizedFamily, Placeholder},
    tracing_v2::{
        FloatExt, MatrixOps, TraceError, TraceValue, ZeroLike,
        batch::{Batch, stack, unstack},
        jit::{CompiledFunction, JitTracer, try_jit, try_trace_program},
        linear::{LinearProgram, jvp_program, try_jvp_traced, try_linearize_traced_program},
        operations::{AddOp, CosOp, MulOp, NegOp, SinOp},
        ops::DifferentiableOp,
        program::Program,
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
///
/// The type parameters have no bounds on the struct itself so that `JvpTracer` can appear in
/// signatures (e.g., trait default methods) without propagating value-level bounds. The required
/// relationship `T: TangentSpace<V>` is enforced on the impl blocks that actually operate on the
/// values.
#[derive(Clone, Debug)]
pub struct JvpTracer<V, T> {
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

impl<V: TraceValue + Add<Output = V> + ZeroLike, T: TangentSpace<V>> Add for JvpTracer<V, T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        single_output(AddOp.jvp(&[self, rhs]).expect("add JVP rule should succeed"), "add")
    }
}

impl<V: TraceValue + Mul<Output = V> + ZeroLike, T: TangentSpace<V>> Mul for JvpTracer<V, T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        single_output(MulOp.jvp(&[self, rhs]).expect("mul JVP rule should succeed"), "mul")
    }
}

impl<V: TraceValue + Neg<Output = V> + ZeroLike, T: TangentSpace<V>> Neg for JvpTracer<V, T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        single_output(NegOp.jvp(&[self]).expect("neg JVP rule should succeed"), "neg")
    }
}

impl<V: TraceValue + FloatExt + ZeroLike, T: TangentSpace<V>> FloatExt for JvpTracer<V, T> {
    #[inline]
    fn sin(self) -> Self {
        single_output(SinOp.jvp(&[self]).expect("sin JVP rule should succeed"), "sin")
    }

    #[inline]
    fn cos(self) -> Self {
        single_output(CosOp.jvp(&[self]).expect("cos JVP rule should succeed"), "cos")
    }
}

/// Concrete-value dispatch for [`jvp`]: traces the user function with [`JitTracer`] to build a staged
/// pushforward via [`jvp_program`] and evaluates it at the supplied tangents.
impl<
    V: TraceValue + FloatExt + ZeroLike + crate::tracing_v2::OneLike + crate::tracing_v2::IdentityValue + MatrixOps + crate::tracing_v2::operations::reshape::ReshapeOps,
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

/// Already-traced dispatch for [`jvp`]: delegates to [`try_jvp_traced`] to replay the user function
/// symbolically inside an enclosing [`JitTracer`] scope, staging both the primal output and the
/// tangent propagation as part of the outer compiled graph.
impl<
    V: TraceValue + FloatExt + ZeroLike + crate::tracing_v2::OneLike + MatrixOps + crate::tracing_v2::operations::reshape::ReshapeOps,
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

/// Batched dispatch for [`jvp`], enabling standalone `vmap(|x| jvp(f, x, dx), inputs)` -- computing
/// per-element Jacobian-vector products over a batch without requiring an outer [`jit`] wrapper.
///
/// Uses the same trace-once strategy as [`GradInvocationLeaf`](crate::tracing_v2::GradInvocationLeaf)
/// for [`Batch`]: the user function is traced once to a [`Program`], and a [`CompiledFunction`] that
/// takes primals and tangents and returns `(primal_output, tangent_output)` per lane is compiled via
/// [`try_jit`]. Primal and tangent outputs are collected per lane and stacked separately.
impl<
    V: TraceValue + FloatExt + ZeroLike + crate::tracing_v2::OneLike + MatrixOps + crate::tracing_v2::operations::reshape::ReshapeOps,
    Input: Parameterized<Batch<V>, ParameterStructure: Clone + PartialEq>,
    Output: Parameterized<Batch<V>, ParameterStructure: Clone + PartialEq>,
> JvpInvocationLeaf<Input, Output> for Batch<V>
where
    Input::Family: ParameterizedFamily<V> + ParameterizedFamily<JitTracer<V>>,
    Output::Family: ParameterizedFamily<V> + ParameterizedFamily<JitTracer<V>>,
    Input::To<V>: Clone
        + Parameterized<
            V,
            ParameterStructure: Clone + PartialEq,
            To<Batch<V>> = Input,
            To<JitTracer<V>> = Input::To<JitTracer<V>>,
        >,
    Output::To<V>: Clone
        + Parameterized<
            V,
            ParameterStructure: Clone + PartialEq,
            To<Batch<V>> = Output,
            To<JitTracer<V>> = Output::To<JitTracer<V>>,
        >,
    <Input::To<V> as Parameterized<V>>::Family: ParameterizedFamily<JitTracer<V>> + ParameterizedFamily<Batch<V>>,
    <Output::To<V> as Parameterized<V>>::Family: ParameterizedFamily<JitTracer<V>> + ParameterizedFamily<Batch<V>>,
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

        let lane_primals: Vec<Input::To<V>> = unstack(primals)?;
        let lane_tangents: Vec<Input::To<V>> = unstack(tangents)?;
        if lane_primals.is_empty() {
            return Err(TraceError::EmptyBatch);
        }

        let lane0_primals = lane_primals[0].clone();
        let input_structure = lane0_primals.parameter_structure();
        let input_parameter_count = input_structure.parameter_count();

        // Trace the function once at lane 0 primals, consuming the FnOnce closure.
        let (primal_output_0, traced_program): (Output::To<V>, Program<V, Input::To<V>, Output::To<V>>) =
            try_trace_program(|staged_input| Ok(function(staged_input)), lane0_primals)?;

        let output_structure = primal_output_0.parameter_structure();
        let output_parameter_count = output_structure.parameter_count();

        // Reshape to flat Vec program for the JIT compilation step.
        let flat_program = Program::from_graph(traced_program.graph().clone_with_structures::<Vec<V>, Vec<V>>(
            vec![Placeholder; input_parameter_count],
            vec![Placeholder; output_parameter_count],
        ))
        .simplify()?;

        // Compile the full JVP into a reusable program. Inside the JIT scope, the program is
        // replayed symbolically with `try_linearize_traced_program`, which produces both the
        // primal outputs and a pushforward parameterized over the JIT-symbolic primals.
        let combined_input_count = input_parameter_count * 2;
        let combined_output_count = output_parameter_count * 2;

        let (_, compiled_jvp): (Vec<V>, CompiledFunction<V, Vec<V>, Vec<V>>) = try_jit(
            |jit_combined: Vec<JitTracer<V>>| {
                let (jit_primals, jit_tangents) = jit_combined.split_at(input_parameter_count);

                // Replay the forward pass symbolically and linearize at the symbolic primals.
                let (primal_outputs, pushforward) = try_linearize_traced_program(&flat_program, jit_primals.to_vec())?;

                // Apply the pushforward to the symbolic tangents.
                let tangent_outputs = pushforward.call(jit_tangents.to_vec())?;

                let mut result = Vec::with_capacity(combined_output_count);
                result.extend(primal_outputs);
                result.extend(tangent_outputs);
                Ok(result)
            },
            {
                let mut combined = Vec::with_capacity(combined_input_count);
                combined.extend(lane_primals[0].clone().into_parameters());
                combined.extend(lane_tangents[0].clone().into_parameters());
                combined
            },
        )?;

        // Apply per-lane and split into (primal_output, tangent_output).
        let mut lane_primal_outputs = Vec::with_capacity(lane_primals.len());
        let mut lane_tangent_outputs = Vec::with_capacity(lane_primals.len());
        for (lane_p, lane_t) in lane_primals.into_iter().zip(lane_tangents) {
            let mut combined_flat = Vec::with_capacity(combined_input_count);
            combined_flat.extend(lane_p.into_parameters());
            combined_flat.extend(lane_t.into_parameters());
            let combined_result = compiled_jvp.call(combined_flat)?;
            let (primal_flat, tangent_flat) = combined_result.split_at(output_parameter_count);
            lane_primal_outputs.push(
                Output::To::<V>::from_parameters(output_structure.clone(), primal_flat.to_vec())
                    .map_err(TraceError::from)?,
            );
            lane_tangent_outputs.push(
                Output::To::<V>::from_parameters(output_structure.clone(), tangent_flat.to_vec())
                    .map_err(TraceError::from)?,
            );
        }

        let primal_output = stack(lane_primal_outputs)?;
        let tangent_output = stack(lane_tangent_outputs)?;
        Ok((primal_output, tangent_output))
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
