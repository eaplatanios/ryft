//! Forward-mode differentiation primitives.
//!
//! The core value in this module is [`JvpTracer`], which carries both a primal value and a tangent value. Primitive
//! operations implement their JVP rules in [`crate::tracing_v2::operations`], while this module provides the user-facing
//! wrapper type and the `jvp` transform itself.

use std::{
    borrow::Cow,
    ops::{Add, Mul, Neg},
};

use crate::{
    parameters::{Parameter, Parameterized, ParameterizedFamily, Placeholder},
    tracing_v2::{
        LinearProgramOpRef, Program, TraceError, Traceable, Value, ZeroLike,
        batch::{Batch, stack, unstack},
        engine::Engine,
        jit::{JitTracer, trace_program},
        linear::{LinearProgram, Linearized, jvp_program, jvp_traced, linearize_traced_program},
        operations::{CoreLinearReplayOp, DifferentiableOp, InterpretableOp, Op},
    },
    types::{ArrayType, Type, Typed},
};

/// Tangent representation for a traced primal value.
///
/// The default implementation is the primal type itself, but transforms such as [`jvp_program`] replace tangents
/// with a staged linear representation like [`crate::tracing_v2::LinearTerm`].
///
/// [`jvp_program`]: crate::tracing_v2::jvp_program
pub trait TangentSpace<T: Type, V: Typed<T>>: Clone + Parameter {
    /// Adds two tangent values.
    fn add(lhs: Self, rhs: Self) -> Self;

    /// Negates a tangent value.
    fn neg(value: Self) -> Self;

    /// Scales a tangent by a primal value.
    fn scale(factor: V, tangent: Self) -> Self;

    /// Produces a zero tangent matching the primal shape.
    fn zero_like(primal: &V, tangent: &Self) -> Self;
}

impl<V: Traceable<ArrayType> + Add<Output = V> + Mul<Output = V> + Neg<Output = V> + ZeroLike>
    TangentSpace<ArrayType, V> for V
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

impl<V: Traceable<ArrayType>, T: TangentSpace<ArrayType, V>> Parameter for JvpTracer<V, T> {}

impl<V: Traceable<ArrayType>, T: TangentSpace<ArrayType, V> + 'static> Typed<ArrayType> for JvpTracer<V, T> {
    #[inline]
    fn tpe(&self) -> Cow<'_, ArrayType> {
        <V as Typed<ArrayType>>::tpe(&self.primal)
    }
}

impl<V: Traceable<ArrayType>, T: TangentSpace<ArrayType, V> + 'static> Traceable<ArrayType> for JvpTracer<V, T> {}

impl<V: Traceable<ArrayType> + ZeroLike, T: TangentSpace<ArrayType, V>> ZeroLike for JvpTracer<V, T> {
    #[inline]
    fn zero_like(&self) -> Self {
        Self { primal: self.primal.zero_like(), tangent: T::zero_like(&self.primal, &self.tangent) }
    }
}

impl<V: Traceable<ArrayType> + crate::tracing_v2::OneLike, T: TangentSpace<ArrayType, V>> crate::tracing_v2::OneLike
    for JvpTracer<V, T>
{
    #[inline]
    fn one_like(&self) -> Self {
        Self { primal: self.primal.one_like(), tangent: T::zero_like(&self.primal, &self.tangent) }
    }
}

/// Standard dual number representation used for first-order forward-mode evaluation.
pub type Dual<V> = JvpTracer<V, V>;

/// Dispatch trait used by [`jvp`] so it can operate both on concrete values and on already traced values.
#[doc(hidden)]
pub trait JvpInvocationLeaf<E, Input, Output>: Parameter + Sized
where
    E: Engine<Type = ArrayType>,
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq>,
    Output: Parameterized<Self, ParameterStructure: Clone>,
{
    /// Input type expected by the user-provided function.
    type FunctionInput;

    /// Output type produced by the user-provided function.
    type FunctionOutput;

    /// Invokes [`jvp`] for one leaf regime.
    fn invoke<F>(engine: &E, function: F, primals: Input, tangents: Input) -> Result<(Output, Output), TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> Self::FunctionOutput;
}

impl<V: Traceable<ArrayType> + Add<Output = V> + ZeroLike, T: TangentSpace<ArrayType, V>> Add for JvpTracer<V, T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self { primal: self.primal + rhs.primal, tangent: T::add(self.tangent, rhs.tangent) }
    }
}

impl<V: Traceable<ArrayType> + Mul<Output = V> + ZeroLike, T: TangentSpace<ArrayType, V>> Mul for JvpTracer<V, T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            primal: self.primal.clone() * rhs.primal.clone(),
            tangent: T::add(T::scale(rhs.primal, self.tangent), T::scale(self.primal, rhs.tangent)),
        }
    }
}

impl<V: Traceable<ArrayType> + Neg<Output = V> + ZeroLike, T: TangentSpace<ArrayType, V>> Neg for JvpTracer<V, T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self { primal: -self.primal, tangent: T::neg(self.tangent) }
    }
}

/// Concrete-value dispatch for [`jvp`]: traces the user function with [`JitTracer`] to build a staged
/// pushforward via [`jvp_program`] and evaluates it at the supplied tangents.
impl<
    E,
    V: Value<ArrayType> + ZeroLike + Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq>,
    Output: Parameterized<Self, ParameterStructure: Clone>,
> JvpInvocationLeaf<E, Input, Output> for V
where
    E: Engine<Type = ArrayType, Value = V>,
    Input::Family: ParameterizedFamily<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
    Output::Family: ParameterizedFamily<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
    E::TracingOperation: InterpretableOp<ArrayType, V>,
    E::TracingOperation: DifferentiableOp<
            ArrayType,
            V,
            crate::tracing_v2::LinearTerm<ArrayType, V, E::LinearOperation>,
            E::TracingOperation,
            E::LinearOperation,
        >,
    E::LinearOperation: InterpretableOp<ArrayType, V> + Op<ArrayType>,
{
    type FunctionInput = Input::To<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>;
    type FunctionOutput = Output::To<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>;

    fn invoke<F>(engine: &E, function: F, primals: Input, tangents: Input) -> Result<(Output, Output), TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> Self::FunctionOutput,
    {
        if primals.parameter_structure() != tangents.parameter_structure() {
            return Err(TraceError::MismatchedParameterStructure);
        }

        let (primal_output, tangent_program): (Output, LinearProgram<ArrayType, V, Input, Output, E::LinearOperation>) =
            jvp_program(engine, |input| Ok(function(input)), primals)?;
        let tangent_output = tangent_program.call(tangents)?;
        Ok((primal_output, tangent_output))
    }
}

/// Already-traced dispatch for [`jvp`]: delegates to [`jvp_traced`] to replay the user function
/// symbolically inside an enclosing [`JitTracer`] scope, staging both the primal output and the
/// tangent propagation as part of the outer compiled program.
impl<
    E,
    V: Traceable<ArrayType> + ZeroLike + Parameterized<V, ParameterStructure = Placeholder>,
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq, To<Self> = Input>,
    Output: Parameterized<Self, ParameterStructure: Clone, To<Self> = Output>,
    O: Clone + Op<ArrayType> + 'static,
    L: Clone + Op<ArrayType> + 'static,
> JvpInvocationLeaf<E, Input, Output> for JitTracer<ArrayType, V, O, L>
where
    Input::Family: ParameterizedFamily<V> + ParameterizedFamily<ArrayType>,
    Output::Family: ParameterizedFamily<V> + ParameterizedFamily<ArrayType>,
    Input::To<ArrayType>: Parameterized<ArrayType, To<JitTracer<ArrayType, V, O, L>> = Input>,
    Output::To<ArrayType>: Parameterized<ArrayType, To<JitTracer<ArrayType, V, O, L>> = Output>,
    O: InterpretableOp<
            ArrayType,
            Linearized<JitTracer<ArrayType, V, O, L>, LinearProgramOpRef<JitTracer<ArrayType, V, O, L>>>,
        >,
    LinearProgramOpRef<JitTracer<ArrayType, V, O, L>>: CoreLinearReplayOp<JitTracer<ArrayType, V, O, L>>,
    E: Engine<Type = ArrayType, TracingOperation = O, LinearOperation = L>,
{
    type FunctionInput = Input;
    type FunctionOutput = Output;

    fn invoke<F>(_engine: &E, function: F, primals: Input, tangents: Input) -> Result<(Output, Output), TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> Self::FunctionOutput,
    {
        jvp_traced::<_, _, _, V, O, L>(|input| Ok(function(input)), primals, tangents)
    }
}

/// Batched dispatch for [`jvp`], enabling standalone `vmap(|x| jvp(f, x, dx), inputs)` -- computing
/// per-element Jacobian-vector products over a batch without requiring an outer [`trace_program`] wrapper.
///
/// Uses the same trace-once strategy as the reverse-mode batched dispatch for [`Batch`]: the user
/// function is traced once to a [`Program`], and a second [`Program`] that
/// takes primals and tangents and returns `(primal_output, tangent_output)` per lane is compiled via
/// [`trace_program`]. Primal and tangent outputs are collected per lane and stacked separately.
impl<
    E,
    V: Traceable<ArrayType> + ZeroLike + Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input: Parameterized<Batch<V>, ParameterStructure: Clone + PartialEq>,
    Output: Parameterized<Batch<V>, ParameterStructure: Clone + PartialEq>,
> JvpInvocationLeaf<E, Input, Output> for Batch<V>
where
    E: Engine<Type = ArrayType, Value = V>,
    Vec<V>: Parameterized<
            V,
            ParameterStructure = Vec<Placeholder>,
            To<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>> = Vec<
                JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
            >,
        >,
    Input::Family:
        ParameterizedFamily<V> + ParameterizedFamily<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
    Output::Family:
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
    Output::To<V>: Clone
        + Parameterized<
            V,
            ParameterStructure: Clone + PartialEq,
            To<Batch<V>> = Output,
            To<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>> = Output::To<
                JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
            >,
        >,
    <Vec<V> as Parameterized<V>>::Family:
        ParameterizedFamily<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
    V::Family: ParameterizedFamily<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
    E::TracingOperation: Op<ArrayType>,
    E::TracingOperation: InterpretableOp<ArrayType, V>,
    E::TracingOperation: InterpretableOp<
            ArrayType,
            Linearized<
                JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
                LinearProgramOpRef<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
            >,
        >,
    LinearProgramOpRef<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>:
        CoreLinearReplayOp<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
{
    type FunctionInput = Input::To<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>;
    type FunctionOutput = Output::To<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>;

    fn invoke<F>(engine: &E, function: F, primals: Input, tangents: Input) -> Result<(Output, Output), TraceError>
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
        let (primal_output_0, traced_program): (
            Output::To<V>,
            Program<ArrayType, V, Input::To<V>, Output::To<V>, E::TracingOperation>,
        ) = trace_program(
            engine,
            |staged_input| {
                let staged_input: Self::FunctionInput = staged_input;
                let staged_output: Self::FunctionOutput = function(staged_input);
                Ok(staged_output)
            },
            lane0_primals,
        )?;

        let output_structure = primal_output_0.parameter_structure();
        let output_parameter_count = output_structure.parameter_count();

        // Reshape to flat Vec program for the JIT compilation step.
        let flat_program = traced_program
            .clone_with_structures::<Vec<V>, Vec<V>>(
                vec![Placeholder; input_parameter_count],
                vec![Placeholder; output_parameter_count],
            )
            .simplify()?;

        // Compile the full JVP into a reusable program. Inside the JIT scope, the program is
        // replayed symbolically with `linearize_traced_program`, which produces both the
        // primal outputs and a pushforward parameterized over the JIT-symbolic primals.
        let combined_input_count = input_parameter_count * 2;
        let combined_output_count = output_parameter_count * 2;

        let (_, compiled_jvp): (Vec<V>, Program<ArrayType, V, Vec<V>, Vec<V>, E::TracingOperation>) = trace_program(
            engine,
            |jit_combined| {
                let (jit_primals, jit_tangents) = jit_combined.split_at(input_parameter_count);

                // Replay the forward pass symbolically and linearize at the symbolic primals.
                let (primal_outputs, pushforward) = linearize_traced_program::<
                    V,
                    E::TracingOperation,
                    E::LinearOperation,
                >(&flat_program, jit_primals.to_vec())?;

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
            let combined_result: Vec<V> = compiled_jvp.call(combined_flat)?;
            let (primal_flat, tangent_flat) = combined_result.split_at(output_parameter_count);
            let primal_flat: Vec<V> = primal_flat.to_vec();
            let tangent_flat: Vec<V> = tangent_flat.to_vec();
            lane_primal_outputs.push(
                Output::To::<V>::from_parameters(output_structure.clone(), primal_flat).map_err(TraceError::from)?,
            );
            lane_tangent_outputs.push(
                Output::To::<V>::from_parameters(output_structure.clone(), tangent_flat).map_err(TraceError::from)?,
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
pub fn jvp<E, F, Input, Output, Leaf>(
    engine: &E,
    function: F,
    primals: Input,
    tangents: Input,
) -> Result<(Output, Output), TraceError>
where
    E: Engine<Type = ArrayType>,
    Leaf: JvpInvocationLeaf<E, Input, Output>,
    Input: Parameterized<Leaf, ParameterStructure: Clone + PartialEq>,
    Output: Parameterized<Leaf, ParameterStructure: Clone>,
    F: FnOnce(
        <Leaf as JvpInvocationLeaf<E, Input, Output>>::FunctionInput,
    ) -> <Leaf as JvpInvocationLeaf<E, Input, Output>>::FunctionOutput,
{
    Leaf::invoke(engine, function, primals, tangents)
}

#[cfg(test)]
mod tests {
    use crate::tracing_v2::{OneLike, engine::ArrayScalarEngine, test_support};

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
        let engine = ArrayScalarEngine::<f64>::new();
        let result: Result<(f64, f64), TraceError> =
            jvp(&engine, |xs: Vec<JitTracer<ArrayType, f64>>| xs[0].clone(), vec![2.0f64], vec![1.0f64, 2.0f64]);
        assert!(matches!(result, Err(TraceError::MismatchedParameterStructure)));
        test_support::assert_quadratic_pushforward_rendering();
    }
}
