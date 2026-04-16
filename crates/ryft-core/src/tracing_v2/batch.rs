//! Vectorization support for `tracing_v2`.
//!
//! Concrete batching is still represented as explicit lane lists via [`Batch`]. For traced programs, however,
//! [`vmap`] stages a compact higher-order op instead of eagerly duplicating one scalar graph per lane. That keeps the
//! public batching surface unchanged while giving lowering enough structure to emit packed StableHLO that is much
//! closer to JAX's current Shardy output.

use std::ops::{Add, Mul, Neg};

use crate::{
    parameters::{Parameter, Parameterized, ParameterizedFamily, Placeholder},
    tracing_v2::{
        CompiledFunction, FloatExt, JitTracer, OneLike, Program, TraceError, Traceable, ZeroLike,
        operations::{AddOp, CosOp, FlatTracedVMap, MulOp, NegOp, SinOp, VMapOp},
        ops::{PrimitiveOp, VectorizableOp},
    },
    types::{ArrayType, Typed},
};
use ryft_macros::Parameter;

/// Batched leaf value represented as an explicit list of lanes.
#[derive(Clone, Debug, PartialEq, Parameter)]
pub struct Batch<V> {
    lanes: Vec<V>,
}

impl<V> Batch<V> {
    /// Creates a new batched value from a list of lanes.
    #[inline]
    pub fn new(lanes: Vec<V>) -> Self {
        Self { lanes }
    }

    /// Returns the number of lanes.
    #[inline]
    pub fn len(&self) -> usize {
        self.lanes.len()
    }

    /// Returns `true` when the batch contains no lanes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.lanes.is_empty()
    }

    /// Returns the lanes by shared reference.
    #[inline]
    pub fn lanes(&self) -> &[V] {
        self.lanes.as_slice()
    }

    /// Consumes `self` and returns the underlying lanes.
    #[inline]
    pub fn into_lanes(self) -> Vec<V> {
        self.lanes
    }
}

fn single_output<V>(mut outputs: Vec<Batch<V>>, op: &'static str) -> Batch<V> {
    debug_assert_eq!(outputs.len(), 1, "{op} should produce a single batched output");
    outputs.pop().expect("single-output primitive should return one batched output")
}

impl<V: Traceable<ArrayType> + Add<Output = V>> Add for Batch<V> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        single_output(AddOp.batch(&[self, rhs]).expect("add batching rule should succeed"), "add")
    }
}

impl<V: Traceable<ArrayType> + Mul<Output = V>> Mul for Batch<V> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        single_output(MulOp.batch(&[self, rhs]).expect("mul batching rule should succeed"), "mul")
    }
}

impl<V: Traceable<ArrayType> + Neg<Output = V>> Neg for Batch<V> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        single_output(NegOp.batch(&[self]).expect("neg batching rule should succeed"), "neg")
    }
}

impl<V: Traceable<ArrayType> + FloatExt> FloatExt for Batch<V> {
    #[inline]
    fn sin(self) -> Self {
        single_output(SinOp.batch(&[self]).expect("sin batching rule should succeed"), "sin")
    }

    #[inline]
    fn cos(self) -> Self {
        single_output(CosOp.batch(&[self]).expect("cos batching rule should succeed"), "cos")
    }
}

impl<V: Parameter + ZeroLike> ZeroLike for Batch<V> {
    #[inline]
    fn zero_like(&self) -> Self {
        Self::new(self.lanes.iter().map(ZeroLike::zero_like).collect())
    }
}

impl<V: Parameter + OneLike> OneLike for Batch<V> {
    #[inline]
    fn one_like(&self) -> Self {
        Self::new(self.lanes.iter().map(OneLike::one_like).collect())
    }
}

/// Stacks a list of structured inputs into one structured value whose leaves are [`Batch`] values.
pub fn stack<Input, V>(inputs: Vec<Input>) -> Result<Input::To<Batch<V>>, TraceError>
where
    V: Parameter,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<Batch<V>>,
{
    let mut inputs = inputs.into_iter();
    let first = inputs.next().ok_or(TraceError::EmptyBatch)?;
    let structure = first.parameter_structure();
    let parameter_count = structure.parameter_count();
    let mut buckets = (0..parameter_count).map(|_| Vec::new()).collect::<Vec<Vec<V>>>();
    let first_parameters = first.into_parameters().collect::<Vec<_>>();

    for (bucket, parameter) in buckets.iter_mut().zip(first_parameters) {
        bucket.push(parameter);
    }

    for input in inputs {
        if input.parameter_structure() != structure {
            return Err(TraceError::MismatchedParameterStructure);
        }

        for (bucket, parameter) in buckets.iter_mut().zip(input.into_parameters()) {
            bucket.push(parameter);
        }
    }

    Ok(Input::To::<Batch<V>>::from_parameters(structure, buckets.into_iter().map(Batch::new))?)
}

/// Splits a structured batch back into one structured value per lane.
pub fn unstack<Input, V>(batched: Input::To<Batch<V>>) -> Result<Vec<Input>, TraceError>
where
    V: Parameter,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<Batch<V>>,
{
    let structure = batched.parameter_structure();
    let batches = batched.into_parameters().collect::<Vec<_>>();
    if batches.is_empty() {
        return Ok(Vec::new());
    }

    let lane_count = batches[0].len();
    if batches.iter().any(|batch| batch.len() != lane_count) {
        return Err(TraceError::MismatchedBatchSize);
    }

    let mut lane_parameters = (0..lane_count).map(|_| Vec::with_capacity(batches.len())).collect::<Vec<Vec<V>>>();
    for batch in batches {
        for (lane_index, value) in batch.into_lanes().into_iter().enumerate() {
            lane_parameters[lane_index].push(value);
        }
    }

    lane_parameters
        .into_iter()
        .map(|parameters| Input::from_parameters(structure.clone(), parameters).map_err(TraceError::from))
        .collect()
}

/// Dispatch trait used by [`vmap`] so it can handle both concrete batches and already traced values.
#[doc(hidden)]
pub(crate) trait VMapInvocationLeaf<
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq>,
    Output: Parameterized<Self, ParameterStructure: Clone>,
>: Parameter + Sized
{
    /// Invokes [`vmap`] for one concrete leaf regime.
    fn invoke<F>(function: F, inputs: Vec<Input>) -> Result<Vec<Output>, TraceError>
    where
        Input::Family: ParameterizedFamily<Batch<Self>>,
        Output::Family: ParameterizedFamily<Batch<Self>>,
        F: FnOnce(Input::To<Batch<Self>>) -> Output::To<Batch<Self>>;
}

/// Concrete-value dispatch for [`vmap`]: stacks inputs into [`Batch`] leaves, applies the user function
/// over the batched representation, and unstacks the result back into per-lane outputs.
impl<
    V: Traceable<ArrayType>
        + Parameterized<V, ParameterStructure = Placeholder>
        + FloatExt
        + ZeroLike
        + OneLike
        + crate::tracing_v2::Value<ArrayType>
        + crate::tracing_v2::MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Output: Parameterized<V, ParameterStructure: Clone>,
> VMapInvocationLeaf<Input, Output> for V
where
    Input::Family: ParameterizedFamily<Batch<V>>,
    Output::Family: ParameterizedFamily<Batch<V>>,
{
    fn invoke<F>(function: F, inputs: Vec<Input>) -> Result<Vec<Output>, TraceError>
    where
        F: FnOnce(Input::To<Batch<Self>>) -> Output::To<Batch<Self>>,
    {
        let batched_input = stack(inputs)?;
        unstack(function(batched_input))
    }
}

/// Already-traced dispatch for [`vmap`]: stages a compact higher-order [`VMapOp`] in the enclosing
/// [`JitTracer`] scope instead of eagerly duplicating the scalar graph per lane. The body is traced
/// once at a single-lane exemplar and compiled into a [`CompiledFunction`] that lowering can later
/// emit as packed StableHLO.
impl<
    V: Traceable<ArrayType>
        + Parameterized<V, ParameterStructure = Placeholder>
        + FloatExt
        + ZeroLike
        + OneLike
        + crate::tracing_v2::MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps,
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq>,
    Output: Parameterized<Self, ParameterStructure: Clone>,
> VMapInvocationLeaf<Input, Output> for JitTracer<V>
where
    Input::Family: ParameterizedFamily<Batch<Self>> + ParameterizedFamily<V>,
    Output::Family: ParameterizedFamily<Batch<Self>> + ParameterizedFamily<Self> + ParameterizedFamily<V>,
    V: Parameterized<V, To<JitTracer<V>> = JitTracer<V>, ParameterStructure: Clone + PartialEq>,
    V::Family: ParameterizedFamily<JitTracer<V>>,
    Vec<V>: Parameterized<V, To<JitTracer<V>> = Vec<JitTracer<V>>, ParameterStructure = Vec<Placeholder>>,
    <Vec<V> as Parameterized<V>>::Family: ParameterizedFamily<JitTracer<V>>,
{
    fn invoke<F>(function: F, inputs: Vec<Input>) -> Result<Vec<Output>, TraceError>
    where
        F: FnOnce(Input::To<Batch<Self>>) -> Output::To<Batch<Self>>,
    {
        type LaneOutput<Output, Value> =
            <<Output as Parameterized<JitTracer<Value>>>::To<Value> as Parameterized<Value>>::To<JitTracer<Value>>;

        let mut inputs = inputs.into_iter();
        let first_input = inputs.next().ok_or(TraceError::EmptyBatch)?;
        let input_structure = first_input.parameter_structure();
        let mut traced_inputs = vec![first_input.into_parameters().collect::<Vec<_>>()];
        for input in inputs {
            if input.parameter_structure() != input_structure {
                return Err(TraceError::MismatchedParameterStructure);
            }
            traced_inputs.push(input.into_parameters().collect::<Vec<_>>());
        }

        let lane_count = traced_inputs.len();
        let input_leaf_count = input_structure.parameter_count();
        let exemplar_primals = Input::To::<V>::from_parameters(
            input_structure.clone(),
            traced_inputs[0].iter().map(|input| input.value.clone()).collect::<Vec<_>>(),
        )?;

        let (exemplar_outputs, body_program): (Output::To<V>, Program<V, Input::To<V>, Output::To<V>>) =
            crate::tracing_v2::jit::try_trace_program(
                |lane_inputs| {
                    let batched_inputs = Input::To::<Batch<JitTracer<V>>>::from_parameters(
                        lane_inputs.parameter_structure(),
                        lane_inputs.into_parameters().map(|input| Batch::new(vec![input])),
                    )?;
                    let batched_outputs = function(batched_inputs);
                    let output_structure = batched_outputs.parameter_structure();
                    let mut lane_outputs = Vec::new();
                    for batch in batched_outputs.into_parameters() {
                        let mut outputs = batch.into_lanes();
                        if outputs.len() != 1 {
                            return Err(TraceError::HigherOrderOpFailure {
                                op: "vmap",
                                message: "traced vmap only supports bodies that preserve the per-lane output structure"
                                    .to_string(),
                            });
                        }
                        lane_outputs.push(outputs.pop().expect("single-lane batches should contain one output"));
                    }
                    Ok(LaneOutput::<Output, V>::from_parameters(output_structure, lane_outputs)?)
                },
                exemplar_primals,
            )?;

        let output_structure = exemplar_outputs.parameter_structure();
        let output_leaf_count = output_structure.parameter_count();
        let flat_input_structure = vec![Placeholder; input_leaf_count];
        let flat_output_structure = vec![Placeholder; output_leaf_count];
        let body = FlatTracedVMap::from_parts(
            lane_count,
            body_program
                .graph()
                .input_atoms()
                .iter()
                .map(|input| {
                    body_program.graph().atom(*input).expect("body input atoms should exist").abstract_value.clone()
                })
                .collect::<Vec<_>>(),
            exemplar_outputs.parameters().map(Typed::tpe).collect::<Vec<_>>(),
            CompiledFunction::from_graph(
                body_program
                    .graph()
                    .clone_with_structures::<Vec<V>, Vec<V>>(flat_input_structure, flat_output_structure),
            ),
        );

        let output_values = traced_inputs
            .iter()
            .map(|lane_inputs| {
                body.compiled().call(lane_inputs.iter().map(|input| input.value.clone()).collect::<Vec<_>>())
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let staged_inputs = traced_inputs.into_iter().flatten().collect::<Vec<_>>();
        let staged_outputs = JitTracer::apply_staged_op(
            staged_inputs.as_slice(),
            PrimitiveOp::VMap(Box::new(VMapOp::new(body))),
            output_values,
        )?;
        (0..lane_count)
            .map(|lane_index| {
                let start = lane_index * output_leaf_count;
                let end = start + output_leaf_count;
                Output::from_parameters(output_structure.clone(), staged_outputs[start..end].iter().cloned())
                    .map_err(TraceError::from)
            })
            .collect()
    }
}

/// Nested-batch dispatch for [`vmap`], enabling `vmap(|xs| vmap(g, xs))` -- applying vectorization
/// recursively. This delegates to the concrete `V` implementation: the outer batch is unstacked,
/// the inner [`vmap`] runs per outer lane using the existing [`VMapInvocationLeaf`] for `V`, and
/// results are stacked back. No trace-once pattern is needed here because the delegation to the
/// concrete implementation handles each lane directly.
impl<
    V: Traceable<ArrayType>
        + Parameterized<V, ParameterStructure = Placeholder>
        + FloatExt
        + ZeroLike
        + OneLike
        + crate::tracing_v2::MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps,
    Input: Parameterized<Batch<V>, ParameterStructure: Clone + PartialEq>,
    Output: Parameterized<Batch<V>, ParameterStructure: Clone>,
> VMapInvocationLeaf<Input, Output> for Batch<V>
where
    Input::Family: ParameterizedFamily<Batch<Batch<V>>>,
    Output::Family: ParameterizedFamily<Batch<Batch<V>>>,
{
    fn invoke<F>(function: F, inputs: Vec<Input>) -> Result<Vec<Output>, TraceError>
    where
        Input::Family: ParameterizedFamily<Batch<Self>>,
        Output::Family: ParameterizedFamily<Batch<Self>>,
        F: FnOnce(Input::To<Batch<Self>>) -> Output::To<Batch<Self>>,
    {
        let batched_input = stack(inputs)?;
        unstack(function(batched_input))
    }
}

/// Maps `function` over a leading batch axis by stacking inputs, running the batched computation, and then
/// unstacking the result.
#[allow(private_bounds)]
pub fn vmap<F, Input, Output, V>(function: F, inputs: Vec<Input>) -> Result<Vec<Output>, TraceError>
where
    V: VMapInvocationLeaf<Input, Output>,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<Batch<V>>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<Batch<V>>,
    F: FnOnce(Input::To<Batch<V>>) -> Output::To<Batch<V>>,
{
    V::invoke(function, inputs)
}

#[cfg(test)]
mod tests {
    use indoc::indoc;

    use crate::tracing_v2::{JitTracer, test_support};

    use super::*;

    #[test]
    fn stack_and_unstack_round_trip_structured_values() {
        let batched = stack::<(f64, f64), f64>(vec![(1.0, 2.0), (3.0, 4.0)]).unwrap();
        assert_eq!(batched.0.lanes(), &[1.0, 3.0]);
        assert_eq!(batched.1.lanes(), &[2.0, 4.0]);

        let unstacked = unstack::<(f64, f64), f64>(batched).unwrap();
        assert_eq!(unstacked, vec![(1.0, 2.0), (3.0, 4.0)]);
        test_support::assert_reference_scalar_sine_jit_rendering();
    }

    #[test]
    fn stack_rejects_empty_inputs() {
        let result = stack::<(f64, f64), f64>(Vec::new());
        assert!(matches!(result, Err(TraceError::EmptyBatch)));
        test_support::assert_reference_scalar_sine_jit_rendering();
    }

    #[test]
    fn unstack_rejects_mismatched_lane_counts() {
        let batched = (Batch::new(vec![1.0f64]), Batch::new(vec![2.0f64, 3.0f64]));
        let result = unstack::<(f64, f64), f64>(batched);
        assert!(matches!(result, Err(TraceError::MismatchedBatchSize)));
        test_support::assert_reference_scalar_sine_jit_rendering();
    }

    #[test]
    fn vmap_exposes_batch_axis_size() {
        let outputs: Vec<f64> = vmap(
            |inputs: Batch<f64>| {
                assert_eq!(inputs.len(), 3);
                inputs.clone() + inputs.one_like()
            },
            vec![1.0f64, 2.0, 3.0],
        )
        .unwrap();
        assert_eq!(outputs, vec![2.0, 3.0, 4.0]);
        test_support::assert_reference_scalar_sine_jit_rendering();
    }

    #[test]
    fn traced_vmap_stages_one_higher_order_op() {
        let (output, compiled): (f64, CompiledFunction<f64, f64, f64>) = crate::tracing_v2::jit::try_jit(
            |x: JitTracer<f64>| {
                let outputs: Vec<JitTracer<f64>> =
                    vmap(|batch: Batch<JitTracer<f64>>| batch.clone() + batch.one_like(), vec![x.clone(), x])?;
                Ok(outputs[0].clone() + outputs[1].clone())
            },
            2.0f64,
        )
        .unwrap();

        assert_eq!(output, 6.0);
        assert_eq!(compiled.call(3.0f64).unwrap(), 8.0);
        assert_eq!(
            compiled.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[], %2:f64[] = vmap %0 %0
                    %3:f64[] = add %1 %2
                in (%3)
            "}
            .trim_end(),
        );
    }

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    #[test]
    fn test_vmap_of_grad_computes_per_lane_gradients() {
        // f(x) = x^2 + sin(x), df/dx = 2x + cos(x)
        let gradients: Vec<f64> = vmap(
            |batch: Batch<f64>| {
                crate::tracing_v2::grad(|x: JitTracer<f64>| x.clone() * x.clone() + x.sin(), batch)
                    .expect("batched grad should succeed")
            },
            vec![1.0f64, 2.0, 3.0],
        )
        .unwrap();

        approx_eq(gradients[0], 2.0 * 1.0 + 1.0f64.cos());
        approx_eq(gradients[1], 2.0 * 2.0 + 2.0f64.cos());
        approx_eq(gradients[2], 2.0 * 3.0 + 3.0f64.cos());
        test_support::assert_reference_scalar_sine_jit_rendering();
    }

    #[test]
    fn test_vmap_of_value_and_grad_returns_batched_values_and_gradients() {
        // f(x) = x^2 + sin(x), df/dx = 2x + cos(x)
        let results: Vec<(f64, f64)> = vmap(
            |batch: Batch<f64>| {
                crate::tracing_v2::value_and_grad(|x: JitTracer<f64>| x.clone() * x.clone() + x.sin(), batch)
                    .expect("batched value_and_grad should succeed")
            },
            vec![1.0f64, 2.0, 3.0],
        )
        .unwrap();

        for (i, x) in [1.0f64, 2.0, 3.0].into_iter().enumerate() {
            approx_eq(results[i].0, x * x + x.sin());
            approx_eq(results[i].1, 2.0 * x + x.cos());
        }
        test_support::assert_reference_scalar_sine_jit_rendering();
    }

    #[test]
    fn test_vmap_of_jvp_propagates_tangents_per_lane() {
        // f(x) = x^2 + sin(x), df/dx = 2x + cos(x)
        // jvp at x with tangent t gives (f(x), (2x + cos(x)) * t)
        let results: Vec<(f64, f64)> = vmap(
            |(primals, tangents): (Batch<f64>, Batch<f64>)| {
                crate::tracing_v2::jvp(|x: JitTracer<f64>| x.clone() * x.clone() + x.sin(), primals, tangents)
                    .expect("batched jvp should succeed")
            },
            vec![(1.0f64, 1.0f64), (2.0, 0.5), (3.0, 2.0)],
        )
        .unwrap();

        for (i, (x, t)) in [(1.0f64, 1.0f64), (2.0, 0.5), (3.0, 2.0)].into_iter().enumerate() {
            approx_eq(results[i].0, x * x + x.sin());
            approx_eq(results[i].1, (2.0 * x + x.cos()) * t);
        }
        test_support::assert_reference_scalar_sine_jit_rendering();
    }
}
