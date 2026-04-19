//! Vectorization support for `tracing_v2`.
//!
//! This module provides the batching story for the rest of the tracing stack. At the surface, users
//! interact with [`vmap`] and the explicit lane container [`Batch`]. Under the hood, the module has
//! two execution regimes that mirror the rest of `tracing_v2`:
//!
//! - on concrete leaves, batching is represented literally as a vector of lanes and primitive
//!   batching rules operate on those lane lists;
//! - on traced leaves, [`vmap`] stages a compact higher-order op instead of eagerly duplicating one
//!   scalar program per lane.
//!
//! That split is what lets the public API stay simple while still preserving enough structure for
//! backend lowering to emit packed batched programs.

use std::ops::{Add, Mul, Neg};

use crate::{
    parameters::{Parameter, Parameterized, ParameterizedFamily, Placeholder},
    tracing_v2::{
        OneLike, Program, Traceable, TracingError, ZeroLike,
        engine::Engine,
        jit::Tracer,
        operations::{
            AddOp, FlatTracedVMap, InterpretableOp, MulOp, NegOp, Op, VMapOp, VMapTracingOperation, VectorizableOp,
        },
    },
    types::{ArrayType, Typed},
};
use ryft_macros::Parameter;

/// Batched leaf value represented as an explicit list of lanes.
///
/// [`Batch`] is the concrete batching counterpart to [`Tracer`](crate::tracing_v2::Tracer): when a
/// transform wants to interpret a batched computation eagerly, each leaf becomes one `Batch<V>`
/// containing a lane-by-lane view of the input. Primitive batching rules consume and produce these
/// values directly.
#[derive(Clone, Debug, PartialEq, Parameter)]
pub struct Batch<V> {
    /// Lane values carried by this batch in semantic lane order.
    lanes: Vec<V>,
}

impl<V> Batch<V> {
    /// Creates a new batched value from a list of lanes.
    ///
    /// The lane order is semantically meaningful and is preserved by [`stack`], [`unstack`], and
    /// primitive batching rules.
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

impl<V: ZeroLike> ZeroLike for Batch<V> {
    #[inline]
    fn zero_like(&self) -> Self {
        Self::new(self.lanes.iter().map(ZeroLike::zero_like).collect())
    }
}

impl<V: OneLike> OneLike for Batch<V> {
    #[inline]
    fn one_like(&self) -> Self {
        Self::new(self.lanes.iter().map(OneLike::one_like).collect())
    }
}

/// Stacks a list of structured inputs into one structured value whose leaves are [`Batch`] values.
///
/// This is the structural entry point for concrete `vmap` execution. It transposes a
/// "batch-of-structures" into a "structure-of-batches" so the user function can run once over
/// batched leaves instead of once per lane.
pub fn stack<
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq, Family: ParameterizedFamily<Batch<V>>>,
    V: Parameter,
>(
    inputs: Vec<Input>,
) -> Result<Input::To<Batch<V>>, TracingError> {
    let mut inputs = inputs.into_iter();
    let first = inputs.next().ok_or(TracingError::EmptyBatch)?;
    let structure = first.parameter_structure();
    let parameter_count = structure.parameter_count();
    let mut buckets = (0..parameter_count).map(|_| Vec::new()).collect::<Vec<Vec<V>>>();
    let first_parameters = first.into_parameters().collect::<Vec<_>>();

    for (bucket, parameter) in buckets.iter_mut().zip(first_parameters) {
        bucket.push(parameter);
    }

    for input in inputs {
        if input.parameter_structure() != structure {
            return Err(TracingError::MismatchedParameterStructure);
        }

        for (bucket, parameter) in buckets.iter_mut().zip(input.into_parameters()) {
            bucket.push(parameter);
        }
    }

    Ok(Input::To::<Batch<V>>::from_parameters(structure, buckets.into_iter().map(Batch::new))?)
}

/// Splits a structured batch back into one structured value per lane.
///
/// This is the inverse of [`stack`]. Concrete `vmap` uses it after the batched body runs so the
/// caller gets one structured output per original lane.
pub fn unstack<
    Input: Parameterized<V, ParameterStructure: Clone, Family: ParameterizedFamily<Batch<V>>>,
    V: Parameter,
>(
    batched: Input::To<Batch<V>>,
) -> Result<Vec<Input>, TracingError> {
    let structure = batched.parameter_structure();
    let batches = batched.into_parameters().collect::<Vec<_>>();
    if batches.is_empty() {
        return Ok(Vec::new());
    }

    let lane_count = batches[0].len();
    if batches.iter().any(|batch| batch.len() != lane_count) {
        return Err(TracingError::MismatchedBatchSize);
    }

    let mut lane_parameters = (0..lane_count).map(|_| Vec::with_capacity(batches.len())).collect::<Vec<Vec<V>>>();
    for batch in batches {
        for (lane_index, value) in batch.into_lanes().into_iter().enumerate() {
            lane_parameters[lane_index].push(value);
        }
    }

    lane_parameters
        .into_iter()
        .map(|parameters| Input::from_parameters(structure.clone(), parameters).map_err(TracingError::from))
        .collect()
}

/// Dispatch trait used by [`vmap`] so it can handle both concrete batches and already traced values.
///
/// The trait is the batching analogue of the dispatch seams used by [`jvp`](crate::tracing_v2::jvp)
/// and [`grad`](crate::tracing_v2::grad): the public transform stays small while the concrete,
/// traced, and nested-batch execution strategies each get their own implementation.
#[doc(hidden)]
pub(crate) trait VMapInvocationLeaf<
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq, Family: ParameterizedFamily<Batch<Self>>>,
    Output: Parameterized<Self, ParameterStructure: Clone, Family: ParameterizedFamily<Batch<Self>>>,
>: Parameter + Sized
{
    /// Invokes [`vmap`] for one concrete leaf regime.
    fn invoke<F: FnOnce(Input::To<Batch<Self>>) -> Output::To<Batch<Self>>>(
        function: F,
        inputs: Vec<Input>,
    ) -> Result<Vec<Output>, TracingError>;
}

/// Concrete-value dispatch for [`vmap`]: stacks inputs into [`Batch`] leaves, applies the user function
/// over the batched representation, and unstacks the result back into per-lane outputs.
///
/// No op-capability (`Sin` / `Cos` / `MatrixOps` / `ReshapeOps`) bounds on `V` are required here because the body
/// of `invoke` never exercises them â€” it stacks / unstacks / invokes the user's closure on
/// `Batch<V>` values, and any capability the closure actually uses is enforced at the call site
/// through the conditional op-local trait impls on [`Batch`].
impl<
    V: Traceable<ArrayType> + crate::tracing_v2::Value<ArrayType>,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq, Family: ParameterizedFamily<Batch<V>>>,
    Output: Parameterized<V, ParameterStructure: Clone, Family: ParameterizedFamily<Batch<V>>>,
> VMapInvocationLeaf<Input, Output> for V
{
    fn invoke<F: FnOnce(Input::To<Batch<Self>>) -> Output::To<Batch<Self>>>(
        function: F,
        inputs: Vec<Input>,
    ) -> Result<Vec<Output>, TracingError> {
        let batched_input = stack(inputs)?;
        unstack(function(batched_input))
    }
}

/// Already-traced dispatch for [`vmap`]: stages a compact higher-order [`VMapOp`] in the enclosing
/// [`Tracer`] scope instead of eagerly duplicating the scalar program per lane. The body is traced
/// once at a single-lane exemplar and captured as a [`Program`] that lowering can later
/// emit as packed StableHLO.
impl<
    'engine,
    E: Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized + 'static,
    V: Traceable<ArrayType>
        + Parameterized<
            V,
            ParameterStructure = Placeholder,
            To<Tracer<'engine, E>> = Tracer<'engine, E>,
            Family: ParameterizedFamily<Tracer<'engine, E>>,
        >,
    Input: Parameterized<
            Tracer<'engine, E>,
            ParameterStructure: Clone + PartialEq,
            To<Tracer<'engine, E>> = Input,
            Family: ParameterizedFamily<Batch<Tracer<'engine, E>>>
                        + ParameterizedFamily<V>
                        + ParameterizedFamily<ArrayType>,
        >,
    Output: Parameterized<
            Tracer<'engine, E>,
            ParameterStructure: Clone,
            To<Tracer<'engine, E>> = Output,
            Family: ParameterizedFamily<Batch<Tracer<'engine, E>>>
                        + ParameterizedFamily<Tracer<'engine, E>>
                        + ParameterizedFamily<V>
                        + ParameterizedFamily<ArrayType>,
        >,
    O: Op<ArrayType> + InterpretableOp<ArrayType, V> + VMapTracingOperation<ArrayType, V, L>,
    L: Clone,
> VMapInvocationLeaf<Input, Output> for Tracer<'engine, E>
where
    Input::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'engine, E>> = Input, To<V> = Input::To<V>>,
    Output::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'engine, E>> = Output, To<V> = Output::To<V>>,
    Vec<V>: Parameterized<V, To<Tracer<'engine, E>> = Vec<Tracer<'engine, E>>, ParameterStructure = Vec<Placeholder>>,
    <Vec<V> as Parameterized<V>>::Family: ParameterizedFamily<Tracer<'engine, E>>,
{
    fn invoke<F: FnOnce(Input::To<Batch<Self>>) -> Output::To<Batch<Self>>>(
        function: F,
        inputs: Vec<Input>,
    ) -> Result<Vec<Output>, TracingError> {
        let mut inputs = inputs.into_iter();
        let first_input = inputs.next().ok_or(TracingError::EmptyBatch)?;
        let input_structure = first_input.parameter_structure();
        let mut traced_inputs = vec![first_input.into_parameters().collect::<Vec<_>>()];
        for input in inputs {
            if input.parameter_structure() != input_structure {
                return Err(TracingError::MismatchedParameterStructure);
            }
            traced_inputs.push(input.into_parameters().collect::<Vec<_>>());
        }

        let lane_count = traced_inputs.len();
        let input_leaf_count = input_structure.parameter_count();
        let exemplar_input_types = Input::To::<ArrayType>::from_parameters(
            input_structure.clone(),
            traced_inputs[0].iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
        )?;
        let exemplar_engine = traced_inputs[0].first().ok_or(TracingError::EmptyParameterizedValue)?.engine();

        let (exemplar_output_types, body_program): (
            Output::To<ArrayType>,
            Program<ArrayType, V, O, Input::To<V>, Output::To<V>>,
        ) = crate::tracing_v2::jit::trace(
            exemplar_engine,
            |lane_inputs| {
                let batched_inputs = Input::To::<Batch<Tracer<'engine, E>>>::from_parameters(
                    lane_inputs.parameter_structure(),
                    lane_inputs.into_parameters().map(|input| Batch::new(vec![input])),
                )?;
                let batched_outputs = function(batched_inputs);
                let output_structure = batched_outputs.parameter_structure();
                let mut lane_outputs = Vec::new();
                for batch in batched_outputs.into_parameters() {
                    let mut outputs = batch.into_lanes();
                    if outputs.len() != 1 {
                        return Err(TracingError::HigherOrderOpFailure {
                            op: "vmap",
                            message: "traced vmap only supports bodies that preserve the per-lane output structure"
                                .to_string(),
                        });
                    }
                    lane_outputs.push(outputs.pop().expect("single-lane batches should contain one output"));
                }
                Output::from_parameters(output_structure, lane_outputs).map_err(TracingError::from)
            },
            exemplar_input_types,
        )?;
        let body_program = body_program.simplify()?;

        let output_structure = exemplar_output_types.parameter_structure();
        let output_leaf_count = output_structure.parameter_count();
        let flat_input_structure = vec![Placeholder; input_leaf_count];
        let flat_output_structure = vec![Placeholder; output_leaf_count];
        let body = FlatTracedVMap::from_parts(
            lane_count,
            body_program
                .input_ids()
                .iter()
                .map(|input| body_program.atom(*input).expect("body input atoms should exist").r#type().into_owned())
                .collect::<Vec<_>>(),
            exemplar_output_types.parameters().cloned().collect::<Vec<_>>(),
            body_program.clone_with_structures::<Vec<V>, Vec<V>>(flat_input_structure, flat_output_structure),
        );

        let staged_inputs = traced_inputs.into_iter().flatten().collect::<Vec<_>>();
        let staged_outputs = Tracer::apply_staged_op(staged_inputs.as_slice(), O::vmap_op(VMapOp::new(body)))?;
        (0..lane_count)
            .map(|lane_index| {
                let start = lane_index * output_leaf_count;
                let end = start + output_leaf_count;
                Output::from_parameters(output_structure.clone(), staged_outputs[start..end].iter().cloned())
                    .map_err(TracingError::from)
            })
            .collect()
    }
}

/// Nested-batch dispatch for [`vmap`], enabling `vmap(|xs| vmap(g, xs))` -- applying vectorization
/// recursively. This delegates to the concrete `V` implementation: the outer batch is unstacked,
/// the inner [`vmap`] runs per outer lane using the existing [`VMapInvocationLeaf`] for `V`, and
/// results are stacked back. No trace-once pattern is needed here because the delegation to the
/// concrete implementation handles each lane directly.
///
/// Capability-trait bounds (`Sin` / `Cos` / `MatrixOps` / `ReshapeOps`) on `V` are deliberately omitted: the
/// body only stacks, unstacks, and invokes the user's closure â€” any capability the closure uses on
/// `Batch<Batch<V>>` is enforced through the conditional blanket impls on `Batch<_>`.
impl<
    V: Traceable<ArrayType>,
    Input: Parameterized<Batch<V>, ParameterStructure: Clone + PartialEq, Family: ParameterizedFamily<Batch<Batch<V>>>>,
    Output: Parameterized<Batch<V>, ParameterStructure: Clone, Family: ParameterizedFamily<Batch<Batch<V>>>>,
> VMapInvocationLeaf<Input, Output> for Batch<V>
{
    fn invoke<F: FnOnce(Input::To<Batch<Self>>) -> Output::To<Batch<Self>>>(
        function: F,
        inputs: Vec<Input>,
    ) -> Result<Vec<Output>, TracingError> {
        let batched_input = stack(inputs)?;
        unstack(function(batched_input))
    }
}

/// Maps `function` over a leading batch axis.
///
/// Conceptually, [`vmap`] lifts a scalar function into a batched function. For concrete inputs it
/// does so by stacking the input leaves into [`Batch`] values, running the user closure once, and
/// then unstacking the result back into one output per lane. For traced inputs it instead stages a
/// compact higher-order `vmap` operation so later transforms and lowerings can treat batching as a
/// first-class program construct.
#[allow(private_bounds)]
pub fn vmap<
    F: FnOnce(Input::To<Batch<V>>) -> Output::To<Batch<V>>,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq, Family: ParameterizedFamily<Batch<V>>>,
    Output: Parameterized<V, ParameterStructure: Clone, Family: ParameterizedFamily<Batch<V>>>,
    V: VMapInvocationLeaf<Input, Output>,
>(
    function: F,
    inputs: Vec<Input>,
) -> Result<Vec<Output>, TracingError> {
    V::invoke(function, inputs)
}

#[cfg(test)]
mod tests {
    use indoc::indoc;

    use crate::tracing_v2::{PrimitiveOp, Sin, Tracer, engine::ArrayScalarEngine, test_support};

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
        assert!(matches!(result, Err(TracingError::EmptyBatch)));
        test_support::assert_reference_scalar_sine_jit_rendering();
    }

    #[test]
    fn unstack_rejects_mismatched_lane_counts() {
        let batched = (Batch::new(vec![1.0f64]), Batch::new(vec![2.0f64, 3.0f64]));
        let result = unstack::<(f64, f64), f64>(batched);
        assert!(matches!(result, Err(TracingError::MismatchedBatchSize)));
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
        let engine = ArrayScalarEngine::<f64>::new();
        let (output, program): (f64, Program<ArrayType, f64, PrimitiveOp<ArrayType, f64>, f64, f64>) =
            crate::tracing_v2::interpret_and_trace(
                &engine,
                |x| {
                    let outputs: Vec<Tracer<ArrayScalarEngine<f64>>> =
                        vmap(|batch| batch.clone() + batch.one_like(), vec![x.clone(), x])?;
                    Ok(outputs[0].clone() + outputs[1].clone())
                },
                2.0f64,
            )
            .unwrap();

        assert_eq!(output, 6.0);
        assert_eq!(program.call(3.0f64).unwrap(), 8.0);
        assert_eq!(
            program.to_string(),
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
        let engine = crate::tracing_v2::engine::ArrayScalarEngine::<f64>::new();
        let gradients: Vec<f64> = vmap(
            |batch: Batch<f64>| {
                crate::tracing_v2::grad(&engine, |x| x.clone() * x.clone() + x.sin(), batch)
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
        let engine = crate::tracing_v2::engine::ArrayScalarEngine::<f64>::new();
        let results: Vec<(f64, f64)> = vmap(
            |batch: Batch<f64>| {
                crate::tracing_v2::value_and_grad(&engine, |x| x.clone() * x.clone() + x.sin(), batch)
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
        let engine = crate::tracing_v2::engine::ArrayScalarEngine::<f64>::new();
        let results: Vec<(f64, f64)> = vmap(
            |(primals, tangents): (Batch<f64>, Batch<f64>)| {
                crate::tracing_v2::jvp(&engine, |x| x.clone() * x.clone() + x.sin(), primals, tangents)
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

    /// Pins the user-facing property the quick-win in this module delivers: a concrete leaf type
    /// that implements only `Traceable + Value + Add + ZeroLike + OneLike` â€” without `Sin`,
    /// `Cos`,
    /// `MatrixOps`, or `ReshapeOps` â€” must still be usable through [`vmap`] for programs that only
    /// exercise `Add`. Previously, the concrete [`VMapInvocationLeaf`] impl carried the full
    /// `Add + Mul + Neg + Sin + Cos + MatrixOps + ReshapeOps` union and rejected this case at
    /// compile time. The impl
    /// now only requires `Traceable + Value`, so callers pay exactly the capabilities their closure
    /// uses. The closure here exercises `Add` (via `Batch<Int64>::Add`, which delegates to
    /// `Int64::Add`), and any attempt to call e.g. `.sin()` on `Batch<Int64>` would still fail to
    /// compile because the `impl<V: Sin> Sin for Batch<V>` blanket would demand `Int64: Sin`.
    #[test]
    fn test_vmap_compiles_for_leaf_without_float_matrix_or_reshape_ext_traits() {
        use std::borrow::Cow;
        use std::ops::Add;

        use crate::types::{ArrayType, DataType, Typed};

        #[derive(Clone, Debug, PartialEq, Parameter)]
        struct Int64(i64);

        impl Typed<ArrayType> for Int64 {
            fn r#type(&self) -> Cow<'_, ArrayType> {
                Cow::Owned(ArrayType::scalar(DataType::I64))
            }
        }

        impl Traceable<ArrayType> for Int64 {}
        impl crate::tracing_v2::Value<ArrayType> for Int64 {}

        impl Add for Int64 {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                Self(self.0 + rhs.0)
            }
        }

        impl ZeroLike for Int64 {
            fn zero_like(&self) -> Self {
                Self(0)
            }
        }

        impl OneLike for Int64 {
            fn one_like(&self) -> Self {
                Self(1)
            }
        }

        let outputs: Vec<Int64> =
            vmap(|batch: Batch<Int64>| batch.clone() + batch, vec![Int64(1), Int64(2), Int64(3)]).unwrap();
        assert_eq!(outputs, vec![Int64(2), Int64(4), Int64(6)]);
    }
}
