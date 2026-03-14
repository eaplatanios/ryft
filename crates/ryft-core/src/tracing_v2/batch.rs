//! Vectorization support for `tracing_v2`.
//!
//! `Batch<V>` is the current prototype representation for a batched leaf value. It materializes each lane in a `Vec`,
//! which is simple and works well for tests. A future backend can keep the same public `stack` / `unstack` / `vmap`
//! surface while swapping in a more efficient batched representation.

use std::ops::{Add, Mul, Neg};

use ryft_macros::Parameter;

use crate::{
    parameters::{Parameter, Parameterized, ParameterizedFamily},
    tracing_v2::{
        FloatExt, OneLike, TraceError, TraceValue, ZeroLike,
        context::BatchingContext,
        ops::{AddOp, BatchOp, CosOp, MulOp, NegOp, SinOp},
    },
};

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

impl<V> Add for Batch<V>
where
    V: TraceValue + Add<Output = V>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        single_output(AddOp.batch(&[self, rhs]).expect("add batching rule should succeed"), "add")
    }
}

impl<V> Mul for Batch<V>
where
    V: TraceValue + Mul<Output = V>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        single_output(MulOp.batch(&[self, rhs]).expect("mul batching rule should succeed"), "mul")
    }
}

impl<V> Neg for Batch<V>
where
    V: TraceValue + Neg<Output = V>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        single_output(NegOp.batch(&[self]).expect("neg batching rule should succeed"), "neg")
    }
}

impl<V> FloatExt for Batch<V>
where
    V: TraceValue + FloatExt,
{
    #[inline]
    fn sin(self) -> Self {
        single_output(SinOp.batch(&[self]).expect("sin batching rule should succeed"), "sin")
    }

    #[inline]
    fn cos(self) -> Self {
        single_output(CosOp.batch(&[self]).expect("cos batching rule should succeed"), "cos")
    }
}

impl<V> ZeroLike for Batch<V>
where
    V: ZeroLike,
{
    #[inline]
    fn zero_like(&self) -> Self {
        Self::new(self.lanes.iter().map(ZeroLike::zero_like).collect())
    }
}

impl<V> OneLike for Batch<V>
where
    V: OneLike,
{
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

/// Maps `function` over a leading batch axis by stacking inputs, running the batched program once, and then
/// unstacking the result.
pub fn vmap<'context, Context, F, Input, Output, V>(
    context: &'context mut Context,
    function: F,
    inputs: Vec<Input>,
) -> Result<Vec<Output>, TraceError>
where
    V: TraceValue,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<Batch<V>>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<Batch<V>>,
    F: FnOnce(&mut BatchingContext<'context, Context>, Input::To<Batch<V>>) -> Output::To<Batch<V>>,
{
    let axis_size = inputs.len();
    let batched_input = stack(inputs)?;
    let mut batching_context = BatchingContext::new(context, axis_size);
    let batched_output = function(&mut batching_context, batched_input);
    let _context = batching_context.finish();
    unstack(batched_output)
}

#[cfg(test)]
mod tests {
    use crate::tracing_v2::test_support;

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
        let mut context = ();
        let outputs: Vec<f64> = vmap(
            &mut context,
            |context, inputs: Batch<f64>| {
                assert_eq!(context.axis_size(), 3);
                inputs.clone() + inputs.one_like()
            },
            vec![1.0f64, 2.0, 3.0],
        )
        .unwrap();
        assert_eq!(outputs, vec![2.0, 3.0, 4.0]);
        test_support::assert_reference_scalar_sine_jit_rendering();
    }
}
