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

#[derive(Clone, Debug, PartialEq, Parameter)]
pub struct Batch<V> {
    lanes: Vec<V>,
}

impl<V> Batch<V> {
    #[inline]
    pub fn new(lanes: Vec<V>) -> Self {
        Self { lanes }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.lanes.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.lanes.is_empty()
    }

    #[inline]
    pub fn lanes(&self) -> &[V] {
        self.lanes.as_slice()
    }

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
    V: TraceValue,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        single_output(AddOp.batch(&[self, rhs]).expect("add batching rule should succeed"), "add")
    }
}

impl<V> Mul for Batch<V>
where
    V: TraceValue,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        single_output(MulOp.batch(&[self, rhs]).expect("mul batching rule should succeed"), "mul")
    }
}

impl<V> Neg for Batch<V>
where
    V: TraceValue,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        single_output(NegOp.batch(&[self]).expect("neg batching rule should succeed"), "neg")
    }
}

impl<V> FloatExt for Batch<V>
where
    V: TraceValue,
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
