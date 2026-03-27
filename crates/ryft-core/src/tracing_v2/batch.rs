//! Vectorization support for `tracing_v2`.
//!
//! Concrete batching is still represented as explicit lane lists via [`Batch`]. For traced programs, however,
//! [`vmap`] stages a compact higher-order op instead of eagerly duplicating one scalar graph per lane. That keeps the
//! public batching surface unchanged while giving lowering enough structure to emit packed StableHLO that is much
//! closer to JAX's current Shardy output.

use std::{
    fmt::{Debug, Display},
    ops::{Add, Mul, Neg},
    sync::Arc,
};

use ryft_macros::Parameter;
#[cfg(feature = "xla")]
use ryft_mlir::ValueRef;

#[cfg(feature = "xla")]
use crate::xla::lowering::{
    LoweringError, MlirLowerableValue, PlainMlirLowerer, PlainMlirLoweringMode, ShardMapMlirLowerer,
};
use crate::{
    parameters::{Parameter, Parameterized, ParameterizedFamily, Placeholder},
    tracing_v2::{
        CompiledFunction, FloatExt, JitTracer, LinearTerm, MatrixOps, OneLike, Op, Program, TraceError, TraceValue,
        TransformLeaf, ZeroLike,
        linear::{linearize_program, transpose_linear_program},
        ops::{AddOp, BatchOp, CosOp, MulOp, NegOp, SinOp},
    },
    types::{ArrayType, Typed},
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

/// Erased traced `vmap` body used by the staged higher-order op.
#[derive(Clone)]
pub(crate) struct FlatTracedVMap<V>
where
    V: TraceValue,
{
    lane_count: usize,
    input_types: Vec<ArrayType>,
    output_types: Vec<ArrayType>,
    compiled: CompiledFunction<V, Vec<V>, Vec<V>>,
}

impl<V> FlatTracedVMap<V>
where
    V: TraceValue,
{
    #[inline]
    pub(crate) fn from_parts(
        lane_count: usize,
        input_types: Vec<ArrayType>,
        output_types: Vec<ArrayType>,
        compiled: CompiledFunction<V, Vec<V>, Vec<V>>,
    ) -> Self {
        Self { lane_count, input_types, output_types, compiled }
    }

    #[inline]
    pub(crate) fn lane_count(&self) -> usize {
        self.lane_count
    }

    #[inline]
    pub(crate) fn input_types(&self) -> &[ArrayType] {
        self.input_types.as_slice()
    }

    #[inline]
    pub(crate) fn output_types(&self) -> &[ArrayType] {
        self.output_types.as_slice()
    }

    #[inline]
    pub(crate) fn compiled(&self) -> &CompiledFunction<V, Vec<V>, Vec<V>> {
        &self.compiled
    }

    #[inline]
    pub(crate) fn total_input_count(&self) -> usize {
        self.lane_count * self.input_types.len()
    }

    #[inline]
    pub(crate) fn total_output_count(&self) -> usize {
        self.lane_count * self.output_types.len()
    }

    fn repeated_input_types(&self) -> Vec<ArrayType> {
        (0..self.lane_count).flat_map(|_| self.input_types.iter().cloned()).collect::<Vec<_>>()
    }

    fn repeated_output_types(&self) -> Vec<ArrayType> {
        (0..self.lane_count).flat_map(|_| self.output_types.iter().cloned()).collect::<Vec<_>>()
    }

    fn eval_lanes(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        if inputs.len() != self.total_input_count() {
            return Err(TraceError::InvalidInputCount { expected: self.total_input_count(), got: inputs.len() });
        }

        let lane_input_count = self.input_types.len();
        let mut outputs = Vec::with_capacity(self.total_output_count());
        for lane_inputs in inputs.chunks(lane_input_count) {
            outputs.extend(self.compiled.call(lane_inputs.to_vec())?);
        }
        Ok(outputs)
    }
}

/// Higher-order `vmap` op that carries one canonical program payload and, when linear, its transpose payload too.
#[derive(Clone)]
pub(crate) struct VMapOp<V>
where
    V: TraceValue,
{
    body: FlatTracedVMap<V>,
    transpose_body: Option<FlatTracedVMap<V>>,
}

impl<V> VMapOp<V>
where
    V: TraceValue,
{
    #[inline]
    pub(crate) fn new(body: FlatTracedVMap<V>) -> Self {
        Self { body, transpose_body: None }
    }

    #[inline]
    pub(crate) fn new_linear(body: FlatTracedVMap<V>, transpose_body: FlatTracedVMap<V>) -> Self {
        Self { body, transpose_body: Some(transpose_body) }
    }

    #[inline]
    pub(crate) fn body(&self) -> &FlatTracedVMap<V> {
        &self.body
    }

    #[inline]
    pub(crate) fn has_transpose_body(&self) -> bool {
        self.transpose_body.is_some()
    }

    pub(crate) fn transpose_op(&self) -> Result<Self, TraceError> {
        let transpose_body = self.transpose_body.clone().ok_or(TraceError::HigherOrderOpFailure {
            op: "vmap",
            message: "transpose requested for a vmap op without a transpose body".to_string(),
        })?;
        Ok(Self::new_linear(transpose_body, self.body.clone()))
    }
}

impl<V> Debug for VMapOp<V>
where
    V: TraceValue,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "VMap")
    }
}

impl<V> Display for VMapOp<V>
where
    V: TraceValue,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "vmap")
    }
}

impl<V> Op<V> for VMapOp<V>
where
    V: TransformLeaf,
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &'static str {
        "vmap"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        let expected_inputs = self.body.repeated_input_types();
        if inputs.len() != expected_inputs.len() {
            return Err(TraceError::InvalidInputCount { expected: expected_inputs.len(), got: inputs.len() });
        }
        if inputs != expected_inputs.as_slice() {
            return Err(TraceError::IncompatibleAbstractValues { op: "vmap" });
        }
        Ok(self.body.repeated_output_types())
    }

    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        let abstract_inputs = inputs.iter().map(Typed::tpe).collect::<Vec<_>>();
        let _ = <Self as Op<V>>::abstract_eval(self, abstract_inputs.as_slice())?;
        self.body.eval_lanes(inputs)
    }

    fn replay_linearized_jit(
        &self,
        inputs: Vec<crate::tracing_v2::JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>,
    ) -> Result<Vec<crate::tracing_v2::JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>, TraceError>
    where
        V: TransformLeaf,
    {
        if self.has_transpose_body() {
            return Err(TraceError::HigherOrderOpFailure {
                op: "replay_program_graph",
                message: "replaying linearized values through a linear vmap op is not implemented".to_string(),
            });
        }
        let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let primal_output_values =
            self.eval(primal_inputs.iter().map(|input| input.value.clone()).collect::<Vec<_>>().as_slice())?;
        let primal_outputs =
            JitTracer::apply_staged_op(primal_inputs.as_slice(), Arc::new(self.clone()), primal_output_values)?;
        let lane_input_count = self.body().input_types().len();
        let mut tangent_outputs = Vec::with_capacity(self.body().total_output_count());
        for lane_inputs in inputs.chunks(lane_input_count) {
            let lane_outputs = crate::tracing_v2::linear::replay_program_graph_linearized_jit(
                self.body().compiled().program().graph(),
                lane_inputs.to_vec(),
            )?;
            tangent_outputs.extend(lane_outputs.into_iter().map(|output| output.tangent));
        }
        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| crate::tracing_v2::JvpTracer { primal, tangent })
            .collect::<Vec<_>>())
    }

    fn apply_program_jvp_rule(
        &self,
        inputs: &[crate::tracing_v2::JvpTracer<V, LinearTerm<V>>],
    ) -> Result<Vec<crate::tracing_v2::JvpTracer<V, LinearTerm<V>>>, TraceError>
    where
        V: FloatExt + ZeroLike + MatrixOps,
    {
        if self.has_transpose_body() {
            return Err(TraceError::HigherOrderOpFailure {
                op: "linearize_program",
                message: "JVP rule for staged op 'vmap' is not implemented".to_string(),
            });
        }
        let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let tangent_inputs = inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
        let primal_outputs = self.eval(primal_inputs.as_slice())?;
        let tangent_outputs = LinearTerm::apply_staged_op(
            tangent_inputs.as_slice(),
            Arc::new(make_linear_vmap(&self.body)?),
            self.body.total_output_count(),
        )?;
        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| crate::tracing_v2::JvpTracer { primal, tangent })
            .collect::<Vec<_>>())
    }

    fn transpose_program_op(
        &self,
        builder: &mut crate::tracing_v2::ProgramBuilder<V>,
        inputs: &[crate::tracing_v2::AtomId],
        outputs: &[crate::tracing_v2::AtomId],
        output_cotangents: &[crate::tracing_v2::AtomId],
    ) -> Result<Vec<Option<crate::tracing_v2::AtomId>>, TraceError>
    where
        V: FloatExt + ZeroLike + MatrixOps,
    {
        if !self.has_transpose_body() {
            return Err(TraceError::HigherOrderOpFailure {
                op: "transpose_linear_program",
                message: "transpose rule for staged op 'vmap' is not implemented".to_string(),
            });
        }
        if inputs.len() != self.body.total_input_count() {
            return Err(TraceError::InvalidInputCount { expected: self.body.total_input_count(), got: inputs.len() });
        }
        if outputs.len() != self.body.total_output_count() {
            return Err(TraceError::InvalidOutputCount {
                expected: self.body.total_output_count(),
                got: outputs.len(),
            });
        }
        if output_cotangents.len() != self.body.total_output_count() {
            return Err(TraceError::InvalidInputCount {
                expected: self.body.total_output_count(),
                got: output_cotangents.len(),
            });
        }
        let contributions = builder.add_equation(Arc::new(self.transpose_op()?), output_cotangents.to_vec())?;
        Ok(contributions.into_iter().map(Some).collect::<Vec<_>>())
    }

    #[cfg(feature = "xla")]
    fn lower_plain_mlir<'b, 'c, 't>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
    where
        V: MlirLowerableValue,
    {
        lowerer.lower_vmap(self, input_values)
    }

    #[cfg(feature = "xla")]
    fn lower_shard_map_mlir<'b, 'c, 't, 'm>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        lowerer: &mut ShardMapMlirLowerer<'b, 'c, 't, 'm>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
    where
        V: MlirLowerableValue,
    {
        lowerer.lower_vmap(self, input_values)
    }
}

impl<V> Op<JitTracer<V>> for VMapOp<V>
where
    V: TransformLeaf,
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &'static str {
        "vmap"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        <Self as Op<V>>::abstract_eval(self, inputs)
    }

    fn eval(&self, inputs: &[JitTracer<V>]) -> Result<Vec<JitTracer<V>>, TraceError> {
        let concrete_inputs = inputs.iter().map(|input| input.value.clone()).collect::<Vec<_>>();
        let output_values = <Self as Op<V>>::eval(self, concrete_inputs.as_slice())?;
        JitTracer::apply_staged_op(inputs, Arc::new(self.clone()), output_values)
    }
}

fn single_output<V>(mut outputs: Vec<Batch<V>>, op: &'static str) -> Batch<V> {
    debug_assert_eq!(outputs.len(), 1, "{op} should produce a single batched output");
    outputs.pop().expect("single-output primitive should return one batched output")
}

pub(crate) fn make_linear_vmap<V>(body: &FlatTracedVMap<V>) -> Result<VMapOp<V>, TraceError>
where
    V: TransformLeaf,
{
    let pushforward = linearize_program(body.compiled.program())?;
    let pullback = transpose_linear_program(&pushforward)?;
    Ok(VMapOp::new_linear(
        FlatTracedVMap::from_parts(
            body.lane_count,
            body.input_types.clone(),
            body.output_types.clone(),
            CompiledFunction::from_program(pushforward.program().clone()),
        ),
        FlatTracedVMap::from_parts(
            body.lane_count,
            body.output_types.clone(),
            body.input_types.clone(),
            CompiledFunction::from_program(pullback.program().clone()),
        ),
    ))
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

/// Dispatch trait used by [`vmap`] so it can handle both concrete batches and already traced values.
#[doc(hidden)]
pub(crate) trait VMapInvocationLeaf<Input, Output>: Parameter + Sized
where
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq>,
    Output: Parameterized<Self, ParameterStructure: Clone>,
{
    /// Invokes [`vmap`] for one concrete leaf regime.
    fn invoke<F>(function: F, inputs: Vec<Input>) -> Result<Vec<Output>, TraceError>
    where
        Input::Family: ParameterizedFamily<Batch<Self>>,
        Output::Family: ParameterizedFamily<Batch<Self>>,
        F: FnOnce(Input::To<Batch<Self>>) -> Output::To<Batch<Self>>;
}

impl<V, Input, Output> VMapInvocationLeaf<Input, Output> for V
where
    V: TransformLeaf,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<Batch<V>>,
    Output: Parameterized<V, ParameterStructure: Clone>,
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

impl<V, Input, Output> VMapInvocationLeaf<Input, Output> for JitTracer<V>
where
    V: TransformLeaf,
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<Batch<Self>> + ParameterizedFamily<V>,
    Output: Parameterized<Self, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<Batch<Self>> + ParameterizedFamily<Self> + ParameterizedFamily<V>,
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
            CompiledFunction::from_graph(body_program.graph().clone_with_structures::<Vec<V>, Vec<V>>(
                vec![Placeholder; input_leaf_count],
                vec![Placeholder; output_leaf_count],
            )),
        );

        let output_values = traced_inputs
            .iter()
            .map(|lane_inputs| {
                body.compiled.call(lane_inputs.iter().map(|input| input.value.clone()).collect::<Vec<_>>())
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let staged_inputs = traced_inputs.into_iter().flatten().collect::<Vec<_>>();
        let staged_outputs =
            JitTracer::apply_staged_op(staged_inputs.as_slice(), Arc::new(VMapOp::new(body)), output_values)?;
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
}
