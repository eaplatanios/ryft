//! Higher-order `vmap` operations for [`crate::tracing_v2`].

use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

#[cfg(feature = "xla")]
use ryft_mlir::ValueRef;

#[cfg(feature = "xla")]
use crate::xla::lowering::{
    LoweringError, MlirLowerableValue, PlainMlirLowerer, PlainMlirLoweringMode, ShardMapMlirLowerer,
};
use crate::{
    tracing_v2::{
        CompiledFunction, FloatExt, JitTracer, LinearTerm, MatrixOps, Op, TraceError, TraceValue, TransformLeaf,
        ZeroLike,
        linear::{linearize_program, transpose_linear_program},
    },
    types::{ArrayType, Typed},
};

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
    /// Builds one erased traced `vmap` body from explicit staged parts.
    #[inline]
    pub(crate) fn from_parts(
        lane_count: usize,
        input_types: Vec<ArrayType>,
        output_types: Vec<ArrayType>,
        compiled: CompiledFunction<V, Vec<V>, Vec<V>>,
    ) -> Self {
        Self { lane_count, input_types, output_types, compiled }
    }

    /// Returns the body lane count.
    #[inline]
    pub(crate) fn lane_count(&self) -> usize {
        self.lane_count
    }

    /// Returns the canonical per-lane input types.
    #[inline]
    pub(crate) fn input_types(&self) -> &[ArrayType] {
        self.input_types.as_slice()
    }

    /// Returns the canonical per-lane output types.
    #[inline]
    pub(crate) fn output_types(&self) -> &[ArrayType] {
        self.output_types.as_slice()
    }

    /// Returns the compiled flat body.
    #[inline]
    pub(crate) fn compiled(&self) -> &CompiledFunction<V, Vec<V>, Vec<V>> {
        &self.compiled
    }

    /// Returns the flattened input count across all lanes.
    #[inline]
    pub(crate) fn total_input_count(&self) -> usize {
        self.lane_count * self.input_types.len()
    }

    /// Returns the flattened output count across all lanes.
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
    /// Builds one ordinary traced `vmap` op.
    #[inline]
    pub(crate) fn new(body: FlatTracedVMap<V>) -> Self {
        Self { body, transpose_body: None }
    }

    /// Builds one linear traced `vmap` op with an explicit transpose body.
    #[inline]
    pub(crate) fn new_linear(body: FlatTracedVMap<V>, transpose_body: FlatTracedVMap<V>) -> Self {
        Self { body, transpose_body: Some(transpose_body) }
    }

    /// Returns the canonical traced body.
    #[inline]
    pub(crate) fn body(&self) -> &FlatTracedVMap<V> {
        &self.body
    }

    /// Returns whether the op carries a transpose body.
    #[inline]
    pub(crate) fn has_transpose_body(&self) -> bool {
        self.transpose_body.is_some()
    }

    fn transpose_op(&self) -> Result<Self, TraceError> {
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
    fn lower_shard_map_mlir<'b, 'c, 't>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        lowerer: &mut ShardMapMlirLowerer<'b, 'c, 't>,
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

/// Builds one linearized staged `vmap` op from its primal body.
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
