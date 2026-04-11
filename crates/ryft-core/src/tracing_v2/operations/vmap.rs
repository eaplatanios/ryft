//! Higher-order `vmap` operations for [`crate::tracing_v2`].

use std::fmt::{Debug, Display};

use crate::{
    tracing_v2::{
        CompiledFunction, FloatExt, JitTracer, LinearTerm, MatrixOps, OneLike, TraceError, TraceValue,
        TransformLeaf, ZeroLike,
        operations::reshape::ReshapeOps,
        ops::{DifferentiableOp, Eval, Op, PrimitiveOp},
        linear::{linearize_program, transpose_linear_program},
    },
    types::{ArrayType, Typed},
};

/// Erased traced `vmap` body used by the staged higher-order op.
#[derive(Clone)]
pub struct FlatTracedVMap<V: TraceValue> {
    lane_count: usize,
    input_types: Vec<ArrayType>,
    output_types: Vec<ArrayType>,
    compiled: CompiledFunction<V, Vec<V>, Vec<V>>,
}

impl<V: TraceValue> FlatTracedVMap<V> {
    /// Builds one erased traced `vmap` body from explicit staged parts.
    #[inline]
    pub fn from_parts(
        lane_count: usize,
        input_types: Vec<ArrayType>,
        output_types: Vec<ArrayType>,
        compiled: CompiledFunction<V, Vec<V>, Vec<V>>,
    ) -> Self {
        Self { lane_count, input_types, output_types, compiled }
    }

    /// Returns the body lane count.
    #[inline]
    pub fn lane_count(&self) -> usize {
        self.lane_count
    }

    /// Returns the canonical per-lane input types.
    #[inline]
    pub fn input_types(&self) -> &[ArrayType] {
        self.input_types.as_slice()
    }

    /// Returns the canonical per-lane output types.
    #[inline]
    pub fn output_types(&self) -> &[ArrayType] {
        self.output_types.as_slice()
    }

    /// Returns the compiled flat body.
    #[inline]
    pub fn compiled(&self) -> &CompiledFunction<V, Vec<V>, Vec<V>> {
        &self.compiled
    }

    /// Returns the flattened input count across all lanes.
    #[inline]
    pub fn total_input_count(&self) -> usize {
        self.lane_count * self.input_types.len()
    }

    /// Returns the flattened output count across all lanes.
    #[inline]
    pub fn total_output_count(&self) -> usize {
        self.lane_count * self.output_types.len()
    }

    pub(crate) fn repeated_input_types(&self) -> Vec<ArrayType> {
        (0..self.lane_count).flat_map(|_| self.input_types.iter().cloned()).collect::<Vec<_>>()
    }

    pub(crate) fn repeated_output_types(&self) -> Vec<ArrayType> {
        (0..self.lane_count).flat_map(|_| self.output_types.iter().cloned()).collect::<Vec<_>>()
    }

    pub(crate) fn eval_lanes(&self, inputs: &[V]) -> Result<Vec<V>, TraceError>
    where
        V: FloatExt + ZeroLike + OneLike + MatrixOps + ReshapeOps,
    {
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
pub struct VMapOp<V: TraceValue> {
    body: FlatTracedVMap<V>,
    transpose_body: Option<FlatTracedVMap<V>>,
}

impl<V: TraceValue> VMapOp<V> {
    /// Builds one ordinary traced `vmap` op.
    #[inline]
    pub fn new(body: FlatTracedVMap<V>) -> Self {
        Self { body, transpose_body: None }
    }

    /// Builds one linear traced `vmap` op with an explicit transpose body.
    #[inline]
    pub fn new_linear(body: FlatTracedVMap<V>, transpose_body: FlatTracedVMap<V>) -> Self {
        Self { body, transpose_body: Some(transpose_body) }
    }

    /// Returns the canonical traced body.
    #[inline]
    pub fn body(&self) -> &FlatTracedVMap<V> {
        &self.body
    }

    /// Returns whether the op carries a transpose body.
    #[inline]
    pub fn has_transpose_body(&self) -> bool {
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

impl<V: TraceValue> Debug for VMapOp<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "VMap")
    }
}

impl<V: TraceValue> Display for VMapOp<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "vmap")
    }
}

impl<V: TraceValue> Op for VMapOp<V> {
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
}

impl<V: TraceValue + FloatExt + ZeroLike + OneLike + MatrixOps + ReshapeOps> Eval<V> for VMapOp<V> {
    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        let abstract_inputs = inputs.iter().map(Typed::tpe).collect::<Vec<_>>();
        let _ = self.abstract_eval(abstract_inputs.as_slice())?;
        self.body.eval_lanes(inputs)
    }
}

impl<V: TransformLeaf> DifferentiableOp<V> for VMapOp<V> {
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
            <Self as Eval<V>>::eval(self, primal_inputs.iter().map(|input| input.value.clone()).collect::<Vec<_>>().as_slice())?;
        let primal_outputs =
            JitTracer::apply_staged_op(primal_inputs.as_slice(), PrimitiveOp::VMap(Box::new(self.clone())), primal_output_values)?;
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
    ) -> Result<Vec<crate::tracing_v2::JvpTracer<V, LinearTerm<V>>>, TraceError> {
        if self.has_transpose_body() {
            return Err(TraceError::HigherOrderOpFailure {
                op: "linearize_program",
                message: "JVP rule for staged op 'vmap' is not implemented".to_string(),
            });
        }
        let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let tangent_inputs = inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
        let primal_outputs = <Self as Eval<V>>::eval(self, primal_inputs.as_slice())?;
        let tangent_outputs = LinearTerm::apply_staged_op(
            tangent_inputs.as_slice(),
            PrimitiveOp::VMap(Box::new(make_linear_vmap(&self.body)?)),
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
    ) -> Result<Vec<Option<crate::tracing_v2::AtomId>>, TraceError> {
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
        let transpose = self.transpose_op()?;
        let output_abstracts = transpose.body().repeated_output_types();
        let output_examples = transpose.body().eval_lanes(
            &output_cotangents.iter().map(|id| builder.atom(*id).expect("cotangent atom should exist").example_value.clone()).collect::<Vec<_>>(),
        )?;
        let contributions = builder.add_equation_prevalidated(
            PrimitiveOp::VMap(Box::new(transpose)),
            output_cotangents.to_vec(),
            output_abstracts,
            output_examples,
        );
        Ok(contributions.into_iter().map(Some).collect::<Vec<_>>())
    }
}

impl<V: TransformLeaf> Eval<JitTracer<V>> for VMapOp<V> {
    fn eval(&self, inputs: &[JitTracer<V>]) -> Result<Vec<JitTracer<V>>, TraceError> {
        let concrete_inputs = inputs.iter().map(|input| input.value.clone()).collect::<Vec<_>>();
        let output_values = <Self as Eval<V>>::eval(self, concrete_inputs.as_slice())?;
        JitTracer::apply_staged_op(inputs, PrimitiveOp::VMap(Box::new(self.clone())), output_values)
    }
}

/// Builds one linearized staged `vmap` op from its primal body.
pub fn make_linear_vmap<V>(body: &FlatTracedVMap<V>) -> Result<VMapOp<V>, TraceError>
where
    V: TraceValue + FloatExt + ZeroLike + OneLike + MatrixOps + ReshapeOps + std::ops::Add<Output = V> + std::ops::Mul<Output = V> + std::ops::Neg<Output = V>,
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
