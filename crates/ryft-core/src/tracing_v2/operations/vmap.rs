//! Higher-order `vmap` operations for [`crate::tracing_v2`].

use std::fmt::{Debug, Display};

use crate::{
    parameters::{Parameter, Parameterized},
    tracing_v2::{
        CompiledFunction, JitTracer, LinearTerm, TraceError, Traceable, ZeroLike,
        engine::Engine,
        linear::{
            linearize_program, replay_program_graph_linearized_jit, transpose_linear_program_with_output_examples,
        },
        ops::{
            CoreLinearProgramOp, CoreOperationSet, DifferentiableOp, InterpretableOp, LinearOperation,
            LinearPrimitiveOp, Op, OperationSet, SupportsCoreSyntax, SupportsVMap,
        },
        program::{LinearProgramOpRef, ProgramOpRef},
    },
    types::{ArrayType, Type, Typed},
};

/// Erased traced `vmap` body used by the staged higher-order op.
pub struct FlatTracedVMap<T: Type, V: Typed<T> + Parameter, O = ProgramOpRef<V>> {
    lane_count: usize,
    input_types: Vec<T>,
    output_types: Vec<T>,
    compiled: CompiledFunction<T, V, Vec<V>, Vec<V>, O>,
}

impl<T: Type, V: Traceable<T>, O: Clone> Clone for FlatTracedVMap<T, V, O>
where
    <Vec<V> as Parameterized<V>>::ParameterStructure: Clone,
{
    fn clone(&self) -> Self {
        Self {
            lane_count: self.lane_count,
            input_types: self.input_types.clone(),
            output_types: self.output_types.clone(),
            compiled: self.compiled.clone(),
        }
    }
}

impl<T: Type, V: Traceable<T>, O: Clone> FlatTracedVMap<T, V, O> {
    /// Builds one erased traced `vmap` body from explicit staged parts.
    #[inline]
    pub fn from_parts(
        lane_count: usize,
        input_types: Vec<T>,
        output_types: Vec<T>,
        compiled: CompiledFunction<T, V, Vec<V>, Vec<V>, O>,
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
    pub fn input_types(&self) -> &[T] {
        self.input_types.as_slice()
    }

    /// Returns the canonical per-lane output types.
    #[inline]
    pub fn output_types(&self) -> &[T] {
        self.output_types.as_slice()
    }

    /// Returns the compiled flat body.
    #[inline]
    pub fn compiled(&self) -> &CompiledFunction<T, V, Vec<V>, Vec<V>, O> {
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

    pub(crate) fn repeated_input_types(&self) -> Vec<T> {
        (0..self.lane_count).flat_map(|_| self.input_types.iter().cloned()).collect::<Vec<_>>()
    }

    pub(crate) fn repeated_output_types(&self) -> Vec<T> {
        (0..self.lane_count).flat_map(|_| self.output_types.iter().cloned()).collect::<Vec<_>>()
    }

    pub(crate) fn eval_lanes(&self, inputs: &[V]) -> Result<Vec<V>, TraceError>
    where
        O: InterpretableOp<T, V>,
        Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
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

/// Higher-order `vmap` op that carries one canonical forward program payload.
pub struct VMapOp<T: Type + Display, V: Traceable<T> + Parameter, S: OperationSet<T, V> = CoreOperationSet> {
    body: FlatTracedVMap<T, V, <S as OperationSet<T, V>>::TracingOperation>,
}

impl<T: Type + Display, V: Traceable<T>, S: OperationSet<T, V>> Clone for VMapOp<T, V, S> {
    fn clone(&self) -> Self {
        Self { body: self.body.clone() }
    }
}

impl<T: Type + Display, V: Traceable<T>, S: OperationSet<T, V>> VMapOp<T, V, S> {
    /// Builds one ordinary traced `vmap` op.
    #[inline]
    pub fn new(body: FlatTracedVMap<T, V, <S as OperationSet<T, V>>::TracingOperation>) -> Self {
        Self { body }
    }

    /// Returns the canonical traced body.
    #[inline]
    pub fn body(&self) -> &FlatTracedVMap<T, V, <S as OperationSet<T, V>>::TracingOperation> {
        &self.body
    }
}

impl<T: Type + Display, V: Traceable<T>, S: OperationSet<T, V>> Debug for VMapOp<T, V, S> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "VMap")
    }
}

impl<T: Type + Display, V: Traceable<T>, S: OperationSet<T, V>> Display for VMapOp<T, V, S> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "vmap")
    }
}

impl<V: Traceable<ArrayType>, S: OperationSet<ArrayType, V>> Op for VMapOp<ArrayType, V, S> {
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

impl<V: Traceable<ArrayType>, S: OperationSet<ArrayType, V>> InterpretableOp<ArrayType, V> for VMapOp<ArrayType, V, S>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    <S as OperationSet<ArrayType, V>>::TracingOperation: InterpretableOp<ArrayType, V>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        let abstract_inputs = inputs.iter().map(|input| input.tpe().into_owned()).collect::<Vec<_>>();
        let _ = self.abstract_eval(abstract_inputs.as_slice())?;
        self.body.eval_lanes(inputs)
    }
}

impl<V: Traceable<ArrayType> + ZeroLike, S: OperationSet<ArrayType, V> + SupportsVMap<ArrayType, V>>
    InterpretableOp<ArrayType, crate::tracing_v2::linear::Linearized<JitTracer<ArrayType, V, S>>>
    for VMapOp<ArrayType, V, S>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    S::TracingOperation: Op<ArrayType>,
    S::TracingOperation: InterpretableOp<ArrayType, V>,
    S::TracingOperation: InterpretableOp<ArrayType, crate::tracing_v2::linear::Linearized<JitTracer<ArrayType, V, S>>>,
{
    fn interpret(
        &self,
        inputs: &[crate::tracing_v2::linear::Linearized<JitTracer<ArrayType, V, S>>],
    ) -> Result<Vec<crate::tracing_v2::linear::Linearized<JitTracer<ArrayType, V, S>>>, TraceError> {
        let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let primal_output_values = <Self as InterpretableOp<ArrayType, V>>::interpret(
            self,
            primal_inputs.iter().map(|input| input.value.clone()).collect::<Vec<_>>().as_slice(),
        )?;
        let primal_outputs =
            JitTracer::apply_staged_op(primal_inputs.as_slice(), S::vmap_op(self.clone()), primal_output_values)?;
        let lane_input_count = self.body().input_types().len();
        let mut tangent_outputs = Vec::with_capacity(self.body().total_output_count());
        for lane_inputs in inputs.chunks(lane_input_count) {
            let lane_outputs = replay_program_graph_linearized_jit::<_, _, _, S>(
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
}

impl<V: Traceable<ArrayType> + ZeroLike, S: OperationSet<ArrayType, V> + SupportsCoreSyntax<ArrayType, V>>
    DifferentiableOp<ArrayType, V, LinearTerm<ArrayType, V>, S> for VMapOp<ArrayType, V, S>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    S::TracingOperation: InterpretableOp<ArrayType, V>,
    S::TracingOperation: DifferentiableOp<ArrayType, V, LinearTerm<ArrayType, V>, S>,
    S::TracingOperation: InterpretableOp<ArrayType, crate::tracing_v2::linear::Linearized<JitTracer<ArrayType, V, S>>>,
    LinearProgramOpRef<V>: CoreLinearProgramOp<V>,
{
    fn jvp(
        &self,
        engine: &dyn Engine<Type = ArrayType, Value = V, OperationSet = S>,
        inputs: &[crate::tracing_v2::JvpTracer<V, LinearTerm<ArrayType, V>>],
    ) -> Result<Vec<crate::tracing_v2::JvpTracer<V, LinearTerm<ArrayType, V>>>, TraceError> {
        let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let tangent_inputs = inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
        let primal_outputs = <Self as InterpretableOp<ArrayType, V>>::interpret(self, primal_inputs.as_slice())?;
        let lane_input_count = self.body.input_types().len();
        let lane_primals = primal_inputs.iter().take(lane_input_count).cloned().collect::<Vec<_>>();
        let tangent_outputs = LinearTerm::apply_staged_op(
            tangent_inputs.as_slice(),
            LinearPrimitiveOp::VMap(Box::new(make_linear_vmap::<_, S>(engine, &self.body, lane_primals)?)),
            self.body.total_output_count(),
        )?;
        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| crate::tracing_v2::JvpTracer { primal, tangent })
            .collect::<Vec<_>>())
    }
}

impl<V: Traceable<ArrayType>, S: OperationSet<ArrayType, V> + SupportsVMap<ArrayType, V>>
    InterpretableOp<ArrayType, JitTracer<ArrayType, V, S>> for VMapOp<ArrayType, V, S>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    S::TracingOperation: Op<ArrayType>,
    S::TracingOperation: InterpretableOp<ArrayType, V>,
{
    fn interpret(&self, inputs: &[JitTracer<ArrayType, V, S>]) -> Result<Vec<JitTracer<ArrayType, V, S>>, TraceError> {
        let concrete_inputs = inputs.iter().map(|input| input.value.clone()).collect::<Vec<_>>();
        let output_values = <Self as InterpretableOp<ArrayType, V>>::interpret(self, concrete_inputs.as_slice())?;
        JitTracer::apply_staged_op(inputs, S::vmap_op(self.clone()), output_values)
    }
}

/// Linear-only `vmap` op that always carries both the linear body and its transpose body.
pub struct LinearVMapOp<T: Type + Display, V: Traceable<T> + Parameter, S: OperationSet<T, V> = CoreOperationSet> {
    body: FlatTracedVMap<T, V, <S as OperationSet<T, V>>::LinearOperation>,
    transpose_body: FlatTracedVMap<T, V, <S as OperationSet<T, V>>::LinearOperation>,
}

impl<T: Type + Display, V: Traceable<T>, S: OperationSet<T, V>> Clone for LinearVMapOp<T, V, S> {
    fn clone(&self) -> Self {
        Self { body: self.body.clone(), transpose_body: self.transpose_body.clone() }
    }
}

impl<T: Type + Display, V: Traceable<T>, S: OperationSet<T, V>> LinearVMapOp<T, V, S> {
    /// Builds one linear traced `vmap` op with its transpose body.
    #[inline]
    pub fn new(
        body: FlatTracedVMap<T, V, <S as OperationSet<T, V>>::LinearOperation>,
        transpose_body: FlatTracedVMap<T, V, <S as OperationSet<T, V>>::LinearOperation>,
    ) -> Self {
        Self { body, transpose_body }
    }

    /// Returns the canonical traced body.
    #[inline]
    pub fn body(&self) -> &FlatTracedVMap<T, V, <S as OperationSet<T, V>>::LinearOperation> {
        &self.body
    }

    fn transpose_op(&self) -> Self {
        Self::new(self.transpose_body.clone(), self.body.clone())
    }
}

impl<T: Type + Display, V: Traceable<T>, S: OperationSet<T, V>> Debug for LinearVMapOp<T, V, S> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "LinearVMap")
    }
}

impl<T: Type + Display, V: Traceable<T>, S: OperationSet<T, V>> Display for LinearVMapOp<T, V, S> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "vmap")
    }
}

impl<V: Traceable<ArrayType>, S: OperationSet<ArrayType, V>> Op for LinearVMapOp<ArrayType, V, S> {
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

impl<V: Traceable<ArrayType>, S: OperationSet<ArrayType, V>> InterpretableOp<ArrayType, V>
    for LinearVMapOp<ArrayType, V, S>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    <S as OperationSet<ArrayType, V>>::LinearOperation: InterpretableOp<ArrayType, V>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        let abstract_inputs = inputs.iter().map(|input| input.tpe().into_owned()).collect::<Vec<_>>();
        let _ = self.abstract_eval(abstract_inputs.as_slice())?;
        self.body.eval_lanes(inputs)
    }
}

impl<V: Traceable<ArrayType>> LinearOperation<ArrayType, V> for LinearVMapOp<ArrayType, V> {
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, V>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, V>>>, TraceError> {
        if output_cotangents.len() != self.body.total_output_count() {
            return Err(TraceError::InvalidInputCount {
                expected: self.body.total_output_count(),
                got: output_cotangents.len(),
            });
        }
        let transpose = self.transpose_op();
        Ok(LinearTerm::apply_staged_op(
            output_cotangents,
            LinearPrimitiveOp::VMap(Box::new(transpose)),
            self.body.total_input_count(),
        )?
        .into_iter()
        .map(Some)
        .collect::<Vec<_>>())
    }
}

/// Builds one linearized staged `vmap` op from its primal body at the provided primal inputs.
#[allow(private_bounds)]
pub(crate) fn make_linear_vmap<V, S: OperationSet<ArrayType, V> + SupportsCoreSyntax<ArrayType, V>>(
    engine: &dyn Engine<Type = ArrayType, Value = V, OperationSet = S>,
    body: &FlatTracedVMap<ArrayType, V, <S as OperationSet<ArrayType, V>>::TracingOperation>,
    input_primals: Vec<V>,
) -> Result<LinearVMapOp<ArrayType, V>, TraceError>
where
    V: Traceable<ArrayType> + ZeroLike,
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    <S as OperationSet<ArrayType, V>>::TracingOperation: Op<ArrayType>,
    <S as OperationSet<ArrayType, V>>::TracingOperation: InterpretableOp<ArrayType, V>,
    <S as OperationSet<ArrayType, V>>::TracingOperation: DifferentiableOp<ArrayType, V, LinearTerm<ArrayType, V>, S>,
    <S as OperationSet<ArrayType, V>>::TracingOperation:
        InterpretableOp<ArrayType, crate::tracing_v2::linear::Linearized<JitTracer<ArrayType, V, S>>>,
    LinearProgramOpRef<V>: CoreLinearProgramOp<V>,
{
    let output_primals = body.compiled.call(input_primals.clone())?;
    let pushforward = linearize_program::<_, _, _, S>(engine, body.compiled.program(), input_primals)?;
    let pullback = transpose_linear_program_with_output_examples(&pushforward, output_primals.as_slice())?;
    Ok(LinearVMapOp::new(
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
