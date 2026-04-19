//! Higher-order `vmap` operations for [`crate::tracing_v2`].

use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use crate::{
    parameters::{Parameter, Parameterized},
    tracing_v2::{
        LinearProgramOpRef, LinearTerm, Program, ProgramOpRef, TraceError, Traceable, Tracer, ZeroLike,
        engine::Engine,
        linear::{linearize_program, replay_program_linearized_jit, transpose_linear_program_with_output_examples},
    },
    types::{ArrayType, Type, Typed},
};

use super::{
    CoreLinearProgramOp, DifferentiableOp, InterpretableOp, LinearOperation, Op, primitive::LinearPrimitiveOp,
};

/// Hidden staging trait for the `vmap` higher-order primitive.
#[doc(hidden)]
pub trait VMapTracingOperation<T: Type + Display, V: Traceable<T>, L: Clone>: Clone {
    /// Constructs the carrier-specific representation of the `vmap` higher-order primitive with a
    /// captured traced body.
    fn vmap_op(op: VMapOp<T, V, Self, L>) -> Self;
}

/// Hidden staging trait for the `vmap` higher-order primitive in linear programs.
#[doc(hidden)]
pub trait LinearVMapOperation<T: Type + Display, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the linear `vmap` higher-order primitive
    /// with a captured linear traced body.
    fn linear_vmap_op(op: LinearVMapOp<T, V, Self>) -> Self;
}

/// Erased traced `vmap` body used by the staged higher-order op.
pub struct FlatTracedVMap<T: Type, V: Typed<T> + Parameter, O = ProgramOpRef<V>> {
    lane_count: usize,
    input_types: Vec<T>,
    output_types: Vec<T>,
    program: Program<T, V, Vec<V>, Vec<V>, O>,
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
            program: self.program.clone(),
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
        program: Program<T, V, Vec<V>, Vec<V>, O>,
    ) -> Self {
        Self { lane_count, input_types, output_types, program }
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

    /// Returns the flat body program.
    #[inline]
    pub fn program(&self) -> &Program<T, V, Vec<V>, Vec<V>, O> {
        &self.program
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
            outputs.extend(self.program.call(lane_inputs.to_vec())?);
        }
        Ok(outputs)
    }
}

/// Higher-order `vmap` op that carries one canonical forward program payload.
pub struct VMapOp<
    T: Type + Display,
    V: Traceable<T> + Parameter,
    O: Clone = ProgramOpRef<V>,
    L: Clone = LinearProgramOpRef<V>,
> {
    body: FlatTracedVMap<T, V, O>,
    marker: PhantomData<fn() -> L>,
}

impl<T: Type + Display, V: Traceable<T>, O: Clone, L: Clone> Clone for VMapOp<T, V, O, L> {
    fn clone(&self) -> Self {
        Self { body: self.body.clone(), marker: PhantomData }
    }
}

impl<T: Type + Display, V: Traceable<T>, O: Clone, L: Clone> VMapOp<T, V, O, L> {
    /// Builds one ordinary traced `vmap` op.
    #[inline]
    pub fn new(body: FlatTracedVMap<T, V, O>) -> Self {
        Self { body, marker: PhantomData }
    }

    /// Returns the canonical traced body.
    #[inline]
    pub fn body(&self) -> &FlatTracedVMap<T, V, O> {
        &self.body
    }
}

impl<T: Type + Display, V: Traceable<T>, O: Clone, L: Clone> Debug for VMapOp<T, V, O, L> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "VMap")
    }
}

impl<T: Type + Display, V: Traceable<T>, O: Clone, L: Clone> Display for VMapOp<T, V, O, L> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "vmap")
    }
}

impl<V: Traceable<ArrayType>, O: Clone, L: Clone> Op for VMapOp<ArrayType, V, O, L> {
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

impl<V: Traceable<ArrayType>, O: Clone, L: Clone> InterpretableOp<ArrayType, V> for VMapOp<ArrayType, V, O, L>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    O: InterpretableOp<ArrayType, V>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        let abstract_inputs = inputs.iter().map(|input| input.tpe().into_owned()).collect::<Vec<_>>();
        let _ = self.abstract_eval(abstract_inputs.as_slice())?;
        self.body.eval_lanes(inputs)
    }
}

impl<E, V: Traceable<ArrayType> + ZeroLike, O: Clone, L: Clone>
    InterpretableOp<ArrayType, crate::tracing_v2::linear::Linearized<Tracer<ArrayType, V, O, L, E>>>
    for VMapOp<ArrayType, V, O, L>
where
    E: Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized + 'static,
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    O: Op<ArrayType>,
    O: InterpretableOp<ArrayType, V>,
    O: InterpretableOp<ArrayType, crate::tracing_v2::linear::Linearized<Tracer<ArrayType, V, O, L, E>>>,
    O: VMapTracingOperation<ArrayType, V, L>,
    LinearProgramOpRef<Tracer<ArrayType, V, O, L, E>>: CoreLinearProgramOp<Tracer<ArrayType, V, O, L, E>>,
{
    fn interpret(
        &self,
        inputs: &[crate::tracing_v2::linear::Linearized<Tracer<ArrayType, V, O, L, E>>],
    ) -> Result<Vec<crate::tracing_v2::linear::Linearized<Tracer<ArrayType, V, O, L, E>>>, TraceError> {
        let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let primal_outputs = Tracer::apply_staged_op(primal_inputs.as_slice(), O::vmap_op(self.clone()))?;
        let lane_input_count = self.body().input_types().len();
        let mut tangent_outputs = Vec::with_capacity(self.body().total_output_count());
        for lane_inputs in inputs.chunks(lane_input_count) {
            let lane_outputs =
                replay_program_linearized_jit::<_, _, _, O, L, E>(self.body().program(), lane_inputs.to_vec())?;
            tangent_outputs.extend(lane_outputs.into_iter().map(|output| output.tangent));
        }
        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| crate::tracing_v2::JvpTracer { primal, tangent })
            .collect::<Vec<_>>())
    }
}

impl<V: Traceable<ArrayType> + ZeroLike, O: Clone + 'static>
    DifferentiableOp<ArrayType, V, LinearTerm<ArrayType, V, LinearProgramOpRef<V>>, O, LinearProgramOpRef<V>>
    for VMapOp<ArrayType, V, O>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    O: DifferentiableOp<ArrayType, V, LinearTerm<ArrayType, V, LinearProgramOpRef<V>>, O, LinearProgramOpRef<V>>,
    O: InterpretableOp<ArrayType, V>,
    O: InterpretableOp<
            ArrayType,
            crate::tracing_v2::linear::Linearized<
                Tracer<
                    ArrayType,
                    V,
                    O,
                    LinearProgramOpRef<V>,
                    dyn Engine<
                            Type = ArrayType,
                            Value = V,
                            TracingOperation = O,
                            LinearOperation = LinearProgramOpRef<V>,
                        >,
                >,
            >,
        >,
    LinearProgramOpRef<V>: CoreLinearProgramOp<V>,
    LinearProgramOpRef<
        Tracer<
            ArrayType,
            V,
            O,
            LinearProgramOpRef<V>,
            dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = LinearProgramOpRef<V>>,
        >,
    >:CoreLinearProgramOp<
        Tracer<
            ArrayType,
            V,
            O,
            LinearProgramOpRef<V>,
            dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = LinearProgramOpRef<V>>,
        >,
    >,
{
    fn jvp(
        &self,
        engine: &dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = LinearProgramOpRef<V>>,
        inputs: &[crate::tracing_v2::JvpTracer<V, LinearTerm<ArrayType, V, LinearProgramOpRef<V>>>],
    ) -> Result<Vec<crate::tracing_v2::JvpTracer<V, LinearTerm<ArrayType, V, LinearProgramOpRef<V>>>>, TraceError> {
        let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let tangent_inputs = inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
        let primal_outputs = <Self as InterpretableOp<ArrayType, V>>::interpret(self, primal_inputs.as_slice())?;
        let lane_input_count = self.body.input_types().len();
        let lane_primals = primal_inputs.iter().take(lane_input_count).cloned().collect::<Vec<_>>();
        let tangent_outputs = LinearTerm::apply_staged_op(
            tangent_inputs.as_slice(),
            LinearPrimitiveOp::VMap(Box::new(make_linear_vmap(engine, &self.body, lane_primals)?)),
            self.body.total_output_count(),
        )?;
        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| crate::tracing_v2::JvpTracer { primal, tangent })
            .collect::<Vec<_>>())
    }
}

impl<
    V: Traceable<ArrayType>,
    O: Clone,
    L: Clone,
    E: Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized,
> InterpretableOp<ArrayType, Tracer<ArrayType, V, O, L, E>> for VMapOp<ArrayType, V, O, L>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    O: Op<ArrayType> + InterpretableOp<ArrayType, V> + VMapTracingOperation<ArrayType, V, L>,
{
    fn interpret(
        &self,
        inputs: &[Tracer<ArrayType, V, O, L, E>],
    ) -> Result<Vec<Tracer<ArrayType, V, O, L, E>>, TraceError> {
        Tracer::apply_staged_op(inputs, O::vmap_op(self.clone()))
    }
}

/// Linear-only `vmap` op that always carries both the linear body and its transpose body.
pub struct LinearVMapOp<T: Type + Display, V: Traceable<T> + Parameter, O: Clone = LinearProgramOpRef<V>> {
    body: FlatTracedVMap<T, V, O>,
    transpose_body: FlatTracedVMap<T, V, O>,
}

impl<T: Type + Display, V: Traceable<T>, O: Clone> Clone for LinearVMapOp<T, V, O> {
    fn clone(&self) -> Self {
        Self { body: self.body.clone(), transpose_body: self.transpose_body.clone() }
    }
}

impl<T: Type + Display, V: Traceable<T>, O: Clone> LinearVMapOp<T, V, O> {
    /// Builds one linear traced `vmap` op with its transpose body.
    #[inline]
    pub fn new(body: FlatTracedVMap<T, V, O>, transpose_body: FlatTracedVMap<T, V, O>) -> Self {
        Self { body, transpose_body }
    }

    /// Returns the canonical traced body.
    #[inline]
    pub fn body(&self) -> &FlatTracedVMap<T, V, O> {
        &self.body
    }

    fn transpose_op(&self) -> Self {
        Self::new(self.transpose_body.clone(), self.body.clone())
    }
}

impl<T: Type + Display, V: Traceable<T>, O: Clone> Debug for LinearVMapOp<T, V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "LinearVMap")
    }
}

impl<T: Type + Display, V: Traceable<T>, O: Clone> Display for LinearVMapOp<T, V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "vmap")
    }
}

impl<V: Traceable<ArrayType>, O: Clone> Op for LinearVMapOp<ArrayType, V, O> {
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

impl<V: Traceable<ArrayType>, O: Clone> InterpretableOp<ArrayType, V> for LinearVMapOp<ArrayType, V, O>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    O: InterpretableOp<ArrayType, V>,
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
pub(crate) fn make_linear_vmap<V, O>(
    engine: &dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = LinearProgramOpRef<V>>,
    body: &FlatTracedVMap<ArrayType, V, O>,
    input_primals: Vec<V>,
) -> Result<LinearVMapOp<ArrayType, V>, TraceError>
where
    V: Traceable<ArrayType> + ZeroLike,
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    O: Clone + Op<ArrayType> + 'static,
    O: InterpretableOp<ArrayType, V>,
    O: DifferentiableOp<ArrayType, V, LinearTerm<ArrayType, V, LinearProgramOpRef<V>>, O, LinearProgramOpRef<V>>,
    O: InterpretableOp<
            ArrayType,
            crate::tracing_v2::linear::Linearized<
                Tracer<
                    ArrayType,
                    V,
                    O,
                    LinearProgramOpRef<V>,
                    dyn Engine<
                            Type = ArrayType,
                            Value = V,
                            TracingOperation = O,
                            LinearOperation = LinearProgramOpRef<V>,
                        >,
                >,
            >,
        >,
    LinearProgramOpRef<V>: CoreLinearProgramOp<V>,
    LinearProgramOpRef<
        Tracer<
            ArrayType,
            V,
            O,
            LinearProgramOpRef<V>,
            dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = LinearProgramOpRef<V>>,
        >,
    >:CoreLinearProgramOp<
        Tracer<
            ArrayType,
            V,
            O,
            LinearProgramOpRef<V>,
            dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = LinearProgramOpRef<V>>,
        >,
    >,
{
    let output_primals = body.program.call(input_primals.clone())?;
    let pushforward = linearize_program(engine, body.program(), input_primals)?;
    let pullback = transpose_linear_program_with_output_examples(&pushforward, output_primals.as_slice())?;
    Ok(LinearVMapOp::new(
        FlatTracedVMap::from_parts(
            body.lane_count,
            body.input_types.clone(),
            body.output_types.clone(),
            pushforward.program().clone(),
        ),
        FlatTracedVMap::from_parts(
            body.lane_count,
            body.output_types.clone(),
            body.input_types.clone(),
            pullback.program().clone(),
        ),
    ))
}
