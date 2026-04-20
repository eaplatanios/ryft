//! Higher-order `vmap` operations for [`crate::tracing_v2`].
//!
//! The public [`crate::tracing_v2::vmap`] transform lives in [`crate::tracing_v2::batch`], but the
//! traced representation of that transform lives here. These types let batching survive as a
//! first-class node inside staged programs instead of being lowered immediately into repeated scalar
//! work.

use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use crate::{
    parameters::{Parameter, Parameterized, Placeholder},
    tracing_v2::{
        Atom, AtomId, Instruction, LinearPrimitiveOperation, LinearTerm, PrimitiveOperation, Program, Traceable,
        Tracer, TracingError, Value, ZeroLike,
        engine::Engine,
        linear::{linearize_program, replay_program_linearized_jit, transpose_linear_program_with_output_examples},
    },
    types::{ArrayType, Type},
};

use super::{CoreLinearProgramOperation, DifferentiableOperation, InterpretableOperation, LinearOperation, Operation};

/// Hidden staging trait for the `vmap` higher-order primitive.
#[doc(hidden)]
pub trait VMapTracingOperation<T: Type + Display, V: Traceable<T>, L: Clone>: Clone {
    /// Constructs the carrier-specific representation of the `vmap` higher-order primitive with a
    /// captured traced body.
    fn vmap_op(op: VMapOperation<T, V, Self, L>) -> Self;
}

/// Hidden staging trait for the `vmap` higher-order primitive in linear programs.
#[doc(hidden)]
pub trait LinearVMapCarrierOperation<T: Type + Display, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the linear `vmap` higher-order primitive
    /// with a captured linear traced body.
    fn linear_vmap_op(op: LinearVMapOperation<T, V, Self>) -> Self;
}

/// Erased traced `vmap` body used by the staged higher-order op.
///
/// The body is stored in a flattened, lane-agnostic form so the higher-order op can be cloned,
/// replayed, transposed, and lowered without carrying the caller's original structured parameter
/// types around.
#[derive(Clone)]
pub struct FlatTracedVMap<T: Type, V: Traceable<T>, O = PrimitiveOperation<ArrayType, V>> {
    /// Number of logical lanes represented by this flattened batched body.
    lane_count: usize,

    /// Canonical per-lane input types of the captured body.
    input_types: Vec<T>,

    /// Canonical per-lane output types of the captured body.
    output_types: Vec<T>,

    /// Atom table of the flattened staged body.
    atoms: Vec<Atom<T, V>>,

    /// Input atom ids of the flattened staged body.
    input_ids: Vec<AtomId>,

    /// Output atom ids of the flattened staged body.
    output_ids: Vec<AtomId>,

    /// Instructions of the flattened staged body.
    instructions: Vec<Instruction<O>>,
}

impl<T: Type, V: Traceable<T>, O: Clone> FlatTracedVMap<T, V, O> {
    /// Builds one erased traced `vmap` body from explicit staged parts.
    #[inline]
    pub fn from_parts(
        lane_count: usize,
        input_types: Vec<T>,
        output_types: Vec<T>,
        program: Program<T, V, O, Vec<V>, Vec<V>>,
    ) -> Self
    where
        O: Operation<T>,
    {
        let Program { atoms, input_ids, output_ids, instructions, .. } = program;
        Self { lane_count, input_types, output_types, atoms, input_ids, output_ids, instructions }
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

    /// Returns a cloned flat body program.
    #[inline]
    pub fn program(&self) -> Program<T, V, O, Vec<V>, Vec<V>>
    where
        O: Operation<T>,
    {
        Program {
            atoms: self.atoms.clone(),
            input_ids: self.input_ids.clone(),
            output_ids: self.output_ids.clone(),
            instructions: self.instructions.clone(),
            input_structure: vec![Placeholder; self.input_types.len()],
            output_structure: vec![Placeholder; self.output_types.len()],
            marker: PhantomData,
        }
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

    pub(crate) fn eval_lanes(&self, inputs: &[V]) -> Result<Vec<V>, TracingError>
    where
        O: InterpretableOperation<T, V>,
        Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    {
        if inputs.len() != self.total_input_count() {
            return Err(TracingError::InvalidInputCount { expected: self.total_input_count(), got: inputs.len() });
        }

        let lane_input_count = self.input_types.len();
        let mut outputs = Vec::with_capacity(self.total_output_count());
        for lane_inputs in inputs.chunks(lane_input_count) {
            outputs.extend(self.program().interpret(lane_inputs.to_vec())?);
        }
        Ok(outputs)
    }
}

/// Higher-order `vmap` op that carries one canonical forward program payload.
///
/// Ordinary traced programs store [`VMapOperation`] when vectorization is preserved symbolically instead
/// of being unrolled into repeated scalar instructions.
#[derive(Clone)]
pub struct VMapOperation<
    T: Type + Display,
    V: Traceable<T> + Parameter,
    O: Clone = PrimitiveOperation<ArrayType, V>,
    L: Clone = LinearPrimitiveOperation<ArrayType, V>,
> {
    /// Captured flattened forward body for the batched computation.
    body: FlatTracedVMap<T, V, O>,

    /// Phantom marker tying the op to the linear carrier used by nested transforms.
    marker: PhantomData<fn() -> L>,
}

impl<T: Type + Display, V: Traceable<T>, O: Clone, L: Clone> VMapOperation<T, V, O, L> {
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

impl<T: Type + Display, V: Traceable<T>, O: Clone, L: Clone> Debug for VMapOperation<T, V, O, L> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "VMap")
    }
}

impl<T: Type + Display, V: Traceable<T>, O: Clone, L: Clone> Display for VMapOperation<T, V, O, L> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "vmap")
    }
}

impl<V: Traceable<ArrayType>, O: Clone + Operation<ArrayType>, L: Clone> Operation
    for VMapOperation<ArrayType, V, O, L>
{
    fn name(&self) -> &'static str {
        "vmap"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TracingError> {
        let expected_inputs = self.body.repeated_input_types();
        if input_types.len() != expected_inputs.len() {
            return Err(TracingError::InvalidInputCount { expected: expected_inputs.len(), got: input_types.len() });
        }
        if input_types != expected_inputs.as_slice() {
            return Err(TracingError::IncompatibleAbstractValues { op: "vmap" });
        }
        Ok(self.body.repeated_output_types())
    }
}

impl<V: Traceable<ArrayType>, O: Clone + Operation<ArrayType>, L: Clone> InterpretableOperation<ArrayType, V>
    for VMapOperation<ArrayType, V, O, L>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    O: InterpretableOperation<ArrayType, V>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        let abstract_inputs = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let _ = self.infer_output_types(abstract_inputs.as_slice())?;
        self.body.eval_lanes(inputs)
    }
}

impl<'engine, E, V: Value<ArrayType> + ZeroLike, O: Clone + 'static, L: Clone + Operation<ArrayType> + 'static>
    InterpretableOperation<ArrayType, crate::tracing_v2::linear::Linearized<Tracer<'engine, E>>>
    for VMapOperation<ArrayType, V, O, L>
where
    E: Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized + 'static,
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    O: Operation<ArrayType>,
    O: InterpretableOperation<ArrayType, V>,
    O: InterpretableOperation<ArrayType, crate::tracing_v2::linear::Linearized<Tracer<'engine, E>>>,
    O: VMapTracingOperation<ArrayType, V, L>,
    LinearPrimitiveOperation<ArrayType, Tracer<'engine, E>>: CoreLinearProgramOperation<Tracer<'engine, E>>,
{
    fn interpret(
        &self,
        inputs: &[crate::tracing_v2::linear::Linearized<Tracer<'engine, E>>],
    ) -> Result<Vec<crate::tracing_v2::linear::Linearized<Tracer<'engine, E>>>, TracingError> {
        let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let exemplar_primal_input = primal_inputs.first().ok_or(TracingError::EmptyParameterizedValue)?.clone();
        let primal_outputs = Tracer::apply_staged_op(
            exemplar_primal_input.engine,
            exemplar_primal_input.builder.clone(),
            primal_inputs.as_slice(),
            O::vmap_op(self.clone()),
        )?;
        let lane_input_count = self.body().input_types().len();
        let mut tangent_outputs = Vec::with_capacity(self.body().total_output_count());
        for lane_inputs in inputs.chunks(lane_input_count) {
            let lane_program = self.body().program();
            let lane_outputs = replay_program_linearized_jit::<_, _, _, O, L, E>(&lane_program, lane_inputs.to_vec())?;
            tangent_outputs.extend(lane_outputs.into_iter().map(|output| output.tangent));
        }
        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| crate::tracing_v2::JvpTracer { primal, tangent })
            .collect::<Vec<_>>())
    }
}

impl<V: Value<ArrayType> + ZeroLike + 'static, O: Clone + 'static>
    DifferentiableOperation<
        ArrayType,
        V,
        LinearTerm<ArrayType, V, LinearPrimitiveOperation<ArrayType, V>>,
        O,
        LinearPrimitiveOperation<ArrayType, V>,
    > for VMapOperation<ArrayType, V, O>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    O: DifferentiableOperation<
            ArrayType,
            V,
            LinearTerm<ArrayType, V, LinearPrimitiveOperation<ArrayType, V>>,
            O,
            LinearPrimitiveOperation<ArrayType, V>,
        >,
    O: InterpretableOperation<ArrayType, V>,
    O: for<'call> InterpretableOperation<
            ArrayType,
            crate::tracing_v2::linear::Linearized<
                Tracer<
                    'call,
                    dyn Engine<
                            Type = ArrayType,
                            Value = V,
                            TracingOperation = O,
                            LinearOperation = LinearPrimitiveOperation<ArrayType, V>,
                        >,
                >,
            >,
        >,
    LinearPrimitiveOperation<ArrayType, V>: CoreLinearProgramOperation<V>,
    for<'call> LinearPrimitiveOperation<
        ArrayType,
        Tracer<
            'call,
            dyn Engine<
                    Type = ArrayType,
                    Value = V,
                    TracingOperation = O,
                    LinearOperation = LinearPrimitiveOperation<ArrayType, V>,
                >,
        >,
    >:CoreLinearProgramOperation<
        Tracer<
            'call,
            dyn Engine<
                    Type = ArrayType,
                    Value = V,
                    TracingOperation = O,
                    LinearOperation = LinearPrimitiveOperation<ArrayType, V>,
                >,
        >,
    >,
{
    fn jvp(
        &self,
        engine: &dyn Engine<
            Type = ArrayType,
            Value = V,
            TracingOperation = O,
            LinearOperation = LinearPrimitiveOperation<ArrayType, V>,
        >,
        inputs: &[crate::tracing_v2::JvpTracer<V, LinearTerm<ArrayType, V, LinearPrimitiveOperation<ArrayType, V>>>],
    ) -> Result<
        Vec<crate::tracing_v2::JvpTracer<V, LinearTerm<ArrayType, V, LinearPrimitiveOperation<ArrayType, V>>>>,
        TracingError,
    > {
        let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let tangent_inputs = inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
        let primal_outputs = <Self as InterpretableOperation<ArrayType, V>>::interpret(self, primal_inputs.as_slice())?;
        let lane_input_count = self.body.input_types().len();
        let lane_primals = primal_inputs.iter().take(lane_input_count).cloned().collect::<Vec<_>>();
        let tangent_builder = tangent_inputs.first().ok_or(TracingError::EmptyParameterizedValue)?.builder.clone();
        let tangent_outputs = LinearTerm::apply_staged_op(
            tangent_builder,
            tangent_inputs.as_slice(),
            LinearPrimitiveOperation::VMap(Box::new(make_linear_vmap(engine, &self.body, lane_primals)?)),
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
    'engine,
    V: Value<ArrayType>,
    O: Clone,
    L: Clone,
    E: Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized,
> InterpretableOperation<ArrayType, Tracer<'engine, E>> for VMapOperation<ArrayType, V, O, L>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    O: Operation<ArrayType> + InterpretableOperation<ArrayType, V> + VMapTracingOperation<ArrayType, V, L>,
{
    fn interpret(&self, inputs: &[Tracer<'engine, E>]) -> Result<Vec<Tracer<'engine, E>>, TracingError> {
        let exemplar_input = inputs.first().ok_or(TracingError::EmptyParameterizedValue)?.clone();
        Tracer::apply_staged_op(exemplar_input.engine, exemplar_input.builder.clone(), inputs, O::vmap_op(self.clone()))
    }
}

/// Linear-only `vmap` op that always carries both the linear body and its transpose body.
/// Higher-order linear `vmap` op that carries both a forward body and a transpose body.
///
/// Linear programs need slightly more structure than ordinary programs because reverse-mode
/// transposition must know how to batch both the forward linear map and its transpose.
#[derive(Clone)]
pub struct LinearVMapOperation<
    T: Type + Display,
    V: Traceable<T> + Parameter,
    O: Clone = LinearPrimitiveOperation<ArrayType, V>,
> {
    /// Captured flattened forward linear body.
    body: FlatTracedVMap<T, V, O>,

    /// Captured flattened transpose body used for reverse-mode batching.
    transpose_body: FlatTracedVMap<T, V, O>,
}

impl<T: Type + Display, V: Traceable<T>, O: Clone> LinearVMapOperation<T, V, O> {
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

impl<T: Type + Display, V: Traceable<T>, O: Clone> Debug for LinearVMapOperation<T, V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "LinearVMap")
    }
}

impl<T: Type + Display, V: Traceable<T>, O: Clone> Display for LinearVMapOperation<T, V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "vmap")
    }
}

impl<V: Traceable<ArrayType>, O: Clone + Operation<ArrayType>> Operation for LinearVMapOperation<ArrayType, V, O> {
    fn name(&self) -> &'static str {
        "vmap"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TracingError> {
        let expected_inputs = self.body.repeated_input_types();
        if input_types.len() != expected_inputs.len() {
            return Err(TracingError::InvalidInputCount { expected: expected_inputs.len(), got: input_types.len() });
        }
        if input_types != expected_inputs.as_slice() {
            return Err(TracingError::IncompatibleAbstractValues { op: "vmap" });
        }
        Ok(self.body.repeated_output_types())
    }
}

impl<V: Traceable<ArrayType>, O: Clone + Operation<ArrayType>> InterpretableOperation<ArrayType, V>
    for LinearVMapOperation<ArrayType, V, O>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    O: InterpretableOperation<ArrayType, V>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        let abstract_inputs = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let _ = self.infer_output_types(abstract_inputs.as_slice())?;
        self.body.eval_lanes(inputs)
    }
}

impl<V: Traceable<ArrayType>> LinearOperation<ArrayType, V> for LinearVMapOperation<ArrayType, V> {
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, V>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, V>>>, TracingError> {
        if output_cotangents.len() != self.body.total_output_count() {
            return Err(TracingError::InvalidInputCount {
                expected: self.body.total_output_count(),
                got: output_cotangents.len(),
            });
        }
        let transpose = self.transpose_op();
        let exemplar_output_cotangent = output_cotangents.first().ok_or(TracingError::EmptyParameterizedValue)?.clone();
        Ok(LinearTerm::apply_staged_op(
            exemplar_output_cotangent.builder.clone(),
            output_cotangents,
            LinearPrimitiveOperation::VMap(Box::new(transpose)),
            self.body.total_input_count(),
        )?
        .into_iter()
        .map(Some)
        .collect::<Vec<_>>())
    }
}

/// Builds one linearized staged `vmap` op from its primal body at the provided primal inputs.
#[allow(private_bounds)]
pub(crate) fn make_linear_vmap<'engine, V, O>(
    engine: &dyn Engine<
        Type = ArrayType,
        Value = V,
        TracingOperation = O,
        LinearOperation = LinearPrimitiveOperation<ArrayType, V>,
    >,
    body: &FlatTracedVMap<ArrayType, V, O>,
    input_primals: Vec<V>,
) -> Result<LinearVMapOperation<ArrayType, V>, TracingError>
where
    V: Traceable<ArrayType> + ZeroLike + 'static,
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    O: Clone + Operation<ArrayType> + 'static,
    O: InterpretableOperation<ArrayType, V>,
    O: DifferentiableOperation<
            ArrayType,
            V,
            LinearTerm<ArrayType, V, LinearPrimitiveOperation<ArrayType, V>>,
            O,
            LinearPrimitiveOperation<ArrayType, V>,
        >,
    O: for<'call> InterpretableOperation<
            ArrayType,
            crate::tracing_v2::linear::Linearized<
                Tracer<
                    'call,
                    dyn Engine<
                            Type = ArrayType,
                            Value = V,
                            TracingOperation = O,
                            LinearOperation = LinearPrimitiveOperation<ArrayType, V>,
                        >,
                >,
            >,
        >,
    LinearPrimitiveOperation<ArrayType, V>: CoreLinearProgramOperation<V>,
    for<'call> LinearPrimitiveOperation<
        ArrayType,
        Tracer<
            'call,
            dyn Engine<
                    Type = ArrayType,
                    Value = V,
                    TracingOperation = O,
                    LinearOperation = LinearPrimitiveOperation<ArrayType, V>,
                >,
        >,
    >:CoreLinearProgramOperation<
        Tracer<
            'call,
            dyn Engine<
                    Type = ArrayType,
                    Value = V,
                    TracingOperation = O,
                    LinearOperation = LinearPrimitiveOperation<ArrayType, V>,
                >,
        >,
    >,
{
    let body_program = body.program();
    let output_primals = body_program.interpret(input_primals.clone())?;
    let pushforward = linearize_program(engine, &body_program, input_primals)?;
    let pullback = transpose_linear_program_with_output_examples(engine, &pushforward, output_primals.as_slice())?;
    Ok(LinearVMapOperation::new(
        FlatTracedVMap::from_parts(
            body.lane_count,
            body.input_types.clone(),
            body.output_types.clone(),
            pushforward.clone(),
        ),
        FlatTracedVMap::from_parts(
            body.lane_count,
            body.output_types.clone(),
            body.input_types.clone(),
            pullback.clone(),
        ),
    ))
}
