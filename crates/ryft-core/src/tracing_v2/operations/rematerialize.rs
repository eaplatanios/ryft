//! Higher-order `rematerialize` operation for [`crate::tracing_v2`].
//!
//! This module gives staged programs an explicit rematerialization boundary. The forward semantics
//! are intentionally boring: calling a rematerialized body is the same as calling the body
//! directly. The interesting behavior shows up later, when reverse-mode differentiation decides
//! whether to save intermediates or recompute them.

use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use crate::{
    parameters::{Parameter, Parameterized, ParameterizedFamily, Placeholder},
    tracing_v2::{
        Atom, AtomId, Instruction, LinearPrimitiveOperation, LinearTerm, PrimitiveOperation, Program, Traceable,
        Tracer, TracingError, Value, ZeroLike,
        engine::Engine,
        linear::{
            linearize_program, replay_program_linearized_jit, trace_flat_program_from_input_types,
            transpose_linear_program_with_output_examples,
        },
    },
    types::{ArrayType, Type, Typed},
};

use super::{CoreLinearProgramOperation, DifferentiableOperation, InterpretableOperation, LinearOperation, Operation};

/// Hidden staging trait for the `rematerialize` higher-order primitive.
#[doc(hidden)]
pub trait RematerializeTracingOperation<T: Type + Display, V: Traceable<T>, L: Clone>: Clone {
    /// Constructs the carrier-specific representation of the `rematerialize` higher-order primitive
    /// with a captured traced body.
    fn rematerialize_op(op: RematerializeOperation<T, V, Self, L>) -> Self;
}

/// Hidden staging trait for the `rematerialize` higher-order primitive in linear programs.
#[doc(hidden)]
pub trait LinearRematerializeCarrierOperation<T: Type + Display, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the linear `rematerialize` higher-order
    /// primitive with a captured linear traced body.
    fn linear_rematerialize_op(op: LinearRematerializeOperation<T, V, Self>) -> Self;
}

/// Erased traced body for a rematerialization boundary.
///
/// Like [`crate::tracing_v2::operations::FlatTracedVMap`], this stores a flattened traced body that
/// higher-order op nodes can carry around independently of the caller's original parameter shapes.
#[derive(Clone)]
pub struct FlatTracedRematerialize<T: Type, V: Traceable<T>, O = PrimitiveOperation<ArrayType, V>> {
    /// Canonical input types of the body.
    input_types: Vec<T>,

    /// Canonical output types of the body.
    output_types: Vec<T>,

    /// Atom table of the body sub-program.
    atoms: Vec<Atom<T, V>>,

    /// Input atom ids of the body sub-program.
    input_ids: Vec<AtomId>,

    /// Output atom ids of the body sub-program.
    output_ids: Vec<AtomId>,

    /// Instructions of the body sub-program.
    instructions: Vec<Instruction<O>>,
}

impl<T: Type, V: Traceable<T>, O: Clone> FlatTracedRematerialize<T, V, O> {
    /// Builds one erased traced rematerialize body from explicit staged parts.
    #[inline]
    pub fn from_parts(input_types: Vec<T>, output_types: Vec<T>, program: Program<T, V, O, Vec<V>, Vec<V>>) -> Self
    where
        O: Operation<T>,
    {
        let Program { atoms, input_ids, output_ids, instructions, .. } = program;
        Self { input_types, output_types, atoms, input_ids, output_ids, instructions }
    }

    /// Returns the canonical input types of the body.
    #[inline]
    pub fn input_types(&self) -> &[T] {
        self.input_types.as_slice()
    }

    /// Returns the canonical output types of the body.
    #[inline]
    pub fn output_types(&self) -> &[T] {
        self.output_types.as_slice()
    }

    /// Returns a cloned body sub-program.
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
}

/// Higher-order operation that marks its body for rematerialization during linearization.
///
/// During forward execution the body is evaluated normally. When linearized, the body's pushforward
/// is computed and staged so that the tangent program recomputes forward intermediates from the
/// inputs rather than storing them as constants. This makes [`RematerializeOperation`] the staged IR hook
/// that powers the user-facing rematerialization policies in [`crate::tracing_v2::linear`].
#[derive(Clone)]
pub struct RematerializeOperation<
    T: Type + Display,
    V: Traceable<T> + Parameter,
    O: Clone = PrimitiveOperation<ArrayType, V>,
    L: Clone = LinearPrimitiveOperation<ArrayType, V>,
> {
    /// The forward body sub-program.
    body: FlatTracedRematerialize<T, V, O>,

    /// Phantom marker tying the op to the linear carrier used when the body is linearized.
    marker: PhantomData<fn() -> L>,
}

impl<T: Type + Display, V: Traceable<T>, O: Clone, L: Clone> RematerializeOperation<T, V, O, L> {
    /// Builds one ordinary (non-linear) rematerialize op wrapping the given body.
    #[inline]
    pub fn new(body: FlatTracedRematerialize<T, V, O>) -> Self {
        Self { body, marker: PhantomData }
    }

    /// Returns the forward body.
    #[inline]
    pub fn body(&self) -> &FlatTracedRematerialize<T, V, O> {
        &self.body
    }
}

impl<T: Type + Display, V: Traceable<T>, O: Clone, L: Clone> Debug for RematerializeOperation<T, V, O, L> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Rematerialize")
    }
}

impl<T: Type + Display, V: Traceable<T>, O: Clone, L: Clone> Display for RematerializeOperation<T, V, O, L> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "rematerialize")
    }
}

impl<V: Traceable<ArrayType>, O: Clone + Operation<ArrayType>, L: Clone> Operation
    for RematerializeOperation<ArrayType, V, O, L>
{
    fn name(&self) -> &'static str {
        "rematerialize"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TracingError> {
        if input_types.len() != self.body.input_types.len() {
            return Err(TracingError::InvalidInputCount {
                expected: self.body.input_types.len(),
                got: input_types.len(),
            });
        }
        if input_types != self.body.input_types.as_slice() {
            return Err(TracingError::IncompatibleAbstractValues { op: "rematerialize" });
        }
        Ok(self.body.output_types.clone())
    }
}

impl<V: Traceable<ArrayType>, O: Clone + Operation<ArrayType>, L: Clone> InterpretableOperation<ArrayType, V>
    for RematerializeOperation<ArrayType, V, O, L>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    O: InterpretableOperation<ArrayType, V>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        let abstract_inputs = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let _ = self.infer_output_types(abstract_inputs.as_slice())?;
        self.body.program().interpret(inputs.to_vec())
    }
}

impl<'engine, E, V: Value<ArrayType> + ZeroLike, O: Clone + 'static, L: Clone + Operation<ArrayType> + 'static>
    InterpretableOperation<ArrayType, crate::tracing_v2::linear::Linearized<Tracer<'engine, E>>>
    for RematerializeOperation<ArrayType, V, O, L>
where
    E: Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized + 'static,
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    O: Operation<ArrayType>,
    O: InterpretableOperation<ArrayType, V>,
    O: InterpretableOperation<ArrayType, crate::tracing_v2::linear::Linearized<Tracer<'engine, E>>>,
    O: RematerializeTracingOperation<ArrayType, V, L>,
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
            O::rematerialize_op(self.clone()),
        )?;
        let body_program = self.body().program();
        let tangent_outputs = replay_program_linearized_jit::<_, _, _, O, L, E>(&body_program, inputs.to_vec())?;
        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs.into_iter().map(|output| output.tangent))
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
    > for RematerializeOperation<ArrayType, V, O>
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
    O: for<'engine> InterpretableOperation<
            ArrayType,
            crate::tracing_v2::linear::Linearized<
                Tracer<
                    'engine,
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
    for<'engine> LinearPrimitiveOperation<
        ArrayType,
        Tracer<
            'engine,
            dyn Engine<
                    Type = ArrayType,
                    Value = V,
                    TracingOperation = O,
                    LinearOperation = LinearPrimitiveOperation<ArrayType, V>,
                >,
        >,
    >:CoreLinearProgramOperation<
        Tracer<
            'engine,
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
        let tangent_builder = tangent_inputs.first().ok_or(TracingError::EmptyParameterizedValue)?.builder.clone();
        let tangent_outputs = LinearTerm::apply_staged_op(
            tangent_builder,
            tangent_inputs.as_slice(),
            LinearPrimitiveOperation::Rematerialize(Box::new(make_linear_rematerialize(
                engine,
                &self.body,
                primal_inputs,
            )?)),
            self.body.output_types.len(),
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
> InterpretableOperation<ArrayType, Tracer<'engine, E>> for RematerializeOperation<ArrayType, V, O, L>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    O: Operation<ArrayType> + InterpretableOperation<ArrayType, V> + RematerializeTracingOperation<ArrayType, V, L>,
{
    fn interpret(&self, inputs: &[Tracer<'engine, E>]) -> Result<Vec<Tracer<'engine, E>>, TracingError> {
        let exemplar_input = inputs.first().ok_or(TracingError::EmptyParameterizedValue)?.clone();
        Tracer::apply_staged_op(
            exemplar_input.engine,
            exemplar_input.builder.clone(),
            inputs,
            O::rematerialize_op(self.clone()),
        )
    }
}

/// Linear-only rematerialization boundary that always carries both the linear body and its transpose body.
#[derive(Clone)]
pub struct LinearRematerializeOperation<
    T: Type + Display,
    V: Traceable<T> + Parameter,
    O: Clone = LinearPrimitiveOperation<ArrayType, V>,
> {
    /// The forward linear body sub-program.
    body: FlatTracedRematerialize<T, V, O>,

    /// The transpose linear body.
    transpose_body: FlatTracedRematerialize<T, V, O>,
}

impl<T: Type + Display, V: Traceable<T>, O: Clone> LinearRematerializeOperation<T, V, O> {
    /// Builds one linear rematerialize op with an explicit transpose body.
    #[inline]
    pub fn new(body: FlatTracedRematerialize<T, V, O>, transpose_body: FlatTracedRematerialize<T, V, O>) -> Self {
        Self { body, transpose_body }
    }

    /// Returns the forward body.
    #[inline]
    pub fn body(&self) -> &FlatTracedRematerialize<T, V, O> {
        &self.body
    }

    fn transpose_op(&self) -> Self {
        Self::new(self.transpose_body.clone(), self.body.clone())
    }
}

impl<T: Type + Display, V: Traceable<T>, O: Clone> Debug for LinearRematerializeOperation<T, V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "LinearRematerialize")
    }
}

impl<T: Type + Display, V: Traceable<T>, O: Clone> Display for LinearRematerializeOperation<T, V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "rematerialize")
    }
}

impl<V: Traceable<ArrayType>, O: Clone + Operation<ArrayType>> Operation
    for LinearRematerializeOperation<ArrayType, V, O>
{
    fn name(&self) -> &'static str {
        "rematerialize"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TracingError> {
        if input_types.len() != self.body.input_types.len() {
            return Err(TracingError::InvalidInputCount {
                expected: self.body.input_types.len(),
                got: input_types.len(),
            });
        }
        if input_types != self.body.input_types.as_slice() {
            return Err(TracingError::IncompatibleAbstractValues { op: "rematerialize" });
        }
        Ok(self.body.output_types.clone())
    }
}

impl<V: Traceable<ArrayType>, O: Clone + Operation<ArrayType>> InterpretableOperation<ArrayType, V>
    for LinearRematerializeOperation<ArrayType, V, O>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    O: InterpretableOperation<ArrayType, V>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        let abstract_inputs = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let _ = self.infer_output_types(abstract_inputs.as_slice())?;
        self.body.program().interpret(inputs.to_vec())
    }
}

impl<V: Traceable<ArrayType>> LinearOperation<ArrayType, V> for LinearRematerializeOperation<ArrayType, V> {
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, V>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, V>>>, TracingError> {
        let transpose = self.transpose_op();
        let exemplar_output_cotangent = output_cotangents.first().ok_or(TracingError::EmptyParameterizedValue)?.clone();
        Ok(LinearTerm::apply_staged_op(
            exemplar_output_cotangent.builder.clone(),
            output_cotangents,
            LinearPrimitiveOperation::Rematerialize(Box::new(transpose)),
            self.body.input_types().len(),
        )?
        .into_iter()
        .map(Some)
        .collect::<Vec<_>>())
    }
}

/// Builds a linearized rematerialize op from its primal body by computing the pushforward and
/// pullback programs at the provided primal inputs.
#[allow(private_bounds)]
pub(crate) fn make_linear_rematerialize<V, O>(
    engine: &dyn Engine<
        Type = ArrayType,
        Value = V,
        TracingOperation = O,
        LinearOperation = LinearPrimitiveOperation<ArrayType, V>,
    >,
    body: &FlatTracedRematerialize<ArrayType, V, O>,
    input_primals: Vec<V>,
) -> Result<LinearRematerializeOperation<ArrayType, V>, TracingError>
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
    O: for<'engine> InterpretableOperation<
            ArrayType,
            crate::tracing_v2::linear::Linearized<
                Tracer<
                    'engine,
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
    for<'engine> LinearPrimitiveOperation<
        ArrayType,
        Tracer<
            'engine,
            dyn Engine<
                    Type = ArrayType,
                    Value = V,
                    TracingOperation = O,
                    LinearOperation = LinearPrimitiveOperation<ArrayType, V>,
                >,
        >,
    >:CoreLinearProgramOperation<
        Tracer<
            'engine,
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
    let pullback = transpose_linear_program_with_output_examples(&pushforward, output_primals.as_slice())?;
    Ok(LinearRematerializeOperation::new(
        FlatTracedRematerialize::from_parts(
            body.input_types.clone(),
            body.output_types.clone(),
            pushforward.program().clone(),
        ),
        FlatTracedRematerialize::from_parts(
            body.output_types.clone(),
            body.input_types.clone(),
            pullback.program().clone(),
        ),
    ))
}

// ---------------------------------------------------------------------------
// Dispatch trait and public `rematerialize` entry point
// ---------------------------------------------------------------------------

/// Dispatch trait used by [`rematerialize`] to handle both concrete values and already traced values.
#[doc(hidden)]
pub(crate) trait RematerializeInvocationLeaf<
    Input: Parameterized<Self, ParameterStructure: Clone>,
    Output: Parameterized<Self, ParameterStructure: Clone>,
>: Parameter + Sized
{
    /// Invokes [`rematerialize`] for one concrete leaf regime.
    fn invoke<F>(function: F, input: Input) -> Result<Output, TracingError>
    where
        F: FnOnce(Input) -> Output;
}

/// Concrete-value dispatch for [`rematerialize`]: the rematerialization boundary is a no-op during
/// eager execution and simply applies the body function directly.
impl<
    V: Value<ArrayType>,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
> RematerializeInvocationLeaf<Input, Output> for V
{
    fn invoke<F>(function: F, input: Input) -> Result<Output, TracingError>
    where
        F: FnOnce(Input) -> Output,
    {
        Ok(function(input))
    }
}

/// Already-traced dispatch for [`rematerialize`]: traces the body function into a sub-program and
/// stages a [`RematerializeOperation`] in the enclosing [`Tracer`] scope. The sub-program is traced
/// once over exemplar values and captured as a [`Program`] that lowering can later handle.
impl<
    'engine,
    E,
    V: Traceable<ArrayType>,
    Input: Parameterized<Tracer<'engine, E>, ParameterStructure: Clone, To<Tracer<'engine, E>> = Input>,
    Output: Parameterized<Tracer<'engine, E>, ParameterStructure: Clone, To<Tracer<'engine, E>> = Output>,
    O: Clone + Operation<ArrayType> + 'static,
    L: Clone + Operation<ArrayType> + 'static,
> RematerializeInvocationLeaf<Input, Output> for Tracer<'engine, E>
where
    E: Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized + 'static,
    Input::Family: ParameterizedFamily<V> + ParameterizedFamily<ArrayType>,
    Output::Family: ParameterizedFamily<V> + ParameterizedFamily<ArrayType>,
    Input::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'engine, E>> = Input, To<V> = Input::To<V>>,
    Output::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'engine, E>> = Output, To<V> = Output::To<V>>,
    O: InterpretableOperation<ArrayType, V> + RematerializeTracingOperation<ArrayType, V, L>,
{
    fn invoke<F>(function: F, input: Input) -> Result<Output, TracingError>
    where
        F: FnOnce(Input) -> Output,
    {
        let input_structure = input.parameter_structure();
        let traced_inputs = input.into_parameters().collect::<Vec<_>>();
        let input_leaf_count = traced_inputs.len();
        let exemplar_input_types = Input::To::<ArrayType>::from_parameters(
            input_structure.clone(),
            traced_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
        )?;
        let exemplar_traced_input = traced_inputs.first().ok_or(TracingError::EmptyParameterizedValue)?.clone();
        let (exemplar_output_types, body_program) =
            trace_flat_program_from_input_types::<Input::To<ArrayType>, Output::To<ArrayType>, V, O, L, E, _>(
                exemplar_traced_input.engine,
                |staged_input| Ok(function(staged_input)),
                exemplar_input_types,
            )?;

        let output_structure = exemplar_output_types.parameter_structure();
        let output_leaf_count = output_structure.parameter_count();
        let input_types = body_program
            .input_ids
            .iter()
            .map(|id| body_program.atoms[id.index].r#type().into_owned())
            .collect::<Vec<_>>();
        let output_types = exemplar_output_types.parameters().cloned().collect::<Vec<_>>();
        let body = FlatTracedRematerialize::from_parts(
            input_types,
            output_types,
            Program {
                atoms: body_program.atoms.clone(),
                input_ids: body_program.input_ids.clone(),
                output_ids: body_program.output_ids.clone(),
                instructions: body_program.instructions.clone(),
                input_structure: vec![Placeholder; input_leaf_count],
                output_structure: vec![Placeholder; output_leaf_count],
                marker: std::marker::PhantomData,
            },
        );

        let staged_outputs = Tracer::apply_staged_op(
            exemplar_traced_input.engine,
            exemplar_traced_input.builder.clone(),
            traced_inputs.as_slice(),
            O::rematerialize_op(RematerializeOperation::new(body)),
        )?;
        Output::from_parameters(output_structure, staged_outputs).map_err(TracingError::from)
    }
}

/// Marks `function(input)` as a rematerialization boundary.
///
/// During forward execution this is equivalent to calling `function(input)` directly. During
/// reverse-mode differentiation the forward pass of `function` is recomputed from the inputs
/// rather than having its intermediate values saved as constants, trading compute for memory.
///
/// # Example
///
/// ```ignore
/// use ryft_core::tracing_v2::{compile_grad, rematerialize};
/// use ryft_core::tracing_v2::engine::ArrayScalarEngine;
///
/// // Without rematerialize, compile_grad saves all forward intermediates.
/// // With rematerialize, the body is recomputed during the backward pass.
/// let engine = ArrayScalarEngine::<f64>::new();
/// let (_, grad_fn) = compile_grad(&engine, |x: f64| rematerialize(|y| y.sin(), x).unwrap(), 1.0)?;
/// ```
#[allow(private_bounds)]
pub fn rematerialize<F, Input, Output, V>(function: F, input: Input) -> Result<Output, TracingError>
where
    V: RematerializeInvocationLeaf<Input, Output>,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    F: FnOnce(Input) -> Output,
{
    V::invoke(function, input)
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::tracing_v2::{
        Program, Sin,
        engine::ArrayScalarEngine,
        interpret_and_trace,
        linear::{compile_grad, grad, value_and_grad},
    };

    use super::*;

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    #[test]
    fn test_rematerialize_concrete_is_identity() {
        // rematerialize with concrete values should just call the function.
        let result: f64 = rematerialize(|x: f64| x.sin(), 2.0f64).unwrap();
        approx_eq(result, 2.0f64.sin());
    }

    #[test]
    fn test_rematerialize_jit_produces_traced_op() {
        // When used inside jit, rematerialize should produce a "rematerialize" op in the program.
        let engine = ArrayScalarEngine::<f64>::new();
        let (output, program): (f64, Program<ArrayType, f64, PrimitiveOperation<ArrayType, f64>, f64, f64>) =
            interpret_and_trace(&engine, |x| Ok(rematerialize(|y| y.sin(), x).unwrap()), 2.0f64).unwrap();

        approx_eq(output, 2.0f64.sin());
        let ir = program.to_string();
        assert!(ir.contains("rematerialize"), "jit program should contain the rematerialize op: {ir}");
    }

    #[test]
    fn test_rematerialize_jit_program_rendering() {
        // Check the exact rendering of the jit-traced program containing a rematerialize op.
        let engine = ArrayScalarEngine::<f64>::new();
        let (_, program): (f64, Program<ArrayType, f64, PrimitiveOperation<ArrayType, f64>, f64, f64>) =
            interpret_and_trace(&engine, |x| Ok(rematerialize(|y| y.sin(), x).unwrap()), 2.0f64).unwrap();

        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = rematerialize %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_rematerialize_grad_computes_correct_gradient() {
        // grad of rematerialize(sin, x) should be cos(x).
        let engine = ArrayScalarEngine::<f64>::new();
        let gradient: f64 = grad(&engine, |x| rematerialize(|y| y.sin(), x).unwrap(), 2.0f64).unwrap();

        approx_eq(gradient, 2.0f64.cos());
    }

    #[test]
    fn test_rematerialize_value_and_grad_returns_both() {
        // value_and_grad of rematerialize(sin, x) should give (sin(x), cos(x)).
        let engine = ArrayScalarEngine::<f64>::new();
        let (value, gradient): (f64, f64) =
            value_and_grad(&engine, |x| rematerialize(|y| y.sin(), x).unwrap(), 2.0f64).unwrap();

        approx_eq(value, 2.0f64.sin());
        approx_eq(gradient, 2.0f64.cos());
    }

    #[test]
    fn test_rematerialize_compile_grad_produces_reusable_gradient() {
        // compile_grad with rematerialize should produce a symbolic gradient program.
        let engine = ArrayScalarEngine::<f64>::new();
        let compiled = compile_grad(&engine, |x| rematerialize(|y| y.sin(), x).unwrap(), 2.0f64).unwrap();

        // Verify at the original primal point: d/dx sin(x) = cos(x).
        let grad_at_2 = compiled.interpret(2.0f64).unwrap();
        approx_eq(grad_at_2, 2.0f64.cos());

        // Verify at a different primal point to confirm the gradient is symbolic.
        let grad_at_half = compiled.interpret(0.5f64).unwrap();
        approx_eq(grad_at_half, 0.5f64.cos());

        let grad_at_pi = compiled.interpret(std::f64::consts::PI).unwrap();
        approx_eq(grad_at_pi, std::f64::consts::PI.cos());
    }

    #[test]
    fn test_rematerialize_grad_of_quadratic_plus_sin() {
        // grad of rematerialize(x^2 + sin(x), x) should be 2x + cos(x).
        let engine = ArrayScalarEngine::<f64>::new();
        let gradient: f64 =
            grad(&engine, |x| rematerialize(|y| y.clone() * y.clone() + y.sin(), x).unwrap(), 2.0f64).unwrap();

        approx_eq(gradient, 2.0 * 2.0 + 2.0f64.cos());
    }

    #[test]
    fn test_rematerialize_compile_grad_quadratic_plus_sin() {
        // compile_grad with rematerialize wrapping a multi-op body.
        let engine = ArrayScalarEngine::<f64>::new();
        let compiled =
            compile_grad(&engine, |x| rematerialize(|y| y.clone() * y.clone() + y.sin(), x).unwrap(), 2.0f64).unwrap();

        // d/dx(x^2 + sin(x)) = 2x + cos(x)
        let grad_at_2 = compiled.interpret(2.0f64).unwrap();
        approx_eq(grad_at_2, 2.0 * 2.0 + 2.0f64.cos());

        let grad_at_half = compiled.interpret(0.5f64).unwrap();
        approx_eq(grad_at_half, 2.0 * 0.5 + 0.5f64.cos());
    }

    #[test]
    fn test_rematerialize_does_not_affect_forward_result() {
        // The forward result with rematerialize should match the result without it.
        let without: f64 = {
            let engine = ArrayScalarEngine::<f64>::new();
            let (output, _): (f64, Program<ArrayType, f64, PrimitiveOperation<ArrayType, f64>, f64, f64>) =
                interpret_and_trace(&engine, |x| Ok(x.clone() * x.clone() + x.sin()), 3.0f64).unwrap();
            output
        };
        let with: f64 = {
            let engine = ArrayScalarEngine::<f64>::new();
            let (output, _): (f64, Program<ArrayType, f64, PrimitiveOperation<ArrayType, f64>, f64, f64>) =
                interpret_and_trace(
                    &engine,
                    |x| Ok(rematerialize(|y| y.clone() * y.clone() + y.sin(), x).unwrap()),
                    3.0f64,
                )
                .unwrap();
            output
        };

        approx_eq(without, with);
    }
}
