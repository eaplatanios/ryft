use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::parameters::{Parameter, Parameterized, ParameterizedFamily, Placeholder};
use crate::tracing::engines::{Tracer, TracingEngine};
use crate::tracing::transposition::LinearOperation;
use crate::tracing::{Instruction, Program, ProgramBuilder, Traceable, TracingError, Value};
use crate::tracing_v2::operations::constants::{SupportsZero, SupportsZeroLike, Zero, ZeroLike};
use crate::tracing_v2::{
    ArrayOperation, Differentiable, DifferentiableEngine, DifferentiableOperation, DifferentiableTracingEngine,
    DifferentiationError, LinearArrayOperation,
};
use crate::types::{ArrayType, Type, TypeError, Typed};

use super::SupportsAdd;

/// Hidden carrier capability for staging the `rematerialize` higher-order primitive.
#[doc(hidden)]
pub trait SupportsRematerialize<T: Type + PartialEq, V: Traceable<T>, L: Clone>: Clone + Operation<T> {
    /// Constructs the carrier-specific representation of the `rematerialize` higher-order primitive
    /// with a captured traced body.
    fn rematerialize_operation(op: RematerializeOperation<T, V, Self, L>) -> Self;
}

/// Hidden carrier capability for staging the `rematerialize` higher-order primitive in linear programs.
#[doc(hidden)]
pub trait SupportsLinearRematerialize<T: Type + PartialEq, V: Traceable<T>>: Clone + Operation<T> {
    /// Constructs the carrier-specific representation of the linear `rematerialize` higher-order
    /// primitive with a captured linear traced body.
    fn rematerialize_operation(op: LinearRematerializeOperation<T, V, Self>) -> Self;
}

/// Erased traced body for a rematerialization boundary.
///
/// This stores a flattened traced body that higher-order op nodes can carry around independently of
/// the caller's original parameter shapes.
#[derive(Clone)]
pub struct FlatTracedRematerialize<T: Type + PartialEq, V: Traceable<T>, O: Clone = ArrayOperation<V, T>> {
    /// Canonical input types of the body.
    input_types: Vec<T>,

    /// Canonical output types of the body.
    output_types: Vec<T>,

    /// Flat body sub-program executed by this rematerialization boundary.
    program: Program<T, V, O, Vec<V>, Vec<V>>,
}

impl<T: Type + PartialEq, V: Traceable<T>, O: Clone> FlatTracedRematerialize<T, V, O> {
    /// Builds one erased traced rematerialize body from explicit staged parts.
    #[inline]
    pub fn from_parts(input_types: Vec<T>, output_types: Vec<T>, program: Program<T, V, O, Vec<V>, Vec<V>>) -> Self {
        Self { input_types, output_types, program }
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

    /// Returns the flat body sub-program.
    #[inline]
    pub fn program(&self) -> &Program<T, V, O, Vec<V>, Vec<V>> {
        &self.program
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
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
    O: Clone = ArrayOperation<V, T>,
    L: Clone = LinearArrayOperation<V, T>,
> {
    /// The forward body sub-program.
    body: FlatTracedRematerialize<T, V, O>,

    /// Phantom marker tying the op to the linear carrier used when the body is linearized.
    marker: PhantomData<fn() -> L>,
}

impl<T: Type + PartialEq, V: Traceable<T>, O: Clone, L: Clone> RematerializeOperation<T, V, O, L> {
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

impl<T: Type + PartialEq, V: Traceable<T>, O: Clone, L: Clone> Debug for RematerializeOperation<T, V, O, L> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Rematerialize")
    }
}

impl<T: Type + PartialEq, V: Traceable<T>, O: Clone, L: Clone> Display for RematerializeOperation<T, V, O, L> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "rematerialize")
    }
}

impl<T: Type + PartialEq, V: Traceable<T>, O: Clone + Operation<T>, L: Clone> Operation<T>
    for RematerializeOperation<T, V, O, L>
{
    fn name(&self) -> &'static str {
        "rematerialize"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        if input_types.len() != self.body.input_types.len() {
            return Err(TypeError {
                message: format!(
                    "rematerialize expected {} input types but got {}",
                    self.body.input_types.len(),
                    input_types.len()
                ),
            });
        }
        if input_types != self.body.input_types.as_slice() {
            return Err(TypeError {
                message: "rematerialize input types do not match the captured body signature".to_string(),
            });
        }
        Ok(self.body.output_types.clone())
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.program("body", self.body().program()))
    }
}

impl<T: Type + PartialEq, V: Traceable<T>, O: Clone + Operation<T>, L: Clone> InterpretableOperation<T, V>
    for RematerializeOperation<T, V, O, L>
where
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
    O: InterpretableOperation<T, V>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        let abstract_inputs = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let _ = self.infer_output_types(abstract_inputs.as_slice())?;
        self.body.program().interpret(inputs.to_vec())
    }
}

/// JVP rule for `RematerializeOperation` under
/// [`TracingContext`](crate::tracing::engines::TracingContext).
///
/// Stages the primal effect by tracing the rematerialize op in the outer trace via
/// [`TracingContext::trace`](crate::tracing::engines::TracingContext::trace), then recursively linearizes
/// the body through
/// [`TracingContext::linearize`](crate::tracing::engines::TracingContext::linearize) to obtain a pushforward
/// over `Tracer` values, and finally wraps
/// that pushforward (paired with its transpose) in a
/// [`LinearArrayOperation::Rematerialize`] variant that the active linear builder can stage
/// directly.
impl<'engine, V, EInner> DifferentiableOperation<crate::tracing::engines::TracingContext<'engine, EInner>>
    for RematerializeOperation<ArrayType, V, EInner::Operation, LinearArrayOperation<V>>
where
    V: Value<ArrayType>
        + ZeroLike
        + crate::tracing_v2::operations::constants::Zero<ArrayType>
        + Differentiable<ArrayType, Tangent = V>
        + 'static,
    EInner: DifferentiableTracingEngine<Type = ArrayType, Value = V> + ?Sized + 'static,
    EInner::Operation: InterpretableOperation<ArrayType, V>
        + SupportsAdd<ArrayType, V>
        + SupportsRematerialize<ArrayType, V, LinearArrayOperation<V>>,
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
    LinearArrayOperation<V>:
        Clone + InterpretableOperation<ArrayType, V> + LinearOperation<ArrayType, V, LinearArrayOperation<V>>,
    EInner::Operation:
        crate::tracing_v2::linear::TracedLinearizableOperation<'engine, EInner> + SupportsZeroLike<ArrayType, V>,
    EInner::LinearOperation<'engine>: Clone
        + InterpretableOperation<ArrayType, Tracer<'engine, EInner>>
        + LinearOperation<ArrayType, Tracer<'engine, EInner>, EInner::LinearOperation<'engine>>
        + SupportsZero<ArrayType, Tracer<'engine, EInner>>
        + SupportsLinearRematerialize<ArrayType, Tracer<'engine, EInner>>,
{
    fn jvp(
        &self,
        context: &mut crate::tracing_v2::JvpContext<'_, crate::tracing::engines::TracingContext<'engine, EInner>>,
        inputs: &[crate::tracing_v2::JvpTracer<Tracer<'engine, EInner>, crate::tracing::AtomId>],
    ) -> Result<Vec<crate::tracing_v2::JvpTracer<Tracer<'engine, EInner>, crate::tracing::AtomId>>, TracingError> {
        let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let tangent_inputs = inputs.iter().map(|input| input.tangent).collect::<Vec<_>>();

        let primal_outputs = if primal_inputs.is_empty() {
            if !self.body().output_types().is_empty() {
                return Err(DifferentiationError::MissingTracedRematerializeInputLeaves.into());
            }
            Vec::new()
        } else {
            let exemplar = primal_inputs[0].clone();
            let primal_input_refs = primal_inputs.iter().collect::<Vec<_>>();
            exemplar
                .context
                .trace(EInner::Operation::rematerialize_operation(self.clone()), primal_input_refs.as_slice())?
        };

        if tangent_inputs.is_empty() && !self.body.output_types.is_empty() {
            return Err(DifferentiationError::MissingLinearRematerializeReplayTangentLeaves.into());
        }

        let (_, pushforward) = context.engine.linearize(self.body().program(), primal_inputs)?;
        let pullback = context.engine.transpose(&pushforward)?;

        let body_input_types = self.body().input_types().to_vec();
        let body_output_types = self.body().output_types().to_vec();
        let linear_remat = LinearRematerializeOperation::<
            ArrayType,
            Tracer<'engine, EInner>,
            <EInner as crate::tracing_v2::DifferentiableTracingEngine>::LinearOperation<'engine>,
        >::new(
            FlatTracedRematerialize::<
                ArrayType,
                Tracer<'engine, EInner>,
                <EInner as crate::tracing_v2::DifferentiableTracingEngine>::LinearOperation<'engine>,
            >::from_parts(body_input_types.clone(), body_output_types.clone(), pushforward),
            FlatTracedRematerialize::<
                ArrayType,
                Tracer<'engine, EInner>,
                <EInner as crate::tracing_v2::DifferentiableTracingEngine>::LinearOperation<'engine>,
            >::from_parts(body_output_types, body_input_types, pullback),
        );

        let tangent_outputs = context.apply_operation(
            tangent_inputs.as_slice(),
            <EInner::LinearOperation<'engine> as SupportsLinearRematerialize<
                ArrayType,
                Tracer<'engine, EInner>,
            >>::rematerialize_operation(linear_remat),
            self.body().output_types().len(),
        )?;

        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| crate::tracing_v2::JvpTracer { primal, tangent })
            .collect::<Vec<_>>())
    }
}

impl<
    V: Value<ArrayType>
        + ZeroLike
        + crate::tracing_v2::operations::constants::Zero<ArrayType>
        + Differentiable<ArrayType, Tangent = V>
        + 'static,
    E: DifferentiableEngine<Type = ArrayType, Value = V, LinearOperation = LinearArrayOperation<V>> + ?Sized + 'static,
    O: Clone + Operation<ArrayType>,
> DifferentiableOperation<E> for RematerializeOperation<ArrayType, V, O, E::LinearOperation>
where
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
    O: DifferentiableOperation<E>,
    O: InterpretableOperation<ArrayType, V>,
    O: SupportsRematerialize<ArrayType, V, E::LinearOperation> + 'static,
    LinearArrayOperation<V>:
        Clone + InterpretableOperation<ArrayType, V> + LinearOperation<ArrayType, V, LinearArrayOperation<V>>,
{
    fn jvp(
        &self,
        context: &mut crate::tracing_v2::JvpContext<'_, E>,
        inputs: &[crate::tracing_v2::JvpTracer<V, crate::tracing::AtomId>],
    ) -> Result<Vec<crate::tracing_v2::JvpTracer<V, crate::tracing::AtomId>>, TracingError> {
        let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let tangent_inputs = inputs.iter().map(|input| input.tangent).collect::<Vec<_>>();
        let primal_outputs = <Self as InterpretableOperation<ArrayType, V>>::interpret(self, primal_inputs.as_slice())?;
        if tangent_inputs.is_empty() && !self.body.output_types.is_empty() {
            return Err(DifferentiationError::MissingLinearRematerializeReplayTangentLeaves.into());
        }
        let tangent_outputs = context.apply_operation(
            tangent_inputs.as_slice(),
            LinearArrayOperation::Rematerialize(Box::new(make_linear_rematerialize(
                context.engine,
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
    V: Value<ArrayType> + Differentiable<ArrayType, Tangent = V>,
    E: DifferentiableEngine<Type = ArrayType, Value = V> + TracingEngine + ?Sized,
> InterpretableOperation<ArrayType, Tracer<'engine, E>>
    for RematerializeOperation<ArrayType, V, E::Operation, E::LinearOperation>
where
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
    E::Operation: InterpretableOperation<ArrayType, V> + SupportsRematerialize<ArrayType, V, E::LinearOperation>,
{
    fn interpret(&self, inputs: &[Tracer<'engine, E>]) -> Result<Vec<Tracer<'engine, E>>, TracingError> {
        if inputs.is_empty() {
            return if self.body.output_types().is_empty() {
                Ok(Vec::new())
            } else {
                Err(DifferentiationError::MissingTracedRematerializeInputLeaves.into())
            };
        }
        let exemplar_input = inputs[0].clone();
        let input_refs = inputs.iter().collect::<Vec<_>>();
        exemplar_input
            .context
            .trace(E::Operation::rematerialize_operation(self.clone()), input_refs.as_slice())
    }
}

/// Linear-only rematerialization boundary that always carries both the linear body and its transpose body.
#[derive(Clone)]
pub struct LinearRematerializeOperation<
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
    O: Clone = LinearArrayOperation<V, T>,
> {
    /// The forward linear body sub-program.
    body: FlatTracedRematerialize<T, V, O>,

    /// The transpose linear body.
    transpose_body: FlatTracedRematerialize<T, V, O>,
}

impl<T: Type + PartialEq, V: Traceable<T>, O: Clone> LinearRematerializeOperation<T, V, O> {
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

impl<T: Type + PartialEq, V: Traceable<T>, O: Clone> Debug for LinearRematerializeOperation<T, V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "LinearRematerialize")
    }
}

impl<T: Type + PartialEq, V: Traceable<T>, O: Clone> Display for LinearRematerializeOperation<T, V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "rematerialize")
    }
}

impl<T: Type + PartialEq, V: Traceable<T>, O: Clone + Operation<T>> Operation<T>
    for LinearRematerializeOperation<T, V, O>
{
    fn name(&self) -> &'static str {
        "rematerialize"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        if input_types.len() != self.body.input_types.len() {
            return Err(TypeError {
                message: format!(
                    "rematerialize expected {} input types but got {}",
                    self.body.input_types.len(),
                    input_types.len()
                ),
            });
        }
        if input_types != self.body.input_types.as_slice() {
            return Err(TypeError {
                message: "rematerialize input types do not match the captured body signature".to_string(),
            });
        }
        Ok(self.body.output_types.clone())
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.program("body", self.body().program())?;
            operation.program("transpose_body", self.transpose_body.program())
        })
    }
}

impl<T: Type + PartialEq, V: Traceable<T>, O: Clone + Operation<T>> InterpretableOperation<T, V>
    for LinearRematerializeOperation<T, V, O>
where
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
    O: InterpretableOperation<T, V>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        let abstract_inputs = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let _ = self.infer_output_types(abstract_inputs.as_slice())?;
        self.body.program().interpret(inputs.to_vec())
    }
}

impl<T, V> LinearOperation<T, V, LinearArrayOperation<V, T>> for LinearRematerializeOperation<T, V>
where
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
    LinearArrayOperation<V, T>: Operation<T>,
{
    fn transpose(
        &self,
        context: &mut crate::tracing::transposition::TranspositionContext<T, V, LinearArrayOperation<V, T>>,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        let transpose = self.transpose_op();
        if output_cotangents.is_empty() {
            return if self.body.input_types().is_empty() {
                Ok(Vec::new())
            } else {
                Err(DifferentiationError::MissingLinearRematerializeTransposeCotangentLeaves.into())
            };
        }
        if output_cotangents.iter().all(Option::is_none) {
            return Ok(vec![None; self.body.input_types().len()]);
        }
        let materialized = output_cotangents
            .iter()
            .zip(transpose.body.input_types().iter())
            .map(|(cotangent, input_type)| materialize_optional_cotangent(context, *cotangent, input_type))
            .collect::<Vec<_>>();
        Ok(context
            .stage(LinearArrayOperation::<V, T>::Rematerialize(Box::new(transpose)), materialized.as_slice())?
            .into_iter()
            .map(Some)
            .collect::<Vec<_>>())
    }
}

/// Returns a concrete cotangent atom for `cotangent`, staging a typed `Zero` op when the cotangent
/// is structurally zero. Linear higher-order rules use this when they must consume all output
/// cotangents jointly (e.g. a nested transpose program that has a fixed input arity).
fn materialize_optional_cotangent<T, V>(
    context: &crate::tracing::transposition::TranspositionContext<T, V, LinearArrayOperation<V, T>>,
    cotangent: Option<crate::tracing::AtomId>,
    input_type: &T,
) -> crate::tracing::AtomId
where
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
    LinearArrayOperation<V, T>: Operation<T>,
{
    if let Some(atom) = cotangent {
        return atom;
    }
    use crate::tracing_v2::operations::SupportsZero;
    let builder = &context.builder;
    let mut builder_borrow = builder.borrow_mut();
    let output = builder_borrow.add_variable(input_type.clone());
    builder_borrow.instructions.push(Instruction {
        operation: <LinearArrayOperation<V, T> as SupportsZero<T, V>>::zero_operation(input_type.clone()),
        inputs: vec![],
        outputs: vec![output],
    });
    output
}

/// Builds a linearized rematerialize op from its primal body by computing the pushforward and
/// pullback programs at the provided primal inputs.
#[allow(private_bounds)]
pub(crate) fn make_linear_rematerialize<V, E, O>(
    engine: &E,
    body: &FlatTracedRematerialize<ArrayType, V, O>,
    input_primals: Vec<V>,
) -> Result<LinearRematerializeOperation<ArrayType, V>, TracingError>
where
    V: Traceable<ArrayType> + ZeroLike + Differentiable<ArrayType, Tangent = V> + Zero<ArrayType> + 'static,
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
    O: Clone + Operation<ArrayType> + InterpretableOperation<ArrayType, V> + DifferentiableOperation<E> + 'static,
    LinearArrayOperation<V>:
        Clone + InterpretableOperation<ArrayType, V> + LinearOperation<ArrayType, V, LinearArrayOperation<V>>,
    E: DifferentiableEngine<Type = ArrayType, Value = V, LinearOperation = LinearArrayOperation<V>> + ?Sized + 'static,
{
    let body_program = body.program();
    let output_primals = body_program.interpret(input_primals.clone())?;
    let pushforward = body_program.linearize(engine, input_primals)?;
    let pullback = pushforward.transpose(output_primals.as_slice())?;
    Ok(LinearRematerializeOperation::new(
        FlatTracedRematerialize::from_parts(body.input_types.clone(), body.output_types.clone(), pushforward),
        FlatTracedRematerialize::from_parts(body.output_types.clone(), body.input_types.clone(), pullback),
    ))
}

// ---------------------------------------------------------------------------
// Dispatch trait and public `rematerialize` entry point
// ---------------------------------------------------------------------------

/// Dispatch trait used by [`rematerialize`] to handle both concrete values and already traced values.
#[doc(hidden)]
pub(crate) trait RematerializeInvocationLeaf<Input: Parameterized<Self>, Output: Parameterized<Self>>:
    Parameter + Sized
{
    /// Invokes [`rematerialize`] for one concrete leaf regime.
    fn invoke<F>(function: F, input: Input) -> Result<Output, TracingError>
    where
        F: FnOnce(Input) -> Output;
}

/// Concrete-value dispatch for [`rematerialize`]: the rematerialization boundary is a no-op during
/// eager execution and simply applies the body function directly.
impl<V: Value<ArrayType>, Input: Parameterized<V>, Output: Parameterized<V>> RematerializeInvocationLeaf<Input, Output>
    for V
{
    fn invoke<F>(function: F, input: Input) -> Result<Output, TracingError>
    where
        F: FnOnce(Input) -> Output,
    {
        Ok(function(input))
    }
}

/// Already-traced dispatch for [`rematerialize`]: traces the body function into a sub-program and
/// stages a [`RematerializeOperation`] in the enclosing [`Tracer`] engine. The sub-program is traced
/// once over exemplar values and captured as a [`Program`] that lowering can later handle.
impl<
    'engine,
    E,
    V: Traceable<ArrayType> + Differentiable<ArrayType, Tangent = V>,
    Input: Parameterized<Tracer<'engine, E>, To<Tracer<'engine, E>> = Input>,
    Output: Parameterized<Tracer<'engine, E>, To<Tracer<'engine, E>> = Output>,
> RematerializeInvocationLeaf<Input, Output> for Tracer<'engine, E>
where
    E: DifferentiableEngine<Type = ArrayType, Value = V> + TracingEngine + ?Sized + 'static,
    Input::Family: ParameterizedFamily<V> + ParameterizedFamily<ArrayType>,
    Output::Family: ParameterizedFamily<V> + ParameterizedFamily<ArrayType>,
    Input::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'engine, E>> = Input, To<V> = Input::To<V>>,
    Output::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'engine, E>> = Output, To<V> = Output::To<V>>,
    E::Operation: InterpretableOperation<ArrayType, V> + SupportsRematerialize<ArrayType, V, E::LinearOperation>,
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
        let Some(exemplar_traced_input) = traced_inputs.first().cloned() else {
            return Err(DifferentiationError::MissingTracedRematerializeInputLeaves.into());
        };
        let (exemplar_output_types, body_program) = exemplar_traced_input
            .context
            .engine
            .trace(|staged_input| Ok(function(staged_input)), exemplar_input_types)?;

        let output_structure = exemplar_output_types.parameter_structure();
        let output_leaf_count = output_structure.parameter_count();
        let input_types = body_program
            .input_ids
            .iter()
            .map(|id| body_program.atoms[id.index].r#type().into_owned())
            .collect::<Vec<_>>();
        let output_types = exemplar_output_types.parameters().cloned().collect::<Vec<_>>();
        let Program { atoms, input_ids, output_ids, instructions, .. } = body_program;
        let mut builder = ProgramBuilder::<ArrayType, V, E::Operation>::new();
        builder.atoms = atoms;
        builder.input_ids = input_ids;
        builder.instructions = instructions;
        let body = FlatTracedRematerialize::from_parts(
            input_types,
            output_types,
            builder
                .build(output_ids, vec![Placeholder; input_leaf_count], vec![Placeholder; output_leaf_count])?
                .simplified()?,
        );

        let traced_input_refs = traced_inputs.iter().collect::<Vec<_>>();
        let staged_outputs = exemplar_traced_input.context.trace(
            E::Operation::rematerialize_operation(RematerializeOperation::new(body)),
            traced_input_refs.as_slice(),
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
///
/// // Without rematerialize, compile_grad saves all forward intermediates.
/// // With rematerialize, the body is recomputed during the backward pass.
/// // Use an engine whose metadata type is ArrayType, such as an array backend.
/// let (_, grad_fn) = compile_grad(&engine, |x| rematerialize(|y| y.sin(), x).unwrap(), input)?;
/// ```
#[allow(private_bounds)]
pub fn rematerialize<F, Input, Output, V>(function: F, input: Input) -> Result<Output, TracingError>
where
    V: RematerializeInvocationLeaf<Input, Output>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
    F: FnOnce(Input) -> Output,
{
    V::invoke(function, input)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::tracing::engines::{Engine, Tracer, TracingEngine};
    use crate::tracing::transposition::TranspositionContext;
    use crate::tracing::{Program, ProgramBuilder};
    use crate::tracing_v2::linear::{compile_grad, grad, value_and_grad};
    use crate::tracing_v2::{DifferentiationError, JvpTracer, LinearArrayOperation, Sin};

    use super::*;

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    fn scalar_type() -> ArrayType {
        ArrayType::scalar(crate::types::DataType::F64)
    }

    fn test_transposition_context() -> TranspositionContext<ArrayType, f64, LinearArrayOperation<f64>> {
        TranspositionContext::new(Rc::new(RefCell::new(ProgramBuilder::new())))
    }

    #[derive(Copy, Clone)]
    struct ArrayScalarEngine;

    impl Engine for ArrayScalarEngine {
        type Type = ArrayType;
        type Value = f64;

        fn zero(&self, _type: &ArrayType) -> Result<f64, TracingError> {
            Ok(0.0)
        }

        fn one(&self, _type: &ArrayType) -> Result<f64, TracingError> {
            Ok(1.0)
        }
    }

    impl TracingEngine for ArrayScalarEngine {
        type Operation = ArrayOperation<f64>;
    }

    impl DifferentiableEngine for ArrayScalarEngine {
        type DifferentiableOperation = ArrayOperation<f64>;
        type LinearOperation = LinearArrayOperation<f64>;
    }

    impl DifferentiableTracingEngine for ArrayScalarEngine {
        type LinearOperation<'engine>
            = LinearArrayOperation<Tracer<'engine, Self>>
        where
            Self: 'engine;
    }

    fn empty_traced_body() -> FlatTracedRematerialize<ArrayType, f64> {
        let mut builder = ProgramBuilder::<ArrayType, f64, ArrayOperation<f64>>::new();
        let output = builder.add_constant(0.0f64);
        let program = builder.build(vec![output], Vec::<Placeholder>::new(), vec![Placeholder]).unwrap();
        FlatTracedRematerialize::from_parts(vec![], vec![scalar_type()], program)
    }

    fn empty_linear_body() -> FlatTracedRematerialize<ArrayType, f64, LinearArrayOperation<f64>> {
        let program = ProgramBuilder::<ArrayType, f64, LinearArrayOperation<f64>>::new()
            .build(vec![], Vec::<Placeholder>::new(), Vec::<Placeholder>::new())
            .unwrap();
        FlatTracedRematerialize::from_parts(vec![scalar_type()], vec![], program)
    }

    #[test]
    fn test_rematerialize_concrete_is_identity() {
        // rematerialize with concrete values should just call the function.
        let result: f64 = rematerialize(|x: f64| x.sin(), 2.0f64).unwrap();
        approx_eq(result, 2.0f64.sin());
    }

    #[test]
    fn test_linear_rematerialize_jvp_requires_tangent_leaves() {
        let engine = ArrayScalarEngine;
        let operation = RematerializeOperation::<ArrayType, f64>::new(empty_traced_body());
        let inputs: Vec<JvpTracer<f64, crate::tracing::AtomId>> = Vec::new();
        let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, f64, LinearArrayOperation<f64>>::new()));
        let mut context = crate::tracing_v2::JvpContext::new(&engine, builder);

        assert!(matches!(
            operation.jvp(&mut context, inputs.as_slice()),
            Err(TracingError::Differentiation(DifferentiationError::MissingLinearRematerializeReplayTangentLeaves))
        ));
    }

    #[test]
    fn test_traced_rematerialize_requires_input_leaves() {
        let operation = RematerializeOperation::<ArrayType, f64>::new(empty_traced_body());
        let inputs: Vec<Tracer<'_, ArrayScalarEngine>> = Vec::new();

        assert!(matches!(
            operation.interpret(inputs.as_slice()),
            Err(TracingError::Differentiation(DifferentiationError::MissingTracedRematerializeInputLeaves))
        ));
    }

    #[test]
    fn test_linear_rematerialize_transpose_requires_output_cotangent_leaves() {
        let operation = LinearRematerializeOperation::<ArrayType, f64>::new(empty_linear_body(), empty_linear_body());
        let output_cotangents: Vec<Option<crate::tracing::AtomId>> = Vec::new();
        let mut context = test_transposition_context();

        assert!(matches!(
            operation.transpose(&mut context, output_cotangents.as_slice()),
            Err(TracingError::Differentiation(
                DifferentiationError::MissingLinearRematerializeTransposeCotangentLeaves
            ))
        ));
    }

    #[test]
    fn test_rematerialize_invocation_requires_traced_input_leaves() {
        let input: Vec<Tracer<'_, ArrayScalarEngine>> = Vec::new();

        let result = <Tracer<'_, ArrayScalarEngine> as RematerializeInvocationLeaf<
            Vec<Tracer<'_, ArrayScalarEngine>>,
            Tracer<'_, ArrayScalarEngine>,
        >>::invoke(|_inputs| panic!("closure should not run without traced inputs"), input);

        assert!(matches!(
            result,
            Err(TracingError::Differentiation(DifferentiationError::MissingTracedRematerializeInputLeaves))
        ));
    }

    #[test]
    fn test_rematerialize_jit_produces_traced_op() {
        // When used inside jit, rematerialize should produce a "rematerialize" op in the program.
        let engine = ArrayScalarEngine;
        let (output, program): (f64, Program<ArrayType, f64, ArrayOperation<f64>, f64, f64>) =
            engine.interpret_and_trace(|x| Ok(rematerialize(|y| y.sin(), x).unwrap()), 2.0f64).unwrap();

        approx_eq(output, 2.0f64.sin());
        let ir = program.to_string();
        assert!(ir.contains("rematerialize"), "jit program should contain the rematerialize op: {ir}");
    }

    #[test]
    fn test_rematerialize_jit_program_rendering() {
        // Check the exact rendering of the jit-traced program containing a rematerialize op.
        let engine = ArrayScalarEngine;
        let (_, program): (f64, Program<ArrayType, f64, ArrayOperation<f64>, f64, f64>) =
            engine.interpret_and_trace(|x| Ok(rematerialize(|y| y.sin(), x).unwrap()), 2.0f64).unwrap();

        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = rematerialize [
                    body={
                        lambda %0:f64[] .
                        let %1:f64[] = sin %0
                        in (%1)
                    },
                ] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_rematerialize_grad_computes_correct_gradient() {
        // grad of rematerialize(sin, x) should be cos(x).
        let engine = ArrayScalarEngine;
        let gradient: f64 = grad(&engine, |x| rematerialize(|y| y.sin(), x).unwrap(), 2.0f64).unwrap();

        approx_eq(gradient, 2.0f64.cos());
    }

    #[test]
    fn test_rematerialize_value_and_grad_returns_both() {
        // value_and_grad of rematerialize(sin, x) should give (sin(x), cos(x)).
        let engine = ArrayScalarEngine;
        let (value, gradient): (f64, f64) =
            value_and_grad(&engine, |x| rematerialize(|y| y.sin(), x).unwrap(), 2.0f64).unwrap();

        approx_eq(value, 2.0f64.sin());
        approx_eq(gradient, 2.0f64.cos());
    }

    #[test]
    fn test_rematerialize_compile_grad_produces_reusable_gradient() {
        // compile_grad with rematerialize should produce a symbolic gradient program.
        let engine = ArrayScalarEngine;
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
        let engine = ArrayScalarEngine;
        let gradient: f64 =
            grad(&engine, |x| rematerialize(|y| y.clone() * y.clone() + y.sin(), x).unwrap(), 2.0f64).unwrap();

        approx_eq(gradient, 2.0 * 2.0 + 2.0f64.cos());
    }

    #[test]
    fn test_rematerialize_compile_grad_quadratic_plus_sin() {
        // compile_grad with rematerialize wrapping a multi-op body.
        let engine = ArrayScalarEngine;
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
            let engine = ArrayScalarEngine;
            let (output, _): (f64, Program<ArrayType, f64, ArrayOperation<f64>, f64, f64>) =
                engine.interpret_and_trace(|x| Ok(x.clone() * x.clone() + x.sin()), 3.0f64).unwrap();
            output
        };
        let with: f64 = {
            let engine = ArrayScalarEngine;
            let (output, _): (f64, Program<ArrayType, f64, ArrayOperation<f64>, f64, f64>) = engine
                .interpret_and_trace(|x| Ok(rematerialize(|y| y.clone() * y.clone() + y.sin(), x).unwrap()), 3.0f64)
                .unwrap();
            output
        };

        approx_eq(without, with);
    }
}
