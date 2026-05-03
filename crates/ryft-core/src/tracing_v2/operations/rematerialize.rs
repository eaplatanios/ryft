use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use crate::macros::check_input_count;
use crate::operations::constants::{SupportsZero, Zero};
use crate::operations::constants::{SupportsZeroLike, ZeroLike};
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::parameters::{Parameter, Parameterized, ParameterizedFamily, Placeholder};
use crate::tracing::engines::{Tracer, TracingEngine};
use crate::tracing::transposition::LinearOperation;
use crate::tracing::{Instruction, Program, ProgramBuilder, Traceable, TracingError, Value};
use crate::tracing_v2::{
    ArrayOperation, Differentiable, DifferentiableEngine, DifferentiableOperation, DifferentiableTracingEngine,
    DifferentiationError, LinearArrayOperation, LinearizableEngine,
};
use crate::types::{ArrayType, Type, TypeError, Typed};

use crate::operations::arithmetic::SupportsAdd;

/// Trait that represents [`Operation`] carrier types that support/include [`RematerializeOperation`]. Backend-owned
/// closed [`Operation`] carrier types (such as [`ArrayOperation`](super::ArrayOperation), for example) implement this
/// trait so that generic transform code can stage [`RematerializeOperation`] without knowing which carrier is in use.
#[doc(hidden)]
pub trait SupportsRematerialize<T: Type + PartialEq, V: Traceable<T>, L>: Sized + Operation<T> {
    /// Constructs the carrier-specific representation of the rematerialization [`Operation`].
    fn rematerialize_operation(op: RematerializeOperation<T, V, Self, L>) -> Self;
}

/// Trait that represents [`Operation`] carrier types that support/include [`LinearRematerializeOperation`].
/// Backend-owned closed [`Operation`] carrier types (such as [`LinearArrayOperation`](super::LinearArrayOperation), for
/// example) implement this trait so that generic transform code can stage [`LinearRematerializeOperation`] without
/// knowing which carrier is in use.
#[doc(hidden)]
pub trait SupportsLinearRematerialize<T: Type + PartialEq, V: Traceable<T>>: Sized + Operation<T> {
    /// Constructs the carrier-specific representation of the linear rematerialization [`Operation`].
    fn rematerialize_operation(op: LinearRematerializeOperation<T, V, Self>) -> Self;
}

/// Erased traced body for a rematerialization boundary.
///
/// This stores a flattened traced body that higher-order op nodes can carry around independently of
/// the caller's original parameter shapes.
#[derive(Clone, Debug)]
pub struct FlatTracedRematerialize<T: Type + PartialEq, V: Traceable<T>, O = ArrayOperation<V, T>> {
    /// Canonical input types of the body.
    pub input_types: Vec<T>,

    /// Canonical output types of the body.
    pub output_types: Vec<T>,

    /// Flat body sub-program executed by this rematerialization boundary.
    pub program: Program<T, V, O, Vec<V>, Vec<V>>,
}

impl<T: Type + PartialEq, V: Traceable<T>, O> FlatTracedRematerialize<T, V, O> {
    /// Builds one erased traced rematerialize body from explicit staged parts.
    #[inline]
    pub fn from_parts(input_types: Vec<T>, output_types: Vec<T>, program: Program<T, V, O, Vec<V>, Vec<V>>) -> Self {
        Self { input_types, output_types, program }
    }
}

/// Higher-order operation that marks its body for rematerialization during linearization.
///
/// During forward execution the body is evaluated normally. When linearized, the body's pushforward
/// is computed and staged so that the tangent program recomputes forward intermediates from the
/// inputs rather than storing them as constants. This makes [`RematerializeOperation`] the staged IR hook
/// that powers the user-facing rematerialization policies in [`crate::tracing_v2::linear`].
#[derive(Clone, Debug)]
pub struct RematerializeOperation<
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
    O = ArrayOperation<V, T>,
    L = LinearArrayOperation<V, T>,
> {
    /// The forward body sub-program.
    pub body: FlatTracedRematerialize<T, V, O>,

    /// Phantom marker tying the op to the linear carrier used when the body is linearized.
    pub marker: PhantomData<fn() -> L>,
}

impl<T: Type + PartialEq, V: Traceable<T>, O, L> RematerializeOperation<T, V, O, L> {
    /// Builds one ordinary (non-linear) rematerialize op wrapping the given body.
    #[inline]
    pub fn new(body: FlatTracedRematerialize<T, V, O>) -> Self {
        Self { body, marker: PhantomData }
    }
}

impl<T: Type + PartialEq, V: Traceable<T>, O, L> Display for RematerializeOperation<T, V, O, L>
where
    Self: Operation<T>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl<T: Type + PartialEq, V: Traceable<T>, O: Clone + Operation<T>, L: Clone + Debug> Operation<T>
    for RematerializeOperation<T, V, O, L>
{
    #[inline]
    fn name(&self) -> &'static str {
        "rematerialize"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        check_input_count!(input_types, self.body.input_types.len(), TypeError);
        if input_types != self.body.input_types.as_slice() {
            return Err(TypeError {
                message: "rematerialize input types do not match the captured body signature".to_string(),
            });
        }
        Ok(self.body.output_types.clone())
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.program("body", &self.body.program))
    }
}

impl<T: Type + PartialEq, V: Traceable<T>, O: Clone + Operation<T>, L: Clone + Debug> InterpretableOperation<T, V>
    for RematerializeOperation<T, V, O, L>
where
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
    O: InterpretableOperation<T, V>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        let abstract_inputs = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let _ = self.infer_output_types(abstract_inputs.as_slice())?;
        self.body.program.interpret(inputs.to_vec())
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
    for RematerializeOperation<ArrayType, V, EInner::OperationCarrier, LinearArrayOperation<V, ArrayType>>
where
    V: Value<ArrayType>
        + ZeroLike
        + crate::operations::constants::Zero<ArrayType>
        + Differentiable<ArrayType, Tangent = V>
        + 'static,
    EInner: DifferentiableTracingEngine<Type = ArrayType, Value = V> + ?Sized + 'static,
    EInner::OperationCarrier: Clone
        + InterpretableOperation<ArrayType, V>
        + SupportsAdd<ArrayType, V>
        + SupportsRematerialize<ArrayType, V, LinearArrayOperation<V, ArrayType>>,
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
    LinearArrayOperation<V, ArrayType>: Clone
        + InterpretableOperation<ArrayType, V>
        + LinearOperation<ArrayType, V, LinearArrayOperation<V, ArrayType>>,
    EInner::OperationCarrier: DifferentiableOperation<crate::tracing::engines::TracingContext<'engine, EInner>>
        + SupportsZeroLike<ArrayType, V>,
    EInner::LinearOperationCarrier<'engine>: Clone
        + InterpretableOperation<ArrayType, Tracer<'engine, EInner>>
        + LinearOperation<ArrayType, Tracer<'engine, EInner>, EInner::LinearOperationCarrier<'engine>>
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
            if !self.body.output_types.as_slice().is_empty() {
                return Err(DifferentiationError::MissingTracedRematerializeInputLeaves.into());
            }
            Vec::new()
        } else {
            let exemplar = primal_inputs[0].clone();
            let primal_input_refs = primal_inputs.iter().collect::<Vec<_>>();
            exemplar
                .context
                .trace(EInner::OperationCarrier::rematerialize_operation(self.clone()), primal_input_refs.as_slice())?
        };

        if tangent_inputs.is_empty() && !self.body.output_types.is_empty() {
            return Err(DifferentiationError::MissingLinearRematerializeReplayTangentLeaves.into());
        }

        let (_, pushforward) = context.engine.linearize(&self.body.program, primal_inputs)?;
        let pullback = context.engine.transpose(&pushforward)?;

        let body_input_types = self.body.input_types.as_slice().to_vec();
        let body_output_types = self.body.output_types.as_slice().to_vec();
        let linear_remat = LinearRematerializeOperation::<
            ArrayType,
            Tracer<'engine, EInner>,
            <EInner as crate::tracing_v2::DifferentiableTracingEngine>::LinearOperationCarrier<'engine>,
        >::new(
            FlatTracedRematerialize::<
                ArrayType,
                Tracer<'engine, EInner>,
                <EInner as crate::tracing_v2::DifferentiableTracingEngine>::LinearOperationCarrier<'engine>,
            >::from_parts(body_input_types.clone(), body_output_types.clone(), pushforward),
            FlatTracedRematerialize::<
                ArrayType,
                Tracer<'engine, EInner>,
                <EInner as crate::tracing_v2::DifferentiableTracingEngine>::LinearOperationCarrier<'engine>,
            >::from_parts(body_output_types, body_input_types, pullback),
        );

        let tangent_outputs = context.stage(
            <EInner::LinearOperationCarrier<'engine> as SupportsLinearRematerialize<
                ArrayType,
                Tracer<'engine, EInner>,
            >>::rematerialize_operation(linear_remat),
            tangent_inputs.as_slice(),
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
        + crate::operations::constants::Zero<ArrayType>
        + Differentiable<ArrayType, Tangent = V>
        + 'static,
    E: LinearizableEngine<Type = ArrayType, Value = V, LinearOperationCarrier = LinearArrayOperation<V, ArrayType>>
        + ?Sized
        + 'static,
    O: Clone + Operation<ArrayType>,
> DifferentiableOperation<E> for RematerializeOperation<ArrayType, V, O, E::LinearOperationCarrier>
where
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
    O: DifferentiableOperation<E>,
    O: InterpretableOperation<ArrayType, V>,
    O: SupportsRematerialize<ArrayType, V, E::LinearOperationCarrier> + 'static,
    LinearArrayOperation<V, ArrayType>: Clone
        + InterpretableOperation<ArrayType, V>
        + LinearOperation<ArrayType, V, LinearArrayOperation<V, ArrayType>>,
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
        let tangent_outputs = context.stage(
            LinearArrayOperation::Rematerialize(Box::new(make_linear_rematerialize(
                context.engine,
                &self.body,
                primal_inputs,
            )?)),
            tangent_inputs.as_slice(),
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
    for RematerializeOperation<ArrayType, V, E::OperationCarrier, E::LinearOperationCarrier>
where
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
    E::OperationCarrier:
        Clone + InterpretableOperation<ArrayType, V> + SupportsRematerialize<ArrayType, V, E::LinearOperationCarrier>,
{
    fn interpret(&self, inputs: &[Tracer<'engine, E>]) -> Result<Vec<Tracer<'engine, E>>, TracingError> {
        if inputs.is_empty() {
            return if self.body.output_types.as_slice().is_empty() {
                Ok(Vec::new())
            } else {
                Err(DifferentiationError::MissingTracedRematerializeInputLeaves.into())
            };
        }
        let exemplar_input = inputs[0].clone();
        let input_refs = inputs.iter().collect::<Vec<_>>();
        exemplar_input
            .context
            .trace(E::OperationCarrier::rematerialize_operation(self.clone()), input_refs.as_slice())
    }
}

/// Linear-only rematerialization boundary that always carries both the linear body and its transpose body.
#[derive(Clone, Debug)]
pub struct LinearRematerializeOperation<
    T: Type + PartialEq,
    V: Traceable<T> + Parameter,
    O = LinearArrayOperation<V, T>,
> {
    /// The forward linear body sub-program.
    pub body: FlatTracedRematerialize<T, V, O>,

    /// The transpose linear body.
    pub transpose_body: FlatTracedRematerialize<T, V, O>,
}

impl<T: Type + PartialEq, V: Traceable<T>, O> LinearRematerializeOperation<T, V, O> {
    /// Builds one linear rematerialize op with an explicit transpose body.
    #[inline]
    pub fn new(body: FlatTracedRematerialize<T, V, O>, transpose_body: FlatTracedRematerialize<T, V, O>) -> Self {
        Self { body, transpose_body }
    }
}

impl<T: Type + PartialEq, V: Traceable<T>, O: Clone> LinearRematerializeOperation<T, V, O> {
    fn transpose_op(&self) -> Self {
        Self::new(self.transpose_body.clone(), self.body.clone())
    }
}

impl<T: Type + PartialEq, V: Traceable<T>, O> Display for LinearRematerializeOperation<T, V, O>
where
    Self: Operation<T>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl<T: Type + PartialEq, V: Traceable<T>, O: Clone + Operation<T>> Operation<T>
    for LinearRematerializeOperation<T, V, O>
{
    #[inline]
    fn name(&self) -> &'static str {
        "rematerialize"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        check_input_count!(input_types, self.body.input_types.len(), TypeError);
        if input_types != self.body.input_types.as_slice() {
            return Err(TypeError {
                message: "rematerialize input types do not match the captured body signature".to_string(),
            });
        }
        Ok(self.body.output_types.clone())
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.program("body", &self.body.program)?;
            operation.program("transpose_body", &self.transpose_body.program)
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
        self.body.program.interpret(inputs.to_vec())
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
            return if self.body.input_types.as_slice().is_empty() {
                Ok(Vec::new())
            } else {
                Err(DifferentiationError::MissingLinearRematerializeTransposeCotangentLeaves.into())
            };
        }
        if output_cotangents.iter().all(Option::is_none) {
            return Ok(vec![None; self.body.input_types.as_slice().len()]);
        }
        let materialized = output_cotangents
            .iter()
            .zip(transpose.body.input_types.as_slice().iter())
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
    use crate::operations::constants::SupportsZero;
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
    LinearArrayOperation<V, ArrayType>: Clone
        + InterpretableOperation<ArrayType, V>
        + LinearOperation<ArrayType, V, LinearArrayOperation<V, ArrayType>>,
    E: LinearizableEngine<Type = ArrayType, Value = V, LinearOperationCarrier = LinearArrayOperation<V, ArrayType>>
        + ?Sized
        + 'static,
{
    let body_program = &body.program;
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
    E::OperationCarrier:
        Clone + InterpretableOperation<ArrayType, V> + SupportsRematerialize<ArrayType, V, E::LinearOperationCarrier>,
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
        let mut builder = ProgramBuilder::<ArrayType, V, E::OperationCarrier>::new();
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
            E::OperationCarrier::rematerialize_operation(RematerializeOperation::new(body)),
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
/// use crate::tracing_v2::{compile_grad, rematerialize};
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
