//! Batching and vmapping support for `ryft-core`.
//!
//! This module owns the explicit batching surface exposed by the crate:
//! [`Batch`], [`stack`], [`unstack`], [`vmap`], and the traced higher-order `vmap` operation types
//! that let batching survive as a first-class staged-program node.
//!
//! The execution model mirrors the rest of [`crate::tracing_v2`]:
//!
//! - concrete leaves batch eagerly as explicit lane lists inside [`Batch`];
//! - traced leaves stage compact higher-order `vmap` operations instead of duplicating scalar work.
//!
//! That split keeps the public batching API small while preserving enough structure for later
//! transforms and backend lowerings to reason about batched programs directly.

use std::{
    cell::RefCell,
    fmt::{Debug, Display},
    marker::PhantomData,
    ops::{Add, Mul, Neg},
    rc::Rc,
};

use ryft_macros::Parameter;
use thiserror::Error;

use crate::{
    parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily, Placeholder},
    tracing::{
        Atom, AtomId, Instruction, InterpretableOperation, OneLike, Operation, Program, ProgramBuilder, Traceable,
        TracingError, Value, ZeroLike,
    },
    tracing_v2::{
        LinearOperation, LinearPrimitiveOperation, LinearTerm, PrimitiveOperation, Tracer,
        engine::Engine,
        linear::{linearize_program, replay_program_linearized_jit, transpose_linear_program_with_output_examples},
        operations::{
            AddOperation, CoreLinearProgramOperation, DifferentiableOperation, MulOperation, NegOperation,
            VectorizableOperation,
        },
    },
    types::{ArrayType, Type, TypeError, Typed},
};

/// Checks that all batched inputs carry the same number of lanes.
///
/// This macro is intended for traced batching rules that already return `Result<_, TracingError>`.
/// Empty or singleton input slices always pass. When any lane count differs from the first batch,
/// it returns early from the enclosing function with
/// [`BatchingError::MismatchedBatchSize`] wrapped in [`TracingError`].
#[macro_export]
macro_rules! check_batch_sizes {
    ($inputs:expr $(,)?) => {{
        let inputs = &$inputs;
        if let Some(first_input) = inputs.first() {
            let expected_lane_count = first_input.len();
            if inputs.iter().skip(1).any(|input| input.len() != expected_lane_count) {
                return Err($crate::batching::BatchingError::MismatchedBatchSize.into());
            }
        }
    }};
}

pub use crate::check_batch_sizes;

/// Error type for explicit batching and `vmap`-specific failures.
///
/// [`BatchingError`] owns the failures that are specific to batching and vmapping semantics.
/// Broader tracing and staged-program construction failures still live in
/// [`TracingError`](crate::tracing::TracingError), which wraps this type when batching
/// participates inside larger transforms.
#[derive(Clone, Debug, Error, Eq, Hash, PartialEq)]
pub enum BatchingError {
    /// Structured lanes did not share the same `Parameterized` shape.
    #[error("mismatched parameter structures across batch lanes")]
    MismatchedParameterStructures,

    /// A batching transform encountered zero lanes and therefore could not infer a batch size.
    #[error("encountered an empty batch")]
    EmptyBatch,

    /// Different batched leaves disagreed on the number of lanes they carried.
    #[error("mismatched batch sizes across batched leaves")]
    MismatchedBatchSize,

    /// A traced `vmap` body produced a different per-lane output structure than it consumed.
    #[error("traced vmap only supports bodies that preserve the per-lane output structure")]
    VMapBodyMustPreservePerLaneOutputStructure,

    /// Traced `vmap` replay could not recover the staging context because no input leaf was available.
    #[error("traced vmap requires at least one input leaf to recover the staging context")]
    VMapMissingInputStagingContext,

    /// Linearized `vmap` replay could not recover the tangent staging context because no tangent leaf was available.
    #[error("linear vmap replay requires at least one tangent leaf to recover the staging context")]
    VMapMissingTangentStagingContext,

    /// Linear `vmap` transposition could not recover the staging context because no output cotangent leaf was available.
    #[error("linear vmap transpose requires at least one output cotangent leaf to recover the staging context")]
    VMapMissingOutputCotangentStagingContext,

    /// Wrapper around parameter-lifting failures from the `Parameterized` infrastructure.
    #[error(transparent)]
    Parameter(#[from] ParameterError),
}

/// Batched leaf value represented as an explicit list of lanes.
///
/// [`Batch`] is the concrete batching counterpart to [`Tracer`](crate::tracing_v2::Tracer): when a
/// transform wants to interpret a batched computation eagerly, each leaf becomes one `Batch<V>`
/// containing a lane-by-lane view of the input. Primitive batching rules consume and produce these
/// values directly.
#[derive(Clone, Debug, PartialEq, Parameter)]
pub struct Batch<V> {
    /// Lane values carried by this batch in semantic lane order.
    lanes: Vec<V>,
}

impl<V> Batch<V> {
    /// Creates a new batched value from a list of lanes.
    ///
    /// The lane order is semantically meaningful and is preserved by [`stack`], [`unstack`], and
    /// primitive batching rules.
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

fn single_output<V>(mut outputs: Vec<Batch<V>>, op: &'static str) -> Batch<V> {
    debug_assert_eq!(outputs.len(), 1, "{op} should produce a single batched output");
    outputs.pop().expect("single-output primitive should return one batched output")
}

impl<V: Traceable<ArrayType> + Add<Output = V>> Add for Batch<V> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        single_output(AddOperation.batch(&[self, rhs]).expect("add batching rule should succeed"), "add")
    }
}

impl<V: Traceable<ArrayType> + Mul<Output = V>> Mul for Batch<V> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        single_output(MulOperation.batch(&[self, rhs]).expect("mul batching rule should succeed"), "mul")
    }
}

impl<V: Traceable<ArrayType> + Neg<Output = V>> Neg for Batch<V> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        single_output(NegOperation.batch(&[self]).expect("neg batching rule should succeed"), "neg")
    }
}

impl<V: ZeroLike> ZeroLike for Batch<V> {
    #[inline]
    fn zero_like(&self) -> Self {
        Self::new(self.lanes.iter().map(ZeroLike::zero_like).collect())
    }
}

impl<V: OneLike> OneLike for Batch<V> {
    #[inline]
    fn one_like(&self) -> Self {
        Self::new(self.lanes.iter().map(OneLike::one_like).collect())
    }
}

/// Stacks a list of structured inputs into one structured value whose leaves are [`Batch`] values.
///
/// This is the structural entry point for concrete `vmap` execution. It transposes a
/// "batch-of-structures" into a "structure-of-batches" so the user function can run once over
/// batched leaves instead of once per lane.
pub fn stack<
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq, Family: ParameterizedFamily<Batch<V>>>,
    V: Parameter,
>(
    inputs: Vec<Input>,
) -> Result<Input::To<Batch<V>>, BatchingError> {
    let mut inputs = inputs.into_iter();
    let first = inputs.next().ok_or(BatchingError::EmptyBatch)?;
    let structure = first.parameter_structure();
    let parameter_count = structure.parameter_count();
    let mut buckets = (0..parameter_count).map(|_| Vec::new()).collect::<Vec<Vec<V>>>();
    let first_parameters = first.into_parameters().collect::<Vec<_>>();

    for (bucket, parameter) in buckets.iter_mut().zip(first_parameters) {
        bucket.push(parameter);
    }

    for input in inputs {
        if input.parameter_structure() != structure {
            return Err(BatchingError::MismatchedParameterStructures);
        }

        for (bucket, parameter) in buckets.iter_mut().zip(input.into_parameters()) {
            bucket.push(parameter);
        }
    }

    Ok(Input::To::<Batch<V>>::from_parameters(structure, buckets.into_iter().map(Batch::new))?)
}

/// Splits a structured batch back into one structured value per lane.
///
/// This is the inverse of [`stack`]. Concrete `vmap` uses it after the batched body runs so the
/// caller gets one structured output per original lane.
pub fn unstack<
    Input: Parameterized<V, ParameterStructure: Clone, Family: ParameterizedFamily<Batch<V>>>,
    V: Parameter,
>(
    batched: Input::To<Batch<V>>,
) -> Result<Vec<Input>, BatchingError> {
    let structure = batched.parameter_structure();
    let batches = batched.into_parameters().collect::<Vec<_>>();
    if batches.is_empty() {
        return Ok(Vec::new());
    }

    let lane_count = batches[0].len();
    if batches.iter().any(|batch| batch.len() != lane_count) {
        return Err(BatchingError::MismatchedBatchSize);
    }

    let mut lane_parameters = (0..lane_count).map(|_| Vec::with_capacity(batches.len())).collect::<Vec<Vec<V>>>();
    for batch in batches {
        for (lane_index, value) in batch.into_lanes().into_iter().enumerate() {
            lane_parameters[lane_index].push(value);
        }
    }

    lane_parameters
        .into_iter()
        .map(|parameters| Input::from_parameters(structure.clone(), parameters).map_err(BatchingError::from))
        .collect()
}

/// Dispatch trait used by [`vmap`] so it can handle both concrete batches and already traced values.
///
/// The trait is the batching analogue of the dispatch seams used by
/// [`jvp`](crate::tracing_v2::jvp) and [`grad`](crate::tracing_v2::grad): the public transform
/// stays small while the concrete, traced, and nested-batch execution strategies each get their
/// own implementation.
#[doc(hidden)]
pub(crate) trait VMapInvocationLeaf<'engine, E, Input, Output>: Parameter + Sized
where
    E: Engine<Type = ArrayType> + ?Sized,
    E::Value: Traceable<ArrayType>,
    E::TracingOperation: Operation<ArrayType>,
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq, Family: ParameterizedFamily<Batch<Self>>>,
    Output: Parameterized<Self, ParameterStructure: Clone, Family: ParameterizedFamily<Batch<Self>>>,
{
    /// Invokes [`vmap`] for one concrete leaf regime.
    fn vmap<F: FnOnce(Input::To<Batch<Self>>) -> Output::To<Batch<Self>>>(
        engine: &'engine E,
        builder: Rc<RefCell<ProgramBuilder<ArrayType, E::Value, E::TracingOperation>>>,
        function: F,
        inputs: Vec<Input>,
    ) -> Result<Vec<Output>, TracingError>;
}

/// Concrete-value dispatch for [`vmap`]: stacks inputs into [`Batch`] leaves, applies the user
/// function over the batched representation, and unstacks the result back into per-lane outputs.
///
/// No op-capability (`Sin` / `Cos` / `MatrixOps` / `ReshapeOps`) bounds on `V` are required here
/// because the body of `vmap` never exercises them directly. It stacks, unstacks, and invokes the
/// user's closure on `Batch<V>` values, and any capability the closure actually uses is enforced at
/// the call site through the conditional op-local trait impls on [`Batch`].
impl<
    'engine,
    E: Engine<Type = ArrayType> + ?Sized,
    V: Traceable<ArrayType> + Value<ArrayType>,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq, Family: ParameterizedFamily<Batch<V>>>,
    Output: Parameterized<V, ParameterStructure: Clone, Family: ParameterizedFamily<Batch<V>>>,
> VMapInvocationLeaf<'engine, E, Input, Output> for V
where
    E::Value: Traceable<ArrayType>,
    E::TracingOperation: Operation<ArrayType>,
{
    fn vmap<F: FnOnce(Input::To<Batch<Self>>) -> Output::To<Batch<Self>>>(
        _engine: &'engine E,
        _builder: Rc<RefCell<ProgramBuilder<ArrayType, E::Value, E::TracingOperation>>>,
        function: F,
        inputs: Vec<Input>,
    ) -> Result<Vec<Output>, TracingError> {
        let batched_input = stack(inputs)?;
        Ok(unstack(function(batched_input))?)
    }
}

/// Already-traced dispatch for [`vmap`]: stages a compact higher-order [`VMapOperation`] in the
/// enclosing [`Tracer`] scope instead of eagerly duplicating the scalar program per lane.
impl<
    'engine,
    E: Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized + 'static,
    V: Traceable<ArrayType>
        + Parameterized<
            V,
            ParameterStructure = Placeholder,
            To<Tracer<'engine, E>> = Tracer<'engine, E>,
            Family: ParameterizedFamily<Tracer<'engine, E>>,
        >,
    Input: Parameterized<
            Tracer<'engine, E>,
            ParameterStructure: Clone + PartialEq,
            To<Tracer<'engine, E>> = Input,
            Family: ParameterizedFamily<Batch<Tracer<'engine, E>>>
                        + ParameterizedFamily<V>
                        + ParameterizedFamily<ArrayType>,
        >,
    Output: Parameterized<
            Tracer<'engine, E>,
            ParameterStructure: Clone,
            To<Tracer<'engine, E>> = Output,
            Family: ParameterizedFamily<Batch<Tracer<'engine, E>>>
                        + ParameterizedFamily<Tracer<'engine, E>>
                        + ParameterizedFamily<V>
                        + ParameterizedFamily<ArrayType>,
        >,
    O: Operation<ArrayType> + InterpretableOperation<ArrayType, V> + VMapTracingOperation<ArrayType, V, L>,
    L: Clone,
> VMapInvocationLeaf<'engine, E, Input, Output> for Tracer<'engine, E>
where
    Input::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'engine, E>> = Input, To<V> = Input::To<V>>,
    Output::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'engine, E>> = Output, To<V> = Output::To<V>>,
    Vec<V>: Parameterized<V, To<Tracer<'engine, E>> = Vec<Tracer<'engine, E>>, ParameterStructure = Vec<Placeholder>>,
    <Vec<V> as Parameterized<V>>::Family: ParameterizedFamily<Tracer<'engine, E>>,
{
    fn vmap<F: FnOnce(Input::To<Batch<Self>>) -> Output::To<Batch<Self>>>(
        engine: &'engine E,
        builder: Rc<RefCell<ProgramBuilder<ArrayType, E::Value, E::TracingOperation>>>,
        function: F,
        inputs: Vec<Input>,
    ) -> Result<Vec<Output>, TracingError> {
        let mut inputs = inputs.into_iter();
        let first_input = inputs.next().ok_or(BatchingError::EmptyBatch)?;
        let input_structure = first_input.parameter_structure();
        let mut traced_inputs = vec![first_input.into_parameters().collect::<Vec<_>>()];
        for input in inputs {
            if input.parameter_structure() != input_structure {
                return Err(BatchingError::MismatchedParameterStructures.into());
            }
            traced_inputs.push(input.into_parameters().collect::<Vec<_>>());
        }

        let lane_count = traced_inputs.len();
        let input_leaf_count = input_structure.parameter_count();
        let exemplar_input_types = Input::To::<ArrayType>::from_parameters(
            input_structure.clone(),
            traced_inputs[0].iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
        )?;
        let (exemplar_output_types, body_program): (
            Output::To<ArrayType>,
            Program<ArrayType, V, O, Input::To<V>, Output::To<V>>,
        ) = crate::tracing_v2::jit::trace(
            engine,
            |lane_inputs| {
                let batched_inputs = Input::To::<Batch<Tracer<'engine, E>>>::from_parameters(
                    lane_inputs.parameter_structure(),
                    lane_inputs.into_parameters().map(|input| Batch::new(vec![input])),
                )?;
                let batched_outputs = function(batched_inputs);
                let output_structure = batched_outputs.parameter_structure();
                let mut lane_outputs = Vec::new();
                for batch in batched_outputs.into_parameters() {
                    let mut outputs = batch.into_lanes();
                    if outputs.len() != 1 {
                        return Err(BatchingError::VMapBodyMustPreservePerLaneOutputStructure.into());
                    }
                    lane_outputs.push(outputs.pop().expect("single-lane batches should contain one output"));
                }
                Output::from_parameters(output_structure, lane_outputs).map_err(TracingError::from)
            },
            exemplar_input_types,
        )?;
        let body_program = body_program.with_folded_constants()?.simplified()?;

        let output_structure = exemplar_output_types.parameter_structure();
        let output_leaf_count = output_structure.parameter_count();
        let flat_input_structure = vec![Placeholder; input_leaf_count];
        let flat_output_structure = vec![Placeholder; output_leaf_count];
        let body = FlatTracedVMap::from_parts(
            lane_count,
            body_program
                .input_ids
                .iter()
                .map(|input| body_program.atoms[input.index].r#type().into_owned())
                .collect::<Vec<_>>(),
            exemplar_output_types.parameters().cloned().collect::<Vec<_>>(),
            Program {
                atoms: body_program.atoms.clone(),
                input_ids: body_program.input_ids.clone(),
                output_ids: body_program.output_ids.clone(),
                instructions: body_program.instructions.clone(),
                input_structure: flat_input_structure,
                output_structure: flat_output_structure,
                marker: PhantomData,
            },
        );

        let staged_inputs = traced_inputs.into_iter().flatten().collect::<Vec<_>>();
        let staged_outputs =
            Tracer::apply_staged_op(engine, builder, staged_inputs.as_slice(), O::vmap_op(VMapOperation::new(body)))?;
        (0..lane_count)
            .map(|lane_index| {
                let start = lane_index * output_leaf_count;
                let end = start + output_leaf_count;
                Output::from_parameters(output_structure.clone(), staged_outputs[start..end].iter().cloned())
                    .map_err(TracingError::from)
            })
            .collect()
    }
}

/// Nested-batch dispatch for [`vmap`], enabling `vmap(|xs| vmap(g, xs))` and related recursive
/// vectorization patterns.
impl<
    'engine,
    E: Engine<Type = ArrayType> + ?Sized,
    V: Traceable<ArrayType>,
    Input: Parameterized<Batch<V>, ParameterStructure: Clone + PartialEq, Family: ParameterizedFamily<Batch<Batch<V>>>>,
    Output: Parameterized<Batch<V>, ParameterStructure: Clone, Family: ParameterizedFamily<Batch<Batch<V>>>>,
> VMapInvocationLeaf<'engine, E, Input, Output> for Batch<V>
where
    E::Value: Traceable<ArrayType>,
    E::TracingOperation: Operation<ArrayType>,
{
    fn vmap<F: FnOnce(Input::To<Batch<Self>>) -> Output::To<Batch<Self>>>(
        _engine: &'engine E,
        _builder: Rc<RefCell<ProgramBuilder<ArrayType, E::Value, E::TracingOperation>>>,
        function: F,
        inputs: Vec<Input>,
    ) -> Result<Vec<Output>, TracingError> {
        let batched_input = stack(inputs)?;
        Ok(unstack(function(batched_input))?)
    }
}

/// Maps `function` over a leading batch axis.
///
/// Conceptually, [`vmap`] lifts a scalar function into a batched function. For concrete inputs it
/// does so by stacking the input leaves into [`Batch`] values, running the user closure once, and
/// then unstacking the result back into one output per lane. For traced inputs it instead stages a
/// compact higher-order `vmap` operation so later transforms and lowerings can treat batching as a
/// first-class program construct.
///
/// # Parameters
///
///   - `engine`: backend engine used when tracing or interpreting the batched body.
///   - `builder`: enclosing staged-program builder that receives the higher-order `vmap`
///     instruction when the inputs are already traced. Concrete execution paths ignore it.
///   - `function`: batched body to run or trace.
///   - `inputs`: one structured input per batch lane.
#[allow(private_bounds)]
pub fn vmap<
    'engine,
    E: Engine<Type = ArrayType> + ?Sized,
    F: FnOnce(Input::To<Batch<V>>) -> Output::To<Batch<V>>,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq, Family: ParameterizedFamily<Batch<V>>>,
    Output: Parameterized<V, ParameterStructure: Clone, Family: ParameterizedFamily<Batch<V>>>,
    V: VMapInvocationLeaf<'engine, E, Input, Output>,
>(
    engine: &'engine E,
    builder: Rc<RefCell<ProgramBuilder<ArrayType, E::Value, E::TracingOperation>>>,
    function: F,
    inputs: Vec<Input>,
) -> Result<Vec<Output>, TracingError>
where
    E::Value: Traceable<ArrayType>,
    E::TracingOperation: Operation<ArrayType>,
{
    V::vmap(engine, builder, function, inputs)
}

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
        Vec<V>: Parameterized<V, ParameterStructure: Clone + Debug + PartialEq>,
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
/// Ordinary traced programs store [`VMapOperation`] when vectorization is preserved symbolically
/// instead of being unrolled into repeated scalar instructions.
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

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        let expected_inputs = self.body.repeated_input_types();
        if input_types.len() != expected_inputs.len() {
            return Err(TypeError {
                message: format!("vmap expected {} input types but got {}", expected_inputs.len(), input_types.len()),
            });
        }
        if input_types != expected_inputs.as_slice() {
            return Err(TypeError { message: "vmap input types do not match the captured body signature".to_string() });
        }
        Ok(self.body.repeated_output_types())
    }
}

impl<V: Traceable<ArrayType>, O: Clone + Operation<ArrayType>, L: Clone> InterpretableOperation<ArrayType, V>
    for VMapOperation<ArrayType, V, O, L>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + Debug + PartialEq>,
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
    Vec<V>: Parameterized<V, ParameterStructure: Clone + Debug + PartialEq>,
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
        if inputs.is_empty() {
            return if self.body.total_output_count() == 0 {
                Ok(Vec::new())
            } else {
                Err(BatchingError::VMapMissingInputStagingContext.into())
            };
        }
        let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
        let exemplar_primal_input = primal_inputs[0].clone();
        let linear_builder = inputs[0].tangent.builder.clone();
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
            let lane_outputs = replay_program_linearized_jit::<_, _, _, O, L, E>(
                exemplar_primal_input.engine,
                exemplar_primal_input.builder.clone(),
                linear_builder.clone(),
                &lane_program,
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

impl<V: Value<ArrayType> + ZeroLike + 'static, O: Clone + 'static>
    DifferentiableOperation<
        ArrayType,
        V,
        LinearTerm<ArrayType, V, LinearPrimitiveOperation<ArrayType, V>>,
        O,
        LinearPrimitiveOperation<ArrayType, V>,
    > for VMapOperation<ArrayType, V, O>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + Debug + PartialEq>,
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
        let tangent_builder = if let Some(first_tangent) = tangent_inputs.first() {
            first_tangent.builder.clone()
        } else if self.body.total_output_count() == 0 {
            return Ok(Vec::new());
        } else {
            return Err(BatchingError::VMapMissingTangentStagingContext.into());
        };
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
    Vec<V>: Parameterized<V, ParameterStructure: Clone + Debug + PartialEq>,
    O: Operation<ArrayType> + InterpretableOperation<ArrayType, V> + VMapTracingOperation<ArrayType, V, L>,
{
    fn interpret(&self, inputs: &[Tracer<'engine, E>]) -> Result<Vec<Tracer<'engine, E>>, TracingError> {
        if inputs.is_empty() {
            return if self.body.total_output_count() == 0 {
                Ok(Vec::new())
            } else {
                Err(BatchingError::VMapMissingInputStagingContext.into())
            };
        }
        let exemplar_input = inputs[0].clone();
        Tracer::apply_staged_op(exemplar_input.engine, exemplar_input.builder.clone(), inputs, O::vmap_op(self.clone()))
    }
}

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

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        let expected_inputs = self.body.repeated_input_types();
        if input_types.len() != expected_inputs.len() {
            return Err(TypeError {
                message: format!("vmap expected {} input types but got {}", expected_inputs.len(), input_types.len()),
            });
        }
        if input_types != expected_inputs.as_slice() {
            return Err(TypeError { message: "vmap input types do not match the captured body signature".to_string() });
        }
        Ok(self.body.repeated_output_types())
    }
}

impl<V: Traceable<ArrayType>, O: Clone + Operation<ArrayType>> InterpretableOperation<ArrayType, V>
    for LinearVMapOperation<ArrayType, V, O>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + Debug + PartialEq>,
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
        if output_cotangents.is_empty() {
            return if self.body.total_input_count() == 0 {
                Ok(Vec::new())
            } else {
                Err(BatchingError::VMapMissingOutputCotangentStagingContext.into())
            };
        }
        let exemplar_output_cotangent = output_cotangents[0].clone();
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
    Vec<V>: Parameterized<V, ParameterStructure: Clone + Debug + PartialEq>,
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

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use crate::parameters::Parameter;
    use indoc::indoc;
    use ryft_macros::Parameter;

    use crate::{
        batching::{Batch, BatchingError, stack, unstack, vmap},
        tracing::{OneLike, Program, ProgramBuilder},
        tracing_v2::{PrimitiveOperation, Sin, Tracer, engine::ArrayScalarEngine, test_support},
        types::ArrayType,
    };

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
        assert!(matches!(result, Err(BatchingError::EmptyBatch)));
        test_support::assert_reference_scalar_sine_jit_rendering();
    }

    #[test]
    fn unstack_rejects_mismatched_lane_counts() {
        let batched = (Batch::new(vec![1.0f64]), Batch::new(vec![2.0f64, 3.0f64]));
        let result = unstack::<(f64, f64), f64>(batched);
        assert!(matches!(result, Err(BatchingError::MismatchedBatchSize)));
        test_support::assert_reference_scalar_sine_jit_rendering();
    }

    #[test]
    fn vmap_exposes_batch_axis_size() {
        let engine = ArrayScalarEngine::<f64>::new();
        let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        let outputs: Vec<f64> = vmap(
            &engine,
            builder,
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
        let engine = ArrayScalarEngine::<f64>::new();
        let (output, program): (f64, Program<ArrayType, f64, PrimitiveOperation<ArrayType, f64>, f64, f64>) =
            crate::tracing_v2::interpret_and_trace(
                &engine,
                |x| {
                    let builder = x.builder.clone();
                    let outputs: Vec<Tracer<ArrayScalarEngine<f64>>> =
                        vmap(&engine, builder, |batch| batch.clone() + batch.one_like(), vec![x.clone(), x])?;
                    Ok(outputs[0].clone() + outputs[1].clone())
                },
                2.0f64,
            )
            .unwrap();

        assert_eq!(output, 6.0);
        assert_eq!(program.interpret(3.0f64).unwrap(), 8.0);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[], %2:f64[] = vmap %0 %0
                    %3:f64[] = add %1 %2
                in (%3)
            "}
            .trim_end(),
        );
    }

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    #[test]
    fn test_vmap_of_grad_computes_per_lane_gradients() {
        let engine = crate::tracing_v2::engine::ArrayScalarEngine::<f64>::new();
        let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        let gradients: Vec<f64> = vmap(
            &engine,
            builder,
            |batch: Batch<f64>| {
                crate::tracing_v2::grad(&engine, |x| x.clone() * x.clone() + x.sin(), batch)
                    .expect("batched grad should succeed")
            },
            vec![1.0f64, 2.0, 3.0],
        )
        .unwrap();

        approx_eq(gradients[0], 2.0 * 1.0 + 1.0f64.cos());
        approx_eq(gradients[1], 2.0 * 2.0 + 2.0f64.cos());
        approx_eq(gradients[2], 2.0 * 3.0 + 3.0f64.cos());
        test_support::assert_reference_scalar_sine_jit_rendering();
    }

    #[test]
    fn test_vmap_of_value_and_grad_returns_batched_values_and_gradients() {
        let engine = crate::tracing_v2::engine::ArrayScalarEngine::<f64>::new();
        let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        let results: Vec<(f64, f64)> = vmap(
            &engine,
            builder,
            |batch: Batch<f64>| {
                crate::tracing_v2::value_and_grad(&engine, |x| x.clone() * x.clone() + x.sin(), batch)
                    .expect("batched value_and_grad should succeed")
            },
            vec![1.0f64, 2.0, 3.0],
        )
        .unwrap();

        for (index, x) in [1.0f64, 2.0, 3.0].into_iter().enumerate() {
            approx_eq(results[index].0, x * x + x.sin());
            approx_eq(results[index].1, 2.0 * x + x.cos());
        }
        test_support::assert_reference_scalar_sine_jit_rendering();
    }

    #[test]
    fn test_vmap_of_jvp_propagates_tangents_per_lane() {
        let engine = crate::tracing_v2::engine::ArrayScalarEngine::<f64>::new();
        let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        let results: Vec<(f64, f64)> = vmap(
            &engine,
            builder,
            |(primals, tangents): (Batch<f64>, Batch<f64>)| {
                crate::tracing_v2::jvp(&engine, |x| x.clone() * x.clone() + x.sin(), primals, tangents)
                    .expect("batched jvp should succeed")
            },
            vec![(1.0f64, 1.0f64), (2.0, 0.5), (3.0, 2.0)],
        )
        .unwrap();

        for (index, (x, t)) in [(1.0f64, 1.0f64), (2.0, 0.5), (3.0, 2.0)].into_iter().enumerate() {
            approx_eq(results[index].0, x * x + x.sin());
            approx_eq(results[index].1, (2.0 * x + x.cos()) * t);
        }
        test_support::assert_reference_scalar_sine_jit_rendering();
    }

    #[test]
    fn test_jvp_through_staged_vmap_propagates_tangents() {
        let engine = crate::tracing_v2::engine::ArrayScalarEngine::<f64>::new();
        let (primal, tangent): (f64, f64) = crate::tracing_v2::jvp(
            &engine,
            |x| {
                let builder = x.builder.clone();
                let outputs: Vec<Tracer<ArrayScalarEngine<f64>>> =
                    vmap(&engine, builder, |batch| batch.clone() * batch.clone() + batch.sin(), vec![x.clone(), x])
                        .unwrap();
                outputs[0].clone() + outputs[1].clone()
            },
            2.0f64,
            1.5f64,
        )
        .unwrap();

        approx_eq(primal, 2.0 * (2.0f64 * 2.0f64 + 2.0f64.sin()));
        approx_eq(tangent, 2.0 * (2.0 * 2.0f64 + 2.0f64.cos()) * 1.5);
        test_support::assert_reference_scalar_sine_jit_rendering();
    }

    #[test]
    fn test_vmap_compiles_for_leaf_without_float_matrix_or_reshape_ext_traits() {
        use std::borrow::Cow;
        use std::ops::Add;

        use crate::types::{ArrayType, DataType, Typed};

        #[derive(Clone, Debug, PartialEq, Parameter)]
        struct Int64(i64);

        impl Typed<ArrayType> for Int64 {
            fn r#type(&self) -> Cow<'_, ArrayType> {
                Cow::Owned(ArrayType::scalar(DataType::I64))
            }
        }

        impl crate::tracing::Traceable<ArrayType> for Int64 {}
        impl crate::tracing::Value<ArrayType> for Int64 {}

        impl Add for Int64 {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                Self(self.0 + rhs.0)
            }
        }

        impl crate::tracing::ZeroLike for Int64 {
            fn zero_like(&self) -> Self {
                Self(0)
            }
        }

        impl crate::tracing::OneLike for Int64 {
            fn one_like(&self) -> Self {
                Self(1)
            }
        }

        let engine = ArrayScalarEngine::<f64>::new();
        let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        let outputs: Vec<Int64> =
            vmap(&engine, builder, |batch: Batch<Int64>| batch.clone() + batch, vec![Int64(1), Int64(2), Int64(3)])
                .unwrap();
        assert_eq!(outputs, vec![Int64(2), Int64(4), Int64(6)]);
    }
}
