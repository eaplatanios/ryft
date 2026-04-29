use std::borrow::Cow;
use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::ops::{Add, Mul, Neg};
use std::rc::Rc;

use half::{bf16, f16};

use ryft_macros::Parameter;

use crate::parameters::{Parameter, Parameterized, ParameterizedFamily as ParameterFamily};
use crate::tracing::{AtomId, InterpretableOperation, Operation, Program, ProgramBuilder, Traceable, TracingError};
use crate::tracing_v2::forward::Differentiable;
use crate::tracing_v2::operations::constants::{OneLike, ZeroLike};
use crate::tracing_v2::operations::primitive::PrimitiveOperation;
use crate::tracing_v2::operations::{SupportsAdd, SupportsMul, SupportsNeg, TracedLinearizationCarrier};
use crate::types::{ArrayType, Type, Typed};

/// [`Engine`]s provide backend-specific functionality related to tracing, just-in-time compilation, automatic
/// differentiation, and potentially other [`Program`] transforms. They also define the kinds of [`Type`]s and
/// [`Traceable`] values that each backend supports and they are effectively what lets higher-order transforms
/// remain backend-invariant.
pub trait Engine {
    /// [`Type`]s that this [`Engine`] uses to represent the abstract metadata associated with its [`Traceable`] values.
    /// A commonly used [`Type`] is [`ArrayType`], though richer backends may use richer types.
    type Type: Type + Parameter;

    /// [`Traceable`] value types supported by this [`Engine`]. Instances of this type are what [`Program`]
    /// interpretation and eager transforms operate on. [`Engine::Type`] represents abstract staging metadata,
    /// while [`Engine::Value`] represents the runtime values that inhabit traced [`Program`]s during execution.
    type Value: Traceable<Self::Type>;

    /// Returns the additive-identity value (i.e., the _zero_ value) that corresponds to the provided type.
    fn zero(&self, r#type: &Self::Type) -> Result<Self::Value, TracingError>;

    /// Returns the multiplicative-identity value (i.e., the _one_ value) that corresponds to the provided type.
    fn one(&self, r#type: &Self::Type) -> Result<Self::Value, TracingError>;
}

/// [`StagingEngine`] extends [`Engine`] with a closed [`Operation`] carrier type that can be used to trace
/// and interpret [`Program`]s.
pub trait StagingEngine: Engine {
    /// Staged [`Operation`] type supported by this [`StagingEngine`].
    type Operation: Operation<Self::Type>;

    /// Traces the provided `function` into a [`Program`] for the provided input types, returning the output types of
    /// the traced [`Program`] along with that traced [`Program`] itself. This is the most symbolic tracing entry
    /// point in that it does not require concrete runtime input values but rather it only requires their types. The
    /// provided closure is executed once on [`Tracer`] values standing in for those input types to trace the function,
    /// and relies on [`Operation::infer_output_types`] for inferring output types.
    #[inline]
    fn trace<
        'e,
        F: FnOnce(I::To<Tracer<'e, Self>>) -> Result<O::To<Tracer<'e, Self>>, TracingError>,
        I: Parameterized<Self::Type, Family: ParameterFamily<Self::Value> + ParameterFamily<Tracer<'e, Self>>>,
        O: Parameterized<Self::Type, Family: ParameterFamily<Self::Value> + ParameterFamily<Tracer<'e, Self>>>,
    >(
        &'e self,
        function: F,
        input_types: I,
    ) -> Result<
        (O, Program<Self::Type, Self::Value, Self::Operation, I::To<Self::Value>, O::To<Self::Value>>),
        TracingError,
    > {
        let program_builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        TracingEngine::new(self, program_builder).trace(function, input_types)
    }

    /// Traces the provided `function` into a [`Program`] using the provided input values and also interprets it using
    /// those same input values, returning the output values along with the traced [`Program`]. This function should be
    /// used instead of [`StagingEngine::trace`] when the caller wants to both trace a computation and execute it at the
    /// same time.
    fn interpret_and_trace<
        'e,
        F: FnOnce(I::To<Tracer<'e, Self>>) -> Result<O::To<Tracer<'e, Self>>, TracingError>,
        I: Parameterized<Self::Value, Family: ParameterFamily<Tracer<'e, Self>>, ParameterStructure: Debug + PartialEq>,
        O: Parameterized<Self::Value, Family: ParameterFamily<Tracer<'e, Self>>>,
    >(
        &'e self,
        function: F,
        input: I,
    ) -> Result<(O, Program<Self::Type, Self::Value, Self::Operation, I, O>), TracingError>
    where
        Self::Operation: Clone + InterpretableOperation<Self::Type, Self::Value>,
    {
        let input_structure = input.parameter_structure();
        let input_values = input.into_parameters().collect::<Vec<_>>();
        let input_types = input_values.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
        let mut output_structure = None;
        let (_, flat_program): (
            Vec<Self::Type>,
            Program<Self::Type, Self::Value, Self::Operation, Vec<Self::Value>, Vec<Self::Value>>,
        ) = self.trace(
            |flat_input| {
                let input = I::To::<Tracer<'e, Self>>::from_parameters(input_structure.clone(), flat_input)?;
                let output = function(input)?;
                output_structure = Some(output.parameter_structure());
                Ok(output.into_parameters().collect::<Vec<_>>())
            },
            input_types,
        )?;
        let output_structure = output_structure.unwrap();
        let Program { atoms, input_ids, output_ids, instructions, .. } = flat_program;
        let mut builder = ProgramBuilder::new();
        builder.atoms = atoms;
        builder.input_ids = input_ids;
        builder.instructions = instructions;
        let program = builder.build::<I, O>(output_ids, input_structure, output_structure)?;
        let program = program.simplified()?;
        let concrete_input = I::from_parameters(program.input_structure.clone(), input_values)?;
        Ok((program.interpret(concrete_input)?, program))
    }
}

/// Stateless engine that synthesizes scalar-compatible values from [`ArrayType`] metadata.
///
/// [`ScalarEngine<V>`] is the "minimal backend" used throughout tests and scalar-only examples. It demonstrates the
/// intended role of an [`Engine`] in the smallest possible form: there is no device handle, no mesh state, and no
/// backend registry, just the built-in primitive carriers plus metadata-driven construction of scalar identity values.
///
/// The engine ignores most of the supplied [`ArrayType`] metadata because scalar leaves have one canonical runtime
/// representation. That makes it a compact teaching example for the tracing stack: if a transform works against
/// [`ScalarEngine`], the same path can be reused by richer engines with sharding, device, or runtime context.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScalarEngine<V> {
    /// Phantom marker that ties the zero-sized engine to its scalar leaf type.
    marker: PhantomData<fn() -> V>,
}

impl<V> ScalarEngine<V> {
    /// Returns a new [`ScalarEngine<V>`].
    ///
    /// This zero-sized engine is a runtime no-op; it gives examples and tests an explicit backend token.
    #[inline]
    pub const fn new() -> Self {
        Self { marker: PhantomData }
    }
}

macro_rules! impl_scalar_engine_for_scalar {
    ($ty:ty, $zero:expr, $one:expr) => {
        impl Engine for ScalarEngine<$ty> {
            type Type = ArrayType;
            type Value = $ty;

            #[inline]
            fn zero(&self, _type: &ArrayType) -> Result<$ty, TracingError> {
                Ok($zero)
            }

            #[inline]
            fn one(&self, _type: &ArrayType) -> Result<$ty, TracingError> {
                Ok($one)
            }
        }

        impl StagingEngine for ScalarEngine<$ty> {
            type Operation = PrimitiveOperation<$ty>;
        }
    };
}

impl_scalar_engine_for_scalar!(bool, false, true);
impl_scalar_engine_for_scalar!(i8, 0i8, 1i8);
impl_scalar_engine_for_scalar!(i16, 0i16, 1i16);
impl_scalar_engine_for_scalar!(i32, 0i32, 1i32);
impl_scalar_engine_for_scalar!(i64, 0i64, 1i64);
impl_scalar_engine_for_scalar!(u8, 0u8, 1u8);
impl_scalar_engine_for_scalar!(u16, 0u16, 1u16);
impl_scalar_engine_for_scalar!(u32, 0u32, 1u32);
impl_scalar_engine_for_scalar!(u64, 0u64, 1u64);
impl_scalar_engine_for_scalar!(bf16, bf16::ZERO, bf16::ONE);
impl_scalar_engine_for_scalar!(f16, f16::ZERO, f16::ONE);
impl_scalar_engine_for_scalar!(f32, 0.0, 1.0);
impl_scalar_engine_for_scalar!(f64, 0.0, 1.0);

/// Active engine used while staging one program.
///
/// [`TracingEngine`] bundles the active [`StagingEngine`] reference with the active
/// [`ProgramBuilder`](crate::tracing::ProgramBuilder). Individual [`Tracer`] leaves carry a clone of this engine so
/// ordinary Rust operator traits can keep staging without requiring an explicit engine argument at every call site.
pub struct TracingEngine<'engine, E: StagingEngine + ?Sized> {
    /// Engine borrowed by this tracing engine for metadata-driven value synthesis and operation selection.
    engine: &'engine E,

    /// Shared builder that owns the staged program currently being traced.
    builder: Rc<RefCell<ProgramBuilder<E::Type, E::Value, E::Operation>>>,
}

impl<'engine, E: StagingEngine + ?Sized> TracingEngine<'engine, E> {
    /// Creates a tracing engine over `engine` and `builder`.
    ///
    /// The caller owns the tracing scope: all tracers used together must carry a [`TracingEngine`] that refers to the
    /// same outer engine handle and builder, or staging is rejected before mutation.
    #[inline]
    pub fn new(engine: &'engine E, builder: Rc<RefCell<ProgramBuilder<E::Type, E::Value, E::Operation>>>) -> Self {
        Self { engine, builder }
    }

    /// Returns the active staging engine used by this tracing engine.
    #[inline]
    pub fn outer_engine(&self) -> &'engine E {
        self.engine
    }

    /// Returns the shared program builder owned by this tracing engine.
    #[inline]
    pub fn builder(&self) -> &Rc<RefCell<ProgramBuilder<E::Type, E::Value, E::Operation>>> {
        &self.builder
    }

    /// Creates a sibling tracing engine over the same active engine handle and a fresh builder.
    #[inline]
    pub(crate) fn sibling(&self, builder: Rc<RefCell<ProgramBuilder<E::Type, E::Value, E::Operation>>>) -> Self {
        Self { engine: self.engine, builder }
    }

    /// Lifts a concrete engine value into a [`Tracer`] constant in this tracing engine.
    ///
    /// Traced JVP rules for value-capturing operations use this to turn captured runtime values into symbolic values
    /// staged in the outer program, preserving the symbolic dataflow.
    pub fn lift_constant(&self, value: E::Value) -> Tracer<'engine, E> {
        let r#type = <E::Value as Typed<E::Type>>::r#type(&value).into_owned();
        let atom = self.builder.borrow_mut().add_constant(value);
        self.tracer_from_staged_parts(atom, r#type)
    }

    /// Constructs a live tracer in this tracing engine from a staged atom and its cached abstract type.
    #[inline]
    pub fn tracer_from_staged_parts(&self, atom: AtomId, r#type: E::Type) -> Tracer<'engine, E> {
        Tracer { state: TracerState::Live(atom, r#type), engine: self.clone() }
    }

    /// Constructs a live tracer in this tracing engine by reading the staged atom's abstract type.
    #[inline]
    pub fn tracer_from_atom(&self, atom: AtomId) -> Tracer<'engine, E> {
        let r#type = self.builder.borrow().atoms[atom.index].r#type().into_owned();
        self.tracer_from_staged_parts(atom, r#type)
    }

    /// Constructs a poisoned tracer in this tracing engine.
    #[inline]
    pub(crate) fn poisoned_tracer(&self, r#type: E::Type) -> Tracer<'engine, E> {
        Tracer { state: TracerState::Poison(r#type), engine: self.clone() }
    }

    /// Stages one operation application in this tracing engine and returns tracers for its outputs.
    ///
    /// The method validates that all inputs belong to this tracing engine and then delegates normal instruction
    /// construction to [`ProgramBuilder::add_instruction`]. If the builder has already recorded an error, it avoids
    /// mutating the partial program and only uses abstract evaluation to synthesize poisoned output tracers with the
    /// expected types. If abstract evaluation also fails after an earlier builder error, a non-empty input list still
    /// produces one poisoned tracer using the first input type so later staging can continue to short-circuit.
    pub fn apply_staged_op(
        &self,
        inputs: &[Tracer<'engine, E>],
        op: E::Operation,
    ) -> Result<Vec<Tracer<'engine, E>>, TracingError> {
        if inputs.iter().any(|input| !Rc::ptr_eq(&self.builder, input.engine.builder())) {
            return Err(TracingError::MismatchedProgramBuilders);
        }
        if inputs
            .iter()
            .any(|input| !std::ptr::addr_eq(std::ptr::from_ref(input.engine.engine), std::ptr::from_ref(self.engine)))
        {
            return Err(TracingError::MismatchedEngines);
        }

        if self.builder.borrow().error.is_some() {
            let input_types = inputs.iter().map(|input| input.state.r#type().clone()).collect::<Vec<_>>();
            let output_types = match op.infer_output_types(input_types.as_slice()) {
                Ok(output_types) => output_types,
                Err(error) => {
                    let poison_type = input_types.first().cloned().ok_or(error)?;
                    return Ok(vec![self.poisoned_tracer(poison_type)]);
                }
            };
            return Ok(output_types.into_iter().map(|r#type| self.poisoned_tracer(r#type)).collect());
        }

        let input_atoms = inputs.iter().map(|input| input.atom_id()).collect::<Result<Vec<_>, _>>()?;
        let add_result = {
            let mut builder = self.builder.borrow_mut();
            match builder.add_instruction(op, input_atoms) {
                Ok(outputs) => Ok(outputs.to_vec()),
                Err(error) => {
                    if builder.error.is_none() {
                        builder.error = Some(error.clone());
                    }
                    Err(error)
                }
            }
        };
        let output_atoms = match add_result {
            Ok(output_atoms) => output_atoms,
            Err(error) => {
                let poison_type = match inputs.first() {
                    Some(input) => input.state.r#type().clone(),
                    None => return Err(error),
                };
                return Ok(vec![self.poisoned_tracer(poison_type)]);
            }
        };

        let output_states = {
            let builder = self.builder.borrow();
            output_atoms
                .into_iter()
                .map(|atom| TracerState::Live(atom, builder.atoms[atom.index].r#type().into_owned()))
                .collect::<Vec<_>>()
        };
        Ok(output_states.into_iter().map(|state| Tracer { state, engine: self.clone() }).collect())
    }

    /// Stages `function` directly from type metadata using this tracing engine's active builder.
    ///
    /// This builder-backed form lets nested traced transforms trace a fresh sibling program while retaining access to
    /// the outer engine context that supplies values and operation carriers.
    pub fn trace<F, Input, Output>(
        self,
        function: F,
        input_types: Input,
    ) -> Result<
        (Output, Program<E::Type, E::Value, E::Operation, Input::To<E::Value>, Output::To<E::Value>>),
        TracingError,
    >
    where
        Input: Parameterized<E::Type, Family: ParameterFamily<E::Value> + ParameterFamily<Tracer<'engine, E>>>,
        Output: Parameterized<E::Type, Family: ParameterFamily<E::Value> + ParameterFamily<Tracer<'engine, E>>>,
        F: FnOnce(Input::To<Tracer<'engine, E>>) -> Result<Output::To<Tracer<'engine, E>>, TracingError>,
    {
        let input_structure = input_types.parameter_structure();
        let builder = self.builder().clone();
        let traced_input = Input::To::<Tracer<'engine, E>>::from_parameters(
            input_types.parameter_structure(),
            input_types.into_parameters().map(|r#type| {
                let atom = builder.borrow_mut().add_input(r#type.clone());
                self.tracer_from_staged_parts(atom, r#type)
            }),
        )
        .map_err(TracingError::from)?;

        let (output_structure, output_types, outputs) = {
            let traced_output = function(traced_input)?;
            let output_structure = traced_output.parameter_structure();
            let traced_outputs = traced_output.into_parameters().collect::<Vec<_>>();
            let output_types = Output::from_parameters(
                output_structure.clone(),
                traced_outputs.iter().map(|output| output.r#type().into_owned()).collect::<Vec<_>>(),
            )?;
            if let Some(tracing_error) = builder.borrow_mut().error.take() {
                return Err(tracing_error);
            }
            let outputs = traced_outputs.into_iter().map(|output| output.atom_id()).collect::<Result<Vec<_>, _>>()?;
            let output_structure = output_types.parameter_structure();
            (output_structure, output_types, outputs)
        };
        drop(self);
        let builder = match Rc::try_unwrap(builder) {
            Ok(builder) => builder.into_inner(),
            Err(_) => return Err(TracingError::EscapedProgramBuilder),
        };
        let program =
            builder.build::<Input::To<E::Value>, Output::To<E::Value>>(outputs, input_structure, output_structure)?;
        Ok((output_types, program))
    }
}

impl<'engine, E: StagingEngine + ?Sized> Clone for TracingEngine<'engine, E> {
    fn clone(&self) -> Self {
        Self { engine: self.engine, builder: self.builder.clone() }
    }
}

impl<'engine, E: StagingEngine + ?Sized> Debug for TracingEngine<'engine, E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("TracingEngine").finish_non_exhaustive()
    }
}

impl<'engine, E: StagingEngine + ?Sized> Engine for TracingEngine<'engine, E> {
    type Type = E::Type;
    type Value = Tracer<'engine, E>;

    #[inline]
    fn zero(&self, r#type: &Self::Type) -> Result<Self::Value, TracingError> {
        let value = self.outer_engine().zero(r#type)?;
        let atom = self.builder.borrow_mut().add_constant(value);
        Ok(self.tracer_from_staged_parts(atom, r#type.clone()))
    }

    #[inline]
    fn one(&self, r#type: &Self::Type) -> Result<Self::Value, TracingError> {
        let value = self.outer_engine().one(r#type)?;
        let atom = self.builder.borrow_mut().add_constant(value);
        Ok(self.tracer_from_staged_parts(atom, r#type.clone()))
    }
}

impl<'engine, E> StagingEngine for TracingEngine<'engine, E>
where
    E: StagingEngine + ?Sized,
    E::Value: Differentiable<E::Type, Tangent = E::Value>,
    E::Operation: TracedLinearizationCarrier<E::Type, E::Value>,
    crate::tracing_v2::operations::AddOperation: Operation<E::Type>,
{
    type Operation = crate::tracing_v2::operations::AddOperation;
}

/// Execution state carried by a [`Tracer`] leaf.
///
/// Live tracers point at a concrete staged atom in the shared program builder. Poisoned tracers arise only after the
/// active tracing engine has already recorded an error and can no longer stage new instructions safely. They still
/// retain the inferred abstract output type so type queries and best-effort short-circuiting can continue without
/// manufacturing dummy atoms for a program that can no longer be finalized successfully.
#[derive(Clone, PartialEq, Eq)]
pub enum TracerState<T: Type> {
    /// Normal traced leaf backed by a concrete atom in the staged program.
    Live(AtomId, T),

    /// Poisoned traced leaf that carries only abstract output type information.
    Poison(T),
}

impl<T: Type> TracerState<T> {
    /// Returns the staged atom id for live tracers, if one exists.
    #[inline]
    pub fn live_atom(&self) -> Option<AtomId> {
        match self {
            Self::Live(atom, _) => Some(*atom),
            Self::Poison(_) => None,
        }
    }

    /// Returns the cached abstract type carried by this tracer state.
    #[inline]
    pub fn r#type(&self) -> &T {
        match self {
            Self::Live(_, r#type) | Self::Poison(r#type) => r#type,
        }
    }
}

impl<T: Type> Debug for TracerState<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Live(atom, _) => formatter.debug_tuple("Live").field(atom).finish(),
            Self::Poison(_) => formatter.write_str("Poison(..)"),
        }
    }
}

/// Symbolic leaf used while staging ordinary traced programs.
///
/// A [`Tracer`] is the value-level facade for one staged traced leaf. Primitive trait impls on [`Tracer`] stage
/// instructions in a shared [`ProgramBuilder`](crate::tracing::ProgramBuilder) instead of doing numerical work, and
/// return new tracers for the staged outputs. When tracing has already failed, later operations return poisoned tracers
/// that retain only abstract type metadata rather than manufacturing dummy atoms. This makes [`Tracer`] the symbolic
/// leaf used when `tracing_v2` executes a closure symbolically instead of eagerly.
#[derive(Parameter)]
pub struct Tracer<'engine, E: StagingEngine + ?Sized> {
    /// Execution state for this traced leaf.
    pub(crate) state: TracerState<E::Type>,

    /// Tracing engine that owns the shared builder and outer staging engine reference.
    pub engine: TracingEngine<'engine, E>,
}

impl<'engine, E: StagingEngine + ?Sized> Tracer<'engine, E> {
    /// Constructs a traced leaf from staged tracing parts.
    ///
    /// Callers that already know the staged atom's abstract type should prefer this constructor so future type queries
    /// can use cached metadata without re-borrowing the shared builder.
    #[inline]
    pub fn from_staged_parts(
        atom: AtomId,
        r#type: E::Type,
        builder: Rc<RefCell<ProgramBuilder<E::Type, E::Value, E::Operation>>>,
        engine: &'engine E,
    ) -> Self {
        TracingEngine::new(engine, builder).tracer_from_staged_parts(atom, r#type)
    }

    /// Returns this tracer's leaf state.
    #[inline]
    pub fn state(&self) -> &TracerState<E::Type> {
        &self.state
    }

    /// Returns the outer staging engine borrowed by this tracer's tracing engine.
    #[inline]
    pub fn outer_engine(&self) -> &'engine E {
        self.engine.outer_engine()
    }

    /// Returns the shared program builder for this tracer's tracing engine.
    #[inline]
    pub fn builder(&self) -> &Rc<RefCell<ProgramBuilder<E::Type, E::Value, E::Operation>>> {
        self.engine.builder()
    }

    /// Returns the staged atom id for this tracer when it is still live.
    pub fn atom_id(&self) -> Result<AtomId, TracingError> {
        self.state.live_atom().ok_or(TracingError::PoisonedTracer)
    }

    /// Stages a single-input operation application and returns its unique output.
    ///
    /// Convenience wrapper for operator trait implementations whose staged operation should produce one output.
    pub fn unary(self, op: E::Operation) -> Self {
        let engine = self.engine.clone();
        engine
            .apply_staged_op(std::slice::from_ref(&self), op)
            .expect("unary traced staging should preserve non-empty inputs")
            .into_iter()
            .next()
            .expect("unary traced staging should produce one output")
    }

    /// Stages a two-input operation application and returns its unique output.
    ///
    /// Convenience wrapper for operator trait implementations whose staged operation should produce one output.
    pub fn binary(self, rhs: Self, op: E::Operation) -> Self {
        debug_assert!(Rc::ptr_eq(self.builder(), rhs.builder()));
        let engine = self.engine.clone();
        engine
            .apply_staged_op(&[self, rhs], op)
            .expect("binary traced staging should preserve non-empty inputs")
            .into_iter()
            .next()
            .expect("binary traced staging should produce one output")
    }
}

impl<'engine, E: StagingEngine + ?Sized> Clone for Tracer<'engine, E> {
    fn clone(&self) -> Self {
        Self { state: self.state.clone(), engine: self.engine.clone() }
    }
}

impl<'engine, E: StagingEngine + ?Sized> std::fmt::Debug for Tracer<'engine, E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Tracer").field("state", &self.state).finish_non_exhaustive()
    }
}

impl<'engine, E> Display for Tracer<'engine, E>
where
    E: StagingEngine + ?Sized,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.state {
            TracerState::Live(atom, _) => Display::fmt(atom, formatter),
            TracerState::Poison(r#type) => write!(formatter, "<poison:{type}>"),
        }
    }
}

impl<'engine, E: StagingEngine + ?Sized> Typed<E::Type> for Tracer<'engine, E> {
    #[inline]
    fn r#type(&self) -> Cow<'_, E::Type> {
        Cow::Borrowed(self.state.r#type())
    }
}

impl<'engine, E> Traceable<E::Type> for Tracer<'engine, E> where E: StagingEngine + ?Sized {}

impl<'engine, E: StagingEngine + ?Sized> ZeroLike for Tracer<'engine, E> {
    #[inline]
    fn zero_like(&self) -> Self {
        let r#type = self.r#type().into_owned();
        let value = match self.outer_engine().zero(&r#type) {
            Ok(value) => value,
            Err(error) => {
                if self.builder().borrow().error.is_none() {
                    self.builder().borrow_mut().error = Some(error);
                }
                return self.engine.poisoned_tracer(r#type);
            }
        };
        let atom = self.builder().borrow_mut().add_constant(value);
        self.engine.tracer_from_staged_parts(atom, r#type)
    }
}

impl<'engine, E: StagingEngine + ?Sized> OneLike for Tracer<'engine, E> {
    #[inline]
    fn one_like(&self) -> Self {
        let r#type = self.r#type().into_owned();
        let value = match self.outer_engine().one(&r#type) {
            Ok(value) => value,
            Err(error) => {
                if self.builder().borrow().error.is_none() {
                    self.builder().borrow_mut().error = Some(error);
                }
                return self.engine.poisoned_tracer(r#type);
            }
        };
        let atom = self.builder().borrow_mut().add_constant(value);
        self.engine.tracer_from_staged_parts(atom, r#type)
    }
}

impl<'engine, E: StagingEngine + ?Sized> Add for Tracer<'engine, E>
where
    E::Operation: SupportsAdd<E::Type, E::Value>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.binary(rhs, E::Operation::add_operation())
    }
}

impl<'engine, E: StagingEngine + ?Sized> Mul for Tracer<'engine, E>
where
    E::Operation: SupportsMul<E::Type, E::Value>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.binary(rhs, E::Operation::mul_operation())
    }
}

impl<'engine, E: StagingEngine + ?Sized> Neg for Tracer<'engine, E>
where
    E::Operation: SupportsNeg<E::Type, E::Value>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        self.unary(E::Operation::neg_operation())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;
    use crate::tracing_v2::differentiation::DifferentiableEngine;
    use crate::tracing_v2::{Sin, jvp, test_support};
    use crate::types::{DataType, Shape, Size, TypeError};

    use super::*;

    type F64ProgramBuilder = ProgramBuilder<ArrayType, f64, PrimitiveOperation<f64>>;

    fn assert_differentiable_engine<E: DifferentiableEngine>() {}

    fn new_f64_builder() -> Rc<RefCell<F64ProgramBuilder>> {
        Rc::new(RefCell::new(ProgramBuilder::new()))
    }

    fn scalar_f64_type() -> ArrayType {
        ArrayType::scalar(DataType::F64)
    }

    struct TaggedEngine {
        id: u8,
    }

    impl Engine for TaggedEngine {
        type Type = ArrayType;
        type Value = f64;

        fn zero(&self, _type: &ArrayType) -> Result<f64, TracingError> {
            let _ = self.id;
            Ok(0.0)
        }

        fn one(&self, _type: &ArrayType) -> Result<f64, TracingError> {
            let _ = self.id;
            Ok(1.0)
        }
    }

    impl StagingEngine for TaggedEngine {
        type Operation = PrimitiveOperation<f64>;
    }

    #[test]
    fn test_array_scalar_engine_is_zero_sized() {
        assert_eq!(size_of::<ScalarEngine<bool>>(), 0);
        assert_eq!(size_of::<ScalarEngine<i8>>(), 0);
        assert_eq!(size_of::<ScalarEngine<u64>>(), 0);
        assert_eq!(size_of::<ScalarEngine<bf16>>(), 0);
        assert_eq!(size_of::<ScalarEngine<f16>>(), 0);
        assert_eq!(size_of::<ScalarEngine<f64>>(), 0);
        assert_eq!(size_of::<ScalarEngine<f32>>(), 0);
    }

    #[test]
    fn test_array_scalar_engine_produces_canonical_zero_and_one() {
        let bool_type = ArrayType::scalar(DataType::Boolean);
        let bool_engine = ScalarEngine::<bool>::new();
        assert_eq!(Engine::zero(&bool_engine, &bool_type), Ok(false));
        assert_eq!(Engine::one(&bool_engine, &bool_type), Ok(true));

        let i32_type = ArrayType::scalar(DataType::I32);
        let i32_engine = ScalarEngine::<i32>::new();
        assert_eq!(Engine::zero(&i32_engine, &i32_type), Ok(0i32));
        assert_eq!(Engine::one(&i32_engine, &i32_type), Ok(1i32));

        let u64_type = ArrayType::scalar(DataType::U64);
        let u64_engine = ScalarEngine::<u64>::new();
        assert_eq!(Engine::zero(&u64_engine, &u64_type), Ok(0u64));
        assert_eq!(Engine::one(&u64_engine, &u64_type), Ok(1u64));

        let bf16_type = ArrayType::scalar(DataType::BF16);
        let bf16_engine = ScalarEngine::<bf16>::new();
        assert_eq!(Engine::zero(&bf16_engine, &bf16_type), Ok(bf16::ZERO));
        assert_eq!(Engine::one(&bf16_engine, &bf16_type), Ok(bf16::ONE));

        let f16_type = ArrayType::scalar(DataType::F16);
        let f16_engine = ScalarEngine::<f16>::new();
        assert_eq!(Engine::zero(&f16_engine, &f16_type), Ok(f16::ZERO));
        assert_eq!(Engine::one(&f16_engine, &f16_type), Ok(f16::ONE));

        let f32_type = ArrayType::scalar(DataType::F32);
        let f32_engine = ScalarEngine::<f32>::new();
        assert_eq!(Engine::zero(&f32_engine, &f32_type), Ok(0.0f32));
        assert_eq!(Engine::one(&f32_engine, &f32_type), Ok(1.0f32));

        let f64_type = ArrayType::scalar(DataType::F64);
        let f64_engine = ScalarEngine::<f64>::new();
        assert_eq!(Engine::zero(&f64_engine, &f64_type), Ok(0.0f64));
        assert_eq!(Engine::one(&f64_engine, &f64_type), Ok(1.0f64));
    }

    #[test]
    fn test_half_and_float_scalar_engines_are_differentiable() {
        assert_differentiable_engine::<ScalarEngine<bf16>>();
        assert_differentiable_engine::<ScalarEngine<f16>>();
        assert_differentiable_engine::<ScalarEngine<f32>>();
        assert_differentiable_engine::<ScalarEngine<f64>>();
    }

    #[test]
    fn test_half_scalar_engines_run_jvp() {
        let bf16_engine = ScalarEngine::<bf16>::new();
        assert_eq!(
            jvp(&bf16_engine, |x| x.clone() + x, bf16::from_f32(3.0), bf16::ONE),
            Ok((bf16::from_f32(6.0), bf16::from_f32(2.0)))
        );

        let f16_engine = ScalarEngine::<f16>::new();
        assert_eq!(
            jvp(&f16_engine, |x| x.clone() + x, f16::from_f32(3.0), f16::ONE),
            Ok((f16::from_f32(6.0), f16::from_f32(2.0)))
        );
    }

    #[test]
    fn test_tracing_engine_zero_and_one_lift_constant_atoms() {
        let builder = new_f64_builder();
        let engine = ScalarEngine::<f64>::new();
        let tracing_engine = TracingEngine::new(&engine, builder.clone());
        let r#type = scalar_f64_type();

        let zero = Engine::zero(&tracing_engine, &r#type).unwrap();
        let one = Engine::one(&tracing_engine, &r#type).unwrap();

        assert_eq!(zero.r#type().into_owned(), r#type);
        assert_eq!(one.r#type().into_owned(), scalar_f64_type());
        let zero_atom = zero.atom_id().expect("zero tracer should remain live");
        let one_atom = one.atom_id().expect("one tracer should remain live");
        assert_eq!(zero_atom.index, 0);
        assert_eq!(one_atom.index, 1);

        let program = builder
            .borrow()
            .clone()
            .build::<Vec<f64>, Vec<f64>>(
                vec![zero_atom, one_atom],
                Vec::<Placeholder>::new(),
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        assert_eq!(program.interpret(Vec::new()).unwrap(), vec![0.0, 1.0]);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:f64[] = const
                    %1:f64[] = const
                in (%0, %1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_tracer_zero_like_adds_constant_atoms() {
        let builder = new_f64_builder();
        let atom = builder.borrow_mut().add_input(3.0f64.r#type().into_owned());
        let engine = ScalarEngine::<f64>::new();
        let tracer: Tracer<ScalarEngine<f64>> = TracingEngine::new(&engine, builder).tracer_from_atom(atom);
        let zero = tracer.zero_like();
        assert_eq!(zero.r#type().into_owned(), scalar_f64_type());
        let zero_atom = zero.state().live_atom().expect("zero-like tracer should remain live");
        assert!(zero_atom > atom);

        let program = zero
            .builder()
            .borrow()
            .clone()
            .build::<f64, f64>(vec![zero_atom], Placeholder, Placeholder)
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = const
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_live_tracer_type_borrows_cached_type() {
        let builder = new_f64_builder();
        let input_type = scalar_f64_type();
        let atom = builder.borrow_mut().add_input(input_type.clone());
        let engine = ScalarEngine::<f64>::new();
        let tracer: Tracer<ScalarEngine<f64>> = Tracer::from_staged_parts(atom, input_type, builder, &engine);

        assert!(matches!(tracer.r#type(), Cow::Borrowed(r#type) if *r#type == scalar_f64_type()));
    }

    #[test]
    fn test_apply_staged_op_rejects_mismatched_program_builders() {
        let builder_a = new_f64_builder();
        let builder_b = new_f64_builder();
        let atom_a = builder_a.borrow_mut().add_input(1.0f64.r#type().into_owned());
        let atom_b = builder_b.borrow_mut().add_input(2.0f64.r#type().into_owned());
        let engine = TaggedEngine { id: 1 };
        let tracer_a = TracingEngine::new(&engine, builder_a.clone()).tracer_from_atom(atom_a);
        let tracer_b = TracingEngine::new(&engine, builder_b).tracer_from_atom(atom_b);

        assert!(matches!(
            TracingEngine::new(&engine, builder_a).apply_staged_op(&[tracer_a, tracer_b], PrimitiveOperation::Add),
            Err(TracingError::MismatchedProgramBuilders),
        ));
    }

    #[test]
    fn test_apply_staged_op_rejects_mismatched_engines() {
        let builder = new_f64_builder();
        let atom_a = builder.borrow_mut().add_input(1.0f64.r#type().into_owned());
        let atom_b = builder.borrow_mut().add_input(2.0f64.r#type().into_owned());
        let engine_a = TaggedEngine { id: 1 };
        let engine_b = TaggedEngine { id: 2 };
        let tracer_a = TracingEngine::new(&engine_a, builder.clone()).tracer_from_atom(atom_a);
        let tracer_b = TracingEngine::new(&engine_b, builder.clone()).tracer_from_atom(atom_b);

        assert!(matches!(
            TracingEngine::new(&engine_a, builder).apply_staged_op(&[tracer_a, tracer_b], PrimitiveOperation::Add),
            Err(TracingError::MismatchedEngines),
        ));
    }

    #[test]
    fn test_apply_staged_op_returns_poisoned_tracers_after_builder_failure() {
        let builder = new_f64_builder();
        let atom = builder.borrow_mut().add_input(1.0f64.r#type().into_owned());
        builder.borrow_mut().error = Some(TracingError::InvalidInputCount { expected: 1, got: 0 });
        let engine = TaggedEngine { id: 1 };
        let tracer = TracingEngine::new(&engine, builder.clone()).tracer_from_atom(atom);

        let outputs = TracingEngine::new(&engine, builder)
            .apply_staged_op(std::slice::from_ref(&tracer), PrimitiveOperation::Neg)
            .unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(matches!(
            outputs[0].state(),
            TracerState::Poison(output_type) if *output_type == scalar_f64_type()
        ));
    }

    #[test]
    fn test_apply_staged_op_caches_live_output_types() {
        let builder = new_f64_builder();
        let input_type = scalar_f64_type();
        let atom = builder.borrow_mut().add_input(input_type.clone());
        let engine = TaggedEngine { id: 1 };
        let tracer = Tracer::from_staged_parts(atom, input_type, builder.clone(), &engine);

        let outputs = TracingEngine::new(&engine, builder)
            .apply_staged_op(std::slice::from_ref(&tracer), PrimitiveOperation::Neg)
            .unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(matches!(outputs[0].state(), TracerState::Live(_, output_type) if *output_type == scalar_f64_type()));
        assert!(matches!(outputs[0].r#type(), Cow::Borrowed(r#type) if *r#type == scalar_f64_type()));
    }

    #[test]
    fn test_poisoned_tracer_atom_id_returns_poisoned_tracer_error() {
        let builder = new_f64_builder();
        let engine = TaggedEngine { id: 1 };
        let tracer = TracingEngine::new(&engine, builder).poisoned_tracer(scalar_f64_type());

        assert_eq!(tracer.atom_id(), Err(TracingError::PoisonedTracer));
        assert!(matches!(tracer.r#type(), Cow::Borrowed(r#type) if *r#type == scalar_f64_type()));
    }

    #[test]
    fn test_interpret_and_trace_replays_staged_graphs() {
        let engine = ScalarEngine::<f64>::new();
        let (output, program): (f64, Program<ArrayType, f64, PrimitiveOperation<f64>, f64, f64>) = engine
            .interpret_and_trace(
                |x: Tracer<ScalarEngine<f64>>| {
                    let squared = x.clone() * x.clone();
                    Ok(squared + x.sin())
                },
                2.0f64,
            )
            .unwrap();

        assert_eq!(output, 2.0f64 * 2.0f64 + 2.0f64.sin());
        assert_eq!(program.interpret(0.5f64).unwrap(), 0.5f64 * 0.5f64 + 0.5f64.sin());
        assert_eq!(program.input_ids.len(), 1);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = mul %0 %0
                    %2:f64[] = sin %0
                    %3:f64[] = add %1 %2
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_trace_stages_program_from_type_metadata() {
        let engine = ScalarEngine::<f64>::new();
        let input_type = scalar_f64_type();
        let (output_type, program): (ArrayType, Program<ArrayType, f64, PrimitiveOperation<f64>, f64, f64>) = engine
            .trace(
                |x: Tracer<ScalarEngine<f64>>| {
                    let squared = x.clone() * x.clone();
                    Ok(squared + x.one_like())
                },
                input_type.clone(),
            )
            .unwrap();

        assert_eq!(output_type, input_type);
        assert_eq!(program.interpret(3.0).unwrap(), 10.0);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = mul %0 %0
                    %2:f64[] = const
                    %3:f64[] = add %1 %2
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_trace_rejects_escaped_program_builder() {
        let engine = ScalarEngine::<f64>::new();
        let escaped_builder = Rc::new(RefCell::new(None));
        let result: Result<(ArrayType, Program<ArrayType, f64, PrimitiveOperation<f64>, f64, f64>), TracingError> =
            engine.trace(
                |x: Tracer<ScalarEngine<f64>>| {
                    *escaped_builder.borrow_mut() = Some(x.builder().clone());
                    Ok(x)
                },
                scalar_f64_type(),
            );

        assert!(matches!(result, Err(TracingError::EscapedProgramBuilder)));
    }

    #[test]
    fn test_tracer_display_and_debug_render_live_and_poisoned_states() {
        let builder = new_f64_builder();
        let atom = builder.borrow_mut().add_input(scalar_f64_type());
        let engine = ScalarEngine::<f64>::new();
        let live = TracingEngine::new(&engine, builder.clone()).tracer_from_atom(atom);
        let poison = TracingEngine::new(&engine, builder).poisoned_tracer(scalar_f64_type());

        assert_eq!(live.to_string(), "%0");
        assert_eq!(format!("{live:?}"), "Tracer { state: Live(AtomId { index: 0 }), .. }");
        assert_eq!(poison.to_string(), "<poison:f64[]>");
        assert_eq!(format!("{poison:?}"), "Tracer { state: Poison(..), .. }");
        assert_eq!(format!("{:?}", live.engine), "TracingEngine { .. }");
    }

    #[test]
    fn test_apply_staged_op_records_abstract_eval_error_and_returns_poisoned_output() {
        let builder = new_f64_builder();
        let lhs_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)]), None, None).unwrap();
        let rhs_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3)]), None, None).unwrap();
        let lhs_atom = builder.borrow_mut().add_input(lhs_type.clone());
        let rhs_atom = builder.borrow_mut().add_input(rhs_type);
        let engine = ScalarEngine::<f64>::new();
        let lhs = TracingEngine::new(&engine, builder.clone()).tracer_from_atom(lhs_atom);
        let rhs = TracingEngine::new(&engine, builder.clone()).tracer_from_atom(rhs_atom);

        let outputs = TracingEngine::new(&engine, builder.clone())
            .apply_staged_op(&[lhs, rhs], PrimitiveOperation::Add)
            .unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(matches!(outputs[0].state(), TracerState::Poison(output_type) if *output_type == lhs_type));
        assert!(matches!(
            builder.borrow().error.clone(),
            Some(TracingError::Type(TypeError { message }))
                if message == "add input types are not broadcast-compatible"
        ));
    }

    #[test]
    fn test_apply_staged_op_uses_input_type_when_poisoned_abstract_eval_fails() {
        let builder = new_f64_builder();
        let input_type = scalar_f64_type();
        let atom = builder.borrow_mut().add_input(input_type.clone());
        builder.borrow_mut().error = Some(TracingError::InvalidInputCount { expected: 1, got: 0 });
        let engine = TaggedEngine { id: 1 };
        let tracer = TracingEngine::new(&engine, builder.clone()).tracer_from_atom(atom);

        let outputs = TracingEngine::new(&engine, builder)
            .apply_staged_op(std::slice::from_ref(&tracer), PrimitiveOperation::Add)
            .unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(matches!(outputs[0].state(), TracerState::Poison(output_type) if *output_type == input_type));
    }

    #[test]
    fn test_tracer_one_like_records_engine_identity_error() {
        struct FailingOneEngine;

        impl Engine for FailingOneEngine {
            type Type = ArrayType;
            type Value = f64;

            fn zero(&self, _type: &ArrayType) -> Result<f64, TracingError> {
                Ok(0.0)
            }

            fn one(&self, _type: &ArrayType) -> Result<f64, TracingError> {
                Err(TypeError { message: "test engine cannot synthesize one".to_string() }.into())
            }
        }

        impl StagingEngine for FailingOneEngine {
            type Operation = PrimitiveOperation<f64>;
        }

        let engine = FailingOneEngine;

        assert!(matches!(
            StagingEngine::interpret_and_trace::<_, f64, f64>(
                &engine,
                |x: Tracer<FailingOneEngine>| Ok(x.one_like()),
                1.0f64,
            ),
            Err(TracingError::Type(TypeError { message })) if message == "test engine cannot synthesize one"
        ));
    }

    #[test]
    fn test_interpret_and_trace_supports_non_array_types() {
        use std::fmt;

        use ryft_macros::Parameter;

        #[derive(Clone, Debug, PartialEq, Eq)]
        struct TestType(&'static str);

        impl Type for TestType {
            fn is_compatible_with(&self, other: &Self) -> bool {
                self == other
            }
        }

        impl Parameter for TestType {}

        impl fmt::Display for TestType {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(self.0)
            }
        }

        #[derive(Clone, Debug, PartialEq, Eq, Parameter)]
        struct TestValue {
            r#type: TestType,
            value: i32,
        }

        impl TestValue {
            fn new(r#type: TestType, value: i32) -> Self {
                Self { r#type, value }
            }
        }

        impl Typed<TestType> for TestValue {
            fn r#type(&self) -> Cow<'_, TestType> {
                Cow::Borrowed(&self.r#type)
            }
        }

        impl fmt::Display for TestValue {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "{}", self.value)
            }
        }

        impl Traceable<TestType> for TestValue {}

        impl crate::tracing::Value<TestType> for TestValue {}

        impl Add for TestValue {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                assert_eq!(self.r#type, rhs.r#type);
                Self { r#type: self.r#type, value: self.value + rhs.value }
            }
        }

        #[derive(Clone, Debug)]
        struct TestAddOp;

        impl fmt::Display for TestAddOp {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("test_add")
            }
        }

        impl SupportsAdd<TestType, TestValue> for TestAddOp {
            fn add_operation() -> Self {
                Self
            }
        }

        impl Operation<TestType> for TestAddOp {
            fn name(&self) -> &'static str {
                "test_add"
            }

            fn infer_output_types(&self, input_types: &[TestType]) -> Result<Vec<TestType>, TypeError> {
                if input_types.len() != 2 {
                    return Err(TypeError {
                        message: format!("test_add expected 2 input types but got {}", input_types.len()),
                    });
                }
                if !input_types[0].is_compatible_with(&input_types[1]) {
                    return Err(TypeError { message: "test_add input types are incompatible".to_string() });
                }
                Ok(vec![input_types[0].clone()])
            }
        }

        impl InterpretableOperation<TestType, TestValue> for TestAddOp {
            fn interpret(&self, inputs: &[TestValue]) -> Result<Vec<TestValue>, TracingError> {
                if inputs.len() != 2 {
                    return Err(TracingError::InvalidInputCount { expected: 2, got: inputs.len() });
                }
                if !inputs[0].r#type.is_compatible_with(&inputs[1].r#type) {
                    return Err(TracingError::Type(TypeError {
                        message: "test_add input types are incompatible".to_string(),
                    }));
                }
                Ok(vec![inputs[0].clone() + inputs[1].clone()])
            }
        }

        struct TestEngine;

        impl Engine for TestEngine {
            type Type = TestType;
            type Value = TestValue;

            fn zero(&self, r#type: &TestType) -> Result<TestValue, TracingError> {
                Ok(TestValue::new(r#type.clone(), 0))
            }

            fn one(&self, r#type: &TestType) -> Result<TestValue, TracingError> {
                Ok(TestValue::new(r#type.clone(), 1))
            }
        }

        impl StagingEngine for TestEngine {
            type Operation = TestAddOp;
        }

        let scalar_type = TestType("test_scalar");
        let (output, program): (TestValue, Program<TestType, TestValue, TestAddOp, (TestValue, TestValue), TestValue>) =
            TestEngine
                .interpret_and_trace(
                    |inputs: (Tracer<TestEngine>, Tracer<TestEngine>)| {
                        let sum = inputs.0.clone() + inputs.1;
                        let stabilized = sum + inputs.0.zero_like();
                        Ok(stabilized + inputs.0.one_like())
                    },
                    (TestValue::new(scalar_type.clone(), 2), TestValue::new(scalar_type.clone(), 3)),
                )
                .unwrap();

        assert_eq!(output, TestValue::new(scalar_type.clone(), 6));
        assert_eq!(
            program
                .interpret((TestValue::new(scalar_type.clone(), 4), TestValue::new(scalar_type.clone(), 5)))
                .unwrap(),
            TestValue::new(scalar_type, 10),
        );
    }

    #[test]
    fn test_interpret_and_trace_returns_abstract_eval_errors_instead_of_panicking() {
        use ryft_macros::Parameter;

        use crate::tracing::TracingError;
        use crate::tracing_v2::operations::constants::{OneLike, ZeroLike};
        use crate::tracing_v2::operations::reshape::ReshapeOps;
        use crate::tracing_v2::operations::{ControlFlowError, ControlFlowValue};
        use crate::tracing_v2::{Cos, MatrixOps, Sin};
        use crate::types::{ArrayType, DataType, Shape, Size, TypeError, Typed};

        #[derive(Clone, Debug, Parameter)]
        struct TestAbstractValue {
            r#type: ArrayType,
        }

        impl Typed<ArrayType> for TestAbstractValue {
            fn r#type(&self) -> Cow<'_, ArrayType> {
                Cow::Borrowed(&self.r#type)
            }
        }

        impl std::fmt::Display for TestAbstractValue {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Display::fmt(&self.r#type, formatter)
            }
        }

        impl Traceable<ArrayType> for TestAbstractValue {}

        impl crate::tracing::Value<ArrayType> for TestAbstractValue {}

        impl ControlFlowValue for TestAbstractValue {
            fn control_flow_predicate(&self) -> Result<bool, TracingError> {
                Err(ControlFlowError::InvalidPredicateValue { type_: self.r#type().into_owned() }.into())
            }
        }

        impl Add for TestAbstractValue {
            type Output = Self;

            fn add(self, _rhs: Self) -> Self::Output {
                self
            }
        }

        impl Mul for TestAbstractValue {
            type Output = Self;

            fn mul(self, _rhs: Self) -> Self::Output {
                self
            }
        }

        impl Neg for TestAbstractValue {
            type Output = Self;

            fn neg(self) -> Self::Output {
                self
            }
        }

        impl Sin for TestAbstractValue {
            fn sin(self) -> Self {
                self
            }
        }

        impl Cos for TestAbstractValue {
            fn cos(self) -> Self {
                self
            }
        }

        impl ZeroLike for TestAbstractValue {
            fn zero_like(&self) -> Self {
                self.clone()
            }
        }

        impl OneLike for TestAbstractValue {
            fn one_like(&self) -> Self {
                self.clone()
            }
        }

        impl MatrixOps for TestAbstractValue {
            fn matmul(self, _rhs: Self) -> Self {
                self
            }

            fn transpose_matrix(self) -> Self {
                self
            }
        }

        impl ReshapeOps for TestAbstractValue {
            fn reshape(self, _target_shape: crate::types::Shape) -> Result<Self, TracingError> {
                Ok(self)
            }
        }

        struct TestEngine;

        impl crate::tracing_v2::engines::Engine for TestEngine {
            type Type = ArrayType;
            type Value = TestAbstractValue;

            fn zero(&self, r#type: &ArrayType) -> Result<TestAbstractValue, TracingError> {
                Ok(TestAbstractValue { r#type: r#type.clone() })
            }

            fn one(&self, r#type: &ArrayType) -> Result<TestAbstractValue, TracingError> {
                Ok(TestAbstractValue { r#type: r#type.clone() })
            }
        }

        impl crate::tracing_v2::engines::StagingEngine for TestEngine {
            type Operation = crate::tracing_v2::PrimitiveOperation<TestAbstractValue>;
        }

        let result: Result<
            (
                TestAbstractValue,
                Program<
                    ArrayType,
                    TestAbstractValue,
                    crate::tracing_v2::PrimitiveOperation<TestAbstractValue>,
                    (TestAbstractValue, TestAbstractValue),
                    TestAbstractValue,
                >,
            ),
            TracingError,
        > = TestEngine.interpret_and_trace(
            |inputs: (Tracer<TestEngine>, Tracer<TestEngine>)| Ok(inputs.0 + inputs.1),
            (
                TestAbstractValue {
                    r#type: ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)]), None, None).unwrap(),
                },
                TestAbstractValue {
                    r#type: ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3)]), None, None).unwrap(),
                },
            ),
        );

        assert!(matches!(
            result,
            Err(TracingError::Type(TypeError { message }))
                if message == "add input types are not broadcast-compatible"
        ));
    }

    #[test]
    fn test_staged_program_display_renders_the_staged_program() {
        let engine = ScalarEngine::<f64>::new();
        let (_, compiled): (f64, Program<ArrayType, f64, PrimitiveOperation<f64>, f64, f64>) = engine
            .interpret_and_trace(|x: Tracer<ScalarEngine<f64>>| Ok(x.clone() * x.clone() + x.sin()), 2.0f64)
            .unwrap();

        assert_eq!(
            compiled.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = mul %0 %0
                    %2:f64[] = sin %0
                    %3:f64[] = add %1 %2
                in (%3)
            "}
            .trim_end(),
        );
        test_support::assert_bilinear_jit_rendering();
    }
}
