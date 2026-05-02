use std::borrow::Cow;
use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::rc::Rc;

use half::{bf16, f16};

use ryft_macros::Parameter;

use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::{Parameter, Parameterized, ParameterizedFamily as ParameterFamily};
use crate::tracing::{AtomId, Program, ProgramBuilder, Traceable, TracingError};
use crate::tracing_v2::operations::primitive::ScalarOperation;
use crate::types::{DataType, Type, TypeError, Typed};

/// [`Engine`]s provide backend-specific functionality related to tracing, just-in-time compilation, automatic
/// differentiation, and potentially other [`Program`] transforms. They also define the kinds of [`Type`]s and
/// [`Traceable`] values that each backend supports and they are effectively what lets higher-order transforms
/// remain backend-invariant.
pub trait Engine {
    /// [`Type`]s that this [`Engine`] uses to represent the abstract metadata associated with its [`Traceable`] values.
    /// A commonly used [`Type`] is [`ArrayType`](crate::ArrayType), though scalar-only engines can use
    /// [`DataType`] and richer backends may use richer metadata.
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

/// Extension of [`Engine`] for backends that can trace staged [`Program`]s. A [`TracingEngine`] selects an
/// [`Operation`] _carrier_: the concrete operation representation stored in each
/// [`Instruction`](crate::tracing::Instruction) of traced programs for this backend. Carriers
/// are usually closed enums whose variants wrap the primitive operations supported by the backend,
/// though simple engines may use one primitive operation type directly.
pub trait TracingEngine: Engine {
    /// [`Operation`] carrier selected by this [`TracingEngine`] for ordinary traced [`Program`]s.
    type OperationCarrier: Operation<Self::Type>;

    /// Traces the provided `function` into a [`Program`] for the provided input types, returning the output types of
    /// the traced [`Program`] along with that traced [`Program`] itself. This is the most symbolic tracing entry
    /// point in that it does not require concrete runtime input values but rather it only requires their types. The
    /// provided closure is executed once on [`Tracer`] values standing in for those input types to trace the function,
    /// and relies on [`Operation::infer_output_types`] for inferring output types.
    #[inline]
    fn trace<
        'engine,
        F: FnOnce(I::To<Tracer<'engine, Self>>) -> Result<O, TracingError>,
        I: Parameterized<Self::Type, Family: ParameterFamily<Self::Value> + ParameterFamily<Tracer<'engine, Self>>>,
        O: Parameterized<Tracer<'engine, Self>, Family: ParameterFamily<Self::Type> + ParameterFamily<Self::Value>>,
    >(
        &'engine self,
        function: F,
        input_types: I,
    ) -> Result<
        (
            O::To<Self::Type>,
            Program<Self::Type, Self::Value, Self::OperationCarrier, I::To<Self::Value>, O::To<Self::Value>>,
        ),
        TracingError,
    > {
        let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        let input_structure = input_types.parameter_structure();
        let input = input_types
            .map_parameters(|r#type| TracingContext::new(self, builder.clone()).input(r#type))
            .map_err(TracingError::from)?;
        let output = function(input).map_err(|error| match builder.borrow_mut().error.take() {
            Some(error) => error,
            None => error,
        })?;
        let _ = builder.borrow_mut().error.take().map_or(Ok(()), Err)?;
        let output_structure = output.parameter_structure();
        let outputs = output.parameters().map(|output| output.atom_id()).collect::<Result<Vec<_>, _>>()?;
        let output_types = output.map_parameters(|output| output.r#type().into_owned()).map_err(TracingError::from)?;
        let builder = Rc::try_unwrap(builder).map_err(|_| TracingError::EscapedProgramBuilder)?.into_inner();
        let program = builder.build(outputs, input_structure, output_structure)?;
        Ok((output_types, program))
    }

    /// Traces the provided `function` into a [`Program`] using the provided input values and also interprets it using
    /// those same input values, returning the output values along with the traced [`Program`]. This function should be
    /// used instead of [`TracingEngine::trace`] when the caller wants to both trace a computation and execute it at the
    /// same time.
    fn interpret_and_trace<
        'engine,
        F: FnOnce(I::To<Tracer<'engine, Self>>) -> Result<O, TracingError>,
        I: Parameterized<
                Self::Value,
                Family: ParameterFamily<Tracer<'engine, Self>>,
                ParameterStructure: Debug + PartialEq,
            >,
        O: Parameterized<Tracer<'engine, Self>, Family: ParameterFamily<Self::Value>>,
    >(
        &'engine self,
        function: F,
        input: I,
    ) -> Result<
        (O::To<Self::Value>, Program<Self::Type, Self::Value, Self::OperationCarrier, I, O::To<Self::Value>>),
        TracingError,
    >
    where
        Self::OperationCarrier: Clone + InterpretableOperation<Self::Type, Self::Value>,
    {
        let input_structure = input.parameter_structure();
        let input_values = input.into_parameters().collect::<Vec<_>>();
        let input_types = input_values.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
        let mut output_structure = None;
        let (_, flat_program) = self.trace(
            |flat_input| {
                let input = I::To::<Tracer<'engine, Self>>::from_parameters(input_structure.clone(), flat_input)?;
                let output = function(input)?;
                output_structure = Some(output.parameter_structure());
                Ok(output.into_parameters().collect::<Vec<_>>())
            },
            input_types,
        )?;
        let output_structure = output_structure.expect("the function being traced should have been invoked");
        let flat_program = flat_program.into_simplified()?;
        let output =
            O::To::<Self::Value>::from_parameters(output_structure.clone(), flat_program.interpret(input_values)?)?;
        let program = Program {
            atoms: flat_program.atoms,
            input_ids: flat_program.input_ids,
            output_ids: flat_program.output_ids,
            instructions: flat_program.instructions,
            input_structure,
            output_structure,
            marker: PhantomData,
        };
        Ok((output, program))
    }
}

/// Stateless [`TracingEngine`] that uses [`DataType`] for scalar metadata and Rust scalar values such as `f32` for
/// runtime values. [`ScalarEngine`] is the minimal scalar-only backend used throughout tests and examples in
/// `ryft-core`. It demonstrates the intended role of an [`Engine`] in the smallest possible form: there are no device
/// handles, no mesh states, and no backend registries; just the built-in [`ScalarOperation`] carriers plus
/// [`DataType`]-driven construction of scalar values.
#[derive(Copy, Clone, Debug, Default)]
pub struct ScalarEngine<V> {
    /// Phantom marker that ties this zero-sized [`ScalarEngine`] to its scalar value type.
    marker: PhantomData<fn() -> V>,
}

impl<V> ScalarEngine<V> {
    /// Creates a new [`ScalarEngine`].
    #[inline]
    pub const fn new() -> Self {
        Self { marker: PhantomData }
    }
}

macro_rules! impl_tracing_engine_for_scalar {
    ($ty:ty, $data_type:path, $zero:expr, $one:expr) => {
        impl Engine for ScalarEngine<$ty> {
            type Type = DataType;
            type Value = $ty;

            #[inline]
            fn zero(&self, r#type: &DataType) -> Result<$ty, TracingError> {
                if *r#type != $data_type {
                    return Err(TypeError {
                        message: format!("scalar engine for {} cannot synthesize zero for {}", $data_type, r#type),
                    }
                    .into());
                }
                Ok($zero)
            }

            #[inline]
            fn one(&self, r#type: &DataType) -> Result<$ty, TracingError> {
                if *r#type != $data_type {
                    return Err(TypeError {
                        message: format!("scalar engine for {} cannot synthesize one for {}", $data_type, r#type),
                    }
                    .into());
                }
                Ok($one)
            }
        }

        impl TracingEngine for ScalarEngine<$ty> {
            type OperationCarrier = ScalarOperation<$ty>;
        }
    };
}

impl_tracing_engine_for_scalar!(bool, DataType::Boolean, false, true);
impl_tracing_engine_for_scalar!(i8, DataType::I8, 0i8, 1i8);
impl_tracing_engine_for_scalar!(i16, DataType::I16, 0i16, 1i16);
impl_tracing_engine_for_scalar!(i32, DataType::I32, 0i32, 1i32);
impl_tracing_engine_for_scalar!(i64, DataType::I64, 0i64, 1i64);
impl_tracing_engine_for_scalar!(u8, DataType::U8, 0u8, 1u8);
impl_tracing_engine_for_scalar!(u16, DataType::U16, 0u16, 1u16);
impl_tracing_engine_for_scalar!(u32, DataType::U32, 0u32, 1u32);
impl_tracing_engine_for_scalar!(u64, DataType::U64, 0u64, 1u64);
impl_tracing_engine_for_scalar!(bf16, DataType::BF16, bf16::ZERO, bf16::ONE);
impl_tracing_engine_for_scalar!(f16, DataType::F16, f16::ZERO, f16::ONE);
impl_tracing_engine_for_scalar!(f32, DataType::F32, 0.0, 1.0);
impl_tracing_engine_for_scalar!(f64, DataType::F64, 0.0, 1.0);

/// Context that is used while _tracing_ [`Program`]s. This context bundles an underlying [`TracingEngine`]
/// with a [`ProgramBuilder`] and uses [`Tracer`]s to represent values.
pub struct TracingContext<'engine, E: TracingEngine + ?Sized> {
    /// [`TracingEngine`] borrowed by this [`TracingContext`] for type-driven value synthesis and operation selection.
    pub engine: &'engine E,

    /// [`ProgramBuilder`] that owns the staged [`Program`] that is currently being traced.
    pub builder: Rc<RefCell<ProgramBuilder<E::Type, E::Value, E::OperationCarrier>>>,
}

impl<'engine, E: TracingEngine + ?Sized> TracingContext<'engine, E> {
    /// Creates a new [`TracingContext`] that borrows the provided [`TracingEngine`].
    #[inline]
    pub fn new(
        engine: &'engine E,
        builder: Rc<RefCell<ProgramBuilder<E::Type, E::Value, E::OperationCarrier>>>,
    ) -> Self {
        Self { engine, builder }
    }

    /// Creates a constant [`Tracer`] in this [`TracingContext`] for the provided concrete value.
    #[inline]
    pub fn constant(&self, value: E::Value) -> Tracer<'engine, E> {
        let r#type = value.r#type().into_owned();
        let atom = self.builder.borrow_mut().add_constant(value);
        self.tracer(atom, Some(r#type))
    }

    /// Creates an input [`Tracer`] in this [`TracingContext`] for the provided type.
    #[inline]
    pub fn input(&self, r#type: E::Type) -> Tracer<'engine, E> {
        let atom = self.builder.borrow_mut().add_input(r#type.clone());
        self.tracer(atom, Some(r#type))
    }

    /// Constructs a [`TracerState::Live`] [`Tracer`] in this [`TracingContext`] for the provided [`AtomId`].
    /// If the provided `r#type` is [`None`], the staged [`Atom`]'s type is read from the owned [`ProgramBuilder`].
    #[inline]
    pub fn tracer(&self, atom: AtomId, r#type: Option<E::Type>) -> Tracer<'engine, E> {
        let r#type = r#type.unwrap_or_else(|| self.builder.borrow().atoms[atom.index].r#type().into_owned());
        Tracer { state: TracerState::Live(atom), r#type, context: self.clone() }
    }

    /// Records the provided [`TracingError`] in the underlying [`ProgramBuilder`] and returns it. If the underlying
    /// [`ProgramBuilder`] already has an error recorded, then it is left unchanged and this function acts simply as
    /// an identity function.
    #[inline]
    pub fn error(&self, error: TracingError) -> TracingError {
        let mut builder = self.builder.borrow_mut();
        if builder.error.is_none() {
            builder.error = Some(error.clone());
        }
        error
    }

    /// Traces one [`Operation`] application in this [`TracingContext`] and returns [`Tracer`]s for its outputs.
    /// This function validates that all provided `inputs` belong to this [`TracingContext`] and then delegates normal
    /// [`Instruction`](crate::tracing::Instruction) construction to [`ProgramBuilder::add_instruction`]. If the builder
    /// has already recorded an error, this function avoids mutating the partial [`Program`] and only uses type
    /// inference to synthesize poisoned output [`Tracer`]s with the expected types.
    pub fn trace(
        &self,
        operation: E::OperationCarrier,
        inputs: &[&Tracer<'engine, E>],
    ) -> Result<Vec<Tracer<'engine, E>>, TracingError> {
        if inputs.iter().any(|input| !Rc::ptr_eq(&self.builder, &input.context.builder)) {
            return Err(self.error(TracingError::MismatchedProgramBuilders));
        }
        if self.builder.borrow().error.is_some() {
            let input_types = inputs.iter().map(|input| input.r#type.clone()).collect::<Vec<_>>();
            let output_types = operation.infer_output_types(input_types.as_slice())?;
            Ok(output_types
                .into_iter()
                .map(|r#type| Tracer { state: TracerState::Poison, r#type, context: self.clone() })
                .collect())
        } else {
            let input_atom_ids = match inputs.iter().map(|input| input.atom_id()).collect::<Result<Vec<_>, _>>() {
                Ok(input_atom_ids) => input_atom_ids,
                Err(error) => return Err(self.error(error)),
            };
            let output_atom_ids = {
                let mut builder = self.builder.borrow_mut();
                match builder.add_instruction(operation, input_atom_ids) {
                    Ok(outputs) => outputs.to_vec(),
                    Err(error) => {
                        if builder.error.is_none() {
                            builder.error = Some(error.clone());
                        }
                        return Err(error);
                    }
                }
            };
            Ok(output_atom_ids.into_iter().map(|atom| self.tracer(atom, None)).collect::<Vec<_>>())
        }
    }
}

impl<'engine, E: TracingEngine + ?Sized> Clone for TracingContext<'engine, E> {
    fn clone(&self) -> Self {
        Self { engine: self.engine, builder: self.builder.clone() }
    }
}

impl<'engine, E: TracingEngine + ?Sized> Debug for TracingContext<'engine, E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("TracingContext").finish_non_exhaustive()
    }
}

impl<'engine, E: TracingEngine + ?Sized> Engine for TracingContext<'engine, E> {
    type Type = E::Type;
    type Value = Tracer<'engine, E>;

    #[inline]
    fn zero(&self, r#type: &Self::Type) -> Result<Self::Value, TracingError> {
        Ok(self.constant(self.engine.zero(r#type)?))
    }

    #[inline]
    fn one(&self, r#type: &Self::Type) -> Result<Self::Value, TracingError> {
        Ok(self.constant(self.engine.one(r#type)?))
    }
}

/// State carried by a [`Tracer`] that indicates whether this tracer is _live_ and has a corresponding
/// [`Atom`](crate::tracing::Atom) or _poisoned_, meaning that it corresponds to an error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TracerState {
    /// The corresponding [`Tracer`] is _live_ and has a corresponding [`Atom`](crate::tracing::Atom).
    Live(AtomId),

    /// The corresponding [`Tracer`] has been _poisoned_, meaning that it corresponds to an error, and will propagate
    /// that error wherever it is used (i.e., it will _poison_ those corresponding downstream [`Tracer`]s too).
    Poison,
}

/// Value used for tracing [`Program`]s, substituting actual runtime values and recording the executed [`Operation`]s
/// via its [`TracingContext`]. Trait implementations on [`Tracer`]s stage [`Instruction`]s in a shared
/// [`ProgramBuilder`] instead of executing those instructions, and return new [`Tracer`]s for the staged outputs.
/// When tracing fails, later operations return _poisoned_ tracers which are represented using [`TracerState::Poison`].
#[derive(Parameter)]
pub struct Tracer<'engine, E: TracingEngine + ?Sized> {
    /// [`TracerState`] of this [`Tracer`].
    pub state: TracerState,

    /// [`Type`] of the value that this [`Tracer`] represents.
    pub r#type: E::Type,

    /// [`TracingContext`] associated with this [`Tracer`] that owns the underlying shared [`ProgramBuilder`].
    pub context: TracingContext<'engine, E>,
}

impl<'engine, E: TracingEngine + ?Sized> Tracer<'engine, E> {
    /// Returns the [`TracingEngine`] associated with this [`Tracer`].
    #[inline]
    pub fn engine(&self) -> &'engine E {
        self.context.engine
    }

    /// Returns the [`ProgramBuilder`] associated with this [`Tracer`].
    #[inline]
    pub fn builder(&self) -> &Rc<RefCell<ProgramBuilder<E::Type, E::Value, E::OperationCarrier>>> {
        &self.context.builder
    }

    /// Returns the staged [`AtomId`] for this [`Tracer`] if it is _live_,
    /// and [`TracingError::PoisonedTracer`] otherwise.
    #[inline]
    pub fn atom_id(&self) -> Result<AtomId, TracingError> {
        match &self.state {
            TracerState::Live(atom) => Ok(*atom),
            TracerState::Poison => Err(TracingError::PoisonedTracer),
        }
    }

    /// Applies the provided _unary_ [`Operation`] to this [`Tracer`] returning the resulting [`Tracer`].
    /// _Unary_ operations are operations that have a single input and a single output. If the provided operation is not
    /// a unary operation then the resulting [`Tracer`] will contain a [`TracerState::Poison`].
    pub fn unary(self, operation: E::OperationCarrier) -> Self {
        match self.context.trace(operation, &[&self]) {
            Ok(mut outputs) if outputs.len() == 1 => outputs.remove(0),
            Ok(outputs) => {
                self.context.error(TracingError::InvalidOutputCount { expected: 1, got: outputs.len() });
                Tracer { state: TracerState::Poison, r#type: self.r#type.clone(), context: self.context.clone() }
            }
            Err(error) => {
                self.context.error(error);
                Tracer { state: TracerState::Poison, r#type: self.r#type.clone(), context: self.context.clone() }
            }
        }
    }

    /// Applies the provided _binary_ [`Operation`] to this [`Tracer`] and the provided [`Tracer`] returning the
    /// resulting [`Tracer`]. _Binary_ operations are operations that have two inputs and a single output. If the
    /// provided operation is not a binary operation then the resulting [`Tracer`] will contain a
    /// [`TracerState::Poison`].
    pub fn binary(self, rhs: Self, operation: E::OperationCarrier) -> Self {
        match self.context.trace(operation, &[&self, &rhs]) {
            Ok(mut outputs) if outputs.len() == 1 => outputs.remove(0),
            Ok(outputs) => {
                self.context.error(TracingError::InvalidOutputCount { expected: 1, got: outputs.len() });
                Tracer { state: TracerState::Poison, r#type: self.r#type.clone(), context: self.context.clone() }
            }
            Err(error) => {
                self.context.error(error);
                Tracer { state: TracerState::Poison, r#type: self.r#type.clone(), context: self.context.clone() }
            }
        }
    }
}

impl<'engine, E: TracingEngine + ?Sized> Clone for Tracer<'engine, E> {
    fn clone(&self) -> Self {
        Self { state: self.state.clone(), r#type: self.r#type.clone(), context: self.context.clone() }
    }
}

impl<'engine, E: TracingEngine<Type: Debug> + ?Sized> Debug for Tracer<'engine, E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Tracer")
            .field("state", &self.state)
            .field("type", &self.r#type)
            .finish_non_exhaustive()
    }
}

impl<'engine, E: TracingEngine + ?Sized> Display for Tracer<'engine, E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.state {
            TracerState::Live(atom_id) => write!(formatter, "{atom_id}"),
            TracerState::Poison => write!(formatter, "<poison:{}>", self.r#type),
        }
    }
}

impl<'engine, E: TracingEngine + ?Sized> Typed<E::Type> for Tracer<'engine, E> {
    #[inline]
    fn r#type(&self) -> Cow<'_, E::Type> {
        Cow::Borrowed(&self.r#type)
    }
}

impl<'engine, E: TracingEngine + ?Sized> Traceable<E::Type> for Tracer<'engine, E> {}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::operations::constants::{OneLike, ZeroLike};
    use crate::parameters::Placeholder;
    use crate::tracing_v2::operations::sin::Sin;
    use crate::types::{DataType, TypeError, Typed};

    use super::*;

    #[test]
    fn test_engine() {
        let bool_type = DataType::Boolean;
        let bool_engine = ScalarEngine::<bool>::new();
        assert_eq!(Engine::zero(&bool_engine, &bool_type), Ok(false));
        assert_eq!(Engine::one(&bool_engine, &bool_type), Ok(true));

        let i8_type = DataType::I8;
        let i8_engine = ScalarEngine::<i8>::new();
        assert_eq!(Engine::zero(&i8_engine, &i8_type), Ok(0i8));
        assert_eq!(Engine::one(&i8_engine, &i8_type), Ok(1i8));

        let i16_type = DataType::I16;
        let i16_engine = ScalarEngine::<i16>::new();
        assert_eq!(Engine::zero(&i16_engine, &i16_type), Ok(0i16));
        assert_eq!(Engine::one(&i16_engine, &i16_type), Ok(1i16));

        let i32_type = DataType::I32;
        let i32_engine = ScalarEngine::<i32>::new();
        assert_eq!(Engine::zero(&i32_engine, &i32_type), Ok(0i32));
        assert_eq!(Engine::one(&i32_engine, &i32_type), Ok(1i32));

        let i64_type = DataType::I64;
        let i64_engine = ScalarEngine::<i64>::new();
        assert_eq!(Engine::zero(&i64_engine, &i64_type), Ok(0i64));
        assert_eq!(Engine::one(&i64_engine, &i64_type), Ok(1i64));

        let u8_type = DataType::U8;
        let u8_engine = ScalarEngine::<u8>::new();
        assert_eq!(Engine::zero(&u8_engine, &u8_type), Ok(0u8));
        assert_eq!(Engine::one(&u8_engine, &u8_type), Ok(1u8));

        let u16_type = DataType::U16;
        let u16_engine = ScalarEngine::<u16>::new();
        assert_eq!(Engine::zero(&u16_engine, &u16_type), Ok(0u16));
        assert_eq!(Engine::one(&u16_engine, &u16_type), Ok(1u16));

        let u32_type = DataType::U32;
        let u32_engine = ScalarEngine::<u32>::new();
        assert_eq!(Engine::zero(&u32_engine, &u32_type), Ok(0u32));
        assert_eq!(Engine::one(&u32_engine, &u32_type), Ok(1u32));

        let u64_type = DataType::U64;
        let u64_engine = ScalarEngine::<u64>::new();
        assert_eq!(Engine::zero(&u64_engine, &u64_type), Ok(0u64));
        assert_eq!(Engine::one(&u64_engine, &u64_type), Ok(1u64));

        let bf16_type = DataType::BF16;
        let bf16_engine = ScalarEngine::<bf16>::new();
        assert_eq!(Engine::zero(&bf16_engine, &bf16_type), Ok(bf16::ZERO));
        assert_eq!(Engine::one(&bf16_engine, &bf16_type), Ok(bf16::ONE));

        let f16_type = DataType::F16;
        let f16_engine = ScalarEngine::<f16>::new();
        assert_eq!(Engine::zero(&f16_engine, &f16_type), Ok(f16::ZERO));
        assert_eq!(Engine::one(&f16_engine, &f16_type), Ok(f16::ONE));

        let f32_type = DataType::F32;
        let f32_engine = ScalarEngine::<f32>::new();
        assert_eq!(Engine::zero(&f32_engine, &f32_type), Ok(0.0f32));
        assert_eq!(Engine::one(&f32_engine, &f32_type), Ok(1.0f32));

        let f64_type = DataType::F64;
        let f64_engine = ScalarEngine::<f64>::new();
        assert_eq!(Engine::zero(&f64_engine, &f64_type), Ok(0.0f64));
        assert_eq!(Engine::one(&f64_engine, &f64_type), Ok(1.0f64));
        assert!(matches!(
            Engine::zero(&f64_engine, &DataType::F32),
            Err(TracingError::Type(TypeError { message }))
                if message == "scalar engine for f64 cannot synthesize zero for f32",
        ));
        assert!(matches!(
            Engine::one(&f64_engine, &DataType::F32),
            Err(TracingError::Type(TypeError { message }))
                if message == "scalar engine for f64 cannot synthesize one for f32",
        ));
    }

    #[test]
    fn test_tracing_engine_trace() {
        let engine = ScalarEngine::<f64>::new();
        let (output_type, program) = engine.trace(|x| Ok(x.clone() * x.clone() + x.one_like()), DataType::F64).unwrap();
        assert_eq!(output_type, DataType::F64);
        assert_eq!(program.interpret(3.0), Ok(10.0));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = mul %0 %0
                    %2:f64 = one_like %0
                    %3:f64 = add %1 %2
                in (%3)
            "}
            .trim_end(),
        );

        // Test using an escaped [`ProgramBuilder`].
        let escaped_builder = Rc::new(RefCell::new(None));
        assert!(matches!(
            engine.trace(
                |x| {
                    *escaped_builder.borrow_mut() = Some(x.builder().clone());
                    Ok(x)
                },
                DataType::F64,
            ),
            Err(TracingError::EscapedProgramBuilder),
        ));

        // Test that [`TypeError`]s are returned in certain cases.
        assert!(matches!(
            engine.trace(
                |inputs| Ok(inputs.0 + inputs.1),
                (DataType::F8E3M4, DataType::F32),
            ),
            Err(TracingError::Type(TypeError { message }))
                if message == "add input types are not broadcast-compatible",
        ));
    }

    #[test]
    fn test_tracing_engine_interpret_and_trace() {
        let engine = ScalarEngine::<f64>::new();
        let (output, program) = engine.interpret_and_trace(|x| Ok(x.clone() * x.clone() + x.sin()), 2.0f64).unwrap();
        assert_eq!(output, 2.0f64 * 2.0f64 + 2.0f64.sin());
        assert_eq!(program.interpret(0.5f64), Ok(0.5f64 * 0.5f64 + 0.5f64.sin()));
        assert_eq!(program.input_ids.len(), 1);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = mul %0 %0
                    %2:f64 = sin %0
                    %3:f64 = add %1 %2
                in (%3)
            "}
            .trim_end(),
        );

        // Test using a function with a tuple argument.
        let (_, compiled) = engine.interpret_and_trace(|(x, y)| Ok(x.clone() * y + x.sin()), (2.0f64, 3.0f64)).unwrap();
        assert_eq!(
            compiled.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = mul %0 %1
                    %3:f64 = sin %0
                    %4:f64 = add %2 %3
                in (%4)
            "}
            .trim_end(),
        );

        // Test using a function that contains unused code.
        let (output, program) = engine
            .interpret_and_trace(
                |x| {
                    let _ = x.clone().sin();
                    Ok(x.clone() * x)
                },
                2.0f64,
            )
            .unwrap();
        assert_eq!(output, 4.0);
        assert_eq!(program.interpret(0.5f64), Ok(0.25));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = mul %0 %0
                in (%1)
            "}
            .trim_end(),
        );

        // Test tracing value-level identity helpers as ordinary operations.
        let (output, program) = engine.interpret_and_trace(|x| Ok((x.zero_like(), x.one_like())), 2.0f64).unwrap();
        assert_eq!(output, (0.0, 1.0));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = zero_like %0
                    %2:f64 = one_like %0
                in (%1, %2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_scalar_engine() {
        // Check that [`ScalarEngine`] is zero-sized.
        assert_eq!(size_of::<ScalarEngine<bool>>(), 0);
        assert_eq!(size_of::<ScalarEngine<i8>>(), 0);
        assert_eq!(size_of::<ScalarEngine<i16>>(), 0);
        assert_eq!(size_of::<ScalarEngine<i32>>(), 0);
        assert_eq!(size_of::<ScalarEngine<i64>>(), 0);
        assert_eq!(size_of::<ScalarEngine<u8>>(), 0);
        assert_eq!(size_of::<ScalarEngine<u16>>(), 0);
        assert_eq!(size_of::<ScalarEngine<u32>>(), 0);
        assert_eq!(size_of::<ScalarEngine<u64>>(), 0);
        assert_eq!(size_of::<ScalarEngine<bf16>>(), 0);
        assert_eq!(size_of::<ScalarEngine<f16>>(), 0);
        assert_eq!(size_of::<ScalarEngine<f32>>(), 0);
        assert_eq!(size_of::<ScalarEngine<f64>>(), 0);

        // Check that [`ScalarEngine`] is an [`Engine`].
        assert_eq!(ScalarEngine::<f64>::new().zero(&DataType::F64), Ok(0.0));
        assert_eq!(ScalarEngine::<f64>::default().one(&DataType::F64), Ok(1.0));
    }

    #[test]
    fn test_tracing_context() {
        let engine = ScalarEngine::<f64>::new();

        // Test construction, cloning, and debug formatting.
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let tracing_context = TracingContext::new(&engine, builder.clone());
        let cloned_context = tracing_context.clone();
        assert!(std::ptr::eq(tracing_context.engine, &engine));
        assert!(Rc::ptr_eq(&tracing_context.builder, &builder));
        assert!(std::ptr::eq(cloned_context.engine, &engine));
        assert!(Rc::ptr_eq(&cloned_context.builder, &builder));
        assert_eq!(format!("{tracing_context:?}"), "TracingContext { .. }");

        // Test creating a concrete constant in the staged program.
        let constant = tracing_context.constant(2.5f64);
        assert_eq!(constant.r#type().into_owned(), DataType::F64);
        let constant_atom = constant.atom_id().expect("constant tracer should remain live");
        assert_eq!(constant_atom.index, 0);
        let program = builder
            .borrow()
            .clone()
            .build::<Vec<f64>, f64>(vec![constant_atom], Vec::<Placeholder>::new(), Placeholder)
            .unwrap();
        assert_eq!(program.interpret(Vec::new()), Ok(2.5));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:f64 = const
                in (%0)
            "}
            .trim_end(),
        );

        // Test constructing tracers from builder-owned and explicitly cached types.
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let atom = builder.borrow_mut().add_input(DataType::F64);
        let tracing_context = TracingContext::new(&engine, builder);
        let builder_typed = tracing_context.tracer(atom, None);
        let cached_typed = tracing_context.tracer(atom, Some(DataType::F64));
        assert!(matches!(builder_typed.r#type(), Cow::Borrowed(r#type) if *r#type == DataType::F64));
        assert!(matches!(cached_typed.r#type(), Cow::Borrowed(r#type) if *r#type == DataType::F64));

        // Test that only the first recorded builder error is retained.
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let tracing_context = TracingContext::new(&engine, builder.clone());
        let first_error = TracingError::InvalidInputCount { expected: 1, got: 0 };
        let second_error = TracingError::InvalidOutputCount { expected: 1, got: 0 };
        assert_eq!(tracing_context.error(first_error.clone()), first_error);
        assert_eq!(tracing_context.error(second_error), TracingError::InvalidOutputCount { expected: 1, got: 0 });
        assert_eq!(builder.borrow().error, Some(first_error));

        // Test staging a valid operation through the context.
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let lhs_atom = builder.borrow_mut().add_input(DataType::F64);
        let rhs_atom = builder.borrow_mut().add_input(DataType::F64);
        let tracing_context = TracingContext::new(&engine, builder.clone());
        let lhs = tracing_context.tracer(lhs_atom, None);
        let rhs = tracing_context.tracer(rhs_atom, None);
        let outputs = tracing_context.trace(ScalarOperation::Add, &[&lhs, &rhs]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(matches!(&outputs[0].state, TracerState::Live(AtomId { index: 2 })));
        assert_eq!(outputs[0].r#type().into_owned(), DataType::F64);
        let output_atom = outputs[0].atom_id().expect("output tracer should remain live");
        let program = builder
            .borrow()
            .clone()
            .build::<(f64, f64), f64>(vec![output_atom], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        assert_eq!(program.interpret((2.0, 3.0)), Ok(5.0));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = add %0 %1
                in (%2)
            "}
            .trim_end(),
        );

        // Test rejecting inputs that belong to a different program builder.
        let builder_a = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let builder_b = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let atom_a = builder_a.borrow_mut().add_input(DataType::F64);
        let atom_b = builder_b.borrow_mut().add_input(DataType::F64);
        let tracer_a = TracingContext::new(&engine, builder_a.clone()).tracer(atom_a, None);
        let tracer_b = TracingContext::new(&engine, builder_b).tracer(atom_b, None);
        assert!(matches!(
            TracingContext::new(&engine, builder_a.clone()).trace(ScalarOperation::Add, &[&tracer_a, &tracer_b]),
            Err(TracingError::MismatchedProgramBuilders),
        ));
        assert_eq!(builder_a.borrow().error, Some(TracingError::MismatchedProgramBuilders));

        // Test tracing after a builder failure by returning poisoned tracers when output types can still be inferred.
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let atom = builder.borrow_mut().add_input(DataType::F64);
        let builder_error = TracingError::InvalidInputCount { expected: 1, got: 0 };
        builder.borrow_mut().error = Some(builder_error.clone());
        let tracing_context = TracingContext::new(&engine, builder.clone());
        let tracer = tracing_context.tracer(atom, None);
        let outputs = tracing_context.trace(ScalarOperation::Neg, &[&tracer]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(matches!(&outputs[0].state, TracerState::Poison));
        assert_eq!(outputs[0].r#type().into_owned(), DataType::F64);
        assert_eq!(builder.borrow().error, Some(builder_error.clone()));
        assert!(matches!(
            tracing_context.trace(ScalarOperation::Add, &[&tracer]),
            Err(TracingError::Type(TypeError { message })) if message == "expected 2 inputs but got 1",
        ));
        assert_eq!(builder.borrow().error, Some(builder_error));

        // Test propagating abstract-evaluation errors and recording them on the builder.
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let lhs_atom = builder.borrow_mut().add_input(DataType::F8E3M4);
        let rhs_atom = builder.borrow_mut().add_input(DataType::F32);
        let tracing_context = TracingContext::new(&engine, builder.clone());
        let lhs = tracing_context.tracer(lhs_atom, None);
        let rhs = tracing_context.tracer(rhs_atom, None);
        let result = tracing_context.trace(ScalarOperation::Add, &[&lhs, &rhs]);
        assert!(matches!(
            result,
            Err(TracingError::Type(TypeError { message }))
                if message == "add input types are not broadcast-compatible"
        ));
        assert!(matches!(
            builder.borrow().error.clone(),
            Some(TracingError::Type(TypeError { message }))
                if message == "add input types are not broadcast-compatible"
        ));

        // Test using the context itself as an engine for traced identity constants.
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let tracing_context = TracingContext::new(&engine, builder.clone());
        let zero = Engine::zero(&tracing_context, &DataType::F64).unwrap();
        let one = Engine::one(&tracing_context, &DataType::F64).unwrap();
        assert_eq!(zero.r#type().into_owned(), DataType::F64);
        assert_eq!(one.r#type().into_owned(), DataType::F64);
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
        assert_eq!(program.interpret(Vec::new()), Ok(vec![0.0, 1.0]));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:f64 = const
                    %1:f64 = const
                in (%0, %1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_tracer_state_clone_debug_and_equality() {
        let live = TracerState::Live(AtomId { index: 3 });
        assert_eq!(live.clone(), TracerState::Live(AtomId { index: 3 }));
        assert_eq!(TracerState::Poison.clone(), TracerState::Poison);
        assert_ne!(live, TracerState::Poison);
        assert_eq!(format!("{live:?}"), "Live(AtomId { index: 3 })");
        assert_eq!(format!("{:?}", TracerState::Poison), "Poison");
    }

    #[test]
    fn test_tracer() {
        let engine = ScalarEngine::<f64>::new();

        // Test handles, atom lookup, cloning, typing, and rendering.
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let atom = builder.borrow_mut().add_input(DataType::F64);
        let tracing_context = TracingContext::new(&engine, builder.clone());
        let tracer = tracing_context.tracer(atom, None);
        let poisoned = Tracer { state: TracerState::Poison, r#type: DataType::F64, context: tracing_context.clone() };
        let cloned_tracer = tracer.clone();
        assert!(std::ptr::eq(tracer.engine(), &engine));
        assert!(Rc::ptr_eq(tracer.builder(), &builder));
        assert_eq!(tracer.atom_id(), Ok(atom));
        assert_eq!(poisoned.atom_id(), Err(TracingError::PoisonedTracer));
        assert_eq!(cloned_tracer.state, tracer.state);
        assert_eq!(cloned_tracer.r#type, tracer.r#type);
        assert!(Rc::ptr_eq(cloned_tracer.builder(), &builder));
        assert!(matches!(tracer.r#type(), Cow::Borrowed(r#type) if *r#type == DataType::F64));
        assert_eq!(tracer.to_string(), "%0");
        assert_eq!(format!("{tracer:?}"), "Tracer { state: Live(AtomId { index: 0 }), type: F64, .. }");
        assert_eq!(poisoned.to_string(), "<poison:f64>");
        assert_eq!(format!("{poisoned:?}"), "Tracer { state: Poison, type: F64, .. }");

        // Test staging value-level identity helpers through the tracer convenience API.
        let zero = tracer.zero_like();
        let one = tracer.one_like();
        assert_eq!(zero.r#type().into_owned(), DataType::F64);
        assert_eq!(one.r#type().into_owned(), DataType::F64);
        let zero_atom = zero.atom_id().expect("zero_like output should remain live");
        let one_atom = one.atom_id().expect("one_like output should remain live");
        let program = builder
            .borrow()
            .clone()
            .build::<f64, Vec<f64>>(vec![zero_atom, one_atom], Placeholder, vec![Placeholder, Placeholder])
            .unwrap();
        assert_eq!(program.interpret(2.0), Ok(vec![0.0, 1.0]));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = zero_like %0
                    %2:f64 = one_like %0
                in (%1, %2)
            "}
            .trim_end(),
        );

        // Test staging a unary operation through the tracer convenience API.
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let atom = builder.borrow_mut().add_input(DataType::F64);
        let tracer = TracingContext::new(&engine, builder.clone()).tracer(atom, None);
        let output = tracer.unary(ScalarOperation::Neg);
        assert_eq!(output.r#type().into_owned(), DataType::F64);
        let output_atom = output.atom_id().expect("unary output should remain live");
        let program = builder.borrow().clone().build::<f64, f64>(vec![output_atom], Placeholder, Placeholder).unwrap();
        assert_eq!(program.interpret(2.0), Ok(-2.0));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = neg %0
                in (%1)
            "}
            .trim_end(),
        );

        // Test staging a binary operation through the tracer convenience API.
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let lhs_atom = builder.borrow_mut().add_input(DataType::F64);
        let rhs_atom = builder.borrow_mut().add_input(DataType::F64);
        let tracing_context = TracingContext::new(&engine, builder.clone());
        let lhs = tracing_context.tracer(lhs_atom, None);
        let rhs = tracing_context.tracer(rhs_atom, None);
        let output = lhs.binary(rhs, ScalarOperation::Add);
        assert_eq!(output.r#type().into_owned(), DataType::F64);
        let output_atom = output.atom_id().expect("binary output should remain live");
        let program = builder
            .borrow()
            .clone()
            .build::<(f64, f64), f64>(vec![output_atom], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        assert_eq!(program.interpret((2.0, 3.0)), Ok(5.0));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = add %0 %1
                in (%2)
            "}
            .trim_end(),
        );

        // Test that binary operations poison the result when inputs belong to different builders.
        let builder_a = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let builder_b = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let atom_a = builder_a.borrow_mut().add_input(DataType::F64);
        let atom_b = builder_b.borrow_mut().add_input(DataType::F64);
        let tracer_a = TracingContext::new(&engine, builder_a.clone()).tracer(atom_a, None);
        let tracer_b = TracingContext::new(&engine, builder_b).tracer(atom_b, None);
        let output = tracer_a.binary(tracer_b, ScalarOperation::Add);
        assert!(matches!(&output.state, TracerState::Poison));
        assert_eq!(output.r#type().into_owned(), DataType::F64);
        assert_eq!(builder_a.borrow().error, Some(TracingError::MismatchedProgramBuilders));
    }

    #[test]
    fn test_tracer_unary_records_invalid_output_count_and_returns_poisoned_tracer() {
        #[derive(Copy, Clone, Debug)]
        struct NoOutputOperation;

        impl Operation<DataType> for NoOutputOperation {
            #[inline]
            fn name(&self) -> &'static str {
                "no_output"
            }

            fn infer_output_types(&self, _input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
                Ok(Vec::new())
            }
        }

        struct NoOutputEngine;

        impl Engine for NoOutputEngine {
            type Type = DataType;
            type Value = f64;

            fn zero(&self, _type: &DataType) -> Result<f64, TracingError> {
                Ok(0.0)
            }

            fn one(&self, _type: &DataType) -> Result<f64, TracingError> {
                Ok(1.0)
            }
        }

        impl TracingEngine for NoOutputEngine {
            type OperationCarrier = NoOutputOperation;
        }

        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, NoOutputOperation>::new()));
        let input_type = DataType::F64;
        let engine = NoOutputEngine;
        let tracer = TracingContext::new(&engine, builder.clone()).input(input_type);
        let output = tracer.unary(NoOutputOperation);
        assert!(matches!(&output.state, TracerState::Poison));
        assert_eq!(output.r#type().into_owned(), DataType::F64);
        assert_eq!(builder.borrow().error, Some(TracingError::InvalidOutputCount { expected: 1, got: 0 }));
    }
}
