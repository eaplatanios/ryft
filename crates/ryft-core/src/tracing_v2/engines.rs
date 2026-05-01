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
use crate::tracing_v2::operations::constants::{OneLike, ZeroLike};
use crate::tracing_v2::operations::primitive::PrimitiveOperation;
use crate::tracing_v2::operations::{SupportsAdd, SupportsMul, SupportsNeg};
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

/// [`TracingEngine`] extends [`Engine`] with a closed [`Operation`] carrier type that can be used to trace
/// and interpret [`Program`]s.
pub trait TracingEngine: Engine {
    /// Staged [`Operation`] type supported by this [`TracingEngine`].
    type Operation: Operation<Self::Type>;

    /// Traces the provided `function` into a [`Program`] for the provided input types, returning the output types of
    /// the traced [`Program`] along with that traced [`Program`] itself. This is the most symbolic tracing entry
    /// point in that it does not require concrete runtime input values but rather it only requires their types. The
    /// provided closure is executed once on [`Tracer`] values standing in for those input types to trace the function,
    /// and relies on [`Operation::infer_output_types`] for inferring output types.
    #[inline]
    fn trace<
        'engine,
        F: FnOnce(I::To<Tracer<'engine, Self>>) -> Result<O::To<Tracer<'engine, Self>>, TracingError>,
        I: Parameterized<Self::Type, Family: ParameterFamily<Self::Value> + ParameterFamily<Tracer<'engine, Self>>>,
        O: Parameterized<
                Self::Type,
                Family: ParameterFamily<Self::Value> + ParameterFamily<Tracer<'engine, Self>>,
                To<Tracer<'engine, Self>>: Parameterized<Tracer<'engine, Self>, To<Self::Type> = O>,
            >,
    >(
        &'engine self,
        function: F,
        input_types: I,
    ) -> Result<
        (O, Program<Self::Type, Self::Value, Self::Operation, I::To<Self::Value>, O::To<Self::Value>>),
        TracingError,
    > {
        let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        let input_structure = input_types.parameter_structure();
        let input = input_types
            .map_parameters(|r#type| Tracer {
                state: TracerState::Live(builder.borrow_mut().add_input(r#type.clone())),
                r#type,
                context: TracingContext::new(self, builder.clone()),
            })
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
        F: FnOnce(I::To<Tracer<'engine, Self>>) -> Result<O::To<Tracer<'engine, Self>>, TracingError>,
        I: Parameterized<
                Self::Value,
                Family: ParameterFamily<Tracer<'engine, Self>>,
                ParameterStructure: Debug + PartialEq,
            >,
        O: Parameterized<Self::Value, Family: ParameterFamily<Tracer<'engine, Self>>>,
    >(
        &'engine self,
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
                let input = I::To::<Tracer<'engine, Self>>::from_parameters(input_structure.clone(), flat_input)?;
                let output = function(input)?;
                output_structure = Some(output.parameter_structure());
                Ok(output.into_parameters().collect::<Vec<_>>())
            },
            input_types,
        )?;
        let output_structure = output_structure.expect("the function being traced should have been invoked");
        let flat_program = flat_program.into_simplified()?;
        let output = O::from_parameters(output_structure.clone(), flat_program.interpret(input_values)?)?;
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
/// handles, no mesh states, and no backend registries; just the built-in scalar primitive carriers plus
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
            type Operation = PrimitiveOperation<$ty, DataType>;
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

/// [`TracingEngine`] that is used while tracing [`Program`]s. This engine bundles an underlying [`TracingEngine`]
/// with a [`ProgramBuilder`] and uses [`Tracer`]s to represent values.
pub struct TracingContext<'engine, E: TracingEngine + ?Sized> {
    /// [`TracingEngine`] borrowed by this [`TracingContext`] for type-driven value synthesis and operation selection.
    pub engine: &'engine E,

    /// [`ProgramBuilder`] that owns the staged [`Program`] that is currently being traced.
    pub builder: Rc<RefCell<ProgramBuilder<E::Type, E::Value, E::Operation>>>,
}

impl<'engine, E: TracingEngine + ?Sized> TracingContext<'engine, E> {
    /// Creates a new [`TracingContext`] that borrows the provided [`TracingEngine`].
    #[inline]
    pub fn new(engine: &'engine E, builder: Rc<RefCell<ProgramBuilder<E::Type, E::Value, E::Operation>>>) -> Self {
        Self { engine, builder }
    }

    /// Lifts a concrete value into a [`Tracer`] constant in this [`TracingContext`].
    #[inline]
    pub fn lift(&self, value: E::Value) -> Tracer<'engine, E> {
        let r#type = value.r#type().into_owned();
        let atom = self.builder.borrow_mut().add_constant(value);
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
        operation: E::Operation,
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
            let builder = self.builder.borrow();
            Ok(output_atom_ids
                .into_iter()
                .map(|atom| Tracer {
                    state: TracerState::Live(atom),
                    r#type: builder.atoms[atom.index].r#type().into_owned(),
                    context: self.clone(),
                })
                .collect::<Vec<_>>())
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
        let value = self.engine.zero(r#type)?;
        let atom = self.builder.borrow_mut().add_constant(value);
        Ok(self.tracer(atom, Some(r#type.clone())))
    }

    #[inline]
    fn one(&self, r#type: &Self::Type) -> Result<Self::Value, TracingError> {
        let value = self.engine.one(r#type)?;
        let atom = self.builder.borrow_mut().add_constant(value);
        Ok(self.tracer(atom, Some(r#type.clone())))
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
    pub fn builder(&self) -> &Rc<RefCell<ProgramBuilder<E::Type, E::Value, E::Operation>>> {
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
    pub fn unary(self, operation: E::Operation) -> Self {
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
    pub fn binary(self, rhs: Self, operation: E::Operation) -> Self {
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
            TracerState::Live(atom) => write!(formatter, "{atom}"),
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

// TODO(eaplatanios): Review from here onwards.
impl<'engine, E: TracingEngine + ?Sized> ZeroLike for Tracer<'engine, E> {
    #[inline]
    fn zero_like(&self) -> Self {
        let r#type = self.r#type().into_owned();
        let value = match self.engine().zero(&r#type) {
            Ok(value) => value,
            Err(error) => {
                if self.builder().borrow().error.is_none() {
                    self.builder().borrow_mut().error = Some(error);
                }
                return Tracer { state: TracerState::Poison, r#type, context: self.context.clone() };
            }
        };
        let atom = self.builder().borrow_mut().add_constant(value);
        self.context.tracer(atom, Some(r#type))
    }
}

impl<'engine, E: TracingEngine + ?Sized> OneLike for Tracer<'engine, E> {
    #[inline]
    fn one_like(&self) -> Self {
        let r#type = self.r#type().into_owned();
        let value = match self.engine().one(&r#type) {
            Ok(value) => value,
            Err(error) => {
                if self.builder().borrow().error.is_none() {
                    self.builder().borrow_mut().error = Some(error);
                }
                return Tracer { state: TracerState::Poison, r#type, context: self.context.clone() };
            }
        };
        let atom = self.builder().borrow_mut().add_constant(value);
        self.context.tracer(atom, Some(r#type))
    }
}

impl<'engine, E: TracingEngine + ?Sized> Add for Tracer<'engine, E>
where
    E::Operation: SupportsAdd<E::Type, E::Value>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.binary(rhs, E::Operation::add_operation())
    }
}

impl<'engine, E: TracingEngine + ?Sized> Mul for Tracer<'engine, E>
where
    E::Operation: SupportsMul<E::Type, E::Value>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.binary(rhs, E::Operation::mul_operation())
    }
}

impl<'engine, E: TracingEngine + ?Sized> Neg for Tracer<'engine, E>
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
    use crate::types::{ArrayType, DataType, Shape, Size, TypeError, Typed};

    use super::*;

    type F64ProgramBuilder = ProgramBuilder<DataType, f64, PrimitiveOperation<f64, DataType>>;
    type ArrayF64ProgramBuilder = ProgramBuilder<ArrayType, f64, PrimitiveOperation<f64>>;

    fn assert_differentiable_engine<E: DifferentiableEngine>() {}

    fn new_f64_builder() -> Rc<RefCell<F64ProgramBuilder>> {
        Rc::new(RefCell::new(ProgramBuilder::new()))
    }

    fn new_array_f64_builder() -> Rc<RefCell<ArrayF64ProgramBuilder>> {
        Rc::new(RefCell::new(ProgramBuilder::new()))
    }

    fn scalar_f64_type() -> DataType {
        DataType::F64
    }

    fn array_scalar_f64_type() -> ArrayType {
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

    impl TracingEngine for TaggedEngine {
        type Operation = PrimitiveOperation<f64>;
    }

    #[derive(Copy, Clone, Debug)]
    struct NoOutputOperation;

    impl Operation<ArrayType> for NoOutputOperation {
        fn name(&self) -> &'static str {
            "no_output"
        }

        fn infer_output_types(&self, _input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
            Ok(Vec::new())
        }
    }

    struct NoOutputEngine;

    impl Engine for NoOutputEngine {
        type Type = ArrayType;
        type Value = f64;

        fn zero(&self, _type: &ArrayType) -> Result<f64, TracingError> {
            Ok(0.0)
        }

        fn one(&self, _type: &ArrayType) -> Result<f64, TracingError> {
            Ok(1.0)
        }
    }

    impl TracingEngine for NoOutputEngine {
        type Operation = NoOutputOperation;
    }

    #[test]
    fn test_scalar_engine_is_zero_sized() {
        assert_eq!(size_of::<ScalarEngine<bool>>(), 0);
        assert_eq!(size_of::<ScalarEngine<i8>>(), 0);
        assert_eq!(size_of::<ScalarEngine<u64>>(), 0);
        assert_eq!(size_of::<ScalarEngine<bf16>>(), 0);
        assert_eq!(size_of::<ScalarEngine<f16>>(), 0);
        assert_eq!(size_of::<ScalarEngine<f64>>(), 0);
        assert_eq!(size_of::<ScalarEngine<f32>>(), 0);
    }

    #[test]
    fn test_scalar_engine_produces_canonical_zero_and_one() {
        let bool_type = DataType::Boolean;
        let bool_engine = ScalarEngine::<bool>::new();
        assert_eq!(Engine::zero(&bool_engine, &bool_type), Ok(false));
        assert_eq!(Engine::one(&bool_engine, &bool_type), Ok(true));

        let i32_type = DataType::I32;
        let i32_engine = ScalarEngine::<i32>::new();
        assert_eq!(Engine::zero(&i32_engine, &i32_type), Ok(0i32));
        assert_eq!(Engine::one(&i32_engine, &i32_type), Ok(1i32));

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
    }

    #[test]
    fn test_scalar_engine_rejects_mismatched_identity_type() {
        let engine = ScalarEngine::<f64>::new();

        assert_eq!(
            Engine::zero(&engine, &DataType::F32).unwrap_err().to_string(),
            "scalar engine for f64 cannot synthesize zero for f32",
        );
        assert_eq!(
            Engine::one(&engine, &DataType::F32).unwrap_err().to_string(),
            "scalar engine for f64 cannot synthesize one for f32",
        );
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
    fn test_tracing_context_zero_and_one_lift_atoms() {
        let builder = new_f64_builder();
        let engine = ScalarEngine::<f64>::new();
        let tracing_context = TracingContext::new(&engine, builder.clone());
        let r#type = scalar_f64_type();

        let zero = Engine::zero(&tracing_context, &r#type).unwrap();
        let one = Engine::one(&tracing_context, &r#type).unwrap();

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
                let %0:f64 = const
                    %1:f64 = const
                in (%0, %1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_tracer_zero_like_adds_constant_atoms() {
        let builder = new_f64_builder();
        let atom = builder.borrow_mut().add_input(<f64 as Typed<DataType>>::r#type(&3.0f64).into_owned());
        let engine = ScalarEngine::<f64>::new();
        let tracer: Tracer<ScalarEngine<f64>> = TracingContext::new(&engine, builder).tracer(atom, None);
        let zero = tracer.zero_like();
        assert_eq!(zero.r#type().into_owned(), scalar_f64_type());
        let zero_atom = match &zero.state {
            TracerState::Live(atom) => *atom,
            TracerState::Poison => panic!("zero-like tracer should remain live"),
        };
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
                lambda %0:f64 .
                let %1:f64 = const
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
        let tracer: Tracer<ScalarEngine<f64>> = TracingContext::new(&engine, builder).tracer(atom, Some(input_type));

        assert!(matches!(tracer.r#type(), Cow::Borrowed(r#type) if *r#type == scalar_f64_type()));
    }

    #[test]
    fn test_trace_rejects_mismatched_program_builders() {
        let builder_a = new_array_f64_builder();
        let builder_b = new_array_f64_builder();
        let atom_a = builder_a.borrow_mut().add_input(<f64 as Typed<ArrayType>>::r#type(&1.0f64).into_owned());
        let atom_b = builder_b.borrow_mut().add_input(<f64 as Typed<ArrayType>>::r#type(&2.0f64).into_owned());
        let engine = TaggedEngine { id: 1 };
        let tracer_a = TracingContext::new(&engine, builder_a.clone()).tracer(atom_a, None);
        let tracer_b = TracingContext::new(&engine, builder_b).tracer(atom_b, None);

        assert!(matches!(
            TracingContext::new(&engine, builder_a.clone()).trace(PrimitiveOperation::Add, &[&tracer_a, &tracer_b]),
            Err(TracingError::MismatchedProgramBuilders),
        ));
        assert_eq!(builder_a.borrow().error, Some(TracingError::MismatchedProgramBuilders));
    }

    #[test]
    fn test_binary_operator_returns_poisoned_tracer_for_mismatched_program_builders() {
        let builder_a = new_array_f64_builder();
        let builder_b = new_array_f64_builder();
        let atom_a = builder_a.borrow_mut().add_input(array_scalar_f64_type());
        let atom_b = builder_b.borrow_mut().add_input(array_scalar_f64_type());
        let engine = TaggedEngine { id: 1 };
        let tracer_a = TracingContext::new(&engine, builder_a.clone()).tracer(atom_a, None);
        let tracer_b = TracingContext::new(&engine, builder_b).tracer(atom_b, None);

        let output = tracer_a.binary(tracer_b, PrimitiveOperation::Add);

        assert!(matches!(&output.state, TracerState::Poison));
        assert!(matches!(output.r#type(), Cow::Borrowed(r#type) if *r#type == array_scalar_f64_type()));
        assert_eq!(builder_a.borrow().error, Some(TracingError::MismatchedProgramBuilders));
    }

    #[test]
    fn test_unary_operator_records_invalid_output_count_and_returns_poisoned_tracer() {
        let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, f64, NoOutputOperation>::new()));
        let input_type = array_scalar_f64_type();
        let atom = builder.borrow_mut().add_input(input_type.clone());
        let engine = NoOutputEngine;
        let tracer = TracingContext::new(&engine, builder.clone()).tracer(atom, Some(input_type));

        let output = tracer.unary(NoOutputOperation);

        assert!(matches!(&output.state, TracerState::Poison));
        assert!(matches!(output.r#type(), Cow::Borrowed(r#type) if *r#type == array_scalar_f64_type()));
        assert_eq!(builder.borrow().error, Some(TracingError::InvalidOutputCount { expected: 1, got: 0 }));
    }

    #[test]
    fn test_trace_returns_poisoned_tracers_after_builder_failure() {
        let builder = new_array_f64_builder();
        let atom = builder.borrow_mut().add_input(<f64 as Typed<ArrayType>>::r#type(&1.0f64).into_owned());
        builder.borrow_mut().error = Some(TracingError::InvalidInputCount { expected: 1, got: 0 });
        let engine = TaggedEngine { id: 1 };
        let tracer = TracingContext::new(&engine, builder.clone()).tracer(atom, None);

        let outputs = TracingContext::new(&engine, builder).trace(PrimitiveOperation::Neg, &[&tracer]).unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(matches!(&outputs[0].state, TracerState::Poison));
        assert!(matches!(outputs[0].r#type(), Cow::Borrowed(r#type) if *r#type == array_scalar_f64_type()));
    }

    #[test]
    fn test_trace_caches_live_output_types() {
        let builder = new_array_f64_builder();
        let input_type = array_scalar_f64_type();
        let atom = builder.borrow_mut().add_input(input_type.clone());
        let engine = TaggedEngine { id: 1 };
        let tracer = TracingContext::new(&engine, builder.clone()).tracer(atom, Some(input_type));

        let outputs = TracingContext::new(&engine, builder).trace(PrimitiveOperation::Neg, &[&tracer]).unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(matches!(&outputs[0].state, TracerState::Live(_)));
        assert!(matches!(outputs[0].r#type(), Cow::Borrowed(r#type) if *r#type == array_scalar_f64_type()));
    }

    #[test]
    fn test_poisoned_state_atom_id_returns_poisoned_tracer_error() {
        let builder = new_array_f64_builder();
        let engine = TaggedEngine { id: 1 };
        let tracing_context = TracingContext::new(&engine, builder);
        let tracer = Tracer { state: TracerState::Poison, r#type: array_scalar_f64_type(), context: tracing_context };

        assert_eq!(tracer.atom_id(), Err(TracingError::PoisonedTracer));
        assert!(matches!(tracer.r#type(), Cow::Borrowed(r#type) if *r#type == array_scalar_f64_type()));
    }

    #[test]
    fn test_interpret_and_trace_replays_staged_graphs() {
        let engine = ScalarEngine::<f64>::new();
        let (output, program): (f64, Program<DataType, f64, PrimitiveOperation<f64, DataType>, f64, f64>) = engine
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
                lambda %0:f64 .
                let %1:f64 = mul %0 %0
                    %2:f64 = sin %0
                    %3:f64 = add %1 %2
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_interpret_and_trace_prunes_unused_staged_operations() {
        let engine = ScalarEngine::<f64>::new();
        let (output, program): (f64, Program<DataType, f64, PrimitiveOperation<f64, DataType>, f64, f64>) = engine
            .interpret_and_trace(
                |x: Tracer<ScalarEngine<f64>>| {
                    let _unused = x.clone().sin();
                    Ok(x.clone() * x)
                },
                2.0f64,
            )
            .unwrap();

        assert_eq!(output, 4.0);
        assert_eq!(program.interpret(0.5f64).unwrap(), 0.25);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = mul %0 %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_trace_stages_program_from_type_metadata() {
        let engine = ScalarEngine::<f64>::new();
        let input_type = scalar_f64_type();
        let (output_type, program): (DataType, Program<DataType, f64, PrimitiveOperation<f64, DataType>, f64, f64>) =
            engine
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
                lambda %0:f64 .
                let %1:f64 = mul %0 %0
                    %2:f64 = const
                    %3:f64 = add %1 %2
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_trace_rejects_escaped_program_builder() {
        let engine = ScalarEngine::<f64>::new();
        let escaped_builder = Rc::new(RefCell::new(None));
        let result: Result<
            (DataType, Program<DataType, f64, PrimitiveOperation<f64, DataType>, f64, f64>),
            TracingError,
        > = engine.trace(
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
        let live = TracingContext::new(&engine, builder.clone()).tracer(atom, None);
        let poison = Tracer {
            state: TracerState::Poison,
            r#type: scalar_f64_type(),
            context: TracingContext::new(&engine, builder),
        };

        assert_eq!(live.to_string(), "%0");
        assert_eq!(format!("{live:?}"), "Tracer { state: Live(AtomId { index: 0 }), type: F64, .. }");
        assert_eq!(poison.to_string(), "<poison:f64>");
        assert_eq!(format!("{poison:?}"), "Tracer { state: Poison, type: F64, .. }");
        assert_eq!(format!("{:?}", live.context), "TracingContext { .. }");
    }

    #[test]
    fn test_trace_records_and_returns_abstract_eval_error() {
        let builder = new_array_f64_builder();
        let lhs_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)]), None, None).unwrap();
        let rhs_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3)]), None, None).unwrap();
        let lhs_atom = builder.borrow_mut().add_input(lhs_type.clone());
        let rhs_atom = builder.borrow_mut().add_input(rhs_type);
        let engine = TaggedEngine { id: 1 };
        let lhs = TracingContext::new(&engine, builder.clone()).tracer(lhs_atom, None);
        let rhs = TracingContext::new(&engine, builder.clone()).tracer(rhs_atom, None);

        let result = TracingContext::new(&engine, builder.clone()).trace(PrimitiveOperation::Add, &[&lhs, &rhs]);

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
    }

    #[test]
    fn test_trace_returns_error_when_poisoned_abstract_eval_fails() {
        let builder = new_array_f64_builder();
        let input_type = array_scalar_f64_type();
        let atom = builder.borrow_mut().add_input(input_type.clone());
        builder.borrow_mut().error = Some(TracingError::InvalidInputCount { expected: 1, got: 0 });
        let engine = TaggedEngine { id: 1 };
        let tracer = TracingContext::new(&engine, builder.clone()).tracer(atom, None);

        let result = TracingContext::new(&engine, builder).trace(PrimitiveOperation::Add, &[&tracer]);

        assert!(matches!(
            result,
            Err(TracingError::Type(TypeError { message })) if message == "add expected 2 input types but got 1"
        ));
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

        impl TracingEngine for FailingOneEngine {
            type Operation = PrimitiveOperation<f64>;
        }

        let engine = FailingOneEngine;

        assert!(matches!(
            TracingEngine::interpret_and_trace::<_, f64, f64>(
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

        impl TracingEngine for TestEngine {
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

        impl crate::tracing_v2::engines::TracingEngine for TestEngine {
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
        let (_, compiled): (f64, Program<DataType, f64, PrimitiveOperation<f64, DataType>, f64, f64>) = engine
            .interpret_and_trace(|x: Tracer<ScalarEngine<f64>>| Ok(x.clone() * x.clone() + x.sin()), 2.0f64)
            .unwrap();

        assert_eq!(
            compiled.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = mul %0 %0
                    %2:f64 = sin %0
                    %3:f64 = add %1 %2
                in (%3)
            "}
            .trim_end(),
        );
        test_support::assert_bilinear_jit_rendering();
    }
}
