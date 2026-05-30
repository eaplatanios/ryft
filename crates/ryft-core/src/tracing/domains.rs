use std::borrow::Cow;
use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::rc::Rc;

use half::{bf16, f16};

use ryft_macros::Parameter;

use crate::operations::Operation;
use crate::operations::scalars::{LinearScalarOperation, ScalarOperation};
use crate::parameters::Parameter;
use crate::tracing::contexts::{Context, TracingContext};
use crate::tracing::{AtomId, ProgramBuilder, Traceable, TracingError};
use crate::types::{DataType, Type, TypeError, Typed};

/// Type/value universe used by program interpretation, tracing, and transformations. A [`Domain`] says what type
/// metadata and runtime values a backend or value model understands. It does not describe an active tracing run, and it
/// does not decide what happens when a primitive is bound. Active bind handling lives in [`Context`] implementations.
/// This separation lets the same domain capabilities be reused by ordinary tracing, batching, linearization, and
/// other transform contexts. Domains that can synthesize canonical runtime values from type metadata implement
/// [`RuntimeDomain`]. Domains that can be used to build ordinary traced programs implement [`TracingDomain`].
pub trait Domain {
    /// [`Type`]s that this [`Domain`] uses to represent the abstract metadata associated with its [`Traceable`] values.
    /// A commonly used [`Type`] is [`ArrayType`](crate::ArrayType), though scalar-only domains can use [`DataType`]
    /// and richer backends may use richer metadata.
    type Type: Type + Parameter;

    /// [`Traceable`] value types supported by this [`Domain`]. Instances of this type are what [`Program`]
    /// interpretation and eager transforms operate on. [`Domain::Type`] represents abstract staging metadata,
    /// while [`Domain::Value`] represents the runtime values that inhabit traced [`Program`]s during execution.
    type Value: Traceable<Self::Type>;
}

/// Represents [`Domain`]s that can synthesize canonical runtime values from abstract [`Type`] metadata.
/// [`RuntimeDomain`] is intentionally narrower than [`Program`] execution or arbitrary value construction. It only says
/// that the domain can materialize distinguished runtime values needed by interpreters and program transforms when they
/// have type metadata but no existing value to copy (currently the additive and multiplicative identities exposed
/// by [`RuntimeDomain::zero`] and [`RuntimeDomain::one`]). A backend can still support staging programs through
/// [`TracingDomain`] without being a [`RuntimeDomain`] if it cannot construct concrete values from types alone.
pub trait RuntimeDomain: Domain {
    /// Returns the additive-identity value (i.e., the _zero_ value) that corresponds to the provided type.
    fn zero(&self, r#type: &Self::Type) -> Result<Self::Value, TracingError>;

    /// Returns the multiplicative-identity value (i.e., the _one_ value) that corresponds to the provided type.
    fn one(&self, r#type: &Self::Type) -> Result<Self::Value, TracingError>;
}

/// Backend capability for ordinary traced [`Program`](crate::tracing::Program) construction. A [`TracingDomain`]
/// selects the constant payload type and concrete [`Operation`] representation stored in programs for a backend. It
/// remains a passive capability description: it does not own a [`ProgramBuilder`], and it does not intercept primitive
/// binds. Ordinary tracing creates a [`TracingContext`] from a [`TracingDomain`], and that context owns the active
/// [`ProgramBuilder`] and bind behavior for a single trace.
///
/// Operation representations are usually closed enums whose variants wrap the primitive operations supported by the
/// backend, though simple tracing domains may use one primitive operation type directly.
pub trait TracingDomain: Domain + Sized {
    /// Constant payload type stored in traced [`Program`]s for this domain. For eager domains this is usually the same
    /// type as [`Domain::Value`]. Compiled backends may use a lifetime-free abstract carrier here while reserving
    /// [`Domain::Value`] for concrete runtime values.
    type Constant: Traceable<Self::Type>;

    /// [`Operation`] representation selected by this [`TracingDomain`] for ordinary traced [`Program`]s.
    type Operation: Operation<Self::Type>;

    /// Lifts a staged [`Program`] constant into this domain's runtime value representation. Most eager domains use
    /// the same representation for [`TracingDomain::Constant`] and [`Domain::Value`], so this is just an identity
    /// conversion. Backends that use abstract, lifetime-free constants for compiled programs can either materialize
    /// a runtime value here when that is semantically valid, or return an error when an abstract constant cannot be
    /// interpreted as a concrete runtime value.
    fn lift_constant(&self, constant: Self::Constant) -> Result<Self::Value, TracingError>;
}

/// [`TracingDomain`] extension for domains that can closure-convert runtime values into staged constants. A captured
/// value is stored outside the staged [`Program`] in a runtime capture table. The program itself stores only the
/// domain's [`TracingDomain::Constant`] reference to that table entry. This hook lets an active [`TracingContext`]
/// append a runtime value to its capture table while keeping the concrete constant representation domain-specific.
pub trait CapturingDomain<C: Traceable<Self::Type>>: TracingDomain {
    /// Creates a staged constant payload that refers to `value` at `index` in the active capture table.
    fn capture_constant(&self, index: usize, value: &C) -> Result<Self::Constant, TracingError>;
}

/// [`TracingDomain`] that only supports tracing staged [`Program`]s and that is not a [`RuntimeDomain`]. This tracing
/// domain is used when code needs the value-level tracing APIs for a known [`ProgramBuilder`] shape but does not have,
/// and should not require, a concrete backend [`RuntimeDomain`] capability.
#[derive(Copy, Clone, Debug, Default)]
pub struct ProgramTracingDomain<T: Type + Parameter, V: Traceable<T>, O: Operation<T>> {
    /// [`PhantomData`] marker tying this zero-sized tracing domain to its associated type, value, and operation types.
    marker: PhantomData<fn() -> (T, V, O)>,
}

impl<T: Type + Parameter, V: Traceable<T>, O: Operation<T>> ProgramTracingDomain<T, V, O> {
    /// Creates a new trace-only [`ProgramTracingDomain`].
    #[inline]
    pub const fn new() -> Self {
        Self { marker: PhantomData }
    }
}

impl<T: Type + Parameter, V: Traceable<T>, O: Operation<T>> Domain for ProgramTracingDomain<T, V, O> {
    type Type = T;
    type Value = V;
}

impl<T: Type + Parameter, V: Traceable<T>, O: Operation<T>> TracingDomain for ProgramTracingDomain<T, V, O> {
    type Constant = V;
    type Operation = O;

    #[inline]
    fn lift_constant(&self, constant: V) -> Result<V, TracingError> {
        Ok(constant)
    }
}

/// [`Tracer`] used for tracing [`Program`]s.
pub type ProgramTracer<'domain, T, V, O> = DomainTracer<'domain, ProgramTracingDomain<T, V, O>>;

// TODO(eaplatanios): Does this really belong here?
/// Stateless [`TracingDomain`] that uses [`DataType`] for scalar metadata and Rust scalar values such as `f32` for
/// runtime values. [`ScalarDomain`] is the minimal scalar-only backend used throughout tests and examples in
/// `ryft-core`. It demonstrates the intended role of [`RuntimeDomain`] in the smallest possible form: there are no
/// device handles, no mesh states, and no backend registries; just the built-in [`ScalarOperation`] variants plus
/// [`DataType`]-driven construction of scalar values.
#[derive(Copy, Clone, Debug, Default)]
pub struct ScalarDomain<V> {
    /// [`LinearScalarDomain`] to be used by automatic differentiation transforms.
    linear_domain: LinearScalarDomain<V>,
}

impl<V> ScalarDomain<V> {
    /// Creates a new [`ScalarDomain`].
    #[inline]
    pub const fn new() -> Self {
        Self { linear_domain: LinearScalarDomain::new() }
    }

    /// Returns the [`LinearScalarDomain`] associated with this [`ScalarDomain`].
    #[inline]
    pub const fn linear_domain(&self) -> &LinearScalarDomain<V> {
        &self.linear_domain
    }
}

/// Stateless linear [`TracingDomain`] for scalar tangent and cotangent [`Program`]s. This is the linear compliment of
/// [`ScalarDomain`]. They both use the same scalar type (i.e, [`DataType`]) and the same runtime scalar values (i.e.,
/// `f32`, `f64`, etc.); they differ only in the operation type selected by [`TracingDomain`]:
///
/// - [`ScalarDomain`] records ordinary scalar programs using [`ScalarOperation`].
/// - [`LinearScalarDomain`] records linear tangent and cotangent programs using [`LinearScalarOperation`].
///
/// This separate domain is needed because [`TracingDomain::Operation`] is an associated type. Once [`ScalarDomain`]
/// says "ordinary scalar traces store [`ScalarOperation`] instructions", the same domain type cannot also say "linear
/// scalar traces store [`LinearScalarOperation`] instructions". Automatic differentiation therefore keeps a tiny
/// companion domain for linear [`Program`]s.
///
/// For example, tracing `f(x) = x * x` with [`ScalarDomain<f64>`] records an ordinary multiplication. Linearizing that
/// program at `x = 3.0` produces a tangent program equivalent to `δx -> 3.0 * δx + 3.0 * δx`; that tangent program is
/// stored with [`LinearScalarOperation`] instructions such as `scale` and `add`. [`LinearScalarDomain`] is what tells
/// the generic tracing machinery to use that linear operation type instead of the standard operation type.
#[derive(Copy, Clone, Debug, Default)]
pub struct LinearScalarDomain<V> {
    /// [`PhantomData`] marker that ties this zero-sized [`LinearScalarDomain`] to its scalar value type.
    marker: PhantomData<fn() -> V>,
}

impl<V> LinearScalarDomain<V> {
    /// Creates a new [`LinearScalarDomain`].
    #[inline]
    pub const fn new() -> Self {
        Self { marker: PhantomData }
    }
}

macro_rules! impl_domain_for_scalar {
    ($ty:ty, $data_type:path, $zero:expr, $one:expr) => {
        impl Domain for ScalarDomain<$ty> {
            type Type = DataType;
            type Value = $ty;
        }

        impl RuntimeDomain for ScalarDomain<$ty> {
            #[inline]
            fn zero(&self, r#type: &DataType) -> Result<$ty, TracingError> {
                if *r#type != $data_type {
                    return Err(TypeError {
                        message: format!("scalar domain for {} cannot synthesize zero for {}", $data_type, r#type),
                    }
                    .into());
                }
                Ok($zero)
            }

            #[inline]
            fn one(&self, r#type: &DataType) -> Result<$ty, TracingError> {
                if *r#type != $data_type {
                    return Err(TypeError {
                        message: format!("scalar domain for {} cannot synthesize one for {}", $data_type, r#type),
                    }
                    .into());
                }
                Ok($one)
            }
        }

        impl TracingDomain for ScalarDomain<$ty> {
            type Constant = $ty;
            type Operation = ScalarOperation<$ty>;

            #[inline]
            fn lift_constant(&self, constant: $ty) -> Result<$ty, TracingError> {
                Ok(constant)
            }
        }

        impl Domain for LinearScalarDomain<$ty> {
            type Type = DataType;
            type Value = $ty;
        }

        impl RuntimeDomain for LinearScalarDomain<$ty> {
            #[inline]
            fn zero(&self, r#type: &DataType) -> Result<$ty, TracingError> {
                ScalarDomain::<$ty>::new().zero(r#type)
            }

            #[inline]
            fn one(&self, r#type: &DataType) -> Result<$ty, TracingError> {
                ScalarDomain::<$ty>::new().one(r#type)
            }
        }

        impl TracingDomain for LinearScalarDomain<$ty> {
            type Constant = $ty;
            type Operation = LinearScalarOperation<$ty>;

            #[inline]
            fn lift_constant(&self, constant: $ty) -> Result<$ty, TracingError> {
                Ok(constant)
            }
        }
    };
}

impl_domain_for_scalar!(bool, DataType::Boolean, false, true);
impl_domain_for_scalar!(i8, DataType::I8, 0i8, 1i8);
impl_domain_for_scalar!(i16, DataType::I16, 0i16, 1i16);
impl_domain_for_scalar!(i32, DataType::I32, 0i32, 1i32);
impl_domain_for_scalar!(i64, DataType::I64, 0i64, 1i64);
impl_domain_for_scalar!(u8, DataType::U8, 0u8, 1u8);
impl_domain_for_scalar!(u16, DataType::U16, 0u16, 1u16);
impl_domain_for_scalar!(u32, DataType::U32, 0u32, 1u32);
impl_domain_for_scalar!(u64, DataType::U64, 0u64, 1u64);
impl_domain_for_scalar!(bf16, DataType::BF16, bf16::ZERO, bf16::ONE);
impl_domain_for_scalar!(f16, DataType::F16, f16::ZERO, f16::ONE);
impl_domain_for_scalar!(f32, DataType::F32, 0.0, 1.0);
impl_domain_for_scalar!(f64, DataType::F64, 0.0, 1.0);

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

/// Value used while tracing [`Program`]s through an active [`Context`], substituting actual runtime values and
/// recording the executed [`Operation`]s in that [`Context`]. When tracing fails, later operations return _poisoned_
/// tracers which are represented using [`TracerState::Poison`].
#[derive(Parameter)]
pub struct Tracer<C: Context> {
    /// [`TracerState`] of this [`Tracer`].
    state: TracerState,

    /// [`Type`] of the value that this [`Tracer`] represents.
    r#type: C::Type,

    /// [`Context`] associated with this [`Tracer`].
    context: C,
}

impl<C: Context> Tracer<C> {
    /// Creates a new [`Tracer`].
    #[inline]
    pub fn new(state: TracerState, r#type: C::Type, context: C) -> Self {
        Self { state, r#type, context }
    }

    /// Returns the [`TracerState`] of this [`Tracer`].
    #[inline]
    pub fn state(&self) -> &TracerState {
        &self.state
    }

    /// Returns the [`Context`] associated with this [`Tracer`].
    #[inline]
    pub fn context(&self) -> &C {
        &self.context
    }

    /// Returns the [`ProgramBuilder`] associated with this [`Tracer`].
    #[inline]
    pub fn builder(&self) -> &Rc<RefCell<ProgramBuilder<C::Type, C::Value, C::Operation>>> {
        self.context.builder()
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
    pub fn unary(self, operation: C::Operation) -> Self {
        match self.context.stage_operation(operation, &[&self]) {
            Ok(mut outputs) if outputs.len() == 1 => outputs.remove(0),
            Ok(outputs) => {
                self.context.error(TracingError::InvalidOutputCount { expected: 1, got: outputs.len() });
                Self { state: TracerState::Poison, r#type: self.r#type.clone(), context: self.context.clone() }
            }
            Err(error) => {
                self.context.error(error);
                Self { state: TracerState::Poison, r#type: self.r#type.clone(), context: self.context.clone() }
            }
        }
    }

    /// Applies the provided _binary_ [`Operation`] to this [`Tracer`] and the provided [`Tracer`] returning the
    /// resulting [`Tracer`]. _Binary_ operations are operations that have two inputs and a single output. If the
    /// provided operation is not a binary operation then the resulting [`Tracer`] will contain a
    /// [`TracerState::Poison`].
    pub fn binary(self, rhs: Self, operation: C::Operation) -> Self {
        match self.context.stage_operation(operation, &[&self, &rhs]) {
            Ok(mut outputs) if outputs.len() == 1 => outputs.remove(0),
            Ok(outputs) => {
                self.context.error(TracingError::InvalidOutputCount { expected: 1, got: outputs.len() });
                Self { state: TracerState::Poison, r#type: self.r#type.clone(), context: self.context.clone() }
            }
            Err(error) => {
                self.context.error(error);
                Self { state: TracerState::Poison, r#type: self.r#type.clone(), context: self.context.clone() }
            }
        }
    }
}

impl<C: Context> Clone for Tracer<C> {
    fn clone(&self) -> Self {
        Self { state: self.state.clone(), r#type: self.r#type.clone(), context: self.context.clone() }
    }
}

impl<C: Context<Type: Debug>> Debug for Tracer<C> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Tracer")
            .field("state", &self.state)
            .field("type", &self.r#type)
            .finish_non_exhaustive()
    }
}

impl<C: Context> Display for Tracer<C> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.state {
            TracerState::Live(atom_id) => write!(formatter, "{atom_id}"),
            TracerState::Poison => write!(formatter, "<poison:{}>", self.r#type),
        }
    }
}

impl<C: Context> Typed<C::Type> for Tracer<C> {
    #[inline]
    fn r#type(&self) -> Cow<'_, C::Type> {
        Cow::Borrowed(&self.r#type)
    }
}

impl<C: Context> Traceable<C::Type> for Tracer<C> {}

/// [`Tracer`] value used by ordinary backend tracing through a [`TracingContext`].
pub type DomainTracer<'domain, D> = Tracer<TracingContext<'domain, D>>;

impl<'domain, D: TracingDomain> DomainTracer<'domain, D> {
    /// Returns the [`TracingDomain`] associated with this [`DomainTracer`].
    #[inline]
    pub fn domain(&self) -> &'domain D {
        self.context.domain()
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::operations::constants::{OneLike, ZeroLike};
    use crate::parameters::Placeholder;
    use crate::types::{DataType, TypeError, Typed};

    use super::*;

    #[test]
    fn test_domain() {
        let bool_type = DataType::Boolean;
        let bool_domain = ScalarDomain::<bool>::new();
        assert_eq!(RuntimeDomain::zero(&bool_domain, &bool_type), Ok(false));
        assert_eq!(RuntimeDomain::one(&bool_domain, &bool_type), Ok(true));

        let i8_type = DataType::I8;
        let i8_domain = ScalarDomain::<i8>::new();
        assert_eq!(RuntimeDomain::zero(&i8_domain, &i8_type), Ok(0i8));
        assert_eq!(RuntimeDomain::one(&i8_domain, &i8_type), Ok(1i8));

        let i16_type = DataType::I16;
        let i16_domain = ScalarDomain::<i16>::new();
        assert_eq!(RuntimeDomain::zero(&i16_domain, &i16_type), Ok(0i16));
        assert_eq!(RuntimeDomain::one(&i16_domain, &i16_type), Ok(1i16));

        let i32_type = DataType::I32;
        let i32_domain = ScalarDomain::<i32>::new();
        assert_eq!(RuntimeDomain::zero(&i32_domain, &i32_type), Ok(0i32));
        assert_eq!(RuntimeDomain::one(&i32_domain, &i32_type), Ok(1i32));

        let i64_type = DataType::I64;
        let i64_domain = ScalarDomain::<i64>::new();
        assert_eq!(RuntimeDomain::zero(&i64_domain, &i64_type), Ok(0i64));
        assert_eq!(RuntimeDomain::one(&i64_domain, &i64_type), Ok(1i64));

        let u8_type = DataType::U8;
        let u8_domain = ScalarDomain::<u8>::new();
        assert_eq!(RuntimeDomain::zero(&u8_domain, &u8_type), Ok(0u8));
        assert_eq!(RuntimeDomain::one(&u8_domain, &u8_type), Ok(1u8));

        let u16_type = DataType::U16;
        let u16_domain = ScalarDomain::<u16>::new();
        assert_eq!(RuntimeDomain::zero(&u16_domain, &u16_type), Ok(0u16));
        assert_eq!(RuntimeDomain::one(&u16_domain, &u16_type), Ok(1u16));

        let u32_type = DataType::U32;
        let u32_domain = ScalarDomain::<u32>::new();
        assert_eq!(RuntimeDomain::zero(&u32_domain, &u32_type), Ok(0u32));
        assert_eq!(RuntimeDomain::one(&u32_domain, &u32_type), Ok(1u32));

        let u64_type = DataType::U64;
        let u64_domain = ScalarDomain::<u64>::new();
        assert_eq!(RuntimeDomain::zero(&u64_domain, &u64_type), Ok(0u64));
        assert_eq!(RuntimeDomain::one(&u64_domain, &u64_type), Ok(1u64));

        let bf16_type = DataType::BF16;
        let bf16_domain = ScalarDomain::<bf16>::new();
        assert_eq!(RuntimeDomain::zero(&bf16_domain, &bf16_type), Ok(bf16::ZERO));
        assert_eq!(RuntimeDomain::one(&bf16_domain, &bf16_type), Ok(bf16::ONE));

        let f16_type = DataType::F16;
        let f16_domain = ScalarDomain::<f16>::new();
        assert_eq!(RuntimeDomain::zero(&f16_domain, &f16_type), Ok(f16::ZERO));
        assert_eq!(RuntimeDomain::one(&f16_domain, &f16_type), Ok(f16::ONE));

        let f32_type = DataType::F32;
        let f32_domain = ScalarDomain::<f32>::new();
        assert_eq!(RuntimeDomain::zero(&f32_domain, &f32_type), Ok(0.0f32));
        assert_eq!(RuntimeDomain::one(&f32_domain, &f32_type), Ok(1.0f32));

        let f64_type = DataType::F64;
        let f64_domain = ScalarDomain::<f64>::new();
        assert_eq!(RuntimeDomain::zero(&f64_domain, &f64_type), Ok(0.0f64));
        assert_eq!(RuntimeDomain::one(&f64_domain, &f64_type), Ok(1.0f64));
        assert!(matches!(
            RuntimeDomain::zero(&f64_domain, &DataType::F32),
            Err(TracingError::Type(TypeError { message }))
                if message == "scalar domain for f64 cannot synthesize zero for f32",
        ));
        assert!(matches!(
            RuntimeDomain::one(&f64_domain, &DataType::F32),
            Err(TracingError::Type(TypeError { message }))
                if message == "scalar domain for f64 cannot synthesize one for f32",
        ));
    }

    #[test]
    fn test_scalar_domain() {
        // Check that [`ScalarDomain`] is zero-sized.
        assert_eq!(size_of::<ScalarDomain<bool>>(), 0);
        assert_eq!(size_of::<ScalarDomain<i8>>(), 0);
        assert_eq!(size_of::<ScalarDomain<i16>>(), 0);
        assert_eq!(size_of::<ScalarDomain<i32>>(), 0);
        assert_eq!(size_of::<ScalarDomain<i64>>(), 0);
        assert_eq!(size_of::<ScalarDomain<u8>>(), 0);
        assert_eq!(size_of::<ScalarDomain<u16>>(), 0);
        assert_eq!(size_of::<ScalarDomain<u32>>(), 0);
        assert_eq!(size_of::<ScalarDomain<u64>>(), 0);
        assert_eq!(size_of::<ScalarDomain<bf16>>(), 0);
        assert_eq!(size_of::<ScalarDomain<f16>>(), 0);
        assert_eq!(size_of::<ScalarDomain<f32>>(), 0);
        assert_eq!(size_of::<ScalarDomain<f64>>(), 0);

        // Check that `ScalarDomain` implements `RuntimeDomain`.
        assert_eq!(ScalarDomain::<f64>::new().zero(&DataType::F64), Ok(0.0));
        assert_eq!(ScalarDomain::<f64>::default().one(&DataType::F64), Ok(1.0));
    }

    #[test]
    fn test_tracer_state_clone_debug_and_equality() {
        let live = TracerState::Live(AtomId::new(3));
        assert_eq!(live.clone(), TracerState::Live(AtomId::new(3)));
        assert_eq!(TracerState::Poison.clone(), TracerState::Poison);
        assert_ne!(live, TracerState::Poison);
        assert_eq!(format!("{live:?}"), "Live(AtomId { index: 3 })");
        assert_eq!(format!("{:?}", TracerState::Poison), "Poison");
    }

    #[test]
    fn test_tracer() {
        let domain = ScalarDomain::<f64>::new();

        // Test handles, atom lookup, cloning, typing, and rendering.
        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new()));
        let atom = builder.borrow_mut().add_input(DataType::F64);
        let tracing_context = TracingContext::new(&domain, builder.clone());
        let tracer = tracing_context.tracer(atom, None);
        let poisoned = Tracer { state: TracerState::Poison, r#type: DataType::F64, context: tracing_context.clone() };
        let cloned_tracer = tracer.clone();
        assert!(std::ptr::eq(tracer.domain(), &domain));
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
        let tracer = TracingContext::new(&domain, builder.clone()).tracer(atom, None);
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
        let tracing_context = TracingContext::new(&domain, builder.clone());
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
        let tracer_a = TracingContext::new(&domain, builder_a.clone()).tracer(atom_a, None);
        let tracer_b = TracingContext::new(&domain, builder_b).tracer(atom_b, None);
        let output = tracer_a.binary(tracer_b, ScalarOperation::Add);
        assert!(matches!(&output.state, TracerState::Poison));
        assert_eq!(output.r#type().into_owned(), DataType::F64);
        assert_eq!(builder_a.borrow().error().cloned(), Some(TracingError::MismatchedProgramBuilders));
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

        struct NoOutputDomain;

        impl Domain for NoOutputDomain {
            type Type = DataType;
            type Value = f64;
        }

        impl RuntimeDomain for NoOutputDomain {
            fn zero(&self, _type: &DataType) -> Result<f64, TracingError> {
                Ok(0.0)
            }

            fn one(&self, _type: &DataType) -> Result<f64, TracingError> {
                Ok(1.0)
            }
        }

        impl TracingDomain for NoOutputDomain {
            type Constant = f64;
            type Operation = NoOutputOperation;

            #[inline]
            fn lift_constant(&self, constant: f64) -> Result<f64, TracingError> {
                Ok(constant)
            }
        }

        let builder = Rc::new(RefCell::new(ProgramBuilder::<DataType, f64, NoOutputOperation>::new()));
        let input_type = DataType::F64;
        let domain = NoOutputDomain;
        let tracer = TracingContext::new(&domain, builder.clone()).input(input_type);
        let output = tracer.unary(NoOutputOperation);
        assert!(matches!(&output.state, TracerState::Poison));
        assert_eq!(output.r#type().into_owned(), DataType::F64);
        assert_eq!(builder.borrow().error().cloned(), Some(TracingError::InvalidOutputCount { expected: 1, got: 0 }),);
    }
}
