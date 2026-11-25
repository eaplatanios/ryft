//! [`Session`]-carrying values used by the Ryft XLA backend. [`Value`] is the XLA counterpart of [`ArrayIrValue`].
//! It holds an [`Array`], a first-class runtime dimension (i.e., a [`Dimension`]), or an array reference (i.e., a
//! [`Reference`]). Unlike [`ArrayIrValue`], whose dimension and reference members are session-free host values, every
//! member of a [`Value`], and every projected representation of a member, owns the [`Session`] session that it belongs
//! to. This has three consequences:
//!
//!   - **Eager Functions from Dimensions to Arrays:** Functions that turn first-class dimensions into arrays (e.g.,
//!     comparing two dimensions or [`DimensionToScalar`](ryft_core::DimensionToScalar)) materialize their results in
//!     the session of their inputs, so they execute eagerly outside any explicit context.
//!   - **Rich Dispatch:** [`Value`] dispatches and executes through its [`Session`], so its capabilities come from
//!     the blanket implementations in `ryft-core` that bind operations, exactly like those of the staged values of a
//!     [`Session`], and free transform entry points (e.g., [`batch`](ryft_core::batch) or
//!     [`differentiate_at`](ryft_core::differentiate_at)) recover the [`Session`] from the values themselves.
//!   - **No Construction Without a Session:** Because [`ValueProjection::from_projected`] is infallible and receives
//!     no session, projected dimensions and references are the session-carrying [`Dimension`] and [`Reference`] rather
//!     than [`DimensionValue`] and [`ArrayReference`]. Host-side dimension values enter a [`Session`] through its
//!     constants (i.e., [`Context::lift`]) instead.
//!
//! Host-side semantics that do not depend on the session (e.g., which members can act as `while` predicates, assertion
//! observations, and the reference handle lifecycle) delegate to the generic [`ArrayIrValue`] implementations in
//! `ryft-core` through a lossless conversion, so that both array IR value types share one source of truth.

use std::borrow::{Borrow, Cow};
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};

use ryft_core::{
    Add, And, Array as CoreArray, ArrayIrType, ArrayIrValue, ArrayReference, ArrayType, AssertionValue,
    CaptureConstant, CaptureReference, Compare, CompareOperation, ComparisonDirection, Concretizable, Context,
    DataType, DimensionType, DimensionValue, Div, EagerContext, LiteralIdentity, Mul, Neg, Not, Or, Parameter, Placed,
    Placement, ProgramError, ProjectedContext, ProjectedValue, ReferenceId, ReferenceType, Rem, Sub, Type as CoreType,
    TypeError, TypeIdentityRenaming, Typed, Value as CoreValue, ValueDirectDispatch, ValueDomainDispatch,
    ValueProjection, WhilePredicate, Xor,
};
use ryft_macros::Parameter;

use crate::compilation::domains::argument_type;
use crate::{Array, RetainedArray, Session};

/// Type of an XLA [`Value`], which is the [`ArrayIrType`] of `ryft-core` used as it is. XLA defines its own values,
/// constants, and operations, but it adds no types of its own, so this alias only gives the type universe of
/// [`Session`] the name that this crate uses for its other members. Being an alias, it is interchangeable with
/// [`ArrayIrType`] everywhere, including in patterns and in the paths of its variants (e.g., [`Type::Array`]).
pub type Type = ArrayIrType;

/// First-class runtime dimension bound to the [`Session`] session in which array results derived from it are
/// materialized. This is the dimension member of [`Value`] and its projection onto [`DimensionType`]. A [`Dimension`]
/// is a runtime _value_ that holds a checked [`DimensionValue`]. It is not a [`ryft_core::Dimension`], which is a
/// static or symbolic extent inside an [`ArrayType`]. Dimension arithmetic remains host integer work. It dispatches
/// through this dimension's [`Session`], whose eager binding evaluates dimension operations on the host without
/// compiling programs. Only functions that produce arrays (e.g., comparisons and
/// [`DimensionToScalar`](ryft_core::DimensionToScalar)) upload their results, which are placed according to the mesh
/// of the session. [`Value`] documents this placement policy. Two dimensions are equal when their [`DimensionValue`]s
/// are equal, regardless of their sessions.
#[derive(Clone, Parameter)]
pub struct Dimension {
    /// Checked host-side dimension.
    value: DimensionValue,

    /// [`Session`] in which array results computed from this dimension are materialized.
    session: Session,
}

impl Dimension {
    /// Creates a new [`Dimension`] that materializes array results derived from `value` in `session`.
    #[inline]
    pub fn new(value: DimensionValue, session: Session) -> Self {
        Self { value, session }
    }

    /// Returns the checked host-side [`DimensionValue`] of this dimension.
    #[inline]
    pub fn value(&self) -> &DimensionValue {
        &self.value
    }

    /// Returns the [`Session`] in which array results computed from this dimension are materialized.
    #[inline]
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// Consumes this dimension and returns its checked host-side [`DimensionValue`].
    #[inline]
    pub fn into_value(self) -> DimensionValue {
        self.value
    }
}

impl Debug for Dimension {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_tuple("Dimension").field(&self.value).finish()
    }
}

impl Display for Dimension {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.value, formatter)
    }
}

impl PartialEq for Dimension {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // Dimensions compare by value, because the session only selects where derived arrays are materialized.
        self.value == other.value
    }
}

impl Borrow<DimensionValue> for Dimension {
    #[inline]
    fn borrow(&self) -> &DimensionValue {
        &self.value
    }
}

impl Typed for Dimension {
    type Type = DimensionType;

    #[inline]
    fn r#type(&self) -> Cow<'_, DimensionType> {
        self.value.r#type()
    }
}

impl CoreValue for Dimension {
    // Like `Array`, a dimension dispatches and executes through the rich `Session` that it carries, projected onto
    // the dimension member family. That session evaluates dimension operations on the host and materializes array
    // results (e.g., `dimension_to_scalar`) in its session.

    type Dispatch = ValueDomainDispatch;
    type Domain = ProjectedContext<Session, DimensionType>;

    #[inline]
    fn domain(&self) -> Self::Domain {
        ProjectedContext::new(self.session.clone())
    }

    #[inline]
    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<Self::Type as CoreType>::Identity>,
    ) -> Result<Self, TypeError> {
        Ok(Self::new(self.value.rename_type_identities(renaming)?, self.session.clone()))
    }
}

impl Compare<Value, DimensionType> for Dimension {
    #[inline]
    fn compare(&self, other: &Self, direction: ComparisonDirection) -> Result<Value, ProgramError> {
        // Comparing two dimensions yields a Boolean array, which is materialized in this dimension's session by binding
        // the array IR comparison there. The session decides the predicate on the host from the two extents.
        let inputs = [Value::Dimension(self.clone()), Value::Dimension(other.clone())];
        Ok(self.session.bind(CompareOperation::<Type>::new(direction), Vec::new(), &inputs)?.remove(0))
    }
}

/// Handle to an XLA [`Array`] reference, bound to the [`Session`] session that owns its referent. This is the reference
/// member of [`Value`] and its projection onto [`ReferenceType<ArrayType>`](ReferenceType). The session is retained
/// next to the handle rather than recovered from the referent, because reading the referent locks its allocation and
/// fails once the allocation is frozen, whereas the session of a value must be available infallibly. Two references
/// are equal when their handles are equal (i.e., when they name the same allocation and view), regardless of their
/// sessions.
#[derive(Clone)]
pub struct Reference {
    /// Shared reference handle (i.e., the allocation, the view path, and the cached reference type).
    reference: ArrayReference<Array>,

    /// [`Session`] that owns the referent of this reference.
    session: Session,
}

impl Reference {
    /// Creates a new [`Reference`] that binds `reference` to `session`.
    #[inline]
    pub fn new(reference: ArrayReference<Array>, session: Session) -> Self {
        Self { reference, session }
    }

    /// Returns the shared [`ArrayReference`] handle of this reference.
    #[inline]
    pub fn handle(&self) -> &ArrayReference<Array> {
        &self.reference
    }

    /// Returns the [`Session`] that owns the referent of this reference.
    #[inline]
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// Consumes this reference and returns its shared [`ArrayReference`] handle.
    #[inline]
    pub fn into_handle(self) -> ArrayReference<Array> {
        self.reference
    }
}

impl Debug for Reference {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_tuple("Reference").field(&self.reference).finish()
    }
}

impl Display for Reference {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.reference, formatter)
    }
}

impl PartialEq for Reference {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // References compare by handle identity, like `ArrayReference`.
        self.reference == other.reference
    }
}

impl Typed for Reference {
    type Type = ReferenceType<ArrayType>;

    #[inline]
    fn r#type(&self) -> Cow<'_, ReferenceType<ArrayType>> {
        self.reference.r#type()
    }
}

/// Array IR runtime value associated with a [`Session`] (i.e., its [`Session::Value`](ryft_core::Domain::Value)),
/// and the [`Session`] counterpart of [`ArrayIrValue`] in which every member carries the [`Session`] session that
/// it belongs to. Refer to the [module documentation](self) for why the session is part of every member.
///
/// # Dispatch and Transforms
///
/// A [`Value`] dispatches and executes through the [`Session`] of its member, so every capability binds an
/// [`Operation`](crate::Operation) that the session executes eagerly, one operation at a time through its eager
/// dispatch cache, and free transform entry points recover that session from the values. When several inputs carry
/// different sessions, the receiver's session executes, and it validates that its array inputs belong to its client
/// and placement. Dimension inputs are host data and need no such validation.
///
/// # Placement
///
/// Arrays that a [`Value`] creates without array inputs (e.g., the scalar that converts a first-class dimension into
/// array data) are placed on the mesh of the executing session. When the session of an array member has no mesh of its
/// own, the session recovered from that member adopts the array's mesh, so a statically known extent read from an array
/// on some devices, and every scalar later derived from it, stays on those devices instead of moving to the client's
/// default devices. Dimensions read from or decoded from an array bind the array's mesh in the same way.
///
/// # Identity
///
/// [`PartialEq`] ignores sessions, so arrays compare by the storage identity of [`Array`], dimensions by their
/// [`DimensionValue`]s, and references by their handles.
#[derive(Clone, Debug, PartialEq, Parameter)]
pub enum Value {
    /// Device-resident [`Array`], which carries its own session.
    Array(Array),

    /// First-class runtime dimension.
    Dimension(Dimension),

    /// Array reference.
    Reference(Reference),
}

impl Value {
    /// Converts an [`ArrayIrValue`] over XLA arrays into a [`Value`]. Array members keep their original [`Session`].
    /// Dimension and reference members bind to `session`.
    #[inline]
    pub fn from_array_ir_value(value: ArrayIrValue<Array>, session: &Session) -> Self {
        match value {
            ArrayIrValue::Array(value) => Self::Array(value),
            ArrayIrValue::Dimension(value) => Self::Dimension(Dimension::new(value, session.clone())),
            ArrayIrValue::Reference(value) => Self::Reference(Reference::new(value, session.clone())),
        }
    }

    /// Returns the [`Session`] of this value's member.
    #[inline]
    pub fn session(&self) -> &Session {
        match self {
            Self::Array(value) => value.session(),
            Self::Dimension(value) => value.session(),
            Self::Reference(value) => value.session(),
        }
    }

    /// Converts this value into the session-free [`ArrayIrValue`] representation over XLA arrays, which drops the
    /// sessions of dimension and reference members. Used to delegate host-side semantics (e.g., reference handles and
    /// predicates) to the generic [`ArrayIrValue`] implementations in `ryft-core`.
    #[inline]
    pub fn into_array_ir_value(self) -> ArrayIrValue<Array> {
        match self {
            Self::Array(value) => ArrayIrValue::Array(value),
            Self::Dimension(value) => ArrayIrValue::Dimension(value.into_value()),
            Self::Reference(value) => ArrayIrValue::Reference(value.into_handle()),
        }
    }

    /// Converts this value into the immediate [`Constant`] that a trace whose capture table holds constants stages for
    /// it. An array is retained and becomes a [`Constant::Session`] handle to its storage, and a dimension becomes a
    /// [`Constant::Dimension`].
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError::Type`] error for a reference, because its state cannot be shared with a trace that
    /// outlives the current transform.
    #[inline]
    pub fn into_constant(self) -> Result<Constant, ProgramError> {
        match self {
            Self::Array(value) => Ok(Constant::Session(RetainedArray::new(&value))),
            Self::Dimension(value) => Ok(Constant::Dimension(value.into_value())),
            Self::Reference(_) => Err(TypeError::invalid(
                "cannot import an XLA reference into a trace whose capture table holds constants",
            )
            .into()),
        }
    }

    /// Returns this member's diagnostic kind name.
    #[inline]
    fn kind(&self) -> &'static str {
        match self {
            Self::Array(_) => "array",
            Self::Dimension(_) => "dimension",
            Self::Reference(_) => "reference",
        }
    }
}

impl Display for Value {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Array(value) => Display::fmt(value, formatter),
            Self::Dimension(value) => Display::fmt(value, formatter),
            Self::Reference(value) => Display::fmt(value, formatter),
        }
    }
}

impl Typed for Value {
    type Type = Type;

    #[inline]
    fn r#type(&self) -> Cow<'_, Type> {
        Cow::Owned(match self {
            Self::Array(value) => Type::Array(value.r#type().into_owned()),
            Self::Dimension(value) => Type::Dimension(value.r#type().into_owned()),
            Self::Reference(value) => Type::Reference(value.r#type().into_owned()),
        })
    }
}

impl Placed for Value {
    #[inline]
    fn placement(&self) -> Option<Placement> {
        // Only the array members of an XLA value live on devices.
        match self {
            Self::Array(value) => value.placement(),
            Self::Dimension(_) | Self::Reference(_) => None,
        }
    }
}

impl CoreValue for Value {
    // Every member carries its `Session`, so array IR values dispatch and execute through it directly. Capability
    // blankets in `ryft-core` bind `Operation`s into this session, and free transform entry points recover it here.

    type Dispatch = ValueDomainDispatch;
    type Domain = Session;

    #[inline]
    fn domain(&self) -> Self::Domain {
        // A mesh-less session recovered from an array places the values that it creates without array inputs (e.g.,
        // a folded static extent and the scalars later derived from it) on that array's mesh, rather than on the
        // client's default devices.
        match self {
            Self::Array(value) if value.session().device_mesh().is_err() => value.session().with_derived_mesh(value),
            _ => self.session().clone(),
        }
    }

    #[inline]
    fn relocatable_type(&self) -> Option<Self::Type> {
        // An uncommitted array whose type records nothing beyond replication enters a transform without a sharding,
        // which is the type for which a `jit` dispatcher specializes such an argument, so that a call moves it onto
        // the devices where it runs under a transform as it does outside of one.
        match self {
            Self::Array(array) => argument_type(array, &array.r#type()).map(Type::Array),
            Self::Dimension(_) | Self::Reference(_) => None,
        }
    }

    #[inline]
    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<Self::Type as CoreType>::Identity>,
    ) -> Result<Self, TypeError> {
        match self {
            Self::Array(value) => Ok(Self::Array(value.rename_type_identities(renaming)?)),
            Self::Dimension(value) => Ok(Self::Dimension(value.rename_type_identities(renaming)?)),
            Self::Reference(value) => Ok(Self::Reference(Reference::new(
                value.handle().rename_type_identities(renaming)?,
                value.session().clone(),
            ))),
        }
    }

    #[inline]
    fn validate_as_constant(&self) -> Result<(), TypeError> {
        self.clone().into_array_ir_value().validate_as_constant()
    }

    #[inline]
    fn is_zero(&self) -> bool {
        match self {
            Self::Array(value) => value.is_zero(),
            Self::Dimension(_) | Self::Reference(_) => false,
        }
    }

    #[inline]
    fn reference_id(&self) -> Option<ReferenceId> {
        // `singleton` keeps the default `None`, because it is a static function that cannot attach a session
        // to a singleton dimension, and `None` is the conservative answer.
        match self {
            Self::Array(_) | Self::Dimension(_) => None,
            Self::Reference(value) => Some(value.handle().id()),
        }
    }
}

// The host-side consumer contracts below delegate to the generic `ArrayIrValue` implementations in `ryft-core`, so
// their member-kind semantics (e.g., which members can act as predicates and how equal dimension carries pass through
// a batched selection) have a single source of truth.

impl Concretizable<bool> for Value {
    #[inline]
    fn concretize(&self) -> Result<bool, ProgramError> {
        self.clone().into_array_ir_value().concretize()
    }
}

impl WhilePredicate for Value {
    #[inline]
    fn any_true(&self) -> Result<bool, ProgramError> {
        self.clone().into_array_ir_value().any_true()
    }

    #[inline]
    fn select(&self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        let output = self
            .clone()
            .into_array_ir_value()
            .select(&on_true.clone().into_array_ir_value(), &on_false.clone().into_array_ir_value())?;
        Ok(Self::from_array_ir_value(output, on_true.session()))
    }
}

impl AssertionValue for Value {
    #[inline]
    fn assertion_observation(&self) -> Result<String, ProgramError> {
        self.clone().into_array_ir_value().assertion_observation()
    }

    #[inline]
    fn assertion_array(&self) -> Result<Option<CoreArray>, ProgramError> {
        self.clone().into_array_ir_value().assertion_array()
    }
}

impl ValueProjection<ArrayType> for Value {
    type Projected = Array;
    type ProjectedRef<'v>
        = &'v Array
    where
        Self: 'v;

    #[inline]
    fn from_projected(value: Array) -> Self {
        Self::Array(value)
    }

    #[inline]
    fn projected<'v>(&'v self) -> Result<&'v Array, TypeError>
    where
        ArrayType: 'v,
    {
        match self {
            Self::Array(value) => Ok(value),
            other => Err(TypeError::invalid(format!("expected array type but got {} type", other.kind()))),
        }
    }

    #[inline]
    fn into_projected(self) -> Result<Array, TypeError> {
        match self {
            Self::Array(value) => Ok(value),
            other => Err(TypeError::invalid(format!("expected array type but got {} type", other.kind()))),
        }
    }
}

impl ValueProjection<DimensionType> for Value {
    type Projected = Dimension;
    type ProjectedRef<'v>
        = &'v Dimension
    where
        Self: 'v;

    #[inline]
    fn from_projected(value: Dimension) -> Self {
        Self::Dimension(value)
    }

    #[inline]
    fn projected<'v>(&'v self) -> Result<&'v Dimension, TypeError>
    where
        DimensionType: 'v,
    {
        match self {
            Self::Dimension(value) => Ok(value),
            other => Err(TypeError::invalid(format!("expected dimension type but got {} type", other.kind()))),
        }
    }

    #[inline]
    fn into_projected(self) -> Result<Dimension, TypeError> {
        match self {
            Self::Dimension(value) => Ok(value),
            other => Err(TypeError::invalid(format!("expected dimension type but got {} type", other.kind()))),
        }
    }
}

impl ValueProjection<ReferenceType<ArrayType>> for Value {
    type Projected = Reference;
    type ProjectedRef<'v>
        = &'v Reference
    where
        Self: 'v;

    #[inline]
    fn from_projected(value: Reference) -> Self {
        Self::Reference(value)
    }

    #[inline]
    fn projected<'v>(&'v self) -> Result<&'v Reference, TypeError>
    where
        ReferenceType<ArrayType>: 'v,
    {
        match self {
            Self::Reference(value) => Ok(value),
            other => Err(TypeError::invalid(format!("expected reference type but got {} type", other.kind()))),
        }
    }

    #[inline]
    fn into_projected(self) -> Result<Reference, TypeError> {
        match self {
            Self::Reference(value) => Ok(value),
            other => Err(TypeError::invalid(format!("expected reference type but got {} type", other.kind()))),
        }
    }
}

impl From<Array> for Value {
    #[inline]
    fn from(value: Array) -> Self {
        Self::Array(value)
    }
}

impl From<Dimension> for Value {
    #[inline]
    fn from(value: Dimension) -> Self {
        Self::Dimension(value)
    }
}

impl From<Reference> for Value {
    #[inline]
    fn from(value: Reference) -> Self {
        Self::Reference(value)
    }
}

impl From<Value> for ArrayIrValue<Array> {
    #[inline]
    fn from(value: Value) -> Self {
        value.into_array_ir_value()
    }
}

impl TryFrom<Value> for Constant {
    type Error = ProgramError;

    #[inline]
    fn try_from(value: Value) -> Result<Self, Self::Error> {
        value.into_constant()
    }
}

// The `std::ops` operator traits are foreign, so the blanket tracer implementations in `ryft-core` cannot cover
// concrete backend values. These implementations provide the panicking operator sugar that the array operation
// bundles require by delegating to the fallible `ryft` capabilities, exactly like those of `Array`.

impl std::ops::Neg for Value {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Neg::neg(&self).expect("`neg` operation failed")
    }
}

impl std::ops::Add for Value {
    type Output = Self;

    #[inline]
    fn add(self, right: Self) -> Self {
        Add::add(&self, &right).expect("`add` operation failed")
    }
}

impl std::ops::Sub for Value {
    type Output = Self;

    #[inline]
    fn sub(self, right: Self) -> Self {
        Sub::sub(&self, &right).expect("`sub` operation failed")
    }
}

impl std::ops::Mul for Value {
    type Output = Self;

    #[inline]
    fn mul(self, right: Self) -> Self {
        Mul::mul(&self, &right).expect("`mul` operation failed")
    }
}

impl std::ops::Div for Value {
    type Output = Self;

    #[inline]
    fn div(self, right: Self) -> Self {
        Div::div(&self, &right).expect("`div` operation failed")
    }
}

impl std::ops::Rem for Value {
    type Output = Self;

    #[inline]
    fn rem(self, right: Self) -> Self {
        Rem::rem(&self, &right).expect("`rem` operation failed")
    }
}

impl std::ops::Not for Value {
    type Output = Self;

    #[inline]
    fn not(self) -> Self {
        Not::not(&self).expect("`not` operation failed")
    }
}

impl std::ops::BitAnd for Value {
    type Output = Self;

    #[inline]
    fn bitand(self, right: Self) -> Self {
        And::and(&self, &right).expect("`and` operation failed")
    }
}

impl std::ops::BitOr for Value {
    type Output = Self;

    #[inline]
    fn bitor(self, right: Self) -> Self {
        Or::or(&self, &right).expect("`or` operation failed")
    }
}

impl std::ops::BitXor for Value {
    type Output = Self;

    #[inline]
    fn bitxor(self, right: Self) -> Self {
        Xor::xor(&self, &right).expect("`xor` operation failed")
    }
}

/// Constant payload stored in the atom table of a staged XLA [`CoreProgram`](ryft_core::Program). Staged XLA programs
/// keep four kinds of constants apart, and this sum is the staged counterpart of the eager [`ArrayIrValue`] universe:
///
///   - **Captured Runtime Values:** Array buffers and external references stay in the surrounding compiled function's
///     capture table, while the program stores only a lifetime-free [`CaptureReference`] carrying its index and
///     structural [`Type`]. This keeps runtime storage out of literal IR payloads. A captured reference remains
///     metadata only and must be validated and discharged before ordinary XLA lowering.
///   - **Immediate First-Class Dimensions:** A [`DimensionValue`] is a checked host integer, so embedding it costs
///     nothing and it lowers to a scalar `stablehlo.constant`. Unlike a capture reference, it also stays usable inside
///     a nested region, most importantly a `shard_map` manual computation, which owns no capture table of its own. That
///     is what lets shape arithmetic, such as the explicit result extents of the collectives, be staged against a
///     manual region's shard-local values.
///   - **Immediate Boolean Predicates:** A predicate that is already known on the host while tracing (i.e., the result
///     of comparing two first-class dimensions with known extents) is stored in the program itself, so that it stays
///     concrete through array projection, logical composition, and assertion folding without requiring a captured
///     device buffer. Like an immediate dimension, it also stays usable inside nested regions, which own no capture
///     table. Booleans are the only array elements with an immediate form because predicates are the only array values
///     that host-side reasoning produces. Nothing in tracing computes, for example, a known `F32` scalar on the host,
///     so every other array constant is a capture or a session array.
///   - **Session Arrays:** A [`RetainedArray`] is a handle to an array that is retained for a trace whose capture table
///     cannot hold runtime arrays (e.g., the body of a rematerialized function or of a composite). Lowering turns it
///     into a hidden executable argument, as it does for a capture.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum Constant {
    /// Reference to a value held in the surrounding compiled function's capture table.
    Captured(CaptureReference<Type>),

    /// Handle that retains an array.
    Session(RetainedArray),

    /// Immediate checked host-side first-class dimension extent.
    Dimension(DimensionValue),

    /// Immediate scalar Boolean predicate.
    Boolean(bool),
}

impl Display for Constant {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Captured(value) => Display::fmt(value, formatter),
            Self::Session(value) => Display::fmt(value, formatter),
            Self::Dimension(value) => Display::fmt(value, formatter),
            Self::Boolean(value) => Display::fmt(value, formatter),
        }
    }
}

impl Typed for Constant {
    type Type = Type;

    #[inline]
    fn r#type(&self) -> Cow<'_, Type> {
        match self {
            Self::Captured(value) => value.r#type(),
            Self::Session(value) => Cow::Owned(Type::Array(value.r#type().into_owned())),
            Self::Dimension(value) => Cow::Owned(Type::Dimension(value.r#type().into_owned())),
            Self::Boolean(_) => Cow::Owned(ArrayType::scalar(DataType::Boolean).into()),
        }
    }
}

impl CoreValue for Constant {
    type Dispatch = ValueDirectDispatch;
    type Domain = EagerContext<Self>;

    #[inline]
    fn domain(&self) -> EagerContext<Self> {
        EagerContext::new()
    }

    #[inline]
    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<Type as CoreType>::Identity>,
    ) -> Result<Self, TypeError> {
        match self {
            Self::Captured(value) => Ok(Self::Captured(value.rename_type_identities(renaming)?)),
            Self::Session(value) => Ok(Self::Session(value.rename_type_identities(renaming)?)),
            Self::Dimension(value) => Ok(Self::Dimension(value.rename_type_identities(renaming)?)),
            Self::Boolean(value) => Ok(Self::Boolean(*value)),
        }
    }

    #[inline]
    fn capture_index(&self) -> Option<usize> {
        match self {
            Self::Captured(value) => Some(value.index()),
            Self::Session(_) | Self::Dimension(_) | Self::Boolean(_) => None,
        }
    }
}

impl LiteralIdentity for Constant {
    // XLA program constants carry only immediate scalars and typed capture slots, so their ordinary equality is already
    // exact and serves as their literal identity.

    #[inline]
    fn literal_eq(&self, other: &Self) -> bool {
        self == other
    }

    #[inline]
    fn literal_hash<H: Hasher>(&self, state: &mut H) {
        self.hash(state);
    }
}

impl CaptureConstant for Constant {
    #[inline]
    fn map_capture_index<F: FnOnce(usize) -> usize>(&self, map: F) -> Self {
        match self {
            Self::Captured(value) => Self::Captured(value.map_capture_index(map)),
            Self::Session(value) => Self::Session(value.clone()),
            Self::Dimension(value) => Self::Dimension(value.clone()),
            Self::Boolean(value) => Self::Boolean(*value),
        }
    }
}

impl Concretizable<bool> for Constant {
    #[inline]
    fn concretize(&self) -> Result<bool, ProgramError> {
        match self {
            Self::Boolean(value) => Ok(*value),
            _ => Err(ProgramError::Concretization {
                message: format!("cannot extract a concrete boolean from the staged XLA constant `{self}`"),
            }),
        }
    }
}

impl AssertionValue for Constant {
    #[inline]
    fn assertion_observation(&self) -> Result<String, ProgramError> {
        match self {
            Self::Dimension(value) => Ok(value.extent().to_string()),
            Self::Boolean(value) => Ok(value.to_string()),
            Self::Captured(_) => Err(ProgramError::Concretization {
                message: "cannot read an assertion observation from a captured constant reference".to_owned(),
            }),
            Self::Session(_) => Err(ProgramError::Concretization {
                message: "cannot read an assertion observation from a session constant handle".to_owned(),
            }),
        }
    }

    #[inline]
    fn assertion_array(&self) -> Result<Option<CoreArray>, ProgramError> {
        match self {
            Self::Dimension(_) => Ok(None),
            Self::Boolean(value) => Ok(Some(CoreArray::scalar(*value)?)),
            Self::Captured(_) => Err(ProgramError::Concretization {
                message: "cannot read an assertion array from a captured constant reference".to_owned(),
            }),
            Self::Session(_) => Err(ProgramError::Concretization {
                message: "cannot read an assertion array from a session constant handle".to_owned(),
            }),
        }
    }
}

impl ValueProjection<ArrayType> for Constant {
    type Projected = ArrayConstant;
    type ProjectedRef<'v> = ProjectedValue<ArrayType, &'v Self>;

    #[inline]
    fn from_projected(value: ArrayConstant) -> Self {
        match value {
            ArrayConstant::Captured(value) => Self::Captured(ValueProjection::<ArrayType>::from_projected(value)),
            ArrayConstant::Session(value) => Self::Session(value),
            ArrayConstant::Boolean(value) => Self::Boolean(value),
        }
    }

    #[inline]
    fn projected<'v>(&'v self) -> Result<Self::ProjectedRef<'v>, TypeError>
    where
        ArrayType: 'v,
    {
        Ok(ProjectedValue::new(self, <&ArrayType>::try_from(self.r#type().as_ref())?.clone()))
    }

    #[inline]
    fn into_projected(self) -> Result<ArrayConstant, TypeError> {
        match self {
            Self::Captured(value) => value.into_projected().map(ArrayConstant::Captured),
            Self::Session(value) => Ok(ArrayConstant::Session(value)),
            Self::Boolean(value) => Ok(ArrayConstant::Boolean(value)),
            Self::Dimension(_) => Err(TypeError::invalid("expected array type but got dimension type")),
        }
    }
}

impl ValueProjection<DimensionType> for Constant {
    type Projected = DimensionValue;
    type ProjectedRef<'v> = &'v DimensionValue;

    #[inline]
    fn from_projected(value: DimensionValue) -> Self {
        Self::Dimension(value)
    }

    #[inline]
    fn projected<'v>(&'v self) -> Result<&'v DimensionValue, TypeError>
    where
        DimensionType: 'v,
    {
        match self {
            Self::Captured(_) => Err(TypeError::invalid("expected an immediate dimension but got a captured value")),
            Self::Session(_) => Err(TypeError::invalid("expected an immediate dimension but got a session array")),
            Self::Boolean(_) => Err(TypeError::invalid("expected a dimension but got a Boolean array")),
            Self::Dimension(value) => Ok(value),
        }
    }

    #[inline]
    fn into_projected(self) -> Result<DimensionValue, TypeError> {
        match self {
            Self::Captured(_) => Err(TypeError::invalid("expected an immediate dimension but got a captured value")),
            Self::Session(_) => Err(TypeError::invalid("expected an immediate dimension but got a session array")),
            Self::Boolean(_) => Err(TypeError::invalid("expected a dimension but got a Boolean array")),
            Self::Dimension(value) => Ok(value),
        }
    }
}

impl From<CaptureReference<Type>> for Constant {
    #[inline]
    fn from(value: CaptureReference<Type>) -> Self {
        Self::Captured(value)
    }
}

impl From<DimensionValue> for Constant {
    #[inline]
    fn from(value: DimensionValue) -> Self {
        Self::Dimension(value)
    }
}

impl From<bool> for Constant {
    #[inline]
    fn from(value: bool) -> Self {
        Self::Boolean(value)
    }
}

/// Array constants retained by staged XLA programs. Captures refer to the enclosing capture table, session constants
/// refer to arrays that their handles retain, and Boolean literals preserve dimension comparison proofs without
/// allocating device storage or introducing a capture. Projection onto [`ArrayType`] is the identity and preserves
/// every representation unchanged.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum ArrayConstant {
    /// A reference to array data in the enclosing capture table.
    Captured(CaptureReference<ArrayType>),

    /// A handle that retains array data, which [`RetainedArray`] describes.
    Session(RetainedArray),

    /// An immediate scalar Boolean predicate.
    Boolean(bool),
}

impl Display for ArrayConstant {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Captured(value) => Display::fmt(value, formatter),
            Self::Session(value) => Display::fmt(value, formatter),
            Self::Boolean(value) => Display::fmt(value, formatter),
        }
    }
}

impl Typed for ArrayConstant {
    type Type = ArrayType;

    #[inline]
    fn r#type(&self) -> Cow<'_, ArrayType> {
        match self {
            Self::Captured(value) => value.r#type(),
            Self::Session(value) => value.r#type(),
            Self::Boolean(_) => Cow::Owned(ArrayType::scalar(DataType::Boolean)),
        }
    }
}

impl CoreValue for ArrayConstant {
    type Dispatch = ValueDirectDispatch;
    type Domain = EagerContext<Self>;

    #[inline]
    fn domain(&self) -> Self::Domain {
        EagerContext::new()
    }

    #[inline]
    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<ArrayType as CoreType>::Identity>,
    ) -> Result<Self, TypeError> {
        match self {
            Self::Captured(value) => Ok(Self::Captured(value.rename_type_identities(renaming)?)),
            Self::Session(value) => Ok(Self::Session(value.rename_type_identities(renaming)?)),
            Self::Boolean(value) => Ok(Self::Boolean(*value)),
        }
    }

    #[inline]
    fn capture_index(&self) -> Option<usize> {
        match self {
            Self::Captured(value) => Some(value.index()),
            Self::Session(_) | Self::Boolean(_) => None,
        }
    }
}

impl CaptureConstant for ArrayConstant {
    #[inline]
    fn map_capture_index<F: FnOnce(usize) -> usize>(&self, map: F) -> Self {
        match self {
            Self::Captured(value) => Self::Captured(value.map_capture_index(map)),
            Self::Session(value) => Self::Session(value.clone()),
            Self::Boolean(value) => Self::Boolean(*value),
        }
    }
}

impl Concretizable<bool> for ArrayConstant {
    #[inline]
    fn concretize(&self) -> Result<bool, ProgramError> {
        match self {
            Self::Captured(value) => value.concretize(),
            Self::Session(_) => Err(ProgramError::Concretization {
                message: "cannot extract a concrete boolean from a session constant handle".to_string(),
            }),
            Self::Boolean(value) => Ok(*value),
        }
    }
}

impl AssertionValue for ArrayConstant {
    #[inline]
    fn assertion_observation(&self) -> Result<String, ProgramError> {
        match self {
            Self::Captured(value) => value.assertion_observation(),
            Self::Session(_) => Err(ProgramError::Concretization {
                message: "cannot inspect a session constant assertion observation before execution".to_owned(),
            }),
            Self::Boolean(value) => Ok(value.to_string()),
        }
    }

    #[inline]
    fn assertion_array(&self) -> Result<Option<CoreArray>, ProgramError> {
        match self {
            Self::Captured(value) => value.assertion_array(),
            Self::Session(_) => Err(ProgramError::Concretization {
                message: "cannot inspect a session constant assertion array before execution".to_owned(),
            }),
            Self::Boolean(value) => Ok(Some(CoreArray::scalar(*value)?)),
        }
    }
}

impl ValueProjection<ArrayType> for ArrayConstant {
    type Projected = Self;
    type ProjectedRef<'v> = &'v Self;

    #[inline]
    fn from_projected(value: Self::Projected) -> Self {
        value
    }

    #[inline]
    fn projected<'v>(&'v self) -> Result<Self::ProjectedRef<'v>, TypeError>
    where
        ArrayType: 'v,
    {
        Ok(self)
    }

    #[inline]
    fn into_projected(self) -> Result<Self::Projected, TypeError> {
        Ok(self)
    }
}

impl From<CaptureReference<ArrayType>> for ArrayConstant {
    #[inline]
    fn from(value: CaptureReference<ArrayType>) -> Self {
        Self::Captured(value)
    }
}

impl From<bool> for ArrayConstant {
    #[inline]
    fn from(value: bool) -> Self {
        Self::Boolean(value)
    }
}

// TODO(eaplatanios): Review from here onwards.

#[cfg(test)]
mod tests {
    use std::hash::DefaultHasher;

    use pretty_assertions::assert_eq;

    use ryft_core::{
        Add, ArrayReferenceTransform, ArrayReferenceTransformIndex, ArrayType, Assert, AxisError, BatchAxis,
        CaptureReference, Compare, CompareOperation, ComparisonDirection, Concretizable, Context, ConvertElementType,
        DataType, Device, DeviceMesh, Dimension as CoreDimension, DimensionAddOperation, DimensionBounds,
        DimensionOperation, DimensionSize, DimensionSizeOperation, DimensionToScalar, DimensionToScalarOperation,
        DimensionType, DimensionValue, DimensionVariable, Gather, GatherMode, LogicalMesh, Max, Memory, MeshAxis,
        MeshAxisType, Min, OneOperation, OperationProvider, ParallelPermute, Placeholder, Pow, Reduce, ReductionKind,
        ReferenceAddUpdate, ReferenceError, ReferenceFreeze, ReferenceNew, ReferenceRead, ReferenceSwap, ReferenceType,
        ReferenceWrite, Rem, SaturatingSub, Shape, Sharding, Sort, SortDirection, SubOperation, TypeError,
        TypeIdentityRenaming, Typed, ValueProjection, WhileOperation, WhilePredicate, ZeroOperation, batch,
        differentiate_at,
    };
    use ryft_pjrt::Client;

    use crate::tests::{
        cpu_client, execution_client, replicated_array, values_from_bytes, values_to_bytes, vector_type,
    };
    use crate::{Array, Error, ExecutionError, FromPjrt, Operation, ProgramBuilder, Session, SessionError};

    use super::*;

    /// Returns a mesh with one auto axis `x` of size one over the addressable device of `client` at `device_index`.
    fn mesh_on(client: &Client<'_>, device_index: usize) -> DeviceMesh {
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[device_index]).unwrap();
        DeviceMesh::new(logical_mesh, vec![device]).unwrap()
    }

    /// Uploads `values` as an [`Array`] of element type `data_type` and shape `shape` that is replicated over `mesh`.
    fn array<T: Copy>(
        session: &Session,
        mesh: &DeviceMesh,
        data_type: DataType,
        values: &[T],
        shape: &[usize],
    ) -> Array {
        replicated_array(session, mesh, data_type, shape, &values_to_bytes(values))
    }

    /// Returns the elements of the first addressable shard of the array member of `value`, read back to the host.
    fn read<T: Copy>(value: &Value) -> Vec<T> {
        let Value::Array(array) = value else { panic!("expected an array value but got `{value}`") };
        let shard = array.addressable_shards().next().unwrap();
        let bytes = shard.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
        values_from_bytes::<T>(bytes.as_slice())
    }

    /// Returns the [`Value::Dimension`] that lifting the static first-class dimension `extent` into `session` gives.
    fn dimension(session: &Session, extent: usize) -> Value {
        session.lift(Constant::Dimension(DimensionValue::constant(extent).unwrap())).unwrap()
    }

    /// Returns whether `left` and `right` share one [`Session`].
    fn same_session(left: &Session, right: &Session) -> bool {
        left.is_same_session(right)
    }

    #[test]
    fn test_dimension_new() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let value = DimensionValue::constant(3).unwrap();
        let dimension = Dimension::new(value.clone(), session.clone());
        assert_eq!(dimension.value(), &value);
        assert!(same_session(dimension.session(), &session));
        assert_eq!(dimension.into_value(), value);
    }

    #[test]
    fn test_dimension_display_and_debug() {
        let client = execution_client();
        let value = DimensionValue::constant(3).unwrap();
        let dimension = Dimension::new(value.clone(), Session::builder().client(client).build().unwrap());
        assert_eq!(dimension.to_string(), "3");
        assert_eq!(format!("{dimension:?}"), format!("Dimension({value:?})"));
    }

    #[test]
    fn test_dimension_equality() {
        // Equality ignores the session, which only selects where derived arrays are materialized.
        let client = execution_client();
        let value = DimensionValue::constant(3).unwrap();
        let first = Dimension::new(value.clone(), Session::builder().client(client).build().unwrap());
        let second = Dimension::new(value, Session::builder().client(client).build().unwrap());
        assert_eq!(first, second);
        assert_ne!(
            first,
            Dimension::new(DimensionValue::constant(4).unwrap(), Session::builder().client(client).build().unwrap())
        );
    }

    #[test]
    fn test_dimension_borrow() {
        let client = execution_client();
        let value = DimensionValue::constant(3).unwrap();
        let dimension = Dimension::new(value.clone(), Session::builder().client(client).build().unwrap());
        assert_eq!(Borrow::<DimensionValue>::borrow(&dimension), &value);
    }

    #[test]
    fn test_dimension_type() {
        let client = execution_client();
        let value = DimensionValue::constant(3).unwrap();
        let dimension = Dimension::new(value.clone(), Session::builder().client(client).build().unwrap());
        assert_eq!(dimension.r#type(), value.r#type());
    }

    #[test]
    fn test_dimension_domain() {
        // Dimension arithmetic dispatches through the dimension's session, which evaluates it on the host.
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let two = Dimension::new(DimensionValue::constant(2).unwrap(), session.clone());
        let three = Dimension::new(DimensionValue::constant(3).unwrap(), session.clone());
        assert!(same_session(two.domain().parent(), &session));
        let program_count = session.cached_program_count();
        let five = two.add(&three).unwrap();
        assert_eq!(five.value().extent(), 5);
        assert!(same_session(five.session(), &session));
        assert_eq!(session.cached_program_count(), program_count);
    }

    #[test]
    fn test_dimension_rename_type_identities() {
        let client = execution_client();
        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let source = DimensionVariable::new("source", bounds);
        let target = DimensionVariable::new("target", bounds);
        let value = DimensionValue::new(source.clone().into(), 4).unwrap();
        let dimension = Dimension::new(value, Session::builder().client(client).build().unwrap());
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(source, target.clone()).unwrap();
        let renamed = dimension.rename_type_identities(&renaming).unwrap();
        assert_eq!(renamed.value(), &DimensionValue::new(target.into(), 4).unwrap());
        assert!(same_session(renamed.session(), dimension.session()));
    }

    #[test]
    fn test_dimension_compare() {
        // The predicate is decided on the host, so only the Boolean result is uploaded and no program is compiled.
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let two = Dimension::new(DimensionValue::constant(2).unwrap(), session.clone());
        let three = Dimension::new(DimensionValue::constant(3).unwrap(), session.clone());
        let program_count = session.cached_program_count();
        let less_than = two.compare(&three, ComparisonDirection::LessThan).unwrap();
        assert_eq!(read::<u8>(&less_than), vec![1]);
        assert!(same_session(less_than.session(), &session));
        assert_eq!(read::<u8>(&two.compare(&three, ComparisonDirection::Equal).unwrap()), vec![0]);
        assert_eq!(read::<u8>(&three.compare(&three, ComparisonDirection::GreaterThanOrEqual).unwrap()), vec![1]);
        assert_eq!(session.cached_program_count(), program_count);
    }

    #[test]
    fn test_reference_new() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let handle = ArrayReference::new(array(&session, &mesh, DataType::F32, &[1.0f32], &[]));
        let reference = Reference::new(handle.clone(), session.clone());
        assert_eq!(reference.handle(), &handle);
        assert!(same_session(reference.session(), &session));
        assert_eq!(reference.into_handle(), handle);
    }

    #[test]
    fn test_reference_display_and_debug() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let handle = ArrayReference::new(array(&session, &mesh, DataType::F32, &[1.0f32], &[]));
        let reference = Reference::new(handle.clone(), session);
        assert_eq!(reference.to_string(), handle.to_string());
        assert_eq!(format!("{reference:?}"), format!("Reference({handle:?})"));
    }

    #[test]
    fn test_reference_equality() {
        // References compare by handle identity, regardless of their sessions.
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let handle = ArrayReference::new(array(&session, &mesh, DataType::F32, &[1.0f32], &[]));
        let reference = Reference::new(handle.clone(), session.clone());
        assert_eq!(reference, Reference::new(handle, Session::builder().client(client).build().unwrap()));
        let other = ArrayReference::new(array(&session, &mesh, DataType::F32, &[1.0f32], &[]));
        assert_ne!(reference, Reference::new(other, session));
    }

    #[test]
    fn test_reference_type() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let handle = ArrayReference::new(array(&session, &mesh, DataType::F32, &[1.0f32], &[]));
        let reference = Reference::new(handle.clone(), session);
        assert_eq!(reference.r#type(), handle.r#type());
    }

    #[test]
    fn test_value_from_array_ir_value() {
        // Array members keep their own sessions, while dimension and reference members bind to the given session.
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let other_session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let array = array(&session, &mesh, DataType::F32, &[1.0f32], &[]);
        let handle = ArrayReference::new(array.clone());
        let extent = DimensionValue::constant(3).unwrap();
        let array_value = Value::from_array_ir_value(ArrayIrValue::Array(array.clone()), &other_session);
        let dimension_value = Value::from_array_ir_value(ArrayIrValue::Dimension(extent.clone()), &other_session);
        let reference_value = Value::from_array_ir_value(ArrayIrValue::Reference(handle.clone()), &other_session);
        assert_eq!(array_value, Value::Array(array));
        assert_eq!(dimension_value, Value::Dimension(Dimension::new(extent, other_session.clone())));
        assert_eq!(reference_value, Value::Reference(Reference::new(handle, other_session.clone())));
        assert!(same_session(array_value.session(), &session));
        assert!(same_session(dimension_value.session(), &other_session));
        assert!(same_session(reference_value.session(), &other_session));
    }

    #[test]
    fn test_value_session() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let array = array(&session, &mesh, DataType::F32, &[1.0f32], &[]);
        let reference = Reference::new(ArrayReference::new(array.clone()), session.clone());
        assert!(same_session(Value::Array(array).session(), &session));
        assert!(same_session(dimension(&session, 3).session(), &session));
        assert!(same_session(Value::Reference(reference).session(), &session));
    }

    #[test]
    fn test_value_into_array_ir_value() {
        // The conversion keeps every member and drops the sessions of dimension and reference members.
        // The `From` conversion into `ArrayIrValue` performs the same conversion.
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let array = array(&session, &mesh, DataType::F32, &[1.0f32], &[]);
        let handle = ArrayReference::new(array.clone());
        let extent = DimensionValue::constant(3).unwrap();
        let values = [
            Value::Array(array.clone()),
            Value::Dimension(Dimension::new(extent.clone(), session.clone())),
            Value::Reference(Reference::new(handle.clone(), session)),
        ];
        let expected = [ArrayIrValue::Array(array), ArrayIrValue::Dimension(extent), ArrayIrValue::Reference(handle)];
        assert_eq!(values.clone().map(Value::into_array_ir_value), expected);
        assert_eq!(values.map(ArrayIrValue::from), expected);
    }

    #[test]
    fn test_value_into_constant() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let array = Array::from_host_buffer(&session, ArrayType::scalar(DataType::F32), None, None, 1f32.to_ne_bytes())
            .unwrap();

        // Arrays are retained by their session and become handles to their retained storage.
        let Ok(Constant::Session(handle)) = Value::Array(array.clone()).into_constant() else {
            panic!("an array was not converted into a session constant");
        };
        assert_eq!(handle.to_array(&session).unwrap(), array);

        // Dimensions become immediate constants, also through `TryFrom`, and references are rejected.
        let extent = DimensionValue::constant(2).unwrap();
        let dimension = Value::Dimension(Dimension::new(extent.clone(), session.clone()));
        assert_eq!(dimension.clone().into_constant(), Ok(Constant::Dimension(extent.clone())));
        assert_eq!(Constant::try_from(dimension), Ok(Constant::Dimension(extent)));
        assert_eq!(
            Value::Reference(Reference::new(ArrayReference::new(array), session)).into_constant(),
            Err(TypeError::invalid("cannot import an XLA reference into a trace whose capture table holds constants")
                .into()),
        );
    }

    #[test]
    fn test_value_display_and_debug() {
        // Values render exactly like the corresponding `ArrayIrValue` members.
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let array = array(&session, &mesh, DataType::F32, &[1.0f32], &[]);
        let reference = Reference::new(ArrayReference::new(array.clone()), session.clone());
        for value in [Value::Array(array), dimension(&session, 3), Value::Reference(reference)] {
            assert_eq!(value.to_string(), value.clone().into_array_ir_value().to_string());
        }
        let value = DimensionValue::constant(3).unwrap();
        assert_eq!(format!("{:?}", dimension(&session, 3)), format!("Dimension(Dimension({value:?}))"));
    }

    #[test]
    fn test_value_type() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let array = array(&session, &mesh, DataType::F32, &[1.0f32], &[]);
        let reference = Reference::new(ArrayReference::new(array.clone()), session.clone());
        for value in [Value::Array(array), dimension(&session, 3), Value::Reference(reference)] {
            assert_eq!(value.r#type(), value.clone().into_array_ir_value().r#type());
        }
    }

    #[test]
    fn test_value_placement() {
        // Only array members live on devices, so only they have a placement.
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let array = array(&session, &mesh, DataType::F32, &[1.0f32], &[]);
        let reference = Value::Reference(Reference::new(ArrayReference::new(array.clone()), session.clone()));
        assert_eq!(
            Value::Array(array).placement(),
            Some(Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 0)).unwrap()),
        );
        assert_eq!(dimension(&session, 3).placement(), None);
        assert_eq!(reference.placement(), None);
    }

    #[test]
    fn test_value_domain() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let value = Value::Array(array(&session, &mesh, DataType::F32, &[1.0f32], &[]));
        assert!(same_session(&value.domain(), &session));
        assert!(same_session(&dimension(&session, 3).domain(), &session));

        // A session without a mesh that is recovered from an array adopts that array's mesh, while a session with a
        // mesh keeps its own.
        assert_eq!(value.domain().device_mesh().unwrap(), &mesh);
        assert!(matches!(
            dimension(&session, 3).domain().device_mesh(),
            Err(Error::Session(SessionError::MissingMesh)),
        ));
        let other_mesh = mesh_on(client, 1);
        assert_ne!(other_mesh, mesh);
        let meshed_session = session.with_device_mesh(other_mesh.clone());
        let meshed_array = Value::Array(array(&meshed_session, &mesh, DataType::F32, &[1.0f32], &[]));
        assert_eq!(meshed_array.domain().device_mesh().unwrap(), &other_mesh);
    }

    #[test]
    fn test_value_domain_places_dimension_derived_arrays_on_the_source_mesh() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 1);
        let input = Value::Array(array(&session, &mesh, DataType::F32, &[1.0f32, 2.0, 3.0], &[3]));
        let size = input.dimension_size(0).unwrap();
        let Value::Array(scalar) = size.to_scalar().unwrap() else { panic!("expected an array") };
        assert_eq!(scalar.mesh(), &mesh);
        let Value::Array(comparison) = size.less_than(&dimension(&session, 4)).unwrap() else {
            panic!("expected an array")
        };
        assert_eq!(comparison.mesh(), &mesh);
    }

    #[test]
    fn test_value_domain_compares_dimensions() {
        // Comparing first-class dimensions executes eagerly outside any context, without compiling a program, and
        // materializes the Boolean result in the session of the inputs.
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let (two, three) = (dimension(&session, 2), dimension(&session, 3));
        let program_count = session.cached_program_count();
        assert_eq!(read::<u8>(&two.less_than(&three).unwrap()), vec![1]);
        assert_eq!(read::<u8>(&two.equal(&three).unwrap()), vec![0]);
        assert_eq!(read::<u8>(&three.greater_than_or_equal(&three).unwrap()), vec![1]);
        assert_eq!(session.cached_program_count(), program_count);

        // A dimension cannot be compared with an array.
        let scalar = Value::Array(array(&session, &mesh, DataType::I64, &[2i64], &[]));
        assert_eq!(
            two.less_than(&scalar),
            Err(ProgramError::Type(TypeError::invalid(
                "`compare` inputs must both be arrays or both be first-class dimensions, but got `dimension<2>` and \
                 `i64[][sharding={mesh<['x'=1:auto]>, []}]`",
            ))),
        );
    }

    #[test]
    fn test_value_domain_evaluates_dimension_arithmetic() {
        // Arithmetic on two first-class dimensions uses their dimension members and executes eagerly outside any
        // context, without compiling a program.
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let (two, three) = (dimension(&session, 2), dimension(&session, 3));
        let program_count = session.cached_program_count();
        assert_eq!(two.add(&three), Ok(dimension(&session, 5)));
        assert_eq!(three.sub(&two), Ok(dimension(&session, 1)));
        assert_eq!(two.saturating_sub(&three), Ok(dimension(&session, 0)));
        assert_eq!(two.mul(&three), Ok(dimension(&session, 6)));
        assert_eq!(three.div(&two), Ok(dimension(&session, 1)));
        assert_eq!(three.rem(&two), Ok(dimension(&session, 1)));
        assert_eq!(two.pow(&three), Ok(dimension(&session, 8)));
        assert_eq!(two.min(&three), Ok(dimension(&session, 2)));
        assert_eq!(two.max(&three), Ok(dimension(&session, 3)));
        assert_eq!(session.cached_program_count(), program_count);

        // A dimension paired with an array uses the array members, which rejects the dimension.
        let scalar = Value::Array(array(&session, &mesh, DataType::I64, &[2i64], &[]));
        assert_eq!(
            two.mul(&scalar),
            Err(ProgramError::Type(TypeError::invalid("expected array type but got dimension type"))),
        );
    }

    #[test]
    fn test_value_domain_converts_dimensions_to_scalars() {
        // The session uploads the host extent as an `i64` scalar without compiling a program.
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let program_count = session.cached_program_count();
        let scalar = dimension(&session, 2).to_scalar().unwrap();
        assert_eq!(scalar.r#type().to_string(), "SCALAR_TYPE_PLACEHOLDER");
        assert_eq!(read::<i64>(&scalar), vec![2]);
        assert_eq!(session.cached_program_count(), program_count);
    }

    #[test]
    fn test_value_domain_executes_reference_operations() {
        // Top-level reference operations execute eagerly on their host-side handles, with the validation and locking of
        // the generic array IR reference implementations.
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let vector = |values: &[f32]| Value::Array(array(&session, &mesh, DataType::F32, values, &[values.len()]));
        let scalar = |value: f32| Value::Array(array(&session, &mesh, DataType::F32, &[value], &[]));

        let reference = vector(&[1.0, 2.0]).reference_new().unwrap();
        assert!(same_session(reference.session(), &session));
        assert_eq!(read::<f32>(&reference.read().unwrap()), vec![1.0, 2.0]);
        reference.write(&vector(&[3.0, 4.0])).unwrap();
        assert_eq!(read::<f32>(&reference.swap(&vector(&[5.0, 6.0])).unwrap()), vec![3.0, 4.0]);
        reference.add_update(&vector(&[1.0, 1.0])).unwrap();
        assert_eq!(read::<f32>(&reference.read().unwrap()), vec![6.0, 7.0]);

        // Views address parts of the referent.
        let second = [ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) }];
        assert_eq!(read::<f32>(&reference.read_through(&second, &[]).unwrap()), vec![7.0]);
        reference.write_through(&scalar(9.0), &second, &[]).unwrap();
        assert_eq!(read::<f32>(&reference.read().unwrap()), vec![6.0, 9.0]);

        // Mismatched replacements are rejected, and freezing invalidates every alias.
        assert_eq!(
            reference.write(&scalar(1.0)),
            Err(ProgramError::Type(TypeError::invalid(
                "`reference_write` replacement type `f32[][sharding={mesh<['x'=1:auto]>, []}]` must exactly match \
                 reference referent type `f32[2][sharding={mesh<['x'=1:auto]>, [{}]}]`",
            ))),
        );
        let alias = reference.clone();
        assert_eq!(read::<f32>(&reference.freeze().unwrap()), vec![6.0, 9.0]);
        assert_eq!(alias.read().unwrap_err().downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
        assert_eq!(
            alias.write(&vector(&[1.0, 1.0])).unwrap_err().downcast_custom::<ReferenceError>(),
            Some(&ReferenceError::Frozen),
        );

        // Reference operands of array operations are rejected by the array projection of the array IR capabilities.
        let reference = vector(&[1.0, 2.0]).reference_new().unwrap();
        assert_eq!(
            reference.add(&vector(&[1.0, 2.0])),
            Err(ProgramError::Type(TypeError::invalid("expected array type but got reference type"))),
        );
    }

    #[test]
    fn test_value_domain_projects_capabilities_onto_arrays() {
        // Capabilities without a receiver-shaped signature project onto the array member through the generalized
        // array IR implementations in `ryft-core`.
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let input = Value::Array(array(&session, &mesh, DataType::F32, &[3.0f32, 1.0, 2.0], &[3]));
        let sorted = Value::sort(std::slice::from_ref(&input), 0, SortDirection::Ascending).unwrap();
        assert_eq!(read::<f32>(&sorted[0]), vec![1.0, 2.0, 3.0]);
        let indices = Value::Array(array(&session, &mesh, DataType::I32, &[2i32, 0], &[2]));
        assert_eq!(read::<f32>(&input.gather_axis(&indices, 0, GatherMode::Clip).unwrap()), vec![2.0, 3.0]);
        assert_eq!(
            input.parallel_permute("missing", vec![(0, 0)]),
            Err(ProgramError::Axis(AxisError::UnboundAxisName { name: "missing".to_string() })),
        );
    }

    #[test]
    fn test_value_domain_executes_assertions() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let truth = Value::Array(array(&session, &mesh, DataType::Boolean, &[1u8], &[]));
        let falsity = Value::Array(array(&session, &mesh, DataType::Boolean, &[0u8], &[]));
        assert_eq!(truth.assert("predicate must hold", &[]), Ok(()));
        let message = "assertion failed: predicate must hold";

        // Inline execution can report the failure from the call. An asynchronous submission reports it at the
        // effect barrier because the assertion has no result array carrying its completion.
        match falsity.assert("predicate must hold", &[]) {
            Err(error) => assert_eq!(error, ProgramError::InvalidArgument { message: message.to_string() }),
            Ok(()) => assert!(matches!(
                session.effects_barrier(),
                Err(Error::Execution(ExecutionError::EffectSubmissionFailures { failures }))
                    if matches!(failures.as_slice(), [(_, failure)] if failure.as_ref() == message),
            )),
        }
    }

    #[test]
    fn test_value_domain_recovered_by_batch() {
        // Free transform entry points recover the session from the values, so no explicit context is needed.
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let input = Value::Array(array(&session, &mesh, DataType::F32, &[1.0f32, 2.0, 3.0], &[3]));
        let squared: Value = batch(|x| Ok(x.clone() * x), input, BatchAxis::new(0), BatchAxis::new(0), None).unwrap();
        assert_eq!(read::<f32>(&squared), vec![1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_value_domain_recovered_by_jvp() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let input = Value::Array(array(&session, &mesh, DataType::F32, &[1.0f32, 2.0, 3.0], &[3]));
        let (value, derivative) = differentiate_at(input.clone()).jvp(input, |x| Ok(x.clone() * x)).unwrap();
        assert_eq!(read::<f32>(&value), vec![1.0, 4.0, 9.0]);
        assert_eq!(read::<f32>(&derivative), vec![2.0, 8.0, 18.0]);
    }

    #[test]
    fn test_value_domain_recovered_by_value_and_gradient() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let input = Value::Array(array(&session, &mesh, DataType::F32, &[1.0f32, 2.0, 3.0], &[3]));
        let (value, gradient) = differentiate_at(input)
            .value_and_gradient(|x| (x.clone() * x).reduce(&[0], ReductionKind::Sum).unwrap())
            .unwrap();
        assert_eq!(read::<f32>(&value), vec![14.0]);
        assert_eq!(read::<f32>(&gradient), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_value_domain_replays_dimension_derived_values_under_batch() {
        // Each item is scaled by its own folded extent, which `batch` replays as a dimension constant.
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let input = Value::Array(array(&session, &mesh, DataType::F32, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let scaled: Value = batch(
            |x| {
                let extent = x.dimension_size(0)?.to_scalar()?.convert_element_type(DataType::F32)?;
                Ok(x * extent)
            },
            input,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(read::<f32>(&scaled), vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0]);
    }

    #[test]
    fn test_value_domain_replays_dimension_derived_values_under_jvp() {
        // The primal is scaled by its folded extent, which `jvp` replays as a dimension constant.
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let primal = Value::Array(array(&session, &mesh, DataType::F32, &[1.0f32, 2.0, 3.0], &[3]));
        let tangent = Value::Array(array(&session, &mesh, DataType::F32, &[1.0f32, 1.0, 1.0], &[3]));
        let (value, derivative) = differentiate_at(primal)
            .jvp(tangent, |x| {
                let extent = x.dimension_size(0)?.to_scalar()?.convert_element_type(DataType::F32)?;
                Ok(x * extent)
            })
            .unwrap();
        assert_eq!(read::<f32>(&value), vec![3.0, 6.0, 9.0]);
        assert_eq!(read::<f32>(&derivative), vec![3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_value_program_replay_lifts_dimension_constants() {
        // Replaying a program over the session lifts its dimension constants through the session-aware `lift`.
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let input = array(&session, &mesh, DataType::F32, &[1.0f32, 2.0, 3.0], &[3]);

        // `dimension_size(x, 0) + 2` followed by `dimension_to_scalar`, with the `2` stored as a program constant.
        let mut builder = ProgramBuilder::new();
        let x = builder.add_input(Type::Array(input.r#type().into_owned()));
        let size_operation = DimensionSizeOperation::new(input.r#type().as_ref(), 0).unwrap();
        let size_type = size_operation.output_type().clone();
        let size = builder.add_instruction(size_operation, Vec::new(), vec![x], None).unwrap()[0];
        let constant = DimensionValue::constant(2).unwrap();
        let sum = DimensionAddOperation::new(&size_type, constant.r#type().as_ref()).unwrap();
        let two = builder.add_constant(Constant::Dimension(constant));
        let sum = builder
            .add_instruction(Operation::Dimension(DimensionOperation::Add(sum)), Vec::new(), vec![size, two], None)
            .unwrap()[0];
        let output = builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![sum], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let outputs = program.interpret_in_context(&session, vec![Value::Array(input)]).unwrap();
        assert_eq!(read::<i64>(&outputs[0]), vec![5]);
    }

    #[test]
    fn test_value_relocatable_type() {
        // An uncommitted array whose type records nothing beyond replication is relocatable under its type without a
        // sharding. Committed arrays, dimensions, and references are not relocatable.
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let scalar_type = ArrayType::scalar(DataType::F32);
        let uncommitted =
            Array::from_host_buffer(&session, scalar_type.clone(), None, None, 1f32.to_ne_bytes()).unwrap();
        let committed = array(&session, &mesh, DataType::F32, &[1.0f32], &[]);
        let reference = Value::Reference(Reference::new(ArrayReference::new(committed.clone()), session.clone()));
        assert_eq!(Value::Array(uncommitted).relocatable_type(), Some(Type::Array(scalar_type)));
        assert_eq!(Value::Array(committed).relocatable_type(), None);
        assert_eq!(dimension(&session, 3).relocatable_type(), None);
        assert_eq!(reference.relocatable_type(), None);
    }

    #[test]
    fn test_value_rename_type_identities() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let source = DimensionVariable::new("source", bounds);
        let target = DimensionVariable::new("target", bounds);
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(source.clone(), target.clone()).unwrap();

        // Dimension members rename their types and keep their sessions.
        let value = Value::Dimension(Dimension::new(DimensionValue::new(source.into(), 4).unwrap(), session.clone()));
        let renamed = value.rename_type_identities(&renaming).unwrap();
        assert_eq!(
            renamed,
            Value::Dimension(Dimension::new(DimensionValue::new(target.into(), 4).unwrap(), session.clone()))
        );
        assert!(same_session(renamed.session(), &session));

        // Static array and reference members have no identities to rename, so they keep their storage and handles.
        let array = array(&session, &mesh, DataType::F32, &[1.0f32], &[]);
        let reference = Value::Reference(Reference::new(ArrayReference::new(array.clone()), session.clone()));
        assert_eq!(Value::Array(array.clone()).rename_type_identities(&renaming), Ok(Value::Array(array)));
        let renamed = reference.rename_type_identities(&renaming).unwrap();
        assert_eq!(renamed, reference);
        assert!(same_session(renamed.session(), &session));
    }

    #[test]
    fn test_value_validate_as_constant() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let array = array(&session, &mesh, DataType::F32, &[1.0f32], &[]);
        let reference = Value::Reference(Reference::new(ArrayReference::new(array.clone()), session.clone()));
        assert_eq!(Value::Array(array).validate_as_constant(), Ok(()));
        assert_eq!(dimension(&session, 3).validate_as_constant(), Ok(()));
        assert_eq!(
            reference.validate_as_constant(),
            Err(TypeError::invalid(
                "reference values cannot be stored as program constants; pass external references through program \
                 inputs or captures instead",
            )),
        );
    }

    #[test]
    fn test_value_is_zero() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let array = array(&session, &mesh, DataType::F32, &[0.0f32], &[]);
        let reference = Value::Reference(Reference::new(ArrayReference::new(array.clone()), session.clone()));
        assert!(!Value::Array(array).is_zero());
        assert!(!dimension(&session, 0).is_zero());
        assert!(!reference.is_zero());
    }

    #[test]
    fn test_value_singleton() {
        // A static function cannot attach a session, so even a singleton dimension type has no singleton value.
        let singleton_type = DimensionValue::constant(3).unwrap().r#type().into_owned();
        assert_eq!(Value::singleton(&Type::Dimension(singleton_type)), None);
        assert_eq!(Value::singleton(&Type::Array(ArrayType::scalar(DataType::F32))), None);
    }

    #[test]
    fn test_value_reference_id() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let array = array(&session, &mesh, DataType::F32, &[1.0f32], &[]);
        let handle = ArrayReference::new(array.clone());
        let reference = Value::Reference(Reference::new(handle.clone(), session.clone()));
        assert_eq!(Value::Array(array).reference_id(), None);
        assert_eq!(dimension(&session, 3).reference_id(), None);
        assert_eq!(reference.reference_id(), Some(handle.id()));
    }

    #[test]
    fn test_value_concretize() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let predicate = Value::Array(array(&session, &mesh, DataType::Boolean, &[1u8], &[]));
        assert_eq!(predicate.concretize(), Ok(true));
        assert!(matches!(
            dimension(&session, 3).concretize(),
            Err(ProgramError::Concretization { message })
                if message == "cannot extract a concrete boolean from a first-class dimension `3`",
        ));
    }

    #[test]
    fn test_value_any_true() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        assert_eq!(Value::Array(array(&session, &mesh, DataType::Boolean, &[0u8, 1], &[2])).any_true(), Ok(true));
        assert_eq!(Value::Array(array(&session, &mesh, DataType::Boolean, &[0u8, 0], &[2])).any_true(), Ok(false));
        assert!(matches!(
            dimension(&session, 3).any_true(),
            Err(ProgramError::Concretization { message })
                if message == "cannot use first-class dimension `3` as a `while` predicate",
        ));
    }

    #[test]
    fn test_value_any_true_decides_a_data_dependent_while_under_jvp() {
        // At concrete primals, the forward mode `while` rule decides the loop condition on the host through the
        // predicate contracts of `Value` and unrolls the executed iterations.
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let scalar_type = Type::Array(ArrayType::scalar(DataType::F32));
        let condition = {
            let mut builder = ProgramBuilder::new();
            let state = builder.add_input(scalar_type.clone());
            let zero = Operation::provide(ZeroOperation::new(scalar_type.clone()), &[]).unwrap();
            let zero = builder.add_instruction(zero, Vec::new(), Vec::new(), None).unwrap()[0];
            let comparison = CompareOperation::<ArrayType>::new(ComparisonDirection::GreaterThan);
            let predicate = builder
                .add_instruction(Operation::Array(comparison.into()), Vec::new(), vec![state, zero], None)
                .unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![predicate], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::new();
            let state = builder.add_input(scalar_type.clone());
            let one = Operation::provide(OneOperation::new(scalar_type.clone()), &[]).unwrap();
            let one = builder.add_instruction(one, Vec::new(), Vec::new(), None).unwrap()[0];
            let subtraction = SubOperation::<ArrayType>::new();
            let next = builder
                .add_instruction(Operation::Array(subtraction.into()), Vec::new(), vec![state, one], None)
                .unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![next], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let primal = Value::Array(array(&session, &mesh, DataType::F32, &[3.5f32], &[]));
        let tangent = Value::Array(array(&session, &mesh, DataType::F32, &[2.0f32], &[]));
        let (value, derivative) = differentiate_at(primal)
            .jvp(tangent, |state| {
                let mut outputs = state.domain().bind(
                    Operation::While(WhileOperation::new()),
                    vec![condition.clone(), body.clone()],
                    std::slice::from_ref(&state),
                )?;
                Ok(outputs.remove(0))
            })
            .unwrap();
        assert_eq!(read::<f32>(&value), vec![-0.5]);
        assert_eq!(read::<f32>(&derivative), vec![2.0]);
    }

    #[test]
    fn test_value_select() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);

        // A scalar predicate selects whole values.
        let falsity = Value::Array(array(&session, &mesh, DataType::Boolean, &[0u8], &[]));
        let one = Value::Array(array(&session, &mesh, DataType::F32, &[1.0f32], &[]));
        let two = Value::Array(array(&session, &mesh, DataType::F32, &[2.0f32], &[]));
        assert_eq!(read::<f32>(&falsity.select(&one, &two).unwrap()), vec![2.0]);

        // A batched predicate, whose shape is a prefix of the selected values' shape, selects per item. Item 0 keeps
        // the first row of `on_false`, and item 1 takes the second row of `on_true`.
        let mixed = Value::Array(array(&session, &mesh, DataType::Boolean, &[0u8, 1], &[2]));
        let on_true = Value::Array(array(&session, &mesh, DataType::F32, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2]));
        let on_false = Value::Array(array(&session, &mesh, DataType::F32, &[5.0f32, 6.0, 7.0, 8.0], &[2, 2]));
        assert_eq!(read::<f32>(&mixed.select(&on_true, &on_false).unwrap()), vec![5.0, 6.0, 3.0, 4.0]);

        // Equal dimension carries pass through with their session, while dimensions are never predicates.
        let three = dimension(&session, 3);
        let selected = mixed.select(&three, &three).unwrap();
        assert_eq!(selected, three);
        assert!(same_session(selected.session(), &session));
        assert!(matches!(
            three.select(&one, &two),
            Err(ProgramError::Concretization { message })
                if message == "cannot use first-class dimension `3` as a `while` predicate",
        ));
    }

    #[test]
    fn test_value_assertion_value() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let scalar = Value::Array(array(&session, &mesh, DataType::I32, &[42i32], &[]));
        assert_eq!(scalar.assertion_observation(), Ok("42".to_string()));
        assert_eq!(scalar.assertion_array(), Ok(Some(CoreArray::scalar(42i32).unwrap())));
        assert_eq!(dimension(&session, 3).assertion_observation(), Ok("3".to_string()));
        assert_eq!(dimension(&session, 3).assertion_array(), Ok(None));
    }

    #[test]
    fn test_value_array_projection() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let array = array(&session, &mesh, DataType::F32, &[1.0f32], &[]);
        let value = <Value as ValueProjection<ArrayType>>::from_projected(array.clone());
        assert_eq!(<Value as ValueProjection<ArrayType>>::projected(&value), Ok(&array));
        assert_eq!(ValueProjection::<ArrayType>::into_projected(value), Ok(array));
        assert_eq!(
            <Value as ValueProjection<ArrayType>>::projected(&dimension(&session, 3)),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
    }

    #[test]
    fn test_value_dimension_projection() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let dimension = Dimension::new(DimensionValue::constant(3).unwrap(), session.clone());
        let value = <Value as ValueProjection<DimensionType>>::from_projected(dimension.clone());
        assert_eq!(<Value as ValueProjection<DimensionType>>::projected(&value), Ok(&dimension));
        assert_eq!(ValueProjection::<DimensionType>::into_projected(value), Ok(dimension));
        let array = Value::Array(array(&session, &mesh, DataType::F32, &[1.0f32], &[]));
        assert_eq!(
            <Value as ValueProjection<DimensionType>>::projected(&array),
            Err(TypeError::invalid("expected dimension type but got array type")),
        );
    }

    #[test]
    fn test_value_reference_projection() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let array = array(&session, &mesh, DataType::F32, &[1.0f32], &[]);
        let reference = Reference::new(ArrayReference::new(array.clone()), session.clone());
        let value = <Value as ValueProjection<ReferenceType<ArrayType>>>::from_projected(reference.clone());
        assert_eq!(<Value as ValueProjection<ReferenceType<ArrayType>>>::projected(&value), Ok(&reference));
        assert_eq!(ValueProjection::<ReferenceType<ArrayType>>::into_projected(value), Ok(reference));
        assert_eq!(
            <Value as ValueProjection<ReferenceType<ArrayType>>>::projected(&Value::Array(array)),
            Err(TypeError::invalid("expected reference type but got array type")),
        );
    }

    #[test]
    fn test_value_from() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let array = array(&session, &mesh, DataType::F32, &[1.0f32], &[]);
        let dimension = Dimension::new(DimensionValue::constant(3).unwrap(), session.clone());
        let reference = Reference::new(ArrayReference::new(array.clone()), session);
        assert_eq!(Value::from(array.clone()), Value::Array(array));
        assert_eq!(Value::from(dimension.clone()), Value::Dimension(dimension));
        assert_eq!(Value::from(reference.clone()), Value::Reference(reference));
    }

    #[test]
    fn test_value_neg() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let two = Value::Array(array(&session, &mesh, DataType::F32, &[2.0f32], &[]));
        assert_eq!(read::<f32>(&-two), vec![-2.0]);
    }

    #[test]
    fn test_value_add() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let two = Value::Array(array(&session, &mesh, DataType::F32, &[2.0f32], &[]));
        let four = Value::Array(array(&session, &mesh, DataType::F32, &[4.0f32], &[]));
        assert_eq!(read::<f32>(&(two + four)), vec![6.0]);
    }

    #[test]
    fn test_value_sub() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let two = Value::Array(array(&session, &mesh, DataType::F32, &[2.0f32], &[]));
        let four = Value::Array(array(&session, &mesh, DataType::F32, &[4.0f32], &[]));
        assert_eq!(read::<f32>(&(two - four)), vec![-2.0]);
    }

    #[test]
    fn test_value_mul() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let two = Value::Array(array(&session, &mesh, DataType::F32, &[2.0f32], &[]));
        let four = Value::Array(array(&session, &mesh, DataType::F32, &[4.0f32], &[]));
        assert_eq!(read::<f32>(&(two * four)), vec![8.0]);
    }

    #[test]
    fn test_value_div() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let two = Value::Array(array(&session, &mesh, DataType::F32, &[2.0f32], &[]));
        let four = Value::Array(array(&session, &mesh, DataType::F32, &[4.0f32], &[]));
        assert_eq!(read::<f32>(&(four / two)), vec![2.0]);
    }

    #[test]
    fn test_value_rem() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let two = Value::Array(array(&session, &mesh, DataType::F32, &[2.0f32], &[]));
        let four = Value::Array(array(&session, &mesh, DataType::F32, &[4.0f32], &[]));
        assert_eq!(read::<f32>(&(two % four)), vec![2.0]);

        // Two dimension members produce a dimension.
        assert_eq!(dimension(&session, 3) % dimension(&session, 2), dimension(&session, 1));
    }

    #[test]
    fn test_value_not() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let mixed = Value::Array(array(&session, &mesh, DataType::Boolean, &[1u8, 0, 1, 0], &[4]));
        assert_eq!(read::<u8>(&!mixed), [0, 1, 0, 1]);
    }

    #[test]
    fn test_value_bitand() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let truths = Value::Array(array(&session, &mesh, DataType::Boolean, &[1u8, 1, 0, 0], &[4]));
        let mixed = Value::Array(array(&session, &mesh, DataType::Boolean, &[1u8, 0, 1, 0], &[4]));
        assert_eq!(read::<u8>(&(truths & mixed)), [1, 0, 0, 0]);
    }

    #[test]
    fn test_value_bitor() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let truths = Value::Array(array(&session, &mesh, DataType::Boolean, &[1u8, 1, 0, 0], &[4]));
        let mixed = Value::Array(array(&session, &mesh, DataType::Boolean, &[1u8, 0, 1, 0], &[4]));
        assert_eq!(read::<u8>(&(truths | mixed)), [1, 1, 1, 0]);
    }

    #[test]
    fn test_value_bitxor() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = mesh_on(client, 0);
        let truths = Value::Array(array(&session, &mesh, DataType::Boolean, &[1u8, 1, 0, 0], &[4]));
        let mixed = Value::Array(array(&session, &mesh, DataType::Boolean, &[1u8, 0, 1, 0], &[4]));
        assert_eq!(read::<u8>(&(truths ^ mixed)), [0, 1, 1, 0]);
    }

    #[test]
    fn test_constant_display() {
        // Constants render exactly like the payload that they wrap.
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let extent = DimensionValue::new(extent_type, 4).unwrap();
        assert_eq!(
            Constant::Captured(CaptureReference::new(2, Type::Array(vector_type()))).to_string(),
            "capture#2:f64[4]"
        );
        assert_eq!(Constant::Dimension(extent).to_string(), "EXTENT_PLACEHOLDER");
        assert_eq!(Constant::Boolean(true).to_string(), "true");
    }

    #[test]
    fn test_constant_type() {
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let extent = DimensionValue::new(extent_type, 4).unwrap();
        assert_eq!(
            Constant::Captured(CaptureReference::new(2, Type::Array(vector_type()))).r#type().into_owned(),
            Type::Array(vector_type()),
        );
        assert_eq!(
            Constant::Dimension(extent.clone()).r#type().into_owned(),
            Type::Dimension(extent.r#type().into_owned()),
        );
        assert_eq!(Constant::Boolean(true).r#type().into_owned(), Type::Array(ArrayType::scalar(DataType::Boolean)));
    }

    #[test]
    fn test_constant_rename_type_identities() {
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let source = DimensionVariable::new("source", bounds);
        let target = DimensionVariable::new("target", bounds);
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(source.clone(), target.clone()).unwrap();
        let reference_type = |variable| {
            Type::Reference(ReferenceType::new(ArrayType::new(
                DataType::F32,
                Shape::new(vec![CoreDimension::Dynamic(variable)]),
            )))
        };

        // A capture renames the identities of its structural type, including a reference type, and keeps its capture
        // table index.
        assert_eq!(
            Constant::Captured(CaptureReference::new(7, reference_type(source))).rename_type_identities(&renaming),
            Ok(Constant::Captured(CaptureReference::new(7, reference_type(target)))),
        );
        assert_eq!(Constant::Boolean(true).rename_type_identities(&renaming), Ok(Constant::Boolean(true)));
    }

    #[test]
    fn test_constant_capture_index() {
        // Only a capture names a capture table slot.
        assert_eq!(Constant::Captured(CaptureReference::new(2, Type::Array(vector_type()))).capture_index(), Some(2));
        assert_eq!(Constant::Dimension(DimensionValue::constant(4).unwrap()).capture_index(), None);
        assert_eq!(Constant::Boolean(true).capture_index(), None);
    }

    #[test]
    fn test_constant_literal_eq_and_literal_hash() {
        // The literal identity of a constant is its ordinary equality and hash.
        let captured = Constant::Captured(CaptureReference::new(2, Type::Array(vector_type())));
        assert!(captured.literal_eq(&Constant::Captured(CaptureReference::new(2, Type::Array(vector_type())))));
        assert!(!captured.literal_eq(&Constant::Captured(CaptureReference::new(3, Type::Array(vector_type())))));
        assert!(!captured.literal_eq(&Constant::Boolean(true)));
        let mut literal_hasher = DefaultHasher::new();
        captured.literal_hash(&mut literal_hasher);
        let mut hasher = DefaultHasher::new();
        captured.hash(&mut hasher);
        assert_eq!(literal_hasher.finish(), hasher.finish());
    }

    #[test]
    fn test_constant_map_capture_index() {
        // Only a capture is renumbered, and immediates pass through every remapping unchanged.
        let captured = Constant::Captured(CaptureReference::new(2, Type::Array(vector_type())));
        let immediate = Constant::Dimension(DimensionValue::constant(4).unwrap());
        assert_eq!(
            captured.map_capture_index(|index| index + 3),
            Constant::Captured(CaptureReference::new(5, Type::Array(vector_type()))),
        );
        assert_eq!(immediate.map_capture_index(|_| 7), immediate);
        assert_eq!(Constant::Boolean(true).map_capture_index(|_| 99), Constant::Boolean(true));
    }

    #[test]
    fn test_constant_concretize() {
        assert_eq!(Constant::Boolean(true).concretize(), Ok(true));
        assert_eq!(Constant::Boolean(false).concretize(), Ok(false));
        assert_eq!(
            Constant::Dimension(DimensionValue::constant(3).unwrap()).concretize(),
            Err(ProgramError::Concretization {
                message: "cannot extract a concrete boolean from the staged XLA constant `3`".to_string(),
            }),
        );
    }

    #[test]
    fn test_constant_assertion_value() {
        let dimension = Constant::Dimension(DimensionValue::constant(3).unwrap());
        assert_eq!(dimension.assertion_array(), Ok(None));
        assert_eq!(dimension.assertion_observation(), Ok("3".to_string()));
        let predicate = Constant::Boolean(true);
        assert_eq!(predicate.assertion_array(), Ok(Some(CoreArray::scalar(true).unwrap())));
        assert_eq!(predicate.assertion_observation(), Ok("true".to_string()));
        let captured = Constant::Captured(CaptureReference::new(2, Type::Array(vector_type())));
        assert_eq!(
            captured.assertion_array(),
            Err(ProgramError::Concretization {
                message: "cannot read an assertion array from a captured constant reference".to_string(),
            }),
        );
        assert_eq!(
            captured.assertion_observation(),
            Err(ProgramError::Concretization {
                message: "cannot read an assertion observation from a captured constant reference".to_string(),
            }),
        );
    }

    #[test]
    fn test_constant_array_projection() {
        let array_type = vector_type();
        let captured = Constant::Captured(CaptureReference::new(2, Type::Array(array_type.clone())));
        let projected = ArrayConstant::Captured(CaptureReference::new(2, array_type.clone()));

        // Array captures and Boolean predicates project onto array constants and back.
        assert_eq!(
            <Constant as ValueProjection<ArrayType>>::projected(&captured).map(|value| value.r#type().into_owned()),
            Ok(array_type.clone()),
        );
        assert_eq!(ValueProjection::<ArrayType>::into_projected(captured.clone()), Ok(projected.clone()));
        assert_eq!(<Constant as ValueProjection<ArrayType>>::from_projected(projected), captured);
        assert_eq!(
            ValueProjection::<ArrayType>::into_projected(Constant::Boolean(true)),
            Ok(ArrayConstant::Boolean(true)),
        );
        assert_eq!(
            <Constant as ValueProjection<ArrayType>>::from_projected(ArrayConstant::Boolean(true)),
            Constant::Boolean(true),
        );

        // Immediate dimensions and reference captures have no array projection.
        assert_eq!(
            ValueProjection::<ArrayType>::into_projected(Constant::Dimension(DimensionValue::constant(4).unwrap())),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
        let reference = Constant::Captured(CaptureReference::new(7, Type::Reference(ReferenceType::new(array_type))));
        assert_eq!(
            <Constant as ValueProjection<ArrayType>>::projected(&reference).map(|value| value.r#type().into_owned()),
            Err(TypeError::invalid("expected array type but got reference type")),
        );
    }

    #[test]
    fn test_constant_dimension_projection() {
        let extent = DimensionValue::constant(4).unwrap();
        let immediate = <Constant as ValueProjection<DimensionType>>::from_projected(extent.clone());
        assert_eq!(immediate, Constant::Dimension(extent.clone()));
        assert_eq!(<Constant as ValueProjection<DimensionType>>::projected(&immediate), Ok(&extent));
        assert_eq!(ValueProjection::<DimensionType>::into_projected(immediate), Ok(extent));

        // Only immediate dimensions project onto dimensions.
        let captured = Constant::Captured(CaptureReference::new(2, Type::Array(vector_type())));
        assert_eq!(
            <Constant as ValueProjection<DimensionType>>::projected(&captured),
            Err(TypeError::invalid("expected an immediate dimension but got a captured value")),
        );
        assert_eq!(
            ValueProjection::<DimensionType>::into_projected(Constant::Boolean(true)),
            Err(TypeError::invalid("expected a dimension but got a Boolean array")),
        );
    }

    #[test]
    fn test_constant_from() {
        let capture = CaptureReference::new(2, Type::Array(vector_type()));
        let extent = DimensionValue::constant(4).unwrap();
        assert_eq!(Constant::from(capture.clone()), Constant::Captured(capture));
        assert_eq!(Constant::from(extent.clone()), Constant::Dimension(extent));
        assert_eq!(Constant::from(true), Constant::Boolean(true));
    }

    #[test]
    fn test_array_constant_display() {
        assert_eq!(ArrayConstant::Captured(CaptureReference::new(2, vector_type())).to_string(), "capture#2:f64[4]");
        assert_eq!(ArrayConstant::Boolean(false).to_string(), "false");
    }

    #[test]
    fn test_array_constant_type() {
        assert_eq!(
            ArrayConstant::Captured(CaptureReference::new(2, vector_type())).r#type().into_owned(),
            vector_type()
        );
        assert_eq!(ArrayConstant::Boolean(true).r#type().into_owned(), ArrayType::scalar(DataType::Boolean));
    }

    #[test]
    fn test_array_constant_rename_type_identities() {
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let source = DimensionVariable::new("source", bounds);
        let target = DimensionVariable::new("target", bounds);
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(source.clone(), target.clone()).unwrap();
        let array_type = |variable| ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Dynamic(variable)]));
        let captured = ArrayConstant::Captured(CaptureReference::new(3, array_type(source)));
        assert_eq!(
            captured.rename_type_identities(&renaming),
            Ok(ArrayConstant::Captured(CaptureReference::new(3, array_type(target)))),
        );
        assert_eq!(ArrayConstant::Boolean(true).rename_type_identities(&renaming), Ok(ArrayConstant::Boolean(true)));
    }

    #[test]
    fn test_array_constant_capture_index() {
        assert_eq!(ArrayConstant::Captured(CaptureReference::new(2, vector_type())).capture_index(), Some(2));
        assert_eq!(ArrayConstant::Boolean(true).capture_index(), None);
    }

    #[test]
    fn test_array_constant_map_capture_index() {
        assert_eq!(
            ArrayConstant::Captured(CaptureReference::new(2, vector_type())).map_capture_index(|index| index + 3),
            ArrayConstant::Captured(CaptureReference::new(5, vector_type())),
        );
        assert_eq!(ArrayConstant::Boolean(true).map_capture_index(|_| 99), ArrayConstant::Boolean(true));
    }

    #[test]
    fn test_array_constant_concretize() {
        assert_eq!(ArrayConstant::Boolean(true).concretize(), Ok(true));
        assert_eq!(
            ArrayConstant::Captured(CaptureReference::new(2, vector_type())).concretize(),
            Err(ProgramError::Concretization {
                message: "cannot extract a concrete boolean from a captured constant reference".to_string(),
            }),
        );
    }

    #[test]
    fn test_array_constant_assertion_value() {
        let predicate = ArrayConstant::Boolean(false);
        assert_eq!(predicate.assertion_array(), Ok(Some(CoreArray::scalar(false).unwrap())));
        assert_eq!(predicate.assertion_observation(), Ok("false".to_string()));
        let captured = ArrayConstant::Captured(CaptureReference::new(2, vector_type()));
        assert_eq!(
            captured.assertion_array(),
            Err(ProgramError::Concretization {
                message: "cannot inspect a captured assertion array before execution".to_string(),
            }),
        );
        assert_eq!(
            captured.assertion_observation(),
            Err(ProgramError::Concretization {
                message: "cannot inspect a captured assertion observation before execution".to_string(),
            }),
        );
    }

    #[test]
    fn test_array_constant_array_projection() {
        let array_type = vector_type().with_memory(Memory::Host { pinned: true });
        let captured = ArrayConstant::Captured(CaptureReference::new(2, array_type.clone()));
        let borrowed = ValueProjection::<ArrayType>::projected(&captured).unwrap();
        assert!(std::ptr::eq(borrowed, &captured));
        assert_eq!(borrowed.r#type().as_ref(), &array_type);
        let owned = ValueProjection::<ArrayType>::into_projected(captured.clone()).unwrap();
        assert_eq!(owned, captured);
        assert_eq!(<ArrayConstant as ValueProjection<ArrayType>>::from_projected(owned), captured);

        let predicate = ArrayConstant::Boolean(true);
        let borrowed = ValueProjection::<ArrayType>::projected(&predicate).unwrap();
        assert!(std::ptr::eq(borrowed, &predicate));
        let owned = ValueProjection::<ArrayType>::into_projected(predicate.clone()).unwrap();
        assert_eq!(owned, predicate);
        assert_eq!(<ArrayConstant as ValueProjection<ArrayType>>::from_projected(owned), predicate);
    }

    #[test]
    fn test_array_constant_from() {
        let capture = CaptureReference::new(2, vector_type());
        assert_eq!(ArrayConstant::from(capture.clone()), ArrayConstant::Captured(capture));
        assert_eq!(ArrayConstant::from(true), ArrayConstant::Boolean(true));
        assert_eq!(
            ArrayIrValue::<ArrayConstant>::try_from(false),
            Ok(ArrayIrValue::Array(ArrayConstant::Boolean(false))),
        );
    }
}
