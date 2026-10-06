//! Session-carrying composite values of the [`XlaDomain`].
//!
//! [`XlaValue`] is the [`XlaDomain`] counterpart of [`ArrayIrValue`]: it holds an [`XlaArray`], a first-class runtime
//! dimension ([`XlaDimension`]), or an array reference ([`XlaReference`]). Unlike [`ArrayIrValue`], whose dimension and
//! reference members are session-free host values, every member of an [`XlaValue`] (and every projected representation
//! of a member) owns the [`XlaDomain`] session that it belongs to. This has three consequences:
//!
//!   - **Eager dimension-to-array functions.** Functions that turn first-class dimensions into arrays (e.g., comparing
//!     two dimensions or [`DimensionToScalar`](ryft_core::DimensionToScalar)) materialize their results in the session
//!     of their inputs, so they execute eagerly outside any explicit context.
//!   - **Rich dispatch.** [`XlaValue`] dispatches and executes through its [`XlaDomain`], so its capabilities come from
//!     the operation-binding blanket implementations in `ryft-core`, exactly like those of values staged over an
//!     [`XlaDomain`], and free transform entry points (e.g., [`batch`](ryft_core::batch) or
//!     [`differentiate_at`](ryft_core::differentiate_at)) recover the [`XlaDomain`] from the values themselves.
//!   - **No session-less construction.** Because [`ValueProjection::from_projected`] is infallible and receives no
//!     session, projected dimensions and references are the session-carrying [`XlaDimension`] and [`XlaReference`]
//!     rather than [`DimensionValue`] and [`ArrayReference`]. Host-side dimension values enter an [`XlaDomain`] through
//!     its constants (i.e., [`Context::lift`]) instead.
//!
//! Host-side semantics that do not depend on the session (e.g., which members can act as `while` predicates, assertion
//! observations, and the reference handle lifecycle) delegate to the generic [`ArrayIrValue`] implementations in
//! `ryft-core` through a lossless conversion, so that both composite value types share one source of truth.

use std::borrow::{Borrow, Cow};
use std::fmt::{Debug, Display};

use ryft_core::{
    Add, ArrayIrType, ArrayIrValue, ArrayReference, ArrayType, AssertionValue, Compare, CompareOperation,
    ComparisonDirection, Concretizable, Context, DimensionType, DimensionValue, Div, Mul, Neg, Parameter, ProgramError,
    ProjectedContext, ReferenceId, ReferenceType, Sub, Type, TypeError, TypeIdentityRenaming, Typed, Value,
    ValueProjection, WhilePredicate,
};
use ryft_macros::Parameter;

use crate::{XlaArray, XlaDomain};

/// First-class runtime dimension bound to the [`XlaDomain`] session in which array results derived from it are
/// materialized. This is the dimension member of [`XlaValue`] and its projection onto [`DimensionType`].
///
/// An [`XlaDimension`] is a runtime *value* that holds a checked [`DimensionValue`]. It is not a
/// [`Dimension`](ryft_core::Dimension), which is a static or symbolic extent inside an [`ArrayType`]. Dimension
/// arithmetic remains host integer work: it dispatches through this dimension's [`XlaDomain`], whose eager binding
/// evaluates dimension operations on the host without compiling programs. Only functions that produce arrays (e.g.,
/// comparisons and [`DimensionToScalar`](ryft_core::DimensionToScalar)) upload their results, which are placed
/// according to the mesh of the domain (refer to [`XlaValue`] for the placement policy).
///
/// Two dimensions are equal when their [`DimensionValue`]s are equal, regardless of their sessions.
#[derive(Clone, Parameter)]
pub struct XlaDimension<'c> {
    /// Checked host-side dimension.
    value: DimensionValue,

    /// [`XlaDomain`] in which array results computed from this dimension are materialized.
    domain: XlaDomain<'c>,
}

impl<'c> XlaDimension<'c> {
    /// Creates a new [`XlaDimension`] that materializes array results derived from `value` in `domain`.
    pub fn new(value: DimensionValue, domain: XlaDomain<'c>) -> Self {
        Self { value, domain }
    }

    /// Returns the checked host-side [`DimensionValue`] of this dimension.
    pub fn value(&self) -> &DimensionValue {
        &self.value
    }

    /// Returns the [`XlaDomain`] in which array results computed from this dimension are materialized.
    pub fn domain(&self) -> &XlaDomain<'c> {
        &self.domain
    }

    /// Consumes this dimension and returns its checked host-side [`DimensionValue`].
    pub fn into_value(self) -> DimensionValue {
        self.value
    }
}

impl Debug for XlaDimension<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_tuple("XlaDimension").field(&self.value).finish()
    }
}

impl Display for XlaDimension<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.value, formatter)
    }
}

// Dimensions compare by value: the session only selects where derived arrays are materialized.
impl PartialEq for XlaDimension<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Borrow<DimensionValue> for XlaDimension<'_> {
    fn borrow(&self) -> &DimensionValue {
        &self.value
    }
}

impl Typed for XlaDimension<'_> {
    type Type = DimensionType;

    fn r#type(&self) -> Cow<'_, DimensionType> {
        self.value.r#type()
    }
}

// Comparing two dimensions yields a Boolean array, which is materialized in this dimension's domain by binding the
// composite comparison there. The domain decides the predicate on the host from the two extents.
impl<'c> Compare<XlaValue<'c>, DimensionType> for XlaDimension<'c> {
    fn compare(&self, other: &Self, direction: ComparisonDirection) -> Result<XlaValue<'c>, ProgramError> {
        let inputs = [XlaValue::Dimension(self.clone()), XlaValue::Dimension(other.clone())];
        Ok(self.domain.bind(CompareOperation::<ArrayIrType>::new(direction), Vec::new(), &inputs)?.remove(0))
    }
}

impl<'c> Value for XlaDimension<'c> {
    // Like `XlaArray`, a dimension dispatches and executes through the rich `XlaDomain` that it carries, projected onto
    // the dimension member family. That domain evaluates dimension operations on the host and materializes array
    // results (e.g., `dimension_to_scalar`) in its session.
    type DispatchDomain = ProjectedContext<XlaDomain<'c>, DimensionType>;
    type ExecutionDomain = ProjectedContext<XlaDomain<'c>, DimensionType>;

    fn dispatch_domain(&self) -> Self::DispatchDomain {
        self.execution_domain()
    }

    fn execution_domain(&self) -> Self::ExecutionDomain {
        ProjectedContext::new(self.domain.clone())
    }

    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<Self::Type as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        Ok(Self::new(self.value.rename_type_identities(renaming)?, self.domain.clone()))
    }
}

/// Handle to an XLA array reference, bound to the [`XlaDomain`] session that owns its referent. This is the reference
/// member of [`XlaValue`] and its projection onto [`ReferenceType<ArrayType>`](ReferenceType).
///
/// The session is retained next to the handle rather than recovered from the referent, because reading the referent
/// locks its allocation and fails once the allocation is frozen, whereas the domain of a value must be available
/// infallibly. Two references are equal when their handles are equal (i.e., when they name the same allocation and
/// view), regardless of their sessions.
#[derive(Clone)]
pub struct XlaReference<'c> {
    /// Shared reference handle (i.e., the allocation, the view path, and the cached reference type).
    reference: ArrayReference<XlaArray<'c>>,

    /// [`XlaDomain`] that owns the referent of this reference.
    domain: XlaDomain<'c>,
}

impl<'c> XlaReference<'c> {
    /// Creates a new [`XlaReference`] that binds `reference` to `domain`.
    pub fn new(reference: ArrayReference<XlaArray<'c>>, domain: XlaDomain<'c>) -> Self {
        Self { reference, domain }
    }

    /// Returns the shared [`ArrayReference`] handle of this reference.
    pub fn handle(&self) -> &ArrayReference<XlaArray<'c>> {
        &self.reference
    }

    /// Returns the [`XlaDomain`] that owns the referent of this reference.
    pub fn domain(&self) -> &XlaDomain<'c> {
        &self.domain
    }

    /// Consumes this reference and returns its shared [`ArrayReference`] handle.
    pub fn into_handle(self) -> ArrayReference<XlaArray<'c>> {
        self.reference
    }
}

impl Debug for XlaReference<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_tuple("XlaReference").field(&self.reference).finish()
    }
}

impl Display for XlaReference<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.reference, formatter)
    }
}

// References compare by handle identity, like `ArrayReference`.
impl PartialEq for XlaReference<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.reference == other.reference
    }
}

impl Typed for XlaReference<'_> {
    type Type = ReferenceType<ArrayType>;

    fn r#type(&self) -> Cow<'_, ReferenceType<ArrayType>> {
        self.reference.r#type()
    }
}

/// Composite runtime value of the [`XlaDomain`] (i.e., its [`Domain::Value`](ryft_core::Domain::Value)), and the
/// [`XlaDomain`] counterpart of [`ArrayIrValue`] in which every member carries the [`XlaDomain`] session that it
/// belongs to. Refer to the [module documentation](self) for why the session is part of every member.
///
/// # Dispatch and Transforms
///
/// An [`XlaValue`] dispatches and executes through the [`XlaDomain`] of its member, so every capability binds an
/// [`XlaOperation`](crate::experimental::ops::XlaOperation) that the domain executes eagerly (op by op, through its
/// eager dispatch cache), and free transform entry points recover that domain from the values. When several inputs
/// carry different sessions, the receiver's domain executes, and it validates that its array inputs belong to its
/// client and placement. Dimension inputs are host data and need no such validation.
///
/// # Placement
///
/// Arrays that an [`XlaValue`] creates without array inputs (e.g., the scalar that converts a first-class dimension
/// into array data) are placed on the mesh of the executing domain. When the domain of an array member has no mesh of
/// its own, the domain recovered from that member adopts the array's mesh, so a statically known extent read from an
/// array on some devices, and every scalar later derived from it, stays on those devices instead of moving to the
/// client's default devices. Dimensions read from or decoded from an array bind the array's mesh in the same way.
///
/// # Identity
///
/// [`PartialEq`] ignores sessions: arrays compare by the storage identity of [`XlaArray`], dimensions by their
/// [`DimensionValue`]s, and references by their handles.
#[derive(Clone, Debug, PartialEq, Parameter)]
pub enum XlaValue<'c> {
    /// Device-resident [`XlaArray`], which carries its own domain.
    Array(XlaArray<'c>),

    /// First-class runtime dimension.
    Dimension(XlaDimension<'c>),

    /// Array reference.
    Reference(XlaReference<'c>),
}

impl<'c> XlaValue<'c> {
    /// Returns the [`XlaDomain`] of this value's member.
    pub fn domain(&self) -> &XlaDomain<'c> {
        match self {
            Self::Array(value) => value.domain(),
            Self::Dimension(value) => value.domain(),
            Self::Reference(value) => value.domain(),
        }
    }

    /// Converts this value into the session-free [`ArrayIrValue`] representation over XLA arrays, which drops the
    /// domains of dimension and reference members. Used to delegate host-side semantics (e.g., reference handles and
    /// predicates) to the generic [`ArrayIrValue`] implementations in `ryft-core`.
    pub(crate) fn into_array_ir_value(self) -> ArrayIrValue<XlaArray<'c>> {
        match self {
            Self::Array(value) => ArrayIrValue::Array(value),
            Self::Dimension(value) => ArrayIrValue::Dimension(value.into_value()),
            Self::Reference(value) => ArrayIrValue::Reference(value.into_handle()),
        }
    }

    /// Converts a session-free [`ArrayIrValue`] over XLA arrays into an [`XlaValue`], binding dimension and reference
    /// members to `domain`. Array members keep their own domain.
    pub(crate) fn from_array_ir_value(value: ArrayIrValue<XlaArray<'c>>, domain: &XlaDomain<'c>) -> Self {
        match value {
            ArrayIrValue::Array(value) => Self::Array(value),
            ArrayIrValue::Dimension(value) => Self::Dimension(XlaDimension::new(value, domain.clone())),
            ArrayIrValue::Reference(value) => Self::Reference(XlaReference::new(value, domain.clone())),
        }
    }

    /// Returns this member's diagnostic kind name.
    fn kind_name(&self) -> &'static str {
        match self {
            Self::Array(_) => "array",
            Self::Dimension(_) => "dimension",
            Self::Reference(_) => "reference",
        }
    }
}

impl Display for XlaValue<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Array(value) => Display::fmt(value, formatter),
            Self::Dimension(value) => Display::fmt(value, formatter),
            Self::Reference(value) => Display::fmt(value, formatter),
        }
    }
}

impl Typed for XlaValue<'_> {
    type Type = ArrayIrType;

    fn r#type(&self) -> Cow<'_, ArrayIrType> {
        Cow::Owned(match self {
            Self::Array(value) => ArrayIrType::Array(value.r#type().into_owned()),
            Self::Dimension(value) => ArrayIrType::Dimension(value.r#type().into_owned()),
            Self::Reference(value) => ArrayIrType::Reference(value.r#type().into_owned()),
        })
    }
}

impl<'c> Value for XlaValue<'c> {
    // Every member carries its `XlaDomain`, so composite values dispatch and execute through it directly: capability
    // blankets in `ryft-core` bind `XlaOperation`s into this domain, and free transform entry points recover it here.
    type DispatchDomain = XlaDomain<'c>;
    type ExecutionDomain = XlaDomain<'c>;

    fn dispatch_domain(&self) -> Self::DispatchDomain {
        self.execution_domain()
    }

    fn execution_domain(&self) -> Self::ExecutionDomain {
        // A mesh-less domain recovered from an array places the values that it creates without array inputs (e.g., a
        // folded static extent and the scalars later derived from it) on that array's mesh, rather than on the
        // client's default devices.
        match self {
            Self::Array(value) if value.domain().mesh().is_err() => value.domain().with_mesh(value.mesh().clone()),
            _ => self.domain().clone(),
        }
    }

    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<Self::Type as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        match self {
            Self::Array(value) => Ok(Self::Array(value.rename_type_identities(renaming)?)),
            Self::Dimension(value) => Ok(Self::Dimension(value.rename_type_identities(renaming)?)),
            Self::Reference(value) => Ok(Self::Reference(XlaReference::new(
                value.handle().rename_type_identities(renaming)?,
                value.domain().clone(),
            ))),
        }
    }

    fn validate_as_constant(&self) -> Result<(), TypeError> {
        self.clone().into_array_ir_value().validate_as_constant()
    }

    fn is_zero(&self) -> bool {
        match self {
            Self::Array(value) => value.is_zero(),
            Self::Dimension(_) | Self::Reference(_) => false,
        }
    }

    // `singleton` keeps the default `None`: it is a static function, so it cannot attach a session to a singleton
    // dimension, and `None` is the conservative answer.

    fn reference_id(&self) -> Option<ReferenceId> {
        match self {
            Self::Array(_) | Self::Dimension(_) => None,
            Self::Reference(value) => Some(value.handle().id()),
        }
    }
}

impl<'c> ValueProjection<ArrayType> for XlaValue<'c> {
    type Projected = XlaArray<'c>;
    type ProjectedRef<'v>
        = &'v XlaArray<'c>
    where
        Self: 'v;

    fn from_projected(value: XlaArray<'c>) -> Self {
        Self::Array(value)
    }

    fn projected<'v>(&'v self) -> Result<&'v XlaArray<'c>, TypeError>
    where
        ArrayType: 'v,
    {
        match self {
            Self::Array(value) => Ok(value),
            other => Err(TypeError::invalid(format!("expected array type but got {} type", other.kind_name()))),
        }
    }

    fn into_projected(self) -> Result<XlaArray<'c>, TypeError> {
        match self {
            Self::Array(value) => Ok(value),
            other => Err(TypeError::invalid(format!("expected array type but got {} type", other.kind_name()))),
        }
    }
}

impl<'c> ValueProjection<DimensionType> for XlaValue<'c> {
    type Projected = XlaDimension<'c>;
    type ProjectedRef<'v>
        = &'v XlaDimension<'c>
    where
        Self: 'v;

    fn from_projected(value: XlaDimension<'c>) -> Self {
        Self::Dimension(value)
    }

    fn projected<'v>(&'v self) -> Result<&'v XlaDimension<'c>, TypeError>
    where
        DimensionType: 'v,
    {
        match self {
            Self::Dimension(value) => Ok(value),
            other => Err(TypeError::invalid(format!("expected dimension type but got {} type", other.kind_name()))),
        }
    }

    fn into_projected(self) -> Result<XlaDimension<'c>, TypeError> {
        match self {
            Self::Dimension(value) => Ok(value),
            other => Err(TypeError::invalid(format!("expected dimension type but got {} type", other.kind_name()))),
        }
    }
}

impl<'c> ValueProjection<ReferenceType<ArrayType>> for XlaValue<'c> {
    type Projected = XlaReference<'c>;
    type ProjectedRef<'v>
        = &'v XlaReference<'c>
    where
        Self: 'v;

    fn from_projected(value: XlaReference<'c>) -> Self {
        Self::Reference(value)
    }

    fn projected<'v>(&'v self) -> Result<&'v XlaReference<'c>, TypeError>
    where
        ReferenceType<ArrayType>: 'v,
    {
        match self {
            Self::Reference(value) => Ok(value),
            other => Err(TypeError::invalid(format!("expected reference type but got {} type", other.kind_name()))),
        }
    }

    fn into_projected(self) -> Result<XlaReference<'c>, TypeError> {
        match self {
            Self::Reference(value) => Ok(value),
            other => Err(TypeError::invalid(format!("expected reference type but got {} type", other.kind_name()))),
        }
    }
}

impl<'c> From<XlaArray<'c>> for XlaValue<'c> {
    fn from(value: XlaArray<'c>) -> Self {
        Self::Array(value)
    }
}

impl<'c> From<XlaDimension<'c>> for XlaValue<'c> {
    fn from(value: XlaDimension<'c>) -> Self {
        Self::Dimension(value)
    }
}

impl<'c> From<XlaReference<'c>> for XlaValue<'c> {
    fn from(value: XlaReference<'c>) -> Self {
        Self::Reference(value)
    }
}

// The host-side consumer contracts below delegate to the generic `ArrayIrValue` implementations in `ryft-core`, so
// their member-kind semantics (e.g., which members can act as predicates and how equal dimension carries pass through
// a batched selection) have a single source of truth.

impl Concretizable<bool> for XlaValue<'_> {
    fn concretize(&self) -> Result<bool, ProgramError> {
        self.clone().into_array_ir_value().concretize()
    }
}

impl WhilePredicate for XlaValue<'_> {
    fn any_true(&self) -> Result<bool, ProgramError> {
        self.clone().into_array_ir_value().any_true()
    }

    fn mask_select(&self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        let output = self
            .clone()
            .into_array_ir_value()
            .mask_select(&on_true.clone().into_array_ir_value(), &on_false.clone().into_array_ir_value())?;
        Ok(Self::from_array_ir_value(output, on_true.domain()))
    }
}

impl AssertionValue for XlaValue<'_> {
    fn assertion_observation(&self) -> Result<String, ProgramError> {
        self.clone().into_array_ir_value().assertion_observation()
    }

    fn assertion_array(&self) -> Result<Option<ryft_core::Array>, ProgramError> {
        self.clone().into_array_ir_value().assertion_array()
    }
}

// The `std::ops` operator traits are foreign, so the blanket tracer implementations in `ryft-core` cannot cover
// concrete backend values; these implementations provide the panicking operator sugar that the array operation bundles
// require by delegating to the fallible `ryft` capabilities, exactly like those of `XlaArray`.

impl std::ops::Neg for XlaValue<'_> {
    type Output = Self;

    fn neg(self) -> Self {
        Neg::neg(&self).expect("`neg` operation failed")
    }
}

impl std::ops::Add for XlaValue<'_> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Add::add(&self, &rhs).expect("`add` operation failed")
    }
}

impl std::ops::Sub for XlaValue<'_> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Sub::sub(&self, &rhs).expect("`sub` operation failed")
    }
}

impl std::ops::Mul for XlaValue<'_> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Mul::mul(&self, &rhs).expect("`mul` operation failed")
    }
}

impl std::ops::Div for XlaValue<'_> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Div::div(&self, &rhs).expect("`div` operation failed")
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use pretty_assertions::assert_eq;
    use ryft_core::{
        Add, ArrayReferenceTransform, ArrayReferenceTransformIndex, Assert, BatchAxis, Compare, ComparisonDirection,
        Concretizable, Context, ConvertElementType, DataType, Device, DeviceMesh, DimensionBounds, DimensionOperation,
        DimensionSize, DimensionSizeOperation, DimensionToScalar, DimensionToScalarOperation, DimensionValue,
        DimensionVariable, Gather, GatherMode, LogicalMesh, MeshAxis, MeshAxisType, OneOperation, OperationProvider,
        ParallelPermute, Placeholder, Reduce, ReductionKind, ReferenceAddUpdate, ReferenceFreeze, ReferenceNew,
        ReferenceRead, ReferenceSwap, ReferenceWrite, Shape, Sharding, Sort, SortDirection, Typed, WhileOperation,
        WhilePredicate, ZeroOperation, batch, differentiate_at,
    };
    use ryft_pjrt::{Client, ClientOptions, CpuClientOptions, load_cpu_plugin};

    use crate::experimental::ops::{XlaConstant, XlaOperation, XlaProgramBuilder};
    use crate::tests::{execution_client, values_from_bytes, values_to_bytes};
    use crate::{FromPjrt, XlaSession};

    use super::*;

    fn mesh_on(client: &Client<'_>, device_index: usize) -> DeviceMesh {
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[device_index]).unwrap();
        DeviceMesh::new(logical_mesh, vec![device]).unwrap()
    }

    fn array<'c, T: Copy>(
        domain: &XlaDomain<'c>,
        mesh: &DeviceMesh,
        data_type: DataType,
        values: &[T],
        shape: &[usize],
    ) -> XlaArray<'c> {
        let r#type = ArrayType::new(data_type, Shape::from(shape.to_vec()))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), shape.len()))
            .unwrap();
        XlaArray::from_host_buffer(domain, r#type, mesh.clone(), values_to_bytes::<T>(values).as_slice()).unwrap()
    }

    fn read<T: Copy>(value: &XlaValue<'_>) -> Vec<T> {
        let XlaValue::Array(array) = value else { panic!("expected an array value but got `{value}`") };
        let shard = array.addressable_shards().next().unwrap();
        let bytes = shard.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
        values_from_bytes::<T>(bytes.as_slice())
    }

    fn dimension<'c>(domain: &XlaDomain<'c>, extent: usize) -> XlaValue<'c> {
        domain.lift(XlaConstant::Dimension(DimensionValue::constant(extent).unwrap())).unwrap()
    }

    fn same_session<'c>(left: &XlaDomain<'c>, right: &XlaDomain<'c>) -> bool {
        Arc::ptr_eq(left.session(), right.session())
    }

    #[test]
    fn test_xla_dimension_new() {
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let value = DimensionValue::constant(3).unwrap();
        let dimension = XlaDimension::new(value.clone(), domain.clone());
        assert_eq!(dimension.value(), &value);
        assert!(same_session(dimension.domain(), &domain));
        assert_eq!(dimension.into_value(), value);
    }

    #[test]
    fn test_xla_dimension_display_and_debug() {
        let client = execution_client();
        let value = DimensionValue::constant(3).unwrap();
        let dimension = XlaDimension::new(value.clone(), XlaSession::new(&client).domain());
        assert_eq!(dimension.to_string(), "3");
        assert_eq!(format!("{dimension:?}"), format!("XlaDimension({value:?})"));
    }

    #[test]
    fn test_xla_dimension_equality() {
        // Equality ignores the session, which only selects where derived arrays are materialized.
        let client = execution_client();
        let value = DimensionValue::constant(3).unwrap();
        let first = XlaDimension::new(value.clone(), XlaSession::new(&client).domain());
        let second = XlaDimension::new(value, XlaSession::new(&client).domain());
        assert_eq!(first, second);
        assert_ne!(first, XlaDimension::new(DimensionValue::constant(4).unwrap(), XlaSession::new(&client).domain()));
    }

    #[test]
    fn test_xla_dimension_borrow() {
        let client = execution_client();
        let value = DimensionValue::constant(3).unwrap();
        let dimension = XlaDimension::new(value.clone(), XlaSession::new(&client).domain());
        assert_eq!(Borrow::<DimensionValue>::borrow(&dimension), &value);
    }

    #[test]
    fn test_xla_dimension_type() {
        let client = execution_client();
        let value = DimensionValue::constant(3).unwrap();
        let dimension = XlaDimension::new(value.clone(), XlaSession::new(&client).domain());
        assert_eq!(dimension.r#type(), value.r#type());
    }

    #[test]
    fn test_xla_dimension_compare() {
        // The predicate is decided on the host, so only the Boolean result is uploaded and no program is compiled.
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let two = XlaDimension::new(DimensionValue::constant(2).unwrap(), domain.clone());
        let three = XlaDimension::new(DimensionValue::constant(3).unwrap(), domain.clone());
        let cache_size = domain.cache_size();
        let less_than = two.compare(&three, ComparisonDirection::LessThan).unwrap();
        assert_eq!(read::<u8>(&less_than), vec![1]);
        assert!(same_session(less_than.domain(), &domain));
        assert_eq!(read::<u8>(&two.compare(&three, ComparisonDirection::Equal).unwrap()), vec![0]);
        assert_eq!(read::<u8>(&three.compare(&three, ComparisonDirection::GreaterThanOrEqual).unwrap()), vec![1]);
        assert_eq!(domain.cache_size(), cache_size);
    }

    #[test]
    fn test_xla_dimension_execution_domain() {
        // Dimension arithmetic dispatches through the dimension's domain, which evaluates it on the host.
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let two = XlaDimension::new(DimensionValue::constant(2).unwrap(), domain.clone());
        let three = XlaDimension::new(DimensionValue::constant(3).unwrap(), domain.clone());
        assert!(same_session(two.execution_domain().parent(), &domain));
        assert!(same_session(two.dispatch_domain().parent(), &domain));
        let cache_size = domain.cache_size();
        let five = Add::add(&two, &three).unwrap();
        assert_eq!(five.value().extent(), 5);
        assert!(same_session(five.domain(), &domain));
        assert_eq!(domain.cache_size(), cache_size);
    }

    #[test]
    fn test_xla_dimension_rename_type_identities() {
        let client = execution_client();
        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let source = DimensionVariable::new("source", bounds);
        let target = DimensionVariable::new("target", bounds);
        let value = DimensionValue::new(source.clone().into(), 4).unwrap();
        let dimension = XlaDimension::new(value, XlaSession::new(&client).domain());
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(source, target.clone()).unwrap();
        let renamed = dimension.rename_type_identities(&renaming).unwrap();
        assert_eq!(renamed.value(), &DimensionValue::new(target.into(), 4).unwrap());
        assert!(same_session(renamed.domain(), dimension.domain()));
    }

    #[test]
    fn test_xla_reference_new() {
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 0);
        let handle = ArrayReference::new(array(&domain, &mesh, DataType::F32, &[1.0f32], &[]));
        let reference = XlaReference::new(handle.clone(), domain.clone());
        assert_eq!(reference.handle(), &handle);
        assert!(same_session(reference.domain(), &domain));
        assert_eq!(reference.into_handle(), handle);
    }

    #[test]
    fn test_xla_reference_display_and_debug() {
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 0);
        let handle = ArrayReference::new(array(&domain, &mesh, DataType::F32, &[1.0f32], &[]));
        let reference = XlaReference::new(handle.clone(), domain);
        assert_eq!(reference.to_string(), handle.to_string());
        assert_eq!(format!("{reference:?}"), format!("XlaReference({handle:?})"));
    }

    #[test]
    fn test_xla_reference_equality() {
        // References compare by handle identity, regardless of their sessions.
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 0);
        let handle = ArrayReference::new(array(&domain, &mesh, DataType::F32, &[1.0f32], &[]));
        let reference = XlaReference::new(handle.clone(), domain.clone());
        assert_eq!(reference, XlaReference::new(handle, XlaSession::new(&client).domain()));
        let other = ArrayReference::new(array(&domain, &mesh, DataType::F32, &[1.0f32], &[]));
        assert_ne!(reference, XlaReference::new(other, domain));
    }

    #[test]
    fn test_xla_reference_type() {
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 0);
        let handle = ArrayReference::new(array(&domain, &mesh, DataType::F32, &[1.0f32], &[]));
        let reference = XlaReference::new(handle.clone(), domain);
        assert_eq!(reference.r#type(), handle.r#type());
    }

    #[test]
    fn test_xla_value_domain() {
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 0);
        let array = array(&domain, &mesh, DataType::F32, &[1.0f32], &[]);
        let reference = XlaReference::new(ArrayReference::new(array.clone()), domain.clone());
        assert!(same_session(XlaValue::Array(array).domain(), &domain));
        assert!(same_session(dimension(&domain, 3).domain(), &domain));
        assert!(same_session(XlaValue::Reference(reference).domain(), &domain));
    }

    #[test]
    fn test_xla_value_into_array_ir_value() {
        // The conversion is lossless up to the sessions, which the inverse conversion binds to the provided domain.
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 0);
        let array = array(&domain, &mesh, DataType::F32, &[1.0f32], &[]);
        let handle = ArrayReference::new(array.clone());
        let extent = DimensionValue::constant(3).unwrap();
        let values = [
            XlaValue::Array(array.clone()),
            XlaValue::Dimension(XlaDimension::new(extent.clone(), domain.clone())),
            XlaValue::Reference(XlaReference::new(handle.clone(), domain.clone())),
        ];
        let converted = values.clone().map(XlaValue::into_array_ir_value);
        assert_eq!(
            converted,
            [ArrayIrValue::Array(array), ArrayIrValue::Dimension(extent), ArrayIrValue::Reference(handle),],
        );
        let other_domain = XlaSession::new(&client).domain();
        let restored = converted.map(|value| XlaValue::from_array_ir_value(value, &other_domain));
        assert_eq!(restored, values);
        assert!(same_session(restored[0].domain(), &domain));
        assert!(same_session(restored[1].domain(), &other_domain));
        assert!(same_session(restored[2].domain(), &other_domain));
    }

    #[test]
    fn test_xla_value_display_and_debug() {
        // Values render exactly like the corresponding `ArrayIrValue` members.
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 0);
        let array = array(&domain, &mesh, DataType::F32, &[1.0f32], &[]);
        let reference = XlaReference::new(ArrayReference::new(array.clone()), domain.clone());
        for value in [XlaValue::Array(array), dimension(&domain, 3), XlaValue::Reference(reference)] {
            assert_eq!(value.to_string(), value.clone().into_array_ir_value().to_string());
        }
        let value = DimensionValue::constant(3).unwrap();
        assert_eq!(format!("{:?}", dimension(&domain, 3)), format!("Dimension(XlaDimension({value:?}))"));
    }

    #[test]
    fn test_xla_value_type() {
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 0);
        let array = array(&domain, &mesh, DataType::F32, &[1.0f32], &[]);
        let reference = XlaReference::new(ArrayReference::new(array.clone()), domain.clone());
        for value in [XlaValue::Array(array), dimension(&domain, 3), XlaValue::Reference(reference)] {
            assert_eq!(value.r#type(), value.clone().into_array_ir_value().r#type());
        }
    }

    #[test]
    fn test_xla_value_execution_domain() {
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 0);
        let array = XlaValue::Array(array(&domain, &mesh, DataType::F32, &[1.0f32], &[]));
        assert!(same_session(&array.execution_domain(), &domain));
        assert!(same_session(&array.dispatch_domain(), &domain));
        assert!(same_session(&dimension(&domain, 3).execution_domain(), &domain));

        // A mesh-less domain recovered from an array adopts that array's mesh, while a domain with a mesh keeps it.
        assert_eq!(array.execution_domain().mesh().unwrap(), &mesh);
        assert!(dimension(&domain, 3).execution_domain().mesh().is_err());
        let other_mesh = mesh_on(&client, 0);
        let meshed_domain = domain.with_mesh(other_mesh.clone());
        let meshed_array = XlaValue::Array(super::tests::array(&meshed_domain, &mesh, DataType::F32, &[1.0f32], &[]));
        assert_eq!(meshed_array.execution_domain().mesh().unwrap(), &other_mesh);
    }

    #[test]
    fn test_xla_value_execution_domain_places_dimension_derived_arrays_on_the_source_mesh() {
        let client = load_cpu_plugin()
            .unwrap()
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 1);
        let input = XlaValue::Array(array(&domain, &mesh, DataType::F32, &[1.0f32, 2.0, 3.0], &[3]));
        let size = input.dimension_size(0).unwrap();
        let XlaValue::Array(scalar) = size.to_scalar().unwrap() else { panic!("expected an array") };
        assert_eq!(scalar.mesh(), &mesh);
        let XlaValue::Array(comparison) = size.less_than(&dimension(&domain, 4)).unwrap() else {
            panic!("expected an array")
        };
        assert_eq!(comparison.mesh(), &mesh);
    }

    #[test]
    fn test_xla_value_rename_type_identities() {
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 0);
        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let source = DimensionVariable::new("source", bounds);
        let target = DimensionVariable::new("target", bounds);
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(source.clone(), target.clone()).unwrap();

        // Dimension members rename their types and keep their sessions.
        let value =
            XlaValue::Dimension(XlaDimension::new(DimensionValue::new(source.into(), 4).unwrap(), domain.clone()));
        let renamed = value.rename_type_identities(&renaming).unwrap();
        assert_eq!(
            renamed,
            XlaValue::Dimension(XlaDimension::new(DimensionValue::new(target.into(), 4).unwrap(), domain.clone()))
        );
        assert!(same_session(renamed.domain(), &domain));

        // Static array and reference members have no identities to rename, so they keep their storage and handles.
        let array = array(&domain, &mesh, DataType::F32, &[1.0f32], &[]);
        let reference = XlaValue::Reference(XlaReference::new(ArrayReference::new(array.clone()), domain.clone()));
        assert_eq!(XlaValue::Array(array.clone()).rename_type_identities(&renaming), Ok(XlaValue::Array(array)));
        let renamed = reference.rename_type_identities(&renaming).unwrap();
        assert_eq!(renamed, reference);
        assert!(same_session(renamed.domain(), &domain));
    }

    #[test]
    fn test_xla_value_validate_as_constant() {
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 0);
        let array = array(&domain, &mesh, DataType::F32, &[1.0f32], &[]);
        let reference = XlaValue::Reference(XlaReference::new(ArrayReference::new(array.clone()), domain.clone()));
        assert_eq!(XlaValue::Array(array).validate_as_constant(), Ok(()));
        assert_eq!(dimension(&domain, 3).validate_as_constant(), Ok(()));
        assert_eq!(
            reference.validate_as_constant(),
            Err(TypeError::invalid(
                "reference values cannot be stored as program constants; pass external references through program \
                 inputs or captures instead",
            )),
        );
    }

    #[test]
    fn test_xla_value_is_zero() {
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 0);
        let array = array(&domain, &mesh, DataType::F32, &[0.0f32], &[]);
        let reference = XlaValue::Reference(XlaReference::new(ArrayReference::new(array.clone()), domain.clone()));
        assert_eq!(XlaValue::Array(array.clone()).is_zero(), array.is_zero());
        assert!(!dimension(&domain, 0).is_zero());
        assert!(!reference.is_zero());
    }

    #[test]
    fn test_xla_value_singleton() {
        // A static function cannot attach a session, so even a singleton dimension type has no singleton value.
        let singleton_type = DimensionValue::constant(3).unwrap().r#type().into_owned();
        assert_eq!(XlaValue::singleton(&ArrayIrType::Dimension(singleton_type)), None);
        assert_eq!(XlaValue::singleton(&ArrayIrType::Array(ArrayType::scalar(DataType::F32))), None);
    }

    #[test]
    fn test_xla_value_reference_id() {
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 0);
        let array = array(&domain, &mesh, DataType::F32, &[1.0f32], &[]);
        let handle = ArrayReference::new(array.clone());
        let reference = XlaValue::Reference(XlaReference::new(handle.clone(), domain.clone()));
        assert_eq!(XlaValue::Array(array).reference_id(), None);
        assert_eq!(dimension(&domain, 3).reference_id(), None);
        assert_eq!(reference.reference_id(), Some(handle.id()));
    }

    #[test]
    fn test_xla_value_array_projection() {
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 0);
        let array = array(&domain, &mesh, DataType::F32, &[1.0f32], &[]);
        let value = <XlaValue<'_> as ValueProjection<ArrayType>>::from_projected(array.clone());
        assert_eq!(<XlaValue<'_> as ValueProjection<ArrayType>>::projected(&value), Ok(&array));
        assert_eq!(ValueProjection::<ArrayType>::into_projected(value), Ok(array));
        assert_eq!(
            <XlaValue<'_> as ValueProjection<ArrayType>>::projected(&dimension(&domain, 3)),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
    }

    #[test]
    fn test_xla_value_dimension_projection() {
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 0);
        let dimension = XlaDimension::new(DimensionValue::constant(3).unwrap(), domain.clone());
        let value = <XlaValue<'_> as ValueProjection<DimensionType>>::from_projected(dimension.clone());
        assert_eq!(<XlaValue<'_> as ValueProjection<DimensionType>>::projected(&value), Ok(&dimension));
        assert_eq!(ValueProjection::<DimensionType>::into_projected(value), Ok(dimension));
        let array = XlaValue::Array(array(&domain, &mesh, DataType::F32, &[1.0f32], &[]));
        assert_eq!(
            <XlaValue<'_> as ValueProjection<DimensionType>>::projected(&array),
            Err(TypeError::invalid("expected dimension type but got array type")),
        );
    }

    #[test]
    fn test_xla_value_reference_projection() {
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 0);
        let array = array(&domain, &mesh, DataType::F32, &[1.0f32], &[]);
        let reference = XlaReference::new(ArrayReference::new(array.clone()), domain.clone());
        let value = <XlaValue<'_> as ValueProjection<ReferenceType<ArrayType>>>::from_projected(reference.clone());
        assert_eq!(<XlaValue<'_> as ValueProjection<ReferenceType<ArrayType>>>::projected(&value), Ok(&reference));
        assert_eq!(ValueProjection::<ReferenceType<ArrayType>>::into_projected(value), Ok(reference));
        assert_eq!(
            <XlaValue<'_> as ValueProjection<ReferenceType<ArrayType>>>::projected(&XlaValue::Array(array)),
            Err(TypeError::invalid("expected reference type but got array type")),
        );
    }

    #[test]
    fn test_xla_value_from() {
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 0);
        let array = array(&domain, &mesh, DataType::F32, &[1.0f32], &[]);
        let dimension = XlaDimension::new(DimensionValue::constant(3).unwrap(), domain.clone());
        let reference = XlaReference::new(ArrayReference::new(array.clone()), domain);
        assert_eq!(XlaValue::from(array.clone()), XlaValue::Array(array));
        assert_eq!(XlaValue::from(dimension.clone()), XlaValue::Dimension(dimension));
        assert_eq!(XlaValue::from(reference.clone()), XlaValue::Reference(reference));
    }

    #[test]
    fn test_xla_value_concretize() {
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 0);
        let predicate = XlaValue::Array(array(&domain, &mesh, DataType::Boolean, &[1u8], &[]));
        assert_eq!(predicate.concretize(), Ok(true));
        assert!(matches!(
            dimension(&domain, 3).concretize(),
            Err(ProgramError::Concretization { message })
                if message == "cannot extract a concrete boolean from a first-class dimension `3`",
        ));
    }

    #[test]
    fn test_xla_value_any_true() {
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 0);
        assert_eq!(XlaValue::Array(array(&domain, &mesh, DataType::Boolean, &[0u8, 1], &[2])).any_true(), Ok(true));
        assert_eq!(XlaValue::Array(array(&domain, &mesh, DataType::Boolean, &[0u8, 0], &[2])).any_true(), Ok(false));
        assert!(matches!(
            dimension(&domain, 3).any_true(),
            Err(ProgramError::Concretization { message })
                if message == "cannot use first-class dimension `3` as a `while` predicate",
        ));
    }

    #[test]
    fn test_xla_value_mask_select() {
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 0);

        // A scalar predicate selects whole values.
        let falsity = XlaValue::Array(array(&domain, &mesh, DataType::Boolean, &[0u8], &[]));
        let one = XlaValue::Array(array(&domain, &mesh, DataType::F32, &[1.0f32], &[]));
        let two = XlaValue::Array(array(&domain, &mesh, DataType::F32, &[2.0f32], &[]));
        assert_eq!(read::<f32>(&falsity.mask_select(&one, &two).unwrap()), vec![2.0]);

        // A batched (prefix-shaped) predicate selects per item: item 0 keeps the first row of `on_false` and item 1
        // takes the second row of `on_true`.
        let mixed = XlaValue::Array(array(&domain, &mesh, DataType::Boolean, &[0u8, 1], &[2]));
        let on_true = XlaValue::Array(array(&domain, &mesh, DataType::F32, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2]));
        let on_false = XlaValue::Array(array(&domain, &mesh, DataType::F32, &[5.0f32, 6.0, 7.0, 8.0], &[2, 2]));
        assert_eq!(read::<f32>(&mixed.mask_select(&on_true, &on_false).unwrap()), vec![5.0, 6.0, 3.0, 4.0]);

        // Equal dimension carries pass through with their session, while dimensions are never predicates.
        let three = dimension(&domain, 3);
        let selected = mixed.mask_select(&three, &three).unwrap();
        assert_eq!(selected, three);
        assert!(same_session(selected.domain(), &domain));
        assert!(matches!(
            three.mask_select(&one, &two),
            Err(ProgramError::Concretization { message })
                if message == "cannot use first-class dimension `3` as a `while` predicate",
        ));
    }

    #[test]
    fn test_xla_value_assertion_value() {
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 0);
        let scalar = XlaValue::Array(array(&domain, &mesh, DataType::I32, &[42i32], &[]));
        assert_eq!(scalar.assertion_observation(), Ok("42".to_string()));
        assert_eq!(scalar.assertion_array(), Ok(Some(ryft_core::Array::scalar(42i32).unwrap())));
        assert_eq!(dimension(&domain, 3).assertion_observation(), Ok("3".to_string()));
        assert_eq!(dimension(&domain, 3).assertion_array(), Ok(None));
    }

    #[test]
    fn test_xla_value_assert() {
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 0);
        let truth = XlaValue::Array(array(&domain, &mesh, DataType::Boolean, &[1u8], &[]));
        let falsity = XlaValue::Array(array(&domain, &mesh, DataType::Boolean, &[0u8], &[]));
        assert_eq!(truth.assert("predicate must hold", &[]), Ok(()));
        assert!(falsity.assert("predicate must hold", &[]).is_err());
    }

    #[test]
    fn test_xla_value_operators() {
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 0);
        let two = XlaValue::Array(array(&domain, &mesh, DataType::F32, &[2.0f32], &[]));
        let four = XlaValue::Array(array(&domain, &mesh, DataType::F32, &[4.0f32], &[]));
        assert_eq!(read::<f32>(&-two.clone()), vec![-2.0]);
        assert_eq!(read::<f32>(&(two.clone() + four.clone())), vec![6.0]);
        assert_eq!(read::<f32>(&(two.clone() - four.clone())), vec![-2.0]);
        assert_eq!(read::<f32>(&(two.clone() * four.clone())), vec![8.0]);
        assert_eq!(read::<f32>(&(four / two)), vec![2.0]);
    }

    #[test]
    fn test_xla_value_compare_dimensions() {
        // Comparing first-class dimensions executes eagerly outside any context, without compiling a program, and
        // materializes the Boolean result in the session of the inputs.
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 0);
        let (two, three) = (dimension(&domain, 2), dimension(&domain, 3));
        let cache_size = domain.cache_size();
        assert_eq!(read::<u8>(&two.less_than(&three).unwrap()), vec![1]);
        assert_eq!(read::<u8>(&two.equal(&three).unwrap()), vec![0]);
        assert_eq!(read::<u8>(&three.greater_than_or_equal(&three).unwrap()), vec![1]);
        assert_eq!(domain.cache_size(), cache_size);

        // A dimension cannot be compared with an array.
        let scalar = XlaValue::Array(array(&domain, &mesh, DataType::I64, &[2i64], &[]));
        assert!(matches!(
            two.less_than(&scalar),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message.starts_with("`compare` inputs must both be arrays or both be first-class dimensions"),
        ));
    }

    #[test]
    fn test_xla_value_dimension_to_scalar() {
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let cache_size = domain.cache_size();
        let scalar = dimension(&domain, 2).to_scalar().unwrap();
        assert!(matches!(
            scalar.r#type().as_ref(),
            ArrayIrType::Array(r#type) if r#type.data_type() == DataType::I64 && r#type.rank() == 0,
        ));
        assert_eq!(read::<i64>(&scalar), vec![2]);
        assert_eq!(domain.cache_size(), cache_size);
    }

    #[test]
    fn test_xla_value_reference_capabilities() {
        // Top-level reference operations execute eagerly on their host-side handles, with the validation and locking of
        // the generic composite reference implementations.
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 0);
        let vector = |values: &[f32]| XlaValue::Array(array(&domain, &mesh, DataType::F32, values, &[values.len()]));
        let scalar = |value: f32| XlaValue::Array(array(&domain, &mesh, DataType::F32, &[value], &[]));

        let reference = vector(&[1.0, 2.0]).reference_new().unwrap();
        assert!(same_session(reference.domain(), &domain));
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
        assert!(reference.write(&scalar(1.0)).is_err());
        let alias = reference.clone();
        assert_eq!(read::<f32>(&reference.freeze().unwrap()), vec![6.0, 9.0]);
        assert!(alias.read().is_err());
        assert!(alias.write(&vector(&[1.0, 1.0])).is_err());

        // Reference operands of array operations are rejected by the array projection of the composite capabilities.
        let reference = vector(&[1.0, 2.0]).reference_new().unwrap();
        assert_eq!(
            Add::add(&reference, &vector(&[1.0, 2.0])),
            Err(ProgramError::Type(TypeError::invalid("expected array type but got reference type"))),
        );
    }

    #[test]
    fn test_xla_value_projected_capabilities() {
        // Capabilities without a receiver-shaped signature project onto the array member through the generalized
        // composite implementations in `ryft-core`.
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 0);
        let input = XlaValue::Array(array(&domain, &mesh, DataType::F32, &[3.0f32, 1.0, 2.0], &[3]));
        let sorted = XlaValue::sort(&[input.clone()], 0, SortDirection::Ascending).unwrap();
        assert_eq!(read::<f32>(&sorted[0]), vec![1.0, 2.0, 3.0]);
        let indices = XlaValue::Array(array(&domain, &mesh, DataType::I32, &[2i32, 0], &[2]));
        assert_eq!(read::<f32>(&input.gather_axis(&indices, 0, GatherMode::Clip).unwrap()), vec![2.0, 3.0]);
        assert!(input.parallel_permute("missing", vec![(0, 0)]).is_err());
    }

    #[test]
    fn test_xla_value_program_replay_lifts_dimension_constants() {
        // Replaying a program over the domain lifts its dimension constants through the session-aware `lift`.
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 0);
        let input = array(&domain, &mesh, DataType::F32, &[1.0f32, 2.0, 3.0], &[3]);

        // `dimension_size(x, 0) + 2` followed by `dimension_to_scalar`, with the `2` stored as a program constant.
        let mut builder = XlaProgramBuilder::new();
        let x = builder.add_input(ArrayIrType::Array(input.r#type().into_owned()));
        let size_operation = DimensionSizeOperation::new(input.r#type().as_ref(), 0).unwrap();
        let size_type = size_operation.output_type().clone();
        let size = builder.add_instruction(size_operation, Vec::new(), vec![x], None).unwrap()[0];
        let constant = DimensionValue::constant(2).unwrap();
        let sum = ryft_core::DimensionAddOperation::new(&size_type, constant.r#type().as_ref()).unwrap();
        let two = builder.add_constant(XlaConstant::Dimension(constant));
        let sum = builder
            .add_instruction(XlaOperation::Dimension(DimensionOperation::Add(sum)), Vec::new(), vec![size, two], None)
            .unwrap()[0];
        let output = builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![sum], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let outputs = program.interpret_in_context(&domain, vec![XlaValue::Array(input)]).unwrap();
        assert_eq!(read::<i64>(&outputs[0]), vec![5]);
    }

    #[test]
    fn test_xla_value_free_transforms() {
        // Free transform entry points recover the domain from the values, so no explicit context is needed.
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 0);
        let input = XlaValue::Array(array(&domain, &mesh, DataType::F32, &[1.0f32, 2.0, 3.0], &[3]));

        let squared: XlaValue<'_> =
            batch(|x| Ok(x.clone() * x), input.clone(), BatchAxis::new(0), BatchAxis::new(0), None).unwrap();
        assert_eq!(read::<f32>(&squared), vec![1.0, 4.0, 9.0]);

        let (value, derivative) = differentiate_at(input.clone()).jvp(input.clone(), |x| Ok(x.clone() * x)).unwrap();
        assert_eq!(read::<f32>(&value), vec![1.0, 4.0, 9.0]);
        assert_eq!(read::<f32>(&derivative), vec![2.0, 8.0, 18.0]);

        let (value, gradient) = differentiate_at(input)
            .value_and_gradient(|x| (x.clone() * x).reduce(&[0], ReductionKind::Sum).unwrap())
            .unwrap();
        assert_eq!(read::<f32>(&value), vec![14.0]);
        assert_eq!(read::<f32>(&gradient), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_xla_value_free_transforms_replay_dimension_derived_values() {
        // Each item is scaled by its own folded extent, which the transforms replay as a dimension constant.
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 0);
        let input = XlaValue::Array(array(&domain, &mesh, DataType::F32, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let scaled: XlaValue<'_> = batch(
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

        let primal = XlaValue::Array(array(&domain, &mesh, DataType::F32, &[1.0f32, 2.0, 3.0], &[3]));
        let tangent = XlaValue::Array(array(&domain, &mesh, DataType::F32, &[1.0f32, 1.0, 1.0], &[3]));
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
    fn test_xla_value_jvp_through_a_data_dependent_while() {
        // At concrete primals, the forward-mode `while` rule decides the loop condition on the host through the
        // predicate contracts of `XlaValue` and unrolls the executed iterations.
        let client = execution_client();
        let domain = XlaSession::new(&client).domain();
        let mesh = mesh_on(&client, 0);
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let condition = {
            let mut builder = XlaProgramBuilder::new();
            let state = builder.add_input(scalar_type.clone());
            let zero = XlaOperation::provide(ZeroOperation::new(scalar_type.clone()), &[]).unwrap();
            let zero = builder.add_instruction(zero, Vec::new(), Vec::new(), None).unwrap()[0];
            let comparison = ryft_core::CompareOperation::<ArrayType>::new(ComparisonDirection::GreaterThan);
            let predicate = builder
                .add_instruction(XlaOperation::Array(comparison.into()), Vec::new(), vec![state, zero], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![predicate], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = XlaProgramBuilder::new();
            let state = builder.add_input(scalar_type.clone());
            let one = XlaOperation::provide(OneOperation::new(scalar_type.clone()), &[]).unwrap();
            let one = builder.add_instruction(one, Vec::new(), Vec::new(), None).unwrap()[0];
            let subtraction = ryft_core::SubOperation::<ArrayType>::new();
            let next = builder
                .add_instruction(XlaOperation::Array(subtraction.into()), Vec::new(), vec![state, one], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![next], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let primal = XlaValue::Array(array(&domain, &mesh, DataType::F32, &[3.5f32], &[]));
        let tangent = XlaValue::Array(array(&domain, &mesh, DataType::F32, &[2.0f32], &[]));
        let (value, derivative) = differentiate_at(primal)
            .jvp(tangent, |state| {
                let mut outputs = state.dispatch_domain().bind(
                    XlaOperation::While(WhileOperation::new()),
                    vec![condition.clone(), body.clone()],
                    &[state.clone()],
                )?;
                Ok(outputs.remove(0))
            })
            .unwrap();
        assert_eq!(read::<f32>(&value), vec![-0.5]);
        assert_eq!(read::<f32>(&derivative), vec![2.0]);
    }
}
