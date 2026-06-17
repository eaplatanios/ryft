use std::borrow::Cow;
use std::fmt::{Debug, Display};

use thiserror::Error;

use crate::parameters::Parameter;

pub mod array_types;
pub mod data_types;
pub mod layouts;
pub mod memories;

pub use array_types::{ArrayType, Shape, Size, StaticShape};
pub use data_types::{DataType, DataTypeError};
pub use layouts::{Layout, LayoutError, StridedLayout, Tile, TileDimension, TiledLayout};
pub use memories::Memory;

/// Error returned when type inference fails.
#[derive(Clone, Debug, Error, PartialEq, Eq, Hash)]
#[error("{message}")]
pub struct TypeError {
    pub message: String,
}

/// Lightweight type-level description of a family of runtime values. A [`Type`] captures the structural metadata that
/// Ryft needs to reason about values without inspecting the values themselves. Examples include scalar data types such
/// as [`DataType`], array-like types that combine an element [`DataType`] with shape information, and richer type
/// descriptors for traced values.
///
/// Note that [`Type`] requires [`Clone`] so that descriptors can be duplicated into staged [`Program`](crate::Program)s
/// returned via [`Cow`] from [`Typed::r#type`](Typed::type), and stored in tracing data structures. It requires
/// [`Debug`] and [`Display`] so diagnostics and rendered programs can show type descriptors consistently without
/// forcing every call site to repeat those bounds. It also requires [`PartialEq`] because type equality is fundamental
/// to type inference and validation, and so generic code bounded on [`Type`] can compare type descriptors without
/// repeating that bound. Finally, it requires [`Parameter`] so that type descriptors can be used as leaves in
/// [`Parameterized`](crate::Parameterized) data structures.
pub trait Type: Clone + Debug + Display + PartialEq + Parameter {
    /// Returns `true` if values described by this [`Type`] are compatible with the provided [`Type`]. The precise
    /// notion of compatibility is type-specific. For example, scalar data types may treat compatibility as promotion
    /// while array-like types may account for broadcasting and nested structure.
    ///
    /// This relation describes *implicit convertibility*: a value of the receiver's type is not itself a value of
    /// `other`, but it can be turned into one (e.g., by promoting its [`DataType`] or broadcasting its [`Shape`]),
    /// possibly losing information along the way. Contrast this with [`Self::is_refined_by`], which holds only when a
    /// value already *is* a value of the other type, with no conversion involved. For example, `f16` is compatible with
    /// (i.e., promotable to) `f32` but does not refine it, while a dynamically shaped `f32` array type of shape
    /// `[?, 3]` is refined by an `f32` array type of shape `[2, 3]` with no conversion involved at all.
    fn is_compatible_with(&self, other: &Self) -> bool;

    /// Returns `true` if every value described by `other` is also described by this [`Type`]. The receiver is the
    /// more general type (e.g., a declared or staged type) and the argument is the more precise one (e.g., the actual
    /// type carried by a runtime value), and so the relation is directional: `declared.is_refined_by(&actual)`.
    /// Interpretation entry points such as [`Program::interpret`](crate::Program::interpret) use this relation to
    /// validate runtime input values against declared program input types.
    ///
    /// For fully static types this is type equality (e.g., [`DataType`] requires equal data types). Types that can
    /// carry unknown components additionally admit every more precise instantiation of those components. For example,
    /// [`ArrayType`]s with [`Size::Dynamic`] dimensions are refined by otherwise-equal [`ArrayType`]s whose
    /// corresponding dimensions are static, per [`Size::is_refined_by`].
    ///
    /// Reading each [`Type`] as the set of values it describes, this relation is equivalent to set inclusion
    /// (i.e., argument ⊆ receiver) and forms a partial ordering (i.e., semantic subtyping where `other` is a subtype of
    /// the receiver, with no conversion involved). This is what distinguishes it from [`Self::is_compatible_with`],
    /// whose promotion- and broadcasting-based notions describe *implicit convertibility* between values of different
    /// types rather than containment. This is also consistent with the notion of refinement in
    /// [StableHLO](https://openxla.org/stablehlo/dynamism) and MLIR.
    fn is_refined_by(&self, other: &Self) -> bool;

    /// Returns `true` if this [`Type`] describes a single scalar (i.e., a rank-`0` array/tensor) value. This predicate
    /// exists to let reverse-mode differentiation enforce scalar-output functions. Reverse-mode differentiation seeds
    /// the output cotangent with the multiplicative identity (i.e., a value of one) and pulls it back to the inputs.
    /// That seed represents the derivative of the output with respect to itself and is only meaningful when the output
    /// is a scalar for simple gradients (i.e., not Jacobians).
    fn is_scalar(&self) -> bool;
}

/// Associates a runtime value with the abstract [`Type`] descriptor that Ryft should use to reason about it. [`Typed`]
/// is the value-level counterpart to [`Type`]. While [`Type`] models relationships between abstract type descriptors,
/// [`Typed`] lets a concrete value produce the descriptor that should represent it during tracing, staging, type
/// checking, and other forms of abstract reasoning.
pub trait Typed<T: Type> {
    /// Returns the [`Type`] description of this value. The returned [`Type`] should capture the structural information
    /// that Ryft needs to reason about the value without having to inspect its contents. Note that returning a [`Cow`]
    /// lets implementors lend out a stored [`Type`] by borrow when one is cached on the value, while still supporting
    /// values that compute their [`Type`] on the fly (and return [`Cow::Owned`]). Callers that need ownership can call
    /// [`Cow::into_owned`] to clone on demand.
    fn r#type(&self) -> Cow<'_, T>;
}

macro_rules! impl_typed_for_scalar {
    ($ty:ty, $data_type:path) => {
        impl Typed<DataType> for $ty {
            fn r#type(&self) -> Cow<'_, DataType> {
                Cow::Owned($data_type)
            }
        }

        impl Typed<ArrayType> for $ty {
            fn r#type(&self) -> Cow<'_, ArrayType> {
                Cow::Owned(ArrayType::scalar($data_type))
            }
        }
    };
}

impl_typed_for_scalar!(bool, DataType::Boolean);
impl_typed_for_scalar!(i8, DataType::I8);
impl_typed_for_scalar!(i16, DataType::I16);
impl_typed_for_scalar!(i32, DataType::I32);
impl_typed_for_scalar!(i64, DataType::I64);
impl_typed_for_scalar!(u8, DataType::U8);
impl_typed_for_scalar!(u16, DataType::U16);
impl_typed_for_scalar!(u32, DataType::U32);
impl_typed_for_scalar!(u64, DataType::U64);
impl_typed_for_scalar!(half::bf16, DataType::BF16);
impl_typed_for_scalar!(half::f16, DataType::F16);
impl_typed_for_scalar!(f32, DataType::F32);
impl_typed_for_scalar!(f64, DataType::F64);
