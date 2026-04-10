pub mod array_types;
pub mod data_types;
pub mod layouts;

pub use array_types::*;
pub use data_types::*;
pub use layouts::*;

/// Lightweight type-level description of a family of runtime values. A [`Type`] captures the structural metadata that
/// Ryft needs to reason about values without inspecting the values themselves. Examples include scalar data types such
/// as [`DataType`], array-like types that combine an element [`DataType`] with shape information, and richer type
/// descriptors for traced values.
pub trait Type {
    /// Returns `true` if values described by this [`Type`] are compatible with the provided [`Type`]. The precise
    /// notion of compatibility is type-specific. For example, scalar data types may treat compatibility as promotion
    /// while array-like types may account for broadcasting and nested structure.
    fn is_compatible_with(&self, other: &Self) -> bool;
}

/// Associates a runtime value with the abstract [`Type`] descriptor that Ryft should use to reason about it. [`Typed`]
/// is the value-level counterpart to [`Type`]. While [`Type`] models relationships between abstract type descriptors,
/// [`Typed`] lets a concrete value produce the descriptor that should represent it during tracing, staging, type
/// checking, and other forms of abstract reasoning.
pub trait Typed<T: Type> {
    /// Returns the [`Type`] description of this value. The returned [`Type`] should capture the structural information
    /// that Ryft needs to reason about the value without having to inspect its contents.
    fn tpe(&self) -> T;
}

// ---------------------------------------------------------------------------
// Scalar Typed<ArrayType> implementations
// ---------------------------------------------------------------------------

macro_rules! impl_typed_for_scalar {
    ($ty:ty, $data_type:path) => {
        impl Typed<ArrayType> for $ty {
            fn tpe(&self) -> ArrayType {
                ArrayType::scalar($data_type)
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
