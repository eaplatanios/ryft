pub mod data_types;
pub mod layouts;

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
