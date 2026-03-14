pub mod data_type;

pub use data_type::*;

/// Lightweight type-level description of a family of runtime values. A [`Type`] captures the structural metadata that
/// Ryft needs to reason about values without inspecting the values themselves. Examples include scalar data types such
/// as [`DataType`], array-like types that combine an element [`DataType`] with shape information, and richer type
/// descriptors for traced values.
pub trait Type {
    /// Returns `true` if this [`Type`] is a subtype of the provided [`Type`] (i.e., when
    /// values that belong to this [`Type`] can be safely cast to the provided [`Type`]).
    fn is_subtype_of(&self, other: &Self) -> bool;
}
