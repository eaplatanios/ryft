use std::fmt::Display;

pub mod data_type;

// TODO(eaplatanios): [DOC].
// TODO(eaplatanios): Should [`Display`] really be a requirement here?
pub trait Type: Display {
    /// Returns `true` if this [`Type`] is a subtype of the provided [`Type`] (i.e., when
    /// values that belong to this [`Type`] can be safely cast to the provided [`Type`]).
    fn is_subtype_of(&self, other: &Self) -> bool;
}
