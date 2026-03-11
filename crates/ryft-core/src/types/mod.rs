use std::fmt::Display;

pub mod data_type;

pub use data_type::*;

pub trait Type: Display {
    fn is_subtype_of(&self, other: &Self) -> bool;
}
