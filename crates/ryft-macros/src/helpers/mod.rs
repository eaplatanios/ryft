#![allow(dead_code)]

/// Contains helpers for working with [`syn::Attribute`]s.
pub mod attributes;

/// Contains helpers for working with [`syn::Generic`]s.
pub mod generics;

/// Contains helpers for making macros [hygienic](https://en.wikipedia.org/wiki/Hygienic_macro)
/// (e.g., making sure that any generated symbols do not pollute the scope in which a macro is invoked).
pub mod hygiene;

/// Contains helpers for working with [`syn::Ident`]s.
pub mod idents;

/// Contains helpers for working with [`syn::Path`]s.
pub mod paths;

/// Contains helpers for working with receiver values and types (i.e., `self` and `Self`).
pub mod receivers;

/// Contains helpers for working with [`proc_macro2::Span`]s.
pub mod spans;

/// Contains a wrapper over `&'static str` that provides convenient [`PartialEq`], [`Display`],
/// and [`From`] implementations for working with [`syn`] abstract syntax trees.
pub mod symbols;
