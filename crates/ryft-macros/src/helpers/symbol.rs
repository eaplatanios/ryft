use std::fmt::{self, Display};

use proc_macro2::Span;

/// Wrapper over `&'static str` that provides convenient [`PartialEq`], [`Display`],
/// and [`From`] implementations for working with [`syn`] abstract syntax trees.
#[derive(Copy, Clone, Debug)]
pub struct Symbol(&'static str);

impl Symbol {
    pub const fn new(symbol: &'static str) -> Self {
        Self(symbol)
    }
}

impl PartialEq<Symbol> for syn::Ident {
    fn eq(&self, word: &Symbol) -> bool {
        self == word.0
    }
}

impl PartialEq<Symbol> for syn::Lifetime {
    fn eq(&self, word: &Symbol) -> bool {
        self.ident == word.0[1..]
    }
}

impl PartialEq<Symbol> for syn::Path {
    fn eq(&self, word: &Symbol) -> bool {
        self.is_ident(word.0)
    }
}

impl PartialEq<Symbol> for syn::TypePath {
    fn eq(&self, word: &Symbol) -> bool {
        self.qself.is_none() && &self.path == word
    }
}

impl PartialEq<Symbol> for syn::LifetimeParam {
    fn eq(&self, word: &Symbol) -> bool {
        &self.lifetime == word
    }
}

impl PartialEq<Symbol> for syn::TypeParam {
    fn eq(&self, word: &Symbol) -> bool {
        &self.ident == word
    }
}

impl PartialEq<Symbol> for syn::ConstParam {
    fn eq(&self, word: &Symbol) -> bool {
        &self.ident == word
    }
}

impl PartialEq<Symbol> for syn::GenericParam {
    fn eq(&self, word: &Symbol) -> bool {
        match &self {
            syn::GenericParam::Lifetime(param) => param == word,
            syn::GenericParam::Type(param) => param == word,
            syn::GenericParam::Const(param) => param == word,
        }
    }
}

impl Display for Symbol {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

impl From<Symbol> for syn::Ident {
    fn from(value: Symbol) -> Self {
        syn::Ident::new(value.0, Span::call_site())
    }
}

impl From<Symbol> for syn::Lifetime {
    fn from(value: Symbol) -> Self {
        syn::Lifetime::new(value.0, Span::call_site())
    }
}
