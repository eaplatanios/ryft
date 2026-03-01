use std::fmt::Display;

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
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
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

#[cfg(test)]
mod tests {
    use quote::ToTokens;

    use super::Symbol;

    #[test]
    fn test_symbol() {
        let symbol = Symbol::new("ryft");
        assert_eq!(symbol.to_string(), "ryft");

        let ident: syn::Ident = symbol.into();
        assert_eq!(ident.to_string(), "ryft");

        let lifetime: syn::Lifetime = Symbol::new("'a").into();
        assert_eq!(lifetime.to_token_stream().to_string(), "'a");

        let ident: syn::Ident = syn::parse_quote!(T);
        assert_eq!(ident, Symbol::new("T"));

        let lifetime: syn::Lifetime = syn::parse_quote!('p);
        assert_eq!(lifetime, Symbol::new("'p"));

        let path: syn::Path = syn::parse_quote!(ryft);
        assert_eq!(path, Symbol::new("ryft"));

        let nested_path: syn::Path = syn::parse_quote!(ryft::core);
        assert_ne!(nested_path, Symbol::new("ryft"));

        let type_path: syn::TypePath = syn::parse_quote!(T);
        assert_eq!(type_path, Symbol::new("T"));

        let qualified_type_path: syn::TypePath = syn::parse_quote!(<Vec<T> as Trait>::Item);
        assert_ne!(qualified_type_path, Symbol::new("Item"));

        let lifetime_param: syn::LifetimeParam = syn::parse_quote!('a: 'b);
        assert_eq!(lifetime_param, Symbol::new("'a"));

        let type_param: syn::TypeParam = syn::parse_quote!(P: Clone);
        assert_eq!(type_param, Symbol::new("P"));

        let const_param: syn::ConstParam = syn::parse_quote!(const N: usize);
        assert_eq!(const_param, Symbol::new("N"));

        let generic_lifetime: syn::GenericParam = syn::parse_quote!('a: 'b);
        assert_eq!(generic_lifetime, Symbol::new("'a"));

        let generic_type: syn::GenericParam = syn::parse_quote!(P: Clone);
        assert_eq!(generic_type, Symbol::new("P"));

        let generic_const: syn::GenericParam = syn::parse_quote!(const N: usize);
        assert_eq!(generic_const, Symbol::new("N"));
    }
}
