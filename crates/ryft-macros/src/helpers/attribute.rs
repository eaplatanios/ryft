use proc_macro2::TokenStream;
use quote::ToTokens;

use crate::helpers::symbol::Symbol;

/// Parsed [`syn::Attribute`] value.
pub struct Attribute<V> {
    /// Name of this [`syn::Attribute`] value that corresponds to the keys in attributes like `#[ryft(crate = ...)]`.
    name: Symbol,

    /// [`TokenStream`] that corresponds to the tokens from which this attribute value was parsed.
    tokens: TokenStream,

    /// Parsed value for this [`Attribute`]. Note that this always set to [`None`] when the attribute is constructed and
    /// it can be set by [`Attribute::set`] afterwards. This is so that we can detect things like a specific attribute
    /// being set multiple times and produce appropriate [`syn::Error`]s.
    value: Option<V>,
}

impl<V> Attribute<V> {
    /// Constructs a new [`Attribute`] with the provided name.
    pub fn new(name: Symbol) -> Self {
        Attribute { name, tokens: TokenStream::new(), value: None }
    }

    /// Returns the name of this [`Attribute`].
    pub fn name(&self) -> Symbol {
        self.name
    }

    /// Sets the value of this [`Attribute`] by parsing the provided [`syn::meta::ParseNestedMeta`]. Returns a
    /// [`syn::Error`] if the provided [`syn::meta::ParseNestedMeta`] cannot be parsed, if it has been set already,
    /// or if the [`syn::meta::ParseNestedMeta::path`] in the provided `meta` does not match the name of this
    /// [`Attribute`].
    pub fn set(&mut self, meta: &syn::meta::ParseNestedMeta) -> syn::Result<()>
    where
        V: AttributeValue,
    {
        if self.value.is_some() {
            Err(syn::Error::new_spanned(&meta.path, format!("Duplicate ryft attribute '{}'", self.name)))
        } else {
            self.tokens = (&meta.path).into_token_stream();
            self.value = Some(V::from_meta(&self.name, meta)?);
            Ok(())
        }
    }

    /// Returns the parsed value of this [`Attribute`] or [`None`] if it has not been set yet.
    pub fn get(self) -> Option<V> {
        self.value
    }
}

/// Helper trait for specifying how to parse [`Attribute`] values of different types from
/// [`syn::meta::ParseNestedMeta`].
pub trait AttributeValue: Sized {
    /// Parses an [`Attribute`] value of this type from the provided [`syn::meta::ParseNestedMeta`], for the provided
    /// [`Attribute`] name. Note that if the [`syn::meta::ParseNestedMeta::path`] in the provided `meta` does not match
    /// the provided `name`, then this function will return a [`syn::Error`]. This is similar in terms of functionality
    /// to functions like [`syn::LitStr::from_meta`].
    fn from_meta(name: &Symbol, meta: &syn::meta::ParseNestedMeta) -> syn::Result<Self>;
}

impl<V: AttributeValue> AttributeValue for Option<V> {
    fn from_meta(name: &Symbol, meta: &syn::meta::ParseNestedMeta) -> syn::Result<Self> {
        // Optional attribute values simply throw away the error information.
        Ok(V::from_meta(name, meta).ok())
    }
}

impl AttributeValue for bool {
    fn from_meta(name: &Symbol, meta: &syn::meta::ParseNestedMeta) -> syn::Result<Self> {
        Ok(meta.path == *name)
    }
}

impl AttributeValue for syn::LitStr {
    fn from_meta(name: &Symbol, meta: &syn::meta::ParseNestedMeta) -> syn::Result<Self> {
        if &meta.path != name {
            return Err(meta.error("Cannot parse attribute value from a 'ParseNestedMeta' with a different path."));
        }

        let expression: syn::Expr = meta.value()?.parse()?;

        let mut value = &expression;
        while let syn::Expr::Group(expression_group) = value {
            value = &expression_group.expr;
        }

        match value {
            syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Str(literal), .. }) => {
                let suffix = literal.suffix();
                if !suffix.is_empty() {
                    Err(syn::Error::new_spanned(literal, format!("Unexpected suffix `{suffix}` on string literal.")))
                } else {
                    Ok(literal.clone())
                }
            }
            _ => Err(syn::Error::new_spanned(
                expression,
                format!("Expected ryft '{name}' attribute to be a string: `{name} = \"...\"`."),
            )),
        }
    }
}

impl AttributeValue for syn::Ident {
    fn from_meta(name: &Symbol, meta: &syn::meta::ParseNestedMeta) -> syn::Result<Self> {
        let string = syn::LitStr::from_meta(name, meta)?;
        string.parse().map_err(|_| {
            syn::Error::new_spanned(&string, format!("Failed to parse identifier: '{:?}'.", string.value()))
        })
    }
}

impl AttributeValue for syn::Path {
    fn from_meta(name: &Symbol, meta: &syn::meta::ParseNestedMeta) -> syn::Result<Self> {
        let string = syn::LitStr::from_meta(name, meta)?;
        string
            .parse()
            .map_err(|_| syn::Error::new_spanned(&string, format!("Failed to parse path: '{:?}'.", string.value())))
    }
}

impl AttributeValue for syn::ExprPath {
    fn from_meta(name: &Symbol, meta: &syn::meta::ParseNestedMeta) -> syn::Result<Self> {
        let string = syn::LitStr::from_meta(name, meta)?;
        string.parse().map_err(|_| {
            syn::Error::new_spanned(&string, format!("Failed to parse expression path: '{:?}'.", string.value()))
        })
    }
}

impl AttributeValue for Vec<syn::WherePredicate> {
    fn from_meta(name: &Symbol, meta: &syn::meta::ParseNestedMeta) -> syn::Result<Self> {
        let string = syn::LitStr::from_meta(name, meta)?;
        string
            .parse_with(syn::punctuated::Punctuated::<syn::WherePredicate, syn::Token![,]>::parse_terminated)
            .map(Vec::from_iter)
            .map_err(|error| syn::Error::new_spanned(&string, error))
    }
}

impl AttributeValue for syn::Type {
    fn from_meta(name: &Symbol, meta: &syn::meta::ParseNestedMeta) -> syn::Result<Self> {
        let string = syn::LitStr::from_meta(name, meta)?;
        string
            .parse()
            .map_err(|_| syn::Error::new_spanned(&string, format!("Failed to parse type: '{:?}'.", string.value())))
    }
}
