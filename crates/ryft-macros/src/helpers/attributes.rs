use proc_macro2::TokenStream;
use quote::ToTokens;

use crate::helpers::symbols::Symbol;

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
            Err(syn::Error::new_spanned(&meta.path, format!("duplicate ryft attribute '{}'", self.name)))
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
            return Err(meta.error("cannot parse attribute value from a 'ParseNestedMeta' with a different path"));
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
                    Err(syn::Error::new_spanned(literal, format!("unexpected suffix `{suffix}` on string literal")))
                } else {
                    Ok(literal.clone())
                }
            }
            _ => Err(syn::Error::new_spanned(
                expression,
                format!("expected ryft '{name}' attribute to be a string: `{name} = \"...\"`"),
            )),
        }
    }
}

impl AttributeValue for syn::Ident {
    fn from_meta(name: &Symbol, meta: &syn::meta::ParseNestedMeta) -> syn::Result<Self> {
        let string = syn::LitStr::from_meta(name, meta)?;
        string.parse().map_err(|_| {
            syn::Error::new_spanned(&string, format!("failed to parse identifier: '{:?}'", string.value()))
        })
    }
}

impl AttributeValue for syn::Path {
    fn from_meta(name: &Symbol, meta: &syn::meta::ParseNestedMeta) -> syn::Result<Self> {
        let string = syn::LitStr::from_meta(name, meta)?;
        string
            .parse()
            .map_err(|_| syn::Error::new_spanned(&string, format!("failed to parse path: '{:?}'", string.value())))
    }
}

impl AttributeValue for syn::ExprPath {
    fn from_meta(name: &Symbol, meta: &syn::meta::ParseNestedMeta) -> syn::Result<Self> {
        let string = syn::LitStr::from_meta(name, meta)?;
        string.parse().map_err(|_| {
            syn::Error::new_spanned(&string, format!("failed to parse expression path: '{:?}'", string.value()))
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
            .map_err(|_| syn::Error::new_spanned(&string, format!("failed to parse type: '{:?}'", string.value())))
    }
}

#[cfg(test)]
mod tests {
    use quote::{ToTokens, quote};
    use syn::parse::Parser;

    use super::{Attribute, AttributeValue};
    use crate::helpers::symbols::Symbol;

    #[test]
    fn test_attribute() {
        let attribute = Attribute::<syn::LitStr>::new(Symbol::new("crate"));
        assert_eq!(attribute.name().to_string(), "crate");
        assert!(attribute.get().is_none());

        let mut attribute = Attribute::<syn::LitStr>::new(Symbol::new("crate"));
        syn::meta::parser(|meta| attribute.set(&meta)).parse2(quote!(crate = "ryft")).unwrap();
        assert_eq!(attribute.get().expect("expected attribute value to be set").value(), "ryft");

        let mut attribute = Attribute::<syn::LitStr>::new(Symbol::new("crate"));
        syn::meta::parser(|meta| attribute.set(&meta)).parse2(quote!(crate = "ryft")).unwrap();
        assert!(
            syn::meta::parser(|meta| attribute.set(&meta))
                .parse2(quote!(crate = "alt_ryft"))
                .unwrap_err()
                .to_string()
                .contains("duplicate ryft attribute 'crate'")
        );

        assert!(
            syn::meta::parser(|meta| {
                let _ = syn::LitStr::from_meta(&Symbol::new("crate"), &meta)?;
                Ok(())
            })
            .parse2(quote!(other = "ryft"))
            .unwrap_err()
            .to_string()
            .contains("different path")
        );

        // Test [`Option::<syn::Ident>::from_meta`].
        let mut parsed = None;
        syn::meta::parser(|meta| {
            parsed = Some(Option::<syn::Ident>::from_meta(&Symbol::new("crate"), &meta)?);
            Ok(())
        })
        .parse2(quote!(crate = "ryft::core"))
        .unwrap();
        assert!(parsed.expect("expected optional value to be parsed").is_none());

        // Test [`bool::from_meta`].
        let mut bool_value = None;
        syn::meta::parser(|meta| {
            bool_value = Some(bool::from_meta(&Symbol::new("crate"), &meta)?);
            Ok(())
        })
        .parse2(quote!(crate))
        .unwrap();
        assert!(bool_value.expect("expected boolean value to be parsed"));

        let mut bool_value = None;
        syn::meta::parser(|meta| {
            bool_value = Some(bool::from_meta(&Symbol::new("crate"), &meta)?);
            Ok(())
        })
        .parse2(quote!(other))
        .unwrap();
        assert!(!bool_value.expect("expected boolean value to be parsed"));

        // Test [`syn::LitStr::from_meta`].
        assert!(
            syn::meta::parser(|meta| {
                let _ = syn::LitStr::from_meta(&Symbol::new("crate"), &meta)?;
                Ok(())
            })
            .parse2(quote!(crate = 42))
            .unwrap_err()
            .to_string()
            .contains("expected ryft 'crate' attribute to be a string")
        );

        // Test [`syn::Ident::from_meta`].
        let mut ident = None;
        syn::meta::parser(|meta| {
            ident = Some(syn::Ident::from_meta(&Symbol::new("crate"), &meta)?);
            Ok(())
        })
        .parse2(quote!(crate = "ryft"))
        .unwrap();
        assert_eq!(ident.expect("expected identifier to be parsed").to_string(), "ryft");

        // Test [`syn::Path::from_meta`].
        let mut path = None;
        syn::meta::parser(|meta| {
            path = Some(syn::Path::from_meta(&Symbol::new("crate"), &meta)?);
            Ok(())
        })
        .parse2(quote!(crate = "ryft::core"))
        .unwrap();
        assert_eq!(
            path.expect("expected path to be parsed").to_token_stream().to_string().replace(' ', ""),
            "ryft::core",
        );

        // Test [`syn::ExprPath::from_meta`].
        let mut expr_path = None;
        syn::meta::parser(|meta| {
            expr_path = Some(syn::ExprPath::from_meta(&Symbol::new("crate"), &meta)?);
            Ok(())
        })
        .parse2(quote!(crate = "ryft::core::Type"))
        .unwrap();
        assert_eq!(
            expr_path
                .expect("expected expression path to be parsed")
                .to_token_stream()
                .to_string()
                .replace(' ', ""),
            "ryft::core::Type",
        );

        // Test [`Vec::<syn::WherePredicate>::from_meta`].
        let mut predicates = None;
        syn::meta::parser(|meta| {
            predicates = Some(Vec::<syn::WherePredicate>::from_meta(&Symbol::new("bounds"), &meta)?);
            Ok(())
        })
        .parse2(quote!(bounds = "T: Clone, U: Default"))
        .unwrap();
        assert_eq!(
            predicates
                .expect("expected where predicates to be parsed")
                .iter()
                .map(|predicate| predicate.to_token_stream().to_string().replace(' ', ""))
                .collect::<Vec<_>>(),
            vec!["T:Clone", "U:Default"],
        );

        // Test [`syn::Type::from_meta`].
        let mut ty = None;
        syn::meta::parser(|meta| {
            ty = Some(syn::Type::from_meta(&Symbol::new("crate"), &meta)?);
            Ok(())
        })
        .parse2(quote!(crate = "Vec<ryft::Placeholder>"))
        .unwrap();
        assert_eq!(
            ty.expect("expected type to be parsed").to_token_stream().to_string().replace(' ', ""),
            "Vec<ryft::Placeholder>",
        );
    }
}
