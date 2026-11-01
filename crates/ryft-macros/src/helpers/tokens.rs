use quote::ToTokens;

/// Helper private module making sure that [`TokensHelpers`] is a sealed trait. Refer to
/// [this page](https://predr.ag/blog/definitive-guide-to-sealed-traits-in-rust/) for more information
/// on private traits in Rust.
mod private {
    pub trait Sealed {}
    impl<T: quote::ToTokens + ?Sized> Sealed for T {}
}

/// Defines helper functions for working with values that can be converted to tokens (i.e., values that implement
/// [`ToTokens`], like [`syn::Type`]s and [`syn::Path`]s).
pub trait TokensHelpers: private::Sealed {
    /// Returns the string representation of the tokens of this value without any whitespace (e.g., `Box<Operation<V>>`
    /// rather than `Box < Operation < V > >`). This is meant for rendering code fragments in diagnostics and for
    /// comparing tokens. Note that removing whitespace also joins adjacent identifiers and keywords (e.g., `dyn Trait`
    /// is rendered as `dynTrait`), and so the result is not always valid Rust code.
    fn to_compact_string(&self) -> String;

    /// Returns `true` if this value and `other` consist of the same tokens, ignoring whitespace and the spacing of
    /// punctuation (e.g., `Vec<Vec<u8>>` and `Vec<Vec<u8> >` are equal). Given that procedural macros are executed
    /// before type checking and inference is performed by the Rust compiler, token equality is the strongest type
    /// equality that macros can check for (e.g., type aliases cannot be resolved, and so `Vec<u8>` and
    /// `std::vec::Vec<u8>` are not equal).
    fn tokens_eq<T: ToTokens + ?Sized>(&self, other: &T) -> bool;
}

impl<V: ToTokens + ?Sized> TokensHelpers for V {
    fn to_compact_string(&self) -> String {
        self.to_token_stream().to_string().replace(' ', "")
    }

    fn tokens_eq<T: ToTokens + ?Sized>(&self, other: &T) -> bool {
        self.to_compact_string() == other.to_compact_string()
    }
}

#[cfg(test)]
mod tests {
    use quote::quote;

    use super::TokensHelpers;

    #[test]
    fn test_to_compact_string() {
        assert_eq!(quote!(Box<ArrayOperation<V>>).to_compact_string(), "Box<ArrayOperation<V>>");
        assert_eq!(quote!(ryft::Value<Type = T>).to_compact_string(), "ryft::Value<Type=T>");
        let path: syn::Path = syn::parse_quote!(::ryft::core);
        assert_eq!(path.to_compact_string(), "::ryft::core");
    }

    #[test]
    fn test_tokens_eq() {
        let r#type: syn::Type = syn::parse_quote!(Vec<Vec<u8>>);
        assert!(r#type.tokens_eq(&quote!(Vec<Vec<u8>>)));
        assert!(!r#type.tokens_eq(&quote!(std::vec::Vec<Vec<u8>>)));
    }
}
