/// Helper private module making sure that [`PathHelpers`] is a sealed trait. Refer to
/// [this page](https://predr.ag/blog/definitive-guide-to-sealed-traits-in-rust/) for more information
/// on private traits in Rust.
mod private {
    pub trait Sealed {}
    impl Sealed for syn::Path {}
}

/// Defines helper functions for working with [`syn::Path`]s.
pub trait PathHelpers: private::Sealed {
    /// Appends the provided [`syn::PathSegment`] to this [`syn::Path`].
    fn with_segment(&self, value: syn::PathSegment) -> Self;
}

impl PathHelpers for syn::Path {
    fn with_segment(&self, value: syn::PathSegment) -> Self {
        let mut path = self.clone();
        path.segments.push(value);
        path
    }
}

#[cfg(test)]
mod tests {
    use quote::ToTokens;

    use super::PathHelpers;

    #[test]
    fn test_path_with_segment() {
        let path: syn::Path = syn::parse_quote!(ryft::core);
        assert_eq!(path.to_token_stream().to_string().replace(' ', ""), "ryft::core");

        let path = path.with_segment(syn::parse_quote!(Parameter));
        assert_eq!(path.to_token_stream().to_string().replace(' ', ""), "ryft::core::Parameter");

        let path: syn::Path = syn::parse_quote!(::ryft);
        let path = path.with_segment(syn::parse_quote!(core));
        assert_eq!(path.to_token_stream().to_string().replace(' ', ""), "::ryft::core");
        assert!(path.leading_colon.is_some());
    }
}
