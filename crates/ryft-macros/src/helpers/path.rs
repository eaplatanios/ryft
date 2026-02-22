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
