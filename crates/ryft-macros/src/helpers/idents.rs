use syn::{visit::Visit, visit_mut::VisitMut};

/// Helper private module making sure that [`IdentHelpers`] is a sealed trait. Refer to
/// [this page](https://predr.ag/blog/definitive-guide-to-sealed-traits-in-rust/) for more information
/// on private traits in Rust.
mod private {
    pub trait Sealed {}
    impl Sealed for syn::Field {}
    impl Sealed for syn::Generics {}
    impl Sealed for syn::GenericParam {}
    impl Sealed for syn::LifetimeParam {}
    impl Sealed for syn::TypeParam {}
    impl Sealed for syn::ConstParam {}
    impl Sealed for syn::Lifetime {}
    impl Sealed for syn::Type {}
    impl Sealed for syn::ReturnType {}
    impl Sealed for syn::TypeParamBound {}
    impl Sealed for syn::TypePath {}
    impl Sealed for syn::Path {}
    impl Sealed for syn::PathArguments {}
    impl Sealed for syn::GenericArgument {}
    impl Sealed for syn::CapturedParam {}
    impl Sealed for syn::WherePredicate {}
}

/// Defines helper functions for working with [`syn::Ident`]s.
pub trait IdentHelpers: private::Sealed {
    /// Returns the [`syn::Ident`] of this instance. Note that if this instance is not equivalent to a single
    /// [`syn::Ident`] (e.g., if we have a [`syn::Path`] with multiple segments), then this function will return
    /// [`None`]. Otherwise, it will return the [`syn::Ident`] that is equivalent to this instance (e.g., when
    /// converted to code using [`quote::quote`]).
    fn ident(&self) -> Option<&syn::Ident>;

    /// Returns [`true`] if the provided [`syn::Ident`] matches this instance. This function will only return
    /// [`true`] for instances that match the provided [`syn::Ident`] *exactly*. For example, a [`syn::TypePath`]
    /// will match only if it has no [`syn::TypePath::qself`] and contains a single segment that matches the
    /// provided [`syn::Ident`].
    fn matches_ident(&self, ident: &syn::Ident) -> bool {
        self.ident() == Some(ident)
    }

    /// Returns [`true`] if this instance references the provided [`syn::Ident`]. Note that this function specifically
    /// checks for references of the [`syn::Ident`] that are not ambiguous. For example, if a [`syn::PathSegment`] in
    /// the middle of a long [`syn::Path`] matches the provided [`syn::Ident`], then that will not be counted as a
    /// reference. Typically, `ident` will be the [`syn::Ident`] of a [`syn::GenericParam`] and this function can be
    /// used to determine whether e.g., a specific [`syn::Type`] references that [`syn::GenericParam`].
    ///
    /// Note that macro paths are not going to be considered by this function and so any type parameters that appear
    /// in macro paths will be ignored. For example, `T!()` in the following structure will be ignored:
    ///
    /// ```no_run
    /// # use std::marker::PhantomData;
    /// # macro_rules! T { () => { u32 }}
    ///
    /// struct TypeMacro<T> {
    ///     r#macro: T!(),
    ///     marker: PhantomData<T>,
    /// }
    /// ```
    fn references_ident(&self, ident: &syn::Ident) -> bool;

    /// Replaces [`syn::Ident`] in this instance with [`syn::Path`]. This is meant to be used
    /// for replacing generic parameter identifiers in [`syn::Generic`]s and [`syn::Type`]s.
    fn replace_ident(&mut self, ident: &syn::Ident, replacement: &syn::Path);
}

impl IdentHelpers for syn::Field {
    fn ident(&self) -> Option<&syn::Ident> {
        self.ident.as_ref()
    }

    fn references_ident(&self, ident: &syn::Ident) -> bool {
        let mut visitor = ReferencesIdentVisitor::new(ident);
        visitor.visit_type(&self.ty);
        visitor.referenced
    }

    fn replace_ident(&mut self, ident: &syn::Ident, replacement: &syn::Path) {
        let mut visitor = ReplaceIdentVisitor::new(ident, replacement);
        visitor.visit_field_mut(self);
    }
}

impl IdentHelpers for syn::Generics {
    fn ident(&self) -> Option<&syn::Ident> {
        None
    }

    fn references_ident(&self, ident: &syn::Ident) -> bool {
        let mut visitor = ReferencesIdentVisitor::new(ident);
        visitor.visit_generics(self);
        visitor.referenced
    }

    fn replace_ident(&mut self, ident: &syn::Ident, replacement: &syn::Path) {
        let mut visitor = ReplaceIdentVisitor::new(ident, replacement);
        visitor.visit_generics_mut(self);
    }
}

impl IdentHelpers for syn::GenericParam {
    fn ident(&self) -> Option<&syn::Ident> {
        match self {
            syn::GenericParam::Lifetime(p) => p.ident(),
            syn::GenericParam::Type(p) => p.ident(),
            syn::GenericParam::Const(p) => p.ident(),
        }
    }

    fn references_ident(&self, ident: &syn::Ident) -> bool {
        let mut visitor = ReferencesIdentVisitor::new(ident);
        visitor.visit_generic_param(self);
        visitor.referenced
    }

    fn replace_ident(&mut self, ident: &syn::Ident, replacement: &syn::Path) {
        let mut visitor = ReplaceIdentVisitor::new(ident, replacement);
        visitor.visit_generic_param_mut(self);
    }
}

impl IdentHelpers for syn::LifetimeParam {
    fn ident(&self) -> Option<&syn::Ident> {
        Some(&self.lifetime.ident)
    }

    fn references_ident(&self, ident: &syn::Ident) -> bool {
        let mut visitor = ReferencesIdentVisitor::new(ident);
        visitor.visit_lifetime_param(self);
        visitor.referenced
    }

    fn replace_ident(&mut self, ident: &syn::Ident, replacement: &syn::Path) {
        let mut visitor = ReplaceIdentVisitor::new(ident, replacement);
        visitor.visit_lifetime_param_mut(self);
    }
}

impl IdentHelpers for syn::TypeParam {
    fn ident(&self) -> Option<&syn::Ident> {
        Some(&self.ident)
    }

    fn references_ident(&self, ident: &syn::Ident) -> bool {
        let mut visitor = ReferencesIdentVisitor::new(ident);
        visitor.visit_type_param(self);
        visitor.referenced
    }

    fn replace_ident(&mut self, ident: &syn::Ident, replacement: &syn::Path) {
        let mut visitor = ReplaceIdentVisitor::new(ident, replacement);
        visitor.visit_type_param_mut(self);
    }
}

impl IdentHelpers for syn::ConstParam {
    fn ident(&self) -> Option<&syn::Ident> {
        Some(&self.ident)
    }

    fn references_ident(&self, ident: &syn::Ident) -> bool {
        let mut visitor = ReferencesIdentVisitor::new(ident);
        visitor.visit_const_param(self);
        visitor.referenced
    }

    fn replace_ident(&mut self, ident: &syn::Ident, replacement: &syn::Path) {
        let mut visitor = ReplaceIdentVisitor::new(ident, replacement);
        visitor.visit_const_param_mut(self);
    }
}

impl IdentHelpers for syn::Type {
    fn ident(&self) -> Option<&syn::Ident> {
        match self {
            syn::Type::Array(_) | syn::Type::BareFn(_) => None,
            syn::Type::Group(t) => t.elem.ident(),
            syn::Type::ImplTrait(_)
            | syn::Type::Infer(_)
            | syn::Type::Macro(_)
            | syn::Type::Never(_)
            | syn::Type::Paren(_) => None,
            syn::Type::Path(syn::TypePath { qself: None, path }) => path.ident(),
            syn::Type::Path(_) => None,
            syn::Type::Ptr(_)
            | syn::Type::Reference(_)
            | syn::Type::Slice(_)
            | syn::Type::TraitObject(_)
            | syn::Type::Tuple(_)
            | syn::Type::Verbatim(_) => None,
            _ => None,
        }
    }

    fn references_ident(&self, ident: &syn::Ident) -> bool {
        let mut visitor = ReferencesIdentVisitor::new(ident);
        visitor.visit_type(self);
        visitor.referenced
    }

    fn replace_ident(&mut self, ident: &syn::Ident, replacement: &syn::Path) {
        let mut visitor = ReplaceIdentVisitor::new(ident, replacement);
        visitor.visit_type_mut(self);
    }
}

impl IdentHelpers for syn::Path {
    fn ident(&self) -> Option<&syn::Ident> {
        self.get_ident()
    }

    fn references_ident(&self, ident: &syn::Ident) -> bool {
        let mut visitor = ReferencesIdentVisitor::new(ident);
        visitor.visit_path(self);
        visitor.referenced
    }

    fn replace_ident(&mut self, ident: &syn::Ident, replacement: &syn::Path) {
        let mut visitor = ReplaceIdentVisitor::new(ident, replacement);
        visitor.visit_path_mut(self);
    }
}

impl IdentHelpers for syn::WherePredicate {
    fn ident(&self) -> Option<&syn::Ident> {
        None
    }

    fn references_ident(&self, ident: &syn::Ident) -> bool {
        let mut visitor = ReferencesIdentVisitor::new(ident);
        visitor.visit_where_predicate(self);
        visitor.referenced
    }

    fn replace_ident(&mut self, ident: &syn::Ident, replacement: &syn::Path) {
        let mut visitor = ReplaceIdentVisitor::new(ident, replacement);
        visitor.visit_where_predicate_mut(self);
    }
}

/// Internal helper for implementing [`IdentHelpers::references_ident`].
struct ReferencesIdentVisitor<'s> {
    ident: &'s syn::Ident,
    referenced: bool,
}

impl<'s> ReferencesIdentVisitor<'s> {
    fn new(ident: &'s syn::Ident) -> Self {
        Self { ident, referenced: false }
    }
}

impl Visit<'_> for ReferencesIdentVisitor<'_> {
    fn visit_lifetime(&mut self, node: &syn::Lifetime) {
        if self.referenced {
            return;
        }

        if &node.ident == self.ident {
            self.referenced = true;
        }
    }

    fn visit_path(&mut self, node: &syn::Path) {
        if self.referenced {
            return;
        }

        if node.get_ident() == Some(self.ident) {
            self.referenced = true;
        } else if node.segments.first().iter().any(|segment| &segment.ident == self.ident) {
            // If the path starts with the [`syn::Ident`] that we are looking for, then we assume that it references
            // that [`syn::Ident`]. That is because the [`syn::Ident`]s that we are looking for typically correspond to
            // [`syn::GenericParam`]s.
            self.referenced = true;
        } else {
            // Note that the segment [`syn::Ident`]s are not checked. That is because those identifiers would not match
            // the identifier we are looking for as they appear in the middle of a path, and we only care about paths
            // fully matching. Note that if we had type information available when our macros get to run, this would
            // all be much easier as we could check for specific types directly.
            node.segments.iter().for_each(|segment| self.visit_path_arguments(&segment.arguments));
        }
    }

    fn visit_captured_param(&mut self, node: &syn::CapturedParam) {
        if self.referenced {
            return;
        }

        match node {
            syn::CapturedParam::Lifetime(lifetime) => self.visit_lifetime(lifetime),
            syn::CapturedParam::Ident(ident) if ident == self.ident => self.referenced = true,
            _ => {}
        }
    }

    fn visit_type_param_bound(&mut self, node: &syn::TypeParamBound) {
        if self.referenced {
            return;
        }

        match node {
            syn::TypeParamBound::Trait(b) => self.visit_path(&b.path),
            syn::TypeParamBound::Lifetime(b) => self.visit_lifetime(b),
            syn::TypeParamBound::PreciseCapture(b) => b.params.iter().for_each(|p| self.visit_captured_param(p)),
            syn::TypeParamBound::Verbatim(_) => {}
            _ => {}
        }
    }

    fn visit_type_bare_fn(&mut self, node: &syn::TypeBareFn) {
        if self.referenced {
            return;
        }

        node.inputs.iter().for_each(|input| self.visit_type(&input.ty));
        self.visit_return_type(&node.output);
    }
}

/// Internal helper for implementing [`IdentHelpers::replace_ident`].
struct ReplaceIdentVisitor<'s> {
    ident: &'s syn::Ident,
    replacement: &'s syn::Path,
}

impl<'s> ReplaceIdentVisitor<'s> {
    fn new(ident: &'s syn::Ident, replacement: &'s syn::Path) -> Self {
        Self { ident, replacement }
    }
}

impl VisitMut for ReplaceIdentVisitor<'_> {
    fn visit_path_mut(&mut self, node: &mut syn::Path) {
        if node.leading_colon.is_none() && node.segments.first().iter().all(|s| &s.ident == self.ident) {
            *node = {
                let leading_colon = self.replacement.leading_colon;
                let mut segments = syn::punctuated::Punctuated::<syn::PathSegment, syn::Token![::]>::new();
                self.replacement.segments.iter().for_each(|s| segments.push(s.clone()));
                node.segments.iter().skip(1).for_each(|s| segments.push(s.clone()));
                syn::Path { leading_colon, segments }
            };
        } else {
            node.segments.pairs_mut().for_each(|mut segment| {
                let segment = segment.value_mut();
                self.visit_path_segment_mut(segment);
            });
        }
    }

    fn visit_lifetime_mut(&mut self, node: &mut syn::Lifetime) {
        if &node.ident == self.ident {
            self.replacement.get_ident().into_iter().for_each(|replacement_ident| {
                node.ident = replacement_ident.clone();
            });
        }
    }

    fn visit_type_param_mut(&mut self, node: &mut syn::TypeParam) {
        if &node.ident == self.ident {
            self.replacement.get_ident().into_iter().for_each(|replacement_ident| {
                node.ident = replacement_ident.clone();
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use quote::{ToTokens, quote};
    use syn::parse::Parser;

    use super::IdentHelpers;
    use crate::helpers::generics::GenericsHelpers;

    #[test]
    fn test_matches_ident() {
        let ident_t: syn::Ident = syn::parse_quote!(T);
        let ident_u: syn::Ident = syn::parse_quote!(U);

        let path: syn::Path = syn::parse_quote!(T);
        assert_eq!(path.ident().map(|ident| ident.to_string()), Some("T".to_string()));
        assert!(path.matches_ident(&ident_t));

        let ty: syn::Type = syn::parse_quote!(T);
        assert_eq!(ty.ident().map(|ident| ident.to_string()), Some("T".to_string()));
        assert!(ty.matches_ident(&ident_t));

        let generic_param: syn::GenericParam = syn::parse_quote!(T: Clone);
        assert_eq!(generic_param.ident().map(|ident| ident.to_string()), Some("T".to_string()));
        assert!(!generic_param.matches_ident(&ident_u));

        let where_predicate: syn::WherePredicate = syn::parse_quote!(T: Clone);
        assert!(!where_predicate.matches_ident(&ident_t));
    }

    #[test]
    fn test_references_ident() {
        let ident_t: syn::Ident = syn::parse_quote!(T);
        let ident_p: syn::Ident = syn::parse_quote!(P);

        let referencing_type: syn::Type = syn::parse_quote!(fn(T) -> Option<T>);
        assert!(referencing_type.references_ident(&ident_t));

        let non_referencing_type: syn::Type = syn::parse_quote!(crate::module::T);
        assert!(!non_referencing_type.references_ident(&ident_t));

        let referencing_path: syn::Path = syn::parse_quote!(T::Assoc);
        assert!(referencing_path.references_ident(&ident_t));

        let non_referencing_path: syn::Path = syn::parse_quote!(crate::T);
        assert!(!non_referencing_path.references_ident(&ident_t));

        let field: syn::Field = syn::Field::parse_named.parse2(quote!(value: Option<T>)).unwrap();
        assert!(field.references_ident(&ident_t));

        let path: syn::Path = syn::parse_quote!(P);
        let mut generics = syn::parse2::<syn::DeriveInput>(quote!(
            struct Dummy<T, U>
            where
                Vec<T>: Into<U>,
                U: From<T>;
        ))
        .expect("failed to parse derive input")
        .generics;
        generics.replace_ident(&ident_t, &path);
        assert!(generics.matches_any_param(&ident_p));
        assert!(!generics.matches_any_param(&ident_t));
        let where_clause = generics.where_clause.as_ref().expect("expected a where clause");
        assert!(where_clause.predicates.iter().all(|predicate| !predicate.references_ident(&ident_t)));
        assert!(where_clause.predicates.iter().any(|predicate| predicate.references_ident(&ident_p)));
    }

    #[test]
    fn test_replace_ident() {
        let ident: syn::Ident = syn::parse_quote!(T);
        let path: syn::Path = syn::parse_quote!(ryft::Placeholder);

        let mut ty: syn::Type = syn::parse_quote!(Option<T>);
        ty.replace_ident(&ident, &path);
        assert_eq!(ty.to_token_stream().to_string().replace(' ', ""), "Option<ryft::Placeholder>");

        let mut predicate: syn::WherePredicate = syn::parse_quote!(T: Into<Vec<T>>);
        predicate.replace_ident(&ident, &path);
        assert_eq!(
            predicate.to_token_stream().to_string().replace(' ', ""),
            "ryft::Placeholder:Into<Vec<ryft::Placeholder>>",
        );
    }
}
