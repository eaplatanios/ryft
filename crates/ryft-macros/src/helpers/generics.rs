use crate::helpers::ident::IdentHelpers;

/// Helper private module making sure that [`GenericsHelpers`] is a sealed trait. Refer to
/// [this page](https://predr.ag/blog/definitive-guide-to-sealed-traits-in-rust/) for more information
/// on private traits in Rust.
mod private {
    pub trait Sealed {}
    impl Sealed for syn::Generics {}
}

/// Defines helper functions for working with [`syn::Generics`]. This is defined as a trait that is implemented only for
/// [`syn::Generics`] in order to provide convenient dot-notation for these helpers, making them appear as if they are
/// already part of the [`syn::Generics`] interface.
pub trait GenericsHelpers: private::Sealed {
    /// Returns `true` if the generic parameter with the specified [`syn::Ident`] is bounded (either with simple bounds
    /// or with where clauses). Note that this function does not check whether any of the [`syn::GenericParam`]s in this
    /// instance have a matching [`syn::Ident`] to the provided one. If none do, it simply returns `false`, same as if
    /// parameter with a matching identifier is not bounded.
    fn is_param_bounded(&self, ident: &syn::Ident) -> bool;

    /// Returns `true` if the provided [`syn::Ident`] matches the identifiers of any of the [`syn::GenericParam`]s
    /// of this [`syn::Generics`] instance.
    fn matches_any_param(&self, ident: &syn::Ident) -> bool;

    /// Returns `true` if the provided [`syn::Lifetime`] references any of the [`syn::GenericParam`]s
    /// of this [`syn::Generics`] instance.
    fn referenced_by_lifetime(&self, lifetime: &syn::Lifetime) -> bool;

    /// Returns `true` if the provided [`syn::Type`] references any of the [`syn::GenericParam`]s of this
    /// [`syn::Generics`] instance. Note that any nested [`syn::Type::Infer`]s, [`syn::Type::Macro`]s,
    /// [`syn::Type::Verbatim`]s, and [`syn::TypeParamBound::Verbatim`]s, will not be visited by this function
    /// and so if there are implicit references there, they will not be detected.
    fn referenced_by_type(&self, ty: &syn::Type) -> bool;

    /// Returns `true` if the provided [`syn::ReturnType`] references any of the [`syn::GenericParam`]s of
    /// this [`syn::Generics`] instance. Note that any nested [`syn::Type::Infer`]s, [`syn::Type::Macro`]s,
    /// [`syn::Type::Verbatim`]s, and [`syn::TypeParamBound::Verbatim`]s, will not be visited by this function
    /// and so if there are implicit references there, they will not be detected.
    fn referenced_by_return_type(&self, return_type: &syn::ReturnType) -> bool;

    /// Returns `true` if the provided [`syn::TypeParamBound`] references any of the [`syn::GenericParam`]s of
    /// this [`syn::Generics`] instance. Note that any nested [`syn::Type::Infer`]s, [`syn::Type::Macro`]s,
    /// [`syn::Type::Verbatim`]s, and [`syn::TypeParamBound::Verbatim`]s, will not be visited by this function
    /// and so if there are implicit references there, they will not be detected.
    fn referenced_by_type_param_bound(&self, type_param_bound: &syn::TypeParamBound) -> bool;

    /// Returns `true` if the provided [`syn::TypePath`] references any of the [`syn::GenericParam`]s of this
    /// [`syn::Generics`] instance. Note that any nested [`syn::Type::Infer`]s, [`syn::Type::Macro`]s,
    /// [`syn::Type::Verbatim`]s, and [`syn::TypeParamBound::Verbatim`]s, will not be visited by this function
    /// and so if there are implicit references there, they will not be detected.
    fn referenced_by_type_path(&self, type_path: &syn::TypePath) -> bool;

    /// Returns `true` if the provided [`syn::Path`] references any of the [`syn::GenericParam`]s of this
    /// [`syn::Generics`] instance.
    fn referenced_by_path(&self, path: &syn::Path) -> bool;

    /// Returns `true` if the provided [`syn::CapturedParam`] references any of the [`syn::GenericParam`]s
    /// of this [`syn::Generics`] instance.
    fn referenced_by_captured_param(&self, captured_param: &syn::CapturedParam) -> bool;

    // TODO(eaplatanios): Document this. Mention that it does not check if that lifetime param already exists.
    fn with_lifetime(&self, lifetime: &syn::Lifetime, param: &syn::Ident) -> Self;

    // TODO(eaplatanios): Document this. Mention that it does not check if the new param name already exists.
    fn with_renamed_param(&self, old_name: &syn::Ident, new_name: &syn::Ident) -> Self;

    // TODO(eaplatanios): Document this.
    fn with_concrete_param(&self, param: &syn::Ident, replacement: &syn::Path) -> Self;

    /// Returns a copy of this [`syn::Generics`] instance with the provided [`syn::TypeParamBound`]s added for each
    /// of the provided [`syn::TypePath`]s in this [`syn::Generics`] instance, except for any that do not reference
    /// any of the [`syn::GenericParam`]s of this instance, which are left unchanged.
    fn with_bounds<'t, T: IntoIterator<Item = &'t syn::Type>, B: IntoIterator<Item = syn::TypeParamBound>>(
        &self,
        types: T,
        bounds: B,
    ) -> Self;

    /// Constructs a [`syn::Generics`] instance that is a clone of this instance except that it includes the
    /// provided [`syn::WherePredicates`] in addition to any [`syn::WherePredicates`] it may have already included.
    fn with_where_predicates(&self, predicates: &[syn::WherePredicate]) -> Self;

    /// Constructs a [`syn::Generics`] instance that is a clone of this instance except that it has all of its defaults
    /// removed. That includes both generic type parameter defaults and const generic defaults. This is useful for
    /// situations where the generic types end up as associated types in generated `impl`s and you get errors like
    /// this one: `error: associated type bindings are not allowed here`.
    fn without_defaults(&self) -> Self;

    /// Constructs a [`syn::Generics`] instance that is a clone of this instance except that it does not include the
    /// provided generic parameters (if it already does not include some or all of them, that is ok; this function
    /// will simply ignore any parameters that do not appear in this instance).
    ///
    /// Note that this function not only filters [`syn::Generics::params`], but it also filters
    /// [`syn::Generics::where_clause`] as needed. That is because if we remove a generic parameter that appears in the
    /// where clauses but we do not remove the corresponding clauses, that will result in invalid generated code.
    ///
    /// This function does not perform any checks on whether the removed parameters appears in any of the bounds of the
    /// remaining parameters. It is the caller's responsibility to ensure that the removal is correct and safe.
    fn without_params<'p, P: IntoIterator<Item = &'p syn::Ident>>(&self, params_to_remove: P) -> Self;
}

impl GenericsHelpers for syn::Generics {
    fn is_param_bounded(&self, ident: &syn::Ident) -> bool {
        self.params.iter().any(|generic_param| match generic_param {
            syn::GenericParam::Lifetime(p) if p.ident() == Some(ident) && !p.bounds.is_empty() => true,
            syn::GenericParam::Type(p) if p.ident() == Some(ident) && !p.bounds.is_empty() => true,
            _ => false,
        }) || self
            .where_clause
            .iter()
            .any(|where_clause| where_clause.predicates.iter().any(|predicate| predicate.references_ident(ident)))
    }

    fn matches_any_param(&self, ident: &syn::Ident) -> bool {
        // Note that we could precompute a [`HashSet`] with all the parameter identifiers. We do not currently do that
        // because we assume that the number of generic parameters is usually quite small and so looping over them
        // should be sufficiently fast.
        self.params.iter().any(|param| param.ident() == Some(ident))
    }

    fn referenced_by_lifetime(&self, lifetime: &syn::Lifetime) -> bool {
        self.matches_any_param(&lifetime.ident)
    }

    fn referenced_by_type(&self, ty: &syn::Type) -> bool {
        match ty {
            // TODO(eaplatanios): Uncomment this once `non_exhaustive_omitted_patterns_lint` is stabilized.
            // #![cfg_attr(test, deny(non_exhaustive_omitted_patterns))]
            syn::Type::Array(type_array) => self.referenced_by_type(&type_array.elem),
            syn::Type::BareFn(type_bare_fn) => {
                type_bare_fn.inputs.iter().any(|input| self.referenced_by_type(&input.ty))
                    || self.referenced_by_return_type(&type_bare_fn.output)
            }
            syn::Type::Group(type_group) => self.referenced_by_type(&type_group.elem),
            syn::Type::ImplTrait(type_impl_trait) => {
                type_impl_trait.bounds.iter().any(|bound| self.referenced_by_type_param_bound(bound))
            }
            syn::Type::Infer(_) | syn::Type::Macro(_) | syn::Type::Never(_) => false,
            syn::Type::Paren(type_paren) => self.referenced_by_type(&type_paren.elem),
            syn::Type::Path(type_path) => self.referenced_by_type_path(type_path),
            syn::Type::Ptr(type_ptr) => self.referenced_by_type(&type_ptr.elem),
            syn::Type::Reference(type_reference) => self.referenced_by_type(&type_reference.elem),
            syn::Type::Slice(type_slice) => self.referenced_by_type(&type_slice.elem),
            syn::Type::TraitObject(type_trait_object) => {
                type_trait_object.bounds.iter().any(|bound| self.referenced_by_type_param_bound(bound))
            }
            syn::Type::Tuple(type_tuple) => type_tuple.elems.iter().any(|elem| self.referenced_by_type(elem)),
            syn::Type::Verbatim(_) => false,
            _ => false,
        }
    }

    fn referenced_by_return_type(&self, return_type: &syn::ReturnType) -> bool {
        match return_type {
            syn::ReturnType::Default => false,
            syn::ReturnType::Type(_, output) => self.referenced_by_type(output),
        }
    }

    fn referenced_by_type_param_bound(&self, type_param_bound: &syn::TypeParamBound) -> bool {
        match type_param_bound {
            // TODO(eaplatanios): Uncomment this once `non_exhaustive_omitted_patterns_lint` is stabilized.
            // #![cfg_attr(test, deny(non_exhaustive_omitted_patterns))]
            syn::TypeParamBound::Trait(bound) => self.referenced_by_path(&bound.path),
            syn::TypeParamBound::Lifetime(lifetime) => self.referenced_by_lifetime(lifetime),
            syn::TypeParamBound::PreciseCapture(capture) => {
                capture.params.iter().any(|param| self.referenced_by_captured_param(param))
            }
            syn::TypeParamBound::Verbatim(_) => false,
            _ => false,
        }
    }

    fn referenced_by_type_path(&self, type_path: &syn::TypePath) -> bool {
        type_path.qself.iter().any(|qself| self.referenced_by_type(&qself.ty))
            || self.referenced_by_path(&type_path.path)
    }

    fn referenced_by_path(&self, path: &syn::Path) -> bool {
        path.get_ident().iter().any(|ident| self.matches_any_param(ident))
    }

    fn referenced_by_captured_param(&self, captured_param: &syn::CapturedParam) -> bool {
        match captured_param {
            // TODO(eaplatanios): Uncomment this once `non_exhaustive_omitted_patterns_lint` is stabilized.
            // #![cfg_attr(test, deny(non_exhaustive_omitted_patterns))]
            syn::CapturedParam::Lifetime(lifetime) => self.referenced_by_lifetime(lifetime),
            syn::CapturedParam::Ident(ident) => self.matches_any_param(ident),
            _ => false,
        }
    }

    fn with_lifetime(&self, lifetime: &syn::Lifetime, param: &syn::Ident) -> Self {
        let mut params = self
            .params
            .iter()
            .map(|generic_param| match generic_param {
                syn::GenericParam::Lifetime(p) if p.matches_ident(param) => {
                    let mut bounds = p.bounds.clone();
                    bounds.push(lifetime.clone());
                    syn::GenericParam::Lifetime(syn::LifetimeParam { bounds, ..p.clone() })
                }
                syn::GenericParam::Type(p) if p.matches_ident(param) => {
                    let mut bounds = p.bounds.clone();
                    bounds.push(syn::TypeParamBound::Lifetime(lifetime.clone()));
                    syn::GenericParam::Type(syn::TypeParam { bounds, ..p.clone() })
                }
                _ => generic_param.clone(),
            })
            .collect::<syn::punctuated::Punctuated<syn::GenericParam, syn::Token![,]>>();
        params.push(syn::GenericParam::Lifetime(syn::LifetimeParam::new(lifetime.clone())));
        syn::Generics { params, ..self.clone() }
    }

    fn with_renamed_param(&self, old_name: &syn::Ident, new_name: &syn::Ident) -> Self {
        let mut generics = self.clone();
        let replacement = syn::Path::from(new_name.clone());
        generics.replace_ident(old_name, &replacement);
        generics
    }

    fn with_concrete_param(&self, param: &syn::Ident, replacement: &syn::Path) -> Self {
        let mut generics = self.clone();
        generics.params = generics.params.iter().filter(|p| p.ident() != Some(param)).cloned().collect();
        generics.replace_ident(param, replacement);
        generics
    }

    fn with_bounds<'t, T: IntoIterator<Item = &'t syn::Type>, B: IntoIterator<Item = syn::TypeParamBound>>(
        &self,
        types: T,
        bounds: B,
    ) -> Self {
        let types = types.into_iter().filter(|ty| self.referenced_by_type(ty)).collect::<Vec<_>>();
        let bounds = syn::punctuated::Punctuated::from_iter(bounds);
        let mut generics = self.clone();
        generics.make_where_clause().predicates.extend(types.into_iter().map(|ty| {
            syn::WherePredicate::Type(syn::PredicateType {
                lifetimes: None,
                bounded_ty: ty.clone(),
                colon_token: <syn::Token![:]>::default(),
                bounds: bounds.clone(),
            })
        }));
        generics
    }

    fn with_where_predicates(&self, predicates: &[syn::WherePredicate]) -> Self {
        let mut generics = self.clone();
        generics.make_where_clause().predicates.extend(predicates.iter().cloned());
        generics
    }

    fn without_defaults(&self) -> Self {
        syn::Generics {
            params: self
                .params
                .iter()
                .map(|param| match param {
                    syn::GenericParam::Lifetime(_) => param.clone(),
                    syn::GenericParam::Type(param) => {
                        syn::GenericParam::Type(syn::TypeParam { eq_token: None, default: None, ..param.clone() })
                    }
                    syn::GenericParam::Const(param) => {
                        syn::GenericParam::Const(syn::ConstParam { eq_token: None, default: None, ..param.clone() })
                    }
                })
                .collect(),
            ..self.clone()
        }
    }

    fn without_params<'p, P: IntoIterator<Item = &'p syn::Ident>>(&self, params_to_remove: P) -> Self {
        let params_to_remove = params_to_remove.into_iter().collect::<Vec<&'p syn::Ident>>();

        let params = self
            .params
            .iter()
            .filter(|param| !params_to_remove.iter().any(|p| param.matches_ident(p)))
            .cloned()
            .collect::<syn::punctuated::Punctuated<syn::GenericParam, syn::Token![,]>>();

        let where_clause = self.where_clause.as_ref().map(|where_clause| syn::WhereClause {
            where_token: where_clause.where_token,
            predicates: where_clause
                .predicates
                .iter()
                .filter(|predicate| !params_to_remove.iter().any(|param| predicate.references_ident(param)))
                .cloned()
                .collect(),
        });

        syn::Generics { params, where_clause, ..self.clone() }
    }
}
