use proc_macro2::Span;
use quote::ToTokens;

use crate::helpers::span::with_span;

/// Replaces instances of the `Self` type in the provided [`DeriveInput`] with the corresponding fully-qualified
/// "receiver" type. This is necessary in order to be able to handle recursive types when deriving trait
/// implementations.
///
/// Relevant issues that provide more information as to why and when this is necessary:
///   - https://github.com/serde-rs/serde/issues/1506
///   - https://github.com/serde-rs/serde/issues/1565
///
/// # Parameters
///
///   * `input` - [`DeriveInput`] for which to replace instances of the `Self` type. Note that this function
///     will mutate this [`DeriveInput`] instance directly. Note also that only structs and enums are supported.
///     [`DeriveInput`] instances that correspond to unions will be ignored by this function.
pub fn replace_self_type(input: &mut syn::DeriveInput) {
    let receiver_type = {
        let ident = &input.ident;
        let ty_generics = input.generics.split_for_impl().1;
        syn::parse_quote!(#ident #ty_generics)
    };
    let visitor = ReplaceSelf { receiver_type: &receiver_type };
    visitor.process_derive_input(input);
}

/// Private helper struct that is used to implement [`replace_self_type`].
struct ReplaceSelf<'p> {
    /// Fully-qualified "receiver" [`TypePath`] to use when replacing `Self`.
    receiver_type: &'p syn::TypePath,
}

impl ReplaceSelf<'_> {
    /// Returns a copy of the fully-qualified `Self` [`TypePath`] using the provided [`Span`].
    fn receiver_type(&self, span: Span) -> syn::TypePath {
        syn::parse2(with_span(self.receiver_type.to_token_stream(), span)).unwrap()
    }

    /// Replaces all instances of `Self` in the provided [`DeriveInput`] with [`Self::receiver_type`].
    fn process_derive_input(&self, derive_input: &mut syn::DeriveInput) {
        self.process_generics(&mut derive_input.generics);
        self.process_data(&mut derive_input.data);
    }

    /// Replaces all instances of `Self` in the provided [`Generics`] with [`Self::receiver_type`].
    fn process_generics(&self, generics: &mut syn::Generics) {
        // Process generic parameters.
        generics.params.iter_mut().for_each(|param| match param {
            syn::GenericParam::Lifetime(_) => {}
            syn::GenericParam::Type(param) => {
                param.bounds.iter_mut().for_each(|bound| self.process_type_param_bound(bound))
            }
            syn::GenericParam::Const(_) => {}
        });

        // Process the where clause, if one exists.
        if let Some(where_clause) = &mut generics.where_clause {
            where_clause.predicates.iter_mut().for_each(|predicate| match predicate {
                // TODO(eaplatanios): Uncomment this once `non_exhaustive_omitted_patterns_lint` is stabilized.
                // #![cfg_attr(test, deny(non_exhaustive_omitted_patterns))]
                syn::WherePredicate::Lifetime(_) => {}
                syn::WherePredicate::Type(predicate) => {
                    self.process_type(&mut predicate.bounded_ty);
                    for bound in &mut predicate.bounds {
                        self.process_type_param_bound(bound);
                    }
                }
                _ => {}
            });
        }
    }

    /// Replaces all instances of `Self` in the provided [`Data`] with [`Self::receiver_type`].
    fn process_data(&self, data: &mut syn::Data) {
        match data {
            syn::Data::Struct(data) => data.fields.iter_mut().for_each(|field| self.process_type(&mut field.ty)),
            syn::Data::Enum(data) => {
                data.variants
                    .iter_mut()
                    .for_each(|variant| variant.fields.iter_mut().for_each(|field| self.process_type(&mut field.ty)));
            }
            syn::Data::Union(_) => {}
        }
    }

    /// Replaces all instances of `Self` in the provided [`Type`] with [`Self::receiver_type`].
    fn process_type(&self, ty: &mut syn::Type) {
        match ty {
            // TODO(eaplatanios): Uncomment this once `non_exhaustive_omitted_patterns_lint` is stabilized.
            // #![cfg_attr(test, deny(non_exhaustive_omitted_patterns))]
            syn::Type::Array(ty) => {
                self.process_type(&mut ty.elem);
                self.process_expression(&mut ty.len);
            }
            syn::Type::BareFn(ty) => {
                ty.inputs.iter_mut().for_each(|input| self.process_type(&mut input.ty));
                self.process_return_type(&mut ty.output);
            }
            syn::Type::Group(ty) => self.process_type(&mut ty.elem),
            syn::Type::ImplTrait(ty) => ty.bounds.iter_mut().for_each(|bound| self.process_type_param_bound(bound)),
            syn::Type::Infer(_) | syn::Type::Macro(_) | syn::Type::Never(_) => {}
            syn::Type::Paren(ty) => self.process_type(&mut ty.elem),
            syn::Type::Path(ty) => {
                // In this branch, we perform replacements of the form `Self::Assoc` -> `<Receiver>::Assoc`.
                match &mut ty.qself {
                    None => self.self_to_qself(&mut ty.qself, &mut ty.path),
                    Some(qself) => self.process_type(&mut qself.ty),
                }
                self.process_path(&mut ty.path);
            }
            syn::Type::Ptr(ty) => self.process_type(&mut ty.elem),
            syn::Type::Reference(ty) => self.process_type(&mut ty.elem),
            syn::Type::Slice(ty) => self.process_type(&mut ty.elem),
            syn::Type::TraitObject(ty) => ty.bounds.iter_mut().for_each(|bound| self.process_type_param_bound(bound)),
            syn::Type::Tuple(ty) => ty.elems.iter_mut().for_each(|elem| self.process_type(elem)),
            syn::Type::Verbatim(_) => {}
            _ => {}
        }
    }

    /// Replaces all instances of `Self` in the provided [`Expr`] with [`Self::receiver_type`].
    ///
    /// Note that this function is not general-purpose. It assumes that it is being called in the context of a
    /// [`ReplaceSelf`] instance and thus only processes expression types that can appear in the context of the types
    /// that such an instance processes. This means that it only processes constant expressions that compute the
    /// length of arrays (and is only ever called in that context, though it also recurses into such expressions).
    fn process_expression(&self, expr: &mut syn::Expr) {
        match expr {
            // TODO(eaplatanios): Uncomment this once `non_exhaustive_omitted_patterns_lint` is stabilized.
            // #![cfg_attr(test, deny(non_exhaustive_omitted_patterns))]
            syn::Expr::Array(_) | syn::Expr::Assign(_) | syn::Expr::Async(_) | syn::Expr::Await(_) => {}
            syn::Expr::Binary(expr) => {
                self.process_expression(&mut expr.left);
                self.process_expression(&mut expr.right);
            }
            syn::Expr::Block(_) | syn::Expr::Break(_) => {}
            syn::Expr::Call(expr) => {
                self.process_expression(&mut expr.func);
                expr.args.iter_mut().for_each(|arg| self.process_expression(arg));
            }
            syn::Expr::Cast(expr) => {
                self.process_expression(&mut expr.expr);
                self.process_type(&mut expr.ty);
            }
            syn::Expr::Closure(_) | syn::Expr::Const(_) | syn::Expr::Continue(_) => {}
            syn::Expr::Field(expr) => self.process_expression(&mut expr.base),
            syn::Expr::ForLoop(_) | syn::Expr::Group(_) | syn::Expr::If(_) => {}
            syn::Expr::Index(expr) => {
                self.process_expression(&mut expr.expr);
                self.process_expression(&mut expr.index);
            }
            syn::Expr::Infer(_)
            | syn::Expr::Let(_)
            | syn::Expr::Lit(_)
            | syn::Expr::Loop(_)
            | syn::Expr::Macro(_)
            | syn::Expr::Match(_)
            | syn::Expr::MethodCall(_) => {}
            syn::Expr::Paren(expr) => self.process_expression(&mut expr.expr),
            syn::Expr::Path(expr) => {
                // In this branch, we perform replacements of the form `Self::method` -> `<Receiver>::method`.
                match &mut expr.qself {
                    None => self.self_to_qself(&mut expr.qself, &mut expr.path),
                    Some(qself) => self.process_type(&mut qself.ty),
                }
                self.process_path(&mut expr.path);
            }
            syn::Expr::Range(_)
            | syn::Expr::RawAddr(_)
            | syn::Expr::Reference(_)
            | syn::Expr::Repeat(_)
            | syn::Expr::Return(_)
            | syn::Expr::Struct(_)
            | syn::Expr::Try(_)
            | syn::Expr::TryBlock(_)
            | syn::Expr::Tuple(_) => {}
            syn::Expr::Unary(expr) => self.process_expression(&mut expr.expr),
            syn::Expr::Unsafe(_) | syn::Expr::Verbatim(_) | syn::Expr::While(_) | syn::Expr::Yield(_) => {}
            _ => {}
        }
    }

    /// Replaces all instances of `Self` in the provided [`ReturnType`] with [`Self::receiver_type`].
    fn process_return_type(&self, return_type: &mut syn::ReturnType) {
        match return_type {
            syn::ReturnType::Default => {}
            syn::ReturnType::Type(_, output) => self.process_type(output),
        }
    }

    /// Replaces all instances of `Self` in the provided [`TypeParamBound`] with [`Self::receiver_type`].
    fn process_type_param_bound(&self, bound: &mut syn::TypeParamBound) {
        match bound {
            // TODO(eaplatanios): Uncomment this once `non_exhaustive_omitted_patterns_lint` is stabilized.
            // #![cfg_attr(test, deny(non_exhaustive_omitted_patterns))]
            syn::TypeParamBound::Trait(bound) => self.process_path(&mut bound.path),
            syn::TypeParamBound::Lifetime(_)
            | syn::TypeParamBound::PreciseCapture(_) // TODO(eaplatanios): Should we raise an error for this one?
            | syn::TypeParamBound::Verbatim(_) => {}
            _ => {}
        }
    }

    /// Replaces all instances of `Path` in the provided [`TypeParamBound`] with [`Self::receiver_type`].
    ///
    /// Note that this function is not general-purpose. It assumes that it is being called in the context of a
    /// [`ReplaceSelf`] instance. This means that it only processes the argument types that appear in path segments
    /// as, at this point, `Self` cannot appear as an identifier in a path segment.
    fn process_path(&self, path: &mut syn::Path) {
        path.segments.iter_mut().for_each(|segment| match &mut segment.arguments {
            syn::PathArguments::None => {}
            syn::PathArguments::AngleBracketed(arguments) => {
                arguments.args.iter_mut().for_each(|arg| match arg {
                    // TODO(eaplatanios): Uncomment this once `non_exhaustive_omitted_patterns_lint` is stabilized.
                    // #![cfg_attr(test, deny(non_exhaustive_omitted_patterns))]
                    syn::GenericArgument::Lifetime(_) => {}
                    syn::GenericArgument::Type(arg) => self.process_type(arg),
                    syn::GenericArgument::Const(_) => {}
                    syn::GenericArgument::AssocType(arg) => self.process_type(&mut arg.ty),
                    syn::GenericArgument::AssocConst(_) | syn::GenericArgument::Constraint(_) => {}
                    _ => {}
                });
            }
            syn::PathArguments::Parenthesized(arguments) => {
                arguments.inputs.iter_mut().for_each(|input| self.process_type(input));
                self.process_return_type(&mut arguments.output);
            }
        });
    }

    /// Sets `qself` to the fully-qualified receiver type, if not already set, and if `path` starts with `Self`.
    ///
    /// This function also removes the `Self` segment from `path`, when it modifies `qself`, replacing it with a
    /// leading `::`. Effectively, it converts paths like `Self::Type` or `Self::function` to `<Receiver>::Type`
    /// and `<Receiver>::function`, respectively.
    ///
    /// Note that if `qself` is [`None`] and `path` is just `Self`, then both `qself` and `path` will be set to the
    /// corresponding fields of [`Self::receiver_type`].
    ///
    /// Note that this function will only modify paths for which `qself` is [`None`], there is no leading `::`, and the
    /// first path segment is the `Self` identifier. That is because, `Self` should generally never appear after a `::`
    /// in a (valid) path. Note also that if `qself` is not [`None`], then `Self` should never appear in (valid) paths.
    fn self_to_qself(&self, qself: &mut Option<syn::QSelf>, path: &mut syn::Path) {
        if qself.is_none() && path.is_ident("Self") {
            let receiver_type = self.receiver_type(path.segments[0].ident.span());
            *qself = receiver_type.qself;
            *path = receiver_type.path;
        }

        if qself.is_some() || path.leading_colon.is_some() || path.segments[0].ident != "Self" {
            return;
        }

        // In the following, we add a `QSelf` with position `0` that is the fully-qualified `Self` type. Then, we
        // remove the first `Self` path segment and replace it with a leading path separator. Roughly, that results in
        // a conversion like `Self::...` to `<Vec<T>>::...`, where in this example, `Self = Vec<T>`.
        let span = path.segments[0].ident.span();
        *qself = Some(syn::QSelf {
            lt_token: syn::Token![<](span),
            ty: Box::new(syn::Type::Path(self.receiver_type(span))),
            position: 0,
            as_token: None,
            gt_token: syn::Token![>](span),
        });

        path.leading_colon = Some(syn::Token![::](span));

        let segments = std::mem::take(&mut path.segments);
        path.segments = segments.into_pairs().skip(1).collect();
    }
}
