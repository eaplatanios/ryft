//! Expansion of the `#[capability]` attribute, which validates the conventions of capability traits and generates the
//! implementations through which composite values apply a capability by projecting onto their members.

// TODO(eaplatanios): Review this module.

use proc_macro2::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::{
    FnArg, GenericArgument, GenericParam, Ident, ItemTrait, LitStr, Pat, Path, PathArguments, PathSegment, ReturnType,
    Token, TraitItem, TraitItemFn, Type, TypeParamBound, WherePredicate, parenthesized,
};

/// Projection requested by a `projection(Composite => Member)` attribute argument.
struct Projection {
    /// Universe of the composite values that apply the capability by projecting onto their members.
    composite: Type,

    /// Universe of the member values onto which composite values project.
    member: Type,
}

/// Parsed arguments of the `#[capability]` attribute.
struct Arguments {
    /// Path of the `ryft` crate (or of a crate that re-exports its core items) used in generated code.
    core: Path,

    /// Requested projections, in declaration order.
    projections: Vec<Projection>,
}

impl Parse for Arguments {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut core: Path = syn::parse_quote!(ryft);
        let mut projections = Vec::new();
        while !input.is_empty() {
            if input.peek(Token![crate]) {
                input.parse::<Token![crate]>()?;
                input.parse::<Token![=]>()?;
                core = input.parse::<LitStr>()?.parse()?;
            } else {
                let keyword = input.parse::<Ident>()?;
                if keyword != "projection" {
                    return Err(syn::Error::new_spanned(
                        keyword,
                        "expected `projection(Composite => Member)` or `crate = \"path\"`",
                    ));
                }
                let content;
                parenthesized!(content in input);
                let composite = content.parse()?;
                content.parse::<Token![=>]>()?;
                let member = content.parse()?;
                projections.push(Projection { composite, member });
            }
            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }
        Ok(Self { core, projections })
    }
}

// TODO(eaplatanios): Review this.
/// Expands one `#[capability]` attribute: validates the annotated capability trait and appends one projection
/// implementation per `projection(Composite => Member)` argument.
pub(crate) fn expand(attributes: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
    let arguments: Arguments = syn::parse2(attributes)?;
    let item: ItemTrait = syn::parse2(item)?;
    let universe = universe_parameter_index(&item)?;
    let implementations = arguments
        .projections
        .iter()
        .map(|projection| projection_implementation(&item, universe, projection, &arguments.core))
        .collect::<syn::Result<Vec<_>>>()?;
    Ok(quote! {
        #item
        #(#implementations)*
    })
}

/// Validates the capability conventions of `item` and returns the position of its universe parameter: exactly one type
/// parameter defaults to `<Self as Capability>::Universe`, that parameter is unbounded, and `Capability` is a direct
/// supertrait.
fn universe_parameter_index(item: &ItemTrait) -> syn::Result<usize> {
    let mut universes = item.generics.params.iter().enumerate().filter(|(_, parameter)| {
        matches!(parameter, GenericParam::Type(parameter)
            if parameter.default.as_ref().is_some_and(is_universe_default))
    });
    let Some((index, GenericParam::Type(universe))) = universes.next() else {
        return Err(syn::Error::new_spanned(
            &item.ident,
            "capability traits must declare a type parameter that defaults to `<Self as Capability>::Universe`",
        ));
    };
    if let Some((_, duplicate)) = universes.next() {
        return Err(syn::Error::new_spanned(
            duplicate,
            "capability traits must declare exactly one universe parameter",
        ));
    }

    // Host types are their own universes, so a bound such as `T: Type` would exclude them from the capability.
    let bounded_in_where_clause = item.generics.where_clause.as_ref().is_some_and(|where_clause| {
        where_clause.predicates.iter().any(|predicate| {
            matches!(predicate, WherePredicate::Type(predicate)
                if matches!(&predicate.bounded_ty, Type::Path(path) if path.qself.is_none()
                    && path.path.is_ident(&universe.ident)))
        })
    });
    if !universe.bounds.is_empty() || bounded_in_where_clause {
        return Err(syn::Error::new_spanned(
            &universe.ident,
            format!(
                "the universe parameter `{}` of a capability must not be bounded, because host types are their own \
                 universes",
                universe.ident,
            ),
        ));
    }

    let has_capability_supertrait = item.supertraits.iter().any(|bound| {
        matches!(bound, TypeParamBound::Trait(bound)
            if bound.path.segments.last().is_some_and(|segment| segment.ident == "Capability"))
    });
    if !has_capability_supertrait {
        return Err(syn::Error::new_spanned(&item.ident, "capability traits must have `Capability` as a supertrait"));
    }
    Ok(index)
}

/// Returns `true` if `default` is `<Self as Capability>::Universe`, with any path to `Capability`.
fn is_universe_default(default: &Type) -> bool {
    let Type::Path(path) = default else {
        return false;
    };
    let Some(qualified_self) = &path.qself else {
        return false;
    };
    let segments = &path.path.segments;
    is_self(&qualified_self.ty)
        && qualified_self.position + 1 == segments.len()
        && segments[qualified_self.position - 1].ident == "Capability"
        && segments[qualified_self.position].ident == "Universe"
}

/// Returns `true` if `type` is exactly `Self`.
fn is_self(r#type: &Type) -> bool {
    matches!(r#type, Type::Path(path) if path.qself.is_none() && path.path.is_ident("Self"))
}

/// Returns `true` if `type` mentions `Self` anywhere.
fn mentions_self(r#type: &Type) -> bool {
    /// Records whether any visited path segment is `Self`.
    struct SelfFinder {
        /// Whether a `Self` path segment was visited.
        found: bool,
    }

    impl<'ast> Visit<'ast> for SelfFinder {
        fn visit_path_segment(&mut self, segment: &'ast PathSegment) {
            self.found |= segment.ident == "Self";
            syn::visit::visit_path_segment(self, segment);
        }
    }

    let mut finder = SelfFinder { found: false };
    finder.visit_type(r#type);
    finder.found
}

/// Generates the implementation of the capability for every value of `projection.composite` whose
/// [`ValueProjection`] onto `projection.member` yields a value that implements the capability itself.
fn projection_implementation(
    item: &ItemTrait,
    universe: usize,
    projection: &Projection,
    core: &Path,
) -> syn::Result<TokenStream> {
    let Projection { composite, member } = projection;
    let projected = quote!(<__V as #core::ValueProjection<#member>>::Projected);

    // Type parameters other than the universe must default to `Self` (e.g., the right input of `Dot`), so that they
    // resolve to the composite value in the implemented trait and to its projection in the delegated one.
    let mut self_parameters = Vec::new();
    let mut implemented_arguments = Vec::new();
    let mut projected_arguments = Vec::new();
    for (index, parameter) in item.generics.params.iter().enumerate() {
        if index == universe {
            implemented_arguments.push(quote!(#composite));
            projected_arguments.push(quote!(#member));
            continue;
        }
        match parameter {
            GenericParam::Type(parameter) if index < universe && parameter.default.as_ref().is_some_and(is_self) => {
                self_parameters.push(parameter.ident.clone());
                implemented_arguments.push(quote!(__V));
                projected_arguments.push(projected.clone());
            }
            _ => {
                return Err(syn::Error::new_spanned(
                    parameter,
                    "projected capabilities only support type parameters that precede the universe parameter and \
                     default to `Self`",
                ));
            }
        }
    }

    let trait_name = &item.ident;
    let functions = item
        .items
        .iter()
        .filter_map(|trait_item| match trait_item {
            TraitItem::Fn(function) if function.default.is_some() => None,
            TraitItem::Fn(function) => {
                Some(projected_function(function, &self_parameters, trait_name, &projected_arguments, member, core))
            }
            other => Some(Err(syn::Error::new_spanned(other, "projected capabilities only support functions"))),
        })
        .collect::<syn::Result<Vec<_>>>()?;
    Ok(quote! {
        impl<__V> #trait_name<#(#implemented_arguments),*> for __V
        where
            __V: #core::Value<Type = #composite> + #core::ValueProjection<#member>,
            #projected: #trait_name<#(#projected_arguments),*>,
        {
            #(#functions)*
        }
    })
}

/// Rewrites occurrences of the trait's `Self`-defaulted type parameters (e.g., `Rhs`) to `Self`, which is what they
/// resolve to in the generated implementation.
struct SelfParameterRewriter<'p> {
    /// Names of the trait's type parameters that default to `Self`.
    parameters: &'p [Ident],
}

impl VisitMut for SelfParameterRewriter<'_> {
    fn visit_type_mut(&mut self, r#type: &mut Type) {
        if let Type::Path(path) = r#type
            && path.qself.is_none()
            && self.parameters.iter().any(|parameter| path.path.is_ident(parameter))
        {
            *r#type = syn::parse_quote!(Self);
            return;
        }
        syn::visit_mut::visit_type_mut(self, r#type);
    }
}

/// Generates one function of a projection implementation. The function projects the receiver and every `&Self`,
/// `&[Self]`, and `Option<&Self>` input onto the member universe, applies the member implementation, and lifts its
/// `Self` outputs back.
fn projected_function(
    function: &TraitItemFn,
    self_parameters: &[Ident],
    trait_name: &Ident,
    projected_arguments: &[TokenStream],
    member: &Type,
    core: &Path,
) -> syn::Result<TokenStream> {
    let mut signature = function.sig.clone();
    SelfParameterRewriter { parameters: self_parameters }.visit_signature_mut(&mut signature);
    let projection = quote!(#core::ValueProjection::<#member>::into_projected);

    let mut inputs = signature.inputs.iter();
    match inputs.next() {
        Some(FnArg::Receiver(receiver)) if receiver.reference.is_some() && receiver.mutability.is_none() => {}
        _ => return Err(syn::Error::new_spanned(&signature, "projected capability functions must take `&self`")),
    }

    let mut bindings = Vec::new();
    let mut arguments = Vec::new();
    for input in inputs {
        let FnArg::Typed(input) = input else {
            return Err(syn::Error::new_spanned(input, "projected capability functions must take `&self`"));
        };
        let Pat::Ident(name) = input.pat.as_ref() else {
            return Err(syn::Error::new_spanned(
                &input.pat,
                "projected capability inputs must be named by identifiers",
            ));
        };
        let name = &name.ident;
        match input.ty.as_ref() {
            Type::Reference(reference) if reference.mutability.is_none() && is_self(&reference.elem) => {
                bindings.push(quote!(let #name = #projection(::core::clone::Clone::clone(#name))?;));
                arguments.push(quote!(&#name));
            }
            Type::Reference(reference)
                if reference.mutability.is_none()
                    && matches!(reference.elem.as_ref(), Type::Slice(slice) if is_self(&slice.elem)) =>
            {
                bindings.push(quote! {
                    let #name = #name
                        .iter()
                        .map(|value| { #projection(::core::clone::Clone::clone(value)) })
                        .collect::<::core::result::Result<::std::vec::Vec<_>, _>>()?;
                });
                arguments.push(quote!(#name.as_slice()));
            }
            r#type if optional_self_reference(r#type) => {
                bindings.push(quote! {
                    let #name = #name.map(|value| { #projection(::core::clone::Clone::clone(value)) }).transpose()?;
                });
                arguments.push(quote!(#name.as_ref()));
            }
            r#type if !mentions_self(r#type) => arguments.push(quote!(#name)),
            r#type => {
                return Err(syn::Error::new_spanned(
                    r#type,
                    "projected capability inputs that mention `Self` must be `&Self`, `&[Self]`, or `Option<&Self>`",
                ));
            }
        }
    }

    let ReturnType::Type(_, output) = &signature.output else {
        return Err(syn::Error::new_spanned(&signature, "projected capability functions must return a `Result`"));
    };
    let Some(output) = result_value_type(output) else {
        return Err(syn::Error::new_spanned(output, "projected capability functions must return a `Result`"));
    };
    let lifted = lift(output, quote!(output), member, core)?;
    let function_name = &signature.ident;
    Ok(quote! {
        #[inline]
        #signature {
            let receiver = #projection(::core::clone::Clone::clone(self))?;
            #(#bindings)*
            let output = <<__V as #core::ValueProjection<#member>>::Projected as #trait_name<#(#projected_arguments),*>>
                ::#function_name(&receiver, #(#arguments),*)?;
            ::core::result::Result::Ok(#lifted)
        }
    })
}

/// Returns `true` if `type` is `Option<&Self>`.
fn optional_self_reference(r#type: &Type) -> bool {
    let Type::Path(path) = r#type else {
        return false;
    };
    let Some(segment) = path.path.segments.last() else {
        return false;
    };
    let PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return false;
    };
    segment.ident == "Option"
        && arguments.args.len() == 1
        && matches!(&arguments.args[0], GenericArgument::Type(Type::Reference(reference))
            if reference.mutability.is_none() && is_self(&reference.elem))
}

/// Returns the success type `R` of a `Result<R, E>` type.
fn result_value_type(r#type: &Type) -> Option<&Type> {
    let Type::Path(path) = r#type else {
        return None;
    };
    let segment = path.path.segments.last()?;
    let PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return None;
    };
    match (segment.ident == "Result", arguments.args.first()) {
        (true, Some(GenericArgument::Type(value))) if arguments.args.len() == 2 => Some(value),
        _ => None,
    }
}

/// Lifts the projected value `expression` of type `type` (with `Self` read as the projected member) back into the
/// composite universe: `Self` is lifted through [`ValueProjection::from_projected`], `Vec<Self>` element by element,
/// tuples component by component, and types that do not mention `Self` are passed through unchanged.
fn lift(r#type: &Type, expression: TokenStream, member: &Type, core: &Path) -> syn::Result<TokenStream> {
    let from_projected = quote!(<Self as #core::ValueProjection<#member>>::from_projected);
    if is_self(r#type) {
        return Ok(quote!(#from_projected(#expression)));
    }
    if !mentions_self(r#type) {
        return Ok(expression);
    }
    match r#type {
        Type::Path(path)
            if path.path.segments.last().is_some_and(|segment| {
                segment.ident == "Vec"
                    && matches!(&segment.arguments, PathArguments::AngleBracketed(arguments)
                        if arguments.args.len() == 1
                            && matches!(&arguments.args[0], GenericArgument::Type(element) if is_self(element)))
            }) =>
        {
            Ok(quote!(#expression.into_iter().map(#from_projected).collect()))
        }
        Type::Tuple(tuple) => {
            let names =
                (0..tuple.elems.len()).map(|index| quote::format_ident!("__output_{index}")).collect::<Vec<_>>();
            let components = tuple
                .elems
                .iter()
                .zip(&names)
                .map(|(element, name)| lift(element, quote!(#name), member, core))
                .collect::<syn::Result<Vec<_>>>()?;
            Ok(quote!({
                let (#(#names,)*) = #expression;
                (#(#components,)*)
            }))
        }
        _ => Err(syn::Error::new_spanned(
            r#type,
            "projected capability functions must return `Self`, `Vec<Self>`, tuples of those, or types that do not \
             mention `Self`",
        )),
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    /// Expands `item` with `attributes` and returns the rendered error message.
    fn error(attributes: TokenStream, item: TokenStream) -> String {
        expand(attributes, item).unwrap_err().to_string()
    }

    #[test]
    fn test_expand_validation() {
        // A valid capability expands to itself.
        let item = quote! {
            pub trait Double<T = <Self as Capability>::Universe>: Capability + Sized {
                fn double(&self) -> Result<Self, ProgramError>;
            }
        };
        let expanded: syn::File = syn::parse2(expand(quote!(), item.clone()).unwrap()).unwrap();
        let original: syn::File = syn::parse2(item).unwrap();
        assert_eq!(expanded, original);

        // Universe parameters are recognized through any path to `Capability`, as in macro-generated traits.
        assert!(
            expand(
                quote!(),
                quote! {
                    pub trait Double<T = <Self as crate::operations::Capability>::Universe>:
                        crate::operations::Capability + Clone {}
                },
            )
            .is_ok(),
        );

        assert_eq!(
            error(
                quote!(),
                quote!(
                    pub trait Double: Capability {}
                )
            ),
            "capability traits must declare a type parameter that defaults to `<Self as Capability>::Universe`",
        );
        assert_eq!(
            error(
                quote!(),
                quote!(
                    pub trait Double<T: Type = <Self as Capability>::Universe>: Capability {}
                )
            ),
            "the universe parameter `T` of a capability must not be bounded, because host types are their own \
             universes",
        );
        assert_eq!(
            error(
                quote!(),
                quote!(
                    pub trait Double<T = <Self as Capability>::Universe>: Capability
                    where
                        T: Type,
                    {
                    }
                )
            ),
            "the universe parameter `T` of a capability must not be bounded, because host types are their own \
             universes",
        );
        assert_eq!(
            error(
                quote!(),
                quote!(
                    pub trait Double<T = <Self as Capability>::Universe>: Sized {}
                )
            ),
            "capability traits must have `Capability` as a supertrait",
        );
        assert_eq!(
            error(
                quote!(projection),
                quote!(
                    pub trait Double<T = <Self as Capability>::Universe>: Capability {}
                )
            ),
            "unexpected end of input, expected parentheses",
        );
        assert_eq!(
            error(
                quote!(projections(A => B)),
                quote!(
                    pub trait Double<T = <Self as Capability>::Universe>: Capability {}
                ),
            ),
            "expected `projection(Composite => Member)` or `crate = \"path\"`",
        );
    }

    #[test]
    fn test_expand_projection() {
        let expanded = expand(
            quote!(crate = "core_alias", projection(ArrayIrType => ArrayType)),
            quote! {
                pub trait Combine<Rhs = Self, T = <Self as Capability>::Universe>: Capability + Sized {
                    fn combine<A: Into<Axis>>(
                        &self,
                        right: &Rhs,
                        others: &[Self],
                        mask: Option<&Self>,
                        axis: A,
                    ) -> Result<(Self, Vec<Self>, usize), ProgramError>;

                    fn rank(&self) -> Result<usize, ProgramError> {
                        Ok(0)
                    }
                }
            },
        )
        .unwrap();
        let file: syn::File = syn::parse2(expanded).unwrap();
        assert_eq!(file.items.len(), 2);

        // Provided functions keep their default bodies, which only call the projected required functions.
        let expanded = expand(
            quote!(projection(ArrayIrType => ArrayType)),
            quote! {
                pub trait Double<T = <Self as Capability>::Universe>: Capability + Sized {
                    fn double(&self) -> Result<Self, ProgramError>;

                    fn quadruple(&self) -> Result<Self, ProgramError> {
                        self.double()?.double()
                    }
                }
            },
        )
        .unwrap();
        let expanded: syn::File = syn::parse2(expanded).unwrap();
        let syn::Item::Impl(implementation) = &expanded.items[1] else {
            panic!("expected the projection implementation");
        };
        let functions = implementation
            .items
            .iter()
            .map(|item| match item {
                syn::ImplItem::Fn(function) => function.sig.ident.to_string(),
                _ => panic!("expected only functions"),
            })
            .collect::<Vec<_>>();
        assert_eq!(functions, vec!["double"]);
        let syn::Item::Impl(implementation) = &file.items[1] else {
            panic!("expected the projection implementation");
        };
        let expected: syn::ItemImpl = syn::parse_quote! {
            impl<__V> Combine<__V, ArrayIrType> for __V
            where
                __V: core_alias::Value<Type = ArrayIrType> + core_alias::ValueProjection<ArrayType>,
                <__V as core_alias::ValueProjection<ArrayType>>::Projected: Combine<
                    <__V as core_alias::ValueProjection<ArrayType>>::Projected,
                    ArrayType
                >,
            {
                #[inline]
                fn combine<A: Into<Axis>>(
                    &self,
                    right: &Self,
                    others: &[Self],
                    mask: Option<&Self>,
                    axis: A,
                ) -> Result<(Self, Vec<Self>, usize), ProgramError> {
                    let receiver =
                        core_alias::ValueProjection::<ArrayType>::into_projected(::core::clone::Clone::clone(self))?;
                    let right =
                        core_alias::ValueProjection::<ArrayType>::into_projected(::core::clone::Clone::clone(right))?;
                    let others = others
                        .iter()
                        .map(|value| {
                            core_alias::ValueProjection::<ArrayType>::into_projected(::core::clone::Clone::clone(value))
                        })
                        .collect::<::core::result::Result<::std::vec::Vec<_>, _>>()?;
                    let mask = mask
                        .map(|value| {
                            core_alias::ValueProjection::<ArrayType>::into_projected(::core::clone::Clone::clone(value))
                        })
                        .transpose()?;
                    let output = <<__V as core_alias::ValueProjection<ArrayType>>::Projected as Combine<
                        <__V as core_alias::ValueProjection<ArrayType>>::Projected,
                        ArrayType
                    >>::combine(&receiver, &right, others.as_slice(), mask.as_ref(), axis)?;
                    ::core::result::Result::Ok({
                        let (__output_0, __output_1, __output_2,) = output;
                        (
                            <Self as core_alias::ValueProjection<ArrayType>>::from_projected(__output_0),
                            __output_1
                                .into_iter()
                                .map(<Self as core_alias::ValueProjection<ArrayType>>::from_projected)
                                .collect(),
                            __output_2,
                        )
                    })
                }
            }
        };
        assert_eq!(quote!(#implementation).to_string(), quote!(#expected).to_string());
    }

    #[test]
    fn test_expand_projection_errors() {
        let projection = quote!(projection(ArrayIrType => ArrayType));
        assert_eq!(
            error(
                projection.clone(),
                quote! {
                    pub trait Gather<Stored = Array, T = <Self as Capability>::Universe>: Capability {}
                },
            ),
            "projected capabilities only support type parameters that precede the universe parameter and default to \
             `Self`",
        );
        assert_eq!(
            error(
                projection.clone(),
                quote! {
                    pub trait Double<T = <Self as Capability>::Universe>: Capability {
                        type Output;
                    }
                },
            ),
            "projected capabilities only support functions",
        );
        assert_eq!(
            error(
                projection.clone(),
                quote! {
                    pub trait Double<T = <Self as Capability>::Universe>: Capability {
                        fn double(self) -> Result<Self, ProgramError>;
                    }
                },
            ),
            "projected capability functions must take `&self`",
        );
        assert_eq!(
            error(
                projection.clone(),
                quote! {
                    pub trait Double<T = <Self as Capability>::Universe>: Capability {
                        fn double(&self, other: Vec<Self>) -> Result<Self, ProgramError>;
                    }
                },
            ),
            "projected capability inputs that mention `Self` must be `&Self`, `&[Self]`, or `Option<&Self>`",
        );
        assert_eq!(
            error(
                projection.clone(),
                quote! {
                    pub trait Double<T = <Self as Capability>::Universe>: Capability {
                        fn double(&self) -> Self;
                    }
                },
            ),
            "projected capability functions must return a `Result`",
        );
        assert_eq!(
            error(
                projection,
                quote! {
                    pub trait Double<T = <Self as Capability>::Universe>: Capability {
                        fn double(&self) -> Result<Option<Self>, ProgramError>;
                    }
                },
            ),
            "projected capability functions must return `Self`, `Vec<Self>`, tuples of those, or types that do not \
             mention `Self`",
        );
    }
}
