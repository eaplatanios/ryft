//! Expansion of the `#[capability]` attribute, which validates the conventions of capability traits and generates the
//! implementations through which heterogeneous values apply a capability by projecting onto their members.

use std::fmt::Display;

use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote, quote_spanned};
use syn::spanned::Spanned;
use syn::visit_mut::VisitMut;

use crate::helpers::attributes::Attribute;
use crate::helpers::idents::IdentHelpers;
use crate::helpers::symbols::Symbol;
use crate::helpers::tokens::TokensHelpers;

// TODO(eaplatanios): Review this module.

const CRATE_ARGUMENT: Symbol = Symbol::new("crate");
const PROJECTION_ARGUMENT: Symbol = Symbol::new("projection");

const DEFAULT_RYFT_CRATE: Symbol = Symbol::new("ryft");
const CAPABILITY_TRAIT: Symbol = Symbol::new("Capability");
const UNIVERSE_TYPE: Symbol = Symbol::new("Universe");
const SELF_TYPE: Symbol = Symbol::new("Self");
const MACRO_VALUE_TYPE: Symbol = Symbol::new("__V");

/// Code generator for the `#[capability]` attribute, which validates the conventions of the annotated capability trait
/// and generates one projection implementation per `projection(Family => Member)` argument. Refer to the documentation
/// of the `Capability` trait in `ryft-core` for information on how to use this attribute.
pub(crate) struct CapabilityCodeGenerator {
    /// [`syn::Path`] that represents the root `ryft` library path (e.g., `ryft`). This is customizable via the
    /// `crate = "..."` argument and it is meant to support libraries that build on top of `ryft` (or that depend on
    /// `ryft-core` directly) and that may want to export `ryft` items as part of their namespace.
    ryft_crate: syn::Path,

    /// [`Projection`]s requested through `projection(Family => Member)` arguments, in declaration order.
    projections: Vec<Projection>,

    /// Errors accumulated in this [`CapabilityCodeGenerator`]. The way error handling works in this generator is that
    /// we collect errors as we encounter them, and keep going as far as we can with the information that is available,
    /// before raising them. That is meant to enable a smoother development experience when working with `ryft` by
    /// reducing the amount of trial and error required to get something work (i.e., users do not need to keep trying,
    /// fixing one error at a time; they get to see multiple errors at once, when there are multiple).
    errors: Vec<syn::Error>,
}

impl CapabilityCodeGenerator {
    /// Generates the `#[capability]` expansion for the provided attribute arguments and annotated trait. The expansion
    /// always contains the annotated trait itself, followed by one projection implementation per
    /// `projection(Family => Member)` argument. If any errors are encountered, the projection implementations are
    /// replaced by a [`compile_error!`] invocation, while the trait is still emitted, so that its uses elsewhere do not
    /// report additional errors about a missing trait.
    ///
    /// As an example, `#[capability(projection(ArrayIrType => ArrayType))]` applied to a capability
    /// `trait Double<T = <Self as Capability>::Universe>: Capability + Sized` with one required function `double`
    /// generates the following implementation next to the trait:
    ///
    /// ```ignore
    /// impl<__V> Double<ArrayIrType> for __V
    /// where
    ///     __V: ryft::Value<Type = ArrayIrType> + ryft::ValueProjection<ArrayType>,
    ///     <__V as ryft::ValueProjection<ArrayType>>::Projected: Double<ArrayType>,
    ///     __V: Sized,
    /// {
    ///     #[inline]
    ///     fn double(&self) -> Result<Self, ProgramError> {
    ///         let __receiver = ryft::ValueProjection::<ArrayType>::into_projected(::core::clone::Clone::clone(self))?;
    ///         let __output = <<__V as ryft::ValueProjection<ArrayType>>::Projected as Double<ArrayType>>
    ///             ::double(&__receiver)?;
    ///         ::core::result::Result::Ok(<Self as ryft::ValueProjection<ArrayType>>::from_projected(__output))
    ///     }
    /// }
    /// ```
    pub(crate) fn generate_capability_impl(attributes: TokenStream, item: TokenStream) -> TokenStream {
        let item = match syn::parse2::<syn::ItemTrait>(item) {
            Ok(item) => item,
            Err(error) => return error.into_compile_error(),
        };
        let mut generator = Self::new();
        generator.extract_arguments(attributes);
        let universe = generator.extract_universe_parameter(&item);
        let projections = std::mem::take(&mut generator.projections);
        let implementations = universe
            .map(|universe| {
                projections
                    .iter()
                    .map(|projection| generator.generate_projection_impl(&item, universe, projection))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        match generator.compile_error() {
            Some(error) => quote!(#item #error),
            None => quote!(#item #(#implementations)*),
        }
    }

    /// Creates a new [`CapabilityCodeGenerator`] that has not extracted any information yet.
    fn new() -> Self {
        Self { ryft_crate: DEFAULT_RYFT_CRATE.into(), projections: Vec::new(), errors: Vec::new() }
    }

    /// Adds an error to this [`CapabilityCodeGenerator`] with the specified message spanning the provided tokens. This
    /// is only meant to be used internally by this generator as a convenient helper for collecting errors.
    ///
    /// # Parameters
    ///
    ///   - `tokens`: Tokens that the error spans.
    ///   - `message`: Message describing the error.
    fn add_error<T: ToTokens, U: Display>(&mut self, tokens: T, message: U) {
        self.errors.push(syn::Error::new_spanned(tokens.into_token_stream(), message));
    }

    /// Returns a [`TokenStream`] that represents a [`compile_error!`] invocation that contains information about the
    /// [`syn::Error`]s that have been collected by this [`CapabilityCodeGenerator`] so far. If there are no errors,
    /// then this function returns [`None`].
    fn compile_error(&self) -> Option<TokenStream> {
        self.errors
            .iter()
            .cloned()
            .reduce(|mut combined_error, error| {
                combined_error.combine(error);
                combined_error
            })
            .map(|error| error.into_compile_error())
    }

    /// Extracts the `crate = "..."` and `projection(Family => Member)` arguments of the attribute into
    /// [`CapabilityCodeGenerator::ryft_crate`] and [`CapabilityCodeGenerator::projections`]. Each family may be
    /// projected at most once, because a second projection of the same family would implement the capability twice.
    fn extract_arguments(&mut self, attributes: TokenStream) {
        let mut ryft_crate = Attribute::new(CRATE_ARGUMENT);
        let parser = syn::meta::parser(|meta| match &meta.path {
            path if path == &CRATE_ARGUMENT => ryft_crate.set(&meta),
            path if path == &PROJECTION_ARGUMENT => {
                let content;
                syn::parenthesized!(content in meta.input);
                let family = content.parse::<syn::Type>()?;
                content.parse::<syn::Token![=>]>()?;
                let member = content.parse::<syn::Type>()?;
                if !content.is_empty() {
                    return Err(content.error("`projection(...)` declares exactly one `Family => Member` pair"));
                }
                if self.projections.iter().any(|projection| projection.family.tokens_eq(&family)) {
                    return Err(syn::Error::new_spanned(
                        &family,
                        format_args!("duplicate capability projection of family `{}`", family.to_compact_string()),
                    ));
                }
                self.projections.push(Projection { family, member });
                Ok(())
            }
            _ => Err(meta.error(format_args!(
                "invalid `#[capability(...)]` argument `{}`; only `crate` and `projection(...)` are supported here",
                meta.path.to_compact_string(),
            ))),
        });
        syn::parse::Parser::parse2(parser, attributes).unwrap_or_else(|error| self.errors.push(error));
        if let Some(ryft_crate) = ryft_crate.get() {
            self.ryft_crate = ryft_crate;
        }
    }

    /// Validates the capability conventions of the provided trait, which are that exactly one type parameter defaults
    /// to `<Self as Capability>::Universe` (with any path to `Capability`), that this universe parameter is unbounded
    /// (because host types are their own universes, and so a bound such as `T: Type` would exclude them), and that
    /// `Capability` is a direct supertrait. Returns the position of the universe parameter, or [`None`] if the trait
    /// has no unique universe parameter.
    fn extract_universe_parameter(&mut self, item: &syn::ItemTrait) -> Option<usize> {
        let universes = item
            .generics
            .params
            .iter()
            .enumerate()
            .filter_map(|(index, parameter)| match parameter {
                syn::GenericParam::Type(parameter) if parameter.default.as_ref().is_some_and(is_universe_default) => {
                    Some((index, parameter))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        let universe = match universes.as_slice() {
            [] => {
                self.add_error(
                    &item.ident,
                    "capability traits must declare a type parameter that defaults to `<Self as Capability>::Universe`",
                );
                None
            }
            [universe] => Some(*universe),
            [_, duplicates @ ..] => {
                duplicates.iter().for_each(|(_, duplicate)| {
                    self.add_error(duplicate, "capability traits must declare exactly one universe parameter")
                });
                None
            }
        };

        if let Some((_, universe)) = universe {
            let is_bounded_in_where_clause = item.generics.where_clause.iter().any(|where_clause| {
                where_clause.predicates.iter().any(|predicate| {
                    matches!(predicate, syn::WherePredicate::Type(predicate)
                        if predicate.bounded_ty.matches_ident(&universe.ident))
                })
            });
            if !universe.bounds.is_empty() || is_bounded_in_where_clause {
                self.add_error(
                    &universe.ident,
                    format_args!(
                        "the universe parameter `{}` of a capability must not be bounded, because host types are their \
                         own universes",
                        universe.ident,
                    ),
                );
            }
        }

        let has_capability_supertrait = item.supertraits.iter().any(|bound| {
            matches!(bound, syn::TypeParamBound::Trait(bound)
                if bound.path.segments.last().is_some_and(|segment| segment.ident == CAPABILITY_TRAIT))
        });
        if !has_capability_supertrait {
            self.add_error(&item.ident, "capability traits must have `Capability` as a supertrait");
        }
        universe.map(|(index, _)| index)
    }

    /// Generates the implementation of the provided capability for every value of the projection family whose
    /// `ValueProjection` onto the projection member yields a value that implements the capability itself. Any parts of
    /// the capability that cannot be projected are reported as errors.
    ///
    /// The implementation restates the supertraits and the `where` predicates of the capability as bounds on the
    /// implementing value, with its universe parameter read as the projection family, so that it only applies to the
    /// values that satisfy them. Type parameters other than the universe must precede it and default to `Self` (e.g.,
    /// the right input of `Dot`), so that they resolve to the heterogeneous value in the implemented capability and to
    /// its projection in the delegated one. Only the required functions of the capability are implemented, while its
    /// provided functions keep their default bodies.
    ///
    /// # Parameters
    ///
    ///   - `item`: Capability trait to implement.
    ///   - `universe`: Position of the universe parameter of the capability.
    ///   - `projection`: [`Projection`] through which the capability is implemented.
    fn generate_projection_impl(
        &mut self,
        item: &syn::ItemTrait,
        universe: usize,
        projection: &Projection,
    ) -> TokenStream {
        let ryft = self.ryft_crate.clone();
        let Projection { family, member } = projection;
        let value_type: syn::Path = MACRO_VALUE_TYPE.into();
        let projected_type: syn::Type = syn::parse_quote!(<#value_type as #ryft::ValueProjection<#member>>::Projected);

        let mut self_parameters = Vec::new();
        let mut implemented_arguments = Vec::new();
        let mut projected_arguments = Vec::new();
        for (index, parameter) in item.generics.params.iter().enumerate() {
            match parameter {
                _ if index == universe => {
                    implemented_arguments.push(family.to_token_stream());
                    projected_arguments.push(member.to_token_stream());
                }
                syn::GenericParam::Type(parameter)
                    if index < universe && parameter.default.as_ref().is_some_and(|default| default.is_self()) =>
                {
                    self_parameters.push(parameter.ident.clone());
                    implemented_arguments.push(value_type.to_token_stream());
                    projected_arguments.push(projected_type.to_token_stream());
                }
                _ => self.add_error(
                    parameter,
                    "projected capabilities only support type parameters that precede the universe parameter and \
                     default to `Self`",
                ),
            }
        }

        // Every mention of `Self` and of the `Self`-defaulted parameters refers to the implementing value, and every
        // mention of the universe parameter refers to the projection family.
        let universe_parameter = item
            .generics
            .type_params()
            .find(|parameter| parameter.default.as_ref().is_some_and(is_universe_default));
        let restated_predicates = item
            .supertraits
            .iter()
            .filter(|bound| {
                !matches!(bound, syn::TypeParamBound::Trait(bound)
                    if bound.path.segments.last().is_some_and(|segment| segment.ident == CAPABILITY_TRAIT))
            })
            .map(|bound| -> syn::WherePredicate { syn::parse_quote!(Self: #bound) })
            .chain(item.generics.where_clause.iter().flat_map(|where_clause| where_clause.predicates.iter().cloned()))
            .map(|mut predicate| {
                predicate.replace_ident(&SELF_TYPE.into(), &value_type);
                self_parameters.iter().for_each(|parameter| predicate.replace_ident(parameter, &value_type));
                if let Some(universe_parameter) = universe_parameter {
                    TypeParameterReplacer::new(&universe_parameter.ident, family)
                        .visit_where_predicate_mut(&mut predicate);
                }
                predicate
            })
            .collect::<Vec<_>>();

        let trait_name = &item.ident;
        let functions = item
            .items
            .iter()
            .filter_map(|trait_item| match trait_item {
                syn::TraitItem::Fn(function) if function.default.is_some() => None,
                syn::TraitItem::Fn(function) => {
                    match self.generate_projected_function(
                        function,
                        &self_parameters,
                        &projected_arguments,
                        item,
                        member,
                    ) {
                        Ok(function) => Some(function),
                        Err(error) => {
                            self.errors.push(error);
                            None
                        }
                    }
                }
                other => {
                    self.add_error(other, "projected capabilities only support functions");
                    None
                }
            })
            .collect::<Vec<_>>();

        // The implementation is spanned at the projection argument (and its functions at their trait functions), so
        // that conflicting implementations and unsatisfied bounds point at the projection that introduced them.
        let span = family.span();
        quote_spanned! { span =>
            impl<#value_type> #trait_name<#(#implemented_arguments),*> for #value_type
            where
                #value_type: #ryft::Value<Type = #family> + #ryft::ValueProjection<#member>,
                #projected_type: #trait_name<#(#projected_arguments),*>,
                #(#restated_predicates,)*
            {
                #(#functions)*
            }
        }
    }

    /// Generates one function of a projection implementation. The function projects the receiver and every `&Self`,
    /// `&[Self]`, and `Option<&Self>` input onto the member universe, applies the member implementation, and lifts its
    /// outputs back through [`lift`], while passing all other inputs through unchanged. The generated function keeps
    /// the `#[cfg]` and `#[cfg_attr]` attributes of the trait function, so that both are compiled under the same
    /// configuration, and its local variables are named so that they cannot shadow the inputs of the function.
    ///
    /// # Parameters
    ///
    ///   - `function`: Required function of the capability to implement.
    ///   - `self_parameters`: Type parameters of the capability that default to `Self`.
    ///   - `projected_arguments`: Generic arguments of the capability as implemented by the projected value.
    ///   - `item`: Capability trait that declares `function`.
    ///   - `member`: Member universe onto which values are projected.
    fn generate_projected_function(
        &self,
        function: &syn::TraitItemFn,
        self_parameters: &[syn::Ident],
        projected_arguments: &[TokenStream],
        item: &syn::ItemTrait,
        member: &syn::Type,
    ) -> syn::Result<TokenStream> {
        let ryft = &self.ryft_crate;
        let self_type: syn::Type = syn::parse_quote!(Self);
        let mut signature = function.sig.clone();
        for parameter in self_parameters {
            TypeParameterReplacer::new(parameter, &self_type).visit_signature_mut(&mut signature);
        }
        let projection = quote!(#ryft::ValueProjection::<#member>::into_projected);

        let mut inputs = signature.inputs.iter();
        match inputs.next() {
            Some(syn::FnArg::Receiver(receiver)) if receiver.reference.is_some() && receiver.mutability.is_none() => {}
            _ => return Err(syn::Error::new_spanned(&signature, "projected capability functions must take `&self`")),
        }

        let mut bindings = Vec::new();
        let mut arguments = Vec::new();
        for input in inputs {
            let syn::FnArg::Typed(input) = input else {
                return Err(syn::Error::new_spanned(input, "projected capability functions must take `&self`"));
            };
            let syn::Pat::Ident(name) = input.pat.as_ref() else {
                return Err(syn::Error::new_spanned(
                    &input.pat,
                    "projected capability inputs must be named by identifiers",
                ));
            };
            let name = &name.ident;
            match input.ty.as_ref() {
                syn::Type::Reference(reference) if reference.mutability.is_none() && reference.elem.is_self() => {
                    bindings.push(quote!(let #name = #projection(::core::clone::Clone::clone(#name))?;));
                    arguments.push(quote!(&#name));
                }
                syn::Type::Reference(reference)
                    if reference.mutability.is_none()
                        && matches!(reference.elem.as_ref(), syn::Type::Slice(slice) if slice.elem.is_self()) =>
                {
                    bindings.push(quote! {
                        let #name = #name
                            .iter()
                            .map(|value| #projection(::core::clone::Clone::clone(value)))
                            .collect::<::core::result::Result<::std::vec::Vec<_>, _>>()?;
                    });
                    arguments.push(quote!(#name.as_slice()));
                }
                r#type
                    if single_type_argument(r#type, "Option").is_some_and(|argument| {
                        matches!(argument, syn::Type::Reference(reference)
                            if reference.mutability.is_none() && reference.elem.is_self())
                    }) =>
                {
                    bindings.push(quote! {
                        let #name = #name.map(|value| #projection(::core::clone::Clone::clone(value))).transpose()?;
                    });
                    arguments.push(quote!(#name.as_ref()));
                }
                r#type if !r#type.mentions_self() => arguments.push(quote!(#name)),
                r#type => {
                    return Err(syn::Error::new_spanned(
                        r#type,
                        "projected capability inputs that mention `Self` must be `&Self`, `&[Self]`, or \
                         `Option<&Self>`",
                    ));
                }
            }
        }

        let syn::ReturnType::Type(_, output) = &signature.output else {
            return Err(syn::Error::new_spanned(&signature, "projected capability functions must return a `Result`"));
        };
        let Some(output) = result_value_type(output) else {
            return Err(syn::Error::new_spanned(output, "projected capability functions must return a `Result`"));
        };
        let lifted = lift(output, quote!(__output), member, ryft)?;
        let value_type: syn::Path = MACRO_VALUE_TYPE.into();
        let attributes = function
            .attrs
            .iter()
            .filter(|attribute| attribute.path().is_ident("cfg") || attribute.path().is_ident("cfg_attr"));
        let trait_name = &item.ident;
        let function_name = &signature.ident;
        Ok(quote_spanned! { function.sig.span() =>
            #(#attributes)*
            #[inline]
            #signature {
                let __receiver = #projection(::core::clone::Clone::clone(self))?;
                #(#bindings)*
                let __output = <
                    <#value_type as #ryft::ValueProjection<#member>>::Projected
                    as #trait_name<#(#projected_arguments),*>
                >::#function_name(&__receiver, #(#arguments),*)?;
                ::core::result::Result::Ok(#lifted)
            }
        })
    }
}

/// Projection requested by a `projection(Family => Member)` argument of the `#[capability]` attribute.
struct Projection {
    /// Universe of the heterogeneous values that apply the capability by projecting onto their members.
    family: syn::Type,

    /// Universe of the member values onto which the heterogeneous values project.
    member: syn::Type,
}

/// [`VisitMut`] that replaces every bare mention of a type parameter with a replacement type.
struct TypeParameterReplacer<'r> {
    /// Type parameter whose mentions are replaced.
    parameter: &'r syn::Ident,

    /// Type that replaces every mention of [`TypeParameterReplacer::parameter`].
    replacement: &'r syn::Type,
}

impl<'r> TypeParameterReplacer<'r> {
    /// Creates a new [`TypeParameterReplacer`] that replaces `parameter` with `replacement`.
    fn new(parameter: &'r syn::Ident, replacement: &'r syn::Type) -> Self {
        Self { parameter, replacement }
    }
}

impl VisitMut for TypeParameterReplacer<'_> {
    fn visit_type_mut(&mut self, r#type: &mut syn::Type) {
        if r#type.matches_ident(self.parameter) {
            *r#type = self.replacement.clone();
            return;
        }
        syn::visit_mut::visit_type_mut(self, r#type);
    }
}

/// Helper trait for checking whether [`syn::Type`]s refer to `Self`.
trait SelfTypeHelpers {
    /// Returns `true` if this type is exactly `Self`.
    fn is_self(&self) -> bool;

    /// Returns `true` if this type mentions `Self` anywhere (e.g., `Vec<Self>` or `Self::Item`).
    fn mentions_self(&self) -> bool;
}

impl SelfTypeHelpers for syn::Type {
    fn is_self(&self) -> bool {
        self.matches_ident(&SELF_TYPE.into())
    }

    fn mentions_self(&self) -> bool {
        self.references_ident(&SELF_TYPE.into())
    }
}

/// Returns `true` if the provided type is `<Self as Capability>::Universe`, with any path to `Capability`.
fn is_universe_default(r#type: &syn::Type) -> bool {
    let syn::Type::Path(syn::TypePath { qself: Some(qualified_self), path }) = r#type else {
        return false;
    };
    let segments = &path.segments;
    qualified_self.ty.is_self()
        && qualified_self.position >= 1
        && qualified_self.position + 1 == segments.len()
        && segments[qualified_self.position - 1].ident == CAPABILITY_TRAIT
        && segments[qualified_self.position].ident == UNIVERSE_TYPE
}

/// Returns the single type argument `A` of the provided type if it is a path whose last segment is `name<A>` (e.g.,
/// `Option<A>`, `Vec<A>`, or `std::vec::Vec<A>`), and [`None`] otherwise.
fn single_type_argument<'t>(r#type: &'t syn::Type, name: &str) -> Option<&'t syn::Type> {
    let syn::Type::Path(syn::TypePath { qself: None, path }) = r#type else {
        return None;
    };
    let segment = path.segments.last()?;
    let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return None;
    };
    match arguments.args.iter().collect::<Vec<_>>().as_slice() {
        [syn::GenericArgument::Type(argument)] if segment.ident == name => Some(argument),
        _ => None,
    }
}

/// Returns the success type `R` of the provided type if it is a path whose last segment is `Result<R, E>`, and [`None`]
/// otherwise.
fn result_value_type(r#type: &syn::Type) -> Option<&syn::Type> {
    let syn::Type::Path(syn::TypePath { qself: None, path }) = r#type else {
        return None;
    };
    let segment = path.segments.last()?;
    let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return None;
    };
    match arguments.args.iter().collect::<Vec<_>>().as_slice() {
        [syn::GenericArgument::Type(value), syn::GenericArgument::Type(_)] if segment.ident == "Result" => Some(value),
        _ => None,
    }
}

/// Lifts the projected value of the provided expression, whose type is `type` with `Self` read as the projected
/// member, back into the heterogeneous universe. `Self` is lifted through `ValueProjection::from_projected`,
/// `Vec<Self>` is lifted element by element, tuples are lifted component by component, and types that do not mention
/// `Self` are passed through unchanged. Returns an error for all other types that mention `Self`.
///
/// # Parameters
///
///   - `type`: Type of the value to lift, as declared by the capability function.
///   - `expression`: Expression that evaluates to the projected value.
///   - `member`: Member universe of the projected value.
///   - `ryft`: Root `ryft` library path used by the generated code.
fn lift(r#type: &syn::Type, expression: TokenStream, member: &syn::Type, ryft: &syn::Path) -> syn::Result<TokenStream> {
    let from_projected = quote!(<Self as #ryft::ValueProjection<#member>>::from_projected);
    if r#type.is_self() {
        return Ok(quote!(#from_projected(#expression)));
    }
    if !r#type.mentions_self() {
        return Ok(expression);
    }
    match r#type {
        r#type if single_type_argument(r#type, "Vec").is_some_and(|element| element.is_self()) => {
            Ok(quote!(#expression.into_iter().map(#from_projected).collect()))
        }
        syn::Type::Tuple(tuple) => {
            let names = (0..tuple.elems.len()).map(|index| format_ident!("__output_{index}")).collect::<Vec<_>>();
            let components = tuple
                .elems
                .iter()
                .zip(&names)
                .map(|(element, name)| lift(element, quote!(#name), member, ryft))
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
    use quote::quote;

    use super::*;

    /// Returns the rendering of the provided tokens without whitespace and without trailing commas, so that generated
    /// code can be compared against expected code that is written in the usual `rustfmt` style.
    fn rendering<T: ToTokens>(tokens: &T) -> String {
        let rendering =
            [",>", ",)", ",]", ",{", ",}"].iter().fold(tokens.to_compact_string(), |rendering, trailing_comma| {
                rendering.replace(trailing_comma, &trailing_comma[1..])
            });
        rendering.strip_suffix(',').map(str::to_string).unwrap_or(rendering)
    }

    /// Returns the messages of the errors that the provided [`CapabilityCodeGenerator`] has accumulated.
    fn errors(generator: &CapabilityCodeGenerator) -> Vec<String> {
        generator.errors.iter().map(syn::Error::to_string).collect()
    }

    /// Returns a [`CapabilityCodeGenerator`] that has extracted the provided attribute arguments.
    fn extract_arguments(attributes: TokenStream) -> CapabilityCodeGenerator {
        let mut generator = CapabilityCodeGenerator::new();
        generator.extract_arguments(attributes);
        assert_eq!(errors(&generator), Vec::<String>::new());
        generator
    }

    #[test]
    fn test_capability_code_generator_generate_capability_impl() {
        // Test expanding a capability without projections, which expands to the capability itself.
        let item = quote! {
            pub trait Double<T = <Self as Capability>::Universe>: Capability + Sized {
                fn double(&self) -> Result<Self, ProgramError>;
            }
        };
        assert_eq!(
            rendering(&CapabilityCodeGenerator::generate_capability_impl(quote!(), item.clone())),
            rendering(&item),
        );

        // Test expanding a capability with a projection, which appends the projection implementation.
        let expanded = CapabilityCodeGenerator::generate_capability_impl(
            quote!(projection(ArrayIrType => ArrayType)),
            item.clone(),
        );
        let expanded: syn::File = syn::parse2(expanded).unwrap();
        assert_eq!(expanded.items.len(), 2);
        assert_eq!(rendering(&expanded.items[0]), rendering(&item));
        assert!(matches!(&expanded.items[1], syn::Item::Impl(_)));

        // Test expanding an invalid capability, which keeps the capability next to the reported errors so that its
        // uses do not report additional errors about a missing trait.
        let item = quote! {
            pub trait Double: Sized {}
        };
        let expanded = CapabilityCodeGenerator::generate_capability_impl(quote!(projection(A => B)), item.clone());
        assert_eq!(
            rendering(&expanded),
            rendering(&quote! {
                #item
                ::core::compile_error! {
                    "capability traits must declare a type parameter that defaults to `<Self as Capability>::Universe`"
                }
                ::core::compile_error! { "capability traits must have `Capability` as a supertrait" }
            }),
        );

        // Test expanding an item that is not a trait.
        assert_eq!(
            CapabilityCodeGenerator::generate_capability_impl(
                quote!(),
                quote!(
                    struct Double;
                )
            )
            .to_string(),
            quote!(::core::compile_error! { "expected `trait`" }).to_string(),
        );
    }

    #[test]
    fn test_capability_code_generator_add_error_and_compile_error() {
        let mut generator = CapabilityCodeGenerator::new();
        assert!(generator.compile_error().is_none());
        generator.add_error(quote!(first), "first error");
        generator.add_error(quote!(second), "second error");
        assert_eq!(
            generator.compile_error().unwrap().to_string(),
            quote!(::core::compile_error! { "first error" } ::core::compile_error! { "second error" }).to_string(),
        );
    }

    #[test]
    fn test_capability_code_generator_extract_arguments() {
        let generator = extract_arguments(quote!());
        assert_eq!(generator.ryft_crate.to_compact_string(), "ryft");
        assert!(generator.projections.is_empty());

        let generator = extract_arguments(quote! {
            projection(ArrayIrType => ArrayType),
            crate = "::core_alias",
            projection(other::FamilyType<A> => other::MemberType),
        });
        assert_eq!(generator.ryft_crate.to_compact_string(), "::core_alias");
        assert_eq!(
            generator
                .projections
                .iter()
                .map(|projection| (projection.family.to_compact_string(), projection.member.to_compact_string()))
                .collect::<Vec<_>>(),
            vec![
                ("ArrayIrType".to_string(), "ArrayType".to_string()),
                ("other::FamilyType<A>".to_string(), "other::MemberType".to_string()),
            ],
        );

        // Test using invalid arguments.
        for (attributes, error) in [
            (
                quote!(projections(A => B)),
                "invalid `#[capability(...)]` argument `projections`; only `crate` and `projection(...)` are supported \
                 here",
            ),
            (quote!(crate = "ryft", crate = "ryft"), "duplicate ryft attribute `crate`"),
            (quote!(projection(A => B), projection(A => C)), "duplicate capability projection of family `A`"),
            (quote!(projection), "unexpected end of input, expected parentheses"),
            (quote!(projection(A => B, C)), "`projection(...)` declares exactly one `Family => Member` pair"),
        ] {
            let mut generator = CapabilityCodeGenerator::new();
            generator.extract_arguments(attributes);
            assert_eq!(errors(&generator), vec![error]);
        }
    }

    #[test]
    fn test_capability_code_generator_extract_universe_parameter() {
        let mut generator = CapabilityCodeGenerator::new();
        let item: syn::ItemTrait = syn::parse_quote! {
            pub trait Combine<Rhs = Self, T = <Self as Capability>::Universe>: Capability + Sized {}
        };
        assert_eq!(generator.extract_universe_parameter(&item), Some(1));

        // Universe parameters are recognized through any path to `Capability`, as in macro-generated capabilities.
        let item: syn::ItemTrait = syn::parse_quote! {
            pub trait Double<T = <Self as crate::operations::Capability>::Universe>:
                crate::operations::Capability + Clone {}
        };
        assert_eq!(generator.extract_universe_parameter(&item), Some(0));
        assert_eq!(errors(&generator), Vec::<String>::new());

        // Test using capabilities that violate the conventions.
        for (item, error) in [
            (
                quote!(
                    pub trait Double: Capability {}
                ),
                "capability traits must declare a type parameter that defaults to `<Self as Capability>::Universe`",
            ),
            (
                quote!(
                    pub trait Double<T = <Self>::Universe>: Capability {}
                ),
                "capability traits must declare a type parameter that defaults to `<Self as Capability>::Universe`",
            ),
            (
                quote! {
                    pub trait Double<T = <Self as Capability>::Universe, U = <Self as Capability>::Universe>:
                        Capability {}
                },
                "capability traits must declare exactly one universe parameter",
            ),
            (
                quote!(
                    pub trait Double<T: Type = <Self as Capability>::Universe>: Capability {}
                ),
                "the universe parameter `T` of a capability must not be bounded, because host types are their own \
                 universes",
            ),
            (
                quote!(
                    pub trait Double<T = <Self as Capability>::Universe>: Capability
                    where
                        T: Type,
                    {
                    }
                ),
                "the universe parameter `T` of a capability must not be bounded, because host types are their own \
                 universes",
            ),
            (
                quote!(
                    pub trait Double<T = <Self as Capability>::Universe>: Sized {}
                ),
                "capability traits must have `Capability` as a supertrait",
            ),
        ] {
            let mut generator = CapabilityCodeGenerator::new();
            generator.extract_universe_parameter(&syn::parse2(item).unwrap());
            assert_eq!(errors(&generator), vec![error]);
        }
    }

    #[test]
    fn test_capability_code_generator_generate_projection_impl() {
        // The implementation restates the supertraits and the `where` predicates of the capability, and it only
        // implements its required functions.
        let mut generator = extract_arguments(quote!(crate = "core_alias", projection(ArrayIrType => ArrayType)));
        let item: syn::ItemTrait = syn::parse_quote! {
            pub trait Combine<Rhs = Self, T = <Self as Capability>::Universe>: Capability + Sized + Clone
            where
                Self: Debug + PartialEq<Rhs>,
                Vec<T>: Default,
            {
                fn combine(&self, right: &Rhs) -> Result<Self, ProgramError>;

                fn combine_twice(&self, right: &Rhs) -> Result<Self, ProgramError> {
                    self.combine(right)?.combine(right)
                }
            }
        };
        let projection = generator.projections.remove(0);
        let generated = generator.generate_projection_impl(&item, 1, &projection);
        assert_eq!(errors(&generator), Vec::<String>::new());
        assert_eq!(
            rendering(&generated),
            rendering(&quote! {
                impl<__V> Combine<__V, ArrayIrType> for __V
                where
                    __V: core_alias::Value<Type = ArrayIrType> + core_alias::ValueProjection<ArrayType>,
                    <__V as core_alias::ValueProjection<ArrayType>>::Projected: Combine<
                        <__V as core_alias::ValueProjection<ArrayType>>::Projected,
                        ArrayType,
                    >,
                    __V: Sized,
                    __V: Clone,
                    __V: Debug + PartialEq<__V>,
                    Vec<ArrayIrType>: Default,
                {
                    #[inline]
                    fn combine(&self, right: &Self) -> Result<Self, ProgramError> {
                        let __receiver = core_alias::ValueProjection::<ArrayType>::into_projected(
                            ::core::clone::Clone::clone(self),
                        )?;
                        let right = core_alias::ValueProjection::<ArrayType>::into_projected(
                            ::core::clone::Clone::clone(right),
                        )?;
                        let __output = <
                            <__V as core_alias::ValueProjection<ArrayType>>::Projected
                            as Combine<<__V as core_alias::ValueProjection<ArrayType>>::Projected, ArrayType>
                        >::combine(&__receiver, &right)?;
                        ::core::result::Result::Ok(
                            <Self as core_alias::ValueProjection<ArrayType>>::from_projected(__output),
                        )
                    }
                }
            }),
        );

        // Test projecting capabilities that cannot be projected.
        for (item, error) in [
            (
                quote!(
                    pub trait Gather<Stored = Array, T = <Self as Capability>::Universe>: Capability {}
                ),
                "projected capabilities only support type parameters that precede the universe parameter and default \
                 to `Self`",
            ),
            (
                quote!(
                    pub trait Double<T = <Self as Capability>::Universe>: Capability {
                        type Output;
                    }
                ),
                "projected capabilities only support functions",
            ),
            (
                quote! {
                    pub trait Double<T = <Self as Capability>::Universe>: Capability {
                        fn double(self) -> Result<Self, ProgramError>;
                    }
                },
                "projected capability functions must take `&self`",
            ),
        ] {
            let mut generator = extract_arguments(quote!(projection(ArrayIrType => ArrayType)));
            let item: syn::ItemTrait = syn::parse2(item).unwrap();
            let universe = generator.extract_universe_parameter(&item).unwrap();
            let projection = generator.projections.remove(0);
            generator.generate_projection_impl(&item, universe, &projection);
            assert_eq!(errors(&generator), vec![error]);
        }
    }

    #[test]
    fn test_capability_code_generator_generate_projected_function() {
        // Every kind of supported input is projected, the `#[cfg]` attributes of the function are kept, and the local
        // variables of the generated function do not shadow its inputs (e.g., `receiver` and `output`).
        let generator = extract_arguments(quote!());
        let item: syn::ItemTrait = syn::parse_quote! {
            pub trait Combine<T = <Self as Capability>::Universe>: Capability + Sized {
                /// Combines values.
                #[cfg(feature = "combine")]
                fn combine<A: Into<Axis>>(
                    &self,
                    receiver: &Self,
                    others: &[Self],
                    mask: Option<&Self>,
                    output: A,
                ) -> Result<(Self, Vec<Self>, usize), ProgramError>;
            }
        };
        let syn::TraitItem::Fn(function) = &item.items[0] else { unreachable!() };
        let member: syn::Type = syn::parse_quote!(ArrayType);
        let generated =
            generator.generate_projected_function(function, &[], &[quote!(ArrayType)], &item, &member).unwrap();
        assert_eq!(
            rendering(&generated),
            rendering(&quote! {
                #[cfg(feature = "combine")]
                #[inline]
                fn combine<A: Into<Axis>>(
                    &self,
                    receiver: &Self,
                    others: &[Self],
                    mask: Option<&Self>,
                    output: A,
                ) -> Result<(Self, Vec<Self>, usize), ProgramError> {
                    let __receiver =
                        ryft::ValueProjection::<ArrayType>::into_projected(::core::clone::Clone::clone(self))?;
                    let receiver =
                        ryft::ValueProjection::<ArrayType>::into_projected(::core::clone::Clone::clone(receiver))?;
                    let others = others
                        .iter()
                        .map(|value| ryft::ValueProjection::<ArrayType>::into_projected(
                            ::core::clone::Clone::clone(value),
                        ))
                        .collect::<::core::result::Result<::std::vec::Vec<_>, _>>()?;
                    let mask = mask
                        .map(|value| ryft::ValueProjection::<ArrayType>::into_projected(
                            ::core::clone::Clone::clone(value),
                        ))
                        .transpose()?;
                    let __output = <
                        <__V as ryft::ValueProjection<ArrayType>>::Projected as Combine<ArrayType>
                    >::combine(
                        &__receiver,
                        &receiver,
                        others.as_slice(),
                        mask.as_ref(),
                        output,
                    )?;
                    ::core::result::Result::Ok({
                        let (__output_0, __output_1, __output_2,) = __output;
                        (
                            <Self as ryft::ValueProjection<ArrayType>>::from_projected(__output_0),
                            __output_1
                                .into_iter()
                                .map(<Self as ryft::ValueProjection<ArrayType>>::from_projected)
                                .collect(),
                            __output_2,
                        )
                    })
                }
            }),
        );

        // Test projecting functions that cannot be projected.
        for (function, error) in [
            (
                quote!(
                    fn double(&mut self) -> Result<Self, ProgramError>;
                ),
                "projected capability functions must take `&self`",
            ),
            (
                quote!(
                    fn double(&self, (left, right): (usize, usize)) -> Result<Self, ProgramError>;
                ),
                "projected capability inputs must be named by identifiers",
            ),
            (
                quote!(
                    fn double(&self, others: Vec<Self>) -> Result<Self, ProgramError>;
                ),
                "projected capability inputs that mention `Self` must be `&Self`, `&[Self]`, or `Option<&Self>`",
            ),
            (
                quote!(
                    fn double(&self) -> Self;
                ),
                "projected capability functions must return a `Result`",
            ),
            (
                quote!(
                    fn double(&self);
                ),
                "projected capability functions must return a `Result`",
            ),
        ] {
            let function: syn::TraitItemFn = syn::parse2(function).unwrap();
            let result = generator.generate_projected_function(&function, &[], &[], &item, &member);
            assert_eq!(result.unwrap_err().to_string(), error);
        }
    }

    #[test]
    fn test_type_parameter_replacer() {
        let parameter: syn::Ident = syn::parse_quote!(Rhs);
        let replacement: syn::Type = syn::parse_quote!(Self);
        let mut signature: syn::Signature = syn::parse_quote!(fn f(&self, right: &Rhs, rights: Vec<Rhs>) -> Rhs);
        TypeParameterReplacer::new(&parameter, &replacement).visit_signature_mut(&mut signature);
        assert_eq!(rendering(&signature), rendering(&quote!(fn f(&self, right: &Self, rights: Vec<Self>) -> Self)),);
    }

    #[test]
    fn test_self_type_helpers_is_self() {
        assert!(syn::Type::is_self(&syn::parse_quote!(Self)));
        assert!(!syn::Type::is_self(&syn::parse_quote!(Self::Item)));
        assert!(!syn::Type::is_self(&syn::parse_quote!(Vec<Self>)));
    }

    #[test]
    fn test_self_type_helpers_mentions_self() {
        assert!(syn::Type::mentions_self(&syn::parse_quote!(Self)));
        assert!(syn::Type::mentions_self(&syn::parse_quote!(Self::Item)));
        assert!(syn::Type::mentions_self(&syn::parse_quote!((usize, Vec<Self>))));
        assert!(!syn::Type::mentions_self(&syn::parse_quote!(Vec<usize>)));
    }

    #[test]
    fn test_is_universe_default() {
        assert!(is_universe_default(&syn::parse_quote!(<Self as Capability>::Universe)));
        assert!(is_universe_default(&syn::parse_quote!(<Self as crate::Capability>::Universe)));
        assert!(!is_universe_default(&syn::parse_quote!(<Self>::Universe)));
        assert!(!is_universe_default(&syn::parse_quote!(<Self as Capability>::Type)));
        assert!(!is_universe_default(&syn::parse_quote!(<Other as Capability>::Universe)));
        assert!(!is_universe_default(&syn::parse_quote!(Universe)));
    }

    #[test]
    fn test_single_type_argument() {
        let r#type: syn::Type = syn::parse_quote!(std::vec::Vec<Self>);
        assert_eq!(
            single_type_argument(&r#type, "Vec").map(TokensHelpers::to_compact_string),
            Some("Self".to_string())
        );
        assert!(single_type_argument(&r#type, "Option").is_none());
        assert!(single_type_argument(&syn::parse_quote!(HashMap<Self, Self>), "HashMap").is_none());
    }

    #[test]
    fn test_result_value_type() {
        let r#type: syn::Type = syn::parse_quote!(std::result::Result<Vec<Self>, ProgramError>);
        assert_eq!(result_value_type(&r#type).map(TokensHelpers::to_compact_string), Some("Vec<Self>".to_string()));
        assert!(result_value_type(&syn::parse_quote!(Result<Self>)).is_none());
        assert!(result_value_type(&syn::parse_quote!(Option<Self>)).is_none());
    }

    #[test]
    fn test_lift() {
        let member: syn::Type = syn::parse_quote!(ArrayType);
        let ryft: syn::Path = syn::parse_quote!(ryft);
        let lift = |r#type: syn::Type| {
            lift(&r#type, quote!(value), &member, &ryft)
                .map(|lifted| rendering(&lifted))
                .map_err(|error| error.to_string())
        };
        assert_eq!(
            lift(syn::parse_quote!(Self)),
            Ok(rendering(&quote!(<Self as ryft::ValueProjection<ArrayType>>::from_projected(value)))),
        );
        assert_eq!(
            lift(syn::parse_quote!(Vec<Self>)),
            Ok(rendering(&quote!(
                value.into_iter().map(<Self as ryft::ValueProjection<ArrayType>>::from_projected).collect()
            ))),
        );
        assert_eq!(lift(syn::parse_quote!(usize)), Ok(rendering(&quote!(value))));
        assert_eq!(
            lift(syn::parse_quote!((Self, usize))),
            Ok(rendering(&quote!({
                let (__output_0, __output_1) = value;
                (<Self as ryft::ValueProjection<ArrayType>>::from_projected(__output_0), __output_1)
            }))),
        );
        assert_eq!(
            lift(syn::parse_quote!(Option<Self>)),
            Err("projected capability functions must return `Self`, `Vec<Self>`, tuples of those, or types that do \
                 not mention `Self`"
                .to_string(),),
        );
    }
}
