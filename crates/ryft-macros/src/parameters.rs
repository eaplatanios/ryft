use std::collections::HashSet;
use std::fmt::Display;

use proc_macro2::{Span, TokenStream};
use quote::{ToTokens, format_ident, quote};

use crate::helpers::attribute::Attribute;
use crate::helpers::generics::GenericsHelpers;
use crate::helpers::hygiene::const_block;
use crate::helpers::ident::IdentHelpers;
use crate::helpers::path::PathHelpers;
use crate::helpers::receiver::replace_self_type;
use crate::helpers::symbol::Symbol;

const RYFT_ATTRIBUTE: Symbol = Symbol::new("ryft");
const CRATE_ATTRIBUTE: Symbol = Symbol::new("crate");
const VALID_CONTAINER_ATTRIBUTES: [Symbol; 1] = [CRATE_ATTRIBUTE];

const DEFAULT_RYFT_CRATE: Symbol = Symbol::new("ryft");
const DEFAULT_MACRO_PARAMETER_LIFETIME: Symbol = Symbol::new("'__p");
const DEFAULT_MACRO_PARAMETER_TYPE: Symbol = Symbol::new("__P");
const DEFAULT_PARAMETER_TYPE: Symbol = Symbol::new("Parameter");

const FIELD_ATTRIBUTE_ERROR: &str = "\
  The '#[ryft(...)]' attribute is only supported at the top level \
  for structs and enums. It is not supported for fields or variants.";

/// [`CodeGenerator`] are used to generate implementations of the [`Parameterized`] trait via the
/// `#[derive(Parameterized)]` macro. Refer to the documentation of the [`Parameterized`] trait for information on how
/// to use that macro.
pub(crate) struct CodeGenerator {
    /// [`syn::Path`] that represents the root `ryft` library path (e.g., `ryft`). This is customizable via the
    /// `#[ryft(crate = "...")]` attribute and it is meant to support libraries that build on top of `ryft` and may
    /// want to export `ryft` types as part of their namespace. This is similar to the `#[serde(crate = "...")]`
    /// attribute. You can refer to [its documentation](https://serde.rs/container-attrs.html#crate) for more
    /// information.
    ryft_crate: syn::Path,

    /// [`syn::Lifetime`] that represents the macro-internal lifetime that is used when we need to introduce one.
    /// This should not conflict with any identifiers that already appear in the scope in which the corresponding
    /// code will be generated. It defaults to [`DEFAULT_MACRO_PARAMETER_LIFETIME`].
    macro_parameter_lifetime: syn::Lifetime,

    /// [`syn::Ident`] that represents the macro-internal type parameter that is used to represent the parameter type
    /// when we need to introduce a new one (e.g., for an associated type that takes a parameter type as one of its
    /// arguments). This should not conflict with any identifiers that already appear in the scope in which the
    /// corresponding code will be generated. It defaults to [`DEFAULT_MACRO_PARAMETER_TYPE`].
    macro_parameter_type: syn::Ident,

    /// [`syn::Ident`] that represents the parameter type in the container on which our macros operate. This must match
    /// one of the generic type parameters of that container. This [`syn::Ident`] is always inferred from the
    /// [`syn::Generics`] of the provided [`syn::DeriveInput`]. It is set to the generic type parameter that is bounded
    /// by [`ryft::Parameter`]. There must be exactly one such type parameter and an error will be generated if there
    /// are zero or more than one such generic type parameters.
    parameter_type: syn::Ident,

    /// [`syn::Generics`] that are used for code generation. These [`syn::Generics`] are extracted from the provided
    /// [`syn::DeriveInput`] and post-processed, adding the necessary bounds for the [`Parameterized`] implementation.
    /// Specifically, they include [`Clone`] bounds for non-parameter [`Field`] types in the underlying [`Data`]
    /// (that is because these bounds are required by the [`Parameterized::parameter_structure`] implementation) and
    /// [`Parameterized`] bounds for all parameter [`Field`] types. Note that for the latter, the actual bounds that
    /// it adds look as follows (abusing notation a bit and representing the field type as `FieldType<P>` where `P`
    /// is the parameter type since at this point we know that these types are generic with respect to `P`):
    /// `ParameterStructure = FieldType<ryft::Placeholder>`.
    generics: syn::Generics,

    /// [`Data`] that is used for the code generation and which is extracted from the provided [`syn::DeriveInput`].
    /// This [`Data`] instance represents the type for which this [`CodeGenerator`] will generate a [`Parameterized`]
    /// trait implementation for.
    data: Data,

    /// Errors accumulated in this [`CodeGenerator`]. The way error handling works in this code generator is that we
    /// collect errors as we encounter them, and keep going as far as we can with the information that is available,
    /// before raising them. That is meant to enable a smoother development experience when working with `ryft` by
    /// reducing the amount of trial and error required to get something work (i.e., users do not need to keep trying,
    /// fixing one error at a time; they get to see multiple errors at once, when there are multiple).
    errors: Vec<syn::Error>,
}

impl CodeGenerator {
    /// Generates an implementation of [`Parameter`] for the provided input. Note that [`Parameter`] is an empty trait
    /// and so the [`Parameter`] derive macro is mainly used for convenience and consistency in the recommended way to
    /// work with ryft traits.
    pub(crate) fn generate_parameter_impl(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
        let input = syn::parse_macro_input!(input as syn::DeriveInput);
        let ident = &input.ident;
        let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
        quote!(impl #impl_generics Parameter for #ident #ty_generics #where_clause {}).into()
    }

    /// Generates an implementation of [`Parameterized`] for the provided input. This function also checks for any
    /// errors on the provided input and will generate a [`TokenStream`] that contains a call to [`compile_error!`]
    /// if any errors are encountered. Refer to the documentation of the [`Parameterized`] trait for information on
    /// how to use this macro.
    ///
    /// As an example, the generated code typically looks something like this:
    ///
    /// ```ignore
    /// #[doc(hidden)]
    /// #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    /// const _: () = {
    ///     #[automatically_derived]
    ///     impl<__P: ryft::Parameter> ryft::ParameterizedFamily<__P> for CustomType<ryft::Placeholder> {
    ///         type To = CustomType<__P>;
    ///     }
    ///
    ///     #[automatically_derived]
    ///     impl<P: Parameter> ryft::Parameterized<P> for CustomType<P> {
    ///         type Family = CustomType<ryft::Placeholder>;
    ///         
    ///         type To<__P: ryft::Parameter> =
    ///             <Self::Family as ryft::ParameterizedFamily<__P>>::To
    ///         where
    ///             Self::Family: ryft::ParameterizedFamily<__P>;
    ///         
    ///         type ParameterStructure = Self::To<ryft::Placeholder>;
    ///
    ///         type ParameterIterator<'__p, __P: '__p + ryft::Parameter> = ... where Self: '__p;
    ///         type ParameterIteratorMut<'__p, __P: '__p + ryft::Parameter> = ... where Self: '__p;
    ///         type ParameterIntoIterator<__P: ryft::Parameter> = ...;
    ///
    ///         fn parameter_count(&self) -> usize { ... }
    ///         fn parameter_structure(&self) -> Self::To<ryft::Placeholder> { ... }
    ///
    ///         fn parameters(&self) -> Self::ParameterIterator<'_, P> { ... }
    ///         fn parameters_mut(&mut self) -> Self::ParameterIteratorMut<'_, P> { ... }
    ///         fn into_parameters(self) -> Self::ParameterIntoIterator<P> { ... }
    ///
    ///         fn from_parameters_with_remainder<I: Iterator<Item = P>>(
    ///             structure: Self::To<ryft::Placeholder>,
    ///             parameters: &mut I,
    ///         ) -> Result<Self, ryft::Error> {
    ///             let expected_count = structure.parameter_count();
    ///             Ok(...)
    ///         }
    ///     }
    ///
    ///     #[automatically_derived]
    ///     pub enum CustomParameterIterator<'__p, P: '__p + Parameter> where ... {
    ///         ...
    ///     }
    ///
    ///     #[automatically_derived]
    ///     impl<'__p, P: '__p + Parameter> Iterator for CustomParameterIterator<'__p, P> {
    ///         type Item = &'__p P;
    ///
    ///         fn next(&mut self) -> Option<Self::Item> {
    ///             ...
    ///         }
    ///     }
    ///
    ///     ...
    /// };
    /// ```
    ///
    /// The additional enum and [`Iterator`] implementation outside the main `impl` block are only going to be generated
    /// if they are necessary for implementing the parameter iterators without incurring a performance loss. Currently,
    /// they are never generated for structs and they are always generated for enums.
    pub(crate) fn generate_parameterized_impl(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
        let mut input = syn::parse_macro_input!(input as syn::DeriveInput);

        // Replace any instances of [`Self`] with its fully-qualified path. This is necessary in order to be able to
        // handle recursive types when deriving our [`Parameterized`] implementation.
        replace_self_type(&mut input);

        // Construct a new [`CodeGenerator`] using inconsequential default values for fields whose values need to be
        // extracted from the provided input. These values are inconsequential because if we fail to extract them from
        // the provided input, then we will accumulate all relevant [`syn::Error`]s in [`CodeGenerator::errors`] and
        // return a compiler error before we get to use them.
        let mut generator = CodeGenerator {
            ryft_crate: DEFAULT_RYFT_CRATE.into(),
            macro_parameter_lifetime: DEFAULT_MACRO_PARAMETER_LIFETIME.into(),
            macro_parameter_type: DEFAULT_MACRO_PARAMETER_TYPE.into(),
            parameter_type: DEFAULT_PARAMETER_TYPE.into(),
            generics: syn::Generics::default(),
            data: Data::Struct(StructData { ident: Symbol::new("Data").into(), fields: Vec::new(), kind: Kind::Unit }),
            errors: Vec::new(),
        };

        // Extracts all the necessary information from our [`syn::DeriveInput`].
        generator.extract_attributes(&input);
        generator.extract_parameter_type(&input);
        generator.extract_data(&input);
        generator.extract_generics(&input);
        generator.check_for_name_conflicts(&input);

        // Perform the actual code generation.
        let ryft = &generator.ryft_crate;
        let parameter_type = &generator.parameter_type;
        let ident = generator.data.ident();
        let (impl_generics, ty_generics, where_clause) = generator.generics.split_for_impl();
        let (assoc_types, additional_types_and_impls) = generator.generate_assoc_types();
        let parameter_count = generator.generate_parameter_count_function();
        let parameter_structure = generator.generate_parameter_structure_function();
        let parameters_functions = generator.generate_parameters_functions();
        let from_parameters_with_remainder = generator.generate_from_parameters_with_remainder();
        let code = const_block(quote! {
            #[automatically_derived]
            impl #impl_generics #ryft::Parameterized<#parameter_type> for #ident #ty_generics #where_clause {
                #assoc_types
                #parameter_count
                #parameter_structure
                #parameters_functions
                #from_parameters_with_remainder
            }

            #additional_types_and_impls
        });

        // Make sure to raise a compile-time error if any errors were encountered along the way.
        // This error will contain the information from all encountered errors combined. This is meant
        // to make working with this macro easier by reducing the amount of trial and error required to
        // get it working properly.
        if let Some(error) = generator.compile_error() {
            return error.into();
        }

        code.into()
    }

    /// Returns a [`TokenStream`] that represents a [`compile_error!`] invocation that contain information about
    /// [`syn::Error`]s that have been collected by this [`CodeGenerator`] so far. If there are no errors, then this
    /// function returns [`None`].
    fn compile_error(&self) -> Option<TokenStream> {
        self.errors
            .iter()
            .flatten()
            .reduce(|mut combined_error, error| {
                combined_error.combine(error);
                combined_error
            })
            .map(|error| error.into_compile_error())
    }

    /// Extracts any `#[ryft(...)]` attributes that are attached to the provided [`syn::DeriveInput`] and checks for
    /// unknown top-level (i.e., not field or variant) `#[ryft(...)]` attributes. This function will set
    /// [`CodeGenerator::ryft_crate`], if it is able to successfully extract the attribute value.
    fn extract_attributes(&mut self, input: &syn::DeriveInput) {
        let mut ryft_crate = Attribute::new(CRATE_ATTRIBUTE);
        input.attrs.iter().filter(|attr| attr.path() == &RYFT_ATTRIBUTE).for_each(|attr| {
            attr.parse_nested_meta(|meta| match &meta.path {
                path if path == &CRATE_ATTRIBUTE => ryft_crate.set(&meta),
                _ => Err(meta.error(format_args!(
                    "Invalid '#[ryft(...)]' attribute: '{}'. These are the attributes that are supported here: {:?}.",
                    meta.path.to_token_stream().to_string().replace(' ', ""),
                    VALID_CONTAINER_ATTRIBUTES,
                ))),
            })
            .unwrap_or_else(|error| self.errors.push(error));
        });

        if let Some(ryft_crate) = ryft_crate.get() {
            self.ryft_crate = ryft_crate;
        }
    }

    /// Extracts the generic parameter type identifier which is the [`Parameter`] type for the [`Parameterized`]
    /// implementation that will be derived. This function also checks if there are more than one generic type
    /// parameters bounded by [`Parameter`] or if there are none, and reports errors as needed. Note that this
    /// function inspects the bounds on the generic type parameters themselves, as well as any corresponding
    /// where bounds.
    fn extract_parameter_type(&mut self, input: &syn::DeriveInput) {
        let generics = &input.generics;

        // Helper that checks if the provided [`syn::TypeParamBound`] is a [`ryft::Parameter`] bound. Given that
        // procedural macros are executed before type checking and inference is performed by the Rust compiler,
        // we cannot consider all possible ways of specifying this bound (e.g., using aliases) and therefore
        // this function simply checks for a match to either `Parameter` or `ryft::Parameter`, with `ryft`
        // substituted for the provided `ryft_crate`.
        let check_bound = |ryft_crate: &syn::Path, bound: &syn::TypeParamBound| match &bound {
            syn::TypeParamBound::Trait(syn::TraitBound { path, .. }) => {
                path.get_ident().map(|ident| ident == "Parameter").unwrap_or(false)
                    || path == &ryft_crate.with_segment(Symbol::new("Parameter").into())
            }
            _ => false,
        };

        // Helpers that applies `check_bound` on each of the provided bounds, returning `true` if any of them do.
        let check_bounds = |ryft_crate: &syn::Path, bounds: &syn::punctuated::Punctuated<syn::TypeParamBound, _>| {
            bounds.into_iter().any(|bound| check_bound(ryft_crate, bound))
        };

        // Collect all parameters that are bounded by [`ryft::Parameter`].
        let where_predicates = generics.where_clause.iter().flat_map(|clause| clause.predicates.iter());
        let parameter_types: HashSet<&syn::Ident> = generics
            .type_params()
            .filter(|parameter| parameter.bounds.iter().any(|bound| check_bound(&self.ryft_crate, bound)))
            .map(|parameter| &parameter.ident)
            .chain(where_predicates.flat_map(|predicate| match &predicate {
                syn::WherePredicate::Type(p) if check_bounds(&self.ryft_crate, &p.bounds) => p.bounded_ty.ident(),
                _ => None,
            }))
            .collect();

        // Check that there is only a single unique type parameter bounded by [`ryft::Parameter`].
        if parameter_types.len() > 1 {
            self.add_error(
                generics,
                "Found more than one generic types bounded by 'Parameter'. To use the ryft '#[derive(Parameterized)]' \
                macro you must bound exactly one type parameter with the 'Parameter' trait; not none and not more \
                than one. The bound must be specified as 'Parameter' or 'ryft::Parameter' where 'ryft' must be \
                substituted for the custom ryft 'crate' provided, if one was provided using the \
                '#[ryft(crate = ...)]' attribute.",
            );
        } else if let Some(parameter_type) = parameter_types.into_iter().next() {
            self.parameter_type = parameter_type.clone();
        } else {
            self.add_error(
                &input.ident,
                "Found no generic types bounded by 'Parameter'. To use the ryft '#[derive(Parameterized)]' \
                macro you must bound exactly one type parameter with the 'Parameter' trait; not none and not more \
                than one. The bound must be specified as 'Parameter' or 'ryft::Parameter' where 'ryft' must be \
                substituted for the custom ryft 'crate' provided, if one was provided using the \
                '#[ryft(crate = ...)]' attribute.",
            );
        }
    }

    /// Extracts the [`Data`] that is contained in the provided [`syn::DeriveInput`]. This function also checks for
    /// invalid or unknown `[#ryft(...)]` attributes in any of the nested [`syn::Field`]s or [`syn::Variant`]s, as well
    /// as for types of data that are not supported by our `#[derive(Parameterized)]` macro.
    fn extract_data(&mut self, input: &syn::DeriveInput) {
        /// Helper function that extracts a [Field] from the provided data. This function also checks for any invalid
        /// or unknown `[#ryft(...)]` attributes attached to the corresponding [`syn::Field`].
        ///
        /// # Parameters
        ///
        ///   * `generator` - [`CodeGenerator`] from within which this function is called. We need to pass it as an
        ///     additional argument because Rust function do not capture the surrounding generator.
        ///   * `field_index` - Index of the [`Field`] in the container in which it belongs.
        ///   * `field_ident` - Optional [`syn::Ident`] of the [`Field`]. This must be set to [`None`] for anonymous fields
        ///     (e.g., the fields/elements of a tuple).
        ///   * `field_ty` - [`syn::Type`] of the [`Field`].
        ///   * `field_attrs` - Optional [`syn::Attribute`]s attached to the [`Field`].
        fn extract_field(
            generator: &mut CodeGenerator,
            field_index: usize,
            field_ident: Option<syn::Ident>,
            field_ty: syn::Type,
            field_attrs: Option<&Vec<syn::Attribute>>,
        ) -> Field {
            // Check for invalid '#[ryft(...)]' attributes.
            field_attrs
                .iter()
                .flat_map(|attrs| attrs.iter())
                .filter(|attr| attr.path() == &RYFT_ATTRIBUTE)
                .for_each(|attr| generator.add_error(attr, FIELD_ATTRIBUTE_ERROR));

            // Check if the field is a parameter field and perform validation checks, if necessary.
            let is_parameter = field_ty.references_ident(&generator.parameter_type);
            match &field_ty {
                syn::Type::Ptr(_) if is_parameter => {
                    generator.add_error(&field_ty, "Ryft parameters must be owned. Pointers are not allowed.")
                }
                syn::Type::Reference(_) if is_parameter => {
                    generator.add_error(&field_ty, "Ryft parameters must be owned. References are not allowed.")
                }
                _ => {}
            }

            // Construct a [`Field`] from the provided information.
            Field {
                is_parameter,
                index: field_index,
                ident: field_ident,
                ty: field_ty.clone(),
                fields: {
                    if let syn::Type::Tuple(type_tuple) = &field_ty {
                        Some(
                            type_tuple
                                .elems
                                .iter()
                                .enumerate()
                                .map(|(i, e)| extract_field(generator, i, None, e.clone(), None))
                                .collect(),
                        )
                    } else {
                        None
                    }
                },
            }
        }

        /// Helper function that extracts a [`Variant`] from the provided [`syn::Variant`]. This function also checks
        /// for any invalid or unknown `[#ryft(...)]` attributes attached to the [`syn::Variant`].
        ///
        /// # Parameters
        ///
        ///   * `generator` - [`CodeGenerator`] from within which this function is called. We need to pass it as an
        ///     additional argument because Rust function do not capture the surrounding generator.
        ///   * `variant` - [`syn::Variant`] from which to extract a [`Variant`].
        fn extract_variant(generator: &mut CodeGenerator, variant: &syn::Variant) -> Variant {
            // Check for invalid '#[ryft(...)]' attributes.
            variant
                .attrs
                .iter()
                .filter(|attr| attr.path() == &RYFT_ATTRIBUTE)
                .for_each(|f| generator.add_error(f, FIELD_ATTRIBUTE_ERROR));
            variant
                .fields
                .iter()
                .filter(|f| f.attrs.iter().any(|attr| attr.path() == &RYFT_ATTRIBUTE))
                .for_each(|f| generator.add_error(f, FIELD_ATTRIBUTE_ERROR));

            // Construct a [`Variant`] from the provided information.
            let fields = variant
                .fields
                .iter()
                .enumerate()
                .map(|(i, f)| extract_field(generator, i, f.ident.clone(), f.ty.clone(), Some(&f.attrs)))
                .collect::<Vec<_>>();
            let kind = match &variant.fields {
                syn::Fields::Named(_) => Kind::Named,
                syn::Fields::Unnamed(_) => Kind::Unnamed,
                syn::Fields::Unit => Kind::Unit,
            };
            Variant { ident: variant.ident.clone(), fields, kind }
        }

        let ident = input.ident.clone();
        match &input.data {
            syn::Data::Struct(data) => {
                let fields = data
                    .fields
                    .iter()
                    .enumerate()
                    .map(|(i, f)| extract_field(self, i, f.ident.clone(), f.ty.clone(), Some(&f.attrs)))
                    .collect::<Vec<_>>();
                self.data = match &data.fields {
                    syn::Fields::Named(_) => Data::Struct(StructData { ident, fields, kind: Kind::Named }),
                    syn::Fields::Unnamed(_) => Data::Struct(StructData { ident, fields, kind: Kind::Unnamed }),
                    syn::Fields::Unit => Data::Struct(StructData { ident, fields, kind: Kind::Unit }),
                };
            }
            syn::Data::Enum(data) => {
                let variants = data.variants.iter().map(|variant| extract_variant(self, variant)).collect();
                self.data = Data::Enum(EnumData { ident, variants });
            }
            syn::Data::Union(_) => {
                self.add_error(&input.ident, "The '#[derive(ryft::Parameterized)]' macro does not support unions.");
            }
        }
    }

    /// Extracts the [`syn::Generics`] that is contained in the provided [`syn::DeriveInput`] and post-processes it.
    /// Refer to the documentation of [`CodeGenerator::generics`] for information on the post-processing. This function
    /// will set [`CodeGenerator::generics`] to the resulting [`syn::Generics`], if it is able to successfully extract
    /// them.
    fn extract_generics(&mut self, input: &syn::DeriveInput) {
        let mut generics = input.generics.clone();

        /// Adds a trait bound to the provided [`syn::Generics`] for [`syn::Type`] `ty`.
        fn add_trait_bound(generics: &mut syn::Generics, ty: syn::Type, bound: syn::Path) {
            // Note that we may add redundant bounds here (e.g., for non-generic types) or even duplicate bounds.
            // That is ok as the compiler will take care of removing any redundancies later on. It is actually hard
            // and maybe even impossible to eliminate these redundancies in a robust manner here because procedural
            // macros are executed before type checking/inference and that makes dealing with things like type aliases
            // correctly practically impossible (as we would need information that is not necessarily part of the
            // [`syn::DeriveInput`] that this code is operating over).
            let mut bounds = syn::punctuated::Punctuated::new();
            bounds.push(syn::TypeParamBound::Trait(syn::TraitBound {
                paren_token: None,
                modifier: syn::TraitBoundModifier::None,
                lifetimes: None,
                path: bound,
            }));
            generics.make_where_clause().predicates.push(syn::WherePredicate::Type(syn::PredicateType {
                lifetimes: None,
                bounded_ty: ty,
                colon_token: <syn::Token![:]>::default(),
                bounds,
            }));
        }

        /// Adds [`Parameterized`] bounds for the provided [`Field`]'s [`syn::Type`], if it references the
        /// [`CodeGenerator::parameter_type`]. If it does not, then this function will not do anything.
        fn add_parameterized_bounds(generator: &CodeGenerator, generics: &mut syn::Generics, field: &Field) {
            // We do not add [`Parameterized`] bounds for types that do not reference the
            // [`CodeGenerator::parameter_type`].
            if field.is_parameter {
                if let Some(fields) = &field.fields {
                    fields.iter().for_each(|field| add_parameterized_bounds(generator, generics, field));
                } else {
                    // We need to construct a bound that looks like this (abusing notation a bit and representing
                    // the field type as a function of the parameter type), where `P` is the parameter type:
                    //
                    //   FieldType<P>: ryft::Parameterized<
                    //     P,
                    //     ParameterStructure = FieldType<ryft::Placeholder>,
                    //   >
                    //
                    // Therefore, we need to construct the `FieldType<ryft::Placeholder>` type and that is what
                    // is happening here using `replace_ident`. Note that we could be constructing this bound using
                    // `parse_quote!` but we instead construct it manually in order to avoid the parsing overhead.
                    let ryft_placeholder = generator.ryft_crate.with_segment(Symbol::new("Placeholder").into());

                    let mut ty_using_parameter_placeholder = field.ty.clone();
                    ty_using_parameter_placeholder.replace_ident(&generator.parameter_type, &ryft_placeholder);

                    // We need to construct the full [`ryft::Parameterized`] bound and to do that, we first construct
                    // its arguments `<P, ParameterStructure = FieldType<ryft::Placeholder>>`.
                    let mut args = syn::punctuated::Punctuated::new();

                    // Add the `P` generic argument.
                    let parameter_type = generator.parameter_type.clone();
                    let parameter_type = syn::Type::Path(syn::TypePath { qself: None, path: parameter_type.into() });
                    args.push(syn::GenericArgument::Type(parameter_type));

                    // Add the `ParameterStructure = FieldType<ryft::Placeholder>` generic argument.
                    args.push(syn::GenericArgument::AssocType(syn::AssocType {
                        ident: syn::Ident::new("ParameterStructure", Span::call_site()),
                        generics: None,
                        eq_token: <syn::Token![=]>::default(),
                        ty: ty_using_parameter_placeholder.clone(),
                    }));

                    // Then, we construct the full bound [`syn::Path`].
                    let bound = generator.ryft_crate.with_segment(syn::PathSegment {
                        ident: syn::Ident::new("Parameterized", Span::call_site()),
                        arguments: syn::PathArguments::AngleBracketed(syn::AngleBracketedGenericArguments {
                            colon2_token: None,
                            lt_token: <syn::Token![<]>::default(),
                            args,
                            gt_token: <syn::Token![>]>::default(),
                        }),
                    });

                    add_trait_bound(generics, field.ty.clone(), bound);
                }
            }
        }

        /// Adds any bounds that are necessary for the provided [`Field`] to the provided [`syn::Generics`].
        fn add_field_bounds(generator: &CodeGenerator, generics: &mut syn::Generics, field: &Field) {
            if field.is_parameter {
                add_parameterized_bounds(generator, generics, field);
            } else {
                let bound = syn::Path::from(syn::Ident::new("Clone", Span::call_site()));
                add_trait_bound(generics, field.ty.clone(), bound);
            }
        }

        // Go over all [`Fields`] in the underlying [`Data`] (including fields that may be nested inside [`Variant`]s),
        // and add any necessary trait bounds for them.
        match &self.data {
            Data::Struct(data) => data.fields.iter().for_each(|f| add_field_bounds(self, &mut generics, f)),
            Data::Enum(data) => data
                .variants
                .iter()
                .flat_map(|v| v.fields.iter())
                .for_each(|f| add_field_bounds(self, &mut generics, f)),
        };

        self.generics = generics;
    }

    /// Checks whether there are any name conflicts in the provided [`syn::DeriveInput`]. Specifically, this function
    /// will check for whether the reserved [`MACRO_PARAMETER_LIFETIME`] and [`MACRO_PARAMETER_TYPE`] identifiers appear
    /// anywhere in the provided [`syn::DeriveInput`] and will report corresponding errors if they do.
    fn check_for_name_conflicts(&mut self, input: &syn::DeriveInput) {
        input.generics.params.iter().for_each(|parameter| match parameter {
            syn::GenericParam::Lifetime(parameter) if parameter.lifetime == self.macro_parameter_lifetime => {
                self.add_error(
                    parameter,
                    format_args!("Identifier '{}' is reserved.", self.macro_parameter_lifetime.clone()),
                );
            }
            syn::GenericParam::Type(parameter) if parameter.matches_ident(&self.macro_parameter_type) => {
                self.add_error(
                    parameter,
                    format_args!("Identifier '{}' is reserved.", self.macro_parameter_type.clone()),
                );
            }
            syn::GenericParam::Const(parameter) if parameter.matches_ident(&self.macro_parameter_type) => {
                self.add_error(
                    parameter,
                    format_args!("Identifier '{}' is reserved.", self.macro_parameter_type.clone()),
                );
            }
            _ => {}
        });
    }

    /// Adds an error to this [`CodeGenerator`] with the specified message spanning the provided tokens. This is only
    /// meant to be used internally by this class as a convenient helper for collecting errors.
    ///
    /// # Parameters
    ///
    ///   * `tokens` - Tokens that the error spans.
    ///   * `message` - Message describing the error.
    fn add_error<T: ToTokens, U: Display>(&mut self, tokens: T, message: U) {
        self.errors.push(syn::Error::new_spanned(tokens.into_token_stream(), message));
    }

    /// Generates the associated [`Parameterized::Family`], [`Parameterized::To`],
    /// [`Parameterized::ParameterStructure`], [`Parameterized::ParameterIterator`],
    /// [`Parameterized::ParameterIteratorMut`], and [`Parameterized::ParameterIntoIterator`] type declarations
    /// for the [`Data`] owned by the provided [`CodeGenerator`]. This function returns two [`TokenStream`]s:
    ///
    ///   1. The first [`TokenStream`] contains all the associated type declarations without the surrounding `impl`
    ///      block. For example, the generated code may look something like this (with `...` substituted for generated
    ///      code, of course, and the `ryft` crate possibly substituted as well):
    ///
    ///      ```ignore
    ///      type Family = ...;
    ///
    ///      type To<__P: ryft::Parameter> =
    ///          <Self::Family as ryft::ParameterizedFamily<__P>>::To
    ///      where
    ///          Self::Family: ryft::ParameterizedFamily<__P>;
    ///
    ///      type ParameterStructure = Self::To<ryft::Placeholder>;
    ///
    ///      type ParameterIterator<'__p, __P: '__p + ryft::Parameter> = ... where Self: '__p;
    ///      type ParameterIteratorMut<'__p, __P: '__p + ryft::Parameter> = ... where Self: '__p;
    ///      type ParameterIntoIterator<__P: ryft::Parameter> = ...;
    ///
    ///      type NamedParameterIterator<'__p, __P: '__p + ryft::Parameter> = ... where Self: '__p;
    ///      type NamedParameterIteratorMut<'__p, __P: '__p + ryft::Parameter> = ... where Self: '__p;
    ///      type NamedParameterIntoIterator<__P: ryft::Parameter> = ...;
    ///      ```
    ///
    ///   2. The second [`TokenStream`] contains any additional new custom types and corresponding `impl` blocks that
    ///      may be required (as in, referenced) by the type declarations in the first [`TokenStream`]. For example, the
    ///      generated code may look something like this (with `...` substituted for generated code, of course, and the
    ///      `ryft` crate possibly substituted as well):
    ///
    ///      ```ignore
    ///      #[automatically_derived]
    ///      impl<P: ryft::Parameter> ParameterizedFamily<P> for ... {
    ///          type To = ...;
    ///      }
    ///
    ///      #[automatically_derived]
    ///      pub enum CustomParameterIterator<'__p, P: '__p + Parameter> where ... {
    ///          ...
    ///      }
    ///
    ///      #[automatically_derived]
    ///      impl<'__p, P: '__p + Parameter> Iterator for CustomParameterIterator<'__p, P> {
    ///          type Item = &'__p P;
    ///
    ///          fn next(&mut self) -> Option<Self::Item> {
    ///              ...
    ///          }
    ///      }
    ///
    ///      #[automatically_derived]
    ///      pub enum CustomParameterIteratorMut<'__p, P: '__p + Parameter> where ... {
    ///          ...
    ///      }
    ///
    ///      #[automatically_derived]
    ///      impl<'__p, P: '__p + Parameter> Iterator for CustomParameterIteratorMut<'__p, P> {
    ///          type Item = &'__p mut P;
    ///
    ///          fn next(&mut self) -> Option<Self::Item> {
    ///              ...
    ///          }
    ///      }
    ///
    ///      #[automatically_derived]
    ///      pub enum CustomParameterIntoIterator<P: Parameter> where ... {
    ///          ...
    ///      }
    ///
    ///      #[automatically_derived]
    ///      impl<P: Parameter> Iterator for CustomParameterIntoIterator<P> {
    ///          type Item = P;
    ///
    ///          fn next(&mut self) -> Option<Self::Item> {
    ///              ...
    ///          }
    ///      }
    ///
    ///      #[automatically_derived]
    ///      pub enum CustomNamedParameterIterator<'__p, P: '__p + Parameter> where ... {
    ///          ...
    ///      }
    ///
    ///      #[automatically_derived]
    ///      impl<'__p, P: '__p + Parameter> Iterator for CustomNamedParameterIterator<'__p, P> {
    ///          type Item = (ryft::ParameterPath, &'__p P);
    ///
    ///          fn next(&mut self) -> Option<Self::Item> {
    ///              ...
    ///          }
    ///      }
    ///
    ///      #[automatically_derived]
    ///      pub enum CustomNamedParameterIteratorMut<'__p, P: '__p + Parameter> where ... {
    ///          ...
    ///      }
    ///
    ///      #[automatically_derived]
    ///      impl<'__p, P: '__p + Parameter> Iterator for CustomNamedParameterIteratorMut<'__p, P> {
    ///          type Item = (ryft::ParameterPath, &'__p mut P);
    ///
    ///          fn next(&mut self) -> Option<Self::Item> {
    ///              ...
    ///          }
    ///      }
    ///
    ///      #[automatically_derived]
    ///      pub enum CustomNamedParameterIntoIterator<P: Parameter> where ... {
    ///          ...
    ///      }
    ///
    ///      #[automatically_derived]
    ///      impl<P: Parameter> Iterator for CustomNamedParameterIntoIterator<P> {
    ///          type Item = (ryft::ParameterPath, P);
    ///
    ///          fn next(&mut self) -> Option<Self::Item> {
    ///              ...
    ///          }
    ///      }
    ///      ```
    ///
    ///      This is intended to cover cases where the parameter iterators cannot be defined in terms of existing types
    ///      and the macro may want to synthesize custom iterator types in order to not sacrifice efficiency. This is
    ///      the case, for example, for derived [`Parameterized`] implementations for enum types. This [`TokenStream`]
    ///      must not be placed inside the [`Parameterized`] `impl` block that this [`CodeGenerator`] produces. It must
    ///      instead be placed adjacent to it. That is taken care of by [`CodeGenerator::generate_parameterized_impl`].
    fn generate_assoc_types(&self) -> (TokenStream, TokenStream) {
        /// Helper that generates one associated parameter-iterator type declaration for the [`Data`]
        /// owned by the provided [`CodeGenerator`]. This includes both unnamed iterators (i.e.,
        /// [`Parameterized::ParameterIterator`], [`Parameterized::ParameterIteratorMut`],
        /// and [`Parameterized::ParameterIntoIterator`]) and named iterators (i.e.,
        /// [`Parameterized::NamedParameterIterator`], [`Parameterized::NamedParameterIteratorMut`],
        /// and [`Parameterized::NamedParameterIntoIterator`]). Note that this function returns a [`TokenStream`] with
        /// the complete type declaration but without the surrounding `impl` block. For example, the generated code may
        /// look something like this (with `...` substituted for generated code, of course, and the `ryft` crate
        /// possibly substituted as well):
        ///
        /// ```ignore
        /// type ParameterIterator<'__p, __P: '__p + ryft::Parameter> = ... where Self: '__p;
        /// type NamedParameterIterator<'__p, __P: '__p + ryft::Parameter> = ... where Self: '__p;
        /// ```
        ///
        /// # Parameters
        ///
        ///   * `generator` - [`CodeGenerator`] from within which this function is being called.
        ///   * `fields` - List of [`Field`]s for which to generate code.
        ///   * `iter_type` - [`IterType`] that specifies which variant among [`Parameterized::ParameterIterator`],
        ///     [`Parameterized::ParameterIteratorMut`], [`Parameterized::ParameterIntoIterator`],
        ///     [`Parameterized::NamedParameterIterator`], [`Parameterized::NamedParameterIteratorMut`],
        ///     and [`Parameterized::NamedParameterIntoIterator`] to generate code for.
        ///   * `iter_parameter_type` - [`syn::Ident`] that represents the item type of the resulting iterator type.
        ///     This is necessary as due to parameter renames that take place in certain cases, we cannot just directly
        ///     use [`CodeGenerator::parameter_type`].
        fn generate_assoc_iter_type(generator: &CodeGenerator, iter_type: &IterType) -> TokenStream {
            let ryft = &generator.ryft_crate;
            let macro_parameter_lifetime = &generator.macro_parameter_lifetime;
            let macro_parameter_type = &generator.macro_parameter_type;

            let body = match &generator.data {
                Data::Struct(StructData { fields, .. }) => {
                    generate_assoc_iter_type_for_fields(generator, fields, iter_type, macro_parameter_type)
                }
                Data::Enum(EnumData { ident, variants }) => {
                    let iterator_ident = format_ident!("{}{}", &ident, iter_type.parameters_assoc_type_name());

                    // Refer to the implementation of [`generate_type_and_impl`] for more information on why the
                    // [`syn::Generics`] here are constructed this way.
                    let parameterized_fields = variants.iter().flat_map(|variant| variant.fields.iter());
                    let generics = generator.generics_for_fields(parameterized_fields);
                    let generics = generics.with_renamed_param(&generator.parameter_type, macro_parameter_type);
                    let generics = match &iter_type {
                        IterType::IntoIter | IterType::NamedIntoIter => generics.clone(),
                        _ => generics.with_lifetime(macro_parameter_lifetime, macro_parameter_type),
                    };
                    let (_, ty_generics, _) = generics.split_for_impl();

                    match &iter_type {
                        IterType::Iter => quote!(#iterator_ident #ty_generics),
                        IterType::IterMut => quote!(#iterator_ident #ty_generics),
                        IterType::IntoIter => quote!(#iterator_ident #ty_generics),
                        IterType::NamedIter => quote!(#iterator_ident #ty_generics),
                        IterType::NamedIterMut => quote!(#iterator_ident #ty_generics),
                        IterType::NamedIntoIter => quote!(#iterator_ident #ty_generics),
                    }
                }
            };

            match &iter_type {
                IterType::Iter => quote! {
                    type ParameterIterator<
                        #macro_parameter_lifetime,
                        #macro_parameter_type: #macro_parameter_lifetime + #ryft::Parameter,
                    > = #body where Self: #macro_parameter_lifetime;
                },
                IterType::IterMut => quote! {
                    type ParameterIteratorMut<
                        #macro_parameter_lifetime,
                        #macro_parameter_type: #macro_parameter_lifetime + #ryft::Parameter,
                    > = #body where Self: #macro_parameter_lifetime;
                },
                IterType::IntoIter => quote! {
                    type ParameterIntoIterator<#macro_parameter_type: #ryft::Parameter> = #body;
                },
                IterType::NamedIter => quote! {
                    type NamedParameterIterator<
                        #macro_parameter_lifetime,
                        #macro_parameter_type: #macro_parameter_lifetime + #ryft::Parameter,
                    > = #body where Self: #macro_parameter_lifetime;
                },
                IterType::NamedIterMut => quote! {
                    type NamedParameterIteratorMut<
                        #macro_parameter_lifetime,
                        #macro_parameter_type: #macro_parameter_lifetime + #ryft::Parameter,
                    > = #body where Self: #macro_parameter_lifetime;
                },
                IterType::NamedIntoIter => quote! {
                    type NamedParameterIntoIterator<#macro_parameter_type: #ryft::Parameter> = #body;
                },
            }
        }

        /// Helper that generates the associated iterator type portion for the provided [`Field`]s. This includes both
        /// unnamed iterators (i.e., [`Parameterized::ParameterIterator`], [`Parameterized::ParameterIteratorMut`],
        /// and [`Parameterized::ParameterIntoIterator`]) and named iterators (i.e.,
        /// [`Parameterized::NamedParameterIterator`], [`Parameterized::NamedParameterIteratorMut`],
        /// and [`Parameterized::NamedParameterIntoIterator`]). The resulting [`TokenStream`] contains an expression
        /// that evaluates to the appropriate type and that can be nested directly within the generic arguments of other
        /// types (e.g., for nested tuples). It is the responsibility of the caller to nest this appropriately within
        /// other types, if needed, and to generate the surrounding code for the appropriate associated type
        /// declaration.
        ///
        /// # Parameters
        ///
        ///   * `generator` - [`CodeGenerator`] from within which this function is being called.
        ///   * `fields` - List of [`Field`]s for which to generate code.
        ///   * `iter_type` - [`IterType`] that specifies which variant among [`Parameterized::ParameterIterator`],
        ///     [`Parameterized::ParameterIteratorMut`], [`Parameterized::ParameterIntoIterator`],
        ///     [`Parameterized::NamedParameterIterator`], [`Parameterized::NamedParameterIteratorMut`],
        ///     and [`Parameterized::NamedParameterIntoIterator`] to generate code for.
        ///   * `iter_parameter_type` - [`syn::Ident`] that represents the item type of the resulting iterator type.
        ///     This is necessary as due to parameter renames that take place in certain cases, we cannot just
        ///     directly use [`CodeGenerator::parameter_type`].
        fn generate_assoc_iter_type_for_fields(
            generator: &CodeGenerator,
            fields: &[Field],
            iter_type: &IterType,
            iter_parameter_type: &syn::Ident,
        ) -> TokenStream {
            let ryft = &generator.ryft_crate;
            let lifetime = &generator.macro_parameter_lifetime;
            fields
                .iter()
                .filter_map(|field| match &field {
                    Field { is_parameter: false, .. } => None,
                    Field { ty, fields: None, .. } => {
                        let parameter_ty = &generator.parameter_type;
                        let assoc_ty = match &iter_type {
                            IterType::Iter => quote!(ParameterIterator<#lifetime, #iter_parameter_type>),
                            IterType::IterMut => quote!(ParameterIteratorMut<#lifetime, #iter_parameter_type>),
                            IterType::IntoIter => quote!(ParameterIntoIterator<#iter_parameter_type>),
                            IterType::NamedIter => quote!(NamedParameterIterator<#lifetime, #iter_parameter_type>),
                            IterType::NamedIterMut => {
                                quote!(NamedParameterIteratorMut<#lifetime, #iter_parameter_type>)
                            }
                            IterType::NamedIntoIter => quote!(NamedParameterIntoIterator<#iter_parameter_type>),
                        };
                        Some(quote!(<#ty as #ryft::Parameterized<#parameter_ty>>::#assoc_ty))
                    }
                    Field { fields: Some(fields), .. } => {
                        Some(generate_assoc_iter_type_for_fields(generator, fields, iter_type, iter_parameter_type))
                    }
                })
                .map(|iterator_ty| match &iter_type {
                    IterType::Iter | IterType::IterMut | IterType::IntoIter => iterator_ty,
                    IterType::NamedIter => quote!(
                        #ryft::PathPrefixedParameterIterator<&#lifetime #iter_parameter_type, #iterator_ty>
                    ),
                    IterType::NamedIterMut => quote!(
                        #ryft::PathPrefixedParameterIterator<&#lifetime mut #iter_parameter_type, #iterator_ty>
                    ),
                    IterType::NamedIntoIter => {
                        quote!(#ryft::PathPrefixedParameterIterator<#iter_parameter_type, #iterator_ty>)
                    }
                })
                .reduce(|chain_ty, ty| quote!(std::iter::Chain<#chain_ty, #ty>))
                .unwrap_or_else(|| {
                    let item_ty = match &iter_type {
                        IterType::Iter => quote!(&#lifetime #iter_parameter_type),
                        IterType::IterMut => quote!(&#lifetime mut #iter_parameter_type),
                        IterType::IntoIter => quote!(#iter_parameter_type),
                        IterType::NamedIter => quote!((#ryft::ParameterPath, &#lifetime #iter_parameter_type)),
                        IterType::NamedIterMut => quote!((#ryft::ParameterPath, &#lifetime mut #iter_parameter_type)),
                        IterType::NamedIntoIter => quote!((#ryft::ParameterPath, #iter_parameter_type)),
                    };
                    quote!(std::iter::Empty<#item_ty>)
                })
        }

        /// Helper that generates any custom iterator types and their corresponding [`Iterator`] `impl` blocks
        /// that may be necessary for the corresponding unnamed iterator declarations (i.e.,
        /// [`Parameterized::ParameterIterator`], [`Parameterized::ParameterIteratorMut`],
        /// and [`Parameterized::ParameterIntoIterator`]) and named iterator declarations (i.e.,
        /// [`Parameterized::NamedParameterIterator`], [`Parameterized::NamedParameterIteratorMut`], and
        /// [`Parameterized::NamedParameterIntoIterator`]). Note that this function returns a [`TokenStream`]
        /// with the complete type declarations and corresponding `impl` blocks that must be placed outside of any other
        /// `impl` blocks that this [`CodeGenerator`] may produce. For example, the generated code may look something
        /// like this (with `...` substituted for generated code, of course, and the `ryft` crate possibly substituted
        /// as well) for a specific [`IterType`]:
        ///
        /// ```ignore
        /// #[automatically_derived]
        /// pub enum CustomParameterIterator<'__p, P: '__p + Parameter> where ... {
        ///     ...
        /// }
        ///
        /// #[automatically_derived]
        /// impl<'__p, P: '__p + Parameter> Iterator for CustomParameterIterator<'__p, P> {
        ///     type Item = &'__p P;
        ///
        ///     fn next(&mut self) -> Option<Self::Item> {
        ///         ...
        ///     }
        /// }
        /// ```
        fn generate_iter_type_and_impl(generator: &CodeGenerator, iter_type: &IterType) -> Option<TokenStream> {
            match &generator.data {
                Data::Struct(_) => None,
                Data::Enum(EnumData { ident, variants }) => {
                    let ryft = &generator.ryft_crate;
                    let macro_parameter_lifetime = &generator.macro_parameter_lifetime;
                    let parameter_type = &generator.parameter_type;
                    let iterator_ident = format_ident!("{}{}", &ident, iter_type.parameters_assoc_type_name());
                    let item_ty = match &iter_type {
                        IterType::Iter => quote!(&#macro_parameter_lifetime #parameter_type),
                        IterType::IterMut => quote!(&#macro_parameter_lifetime mut #parameter_type),
                        IterType::IntoIter => quote!(#parameter_type),
                        IterType::NamedIter => {
                            quote!((#ryft::ParameterPath, &#macro_parameter_lifetime #parameter_type))
                        }
                        IterType::NamedIterMut => {
                            quote!((#ryft::ParameterPath, &#macro_parameter_lifetime mut #parameter_type))
                        }
                        IterType::NamedIntoIter => quote!((#ryft::ParameterPath, #parameter_type)),
                    };

                    // When generating types that hold references (e.g., iterators over parameter references),
                    // we need to add a new lifetime parameter to our generics that is used to bound the lifetimes
                    // of those references.
                    let parameterized_fields = variants.iter().flat_map(|variant| variant.fields.iter());
                    let generics = generator.generics_for_fields(parameterized_fields);
                    let generics = match &iter_type {
                        IterType::IntoIter | IterType::NamedIntoIter => generics.clone(),
                        _ => generics.with_lifetime(macro_parameter_lifetime, parameter_type),
                    };
                    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

                    let iterator_variants = variants.iter().map(|variant| {
                        let variant_ident = &variant.ident;
                        let iterator_type = generate_assoc_iter_type_for_fields(
                            generator,
                            &variant.fields,
                            iter_type,
                            &generator.parameter_type,
                        );
                        let iterator_type = match &iter_type {
                            IterType::Iter | IterType::IterMut | IterType::IntoIter => iterator_type,
                            IterType::NamedIter => {
                                quote!(#ryft::PathPrefixedParameterIterator<
                                    &#macro_parameter_lifetime #parameter_type,
                                    #iterator_type,
                                >)
                            }
                            IterType::NamedIterMut => {
                                quote!(#ryft::PathPrefixedParameterIterator<
                                    &#macro_parameter_lifetime mut #parameter_type,
                                    #iterator_type,
                                >)
                            }
                            IterType::NamedIntoIter => {
                                quote!(#ryft::PathPrefixedParameterIterator<#parameter_type, #iterator_type>)
                            }
                        };
                        quote!(#variant_ident(#iterator_type))
                    });

                    let iterator_variant_impls = variants.iter().map(|variant| {
                        let variant_ident = &variant.ident;
                        quote!(#iterator_ident::#variant_ident(iterator) => iterator.next())
                    });

                    Some(quote! {
                        #[automatically_derived]
                        pub enum #iterator_ident #impl_generics #where_clause {
                            #(#iterator_variants,)*
                        }

                        #[automatically_derived]
                        impl #impl_generics Iterator for #iterator_ident #ty_generics #where_clause {
                            type Item = #item_ty;

                            fn next(&mut self) -> Option<Self::Item> {
                                match self {
                                    #(#iterator_variant_impls,)*
                                }
                            }
                        }
                    })
                }
            }
        }

        let ryft = &self.ryft_crate;
        let macro_parameter_type = &self.macro_parameter_type;
        let parameter_type = &self.parameter_type;
        let ident = self.data.ident();

        // For generated `To` associated types we need to rename the parameter type in our [`syn::Generics`].
        // That is because if we do not rename it, we will end up with something like this:
        // ```
        // impl<P: Parameter> ParameterizedFamily<P> for SomeType<Placeholder> {
        //     type To = SomeType<P>;
        //     ...
        // }
        // ```
        // In this case, the name `P` conflicts with the generic parameter of the `impl` block itself.
        // What we really want is something like this instead:
        // ```
        // impl<__P: Parameter> ParameterizedFamily<__P> for SomeType<Placeholder> {
        //     type To = SomeType<__P>;
        //     ...
        // }
        // ```
        // where `__P` is a fresh and unique identifier. This generic parameter replacement is what
        // takes place in the following line (the uniqueness check for the name is performed in
        // [`CodeGenerator::check_for_name_conflicts`]).
        let to_assoc_ty_generics = self.generics.with_renamed_param(parameter_type, macro_parameter_type);
        let (_, to_assoc_ty_generics, _) = to_assoc_ty_generics.split_for_impl();
        let to_assoc_parameter = quote!(#macro_parameter_type: #ryft::Parameter);

        // Generate the [`ParameterizedFamily`] implementation block first.
        let family_generics = self.generics.with_renamed_param(parameter_type, macro_parameter_type);
        let (family_impl_generics, _, family_where_clause) = family_generics.split_for_impl();
        let family_type = if self.generics.params.is_empty() {
            quote!(#ident)
        } else {
            let family_type_args = self.generics.params.iter().map(|parameter| match parameter {
                syn::GenericParam::Lifetime(lifetime_param) => {
                    let lifetime = &lifetime_param.lifetime;
                    quote!(#lifetime)
                }
                syn::GenericParam::Type(type_param) if type_param.ident == *parameter_type => {
                    quote!(#ryft::Placeholder)
                }
                syn::GenericParam::Type(type_param) => {
                    let ident = &type_param.ident;
                    quote!(#ident)
                }
                syn::GenericParam::Const(const_param) => {
                    let ident = &const_param.ident;
                    quote!(#ident)
                }
            });
            quote!(#ident<#(#family_type_args),*>)
        };
        let family_impl = quote! {
            #[automatically_derived]
            impl #family_impl_generics #ryft::ParameterizedFamily<#macro_parameter_type>
                for #family_type #family_where_clause
            {
                type To = #ident #to_assoc_ty_generics;
            }
        };

        // Generate the [`Parameterized`] associated type declarations.
        let family_assoc_ty = quote!(type Family = #family_type;);
        let to_assoc_ty = quote!(
            type To<#to_assoc_parameter>
                = <Self::Family as #ryft::ParameterizedFamily<#macro_parameter_type>>::To
            where
                Self::Family: #ryft::ParameterizedFamily<#macro_parameter_type>;
        );
        let parameter_structure_assoc_type = quote!(type ParameterStructure = Self::To<#ryft::Placeholder>;);

        let parameter_iterator_assoc_type = generate_assoc_iter_type(self, &IterType::Iter);
        let parameter_iterator_mut_assoc_type = generate_assoc_iter_type(self, &IterType::IterMut);
        let parameter_into_iterator_assoc_type = generate_assoc_iter_type(self, &IterType::IntoIter);
        let named_parameter_iterator_assoc_type = generate_assoc_iter_type(self, &IterType::NamedIter);
        let named_parameter_iterator_mut_assoc_type = generate_assoc_iter_type(self, &IterType::NamedIterMut);
        let named_parameter_into_iterator_assoc_type = generate_assoc_iter_type(self, &IterType::NamedIntoIter);
        let parameter_iterator_assoc_types = quote! {
            #family_assoc_ty
            #to_assoc_ty
            #parameter_structure_assoc_type
            #parameter_iterator_assoc_type
            #parameter_iterator_mut_assoc_type
            #parameter_into_iterator_assoc_type
            #named_parameter_iterator_assoc_type
            #named_parameter_iterator_mut_assoc_type
            #named_parameter_into_iterator_assoc_type
        };

        let parameter_iterator_type_and_impl = generate_iter_type_and_impl(self, &IterType::Iter);
        let parameter_iterator_mut_type_and_impl = generate_iter_type_and_impl(self, &IterType::IterMut);
        let parameter_into_iterator_type_and_impl = generate_iter_type_and_impl(self, &IterType::IntoIter);
        let named_parameter_iterator_type_and_impl = generate_iter_type_and_impl(self, &IterType::NamedIter);
        let named_parameter_iterator_mut_type_and_impl = generate_iter_type_and_impl(self, &IterType::NamedIterMut);
        let named_parameter_into_iterator_type_and_impl = generate_iter_type_and_impl(self, &IterType::NamedIntoIter);
        let parameter_iterator_types_and_impls = quote! {
            #family_impl
            #parameter_iterator_type_and_impl
            #parameter_iterator_mut_type_and_impl
            #parameter_into_iterator_type_and_impl
            #named_parameter_iterator_type_and_impl
            #named_parameter_iterator_mut_type_and_impl
            #named_parameter_into_iterator_type_and_impl
        };

        (parameter_iterator_assoc_types, parameter_iterator_types_and_impls)
    }

    /// Generates an implementation for the [`Parameterized::parameter_count`] for the [`Data`] owned by this
    /// [`CodeGenerator`]. Note that this function returns a [`TokenStream`] with the complete function declaration and
    /// implementation but without the surrounding `impl` block. For example, the generated code may look something like
    /// this (with `...` substituted for generated code, of course, and the `ryft` crate possibly substituted as well):
    ///
    /// ```ignore
    /// fn parameter_count(&self) -> usize { ... }
    /// ```
    fn generate_parameter_count_function(&self) -> TokenStream {
        /// Helper that generates the body portion of the [`Parameterized::parameter_count`] function for the provided
        /// [`Field`]s. The resulting [`TokenStream`] contains an expression that evaluates to the appropriate [`usize`]
        /// and that can be nested directly within the body of the owning function. It is the responsibility of the
        /// caller to surround this [`TokenStream`] with braces and/or parenthesis and to generate the rest of the
        /// surrounding code.
        ///
        /// # Parameters
        ///
        ///   * `fields` - List of [`Field`]s for which to generate code.
        ///   * `receiver` - Optional [`TokenStream`] that represents the "owner" of the provided [`Field`]s. Note that
        ///     the owner may be a tuple, a struct, or the variant of an enum. This is used to generate expressions that
        ///     access values of the provided [`Field`]s by invoking [`Field::member`].
        fn generate_body(fields: &[Field], receiver: Option<TokenStream>) -> TokenStream {
            fields.iter().fold(quote!(0usize), |token_stream, field| {
                let receiver = field.member(receiver.as_ref());
                match &field {
                    Field { is_parameter: false, .. } => token_stream,
                    Field { fields: None, .. } => quote!(#token_stream + #receiver.parameter_count()),
                    Field { fields: Some(fields), .. } => {
                        let fields_expression = generate_body(fields, Some(receiver));
                        quote!(#token_stream + #fields_expression)
                    }
                }
            })
        }

        let body = match &self.data {
            Data::Struct(StructData { fields, .. }) => generate_body(fields, Some(quote!(self))),
            Data::Enum(EnumData { ident, variants }) => {
                let variant_counts = variants.iter().map(|variant| {
                    let variant_ident = &variant.ident;
                    let variant_path = quote!(#ident::#variant_ident);
                    let fields = variant.fields.iter().map(|field| field.member(None));
                    let fields_body = generate_body(&variant.fields, None);
                    match &variant.kind {
                        Kind::Unit => quote!(#variant_path => 0usize),
                        Kind::Unnamed => quote!(#variant_path (#(#fields,)*) => #fields_body),
                        Kind::Named => quote!(#variant_path { #(#fields,)* } => #fields_body),
                    }
                });
                quote!(match self { #(#variant_counts,)* })
            }
        };

        quote!(fn parameter_count(&self) -> usize { #body })
    }

    /// Generates an implementation for the [`Parameterized::parameter_structure`] for the [`Data`] owned by this
    /// [`CodeGenerator`]. Note that this function returns a [`TokenStream`] with the complete function declaration
    /// and implementation but without the surrounding `impl` block. For example, the generated code may look something
    /// like this (with `...` substituted for generated code, of course, and the `ryft` crate possible substituted
    /// as well):
    ///
    /// ```ignore
    /// fn parameter_structure(&self) -> Self::To<ryft::Placeholder> { ... }
    /// ```
    fn generate_parameter_structure_function(&self) -> TokenStream {
        /// Helper that generates the body portion of the [`Parameterized::parameter_structure`] function for the
        /// provided [`Field`]s. The resulting [`TokenStream`] contains an expression that evaluates to the appropriate
        /// structure and that can be nested directly within the body of the owning function. It is the responsibility
        /// of the caller to surround this [`TokenStream`] with braces and/or parenthesis and to generate the rest of
        /// the surrounding code.
        ///
        /// # Parameters
        ///
        ///   * `fields` - List of [`Field`]s for which to generate code.
        ///   * `receiver` - Optional [`TokenStream`] that represents the "owner" of the provided [`Field`]s. Note that
        ///     the owner may be a tuple, a struct, or the variant of an enum. This is used to generate expressions that
        ///     access values of the provided [`Field`]s by invoking [`Field::member`].
        fn generate_body(fields: &[Field], receiver: Option<TokenStream>) -> TokenStream {
            fields
                .iter()
                .map(|field| {
                    let prefix = field.ident.as_ref().map(|ident| quote!(#ident:)).unwrap_or(TokenStream::new());
                    let receiver = field.member(receiver.as_ref());
                    match &field {
                        Field { is_parameter: false, .. } => quote!(#prefix #receiver.clone()),
                        Field { fields: None, .. } => quote!(#prefix #receiver.parameter_structure()),
                        Field { fields: Some(fields), .. } => {
                            let fields_expression = generate_body(fields, Some(receiver));
                            quote!(#prefix (#fields_expression))
                        }
                    }
                })
                .reduce(|token_stream, field_expression| quote!(#token_stream, #field_expression))
                .unwrap_or(TokenStream::new())
        }

        let body = match &self.data {
            Data::Struct(StructData { ident, fields, kind }) => {
                let fields_body = generate_body(fields, Some(quote!(self)));
                match &kind {
                    Kind::Unit => quote!(#ident),
                    Kind::Unnamed => quote!(#ident ( #fields_body )),
                    Kind::Named => quote!(#ident { #fields_body }),
                }
            }
            Data::Enum(EnumData { ident, variants }) => {
                let variant_expressions = variants.iter().map(|variant| {
                    let variant_ident = &variant.ident;
                    let variant_path = quote!(#ident::#variant_ident);
                    let fields = variant.fields.iter().map(|field| field.member(None));
                    let fields_body = generate_body(&variant.fields, None);
                    match &variant.kind {
                        Kind::Unit => quote!(#variant_path => #variant_path),
                        Kind::Unnamed => quote!(#variant_path (#(#fields,)*) => #variant_path ( #fields_body )),
                        Kind::Named => quote!(#variant_path { #(#fields,)* } => #variant_path { #fields_body }),
                    }
                });
                quote!(match self { #(#variant_expressions,)* })
            }
        };

        let ryft = &self.ryft_crate;
        quote!(fn parameter_structure(&self) -> Self::To<#ryft::Placeholder> { #body })
    }

    /// Generates implementations for the [`Parameterized::parameters`], [`Parameterized::parameters_mut`],
    /// [`Parameterized::into_parameters`], [`Parameterized::named_parameters`],
    /// [`Parameterized::named_parameters_mut`], and [`Parameterized::into_named_parameters`] functions for the [`Data`]
    /// owned by this [`CodeGenerator`]. Note that this function returns a [`TokenStream`] with the complete function
    /// declarations and implementations but without the surrounding `impl` block. For example, the generated code may
    /// look something like this (with `...` substituted for generated code, of course, and the `ryft` crate possibly
    /// substituted as well):
    ///
    /// ```ignore
    /// fn parameters(&self) -> Self::ParameterIterator<'_, P> { ... }
    /// fn parameters_mut(&mut self) -> Self::ParameterIteratorMut<'_, P> { ... }
    /// fn into_parameters(self) -> Self::ParameterIntoIterator<P> { ... }
    /// fn named_parameters(&self) -> Self::NamedParameterIterator<'_, P> { ... }
    /// fn named_parameters_mut(&mut self) -> Self::NamedParameterIteratorMut<'_, P> { ... }
    /// fn into_named_parameters(self) -> Self::NamedParameterIntoIterator<P> { ... }
    /// ```
    fn generate_parameters_functions(&self) -> TokenStream {
        /// Helper that generates the body of one iterator-returning traversal function. This includes
        /// both unnamed traversals (i.e., [`Parameterized::parameters`], [`Parameterized::parameters_mut`], and
        /// [`Parameterized::into_parameters`]) and named traversals (i.e., [`Parameterized::named_parameters`],
        /// [`Parameterized::named_parameters_mut`], and [`Parameterized::into_named_parameters`]). The resulting
        /// [`TokenStream`] contains an expression that evaluates to the appropriate iterator and should be used as
        /// the body of the owning function. It is the responsibility of the caller to surround this [`TokenStream`]
        /// with braces and to generate the function signature/declaration.
        ///
        /// # Parameters
        ///
        ///   * `generator` - [`CodeGenerator`] from within which this function is being called.
        ///   * `iter_type` - [`IterType`] that specifies which variant among [`Parameterized::parameters`],
        ///     [`Parameterized::parameters_mut`], [`Parameterized::into_parameters`],
        ///     [`Parameterized::named_parameters`], [`Parameterized::named_parameters_mut`],
        ///     and [`Parameterized::into_named_parameters`], to generate code for.
        fn generate_body(generator: &CodeGenerator, iter_type: &IterType) -> TokenStream {
            match &generator.data {
                Data::Struct(StructData { fields, .. }) => {
                    generate_body_for_fields(generator, fields, Some(quote!(self)), iter_type)
                }
                Data::Enum(EnumData { ident, variants }) => {
                    let ryft = &generator.ryft_crate;
                    let assoc_ty = iter_type.parameters_assoc_type_name();
                    let variant_parameters = variants.iter().map(|variant| {
                        let variant_ident = &variant.ident;
                        let variant_path = quote!(#ident::#variant_ident);
                        let iterator_ident = format_ident!("{}{}", &ident, assoc_ty);
                        let fields = variant.fields.iter().map(|field| field.member(None));
                        let fields_body = generate_body_for_fields(generator, &variant.fields, None, iter_type);
                        let variant_body = if iter_type.is_named() {
                            let variant_name = variant.ident.to_string();
                            quote! {
                                #iterator_ident::#variant_ident(#ryft::PathPrefixedParameterIterator {
                                    iterator: #fields_body,
                                    segment: #ryft::ParameterPathSegment::Variant(#variant_name),
                                })
                            }
                        } else {
                            quote!(#iterator_ident::#variant_ident(#fields_body))
                        };
                        match &variant.kind {
                            Kind::Unit => quote!(#variant_path => #variant_body),
                            Kind::Unnamed => quote!(#variant_path (#(#fields,)*) => #variant_body),
                            Kind::Named => quote!(#variant_path { #(#fields,)* } => #variant_body),
                        }
                    });
                    quote!(match self { #(#variant_parameters,)* })
                }
            }
        }

        /// Helper that generates the field-level body portion of one iterator-returning traversal function
        /// for the provided [`Field`]s. This includes both unnamed traversals (i.e., [`Parameterized::parameters`],
        /// [`Parameterized::parameters_mut`], and [`Parameterized::into_parameters`]) and named traversals
        /// (i.e., [`Parameterized::named_parameters`], [`Parameterized::named_parameters_mut`], and
        /// [`Parameterized::into_named_parameters`]). The resulting [`TokenStream`] contains an expression that
        /// evaluates to the appropriate iterator and that can be nested directly within the body of the owning
        /// function. It is the responsibility of the caller to surround this [`TokenStream`] with braces and/or
        /// parentheses and to generate the rest of the surrounding code.
        ///
        /// # Parameters
        ///
        ///   * `generator` - [`CodeGenerator`] from within which this function is being called.
        ///   * `fields` - List of [`Field`]s for which to generate code.
        ///   * `receiver` - Optional [`TokenStream`] that represents the "owner" of the provided [`Field`]s. Note that
        ///     the owner may be a tuple, a struct, or the variant of an enum. This is used to generate expressions that
        ///     access values of the provided [`Field`]s by invoking [`Field::member`].
        ///   * `iter_type` - [`IterType`] that specifies which variant among [`Parameterized::parameters`],
        ///     [`Parameterized::parameters_mut`], [`Parameterized::into_parameters`],
        ///     [`Parameterized::named_parameters`], [`Parameterized::named_parameters_mut`],
        ///     and [`Parameterized::into_named_parameters`], to generate code for.
        fn generate_body_for_fields(
            generator: &CodeGenerator,
            fields: &[Field],
            receiver: Option<TokenStream>,
            iter_type: &IterType,
        ) -> TokenStream {
            let ryft = &generator.ryft_crate;
            let parameter_type = &generator.parameter_type;
            fields
                .iter()
                .filter_map(|field| {
                    let receiver = field.member(receiver.as_ref());
                    let path_segment = match &field.ident {
                        Some(ident) => {
                            let name = ident.to_string();
                            quote!(#ryft::ParameterPathSegment::Field(#name))
                        }
                        None => {
                            let index = field.index;
                            quote!(#ryft::ParameterPathSegment::TupleIndex(#index))
                        }
                    };
                    match &field {
                        Field { is_parameter: false, .. } => None,
                        Field { fields: None, .. } => Some(match &iter_type {
                            IterType::Iter => quote!(#receiver.parameters()),
                            IterType::IterMut => quote!(#receiver.parameters_mut()),
                            IterType::IntoIter => quote!(#receiver.into_parameters()),
                            IterType::NamedIter => quote!(#ryft::PathPrefixedParameterIterator {
                                iterator: #receiver.named_parameters(),
                                segment: #path_segment,
                            }),
                            IterType::NamedIterMut => quote!(#ryft::PathPrefixedParameterIterator {
                                iterator: #receiver.named_parameters_mut(),
                                segment: #path_segment,
                            }),
                            IterType::NamedIntoIter => quote!(#ryft::PathPrefixedParameterIterator {
                                iterator: #receiver.into_named_parameters(),
                                segment: #path_segment,
                            }),
                        }),
                        Field { fields: Some(fields), .. } => {
                            let iterator = generate_body_for_fields(generator, fields, Some(receiver), iter_type);
                            if iter_type.is_named() {
                                Some(quote!(#ryft::PathPrefixedParameterIterator {
                                    iterator: #iterator,
                                    segment: #path_segment,
                                }))
                            } else {
                                Some(iterator)
                            }
                        }
                    }
                })
                .reduce(|chain_ty, ty| quote!(#chain_ty.chain(#ty)))
                .unwrap_or_else(|| match iter_type {
                    IterType::Iter | IterType::IterMut | IterType::IntoIter => quote!(std::iter::empty()),
                    IterType::NamedIter => quote!(std::iter::empty::<(#ryft::ParameterPath, &'_ #parameter_type)>()),
                    IterType::NamedIterMut => {
                        quote!(std::iter::empty::<(#ryft::ParameterPath, &'_ mut #parameter_type)>())
                    }
                    IterType::NamedIntoIter => quote!(std::iter::empty::<(#ryft::ParameterPath, #parameter_type)>()),
                })
        }

        let parameters_body = generate_body(self, &IterType::Iter);
        let parameters_mut_body = generate_body(self, &IterType::IterMut);
        let into_parameters_body = generate_body(self, &IterType::IntoIter);
        let named_parameters_body = generate_body(self, &IterType::NamedIter);
        let named_parameters_mut_body = generate_body(self, &IterType::NamedIterMut);
        let into_named_parameters_body = generate_body(self, &IterType::NamedIntoIter);

        let item_ty = &self.parameter_type;
        quote! {
            fn parameters(&self) -> Self::ParameterIterator<'_, #item_ty> {
                #parameters_body
            }
            
            fn parameters_mut(&mut self) -> Self::ParameterIteratorMut<'_, #item_ty> {
                #parameters_mut_body
            }
            
            fn into_parameters(self) -> Self::ParameterIntoIterator<#item_ty> {
                #into_parameters_body
            }
            
            fn named_parameters(&self) -> Self::NamedParameterIterator<'_, #item_ty> {
                #named_parameters_body
            }
            
            fn named_parameters_mut(&mut self) -> Self::NamedParameterIteratorMut<'_, #item_ty> {
                #named_parameters_mut_body
            }
            
            fn into_named_parameters(self) -> Self::NamedParameterIntoIterator<#item_ty> {
                #into_named_parameters_body
            }
        }
    }

    /// Generates an implementation for the [`Parameterized::from_parameters_with_remainder`] for the [`Data`] owned by
    /// this [`CodeGenerator`]. Note that this function returns a [`TokenStream`] with the complete function declaration
    /// and implementation but without the surrounding `impl` block. For example, the generated code may look something
    /// like this (with `...` substituted for generated code, of course, and the `ryft` crate possible substituted
    /// as well):
    ///
    /// ```ignore
    /// fn from_parameters_with_remainder<I: Iterator<Item = P>>(
    ///     structure: Self::To<ryft::Placeholder>,
    ///     parameters: &mut I,
    /// ) -> Result<Self, ryft::Error> {
    ///     let expected_count = structure.parameter_count();
    ///     Ok(...)
    /// }
    /// ```
    fn generate_from_parameters_with_remainder(&self) -> TokenStream {
        /// Helper that generates the body portion of the [`Parameterized::from_parameters_with_remainder`] function for
        /// the provided [`Field`]s. The resulting [`TokenStream`] contains a comma-separated list of expressions that
        /// compute the result of [`Parameterized::from_parameters_with_remainder`] for each of the provided [`Field`]s.
        /// It is the responsibility of the caller to surround this [`TokenStream`] with braces and/or parenthesis and
        /// to generate the rest of the surrounding code.
        ///
        /// # Parameters
        ///
        ///   * `generator` - [`CodeGenerator`] from within which this function is being called.
        ///   * `fields` - List of [`Field`]s for which to generate code.
        ///   * `structure` - Optional [`TokenStream`] that represents the [`Parameterized`] structure of the "owner" of
        ///     the provided [`Field`]s. Note that the owner may be a tuple, a struct, or the variant of an enum. This
        ///     is used to obtain the [`Parameterized`] structure that corresponds to each [`Field`] in the provided
        ///     list by invoking [`Field::member`] to construct the appropriate accessor.
        fn generate_body(generator: &CodeGenerator, fields: &[Field], structure: Option<TokenStream>) -> TokenStream {
            let ryft = &generator.ryft_crate;
            let parameter_type = &generator.parameter_type;

            // Note that `expected_count`, which is referenced by the generated code here is defined in the beginning
            // of the implementation of the generated [`Parameterized::from_parameters_with_remainder`] function body.
            let insufficient_parameters_error = quote!(#ryft::Error::InsufficientParameters { expected_count });

            fields
                .iter()
                .map(|field| {
                    let prefix = field.ident.as_ref().map(|ident| quote!(#ident:)).unwrap_or(TokenStream::new());
                    let structure = field.member(structure.as_ref());
                    match &field {
                        Field { is_parameter: false, .. } => quote!(#prefix #structure),
                        Field { ty, fields: None, .. } => quote! {
                            #prefix <#ty as #ryft::Parameterized<#parameter_type>>::from_parameters_with_remainder(
                                #structure,
                                parameters,
                            ).map_err(|error| match error {
                                #ryft::Error::InsufficientParameters { .. } => #insufficient_parameters_error,
                                error => error,
                            })?
                        },
                        Field { fields: Some(fields), .. } => {
                            let fields_expression = generate_body(generator, fields, Some(structure));
                            quote!(#prefix (#fields_expression))
                        }
                    }
                })
                .reduce(|token_stream, field_expression| quote!(#token_stream, #field_expression))
                .unwrap_or(TokenStream::new())
        }

        let body = match &self.data {
            Data::Struct(StructData { ident, fields, kind }) => {
                let fields = generate_body(self, fields, Some(quote!(structure)));
                match &kind {
                    Kind::Unit => quote!(#ident),
                    Kind::Unnamed => quote!(#ident ( #fields )),
                    Kind::Named => quote!(#ident { #fields }),
                }
            }
            Data::Enum(EnumData { ident, variants }) => {
                let variant_from_parameters_with_remainders = variants.iter().map(|variant| {
                    let variant_ident = &variant.ident;
                    let variant_path = quote!(#ident::#variant_ident);
                    let fields = variant.fields.iter().map(|field| field.member(None));
                    let result = generate_body(self, &variant.fields, None);
                    match &variant.kind {
                        Kind::Unit => quote!(#variant_path => #variant_path),
                        Kind::Unnamed => quote!(#variant_path (#(#fields,)*) => #variant_path ( #result )),
                        Kind::Named => quote!(#variant_path { #(#fields,)* } => #variant_path { #result }),
                    }
                });
                quote!(match structure { #(#variant_from_parameters_with_remainders,)* })
            }
        };

        let ryft = &self.ryft_crate;
        let parameter_type = &self.parameter_type;
        let self_to_as_parameterized = quote!(<Self::ParameterStructure as #ryft::Parameterized<#ryft::Placeholder>>);
        quote! {
            fn from_parameters_with_remainder<I: Iterator<Item = #parameter_type>>(
                structure: Self::To<#ryft::Placeholder>,
                parameters: &mut I,
            ) -> Result<Self, #ryft::Error> {
                let expected_count = #self_to_as_parameterized::parameter_count(&structure);
                Ok(#body)
            }
        }
    }

    /// Returns a clone of the extracted [`syn::Generics`] that is filtered down to only the set of
    /// [`syn::GenericParam`]s that are referenced by the provided [`Field`]s. This is useful for constructing
    /// [`syn::Generics`]s for types that e.g., only involve [`Parameterized`] [`Field`]s.
    fn generics_for_fields<'s, I: IntoIterator<Item = &'s Field>>(&'s self, fields: I) -> syn::Generics {
        let field_types = fields.into_iter().filter(|f| f.is_parameter).map(|f| &f.ty).collect::<Vec<_>>();
        let parameters_to_remove = self
            .generics
            .params
            .iter()
            .flat_map(|parameter| parameter.ident())
            .filter(|ident| !field_types.iter().any(|ty| ty.matches_ident(ident) || ty.references_ident(ident)));
        self.generics.without_params(parameters_to_remove)
    }
}

/// Parsed [`syn::Data`] from a container that was annotated with `#[derive(Parameterized)]`.
enum Data {
    Struct(StructData),
    Enum(EnumData),
}

impl Data {
    fn ident(&self) -> &syn::Ident {
        match &self {
            Data::Struct(StructData { ident, .. }) => ident,
            Data::Enum(EnumData { ident, .. }) => ident,
        }
    }
}

/// Parsed [`syn::Data::Struct`] from a container that was annotated with `#[derive(Parameterized)]`.
struct StructData {
    /// Identifier of the struct.
    ident: syn::Ident,

    /// [Field]s in the struct.
    fields: Vec<Field>,

    /// [Kind] of the struct that determines the appropriate syntax for constructing instances of it.
    kind: Kind,
}

/// Parsed [`syn::Data::Enum`] from a container that was annotated with `#[derive(Parameterized)]`.
struct EnumData {
    /// Identifier of the enum.
    ident: syn::Ident,
    variants: Vec<Variant>,
}

/// Parsed [`syn::Variant`] from a container that was annotated with `#[derive(Parameterized)]`.
struct Variant {
    /// Identifier of this [`Variant`].
    ident: syn::Ident,

    /// List of [`Field`]s in this [`Variant`].
    fields: Vec<Field>,

    /// [`Kind`] of this [`Variant`] that determines the appropriate syntax for constructing instances of it,
    /// and for pattern matching within match statements.
    kind: Kind,
}

/// Parsed [`syn::Field`] from a container that was annotated with `#[derive(Parameterized)]`.
struct Field {
    /// Indicates whether the [`syn::Type`] of this [`Field`] should be bounded by [`Parameterized`].
    /// This is determined by whether or not this [`Field`]'s type references [`CodeGenerator::parameter_type`].
    is_parameter: bool,

    /// Index of this [`Field`] in the container that it belongs.
    index: usize,

    /// Identifier of this [`Field`]. This is optional because [`Kind::Unnamed`] [`Field`]s do not have identifiers.
    ident: Option<syn::Ident>,

    /// [`syn::Type`] of this [`Field`].
    ty: syn::Type,

    /// Nested [`Field`]s of this [`Field`]. This is optional because it is only relevant for tuples. When a
    /// [`CodeGenerator`] synthesized [`Parameterized`] implementations it needs to recurse into nested tuple types
    /// as those types are not [`Parameterized`] by default (this is due to the fact that providing [`Parameterized`]
    /// implementations for all possible tuple types with arbitrary combinations of [`Parameter`] and non-[`Parameter`]
    /// fields is too expensive; the number of such combinations grows exponentially with the tuple size).
    ///
    /// In summary, the way to interpret this field is that if [`Field::ty`] is a [`syn::Type::Tuple`], then this field
    /// will be populated with the [`Field`]s extracted from that [`syn::Type::Tuple`].
    fields: Option<Vec<Field>>,
}

impl Field {
    /// Generates a [`TokenStream`] that represents an expression that accesses this [`Field`] on the value that
    /// corresponds to the provided "receiver". If `receiver` is [`None`], then it is assumed that this [`Field`]
    /// corresponds to the field of an enum [`Variant`] within a match statement, meaning that there is no
    /// receiver and the [`Field`]'s identifier or index can be used directly. If the `receiver` is not [`None`],
    /// then the [`Field`] is accessed using dot-notation (e.g., `receiver.field`). Note that when the field
    /// has an identifier, then its index is ignored. If it has no identifier and the `receiver` is not [`None`],
    /// then it is accessed as a tuple element using the index. If it has no identifier and the `receiver` is
    /// also none, then a special identifier is synthesized for it that looks like `field_{index}`. This only
    /// happens for [`Variant`] fields of [`Kind::Unnamed`] enums and the [`CodeGenerator`] ensures that the same
    /// identifier is used in the corresponding match statement branch
    /// (e.g., `match &self { Enum(field_0) => ... field_0 ...}`).
    fn member(&self, receiver: Option<&TokenStream>) -> TokenStream {
        let index = syn::Member::Unnamed(self.index.into());
        match (&self.ident, receiver) {
            (None, None) => format_ident!("field_{}", self.index).into_token_stream(),
            (Some(ident), None) => quote!(#ident),
            (Some(ident), Some(receiver)) => quote!(#receiver.#ident),
            (None, Some(receiver)) => quote!(#receiver.#index),
        }
    }
}

/// Enum used to describe whether a list of [`Field`]s corresponds to a [`syn::Fields::Named`],
/// [`syn::Fields::Unnamed`], or [`syn::Fields::Unit`].
enum Kind {
    Unit,
    Unnamed,
    Named,
}

/// Represents the type of a (typically) parameter iterator. This is used internally for generating code
/// related to parameter iterators which can iterate over immutable parameter references, mutable parameter
/// references, or owned parameters.
enum IterType {
    Iter,
    IterMut,
    IntoIter,
    NamedIter,
    NamedIterMut,
    NamedIntoIter,
}

impl IterType {
    /// Returns `true` if this [`IterType`] corresponds to a _named_ parameter iterator.
    fn is_named(&self) -> bool {
        matches!(self, IterType::NamedIter | IterType::NamedIterMut | IterType::NamedIntoIter)
    }

    /// Returns the name of the associated type of [`Parameterized`] that corresponds to this [`IterType`];
    /// namely, one of `ParameterIterator`, `ParameterIteratorMut`, `ParameterIntoIterator`, `NamedParameterIterator`,
    /// `NamedParameterIteratorMut`, and `NamedParameterIntoIterator`.
    fn parameters_assoc_type_name(&self) -> syn::Ident {
        match &self {
            IterType::Iter => syn::Ident::new("ParameterIterator", Span::call_site()),
            IterType::IterMut => syn::Ident::new("ParameterIteratorMut", Span::call_site()),
            IterType::IntoIter => syn::Ident::new("ParameterIntoIterator", Span::call_site()),
            IterType::NamedIter => syn::Ident::new("NamedParameterIterator", Span::call_site()),
            IterType::NamedIterMut => syn::Ident::new("NamedParameterIteratorMut", Span::call_site()),
            IterType::NamedIntoIter => syn::Ident::new("NamedParameterIntoIterator", Span::call_site()),
        }
    }
}
