use std::collections::HashSet;
use std::fmt::Display;

use proc_macro2::{Span, TokenStream};
use quote::{ToTokens, format_ident, quote, quote_spanned};
use syn::spanned::Spanned;

use crate::helpers::attributes::Attribute;
use crate::helpers::generics::GenericsHelpers;
use crate::helpers::hygiene::const_block;
use crate::helpers::idents::IdentHelpers;
use crate::helpers::receivers::replace_self_type;
use crate::helpers::symbols::Symbol;
use crate::helpers::tokens::TokensHelpers;

const RYFT_ATTRIBUTE: Symbol = Symbol::new("ryft");
const CRATE_ATTRIBUTE: Symbol = Symbol::new("crate");
const TYPE_ATTRIBUTE: Symbol = Symbol::new("type");
const CONSTANT_ATTRIBUTE: Symbol = Symbol::new("constant");
const MEMBERS_ATTRIBUTE: Symbol = Symbol::new("members");
const DISPATCH_ATTRIBUTE: Symbol = Symbol::new("dispatch");
const PROJECTED_ATTRIBUTE: Symbol = Symbol::new("projected");
const MIXED_ATTRIBUTE: Symbol = Symbol::new("mixed");
const COLLECTIVE_ATTRIBUTE: Symbol = Symbol::new("collective");
const LOCKSTEP_ATTRIBUTE: Symbol = Symbol::new("lockstep");
const SKIP_FROM_ATTRIBUTE: Symbol = Symbol::new("skip_from");
const STRUCTURAL_ROLE: Symbol = Symbol::new("structural");

const IDENTITY_DISPATCHER: Symbol = Symbol::new("identity");
const DISCHARGE_DISPATCHER: Symbol = Symbol::new("discharge");
const BATCHING_DISPATCHER: Symbol = Symbol::new("batching");
const DIFFERENTIATION_DISPATCHER: Symbol = Symbol::new("differentiation");
const TRANSPOSITION_DISPATCHER: Symbol = Symbol::new("transposition");

const DEFAULT_RYFT_CRATE: Symbol = Symbol::new("ryft");

/// [`OperationCodeGenerator`]s are used to generate the implementations of the `#[derive(Operation)]` macro. Each one
/// holds the normalized operation enum that [`OperationParser::extract_code_generator`] extracts from the provided
/// [`syn::DeriveInput`]. Refer to the documentation of the `Operation` trait in `ryft-core` for information on how to
/// use that macro.
///
/// Some of the generated implementations operate on programs whose operation family is this enum (e.g., interpretation
/// and batching) and are therefore _program-shaped_. Program-shaped implementations have a single value space, and so
/// they use the program-shaped generics, self type, and payload types, in which payload-specific value parameters are
/// substituted with the constant parameter. Refer to [`OperationParser::extract_code_generator`] for more information.
pub(crate) struct OperationCodeGenerator {
    /// [`syn::Path`] that represents the root `ryft` library path.
    /// Refer to [`OperationParser::ryft_crate`] for more information.
    ryft_crate: syn::Path,

    /// Primary operation type, which is the `Operation::Type` of the generated `Operation` implementation.
    operation_type: syn::Type,

    /// [`OperationMember`]s declared through `#[ryft(members(...))]`, in declaration order and with their member types
    /// in program-shaped form.
    members: Vec<OperationMember>,

    /// [`Dispatchers`] selected through `#[ryft(dispatch(...))]`.
    dispatchers: Dispatchers,

    /// Identifier of the enum.
    ident: syn::Ident,

    /// [`syn::Generics`] of the enum without defaults, which are not allowed in `impl` blocks. These are used by the
    /// implementations that are bounded by the payload types themselves (i.e., the payload conversions, the payload
    /// projection, and the identity implementations).
    generics: syn::Generics,

    /// [`syn::Generics`] used by the generated `Operation` implementation and by the implementations that require it.
    /// These extend [`OperationCodeGenerator::generics`] with one `Operation` predicate per distinct payload contract
    /// and with the type conversion predicates of every projected member type.
    operation_generics: syn::Generics,

    /// Program-shaped [`syn::Generics`] used by the interpretation, reference discharge, batching, and differentiation
    /// dispatchers. These omit the payload `Operation` predicates of [`OperationCodeGenerator::operation_generics`],
    /// because the per-payload rule predicates of these dispatchers imply them.
    dispatcher_generics: syn::Generics,

    /// Program-shaped form of [`OperationCodeGenerator::operation_generics`], used by the partial evaluation
    /// dispatcher.
    program_generics: syn::Generics,

    /// Type of the enum, using all of its generic parameters.
    self_type: syn::Type,

    /// Program-shaped form of [`OperationCodeGenerator::self_type`].
    program_self_type: syn::Type,

    /// Type of the constants that are stored in programs whose operation family is this enum.
    constant_type: syn::Type,

    /// Generic parameter named by [`OperationCodeGenerator::constant_type`], whose bounds generated runtime value types
    /// inherit. This is [`None`] for enums that declare an explicit constant type, because such a type has no
    /// one-to-one generic source.
    constant_parameter: Option<syn::Ident>,

    /// Boolean flag indicating whether the generated interpretation and transposition implementations use a runtime
    /// value type that is distinct from [`OperationCodeGenerator::constant_type`]. This is the case for enums that
    /// declare an explicit constant type and for enums with a single `Value<Type = T>` parameter.
    has_separate_runtime_value_type: bool,

    /// [`OperationVariant`]s of the enum, in declaration order.
    variants: Vec<OperationVariant>,
}

impl OperationCodeGenerator {
    /// Generates the `#[derive(Operation)]` implementations for the provided input. This function also checks for any
    /// errors on the provided input and will generate a [`TokenStream`] that contains a call to [`compile_error!`] if
    /// any errors are encountered. Refer to the documentation of the `Operation` trait in `ryft-core` for information
    /// on how to use this macro.
    ///
    /// As an example, the generated code for an enum with one native variant and no selected dispatchers typically
    /// looks something like this:
    ///
    /// ```ignore
    /// #[doc(hidden)]
    /// #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    /// const _: () = {
    ///     #[automatically_derived]
    ///     impl<V: Value<Type = T>> ::std::fmt::Display for CustomOperation<V> where ... {
    ///         ...
    ///     }
    ///
    ///     #[automatically_derived]
    ///     impl<V: Value<Type = T>> ryft::Operation for CustomOperation<V>
    ///     where
    ///         Payload<V>: ryft::Operation<Type = T>,
    ///     {
    ///         type Type = T;
    ///
    ///         fn name(&self) -> &'static str {
    ///             match self {
    ///                 Self::Payload(operation) => { <Payload<V> as ryft::Operation>::name(operation) },
    ///             }
    ///         }
    ///
    ///         ...
    ///     }
    ///
    ///     #[automatically_derived]
    ///     impl<V: Value<Type = T>, __Context> ryft::InterpretableOperation<__Context> for CustomOperation<V>
    ///     where
    ///         ...
    ///     {
    ///         ...
    ///     }
    ///
    ///     #[automatically_derived]
    ///     impl<V: Value<Type = T>, __Context> ryft::PartiallyEvaluatableOperation<__Context> for CustomOperation<V>
    ///     where
    ///         ...
    ///     {
    ///         ...
    ///     }
    ///
    ///     #[automatically_derived]
    ///     impl<V: Value<Type = T>> ryft::OperationPayloadProjection for CustomOperation<V> {
    ///         ...
    ///     }
    ///
    ///     #[automatically_derived]
    ///     impl<V: Value<Type = T>> ryft::OperationProvider<T, ryft::GuardOperation<T>> for CustomOperation<V>
    ///     where
    ///         ...
    ///     {
    ///         ...
    ///     }
    ///
    ///     #[automatically_derived]
    ///     impl<V: Value<Type = T>> ::std::convert::From<Payload<V>> for CustomOperation<V> {
    ///         ...
    ///     }
    ///
    ///     #[automatically_derived]
    ///     impl<'__operation, V: Value<Type = T>> ::std::convert::TryFrom<&'__operation CustomOperation<V>>
    ///         for &'__operation Payload<V>
    ///     where
    ///         CustomOperation<V>: ryft::Operation,
    ///     {
    ///         ...
    ///     }
    /// };
    /// ```
    ///
    /// Projected member variants additionally generate `OperationProjection` implementations, and every dispatcher that
    /// is selected through `#[ryft(dispatch(...))]` generates its implementation inside the same `const` block.
    pub(crate) fn generate_operation_impl(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
        let mut input = syn::parse_macro_input!(input as syn::DeriveInput);

        // Replace any instances of `Self` with its fully-qualified path. This is necessary in order to be able to
        // handle recursive payloads (e.g., control flow operations whose regions are programs over the enclosing enum).
        replace_self_type(&mut input);

        // Variant markers depend on the enum-level attributes (e.g., a `mixed` marker may take its member type from the
        // declared member universes), and so we only extract the variants once the enum-level attributes are valid.
        // That avoids reporting misleading follow-on errors for the variants.
        let mut parser = OperationParser::new();
        parser.extract_attributes(&input);
        if let Some(error) = parser.compile_error() {
            return error.into();
        }
        let generator = parser.extract_code_generator(&input);
        if let Some(error) = parser.compile_error() {
            return error.into();
        }
        generator.generate().into()
    }

    /// Generates all implementations for this [`OperationCodeGenerator`], including the dispatchers that are selected
    /// through `#[ryft(dispatch(...))]`, wrapped in a single `const` block. The implementations are generated in the
    /// order in which `ryft` places the implementations of an operation: the formatting and identity implementations,
    /// followed by `Operation`, `ReferenceDischargeableOperation`, `InterpretableOperation`,
    /// `PartiallyEvaluatableOperation`, `BatchableOperation`, `DifferentiableOperation`, `TransposableOperation`,
    /// `OperationProjection`, `OperationPayloadProjection`, `CollectiveOperation`, `OperationProvider`, and finally
    /// the payload conversions. Refer to [`OperationCodeGenerator::generate_operation_impl`] for an example of the
    /// generated code.
    fn generate(&self) -> TokenStream {
        let ryft = &self.ryft_crate;
        let self_type = &self.self_type;
        let (impl_generics, _, where_clause) = self.operation_generics.split_for_impl();
        let identity = self.dispatchers.identity.then(|| self.generate_identity_impls());
        let operation = self.generate_operation_dispatcher();
        let discharge = self.dispatchers.discharge.then(|| self.generate_reference_discharge_dispatcher());
        let interpretation = self.generate_interpretation_dispatcher();
        let partial_evaluation = self.generate_partial_evaluation_dispatcher();
        let batching = self.dispatchers.batching.then(|| self.generate_batching_dispatcher());
        let differentiation = self.dispatchers.differentiation.then(|| self.generate_differentiation_dispatcher());
        let transposition = self.dispatchers.transposition.then(|| self.generate_transposition_dispatcher());
        let operation_projections =
            self.variants.iter().filter_map(|variant| self.generate_operation_projection_impl(variant));
        let payload_projection = self.generate_payload_projection_impl();
        let collective = self.generate_collective_impl();
        let lockstep = self.generate_lockstep_impl();
        let guard_provider = self.generate_guard_provider_impl();
        let conversions = self
            .variants
            .iter()
            .filter(|variant| !variant.is_generic_extension)
            .map(|variant| self.generate_conversion_impls(variant));
        const_block(quote! {
            #[automatically_derived]
            impl #impl_generics ::std::fmt::Display for #self_type #where_clause {
                fn fmt(&self, formatter: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                    <Self as #ryft::Operation>::render(self, formatter, 0)
                }
            }

            #identity

            #operation

            #discharge

            #interpretation

            #partial_evaluation

            #batching

            #differentiation

            #transposition

            #(#operation_projections)*

            #payload_projection

            #collective

            #lockstep

            #guard_provider

            #(#conversions)*
        })
    }

    /// Generates the `#[ryft(dispatch(identity))]` implementations, which are [`PartialEq`], [`Eq`], and [`Hash`]
    /// implementations that delegate to the payload of each variant. Unlike the standard derives, which bound every
    /// generic parameter by the derived trait, these implementations are bounded by the payload types themselves, so
    /// that a family is comparable exactly when its payloads are (e.g., a payload that stores a literal may require a
    /// bound other than [`PartialEq`] on the value parameter of the family). Two operations are equal when they are the
    /// same variant with equal payloads, and the hash covers the variant discriminant and the payload.
    fn generate_identity_impls(&self) -> TokenStream {
        let self_type = &self.self_type;
        let mut payload_type_names = HashSet::new();
        let payload_variants = self
            .variants
            .iter()
            .filter(|variant| payload_type_names.insert(variant.payload_type.to_compact_string()))
            .collect::<Vec<_>>();
        let bounded_generics = |bound: TokenStream| {
            let mut generics = self.generics.clone();
            let where_clause = generics.make_where_clause();
            for variant in &payload_variants {
                let payload_type = &variant.payload_type;
                where_clause.predicates.push(syn::parse_quote_spanned!(variant.span => #payload_type: #bound));
            }
            generics
        };
        let partial_eq_generics = bounded_generics(quote!(::std::cmp::PartialEq));
        let eq_generics = bounded_generics(quote!(::std::cmp::Eq));
        let hash_generics = bounded_generics(quote!(::std::hash::Hash));
        let (partial_eq_impl_generics, _, partial_eq_where_clause) = partial_eq_generics.split_for_impl();
        let (eq_impl_generics, _, eq_where_clause) = eq_generics.split_for_impl();
        let (hash_impl_generics, _, hash_where_clause) = hash_generics.split_for_impl();
        let variant_idents = self.variants.iter().map(|variant| &variant.ident).collect::<Vec<_>>();
        let mismatched_variants_arm = (self.variants.len() > 1).then(|| quote!(_ => false,));
        quote! {
            #[automatically_derived]
            impl #partial_eq_impl_generics ::std::cmp::PartialEq for #self_type #partial_eq_where_clause {
                #[inline]
                fn eq(&self, other: &Self) -> bool {
                    match (self, other) {
                        #((Self::#variant_idents(left), Self::#variant_idents(right)) => left == right,)*
                        #mismatched_variants_arm
                    }
                }
            }

            #[automatically_derived]
            impl #eq_impl_generics ::std::cmp::Eq for #self_type #eq_where_clause {}

            #[automatically_derived]
            impl #hash_impl_generics ::std::hash::Hash for #self_type #hash_where_clause {
                #[inline]
                fn hash<__H: ::std::hash::Hasher>(&self, state: &mut __H) {
                    ::std::hash::Hash::hash(&::std::mem::discriminant(self), state);
                    match self {
                        #(Self::#variant_idents(operation) => ::std::hash::Hash::hash(operation, state),)*
                    }
                }
            }
        }
    }

    /// Generates the `Operation` implementation, whose functions dispatch to the payload of the active variant.
    /// Functions whose results depend on the instruction boundary (e.g., output type inference) use the contract of
    /// the universe that the payload of each variant defines. That is, family-native payloads are called directly,
    /// projected member payloads are called through the `ryft` support functions that project the boundary into their
    /// member type and lift the results back, and mixed member payloads are called through `MemberOperation`.
    /// All other functions (e.g., rendering) are forwarded to the payload of every variant directly.
    fn generate_operation_dispatcher(&self) -> TokenStream {
        let ryft = &self.ryft_crate;
        let operation_type = &self.operation_type;
        let self_type = &self.self_type;
        let (impl_generics, _, where_clause) = self.operation_generics.split_for_impl();

        // Functions that see the instruction boundary are called through the contract of the universe that the payload
        // of each variant defines. Mixed member payloads are called through the provided `MemberOperation` function and
        // projected member payloads through the provided `ryft` support function, when these are provided. All other
        // payloads, and all payloads of functions that are simply forwarded, are called through `Operation` directly.
        let call = |variant: &OperationVariant,
                    function: &str,
                    mixed_function: Option<&str>,
                    projected_function: Option<&str>,
                    arguments: &TokenStream| {
            let payload_type = &variant.payload_type;
            let receiver = variant.receiver();
            let arguments = if arguments.is_empty() { quote!() } else { quote!(, #arguments) };
            match (&variant.class, mixed_function, projected_function) {
                (OperationVariantClass::MixedMember { .. }, Some(mixed_function), _) => {
                    let function = format_ident!("{mixed_function}");
                    quote!(<#payload_type as #ryft::MemberOperation<#operation_type>>::#function(#receiver #arguments))
                }
                (OperationVariantClass::ProjectedMember { .. }, _, Some(projected_function)) => {
                    let function = format_ident!("{projected_function}");
                    quote!(#ryft::#function(#receiver #arguments))
                }
                _ => {
                    let function = format_ident!("{function}");
                    quote!(<#payload_type as #ryft::Operation>::#function(#receiver #arguments))
                }
            }
        };
        let arms =
            |function: &str, mixed_function: Option<&str>, projected_function: Option<&str>, arguments: TokenStream| {
                self.variants
                    .iter()
                    .map(|variant| variant.arm(call(variant, function, mixed_function, projected_function, &arguments)))
                    .collect::<Vec<_>>()
            };
        let forwarding_arms = |function: &str, arguments: TokenStream| arms(function, None, None, arguments);

        let name_arms = forwarding_arms("name", quote!());
        let defined_type_identity_arms =
            arms("defined_type_identities", Some("parent_defined_type_identities"), None, quote!());
        let region_slot_arms = forwarding_arms("region_slots", quote!());
        let region_interface_validation_arms = arms(
            "validate_region_interfaces",
            Some("validate_parent_region_interfaces"),
            Some("validate_projected_operation_region_interfaces"),
            quote!(region_interfaces),
        );
        let region_input_type_arms = arms(
            "infer_region_input_types",
            Some("infer_parent_region_input_types"),
            Some("infer_projected_operation_region_input_types"),
            quote!(input_types, region_interfaces),
        );
        let output_type_arms = arms(
            "infer_output_types",
            Some("infer_parent_output_types"),
            Some("infer_projected_operation_output_types"),
            quote!(input_types, region_interfaces),
        );
        let fold_arms =
            arms("fold", Some("fold_parent"), Some("fold_projected_operation"), quote!(input_types, region_interfaces));
        let input_region_provenance_arms =
            forwarding_arms("input_region_provenance", quote!(region_index, input_index));
        let output_region_provenance_arms = forwarding_arms("output_region_provenance", quote!(output_index));
        let region_data_flow_arms = forwarding_arms("region_data_flow", quote!());
        let prune_boundary_arms = self.variants.iter().map(|variant| {
            let pruning = call(
                variant,
                "prune_boundary",
                Some("prune_parent_boundary"),
                Some("prune_projected_operation_boundary"),
                &quote!(input_types, used_outputs, regions),
            );
            let operation = variant.construct(quote!(operation));
            variant.arm(quote! {
                ::std::result::Result::Ok(#pruning?.map(|pruning| pruning.map_operation(|operation| #operation)))
            })
        });
        let is_zero_arms = forwarding_arms("is_zero", quote!(output_index));
        let region_capture_input_count_arms = forwarding_arms("region_capture_input_count", quote!(region_index));
        let reference_output_identity_input_arms =
            forwarding_arms("reference_output_identity_input", quote!(output_index));
        let allows_reference_access_through_region_input_arms =
            forwarding_arms("allows_reference_access_through_region_input", quote!(region_index, mode));
        let defers_reference_discharge_arms = forwarding_arms("defers_reference_discharge", quote!());
        let effects_arms = forwarding_arms("effects", quote!());
        let type_identity_renaming_arms = self.variants.iter().map(|variant| {
            let renaming =
                call(variant, "rename_type_identities", Some("rename_parent_type_identities"), None, &quote!(renaming));
            let operation = variant.construct(quote!(#renaming?));
            variant.arm(quote!(::std::result::Result::Ok(#operation)))
        });
        let render_arms = forwarding_arms("render", quote!(formatter, indentation));

        quote! {
            #[automatically_derived]
            impl #impl_generics #ryft::Operation for #self_type #where_clause {
                type Type = #operation_type;

                #[inline]
                fn name(&self) -> &'static str {
                    match self { #(#name_arms)* }
                }

                #[inline]
                fn defined_type_identities(
                    &self,
                ) -> ::std::borrow::Cow<'_, [<#operation_type as #ryft::Type>::Identity]> {
                    match self { #(#defined_type_identity_arms)* }
                }

                #[inline]
                fn region_slots(&self) -> &'static [#ryft::RegionSlot] {
                    match self { #(#region_slot_arms)* }
                }

                #[inline]
                fn validate_region_interfaces(
                    &self,
                    region_interfaces: &[#ryft::RegionInterface<#operation_type>],
                ) -> ::std::result::Result<(), #ryft::ProgramError> {
                    match self { #(#region_interface_validation_arms)* }
                }

                #[inline]
                fn infer_region_input_types(
                    &self,
                    input_types: &[#operation_type],
                    region_interfaces: &[#ryft::RegionInterface<#operation_type>],
                ) -> ::std::result::Result<
                    ::std::vec::Vec<::std::option::Option<::std::vec::Vec<#operation_type>>>,
                    #ryft::TypeError,
                > {
                    match self { #(#region_input_type_arms)* }
                }

                #[inline]
                fn infer_output_types(
                    &self,
                    input_types: &[#operation_type],
                    region_interfaces: &[#ryft::RegionInterface<#operation_type>],
                ) -> ::std::result::Result<::std::vec::Vec<#operation_type>, #ryft::TypeError> {
                    match self { #(#output_type_arms)* }
                }

                #[inline]
                fn fold(
                    &self,
                    input_types: &[#operation_type],
                    region_interfaces: &[#ryft::RegionInterface<#operation_type>],
                ) -> ::std::result::Result<
                    ::std::option::Option<::std::vec::Vec<#ryft::OperationFoldOutput>>,
                    #ryft::TypeError,
                > {
                    match self { #(#fold_arms)* }
                }

                #[inline]
                fn input_region_provenance(
                    &self,
                    region_index: usize,
                    input_index: usize,
                ) -> #ryft::InputRegionProvenance {
                    match self { #(#input_region_provenance_arms)* }
                }

                #[inline]
                fn output_region_provenance(
                    &self,
                    output_index: usize,
                ) -> ::std::vec::Vec<#ryft::OutputRegionProvenance> {
                    match self { #(#output_region_provenance_arms)* }
                }

                #[inline]
                fn region_data_flow(&self) -> #ryft::RegionDataFlow<'_> {
                    match self { #(#region_data_flow_arms)* }
                }

                #[inline]
                fn prune_boundary(
                    &self,
                    input_types: &[#operation_type],
                    used_outputs: &[bool],
                    regions: &mut dyn #ryft::RegionLiveness,
                ) -> ::std::result::Result<
                    ::std::option::Option<#ryft::OperationBoundaryPruning<Self>>,
                    #ryft::ProgramError,
                > {
                    match self { #(#prune_boundary_arms)* }
                }

                #[inline]
                fn is_zero(&self, output_index: usize) -> bool {
                    match self { #(#is_zero_arms)* }
                }

                #[inline]
                fn region_capture_input_count(&self, region_index: usize) -> ::std::option::Option<usize> {
                    match self { #(#region_capture_input_count_arms)* }
                }

                #[inline]
                fn reference_output_identity_input(&self, output_index: usize) -> ::std::option::Option<usize> {
                    match self { #(#reference_output_identity_input_arms)* }
                }

                #[inline]
                fn allows_reference_access_through_region_input(
                    &self,
                    region_index: usize,
                    mode: #ryft::ReferenceAccessMode,
                ) -> bool {
                    match self { #(#allows_reference_access_through_region_input_arms)* }
                }

                #[inline]
                fn defers_reference_discharge(&self) -> bool {
                    match self { #(#defers_reference_discharge_arms)* }
                }

                #[inline]
                fn effects(&self) -> ::std::borrow::Cow<'_, #ryft::Effects> {
                    match self { #(#effects_arms)* }
                }

                #[inline]
                fn rename_type_identities(
                    &self,
                    renaming: &#ryft::TypeIdentityRenaming<<#operation_type as #ryft::Type>::Identity>,
                ) -> ::std::result::Result<Self, #ryft::TypeError> {
                    match self { #(#type_identity_renaming_arms)* }
                }

                #[inline]
                fn render(
                    &self,
                    formatter: &mut ::std::fmt::Formatter<'_>,
                    indentation: usize,
                ) -> ::std::fmt::Result {
                    match self { #(#render_arms)* }
                }
            }
        }
    }

    /// Generates the `ReferenceDischargeableOperation` dispatcher. The reference discharge policy stays generic for
    /// every family, because a policy names the reference universe that is being discharged rather than the universe
    /// that the values of the family belong to. Family-native and bare generic extension variants delegate to the
    /// discharge rules of their payloads, while member variants replay the complete enum through
    /// `ReferenceDischargeContext::discharge_reference_free_operation`, which copies any attached
    /// regions unchanged and rejects reference inputs and region closures that reach references.
    fn generate_reference_discharge_dispatcher(&self) -> TokenStream {
        let ryft = &self.ryft_crate;
        let operation_type = &self.operation_type;
        let self_type = &self.program_self_type;
        let delegates = |variant: &OperationVariant| {
            variant.is_generic_extension || matches!(variant.class, OperationVariantClass::FamilyNative)
        };
        let mut generics = self.dispatcher_generics.clone();
        generics.params.push(syn::parse_quote!(__ParentContext));
        generics.params.push(syn::parse_quote!(__ReferenceDischargePolicy));
        let where_clause = generics.make_where_clause();

        // The constant type of the parent context is deliberately left free. No rule reads it, and pinning it to the
        // constant type of the family would exclude capture-lifted programs, whose constants name the captures of their
        // caller instead of carrying the values of the family.
        where_clause.predicates.push(syn::parse_quote! {
            __ParentContext: #ryft::Context<Type = #operation_type, Operation = #self_type>
        });
        where_clause
            .predicates
            .push(syn::parse_quote!(#self_type: #ryft::Operation<Type = #operation_type>));
        where_clause.predicates.push(syn::parse_quote! {
            __ReferenceDischargePolicy: #ryft::ReferenceDischargePolicy<__ParentContext>
        });
        where_clause.predicates.extend(self.variants.iter().filter(|variant| delegates(variant)).map(
            |variant| -> syn::WherePredicate {
                let payload_type = &variant.program_payload_type;
                syn::parse_quote_spanned! { variant.span =>
                    #payload_type: #ryft::ReferenceDischargeableOperation<__ParentContext, __ReferenceDischargePolicy>
                }
            },
        ));
        let arms = self.variants.iter().map(|variant| {
            if delegates(variant) {
                let payload_type = &variant.program_payload_type;
                let receiver = variant.receiver();
                variant.arm(quote! {
                    <#payload_type as #ryft::ReferenceDischargeableOperation<
                        __ParentContext,
                        __ReferenceDischargePolicy,
                    >>::discharge_references(#receiver, context, driver, inputs)
                })
            } else {
                variant.wildcard_arm(quote!(context.discharge_reference_free_operation(self, driver, inputs)))
            }
        });
        let (impl_generics, _, where_clause) = generics.split_for_impl();
        quote! {
            #[automatically_derived]
            impl #impl_generics #ryft::ReferenceDischargeableOperation<__ParentContext, __ReferenceDischargePolicy>
                for #self_type
            #where_clause
            {
                #[inline]
                fn discharge_references<
                    __D: #ryft::ReferenceDischargeDriver<__ParentContext, __ReferenceDischargePolicy>,
                >(
                    &self,
                    context: &#ryft::ReferenceDischargeContext<__ParentContext, __ReferenceDischargePolicy>,
                    driver: &__D,
                    inputs: &[#ryft::ReferenceDischargeValue<__ParentContext, __ReferenceDischargePolicy>],
                ) -> ::std::result::Result<
                    ::std::vec::Vec<#ryft::ReferenceDischargeValue<__ParentContext, __ReferenceDischargePolicy>>,
                    #ryft::ProgramError,
                > {
                    match self { #(#arms)* }
                }
            }
        }
    }

    /// Generates the `InterpretableOperation` dispatcher, which is generic over the interpretation context.
    /// Family-native variants delegate to the interpretation rules of their payloads and mixed member variants
    /// delegate to the parent-universe rules of their payloads. Projected member variants are interpreted through
    /// `interpret_projected_operation`, which projects their inputs into an eager context of their member universe
    /// and lifts their outputs back.
    ///
    /// The context value type is the constant type for enums with two or more `Value<Type = T>` parameters. Otherwise,
    /// it is a distinct runtime value type, named by a generated `__InterpretationValue` type parameter that inherits
    /// the bounds of the constant parameter, and the context must be able to lift constants into it.
    fn generate_interpretation_dispatcher(&self) -> TokenStream {
        let ryft = &self.ryft_crate;
        let operation_type = &self.operation_type;
        let constant_type = &self.constant_type;
        let self_type = &self.program_self_type;
        let context_value_type: syn::Type = syn::parse_quote!(<__Context as #ryft::Domain>::Value);
        let mut generics = self.dispatcher_generics.clone();
        generics.params.push(syn::parse_quote!(__Context));

        // A separate runtime value type is named by a generated type parameter rather than by the value projection of
        // the context, so that the bounds of the constant parameter can be renamed onto it, including bounds on its
        // associated types (e.g., `V::Item: Bound`), which cannot be expressed on a projection.
        let value_type: syn::Type = if self.has_separate_runtime_value_type {
            generics.params.push(syn::parse_quote!(__InterpretationValue));
            syn::parse_quote!(__InterpretationValue)
        } else {
            context_value_type.clone()
        };
        let where_clause = generics.make_where_clause();
        if self.has_separate_runtime_value_type {
            where_clause.predicates.push(syn::parse_quote! {
                __Context: #ryft::Domain<Type = #operation_type, Value = #value_type>
            });
            where_clause
                .predicates
                .extend(self.constant_parameter_predicates(&format_ident!("__InterpretationValue")));
            where_clause.predicates.push(syn::parse_quote! {
                __Context: #ryft::Constant<#value_type, #constant_type>
            });
        } else {
            where_clause.predicates.push(syn::parse_quote! {
                __Context: #ryft::Domain<Type = #operation_type, Value = #constant_type>
            });
        }
        for variant in &self.variants {
            let payload_type = &variant.program_payload_type;
            match &variant.class {
                OperationVariantClass::FamilyNative => {
                    where_clause.predicates.push(syn::parse_quote_spanned! { variant.span =>
                        #payload_type: #ryft::InterpretableOperation<__Context>
                    })
                }
                OperationVariantClass::MixedMember { .. } => {
                    where_clause.predicates.push(syn::parse_quote_spanned! { variant.span =>
                        #payload_type: #ryft::MemberInterpretableOperation<__Context>
                    })
                }
                OperationVariantClass::ProjectedMember { member_type, .. } => {
                    let member_value_type: syn::Type = syn::parse_quote! {
                        <#value_type as #ryft::ValueProjection<#member_type>>::Projected
                    };
                    where_clause.predicates.push(syn::parse_quote! {
                        #value_type: #ryft::ValueProjection<#member_type, Projected: #ryft::Value<Type = #member_type>>
                    });

                    // The eager context of the member universe is built from the payload itself, and so we state its
                    // domain as an assumption. The `Operation<Type = C::Type>` super-trait of the payload rule
                    // predicate then normalizes through the parameter environment instead of recursing through the
                    // bounds of the payload.
                    where_clause.predicates.push(syn::parse_quote! {
                        #ryft::EagerContext<#member_value_type, #payload_type>: #ryft::Domain<
                            Type = #member_type,
                            Value = #member_value_type,
                            Constant = #member_value_type,
                            Operation = #payload_type,
                        >
                    });
                    where_clause.predicates.push(syn::parse_quote_spanned! { variant.span =>
                        #payload_type: #ryft::InterpretableOperation<
                            #ryft::EagerContext<#member_value_type, #payload_type>,
                        >
                    });
                }
            }
        }
        let arms = self.variants.iter().map(|variant| {
            let payload_type = &variant.program_payload_type;
            let receiver = variant.receiver();
            variant.arm(match &variant.class {
                OperationVariantClass::FamilyNative => quote! {
                    <#payload_type as #ryft::InterpretableOperation<__Context>>::
                        interpret(#receiver, context, driver, inputs)
                },
                OperationVariantClass::MixedMember { .. } => quote! {
                    <#payload_type as #ryft::MemberInterpretableOperation<__Context>>::
                        interpret_in_parent(#receiver, context, driver, inputs)
                },
                OperationVariantClass::ProjectedMember { .. } => quote! {
                    #ryft::interpret_projected_operation(context, #receiver, driver, inputs)
                },
            })
        });
        let (impl_generics, _, where_clause) = generics.split_for_impl();
        quote! {
            #[automatically_derived]
            impl #impl_generics #ryft::InterpretableOperation<__Context> for #self_type #where_clause {
                #[inline]
                fn interpret<__D: #ryft::InterpretationDriver<__Context>>(
                    &self,
                    context: &__Context,
                    driver: &__D,
                    inputs: &[#context_value_type],
                ) -> ::std::result::Result<::std::vec::Vec<#context_value_type>, #ryft::ProgramError> {
                    match self { #(#arms)* }
                }
            }
        }
    }

    /// Generates the `PartiallyEvaluatableOperation` dispatcher, which is generic over the partial evaluation context.
    /// Family-native variants delegate to the partial evaluation rules of their payloads. Bare generic extension and
    /// member variants instead use the canonical fold-or-residualize path of the complete enum, because a member
    /// payload's own partial evaluation rule belongs to its native context and cannot consume the partial values
    /// of the enclosing family without a separate parent-boundary contract.
    fn generate_partial_evaluation_dispatcher(&self) -> TokenStream {
        let ryft = &self.ryft_crate;
        let operation_type = &self.operation_type;
        let constant_type = &self.constant_type;
        let self_type = &self.program_self_type;
        let value_type: syn::Type =
            syn::parse_quote!(#ryft::PartialEvaluationValue<<__Context as #ryft::Domain>::Value>);
        let delegates = |variant: &OperationVariant| {
            !variant.is_generic_extension && matches!(variant.class, OperationVariantClass::FamilyNative)
        };
        let mut generics = self.program_generics.clone();
        generics.params.push(syn::parse_quote!(__Context));
        let where_clause = generics.make_where_clause();
        where_clause.predicates.push(syn::parse_quote! {
            __Context: #ryft::Context<Type = #operation_type, Constant = #constant_type, Operation = #self_type>
        });
        where_clause
            .predicates
            .extend(self.variants.iter().filter(|variant| delegates(variant)).map(|variant| -> syn::WherePredicate {
            let payload_type = &variant.program_payload_type;
            syn::parse_quote_spanned!(variant.span => #payload_type: #ryft::PartiallyEvaluatableOperation<__Context>)
        }));
        let arms = self.variants.iter().map(|variant| {
            if delegates(variant) {
                let payload_type = &variant.program_payload_type;
                let receiver = variant.receiver();
                variant.arm(quote! {
                    <#payload_type as #ryft::PartiallyEvaluatableOperation<__Context>>::
                        partially_evaluate(#receiver, context, driver, inputs)
                })
            } else {
                variant.wildcard_arm(quote!(context.fold_or_residualize(
                    self.clone(),
                    driver.region_programs(),
                    inputs
                )))
            }
        });

        // Splits on residual demand follow the same delegation as partial evaluation, and every other variant keeps
        // its applications whole. The bound of the trait function is repeated here, so that the dispatcher states it
        // on the function rather than on the implementation (which would make the enum's partial evaluation depend on
        // itself).
        let split_arms = self.variants.iter().map(|variant| {
            if delegates(variant) {
                let payload_type = &variant.program_payload_type;
                let receiver = variant.receiver();
                variant.arm(quote! {
                    <#payload_type as #ryft::PartiallyEvaluatableOperation<__Context>>::
                        split_on_residual_demand(#receiver, context, driver, inputs, demanded_outputs)
                })
            } else {
                variant.wildcard_arm(quote!(::std::result::Result::Ok(::std::option::Option::None)))
            }
        });
        let (impl_generics, _, where_clause) = generics.split_for_impl();
        quote! {
            #[automatically_derived]
            impl #impl_generics #ryft::PartiallyEvaluatableOperation<__Context> for #self_type #where_clause {
                #[inline]
                fn partially_evaluate<__D: #ryft::PartialEvaluationDriver<__Context>>(
                    &self,
                    context: &#ryft::PartialEvaluationContext<__Context>,
                    driver: &__D,
                    inputs: &[#value_type],
                ) -> ::std::result::Result<::std::vec::Vec<#value_type>, #ryft::ProgramError> {
                    match self { #(#arms)* }
                }

                #[inline]
                fn split_on_residual_demand<__D: #ryft::PartialEvaluationDriver<__Context>>(
                    &self,
                    context: &#ryft::PartialEvaluationContext<__Context>,
                    driver: &__D,
                    inputs: &[#value_type],
                    demanded_outputs: &[bool],
                ) -> ::std::result::Result<
                    ::std::option::Option<
                        #ryft::ResidualDemandSplit<<__Context as #ryft::Domain>::Value>
                    >,
                    #ryft::ProgramError,
                >
                where
                    <__Context as #ryft::Domain>::Operation: #ryft::PartiallyEvaluatableOperation<
                        #ryft::TracingContext<
                            <__Context as #ryft::Domain>::Constant,
                            <__Context as #ryft::Domain>::Operation,
                        >,
                    >,
                {
                    // Enums without delegating variants read none of the parameters.
                    let _ = (context, driver, inputs, demanded_outputs);
                    match self { #(#split_arms)* }
                }
            }
        }
    }

    /// Generates the `BatchableOperation` dispatcher. Enums whose constant type is inferred stay generic over their
    /// batching policy, which lets array families support every array batching mode that their payload rules support,
    /// while enums with an explicit constant type use the canonical batching policy that `BatchableType` selects for
    /// their primary operation type. Family-native variants delegate to the batching rules of their payloads, and mixed
    /// member variants delegate to the parent-universe rules of their payloads. Computational projected member variants
    /// are batched through `BatchingContext::batch_projected_operation`, which applies the member batching rule of the
    /// payload under the projected batching policy. Structural projected member variants are instead bound once in the
    /// parent context through `BatchingContext::batch_replicated_projected_operation`, which requires the batching
    /// policy to implement `ReplicatedBatchingPolicyProjection` for their member type.
    fn generate_batching_dispatcher(&self) -> TokenStream {
        let ryft = &self.ryft_crate;
        let operation_type = &self.operation_type;
        let constant_type = &self.constant_type;
        let self_type = &self.program_self_type;
        let has_explicit_constant_type = self.constant_parameter.is_none();
        let parent_value_type: syn::Type = syn::parse_quote!(<__ParentContext as #ryft::Domain>::Value);
        let policy: syn::Type = syn::parse_quote!(__BatchingPolicy);
        let batch_type: syn::Type = syn::parse_quote!(<#policy as #ryft::BatchingPolicy<__ParentContext>>::Batch);
        let mut generics = self.dispatcher_generics.clone();
        generics.params.push(syn::parse_quote!(__ParentContext));
        let where_clause = generics.make_where_clause();
        where_clause.predicates.push(syn::parse_quote! {
            __ParentContext: #ryft::Context<
                Type = #operation_type,
                Constant = #constant_type,
                Operation = #self_type,
            >
        });
        where_clause
            .predicates
            .push(syn::parse_quote!(#self_type: #ryft::Operation<Type = #operation_type>));
        if has_explicit_constant_type {
            where_clause
                .predicates
                .push(syn::parse_quote!(#operation_type: #ryft::BatchableType<Policy = #policy>));
        }
        where_clause.predicates.push(syn::parse_quote!(#policy: #ryft::BatchingPolicy<__ParentContext>));

        for variant in &self.variants {
            let payload_type = &variant.program_payload_type;
            match &variant.class {
                OperationVariantClass::FamilyNative => {
                    where_clause.predicates.push(syn::parse_quote_spanned! { variant.span =>
                        #payload_type: #ryft::BatchableOperation<__ParentContext, #policy>
                    })
                }
                OperationVariantClass::MixedMember { .. } => {
                    where_clause.predicates.push(syn::parse_quote_spanned! { variant.span =>
                        #payload_type: #ryft::MemberBatchableOperation<__ParentContext, #policy>
                    })
                }
                OperationVariantClass::ProjectedMember { member_type, structural } => {
                    let projection_predicates: [syn::WherePredicate; 4] = [
                        syn::parse_quote! {
                            #constant_type: #ryft::ValueProjection<
                                #member_type,
                                Projected: #ryft::Value<Type = #member_type>,
                            >
                        },
                        syn::parse_quote! {
                            <__ParentContext as #ryft::Domain>::Constant: #ryft::ValueProjection<
                                #member_type,
                                Projected: #ryft::Value<Type = #member_type>,
                            >
                        },
                        syn::parse_quote! {
                            #parent_value_type: #ryft::ValueProjection<
                                #member_type,
                                Projected: #ryft::Value<Type = #member_type>,
                            >
                        },
                        syn::parse_quote! {
                            <__ParentContext as #ryft::Domain>::Operation: #ryft::OperationProjection<
                                #member_type,
                                Projected = #payload_type,
                            >
                        },
                    ];
                    where_clause.predicates.extend(projection_predicates);
                    if *structural {
                        where_clause.predicates.push(syn::parse_quote! {
                            #policy: #ryft::ReplicatedBatchingPolicyProjection<__ParentContext, #member_type>
                        });
                    } else {
                        where_clause.predicates.push(syn::parse_quote! {
                            #policy: #ryft::BatchingPolicyProjection<__ParentContext, #member_type>
                        });

                        // We state the domain of the projected context as an assumption, so that the elaboration
                        // of the `Operation<Type = C::Type>` super-trait of the payload rule predicate normalizes
                        // through the parameter environment instead of re-evaluating the `Domain` implementation of
                        // `ProjectedContext`, whose `DomainProjection` bound would recurse through the
                        // `OperationProjection` implementation of this family.
                        where_clause.predicates.push(syn::parse_quote! {
                            #ryft::ProjectedContext<__ParentContext, #member_type>: #ryft::Context<Type = #member_type>
                        });
                        where_clause.predicates.push(syn::parse_quote_spanned! { variant.span =>
                            #payload_type: #ryft::BatchableOperation<
                                #ryft::ProjectedContext<__ParentContext, #member_type>,
                                <#policy as #ryft::BatchingPolicyProjection<__ParentContext, #member_type>>::Projected,
                            >
                        });
                    }
                }
            }
        }

        // Recursive higher-order rules seed their accumulators through the `Zero` capability of the parent context.
        // All other value requirements are carried by the per-payload batching predicates.
        where_clause.predicates.push(syn::parse_quote!(__ParentContext: #ryft::Zero<#parent_value_type>));
        let arms = self.variants.iter().map(|variant| {
            let payload_type = &variant.program_payload_type;
            let receiver = variant.receiver();
            variant.arm(match &variant.class {
                OperationVariantClass::FamilyNative => quote! {
                    <#payload_type as #ryft::BatchableOperation<__ParentContext, #policy>>::
                        batch(#receiver, context, driver, inputs)
                },
                OperationVariantClass::MixedMember { .. } => quote! {
                    <#payload_type as #ryft::MemberBatchableOperation<__ParentContext, #policy>>::
                        batch_in_parent(#receiver, context, driver, inputs)
                },
                OperationVariantClass::ProjectedMember { structural: false, .. } => {
                    quote!(context.batch_projected_operation(#receiver, inputs))
                }
                OperationVariantClass::ProjectedMember { member_type, structural: true } => {
                    quote! {
                        context.batch_replicated_projected_operation::<#member_type>(#receiver, inputs)
                    }
                }
            })
        });
        generics.params.push(syn::parse_quote!(__BatchingPolicy));
        let (impl_generics, _, where_clause) = generics.split_for_impl();
        quote! {
            #[automatically_derived]
            impl #impl_generics #ryft::BatchableOperation<__ParentContext, #policy> for #self_type #where_clause {
                #[inline]
                fn batch<__D: #ryft::BatchingDriver<__ParentContext, #policy>>(
                    &self,
                    context: &#ryft::BatchingContext<__ParentContext, #policy>,
                    driver: &__D,
                    inputs: &[#batch_type],
                ) -> ::std::result::Result<::std::vec::Vec<#batch_type>, #ryft::BatchingError> {
                    match self { #(#arms)* }
                }
            }
        }
    }

    /// Generates the `DifferentiableOperation` dispatcher, which implements both `jvp` and `jvp_for_transpose`.
    /// The dispatcher is generic over a differentiation context that is pinned to the primary operation type, to the
    /// constant type, and to the enum itself as its operation family. Family-native variants delegate to the forward
    /// mode rules of their payloads, and computational member variants delegate to the parent-universe rules of their
    /// payloads. Structural member variants are constant with respect to their inputs. A structural projected member
    /// is differentiated by `DifferentiationContext::jvp_structural_operation`, which stages its primal with symbolic
    /// zero tangents, while a structural mixed member is differentiated by
    /// `DifferentiationContext::jvp_structural_mixed_operation`, which also stages one member zero tangent per output,
    /// through `DifferentiationContext::stage_member_zero_tangent` for each computational member universe of the enum.
    /// Higher-order rules request nested differentiation through their instruction-scoped driver, whose concrete
    /// implementation establishes the required `Program::jvp` and `Program::linearize` bounds at its construction site.
    fn generate_differentiation_dispatcher(&self) -> TokenStream {
        let ryft = &self.ryft_crate;
        let operation_type = &self.operation_type;
        let constant_type = &self.constant_type;
        let self_type = &self.program_self_type;
        let mut generics = self.dispatcher_generics.clone();
        generics.params.push(syn::parse_quote!(__DifferentiationContext));
        let where_clause = generics.make_where_clause();
        where_clause.predicates.push(syn::parse_quote! {
            __DifferentiationContext: #ryft::Context<
                Type = #operation_type,
                Constant = #constant_type,
                Operation = #self_type,
            >
        });
        where_clause
            .predicates
            .push(syn::parse_quote!(#self_type: #ryft::Operation<Type = #operation_type>));

        // Higher-order payload rules materialize structural zero tangents at sub-program boundaries through the `Zero`
        // capability of the context. Staging contexts satisfy it through the blanket staging `Zero` implementation,
        // and eager domains implement it directly.
        where_clause.predicates.push(syn::parse_quote! {
            __DifferentiationContext: #ryft::Zero<<__DifferentiationContext as #ryft::Domain>::Value>
        });

        // The `while` rule concretizes data-dependent loop predicates on the carried values when the context is eager,
        // and so the dispatcher also requires `Concretizable<bool>` values. Staged values satisfy it through the
        // `Concretizable<bool>` implementation of tracers (whose `concretize` defers with an error), and eager values
        // implement it directly.
        where_clause.predicates.push(syn::parse_quote! {
            <__DifferentiationContext as #ryft::Domain>::Value: #ryft::Concretizable<bool>
        });
        for variant in &self.variants {
            let payload_type = &variant.program_payload_type;
            match &variant.class {
                OperationVariantClass::FamilyNative => {
                    where_clause.predicates.push(syn::parse_quote_spanned! { variant.span =>
                        #payload_type: #ryft::DifferentiableOperation<__DifferentiationContext>
                    })
                }
                OperationVariantClass::ProjectedMember { structural: false, .. }
                | OperationVariantClass::MixedMember { structural: false, .. } => {
                    where_clause.predicates.push(syn::parse_quote_spanned! { variant.span =>
                        #payload_type: #ryft::MemberDifferentiableOperation<__DifferentiationContext>
                    });
                }
                OperationVariantClass::ProjectedMember { structural: true, .. } => {}

                // A structural mixed member needs a zero constructor and a type view for every computational member
                // universe that an output may belong to.
                OperationVariantClass::MixedMember { member_type, structural: true } => {
                    for tangent_universe in self.tangent_universes(member_type) {
                        let tangent_predicates: [syn::WherePredicate; 3] = [
                            syn::parse_quote!(#tangent_universe: #ryft::DifferentiableType),
                            syn::parse_quote! {
                                #self_type: ::std::convert::From<#ryft::ZeroOperation<#tangent_universe>>
                            },
                            syn::parse_quote! {
                                for<'__t> &'__t #tangent_universe:
                                    ::std::convert::TryFrom<&'__t #operation_type, Error = #ryft::TypeError>
                            },
                        ];
                        where_clause.predicates.extend(tangent_predicates);
                    }
                }
            }
        }

        // Higher-order payload rules (e.g., condition, while, and scan) differentiate their nested programs through the
        // call-scoped `DifferentiationDriver`, whose implementations carry the `Program::jvp` and `Program::linearize`
        // bounds at their construction sites, and so the dispatcher itself only supplies the zero provider that those
        // rules use to materialize structural zeros.
        where_clause.predicates.push(syn::parse_quote! {
            #self_type: #ryft::OperationProvider<
                #operation_type,
                #ryft::ZeroOperation<#operation_type>,
                Operation = #self_type,
            >
        });

        let value_type: syn::Type = syn::parse_quote! {
            #ryft::DifferentiationDual<<__DifferentiationContext as #ryft::Domain>::Value>
        };
        let functions = [("jvp", "jvp_in_parent"), ("jvp_for_transpose", "jvp_for_transpose_in_parent")].map(
            |(function, member_function)| {
                let function = format_ident!("{function}");
                let member_function = format_ident!("{member_function}");
                let arms = self.variants.iter().map(|variant| {
                    let payload_type = &variant.program_payload_type;
                    let receiver = variant.receiver();
                    match &variant.class {
                        OperationVariantClass::FamilyNative => variant.arm(quote! {
                            <#payload_type as #ryft::DifferentiableOperation<__DifferentiationContext>>::
                                #function(#receiver, context, driver, inputs)
                        }),
                        OperationVariantClass::ProjectedMember { structural: false, .. }
                        | OperationVariantClass::MixedMember { structural: false, .. } => variant.arm(quote! {
                            <#payload_type as #ryft::MemberDifferentiableOperation<__DifferentiationContext>>::
                                #member_function(#receiver, context, driver, inputs)
                        }),
                        OperationVariantClass::ProjectedMember { structural: true, .. } => {
                            variant.wildcard_arm(quote!(context.jvp_structural_operation(self.clone(), inputs)))
                        }

                        // Each output's zero tangent is staged in the computational member universe that the output
                        // belongs to, and an output outside every such universe (e.g., a structural member output)
                        // has a zero differential space and so it receives a symbolic zero tangent instead.
                        OperationVariantClass::MixedMember { member_type, structural: true } => {
                            let tangent_universes = self.tangent_universes(member_type);
                            variant.wildcard_arm(quote! {
                                context.jvp_structural_mixed_operation(
                                    self.clone(),
                                    inputs,
                                    |context, primal_inputs, output_type| {
                                        #(
                                            if let ::std::option::Option::Some(tangent) =
                                                context.stage_member_zero_tangent::<#tangent_universes>(
                                                    primal_inputs,
                                                    output_type,
                                                )?
                                            {
                                                return ::std::result::Result::Ok(::std::option::Option::Some(tangent));
                                            }
                                        )*
                                        ::std::result::Result::Ok(::std::option::Option::None)
                                    },
                                )
                            })
                        }
                    }
                });
                quote! {
                    #[inline]
                    fn #function<
                        __D: #ryft::DifferentiationDriver<__DifferentiationContext>,
                        __Policy: #ryft::DifferentiationPolicy<__DifferentiationContext>,
                    >(
                        &self,
                        context: &#ryft::DifferentiationContext<__DifferentiationContext, __Policy>,
                        driver: &__D,
                        inputs: &[#value_type],
                    ) -> ::std::result::Result<::std::vec::Vec<#value_type>, #ryft::DifferentiationError> {
                        match self { #(#arms)* }
                    }
                }
            },
        );
        let (impl_generics, _, where_clause) = generics.split_for_impl();
        quote! {
            #[automatically_derived]
            impl #impl_generics #ryft::DifferentiableOperation<__DifferentiationContext> for #self_type #where_clause {
                #(#functions)*
            }
        }
    }

    /// Generates the `TransposableOperation` dispatcher. Enums with a separate runtime value type (refer to
    /// [`OperationCodeGenerator::has_separate_runtime_value_type`]) transpose over a generated `__TranspositionValue`
    /// parameter that inherits the bounds of the constant parameter, while all other enums transpose over their
    /// constant type. Family-native variants delegate to the transposition rules of their payloads. Computational
    /// projected member variants delegate to `MemberTransposableOperation::transpose_in_parent`, which preserves
    /// access to cotangent accumulators that the value family of the member may not be able to represent. Mixed member
    /// variants are transposed through `TranspositionContext::transpose_mixed_operation`, and structural projected
    /// member variants contribute no cotangents. Higher-order rules request nested program transposition through
    /// their instruction-scoped driver, whose concrete implementation establishes the required
    /// `Program::transpose_with_respect_to` bounds.
    fn generate_transposition_dispatcher(&self) -> TokenStream {
        let ryft = &self.ryft_crate;
        let operation_type = &self.operation_type;
        let self_type = &self.self_type;
        let mut generics = self.operation_generics.clone();
        let value_type: syn::Type = if self.has_separate_runtime_value_type {
            generics.params.push(syn::parse_quote!(__TranspositionValue));
            syn::parse_quote!(__TranspositionValue)
        } else {
            self.constant_type.clone()
        };
        let where_clause = generics.make_where_clause();
        if self.has_separate_runtime_value_type {
            where_clause.predicates.push(syn::parse_quote!(#value_type: #ryft::Value<Type = #operation_type>));
            where_clause
                .predicates
                .extend(self.constant_parameter_predicates(&format_ident!("__TranspositionValue")));
        }
        where_clause
            .predicates
            .push(syn::parse_quote!(#self_type: #ryft::Operation<Type = #operation_type>));
        for variant in &self.variants {
            let payload_type = &variant.payload_type;
            match &variant.class {
                OperationVariantClass::FamilyNative => {
                    where_clause.predicates.push(syn::parse_quote_spanned! { variant.span =>
                        #payload_type: #ryft::TransposableOperation<#value_type, #self_type>
                    })
                }
                OperationVariantClass::ProjectedMember { structural: false, .. } => {
                    where_clause.predicates.push(syn::parse_quote_spanned! { variant.span =>
                        #payload_type: #ryft::MemberTransposableOperation<#value_type, #self_type>
                    });
                }
                OperationVariantClass::ProjectedMember { structural: true, .. } => {}
                OperationVariantClass::MixedMember { member_type, .. } => {
                    let projected_operation_type: syn::Type = syn::parse_quote! {
                        <#self_type as #ryft::OperationProjection<#member_type>>::Projected
                    };
                    let mixed_predicates: [syn::WherePredicate; 7] = [
                        syn::parse_quote!(#member_type: #ryft::DifferentiableType),
                        syn::parse_quote! {
                            #value_type: #ryft::ValueProjection<
                                #member_type,
                                Projected: #ryft::Value<Type = #member_type>,
                            >
                        },
                        syn::parse_quote!(#self_type: #ryft::OperationProjection<#member_type>),
                        syn::parse_quote! {
                            #projected_operation_type: #ryft::OperationProvider<
                                #member_type,
                                #ryft::GuardOperation<#member_type>,
                                Operation = #projected_operation_type,
                            >
                        },
                        syn::parse_quote! {
                            #projected_operation_type: ::std::convert::From<#payload_type>
                                + #ryft::TransposableOperation<
                                    <#value_type as #ryft::ValueProjection<#member_type>>::Projected,
                                    #projected_operation_type,
                                >
                        },
                        syn::parse_quote!(#operation_type: ::std::convert::From<#member_type>),
                        syn::parse_quote! {
                            for<'__t> &'__t #member_type:
                                ::std::convert::TryFrom<&'__t #operation_type, Error = #ryft::TypeError>
                        },
                    ];
                    where_clause.predicates.extend(mixed_predicates);
                }
            }
        }
        let arms = self.variants.iter().map(|variant| {
            let payload_type = &variant.payload_type;
            let receiver = variant.receiver();
            match &variant.class {
                OperationVariantClass::FamilyNative => variant.arm(quote! {
                    <#payload_type as #ryft::TransposableOperation<#value_type, #self_type>>::
                        transpose(#receiver, context, driver, inputs, outputs, accumulators)
                }),
                OperationVariantClass::ProjectedMember { structural: false, .. } => variant.arm(quote! {
                    <#payload_type as #ryft::MemberTransposableOperation<#value_type, #self_type>>::
                        transpose_in_parent(#receiver, context, driver, inputs, outputs, accumulators)
                }),
                OperationVariantClass::ProjectedMember { structural: true, .. } => {
                    variant.wildcard_arm(quote!(::std::result::Result::Ok(())))
                }
                OperationVariantClass::MixedMember { .. } => variant.arm(quote! {
                    context.transpose_mixed_operation(#receiver, inputs, outputs, accumulators)
                }),
            }
        });
        let tracer_type: syn::Type = syn::parse_quote! {
            #ryft::Tracer<#ryft::TracingContext<#value_type, #self_type>>
        };
        let (impl_generics, _, where_clause) = generics.split_for_impl();
        quote! {
            #[automatically_derived]
            impl #impl_generics #ryft::TransposableOperation<#value_type, #self_type> for #self_type #where_clause {
                #[inline]
                fn transpose<__D: #ryft::TranspositionDriver<#value_type, #self_type>>(
                    &self,
                    context: &mut #ryft::TranspositionContext<#value_type, #self_type>,
                    driver: &__D,
                    inputs: &[#ryft::PartialValue<#tracer_type>],
                    outputs: &[#ryft::MaybeZero<#tracer_type>],
                    accumulators: &[#ryft::CotangentAccumulator],
                ) -> ::std::result::Result<(), #ryft::DifferentiationError> {
                    match self { #(#arms)* }
                }
            }
        }
    }

    /// Generates the canonical `OperationProjection<U>` implementation for the provided variant if it is a projected
    /// member variant of member type `U` (in either role), and returns [`None`] otherwise. Family-native and mixed
    /// member variants do not define homogeneous operation family projections.
    fn generate_operation_projection_impl(&self, variant: &OperationVariant) -> Option<TokenStream> {
        let member_type = variant.class.projected_member_type()?;
        let ryft = &self.ryft_crate;
        let self_type = &self.self_type;
        let payload_type = &variant.payload_type;
        let (impl_generics, _, where_clause) = self.operation_generics.split_for_impl();
        Some(quote! {
            #[automatically_derived]
            impl #impl_generics #ryft::OperationProjection<#member_type> for #self_type #where_clause {
                type Projected = #payload_type;
            }
        })
    }

    /// Generates the `OperationPayloadProjection` implementation. Projected member variants delegate to their
    /// member families in both directions, lifting constructed member operations through the `From<Member>`
    /// conversion of the enum. Every other variant projects into its payload when the payload type is `'static`
    /// for every instantiation of the generic parameters of the enum (refer to [`is_unconditionally_static`]), and
    /// such family-native variants are also constructed from their payloads, before any projected member is tried.
    /// Mixed member variants are never constructed, and generic extension variants are neither projected nor
    /// constructed.
    fn generate_payload_projection_impl(&self) -> TokenStream {
        let ryft = &self.ryft_crate;
        let self_type = &self.self_type;
        let mut generics = self.generics.clone();
        let mut projection_arms = Vec::new();
        let mut native_steps = Vec::new();
        let mut member_steps = Vec::new();
        for variant in self.variants.iter().filter(|variant| !variant.is_generic_extension) {
            let variant_ident = &variant.ident;
            let payload_type = &variant.payload_type;
            let receiver = variant.receiver();
            if variant.class.projected_member_type().is_some() {
                generics
                    .make_where_clause()
                    .predicates
                    .push(syn::parse_quote_spanned!(variant.span => #payload_type: #ryft::OperationPayloadProjection));
                projection_arms.push(quote! {
                    Self::#variant_ident(operation) => {
                        <#payload_type as #ryft::OperationPayloadProjection>::project_payload(#receiver, payload)
                    }
                });
                member_steps.push(quote! {
                    let payload = match <#payload_type as #ryft::OperationPayloadProjection>::from_payload(payload) {
                        ::std::result::Result::Ok(operation) => {
                            return ::std::result::Result::Ok(<Self as ::std::convert::From<#payload_type>>::from(
                                operation,
                            ));
                        }
                        ::std::result::Result::Err(payload) => payload,
                    };
                });
            } else if is_unconditionally_static(payload_type, &self.generics) {
                projection_arms.push(quote! {
                    Self::#variant_ident(operation) if payload == ::std::any::TypeId::of::<#payload_type>() => {
                        ::std::option::Option::Some(#receiver as &dyn ::std::any::Any)
                    }
                });
                if matches!(variant.class, OperationVariantClass::FamilyNative) {
                    let operation = variant.construct(quote!(operation));
                    native_steps.push(quote! {
                        let payload = match payload.downcast::<#payload_type>() {
                            ::std::result::Result::Ok(operation) => {
                                return ::std::result::Result::Ok(#operation);
                            }
                            ::std::result::Result::Err(payload) => payload,
                        };
                    });
                }
            }
        }
        let (impl_generics, _, where_clause) = generics.split_for_impl();
        quote! {
            #[automatically_derived]
            impl #impl_generics #ryft::OperationPayloadProjection for #self_type #where_clause {
                #[allow(unreachable_patterns)]
                #[inline]
                fn project_payload(
                    &self,
                    payload: ::std::any::TypeId,
                ) -> ::std::option::Option<&dyn ::std::any::Any> {
                    match self {
                        #(#projection_arms)*
                        _ => ::std::option::Option::None,
                    }
                }

                #[inline]
                fn from_payload(
                    payload: #ryft::ErasedOperation,
                ) -> ::std::result::Result<Self, #ryft::ErasedOperation> {
                    #(#native_steps)*
                    #(#member_steps)*
                    ::std::result::Result::Err(payload)
                }
            }
        }
    }

    /// Generates the `CollectiveOperation` implementation, which delegates both of its functions to the payloads
    /// of the variants that declare `#[ryft(collective)]` and reports no axes and no hidden axes for every other
    /// variant, or returns [`None`] when no variant declares it. The payload of a marked variant either implements
    /// `CollectiveOperation` itself (e.g., a collective or a composite call) or is a family that does (e.g., a
    /// projected member family or a generic extension), and it is bounded accordingly, so that a family with a
    /// marked generic payload implements the trait only when that payload does.
    fn generate_collective_impl(&self) -> Option<TokenStream> {
        let ryft = &self.ryft_crate;
        let self_type = &self.self_type;
        let mut generics = self.generics.clone();
        let mut axes_arms = Vec::new();
        let mut hidden_axes_arms = Vec::new();
        for variant in self.variants.iter().filter(|variant| variant.is_collective) {
            let payload_type = &variant.payload_type;
            let receiver = variant.receiver();
            generics
                .make_where_clause()
                .predicates
                .push(syn::parse_quote_spanned!(variant.span => #payload_type: #ryft::CollectiveOperation));
            axes_arms.push(variant.arm(quote!(<#payload_type as #ryft::CollectiveOperation>::axes(#receiver))));
            hidden_axes_arms
                .push(variant.arm(quote!(<#payload_type as #ryft::CollectiveOperation>::hidden_axes(#receiver))));
        }

        if axes_arms.is_empty() {
            return None;
        }

        let (impl_generics, _, where_clause) = generics.split_for_impl();
        Some(quote! {
            #[automatically_derived]
            impl #impl_generics #ryft::CollectiveOperation for #self_type #where_clause {
                #[allow(unreachable_patterns)]
                #[inline]
                fn axes(&self) -> ::std::borrow::Cow<'_, #ryft::CollectiveAxes> {
                    match self {
                        #(#axes_arms)*
                        _ => ::std::borrow::Cow::Owned(
                            <#ryft::CollectiveAxes as ::std::default::Default>::default(),
                        ),
                    }
                }

                #[allow(unreachable_patterns)]
                #[inline]
                fn hidden_axes(&self) -> #ryft::HiddenAxes {
                    match self {
                        #(#hidden_axes_arms)*
                        _ => #ryft::HiddenAxes::None,
                    }
                }
            }
        })
    }

    /// Generates the `LockstepOperation` implementation, which delegates to the payloads of the variants that declare
    /// `#[ryft(lockstep)]` and reports no lockstep rule for every other variant, or returns [`None`] when no variant
    /// declares it. The implementation is generic over the root operation family of the emulated program, and the
    /// payload of each marked variant is bounded to implement `LockstepOperation` for that family, so that a family
    /// with a marked member family or generic extension implements the trait only when that payload does.
    fn generate_lockstep_impl(&self) -> Option<TokenStream> {
        let ryft = &self.ryft_crate;
        let self_type = &self.self_type;
        let mut generics = self.generics.clone();
        let mut arms = Vec::new();
        for variant in self.variants.iter().filter(|variant| variant.is_lockstep) {
            let payload_type = &variant.payload_type;
            let receiver = variant.receiver();
            generics.make_where_clause().predicates.push(
                syn::parse_quote_spanned!(variant.span => #payload_type: #ryft::LockstepOperation<LockstepFamily>),
            );
            arms.push(variant.arm(quote! {
                <#payload_type as #ryft::LockstepOperation<LockstepFamily>>::interpret_in_lockstep(
                    #receiver,
                    context,
                    driver,
                    inputs,
                )
            }));
        }
        if arms.is_empty() {
            return None;
        }
        generics.params.push(syn::parse_quote!(LockstepFamily: #ryft::Operation<Type = #ryft::ArrayIrType>));
        let (impl_generics, _, where_clause) = generics.split_for_impl();
        Some(quote! {
            #[automatically_derived]
            impl #impl_generics #ryft::LockstepOperation<LockstepFamily> for #self_type #where_clause {
                #[allow(unreachable_patterns)]
                #[inline]
                fn interpret_in_lockstep<
                    LockstepDriver: #ryft::InterpretationDriver<#ryft::LockstepDomain<LockstepFamily>>,
                >(
                    &self,
                    context: &#ryft::LockstepDomain<LockstepFamily>,
                    driver: &LockstepDriver,
                    inputs: &[#ryft::LockstepValue<LockstepFamily>],
                ) -> ::std::option::Option<
                    ::std::result::Result<
                        ::std::vec::Vec<#ryft::LockstepValue<LockstepFamily>>,
                        #ryft::ProgramError,
                    >,
                > {
                    match self {
                        #(#arms)*
                        _ => ::std::option::Option::None,
                    }
                }
            }
        })
    }

    /// Generates the `OperationProvider` implementation that rejects every `GuardOperation` request of the primary
    /// operation type with `GuardOperation::unsupported_error`, or returns [`None`] when a variant stores a native
    /// guard (refer to [`OperationVariant::is_native_guard`]). Every `Context` requires its operation family to
    /// provide guard requests, and so a family that stores no guard needs this provider to be usable as the family
    /// of a context at all. A family that stores a native guard instead converts it through the generated
    /// `From<GuardOperation<T>>` implementation, which the blanket supporting provider of `ryft-core`
    /// turns into an accepting provider.
    ///
    /// The two implementations never overlap. The blanket implementation requires `Enum: From<GuardOperation<T>>`,
    /// which only the deriving crate can implement for its own enum, and so the compiler proves that it does not
    /// hold whenever no such conversion exists. A family that stores no native guard but handwrites that conversion
    /// therefore fails to compile with a conflicting implementations error (E0119), instead of silently rejecting
    /// guards that it could represent.
    fn generate_guard_provider_impl(&self) -> Option<TokenStream> {
        if self.variants.iter().any(OperationVariant::is_native_guard) {
            return None;
        }
        let ryft = &self.ryft_crate;
        let operation_type = &self.operation_type;
        let self_type = &self.self_type;
        let (impl_generics, _, where_clause) = self.operation_generics.split_for_impl();
        Some(quote! {
            #[automatically_derived]
            impl #impl_generics #ryft::OperationProvider<#operation_type, #ryft::GuardOperation<#operation_type>>
                for #self_type
            #where_clause
            {
                type Operation = Self;

                #[inline]
                fn provide(
                    request: #ryft::GuardOperation<#operation_type>,
                    _input_types: &[&#operation_type],
                ) -> ::std::result::Result<Self, #ryft::ProgramError> {
                    ::std::result::Result::Err(request.unsupported_error())
                }
            }
        })
    }

    /// Generates the owned `From<Payload>` and borrowed `TryFrom<&Enum>` payload conversions for the provided variant.
    /// Boxed payloads are boxed by `From` and dereferenced through the box by `TryFrom`. The `From` conversion is
    /// skipped for variants that declare `#[ryft(skip_from)]`. The borrowed projection fails with a `TypeError` that
    /// names both the stored operation and the expected payload type, and so it requires the enum to implement
    /// `Operation`.
    fn generate_conversion_impls(&self, variant: &OperationVariant) -> TokenStream {
        let ryft = &self.ryft_crate;
        let enum_ident = &self.ident;
        let self_type = &self.self_type;
        let variant_ident = &variant.ident;
        let payload_type = &variant.payload_type;
        let payload_name = payload_type.to_compact_string();
        let (impl_generics, _, where_clause) = self.generics.split_for_impl();
        let mut try_from_generics = self.generics.clone();
        try_from_generics.params.insert(0, syn::parse_quote!('__operation));
        try_from_generics
            .make_where_clause()
            .predicates
            .push(syn::parse_quote!(#self_type: #ryft::Operation));
        let (try_from_impl_generics, _, try_from_where_clause) = try_from_generics.split_for_impl();
        let operation = variant.construct(quote!(operation));
        let receiver = variant.receiver();

        // Only the implementation headers are spanned at the variant, so that conflicting implementations (e.g., of
        // a payload type that is held by several variants) point at it. The bodies keep the call site span, so that
        // lints on generated code (e.g., the unreachable fallback arm of a single variant enum) are not reported in
        // user code.
        let from_body = quote! {
            #[inline]
            fn from(operation: #payload_type) -> Self {
                #operation
            }
        };
        let try_from_body = quote! {
            type Error = #ryft::TypeError;

            #[inline]
            fn try_from(value: &'__operation #self_type) -> ::std::result::Result<Self, #ryft::TypeError> {
                match value {
                    #enum_ident::#variant_ident(operation) => ::std::result::Result::Ok(#receiver),
                    _ => ::std::result::Result::Err(#ryft::TypeError::invalid(::std::format!(
                        "cannot project operation `{}` into a `{}` payload",
                        <#self_type as #ryft::Operation>::name(value),
                        #payload_name,
                    ))),
                }
            }
        };
        let from_impl = variant.generate_from.then(|| {
            quote_spanned! { variant.span =>
                #[automatically_derived]
                impl #impl_generics ::std::convert::From<#payload_type> for #self_type #where_clause {
                    #from_body
                }
            }
        });
        let try_from_impl = quote_spanned! { variant.span =>
            #[automatically_derived]
            impl #try_from_impl_generics ::std::convert::TryFrom<&'__operation #self_type>
                for &'__operation #payload_type
            #try_from_where_clause
            {
                #try_from_body
            }
        };
        quote!(#from_impl #try_from_impl)
    }

    /// Returns the `where` predicates that copy the declared bounds of [`OperationCodeGenerator::constant_parameter`]
    /// (i.e., both its inline bounds and the `where` predicates whose bounded type references it) onto the provided
    /// generated type parameter, renaming every mention of the constant parameter in them. That way, a generated
    /// runtime value type inherits the capability requirements that the enum declares for its constant parameter.
    /// This function returns no predicates for enums that declare an explicit constant type.
    fn constant_parameter_predicates(&self, value_parameter: &syn::Ident) -> Vec<syn::WherePredicate> {
        let Some(parameter) = &self.constant_parameter else {
            return Vec::new();
        };
        let value_path = syn::Path::from(value_parameter.clone());
        let inline_predicates = self
            .generics
            .type_params()
            .filter(|type_parameter| type_parameter.ident == *parameter)
            .flat_map(|type_parameter| type_parameter.bounds.iter())
            .map(|bound| {
                let mut predicate: syn::WherePredicate = syn::parse_quote!(#value_parameter: #bound);
                predicate.replace_ident(parameter, &value_path);
                predicate
            });
        let where_predicates = self
            .generics
            .where_clause
            .iter()
            .flat_map(|where_clause| where_clause.predicates.iter())
            .filter(|predicate| match predicate {
                syn::WherePredicate::Type(predicate) => predicate.bounded_ty.references_ident(parameter),
                _ => false,
            })
            .map(|predicate| {
                let mut predicate = predicate.clone();
                predicate.replace_ident(parameter, &value_path);
                predicate
            });
        inline_predicates.chain(where_predicates).collect()
    }

    /// Returns the computational member universes that an output of a structural mixed member payload with the
    /// provided member type can belong to. These are the member type of the payload itself, followed by every other
    /// computational member universe that the enum declares through `#[ryft(members(...))]`. An enum that declares no
    /// members therefore exposes only the member type of the payload.
    fn tangent_universes(&self, member_type: &syn::Type) -> Vec<syn::Type> {
        std::iter::once(member_type.clone())
            .chain(
                self.members
                    .iter()
                    .filter(|member| !member.structural && !member.r#type.tokens_eq(member_type))
                    .map(|member| member.r#type.clone()),
            )
            .collect()
    }
}

/// Parser that extracts the [`OperationCodeGenerator`] of an operation enum from the provided [`syn::DeriveInput`].
/// Refer to [`OperationCodeGenerator::generate_operation_impl`] for information on how it is used.
struct OperationParser {
    /// [`syn::Path`] that represents the root `ryft` library path (e.g., `ryft`). This is customizable via the
    /// `#[ryft(crate = "...")]` attribute and it is meant to support libraries that build on top of `ryft` and may
    /// want to export `ryft` types as part of their namespace. This is similar to the `#[serde(crate = "...")]`
    /// attribute. You can refer to [its documentation](https://serde.rs/container-attrs.html#crate) for more
    /// information.
    ryft_crate: syn::Path,

    /// Primary operation type, which is either declared through `#[ryft(type(T))]` or inferred from the unique
    /// `Value<Type = T>` bound of the generic parameters of the enum. This is [`None`] only if the type could not be
    /// determined, in which case an error has been reported.
    operation_type: Option<syn::Type>,

    /// Constant type declared through `#[ryft(constant(C))]`. This is [`None`] for enums whose constant type is instead
    /// inferred from their `Value<Type = T>` generic parameters.
    constant_type: Option<syn::Type>,

    /// [`OperationMember`]s declared through `#[ryft(members(...))]`, in declaration order. This is empty when the enum
    /// declares no member universes, in which case every member variant must name its own member type.
    members: Vec<OperationMember>,

    /// [`Dispatchers`] selected through `#[ryft(dispatch(...))]`. This is [`None`] when the enum selects none.
    dispatchers: Option<Dispatchers>,

    /// Errors accumulated in this [`OperationParser`]. The way error handling works in this parser is that we collect
    /// errors as we encounter them, and keep going as far as we can with the information that is available, before
    /// raising them. That is meant to enable a smoother development experience when working with `ryft` by reducing
    /// the amount of trial and error required to get something work (i.e., users do not need to keep trying, fixing
    /// one error at a time; they get to see multiple errors at once, when there are multiple).
    errors: Vec<syn::Error>,
}

impl OperationParser {
    /// Creates a new [`OperationParser`] that has not extracted any information yet.
    fn new() -> Self {
        Self {
            ryft_crate: DEFAULT_RYFT_CRATE.into(),
            operation_type: None,
            constant_type: None,
            members: Vec::new(),
            dispatchers: None,
            errors: Vec::new(),
        }
    }

    /// Adds an error to this [`OperationParser`] with the specified message spanning the provided tokens. This is only
    /// meant to be used internally by this parser as a convenient helper for collecting errors.
    ///
    /// # Parameters
    ///
    ///   - `tokens`: Tokens that the error spans.
    ///   - `message`: Message describing the error.
    fn add_error<T: ToTokens, U: Display>(&mut self, tokens: T, message: U) {
        self.errors.push(syn::Error::new_spanned(tokens.into_token_stream(), message));
    }

    /// Returns a [`TokenStream`] that represents a [`compile_error!`] invocation that contains information about the
    /// [`syn::Error`]s that have been collected by this [`OperationParser`] so far. If there are no errors, then this
    /// function returns [`None`].
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

    /// Extracts the `#[ryft(...)]` attributes that are attached to the provided [`syn::DeriveInput`]
    /// and checks for unknown enum-level attributes. This function sets [`OperationParser::ryft_crate`],
    /// [`OperationParser::constant_type`], [`OperationParser::members`], and [`OperationParser::dispatchers`] if it is
    /// able to successfully extract them. It also sets [`OperationParser::operation_type`], which is either declared
    /// explicitly through `#[ryft(type(T), constant(C))]` (both of which must be declared together) or inferred from
    /// the unique `Value<Type = T>` bound of the generic parameters of the enum.
    fn extract_attributes(&mut self, input: &syn::DeriveInput) {
        let mut ryft_crate = Attribute::new(CRATE_ATTRIBUTE);
        input.attrs.iter().filter(|attr| attr.path() == &RYFT_ATTRIBUTE).for_each(|attr| {
            attr.parse_nested_meta(|meta| match &meta.path {
                path if path == &CRATE_ATTRIBUTE => ryft_crate.set(&meta),
                path if path == &TYPE_ATTRIBUTE => {
                    Self::extract_type_attribute(&meta, TYPE_ATTRIBUTE, &mut self.operation_type)
                }
                path if path == &CONSTANT_ATTRIBUTE => {
                    Self::extract_type_attribute(&meta, CONSTANT_ATTRIBUTE, &mut self.constant_type)
                }
                path if path == &MEMBERS_ATTRIBUTE => self.extract_members_attribute(&meta),
                path if path == &DISPATCH_ATTRIBUTE => self.extract_dispatch_attribute(&meta),
                path if path == &IDENTITY_DISPATCHER => {
                    Err(meta.error("the `identity` attribute was replaced by the `dispatch(identity)` dispatcher"))
                }
                _ => Err(meta.error(format_args!(
                    "invalid `#[ryft(...)]` attribute `{}`; only `crate`, `type(...)`, `constant(...)`, \
                     `members(...)`, and `dispatch(...)` are supported here",
                    meta.path.to_compact_string(),
                ))),
            })
            .unwrap_or_else(|error| self.errors.push(error));
        });

        if let Some(ryft_crate) = ryft_crate.get() {
            self.ryft_crate = ryft_crate;
        }

        // An attribute that fails to parse also stops the parsing of the attributes that follow it in the same list,
        // and so we only determine the operation type if all attributes were parsed, to avoid follow-on errors.
        if !self.errors.is_empty() {
            return;
        }

        match (&self.operation_type, &self.constant_type) {
            (Some(_), Some(_)) => {}
            (None, None) => {
                let mut operation_types = Vec::<syn::Type>::new();
                for (_, operation_type) in value_parameters(&input.generics) {
                    if !operation_types.iter().any(|existing_type| existing_type.tokens_eq(&operation_type)) {
                        operation_types.push(operation_type);
                    }
                }
                match operation_types.as_slice() {
                    [operation_type] => self.operation_type = Some(operation_type.clone()),
                    [] => self.add_error(
                        &input.ident,
                        "could not infer an operation type because no generic parameter is bounded by \
                         `Value<Type = T>`",
                    ),
                    _ => self.add_error(
                        &input.generics,
                        "could not infer a unique operation type because multiple distinct `Value<Type = T>` \
                         bounds are present",
                    ),
                }
            }
            _ => self.add_error(
                &input.ident,
                "explicit operation-family declarations must specify both `type(...)` and `constant(...)`",
            ),
        }
    }

    /// Extracts the single [`syn::Type`] declared by a `#[ryft(type(T))]` or a `#[ryft(constant(C))]` attribute into
    /// `declared_type`. The type is parsed locally (rather than through the shared `name = value` attribute grammar)
    /// so that any complete Rust type, including generic, qualified, and tuple types, fits inside the parentheses.
    ///
    /// # Parameters
    ///
    ///   - `meta`: Nested meta item whose path is `name`.
    ///   - `name`: Name of the attribute being extracted (i.e., `type` or `constant`), which is used in diagnostics.
    ///   - `declared_type`: Destination for the declared type, which must not have been set by an earlier attribute.
    fn extract_type_attribute(
        meta: &syn::meta::ParseNestedMeta,
        name: Symbol,
        declared_type: &mut Option<syn::Type>,
    ) -> syn::Result<()> {
        if meta.input.peek(syn::Token![=]) {
            return Err(meta.error(format_args!("the `{name} = ...` attribute form was replaced by `{name}(...)`")));
        }
        if declared_type.is_some() {
            return Err(meta.error(format_args!("duplicate ryft attribute `{name}(...)`")));
        }
        let error = || meta.error(format_args!("`#[ryft({name}(...))]` must declare exactly one type"));
        if !meta.input.peek(syn::token::Paren) {
            return Err(error());
        }
        let content;
        syn::parenthesized!(content in meta.input);
        match content.parse::<syn::Type>() {
            Ok(parsed_type) if content.is_empty() => {
                *declared_type = Some(parsed_type);
                Ok(())
            }
            _ => Err(error()),
        }
    }

    /// Extracts the [`OperationMember`]s declared by a `#[ryft(members(...))]` attribute into
    /// [`OperationParser::members`]. Declaring the members of an operation family lets `mixed` variant markers default
    /// their member type, lets the member types of all variant markers be validated against the family, and gives the
    /// generated structural mixed forward mode arms the member types they need to classify each output.
    fn extract_members_attribute(&mut self, meta: &syn::meta::ParseNestedMeta) -> syn::Result<()> {
        if !self.members.is_empty() {
            return Err(meta.error("duplicate ryft attribute `members(...)`"));
        }
        let content;
        syn::parenthesized!(content in meta.input);
        let members = syn::punctuated::Punctuated::<OperationMember, syn::Token![,]>::parse_terminated(&content)?;
        if members.is_empty() {
            return Err(meta.error("`#[ryft(members(...))]` must declare at least one member universe"));
        }
        for (index, member) in members.iter().enumerate() {
            if members.iter().take(index).any(|other| other.r#type.tokens_eq(&member.r#type)) {
                return Err(syn::Error::new_spanned(
                    &member.r#type,
                    format_args!("duplicate ryft member universe `{}`", member.r#type.to_compact_string()),
                ));
            }
        }
        self.members = members.into_iter().collect();
        Ok(())
    }

    /// Extracts the [`Dispatchers`] selected by a `#[ryft(dispatch(...))]` attribute into
    /// [`OperationParser::dispatchers`]. Dispatchers may be listed in any order, but by convention `identity`
    /// is listed first.
    fn extract_dispatch_attribute(&mut self, meta: &syn::meta::ParseNestedMeta) -> syn::Result<()> {
        if self.dispatchers.is_some() {
            return Err(meta.error("duplicate ryft attribute `dispatch(...)`"));
        }
        let content;
        let fork = meta.input.fork();
        syn::parenthesized!(content in fork);
        if content.is_empty() {
            return Err(meta.error("`#[ryft(dispatch(...))]` must select at least one dispatcher"));
        }
        let mut dispatchers = Dispatchers::default();
        meta.parse_nested_meta(|meta| {
            let selected = match &meta.path {
                path if path == &IDENTITY_DISPATCHER => &mut dispatchers.identity,
                path if path == &DISCHARGE_DISPATCHER => &mut dispatchers.discharge,
                path if path == &BATCHING_DISPATCHER => &mut dispatchers.batching,
                path if path == &DIFFERENTIATION_DISPATCHER => &mut dispatchers.differentiation,
                path if path == &TRANSPOSITION_DISPATCHER => &mut dispatchers.transposition,
                _ => {
                    return Err(meta.error(format_args!(
                        "invalid `#[ryft(dispatch(...))]` dispatcher `{}`; only `identity`, `discharge`, `batching`, \
                         `differentiation`, and `transposition` are supported here",
                        meta.path.to_compact_string(),
                    )));
                }
            };
            if *selected {
                return Err(meta.error(format_args!("duplicate ryft dispatcher `{}`", meta.path.to_compact_string())));
            }
            *selected = true;
            Ok(())
        })?;
        self.dispatchers = Some(dispatchers);
        Ok(())
    }

    /// Extracts the [`OperationVariant`]s that are contained in the provided [`syn::DeriveInput`]. This function also
    /// checks that the input is an enum, because the operation derivation does not support structs or unions.
    fn extract_variants(&mut self, input: &syn::DeriveInput) -> Vec<OperationVariant> {
        let syn::Data::Enum(data) = &input.data else {
            self.add_error(&input.ident, "the `#[derive(Operation)]` macro only supports enums");
            return Vec::new();
        };
        data.variants.iter().filter_map(|variant| self.extract_variant(&input.generics, variant)).collect()
    }

    /// Extracts one [`OperationVariant`] from the provided [`syn::Variant`]. This function also checks that the variant
    /// is a tuple variant with exactly one payload field and that the payload field has no `#[ryft(...)]` attributes,
    /// returning [`None`] if the variant has no usable payload.
    ///
    /// # Parameters
    ///
    ///   - `generics`: [`syn::Generics`] of the enum, which are used to detect bare generic extension payloads.
    ///   - `variant`: [`syn::Variant`] from which to extract an [`OperationVariant`].
    fn extract_variant(&mut self, generics: &syn::Generics, variant: &syn::Variant) -> Option<OperationVariant> {
        let (class, generate_from, is_collective, is_lockstep) = self.extract_variant_attributes(variant);
        let field = match &variant.fields {
            syn::Fields::Unnamed(fields) if fields.unnamed.len() == 1 => &fields.unnamed[0],
            _ => {
                self.add_error(
                    &variant.ident,
                    "operation enum variants must be tuple variants with exactly one payload field",
                );
                return None;
            }
        };
        field.attrs.iter().filter(|attr| attr.path() == &RYFT_ATTRIBUTE).for_each(|attr| {
            self.add_error(
                attr,
                "operation field attributes are not supported; put operation member-class attributes on the \
                 enclosing variant",
            )
        });

        // Bare generic payloads (e.g., the `Extension` payload of an `Extension(Extension)` variant) get no generated
        // conversions, because a `From<Extension>` implementation would overlap with the concrete payload conversions
        // whenever `Extension` is instantiated as one of the concrete payload types.
        let is_generic_extension = field
            .ty
            .ident()
            .is_some_and(|ident| generics.type_params().any(|parameter| parameter.ident == *ident));

        // Boxed payloads (i.e., type paths whose final segment is `Box` with exactly one type argument, like `Box<T>`
        // or `std::boxed::Box<T>`) dispatch to the boxed type.
        let (payload_type, is_boxed) = if let syn::Type::Path(syn::TypePath { qself: None, path }) = &field.ty
            && let Some(segment) = path.segments.last()
            && segment.ident == "Box"
            && let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments
            && arguments.args.len() == 1
            && let Some(syn::GenericArgument::Type(payload_type)) = arguments.args.first()
        {
            (payload_type.clone(), true)
        } else {
            (field.ty.clone(), false)
        };

        Some(OperationVariant {
            ident: variant.ident.clone(),
            span: field.ty.span(),
            class,
            program_payload_type: payload_type.clone(),
            payload_type,
            is_boxed,
            is_generic_extension,
            generate_from,
            is_collective,
            is_lockstep,
        })
    }

    /// Extracts the [`OperationVariantClass`], the conversion policy, and the collective and lockstep markers declared
    /// for the provided [`syn::Variant`] through its `#[ryft(...)]` attributes. Variants without a class marker are
    /// family-native. The conversion policy is `false` when the variant declares `#[ryft(skip_from)]`, which suppresses
    /// only its generated `From<Payload>` implementation so that a handwritten normalizing conversion can be used
    /// instead. Its borrowed `TryFrom` projection is still generated. The collective marker is `true` when the variant
    /// declares `#[ryft(collective)]`, which makes the generated `CollectiveOperation` implementation delegate to its
    /// payload, and the lockstep marker is `true` when it declares `#[ryft(lockstep)]`, which does the same for the
    /// generated `LockstepOperation` implementation.
    fn extract_variant_attributes(&mut self, variant: &syn::Variant) -> (OperationVariantClass, bool, bool, bool) {
        let mut class = None::<OperationVariantClass>;
        let mut generate_from = true;
        let mut is_collective = false;
        let mut is_lockstep = false;
        variant.attrs.iter().filter(|attr| attr.path() == &RYFT_ATTRIBUTE).for_each(|attr| {
            attr.parse_nested_meta(|meta| match &meta.path {
                path if path == &SKIP_FROM_ATTRIBUTE && !generate_from => {
                    Err(meta.error("duplicate ryft operation variant attribute `skip_from`"))
                }
                path if path == &SKIP_FROM_ATTRIBUTE => {
                    generate_from = false;
                    Ok(())
                }
                path if path == &COLLECTIVE_ATTRIBUTE && is_collective => {
                    Err(meta.error("duplicate ryft operation variant attribute `collective`"))
                }
                path if path == &COLLECTIVE_ATTRIBUTE => {
                    is_collective = true;
                    Ok(())
                }
                path if path == &LOCKSTEP_ATTRIBUTE && is_lockstep => {
                    Err(meta.error("duplicate ryft operation variant attribute `lockstep`"))
                }
                path if path == &LOCKSTEP_ATTRIBUTE => {
                    is_lockstep = true;
                    Ok(())
                }
                path if path == &PROJECTED_ATTRIBUTE || path == &MIXED_ATTRIBUTE => {
                    let candidate = OperationVariantClass::parse(&meta, &self.members)?;
                    match &class {
                        Some(existing) if std::mem::discriminant(existing) == std::mem::discriminant(&candidate) => {
                            Err(meta
                                .error(format_args!("duplicate ryft operation variant class `{}`", candidate.name())))
                        }
                        Some(existing) => Err(meta.error(format_args!(
                            "contradictory ryft operation variant classes `{}` and `{}`",
                            existing.name(),
                            candidate.name(),
                        ))),
                        None => {
                            class = Some(candidate);
                            Ok(())
                        }
                    }
                }
                path if path == &STRUCTURAL_ROLE => Err(meta.error(
                    "the `structural` operation variant class was replaced by the `projected(U, structural)` \
                     role form",
                )),
                _ => Err(meta.error(format_args!(
                    "invalid operation variant `#[ryft(...)]` attribute `{}`; only `projected`, `mixed`, \
                     `collective`, `lockstep`, and `skip_from` are supported here",
                    meta.path.to_compact_string(),
                ))),
            })
            .unwrap_or_else(|error| self.errors.push(error));
        });
        (class.unwrap_or(OperationVariantClass::FamilyNative), generate_from, is_collective, is_lockstep)
    }

    /// Checks that the provided [`OperationVariant`]s can be told apart by the generated payload conversions and
    /// projections, reporting errors otherwise. Every variant other than a bare generic extension receives its own
    /// borrowed `TryFrom<&Enum>` conversion (and usually also an owned `From<Payload>` conversion) for its payload
    /// type, and so a payload type may be held by at most one such variant. That also keeps the payload construction
    /// of `OperationPayloadProjection::from_payload` unambiguous. Payload types are compared by their tokens, and so
    /// payload types that only overlap for some instantiation of the generic parameters of the enum (e.g.,
    /// `Operation<V>` and `Operation<C>`) are left to the coherence checks of the compiler. Similarly, a projected
    /// member type may be declared by at most one variant, because `OperationProjection<U>` names one canonical
    /// projected family.
    fn check_for_conflicting_variants(&mut self, variants: &[OperationVariant]) {
        let mut payload_types = HashSet::new();
        let mut member_types = HashSet::new();
        for variant in variants {
            if !variant.is_generic_extension && !payload_types.insert(variant.payload_type.to_compact_string()) {
                self.add_error(
                    &variant.ident,
                    format_args!(
                        "operation family holds payload type `{}` in more than one variant",
                        variant.payload_type.to_compact_string(),
                    ),
                );
            }
            if let Some(member_type) = variant.class.projected_member_type()
                && !member_types.insert(member_type.to_compact_string())
            {
                self.add_error(
                    &variant.ident,
                    format_args!(
                        "operation family declares member type `{}` more than once",
                        member_type.to_compact_string(),
                    ),
                );
            }
        }
    }

    /// Extracts the [`OperationCodeGenerator`] that generates all implementations for the provided
    /// [`syn::DeriveInput`]. This function extracts the [`OperationVariant`]s, checks them for conflicts,
    /// and computes the generics, the self types, and the payload types of the generated implementations.
    ///
    /// The implementations that operate on programs (i.e., interpretation, partial evaluation, reference discharge,
    /// batching, and differentiation) have a single value space. For enums whose constant type is inferred, the first
    /// `Value<Type = T>` parameter is the constant type and the second one is the capture constant type, while any
    /// later parameters carry payload-specific metadata. These program-shaped implementations therefore substitute
    /// every `Value<Type = T>` parameter after the first two with the first one. Refer to the documentation of the
    /// `Operation` trait in `ryft-core` for more information on how the value types of the generated implementations
    /// are inferred.
    ///
    /// This function must only be called after [`OperationParser::extract_attributes`] has completed without errors,
    /// which guarantees that [`OperationParser::operation_type`] is set.
    fn extract_code_generator(&mut self, input: &syn::DeriveInput) -> OperationCodeGenerator {
        // Variants whose markers fail to parse fall back to the family-native class, and so we only check for
        // conflicting variants if all variants were extracted without errors, to avoid follow-on errors.
        let mut variants = self.extract_variants(input);
        if self.errors.is_empty() {
            self.check_for_conflicting_variants(&variants);
        }

        let ryft = &self.ryft_crate;
        let operation_type = self.operation_type.clone().unwrap();
        let ident = input.ident.clone();
        let generics = input.generics.without_defaults();
        let (_, type_generics, _) = generics.split_for_impl();
        let self_type: syn::Type = syn::parse_quote!(#ident #type_generics);

        // Enums with an explicit constant type have no generic constant parameter. For all other enums, the operation
        // type was inferred from their `Value<Type = T>` parameters, and so there is at least one of them.
        let mut operation_value_parameters = value_parameters(&generics)
            .into_iter()
            .filter(|(_, value_type)| value_type.tokens_eq(&operation_type))
            .map(|(parameter, _)| parameter.clone())
            .collect::<Vec<_>>();
        operation_value_parameters.dedup();
        let (constant_type, constant_parameter, has_separate_runtime_value_type) = match &self.constant_type {
            Some(constant_type) => (constant_type.clone(), None, true),
            None => {
                let parameter = operation_value_parameters[0].clone();
                (syn::parse_quote!(#parameter), Some(parameter), operation_value_parameters.len() == 1)
            }
        };
        let substituted_parameters = match &constant_parameter {
            Some(_) => operation_value_parameters.iter().skip(2).collect::<Vec<_>>(),
            None => Vec::new(),
        };
        let constant_path = constant_parameter.clone().map(syn::Path::from);
        let substitute_type = |ty: &syn::Type| {
            let mut ty = ty.clone();
            substituted_parameters
                .iter()
                .for_each(|parameter| ty.replace_ident(parameter, constant_path.as_ref().unwrap()));
            ty
        };
        let substitute_generics = |generics: &syn::Generics| {
            substituted_parameters.iter().fold(generics.clone(), |generics, parameter| {
                generics.with_concrete_param(parameter, constant_path.as_ref().unwrap())
            })
        };

        // Every payload must implement the operation contract of its class, and we add one predicate per distinct
        // payload contract. Family-native payloads use the primary operation type, projected member payloads use their
        // member type, and mixed member payloads use `MemberOperation` for their parent boundary.
        let mut payload_contracts = HashSet::new();
        let payload_predicates = variants
            .iter()
            .filter_map(|variant| {
                let payload_type = &variant.payload_type;
                let predicate: syn::WherePredicate = match &variant.class {
                    OperationVariantClass::FamilyNative => syn::parse_quote_spanned! { variant.span =>
                        #payload_type: #ryft::Operation<Type = #operation_type>
                    },
                    OperationVariantClass::ProjectedMember { member_type, .. } => {
                        syn::parse_quote_spanned!(variant.span => #payload_type: #ryft::Operation<Type = #member_type>)
                    }
                    OperationVariantClass::MixedMember { .. } => syn::parse_quote_spanned! { variant.span =>
                        #payload_type: #ryft::MemberOperation<#operation_type>
                    },
                };
                payload_contracts.insert(predicate.to_compact_string()).then_some(predicate)
            })
            .collect::<Vec<_>>();

        // Every projected member type must share the type identities of the primary operation type and must convert to
        // and from it, so that the projected boundary helpers can move types between the two universes.
        let mut member_types = HashSet::new();
        let member_predicates = variants
            .iter()
            .filter_map(|variant| variant.class.projected_member_type())
            .filter(|member_type| member_types.insert(member_type.to_compact_string()))
            .flat_map(|member_type| -> [syn::WherePredicate; 3] {
                [
                    syn::parse_quote! {
                        #member_type: #ryft::Type<
                            Identity = <#operation_type as #ryft::Type>::Identity,
                        >
                    },
                    syn::parse_quote!(#operation_type: ::std::convert::From<#member_type>),
                    syn::parse_quote! {
                        for<'__member_type> &'__member_type #member_type: ::std::convert::TryFrom<
                            &'__member_type #operation_type,
                            Error = #ryft::TypeError,
                        >
                    },
                ]
            })
            .collect::<Vec<_>>();

        // The interpretation, reference discharge, batching, and differentiation rule traits carry `Operation<Type =
        // C::Type>` as their super-trait, and so the per-payload rule predicates of those dispatchers already imply the
        // payload `Operation` predicates and do not restate them.
        let operation_generics =
            generics.with_where_predicates(&[payload_predicates, member_predicates.clone()].concat());
        let dispatcher_generics = substitute_generics(&generics.with_where_predicates(&member_predicates));
        let program_generics = substitute_generics(&operation_generics);
        let program_self_type = substitute_type(&self_type);
        variants.iter_mut().for_each(|variant| {
            variant.program_payload_type = substitute_type(&variant.payload_type);
            if let OperationVariantClass::ProjectedMember { member_type, .. }
            | OperationVariantClass::MixedMember { member_type, .. } = &mut variant.class
            {
                *member_type = substitute_type(member_type);
            }
        });
        let members = self
            .members
            .iter()
            .map(|member| OperationMember { r#type: substitute_type(&member.r#type), structural: member.structural })
            .collect();

        OperationCodeGenerator {
            ryft_crate: self.ryft_crate.clone(),
            operation_type,
            members,
            dispatchers: self.dispatchers.unwrap_or_default(),
            ident,
            generics,
            operation_generics,
            dispatcher_generics,
            program_generics,
            self_type,
            program_self_type,
            constant_type,
            constant_parameter,
            has_separate_runtime_value_type,
            variants,
        }
    }
}

/// Member universe of an operation family, which is declared through the enum-level `#[ryft(members(...))]` attribute
/// as either a bare type (for a computational member) or as `structural(Type)` (for a structural member).
struct OperationMember {
    /// Type of this member universe.
    r#type: syn::Type,

    /// Boolean flag indicating whether this member universe is structural, meaning that its values are bookkeeping
    /// with a zero differential space rather than differentiable data.
    structural: bool,
}

impl syn::parse::Parse for OperationMember {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        // A leading identifier followed by parentheses declares the role of the member type inside those parentheses,
        // while any other entry is the member type itself.
        let fork = input.fork();
        if fork.parse::<syn::Ident>().is_err() || !fork.peek(syn::token::Paren) {
            return Ok(Self { r#type: input.parse()?, structural: false });
        }
        let role = input.parse::<syn::Ident>()?;
        if role != STRUCTURAL_ROLE {
            return Err(syn::Error::new_spanned(
                &role,
                format_args!("invalid ryft member universe role `{role}`; only `structural` is supported here"),
            ));
        }
        let content;
        syn::parenthesized!(content in input);
        let r#type = content.parse::<syn::Type>()?;
        if !content.is_empty() {
            return Err(content.error("`structural(...)` declares exactly one member type"));
        }
        Ok(Self { r#type, structural: true })
    }
}

/// Optional implementations that `#[derive(Operation)]` generates when they are selected through
/// `#[ryft(dispatch(...))]`.
#[derive(Copy, Clone, Default)]
struct Dispatchers {
    /// Boolean flag indicating whether to generate payload-delegating [`PartialEq`], [`Eq`], and [`Hash`]
    /// implementations.
    identity: bool,

    /// Boolean flag indicating whether to generate the `ReferenceDischargeableOperation` dispatcher.
    discharge: bool,

    /// Boolean flag indicating whether to generate the `BatchableOperation` dispatcher.
    batching: bool,

    /// Boolean flag indicating whether to generate the `DifferentiableOperation` dispatcher.
    differentiation: bool,

    /// Boolean flag indicating whether to generate the `TransposableOperation` dispatcher.
    transposition: bool,
}

/// Parsed [`syn::Variant`] of an operation enum, together with the payload information that determines the generated
/// match arms, `where` predicates, and conversions.
struct OperationVariant {
    /// Identifier of this [`OperationVariant`].
    ident: syn::Ident,

    /// [`Span`] of the payload type of this [`OperationVariant`], which spans the generated code that is specific to
    /// this variant (e.g., its payload bounds and its match arms), so that compiler errors in that code point at the
    /// variant instead of at the `#[derive(Operation)]` attribute.
    span: Span,

    /// [`OperationVariantClass`] of this [`OperationVariant`].
    class: OperationVariantClass,

    /// Type of the payload of this [`OperationVariant`], which is the stored type without its `Box`, if it is boxed.
    payload_type: syn::Type,

    /// Program-shaped form of [`OperationVariant::payload_type`], which is used by the program-shaped implementations.
    /// Refer to [`OperationCodeGenerator`] for more information.
    program_payload_type: syn::Type,

    /// Boolean flag indicating whether the payload is stored as `Box<Payload>`.
    is_boxed: bool,

    /// Boolean flag indicating whether the payload is a bare generic type parameter of the enum (e.g., the `Extension`
    /// payload of an `Extension(Extension)` variant).
    is_generic_extension: bool,

    /// Boolean flag indicating whether to generate the owned `From<Payload>` conversion for this [`OperationVariant`].
    generate_from: bool,

    /// Boolean flag indicating whether the generated `CollectiveOperation` implementation delegates to the payload of
    /// this [`OperationVariant`], which is the case when the variant declares `#[ryft(collective)]`.
    is_collective: bool,

    /// Boolean flag indicating whether the generated `LockstepOperation` implementation delegates to the payload of
    /// this [`OperationVariant`], which is the case when the variant declares `#[ryft(lockstep)]`.
    is_lockstep: bool,
}

impl OperationVariant {
    /// Returns the expression that refers to the payload of this [`OperationVariant`] when it is bound as `operation`
    /// in a match arm, dereferencing through the box of boxed payloads.
    fn receiver(&self) -> TokenStream {
        if self.is_boxed { quote!(&**operation) } else { quote!(operation) }
    }

    /// Returns the expression that constructs this [`OperationVariant`] from the provided payload expression, boxing
    /// the payload if it is stored boxed.
    fn construct(&self, payload: TokenStream) -> TokenStream {
        let ident = &self.ident;
        if self.is_boxed {
            quote!(Self::#ident(::std::boxed::Box::new(#payload)))
        } else {
            quote!(Self::#ident(#payload))
        }
    }

    /// Returns a match arm that binds the payload of this [`OperationVariant`] as `operation` and evaluates `body`.
    fn arm(&self, body: TokenStream) -> TokenStream {
        let ident = &self.ident;
        quote_spanned!(self.span => Self::#ident(operation) => { #body },)
    }

    /// Returns a match arm that matches this [`OperationVariant`] without binding its payload and evaluates `body`.
    fn wildcard_arm(&self, body: TokenStream) -> TokenStream {
        let ident = &self.ident;
        quote_spanned!(self.span => Self::#ident(..) => { #body },)
    }

    /// Returns `true` if this [`OperationVariant`] stores a native guard of the enclosing family, which is a
    /// family-native payload, stored directly or boxed, whose type is a path whose last segment is `GuardOperation`.
    /// The detection is syntactic, because procedural macros cannot resolve types. Projected and mixed member payloads
    /// do not count, because their guards belong to the member universe rather than to the primary operation type, and
    /// neither do bare generic extension payloads. A payload that names the guard through a type alias is therefore not
    /// detected, and the generated rejecting provider then conflicts with the blanket provider that the generated
    /// `From` conversion of the variant enables (E0119), which is a compile error rather than a silent rejection.
    fn is_native_guard(&self) -> bool {
        matches!(self.class, OperationVariantClass::FamilyNative)
            && matches!(
                &self.payload_type,
                syn::Type::Path(syn::TypePath { qself: None, path })
                    if path.segments.last().is_some_and(|segment| segment.ident == "GuardOperation"),
            )
    }
}

/// Semantic class of an [`OperationVariant`], which determines how the generated implementations dispatch to its
/// payload. Member classes combine a _boundary shape_, which describes how the type boundary of the payload relates to
/// the enclosing universe, with a _transform role_, which describes whether the payload carries real transform rules
/// or is structural bookkeeping. Both member boundaries share the `class(U [, structural])` grammar, and so the role
/// is a flag of each boundary rather than a separate class.
enum OperationVariantClass {
    /// Variant whose payload already uses the primary operation type and the semantic contracts of the enclosing
    /// operation family.
    FamilyNative,

    /// Variant whose payload belongs to a member universe and whose complete instruction boundary belongs to that
    /// universe, which means that its inference, execution, and transform rules run in a projected context.
    ProjectedMember {
        /// Type of the projected member universe.
        member_type: syn::Type,

        /// Boolean flag indicating whether this payload is structural. A structural projected member has nothing to
        /// differentiate and nothing to batch per item, so forward mode stages a zero tangent and transposition
        /// contributes no cotangents instead of consulting a payload rule.
        structural: bool,
    },

    /// Variant whose payload belongs to a member universe but whose parent instruction has a genuinely mixed boundary
    /// that also consumes or produces values of other members of the enclosing universe.
    MixedMember {
        /// Type of the member universe that the payload belongs to natively.
        member_type: syn::Type,

        /// Boolean flag indicating whether this payload is structural, in which case forward mode stages a zero
        /// tangent instead of consulting a payload rule.
        structural: bool,
    },
}

impl OperationVariantClass {
    /// Parses the member class that a `projected(...)` or `mixed(...)` variant marker declares. Both markers share the
    /// grammar `class(U [, structural])`, where `U` is the member type of the payload and the optional `structural`
    /// token selects the structural transform role instead of the default computational role. A `mixed` marker may
    /// additionally use the abbreviated forms `mixed` and `mixed(structural)`, which take their member type from the
    /// single computational member universe that the enum declares through `#[ryft(members(...))]`. In the abbreviated
    /// role form, a lone `structural` identifier is therefore parsed as the role instead of as a member type.
    ///
    /// # Parameters
    ///
    ///   - `meta`: Nested `#[ryft(...)]` meta item whose path is either `projected` or `mixed`.
    ///   - `members`: [`OperationMember`]s declared by the enum, which are used to default and to validate
    ///     the member type.
    fn parse(meta: &syn::meta::ParseNestedMeta, members: &[OperationMember]) -> syn::Result<Self> {
        /// Returns the member type that an abbreviated `mixed` marker defaults to, which is the single computational
        /// member universe among the provided [`OperationMember`]s. Enums that declare no members, or that declare
        /// several computational members, have no such default and must name the member type explicitly.
        fn default_member_type<T: ToTokens>(tokens: T, members: &[OperationMember]) -> syn::Result<syn::Type> {
            let computational_members = members.iter().filter(|member| !member.structural).collect::<Vec<_>>();
            match computational_members.as_slice() {
                [member] => Ok(member.r#type.clone()),
                [] if members.is_empty() => Err(syn::Error::new_spanned(
                    tokens,
                    "the `mixed` operation variant class cannot default its member type because the operation family \
                     declares no `#[ryft(members(...))]`; declare the family's members or name the member type as \
                     `mixed(U)`",
                )),
                _ => Err(syn::Error::new_spanned(
                    tokens,
                    format_args!(
                        "the `mixed` operation variant class cannot default its member type because the operation \
                         family declares {} computational member universes; name the member type as `mixed(U)`",
                        computational_members.len(),
                    ),
                )),
            }
        }

        let is_mixed = meta.path == MIXED_ATTRIBUTE;
        let class = if is_mixed { MIXED_ATTRIBUTE } else { PROJECTED_ATTRIBUTE };
        let (member_type, structural) = if is_mixed && !meta.input.peek(syn::token::Paren) {
            (default_member_type(&meta.path, members)?, false)
        } else {
            let content;
            syn::parenthesized!(content in meta.input);
            let fork = content.fork();
            if is_mixed && fork.parse::<syn::Ident>().is_ok_and(|role| role == STRUCTURAL_ROLE) && fork.is_empty() {
                let role = content.parse::<syn::Ident>()?;
                (default_member_type(&role, members)?, true)
            } else {
                let member_type = content.parse::<syn::Type>()?;
                let structural = content.parse::<Option<syn::Token![,]>>()?.is_some();
                if structural {
                    let role = content.parse::<syn::Ident>()?;
                    if role != STRUCTURAL_ROLE {
                        return Err(syn::Error::new_spanned(
                            &role,
                            format_args!(
                                "invalid {class} operation variant role `{role}`; only `structural` is supported here",
                            ),
                        ));
                    }
                }
                if !content.is_empty() {
                    return Err(content.error(format_args!(
                        "the `{class}` operation variant class accepts one member type and an optional role",
                    )));
                }
                if !members.is_empty() && !members.iter().any(|member| member.r#type.tokens_eq(&member_type)) {
                    return Err(syn::Error::new_spanned(
                        &member_type,
                        format_args!(
                            "the `{}` operation variant class declares member type `{}`, which the operation \
                             family does not declare in `#[ryft(members(...))]`",
                            class,
                            member_type.to_compact_string(),
                        ),
                    ));
                }
                (member_type, structural)
            }
        };
        if is_mixed {
            Ok(Self::MixedMember { member_type, structural })
        } else {
            Ok(Self::ProjectedMember { member_type, structural })
        }
    }

    /// Returns the name of the variant marker that declares this [`OperationVariantClass`]. Family-native variants have
    /// no marker and so this function must only be called on member classes.
    fn name(&self) -> Symbol {
        match self {
            Self::FamilyNative => unreachable!("family-native variants have no class marker"),
            Self::ProjectedMember { .. } => PROJECTED_ATTRIBUTE,
            Self::MixedMember { .. } => MIXED_ATTRIBUTE,
        }
    }

    /// Returns the member type that this [`OperationVariantClass`] claims as the canonical `OperationProjection`
    /// target of the enclosing family, which is the member type of projected member classes, or [`None`] for all
    /// other classes. Mixed members belong to a member type without owning that projection.
    fn projected_member_type(&self) -> Option<&syn::Type> {
        match self {
            Self::FamilyNative | Self::MixedMember { .. } => None,
            Self::ProjectedMember { member_type, .. } => Some(member_type),
        }
    }
}

/// Returns the generic type parameters in the provided [`syn::Generics`] that are bounded by `Value<Type = T>`, each
/// paired with its type `T`, in parameter declaration order. Both inline parameter bounds and `where` predicates whose
/// bounded type is exactly the parameter count, and a parameter with several such bounds appears once per bound. Given
/// that procedural macros are executed before type checking and inference is performed by the Rust compiler, this
/// function recognizes the bounds by their final path segment being `Value` (e.g., `Value<Type = T>` or
/// `ryft::Value<Type = T>`).
fn value_parameters(generics: &syn::Generics) -> Vec<(&syn::Ident, syn::Type)> {
    /// Returns the type `T` of the provided `Value<Type = T>` bound, or [`None`] if it is any other bound.
    fn value_bound_type(bound: &syn::TypeParamBound) -> Option<syn::Type> {
        let syn::TypeParamBound::Trait(bound) = bound else {
            return None;
        };
        let segment = bound.path.segments.last()?;
        if segment.ident != "Value" {
            return None;
        }
        let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
            return None;
        };
        match arguments.args.iter().collect::<Vec<_>>().as_slice() {
            [syn::GenericArgument::AssocType(binding)] if binding.ident == "Type" => Some(binding.ty.clone()),
            _ => None,
        }
    }

    let where_predicates = generics
        .where_clause
        .iter()
        .flat_map(|where_clause| where_clause.predicates.iter())
        .filter_map(|predicate| match predicate {
            syn::WherePredicate::Type(predicate) => Some(predicate),
            _ => None,
        })
        .collect::<Vec<_>>();
    generics
        .type_params()
        .flat_map(|parameter| {
            let where_bounds = where_predicates
                .iter()
                .filter(|predicate| predicate.bounded_ty.matches_ident(&parameter.ident))
                .flat_map(|predicate| predicate.bounds.iter());
            parameter
                .bounds
                .iter()
                .chain(where_bounds)
                .filter_map(value_bound_type)
                .map(|value_type| (&parameter.ident, value_type))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Returns `true` if the provided [`syn::Type`] is `'static` for every instantiation of the provided [`syn::Generics`],
/// which holds when it mentions none of their type parameters and no lifetime. Such types can be named in `'static`
/// contexts, like `TypeId::of` and `downcast`, without any bounds on the generic parameters. This check is syntactic
/// and therefore conservative. That is, every identifier that matches a type parameter counts as a mention, and so
/// does every lifetime (including `'static`). Const parameters do not count, because they cannot make a type
/// non-`'static`, even though types that mention them may differ across instantiations.
fn is_unconditionally_static(ty: &syn::Type, generics: &syn::Generics) -> bool {
    /// Returns `true` if the provided tokens, including the tokens nested inside delimited groups, contain any of the
    /// provided parameter identifiers or any lifetime (which is recognized by its leading `'` punctuation).
    fn mentions(tokens: TokenStream, parameters: &[&syn::Ident]) -> bool {
        tokens.into_iter().any(|token| match token {
            proc_macro2::TokenTree::Group(group) => mentions(group.stream(), parameters),
            proc_macro2::TokenTree::Ident(ident) => parameters.contains(&&ident),
            proc_macro2::TokenTree::Punct(punct) => punct.as_char() == '\'',
            proc_macro2::TokenTree::Literal(_) => false,
        })
    }

    let parameters = generics.type_params().map(|parameter| &parameter.ident).collect::<Vec<_>>();
    !mentions(ty.to_token_stream(), &parameters)
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use quote::quote;

    use super::*;

    /// Parses the provided derive input and extracts its enum-level attributes, returning the resulting parser.
    fn extract_attributes(tokens: TokenStream) -> OperationParser {
        let mut parser = OperationParser::new();
        parser.extract_attributes(&syn::parse2(tokens).unwrap());
        parser
    }

    /// Parses the provided derive input and extracts its [`OperationCodeGenerator`] like
    /// [`OperationCodeGenerator::generate_operation_impl`] does, asserting that no errors are reported along the way.
    fn code_generator(tokens: TokenStream) -> OperationCodeGenerator {
        let mut input = syn::parse2(tokens).unwrap();
        replace_self_type(&mut input);
        let mut parser = OperationParser::new();
        parser.extract_attributes(&input);
        let generator = parser.extract_code_generator(&input);
        assert_eq!(errors(&parser), Vec::<String>::new());
        generator
    }

    /// Returns the messages of the errors that the provided [`OperationParser`] has accumulated.
    fn errors(parser: &OperationParser) -> Vec<String> {
        parser.errors.iter().map(syn::Error::to_string).collect()
    }

    /// Returns the rendering of the provided tokens without whitespace and without trailing commas, so that generated
    /// code can be compared against expected code that is written in the usual `rustfmt` style.
    fn rendering<T: ToTokens>(tokens: &T) -> String {
        let rendering =
            [",>", ",)", ",]", ",{", ",}"].iter().fold(tokens.to_compact_string(), |rendering, trailing_comma| {
                rendering.replace(trailing_comma, &trailing_comma[1..])
            });
        rendering.strip_suffix(',').map(str::to_string).unwrap_or(rendering)
    }

    /// Returns the rendering of the provided [`syn::Generics`], including their `where` clause, like [`rendering`].
    fn generics_rendering(generics: &syn::Generics) -> String {
        let where_clause = &generics.where_clause;
        rendering(&quote!(#generics #where_clause))
    }

    #[test]
    fn test_operation_parser_add_error_and_compile_error() {
        let mut parser = OperationParser::new();
        assert!(parser.compile_error().is_none());
        parser.add_error(quote!(variant_a), "first error");
        parser.add_error(quote!(variant_b), "second error");
        assert_eq!(
            parser.compile_error().unwrap().to_string(),
            quote!(::core::compile_error! { "first error" } ::core::compile_error! { "second error" }).to_string(),
        );
    }

    #[test]
    fn test_operation_parser_extract_attributes() {
        // Test inferring the operation type from the unique `Value<Type = T>` bound, which may be repeated.
        let parser = extract_attributes(quote! {
            #[ryft(crate = "wrapped::ryft")]
            #[ryft(dispatch(identity, batching))]
            enum Operation<V: Value<Type = DataType>, C> where C: ryft::Value<Type = DataType> {}
        });
        assert_eq!(errors(&parser), Vec::<String>::new());
        assert_eq!(parser.ryft_crate.to_compact_string(), "wrapped::ryft");
        assert_eq!(parser.operation_type.as_ref().unwrap().to_compact_string(), "DataType");
        assert!(parser.constant_type.is_none());
        assert!(parser.dispatchers.is_some());

        // Test declaring the operation type and the constant type explicitly, possibly across several attributes.
        let parser = extract_attributes(quote! {
            #[ryft(type(ArrayIrType))]
            #[ryft(constant(ArrayIrValue<A>), members(ArrayType))]
            enum Operation<A: Value<Type = ArrayType>> {}
        });
        assert_eq!(errors(&parser), Vec::<String>::new());
        assert_eq!(parser.ryft_crate.to_compact_string(), "ryft");
        assert_eq!(parser.operation_type.as_ref().unwrap().to_compact_string(), "ArrayIrType");
        assert_eq!(parser.constant_type.as_ref().unwrap().to_compact_string(), "ArrayIrValue<A>");
        assert_eq!(parser.members.len(), 1);
        assert!(parser.dispatchers.is_none());

        // Test using invalid and removed attributes.
        let parser = extract_attributes(quote! {
            #[ryft(crate = "ryft", unknown = "value", identity)]
            enum Operation<V: Value<Type = DataType>> {}
        });
        assert_eq!(
            errors(&parser),
            vec![
                "invalid `#[ryft(...)]` attribute `unknown`; only `crate`, `type(...)`, `constant(...)`, \
                 `members(...)`, and `dispatch(...)` are supported here",
            ],
        );
        let parser = extract_attributes(quote! {
            #[ryft(identity)]
            enum Operation<V: Value<Type = DataType>> {}
        });
        assert_eq!(
            errors(&parser),
            vec!["the `identity` attribute was replaced by the `dispatch(identity)` dispatcher"]
        );

        // Test failing to determine the operation type.
        let parser = extract_attributes(quote!(
            enum Operation<V> {}
        ));
        assert_eq!(
            errors(&parser),
            vec!["could not infer an operation type because no generic parameter is bounded by `Value<Type = T>`"],
        );
        let parser = extract_attributes(quote!(
            enum Operation<V: Value<Type = DataType>, A: Value<Type = ArrayType>> {}
        ));
        assert_eq!(
            errors(&parser),
            vec![
                "could not infer a unique operation type because multiple distinct `Value<Type = T>` bounds are \
                 present",
            ],
        );
        let parser = extract_attributes(quote! {
            #[ryft(type(ArrayIrType))]
            enum Operation<A: Value<Type = ArrayType>> {}
        });
        assert_eq!(
            errors(&parser),
            vec!["explicit operation-family declarations must specify both `type(...)` and `constant(...)`"],
        );
    }

    #[test]
    fn test_operation_parser_extract_type_attribute() {
        // Test declaring complete Rust types, including qualified paths with generic arguments and tuple types.
        let parser = extract_attributes(quote! {
            #[ryft(type(<A as Projection<ArrayType>>::Family), constant((ArrayIrValue<A>, Vec<u8>)))]
            enum Operation<A: Value<Type = ArrayType>> {}
        });
        assert_eq!(errors(&parser), Vec::<String>::new());
        assert_eq!(parser.operation_type.as_ref().unwrap().to_compact_string(), "<AasProjection<ArrayType>>::Family");
        assert_eq!(parser.constant_type.as_ref().unwrap().to_compact_string(), "(ArrayIrValue<A>,Vec<u8>)");

        // Test using invalid type attributes.
        for (attribute, error) in [
            (quote!(#[ryft(type = ArrayIrType)]), "the `type = ...` attribute form was replaced by `type(...)`"),
            (
                quote!(#[ryft(constant = ArrayIrValue)]),
                "the `constant = ...` attribute form was replaced by `constant(...)`",
            ),
            (quote!(#[ryft(type(DataType), type(ArrayType))]), "duplicate ryft attribute `type(...)`"),
            (quote!(#[ryft(type)]), "`#[ryft(type(...))]` must declare exactly one type"),
            (quote!(#[ryft(type())]), "`#[ryft(type(...))]` must declare exactly one type"),
            (quote!(#[ryft(type(DataType, ArrayType))]), "`#[ryft(type(...))]` must declare exactly one type"),
            (quote!(#[ryft(constant(1))]), "`#[ryft(constant(...))]` must declare exactly one type"),
        ] {
            let parser = extract_attributes(quote! {
                #attribute
                enum Operation<V: Value<Type = DataType>> {}
            });
            assert_eq!(errors(&parser)[0], error);
        }
    }

    #[test]
    fn test_operation_parser_extract_members_attribute() {
        // Test declaring members in declaration order, together with their roles.
        let parser = extract_attributes(quote! {
            #[ryft(type(ArrayIrType), constant(ArrayIrValue), members(ArrayType, structural(DimensionType),))]
            enum Operation {}
        });
        assert_eq!(errors(&parser), Vec::<String>::new());
        assert_eq!(
            parser
                .members
                .iter()
                .map(|member| (member.r#type.to_compact_string(), member.structural))
                .collect::<Vec<_>>(),
            vec![("ArrayType".to_string(), false), ("DimensionType".to_string(), true)],
        );

        // Test using invalid members attributes.
        for (attribute, error) in [
            (quote!(#[ryft(members())]), "`#[ryft(members(...))]` must declare at least one member universe"),
            (quote!(#[ryft(members(ArrayType, ArrayType))]), "duplicate ryft member universe `ArrayType`"),
            (quote!(#[ryft(members(ArrayType), members(DimensionType))]), "duplicate ryft attribute `members(...)`"),
            (
                quote!(#[ryft(members(replicated(DimensionType)))]),
                "invalid ryft member universe role `replicated`; only `structural` is supported here",
            ),
        ] {
            let parser = extract_attributes(quote! {
                #[ryft(type(ArrayIrType), constant(ArrayIrValue))]
                #attribute
                enum Operation {}
            });
            assert_eq!(errors(&parser), vec![error]);
        }
    }

    #[test]
    fn test_operation_parser_extract_dispatch_attribute() {
        let parser = extract_attributes(quote! {
            #[ryft(dispatch(identity, discharge, batching, differentiation, transposition))]
            enum Operation<V: Value<Type = DataType>> {}
        });
        let dispatchers = parser.dispatchers.unwrap();
        assert!(dispatchers.identity);
        assert!(dispatchers.discharge);
        assert!(dispatchers.batching);
        assert!(dispatchers.differentiation);
        assert!(dispatchers.transposition);

        let parser = extract_attributes(quote! {
            #[ryft(dispatch(transposition, identity))]
            enum Operation<V: Value<Type = DataType>> {}
        });
        let dispatchers = parser.dispatchers.unwrap();
        assert!(dispatchers.identity);
        assert!(!dispatchers.discharge);
        assert!(!dispatchers.batching);
        assert!(!dispatchers.differentiation);
        assert!(dispatchers.transposition);

        // Test using invalid dispatch attributes.
        for (attribute, error) in [
            (quote!(#[ryft(dispatch())]), "`#[ryft(dispatch(...))]` must select at least one dispatcher"),
            (
                quote!(#[ryft(dispatch(lowering))]),
                "invalid `#[ryft(dispatch(...))]` dispatcher `lowering`; only `identity`, `discharge`, `batching`, \
                 `differentiation`, and `transposition` are supported here",
            ),
            (quote!(#[ryft(dispatch(batching, batching))]), "duplicate ryft dispatcher `batching`"),
            (quote!(#[ryft(dispatch(batching), dispatch(transposition))]), "duplicate ryft attribute `dispatch(...)`"),
        ] {
            let parser = extract_attributes(quote! {
                #attribute
                enum Operation<V: Value<Type = DataType>> {}
            });
            assert_eq!(errors(&parser), vec![error]);
        }
    }

    #[test]
    fn test_operation_parser_extract_variants() {
        let mut parser = OperationParser::new();
        let variants = parser.extract_variants(&syn::parse_quote! {
            enum Operation<V: Value<Type = DataType>> {
                Zero(ZeroOperation<DataType>),
                Constant(ConstantOperation<V>),
            }
        });
        assert_eq!(errors(&parser), Vec::<String>::new());
        assert_eq!(
            variants.iter().map(|variant| variant.ident.to_string()).collect::<Vec<_>>(),
            vec!["Zero", "Constant"]
        );

        // Test using a struct, which is not supported.
        let mut parser = OperationParser::new();
        assert!(
            parser
                .extract_variants(&syn::parse_quote!(
                    struct Operation(ZeroOperation<DataType>);
                ))
                .is_empty()
        );
        assert_eq!(errors(&parser), vec!["the `#[derive(Operation)]` macro only supports enums"]);
    }

    #[test]
    fn test_operation_parser_extract_variant() {
        let mut parser = OperationParser::new();
        let input: syn::DeriveInput = syn::parse_quote! {
            enum Operation<V: Value<Type = DataType>, Extension> {
                Zero(ZeroOperation<DataType>),
                #[ryft(projected(ArrayType), collective, skip_from)]
                Array(std::boxed::Box<ArrayOperation<V>>),
                Extension(Extension),
                Unit,
                Named { operation: ZeroOperation<DataType> },
                Pair(ZeroOperation<DataType>, ZeroOperation<DataType>),
                Field(#[ryft(projected(ArrayType))] ZeroOperation<DataType>),
            }
        };
        let syn::Data::Enum(data) = &input.data else { unreachable!() };
        let variants = data
            .variants
            .iter()
            .map(|variant| parser.extract_variant(&input.generics, variant))
            .collect::<Vec<_>>();

        // Test extracting a plain payload, a boxed payload with a class marker, and a generic extension payload.
        let zero = variants[0].as_ref().unwrap();
        assert!(matches!(zero.class, OperationVariantClass::FamilyNative));
        assert_eq!(zero.payload_type.to_compact_string(), "ZeroOperation<DataType>");
        assert_eq!(zero.program_payload_type.to_compact_string(), "ZeroOperation<DataType>");
        assert!(!zero.is_boxed);
        assert!(!zero.is_generic_extension);
        assert!(zero.generate_from);
        assert!(!zero.is_collective);
        let array = variants[1].as_ref().unwrap();
        assert!(matches!(array.class, OperationVariantClass::ProjectedMember { structural: false, .. }));
        assert_eq!(array.payload_type.to_compact_string(), "ArrayOperation<V>");
        assert!(array.is_boxed);
        assert!(!array.is_generic_extension);
        assert!(!array.generate_from);
        assert!(array.is_collective);
        let extension = variants[2].as_ref().unwrap();
        assert!(!extension.is_boxed);
        assert!(extension.is_generic_extension);

        // Test using variants that are not tuple variants with exactly one payload field.
        assert!(variants[3..6].iter().all(Option::is_none));
        assert!(variants[6].is_some());
        assert_eq!(
            errors(&parser),
            vec![
                "operation enum variants must be tuple variants with exactly one payload field",
                "operation enum variants must be tuple variants with exactly one payload field",
                "operation enum variants must be tuple variants with exactly one payload field",
                "operation field attributes are not supported; put operation member-class attributes on the \
                 enclosing variant",
            ],
        );
    }

    #[test]
    fn test_operation_parser_extract_variant_attributes() {
        let mut parser = OperationParser::new();
        let variant: syn::Variant = syn::parse_quote!(Zero(ZeroOperation<DataType>));
        assert!(matches!(
            parser.extract_variant_attributes(&variant),
            (OperationVariantClass::FamilyNative, true, false, false),
        ));
        let variant: syn::Variant = syn::parse_quote! {
            #[ryft(skip_from)]
            #[ryft(mixed(ArrayType, structural), collective)]
            Zero(ZeroOperation<ArrayType>)
        };
        assert!(matches!(
            parser.extract_variant_attributes(&variant),
            (OperationVariantClass::MixedMember { structural: true, .. }, false, true, false),
        ));
        let variant: syn::Variant = syn::parse_quote!(#[ryft(lockstep)] While(WhileOperation<ArrayIrType>));
        assert!(matches!(
            parser.extract_variant_attributes(&variant),
            (OperationVariantClass::FamilyNative, true, false, true),
        ));
        assert_eq!(errors(&parser), Vec::<String>::new());

        // Test using invalid variant attributes.
        for (attribute, error) in [
            (quote!(#[ryft(skip_from, skip_from)]), "duplicate ryft operation variant attribute `skip_from`"),
            (quote!(#[ryft(collective, collective)]), "duplicate ryft operation variant attribute `collective`"),
            (quote!(#[ryft(lockstep, lockstep)]), "duplicate ryft operation variant attribute `lockstep`"),
            (
                quote!(#[ryft(projected(ArrayType), projected(ArrayType))]),
                "duplicate ryft operation variant class `projected`",
            ),
            (
                quote!(#[ryft(projected(ArrayType), mixed(ArrayType))]),
                "contradictory ryft operation variant classes `projected` and `mixed`",
            ),
            (
                quote!(#[ryft(structural(DimensionType))]),
                "the `structural` operation variant class was replaced by the `projected(U, structural)` role form",
            ),
            (
                quote!(#[ryft(transpose = custom)]),
                "invalid operation variant `#[ryft(...)]` attribute `transpose`; only `projected`, `mixed`, \
                 `collective`, `lockstep`, and `skip_from` are supported here",
            ),
        ] {
            let mut parser = OperationParser::new();
            parser.extract_variant_attributes(&syn::parse_quote!(#attribute Zero(ZeroOperation<DataType>)));
            assert_eq!(errors(&parser), vec![error]);
        }
    }

    #[test]
    fn test_operation_parser_check_for_conflicting_variants() {
        let mut input: syn::DeriveInput = syn::parse_quote! {
            #[ryft(type(ArrayIrType), constant(ArrayIrValue<V>))]
            enum Operation<V: Value<Type = ArrayType>, Extension> {
                First(FirstOperation),
                Boxed(Box<FirstOperation>),
                #[ryft(mixed(ArrayType))]
                Mixed(FirstOperation),
                Generic(GenericOperation<V>),
                OtherGeneric(GenericOperation<V>),
                OtherParameter(GenericOperation<Extension>),
                Extension(Extension),
                OtherExtension(Extension),
                #[ryft(projected(ArrayType))]
                Array(ArrayOperation<V>),
                #[ryft(projected(ArrayType, structural))]
                OtherArray(OtherArrayOperation),
            }
        };
        replace_self_type(&mut input);
        let mut parser = OperationParser::new();
        parser.extract_attributes(&input);
        parser.extract_code_generator(&input);

        // Payload types may only repeat across bare generic extension variants, which receive no conversions, and
        // projected member types may not repeat at all. Payload types that differ only in their generic arguments are
        // left to the coherence checks of the compiler.
        assert_eq!(
            errors(&parser),
            vec![
                "operation family holds payload type `FirstOperation` in more than one variant",
                "operation family holds payload type `FirstOperation` in more than one variant",
                "operation family holds payload type `GenericOperation<V>` in more than one variant",
                "operation family declares member type `ArrayType` more than once",
            ],
        );
    }

    #[test]
    fn test_operation_parser_extract_code_generator() {
        // Test normalizing an enum whose types are inferred, in which case value parameters after the first two are
        // substituted by the first one in the program-shaped generics, self type, and payload types.
        let operation = code_generator(quote! {
            enum Operation<V, C, Extension, F = V>
            where
                V: Value<Type = DataType>,
                C: Value<Type = DataType>,
                F: Value<Type = DataType>,
            {
                While(Box<WhileOperation<DataType, F, Self>>),
                Extension(Extension),
            }
        });
        assert_eq!(operation.operation_type.to_compact_string(), "DataType");
        assert_eq!(operation.constant_type.to_compact_string(), "V");
        assert_eq!(operation.constant_parameter.as_ref().unwrap().to_string(), "V");
        assert!(!operation.has_separate_runtime_value_type);
        assert_eq!(operation.generics.to_compact_string(), "<V,C,Extension,F>");
        assert_eq!(operation.self_type.to_compact_string(), "Operation<V,C,Extension,F>");
        assert_eq!(operation.program_self_type.to_compact_string(), "Operation<V,C,Extension,V>");
        assert_eq!(
            operation.variants[0].payload_type.to_compact_string(),
            "WhileOperation<DataType,F,Operation<V,C,Extension,F>>",
        );
        assert_eq!(
            operation.variants[0].program_payload_type.to_compact_string(),
            "WhileOperation<DataType,V,Operation<V,C,Extension,V>>",
        );
        assert_eq!(
            generics_rendering(&operation.operation_generics),
            rendering(&quote! {
                <V, C, Extension, F>
                where
                    V: Value<Type = DataType>,
                    C: Value<Type = DataType>,
                    F: Value<Type = DataType>,
                    WhileOperation<DataType, F, Operation<V, C, Extension, F>>: ryft::Operation<Type = DataType>,
                    Extension: ryft::Operation<Type = DataType>,
            }),
        );
        assert_eq!(
            generics_rendering(&operation.program_generics),
            rendering(&quote! {
                <V, C, Extension>
                where
                    V: Value<Type = DataType>,
                    C: Value<Type = DataType>,
                    V: Value<Type = DataType>,
                    WhileOperation<DataType, V, Operation<V, C, Extension, V>>: ryft::Operation<Type = DataType>,
                    Extension: ryft::Operation<Type = DataType>,
            }),
        );
        assert_eq!(
            generics_rendering(&operation.dispatcher_generics),
            rendering(&quote! {
                <V, C, Extension>
                where
                    V: Value<Type = DataType>,
                    C: Value<Type = DataType>,
                    V: Value<Type = DataType>,
            }),
        );

        // Test normalizing an enum whose types are declared explicitly, which has no constant parameter and whose
        // projected member types add type conversion predicates.
        let operation = code_generator(quote! {
            #[ryft(type(ArrayIrType), constant(ArrayIrValue<A>), members(ArrayType), dispatch(batching))]
            enum Operation<A: Value<Type = ArrayType>> {
                #[ryft(projected(ArrayType))]
                Array(ArrayOperation<A>),
            }
        });
        assert_eq!(operation.operation_type.to_compact_string(), "ArrayIrType");
        assert_eq!(operation.constant_type.to_compact_string(), "ArrayIrValue<A>");
        assert!(operation.constant_parameter.is_none());
        assert!(operation.has_separate_runtime_value_type);
        assert_eq!(operation.members.len(), 1);
        assert!(operation.dispatchers.batching);
        assert!(!operation.dispatchers.identity);
        assert_eq!(operation.program_self_type.to_compact_string(), "Operation<A>");
        assert_eq!(
            generics_rendering(&operation.operation_generics),
            rendering(&quote! {
                <A: Value<Type = ArrayType>>
                where
                    ArrayOperation<A>: ryft::Operation<Type = ArrayType>,
                    ArrayType: ryft::Type<Identity = <ArrayIrType as ryft::Type>::Identity>,
                    ArrayIrType: ::std::convert::From<ArrayType>,
                    for<'__member_type> &'__member_type ArrayType:
                        ::std::convert::TryFrom<&'__member_type ArrayIrType, Error = ryft::TypeError>,
            }),
        );
    }

    #[test]
    fn test_operation_code_generator_generate() {
        /// Returns the names of the traits that the provided `const` block of generated code implements.
        fn implemented_traits(tokens: TokenStream) -> Vec<String> {
            let generated: syn::ItemConst = syn::parse2(tokens).unwrap();
            let syn::Expr::Block(block) = *generated.expr else { unreachable!() };
            block
                .block
                .stmts
                .iter()
                .filter_map(|statement| match statement {
                    syn::Stmt::Item(syn::Item::Impl(implementation)) => implementation.trait_.as_ref(),
                    _ => None,
                })
                .map(|(_, path, _)| path.segments.last().unwrap().ident.to_string())
                .collect()
        }

        let operation = code_generator(quote! {
            enum Operation<V: Value<Type = DataType>> {
                Constant(ConstantOperation<V>),
            }
        });
        assert_eq!(
            implemented_traits(operation.generate()),
            vec![
                "Display",
                "Operation",
                "InterpretableOperation",
                "PartiallyEvaluatableOperation",
                "OperationPayloadProjection",
                "OperationProvider",
                "From",
                "TryFrom",
            ],
        );

        let operation = code_generator(quote! {
            #[ryft(type(ArrayIrType), constant(ArrayIrValue<A>))]
            #[ryft(dispatch(transposition, differentiation, batching, discharge, identity))]
            enum Operation<A: Value<Type = ArrayType>> {
                #[ryft(projected(ArrayType))]
                Array(ArrayOperation<A>),
            }
        });
        assert_eq!(
            implemented_traits(operation.generate()),
            vec![
                "Display",
                "PartialEq",
                "Eq",
                "Hash",
                "Operation",
                "ReferenceDischargeableOperation",
                "InterpretableOperation",
                "PartiallyEvaluatableOperation",
                "BatchableOperation",
                "DifferentiableOperation",
                "TransposableOperation",
                "OperationProjection",
                "OperationPayloadProjection",
                "OperationProvider",
                "From",
                "TryFrom",
            ],
        );
    }

    #[test]
    fn test_operation_code_generator_generate_identity_impls() {
        // The implementations are bounded by each distinct payload type rather than by the generic parameters, compare
        // mismatched variants as unequal, and hash the variant discriminant before the payload.
        let operation = code_generator(quote! {
            enum Operation<V: Value<Type = DataType>, Extension> {
                Zero(ZeroOperation<DataType>),
                Tag(Box<TagOperation<DataType>>),
                First(Extension),
                Second(Extension),
            }
        });
        assert_eq!(
            rendering(&operation.generate_identity_impls()),
            rendering(&quote! {
                #[automatically_derived]
                impl<V: Value<Type = DataType>, Extension> ::std::cmp::PartialEq for Operation<V, Extension>
                where
                    ZeroOperation<DataType>: ::std::cmp::PartialEq,
                    TagOperation<DataType>: ::std::cmp::PartialEq,
                    Extension: ::std::cmp::PartialEq,
                {
                    #[inline]
                    fn eq(&self, other: &Self) -> bool {
                        match (self, other) {
                            (Self::Zero(left), Self::Zero(right)) => left == right,
                            (Self::Tag(left), Self::Tag(right)) => left == right,
                            (Self::First(left), Self::First(right)) => left == right,
                            (Self::Second(left), Self::Second(right)) => left == right,
                            _ => false,
                        }
                    }
                }

                #[automatically_derived]
                impl<V: Value<Type = DataType>, Extension> ::std::cmp::Eq for Operation<V, Extension>
                where
                    ZeroOperation<DataType>: ::std::cmp::Eq,
                    TagOperation<DataType>: ::std::cmp::Eq,
                    Extension: ::std::cmp::Eq,
                {}

                #[automatically_derived]
                impl<V: Value<Type = DataType>, Extension> ::std::hash::Hash for Operation<V, Extension>
                where
                    ZeroOperation<DataType>: ::std::hash::Hash,
                    TagOperation<DataType>: ::std::hash::Hash,
                    Extension: ::std::hash::Hash,
                {
                    #[inline]
                    fn hash<__H: ::std::hash::Hasher>(&self, state: &mut __H) {
                        ::std::hash::Hash::hash(&::std::mem::discriminant(self), state);
                        match self {
                            Self::Zero(operation) => ::std::hash::Hash::hash(operation, state),
                            Self::Tag(operation) => ::std::hash::Hash::hash(operation, state),
                            Self::First(operation) => ::std::hash::Hash::hash(operation, state),
                            Self::Second(operation) => ::std::hash::Hash::hash(operation, state),
                        }
                    }
                }
            }),
        );

        // A single-variant family has no mismatched-variant arm, which would otherwise be unreachable.
        let operation = code_generator(quote! {
            enum Operation<V: Value<Type = DataType>> {
                Zero(ZeroOperation<DataType>),
            }
        });
        assert!(rendering(&operation.generate_identity_impls()).contains(&rendering(&quote! {
            match (self, other) {
                (Self::Zero(left), Self::Zero(right)) => left == right,
            }
        })));
    }

    #[test]
    fn test_operation_code_generator_generate_operation_dispatcher() {
        let operation = code_generator(quote! {
            #[ryft(type(ArrayIrType), constant(ArrayIrValue<A>))]
            enum Operation<A: Value<Type = ArrayType>> {
                Add(AddOperation<ArrayIrType>),
                #[ryft(projected(ArrayType))]
                Array(Box<ArrayOperation<A>>),
                #[ryft(mixed(ArrayType))]
                Gather(GatherOperation),
            }
        });
        let generated: syn::ItemImpl = syn::parse2(operation.generate_operation_dispatcher()).unwrap();
        let function = |name: &str| {
            generated
                .items
                .iter()
                .find_map(|item| match item {
                    syn::ImplItem::Fn(function) if function.sig.ident == name => Some(rendering(function)),
                    _ => None,
                })
                .unwrap()
        };
        assert_eq!(rendering(&generated.generics.where_clause), rendering(&operation.operation_generics.where_clause));
        assert_eq!(generated.items.len(), 20);

        // Functions that do not see the instruction boundary forward to the payload of every variant directly.
        assert_eq!(
            function("name"),
            rendering(&quote! {
                #[inline]
                fn name(&self) -> &'static str {
                    match self {
                        Self::Add(operation) => { <AddOperation<ArrayIrType> as ryft::Operation>::name(operation) },
                        Self::Array(operation) => { <ArrayOperation<A> as ryft::Operation>::name(&**operation) },
                        Self::Gather(operation) => { <GatherOperation as ryft::Operation>::name(operation) },
                    }
                }
            }),
        );

        // Functions that see the instruction boundary use the contract of the universe of each payload.
        assert_eq!(
            function("infer_output_types"),
            rendering(&quote! {
                #[inline]
                fn infer_output_types(
                    &self,
                    input_types: &[ArrayIrType],
                    region_interfaces: &[ryft::RegionInterface<ArrayIrType>],
                ) -> ::std::result::Result<::std::vec::Vec<ArrayIrType>, ryft::TypeError> {
                    match self {
                        Self::Add(operation) => {
                            <AddOperation<ArrayIrType> as ryft::Operation>::infer_output_types(
                                operation,
                                input_types,
                                region_interfaces,
                            )
                        },
                        Self::Array(operation) => {
                            ryft::infer_projected_operation_output_types(&**operation, input_types, region_interfaces)
                        },
                        Self::Gather(operation) => {
                            <GatherOperation as ryft::MemberOperation<ArrayIrType>>::infer_parent_output_types(
                                operation,
                                input_types,
                                region_interfaces,
                            )
                        },
                    }
                }
            }),
        );
        assert_eq!(
            function("defined_type_identities"),
            rendering(&quote! {
                #[inline]
                fn defined_type_identities(&self) -> ::std::borrow::Cow<'_, [<ArrayIrType as ryft::Type>::Identity]> {
                    match self {
                        Self::Add(operation) => {
                            <AddOperation<ArrayIrType> as ryft::Operation>::defined_type_identities(operation)
                        },
                        Self::Array(operation) => {
                            <ArrayOperation<A> as ryft::Operation>::defined_type_identities(&**operation)
                        },
                        Self::Gather(operation) => {
                            <GatherOperation as ryft::MemberOperation<ArrayIrType>>::parent_defined_type_identities(
                                operation,
                            )
                        },
                    }
                }
            }),
        );

        // Functions that return operations wrap the returned payloads back into their (possibly boxed) variants.
        assert_eq!(
            function("prune_boundary"),
            rendering(&quote! {
                #[inline]
                fn prune_boundary(
                    &self,
                    input_types: &[ArrayIrType],
                    used_outputs: &[bool],
                    regions: &mut dyn ryft::RegionLiveness,
                ) -> ::std::result::Result<
                    ::std::option::Option<ryft::OperationBoundaryPruning<Self>>,
                    ryft::ProgramError,
                > {
                    match self {
                        Self::Add(operation) => {
                            ::std::result::Result::Ok(
                                <AddOperation<ArrayIrType> as ryft::Operation>::prune_boundary(
                                    operation,
                                    input_types,
                                    used_outputs,
                                    regions,
                                )?
                                .map(|pruning| pruning.map_operation(|operation| Self::Add(operation))),
                            )
                        },
                        Self::Array(operation) => {
                            ::std::result::Result::Ok(
                                ryft::prune_projected_operation_boundary(
                                    &**operation,
                                    input_types,
                                    used_outputs,
                                    regions,
                                )?
                                .map(|pruning| pruning.map_operation(
                                    |operation| Self::Array(::std::boxed::Box::new(operation)),
                                )),
                            )
                        },
                        Self::Gather(operation) => {
                            ::std::result::Result::Ok(
                                <GatherOperation as ryft::MemberOperation<ArrayIrType>>::prune_parent_boundary(
                                    operation,
                                    input_types,
                                    used_outputs,
                                    regions,
                                )?
                                .map(|pruning| pruning.map_operation(|operation| Self::Gather(operation))),
                            )
                        },
                    }
                }
            }),
        );
        assert_eq!(
            function("rename_type_identities"),
            rendering(&quote! {
                #[inline]
                fn rename_type_identities(
                    &self,
                    renaming: &ryft::TypeIdentityRenaming<<ArrayIrType as ryft::Type>::Identity>,
                ) -> ::std::result::Result<Self, ryft::TypeError> {
                    match self {
                        Self::Add(operation) => {
                            ::std::result::Result::Ok(Self::Add(
                                <AddOperation<ArrayIrType> as ryft::Operation>::rename_type_identities(
                                    operation,
                                    renaming,
                                )?,
                            ))
                        },
                        Self::Array(operation) => {
                            ::std::result::Result::Ok(Self::Array(::std::boxed::Box::new(
                                <ArrayOperation<A> as ryft::Operation>::rename_type_identities(&**operation, renaming)?,
                            )))
                        },
                        Self::Gather(operation) => {
                            ::std::result::Result::Ok(Self::Gather(
                                <GatherOperation as ryft::MemberOperation<ArrayIrType>>::rename_parent_type_identities(
                                    operation,
                                    renaming,
                                )?,
                            ))
                        },
                    }
                }
            }),
        );
    }

    #[test]
    fn test_operation_code_generator_generate_reference_discharge_dispatcher() {
        // Native and generic extension payloads delegate to their own discharge rules, while member payloads replay
        // the complete enum through the reference-free replay path.
        let operation = code_generator(quote! {
            #[ryft(type(ArrayIrType), constant(ArrayIrValue<A>))]
            enum Operation<A: Value<Type = ArrayType>, Extension> {
                Add(AddOperation<ArrayIrType>),
                #[ryft(projected(ArrayType))]
                Array(ArrayOperation<A>),
                Extension(Extension),
            }
        });
        assert_eq!(
            rendering(&operation.generate_reference_discharge_dispatcher()),
            rendering(&quote! {
                #[automatically_derived]
                impl<A: Value<Type = ArrayType>, Extension, __ParentContext, __ReferenceDischargePolicy>
                    ryft::ReferenceDischargeableOperation<__ParentContext, __ReferenceDischargePolicy>
                    for Operation<A, Extension>
                where
                    ArrayType: ryft::Type<Identity = <ArrayIrType as ryft::Type>::Identity>,
                    ArrayIrType: ::std::convert::From<ArrayType>,
                    for<'__member_type> &'__member_type ArrayType:
                        ::std::convert::TryFrom<&'__member_type ArrayIrType, Error = ryft::TypeError>,
                    __ParentContext: ryft::Context<Type = ArrayIrType, Operation = Operation<A, Extension>>,
                    Operation<A, Extension>: ryft::Operation<Type = ArrayIrType>,
                    __ReferenceDischargePolicy: ryft::ReferenceDischargePolicy<__ParentContext>,
                    AddOperation<ArrayIrType>:
                        ryft::ReferenceDischargeableOperation<__ParentContext, __ReferenceDischargePolicy>,
                    Extension: ryft::ReferenceDischargeableOperation<__ParentContext, __ReferenceDischargePolicy>,
                {
                    #[inline]
                    fn discharge_references<
                        __D: ryft::ReferenceDischargeDriver<__ParentContext, __ReferenceDischargePolicy>,
                    >(
                        &self,
                        context: &ryft::ReferenceDischargeContext<__ParentContext, __ReferenceDischargePolicy>,
                        driver: &__D,
                        inputs: &[ryft::ReferenceDischargeValue<__ParentContext, __ReferenceDischargePolicy>],
                    ) -> ::std::result::Result<
                        ::std::vec::Vec<ryft::ReferenceDischargeValue<__ParentContext, __ReferenceDischargePolicy>>,
                        ryft::ProgramError,
                    > {
                        match self {
                            Self::Add(operation) => {
                                <AddOperation<ArrayIrType> as ryft::ReferenceDischargeableOperation<
                                    __ParentContext,
                                    __ReferenceDischargePolicy,
                                >>::discharge_references(operation, context, driver, inputs)
                            },
                            Self::Array(..) => {
                                context.discharge_reference_free_operation(self, driver, inputs)
                            },
                            Self::Extension(operation) => {
                                <Extension as ryft::ReferenceDischargeableOperation<
                                    __ParentContext,
                                    __ReferenceDischargePolicy,
                                >>::discharge_references(operation, context, driver, inputs)
                            },
                        }
                    }
                }
            }),
        );
    }

    #[test]
    fn test_operation_code_generator_generate_interpretation_dispatcher() {
        let operation = code_generator(quote! {
            #[ryft(type(ArrayIrType), constant(ArrayIrValue<A>))]
            enum Operation<A: Value<Type = ArrayType>> {
                Add(AddOperation<ArrayIrType>),
                #[ryft(projected(ArrayType))]
                Array(Box<ArrayOperation<A>>),
                #[ryft(mixed(ArrayType))]
                Gather(GatherOperation),
            }
        });
        assert_eq!(
            rendering(&operation.generate_interpretation_dispatcher()),
            rendering(&quote! {
                #[automatically_derived]
                impl<A: Value<Type = ArrayType>, __Context, __InterpretationValue>
                    ryft::InterpretableOperation<__Context> for Operation<A>
                where
                    ArrayType: ryft::Type<Identity = <ArrayIrType as ryft::Type>::Identity>,
                    ArrayIrType: ::std::convert::From<ArrayType>,
                    for<'__member_type> &'__member_type ArrayType:
                        ::std::convert::TryFrom<&'__member_type ArrayIrType, Error = ryft::TypeError>,
                    __Context: ryft::Domain<Type = ArrayIrType, Value = __InterpretationValue>,
                    __Context: ryft::Constant<__InterpretationValue, ArrayIrValue<A>>,
                    AddOperation<ArrayIrType>: ryft::InterpretableOperation<__Context>,
                    __InterpretationValue: ryft::ValueProjection<ArrayType, Projected: ryft::Value<Type = ArrayType>>,
                    ryft::EagerContext<
                        <__InterpretationValue as ryft::ValueProjection<ArrayType>>::Projected,
                        ArrayOperation<A>,
                    >: ryft::Domain<
                        Type = ArrayType,
                        Value = <__InterpretationValue as ryft::ValueProjection<ArrayType>>::Projected,
                        Constant = <__InterpretationValue as ryft::ValueProjection<ArrayType>>::Projected,
                        Operation = ArrayOperation<A>,
                    >,
                    ArrayOperation<A>: ryft::InterpretableOperation<
                        ryft::EagerContext<
                            <__InterpretationValue as ryft::ValueProjection<ArrayType>>::Projected,
                            ArrayOperation<A>,
                        >,
                    >,
                    GatherOperation: ryft::MemberInterpretableOperation<__Context>,
                {
                    #[inline]
                    fn interpret<__D: ryft::InterpretationDriver<__Context>>(
                        &self,
                        context: &__Context,
                        driver: &__D,
                        inputs: &[<__Context as ryft::Domain>::Value],
                    ) -> ::std::result::Result<
                        ::std::vec::Vec<<__Context as ryft::Domain>::Value>,
                        ryft::ProgramError,
                    > {
                        match self {
                            Self::Add(operation) => {
                                <AddOperation<ArrayIrType> as ryft::InterpretableOperation<__Context>>::interpret(
                                    operation,
                                    context,
                                    driver,
                                    inputs,
                                )
                            },
                            Self::Array(operation) => {
                                ryft::interpret_projected_operation(context, &**operation, driver, inputs)
                            },
                            Self::Gather(operation) => {
                                <GatherOperation as ryft::MemberInterpretableOperation<__Context>>::interpret_in_parent(
                                    operation,
                                    context,
                                    driver,
                                    inputs,
                                )
                            },
                        }
                    }
                }
            }),
        );

        // Test generating the dispatcher of an enum whose two inferred value parameters make the constant type the
        // value type of the interpretation context.
        let operation = code_generator(quote! {
            enum Operation<V: Value<Type = ArrayType>, C: Value<Type = ArrayType>> {
                Constant(ConstantOperation<V>),
            }
        });
        let generated: syn::ItemImpl = syn::parse2(operation.generate_interpretation_dispatcher()).unwrap();
        assert_eq!(
            generics_rendering(&generated.generics),
            rendering(&quote! {
                <V: Value<Type = ArrayType>, C: Value<Type = ArrayType>, __Context>
                where
                    __Context: ryft::Domain<Type = ArrayType, Value = V>,
                    ConstantOperation<V>: ryft::InterpretableOperation<__Context>,
            }),
        );
    }

    #[test]
    fn test_operation_code_generator_generate_partial_evaluation_dispatcher() {
        let operation = code_generator(quote! {
            #[ryft(type(ArrayIrType), constant(ArrayIrValue<A>))]
            enum Operation<A: Value<Type = ArrayType>, Extension> {
                Add(AddOperation<ArrayIrType>),
                #[ryft(projected(ArrayType))]
                Array(ArrayOperation<A>),
                Extension(Extension),
            }
        });
        assert_eq!(
            rendering(&operation.generate_partial_evaluation_dispatcher()),
            rendering(&quote! {
                #[automatically_derived]
                impl<A: Value<Type = ArrayType>, Extension, __Context> ryft::PartiallyEvaluatableOperation<__Context>
                    for Operation<A, Extension>
                where
                    AddOperation<ArrayIrType>: ryft::Operation<Type = ArrayIrType>,
                    ArrayOperation<A>: ryft::Operation<Type = ArrayType>,
                    Extension: ryft::Operation<Type = ArrayIrType>,
                    ArrayType: ryft::Type<Identity = <ArrayIrType as ryft::Type>::Identity>,
                    ArrayIrType: ::std::convert::From<ArrayType>,
                    for<'__member_type> &'__member_type ArrayType:
                        ::std::convert::TryFrom<&'__member_type ArrayIrType, Error = ryft::TypeError>,
                    __Context: ryft::Context<
                        Type = ArrayIrType,
                        Constant = ArrayIrValue<A>,
                        Operation = Operation<A, Extension>,
                    >,
                    AddOperation<ArrayIrType>: ryft::PartiallyEvaluatableOperation<__Context>,
                {
                    #[inline]
                    fn partially_evaluate<__D: ryft::PartialEvaluationDriver<__Context>>(
                        &self,
                        context: &ryft::PartialEvaluationContext<__Context>,
                        driver: &__D,
                        inputs: &[ryft::PartialEvaluationValue<<__Context as ryft::Domain>::Value>],
                    ) -> ::std::result::Result<
                        ::std::vec::Vec<ryft::PartialEvaluationValue<<__Context as ryft::Domain>::Value>>,
                        ryft::ProgramError,
                    > {
                        match self {
                            Self::Add(operation) => {
                                <AddOperation<ArrayIrType> as ryft::PartiallyEvaluatableOperation<__Context>>::
                                    partially_evaluate(operation, context, driver, inputs)
                            },
                            Self::Array(..) => {
                                context.fold_or_residualize(self.clone(), driver.region_programs(), inputs)
                            },
                            Self::Extension(..) => {
                                context.fold_or_residualize(self.clone(), driver.region_programs(), inputs)
                            },
                        }
                    }

                    #[inline]
                    fn split_on_residual_demand<__D: ryft::PartialEvaluationDriver<__Context>>(
                        &self,
                        context: &ryft::PartialEvaluationContext<__Context>,
                        driver: &__D,
                        inputs: &[ryft::PartialEvaluationValue<<__Context as ryft::Domain>::Value>],
                        demanded_outputs: &[bool],
                    ) -> ::std::result::Result<
                        ::std::option::Option<
                            ryft::ResidualDemandSplit<<__Context as ryft::Domain>::Value>
                        >,
                        ryft::ProgramError,
                    >
                    where
                        <__Context as ryft::Domain>::Operation: ryft::PartiallyEvaluatableOperation<
                            ryft::TracingContext<
                                <__Context as ryft::Domain>::Constant,
                                <__Context as ryft::Domain>::Operation,
                            >,
                        >,
                    {
                        let _ = (context, driver, inputs, demanded_outputs);
                        match self {
                            Self::Add(operation) => {
                                <AddOperation<ArrayIrType> as ryft::PartiallyEvaluatableOperation<__Context>>::
                                    split_on_residual_demand(operation, context, driver, inputs, demanded_outputs)
                            },
                            Self::Array(..) => {
                                ::std::result::Result::Ok(::std::option::Option::None)
                            },
                            Self::Extension(..) => {
                                ::std::result::Result::Ok(::std::option::Option::None)
                            },
                        }
                    }
                }
            }),
        );
    }

    #[test]
    fn test_operation_code_generator_generate_batching_dispatcher() {
        // Test generating the dispatcher of an enum with an explicit constant type, which uses the canonical batching
        // policy of its primary type. Computational projected members apply their member batching rules under the
        // projected policy, while structural projected members are bound once under the replicated projection.
        let operation = code_generator(quote! {
            #[ryft(type(ArrayIrType), constant(ArrayIrValue<A>))]
            enum Operation<A: Value<Type = ArrayType>> {
                Add(AddOperation<ArrayIrType>),
                #[ryft(projected(ArrayType))]
                Array(ArrayOperation<A>),
                #[ryft(projected(DimensionType, structural))]
                Dimension(DimensionOperation),
                #[ryft(mixed(ArrayType))]
                Gather(GatherOperation),
            }
        });
        assert_eq!(
            rendering(&operation.generate_batching_dispatcher()),
            rendering(&quote! {
                #[automatically_derived]
                impl<A: Value<Type = ArrayType>, __ParentContext, __BatchingPolicy>
                    ryft::BatchableOperation<__ParentContext, __BatchingPolicy> for Operation<A>
                where
                    ArrayType: ryft::Type<Identity = <ArrayIrType as ryft::Type>::Identity>,
                    ArrayIrType: ::std::convert::From<ArrayType>,
                    for<'__member_type> &'__member_type ArrayType:
                        ::std::convert::TryFrom<&'__member_type ArrayIrType, Error = ryft::TypeError>,
                    DimensionType: ryft::Type<Identity = <ArrayIrType as ryft::Type>::Identity>,
                    ArrayIrType: ::std::convert::From<DimensionType>,
                    for<'__member_type> &'__member_type DimensionType:
                        ::std::convert::TryFrom<&'__member_type ArrayIrType, Error = ryft::TypeError>,
                    __ParentContext:
                        ryft::Context<Type = ArrayIrType, Constant = ArrayIrValue<A>, Operation = Operation<A>>,
                    Operation<A>: ryft::Operation<Type = ArrayIrType>,
                    ArrayIrType: ryft::BatchableType<Policy = __BatchingPolicy>,
                    __BatchingPolicy: ryft::BatchingPolicy<__ParentContext>,
                    AddOperation<ArrayIrType>: ryft::BatchableOperation<__ParentContext, __BatchingPolicy>,
                    ArrayIrValue<A>: ryft::ValueProjection<ArrayType, Projected: ryft::Value<Type = ArrayType>>,
                    <__ParentContext as ryft::Domain>::Constant:
                        ryft::ValueProjection<ArrayType, Projected: ryft::Value<Type = ArrayType>>,
                    <__ParentContext as ryft::Domain>::Value:
                        ryft::ValueProjection<ArrayType, Projected: ryft::Value<Type = ArrayType>>,
                    <__ParentContext as ryft::Domain>::Operation:
                        ryft::OperationProjection<ArrayType, Projected = ArrayOperation<A>>,
                    __BatchingPolicy: ryft::BatchingPolicyProjection<__ParentContext, ArrayType>,
                    ryft::ProjectedContext<__ParentContext, ArrayType>: ryft::Context<Type = ArrayType>,
                    ArrayOperation<A>: ryft::BatchableOperation<
                        ryft::ProjectedContext<__ParentContext, ArrayType>,
                        <__BatchingPolicy as ryft::BatchingPolicyProjection<__ParentContext, ArrayType>>::Projected,
                    >,
                    ArrayIrValue<A>:
                        ryft::ValueProjection<DimensionType, Projected: ryft::Value<Type = DimensionType>>,
                    <__ParentContext as ryft::Domain>::Constant:
                        ryft::ValueProjection<DimensionType, Projected: ryft::Value<Type = DimensionType>>,
                    <__ParentContext as ryft::Domain>::Value:
                        ryft::ValueProjection<DimensionType, Projected: ryft::Value<Type = DimensionType>>,
                    <__ParentContext as ryft::Domain>::Operation:
                        ryft::OperationProjection<DimensionType, Projected = DimensionOperation>,
                    __BatchingPolicy: ryft::ReplicatedBatchingPolicyProjection<__ParentContext, DimensionType>,
                    GatherOperation: ryft::MemberBatchableOperation<__ParentContext, __BatchingPolicy>,
                    __ParentContext: ryft::Zero<<__ParentContext as ryft::Domain>::Value>,
                {
                    #[inline]
                    fn batch<__D: ryft::BatchingDriver<__ParentContext, __BatchingPolicy>>(
                        &self,
                        context: &ryft::BatchingContext<__ParentContext, __BatchingPolicy>,
                        driver: &__D,
                        inputs: &[<__BatchingPolicy as ryft::BatchingPolicy<__ParentContext>>::Batch],
                    ) -> ::std::result::Result<
                        ::std::vec::Vec<<__BatchingPolicy as ryft::BatchingPolicy<__ParentContext>>::Batch>,
                        ryft::BatchingError,
                    > {
                        match self {
                            Self::Add(operation) => {
                                <AddOperation<ArrayIrType> as ryft::BatchableOperation<
                                    __ParentContext,
                                    __BatchingPolicy,
                                >>::batch(operation, context, driver, inputs)
                            },
                            Self::Array(operation) => {
                                context.batch_projected_operation(operation, inputs)
                            },
                            Self::Dimension(operation) => {
                                context.batch_replicated_projected_operation::<DimensionType>(
                                    operation,
                                    inputs,
                                )
                            },
                            Self::Gather(operation) => {
                                <GatherOperation as ryft::MemberBatchableOperation<
                                    __ParentContext,
                                    __BatchingPolicy,
                                >>::batch_in_parent(operation, context, driver, inputs)
                            },
                        }
                    }
                }
            }),
        );

        // Test generating the dispatcher of an enum whose constant type is inferred, which stays generic over its
        // batching policy.
        let operation = code_generator(quote! {
            enum Operation<V: Value<Type = ArrayType>> {
                Constant(ConstantOperation<V>),
            }
        });
        assert_eq!(
            rendering(&operation.generate_batching_dispatcher()),
            rendering(&quote! {
                #[automatically_derived]
                impl<V: Value<Type = ArrayType>, __ParentContext, __BatchingPolicy>
                    ryft::BatchableOperation<__ParentContext, __BatchingPolicy> for Operation<V>
                where
                    __ParentContext: ryft::Context<Type = ArrayType, Constant = V, Operation = Operation<V>>,
                    Operation<V>: ryft::Operation<Type = ArrayType>,
                    __BatchingPolicy: ryft::BatchingPolicy<__ParentContext>,
                    ConstantOperation<V>: ryft::BatchableOperation<__ParentContext, __BatchingPolicy>,
                    __ParentContext: ryft::Zero<<__ParentContext as ryft::Domain>::Value>,
                {
                    #[inline]
                    fn batch<__D: ryft::BatchingDriver<__ParentContext, __BatchingPolicy>>(
                        &self,
                        context: &ryft::BatchingContext<__ParentContext, __BatchingPolicy>,
                        driver: &__D,
                        inputs: &[<__BatchingPolicy as ryft::BatchingPolicy<__ParentContext>>::Batch],
                    ) -> ::std::result::Result<
                        ::std::vec::Vec<<__BatchingPolicy as ryft::BatchingPolicy<__ParentContext>>::Batch>,
                        ryft::BatchingError,
                    > {
                        match self {
                            Self::Constant(operation) => {
                                <ConstantOperation<V> as ryft::BatchableOperation<
                                    __ParentContext,
                                    __BatchingPolicy,
                                >>::batch(operation, context, driver, inputs)
                            },
                        }
                    }
                }
            }),
        );
    }

    #[test]
    fn test_operation_code_generator_generate_differentiation_dispatcher() {
        // Native payloads use their own forward mode rules, computational member payloads use their parent-universe
        // rules, and structural projected payloads stage their primals with symbolic zero tangents.
        let operation = code_generator(quote! {
            #[ryft(type(ArrayIrType), constant(ArrayIrValue<A>))]
            enum Operation<A: Value<Type = ArrayType>> {
                Add(AddOperation<ArrayIrType>),
                #[ryft(projected(ArrayType))]
                Array(ArrayOperation<A>),
                #[ryft(projected(DimensionType, structural))]
                Dimension(DimensionOperation),
                #[ryft(mixed(ArrayType))]
                Gather(GatherOperation),
            }
        });
        let generated: syn::ItemImpl = syn::parse2(operation.generate_differentiation_dispatcher()).unwrap();
        assert_eq!(
            generics_rendering(&generated.generics),
            rendering(&quote! {
                <A: Value<Type = ArrayType>, __DifferentiationContext>
                where
                    ArrayType: ryft::Type<Identity = <ArrayIrType as ryft::Type>::Identity>,
                    ArrayIrType: ::std::convert::From<ArrayType>,
                    for<'__member_type> &'__member_type ArrayType:
                        ::std::convert::TryFrom<&'__member_type ArrayIrType, Error = ryft::TypeError>,
                    DimensionType: ryft::Type<Identity = <ArrayIrType as ryft::Type>::Identity>,
                    ArrayIrType: ::std::convert::From<DimensionType>,
                    for<'__member_type> &'__member_type DimensionType:
                        ::std::convert::TryFrom<&'__member_type ArrayIrType, Error = ryft::TypeError>,
                    __DifferentiationContext:
                        ryft::Context<Type = ArrayIrType, Constant = ArrayIrValue<A>, Operation = Operation<A>>,
                    Operation<A>: ryft::Operation<Type = ArrayIrType>,
                    __DifferentiationContext: ryft::Zero<<__DifferentiationContext as ryft::Domain>::Value>,
                    <__DifferentiationContext as ryft::Domain>::Value: ryft::Concretizable<bool>,
                    AddOperation<ArrayIrType>: ryft::DifferentiableOperation<__DifferentiationContext>,
                    ArrayOperation<A>: ryft::MemberDifferentiableOperation<__DifferentiationContext>,
                    GatherOperation: ryft::MemberDifferentiableOperation<__DifferentiationContext>,
                    Operation<A>: ryft::OperationProvider<
                        ArrayIrType,
                        ryft::ZeroOperation<ArrayIrType>,
                        Operation = Operation<A>,
                    >,
            }),
        );
        assert_eq!(
            rendering(&generated.items[0]),
            rendering(&quote! {
                #[inline]
                fn jvp<
                    __D: ryft::DifferentiationDriver<__DifferentiationContext>,
                    __Policy: ryft::DifferentiationPolicy<__DifferentiationContext>,
                >(
                    &self,
                    context: &ryft::DifferentiationContext<__DifferentiationContext, __Policy>,
                    driver: &__D,
                    inputs: &[ryft::DifferentiationDual<<__DifferentiationContext as ryft::Domain>::Value>],
                ) -> ::std::result::Result<
                    ::std::vec::Vec<ryft::DifferentiationDual<<__DifferentiationContext as ryft::Domain>::Value>>,
                    ryft::DifferentiationError,
                > {
                    match self {
                        Self::Add(operation) => {
                            <AddOperation<ArrayIrType> as ryft::DifferentiableOperation<__DifferentiationContext>>::
                                jvp(operation, context, driver, inputs)
                        },
                        Self::Array(operation) => {
                            <ArrayOperation<A> as ryft::MemberDifferentiableOperation<__DifferentiationContext>>::
                                jvp_in_parent(operation, context, driver, inputs)
                        },
                        Self::Dimension(..) => {
                            context.jvp_structural_operation(self.clone(), inputs)
                        },
                        Self::Gather(operation) => {
                            <GatherOperation as ryft::MemberDifferentiableOperation<__DifferentiationContext>>::
                                jvp_in_parent(operation, context, driver, inputs)
                        },
                    }
                }
            }),
        );

        // The `jvp_for_transpose` function dispatches exactly like `jvp`, but to the corresponding payload functions.
        assert_eq!(
            rendering(&generated.items[1]),
            rendering(&generated.items[0])
                .replace("fnjvp<", "fnjvp_for_transpose<")
                .replace(">::jvp(", ">::jvp_for_transpose(")
                .replace(">::jvp_in_parent(", ">::jvp_for_transpose_in_parent("),
        );
    }

    #[test]
    fn test_operation_code_generator_generate_differentiation_dispatcher_structural_mixed_members() {
        // Declaring the members of a family and abbreviating its mixed markers generates exactly the same code as
        // naming the member type on every marker.
        let explicit = code_generator(quote! {
            #[ryft(type(ArrayIrType), constant(ArrayIrValue<A>))]
            enum Operation<A: Value<Type = ArrayType>> {
                #[ryft(mixed(ArrayType, structural))]
                Zero(ZeroOperation<ArrayType>),
            }
        });
        let declared = code_generator(quote! {
            #[ryft(type(ArrayIrType), constant(ArrayIrValue<A>), members(ArrayType, structural(DimensionType)))]
            enum Operation<A: Value<Type = ArrayType>> {
                #[ryft(mixed(structural))]
                Zero(ZeroOperation<ArrayType>),
            }
        });
        assert_eq!(
            rendering(&declared.generate_differentiation_dispatcher()),
            rendering(&explicit.generate_differentiation_dispatcher()),
        );

        // A structural mixed payload stages one member zero tangent per output, in the computational member universe
        // that the output belongs to, and so it tries every such universe and needs a zero constructor and a type view
        // for each of them.
        let operation = code_generator(quote! {
            #[ryft(type(ArrayIrType), constant(ArrayIrValue<A>))]
            #[ryft(members(ArrayType, TableType, structural(DimensionType)))]
            enum Operation<A: Value<Type = ArrayType>> {
                #[ryft(mixed(ArrayType, structural))]
                Zero(ZeroOperation<ArrayType>),
            }
        });
        let generated: syn::ItemImpl = syn::parse2(operation.generate_differentiation_dispatcher()).unwrap();
        let predicates = generated.generics.where_clause.as_ref().unwrap().predicates.iter().map(rendering);
        assert_eq!(
            predicates.skip(4).collect::<Vec<_>>(),
            [
                quote!(ArrayType: ryft::DifferentiableType),
                quote!(Operation<A>: ::std::convert::From<ryft::ZeroOperation<ArrayType>>),
                quote!(for<'__t> &'__t ArrayType: ::std::convert::TryFrom<&'__t ArrayIrType, Error = ryft::TypeError>),
                quote!(TableType: ryft::DifferentiableType),
                quote!(Operation<A>: ::std::convert::From<ryft::ZeroOperation<TableType>>),
                quote!(for<'__t> &'__t TableType: ::std::convert::TryFrom<&'__t ArrayIrType, Error = ryft::TypeError>),
                quote! {
                    Operation<A>: ryft::OperationProvider<
                        ArrayIrType,
                        ryft::ZeroOperation<ArrayIrType>,
                        Operation = Operation<A>,
                    >
                },
            ]
            .iter()
            .map(rendering)
            .collect::<Vec<_>>(),
        );
        assert_eq!(
            rendering(&generated.items[0]),
            rendering(&quote! {
                #[inline]
                fn jvp<
                    __D: ryft::DifferentiationDriver<__DifferentiationContext>,
                    __Policy: ryft::DifferentiationPolicy<__DifferentiationContext>,
                >(
                    &self,
                    context: &ryft::DifferentiationContext<__DifferentiationContext, __Policy>,
                    driver: &__D,
                    inputs: &[ryft::DifferentiationDual<<__DifferentiationContext as ryft::Domain>::Value>],
                ) -> ::std::result::Result<
                    ::std::vec::Vec<ryft::DifferentiationDual<<__DifferentiationContext as ryft::Domain>::Value>>,
                    ryft::DifferentiationError,
                > {
                    match self {
                        Self::Zero(..) => {
                            context.jvp_structural_mixed_operation(
                                self.clone(),
                                inputs,
                                |context, primal_inputs, output_type| {
                                    if let ::std::option::Option::Some(tangent) =
                                        context.stage_member_zero_tangent::<ArrayType>(primal_inputs, output_type)?
                                    {
                                        return ::std::result::Result::Ok(::std::option::Option::Some(tangent));
                                    }
                                    if let ::std::option::Option::Some(tangent) =
                                        context.stage_member_zero_tangent::<TableType>(primal_inputs, output_type)?
                                    {
                                        return ::std::result::Result::Ok(::std::option::Option::Some(tangent));
                                    }
                                    ::std::result::Result::Ok(::std::option::Option::None)
                                },
                            )
                        },
                    }
                }
            }),
        );
    }

    #[test]
    fn test_operation_code_generator_generate_transposition_dispatcher() {
        // Native payloads use their own transposition rules, computational projected payloads use their
        // parent-universe rules, structural projected payloads contribute no cotangents, and mixed payloads are
        // transposed through the shared mixed adapter.
        let operation = code_generator(quote! {
            #[ryft(type(ArrayIrType), constant(ArrayIrValue<A>))]
            enum Operation<A: Value<Type = ArrayType>> {
                Add(AddOperation<ArrayIrType>),
                #[ryft(projected(ArrayType))]
                Array(Box<ArrayOperation<A>>),
                #[ryft(projected(DimensionType, structural))]
                Dimension(DimensionOperation),
                #[ryft(mixed(ArrayType))]
                Gather(GatherOperation),
            }
        });
        assert_eq!(
            rendering(&operation.generate_transposition_dispatcher()),
            rendering(&quote! {
                #[automatically_derived]
                impl<A: Value<Type = ArrayType>, __TranspositionValue>
                    ryft::TransposableOperation<__TranspositionValue, Operation<A>> for Operation<A>
                where
                    AddOperation<ArrayIrType>: ryft::Operation<Type = ArrayIrType>,
                    ArrayOperation<A>: ryft::Operation<Type = ArrayType>,
                    DimensionOperation: ryft::Operation<Type = DimensionType>,
                    GatherOperation: ryft::MemberOperation<ArrayIrType>,
                    ArrayType: ryft::Type<Identity = <ArrayIrType as ryft::Type>::Identity>,
                    ArrayIrType: ::std::convert::From<ArrayType>,
                    for<'__member_type> &'__member_type ArrayType:
                        ::std::convert::TryFrom<&'__member_type ArrayIrType, Error = ryft::TypeError>,
                    DimensionType: ryft::Type<Identity = <ArrayIrType as ryft::Type>::Identity>,
                    ArrayIrType: ::std::convert::From<DimensionType>,
                    for<'__member_type> &'__member_type DimensionType:
                        ::std::convert::TryFrom<&'__member_type ArrayIrType, Error = ryft::TypeError>,
                    __TranspositionValue: ryft::Value<Type = ArrayIrType>,
                    Operation<A>: ryft::Operation<Type = ArrayIrType>,
                    AddOperation<ArrayIrType>: ryft::TransposableOperation<__TranspositionValue, Operation<A>>,
                    ArrayOperation<A>: ryft::MemberTransposableOperation<__TranspositionValue, Operation<A>>,
                    ArrayType: ryft::DifferentiableType,
                    __TranspositionValue: ryft::ValueProjection<ArrayType, Projected: ryft::Value<Type = ArrayType>>,
                    Operation<A>: ryft::OperationProjection<ArrayType>,
                    <Operation<A> as ryft::OperationProjection<ArrayType>>::Projected: ryft::OperationProvider<
                        ArrayType,
                        ryft::GuardOperation<ArrayType>,
                        Operation = <Operation<A> as ryft::OperationProjection<ArrayType>>::Projected,
                    >,
                    <Operation<A> as ryft::OperationProjection<ArrayType>>::Projected:
                        ::std::convert::From<GatherOperation>
                            + ryft::TransposableOperation<
                                <__TranspositionValue as ryft::ValueProjection<ArrayType>>::Projected,
                                <Operation<A> as ryft::OperationProjection<ArrayType>>::Projected,
                            >,
                    ArrayIrType: ::std::convert::From<ArrayType>,
                    for<'__t> &'__t ArrayType: ::std::convert::TryFrom<&'__t ArrayIrType, Error = ryft::TypeError>,
                {
                    #[inline]
                    fn transpose<__D: ryft::TranspositionDriver<__TranspositionValue, Operation<A>>>(
                        &self,
                        context: &mut ryft::TranspositionContext<__TranspositionValue, Operation<A>>,
                        driver: &__D,
                        inputs: &[ryft::PartialValue<
                            ryft::Tracer<ryft::TracingContext<__TranspositionValue, Operation<A>>>,
                        >],
                        outputs: &[ryft::MaybeZero<
                            ryft::Tracer<ryft::TracingContext<__TranspositionValue, Operation<A>>>,
                        >],
                        accumulators: &[ryft::CotangentAccumulator],
                    ) -> ::std::result::Result<(), ryft::DifferentiationError> {
                        match self {
                            Self::Add(operation) => {
                                <AddOperation<ArrayIrType> as ryft::TransposableOperation<
                                    __TranspositionValue,
                                    Operation<A>,
                                >>::transpose(operation, context, driver, inputs, outputs, accumulators)
                            },
                            Self::Array(operation) => {
                                <ArrayOperation<A> as ryft::MemberTransposableOperation<
                                    __TranspositionValue,
                                    Operation<A>,
                                >>::transpose_in_parent(&**operation, context, driver, inputs, outputs, accumulators)
                            },
                            Self::Dimension(..) => {
                                ::std::result::Result::Ok(())
                            },
                            Self::Gather(operation) => {
                                context.transpose_mixed_operation(operation, inputs, outputs, accumulators)
                            },
                        }
                    }
                }
            }),
        );

        // Test generating the dispatcher of an enum with two inferred value parameters, which transposes over its
        // constant type instead of over a separate transposition value type.
        let operation = code_generator(quote! {
            enum Operation<V: Value<Type = ArrayType>, C: Value<Type = ArrayType>> {
                Constant(ConstantOperation<V>),
            }
        });
        let generated: syn::ItemImpl = syn::parse2(operation.generate_transposition_dispatcher()).unwrap();
        assert_eq!(
            rendering(&generated.trait_.unwrap().1),
            rendering(&quote!(ryft::TransposableOperation<V, Operation<V, C>>))
        );
    }

    #[test]
    fn test_operation_code_generator_generate_operation_projection_impl() {
        let operation = code_generator(quote! {
            #[ryft(type(ArrayIrType), constant(ArrayIrValue<A>))]
            enum Operation<A: Value<Type = ArrayType>> {
                Add(AddOperation<ArrayIrType>),
                #[ryft(projected(ArrayType))]
                Array(Box<ArrayOperation<A>>),
                #[ryft(mixed(ArrayType))]
                Gather(GatherOperation),
            }
        });
        assert!(operation.generate_operation_projection_impl(&operation.variants[0]).is_none());
        assert_eq!(
            rendering(&operation.generate_operation_projection_impl(&operation.variants[1]).unwrap()),
            rendering(&quote! {
                #[automatically_derived]
                impl<A: Value<Type = ArrayType>> ryft::OperationProjection<ArrayType> for Operation<A>
                where
                    AddOperation<ArrayIrType>: ryft::Operation<Type = ArrayIrType>,
                    ArrayOperation<A>: ryft::Operation<Type = ArrayType>,
                    GatherOperation: ryft::MemberOperation<ArrayIrType>,
                    ArrayType: ryft::Type<Identity = <ArrayIrType as ryft::Type>::Identity>,
                    ArrayIrType: ::std::convert::From<ArrayType>,
                    for<'__member_type> &'__member_type ArrayType:
                        ::std::convert::TryFrom<&'__member_type ArrayIrType, Error = ryft::TypeError>,
                {
                    type Projected = ArrayOperation<A>;
                }
            }),
        );
        assert!(operation.generate_operation_projection_impl(&operation.variants[2]).is_none());
    }

    #[test]
    fn test_operation_code_generator_generate_payload_projection_impl() {
        // Mixed payloads project like native ones but are never constructed, native payloads are constructed before the
        // projected member even though the member is declared first, and the generic-dependent payload and the generic
        // extension neither project nor are constructed.
        let operation = code_generator(quote! {
            #[ryft(type(ArrayIrType), constant(ArrayIrValue<A>))]
            enum Operation<A: Value<Type = ArrayType>, Extension> {
                #[ryft(projected(ArrayType))]
                Array(ArrayOperation<A>),
                #[ryft(mixed(ArrayType))]
                Gather(GatherOperation),
                Dot(DotOperation),
                Tag(Box<TagOperation<ArrayIrType>>),
                Scan(ScanOperation<A>),
                Extension(Extension),
            }
        });
        assert_eq!(
            rendering(&operation.generate_payload_projection_impl()),
            rendering(&quote! {
                #[automatically_derived]
                impl<A: Value<Type = ArrayType>, Extension> ryft::OperationPayloadProjection for Operation<A, Extension>
                where
                    ArrayOperation<A>: ryft::OperationPayloadProjection,
                {
                    #[allow(unreachable_patterns)]
                    #[inline]
                    fn project_payload(
                        &self,
                        payload: ::std::any::TypeId,
                    ) -> ::std::option::Option<&dyn ::std::any::Any> {
                        match self {
                            Self::Array(operation) => {
                                <ArrayOperation<A> as ryft::OperationPayloadProjection>::
                                    project_payload(operation, payload)
                            }
                            Self::Gather(operation) if payload == ::std::any::TypeId::of::<GatherOperation>() => {
                                ::std::option::Option::Some(operation as &dyn ::std::any::Any)
                            }
                            Self::Dot(operation) if payload == ::std::any::TypeId::of::<DotOperation>() => {
                                ::std::option::Option::Some(operation as &dyn ::std::any::Any)
                            }
                            Self::Tag(operation)
                                if payload == ::std::any::TypeId::of::<TagOperation<ArrayIrType>>() =>
                            {
                                ::std::option::Option::Some(&**operation as &dyn ::std::any::Any)
                            }
                            _ => ::std::option::Option::None,
                        }
                    }

                    #[inline]
                    fn from_payload(
                        payload: ryft::ErasedOperation,
                    ) -> ::std::result::Result<Self, ryft::ErasedOperation> {
                        let payload = match payload.downcast::<DotOperation>() {
                            ::std::result::Result::Ok(operation) => {
                                return ::std::result::Result::Ok(Self::Dot(operation));
                            }
                            ::std::result::Result::Err(payload) => payload,
                        };
                        let payload = match payload.downcast::<TagOperation<ArrayIrType>>() {
                            ::std::result::Result::Ok(operation) => {
                                return ::std::result::Result::Ok(Self::Tag(::std::boxed::Box::new(operation)));
                            }
                            ::std::result::Result::Err(payload) => payload,
                        };
                        let payload = match <ArrayOperation<A> as ryft::OperationPayloadProjection>::
                            from_payload(payload)
                        {
                            ::std::result::Result::Ok(operation) => {
                                return ::std::result::Result::Ok(
                                    <Self as ::std::convert::From<ArrayOperation<A>>>::from(operation),
                                );
                            }
                            ::std::result::Result::Err(payload) => payload,
                        };
                        ::std::result::Result::Err(payload)
                    }
                }
            }),
        );
    }

    #[test]
    fn test_operation_code_generator_generate_collective_impl() {
        // Marked variants delegate to their payloads, which may be collectives, member families, or generic extensions,
        // stored directly or boxed, while every other variant reports no axes.
        let operation = code_generator(quote! {
            #[ryft(type(ArrayIrType), constant(ArrayIrValue<A>))]
            enum Operation<A: Value<Type = ArrayType>, Extension> {
                #[ryft(projected(ArrayType), collective)]
                Array(ArrayOperation<A>),
                #[ryft(mixed(ArrayType), collective)]
                Gather(Box<ParallelAllGatherOperation>),
                Dot(DotOperation),
                #[ryft(collective)]
                Extension(Extension),
            }
        });
        assert_eq!(
            rendering(&operation.generate_collective_impl().unwrap()),
            rendering(&quote! {
                #[automatically_derived]
                impl<A: Value<Type = ArrayType>, Extension> ryft::CollectiveOperation for Operation<A, Extension>
                where
                    ArrayOperation<A>: ryft::CollectiveOperation,
                    ParallelAllGatherOperation: ryft::CollectiveOperation,
                    Extension: ryft::CollectiveOperation,
                {
                    #[allow(unreachable_patterns)]
                    #[inline]
                    fn axes(&self) -> ::std::borrow::Cow<'_, ryft::CollectiveAxes> {
                        match self {
                            Self::Array(operation) => {
                                <ArrayOperation<A> as ryft::CollectiveOperation>::axes(operation)
                            },
                            Self::Gather(operation) => {
                                <ParallelAllGatherOperation as ryft::CollectiveOperation>::axes(&**operation)
                            },
                            Self::Extension(operation) => {
                                <Extension as ryft::CollectiveOperation>::axes(operation)
                            },
                            _ => ::std::borrow::Cow::Owned(
                                <ryft::CollectiveAxes as ::std::default::Default>::default()
                            ),
                        }
                    }

                    #[allow(unreachable_patterns)]
                    #[inline]
                    fn hidden_axes(&self) -> ryft::HiddenAxes {
                        match self {
                            Self::Array(operation) => {
                                <ArrayOperation<A> as ryft::CollectiveOperation>::hidden_axes(operation)
                            },
                            Self::Gather(operation) => {
                                <ParallelAllGatherOperation as ryft::CollectiveOperation>::hidden_axes(&**operation)
                            },
                            Self::Extension(operation) => {
                                <Extension as ryft::CollectiveOperation>::hidden_axes(operation)
                            },
                            _ => ryft::HiddenAxes::None,
                        }
                    }
                }
            }),
        );

        // A family without marked variants gets no implementation.
        let operation = code_generator(quote! {
            #[ryft(type(ArrayIrType), constant(ArrayIrValue<A>))]
            enum Operation<A: Value<Type = ArrayType>> {
                Dot(DotOperation),
            }
        });
        assert!(operation.generate_collective_impl().is_none());
    }

    #[test]
    fn test_operation_code_generator_generate_lockstep_impl() {
        // Marked variants delegate to their payloads, which may be operations, member families, or generic extensions,
        // stored directly or boxed, while every other variant has no lockstep rule.
        let operation = code_generator(quote! {
            #[ryft(type(ArrayIrType), constant(ArrayIrValue<A>))]
            enum Operation<A: Value<Type = ArrayType>, Extension> {
                #[ryft(projected(ArrayType), lockstep)]
                Array(ArrayOperation<A>),
                #[ryft(lockstep)]
                While(Box<WhileOperation<ArrayIrType>>),
                Dot(DotOperation),
                #[ryft(lockstep)]
                Extension(Extension),
            }
        });
        assert_eq!(
            rendering(&operation.generate_lockstep_impl().unwrap()),
            rendering(&quote! {
                #[automatically_derived]
                impl<
                    A: Value<Type = ArrayType>,
                    Extension,
                    LockstepFamily: ryft::Operation<Type = ryft::ArrayIrType>
                > ryft::LockstepOperation<LockstepFamily> for Operation<A, Extension>
                where
                    ArrayOperation<A>: ryft::LockstepOperation<LockstepFamily>,
                    WhileOperation<ArrayIrType>: ryft::LockstepOperation<LockstepFamily>,
                    Extension: ryft::LockstepOperation<LockstepFamily>,
                {
                    #[allow(unreachable_patterns)]
                    #[inline]
                    fn interpret_in_lockstep<
                        LockstepDriver: ryft::InterpretationDriver<ryft::LockstepDomain<LockstepFamily>>,
                    >(
                        &self,
                        context: &ryft::LockstepDomain<LockstepFamily>,
                        driver: &LockstepDriver,
                        inputs: &[ryft::LockstepValue<LockstepFamily>],
                    ) -> ::std::option::Option<
                        ::std::result::Result<
                            ::std::vec::Vec<ryft::LockstepValue<LockstepFamily>>,
                            ryft::ProgramError,
                        >,
                    > {
                        match self {
                            Self::Array(operation) => {
                                <ArrayOperation<A> as ryft::LockstepOperation<LockstepFamily>>::interpret_in_lockstep(
                                    operation,
                                    context,
                                    driver,
                                    inputs,
                                )
                            },
                            Self::While(operation) => {
                                <WhileOperation<ArrayIrType> as ryft::LockstepOperation<LockstepFamily>>
                                    ::interpret_in_lockstep(&**operation, context, driver, inputs,)
                            },
                            Self::Extension(operation) => {
                                <Extension as ryft::LockstepOperation<LockstepFamily>>::interpret_in_lockstep(
                                    operation,
                                    context,
                                    driver,
                                    inputs,
                                )
                            },
                            _ => ::std::option::Option::None,
                        }
                    }
                }
            }),
        );

        // A family without marked variants does not implement the trait.
        let operation = code_generator(quote! {
            #[ryft(type(ArrayType), constant(A))]
            enum Operation<A: Value<Type = ArrayType>> {
                Dot(DotOperation),
            }
        });
        assert!(operation.generate_lockstep_impl().is_none());
    }

    #[test]
    fn test_operation_code_generator_generate_guard_provider_impl() {
        // A family without a native guard variant rejects guard requests of its primary type under the bounds of its
        // generated `Operation` implementation.
        // A projected member guard belongs to the member universe, and so it does not count as a native guard.
        let operation = code_generator(quote! {
            #[ryft(type(ArrayIrType), constant(ArrayIrValue))]
            enum Operation {
                Add(AddOperation<ArrayIrType>),
                #[ryft(projected(ArrayType))]
                Guard(GuardOperation<ArrayType>),
            }
        });
        assert_eq!(
            rendering(&operation.generate_guard_provider_impl().unwrap()),
            rendering(&quote! {
                #[automatically_derived]
                impl ryft::OperationProvider<ArrayIrType, ryft::GuardOperation<ArrayIrType>> for Operation
                where
                    AddOperation<ArrayIrType>: ryft::Operation<Type = ArrayIrType>,
                    GuardOperation<ArrayType>: ryft::Operation<Type = ArrayType>,
                    ArrayType: ryft::Type<Identity = <ArrayIrType as ryft::Type>::Identity>,
                    ArrayIrType: ::std::convert::From<ArrayType>,
                    for<'__member_type> &'__member_type ArrayType:
                        ::std::convert::TryFrom<&'__member_type ArrayIrType, Error = ryft::TypeError>,
                {
                    type Operation = Self;

                    #[inline]
                    fn provide(
                        request: ryft::GuardOperation<ArrayIrType>,
                        _input_types: &[&ArrayIrType],
                    ) -> ::std::result::Result<Self, ryft::ProgramError> {
                        ::std::result::Result::Err(request.unsupported_error())
                    }
                }
            }),
        );

        // A family with a native guard variant, stored directly or boxed, provides guards through the blanket
        // supporting provider over its generated `From` conversion, and so it must not get a rejecting provider.
        let operation = code_generator(quote! {
            enum Operation<V: Value<Type = ArrayType>> {
                Guard(ryft::GuardOperation<ArrayType>),
            }
        });
        assert!(operation.generate_guard_provider_impl().is_none());
        let operation = code_generator(quote! {
            enum Operation<V: Value<Type = ArrayType>> {
                Guard(Box<GuardOperation<ArrayType>>),
            }
        });
        assert!(operation.generate_guard_provider_impl().is_none());
    }

    #[test]
    fn test_operation_code_generator_generate_conversion_impls() {
        let operation = code_generator(quote! {
            enum Operation<V: Value<Type = ArrayType>> {
                Zero(ZeroOperation<ArrayType>),
                Constant(Box<ConstantOperation<V>>),
                #[ryft(skip_from)]
                Add(AddOperation<ArrayType>),
            }
        });
        assert_eq!(
            rendering(&operation.generate_conversion_impls(&operation.variants[1])),
            rendering(&quote! {
                #[automatically_derived]
                impl<V: Value<Type = ArrayType>> ::std::convert::From<ConstantOperation<V>> for Operation<V> {
                    #[inline]
                    fn from(operation: ConstantOperation<V>) -> Self {
                        Self::Constant(::std::boxed::Box::new(operation))
                    }
                }

                #[automatically_derived]
                impl<'__operation, V: Value<Type = ArrayType>> ::std::convert::TryFrom<&'__operation Operation<V>>
                    for &'__operation ConstantOperation<V>
                where
                    Operation<V>: ryft::Operation,
                {
                    type Error = ryft::TypeError;

                    #[inline]
                    fn try_from(value: &'__operation Operation<V>) -> ::std::result::Result<Self, ryft::TypeError> {
                        match value {
                            Operation::Constant(operation) => ::std::result::Result::Ok(&**operation),
                            _ => ::std::result::Result::Err(ryft::TypeError::invalid(::std::format!(
                                "cannot project operation `{}` into a `{}` payload",
                                <Operation<V> as ryft::Operation>::name(value),
                                "ConstantOperation<V>",
                            ))),
                        }
                    }
                }
            }),
        );

        // Test skipping the owned conversion, which still keeps the borrowed projection.
        let generated: syn::File = syn::parse2(operation.generate_conversion_impls(&operation.variants[2])).unwrap();
        assert_eq!(generated.items.len(), 1);
        assert!(
            rendering(&generated).contains("TryFrom<&'__operationOperation<V>>for&'__operationAddOperation<ArrayType>")
        );
    }

    #[test]
    fn test_operation_code_generator_constant_parameter_predicates() {
        // Every mention of the constant parameter is renamed, including mentions inside its bounds and in associated
        // type projections, while predicates that do not bound the parameter are not copied.
        let operation = code_generator(quote! {
            enum Operation<V: Value<Type = ArrayType> + PartialEq<V>, U>
            where
                Wrapper<V>: Bound,
                V::Item: Other<V>,
                U: Unrelated<V>,
            {
                Constant(ConstantOperation<V>),
            }
        });
        assert_eq!(
            operation
                .constant_parameter_predicates(&format_ident!("RuntimeValue"))
                .iter()
                .map(rendering)
                .collect::<Vec<_>>(),
            [
                quote!(RuntimeValue: Value<Type = ArrayType>),
                quote!(RuntimeValue: PartialEq<RuntimeValue>),
                quote!(Wrapper<RuntimeValue>: Bound),
                quote!(RuntimeValue::Item: Other<RuntimeValue>),
            ]
            .iter()
            .map(rendering)
            .collect::<Vec<_>>(),
        );

        // Enums with an explicit constant type have no constant parameter whose bounds could be copied.
        let operation = code_generator(quote! {
            #[ryft(type(ArrayIrType), constant(ArrayIrValue<A>))]
            enum Operation<A: Value<Type = ArrayType> + Clone> {
                Add(AddOperation<ArrayIrType>),
            }
        });
        assert!(operation.constant_parameter_predicates(&format_ident!("RuntimeValue")).is_empty());
    }

    #[test]
    fn test_operation_code_generator_tangent_universes() {
        let operation = code_generator(quote! {
            #[ryft(type(ArrayIrType), constant(ArrayIrValue), members(ArrayType, structural(DimensionType), TableType))]
            enum Operation {
                Add(AddOperation<ArrayIrType>),
            }
        });
        assert_eq!(
            operation
                .tangent_universes(&syn::parse_quote!(TableType))
                .iter()
                .map(TokensHelpers::to_compact_string)
                .collect::<Vec<_>>(),
            vec!["TableType", "ArrayType"],
        );
        let operation = code_generator(quote! {
            #[ryft(type(ArrayIrType), constant(ArrayIrValue))]
            enum Operation {
                Add(AddOperation<ArrayIrType>),
            }
        });
        assert_eq!(
            operation
                .tangent_universes(&syn::parse_quote!(ArrayType))
                .iter()
                .map(TokensHelpers::to_compact_string)
                .collect::<Vec<_>>(),
            vec!["ArrayType"],
        );
    }

    #[test]
    fn test_operation_member_parse() {
        let member: OperationMember = syn::parse_quote!(ArrayType<Static>);
        assert_eq!(member.r#type.to_compact_string(), "ArrayType<Static>");
        assert!(!member.structural);
        let member: OperationMember = syn::parse_quote!(structural(DimensionType));
        assert_eq!(member.r#type.to_compact_string(), "DimensionType");
        assert!(member.structural);

        // Test parsing invalid members.
        assert_eq!(
            syn::parse2::<OperationMember>(quote!(replicated(DimensionType))).err().unwrap().to_string(),
            "invalid ryft member universe role `replicated`; only `structural` is supported here",
        );
        assert_eq!(
            syn::parse2::<OperationMember>(quote!(structural(ArrayType, DimensionType)))
                .err()
                .unwrap()
                .to_string(),
            "`structural(...)` declares exactly one member type",
        );
    }

    #[test]
    fn test_operation_variant_receiver() {
        let operation = code_generator(quote! {
            enum Operation<V: Value<Type = ArrayType>> {
                Zero(ZeroOperation<ArrayType>),
                Constant(Box<ConstantOperation<V>>),
            }
        });
        assert_eq!(operation.variants[0].receiver().to_string(), quote!(operation).to_string());
        assert_eq!(operation.variants[1].receiver().to_string(), quote!(&**operation).to_string());
    }

    #[test]
    fn test_operation_variant_construct() {
        let operation = code_generator(quote! {
            enum Operation<V: Value<Type = ArrayType>> {
                Zero(ZeroOperation<ArrayType>),
                Constant(Box<ConstantOperation<V>>),
            }
        });
        assert_eq!(
            operation.variants[0].construct(quote!(payload)).to_string(),
            quote!(Self::Zero(payload)).to_string()
        );
        assert_eq!(
            operation.variants[1].construct(quote!(payload)).to_string(),
            quote!(Self::Constant(::std::boxed::Box::new(payload))).to_string(),
        );
    }

    #[test]
    fn test_operation_variant_arm() {
        let operation = code_generator(quote! {
            enum Operation<V: Value<Type = ArrayType>> {
                Constant(Box<ConstantOperation<V>>),
            }
        });
        assert_eq!(
            operation.variants[0].arm(quote!(body(operation))).to_string(),
            quote!(Self::Constant(operation) => { body(operation) },).to_string(),
        );
        assert_eq!(
            operation.variants[0].wildcard_arm(quote!(body())).to_string(),
            quote!(Self::Constant(..) => { body() },).to_string(),
        );
    }

    #[test]
    fn test_operation_variant_is_native_guard() {
        // The detection is syntactic and so it misses guards that are named through type aliases.
        let operation = code_generator(quote! {
            #[ryft(type(ArrayIrType), constant(ArrayIrValue))]
            enum Operation {
                Guard(ryft::GuardOperation<ArrayIrType>),
                BoxedGuard(Box<GuardOperation<ArrayIrType>>),
                #[ryft(projected(ArrayType))]
                ArrayGuard(GuardOperation<ArrayType>),
                AliasedGuard(CheckedInput),
            }
        });
        assert!(operation.variants[0].is_native_guard());
        assert!(operation.variants[1].is_native_guard());
        assert!(!operation.variants[2].is_native_guard());
        assert!(!operation.variants[3].is_native_guard());
    }

    #[test]
    fn test_operation_variant_class_parse() {
        let members = [
            OperationMember { r#type: syn::parse_quote!(ArrayType), structural: false },
            OperationMember { r#type: syn::parse_quote!(DimensionType), structural: true },
        ];
        let class = |attribute: syn::Attribute, members: &[OperationMember]| {
            let mut parsed_class = None;
            let parsing = attribute.parse_nested_meta(|meta| {
                parsed_class = Some(OperationVariantClass::parse(&meta, members)?);
                Ok(())
            });
            match parsing.map(|()| parsed_class.unwrap()) {
                Ok(OperationVariantClass::ProjectedMember { member_type, structural }) => {
                    Ok(("projected", member_type.to_compact_string(), structural))
                }
                Ok(OperationVariantClass::MixedMember { member_type, structural }) => {
                    Ok(("mixed", member_type.to_compact_string(), structural))
                }
                Ok(OperationVariantClass::FamilyNative) => unreachable!(),
                Err(error) => Err(error.to_string()),
            }
        };

        // Test parsing both member boundaries with both roles, with explicit and with defaulted member types.
        let array = "ArrayType".to_string();
        let dimension = "DimensionType".to_string();
        assert_eq!(
            class(syn::parse_quote!(#[ryft(projected(ArrayType))]), &[]),
            Ok(("projected", array.clone(), false))
        );
        assert_eq!(
            class(syn::parse_quote!(#[ryft(projected(DimensionType, structural))]), &members),
            Ok(("projected", dimension.clone(), true)),
        );
        assert_eq!(class(syn::parse_quote!(#[ryft(mixed(ArrayType))]), &members), Ok(("mixed", array.clone(), false)));
        assert_eq!(class(syn::parse_quote!(#[ryft(mixed)]), &members), Ok(("mixed", array.clone(), false)));
        assert_eq!(class(syn::parse_quote!(#[ryft(mixed(structural))]), &members), Ok(("mixed", array, true)));

        // Test parsing invalid member classes.
        assert_eq!(
            class(syn::parse_quote!(#[ryft(projected)]), &members),
            Err("unexpected end of input, expected parentheses".to_string())
        );
        assert_eq!(
            class(syn::parse_quote!(#[ryft(projected(ArrayType, replicated))]), &members),
            Err("invalid projected operation variant role `replicated`; only `structural` is supported here"
                .to_string()),
        );
        assert_eq!(
            class(syn::parse_quote!(#[ryft(mixed(ArrayType, structural, DimensionType))]), &members),
            Err("the `mixed` operation variant class accepts one member type and an optional role".to_string()),
        );
        assert_eq!(
            class(syn::parse_quote!(#[ryft(mixed(TableType))]), &members),
            Err("the `mixed` operation variant class declares member type `TableType`, which the operation family \
                 does not declare in `#[ryft(members(...))]`"
                .to_string(),),
        );
        assert_eq!(
            class(syn::parse_quote!(#[ryft(mixed)]), &[]),
            Err("the `mixed` operation variant class cannot default its member type because the operation family \
                 declares no `#[ryft(members(...))]`; declare the family's members or name the member type as \
                 `mixed(U)`"
                .to_string(),),
        );
        let members = [
            OperationMember { r#type: syn::parse_quote!(ArrayType), structural: false },
            OperationMember { r#type: syn::parse_quote!(TableType), structural: false },
        ];
        assert_eq!(
            class(syn::parse_quote!(#[ryft(mixed(structural))]), &members),
            Err("the `mixed` operation variant class cannot default its member type because the operation family \
                 declares 2 computational member universes; name the member type as `mixed(U)`"
                .to_string(),),
        );
    }

    #[test]
    fn test_operation_variant_class_name() {
        let member_type: syn::Type = syn::parse_quote!(ArrayType);
        let projected = OperationVariantClass::ProjectedMember { member_type: member_type.clone(), structural: false };
        let mixed = OperationVariantClass::MixedMember { member_type, structural: true };
        assert_eq!(projected.name().to_string(), "projected");
        assert_eq!(mixed.name().to_string(), "mixed");
    }

    #[test]
    fn test_operation_variant_class_projected_member_type() {
        let member_type: syn::Type = syn::parse_quote!(ArrayType);
        let projected = OperationVariantClass::ProjectedMember { member_type: member_type.clone(), structural: true };
        let mixed = OperationVariantClass::MixedMember { member_type, structural: false };
        assert_eq!(
            projected.projected_member_type().map(TokensHelpers::to_compact_string),
            Some("ArrayType".to_string()),
        );
        assert!(mixed.projected_member_type().is_none());
        assert!(OperationVariantClass::FamilyNative.projected_member_type().is_none());
    }

    #[test]
    fn test_value_parameters() {
        let input: syn::DeriveInput = syn::parse_quote! {
            enum Operation<T, V: Clone + ryft::Value<Type = T>, W, X: Value<T>, Y: Value<Type = A, Other = B>>
            where
                W: Value<Type = DataType>,
                Vec<W>: Value<Type = ArrayType>,
            {}
        };
        assert_eq!(
            value_parameters(&input.generics)
                .into_iter()
                .map(|(parameter, value_type)| (parameter.to_string(), value_type.to_compact_string()))
                .collect::<Vec<_>>(),
            vec![("V".to_string(), "T".to_string()), ("W".to_string(), "DataType".to_string())],
        );
    }

    #[test]
    fn test_is_unconditionally_static() {
        let generics: syn::Generics = syn::parse_quote!(<'a, V: Value, const N: usize>);
        assert!(is_unconditionally_static(&syn::parse_quote!(DotOperation), &generics));
        assert!(is_unconditionally_static(&syn::parse_quote!(TagOperation<ArrayType>), &generics));
        assert!(is_unconditionally_static(&syn::parse_quote!(FixedOperation<N>), &generics));
        assert!(!is_unconditionally_static(&syn::parse_quote!(ScanOperation<V>), &generics));
        assert!(!is_unconditionally_static(&syn::parse_quote!(Box<[(u8, fn(V))]>), &generics));
        assert!(!is_unconditionally_static(&syn::parse_quote!(BorrowedOperation<'a>), &generics));
        assert!(!is_unconditionally_static(&syn::parse_quote!(&'static str), &generics));
    }
}
