use std::fmt::Display;

use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use syn::visit_mut::VisitMut;

use crate::helpers::attributes::Attribute;
use crate::helpers::generics::GenericsHelpers;
use crate::helpers::hygiene::const_block;
use crate::helpers::receivers::replace_self_type;
use crate::helpers::symbols::Symbol;

const RYFT_ATTRIBUTE: Symbol = Symbol::new("ryft");
const CRATE_ATTRIBUTE: Symbol = Symbol::new("crate");
const BOUNDS_ATTRIBUTE: Symbol = Symbol::new("bounds");
const INTERPRETATION_ATTRIBUTE: Symbol = Symbol::new("interpretation");
const PARTIAL_EVALUATION_ATTRIBUTE: Symbol = Symbol::new("partial_evaluation");
const BATCHING_ATTRIBUTE: Symbol = Symbol::new("batching");
const DIFFERENTIATION_ATTRIBUTE: Symbol = Symbol::new("differentiation");
const TRANSPOSITION_ATTRIBUTE: Symbol = Symbol::new("transposition");
const BATCHING_ACTIVE_ATTRIBUTE: Symbol = Symbol::new("active");
const VALID_CONTAINER_ATTRIBUTES: [Symbol; 2] = [CRATE_ATTRIBUTE, BOUNDS_ATTRIBUTE];

const DEFAULT_RYFT_CRATE: Symbol = Symbol::new("ryft");
const DEFAULT_MACRO_OPERATION_TYPE: Symbol = Symbol::new("__O");

const NESTED_ATTRIBUTE_ERROR: &str = "\
  the '#[ryft(...)]' attribute is only supported at the top level for operation enums. It is not supported for \
  variants or fields";

/// Code generator for `#[derive(Operation)]`.
pub(crate) struct CodeGenerator {
    /// Path to the `ryft` crate or facade used by generated code.
    ryft_crate: syn::Path,

    /// Type descriptor for the generated [`Operation`](ryft_core::Operation) implementation.
    operation_type: syn::Type,

    /// Extra bounds to attach to the generated interpretation value type.
    interpretation_value_bounds: Vec<syn::TypeParamBound>,

    /// Extra bounds to attach to the generated partial-evaluation value type.
    partial_evaluation_value_bounds: Vec<syn::TypeParamBound>,

    /// Whether `#[ryft(bounds(interpretation(...)))]` was already specified.
    interpretation_value_bounds_set: bool,

    /// Whether `#[ryft(bounds(partial_evaluation(...)))]` was already specified.
    partial_evaluation_value_bounds_set: bool,

    /// Extra bounds to attach to the eager batching impl's flowing value type, from
    /// `#[ryft(bounds(batching(...)))]`.
    ///
    /// The eager flowing value is the only position that needs author-supplied batching leaves: the staged flowing
    /// value is the unified tracer, whose capability impls are staging sugar conditioned only on
    /// `C::Operation: From<XOperation>` conversions (so the staged recursive rules spell operation-shaped `From`
    /// bounds that the closed enum discharges structurally), and the program-constant space carries no batching
    /// capabilities at all (backend constants are symbolic capture references).
    batching_value_bounds: Vec<syn::TypeParamBound>,

    /// Whether `#[ryft(bounds(batching(...)))]` was already specified.
    batching_bounds_set: bool,

    /// Extra bounds to attach to the generated `DifferentiableProgramOperation` and `LinearizableProgramOperation`
    /// witnesses' program constant type, from `#[ryft(bounds(differentiation(...)))]`.
    differentiation_value_bounds: Vec<syn::TypeParamBound>,

    /// Whether `#[ryft(bounds(differentiation(...)))]` was already specified.
    differentiation_value_bounds_set: bool,

    /// Extra bounds to attach to the generated `TransposableOperation` dispatcher's and
    /// `TransposableProgramOperation` witness's transposition value type, from
    /// `#[ryft(bounds(transposition(...)))]`. These serve the same role as bounds declared on the enum's own value
    /// parameter (which the generated implementations inherit) without forcing the enum's stored constant type to
    /// carry transposition-only capabilities.
    transposition_value_bounds: Vec<syn::TypeParamBound>,

    /// Whether `#[ryft(bounds(transposition(...)))]` was already specified.
    transposition_value_bounds_set: bool,

    /// [`DeriveKind`] identifying which operation derive this [`CodeGenerator`] serves. The four operation derives
    /// share the same `#[ryft(...)]` attribute namespace so combined derives compile: each `#[ryft(bounds(...))]`
    /// kind and the variant-level `#[ryft(batching(active))]` marker is *owned* (documented and consumed) by exactly
    /// one derive and *tolerated* (parsed and discarded) by the others, and this field selects the owned kinds.
    kind: DeriveKind,

    /// Errors accumulated in this [`CodeGenerator`]. The way error handling works in this code generator is that we
    /// collect errors as we encounter them, and keep going as far as we can with the information that is available,
    /// before raising them. That is meant to enable a smoother development experience when working with `ryft` by
    /// reducing the amount of trial and error required to get something work (i.e., users do not need to keep trying,
    /// fixing one error at a time; they get to see multiple errors at once, when there are multiple).
    errors: Vec<syn::Error>,
}

/// Operation derive that a [`CodeGenerator`] generates code for. Refer to the documentation of
/// [`CodeGenerator::kind`] for information on how the four derives share the `#[ryft(...)]` attribute namespace.
#[derive(Copy, Clone, PartialEq, Eq)]
enum DeriveKind {
    /// The `#[derive(Operation)]` macro, which owns the `#[ryft(bounds(interpretation(...)))]` and
    /// `#[ryft(bounds(partial_evaluation(...)))]` bound kinds.
    Operation,

    /// The `#[derive(BatchableOperation)]` macro, which owns the `#[ryft(bounds(batching(...)))]` bound kind and
    /// the variant-level `#[ryft(batching(active))]` marker.
    BatchableOperation,

    /// The `#[derive(DifferentiableOperation)]` macro, which owns the `#[ryft(bounds(differentiation(...)))]`
    /// bound kind.
    DifferentiableOperation,

    /// The `#[derive(TransposableOperation)]` macro, which owns the `#[ryft(bounds(transposition(...)))]`
    /// bound kind.
    TransposableOperation,
}

impl CodeGenerator {
    /// Creates a new [`CodeGenerator`] for the provided [`DeriveKind`], using inconsequential default values for
    /// fields whose values need to be extracted from the derive input. These values are inconsequential because if we
    /// fail to extract them from the provided input, then we will accumulate all relevant [`syn::Error`]s in
    /// [`CodeGenerator::errors`] and return a compiler error before we get to use them.
    fn new(kind: DeriveKind) -> Self {
        CodeGenerator {
            ryft_crate: DEFAULT_RYFT_CRATE.into(),
            operation_type: syn::Type::Path(syn::TypePath {
                qself: None,
                path: syn::Path::from(syn::Ident::from(DEFAULT_MACRO_OPERATION_TYPE)),
            }),
            interpretation_value_bounds: Vec::new(),
            partial_evaluation_value_bounds: Vec::new(),
            interpretation_value_bounds_set: false,
            partial_evaluation_value_bounds_set: false,
            batching_value_bounds: Vec::new(),
            batching_bounds_set: false,
            differentiation_value_bounds: Vec::new(),
            differentiation_value_bounds_set: false,
            transposition_value_bounds: Vec::new(),
            transposition_value_bounds_set: false,
            kind,
            errors: Vec::new(),
        }
    }

    /// Generates an implementation of [`Operation`] (together with its [`InterpretableOperation`],
    /// [`InterpretableProgramOperation`], [`PartiallyEvaluatableOperation`], [`Display`], and conversion companions)
    /// for the provided input. Refer to the documentation of the [`Operation`] trait for information on how to use
    /// this macro and on the shape of the generated code.
    pub(crate) fn generate_operation_impl(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
        Self::generate_derive_impl(input, DeriveKind::Operation, Self::generate_operation)
    }

    /// Generates an implementation of [`BatchableOperation`] (together with its [`BatchableProgramOperation`]
    /// companion) for the provided input. Refer to the documentation of the [`BatchableOperation`] trait for
    /// information on how to use this macro and on the shape of the generated code.
    pub(crate) fn generate_batchable_operation_impl(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
        Self::generate_derive_impl(input, DeriveKind::BatchableOperation, Self::generate_batchable_operation)
    }

    /// Generates an implementation of [`DifferentiableOperation`] (together with its
    /// [`DifferentiableProgramOperation`] and [`LinearizableProgramOperation`] companions) for the provided input.
    /// Refer to the documentation of the [`DifferentiableOperation`] trait for information on how to use this macro
    /// and on the shape of the generated code.
    pub(crate) fn generate_differentiable_operation_impl(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
        Self::generate_derive_impl(input, DeriveKind::DifferentiableOperation, Self::generate_differentiable_operation)
    }

    /// Generates an implementation of [`TransposableOperation`] (together with its [`TransposableProgramOperation`]
    /// companion) for the provided input. Refer to the documentation of the [`TransposableOperation`] trait for
    /// information on how to use this macro and on the shape of the generated code.
    pub(crate) fn generate_transposable_operation_impl(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
        Self::generate_derive_impl(input, DeriveKind::TransposableOperation, Self::generate_transposable_operation)
    }

    /// Drives one operation derive: parses the input, replaces any instances of [`Self`] with its fully-qualified
    /// path (which is necessary in order to be able to handle recursive operation enums), extracts the shared
    /// `#[ryft(...)]` attributes, invokes the derive-specific `generate` function, and raises any errors that were
    /// accumulated along the way as a combined compile-time error.
    ///
    /// # Parameters
    ///
    ///   * `input` - Derive macro input.
    ///   * `kind` - [`DeriveKind`] identifying the operation derive being generated.
    ///   * `generate` - Derive-specific code generation function.
    fn generate_derive_impl(
        input: proc_macro::TokenStream,
        kind: DeriveKind,
        generate: fn(&mut CodeGenerator, &syn::DeriveInput) -> TokenStream,
    ) -> proc_macro::TokenStream {
        let mut input = syn::parse_macro_input!(input as syn::DeriveInput);
        replace_self_type(&mut input);

        let mut generator = CodeGenerator::new(kind);
        generator.extract_attributes(&input);
        if let Some(error) = generator.compile_error() {
            return error.into();
        }

        let code = generate(&mut generator, &input);
        if let Some(error) = generator.compile_error() {
            return error.into();
        }
        code.into()
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
    /// [`CodeGenerator::ryft_crate`] and the owned `#[ryft(bounds(...))]` bound kinds, if it is able to successfully
    /// extract the attribute values, and it also infers [`CodeGenerator::operation_type`] from the enum's
    /// `Value<Type = T>` generic bounds (there must be exactly one distinct such bound argument and an error is
    /// generated if there are zero or more than one).
    fn extract_attributes(&mut self, input: &syn::DeriveInput) {
        let mut ryft_crate = Attribute::new(CRATE_ATTRIBUTE);
        input.attrs.iter().filter(|attr| attr.path() == &RYFT_ATTRIBUTE).for_each(|attr| {
            attr.parse_nested_meta(|meta| match &meta.path {
                path if path == &CRATE_ATTRIBUTE => ryft_crate.set(&meta),
                path if path == &BOUNDS_ATTRIBUTE => self.extract_bounds_attribute(&meta),
                _ => Err(meta.error(format_args!(
                    "invalid '#[ryft(...)]' attribute: '{}'; these are the attributes that are supported here: {:?}",
                    meta.path.to_token_stream().to_string().replace(' ', ""),
                    VALID_CONTAINER_ATTRIBUTES,
                ))),
            })
            .unwrap_or_else(|error| self.errors.push(error));
        });

        if let Some(ryft_crate) = ryft_crate.get() {
            self.ryft_crate = ryft_crate;
        }
        let inferred_types = unique_value_bound_arguments(&input.generics);
        match inferred_types.as_slice() {
            [operation_type] => self.operation_type = operation_type.clone(),
            [] => self.add_error(
                &input.ident,
                "could not infer an operation type because no generic parameter is bounded by 'Value<Type = T>'",
            ),
            _ => self.add_error(
                &input.generics,
                "could not infer a unique operation type because multiple distinct 'Value<Type = T>' bounds are present",
            ),
        }
    }

    /// Extracts a `#[ryft(bounds(...))]` attribute. Each bound kind is stored only when this [`CodeGenerator`]'s
    /// [`DeriveKind`] owns it and is otherwise parsed and discarded, so that the four operation derives can share
    /// the same `#[ryft(...)]` attribute namespace on one enum (refer to the documentation of
    /// [`CodeGenerator::kind`] for more information).
    fn extract_bounds_attribute(&mut self, meta: &syn::meta::ParseNestedMeta) -> syn::Result<()> {
        meta.parse_nested_meta(|meta| {
            // Identify the bound kind and destructure the owning derive's storage for it.
            let (kind, owner, bounds, bounds_set) = match &meta.path {
                path if path == &INTERPRETATION_ATTRIBUTE => (
                    "interpretation",
                    DeriveKind::Operation,
                    &mut self.interpretation_value_bounds,
                    &mut self.interpretation_value_bounds_set,
                ),
                path if path == &PARTIAL_EVALUATION_ATTRIBUTE => (
                    "partial_evaluation",
                    DeriveKind::Operation,
                    &mut self.partial_evaluation_value_bounds,
                    &mut self.partial_evaluation_value_bounds_set,
                ),
                path if path == &BATCHING_ATTRIBUTE => (
                    "batching",
                    DeriveKind::BatchableOperation,
                    &mut self.batching_value_bounds,
                    &mut self.batching_bounds_set,
                ),
                path if path == &DIFFERENTIATION_ATTRIBUTE => (
                    "differentiation",
                    DeriveKind::DifferentiableOperation,
                    &mut self.differentiation_value_bounds,
                    &mut self.differentiation_value_bounds_set,
                ),
                path if path == &TRANSPOSITION_ATTRIBUTE => (
                    "transposition",
                    DeriveKind::TransposableOperation,
                    &mut self.transposition_value_bounds,
                    &mut self.transposition_value_bounds_set,
                ),
                _ => {
                    return Err(meta.error(format_args!(
                        "invalid '#[ryft(bounds(...))]' attribute: '{}'; only 'interpretation(...)', \
                         'partial_evaluation(...)', 'batching(...)', 'differentiation(...)', and \
                         'transposition(...)' are supported here",
                        meta.path.to_token_stream().to_string().replace(' ', ""),
                    )));
                }
            };
            let parsed_bounds = parse_bounds(&meta, kind)?;
            if self.kind == owner {
                if *bounds_set {
                    return Err(meta.error(format_args!("duplicate ryft attribute 'bounds({kind}(...))'")));
                }
                *bounds_set = true;
                *bounds = parsed_bounds;
            }
            Ok(())
        })
    }

    /// Extracts the [`OperationVariant`]s that are contained in the provided [`syn::DeriveInput`]. This function
    /// also checks that the input is an enum, because the operation derives do not support structs or unions.
    fn extract_variants(&mut self, input: &syn::DeriveInput) -> Vec<OperationVariant> {
        let syn::Data::Enum(data) = &input.data else {
            self.add_error(&input.ident, "the '#[derive(Operation)]' macro only supports enums");
            return Vec::new();
        };

        data.variants
            .iter()
            .filter_map(|variant| self.extract_variant(&input.ident, &input.generics, variant))
            .collect()
    }

    /// Extracts one [`OperationVariant`] from the provided [`syn::Variant`]. This function also checks that the
    /// variant is a tuple variant with exactly one payload field and rejects nested `#[ryft(...)]` attributes on
    /// that field.
    ///
    /// # Parameters
    ///
    ///   * `enum_name` - [`syn::Ident`] of the enum being derived, used to detect recursive payloads.
    ///   * `generics` - [`syn::Generics`] of the enum being derived, used to detect bare generic payloads.
    ///   * `variant` - [`syn::Variant`] from which to extract an [`OperationVariant`].
    fn extract_variant(
        &mut self,
        enum_name: &syn::Ident,
        generics: &syn::Generics,
        variant: &syn::Variant,
    ) -> Option<OperationVariant> {
        let batching_active = self.extract_variant_attributes(&variant.attrs);
        let syn::Fields::Unnamed(fields) = &variant.fields else {
            self.add_error(&variant.ident, "operation enum variants must be tuple variants with one payload field");
            return None;
        };
        if fields.unnamed.len() != 1 {
            self.add_error(&variant.fields, "operation enum variants must have exactly one payload field");
            return None;
        }
        let field = fields.unnamed.first().expect("expected one payload field");
        self.reject_nested_attributes(&field.attrs);

        let payload_type = field.ty.clone();
        let (operation_type, is_boxed) = boxed_inner_type(&payload_type)
            .map(|operation_type| (operation_type, true))
            .unwrap_or_else(|| (payload_type.clone(), false));
        let skip_conversions = bare_generic_parameter(&payload_type, generics).is_some();
        let is_recursive_payload = type_mentions_ident(&operation_type, enum_name);

        Some(OperationVariant {
            ident: variant.ident.clone(),
            operation_type,
            is_boxed,
            skip_conversions,
            is_recursive_payload,
            batching_active,
        })
    }

    /// Extracts supported variant-level `#[ryft(...)]` attributes, returning whether the variant carries
    /// `#[ryft(batching(active))]`. The marker is owned (documented and consumed) by `#[derive(BatchableOperation)]`
    /// and tolerated by the sibling operation derives so combined derives compile.
    fn extract_variant_attributes(&mut self, attributes: &[syn::Attribute]) -> bool {
        let mut batching_active = false;
        attributes.iter().filter(|attr| attr.path() == &RYFT_ATTRIBUTE).for_each(|attr| {
            attr.parse_nested_meta(|meta| match &meta.path {
                path if path == &BATCHING_ATTRIBUTE => meta.parse_nested_meta(|meta| match &meta.path {
                    path if path == &BATCHING_ACTIVE_ATTRIBUTE => {
                        batching_active = true;
                        Ok(())
                    }
                    _ => Err(meta.error(format_args!(
                        "invalid '#[ryft(batching(...))]' variant attribute: '{}'; only 'active' is supported here",
                        meta.path.to_token_stream().to_string().replace(' ', ""),
                    ))),
                }),
                _ => Err(meta.error(NESTED_ATTRIBUTE_ERROR)),
            })
            .unwrap_or_else(|error| self.errors.push(error));
        });
        batching_active
    }

    /// Rejects nested `#[ryft(...)]` attributes, which are only supported at the top level for operation enums and
    /// at the variant level for the `#[ryft(batching(active))]` marker, but never on payload fields.
    fn reject_nested_attributes(&mut self, attributes: &[syn::Attribute]) {
        attributes
            .iter()
            .filter(|attr| attr.path() == &RYFT_ATTRIBUTE)
            .for_each(|attr| self.add_error(attr, NESTED_ATTRIBUTE_ERROR));
    }

    /// Generates the `Operation` derive output: the [`Operation`] dispatcher, the [`InterpretableOperation`]
    /// dispatcher, the [`InterpretableProgramOperation`] witness for nested flat programs, the
    /// [`PartiallyEvaluatableOperation`] dispatcher, the [`Display`] implementation, and the `From` and borrowed
    /// `TryFrom` payload conversions. Refer to the documentation of the [`Operation`] trait for information on the
    /// shape of the generated code and on how the interpretation value types are inferred from the enum's
    /// `Value<Type = T>` generic parameters.
    fn generate_operation(&mut self, input: &syn::DeriveInput) -> TokenStream {
        let variants = self.extract_variants(input);
        if self.compile_error().is_some() {
            return TokenStream::new();
        }

        let enum_name = &input.ident;
        let conversion_generics = input.generics.without_defaults();
        let (_, conversion_ty_generics, _) = conversion_generics.split_for_impl();
        let conversion_self_type: syn::Type = syn::parse_quote!(#enum_name #conversion_ty_generics);
        let ryft = &self.ryft_crate;
        let primary_type = &self.operation_type;

        let operation_self_type = conversion_self_type.clone();
        let operation_generics = self.operation_generics(&input.generics, &variants);
        let (operation_impl_generics, _, operation_where_clause) = operation_generics.split_for_impl();
        let value_type_parameters = value_type_parameters(&input.generics, primary_type);
        let Some(program_constant_type) = value_type_parameters.first().cloned() else {
            self.add_error(
                &input.generics,
                "could not infer the program constant value type for '#[derive(Operation)]'",
            );
            return TokenStream::new();
        };
        let has_separate_interpretation_value_type = value_type_parameters.len() == 1;
        let interpretation_value_type: syn::Type = if has_separate_interpretation_value_type {
            syn::parse_quote!(__InterpretationValue)
        } else {
            syn::parse_quote!(#program_constant_type)
        };
        let program_value_substitutions = program_value_substitutions(&value_type_parameters, &program_constant_type);
        let program_operation_self_type =
            substitute_type_idents(&operation_self_type, program_value_substitutions.as_slice());
        let interpretation_self_type = program_operation_self_type.clone();

        // The `InterpretableOperation` dispatcher shares its generics with the `InterpretableProgramOperation`
        // witness below and additionally requires the witness itself so that higher-order payload rules can
        // recursively interpret their nested programs.
        let mut interpretation_generics = self.interpretation_generics(
            input,
            &variants,
            program_value_substitutions.as_slice(),
            &program_constant_type,
            &interpretation_value_type,
            &interpretation_self_type,
            has_separate_interpretation_value_type,
        );
        interpretation_generics.make_where_clause().predicates.push(syn::parse_quote! {
            #interpretation_self_type:
                #ryft::InterpretableProgramOperation<
                    #interpretation_value_type,
                    __InterpretationContext,
                    #program_constant_type,
                >
        });
        let (interpretation_impl_generics, _, interpretation_where_clause) = interpretation_generics.split_for_impl();

        let program_interpretation_generics = self.interpretation_generics(
            input,
            &variants,
            program_value_substitutions.as_slice(),
            &program_constant_type,
            &interpretation_value_type,
            &program_operation_self_type,
            has_separate_interpretation_value_type,
        );
        let program_constant_lift = if has_separate_interpretation_value_type {
            quote!(context.constant(constant.clone()))
        } else {
            quote!(Ok(constant.clone()))
        };
        let (program_interpretation_impl_generics, _, program_interpretation_where_clause) =
            program_interpretation_generics.split_for_impl();

        // Partial evaluation is generic over the known-side context `__Context`, pinned to the program's constant
        // value type and to this enum as its operation family, so one generated implementation serves both eager
        // known-side contexts (values equal to the program constants) and staging ones (values that are tracers into
        // an outer program). Every variant forwards to its payload's per-operation rule, exactly like interpretation;
        // most payloads use the default rule and only nested-program payloads override it.
        let partial_evaluation_self_type = program_operation_self_type.clone();
        let partial_evaluation_value_type = program_constant_type.clone();
        let mut partial_evaluation_generics = substitute_generics(
            &self.operation_generics(&input.generics, &variants),
            program_value_substitutions.as_slice(),
        );
        partial_evaluation_generics.params.push(syn::parse_quote!(__Context));
        let partial_evaluation_where_clause = partial_evaluation_generics.make_where_clause();
        partial_evaluation_where_clause.predicates.push(syn::parse_quote! {
            __Context: #ryft::Context<
                Type = #primary_type,
                Constant = #partial_evaluation_value_type,
                Operation = #partial_evaluation_self_type,
            >
        });
        partial_evaluation_where_clause
            .predicates
            .push(syn::parse_quote!(#partial_evaluation_self_type: #ryft::Operation<#primary_type>));
        partial_evaluation_where_clause
            .predicates
            .push(syn::parse_quote!(#partial_evaluation_self_type: ::std::clone::Clone));
        // Extra partial-evaluation-only bounds requested via `#[ryft(bounds(partial_evaluation(...)))]`. These apply
        // to *both* partial-evaluation value spaces, because a recursive payload's rule may need them on either side:
        // the scan/while invariance fixed points compare known values, which flow as `__Context::Value` (tracers
        // under a staging known-side context, hence e.g. `PartialEq`), while the condition rule inspects its
        // concretized predicate in the program-constant space (hence e.g. `BooleanLike`). Applying one declared bound
        // list to both spaces keeps the attribute surface small; under an eager known-side context the two spaces
        // coincide anyway.
        {
            let partial_evaluation_value_type: syn::Type = syn::parse_quote!(#partial_evaluation_value_type);
            add_value_bounds(
                partial_evaluation_where_clause,
                &partial_evaluation_value_type,
                self.partial_evaluation_value_bounds.as_slice(),
            );
            let partial_evaluation_flow_type: syn::Type = syn::parse_quote!(<__Context as #ryft::Domain>::Value);
            add_value_bounds(
                partial_evaluation_where_clause,
                &partial_evaluation_flow_type,
                self.partial_evaluation_value_bounds.as_slice(),
            );
        }
        // Skip recursive payload variants (those that mention this enum) from the where-clause, exactly like the
        // interpretation and transposition impls: a `where ConditionOperation<.., Self, ..>:
        // PartiallyEvaluatableOperation` edge would form a genuine trait-solver cycle with the condition payload's own
        // `O: PartiallyEvaluatableProgramOperation` recursion bound (E0275). The match arm for each recursive variant
        // still calls that variant's rule, which is discharged as a body obligation against this very impl.
        partial_evaluation_where_clause.predicates.extend(
            variants.iter().filter(|variant| !variant.is_recursive_payload && !variant.skip_conversions).map(
                |variant| {
                    let operation_type =
                        substitute_type_idents(&variant.operation_type, program_value_substitutions.as_slice());
                    let predicate: syn::WherePredicate = syn::parse_quote! {
                        #operation_type: #ryft::partial::PartiallyEvaluatableOperation<__Context>
                    };
                    predicate
                },
            ),
        );
        let partial_evaluation_arms = variants.iter().map(|variant| {
            let variant_ident = &variant.ident;
            if variant.skip_conversions {
                quote! {
                    Self::#variant_ident(_) => context.fold_or_residualize(self.clone(), inputs),
                }
            } else {
                let operation_type =
                    substitute_type_idents(&variant.operation_type, program_value_substitutions.as_slice());
                let receiver = variant.receiver();
                quote! {
                    Self::#variant_ident(operation) => {
                        <#operation_type as #ryft::partial::PartiallyEvaluatableOperation<__Context>>::
                            partially_evaluate(#receiver, context, inputs)
                    },
                }
            }
        });
        let partial_evaluation_body = quote! {
            match self {
                #(#partial_evaluation_arms)*
            }
        };
        let (partial_evaluation_impl_generics, _, partial_evaluation_where_clause) =
            partial_evaluation_generics.split_for_impl();

        let name_arms = variants.iter().map(|variant| {
            let variant_ident = &variant.ident;
            let payload_operation_type = &variant.operation_type;
            let receiver = variant.receiver();
            quote! {
                Self::#variant_ident(operation) => {
                    <#payload_operation_type as #ryft::Operation<#primary_type>>::name(#receiver)
                },
            }
        });
        let infer_output_type_arms = variants.iter().map(|variant| {
            let variant_ident = &variant.ident;
            let payload_operation_type = &variant.operation_type;
            let receiver = variant.receiver();
            quote! {
                Self::#variant_ident(operation) => {
                    <#payload_operation_type as #ryft::Operation<#primary_type>>::infer_output_types(
                        #receiver,
                        input_types,
                    )
                },
            }
        });
        let effects_arms = variants.iter().map(|variant| {
            let variant_ident = &variant.ident;
            let payload_operation_type = &variant.operation_type;
            let receiver = variant.receiver();
            quote! {
                Self::#variant_ident(operation) => {
                    <#payload_operation_type as #ryft::Operation<#primary_type>>::effects(#receiver)
                },
            }
        });
        let render_arms = variants.iter().map(|variant| {
            let variant_ident = &variant.ident;
            let payload_operation_type = &variant.operation_type;
            let receiver = variant.receiver();
            quote! {
                Self::#variant_ident(operation) => {
                    <#payload_operation_type as #ryft::Operation<#primary_type>>::render(
                        #receiver,
                        formatter,
                        indentation,
                    )
                },
            }
        });
        // The same arms serve both the `InterpretableOperation` dispatcher (where `inputs` is the method
        // argument) and the `InterpretableProgramOperation` witness (where `inputs` is bound by the program-walk
        // closure below), so they are collected once.
        let interpretation_arms = variants
            .iter()
            .map(|variant| {
                let variant_ident = &variant.ident;
                let payload_operation_type =
                    substitute_type_idents(&variant.operation_type, program_value_substitutions.as_slice());
                let receiver = variant.receiver();
                quote! {
                    Self::#variant_ident(operation) => {
                        <#payload_operation_type as #ryft::InterpretableOperation<
                            #interpretation_value_type,
                            __InterpretationContext,
                        >>::interpret(#receiver, context, inputs)
                    },
                }
            })
            .collect::<Vec<_>>();
        let program_interpretation_body = quote! {
            program.interpret_with(
                input,
                |_, constant| #program_constant_lift,
                |instruction, inputs| {
                    match instruction.operation() {
                        #(#interpretation_arms)*
                    }
                },
            )
        };
        let conversion_impls = variants
            .iter()
            .filter(|variant| !variant.skip_conversions)
            .map(|variant| self.generate_conversion_impl(&conversion_generics, &conversion_self_type, variant));

        const_block(quote! {
            #[automatically_derived]
            impl #operation_impl_generics #ryft::Operation<#primary_type> for #operation_self_type
            #operation_where_clause
            {
                fn name(&self) -> &'static str {
                    match self {
                        #(#name_arms)*
                    }
                }

                fn infer_output_types(
                    &self,
                    input_types: &[#primary_type],
                ) -> ::std::result::Result<::std::vec::Vec<#primary_type>, #ryft::TypeError> {
                    match self {
                        #(#infer_output_type_arms)*
                    }
                }

                fn effects(&self) -> #ryft::Effects {
                    match self {
                        #(#effects_arms)*
                    }
                }

                fn render(
                    &self,
                    formatter: &mut ::std::fmt::Formatter<'_>,
                    indentation: usize,
                ) -> ::std::fmt::Result {
                    match self {
                        #(#render_arms)*
                    }
                }
            }

            #[automatically_derived]
            impl #interpretation_impl_generics
                #ryft::InterpretableOperation<#interpretation_value_type, __InterpretationContext>
                for #interpretation_self_type
            #interpretation_where_clause
            {
                fn interpret(
                    &self,
                    context: &__InterpretationContext,
                    inputs: &[#interpretation_value_type],
                ) -> ::std::result::Result<
                    ::std::vec::Vec<#interpretation_value_type>,
                    #ryft::ProgramError,
                > {
                    match self {
                        #(#interpretation_arms)*
                    }
                }
            }

            #[automatically_derived]
            impl #program_interpretation_impl_generics
                #ryft::InterpretableProgramOperation<
                    #interpretation_value_type,
                    __InterpretationContext,
                    #program_constant_type,
                >
                for #program_operation_self_type
            #program_interpretation_where_clause
            {
                fn interpret_program(
                    context: &__InterpretationContext,
                    program: &#ryft::Program<
                        #program_constant_type,
                        Self,
                        ::std::vec::Vec<#program_constant_type>,
                        ::std::vec::Vec<#program_constant_type>,
                    >,
                    input: ::std::vec::Vec<#interpretation_value_type>,
                ) -> ::std::result::Result<
                    ::std::vec::Vec<#interpretation_value_type>,
                    #ryft::ProgramError,
                > {
                    #program_interpretation_body
                }
            }

            #[automatically_derived]
            impl #partial_evaluation_impl_generics
                #ryft::partial::PartiallyEvaluatableOperation<__Context>
                for #partial_evaluation_self_type
            #partial_evaluation_where_clause
            {
                fn partially_evaluate(
                    &self,
                    context: &#ryft::partial::PartialEvaluationContext<__Context>,
                    inputs: &[#ryft::partial::PartialEvaluationValue<
                        <__Context as #ryft::Domain>::Value,
                    >],
                ) -> ::std::result::Result<
                    ::std::vec::Vec<#ryft::partial::PartialEvaluationValue<
                        <__Context as #ryft::Domain>::Value,
                    >>,
                    #ryft::ProgramError,
                > {
                    #partial_evaluation_body
                }
            }

            #[automatically_derived]
            impl #operation_impl_generics ::std::fmt::Display for #operation_self_type
            #operation_where_clause
            {
                fn fmt(&self, formatter: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                    <Self as #ryft::Operation<#primary_type>>::render(self, formatter, 0)
                }
            }

            #(#conversion_impls)*
        })
    }

    /// Generates the `BatchableOperation` derive output: the staged tracer-level and eager value-level
    /// `BatchableOperation` dispatchers plus the `BatchableProgramOperation` witness for nested-program batching.
    ///
    /// The generated where clauses follow the same structure as the differentiation derive: each non-recursive
    /// payload carries its batching obligation as a per-variant `BatchableOperation` predicate (at the context its
    /// arm dispatches at), recursive payloads (those mentioning `Self`) are discharged as definition-time body
    /// checks against the author-supplied leaf capability bounds (`#[ryft(bounds(batching(...)))]`) and the
    /// `BatchableProgramOperation` fixed-point witness, and the witness impl itself spells only the constant-side
    /// leaves because its replay runs over a concrete tracing context where everything else resolves concretely.
    fn generate_batchable_operation(&mut self, input: &syn::DeriveInput) -> TokenStream {
        let variants = self.extract_variants(input);
        if self.compile_error().is_some() {
            return TokenStream::new();
        }
        for variant in &variants {
            if variant.batching_active && variant.is_recursive_payload {
                self.add_error(
                    &variant.ident,
                    "'#[ryft(batching(active))]' is redundant on recursive payloads, which always dispatch at the \
                     active batching context; remove the marker",
                );
            }
        }
        if self.compile_error().is_some() {
            return TokenStream::new();
        }

        let enum_name = &input.ident;
        let conversion_generics = input.generics.without_defaults();
        let (_, conversion_ty_generics, _) = conversion_generics.split_for_impl();
        let conversion_self_type: syn::Type = syn::parse_quote!(#enum_name #conversion_ty_generics);
        let ryft = &self.ryft_crate;
        let primary_type = &self.operation_type;

        let value_type_parameters = value_type_parameters(&input.generics, primary_type);
        let Some(program_constant_type) = value_type_parameters.first().cloned() else {
            self.add_error(
                &input.generics,
                "could not infer the program constant value type for '#[derive(BatchableOperation)]'",
            );
            return TokenStream::new();
        };
        let program_value_substitutions = program_value_substitutions(&value_type_parameters, &program_constant_type);
        let batching_self_type = substitute_type_idents(&conversion_self_type, program_value_substitutions.as_slice());
        let constant_type: syn::Type = syn::parse_quote!(#program_constant_type);

        // Staged tracer-level dispatcher. Primitive (non-recursive, unmarked) arms dispatch at the parent staging
        // context — the flowing physical values are parent-trace values — while recursive and `active`-marked arms
        // dispatch at the batching context itself for its axis metadata. No tracer capability leaves are spelled:
        // the recursive staged rules carry operation-shaped `From<XOperation>` bounds that the `Operation = Self`
        // projection discharges structurally, and the tracer capability blankets bridge them back to the value
        // capabilities the shared rule bodies call.
        let tracer_type: syn::Type = syn::parse_quote! {
            <__ParentContext as #ryft::Domain>::Value
        };
        let mut staged_generics = substitute_generics(
            &self.operation_generics(&input.generics, &variants),
            program_value_substitutions.as_slice(),
        );
        staged_generics.params.push(syn::parse_quote!(__ParentContext));
        let staged_where_clause = staged_generics.make_where_clause();
        staged_where_clause.predicates.push(syn::parse_quote! {
            __ParentContext: #ryft::Context<
                Type = #primary_type,
                Constant = #program_constant_type,
                Operation = #batching_self_type,
            >
        });
        staged_where_clause
            .predicates
            .extend(variants.iter().filter(|variant| !variant.is_recursive_payload).map(|variant| {
                let operation_type =
                    substitute_type_idents(&variant.operation_type, program_value_substitutions.as_slice());
                let predicate: syn::WherePredicate = if variant.batching_active {
                    syn::parse_quote! {
                        #operation_type:
                            #ryft::BatchableOperation<#tracer_type, #ryft::BatchingContext<__ParentContext>>
                    }
                } else {
                    syn::parse_quote!(#operation_type: #ryft::BatchableOperation<#tracer_type, __ParentContext>)
                };
                predicate
            }));
        staged_where_clause
            .predicates
            .push(syn::parse_quote!(#batching_self_type: #ryft::BatchableProgramOperation<#program_constant_type>));
        // Recursive higher-order rules (scan/condition/while/custom derivatives) dispatch at this batching context and
        // compute over the flowing parent value, so they need the author-declared batching value capabilities on that
        // value plus the parent context's `Zero` leaf for accumulator seeding — mirroring the eager dispatcher below.
        // The declared bounds are written against the enum's value parameter, but the staged dispatcher flows the
        // parent context's value, so substitute that value into their associated-type constraints (e.g.
        // `Select<Condition = V>` becomes `Select<Condition = <__ParentContext as Domain>::Value>`).
        let staged_value_substitutions = [(program_constant_type.clone(), tracer_type.clone())];
        let mut staged_value_substituter = TypeIdentSubstituter { substitutions: &staged_value_substitutions };
        let staged_value_bounds = self
            .batching_value_bounds
            .iter()
            .map(|bound| {
                let mut bound = bound.clone();
                staged_value_substituter.visit_type_param_bound_mut(&mut bound);
                bound
            })
            .collect::<Vec<_>>();
        add_value_bounds(staged_where_clause, &tracer_type, staged_value_bounds.as_slice());
        staged_where_clause.predicates.push(syn::parse_quote!(__ParentContext: #ryft::Zero<#tracer_type>));
        let staged_needs_parent_context =
            variants.iter().any(|variant| !variant.is_recursive_payload && !variant.batching_active);
        let staged_parent_context_binding =
            if staged_needs_parent_context { quote!(let parent = context.parent();) } else { TokenStream::new() };
        let staged_arms = variants.iter().map(|variant| {
            let variant_ident = &variant.ident;
            let operation_type =
                substitute_type_idents(&variant.operation_type, program_value_substitutions.as_slice());
            let receiver = variant.receiver();
            if variant.is_recursive_payload || variant.batching_active {
                quote! {
                    Self::#variant_ident(operation) => {
                        <#operation_type as #ryft::BatchableOperation<
                            #tracer_type,
                            #ryft::BatchingContext<__ParentContext>,
                        >>::batch(#receiver, context, inputs)
                    },
                }
            } else {
                quote! {
                    Self::#variant_ident(operation) => {
                        <#operation_type as #ryft::BatchableOperation<#tracer_type, __ParentContext>>::batch(
                            #receiver,
                            parent,
                            inputs,
                        )
                    },
                }
            }
        });
        let (staged_impl_generics, _, staged_where_clause) = staged_generics.split_for_impl();
        let staged_impl = quote! {
            #[automatically_derived]
            impl #staged_impl_generics
                #ryft::BatchableOperation<#tracer_type, #ryft::BatchingContext<__ParentContext>>
                for #batching_self_type
            #staged_where_clause
            {
                fn batch(
                    &self,
                    context: &#ryft::BatchingContext<__ParentContext>,
                    inputs: &[#ryft::ArrayBatch<#tracer_type>],
                ) -> ::std::result::Result<::std::vec::Vec<#ryft::ArrayBatch<#tracer_type>>, #ryft::BatchingError> {
                    #staged_parent_context_binding
                    match self {
                        #(#staged_arms)*
                    }
                }
            }
        };

        // Eager value-level dispatcher: every arm dispatches at the eager context. The hardcoded context `Zero`
        // leaf backs recursive scan rules' accumulator seeding, mirroring the differentiation derive's hardcoded
        // context `Zero` bound.
        let eager_context: syn::Type =
            syn::parse_quote!(#ryft::EagerContext<#program_constant_type, #batching_self_type>);
        let mut eager_generics = substitute_generics(
            &self.operation_generics(&input.generics, &variants),
            program_value_substitutions.as_slice(),
        );
        let eager_where_clause = eager_generics.make_where_clause();
        add_value_bounds(eager_where_clause, &constant_type, self.batching_value_bounds.as_slice());
        eager_where_clause
            .predicates
            .push(syn::parse_quote!(#eager_context: #ryft::Zero<#program_constant_type>));
        eager_where_clause
            .predicates
            .extend(variants.iter().filter(|variant| !variant.is_recursive_payload).map(|variant| {
                let operation_type =
                    substitute_type_idents(&variant.operation_type, program_value_substitutions.as_slice());
                let predicate: syn::WherePredicate = syn::parse_quote! {
                    #operation_type: #ryft::BatchableOperation<#program_constant_type, #eager_context>
                };
                predicate
            }));
        let eager_arms = variants.iter().map(|variant| {
            let variant_ident = &variant.ident;
            let operation_type =
                substitute_type_idents(&variant.operation_type, program_value_substitutions.as_slice());
            let receiver = variant.receiver();
            quote! {
                Self::#variant_ident(operation) => {
                    <#operation_type as #ryft::BatchableOperation<#program_constant_type, #eager_context>>::batch(
                        #receiver,
                        context,
                        inputs,
                    )
                },
            }
        });
        let (eager_impl_generics, _, eager_where_clause) = eager_generics.split_for_impl();
        let eager_impl = quote! {
            #[automatically_derived]
            impl #eager_impl_generics #ryft::BatchableOperation<#program_constant_type, #eager_context>
                for #batching_self_type
            #eager_where_clause
            {
                fn batch(
                    &self,
                    context: &#eager_context,
                    inputs: &[#ryft::ArrayBatch<#program_constant_type>],
                ) -> ::std::result::Result<
                    ::std::vec::Vec<#ryft::ArrayBatch<#program_constant_type>>,
                    #ryft::BatchingError,
                > {
                    match self {
                        #(#eager_arms)*
                    }
                }
            }
        };

        // Program-level witness: the `batch_program` replay runs over the concrete tracing context
        // `TracingContext<Constant, Self, Constant>`, so the staged dispatcher's obligations are spelled here
        // instantiated at that context — per-variant predicates that a payload rule pinned to a concrete constant
        // type (e.g. a backend `jit_call`) cannot resolve at a generic constant parameter stay transported, while
        // everything context-shaped resolves concretely. Keeping the enum's own batching obligation out of the
        // where clause is what lets the staged dispatcher require `Self: BatchableProgramOperation<..>` without
        // unbounded recursion.
        let program_trace_context: syn::Type = syn::parse_quote! {
            #ryft::TracingContext<#program_constant_type, #batching_self_type, #program_constant_type>
        };
        let program_tracer_type: syn::Type = syn::parse_quote! {
            #ryft::Tracer<#program_trace_context>
        };
        let mut program_generics = substitute_generics(
            &self.operation_generics(&input.generics, &variants),
            program_value_substitutions.as_slice(),
        );
        let program_where_clause = program_generics.make_where_clause();
        program_where_clause.predicates.push(syn::parse_quote!(#program_constant_type: 'static));
        program_where_clause
            .predicates
            .extend(variants.iter().filter(|variant| !variant.is_recursive_payload).map(|variant| {
                let operation_type =
                    substitute_type_idents(&variant.operation_type, program_value_substitutions.as_slice());
                let predicate: syn::WherePredicate = if variant.batching_active {
                    syn::parse_quote! {
                        #operation_type: #ryft::BatchableOperation<
                            #program_tracer_type,
                            #ryft::BatchingContext<#program_trace_context>,
                        >
                    }
                } else {
                    syn::parse_quote! {
                        #operation_type: #ryft::BatchableOperation<#program_tracer_type, #program_trace_context>
                    }
                };
                predicate
            }));
        let (program_impl_generics, _, program_where_clause) = program_generics.split_for_impl();
        let program_impl = quote! {
            #[automatically_derived]
            impl #program_impl_generics #ryft::BatchableProgramOperation<#program_constant_type>
                for #batching_self_type
            #program_where_clause
            {
                fn batch_program(
                    program: &#ryft::Program<
                        #program_constant_type,
                        Self,
                        ::std::vec::Vec<#program_constant_type>,
                        ::std::vec::Vec<#program_constant_type>,
                    >,
                    axis_size: usize,
                    input_batch_axes: &[#ryft::BatchAxis],
                    output_axes_policy: #ryft::ProgramBatchingOutputAxesPolicy,
                ) -> ::std::result::Result<
                    (
                        #ryft::Program<
                            #program_constant_type,
                            Self,
                            ::std::vec::Vec<#program_constant_type>,
                            ::std::vec::Vec<#program_constant_type>,
                        >,
                        ::std::vec::Vec<#ryft::BatchAxis>,
                    ),
                    #ryft::BatchingError,
                > {
                    program.batched(axis_size, input_batch_axes, output_axes_policy)
                }
            }
        };

        const_block(quote! {
            #staged_impl
            #eager_impl
            #program_impl
        })
    }

    /// Generates the forward-mode (JVP) dispatcher of the `DifferentiableOperation` derive output.
    ///
    /// The generated implementation is generic over a `__DifferentiationContext` staging context pinned to the enum's
    /// primary type, program constant type, and the enum itself as its operation family. Every variant forwards to its
    /// payload's own `DifferentiableOperation` rule. Non-recursive payloads carry per-variant
    /// `DifferentiableOperation` predicates that transport each rule's own capability requirements, while recursive
    /// payloads are discharged as definition-time body obligations against the operation `From` conversions and the
    /// `MaybeZeroOperation` / `DifferentiableProgramOperation` / `LinearizableProgramOperation` fixed-point witnesses
    /// (a per-variant predicate for them would form a genuine trait-solver cycle).
    fn generate_differentiable_operation(&mut self, input: &syn::DeriveInput) -> TokenStream {
        let variants = self.extract_variants(input);
        if self.compile_error().is_some() {
            return TokenStream::new();
        }

        let enum_name = &input.ident;
        let conversion_generics = input.generics.without_defaults();
        let (_, conversion_ty_generics, _) = conversion_generics.split_for_impl();
        let conversion_self_type: syn::Type = syn::parse_quote!(#enum_name #conversion_ty_generics);
        let ryft = &self.ryft_crate;
        let primary_type = &self.operation_type;

        let value_type_parameters = value_type_parameters(&input.generics, primary_type);
        let Some(program_constant_type) = value_type_parameters.first().cloned() else {
            self.add_error(
                &input.generics,
                "could not infer the program constant value type for '#[derive(DifferentiableOperation)]'",
            );
            return TokenStream::new();
        };
        let program_value_substitutions = program_value_substitutions(&value_type_parameters, &program_constant_type);
        let differentiation_self_type =
            substitute_type_idents(&conversion_self_type, program_value_substitutions.as_slice());
        let mut differentiation_generics = substitute_generics(
            &self.operation_generics(&input.generics, &variants),
            program_value_substitutions.as_slice(),
        );
        differentiation_generics.params.push(syn::parse_quote!(__DifferentiationContext));
        let where_clause = differentiation_generics.make_where_clause();
        where_clause.predicates.push(syn::parse_quote! {
            __DifferentiationContext: #ryft::Context<
                Type = #primary_type,
                Constant = #program_constant_type,
                Operation = #differentiation_self_type,
            >
        });
        // The recursive higher-order payload rules materialize structural zero tangents at sub-program boundaries
        // through the context's `Zero` capability, so the dispatcher transports that requirement explicitly (the
        // per-variant predicates only cover non-recursive payloads). Staging contexts satisfy it through the blanket
        // staging `Zero` implementation and eager domains implement it directly.
        where_clause.predicates.push(syn::parse_quote! {
            __DifferentiationContext: #ryft::Zero<<__DifferentiationContext as #ryft::Domain>::Value>
        });
        // The `while` rule concretizes data-dependent loop predicates on the carried values when the context is
        // eager, so the dispatcher also transports the `BooleanLike` value requirement. Staged values
        // satisfy it through the tracer `BooleanLike` implementation (whose `boolean` defers with an error) and
        // eager values implement it directly.
        where_clause.predicates.push(syn::parse_quote! {
            <__DifferentiationContext as #ryft::Domain>::Value: #ryft::BooleanLike
        });
        where_clause
            .predicates
            .push(syn::parse_quote!(#differentiation_self_type: #ryft::Operation<#primary_type>));
        where_clause.predicates.push(syn::parse_quote!(#differentiation_self_type: ::std::clone::Clone));
        // The per-variant rules stage ordinary primal-enum operations for both the primal and the tangent side, so the
        // enum must offer the `From` conversion for every concrete payload. Bare generic payloads have no conversion
        // and instead carry their own forward-mode obligation directly.
        where_clause
            .predicates
            .extend(variants.iter().filter(|variant| !variant.skip_conversions).map(|variant| {
                let operation_type =
                    substitute_type_idents(&variant.operation_type, program_value_substitutions.as_slice());
                let predicate: syn::WherePredicate =
                    syn::parse_quote!(#differentiation_self_type: ::std::convert::From<#operation_type>);
                predicate
            }));
        // Non-recursive payloads carry their forward-mode obligation as a per-variant predicate, exactly like the
        // interpretation and partial-evaluation impls: the predicate transports each rule's own capability
        // requirements (e.g., `C::Value: Sin` for the sine rule) to the use site without the enum spelling them.
        // Recursive payloads (those mentioning `Self`) are skipped — a `ScanOperation<.., Self, ..>:
        // DifferentiableOperation<..>` predicate would re-enter the enum's own obligation and overflow the trait
        // solver — and are discharged as definition-time body checks against the fixed-point witnesses below.
        where_clause.predicates.extend(variants.iter().filter(|variant| !variant.is_recursive_payload).map(
            |variant| {
                let operation_type =
                    substitute_type_idents(&variant.operation_type, program_value_substitutions.as_slice());
                let predicate: syn::WherePredicate =
                    syn::parse_quote!(#operation_type: #ryft::DifferentiableOperation<__DifferentiationContext>);
                predicate
            },
        ));
        // The `From<ZeroOperation>`, `DifferentiableProgramOperation`, and `LinearizableProgramOperation` fixed-point
        // witnesses let higher-order payload rules (condition/while/scan) forward-differentiate and linearize their
        // nested programs in this same operation family without re-entering the enum's own `DifferentiableOperation`
        // obligation. Both program witnesses are required because the fused rules (`scan`/`condition`) stage through
        // `jvp_program` while the bounded `while` rule linearizes its body through `linearize_program`.
        where_clause.predicates.push(syn::parse_quote! {
            #differentiation_self_type:
                ::std::convert::From<#ryft::ZeroOperation<#primary_type>>
                + #ryft::DifferentiableProgramOperation<
                    #program_constant_type,
                    #differentiation_self_type,
                >
                + #ryft::LinearizableProgramOperation<
                    #program_constant_type,
                    #differentiation_self_type,
                >
        });

        let jvp_arms = variants.iter().map(|variant| {
            let variant_ident = &variant.ident;
            let operation_type =
                substitute_type_idents(&variant.operation_type, program_value_substitutions.as_slice());
            let receiver = variant.receiver();
            quote! {
                Self::#variant_ident(operation) => {
                    <#operation_type as #ryft::DifferentiableOperation<__DifferentiationContext>>::jvp(
                        #receiver,
                        context,
                        inputs,
                    )
                },
            }
        });
        let (differentiation_impl_generics, _, differentiation_where_clause) =
            differentiation_generics.split_for_impl();

        // Program-level witnesses backing the recursive higher-order rules: the fixed bodies discharge the enum's
        // full forward-mode obligation once, as a definition-time body check over the concrete linearization trace,
        // so the where clause spells only the constant-side leaves (`#[ryft(bounds(differentiation(...)))]`) and the
        // `From<ZeroOperation>` conversion the traits themselves require. Keeping the recursive
        // `Self: DifferentiableOperation<..>` bound out of the where clause is what lets the dispatcher above
        // require `Self: DifferentiableProgramOperation<..>` and `Self: LinearizableProgramOperation<..>` without
        // unbounded recursion.
        let program_type = quote! {
            #ryft::Program<
                #program_constant_type,
                Self,
                ::std::vec::Vec<#program_constant_type>,
                ::std::vec::Vec<#program_constant_type>,
            >
        };
        let mut witness_generics = substitute_generics(
            &self.operation_generics(&input.generics, &variants),
            program_value_substitutions.as_slice(),
        );
        let witness_where_clause = witness_generics.make_where_clause();
        add_value_bounds(
            witness_where_clause,
            &syn::parse_quote!(#program_constant_type),
            self.differentiation_value_bounds.as_slice(),
        );
        witness_where_clause.predicates.push(syn::parse_quote! {
            #differentiation_self_type: ::std::convert::From<#ryft::ZeroOperation<#primary_type>>
        });
        let (witness_impl_generics, _, witness_where_clause) = witness_generics.split_for_impl();
        let witness_impl = quote! {
            #[automatically_derived]
            impl #witness_impl_generics
                #ryft::DifferentiableProgramOperation<
                    #program_constant_type,
                    #differentiation_self_type,
                >
                for #differentiation_self_type
            #witness_where_clause
            {
                fn jvp_program(
                    program: &#program_type,
                ) -> ::std::result::Result<#program_type, #ryft::DifferentiationError> {
                    program.jvp()
                }
            }

            #[automatically_derived]
            impl #witness_impl_generics
                #ryft::LinearizableProgramOperation<
                    #program_constant_type,
                    #differentiation_self_type,
                >
                for #differentiation_self_type
            #witness_where_clause
            {
                fn linearize_program(
                    program: &#program_type,
                ) -> ::std::result::Result<
                    #ryft::Linearization<#program_constant_type, Self>,
                    #ryft::DifferentiationError,
                > {
                    program.linearize()
                }
            }
        };

        const_block(quote! {
            #[automatically_derived]
            impl #differentiation_impl_generics
                #ryft::DifferentiableOperation<__DifferentiationContext>
                for #differentiation_self_type
            #differentiation_where_clause
            {
                fn jvp(
                    &self,
                    context: &__DifferentiationContext,
                    inputs: &[#ryft::DifferentiationDual<
                        <__DifferentiationContext as #ryft::Domain>::Value,
                    >],
                ) -> ::std::result::Result<
                    ::std::vec::Vec<#ryft::DifferentiationDual<
                        <__DifferentiationContext as #ryft::Domain>::Value,
                    >>,
                    #ryft::DifferentiationError,
                > {
                    match self {
                        #(#jvp_arms)*
                    }
                }
            }

            #witness_impl
        })
    }

    /// Generates the `TransposableOperation` derive output: the `TransposableOperation` dispatcher and the
    /// `TransposableProgramOperation` witness for nested linear programs. Single-value-parameter enums without
    /// recursive payloads transpose over a separate generated `__TranspositionValue` parameter. Enums with two or
    /// more value parameters pin the transposition value to the first (tangent/cotangent) value parameter, and so
    /// do enums with recursive payloads (those mentioning `Self`): recursive higher-order payload rules name the
    /// operation-family fixed point through `TransposableProgramOperation` and pin their transposition value to the
    /// program constant type, so the witness's `Program::transpose_with_respect_to` call is only provable at that
    /// instantiation.
    fn generate_transposable_operation(&mut self, input: &syn::DeriveInput) -> TokenStream {
        let variants = self.extract_variants(input);
        if self.compile_error().is_some() {
            return TokenStream::new();
        }

        let enum_name = &input.ident;
        let conversion_generics = input.generics.without_defaults();
        let (_, conversion_ty_generics, _) = conversion_generics.split_for_impl();
        let conversion_self_type: syn::Type = syn::parse_quote!(#enum_name #conversion_ty_generics);
        let ryft = &self.ryft_crate;
        let primary_type = &self.operation_type;

        let operation_self_type = conversion_self_type.clone();
        let value_type_parameters = value_type_parameters(&input.generics, primary_type);
        let Some(program_constant_type) = value_type_parameters.first().cloned() else {
            self.add_error(
                &input.generics,
                "could not infer the program constant value type for '#[derive(TransposableOperation)]'",
            );
            return TokenStream::new();
        };
        let has_separate_transposition_value_type =
            value_type_parameters.len() == 1 && variants.iter().all(|variant| !variant.is_recursive_payload);
        let transposed_value_type: syn::Type = if has_separate_transposition_value_type {
            syn::parse_quote!(__TranspositionValue)
        } else {
            syn::parse_quote!(#program_constant_type)
        };
        let mut transposition_generics = self.operation_generics(&input.generics, &variants);
        if has_separate_transposition_value_type {
            transposition_generics.params.push(syn::parse_quote!(__TranspositionValue));
        }

        let transpose_bounds = variants
            .iter()
            .map(|variant| {
                let operation_type = &variant.operation_type;
                let predicate: syn::WherePredicate = syn::parse_quote! {
                    #operation_type: #ryft::TransposableOperation<
                        #transposed_value_type,
                        #operation_self_type,
                    >
                };
                predicate
            })
            .collect::<Vec<_>>();
        let program_transpose_bounds = variants.iter().filter(|variant| !variant.is_recursive_payload).map(|variant| {
            let operation_type = &variant.operation_type;
            let predicate: syn::WherePredicate = syn::parse_quote! {
                #operation_type: #ryft::TransposableOperation<
                    #transposed_value_type,
                    #operation_self_type,
                >
            };
            predicate
        });
        let where_clause = transposition_generics.make_where_clause();
        if has_separate_transposition_value_type {
            where_clause
                .predicates
                .push(syn::parse_quote!(#transposed_value_type: #ryft::Value<Type = #primary_type>));
            where_clause.predicates.extend(generic_parameter_bounds_as_predicates(
                &input.generics,
                &program_constant_type,
                &transposed_value_type,
            ));
        }
        // Extra transposition-only bounds requested via `#[ryft(bounds(transposition(...)))]`. These serve the same
        // role as bounds declared on the enum's own value parameter (which the generated implementations inherit)
        // without forcing the enum's stored constant type to carry transposition-only capabilities.
        add_value_bounds(where_clause, &transposed_value_type, self.transposition_value_bounds.as_slice());
        where_clause
            .predicates
            .push(syn::parse_quote!(#operation_self_type: #ryft::Operation<#primary_type>));
        where_clause.predicates.extend(transpose_bounds.iter().cloned());

        let mut program_transposition_generics = self.operation_generics(&input.generics, &variants);
        if has_separate_transposition_value_type {
            program_transposition_generics.params.push(syn::parse_quote!(__TranspositionValue));
        }
        let program_where_clause = program_transposition_generics.make_where_clause();
        if has_separate_transposition_value_type {
            program_where_clause
                .predicates
                .push(syn::parse_quote!(#transposed_value_type: #ryft::Value<Type = #primary_type>));
            program_where_clause.predicates.extend(generic_parameter_bounds_as_predicates(
                &input.generics,
                &program_constant_type,
                &transposed_value_type,
            ));
        }
        add_value_bounds(program_where_clause, &transposed_value_type, self.transposition_value_bounds.as_slice());
        program_where_clause
            .predicates
            .push(syn::parse_quote!(#operation_self_type: #ryft::Operation<#primary_type>));
        program_where_clause.predicates.extend(program_transpose_bounds);
        program_where_clause.predicates.push(syn::parse_quote!(#primary_type: #ryft::DifferentiableType));
        program_where_clause.predicates.push(syn::parse_quote! {
            #operation_self_type:
                ::std::convert::From<#ryft::ZeroOperation<#primary_type>>
                + ::std::convert::From<#ryft::AddOperation>
        });

        let (transposition_impl_generics, _, transposition_where_clause) = transposition_generics.split_for_impl();
        let (program_transposition_impl_generics, _, program_transposition_where_clause) =
            program_transposition_generics.split_for_impl();
        let program_transposition_impl = quote! {
            #[automatically_derived]
            impl #program_transposition_impl_generics
                #ryft::TransposableProgramOperation<#transposed_value_type>
                for #operation_self_type
            #program_transposition_where_clause
            {
                fn transpose_program(
                    program: &#ryft::Program<
                        #transposed_value_type,
                        Self,
                        ::std::vec::Vec<#transposed_value_type>,
                        ::std::vec::Vec<#transposed_value_type>,
                    >,
                    input_linearity: &[bool],
                ) -> ::std::result::Result<
                    #ryft::Program<
                        #transposed_value_type,
                        Self,
                        ::std::vec::Vec<#transposed_value_type>,
                        ::std::vec::Vec<#transposed_value_type>,
                    >,
                    #ryft::DifferentiationError,
                > {
                    let with_respect_to = input_linearity
                        .iter()
                        .enumerate()
                        .filter_map(|(index, &linear)| if linear { ::std::option::Option::Some(index) } else { ::std::option::Option::None })
                        .collect::<::std::vec::Vec<usize>>();
                    program.transpose_with_respect_to(with_respect_to.as_slice())
                }
            }
        };
        let transpose_arms = variants.iter().map(|variant| {
            let variant_ident = &variant.ident;
            let operation_type = &variant.operation_type;
            let receiver = variant.receiver();
            quote! {
                Self::#variant_ident(operation) => {
                    <#operation_type as #ryft::TransposableOperation<
                        #transposed_value_type,
                        #operation_self_type,
                    >>::transpose(#receiver, context, inputs, outputs)
                },
            }
        });

        const_block(quote! {
            #[automatically_derived]
            impl #transposition_impl_generics
                #ryft::TransposableOperation<#transposed_value_type, #operation_self_type>
                for #operation_self_type
            #transposition_where_clause
            {
                fn transpose(
                    &self,
                    context: &mut #ryft::TracingContext<
                        #transposed_value_type,
                        #operation_self_type,
                    >,
                    inputs: &[#ryft::partial::PartialValue<
                        #ryft::Tracer<#ryft::TracingContext<
                            #transposed_value_type,
                            #operation_self_type,
                        >>,
                    >],
                    outputs: &[#ryft::MaybeZero<
                        #ryft::Tracer<#ryft::TracingContext<
                            #transposed_value_type,
                            #operation_self_type,
                        >>,
                    >],
                ) -> ::std::result::Result<
                    ::std::vec::Vec<#ryft::MaybeZero<
                        #ryft::Tracer<#ryft::TracingContext<
                            #transposed_value_type,
                            #operation_self_type,
                        >>,
                    >>,
                    #ryft::DifferentiationError,
                > {
                    match self {
                        #(#transpose_arms)*
                    }
                }
            }

            #program_transposition_impl
        })
    }

    /// Generates the `From<Payload>` and borrowed `TryFrom<&Enum>` conversion implementations for one
    /// conversion-enabled variant. Boxed payloads box the payload in `From` and deref through the box in `TryFrom`.
    ///
    /// # Parameters
    ///
    ///   * `generics` - Enum generics (without defaults) used by the generated implementations.
    ///   * `enum_type` - Enum self type that the conversions convert to and from.
    ///   * `variant` - [`OperationVariant`] to generate the conversions for.
    fn generate_conversion_impl(
        &self,
        generics: &syn::Generics,
        enum_type: &syn::Type,
        variant: &OperationVariant,
    ) -> TokenStream {
        let (impl_generics, _, where_clause) = generics.split_for_impl();
        let variant_ident = &variant.ident;
        let operation_type = &variant.operation_type;
        let enum_ident = enum_type_ident(enum_type).expect("expected enum self type to start with an identifier");
        let mut try_from_generics = generics.clone();
        try_from_generics
            .params
            .insert(0, syn::GenericParam::Lifetime(syn::LifetimeParam::new(syn::parse_quote!('__operation))));
        let (try_from_impl_generics, _, try_from_where_clause) = try_from_generics.split_for_impl();

        let from_body = if variant.is_boxed {
            quote!(Self::#variant_ident(::std::boxed::Box::new(operation)))
        } else {
            quote!(Self::#variant_ident(operation))
        };
        let try_from_body = if variant.is_boxed { quote!(Ok(&**operation)) } else { quote!(Ok(operation)) };

        quote! {
            #[automatically_derived]
            impl #impl_generics ::std::convert::From<#operation_type> for #enum_type
            #where_clause
            {
                fn from(operation: #operation_type) -> Self {
                    #from_body
                }
            }

            #[automatically_derived]
            impl #try_from_impl_generics ::std::convert::TryFrom<&'__operation #enum_type>
                for &'__operation #operation_type
            #try_from_where_clause
            {
                type Error = ();

                fn try_from(value: &'__operation #enum_type) -> ::std::result::Result<Self, ()> {
                    match value {
                        #enum_ident::#variant_ident(operation) => #try_from_body,
                        _ => Err(()),
                    }
                }
            }
        }
    }
    /// Builds the generics shared by the generated [`InterpretableOperation`] dispatcher and the generated
    /// [`InterpretableProgramOperation`] witness: the enum generics with program-shaped value substitutions applied,
    /// one generated `__InterpretationValue` parameter when the enum declares a single value parameter (which is then
    /// treated as the nested program's captured constant type), one generated `__InterpretationContext` parameter,
    /// the constant-lifting `Constant` context bound, the author-declared `#[ryft(bounds(interpretation(...)))]`
    /// value bounds, and one `InterpretableOperation` predicate per non-recursive payload (recursive payloads are
    /// skipped because such a predicate would re-enter the enum's own obligation and overflow the trait solver;
    /// their arms are discharged as body obligations against the generated implementations themselves).
    ///
    /// # Parameters
    ///
    ///   * `input` - Derive macro input.
    ///   * `variants` - Extracted operation variants.
    ///   * `program_value_substitutions` - Program-shaped value substitutions (refer to
    ///     [`program_value_substitutions`]).
    ///   * `program_constant_type` - Program constant value type parameter.
    ///   * `interpretation_value_type` - Value type that the generated implementation interprets over.
    ///   * `interpretation_self_type` - Self type of the generated implementation.
    ///   * `has_separate_interpretation_value_type` - Whether interpretation is generic over a generated
    ///     `__InterpretationValue` parameter instead of reusing the program constant type.
    #[allow(clippy::too_many_arguments)]
    fn interpretation_generics(
        &self,
        input: &syn::DeriveInput,
        variants: &[OperationVariant],
        program_value_substitutions: &[(syn::Ident, syn::Type)],
        program_constant_type: &syn::Ident,
        interpretation_value_type: &syn::Type,
        interpretation_self_type: &syn::Type,
        has_separate_interpretation_value_type: bool,
    ) -> syn::Generics {
        let ryft = &self.ryft_crate;
        let primary_type = &self.operation_type;
        let mut generics =
            substitute_generics(&self.operation_generics(&input.generics, variants), program_value_substitutions);
        if has_separate_interpretation_value_type {
            generics.params.push(syn::parse_quote!(__InterpretationValue));
        }
        generics.params.push(syn::parse_quote!(__InterpretationContext));
        let where_clause = generics.make_where_clause();
        if has_separate_interpretation_value_type {
            where_clause
                .predicates
                .push(syn::parse_quote!(#interpretation_value_type: #ryft::Value<Type = #primary_type>));
            where_clause.predicates.extend(generic_parameter_bounds_as_predicates(
                &input.generics,
                program_constant_type,
                interpretation_value_type,
            ));
            where_clause.predicates.push(syn::parse_quote! {
                __InterpretationContext: #ryft::Constant<
                    #interpretation_value_type,
                    #program_constant_type,
                    #ryft::payloads::Captured,
                >
            });
        }
        where_clause
            .predicates
            .push(syn::parse_quote!(#interpretation_self_type: #ryft::Operation<#primary_type>));
        if !self.interpretation_value_bounds.is_empty() {
            add_interpretation_value_bounds(
                where_clause,
                ryft,
                interpretation_value_type,
                self.interpretation_value_bounds.as_slice(),
            );
        }
        where_clause.predicates.extend(variants.iter().filter(|variant| !variant.is_recursive_payload).map(
            |variant| {
                let operation_type = substitute_type_idents(&variant.operation_type, program_value_substitutions);
                let predicate: syn::WherePredicate = syn::parse_quote! {
                    #operation_type: #ryft::InterpretableOperation<
                        #interpretation_value_type,
                        __InterpretationContext,
                    >
                };
                predicate
            },
        ));
        generics
    }

    /// Builds the generics used by the generated [`Operation`] and [`Display`] implementations: the enum generics
    /// without defaults, plus one `Payload: Operation<T>` predicate per bare generic payload (concrete payloads
    /// already implement [`Operation`] on their own).
    fn operation_generics(&self, generics: &syn::Generics, variants: &[OperationVariant]) -> syn::Generics {
        let mut generics = generics.without_defaults();

        let ryft = &self.ryft_crate;
        let primary_type = &self.operation_type;
        let generic_operation_bounds = variants.iter().filter(|variant| variant.skip_conversions).map(|variant| {
            let operation_type = &variant.operation_type;
            let predicate: syn::WherePredicate = syn::parse_quote!(#operation_type: #ryft::Operation<#primary_type>);
            predicate
        });
        generics.make_where_clause().predicates.extend(generic_operation_bounds);
        generics
    }
}

/// Operation enum variant extracted from the derive input, together with the payload metadata that drives the
/// generated dispatch arms, predicates, and conversions.
struct OperationVariant {
    /// Identifier of the enum variant.
    ident: syn::Ident,

    /// Operation type exposed by generated conversions.
    operation_type: syn::Type,

    /// Whether the stored payload is `Box<operation_type>`.
    is_boxed: bool,

    /// Whether conversion impls should be skipped for this variant.
    skip_conversions: bool,

    /// Whether this payload recursively contains the enum type being derived.
    is_recursive_payload: bool,

    /// Whether the variant carries `#[ryft(batching(active))]`, marking a non-recursive payload whose staged
    /// batching rule dispatches at the active batching context (for its axis metadata) instead of the parent
    /// staging context.
    batching_active: bool,
}

impl OperationVariant {
    /// Receiver expression to use when delegating to the wrapped operation.
    fn receiver(&self) -> TokenStream {
        if self.is_boxed { quote!(&**operation) } else { quote!(operation) }
    }
}

/// Returns the inner type of a `Box<T>` type expression. This recognizes any type-path whose final path segment is
/// `Box` with exactly one type argument and returns that type argument, returning [`None`] for any other type.
fn boxed_inner_type(ty: &syn::Type) -> Option<syn::Type> {
    let syn::Type::Path(type_path) = ty else {
        return None;
    };
    if type_path.qself.is_some() {
        return None;
    }
    let segment = type_path.path.segments.last()?;
    if segment.ident != "Box" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return None;
    };
    if arguments.args.len() != 1 {
        return None;
    }
    match arguments.args.first() {
        Some(syn::GenericArgument::Type(inner_type)) => Some(inner_type.clone()),
        _ => None,
    }
}

/// Returns the matching generic parameter if `ty` is exactly a bare enum type parameter (e.g., the `Extension`
/// payload of an `Extension(Extension)` variant), and [`None`] otherwise. Bare generic payloads are the ones whose
/// generated conversions must be skipped, because a `From<Extension>` implementation would overlap with the concrete
/// payload conversions whenever `Extension` is instantiated as one of the concrete payload types.
fn bare_generic_parameter(ty: &syn::Type, generics: &syn::Generics) -> Option<syn::Ident> {
    let syn::Type::Path(type_path) = ty else {
        return None;
    };
    if type_path.qself.is_some() || type_path.path.segments.len() != 1 {
        return None;
    }
    let segment = type_path.path.segments.first()?;
    if !matches!(segment.arguments, syn::PathArguments::None) {
        return None;
    }
    generics
        .type_params()
        .find(|parameter| parameter.ident == segment.ident)
        .map(|parameter| parameter.ident.clone())
}

/// Returns whether `ty` is exactly the provided identifier.
fn type_is_ident(ty: &syn::Type, ident: &syn::Ident) -> bool {
    matches!(ty, syn::Type::Path(syn::TypePath { qself: None, path }) if path.is_ident(ident))
}

/// Returns whether `ty` mentions a path segment named `ident`.
fn type_mentions_ident(ty: &syn::Type, ident: &syn::Ident) -> bool {
    match ty {
        syn::Type::Array(ty) => type_mentions_ident(&ty.elem, ident),
        syn::Type::BareFn(ty) => {
            ty.inputs.iter().any(|input| type_mentions_ident(&input.ty, ident))
                || match &ty.output {
                    syn::ReturnType::Default => false,
                    syn::ReturnType::Type(_, ty) => type_mentions_ident(ty, ident),
                }
        }
        syn::Type::Group(ty) => type_mentions_ident(&ty.elem, ident),
        syn::Type::ImplTrait(ty) => ty.bounds.iter().any(|bound| type_param_bound_mentions_ident(bound, ident)),
        syn::Type::TraitObject(ty) => ty.bounds.iter().any(|bound| type_param_bound_mentions_ident(bound, ident)),
        syn::Type::Paren(ty) => type_mentions_ident(&ty.elem, ident),
        syn::Type::Path(ty) => {
            ty.qself.as_ref().is_some_and(|qself| type_mentions_ident(&qself.ty, ident))
                || path_mentions_ident(&ty.path, ident)
        }
        syn::Type::Ptr(ty) => type_mentions_ident(&ty.elem, ident),
        syn::Type::Reference(ty) => type_mentions_ident(&ty.elem, ident),
        syn::Type::Slice(ty) => type_mentions_ident(&ty.elem, ident),
        syn::Type::Tuple(ty) => ty.elems.iter().any(|ty| type_mentions_ident(ty, ident)),
        _ => false,
    }
}

/// Returns whether `bound` mentions a path segment named `ident`.
fn type_param_bound_mentions_ident(bound: &syn::TypeParamBound, ident: &syn::Ident) -> bool {
    match bound {
        syn::TypeParamBound::Trait(bound) => path_mentions_ident(&bound.path, ident),
        _ => false,
    }
}

/// Returns whether `path` mentions a segment named `ident`.
fn path_mentions_ident(path: &syn::Path, ident: &syn::Ident) -> bool {
    path.segments.iter().any(|segment| {
        segment.ident == *ident
            || match &segment.arguments {
                syn::PathArguments::None => false,
                syn::PathArguments::AngleBracketed(arguments) => arguments.args.iter().any(|argument| {
                    matches!(argument, syn::GenericArgument::Type(ty) if type_mentions_ident(ty, ident))
                        || matches!(
                            argument,
                            syn::GenericArgument::AssocType(argument)
                                if type_mentions_ident(&argument.ty, ident)
                        )
                }),
                syn::PathArguments::Parenthesized(arguments) => {
                    arguments.inputs.iter().any(|ty| type_mentions_ident(ty, ident))
                        || match &arguments.output {
                            syn::ReturnType::Default => false,
                            syn::ReturnType::Type(_, ty) => type_mentions_ident(ty, ident),
                        }
                }
            }
    })
}

/// Extracts the leading identifier from an enum self type.
fn enum_type_ident(ty: &syn::Type) -> Option<&syn::Ident> {
    let syn::Type::Path(type_path) = ty else {
        return None;
    };
    if type_path.qself.is_some() {
        return None;
    }
    type_path.path.segments.first().map(|segment| &segment.ident)
}

/// Substitutes bare type identifiers in `ty` according to the provided substitutions, returning the substituted
/// type. Refer to the documentation of [`TypeIdentSubstituter`] for information on which mentions are substituted.
fn substitute_type_idents(ty: &syn::Type, substitutions: &[(syn::Ident, syn::Type)]) -> syn::Type {
    let mut ty = ty.clone();
    TypeIdentSubstituter { substitutions }.visit_type_mut(&mut ty);
    ty
}

/// Substitutes bare type identifiers in `generics` according to the provided substitutions and removes the
/// substituted type parameters from the parameter list (they are replaced by concrete types and must not remain
/// generic in the generated implementations).
fn substitute_generics(generics: &syn::Generics, substitutions: &[(syn::Ident, syn::Type)]) -> syn::Generics {
    let mut generics = generics.clone();
    generics.params = generics
        .params
        .into_iter()
        .filter(|parameter| {
            !matches!(
                parameter,
                syn::GenericParam::Type(parameter)
                    if substitutions.iter().any(|(ident, _)| parameter.ident == *ident)
            )
        })
        .collect();
    TypeIdentSubstituter { substitutions }.visit_generics_mut(&mut generics);
    generics
}

/// Copies the declared bounds of the `source` generic parameter (both inline bounds and where-clause predicates
/// mentioning it) onto the generated `target` type, so that a generated value parameter inherits the same
/// capability requirements that the enum declares for its own value parameter.
///
/// # Parameters
///
///   * `generics` - Enum generics declaring the `source` parameter.
///   * `source` - Enum value parameter whose bounds are copied.
///   * `target` - Generated type that receives the copied bounds.
fn generic_parameter_bounds_as_predicates(
    generics: &syn::Generics,
    source: &syn::Ident,
    target: &syn::Type,
) -> Vec<syn::WherePredicate> {
    let mut predicates = generics
        .type_params()
        .find(|parameter| parameter.ident == *source)
        .into_iter()
        .flat_map(|parameter| parameter.bounds.iter())
        .map(|bound| syn::parse_quote!(#target: #bound))
        .collect::<Vec<syn::WherePredicate>>();
    if let Some(where_clause) = &generics.where_clause {
        predicates.extend(where_clause.predicates.iter().filter_map(|predicate| match predicate {
            syn::WherePredicate::Type(predicate) if type_mentions_ident(&predicate.bounded_ty, source) => {
                let mut predicate = syn::WherePredicate::Type(predicate.clone());
                TypeIdentSubstituter { substitutions: &[(source.clone(), target.clone())] }
                    .visit_where_predicate_mut(&mut predicate);
                Some(predicate)
            }
            _ => None,
        }));
    }
    predicates
}

/// Parses the parenthesized `Bound1 + Bound2 + ...` list of a `#[ryft(bounds(kind(...)))]` attribute, reporting an
/// error for empty lists and for unexpected trailing tokens.
fn parse_bounds(meta: &syn::meta::ParseNestedMeta, kind: &str) -> syn::Result<Vec<syn::TypeParamBound>> {
    let content;
    syn::parenthesized!(content in meta.input);
    let bounds =
        syn::punctuated::Punctuated::<syn::TypeParamBound, syn::Token![+]>::parse_separated_nonempty(&content)?;
    if !content.is_empty() {
        return Err(content.error(format!("unexpected tokens after {kind} bounds")));
    }
    Ok(bounds.into_iter().collect())
}

/// Adds the caller-provided `#[ryft(bounds(...))]` value bounds to the provided where clause as one predicate on
/// `value_type`, doing nothing when no bounds were declared.
fn add_value_bounds(where_clause: &mut syn::WhereClause, value_type: &syn::Type, value_bounds: &[syn::TypeParamBound]) {
    if value_bounds.is_empty() {
        return;
    }
    where_clause.predicates.push(syn::parse_quote! {
        #value_type:
            #(#value_bounds)+*
    });
}

/// Adds the caller-provided `#[ryft(bounds(interpretation(...)))]` value bounds to the provided where clause,
/// together with the standard companion requirement `__InterpretationContext: Zero<V>` that recursive higher-order
/// payload interpretation rules need whenever interpretation bounds are declared.
fn add_interpretation_value_bounds(
    where_clause: &mut syn::WhereClause,
    ryft: &syn::Path,
    interpretation_value_type: &syn::Type,
    interpretation_value_bounds: &[syn::TypeParamBound],
) {
    add_value_bounds(where_clause, interpretation_value_type, interpretation_value_bounds);
    where_clause.predicates.push(syn::parse_quote! {
        __InterpretationContext: #ryft::Zero<#interpretation_value_type>
    });
}

/// [`VisitMut`] visitor that replaces bare type identifier mentions with concrete replacement types: a
/// single-segment argument-free path is replaced wholesale, while a multi-segment path whose first segment matches a
/// substituted identifier has that first segment replaced by the replacement path (preserving the remaining
/// segments), so that associated-type projections through a substituted parameter keep working.
struct TypeIdentSubstituter<'a> {
    /// Type identifier substitutions.
    substitutions: &'a [(syn::Ident, syn::Type)],
}

impl VisitMut for TypeIdentSubstituter<'_> {
    fn visit_type_mut(&mut self, ty: &mut syn::Type) {
        if let syn::Type::Path(type_path) = ty
            && type_path.qself.is_none()
        {
            if type_path.path.segments.len() == 1
                && let Some(segment) = type_path.path.segments.first()
                && matches!(segment.arguments, syn::PathArguments::None)
                && let Some((_, replacement)) = self.substitutions.iter().find(|(ident, _)| segment.ident == *ident)
            {
                *ty = replacement.clone();
                return;
            }
            if let Some(first_segment) = type_path.path.segments.first()
                && matches!(first_segment.arguments, syn::PathArguments::None)
                && let Some((_, syn::Type::Path(replacement))) =
                    self.substitutions.iter().find(|(ident, _)| first_segment.ident == *ident)
                && replacement.qself.is_none()
            {
                let remaining_segments = type_path.path.segments.iter().skip(1).cloned();
                type_path.path.leading_colon = replacement.path.leading_colon;
                type_path.path.segments = replacement.path.segments.iter().cloned().chain(remaining_segments).collect();
            }
        }
        syn::visit_mut::visit_type_mut(self, ty);
    }
}

/// Builds the program-shaped value substitutions for the provided value type parameters: every value parameter
/// after the first two is substituted with the program constant type (the first value parameter). The first value
/// parameter is the program-constant/flowing value space and the second is the capture/constant space, while later
/// value parameters are payload-specific metadata that program-shaped implementations (which have one value space)
/// must pin to the program constant type. Refer to the documentation of the [`Operation`] trait for information on
/// how the value types of generated implementations are inferred.
fn program_value_substitutions(
    value_type_parameters: &[syn::Ident],
    program_constant_type: &syn::Ident,
) -> Vec<(syn::Ident, syn::Type)> {
    value_type_parameters
        .iter()
        .skip(2)
        .cloned()
        .map(|parameter| {
            let replacement: syn::Type = syn::parse_quote!(#program_constant_type);
            (parameter, replacement)
        })
        .collect()
}

/// Returns generic value parameters bounded by `Value<Type = operation_type>`.
fn value_type_parameters(generics: &syn::Generics, operation_type: &syn::Type) -> Vec<syn::Ident> {
    generics
        .type_params()
        .filter(|parameter| {
            parameter.bounds.iter().any(|bound| {
                value_bound_argument(bound).is_some_and(|argument| type_tokens_equal(&argument, operation_type))
            }) || generics.where_clause.iter().any(|where_clause| {
                where_clause.predicates.iter().any(|predicate| match predicate {
                    syn::WherePredicate::Type(predicate) if type_is_ident(&predicate.bounded_ty, &parameter.ident) => {
                        predicate.bounds.iter().any(|bound| {
                            value_bound_argument(bound)
                                .is_some_and(|argument| type_tokens_equal(&argument, operation_type))
                        })
                    }
                    _ => false,
                })
            })
        })
        .map(|parameter| parameter.ident.clone())
        .collect()
}

/// Returns whether two types have the same token representation. Given that procedural macros are executed before
/// type checking and inference is performed by the Rust compiler, token equality is the strongest type equality we
/// can check for here (e.g., type aliases cannot be resolved).
fn type_tokens_equal(left: &syn::Type, right: &syn::Type) -> bool {
    left.to_token_stream().to_string().replace(' ', "") == right.to_token_stream().to_string().replace(' ', "")
}

/// Extracts distinct `Value<Type = T>` bound arguments from `generics`, preserving their first-seen order.
fn unique_value_bound_arguments(generics: &syn::Generics) -> Vec<syn::Type> {
    value_bound_arguments(generics).into_iter().fold(Vec::new(), |mut arguments, argument| {
        if arguments.iter().all(|existing_argument| !type_tokens_equal(existing_argument, &argument)) {
            arguments.push(argument);
        }
        arguments
    })
}

/// Extracts all `Value<Type = T>` bound arguments from `generics`.
fn value_bound_arguments(generics: &syn::Generics) -> Vec<syn::Type> {
    let parameter_bounds = generics.type_params().flat_map(|parameter| parameter.bounds.iter());
    let where_bounds = generics
        .where_clause
        .iter()
        .flat_map(|where_clause| where_clause.predicates.iter())
        .filter_map(|predicate| match predicate {
            syn::WherePredicate::Type(predicate) => Some(predicate.bounds.iter()),
            _ => None,
        })
        .flatten();

    parameter_bounds.chain(where_bounds).filter_map(value_bound_argument).collect()
}

/// Returns the bound `Type` from a `Value<Type = T>` trait bound.
fn value_bound_argument(bound: &syn::TypeParamBound) -> Option<syn::Type> {
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
    match arguments.args.first() {
        Some(syn::GenericArgument::AssocType(binding)) if arguments.args.len() == 1 && binding.ident == "Type" => {
            Some(binding.ty.clone())
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use quote::{ToTokens, quote};

    use super::*;

    #[test]
    fn test_code_generator_add_error_and_compile_error() {
        let mut generator = CodeGenerator::new(DeriveKind::Operation);
        generator.add_error(quote!(variant_a), "first error");
        generator.add_error(quote!(variant_b), "second error");
        let error = generator.compile_error().expect("expected combined compile error").to_string();
        assert!(error.contains("compile_error"));
        assert!(error.contains("first error"));
        assert!(error.contains("second error"));
    }

    #[test]
    fn test_code_generator_extract_attributes() {
        // Test using valid attributes: the crate path and the owned interpretation bound kind are stored, while the
        // batching bound kind (owned by `#[derive(BatchableOperation)]`) is parsed and discarded.
        let mut generator = CodeGenerator::new(DeriveKind::Operation);
        let input = syn::parse2(quote! {
            #[ryft(crate = "wrapped::ryft")]
            #[ryft(bounds(interpretation(BooleanLike + Slice)))]
            #[ryft(bounds(batching(BooleanLike)))]
            enum Operation<V: Value<Type = DataType>> {
                Zero(ZeroOperation<DataType>),
            }
        })
        .expect("failed to parse derive input");
        generator.extract_attributes(&input);
        assert!(generator.errors.is_empty());
        assert_eq!(generator.ryft_crate.to_token_stream().to_string(), "wrapped :: ryft");
        assert_eq!(generator.operation_type.to_token_stream().to_string(), "DataType");
        assert_eq!(generator.interpretation_value_bounds.len(), 2);
        assert!(generator.batching_value_bounds.is_empty());

        // Test using invalid attributes.
        let mut generator = CodeGenerator::new(DeriveKind::Operation);
        let input = syn::parse2(quote! {
            #[ryft(crate = "ryft", unknown = "value")]
            enum Operation<V: Value<Type = DataType>> {
                Zero(ZeroOperation<DataType>),
            }
        })
        .expect("failed to parse derive input");
        generator.extract_attributes(&input);
        let errors = generator.errors.iter().map(|error| error.to_string()).collect::<Vec<_>>();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("invalid '#[ryft(...)]' attribute: 'unknown'"));

        // Test using duplicate owned bound kinds.
        let mut generator = CodeGenerator::new(DeriveKind::Operation);
        let input = syn::parse2(quote! {
            #[ryft(bounds(interpretation(BooleanLike)))]
            #[ryft(bounds(interpretation(Slice)))]
            enum Operation<V: Value<Type = DataType>> {
                Zero(ZeroOperation<DataType>),
            }
        })
        .expect("failed to parse derive input");
        generator.extract_attributes(&input);
        let errors = generator.errors.iter().map(|error| error.to_string()).collect::<Vec<_>>();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("duplicate ryft attribute 'bounds(interpretation(...))'"));
    }

    #[test]
    fn test_code_generator_extract_variants() {
        let mut generator = CodeGenerator::new(DeriveKind::Operation);
        let mut input = syn::parse2(quote! {
            enum Operation<V: Value<Type = DataType>, Extension> {
                Zero(ZeroOperation<DataType>),
                Boxed(Box<CustomJvpOperation<DataType, V>>),
                Recursive(WhileOperation<DataType, V, Self>),
                Extension(Extension),
            }
        })
        .expect("failed to parse derive input");
        // Recursive payloads are detected by their mention of the enum name, so `Self` mentions must be replaced
        // with their fully-qualified path first, exactly like the derive entry points do.
        replace_self_type(&mut input);
        let variants = generator.extract_variants(&input);
        assert!(generator.errors.is_empty());
        assert_eq!(variants.len(), 4);
        assert!(!variants[0].is_boxed && !variants[0].skip_conversions && !variants[0].is_recursive_payload);
        assert!(variants[1].is_boxed && !variants[1].skip_conversions && !variants[1].is_recursive_payload);
        assert!(!variants[2].is_boxed && !variants[2].skip_conversions && variants[2].is_recursive_payload);
        assert!(!variants[3].is_boxed && variants[3].skip_conversions && !variants[3].is_recursive_payload);

        // Non-enum inputs and variants that are not single-payload tuple variants are rejected.
        let mut generator = CodeGenerator::new(DeriveKind::Operation);
        let input = syn::parse2(quote! {
            struct Operation<V: Value<Type = DataType>> {
                value: V,
            }
        })
        .expect("failed to parse derive input");
        assert!(generator.extract_variants(&input).is_empty());
        assert_eq!(generator.errors.len(), 1);

        let mut generator = CodeGenerator::new(DeriveKind::Operation);
        let input = syn::parse2(quote! {
            enum Operation<V: Value<Type = DataType>> {
                Unit,
                Named { operation: ZeroOperation<DataType> },
            }
        })
        .expect("failed to parse derive input");
        assert!(generator.extract_variants(&input).is_empty());
        assert_eq!(generator.errors.len(), 2);
    }

    #[test]
    fn test_boxed_inner_type() {
        let inner = boxed_inner_type(&syn::parse_quote!(Box<CustomJvpOperation<T, V>>)).unwrap();
        assert_eq!(inner.to_token_stream().to_string().replace(' ', ""), "CustomJvpOperation<T,V>");
        assert!(boxed_inner_type(&syn::parse_quote!(CustomJvpOperation<T, V>)).is_none());
    }

    #[test]
    fn test_bare_generic_parameter() {
        let input: syn::DeriveInput = syn::parse_quote! {
            enum Operation<T, Extension> {
                Extension(Extension),
            }
        };
        assert_eq!(
            bare_generic_parameter(&syn::parse_quote!(Extension), &input.generics)
                .expect("expected generic parameter")
                .to_string(),
            "Extension",
        );
        assert!(bare_generic_parameter(&syn::parse_quote!(Box<Extension>), &input.generics).is_none());
        assert!(bare_generic_parameter(&syn::parse_quote!(Other), &input.generics).is_none());
    }

    #[test]
    fn test_value_bound_arguments() {
        let input: syn::DeriveInput = syn::parse_quote! {
            enum Operation<T, V: Value<Type = T>, W>
            where
                W: Value<Type = DataType>,
            {}
        };
        let arguments = value_bound_arguments(&input.generics);
        assert_eq!(arguments.len(), 2);
        assert_eq!(arguments[0].to_token_stream().to_string(), "T");
        assert_eq!(arguments[1].to_token_stream().to_string(), "DataType");
    }

    #[test]
    fn test_unique_value_bound_arguments() {
        let input: syn::DeriveInput = syn::parse_quote! {
            enum Operation<V: Value<Type = ArrayType>, C, F>
            where
                C: Value<Type = ArrayType>,
                F: Value<Type = DataType>,
            {}
        };
        let arguments = unique_value_bound_arguments(&input.generics);
        assert_eq!(arguments.len(), 2);
        assert_eq!(arguments[0].to_token_stream().to_string(), "ArrayType");
        assert_eq!(arguments[1].to_token_stream().to_string(), "DataType");
    }

    #[test]
    fn test_value_type_parameters() {
        let input: syn::DeriveInput = syn::parse_quote! {
            enum Operation<V: Value<Type = ArrayType>, C, F>
            where
                C: Value<Type = ArrayType>,
                F: Value<Type = DataType>,
            {}
        };
        let parameters = value_type_parameters(&input.generics, &syn::parse_quote!(ArrayType));
        assert_eq!(parameters.len(), 2);
        assert_eq!(parameters[0].to_string(), "V");
        assert_eq!(parameters[1].to_string(), "C");
    }
}
