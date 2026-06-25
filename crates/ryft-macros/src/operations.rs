// TODO(eaplatanios): Review this module.

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
const TRANSPOSITION_ATTRIBUTE: Symbol = Symbol::new("transposition");
const VALID_CONTAINER_ATTRIBUTES: [Symbol; 2] = [CRATE_ATTRIBUTE, BOUNDS_ATTRIBUTE];

const DEFAULT_RYFT_CRATE: Symbol = Symbol::new("ryft");
const DEFAULT_OPERATION_TYPE: Symbol = Symbol::new("__OperationType");

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

    /// Extra bounds to attach to the generated transposition value type.
    transposition_value_bounds: Vec<syn::TypeParamBound>,

    /// Whether `#[ryft(bounds(interpretation(...)))]` was already specified.
    interpretation_value_bounds_set: bool,

    /// Whether `#[ryft(bounds(transposition(...)))]` was already specified.
    transposition_value_bounds_set: bool,

    /// Whether this derive owns interpretation-bound code generation.
    generate_interpretation_bounds: bool,

    /// Whether this derive owns transposition-bound code generation.
    generate_transposition_bounds: bool,

    /// Errors accumulated while validating and generating the derive output.
    errors: Vec<syn::Error>,
}

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
}

impl CodeGenerator {
    /// Generates the implementation for `#[derive(Operation)]`.
    pub(crate) fn generate_operation_impl(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
        let mut input = syn::parse_macro_input!(input as syn::DeriveInput);
        replace_self_type(&mut input);

        let mut generator = CodeGenerator {
            ryft_crate: DEFAULT_RYFT_CRATE.into(),
            operation_type: syn::Type::Path(syn::TypePath {
                qself: None,
                path: syn::Path::from(syn::Ident::from(DEFAULT_OPERATION_TYPE)),
            }),
            interpretation_value_bounds: Vec::new(),
            transposition_value_bounds: Vec::new(),
            interpretation_value_bounds_set: false,
            transposition_value_bounds_set: false,
            generate_interpretation_bounds: true,
            generate_transposition_bounds: false,
            errors: Vec::new(),
        };
        generator.extract_attributes(&input);
        if let Some(error) = generator.compile_error() {
            return error.into();
        }

        let code = generator.generate(&input);
        if let Some(error) = generator.compile_error() {
            return error.into();
        }
        code.into()
    }

    /// Generates the implementation for `#[derive(TransposableOperation)]`.
    pub(crate) fn generate_transposable_operation_impl(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
        let mut input = syn::parse_macro_input!(input as syn::DeriveInput);
        replace_self_type(&mut input);

        let mut generator = CodeGenerator {
            ryft_crate: DEFAULT_RYFT_CRATE.into(),
            operation_type: syn::Type::Path(syn::TypePath {
                qself: None,
                path: syn::Path::from(syn::Ident::from(DEFAULT_OPERATION_TYPE)),
            }),
            interpretation_value_bounds: Vec::new(),
            transposition_value_bounds: Vec::new(),
            interpretation_value_bounds_set: false,
            transposition_value_bounds_set: false,
            generate_interpretation_bounds: false,
            generate_transposition_bounds: true,
            errors: Vec::new(),
        };
        generator.extract_attributes(&input);
        if let Some(error) = generator.compile_error() {
            return error.into();
        }

        let code = generator.generate_transposable_operation(&input);
        if let Some(error) = generator.compile_error() {
            return error.into();
        }
        code.into()
    }

    /// Adds an error to this [`CodeGenerator`].
    fn add_error<T: ToTokens, U: std::fmt::Display>(&mut self, tokens: T, message: U) {
        self.errors.push(syn::Error::new_spanned(tokens.into_token_stream(), message));
    }

    /// Returns all collected errors as one [`compile_error!`] token stream.
    fn compile_error(&self) -> Option<TokenStream> {
        if self.errors.is_empty() {
            None
        } else {
            let errors = self.errors.iter().map(syn::Error::to_compile_error);
            Some(quote!(#(#errors)*))
        }
    }

    /// Extracts supported top-level `#[ryft(...)]` attributes.
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
                "could not infer an operation type because no generic parameter is bounded by 'Value<T>'",
            ),
            _ => self.add_error(
                &input.generics,
                "could not infer a unique operation type because multiple distinct 'Value<T>' bounds are present",
            ),
        }
    }

    /// Extracts a `#[ryft(bounds(...))]` attribute.
    fn extract_bounds_attribute(&mut self, meta: &syn::meta::ParseNestedMeta) -> syn::Result<()> {
        meta.parse_nested_meta(|meta| match &meta.path {
            path if path == &INTERPRETATION_ATTRIBUTE => {
                if self.generate_interpretation_bounds {
                    if self.interpretation_value_bounds_set {
                        return Err(meta.error("duplicate ryft attribute 'bounds(interpretation(...))'"));
                    }
                    self.interpretation_value_bounds_set = true;
                    self.interpretation_value_bounds = parse_bounds(&meta, "interpretation")?;
                    return Ok(());
                }

                parse_bounds(&meta, "interpretation").map(|_| ())
            }
            path if path == &TRANSPOSITION_ATTRIBUTE => {
                if self.generate_transposition_bounds {
                    if self.transposition_value_bounds_set {
                        return Err(meta.error("duplicate ryft attribute 'bounds(transposition(...))'"));
                    }
                    self.transposition_value_bounds_set = true;
                    self.transposition_value_bounds = parse_bounds(&meta, "transposition")?;
                    return Ok(());
                }

                parse_bounds(&meta, "transposition").map(|_| ())
            }
            _ => {
                let supported_attributes = if self.generate_transposition_bounds {
                    "only 'transposition(...)' is supported here"
                } else {
                    "only 'interpretation(...)' is supported here"
                };
                Err(meta.error(format_args!(
                    "invalid '#[ryft(bounds(...))]' attribute: '{}'; {supported_attributes}",
                    meta.path.to_token_stream().to_string().replace(' ', ""),
                )))
            }
        })
    }

    /// Generates all derive output.
    fn generate(&mut self, input: &syn::DeriveInput) -> TokenStream {
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
        let program_value_substitutions = value_type_parameters
            .iter()
            .skip(2)
            .cloned()
            .map(|parameter| {
                let replacement: syn::Type = syn::parse_quote!(#program_constant_type);
                (parameter, replacement)
            })
            .collect::<Vec<_>>();
        let program_operation_self_type =
            substitute_type_idents(&operation_self_type, program_value_substitutions.as_slice());
        let interpretation_self_type = program_operation_self_type.clone();
        let mut interpretation_generics = substitute_generics(
            &self.operation_generics(&input.generics, &variants),
            program_value_substitutions.as_slice(),
        );
        if has_separate_interpretation_value_type {
            interpretation_generics.params.push(syn::parse_quote!(__InterpretationValue));
        }
        let interpretation_where_clause = interpretation_generics.make_where_clause();
        if has_separate_interpretation_value_type {
            interpretation_where_clause
                .predicates
                .push(syn::parse_quote!(#interpretation_value_type: #ryft::Value<#primary_type>));
            interpretation_where_clause.predicates.extend(generic_parameter_bounds_as_predicates(
                &input.generics,
                &program_constant_type,
                &interpretation_value_type,
            ));
            interpretation_where_clause.predicates.push(syn::parse_quote! {
                <#interpretation_value_type as #ryft::Value<#primary_type>>::InterpretationContext:
                    #ryft::Context<
                        Type = #primary_type,
                        Constant = #program_constant_type,
                        Value = #interpretation_value_type,
                    >
            });
        }
        interpretation_where_clause
            .predicates
            .push(syn::parse_quote!(#interpretation_self_type: #ryft::Operation<#primary_type>));
        interpretation_where_clause.predicates.push(syn::parse_quote! {
            #interpretation_self_type:
                #ryft::InterpretableProgramOperation<
                    #primary_type,
                    #interpretation_value_type,
                    #program_constant_type,
                >
        });
        if !self.interpretation_value_bounds.is_empty() {
            add_interpretation_value_bounds(
                interpretation_where_clause,
                ryft,
                primary_type,
                &interpretation_value_type,
                self.interpretation_value_bounds.as_slice(),
            );
        }
        interpretation_where_clause.predicates.extend(
            variants
                .iter()
                .map(|variant| {
                    let operation_type =
                        substitute_type_idents(&variant.operation_type, program_value_substitutions.as_slice());
                    (variant, operation_type)
                })
                .filter(|(variant, _)| !variant.is_recursive_payload)
                .map(|(_, operation_type)| {
                    let predicate: syn::WherePredicate = syn::parse_quote! {
                        #operation_type: #ryft::InterpretableOperation<#primary_type, #interpretation_value_type>
                    };
                    predicate
                }),
        );
        let (interpretation_impl_generics, _, interpretation_where_clause) = interpretation_generics.split_for_impl();

        let mut program_interpretation_generics = substitute_generics(
            &self.operation_generics(&input.generics, &variants),
            program_value_substitutions.as_slice(),
        );
        if has_separate_interpretation_value_type {
            program_interpretation_generics.params.push(syn::parse_quote!(__InterpretationValue));
        }
        let program_interpretation_where_clause = program_interpretation_generics.make_where_clause();
        if has_separate_interpretation_value_type {
            program_interpretation_where_clause
                .predicates
                .push(syn::parse_quote!(#interpretation_value_type: #ryft::Value<#primary_type>));
            program_interpretation_where_clause.predicates.extend(generic_parameter_bounds_as_predicates(
                &input.generics,
                &program_constant_type,
                &interpretation_value_type,
            ));
            program_interpretation_where_clause.predicates.push(syn::parse_quote! {
                <#interpretation_value_type as #ryft::Value<#primary_type>>::InterpretationContext:
                    #ryft::Context<
                        Type = #primary_type,
                        Constant = #program_constant_type,
                        Value = #interpretation_value_type,
                    >
            });
        }
        program_interpretation_where_clause
            .predicates
            .push(syn::parse_quote!(#program_operation_self_type: #ryft::Operation<#primary_type>));
        if !self.interpretation_value_bounds.is_empty() {
            add_interpretation_value_bounds(
                program_interpretation_where_clause,
                ryft,
                primary_type,
                &interpretation_value_type,
                self.interpretation_value_bounds.as_slice(),
            );
        }
        program_interpretation_where_clause.predicates.extend(
            variants
                .iter()
                .map(|variant| {
                    let operation_type =
                        substitute_type_idents(&variant.operation_type, program_value_substitutions.as_slice());
                    (variant, operation_type)
                })
                .filter(|(variant, _)| !variant.is_recursive_payload)
                .map(|(_, operation_type)| {
                    let predicate: syn::WherePredicate = syn::parse_quote! {
                        #operation_type: #ryft::InterpretableOperation<#primary_type, #interpretation_value_type>
                    };
                    predicate
                }),
        );
        let program_constant_lift = if has_separate_interpretation_value_type {
            quote! {
                <<#interpretation_value_type as #ryft::Value<
                    #primary_type,
                >>::InterpretationContext as #ryft::Context>::lift(context, constant.clone())
            }
        } else {
            quote!(Ok(constant.clone()))
        };
        let program_interpretation_arms = variants.iter().map(|variant| {
            let variant_ident = &variant.ident;
            let operation_type =
                substitute_type_idents(&variant.operation_type, program_value_substitutions.as_slice());
            let receiver = variant.receiver();
            quote! {
                Self::#variant_ident(operation) => {
                    <#operation_type as #ryft::InterpretableOperation<
                        #primary_type,
                        #interpretation_value_type,
                    >>::interpret(#receiver, context, instruction_inputs)
                },
            }
        });
        let program_interpretation_body = quote! {
            program.interpret_with(
                input,
                |_, constant| #program_constant_lift,
                |instruction, instruction_inputs| {
                    match instruction.operation() {
                        #(#program_interpretation_arms)*
                    }
                },
            )
        };
        let (program_interpretation_impl_generics, _, program_interpretation_where_clause) =
            program_interpretation_generics.split_for_impl();

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
        let interpretation_arms = variants.iter().map(|variant| {
            let variant_ident = &variant.ident;
            let payload_operation_type =
                substitute_type_idents(&variant.operation_type, program_value_substitutions.as_slice());
            let receiver = variant.receiver();
            quote! {
                Self::#variant_ident(operation) => {
                    <#payload_operation_type as #ryft::InterpretableOperation<
                        #primary_type,
                        #interpretation_value_type,
                    >>::interpret(#receiver, context, inputs)
                },
            }
        });
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
                #ryft::InterpretableOperation<#primary_type, #interpretation_value_type>
                for #interpretation_self_type
            #interpretation_where_clause
            {
                fn interpret(
                    &self,
                    context: &<#interpretation_value_type as #ryft::Value<
                        #primary_type,
                    >>::InterpretationContext,
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
                    #primary_type,
                    #interpretation_value_type,
                    #program_constant_type,
                >
                for #program_operation_self_type
            #program_interpretation_where_clause
            {
                fn interpret_program(
                    context: &<#interpretation_value_type as #ryft::Value<
                        #primary_type,
                    >>::InterpretationContext,
                    program: &#ryft::Program<
                        #primary_type,
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

    /// Generates the `TransposableOperation` derive output.
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
        let has_separate_transposition_value_type = value_type_parameters.len() == 1;
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
                        #primary_type,
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
                    #primary_type,
                    #transposed_value_type,
                    #operation_self_type,
                >
            };
            predicate
        });
        let where_clause = transposition_generics.make_where_clause();
        if has_separate_transposition_value_type {
            where_clause.predicates.push(syn::parse_quote!(#transposed_value_type: #ryft::Value<#primary_type>));
            where_clause.predicates.extend(generic_parameter_bounds_as_predicates(
                &input.generics,
                &program_constant_type,
                &transposed_value_type,
            ));
        }
        where_clause
            .predicates
            .push(syn::parse_quote!(#operation_self_type: #ryft::Operation<#primary_type>));
        add_value_bounds(where_clause, &transposed_value_type, self.transposition_value_bounds.as_slice());
        where_clause.predicates.extend(transpose_bounds.iter().cloned());

        let mut program_transposition_generics = self.operation_generics(&input.generics, &variants);
        if has_separate_transposition_value_type {
            program_transposition_generics.params.push(syn::parse_quote!(__TranspositionValue));
        }
        let program_where_clause = program_transposition_generics.make_where_clause();
        if has_separate_transposition_value_type {
            program_where_clause
                .predicates
                .push(syn::parse_quote!(#transposed_value_type: #ryft::Value<#primary_type>));
            program_where_clause.predicates.extend(generic_parameter_bounds_as_predicates(
                &input.generics,
                &program_constant_type,
                &transposed_value_type,
            ));
        }
        program_where_clause
            .predicates
            .push(syn::parse_quote!(#operation_self_type: #ryft::Operation<#primary_type>));
        add_value_bounds(program_where_clause, &transposed_value_type, self.transposition_value_bounds.as_slice());
        program_where_clause.predicates.extend(program_transpose_bounds);
        program_where_clause.predicates.push(syn::parse_quote!(#primary_type: #ryft::DifferentiableType));
        program_where_clause.predicates.push(syn::parse_quote! {
            #operation_self_type:
                #ryft::MaybeZeroOperation<#primary_type>
                + ::std::convert::From<#ryft::ZeroOperation<#primary_type>>
                + ::std::convert::From<#ryft::AddOperation>
        });

        let (transposition_impl_generics, _, transposition_where_clause) = transposition_generics.split_for_impl();
        let (program_transposition_impl_generics, _, program_transposition_where_clause) =
            program_transposition_generics.split_for_impl();
        let program_transposition_impl = quote! {
            #[automatically_derived]
            impl #program_transposition_impl_generics
                #ryft::TransposableProgramOperation<#primary_type, #transposed_value_type>
                for #operation_self_type
            #program_transposition_where_clause
            {
                fn transpose_program(
                    program: &#ryft::Program<
                        #primary_type,
                        #transposed_value_type,
                        Self,
                        ::std::vec::Vec<#transposed_value_type>,
                        ::std::vec::Vec<#transposed_value_type>,
                    >,
                ) -> ::std::result::Result<
                    #ryft::Program<
                        #primary_type,
                        #transposed_value_type,
                        Self,
                        ::std::vec::Vec<#transposed_value_type>,
                        ::std::vec::Vec<#transposed_value_type>,
                    >,
                    #ryft::ProgramError,
                > {
                    program.transpose()
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
                        #primary_type,
                        #transposed_value_type,
                        #operation_self_type,
                    >>::transpose(#receiver, context, input_types, output_cotangents)
                },
            }
        });

        const_block(quote! {
            #[automatically_derived]
            impl #transposition_impl_generics
                #ryft::TransposableOperation<#primary_type, #transposed_value_type, #operation_self_type>
                for #operation_self_type
            #transposition_where_clause
            {
                fn transpose<'__transpose>(
                    &self,
                    context: &mut #ryft::AbstractTracingContext<
                        '__transpose,
                        #primary_type,
                        #transposed_value_type,
                        #operation_self_type,
                    >,
                    input_types: &[&#primary_type],
                    output_cotangents: &[#ryft::Cotangent<
                        '__transpose,
                        #primary_type,
                        #transposed_value_type,
                        #operation_self_type,
                    >],
                ) -> ::std::result::Result<
                    ::std::vec::Vec<#ryft::Cotangent<
                        '__transpose,
                        #primary_type,
                        #transposed_value_type,
                        #operation_self_type,
                    >>,
                    #ryft::ProgramError,
                > {
                    match self {
                        #(#transpose_arms)*
                    }
                }
            }

            #program_transposition_impl
        })
    }

    /// Extracts operation variants from an enum input.
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

    /// Extracts one operation variant.
    fn extract_variant(
        &mut self,
        enum_name: &syn::Ident,
        generics: &syn::Generics,
        variant: &syn::Variant,
    ) -> Option<OperationVariant> {
        self.reject_nested_attributes(&variant.attrs);
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
        })
    }

    /// Rejects nested `#[ryft(...)]` attributes.
    fn reject_nested_attributes(&mut self, attributes: &[syn::Attribute]) {
        attributes
            .iter()
            .filter(|attr| attr.path() == &RYFT_ATTRIBUTE)
            .for_each(|attr| self.add_error(attr, NESTED_ATTRIBUTE_ERROR));
    }

    /// Builds the generics used by the generated `Operation` and `Display` impls.
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

    /// Generates `From` and borrowed `TryFrom` impls for one conversion-enabled variant.
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
}

impl OperationVariant {
    /// Receiver expression to use when delegating to the wrapped operation.
    fn receiver(&self) -> TokenStream {
        if self.is_boxed { quote!(&**operation) } else { quote!(operation) }
    }
}

/// Returns the inner type of `Box<T>`.
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

/// Returns the matching generic parameter if `ty` is exactly a bare enum type parameter.
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
        syn::TypeParamBound::Lifetime(_) => false,
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

/// Substitutes bare type identifiers in `ty`.
fn substitute_type_idents(ty: &syn::Type, substitutions: &[(syn::Ident, syn::Type)]) -> syn::Type {
    let mut ty = ty.clone();
    TypeIdentSubstituter { substitutions }.visit_type_mut(&mut ty);
    ty
}

/// Substitutes bare type identifiers in `generics` and removes substituted type parameters.
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

/// Copies bounds from a generic parameter to a generated type.
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

/// Parses a `#[ryft(bounds(kind(...)))]` bound list.
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

/// Adds caller-provided value bounds.
fn add_value_bounds(where_clause: &mut syn::WhereClause, value_type: &syn::Type, value_bounds: &[syn::TypeParamBound]) {
    if value_bounds.is_empty() {
        return;
    }
    where_clause.predicates.push(syn::parse_quote! {
        #value_type:
            #(#value_bounds)+*
    });
}

/// Adds caller-provided interpretation value bounds and their standard companion bounds.
fn add_interpretation_value_bounds(
    where_clause: &mut syn::WhereClause,
    ryft: &syn::Path,
    operation_type: &syn::Type,
    interpretation_value_type: &syn::Type,
    interpretation_value_bounds: &[syn::TypeParamBound],
) {
    add_value_bounds(where_clause, interpretation_value_type, interpretation_value_bounds);
    where_clause.predicates.push(syn::parse_quote! {
        <#interpretation_value_type as #ryft::Value<#operation_type>>::InterpretationContext:
            #ryft::Zero<#operation_type, #interpretation_value_type>
    });
}

/// Visitor replacing bare type identifiers with concrete types.
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

/// Returns generic value parameters bounded by `Value<operation_type>`.
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

/// Returns whether two types have the same token representation.
fn type_tokens_equal(left: &syn::Type, right: &syn::Type) -> bool {
    left.to_token_stream().to_string().replace(' ', "") == right.to_token_stream().to_string().replace(' ', "")
}

/// Extracts distinct `Value<T>` bound arguments from `generics`, preserving their first-seen order.
fn unique_value_bound_arguments(generics: &syn::Generics) -> Vec<syn::Type> {
    value_bound_arguments(generics).into_iter().fold(Vec::new(), |mut arguments, argument| {
        if arguments.iter().all(|existing_argument| !type_tokens_equal(existing_argument, &argument)) {
            arguments.push(argument);
        }
        arguments
    })
}

/// Extracts all `Value<T>` bound arguments from `generics`.
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

/// Returns the single type argument from a `Value<T>` trait bound.
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
        Some(syn::GenericArgument::Type(argument)) if arguments.args.len() == 1 => Some(argument.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use quote::ToTokens;

    use super::{
        bare_generic_parameter, boxed_inner_type, unique_value_bound_arguments, value_bound_arguments,
        value_type_parameters,
    };

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
            enum Operation<T, V: Value<T>, W>
            where
                W: Value<DataType>,
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
            enum Operation<V: Value<ArrayType>, C, F>
            where
                C: Value<ArrayType>,
                F: Value<DataType>,
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
            enum Operation<V: Value<ArrayType>, C, F>
            where
                C: Value<ArrayType>,
                F: Value<DataType>,
            {}
        };
        let parameters = value_type_parameters(&input.generics, &syn::parse_quote!(ArrayType));
        assert_eq!(parameters.len(), 2);
        assert_eq!(parameters[0].to_string(), "V");
        assert_eq!(parameters[1].to_string(), "C");
    }
}
