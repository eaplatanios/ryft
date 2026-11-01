use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote, quote_spanned};
use syn::parse::Parser;
use syn::spanned::Spanned;
use syn::visit::Visit;

use crate::helpers::attributes::Attribute;
use crate::helpers::symbols::Symbol;

// TODO(eaplatanios): Review from here onwards.

const CRATE_ARGUMENT: Symbol = Symbol::new("crate");
const REQUIRES_ARGUMENT: Symbol = Symbol::new("requires");

const INPUT_ANNOTATION: Symbol = Symbol::new("input");
const OUTPUT_ANNOTATION: Symbol = Symbol::new("output");

const DATA_TYPE_METADATA: Symbol = Symbol::new("data_type");
const RANK_METADATA: Symbol = Symbol::new("rank");
const SHAPE_METADATA: Symbol = Symbol::new("shape");
const TILE_METADATA: Symbol = Symbol::new("tile");
const BOUNDARY_METADATA: Symbol = Symbol::new("boundary");

/// Prefix of the identifiers that generated code declares, which kernel functions cannot use for their own bindings.
const RESERVED_PREFIX: &str = "__kernel_";

/// Code generator for the `#[kernel]` attribute, which replaces the annotated function with a functional callable and a
/// module of the same name whose `definition` function traces the kernel body into a `KernelDefinition`. The body is
/// lowered statement by statement into calls that stage canonical operations, so that no host Rust code runs while
/// the kernel is traced. Refer to the documentation of the `kernel` attribute in the crate root for information on
/// how to use it.
pub(crate) struct KernelCodeGenerator {
    /// [`syn::Path`] of the kernel namespace (i.e., `ryft::kernels` unless overridden by the `crate = "..."` argument),
    /// through which generated code names the kernel API.
    kernels: syn::Path,

    /// Path of `ryft-core`, which generated code reaches through the hidden `__private::core` re-export of the kernel
    /// namespace, because a re-exported procedural attribute cannot see the path through which it was invoked.
    core: TokenStream,

    /// Constraint over the shapes of the inputs, provided through the `requires = ...` argument.
    requirement: Option<syn::Expr>,

    /// [`Input`]s of the kernel, in declaration order.
    inputs: Vec<Input>,

    /// [`Output`] of the kernel.
    output: Output,
}

impl KernelCodeGenerator {
    /// Generates the `#[kernel]` expansion for the provided attribute arguments and annotated function, which consists
    /// of the callable and the definition module described in [`KernelCodeGenerator`]. If any errors are encountered,
    /// the expansion consists of a [`compile_error!`] invocation for the first one, because the annotated function
    /// itself is not valid Rust.
    ///
    /// As an example, the following kernel:
    ///
    /// ```ignore
    /// #[kernel]
    /// pub fn copy(
    ///     #[input(data_type = F32, rank = 1)] input: &Array,
    ///     #[output(data_type = F32, shape = [input.shape()[0]])] output: &mut Array,
    /// ) {
    ///     output.store(input.load());
    /// }
    /// ```
    ///
    /// expands to the following items, where `core` stands for `ryft::kernels::__private::core`, the type checks of
    /// the inputs are elided, and `scope(...)` stands for the provenance scope of the source location of each lowered
    /// expression:
    ///
    /// ```ignore
    /// pub fn copy<__KernelValue: ryft::kernels::KernelCall>(
    ///     input: &__KernelValue,
    /// ) -> ::core::result::Result<__KernelValue, core::programs::ProgramError> {
    ///     let __kernel_input_type = core::programs::Typed::r#type(input);
    ///     let __kernel_definition = copy::definition(__kernel_input_type.as_ref())
    ///         .map_err(core::programs::ProgramError::custom)?;
    ///     let mut __kernel_outputs = <__KernelValue as ryft::kernels::KernelCall>::call_kernel(
    ///         &__kernel_definition,
    ///         &[input.clone()],
    ///     )?;
    ///     // ...
    ///     ::core::result::Result::Ok(__kernel_outputs.remove(0))
    /// }
    ///
    /// pub mod copy {
    ///     use super::*;
    ///
    ///     pub fn definition(
    ///         input: &core::arrays::ArrayType,
    ///     ) -> ::core::result::Result<ryft::kernels::KernelDefinition, ryft::kernels::KernelError> {
    ///         // ...
    ///         let __kernel_output_type = core::arrays::ArrayType::new_static(
    ///             core::arrays::DataType::F32,
    ///             [ryft::kernels::static_extent(input, 0usize)?],
    ///         );
    ///         let __kernel_operation = { /* Whole-array parameters on a rank-zero grid. */ };
    ///         let __kernel_input_metadata = input;
    ///         ryft::kernels::KernelDefinition::trace(__kernel_operation, |(__kernel_references, _)| {
    ///             use core::contexts::Context as _;
    ///             let __kernel_context = __kernel_references[0].context().clone();
    ///             let input = __kernel_references[0usize].clone();
    ///             let output = __kernel_references[1usize].clone();
    ///             __kernel_context.invoke_with_provenance_scope(scope(...), || {
    ///                 Ok(core::operations::ReferenceWrite::write(&(output.clone()), &(/* input.load() */))?)
    ///             })?;
    ///             ::core::result::Result::Ok(())
    ///         })
    ///     }
    /// }
    /// ```
    pub(crate) fn generate_kernel_impl(attributes: TokenStream, item: TokenStream) -> TokenStream {
        Self::generate(attributes, item).unwrap_or_else(syn::Error::into_compile_error)
    }

    /// Generates the `#[kernel]` expansion, returning the first error that is encountered. Refer to
    /// [`KernelCodeGenerator::generate_kernel_impl`] for more information.
    fn generate(attributes: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
        let function = syn::parse2::<syn::ItemFn>(item)?;
        let (kernels, requirement) = Self::extract_arguments(attributes)?;
        Self::validate_signature(&function.sig)?;
        Self::validate_body(&function.block)?;
        let (inputs, output) = Self::extract_parameters(&function.sig)?;
        let generator = Self::new(kernels, requirement, inputs, output);
        let callable = generator.generate_callable(&function);
        let definition = generator.generate_definition(&function)?;
        Ok(quote!(#callable #definition))
    }

    /// Creates a new [`KernelCodeGenerator`] for the provided kernel namespace, shape requirement, and parameters.
    fn new(kernels: syn::Path, requirement: Option<syn::Expr>, inputs: Vec<Input>, output: Output) -> Self {
        let core = quote!(#kernels::__private::core);
        Self { kernels, core, requirement, inputs, output }
    }

    /// Extracts the kernel namespace path from the `crate = "..."` argument (defaulting to `ryft::kernels`) and the
    /// shape requirement from the `requires = ...` argument of the attribute.
    fn extract_arguments(attributes: TokenStream) -> syn::Result<(syn::Path, Option<syn::Expr>)> {
        let mut kernels = Attribute::<syn::Path>::new(CRATE_ARGUMENT);
        let mut requirement = Attribute::<syn::Expr>::new(REQUIRES_ARGUMENT);
        let parser = syn::meta::parser(|meta| match &meta.path {
            path if path == &CRATE_ARGUMENT => kernels.set(&meta),
            path if path == &REQUIRES_ARGUMENT => requirement.set(&meta),
            _ => Err(meta.error("unsupported `kernel` argument (expected `crate = \"...\"` or `requires = ...`)")),
        });
        parser.parse2(attributes)?;
        Ok((kernels.get().unwrap_or_else(|| syn::parse_quote!(ryft::kernels)), requirement.get()))
    }

    /// Validates that the signature of the kernel function is safe, synchronous, and non-generic, and that the function
    /// returns nothing, because its only result is written through its output parameter.
    fn validate_signature(signature: &syn::Signature) -> syn::Result<()> {
        if signature.asyncness.is_some()
            || signature.unsafety.is_some()
            || signature.constness.is_some()
            || signature.abi.is_some()
            || signature.variadic.is_some()
            || signature.generics.where_clause.is_some()
            || !signature.generics.params.is_empty()
            || !matches!(signature.output, syn::ReturnType::Default)
        {
            return Err(syn::Error::new_spanned(
                signature,
                "kernel functions must be safe, synchronous, non-generic, and return through output parameters",
            ));
        }
        Ok(())
    }

    /// Validates that the body of the kernel function contains no attributes, because attributes such as `#[cfg]`
    /// could not keep their meaning once the statements and expressions that they annotate become staging calls.
    fn validate_body(block: &syn::Block) -> syn::Result<()> {
        /// Visitor that finds the first attribute in a kernel body.
        struct AttributeFinder<'ast> {
            /// First attribute that has been visited, if any.
            attribute: Option<&'ast syn::Attribute>,
        }

        impl<'ast> Visit<'ast> for AttributeFinder<'ast> {
            fn visit_attribute(&mut self, attribute: &'ast syn::Attribute) {
                self.attribute.get_or_insert(attribute);
            }
        }

        let mut finder = AttributeFinder { attribute: None };
        finder.visit_block(block);
        match finder.attribute {
            Some(attribute) => Err(syn::Error::new_spanned(attribute, "kernel body attributes are unsupported")),
            None => Ok(()),
        }
    }

    /// Extracts the [`Input`]s and the single [`Output`] of the kernel from the parameters of its function.
    fn extract_parameters(signature: &syn::Signature) -> syn::Result<(Vec<Input>, Output)> {
        let mut names = Vec::<syn::Ident>::new();
        let mut inputs = Vec::new();
        let mut output = None;
        for argument in &signature.inputs {
            let syn::FnArg::Typed(argument) = argument else {
                return Err(syn::Error::new_spanned(argument, "kernel functions cannot take `self`"));
            };
            let parameter = Self::extract_parameter(argument)?;
            let name = match &parameter {
                Parameter::Input(input) => &input.name,
                Parameter::Output(output) => &output.name,
            };
            if names.contains(name) {
                return Err(syn::Error::new_spanned(name, "kernel parameter names must be unique"));
            }
            names.push(name.clone());
            match parameter {
                Parameter::Input(input) => inputs.push(input),
                Parameter::Output(parameter) if output.is_none() => output = Some(*parameter),
                Parameter::Output(_) => {
                    return Err(syn::Error::new_spanned(argument, "kernels support only one output parameter"));
                }
            }
        }
        let output =
            output.ok_or_else(|| syn::Error::new_spanned(signature, "kernels require one `output` parameter"))?;
        Ok((inputs, output))
    }

    /// Extracts one kernel [`Parameter`] from a parameter of the kernel function, which must be a named `&Array` with
    /// an `#[input(...)]` annotation or a named `&mut Array` with an `#[output(...)]` annotation.
    fn extract_parameter(argument: &syn::PatType) -> syn::Result<Parameter> {
        let syn::Pat::Ident(pattern) = argument.pat.as_ref() else {
            return Err(syn::Error::new_spanned(&argument.pat, "kernel parameters require a name"));
        };
        if pattern.by_ref.is_some()
            || pattern.subpat.is_some()
            || pattern.ident.to_string().starts_with(RESERVED_PREFIX)
        {
            return Err(syn::Error::new_spanned(
                pattern,
                "kernel parameter names cannot use the reserved `__kernel_` prefix or subpatterns",
            ));
        }
        let syn::Type::Reference(reference) = argument.ty.as_ref() else {
            return Err(syn::Error::new_spanned(&argument.ty, "kernel parameters must use `&Array` or `&mut Array`"));
        };
        if reference.lifetime.is_some() {
            return Err(syn::Error::new_spanned(
                reference,
                "kernel parameter references do not accept explicit lifetimes",
            ));
        }
        let is_array = matches!(
            reference.elem.as_ref(),
            syn::Type::Path(syn::TypePath { qself: None, path })
                if path.segments.last().is_some_and(|segment| segment.ident == "Array" && segment.arguments.is_none())
        );
        if !is_array {
            return Err(syn::Error::new_spanned(
                &reference.elem,
                "kernel parameters must use the canonical non-generic `Array`",
            ));
        }
        let [annotation] = argument.attrs.as_slice() else {
            return Err(syn::Error::new_spanned(
                argument,
                "each kernel parameter requires exactly one `input` or `output` annotation",
            ));
        };
        let is_output = match annotation.path() {
            path if path == &INPUT_ANNOTATION => false,
            path if path == &OUTPUT_ANNOTATION => true,
            _ => return Err(syn::Error::new_spanned(annotation, "expected an `input` or `output` annotation")),
        };
        if is_output != reference.mutability.is_some() {
            return Err(syn::Error::new_spanned(
                reference,
                "kernel inputs must use `&Array` and kernel outputs must use `&mut Array`",
            ));
        }

        let mut data_type = Attribute::<syn::Expr>::new(DATA_TYPE_METADATA);
        let mut rank = Attribute::<syn::Expr>::new(RANK_METADATA);
        let mut shape = Attribute::<syn::Expr>::new(SHAPE_METADATA);
        let mut tile = Attribute::<syn::Expr>::new(TILE_METADATA);
        let mut boundary = Attribute::<syn::Expr>::new(BOUNDARY_METADATA);
        annotation.parse_nested_meta(|meta| match &meta.path {
            path if path == &DATA_TYPE_METADATA => data_type.set(&meta),
            path if path == &RANK_METADATA && !is_output => rank.set(&meta),
            path if path == &SHAPE_METADATA && is_output => shape.set(&meta),
            path if path == &TILE_METADATA && is_output => tile.set(&meta),
            path if path == &BOUNDARY_METADATA && is_output => boundary.set(&meta),
            path if path == &RANK_METADATA => Err(meta.error("kernel outputs take their rank from their `shape`")),
            path if path == &SHAPE_METADATA || path == &TILE_METADATA || path == &BOUNDARY_METADATA => {
                Err(meta.error("`shape`, `tile`, and `boundary` metadata are only supported for kernel outputs"))
            }
            _ => Err(meta.error("unsupported kernel parameter metadata")),
        })?;

        let name = pattern.ident.clone();
        let data_type = data_type
            .get()
            .ok_or_else(|| syn::Error::new_spanned(annotation, "kernel parameters require `data_type` metadata"))?;
        let data_type = identifier(&data_type)
            .ok_or_else(|| syn::Error::new_spanned(&data_type, "kernel data types must be `DataType` variants"))?;
        if !is_output {
            let rank = rank
                .get()
                .ok_or_else(|| syn::Error::new_spanned(annotation, "kernel inputs require `rank` metadata"))?;
            let syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(rank), .. }) = ungroup(&rank) else {
                return Err(syn::Error::new_spanned(rank, "kernel input ranks must be integer literals"));
            };
            return Ok(Parameter::Input(Input { name, data_type, rank: rank.base10_parse()? }));
        }
        let shape = shape
            .get()
            .ok_or_else(|| syn::Error::new_spanned(annotation, "kernel outputs require `shape` metadata"))?;
        let tiling = match (tile.get(), boundary.get()) {
            (Some(tile), Some(boundary)) => {
                let boundary = match identifier(&boundary) {
                    Some(policy) if policy == "masked" => Boundary::Masked,
                    Some(policy) if policy == "in_bounds" => Boundary::InBounds,
                    _ => {
                        return Err(syn::Error::new_spanned(
                            boundary,
                            "unsupported kernel boundary policy (expected `masked` or `in_bounds`)",
                        ));
                    }
                };
                Some(Tiling { tile, boundary })
            }
            (None, None) => None,
            _ => {
                return Err(syn::Error::new_spanned(
                    annotation,
                    "output tiling requires explicit `tile` and `boundary` metadata",
                ));
            }
        };
        Ok(Parameter::Output(Box::new(Output { name, data_type, shape, tiling })))
    }

    /// Generates the functional callable of the kernel, which keeps the name, visibility, and attributes of the kernel
    /// function, takes the kernel inputs as values of any `KernelCall` type, builds the kernel definition for their
    /// types, and invokes or stages it through `KernelCall::call_kernel`. The callable allows the Clippy lint that
    /// passing a single cloned input triggers.
    fn generate_callable(&self, function: &syn::ItemFn) -> TokenStream {
        let Self { kernels, core, .. } = self;
        let attributes = &function.attrs;
        let visibility = &function.vis;
        let name = &function.sig.ident;
        let inputs = self.inputs.iter().map(|input| &input.name).collect::<Vec<_>>();
        let types = inputs.iter().map(|input| format_ident!("__kernel_{}_type", input)).collect::<Vec<_>>();
        quote! {
            #(#attributes)*
            #[allow(clippy::cloned_ref_to_slice_refs)]
            #visibility fn #name<__KernelValue: #kernels::KernelCall>(#(#inputs: &__KernelValue),*)
                -> ::core::result::Result<__KernelValue, #core::programs::ProgramError>
            {
                #(let #types = #core::programs::Typed::r#type(#inputs);)*
                let __kernel_definition = #name::definition(#(#types.as_ref()),*)
                    .map_err(#core::programs::ProgramError::custom)?;
                let mut __kernel_outputs = <__KernelValue as #kernels::KernelCall>::call_kernel(
                    &__kernel_definition,
                    &[#(#inputs.clone()),*],
                )?;
                if __kernel_outputs.len() != 1 {
                    return ::core::result::Result::Err(#core::programs::ProgramError::InvalidOutputCount {
                        expected: 1,
                        actual: __kernel_outputs.len(),
                    });
                }
                ::core::result::Result::Ok(__kernel_outputs.remove(0))
            }
        }
    }

    /// Generates the definition module of the kernel, which has the name and visibility of the kernel function and the
    /// `#[cfg]` attributes that select it. Its `definition` function validates the input types against the input
    /// annotations and the shape requirement, builds the kernel call operation for the output annotation, and traces
    /// the lowered kernel body. The module imports the items of its parent, so that paths in the kernel (e.g., a
    /// relative kernel namespace path or an element type in `zeros::<T>`) resolve as they would next to the function.
    /// The `definition` function allows the Clippy lints that the uniform lowering triggers (e.g., `Ok(value?)` in the
    /// provenance scopes of fallible expressions).
    fn generate_definition(&self, function: &syn::ItemFn) -> syn::Result<TokenStream> {
        let Self { kernels, core, .. } = self;
        let visibility = &function.vis;
        let name = &function.sig.ident;
        let cfg_attributes = function.attrs.iter().filter(|attribute| attribute.path().is_ident("cfg"));
        let module_documentation = format!(
            " Definition of the `{name}` kernel, which can be inspected, verified, and compiled independently of the \
            `{name}` kernel function.",
        );
        let definition_documentation = format!(
            " Returns the definition of the `{name}` kernel for the provided input types, after validating them \
            against the input annotations and the `requires` constraint of the kernel.",
        );
        let inputs = self.inputs.iter().map(|input| &input.name).collect::<Vec<_>>();
        let input_indices = 0..inputs.len();
        let metadata = inputs.iter().map(|input| format_ident!("__kernel_{}_metadata", input)).collect::<Vec<_>>();
        let checks = self.inputs.iter().map(|Input { name, data_type, rank }| {
            let message = format!("kernel input `{name}` requires data type `{data_type}` and rank `{rank}`");
            quote! {
                if #name.data_type() != #core::arrays::DataType::#data_type || #name.rank() != #rank {
                    return ::core::result::Result::Err(#kernels::KernelError::Type(
                        #core::programs::TypeError::invalid(#message),
                    ));
                }
            }
        });
        let requirement = self
            .requirement
            .as_ref()
            .map(|requirement| self.generate_shape(requirement, false))
            .transpose()?
            .map(|requirement| {
                quote! {
                    if !(#requirement) {
                        return ::core::result::Result::Err(#kernels::KernelError::Type(
                            #core::programs::TypeError::invalid("kernel shape requirement is not satisfied"),
                        ));
                    }
                }
            });
        let output = &self.output.name;
        let output_index = inputs.len();
        let output_data_type = &self.output.data_type;
        let output_shape = self.generate_shape(&self.output.shape, false)?;
        let operation = self.generate_operation()?;
        let mut scope = Scope::new(
            inputs
                .iter()
                .copied()
                .chain([output])
                .map(|name| Binding { name: name.clone(), mutable: false })
                .collect(),
        );
        let body = self.generate_statements(&function.block.stmts, &mut scope)?;
        Ok(quote! {
            #[doc = #module_documentation]
            #(#cfg_attributes)*
            #visibility mod #name {
                #[allow(unused_imports)]
                use super::*;

                #[doc = #definition_documentation]
                #[allow(clippy::cloned_ref_to_slice_refs, clippy::needless_question_mark)]
                pub fn definition(#(#inputs: &#core::arrays::ArrayType),*)
                    -> ::core::result::Result<#kernels::KernelDefinition, #kernels::KernelError>
                {
                    #(#checks)*
                    #requirement
                    let __kernel_output_type = #core::arrays::ArrayType::new_static(
                        #core::arrays::DataType::#output_data_type,
                        #output_shape,
                    );
                    let __kernel_operation = #operation;
                    #(let #metadata = #inputs;)*
                    #kernels::KernelDefinition::trace(
                        __kernel_operation,
                        |(__kernel_references, __kernel_coordinates)| {
                            use #core::contexts::Context as _;
                            let __kernel_context = __kernel_references[0].context().clone();
                            #(let #inputs = __kernel_references[#input_indices].clone();)*
                            let #output = __kernel_references[#output_index].clone();
                            #body
                            ::core::result::Result::Ok(())
                        },
                    )
                }
            }
        })
    }

    /// Generates the kernel call operation, which is a parallel grid over the output tiles when the output is tiled and
    /// a single program over whole-array parameters otherwise. The generated expression uses the `definition` function
    /// parameters and the `__kernel_output_type` binding.
    fn generate_operation(&self) -> syn::Result<TokenStream> {
        let kernels = &self.kernels;
        let inputs = self.inputs.iter().map(|input| &input.name).collect::<Vec<_>>();
        match &self.output.tiling {
            Some(Tiling { tile, boundary }) => {
                let tile = self.generate_shape(tile, false)?;
                let boundary = match boundary {
                    Boundary::Masked => quote!(#kernels::BoundaryPolicy::Masked),
                    Boundary::InBounds => quote!(#kernels::BoundaryPolicy::InBounds),
                };
                Ok(quote!(
                    #kernels::tiled_call(&[#(#inputs.clone()),*], __kernel_output_type, (#tile).to_vec(), #boundary)?
                ))
            }
            None => Ok(quote!({
                let __kernel_parameters = ::std::vec![
                    #(#kernels::whole_array_parameter(#inputs.clone(), #kernels::KernelParameterAccess::ReadOnly)?,)*
                    #kernels::whole_array_parameter(
                        __kernel_output_type,
                        #kernels::KernelParameterAccess::WriteOnly,
                    )?,
                ];
                #kernels::KernelCallOperation::new(#kernels::Grid::new(::std::vec![])?, __kernel_parameters)?
            })),
        }
    }

    /// Generates the evaluation of a shape metadata expression, which consists of integer literals, arrays, input
    /// extents written as `input.shape()[axis]`, `extent.div_ceil(divisor)`, comparisons, and logical conjunctions and
    /// disjunctions, so that it cannot run host callbacks. Input extents are read from the input types, and every axis
    /// must be an integer literal below the declared rank of its input.
    ///
    /// # Parameters
    ///
    ///   - `expression`: Shape metadata expression to evaluate.
    ///   - `traced`: Indicates whether the expression is evaluated inside the traced kernel body, where the input types
    ///     are bound to `__kernel_<input>_metadata` because the input names are bound to references, and where errors
    ///     are converted into `ProgramError`s. Otherwise, the expression is evaluated in the `definition` function.
    fn generate_shape(&self, expression: &syn::Expr, traced: bool) -> syn::Result<TokenStream> {
        let kernels = &self.kernels;
        match expression {
            syn::Expr::Lit(literal) => Ok(quote!(#literal)),
            syn::Expr::Array(array) => {
                let elements = array
                    .elems
                    .iter()
                    .map(|element| self.generate_shape(element, traced))
                    .collect::<syn::Result<Vec<_>>>()?;
                Ok(quote!([#(#elements),*]))
            }
            syn::Expr::Paren(syn::ExprParen { expr, .. }) | syn::Expr::Group(syn::ExprGroup { expr, .. }) => {
                let expression = self.generate_shape(expr, traced)?;
                Ok(quote!((#expression)))
            }
            syn::Expr::Binary(binary) => {
                let left = self.generate_shape(&binary.left, traced)?;
                let right = self.generate_shape(&binary.right, traced)?;
                let operator = &binary.op;
                if !matches!(
                    operator,
                    syn::BinOp::Eq(_)
                        | syn::BinOp::Ne(_)
                        | syn::BinOp::Lt(_)
                        | syn::BinOp::Le(_)
                        | syn::BinOp::Gt(_)
                        | syn::BinOp::Ge(_)
                        | syn::BinOp::And(_)
                        | syn::BinOp::Or(_)
                ) {
                    return Err(syn::Error::new_spanned(operator, "unsupported shape metadata operator"));
                }
                Ok(quote!(#left #operator #right))
            }
            syn::Expr::Index(index) => self.generate_shape_extent(index, traced),
            syn::Expr::MethodCall(call)
                if call.method == "div_ceil" && call.args.len() == 1 && call.turbofish.is_none() =>
            {
                let extent = self.generate_shape(&call.receiver, traced)?;
                let divisor = self.generate_shape(&call.args[0], traced)?;
                Ok(self.generate_shape_result(quote!(#kernels::shape_div_ceil(#extent, #divisor)), traced))
            }
            _ => Err(syn::Error::new_spanned(expression, "unsupported shape metadata expression")),
        }
    }

    /// Generates the evaluation of an input extent written as `input.shape()[axis]` in shape metadata. Refer to
    /// [`KernelCodeGenerator::generate_shape`] for more information.
    fn generate_shape_extent(&self, index: &syn::ExprIndex, traced: bool) -> syn::Result<TokenStream> {
        let kernels = &self.kernels;
        let syn::Expr::MethodCall(call) = ungroup(&index.expr) else {
            return Err(syn::Error::new_spanned(index, "shape metadata requires `input.shape()[axis]`"));
        };
        if call.method != "shape" || !call.args.is_empty() || call.turbofish.is_some() {
            return Err(syn::Error::new_spanned(call, "unsupported shape metadata call"));
        }
        let Some(name) = identifier(&call.receiver) else {
            return Err(syn::Error::new_spanned(&call.receiver, "shape metadata requires an input name"));
        };
        let Some(input) = self.inputs.iter().find(|input| input.name == name) else {
            return Err(syn::Error::new_spanned(&call.receiver, "unknown shape metadata input"));
        };
        let syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(axis), .. }) = ungroup(&index.index) else {
            return Err(syn::Error::new_spanned(&index.index, "shape axis must be an integer literal"));
        };
        let axis = axis.base10_parse::<usize>()?;
        if axis >= input.rank {
            return Err(syn::Error::new_spanned(
                &index.index,
                format!("shape axis `{axis}` is out of bounds for kernel input `{name}` with rank `{}`", input.rank),
            ));
        }
        let shape = if traced { format_ident!("__kernel_{}_metadata", name) } else { name };
        Ok(self.generate_shape_result(quote!(#kernels::static_extent(#shape, #axis)), traced))
    }

    /// Generates the propagation of the error of a fallible shape metadata `call`, converting it into a `ProgramError`
    /// when the call is `traced` inside the kernel body.
    fn generate_shape_result(&self, call: TokenStream, traced: bool) -> TokenStream {
        let core = &self.core;
        if traced { quote!(#call.map_err(#core::programs::ProgramError::custom)?) } else { quote!(#call?) }
    }

    /// Generates the lowering of the provided kernel body statements, which declare their bindings in `scope`.
    fn generate_statements(&self, statements: &[syn::Stmt], scope: &mut Scope) -> syn::Result<TokenStream> {
        statements.iter().map(|statement| self.generate_statement(statement, scope)).collect()
    }

    /// Generates the lowering of one kernel body statement, which may declare bindings in `scope`.
    fn generate_statement(&self, statement: &syn::Stmt, scope: &mut Scope) -> syn::Result<TokenStream> {
        match statement {
            syn::Stmt::Local(local @ syn::Local { pat: syn::Pat::Slice(pattern), .. }) => {
                self.generate_tile_coordinates(local, pattern, scope)
            }
            syn::Stmt::Local(local) => self.generate_local(local, scope),
            syn::Stmt::Expr(syn::Expr::Assign(assignment), _) => {
                self.generate_assignment(&assignment.left, &assignment.right, scope)
            }
            syn::Stmt::Expr(syn::Expr::Binary(binary @ syn::ExprBinary { op: syn::BinOp::AddAssign(_), .. }), _) => {
                self.generate_compound_assignment(binary, quote!(+), scope)
            }
            syn::Stmt::Expr(syn::Expr::Binary(binary @ syn::ExprBinary { op: syn::BinOp::SubAssign(_), .. }), _) => {
                self.generate_compound_assignment(binary, quote!(-), scope)
            }
            syn::Stmt::Expr(syn::Expr::Binary(binary @ syn::ExprBinary { op: syn::BinOp::MulAssign(_), .. }), _) => {
                self.generate_compound_assignment(binary, quote!(*), scope)
            }
            syn::Stmt::Expr(syn::Expr::Binary(binary @ syn::ExprBinary { op: syn::BinOp::DivAssign(_), .. }), _) => {
                self.generate_compound_assignment(binary, quote!(/), scope)
            }
            syn::Stmt::Expr(syn::Expr::ForLoop(loop_expression), _) => self.generate_loop(loop_expression, scope),
            syn::Stmt::Expr(syn::Expr::If(condition), _) => self.generate_condition(condition, scope),
            syn::Stmt::Expr(expression, _) => {
                let value = self.generate_value(expression, scope)?;
                Ok(quote!(#value;))
            }
            _ => Err(syn::Error::new_spanned(
                statement,
                "unsupported kernel statement; arbitrary host items and macros cannot run while tracing",
            )),
        }
    }

    /// Generates the lowering of `let [row, column] = output.tile_index();`, which binds the coordinates of the output
    /// tile that the current program of a tiled kernel writes. Tile coordinates must be bound outside staged control
    /// flow, and their number must equal the rank of the output, which the traced body checks.
    fn generate_tile_coordinates(
        &self,
        local: &syn::Local,
        pattern: &syn::PatSlice,
        scope: &mut Scope,
    ) -> syn::Result<TokenStream> {
        let core = &self.core;
        let Some(initializer) = &local.init else {
            return Err(syn::Error::new_spanned(local, "tile coordinates require an initializer"));
        };
        if initializer.diverge.is_some() {
            return Err(syn::Error::new_spanned(local, "kernel `let else` is unsupported"));
        }
        let syn::Expr::MethodCall(call) = ungroup(&initializer.expr) else {
            return Err(syn::Error::new_spanned(local, "kernel destructuring only supports tile coordinates"));
        };
        if self.output.tiling.is_none()
            || scope.region
            || identifier(&call.receiver).is_none_or(|receiver| receiver != self.output.name)
            || call.method != "tile_index"
            || !call.args.is_empty()
            || call.turbofish.is_some()
        {
            return Err(syn::Error::new_spanned(call, "tile coordinates require a tiled output"));
        }
        let mut coordinates = Vec::new();
        for element in &pattern.elems {
            let syn::Pat::Ident(element) = element else {
                return Err(syn::Error::new_spanned(element, "tile coordinates require named bindings"));
            };
            let name = scope.validate_new_name(element, "tile coordinates", false)?.clone();
            scope.bindings.push(Binding { name: name.clone(), mutable: false });
            coordinates.push(name);
        }
        let axes = 0..coordinates.len();
        let rank = coordinates.len();
        Ok(quote! {
            if __kernel_coordinates.len() != #rank {
                return ::core::result::Result::Err(#core::programs::TypeError::invalid(
                    "tile coordinate binding count must equal grid rank",
                ).into());
            }
            #(let #coordinates = __kernel_coordinates[#axes].clone();)*
        })
    }

    /// Generates the lowering of a `let` statement, which either binds a staged value or declares a [`TileView`]. Tile
    /// view declarations generate no code, because their loads are lowered directly into `tile_load` calls.
    fn generate_local(&self, local: &syn::Local, scope: &mut Scope) -> syn::Result<TokenStream> {
        let syn::Pat::Ident(pattern) = &local.pat else {
            return Err(syn::Error::new_spanned(&local.pat, "unsupported kernel local pattern"));
        };
        let Some(initializer) = &local.init else {
            return Err(syn::Error::new_spanned(local, "kernel locals require an initializer"));
        };
        if initializer.diverge.is_some() {
            return Err(syn::Error::new_spanned(local, "kernel `let else` is unsupported"));
        }
        if let syn::Expr::MethodCall(padding) = ungroup(&initializer.expr)
            && padding.method == "pad"
        {
            let view = self.extract_tile_view(pattern, padding, scope)?;
            scope.tiles.push(view);
            return Ok(TokenStream::new());
        }
        let name = scope.validate_new_name(pattern, "locals", true)?.clone();
        let value = self.generate_value(&initializer.expr, scope)?;
        scope.bindings.push(Binding { name, mutable: pattern.mutability.is_some() });
        Ok(quote!(let #pattern = #value;))
    }

    /// Extracts a [`TileView`] from the declaration `let name = input.tiles(shape).pad(literal);`.
    fn extract_tile_view(
        &self,
        pattern: &syn::PatIdent,
        padding: &syn::ExprMethodCall,
        scope: &Scope,
    ) -> syn::Result<TileView> {
        let syn::Expr::MethodCall(tiles) = ungroup(&padding.receiver) else {
            return Err(syn::Error::new_spanned(padding, "padding requires a tile view"));
        };
        let input = identifier(&tiles.receiver).and_then(|name| self.inputs.iter().find(|input| input.name == name));
        let Some(input) = input.filter(|_| {
            tiles.method == "tiles"
                && tiles.args.len() == 1
                && padding.args.len() == 1
                && tiles.turbofish.is_none()
                && padding.turbofish.is_none()
        }) else {
            return Err(syn::Error::new_spanned(padding, "expected `input.tiles(shape).pad(scalar)` metadata"));
        };
        let shape = &tiles.args[0];
        let padding_value = &padding.args[0];
        let is_literal = match ungroup(padding_value) {
            syn::Expr::Lit(_) => true,
            syn::Expr::Unary(syn::ExprUnary { op: syn::UnOp::Neg(_), expr, .. }) => {
                matches!(ungroup(expr), syn::Expr::Lit(_))
            }
            _ => false,
        };
        if !is_literal {
            return Err(syn::Error::new_spanned(padding_value, "tile padding requires a scalar literal"));
        }
        let element = input.element_type()?;
        let shape = self.generate_shape(shape, true)?;
        let name = scope.validate_new_name(pattern, "tile views", false)?.clone();
        Ok(TileView {
            name,
            source: input.name.clone(),
            rank: input.rank,
            element,
            shape,
            padding: padding_value.clone(),
        })
    }

    /// Generates the lowering of an assignment of the provided `value` to the binding named by `target`.
    fn generate_assignment(&self, target: &syn::Expr, value: &syn::Expr, scope: &Scope) -> syn::Result<TokenStream> {
        let Some(name) = identifier(target) else {
            return Err(syn::Error::new_spanned(target, "kernel assignment requires a local name"));
        };
        let value = self.generate_value(value, scope)?;
        Ok(quote!(#name = #value;))
    }

    /// Generates the lowering of a compound assignment (e.g., `left += right`), which is lowered as the assignment of
    /// the binary `operator` applied to its target and its right operand (e.g., `left = left + right`).
    fn generate_compound_assignment(
        &self,
        binary: &syn::ExprBinary,
        operator: TokenStream,
        scope: &Scope,
    ) -> syn::Result<TokenStream> {
        let target = &binary.left;
        let right = &binary.right;
        let value: syn::Expr = syn::parse_quote!(#target #operator #right);
        self.generate_assignment(target, &value, scope)
    }

    /// Generates the lowering of `for index in start..end { ... }`, where `start` and `end` are shape metadata, into a
    /// staged `while` loop with a signed 64-bit scalar index. The loop carries every binding of `scope`, and the
    /// mutable bindings take their final values after the loop.
    fn generate_loop(&self, loop_expression: &syn::ExprForLoop, scope: &Scope) -> syn::Result<TokenStream> {
        let kernels = &self.kernels;
        let syn::Pat::Ident(index) = loop_expression.pat.as_ref() else {
            return Err(syn::Error::new_spanned(&loop_expression.pat, "kernel range loops require a named index"));
        };
        let syn::Expr::Range(range) = ungroup(&loop_expression.expr) else {
            return Err(syn::Error::new_spanned(
                &loop_expression.expr,
                "kernel loops require a bounded half-open range",
            ));
        };
        let (Some(start), Some(end)) = (&range.start, &range.end) else {
            return Err(syn::Error::new_spanned(range, "kernel loops require both range bounds"));
        };
        if !matches!(range.limits, syn::RangeLimits::HalfOpen(_)) || loop_expression.label.is_some() {
            return Err(syn::Error::new_spanned(loop_expression, "kernel loops require an unlabeled half-open range"));
        }
        let start = self.generate_shape(start, true)?;
        let end = self.generate_shape(end, true)?;
        let index = scope.validate_new_name(index, "loop indices", false)?;
        let names = scope.bindings.iter().map(|binding| &binding.name).collect::<Vec<_>>();
        let initialization = scope.generate_initialization();
        let restoration = scope.generate_restoration();
        let mut region = scope.region();
        region.bindings.push(Binding { name: index.clone(), mutable: false });
        let body = self.generate_statements(&loop_expression.body.stmts, &mut region)?;
        let provenance = self.generate_provenance_scope(loop_expression.span());
        Ok(quote! {
            let __kernel_carried = __kernel_context.invoke_with_provenance_scope(#provenance, || {
                #kernels::for_loop(
                    &__kernel_context,
                    #start..#end,
                    ::std::vec![#(#names.clone()),*],
                    |#index, __kernel_values| {
                        let __kernel_context = #index.context().clone();
                        #initialization
                        #body
                        ::core::result::Result::Ok(::std::vec![#(#names.clone()),*])
                    },
                )
            })?;
            #restoration
        })
    }

    /// Generates the lowering of `if predicate { ... } else { ... }`, where `predicate` is a staged boolean value, into
    /// a staged condition. Both branches carry every binding of `scope`, and the mutable bindings take the values of
    /// the selected branch after the condition. `else if` branches are lowered as nested conditions.
    fn generate_condition(&self, condition: &syn::ExprIf, scope: &Scope) -> syn::Result<TokenStream> {
        let core = &self.core;
        let predicate = self.generate_value(&condition.cond, scope)?;
        let names = scope.bindings.iter().map(|binding| &binding.name).collect::<Vec<_>>();
        let initialization = scope.generate_initialization();
        let restoration = scope.generate_restoration();
        let then_body = self.generate_statements(&condition.then_branch.stmts, &mut scope.region())?;
        let else_body = match &condition.else_branch {
            None => TokenStream::new(),
            Some((_, branch)) => match branch.as_ref() {
                syn::Expr::Block(block) => self.generate_statements(&block.block.stmts, &mut scope.region())?,
                syn::Expr::If(condition) => self.generate_condition(condition, &scope.region())?,
                _ => {
                    return Err(syn::Error::new_spanned(branch, "kernel else branches require a block or conditional"));
                }
            },
        };
        let provenance = self.generate_provenance_scope(condition.span());
        Ok(quote! {
            let __kernel_predicate = #predicate;
            let __kernel_carried = __kernel_context.invoke_with_provenance_scope(#provenance, || {
                use #core::Condition as _;
                __kernel_predicate.condition(
                    |__kernel_values| {
                        let __kernel_context = __kernel_values[0].context().clone();
                        #initialization
                        #then_body
                        ::core::result::Result::Ok(::std::vec![#(#names.clone()),*])
                    },
                    |__kernel_values| {
                        let __kernel_context = __kernel_values[0].context().clone();
                        #initialization
                        #else_body
                        ::core::result::Result::Ok(::std::vec![#(#names.clone()),*])
                    },
                    ::std::vec![#(#names.clone()),*],
                )
            })?;
            #restoration
        })
    }

    /// Generates the lowering of a kernel body expression into the staged value that it computes, under a provenance
    /// scope that records the source location of the expression.
    fn generate_value(&self, expression: &syn::Expr, scope: &Scope) -> syn::Result<TokenStream> {
        let core = &self.core;
        let value = self.generate_unscoped_value(expression, scope)?;
        let provenance = self.generate_provenance_scope(expression.span());
        Ok(quote! {
            __kernel_context.invoke_with_provenance_scope(
                #provenance,
                || ::core::result::Result::Ok::<_, #core::programs::ProgramError>(#value),
            )?
        })
    }

    /// Generates the lowering of a kernel body expression without the provenance scope that
    /// [`KernelCodeGenerator::generate_value`] adds. The supported expressions are bindings, parenthesized
    /// expressions, capability calls (i.e., `load`, `store`, `dot`, `sum`, and tile view loads), `zeros::<T>(shape)`,
    /// arithmetic, and literals.
    fn generate_unscoped_value(&self, expression: &syn::Expr, scope: &Scope) -> syn::Result<TokenStream> {
        let core = &self.core;
        match expression {
            syn::Expr::Path(path) if path.qself.is_none() && path.path.get_ident().is_some() => {
                let name = path.path.get_ident().unwrap();
                if name.to_string().starts_with(RESERVED_PREFIX) {
                    return Err(syn::Error::new_spanned(name, "kernel bodies cannot use reserved `__kernel_` names"));
                }
                if scope.tile(name).is_some() {
                    return Err(syn::Error::new_spanned(name, "tile views can only be loaded"));
                }
                Ok(quote!(#name.clone()))
            }
            syn::Expr::Paren(syn::ExprParen { expr, .. }) | syn::Expr::Group(syn::ExprGroup { expr, .. }) => {
                self.generate_unscoped_value(expr, scope)
            }
            syn::Expr::MethodCall(call) => self.generate_method_call(call, scope),
            syn::Expr::Call(call) => self.generate_zeros(call),
            syn::Expr::Binary(binary) => self.generate_arithmetic(binary, scope),
            syn::Expr::Lit(literal) => Ok(quote! {
                __kernel_context.lift(#core::arrays::ArrayIrValue::Array(#core::arrays::Array::scalar(#literal)?))?
            }),
            _ => Err(syn::Error::new_spanned(
                expression,
                "unsupported kernel expression; control flow requires staged lowering and cannot execute as host Rust",
            )),
        }
    }

    /// Generates the lowering of a capability call (i.e., `reference.load()`, `reference.store(value)`,
    /// `left.dot(right)`, `value.sum([axes])` with literal axes, or `tiles.load([indices])` for a [`TileView`]).
    /// Stores into a tiled output write the tile of the current program.
    fn generate_method_call(&self, call: &syn::ExprMethodCall, scope: &Scope) -> syn::Result<TokenStream> {
        let Self { kernels, core, .. } = self;
        if call.turbofish.is_some() {
            return Err(syn::Error::new_spanned(
                call,
                "kernel capability calls do not accept explicit generic arguments",
            ));
        }
        if let Some(view) = identifier(&call.receiver).and_then(|name| scope.tile(&name)) {
            return self.generate_tile_load(call, view, scope);
        }
        let receiver = self.generate_value(&call.receiver, scope)?;
        match (call.method.to_string().as_str(), call.args.len()) {
            ("load", 0) => Ok(quote!(#core::operations::ReferenceRead::read(&(#receiver))?)),
            ("store", 1) => {
                let value = self.generate_value(&call.args[0], scope)?;
                if self.output.tiling.is_some() {
                    Ok(quote!(#kernels::tile_store(&__kernel_context, &(#receiver), &(#value))?))
                } else {
                    Ok(quote!(#core::operations::ReferenceWrite::write(&(#receiver), &(#value))?))
                }
            }
            ("dot", 1) => {
                let right = self.generate_value(&call.args[0], scope)?;
                Ok(quote!(#kernels::dot(&(#receiver), &(#right))?))
            }
            ("sum", 1) => {
                let syn::Expr::Array(axes) = ungroup(&call.args[0]) else {
                    return Err(syn::Error::new_spanned(
                        &call.args[0],
                        "kernel reduction axes require an array of integer literals",
                    ));
                };
                let axes = axes
                    .elems
                    .iter()
                    .map(|axis| match ungroup(axis) {
                        syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(axis), .. }) => axis.base10_parse::<usize>(),
                        _ => Err(syn::Error::new_spanned(axis, "kernel reduction axes require integer literals")),
                    })
                    .collect::<syn::Result<Vec<_>>>()?;
                Ok(quote! {
                    __kernel_context.bind(
                        #core::arrays::ArrayIrOperation::from(#core::arrays::ArrayOperation::Reduce(
                            #core::operations::ReduceOperation::new(
                                ::std::vec![#(#axes),*],
                                #core::operations::ReductionKind::Sum,
                            ),
                        )),
                        ::std::vec![],
                        &[#receiver],
                    )?.remove(0)
                })
            }
            _ => Err(syn::Error::new_spanned(call, "unsupported kernel capability call")),
        }
    }

    /// Generates the lowering of `tiles.load([indices])` for the provided [`TileView`], which loads the tile at the
    /// provided tile indices (i.e., element offsets divided by the tile shape) and pads its out-of-bounds elements.
    fn generate_tile_load(
        &self,
        call: &syn::ExprMethodCall,
        view: &TileView,
        scope: &Scope,
    ) -> syn::Result<TokenStream> {
        let kernels = &self.kernels;
        if call.method != "load" || call.args.len() != 1 {
            return Err(syn::Error::new_spanned(call, "tile views only support `load([indices])`"));
        }
        let syn::Expr::Array(indices) = ungroup(&call.args[0]) else {
            return Err(syn::Error::new_spanned(&call.args[0], "tile indices require an explicit array"));
        };
        if indices.elems.len() != view.rank {
            return Err(syn::Error::new_spanned(
                indices,
                format!("tile loads from `{}` require `{}` indices", view.source, view.rank),
            ));
        }
        let indices = indices
            .elems
            .iter()
            .map(|index| self.generate_value(index, scope))
            .collect::<syn::Result<Vec<_>>>()?;
        let TileView { source, element, shape, padding, .. } = view;
        Ok(quote! {
            #kernels::tile_load::<#element, _>(
                &__kernel_context,
                &#source,
                (#shape).to_vec(),
                &[#(#indices),*],
                #padding,
            )?
        })
    }

    /// Generates the lowering of `zeros::<T>(shape)`, where `T` is the element type and `shape` is shape metadata.
    fn generate_zeros(&self, call: &syn::ExprCall) -> syn::Result<TokenStream> {
        let kernels = &self.kernels;
        let syn::Expr::Path(syn::ExprPath { qself: None, path, .. }) = ungroup(&call.func) else {
            return Err(syn::Error::new_spanned(call, "kernel helpers must be recognized named capabilities"));
        };
        let segment = path.segments.first().filter(|segment| {
            path.leading_colon.is_none() && path.segments.len() == 1 && segment.ident == "zeros" && call.args.len() == 1
        });
        let Some(segment) = segment else {
            return Err(syn::Error::new_spanned(call, "unsupported kernel helper call"));
        };
        let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
            return Err(syn::Error::new_spanned(call, "kernel zeros requires one explicit element type"));
        };
        let (1, Some(syn::GenericArgument::Type(element))) = (arguments.args.len(), arguments.args.first()) else {
            return Err(syn::Error::new_spanned(arguments, "kernel zeros requires one element type"));
        };
        let shape = self.generate_shape(&call.args[0], true)?;
        Ok(quote!(#kernels::zeros::<#element, _>(&__kernel_context, #shape)?))
    }

    /// Generates the lowering of `left op right`, where `op` is one of `+`, `-`, `*`, and `/`, into the corresponding
    /// elementwise array operation.
    fn generate_arithmetic(&self, binary: &syn::ExprBinary, scope: &Scope) -> syn::Result<TokenStream> {
        let core = &self.core;
        let left = self.generate_value(&binary.left, scope)?;
        let right = self.generate_value(&binary.right, scope)?;
        let (variant, operation) = match binary.op {
            syn::BinOp::Add(_) => (quote!(Add), quote!(AddOperation)),
            syn::BinOp::Sub(_) => (quote!(Sub), quote!(SubOperation)),
            syn::BinOp::Mul(_) => (quote!(Mul), quote!(MulOperation)),
            syn::BinOp::Div(_) => (quote!(Div), quote!(DivOperation)),
            _ => return Err(syn::Error::new_spanned(binary.op, "unsupported kernel arithmetic operator")),
        };
        Ok(quote! {
            __kernel_context.bind(
                #core::arrays::ArrayIrOperation::from(#core::arrays::ArrayOperation::#variant(
                    #core::operations::#operation::new(),
                )),
                ::std::vec![],
                &[#left, #right],
            )?.remove(0)
        })
    }

    /// Generates the construction of the provenance scope that records the source location at `span`. Only the
    /// location macros carry `span`, so that the rest of the generated code keeps call-site spans, which keeps lints
    /// that target user code from firing on generated code.
    fn generate_provenance_scope(&self, span: Span) -> TokenStream {
        let core = &self.core;
        let location =
            quote_spanned!(span=> ::core::concat!(::core::file!(), ":", ::core::line!(), ":", ::core::column!()));
        quote!(#core::programs::ProvenanceScope::new(#location))
    }
}

/// Kernel parameter declared through an `#[input(...)]` or `#[output(...)]` annotation.
enum Parameter {
    /// Parameter with an `#[input(...)]` annotation.
    Input(Input),

    /// Parameter with an `#[output(...)]` annotation.
    Output(Box<Output>),
}

/// Kernel input declared through an `#[input(data_type = ..., rank = ...)]` annotation.
struct Input {
    /// Name of the input parameter.
    name: syn::Ident,

    /// `DataType` variant that the elements of the input must have (e.g., `F32`).
    data_type: syn::Ident,

    /// Rank that the input must have.
    rank: usize,
}

impl Input {
    /// Returns the Rust scalar type of the elements of this input (e.g., `f32` for `F32`), which types the padding
    /// literals of its [`TileView`]s.
    fn element_type(&self) -> syn::Result<TokenStream> {
        match self.data_type.to_string().as_str() {
            "F32" => Ok(quote!(f32)),
            "F64" => Ok(quote!(f64)),
            "I8" => Ok(quote!(i8)),
            "I16" => Ok(quote!(i16)),
            "I32" => Ok(quote!(i32)),
            "I64" => Ok(quote!(i64)),
            "U8" => Ok(quote!(u8)),
            "U16" => Ok(quote!(u16)),
            "U32" => Ok(quote!(u32)),
            "U64" => Ok(quote!(u64)),
            "Boolean" => Ok(quote!(bool)),
            _ => Err(syn::Error::new_spanned(
                &self.data_type,
                "tile padding literal requires a supported Rust scalar type",
            )),
        }
    }
}

/// Kernel output declared through an `#[output(data_type = ..., shape = ...)]` annotation.
struct Output {
    /// Name of the output parameter.
    name: syn::Ident,

    /// `DataType` variant of the elements of the output (e.g., `F32`).
    data_type: syn::Ident,

    /// Shape metadata expression for the static shape of the output.
    shape: syn::Expr,

    /// [`Tiling`] of the output, which is [`None`] when the kernel writes the whole output from a single program.
    tiling: Option<Tiling>,
}

/// Partition of a kernel output into tiles, each of which is written by one program of a parallel grid, declared
/// through the `tile = ...` and `boundary = ...` metadata of the output annotation.
struct Tiling {
    /// Shape metadata expression for the static shape of each tile.
    tile: syn::Expr,

    /// [`Boundary`] policy for the tiles that extend past the output.
    boundary: Boundary,
}

/// Policy for the output tiles that extend past the output, which corresponds to the `BoundaryPolicy` of the kernel
/// namespace.
enum Boundary {
    /// Tiles may extend past the output (i.e., `boundary = masked`), and only their in-bounds elements are written.
    Masked,

    /// Tiles must lie within the output (i.e., `boundary = in_bounds`).
    InBounds,
}

/// Lexical scope of a kernel body, which tracks the bindings that staged loops and conditions carry and the tile views
/// that can be loaded. Scopes of nested regions are copies, so that their declarations do not leak out of them.
#[derive(Clone)]
struct Scope {
    /// [`Binding`]s that are visible in this scope, in declaration order.
    bindings: Vec<Binding>,

    /// [`TileView`]s that are visible in this scope.
    tiles: Vec<TileView>,

    /// Indicates whether this scope belongs to the region of a staged loop or condition.
    region: bool,
}

impl Scope {
    /// Creates a new top-level [`Scope`] with the provided [`Binding`]s.
    fn new(bindings: Vec<Binding>) -> Self {
        Self { bindings, tiles: Vec::new(), region: false }
    }

    /// Returns a copy of this [`Scope`] for the region of a staged loop or condition.
    fn region(&self) -> Self {
        Self { region: true, ..self.clone() }
    }

    /// Returns the [`TileView`] named `name` in this [`Scope`], if there is one.
    fn tile(&self, name: &syn::Ident) -> Option<&TileView> {
        self.tiles.iter().find(|tile| &tile.name == name)
    }

    /// Validates that `pattern` declares a new name in this [`Scope`] and returns that name. New names cannot use the
    /// reserved prefix of generated identifiers, bind by reference, have subpatterns, or shadow other [`Binding`]s or
    /// [`TileView`]s (because staged regions carry bindings by name), and they can be mutable only if `mutable` is
    /// `true`. The provided `kind` (e.g., `"locals"`) names the declared bindings in diagnostics.
    fn validate_new_name<'p>(
        &self,
        pattern: &'p syn::PatIdent,
        kind: &str,
        mutable: bool,
    ) -> syn::Result<&'p syn::Ident> {
        let name = &pattern.ident;
        if pattern.by_ref.is_some() || pattern.subpat.is_some() || name.to_string().starts_with(RESERVED_PREFIX) {
            return Err(syn::Error::new_spanned(
                pattern,
                format!("kernel {kind} cannot use the reserved `__kernel_` prefix, `ref`, or subpatterns"),
            ));
        }
        if !mutable && pattern.mutability.is_some() {
            return Err(syn::Error::new_spanned(pattern, format!("kernel {kind} cannot be mutable")));
        }
        if self.bindings.iter().any(|binding| &binding.name == name) || self.tile(name).is_some() {
            return Err(syn::Error::new_spanned(pattern, format!("kernel {kind} cannot shadow an existing binding")));
        }
        Ok(name)
    }

    /// Generates the declarations that bind every [`Binding`] of this [`Scope`] to the corresponding region argument
    /// in `__kernel_values` at the start of a staged region.
    fn generate_initialization(&self) -> TokenStream {
        let declarations = self.bindings.iter().enumerate().map(|(index, Binding { name, mutable })| {
            let mutability = mutable.then(|| quote!(mut));
            quote!(let #mutability #name = __kernel_values[#index].clone();)
        });
        quote!(#(#declarations)*)
    }

    /// Generates the assignments that update the mutable [`Binding`]s of this [`Scope`] to the corresponding results
    /// in `__kernel_carried` after a staged region.
    fn generate_restoration(&self) -> TokenStream {
        let assignments = self
            .bindings
            .iter()
            .enumerate()
            .filter(|(_, binding)| binding.mutable)
            .map(|(index, Binding { name, .. })| quote!(#name = __kernel_carried[#index].clone();));
        quote!(#(#assignments)*)
    }
}

/// Binding of a kernel body to a staged value (e.g., an input reference, a loop index, or a local).
#[derive(Clone)]
struct Binding {
    /// Name of the binding.
    name: syn::Ident,

    /// Indicates whether the binding is mutable, in which case staged regions return its final value.
    mutable: bool,
}

/// Tile view declared through `let name = input.tiles(shape).pad(literal);`, which loads tiles of `shape` from an
/// input and pads their out-of-bounds elements with `literal`.
#[derive(Clone)]
struct TileView {
    /// Name of the tile view.
    name: syn::Ident,

    /// Name of the input that the tile view loads from.
    source: syn::Ident,

    /// Rank of the input that the tile view loads from.
    rank: usize,

    /// Rust scalar type of the elements of the input that the tile view loads from.
    element: TokenStream,

    /// Generated evaluation of the tile shape metadata inside the traced kernel body.
    shape: TokenStream,

    /// Scalar literal that pads the out-of-bounds elements of the loaded tiles.
    padding: syn::Expr,
}

/// Returns `expression` without the invisible groups that declarative macros wrap around forwarded fragments.
fn ungroup(mut expression: &syn::Expr) -> &syn::Expr {
    while let syn::Expr::Group(group) = expression {
        expression = &group.expr;
    }
    expression
}

/// Returns the identifier that `expression` consists of, if it consists of a single identifier.
fn identifier(expression: &syn::Expr) -> Option<syn::Ident> {
    match ungroup(expression) {
        syn::Expr::Path(syn::ExprPath { qself: None, path, .. }) => path.get_ident().cloned(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use proc_macro2::{Delimiter, Group, TokenTree};
    use syn::parse_quote;

    use crate::helpers::tokens::TokensHelpers;

    use super::*;

    /// Returns the kernel function that most tests generate code for, which has two inputs and a tiled output.
    fn function() -> syn::ItemFn {
        parse_quote! {
            /// Gathers rows.
            #[cfg(feature = "kernels")]
            #[cfg_attr(feature = "kernels", inline)]
            pub fn gather(
                #[input(data_type = F32, rank = 2)] left: &Array,
                #[input(data_type = I32, rank = 1)] indices: &Array,
                #[output(data_type = F32, shape = [indices.shape()[0], 32], tile = [8, 32], boundary = masked)]
                output: &mut Array,
            ) {}
        }
    }

    /// Returns a [`KernelCodeGenerator`] for [`function`] that names the kernel namespace `kernels`.
    fn generator() -> KernelCodeGenerator {
        let (inputs, output) = KernelCodeGenerator::extract_parameters(&function().sig).unwrap();
        KernelCodeGenerator::new(parse_quote!(kernels), None, inputs, output)
    }

    /// Returns the top-level [`Scope`] of [`function`] with an additional mutable `accumulator` local.
    fn scope() -> Scope {
        let mut scope = Scope::new(
            ["left", "indices", "output"]
                .into_iter()
                .map(|name| Binding { name: format_ident!("{name}"), mutable: false })
                .collect(),
        );
        scope.bindings.push(Binding { name: format_ident!("accumulator"), mutable: true });
        scope
    }

    /// Returns the documentation of the definition module and of the `definition` function of the kernel `name`.
    fn definition_documentation(name: &str) -> (String, String) {
        (
            format!(
                " Definition of the `{name}` kernel, which can be inspected, verified, and compiled independently of \
                the `{name}` kernel function.",
            ),
            format!(
                " Returns the definition of the `{name}` kernel for the provided input types, after validating them \
                against the input annotations and the `requires` constraint of the kernel.",
            ),
        )
    }

    /// Returns the construction of the provenance scope that the generated code records for each source location.
    fn provenance() -> TokenStream {
        quote! {
            kernels::__private::core::programs::ProvenanceScope::new(
                ::core::concat!(::core::file!(), ":", ::core::line!(), ":", ::core::column!())
            )
        }
    }

    /// Returns the lowering of `value` under the provenance scope that [`KernelCodeGenerator::generate_value`] adds.
    fn scoped(value: TokenStream) -> TokenStream {
        let provenance = provenance();
        quote! {
            __kernel_context.invoke_with_provenance_scope(
                #provenance,
                || ::core::result::Result::Ok::<_, kernels::__private::core::programs::ProgramError>(#value),
            )?
        }
    }

    /// Returns `tokens` wrapped in the invisible group that declarative macros wrap around forwarded fragments.
    fn invisible_group(tokens: TokenStream) -> syn::Expr {
        syn::parse2(TokenTree::Group(Group::new(Delimiter::None, tokens)).into()).unwrap()
    }

    /// Returns the message of the error that extracting a kernel parameter from `argument` reports.
    fn parameter_error(argument: syn::FnArg) -> String {
        let syn::FnArg::Typed(argument) = argument else { panic!("expected a typed function argument") };
        KernelCodeGenerator::extract_parameter(&argument).err().unwrap().to_string()
    }

    #[test]
    fn test_kernel_code_generator_generate_kernel_impl() {
        let input = scoped(quote!(input.clone()));
        let output = scoped(quote!(output.clone()));
        let load = scoped(quote!(kernels::__private::core::operations::ReferenceRead::read(&(#input))?));
        let store = scoped(quote!(kernels::__private::core::operations::ReferenceWrite::write(&(#output), &(#load))?));
        let (module_documentation, definition_documentation) = definition_documentation("copy");
        assert_eq!(
            KernelCodeGenerator::generate_kernel_impl(
                quote!(crate = "kernels"),
                quote! {
                    pub fn copy(
                        #[input(data_type = F32, rank = 1)] input: &Array,
                        #[output(data_type = F32, shape = [input.shape()[0]])] output: &mut Array,
                    ) {
                        output.store(input.load());
                    }
                },
            )
            .to_string(),
            quote! {
                #[allow(clippy::cloned_ref_to_slice_refs)]
                pub fn copy<__KernelValue: kernels::KernelCall>(input: &__KernelValue)
                    -> ::core::result::Result<__KernelValue, kernels::__private::core::programs::ProgramError>
                {
                    let __kernel_input_type = kernels::__private::core::programs::Typed::r#type(input);
                    let __kernel_definition = copy::definition(__kernel_input_type.as_ref())
                        .map_err(kernels::__private::core::programs::ProgramError::custom)?;
                    let mut __kernel_outputs = <__KernelValue as kernels::KernelCall>::call_kernel(
                        &__kernel_definition,
                        &[input.clone()],
                    )?;
                    if __kernel_outputs.len() != 1 {
                        return ::core::result::Result::Err(
                            kernels::__private::core::programs::ProgramError::InvalidOutputCount {
                                expected: 1,
                                actual: __kernel_outputs.len(),
                            }
                        );
                    }
                    ::core::result::Result::Ok(__kernel_outputs.remove(0))
                }

                #[doc = #module_documentation]
                pub mod copy {
                    #[allow(unused_imports)]
                    use super::*;

                    #[doc = #definition_documentation]
                    #[allow(clippy::cloned_ref_to_slice_refs, clippy::needless_question_mark)]
                    pub fn definition(input: &kernels::__private::core::arrays::ArrayType)
                        -> ::core::result::Result<kernels::KernelDefinition, kernels::KernelError>
                    {
                        if input.data_type() != kernels::__private::core::arrays::DataType::F32
                            || input.rank() != 1usize
                        {
                            return ::core::result::Result::Err(kernels::KernelError::Type(
                                kernels::__private::core::programs::TypeError::invalid(
                                    "kernel input `input` requires data type `F32` and rank `1`"
                                ),
                            ));
                        }
                        let __kernel_output_type = kernels::__private::core::arrays::ArrayType::new_static(
                            kernels::__private::core::arrays::DataType::F32,
                            [kernels::static_extent(input, 0usize)?],
                        );
                        let __kernel_operation = {
                            let __kernel_parameters = ::std::vec![
                                kernels::whole_array_parameter(
                                    input.clone(),
                                    kernels::KernelParameterAccess::ReadOnly
                                )?,
                                kernels::whole_array_parameter(
                                    __kernel_output_type,
                                    kernels::KernelParameterAccess::WriteOnly,
                                )?,
                            ];
                            kernels::KernelCallOperation::new(kernels::Grid::new(::std::vec![])?, __kernel_parameters)?
                        };
                        let __kernel_input_metadata = input;
                        kernels::KernelDefinition::trace(
                            __kernel_operation,
                            |(__kernel_references, __kernel_coordinates)| {
                                use kernels::__private::core::contexts::Context as _;
                                let __kernel_context = __kernel_references[0].context().clone();
                                let input = __kernel_references[0usize].clone();
                                let output = __kernel_references[1usize].clone();
                                #store;
                                ::core::result::Result::Ok(())
                            },
                        )
                    }
                }
            }
            .to_string(),
        );

        // Invalid kernels expand to the first error that is encountered.
        assert_eq!(
            KernelCodeGenerator::generate_kernel_impl(
                quote!(),
                quote!(
                    fn invalid() {}
                )
            )
            .to_string(),
            quote!(::core::compile_error! { "kernels require one `output` parameter" }).to_string(),
        );
    }

    #[test]
    fn test_kernel_code_generator_extract_arguments() {
        let (kernels, requirement) = KernelCodeGenerator::extract_arguments(quote!()).unwrap();
        assert_eq!(kernels.to_compact_string(), "ryft::kernels");
        assert!(requirement.is_none());

        let (kernels, requirement) = KernelCodeGenerator::extract_arguments(quote!(
            crate = "::kernels_alias",
            requires = left.shape()[0] == right.shape()[0]
        ))
        .unwrap();
        assert_eq!(kernels.to_compact_string(), "::kernels_alias");
        assert_eq!(requirement.unwrap().to_compact_string(), "left.shape()[0]==right.shape()[0]");

        let error = |attributes| KernelCodeGenerator::extract_arguments(attributes).unwrap_err().to_string();
        assert_eq!(error(quote!(crate = "a", crate = "b")), "duplicate ryft attribute `crate`");
        assert_eq!(error(quote!(requires = true, requires = false)), "duplicate ryft attribute `requires`");
        assert_eq!(error(quote!(crate = kernels)), "expected ryft `crate` attribute to be a string: `crate = \"...\"`");
        assert_eq!(
            error(quote!(grid = [1])),
            "unsupported `kernel` argument (expected `crate = \"...\"` or `requires = ...`)",
        );
    }

    #[test]
    fn test_kernel_code_generator_validate_signature() {
        assert!(KernelCodeGenerator::validate_signature(&function().sig).is_ok());
        for signature in [
            parse_quote!(async fn invalid()),
            parse_quote!(unsafe fn invalid()),
            parse_quote!(fn invalid<T>()),
            parse_quote!(fn invalid() -> u32),
        ] {
            assert_eq!(
                KernelCodeGenerator::validate_signature(&signature).unwrap_err().to_string(),
                "kernel functions must be safe, synchronous, non-generic, and return through output parameters",
            );
        }
    }

    #[test]
    fn test_kernel_code_generator_validate_body() {
        assert!(
            KernelCodeGenerator::validate_body(&parse_quote!({
                output.store(input.load());
            }))
            .is_ok()
        );
        assert_eq!(
            KernelCodeGenerator::validate_body(&parse_quote!({
                for index in 0..1 {
                    #[cfg(any())]
                    output.store(input.load());
                }
            }))
            .unwrap_err()
            .to_string(),
            "kernel body attributes are unsupported",
        );
    }

    #[test]
    fn test_kernel_code_generator_extract_parameters() {
        let (inputs, output) = KernelCodeGenerator::extract_parameters(&function().sig).unwrap();
        assert_eq!(
            inputs
                .iter()
                .map(|input| (input.name.to_string(), input.data_type.to_string(), input.rank))
                .collect::<Vec<_>>(),
            vec![("left".to_string(), "F32".to_string(), 2), ("indices".to_string(), "I32".to_string(), 1)],
        );
        assert_eq!(output.name, "output");
        assert_eq!(output.data_type, "F32");
        assert_eq!(output.shape.to_compact_string(), "[indices.shape()[0],32]");
        let tiling = output.tiling.unwrap();
        assert_eq!(tiling.tile.to_compact_string(), "[8,32]");
        assert!(matches!(tiling.boundary, Boundary::Masked));

        let error = |signature| KernelCodeGenerator::extract_parameters(&signature).err().unwrap().to_string();
        assert_eq!(error(parse_quote!(fn invalid(&self))), "kernel functions cannot take `self`");
        assert_eq!(
            error(parse_quote!(fn invalid(
                #[input(data_type = F32, rank = 1)] value: &Array,
                #[output(data_type = F32, shape = [1])] value: &mut Array,
            ))),
            "kernel parameter names must be unique",
        );
        assert_eq!(
            error(parse_quote!(fn invalid(
                #[output(data_type = F32, shape = [1])] first: &mut Array,
                #[output(data_type = F32, shape = [1])] second: &mut Array,
            ))),
            "kernels support only one output parameter",
        );
        assert_eq!(
            error(parse_quote!(fn invalid(#[input(data_type = F32, rank = 1)] input: &Array))),
            "kernels require one `output` parameter",
        );
    }

    #[test]
    fn test_kernel_code_generator_extract_parameter() {
        let syn::FnArg::Typed(argument) =
            parse_quote!(#[output(data_type = U8, shape = [4], tile = [2], boundary = in_bounds)] output: &mut Array)
        else {
            panic!("expected a typed function argument");
        };
        let Parameter::Output(output) = KernelCodeGenerator::extract_parameter(&argument).unwrap() else {
            panic!("expected an output parameter");
        };
        assert!(matches!(output.tiling, Some(Tiling { boundary: Boundary::InBounds, .. })));

        // Patterns and types.
        assert_eq!(
            parameter_error(parse_quote!(#[input(data_type = F32, rank = 1)] (left, right): &Array)),
            "kernel parameters require a name",
        );
        assert_eq!(
            parameter_error(parse_quote!(#[input(data_type = F32, rank = 1)] __kernel_input: &Array)),
            "kernel parameter names cannot use the reserved `__kernel_` prefix or subpatterns",
        );
        assert_eq!(
            parameter_error(parse_quote!(#[input(data_type = F32, rank = 1)] input: Array)),
            "kernel parameters must use `&Array` or `&mut Array`",
        );
        assert_eq!(
            parameter_error(parse_quote!(#[input(data_type = F32, rank = 1)] input: &'a Array)),
            "kernel parameter references do not accept explicit lifetimes",
        );
        assert_eq!(
            parameter_error(parse_quote!(#[input(data_type = F32, rank = 1)] input: &Array<f32>)),
            "kernel parameters must use the canonical non-generic `Array`",
        );
        assert_eq!(
            parameter_error(parse_quote!(#[input(data_type = F32, rank = 1)] input: &mut Array)),
            "kernel inputs must use `&Array` and kernel outputs must use `&mut Array`",
        );

        // Annotations.
        assert_eq!(
            parameter_error(parse_quote!(input: &Array)),
            "each kernel parameter requires exactly one `input` or `output` annotation",
        );
        assert_eq!(
            parameter_error(parse_quote!(#[inputs(data_type = F32, rank = 1)] input: &Array)),
            "expected an `input` or `output` annotation",
        );
        assert_eq!(
            parameter_error(parse_quote!(#[input(data_type = F32, data_type = F64, rank = 1)] input: &Array)),
            "duplicate ryft attribute `data_type`",
        );
        assert_eq!(
            parameter_error(parse_quote!(#[input(data_type = F32, rank = 1, shape = [1])] input: &Array)),
            "`shape`, `tile`, and `boundary` metadata are only supported for kernel outputs",
        );
        assert_eq!(
            parameter_error(parse_quote!(#[output(data_type = F32, rank = 1, shape = [1])] output: &mut Array)),
            "kernel outputs take their rank from their `shape`",
        );
        assert_eq!(
            parameter_error(parse_quote!(#[input(data_type = F32, rank = 1, layout = row_major)] input: &Array)),
            "unsupported kernel parameter metadata",
        );

        // Metadata values.
        assert_eq!(
            parameter_error(parse_quote!(#[input(rank = 1)] input: &Array)),
            "kernel parameters require `data_type` metadata",
        );
        assert_eq!(
            parameter_error(parse_quote!(#[input(data_type = DataType::F32, rank = 1)] input: &Array)),
            "kernel data types must be `DataType` variants",
        );
        assert_eq!(
            parameter_error(parse_quote!(#[input(data_type = F32)] input: &Array)),
            "kernel inputs require `rank` metadata",
        );
        assert_eq!(
            parameter_error(parse_quote!(#[input(data_type = F32, rank = RANK)] input: &Array)),
            "kernel input ranks must be integer literals",
        );
        assert_eq!(
            parameter_error(parse_quote!(#[output(data_type = F32)] output: &mut Array)),
            "kernel outputs require `shape` metadata",
        );
        assert_eq!(
            parameter_error(parse_quote!(#[output(data_type = F32, shape = [1], tile = [1])] output: &mut Array)),
            "output tiling requires explicit `tile` and `boundary` metadata",
        );
        assert_eq!(
            parameter_error(parse_quote!(
                #[output(data_type = F32, shape = [1], tile = [1], boundary = BoundaryPolicy::Masked)]
                output: &mut Array
            )),
            "unsupported kernel boundary policy (expected `masked` or `in_bounds`)",
        );
    }

    #[test]
    fn test_kernel_code_generator_generate_callable() {
        assert_eq!(
            generator().generate_callable(&function()).to_string(),
            quote! {
                /// Gathers rows.
                #[cfg(feature = "kernels")]
                #[cfg_attr(feature = "kernels", inline)]
                #[allow(clippy::cloned_ref_to_slice_refs)]
                pub fn gather<__KernelValue: kernels::KernelCall>(left: &__KernelValue, indices: &__KernelValue)
                    -> ::core::result::Result<__KernelValue, kernels::__private::core::programs::ProgramError>
                {
                    let __kernel_left_type = kernels::__private::core::programs::Typed::r#type(left);
                    let __kernel_indices_type = kernels::__private::core::programs::Typed::r#type(indices);
                    let __kernel_definition = gather::definition(
                        __kernel_left_type.as_ref(),
                        __kernel_indices_type.as_ref()
                    )
                    .map_err(kernels::__private::core::programs::ProgramError::custom)?;
                    let mut __kernel_outputs = <__KernelValue as kernels::KernelCall>::call_kernel(
                        &__kernel_definition,
                        &[left.clone(), indices.clone()],
                    )?;
                    if __kernel_outputs.len() != 1 {
                        return ::core::result::Result::Err(
                            kernels::__private::core::programs::ProgramError::InvalidOutputCount {
                                expected: 1,
                                actual: __kernel_outputs.len(),
                            }
                        );
                    }
                    ::core::result::Result::Ok(__kernel_outputs.remove(0))
                }
            }
            .to_string(),
        );
    }

    #[test]
    fn test_kernel_code_generator_generate_definition() {
        // The module keeps only the `#[cfg]` attributes of the kernel function, and the shape requirement is checked
        // after the input types.
        let mut generator = generator();
        generator.requirement = Some(parse_quote!(left.shape()[0] == indices.shape()[0]));
        let operation = generator.generate_operation().unwrap();
        let (module_documentation, definition_documentation) = definition_documentation("gather");
        assert_eq!(
            generator.generate_definition(&function()).unwrap().to_string(),
            quote! {
                #[doc = #module_documentation]
                #[cfg(feature = "kernels")]
                pub mod gather {
                    #[allow(unused_imports)]
                    use super::*;

                    #[doc = #definition_documentation]
                    #[allow(clippy::cloned_ref_to_slice_refs, clippy::needless_question_mark)]
                    pub fn definition(
                        left: &kernels::__private::core::arrays::ArrayType,
                        indices: &kernels::__private::core::arrays::ArrayType
                    ) -> ::core::result::Result<kernels::KernelDefinition, kernels::KernelError> {
                        if left.data_type() != kernels::__private::core::arrays::DataType::F32
                            || left.rank() != 2usize
                        {
                            return ::core::result::Result::Err(kernels::KernelError::Type(
                                kernels::__private::core::programs::TypeError::invalid(
                                    "kernel input `left` requires data type `F32` and rank `2`"
                                ),
                            ));
                        }
                        if indices.data_type() != kernels::__private::core::arrays::DataType::I32
                            || indices.rank() != 1usize
                        {
                            return ::core::result::Result::Err(kernels::KernelError::Type(
                                kernels::__private::core::programs::TypeError::invalid(
                                    "kernel input `indices` requires data type `I32` and rank `1`"
                                ),
                            ));
                        }
                        if !(kernels::static_extent(left, 0usize)? == kernels::static_extent(indices, 0usize)?) {
                            return ::core::result::Result::Err(kernels::KernelError::Type(
                                kernels::__private::core::programs::TypeError::invalid(
                                    "kernel shape requirement is not satisfied"
                                ),
                            ));
                        }
                        let __kernel_output_type = kernels::__private::core::arrays::ArrayType::new_static(
                            kernels::__private::core::arrays::DataType::F32,
                            [kernels::static_extent(indices, 0usize)?, 32],
                        );
                        let __kernel_operation = #operation;
                        let __kernel_left_metadata = left;
                        let __kernel_indices_metadata = indices;
                        kernels::KernelDefinition::trace(
                            __kernel_operation,
                            |(__kernel_references, __kernel_coordinates)| {
                                use kernels::__private::core::contexts::Context as _;
                                let __kernel_context = __kernel_references[0].context().clone();
                                let left = __kernel_references[0usize].clone();
                                let indices = __kernel_references[1usize].clone();
                                let output = __kernel_references[2usize].clone();
                                ::core::result::Result::Ok(())
                            },
                        )
                    }
                }
            }
            .to_string(),
        );
    }

    #[test]
    fn test_kernel_code_generator_generate_operation() {
        assert_eq!(
            generator().generate_operation().unwrap().to_string(),
            quote! {
                kernels::tiled_call(
                    &[left.clone(), indices.clone()],
                    __kernel_output_type,
                    ([8, 32]).to_vec(),
                    kernels::BoundaryPolicy::Masked
                )?
            }
            .to_string(),
        );

        let mut generator = generator();
        generator.output.tiling = None;
        assert_eq!(
            generator.generate_operation().unwrap().to_string(),
            quote!({
                let __kernel_parameters = ::std::vec![
                    kernels::whole_array_parameter(left.clone(), kernels::KernelParameterAccess::ReadOnly)?,
                    kernels::whole_array_parameter(indices.clone(), kernels::KernelParameterAccess::ReadOnly)?,
                    kernels::whole_array_parameter(__kernel_output_type, kernels::KernelParameterAccess::WriteOnly,)?,
                ];
                kernels::KernelCallOperation::new(kernels::Grid::new(::std::vec![])?, __kernel_parameters)?
            })
            .to_string(),
        );
    }

    #[test]
    fn test_kernel_code_generator_generate_shape() {
        let generator = generator();
        let shape = |expression, traced| {
            generator
                .generate_shape(&expression, traced)
                .map(|shape| shape.to_string())
                .map_err(|error| error.to_string())
        };

        // Shape metadata is evaluated from the `definition` function parameters or, inside the traced body, from their
        // metadata bindings, and expressions forwarded through declarative macros are supported.
        assert_eq!(
            shape(parse_quote!([left.shape()[1].div_ceil(32), (8)]), false),
            Ok(quote!([kernels::shape_div_ceil(kernels::static_extent(left, 1usize)?, 32)?, (8)]).to_string()),
        );
        assert_eq!(
            shape(parse_quote!(left.shape()[0] == indices.shape()[0] && true), true),
            Ok(quote! {
                kernels::static_extent(__kernel_left_metadata, 0usize)
                    .map_err(kernels::__private::core::programs::ProgramError::custom)?
                    == kernels::static_extent(__kernel_indices_metadata, 0usize)
                        .map_err(kernels::__private::core::programs::ProgramError::custom)?
                    && true
            }
            .to_string()),
        );
        assert_eq!(shape(invisible_group(quote!(32)), false), Ok(quote!((32)).to_string()));

        // Shape metadata cannot run host code or read axes outside the declared input ranks.
        let error = |expression| shape(expression, false).unwrap_err().to_string();
        assert_eq!(error(invisible_group(quote!(host_callback()))), "unsupported shape metadata expression");
        assert_eq!(error(parse_quote!(left.shape()[0] + 1)), "unsupported shape metadata operator");
        assert_eq!(error(parse_quote!(left[0])), "shape metadata requires `input.shape()[axis]`");
        assert_eq!(error(parse_quote!(left.shape::<usize>()[0])), "unsupported shape metadata call");
        assert_eq!(error(parse_quote!(left.dims()[0])), "unsupported shape metadata call");
        assert_eq!(error(parse_quote!(inputs[0].shape()[0])), "shape metadata requires an input name");
        assert_eq!(error(parse_quote!(output.shape()[0])), "unknown shape metadata input");
        assert_eq!(error(parse_quote!(left.shape()[axis])), "shape axis must be an integer literal");
        assert_eq!(
            error(parse_quote!(indices.shape()[1])),
            "shape axis `1` is out of bounds for kernel input `indices` with rank `1`",
        );
    }

    #[test]
    fn test_kernel_code_generator_generate_statement() {
        let generator = generator();
        let mut scope = scope();
        let mut statement = |statement| {
            generator
                .generate_statement(&statement, &mut scope)
                .map(|s| s.to_string())
                .map_err(|error| error.to_string())
        };

        // Expression statements, assignments, and compound assignments.
        let output = scoped(quote!(output.clone()));
        let accumulator = scoped(quote!(accumulator.clone()));
        let store = scoped(quote!(kernels::tile_store(&__kernel_context, &(#output), &(#accumulator))?));
        assert_eq!(statement(parse_quote!(output.store(accumulator);)), Ok(quote!(#store;).to_string()));
        let left = scoped(quote!(left.clone()));
        let load = scoped(quote!(kernels::__private::core::operations::ReferenceRead::read(&(#left))?));
        assert_eq!(statement(parse_quote!(accumulator = left.load();)), Ok(quote!(accumulator = #load;).to_string()));
        let sum = scoped(quote! {
            __kernel_context.bind(
                kernels::__private::core::arrays::ArrayIrOperation::from(
                    kernels::__private::core::arrays::ArrayOperation::Div(
                        kernels::__private::core::operations::DivOperation::new(),
                    )
                ),
                ::std::vec![],
                &[#accumulator, #load],
            )?.remove(0)
        });
        assert_eq!(statement(parse_quote!(accumulator /= left.load();)), Ok(quote!(accumulator = #sum;).to_string()));

        // Unsupported statements and assignment targets.
        assert_eq!(
            statement(parse_quote!(println!("{}", 1);)).unwrap_err(),
            "unsupported kernel statement; arbitrary host items and macros cannot run while tracing",
        );
        assert_eq!(
            statement(parse_quote!(output.data = left.load();)).unwrap_err(),
            "kernel assignment requires a local name",
        );
    }

    #[test]
    fn test_kernel_code_generator_generate_tile_coordinates() {
        let generator = generator();
        let mut scope = Scope::new(vec![Binding { name: format_ident!("output"), mutable: false }]);
        assert_eq!(
            generator
                .generate_statement(&parse_quote!(let [row, column] = output.tile_index();), &mut scope)
                .unwrap()
                .to_string(),
            quote! {
                if __kernel_coordinates.len() != 2usize {
                    return ::core::result::Result::Err(kernels::__private::core::programs::TypeError::invalid(
                        "tile coordinate binding count must equal grid rank",
                    ).into());
                }
                let row = __kernel_coordinates[0usize].clone();
                let column = __kernel_coordinates[1usize].clone();
            }
            .to_string(),
        );
        assert_eq!(
            scope.bindings.iter().map(|binding| binding.name.to_string()).collect::<Vec<_>>(),
            vec!["output", "row", "column"],
        );

        // Tile coordinates must be fresh immutable names of a tiled output that are bound outside staged regions.
        let error =
            |statement, scope: &mut Scope| generator.generate_statement(&statement, scope).unwrap_err().to_string();
        assert_eq!(
            error(parse_quote!(let [row] = output.tile_index();), &mut scope),
            "kernel tile coordinates cannot shadow an existing binding",
        );
        assert_eq!(
            error(parse_quote!(let [mut depth] = output.tile_index();), &mut scope),
            "kernel tile coordinates cannot be mutable",
        );
        assert_eq!(
            error(parse_quote!(let [depth] = output.tile_index();), &mut scope.region()),
            "tile coordinates require a tiled output",
        );
        assert_eq!(
            error(parse_quote!(let [depth] = output.shape();), &mut scope),
            "tile coordinates require a tiled output",
        );
        assert_eq!(
            error(parse_quote!(let [depth] = coordinates;), &mut scope),
            "kernel destructuring only supports tile coordinates",
        );
        assert_eq!(
            error(parse_quote!(let [depth, ..] = output.tile_index();), &mut scope),
            "tile coordinates require named bindings",
        );
        let mut untiled = generator;
        untiled.output.tiling = None;
        assert_eq!(
            untiled
                .generate_statement(&parse_quote!(let [depth] = output.tile_index();), &mut scope)
                .unwrap_err()
                .to_string(),
            "tile coordinates require a tiled output",
        );
    }

    #[test]
    fn test_kernel_code_generator_generate_local() {
        let generator = generator();
        let mut scope = scope();
        let left = scoped(quote!(left.clone()));
        let load = scoped(quote!(kernels::__private::core::operations::ReferenceRead::read(&(#left))?));
        assert_eq!(
            generator
                .generate_statement(&parse_quote!(let mut row = left.load();), &mut scope)
                .unwrap()
                .to_string(),
            quote!(let mut row = #load;).to_string(),
        );
        assert!(matches!(scope.bindings.last(), Some(Binding { name, mutable: true }) if name == "row"));

        // Tile view declarations generate no code.
        assert!(
            generator
                .generate_statement(&parse_quote!(let tiles = left.tiles([8, 32]).pad(0.0);), &mut scope)
                .unwrap()
                .is_empty()
        );
        assert!(scope.tile(&format_ident!("tiles")).is_some());

        let error = |statement| generator.generate_statement(&statement, &mut scope.clone()).unwrap_err().to_string();
        assert_eq!(error(parse_quote!(let row = left.load();)), "kernel locals cannot shadow an existing binding");
        assert_eq!(error(parse_quote!(let tiles = left.load();)), "kernel locals cannot shadow an existing binding");
        assert_eq!(
            error(parse_quote!(let __kernel_row = left.load();)),
            "kernel locals cannot use the reserved `__kernel_` prefix, `ref`, or subpatterns",
        );
        assert_eq!(error(parse_quote!(let (first, second) = left.load();)), "unsupported kernel local pattern");
        assert_eq!(error(parse_quote!(let value;)), "kernel locals require an initializer");
        assert_eq!(error(parse_quote!(let value = left.load() else { return; };)), "kernel `let else` is unsupported");
    }

    #[test]
    fn test_kernel_code_generator_extract_tile_view() {
        let generator = generator();
        let scope = scope();
        let view = |statement| {
            let syn::Stmt::Local(syn::Local { pat: syn::Pat::Ident(pattern), init: Some(initializer), .. }) = statement
            else {
                panic!("expected a local with an initializer");
            };
            let syn::Expr::MethodCall(padding) = *initializer.expr else { panic!("expected a method call") };
            generator.extract_tile_view(&pattern, &padding, &scope)
        };
        let TileView { name, source, rank, element, shape, padding } =
            view(parse_quote!(let tiles = left.tiles([8, left.shape()[1]]).pad(-1.0);)).unwrap();
        assert_eq!(name, "tiles");
        assert_eq!(source, "left");
        assert_eq!(rank, 2);
        assert_eq!(element.to_string(), "f32");
        assert_eq!(
            shape.to_string(),
            quote! {
                [
                    8,
                    kernels::static_extent(__kernel_left_metadata, 1usize)
                        .map_err(kernels::__private::core::programs::ProgramError::custom)?
                ]
            }
            .to_string(),
        );
        assert_eq!(padding.to_compact_string(), "-1.0");

        // Tile views must pad tiles of a named input with a scalar literal, and they cannot be mutable.
        let error = |statement| view(statement).err().unwrap().to_string();
        assert_eq!(error(parse_quote!(let tiles = left.pad(0.0);)), "padding requires a tile view");
        assert_eq!(
            error(parse_quote!(let tiles = output.tiles([8]).pad(0.0);)),
            "expected `input.tiles(shape).pad(scalar)` metadata",
        );
        assert_eq!(
            error(parse_quote!(let tiles = left.tiles([8, 32]).pad(left);)),
            "tile padding requires a scalar literal",
        );
        assert_eq!(
            error(parse_quote!(let mut tiles = left.tiles([8, 32]).pad(0.0);)),
            "kernel tile views cannot be mutable",
        );
        assert_eq!(
            error(parse_quote!(let tiles = left.tiles([8, left.shape()[2]]).pad(0.0);)),
            "shape axis `2` is out of bounds for kernel input `left` with rank `2`",
        );
    }

    #[test]
    fn test_kernel_code_generator_generate_loop() {
        let generator = generator();
        let provenance = provenance();
        let left = scoped(quote!(left.clone()));
        let load = scoped(quote!(kernels::__private::core::operations::ReferenceRead::read(&(#left))?));
        assert_eq!(
            generator
                .generate_statement(
                    &parse_quote! {
                        for depth in 0..left.shape()[1].div_ceil(32) {
                            accumulator = left.load();
                        }
                    },
                    &mut scope(),
                )
                .unwrap()
                .to_string(),
            quote! {
                let __kernel_carried = __kernel_context.invoke_with_provenance_scope(#provenance, || {
                    kernels::for_loop(
                        &__kernel_context,
                        0..kernels::shape_div_ceil(
                            kernels::static_extent(__kernel_left_metadata, 1usize)
                                .map_err(kernels::__private::core::programs::ProgramError::custom)?,
                            32
                        )
                        .map_err(kernels::__private::core::programs::ProgramError::custom)?,
                        ::std::vec![left.clone(), indices.clone(), output.clone(), accumulator.clone()],
                        |depth, __kernel_values| {
                            let __kernel_context = depth.context().clone();
                            let left = __kernel_values[0usize].clone();
                            let indices = __kernel_values[1usize].clone();
                            let output = __kernel_values[2usize].clone();
                            let mut accumulator = __kernel_values[3usize].clone();
                            accumulator = #load;
                            ::core::result::Result::Ok(::std::vec![
                                left.clone(),
                                indices.clone(),
                                output.clone(),
                                accumulator.clone()
                            ])
                        },
                    )
                })?;
                accumulator = __kernel_carried[3usize].clone();
            }
            .to_string(),
        );

        let error = |statement| generator.generate_statement(&statement, &mut scope()).unwrap_err().to_string();
        assert_eq!(error(parse_quote!(for (row, column) in 0..1 {})), "kernel range loops require a named index");
        assert_eq!(error(parse_quote!(for depth in depths {})), "kernel loops require a bounded half-open range");
        assert_eq!(error(parse_quote!(for depth in 0.. {})), "kernel loops require both range bounds");
        assert_eq!(error(parse_quote!(for depth in 0..=1 {})), "kernel loops require an unlabeled half-open range");
        assert_eq!(
            error(parse_quote!('outer: for depth in 0..1 {})),
            "kernel loops require an unlabeled half-open range",
        );
        assert_eq!(error(parse_quote!(for left in 0..1 {})), "kernel loop indices cannot shadow an existing binding");
        assert_eq!(error(parse_quote!(for mut depth in 0..1 {})), "kernel loop indices cannot be mutable");
    }

    #[test]
    fn test_kernel_code_generator_generate_condition() {
        let generator = generator();
        let provenance = provenance();
        let indices = scoped(quote!(indices.clone()));
        let predicate = scoped(quote!(kernels::__private::core::operations::ReferenceRead::read(&(#indices))?));
        let left = scoped(quote!(left.clone()));
        let load = scoped(quote!(kernels::__private::core::operations::ReferenceRead::read(&(#left))?));
        let initialization = quote! {
            let __kernel_context = __kernel_values[0].context().clone();
            let left = __kernel_values[0usize].clone();
            let indices = __kernel_values[1usize].clone();
            let output = __kernel_values[2usize].clone();
            let mut accumulator = __kernel_values[3usize].clone();
        };
        let results = quote! {
            ::core::result::Result::Ok(::std::vec![left.clone(), indices.clone(), output.clone(), accumulator.clone()])
        };
        assert_eq!(
            generator
                .generate_statement(
                    &parse_quote!(if indices.load() {
                        accumulator = left.load();
                    }),
                    &mut scope()
                )
                .unwrap()
                .to_string(),
            quote! {
                let __kernel_predicate = #predicate;
                let __kernel_carried = __kernel_context.invoke_with_provenance_scope(#provenance, || {
                    use kernels::__private::core::Condition as _;
                    __kernel_predicate.condition(
                        |__kernel_values| {
                            #initialization
                            accumulator = #load;
                            #results
                        },
                        |__kernel_values| {
                            #initialization
                            #results
                        },
                        ::std::vec![left.clone(), indices.clone(), output.clone(), accumulator.clone()],
                    )
                })?;
                accumulator = __kernel_carried[3usize].clone();
            }
            .to_string(),
        );

        // `else if` branches are lowered as nested conditions, and the locals of a branch are not visible outside it.
        let mut scope = scope();
        assert!(
            generator
                .generate_statement(
                    &parse_quote! {
                        if indices.load() { let row = left.load(); } else if left.load() { let row = left.load(); }
                    },
                    &mut scope,
                )
                .unwrap()
                .to_string()
                .matches("__kernel_predicate . condition")
                .count()
                == 2
        );
        assert!(scope.bindings.iter().all(|binding| binding.name != "row"));
        assert!(
            generator
                .generate_statement(
                    &parse_quote!(if indices.load() {
                    } else {
                        let row = 1;
                    }),
                    &mut scope
                )
                .is_ok()
        );
    }

    #[test]
    fn test_kernel_code_generator_generate_value() {
        assert_eq!(
            generator().generate_value(&parse_quote!(left), &scope()).unwrap().to_string(),
            scoped(quote!(left.clone())).to_string(),
        );
    }

    #[test]
    fn test_kernel_code_generator_generate_unscoped_value() {
        let generator = generator();
        let mut scope = scope();
        scope.tiles.push(TileView {
            name: format_ident!("tiles"),
            source: format_ident!("left"),
            rank: 2,
            element: quote!(f32),
            shape: quote!([8, 32]),
            padding: parse_quote!(0.0),
        });
        let value = |expression| {
            generator
                .generate_unscoped_value(&expression, &scope)
                .map(|value| value.to_string())
                .map_err(|error| error.to_string())
        };

        // Bindings, parenthesized and forwarded expressions, and literals.
        assert_eq!(value(parse_quote!(accumulator)), Ok(quote!(accumulator.clone()).to_string()));
        assert_eq!(value(parse_quote!((accumulator))), Ok(quote!(accumulator.clone()).to_string()));
        assert_eq!(value(invisible_group(quote!(accumulator))), Ok(quote!(accumulator.clone()).to_string()));
        assert_eq!(
            value(parse_quote!(2.0f32)),
            Ok(quote! {
                __kernel_context.lift(kernels::__private::core::arrays::ArrayIrValue::Array(
                    kernels::__private::core::arrays::Array::scalar(2.0f32)?
                ))?
            }
            .to_string()),
        );

        // Reserved names, tile views outside loads, and host Rust expressions are rejected.
        let error = |expression| value(expression).unwrap_err().to_string();
        assert_eq!(error(parse_quote!(__kernel_context)), "kernel bodies cannot use reserved `__kernel_` names");
        assert_eq!(error(parse_quote!(tiles)), "tile views can only be loaded");
        assert_eq!(
            error(parse_quote!(loop {})),
            "unsupported kernel expression; control flow requires staged lowering and cannot execute as host Rust",
        );
        assert_eq!(
            error(parse_quote!(-left)),
            "unsupported kernel expression; control flow requires staged lowering and cannot execute as host Rust",
        );
    }

    #[test]
    fn test_kernel_code_generator_generate_method_call() {
        let generator = generator();
        let scope = scope();
        let call = |expression| {
            generator
                .generate_unscoped_value(&expression, &scope)
                .map(|call| call.to_string())
                .map_err(|error| error.to_string())
        };
        let left = scoped(quote!(left.clone()));
        let accumulator = scoped(quote!(accumulator.clone()));
        assert_eq!(
            call(parse_quote!(left.dot(accumulator))),
            Ok(quote!(kernels::dot(&(#left), &(#accumulator))?).to_string()),
        );
        assert_eq!(
            call(parse_quote!(left.sum([0, 1]))),
            Ok(quote! {
                __kernel_context.bind(
                    kernels::__private::core::arrays::ArrayIrOperation::from(
                        kernels::__private::core::arrays::ArrayOperation::Reduce(
                            kernels::__private::core::operations::ReduceOperation::new(
                                ::std::vec![0usize, 1usize],
                                kernels::__private::core::operations::ReductionKind::Sum,
                            ),
                        )
                    ),
                    ::std::vec![],
                    &[#left],
                )?.remove(0)
            }
            .to_string()),
        );

        // Stores into an output that is not tiled write the whole output.
        let mut untiled = self::generator();
        untiled.output.tiling = None;
        let output = scoped(quote!(output.clone()));
        assert_eq!(
            untiled
                .generate_unscoped_value(&parse_quote!(output.store(accumulator)), &scope)
                .unwrap()
                .to_string(),
            quote!(kernels::__private::core::operations::ReferenceWrite::write(&(#output), &(#accumulator))?)
                .to_string(),
        );

        let error = |expression| call(expression).unwrap_err().to_string();
        assert_eq!(
            error(parse_quote!(left.load::<f32>())),
            "kernel capability calls do not accept explicit generic arguments",
        );
        assert_eq!(error(parse_quote!(left.load(0))), "unsupported kernel capability call");
        assert_eq!(error(parse_quote!(left.exp())), "unsupported kernel capability call");
        assert_eq!(error(parse_quote!(left.sum(axes))), "kernel reduction axes require an array of integer literals");
        assert_eq!(error(parse_quote!(left.sum([axis]))), "kernel reduction axes require integer literals");
    }

    #[test]
    fn test_kernel_code_generator_generate_tile_load() {
        let generator = generator();
        let mut scope = scope();
        generator
            .generate_statement(&parse_quote!(let tiles = left.tiles([8, 32]).pad(0.0);), &mut scope)
            .unwrap();
        let load = |expression| {
            generator
                .generate_unscoped_value(&expression, &scope)
                .map(|load| load.to_string())
                .map_err(|error| error.to_string())
        };
        let indices = scoped(quote!(indices.clone()));
        let one = scoped(quote! {
            __kernel_context.lift(kernels::__private::core::arrays::ArrayIrValue::Array(
                kernels::__private::core::arrays::Array::scalar(1)?
            ))?
        });
        assert_eq!(
            load(parse_quote!(tiles.load([indices, 1]))),
            Ok(quote! {
                kernels::tile_load::<f32, _>(
                    &__kernel_context,
                    &left,
                    ([8, 32]).to_vec(),
                    &[#indices, #one],
                    0.0,
                )?
            }
            .to_string()),
        );

        let error = |expression| load(expression).unwrap_err().to_string();
        assert_eq!(error(parse_quote!(tiles.store([0, 0]))), "tile views only support `load([indices])`");
        assert_eq!(error(parse_quote!(tiles.load(indices))), "tile indices require an explicit array");
        assert_eq!(error(parse_quote!(tiles.load([0]))), "tile loads from `left` require `2` indices");
    }

    #[test]
    fn test_kernel_code_generator_generate_zeros() {
        let generator = generator();
        let zeros = |expression| {
            generator
                .generate_unscoped_value(&expression, &scope())
                .map(|zeros| zeros.to_string())
                .map_err(|error| error.to_string())
        };
        assert_eq!(
            zeros(parse_quote!(zeros::<Element>([8, indices.shape()[0]]))),
            Ok(quote! {
                kernels::zeros::<Element, _>(
                    &__kernel_context,
                    [
                        8,
                        kernels::static_extent(__kernel_indices_metadata, 0usize)
                            .map_err(kernels::__private::core::programs::ProgramError::custom)?
                    ]
                )?
            }
            .to_string()),
        );

        let error = |expression| zeros(expression).unwrap_err().to_string();
        assert_eq!(error(parse_quote!((helpers)([8]))), "kernel helpers must be recognized named capabilities");
        assert_eq!(error(parse_quote!(helpers::zeros::<f32>([8]))), "unsupported kernel helper call");
        assert_eq!(error(parse_quote!(zeros::<f32>([8], [8]))), "unsupported kernel helper call");
        assert_eq!(error(parse_quote!(ones::<f32>([8]))), "unsupported kernel helper call");
        assert_eq!(error(parse_quote!(zeros([8]))), "kernel zeros requires one explicit element type");
        assert_eq!(error(parse_quote!(zeros::<'static>([8]))), "kernel zeros requires one element type");
    }

    #[test]
    fn test_kernel_code_generator_generate_arithmetic() {
        let generator = generator();
        let left = scoped(quote!(left.clone()));
        let accumulator = scoped(quote!(accumulator.clone()));
        assert_eq!(
            generator.generate_unscoped_value(&parse_quote!(left - accumulator), &scope()).unwrap().to_string(),
            quote! {
                __kernel_context.bind(
                    kernels::__private::core::arrays::ArrayIrOperation::from(
                        kernels::__private::core::arrays::ArrayOperation::Sub(
                            kernels::__private::core::operations::SubOperation::new(),
                        )
                    ),
                    ::std::vec![],
                    &[#left, #accumulator],
                )?.remove(0)
            }
            .to_string(),
        );
        assert_eq!(
            generator
                .generate_unscoped_value(&parse_quote!(left % accumulator), &scope())
                .unwrap_err()
                .to_string(),
            "unsupported kernel arithmetic operator",
        );
    }

    #[test]
    fn test_kernel_code_generator_generate_provenance_scope() {
        assert_eq!(generator().generate_provenance_scope(Span::call_site()).to_string(), provenance().to_string());
    }

    #[test]
    fn test_input_element_type() {
        let input =
            |data_type| Input { name: format_ident!("input"), data_type: format_ident!("{data_type}"), rank: 1 };
        assert_eq!(input("F32").element_type().unwrap().to_string(), "f32");
        assert_eq!(input("Boolean").element_type().unwrap().to_string(), "bool");
        assert_eq!(
            input("BF16").element_type().unwrap_err().to_string(),
            "tile padding literal requires a supported Rust scalar type",
        );
    }

    #[test]
    fn test_scope_region() {
        let scope = scope();
        assert!(!scope.region);
        let region = scope.region();
        assert!(region.region);
        assert_eq!(region.bindings.len(), scope.bindings.len());
    }

    #[test]
    fn test_scope_validate_new_name() {
        let scope = scope();
        let pattern = |tokens| match syn::Pat::parse_single.parse2(tokens).unwrap() {
            syn::Pat::Ident(pattern) => pattern,
            _ => panic!("expected an identifier pattern"),
        };
        assert_eq!(scope.validate_new_name(&pattern(quote!(mut row)), "locals", true).unwrap(), "row");
        assert_eq!(
            scope.validate_new_name(&pattern(quote!(mut row)), "loop indices", false).unwrap_err().to_string(),
            "kernel loop indices cannot be mutable",
        );
        assert_eq!(
            scope.validate_new_name(&pattern(quote!(ref row)), "locals", true).unwrap_err().to_string(),
            "kernel locals cannot use the reserved `__kernel_` prefix, `ref`, or subpatterns",
        );
        assert_eq!(
            scope.validate_new_name(&pattern(quote!(accumulator)), "locals", true).unwrap_err().to_string(),
            "kernel locals cannot shadow an existing binding",
        );
    }

    #[test]
    fn test_scope_generate_initialization() {
        assert_eq!(
            Scope::new(vec![
                Binding { name: format_ident!("left"), mutable: false },
                Binding { name: format_ident!("accumulator"), mutable: true },
            ])
            .generate_initialization()
            .to_string(),
            quote! {
                let left = __kernel_values[0usize].clone();
                let mut accumulator = __kernel_values[1usize].clone();
            }
            .to_string(),
        );
    }

    #[test]
    fn test_scope_generate_restoration() {
        assert_eq!(scope().generate_restoration().to_string(), "accumulator = __kernel_carried [3usize] . clone () ;");
    }

    #[test]
    fn test_ungroup() {
        assert_eq!(ungroup(&invisible_group(quote!(left))).to_compact_string(), "left");
        assert_eq!(ungroup(&parse_quote!((left))).to_compact_string(), "(left)");
    }

    #[test]
    fn test_identifier() {
        assert_eq!(identifier(&invisible_group(quote!(left))).unwrap(), "left");
        assert_eq!(identifier(&parse_quote!(self::left)), None);
        assert_eq!(identifier(&parse_quote!(left.load())), None);
    }
}
