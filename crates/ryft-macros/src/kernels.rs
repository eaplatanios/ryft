//! Syntax lowering for the experimental kernel authoring attribute.

use proc_macro2::TokenStream;
use quote::{format_ident, quote, quote_spanned};
use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::{BinOp, Expr, FnArg, Ident, ItemFn, Meta, Pat, Stmt, Token, Type};

/// Parsed signature declaration; metadata remains Rust expressions validated before generated tracing.
struct Parameter {
    /// Local name used by the source body.
    name: Ident,

    /// Canonical data-type variant.
    data_type: Ident,

    /// Required input rank; output rank follows its shape.
    rank: Option<usize>,

    /// Full output shape, absent for input parameters.
    shape: Option<Expr>,

    /// Optional output block shape.
    tile: Option<Expr>,

    /// Explicit output boundary policy.
    boundary: Option<Ident>,
}

/// One lexical tracer binding carried explicitly across staged control flow.
#[derive(Clone)]
struct Binding {
    /// Source binding name.
    name: Ident,

    /// Whether its post-region value must replace the source binding.
    mutable: bool,
}

/// Lexical tile declaration lowered directly to operation metadata.
#[derive(Clone)]
struct TileBinding {
    /// Source-level local name.
    name: Ident,

    /// Canonical reference binding.
    source: Ident,

    /// Static tile shape expression.
    shape: Expr,

    /// Explicit scalar fill expression.
    other: Expr,
}

/// Source metadata available while lowering a lexical body.
#[derive(Clone)]
struct BodyMetadata {
    /// Declared source parameter element types.
    input_types: Vec<(Ident, Ident)>,

    /// Output source name.
    output: Ident,

    /// Whether this is a nested control-flow region.
    nested: bool,

    /// Whether the output uses a mapped tile window.
    tiled: bool,

    /// Lexically visible tile declarations.
    tiles: Vec<TileBinding>,
}

impl BodyMetadata {
    /// Preserves lexical metadata while entering a traced child region.
    fn in_region(&self) -> Self {
        Self { nested: true, ..self.clone() }
    }
}

/// Expands one annotated function into a functional callable and an inspectable definition constructor.
pub(crate) fn expand(attributes: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
    let function: ItemFn = syn::parse2(item)?;
    if function.sig.asyncness.is_some()
        || function.sig.unsafety.is_some()
        || function.sig.constness.is_some()
        || function.sig.abi.is_some()
        || function.sig.variadic.is_some()
        || function.sig.generics.where_clause.is_some()
        || !function.sig.generics.params.is_empty()
        || !matches!(function.sig.output, syn::ReturnType::Default)
    {
        return Err(syn::Error::new_spanned(
            &function.sig,
            "kernel functions must be safe, synchronous, non-generic, and return through output parameters",
        ));
    }
    let mut core: syn::Path = syn::parse_quote!(::ryft_core);
    // Body attributes must not disappear when statements and expressions become tracing calls.
    /// Retains the first unsupported body attribute for a source-positioned diagnostic.
    struct BodyAttributes {
        /// First attribute whose semantics cannot be preserved by this lowering.
        error: Option<syn::Error>,
    }

    impl<'ast> Visit<'ast> for BodyAttributes {
        fn visit_attribute(&mut self, attribute: &'ast syn::Attribute) {
            if self.error.is_none() {
                self.error = Some(syn::Error::new_spanned(attribute, "kernel body attributes are unsupported"));
            }
        }
    }

    let mut body_attributes = BodyAttributes { error: None };
    body_attributes.visit_block(&function.block);
    if let Some(error) = body_attributes.error {
        return Err(error);
    }
    let mut requirement = None;
    for attribute in Punctuated::<Meta, Token![,]>::parse_terminated.parse2(attributes)? {
        let Meta::NameValue(attribute) = attribute else {
            return Err(syn::Error::new_spanned(attribute, "expected `requires = expression` or `crate = \"path\"`"));
        };
        if attribute.path.is_ident("requires") {
            requirement = Some(attribute.value);
        } else if attribute.path.is_ident("crate") {
            let Expr::Lit(value) = attribute.value else {
                return Err(syn::Error::new_spanned(attribute, "expected a crate path string"));
            };
            let syn::Lit::Str(value) = value.lit else {
                return Err(syn::Error::new_spanned(value, "expected a crate path string"));
            };
            core = value.parse()?;
        } else {
            return Err(syn::Error::new_spanned(attribute.path, "unsupported kernel attribute"));
        }
    }
    let mut names_seen = Vec::new();
    let mut inputs = Vec::new();
    let mut output = None;
    for argument in &function.sig.inputs {
        let FnArg::Typed(argument) = argument else {
            return Err(syn::Error::new_spanned(argument, "kernel functions cannot take `self`"));
        };
        let Pat::Ident(pattern) = argument.pat.as_ref() else {
            return Err(syn::Error::new_spanned(&argument.pat, "kernel parameters require a name"));
        };
        if pattern.ident.to_string().starts_with("__kernel_") || pattern.subpat.is_some() || pattern.by_ref.is_some() {
            return Err(syn::Error::new_spanned(
                pattern,
                "kernel parameter names cannot use the reserved `__kernel_` prefix or subpatterns",
            ));
        }
        if names_seen.contains(&pattern.ident) {
            return Err(syn::Error::new_spanned(pattern, "kernel parameter names must be unique"));
        }
        names_seen.push(pattern.ident.clone());
        let Type::Reference(reference) = argument.ty.as_ref() else {
            return Err(syn::Error::new_spanned(&argument.ty, "kernel parameters must use `&Array` or `&mut Array`"));
        };
        if reference.lifetime.is_some() {
            return Err(syn::Error::new_spanned(
                reference,
                "kernel parameter references do not accept explicit lifetimes",
            ));
        }
        let Type::Path(array) = reference.elem.as_ref() else {
            return Err(syn::Error::new_spanned(reference, "kernel parameters must use `Array`"));
        };
        if array
            .path
            .segments
            .last()
            .is_none_or(|segment| segment.ident != "Array" || !matches!(segment.arguments, syn::PathArguments::None))
        {
            return Err(syn::Error::new_spanned(array, "kernel parameters must use the canonical non-generic `Array`"));
        }
        if argument.attrs.len() != 1 {
            return Err(syn::Error::new_spanned(
                argument,
                "each kernel parameter requires exactly one `input` or `output` annotation",
            ));
        }
        let annotation = &argument.attrs[0];
        let is_output = annotation.path().is_ident("output");
        if !is_output && !annotation.path().is_ident("input") {
            return Err(syn::Error::new_spanned(annotation, "expected an `input` or `output` annotation"));
        }
        if is_output != reference.mutability.is_some() {
            return Err(syn::Error::new_spanned(
                reference,
                "input parameters use `&Array`; output parameters use `&mut Array`",
            ));
        }
        let mut data_type = None;
        let mut rank = None;
        let mut shape = None;
        let mut tile = None;
        let mut boundary = None;
        for metadata in annotation.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)? {
            let Meta::NameValue(metadata) = metadata else {
                return Err(syn::Error::new_spanned(metadata, "expected parameter metadata `name = value`"));
            };
            if metadata.path.is_ident("data_type") {
                let Expr::Path(value) = metadata.value else {
                    return Err(syn::Error::new_spanned(metadata, "expected a canonical data-type variant"));
                };
                data_type = value.path.get_ident().cloned();
            } else if metadata.path.is_ident("rank") && !is_output {
                let Expr::Lit(value) = metadata.value else {
                    return Err(syn::Error::new_spanned(metadata, "rank must be an integer literal"));
                };
                let syn::Lit::Int(value) = value.lit else {
                    return Err(syn::Error::new_spanned(value, "rank must be an integer literal"));
                };
                rank = Some(value.base10_parse::<usize>()?);
            } else if metadata.path.is_ident("shape") && is_output {
                shape = Some(metadata.value);
            } else if metadata.path.is_ident("tile") && is_output {
                tile = Some(metadata.value);
            } else if metadata.path.is_ident("boundary") && is_output {
                let Expr::Path(value) = metadata.value else {
                    return Err(syn::Error::new_spanned(metadata, "expected a kernel boundary policy"));
                };
                boundary = value.path.get_ident().cloned();
            } else {
                return Err(syn::Error::new_spanned(
                    metadata.path,
                    "unsupported parameter metadata in the whole-array kernel subset",
                ));
            }
        }
        let parameter = Parameter {
            name: pattern.ident.clone(),
            data_type: data_type
                .ok_or_else(|| syn::Error::new_spanned(annotation, "parameter metadata requires `data_type`"))?,
            rank,
            shape,
            tile,
            boundary,
        };
        if is_output {
            if parameter.shape.is_none() || output.is_some() {
                return Err(syn::Error::new_spanned(
                    annotation,
                    "the initial kernel subset requires exactly one output with an explicit shape",
                ));
            }
            output = Some(parameter);
        } else {
            if parameter.rank.is_none() {
                return Err(syn::Error::new_spanned(annotation, "input metadata requires `rank`"));
            }
            inputs.push(parameter);
        }
    }
    let output =
        output.ok_or_else(|| syn::Error::new_spanned(&function.sig, "a kernel requires one output parameter"))?;
    let names = inputs.iter().map(|parameter| parameter.name.clone()).collect::<Vec<_>>();
    let shape = metadata(output.shape.as_ref().unwrap(), &names, &core)?;
    let requirement = requirement.as_ref().map(|expression| metadata(expression, &names, &core)).transpose()?;
    let checks = inputs.iter().map(|parameter| {
        let name = &parameter.name;
        let data_type = &parameter.data_type;
        let rank = parameter.rank.unwrap();
        let message = format!("kernel input `{name}` requires data type `{data_type}` and rank `{rank}`");
        quote! { if #name.data_type() != #core::arrays::DataType::#data_type || #name.rank() != #rank {
            return Err(#core::kernels::KernelError::Type(#core::programs::TypeError::invalid(#message)));
        } }
    });
    let require_check = requirement.map(|requirement| {
        quote! {
            if !(#requirement) {
                return Err(#core::kernels::KernelError::Type(#core::programs::TypeError::invalid(
                    "kernel shape requirement is not satisfied",
                )));
            }
        }
    });
    let output_name = &output.name;
    let output_data_type = &output.data_type;
    let mut bindings = names
        .iter()
        .chain(std::iter::once(output_name))
        .map(|name| Binding { name: name.clone(), mutable: false })
        .collect::<Vec<_>>();
    if output.tile.is_some() != output.boundary.is_some() {
        return Err(syn::Error::new_spanned(
            &function.sig,
            "output tiling requires explicit `tile` and `boundary` metadata",
        ));
    }
    let mut body_metadata = BodyMetadata {
        input_types: inputs.iter().map(|parameter| (parameter.name.clone(), parameter.data_type.clone())).collect(),
        output: output.name.clone(),
        nested: false,
        tiled: output.tile.is_some(),
        tiles: Vec::new(),
    };
    let body = body(&function.block.stmts, &core, &mut bindings, &names, &mut body_metadata)?;
    let metadata_names = names.iter().map(|name| format_ident!("__kernel_{}_metadata", name)).collect::<Vec<_>>();
    let input_bindings = names
        .iter()
        .enumerate()
        .map(|(index, name)| quote!(let #name = __kernel_references[#index].clone();));
    let output_index = names.len();
    let function_name = &function.sig.ident;
    let visibility = &function.vis;
    let attributes = &function.attrs;
    let module_attributes = attributes
        .iter()
        .filter(|attribute| attribute.path().is_ident("cfg") || attribute.path().is_ident("cfg_attr"));
    let types = names.iter().map(|name| format_ident!("__kernel_{}_type", name)).collect::<Vec<_>>();
    let operation = if let Some(tile) = &output.tile {
        let tile = metadata(tile, &names, &core)?;
        let boundary = match output.boundary.as_ref().unwrap().to_string().as_str() {
            "masked" => quote!(#core::kernels::BoundaryPolicy::Masked),
            "in_bounds" => quote!(#core::kernels::BoundaryPolicy::InBounds),
            _ => {
                return Err(syn::Error::new_spanned(
                    output.boundary.as_ref().unwrap(),
                    "unsupported kernel boundary policy",
                ));
            }
        };
        quote!(#core::kernels::tiled_call(&[#(#names.clone()),*], __kernel_output_type, (#tile).to_vec(), #boundary)?)
    } else {
        quote!({
            let __kernel_parameters = vec![
                #(#core::kernels::whole_array_parameter(
                    #names.clone(), #core::kernels::KernelParameterAccess::ReadOnly,
                )?,)*
                #core::kernels::whole_array_parameter(
                    __kernel_output_type, #core::kernels::KernelParameterAccess::WriteOnly,
                )?,
            ];
            #core::kernels::KernelCallOperation::new(
                #core::kernels::Grid::new(vec![])?,
                __kernel_parameters,
            )?
        })
    };
    Ok(quote! {
        #(#attributes)*
        #visibility fn #function_name<__KernelValue: #core::kernels::KernelCall>(#(#names: &__KernelValue),*)
            -> ::core::result::Result<__KernelValue, #core::programs::ProgramError>
        {
            #(let #types = #core::programs::Typed::r#type(#names);)*
            let __kernel_definition = #function_name::definition(#(#types.as_ref()),*)
                .map_err(#core::programs::ProgramError::custom)?;
            let mut __kernel_outputs = __KernelValue::call_kernel(&__kernel_definition, &[#(#names.clone()),*])?;
            if __kernel_outputs.len() != 1 {
                return Err(#core::programs::ProgramError::InvalidOutputCount {
                    expected: 1, actual: __kernel_outputs.len(),
                });
            }
            Ok(__kernel_outputs.remove(0))
        }

        /// Inspectable portable definition generated from the annotated kernel function.
        #(#module_attributes)*
        #visibility mod #function_name {
            /// Validates the full signature, then traces one whole-array kernel body through Ryft.
            pub fn definition(#(#names: &#core::arrays::ArrayType),*)
                -> ::core::result::Result<#core::kernels::KernelDefinition, #core::kernels::KernelError>
            {
                #(#checks)*
                #require_check
                let __kernel_output_type = #core::arrays::ArrayType::new_static(
                    #core::arrays::DataType::#output_data_type, #shape,
                );
                let __kernel_operation = #operation;
                #(let #metadata_names = #names.clone();)*
                #core::kernels::KernelDefinition::trace(__kernel_operation,
                    |(__kernel_references, __kernel_coordinates)| {
                    use #core::contexts::Context as _;
                    let __kernel_context = __kernel_references[0].context().clone();
                    #(#input_bindings)*
                    let #output_name = __kernel_references[#output_index].clone();
                    #body
                    Ok(())
                })
            }
        }
    })
}

/// Lowers pure shape metadata, rejecting host callbacks and checking every shape index.
fn metadata(expression: &Expr, inputs: &[Ident], core: &syn::Path) -> syn::Result<TokenStream> {
    shape_metadata(expression, inputs, core, false)
}

/// Shape metadata inside a trace uses captured immutable type descriptors and preserves typed errors.
fn shape_metadata(expression: &Expr, inputs: &[Ident], core: &syn::Path, in_trace: bool) -> syn::Result<TokenStream> {
    match expression {
        Expr::Lit(value) => Ok(quote!(#value)),
        Expr::Array(array) => {
            let elements = array
                .elems
                .iter()
                .map(|element| shape_metadata(element, inputs, core, in_trace))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(quote!([#(#elements),*]))
        }
        Expr::Paren(value) => {
            let value = shape_metadata(&value.expr, inputs, core, in_trace)?;
            Ok(quote!((#value)))
        }
        Expr::Group(value) => {
            let value = shape_metadata(&value.expr, inputs, core, in_trace)?;
            Ok(quote!((#value)))
        }
        Expr::Binary(binary) => {
            let left = shape_metadata(&binary.left, inputs, core, in_trace)?;
            let right = shape_metadata(&binary.right, inputs, core, in_trace)?;
            let operator = &binary.op;
            if !matches!(
                operator,
                BinOp::Eq(_)
                    | BinOp::Ne(_)
                    | BinOp::Lt(_)
                    | BinOp::Le(_)
                    | BinOp::Gt(_)
                    | BinOp::Ge(_)
                    | BinOp::And(_)
                    | BinOp::Or(_)
            ) {
                return Err(syn::Error::new_spanned(operator, "unsupported shape metadata operator"));
            }
            Ok(quote!(#left #operator #right))
        }
        Expr::Index(index) => {
            let Expr::MethodCall(shape) = index.expr.as_ref() else {
                return Err(syn::Error::new_spanned(index, "shape metadata requires `input.shape()[axis]`"));
            };
            let Expr::Path(input) = shape.receiver.as_ref() else {
                return Err(syn::Error::new_spanned(shape, "shape metadata requires an input name"));
            };
            let Some(name) = input.path.get_ident().filter(|name| inputs.contains(name)) else {
                return Err(syn::Error::new_spanned(input, "unknown shape metadata input"));
            };
            let Expr::Lit(axis) = index.index.as_ref() else {
                return Err(syn::Error::new_spanned(index, "shape axis must be an integer literal"));
            };
            let syn::Lit::Int(axis) = &axis.lit else {
                return Err(syn::Error::new_spanned(axis, "shape axis must be an integer literal"));
            };
            let axis = axis.base10_parse::<usize>()?;
            if shape.method != "shape" || !shape.args.is_empty() || shape.turbofish.is_some() {
                return Err(syn::Error::new_spanned(shape, "unsupported shape metadata call"));
            }
            if in_trace {
                let name = format_ident!("__kernel_{}_metadata", name);
                Ok(quote!(#core::kernels::static_extent(&#name, #axis).map_err(#core::programs::ProgramError::custom)?))
            } else {
                Ok(quote!(#core::kernels::static_extent(#name, #axis)?))
            }
        }
        Expr::MethodCall(call) if call.method == "div_ceil" && call.args.len() == 1 && call.turbofish.is_none() => {
            let extent = shape_metadata(&call.receiver, inputs, core, in_trace)?;
            let divisor = shape_metadata(&call.args[0], inputs, core, in_trace)?;
            if in_trace {
                Ok(quote!(#core::kernels::shape_div_ceil(#extent, #divisor)
                        .map_err(#core::programs::ProgramError::custom)?))
            } else {
                Ok(quote!(#core::kernels::shape_div_ceil(#extent, #divisor)?))
            }
        }
        _ => Err(syn::Error::new_spanned(expression, "unsupported shape metadata expression")),
    }
}

/// Lowers supported statements to fallible canonical staging calls; host control flow never executes by accident.
fn body(
    statements: &[Stmt],
    core: &syn::Path,
    bindings: &mut Vec<Binding>,
    inputs: &[Ident],
    metadata: &mut BodyMetadata,
) -> syn::Result<TokenStream> {
    let mut output = TokenStream::new();
    for statement in statements {
        let tokens = match statement {
            Stmt::Local(local) if matches!(&local.pat, Pat::Slice(_)) => {
                let Pat::Slice(pattern) = &local.pat else { unreachable!() };
                let Some(initializer) = &local.init else {
                    return Err(syn::Error::new_spanned(local, "tile coordinates require an initializer"));
                };
                let Expr::MethodCall(call) = initializer.expr.as_ref() else {
                    return Err(syn::Error::new_spanned(local, "kernel destructuring only supports tile coordinates"));
                };
                if !metadata.tiled
                    || metadata.nested
                    || !matches!(call.receiver.as_ref(), Expr::Path(path) if path.path.is_ident(&metadata.output))
                    || call.method != "tile_index"
                    || !call.args.is_empty()
                    || call.turbofish.is_some()
                {
                    return Err(syn::Error::new_spanned(call, "tile coordinates require a tiled output"));
                }
                let mut declarations = Vec::new();
                for (axis, pattern) in pattern.elems.iter().enumerate() {
                    let Pat::Ident(pattern) = pattern else {
                        return Err(syn::Error::new_spanned(pattern, "tile coordinates require named bindings"));
                    };
                    if pattern.mutability.is_some()
                        || pattern.by_ref.is_some()
                        || pattern.subpat.is_some()
                        || pattern.ident.to_string().starts_with("__kernel_")
                        || bindings.iter().any(|binding| binding.name == pattern.ident)
                    {
                        return Err(syn::Error::new_spanned(pattern, "tile coordinates require fresh immutable names"));
                    }
                    let name = &pattern.ident;
                    declarations.push(quote!(let #name = __kernel_coordinates[#axis].clone();));
                    bindings.push(Binding { name: name.clone(), mutable: false });
                }
                let rank = declarations.len();
                quote! {
                    if __kernel_coordinates.len() != #rank {
                        return Err(#core::programs::TypeError::invalid(
                            "tile coordinate binding count must equal grid rank",
                        ).into());
                    }
                    #(#declarations)*
                }
            }
            Stmt::Local(local) => {
                let Pat::Ident(pattern) = &local.pat else {
                    return Err(syn::Error::new_spanned(&local.pat, "unsupported kernel local pattern"));
                };
                if pattern.ident.to_string().starts_with("__kernel_")
                    || pattern.subpat.is_some()
                    || pattern.by_ref.is_some()
                {
                    return Err(syn::Error::new_spanned(
                        pattern,
                        "kernel local names cannot use the reserved `__kernel_` prefix or subpatterns",
                    ));
                }
                if bindings.iter().any(|binding| binding.name == pattern.ident)
                    || metadata.tiles.iter().any(|tile| tile.name == pattern.ident)
                {
                    return Err(syn::Error::new_spanned(pattern, "kernel locals cannot shadow an existing binding"));
                }
                let Some(initializer) = &local.init else {
                    return Err(syn::Error::new_spanned(local, "kernel locals require an initializer"));
                };
                if initializer.diverge.is_some() {
                    return Err(syn::Error::new_spanned(local, "kernel `let else` is unsupported"));
                }
                if let Expr::MethodCall(padding) = initializer.expr.as_ref()
                    && padding.method == "pad"
                {
                    let Expr::MethodCall(tiles) = padding.receiver.as_ref() else {
                        return Err(syn::Error::new_spanned(padding, "padding requires a tile view"));
                    };
                    let Expr::Path(source) = tiles.receiver.as_ref() else {
                        return Err(syn::Error::new_spanned(tiles, "tile views require a named input reference"));
                    };
                    let Some(source) = source.path.get_ident() else {
                        return Err(syn::Error::new_spanned(source, "tile views require a named input reference"));
                    };
                    if tiles.method != "tiles"
                        || tiles.args.len() != 1
                        || padding.args.len() != 1
                        || tiles.turbofish.is_some()
                        || padding.turbofish.is_some()
                        || pattern.mutability.is_some()
                        || !inputs.contains(source)
                    {
                        return Err(syn::Error::new_spanned(
                            &initializer.expr,
                            "expected immutable `input.tiles(shape).pad(scalar)` metadata",
                        ));
                    }
                    metadata.tiles.push(TileBinding {
                        name: pattern.ident.clone(),
                        source: source.clone(),
                        shape: tiles.args[0].clone(),
                        other: padding.args[0].clone(),
                    });
                    continue;
                }
                let value = value(&initializer.expr, core, inputs, metadata)?;
                bindings.retain(|binding| binding.name != pattern.ident);
                bindings.push(Binding { name: pattern.ident.clone(), mutable: pattern.mutability.is_some() });
                quote!(let #pattern = #value;)
            }
            Stmt::Expr(Expr::Assign(assignment), _) => {
                let Expr::Path(name) = assignment.left.as_ref() else {
                    return Err(syn::Error::new_spanned(assignment, "kernel assignment requires a local name"));
                };
                let value = value(&assignment.right, core, inputs, metadata)?;
                quote!(#name = #value;)
            }
            Stmt::Expr(Expr::Binary(binary), _)
                if matches!(
                    binary.op,
                    BinOp::AddAssign(_) | BinOp::SubAssign(_) | BinOp::MulAssign(_) | BinOp::DivAssign(_)
                ) =>
            {
                let Expr::Path(name) = binary.left.as_ref() else {
                    return Err(syn::Error::new_spanned(binary, "kernel assignment requires a local name"));
                };
                let right = &binary.right;
                let expression: Expr = match binary.op {
                    BinOp::AddAssign(_) => syn::parse_quote!(#name + #right),
                    BinOp::SubAssign(_) => syn::parse_quote!(#name - #right),
                    BinOp::MulAssign(_) => syn::parse_quote!(#name * #right),
                    _ => syn::parse_quote!(#name / #right),
                };
                let value = value(&expression, core, inputs, metadata)?;
                quote!(#name = #value;)
            }
            Stmt::Expr(Expr::ForLoop(loop_expression), _) => {
                let Pat::Ident(index) = loop_expression.pat.as_ref() else {
                    return Err(syn::Error::new_spanned(
                        &loop_expression.pat,
                        "kernel range loops require a named index",
                    ));
                };
                let Expr::Range(range) = loop_expression.expr.as_ref() else {
                    return Err(syn::Error::new_spanned(
                        &loop_expression.expr,
                        "kernel loops require a bounded half-open range",
                    ));
                };
                let (Some(start), Some(end)) = (&range.start, &range.end) else {
                    return Err(syn::Error::new_spanned(range, "kernel loops require both range bounds"));
                };
                if !matches!(range.limits, syn::RangeLimits::HalfOpen(_)) || loop_expression.label.is_some() {
                    return Err(syn::Error::new_spanned(
                        loop_expression,
                        "kernel loops require an unlabeled half-open range",
                    ));
                }
                let start = shape_metadata(start, inputs, core, true)?;
                let end = shape_metadata(end, inputs, core, true)?;
                let names = bindings.iter().map(|binding| &binding.name).collect::<Vec<_>>();
                if names.contains(&&index.ident)
                    || metadata.tiles.iter().any(|tile| tile.name == index.ident)
                    || index.mutability.is_some()
                    || index.by_ref.is_some()
                    || index.subpat.is_some()
                    || index.ident.to_string().starts_with("__kernel_")
                {
                    return Err(syn::Error::new_spanned(
                        index,
                        "kernel loop indices cannot shadow an existing binding",
                    ));
                }
                let initialization = initialize_bindings(bindings);
                let restoration = restore_bindings(bindings);
                let mut nested = bindings.clone();
                nested.push(Binding { name: index.ident.clone(), mutable: false });
                let body = body(&loop_expression.body.stmts, core, &mut nested, inputs, &mut metadata.in_region())?;
                let source_scope = quote_spanned!(loop_expression.span()=> #core::programs::ProvenanceScope::new(
                    format!("{}:{}:{}", file!(), line!(), column!()),
                ));
                let index = &index.ident;
                quote! {
                    let __kernel_carried = __kernel_context.invoke_with_provenance_scope(#source_scope,
                        || #core::kernels::for_loop(&__kernel_context, #start..#end,
                        vec![#(#names.clone()),*], |#index, __kernel_values| {
                            let __kernel_context = #index.context().clone();
                            #initialization
                            #body
                            Ok(vec![#(#names.clone()),*])
                        }))?;
                    #restoration
                }
            }
            Stmt::Expr(Expr::If(condition), _) => {
                let predicate = value(&condition.cond, core, inputs, metadata)?;
                let names = bindings.iter().map(|binding| &binding.name).collect::<Vec<_>>();
                let initialization = initialize_bindings(bindings);
                let restoration = restore_bindings(bindings);
                let then_body =
                    body(&condition.then_branch.stmts, core, &mut bindings.clone(), inputs, &mut metadata.in_region())?;
                let else_body = if let Some((_, branch)) = &condition.else_branch {
                    match branch.as_ref() {
                        Expr::Block(block) => {
                            body(&block.block.stmts, core, &mut bindings.clone(), inputs, &mut metadata.in_region())?
                        }
                        Expr::If(_) => body(
                            &[Stmt::Expr(branch.as_ref().clone(), None)],
                            core,
                            &mut bindings.clone(),
                            inputs,
                            &mut metadata.in_region(),
                        )?,
                        _ => {
                            return Err(syn::Error::new_spanned(
                                branch,
                                "kernel else branches require a block or conditional",
                            ));
                        }
                    }
                } else {
                    TokenStream::new()
                };
                let source_scope = quote_spanned!(condition.span()=> #core::programs::ProvenanceScope::new(
                    format!("{}:{}:{}", file!(), line!(), column!()),
                ));
                quote! {
                    let __kernel_predicate = #predicate;
                    let __kernel_carried = __kernel_context.invoke_with_provenance_scope(#source_scope,
                        || #core::kernels::condition(&__kernel_context, &__kernel_predicate,
                        vec![#(#names.clone()),*],
                        |__kernel_values| {
                            let __kernel_context = __kernel_values[0].context().clone();
                            #initialization
                            #then_body
                            Ok(vec![#(#names.clone()),*])
                        },
                        |__kernel_values| {
                            let __kernel_context = __kernel_values[0].context().clone();
                            #initialization
                            #else_body
                            Ok(vec![#(#names.clone()),*])
                        }))?;
                    #restoration
                }
            }
            Stmt::Expr(expression, _) => {
                let value = value(expression, core, inputs, metadata)?;
                quote!(#value;)
            }
            _ => {
                return Err(syn::Error::new_spanned(
                    statement,
                    "unsupported kernel statement; arbitrary host items and macros cannot run while tracing",
                ));
            }
        };
        output.extend(tokens);
    }
    Ok(output)
}

/// Creates branch-local SSA bindings from an explicit region argument vector.
fn initialize_bindings(bindings: &[Binding]) -> TokenStream {
    let bindings = bindings.iter().enumerate().map(|(index, binding)| {
        let name = &binding.name;
        let mutable = binding.mutable.then(|| quote!(mut));
        quote!(let #mutable #name = __kernel_values[#index].clone();)
    });
    quote!(#(#bindings)*)
}

/// Replaces mutable outer bindings with a completed region's returned SSA values.
fn restore_bindings(bindings: &[Binding]) -> TokenStream {
    let bindings = bindings.iter().enumerate().filter(|(_, binding)| binding.mutable).map(|(index, binding)| {
        let name = &binding.name;
        quote!(#name = __kernel_carried[#index].clone();)
    });
    quote!(#(#bindings)*)
}

/// Rewrites expressions to existing capability operations while retaining source spans for Rust diagnostics.
fn value(expression: &Expr, core: &syn::Path, inputs: &[Ident], metadata: &BodyMetadata) -> syn::Result<TokenStream> {
    let generated = value_inner(expression, core, inputs, metadata)?;
    let span = expression.span();
    Ok(quote_spanned!(span=> __kernel_context.invoke_with_provenance_scope(
        #core::programs::ProvenanceScope::new(format!("{}:{}:{}", file!(), line!(), column!())),
        || Ok::<_, #core::programs::ProgramError>(#generated),
    )?))
}

/// Lowers one expression before adding its canonical source provenance scope.
fn value_inner(
    expression: &Expr,
    core: &syn::Path,
    inputs: &[Ident],
    metadata: &BodyMetadata,
) -> syn::Result<TokenStream> {
    let span = expression.span();
    match expression {
        Expr::Path(path) if path.path.get_ident().is_some() => Ok(quote_spanned!(span=> #path.clone())),
        Expr::Paren(value_expression) => value(&value_expression.expr, core, inputs, metadata),
        Expr::MethodCall(call) => {
            if call.turbofish.is_some() {
                return Err(syn::Error::new_spanned(
                    call,
                    "kernel capability calls do not accept explicit generic arguments",
                ));
            }
            if call.method == "load"
                && call.args.len() == 1
                && let Expr::Path(receiver) = call.receiver.as_ref()
                && let Some(tile) = metadata.tiles.iter().find(|tile| receiver.path.is_ident(&tile.name))
            {
                let Expr::Array(indices) = &call.args[0] else {
                    return Err(syn::Error::new_spanned(call, "tile indices require an explicit array"));
                };
                let indices = indices
                    .elems
                    .iter()
                    .map(|index| value(index, core, inputs, metadata))
                    .collect::<syn::Result<Vec<_>>>()?;
                let shape = shape_metadata(&tile.shape, inputs, core, true)?;
                let source = &tile.source;
                let other = &tile.other;
                if !matches!(other, Expr::Lit(_))
                    && !matches!(other, Expr::Unary(unary)
                        if matches!(unary.op, syn::UnOp::Neg(_)) && matches!(unary.expr.as_ref(), Expr::Lit(_)))
                {
                    return Err(syn::Error::new_spanned(other, "tile padding requires a scalar literal"));
                }
                let data_type = &metadata.input_types.iter().find(|(name, _)| name == source).unwrap().1;
                let element = match data_type.to_string().as_str() {
                    "F32" => quote!(f32),
                    "F64" => quote!(f64),
                    "I8" => quote!(i8),
                    "I16" => quote!(i16),
                    "I32" => quote!(i32),
                    "I64" => quote!(i64),
                    "U8" => quote!(u8),
                    "U16" => quote!(u16),
                    "U32" => quote!(u32),
                    "U64" => quote!(u64),
                    "Boolean" => quote!(bool),
                    _ => {
                        return Err(syn::Error::new_spanned(
                            data_type,
                            "tile padding literal requires a supported Rust scalar type",
                        ));
                    }
                };
                return Ok(quote_spanned!(span=> #core::kernels::tile_load::<#element, _>(
                        &__kernel_context, &#source, (#shape).to_vec(), &[#(#indices),*], #other,
                    )?));
            }
            let receiver = value(&call.receiver, core, inputs, metadata)?;
            match call.method.to_string().as_str() {
                "load" if call.args.is_empty() => {
                    Ok(quote_spanned!(span=> #core::operations::ReferenceRead::read(&(#receiver))?))
                }
                "store" if call.args.len() == 1 => {
                    let replacement = value(&call.args[0], core, inputs, metadata)?;
                    if metadata.tiled {
                        Ok(quote_spanned!(span=> #core::kernels::tile_store(
                                &__kernel_context, &(#receiver), &(#replacement),
                            )?))
                    } else {
                        Ok(quote_spanned!(span=> #core::operations::ReferenceWrite::write(
                                &(#receiver), &(#replacement),
                            )?))
                    }
                }
                "dot" if call.args.len() == 1 => {
                    let right = value(&call.args[0], core, inputs, metadata)?;
                    Ok(quote_spanned!(span=> #core::kernels::dot(&(#receiver), &(#right))?))
                }
                "sum" if call.args.len() == 1 => {
                    let Expr::Array(axes) = &call.args[0] else {
                        return Err(syn::Error::new_spanned(
                            &call.args[0],
                            "kernel reduction axes require an array of integer literals",
                        ));
                    };
                    let axes = axes
                        .elems
                        .iter()
                        .map(|axis| {
                            let Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(axis), .. }) = axis else {
                                return Err(syn::Error::new_spanned(
                                    axis,
                                    "kernel reduction axes require integer literals",
                                ));
                            };
                            axis.base10_parse::<usize>()
                        })
                        .collect::<syn::Result<Vec<_>>>()?;
                    Ok(quote_spanned!(span=> __kernel_context.bind(
                        #core::arrays::ArrayIrOperation::from(#core::arrays::ArrayOperation::Reduce(
                            #core::operations::ReduceOperation::new(
                                vec![#(#axes),*], #core::operations::ReductionKind::Sum,
                            ),
                        )),
                        vec![], &[#receiver],
                    )?.remove(0)))
                }
                _ => Err(syn::Error::new_spanned(call, "unsupported kernel capability call")),
            }
        }
        Expr::Call(call) => {
            let Expr::Path(path) = call.func.as_ref() else {
                return Err(syn::Error::new_spanned(call, "kernel helpers must be recognized named capabilities"));
            };
            let Some(segment) = path.path.segments.last() else { unreachable!() };
            if segment.ident != "zeros" || call.args.len() != 1 {
                return Err(syn::Error::new_spanned(call, "unsupported kernel helper call"));
            }
            let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
                return Err(syn::Error::new_spanned(call, "kernel zeros requires one explicit element type"));
            };
            if arguments.args.len() != 1 {
                return Err(syn::Error::new_spanned(arguments, "kernel zeros requires one element type"));
            }
            let element_type = &arguments.args[0];
            let shape = shape_metadata(&call.args[0], inputs, core, true)?;
            Ok(quote_spanned!(span=> #core::kernels::zeros::<#element_type, _>(&__kernel_context, #shape)?))
        }
        Expr::Binary(binary) => {
            let left = value(&binary.left, core, inputs, metadata)?;
            let right = value(&binary.right, core, inputs, metadata)?;
            let (variant, operation) = match binary.op {
                BinOp::Add(_) => (format_ident!("Add"), format_ident!("AddOperation")),
                BinOp::Sub(_) => (format_ident!("Sub"), format_ident!("SubOperation")),
                BinOp::Mul(_) => (format_ident!("Mul"), format_ident!("MulOperation")),
                BinOp::Div(_) => (format_ident!("Div"), format_ident!("DivOperation")),
                _ => {
                    return Err(syn::Error::new_spanned(&binary.op, "unsupported kernel arithmetic operator"));
                }
            };
            Ok(quote_spanned!(span=> __kernel_context.bind(
                #core::arrays::ArrayIrOperation::from(#core::arrays::ArrayOperation::#variant(
                    #core::operations::#operation::new(),
                )),
                vec![], &[#left, #right],
            )?.remove(0)))
        }
        Expr::Lit(literal) => Ok(quote_spanned!(span=> __kernel_context.lift(#core::arrays::ArrayIrValue::Array(
                #core::arrays::Array::scalar(#literal)?,
            ))?)),
        _ => Err(syn::Error::new_spanned(
            expression,
            "unsupported kernel expression; control flow requires staged lowering and cannot execute as host Rust",
        )),
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_expand() {
        let expanded = expand(
            quote!(crate = "core_alias", requires = left.shape()[0] == right.shape()[0]),
            quote! {
                pub fn add(
                    #[input(data_type = F32, rank = 1)] left: &Array,
                    #[input(data_type = F32, rank = 1)] right: &Array,
                    #[output(data_type = F32, shape = [left.shape()[0]])] output: &mut Array,
                ) { output.store(left.load() + right.load()); }
            },
        )
        .unwrap();
        let file: syn::File = syn::parse2(expanded).unwrap();
        assert_eq!(file.items.len(), 2);
        let syn::Item::Fn(function) = &file.items[0] else {
            panic!("expected generated callable");
        };
        assert_eq!(function.sig.ident, "add");
        assert_eq!(function.sig.inputs.len(), 2);
        let syn::Item::Mod(module) = &file.items[1] else {
            panic!("expected definition module");
        };
        assert_eq!(module.ident, "add");
        let syn::Item::Fn(definition) = &module.content.as_ref().unwrap().1[0] else {
            panic!("expected definition constructor");
        };
        assert_eq!(definition.sig.ident, "definition");
        assert_eq!(definition.sig.inputs.len(), 2);
    }

    #[test]
    fn test_expand_control_flow() {
        let expanded = expand(
            quote!(),
            quote! {
                fn accumulate(
                    #[input(data_type = Boolean, rank = 0)] flag: &Array,
                    #[input(data_type = F32, rank = 1)] input: &Array,
                    #[output(data_type = F32, shape = [input.shape()[0]])] output: &mut Array,
                ) {
                    let mut accumulator = zeros::<f32>([input.shape()[0]]);
                    for depth in 0..input.shape()[0].div_ceil(2) {
                        if flag.load() { accumulator += input.load(); }
                        else { accumulator -= input.load(); }
                    }
                    output.store(accumulator);
                }
            },
        )
        .unwrap();
        let file: syn::File = syn::parse2(expanded).unwrap();
        assert_eq!(file.items.len(), 2);
    }

    #[test]
    fn test_expand_tiled_matmul() {
        let expanded = expand(
            quote!(requires = left.shape()[1] == right.shape()[0]),
            quote! {
                fn matmul(
                    #[input(data_type = F32, rank = 2)] left: &Array,
                    #[input(data_type = F32, rank = 2)] right: &Array,
                    #[output(data_type = F32, shape = [left.shape()[0], right.shape()[1]],
                        tile = [32, 32], boundary = masked)] output: &mut Array,
                ) {
                    let [row, column] = output.tile_index();
                    let left_tiles = left.tiles([32, 32]).pad(0.0);
                    let right_tiles = right.tiles([32, 32]).pad(0.0);
                    let mut accumulator = zeros::<f32>([32, 32]);
                    for depth in 0..left.shape()[1].div_ceil(32) {
                        accumulator += left_tiles.load([row, depth]).dot(right_tiles.load([depth, column]));
                    }
                    output.store(accumulator);
                }
            },
        )
        .unwrap();
        let file: syn::File = syn::parse2(expanded).unwrap();
        assert_eq!(file.items.len(), 2);
    }

    #[test]
    fn test_expand_rejects_host_control_flow() {
        let error = expand(
            quote!(),
            quote! {
                fn invalid(#[output(data_type = F32, shape = [])] output: &mut Array) {
                    loop { output.store(0.0f32); }
                }
            },
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "unsupported kernel expression; control flow requires staged lowering and cannot execute as host Rust"
        );
    }

    #[test]
    fn test_expand_rejects_body_attributes() {
        for statement in [
            quote!(#[cfg(any())] let value = 1.0f32;),
            quote!(#[cfg(any())] output.store(1.0f32);),
            quote!(for index in 0..1 {
                #[cfg(any())]
                output.store(1.0f32);
            }),
        ] {
            let error = expand(
                quote!(),
                quote! {
                    fn invalid(#[output(data_type = F32, shape = [])] output: &mut Array) {
                        #statement
                    }
                },
            )
            .unwrap_err();
            assert_eq!(error.to_string(), "kernel body attributes are unsupported");
        }
    }

    #[test]
    fn test_expand_rejects_generic_array() {
        let error = expand(
            quote!(),
            quote! {
                fn invalid(#[output(data_type = F32, shape = [])] output: &mut Array<f32>) {}
            },
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "kernel parameters must use the canonical non-generic `Array`");
    }

    #[test]
    fn test_expand_rejects_generic_metadata_calls() {
        let error = expand(
            quote!(),
            quote! {
                fn invalid(
                    #[input(data_type = F32, rank = 1)] input: &Array,
                    #[output(data_type = F32, shape = [input.shape::<usize>()[0]])] output: &mut Array,
                ) {}
            },
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "unsupported shape metadata call");
    }

    #[test]
    fn test_expand_rejects_shadowed_tile_metadata() {
        let error = expand(
            quote!(),
            quote! {
                fn invalid(
                    #[input(data_type = F32, rank = 1)] input: &Array,
                    #[output(data_type = F32, shape = [1])] output: &mut Array,
                ) {
                    let tiles = input.tiles([1]).pad(0.0);
                    let tiles = input.tiles([2]).pad(1.0);
                }
            },
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "kernel locals cannot shadow an existing binding");
    }

    #[test]
    fn test_expand_rejects_host_metadata() {
        let error = expand(
            quote!(),
            quote! {
                fn invalid(#[output(data_type = F32, shape = host_callback())] output: &mut Array) {}
            },
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "unsupported shape metadata expression");
    }

    #[test]
    fn test_expand_rejects_dynamic_reduction_axes() {
        let error = expand(
            quote!(),
            quote! {
                fn invalid(
                    #[input(data_type = F32, rank = 1)] input: &Array,
                    #[output(data_type = F32, shape = [])] output: &mut Array,
                ) { output.store(input.load().sum([input.shape()[0]])); }
            },
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "kernel reduction axes require integer literals");
    }

    #[test]
    fn test_shape_metadata_group() {
        let group = proc_macro2::Group::new(proc_macro2::Delimiter::None, quote!(32));
        let expression: Expr = syn::parse2(TokenStream::from(proc_macro2::TokenTree::Group(group))).unwrap();
        assert_eq!(
            shape_metadata(&expression, &[], &syn::parse_quote!(::ryft_core), true).unwrap().to_string(),
            "(32)"
        );
        let group = proc_macro2::Group::new(proc_macro2::Delimiter::None, quote!(host_callback()));
        let expression: Expr = syn::parse2(TokenStream::from(proc_macro2::TokenTree::Group(group))).unwrap();
        assert_eq!(
            shape_metadata(&expression, &[], &syn::parse_quote!(::ryft_core), true).unwrap_err().to_string(),
            "unsupported shape metadata expression"
        );
    }
}
