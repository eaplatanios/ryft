use proc_macro2::{Literal, Span, TokenStream};
use quote::{format_ident, quote, quote_spanned};
use syn::parse::Parser;
use syn::spanned::Spanned;
use syn::visit::Visit;

use crate::helpers::attributes::Attribute;
use crate::helpers::symbols::Symbol;

// TODO(eaplatanios): Review from here onwards.

const CRATE_ARGUMENT: Symbol = Symbol::new("crate");
const REQUIRES_ARGUMENT: Symbol = Symbol::new("requires");
const GRID_ARGUMENT: Symbol = Symbol::new("grid");

const INPUT_ANNOTATION: Symbol = Symbol::new("input");
const OUTPUT_ANNOTATION: Symbol = Symbol::new("output");
const INOUT_ANNOTATION: Symbol = Symbol::new("inout");
const PREFETCH_ANNOTATION: Symbol = Symbol::new("prefetch");

const DATA_TYPE_METADATA: Symbol = Symbol::new("data_type");
const RANK_METADATA: Symbol = Symbol::new("rank");
const SHAPE_METADATA: Symbol = Symbol::new("shape");
const TILE_METADATA: Symbol = Symbol::new("tile");
const BOUNDARY_METADATA: Symbol = Symbol::new("boundary");
const BLOCK_METADATA: Symbol = Symbol::new("block");
const INDEX_METADATA: Symbol = Symbol::new("index");
const START_METADATA: Symbol = Symbol::new("start");

/// Prefix of the identifiers that generated code declares, which kernel functions cannot use for their own bindings.
const RESERVED_PREFIX: &str = "__kernel_";

/// `DataType` variants that have a Rust scalar element type, each paired with that type.
const ELEMENT_TYPES: [(&str, &str); 11] = [
    ("F32", "f32"),
    ("F64", "f64"),
    ("I8", "i8"),
    ("I16", "i16"),
    ("I32", "i32"),
    ("I64", "i64"),
    ("U8", "u8"),
    ("U16", "u16"),
    ("U32", "u32"),
    ("U64", "u64"),
    ("Boolean", "bool"),
];

/// Elementwise unary capability functions of kernel body values (e.g., `value.exp()`), each paired with the
/// `ArrayOperation` variant that stages it.
const UNARY_FUNCTIONS: [(&str, &str); 17] = [
    ("abs", "Abs"),
    ("ceil", "Ceil"),
    ("cos", "Cos"),
    ("erf", "Erf"),
    ("exp", "Exp"),
    ("exp_m1", "ExpM1"),
    ("floor", "Floor"),
    ("ln_1p", "Ln1p"),
    ("log", "Log"),
    ("logistic", "Logistic"),
    ("round", "Round"),
    ("rsqrt", "Rsqrt"),
    ("sign", "Sign"),
    ("sin", "Sin"),
    ("sqrt", "Sqrt"),
    ("tan", "Tan"),
    ("tanh", "Tanh"),
];

/// Elementwise binary capability functions of kernel body values (e.g., `left.max(right)`), each paired with the
/// `ArrayOperation` variant that stages it.
const BINARY_FUNCTIONS: [(&str, &str); 5] =
    [("atan2", "Atan2"), ("log_add_exp", "LogAddExp"), ("max", "Max"), ("min", "Min"), ("pow", "Pow")];

/// Reduction capability functions of kernel body values (e.g., `value.reduce_max([0])`), each paired with the
/// `ReductionKind` variant that they stage. `sum` is the shorter spelling of `reduce_sum`.
const REDUCTION_FUNCTIONS: [(&str, &str); 9] = [
    ("sum", "Sum"),
    ("reduce_sum", "Sum"),
    ("reduce_product", "Product"),
    ("reduce_mean", "Mean"),
    ("reduce_log_sum_exp", "LogSumExp"),
    ("reduce_max", "Max"),
    ("reduce_min", "Min"),
    ("reduce_any", "Any"),
    ("reduce_all", "All"),
];

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

    /// Constraint over the shapes of the arguments, provided through the `requires = ...` argument.
    requirement: Option<syn::Expr>,

    /// [`GridAxis`]es declared through the `grid(...)` argument, in order, which is [`None`] when the kernel declares
    /// no grid.
    grid: Option<Vec<GridAxis>>,

    /// [`Parameter`]s of the kernel, in declaration order.
    parameters: Vec<Parameter>,
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
    /// the arguments are elided, and `scope(...)` stands for the provenance scope of the source location of each
    /// lowered expression:
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
    ///             [ryft::kernels::__private::static_extent(input, 0usize)?],
    ///         );
    ///         let __kernel_operation = { /* Whole-array parameters on a rank-0 grid. */ };
    ///         let __kernel_input_metadata = input;
    ///         ryft::kernels::KernelDefinition::trace_with_prefetch(
    ///             __kernel_operation,
    ///             |(__kernel_references, __kernel_coordinates, __kernel_prefetched)| {
    ///                 use core::contexts::Context as _;
    ///                 let __kernel_context = __kernel_references[0].context().clone();
    ///                 let input = __kernel_references[0usize].clone();
    ///                 let output = __kernel_references[1usize].clone();
    ///                 __kernel_context.invoke_with_provenance_scope(scope(...), || {
    ///                     Ok(core::operations::ReferenceWrite::write(&(output.clone()), &(/* input.load() */))?)
    ///                 })?;
    ///                 ::core::result::Result::Ok(())
    ///             },
    ///         )
    ///     }
    /// }
    /// ```
    ///
    /// A kernel with a `grid(...)` argument, a block window, or a `#[prefetch]` parameter constructs its call with
    /// `mapped_call` instead and binds each grid axis name to `__kernel_coordinates[axis]` and each prefetch parameter
    /// to `__kernel_prefetched[position]` in the traced body. A callable with several results returns them as a tuple.
    pub(crate) fn generate_kernel_impl(attributes: TokenStream, item: TokenStream) -> TokenStream {
        Self::generate(attributes, item).unwrap_or_else(syn::Error::into_compile_error)
    }

    /// Generates the `#[kernel]` expansion, returning the first error that is encountered. Refer to
    /// [`KernelCodeGenerator::generate_kernel_impl`] for more information.
    fn generate(attributes: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
        let function = syn::parse2::<syn::ItemFn>(item)?;
        let (kernels, requirement, grid) = Self::extract_arguments(attributes)?;
        Self::validate_signature(&function.sig)?;
        Self::validate_body(&function.block)?;
        let parameters = Self::extract_parameters(&function.sig)?;
        Self::validate_grid(grid.as_deref(), &parameters)?;
        let generator = Self::new(kernels, requirement, grid, parameters);
        let callable = generator.generate_callable(&function);
        let definition = generator.generate_definition(&function)?;
        Ok(quote!(#callable #definition))
    }

    /// Creates a new [`KernelCodeGenerator`] for the provided kernel namespace, shape requirement, grid, and
    /// parameters.
    fn new(
        kernels: syn::Path,
        requirement: Option<syn::Expr>,
        grid: Option<Vec<GridAxis>>,
        parameters: Vec<Parameter>,
    ) -> Self {
        let core = quote!(#kernels::__private::core);
        Self { kernels, core, requirement, grid, parameters }
    }

    /// Extracts the kernel namespace path from the `crate = "..."` argument (defaulting to `ryft::kernels`), the shape
    /// requirement from the `requires = ...` argument, and the [`GridAxis`]es from the `grid(...)` argument of the
    /// attribute. Each grid axis is written as `name = extent`, `name = parallel(extent)`, or
    /// `name = sequential(extent)`, where `extent` is shape metadata, and grid axis names must be unique.
    fn extract_arguments(
        attributes: TokenStream,
    ) -> syn::Result<(syn::Path, Option<syn::Expr>, Option<Vec<GridAxis>>)> {
        let mut kernels = Attribute::<syn::Path>::new(CRATE_ARGUMENT);
        let mut requirement = Attribute::<syn::Expr>::new(REQUIRES_ARGUMENT);
        let mut grid = None::<Vec<GridAxis>>;
        let parser = syn::meta::parser(|meta| match &meta.path {
            path if path == &CRATE_ARGUMENT => kernels.set(&meta),
            path if path == &REQUIRES_ARGUMENT => requirement.set(&meta),
            path if path == &GRID_ARGUMENT => {
                if grid.is_some() {
                    return Err(syn::Error::new_spanned(path, "duplicate ryft attribute `grid`"));
                }
                let mut axes = Vec::<GridAxis>::new();
                meta.parse_nested_meta(|axis| {
                    let Some(name) = axis.path.get_ident().cloned() else {
                        return Err(axis.error("kernel grid axes require a name"));
                    };
                    if name.to_string().starts_with(RESERVED_PREFIX) {
                        return Err(syn::Error::new_spanned(
                            &name,
                            "kernel grid axis names cannot use the reserved `__kernel_` prefix",
                        ));
                    }
                    if axes.iter().any(|existing| existing.name == name) {
                        return Err(syn::Error::new_spanned(&name, "kernel grid axis names must be unique"));
                    }
                    let extent = axis.value()?.parse::<syn::Expr>()?;
                    let (sequential, extent) = match ungroup(&extent) {
                        syn::Expr::Call(call) if call.args.len() == 1 => match identifier(&call.func) {
                            Some(execution) if execution == "sequential" => (true, call.args[0].clone()),
                            Some(execution) if execution == "parallel" => (false, call.args[0].clone()),
                            _ => (false, extent),
                        },
                        _ => (false, extent),
                    };
                    axes.push(GridAxis { name, extent, sequential });
                    Ok(())
                })?;
                grid = Some(axes);
                Ok(())
            }
            _ => Err(meta
                .error("unsupported `kernel` argument (expected `crate = \"...\"`, `requires = ...`, or `grid(...)`)")),
        });
        parser.parse2(attributes)?;
        Ok((kernels.get().unwrap_or_else(|| syn::parse_quote!(ryft::kernels)), requirement.get(), grid))
    }

    /// Validates that the signature of the kernel function is safe, synchronous, and non-generic, and that the function
    /// returns nothing, because its results are written through its `output` and `inout` parameters.
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

    /// Extracts the [`Parameter`]s of the kernel from the parameters of its function, which must have unique names and
    /// include at least one result (i.e., an `output` or `inout` parameter).
    fn extract_parameters(signature: &syn::Signature) -> syn::Result<Vec<Parameter>> {
        let mut parameters = Vec::<Parameter>::new();
        for argument in &signature.inputs {
            let syn::FnArg::Typed(argument) = argument else {
                return Err(syn::Error::new_spanned(argument, "kernel functions cannot take `self`"));
            };
            let parameter = Self::extract_parameter(argument)?;
            if parameters.iter().any(|existing| existing.name == parameter.name) {
                return Err(syn::Error::new_spanned(&parameter.name, "kernel parameter names must be unique"));
            }
            parameters.push(parameter);
        }
        if !parameters.iter().any(Parameter::is_result) {
            return Err(syn::Error::new_spanned(
                signature,
                "kernels require at least one `output` or `inout` parameter",
            ));
        }
        Ok(parameters)
    }

    /// Extracts one kernel [`Parameter`] from a parameter of the kernel function, which must be a named `&Array` with
    /// an `#[input(...)]` or `#[prefetch(...)]` annotation, or a named `&mut Array` with an `#[output(...)]` or
    /// `#[inout(...)]` annotation.
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
                "each kernel parameter requires exactly one `input`, `output`, `inout`, or `prefetch` annotation",
            ));
        };
        let access = match annotation.path() {
            path if path == &INPUT_ANNOTATION => Access::Input,
            path if path == &OUTPUT_ANNOTATION => Access::Output,
            path if path == &INOUT_ANNOTATION => Access::Inout,
            path if path == &PREFETCH_ANNOTATION => Access::Prefetch,
            _ => {
                return Err(syn::Error::new_spanned(
                    annotation,
                    "expected an `input`, `output`, `inout`, or `prefetch` annotation",
                ));
            }
        };
        let is_mutable = matches!(access, Access::Output | Access::Inout);
        if is_mutable != reference.mutability.is_some() {
            return Err(syn::Error::new_spanned(
                reference,
                "kernel `input` and `prefetch` parameters must use `&Array`, and kernel `output` and `inout` \
                parameters must use `&mut Array`",
            ));
        }

        let is_output = access == Access::Output;
        let mut data_type = Attribute::<syn::Expr>::new(DATA_TYPE_METADATA);
        let mut rank = Attribute::<syn::Expr>::new(RANK_METADATA);
        let mut shape = Attribute::<syn::Expr>::new(SHAPE_METADATA);
        let mut tile = Attribute::<syn::Expr>::new(TILE_METADATA);
        let mut boundary = Attribute::<syn::Expr>::new(BOUNDARY_METADATA);
        let mut block = Attribute::<syn::Expr>::new(BLOCK_METADATA);
        let mut index = Attribute::<syn::Expr>::new(INDEX_METADATA);
        let mut start = Attribute::<syn::Expr>::new(START_METADATA);
        annotation.parse_nested_meta(|meta| match &meta.path {
            path if path == &DATA_TYPE_METADATA => data_type.set(&meta),
            _ if access == Access::Prefetch => {
                Err(meta.error("kernel `prefetch` parameters only take `data_type` metadata"))
            }
            path if path == &RANK_METADATA && !is_output => rank.set(&meta),
            path if path == &SHAPE_METADATA && is_output => shape.set(&meta),
            path if path == &TILE_METADATA && is_output => tile.set(&meta),
            path if path == &BOUNDARY_METADATA => boundary.set(&meta),
            path if path == &BLOCK_METADATA => block.set(&meta),
            path if path == &INDEX_METADATA => index.set(&meta),
            path if path == &START_METADATA => start.set(&meta),
            path if path == &RANK_METADATA => Err(meta.error("kernel outputs take their rank from their `shape`")),
            path if path == &SHAPE_METADATA || path == &TILE_METADATA => {
                Err(meta.error("`shape` and `tile` metadata are only supported for kernel outputs"))
            }
            _ => Err(meta.error("unsupported kernel parameter metadata")),
        })?;

        let name = pattern.ident.clone();
        let data_type = data_type
            .get()
            .ok_or_else(|| syn::Error::new_spanned(annotation, "kernel parameters require `data_type` metadata"))?;
        let data_type = identifier(&data_type)
            .ok_or_else(|| syn::Error::new_spanned(&data_type, "kernel data types must be `DataType` variants"))?;
        let shape = match access {
            Access::Output => ShapeAnnotation::Shape(
                shape
                    .get()
                    .ok_or_else(|| syn::Error::new_spanned(annotation, "kernel outputs require `shape` metadata"))?,
            ),
            Access::Prefetch => ShapeAnnotation::Rank(0),
            Access::Input | Access::Inout => {
                let rank = rank.get().ok_or_else(|| {
                    syn::Error::new_spanned(annotation, "kernel `input` and `inout` parameters require `rank` metadata")
                })?;
                let syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(rank), .. }) = ungroup(&rank) else {
                    return Err(syn::Error::new_spanned(rank, "kernel parameter ranks must be integer literals"));
                };
                ShapeAnnotation::Rank(rank.base10_parse()?)
            }
        };
        let window =
            Self::extract_window(annotation, tile.get(), block.get(), index.get(), start.get(), boundary.get())?;
        Ok(Parameter { name, access, data_type, shape, window })
    }

    /// Extracts the [`Window`] of a kernel parameter from the `tile`, `block`, `index`, `start`, and `boundary`
    /// metadata of its `annotation`. A `tile` requires an explicit `boundary` and excludes the other metadata. A
    /// `block` array requires exactly one of the `index` and `start` arrays, whose length must equal that of the
    /// `block` array, and `boundary` defaults to `in_bounds`. Without either, the window is the whole array.
    fn extract_window(
        annotation: &syn::Attribute,
        tile: Option<syn::Expr>,
        block: Option<syn::Expr>,
        index: Option<syn::Expr>,
        start: Option<syn::Expr>,
        boundary: Option<syn::Expr>,
    ) -> syn::Result<Window> {
        let boundary = boundary
            .map(|boundary| match identifier(&boundary) {
                Some(policy) if policy == "masked" => Ok(Boundary::Masked),
                Some(policy) if policy == "in_bounds" => Ok(Boundary::InBounds),
                _ => Err(syn::Error::new_spanned(
                    boundary,
                    "unsupported kernel boundary policy (expected `masked` or `in_bounds`)",
                )),
            })
            .transpose()?;
        if let Some(tile) = tile {
            if let Some(expression) = block.as_ref().or(index.as_ref()).or(start.as_ref()) {
                return Err(syn::Error::new_spanned(
                    expression,
                    "output `tile` metadata cannot be combined with `block`, `index`, or `start` metadata",
                ));
            }
            let Some(boundary) = boundary else {
                return Err(syn::Error::new_spanned(
                    annotation,
                    "output tiling requires explicit `tile` and `boundary` metadata",
                ));
            };
            return Ok(Window::Tiled(Tiling { tile, boundary }));
        }
        let Some(block) = block else {
            return match index.as_ref().or(start.as_ref()) {
                Some(expression) => Err(syn::Error::new_spanned(
                    expression,
                    "`index`, `start`, and `boundary` metadata require `block` metadata",
                )),
                None if boundary.is_some() => Err(syn::Error::new_spanned(
                    annotation,
                    "`index`, `start`, and `boundary` metadata require `block` metadata",
                )),
                None => Ok(Window::WholeArray),
            };
        };
        let syn::Expr::Array(extents) = ungroup(&block) else {
            return Err(syn::Error::new_spanned(block, "kernel `block` metadata requires an array"));
        };
        let (mapping, starts) = match (index, start) {
            (Some(_), Some(start)) => {
                return Err(syn::Error::new_spanned(start, "kernel windows take either `index` or `start` metadata"));
            }
            (Some(index), None) => (index, false),
            (None, Some(start)) => (start, true),
            (None, None) => {
                return Err(syn::Error::new_spanned(
                    block,
                    "kernel `block` metadata requires `index` or `start` metadata",
                ));
            }
        };
        let syn::Expr::Array(expressions) = ungroup(&mapping) else {
            return Err(syn::Error::new_spanned(mapping, "kernel `index` and `start` metadata require arrays"));
        };
        if expressions.elems.len() != extents.elems.len() {
            return Err(syn::Error::new_spanned(
                &mapping,
                format!(
                    "kernel `{}` metadata requires one expression per `block` extent",
                    if starts { "start" } else { "index" },
                ),
            ));
        }
        Ok(Window::Block(Block {
            shape: block.clone(),
            boundary: boundary.unwrap_or(Boundary::InBounds),
            expressions: expressions.elems.iter().cloned().collect(),
            starts,
        }))
    }

    /// Validates the combination of the grid and the parameters of a kernel. Grid axis names cannot collide with
    /// parameter names, because both are bound in the kernel body, and an output `tile` is the shorthand for a parallel
    /// grid that is inferred from the single result of the kernel, so it cannot be combined with a `grid(...)`
    /// argument, other results, `prefetch` parameters, or input windows.
    fn validate_grid(grid: Option<&[GridAxis]>, parameters: &[Parameter]) -> syn::Result<()> {
        for axis in grid.unwrap_or_default() {
            if parameters.iter().any(|parameter| parameter.name == axis.name) {
                return Err(syn::Error::new_spanned(
                    &axis.name,
                    "kernel grid axis names cannot collide with parameter names",
                ));
            }
        }
        let tiling = parameters.iter().find_map(|parameter| match &parameter.window {
            Window::Tiled(tiling) => Some(tiling),
            _ => None,
        });
        if let Some(tiling) = tiling {
            if grid.is_some() {
                return Err(syn::Error::new_spanned(
                    &tiling.tile,
                    "`grid` cannot be combined with output `tile` metadata",
                ));
            }
            let results = parameters.iter().filter(|parameter| parameter.is_result()).count();
            let other = parameters.iter().any(|parameter| {
                parameter.access == Access::Prefetch
                    || (!parameter.is_result() && !matches!(parameter.window, Window::WholeArray))
            });
            if results != 1 || other {
                return Err(syn::Error::new_spanned(
                    &tiling.tile,
                    "output `tile` metadata requires a single result, whole array inputs, and no `prefetch` parameters",
                ));
            }
        }
        Ok(())
    }

    /// Returns the [`Parameter`]s that are arguments of the kernel callable and of the `definition` function (i.e.,
    /// every parameter except the `output` parameters), in declaration order.
    fn arguments(&self) -> impl Iterator<Item = &Parameter> {
        self.parameters.iter().filter(|parameter| parameter.access != Access::Output)
    }

    /// Returns the [`Parameter`]s that are references of the kernel body (i.e., every parameter except the `prefetch`
    /// parameters), in declaration order.
    fn references(&self) -> impl Iterator<Item = &Parameter> {
        self.parameters.iter().filter(|parameter| parameter.access != Access::Prefetch)
    }

    /// Returns the `prefetch` [`Parameter`]s, in declaration order.
    fn prefetches(&self) -> impl Iterator<Item = &Parameter> {
        self.parameters.iter().filter(|parameter| parameter.access == Access::Prefetch)
    }

    /// Returns the [`GridAxis`]es of the kernel, which are empty when it declares no grid.
    fn axes(&self) -> &[GridAxis] {
        self.grid.as_deref().unwrap_or_default()
    }

    /// Returns the reference [`Parameter`] named `name`, if there is one.
    fn reference(&self, name: &syn::Ident) -> Option<&Parameter> {
        self.references().find(|parameter| &parameter.name == name)
    }

    /// Returns the output [`Parameter`] whose window is tiled through the `tile` shorthand, if there is one.
    fn tiled_output(&self) -> Option<&Parameter> {
        self.parameters.iter().find(|parameter| matches!(parameter.window, Window::Tiled(_)))
    }

    /// Generates the functional callable of the kernel, which keeps the name, visibility, and attributes of the kernel
    /// function, takes the kernel arguments as values of any `KernelCall` type, builds the kernel definition for their
    /// types, and invokes or stages it through `KernelCall::call_kernel`. The call receives the `input` and `inout`
    /// arguments in declaration order followed by the `prefetch` arguments. The callable returns its one result
    /// directly and several results as a tuple, in the declaration order of the `output` and `inout` parameters. It
    /// allows the Clippy lint that passing a single cloned argument triggers.
    fn generate_callable(&self, function: &syn::ItemFn) -> TokenStream {
        let Self { kernels, core, .. } = self;
        // The lint attribute of the generated function goes below its documentation and above its other attributes.
        let (documentation, attributes): (Vec<_>, Vec<_>) =
            function.attrs.iter().partition(|attribute| attribute.path().is_ident("doc"));
        let visibility = &function.vis;
        let name = &function.sig.ident;
        let arguments = self.arguments().map(|parameter| &parameter.name).collect::<Vec<_>>();
        let types = arguments.iter().map(|argument| format_ident!("__kernel_{}_type", argument)).collect::<Vec<_>>();
        let inputs = self
            .arguments()
            .filter(|parameter| parameter.access != Access::Prefetch)
            .chain(self.prefetches())
            .map(|parameter| &parameter.name);
        let count = self.parameters.iter().filter(|parameter| parameter.is_result()).count();
        let (result, results) = if count == 1 {
            (quote!(__KernelValue), quote!(__kernel_outputs.remove(0)))
        } else {
            let results = (0..count).map(|_| quote!(__kernel_outputs.remove(0)));
            let result = (0..count).map(|_| quote!(__KernelValue));
            (quote!((#(#result),*)), quote!((#(#results),*)))
        };
        let count = Literal::usize_unsuffixed(count);
        quote! {
            #(#documentation)*
            #[allow(clippy::cloned_ref_to_slice_refs)]
            #(#attributes)*
            #visibility fn #name<__KernelValue: #kernels::KernelCall>(#(#arguments: &__KernelValue),*)
                -> ::core::result::Result<#result, #core::programs::ProgramError>
            {
                #(let #types = #core::programs::Typed::r#type(#arguments);)*
                let __kernel_definition = #name::definition(#(#types.as_ref()),*)
                    .map_err(#core::programs::ProgramError::custom)?;
                let mut __kernel_outputs = <__KernelValue as #kernels::KernelCall>::call_kernel(
                    &__kernel_definition,
                    &[#(#inputs.clone()),*],
                )?;
                if __kernel_outputs.len() != #count {
                    return ::core::result::Result::Err(#core::programs::ProgramError::InvalidOutputCount {
                        expected: #count,
                        actual: __kernel_outputs.len(),
                    });
                }
                ::core::result::Result::Ok(#results)
            }
        }
    }

    /// Generates the definition module of the kernel, which has the name and visibility of the kernel function and the
    /// `#[cfg]` attributes that select it. Its `definition` function validates the argument types against their
    /// annotations and the shape requirement, builds the kernel call operation for the grid and the parameter
    /// annotations, and traces the lowered kernel body, in which the parameter names are bound to their references
    /// and prefetched values and the grid axis names to the grid coordinates. The module imports the items of its
    /// parent, so that paths in the kernel (e.g., a relative kernel namespace path or an element type in `zeros::<T>`)
    /// resolve as they would next to the function. The `definition` function allows the Clippy lints that the
    /// uniform lowering triggers (e.g., `Ok(value?)` in the provenance scopes of fallible expressions).
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
        let arguments = self.arguments().map(|parameter| &parameter.name).collect::<Vec<_>>();
        let metadata =
            arguments.iter().map(|argument| format_ident!("__kernel_{}_metadata", argument)).collect::<Vec<_>>();
        let checks = self.arguments().map(|Parameter { name, access, data_type, shape, .. }| {
            let ShapeAnnotation::Rank(rank) = shape else { unreachable!() };
            let annotation = match access {
                Access::Prefetch => "prefetch",
                Access::Inout => "inout",
                _ => "input",
            };
            let message = format!("kernel {annotation} `{name}` requires data type `{data_type}` and rank `{rank}`");
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
        let output_types = self
            .parameters
            .iter()
            .filter_map(|parameter| match &parameter.shape {
                ShapeAnnotation::Shape(shape) => Some((parameter, shape)),
                ShapeAnnotation::Rank(_) => None,
            })
            .map(|(Parameter { name, data_type, .. }, shape)| {
                let r#type = format_ident!("__kernel_{}_type", name);
                let shape = self.generate_shape(shape, false)?;
                Ok(quote! {
                    let #r#type = #core::arrays::ArrayType::new_static(#core::arrays::DataType::#data_type, #shape);
                })
            })
            .collect::<syn::Result<Vec<_>>>()?;
        let operation = self.generate_operation()?;
        let references = self.references().map(|parameter| &parameter.name).collect::<Vec<_>>();
        let reference_indices = 0..references.len();
        let axes = self.axes().iter().map(|axis| &axis.name).collect::<Vec<_>>();
        let axis_indices = 0..axes.len();
        let prefetches = self.prefetches().map(|parameter| &parameter.name).collect::<Vec<_>>();
        let prefetch_indices = 0..prefetches.len();
        let mut scope = Scope::new(
            references
                .iter()
                .chain(&axes)
                .chain(&prefetches)
                .map(|&name| Binding { name: name.clone(), mutable: false })
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
                pub fn definition(#(#arguments: &#core::arrays::ArrayType),*)
                    -> ::core::result::Result<#kernels::KernelDefinition, #kernels::KernelError>
                {
                    #(#checks)*
                    #requirement
                    #(#output_types)*
                    let __kernel_operation = #operation;
                    #(let #metadata = #arguments;)*
                    #kernels::KernelDefinition::trace_with_prefetch(
                        __kernel_operation,
                        |(__kernel_references, __kernel_coordinates, __kernel_prefetched)| {
                            use #core::contexts::Context as _;
                            let __kernel_context = __kernel_references[0].context().clone();
                            #(let #references = __kernel_references[#reference_indices].clone();)*
                            #(#[allow(unused_variables)] let #axes = __kernel_coordinates[#axis_indices].clone();)*
                            #(
                                #[allow(unused_variables)]
                                let #prefetches = __kernel_prefetched[#prefetch_indices].clone();
                            )*
                            #body
                            ::core::result::Result::Ok(())
                        },
                    )
                }
            }
        })
    }

    /// Generates the kernel call operation. A tiled output uses the `tiled_call` shorthand (i.e., a parallel grid
    /// over the output tiles with whole array inputs), and a kernel without a grid, block windows, and `prefetch`
    /// parameters runs a single program over whole array parameters. Every other kernel constructs its call with
    /// `mapped_call` over its grid, with one window per reference parameter and the types of its `prefetch`
    /// parameters. The generated expression uses the `definition` function parameters and the
    /// `__kernel_<output>_type` bindings.
    fn generate_operation(&self) -> syn::Result<TokenStream> {
        let kernels = &self.kernels;
        let r#type = |parameter: &Parameter| {
            let name = &parameter.name;
            match parameter.access {
                Access::Output => {
                    let r#type = format_ident!("__kernel_{}_type", name);
                    quote!(#r#type)
                }
                _ => quote!(#name.clone()),
            }
        };
        if let Some(output) = self.tiled_output() {
            let Window::Tiled(Tiling { tile, boundary }) = &output.window else { unreachable!() };
            let inputs = self.arguments().map(|parameter| &parameter.name);
            let output_type = r#type(output);
            let tile = self.generate_shape(tile, false)?;
            let boundary = boundary.generate(kernels);
            return Ok(quote!(
                #kernels::__private::tiled_call(
                    &[#(#inputs.clone()),*],
                    #output_type,
                    (#tile).to_vec(),
                    #boundary,
                )?
            ));
        }
        if self.grid.is_none()
            && self.prefetches().next().is_none()
            && self.parameters.iter().all(|parameter| matches!(parameter.window, Window::WholeArray))
        {
            let parameters = self.references().map(|parameter| {
                let r#type = r#type(parameter);
                let access = parameter.access.generate(kernels);
                quote!(#kernels::whole_array_parameter(#r#type, #access)?)
            });
            return Ok(quote!({
                let __kernel_parameters = ::std::vec![#(#parameters),*];
                #kernels::KernelCallOperation::new(#kernels::Grid::new(::std::vec![])?, __kernel_parameters)?
            }));
        }
        let core = &self.core;
        let axes = self
            .axes()
            .iter()
            .map(|GridAxis { name, extent, sequential }| {
                let extent = self.generate_shape(extent, false)?;
                let execution = if *sequential { quote!(Sequential) } else { quote!(Parallel) };
                let name = name.to_string();
                Ok(quote! {
                    #kernels::GridDimension::new(
                        #core::arrays::Dimension::Static(#extent),
                        #kernels::GridExecution::#execution,
                    )
                    .with_name(#name)?
                })
            })
            .collect::<syn::Result<Vec<_>>>()?;
        let parameters = self
            .references()
            .map(|parameter| {
                let r#type = r#type(parameter);
                let access = parameter.access.generate(kernels);
                let window = match &parameter.window {
                    Window::WholeArray => quote!(#kernels::ParameterWindow::WholeArray),
                    Window::Tiled(_) => unreachable!(),
                    Window::Block(Block { shape, boundary, expressions, starts }) => {
                        let shape = self.generate_shape(shape, false)?;
                        let boundary = boundary.generate(kernels);
                        let expressions = expressions
                            .iter()
                            .map(|expression| self.generate_mapping(expression))
                            .collect::<syn::Result<Vec<_>>>()?;
                        if *starts {
                            quote! {
                                #kernels::ParameterWindow::ElementStarts {
                                    block_shape: (#shape).to_vec(),
                                    boundary: #boundary,
                                    starts: ::std::vec![#(#expressions),*],
                                }
                            }
                        } else {
                            quote! {
                                #kernels::ParameterWindow::BlockIndices {
                                    block_shape: (#shape).to_vec(),
                                    boundary: #boundary,
                                    indices: ::std::vec![#(#expressions),*],
                                }
                            }
                        }
                    }
                };
                Ok(quote!((#r#type, #access, #window)))
            })
            .collect::<syn::Result<Vec<_>>>()?;
        let prefetches = self.prefetches().map(|parameter| &parameter.name);
        Ok(quote! {
            #kernels::mapped_call(
                #kernels::Grid::new(::std::vec![#(#axes),*])?,
                ::std::vec![#(#parameters),*],
                ::std::vec![#(#prefetches.clone()),*],
            )?
        })
    }

    /// Generates the `MappingExpression` of one block index or element start of a parameter window, which consists of
    /// grid axis names, `prefetch` parameter names, integer literals, and shape metadata, combined with `+`, `-`, `*`,
    /// `/`, `%`, `min`, `max`, and `saturating_sub`. Integer literals and shape metadata become constants.
    fn generate_mapping(&self, expression: &syn::Expr) -> syn::Result<TokenStream> {
        let kernels = &self.kernels;
        match expression {
            syn::Expr::Paren(syn::ExprParen { expr, .. }) | syn::Expr::Group(syn::ExprGroup { expr, .. }) => {
                let expression = self.generate_mapping(expr)?;
                Ok(quote!((#expression)))
            }
            syn::Expr::Path(_) => {
                let Some(name) = identifier(expression) else {
                    return Err(syn::Error::new_spanned(expression, "unsupported kernel mapping expression"));
                };
                if let Some(axis) = self.axes().iter().position(|axis| axis.name == name) {
                    return Ok(quote!(#kernels::MappingExpression::GridCoordinate(#axis)));
                }
                if let Some(prefetch) = self.prefetches().position(|parameter| parameter.name == name) {
                    return Ok(quote!(#kernels::MappingExpression::Prefetch(#prefetch)));
                }
                Err(syn::Error::new_spanned(
                    expression,
                    format!("unknown grid axis or `prefetch` parameter `{name}` in kernel mapping expression"),
                ))
            }
            syn::Expr::Binary(binary) => {
                if !matches!(
                    binary.op,
                    syn::BinOp::Add(_)
                        | syn::BinOp::Sub(_)
                        | syn::BinOp::Mul(_)
                        | syn::BinOp::Div(_)
                        | syn::BinOp::Rem(_)
                ) {
                    return Err(syn::Error::new_spanned(binary.op, "unsupported kernel mapping operator"));
                }
                let left = self.generate_mapping(&binary.left)?;
                let right = self.generate_mapping(&binary.right)?;
                let operator = &binary.op;
                Ok(quote!(#left #operator #right))
            }
            syn::Expr::MethodCall(call)
                if ["min", "max", "saturating_sub"].iter().any(|method| call.method == method)
                    && call.args.len() == 1
                    && call.turbofish.is_none() =>
            {
                let receiver = self.generate_mapping(&call.receiver)?;
                let argument = self.generate_mapping(&call.args[0])?;
                let method = &call.method;
                Ok(quote!(#receiver.#method(#argument)))
            }
            _ => {
                let constant = self.generate_shape(expression, false)?;
                Ok(quote!(#kernels::MappingExpression::Constant(#constant)))
            }
        }
    }

    /// Generates the evaluation of a shape metadata expression, which consists of integer literals, arrays, argument
    /// extents written as `argument.shape()[axis]`, `extent.div_ceil(divisor)`, comparisons, and logical conjunctions
    /// and disjunctions, so that it cannot run host callbacks. Argument extents are read from the argument types, and
    /// every axis must be an integer literal below the declared rank of its argument.
    ///
    /// # Parameters
    ///
    ///   - `expression`: Shape metadata expression to evaluate.
    ///   - `traced`: Indicates whether the expression is evaluated inside the traced kernel body, where the argument
    ///     types are bound to `__kernel_<argument>_metadata` because the argument names are bound to references, and
    ///     where errors are converted into `ProgramError`s. Otherwise, the expression is evaluated in the
    ///     `definition` function.
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
                Ok(self.generate_shape_result(quote!(#kernels::__private::shape_div_ceil(#extent, #divisor)), traced))
            }
            _ => Err(syn::Error::new_spanned(expression, "unsupported shape metadata expression")),
        }
    }

    /// Generates the evaluation of an argument extent written as `argument.shape()[axis]` in shape metadata. Refer to
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
        let Some(argument) = self.arguments().find(|parameter| parameter.name == name) else {
            return Err(syn::Error::new_spanned(&call.receiver, "unknown shape metadata input"));
        };
        let ShapeAnnotation::Rank(rank) = argument.shape else { unreachable!() };
        let syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(axis), .. }) = ungroup(&index.index) else {
            return Err(syn::Error::new_spanned(&index.index, "shape axis must be an integer literal"));
        };
        let axis = axis.base10_parse::<usize>()?;
        if axis >= rank {
            return Err(syn::Error::new_spanned(
                &index.index,
                format!("shape axis `{axis}` is out of bounds for kernel input `{name}` with rank `{rank}`"),
            ));
        }
        let shape = if traced { format_ident!("__kernel_{}_metadata", name) } else { name };
        Ok(self.generate_shape_result(quote!(#kernels::__private::static_extent(#shape, #axis)), traced))
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
        if scope.region
            || identifier(&call.receiver)
                .is_none_or(|receiver| self.tiled_output().is_none_or(|output| receiver != output.name))
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

    /// Generates the lowering of a `let` statement, which either binds a staged value or declares a [`TileView`] or a
    /// [`SliceView`]. View declarations generate no code, because their accesses are lowered directly.
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
        if let syn::Expr::MethodCall(call) = ungroup(&initializer.expr) {
            if call.method == "pad" {
                let view = self.extract_tile_view(pattern, call, scope)?;
                scope.tiles.push(view);
                return Ok(TokenStream::new());
            }
            if call.method == "slice" {
                let view = self.extract_slice_view(call, scope)?;
                let name = scope.validate_new_name(pattern, "slices", false)?.clone();
                scope.slices.push((name, view));
                return Ok(TokenStream::new());
            }
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
        let input = identifier(&tiles.receiver).and_then(|name| {
            self.parameters.iter().find(|parameter| parameter.access == Access::Input && parameter.name == name)
        });
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
        if !is_scalar_literal(padding_value) {
            return Err(syn::Error::new_spanned(padding_value, "tile padding requires a scalar literal"));
        }
        let element = input.element_type()?;
        let shape = self.generate_shape(shape, true)?;
        let name = scope.validate_new_name(pattern, "tile views", false)?.clone();
        let ShapeAnnotation::Rank(rank) = input.shape else { unreachable!() };
        Ok(TileView { name, source: input.name.clone(), rank, element, shape, padding: padding_value.clone() })
    }

    /// Extracts a [`SliceView`] from `reference.slice([ranges])`, where `reference` names a binding and every range is
    /// either `start..end` with shape metadata bounds or `..`, which selects a whole axis.
    fn extract_slice_view(&self, call: &syn::ExprMethodCall, scope: &Scope) -> syn::Result<SliceView> {
        let source = identifier(&call.receiver).filter(|name| {
            scope.bindings.iter().any(|binding| &binding.name == name)
                && call.args.len() == 1
                && call.turbofish.is_none()
        });
        let Some(source) = source else {
            return Err(syn::Error::new_spanned(call, "expected `reference.slice([ranges])` with a named reference"));
        };
        let syn::Expr::Array(ranges) = ungroup(&call.args[0]) else {
            return Err(syn::Error::new_spanned(&call.args[0], "kernel slices require an array of ranges"));
        };
        let ranges = ranges
            .elems
            .iter()
            .map(|range| match ungroup(range) {
                syn::Expr::Range(syn::ExprRange {
                    start: None,
                    end: None,
                    limits: syn::RangeLimits::HalfOpen(_),
                    ..
                }) => Ok(quote!(::core::option::Option::None)),
                syn::Expr::Range(syn::ExprRange {
                    start: Some(start),
                    end: Some(end),
                    limits: syn::RangeLimits::HalfOpen(_),
                    ..
                }) => {
                    let start = self.generate_shape(start, true)?;
                    let end = self.generate_shape(end, true)?;
                    Ok(quote!(::core::option::Option::Some(#start..#end)))
                }
                _ => Err(syn::Error::new_spanned(range, "kernel slice ranges must be `start..end` or `..`")),
            })
            .collect::<syn::Result<Vec<_>>>()?;
        Ok(SliceView { source, ranges })
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
    /// the binary `operator` applied to its target and its right operand (e.g., `left = left + (right)`). The right
    /// operand is parenthesized because it is a complete expression of its own, so that an operand such as
    /// `first + second` in `left *= first + second` keeps its grouping instead of being parsed again under
    /// the precedence of `operator`.
    fn generate_compound_assignment(
        &self,
        binary: &syn::ExprBinary,
        operator: TokenStream,
        scope: &Scope,
    ) -> syn::Result<TokenStream> {
        let target = &binary.left;
        let right = &binary.right;
        let value: syn::Expr = syn::parse_quote!(#target #operator (#right));
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
    /// expressions, capability calls (e.g., `load`, `store`, `dot`, `exp`, and tile view loads), helper calls (e.g.,
    /// `zeros::<T>(shape)` and `select(mask, on_true, on_false)`), arithmetic, logic, comparisons, negation, casts,
    /// and literals.
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
                if scope.slice(name).is_some() {
                    return Err(syn::Error::new_spanned(
                        name,
                        "kernel slices only support `load`, `store`, `accumulate`, and `atomic_accumulate`",
                    ));
                }
                Ok(quote!(#name.clone()))
            }
            syn::Expr::Paren(syn::ExprParen { expr, .. }) | syn::Expr::Group(syn::ExprGroup { expr, .. }) => {
                self.generate_unscoped_value(expr, scope)
            }
            syn::Expr::MethodCall(call) => self.generate_method_call(call, scope),
            syn::Expr::Call(call) => self.generate_call(call, scope),
            syn::Expr::Binary(
                binary @ syn::ExprBinary {
                    op:
                        syn::BinOp::Eq(_)
                        | syn::BinOp::Ne(_)
                        | syn::BinOp::Lt(_)
                        | syn::BinOp::Le(_)
                        | syn::BinOp::Gt(_)
                        | syn::BinOp::Ge(_),
                    ..
                },
            ) => self.generate_comparison(binary, scope),
            syn::Expr::Binary(binary) => self.generate_arithmetic(binary, scope),
            syn::Expr::Unary(unary) => self.generate_unary(unary, scope),
            syn::Expr::Cast(cast) => self.generate_cast(cast, scope),
            syn::Expr::Lit(literal) => Ok(quote! {
                __kernel_context.lift(#core::arrays::ArrayIrValue::Array(#core::arrays::Array::scalar(#literal)?))?
            }),
            _ => Err(syn::Error::new_spanned(
                expression,
                "unsupported kernel expression; control flow requires staged lowering and cannot execute as host Rust",
            )),
        }
    }

    /// Generates the lowering of a capability call of a kernel body value, whose name and number of arguments are
    /// matched against a closed table (i.e., the reference accesses `load`, `load_or(fill)`, `load_where(mask, fill)`,
    /// `store(value)`, `store_where(value, mask)`, `swap_where(value, mask, fill)`, `window_mask()`,
    /// `accumulate(value)`, and `atomic_accumulate(value)`, the asynchronous copies `source.copy_to(target)` and
    /// `token.wait()`, `left.dot(right)`, the reductions with literal axes, and the elementwise math capabilities).
    /// Calls on a [`TileView`] or a [`SliceView`] are lowered as accesses of the view. Stores into a tiled output write
    /// the tile of the current program, `load()` is rejected on a parameter with a masked window, and the fill
    /// literals of masked accesses of a parameter take the element type of that parameter.
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
        let method = call.method.to_string();
        let Some(arity) = method_arity(&method) else {
            return Err(syn::Error::new_spanned(call, "unsupported kernel capability call"));
        };
        if call.args.len() != arity {
            return Err(syn::Error::new_spanned(
                call,
                format!("kernel capability `{method}` takes {} but received {}", arguments(arity), call.args.len()),
            ));
        }
        let view = match ungroup(&call.receiver) {
            syn::Expr::MethodCall(receiver) if receiver.method == "slice" => {
                Some(self.extract_slice_view(receiver, scope)?)
            }
            receiver => identifier(receiver).and_then(|name| scope.slice(&name)).cloned(),
        };
        if let Some(view) = view {
            return self.generate_slice_access(call, &view, scope);
        }
        if method == "slice" {
            return Err(syn::Error::new_spanned(
                call,
                "kernel slices only support `load`, `store`, `accumulate`, and `atomic_accumulate`",
            ));
        }
        let parameter = identifier(&call.receiver).and_then(|name| self.reference(&name));
        if method == "load"
            && let Some(parameter) = parameter
            && parameter.boundary() == Boundary::Masked
        {
            return Err(syn::Error::new_spanned(
                &call.method,
                format!(
                    "`load` cannot read the padding of the masked window of `{}`, so use `load_or(fill)` instead",
                    parameter.name,
                ),
            ));
        }
        // Fill literals take the element type of the parameter that they fill, and otherwise their own type.
        let fill = |argument: &syn::Expr| {
            if !is_scalar_literal(argument) {
                return Err(syn::Error::new_spanned(argument, "kernel fill values require a scalar literal"));
            }
            let element = parameter.map(Parameter::element_type).transpose()?;
            Ok((element.map(|element| quote!(::<#element, _>)), argument.clone()))
        };
        let receiver = self.generate_value(&call.receiver, scope)?;
        let values = |indices: &[usize]| {
            indices
                .iter()
                .map(|&index| self.generate_value(&call.args[index], scope))
                .collect::<syn::Result<Vec<_>>>()
        };
        match method.as_str() {
            "load" => Ok(quote!(#core::operations::ReferenceRead::read(&(#receiver))?)),
            "load_or" => {
                let (element, fill) = fill(&call.args[0])?;
                Ok(quote!(#kernels::masked_load_or #element(&__kernel_context, &(#receiver), #fill)?))
            }
            "load_where" => {
                let [mask] = values(&[0])?.try_into().unwrap();
                let (element, fill) = fill(&call.args[1])?;
                Ok(quote!(#kernels::masked_load #element(&__kernel_context, &(#receiver), &(#mask), #fill)?))
            }
            "store" => {
                let [value] = values(&[0])?.try_into().unwrap();
                if parameter.is_some_and(|parameter| matches!(parameter.window, Window::Tiled(_))) {
                    Ok(quote!(#kernels::tile_store(&__kernel_context, &(#receiver), &(#value))?))
                } else {
                    Ok(quote!(#core::operations::ReferenceWrite::write(&(#receiver), &(#value))?))
                }
            }
            "store_where" => {
                let [value, mask] = values(&[0, 1])?.try_into().unwrap();
                Ok(quote!(#kernels::masked_store(&__kernel_context, &(#receiver), &(#value), &(#mask))?))
            }
            "swap_where" => {
                let [value, mask] = values(&[0, 1])?.try_into().unwrap();
                let (element, fill) = fill(&call.args[2])?;
                Ok(quote! {
                    #kernels::masked_swap #element(&__kernel_context, &(#receiver), &(#value), &(#mask), #fill)?
                })
            }
            "window_mask" => Ok(quote!(#kernels::window_mask(&__kernel_context, &(#receiver))?)),
            "accumulate" => {
                let [value] = values(&[0])?.try_into().unwrap();
                let boundary = parameter.map_or(Boundary::InBounds, Parameter::boundary).generate(kernels);
                Ok(quote!(#kernels::accumulate(&__kernel_context, &(#receiver), &(#value), #boundary)?))
            }
            "atomic_accumulate" => {
                let [value] = values(&[0])?.try_into().unwrap();
                Ok(quote!(#core::operations::ReferenceAtomicAddUpdate::atomic_add_update(&(#receiver), &(#value))?))
            }
            "copy_to" => {
                let [target] = values(&[0])?.try_into().unwrap();
                Ok(quote!(#kernels::async_copy(&__kernel_context, &(#receiver), &(#target))?))
            }
            "wait" => Ok(quote!(#kernels::wait(&__kernel_context, &(#receiver))?)),
            "dot" => {
                let [right] = values(&[0])?.try_into().unwrap();
                Ok(quote!(#kernels::dot(&(#receiver), &(#right))?))
            }
            method => {
                if let Some((_, kind)) = REDUCTION_FUNCTIONS.iter().find(|(name, _)| *name == method) {
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
                            syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(axis), .. }) => {
                                axis.base10_parse::<usize>()
                            }
                            _ => Err(syn::Error::new_spanned(axis, "kernel reduction axes require integer literals")),
                        })
                        .collect::<syn::Result<Vec<_>>>()?;
                    let kind = format_ident!("{}", kind);
                    return Ok(quote! {
                        __kernel_context.bind(
                            #core::arrays::ArrayIrOperation::from(#core::arrays::ArrayOperation::Reduce(
                                #core::operations::ReduceOperation::new(
                                    ::std::vec![#(#axes),*],
                                    #core::operations::ReductionKind::#kind,
                                ),
                            )),
                            ::std::vec![],
                            &[#receiver],
                        )?.remove(0)
                    });
                }
                let (variant, arguments) = match UNARY_FUNCTIONS.iter().find(|(name, _)| *name == method) {
                    Some((_, variant)) => (variant, Vec::new()),
                    None => {
                        let (_, variant) = BINARY_FUNCTIONS.iter().find(|(name, _)| *name == method).unwrap();
                        (variant, values(&[0])?)
                    }
                };
                Ok(self.generate_elementwise(variant, std::iter::once(receiver).chain(arguments).collect()))
            }
        }
    }

    /// Generates the lowering of a capability call on the provided [`SliceView`], which supports `load()`,
    /// `store(value)`, `accumulate(value)`, and `atomic_accumulate(value)` through the reference capabilities of the
    /// `ReferenceView` that `reference_slice` returns.
    fn generate_slice_access(
        &self,
        call: &syn::ExprMethodCall,
        view: &SliceView,
        scope: &Scope,
    ) -> syn::Result<TokenStream> {
        let Self { kernels, core, .. } = self;
        let SliceView { source, ranges } = view;
        let source = self.generate_value(&syn::parse_quote!(#source), scope)?;
        let view = quote!(#kernels::reference_slice(&(#source), &[#(#ranges),*])?);
        let capability = match call.method.to_string().as_str() {
            "load" => return Ok(quote!(#core::operations::ReferenceRead::read(&(#view))?)),
            "store" => quote!(ReferenceWrite::write),
            "accumulate" => quote!(ReferenceAddUpdate::add_update),
            "atomic_accumulate" => quote!(ReferenceAtomicAddUpdate::atomic_add_update),
            _ => {
                return Err(syn::Error::new_spanned(
                    call,
                    "kernel slices only support `load`, `store`, `accumulate`, and `atomic_accumulate`",
                ));
            }
        };
        let value = self.generate_value(&call.args[0], scope)?;
        Ok(quote!(#core::operations::#capability(&(#view), &(#value))?))
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

    /// Generates the lowering of a helper call, whose name and number of arguments are matched against a closed table
    /// (i.e., `zeros::<T>(shape)`, `ones::<T>(shape)`, `full(shape, literal)`, `iota::<T>(extent)`,
    /// `scratch::<T>(shape)`, `scratch_aligned::<T>(shape, alignment)`, and `select(mask, on_true, on_false)`). Shapes,
    /// extents, and alignments are shape metadata, `T` is the element type, and `full` may also name its element type
    /// explicitly (e.g., `full::<f32>(shape, 0.5)`) instead of taking it from its literal. `iota` counts along the only
    /// axis of a vector, and `scratch` uses the natural alignment of `T`.
    fn generate_call(&self, call: &syn::ExprCall, scope: &Scope) -> syn::Result<TokenStream> {
        let kernels = &self.kernels;
        let syn::Expr::Path(syn::ExprPath { qself: None, path, .. }) = ungroup(&call.func) else {
            return Err(syn::Error::new_spanned(call, "kernel helpers must be recognized named capabilities"));
        };
        let segment = path.segments.first().filter(|_| path.leading_colon.is_none() && path.segments.len() == 1);
        let helper = segment.map(|segment| segment.ident.to_string());
        let arity = match helper.as_deref() {
            Some("zeros" | "ones" | "iota" | "scratch") => 1,
            Some("full" | "scratch_aligned") => 2,
            Some("select") => 3,
            _ => return Err(syn::Error::new_spanned(call, "unsupported kernel helper call")),
        };
        let (segment, helper) = (segment.unwrap(), helper.unwrap());
        if call.args.len() != arity {
            return Err(syn::Error::new_spanned(
                call,
                format!("kernel `{helper}` takes {} but received {}", arguments(arity), call.args.len()),
            ));
        }
        let element = match &segment.arguments {
            syn::PathArguments::None => None,
            arguments if helper == "select" => {
                return Err(syn::Error::new_spanned(arguments, "kernel select does not take an element type"));
            }
            syn::PathArguments::AngleBracketed(syn::AngleBracketedGenericArguments { args, .. })
                if args.len() == 1
                    && let Some(syn::GenericArgument::Type(element)) = args.first() =>
            {
                Some(element)
            }
            arguments => {
                return Err(syn::Error::new_spanned(arguments, format!("kernel {helper} requires one element type")));
            }
        };
        if element.is_none() && !matches!(helper.as_str(), "full" | "select") {
            return Err(syn::Error::new_spanned(call, format!("kernel {helper} requires one explicit element type")));
        }
        match helper.as_str() {
            "zeros" | "ones" => {
                let helper = format_ident!("{}", helper);
                let shape = self.generate_shape(&call.args[0], true)?;
                Ok(quote!(#kernels::#helper::<#element, _>(&__kernel_context, #shape)?))
            }
            "full" => {
                let shape = self.generate_shape(&call.args[0], true)?;
                let value = &call.args[1];
                if !is_scalar_literal(value) {
                    return Err(syn::Error::new_spanned(value, "kernel fill values require a scalar literal"));
                }
                let element = element.map(|element| quote!(::<#element, _>));
                Ok(quote!(#kernels::full #element(&__kernel_context, #shape, #value)?))
            }
            "iota" => {
                let extent = self.generate_shape(&call.args[0], true)?;
                Ok(quote!(#kernels::iota::<#element, _>(&__kernel_context, [#extent], 0)?))
            }
            "scratch" | "scratch_aligned" => {
                let shape = self.generate_shape(&call.args[0], true)?;
                let alignment = match call.args.get(1) {
                    Some(alignment) => self.generate_shape(alignment, true)?,
                    None => quote!(::core::mem::align_of::<#element>()),
                };
                Ok(quote!(#kernels::scratch::<#element, _>(&__kernel_context, #shape, #alignment)?))
            }
            _ => {
                let [mask, on_true, on_false] = call
                    .args
                    .iter()
                    .map(|argument| self.generate_value(argument, scope))
                    .collect::<syn::Result<Vec<_>>>()?
                    .try_into()
                    .unwrap();
                Ok(quote!(#kernels::select(&__kernel_context, &(#mask), &(#on_true), &(#on_false))?))
            }
        }
    }

    /// Generates the lowering of `left op right`, where `op` is one of `+`, `-`, `*`, `/`, `%`, `&`, `|`, and `^`,
    /// into the corresponding elementwise array operation. `&&` and `||` are rejected, because both operands of a
    /// staged predicate are always evaluated.
    fn generate_arithmetic(&self, binary: &syn::ExprBinary, scope: &Scope) -> syn::Result<TokenStream> {
        let variant = match binary.op {
            syn::BinOp::Add(_) => "Add",
            syn::BinOp::Sub(_) => "Sub",
            syn::BinOp::Mul(_) => "Mul",
            syn::BinOp::Div(_) => "Div",
            syn::BinOp::Rem(_) => "Rem",
            syn::BinOp::BitAnd(_) => "And",
            syn::BinOp::BitOr(_) => "Or",
            syn::BinOp::BitXor(_) => "Xor",
            syn::BinOp::And(_) => {
                return Err(syn::Error::new_spanned(
                    binary.op,
                    "use `&` instead of `&&`, because kernel predicates are staged values",
                ));
            }
            syn::BinOp::Or(_) => {
                return Err(syn::Error::new_spanned(
                    binary.op,
                    "use `|` instead of `||`, because kernel predicates are staged values",
                ));
            }
            _ => return Err(syn::Error::new_spanned(binary.op, "unsupported kernel arithmetic operator")),
        };
        let left = self.generate_value(&binary.left, scope)?;
        let right = self.generate_value(&binary.right, scope)?;
        Ok(self.generate_elementwise(variant, vec![left, right]))
    }

    /// Generates the lowering of `left op right`, where `op` is one of `==`, `!=`, `<`, `<=`, `>`, and `>=`, into a
    /// comparison through `compare`. A grid axis name compared with an integer literal on either side is compared as a
    /// dimension with a constant through `compare_dimension` (e.g., `k == 0`), which is the form of the first step
    /// predicates that kernel initialization recognizes.
    fn generate_comparison(&self, binary: &syn::ExprBinary, scope: &Scope) -> syn::Result<TokenStream> {
        let Self { kernels, core, .. } = self;
        let (direction, mirrored) = match binary.op {
            syn::BinOp::Eq(_) => ("Equal", "Equal"),
            syn::BinOp::Ne(_) => ("NotEqual", "NotEqual"),
            syn::BinOp::Lt(_) => ("LessThan", "GreaterThan"),
            syn::BinOp::Le(_) => ("LessThanOrEqual", "GreaterThanOrEqual"),
            syn::BinOp::Gt(_) => ("GreaterThan", "LessThan"),
            syn::BinOp::Ge(_) => ("GreaterThanOrEqual", "LessThanOrEqual"),
            _ => return Err(syn::Error::new_spanned(binary.op, "unsupported kernel comparison operator")),
        };
        let is_axis = |expression: &syn::Expr| {
            identifier(expression).is_some_and(|name| self.axes().iter().any(|axis| axis.name == name))
        };
        let constant = |expression: &syn::Expr| match ungroup(expression) {
            syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(constant), .. }) => Some(constant.base10_parse::<usize>()),
            _ => None,
        };
        let dimension = match (constant(&binary.left), constant(&binary.right)) {
            (None, Some(constant)) if is_axis(&binary.left) => Some((&binary.left, direction, constant?)),
            (Some(constant), None) if is_axis(&binary.right) => Some((&binary.right, mirrored, constant?)),
            _ => None,
        };
        if let Some((axis, direction, constant)) = dimension {
            let axis = self.generate_value(axis, scope)?;
            let direction = format_ident!("{}", direction);
            return Ok(quote! {
                #kernels::compare_dimension(
                    &__kernel_context,
                    &(#axis),
                    #core::operations::ComparisonDirection::#direction,
                    #constant,
                )?
            });
        }
        let left = self.generate_value(&binary.left, scope)?;
        let right = self.generate_value(&binary.right, scope)?;
        let direction = format_ident!("{}", direction);
        Ok(quote! {
            #kernels::compare(
                &__kernel_context,
                &(#left),
                &(#right),
                #core::operations::ComparisonDirection::#direction,
            )?
        })
    }

    /// Generates the lowering of `-value` and `!value` into elementwise negation and logical negation.
    fn generate_unary(&self, unary: &syn::ExprUnary, scope: &Scope) -> syn::Result<TokenStream> {
        let variant = match unary.op {
            syn::UnOp::Neg(_) => "Neg",
            syn::UnOp::Not(_) => "Not",
            _ => return Err(syn::Error::new_spanned(unary.op, "unsupported kernel unary operator")),
        };
        let value = self.generate_value(&unary.expr, scope)?;
        Ok(self.generate_elementwise(variant, vec![value]))
    }

    /// Generates the lowering of `value as T`, where `T` is a Rust scalar type, into an element type conversion
    /// through the `ConvertElementType` capability.
    fn generate_cast(&self, cast: &syn::ExprCast, scope: &Scope) -> syn::Result<TokenStream> {
        let core = &self.core;
        let data_type = match cast.ty.as_ref() {
            syn::Type::Path(syn::TypePath { qself: None, path }) => path.get_ident().and_then(|element| {
                ELEMENT_TYPES.iter().find(|(_, rust)| element == rust).map(|(data_type, _)| data_type)
            }),
            _ => None,
        };
        let Some(data_type) = data_type else {
            return Err(syn::Error::new_spanned(&cast.ty, "kernel casts require a Rust scalar element type"));
        };
        let data_type = format_ident!("{}", data_type);
        let value = self.generate_value(&cast.expr, scope)?;
        Ok(quote! {
            #core::operations::ConvertElementType::convert_element_type(
                &(#value),
                #core::arrays::DataType::#data_type,
            )?
        })
    }

    /// Generates the binding of the elementwise `ArrayOperation` with the provided `variant` (e.g., `Add`) to `inputs`.
    fn generate_elementwise(&self, variant: &str, inputs: Vec<TokenStream>) -> TokenStream {
        let core = &self.core;
        let operation = format_ident!("{}Operation", variant);
        let variant = format_ident!("{}", variant);
        quote! {
            __kernel_context.bind(
                #core::arrays::ArrayIrOperation::from(#core::arrays::ArrayOperation::#variant(
                    #core::operations::#operation::new(),
                )),
                ::std::vec![],
                &[#(#inputs),*],
            )?.remove(0)
        }
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

/// Kernel parameter declared through an `#[input(...)]`, `#[output(...)]`, `#[inout(...)]`, or `#[prefetch(...)]`
/// annotation.
struct Parameter {
    /// Name of the parameter.
    name: syn::Ident,

    /// [`Access`] that the annotation of the parameter declares.
    access: Access,

    /// `DataType` variant of the elements of the parameter (e.g., `F32`).
    data_type: syn::Ident,

    /// [`ShapeAnnotation`] of the parameter.
    shape: ShapeAnnotation,

    /// [`Window`] of the parameter that each program of the kernel accesses.
    window: Window,
}

impl Parameter {
    /// Returns `true` if this parameter is a result of the kernel callable (i.e., an `output` or `inout` parameter).
    fn is_result(&self) -> bool {
        matches!(self.access, Access::Output | Access::Inout)
    }

    /// Returns the [`Boundary`] policy of the window of this parameter, which is [`Boundary::InBounds`] for a whole
    /// array window.
    fn boundary(&self) -> Boundary {
        match &self.window {
            Window::WholeArray => Boundary::InBounds,
            Window::Tiled(Tiling { boundary, .. }) | Window::Block(Block { boundary, .. }) => *boundary,
        }
    }

    /// Returns the Rust scalar type of the elements of this parameter (e.g., `f32` for `F32`), which types the
    /// padding literals of its [`TileView`]s and the fill literals of its masked accesses.
    fn element_type(&self) -> syn::Result<TokenStream> {
        match ELEMENT_TYPES.iter().find(|(data_type, _)| self.data_type == data_type) {
            Some((_, element)) => {
                let element = format_ident!("{}", element);
                Ok(quote!(#element))
            }
            None => Err(syn::Error::new_spanned(
                &self.data_type,
                "kernel fill literals require a data type with a Rust scalar element type",
            )),
        }
    }
}

/// Access of a kernel [`Parameter`] that its annotation declares.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Access {
    /// Read-only argument declared through `#[input(...)]`.
    Input,

    /// Write-only result declared through `#[output(...)]`.
    Output,

    /// Read-write argument that is also a result, declared through `#[inout(...)]`.
    Inout,

    /// Rank-0 integer argument whose value is prefetched for the window mappings and the body, declared through
    /// `#[prefetch(...)]`.
    Prefetch,
}

impl Access {
    /// Generates the `KernelParameterAccess` of a reference parameter with this access.
    fn generate(self, kernels: &syn::Path) -> TokenStream {
        match self {
            Self::Input | Self::Prefetch => quote!(#kernels::KernelParameterAccess::ReadOnly),
            Self::Output => quote!(#kernels::KernelParameterAccess::WriteOnly),
            Self::Inout => quote!(#kernels::KernelParameterAccess::ReadWrite),
        }
    }
}

/// Shape annotation of a kernel [`Parameter`].
enum ShapeAnnotation {
    /// Rank that the type of an argument must have (i.e., the `rank` metadata of an `input` or `inout` parameter, and
    /// `0` for a `prefetch` parameter).
    Rank(usize),

    /// Shape metadata expression for the static shape of an `output` parameter.
    Shape(syn::Expr),
}

/// Window of a kernel [`Parameter`] that each program of the kernel accesses.
enum Window {
    /// Whole array, at every grid point.
    WholeArray,

    /// Output tile of the current program of the `tile` shorthand.
    Tiled(Tiling),

    /// [`Block`] declared through the `block` metadata.
    Block(Block),
}

/// Partition of a kernel output into tiles, each of which is written by one program of a parallel grid, declared
/// through the `tile = ...` and `boundary = ...` metadata of the output annotation.
struct Tiling {
    /// Shape metadata expression for the static shape of each tile.
    tile: syn::Expr,

    /// [`Boundary`] policy for the tiles that extend past the output.
    boundary: Boundary,
}

/// Block window of a kernel parameter, declared through its `block`, `index` or `start`, and `boundary` metadata.
struct Block {
    /// Shape metadata array for the static shape of the block.
    shape: syn::Expr,

    /// [`Boundary`] policy for the blocks that extend past the array.
    boundary: Boundary,

    /// Mapping expression per block axis, which is a block index or an element start.
    expressions: Vec<syn::Expr>,

    /// Indicates whether the expressions are element starts (i.e., `start` metadata) rather than block indices (i.e.,
    /// `index` metadata).
    starts: bool,
}

/// Policy for the windows that extend past their array, which corresponds to the `BoundaryPolicy` of the kernel
/// namespace.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Boundary {
    /// Windows may extend past their array (i.e., `boundary = masked`), and only their in-bounds elements are valid.
    Masked,

    /// Windows must lie within their array (i.e., `boundary = in_bounds`).
    InBounds,
}

impl Boundary {
    /// Generates the `BoundaryPolicy` that corresponds to this boundary.
    fn generate(self, kernels: &syn::Path) -> TokenStream {
        match self {
            Self::Masked => quote!(#kernels::BoundaryPolicy::Masked),
            Self::InBounds => quote!(#kernels::BoundaryPolicy::InBounds),
        }
    }
}

/// Axis of the grid of a kernel, declared through the `grid(...)` argument of the attribute.
struct GridAxis {
    /// Name of the axis, which the kernel body binds to the coordinate of the current program along it.
    name: syn::Ident,

    /// Shape metadata expression for the extent of the axis.
    extent: syn::Expr,

    /// Indicates whether the axis is sequential (i.e., `sequential(extent)`) rather than parallel.
    sequential: bool,
}

/// Lexical scope of a kernel body, which tracks the bindings that staged loops and conditions carry and the tile and
/// slice views that can be accessed. Scopes of nested regions are copies, so that their declarations do not leak out
/// of them.
#[derive(Clone)]
struct Scope {
    /// [`Binding`]s that are visible in this scope, in declaration order.
    bindings: Vec<Binding>,

    /// [`TileView`]s that are visible in this scope.
    tiles: Vec<TileView>,

    /// Named [`SliceView`]s that are visible in this scope.
    slices: Vec<(syn::Ident, SliceView)>,

    /// Indicates whether this scope belongs to the region of a staged loop or condition.
    region: bool,
}

impl Scope {
    /// Creates a new top-level [`Scope`] with the provided [`Binding`]s.
    fn new(bindings: Vec<Binding>) -> Self {
        Self { bindings, tiles: Vec::new(), slices: Vec::new(), region: false }
    }

    /// Returns a copy of this [`Scope`] for the region of a staged loop or condition.
    fn region(&self) -> Self {
        Self { region: true, ..self.clone() }
    }

    /// Returns the [`TileView`] named `name` in this [`Scope`], if there is one.
    fn tile(&self, name: &syn::Ident) -> Option<&TileView> {
        self.tiles.iter().find(|tile| &tile.name == name)
    }

    /// Returns the [`SliceView`] named `name` in this [`Scope`], if there is one.
    fn slice(&self, name: &syn::Ident) -> Option<&SliceView> {
        self.slices.iter().find(|(slice, _)| slice == name).map(|(_, view)| view)
    }

    /// Validates that `pattern` declares a new name in this [`Scope`] and returns that name. New names cannot use the
    /// reserved prefix of generated identifiers, bind by reference, have subpatterns, or shadow other [`Binding`]s or
    /// views (because staged regions carry bindings by name), and they can be mutable only if `mutable` is `true`.
    /// The provided `kind` (e.g., `"locals"`) names the declared bindings in diagnostics.
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
        if self.bindings.iter().any(|binding| &binding.name == name)
            || self.tile(name).is_some()
            || self.slice(name).is_some()
        {
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

/// Binding of a kernel body to a staged value (e.g., a parameter reference, a grid coordinate, a loop index, or a
/// local).
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

/// Static sub-window of a reference written as `reference.slice([ranges])`, whose accesses are lowered through the
/// `ReferenceView` that `reference_slice` returns. A slice stages nothing until it is accessed.
#[derive(Clone)]
struct SliceView {
    /// Name of the binding of the sliced reference.
    source: syn::Ident,

    /// Generated `Option<Range<usize>>` per axis of the reference, where `None` selects the whole axis.
    ranges: Vec<TokenStream>,
}

/// Returns the number of arguments of the kernel capability `method`, or [`None`] when kernel bodies do not support
/// it.
fn method_arity(method: &str) -> Option<usize> {
    match method {
        "load" | "window_mask" | "wait" => Some(0),
        "load_or" | "store" | "accumulate" | "atomic_accumulate" | "copy_to" | "dot" | "slice" => Some(1),
        "load_where" | "store_where" => Some(2),
        "swap_where" => Some(3),
        method if UNARY_FUNCTIONS.iter().any(|(name, _)| *name == method) => Some(0),
        method if BINARY_FUNCTIONS.iter().chain(&REDUCTION_FUNCTIONS).any(|(name, _)| *name == method) => Some(1),
        _ => None,
    }
}

/// Returns `count` followed by "argument" or "arguments", for diagnostics.
fn arguments(count: usize) -> String {
    if count == 1 { "1 argument".to_string() } else { format!("{count} arguments") }
}

/// Returns `true` if `expression` is a scalar literal, which may be negated.
fn is_scalar_literal(expression: &syn::Expr) -> bool {
    match ungroup(expression) {
        syn::Expr::Lit(_) => true,
        syn::Expr::Unary(syn::ExprUnary { op: syn::UnOp::Neg(_), expr, .. }) => {
            matches!(ungroup(expr), syn::Expr::Lit(_))
        }
        _ => false,
    }
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
        let parameters = KernelCodeGenerator::extract_parameters(&function().sig).unwrap();
        KernelCodeGenerator::new(parse_quote!(kernels), None, None, parameters)
    }

    /// Returns the kernel function with a grid, block windows, a `prefetch` parameter, and two results that the tests
    /// of mapped calls generate code for.
    fn mapped_function() -> syn::ItemFn {
        parse_quote! {
            pub fn update(
                #[input(data_type = F32, rank = 2, block = [8, 32], index = [i, k], boundary = masked)] left: &Array,
                #[prefetch(data_type = I32)] offset: &Array,
                #[inout(data_type = F32, rank = 1, block = [8], start = [offset + i * 8])] state: &mut Array,
                #[output(data_type = I32, shape = [left.shape()[0]], block = [8], index = [i])] output: &mut Array,
            ) {}
        }
    }

    /// Returns a [`KernelCodeGenerator`] for [`mapped_function`] with the grid `(i, k)`, where `k` is sequential, that
    /// names the kernel namespace `kernels`.
    fn mapped_generator() -> KernelCodeGenerator {
        let (_, _, grid) =
            KernelCodeGenerator::extract_arguments(quote!(grid(i = left.shape()[0].div_ceil(8), k = sequential(4))))
                .unwrap();
        let parameters = KernelCodeGenerator::extract_parameters(&mapped_function().sig).unwrap();
        KernelCodeGenerator::new(parse_quote!(kernels), None, grid, parameters)
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

    /// Returns the top-level [`Scope`] of [`mapped_function`].
    fn mapped_scope() -> Scope {
        Scope::new(
            ["left", "state", "output", "i", "k", "offset"]
                .into_iter()
                .map(|name| Binding { name: format_ident!("{name}"), mutable: false })
                .collect(),
        )
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

    /// Returns the lowering of the literal `literal` under the provenance scope of its expression.
    fn scoped_literal(literal: TokenStream) -> TokenStream {
        scoped(quote! {
            __kernel_context.lift(kernels::__private::core::arrays::ArrayIrValue::Array(
                kernels::__private::core::arrays::Array::scalar(#literal)?
            ))?
        })
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

    /// Returns the [`Parameter`] that is extracted from `argument`.
    fn parameter(argument: syn::FnArg) -> Parameter {
        let syn::FnArg::Typed(argument) = argument else { panic!("expected a typed function argument") };
        KernelCodeGenerator::extract_parameter(&argument).unwrap()
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
                            [kernels::__private::static_extent(input, 0usize)?]
                        );
                        let __kernel_operation = {
                            let __kernel_parameters = ::std::vec![
                                kernels::whole_array_parameter(
                                    input.clone(),
                                    kernels::KernelParameterAccess::ReadOnly
                                )?,
                                kernels::whole_array_parameter(
                                    __kernel_output_type,
                                    kernels::KernelParameterAccess::WriteOnly
                                )?
                            ];
                            kernels::KernelCallOperation::new(kernels::Grid::new(::std::vec![])?, __kernel_parameters)?
                        };
                        let __kernel_input_metadata = input;
                        kernels::KernelDefinition::trace_with_prefetch(
                            __kernel_operation,
                            |(__kernel_references, __kernel_coordinates, __kernel_prefetched)| {
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
            quote!(::core::compile_error! { "kernels require at least one `output` or `inout` parameter" }).to_string(),
        );
        assert_eq!(
            KernelCodeGenerator::generate_kernel_impl(
                quote!(grid(output = 2)),
                quote!(
                    fn invalid(#[output(data_type = F32, shape = [2])] output: &mut Array) {}
                )
            )
            .to_string(),
            quote!(::core::compile_error! { "kernel grid axis names cannot collide with parameter names" }).to_string(),
        );
    }

    #[test]
    fn test_kernel_code_generator_extract_arguments() {
        let (kernels, requirement, grid) = KernelCodeGenerator::extract_arguments(quote!()).unwrap();
        assert_eq!(kernels.to_compact_string(), "ryft::kernels");
        assert!(requirement.is_none());
        assert!(grid.is_none());

        let (kernels, requirement, grid) = KernelCodeGenerator::extract_arguments(quote!(
            crate = "::kernels_alias",
            requires = left.shape()[0] == right.shape()[0],
            grid(i = left.shape()[0], j = parallel(2), k = sequential(left.shape()[1].div_ceil(8)))
        ))
        .unwrap();
        assert_eq!(kernels.to_compact_string(), "::kernels_alias");
        assert_eq!(requirement.unwrap().to_compact_string(), "left.shape()[0]==right.shape()[0]");
        assert_eq!(
            grid.unwrap()
                .iter()
                .map(|axis| (axis.name.to_string(), axis.extent.to_compact_string(), axis.sequential))
                .collect::<Vec<_>>(),
            vec![
                ("i".to_string(), "left.shape()[0]".to_string(), false),
                ("j".to_string(), "2".to_string(), false),
                ("k".to_string(), "left.shape()[1].div_ceil(8)".to_string(), true),
            ],
        );

        let error = |attributes| KernelCodeGenerator::extract_arguments(attributes).err().unwrap().to_string();
        assert_eq!(error(quote!(crate = "a", crate = "b")), "duplicate ryft attribute `crate`");
        assert_eq!(error(quote!(requires = true, requires = false)), "duplicate ryft attribute `requires`");
        assert_eq!(error(quote!(crate = kernels)), "expected ryft `crate` attribute to be a string: `crate = \"...\"`");
        assert_eq!(
            error(quote!(schedule = [1])),
            "unsupported `kernel` argument (expected `crate = \"...\"`, `requires = ...`, or `grid(...)`)",
        );
        assert_eq!(error(quote!(grid(i = 1), grid(j = 1))), "duplicate ryft attribute `grid`");
        assert_eq!(error(quote!(grid(i = 1, i = 2))), "kernel grid axis names must be unique");
        assert_eq!(error(quote!(grid(axes::i = 1))), "kernel grid axes require a name");
        assert_eq!(
            error(quote!(grid(__kernel_i = 1))),
            "kernel grid axis names cannot use the reserved `__kernel_` prefix",
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
        let parameters = KernelCodeGenerator::extract_parameters(&mapped_function().sig).unwrap();
        assert_eq!(
            parameters
                .iter()
                .map(|parameter| (parameter.name.to_string(), parameter.access, parameter.data_type.to_string()))
                .collect::<Vec<_>>(),
            vec![
                ("left".to_string(), Access::Input, "F32".to_string()),
                ("offset".to_string(), Access::Prefetch, "I32".to_string()),
                ("state".to_string(), Access::Inout, "F32".to_string()),
                ("output".to_string(), Access::Output, "I32".to_string()),
            ],
        );

        // Kernels may have several results, which may all be `inout` parameters.
        assert_eq!(
            KernelCodeGenerator::extract_parameters(&parse_quote!(fn valid(
                #[output(data_type = F32, shape = [1])] first: &mut Array,
                #[output(data_type = F32, shape = [1])] second: &mut Array,
            )))
            .unwrap()
            .len(),
            2,
        );
        assert_eq!(
            KernelCodeGenerator::extract_parameters(&parse_quote!(fn valid(
                #[inout(data_type = F32, rank = 1)] state: &mut Array,
            )))
            .unwrap()
            .len(),
            1,
        );

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
                #[input(data_type = F32, rank = 1)] input: &Array,
                #[prefetch(data_type = I32)] offset: &Array,
            ))),
            "kernels require at least one `output` or `inout` parameter",
        );
    }

    #[test]
    fn test_kernel_code_generator_extract_parameter() {
        let output = parameter(parse_quote!(
            #[output(data_type = U8, shape = [4], tile = [2], boundary = in_bounds)] output: &mut Array
        ));
        assert_eq!(output.access, Access::Output);
        assert!(matches!(output.shape, ShapeAnnotation::Shape(_)));
        assert!(matches!(output.window, Window::Tiled(Tiling { boundary: Boundary::InBounds, .. })));
        let state = parameter(parse_quote!(
            #[inout(data_type = I32, rank = 1, block = [2], start = [i], boundary = masked)] state: &mut Array
        ));
        assert_eq!(state.access, Access::Inout);
        assert!(matches!(state.shape, ShapeAnnotation::Rank(1)));
        assert!(matches!(state.window, Window::Block(Block { boundary: Boundary::Masked, starts: true, .. })));
        let offset = parameter(parse_quote!(#[prefetch(data_type = I32)] offset: &Array));
        assert_eq!(offset.access, Access::Prefetch);
        assert!(matches!(offset.shape, ShapeAnnotation::Rank(0)));
        assert!(matches!(offset.window, Window::WholeArray));

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
        for argument in [
            parse_quote!(#[input(data_type = F32, rank = 1)] input: &mut Array),
            parse_quote!(#[prefetch(data_type = I32)] offset: &mut Array),
            parse_quote!(#[output(data_type = F32, shape = [1])] output: &Array),
            parse_quote!(#[inout(data_type = F32, rank = 1)] state: &Array),
        ] {
            assert_eq!(
                parameter_error(argument),
                "kernel `input` and `prefetch` parameters must use `&Array`, and kernel `output` and `inout` \
                parameters must use `&mut Array`",
            );
        }

        // Annotations.
        assert_eq!(
            parameter_error(parse_quote!(input: &Array)),
            "each kernel parameter requires exactly one `input`, `output`, `inout`, or `prefetch` annotation",
        );
        assert_eq!(
            parameter_error(parse_quote!(#[inputs(data_type = F32, rank = 1)] input: &Array)),
            "expected an `input`, `output`, `inout`, or `prefetch` annotation",
        );
        assert_eq!(
            parameter_error(parse_quote!(#[input(data_type = F32, data_type = F64, rank = 1)] input: &Array)),
            "duplicate ryft attribute `data_type`",
        );
        assert_eq!(
            parameter_error(parse_quote!(#[input(data_type = F32, rank = 1, shape = [1])] input: &Array)),
            "`shape` and `tile` metadata are only supported for kernel outputs",
        );
        assert_eq!(
            parameter_error(parse_quote!(#[output(data_type = F32, rank = 1, shape = [1])] output: &mut Array)),
            "kernel outputs take their rank from their `shape`",
        );
        assert_eq!(
            parameter_error(parse_quote!(#[prefetch(data_type = I32, rank = 0)] offset: &Array)),
            "kernel `prefetch` parameters only take `data_type` metadata",
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
            parameter_error(parse_quote!(#[inout(data_type = F32)] state: &mut Array)),
            "kernel `input` and `inout` parameters require `rank` metadata",
        );
        assert_eq!(
            parameter_error(parse_quote!(#[input(data_type = F32, rank = RANK)] input: &Array)),
            "kernel parameter ranks must be integer literals",
        );
        assert_eq!(
            parameter_error(parse_quote!(#[output(data_type = F32)] output: &mut Array)),
            "kernel outputs require `shape` metadata",
        );
    }

    #[test]
    fn test_kernel_code_generator_extract_window() {
        let annotation: syn::Attribute = parse_quote!(#[input(data_type = F32, rank = 2)]);
        let window = |tile, block, index, start, boundary| {
            KernelCodeGenerator::extract_window(&annotation, tile, block, index, start, boundary)
        };
        assert!(matches!(window(None, None, None, None, None), Ok(Window::WholeArray)));
        assert!(matches!(
            window(Some(parse_quote!([2])), None, None, None, Some(parse_quote!(masked))),
            Ok(Window::Tiled(Tiling { boundary: Boundary::Masked, .. })),
        ));
        let Ok(Window::Block(block)) =
            window(None, Some(parse_quote!([2, 4])), Some(parse_quote!([i, k + 1])), None, None)
        else {
            panic!("expected a block window");
        };
        assert_eq!(block.shape.to_compact_string(), "[2,4]");
        assert_eq!(block.boundary, Boundary::InBounds);
        assert_eq!(
            block.expressions.iter().map(TokensHelpers::to_compact_string).collect::<Vec<_>>(),
            vec!["i", "k+1"],
        );
        assert!(!block.starts);
        assert!(matches!(
            window(None, Some(parse_quote!([2])), None, Some(parse_quote!([i])), Some(parse_quote!(masked))),
            Ok(Window::Block(Block { boundary: Boundary::Masked, starts: true, .. })),
        ));

        let error = |tile, block, index, start, boundary| {
            window(tile, block, index, start, boundary).err().unwrap().to_string()
        };
        assert_eq!(
            error(None, None, None, None, Some(parse_quote!(BoundaryPolicy::Masked))),
            "unsupported kernel boundary policy (expected `masked` or `in_bounds`)",
        );
        assert_eq!(
            error(Some(parse_quote!([2])), Some(parse_quote!([2])), None, None, Some(parse_quote!(masked))),
            "output `tile` metadata cannot be combined with `block`, `index`, or `start` metadata",
        );
        assert_eq!(
            error(Some(parse_quote!([2])), None, None, None, None),
            "output tiling requires explicit `tile` and `boundary` metadata",
        );
        assert_eq!(
            error(None, None, Some(parse_quote!([i])), None, None),
            "`index`, `start`, and `boundary` metadata require `block` metadata",
        );
        assert_eq!(
            error(None, None, None, None, Some(parse_quote!(masked))),
            "`index`, `start`, and `boundary` metadata require `block` metadata",
        );
        assert_eq!(
            error(None, Some(parse_quote!(BLOCK)), Some(parse_quote!([i])), None, None),
            "kernel `block` metadata requires an array",
        );
        assert_eq!(
            error(None, Some(parse_quote!([2])), Some(parse_quote!([i])), Some(parse_quote!([i])), None),
            "kernel windows take either `index` or `start` metadata",
        );
        assert_eq!(
            error(None, Some(parse_quote!([2])), None, None, None),
            "kernel `block` metadata requires `index` or `start` metadata",
        );
        assert_eq!(
            error(None, Some(parse_quote!([2])), Some(parse_quote!(indices)), None, None),
            "kernel `index` and `start` metadata require arrays",
        );
        assert_eq!(
            error(None, Some(parse_quote!([2, 2])), None, Some(parse_quote!([i])), None),
            "kernel `start` metadata requires one expression per `block` extent",
        );
    }

    #[test]
    fn test_kernel_code_generator_validate_grid() {
        let parameters = |signature: syn::Signature| KernelCodeGenerator::extract_parameters(&signature).unwrap();
        let grid = |attributes| KernelCodeGenerator::extract_arguments(attributes).unwrap().2;
        let mapped = parameters(mapped_function().sig);
        let tiled = parameters(function().sig);
        assert!(KernelCodeGenerator::validate_grid(grid(quote!(grid(i = 2, k = 2))).as_deref(), &mapped).is_ok());
        assert!(KernelCodeGenerator::validate_grid(None, &tiled).is_ok());

        let error = |grid: Option<Vec<GridAxis>>, parameters: &[Parameter]| {
            KernelCodeGenerator::validate_grid(grid.as_deref(), parameters).unwrap_err().to_string()
        };
        assert_eq!(
            error(grid(quote!(grid(i = 2, state = 2))), &mapped),
            "kernel grid axis names cannot collide with parameter names",
        );
        assert_eq!(error(grid(quote!(grid(i = 2))), &tiled), "`grid` cannot be combined with output `tile` metadata");
        for signature in [
            parse_quote!(fn invalid(
                #[output(data_type = F32, shape = [4], tile = [2], boundary = masked)] first: &mut Array,
                #[output(data_type = F32, shape = [4])] second: &mut Array,
            )),
            parse_quote!(fn invalid(
                #[prefetch(data_type = I32)] offset: &Array,
                #[output(data_type = F32, shape = [4], tile = [2], boundary = masked)] output: &mut Array,
            )),
            parse_quote!(fn invalid(
                #[input(data_type = F32, rank = 1, block = [2], index = [0])] input: &Array,
                #[output(data_type = F32, shape = [4], tile = [2], boundary = masked)] output: &mut Array,
            )),
        ] {
            assert_eq!(
                error(None, &parameters(signature)),
                "output `tile` metadata requires a single result, whole array inputs, and no `prefetch` parameters",
            );
        }
    }

    #[test]
    fn test_kernel_code_generator_arguments() {
        assert_eq!(
            mapped_generator().arguments().map(|parameter| parameter.name.to_string()).collect::<Vec<_>>(),
            vec!["left", "offset", "state"],
        );
    }

    #[test]
    fn test_kernel_code_generator_references() {
        assert_eq!(
            mapped_generator().references().map(|parameter| parameter.name.to_string()).collect::<Vec<_>>(),
            vec!["left", "state", "output"],
        );
    }

    #[test]
    fn test_kernel_code_generator_prefetches() {
        assert_eq!(
            mapped_generator().prefetches().map(|parameter| parameter.name.to_string()).collect::<Vec<_>>(),
            vec!["offset"],
        );
    }

    #[test]
    fn test_kernel_code_generator_axes() {
        assert_eq!(
            mapped_generator().axes().iter().map(|axis| axis.name.to_string()).collect::<Vec<_>>(),
            vec!["i", "k"],
        );
        assert!(generator().axes().is_empty());
    }

    #[test]
    fn test_kernel_code_generator_reference() {
        let generator = mapped_generator();
        assert_eq!(generator.reference(&format_ident!("state")).unwrap().access, Access::Inout);
        assert!(generator.reference(&format_ident!("offset")).is_none());
        assert!(generator.reference(&format_ident!("missing")).is_none());
    }

    #[test]
    fn test_kernel_code_generator_tiled_output() {
        assert_eq!(generator().tiled_output().unwrap().name, "output");
        assert!(mapped_generator().tiled_output().is_none());
    }

    #[test]
    fn test_kernel_code_generator_generate_callable() {
        assert_eq!(
            generator().generate_callable(&function()).to_string(),
            quote! {
                /// Gathers rows.
                #[allow(clippy::cloned_ref_to_slice_refs)]
                #[cfg(feature = "kernels")]
                #[cfg_attr(feature = "kernels", inline)]
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

        // The call receives the prefetched values last, and several results are returned as a tuple.
        assert_eq!(
            mapped_generator().generate_callable(&mapped_function()).to_string(),
            quote! {
                #[allow(clippy::cloned_ref_to_slice_refs)]
                pub fn update<__KernelValue: kernels::KernelCall>(
                    left: &__KernelValue,
                    offset: &__KernelValue,
                    state: &__KernelValue
                ) -> ::core::result::Result<
                    (__KernelValue, __KernelValue),
                    kernels::__private::core::programs::ProgramError
                > {
                    let __kernel_left_type = kernels::__private::core::programs::Typed::r#type(left);
                    let __kernel_offset_type = kernels::__private::core::programs::Typed::r#type(offset);
                    let __kernel_state_type = kernels::__private::core::programs::Typed::r#type(state);
                    let __kernel_definition = update::definition(
                        __kernel_left_type.as_ref(),
                        __kernel_offset_type.as_ref(),
                        __kernel_state_type.as_ref()
                    )
                    .map_err(kernels::__private::core::programs::ProgramError::custom)?;
                    let mut __kernel_outputs = <__KernelValue as kernels::KernelCall>::call_kernel(
                        &__kernel_definition,
                        &[left.clone(), state.clone(), offset.clone()],
                    )?;
                    if __kernel_outputs.len() != 2 {
                        return ::core::result::Result::Err(
                            kernels::__private::core::programs::ProgramError::InvalidOutputCount {
                                expected: 2,
                                actual: __kernel_outputs.len(),
                            }
                        );
                    }
                    ::core::result::Result::Ok((__kernel_outputs.remove(0), __kernel_outputs.remove(0)))
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
                        if !(kernels::__private::static_extent(left, 0usize)?
                            == kernels::__private::static_extent(indices, 0usize)?)
                        {
                            return ::core::result::Result::Err(kernels::KernelError::Type(
                                kernels::__private::core::programs::TypeError::invalid(
                                    "kernel shape requirement is not satisfied"
                                ),
                            ));
                        }
                        let __kernel_output_type = kernels::__private::core::arrays::ArrayType::new_static(
                            kernels::__private::core::arrays::DataType::F32,
                            [kernels::__private::static_extent(indices, 0usize)?, 32]
                        );
                        let __kernel_operation = #operation;
                        let __kernel_left_metadata = left;
                        let __kernel_indices_metadata = indices;
                        kernels::KernelDefinition::trace_with_prefetch(
                            __kernel_operation,
                            |(__kernel_references, __kernel_coordinates, __kernel_prefetched)| {
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
    fn test_kernel_code_generator_generate_definition_mapped() {
        // Every argument is checked and every output type is constructed, and the grid axes and the prefetched values
        // are bound after the references.
        let generator = mapped_generator();
        let operation = generator.generate_operation().unwrap();
        let (module_documentation, definition_documentation) = definition_documentation("update");
        let check = |name: &str, kind: &str, data_type: &str, rank: usize| {
            let name = format_ident!("{}", name);
            let message = format!("kernel {kind} `{name}` requires data type `{data_type}` and rank `{rank}`");
            let data_type = format_ident!("{}", data_type);
            quote! {
                if #name.data_type() != kernels::__private::core::arrays::DataType::#data_type
                    || #name.rank() != #rank
                {
                    return ::core::result::Result::Err(kernels::KernelError::Type(
                        kernels::__private::core::programs::TypeError::invalid(#message),
                    ));
                }
            }
        };
        let checks = [check("left", "input", "F32", 2), check("offset", "prefetch", "I32", 0)];
        let checks = checks.into_iter().chain([check("state", "inout", "F32", 1)]);
        assert_eq!(
            generator.generate_definition(&mapped_function()).unwrap().to_string(),
            quote! {
                #[doc = #module_documentation]
                pub mod update {
                    #[allow(unused_imports)]
                    use super::*;

                    #[doc = #definition_documentation]
                    #[allow(clippy::cloned_ref_to_slice_refs, clippy::needless_question_mark)]
                    pub fn definition(
                        left: &kernels::__private::core::arrays::ArrayType,
                        offset: &kernels::__private::core::arrays::ArrayType,
                        state: &kernels::__private::core::arrays::ArrayType
                    ) -> ::core::result::Result<kernels::KernelDefinition, kernels::KernelError> {
                        #(#checks)*
                        let __kernel_output_type = kernels::__private::core::arrays::ArrayType::new_static(
                            kernels::__private::core::arrays::DataType::I32,
                            [kernels::__private::static_extent(left, 0usize)?]
                        );
                        let __kernel_operation = #operation;
                        let __kernel_left_metadata = left;
                        let __kernel_offset_metadata = offset;
                        let __kernel_state_metadata = state;
                        kernels::KernelDefinition::trace_with_prefetch(
                            __kernel_operation,
                            |(__kernel_references, __kernel_coordinates, __kernel_prefetched)| {
                                use kernels::__private::core::contexts::Context as _;
                                let __kernel_context = __kernel_references[0].context().clone();
                                let left = __kernel_references[0usize].clone();
                                let state = __kernel_references[1usize].clone();
                                let output = __kernel_references[2usize].clone();
                                #[allow(unused_variables)]
                                let i = __kernel_coordinates[0usize].clone();
                                #[allow(unused_variables)]
                                let k = __kernel_coordinates[1usize].clone();
                                #[allow(unused_variables)]
                                let offset = __kernel_prefetched[0usize].clone();
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
                kernels::__private::tiled_call(
                    &[left.clone(), indices.clone()],
                    __kernel_output_type,
                    ([8, 32]).to_vec(),
                    kernels::BoundaryPolicy::Masked,
                )?
            }
            .to_string(),
        );

        // Kernels without a grid, block windows, and `prefetch` parameters run one program over whole arrays.
        let mut generator = generator();
        generator.parameters[2].window = Window::WholeArray;
        assert_eq!(
            generator.generate_operation().unwrap().to_string(),
            quote!({
                let __kernel_parameters = ::std::vec![
                    kernels::whole_array_parameter(left.clone(), kernels::KernelParameterAccess::ReadOnly)?,
                    kernels::whole_array_parameter(indices.clone(), kernels::KernelParameterAccess::ReadOnly)?,
                    kernels::whole_array_parameter(__kernel_output_type, kernels::KernelParameterAccess::WriteOnly)?
                ];
                kernels::KernelCallOperation::new(kernels::Grid::new(::std::vec![])?, __kernel_parameters)?
            })
            .to_string(),
        );

        // Every other kernel constructs a mapped call over its grid.
        assert_eq!(
            mapped_generator().generate_operation().unwrap().to_string(),
            quote! {
                kernels::mapped_call(
                    kernels::Grid::new(::std::vec![
                        kernels::GridDimension::new(
                            kernels::__private::core::arrays::Dimension::Static(
                                kernels::__private::shape_div_ceil(kernels::__private::static_extent(left, 0usize)?, 8)?
                            ),
                            kernels::GridExecution::Parallel,
                        )
                        .with_name("i")?,
                        kernels::GridDimension::new(
                            kernels::__private::core::arrays::Dimension::Static(4),
                            kernels::GridExecution::Sequential,
                        )
                        .with_name("k")?
                    ])?,
                    ::std::vec![
                        (
                            left.clone(),
                            kernels::KernelParameterAccess::ReadOnly,
                            kernels::ParameterWindow::BlockIndices {
                                block_shape: ([8, 32]).to_vec(),
                                boundary: kernels::BoundaryPolicy::Masked,
                                indices: ::std::vec![
                                    kernels::MappingExpression::GridCoordinate(0usize),
                                    kernels::MappingExpression::GridCoordinate(1usize)
                                ],
                            }
                        ),
                        (
                            state.clone(),
                            kernels::KernelParameterAccess::ReadWrite,
                            kernels::ParameterWindow::ElementStarts {
                                block_shape: ([8]).to_vec(),
                                boundary: kernels::BoundaryPolicy::InBounds,
                                starts: ::std::vec![
                                    kernels::MappingExpression::Prefetch(0usize)
                                        + kernels::MappingExpression::GridCoordinate(0usize)
                                            * kernels::MappingExpression::Constant(8)
                                ],
                            }
                        ),
                        (
                            __kernel_output_type,
                            kernels::KernelParameterAccess::WriteOnly,
                            kernels::ParameterWindow::BlockIndices {
                                block_shape: ([8]).to_vec(),
                                boundary: kernels::BoundaryPolicy::InBounds,
                                indices: ::std::vec![kernels::MappingExpression::GridCoordinate(0usize)],
                            }
                        )
                    ],
                    ::std::vec![offset.clone()],
                )?
            }
            .to_string(),
        );
    }

    #[test]
    fn test_kernel_code_generator_generate_mapping() {
        let generator = mapped_generator();
        let mapping = |expression| {
            generator
                .generate_mapping(&expression)
                .map(|mapping| mapping.to_string())
                .map_err(|error| error.to_string())
        };

        // Grid axes, prefetched values, and constants combine through operators and functions.
        assert_eq!(
            mapping(parse_quote!((k % 2).max(offset).saturating_sub(i / 2).min(left.shape()[1]))),
            Ok(quote! {
                (kernels::MappingExpression::GridCoordinate(1usize) % kernels::MappingExpression::Constant(2))
                    .max(kernels::MappingExpression::Prefetch(0usize))
                    .saturating_sub(
                        kernels::MappingExpression::GridCoordinate(0usize) / kernels::MappingExpression::Constant(2)
                    )
                    .min(kernels::MappingExpression::Constant(kernels::__private::static_extent(left, 1usize)?))
            }
            .to_string()),
        );
        assert_eq!(
            mapping(invisible_group(quote!(i - 1))),
            Ok(quote! {
                (kernels::MappingExpression::GridCoordinate(0usize) - kernels::MappingExpression::Constant(1))
            }
            .to_string()),
        );

        let error = |expression| mapping(expression).unwrap_err();
        assert_eq!(
            error(parse_quote!(j)),
            "unknown grid axis or `prefetch` parameter `j` in kernel mapping expression",
        );
        assert_eq!(
            error(parse_quote!(state + 1)),
            "unknown grid axis or `prefetch` parameter `state` in kernel mapping expression",
        );
        assert_eq!(error(parse_quote!(self::i)), "unsupported kernel mapping expression");
        assert_eq!(error(parse_quote!(i << 1)), "unsupported kernel mapping operator");
        assert_eq!(error(parse_quote!(i.pow(2))), "unsupported shape metadata expression");
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
            Ok(quote!([
                kernels::__private::shape_div_ceil(kernels::__private::static_extent(left, 1usize)?, 32)?,
                (8)
            ])
            .to_string()),
        );
        assert_eq!(
            shape(parse_quote!(left.shape()[0] == indices.shape()[0] && true), true),
            Ok(quote! {
                kernels::__private::static_extent(__kernel_left_metadata, 0usize)
                    .map_err(kernels::__private::core::programs::ProgramError::custom)?
                    == kernels::__private::static_extent(__kernel_indices_metadata, 0usize)
                        .map_err(kernels::__private::core::programs::ProgramError::custom)?
                    && true
            }
            .to_string()),
        );
        assert_eq!(shape(invisible_group(quote!(32)), false), Ok(quote!((32)).to_string()));

        // Shape metadata cannot run host code or read axes outside the declared argument ranks.
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

        // The right operand of a compound assignment keeps its grouping, whatever the precedence of its operator.
        assert_eq!(
            statement(parse_quote!(accumulator *= left.load() + left.load();)),
            statement(parse_quote!(accumulator = accumulator * (left.load() + left.load());)),
        );
        assert_ne!(
            statement(parse_quote!(accumulator *= left.load() + left.load();)),
            statement(parse_quote!(accumulator = accumulator * left.load() + left.load();)),
        );

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
        untiled.parameters[2].window = Window::WholeArray;
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

        // Slice declarations generate no code either.
        assert!(
            generator
                .generate_statement(&parse_quote!(let head = left.slice([0..1, ..]);), &mut scope)
                .unwrap()
                .is_empty()
        );
        assert!(scope.slice(&format_ident!("head")).is_some());

        let error = |statement| generator.generate_statement(&statement, &mut scope.clone()).unwrap_err().to_string();
        assert_eq!(error(parse_quote!(let row = left.load();)), "kernel locals cannot shadow an existing binding");
        assert_eq!(error(parse_quote!(let tiles = left.load();)), "kernel locals cannot shadow an existing binding");
        assert_eq!(error(parse_quote!(let head = left.load();)), "kernel locals cannot shadow an existing binding");
        assert_eq!(error(parse_quote!(let mut tail = left.slice([0..1, ..]);)), "kernel slices cannot be mutable",);
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
                    kernels::__private::static_extent(__kernel_left_metadata, 1usize)
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
    fn test_kernel_code_generator_extract_slice_view() {
        let generator = generator();
        let scope = scope();
        let view = |expression| {
            let syn::Expr::MethodCall(call) = expression else { panic!("expected a method call") };
            generator.extract_slice_view(&call, &scope)
        };
        let SliceView { source, ranges } = view(parse_quote!(left.slice([.., 1..left.shape()[1]]))).unwrap();
        assert_eq!(source, "left");
        assert_eq!(
            ranges.iter().map(TokenStream::to_string).collect::<Vec<_>>(),
            vec![
                quote!(::core::option::Option::None).to_string(),
                quote! {
                    ::core::option::Option::Some(
                        1..kernels::__private::static_extent(__kernel_left_metadata, 1usize)
                            .map_err(kernels::__private::core::programs::ProgramError::custom)?
                    )
                }
                .to_string(),
            ],
        );

        // Slices take one range or `..` per axis of a named binding.
        let error = |expression| view(expression).err().unwrap().to_string();
        assert_eq!(
            error(parse_quote!(missing.slice([..]))),
            "expected `reference.slice([ranges])` with a named reference",
        );
        assert_eq!(
            error(parse_quote!(left.slice([..], [..]))),
            "expected `reference.slice([ranges])` with a named reference",
        );
        assert_eq!(error(parse_quote!(left.slice(ranges))), "kernel slices require an array of ranges");
        assert_eq!(error(parse_quote!(left.slice([0..=1]))), "kernel slice ranges must be `start..end` or `..`");
        assert_eq!(error(parse_quote!(left.slice([1..]))), "kernel slice ranges must be `start..end` or `..`");
        assert_eq!(error(parse_quote!(left.slice([0..host()]))), "unsupported shape metadata expression");
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
                        0..kernels::__private::shape_div_ceil(
                            kernels::__private::static_extent(__kernel_left_metadata, 1usize)
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
        scope
            .slices
            .push((format_ident!("head"), SliceView { source: format_ident!("left"), ranges: vec![] }));
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

        // Operators and casts are dispatched to their own lowerings.
        assert_eq!(
            value(parse_quote!(left < accumulator)),
            Ok(generator.generate_comparison(&parse_quote!(left < accumulator), &scope).unwrap().to_string()),
        );
        assert_eq!(
            value(parse_quote!(left % accumulator)),
            Ok(generator.generate_arithmetic(&parse_quote!(left % accumulator), &scope).unwrap().to_string()),
        );
        assert_eq!(
            value(parse_quote!(-left)),
            Ok(generator.generate_unary(&parse_quote!(-left), &scope).unwrap().to_string()),
        );
        assert_eq!(
            value(parse_quote!(left as f32)),
            Ok(generator.generate_cast(&parse_quote!(left as f32), &scope).unwrap().to_string()),
        );

        // Reserved names, views outside their accesses, and host Rust expressions are rejected.
        let error = |expression| value(expression).unwrap_err().to_string();
        assert_eq!(error(parse_quote!(__kernel_context)), "kernel bodies cannot use reserved `__kernel_` names");
        assert_eq!(error(parse_quote!(tiles)), "tile views can only be loaded");
        assert_eq!(
            error(parse_quote!(head)),
            "kernel slices only support `load`, `store`, `accumulate`, and `atomic_accumulate`",
        );
        assert_eq!(
            error(parse_quote!(loop {})),
            "unsupported kernel expression; control flow requires staged lowering and cannot execute as host Rust",
        );
        assert_eq!(
            error(parse_quote!(left?)),
            "unsupported kernel expression; control flow requires staged lowering and cannot execute as host Rust",
        );
    }

    #[test]
    fn test_kernel_code_generator_generate_method_call() {
        let generator = mapped_generator();
        let scope = mapped_scope();
        let call = |expression| {
            generator
                .generate_unscoped_value(&expression, &scope)
                .map(|call| call.to_string())
                .map_err(|error| error.to_string())
        };
        let left = scoped(quote!(left.clone()));
        let state = scoped(quote!(state.clone()));
        let output = scoped(quote!(output.clone()));
        let offset = scoped(quote!(offset.clone()));

        // Reference accesses, where masked accesses of a parameter take its element type and accumulation takes the
        // boundary policy of its window.
        assert_eq!(
            call(parse_quote!(state.load())),
            Ok(quote!(kernels::__private::core::operations::ReferenceRead::read(&(#state))?).to_string()),
        );
        assert_eq!(
            call(parse_quote!(left.load_or(-1.0))),
            Ok(quote!(kernels::masked_load_or::<f32, _>(&__kernel_context, &(#left), -1.0)?).to_string()),
        );
        assert_eq!(
            call(parse_quote!(left.load_where(offset, 0.0))),
            Ok(quote!(kernels::masked_load::<f32, _>(&__kernel_context, &(#left), &(#offset), 0.0)?).to_string()),
        );
        assert_eq!(
            call(parse_quote!(offset.load_or(0i32))),
            Ok(quote!(kernels::masked_load_or(&__kernel_context, &(#offset), 0i32)?).to_string()),
        );
        assert_eq!(
            call(parse_quote!(output.store(state))),
            Ok(quote!(kernels::__private::core::operations::ReferenceWrite::write(&(#output), &(#state))?).to_string()),
        );
        assert_eq!(
            call(parse_quote!(output.store_where(state, offset))),
            Ok(quote!(kernels::masked_store(&__kernel_context, &(#output), &(#state), &(#offset))?).to_string()),
        );
        assert_eq!(
            call(parse_quote!(state.swap_where(output, offset, 1.5))),
            Ok(quote! {
                kernels::masked_swap::<f32, _>(&__kernel_context, &(#state), &(#output), &(#offset), 1.5)?
            }
            .to_string()),
        );
        assert_eq!(
            call(parse_quote!(left.window_mask())),
            Ok(quote!(kernels::window_mask(&__kernel_context, &(#left))?).to_string()),
        );
        assert_eq!(
            call(parse_quote!(left.accumulate(state))),
            Ok(quote! {
                kernels::accumulate(&__kernel_context, &(#left), &(#state), kernels::BoundaryPolicy::Masked)?
            }
            .to_string()),
        );
        assert_eq!(
            call(parse_quote!(offset.accumulate(state))),
            Ok(quote! {
                kernels::accumulate(&__kernel_context, &(#offset), &(#state), kernels::BoundaryPolicy::InBounds)?
            }
            .to_string()),
        );
        assert_eq!(
            call(parse_quote!(state.atomic_accumulate(output))),
            Ok(quote! {
                kernels::__private::core::operations::ReferenceAtomicAddUpdate::atomic_add_update(
                    &(#state),
                    &(#output)
                )?
            }
            .to_string()),
        );

        // Asynchronous copies and their completion.
        assert_eq!(
            call(parse_quote!(left.copy_to(state))),
            Ok(quote!(kernels::async_copy(&__kernel_context, &(#left), &(#state))?).to_string()),
        );
        assert_eq!(
            call(parse_quote!(offset.wait())),
            Ok(quote!(kernels::wait(&__kernel_context, &(#offset))?).to_string()),
        );

        // Matrix products, reductions, and elementwise math capabilities.
        assert_eq!(call(parse_quote!(left.dot(state))), Ok(quote!(kernels::dot(&(#left), &(#state))?).to_string()));
        let reduction = |kind: TokenStream| {
            quote! {
                __kernel_context.bind(
                    kernels::__private::core::arrays::ArrayIrOperation::from(
                        kernels::__private::core::arrays::ArrayOperation::Reduce(
                            kernels::__private::core::operations::ReduceOperation::new(
                                ::std::vec![0usize, 1usize],
                                kernels::__private::core::operations::ReductionKind::#kind,
                            ),
                        )
                    ),
                    ::std::vec![],
                    &[#left],
                )?.remove(0)
            }
            .to_string()
        };
        assert_eq!(call(parse_quote!(left.sum([0, 1]))), Ok(reduction(quote!(Sum))));
        assert_eq!(call(parse_quote!(left.reduce_sum([0, 1]))), Ok(reduction(quote!(Sum))));
        assert_eq!(call(parse_quote!(left.reduce_log_sum_exp([0, 1]))), Ok(reduction(quote!(LogSumExp))));
        assert_eq!(call(parse_quote!(left.reduce_all([0, 1]))), Ok(reduction(quote!(All))));
        assert_eq!(
            call(parse_quote!(left.exp_m1())),
            Ok(generator.generate_elementwise("ExpM1", vec![left.clone()]).to_string()),
        );
        assert_eq!(
            call(parse_quote!(left.log_add_exp(state))),
            Ok(generator.generate_elementwise("LogAddExp", vec![left.clone(), state]).to_string()),
        );

        // Calls on slices are slice accesses.
        assert_eq!(
            call(parse_quote!(left.slice([.., 0..8]).load())),
            Ok(generator
                .generate_slice_access(
                    &parse_quote!(left.slice([.., 0..8]).load()),
                    &SliceView {
                        source: format_ident!("left"),
                        ranges: vec![quote!(::core::option::Option::None), quote!(::core::option::Option::Some(0..8)),],
                    },
                    &scope,
                )
                .unwrap()
                .to_string()),
        );

        let error = |expression| call(expression).unwrap_err().to_string();
        assert_eq!(
            error(parse_quote!(left.load::<f32>())),
            "kernel capability calls do not accept explicit generic arguments",
        );
        assert_eq!(error(parse_quote!(left.softplus())), "unsupported kernel capability call");
        assert_eq!(error(parse_quote!(left.load(0))), "kernel capability `load` takes 0 arguments but received 1");
        assert_eq!(error(parse_quote!(left.max())), "kernel capability `max` takes 1 argument but received 0");
        assert_eq!(
            error(parse_quote!(left.load())),
            "`load` cannot read the padding of the masked window of `left`, so use `load_or(fill)` instead",
        );
        assert_eq!(error(parse_quote!(left.load_or(state))), "kernel fill values require a scalar literal");
        let mut half = mapped_generator();
        half.parameters[2].data_type = format_ident!("BF16");
        assert!(half.generate_unscoped_value(&parse_quote!(state.load()), &scope).is_ok());
        assert_eq!(
            half.generate_unscoped_value(&parse_quote!(state.load_or(0.0)), &scope).unwrap_err().to_string(),
            "kernel fill literals require a data type with a Rust scalar element type",
        );
        assert_eq!(
            error(parse_quote!(left.slice([.., 0..8]))),
            "kernel slices only support `load`, `store`, `accumulate`, and `atomic_accumulate`",
        );
        assert_eq!(error(parse_quote!(left.sum(axes))), "kernel reduction axes require an array of integer literals");
        assert_eq!(error(parse_quote!(left.reduce_max([axis]))), "kernel reduction axes require integer literals");
    }

    #[test]
    fn test_kernel_code_generator_generate_method_call_tiled_store() {
        // Stores into a tiled output write the tile of the current program, and stores elsewhere write references.
        let generator = generator();
        let scope = scope();
        let output = scoped(quote!(output.clone()));
        let left = scoped(quote!(left.clone()));
        let accumulator = scoped(quote!(accumulator.clone()));
        assert_eq!(
            generator
                .generate_unscoped_value(&parse_quote!(output.store(accumulator)), &scope)
                .unwrap()
                .to_string(),
            quote!(kernels::tile_store(&__kernel_context, &(#output), &(#accumulator))?).to_string(),
        );
        assert_eq!(
            generator
                .generate_unscoped_value(&parse_quote!(left.store(accumulator)), &scope)
                .unwrap()
                .to_string(),
            quote!(kernels::__private::core::operations::ReferenceWrite::write(&(#left), &(#accumulator))?).to_string(),
        );
    }

    #[test]
    fn test_kernel_code_generator_generate_slice_access() {
        let generator = generator();
        let scope = scope();
        let view = SliceView { source: format_ident!("left"), ranges: vec![quote!(::core::option::Option::None)] };
        let access = |expression| {
            let syn::Expr::MethodCall(call) = expression else { panic!("expected a method call") };
            generator
                .generate_slice_access(&call, &view, &scope)
                .map(|access| access.to_string())
                .map_err(|error| error.to_string())
        };
        let left = scoped(quote!(left.clone()));
        let reference = quote!(kernels::reference_slice(&(#left), &[::core::option::Option::None])?);
        let accumulator = scoped(quote!(accumulator.clone()));
        assert_eq!(
            access(parse_quote!(head.load())),
            Ok(quote!(kernels::__private::core::operations::ReferenceRead::read(&(#reference))?).to_string()),
        );
        assert_eq!(
            access(parse_quote!(head.store(accumulator))),
            Ok(quote! {
                kernels::__private::core::operations::ReferenceWrite::write(&(#reference), &(#accumulator))?
            }
            .to_string()),
        );
        assert_eq!(
            access(parse_quote!(head.accumulate(accumulator))),
            Ok(quote! {
                kernels::__private::core::operations::ReferenceAddUpdate::add_update(&(#reference), &(#accumulator))?
            }
            .to_string()),
        );
        assert_eq!(
            access(parse_quote!(head.atomic_accumulate(accumulator))),
            Ok(quote! {
                kernels::__private::core::operations::ReferenceAtomicAddUpdate::atomic_add_update(
                    &(#reference),
                    &(#accumulator)
                )?
            }
            .to_string()),
        );
        assert_eq!(
            access(parse_quote!(head.window_mask())).unwrap_err(),
            "kernel slices only support `load`, `store`, `accumulate`, and `atomic_accumulate`",
        );
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
        let one = scoped_literal(quote!(1));
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
    fn test_kernel_code_generator_generate_call() {
        let generator = generator();
        let scope = scope();
        let call = |expression| {
            generator
                .generate_unscoped_value(&expression, &scope)
                .map(|call| call.to_string())
                .map_err(|error| error.to_string())
        };
        let extent = quote! {
            kernels::__private::static_extent(__kernel_indices_metadata, 0usize)
                .map_err(kernels::__private::core::programs::ProgramError::custom)?
        };
        assert_eq!(
            call(parse_quote!(zeros::<Element>([8, indices.shape()[0]]))),
            Ok(quote!(kernels::zeros::<Element, _>(&__kernel_context, [8, #extent])?).to_string()),
        );
        assert_eq!(
            call(parse_quote!(ones::<f32>([8]))),
            Ok(quote!(kernels::ones::<f32, _>(&__kernel_context, [8])?).to_string()),
        );
        assert_eq!(
            call(parse_quote!(full([8], -2.0f32))),
            Ok(quote!(kernels::full(&__kernel_context, [8], -2.0f32)?).to_string()),
        );
        assert_eq!(
            call(parse_quote!(full::<f32>([8], 2.0))),
            Ok(quote!(kernels::full::<f32, _>(&__kernel_context, [8], 2.0)?).to_string()),
        );
        assert_eq!(
            call(parse_quote!(iota::<i32>(indices.shape()[0]))),
            Ok(quote!(kernels::iota::<i32, _>(&__kernel_context, [#extent], 0)?).to_string()),
        );
        assert_eq!(
            call(parse_quote!(scratch::<f32>([8]))),
            Ok(quote!(kernels::scratch::<f32, _>(&__kernel_context, [8], ::core::mem::align_of::<f32>())?).to_string()),
        );
        assert_eq!(
            call(parse_quote!(scratch_aligned::<f32>([8], 64))),
            Ok(quote!(kernels::scratch::<f32, _>(&__kernel_context, [8], 64)?).to_string()),
        );
        let indices = scoped(quote!(indices.clone()));
        let left = scoped(quote!(left.clone()));
        let accumulator = scoped(quote!(accumulator.clone()));
        assert_eq!(
            call(parse_quote!(select(indices, left, accumulator))),
            Ok(quote!(kernels::select(&__kernel_context, &(#indices), &(#left), &(#accumulator))?).to_string()),
        );

        let error = |expression| call(expression).unwrap_err().to_string();
        assert_eq!(error(parse_quote!((helpers)([8]))), "kernel helpers must be recognized named capabilities");
        assert_eq!(error(parse_quote!(helpers::zeros::<f32>([8]))), "unsupported kernel helper call");
        assert_eq!(error(parse_quote!(arange::<f32>(8))), "unsupported kernel helper call");
        assert_eq!(error(parse_quote!(zeros::<f32>([8], [8]))), "kernel `zeros` takes 1 argument but received 2");
        assert_eq!(error(parse_quote!(select(left))), "kernel `select` takes 3 arguments but received 1");
        assert_eq!(error(parse_quote!(zeros([8]))), "kernel zeros requires one explicit element type");
        assert_eq!(error(parse_quote!(zeros::<'static>([8]))), "kernel zeros requires one element type");
        assert_eq!(error(parse_quote!(scratch::<f32, f64>([8]))), "kernel scratch requires one element type");
        assert_eq!(
            error(parse_quote!(select::<bool>(indices, left, accumulator))),
            "kernel select does not take an element type",
        );
        assert_eq!(error(parse_quote!(full([8], left))), "kernel fill values require a scalar literal");
    }

    #[test]
    fn test_kernel_code_generator_generate_arithmetic() {
        let generator = generator();
        let arithmetic = |expression| {
            let syn::Expr::Binary(binary) = expression else { panic!("expected a binary expression") };
            generator
                .generate_arithmetic(&binary, &scope())
                .map(|arithmetic| arithmetic.to_string())
                .map_err(|error| error.to_string())
        };
        let left = scoped(quote!(left.clone()));
        let accumulator = scoped(quote!(accumulator.clone()));
        assert_eq!(
            arithmetic(parse_quote!(left - accumulator)),
            Ok(quote! {
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
            .to_string()),
        );
        for (expression, variant) in [
            (parse_quote!(left + accumulator), "Add"),
            (parse_quote!(left * accumulator), "Mul"),
            (parse_quote!(left / accumulator), "Div"),
            (parse_quote!(left % accumulator), "Rem"),
            (parse_quote!(left & accumulator), "And"),
            (parse_quote!(left | accumulator), "Or"),
            (parse_quote!(left ^ accumulator), "Xor"),
        ] {
            assert_eq!(
                arithmetic(expression),
                Ok(generator.generate_elementwise(variant, vec![left.clone(), accumulator.clone()]).to_string()),
            );
        }

        // Short-circuiting operators are rejected, because both sides of a staged predicate are evaluated.
        assert_eq!(
            arithmetic(parse_quote!(left && accumulator)).unwrap_err(),
            "use `&` instead of `&&`, because kernel predicates are staged values",
        );
        assert_eq!(
            arithmetic(parse_quote!(left || accumulator)).unwrap_err(),
            "use `|` instead of `||`, because kernel predicates are staged values",
        );
        assert_eq!(
            arithmetic(parse_quote!(left << accumulator)).unwrap_err(),
            "unsupported kernel arithmetic operator",
        );
    }

    #[test]
    fn test_kernel_code_generator_generate_comparison() {
        let generator = mapped_generator();
        let comparison = |expression| {
            let syn::Expr::Binary(binary) = expression else { panic!("expected a binary expression") };
            generator
                .generate_comparison(&binary, &mapped_scope())
                .map(|comparison| comparison.to_string())
                .map_err(|error| error.to_string())
        };
        let state = scoped(quote!(state.clone()));
        let k = scoped(quote!(k.clone()));
        let zero = scoped_literal(quote!(0));
        assert_eq!(
            comparison(parse_quote!(state >= 0)),
            Ok(quote! {
                kernels::compare(
                    &__kernel_context,
                    &(#state),
                    &(#zero),
                    kernels::__private::core::operations::ComparisonDirection::GreaterThanOrEqual,
                )?
            }
            .to_string()),
        );

        // A grid axis compared with an integer literal compares a dimension with a constant, and the direction is
        // mirrored when the literal is on the left.
        let dimension = |direction: TokenStream, constant: usize| {
            quote! {
                kernels::compare_dimension(
                    &__kernel_context,
                    &(#k),
                    kernels::__private::core::operations::ComparisonDirection::#direction,
                    #constant,
                )?
            }
            .to_string()
        };
        assert_eq!(comparison(parse_quote!(k == 0)), Ok(dimension(quote!(Equal), 0)));
        assert_eq!(comparison(parse_quote!(k != 1)), Ok(dimension(quote!(NotEqual), 1)));
        assert_eq!(comparison(parse_quote!(k < 2)), Ok(dimension(quote!(LessThan), 2)));
        assert_eq!(comparison(parse_quote!(2 < k)), Ok(dimension(quote!(GreaterThan), 2)));
        assert_eq!(comparison(parse_quote!(3 <= k)), Ok(dimension(quote!(GreaterThanOrEqual), 3)));
        assert_eq!(comparison(parse_quote!(3 > k)), Ok(dimension(quote!(LessThan), 3)));
        assert_eq!(comparison(parse_quote!(3usize >= k)), Ok(dimension(quote!(LessThanOrEqual), 3)));
        let i = scoped(quote!(i.clone()));
        assert_eq!(
            comparison(parse_quote!(k == i)),
            Ok(quote! {
                kernels::compare(
                    &__kernel_context,
                    &(#k),
                    &(#i),
                    kernels::__private::core::operations::ComparisonDirection::Equal,
                )?
            }
            .to_string()),
        );
        assert_eq!(comparison(parse_quote!(k + i)).unwrap_err(), "unsupported kernel comparison operator");
    }

    #[test]
    fn test_kernel_code_generator_generate_unary() {
        let generator = generator();
        let unary = |expression| {
            let syn::Expr::Unary(unary) = expression else { panic!("expected a unary expression") };
            generator
                .generate_unary(&unary, &scope())
                .map(|unary| unary.to_string())
                .map_err(|error| error.to_string())
        };
        let left = scoped(quote!(left.clone()));
        assert_eq!(
            unary(parse_quote!(-left)),
            Ok(generator.generate_elementwise("Neg", vec![left.clone()]).to_string()),
        );
        assert_eq!(unary(parse_quote!(!left)), Ok(generator.generate_elementwise("Not", vec![left]).to_string()));
        assert_eq!(unary(parse_quote!(*left)).unwrap_err(), "unsupported kernel unary operator");
    }

    #[test]
    fn test_kernel_code_generator_generate_cast() {
        let generator = generator();
        let cast = |expression| {
            let syn::Expr::Cast(cast) = expression else { panic!("expected a cast") };
            generator
                .generate_cast(&cast, &scope())
                .map(|cast| cast.to_string())
                .map_err(|error| error.to_string())
        };
        let left = scoped(quote!(left.clone()));
        assert_eq!(
            cast(parse_quote!(left as bool)),
            Ok(quote! {
                kernels::__private::core::operations::ConvertElementType::convert_element_type(
                    &(#left),
                    kernels::__private::core::arrays::DataType::Boolean,
                )?
            }
            .to_string()),
        );
        assert_eq!(cast(parse_quote!(left as usize)).unwrap_err(), "kernel casts require a Rust scalar element type");
        assert_eq!(
            cast(parse_quote!(left as std::f32)).unwrap_err(),
            "kernel casts require a Rust scalar element type",
        );
    }

    #[test]
    fn test_kernel_code_generator_generate_elementwise() {
        assert_eq!(
            generator().generate_elementwise("Atan2", vec![quote!(left), quote!(right)]).to_string(),
            quote! {
                __kernel_context.bind(
                    kernels::__private::core::arrays::ArrayIrOperation::from(
                        kernels::__private::core::arrays::ArrayOperation::Atan2(
                            kernels::__private::core::operations::Atan2Operation::new(),
                        )
                    ),
                    ::std::vec![],
                    &[left, right],
                )?.remove(0)
            }
            .to_string(),
        );
    }

    #[test]
    fn test_kernel_code_generator_generate_provenance_scope() {
        assert_eq!(generator().generate_provenance_scope(Span::call_site()).to_string(), provenance().to_string());
    }

    #[test]
    fn test_parameter_is_result() {
        assert_eq!(
            mapped_generator().parameters.iter().map(Parameter::is_result).collect::<Vec<_>>(),
            vec![false, false, true, true],
        );
    }

    #[test]
    fn test_parameter_boundary() {
        assert_eq!(
            mapped_generator().parameters.iter().map(Parameter::boundary).collect::<Vec<_>>(),
            vec![Boundary::Masked, Boundary::InBounds, Boundary::InBounds, Boundary::InBounds],
        );
        assert_eq!(generator().parameters[2].boundary(), Boundary::Masked);
    }

    #[test]
    fn test_parameter_element_type() {
        let parameter = |data_type| Parameter {
            name: format_ident!("input"),
            access: Access::Input,
            data_type: format_ident!("{data_type}"),
            shape: ShapeAnnotation::Rank(1),
            window: Window::WholeArray,
        };
        assert_eq!(parameter("F32").element_type().unwrap().to_string(), "f32");
        assert_eq!(parameter("Boolean").element_type().unwrap().to_string(), "bool");
        assert_eq!(
            parameter("BF16").element_type().unwrap_err().to_string(),
            "kernel fill literals require a data type with a Rust scalar element type",
        );
    }

    #[test]
    fn test_access_generate() {
        let kernels = parse_quote!(kernels);
        assert_eq!(
            [Access::Input, Access::Output, Access::Inout, Access::Prefetch]
                .map(|access| access.generate(&kernels).to_string()),
            [
                quote!(kernels::KernelParameterAccess::ReadOnly).to_string(),
                quote!(kernels::KernelParameterAccess::WriteOnly).to_string(),
                quote!(kernels::KernelParameterAccess::ReadWrite).to_string(),
                quote!(kernels::KernelParameterAccess::ReadOnly).to_string(),
            ],
        );
    }

    #[test]
    fn test_boundary_generate() {
        let kernels = parse_quote!(kernels);
        assert_eq!(
            Boundary::Masked.generate(&kernels).to_string(),
            quote!(kernels::BoundaryPolicy::Masked).to_string(),
        );
        assert_eq!(
            Boundary::InBounds.generate(&kernels).to_string(),
            quote!(kernels::BoundaryPolicy::InBounds).to_string(),
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
    fn test_scope_slice() {
        let mut scope = scope();
        scope
            .slices
            .push((format_ident!("head"), SliceView { source: format_ident!("left"), ranges: vec![] }));
        assert_eq!(scope.slice(&format_ident!("head")).unwrap().source, "left");
        assert!(scope.slice(&format_ident!("left")).is_none());
    }

    #[test]
    fn test_scope_validate_new_name() {
        let mut scope = scope();
        scope
            .slices
            .push((format_ident!("head"), SliceView { source: format_ident!("left"), ranges: vec![] }));
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
        assert_eq!(
            scope.validate_new_name(&pattern(quote!(head)), "locals", true).unwrap_err().to_string(),
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
    fn test_method_arity() {
        assert_eq!(method_arity("load"), Some(0));
        assert_eq!(method_arity("tanh"), Some(0));
        assert_eq!(method_arity("copy_to"), Some(1));
        assert_eq!(method_arity("pow"), Some(1));
        assert_eq!(method_arity("reduce_any"), Some(1));
        assert_eq!(method_arity("store_where"), Some(2));
        assert_eq!(method_arity("swap_where"), Some(3));
        assert_eq!(method_arity("tile_index"), None);
        assert_eq!(method_arity("neg"), None);
    }

    #[test]
    fn test_arguments() {
        assert_eq!(arguments(0), "0 arguments");
        assert_eq!(arguments(1), "1 argument");
        assert_eq!(arguments(3), "3 arguments");
    }

    #[test]
    fn test_is_scalar_literal() {
        assert!(is_scalar_literal(&parse_quote!(1.5f32)));
        assert!(is_scalar_literal(&parse_quote!(-1)));
        assert!(is_scalar_literal(&invisible_group(quote!(true))));
        assert!(!is_scalar_literal(&parse_quote!(-left)));
        assert!(!is_scalar_literal(&parse_quote!([1])));
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
