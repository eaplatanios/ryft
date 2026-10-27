use proc_macro::TokenStream;

mod capabilities;
mod diagrams;
mod helpers;
mod kernels;
mod operations;
mod parameters;

use parameters::CodeGenerator;

/// Generates a [`Parameter`] implementation for the container this macro is applied on.
#[proc_macro_derive(Parameter)]
pub fn derive_parameter(input: TokenStream) -> TokenStream {
    CodeGenerator::generate_parameter_impl(input)
}

/// Generates a [`Parameterized`] implementation for the container this macro is applied on.
#[proc_macro_derive(Parameterized, attributes(ryft))]
pub fn derive_parameterized(input: TokenStream) -> TokenStream {
    CodeGenerator::generate_parameterized_impl(input)
}

// TODO(eaplatanios): Review this.
/// Generates an operation enum's `Operation` contract and its selected semantic dispatchers.
///
/// The derivation applies to enums whose every variant wraps exactly one operation payload, and it makes the enum
/// behave like whichever payload it holds. It always generates the `Operation` implementation, the
/// `InterpretableOperation` and `PartiallyEvaluatableOperation` dispatchers, an `OperationProjection` implementation
/// per projected member variant, `Display`, and the owned `From<Payload>` plus borrowed `TryFrom<&Enum>` payload
/// conversions. Reference discharge, batching, forward-mode differentiation, and transposition dispatchers, as well as
/// the payload-delegating `PartialEq`, `Eq`, and `Hash` implementations of `dispatch(identity)`, are generated only
/// when selected.
///
/// The `#[ryft(...)]` attribute surface is the following, where enum-level attributes are all optional:
///
/// ```text
/// // Enum level:
/// #[ryft(crate = "...")]                     // Path used to name Ryft items; defaults to `ryft`.
/// #[ryft(type(T), constant(V))]              // Primary operation type and stored constant type; inferred otherwise.
/// #[ryft(members(U [, structural(S)]...))]   // Member universes the operation family declares.
/// #[ryft(dispatch(identity, discharge, batching, differentiation, transposition))]
///
/// // Variant level:
/// #[ryft(projected(U [, structural]))]       // Every input and output of the instruction belongs to `U`.
/// #[ryft(mixed(U [, structural]))]           // The instruction crosses member universes; `U` may be defaulted.
/// #[ryft(skip_from)]                         // Suppresses only this variant's owned `From<Payload>` conversion.
/// ```
///
/// For example, the following declaration derives an `Operation<Type = DataType>` implementation, the interpretation
/// and partial-evaluation dispatchers, and the payload conversions for both variants:
///
/// ```rust,ignore
/// #[derive(Clone, Debug, Operation)]
/// enum BackendOperation<V: Value<Type = DataType>> {
///     Zero(ZeroOperation<DataType>),
///     Constant(ConstantOperation<V>),
/// }
/// ```
///
/// Refer to the documentation of `ryft-core`'s `Operation` trait for the canonical derive reference, which covers the
/// enum-level attributes, operation-type and constant-type inference, the variant-class grammar with its boundary
/// shapes and transform roles, the generated implementations and their conversion contracts, and the macro's
/// requirements and limitations.
#[proc_macro_derive(Operation, attributes(ryft))]
pub fn derive_operation(input: TokenStream) -> TokenStream {
    operations::generate_operation_impl(input)
}

// TODO(eaplatanios): Review this.
/// Validates a capability trait and optionally generates the implementations through which heterogeneous values apply
/// it by projecting onto their members (e.g., `#[capability(projection(ArrayIrType => ArrayType))]`).
///
/// Refer to the documentation of `ryft-core`'s `Capability` trait for the canonical reference, which covers the checked
/// conventions, the generated projection implementations, the supported function signatures, and the `crate = "path"`
/// argument.
#[proc_macro_attribute]
pub fn capability(attributes: TokenStream, input: TokenStream) -> TokenStream {
    capabilities::expand(attributes.into(), input.into())
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

// TODO(eaplatanios): Review this.
/// Defines an experimental portable kernel using canonical `Array` signature annotations.
///
/// The current subset supports whole-array and padded tiled access, shape requirements, local arithmetic,
/// zero tiles, matrix multiplication, and `.sum([axes])` reductions with literal axes. It produces a functional
/// callable and a same-name module exposing `definition`. Bounded range loops and value-dependent conditionals stage
/// canonical control-flow regions.
/// Unbounded control flow and arbitrary host calls are rejected instead of executing as host Rust.
///
/// # Kernel namespace
///
/// The generated code names the kernel API through a single namespace path and reaches `ryft-core` items through that
/// namespace's hidden `__private::core` macro-support re-export. Like the derive macros' `ryft` default, the namespace
/// defaults to the `ryft` facade path `ryft::kernels`, so callers that depend on the facade pass nothing. Callers that
/// reach `ryft-kernels` through another path pass that path as `crate = "path"`:
///
/// ```rust,ignore
/// #[ryft::kernels::kernel]                                           // `ryft` facade (default namespace).
/// #[ryft::kernel]                                                    // `ryft` facade's top-level re-export.
/// #[ryft_kernels::kernel(crate = "ryft_kernels")]                    // Direct `ryft-kernels` dependency.
/// #[kernels_alias::kernel(crate = "kernels_alias")]                  // Renamed `ryft-kernels` dependency.
/// #[facade_alias::kernels::kernel(crate = "facade_alias::kernels")]  // Renamed `ryft` facade dependency.
/// ```
///
/// The macro cannot infer the namespace from the attribute path because a re-exported procedural attribute cannot see
/// the path through which it was invoked. The path always names the kernel namespace; a stale override that names the
/// core crate (e.g., `crate = "ryft_core"`) is not diagnosed by the macro and instead fails with unresolved-path errors
/// inside the generated code (the first one being that `__private` cannot be found in `ryft_core`). The attribute
/// also accepts `requires = expression`, a shape requirement over the input annotations that `definition` validates.
#[proc_macro_attribute]
pub fn kernel(attributes: TokenStream, input: TokenStream) -> TokenStream {
    kernels::expand(attributes.into(), input.into())
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Renders the [Mermaid](https://mermaid.js.org/) diagrams in the documentation of the item that it is applied on.
///
/// The attribute rewrites every ```` ```mermaid ```` code fence (or `~~~mermaid` fence) in the outer documentation
/// attributes of the item, which include its `///` comments, into a `<pre class="mermaid">` HTML block, and inserts a
/// small module script before the first diagram of the item that loads Mermaid and renders the diagrams of the page in
/// the browser. The script renders each page once even when several of its items carry diagrams, and it selects the
/// dark Mermaid theme when the browser prefers a dark color scheme. All other documentation lines, including other code
/// fences, are preserved verbatim and in order, and items without diagrams are returned unchanged. The attribute takes
/// no arguments, and it rejects unterminated diagrams, code fences nested inside diagrams, and attributes other than
/// literal documentation attributes placed between the lines of a diagram.
///
/// Diagram text reaches Mermaid as written, except that HTML character references in it are decoded first, so that
/// labels can contain HTML elements spelled with entities (e.g., `&lt;code&gt;` for a `code` element).
///
/// The attribute is meant to be applied only when building documentation, so that ordinary builds never run it:
///
/// ```rust,ignore
/// /// Graph with a single edge.
/// ///
/// /// ```mermaid
/// /// flowchart LR
/// ///   source --> target
/// /// ```
/// #[cfg_attr(doc, ryft_macros::mermaid)]
/// pub struct Graph;
/// ```
///
/// # Mermaid Bundle
///
/// When it expands an item with diagrams and the rustdoc output directory `<target>/doc` already exists, the attribute
/// writes a vendored Mermaid bundle to `<target>/doc/static.files.mermaid/mermaid.min.js` (unless that file already
/// exists with the same size), where `<target>` is `$CARGO_TARGET_DIR` or, when that variable is unset, `./target`
/// relative to the working directory of the compiler. Failing to write the bundle does not fail the build. The loader
/// script first requests this bundle relative to the rustdoc root path, which also works for documentation opened
/// directly from the file system, and then falls back to `https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js`.
/// If neither loads, each diagram is replaced by a note saying that it could not be rendered.
///
/// Because the output directory is derived only from `CARGO_TARGET_DIR` and `./target`, documentation written
/// elsewhere (e.g., by `cargo doc --target <triple>`, which writes to `<target>/<triple>/doc`, or into a target
/// directory configured through `--target-dir` or `build.target-dir`) does not receive the bundle and renders its
/// diagrams through the CDN. The same holds when the bundle is deleted while the documentation of the crate is fresh,
/// since rustdoc only expands the attribute when it documents the crate again.
#[proc_macro_attribute]
pub fn mermaid(attributes: TokenStream, input: TokenStream) -> TokenStream {
    diagrams::expand(attributes.into(), input.into())
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
