use proc_macro::TokenStream;

mod capabilities;
mod diagrams;
mod helpers;
mod operations;
mod parameters;

use capabilities::CapabilityCodeGenerator;
use diagrams::DiagramRenderer;
use operations::OperationCodeGenerator;
use parameters::ParameterizedCodeGenerator;

/// Generates a [`Parameter`] implementation for the container this macro is applied on.
#[proc_macro_derive(Parameter)]
pub fn derive_parameter(input: TokenStream) -> TokenStream {
    ParameterizedCodeGenerator::generate_parameter_impl(input)
}

/// Generates a [`Parameterized`] implementation for the container this macro is applied on.
#[proc_macro_derive(Parameterized, attributes(ryft))]
pub fn derive_parameterized(input: TokenStream) -> TokenStream {
    ParameterizedCodeGenerator::generate_parameterized_impl(input)
}

/// Generates an [`Operation`] implementation for the enum this macro is applied on, together with the implementations
/// through which the enum dispatches every other operation contract to the payload of its active variant.
///
/// Every variant of the enum must wrap exactly one operation payload, which may be stored directly or boxed, and
/// the enum then behaves like whichever payload it holds. The macro always generates the `Display`, `Operation`,
/// `InterpretableOperation`, `PartiallyEvaluatableOperation`, and `OperationPayloadProjection` implementations, one
/// `OperationProjection` implementation per projected member variant, and the owned `From<Payload>` and borrowed
/// `TryFrom<&Enum>` payload conversions. Unless a variant stores a native `GuardOperation`, it also generates an
/// `OperationProvider` implementation that rejects guard requests. The payload-delegating `PartialEq`, `Eq`, and
/// `Hash` implementations, as well as the reference discharge, batching, differentiation, and transposition
/// dispatchers, are generated only when they are selected through `#[ryft(dispatch(...))]`.
///
/// The macro supports the following optional `#[ryft(...)]` attributes:
///
/// ```text
/// // Enum-level attributes:
/// #[ryft(crate = "...")]                    // Path through which generated code names `ryft` (by default, `ryft`).
/// #[ryft(type(T), constant(C))]             // Operation type and stored constant type, which are otherwise inferred.
/// #[ryft(members(U [, structural(S)]...))]  // Member universes of the operation family.
/// #[ryft(dispatch(identity, discharge, batching, differentiation, transposition))]
///
/// // Variant-level attributes:
/// #[ryft(projected(U [, structural]))]      // Every input and output of the instruction belongs to `U`.
/// #[ryft(mixed(U [, structural]))]          // The instruction crosses member universes, and `U` may be defaulted.
/// #[ryft(skip_from)]                        // Skips only the owned `From<Payload>` conversion of the variant.
/// ```
///
/// For example, the following enum derives an `Operation<Type = ArrayType>` implementation, along with the other
/// implementations listed above, for both of its variants:
///
/// ```rust,ignore
/// #[derive(Clone, Debug, Operation)]
/// enum BackendOperation<V: Value<Type = ArrayType>> {
///     Zero(ZeroOperation<ArrayType>),
///     Constant(ConstantOperation<V>),
/// }
/// ```
///
/// Refer to the documentation of the `Operation` trait in `ryft-core` for the canonical reference of this macro, which
/// covers the enum-level attributes, how the operation and constant types are inferred, the grammar of the variant
/// classes along with their boundary shapes and transform roles, the generated implementations and their contracts,
/// and the requirements and limitations of the macro.
#[proc_macro_derive(Operation, attributes(ryft))]
pub fn derive_operation(input: TokenStream) -> TokenStream {
    OperationCodeGenerator::generate_operation_impl(input)
}

/// Validates the conventions of the capability trait this attribute is applied on and, for every
/// `projection(Family => Member)` argument, generates the implementation through which the values
/// of the `Family` universe apply the capability by projecting onto their `Member` values.
///
/// The attribute supports the following optional arguments:
///
/// ```text
/// #[capability(crate = "...")]                 // Path through which generated code names `ryft` (by default, `ryft`).
/// #[capability(projection(Family => Member))]  // Implements the capability for `Family` values (may be repeated).
/// ```
///
/// For example, `#[capability(projection(ArrayIrType => ArrayType))]` implements a capability for array IR values by
/// projecting them onto their array members. The annotated trait is always emitted unchanged, including when the
/// attribute reports errors. Refer to the documentation of the `Capability` trait in `ryft-core` for the canonical
/// reference of this attribute, which covers the checked conventions, the generated projection implementations, and
/// the supported function signatures.
#[proc_macro_attribute]
pub fn capability(attributes: TokenStream, input: TokenStream) -> TokenStream {
    CapabilityCodeGenerator::generate_capability_impl(attributes.into(), input.into()).into()
}

/// Renders the [Mermaid](https://mermaid.js.org/) diagrams in the documentation of the item that it is applied on.
///
/// The attribute rewrites every ```` ```mermaid ```` code fence (or `~~~mermaid` fence) in the outer documentation
/// attributes of the item, which include its `///` comments, into a `<pre class="mermaid">` HTML block, and inserts a
/// small module script before the first diagram of the item that loads Mermaid and renders the diagrams of the page in
/// the browser. The script renders each page once even when several of its items carry diagrams. It selects the dark
/// Mermaid theme when the page uses a dark rustdoc theme (i.e., `dark` or `ayu`), or when the page uses no other
/// built-in rustdoc theme and the browser prefers a dark color scheme. The theme is selected when the diagrams are
/// rendered, so diagrams pick up a rustdoc theme change only after the page is reloaded. All other documentation lines,
/// including other code fences, are preserved verbatim and in order, and items without diagrams are returned unchanged.
/// The attribute takes no arguments, and it rejects unterminated diagrams, code fences nested inside diagrams, and
/// attributes other than literal documentation attributes placed between the lines of a diagram. When it reports
/// errors, the item is still emitted unchanged, so that its uses elsewhere do not report additional errors about
/// a missing item.
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
/// #[cfg_attr(doc, mermaid)]
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
/// because rustdoc only expands the attribute when it documents the crate again.
#[proc_macro_attribute]
pub fn mermaid(attributes: TokenStream, input: TokenStream) -> TokenStream {
    DiagramRenderer::render_diagrams(attributes.into(), input.into()).into()
}
