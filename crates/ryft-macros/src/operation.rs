use proc_macro2::TokenStream;
use quote::quote;
use syn::ext::IdentExt;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::visit_mut::VisitMut;

use crate::helpers::generics::GenericsHelpers;
use crate::helpers::hygiene::const_block;

/// Suffix that every operation payload type identifier is expected to end with and that is stripped to derive the
/// corresponding enum variant identifier (e.g., `AddOperation` yields the `Add` variant).
const OPERATION_SUFFIX: &str = "Operation";

/// Name of the [`Value`](ryft_core::Value) trait whose `Value<X>` bound is scanned to infer the primary type `X`.
const VALUE_TRAIT_NAME: &str = "Value";

/// Parsed `define_operation_types! { ... }` invocation.
///
/// This is the abstract syntax tree produced by parsing the input to the
/// [`define_operation_types!`](crate::define_operation_types) macro. It captures the enum name and generics declared
/// via the `type = ...` entry, the list of operation variants declared via the `variants = [ ... ]` entry, the
/// per-variant dispatch [`Property`] impls requested via the `properties = [ ... ]` entry, the optional secondary
/// [`LinearSpec`] enum declared via the `linear = { ... }` entry, and whether an
/// [`InterpretableOperation`](crate::define_operation_types) delegation impl was requested via either the bare
/// `interpretable` key or an `interpretable` `properties` entry.
struct OperationInput {
    /// Documentation attributes attached to the `define_operation_types!` invocation. These are forwarded onto the
    /// generated enum.
    documentation: Vec<syn::Attribute>,

    /// Identifier of the generated enum (e.g., `ScalarOperation`).
    name: syn::Ident,

    /// Generics of the generated enum, including any predicates folded in from a trailing `where { ... }` clause.
    generics: syn::Generics,

    /// Operation variants declared via the `variants = [ ... ]` entry, in declaration order.
    variants: Vec<OperationVariant>,

    /// Per-variant dispatch [`Property`] impls requested via the `properties = [ ... ]` entry, in declaration order.
    /// An `interpretable` entry is split out into [`OperationInput::interpretable`] rather than retained here.
    properties: Vec<Property>,

    /// Secondary linear-operation enum declared via the `linear = { ... }` entry, if present.
    linear: Option<LinearSpec>,

    /// The [`InterpretableOperation`](crate::define_operation_types) delegation impl request, if present — via the bare
    /// top-level `interpretable` key, an `interpretable<W> where { ... }` top-level key, or an `interpretable` entry
    /// inside `properties = [ ... ]`. [`None`] when no `interpretable` was requested. See [`InterpretableSpec`].
    interpretable: Option<InterpretableSpec>,

    /// Concrete primary-type pin requested via a `primary = <Type> [where { ... }]` entry, if present. See
    /// [`PrimarySpec`].
    primary: Option<PrimarySpec>,

    /// Variant identifiers declared self-referential via a `recursive = [ ... ]` entry, in declaration order. These are
    /// variants whose payload recurses back into this enum *through an intermediate struct* (so the macro's syntactic
    /// [`EnumSpec::is_self_referential`] detection cannot see it) — for example the linear `Scan`/`Condition` variants,
    /// whose dedicated operation structs hold a nested program over this very enum. Listing them here unions them into
    /// the self-referential set, so their auto per-variant dispatch bound is skipped and the obligation is discharged by
    /// the method body's body-check fixed point (see [`EnumSpec::property_header`]). A listed identifier naming no
    /// variant is reported as an error.
    recursive: Vec<syn::Ident>,
}

/// The value-type generic and extra where-clause predicates for a requested `interpretable` impl.
///
/// The generated [`InterpretableOperation`](crate::define_operation_types) impl is a delegation over a value type:
/// `impl<.., W> InterpretableOperation<Primary, W> for Enum where W: Typed<Primary>, [per-variant], [predicates]`. The
/// bare `interpretable` form leaves [`InterpretableSpec::value`] as [`None`] (a fresh hygienic value generic is used and
/// no extra predicates are added). The `interpretable<W> where { ... }` form names the value generic `W` so the caller
/// can constrain it — needed when a self-referential variant's interpret (whose per-variant bound the macro skips, e.g.
/// `CustomJvpOperation`) requires a structural bound like `Vec<W>: Parameterized<...>` that no other variant implies.
struct InterpretableSpec {
    /// Caller-named value-type generic (the `W` in `interpretable<W>`), or [`None`] for the bare `interpretable` form.
    value: Option<syn::Ident>,

    /// Extra impl where-clause predicates from an optional trailing `where { ... }` clause.
    predicates: Vec<syn::WherePredicate>,
}

/// Concrete primary-type pin requested via a `primary = <Type> [where { ... }]` entry of an
/// [`define_operation_types!`](crate::define_operation_types) invocation.
///
/// Most enums infer their primary type `T` directly from a concrete `Value<T>` bound (for example,
/// `V: Value<DataType>` pins `DataType`), so their generated [`Operation`](crate::define_operation_types) and
/// [`Display`] impls are concrete with no further input. An enum that is generic over its type parameter `T` but whose
/// operations are only meaningful at one concrete type (for example, `ArrayOperation<T, V, Extension>` with
/// `V: Value<T>`, whose variant payloads such as `DotOperation` implement only `Operation<ArrayType>`) cannot have a
/// working generic `Operation<T>` delegation. Such an enum supplies `primary = ArrayType` to pin the generated
/// `Operation`/`Display` impls at `T = ArrayType`: the enum declaration stays generic over `T`, the `From`/`TryFrom`
/// conversions stay generic, but the `Operation`/`Display` impls drop `T` from their impl generics and substitute
/// `T -> ArrayType` in the self type, the delegated payload types, and the where-clause. The optional trailing
/// `where { ... }` carries extra predicates those pinned impls require but that are not in the enum's own where-clause
/// (for example, `Extension: Operation<ArrayType>`, needed because the backend-extension variant delegates to it).
struct PrimarySpec {
    /// Concrete type the generated `Operation`/`Display` impls are pinned to (e.g., `ArrayType`).
    concrete: syn::Type,

    /// Extra impl where-clause predicates for the pinned `Operation`/`Display` impls (e.g.,
    /// `Extension: Operation<ArrayType>`).
    predicates: Vec<syn::WherePredicate>,
}

/// Secondary linear-operation enum declared via the `linear = { ... }` entry of an
/// [`define_operation_types!`](crate::define_operation_types) invocation.
///
/// A linear enum is generated with the same core items as the primary enum (the enum declaration plus the
/// [`Operation`](crate::define_operation_types), [`Display`], and per-variant [`From`]/[`TryFrom`] impls). It
/// additionally always receives a [`Property::Transposable`] impl because it is a linear operation type, and it
/// inherits the primary enum's [`Property::Batchable`] impl (if the primary requested one). Its generics and variants
/// are independent of the primary enum's; nothing is derived from the primary.
struct LinearSpec {
    /// Identifier of the generated linear enum (e.g., `LinearScalarOperation`).
    name: syn::Ident,

    /// Generics of the generated linear enum, including any predicates folded in from a trailing `where { ... }`
    /// clause.
    generics: syn::Generics,

    /// Operation variants declared via the linear enum's own `variants = [ ... ]` entry, in declaration order.
    variants: Vec<OperationVariant>,
}

/// Single operation variant declared in the `variants = [ ... ]` entry.
struct OperationVariant {
    /// Variant identifier derived from the payload type (e.g., `Add` for `AddOperation`).
    variant: syn::Ident,

    /// Full payload type as written in the `variants` list, including any surrounding `Box<...>` and generic arguments.
    /// This is the type stored inside the generated enum variant.
    payload_type: syn::Type,

    /// Operation type carried by the variant. For a `Box<Inner<...>>` payload this is `Inner<...>`; otherwise it is the
    /// same as [`OperationVariant::payload_type`]. This is the type used for the generated [`From`] source type and the
    /// [`TryFrom`] target reference type.
    operation_type: syn::Type,

    /// Whether [`OperationVariant::payload_type`] is a `Box<...>` wrapping [`OperationVariant::operation_type`].
    is_boxed: bool,
}

impl Parse for OperationInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let documentation = input
            .call(syn::Attribute::parse_outer)?
            .into_iter()
            .filter(|attr| attr.path().is_ident("doc"))
            .collect();

        let mut name = None;
        let mut generics = None;
        let mut variants = None;
        let mut properties = Vec::new();
        let mut linear = None;
        let mut interpretable = None;
        let mut primary = None;
        let mut recursive = Vec::new();

        while !input.is_empty() {
            // The entry keys (e.g., `type`) may be reserved keywords, so we parse them with `parse_any`.
            let key = input.call(syn::Ident::parse_any)?;
            match key.to_string().as_str() {
                "type" => {
                    input.parse::<syn::Token![=]>()?;
                    let (parsed_name, parsed_generics) = parse_type_entry(input)?;
                    name = Some(parsed_name);
                    generics = Some(parsed_generics);
                }
                "variants" => {
                    input.parse::<syn::Token![=]>()?;
                    variants = Some(parse_variants_entry(input)?);
                }
                "properties" => {
                    input.parse::<syn::Token![=]>()?;
                    parse_properties_entry(input, &mut properties, &mut interpretable)?;
                }
                "linear" => {
                    input.parse::<syn::Token![=]>()?;
                    linear = Some(parse_linear_entry(input)?);
                }
                // `interpretable` is accepted as a bare key (no `= value`); it may optionally carry a value-type generic
                // and where-clause (`interpretable<W> where { ... }`). It is otherwise equivalent to listing the bare
                // `interpretable` inside `properties = [ ... ]`.
                "interpretable" => interpretable = Some(parse_interpretable_entry(input)?),
                "primary" => {
                    input.parse::<syn::Token![=]>()?;
                    primary = Some(parse_primary_entry(input)?);
                }
                "recursive" => {
                    input.parse::<syn::Token![=]>()?;
                    recursive = parse_recursive_entry(input)?;
                }
                other => {
                    return Err(syn::Error::new_spanned(
                        &key,
                        format_args!(
                            "unknown 'define_operation_types!' entry '{other}'; supported entries are: 'type', \
                            'variants', 'properties', 'linear', 'interpretable', 'primary', and 'recursive'"
                        ),
                    ));
                }
            }
            if input.peek(syn::Token![,]) {
                input.parse::<syn::Token![,]>()?;
            } else {
                break;
            }
        }

        let name =
            name.ok_or_else(|| input.error("'define_operation_types!' requires a 'type = <name><generics>' entry"))?;
        let generics = generics
            .ok_or_else(|| input.error("'define_operation_types!' requires a 'type = <name><generics>' entry"))?;
        let variants =
            variants.ok_or_else(|| input.error("'define_operation_types!' requires a 'variants = [ ... ]' entry"))?;

        Ok(Self { documentation, name, generics, variants, properties, linear, interpretable, primary, recursive })
    }
}

/// Parses the right-hand side of a `recursive = [ <Variant>, ... ]` entry into the list of variant identifiers declared
/// self-referential. Each entry is a bare variant identifier (e.g., `Scan`), matched against the enum's variant names
/// during [`EnumSpec`] construction.
fn parse_recursive_entry(input: ParseStream) -> syn::Result<Vec<syn::Ident>> {
    let content;
    syn::bracketed!(content in input);
    Ok(content.parse_terminated(syn::Ident::parse, syn::Token![,])?.into_iter().collect())
}

/// Parses the right-hand side of a `primary = <Type> [where { <predicates> }]` entry into a [`PrimarySpec`]: a concrete
/// type to pin the generated `Operation`/`Display` impls to, optionally followed by a brace-delimited `where { ... }`
/// clause carrying extra impl where-clause predicates for those pinned impls.
fn parse_primary_entry(input: ParseStream) -> syn::Result<PrimarySpec> {
    let concrete = input.parse::<syn::Type>()?;
    let mut predicates = Vec::new();
    if input.peek(syn::Token![where]) {
        input.parse::<syn::Token![where]>()?;
        let content;
        syn::braced!(content in input);
        predicates = content.parse_terminated(syn::WherePredicate::parse, syn::Token![,])?.into_iter().collect();
    }
    Ok(PrimarySpec { concrete, predicates })
}

/// Parses the optional `<W>` value-type generic and optional trailing `where { <predicates> }` following a top-level
/// `interpretable` key into an [`InterpretableSpec`]. A bare `interpretable` (no `<...>`, no `where`) yields an empty
/// spec (a fresh hygienic value generic and no extra predicates).
fn parse_interpretable_entry(input: ParseStream) -> syn::Result<InterpretableSpec> {
    let value = if input.peek(syn::Token![<]) {
        input.parse::<syn::Token![<]>()?;
        let value = input.parse::<syn::Ident>()?;
        input.parse::<syn::Token![>]>()?;
        Some(value)
    } else {
        None
    };
    let mut predicates = Vec::new();
    if input.peek(syn::Token![where]) {
        input.parse::<syn::Token![where]>()?;
        let content;
        syn::braced!(content in input);
        predicates = content.parse_terminated(syn::WherePredicate::parse, syn::Token![,])?.into_iter().collect();
    }
    Ok(InterpretableSpec { value, predicates })
}

/// Parses the right-hand side of a `type = <name><generics>` entry, optionally followed by a brace-delimited
/// `where { <predicates> }` clause whose predicates are folded into the returned [`syn::Generics`].
fn parse_type_entry(input: ParseStream) -> syn::Result<(syn::Ident, syn::Generics)> {
    let name = input.parse::<syn::Ident>()?;
    let mut generics = input.parse::<syn::Generics>()?;
    fold_optional_where_braces(input, &mut generics)?;
    Ok((name, generics))
}

/// Folds an optional trailing brace-delimited `where { <predicates> }` clause's predicates into the where-clause of
/// the provided [`syn::Generics`]. Does nothing if the next token is not `where`. This brace-delimited form is used
/// (instead of an ordinary trailing where-clause) so that the predicate list has an unambiguous terminator inside the
/// macro's comma-separated entry grammar.
fn fold_optional_where_braces(input: ParseStream, generics: &mut syn::Generics) -> syn::Result<()> {
    if input.peek(syn::Token![where]) {
        input.parse::<syn::Token![where]>()?;
        let content;
        syn::braced!(content in input);
        let predicates = content.parse_terminated(syn::WherePredicate::parse, syn::Token![,])?;
        generics.make_where_clause().predicates.extend(predicates);
    }
    Ok(())
}

/// Parses the right-hand side of a `variants = [ ... ]` entry into a list of [`OperationVariant`]s.
fn parse_variants_entry(input: ParseStream) -> syn::Result<Vec<OperationVariant>> {
    let content;
    syn::bracketed!(content in input);
    Ok(content.parse_terminated(OperationVariant::parse, syn::Token![,])?.into_iter().collect())
}

/// Parses the right-hand side of a `properties = [ ... ]` entry, pushing each non-`interpretable` entry onto
/// `properties` and setting `interpretable` to `true` if an `interpretable` entry is present. An `interpretable` entry
/// is handled this way (rather than as a [`Property`]) because the generated impl takes no generic arguments and is
/// equivalent to the bare top-level `interpretable` key.
fn parse_properties_entry(
    input: ParseStream,
    properties: &mut Vec<Property>,
    interpretable: &mut Option<InterpretableSpec>,
) -> syn::Result<()> {
    let content;
    syn::bracketed!(content in input);
    for property in content.parse_terminated(Property::parse, syn::Token![,])? {
        match property {
            // The `properties = [ ... ]` form of `interpretable` is always bare (no value generic, no where-clause);
            // the `interpretable<W> where { ... }` form is only accepted as a top-level key.
            Property::Interpretable => *interpretable = Some(InterpretableSpec { value: None, predicates: Vec::new() }),
            property => properties.push(property),
        }
    }
    Ok(())
}

/// Parses the right-hand side of a `linear = { ... }` entry into a [`LinearSpec`]. The braces contain a nested
/// `type = <name><generics> [where { ... }]` entry and a `variants = [ ... ]` entry using the same grammar as the
/// primary enum; no other entries are accepted inside `linear = { ... }`.
fn parse_linear_entry(input: ParseStream) -> syn::Result<LinearSpec> {
    let content;
    syn::braced!(content in input);

    let mut name = None;
    let mut generics = None;
    let mut variants = None;
    while !content.is_empty() {
        let key = content.call(syn::Ident::parse_any)?;
        content.parse::<syn::Token![=]>()?;
        match key.to_string().as_str() {
            "type" => {
                let (parsed_name, parsed_generics) = parse_type_entry(&content)?;
                name = Some(parsed_name);
                generics = Some(parsed_generics);
            }
            "variants" => variants = Some(parse_variants_entry(&content)?),
            other => {
                return Err(syn::Error::new_spanned(
                    &key,
                    format_args!(
                        "unknown 'linear' entry '{other}'; the 'linear = {{ ... }}' block supports only 'type' and \
                        'variants'"
                    ),
                ));
            }
        }
        if content.peek(syn::Token![,]) {
            content.parse::<syn::Token![,]>()?;
        } else {
            break;
        }
    }

    let name = name.ok_or_else(|| content.error("'linear = { ... }' requires a 'type = <name><generics>' entry"))?;
    let generics =
        generics.ok_or_else(|| content.error("'linear = { ... }' requires a 'type = <name><generics>' entry"))?;
    let variants = variants.ok_or_else(|| content.error("'linear = { ... }' requires a 'variants = [ ... ]' entry"))?;
    Ok(LinearSpec { name, generics, variants })
}

/// A single per-variant dispatch [`Property`] requested via a `properties = [ ... ]` entry.
///
/// Each variant names a trait whose impl the macro emits for the generated enum as a uniform, exhaustive, per-variant
/// turbofish delegation. The carried [`syn::Ident`]s are the impl-level generic parameters introduced by the entry
/// (e.g., the `D` in `differentiable<D>`), and the carried [`syn::WherePredicate`]s are caller-supplied extra impl
/// where-clause predicates parsed from an optional trailing `where { ... }` clause. The macro always also auto-adds one
/// per-variant bound (`<Payload>: <Trait><...>`) so the delegation type-checks; the extra predicates cover intrinsic
/// structural bounds the macro cannot infer (for example, `D: DifferentiationContext<Type = ...>`).
enum Property {
    /// [`DifferentiableOperation`](crate::define_operation_types) delegation, written `differentiable<D>`. The single
    /// generic is the differentiation context type.
    Differentiable {
        /// Differentiation context generic parameter (the `D` in `differentiable<D>`).
        context: syn::Ident,

        /// Caller-supplied extra impl where-clause predicates from an optional trailing `where { ... }` clause.
        extra_predicates: Vec<syn::WherePredicate>,
    },

    /// [`BatchableOperation`](crate::define_operation_types) delegation, written `batchable<V, C>`. The generics are
    /// the batched value type and the batching context type.
    Batchable {
        /// Batched value generic parameter (the `V` in `batchable<V, C>`).
        value: syn::Ident,

        /// Batching context generic parameter (the `C` in `batchable<V, C>`).
        context: syn::Ident,

        /// Caller-supplied extra impl where-clause predicates from an optional trailing `where { ... }` clause.
        extra_predicates: Vec<syn::WherePredicate>,
    },

    /// [`TransposableOperation`](crate::define_operation_types) delegation, written `transposable<V, O>` or
    /// `transposable<V>`. The first generic is the cotangent value type. The second, optional, generic is the linear
    /// operation type the transpose stages into; when omitted, that operation type is pinned to the enum's own self type
    /// (`transposable<V>` ⇒ `TransposableOperation<T, V, Self>`). The self-pinned form is required when a variant's
    /// transpose stages operations back into this very enum (so the program-op type must be the enum itself) — for
    /// example the linear array operation, whose `Scan`/`Condition` rules stage nested linear programs over the enum.
    Transposable {
        /// Cotangent value generic parameter (the `V` in `transposable<V, O>` / `transposable<V>`).
        value: syn::Ident,

        /// Linear operation generic parameter (the `O` in `transposable<V, O>`), or [`None`] for the `transposable<V>`
        /// form that pins the staged operation type to the enum's own self type.
        operation: Option<syn::Ident>,

        /// Caller-supplied extra impl where-clause predicates from an optional trailing `where { ... }` clause.
        extra_predicates: Vec<syn::WherePredicate>,
    },

    /// [`InterpretableOperation`](crate::define_operation_types) delegation, written `interpretable`. Carries no
    /// generics because the generated impl introduces its own value generic. This is normalized into
    /// [`OperationInput::interpretable`] during parsing rather than retained as a [`Property`].
    Interpretable,
}

impl Parse for Property {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name = input.parse::<syn::Ident>()?;
        let generics = parse_property_generics(input)?;
        let extra_predicates = parse_property_extra_predicates(input)?;
        match name.to_string().as_str() {
            "differentiable" => {
                let [context] = exact_property_generics(&name, "differentiable", generics)?;
                Ok(Property::Differentiable { context, extra_predicates })
            }
            "batchable" => {
                let [value, context] = exact_property_generics(&name, "batchable", generics)?;
                Ok(Property::Batchable { value, context, extra_predicates })
            }
            "transposable" => {
                // `transposable<V>` pins the staged operation type to the enum's self type; `transposable<V, O>` takes
                // an explicit operation generic. Accept either one or two generics accordingly.
                let mut generics = generics.into_iter();
                let value = generics.next().ok_or_else(|| {
                    syn::Error::new_spanned(
                        &name,
                        "'transposable' property expects 'transposable<V>' or 'transposable<V, O>'",
                    )
                })?;
                let operation = generics.next();
                if generics.next().is_some() {
                    return Err(syn::Error::new_spanned(
                        &name,
                        "'transposable' property expects at most two generic arguments ('transposable<V, O>')",
                    ));
                }
                Ok(Property::Transposable { value, operation, extra_predicates })
            }
            "interpretable" => {
                if !generics.is_empty() {
                    return Err(syn::Error::new_spanned(
                        &name,
                        "'interpretable' property does not take generic arguments",
                    ));
                }
                if !extra_predicates.is_empty() {
                    return Err(syn::Error::new_spanned(
                        &name,
                        "'interpretable' property does not support a 'where { ... }' clause",
                    ));
                }
                Ok(Property::Interpretable)
            }
            other => Err(syn::Error::new_spanned(
                &name,
                format_args!(
                    "unknown 'define_operation_types!' property '{other}'; supported properties are: \
                    'differentiable<D>', 'batchable<V, C>', 'transposable<V, O>', and 'interpretable'"
                ),
            )),
        }
    }
}

/// Parses the angle-bracketed generic parameter list of a [`Property`] entry (e.g., the `<V, C>` in `batchable<V, C>`)
/// into a list of [`syn::Ident`]s. Returns an empty [`Vec`] if no `<...>` follows. Only plain identifiers are accepted
/// as property generics.
fn parse_property_generics(input: ParseStream) -> syn::Result<Vec<syn::Ident>> {
    if !input.peek(syn::Token![<]) {
        return Ok(Vec::new());
    }
    input.parse::<syn::Token![<]>()?;
    let generics = Punctuated::<syn::Ident, syn::Token![,]>::parse_separated_nonempty(input)?;
    input.parse::<syn::Token![>]>()?;
    Ok(generics.into_iter().collect())
}

/// Parses the optional trailing `where { <predicates> }` clause of a [`Property`] entry into a list of
/// [`syn::WherePredicate`]s. Returns an empty [`Vec`] if no `where` follows.
fn parse_property_extra_predicates(input: ParseStream) -> syn::Result<Vec<syn::WherePredicate>> {
    if !input.peek(syn::Token![where]) {
        return Ok(Vec::new());
    }
    input.parse::<syn::Token![where]>()?;
    let content;
    syn::braced!(content in input);
    Ok(content.parse_terminated(syn::WherePredicate::parse, syn::Token![,])?.into_iter().collect())
}

/// Converts a [`Property`]'s parsed generic identifiers into a fixed-size array, reporting an error spanned at `name`
/// if the count does not match. `property` is the property's spelling, used in the error message.
fn exact_property_generics<const N: usize>(
    name: &syn::Ident,
    property: &str,
    generics: Vec<syn::Ident>,
) -> syn::Result<[syn::Ident; N]> {
    let count = generics.len();
    generics.try_into().map_err(|_| {
        syn::Error::new_spanned(
            name,
            format_args!("'{property}' property expects {N} generic argument(s) but found {count}"),
        )
    })
}

impl OperationVariant {
    /// Parses one `variants = [ ... ]` entry: either a bare payload type (variant name derived from the payload) or an
    /// explicit `Variant = PayloadType` (variant name given verbatim — used when the desired variant name differs from
    /// the suffix-stripped payload, e.g. `Select = LinearSelectOperation<F>`, or when the payload does not end in
    /// [`OPERATION_SUFFIX`] at all, e.g. a bare generic `Recompute = O`).
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.peek(syn::Ident) && input.peek2(syn::Token![=]) {
            let variant = input.parse::<syn::Ident>()?;
            input.parse::<syn::Token![=]>()?;
            Self::new(Some(variant), input.parse::<syn::Type>()?)
        } else {
            Self::new(None, input.parse::<syn::Type>()?)
        }
    }

    /// Builds an [`OperationVariant`] from an optional explicit variant name and a payload [`syn::Type`].
    ///
    /// If the payload is `Box<Inner<...>>`, the operation type is `Inner<...>`; otherwise it equals the payload type.
    /// When `explicit_variant` is [`None`], the variant identifier is the operation type's last path segment with the
    /// trailing [`OPERATION_SUFFIX`] stripped (an error is reported if it does not end with that suffix); when it is
    /// [`Some`], that identifier is used verbatim (so the payload need not end in `Operation`).
    fn new(explicit_variant: Option<syn::Ident>, payload_type: syn::Type) -> syn::Result<Self> {
        let (operation_type, is_boxed) = match look_through_box(&payload_type) {
            Some(inner) => (inner.clone(), true),
            None => (payload_type.clone(), false),
        };
        let variant = match explicit_variant {
            Some(variant) => variant,
            None => {
                let segment = last_path_segment(&operation_type).ok_or_else(|| {
                    syn::Error::new_spanned(
                        &operation_type,
                        "expected a path type ending in 'Operation' (e.g., 'AddOperation'), or an explicit \
                        'Variant = PayloadType', as a 'define_operation_types!' variant",
                    )
                })?;
                let operation_name = segment.ident.to_string();
                let variant_name =
                    operation_name.strip_suffix(OPERATION_SUFFIX).filter(|stripped| !stripped.is_empty());
                let Some(variant_name) = variant_name else {
                    return Err(syn::Error::new_spanned(
                        &segment.ident,
                        format_args!(
                            "operation variant type '{operation_name}' must end in '{OPERATION_SUFFIX}' (or use an \
                            explicit 'Variant = PayloadType')"
                        ),
                    ));
                };
                syn::Ident::new(variant_name, segment.ident.span())
            }
        };
        Ok(Self { variant, payload_type, operation_type, is_boxed })
    }
}

/// Returns the inner type of a `Box<Inner>` type expression, or [`None`] if the provided type is not a path type whose
/// last segment is `Box` with exactly one type argument.
fn look_through_box(ty: &syn::Type) -> Option<&syn::Type> {
    let segment = last_path_segment(ty)?;
    if segment.ident != "Box" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return None;
    };
    match arguments.args.first() {
        Some(syn::GenericArgument::Type(inner)) if arguments.args.len() == 1 => Some(inner),
        _ => None,
    }
}

/// Returns the last [`syn::PathSegment`] of a path type expression (with no qualified self), or [`None`] otherwise.
fn last_path_segment(ty: &syn::Type) -> Option<&syn::PathSegment> {
    match ty {
        syn::Type::Path(syn::TypePath { qself: None, path }) => path.segments.last(),
        _ => None,
    }
}

/// Generates the implementation for a [`define_operation_types!`](crate::define_operation_types) invocation. This
/// generates the core items (the enum declaration plus the [`Operation`](crate::define_operation_types), [`Display`],
/// and per-variant [`From`]/[`TryFrom`] impls) for the primary enum and, if a `linear = { ... }` entry is present, for
/// the secondary linear enum. It then generates the requested per-variant dispatch [`Property`] impls and the
/// [`InterpretableOperation`](crate::define_operation_types) impl. Any errors encountered along the way are accumulated
/// and returned as a single [`compile_error!`] [`TokenStream`], mirroring the error-accumulation approach used by the
/// derive macros in [`crate::parameters`].
pub(crate) fn generate(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = syn::parse_macro_input!(input as OperationInput);
    let mut errors = Vec::new();

    let primary = EnumSpec::new(
        &input.name,
        &input.generics,
        &input.variants,
        input.primary.as_ref(),
        &input.recursive,
        &mut errors,
    );

    // Enum declarations are emitted at the call site (so the enums and their variants are nameable there); all trait
    // impls are emitted inside a hygienic `const` block so that their items do not pollute the surrounding scope.
    let primary_declaration = primary.declaration(&input.documentation);

    // The linear enum (if any) reuses the same per-enum core generation, then additionally gets an auto `transposable`
    // impl (it is a linear operation type) and inherits the primary enum's `batchable` property (if any).
    let (linear_declaration, linear_items) = match input.linear.as_ref() {
        Some(linear) => {
            let linear_spec = EnumSpec::new(&linear.name, &linear.generics, &linear.variants, None, &[], &mut errors);
            let declaration = linear_spec.declaration(&[]);
            let mut items = linear_spec.core_items();
            items.extend(linear_spec.transposable_implementation(&Property::transposable_auto(), &mut errors));
            for property in &input.properties {
                if matches!(property, Property::Batchable { .. }) {
                    items.extend(linear_spec.property_implementation(property, &mut errors));
                }
            }
            (Some(declaration), items)
        }
        None => (None, TokenStream::new()),
    };

    let core_items = primary.core_items();
    let property_items = input
        .properties
        .iter()
        .map(|property| primary.property_implementation(property, &mut errors))
        .collect::<TokenStream>();
    let interpretable_item = match input.interpretable.as_ref() {
        Some(interpretable) => primary.interpretable_implementation(interpretable),
        None => TokenStream::new(),
    };

    let code = const_block(quote! {
        #core_items
        #linear_items
        #property_items
        #interpretable_item
    });

    if let Some(error) = errors.into_iter().reduce(|mut combined, error| {
        combined.combine(error);
        combined
    }) {
        return error.into_compile_error().into();
    }

    quote!(#primary_declaration #linear_declaration #code).into()
}

/// Resolved generation context for a single operation enum (either the primary enum or the secondary `linear` enum).
///
/// An [`EnumSpec`] bundles the enum's identifier, generics, variants, and inferred primary type together with the
/// pre-split generic fragments used throughout code generation. It exposes one method per generated artifact so that
/// the primary enum and the `linear` enum share identical core generation: [`EnumSpec::declaration`] emits the enum,
/// [`EnumSpec::core_items`] emits the [`Operation`](crate::define_operation_types)/[`Display`]/[`From`]/[`TryFrom`]
/// impls, and the `*_implementation` methods emit the optional per-variant dispatch [`Property`] impls.
struct EnumSpec<'spec> {
    /// Enum identifier (e.g., `ScalarOperation`).
    name: &'spec syn::Ident,

    /// Enum generics with generic defaults preserved, used for the enum declaration's `type` position.
    generics: &'spec syn::Generics,

    /// Enum generics with generic defaults removed, used in `impl` headers (associated-type bindings reject defaults).
    implementation_generics: syn::Generics,

    /// Enum variants in declaration order.
    variants: &'spec [OperationVariant],

    /// Primary type `T` inferred from the enum's `Value<...>` bound, or a placeholder if inference failed.
    primary_type: syn::Type,

    /// Concrete primary-type pin for the generated `Operation`/`Display` impls, if a `primary = ...` entry was given.
    /// When present, those two impls are specialized at `primary_type = concrete` (see [`PrimaryPin`]); the enum
    /// declaration and the `From`/`TryFrom` conversions stay generic over the inferred `primary_type`.
    pin: Option<PrimaryPin>,

    /// Variant identifiers explicitly declared self-referential via the `recursive = [ ... ]` entry. These union with
    /// the syntactic [`EnumSpec::is_self_referential`] detection so that variants whose recursion is hidden inside an
    /// intermediate struct (e.g., the linear `Scan`/`Condition`) also have their auto per-variant dispatch bound skipped.
    recursive_variants: &'spec [syn::Ident],
}

/// Resolved concrete primary-type pin for an [`EnumSpec`] (see [`PrimarySpec`] for the user-facing entry). It records
/// the concrete type the `Operation`/`Display` impls are pinned to and the extra impl where-clause predicates those
/// pinned impls require. The generic parameter being replaced is the [`EnumSpec::primary_type`] itself.
struct PrimaryPin {
    /// Concrete type the generated `Operation`/`Display` impls are pinned to (e.g., `ArrayType`).
    concrete: syn::Type,

    /// Extra impl where-clause predicates for the pinned `Operation`/`Display` impls.
    predicates: Vec<syn::WherePredicate>,
}

impl<'spec> EnumSpec<'spec> {
    /// Builds an [`EnumSpec`] for the enum named `name` with the provided `generics` and `variants`. The primary type
    /// is inferred from the generics' `Value<...>` bound; if inference fails the error is pushed onto `errors` and an
    /// inconsequential placeholder primary type is used so that generation can continue and report all errors at once.
    fn new(
        name: &'spec syn::Ident,
        generics: &'spec syn::Generics,
        variants: &'spec [OperationVariant],
        primary_spec: Option<&PrimarySpec>,
        recursive_variants: &'spec [syn::Ident],
        errors: &mut Vec<syn::Error>,
    ) -> Self {
        let primary_type = match infer_primary_type(generics) {
            Ok(primary_type) => primary_type,
            Err(error) => {
                errors.push(error);
                syn::parse_quote!(__PrimaryType)
            }
        };
        // A `recursive = [ ... ]` identifier that names no variant is almost certainly a typo; report it rather than
        // silently ignoring it (which would leave the intended variant's auto bound un-skipped and surface as a far
        // more confusing E0275 from the generated impl).
        for recursive in recursive_variants {
            if !variants.iter().any(|variant| &variant.variant == recursive) {
                errors.push(syn::Error::new_spanned(
                    recursive,
                    format_args!("'recursive' entry '{recursive}' does not name a variant of '{name}'"),
                ));
            }
        }
        let implementation_generics = generics.without_defaults();
        let pin = primary_spec
            .map(|spec| PrimaryPin { concrete: spec.concrete.clone(), predicates: spec.predicates.clone() });
        Self { name, generics, implementation_generics, variants, primary_type, pin, recursive_variants }
    }

    /// Returns the `(impl_generics, type_generics, where_clause)` fragments for an `impl` header targeting this enum.
    /// The impl generics carry no defaults (as required when generic parameters back associated-type bindings), while
    /// the type generics and where-clause come from the original generics.
    fn impl_fragments(&self) -> (syn::ImplGenerics<'_>, syn::TypeGenerics<'_>, Option<&syn::WhereClause>) {
        let (implementation_generics, _, _) = self.implementation_generics.split_for_impl();
        let (_, type_generics, where_clause) = self.generics.split_for_impl();
        (implementation_generics, type_generics, where_clause)
    }

    /// Returns a clone of `ty` with every occurrence of the `Self` type replaced by this enum's concrete self type
    /// (`Name<...generics>`). The generated `From`/`TryFrom` conversions name the variant's operation type in the
    /// trait-reference position of their `impl` headers, where `Self` is not a valid spelling; variants whose operation
    /// type embeds the enum (e.g., `CustomJvpOperation<T, V, Self>`) therefore need `Self` resolved up front.
    fn self_substituted(&self, ty: &syn::Type) -> syn::Type {
        let name = self.name;
        let (_, type_generics, _) = self.generics.split_for_impl();
        let self_type: syn::Type = syn::parse_quote!(#name #type_generics);
        let mut substituted = ty.clone();
        SelfSubstitution { self_type }.visit_type_mut(&mut substituted);
        substituted
    }

    /// Generates this enum's declaration, preserving generic defaults and forwarding the provided documentation.
    fn declaration(&self, documentation: &[syn::Attribute]) -> TokenStream {
        let name = self.name;
        // Use the full generics (parameter bounds AND defaults) for the declaration: `split_for_impl`'s
        // `type_generics` strips defaults, which would drop e.g. an `F = C` default and break single-argument uses
        // such as `LinearScalarOperation<f64>`. `Generics`'s own `ToTokens` emits the angle-bracketed parameter list
        // with defaults; the where-clause is emitted separately.
        let generics = &self.generics;
        let where_clause = &self.generics.where_clause;
        let variants = self.variants.iter().map(|variant| {
            let variant_ident = &variant.variant;
            let payload_type = &variant.payload_type;
            quote!(#variant_ident(#payload_type))
        });
        quote! {
            #(#documentation)*
            #[derive(Clone, Debug)]
            pub enum #name #generics #where_clause {
                #(#variants,)*
            }
        }
    }

    /// Generates this enum's core trait impls: the [`Operation`](crate::define_operation_types) impl (with `name`,
    /// `infer_output_types`, and `render` all delegating to the held operation), the [`Display`] impl delegating to
    /// [`Operation::render`](crate::define_operation_types), and one [`From`]/[`TryFrom`] conversion pair per variant.
    fn core_items(&self) -> TokenStream {
        let operation_and_display = self.operation_and_display_impls();
        // `From`/`TryFrom` are not generated for variants whose payload is one of the enum's own generic parameters
        // (for example, a backend-extension `Extension(Extension)` or a recomputed-primal `Recompute(O)` variant): a
        // blanket `From<Extension>` would overlap the concrete per-variant `From`s, and a `TryFrom<&Enum> for &Extension`
        // would violate the orphan rule. Such variants are constructed and matched directly through the variant.
        let support_implementations = self
            .variants
            .iter()
            .filter(|variant| !self.is_generic_parameter_payload(variant))
            .map(|variant| self.support_implementation(variant));
        quote! {
            #operation_and_display
            #(#support_implementations)*
        }
    }

    /// Returns whether the variant's payload operation type is one of this enum's own generic type parameters (e.g., a
    /// backend-extension `Extension` or a recomputed-primal `O`). Such variants get no `From`/`TryFrom` conversions (see
    /// [`EnumSpec::core_items`]).
    fn is_generic_parameter_payload(&self, variant: &OperationVariant) -> bool {
        self.generics
            .type_params()
            .any(|parameter| type_is_ident(&variant.operation_type, &parameter.ident))
    }

    /// Returns whether the variant's payload operation type refers back to this enum — either through `Self` or by
    /// naming the enum directly (e.g., `CustomJvpOperation<DataType, V, Self>` or `ConditionOperation<T, V, Self>`), or
    /// because it was explicitly listed in the `recursive = [ ... ]` entry (for variants whose recursion is hidden
    /// inside an intermediate struct, e.g., the linear `Scan`/`Condition`). Such higher-order variants get no auto
    /// per-variant dispatch bound (see [`EnumSpec::property_header`]).
    fn is_self_referential(&self, variant: &OperationVariant) -> bool {
        if self.recursive_variants.iter().any(|recursive| recursive == &variant.variant) {
            return true;
        }
        let mut operation_type = self.self_substituted(&variant.operation_type);
        let mut detector = EnumNameDetector { name: self.name.clone(), found: false };
        detector.visit_type_mut(&mut operation_type);
        detector.found
    }

    /// Generates the [`Operation`](crate::define_operation_types) and [`Display`] impls for this enum. With no
    /// `primary = ...` entry these are generic over the inferred [`EnumSpec::primary_type`]; with a [`PrimaryPin`] they
    /// are specialized at `primary_type = concrete` — the inferred type parameter is dropped from the impl generics and
    /// replaced by the concrete type in the self type, the delegated payload operation types, and the where-clause
    /// (extended with the pin's extra predicates). The enum declaration and the `From`/`TryFrom` conversions are not
    /// affected and stay generic over `primary_type`.
    fn operation_and_display_impls(&self) -> TokenStream {
        // The impl generics, self type, delegated payload operation types, and the `Operation<...>` type argument are
        // all pin-aware: with no `primary = ...` entry they stay generic over `primary_type`; with a [`PrimaryPin`] the
        // shared [`EnumSpec::base_impl_generics`]/[`EnumSpec::impl_self_type`]/[`EnumSpec::pinned_operation_type`]
        // helpers specialize them at `primary_type = concrete`.
        let generics = self.base_impl_generics();
        let (implementation_generics, _, where_clause) = generics.split_for_impl();
        let self_type = self.impl_self_type();
        let operation_types: Vec<syn::Type> =
            self.variants.iter().map(|variant| self.pinned_operation_type(&variant.operation_type)).collect();
        let primary = self.pin.as_ref().map_or_else(|| self.primary_type.clone(), |pin| pin.concrete.clone());
        self.emit_operation_and_display(
            &quote!(#implementation_generics),
            &quote!(#self_type),
            &quote!(#where_clause),
            &primary,
            &operation_types,
        )
    }

    /// Emits the [`Operation`](crate::define_operation_types) and [`Display`] impls from pre-resolved fragments. The
    /// `operation_types` are the per-variant payload types used in the delegation turbofish, parallel to
    /// [`EnumSpec::variants`]; `primary` is the trait type argument (`Operation<#primary>`).
    fn emit_operation_and_display(
        &self,
        implementation_generics: &TokenStream,
        self_type: &TokenStream,
        where_clause: &TokenStream,
        primary: &syn::Type,
        operation_types: &[syn::Type],
    ) -> TokenStream {
        // Delegate `name`/`infer_output_types`/`render` through always-turbofish `<OperationType as Operation<T>>::…`
        // (the same form the property delegations use). Method-call syntax would be ambiguous for operation structs
        // that implement both `Operation<DataType>` directly and `Operation<ArrayType>` via the `ElementwiseOperation`
        // blanket, since nothing else pins the type parameter.
        let primary_operation = quote!(Operation<#primary>);
        let name_arms = self.core_arms(operation_types, &primary_operation, &quote!(name), &quote!());
        let infer_output_types_arms =
            self.core_arms(operation_types, &primary_operation, &quote!(infer_output_types), &quote!(input_types));
        let render_arms =
            self.core_arms(operation_types, &primary_operation, &quote!(render), &quote!(formatter, indentation));

        quote! {
            #[automatically_derived]
            impl #implementation_generics Operation<#primary> for #self_type #where_clause {
                fn name(&self) -> &'static str {
                    match self {
                        #(#name_arms,)*
                    }
                }

                fn infer_output_types(
                    &self,
                    input_types: &[#primary],
                ) -> ::core::result::Result<::std::vec::Vec<#primary>, TypeError> {
                    match self {
                        #(#infer_output_types_arms,)*
                    }
                }

                fn render(
                    &self,
                    formatter: &mut ::core::fmt::Formatter<'_>,
                    indentation: usize,
                ) -> ::core::fmt::Result {
                    match self {
                        #(#render_arms,)*
                    }
                }
            }

            #[automatically_derived]
            impl #implementation_generics ::core::fmt::Display for #self_type #where_clause {
                fn fmt(&self, formatter: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                    <Self as Operation<#primary>>::render(self, formatter, 0)
                }
            }
        }
    }

    /// Builds the core `Operation` delegation match arms from the provided per-variant payload `operation_types`
    /// (parallel to [`EnumSpec::variants`]). Each arm delegates through always-turbofish
    /// `<OperationType as #bound>::#method(#receiver, #arguments)`, dereferencing `Box`ed payloads.
    fn core_arms(
        &self,
        operation_types: &[syn::Type],
        bound: &TokenStream,
        method: &TokenStream,
        arguments: &TokenStream,
    ) -> Vec<TokenStream> {
        self.variants
            .iter()
            .zip(operation_types)
            .map(|(variant, operation_type)| {
                let variant_ident = &variant.variant;
                let receiver = if variant.is_boxed { quote!(&**operation) } else { quote!(operation) };
                quote! {
                    Self::#variant_ident(operation) => <#operation_type as #bound>::#method(#receiver, #arguments)
                }
            })
            .collect()
    }

    /// Returns whether the generic parameter is the inferred primary type parameter (the one a [`PrimaryPin`] replaces).
    fn is_primary_parameter(&self, parameter: &syn::GenericParam) -> bool {
        matches!(parameter, syn::GenericParam::Type(parameter) if type_is_ident(&self.primary_type, &parameter.ident))
    }

    /// Returns whether the where-predicate is the primary type parameter's own bound (e.g., `T: Parameter + Type`),
    /// which a [`PrimaryPin`] drops because the parameter is replaced by a concrete type.
    fn is_primary_parameter_predicate(&self, predicate: &syn::WherePredicate) -> bool {
        matches!(predicate, syn::WherePredicate::Type(predicate) if predicate.bounded_ty == self.primary_type)
    }

    /// Returns the [`ParameterSubstitution`] rewriting this enum's primary type parameter into the [`PrimaryPin`]'s
    /// concrete type, or [`None`] when the enum is unpinned. Both the [`Operation`](crate::define_operation_types)/
    /// [`Display`] impls and the per-variant dispatch [`Property`] impls apply this substitution so a pinned enum's
    /// generated impls are specialized at `primary_type = concrete` (e.g., `ArrayOperation<ArrayType, ...>`) rather
    /// than left generic over the primary parameter.
    fn pin_substitution(&self) -> Option<ParameterSubstitution> {
        self.pin
            .as_ref()
            .map(|pin| ParameterSubstitution { parameter: self.primary_type.clone(), concrete: pin.concrete.clone() })
    }

    /// Returns the base impl [`syn::Generics`] every generated impl header for this enum builds on. For an unpinned
    /// enum this is the bound-bearing [`EnumSpec::implementation_generics`] unchanged. For a pinned enum the primary
    /// type parameter and its own bound are dropped, `primary_type -> concrete` is substituted throughout the remaining
    /// bounds, and the pin's extra predicates (already concrete) are appended — exactly the surgery the
    /// [`Operation`](crate::define_operation_types)/[`Display`] pin branch performs, so property impls share it.
    fn base_impl_generics(&self) -> syn::Generics {
        let mut generics = self.implementation_generics.clone();
        let Some(pin) = self.pin.as_ref() else {
            return generics;
        };
        let mut substitution = self.pin_substitution().unwrap();
        generics.params =
            generics.params.iter().filter(|parameter| !self.is_primary_parameter(parameter)).cloned().collect();
        if let Some(where_clause) = &mut generics.where_clause {
            where_clause.predicates = where_clause
                .predicates
                .iter()
                .filter(|predicate| !self.is_primary_parameter_predicate(predicate))
                .cloned()
                .collect();
        }
        substitution.visit_generics_mut(&mut generics);
        generics.make_where_clause().predicates.extend(pin.predicates.iter().cloned());
        generics
    }

    /// Returns this enum's self type for an impl header (`Name<...generics>`), specialized at the [`PrimaryPin`]'s
    /// concrete type when the enum is pinned and left generic over the primary parameter otherwise.
    fn impl_self_type(&self) -> syn::Type {
        let name = self.name;
        let (_, type_generics, _) = self.generics.split_for_impl();
        let mut self_type: syn::Type = syn::parse_quote!(#name #type_generics);
        if let Some(mut substitution) = self.pin_substitution() {
            substitution.visit_type_mut(&mut self_type);
        }
        self_type
    }

    /// Returns the variant's operation type with the [`PrimaryPin`] substitution applied (so a pinned enum's per-variant
    /// delegation bounds and turbofish arms name `Payload<concrete, ...>`), or the operation type unchanged when the
    /// enum is unpinned. The carried `Self` is left intact: in a property impl body it resolves to the pinned self type.
    fn pinned_operation_type(&self, operation_type: &syn::Type) -> syn::Type {
        let mut operation_type = operation_type.clone();
        if let Some(mut substitution) = self.pin_substitution() {
            substitution.visit_type_mut(&mut operation_type);
        }
        operation_type
    }

    /// Returns the primary type that property impls (`batchable`/`transposable`/`interpretable`) spell as the trait's
    /// type argument and in their method signatures: the [`PrimaryPin`]'s concrete type for a pinned enum (e.g.
    /// `ArrayType`), or the inferred [`EnumSpec::primary_type`] otherwise. A pinned enum drops its inferred primary
    /// type parameter from the impl generics, so a property predicate or signature spelling that parameter would
    /// reference an out-of-scope type; this returns the concrete pin instead.
    fn pinned_primary_type(&self) -> syn::Type {
        self.pin.as_ref().map_or_else(|| self.primary_type.clone(), |pin| pin.concrete.clone())
    }

    /// Generates the standard-conversion capability implementation for a single [`OperationVariant`] of this enum: a
    /// [`From`] that wraps the held operation type into the enum variant, and a [`TryFrom`] that borrows the held
    /// operation type back out (returning `Err(())` for any other variant).
    ///
    /// The conversions are keyed entirely on the variant's operation type ([`OperationVariant::operation_type`]), so no
    /// per-trait generic arguments need to be derived. For a `Box<...>` payload, the [`From`] wraps the operation in a
    /// `Box` and the [`TryFrom`] dereferences through it.
    fn support_implementation(&self, variant: &OperationVariant) -> TokenStream {
        let name = self.name;
        let (implementation_generics, type_generics, where_clause) = self.impl_fragments();

        let variant_ident = &variant.variant;

        // The operation type may mention `Self` (e.g., `CustomJvpOperation<T, V, Self>`), which is not valid in the
        // trait-reference position of a generated `impl` header. Substitute the enum's concrete self type so the
        // generated `From`/`TryFrom` impls name the operation type explicitly.
        let operation_type = self.self_substituted(&variant.operation_type);

        // Box payloads carry the operation behind a `Box`, so the `From` wraps and the `TryFrom` dereferences.
        let (constructed_payload, accessed_payload) = if variant.is_boxed {
            (quote!(::std::boxed::Box::new(operation)), quote!(&**operation))
        } else {
            (quote!(operation), quote!(operation))
        };

        // The `TryFrom` borrows out of the enum, so its impl header introduces a `'__op` lifetime borrowing both the
        // enum value and the returned operation reference.
        let mut try_from_generics = self.implementation_generics.clone();
        try_from_generics.params.insert(0, syn::parse_quote!('__op));
        let (try_from_generics, _, _) = try_from_generics.split_for_impl();

        quote! {
            #[automatically_derived]
            impl #implementation_generics ::core::convert::From<#operation_type>
                for #name #type_generics #where_clause
            {
                fn from(operation: #operation_type) -> Self {
                    Self::#variant_ident(#constructed_payload)
                }
            }

            #[automatically_derived]
            impl #try_from_generics ::core::convert::TryFrom<&'__op #name #type_generics>
                for &'__op #operation_type #where_clause
            {
                type Error = ();

                fn try_from(value: &'__op #name #type_generics) -> ::core::result::Result<Self, ()> {
                    match value {
                        #name::#variant_ident(operation) => ::core::result::Result::Ok(#accessed_payload),
                        _ => ::core::result::Result::Err(()),
                    }
                }
            }
        }
    }

    /// Generates the per-variant dispatch impl for the requested [`Property`] on this enum, dispatching on the property
    /// kind. Returns an empty [`TokenStream`] (after pushing an error onto `errors`) only for the
    /// [`Property::Interpretable`] kind, which is never stored as a [`Property`] and is generated via
    /// [`EnumSpec::interpretable_implementation`] instead.
    fn property_implementation(&self, property: &Property, errors: &mut Vec<syn::Error>) -> TokenStream {
        match property {
            Property::Differentiable { .. } => self.differentiable_implementation(property, errors),
            Property::Batchable { .. } => self.batchable_implementation(property, errors),
            Property::Transposable { .. } => self.transposable_implementation(property, errors),
            Property::Interpretable => {
                errors.push(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    "internal error: 'interpretable' is generated via the interpretable impl, not as a property",
                ));
                TokenStream::new()
            }
        }
    }

    /// Generates the [`DifferentiableOperation`](crate::define_operation_types) delegation impl for a
    /// `differentiable<D>` property.
    fn differentiable_implementation(&self, property: &Property, errors: &mut Vec<syn::Error>) -> TokenStream {
        let Property::Differentiable { context, extra_predicates } = property else {
            errors.push(self.property_kind_error("differentiable"));
            return TokenStream::new();
        };
        let bound = quote!(DifferentiableOperation<#context>);
        let arms = self.delegation_arms(&bound, &quote!(jvp), &quote!(context, inputs));
        // The differentiation context's own bound is intrinsic to the trait reference.
        let predicates = combine_predicates([syn::parse_quote!(#context: DifferentiationContext)], extra_predicates);
        let header = self.property_header(&[context.clone()], &bound, &predicates);
        quote! {
            #header {
                fn jvp<'jvp>(
                    &self,
                    context: &mut TangentContext<'jvp, #context>,
                    inputs: &[JvpTracer<'jvp, #context>],
                ) -> ::core::result::Result<::std::vec::Vec<JvpTracer<'jvp, #context>>, ProgramError>
                where
                    #context: 'jvp,
                {
                    match self {
                        #(#arms,)*
                    }
                }
            }
        }
    }

    /// Generates the [`BatchableOperation`](crate::define_operation_types) delegation impl for the `batchable<V, C>`
    /// property.
    fn batchable_implementation(&self, property: &Property, errors: &mut Vec<syn::Error>) -> TokenStream {
        let Property::Batchable { value, context, extra_predicates } = property else {
            errors.push(self.property_kind_error("batchable"));
            return TokenStream::new();
        };
        // Use the pin-aware primary type so a pinned enum's value bound names the concrete type instead of the inferred
        // primary parameter, which the pin drops from the impl generics.
        let primary_type = self.pinned_primary_type();
        let bound = quote!(BatchableOperation<#value, #context>);
        let arms = self.delegation_arms(&bound, &quote!(batch), &quote!(context, inputs));
        // `BatchableOperation`'s value parameter must implement `Value<ArrayType>`; the primary type is `ArrayType` at
        // every valid use site (the supertrait is `Operation<ArrayType>`), so we spell the bound with it.
        let predicates = combine_predicates([syn::parse_quote!(#value: Value<#primary_type>)], extra_predicates);
        let header = self.property_header(&[value.clone(), context.clone()], &bound, &predicates);
        quote! {
            #header {
                fn batch(
                    &self,
                    context: &#context,
                    inputs: &[ArrayBatch<#value>],
                ) -> ::core::result::Result<::std::vec::Vec<ArrayBatch<#value>>, ProgramError> {
                    match self {
                        #(#arms,)*
                    }
                }
            }
        }
    }

    /// Generates the [`TransposableOperation`](crate::define_operation_types) delegation impl for the
    /// `transposable<V, O>` / `transposable<V>` property. The trait carries the primary type `T` as its first argument,
    /// so the emitted trait reference is `TransposableOperation<T, V, O>` — where `O` is the explicit operation generic
    /// for the `transposable<V, O>` form, or the enum's own (pin-aware) self type for the self-pinned `transposable<V>`
    /// form (used when a variant's transpose stages operations back into this very enum).
    fn transposable_implementation(&self, property: &Property, errors: &mut Vec<syn::Error>) -> TokenStream {
        let Property::Transposable { value, operation, extra_predicates } = property else {
            errors.push(self.property_kind_error("transposable"));
            return TokenStream::new();
        };
        // Use the pin-aware primary type so a pinned enum's trait reference and method signature name the concrete type
        // (e.g. `ArrayType`) instead of the inferred primary parameter, which the pin drops from the impl generics.
        let primary_type = self.pinned_primary_type();
        // The staged operation type is either the explicit `O` generic or, for the self-pinned `transposable<V>` form,
        // the enum's own (pin-aware) self type. In the self-pinned case the operation type *is* `Self`, whose
        // `Operation<T>` impl is generated by this macro, so no intrinsic `O: Operation<T>` predicate is added and the
        // only property generic introduced is `V`.
        let operation_type = match operation {
            Some(operation) => quote!(#operation),
            None => {
                let self_type = self.impl_self_type();
                quote!(#self_type)
            }
        };
        let bound = quote!(TransposableOperation<#primary_type, #value, #operation_type>);
        let arms = self.delegation_arms(&bound, &quote!(transpose), &quote!(context, input_types, output_cotangents));
        let mut predicates = vec![syn::parse_quote!(#value: Value<#primary_type>)];
        let mut property_generics = vec![value.clone()];
        if let Some(operation) = operation {
            predicates.push(syn::parse_quote!(#operation: Operation<#primary_type>));
            property_generics.push(operation.clone());
        }
        predicates.extend(extra_predicates.iter().cloned());
        let header = self.property_header(&property_generics, &bound, &predicates);
        quote! {
            #header {
                fn transpose<'transpose>(
                    &self,
                    context: &mut AbstractTracingContext<'transpose, #primary_type, #value, #operation_type>,
                    input_types: &[&#primary_type],
                    output_cotangents: &[Cotangent<'transpose, #primary_type, #value, #operation_type>],
                ) -> ::core::result::Result<
                    ::std::vec::Vec<Cotangent<'transpose, #primary_type, #value, #operation_type>>,
                    ProgramError,
                > {
                    match self {
                        #(#arms,)*
                    }
                }
            }
        }
    }

    /// Generates the [`InterpretableOperation`](crate::define_operation_types) delegation impl. The value generic is the
    /// caller-named [`InterpretableSpec::value`] (e.g. `interpretable<W>`) or a fresh macro-internal generic for the bare
    /// `interpretable` form; it is bounded `Value<T>` and each variant delegates through
    /// `<Operation as InterpretableOperation<T, _>>::interpret(operation, context, inputs)`, threading the
    /// [`InterpretationContext`](crate::programs::Value::InterpretationContext) of the value type. Any
    /// [`InterpretableSpec::predicates`] from a trailing `where { ... }` are appended (needed when a self-referential
    /// variant's interpret requires a structural bound, like `Vec<W>: Parameterized<...>`, that no per-variant bound
    /// implies).
    fn interpretable_implementation(&self, spec: &InterpretableSpec) -> TokenStream {
        // Use the pin-aware primary type so a pinned enum's trait reference and value bound name the concrete type
        // instead of the inferred primary parameter, which the pin drops from the impl generics.
        let primary_type = self.pinned_primary_type();
        let value: syn::Ident = spec.value.clone().unwrap_or_else(|| syn::parse_quote!(__InterpretableValue));
        let bound = quote!(InterpretableOperation<#primary_type, #value>);
        let arms = self.delegation_arms(&bound, &quote!(interpret), &quote!(context, inputs));
        let predicates = combine_predicates([syn::parse_quote!(#value: Value<#primary_type>)], &spec.predicates);
        let header = self.property_header(&[value.clone()], &bound, &predicates);
        quote! {
            #header {
                fn interpret(
                    &self,
                    context: &<#value as Value<#primary_type>>::InterpretationContext,
                    inputs: &[#value],
                ) -> ::core::result::Result<::std::vec::Vec<#value>, ProgramError> {
                    match self {
                        #(#arms,)*
                    }
                }
            }
        }
    }

    /// Builds the `#[automatically_derived] impl <generics> <trait_reference> for <Enum> <where ...>` header shared by
    /// every per-variant dispatch [`Property`] impl, leaving the trailing method block to the caller.
    ///
    /// The impl generics are this enum's impl generics extended with `property_generics` (the property's own generic
    /// parameters, e.g., `D` or `V, C`). The where-clause is this enum's where-clause extended with one per-variant
    /// bound `<Operation>: <trait_reference>` (so every per-variant delegation type-checks) plus `extra_predicates`
    /// (intrinsic generic-parameter bounds plus any caller-supplied predicates).
    ///
    /// # Parameters
    ///
    ///   - `property_generics`: The property's own impl-level generic parameters to append to the enum's generics.
    ///   - `trait_reference`: The full trait reference the impl is for (e.g., `DifferentiableOperation<D>`); it is also
    ///     used as the per-variant bound on each variant's operation type.
    ///   - `extra_predicates`: Extra where-clause predicates appended after the per-variant bounds.
    fn property_header(
        &self,
        property_generics: &[syn::Ident],
        trait_reference: &TokenStream,
        extra_predicates: &[syn::WherePredicate],
    ) -> TokenStream {
        // Build on the pin-aware base impl generics: an unpinned enum keeps its full bound-bearing generics, while a
        // pinned enum (e.g. `ArrayOperation` at `ArrayType`) already has its primary parameter dropped, its bounds
        // substituted, and the pin's predicates appended. The self type is correspondingly pinned, so the body's `Self`
        // resolves to the concrete enum and the self-referential delegation arms type-check at `primary_type = concrete`.
        let mut impl_generics = self.base_impl_generics();
        let self_type = self.impl_self_type();

        // Extend with the property's own generic parameters, skipping any that are already enum generic parameters. A
        // property may intentionally name an existing enum parameter as its generic — most notably `interpretable<V>`
        // over an enum whose interpret value type IS its declared value parameter `V` (so a value-tied variant like
        // `ConstantOperation<DataType, V>` / `CustomVjpOperation<_, V, Self>` interprets over that same `V`); re-adding
        // it would shadow the enum parameter and break the delegation.
        for generic in property_generics {
            let already_a_parameter = impl_generics.type_params().any(|parameter| &parameter.ident == generic);
            if !already_a_parameter {
                impl_generics.params.push(syn::GenericParam::Type(syn::TypeParam::from(generic.clone())));
            }
        }

        // Auto-add one bound per variant so each delegation type-checks, then append the extra predicates. The variant's
        // operation type is pin-substituted so a pinned enum's bounds name `Payload<concrete, ...>`. The bound is
        // SKIPPED for self-referential variants (payloads naming `Self`/the enum, e.g. `CustomJvpOperation<T, V, Self>`
        // or a higher-order `ConditionOperation<T, V, Self>`): such a bound would expand to require the enum's own
        // `Self: #trait_reference`, closing an inductive where-clause cycle (E0275) at every external call site. Instead
        // the delegation arm resolves the variant's trait obligation in the method BODY, where the impl's own
        // `Self: #trait_reference` is dischargeable from the assumed where-clause (the body-check fixed point that the
        // equivalent hand-written impls rely on). The structural bounds the variant's own impl needs (which a non-self
        // variant would carry transitively through its skipped bound) must then be supplied via `extra_predicates`.
        let variant_bounds = self.variants.iter().filter(|variant| !self.is_self_referential(variant)).map(|variant| {
            let operation_type = self.pinned_operation_type(&variant.operation_type);
            let predicate: syn::WherePredicate = syn::parse_quote!(#operation_type: #trait_reference);
            predicate
        });
        impl_generics
            .make_where_clause()
            .predicates
            .extend(variant_bounds.chain(extra_predicates.iter().cloned()));
        let (impl_generics, _, where_clause) = impl_generics.split_for_impl();

        quote! {
            #[automatically_derived]
            impl #impl_generics #trait_reference for #self_type #where_clause
        }
    }

    /// Builds the per-variant `match` arms for a property delegation: each arm delegates through always-turbofish
    /// `<Operation as #bound>::#method(#receiver, #arguments)`, where the receiver is `&**operation` for `Box`ed
    /// payloads (so trait resolution targets the unboxed operation type rather than probing the boxed receiver) and
    /// `operation` otherwise. The operation type is pin-substituted (`primary_type -> concrete`) so a pinned enum's
    /// arms name `Payload<concrete, ...>`; the carried `Self` resolves to the pinned self type in the method body.
    fn delegation_arms(&self, bound: &TokenStream, method: &TokenStream, arguments: &TokenStream) -> Vec<TokenStream> {
        self.variants
            .iter()
            .map(|variant| {
                let variant_ident = &variant.variant;
                let operation_type = self.pinned_operation_type(&variant.operation_type);
                let receiver = if variant.is_boxed { quote!(&**operation) } else { quote!(operation) };
                quote! {
                    Self::#variant_ident(operation) => <#operation_type as #bound>::#method(#receiver, #arguments)
                }
            })
            .collect()
    }

    /// Builds the error reported when a property generator is invoked with a [`Property`] of the wrong kind. This is an
    /// internal invariant violation (the dispatcher in [`EnumSpec::property_implementation`] routes by kind), so the
    /// message names the expected kind for diagnosis.
    fn property_kind_error(&self, expected: &str) -> syn::Error {
        syn::Error::new(proc_macro2::Span::call_site(), format!("internal error: expected a '{expected}' property"))
    }
}

/// [`VisitMut`] that rewrites every `Self` type into a concrete type, used by [`EnumSpec::self_substituted`] to resolve
/// `Self` inside a variant's operation type before it is placed in a generated `impl` header's trait-reference position.
struct SelfSubstitution {
    /// Concrete type that each `Self` occurrence is rewritten to (the generated enum's `Name<...generics>`).
    self_type: syn::Type,
}

impl VisitMut for SelfSubstitution {
    fn visit_type_mut(&mut self, ty: &mut syn::Type) {
        if let syn::Type::Path(syn::TypePath { qself: None, path }) = ty
            && path.is_ident("Self")
        {
            *ty = self.self_type.clone();
            return;
        }
        syn::visit_mut::visit_type_mut(self, ty);
    }
}

/// [`VisitMut`] that rewrites every occurrence of one type (a generic type parameter) into a concrete type. Used to
/// specialize the generated `Operation`/`Display` impls at a [`PrimaryPin`]'s concrete type: the enum's primary type
/// parameter is replaced by the concrete type in the self type, the delegated payload operation types, and the
/// remaining generic bounds.
struct ParameterSubstitution {
    /// Generic type parameter being replaced (e.g., the `T` of `ArrayOperation<T, V, Extension>`), as a [`syn::Type`].
    parameter: syn::Type,

    /// Concrete type each occurrence of `parameter` is rewritten to (e.g., `ArrayType`).
    concrete: syn::Type,
}

impl VisitMut for ParameterSubstitution {
    fn visit_type_mut(&mut self, ty: &mut syn::Type) {
        if *ty == self.parameter {
            *ty = self.concrete.clone();
            return;
        }
        syn::visit_mut::visit_type_mut(self, ty);
    }
}

/// Returns whether `ty` is a plain (unqualified, single-segment, argument-free) path type naming exactly `ident` — used
/// to identify the generic type parameter a [`PrimaryPin`] removes from the pinned impl generics.
fn type_is_ident(ty: &syn::Type, ident: &syn::Ident) -> bool {
    matches!(ty, syn::Type::Path(syn::TypePath { qself: None, path }) if path.is_ident(ident))
}

/// [`VisitMut`] used purely as a detector (it mutates nothing): it sets [`EnumNameDetector::found`] if any path segment
/// of the visited type names the tracked enum. Applied (after `Self`-substitution) to a variant's payload type to decide
/// whether the variant is self-referential. It is a `VisitMut` rather than a `Visit` only because the `visit-mut` syn
/// feature is already enabled for [`SelfSubstitution`]/[`ParameterSubstitution`].
struct EnumNameDetector {
    /// Enum identifier being searched for (e.g., `ScalarOperation`).
    name: syn::Ident,

    /// Set to `true` once a path segment naming [`EnumNameDetector::name`] is encountered.
    found: bool,
}

impl VisitMut for EnumNameDetector {
    fn visit_path_segment_mut(&mut self, segment: &mut syn::PathSegment) {
        if segment.ident == self.name {
            self.found = true;
        }
        syn::visit_mut::visit_path_segment_mut(self, segment);
    }
}

impl Property {
    /// Builds the synthetic `transposable<__TransposeValue, __TransposeOperation>` property used to auto-emit a
    /// transpose delegation for a `linear = { ... }` enum. The generic identifiers are macro-internal so they cannot
    /// clash with the linear enum's own generics, and there are no extra where-clause predicates.
    fn transposable_auto() -> Self {
        Property::Transposable {
            value: syn::parse_quote!(__TransposeValue),
            operation: Some(syn::parse_quote!(__TransposeOperation)),
            extra_predicates: Vec::new(),
        }
    }
}

/// Concatenates a property's intrinsic generic-parameter bounds (which the macro derives from the known trait being
/// implemented, e.g., `D: DifferentiationContext`) with the caller-supplied extra where-clause predicates, preserving
/// order with the intrinsic bounds first. These bounds are required for the generated trait reference to be well-formed
/// and are added in addition to the per-variant delegation bounds.
fn combine_predicates<const N: usize>(
    intrinsic: [syn::WherePredicate; N],
    extra: &[syn::WherePredicate],
) -> Vec<syn::WherePredicate> {
    intrinsic.into_iter().chain(extra.iter().cloned()).collect()
}

/// Infers the primary type `T` from the `Value<X>` bounds declared on the provided [`syn::Generics`] (covering both the
/// bounds on the generic parameters themselves and any where-clause predicates). The inferred primary type is the `X` of
/// such a bound. If no `Value<...>` bound is found, an error is returned. If multiple `Value<...>` bounds are found with
/// differing `X`s, an error is returned because the primary type would be ambiguous.
fn infer_primary_type(generics: &syn::Generics) -> syn::Result<syn::Type> {
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

    let mut primary_type: Option<syn::Type> = None;
    for bound in parameter_bounds.chain(where_bounds) {
        let Some(value_argument) = value_bound_argument(bound) else {
            continue;
        };
        match &primary_type {
            Some(existing) if existing != &value_argument => {
                return Err(syn::Error::new_spanned(
                    generics,
                    "define_operation_types! could not infer the primary type: found conflicting `Value<...>` bounds",
                ));
            }
            Some(_) => {}
            None => primary_type = Some(value_argument),
        }
    }

    primary_type.ok_or_else(|| {
        syn::Error::new_spanned(
            generics,
            "define_operation_types! could not infer the primary type: expected a `Value<...>` bound",
        )
    })
}

/// Returns the single type argument `X` of a `Value<X>` trait bound, or [`None`] if the provided bound is not a
/// `Value<...>` trait bound with exactly one type argument. The bound's path is matched on its last segment so that both
/// `Value<X>` and fully qualified paths such as `ryft_core::Value<X>` are recognized.
fn value_bound_argument(bound: &syn::TypeParamBound) -> Option<syn::Type> {
    let syn::TypeParamBound::Trait(bound) = bound else {
        return None;
    };
    let segment = bound.path.segments.last()?;
    if segment.ident != VALUE_TRAIT_NAME {
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

    use super::{OperationInput, OperationVariant, Property, infer_primary_type};

    #[test]
    fn test_operation_variant_from_payload_type() {
        let variant = OperationVariant::new(None, syn::parse_quote!(AddOperation)).unwrap();
        assert_eq!(variant.variant.to_string(), "Add");
        assert!(!variant.is_boxed);
        assert_eq!(variant.operation_type.to_token_stream().to_string(), "AddOperation");

        let variant = OperationVariant::new(None, syn::parse_quote!(ScaleOperation<DataType, V>)).unwrap();
        assert_eq!(variant.variant.to_string(), "Scale");
        assert!(!variant.is_boxed);

        let variant =
            OperationVariant::new(None, syn::parse_quote!(Box<CustomJvpOperation<DataType, V, Self>>)).unwrap();
        assert_eq!(variant.variant.to_string(), "CustomJvp");
        assert!(variant.is_boxed);
        assert_eq!(
            variant.operation_type.to_token_stream().to_string().replace(' ', ""),
            "CustomJvpOperation<DataType,V,Self>",
        );
    }

    #[test]
    fn test_operation_variant_rejects_bad_suffix() {
        assert!(OperationVariant::new(None, syn::parse_quote!(NotAnOp)).is_err());
        assert!(OperationVariant::new(None, syn::parse_quote!(Operation)).is_err());
    }

    #[test]
    fn test_infer_primary_type() {
        let generics = syn::parse2::<syn::DeriveInput>(quote::quote!(
            struct Dummy<V: Value<DataType>>;
        ))
        .unwrap()
        .generics;
        assert_eq!(infer_primary_type(&generics).unwrap().to_token_stream().to_string(), "DataType");

        // The primary type can also be inferred from a where-clause `Value<...>` bound.
        let generics = syn::parse2::<syn::DeriveInput>(quote::quote!(
            struct Dummy<V>
            where
                V: Value<DataType>;
        ))
        .unwrap()
        .generics;
        assert_eq!(infer_primary_type(&generics).unwrap().to_token_stream().to_string(), "DataType");

        let generics = syn::parse2::<syn::DeriveInput>(quote::quote!(
            struct Dummy<V: Clone>;
        ))
        .unwrap()
        .generics;
        assert!(infer_primary_type(&generics).is_err());

        // Conflicting `Value<...>` bounds make the primary type ambiguous and must be rejected.
        let generics = syn::parse2::<syn::DeriveInput>(quote::quote!(
            struct Dummy<V: Value<DataType>, W: Value<OtherType>>;
        ))
        .unwrap()
        .generics;
        assert!(infer_primary_type(&generics).is_err());
    }

    #[test]
    fn test_parse_operation_input() {
        let input: OperationInput = syn::parse_quote! {
            type = ScalarOperation<V: Value<DataType>>,
            variants = [
                ZeroOperation<DataType>,
                AddOperation,
                ScaleOperation<DataType, V>,
                Box<CustomJvpOperation<DataType, V, Self>>,
            ],
        };
        assert_eq!(input.name.to_string(), "ScalarOperation");
        assert_eq!(input.variants.len(), 4);
        assert_eq!(input.variants[0].variant.to_string(), "Zero");
        assert_eq!(input.variants[1].variant.to_string(), "Add");
        assert_eq!(input.variants[2].variant.to_string(), "Scale");
        assert_eq!(input.variants[3].variant.to_string(), "CustomJvp");
        assert!(input.variants[3].is_boxed);
    }

    #[test]
    fn test_parse_operation_input_with_where_and_properties() {
        let input: OperationInput = syn::parse_quote! {
            type = CustomOperation<V> where { V: Value<DataType> },
            variants = [AddOperation],
            properties = [differentiable<D>, batchable<V, C>],
            linear = {
                type = LinearCustomOperation<V> where { V: Value<DataType> },
                variants = [AddOperation],
            },
            interpretable,
        };
        assert_eq!(input.name.to_string(), "CustomOperation");
        assert!(input.generics.where_clause.is_some());
        assert_eq!(input.properties.len(), 2);
        assert!(matches!(input.properties[0], Property::Differentiable { .. }));
        assert!(matches!(input.properties[1], Property::Batchable { .. }));
        let linear = input.linear.as_ref().expect("expected a linear spec");
        assert_eq!(linear.name.to_string(), "LinearCustomOperation");
        assert_eq!(linear.variants.len(), 1);
        assert!(input.interpretable.is_some());
        // The primary type is still inferable from the folded-in where predicate.
        assert_eq!(infer_primary_type(&input.generics).unwrap().to_token_stream().to_string(), "DataType");
    }

    #[test]
    fn test_parse_interpretable_as_property_entry() {
        // `interpretable` is also accepted inside `properties = [ ... ]`, normalized into the dedicated flag rather
        // than retained as a `Property`.
        let input: OperationInput = syn::parse_quote! {
            type = CustomOperation<V: Value<DataType>>,
            variants = [AddOperation],
            properties = [interpretable, differentiable<D>],
        };
        assert!(input.interpretable.is_some());
        let interpretable = input.interpretable.as_ref().unwrap();
        assert!(interpretable.value.is_none());
        assert!(interpretable.predicates.is_empty());
        assert_eq!(input.properties.len(), 1);
        assert!(matches!(input.properties[0], Property::Differentiable { .. }));
    }

    #[test]
    fn test_parse_interpretable_with_value_generic_and_where_clause() {
        // The top-level `interpretable<W> where { ... }` form names the value generic and carries extra predicates,
        // used when a self-referential variant's interpretation needs a structural bound no per-variant bound implies.
        let input: OperationInput = syn::parse_quote! {
            type = CustomOperation<V: Value<DataType>>,
            variants = [AddOperation],
            interpretable<W> where { Vec<W>: Clone },
        };
        let interpretable = input.interpretable.as_ref().expect("expected an interpretable spec");
        assert_eq!(interpretable.value.as_ref().unwrap().to_string(), "W");
        assert_eq!(interpretable.predicates.len(), 1);
    }

    #[test]
    fn test_parse_property_generic_arity_and_where() {
        // A `where { ... }` clause on a property is captured as extra predicates.
        let property: Property = syn::parse_quote!(differentiable<D> where { D: DifferentiationContext });
        match property {
            Property::Differentiable { context, extra_predicates } => {
                assert_eq!(context.to_string(), "D");
                assert_eq!(extra_predicates.len(), 1);
            }
            _ => panic!("expected a differentiable property"),
        }

        // The transposable property carries two generics in order.
        let property: Property = syn::parse_quote!(transposable<V, O>);
        match property {
            Property::Transposable { value, extra_predicates, .. } => {
                assert_eq!(value.to_string(), "V");
                // TODO(eaplatanios): assert_eq!(operation.to_string(), "O");
                assert!(extra_predicates.is_empty());
            }
            _ => panic!("expected a transposable property"),
        }

        // The wrong number of generic arguments is rejected.
        assert!(syn::parse2::<Property>(quote::quote!(batchable<V>)).is_err());
        assert!(syn::parse2::<Property>(quote::quote!(differentiable<D, E>)).is_err());
        // An unknown property name is rejected.
        assert!(syn::parse2::<Property>(quote::quote!(frobnicate<X>)).is_err());
        // `interpretable` does not take generic arguments.
        assert!(syn::parse2::<Property>(quote::quote!(interpretable<V>)).is_err());
    }
}
