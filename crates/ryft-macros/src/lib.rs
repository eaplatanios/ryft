use proc_macro::TokenStream;

mod helpers;
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

/// Generates an operation enum's `Operation` contract and its selected semantic dispatchers.
///
/// The derivation applies to enums whose every variant wraps exactly one operation payload, and it makes the enum
/// behave like whichever payload it holds. It always generates the `Operation` implementation, the
/// `InterpretableOperation` and `PartiallyEvaluatableOperation` dispatchers, an `OperationProjection` implementation
/// per projected member variant, `Display`, and the owned `From<Payload>` plus borrowed `TryFrom<&Enum>` payload
/// conversions. Batching, forward-mode differentiation, and transposition dispatchers are generated only when selected.
///
/// The `#[ryft(...)]` attribute surface is the following, where enum-level attributes are all optional:
///
/// ```text
/// // Enum level:
/// #[ryft(crate = "...")]                     // Path used to name Ryft items; defaults to `ryft`.
/// #[ryft(type = T, constant = V)]            // Primary operation type and stored constant type; inferred otherwise.
/// #[ryft(members(U [, structural(S)]...))]   // Member universes the operation family declares.
/// #[ryft(dispatch(batching, differentiation, transposition))]
///
/// // Variant level:
/// #[ryft(projected(U [, structural]))]       // Every operand and result of the instruction belongs to `U`.
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
