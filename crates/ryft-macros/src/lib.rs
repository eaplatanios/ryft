use proc_macro::TokenStream;

mod helpers;
mod operations;
mod parameters;

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
