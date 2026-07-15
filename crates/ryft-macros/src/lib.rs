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

/// Generates an operation enum and its selected semantic dispatchers.
///
/// See the `ryft-core` documentation for the `Operation` trait for the full derive contract, including operation-type
/// inference from `Value<T>` bounds, generated conversions, boxed payload handling, and supported `#[ryft(...)]`
/// attributes. Interpretation and partial evaluation are always generated. Batching, differentiation, and
/// transposition can be selected independently with `#[ryft(dispatch(batching, differentiation, transposition))]`.
#[proc_macro_derive(Operation, attributes(ryft))]
pub fn derive_operation(input: TokenStream) -> TokenStream {
    operations::generate_operation_impl(input)
}
