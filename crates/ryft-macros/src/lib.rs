use proc_macro::TokenStream;

mod helpers;
mod operations;
mod parameters;

use operations::CodeGenerator as OperationCodeGenerator;
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

// TODO(eaplatanios): Review from here onwards.

/// Generates an operation enum dispatcher.
///
/// See the `ryft-core` documentation for the `Operation` trait for the full derive contract, including operation-type
/// inference from `Value<T>` bounds, generated conversions, boxed payload handling, and supported `#[ryft(...)]`
/// attributes.
#[proc_macro_derive(Operation, attributes(ryft))]
pub fn derive_operation(input: TokenStream) -> TokenStream {
    OperationCodeGenerator::generate_operation_impl(input)
}

/// Generates transposition dispatchers for a linear operation enum.
///
/// See the `ryft-core` documentation for the `TransposableOperation` trait for the full derive contract, including
/// operation-type inference, generated payload bounds, and when the macro can generate the
/// `TransposableProgramOperation` witness for nested linear programs.
#[proc_macro_derive(TransposableOperation, attributes(ryft))]
pub fn derive_transposable_operation(input: TokenStream) -> TokenStream {
    OperationCodeGenerator::generate_transposable_operation_impl(input)
}
