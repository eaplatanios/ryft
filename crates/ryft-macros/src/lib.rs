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

/// Generates an [`Operation`] implementation for an enum whose variants wrap operation payloads, along with a bunch
/// of other implementations related to that [`Operation`] and the transformations that it supports.
#[proc_macro_derive(Operation, attributes(ryft))]
pub fn derive_operation(input: TokenStream) -> TokenStream {
    OperationCodeGenerator::generate_operation_impl(input)
}
