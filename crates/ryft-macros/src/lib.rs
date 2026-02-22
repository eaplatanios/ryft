use proc_macro::TokenStream;

mod helpers;
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
