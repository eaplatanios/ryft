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
/// attributes such as `#[ryft(bounds(interpretation(...)))]`.
#[proc_macro_derive(Operation, attributes(ryft))]
pub fn derive_operation(input: TokenStream) -> TokenStream {
    OperationCodeGenerator::generate_operation_impl(input)
}

/// Generates the batching (vectorization) dispatchers for an operation enum: the staged tracer-level and eager
/// value-level `BatchableOperation` impls plus the `BatchableProgramOperation` witness for nested-program batching.
///
/// Each non-recursive payload's batching obligation is transported as a per-variant `BatchableOperation` predicate,
/// while recursive payloads (those mentioning `Self`) are discharged against the leaf bounds the enum declares via
/// `#[ryft(bounds(batching(...)))]` — the value capabilities the recursive *eager* batching rules use directly,
/// bound to the eager impl's flowing value. That is the only position needing author-supplied leaves: the staged
/// flowing value is the unified tracer, whose capability impls are staging sugar conditioned only on
/// operation-shaped `From<XOperation>` conversions (the staged recursive rules spell those `From` bounds and the
/// closed enum discharges them structurally), and the program-constant space carries no batching capabilities at
/// all. In the staged impl, non-recursive arms dispatch at the parent staging context, while recursive arms — and
/// non-recursive variants marked with `#[ryft(batching(active))]`, such as named-axis collectives — dispatch at the
/// active batching context to reach its axis metadata.
#[proc_macro_derive(BatchableOperation, attributes(ryft))]
pub fn derive_batchable_operation(input: TokenStream) -> TokenStream {
    OperationCodeGenerator::generate_batchable_operation_impl(input)
}

/// Generates the forward-mode (JVP) differentiation dispatcher for an operation enum, together with the
/// `DifferentiableProgramOperation` and `LinearizableProgramOperation` witnesses that back the recursive higher-order
/// rules (their fixed bodies forward to `Program::jvp_program` / `Program::linearize`, and extra value bounds those
/// body checks need are supplied via `#[ryft(bounds(differentiation(...)))]`).
///
/// See the `ryft-core` documentation for the `Operation` trait for the full derive contract. This derive enables
/// forward-mode differentiation only; enums that also need reverse-mode differentiation additionally derive
/// `TransposableOperation`, whose output supplies the transposition dispatchers that reverse mode is built on.
#[proc_macro_derive(DifferentiableOperation, attributes(ryft))]
pub fn derive_differentiable_operation(input: TokenStream) -> TokenStream {
    OperationCodeGenerator::generate_differentiable_operation_impl(input)
}

/// Generates transposition dispatchers for a linear operation enum, enabling reverse-mode differentiation for
/// programs staged in that operation family.
///
/// See the `ryft-core` documentation for the `TransposableOperation` trait for the full derive contract, including
/// operation-type inference, generated payload bounds, and when the macro can generate the
/// `TransposableProgramOperation` witness for nested linear programs.
#[proc_macro_derive(TransposableOperation, attributes(ryft))]
pub fn derive_transposable_operation(input: TokenStream) -> TokenStream {
    OperationCodeGenerator::generate_transposable_operation_impl(input)
}
