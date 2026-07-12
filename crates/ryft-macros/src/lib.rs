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

/// Generates the batching (vectorization) dispatchers for an operation enum: one `BatchableOperation<C>` impl that
/// is generic over the parent context `C` plus the `BatchableProgramOperation` witness for nested-program batching.
///
/// The dispatcher forwards the active `BatchingContext<C>` to every variant's own rule, so the parent/active
/// distinction lives in each rule's body rather than in dispatch: ordinary rules run their lifted physical work
/// through `context.parent()` (eagerly under an eager parent, staged under a staging parent), while rules keyed on
/// the active frame's axis metadata (e.g., named-axis collectives) inspect the batching context directly. Each
/// non-recursive payload's batching obligation is transported as a per-variant `BatchableOperation<C>` predicate,
/// while recursive payloads (those mentioning `Self`) are discharged against the witness and the leaf bounds the
/// enum declares via `#[ryft(bounds(batching(...)))]`, which attach to the parent context's value type.
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
