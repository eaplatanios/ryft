---
name: add-or-review-core-operation
description: Adds support for a new operation type in `ryft-core`, or reviews support for an existing one.
---

You must look at our implementation of the `ryft_core` operations, in `ryft_core::operations`, and understand our code
style and conventions around code, documentation, and unit tests. Then, you must either add a new operation that was
requested by the user, or review the code that corresponds to an existing operation for compliance with all of our
conventions, and for correctness. You can refer to corresponding JAX operation implementations when reviewing for
correctness. In general, we aim either for parity with JAX, or to exceed JAX. The Ryft operation implementations should
never be lacking in any way with respect to the corresponding JAX operations.

# Instructions

All the operation code (including code comments), documentation (including the module docstring), and tests must
strictly abide to all of our conventions, and there must be no correctness mistakes. I have already made a pass over the
modules in `ryft_core::operations::constants` and `ryft_core::operations::manipulation` and you can use those as
examples for what I want. Specifically, for each operation I want to have the following ordering:

1. `*_OPERATION_NAME` constant,
2. `*Operation` struct type and its impl blocks, with the operation-specific impl blocks appearing in the order:
   `Operation`, `InterpretableOperation` using the appropriate capability trait bound, `PartiallyEvaluatableOperation`,
   `BatchableOperation`, `DifferentiableOperation`, `TransposableOperation`, `MemberOperation` (when applicable), and
   `MemberInterpretableOperation` (when applicable),
3. `*OperationProvider` trait wherever relevant and its impl blocks, and
4. capability trait and its impl blocks in order: `EagerContext` (for `Array` and then for `ArrayIrValue`),
   `ProjectedContext`, `StagingContext`, `PartialEvaluationContext`, `BatchingContext`, `DifferentiationContext`,
   and `TranspositionContext`.

You must also leverage any pre-existing macros from `ryft_core::macros` wherever applicable across both production code
and testing code. Regarding the unit tests, for most simple operations that do not involve edge cases that need testing,
the tests should look roughly like what I have for `ryft_core::operations::math::add` with `test_<operation name>`,
`test_<operation name>_type_inference`, `test_<operation name>_interpretation` (using `Array` values),
`test_<operation name>_partial_evaluation`, `test_<operation name>_batching`, `test_<operation name>_differentiation`,
and `test_<operation name>_transposition`, in this order. Each test can include multiple assertions and cases internally
to ensure sufficient coverage.

## Final Pass

After you have made all the necessary code changes and are satisfied with the implementation, you must make one final
pass. In that pass, I want you to review everything related to that operation for correctness and for parity with how
JAX implements its corresponding operation types. I also want you to review it for compliance with all of our standards
and conventions (including the documentation strings and the tests in that module). I know this may sound repetitive,
but I want you to do all this again and keep iterating until all my instructions are followed.
