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

## Review Coverage

Before fixing findings, record a coverage inventory in the task plan for the complete operation surface, not just
the changed lines or previously reported issues. Read the repository's testing guidelines and identify the existing
operation and capability patterns that apply. For each applicable behavior, record the inspected implementation,
its invariants, verification evidence, and any remaining uncertainty. Mark a case inapplicable only with a reason.

Cover the operation's payload and public APIs, region and type contracts, interpretation, reference discharge,
partial evaluation, batching, differentiation, transposition, providers and capabilities, and backend integration where
present. Choose meaningful cases for each rule rather than mechanically testing every combination. For region-carrying
operations, these include known, symbolic, and unknown inputs; eager and staged execution; selected and unselected
regions; observable effects and deferred work; zero outputs and structural zeros; static, dynamic, and bounded ragged
geometry; references and identity-bearing values; placement and manual variation; and nested transform composition
where supported.

Trace information across every transform boundary: values and types, region signatures, runtime geometry, reference
identities, effects, derivative activity, and validation evidence. Verify both their producers and consumers; a locally
correct rule can still lose information at a boundary. Check direct context binding as well as validated program replay:
rewriting an operation must not bypass its original primitive contract. Matching primal signatures do not guarantee
matching compact derivative slots, and replaying separate regions into one scope must preserve local type identities.
Inspect supported and rejected cases, including empty, singleton, and disconnected inputs or outputs where meaningful.
Evaluate documented limitations and optimizations against their semantic consequences and the requested JAX parity;
their presence in comments is not justification for accepting them.

Compare the relevant rules with official JAX implementation sources and distinguish frontend conveniences from
primitive contracts. Validate deliberate extensions on their own semantics. Use existing tests as evidence, then
identify adversarial cases those tests do not exercise. Add focused regressions for changed behavior and material
coverage gaps. Tests involving non-termination or expensive execution must fail deterministically without running
an unbounded computation.

Passing tests or repairing discovered issues does not establish review completeness. Check off a behavior only after
inspecting its code and assessing the adequacy of its evidence. After edits, verify the actual final source and affected
backend behavior; an earlier compiled binary is evidence for the source it contains, not for later changes. Report
blocked verification and intentionally unsupported behavior explicitly instead of claiming exhaustive correctness
or complete parity.

## Final Pass

After you have made all the necessary code changes and are satisfied with the implementation, you must make one final
pass. In that pass, I want you to review everything related to that operation for correctness and for parity with how
JAX implements its corresponding operation types. I also want you to review it for compliance with all of our standards
and conventions (including the documentation strings and the tests in that module). I know this may sound repetitive,
but I want you to do all this again and keep iterating until all my instructions are followed.

Use an independent final review against the coverage inventory, without supplying prior conclusions when they would bias
that review. Resolve confirmed findings, update their evidence, and revisit affected inventory entries after every fix.
The final report must state verified outcomes and remaining limitations; do not infer that no undiscovered defects exist
merely because the final pass found none.
