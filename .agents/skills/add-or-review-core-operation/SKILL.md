---
name: add-or-review-core-operation
description: >-
  Add, change, or review ryft-core operations, value capabilities, providers, and transform rules, including their
  backend integration. Use for operation correctness, JAX parity, and conventions; not for MLIR dialect wrappers.
---

# Add or Review Core Operations

Implement or review the requested `ryft-core` operation surface, including its public capability APIs, transforms,
and affected backends. Aim for at least the corresponding JAX primitive's functionality and correctness; record
unsupported behavior or unverified parity explicitly rather than treating the goal as a verified result.

## Start Here

Read repository [AGENTS.md](../../../AGENTS.md) for workflow and safeguards, the
[Rust conventions](../../rust-conventions.md) for code and imports, and the
[documentation guidelines](../../documentation-guidelines.md) for API documentation. Read the
[unit-testing guidelines](../../unit-testing-guidelines.md) before writing or revising tests. This skill owns the
additional core operation conventions; keep future operation-specific corrections here or in its references. Apply
the shared conventions to module documentation, public APIs, implementation comments, and tests alike.

Inspect the current source before choosing an implementation pattern. The useful starting points below are paths
relative to the repository root, not a requirement to read every file for every task:

| Concern                                                 | Source                                                                       |
|---------------------------------------------------------|------------------------------------------------------------------------------|
| Operation, provider, member, and projection contracts   | `crates/ryft-core/src/programs/operations.rs`                                |
| Interpretation and reference discharge                  | `crates/ryft-core/src/interpretation.rs`, `src/programs/references/`         |
| Capability universes and value dispatch                 | `crates/ryft-core/src/operations/mod.rs`, `src/programs/values.rs`           |
| Families, bundles, composite values, and batch extents  | `crates/ryft-core/src/arrays/{operations,ir,batching}.rs`                    |
| Simple operations and tests                             | `crates/ryft-core/src/operations/arithmetic.rs`, especially `AddOperation`   |
| Construction and shape changes                          | `crates/ryft-core/src/operations/constants/`, `src/operations/manipulation/` |
| Shared production and testing macros                    | `crates/ryft-core/src/macros.rs`, `src/arrays/macros.rs`                     |
| Derives and capability projection generation            | `crates/ryft-macros/src/{operations,capabilities}.rs`                        |
| Transform contracts and replay                          | `crates/ryft-core/src/partial/`, `src/batching.rs`, `src/differentiation/`   |
| Backend conversion and lowering                         | `crates/ryft-xla/src/operations/`, `src/experimental/{ops.rs,lowering/}`     |

In a row with a shortened `src/...` path, use the same crate as the first path. Confirm paths and contracts in the
current checkout; examples are implementation references, not permission to perpetuate a defect or stale convention.

Load the references that apply before editing:

- [Capabilities and Providers](references/capabilities.md): Used for capability APIs, universe bounds, providers,
  composite projections, dispatch, generic transform-policy bounds, or backend value integration, including work
  outside operation modules.
- [Transform Invariants](references/transforms.md): Used for operation replay, reference validation, partial evaluation,
  batching, differentiation, transposition, sharding/manual variation, dynamic extents, or explicit dense-buffer kernel
  contracts.
- [Coverage and Final Review](references/review.md): Used for every operation addition or review, and for the affected
  behaviors of a scoped follow-up. Full operation reviews require the complete inventory, not just the reported bug.

## Operation Layout and Construction

Group each operation in this order, omitting only implementations that do not apply:

1. Its `*_OPERATION_NAME` constant.
2. Its operation type, inherent functions, and formatting implementations, followed by `Operation`,
   `ReferenceDischargeableOperation`, `InterpretableOperation` using the appropriate capability bound,
   `PartiallyEvaluatableOperation`, `BatchableOperation`, `DifferentiableOperation`, `TransposableOperation`,
   `MemberOperation`, and `MemberInterpretableOperation` implementations.
3. Its operation provider trait and implementations where relevant.
4. Its capability trait and implementations. Order explicit context implementations as `EagerContext` for `Array`,
   `EagerContext` for `ArrayIrValue`, `ProjectedContext`, `StagingContext`, `PartialEvaluationContext`,
   `BatchingContext`, `DifferentiationContext`, and `TranspositionContext`.

Keep shared blanket implementations and substantial helper submodules shared; do not duplicate them to reproduce
this ordering. Keep additional universe-specific implementations of a shared operation type and their tests in its
defining module rather than introducing a separate module solely for those implementations. Reuse existing production
and testing macros from `crate::macros` where applicable, following their current contracts and importing them
unqualified as required by the Rust conventions. For example, elementwise operations can use
`define_elementwise_operation!` and `define_elementwise_capability!` rather than restating shared boilerplate.

Prefer validated builders such as `ArrayType::with_sharding` over direct metadata field assignments when constructing
or updating operation output types, and propagate validation errors through existing conversions.

## Terminology

In `ryft-core`, call the values that an operation or instruction consumes its "inputs" (e.g., in names, comments,
diagnostics, and tests), following `Instruction::inputs`, `Operation::infer_output_types(.., input_types)`, and
`InputRegionProvenance::Input`, rather than "operands". Write "region input", "instruction input", or "program
input" wherever more than one kind appears nearby. Keep "operand" only where a specification uses it as the role
name of one particular input or inside an attribute name (e.g., the `operand` of StableHLO `gather`, `scatter`, and
`ragged_all_to_all`, and attributes such as `operand_batching_dims`), and in `ryft-xla` MLIR code. A specification
that calls every input an operand (e.g., most StableHLO operations or XLA custom calls) is not a reason to keep it.

## Integration and Verification

For a new operation, trace the nearest analogous operation through exports, operation families, providers,
capability bundles, composite lifting/projection, backend conversions, and lowering dispatch. Update the pieces
needed by the requested operation; a leaf implementation alone does not establish support. Keep explicit unsupported
paths and document their actual technical blockers. Review backend contracts as well as core type inference.

Follow the canonical primary test order, macros, and full-program rendering assertions in the
[Operations, Types, And Attributes testing guidance](../../unit-testing-guidelines.md#operations-types-and-attributes).
Use `arithmetic.rs` for simple operation layout and the nearest relevant operation for specialized cases. Keep
reference discharge, region contracts, sharding, and nested-transform regressions beside the behavior they verify.
Choose finite-difference oracles only where they express the derivative's intended contract.

Run source-matched verification appropriate to the change, starting with focused tests and the required affected
crate suites. Follow the testing guidelines' backend requirement when staged programs change (at least the
`ryft-xla --lib` suite) and the repository's macro integration requirement when a core contract consumed by derives
changes. Use the repository's 300-second command limits and memory/concurrency safeguards. Do not substitute a
successful type check or an earlier binary for execution of the final changed behavior.

Finish with the independent review in [Coverage and final review](references/review.md). Report the implemented or
reviewed surface, verification evidence, and remaining limitations without claiming exhaustive correctness.
