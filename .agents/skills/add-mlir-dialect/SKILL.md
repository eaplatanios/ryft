---
name: add-mlir-dialect
description: >-
  Add, extend, or review MLIR dialect wrappers in ryft-mlir, including typed attributes, types, operations, passes,
  and required native shims. Use for dialect support rather than ryft-core operation rules or an OpenXLA upgrade.
---

# MLIR Dialect Support

Implement the requested dialect surface under `crates/ryft-mlir/src/dialects/`, or review existing support against its
pinned upstream contracts. This skill owns dialect-specific implementation, documentation, and coverage conventions.
Read repository [AGENTS.md](../../../AGENTS.md) for workflow, safeguards, and shared convention routing. Read the
[MLIR rules](../../../crates/ryft-mlir/AGENTS.md) for ownership, context borrow guards, and collection APIs, and the
[FFI conventions](../../ffi-conventions.md) for native wrappers. Apply the shared Rust and documentation references
linked by the root. Read the [testing guidelines](../../unit-testing-guidelines.md) before changing tests.

## Establish The Pinned Surface

- Inspect `stable_hlo` and `gpu` for existing attribute/type wrappers, operation traits, constructors, registration,
  and tests. Use current conventions when an older example is incomplete; examples do not override the requirements.
- Identify the actual upstream revision used by the repository and inspect that dialect's TableGen definitions,
  implementation, C API, and official documentation. Follow the dialect's owning project when it lives outside LLVM.
  Do not infer the pinned contract from current upstream documentation alone.
- For a new dialect or a claim of complete support, inventory the full pinned attribute, type, concrete operation,
  and pass surface. Expand included definitions and exclude only abstract/base operation classes from the concrete
  operation count. Record implemented items, intentionally unsupported items, and each concrete technical blocker.
  A useful core subset is not complete support. For a narrow extension, inspect the requested definitions and their
  dependencies without turning that request into an unrelated full-dialect implementation.
- Audit operation arguments, results, regions, attributes/properties, segment-size traits, builders, and typing
  constraints before selecting the Rust API. Keep registration, wrapper exports, native availability, and tests in
  the inventory so that declarations alone are not mistaken for usable support.

## Module And Wrapper Design

- Use a dialect directory with `mod.rs`, `attributes.rs`, `types.rs`, `operations.rs`, and `passes.rs` as applicable.
  Large operation families may use an `operations/` module like StableHLO. Do not leave a non-trivial dialect as a
  flat one-off module or add empty files for surfaces the dialect does not have.
- Prefer the existing `mlir_subtype_trait_impls!`, `mlir_op!`, `mlir_op_trait!`, operation helper, and pass macros.
  Keep paired owned/reference operation types consistent. Write each operation trait and its `mlir_op!` and
  `mlir_op_trait!` declarations explicitly; do not create dialect-local macros that generate wrapper traits.
- Define a special-purpose trait for every operation, exposing its specific attributes, operands, results, and
  regions. Do not stop at a generic wrapper or empty marker trait when the pinned definition provides named APIs.
- Add `DialectHandle::<dialect>()` and dialect-level registration/loading tests whenever a native C API handle exists
  or a small native shim can provide one. Prefer that typed handle over by-name registry fallbacks. Load the dialect
  before constructing dialect-specific entities where required for safety, parsing, or verification.
- Put `location` last in operation constructors and use `L: Location<'c, 't>`.
- Colocate public operation attribute-name constants immediately above the first operation trait, constructor, or
  macro-generated operation group using them. A shared constant belongs before its first user, not in a block of
  unrelated attribute names at the top of the file.
- Inline operation-specific attribute, segment, and async-token access logic in the owning trait or constructor
  unless a helper is genuinely shared across dialect modules. Avoid tiny helpers for attribute casts or extraction,
  segment-size extraction, operand slicing, scalar/array attribute construction, or generic builder setup.
- For a single operand/result in a segment, use `dense_integer_32_array_attribute_segment_range(...)` and validate
  the returned range length explicitly. `dense_integer_32_array_attribute_usize_value(...)` returns the segment
  length, not its start, and must not serve as a flat operand/result index.
- Prefix remaining dialect-local declarative macros with the dialect/module prefix, such as `gpu_`, unless they are
  intentionally shared across dialect modules.

When a native C API is missing or text rendering/parsing would otherwise implement a custom attribute/type, read
[Native Dialect Shims](references/native-shims.md). Follow the existing GPU implementation for direct native
construction and access rather than adding round-trip parsing overhead.

## Documentation

Explain each operation's semantics, include a `# Example` with its MLIR rendering, and link to the precise official
operation documentation. Constructor documentation links back to the operation semantics and explains non-obvious
arguments using the repository's `# Parameters` style. Use lists for operands/results rather than Markdown tables.
Do not add boilerplate Rust constructor examples unless calling the API is non-obvious. Document dialect scope and
concrete unsupported surfaces accurately rather than claiming support from the existence of wrapper declarations.

## Tests And Verification

Use the shared testing guidelines for assertion style, fixtures, naming, error paths, and timeout limits.
Follow module definition order and keep each attribute, type, and operation's coverage together:

- Attributes: construction/accessors, equality, display/debug, and casting, following StableHLO attribute tests.
- Types: construction/accessors, equality, display/debug, parsing, and casting, following StableHLO type tests.
- Operations: one primary focused test per concrete operation. Put meaningful additional edge-case regressions
  beside it. Do not replace individual operation tests with a broad scenario containing many unrelated operations.
- Attribute/type behavior tests may be separate, as in the StableHLO and GPU modules; one test is not a total cap on
  coverage for an attribute, type, or operation.
- Operation tests build the smallest valid containing module programmatically with typed constructors. Assert typed
  operands, results, attributes/properties, regions, and collection lengths before insertion where detached behavior
  matters. Verify the module and compare its complete canonical `module.to_string()` rendering with `indoc!` and
  `pretty_assertions::assert_eq`. Avoid parsing and walking a module unless no constructor exists or parsing is the
  behavior under test. Inline trivial context/registry setup instead of hiding one or two lines in tiny helpers.
- Test typed dialect namespace, registry insertion, context registration, loading, and repeated-load identity.
  Reuse existing shared attribute/type test helpers for common wrapper contracts; also cover the dialect-owned API.

Run scoped dialect tests and affected crate verification against final source, with the repository's initial
300-second timeout per command. A new native symbol needs a real consumer link against the rebuilt native artifact;
Rust type-checking against an older binary cannot establish its availability. Native changes also follow the shim
reference's verification guidance. If this task upgrades OpenXLA/PJRT/MLIR, use the
[upgrade-xla skill](../upgrade-xla/SKILL.md) and its full suite/artifact workflow; adding a wrapper alone does not
authorize an upgrade or release.

Before finishing, compare the inventory with actual wrappers, exports, tests, and native symbols. Report implemented
scope, verified outcomes, concrete unsupported pieces, and blocked checks. Preserve TODO annotations and unrelated
working-tree/index changes under the repository workflow.
