# Native Dialect Shims

Read this reference when the requested dialect needs a native handle, attribute/type constructors or accessors, or
other C API support that the pinned upstream API does not expose. Read the shared
[FFI conventions](../../../ffi-conventions.md), [MLIR ownership/context rules](../../../../crates/ryft-mlir/AGENTS.md),
and [native naming/header rules](../../../../crates/ryft-xla-sys/AGENTS.md). Use the
[upgrade-xla skill](../../upgrade-xla/SKILL.md) for generated-binding and native artifact requirements, applying only
the binding repair or artifact rebuild mode needed by this task.

## Existing Integration Points

All paths below are relative to the repository root:

- Safe wrappers: `crates/ryft-mlir/src/dialects/gpu/`.
- C declarations/definitions: `crates/ryft-xla-sys/src/c++/mlir/dialects/gpu.h` and matching `gpu.cc`.
- Rust declarations for custom symbols: `crates/ryft-xla-sys/src/mlir/dialects/gpu.rs`, exported through
  `crates/ryft-xla-sys/src/mlir/dialects.rs`.
- Small dialect-registration shim: `crates/ryft-xla-sys/src/c++/mlir/dialects/ub.h` and `ub.cc`, paired with
  `crates/ryft-xla-sys/src/mlir/dialects/ub.rs`.
- Native build/dependencies/header packaging: `crates/ryft-xla-sys/BUILD.bazel`, including the
  `ryft-xla-sys-library` source/header lists and the existing dependency/link wiring.

Inspect these files and the requested dialect's pinned C++ APIs before extending them. Existing generated
upstream C API symbols are available through `ryft_xla_sys::bindings`; custom dialect declarations live under
`ryft_xla_sys::mlir::dialects`. Follow the existing split instead of hand-editing generated `src/bindings.rs`.

## Implementation Contract

- Put custom dialect C++ bindings under `crates/ryft-xla-sys/src/c++/mlir/dialects/<dialect>.h` and the matching `.cc`,
  not directly under `src/c++`.
- Prefix custom MLIR C API extension functions with `mlir` and opaque/helper enum types with `Mlir`, following the
  upstream C API shape. Do not use exported `ryftMlir` or `RyftMlir` prefixes. Preserve upstream spellings for existing
  upstream symbols; apply repository acronym casing to new custom names.
- Prefer direct native type/attribute construction and queries over printing and parsing their custom syntax.
  Preserve the pinned C++ API's null, kind, context, lifetime, and error contracts at the Rust boundary.
- Keep C/C++ declarations, Rust signatures, enum discriminants, constructors/accessors, and native build dependencies
  synchronized. Export the Rust module and register source/header files with the applicable Bazel target. Inspect
  the actual dependency graph rather than assuming every shim has identical link requirements.
- A dialect registration shim should follow existing `MLIR_DECLARE_CAPI_DIALECT_REGISTRATION` /
  `MLIR_DEFINE_CAPI_DIALECT_REGISTRATION` usage and feed the safe typed `DialectHandle` constructor.

## Verify The Native Boundary

Build or otherwise obtain the artifact containing the new symbols before running the safe wrapper tests. Exercise
registration and affected constructors/accessors through a downstream Rust consumer linked against that artifact,
not merely through declarations or an old cached library. Check C/Rust signature agreement and symbol spelling when
linking fails before expanding the patch.

Follow [Finalize Artifacts and Verify](../../upgrade-xla/SKILL.md#finalize-artifacts-and-verify) for the requested
rebuild/link verification: validate at least one downstream consumer link on each affected platform. Loading those
instructions does not require a pin change, publication, or full upgrade. Publication and upgrades remain scoped to
the user's request. Report unavailable native/platform verification explicitly rather than treating `DOCS_RS` or a
type-check-only run as proof of runtime support.
