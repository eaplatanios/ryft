# Binding and Compatibility Audit

Use the native [Contribution checklist](../../../../crates/ryft-xla-sys/README.md#contribution) as the detailed upgrade
authority. Read the current source and pinned upstream headers/TableGen/Protobuf definitions rather than copying
version numbers, symbol lists, or historical assumptions from this skill.

## PJRT and Generated Bindings

Compare `xla/pjrt/c/pjrt_c_api.h` and every relevant PJRT extension header between the old and new OpenXLA revisions.
Audit extension declarations beyond the headers currently passed to bindgen so a newly added extension is not missed.
Update handwritten definitions in `crates/ryft-pjrt/src` and `src/extensions`, adding new supported extension modules,
documentation, and tests according to the [PJRT conventions](../../../../AGENTS.md#ryft-pjrt).

When upgrading PJRT or XLA FFI structs, update both the raw `ffi` argument structs and every corresponding safe Rust
wrapper, convenience method, and test in the same change. Newly added nested option structs must be usable through
the matching safe wrapper or safe method argument, not only through `ffi`. Follow the shared pointer-ownership and
`*_Args::new(...)` initialization rules in `AGENTS.md`; do not silently substitute null for a missing safe API.

`crates/ryft-xla-sys/src/bindings.rs` is bindgen output with deliberate manual adjustments. Generate against the new
artifact's headers using the `generate-bindings` feature; inspect `build.rs` and its `OUT_DIR/bindings.rs` output before
updating the checked-in file. Reapply these adjustments after regeneration:

1. Prepend the existing manual prelude: the `#![allow(...)]` attribute, opaque `PJRT_Api` handle and `GetPjrtApi`,
   and opaque `XlaCustomCallStatus` with its two functions.
2. Remove generated `PJRT_*` and `PLUGIN_Profiler*` items, generated `XlaCustomCallStatus` items, and the
   `XLA_FFI_API_MAJOR` / `XLA_FFI_API_MINOR` constants. `ryft-pjrt` owns the handwritten PJRT FFI modules.

Preserve the intended retained declarations, inspect regeneration diffs, and check that no duplicate manual/generated
symbols remain. Update the generator's input header list and native archive exports if the upstream surface requires
it; regenerating the old input list alone does not establish coverage.

## Other Native Contracts

- Keep Rust and C++ distributed-runtime bridge structs and signatures synchronized, including field types, ownership,
  callback contracts, and the wrappers that expose them.
- Synchronize `crates/ryft-xla-sys/src/protos.rs` with its corresponding exact upstream `.proto` files, including
  message fields, tags, enums, and nested definitions used by consumers.
- Resolve the precise LLVM/MLIR, StableHLO, and Triton revisions selected by the new OpenXLA source. Compare their
  exposed C APIs and dialect definitions with our bindings and safe wrappers, including changed operation contracts,
  attributes, types, passes, registration, and serialization compatibility. A different StableHLO version can require
  `ryft-mlir` changes even without an independently requested dialect feature.
- If JAX changes, audit its exact pinned Mosaic definitions and native contracts, including dialect APIs, serde/resource
  versions, custom-call registration, and build/link dependencies described in the README. Do not copy transient
  version constants or private wheel entry points into a new public ABI.
- Keep the source/artifact boundary consistent: headers, TableGen inputs, Protobuf sources, compiled symbols, and Rust
  declarations must correspond to the same revisions. Use the README's archive and symbol contract checks with the
  exact source roots and archive being shipped.

Changes to native compiler APIs may reach `ryft-triton`, `ryft-xla`, and other consumers beyond the usual sys/PJRT/MLIR
crates. Inspect actual callers, update affected safe interfaces and tests, and record unavailable platform or hardware
verification rather than inferring runtime support from a successful build.
