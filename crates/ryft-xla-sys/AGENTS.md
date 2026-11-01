# `ryft-xla-sys` Conventions

These rules refine the shared repository conventions for `ryft-xla-sys`. The root workflow, Git safeguards, and
generated-file restrictions still apply. Consult these contracts when changing native ABI consumers in other crates.

- Use [upgrade-xla](../../.agents/skills/upgrade-xla/SKILL.md) for dependency upgrades, native artifact/build metadata,
  generated bindings, Protobuf synchronization, distributed-runtime bridge maintenance, and release verification.
  Apply its relevant guidance to focused maintenance as well as full upgrades; keep future upgrade-specific corrections
  in that skill. For custom MLIR dialect C API bindings, also use
  [add-mlir-dialect](../../.agents/skills/add-mlir-dialect/SKILL.md).
- In custom `ryft-xla-sys` C/C++ and Rust FFI identifiers, use regular UpperCamel acronym casing in function and enum
  type names, such as `Gpu`, `Mlir`, and `Mma`; reserve all-uppercase acronym spelling for all-caps C constants and
  upstream symbols that already use it.
- Use `#pragma once` for source-owned C/C++ headers instead of include guards.
- In custom C++ FFI shims, inline tiny nullable handle conversions at the getter call site instead of adding local
  helpers when the helper only wraps a null check for one or two nearby functions.
