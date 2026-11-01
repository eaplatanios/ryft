# `ryft-mlir` Conventions

These rules refine the shared repository conventions for `ryft-mlir`. The root workflow and Git safeguards still
apply. Consult these contracts when changing callers in other crates if ownership, context borrowing, or wrapper
conversion behavior is relevant.

- Use [add-mlir-dialect](../../.agents/skills/add-mlir-dialect/SKILL.md) when adding, updating, or reviewing MLIR
  dialect wrappers, registration, passes, or custom dialect C API shims. It owns the dialect layout, typed APIs,
  documentation, coverage, and native shim conventions. Apply it to existing-dialect maintenance as well as new
  dialects; keep future dialect-specific corrections in that skill. General ownership, FFI, context borrowing, and
  testing rules still apply.
- In `ryft-mlir`, do not use `filter_map` or `.ok()` to hide failed wrapper conversions while traversing counted MLIR
  C API collections. Keep low-level collection APIs iterator-shaped with `impl Iterator<Item = Result<_, Error>>` and
  make callers collect with `collect::<Result<Vec<_>, _>>()?` or handle each item explicitly.
- Use this crate and `ryft-pjrt` as the reference style for macro-driven hierarchy modeling over third-party C APIs.
- Non-owning wrapper types in `ryft-mlir` are typically `Copy + Clone` and often end with `Ref`.
- Owning wrapper types in `ryft-mlir` are not `Copy` and implement `Drop` to release C resources.
- In `ryft-mlir`, acquire context borrow guards (via `borrow()` and `borrow_mut()`) around C API calls that can mutate
  or that depend on mutable internal MLIR state.
- Whenever possible, use existing helper macros instead of duplicating FFI boilerplate:
  `mlir_subtype_trait_impls!`, `mlir_op!`, `mlir_op_trait!`, `mlir_*_op!`, `mlir_pass!`, etc.
