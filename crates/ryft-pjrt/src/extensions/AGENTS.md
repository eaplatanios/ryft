# PJRT Extension Conventions

These rules refine [`ryft-pjrt` conventions](../../AGENTS.md), including the extension-specific implementation order.
The root workflow and Git safeguards still apply. Apply these contracts to extension convenience methods outside this
directory and consult them when changing callers in other crates.

Each PJRT extension lives in its own module under `crates/ryft-pjrt/src/extensions/`. Use `layouts.rs`, `triton.rs`,
and `ffi.rs` as authoritative references. All new extensions must follow these same patterns.

## Extension Struct

- Derive `Copy, Clone`. Fields are `handle: *const ffi::PJRT_<Ext>_Extension` and `api: Api`, both private and each
  with a `///` doc comment.
- Provide three `pub(crate)` methods in the first `impl` block:
  - `unsafe fn from_c_api(handle: *const crate::ffi::PJRT_Extension_Base, api: Api) -> Option<Self>`: Checks
    `extension_type` against the expected `PJRT_Extension_Type_*` constant.
  - `unsafe fn to_c_api(&self) -> *const ffi::PJRT_<Ext>_Extension`: Returns the raw handle.
  - `fn api(&self) -> Api`: Returns the stored `Api`.
- Add `unsafe impl Send` and `unsafe impl Sync` for the extension struct immediately after its main `impl` block.

## `impl` Block Ordering

`impl` blocks for extension modules must appear in the following order:

1. Extension struct definition and core methods (e.g., `from_c_api`, `to_c_api`, `api`).
2. `unsafe impl Send` / `unsafe impl Sync` for the extension struct.
3. Extension-specific domain methods (e.g., `register_handler`, `register_type`).
4. Optional convenience delegation methods for core types like `Device`, etc.
5. Convenience delegation methods for clients in an `impl Client<'_>` block.
6. Convenience delegation methods for plugins in an `impl Plugin` block.
7. `pub(crate)` functions in an `impl Api` block, including `pub(crate) fn <ext>_extension(&self)` that walks the
   `PJRT_Extension_Base` chain and returns `Result<..., Error>`, using `Error::unimplemented` if the extension is
  not present.
8. Additional public wrapper types (e.g., enums, bitflags, borrowed views, etc.).
9. `ffi` module (always appearing last, right before `#[cfg(test)] mod tests`).

## Convenience Delegation Methods

- Both `Client<'_>` and `Plugin` must always provide `pub fn <ext>_extension(&self) -> Result<..., Error>` that
  delegates to `self.api().<ext>_extension()`.
- If the extension exposes high-level operations (e.g., `register_handler`, `register_type`), you must add matching
  convenience methods on both `Client` and `Plugin` that call `self.<ext>_extension()?.<method>(...)`.
- Convenience method documentation strings should cross-reference the canonical method on the extension struct (e.g.,
  "Refer to the documentation of [`<Ext>Extension::<method>`] for more information.") to avoid duplication.
