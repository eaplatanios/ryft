# `ryft-pjrt` Conventions

These rules refine the shared repository conventions for `ryft-pjrt`. The root workflow and Git safeguards still
apply. Consult these contracts when changing callers in other crates if PJRT ownership, option borrowing, error
handling, or extension delegation is relevant.

- For PJRT API upgrades and argument-struct synchronization, also use
  [upgrade-xla](../../.agents/skills/upgrade-xla/SKILL.md); its binding guidance applies even without changing
  the OpenXLA pin.
- Prefer API-invocation macros for PJRT calls and keep handle conversion helpers centralized. Whenever possible, use
  `invoke_pjrt_api_void_fn!`, `invoke_pjrt_api_error_fn!`, `invoke_distributed_api_error_fn!`, etc., instead of
  duplicating FFI boilerplate.
- Keep `ffi` modules at the bottom of files with explicit C struct/function pointer definitions.
- Continue using `OnceLock` to memoize expensive API queries (e.g., attributes, descriptions, etc.).
- When a new PJRT option struct contains pointer fields to owned C API objects, expose a safe owned wrapper for the
  pointee and model the option field as a borrow of that wrapper; do not silently pass null because the wrapper is
  missing.
- For public safe Rust APIs, prefer full-word count naming such as `device_count_per_slice` over abbreviated
  `num_*` names. Only preserve upstream `num_*` names in FFI definitions and direct C API field mirrors.
- Owning wrapper types in `ryft-pjrt` are not `Copy` and implement `Drop` to release C resources.
- Return `Result<_, Error>` with the crate-specific `Error` type from fallible functions and map null/invalid handles
  to explicit error variants.
- For PJRT C API argument structs (i.e., `*_Args`) in `ryft-pjrt`, provide `new(...)` constructors that initialize
  the struct fields following the pattern we are already using in this crate:
  - `struct_size: size_of::<Self>()`
  - `extension_start: std::ptr::null_mut()`
  - sensible null/zero defaults for outputs
- Follow [PJRT Extension Conventions](src/extensions/AGENTS.md) for extension modules and their convenience methods,
  including methods on `Client`, `Plugin`, or other core types outside the extension directory. Consult that file
  when changing extension API callers in other crates as well.
