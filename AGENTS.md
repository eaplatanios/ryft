# Instructions

Act like a high-performing senior engineer, always being concise, direct, decisive, and execution-focused.

## Principles

- **Simplicity**: Make every change as simple as possible. Impact minimal code. Solve problems with simple,
  maintainable, production-friendly solutions. Prefer low-complexity code that is easy to read, debug, and modify.
  Do not overengineer. Do not introduce heavy abstractions, extra layers, or large dependencies for small features.
  Choose the smallest solution that solves the problem well. Avoid cleverness unless it clearly improves the outcome.
- **Elegance:** Keep implementations clean, APIs small, behavior explicit, and naming clear. Aim for reusable and
  well-designed abstractions. Write code that another strong engineer can quickly understand, safely extend, and
  confidently ship.
- **No Laziness**: Find root causes and avoid temporary fixes. Remember that you must always act like a senior engineer
  and not as a junior developer who is reactive and is looking for the quickest, but not necessarily most correct way to
  fix issues.
- **Minimize Impact**: Changes should only touch what is necessary. You must always avoid introducing bugs.

## Workflow

When asked to implement a change or add a new feature, you must always follow the following steps:

1. **Planning:** Enter plan mode for ANY non-trivial task (e.g., anything that involves 3+ steps or architectural
   decisions). If something goes sideways, STOP and re-plan immediately; do not keep pushing. Use plan mode for
   verification steps; not just for building. Always write detailed specifications upfront to reduce ambiguity.
   You must start tackling non-trivial tasks by writing a plan to `.agents/tasks/plan_<task_name>.md` with checkable
   items, letting the user modify it before you start executing on it. While executing on a plan, you must mark
   completed items as such in that file, adding a high-level summary of what you did for each step in a new review
   section.
2. **Subagents:** Use subagents liberally to keep the main context clean. Offload research, exploration, and parallel
   analysis to subagents. Always use subagents for complex programs and stick to one task per subagent for focused
   execution.
3. **Self-Improvement:** After ANY correction from the user, update the `AGENTS.md` file such that you do not require
   the same correction in the future. Write rules for yourself that will prevent you from making the same mistake in the
   future. You must ruthlessly iterate on these rules until your rate of making mistakes drops based on those rules.
4. **Verification:** Never consider a task as completed without first proving that it is. Look at the diff between the
   code before and after your changes to determine what changed and needs testing. Then, ask yourself "Would a staff
   engineer approve this? Also, what tests would they want me to run or even add to do so?". Run tests, check the logs,
   demonstrate correctness, and iterate if you are not there yet.
5. **Elegance:** For non-trivial changes pause and ask yourself "Is there a more elegant way to do this?". If a change
   feels hacky, implement an elegent solution knowing everything that you know by this point. For non-trivial changes,
   always challenge your work before presenting it.

When given a bug report, just fix it, without asking for hand-holding. Look at any relevant logs, errors, failing tests,
etc., and then resolve them. Zero context switching should be required from the user for bug fixes; you shouold operate
as an autonomous bug fixing agent. If you encounter any failing CI tests, go fix them without being told how.

## Conventions

You must always adhere to the `ryft` conventions around code, style, documentations, and testing, treating this file as
the authoritative convention source for this repository. If there is any overlapping guidance, alway follow the stricter
rule. Also, when the user asks for changes to the coding, documentation, or testing style that you use, make sure to
update this file so that they do not need to remind you again in the future.

### Code Style

- Prioritize correctness and clarity first. Optimize performance only when needed and explicit.
- Prefer extending existing modules over creating new small files.
- Keep unsafe boundaries explicit and small.
- Prefer explicit ownership and lifetime modeling over implicit behavior.

#### Formatting & Naming

- Follow workspace formatting (`rustfmt.toml`): `max_width = 120`.
- Use import grouping in this order:
  - `std` imports
  - third-party crate imports
  - `crate::...` imports
- Use full words for variable names and avoid abbreviations or shortened versions of words.
- For canonical conversion helpers in `ryft`, prefer `from_*` naming even when the conversion is fallible and returns
  `Result<_, Error>`; reserve `try_from_*` for trait-based conversions or when an infallible `from_*` already exists.
- Use `r#type`, `r#await`, etc. when a reserved Rust keyword must be used as an identifier.
- Prefer just `size_of::<T>()` instead of `std::mem::size_of<T>()` and do not `use std::mem::size_of` as it is built in.

### Error Handling

- Do not silently discard fallible operations (e.g., `let _ = ...` on `Result`-returning code is disallowed).
- Use `?` for error propagation when the caller should decide what to do with the error.
- Use explicit `match`/`if let` when mapping to domain-specific errors.
- Use `Result<_, Error>` with the crate-specific `Error` type as the return type for functions that can return errors.
  In the `ryft-pjrt` crate, return `Result<_, Error>` and map null/invalid handles to explicit error variants.
- Custom error variants typically carry a `message` and sometimes a `backtrace` via `Backtrace::capture().to_string()`.
- `unwrap()`/`expect()` are allowed only:
  - in tests, or
  - when enforcing internal invariants that were already validated or are contractually guaranteed.
- `Drop` implementations may use `expect(...)` when a cleanup failure is unrecoverable.

### Ownership & Lifetimes

- Preserve established lifetime roles like:
  - `'o`: owner/object lifetime,
  - `'c`: context/client lifetime in `ryft-mlir` and `ryft-pjrt`,
  - `'t`: thread pool lifetime in `ryft-mlir`, and
  - `'s`: store lifetime in `ryft-pjrt`.
- Non-owning wrapper types in `ryft-mlir` are typically `Copy + Clone` and often end with `Ref`.
- Owning wrapper types in `ryft-mlir` are not `Copy` and implement `Drop` to release C resources.
- Owning wrapper types in `ryft-pjrt` are not `Copy` and implement `Drop` to release C resources.
- Use `PhantomData` to encode ownership/lifetime relationships explicitly.

### Concurrency & Caching

- Use `Once`, `OnceLock`, `LazyLock`, and `Mutex` for one-time initialization and thread-safe caching.
- Keep global registration operations idempotent and thread-safe.
- In `ryft-mlir`, acquire context borrow guards (via `borrow()` and  `borrow_mut()`) around C API calls that can mutate
  or that depend on mutable internal MLIR state.

### FFI & Unsafe Patterns

- Keep raw FFI details localized and expose safe wrappers by default.
- Use explicit wrapper types around raw handles (e.g., `handle: *mut ...` / `Mlir...` along with `context`/`api`
  fields). Refer to the `ryft-mlir` and `ryft-pjrt` crates for examples.
- For C type wrappers use functions like the following similar to how we are doing in the
  `ryft-mlir` and `ryft-pjrt` crates:
  - `unsafe fn from_c_api(...) -> Option<Self>` or `unsafe fn from_c_api(...) -> Result<Self, Error>`, and
  - `unsafe fn to_c_api(&self) -> ...`.
- For opaque C type bindings, follow this exact pattern (including the comments but modifying the struct name):
  ```
  // We represent opaque C types as structs with a particular structure that is following the convention
  // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
  #[repr(C)]
  pub struct PJRT_TopologyDescription {
      _data: [u8; 0],
      _marker: PhantomData<(*mut u8, PhantomPinned)>,
  }
  ```
- For PJRT C API argument structs (i.e., `*_Args`) in `ryft-pjrt`, provide `new(...)` constructors that initialize
  the struct fields following the pattern we are already using in that crate:
  - `struct_size: size_of::<Self>()`
  - `extension_start: std::ptr::null_mut()`
  - sensible null/zero defaults for outputs
- Whenever possible, use existing helper macros instead of duplicating FFI boilerplate like:
  - `ryft-mlir`: `mlir_subtype_trait_impls!`, `mlir_op!`, `mlir_op_trait!`, `mlir_*_op!`, `mlir_pass!`, etc.
  - `ryft-pjrt`: `invoke_pjrt_api_void_fn!`, `invoke_pjrt_api_error_fn!`, `invoke_distributed_api_error_fn!`, etc.

## Documentation Style

- Every struct, enum, trait, module, and function should have a documentation string. Enum variants should have 
  documentation unless their name is self-explanatory. If an enum variant requires a documentation string, then you must
  add documentation for all variants in that enum and separate them with an empty line.
- Prefer descriptive documentation that explains semantics and edge cases and includes examples where appropriate.
- On first mention of in-repo entities, use rustdoc links (e.g., ``[`Operation`]``), and use explicit paths (e.g.,
  ``[`Operation`](crate::Operation)``) when not imported in the current scope.
- For function/method argument documentation strings, use a dedicated `# Parameters` section with this exact bullet 
  style: ``///   - `arg_name`: description...`` and indent wrapped lines under the bullet. You may skip this section
  entirely in cases where the arguments do not need a description or where their role is clear from the main description
  of the function itself.
- For public `unsafe` APIs, include:
  - what handle/representation is being exposed,
  - why it is unsafe, and
  - why it is still exposed (e.g., extensibility/interoperability).
- For callback- and threading-heavy code, explain the lifetime/ownership invariants in comments. You can refer to
  documentation strings in the core traits of the `ryft-pjrt` crate for examples of this.
- Link to external official documentation when relevant (e.g., for MLIR, StableHLO, PJRT, XLA, Rustonomicon, etc.).
- When linking external documentation, prefer the most precise relevant page/section instead of just using top-level
  project pages.

## Testing Style

- Keep unit tests colocated in each module under `#[cfg(test)]`.
- Every new/changed behavior should be covered by unit tests.
- Use `pretty_assertions::assert_eq` for string/struct comparisons where output readability matters.
- Use `indoc!` for multiline string matching assertions (e.g., for textual IR/program renderings).
- Prefer deterministic tests with explicit assertions.
- Always name unit tests with a `test_...` prefix for consistency across modules.
- For `Result` assertions, prefer `assert_eq!(..., Ok(...))` for success paths and `assert!(matches!(..., Err(...)))`,
  with guards when needed, for error paths, instead of manual `match` + `panic!` blocks.
- For backend-dependent operations (e.g., in `ryft-pjrt`), assert an explicit set of acceptable error variants with
  `matches!` and only run success-path assertions when the result is `Ok(...)`.
- For asynchronous transfer/copy tests, await the returned completion handle before asserting final output contents
  or invoking dependent callbacks.
- Use the `test_for_each_platform!` macro for testing backend-specific behavior in `ryft-pjrt`, or an equivalent one
  for other crates. You can create such a macro if you need it and it does not yet exist in a crate.
- Reuse shared test helpers (e.g, `test_cpu_client`, `test_cpu_plugin`, etc.) instead of reimplementing setup logic.
- When similar test patterns repeat, extract helper functions or declarative macros. Prefer to add them to the `tests`
  module at the root `lib.rs` file of the corresponding crate, like we have already done for some helpers in
  `ryft-mlir` and `ryft-pjrt`.

## Crate-Specific Conventions

### `ryft-mlir`

- Use this crate and `ryft-pjrt` as the reference style for macro-driven hierarchy modeling over third-party C APIs.
- Prefer macro-driven patterns for operation/type/attribute wrappers and pass registration.
- Keep dialect-loading calls before constructing dialect-specific entities when required for safety.
- Keep paired owned/reference operation types (e.g., `Detached...Operation` and `...OperationRef`) consistent.
- For operation constructor APIs, pass `location` as the last parameter and use generic `L: Location<'c, 't>`.
- For operation documentation strings, avoid Markdown tables for operands/results; prefer clear Markdown lists.
- For operation constructor documentation strings, avoid boilerplate Rust call examples unless usage is non-obvious.
- For operation tests, test operations individually where possible, and prefer full-string equality assertions using
  `indoc!` and `pretty_assertions::assert_eq` over partial `.contains(...)` checks.

### `ryft-pjrt`

- Prefer API-invocation macros for PJRT calls and keep handle conversion helpers centralized.
- Keep `ffi` modules at the bottom of files with explicit C struct/function pointer definitions.
- Continue using `OnceLock` to memoize expensive API queries (e.g., attributes, descriptions, etc.).
- See the **PJRT Extension Conventions** section below for conventions related to code in
  `crates/ryft-pjrt/src/extensions`.

#### PJRT Extension Conventions

Each PJRT extension lives in its own module under `crates/ryft-pjrt/src/extensions/`. Use `layouts.rs`, `triton.rs`,
and `ffi.rs` as authoritative references. All new extensions must follow these same patterns.

##### Extension Struct

- Derive `Copy, Clone`. Fields are `handle: *const ffi::PJRT_<Ext>_Extension` and `api: Api`, both private and each
  with a `///` doc comment.
- Provide three `pub(crate)` methods in the first `impl` block:
  - `unsafe fn from_c_api(handle: *const crate::ffi::PJRT_Extension_Base, api: Api) -> Option<Self>`: Checks
    `extension_type` against the expected `PJRT_Extension_Type_*` constant.
  - `unsafe fn to_c_api(&self) -> *const ffi::PJRT_<Ext>_Extension`: Returns the raw handle.
  - `fn api(&self) -> Api`: Returns the stored `Api`.
- Add `unsafe impl Send` and `unsafe impl Sync` for the extension struct immediately after its main `impl` block.

##### `impl` Block Ordering

`impl` blocks for extension modules must appear in the following order:

1. Extension struct definition and core methods (e.g., `from_c_api`, `to_c_api`, `api`).
2. `unsafe impl Send` / `unsafe impl Sync` for the extension struct.
3. Extension-specific domain methods (e.g., `register_handler`, `register_type`).
4. Optional convenience delegation methods for core types like `Device`, etc.
5. Convenience delegation methods for clients in an `impl Client<'_>` block.
6. Convenience delegation methods for plugins in an `impl Plugin` block.
7. `pub(crate)` functions in an `impl Api` block, including `pub(crate) fn <ext>_extension(&self)` that walks the
   `PJRT_Extension_Base` chain and returns `Result<..., Error>`, using `Error::unimplemented` if the extension
   is not present.
8. Additional public wrapper types (e.g., enums, bitflags, borrowed views, etc.).
9. `ffi` module (always appearing last, right before `#[cfg(test)] mod tests`).

##### Convenience Delegation Methods

- Both `Client<'_>` and `Plugin` must always provide `pub fn <ext>_extension(&self) -> Result<..., Error>`
  that delegates to `self.api().<ext>_extension()`.
- If the extension exposes high-level operations (e.g., `register_handler`, `register_type`), you must add
  matching convenience methods on both `Client` and `Plugin` that call `self.<ext>_extension()?.<method>(...)`.
- Convenience method documention strings should cross-reference the canonical method on the extension
  struct (e.g., "Refer to the documentation of [`<Ext>Extension::<method>`] for more information.") to
  avoid duplication.

### `ryft-xla-sys`

- Preserve the current `build.rs` resolution order: environment-provided artifact -> verified download -> Bazel build.
- Keep the checksum verification and artifact naming/URL logic explicit and up-to-date.
- `src/bindings.rs` is the result of code that was generated using `bindgen` and then very slightly edited.
  If you regenerate it using `bindgen` make sure to apply the same slight edits that we have already applied,
  after regenerating it.
- Keep the Rust and C++ distributed-runtime bridge structs and signatures synchronized.
- Keep the Rust proto message types in `crates/ryft-xla-sys/src/protos.rs` synchronized with the corresponding `.proto`
  files in the OpenXLA repository, whenever upgrading our XLA dependency.

## Convention References / Examples

The `ryft-pjrt`, `ryft-mlir`, and `ryft-xla-sys` crates should provide a good reference for our conventions if you want
to look at real examples. For documentation of core concepts, you can refer to the documentation in
`crates/ryft-core/src/parameters.rs` as a good example.

## Commands

The following are some useful commands that you can use while working on the `ryft` project:

- Build all crates in the workspace: `cargo build`
- Build one crate: `cargo build -p <crate>`
- Type-check all crates in the workspace: `cargo check`
- Type-check one crate: `cargo check -p <crate>`
- Run tests for all crates in the workspace: `cargo test`
- Run tests for one crate: `cargo test -p <crate>`
- Run tests keeping their outputs in `stdout`: `cargo test -p <crate> -- --nocapture`
- Format all crates in the workspace: `cargo fmt`
- Format one crate: `cargo fmt -p <crate>`

Generally, you should prefer running commands scoped to the crate that you are currently modifying to reduce iteration
cost and avoid unnecessary cross-crate churn.

## Generated And Special Files

- Do not manually edit generated outputs unless explicitly regenerating them as part of a version upgrade:
  - `crates/ryft-xla-sys/src/bindings.rs`
- Avoid touching unrelated binary/editor artifacts (e.g., `.DS_Store`).
