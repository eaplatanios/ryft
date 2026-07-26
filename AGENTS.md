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
   You must start tackling non-trivial tasks by writing a plan to `.tasks/plan_<task_name>.md` with checkable items,
   letting the user modify it before you start executing on it. While executing on a plan, you must mark completed
   items as such in that file, adding a high-level summary of what you did for each step in a new review section.
   If the user explicitly waives planning for a narrowly scoped follow-up change, execute directly and keep the change
   tightly scoped instead of forcing a new plan file.
2. **Subagents:** Use subagents liberally to keep the main context clean. Offload research, exploration, and parallel
   analysis to subagents. Always use subagents for complex programs and stick to one task per subagent for focused
   execution.
3. **Self-Improvement:** After important corrections from the user, update the `AGENTS.md` file such that you do not
   require the same corrections in the future. Write rules for yourself that will prevent you from making the same
   mistake in the future. You must ruthlessly iterate on these rules until your rate of making mistakes drops based
   on those rules.
4. **Verification:** Never consider a task as completed without first proving that it is. Look at the diff between the
   code before and after your changes to determine what changed and needs testing. Then, ask yourself "Would a staff
   engineer approve this? Also, what tests would they want me to run or even add to do so?". Run tests, check the logs,
   demonstrate correctness, and iterate if you are not there yet. When running potentially expensive verification
   commands locally, start with an explicit timeout of 300 seconds per command unless the user asks for a longer run.
   Do not let test or benchmark commands sit for many minutes by default. If a Rust verification command causes
   `rustc` to be killed or to grow to extreme memory use, stop rerunning broad checks and first reduce the generic
   type surface or trait-solver obligation graph introduced by the change. Never run `git stash` (or any command that
   mutates unrelated working-tree files) to work around a build broken by concurrent edits in another crate. Those
   uncommitted changes may belong to a concurrently running agent and a partial `stash pop` can silently drop them. If
   a crate you depend on is transiently broken by such edits, verify your own change in isolation instead (e.g., a
   throwaway crate outside the workspace that depends only on the crates you changed). More generally, never run
   `git checkout <path>`, `git restore <path>`, or `git reset` on tracked files during a session. Large refactors live
   uncommitted in the working tree for hours, and such commands silently destroy that work. To inspect the committed
   version of a file, use read-only commands like `git show HEAD:<path>` and `git diff HEAD -- <path>` instead. The
   sole exception is a clean, dedicated staging worktree used by a reviewed extraction plan after the complete source
   tree has been committed and pushed to an immutable archive branch. In that worktree only, `git restore
   --source=<immutable-archive-or-reviewed-integration-ref> -- <explicit-paths>` may restore a documented increment's
   explicit paths. Verify the staging worktree is clean first; restore a whole path only when the reviewed plan assigns
   its complete delta to that increment, and use patch mode for paths shared with later work. Never target the worktree
   root, a directory broader than the increment, a glob, or an unresolved variable. This exception never applies in
   the owner checkout and does not permit `git checkout <path>`, `git reset`, `git clean`, or `git stash`.
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
- For small `Copy` types, prefer passing values directly to functions instead of borrowing them unnecessarily.
- In transform replay code, prefer the established `InterpretableOperation` path for tracer-valued staging unless the
  user explicitly asks for a different abstraction. If an experimental staging hook or lowering path is removed or
  superseded, delete its public trait methods, helper functions, and test-only scaffolding instead of leaving unused
  compatibility layers behind.
- Prefer normal method-call syntax for receiver-based calls (for example, `self.name()`, `operation.result(0)`, or
  `attribute.cast::<TypeAttributeRef>()`), and avoid UFCS/static-like syntax such as
  `crate::Operation::name(self)` or `crate::Attribute::cast::<TypeAttributeRef>(&attribute)` unless disambiguation is
  actually required.
- When replacing a bespoke cross-cutting capability with something more general, prefer a small named trait over a
  higher-order helper function when the call sites need one reusable semantic contract (for example, broadcasting).
- When centering a capability on a trait, move the whole API surface onto that trait instead of keeping a split between
  inherent methods and trait methods, to the extent possible.
- Model first-class dimension arithmetic as one nominal operation type per primitive and centralize only the genuinely
  shared contract in `ArithmeticDimensionOperation`, following `ElementwiseOperation`. Do not encode arithmetic
  selection as a tag inside one payload or reuse stateless array/scalar arithmetic payloads that cannot retain a fresh
  dimension result identity and bounds.
- For simple capability/provider impls, keep bounds to the minimum needed for the impl target to be well-formed and for
  the method body to type-check. Do not copy broader bounds from neighboring trait impls unless the provider method
  itself uses the capability.
- In capability-focused modules, prefer inlining small single-use local helper functions back into the owning trait
  impls when that makes the implementation easier to read.
- For module moves and path migrations, do not introduce compatibility shims or re-export bridges unless the user
  explicitly asks for them. Default to updating all in-repo use sites to the new canonical path.
- For enums with straightforward tuple or unit variants, prefer using the variants directly instead of adding
  redundant constructor methods unless those constructors add validation or the user explicitly asks for them.
- For small, one-off data-shaping logic used in only one or two nearby methods, prefer inlining the conversion at the
  call site instead of extracting a helper that adds indirection without meaningful reuse.
- When an existing `ryft` abstraction already encodes a concept (for example, mesh axis types), do not introduce a
  parallel ad-hoc representation of the same concept in a new module. Derive semantics from the canonical
  abstraction and keep one source of truth.
- Prefer putting short and simple type bounds directly in generic type declarations typically including the primary
  identifying bound for each parameter (e.g., `T: Type`, `V: Value<Type = T>`, `O: Operation<T>`, etc.). Move long or
  structurally complex bounds, especially associated type constraints and bounds that wrap poorly under `rustfmt`,
  into `where` clauses.
- Declare the type-descriptor parameter before the value parameter in generic parameter lists (e.g.,
  `<T: Type, V: Value<Type = T>>`, not `<V, T>`).
- Use `C` for generic parameters whose semantic role is a context (i.e., `Context`, `StagingContext`, `BatchingContext`,
  `DifferentiationContext`, etc.) and `D` for generic parameters whose semantic role is a domain (i.e., `Domain`,
  `CompilationDomain`, etc.). If `C` or `D` would collide with an existing payload, constant, or capture parameter,
  rename that non-context/non-domain parameter to a specific name such as `Constant`, `Capture`, or `Payload`.
- Order type bounds preferably as follows: `Clone`, `Debug`, `Display`, `PartialEq`, `Eq`, `PartialOrd`, `Ord`, `Hash`,
  `Type`, `Value`, `Typed`, `Parameter`, `Operation`, `LinearOperation`, `DifferentiableOperation`,
  `SupportsZero`, `SupportsOne`, `SupportsZeroLike`, `SupportsOneLike`, `SupportsNeg`, `SupportsAdd`, `SupportsSub`,
  `SupportsMul`, `SupportsDiv`, etc.
- `Type` requires `Clone + Debug + Display + PartialEq + Parameter`, so a `T: Type` bound already implies all of those.
  Never write `T: Parameter + PartialEq + Type` (or any subset). Just write `T: Type`.
- When a helper semantically belongs to an existing core type such as `Program`, prefer an associated function in the
  relevant `impl` block over a free function unless there is a clear reuse reason that truly spans multiple owners.
- When a generic API is centered on a parameterized input or output family, prefer using that family's canonical
  reparameterized form (e.g., `Input::To<T>`) instead of introducing a separate generic metadata type that is
  only coupled by matching `ParameterStructure`s.

#### Formatting & Naming

- Follow workspace formatting (`rustfmt.toml`): `max_width = 120`.
- Use import grouping in this order:
  - `std` imports
  - third-party crate imports
  - `crate::...` imports
  - `super::...` imports
- At in-crate declarative macro call sites, import macros from `crate::macros` (grouping related macros where useful)
  and invoke them unqualified. Reserve `$crate::...` paths for hygienic references inside macro definitions.
- Use full words for variable names and avoid abbreviations or shortened versions of words. Canonical mathematical
  function names that Rust's own standard library uses (e.g., `abs` as in `f64::abs`) count as full names and are
  preferred over spelled-out variants such as `absolute_value`.
- When a function-like call or macro invocation argument list spans multiple lines, include a trailing comma after the
  final argument.
- For canonical conversion helpers in `ryft`, prefer `from_*` naming even when the conversion is fallible and returns
  `Result<_, Error>`; reserve `try_from_*` for trait-based conversions or when an infallible `from_*` already exists.
- Always name the formatter argument `formatter` in `Display` and `Debug` implementations; do not use `f` or any other
  shorthand.
- When writing indentation into `std::fmt::Formatter`, prefer inline width-based formatting like
  `write!(formatter, "{:indentation$}", "")` over per-space loops or one-off helper functions.
- For user-requested renames or removals, always run a targeted search afterward to verify that no old identifier
  references remain in the `ryft` codebase.
- Use `r#type`, `r#await`, etc. when a reserved Rust keyword must be used as an identifier.
- Prefer just `size_of::<T>()` instead of `std::mem::size_of<T>()` and do not `use std::mem::size_of` as it is built in.
- In derive lists that include both `Copy` and `Clone`, list `Copy` before `Clone` and keep those two traits before
  any other derived traits.
- When changing a core trait contract that is consumed by derive macros, run the corresponding macro integration test
  crate (e.g., `ryft-macros-tests`) in addition to the macro crate's own unit tests.
- Precede every non-trivial declarative-macro branch with a concise code comment explaining the accepted public form
  or the internal generation role.

### Error Handling

- For ordinary error enums with declarative variant messages, derive `thiserror::Error` instead of writing manual
  `Display` and `std::error::Error` implementations. Keep manual formatting only when the message requires genuinely
  procedural rendering.
- Do not silently discard fallible operations (e.g., `let _ = ...` on `Result`-returning code is disallowed).
- Use `?` for error propagation when the caller should decide what to do with the error.
- Use explicit `match`/`if let` when mapping to domain-specific errors.
- Use `Result<_, Error>` with the crate-specific `Error` type as the return type for functions that can return errors.
  In the `ryft-pjrt` crate, return `Result<_, Error>` and map null/invalid handles to explicit error variants.
- Colocate domain-specific error types with the module that owns the corresponding API and define those error enums
  immediately after the imports in that file. Use crate-level umbrella error types only for aggregation/wrapping.
- Error messages must start with lowercase text and must not end with trailing punctuation.
- Custom error variants typically carry a `message` and sometimes a `backtrace` via `Backtrace::capture().to_string()`.
- `unwrap()`/`expect()` are allowed only:
  - in tests, or
  - when enforcing internal invariants that were already validated or are contractually guaranteed.
- For invariants that can never fail by design (e.g., extraction right after a `check_count!`, a `NonZeroUsize` built
  from a clamped value, or lookups guaranteed by construction), use bare `unwrap()`. Reserve `expect(...)` with a
  message for conditions that are possible but unrecoverable (e.g., mutex poisoning).
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
- When upgrading PJRT or XLA FFI structs, update both the raw `ffi` argument structs and every corresponding safe
  Rust wrapper, convenience method, and test in the same change; do not stop at the FFI layer after adding new fields.

## Documentation Style

- Every struct, enum, trait, module, and function should have a documentation string. Enum variants should have 
  documentation unless their name is self-explanatory. If an enum variant requires a documentation string, then you must
  add documentation for all variants in that enum.
- When enum variants have documentation strings, keep an empty line between adjacent documented variants even for short
  unit variants or small enums.
- Prefer descriptive documentation that explains semantics and edge cases and includes examples where appropriate.
- When documenting `ryft` behavior, prefer stating the concrete semantics directly instead of saying that the code
  "matches" another system such as JAX unless the external comparison is itself the point of the documentation.
- Link to external official documentation when relevant (e.g., for MLIR, StableHLO, PJRT, XLA, Rustonomicon, etc.).
- When linking external documentation, prefer the most precise relevant page/section instead of just using top-level
  project pages.
- On first mention of in-repo entities, use rustdoc links (e.g., ``[`Operation`]``), and use explicit paths (e.g.,
  ``[`Operation`](crate::Operation)``) when not imported in the current scope.
- For function/method argument documentation strings, use a dedicated `# Parameters` section with this exact bullet 
  style: ``///   - `arg_name`: description...`` and indent wrapped lines under the bullet. You may skip this section
  entirely in cases where the arguments do not need a description or where their role is clear from the main description
  of the function itself.
- Do not end function or method rustdoc blocks with an empty `///` line immediately before the item or its attributes.
  Keep empty rustdoc lines only for internal paragraph, list, table, or code-block separation.
- Do not add documentation strings to `From` implementations or their `from` functions. Document the conversion
  semantics, including paired conversions that form a normalizing cycle between two types, in the documentation string
  of the more domain-specific type (for example, a transform-owned error enum rather than the core `ProgramError`).
- For public `unsafe` APIs, include:
  - what handle/representation is being exposed,
  - why it is unsafe, and
  - why it is still exposed (e.g., extensibility/interoperability).
- For callback- and threading-heavy code, explain the lifetime/ownership invariants in comments. You can refer to
  documentation strings in the core traits of the `ryft-pjrt` crate for examples of this.
- When using ASCII diagrams or Markdown tables in doc comments, align columns for source readability and indent the
  block content slightly so that the raw Rust file remains easy to scan.
- When a `rustdoc` diagram would materially improve documentation clarity, prefer a rendered Mermaid diagram via
  `aquamarine` over a large ASCII graph, while keeping any supporting tables aligned in the raw source.
- When a large `rustdoc` table needs color or layout that Markdown cannot express cleanly, prefer theme-aware raw HTML
  with colors derived from rustdoc CSS variables instead of hard-coded light-theme colors, and prefer a small local
  `<style>` block with namespaced classes over repeating large inline styles on every cell.
- Keep raw HTML and CSS in doc comments within the repo's 120-column limit by breaking declarations, tags, and table
  cells across multiple lines instead of leaving oversized single-line blocks.
- When revising one sentence inside a documentation paragraph, reread and polish the whole paragraph so the final
  wording is coherent as a unit instead of sounding locally patched.
- When editing rustdoc prose, reflow the surrounding paragraph toward the 120-column limit where the text naturally
  allows it; avoid leaving documentation lines arbitrarily short unless they are lists, code blocks, tables, links, or
  readability-driven sentence breaks.

## Testing Guidelines

All ryft unit-testing conventions live in `.agents/unit-testing-guidelines.md`.
Consult that file before writing or revising unit tests.

## Crate-Specific Conventions

### `ryft-mlir`

- Use this crate and `ryft-pjrt` as the reference style for macro-driven hierarchy modeling over third-party C APIs.
- Prefer macro-driven patterns for operation/type/attribute wrappers and pass registration.
- Keep dialect-loading calls before constructing dialect-specific entities when required for safety.
- Keep paired owned/reference operation types (e.g., `Detached...Operation` and `...OperationRef`) consistent.
- For operation constructor APIs, pass `location` as the last parameter and use generic `L: Location<'c, 't>`.
- For operation documentation strings, avoid Markdown tables for operands/results; prefer clear Markdown lists.
- For operation constructor documentation strings, avoid boilerplate Rust call examples unless usage is non-obvious.

### `ryft-pjrt`

- Prefer API-invocation macros for PJRT calls and keep handle conversion helpers centralized.
- Keep `ffi` modules at the bottom of files with explicit C struct/function pointer definitions.
- Continue using `OnceLock` to memoize expensive API queries (e.g., attributes, descriptions, etc.).
- When upgrading PJRT C API argument structs, add the matching safe Rust wrapper or safe method argument in the same
  change instead of leaving newly added nested option structs reachable only through `ffi`.
- When a new PJRT option struct contains pointer fields to owned C API objects, expose a safe owned wrapper for the
  pointee and model the option field as a borrow of that wrapper; do not silently pass null because the wrapper is
  missing.
- For public safe Rust APIs, prefer full-word count naming such as `device_count_per_slice` over abbreviated
  `num_*` names. Only preserve upstream `num_*` names in FFI definitions and direct C API field mirrors.
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

### `ryft-mlir`

- For MLIR dialect operation wrappers, define a special-purpose trait for each operation that exposes the
  specific attributes, operands, results, and regions supported by that operation. Avoid generic operation-only
  wrappers when the dialect specification provides more precise semantics.
- When adding MLIR dialect operation wrappers, audit the pinned TableGen operation definitions for arguments, results,
  regions, segment-size traits, and builders; do not stop at empty marker traits unless the operation definition itself
  has no named API surface.
- In MLIR dialect operation modules, colocate public operation attribute-name constants immediately above the first
  operation trait, constructor, or macro-generated operation group that references them. Do not collect operation
  attribute-name constants in a module-level block at the top of the operations file; when a constant is shared across
  several operations, place it before the first operation group that uses it.
- In MLIR dialect operation modules, inline operation-specific attribute, segment, and async-token access logic
  in the owning trait or constructor instead of adding module-level helper functions, unless the helper is genuinely
  shared across dialect modules. Do not add tiny private wrappers for one-line attribute casts, attribute value
  extraction, operand segment-size extraction, operand slicing, scalar/array attribute construction, or generic
  operation-builder setup.
- When an operation accessor needs one operand/result from an MLIR segment-size range, use
  `dense_integer_32_array_attribute_segment_range(...)` and validate the returned range length explicitly. Do not use
  `dense_integer_32_array_attribute_usize_value(...)` as a flat operand/result index because it returns the segment
  length, not the segment start.
- In `ryft-mlir`, do not use `filter_map` or `.ok()` to hide failed wrapper conversions while traversing counted MLIR
  C API collections. Keep low-level collection APIs iterator-shaped with `impl Iterator<Item = Result<_, Error>>`
  and make callers collect with `collect::<Result<Vec<_>, _>>()?` or handle each item explicitly.
- In MLIR dialect operation modules, write operation traits and their `mlir_op!` / `mlir_op_trait!` declarations
  explicitly instead of adding dialect-local macros that generate the wrapper trait definitions.
- Prefix MLIR-dialect-local declarative macros with the dialect or module prefix (for example, `gpu_`) unless they
  are intentionally shared across dialect modules.
- For MLIR dialect attribute tests, follow the StableHLO attribute test structure: add construction/accessor, equality,
  display/debug, and casting tests for each attribute in the same order in which attributes appear in the module.
- For MLIR dialect type tests, follow the StableHLO type test structure: add construction/accessor, equality,
  display/debug, parsing, and casting tests for each type in the same order in which types appear in the module.
- For MLIR dialect operation tests, mirror the StableHLO operation test structure: build the containing module
  programmatically with typed operation constructors, assert typed accessors before insertion where practical, verify
  the module, and compare the canonical `module.to_string()` output. Avoid parsing a module and walking it with helper
  functions unless the operation has no constructor or the test is explicitly about parsing behavior.
- For MLIR dialect operation tests, write one focused test per concrete operation in the same order as the operation
  module. Do not replace per-operation coverage with broad scenario tests that cover many operations at once.
- In MLIR dialect operation tests, inline trivial context/registry setup at the test site instead of adding tiny helpers
  that hides only one or two lines of code.
- When introducing a new MLIR dialect with public wrappers, use the established directory module layout with separate
  `mod.rs`, `attributes.rs`, `types.rs`, `operations.rs`, and `passes.rs` files as applicable. Do not leave a
  non-trivial dialect as a flat one-off module.
- Before claiming support for a new MLIR dialect, audit the complete pinned TableGen surface for dialect attributes,
  types, operations, and pass hooks, and either implement the full exposed surface or document any intentionally
  unsupported pieces with the concrete technical blocker.

### `ryft-xla-sys`

- Preserve the current `build.rs` resolution order: environment-provided artifact -> verified download -> Bazel build.
- Keep the checksum verification and artifact naming/URL logic explicit and up-to-date.
- `src/bindings.rs` is the result of code that was generated using `bindgen` and then very slightly edited.
  If you regenerate it using `bindgen` make sure to apply the same slight edits that we have already applied,
  after regenerating it. The slight edits are: (1) prepend the manual prelude at the top of the checked-in file
  (the `#![allow(...)]` attribute, the opaque `PJRT_Api` handle with `GetPjrtApi`, and the opaque
  `XlaCustomCallStatus` with its two functions), and (2) remove all generated `PJRT_*` and `PLUGIN_Profiler*`
  items, the generated `XlaCustomCallStatus` items, and the `XLA_FFI_API_MAJOR`/`XLA_FFI_API_MINOR` constants,
  because `ryft-pjrt` provides its own hand-written ffi modules for the PJRT C API.
- When upgrading the OpenXLA pin, also sync the `rules_ml_toolchain` pin in `WORKSPACE` and the hermetic toolchain
  `--repo_env` pins in `.bazelrc` (e.g., `ROCM_DISTRO_VERSION`) with the versions used by the new OpenXLA commit
  (see `workspace3.bzl` and `third_party/gpus/rocm_configure.bzl` in the OpenXLA repository); mismatches fail the
  artifact build at repository-rule evaluation time.
- Keep the Rust and C++ distributed-runtime bridge structs and signatures synchronized.
- Keep the Rust proto message types in `crates/ryft-xla-sys/src/protos.rs` synchronized with the corresponding `.proto`
  files in the OpenXLA repository, whenever upgrading our XLA dependency.
- After publishing or rebuilding `ryft-xla-sys` artifacts, always validate at least one downstream consumer link on
  each affected platform against the published binary so deployment-target mismatches and stale exported-symbol names
  are caught before handoff.
- For OpenXLA / PJRT / MLIR upgrade work, do not stop at smoke tests once core crate code has changed; run the full
  affected crate `--lib` unit suites so runtime-attribute drift, printer-format churn, and environment-sensitive test
  assumptions are caught before handoff.
- For OpenXLA / PJRT / MLIR upgrade work, update the changelog for every crate whose public API, wrappers, tests, or
  generated bindings changed. E.g., do not stop at `ryft-xla-sys` and `ryft-pjrt` if `ryft-mlir` also changed.
- When a user asks you to wait for a specific GitHub Actions run before updating `ryft-xla-sys` release metadata,
  do not report the task as complete until that exact run has reached `completed` and you have refreshed every
  affected published checksum from the finalized release assets.
- Put custom `ryft-xla-sys` MLIR dialect C++ bindings under
  `crates/ryft-xla-sys/src/c++/mlir/dialects/<dialect>.h` and the matching `.cc` file instead of placing
  dialect-specific shims directly under `src/c++`.
- In custom `ryft-xla-sys` C/C++ and Rust FFI identifiers, use regular UpperCamel acronym casing in function and enum
  type names, such as `Gpu`, `Mlir`, and `Mma`; reserve all-uppercase acronym spelling for all-caps C constants and
  upstream symbols that already use it.
- Prefix custom MLIR C API extension functions and opaque/helper enum types with `mlir` / `Mlir`, matching the upstream
  MLIR C API style; do not use project-specific prefixes such as `ryftMlir` / `RyftMlir` for those exported symbols.
- Use `#pragma once` for source-owned C/C++ headers instead of include guards.
- In custom C++ FFI shims, inline tiny nullable handle conversions at the getter call site instead of adding local
  helpers when the helper only wraps a null check for one or two nearby functions.

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
