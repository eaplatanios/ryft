# Rust Conventions

Read this reference when adding, changing, or reviewing Rust code, including its imports, APIs, error handling,
ownership, and synchronization. These shared rules also apply to test code; detailed testing conventions live in
[unit-testing-guidelines.md](unit-testing-guidelines.md).

## Code Style

- Prioritize correctness and clarity first. Optimize performance only when needed and explicit.
- Prefer extending existing modules over creating new small files.
- Order inherent functions as constructors, `with_*` builder functions, field accessors, mutators, extraction or
  consuming functions, then private helpers. Place `with_*` functions immediately after constructors. Keep private
  constructors with constructors, merge adjacent inherent impl blocks with identical bounds, and order the corresponding
  tests by the public functions while keeping edge cases together.
- Keep unsafe boundaries explicit and small.
- Prefer explicit ownership and lifetime modeling over implicit behavior.
- Use the narrowest visibility that reaches actual callers; keep module-internal helpers private.
- For small `Copy` types, prefer passing values directly to functions instead of borrowing them unnecessarily.
- For a function that queries or validates a particular instance, including a type descriptor, use a receiver such as
  `&self` rather than a named first argument of type `&Self`. Prefer normal method-call syntax for receiver-based calls
  (for example, `self.name()`, `operation.result(0)`, or `attribute.cast::<TypeAttributeRef>()`), and avoid
  UFCS/static-like syntax such as `crate::Operation::name(self)` or
  `crate::Attribute::cast::<TypeAttributeRef>(&attribute)` unless disambiguation is actually required.
- When replacing a bespoke cross-cutting capability with something more general, prefer a small named trait over a
  higher-order helper function when the call sites need one reusable semantic contract (for example, broadcasting).
- When centering a capability on a trait, move the whole API surface onto that trait instead of keeping a split between
  inherent methods and trait methods, to the extent possible.
- For module moves and path migrations, do not introduce compatibility shims or re-export bridges unless the user
  explicitly asks for them. Default to updating all in-repo use sites to the new canonical path.
- For enums with straightforward tuple or unit variants, prefer using the variants directly instead of adding redundant
  constructor methods unless those constructors add validation or the user explicitly asks for them.
- For small, one-off data-shaping logic used in only one or two nearby methods, prefer inlining the conversion at the
  call site instead of extracting a helper that adds indirection without meaningful reuse.
- Define bookkeeping types used only within one function at the beginning of that function.
- When an existing `ryft` abstraction already encodes a concept (for example, mesh axis types), do not introduce a
  parallel ad-hoc representation of the same concept in a new module. Derive semantics from the canonical abstraction
  and keep one source of truth. For type-member classification, check `ArrayMemberType`, `DimensionMemberType`,
  `ReferenceMemberType`, and conversion into `ArrayIrType` before adding an operation-specific enum or projection hook.
  Do not add an always-error projection implementation for a family merely to satisfy a generic operation bound.
- Prefer putting short and simple type bounds directly in generic type declarations typically including the primary
  identifying bound for each parameter (e.g., `T: Type`, `V: Value<Type = T>`, `O: Operation<T>`, etc.). Move long or
  structurally complex bounds, especially associated type constraints and bounds that wrap poorly under `rustfmt`, into
  `where` clauses.
- Do not nest a bound on a trait that has a `Self`-referential default type parameter inside an associated-type
  binding of a generic parameter. This concerns the value capability traits (`Reshape<T = <Self as
  Capability>::Universe>` and every other `$capability<T = …>` trait) and `PartialEq`/`PartialOrd`/the arithmetic
  operator traits: write `where C::Value: Reshape`, not `C: Domain<Value: Reshape>`. Both forms mean the same thing to
  `rustc`, but RustRover cannot resolve `Self` inside the nested form and then reports a false "type does not implement
  trait" error at every call that must prove such an implementation. Nested bindings on traits without such defaults
  (`Context<Type: DifferentiableType>`), equalities (`Domain<Type = ArrayType>`), and the supertrait or associated-type
  declarations of trait definitions (`trait X: Domain<Type: ScanType>`, `type Value: Value<Type = Self::Type>`) are
  unaffected and stay as they are.
- Declare the type-descriptor parameter before the value parameter in generic parameter lists (e.g.,
  `<T: Type, V: Value<Type = T>>`, not `<V, T>`). Likewise, declare a parameter type before the structure that is
  parameterized by it (e.g., `Parameterwise<P, S>` and `impl<P: Parameter, S: Parameterized<P>>`).
- Use `C` for generic parameters whose semantic role is a context or domain (e.g., `Context`, `StagingContext`,
  `BatchingContext`, `DifferentiationContext`, `Domain`, or `CompilationDomain`), and `D` for driver parameters. When
  a function accepts a driver, reserve `D` for that driver and use `C` for its context or domain; do not rename the
  driver to `Driver` to free `D` for a domain. If `C` or `D` would collide with an existing payload, constant, or
  capture parameter, rename that payload, constant, or capture parameter to a specific name such as `Constant`,
  `Capture`, or `Payload`.
- Order type bounds preferably as follows: `Clone`, `Debug`, `Display`, `PartialEq`, `Eq`, `PartialOrd`, `Ord`, `Hash`,
  `Type`, `Value`, `Typed`, `Parameter`, `Operation`, `LinearOperation`, `DifferentiableOperation`, `SupportsZero`,
  `SupportsOne`, `SupportsZeroLike`, `SupportsOneLike`, `SupportsNeg`, `SupportsAdd`, `SupportsSub`, `SupportsMul`,
  `SupportsDiv`, etc.
- Place lifetime bounds before trait bounds within a bound list (e.g., `T: 'static + Send + Sync`, not
  `T: Send + Sync + 'static`).
- `Type` requires `Clone + Debug + Display + PartialEq + Parameter`, so a `T: Type` bound already implies all of those.
  Never write `T: Parameter + PartialEq + Type` (or any subset). Just write `T: Type`.
- Do not add inherent type accessors that duplicate `Typed::r#type` (e.g., an inherent `r#type()` or `array_type()`
  returning `&T` on a concrete `Value`): call the `Typed`-provided `r#type()` directly. It returns `Cow::Borrowed` for
  concrete values, so it is a zero-cost borrow wrap; expression-position uses work through `Deref`, and a longer-lived
  reference is obtained by binding the `Cow` and calling `.as_ref()`.
- When a helper semantically belongs to an existing core type such as `Program`, prefer an associated function in the
  relevant `impl` block over a free function unless there is a clear reuse reason that truly spans multiple owners.
- When a generic API is centered on a parameterized input or output family, prefer using that family's canonical
  reparameterized form (e.g., `Input::To<T>`) instead of introducing a separate generic metadata type that is
  only coupled by matching `ParameterStructure`s.

## Formatting & Naming

- Follow workspace formatting (i.e., `rustfmt.toml`): `max_width = 120`.
- Write numeric type suffixes without a separating underscore (e.g., `0f32`, `1.0f64`, and `2usize`, instead of `0_f32`,
  `1.0_f64`, or `2_usize`). Keep underscores used as digit separators where useful.
- Use import grouping in this order:
  - `std` imports
  - third-party crate imports
  - `crate::...` imports
  - `super::...` imports
- For imports from the current crate, use the first module segment after `crate` as the ownership boundary:
  - Import items owned by a different top-level module through the shortest path that re-exports them, measured in
    path segments. Both named and glob `pub use` re-exports in module `mod.rs` files count as intentional facades:
    prefer `crate::arrays::ArrayType` and `crate::programs::Typed` to their deeper defining paths, and fall back to
    the defining path only when no facade re-exports the item (e.g.,
    `crate::programs::types::visit_type_signature_pairs`). The one exclusion is crate-root re-exports: in `ryft-core`,
    never import through `crate::<Item>` (the `lib.rs` re-exports exist for downstream crates); in crates whose root
    re-exports are the intentional named public API, such as `ryft-xla`, `use crate::{...}` root imports are the
    shortest facade and are correct.
  - Import items owned by the same top-level module through their full accessible defining submodule paths, even when
    an ancestor re-exports them through a shorter facade path. For example, `crate::arrays::addressing` must import
    `ArrayType` from `crate::arrays::types::arrays::ArrayType`, not from `crate::arrays::ArrayType`. Items defined
    directly in the shared top-level module remain imported from that module. When the defining module is private to
    a sibling, use its nearest accessible intentional facade instead of widening module visibility solely for imports.
  - Group compatible leaf imports from the same exact module path into one brace group (aliased items such as
    `Complex as ComplexNumber` participate in the group). Keep imports that descend into nested module paths in separate
    `use` statements, and never merge imports across different `#[cfg]`, visibility, or other attribute boundaries.
  - Apply the same shortest-path rule to imports from other workspace crates, including their crate roots: when the
    foreign crate re-exports an item at its root (by name or by glob), import it directly from the root (e.g.,
    `ryft_core::Typed`, `ryft_core::ArrayType`, `ryft_pjrt::Event`); otherwise import through the shortest module
    facade that re-exports it. The prohibition on `crate::<Item>` root imports applies only inside the defining
    crate itself, never to downstream consumers.
  - When two facades expose an item at the same shortest length, import from the module that owns the item (e.g.,
    `Tag` from `crate::operations`, not through another module's incidental re-export of it).
  - Enum-variant imports such as `use crate::arrays::DataType::{F32, F64};` keep their own statement beside the
    owning type's group, because merging them would require nested brace groups, which are never used.
  - Facade `pub use` statements refer to their own child modules by relative path (e.g., `pub use types::{...}`
    inside `arrays/mod.rs`), matching the existing facade files; the rules above govern item *imports*, not the
    re-export statements that define the facades.
  - Test modules follow the same rules for their `crate::...` imports; `use super::*` remains the standard way to
    bring in the owning module's items.
  - `rustfmt.toml` sets `imports_granularity = "Module"`, which enforces the grouping rule under nightly `rustfmt`
    but is inert on stable; it cannot choose the semantically correct facade or defining descendant path, so treat it
    as grouping assistance only. Do not add `group_imports = "StdExternalCrate"`: it merges `crate` and `super`
    imports into one group, which conflicts with the import group order above.
- At in-crate declarative macro call sites, import macros from `crate::macros` (grouping related macros where useful)
  and invoke them unqualified. Reserve `$crate::...` paths for hygienic references inside macro definitions.
- For macros that generate per-type implementations, accept one type per invocation and invoke the macro separately
  for each supported type instead of accepting a list of types.
- Use full words for variable names and avoid abbreviations or shortened versions of words. Canonical mathematical
  function names that Rust's own standard library uses (e.g., `abs` as in `f64::abs`) count as full names and are
  preferred over spelled-out variants such as `absolute_value`. This also applies to names borrowed from external APIs:
  spell out abbreviations such as JAX's `prevent_cse` (e.g., as `optimization_barrier`, named for the mechanism, or
  `prevent_common_subexpression_elimination`), and mention the external name only in documentation.
- When a function-like call or macro invocation argument list spans multiple lines, include a trailing comma after the
  final argument.
- For canonical conversion helpers in `ryft`, prefer `from_*` naming even when the conversion is fallible and returns
  `Result<_, Error>`; reserve `try_from_*` for trait-based conversions or when an infallible `from_*` already exists.
- Name functions that enforce a precondition by returning an error (e.g., `Result<(), Error>`, or a validated input)
  `validate_*`, not `check_*`. Reserve `check_*` for assertion-style macros and test oracles (e.g., `check_count!`,
  `check_gradient!`, and `check_operation_*!`).
- Always name the formatter argument `formatter` in `Display` and `Debug` implementations; do not use `f` or any other
  shorthand.
- When writing indentation into `std::fmt::Formatter`, prefer inline width-based formatting like
  `write!(formatter, "{:indentation$}", "")` over per-space loops or one-off helper functions.
- Use `r#type`, `r#await`, etc. when a reserved Rust keyword must be used as an identifier.
- Prefer just `size_of::<T>()` instead of `std::mem::size_of<T>()` and do not `use std::mem::size_of` as it is built in.
- In derive lists that include both `Copy` and `Clone`, list `Copy` before `Clone` and keep those two traits before
  any other derived traits.
- Precede every non-trivial declarative-macro branch with a concise code comment explaining the accepted public form
  or the internal generation role.

## Error Handling

- For ordinary error enums with declarative variant messages, derive `thiserror::Error` instead of writing manual
  `Display` and `std::error::Error` implementations. Keep manual formatting only when the message requires genuinely
  procedural rendering.
- Do not silently discard fallible operations (e.g., `let _ = ...` on `Result`-returning code is disallowed).
- Use `?` for error propagation when the caller should decide what to do with the error.
- Use explicit `match`/`if let` when mapping to domain-specific errors.
- Use `Result<_, Error>` with the crate-specific `Error` type as the return type for functions that can return errors.
- Colocate domain-specific error types with the module that owns the corresponding API and define those error enums
  immediately after the imports in that file. Use crate-level umbrella error types only for aggregation/wrapping.
- Error messages must start with lowercase text and must not end with trailing punctuation.
- In diagnostics, surround operation names, identifiers, types, attributes, command-line flags, and other literal code
  fragments with backticks, following Rust compiler conventions. Do not use single quotes for code references; reserve
  them for syntax that actually contains single quotes, such as character literals, lifetimes, or serialized formats.
- Custom error variants typically carry a `message` and sometimes a `backtrace` via `Backtrace::capture().to_string()`.
- `unwrap()`/`expect()` are allowed only:
  - in tests, or
  - when enforcing internal invariants that were already validated or are contractually guaranteed.
- For invariants that can never fail by design (e.g., extraction right after a `check_count!`, a `NonZeroUsize` built
  from a clamped value, or lookups guaranteed by construction), use bare `unwrap()`. Reserve `expect(...)` with a
  message for conditions that are possible but unrecoverable (e.g., mutex poisoning).
- `Drop` implementations may use `expect(...)` when a cleanup failure is unrecoverable.

## Ownership & Lifetimes

- Preserve established lifetime roles like:
  - `'o`: owner/object lifetime,
  - `'c`: context/client lifetime in `ryft-mlir` and `ryft-pjrt`,
  - `'t`: thread pool lifetime in `ryft-mlir`, and
  - `'s`: store lifetime in `ryft-pjrt`.
- Name the lifetime of a borrow of an operation, object, or other owner `'o` (e.g., `ResidualProducer<'o, T>`, which
  borrows its producing operation), including in the types and higher-ranked bounds that carry it along. Do not use a
  generic `'a`.
- Use `PhantomData` to encode ownership/lifetime relationships explicitly.

## Concurrency & Caching

- Use `Once`, `OnceLock`, `LazyLock`, and `Mutex` for one-time initialization and thread-safe caching.
- Keep global registration operations idempotent and thread-safe.
