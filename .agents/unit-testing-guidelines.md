# Ryft Unit Testing Guidelines

Use this file as the single detailed reference for `ryft` testing conventions.

## Summary

- Keep unit tests colocated in the owning module under `#[cfg(test)] mod tests`. Make colocated module tests primarily
  cover behavior implemented in that module; move incidental downstream behavior coverage to the operation, transform,
  or helper module that owns it.
- Cover every new or changed behavior with deterministic tests and explicit assertions.
- Name unit tests with a `test_...` prefix.
- Use `pretty_assertions::assert_eq` when comparing rendered strings or structured values where readable diffs matter.
- Use `indoc!` for multiline string assertions such as textual IR, StableHLO, HLO, and program renderings.
- Prefer exact assertions: `assert_eq!`, `assert_ne!`, and `assert!(matches!(...))` with guards for error variants.
- For `Result` success paths, prefer `assert_eq!(operation(), Ok(expected))` when the expected value is stable.
- For error paths, assert the concrete error variant and `ryft`-owned message text.
- Await asynchronous transfer, copy, execution, or callback completion before asserting final state or output contents.
- Reuse shared crate test helpers instead of recreating client, plugin, context, backend, or fixture setup.
- Keep tests flat and readable; extract helpers only when they encode a repeated semantic testing contract.
- When testing a later-stage validation error, keep earlier-stage fixtures well-formed unless malformed construction
  is the behavior under test.
- Run scoped verification for the crate or module touched, using a 300-second timeout for local test commands unless the
  user asks for a longer run.

## Reference Backends

- Use `ryft_core::backends::scalars::Scalar` as the concrete eager value for scalar-universe coverage (i.e., programs 
  typed by `DataType`) and `ryft_core::backends::arrays::Array` for array-universe coverage (i.e., programs typed by
  `ArrayType`). Both store honestly typed payloads (`Array` is a row-major `Vec<Scalar>`), so value-level tests can and
  should assert exact element data types, complex values, and exact low-precision floating-point encodings, and not
  `f64` approximations of them.
- In `ryft-xla`, import the reference array value as `use ryft_core::backends::arrays::Array as TestArray;` so that it
  does not collide with the XLA buffer-backed `Array`. When both backends implement an operation, keep them value-level
  consistent.
- Use the `check_gradient!` macro from `ryft_core::macros` as the finite-difference oracle for gradient rules. You
  must use `check_gradient!(@scalar, ...)` for `Scalar`-valued functions and `check_gradient!(@array, ...)` for
  `Array`-valued functions, and `TestRegionOperation` for testing region-carrying program machinery.

## Structure

- Keep setup local and explicit. A single test may cover a normal path plus nearby edge cases when they share the same
  setup and read better together.
- Use `pub(crate) mod tests` only when the module intentionally exposes shared test helpers to sibling modules.
- Put local test helpers near the top of the test module, before the first `#[test]`.
- Define test-only types inside the single test that uses them; keep module-level test types for shared test fixtures.
- Prefer one focused test per behavior family. Avoid broad omnibus tests unless the setup cost is high and the behavior
  is naturally exercised as one scenario.
- Prefer flat sequences of explicit assertions over local helper closures or loops. Compact table or array loops are
  acceptable for exhaustive enum and conversion roundtrips when every case uses the same assertion shape.
- Inline trivial type constructors or one-line setup expressions when a helper would only hide simple test intent.
- For tests that exercise downstream validation or projection logic, construct inputs that satisfy upstream builder
  invariants. Use malformed input programs only in tests that assert those malformed-program errors directly.

## Imports

- Preserve normal import grouping inside test modules:
  `std`, third-party crates, `crate::...`, then `super::...` when needed.
- Import only the helpers and types needed by the test module.
- Import `pretty_assertions::assert_eq` in tests that compare strings, rendered IR, maps, vectors, or larger structs.
- Import `indoc::indoc` for multiline expected strings instead of concatenating or manually managing indentation.

## Assertions

- Prefer exact value assertions over weak predicates. Use `.contains(...)` only when a dependency or backend appends
  unstable context.
- Keep expected error messages exact when ryft owns the message. Error messages should follow repository style:
  lowercase first word and no trailing punctuation.
- Use this pattern for error assertions:

  ```rust
  assert!(matches!(
      operation(),
      Err(Error::InvalidArgument { message, .. }) if message == "precise lowercase message",
  ));
  ```

- Assert both `Display` and `Debug` for user-facing wrappers when those implementations are part of the public surface.
- For equality and hashing behavior, test self-equality, inequality against distinct handles or values, and map lookup
  behavior when `Hash` is implemented.
- For collection-returning APIs, assert collection order with `collect::<Vec<_>>()` when order is part of the contract.
- For fallible backend-dependent APIs, assert the acceptable error variants with `matches!` and only run success-path
  assertions after checking the result is `Ok`.

## Operations, Types, And Attributes

- Create the smallest valid context/module/function/block that exercises the behavior under test.
- Load MLIR dialects before constructing dialect-specific entities when the dialect is required for parsing,
  construction, or verification.
- Use default or unknown locations for ordinary operation tests. Use specific file locations only when location behavior
  itself matters.
- For operation constructor tests, verify the containing operation or module, then compare the complete rendered output
  with `indoc!` and `pretty_assertions::assert_eq`.
- Test operations individually where possible. Prefer full-string equality assertions over partial `.contains(...)`
  checks for rendered IR.
- When testing operation accessors, assert operands, results, attributes, properties, and collection lengths before
  inserting the operation into a final block if detached-operation behavior matters.
- For wrapper hierarchies, cover construction, context ownership, accessor values, equality within one context,
  inequality for different values, inequality across contexts, display/debug rendering, and upcast/downcast behavior.
- For parsing APIs, include one valid parse and one invalid parse when the API exposes a fallible parser.
- Dump-style tests may only assert that `dump()` runs without crashing when stderr capture is not available. Add a short
  comment explaining that limitation.

## Backend And Platform Behavior

- Use shared platform helpers for platform/backend-specific tests. In crates with a macro like `test_for_each_platform!`
  for PJRT, for example, use that macro for behavior that should work across enabled backends.
- Match on explicit platform identifiers for documented backend differences instead of hiding them behind broad
  `.is_ok()` assertions.
- Keep backend expectations concrete. For example, one backend may return `Unimplemented` for an API while another
  returns concrete metadata; assert those differences directly.
- Use CPU-only helpers for CPU-specific behavior and broader platform helpers for cross-backend behavior.
- Do not manually load backend plugins or clients when a crate-level test helper already exists.

## FFI, Unsafe, And Asynchronous Behavior

- For C API wrapper constructors, include null-pointer coverage next to successful construction tests.
- For unsafe wrapper APIs, make the unsafe call visible in the assertion and pair it with a safe API assertion
  when possible.
- For serialization, deserialization, proto, C API, and string conversions, test roundtrips with stable expected values.
- For asynchronous operations, await the returned event, future, or completion handle before asserting output contents
  or invoking dependent callbacks.
- For transfer tests, assert both data movement and completion state. Verify failure paths by injecting a concrete
  `Error` and matching the resulting variant and message.
- For callback tests, use deterministic counters or flags and bounded waits; avoid sleeps without an explicit deadline.

## Test Data

- Keep data small and literal unless the behavior requires a larger payload, such as a long-running execution
  used to test cancellation or poisoning.
- Use concrete byte arrays for host/device transfer tests so that the expected contents are unambiguous.
- Use `to_ne_bytes()` when constructing typed device buffers from scalar values consumed as native-endian bytes.
- Use `HashMap::from([...])` for small expected maps.
- Sort generated vectors before comparison when the API being tested does not guarantee a stable ordering.
- Colocate constants with the local test type that owns their semantics, such as associated constants on test-only
  operation structs.
- Use OS-assigned loopback ports for local networking or distributed-runtime tests instead of hardcoded ports.
- Use multiline `indoc!` fixtures for IR, StableHLO, HLO, or program text.

## Helper Extraction

- Add reusable helpers to the crate root `tests` module when they are broadly useful across the crate.
- Add narrow helpers to the local `tests` module when they only support one module or closely related sibling modules.
- Prefer helper functions or declarative macros when similar test patterns repeat meaningfully.
- Do not extract a helper just to shorten two nearby assertions if the helper would hide simple test intent.

## Verification

- For test-only changes, run `cargo fmt --check` and the scoped crate test command that covers the changed tests.
- Prefer targeted tests first when iterating, then run the affected crate `--lib` suite before handoff when behavior or
  shared helpers changed.
- For dependency, FFI, OpenXLA, PJRT, MLIR, or runtime-attribute upgrade work, do not stop at smoke tests once core
  crate code has changed. Run the full affected crate `--lib` unit suites so printer-format churn, runtime-attribute
  drift, and environment-sensitive assumptions are caught before handoff.
- Use enabled backend features only when the changed behavior depends on them.
- Use a 300-second timeout for local test commands unless the user asks for a longer run.
