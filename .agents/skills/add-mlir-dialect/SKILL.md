---
name: add-mlir-dialect
description: Adds support for a new MLIR dialect in `ryft-mlir`.
---

You must look at our implementation of the `stable_hlo` dialect attributes, operations, and types, in
`ryft_mlir::dialects::stable_hlo`, and understand our code style and conventions around code, documentation,
and unit tests.

Then, you must add support for the types, attributes, and operations of the MLIR dialect requested by the user under
the appropriate submodule of `ryft_mlir::dialects`. You should refer to both the MLIR documentation and the actual MLIR
codebase that you can find on GitHub to understand what is supported in the requested dialect and what the right typing
constraints are. You must also refer to the documentation and code of that specific dialect wherever that lives. You
should also use the `ryft_mlir::dialects::gpu` dialect as another reference. Also note that for e.g., the `gpu` dialect,
we added C++ helpers in `ryft-xla-sys` to avoid the overhead of rendering and parsing custom MLIR attributes and types. You may need to do the same for the requested dialect as well.

When adding a dialect, add a typed `DialectHandle::<dialect>()` constructor and dialect-level registration/loading tests
whenever the dialect has a native C API handle or can be backed by a small `ryft-xla-sys` C API shim. Prefer typed
dialect handles in operation constructors over by-name registry fallbacks.

When adding operation wrappers, keep public operation attribute-name constants colocated immediately above the first
operation trait or macro-generated operation group that references them. Do not collect operation attribute-name
constants in a module-level block at the top of the operations file.

When adding or claiming complete support for an MLIR dialect's operations, first enumerate every concrete operation in
the pinned upstream TableGen file, explicitly exclude only abstract/base classes, and do not treat a useful core subset
as complete coverage.

Inline operation-specific attribute, segment, and operand access logic directly in the owning operation trait or
constructor. Do not introduce tiny private wrappers for one-line attribute casts, attribute value extraction, or operand
slicing unless that helper is genuinely shared across dialect modules.

For the unit tests, make sure to have full coverage like we do in `ryft_mlir::dialects::stable_hlo` and
`ryft_mlir::dialects::gpu` with attributes, types, and operations being tested in the order in which they appear in the
corresponding modules.

There must be exactly one test per attribute, type, and operation.

For operation tests, mirror the StableHLO operation test structure: build the containing module programmatically with
typed operation constructors, assert typed accessors before insertion where practical, verify the module, and compare
the canonical `module.to_string()` output. Avoid parsing a module and walking it with helper functions unless the
operation has no constructor or the test is explicitly about parsing behavior. Inline trivial context/registry setup at
each test site instead of adding tiny helpers that hide only one or two lines of code.
