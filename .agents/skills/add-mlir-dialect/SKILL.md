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

For the unit tests, make sure to have full coverage like we do in `ryft_mlir::dialects::stable_hlo` and
`ryft_mlir::dialects::gpu` with attributes, types, and operations being tested in the order in which they appear
in the corresponding modules.
