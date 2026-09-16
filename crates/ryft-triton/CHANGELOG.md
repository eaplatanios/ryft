# Changelog

## [Unreleased] - Release Date

### Added

- Direct typed Triton lowering and native compiler invocation for portable Ryft kernels.

### Changed

- Compile through the standard `ryft-xla-sys` C ABI using a cloned typed MLIR module instead of a standalone executable.
- Record source/toolchain identity and linked capabilities in schema 2; remove executable paths, hashes and protocols.
- Use cooperative cancellation around synchronous compilation, retaining bounded artifacts and diagnostics.
