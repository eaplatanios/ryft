# Changelog

## [Unreleased] - Release Date

### Added

- Support CUDA 12.9 direct compilation with assembler 12.9.86 and PTX 8.8.
- Direct typed Triton lowering and native compiler invocation for portable Ryft kernels.
- Insert a workgroup barrier when reading a slice after a whole write to the same write-only window in a straight-line
  parallel grid.

### Changed

- Compile through the standard `ryft-xla-sys` C ABI using a cloned typed MLIR module instead of a standalone executable.
- Record source/toolchain identity and linked capabilities in schema 1; remove executable paths, hashes and protocols.
- Use cooperative cancellation around synchronous compilation, retaining bounded artifacts and diagnostics.
