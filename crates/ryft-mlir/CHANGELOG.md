# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/).

<!-- next-header -->
## [Unreleased] - Release Date

### Added

- Added `ryft_mlir::Error` and converted instances of panics throughout the library to fallible functions.
- Added missing `affine` dialect operations.
- Added missing `arith` dialect operations.
- Added support for the `async` dialect.
- Added support for the `emit_c` dialect.
- Added support for the `gpu` dialect.
- Added support for the `linalg` dialect.
- Added support for the `llvm` dialect.
- Added support for the `memref` dialect.
- Added support for the `nvgpu` dialect.
- Added support for the `pdl` dialect.
- Added support for the `scf` dialect.
- Added support for the `shape` dialect.
- Added support for the `sparse_tensor` dialect.
- Added support for the `tensor` dialect.
- Added support for the `transform` dialect.
- Added support for the Triton `tt` dialect.
- Added support for the Mosaic GPU dialect.
- Added support for the Mosaic TPU dialect.
- Added StableHLO mesh and sub-axis attribute wrappers.
- Added signless and unsigned integer `Context` constructors for widths `1`, `2`, `4`, `8`, `16`, `32`, `64`, and
  `128`, plus `bool_type` as an alias for `i1_type`.

### Changed

- Updated the StableHLO `composite` operation to support regions.
- Minor performance optimizations for some of the `shardy` dialect builders.

## [0.0.2] - 2026-03-02

## Changed

- Upgraded the XLA dependency to a newer version.

## [0.0.1] - 2026-02-22

### Added

- Initial release.

<!-- next-url -->
[0.0.1]: https://github.com/eaplatanios/ryft/compare/v0.0.1...HEAD
