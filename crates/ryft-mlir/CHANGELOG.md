# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/).

<!-- next-header -->
## [Unreleased] - Release Date

### Added

- Added `ryft_mlir::Error` and converted instances of panics throughout the library to fallible functions.
- Added typed attribute accessor functions like `Operation::dense_integer_32_array_attribute`.
- Added `TensorTypeRef::element_type`.
- Added missing `affine` dialect operations.
- Added missing `arith` dialect operations.
- Added support for the `async` dialect.
- Added support for the `bufferization` dialect.
- Added support for the `complex` dialect.
- Added support for the `emit_c` dialect.
- Added support for the `gpu` dialect.
- Added support for the `linalg` dialect.
- Added support for the `llvm` dialect.
- Added support for the `math` dialect.
- Added support for the `memref` dialect.
- Added support for the `nvgpu` dialect.
- Added support for the `nvvm` dialect.
- Added support for the `pdl` dialect.
- Added support for the `scf` dialect.
- Added support for the `shape` dialect.
- Added support for the `sparse_tensor` dialect.
- Added support for the `tensor` dialect.
- Added support for the `transform` dialect.
- Added support for the `ub` dialect.
- Added support for the `vector` dialect.
- Added support for the Triton `tt` dialect.
- Added support for the Mosaic GPU dialect.
- Added support for the Mosaic TPU dialect.
- Added support for the `erf` operation from the `chlo` dialect.
- Added support for the `real_dynamic_slice` operation from the `stablehlo` dialect.
- Added StableHLO mesh and sub-axis attribute wrappers.
- Added signless and unsigned integer `Context` constructors for widths `1`, `2`, `4`, `8`, `16`, `32`, `64`, and
  `128`, plus `bool_type` as an alias for `i1_type`.
- Added the builtin `TokenTypeRef` and `Context::token_type` for the new builtin `token` type.
- Added a wrapper for the new Triton `tt.atomic_poll` operation.
- Added support for the new optional `result_tilings` attribute of the StableHLO `custom_call` operation.
- Added the StableHLO `collective_reduce` operation and CHLO `top_k` (including `is_stable`) and `mulhi` operations.
- Added the Shardy `ReductionOperation` enum along with the new reduction operation accessors and constructor
  parameters for tensor sharding attributes and the `all_reduce` and `reduce_scatter` operations.

### Changed

- Changed all `OperationBuilder::add_*` functions to return `Result<Self, Error>` and report context mismatches
  immediately. Callers must now handle each addition's result (for example, using `?` in builder chains). A failed
  addition consumes the builder and destroys its owned regions, including any rejected regions passed to that addition.
- Updated the StableHLO `composite` operation to support regions.
- Updated StableHLO `collective_broadcast` to accept multiple data tensors and optional dynamic roots. Both new
  collective root APIs exclude the root-index tensor from their inferred results.
- Minor performance optimizations for some of the `shardy` dialect builders.
- Upgraded to the LLVM, StableHLO (v1.20.0), Shardy, and Triton revisions pinned by OpenXLA commit
  `eb6b90ed013f511eca088c52f541f3c0819f919e`.

### Removed

- Removed the LLVM dialect `TokenTypeRef` and `Context::llvm_token_type` following the upstream MLIR replacement of
  `!llvm.token` with the builtin `token` type, which is now wrapped by the builtin `TokenTypeRef` and
  `Context::token_type`. Note that builtin token values cannot cross function boundaries (producing operations must
  have the MLIR `TokenProducerTrait` and consuming operations must have the `TokenConsumerTrait`).
- Removed the EmitC `apply` operation wrapper following the upstream removal of the deprecated `emitc.apply`
  operation, whose functionality is covered by the existing `address_of` and `dereference` operation wrappers.

## [0.0.2] - 2026-03-02

## Changed

- Upgraded the XLA dependency to a newer version.

## [0.0.1] - 2026-02-22

### Added

- Initial release.

<!-- next-url -->
[0.0.1]: https://github.com/eaplatanios/ryft/compare/v0.0.1...HEAD
