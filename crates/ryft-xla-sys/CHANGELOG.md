# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/).

<!-- next-header -->
## [Unreleased] - Release Date

### Added

- Added native bindings for converting XProf `XSpace` traces into XLA feedback-directed optimization profiles and
  aggregating multiple instruction profiles at a configurable percentile.
- Added support for Linux AArch64.
- Added C++ bindings for the `affine`, `arith`, `bufferization`, `builtin`, `complex`, `gpu`, `llvm`, `mosaic_gpu`,
  `mosaic_tpu`, `nvgpu`, `shape`, `sparse_tensor`, `transform`, Triton `tt`, and `ub` MLIR dialects.
- Added the `mps` feature for loading the `jax-mps` PJRT plugin on macOS AArch64.
- Added Mosaic GPU type ID and serde-pass bindings, versioned bytecode/resource constants, CUDA-only `mosaic_gpu_v2`
  runtime registration, upstream Complex attribute bindings, and a source-owned UB poison attribute C API bridge.

### Changed

- Upgraded the OpenXLA dependency pin to commit `f16a4aeb435b2896ab96b605f004f982f6c97eb8`, which also upgraded the
  LLVM, StableHLO (v1.18.0), Shardy, and Triton pins.
- Replaced the LLVM dialect token type C++ bindings (`mlirTypeIsALlvmTokenType` and `mlirLlvmTokenTypeGet`) with
  builtin dialect token type bindings (`mlirTypeIsAToken` and `mlirTokenTypeGet`) following the upstream MLIR
  replacement of `!llvm.token` with the builtin `token` type.
- Updated the `sdyTensorShardingAttrGet` binding for the new trailing reduction operation argument and added the
  `sdyTensorShardingAttrGetReductionOp` binding.
- Synchronized mirrored XLA protobuf definitions with the new OpenXLA pin: added the `F6E3M2FN` and `F6E2M3FN` buffer
  types, the mesh iota transform, the named sharding reduction operation, new GPU topology and GPU device information
  fields, new `DebugOptions` fields and support types, and the `OpMetadata` payload message, and removed fields whose
  tags are now reserved upstream.
- Updated CUDA 12 and ROCm 7 Bazel build configuration for the new OpenXLA toolchain dependencies, including bumping
  `rules_ml_toolchain` and the hermetic ROCm distribution to `rocm_7.13.0_gfx908`.
- Updated the PJRT TPU plugin to `libtpu` version `0.0.41`.
- Updated the PJRT Neuron plugin to `libneuronxla` version `3.0.2891.0+e2a4b1f5`.
- Synchronized mirrored XLA protobuf definitions for command buffer command types, autotune backends, debug options,
  and GPU deviceless CUB mode.
- Synchronized StableHLO C API bindings with upstream mesh and sub-axis attributes.
- Pinned macOS Bazel artifacts to a macOS `11.0` deployment target so the published static library remains linkable
  from Rust consumers that target the workspace baseline.
- Compiled the native XLA/MLIR archive with hidden C++ symbol visibility while explicitly exporting source-owned C API
  entry points, avoiding symbol interposition with PJRT plugins that bundle their own XLA/MLIR copy.

## [0.0.2] - 2026-03-02

### Changed

- Upgraded the OpenXLA dependency pin to commit `15bc20b490170c25a4f4669d10573c6a601c0077`.
- Updated bindgen input headers to include `pjrt_c_api_abi_version_extension.h`.
- Synchronized `DebugOptions` protobuf definitions with upstream `xla.proto` additions at tags `455`, `456`, and `457`.

## [0.0.1] - 2026-02-22

### Added

- Initial release.

<!-- next-url -->
[0.0.1]: https://github.com/eaplatanios/ryft/compare/v0.0.1...HEAD
