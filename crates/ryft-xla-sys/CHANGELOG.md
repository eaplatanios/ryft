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
- Added C++ bindings for the `affine`, `arith`, `gpu`, `llvm`, `mosaic_gpu`, `mosaic_tpu`, `nvgpu`, `shape`,
  `sparse_tensor`, `transform`, and Triton `tt` MLIR dialects.
- Added the `mps` feature for loading the `jax-mps` PJRT plugin on macOS AArch64.

### Changed

- Upgraded the OpenXLA dependency pin to commit `1c884c1b85f81728c6391ccb961a1c25d12cbe71`.
- Updated CUDA 12 and ROCm 7 Bazel build configuration for the new OpenXLA toolchain dependencies.
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
