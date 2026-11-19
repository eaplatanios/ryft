# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/).

<!-- next-header -->
## [Unreleased] - Release Date

### Added

- Added CUDA 12, CUDA 13, and ROCm 10 static archives to the native release workflow. Triton builds select the
  corresponding GPU archive when a GPU feature is enabled.
- Added native bindings for converting XProf `XSpace` traces into XLA feedback-directed optimization profiles and
  aggregating multiple instruction profiles at a configurable percentile.
- Added support for Linux AArch64.
- Added C++ bindings for the `affine`, `arith`, `bufferization`, `builtin`, `complex`, `gpu`, `llvm`, `mosaic_gpu`,
  `mosaic_tpu`, `nvgpu`, `shape`, `sparse_tensor`, `transform`, Triton `tt` and `ttg`, and `ub` MLIR dialects.
- Added the `mps` feature for loading the `jax-mps` PJRT plugin (version `0.10.10`) on macOS AArch64.
- Added the `cuda` module for launching precompiled CUDA kernels through `ryft-cuda` using PJRT clients and XLA FFI
  buffers and streams. Includes client CUDA version queries and launcher construction, plus unsafe buffer and stream
  adapters, behind the `cuda-12` and `cuda-13` features.
- Added Mosaic GPU type ID and serde-pass bindings, versioned bytecode/resource constants, CUDA-only `mosaic_gpu_v2`
  runtime registration, upstream Complex attribute bindings, and a source-owned UB poison attribute C API bridge.
- Added a native C ABI and Rust bindings for direct verified Triton module compilation in GPU-configured archives,
  with typed resource metadata, owned outputs, and explicit compiler availability on CPU archives.

### Changed

- Upgraded the OpenXLA dependency pin to commit `aaed3e0327d520ee375451ca42dd239ee40a50e6` and the JAX dependency pin
  to commit `9564ca1d3df0d312cad4346458b8d41ce5204ff7`, together with their LLVM, StableHLO, Shardy, and Triton pins.
  Native builds use Bazel 8.8.2, CCCL 3.4.0, and XLA's pinned ROCm 10.1 development distribution. The pinned Triton
  sources require C++20, so Linux builds use the glibc 2.31 sysroot with GCC 10's standard library.
- Used LLVM's `bit_cast` in the Triton NVIDIA lowering to support the pinned Linux standard library.
- Initialized the OpenXLA GPU client's stream options explicitly to support the pinned Linux standard library.
- Limited CUDA Triton output to PTX 9.0 for the qualified CUDA 13.0 assembler.
- Updated the JAX Mosaic GPU serde pass to compile with the NVVM barrier's new builder signature.
- Updated the ROCm plugin feature, artifact names, environment variables and path accessor to ROCm 10.
- Removed the obsolete `enable_libnvjitlink_support` Bazel flag following XLA's unconditional nvJitLink support.
- Removed the CUDA 12 `sm_50` target because the pinned Eigen headers require compute capability 6.0 or newer.
- Used the shared Bash copy recipe for LLVM operation shards on Windows and preserved copied C++ source bytes.
- Retained absolute junctions in the Windows LLVM source overlay by opting it out of Bazel's reproducible repository
  contents caching, whose relative link rewriting made sources inaccessible through the execroot junction.
- Avoided Windows SDK floating-point classification overloads in XNNPACK integer min/max reductions while preserving
  floating-point NaN handling.
- Preserved the target triple in XLA's embedded constant objects so macOS archives include platform version
  commands and link without missing-platform warnings.
- Regenerated native bindings for XLA FFI custom options, split GPU pass registrations and the LLVM C API changes,
  including the additional DI string type character-type argument and removal of obsolete MHLO domain attributes.
- Synchronized XLA Protobuf fields, enums and optional-presence semantics, including BLASLt algorithm configurations,
  target-machine options, GPU device metadata, memory statistics, compiler variants and profiler multipass settings.
- Replaced the LLVM dialect token type C++ bindings (`mlirTypeIsALlvmTokenType` and `mlirLlvmTokenTypeGet`) with
  builtin dialect token type bindings (`mlirTypeIsAToken` and `mlirTokenTypeGet`) following the upstream MLIR
  replacement of `!llvm.token` with the builtin `token` type.
- Updated the `sdyTensorShardingAttrGet` binding for the new trailing reduction operation argument and added the
  `sdyTensorShardingAttrGetReductionOp` binding.
- Synchronized mirrored XLA protobuf definitions with the new OpenXLA pin: added the `F6E3M2FN` and `F6E2M3FN` buffer
  types, the mesh iota transform, the named sharding reduction operation, new GPU topology and GPU device information
  fields, new `DebugOptions` fields and support types, and the `OpMetadata` payload message, and removed fields whose
  tags are now reserved upstream.
- Updated CUDA 12 and ROCm 10 Bazel build configuration for the new OpenXLA toolchain dependencies, including the
  inherited `rules_ml_toolchain` revision and hermetic ROCm distribution.
- Updated the PJRT TPU plugin to `libtpu` version `0.0.47`.
- Updated the PJRT Neuron plugin to `libneuronxla` version `3.0.5356.0+c743c3ec`.
- Synchronized mirrored XLA protobuf definitions for command buffer command types, autotune backends, debug options,
  and GPU deviceless CUB mode.
- Synchronized StableHLO C API bindings with upstream mesh and sub-axis attributes.
- Pinned macOS Bazel artifacts to a macOS `11.0` deployment target so the published static library remains linkable
  from Rust consumers that target the workspace baseline.
- Compiled the native XLA/MLIR archive with hidden C++ symbol visibility while explicitly exporting source-owned C API
  entry points, avoiding symbol interposition with PJRT plugins that bundle their own XLA/MLIR copy.
- Generated the native compiler's XLA, JAX, and Triton version metadata from the `WORKSPACE` pins, with Bazel
  verifying the Triton revision against OpenXLA.

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
