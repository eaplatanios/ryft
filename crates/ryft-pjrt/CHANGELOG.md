# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/).

<!-- next-header -->
## [Unreleased] - Release Date

### Added

- Added support for the new `PJRT_Buffer_Bitcast` C API function.
- Added support for the new `PJRT_Device_ClearMemoryStats` C API function.
- Added support for the new `PJRT_Error_ForEachPayload` C API function and for providing payload-aware safe Rust
  wrappers for error buffers and execution poisoning.
- Added support for the new `PJRT_TopologyDescription_Fingerprint` C API function.
- Added support for the new `PJRT_Executable_ParameterMemoryKinds` C API function.
- Added support for the new `PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace` C API function.
- Added support for the new `PJRT_TopologyDescription_GetMemorySpaceKindIds` C API function.
- Added support for the new `PJRT_LoadOptions`.
- Added support for the new `PJRT_HostMemoryAllocator` extension and its owned host-memory allocation wrapper.
- Added support for the new `PJRT_Xla_Transform` extension through a safe `XlaTransform` trait API.
- Added the `mps` feature and `load_mps_plugin()` for loading the `jax-mps` PJRT plugin.
- Added `BufferType::element_size_in_bytes`.

### Changed

- Updated our PJRT C API bindings for version `0.111`.
- Changed `Buffer::copy_to_host` to fall back to the buffer's reported on-device byte size when a PJRT plugin
  returns a successful host-copy size query without populating `dst_size`.
- Changed `BufferSpecification` to carry a concrete `Layout`, materializing dense defaults during construction and
  parsing before values cross layout-sensitive PJRT C API calls.
- Expanded executable compiled-memory statistics support to include total allocator bytes, indefinite allocations,
  and peak unpadded heap bytes.
- Changed `TiledLayout::minor_to_major` to `Vec<u64>` from `Vec<i64>`.
- Changed `ExecutionInput::buffer` to an `Arc<Buffer<'o>>` instead of a `Buffer<'o>`.
- Changed `Memory` equality to fall back to memory-kind strings when a PJRT plugin does not implement memory kind IDs.

## [0.0.2] - 2026-03-02

### Added

- Added support for `BufferType::S1` and `BufferType::U1`.
- Added support for the new `PJRT_Device_GetAttributes` C API function.
- Added support for the new `PJRT_Client_Load` C API function.
- Added support for the new `PJRT_LoadedExecutable_AddressableDeviceLogicalIds` C API function.

### Changed

- Updated our PJRT C API bindings for version `0.97`.
- Updated the layouts extension bindings to version `4` and added support for executable parameter layout queries.
- Updated the FFI extension bindings to version `3` and added support for setting and getting the execution context
  for specific execution stages.

## [0.0.1] - 2026-02-22

### Added

- Initial release.

<!-- next-url -->
[0.0.1]: https://github.com/eaplatanios/ryft/compare/v0.0.1...HEAD
