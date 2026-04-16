# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/).

<!-- next-header -->
## [Unreleased] - Release Date

### Added

- Added support for the new `PJRT_Buffer_Bitcast` C API function.
- Added support for the new `PJRT_Error_ForEachPayload` C API function and for providing payload-aware safe Rust
  wrappers for error buffers and execution poisoning.
- Added support for querying executable parameter memory kinds and topology fingerprints.
- Added support for the new `PJRT_HostMemoryAllocator` extension and its owned host-memory allocation wrapper.

### Changed

- Updated our PJRT C API bindings for version `0.104`.
- Expanded executable compiled-memory statistics support to include total allocator bytes, indefinite allocations,
  and peak unpadded heap bytes.
- Changed `TiledLayout::minor_to_major` to `Vec<u64>` from `Vec<i64>`.
- Changed `ExecutionInput::buffer` to an `Arc<Buffer<'o>>` instead of a `Buffer<'o>`.

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
