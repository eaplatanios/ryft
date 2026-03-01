# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/).

<!-- next-header -->
## [Unreleased] - Release Date

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
