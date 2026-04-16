# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/).

<!-- next-header -->
## [Unreleased] - Release Date

### Added

- Added support for Linux AArch64.

### Changed

- Upgraded the OpenXLA dependency pin to commit `7b1be14958aac5c83f1b9f7bcdfc51fdbd29acba`.
- Synchronized the mirrored protobuf definitions with the upstream PJRT and StreamExecutor schema changes.
- Pinned macOS Bazel artifacts to a macOS `11.0` deployment target so the published static library remains linkable
  from Rust consumers that target the workspace baseline.

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
