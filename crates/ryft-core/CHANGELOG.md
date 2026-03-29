# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/).

<!-- next-header -->
## [Unreleased] - Release Date

### Added

- Added `Parameterized::broadcast_to_parameter_structure`.
- Added the `Type` and `Broadcastable` traits.
- Added the `DataType`, `Size`, `Shape`, `TileDimension`, `Tile`, `TiledLayout`, `StridedLayout`, `Layout`, `ArrayType`,
  `MeshAxisType`, `MeshAxis`, `LogicalMesh`, `MeshDeviceId`, `MeshProcessIndex`, `MeshDevice`, `DeviceMesh`,
  `ShardingDimension`, and `Sharding` types.
- Introduced more fine-grained error types like `ParameterError`, `DataTypeError`, `LayoutError`, `BroadcastingError`,
  and `ShardingError`.

## [0.0.2] - 2026-03-02

### Changed

- Significantly enhanced the `Parameterized` trait.

## [0.0.1] - 2026-02-22

### Added

- Initial release.

<!-- next-url -->
[0.0.1]: https://github.com/eaplatanios/ryft/compare/v0.0.1...HEAD
