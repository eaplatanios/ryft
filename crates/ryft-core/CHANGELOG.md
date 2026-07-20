# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/).

<!-- next-header -->
## [Unreleased] - Release Date

### Added

- Added `Parameterized::try_map_parameters`, `Parameterized::try_map_named_parameters`, and
  `Parameterized::broadcast_to_parameter_structure`.
- Added an internal program representation (i.e., an intermediate representation or IR) in `ryft_core::programs`. This
  includes a type system in `ryft_core::programs::types`, that supports modeling data types, array types, layouts,
  memory spaces, and sharding information (along with a new `ryft_core::sharding` module), and an effect system in
  `ryft_core::programs::effects`.
- Added core abstractions for performing program tracing and transforming program traces in `ryft_core::contexts` and
  `ryft_core::tracing`.
- Added program interpretation (i.e., execution or evaluation) machinery in `ryft_core::interpretation`.
- Introduced more fine-grained error types like `ParameterError`, `DataTypeError`, `LayoutError`, `BroadcastingError`,
  and `ShardingError`.
- Added support for the `DataType::F6E3M2FN` and `DataType::F6E2M3FN` 6-bit microscaling floating-point data types.

## [0.0.2] - 2026-03-02

### Changed

- Significantly enhanced the `Parameterized` trait.

## [0.0.1] - 2026-02-22

### Added

- Initial release.

<!-- next-url -->
[0.0.1]: https://github.com/eaplatanios/ryft/compare/v0.0.1...HEAD
