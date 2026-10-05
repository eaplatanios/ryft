# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/).

<!-- next-header -->
## [Unreleased] - Release Date

### Added

- Added `#[derive(Operation)]` for operation enums. Its enum-level attributes are `crate = "..."`, `type(T)` and
  `constant(V)` (which declare a composite family's operation and stored constant types), `members(...)`, and
  `dispatch(...)`, which selects the payload-delegating `identity` implementations together with the `discharge`,
  `batching`, `differentiation`, and `transposition` dispatchers. Structural projected member variants (i.e.,
  `#[ryft(projected(U, structural))]`) are batched by binding their payload once under the policy's
  `ReplicatedBatchingPolicyProjection`, so their member operation families need no batching rules of their own. The
  earlier `#[ryft(identity)]`, `#[ryft(type = T)]`, and `#[ryft(constant = V)]` spellings are rejected with diagnostics
  that name their replacements.

### Fixed

- Fixed `#[derive(Parameterized)]` for types containing optional parameter fields alongside other parameterized fields.

## [0.0.2] - 2026-03-02

### Changed

- Updated the `#[derive(Parameterized)]` macro to be compatible with the updated trait in `ryft-core`.

## [0.0.1] - 2026-02-22

### Added

- Initial release.

<!-- next-url -->
[0.0.1]: https://github.com/eaplatanios/ryft/compare/v0.0.1...HEAD
