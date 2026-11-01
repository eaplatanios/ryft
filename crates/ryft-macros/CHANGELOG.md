# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/).

<!-- next-header -->
## [Unreleased] - Release Date

### Added

- Added the `#[derive(Operation)]` macro, which implements `Operation` for enums whose variants each wrap one operation
  payload, so that the enum behaves like the payload of its active variant. Along with `Operation`, it generates the
  interpretation, partial evaluation, and projection implementations and the payload conversions (i.e., owned
  `From<Payload>` and borrowed `TryFrom<&Enum>`). `#[ryft(dispatch(...))]` additionally opts into identity
  implementations that delegate to the payload (i.e., `PartialEq`, `Eq`, and `Hash`) and into reference discharge,
  batching, differentiation, and transposition dispatchers. `#[ryft(projected(...))]` and `#[ryft(mixed(...))]`
  declare how each variant relates to the member universes of the operation family.
- Added the `#[capability]` attribute macro, which validates the conventions of capability traits and, for each
  `projection(Family => Member)` argument, implements the capability for values of the `Family` universe by
  projecting them onto their `Member` values.
- Added the `#[mermaid]` attribute macro, which renders the [Mermaid](https://mermaid.js.org/) diagrams written in
  `mermaid` code fences in rustdoc comments, choosing a light or dark diagram theme to match the rustdoc theme. It is
  meant to be applied as `#[cfg_attr(doc, mermaid)]`. It loads a vendored Mermaid bundle that it writes into the
  rustdoc output directory and falls back to a CDN when that bundle is unavailable.

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
