# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/).

<!-- next-header -->
## [Unreleased] - Release Date

### Added

- Initial release.

### Changed

- `XlaOptions::donation_flags` is now `Option<Vec<bool>>`; lowering validates a present vector against the flat
  public input arity and materializes full-length flags in the lowered artifact.
- `XlaDomain::client` and `XlaDomain::mesh` now return `Result` (`Error::MissingClient` / `Error::MissingMesh`)
  instead of panicking on clientless or meshless domains.
- `StagedXlaFunction::call` and `CompiledXlaFunction::call` now return `Result` instead of panicking when capture
  registration or structured output reassembly fails, and the `gradient`/`jvp`/`batch` transforms propagate staging
  errors (`XlaDomainError` gained a `Batching` variant).
- `CompiledXlaFunction::staged` returns the staged handle by value; the duplicated retained copy was removed.
- Compilation APIs take `XlaOptions` directly following the removal of the core `CompilationOptions` wrapper, and
  `XlaDomain::compile` is rebased on the shared `CompilationContext::compile_request` helper.
