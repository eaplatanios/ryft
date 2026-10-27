# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/).

<!-- next-header -->
## [Unreleased] - Release Date

### Added

- Initial release of the portable kernel subsystem (authoring `#[kernel]`, verification, interpretation,
  serialization, transforms, scheduling, and the `KernelCompiler` contract), previously developed unreleased under
  `ryft_core::kernels`; the canonical path is `ryft_kernels::...` with `ryft::kernels` as the facade.
