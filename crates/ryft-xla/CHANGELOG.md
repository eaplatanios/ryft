# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/).

<!-- next-header -->
## [Unreleased] - Release Date

### Added

- Initial release.

### Changed

- Updated the Shardy tensor sharding lowering for the new reduction operation parameter introduced by the OpenXLA
  upgrade, emitting sum reductions for unreduced axes.
