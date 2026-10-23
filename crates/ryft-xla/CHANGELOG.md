# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/).

<!-- next-header -->
## [Unreleased] - Release Date

### Added

- Initial release.

### Removed

- Batched XLA programs keep rectangular arrays, as before `bcea9644e`: mapped data cannot produce dimensions or
  per-item extents, and core rejects such computations before lowering. Variable-length workloads use shared bounded
  dynamic dimensions or the explicit dense-buffer kernels `ragged_dot` and `parallel_ragged_all_to_all`.
