# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/).

<!-- next-header -->
## [Unreleased] - Release Date

### Added

- Initial release.
- Added StableHLO lowerings (including the shard-map path) and eager-execution support for the new `ryft-core`
  elementwise operations: `tanh`, `logistic`, `rsqrt`, `pow`, `sign`, `floor`, `ceil`, `round_nearest_even`,
  `maximum`, `minimum`, and `remainder`.
- Added the traced `custom_call` lowering to typed-FFI `stablehlo.custom_call` operations (`api_version = 4`), with
  the operation's typed attributes carried as the `backend_config` dictionary (strings, `i1` Booleans, signless
  `i64` integers, and `f64` floats), including inside `shard_map` manual regions. Handlers registered through
  `ryft-pjrt`'s `Client::register_ffi_handler` execute through both the eager capability path and compiled
  programs.
- Added the traced `sort` lowering to stable `stablehlo.sort` operations with a synthesized comparator region
  (`TOTALORDER` float comparisons, `SIGNED`/`UNSIGNED` integer comparisons), and the traced `rng_bit_generator`
  lowering to `stablehlo.rng_bit_generator`, both wired through the plain, array-operation, and shard-map
  dispatchers. Cross-backend tests verify stable-tie ranking, signed-zero and NaN total-order placement, and
  bit-exact ThreeFry agreement between the reference backend and XLA's expansion.
- Added lowerings for the shape-changing named-axis collectives inside `shard_map` manual regions: channeled
  `stablehlo.all_gather`, `stablehlo.reduce_scatter` (synthesized sum region), `stablehlo.collective_permute`
  (axis-local pairs expanded to global device pairs per replica group), and `stablehlo.all_to_all`, sharing the
  mesh-axis replica-group computation with the reduction collectives. Two-device CPU execution tests cover all
  four, including gradients through `shard_map` bodies transposing to the dual collectives.
- Accumulation-typed dots lower to `stablehlo.dot_general` with the accumulation type as the explicit result type
  (XLA's `preferred_element_type`), verified by a module fixture and an `f8e4m3fn → f32` CPU parity test against
  the reference backend.
