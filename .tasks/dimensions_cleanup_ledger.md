# First-class dimension cleanup delivery ledger

This ledger records the recoverable delivery state for
`.tasks/plan_symbolic_dimensions_architecture_cleanup.md`. Repository and remote state are authoritative when this
ledger disagrees with Git.

## Bootstrap record

- Integration branch: `u/eaplatanios/dynamic-shapes`
- Integration baseline: `8105cfd26817ab728bb2799c889021f240345993`
- Immutable archive branch: `u/eaplatanios/archive/dimensions-wip-2026-07-24`
- Immutable archive commit: `770e77d001547c72150a44843c170ea6417ab41e`
- Immutable archive tree: `4edb3eb201ab45e474c03614a33d580dab70bf67`
- Mutable remainder branch: `u/eaplatanios/wip/dimensions-remainder`
- Mutable remainder initial commit: `770e77d001547c72150a44843c170ea6417ab41e`
- Expanded pre-snapshot status: 112 tracked changes plus 29 nonignored untracked paths, for 141 entries
- Snapshot manifest: 142 paths, including the ignored cleanup plan
- Snapshot manifest SHA-256: `428782ca2768c9dfdb2f8260a2e806f8c5af0ac91ed5503df89037662dfab206`
- Archive verification: the pre-snapshot and committed manifests matched byte-for-byte
- Archived baseline build:
  `CARGO_TARGET_DIR=/Users/eaplatanios/Development/Repositories/ryft-1-dimensions-target cargo check -p ryft-core`
  completed successfully

## S0: bootstrap plan and delivery controls

- Status: landed
- Branch: `u/eaplatanios/increment/s0-bootstrap`
- Source commit: `179483f2da4770fa675021334914baba6d801186`
- Integration commit: `fbf43052ec04ea2822f3b3753883b68b0bb42c7e`
- Remainder reconciliation commit: `979d9dd171ffab30944e80d4ab666614d96cf834`
- Immutable archive unchanged: yes
- Landed: the approved cleanup plan, this delivery ledger, and only the narrow staging-worktree `git restore`
  exception in `AGENTS.md`
- Deferred: all code extraction and semantic work begins with `S1`
- Verification: archive manifest equality and the archived `ryft-core` build passed; S0 formatting and staged-diff
  checks passed before handoff
- Residual search: branch, PR, `S6`, phase-ordering, and restore-policy searches passed before handoff
- Next action: none

## S1: region arena ID selection

- Status: landed
- Branch: `u/eaplatanios/increment/s1-region-reroot`
- Source commit: `389f1eb56c226e2779c993ae26267bed555061ab`
- Integration commit: `048150c170ed19d3138a2d3b5c7eb8bc4d3706ee`
- Remainder reconciliation commit: `5b4c473316beda0f356ade1ad8fff8d4ca99db75`
- Immutable archive unchanged: yes
- Landed: introduced public `RegionRef::with_id` and replaced reconstruction from an existing `RegionRef`'s arena
- Deferred: identity-signature retention remains in `P1`; `S1` provides the metadata-preserving seam without
  introducing identity machinery
- Verification: formatting passed; the focused `with_id` test passed; all 912 core library tests passed; core doctests
  passed 43 tests with 13 ignored; the compiler emitted only the `arrays` ambiguous-glob warning already present on
  the B1 integration baseline
- Residual search: the production change leaves only initial arena-entry `RegionRef::new` calls and its direct
  constructor error test
- Next action: none

## S3: former elementwise-macro restructure

- Status: absorbed
- Branch: none
- Source commit: none
- Integration commit: not applicable; the independent restructure is already ancestral to integration
- Remainder reconciliation commit: not applicable
- Immutable archive unchanged: yes
- Landed: the historical restructure is already present
- Deferred: explicit elementwise inference result types and `TypeError::Invalid` rewrites belong to S4; `Size` to
  `Dimension` rewrites belong to S5a
- Verification: classified every remaining archive diff hunk in `macros.rs` and `differentiation/elementwise.rs`
- Residual search: no standalone S3-owned hunk remains
- Next action: complete S4

## S4: structured type errors

- Status: ready for review
- Branch: `u/eaplatanios/increment/s4-type-error`
- Source commit: pending
- Integration commit: pending owner review
- Remainder reconciliation commit: pending integration
- Immutable archive unchanged: yes
- Landed: replaces the single-field `TypeError` struct with `Invalid(String)`, structured `WrongKind`, and typed
  `Custom` variants; adds typed custom recovery and program-error preservation; migrates all 759 existing construction
  and destructuring sites; and adds the elementwise inference annotations formerly assigned to S3
- Deferred: `DimensionError` remains owned by the later dimension implementation and will travel through `Custom`;
  all dimension representation and identity semantics remain in S5a and P1; the inherited `arrays` root-glob warning
  remains assigned to P9
- Verification: `cargo fmt --all -- --check` and `git diff --check` passed; `cargo check -p ryft-core` and
  `cargo check -p ryft-xla` passed;
  `cargo test -p ryft-core --lib` passed all 914 tests; `cargo test -p ryft-core --doc` passed 43 tests with 13
  ignored; `cargo test -p ryft-macros -p ryft-macros-tests` passed all 53 macro unit tests and all 17 operation
  integration tests, while one parameter compile-fail snapshot retained its independently reproduced integration
  baseline mismatch because the compiler now lists `Axes`; `cargo test -p ryft-xla --lib` passed 395 tests with 1
  ignored
- Residual search: `rg -n 'TypeError\s*\{|TypeError::Invalid\s*\{' --glob '*.rs' .` leaves only the `TypeError`
  enum/impl declarations and unrelated type names; no old struct construction or named `Invalid` variant remains
- Next action: push this increment and stage its no-commit merge on `u/eaplatanios/dynamic-shapes` for owner review

## B0: repair the custom-derivatives module move

- Status: landed
- Branch: `u/eaplatanios/increment/b0-custom-derivatives-baseline`
- Source commit: `656a49d417ae2f3e469ac9846e90aa8beeaf580b`
- Integration commit: `957ca8a66f1e6519773fffe9cf1e35c0ef3b0afe`
- Remainder reconciliation commit: `5a46403b48d78d27025ccedc07f2216dd840e05b`
- Immutable archive unchanged: yes
- Landed: reconstructed the moved module from its last compiling pre-move implementation and updated every
  production path to `tracing_v2::custom_derivatives`
- Deferred: symbolic identity/rebinding changes accidentally mixed into the original move remain assigned to the
  dimension phases that own them
- Verification: `cargo check -p ryft-core` and `cargo check -p ryft-xla` passed;
  `cargo test -p ryft-core --lib` passed all 911 tests; `cargo test -p ryft-core --doc` passed 43 tests with 13
  ignored; `cargo test -p ryft-xla --lib` passed 395 tests with 1 ignored
- Residual search: no `tracing_v2::operations` reference remains in Rust source under `crates/`
- Next action: none

## B1: rename the public array/data type modules

- Status: landed
- Branch: `u/eaplatanios/increment/b1-type-module-renames`
- Source commit: `2f613bdb32e52becb55b248c3611b00f35e09f66`
- Integration commit: `55c6c45387bfa759d83c1d383e50856080696faf`
- Remainder reconciliation commit: `ed5cc39a8da437dcb73b5630ed9326d15ddb1106`
- Immutable archive unchanged: yes
- Landed: renamed `types::array_types` to `types::arrays` and `types::data_types` to `types::data`
- Deferred: all type semantics, item names, and the later `Size`/`Dimension` restructuring remain unchanged
- Verification: formatting and core/XLA checks passed; all 911 core library tests passed; core doctests passed 43
  tests with 13 ignored; XLA library tests passed 395 tests with 1 ignored; owner integration omitted the source
  branch's root-facade disambiguation, so the landed tree retains the `arrays` ambiguous-glob warning
- Residual search: no old module path or filename remains; the five `data_types` matches are ordinary local
  variable/parameter names and are intentionally retained
- Next action: none
