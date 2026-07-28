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

- Status: landed
- Branch: `u/eaplatanios/increment/s4-type-error`
- Source commit: `775a96c976c9f7f0957394eca41a71a5e846ed5d`
- Integration commit: `8be8783b11a55c7239e1af4162449a7481f9fb31`
- Remainder reconciliation commit: `b25a257f85bf93af5ea73642624e5a6910d285ba`
- Immutable archive unchanged: yes
- Landed: replaces the single-field `TypeError` struct with named `Invalid { message: String }` and typed `Custom`
  variants; routes invalid-error construction through `TypeError::invalid(...)`; adds typed custom recovery; migrates
  all 759 existing construction and destructuring sites; and adds the elementwise inference annotations formerly
  assigned to S3
- Deferred: `DimensionError` remains owned by the later dimension implementation and will travel through `Custom`;
  archived `TypeError::from_program` call sites must be corrected by returning owner-specific errors rather than
  adding a reverse umbrella conversion; all dimension representation and identity semantics remain in S5a and P1; the
  inherited `arrays` root-glob warning remains assigned to P9
- Verification: `cargo fmt --all -- --check` and `git diff --check` passed; `cargo check -p ryft-core` and
  `cargo check -p ryft-xla` passed;
  `cargo test -p ryft-core --lib` passed all 913 tests; `cargo test -p ryft-core --doc` passed 43 tests with 13
  ignored; `cargo test -p ryft-macros -p ryft-macros-tests` passed all 53 macro unit tests and all 17 operation
  integration tests, while one parameter compile-fail snapshot retained its independently reproduced integration
  baseline mismatch because the compiler now lists `Axes`; `cargo test -p ryft-xla --lib` passed 395 tests with 1
  ignored
- Residual search: `rg -n 'TypeError::Invalid\(' --glob '*.rs' crates` and
  `rg -n 'TypeError::Invalid\s*\{\s*message\s*:' --glob '*.rs' crates` are empty; the remaining
  `TypeError::Invalid { message }` matches are intentional destructuring patterns
- Next action: none

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

## S5a: rename dimensions and move shape types

- Status: landed
- Branch: `u/eaplatanios/increment/s5a-dimension-rename`
- Source commit: `3676d051cc1ae8a43bc19de09310386bb0d90455`
- Integration commit: `0367c1464f9c1dcbaaab9d14622adcab779b0b93`
- Remainder reconciliation commit: `3445c89ced98069ad8728082102a3fa10cbf131f`
- Immutable archive unchanged: yes
- Landed: renames the public `Size` type to `Dimension`; moves `Dimension`, `Shape`, and `StaticShape` from
  `types::arrays` to the new canonical `types::dimensions` module; updates all in-repo consumers directly without a
  compatibility alias or re-export; and moves the 14 representation tests with their owning types
- Deferred: identity-bearing dimensions, authoritative bounds, refinements, and every other semantic representation
  change remain assigned to P1; S5a preserves `Dynamic(Option<usize>)` exactly
- Verification: formatting and diff checks passed; core, XLA, and facade checks passed; all 913 core library tests
  passed; all 14 focused dimension tests passed; core doctests passed 43 tests with 13 ignored; XLA library tests
  passed 395 tests with 1 ignored; macro verification passed all 53 macro unit tests and all 17 operation integration
  tests, while the parameter compile-fail snapshot retained the independently reproduced S4 baseline mismatch caused
  by rustc listing `Axes`
- Residual search: the S5a handoff reported only the unrelated `ryft_mlir::Size` import aliased as `MlirSize`, but the
  broad rename had incorrectly removed three legitimate `ryft_mlir::Size` uses from the StableHLO example; S5b
  corrects that omission; no old public core `Size` declaration, variant use, test name, or stale `types::arrays` path
  for `Dimension`, `Shape`, or `StaticShape` remains
- Next action: correct the unrelated MLIR `Size` use discovered by the remainder reconciliation

## S5b: restore the MLIR size in the StableHLO example

- Status: landed
- Branch: `u/eaplatanios/increment/s5b-mlir-size-example`
- Source commit: `1c92e69d868706d9efc41e5174d07697bc605306`
- Integration commit: `20eefa3085e70a44995862ff0fc9986f80158c0d`
- Remainder reconciliation commit: `9bcc73d7093a1a001c8c0539e5307851558ad9cd`
- Immutable archive unchanged: yes
- Landed: restores the three low-level StableHLO tensor-shape expressions to `ryft_mlir::Size`, leaving the
  `ryft_core::types::Dimension` rename unchanged
- Deferred: P0 and all dimension semantics remain unchanged
- Verification: the integration baseline's
  `cargo check -p ryft --example stable_hlo_matmul` failed with six unresolved `Dimension` uses; after the correction,
  the same command passes; `cargo fmt -p ryft -- --check` and `git diff --check` pass
- Residual search: exact `Size` matches are the three restored example expressions plus the XLA lowering import
  aliased as `MlirSize`; all four are intentional `ryft_mlir::Size` uses
- Next action: none

## P0: behavioral and architectural evidence freeze

- Status: landed
- Branch: `u/eaplatanios/increment/p0-evidence-freeze`
- Source commit: `ba0e5f862ca083baf54ee58b8809560cdbe2de6c`
- Integration commit: `34b86d75663900bd3ac5446ac3a626ff50c953a5`, completed by the artifact-only correction
  `07f6b22dcb47ed52cc717632a9fab7c71b5b44e9`
- Remainder reconciliation commit: `71f613224`, completed by the artifact-only reconciliation `0484d4e8c`
- Immutable archive unchanged: yes
- Landed: freezes revisions, environment, source/generated size, compile/memory, graph, runtime-smoke, allocation,
  diagnostic, proof, operation-family, transform, reconstruction, collector, and projection-ownership evidence;
  classifies all 142 archive paths and every affected operation
- Deferred: all semantic implementation begins in P1; P0 includes only the one-line feature-gated S1 call-site
  correction required to compile the existing golden benchmark emitter
- Verification: archive `cargo test -p ryft-core --lib` passed 1,035 tests with one documented ignored batching gap;
  integration `cargo test -p ryft-core --lib` passed all 913 tests; integration
  `cargo check -p ryft-core --features benchmarking`, `cargo fmt --all -- --check`, and `git diff --check` passed
- Residual search: the archive-disposition path column mechanically matches the 142-path archive manifest exactly,
  with 142 unique rows and no missing or extra path; every dual contract, constructor overlap, context view,
  reconstruction path, runtime-dimension collector class, and transpose witness has a named destination
- Next action: none

## P1a: dimension identity foundations

- Status: landed
- Branch: `u/eaplatanios/increment/p1a-leaf-dimensions`
- Source commit: `daa288dbae6ff50065413393127db30f55d4f8cc`
- Integration commit: `7a2a0a39a96a3700c9855439faa8c2bfecece50c`
- Remainder reconciliation commit: `7952e1f4e`
- Immutable archive unchanged: yes
- Landed: introduces validated inclusive-lower/exclusive-upper `DimensionBounds`, fresh clone-preserving
  `DimensionVariable` identities with diagnostic-only names, one immutable bounds authority per identity, and typed
  `DimensionError::InvalidBounds` recovery through `TypeError::Custom`; `DimensionError` derives `thiserror::Error`
  rather than duplicating standard error boilerplate
- Deferred: changing `Dimension::Dynamic` to carry an identity belongs to P1b; generic `Type` identity/refinement
  hooks, structural closure, alpha-equivalent cache matching, and `OutputIdentityRole` deletion belong to P1c
- Verification: the 16 focused `types::dimensions` tests passed; `cargo check -p ryft-core` passed;
  `cargo test -p ryft-core --lib` passed all 915 tests; `cargo test -p ryft-core --doc` passed 43 tests with 13
  ignored; scoped formatting and `git diff --check` passed
- Residual search: existing `Dimension::Dynamic(Option<usize>)` construction remains intentionally unchanged for P1b;
  `DimensionVariable` production ownership is confined to `types::dimensions`; the only additional production path is
  one explicit `Ok::<ReshapeDimensionExpression, TypeError>` annotation required because the new typed error
  conversion made an existing inferred `Result<_, _>` ambiguous
- Next action: none

## P1b: dynamic dimension leaves

- Status: landed
- Branch: `u/eaplatanios/increment/p1b-dynamic-dimension-leaves`
- Source commit: `e82396bef14516b183d86535429961ee65989835`
- Integration commit: `42c2a3f3ca973ed6af8e6ab1085a3229acb13e63`
- Remainder reconciliation commit: `2156a584865c946cd088d56b00eb15efd0089fbb`
- Immutable archive unchanged: yes
- Landed: replaces `Dimension::Dynamic(Option<usize>)` with one `DimensionVariable` leaf and migrates shape/type
  consumers without a compatibility variant or expression representation; shared leaves retain equality through
  broadcasting, reshaping, transpose, reduction, dot, and repeated-axis refinement; XLA bounds lowering reads the
  variable's authoritative bounds directly; persistent XLA signatures use a version-3 typed variable table that
  preserves sharing while excluding diagnostic names from canonical cache keys
- Deferred: generic `Type` identity/refinement hooks, structural closure, alpha-equivalent cache matching, and
  `OutputIdentityRole` deletion remain assigned to P1c; derived dynamic reshape, concatenate, pad, and strided
  full-extent slice results reject with exact diagnostics until P3 supplies explicit result-dimension operands
- Verification: `cargo check -p ryft-core -p ryft-xla` passed; `cargo test -p ryft-core --lib` passed all 915 tests;
  `cargo test -p ryft-core --doc` passed 43 tests with 13 ignored; `cargo test -p ryft-xla --lib` passed 396 tests
  with one documented ignored benchmark; all 53 `ryft-macros` unit tests and all 17 operation macro-integration tests
  passed; the full macro-integration command retains one inherited trybuild help-list mismatch that reproduces
  unchanged on integration commit `7a2a0a39a96a3700c9855439faa8c2bfecece50c`; scoped formatting and
  `git diff --check` passed
- Residual search: no `Dimension::Dynamic(None)`/`Dimension::Dynamic(Some(...))`, old XLA version-2 type-schema
  identifiers, invalid zero-bound lowering variant, or non-static `Dimension` `.copied()` use remains under `crates`;
  the sole non-test/non-doc `DimensionVariable::new` outside `types::dimensions` recreates validated shared variables
  while decoding the version-3 XLA persistent signature
- Next action: none

## P1c: structural dimension identities

- Status: landed
- Branch: `u/eaplatanios/increment/p1c-structural-dimension-identities`
- Source commit: `d8292119cb5cfb2d58ea3292aec3328fd4bc3f78`
- Integration commit: `7bb5bf4aad7005b369a981cc3daacfe1369787f6`; owner review corrections continued on the integration
  branch through `7bdddddc33a9c0a377017bf894eb76096e1f7e1a`
- Remainder reconciliation commit: `d638ada5a`
- Immutable archive unchanged: yes
- Landed on increment: adds only the generic `Type::Identity`/`Type::Refinements` contracts required by program
  boundaries; derives boundary/internal ownership structurally from type positions and graph dataflow; validates shared
  dynamic extents transactionally across complete input/output signatures; and applies simultaneous alpha-renaming to
  atom types, constant/capture metadata, identity-bearing operation payloads, and attached regions
- Region composition: condition, while, scan, custom JVP/VJP, custom-VJP tangent, and rematerialization declare their
  ordinary operand-to-region input mapping; all owned, shared-callee, and replayed-region drivers use one generic
  instantiation/import path with cache sharing for disjoint alpha-equivalent identities and isolation for overlapping
  permutations
- Surface reduction: boundary-refinement helpers remain private interpretation functions rather than new `Value`
  methods; live identity closures and cache metadata remain crate-private; the unused generic canonical-signature
  representation and `TypeIdentity::CanonicalProperties` hook were deleted because exact-source core caches have no
  consumer for them and persistent XLA keys already own their canonical schema
- Transitional rule: before P2/P3 add first-class dimension results/operands, one fresh output-reference occurrence may
  establish an internal identity for a legacy shape-producing array operation; Phase 3 and the deletes ledger explicitly
  require removing this fallback once result-dimension operands are present
- Verification: `cargo check -p ryft-core -p ryft-xla` passed; `cargo test -p ryft-core --lib` passed all 922 tests;
  `cargo test -p ryft-core --doc` passed 43 tests with 13 ignored; all 53 `ryft-macros` unit tests and all 17 operation
  macro-integration tests passed; `cargo test -p ryft-xla --lib` passed 396 tests with one documented ignored benchmark;
  `cargo fmt --all` and `git diff --check` passed
- Residual search: no production `OutputIdentityRole`, `output_identity_role`, generic canonical-identity object, or
  `TypeIdentity::CanonicalProperties` remains; all identity-bearing operation payload fields found by the P0/P1c
  inventory either implement renaming or contain only derived transform-local bookkeeping; the temporary output-reference
  producer fallback is named in code, Phase 3, and the deletes ledger
- Known warning: the inherited ambiguous `arrays` glob re-export remains assigned to P9
- Remainder reconciliation: `d638ada5a` preserves the nonconflicting later archived edits as an audit diff while
  resolving superseded P1c foundation paths in favor of the reviewed integration. That mutable remainder is not
  expected to compile against the new contracts until its assigned later increments are reconstructed; the immutable
  archive remains the compiling historical reference.
- Next action: none

## P2a: dimension SSA foundations

- Status: landed
- Branch: `u/eaplatanios/increment/p2a-dimension-ssa-foundations`
- Source commit: `e73dde1d4cebf560d5abfead1ce44132e0bd3124`
- Integration commit: `1693157b44d3b44cdee962c8574dbedf3f80314e`, followed by owner review correction
  `bede731d2968a5deb3d5013c8bfb8d4ac62f9cce`
- Remainder reconciliation commit: `e3e4df58eaadba0d20cf3204aa3022650b5e62da`
- Immutable archive unchanged: yes
- Scope: introduce the homogeneous `DimensionType`/`DimensionValue` SSA foundation and checked dimension arithmetic
  operation family without yet introducing assertions, the heterogeneous array/dimension storage sum, or projected
  contexts
- Deferred: ordered requirements and their effects/partial-evaluation contract belong to `P2b`; generic storage-sum
  projection belongs to `P2c`; direct zero-state projected binding and the third-member extensibility gate belong to
  `P2d`; mixed array/dimension operation signatures remain in `P3`
- Implemented: adds `DimensionType` as a definition-position type, checked host `DimensionValue`s, a homogeneous
  tracing context, generic constant reuse, and one `DimensionArithmeticOperation` parameterized by nine arithmetic
  functions. Every arithmetic result owns a fresh bounded variable, runtime values may refine declared operands by
  narrowing their bounds, and arithmetic remains ordinary SSA through tracing, interpretation, and partial evaluation.
- Generic closure correction: definition-position constant types now establish immutable internal identities, all
  constant definitions are collected before references so atom-table order is irrelevant, duplicate definitions are
  rejected, and the existing unresolved-reference diagnostic remains unchanged. This is the general SSA rule required
  by any future definition-bearing constant type, not a dimension-specific exception.
- Surface control: nine arithmetic functions share one operation payload and one outer operation-family variant rather
  than nine nominal payload types and dispatch variants. P2a adds no expression tree, witness, substitution, custom
  constant operation, heterogeneous storage sum, projected context, or dimension-specific program machinery.
- Verification: `cargo check -p ryft-core -p ryft-xla` passed; `cargo test -p ryft-core --lib` passed all 934 tests;
  `cargo test -p ryft-core --doc` passed 43 tests with 13 ignored; all 53 `ryft-macros` unit tests and all 17 operation
  macro-integration tests passed; `cargo test -p ryft-xla --lib` passed 396 tests with one documented ignored benchmark;
  focused tests cover exact/narrowed inputs, fresh result identities, bound inference, all arithmetic functions, portable
  width and invalid-operation failures, eager execution, tracing, and known-side partial evaluation; formatting and
  `git diff --check` passed
- Inherited verification issue: the parameter trybuild golden mismatch reproduces unchanged because rustc includes
  `Axes` in a non-exhaustive implementor help list; P2a does not touch that diagnostic or its fixture
- Lint audit: strict workspace Clippy still reports the existing repository-wide warning backlog, but a targeted
  diagnostic search reports no warning in either P2a dimension file
- Residual search: no nominal per-function dimension operation payload, custom dimension constant operation, old
  arithmetic error name, projection wrapper, replay hook, or expression representation was introduced; comparisons,
  gateways, `dimension_size`, requirements, the storage sum, and direct projected binding remain assigned to their
  explicit later increments
- Review method: line by line
- Next action: none

## P2b: ordered dimension requirements

- Status: landed
- Branch: `u/eaplatanios/increment/p2b-ordered-dimension-requirements`
- Source commit: `b5f38959b3cdba68b96a0c84b387224fabf32cda`
- Integration commit: `b671b123acbde965ab8d0ea738482a22a825a9cb`
- Remainder reconciliation commit: `982333afccc46c04884db3d3fcd6cef44ac9d46a`
- Immutable archive unchanged: yes
- Scope: add homogeneous equality, less-than-or-equal, positive-divisibility, and explicit-bounds requirements;
  classify them from local exact/identity/interval facts; and integrate inconclusive checks with ordered effects and
  ordinary partial-evaluation placement
- Deferred: comparisons, gateways, `dimension_size`, and the storage sum remain in later P2/P3 increments;
  graph-wide entailment over producer arithmetic, preceding requirements, and nested regions remains in P4; backend
  assertion lowering remains in P7
- Implemented: adds one `DimensionRequirementOperation` payload parameterized by equality, less-than-or-equal,
  positive-divisibility, and explicit-bounds predicates. The private operand representation and public constructors
  enforce unary/binary arity without four nominal operation types. The same exact/shared-variable/interval fact lattice
  drives inference effects and partial evaluation, while refined literal operands keep the operation's declared names
  in diagnostics.
- Effects and placement: proven requirements are pure and disappear under ordinary simplification; statically
  impossible requirements return typed `DimensionError`s; inconclusive requirements carry the restored
  `Effect::OrderedAssertion`, survive zero-result DCE in source order, execute eagerly with observed-value diagnostics,
  and either fold on the known side or residualize exactly once according to ordinary partial-evaluation placement.
- Verification: `cargo check -p ryft-core` and `cargo check -p ryft-xla` passed; all 936 core library tests passed; core
  doctests passed 43 tests with 13 ignored; `cargo test -p ryft-xla --lib` passed 396 tests with one documented ignored
  benchmark; focused requirement and effect tests passed; `cargo fmt --all -- --check` and `git diff --check` passed.
- Lint audit: strict core Clippy remains blocked by the inherited repository-wide warning backlog; its diagnostic list
  contains no warning in the P2b dimension or effect files.
- Residual search: the production tree contains one requirement payload and one outer operation-family variant; there
  is no nominal per-predicate operation type, expression/witness representation, requirement-specific program path, or
  backend assertion lowering. `OrderedAssertion` appears only in the generic effect definition/tests and the
  requirement effect/test sites.
- Review method: line by line
- Next action: none

## P2b.1: canonical dimension operation modules

- Status: landed
- Branch: `u/eaplatanios/increment/p2b1-dimension-operation-modules`
- Source commit: `d95dcc9214cc26afeffe7d6c9ca0f419f26b1ba3`
- Integration commit: `4d4342009b8e45ac2998e25608fafc568b4db798` (review merge
  `8dd7be61ab75d5949adffc1675f3933ab6681303` plus reviewed follow-up commits)
- Remainder reconciliation commit: `fa34b4ffe`
- Immutable archive unchanged: yes
- Scope: replace tagged dimension arithmetic with one nominal type and capability per primitive, share their common
  operation contract through `ArithmeticDimensionOperation`, move arithmetic and requirement primitives into one
  canonical `operations::dimensions` submodule per payload, and leave concrete host values, eager adapters, and the
  reference operation family under backend ownership
- Deferred: the heterogeneous storage sum and generic projection remain P2c; projected binding remains P2d; mixed
  shape operations remain P3; neutral runtime-dimension API placement and final public-path audit remain P9
- Plan: `.tasks/plan_dimension_operation_modules.md`
- Verification: formatting and diff hygiene, 943 `ryft-core` library tests, 53 executable `ryft-core` doctests, 396
  `ryft-xla` library tests, focused operation/requirement tests, and changed-file clippy diagnostics
- Residual search: no `DimensionArithmetic` or `DimensionArithmeticOperation`; exactly nine nominal arithmetic
  payload modules and backend-family variants; no arithmetic selector enum or duplicate backend arithmetic semantics.
  The final macro follow-up moved eager interpretation into generic operation generation and thereby introduced a
  direct `macros -> backends::dimensions::DimensionValue` dependency; P2b.2 owns its removal before P2c.
- Review method: line by line, with the plan reviewed before production implementation and a final minimality and
  dispatch-boundary audit after verification
- Next action: land P2b.2's narrow backend-neutral interpretation correction

## P2b.2: restore backend-neutral dimension interpretation

- Status: landed
- Branch: `u/eaplatanios/increment/p2b2-backend-neutral-dimension-interpretation`
- Source commit: `dc89ecfaa`
- Integration commit: `fb8ba7812`
- Remainder reconciliation commit: `fa34b4ffe` (reconciles reviewed P2b.1 into the mutable remainder)
- Immutable archive unchanged: yes
- Scope: remove the concrete `DimensionValue` dependency from generic arithmetic operation generation, make each
  operation own generic capability-constrained interpretation, and make the reference backend own concrete dimension
  capability implementations
- Deferred: the heterogeneous storage sum and generic projection remain P2c; projected binding remains P2d; no
  operation, capability, bounds, or diagnostic semantics change in this correction
- Verification: formatting and diff hygiene, `cargo check -p ryft-core -p ryft-xla`, focused dimension/backend/macro
  tests, 946 `ryft-core` library tests, 53 executable `ryft-core` doctests, and 396 `ryft-xla` library tests
- Residual search: no arithmetic `evaluate` hook, backend-owned arithmetic/requirement `InterpretableOperation`, rich
  `DimensionValue::DispatchDomain`, or production operation-to-backend dependency remains; the requirement predicate's
  shared `evaluate_extents` is intentionally retained for eager enforcement and known-side reasoning
- Review method: line by line
- Next action: none

## P2c: generic storage-sum projection

- Status: landed
- Branch: `u/eaplatanios/increment/p2c-generic-storage-projection`
- Source commit: `064ed670259e3660506e46fcc26a9f1487ccecde`
- Integration commit: `e5aeef0c4557dda3eefcc80e32005a1c028417e7`
- Remainder reconciliation commit: `0057c4636`
- Immutable archive unchanged: yes
- Scope: introduce the array/dimension storage type and value sums plus generic borrowed and consuming projection
  contracts, preserving concrete eager payload ownership and symbolic SSA identity without introducing projected
  contexts or mixed operation contracts
- Deferred: direct projected binding and the third-member extensibility gate remain P2d; mixed operations, gateways,
  comparisons, and `dimension_size` remain P3; composite batching projection remains P5 because its current carrier is
  array-only and adding a heterogeneous batch representation would prematurely encode batching policy
- Implemented: adds the sole `ArrayProgramType`/`ArrayProgramValue<A>` storage sums, standard `From`/borrowed `TryFrom`
  type conversions, generic borrowed and consuming value projection, checked owned `ProjectedValue<T, V>` and borrowed
  `ProjectedValueRef<'v, T, V>` representations for symbolic identity preservation, and projection implementations for
  eager values, captures, tracers, partial tracers, and differentiation tracers
- Ownership and performance: eager array projection returns `&A` or transfers `A`; a 4,096-element reference array
  retains the same `Scalar` payload pointer through both paths, and isolated allocator tests measure zero allocations
  for 1,000 borrowed projections and for consuming projection
- Verification: formatting and diff hygiene, 958 `ryft-core` library tests, 53 executable `ryft-core` doctests with 15
  ignored, two zero-allocation integration tests, and 396 `ryft-xla` library tests with one ignored benchmark
- Lint audit: scoped Clippy remains nonzero because of the inherited warning backlog; no warning names a P2c-owned
  storage, projection, type, value, or allocation-test implementation
- Residual search: no archived `ArrayProgramProjection`, array/dimension-specific context view, projected context,
  `.cloned()` eager projection, mixed operation family, gateway, or `dimension_size` implementation was introduced
- Review method: line by line
- Next action: reconcile the reviewed increment into the mutable remainder before the P2d handoff

## P2d: zero-state projected binding

- Status: landed
- Branch: `u/eaplatanios/increment/p2d-projected-context`
- Source commit: `768ba6fc1`
- Integration commit: `85e75a9fe35a7f40dbf5585348bfcc0443366299` (including owner review follow-ups)
- Remainder reconciliation commit: `29e3ea7ad8a863c9d64538cfe52c342c14ad17f3`
- Immutable archive unchanged: yes
- Scope: add one generic zero-state context that binds homogeneous member operations directly through a composite
  parent operation family, plus the operation-family projection contract and the third-member extensibility gate
- Deferred: production array-program operation-family construction and genuinely mixed operations remain P3;
  region-carrying projected dispatch remains P4; batching and differentiation policies remain P5 and P6
- Implemented: adds `OperationProjection<T>` with standard `From`-based lifting, the parent-only
  `ProjectedContext<C, T>`, and one blanket
  `Value for ProjectedValue<T, V>`; projected binding lifts exactly the supplied values and operation, rejects regions,
  binds once through the parent, and projects the results without inspecting a program or reconstructing dependencies
- Allocation behavior: nullary, unary, and binary projected binds lift inputs through fixed-size stack arrays, while
  wider operations alone allocate a temporary input vector; output projection materializes the result vector required
  by `Context::bind`, and final production outer-dispatch allocation and latency measurement remains part of P10
- Extensibility gate: one test-only storage and operation family with three distinct member kinds exercises eager
  binding, tracing, resolution, exact SSA identity, and compile-time `Value` support for tracer, partial-tracer, and
  differentiation-tracer projections; the projected context has the same runtime size as its parent
- Verification: `cargo check -p ryft-core -p ryft-xla` passed; `cargo test -p ryft-core --lib` passed all 964 tests;
  `cargo test -p ryft-core --doc` passed 53 tests with 15 ignored; both allocation tests passed; and
  `cargo test -p ryft-xla --lib -q` passed 396 tests with one ignored
- Lint audit: strict Clippy remains nonzero because of 229 inherited warnings; the changed-file audit reports only the
  pre-existing `contexts.rs` type-complexity and clone-on-copy warnings, neither in P2d-owned code
- Residual search: one production `ProjectedContext`, one production `OperationProjection` trait, and one blanket
  `Value for ProjectedValue` remain; no array/dimension-specific context view, replay hook, source-array field,
  ambient-dimension field, or production mixed operation family was introduced; the remaining `with_dimensions`
  matches are the unrelated reshape-permutation builder and its tests
- Review method: line by line
- Next action: none

## P3a: production array-program dispatcher

- Status: landed
- Branch: `u/eaplatanios/increment/p3a-array-program-dispatcher`
- Source commit: `ac66c0bd2`
- Integration commit: `d44f9be37c1cc55b272cd6995d177cd7ca22a1ec`
- Remainder reconciliation commit: `526a597337763b0a3cd4e4c1da6a9e927ca5e417`
- Immutable archive unchanged: yes
- Scope: add the production two-family `ArrayProgramOperation<A>` dispatcher, standard homogeneous-family lifts, and
  complete array-program operation-contract projection without adding any genuinely mixed operation
- Deferred: `dimension_size`, data gateways, dimension comparison, mixed shape operations, region-carrying execution,
  transform policies, and backend lowering remain in their named P3b+ and P4+ increments
- Implemented: adds exactly one array-family and one dimension-family variant, standard `From` lifts and
  `OperationProjection` associations, one shared type/region-interface projection path, complete operation-contract
  forwarding, and generic region-free eager delegation; `ArrayProgramValue<A>` keeps its constant-only dispatch domain
  while using the new family for rich execution
- Verification: 7 focused dispatcher tests, both allocation integration tests, 967 core library tests, 53 executable
  core doctests, 396 XLA library tests, and `cargo check -p ryft-core -p ryft-xla` passed; formatting and diff hygiene
  passed
- Lint audit: the repository's inherited Clippy backlog remains, but no diagnostic names either P3a-owned backend file
- Residual search: exactly two outer family variants; no mixed variant, per-primitive production projection match,
  semantic context state, ambient dimension/source-array field, replay hook, or new context/value wrapper
- Review method: line by line
- Owner corrections: inlined the one-use `project_types`, `lift_types`, and `lift_region_input_types` helpers into the
  owning operation-dispatch paths before merging
- Next action: none

## P3b: canonical first-class dimension size

- Status: landed
- Branch: `u/eaplatanios/increment/p3b-dimension-size`
- Source commit: `21ed403fe27241d1965914d012a5fe2c4576f626`
- Integration commit: `017e775a551df03c07dd299c42ba90234e593f75`
- Remainder reconciliation commit: `00c6aa4991edc321a11c03f1aba576ceb5478979`
- Immutable archive unchanged: yes
- Scope: introduce the sole `array -> dimension` `DimensionSizeOperation`, add it as one genuinely mixed outer-family
  variant, and support explicit staging plus constant-time reference-backend eager extent extraction
- Deferred: dimension-to-data conversion remains P3c; data gateways remain P3d/P3e; comparison remains P3f;
  composite batching, differentiation/transposition, and backend lowering remain P5, P6, and P7
- Plan: `.tasks/plan_p3b_dimension_size.md`
- Audit correction: the archived operation's result-only validation rejects a valid dynamic-declared/static-actual
  eager refinement; the production payload retains the selected declared axis dimension and validates refinement
  directionally
- Implementation: added the sole mixed `array -> dimension` operation, dynamic identity forwarding, exact static
  results, a shared `DimensionSize<Output>` capability with constant-time `DimensionSize<usize>` reference-backend
  extraction, outer-family eager dispatch, projected-array staging, ordinary partial evaluation, and composite
  trace/import coverage
- Verification: focused operation and dispatcher tests, allocation integration tests, core check/lib/doc suites, XLA
  check/lib suite, formatting, diff check, changed-file Clippy attribution, and architectural residual searches pass
- Residual search: no homogeneous array contract, rank-zero integer result, runtime-dimension data type, array-family
  variant, carrier-specific wrapper tower, or implicit witness/source-array/replay mechanism; P3c/P5/P6/P7 retain their
  named data-conversion, transform, and lowering work
- Review method: line by line
- Owner corrections: consolidated concrete host extent extraction and composite first-class results under the single
  `DimensionSize<Output>` capability, removing `DimensionExtent` and its separate method vocabulary
- Remainder note: merging the reviewed integration retained later archived array-program work; the remainder's known
  pre-existing module-path collision and unmatched delimiter still prevent `ryft-core` from compiling independently
- Next action: none

## P3c: explicit dimension-to-scalar-array conversion

- Status: landed
- Branch: `u/eaplatanios/increment/p3c-dimension-to-scalar`
- Source commit: `7b224d993`
- Integration commit: `a7ae31167401e08c200768b2cf953b9266861507` (including owner review commits
  `dd5294aa7` and `98452be79`)
- Remainder reconciliation commit: `12398a196`
- Immutable archive unchanged: yes
- Scope: introduce the sole `dimension -> scalar-array` `DimensionToScalarOperation`, its shared value capability,
  one explicit mixed outer-family variant, and reference-backend eager materialization as a rank-zero signed 64-bit
  array
- Deferred: array-data-to-dimension gateways remain P3d/P3e and comparison remains P3f. At the owner's request, this
  increment includes the reusable composite batching, differentiation/transposition, and XLA lowering foundations
  necessary for the complete `dimension_size -> dimension_to_scalar` vertical slice.
- Plan: `.tasks/plan_p3c_dimension_to_scalar.md`
- Audit correction: the portable `DimensionValue` invariant already guarantees that every concrete extent fits the
  signed 64-bit result representation, so reference eager conversion does not need a second user-facing width failure
- Implemented: added the sole explicit `dimension -> rank-0 i64 array` operation and capability; direct outer-family
  dispatch; reference and composite eager conversion; stored-program batching, JVP/VJP, and transposition policies; and
  public composite StableHLO lowering for the complete `dimension_size -> dimension_to_scalar` path
- Performance: lowering keeps dynamic extents as scalar SSA and lowers the conversion itself as an identity; composite
  batch construction computes its logical array type without projecting or cloning eager array payloads
- Owner corrections: restored typed mapped-dimension diagnostics in every dimension-consuming batching rule, forwarded
  mapped-axis names and sharding into homogeneous array rules, validated the structural-zero rule's nullary contract,
  and documented the Phase 5 ownership of dynamic batch extents and region-driver use
- Verification: focused operation, batching, differentiation, lowering, and CPU execution tests; all 972 core library
  tests; 55 executable core doctests with 16 ignored; all 397 XLA library tests with one ignored; the two projection
  allocation tests; package checks; formatting; and diff hygiene pass
- Lint audit: strict workspace Clippy remains nonzero because of the inherited warning backlog; filtering diagnostics
  to increment-owned files found no new warning
- Runtime qualification: static StableHLO compiles and executes on the CPU PJRT plugin with the expected `i64` result.
  The bounded-dynamic module verifies and has the expected `get_dimension_size -> convert` path, but local CPU
  execution remains blocked by that plugin's unavailable `PadToStatic` custom call
- Residual search: no homogeneous operation contract or family variant, second output-provider abstraction, generic
  mixed bucket, reverse data-to-dimension gateway, expression reconstruction environment, production host readback, or
  copied archived transform/lowering monolith was introduced
- Review method: line by line, including a final ownership/allocation audit
- Next action: none

## P3d: checked scalar-array to dimension gateway

- Status: landed
- Branch: `u/eaplatanios/dynamic-shapes`
- Source commit: `eda43571b`
- Integration commit: `eda43571b`
- Owner follow-up commit: `8dc2c4f4b`
- Remainder reconciliation commit: pending
- Immutable archive unchanged: yes
- Scope: introduce the sole checked `rank-0 integer array -> dimension` gateway, with one fresh declared result
  identity, reference-eager validation, ordinary partial evaluation, replicated-only batching, and explicit deferred
  XLA lowering
- Deferred: indexed extraction from rank-one integer arrays remains P3e; dimension comparison remains P3f; checked
  device-side gateway lowering and observed-value assertions remain P7. That lowering must convert the input integer
  scalar to canonical `i64`, check nonnegativity, declared lower/upper bounds, and `MAX_DIMENSION_EXTENT`, preserve
  actor-named diagnostics with the observed value, and return the checked scalar as dimension SSA. Batching-adapter
  consolidation remains P5
- Plan: `.tasks/plan_p3d_dimension_from_scalar.md`
- Verification: all 975 core library tests; 56 executable core doctests with 16 ignored; all 398 XLA library tests
  with one ignored; the two projection allocation tests; package checks; formatting; and diff hygiene pass
- Residual search: exactly one production operation declaration and one outer-family variant; no homogeneous operation
  contract, data-source adapter, host-readback path, expression witness, source-array field, generic mixed bucket, or
  unchecked lowering was introduced; vector elements cross this scalar gateway only after ordinary array extraction
- Review method: line by line
- Owner follow-up: namespaced all dimension operation modules and arithmetic capability methods with the
  `dimension_` prefix while preserving standard operator syntax
- Next action: reconcile the landed commits into the mutable remainder during the next bookkeeping pass

## P3e: vector-element composition

- Status: landed
- Branch: `u/eaplatanios/dynamic-shapes`
- Source commit: `faa79e49296a6861022529fba4fb01567e196cc1`
- Integration commit: `faa79e49296a6861022529fba4fb01567e196cc1`
- Remainder reconciliation commit: pending
- Immutable archive unchanged: yes
- Scope: decide whether indexed vector-data-to-dimension conversion needs a dedicated mixed operation or should compose
  existing array extraction, scalarization, and `DimensionFromScalarOperation`
- Decision: remove `DimensionFromVectorElementOperation`, its capability, outer-family variant, eager implementation,
  batching rule, lowering deferral, exports, and tests. It fused generic indexing with an already canonical authority
  gateway, had no production consumer, and had no direct JAX or StableHLO counterpart
- Replacement: statically sized vectors use `slice -> reshape-to-scalar -> dimension_from_scalar`; this path is now
  documented on `DimensionFromScalar`. A future dynamic use case must use checked ordinary array indexing or an
  explicit logical-length requirement before scalarization. If existing array operations remain insufficient after
  mixed slicing migrates, add a general checked array-element operation rather than dimension-specific machinery
- Deferred: dimension comparison remains P3f; device-side checks for `DimensionFromScalarOperation` remain P7; general
  dynamic checked indexing is demand-driven and is not part of the dimension operation family
- Plan: `.tasks/plan_p3e_dimension_from_vector_element.md` records the rejected prototype and removal decision
- Verification: all 975 core library tests; 57 executable core doctests with 16 ignored; all 398 executable XLA
  library tests with one ignored benchmark; formatting; and diff hygiene pass
- Residual search: no production operation, capability, export, dispatcher variant, transform rule, backend
  implementation, lowering case, or test named `DimensionFromVectorElement` or
  `dimension_from_vector_element` remains
- Review method: line by line
- Next action: none

## P3f: first-class dimension comparison

- Status: landed
- Branch: `u/eaplatanios/dynamic-shapes`
- Baseline commit: `faa79e49296a6861022529fba4fb01567e196cc1`
- Source commit: `6a2af58e18cbb595878e682b8c54c1c9eeb40d9b`
- Integration commit: `162a8241b54212309f7caeb243959efe8a7e5cd3`
- Immutable archive unchanged: yes
- Scope: extend the canonical `CompareOperation` and `Compare<Output>` capability with the composite member signature
  `(Dimension, Dimension) -> Array(Boolean scalar)`, then carry it through eager execution, tracing, partial
  evaluation, batching, differentiation/transposition, import, and direct StableHLO lowering
- Design: treat comparison as a benign multi-contract operation; reuse its existing payload, direction enum,
  rendering, purity, and value capability rather than adding a dimension-specific operation or method vocabulary
- Prototype gate: the initial associated-output capability conflicted with the blanket staged-value implementation.
  Replacing the associated output with an `Output = Self` generic parameter makes homogeneous `Compare<Self>` and
  projected-dimension `Compare<ParentValue>` distinct coherent trait instantiations without adding a provider trait
- Explicit exclusions: no `DimensionCompareOperation`, `DimensionCompare` capability, Boolean dimension, expression,
  witness, bounds prover, host readback, data-gateway detour, or new projection/context abstraction
- Plan: `.tasks/plan_p3f_dimension_compare.md`
- Toolchain baseline: `rustc 1.93.1`, `cargo 1.93.1`
- Verification baseline: inherited P3e handoff passed 975 core library tests, 57 executable core doctests with 16
  ignored, and 398 executable XLA library tests with one ignored benchmark
- Implementation: `CompareOperation` now has the precise composite contract
  `(Dimension, Dimension) -> Array(Boolean scalar)`. The associated `Compare::Output` type became an
  `Output = Self` trait parameter so projected dimensions can select their parent array-program value as the output
  without overlapping the homogeneous blanket implementation. Projected dimensions bind the existing operation
  directly in the parent context, while homogeneous array comparisons retain their existing projected array-family
  path
- Execution and transforms: the reference eager backend compares checked extents directly; partial evaluation folds
  known inputs and residualizes unknown inputs once; batching accepts replicated dimensions and returns replicated
  predicate data; JVP assigns structural-zero differential space; transposition retains the canonical rejection
- Lowering: both scalar `i64` dimension operands feed the existing signed StableHLO comparison lowering directly.
  Structural checks prove one compare per result and no `dimension_to_scalar`; CPU PJRT execution covers equality and
  ordered comparison
- Verification: 976/976 core library tests; 58 executable core doctests with 16 ignored; 399/400 XLA library tests
  with one ignored benchmark; both projection-allocation guards; targeted core/XLA comparison tests; core/XLA
  compilation; formatting and whitespace checks
- Clippy attribution: the workspace-wide `-D warnings` run remains blocked by inherited diagnostics. Filtered
  changed-file output found no P3f diagnostics after removing the two `CompareOperation::render` needless borrows;
  remaining matches in touched legacy files predate and do not overlap this increment
- Residual audit: no dimension-specific comparison type, capability, or method vocabulary exists. One direct
  `ArrayProgramOperation::Compare(CompareOperation)` complements the pre-existing homogeneous array-family variant;
  no comparison result uses `DimensionType`/`DimensionValue`, and no expression, witness, bounds prover, host readback,
  data gateway, or extra projection/context abstraction was introduced
- Review size: 17 tracked files, 553 inserted and 89 removed lines relative to the baseline, including
  tests, documentation, ledger updates, and mechanical `Compare<Output = V>` to `Compare<V>` bound migrations; below
  the 800-line P3f review budget
- Review method: line by line
- Next action: none

## P3g: explicit reshape and broadcast dimensions

- Status: ready for owner review — Delivery D
- Branch: `u/eaplatanios/dynamic-shapes`
- Baseline commit: `beadf85bd3f96bbf8105bd6f48845a00a4ee2c4f`
- Delivery A source commit: `cd58f0a52`
- Delivery A integration commit: `cd58f0a52`
- Immutable archive unchanged: yes
- Scope: introduce the first two shape-producing mixed operations, giving reshape and broadcast the canonical
  signatures `(Array, output extent for axis 0, ..., output extent for axis rank - 1) -> Array`
- Design: every output axis is one first-class dimension SSA operand ordered by axis. Exact constants represent static
  axes, bounded non-exact values represent dynamic axes, and inference derives the complete output shape from operand
  types. Operation payloads retain only genuine non-shape semantic attributes; they never duplicate a declared shape
  or store dimension expressions, packed shape data, witnesses, or transform-only residual manifests
- Review staging: land the operand contract/flat-family containment, reshape, broadcast, and transform/lowering closure
  as separate owner-reviewed increments so the 42 explicit shape-operation bounds and 182 shape-operation calls are
  never migrated as one opaque sweep
- Transitional boundary: P3g adds and validates the canonical mixed paths. The following deletion increment migrates
  every remaining homogeneous consumer before removing `ReshapeDimensionExpression`, `DynamicBroadcastOperation`, the
  homogeneous reshape/broadcast contracts, and their legacy transform/lowering paths. No new production consumer may
  use those legacy paths after its P3g migration
- Explicit exclusions: no ambient dimension lookup, source-array recovery, expression evaluation in mixed lowering,
  packed rank-one shape operand, Boolean or arithmetic dimension metadata, ragged mapped dimensions, final
  `RuntimeShape` public API, or reshape-specific differentiation residual field
- Plan: `.tasks/plan_p3g_reshape_broadcast.md`
- Delivery A inventory: 42 explicit reshape/broadcast conversion bounds, 182 core reshape/broadcast method calls, and
  120 `ArrayProgramOperation` references across core and XLA at baseline
- Delivery A containment decision: retain the four existing cross-member primitives as direct
  `ArrayProgramOperation` variants. A derived nested `MixedArrayDimensionOperation` prototype reduced handwritten
  forwarding but leaked a storage-oriented family into the public API. Flattening avoids nested variants and requires
  no new derive surface. A projection-aware derive was rejected because projected interpretation would recreate the
  eager `Context`/`InterpretableOperation` obligation cycle and composite `Zero` still requires its existing
  array-result validation
- Delivery A operand contract: every reshape/broadcast output axis is represented by one ordered dimension operand.
  Exact dimension constants encode static axes, while bounded non-exact values encode dynamic axes. Inference derives
  output shape metadata directly from these operand types, avoiding shape duplication and identity reconciliation
- Delivery A schema disposition: the `DimensionOperandSchema` prototype, its module, and its tests were removed. The
  uniform positional contract only needs ordinary count/member-kind inference; a parallel segment and identity
  language would add complexity without representing additional semantics
- Verification baseline: 976 core library tests, 58 executable core doctests with 16 ignored, 399 XLA library tests
  with one ignored benchmark, and two projection-allocation guards
- Delivery A verification before removing the superseded schema prototype: 977 core library tests, 58 executable core
  doctests with 16 ignored, both projection-allocation guards, 399 XLA library tests with one ignored benchmark, all
  17 operation-derive integration tests including all eight compile-fail cases, core/XLA compilation, focused
  flat-dispatch tests, formatting, and whitespace checks. Post-removal verification is recorded in the P3g plan review
- Delivery A post-removal verification: all 976 core library tests passed, core/XLA compilation passed, formatting and
  whitespace checks passed, and a production-code residual search found no schema type, module, or use site
- Delivery A Clippy attribution: library-only Clippy reports 132 inherited diagnostics and none in a changed
  production file
- Delivery A residuals: no `MixedArrayDimensionOperation`, nested `ArrayProgramOperation::Mixed`, or explicit-type
  derive extension remains. The 42 legacy conversion bounds and 182 shape calls remain intentionally unchanged until
  the reshape, broadcast, and closure deliveries
- Delivery B canonical reshape: `ReshapeOperation` is now exclusively the mixed
  `(array, extent axis 0, ..., extent axis rank - 1) -> array` contract. Its output shape is derived from operand types,
  and its payload retains only input permutation and output-sharding attributes. Exact constants represent static
  axes; dynamic extent operands preserve their dimension-variable identities
- Delivery B legacy boundary: the behaviorally unchanged homogeneous/expression implementation is now
  `LegacyReshapeOperation`. Its expression evaluator, lowering, transform rules, and remaining production users are
  intentionally retained until the post-P3g deletion increment
- Delivery B transform/lowering closure: eager interpretation, partial evaluation, batching, JVP, static transpose,
  direct static/dynamic StableHLO lowering, projected binding, identity instantiation, and cross-program import are
  covered. Dynamic transpose is explicitly deferred to Phase 6 because recovering dynamic input extents requires
  transform-owned residuals
- Delivery B CPU limitation: static mixed reshape compiles and executes on CPU. Bounded-dynamic lowering is
  structurally verified, but end-to-end CPU compilation remains blocked by the existing `PadToStatic` custom-call
  requirement
- Delivery B verification: 979 core library tests, 58 executable core doctests with 16 ignored, 400 XLA library tests
  with one ignored benchmark, the empty XLA doctest suite, both projection-allocation guards, focused mixed/legacy
  reshape and transform/lowering tests, core/XLA compilation, formatting, whitespace checks, residual searches, and
  changed-file Clippy attribution
- Delivery C source: `7aef33d93c01e926fd98275ec476045cbe8f396d`
- Delivery C canonical broadcast: `BroadcastOperation` now consumes one array and one dimension operand per output
  axis; inference derives the result shape while the payload retains only output-axis mapping and output sharding.
  Eager execution, partial evaluation, batching, JVP, static transpose, identity instantiation/import, direct
  static/dynamic StableHLO lowering, and static CPU execution are covered.
- Delivery D combined slice: one stored program computes dimension multiplication and addition, reshapes with their
  results, then broadcasts using the same explicit SSA edges. It passes eager execution, partial evaluation, batching,
  JVP, identity instantiation/import, structural dynamic lowering, and static PJRT compile/execute coverage.
- Delivery D correction: known-side folding now embeds constant-resolvable values whose types define identities as
  local residual constants. This generic rule prevents a folded dimension result from becoming an unjustified
  boundary identity while leaving symbolic known values as ordinary residual inputs.
- Delivery D lowering boundary: bounds-proven dimension addition and multiplication lower directly to scalar
  StableHLO. Arithmetic without a sufficient upper-bound proof is rejected pending Phase 7 checked runtime
  assertions; all other first-class dimension operations retain the operation-named unsupported diagnostic.
- Delivery D residual inventory: 80 `ReshapeDimensionExpression`, 68 `LegacyReshapeOperation`, 81
  `LegacyBroadcastOperation`, and 35 `DynamicBroadcastOperation` core/XLA occurrences remain. Their exact owning files,
  42 legacy `From` bounds, 206 legacy method calls, deletion order, and acceptance suite are classified in the P3g
  plan. No occurrence is unrelated MLIR syntax.
- Delivery D verification: core/XLA checks; both focused combined tests; all 982 core library tests; 58 executable core
  doctests with 16 ignored; both projection-allocation guards; all 402 executable XLA library tests with one ignored
  benchmark; the empty XLA doctest suite; formatting; and whitespace checks passed. The inherited Clippy baseline
  reports no diagnostic in a Delivery D production file.
- Review method: line by line
- P3g source: Delivery D and its rendering follow-up landed at
  `a4f2c833b01667c363b6ec2aa56065cf8c2508cb`.
- Deletion-order correction: the classified legacy paths own the homogeneous public capabilities used by consumers
  assigned to Phases 4–9. They remain frozen and are deleted consumer by consumer during those migrations; attempting
  immediate wholesale deletion would require either implicit-dimension adapters or collapsing Phases 4–9 into one
  increment.
- Next action: execute P3h's explicit concatenate result-extent deliveries from
  `.tasks/plan_p3h_concatenate.md`

## P3h Delivery A: explicit concatenate result extent

- Status: ready for owner review — Delivery A
- Branch: `u/eaplatanios/dynamic-shapes`
- Baseline commit: `a4f2c833b01667c363b6ec2aa56065cf8c2508cb`
- Scope: add the canonical mixed `(Array..., Dimension) -> Array` concatenate contract, reference eager execution,
  tracing, partial evaluation, boundary instantiation, and cross-program import
- Representation: retain one axis-only `ConcatenateOperation` and implement both `Operation<ArrayType>` and
  `Operation<ArrayProgramType>` for it. A second nominal concatenate payload was rejected as redundant because the two
  contracts have identical payloads; only their operand type families differ
- Inference: require one or more array operands followed by exactly one result-extent dimension, preserve the
  existing data-type/rank/memory/non-axis/sharding rules, clear layout when concatenation changes shape, preserve a
  sole identical input, reject contradictory exact sums, and take dynamic result identity solely from the final
  operand
- Eager execution: borrow-project every array member, compute the observed axis sum with checked `usize` arithmetic,
  report expected and supplied extents on mismatch, and delegate successful execution to the existing array
  concatenate kernel
- Stored-program acceptance: the dynamic golden contains two `dimension_size` instructions, one `dimension_add`, and
  one concatenate with the exact array and result-extent SSA edges. All-known partial evaluation folds; either an
  unknown array or unknown extent retains one concatenate. Generic identity instantiation and program import rename
  and preserve the trailing result identity
- Delivery boundary: composite batching and differentiation return explicit P3h Delivery B errors; composite lowering
  returns an explicit P3h Delivery C error. No generic zero-tangent or homogeneous lowering path silently handles the
  new mixed variant
- Residuals: no second nominal concatenate payload exists. Homogeneous concatenate consumers in `ArrayOperation`, the
  reference backend, `XlaOperation`, and legacy lowering are unchanged from the baseline. No new expression, witness,
  packed shape operand, implicit source recovery, or ambient dimension lookup was introduced
- Verification: core/XLA checks; focused mixed and homogeneous concatenate tests; all 985 core library tests; 58
  executable core doctests with 16 ignored; both projection-allocation guards; all 402 executable XLA library tests
  with one ignored benchmark; the empty XLA doctest suite; formatting; whitespace checks; and residual searches pass.
  Library Clippy remains blocked by the inherited 132 `ryft-core` warnings and 10 `ryft-mlir` errors, with no
  diagnostic in a Delivery A production file
- Review method: line by line
- Next action: owner review, commit, and push Delivery A; then execute P3h Delivery B batching and differentiation

## P3h Delivery B: explicit concatenate transforms

- Status: ready for owner review — Delivery B
- Branch: `u/eaplatanios/dynamic-shapes`
- Baseline commit: `0461874998fb87435722c8eef0ef26b1c294eb11`
- Scope: preserve the canonical mixed concatenate's trailing result-extent operand through composite batching, JVP,
  and static transpose while retaining the Phase 6 dynamic-transpose boundary
- Batching: require replicated extent authority, return the typed `BatchingError::MappedDimension` for a malformed
  mapped extent, align mapped and replicated array operands through `ArrayBatch::match_axis`, preserve the composite
  mapped-axis sharding when materializing a replicated operand, shift the logical concatenate axis around the common
  physical batch axis, and stage the same mixed concatenate with the unchanged extent value. The payload
  `ConcatenateOperation` owns this rule and the outer composite operation dispatcher only forwards to it
- Differentiation: stage primal and live-tangent concatenates with one shared transformed extent SSA value; materialize
  structural zero array tangents in the existing projected array context rather than broadening the composite
  operation contract; delegate static transpose to the homogeneous slice pullback and append a structural-zero extent
  cotangent
- Dynamic boundary: reject transpose when any concatenated input axis is dynamic with the exact Phase 6
  dimension-residual diagnostic; JVP remains supported for both exact and dynamic extents
- Composition: the stored `dimension_size(left), dimension_size(right), dimension_add, concatenate` program passes
  JVP and composite batching. Its JVP graph contains primal and tangent concatenate instructions whose final operand
  is the same transformed `dimension_add` result
- Residuals: no P3h Delivery B placeholder rejection remains. No expression, witness, packed shape operand,
  source-array recovery, or ambient dimension lookup was introduced. Direct composite lowering remains the explicit
  Delivery C boundary
- Verification: focused concatenate batching/differentiation tests; all 986 core library tests; 58 executable core
  doctests with 16 ignored; both projection-allocation guards; all 402 executable XLA library tests with one ignored
  benchmark; the empty XLA doctest suite; core/XLA compilation; formatting; whitespace checks; and residual searches
  pass.
  Library Clippy remains blocked by the inherited 132 `ryft-core` warnings and 10 `ryft-mlir` errors, with no
  diagnostic in a Delivery B production file
- Review method: line by line
- Next action: owner review, commit, and push Delivery B; then execute P3h Delivery C direct lowering and closure

## P3h Delivery C: explicit concatenate lowering and closure

- Status: ready for owner review — Delivery C
- Branch: `u/eaplatanios/dynamic-shapes`
- Baseline commit: `21b3a87c154560a345a351e072e3a6fac14d03db`
- Scope: lower exact canonical mixed concatenate programs directly, preserve the dynamic runtime-equality boundary,
  compile and execute the exact path on CPU, measure the golden program, and close the residual audit
- Lowering: the canonical four-instruction
  `dimension_size(left), dimension_size(right), dimension_add, concatenate` program lowers to scalar constants, one
  scalar `stablehlo.add`, and one `stablehlo.concatenate` over the two physical array operands. The explicit extent is
  consumed as compile-time authority after mixed inference proves equality with the exact input-axis sum
- Dynamic boundary: a dynamic input-axis sum returns
  `concatenate with first-class dimensions requires runtime equality assertion lowering when its explicit result
  extent is not statically proven equal to the input extent sum`; lowering does not trust or reconstruct the extent
- CPU execution: inputs `[1, 2]` and `[3, 4, 5]` compile and execute to `[1, 2, 3, 4, 5]`
- Measurements: 4 stored instructions, 223 rendered-program bytes, 379 StableHLO bytes, 28,795 microseconds warm local
  CPU compile time, and 354 microseconds for execution, synchronization, and host copy. No timing instrumentation
  remains in production or tests
- Residuals: no legacy concatenate payload exists. Intentional homogeneous consumers are the reference/public array
  capability and transforms, `ArrayOperation`, `XlaOperation`, their frozen lowerers, and tests. No expression,
  witness, packed shape operand, source-array recovery, ambient lookup, `get_dimension_size` reconstruction, or host
  readback was introduced
- Verification: focused lowering and CPU execution passed; all 404 XLA library tests passed with one ignored test;
  the empty XLA doctest suite, core/XLA compilation, formatting, whitespace checks, and residual searches passed.
  Strict XLA Clippy remains blocked by inherited diagnostics, with none in Delivery C changes
- Review method: line by line
- Next action: owner review, commit, and push Delivery C; then continue the Phase 3 mixed-operation migration
