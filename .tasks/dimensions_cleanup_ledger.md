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

- Status: ready for owner review
- Branch: `u/eaplatanios/increment/p2c-generic-storage-projection`
- Source commit: pending
- Integration commit: pending owner review
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
- Next action: commit and push the increment branch, then stage its no-commit merge for owner review
