# Ragged Collectives Plan

Close the explicit ragged-communication gap with JAX: add a new `operations::cumulative` module covering JAX's full
cumulative-reduction family (`cumulative_sum` is needed by the ragged transpose mask; the rest complete the family),
rename the `p*` collectives to their full-word `parallel_*` forms, split `operations::collectives` into
per-operation-kind submodules, add a first-class `ragged_all_to_all` collective (packed data + explicit
offsets/sizes, mirroring `jax.lax.ragged_all_to_all`), add a declared ragged batching contract to
`CustomCallOperation` (deliberately exceeding JAX, which only supports the manual explicit-operand pattern), keep
the per-collective ragged rules as an explicitly gated phase, and add `ragged_dot_general` (grouped matrix
multiplication with explicit `group_sizes`, JAX's surviving explicit-raggedness track) as a final phase. The
existing pre-binding rejection of `RaggedAxis` operands stays the default for every operation without an exact
contract.

Reference semantics (verified against `jax/_src/lax/parallel.py` and the JAX docs):

- Six operands: `operand (N, A, ...)`, `output (M, A, ...)`, and rank-1 integer `input_offsets`, `send_sizes`,
  `output_offsets`, `receive_sizes`, all of equal length `K` divisible by the effective group size. Result has the
  output's type; unwritten output elements pass through.
- `output_offsets` are in the **receiver's** coordinate frame; `send_sizes == all_to_all(receive_sizes)` must hold.
- JVP is jointly linear in `(operand, output)`: tangent result is `ragged_all_to_all` of the tangents with the same
  metadata. Metadata is nondifferentiable.
- Transpose uses another `ragged_all_to_all` with roles swapped, collectively permuted offsets, and an internal
  additive-update mode so cotangents from overlapping forward send regions accumulate:
  `operand_cotangent = ragged_all_to_all(cotangent, zeros, all_to_all(output_offsets), receive_sizes,
  all_to_all(input_offsets), send_sizes)`; `output_cotangent` is the cotangent with the received regions masked to
  zero (regions defined by the permuted `output_offsets` and `receive_sizes`).
- vmap over an unrelated axis merges the batch dimension into the packed leading axis (offsets rebased by
  `iota * N` / `iota * M`) and rejects `axis_index_groups`. Vmap over the collective's own named axis uses the
  eager physical participant path, including groups; staged nonconstant metadata remain explicitly gated.
- Lowering is `stablehlo.custom_call @ragged_all_to_all` with `api_version = 4` (typed FFI) and a dictionary
  `backend_config` carrying `replica_groups` (equally sized groups required) plus `channel_id` under SPMD.

## Provenance integration (cross-cutting)

`ryft-core` now attaches a [`Provenance`](crates/ryft-core/src/programs/provenance.rs) to every instruction, with
contexts exposing `invoke_with_provenance_scope` / `invoke_with_provenance_origin` and transform replay threading
each source instruction's origin into its rule automatically. Two consequences for this plan:

- Every framework-owned *multi-instruction* expansion staged by this plan enters nested
  `ryft::<subsystem>::<concept>` provenance scopes around exactly the staged primitives, mirroring the
  `ryft::differentiation::coordinate_basis` idiom (`differentiation/types.rs:460`), and documents that the scopes
  are purely diagnostic — nothing may match on them for correctness. Concretely: the Phase 1A generalized identity
  masking (`ryft::batching::ragged_identity_mask`), the Phase 1B associative-scan JVP decomposition
  (`ryft::differentiation::associative_scan`), the Phase 5 transpose's metadata permutation and marker-mask
  construction (`ryft::differentiation::ragged_all_to_all_transpose`), and the Phase 6 merged vmap rule's
  reshape/offset-rebasing arithmetic (`ryft::batching::ragged_all_to_all`). Single-instruction staging (the Phase 4
  mesh-staged operation, Phase 8 discharge) needs no explicit scopes — ambient origin threading covers it.
- Tests and snapshots are unaffected by default: provenance renders only under
  `ProgramRenderingMode::WithProvenance` (comment-style ` ; ...` suffixes), and provenance-free modules keep
  byte-identical StableHLO text at lowering (`ryft-xla` `lowering.rs:3859`). Where a phase introduces a scope, add
  one exact structural assertion that every instruction in the expansion carries the expected nested provenance; an
  exact full `WithProvenance` rendering may additionally pin its presentation when useful. Phase 7 lowering snapshots
  of provenance-carrying transformed programs print debug information, so pin those snapshots with locations in mind.

## Phase 1: Cumulative operations and logarithmic math primitives

Scheduled first because it is independent of the collectives work, and split into two independently completable
subphases so the ragged work is never blocked by the nonlinear members' AD and numerics edge cases: the ragged
phases (4–7) depend only on Phase 1A's `cumulative_sum`; Phase 1B completes the family and the math primitives it
needs.

Shared design for the whole cumulative family (`jax/_src/lax/control_flow/loops.py`): all five operations share one
payload shape (`axis: usize`, `reverse: bool`), one type-inference contract, one batching rule, and one lowering
pattern; they differ in combine function, identity, data-type domain, and differentiation. JAX's alternative
`chlo.ScanOp` GPU lowering is dead code (its feasibility gate returns `False` unconditionally), so the
full-prefix-window `reduce_window` lowering — which XLA's `TryOptimizeAssociativeScan` rewriter converts into an
efficient parallel scan — is the single faithful implementation for every member. Submodules are named after the
full operation names.

### Phase 1A: `operations::cumulative` module with `cumulative_sum`

- [x] Create `operations/cumulative/` with `mod.rs` (facade re-exports plus the shared machinery) and
  `cumulative_sum.rs`. Shared machinery in `mod.rs`: payload validation and type inference (output type equals
  input type; `axis` in bounds; the **scanned dimension must be static** — bounded or unbounded dynamic scanned
  dimensions are rejected with a precise diagnostic, with physicalize-at-the-bound-plus-identity-masking noted in
  the rustdoc as the possible future extension; where the type model expresses sharding, the scanned axis must be
  unsharded, mirroring JAX's sharding rule), the axis-shift batching helper (bump `axis` when the inserted batch
  axis precedes it, JAX's `_cumred_batch_rule`), and the eager sequential prefix-scan interpreter parameterized by
  the combine function (reversed iteration when `reverse`; zero-length scan axis returns the input unchanged).
- [x] Generalized ragged identity masking: the existing machinery cannot serve the cumulative family —
  `DynamicArrayBatchingPolicy::mask_reduction_input` zero-masks and rejects every kind but `Sum`
  (`batching.rs:2933`), and `ReductionKind` has no product or log-add-exp variant. Add a masking capability that
  accepts a staged identity *value* (rather than widening `ReductionKind`), used by cumulative operations to mask
  padding on the scanned axis with the member's own identity. Because cumulative operations do not consume the
  axis, the input `RaggedAxis` is preserved unchanged on the output; ragged axes on other axes pass through
  untouched.
- [x] `cumulative_sum.rs`: `CumulativeSumOperation` (name `cumulative_sum`), identity zero, data types real or
  complex numeric. The only *linear* member: its JVP applies the operation to the tangent and its transpose rule
  is itself with `reverse` flipped (JAX's `_cumsum_transpose_rule`; `reverse` is in the payload precisely so the
  transpose is closed). Capability trait `CumulativeSum` with `cumulative_sum(axis)` and
  `reverse_cumulative_sum(axis)` sugar.
- [x] XLA lowering: `stable_hlo::reduce_window` (first `ryft-xla` use of the existing `ryft-mlir` wrapper) with
  `window_dimensions[axis] = n`, padding `(n - 1, 0)` forward / `(0, n - 1)` reverse, unit strides and dilations,
  a zero init, and an `add` reducer body — exactly JAX's `cumred_reduce_window_impl` — documenting the reliance on
  XLA's associative-scan rewriter.
- [x] Wire into the `arrays::operations` enums and dispatch; verify with `ryft-macros` and `ryft-macros-tests` in
  addition to the core suites (new enum members flow through derive-generated dispatch). Tests per the math-op
  conventions: eager interpretation against hand-computed values including `reverse`, gradient checks exercising
  the transpose flip, batching axis shift, ragged identity-masking with `RaggedAxis` preservation, lowering
  snapshots, **and CPU XLA execution tests** (this is Ryft's first `reduce_window` lowering — snapshots alone do
  not prove it executes). Changelog updates are excluded from this effort by request.

### Phase 1B: Remaining cumulative members and math additions

Independently completable; nothing in Phases 2–9 depends on it.

- [x] `math/log1p.rs`: unary elementwise `Log1pOperation` (name `log1p`, matching Rust's canonical `ln_1p` /
  StableHLO's `log_plus_one` concept), computing `log(1 + x)` accurately near zero, real floating-point only
  initially (complex parity needs a different formula and eager implementation; JAX treats it separately). JVP:
  `tangent / (1 + x)`. Eager interpretation via Rust's `ln_1p`; XLA lowering to `stablehlo.log_plus_one` (the
  `ryft-mlir` wrapper already exists).
- [x] `math/log_add_exp.rs`: binary elementwise `LogAddExpOperation` (name `log_add_exp`), real floating-point
  only, mirroring JAX's pinned `logaddexp` (`jax/_src/lax/other.py`) exactly. Primal:
  `select(isnan(a - b), a + b, max(a, b) + log1p(exp(-|a - b|)))` — the `isnan(a - b)` arm covers same-sign
  infinities and NaN operands, so `(+inf, +inf) → +inf`, `(-inf, -inf) → -inf`, mixed infinities → the larger
  operand, NaN propagates. JVP: weights `exp(replace(x) - replace(result))` per operand where `replace` maps
  `+inf → 0` (JAX's `_replace_inf`), giving these pinned exceptional tangents (conventions of the rule, not
  mathematical extensions — encode them as exact test expectations):
  `(+inf, +inf) → t_a + t_b`; `(-inf, -inf) → NaN` (the `-inf` primal is not replaced, so the weights are
  `exp(NaN)`); `(finite a, +inf) → exp(a) * t_a + t_b` (the replaced `+inf` primal makes the finite operand's
  weight `exp(a)`, JAX's literal behavior); `(finite a, -inf) → t_a`; any NaN operand → NaN. Eager interpretation
  via the same select construction; XLA lowering by expansion into existing StableHLO operations (StableHLO has no
  logaddexp primitive); standard binary-elementwise batching.
- [x] `math/log_sum_exp.rs`: axis-reduction `LogSumExpOperation` (name `log_sum_exp`), real floating-point only,
  following `ReduceOperation`'s axes conventions, computing `log(sum(exp(x)))` over the reduced axes with JAX's
  pinned guarded construction (`jax/_src/ops/special.py`) — the naive max-shift alone computes
  `-inf - -inf = NaN` on all-`-inf` input: reduce the maximum with a `-inf` initial value, replace a nonfinite
  maximum with zero, stop its gradient, compute the shifted exponentials against that safe maximum, and add the
  same safe maximum back, so all-`-inf` and empty reductions pin to `-inf` (`log(0) + 0`) without NaNs. Documented
  as the **unweighted, unmasked subset** of `jax.nn.logsumexp` — the `b` weights, `where` mask, sign return, and
  complex behavior are explicit non-goals. JVP: softmax-weighted tangent reduction. Ragged masking uses the `-inf`
  identity through the Phase 1A generalized masking; XLA lowering by expansion through the same guarded
  decomposition.
- [x] Remaining cumulative members, each consuming the shared Phase 1A machinery plus a new staged associative-scan
  decomposition helper in `mod.rs` (JAX's log-doubling construction with `_interleave` built on interior-padded
  `PadOperation`, which Ryft's pad already supports) that their JVP rules differentiate through (JAX's
  `_cumulative_jvp_rule`; reverse mode then flows through those primitives under partial-eval linearization with
  no bespoke transpose rules):
  - `cumulative_product.rs`: `CumulativeProductOperation` (name `cumulative_product`), identity one, real or
    complex numeric.
  - `cumulative_max.rs`: `CumulativeMaxOperation` (name `cumulative_max`), identity the data type's minimum
    (negative infinity for floats), real numeric only.
  - `cumulative_min.rs`: `CumulativeMinOperation` (name `cumulative_min`), identity the data type's maximum
    (positive infinity for floats), real numeric only.
  - `cumulative_log_sum_exp.rs`: `CumulativeLogSumExpOperation` (name `cumulative_log_sum_exp`), real
    floating-point only, identity negative infinity. The combine function is `log_add_exp`, both in the eager
    scan and in the associative-scan decomposition its JVP differentiates through; the `reduce_window` reducer
    body reuses the same stable expansion.
- [x] Per-operation capability traits with `cumulative_*(axis)` / `reverse_cumulative_*(axis)` sugar; enum wiring
  with `ryft-macros` verification; per-member core tests (eager values including `reverse` and the infinity/NaN
  edge cases, decomposition-JVP gradient checks, batching axis shift, ragged identity-masking per identity —
  product one, max/min dtype extrema, log-sum-exp `-inf` — with `RaggedAxis` preservation).
- [x] XLA lowerings for the seven Phase 1B additions, with lowering snapshots and CPU XLA execution tests, and the
  `ryft-macros-tests` run they unblock. Changelog updates were excluded from this effort by request.

## Phase 2: Rename `p*` collectives to full-word `parallel_*` names

JAX's `p` prefix abbreviates "parallel"; Ryft's full-word naming convention applies, and no backwards compatibility
is required. Mechanical rename in the flat file, done before the module split so the split stays verifiably
move-only. Rename mapping:

- The reduction family renames to `ParallelReduce` (not bare `Reduce`, which `operations/math/reduce.rs` already
  owns for positional-axis reduction, and not `AllReduce`, which names the mesh lowering rather than the named-axis
  semantics): `CollectiveOperation` → `ParallelReduceOperation`, the `Collective` trait → `ParallelReduce`, and
  `CollectiveKind` → `ParallelReductionKind` with variants `{Sum, Mean, Max}` (no repeated prefix). Operation name
  strings `psum`/`pmean`/`pmax` → `parallel_sum`/`parallel_mean`/`parallel_max`; trait methods likewise become
  `parallel_sum`/`parallel_mean`/`parallel_max`. Shared module vocabulary (`CollectiveOptions`, `CollectiveMode`,
  `stage_collective`, `forward_collective_to_parent`, `reject_ragged_collective_inputs`, ...) keeps the
  "collective" term, which now unambiguously means the whole module's operation family.
- `PSumScatterOperation` / `PSumScatter` / `psum_scatter` / `PSUM_SCATTER_OPERATION_NAME` →
  `ParallelSumScatterOperation` / `ParallelSumScatter` / `parallel_sum_scatter` /
  `PARALLEL_SUM_SCATTER_OPERATION_NAME`.
- `PpermuteOperation` / `ppermute` / `PPERMUTE_OPERATION_NAME` → `ParallelPermuteOperation` / `parallel_permute` /
  `PARALLEL_PERMUTE_OPERATION_NAME`; `Pshuffle` / `pshuffle` → `ParallelShuffle` / `parallel_shuffle`.
- `PSwapAxes` / `pswapaxes` → `ParallelSwapAxes` / `parallel_swap_axes`.
- Helper functions: `pmean_batch_size` / `pmean_factor_type` / `psum_scatter_output_type` →
  `parallel_mean_batch_size` / `parallel_mean_factor_type` / `parallel_sum_scatter_output_type`.
- `ArrayOperation` / `ArrayIrOperation` variants `PSumScatter` / `Ppermute` → `ParallelSumScatter` /
  `ParallelPermute`, and any variant/embedding of the reduce family follows the `ParallelReduce` rename;
  capability-trait references in `arrays/operations/mod.rs` docs updated to match.
- `all_gather` and `all_to_all` are already full words and keep their names.

Checkable items:

- [x] Apply the rename across `ryft-core` (operation module, `arrays/operations`, `arrays/batching.rs`, tests) and
  `ryft-xla` (`experimental/lowering.rs`, `experimental/shard_map.rs`, `experimental/ops.rs`,
  `experimental/domains.rs`, `bin/differential_testing.rs`), including error-message text, staged-program rendering
  expectations, and rustdoc prose (rereading each touched paragraph as a unit).
- [x] Run a targeted search for every old identifier (`psum`, `pmean`, `pmax`, `ppermute`, `pshuffle`, `pswapaxes`,
  `Ppermute`, `PSum`, `PMean`, `PMax`, `PSwapAxes`, `CollectiveKind`, `CollectiveOperation`, and standalone
  `Collective` outside the retained shared-vocabulary names) to confirm no references remain.
- [x] Audit for anything keyed on the old rendered operation names beyond in-repo test expectations (the
  differential-testing binary, any serialized fixtures or persisted compilation-cache identities); Ryft's jit caches
  are in-memory, so this is expected to be a quick confirmation, but it must be explicit.
- [x] Verify: `cargo check -p ryft-core --all-targets`, `cargo check -p ryft-xla --all-targets`, scoped `--lib` test
  runs for both crates plus `ryft-macros` and `ryft-macros-tests` (renamed enum variants flow through the operation
  derives), 300s timeouts, `cargo fmt` check.

### Phase 2 review

- Renamed the reduction family (`ParallelReduceOperation` / `ParallelReduce` / `ParallelReductionKind` with
  `{Sum, Mean, Max}`), the shape-changing payloads (`ParallelSumScatterOperation`, `ParallelPermuteOperation`), the
  capability traits (`ParallelSumScatter`, `ParallelPermute`, `ParallelShuffle`, `ParallelSwapAxes`), the operation
  name constants and strings, the helper functions, and the `ArrayOperation` / `ArrayIrOperation` / `XlaOperation`
  variants across 14 files in `ryft-core` and `ryft-xla`. Error messages, staged-program rendering expectations, test
  names, and rustdoc prose follow the new vocabulary; reflowed every touched paragraph to the 120-column limit.
- JAX-side references keep JAX's own spelling (`jax.lax.psum` / `pmean` / `pmax` / `psum_scatter` / `ppermute` and the
  "matching JAX's ..." comparisons), because they name JAX's API rather than Ryft's.
- Shared module vocabulary (`CollectiveOptions`, `CollectiveMode`, `CollectiveBatchingPolicy`,
  `CollectiveLoweringState`, `stage_collective`, `forward_collective_to_parent`, the `*_collective_*` helpers,
  `AllGather*` / `AllToAll*`) is unchanged, as is `test_collective_options_validate_axis_index_groups`, which covers
  `CollectiveOptions` rather than the reduction family.
- Nothing outside the repo is keyed on the old rendered operation names: the differential-testing observation payload
  carries only numeric values plus StableHLO module text, and its `case_id` / observation keys (`"pshuffle"`,
  `"pswapaxes"`, `"psum_scatter"`) name the JAX operators under comparison, so they stay as-is and keep the Python
  harness contract intact. No serialized fixtures reference the old names, and the jit caches are in-memory.
- Verified with `cargo check -p ryft-core --all-targets`, `cargo check -p ryft-xla --all-targets`,
  `cargo test -p ryft-core --lib` (1575 passed), `cargo test -p ryft-xla --lib` (531 passed),
  `cargo test -p ryft-xla --features differential-testing --bin differential_testing` (2 passed),
  `cargo test -p ryft-macros`, and `cargo test -p ryft-macros-tests` (17 passed).

## Phase 3: Split `operations::collectives` into per-operation-kind submodules

Pure refactor after Phase 2, no behavior change. `mod.rs` holds the facade plus all shared vocabulary, helpers, and
macros; every other submodule corresponds to exactly one operation kind.

- [x] Create `crates/ryft-core/src/operations/collectives/` with this layout:
  - `mod.rs`: facade (`mod` declarations + `pub use` re-exports preserving today's public paths) plus everything
    shared: `CollectiveMode`, `CollectiveOptions`, `effective_collective_axis_size`,
    `reject_ragged_collective_inputs`, `forward_collective_to_parent`, `stage_collective`,
    `resolve_named_axis_size`, `interpret_degenerate_collective`, the shape-changing engine
    (`shape_changing_collective_dimensions`, `shape_changing_collective_output_type`,
    `infer_explicit_shape_changing_collective_output_type`, `forward_shape_changing_collective`,
    `explicit_collective_inputs`, `forward_explicit_collective`, `jvp_shape_changing_collective_with_adjoint`,
    `transpose_shape_changing_collective`), the extent helpers (`collective_extent_constant`,
    `collective_input_extents`, `multiplied_collective_extent`, `divided_collective_extent`,
    `require_collective_axis_extent`, `require_collective_axis_divisible`), and the `shape_changing_collective!`
    and `impl_shape_changing_collective_member_operation!` macros (defined with `macro_rules!` + `pub(super) use`
    so the operation submodules can invoke them).
  - `parallel_reduce.rs`: the reduction collective kind: `ParallelReductionKind`, `ParallelReduceOperation`, the
    `ParallelReduce` trait (`parallel_sum`/`parallel_mean`/`parallel_max`), `collective_reduce_batch`,
    `parallel_mean_batch_size`, `parallel_mean_factor_type`.
  - `all_gather.rs`: `AllGatherOperation`, `AllGather` trait, `AllGatherOutputVariance`, `all_gather_output_type`,
    `jvp_invariant_all_gather`.
  - `parallel_sum_scatter.rs`: `ParallelSumScatterOperation`, `ParallelSumScatter` trait,
    `parallel_sum_scatter_output_type`.
  - `parallel_permute.rs`: `ParallelPermuteOperation`, `ParallelShuffle` trait.
  - `all_to_all.rs`: `AllToAllOperation`, `AllToAll` and `ParallelSwapAxes` traits.
  - `ragged_all_to_all.rs`: added in Phase 4 (do not create an empty file in this phase).
- [x] Split the existing `#[cfg(test)] mod tests` by operation into each submodule's own `tests` module (reduction
  and vmap-gradient tests into `parallel_reduce.rs`, type-inference and batched-axis materialization tests into their
  owning operation files, options/forwarding/vocabulary and shared explicit-member/involution tests into `mod.rs`),
  following `.agents/unit-testing-guidelines.md`.
- [x] Fix imports per the repo conventions: `mod.rs` re-exports its children by relative path; operation submodules
  import shared items via `super::`; all out-of-module users keep importing through the unchanged
  `crate::operations::collectives::*` facade (verify `operations/mod.rs`, `arrays/operations/mod.rs`,
  `arrays/batching.rs`, and `ryft-xla` imports still resolve without edits, and update any that named the old flat
  file path directly).
- [x] Verify: `cargo check -p ryft-core`, `cargo test -p ryft-core --lib collectives` (300s timeout), `cargo fmt`
  check, and a `git diff` review confirming the split is move-only relative to Phase 2 (no semantic drift).

## Phase 4: `RaggedAllToAllOperation` core semantics

The operation contract is explicitly packed: six array operands, returning the updated packed output. `RaggedAxis`
is batching-time metadata only (`arrays/batching.rs:49`), so it plays no role in the operation's type contract; any
`RaggedAxis` integration is a batching-rule adapter (Phase 6), never part of this surface.

- [x] Add `collectives/ragged_all_to_all.rs` defining `RaggedAllToAllOperation` with fields `axis_name: String`,
  `axis_size: usize`, and `axis_index_groups: Option<Vec<Vec<usize>>>`. The stored `axis_size` follows the
  established shape-changing collective pattern (`collectives.rs:682`): `Operation::infer_output_types` only sees
  input types, so the user-facing capability trait resolves the size via `resolve_named_axis_size` at staging time
  and bakes it into the payload, and the grouped constructor validates group coverage against it like
  `CollectiveOperation::grouped` does (`CollectiveMode` does not apply — there is no tiled/untiled choice). Operand
  order matches JAX: `operand`, `output`, `input_offsets`, `send_sizes`, `output_offsets`, `receive_sizes`
  (full-word naming, not `recv_sizes`). Operation name string: `ragged_all_to_all`.
- [x] Type inference: result type equals the `output` operand's type. Validate: `operand`/`output` share the data
  type and trailing (non-leading) dimensions; the four metadata operands are rank-1 integer arrays of one shared
  integer data type and equal static length `K`; `K > 0` (zero is divisible by every group size, and JAX's
  abstract evaluation rejects empty metadata vectors — pin the exact zero-length diagnostic with a test); and `K`
  is divisible by the effective group size computed from the stored `axis_size` and `axis_index_groups`. Document
  the receiver-frame semantics of `output_offsets` and the `send_sizes == all_to_all(receive_sizes)` invariant
  (checkable only at runtime) in the operation rustdoc, linking the JAX doc page.
- [x] Runtime metadata contract (resolved): offsets and sizes must be nonnegative; every
  `[input_offset, input_offset + send_size)` region must lie within `N` and every receive region within `M`;
  received regions within one output must be disjoint (overlap is a precondition violation, not last-writer-wins);
  send regions may overlap (re-sending the same source slice is well defined). Enforcement is two-tier: *eager*
  interpretation always validates the concrete metadata and returns a precise error, computing `offset + size`
  with checked addition so host validation cannot overflow before the bounds comparison, and checking each
  receiver's bounds and disjointness against its *locally received* (collectively permuted) output offsets rather
  than the sender-local offset vector, since `output_offsets` live in the receiver's frame; *staged/XLA* execution
  treats metadata validity as a documented precondition (undefined results on violation, matching JAX). An
  assertion-staging checked mode is explicitly deferred until a real caller requests it.
- [x] Keep `reject_ragged_collective_inputs` in the batching path for this operation too: `RaggedAxis`-carrying
  operands are rejected exactly like the other collectives until Phase 6's explicit rule lands.
- [x] Effects / partial evaluation: like the existing collectives, this is a *contextual* operation, not an
  effectful one — Ryft's `Effect` classes (`programs/effects.rs`) carry no named-axis class, and the operation's
  correctness rides on staging-time `NamedAxes` validation plus the batching/shard_map contexts, exactly as
  `CollectiveOperation` does today. For linearization, "known" means primal-dependent in the partial-evaluation
  split, never compile-time constant: the metadata are nondifferentiable primal operands that are runtime/device
  values, and the Phase 5 transpose consumes them as residuals.
- [x] Interpretation: split by value concreteness. *Eager* interpretation over concrete arrays (the named axis
  bound as an eager batch axis, or effective group size 1) reads the concrete metadata and performs the segment
  copies directly — no staged dynamic shapes are involved, this exceeds JAX (which always errors outside a mapped
  context), and it provides device-free execution coverage mirroring
  `test_all_to_all_over_batched_axis_exchanges_chunks`. Tests compare against a plain-Rust reference computation of
  the exchange in the test module, using JAX's documented worked example.
- [x] *Staged* materialization over a batch-bound named axis (tracer-valued metadata) is gated: return an explicit
  `BatchingError::UnsupportedOperation` initially, because `DynamicSliceOperation` stores static slice sizes
  (`slicing.rs:1374`) so metadata-driven dynamic-length copies cannot be staged as dynamic-slice loops. Record the
  follow-up design in the rustdoc: iota-based index arithmetic + gather + `select` masking can express
  dynamic-length segment copies with existing operations at `O(group_size × M)` staged work.
- [x] Wire the operation through `arrays::operations`: new `ArrayOperation::RaggedAllToAll` and
  `ArrayIrOperation::RaggedAllToAll` variants, `MemberKindSignature` entry, dispatch/projection plumbing, and a
  user-facing `RaggedAllToAll` capability trait alongside `AllToAll` (documented in the `arrays/operations/mod.rs`
  collectives list).
- [x] Unit tests: type-inference success/failure matrix (dtype, rank, zero and mismatched lengths, divisibility,
  unbound axis), ragged rejection, eager metadata validation errors (including the overflow-safe bounds check),
  batched-axis exchange semantics against a hand-computed example (use JAX's documented worked example), the exact
  `UnsupportedOperation` diagnostic for the gated staged batch-bound path, and staged-program renderings for the
  mesh-staged (shard_map) operation.

### Phase 4 review

- Added the six-array operation contract, direct composite carrier, universe-neutral capability, eager participant
  exchange, grouped routing, runtime metadata validation, and explicit staged/differentiation gates without adding
  Phase 5 differentiation or Phase 7 StableHLO lowering. A private logical/physical representation marker keeps the
  batching-only participant-prefix form unforgeable through the public rank-1 metadata API while preserving it across
  carrier conversions and partial residualization.
- Added focused coverage for type and grouping validation, overflow-safe eager errors, JAX's documented exchange,
  reordered noncontiguous groups, multiple peer slices, trailing dimensions, public rank-2 rejection, raw tracing,
  named-axis staging, constant-metadata partial residualization, the staged-metadata gate, and `RaggedAxis` rejection.
- Verification passed: all-target checks for `ryft-core` and `ryft-xla`; the complete `ryft-core` library suite (1,662
  passed, 3 ignored); the complete `ryft-xla` library suite (540 passed, 5 ignored); `ryft-macros` tests (57 unit tests
  and 1 doctest passed, 7 doctests ignored); `ryft-macros-tests` operation and parameter suites (21 and 17 passed);
  documentation builds for `ryft-core` and `ryft-xla`; workspace formatting; and `git diff --check`.
- Four iterative independent audit cycles reviewed correctness, repository conventions, and simplicity. Every finding
  from the first three cycles was fixed and re-reviewed; all three auditors reported no findings in the fourth cycle.
  No changelog was modified, as requested.

## Phase 5: Differentiation rules

- [x] JVP: jointly linear in `(operand, output)`; the tangent result is `ragged_all_to_all` of the two tangents with
  the primal metadata. Symbolic-zero handling via `MaybeZero` (both tangents zero → zero result tangent; otherwise
  materialize both). Metadata operands are nondifferentiable integers.
- [x] Transpose (`transpose_with_respect_to` the two data operands only):
  - `operand` cotangent: stage dense tiled `all_to_all` of `output_offsets` and of `input_offsets` over the same
    axis, then `ragged_all_to_all(cotangent, zero_like(operand), permuted_output_offsets, receive_sizes,
    permuted_input_offsets, send_sizes)` with an internal additive-update mode, so cotangents from overlapping forward
    send regions sum rather than becoming overlapping overwrite destinations.
  - Forward `axis_index_groups` through the transpose, including into the metadata-permuting dense `all_to_all`
    calls. This is a deliberate correction over JAX, whose `_ragged_all_to_all_transpose` accepts groups on the
    primitive but drops them when permuting the offsets; document the divergence in the rule docs rather than
    claiming exact JAX parity.
  - `output` cotangent: mask the cotangent to zero on received regions, using JAX's `O(M)` construction directly
    now that Phase 1A provides `cumulative_sum`: allocate a length-`M + 1` zeros marker vector (computing `M + 1`
    with checked arithmetic), scatter `+1` at each received region's start (`permuted_output_offsets`) and `-1` at
    each region's end (`offset + receive_size`), using `ScatterReductionKind::Add` for both boundaries so zero-length
    regions cancel and adjacent regions combine deterministically at shared boundaries; the end index can equal `M`
    — the extra slot exists precisely so a region ending at the output boundary stays in bounds, where JAX's
    version silently relies on out-of-bounds-drop scatter semantics), take the `cumulative_sum` along that axis,
    slice the first `M` elements, broadcast the nonzero-inside-region mask over the trailing dimensions, and
    `select` zeros where the mask is set. This construction assumes received regions are disjoint, which the runtime
    metadata contract (Phase 4) makes an explicit precondition.
  - Note in the rule docs that this transpose stages additional collectives (it is not a local rewrite) and relies
    on the metadata operands being primal residuals — runtime values available to the transposed program, not
    compile-time constants. Gate a dynamic output leading dimension with an exact unsupported diagnostic until the
    static scatter/slice payloads used by the `M + 1` marker construction gain first-class dimension operands.
- [x] Tests: gradient checks through the batched-named-axis path (eager execution from Phase 4 makes
  `check_gradient`-style coverage possible without devices), involution-style transpose shape tests mirroring
  `test_shape_changing_collective_transposes_are_involutive`, and staged-program renderings pinning the transpose
  sequence (staged program shape is contract).

## Phase 6: Batching rule and `RaggedAxis` adapter

- [x] Dedicated vmap rule mirroring JAX's `_ragged_all_to_all_batched_collective`: when batching over an axis that
  is not the collective's named axis, move the data batch axes to the front and flatten into the packed leading
  axis, move metadata batch axes to the trailing position, rebase `input_offsets` by `iota * N` and
  `output_offsets` by `iota * M`, run one merged `ragged_all_to_all`, and split the result. Return an explicit
  unsupported error for `axis_index_groups` under this merged rule (matching JAX). Batching over the collective's
  *own* named axis is not this rule's case at all: in Ryft that is the batch-bound named-axis path already covered
  by Phase 4 (eager interpretation supported; staged materialization gated behind an explicit unsupported error).
  Gate a dynamic mapped extent or any dynamic data dimension with exact unsupported diagnostics: offset rebasing
  stages `N` and `M` as scalar constants, while the shared homogeneous/composite reshape interface cannot yet recover
  dynamic trailing extents uniformly from the data values.
- [x] Gated `RaggedAxis` adapter: do not implement until an explicit partition/routing descriptor is designed. A
  `RaggedAxis` carries one logical extent per packed batch item, while `ragged_all_to_all` needs per-source,
  per-destination segment sizes plus both offset vectors — extents alone underdetermine the routing, so any adapter
  must take the partition as an explicit input rather than inferring it. Also document the frame mismatch: the
  operation's raggedness is per participant chunk, `RaggedAxis` raggedness is per batch item, and outside a
  batching transform there is no carrier to attach metadata to.

### Phases 5 and 6 review

- Implemented a jointly linear JVP and a transpose for both data operands. The operand adjoint permutes metadata with
  grouped dense exchanges and uses a private additive ragged exchange so cotangents from resent source slices sum;
  the output-seed adjoint uses an additive boundary marker, cumulative sum, and broadcast mask that handles empty,
  adjacent, and boundary-ending receive regions. Dynamic marker extents and unavailable residual metadata fail with
  exact unsupported diagnostics.
- Implemented unrelated-axis batching for logical and physical carriers by aligning axes, merging the mapped extent
  into packed data and metadata dimensions, widening and rebasing offsets, executing one exchange, and restoring the
  output shape and axis. It rejects groups only for this merged path and gates dynamic mapped or data extents. The
  own named-axis path continues through the eager physical participant exchange, including groups; `RaggedAxis`
  adaptation remains explicitly gated because extents alone do not define routing.
- Added exact transpose renderings and cotangents, finite-difference gradients for both data operands, symbolic-zero
  JVPs, grouped logical and physical routing, sharding preservation, dynamic-dimension diagnostics, empty-batch and
  zero-byte overflow cases, provenance attribution, and both real nested named-batching orders. The focused ragged
  suite passes 20 tests.
- Verification passed: `cargo test -p ryft-core --lib` (1,678 passed, 3 ignored), `cargo test -p ryft-xla --lib`
  (540 passed, 5 ignored), all-target checks for both crates, the `ryft-core` documentation build, workspace
  formatting, and `git diff --check`.
- Independent correctness, convention, and simplicity auditors iteratively reviewed the full diff. Findings covering
  overlapping-send adjoints, dynamic and empty-edge gates, checked byte arithmetic, transform composition,
  provenance assertions, plan accuracy, conditional normalization, redundant bookkeeping, and test fixtures were
  fixed and re-reviewed; the final cycle reported no findings. No changelog was modified, as requested.

## Phase 7: XLA lowering and execution

- [x] `ryft-xla` lowering: match the new `ArrayOperation::RaggedAllToAll` in `experimental/lowering.rs` and emit
  `stable_hlo::custom_call` with target `ragged_all_to_all`, `CustomCallApiVersion::TypedFfi`, result type equal to
  the output operand's type, and a dictionary `backend_config` holding `replica_groups` (reuse the replica-group
  computation from `lower_all_to_all_to_mlir`, validating equally sized groups) plus the collective `channel_id`
  under SPMD/shard_map lowering, following JAX's `_ragged_all_to_all_lowering`. The public overwrite mode maps to the
  custom call directly; the transpose-internal additive mode must lower to an accumulation-preserving decomposition
  rather than silently reusing overwrite semantics.
- [x] shard_map integration: thread the operation through `experimental/shard_map.rs` like the other collectives
  (manual-axis resolution, sharding/type propagation via the tracked output type).
- [x] Backend gating: surface an explicit `unsupported` error where the backend cannot execute the custom call
  (CPU), at the earliest layer that knows the backend; do not let it fail as an opaque runtime custom-call miss.
- [x] Tests, tiered by what each environment can run: the eager-interpretation semantic coverage (Phase 4) and
  lowering snapshot tests (module `to_string()` comparison per the MLIR test conventions) run everywhere; device
  execution tests run only where the backend advertises support (JAX documents `ragged_all_to_all` as an
  accelerator collective, not a CPU one), gated like the existing multi-device collective coverage; include the JAX
  documentation worked example as the reference output in both tiers.
- [x] Changelog updates are excluded from this effort by request.

### Phase 7 review

- Added direct typed-FFI overwrite lowering with replica groups and channel ids, plus an accumulation-preserving
  transpose lowering that assigns every source transfer a disjoint expanded-output lane before reducing and adding
  the output cotangent seed. The same helper serves homogeneous and composite lowering; the direct composite carrier
  consumes exactly its six array operands without inventing dimension operands.
- Added early target-aware CPU rejection and shard-map coverage for the exact overwrite module, a real grouped
  differentiated program (including noncontiguous group-local source lanes), the everywhere JAX worked example, and
  CUDA-gated execution of both the documented forward exchange and overlapping-send cotangent accumulation. The
  transpose-only additive constructor remains private; the backend sees only a hidden read-only semantic query.
- Verification passed: 20 focused `ryft-core` ragged-all-to-all tests, 4 focused default `ryft-xla` tests,
  `cargo check -p ryft-xla --lib`, CUDA-13 all-test type-checking with the repository native archive and a non-loaded
  placeholder plugin path, workspace formatting, and `git diff --check`. CUDA execution remains CI/hardware-gated on
  this macOS arm64 host. An independent strict audit's correctness, convention, gating, API-surface, and redundancy
  findings were fixed and re-reviewed to no remaining source findings. No changelog was modified.

## Phase 8: Ragged batching contract for `CustomCallOperation`

JAX status, for calibration: JAX does not support ragged batching of custom calls in any form. Its supported story
(`jax/_src/ffi.py`) is dense `vmap_method` selection plus users manually passing packed data, offsets, and sizes as
ordinary explicit operands; the experimental ragged-vmap ("jumble") prototype covers a few array primitives, not
FFI calls. Ryft already matches that supported surface today (`Sequential`/`BroadcastAll` plus pre-binding
rejection), so this phase deliberately *exceeds* JAX by formalizing the manual JAX pattern as a declared,
validated calling convention. The scope stays a calling convention, not a constraint language, and the existing
rejection (`custom_call.rs:759`, `custom_call.rs:903`) remains the default for any call without a contract.
(Separate dense-parity aside, not in scope: JAX's `expand_dims` `vmap_method` variant has no Ryft equivalent;
demand-gated.)

Core ABI principle: **transforms never change a kernel's signature**. The foreign kernel is opaque typed FFI — its
operand count, ordering, layouts, and alias indices are a fixed contract (the existing trailing dimension operands
are deliberately *excluded* from that ABI, `custom_call.rs:196`), so the batching rule must not append operands. As
in JAX's manual pattern, the extents are *already* explicit ordinary operands of the source custom call; the
contract only names which existing operand bounds which packed axis, and a hypothetical transformed ABI would
require an explicitly different target declaration, not a silently mutated one.

- [x] Contract data model: an optional declaration on `CustomCallOperation` (builder-style
  `with_ragged_contract(...)`) stating:
  - *input bindings*: for each ragged packed operand axis, the operand index + axis it lives on and the index of
    the **existing** operand carrying its extent. In the unbatched call that extent operand is a scalar (one item's
    extent); `BroadcastAll` batching turns it into the batch-prefixed extents vector and `Sequential` slices the
    mapped vector back to a scalar per invocation, so the kernel sees exactly the arity it declared;
  - *output bindings*: for each declared output, whether it preserves a named input binding's ragged axis (same
    dimension variable, relocated axis), consumes it (dense output), or takes fresh extents from a declared
    **existing integer scalar extents output** (stacked by `Sequential`, batch-prefixed by `BroadcastAll`; it
    remains an ordinary visible kernel result — no result-hiding). A fresh-extents binding stores its
    `DimensionVariable` in the contract so the identity is stable across replays rather than minted
    opportunistically, and the bound packed output axis must have a finite static physical bound compatible with
    that variable; runtime extents lying within the variable and physical bounds is a documented precondition
    (checked in eager execution);
  - a *padding-independence promise*: the kernel's live output elements do not depend on padded input elements.
  Validate the declaration structurally at construction/inference time (indices in range, extent operands/outputs
  scalar integer in the unbatched form, axis bounds, no double-binding, dimension-variable/bound compatibility),
  surfacing `TypeError`s with precise messages.
- [x] Alias cross-checks: validate the contract against `CustomCallInputOutputAlias` declarations — a *preserved*
  output may alias its input only when the packed types, physical axes, dimension identity, and extent binding all
  agree; *consumed* and *fresh-extents* outputs cannot inherit an alias unchanged. Extend the validation matrix
  accordingly, not just duplicate-binding checks.
- [x] Batching-rule discharge: when a `RaggedAxis`-carrying batch reaches a custom call *with* a contract, stop
  rejecting: verify that each bound extent operand's batched value is the *same value* as the packed operand's
  `RaggedAxis::extents` (identity comparison at trace time — the user threads one extents value to both places,
  exactly the manual JAX pattern), stage the call through the existing `CustomCallBatching` machinery without
  adding or reordering operands, and attach result `RaggedAxis` metadata per the output bindings. Ragged axes on
  operands or axes not covered by the contract, and extent operands that do not match the `RaggedAxis` extents,
  keep today's rejection with precise diagnostics.
- [x] Contract propagation: the declaration must survive every existing custom-call metadata-copy seam —
  `ArrayType` ↔ `ArrayIrType` conversion, output-identity renaming (`renamed`, `custom_call.rs:375`), operation
  rendering, and batch-prefixed output reconstruction — with tests pinning each seam.
- [x] Scope boundaries, stated in rustdoc: one ragged axis per operand; **one batching level** — nested ragged
  batching is explicitly unsupported with its own diagnostic (supporting it later requires recording the
  extent-axis mapping à la `RaggedAxis::extent_axes`, not just an operand index); differentiation of ragged custom
  calls follows the existing custom-call AD story unchanged (the contract adds no AD semantics); the padded
  buffer's garbage elements remain garbage-by-contract downstream, with the attached output `RaggedAxis` making
  masked reductions and dimension-size rules compose as usual.
- [x] Tests: the no-contract rejection tests stay green unchanged; contract validation matrix (structural, alias
  interactions, extents-identity mismatch, nested-batching rejection); staged-program renderings for ragged
  `Sequential` and `BroadcastAll` discharge showing the unchanged kernel arity; an eager end-to-end test with a
  test kernel asserting the extents arrive as declared and the output `RaggedAxis` composes with a downstream
  masked reduction.

### Phase 8 review

- Added a declarative ragged calling convention to `CustomCallOperation`, including named packed-input bindings,
  preserved/consumed/fresh output bindings, padding independence, structural and alias validation, stable dimension
  renaming, universe conversion, rendering, and batch-prefix propagation without changing kernel arity.
- Both homogeneous and mixed batching paths now discharge covered one-level `RaggedAxis` metadata only when the
  declared extent operand is the exact staged value, reject uncovered/mismatched/nested cases, and attach or consume
  result metadata according to the positional output contract. Fresh output extents also work without an active
  ragged input.
- Focused verification passed: 25 `ryft-core` custom-call tests; the CPU XLA FFI end-to-end test, including concrete
  extent-bound validation and a downstream masked sum; all-target checks for `ryft-core` and `ryft-xla`; workspace
  formatting; and `git diff --check`. No changelog was modified.

## Phase 9 (gated): ragged-aware rules for individual collectives

Demand-driven, one operation at a time, never a generic "collectives preserve raggedness" rule.

- [x] `parallel_sum`/`parallel_max`: mask padding with the reduction identity via Phase 1A's generalized
  identity-masking capability (the existing `mask_reduction_input` zero-masks and rejects every kind but `Sum`,
  `batching.rs:2933`); result extents are the elementwise `parallel_max` of the participating extents (costs one
  extra collective on the extents; document the partial-participation semantics).
- [x] `parallel_mean` stays explicitly unsupported for ragged inputs unless its denominator is defined first
  (participant count, present-value count, or logical element count are all defensible and give different results).
- [x] `parallel_permute` / untiled `all_gather`: co-move the packed value and its extents, remapping the output
  `RaggedAxis` (`all_gather` extents become per-(participant, item) via multi-axis `extent_axes`). A tiled gather is
  explicitly gated because fusing participant and concatenation axes can turn participant-specific live prefixes
  into a non-prefix layout that one `RaggedAxis` cannot represent.
- [x] `all_to_all` over ragged operands: requires an explicit routing descriptor before any lowering to
  `ragged_all_to_all` — an ordinary `all_to_all` call plus one logical extent per item does not determine the
  per-destination partition, so this is a new API surface, not an automatic rewrite.

### Phase 9 review

- `parallel_sum` and `parallel_max` now mask each ragged packed axis with the correct identity before reducing the
  participant axis, preserve the surviving ragged dimensions, and reduce participant-varying extent arrays with
  elementwise maximum. `parallel_mean` reports a precise denominator-ambiguity diagnostic before doing any work.
- `parallel_permute` applies the same source/target routing to participant-varying extent arrays, including zero
  extents for untargeted participants. Untiled `all_gather` relocates the participant mapping into the materialized
  output axis; tiled gathering fails explicitly where fusion would make the metadata inexpressible. Ordinary
  `all_to_all` directs callers to the explicit offsets-and-sizes `ragged_all_to_all` contract.
- Focused verification passed: 11 `parallel_reduce`, 6 `parallel_permute`, 10 `all_gather`, and 6 `all_to_all` tests;
  the complete `ryft-core` library suite passed 1,691 tests with 3 ignored; `cargo check -p ryft-core --lib`;
  workspace formatting; and `git diff --check`. No changelog was modified.

## Phase 10: `ragged_dot_general`

Grouped ("ragged") matrix multiplication, mirroring `jax.lax.ragged_dot_general` (pinned `jax/_src/lax/lax.py`,
commit `a33ed61`). Depends only on Phase 1A's `cumulative_sum`; independent of the ragged-collective phases.
This raggedness is *grouping metadata carried as an explicit operand* (`group_sizes`), exactly the explicit-operand
doctrine of the rest of this plan — it is unrelated to the batching-time `RaggedAxis` mechanism, and the two must
not be conflated.

Reference semantics (verified against the pinned source):

- Three operands: `lhs`, `rhs`, and integer `group_sizes`; a `RaggedDotDimensionNumbers` payload wrapping ordinary
  dot dimension numbers plus exactly one `lhs` ragged dimension and the `rhs` group dimensions. Three modes by
  where the ragged dimension falls: *non-contracting* (`[b...,m...,k...] × [g,b...,k...,n...] → [b...,m...,n...]`,
  exactly one `rhs` group dimension whose extent is `g`), *contracting* (`→ [g,b...,m...,n...]`, zero group
  dimensions), and *batch* (`→ [b...,m...,n...]`, zero group dimensions). `group_sizes` is `[g]` or the
  prefix-shaped `[b...,x...,g]`; output shape and element type otherwise defer to `dot_general`'s rules, with grouped
  expansion modes rejecting `f8e8m0fnu` because their zero-group and uncovered-position semantics require zero.
- JAX's **default lowering on every platform is a masked-dense decomposition**: expand a leading `g` axis, compute
  group start/end offsets as `cumsum(group_sizes)`, mask with `start <= iota < end` selecting against zero, then
  run ordinary dense `dot_general`s. The dedicated backend instruction is opt-in behind
  `jax_ragged_dot_use_ragged_dot_instruction`. Rows not covered by any group therefore produce zeros — pin that as
  the defined semantic.
- Differentiation is dot-shaped: jointly linear in `(lhs, rhs)` (tangent = `rd(dlhs, rhs) + rd(lhs, drhs)`;
  `group_sizes` nondifferentiable). The transpose is two `ragged_dot_general`s with rearranged dimension numbers
  plus a plain axis transpose, implemented **only for the non-contracting mode** — JAX rejects transposition of
  the contracting and batch modes, and so do we.
- `group_offset` exists in JAX's signature but is signature-only at the pinned commit: every consumer of the
  primitive rejects it with `NotImplementedError` — the JVP (`lax.py:6139`), the transpose (`:6195`), the
  reference impl (`:6356`), and the lowering (`:6471`, before either strategy branches) — and `chlo.ragged_dot`
  carries no such operand. Re-checked against JAX main on 2026-08-25: still signature-only — the same four
  rejection sites persist, the new sharding rule threads the parameter without reading it, and even the Pallas GPU
  lowering entry point rejects it (its internal `group_offsets` are the derived `cumsum(group_sizes) -
  group_sizes` start positions, not this parameter). Ryft omitting the parameter therefore IS full JAX parity:
  there is no supported behavior to mirror. Revisit only if JAX gives it semantics.
- vmap: JAX reuses its dot batch rule via unpack hooks and supports only leading-dimension batching
  (`'ragged_dot vmap over any dim but 0 - NYI'`); the ragged-contracting mode shifts the result batch axis past
  the prepended `g` axis.

Design constraints (no new redundancy):

- Extend `operations/dot/` in its existing per-concern layout — no new module, no parallel vocabulary:
  `RaggedDotDimensionNumbers` in `dimensions.rs` **wrapping** the existing `DotDimensionNumbers` (as JAX does),
  `RaggedDotOperation` in `operation.rs` beside `DotOperation` (payload: the ragged dimension numbers only,
  mirroring `DotOperation`'s dimension-numbers-only payload and its documented element-type contract),
  inference in `inference.rs` reusing `dot_abstract` for the dense part, batching in `batching.rs` reusing the
  existing dot batch rule via unpack hooks (JAX's own structure), differentiation in `differentiation.rs`
  mirroring `DotOperation`'s joint-linear JVP discipline, adjoint-dimension helpers beside
  `adjoint_dimensions_for_left_dot`/`adjoint_dimensions_for_right_dot` in `dimensions.rs` (JAX's
  `grad_x_dims`/`grad_y_dims`), tests in `tests.rs`.
- The group mask construction (`cumulative_sum` of `group_sizes` → start/end offsets → interval-membership iota
  mask) reuses the Phase 1A operation and the established iota-compare-select staging style; it is per-operation
  logic and lives with the dot module, not in the generic batching machinery.
- Eager interpretation is extent-exact, not masked: with concrete `group_sizes`, the `Array` kernel loops the
  groups and reuses the existing dense dot kernel per group slice (consistent with this plan's eager-vs-staged
  doctrine; the staged/XLA world masks because extents are symbolic there).

Checkable items:

- [x] `dimensions.rs`: `RaggedDotDimensionNumbers` wrapping `DotDimensionNumbers` + the mode classification
  (non-contracting / contracting / batch from where the single `lhs` ragged dimension falls), with validation
  mirroring JAX's shape rule (exactly one ragged dimension; group-dimension count per mode; `group_sizes` rank-1
  `[g]` or prefix-shaped `[b...,x...,g]`; precise `TypeError`s in Ryft's message style). Adjoint-dimension helpers
  for the non-contracting transpose beside the existing dot adjoint helpers.
- [x] `operation.rs`: `RaggedDotOperation` (name `ragged_dot_general`), Display/render, `Operation` inference via
  `inference.rs` (dense shape from `dot_abstract` machinery, leading `g` prepended in the contracting mode,
  element-type rules otherwise deferred to dot's, with the grouped-expansion zero restriction above),
  `InterpretableOperation` on a `RaggedDot` capability,
  `PartiallyEvaluatableOperation` default. Capability trait `RaggedDot` with
  `ragged_dot_general(rhs, group_sizes, dimension_numbers)` and a `ragged_dot(rhs, group_sizes)` convenience for
  JAX's basic case, both fallible.
- [x] Eager `Array` kernel: extent-exact grouped loop reusing the existing dense dot kernel per group slice;
  rows/positions not covered by any group are zeros (the pinned semantic); groups of size zero contribute nothing;
  validation via the shared abstract rule.
- [x] Differentiation: joint-linear JVP with `MaybeZero` discipline mirroring `DotOperation`'s; transpose for the
  non-contracting mode via the two rearranged `ragged_dot_general`s + axis transpose (staged-program rendering
  pinned), explicit precise rejections for the contracting and batch modes (JAX parity).
- [x] Batching: reuse the dot batch rule with unpack hooks; leading-dimension mapping only, with JAX's exact
  restriction surfaced as a precise unsupported error otherwise; the contracting mode's result batch axis shifts
  past the prepended `g`. `RaggedAxis`-carrying operands are rejected pre-binding (the collectives idiom) — group
  raggedness and batching raggedness stay separate concepts.
- [x] Wiring: `ArrayOperation::RaggedDot` variant, `ArrayOperations` bundle (both lists), facade/root re-exports,
  `ryft-macros`/`ryft-macros-tests` verification.
- [x] `ryft-mlir`: wrap `chlo.ragged_dot` following the module's existing `chlo::erf` pattern (`mlir_op!` +
  typed constructor in `dialects/chlo/operations.rs`), plus a `RaggedDotDimensionNumbers` CHLO attribute wrapper
  (batching/contracting/ragged/group dimension lists) modeled on the StableHLO dot-dimension-numbers attribute
  wrapper, accepting the same precision-config attribute the dot lowering already uses. Dialect tests per the
  MLIR conventions (attribute construction/accessor/equality/display/casting; operation constructor + typed
  accessors + verified module rendering).
- [x] XLA lowering, both strategies mirroring JAX: the **default** is `lower_ragged_dot_general_to_mlir` emitting
  the masked-dense decomposition — `cumulative_sum` of `group_sizes` via the existing Phase 1A `reduce_window`
  emitter, start/end broadcast, the interval mask, `select` against zero, then the existing dense dot lowering —
  reusing those emitters rather than re-building them. The **instruction** strategy emits one `chlo.ragged_dot`
  with the full dimension-numbers attribute (all three modes; element/accumulation-type handling otherwise follows
  the existing dot lowering conventions). Strategy selection is an XLA-owned per-compilation option, defaulting to
  the decomposition — JAX's own default on every platform, because backend support for the instruction is uneven;
  JAX's TPU-only rhs-transpose quirk stays out of scope with a comment. `domains.rs` padding-discipline classification
  with the same evidence standard as the other phases; `ops.rs` conversion entry.
- [x] Execution evidence per the domains doctrine: CPU execution tests run against the decomposition strategy;
  the instruction strategy gets lowering snapshot tests everywhere and execution tests only where a backend
  demonstrably runs `chlo.ragged_dot` (verify whether the CPU pipeline legalizes it — if it does, add the CPU
  execution test and cross-strategy value parity; if not, gate execution on the accelerator suites and pin the
  exact unsupported error).
- [x] Tests: inference matrix across all three modes and both `group_sizes` shapes with exact errors; eager values
  against hand-computed grouped products (including a zero-size group and an uncovered-rows case); JVP
  finite-difference checks and the pinned transpose rendering; mode-2/3 transpose rejections; batching incl. the
  contracting-mode axis shift and the `RaggedAxis` rejection; lowering snapshot pinning the mask + dense-dot
  expansion; eager/XLA parity and a CPU jit execution test (compile, run, exact grouped-matmul values). Changelog
  updates remain excluded from this effort by request.

### Phase 10 review

- Added the grouped-dot dimension contract, inference, semantic operation, exact eager grouped-slice kernel,
  leading-axis batching rule, joint-linear JVP, and non-contracting transpose. Differential staging converts widened
  tangent and cotangent representations consistently with dense dot; contracting and batch transposes remain
  explicitly unsupported.
- Added the CHLO dimension-number attribute and typed `chlo.ragged_dot` wrapper, including ownership, equality,
  display/debug, casting, accessor, verification, and exact module-rendering coverage. The XLA path defaults to the
  portable cumulative-mask decomposition and exposes the dedicated instruction through an XLA compilation strategy.
- Verified all three modes and prefix-shaped metadata in core, exact eager values and finite-difference JVP behavior,
  contracting-mode batch-axis placement, and the element-type boundary: batch mode ignores group values and supports
  `f8e8m0fnu`, while grouped expansion rejects it because the required zero does not exist. Exact lowering snapshots
  cover every mode and both lowering strategies, including native complex-zero construction for the decomposition.
  CPU eager/XLA parity covers the decomposition, and the instruction pins the CPU plugin's exact unsupported
  diagnostic. Focused suites pass, and no changelog was modified.

## Final verification

Run after the last implemented phase, in addition to each phase's own scoped checks:

- [x] Full test suites: `cargo test -p ryft-core --lib`, `cargo test -p ryft-xla --lib`,
  `cargo test -p ryft-macros`, `cargo test -p ryft-macros-tests` (300s timeouts per command).
- [x] `cargo check -p ryft-core --all-targets` and `cargo check -p ryft-xla --all-targets` with zero warnings.
- [x] Rustdoc builds cleanly for the touched crates; new items follow the documentation conventions.
- [x] `cargo fmt --all -- --check`, `git diff --check` (no whitespace errors), and no source or plan lines over
  120 columns.
- [x] Targeted search confirming no stale renamed identifiers remain (the Phase 2 list, plus any identifiers
  renamed mid-implementation).
- [x] Changelog updates remain excluded from this effort by request.
- [x] Final self-review of the complete diff for scope and simplicity, summarized in the Review section below.

## Review

### Phase 3: Split `operations::collectives` into per-operation-kind submodules

Move-only split of the ~5,100-line flat `operations/collectives.rs` into a directory module. `mod.rs` keeps the
module documentation, the shared vocabulary (`CollectiveMode`, `CollectiveOptions`), named-axis resolution, the
shape-changing collective engine and extent helpers, the `CollectiveBatchingPolicy` trait with both policy impls, the
three matching-axis batching kernels (which the later collectives/XLA audit round moved on into the operation files
that own their payloads, where they are private), and both `macro_rules!` macros (each followed by `pub(super) use` so
the submodules invoke them through `super::`). `parallel_reduce.rs`, `all_gather.rs`, `parallel_sum_scatter.rs`,
`parallel_permute.rs`, and `all_to_all.rs` each own exactly one operation kind, and the `pub use` facade preserves
every previous `crate::operations::collectives::*` path, so no consumer inside `ryft-core` or `ryft-xla` needed an
edit. Deviations from the plan sketch: `stage_collective` moved into `parallel_reduce.rs` because its only callers
are that operation's differentiation rules, and `collective_extent_constant` became `pub(super)` because
`all_gather` and `all_to_all` stage exact extent constants with it. The macro-generated payload fields stayed private
and were never widened, in this phase or any later one; what the split moved is where the helpers that read them live,
and each such helper ended up in the file that owns its payload. The genuinely shared helpers left in `mod.rs` are the
20 `pub(super) fn` items it declares (19 of them plus one test-module helper), joined by the two `pub(super) use`
macro re-exports. No other visibility, naming, or behavior changed. Verified with `cargo check -p ryft-core --tests`
and `cargo check -p ryft-xla --tests` (both clean, no warnings), `cargo test -p ryft-core --lib collectives`
(41 passed), `cargo test -p ryft-core --lib` (1,575 passed), and
`cargo test -p ryft-xla --lib` (531 passed).

### Phase 1A (core): `operations::cumulative` with `cumulative_sum`, and generalized ragged identity masking

`operations/cumulative/mod.rs` owns the family's shared machinery — `cumulative_abstract` (output type equals the
input type; `axis` in bounds; scanned dimension static and unsharded, with the physicalize-at-the-bound extension
documented as future work), `lift_cumulative_axis`, and the row-major `cumulative_evaluate` prefix scan.
`cumulative_sum.rs` owns `CumulativeSumOperation { axis, reverse }` with its type inference (numeric or structural
zero element types), rendering, interpretation, default partial evaluation, batching, linear JVP, the
`reverse`-flipping transpose, and the `CumulativeSum` capability with its staging blanket. The eager kernel lives in
`arrays/operations/cumulative.rs` and shares the abstract rule, so a directly invoked capability rejects exactly what
a staged program rejects. Wiring: `ArrayOperation::CumulativeSum`, the `ArrayOperations` bundle and its blanket
predicate, the `operations` and crate-root facades. `ArrayIrOperation` has no reduction-style variant, and the
composite `MemberDifferentiableOperation` dispatch needed no arm: the operation is shape-preserving, so
`replicated_elementwise_duals` returns `None` and the default fall-through reaches the projected rule.

Ragged masking (`arrays/batching.rs`): new `RaggedMaskIdentity { Zero, One, Lowest, Highest }` and a third
`RaggedArrayBatchingPolicy` hook, `mask_identity_input`, whose discipline is named by the value it writes rather than
by what becomes of the masked axis. The dynamic policy's `zero_ragged_padding` became `mask_ragged_padding`,
parameterized by the identity and entered under nested `ryft::batching::ragged_identity_mask` provenance scopes;
`Zero` still stages `zero_like`, while the other identities stage a rank-zero `ConstantOperation<Array>` (extrema
reusing the element-level reduction identities through a new `pub(crate)` facade re-export of `ElementExtremum`)
broadcast over the packed shape. The failure to resolve the packed physical extent, previously unwrapped, is now
reported as a `BatchingError::InvalidBatchMetadata` naming the ragged axis.
Two deviations from the plan sketch: `zero_ragged_padding` was dropped rather than kept as a thin wrapper, because
its two call sites now read as the identity's `Zero` case and the wrapper only duplicated a 25-line `where` clause
(behavior and reachable bounds are unchanged); and `cumulative_evaluate`'s combiner is fallible, because the
reference backend's element-level arithmetic contracts are.

Verified with `cargo check -p ryft-core --all-targets` (clean, no warnings), `cargo test -p ryft-core --lib`
(1,592 passed, 0 failed; 17 new tests), and `cargo test -p ryft-macros` (clean). `cargo test -p ryft-macros-tests`
cannot run until the XLA lowering lands: it depends on the `ryft` umbrella crate, and `ryft-xla` does not yet cover
the new enum variant (`experimental/domains.rs` and `experimental/lowering.rs` each have one non-exhaustive match).

XLA lowering (`ryft-xla`): `lower_cumulative_sum_to_mlir` emits the full-prefix `stablehlo.reduce_window` (window
`n` on the scanned axis, padding `(n - 1, 0)` forward / `(0, n - 1)` reverse, the shared `Sum` identity constant and
`add` reducer body, optional strides and dilations left at their defaults, and an operand pass-through at `n == 0`);
`array_data_dependent_padding_discipline` classifies the scan as `Propagated` on its own arm because a static
scanned axis is a type-inference guarantee; `ops.rs` gained the conversion. Verified with `cargo check -p ryft-xla
--all-targets` (clean), `cargo test -p ryft-xla --lib` (533 passed, including forward/reverse lowering snapshots,
reference-backend eager parity, and a CPU PJRT test covering both directions and the gradient through the compiled
scan), and `cargo test -p ryft-macros-tests` (17 passed, now unblocked). Changelog updates were excluded from this
effort by request.

### Phase 1B (core): `log1p`, `log_add_exp`, and `log_sum_exp`

`math/log1p.rs` and `math/log_add_exp.rs` are ordinary `define_elementwise_operation!` primitives restricted to
`@float @real`; `log1p`'s JVP is the generated `dx / (1 + x)` elementwise rule, while `log_add_exp` needs a
hand-written `impl_differentiable_operation!` because its softmax weights pass every operand and the primal output
through JAX's `_replace_inf` guard (`select(x == +inf, 0, x)`, staged as `fill`/`compare`/`zero_like`/`select`)
before the subtraction. The five pinned exceptional tangents are encoded as exact assertions, alongside the pinned
primal specials. `math/log_sum_exp.rs` is a hand-written axis reduction (`LogSumExpOperation { axes }`) with its own
`log_sum_exp_abstract` mirroring `reduce_abstract`'s validation idiom and reusing `reduce_sharding`; its batching
rule mirrors `ReduceOperation::batch` but masks ragged padding through the Phase 1A `mask_identity_input` hook with
`RaggedMaskIdentity::Lowest` (`exp(-inf) = 0` is the inner sum's identity) and reports the consumed ragged
dimensions as evidence, and its JVP is the softmax-weighted tangent reduction. All three are non-transposable.

Eager kernels live in `arrays/operations/math.rs`: `log1p` and `log_add_exp` extend `ElementRealFloatMath` (the
binary macro gained a `@real_float` branch whose `check_types!(@float @real, ...)` excludes complex, since the
existing branch admits it), and `log_sum_exp` is a hand-written `log_sum_exp_elements` two-pass kernel over the
input's physical addressing that holds every intermediate in the element's own encoding. Wiring: three
`ArrayOperation` variants, the `ArrayOperations` bundle and its blanket predicate, `LogAddExp` added to
`replicated_elementwise_duals`' broadcasting list, and the `operations::math` and crate-root facades.
`MemberDifferentiableOperation` needed no arm for any of the three: `log1p` is unary, `log_add_exp` is covered by
the replication list, and `log_sum_exp` reaches the projected rule through the default fall-through (correct for
the static shapes its `broadcast`-back JVP supports). `reduce_sharding` and `output_to_input_axis_map` became
`pub(crate)` instead of being duplicated.

Verified with `cargo check -p ryft-core --all-targets` (clean, no warnings), `cargo test -p ryft-core --lib`
(1,617 passed, 0 failed; 25 new tests), `cargo test -p ryft-macros` (clean), and `cargo doc -p ryft-core --no-deps`
(no new warning kinds; the two new elementwise ops inherit the pre-existing unresolved-`Operation`-link warning that
`define_elementwise_operation!` produces for every member of the family). `ryft-xla` is expected red until its
lowering lands: `experimental/domains.rs:4249` and `experimental/lowering.rs:2940` each have one non-exhaustive
match over `ArrayOperation`. XLA lowering and the cumulative members remain outstanding; changelog updates were
excluded from this effort by request.

### Phase 1B (core): `cumulative_product`, `cumulative_max`, `cumulative_min`, and `cumulative_log_sum_exp`

Four new member modules under `operations/cumulative/`, each a structural copy of `cumulative_sum.rs` differing only
in its combining operator, ragged identity (`One` / `Lowest` / `Highest` / `Lowest`), element data-type domain
(numeric-or-structural-zero / real numeric / real numeric / real floating-point), and differentiation. `mod.rs`
gained the shared staged decomposition `associative_scan` — JAX's log-doubling `lax.associative_scan` with
`_interleave` built on interior-padded `PadOperation` — plus `jvp_through_associative_scan`, which every nonlinear
member's JVP rule delegates to. All four are `impl_non_transposable_operation!`; reverse mode flows through the
decomposition's own primitives.

Two deviations from the plan sketch, both forced and both verified:

  - **No array-reversal primitive exists in `ryft-core`**, so a `reverse` scan cannot be spelled as
    `rev -> forward scan -> rev` the way JAX's `associative_scan` does. The recursion is instead *mirrored* around
    the end of the axis: the pairing offset becomes `extent % 2`, the complementary half is built from the aligned
    results that follow it rather than precede it, and the leftover element is concatenated at the tail instead of
    the head. The two directions are checked against `cumulative_evaluate` for every extent in `0..=9` with both a
    commutative combiner (summation) and a non-commutative one (left projection), which pins operand order.
  - **The nested `DifferentiationContext<C>` mechanism is unusable from an operation rule.** Instantiating it
    requires `DifferentiationContext<C>: Context`, whose bound set includes `C::Operation: DifferentiableOperation<C>`;
    since the rule's own operation is a member of `C::Operation`, that bound is self-referential and the enum's
    derive-generated impl overflows the trait solver (`E0275: overflow evaluating the requirement
    CumulativeSumOperation: DifferentiableOperation<PartialEvaluationContext<EagerContext<Array, ArrayOperation>>>`,
    reproduced on a throwaway impl before the design was chosen). The rule instead traces the decomposition into its
    own `TracingContext<C::Constant, C::Operation>` program and asks the instruction-scoped
    `DifferentiationDriver::jvp_program` to differentiate it — the driver hook documented for exactly this (“the
    entry region of a program rebuilt by an operation rule”) — then replays the fused
    `[primal, tangent] -> [primal, tangent]` program in the rule's own context. Semantically this is JAX's
    `api.jvp(partial(associative_scan, combine_fn, ...), primals, tangents)`: every primitive contributes its own
    forward-mode rule and no bespoke gradient formula (cumprod-via-division, select-based cummax) is introduced. A
    structural-zero operand tangent short-circuits to the primitive, since a JVP is linear in its tangent.

`associative_scan` enters nested `ryft` -> `differentiation` -> `associative_scan` provenance scopes (diagnostic
only; a no-op eagerly), pinned by a `WithProvenance` rendering assertion, and the full staged JVP program of a
length-four `cumulative_product` is pinned as an `indoc!` contract. Eager kernels extend
`arrays/operations/cumulative.rs` (`scan_product`, `scan_extremum`, `scan_log_sum_exp`) reusing `cumulative_evaluate`
with `ElementMul`, `ElementExtremum`, and `ElementRealFloatMath::log_add_exp`; `ElementRealFloatMath` became
`pub(crate)` and `padding::dependency_scalar_type` became `pub(crate)` so the interleave stages a correctly typed
padding scalar instead of duplicating that construction. Wiring: four `ArrayOperation` variants beside
`CumulativeSum`, the `ArrayOperations` bundle and its blanket predicate, and the `operations::cumulative` and
crate-root facades. As for `cumulative_sum`, `MemberDifferentiableOperation` needed no arm and the members are
deliberately absent from `replicated_elementwise_duals` (they are unary and shape-preserving, not broadcasting);
the decomposition slices at staging-time positions, so it requires a fully static operand shape and reports a
precise error otherwise.

Verified with `cargo check -p ryft-core --all-targets` (clean, no warnings), `cargo test -p ryft-core --lib`
(1,655 passed, 0 failed; 38 new tests), `cargo test -p ryft-macros` (clean), and `cargo doc -p ryft-core --no-deps`
(no new warnings). `ryft-xla` remains red at the same two non-exhaustive matches (`experimental/domains.rs:4249` and
`experimental/lowering.rs:2940`), which now also miss the four cumulative variants. XLA lowerings and
`ryft-macros-tests` remain outstanding; changelog updates were excluded from this effort by request.

### Phase 1B (XLA): lowerings for the seven new operations

`ryft-xla` now covers every Phase 1B addition, closing the two non-exhaustive `ArrayOperation` matches that had kept
the crate red. `log1p` lowers to the `stablehlo.log_plus_one` primitive. `log_add_exp` has no StableHLO primitive, so
a shared `lower_log_add_exp_to_mlir` expands the guarded construction
`select(isnan(a - b), a + b, max(a, b) + log1p(exp(-|a - b|)))`; because the guard already routes every NaN operand
through the `a + b` arm, the shift uses the plain `stablehlo.maximum` rather than the total-order expansion that
`lower_extremum_to_mlir` emits. `log_sum_exp` expands through `lower_log_sum_exp_to_mlir`: a `reduce` maximum with the
`-inf` identity, `stablehlo.is_finite` (the dialect wrapper already existed, so no arithmetic is-finite trick was
needed), a `select` against a zero splat, a `broadcast_in_dim` back over the operand's kept axes, then
`exp -> reduce sum -> log -> add`.

The five prefix scans share one `lower_cumulative_to_mlir`, generalized from the former `lower_cumulative_sum_to_mlir`
by a small `CumulativeKind` enum that owns the operation name, the window's initial value, and the reducer body. Sum,
max, and min delegate to the existing `build_reduction_identity_constant` / `build_reduce_body_region` pair verbatim,
which is why the pre-existing `cumulative_sum` snapshots stayed byte-identical; the product seeds its windows through
`lower_unplaced_constant_output` (a scalar `one`, so complex element types reach the same `stablehlo.complex`
synthesis the `one` primitive uses) with a `stablehlo.multiply` body, and `cumulative_log_sum_exp` seeds with the
maximum's identity (`-inf` wherever the format has one) and inlines the shared `log_add_exp` expansion into its body.

Padding disciplines: `log_sum_exp` is `XlaMasked`, matching `Reduce`'s `Sum`/`Max` classification because the
expansion is exactly those two reductions and XLA masks each one's operand with its own identity; the four new
scans join `cumulative_sum`'s dedicated `Propagated` arm, whose justification (type inference keeps the scanned
axis static) holds for every combining operator. `ops.rs` gained all seven `impl_array_operation_conversion!`
entries.

Verified with `cargo check -p ryft-xla --all-targets` (clean, no warnings), `cargo test -p ryft-xla --lib`
(537 passed, 0 failed, 5 ignored; 4 new tests), `cargo test -p ryft-macros-tests` (now unblocked, all green), and
`cargo check -p ryft-core --all-targets` (unchanged and clean). New coverage: five lowering snapshots (`log1p`,
the full `log_add_exp` expansion, the full `log_sum_exp` safe-max expansion, and the product / log-sum-exp reducer
bodies alongside max and min), the eager reference-backend parity lines for all seven operations including both
scan directions, and one end-to-end CPU execution test pinning `log_sum_exp` at magnitude 1000 (where a shift-free
`sum(exp(x))` would overflow `f32`) together with a reverse `cumulative_log_sum_exp`. Changelog updates were excluded
from this effort by request.

### Audit fixes (core)

Three independent audits of the Phase 1 change set were folded back into `ryft-core`.

The one correctness bug worth calling out: `cumulative_evaluate` derived its block bounds from the payload length and
the scanned axis's row-major stride, and that stride is zero whenever a zero-extent axis lies to the *right* of the
scanned one, so scanning axis 0 of an `f32[2, 0]` panicked with a division by zero. All five members funnel through
that helper. Both bounds are now direct dimension products, covered by regression cases for `[2, 0]` and `[3, 0, 2]`.

Four further correctness fixes: `ragged_mask_identity_scalar` now verifies that the arithmetic identity survives the
conversion into the element type (`One` in the two-valued `i1`, whose range is `{-1, 0}`, silently became `-1`);
`log_sum_exp` and `cumulative_log_sum_exp` now reject `f8e8m0fnu` at the type level, because that format has neither
the zero the inner sum needs nor a sign, and its smallest element exponentiates to one rather than acting as the
combining operator's identity (the XLA `reduce_window` seed for the scan would have let padded windows climb into live
prefixes); `log_sum_exp`'s empty-axes shortcut now validates the element data type before returning, as its own
"rejects exactly what a staged program rejects" contract requires; and the low-precision scan tests now use cases that
actually discriminate per-step re-encoding from rounding one exact accumulation.

The four nonlinear members (`cumulative_product`, `cumulative_max`, `cumulative_min`, `cumulative_log_sum_exp`) were
near-identical 300-line clones — min and max differed by three non-test lines — and are now one
`define_cumulative_operation!` invocation each, following the `shape_changing_collective!` house pattern. Each
invocation supplies the member docs, the name constant, the abstract rule's element-domain predicate and its exact
diagnostic, the `RaggedMaskIdentity` variant, the combining operation and closure, and the capability trait with its
two method names; everything else is generated. `cumulative_sum` stays hand-written because it is the family's one
linear member, with a real transposition rule and a cheaper forward mode. Public API is byte-compatible, which
`cargo check -p ryft-xla` confirms. Per the testing guidelines the four per-member transposition tests collapsed into
one central test over a single generated member, joined by one central partial-evaluation test (the family had none).

Smaller simplifications: `reduce_shape_abstract` now carries the reduction family's shared axis geometry for both
`reduce_abstract` and `log_sum_exp_abstract` (the element-type check is passed in, which keeps the validation order);
`batch_reducing_operation` carries the axis-collapsing batching skeleton for both `ReduceOperation` and
`LogSumExpOperation`; `scan_sum`/`scan_product` merged into `scan_arithmetic`, matching the file's own `scan_extremum`
idiom; `impl_array_binary_float_math!`'s two branches share one `@kernel` prologue branch; and `DecompositionTracer`
became private. Plus the convention pass: module headers on every `operations/cumulative/` file, rustdoc on trait impl
blocks converted to `//` comments in the new files, `cumulative_abstract`'s `op` parameter renamed to
`operation_name`, facade imports in `arrays/operations/cumulative.rs`, a rewritten `RaggedArrayBatchingPolicy` opening
paragraph naming its three disciplines (reduction masking, contraction zeroing, identity masking),
corrected `RaggedMaskIdentity` extremes documentation, and a backticked `log1p` kernel diagnostic.

Verified with `cargo check -p ryft-core --all-targets` (clean, no warnings), `cargo test -p ryft-core --lib`
(1655 passed, 0 failed, 3 ignored — unchanged in total: four per-member transposition tests removed against one
central transposition test, one central partial-evaluation test, and two ragged-masking error-path tests added, with
the remaining new coverage folded into existing tests), `cargo test -p ryft-macros` (green),
`cargo check -p ryft-xla` (still green, as the public surface did not move), and `cargo doc -p ryft-core --no-deps`
(no new warnings).

### Audit fixes (collectives/XLA)

The follow-up audit of the collectives split and the XLA side of Phase 1 was folded back in.

The one finding with real risk attached was the bounded-dynamic admission of `log_sum_exp`: `domains.rs` classifies it
as `XlaMasked`, which admits it to data-derived compilation, but the expansion builds its `safe_m` zero splat at the
*output* type, and that type is dynamic whenever the data-derived extent sits on a kept axis. The doctrine in that
module admits only configurations with execution evidence, and the rank-1 `data_derived_padding_fixture` had none for
this shape. The fixture now also builds the rank-2 `i + j` matrix `f32[extent, 4]` and returns `log_sum_exp` over its
static axis (a dynamically shaped result) together with a `cumulative_sum` along that same static axis. Both execute
correctly on CPU at size 2 and at the bound: shapes, the byte-for-byte comparison against the reference kernel, and
the hand-computed values (`i + ln(1 + e + e² + e³)` per row, and `[i, 2i + 1, 3i + 3, 4i + 6]` per scanned row) all
agree, so the classification stands on evidence and needed no lowering change or demotion. The two classification
arms now cite that fixture.

`cumulative_log_sum_exp` gained two more rejected element formats. Its documentation had claimed that every
finite-only format's lowest value "exponentiates to zero", which is not the property that decides the question: what
a padded window needs is a sentinel that survives being folded with copies of itself. `f6e2m3fn` has lowest value
`-7.5`, and two copies of it already fold to `-7.5 + ln(2) = -6.807`, which rounds to `-7.0`; `f4e2m1fn` has lowest
value `-6`, which survives two copies but not three, since `-6 + ln(3) = -4.901` rounds to `-4`. Both are now
rejected by the member's `element_domain` with their own diagnostic, and the operation docs and the XLA seed comment
state the concrete per-format reasoning in place of that blanket claim.

The gray zone around the remaining finite-lowest formats is closed by a fold-count criterion in
`lower_cumulative_to_mlir` rather than by more type-level rejections. A `reduce_window` window folds the seed once as
its reduction accumulator and once per padded position, so the widest window of a scan folds `extent` copies, and
folding `k` copies of `lowest` yields `lowest + ln(k)`, which is still the seed exactly while `ln(k)` stays inside
half the gap from `lowest` to its neighbor toward zero. The bound per format is therefore `floor(e^(half gap))`,
tabulated in `log_sum_exp_seed_fold_bound`: `f8e4m3fn` holds 8_886_110 folds, `f8e4m3fnuz` 2_980, and `f8e5m2fnuz`
`e^4096`, which saturates, so no extent is ever rejected for it in practice. `f6e3m2fn` (seven folds) and
`f8e4m3b11fnuz` (two) hold too few for any useful scan length and are rejected outright; the other three are rejected
when the scanned extent exceeds the format's fold bound, and the true-`-inf` formats (`bf16`, `f16`, `f32`, `f64`,
`f8e3m4`, `f8e4m3`, `f8e5m2`) are unconditionally accepted. This round put the whole criterion in the lowering,
reasoning that the eager kernel seeds from the first real element and is exact at every one of these formats; round 3
revisited that placement, because `ryft-core`'s own ragged masking folds sentinels as well.

Test strength and simplicity fixes: the eager parity witness for `cumulative_max` ran on data whose running maximum is
constant (a global-maximum broadcast would have passed), so the forward witness moved to the strictly increasing
operand and the family's one untested direction, `reverse_cumulative_max`, was added; `log_add_exp`'s infinity and NaN
semantics are now pinned on both backends directly, since a difference-based tolerance cannot carry them; the
cumulative lowering's dynamic-scanned-axis rejection has a negative test built through `add_instruction_unchecked`,
which is the only way past `ryft-core`'s own static-axis rule; `lowered_cumulative_module` became
`lowered_unary_module` and now backs `lowered_reduce_module` and the two hand-rolled elementwise lowering tests;
`CumulativeKind::build_body_region` dispatches once instead of twice (dropping the derive that the removed `==` was
the only consumer of); the six per-operation collective batching helpers moved out of `collectives/mod.rs` into their
owning operation files as private items, which is where the payload fields they read are in scope (those fields were
private all along and never needed widening); and the
remaining convention items (the fused all-reduce doc block, a rustdoc comment on a trait impl, the reflowed
collectives module header, `pub(super)` engine macros, backticked operation names in a differential-testing
diagnostic, and the softened `TryOptimizeAssociativeScan` claim) were applied as written.

Verified with `cargo check -p ryft-core --all-targets` and `cargo check -p ryft-xla --all-targets` (both clean, no
warnings), `cargo test -p ryft-core --lib` (1657 passed, 0 failed, 3 ignored: 1655 plus the two extra tests the
cross-cutting forwarding test split into), `cargo test -p ryft-xla --lib` (539 passed, 0 failed, 5 ignored: 537 plus
the dynamic-scanned-axis and drifting-seed-format rejection tests), and `cargo test -p ryft-macros-tests` (green).

### Audit fixes (round 2)

A second audit of the same change set produced ten findings, all folded back in.

The one red target was a rustdoc doctest: the fold-bound table in `log_sum_exp_seed_fold_bound` was indented four
spaces under `///`, which rustdoc collects as a Rust code block, so `cargo test -p ryft-xla --doc` failed to compile
it. The table now uses the unindented Markdown style of the repo's other rustdoc tables.

The one correctness finding: `log_sum_exp` rejected only `f8e8m0fnu`, even though its ragged batching rule masks
padding with `RaggedMaskIdentity::Lowest` — exactly the sentinel that `cumulative_log_sum_exp` had already grown two
more rejections for. `f6e2m3fn` and `f4e2m1fn` are now rejected there too, and rather than restating the domain in
two places, the predicate and its diagnostic became one shared pair — `is_log_add_exp_identity_data_type` and
`log_add_exp_identity_data_type_error` in `math/log_add_exp.rs`, the module that owns the identity the criterion is
about. The cumulative member's `element_domain` and `element_domain_error` now call them, so the two operations
cannot drift apart again, and every existing diagnostic string is reproduced verbatim. All three type-level
rejections also gained an outright arm in `lower_cumulative_to_mlir`: checked construction cannot reach it, but
`add_instruction_unchecked` could otherwise have seeded a window with `f8e8m0fnu`'s "lowest" value, which is the
*positive* `2^-127`. The layering asymmetry was documented as intentional at this point — `ryft-core` rejects at the
type level because `ryft-core` itself writes the sentinel during ragged batching, while the extent-aware fold-count
bound is a `reduce_window` property and belongs to the lowering — but the line between the two layers still ran
through the middle of the fold-count criterion, which round 3 moved (see below).

Documentation and convention items: `RaggedArrayBatchingPolicy::mask_identity_input` is now defined by the value it
writes rather than by what becomes of the masked axis (`LogSumExpOperation::batch` legitimately masks axes it then
consumes, since `ReductionKind` has no log-sum-exp member), with consumption and its evidence stated as the calling
rule's own business; the fold-bound doc no longer claims that `None` marks exactly the formats whose seed is a true
negative infinity; the 17 rustdoc comments on trait impl blocks that the collectives split carried over verbatim from
the flat file became `//` comments; `parallel_mean` and `parallel_max` are backticked in their three diagnostics;
`reduce_abstract` and `reduce_sharding`'s `op` parameters became `operation_name`, matching the `operation_name` that
`reduce_shape_abstract` introduced into the same file; and the stale collective identifiers in `gemma_4_plan.md` and
`muse_glimmer_30b_plan.md` (`CollectiveKind::{PSum, PMean, PMax}`, `PSumScatterOperation`, `PpermuteOperation`,
`Pshuffle`, `PSwapAxes`, and the lowercase `psum` / `psum_scatter` / `ppermute` feature names) now use the Phase 2
names, leaving the `jax.lax.*` references untouched.

This Review section was audited against the tree as well: both `pub(crate) use` claims about the engine macros are
actually `pub(super) use`; the macro-generated payload fields never needed widening, because the per-operation
helpers live in the files that own those payloads; `cargo test -p ryft-core --lib collectives` is 43 today, up from
the 41 the Phase 3 entry recorded, the two extra tests being the halves the cross-cutting forwarding test split into
during the collectives/XLA round; the previous entry's `ryft-xla` count was 538 against an actual 539 (537 plus two
rejection tests); the outstanding changelog line now matches the excluded-by-request phrasing used everywhere else;
and one sentence that ended mid-clause plus several edit-scarred paragraphs were completed and reflowed.

Verified with `cargo check -p ryft-core --all-targets` and `cargo check -p ryft-xla --all-targets` (both clean, no
warnings), `cargo test -p ryft-core --lib` (1657 passed, 0 failed, 3 ignored — unchanged, because the new rejection
assertions extend `test_log_sum_exp_abstract` and the new unchecked-seed assertion extends the existing lowering
rejection test), `cargo test -p ryft-xla --lib` (539 passed, 0 failed, 5 ignored), `cargo test -p ryft-core --doc`
and `cargo test -p ryft-xla --doc` (both green; the latter was red before this round), and
`cargo test -p ryft-macros-tests` (green). Changelog updates remain excluded from this effort by request.

### Audit fixes (round 3)

A third audit produced eleven findings: one substantive (a rationale that was arithmetically false, and a format
split that turned out not to rest on any criterion), two mechanical convention items, one broken rustdoc link, and
seven record-accuracy items against this Review section.

The substantive one. The shared predicate's rationale claimed that `f6e2m3fn` and `f4e2m1fn` both drift after a
single fold and that `f4e2m1fn`'s `exp(-6) = 2.5e-3` "is an ordinary number rather than an underflow". Both claims
are false: `f4e2m1fn` represents `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`, so `2.5e-3` underflows to zero in it, and
`-6 + ln(2) = -5.307` rounds back to `-6` — the format holds two sentinel copies and first drifts at three, where
`-6 + ln(3) = -4.901` rounds to `-4`. Only `f6e2m3fn` drifts at the first fold (`-7.5 + ln(2) = -6.807` rounds to
`-7.0`). Correcting the arithmetic exposed the real problem: `f4e2m1fn` (reach two) was rejected by type inference
while `f8e4m3b11fnuz` (reach two, and likewise a pairwise identity) was accepted there and rejected only by the XLA
lowering. The two formats are indistinguishable on every property the code could be reading, so the previous
layering — "type inference rejects the formats with no usable sentinel, the lowering bounds the rest by extent" —
described no criterion that the code actually implemented.

The unification adopts one criterion at both layers: the sentinel must stay the identity across every copy of itself
that one accumulation folds, and a format's *reach* is the largest number of copies it holds, `floor(e^(half gap))`
for a finite lowest value. Type inference cannot compare a reach against a fold count, because the count
`ryft-core`'s own ragged masking produces is a ragged axis's bound minus its per-item extent, a runtime quantity, so
the type-level line is reach alone. `is_log_add_exp_identity_data_type` now rejects the five formats whose reach is
short enough that any ragged mask worth writing exceeds it — `f8e8m0fnu` (no sentinel at all), `f6e2m3fn` (one),
`f4e2m1fn` (two), `f8e4m3b11fnuz` (two), and `f6e3m2fn` (seven) — which adds `f8e4m3b11fnuz` and `f6e3m2fn` to the
type-level domain of both `log_sum_exp` and `cumulative_log_sum_exp`, with their existing diagnostic reproduced
verbatim. The three accepted finite-lowest formats keep a documented quantitative limit instead of a check, since
there is no count to check against: `f8e4m3fnuz` at 2_980 copies, `f8e4m3fn` at 8_886_110, and `f8e5m2fnuz` at
`e^4096`, which saturates. The predicate's rustdoc tabulates all of it. `lower_cumulative_to_mlir`'s two outright
arms collapsed into one defensive check over the same five formats, citing type inference, and its extent-aware
bound for the other three is unchanged — it is the one consumer that does know the fold count, the scanned extent.
Behavior change confined to those two formats; nothing else moved.

The mechanical items: the two `///` blocks that survived inside the `shape_changing_collective!` transcriber (they
were emitted onto every generated `PartiallyEvaluatableOperation` and `DifferentiableOperation` impl, which is what
the round-2 sweep of 17 such comments had missed) became `//` comments, and the generated capability method's
`[`AxisError::UnboundAxisName`]` and `[`NamedAxes`]` links became explicit `crate::axes::...` paths, since neither
name is in scope at the expansion sites; that was the last unresolved-link warning this change set contributed to
`cargo doc`. Four under-filled rustdoc paragraphs in `parallel_reduce.rs` were reflowed toward the 120-column limit.

Record accuracy in this Review: the Phase 3 entry no longer implies that `mod.rs` still holds the three matching-axis
batching kernels, and its collectives test count is back to the 41 that was true at the time; the three incompatible
private-field narratives (Phase 3, collectives/XLA, round 2) now agree that the macro-generated payload fields were
never widened and that what moved is where the helpers reading them live; "only a handful of shared helpers became
`pub(super)`" is now the measured 20 `pub(super) fn` items plus two `pub(super) use` macro re-exports; the round-2
self-correction that read "43, not 41" now explains the 43 as today's count, two above the Phase 3 figure because the
cross-cutting forwarding test split in two; "the packed physical extent is now a `BatchingError`" says instead that
the failure to resolve it is now reported as one; "rejected only past their extent" says when the scanned extent
exceeds the format's fold bound, and notes that `f8e5m2fnuz`'s bound saturates and is never reached; the false
`f4e2m1fn` arithmetic is corrected in the collectives/XLA entry too; and the orphaned mid-paragraph lines were
reflowed. One short line was left as it is: the `log_add_exp` expansion in the Phase 1B (XLA) entry is a 62-character
code span that cannot share a line with the text introducing it.

Verified with `cargo check -p ryft-core --all-targets` and `cargo check -p ryft-xla --all-targets` (both clean, no
warnings), `cargo test -p ryft-core --lib` (1657 passed, 0 failed, 3 ignored — unchanged, since the two new rejected
formats extend existing loops and the lowering's two rejection blocks merged into one), `cargo test -p ryft-core
--lib collectives` (43), `cargo test -p ryft-xla --lib` (539 passed, 0 failed, 5 ignored), `cargo test -p ryft-core
--doc` and `cargo test -p ryft-xla --doc` (both green), and `cargo doc -p ryft-core --no-deps` (the
`AxisError::UnboundAxisName` warning gone, no new warnings, and all remaining warnings are pre-existing). Changelog
updates remain excluded from this effort by request.

### Final verification and audit closure

Completed Phases 7–10 and closed the full plan. The final implementation includes exact eager and staged ragged
custom-call metadata semantics; a first-class ragged all-to-all across eager execution, abstract evaluation,
batching, differentiation, StableHLO lowering, and shard-map execution; conservative bounded-ragged propagation or
explicit rejection for the existing collective family; and grouped-dot inference, eager execution, batching,
differentiation, CHLO modeling, portable decomposition, and instruction lowering.

Three independent audit tracks reviewed the complete change set for correctness, repository conventions, and
simplicity. Findings were iteratively fixed, including grouped-dot contracting-prefix accumulation, complex-zero and
`f8e8m0fnu` handling, replicated custom-call freshness, mapped and replicated untiled all-gather metadata, mixed
`ArrayIr` collective parity, mapped and replicated ragged extent carriers for mixed collectives, exact lowering
snapshots, hot-loop allocation removal, block replacement without full-buffer cloning, and rustdoc placement. The
final audit round reported no findings in all three tracks.

A post-closure architecture cleanup then removed grouped-dot lowering policy from `ryft-core`: the semantic operation
now carries only dimension numbers, while `ryft-xla::XlaOptions` owns the compilation-wide decomposition/instruction
choice and includes it in cache identity. The three audit tracks reviewed that cleanup independently; their record,
API-documentation, state-inheritance, and hash-contract findings were fixed, and the repeated final review reported no
findings in all three tracks.

Final verification passed: `ryft-core` reported 1,710 passed and 3 ignored; `ryft-xla` reported 550 passed and 5
ignored; `ryft-macros` reported 57 unit tests passed plus its doctest suite; and `ryft-macros-tests` reported both
21-operation and 17-parameter suites passed, including all trybuild cases. Both all-target checks completed with no
warnings. Rustdoc completed for `ryft-core`, `ryft-mlir`, and `ryft-xla`; the final task-owned broken link was fixed,
leaving only the repository's pre-existing warnings. Formatting, whitespace, and added-line width checks passed.
The Phase 2 stale-name search found only deliberate references to upstream `jax.lax.p*` names and differential case
identifiers, not stale Ryft APIs. No changelog file was modified.

## Post-plan feedback remediation

- [x] Represent every accumulated dense custom-call batch prefix, while continuing to reject a genuinely nested
  ragged batch, and cover homogeneous and mixed double batching.
- [x] Reject fresh output dimensions that collide with input ragged dimensions, narrow alias rejection to
  ragged-bound inputs, and pin fresh renaming and mapped mixed-universe attachment.
- [x] Route replicated `parallel_permute` extents through the participant permutation so untargeted participants have
  zero extents.
- [x] Make eager `ragged_dot_general` clip over-covering group sizes consistently with staged lowering and document
  the behavior.
- [x] Reject batching-internal participant-packed `ragged_all_to_all` operations at the XLA per-device custom-call
  lowering boundary while preserving the public logical form used inside `shard_map`.
- [x] Add the missing numerical ragged-reduction, nonuniform transpose, and prefix-shaped lowering regressions.
- [x] Run focused and affected-crate verification without modifying changelog files.
- [x] Complete independent correctness, conventions, and simplicity audit rounds, fix every finding, and repeat until
  all three tracks report no findings.

### Post-plan feedback remediation review

Custom-call ragged contracts now count every accumulated dense prefix independently from whether a prior transform
discharged a ragged level. `BroadcastAll` shifts the contract for each dense prefix, while `Sequential` records a
ragged discharge without shifting the scan body's per-item axes. Repeated dimension identities require one extent
source, fresh identities cannot collide with input identities, consumption evidence is unique per dimension and is
emitted only when no binding for that dimension is preserved, and aligned outputs retain the aligned extent carrier.
The homogeneous and mixed implementations have regressions for double dense batching, differing mapped axes, fresh
outputs, sequential discharge, shared dimensions, and genuine nested-ragged rejection.

Replicated ragged extents now follow `parallel_permute` values through the same participant permutation, including
zero extents for untargeted participants; partial permutations reject positive lower bounds that cannot represent
those zeros. Numerical parallel-sum and parallel-max tests pin padding identities and reduced extents. Eager
`ragged_dot_general` clips over-covering group intervals to the physical extent and stops before irrelevant later
metadata can overflow. XLA widens group metadata to `u64`, caps every value before the reassociable StableHLO window
reduction, accumulates saturating intervals, and reuses one interval pipeline for both contracting operands. Exact
snapshots and CPU execution cover narrow metadata, prefix-shaped groups, and `u64::MAX` overflow cases. Nonuniform
ragged-dot transpose tests pin the cotangent semantics.

The XLA per-device ABI continues to accept the public logical `ragged_all_to_all` representation used by `shard_map`
and explicitly rejects only the batching-internal participant-packed physical representation. A direct lowering test
constructs that representation through the public batching path and pins the diagnostic.

Final verification passed with `cargo test -p ryft-core --lib` (1,720 passed, 3 ignored), `cargo test -p ryft-xla
--lib` (554 passed, 5 ignored), all-target checks for both crates, `cargo test -p ryft-core --doc` (67 passed, 16
ignored, plus 6 compile-fail cases), `cargo test -p ryft-xla --doc`, `cargo +nightly fmt --all -- --check`, and
`git diff --check`. Focused suites included 33 custom-call tests, 8 parallel-permute tests, 12 parallel-reduce tests,
and 7 XLA ragged-dot lowering and CPU tests. Independent correctness, conventions, and simplicity auditors reviewed
the post-fix tree repeatedly; every finding was fixed, including the StableHLO reduction-order portability issue, and
the final closure round reported no findings in all three tracks. No changelog file was modified.
