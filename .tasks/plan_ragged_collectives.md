# Ragged Collectives Plan

Close the explicit ragged-communication gap with JAX: add a new `operations::cumulative` module covering JAX's full
cumulative-reduction family (`cumulative_sum` is needed by the ragged transpose mask; the rest complete the family),
rename the `p*` collectives to their full-word `parallel_*` forms, split `operations::collectives` into
per-operation-kind submodules, add a first-class `ragged_all_to_all` collective (packed data + explicit
offsets/sizes, mirroring `jax.lax.ragged_all_to_all`), add a declared ragged batching contract to
`CustomCallOperation` (deliberately exceeding JAX, which only supports the manual explicit-operand pattern), and
keep the per-collective ragged rules as an explicitly gated final phase. The existing
pre-binding rejection of `RaggedAxis` operands stays the default for every operation without an exact contract.

Reference semantics (verified against `jax/_src/lax/parallel.py` and the JAX docs):

- Six operands: `operand (N, A, ...)`, `output (M, A, ...)`, and rank-1 integer `input_offsets`, `send_sizes`,
  `output_offsets`, `receive_sizes`, all of equal length `K` divisible by the effective group size. Result has the
  output's type; unwritten output elements pass through.
- `output_offsets` are in the **receiver's** coordinate frame; `send_sizes == all_to_all(receive_sizes)` must hold.
- JVP is jointly linear in `(operand, output)`: tangent result is `ragged_all_to_all` of the tangents with the same
  metadata. Metadata is nondifferentiable.
- Transpose is another `ragged_all_to_all` with roles swapped and offsets collectively permuted:
  `operand_cotangent = ragged_all_to_all(cotangent, zeros, all_to_all(output_offsets), receive_sizes,
  all_to_all(input_offsets), send_sizes)`; `output_cotangent` is the cotangent with the received regions masked to
  zero (regions defined by the permuted `output_offsets` and `receive_sizes`).
- vmap over an unrelated axis merges the batch dimension into the packed leading axis (offsets rebased by
  `iota * N` / `iota * M`); vmap over the collective's own axis and `axis_index_groups` batching are unsupported.
- Lowering is `stablehlo.custom_call @ragged_all_to_all` with `api_version = 4` (typed FFI) and a dictionary
  `backend_config` carrying `replica_groups` (equally sized groups required) plus `channel_id` under SPMD.

## Phase 1: Logarithmic math primitives and the cumulative operations module

Scheduled first so `cumulative_sum` is available to the ragged all-to-all transpose (Phase 5) and because it is
independent of the collectives work. Two parts: three new `operations::math` primitives, then the
`operations::cumulative` module that consumes them.

### Math additions (`operations::math`)

- [ ] `math/log1p.rs`: unary elementwise `Log1pOperation` (name `log1p`, matching Rust's canonical `ln_1p` /
  StableHLO's `log_plus_one` concept), computing `log(1 + x)` accurately near zero. JVP: `tangent / (1 + x)`.
  Eager interpretation via Rust's `ln_1p`; XLA lowering to `stablehlo.log_plus_one` (the `ryft-mlir` wrapper
  already exists).
- [ ] `math/log_add_exp.rs`: binary elementwise `LogAddExpOperation` (name `log_add_exp`), computing
  `log(exp(a) + exp(b))` stably as `max(a, b) + log1p(exp(-|a - b|))` with explicit handling of the
  equal-infinite-operand cases (mirroring `jnp.logaddexp`, which is composed in JAX; Ryft makes it a first-class
  operation). JVP: softmax-weighted tangents (`exp(a - result) * da + exp(b - result) * db`). Eager interpretation
  via the stable formula; XLA lowering by expansion into existing StableHLO operations (StableHLO has no logaddexp
  primitive); standard binary-elementwise batching.
- [ ] `math/log_sum_exp.rs`: axis-reduction `LogSumExpOperation` (name `log_sum_exp`), following
  `ReduceOperation`'s axes conventions, computing `log(sum(exp(x)))` over the reduced axes stably via the max-shift
  decomposition (`m + log(sum(exp(x - m)))` with the `-inf` empty/all-masked cases defined explicitly, mirroring
  `jax.nn.logsumexp`). JVP: softmax-weighted tangent reduction. Ragged masking uses the `-inf` identity through the
  reduction-masking machinery; XLA lowering by expansion through the stable decomposition.
- [ ] Per-operation capability traits, math-convention tests (hand-computed values including the stability and
  infinity edge cases, gradient checks, batching, lowering snapshots), enum wiring, and changelog entries.

### Cumulative operations module (`operations::cumulative`)

New module covering JAX's cumulative-reduction family (`jax/_src/lax/control_flow/loops.py`). All five operations
share one payload shape (`axis: usize`, `reverse: bool`), one type-inference contract, one batching rule, and one
lowering pattern; they differ in combine function, identity, and differentiation. JAX's alternative `chlo.ScanOp`
GPU lowering is dead code (its feasibility gate returns `False` unconditionally), so the full-prefix-window
`reduce_window` lowering — which XLA's `TryOptimizeAssociativeScan` rewriter converts into an efficient parallel
scan — is the single faithful implementation for every member.

Module layout, one submodule per operation type, each named after the full operation name:

- `cumulative/mod.rs`: facade re-exports plus the shared machinery: payload validation and type inference (output
  type equals input type; `axis` in bounds; numeric data type; where the type model expresses sharding, the scanned
  axis must be unsharded, mirroring JAX's sharding rule), the axis-shift batching helper (bump `axis` when the
  inserted batch axis precedes it, JAX's `_cumred_batch_rule`), the eager sequential prefix-scan interpreter
  parameterized by the combine function (reversed iteration when `reverse`; zero-length scan axis returns the input
  unchanged), `RaggedAxis` masking on the scanned axis with the operation's own identity via the existing
  reduction-masking machinery (ragged axes elsewhere pass through), and the staged associative-scan decomposition
  helper (JAX's log-doubling construction with `_interleave` built on interior-padded `PadOperation`, which Ryft's
  pad already supports) that the nonlinear members' JVP rules differentiate through.
- `cumulative_sum.rs`: `CumulativeSumOperation` (name `cumulative_sum`), identity zero. The only *linear* member:
  its JVP applies the operation to the tangent and its transpose rule is itself with `reverse` flipped (JAX's
  `_cumsum_transpose_rule`; `reverse` is in the payload precisely so the transpose is closed).
- `cumulative_product.rs`: `CumulativeProductOperation` (name `cumulative_product`), identity one.
- `cumulative_max.rs`: `CumulativeMaxOperation` (name `cumulative_max`), identity the data type's minimum.
- `cumulative_min.rs`: `CumulativeMinOperation` (name `cumulative_min`), identity the data type's maximum.
- `cumulative_log_sum_exp.rs`: `CumulativeLogSumExpOperation` (name `cumulative_log_sum_exp`), float-only, identity
  negative infinity. The combine function is the Phase 1 `log_add_exp` operation, both in the eager scan and in
  the associative-scan decomposition its JVP differentiates through; the `reduce_window` reducer body reuses the
  same stable expansion.

Checkable items:

- [ ] Add the module with the layout above; per-operation capability traits (`CumulativeSum`, `CumulativeProduct`,
  ...) each offering `cumulative_*(axis)` and `reverse_cumulative_*(axis)` sugar.
- [ ] Differentiation: `cumulative_sum` gets the first-class linear JVP/transpose rules. The nonlinear members
  (`product`, `max`, `min`, `log_sum_exp`) follow JAX's `_cumulative_jvp_rule`: their JVP differentiates through the
  associative-scan decomposition staged from existing differentiable operations, so reverse mode flows through
  those primitives under partial-eval linearization with no bespoke transpose rules.
- [ ] XLA lowering for all five: `stable_hlo::reduce_window` (first `ryft-xla` use of the existing `ryft-mlir`
  wrapper) with `window_dimensions[axis] = n`, padding `(n - 1, 0)` forward / `(0, n - 1)` reverse, unit strides
  and dilations, the member's identity as init, and the member's combine function as the reducer body — exactly
  JAX's `cumred_reduce_window_impl` — documenting the reliance on XLA's associative-scan rewriter.
- [ ] Wire all five into the `arrays::operations` enums and dispatch like the math operations; tests per the math-op
  conventions for each member (eager interpretation against hand-computed values including `reverse`, gradient
  checks — the linear transpose flip for `sum`, decomposition JVPs for the rest — batching axis shift, ragged
  masking, lowering snapshots); changelog entries.

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

- [ ] Apply the rename across `ryft-core` (operation module, `arrays/operations`, `arrays/batching.rs`, tests) and
  `ryft-xla` (`experimental/lowering.rs`, `experimental/shard_map.rs`, `experimental/ops.rs`,
  `experimental/domains.rs`, `bin/differential_testing.rs`), including error-message text, staged-program rendering
  expectations, and rustdoc prose (rereading each touched paragraph as a unit).
- [ ] Run a targeted search for every old identifier (`psum`, `pmean`, `pmax`, `ppermute`, `pshuffle`, `pswapaxes`,
  `Ppermute`, `PSum`, `PMean`, `PMax`, `PSwapAxes`, `CollectiveKind`, `CollectiveOperation`, and standalone
  `Collective` outside the retained shared-vocabulary names) to confirm no references remain.
- [ ] Audit for anything keyed on the old rendered operation names beyond in-repo test expectations (the
  differential-testing binary, any serialized fixtures or persisted compilation-cache identities); Ryft's jit caches
  are in-memory, so this is expected to be a quick confirmation, but it must be explicit.
- [ ] Verify: `cargo check -p ryft-core --all-targets`, `cargo check -p ryft-xla --all-targets`, scoped `--lib` test
  runs for both crates plus `ryft-macros` and `ryft-macros-tests` (renamed enum variants flow through the operation
  derives), 300s timeouts, `cargo fmt` check.

## Phase 3: Split `operations::collectives` into per-operation-kind submodules

Pure refactor after Phase 2, no behavior change. `mod.rs` holds the facade plus all shared vocabulary, helpers, and
macros; every other submodule corresponds to exactly one operation kind.

- [ ] Create `crates/ryft-core/src/operations/collectives/` with this layout:
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
    and `impl_shape_changing_collective_member_operation!` macros (defined with `macro_rules!` + `pub(crate) use`
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
- [ ] Split the existing `#[cfg(test)] mod tests` by operation into each submodule's own `tests` module (reduction
  and vmap-gradient tests into `parallel_reduce.rs`, type-inference and batched-axis materialization tests into their owning
  operation files, options/forwarding/vocabulary and shared explicit-member/involution tests into `mod.rs`),
  following `.agents/unit-testing-guidelines.md`.
- [ ] Fix imports per the repo conventions: `mod.rs` re-exports its children by relative path; operation submodules
  import shared items via `super::`; all out-of-module users keep importing through the unchanged
  `crate::operations::collectives::*` facade (verify `operations/mod.rs`, `arrays/operations/mod.rs`,
  `arrays/batching.rs`, and `ryft-xla` imports still resolve without edits, and update any that named the old flat
  file path directly).
- [ ] Verify: `cargo check -p ryft-core`, `cargo test -p ryft-core --lib collectives` (300s timeout), `cargo fmt`
  check, and a `git diff` review confirming the split is move-only relative to Phase 2 (no semantic drift).

## Phase 4: `RaggedAllToAllOperation` core semantics

The operation contract is explicitly packed: six array operands, returning the updated packed output. `RaggedAxis`
is batching-time metadata only (`arrays/batching.rs:49`), so it plays no role in the operation's type contract; any
`RaggedAxis` integration is a batching-rule adapter (Phase 6), never part of this surface.

- [ ] Add `collectives/ragged_all_to_all.rs` defining `RaggedAllToAllOperation` with fields `axis_name: String`,
  `axis_size: usize`, and `axis_index_groups: Option<Vec<Vec<usize>>>`. The stored `axis_size` follows the
  established shape-changing collective pattern (`collectives.rs:682`): `Operation::infer_output_types` only sees
  input types, so the user-facing capability trait resolves the size via `resolve_named_axis_size` at staging time
  and bakes it into the payload, and the grouped constructor validates group coverage against it like
  `CollectiveOperation::grouped` does (`CollectiveMode` does not apply — there is no tiled/untiled choice). Operand
  order matches JAX: `operand`, `output`, `input_offsets`, `send_sizes`, `output_offsets`, `receive_sizes`
  (full-word naming, not `recv_sizes`). Operation name string: `ragged_all_to_all`.
- [ ] Type inference: result type equals the `output` operand's type. Validate: `operand`/`output` share the data
  type and trailing (non-leading) dimensions; the four metadata operands are rank-1 integer arrays of one shared
  integer data type and equal static length `K`; `K` is divisible by the effective group size computed from the
  stored `axis_size` and `axis_index_groups`. Document the receiver-frame semantics of `output_offsets` and the
  `send_sizes == all_to_all(receive_sizes)` invariant (checkable only at runtime) in the operation rustdoc, linking
  the JAX doc page.
- [ ] Runtime metadata contract (decide and document before implementing): offsets and sizes must be nonnegative;
  every `[input_offset, input_offset + send_size)` region must lie within `N` and every receive region within `M`;
  received regions within one output must be disjoint (overlap is a precondition violation, not last-writer-wins);
  send regions may overlap (re-sending the same source slice is well defined). Violations are backend preconditions
  (undefined results at execution), with an optional checked mode that stages `OrderedAssertion`-effect bounds
  checks through the existing assertion machinery rather than inventing new failure plumbing.
- [ ] Keep `reject_ragged_collective_inputs` in the batching path for this operation too: `RaggedAxis`-carrying
  operands are rejected exactly like the other collectives until Phase 6's explicit rule lands.
- [ ] Effects / partial evaluation: like the existing collectives, this is a *contextual* operation, not an
  effectful one — Ryft's `Effect` classes (`programs/effects.rs`) carry no named-axis class, and the operation's
  correctness rides on staging-time `NamedAxes` validation plus the batching/shard_map contexts, exactly as
  `CollectiveOperation` does today. For linearization, "known" means primal-dependent in the partial-evaluation
  split, never compile-time constant: the metadata are nondifferentiable primal operands that are runtime/device
  values, and the Phase 5 transpose consumes them as residuals.
- [ ] Interpretation: split by value concreteness. *Eager* interpretation over concrete arrays (the named axis
  bound as an eager batch axis, or effective group size 1) reads the concrete metadata and performs the segment
  copies directly — no staged dynamic shapes are involved, this exceeds JAX (which always errors outside a mapped
  context), and it provides device-free execution coverage mirroring
  `test_all_to_all_over_batched_axis_exchanges_chunks`. Tests compare against a plain-Rust reference computation of
  the exchange in the test module, using JAX's documented worked example.
- [ ] *Staged* materialization over a batch-bound named axis (tracer-valued metadata) is gated: return an explicit
  `BatchingError::UnsupportedOperation` initially, because `DynamicSliceOperation` stores static slice sizes
  (`slicing.rs:1374`) so metadata-driven dynamic-length copies cannot be staged as dynamic-slice loops. Record the
  follow-up design in the rustdoc: iota-based index arithmetic + gather + `select` masking can express
  dynamic-length segment copies with existing operations at `O(group_size × M)` staged work.
- [ ] Wire the operation through `arrays::operations`: new `ArrayOperation::RaggedAllToAll` and
  `ArrayIrOperation::RaggedAllToAll` variants, `MemberKindSignature` entry, dispatch/projection plumbing, and a
  user-facing `RaggedAllToAll` capability trait alongside `AllToAll` (documented in the `arrays/operations/mod.rs`
  collectives list).
- [ ] Unit tests: type-inference success/failure matrix (dtype, rank, length, divisibility, unbound axis), ragged
  rejection, batched-axis exchange semantics against a hand-computed example (use JAX's documented worked example),
  and staged-program renderings for the staged path.

## Phase 5: Differentiation rules

- [ ] JVP: jointly linear in `(operand, output)`; the tangent result is `ragged_all_to_all` of the two tangents with
  the primal metadata. Symbolic-zero handling via `MaybeZero` (both tangents zero → zero result tangent; otherwise
  materialize both). Metadata operands are nondifferentiable integers.
- [ ] Transpose (`transpose_with_respect_to` the two data operands only):
  - `operand` cotangent: stage dense tiled `all_to_all` of `output_offsets` and of `input_offsets` over the same
    axis, then `ragged_all_to_all(cotangent, zero_like(operand), permuted_output_offsets, receive_sizes,
    permuted_input_offsets, send_sizes)`.
  - Forward `axis_index_groups` through the transpose, including into the metadata-permuting dense `all_to_all`
    calls. This is a deliberate correction over JAX, whose `_ragged_all_to_all_transpose` accepts groups on the
    primitive but drops them when permuting the offsets; document the divergence in the rule docs rather than
    claiming exact JAX parity.
  - `output` cotangent: mask the cotangent to zero on received regions, using JAX's `O(M)` construction directly
    now that Phase 1 provides `cumulative_sum`: scatter `+1` markers at each received region's start
    (`permuted_output_offsets`) and `-1` markers past each region's end into a length-`M` zeros vector, take the
    `cumulative_sum` along that axis, broadcast the nonzero-inside-region mask over the trailing dimensions, and
    `select` zeros where the mask is set. This construction silently assumes received regions are disjoint, which
    the runtime metadata contract (Phase 4) already makes an explicit precondition.
  - Note in the rule docs that this transpose stages additional collectives (it is not a local rewrite) and relies
    on the metadata operands being primal residuals — runtime values available to the transposed program, not
    compile-time constants.
- [ ] Tests: gradient checks through the batched-named-axis path (eager execution from Phase 4 makes
  `check_gradient`-style coverage possible without devices), involution-style transpose shape tests mirroring
  `test_shape_changing_collective_transposes_are_involutive`, and staged-program renderings pinning the transpose
  sequence (staged program shape is contract).

## Phase 6: Batching rule and `RaggedAxis` adapter

- [ ] Dedicated vmap rule mirroring JAX's `_ragged_all_to_all_batched_collective`: when batching over an axis that
  is not the collective's named axis, move the data batch axes to the front and flatten into the packed leading
  axis, move metadata batch axes to the trailing position, rebase `input_offsets` by `iota * N` and
  `output_offsets` by `iota * M`, run one merged `ragged_all_to_all`, and split the result. Return an explicit
  unsupported error for `axis_index_groups` under this merged rule (matching JAX). Batching over the collective's
  *own* named axis is not this rule's case at all: in Ryft that is the batch-bound named-axis path already covered
  by Phase 4 (eager interpretation supported; staged materialization gated behind an explicit unsupported error).
- [ ] Gated `RaggedAxis` adapter: do not implement until an explicit partition/routing descriptor is designed. A
  `RaggedAxis` carries one logical extent per packed batch item, while `ragged_all_to_all` needs per-source,
  per-destination segment sizes plus both offset vectors — extents alone underdetermine the routing, so any adapter
  must take the partition as an explicit input rather than inferring it. Also document the frame mismatch: the
  operation's raggedness is per participant chunk, `RaggedAxis` raggedness is per batch item, and outside a
  batching transform there is no carrier to attach metadata to.

## Phase 7: XLA lowering and execution

- [ ] `ryft-xla` lowering: match the new `ArrayOperation::RaggedAllToAll` in `experimental/lowering.rs` and emit
  `stable_hlo::custom_call` with target `ragged_all_to_all`, `CustomCallApiVersion::TypedFfi`, result type equal to
  the output operand's type, and a dictionary `backend_config` holding `replica_groups` (reuse the replica-group
  computation from `lower_all_to_all_to_mlir`, validating equally sized groups) plus the collective `channel_id`
  under SPMD/shard_map lowering, following JAX's `_ragged_all_to_all_lowering`.
- [ ] shard_map integration: thread the operation through `experimental/shard_map.rs` like the other collectives
  (manual-axis resolution, sharding/type propagation via the tracked output type).
- [ ] Backend gating: surface an explicit `unsupported` error where the backend cannot execute the custom call
  (CPU), at the earliest layer that knows the backend; do not let it fail as an opaque runtime custom-call miss.
- [ ] Tests, tiered by what each environment can run: the eager-interpretation semantic coverage (Phase 4) and
  lowering snapshot tests (module `to_string()` comparison per the MLIR test conventions) run everywhere; device
  execution tests run only where the backend advertises support (JAX documents `ragged_all_to_all` as an
  accelerator collective, not a CPU one), gated like the existing multi-device collective coverage; include the JAX
  documentation worked example as the reference output in both tiers.
- [ ] Changelog entries for every crate touched (`ryft-core`, `ryft-xla`, and `ryft-mlir` if the custom-call wrapper
  needs extensions).

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

- [ ] Contract data model: an optional declaration on `CustomCallOperation` (builder-style
  `with_ragged_contract(...)`) stating:
  - *input bindings*: for each ragged packed operand axis, the operand index + axis it lives on and the operand
    index of the extents vector that bounds it (the extents operand is appended by the batching rule at discharge
    time, exactly what a JAX FFI user does by hand);
  - *output bindings*: for each declared output, whether it preserves a named input binding's ragged axis (same
    dimension variable, relocated axis), consumes it (dense output), or takes fresh extents from a declared
    extents-valued output (new dimension variable);
  - a *padding-independence promise*: the kernel's live output elements do not depend on padded input elements.
  Validate the declaration structurally at construction/inference time (indices in range, extents operands rank-1
  integer, axis bounds, no double-binding), surfacing `TypeError`s with precise messages.
- [ ] Batching-rule discharge: when a `RaggedAxis`-carrying batch reaches a custom call *with* a contract, stop
  rejecting: discharge each bound `RaggedAxis` into its declared explicit extents operand
  (`RaggedAxis::extents` is already an ordinary value), stage the call through the existing
  `CustomCallBatching` machinery, and attach result `RaggedAxis` metadata per the output bindings. Support both
  modes: under `BroadcastAll` the packed buffer and whole extents vector pass through one call; under `Sequential`
  each scanned slice stays padded to the bound and the per-item extent is threaded as a scanned operand —
  `Sequential` does not dodge the contract. Ragged axes on operands or axes not covered by the contract keep
  today's rejection.
- [ ] Scope boundaries, stated in rustdoc: one ragged axis per operand initially; differentiation of ragged custom
  calls follows the existing custom-call AD story unchanged (the contract adds no AD semantics); the padded
  buffer's garbage elements remain garbage-by-contract downstream, with the attached output `RaggedAxis` making
  masked reductions and dimension-size rules compose as usual.
- [ ] Tests: the no-contract rejection tests stay green unchanged; contract validation matrix; staged-program
  renderings for ragged `Sequential` and `BroadcastAll` discharge; an eager end-to-end test with a test kernel
  asserting the extents arrive as declared and the output `RaggedAxis` composes with a downstream masked
  reduction.

## Phase 9 (gated): ragged-aware rules for individual collectives

Demand-driven, one operation at a time, never a generic "collectives preserve raggedness" rule.

- [ ] `parallel_sum`/`parallel_max`: mask padding with the reduction identity via the existing
  `RaggedArrayBatchingPolicy::mask_reduction_input` machinery; result extents are the elementwise `parallel_max` of
  the participating extents (costs one extra collective on the extents; document the partial-participation
  semantics).
- [ ] `parallel_mean` stays explicitly unsupported for ragged inputs unless its denominator is defined first
  (participant count, present-value count, or logical element count are all defensible and give different results).
- [ ] `parallel_permute` / `all_gather`: co-move the packed value and its extents, remapping the output `RaggedAxis`
  (`all_gather` extents become per-(participant, item) via multi-axis `extent_axes`).
- [ ] `all_to_all` over ragged operands: requires an explicit routing descriptor before any lowering to
  `ragged_all_to_all` — an ordinary `all_to_all` call plus one logical extent per item does not determine the
  per-destination partition, so this is a new API surface, not an automatic rewrite.

## Review

(To be filled in as phases complete.)
