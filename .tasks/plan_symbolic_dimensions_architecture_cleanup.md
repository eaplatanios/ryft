# First-class dimension architecture cleanup — remaining work

## Status

Fresh plan cut on 2026-08-08 from the original master plan, which grew to ~7,600 lines of mostly completed work and
review history. The complete history — objective, architectural diagnosis, target architecture, execution phases 0–9
with their log entries, the delivery ledger, the verification matrix with all closed rows, and every review entry —
lives verbatim in `.tasks/plan_symbolic_dimensions_architecture_cleanup_archive_2026-08-08.md`. This file contains
only the work that remains, renumbered into phases 1–7; each phase names the archive phase it continues.

Where things stand: Phases 0–8 of the archive plan are complete except for the residuals listed in Phase 1 below.
That includes the full production XLA/composite cutover, explicit dimension authority through all control flow,
linear-call differentiation residuals, the compiled gateway with per-class effect-token lowering, the bounded
physical ABI with CUDA `PadToStatic` execution, the homogeneous/mixed naming endgame renames, manual-region
first-class extents, contiguous array storage with scalar-backend retirement (Phase 9a), and the 2026-08-08
verification-matrix re-audit with its small-fixture batch and P0 diagnostic-golden re-homing. Tier-3 dynamism
(data-dependent extents through the `dimension_from_scalar` gateway) has its semantic entry point landed; Phases 4–6
below close its semantics, compiled execution, and ragged batching.

## Standing invariants

These constrain all remaining phases; the full rationale is in the archive plan's "Non-negotiable invariants" and
"Objective" sections.

- One graph: runtime dimensions are ordinary SSA values in the one program graph. No expression trees, witnesses,
  scopes, substitution, hidden shape environments, packed shape tensors, host readback, or a second dimension
  program — ever, including in tier-3 work.
- One operation contract: each operation payload has one compiler-enforced semantic contract. No dual
  `Operation<ArrayType>`/`Operation<ArrayIrType>` semantics for one payload.
- Contained heterogeneity: heterogeneous storage matching stays limited to projections, outer dispatch, and genuinely
  mixed rules; array-only and dimension-only rules operate over their homogeneous types.
- `DimensionType` stays strictly identity plus bounds. Concrete extents are runtime values and output-refinement
  observations, never part of `Typed::r#type`, structural equality, hashing, rendering, persistence, or cache
  identity. Note the 2026-08-08 correction: `DimensionVariable` identity is nominal (`Arc::ptr_eq`), so
  independently created same-named variables intentionally do *not* share cache identity; the guaranteed invariance
  is to the runtime extent.
- Stable diagnostics and effects: exact diagnostics, deterministic same-class `OrderedAssertion` ordering, and
  bounds checks that provably survive DCE.
- Backwards compatibility is not a goal. No compatibility aliases, shims, or retained superseded surfaces.
- Data-to-dimension conversion happens only through the checked `dimension_from_scalar` gateway.

## Phase 1: close the phases 0–8 residuals (archive Phase 9 exit)

Archive Phase 9's exit criterion requires every unchecked residual of phases 0–8 to be closed. This is the complete
list.

- [ ] P3k collective-parity tail: group-aware *production* reachability for the collectives, Phase 7 bounded-dynamic
      multi-device execution, and the final current-JAX behavioral/StableHLO comparisons (the comparisons overlap the
      Phase 7 harness below and may land there). Already complete: public semantics, validated `axis_index_groups`,
      all-gather variance, `pshuffle`/`pswapaxes` compositions, shared native lowerers, direct composite binder
      fixture, production composite shard-map reachability, and static two-device execution
      (`.tasks/plan_p3k_collective_dimensions.md`).
- [x] Zero-materialization migration of the type-generic transform drivers and transpose boundaries: replaced by
      identity-directed residual materialization
      (`ResidualZeroProvider::materialize_zero_from_residual_sources`); `MaybeZero::materialize_like{,_any}` deleted.
      Remaining *homogeneous* `ArrayType` forward-mode sites still using an exemplar `zero_like` are `slicing.rs`
      (update-slice and dynamic-update-slice), `scattering.rs`, `select.rs`, and `complex.rs`; they are unreachable
      with dynamic shapes only where their operation contract says so (see the static-only dispositions below).
- [x] Stacked-scan dead-output geometry: closed. `transpose_primal_scan` now materializes a dead stacked cotangent by
      naming its runtime quantities one at a time (the length identity rides the runtime length operand, the inner
      extents ride the per-iteration peers), so the `TODO(eaplatanios)` is deleted. Fixture:
      `scan::tests::test_scan_transpose_materializes_dead_dynamic_stacked_cotangent`. This closes the zero-reference
      guard.
- [x] Static-only dispositions for the homogeneous transpose rules: `gathering.rs`, `slicing.rs` (slice and
      dynamic-slice) now *reject* dynamically shaped operands as part of their public operation contract, with exact
      per-operation diagnostics, rule rustdoc, and pinned fixtures. `attention.rs` needed no rejection: its own
      `infer_output_types` already requires static shapes; only the comment's justification was wrong and is
      corrected.
- [ ] Close-out decision on the five permanent homogeneous batching zero/one sites (`padding.rs` mask-input one and
      transpose mask-input zero, `slicing.rs` `batch_by_item_expansion` empty-batch zero, `scan.rs` zero-length and
      accumulator stack zeros): they live in `Context<Type = ArrayType>` with no way to observe a runtime extent, so
      either re-affirm them as the permanent homogeneous baseline at the Phase 1 gate or move their owning rules to
      the mixed family. Do not leave the question implicit.
- [x] Composite carriers for custom-derivative and rematerialization payloads: carriers, landed 2026-08-08. Each of
      `CustomJvpOperation`, `CustomVjpOperation`, and `RematerializeOperation` gained a `nondifferentiated_count`
      leading-operand group (the `LinearCallOperation::residual_count` shape, and the direct analogue of JAX's
      `nondiff_argnums`) that carries the batched mapped extent, plus composite `ArrayIrOperation::{CustomJvp,
      CustomVjp, Rematerialize}` variants over `ArrayIrType`. The former by-name rejections in interpretation,
      projected binding, batching, JVP, and transposition are gone, and the projected `ArrayIrOperation::Array`
      variant now holds only region-free operations. This closes the verification-matrix row.
- [ ] Verify the 2026-08-08 scatter `MemberDifferentiableOperation` rule fully closes the former "mixed scatter
      reaches the homogeneous rule through the catch-all" gap (dynamic-shape acceptance test per the archive Phase 6
      owner item), then record the closure.
- [x] Naming-endgame close-out: all four rename/delete sub-items are executed (reshape expression deletion, the
      `DynamicBroadcast`/`DynamicReshape` operation and capability renames, and the freed homogeneous names).
      Confirmed 2026-08-08: repository-wide searches under `crates/` for `LegacyBroadcast`, `LegacyReshape`,
      `legacy_broadcast`, `legacy_reshape`, the abandoned `BroadcastTo`, `HomogeneousBroadcastOperation`,
      `ReshapeDimensionExpression`, and `DimensionExpressions` are all empty, so the endgame is ticked.
- [ ] Owner design call (do not resolve silently): the P0 baseline's congruence-transfer prover layer does not exist
      in the current `AbstractDimensionValue` (interval + exact + same-variable only), so e.g.
      `require_divisible_by(n*4, 2)` residualizes a runtime assertion the archive proved statically. Either reinstate
      congruence transfer in abstract values and arithmetic bound inference and restore the P0 probe table (3–5
      days), or amend the carries-over ledger to drop congruence and record the residual-assertion delta explicitly.
      This decision also gates the "exact diagnostics match the baseline" row below.
- [ ] Exact-diagnostics-match-baseline verification row: after the congruence decision, sweep the frozen P0
      diagnostic templates against current behavior and record the outcome.
- [ ] Gate: core language semantics no longer appear to be backend implementation details, and every item above is
      closed.

## Phase 2: persistence and measured performance closure (archive Phase 10)

- [ ] Verify cache keys remain invariant to runtime extents and distinguish semantically different dimension graphs.
      The behavior pinned by fixtures at both levels (region interning and retained JIT) is: two calls differing only
      in the *runtime extent* of a dynamic dimension share one specialization, while independently created same-named
      `DimensionVariable`s and permutations of live identities each take their own — dimension identity is nominal
      (`Arc::ptr_eq`), so alpha-equivalent-but-distinct programs intentionally do *not* share a cache key. This item
      is the final sweep over the remaining persistence surfaces under those semantics.
- [ ] Re-run the Phase 0 graph-size, allocation, compile-time, memory, executable-size, and runtime measurements.
- [ ] Gate: no performance regression exceeds the existing evidence-based thresholds without explicit approval.

## Phase 3: deletion and minimality gate (archive Phase 11)

- [ ] Delete every item in the archive plan's deletes ledger.
- [ ] Treat `u/eaplatanios/wip/dimensions-remainder` as retired historical state at `12398a196`; prove final
      completeness from the current integration tree, this plan's residual searches, and the completed 142-path
      archive-disposition table (`.tasks/dimensions_archive_disposition.md`). Do not merge the stale remainder or
      alter the immutable archive.
- [ ] Verify `origin/u/eaplatanios/archive/dimensions-wip-2026-07-24` still points to the recorded bootstrap commit
      (`770e77d00`).
- [ ] Record a final current-tree review entry (in this file; the archive plan's historical ledger is already
      annotated as retired and must not be backfilled).
- [ ] Remove dead imports, helper traits, macros, tests, documentation, and allowances made obsolete by the cleanup.
- [ ] Run a repository-wide residual search for retired identifiers and classify every match.
- [ ] Compare production/test/generated line counts against the Phase 0 baseline. Require: at least a 40% reduction
      in the combined non-test source of the array-program projection, batching, and differentiation adapters (if
      smaller, stop for architectural review rather than declaring success from passing tests); a final total
      production line count below the Phase 0 baseline (test additions reported separately); zero hidden
      reconstruction paths; and zero dual semantic operation contracts.
- [ ] Run `cargo fmt --all -- --check`, `git diff --check`, the core/macro/XLA focused suites, all doctests affected
      by moved public APIs, and the full workspace all-target suite serially.
- [ ] Review the final diff by subsystem and ask whether every remaining changed line is necessary for the target
      semantics.
- [ ] Gate: a staff-level review confirms simpler dependency direction, one source of truth, no compatibility layer,
      no redundant abstraction, and no unexplained bloat.

## Phase 4: close tier-3 semantics around the `dimension_from_scalar` gateway (archive Phase 12)

Begin only after the Phase 3 gate: tier 3 is a provenance-policy relaxation over a *stable* architecture. The design
bet is that tier 3 requires no new type-system, program, or transform machinery — the gateway landed in P3d with
eager bounds-checked execution, PE fold/residualize behavior, the mapped-batching rejection, a decided
`Effect::OrderedAssertion` effects model, and differentiation through the unchanged linear-call residual contract
(all pinned by fixtures; see the archive plan's Phase 12 and verification-matrix tier-3 rows).

- [ ] Add a retained-JIT cache-identity test proving one compiled specialization serves multiple runtime extents of a
      *data-derived* dimension (the pinned fixture covers input-derived extents; this one must source the extent from
      the gateway).
- [ ] Verify gateway-defined variables need no boundary `TypeRefinements` entry: they are internal identities
      established by their producing instruction under the existing structural-closure rules. Cover closure, import,
      alpha-renaming, and repeated splicing.
- [ ] Batching: pin the tier-3 MVP policy with fixtures — a replicated scalar operand produces ordinary replicated
      dimension authority; a mapped operand keeps its exact typed rejection diagnostic, updated to name Phase 6
      raggedness as the missing capability.
- [ ] Control flow: verify the existing carry-type equality checks reject shape-varying loop-carried state with exact
      diagnostics (a fresh per-iteration variable cannot instantiate the declared carry type). Bounds-widened
      loop-carried extents are an explicit non-goal; record the rejection fixture rather than designing widening.
- [ ] Confirm the Phase 8 authoritative operation declaration covers the gateway (generated dispatch, conversions,
      classification); the closed-family classifier test already asserts exactly one gateway variant, so this is
      likely a verification tick.
- [ ] Update the `DimensionType` motivation rustdoc in `types/dimensions.rs` so the provenance story describes the
      tiers and names the gateway as the single data-to-dimension boundary; update the `ArrayIrType` cross-reference
      if its wording changes.
- [ ] Add JAX comparison fixtures for eager and staged `n = count(mask); take(x, n)`-shaped programs that JAX rejects
      (`ConcretizationTypeError`) and Ryft accepts eagerly and stages symbolically. Compiled execution may reject
      with an exact "requires Phase 5 bounded data-dependent lowering" diagnostic until Phase 5 lands. (Overlaps the
      Phase 7 harness; the fixtures may live there.)
- [ ] Gate: tier-3 programs interpret eagerly end to end; staged tier-3 programs type-check, batch (replicated),
      differentiate, and partially evaluate; the bounds check provably survives DCE with deterministic ordering;
      every unsupported surface fails with an exact diagnostic; and no expression trees, side tables, or ambient
      environments were added.

## Phase 5: bounded data-dependent compiled execution (archive Phase 13)

The dominant tier-3 cost and the piece most exposed to backend maturity. The physical model is fixed — bound-shaped
buffers carrying smaller logical extents — but the encoding route is an explicit measured decision. The gateway's own
compiled lowering (range check as an ordered assertion, riding the per-class token chain) is already complete.

- [ ] Decide the compiled route for operations consuming data-derived extents, on measured evidence: (a) XLA bounded
      dynamism through the existing bounded-input ABI, `set_dimension_size`, and `PadToStatic`; or (b) fully static
      bound-shaped StableHLO plus explicit Ryft-generated masks. Prototype (a) first because the ABI exists; record
      CPU and CUDA coverage evidence before committing, and fall back to (b) per backend rather than globally.
- [ ] Require a finite upper bound at the gateway for any program that reaches compilation; reject unbounded
      data-derived dimensions at lowering with an exact diagnostic naming the variable and its bounds.
- [ ] Complete the per-operation padding-discipline inventory started in archive Phase 7 and record it here as the
      authoritative table: padding-oblivious (elementwise, reshape-within-bounds), mask-required (reductions,
      argmin/argmax, cumulative and windowed operations), or zero-padding-required (contractions, convolutions).
- [ ] Implement the padding rules for the supported operation matrix so padding garbage is unobservable in results.
      Every unclassified or unsupported operation must reject lowering of data-derived extents with an exact
      diagnostic naming the operation; silent truncation or garbage propagation is an abort criterion.
- [ ] Run CPU (and CUDA where backend support permits) eager/JIT parity for a data-dependent golden set including the
      Phase 4 fixtures, proving one compiled specialization serves multiple runtime extents.
- [ ] Add a dispatch-time bound-bucketing policy for *input-derived* extents as pure retained-JIT policy: round the
      host-observed extent up to a bucket (e.g., logarithmically spaced), compile one specialization per bucketed
      bound, and pad inputs to the bucket, with the bucket participating in cache identity. Bounds padding waste at
      the bucket ratio in exchange for log-many compilations; no new semantics. Gateway-split bucketing for
      *data-derived* extents is an explicit recorded non-goal (device-born extents would force a stream-stalling
      host readback and program split); revisit only with a measured workload recorded here first.
- [ ] Gate: bounded data-dependent programs compile and execute correctly on supported backends, padding effects are
      unobservable in every supported operation's results, unsupported operations fail before execution with exact
      diagnostics, and the route decision is recorded with its measured evidence. This gate also closes the tier-3
      verification row "every operation without data-dependent lowering support fails with an exact diagnostic".

## Phase 6: ragged batching for data-dependent extents (archive Phase 14)

`vmap` over a data-derived extent yields per-batch-element extents — ragged intermediates, the hard transform case.
Structural advantages to reuse: the recursive batching meta stack composes nested axes already, and the
relaxed-while-predicate work established consumer-owned masking.

- [ ] Confirm and record the concrete motivating workload before implementation so the supported operation surface is
      demand-shaped rather than speculative. If the owner explicitly approves deferral instead, record it here and
      re-scope the tier-3 exit criteria; do not defer silently.
- [ ] Extend `BatchingPolicy` with a ragged mapped-extent representation: a per-element extent vector (dimension SSA
      indexed along the batch axis) plus the declared bound as the packed physical extent, with masks owned by
      consumers. Raggedness lives on the batch carrier only; do not add it to `Type` and do not build a parallel
      batching context/tracer tower.
- [ ] Batching rule for the gateway: a mapped scalar operand yields a ragged mapped dimension instead of the Phase 4
      rejection; replicated behavior is unchanged.
- [ ] Ragged rules for the elementwise blanket, masked reductions, and the shape-carrying `linear_call` carrier
      (batch both attached regions with replicated residual extents and ragged linear operands, reusing the
      swap-stable P6 batching rule). Every operation without a ragged rule keeps an exact typed diagnostic.
- [ ] Prove nested `vmap` over ragged extents composes through the recursive batching meta stack.
- [ ] Control flow: ragged trip counts remain rejected with an exact diagnostic; record the fixture.
- [ ] Gate: the ragged surface covers the recorded workload end to end with static and dynamic tests, every
      unsupported path has an exact diagnostic, and no parallel batching tower or type-level raggedness was
      introduced.

## Phase 7: JAX differential-testing harness

Split out of the verification matrix as its own multi-week work item; the behavioral-parity matrix row stays open
until this lands.

- [ ] Build a differential-testing harness against a pinned JAX build covering the frozen behavioral-parity and
      Ryft-exceeds-JAX cases.
- [ ] Cover the P3k group-aware collective, `pshuffle`, and `pswapaxes` surfaces (behavioral and StableHLO
      comparisons; closes the P3k comparison tail from Phase 1).
- [ ] Cover the tier-3 `n = count(mask); take(x, n)` eager/staged comparisons from Phase 4.
- [ ] Gate: behavioral JAX parity and the Ryft-exceeds-JAX cases are demonstrated by the harness rather than
      asserted.

## Abort and reassessment criteria

Stop the current phase and revise this plan if any of the following occurs (backend/tier-3 subset of the archive
list; the completed-phase criteria are retired with their phases):

- transform policy leaks batching/differentiation hooks back onto `Type`;
- diagnostics regress to generic assertion/type errors;
- a phase increases production code after its temporary coexistence code should have been deleted;
- a broad Rust check causes extreme memory growth; reduce generic obligations before rerunning;
- any backend path restores shape expression evaluation, host readback, or reconstruction;
- a tier-3 phase needs changes to generic type-system, program, projection, or residual machinery beyond the single
  `dimension_from_scalar` gateway and transform-owned policy — that falsifies the bet Phase 4 exists to cash and
  requires a design review, not incremental patching; or
- any supported data-dependent lowering path lets padding garbage become observable or silently truncates logical
  extents instead of failing with an exact diagnostic.

## Exit criteria

The cleanup (Phases 1–3) is complete only when the archive plan's exit criteria 1–18 hold; the load-bearing ones for
the remaining work are: exact behavior, diagnostics, cache identity, ABI, CPU/CUDA execution, and performance gates
pass (16); production code is materially smaller with the special-purpose adapter modules reduced by at least 40%
(17); and no hidden reconstruction paths or dual semantic operation contracts remain (15, and Phase 3's gate).

Full tier-3 dynamism (Phases 4–6) is additionally complete only when:

1. Data-derived dimension authority exists through exactly one checked gateway with mandatory bounds and ordered
   assertion semantics, and remains unrepresentable everywhere else.
2. Tier-3 programs interpret eagerly, stage, batch, differentiate, and partially evaluate with unchanged type-system,
   program, and residual machinery — the gateway is the only new operation.
3. Bounded data-dependent programs compile and execute on supported backends with padding effects unobservable in
   every supported operation's results.
4. Ragged batching covers the recorded motivating workload (or its deferral is explicitly owner-approved and
   recorded), with exact diagnostics on every unsupported path.
5. No tier introduced expression trees, witnesses, scopes, substitution, side tables, or ambient environments.

## References

- Complete history, log entries, delivery ledger, and closed verification matrix:
  `.tasks/plan_symbolic_dimensions_architecture_cleanup_archive_2026-08-08.md`.
- Frozen P0 evidence and diagnostics: `.tasks/dimensions_p0_evidence.md`.
- Archive-path disposition table (142 paths): `.tasks/dimensions_archive_disposition.md`.
- Collective parity details: `.tasks/plan_p3k_collective_dimensions.md`.
- Branch obligations: `origin/u/eaplatanios/archive/dimensions-wip-2026-07-24` must remain untouched at `770e77d00`
  until the Phase 3 verification tick; `u/eaplatanios/wip/dimensions-remainder` is retired historical state at
  `12398a196` and must not be merged.

## Review

### Review fixes: replay rejection withdrawn, effectful rendering (2026-08-08)

Two review findings closed. (Long-form record, including the withdrawn item's rationale, is in the archive plan's
entry of the same name.)

**Item 1 — the replay rejection surface is deleted.** `reject_unrefined_operation_payload` was a heuristic rather than
a diagnosis: it rewrote any `ProgramError::Type` raised at an instruction whose declared outputs merely mentioned a
refined identity, without establishing causation, and reported an arbitrary identity when several were refined.
Deleted: the function and its two `map_err` wirings, `ProgramError::UnrefinedOperationPayload`, the defaulted
`TypeRefinements::established_refinement` hook, and its array/composite implementations; a repo-wide search for all
three identifiers is empty. The underlying limitation is now accepted and pinned instead:
`interpretation::tests::test_program_interpret_does_not_refine_operation_payloads` asserts the plain
`ProgramError::Type` and its exact inference-level message, documenting that replay refines only program and region
boundaries, never instruction operation payloads. Future path if the specialized diagnostic is ever wanted: it must
originate from a structured operation error that causally identifies the failing payload constraint.

**Item 6 — zero-output effectful instructions now render (intentional rendering-contract change).** `Program::render`
keyed instructions off their first output atom, so resultless instructions produced no line and semantically different
programs could render identically. The per-instruction body is now factored into a `render_instruction` helper and the
atom walk carries an instruction cursor that flushes not-yet-rendered resultless instructions in instruction order.
Syntax: a region body is a statement sequence; a statement is either the existing `%2:t = operation ...operands`
binding or, with no outputs, `operation ...operands` with the binder omitted. Resultless statements share the same
`let`/blank gutter, so `let` still marks the first statement of the block. No program text is parsed anywhere, so
there is no round-trip to preserve. Exactly two pinned fixtures in the workspace contained zero-output instructions,
both in `operations/dimensions/dimension_requirement.rs`, and both deltas are purely added lines;
`test_dimension_requirement_order_is_deterministic` additionally gained a swapped-requirement program that binds the
same atoms yet renders differently.

Gates (all `CARGO_INCREMENTAL=0`): `cargo check --workspace --all-targets` clean, zero warnings; `ryft-core --lib`
1,199 passed / 3 failed, the failures being the pre-existing `test_array_constants` plus the two new
`test_composite_{condition,scan}_pullback_shapes_a_dead_dynamic_*_cotangent_from_a_live_peer` fixtures red from a
concurrent zero-materialization refactor (their deltas are `zero_like` versus `dimension_size` + `zero` staging, with
no resultless instruction involved); `ryft-core` integration binaries 6 + 6; doc-tests 53 passed / 16 ignored;
`ryft-xla --lib` with `--test-threads=1` 442 passed / 1 ignored; `ryft-macros-tests` 20 + 17. Per-file
`rustfmt --edition 2024 --check` clean on every edited file. The verification matrix needed no rewording: no row
states a rendering contract, and the "shape dependency in rendered IR is an operand edge" gate is unaffected because
resultless instructions only add lines.

### Review fixes: type-general zeros, scan transposition, contracts, batching policy (2026-08-08)

**Mechanism (review item 2).** Verified the enabling invariant: `ArrayType::{tangent,cotangent}` rebuild only
`data_type`, `layout`, and `sharding` and carry `shape` through `..self.clone()`, so a differential type has exactly
its primal's geometry. Exemplar matching was therefore wrong for the *whole-type* reason, not the geometry reason, and
the fix is to name the runtime quantity instead of matching the type. Chose the existing differentiation-owned
residual protocol over a shape-directed `ZeroLikeOperation`: it already declares, captures, and spends per-identity
dimension operands through the mixed `ArrayIrOperation::Zero` constructor, which is already interpreted and lowered,
whereas a target type stored on `zero_like` would have needed matching interpretation and XLA lowering work.
`ResidualZeroProvider` gained one required-ish primitive, `capture_zero_residual_value` (per declared residual, one
candidate, `None` without side effects when the candidate does not name it), and one boundary form,
`materialize_zero_from_residual_sources`. `MaybeZero::materialize_like{,_any}` are deleted; `MaybeZero` keeps only the
type-only `materialize`. Migrated sites: `condition.rs` (jvp + transpose), `scan.rs` (jvp + both transposes),
`while.rs` (fused and bounded jvp), `differentiation/linear.rs` (jvp + transpose), `differentiation/reverse.rs`
(auxiliary cotangents), `tracing_v2/custom_derivatives.rs` (custom-JVP and custom-VJP), `tracing_v2/rematerialization.rs`,
and `materialize_array_tangent`, which now stages through the mixed parent context so widened tangents work at dynamic
shapes.

**Scan (item 3).** No architectural corner remained; the boundary already names every quantity collectively, it just
had no way to combine them. Unmet quantities now fail with an exact `UnsupportedOperation` naming the zero type and the
missing residual instead of the constructor's generic diagnostic.

**Batching policy (item 5).** `LinearCallBatchingPolicy` renamed to `CotangentBatchingPolicy` and re-documented around
the shared capability (adapting a batched *backward* region); `CustomJvpOperation` relaxed to plain `BatchingPolicy`.
Repo-wide search for the old identifier is empty. The user concurrently moved the trait to
`differentiation/batching.rs`, which the rename survived.

**tracing_v2.** `check_nondifferentiated_tangents_are_zero` and the three `split_inputs` bodies moved to a new private
`tracing_v2::operands` module; `rematerialization` no longer depends on `custom_derivatives`. Nothing else there moved.

**Rendering deltas (all dynamic-path, all intentional).** The two
`test_composite_*_pullback_shapes_a_dead_dynamic_*_cotangent_from_a_live_peer` fixtures change from `zero_like %peer` to `dimension_size` + mixed `zero`, which is the
whole point of the mechanism change. `materialize_array_tangent`'s dynamic case likewise. No static-path rendering
moved.

**Deferred (not mine to edit).** `ryft-xla`'s `XlaOperation` inherits the defaulted `capture_zero_residual_value`, so
multi-source assembly is unavailable there until it delegates to `ArrayIrOperation` like its sibling methods do
(`crates/ryft-xla/src/experimental/ops.rs`, next to the existing `capture_zero_residual_values` delegation). Every
single-source case already works there. One cross-file unblock was taken instead of deferred: `Atom` gained
`PartialEq`, which a concurrent `dimension_requirement.rs` test required and which no other agent could add.

**Gates** (all `CARGO_INCREMENTAL=0`): `cargo check --workspace --all-targets` clean with zero warnings;
`ryft-core --lib` 1,204 passed / 1 failed (the pre-existing `test_array_constants`); `ryft-core` integration binaries
6 + 6 and doc-tests 53 passed / 16 ignored; `ryft-xla --lib --test-threads=1` 442 passed / 1 ignored;
`ryft-macros-tests` 20 + 17. Per-file `rustfmt --edition 2024 --check` clean on every edited file.

### Review-fix wave closed (2026-08-08)

**XLA zero-residual delegation (closes the deferral recorded in the entry above).** `XlaOperation` now delegates
`ResidualZeroProvider::capture_zero_residual_value` to `ArrayIrOperation::<Capture::Projected>`, beside its
`zero_residual_types`, `capture_zero_residuals`, and `zero_operation_with_residuals` delegations, so identity-directed
multi-source geometry assembly is reachable in the XLA family; the inherited default answered `None` for every
candidate and made `materialize_zero_from_residual_sources` report each declared residual as unsupplied. The plural
`capture_zero_residual_values` is deliberately *not* delegated: `ArrayIrOperation` does not override it either, so
both families reach the same trait default, which now loops through the delegated singular hook. Two fixtures in
`crates/ryft-xla/src/experimental/ops.rs` pin it:
`test_xla_residual_zero_provider_materializes_dynamic_zero_from_array_source` (one source whose element
representation differs from the zero's, proving the geometry-not-type match) and the new
`test_xla_residual_zero_provider_assembles_dynamic_zero_from_several_sources` (an `f32[rows, columns]` zero assembled
from three candidates: a statically shaped array that names neither identity and stages nothing, a dynamic
`f8e8m0fnu[rows]` array that supplies the row extent through `dimension_size`, and a first-class `columns` dimension
that supplies the column extent as itself; the staged instruction sequence and both `zero` operands are asserted).

**Three staleness corrections in this file.** (1) The naming-endgame close-out is ticked: all four archive sub-items
are `[x]` and repository-wide searches under `crates/` for `LegacyBroadcast`, `LegacyReshape`, `legacy_broadcast`,
`legacy_reshape`, the abandoned `BroadcastTo`, `HomogeneousBroadcastOperation`, `ReshapeDimensionExpression`, and
`DimensionExpressions` are all empty. (2) The composite-carrier item no longer says custom derivatives and
rematerialization are rejected by name: they gained `nondifferentiated_count` leading-operand groups and composite
`ArrayIrOperation` variants on 2026-08-08, so the item is ticked with carriers as the outcome. (3) The Phase 2
cache-identity item now states the pinned behavior directly — invariance to the *runtime extent*, with independently
created same-named `DimensionVariable`s and permutations of live identities each taking their own key, because
dimension identity is nominal — instead of deferring to the invariants section for the correction. The corresponding
archive verification-matrix rows were left untouched as history.

**Gates** (all `CARGO_INCREMENTAL=0`): `cargo check --workspace --all-targets` clean with zero warnings;
`ryft-core --lib` 1,204 passed / 1 failed (the pre-existing `test_array_constants` `Fill` diagnostic wording drift,
red at `HEAD` and untouched here); `ryft-core` integration binaries 6 + 6 and doc-tests 53 passed / 16 ignored;
`ryft-xla --lib --test-threads=1` 443 passed / 1 ignored (the 442 baseline plus the new multi-source fixture);
`ryft-macros-tests` 20 + 17. `rustfmt --edition 2024 --check` clean on the one edited source file. No rendered
fixture changed.
