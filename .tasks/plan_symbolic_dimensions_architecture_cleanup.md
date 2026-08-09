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

- [x] P3k collective-parity tail: closed with one four-device production `shard_map` that executes grouped tiled
      all-gather, psum-scatter, and all-to-all together and pins the exact ordered StableHLO replica groups and
      per-device results. A separate four-device fixture executes one bounded-dynamic program that declares three
      dynamic extents at two distinct runtime shapes sharing one cached executable, through the checked physical ABI.
      The backend boundary is explicit: pinned Shardy rejects dynamic tensors, so static programs use Shardy SPMD,
      fully replicated bounded-dynamic programs use one replica per device, and non-replicated bounded-dynamic
      multi-device placement is rejected exactly. Current JAX 0.6.2
      comparisons pin the three grouped collectives plus `pshuffle` and `pswapaxes`; JAX's public collective path is
      static, so Ryft's replicated bounded-dynamic execution exceeds that supported surface
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
- [x] Close-out decision on the five homogeneous batching zero/one sites: retain `padding.rs`'s mask-input one and
      transpose mask-input zero, `slicing.rs`'s `batch_by_item_expansion` empty-batch zero, and `scan.rs`'s zero-length
      and accumulator-stack zeros as the permanent homogeneous baseline. Their `Context<Type = ArrayType>` contract
      deliberately has no first-class extent operands; the corresponding mixed `ArrayIrType` rules own dynamic
      materialization. Moving these sites would couple the homogeneous reference path to the mixed carrier without
      making any currently unsupported dynamic program representable.
- [x] Composite carriers for custom-derivative and rematerialization payloads: carriers, landed 2026-08-08. Each of
      `CustomJvpOperation`, `CustomVjpOperation`, and `RematerializeOperation` gained a `nondifferentiated_count`
      leading-operand group (the `LinearCallOperation::residual_count` shape, and the direct analogue of JAX's
      `nondiff_argnums`) that carries the batched mapped extent, plus composite `ArrayIrOperation::{CustomJvp,
      CustomVjp, Rematerialize}` variants over `ArrayIrType`. The former by-name rejections in interpretation,
      projected binding, batching, JVP, and transposition are gone, and the projected `ArrayIrOperation::Array`
      variant now holds only region-free operations. This closes the verification-matrix row.
- [x] Verified the 2026-08-08 scatter `MemberDifferentiableOperation` rule closes the former "mixed scatter reaches
      the homogeneous rule through the catch-all" gap. `test_array_ir_scatter_differentiation` exercises the mixed
      rule, and `test_array_ir_dynamic_scatter_disconnected_operand_tangent_uses_runtime_extent_residuals` pins a
      disconnected dynamically shaped operand tangent materialized from ordinary runtime-extent residuals.
- [x] Naming-endgame close-out: all four rename/delete sub-items are executed (reshape expression deletion, the
      `DynamicBroadcast`/`DynamicReshape` operation and capability renames, and the freed homogeneous names).
      Confirmed 2026-08-08: repository-wide searches under `crates/` for `LegacyBroadcast`, `LegacyReshape`,
      `legacy_broadcast`, `legacy_reshape`, the abandoned `BroadcastTo`, `HomogeneousBroadcastOperation`,
      `ReshapeDimensionExpression`, and `DimensionExpressions` are all empty, so the endgame is ticked.
- [x] Owner design call: do not reinstate the retired modular-congruence prover. The retained abstract interpreter's
      exact-value, interval, and shared-identity facts remain the deliberately decidable compile-time layer; modular
      relationships stay ordinary SSA plus ordered runtime requirements rather than recreating a parallel symbolic
      algebra. The exact measured delta is one residual `OrderedAssertion` for `require_divisible_by(n * 4, 2)` where
      the retired prover emitted zero, pinned by
      `test_dimension_requirement_partial_evaluation_retains_unproven_congruence`.
- [x] Exact-diagnostics-match-baseline verification row: all five frozen runtime requirement messages, the bounds
      diagnostic, both type-time impossibility messages, deterministic first-failure ordering, PE preservation, and
      relocation are pinned by the dimension-requirement fixtures. Dropping congruence changes only whether the one
      requirement above is residualized; it does not change its actor-named runtime diagnostic.
- [x] Gate: every phases 0–8 residual is closed, generic core semantics remain independent of XLA implementation
      details, and the only backend-specific limitation is an exact placement rejection at the XLA domain boundary.

## Phase 2: persistence and measured performance closure (archive Phase 10)

- [x] Verify cache keys remain invariant to runtime extents and distinguish semantically different dimension graphs.
      The final contract has two deliberate levels. Region instantiation and retained-JIT dispatch preserve nominal
      `DimensionVariable` identity (`Arc::ptr_eq`), so independently created identities and permutations take separate
      specializations. The persistent compiled-artifact key is computed after lowering and canonicalizes
      diagnostic-only identity names: the same lowered dimension graph under a fresh nominal identity shares an
      artifact, while changed dimension SSA does not. At runtime, two distinct concrete extents execute through one
      retained specialization and one compiled executable.
- [x] Re-run the Phase 0 graph-size, allocation, compile-time, memory, executable-size, and runtime measurements.
- [x] Gate: no performance regression exceeds the existing evidence-based thresholds without explicit approval.

## Phase 3: deletion and minimality gate (archive Phase 11)

Phase 2 preflight: this is not a small follow-on review unit. Current source includes substantial post-baseline
functionality and exceeds the historical Phase 0 total, while this gate demands both an adapter-specific accounting
and a lower final production total. Execute Phase 3 separately: classify growth by subsystem and decide the valid
comparison scope before deleting or silently re-baselining anything.

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

### Phase 1 residual closure (2026-08-08)

**Collectives and multi-device dynamism.** A four-device production `shard_map` now executes grouped tiled
all-gather, psum-scatter, and all-to-all in one program, pins all three ordered replica-group attributes, and checks
the exact result on every device. Current JAX 0.6.2 produces the same grouped results and StableHLO collective forms;
its `pshuffle` and `pswapaxes` compositions likewise match Ryft's existing `ppermute` and untiled-all-to-all
fixtures. A second four-device fixture executes a fully replicated bounded-dynamic array with three pairwise-distinct
logical extents. Pinned Shardy rejects dynamic tensors, so the XLA domain now selects Shardy SPMD only for static
boundaries, selects one executable replica per device for fully replicated bounded-dynamic boundaries, and rejects a
requested non-replicated bounded-dynamic multi-device placement with one exact diagnostic. No collective payload,
core type, or transform contract learned this backend limitation.

**Core decisions.** The five listed homogeneous batching constants remain the permanent `ArrayType` reference
baseline; `ArrayIrType` remains the owner of first-class dynamic extent materialization. Mixed scatter's dedicated
member differentiation path is pinned at a dynamic shape, including a disconnected tangent whose geometry is
supplied by runtime residuals. Modular congruence is deliberately dropped from the carries-over ledger: rebuilding it
would recreate part of the parallel symbolic algebra this architecture removed. The accepted cost is exactly one
residual ordered assertion for `(n * 4) % 2 == 0`, now pinned by a focused partial-evaluation test.

**Diagnostics.** The frozen five runtime requirement messages, bounds error, two type-time impossibility messages,
same-class ordering, PE preservation, and relocation fixtures remain exact. The congruence decision changes assertion
elimination density, not diagnostic wording. The grouped-collective, bounded-dynamic multi-device, mixed-scatter,
congruence, frozen-diagnostic, `pshuffle`, and `pswapaxes` fixtures pass. All 445 runnable `ryft-xla` unit tests pass
(one benchmark ignored); all 53 runnable `ryft-core` doctests pass (16 ignored); and
`cargo check --workspace --all-targets`, nightly formatting, and `git diff --check` are clean. Of 1,207 `ryft-core`
unit tests, 1,206 pass and the pre-existing `test_array_constants` exact-string mismatch remains red at `HEAD`: the
production `fill` diagnostic ends in `instead` while its fixture does not. This Phase 1 slice neither changes nor
depends on that unrelated wording.

### Phase 2 persistence and measured performance closure (2026-08-08)

**Cache identity.** The final fixtures separate the two cache layers instead of conflating them. Core region
instantiation and retained-JIT dispatch remain nominal: exact identities reuse their specialization, while a fresh
`DimensionVariable` takes another. The persistent XLA artifact key is a property of the final lowered program, so it
canonicalizes a fresh diagnostic-only identity name for an otherwise identical graph but changes when the dimension
SSA changes (the pinned pair uses `extent + 1` versus `extent + 2`). A four-device bounded-dynamic executable runs
logical shapes `[4, 2, 3, 5]` and `[4, 3, 4, 6]` through one retained specialization and leaves the domain cache at
exactly one entry. The focused builder, dynamic-zero retained-JIT, persistent-signature, canonical-key, dynamic-graph,
and multi-device fixtures all pass.

**Graph size and allocation.** `program_statistics` reproduces the Phase 0 structural counts for all eight successor
cases: scalar entry regions remain at 3/4/15/18/11 instructions with maximum dependency depths 2/2/7/7/7;
`shard_map_basic` and `shard_map_matmul` remain one entry instruction plus one single-instruction child; and
`nested_shard_map` remains a one-instruction entry, two-instruction intermediate region, and one-instruction leaf.
Raw rendered-IR bytes and lines are intentionally unavailable because the approved statistics migration deleted that
unstable metric. The replacement allocation suite passes 6/6: borrowed and consuming array/dimension projections
allocate zero times; large clones add no payload allocation; and elementwise/constructor kernels retain a constant
allocation count with exactly one output-payload byte slope. The retired expression-allocation harness is not
reintroduced.

**Build and size measurements.** Clean `ryft-core` checking took 12.49 s and 970,489,856 B peak RSS; the immediate
incremental check took 0.18 s and 65,208,320 B. Relative to Phase 0 integration, clean time is +13.0% (inside the 20%
gate) and incremental behavior is flat; clean RSS is +39.3%, while no memory percentage threshold was approved. The
retained release performance harness built in 100.52 s at 2,390,065,152 B, +4.3% time and -27.3% peak RSS versus the
archive emitter build. For the direct tool replacement, two clean `program_statistics` builds took 117.49 s and
116.96 s at roughly 2.407 GB; this informational tool-build comparison is +21.4% versus the retired emitter, but it is
not the timed trace/lower/compile acceptance workload. Its executable is 45,292,720 B versus the corrected Phase 0
integration emitter's 57,564,144 B (-21.3%).

**Runtime.** Across five release runs of 50 iterations at 1,024 elements, the median run recorded cold trace/lower/
compile of 74,334/618,209/16,579,166 ns; warm dispatch p50/p95 of 6,333/9,250 ns; enqueue p50/p95 of 5,792/8,834 ns;
and synchronized p50/p95 of 9,333/25,042 ns. Every value improves on the Phase 0 integration smoke baseline
(485,500/3,919,958/47,597,583 ns; 11,042/27,583 ns; 10,500/51,875 ns; 38,334/49,584 ns), so no runtime threshold is
approached.

**Verification.** `ryft-xla --lib --test-threads=1` passes 445 tests with one timing benchmark ignored. The Phase 1
full-core and doctest results remain applicable because Phase 2 changes only XLA fixtures and the plan: 1,206 of 1,207
core unit tests pass, with only the pre-existing `test_array_constants` exact diagnostic mismatch red at `HEAD`; all
53 runnable doctests pass (16 ignored). The projection allocation suite passes 6/6. Workspace all-target checking,
nightly formatting, and diff hygiene are clean.

**Phase 3 sizing decision.** Phase 3 was not started in this combined review. Current `tokei` code counts are 116,431
for `ryft-core/src`, 39,690 for `ryft-xla/src`, and 5,977 for `ryft-macros/src`, versus Phase 0 integration counts of
78,838, 34,271, and 4,766. Those totals include substantial post-baseline functionality, but the Phase 3 gate still
requires a lower total plus an adapter-specific 40% reduction. That makes Phase 3 a material deletion and accounting
review, not a small cleanup suitable for appending here. A preliminary retired-identifier search is clean in `crates/`
for `ArrayProgramProjection`, both context views, replay/source helpers, the old dimension-expression/lowering
machinery, `SymbolValueResolver`, and all legacy broadcast/reshape names; the remaining plan-text mentions are
historical requirements rather than code residuals.

### Shardy-guard coverage fixes (2026-08-08)

**Shardy rejects dynamic tensors everywhere, not only at the boundary (measured).** The `use_shardy_partitioner`
predicate in `XlaDomain::lower_xla_program` only inspected the input and output types, so a program with a fully
static boundary but a gateway-derived dynamic interior still enabled Shardy. Compiling exactly such a program on a
four-device CPU mesh fails with `Shardy propagation only supports ranked tensors with a static shape. type:
'tensor<?xf32, #stablehlo.bounds<4>>'` on the `stablehlo.set_dimension_size`. The predicate is therefore now the
whole-program walk `has_only_static_array_types`, which iterates the region arena (covering every region boundary,
constant, and instruction result, recursively through attached regions) and ignores first-class dimension atoms
because those lower to ordinary scalars. The old boundary check was dropped rather than kept alongside it: the entry
region's atoms already subsume the boundary types. Fixture:
`test_internal_dynamic_tensor_behind_a_static_boundary_takes_the_replica_path` pins the replica path (4/1, no SPMD,
no Shardy, no `sdy.` in the module) and correct execution, *and* pins the counterfactual by re-lowering the same
program with Shardy annotations and asserting the compile failure above.

**`shard_map` escapes the replicated-sharding guard (upstream rejection is narrower than it looks).** Dynamic
`shard_map` *boundaries* are rejected upstream at `ShardMap::trace` via `static_dimensions`
(`crates/ryft-xla/src/experimental/shard_map.rs`), unconditionally and before any sharding is consulted, so even a
fully replicated spec with a dynamic extent fails with `input type #0 dimension #0 must be static for traced
shard_map` (pinned by `test_shard_map_trace_rejects_dynamic_input_types`). That does *not* close the hole: because a
`shard_map` body is always static, it can sit beside dynamic tensors staged elsewhere in the same program, whose
replicated boundary shardings pass the existing guard. Its `sdy.manual_computation` lowering would then be compiled
with `use_shardy_partitioner = false`, and its collectives address partitions rather than replicas. The dynamic
multi-device guard therefore gained a second, separately worded rejection driven by `contains_shard_map`:
`bounded-dynamic multi-device programs cannot contain shard_map regions because sdy.manual_computation requires the
Shardy partitioner`. Fixture: `test_bounded_dynamic_multi_device_programs_reject_shard_map_regions`. Named-axis
collectives need no companion guard: mesh-bound collectives only lower inside a manual region (`axis_index` and the
mesh replica-group derivation both require `collective_state.manual_shard_map()`), so they are unreachable outside
the case just rejected.

**`Sharding::is_replicated`.** The guard compared against a freshly allocated `Sharding::replicated(...)` per element.
`crates/ryft-core/src/arrays/sharding/shardings.rs` now owns a documented `is_replicated` (all dimensions
`Replicated` and all three auxiliary axis sets empty), stating why reduction and manual-axis metadata are
deliberately *not* replicated and why `Unconstrained` is not either. It replaces both the guard comparison and the
duplicated private `is_fully_replicated` in `crates/ryft-xla/src/arrays_v0/compiled_reshard.rs`. Fixture:
`test_sharding_is_replicated`.

**Gates** (all `CARGO_INCREMENTAL=0`): `cargo check --workspace --all-targets` clean with zero warnings; `ryft-core
--lib` 1,207 passed / 1 failed (the pre-existing `test_array_constants` `Fill` diagnostic wording drift, red at
`HEAD` and untouched here; +1 test from `test_sharding_is_replicated`); `ryft-xla --lib --test-threads=1` 447 passed
/ 1 ignored (+2 tests, both new fixtures above); `ryft-macros-tests` 20 + 17. `rustfmt --edition 2024 --check` clean
on the three edited source files. No rendered fixture changed.
