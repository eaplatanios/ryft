# Reference Discharge Architecture Plan

This plan converges reference discharge with Ryft's established transform architecture — the fused
interpreter-plus-per-operation-rule shape used by differentiation, batching, and partial evaluation, which is also
JAX's shape for `discharge_state` — and adds first-class partial discharge. It deliberately separates that
architectural change (Track A, approved direction) from the riskier view-representation question (Track B, a
prototype-gated decision), because bundling them destroys the migration oracle: the current planner and the pinned
renderings are only a usable parity baseline while source programs keep their present shape.

Nothing in the existing reference machinery is treated as committed, but nothing is deleted until the replacement
has proven exact parity.

## 1. Motivation

Discharge is the outlier among Ryft's transforms. Differentiation, batching, and partial evaluation are
interpreter-style replays: a generic context and driver plus per-operation rules, with enum dispatch derives
forwarding backend operation families onto shared implementations. Discharge instead uses a centralized planner and
a bespoke replay, which forces provenance contracts (so the planner can widen boundaries for operations it does not
understand) and their validation against lying third-party families. When each structured operation discharges
itself — as it already batches and differentiates itself — the planner-specific discharge *layout contracts*
disappear (generic region provenance remains, with its independent consumers). The planning logic itself does not
vanish; it separates into three named responsibilities: a summary analysis (root access and transitive mutation
facts, computed generically from `reference_semantics`), the per-operation rules (which decide and emit the
rewrite), and driver services (checked summaries, transactional forks, unions, and boundary assembly).

The interpreter shape also makes partial discharge natural: unselected roots keep reference values whose accesses
replay verbatim, which is what the future kernel pipeline needs (JAX precedent: `should_discharge` in
`jax._src.state.discharge`, used by Pallas to discharge pipeline state while keeping kernel references live).

## 2. Track A: interpreter discharge

### 2.1 Contracts (prototyped and type-checked before any migration)

**Results.** Two envelopes with distinct guarantees:

- `ReferenceDischargeResult<P>` keeps its current contract unchanged: a *proven reference-free* payload with the
  public-prefix/hidden-suffix boundary invariants. `discharge_local_references` continues to return this contract
  and continues to reject surviving external roots.
- `PartialReferenceDischargeResult<P>` is new: a mixed payload in which selected roots were discharged (discharged
  *external* roots are reported as `ReferenceStateBinding`s; discharged local allocations leave no binding) and
  unselected roots survive as well-typed reference values. There is deliberately no generic partial-to-full
  conversion: "all roots were selected" proves nothing against a malformed provider. `try_into_full` is implemented
  only where `P` is a `Program<..>`, and it performs the actual reference-freedom proof: no reference type in any
  boundary, atom, or constant across all nested regions, and no operation with nonempty reference semantics
  anywhere in the closure. The proof is deliberately reference-specific: discharge is not a general
  state-purification pass, so an unrelated third-party ordered-state operation passes through untouched (pinned by
  a test), and downstream consumers keep their own ordered-state gates. Non-`Program` payloads keep the
  provider-owned proof obligation the `ReferenceDischarge` trait already documents.

**Root selection.** Selection needs a pre-transform identity; interpreter-emergent identity is only for threading.
Selection gets its own checked vocabulary rather than reusing `ReferenceRoot` directly, because nested
`ReferenceRoot::RegionInput` roots are invocation-parameterized formal parameters (the analysis contract says so
explicitly) and must not be independently selectable:

```rust
/// One caller-selectable reference site for partial discharge.
pub enum ReferenceDischargeSite {
    /// Entry-boundary root: a capture or public reference argument.
    External(ReferenceSource),

    /// Interior allocation site, required by the kernel pipeline.
    Allocation {
        instruction: InstructionId,
        output_index: usize,
    },
}
```

Sites resolve internally to roots; selection is arena-relative like every other root-bearing artifact. Selection
sets are validated against the program before replay: every named site must exist, name a reference-allocating
operation (or a reference-typed entry position), have a valid output index, and appear at most once. Because
instruction identities are coordinates, a site taken from another program arena that happens to name a valid
allocation here is undetectable in principle; validation catches every kind mismatch, and the arena-relativity
contract carries the rest, exactly as it does for the analysis artifacts. Callers enumerate sites through a
dedicated lightweight query (entry reference positions plus allocation-rule instructions) that is independent of
the full analysis, so it survives the phase-6 reduction. Tests cover identical allocation operations at distinct
sites, non-allocation and out-of-range sites, duplicate sites, and allocation inside a shared nested region.

**The family capability.** One trait owns every family-varying type, on the batching-policy precedent
(`ArrayBatching`/`ArrayIrBatching` are exactly this shape), so the value, context, driver, and rule signatures all
name a single family parameter instead of loose generics — and so a non-array family is a first-class instantiation
rather than an afterthought:

```rust
/// Family capability naming the types one reference universe threads through discharge.
pub trait ReferenceDischargeFamily<C> {
    /// Referent type system of this family's references.
    type Referent: Type;

    /// Composed alias metadata carried by one flowing handle (the view chain, for arrays; `()` for
    /// view-less families).
    type Alias: Clone + Parameter;

    /// Embeds a reference type into the destination type universe.
    fn embed_reference_type(r#type: ReferenceType<Self::Referent>) -> C::Type;

    /// Projects a destination type back onto a reference type, when it is one. The embed/project pair is the
    /// conversion seam access rules use to type-check operands.
    fn project_reference_type(r#type: &C::Type) -> Option<ReferenceType<Self::Referent>>;
}
```

**Values and environment.**

```rust
/// One value flowing through a reference-discharge replay.
pub enum ReferenceDischargeValue<C, F: ReferenceDischargeFamily<C>> {
    /// Ordinary pure value, replayed as-is.
    Pure(C::Value),

    /// Handle to one live root, mirroring the eager `ArrayReference` shape: root identity, the composed
    /// family-owned alias metadata, and the derived reference type this exact handle exposes. Root identity alone
    /// cannot implement `Typed` or tell an access rule which composed view to apply while Track A retains SSA
    /// views. The per-root state lives in the context environment so every handle to one root observes every
    /// ordered update.
    Reference {
        root: ReferenceRootHandle,
        alias: F::Alias,
        r#type: ReferenceType<F::Referent>,
    },
}

/// Environment entry for one live root.
enum RootState<A> {
    /// Selected for discharge: threads as immutable state.
    Discharged {
        /// Current immutable state value.
        current: A,
        /// Whether any ordered write or accumulation has occurred (drives hidden-output and widening pruning).
        mutated: bool,
    },

    /// Not selected: survives as a destination reference value whose accesses replay verbatim.
    Preserved {
        /// Destination reference-typed value.
        reference: A,
    },
}
```

The flowing value is a full `Value` family (Section 8, decision 2), with the reference arm reporting the exact
handle type (which differs from the root type under a composed view); the exact impl set, the minimal family
bounds, and a verified non-array test family are phase-0 prototype deliverables, not implementation details to
discover late.

**Rule trait and bounds.** The per-operation rule follows the transform-rule shape, and the discharge context
implements `Context` (Section 8, decision 2), so discharge runs through `interpret_in_context` like batching and
differentiation and rules can bind through the context directly:

```rust
pub trait ReferenceDischargeableOperation<C, F: ReferenceDischargeFamily<C>>: Operation {
    fn discharge<D: ReferenceDischargeDriver<C, F>>(
        &self,
        context: &ReferenceDischargeContext<C, F>,
        driver: &D,
        inputs: &[ReferenceDischargeValue<C, F>],
    ) -> Result<Vec<ReferenceDischargeValue<C, F>>, ProgramError>;
}
```

The driver additionally exposes the replay position — `fn instruction(&self) -> Option<InstructionId>` — because an
allocation rule must know its own site to test selection membership (Blocker: the existing region drivers expose
attached regions but not the current instruction). Program replay always supplies the identity; a direct
`Context::bind` with no source instruction returns [`None`], and such allocations are always discharged — no site
can name them, and rejecting them instead would make direct binding unusable.

with three constraints that are exit criteria for the phase-0 prototype, not afterthoughts: the rule trait's
context parameter carries capability-only bounds (never `C: Context` on the trait — the pattern that keeps the
dispatch derives clear of trait-solver recursion); the default pure-replay rule's conversion from `Self` into
`C::Operation` uses the established conversion seam and its `Self::Type`/`C::Type` relationship is explicit; and
`ReferenceDischargeValue` proves out as a real `Value` family on the `ArrayIrValue` heterogeneous-value precedent
(capabilities delegate on `Pure`, error on `Reference`, and the reference arm types as `ReferenceType`).

**Driver services, not contracts.** Structured rules (condition, while, scan, call) own their widening, but the
planning-shaped logic they share is provided once by the driver, as services rules compose — the same division the
batching and partial-evaluation drivers already use:

- transactional region forks: structured replay runs against an environment *snapshot* and returns the region's
  outputs and final states without committing anything to the parent environment; only the owning rule merges the
  returned states. This is a hard isolation contract, not `Context` cloning (stateful context clones share active
  transform state, which is exactly wrong here): both condition branches must observe the same entering
  environment, speculative while probes must commit nothing, and a failed replay must leave the parent environment
  untouched. Discharge needs this where batching and differentiation do not because its state lives in the context
  environment rather than riding in per-value tracers;
- transitive closure summaries: which roots a region closure reads, writes, or accumulates, which a while rule
  must know *before* widening. This is retained reference analysis, acknowledged as such, and its inputs are named
  precisely: operation-local `reference_semantics`, input-region provenance, output-region provenance,
  reference-output identity, and recursively computed summaries of nested regions — all existing generic hooks.
  The summary reports accesses in the region's own formal terms, and the owning rule translates formal to caller
  roots itself — the rule owns its operand-to-region-input mapping because it is the operation. Third-party
  structured operations therefore need no companion declaration surface beyond the hooks they already implement;
- deterministic root union and ordering for threaded state;
- the while condition/body interplay (condition observes entering state, produces none) and the scan
  carry-position arithmetic;
- captured-root substitution across call boundaries;
- boundary assembly and validation for the rewritten operation.

Rules remain able to produce malformed programs, exactly as jvp and batching rules can; the house answer stays
builder-time type checking, the result envelopes, and oracles — not per-operation contracts.

### 2.2 Placement and openness

- Reference primitives implement their own rules (allocation binds a fresh discharged or preserved root; read,
  swap, and add-update act on `Discharged` state through the shared view machinery or replay verbatim on
  `Preserved` roots; freeze yields the current state and unbinds).
- Structured operations implement their own widening through the driver services, including the read-only pruning
  policy (no synthesized output or condition/call widening for roots a closure only reads; loop-shaped rules keep
  boundary symmetry).
- Backend enums forward through a dispatch derive as for differentiation and transposition; third-party operations
  implement the trait directly. The system is open over primitives.
- SSA view operations are retained in Track A, with their own rules: `reference_index` and `reference_slice`
  preserve the incoming root handle, validate and compose the transform onto the flowing `alias` (rejecting invalid
  composition before binding the output, with the same math as eager `with_transform`), and record the derived
  reference type. The flowing `alias` field is the single authoritative view chain during discharge; the analysis
  never supplies view resolution to the transform (it remains the lint's and the summaries' concern), so a handle's
  view has exactly one source of truth.

### 2.3 What survives, with consumers named

- The runtime ABI and holder machinery (leases, generations, completions, poisoning): untouched.
- `ReferenceStateBinding`, hidden-suffix ordering, V6 persistence: untouched.
- Provenance hooks (`output_region_provenance`, `input_region_provenance`, `reference_output_identity_input`):
  retained — rematerialization, linear-call, and differentiation consume them independently of discharge. Only
  uses that are provably planner-exclusive may be reconsidered at cutover, hook by hook, with the surviving
  consumers listed in the phase summary.
- The preserved-reference kernel path: remains a distinct validator and lowerer that *consumes* partial discharge
  (normalize pipeline state, keep kernel references), retaining its own eligibility, view, access-mode, liveness,
  and lowering validation.
- The authored-program lint and the eager lifetime preflight: retained as the batch-diagnostics surface for
  builder-constructed programs; the interpreter additionally catches use-after-consume and unbound roots at
  discharge time with instruction-level diagnostics.
- Deliberately kept ahead of JAX: read-only pruning (JAX returns a final value for every discharged reference),
  the effects gates, the typed runtime failure semantics, and the eager/staged parity oracles.

## 3. Track B: view representation (deferred, prototype-gated)

The access-site-transform direction remains attractive because its prize is deleting the alias-analysis category
(root/view graphs, boundary-crossing rules, the alias discharge handling), but it is a separate refactor with its
own risks, and the current code makes one of its premises stale: the access primitives are now universe-generic
(`programs::references::operations`), so an arrays-owned transform payload cannot be embedded without recreating
the `programs -> arrays` dependency. The decision therefore waits for a prototype comparing:

1. **Generic view descriptors on access operations**: the access primitives gain a generic transform-descriptor
   parameter (instantiated by arrays with its transform algebra), designed from day one as a descriptor *plus
   dynamically supplied index operands*, since a dynamic index is an SSA operand, not a static attribute (the
   Pallas/Mosaic operand-tree precedent).
2. **Retaining SSA view operations** with the interpreter transform from Track A.

Comparison criteria: net deletion (does the alias analysis actually disappear, including its boundary rules), IR
size and rendering legibility (per-access chains vs. shared view values; note Ryft's IR carries no per-instruction
source locations, so "source attribution" is not a real criterion today), the traced-handle model it forces
(cloneable handles need shared trace-time generation/invalidation for freeze; capture and attached-region import
rules must reject captured handles explicitly), dynamic-indexing readiness, and backend extensibility. Migration
happens only on a clear net simplification, after Track A is stable.

## 4. Implementation phases

- [ ] **Phase 0 — contract prototypes.** Type-check the family/value/context/driver/trait surfaces against the
      partial-evaluation precedent (including the capability-only-bounds and conversion-seam exit criteria above);
      verify a minimal non-array family instantiation so the architecture is provably not array-shaped; land the
      two result envelopes and the `ReferenceDischargeSite` vocabulary with its validation; prototype one
      structured fixed point (while) end to end on the test operation family. No production wiring.
- [ ] **Phase 1 — flat interpreter discharge.** Allocation, read, swap, add-update, freeze, and the SSA view-rule
      pair (`reference_index`/`reference_slice` composing the authoritative flowing alias) over the discharged
      environment, plus the `dispatch(discharge)` derive and its `ryft-macros`/`ryft-macros-tests` coverage, so
      the parity runs of phases 1–2 exercise the real array and XLA operation families. The current planner
      remains production; the interpreter runs beside it in tests with the full parity comparison on flat programs.
- [ ] **Phase 2 — structured interpreter discharge.** Condition, while, scan, and call rules over the driver
      services, including read-only pruning and the scan carry arithmetic. Exact parity with the planner is
      asserted across the complete existing discharge suite (both implementations run in tests).
- [ ] **Phase 3 — cutover.** Production discharge switches to the interpreter; the transform adapters and XLA
      stateful suites are the regression gate; planner machinery is deleted only where proven redundant, with
      surviving provenance-hook consumers named in the phase summary. No new derive or dispatch work lands here.
- [ ] **Phase 4 — flat partial discharge.** Selection parameter, `Preserved` threading, mixed-result typing, and
      the `PartialReferenceDischargeResult` envelope with `try_into_full`; preserved roots at structured boundaries
      are rejected with an exact diagnostic; adapters stay on the full-discharge contract; kernel-pipeline shaped
      tests (discharge pipeline state, keep kernel references) and preserved-kernel integration as a consumer.
- [ ] **Phase 4b — structured partial discharge.** Mixed structured carries: preserved reference-typed carries
      crossing condition/while/scan/call boundaries beside discharged state, with the boundary-rejection diagnostic
      from phase 4 lifted and the structured-rule and summary machinery extended accordingly. This phase owns the
      capability the plan advertises; partial discharge is not complete until it lands.
- [ ] **Phase 5 — view-representation prototype and decision (Track B).** Build the generic-descriptor prototype,
      run the Section 3 comparison, record the decision in this plan, and only then migrate or close the question.
- [ ] **Phase 6 — trace-time prevention and lint reduction.** Traced-handle scoping and freeze
      generation/invalidation (shared state across clones; capture rejection); shrink the standalone analysis to
      what is actually consumed. The reduction scope depends on Track B's outcome: with the flowing reference
      value carrying the complete alias chain (Section 2.1), discharge no longer needs analysis-provided view
      resolution even under retained SSA views, but the authored-program lint keeps whatever alias/view validation
      Track B's decision leaves in the IR. The surviving consumers (lint, eager preflight, summary service, site
      enumeration) are named in the phase summary.
- [ ] **Phase 7 — stabilization.** Documentation (JAX correspondence, prevention ladder, experimental markers),
      doctests, `ryft-macros` and `ryft-macros-tests` verification (the derive surface changes), full suites
      across `ryft-core`/`ryft-xla`/`ryft-pjrt`, deletion audit, and independent audit rounds to convergence per
      the house protocol.

Each phase lands green on the full test matrix before the next begins, and each phase's summary is recorded in the
review record below.

## 5. Testing strategy

- **Dual-implementation parity (phases 1–3):** the planner is the oracle; every discharged program in the existing
  suite is produced by both implementations and compared exactly on: renderings, output types and identities,
  `public_output_count`, `ReferenceStateBinding`s, hidden-output order, effects and access summaries, structured
  region layouts, and rejection diagnostics (message-for-message).
- **Behavioral oracle:** eager reference semantics remain the ground truth throughout, including for partial
  discharge (a preserved root's eager behavior must match the mixed program's).
- **Rule-level tests** mirror the batching/differentiation suites: one focused test per operation rule, plus the
  driver services (summaries, unions, while interplay, scan positions, call substitution) tested directly.
- **Partial discharge:** mixed programs asserting preserved roots stay well-typed, selected roots thread, the
  partial envelope reports only discharged bindings, and full discharge converts into the reference-free envelope.
- **Prevention tests (phase 6):** trace-time rejections, discharge-time environment errors, and lint diagnostics.

## 6. Phase ownership and estimates

Owning files per phase, with rough size estimates (production/tests). Exclusions apply to every phase: the holder
runtime (`programs/references/runtime.rs`), the external-state ABI and V6 persistence, the XLA transaction, and all
changelogs (owner-written).

Estimates are production/tests/docs line ranges.

- **Phase 0:** `programs/references/discharge.rs` (family, value, context, driver, envelopes, sites;
  ~350/300/100) plus a throwaway prototype module for the while fixed point and the non-array family check
  (deleted at phase exit).
- **Phase 1:** `programs/references/discharge.rs` (context, environment, flat services; ~400/300/80),
  `programs/references/operations.rs` and `arrays`-side view/access rules (~250/250/60), `ryft-macros` +
  `ryft-macros-tests` (`dispatch(discharge)`; ~150/100/20), dual-run harness (~0/150/0).
- **Phase 2:** structured-operation modules under `operations/control_flow/` and the call carriers (rules;
  ~400/350/80), driver fork/summary services (~200/150/40).
- **Phase 3:** `arrays/reference_discharge.rs` (cutover and deletion; net-negative production delta expected;
  ~-600/100/40).
- **Phase 4:** `programs/references/discharge.rs` (selection, partial envelope, `try_into_full`; ~250/300/60),
  `ryft-xla/src/experimental/reference_kernels.rs` (consumer integration; ~100/150/30).
- **Phase 4b:** structured-rule and summary extensions for mixed carries (~250/300/40).
- **Phase 5 (Track B):** prototype under `arrays/` only, with a fixed budget of ~500/200/0 and a hard deletion
  criterion: the prototype is deleted at the phase-5 decision regardless of outcome, and an adopting migration is
  sized and planned as its own follow-up from the prototype's findings.
- **Phase 6:** `arrays/reference_views.rs`, tracing handle modules, `programs/references/analysis.rs`
  (reduction; net-negative), lint surface (~-300/200/60).
- **Phase 7:** documentation and audits across the touched files; no new production surface (~0/100/300).

Rough total (excluding phase 5's unadopted prototype and phase 3/6 deletions): ~2,000 production, ~2,400 test, and
~900 documentation lines.

## 7. Risks and mitigations

- **The driver services regrow into a planner.** Mitigated by the placement rule: services classify reference
  *effects* and assemble boundaries on request — that is their job — but they never select rewrite *rules*. Any
  service that starts choosing how an operation is rewritten is the planner returning and gets rejected in review.
- **Trait-solver recursion in the dispatch derives.** Mitigated by the phase-0 exit criterion (capability-only
  context bounds) and by running the macro integration suites whenever the derive surface changes.
- **Two-implementation window cost.** Phases 1–3 carry both implementations; mitigated by keeping the window short
  and test-only, with cutover as its own phase.
- **Summary-service correctness.** The transitive closure summaries are the one analysis remnant on the hot path;
  they get direct unit coverage and are cross-checked against the existing analysis during the parity window.
- **Track B drag.** The view decision is explicitly allowed to conclude "keep SSA views"; Track A's value does not
  depend on it.

## 8. Resolved design decisions

1. **Naming (decided).** The rule trait is `ReferenceDischargeableOperation`; the flowing value is
   `ReferenceDischargeValue::{Pure, Reference}`.
2. **The discharge context implements `Context` (decided, phase 0 validates).** Discharge is a single
   program-to-program interpretation, so it runs through `interpret_in_context` like batching and differentiation
   rather than through a bespoke replay like partial evaluation (whose wrapper shape is justified only because
   partitioning is not a single interpretation). `ReferenceDischargeValue` becomes a real `Value` family on the
   `ArrayIrValue` heterogeneous-value precedent: capabilities delegate on `Pure` and error on `Reference`, and the
   reference arm types as `ReferenceType`. Phase 0 validates the value-family impls and the capability-only-bounds
   discipline; most driver services from Section 2.1 become context methods.
3. **Partial-discharge selection (decided).** The full selection surface ships immediately through the checked
   `ReferenceDischargeSite` vocabulary (externals plus allocation sites; nested formal roots are deliberately not
   selectable) — externals-only would force a second API migration for the kernel pipeline without avoiding the
   hard case. The staging knob is semantic instead: the initial implementation rejects preserved roots at
   structured boundaries with an exact diagnostic, and mixed structured carries are lifted in phase 4b, which owns
   that capability explicitly.
4. **Freeze linearity (decided).** `freeze` takes the handle by value on both the traced and eager surfaces, so the
   common single-handle misuse is a compile error; cloned aliases share invalidation state and fail dynamically at
   their next use (the eager holder's alias-family invalidation already provides this). Freezing through a shared
   borrow becomes an explicit clone-then-freeze.

## 9. Plan review record

- [ ] Initial owner review of this plan.
