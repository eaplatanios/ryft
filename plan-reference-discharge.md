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
facts, computed generically from `reference_semantics`, the region-provenance hooks, reference-output identity, and
recursive region summaries), the per-operation rules (which decide and emit the rewrite), and driver services
(checked summaries, transactional forks, unions, and boundary assembly).

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

**The discharge policy.** One trait owns every universe-varying type, named and shaped on the batching-policy
precedent (`ArrayBatchingPolicy`/`RecursiveBatchingPolicy`, whose implementors are zero-sized markers generic over
the context — deliberately *not* a context capability, because one marker's single generic impl covers every
context of its type system, while a context-implemented capability would need a coherence-foreclosing blanket to do
the same). The value, context, driver, and rule signatures all name a single policy parameter instead of loose
generics, so a non-array universe is a first-class instantiation rather than an afterthought:

```rust
/// Policy naming the types and alias mechanics one reference universe threads through discharge, implemented by
/// zero-sized markers such as the arrays universe's `ArrayReferenceDischarge`.
pub trait ReferenceDischargePolicy<C: Domain>: Copy + Clone + Debug {
    /// Referent type system of this universe's references.
    type Referent: Type;

    /// Composed alias metadata carried by one flowing handle (the view chain, for arrays; `()` for
    /// view-less families).
    type Alias: Clone + Debug + Parameter;

    /// Returns the identity alias of an unviewed root with the provided referent type, used by allocation and
    /// entry-boundary binding. Infallible by design: referent validity is type inference's job, and identity-alias
    /// construction for a validated referent is total.
    fn root_alias(referent: &Self::Referent) -> Self::Alias;

    /// Lifts a reference type into the destination type universe.
    fn lift_reference_type(r#type: ReferenceType<Self::Referent>) -> C::Type;

    /// Lifts a referent type into the destination type universe. A discharged root's immutable state is an
    /// ordinary destination value of exactly this type, so this is the direction that types an entry-boundary
    /// position whose reference became state.
    fn lift_referent_type(referent: Self::Referent) -> C::Type;

    /// Projects a destination type back onto a reference type, when it is one. The lift/project pair is the
    /// conversion seam access rules use to type-check operands. Projection returns `Option` rather than mirroring
    /// the `TryFrom`-with-`TypeError` idiom, because classifying an operand as "not a reference" is an outcome
    /// rather than a failure and the calling rule owns the resulting diagnostic.
    fn project_reference_type(r#type: &C::Type) -> Option<ReferenceType<Self::Referent>>;

    /// Applies this universe's alias metadata to one immutable state value, returning the selected value.
    fn read(context: &C, current: &C::Value, alias: &Self::Alias) -> Result<C::Value, ProgramError>;

    /// Replaces the coordinates selected by `alias`, returning the previous selection and the successor state.
    fn replace(
        context: &C,
        current: &C::Value,
        replacement: C::Value,
        alias: &Self::Alias,
    ) -> Result<(C::Value, C::Value), ProgramError>;
}

/// Ordered accumulation contract of a reference universe whose references support additive updates, refining the
/// base policy so that accumulation is unavailable — at compile time, and only for `reference_add_update` — to a
/// universe that declines to implement it.
pub trait ReferenceAccumulationPolicy<C: Domain>: ReferenceDischargePolicy<C> {
    /// Accumulates `update` into the coordinates selected by `alias`, returning the successor state.
    fn accumulate(
        context: &C,
        current: &C::Value,
        update: C::Value,
        alias: &Self::Alias,
    ) -> Result<C::Value, ProgramError>;
}
```

The alias-application methods are what keep the generic primitive rules genuinely generic: without them, read,
swap, and add-update would be implicitly array-owned. Capability requirements are placed for per-access
granularity rather than as one impl-level union, but the mechanism is a *refining subtrait* rather than a
`where C::Value: Add` clause on one policy method. A trait-level method bound presumes that every reference
universe reaches addition the same way, and Ryft's production universe does not: `ArrayIrOperations` states
explicitly that homogeneous array capabilities such as `Add` are deliberately not composite members, and
`AddOperation<ArrayIrType>` is a lift marker rather than an `Operation`, so no composite array-IR value — eager or
staged — implements `Add`. The requirement each universe must state genuinely differs (a value-level capability
for one, an operation-lifting conversion for another), so accumulation lives on `ReferenceAccumulationPolicy` and
each implementation states its own destination requirement on its own `impl` block, exactly as the arrays policy's
base `impl` block demands only its view mechanics (`Reshape + Slice + UpdateSlice`).

The approved semantics are preserved and in one respect strengthened. A universe that supports views but not
addition keeps reads and replacements through the base policy, and a program containing `reference_add_update`
fails to discharge for it at compile time, scoped to exactly that operation — now because the refining trait is
unimplemented rather than because a method bound is unsatisfied, which is a requirement the downstream register
universe actively declines rather than merely never being asked to meet. The two unavailability cases stay
distinct: declining the subtrait makes accumulation unavailable at compile time, while a universe whose
destination could add but whose references forbid accumulation implements the subtrait with an explicit runtime
`UnsupportedOperation` rejection. Two recorded caveats survive unchanged: closed operation-enum dispatch
reintroduces the union for enums whose members include add-update (a pre-existing property of enum dispatch,
identical to interpretation today), and dtype-level incompatibilities still surface as runtime `Result`s even
where the requirement holds.

The arrays policy delegates to the existing shared view carriers through `ArrayReferenceViewOperation`, which owns
the three canonical view constructors and which the planner's `ArrayReferenceDischargeOperation` inherits as a
super-trait. The split exists so that the interpreter states only the contract it uses, leaving phase 3 free to
delete the planner's structural-classification half without touching the policy. The `C` bound placement is
settled: both policy traits and the rule trait take `C: Domain`, following the `InterpretableOperation` precedent,
so naming a policy never obliges a caller to prove an active binding contract; an implementation narrows `C` to
`Context` when its own alias mechanics bind operations, as the arrays policy does.

**Values and environment.** The flowing value is a context-stamped tracer, on the `BatchingTracer`/
`DifferentiationTracer` precedent — the eager `ArrayIrValue` is the wrong model, because eager values self-dispatch
while a discharge value must dispatch through the live context that owns the root environment (`Value` requires
reporting the active dispatch domain, which cannot be reconstructed from a root handle). The wrapper cannot reuse
the generic `Tracer<C>` either: a discharged reference handle has no destination atom to wrap, which is precisely
the case the generic tracer cannot represent.

```rust
/// Context-stamped discharge value; implements `Value` and dispatches through its context.
pub struct ReferenceDischargeTracer<C: Domain, P: ReferenceDischargePolicy<C>> {
    context: ReferenceDischargeContext<C, P>,
    value: ReferenceDischargeValue<C, P>,
}

/// Context-free carrier inside one [`ReferenceDischargeTracer`]. Public because the rule trait names it, but
/// enum variant fields are always as public as the enum, so the reference payload is an opaque struct: downstream
/// code can match and read handles but cannot fabricate roots, aliases, types, or preserved values, preserving the
/// checked-construction contract (pinned by a trybuild compile-fail test).
pub enum ReferenceDischargeValue<C: Domain, P: ReferenceDischargePolicy<C>> {
    /// Ordinary pure value, replayed as-is.
    Pure(C::Value),

    /// Handle to one live root.
    Reference(ReferenceDischargeReference<C, P>),
}

/// Opaque reference handle mirroring the eager `ArrayReference` shape, with private fields and read-only
/// accessors; only the discharge context constructs it.
pub struct ReferenceDischargeReference<C: Domain, P: ReferenceDischargePolicy<C>> {
    /// Root identity.
    root: ReferenceRootHandle,

    /// Composed policy-owned alias metadata.
    alias: P::Alias,

    /// Derived reference type this exact handle exposes (differs from the root type under a composed view).
    r#type: ReferenceType<P::Referent>,

    /// For a preserved root, the exact destination reference value this handle denotes: the entry binding for the
    /// root handle, or the bound output of the replayed view operation for a derived handle. Later accesses consume
    /// this exact value; re-deriving the chain per access would duplicate and reorder view operations. Invariant:
    /// `None` exactly when the root is `Discharged`, `Some` exactly when `Preserved`.
    preserved: Option<C::Value>,
}

/// Environment entry for one live root. Public, because the context returns it from `root_state`, which
/// third-party structured rules need and which a public trait cannot expose through a private type.
pub enum ReferenceRootState<A> {
    /// Selected for discharge: threads as immutable state.
    Discharged {
        /// Current immutable state value.
        current: A,
        /// Whether any ordered write or accumulation has occurred (drives hidden-output and widening pruning).
        mutated: bool,
    },

    /// Not selected: survives in the destination; this is the root's own destination reference value, used for
    /// boundary threading. Derived preserved handles carry their own exact destination values in the flowing
    /// `preserved` field.
    Preserved {
        /// Destination reference-typed root value.
        reference: A,
    },
}
```

The tracer implements `Value` (Section 8, decision 2); the enum is merely its carrier — and because the public
rule trait names it in signatures, the carrier, the opaque `ReferenceRootHandle`, and every type a third-party rule
touches must themselves be public with private fields (a public trait cannot name private types, and XLA and custom
backends are external crates). Construction stays checked: rules obtain and produce values only through public
context/driver methods, never by building roots or environment entries directly. Phase 0 explicitly decides and
tests the complete downstream surface — public carrier and handle, rule-safe context accessors and constructors,
the driver services third-party structured rules need, and the `C: Domain`/`C: Context` bound placement required to
name `C::Type` and `C::Value` — and proves it with a downstream-style compile test: an external operation family
plus a non-array policy implementing the complete capability from an integration-test crate position, where the
compiler itself enforces that no private API is reachable. The exact impl set, the
minimal policy bounds, and a non-array test policy that exercises root-alias construction, read, replacement, and
accumulation — not merely type instantiation — are phase-0 prototype deliverables, not implementation details to
discover late.

**Rule trait and bounds.** The per-operation rule follows the transform-rule shape, and the discharge context
implements `Context` (Section 8, decision 2), so discharge runs through `interpret_in_context` like batching and
differentiation and rules can bind through the context directly:

```rust
pub trait ReferenceDischargeableOperation<C: Domain, P: ReferenceDischargePolicy<C>>: Operation {
    // Named after the program-level entry, following the same-verb-at-both-levels precedent of `jvp` and
    // `partially_evaluate`. The super-trait is a plain `Operation` with the projection equality restated per
    // function, matching `BatchableOperation`: the current trait solver cannot discharge that equality at
    // implementation heads whose context type is built from `Self`.
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError>
    where
        Self: Operation<Type = C::Type>;
}
```

The driver additionally exposes the replay position — `fn instruction(&self) -> Option<InstructionId>` — because an
allocation rule must know its own site to test selection membership (Blocker: the existing region drivers expose
attached regions but not the current instruction). Program replay always supplies the identity; a direct
`Context::bind` with no source instruction returns [`None`], and such allocations are always discharged — no site
can name them, and rejecting them instead would make direct binding unusable.

Three constraints were exit criteria for the phase-0 prototype rather than afterthoughts, and all three were met:
the rule trait's context parameter carries capability-only bounds (never `C: Context` on the trait — the pattern
that keeps the dispatch derives clear of trait-solver recursion); the reference-free replay rule's conversion from `Self`
into `C::Operation` uses the established conversion seam with an explicit `Self::Type`/`C::Type` relationship; and
`ReferenceDischargeTracer` proves out as a real `Value` (capabilities delegate on `Pure`, error on `Reference`,
and the reference arm types as the exact handle type). The capability-only bound has one mechanical consequence
recorded with the phase-0 summary: the rule trait cannot carry a defaulted body, because a default reference-free replay
body would need `C: Context` and `Self: Clone + Into<C::Operation>` on the trait itself. Pure replay is therefore
the free function `discharge_reference_free_operation`, which `impl_reference_free_dischargeable_operation!` delegates to, on the model of
`impl_non_transposable_operation!`.

**Capture scopes (phase-3 contract).** A capture-lifted program names its caller's references through constants
rather than through its boundary, so the interpreter needs one more piece of per-scope context: which root each
capture position binds. That is a property of the *scope* a region discharges under, not of any rule, so it lives
beside the root environment on the discharge context and is recomputed at every region boundary:

```rust
/// Roots the enclosing capture prefix binds, together with the constant-family seam that recognizes a capture.
pub struct ReferenceCaptureScope<Constant> {
    /// Seam reporting the capture position a constant names, or [`None`] when it is an ordinary constant.
    capture_index: fn(&Constant) -> Option<usize>,

    /// Root each capture position binds, or [`None`] when that position carries an ordinary value.
    roots: Vec<Option<ReferenceRootHandle>>,
}
```

The seam is a plain function pointer rather than a bound, because `CaptureConstant` cannot be required of every
constant family the interpreter serves — the non-array prototype universes deliberately do not implement it — and it
is the same higher-order seam `RegionRef::analyze_references_with_capture_indices` already uses, reduced to a `Copy`,
allocation-free carrier. The default scope resolves nothing, so every existing instantiation keeps today's behavior:
a reference-typed constant that no scope binds is still rejected by `lift_constant`, with its message unchanged.

Four seams change, and nothing else does:

- `lift_constant` consults the active scope before rejecting, and a resolved capture lifts to the *whole-root* handle
  of the root that capture binds — not to a fresh root, exactly as the analysis reuses the entry root rather than
  minting a second one. A capture position that binds no root falls through to the existing rejection.
- `summarize_region_closure` seeds its root table with the region's reference-typed constant atoms as well as its
  boundary, so a capture-scoped access is *summarized* rather than reported as unresolvable, and records a read for
  every such constant the region *consumes*. The two halves answer different questions: seeding lets an access
  through the constant resolve, while the recorded read is what threads the root even when the closure only passes
  the constant along. Threading follows the replay's own liveness rule rather than the access semantics, because the
  replay materializes a constant something consumes and then needs a root for it. Together they make a structured
  rule's `entering` set non-empty and finally reach the synthesized state-carry paths phase 2 could build but not
  exercise.
- `ReferenceDischargeContext::region_summary` takes the region's owning operation and region index, because whether a
  region establishes a *fresh* scope is stated by the existing generic hook `Operation::region_capture_input_count`
  (a positional call declares its leading capture prefix; every other region inherits its parent's scope). Deriving
  it inside the summary is what keeps the rule from having to know about captures at all.
- `ReferenceRegionDischargeBoundary` gains `capture_input_count: Option<usize>`, which its constructor *derives*
  from the same hook rather than accepting from the rule, so that a rule cannot state one prefix on the boundary and
  let the summary derive another from the same operation. The region fork uses it to build the rebuilt region's
  scope, expressed in *fork* roots — the fresh case reads them straight off the threaded declared inputs, and the
  inherited case maps the caller's scope through the same private caller-to-fork table the boundary already uses —
  so the sealed-fork isolation contract is unchanged: no caller handle reaches the fork and no fork handle reaches
  the caller. Added state may not be inserted inside the prefix, which the fork rejects by name.

The program-level entry point pairs like the analysis does: `discharge_references_with_policy` keeps its signature
and installs the empty scope, and `discharge_references_with_lifted_captures_and_policy` requires
`V: CaptureConstant`, binds the entry boundary's leading capture prefix into the scope, and is what the arrays
universe's `discharge_references_with_lifted_captures` routes to at cutover.

One typing relaxation is required and is confined to the derive: the generated discharge dispatcher pinned
`__ParentContext::Constant` to the family's declared constant type, which no rule body and no shared rule helper ever
reads, and which made the dispatcher inapplicable to a capture-lifted program (whose constants are
`CaptureReference<..>` rather than the family's own value type). Dropping that one predicate from
`generate_reference_dischargeable_operation` — and only from that dispatcher — is what lets the interpreter be
instantiated for the capture-lifted array-IR programs the planner already serves.

**Driver services, not contracts.** Structured rules (condition, while, scan, call) own their widening, but the
planning-shaped logic they share is provided once by the driver, as services rules compose — the same division the
batching and partial-evaluation drivers already use:

- transactional region forks (`ReferenceDischargeDriver::discharge_region_program`): a structured rule requests a
  boundary through `ReferenceRegionDischargeBoundary` and receives only the sealed, context-free
  `ReferenceRegionDischargeFork` — the rebuilt region program, the caller root each declared region output denotes,
  and the threaded roots the region actually mutated. The fork result carries no values of any kind: child discharge
  tracers would keep mutating the abandoned child environment, and even a "plain" `C::Value` is not detached under a
  staging destination (it is itself a tracer stamped with the child's destination builder), so the result type
  structurally excludes both. The owning rule binds the rebuilt structured operation in the *parent* context,
  producing fresh parent-stamped tracers; should a future need for returning values arise, it requires an explicit
  detach/rebind operation with a stated invariant proving the child context is unreachable. Only the owning rule
  merges returned final states. Isolation is achieved by *building* the fork's environment rather than snapshotting
  the caller's: it holds exactly the roots the boundary names, each entering as an ordinary value at its own
  position, and it mints its own environment identity, so a caller handle cannot address a fork root and a fork
  handle cannot address a caller root — a leak in either direction is reported rather than silently aliased. That is
  strictly stronger than the snapshot this bullet originally specified, and strictly stronger than `Context` cloning
  (stateful context clones share active transform state, which is exactly wrong here): both condition branches
  necessarily observe the same entering environment, a rebuilt region commits nothing, and a failed replay leaves the
  caller environment untouched and yields no usable child values. Discharge needs this where batching and
  differentiation do not because its state lives in the context environment rather than riding in per-value tracers.
  Pinned by tests: identical branch snapshots, parent-stamped branch outputs after the rebuilt condition binds, no
  leakage from a rebuilt loop region, no caller mutation or usable values after a failed replay, and no escape of a
  branch-local allocation through merge;
- transitive closure summaries (`ReferenceDischargeContext::region_summary`): which roots a region closure reads,
  writes, or accumulates, which a loop rule must know *before* widening, together with the caller root each declared
  region output denotes, which is how a root the region already returns is kept from being published twice. This is
  retained reference analysis, acknowledged as such, and its inputs are named precisely: operation-local
  `reference_semantics`, input-region provenance, output-region provenance, reference-output identity, and
  recursively computed summaries of nested regions — all existing generic hooks. The summary is *reported* in caller
  roots, because the rule supplies the caller root of each declared region input when it asks — the rule owns its
  operand-to-region-input mapping because it is the operation. Third-party structured operations therefore need no
  companion declaration surface beyond the hooks they already implement;
- deterministic root union and ordering for threaded state, which falls out of ordering roots by their environment
  identity: roots are minted in entry-boundary order and then in replay order, which is the order the planner's
  arena coordinates already produce;
- the shared positional rewrite (`discharge_positional_region_operation`) covering the two structured shapes whose
  regions mirror the operand list after a constant leading offset — a condition and a positional call — including
  the read-only pruning both apply. The loop shapes own their rules instead, because boundary symmetry forbids
  pruning: the while condition/body interplay (the condition observes entering state and produces none) and the scan
  carry-position arithmetic are each stated once, by the operation they belong to;
- boundary assembly and validation for the rewritten operation, including
  `ReferenceRegionDischargeFork::validate_predicted_mutations`, which holds the summary that sized a boundary to what
  the rebuilt region actually did.

Rules remain able to produce malformed programs, exactly as jvp and batching rules can; the house answer stays
builder-time type checking, the result envelopes, and oracles — not per-operation contracts.

### 2.2 Placement and openness

- Reference primitives implement their own rules (allocation binds a fresh discharged or preserved root; read,
  swap, and add-update act on `Discharged` state through the shared view machinery or replay verbatim on
  `Preserved` roots; freeze yields the current state and unbinds).
- Structured operations implement their own widening through the driver services, including the read-only pruning
  policy (no synthesized output or condition/call widening for roots a closure only reads; loop-shaped rules keep
  boundary symmetry).
- A region-carrying operation whose attached closures touch no reference needs no rule of its own: the shared
  reference-free replay copies its regions into the destination as they stand and rejects the application by name as
  soon as a reference does appear anywhere in them. That is what keeps the derivative-rule carriers (`linear_call`,
  `custom_jvp`, `custom_vjp`, `rematerialize`) and the manual-SPMD `shard_map` on one macro invocation: threading
  state through a dormant rule region or a per-shard boundary has no defined meaning, so rejecting is the rule.
- Capture-reference constants are deliberately *not* a driver service. A capture-lifted program names its caller's
  references through constants rather than through its boundary, and the resolution belongs to the *scope* a region
  discharges under rather than to any one rule; phases 1 and 2 rejected such a constant outright, and phase 3 gives
  the context the capture scope described in Section 2.1 instead.
- Backend enums forward through a dispatch derive as for differentiation and transposition; third-party operations
  implement the trait directly. The system is open over primitives.
- SSA view operations are retained in Track A, with their own rules: `reference_index` and `reference_slice`
  preserve the incoming root handle, validate and compose the transform onto the flowing `alias` (rejecting invalid
  composition before binding the output, with the same math as eager `with_transform`), and record the derived
  reference type. On a discharged root the rule only composes metadata and binds nothing; on a preserved root it
  additionally replays the view operation into the destination and stores the bound output as the handle's exact
  `preserved` value, so later accesses consume that value instead of re-deriving the chain. The flowing `alias`
  field is the single authoritative view chain during discharge; the analysis never supplies view resolution to
  the transform (it remains the lint's and the summaries' concern), so a handle's view has exactly one source of
  truth.

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

- [x] **Phase 0 — contract prototypes.** Type-check the policy/value/context/driver/trait surfaces against the
      partial-evaluation precedent (including the capability-only-bounds and conversion-seam exit criteria above);
      verify a minimal non-array policy instantiation so the architecture is provably not array-shaped; land the
      two result envelopes and the `ReferenceDischargeSite` vocabulary with its validation; prototype one
      structured fixed point (while) end to end on the test operation family. No production wiring.
- [x] **Phase 1 — flat interpreter discharge.** Allocation, read, swap, add-update, freeze, and the SSA view-rule
      pair (`reference_index`/`reference_slice` composing the authoritative flowing alias) over the discharged
      environment, plus the `dispatch(discharge)` derive and its `ryft-macros`/`ryft-macros-tests` coverage, so
      the parity runs of phases 1–2 exercise the real array and XLA operation families. The current planner
      remains production; the interpreter runs beside it in tests with the full parity comparison on flat programs.
- [x] **Phase 2 — structured interpreter discharge.** Condition, while, scan, and call rules over the driver
      services, including read-only pruning and the scan carry arithmetic. Exact parity with the planner is
      asserted across the complete existing discharge suite (both implementations run in tests).
- [x] **Phase 3 — cutover.** Production discharge switches to the interpreter; the transform adapters and XLA
      stateful suites are the regression gate; planner machinery is deleted only where proven redundant, with
      surviving provenance-hook consumers named in the phase summary. No new derive or dispatch work lands here.
      Two prerequisites recorded by phase 2 gate the cutover itself and must be discharged first: the interpreter
      must be able to resolve a capture-scoped reference (today it rejects a reference-typed constant, and
      `discharge_references_with_policy` cannot even be instantiated for a capture-lifted program), and it must be
      able to bind a region-closure root so that synthesized state carries become reachable end to end. Phase 3 must
      also sweep every pinned discharged rendering for the dead-constant divergence, and relocate the misplaced
      `custom_jvp`/`custom_vjp` reference tests that phase 0 left in `differentiation/forward.rs`.
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
  region layouts, and rejection diagnostics (message-for-message). Parity has exactly one recorded, deliberate
  divergence: the planner copies every stored constant into the destination before replaying anything, while the
  interpreter reaches the destination through the shared program replay path, which lifts only the constants
  something still consumes. A source containing a *dead* constant therefore discharges to a program with one fewer
  atom under the interpreter. Dropping it is what every other transform already does and is strictly the better
  artifact, so the interpreter keeps that behavior and the divergence is pinned by its own test; phase 3 must
  consequently sweep every pinned discharged rendering, in `ryft-xla`'s stateful suites included, for a retained
  dead constant before cutover, because those pins move.
  Rejection-diagnostic parity is also narrower than it sounds for the rules the planner validates by contract: most
  planner rejections are checks on its own `reference_discharge_rule` classification, which the interpreter has no
  analogue for because it has no classification to violate. Those tests are planner-only and are deleted with it.
- **Behavioral oracle:** eager reference semantics remain the ground truth throughout, including for partial
  discharge (a preserved root's eager behavior must match the mixed program's).
- **Rule-level tests** mirror the batching/differentiation suites: one focused test per operation rule, plus the
  driver services (summaries, unions, while interplay, scan positions, call substitution) tested directly.
- **Partial discharge:** mixed programs asserting preserved roots stay well-typed, selected roots thread, the
  partial envelope reports only discharged bindings, and full discharge converts into the reference-free envelope.
- **Prevention tests (phase 6):** trace-time rejections, discharge-time environment errors, and lint diagnostics.

## 6. Phase ownership and estimates

Owning files per phase, with rough size estimates (production/tests). Exclusions apply to every phase: the holder
runtime (`programs/references/runtime.rs`), the external-state ABI and V6 persistence, the XLA transaction, and
all changelog files: excluded; no changes.

Estimates are midpoint production/tests/docs line counts, not ranges; deletions are reported separately at the
end rather than as negative production.

- **Phase 0:** `programs/references/discharge.rs` (policy, value, context, driver, envelopes, sites;
  ~350/300/100) plus a throwaway prototype module for the while fixed point (deleted once the real structured
  rules supersede it). The minimal non-array policy and its read/replace/accumulate tests are permanent — they are
  the standing proof that the architecture has not silently become array-specific.
- **Phase 1:** `programs/references/discharge.rs` (context, environment, flat services; ~400/300/80),
  `programs/references/operations.rs` and `arrays`-side view/access rules (~250/250/60), `ryft-macros` +
  `ryft-macros-tests` (`dispatch(discharge)`; ~150/100/20), dual-run harness (~0/150/0).
- **Phase 2:** structured-operation modules under `operations/control_flow/` and the call carriers (rules;
  ~400/350/80), driver fork/summary services (~200/150/40).
- **Phase 3:** `arrays/reference_discharge.rs` (cutover; ~100/100/40, with the planner deletion reported below).
- **Phase 4:** `programs/references/discharge.rs` (selection, partial envelope, `try_into_full`; ~250/300/60),
  `ryft-xla/src/experimental/reference_kernels.rs` (consumer integration; ~100/150/30).
- **Phase 4b:** structured-rule and summary extensions for mixed carries (~250/300/40).
- **Phase 5 (Track B):** prototype spanning a test-only generic access-operation fixture in
  `programs/references` plus an array specialization — the descriptor parameter lives on the universe-generic
  access operations, so an arrays-only prototype could not test the dependency boundary the phase exists to
  evaluate. Fixed budget of ~500/200/0 and a hard deletion criterion: the prototype is deleted at the phase-5
  decision regardless of outcome, and an adopting migration is sized and planned as its own follow-up from the
  prototype's findings.
- **Phase 6:** `arrays/reference_views.rs`, tracing handle modules, `programs/references/analysis.rs`
  (reduction), lint surface, and the by-value `freeze` migration, which changes the generic `FreezeReference`
  capability contract in `programs/references/operations.rs` and therefore every implementation and call site
  (eager, traced, capability impls, tests) — named here so the API migration and its tests are owned, not
  discovered (~250/300/60, deletions reported below).
- **Phase 7:** documentation and audits across the touched files; no new production surface (~0/100/300).

Rough planned-work totals (phase 5's prototype is mandatory work even though it is deleted at the decision):
~3,200 production, ~3,050 test, and ~900 documentation lines written, against roughly 900 production lines deleted
(the planner and bespoke replay in phase 3, the analysis reduction in phase 6). The retained baseline excluding the
prototype is ~2,700/~2,850/~900.

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

1. **Naming (decided).** The rule trait is `ReferenceDischargeableOperation` with rule method
   `discharge_references`; the flowing `Value` is the context-stamped `ReferenceDischargeTracer`, whose
   context-free inner carrier is `ReferenceDischargeValue::{Pure, Reference}`.
2. **The discharge context implements `Context` (decided, phase 0 validates).** Discharge is a single
   program-to-program interpretation, so it runs through `interpret_in_context` like batching and differentiation
   rather than through a bespoke replay like partial evaluation (whose wrapper shape is justified only because
   partitioning is not a single interpretation). The context's value is the context-stamped
   `ReferenceDischargeTracer` on the `BatchingTracer`/`DifferentiationTracer` precedent — not an eager-style
   contextless enum, because discharge values must dispatch through the live context that owns the root
   environment, and not the generic `Tracer<C>`, because a discharged handle has no destination atom to wrap.
   Capabilities delegate on `Pure` and error on `Reference`, and the reference arm types as the exact handle type.
   Phase 0 validates the tracer impls and the capability-only-bounds discipline; most driver services from
   Section 2.1 become context methods.
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

### Phase 0 — contract prototypes (landed)

**What landed.** All of phase 0's deliverables, in `crates/ryft-core/src/programs/references/discharge.rs` plus one
new integration test and one small addition to `crates/ryft-core/src/programs/regions.rs`:

- **Both result envelopes.** `PartialReferenceDischargeResult<P>` sits beside the unchanged
  `ReferenceDischargeResult<P>`, and the boundary validation the two share was extracted into one private
  `validate_discharged_boundary` rather than duplicated. `try_into_full` is implemented only for `Program` payloads
  and performs the real proof: no reference-typed atom anywhere in the attached region closure (dormant rule regions
  included, which also covers boundary positions and stored constants because both are atoms) and no operation with
  nonempty reference semantics anywhere in that closure. The proof is deliberately reference-specific, pinned by a
  test in which an unrelated ordered-state operation converts successfully.
- **The selection vocabulary.** `ReferenceDischargeSite` with `External`/`Allocation`, plus the lightweight
  enumeration query `Program::reference_discharge_sites` and the checked
  `Program::validate_reference_discharge_sites`. Enumeration reads only the entry boundary types and the generic
  `Operation::reference_semantics` hook over the region closure, so it is independent of the standalone analysis and
  survives the phase-6 reduction. Validation reports duplicates, out-of-range and non-reference entry positions,
  unknown instructions, non-allocating operations, and non-allocating output positions, each by name.
- **The interpreter contract surface.** `ReferenceDischargePolicy`, `ReferenceRootHandle`, `ReferenceRootState`,
  `ReferenceDischargeReference` (opaque, private fields, read-only accessors), `ReferenceDischargeValue`,
  `ReferenceDischargeTracer` (a real `Value`), `ReferenceDischargeContext` (a real `Context`),
  `ReferenceDischargeDriver` with its mandated `instruction()`, `RecursiveReferenceDischargeDriver`, and
  `ReferenceDischargeableOperation` with rule function `discharge_references`.
- **The non-array policy.** A permanent in-crate prototype universe of fixed-length integer lists whose alias is a
  contiguous sub-range, exercising root-alias construction, read, replacement, and accumulation with real view
  mechanics, plus its operation family and per-operation rules.
- **The structured fixed point.** A `while` rule that probes its body on an isolated `fork()` until the mutated-root
  set stops growing, then commits the loop against the live environment and checks that the commit touched no root
  the probe did not predict. Its test uses a witness counter, so a probe that leaked into the committed environment
  would change the asserted result. Marked `TODO(eaplatanios)` for deletion at phase 2.
- **The downstream compile proof.** `crates/ryft-core/tests/test_reference_discharge_downstream_surface.rs` defines a
  second, view-less reference universe and its complete policy and rules from an integration-test crate position, so
  the compiler enforces that no private `ryft-core` item is needed.

**Exit criteria and the decisions they forced.**

1. **Capability-only bounds (met, and stricter than the batching precedent).** The rule trait is
   `ReferenceDischargeableOperation<C: Domain, P: ReferenceDischargePolicy<C>>`, and the policy is
   `ReferenceDischargePolicy<C: Domain>`. `Domain` rather than `Context` follows the `InterpretableOperation`
   precedent instead of `BatchableOperation`'s `C: Context`, which is what the plan asked for; `Context` is required
   only on the impls that actually bind. The super-trait is a plain `Operation` with `Self: Operation<Type = C::Type>`
   restated on the rule function, matching the recorded E0284 limitation. No rule trait or driver names
   `C::Operation: ReferenceDischargeableOperation<..>`; that obligation appears exactly once, on the
   `ReferenceDischargeContext` `Context` impl and on the recursive driver.
2. **`project_reference_type` returns `Option` (decided).** Classifying an operand as "not a reference" is an
   outcome, not a failure, and the calling rule owns the resulting diagnostic, so the `TryFrom`-with-`TypeError`
   idiom would have forced every access rule to rewrap a policy error it did not raise.
3. **Policy implementations must leave `C::Value` generic (new finding, now documented on the trait).** Restating
   `where C::Value: Add` in an impl that pins `C::Value` to a concrete type produces a trivial bound, which stable
   Rust rejects unless that concrete type really implements `Add` — and a downstream backend value family cannot
   implement `Add` directly at all, because the value-level arithmetic sugar is a blanket implementation whose
   disjointness a foreign crate cannot prove. The downstream test is written the correct way and is therefore also
   the standing proof of the per-function capability granularity: a view-less, non-accumulating universe gets the
   complete policy for reads and replacements without ever proving `Add`.
4. **`ReferenceDischargeTracer` is a real `Value` (met).** Capabilities backed by an operation need no bespoke
   delegation: their value-level sugar binds through the tracer's dispatch domain, which is the discharge context,
   so the operation's own rule performs the unwrapping and owns the rejection. Only operation-free capabilities such
   as `Concretizable` delegate directly, and both arms are tested. An explicit `Add` implementation was written and
   then removed once the blanket proved to cover the tracer.
5. **The driver's replay position (met, with one recorded limitation).** `RecursiveReferenceDischargeDriver` threads
   `InstructionId::new(region, index)` into every nested rule, and `Context::bind` supplies `None` as the plan
   prescribes for a direct bind. Consequently a program replayed through `Program::interpret_in_context` loses the
   coordinate, because `Context::bind` has no parameter for it. Both paths are pinned by tests. Phase 4 does not need
   a core-trait change to fix this: program-level discharge should run through `discharge_region` on the entry region,
   which already threads coordinates.

**Deviations from the plan, with rationale.**

- **The while prototype lives in the test module, not in a throwaway production module.** The plan's phase-6 estimate
  named a throwaway module, but the phase-0 text also says the prototype is built "on the test operation family". A
  test-module rule satisfies the intent, keeps the production surface free of code that phase 2 must delete, and is
  still end to end. It carries the `TODO(eaplatanios)` marker naming phase 2 as its deleting phase.
- **The prototype universe is eager-only.** Its policy pins `C::Value`, which is legal but restricts it to one
  destination. That is a deliberate simplification recorded in a code comment; the downstream test carries the
  value-generic shape a real universe must use.
- **The context takes no selection parameter yet.** Phase 0 lands the site vocabulary, enumeration, and validation;
  the selection knob itself belongs to phase 4 and was omitted rather than shipped as unused scaffolding.
- **`ReferenceRootState` is public.** The plan's sketch left it unmarked, but the context returns it from
  `root_state`, which third-party structured rules need, and a public trait cannot expose a private type.
- **One unrelated pre-existing failure was repaired.** `programs::operations::tests::test_operation` failed at `HEAD`:
  it asserted that `StopGradientOperation::infer_output_types` rejects an empty operand list, which that operation
  has never done. The owner's concurrent work in that module documents `stop_gradient` as variadic, so the stale side
  is the assertion, and it now checks the variadic contract instead. Without this the phase could not land on a green
  matrix.
- **Small shared additions.** `RegionRef::instructions_in_closure` was added beside the existing
  `contains_atom_in_closure` family and `effect_occurrences_in_closure` was re-expressed on top of it instead of
  repeating the traversal; the traversal now skips an attachment that names no arena region rather than resolving it,
  because three new validation entry points reach it. `ReferenceSource::from_input_index` and
  `ReferenceOperationSemantics::new_root_output_indices` were added so that the entry-boundary split and the
  allocation classification have one owner each rather than being re-derived here.

**Audit round.** Two independent Opus auditors reviewed the complete phase-0 surface, one for conventions and one for
correctness. Every real finding was fixed; the ones that changed the design rather than the prose are worth recording:

- **Reference-typed constants were being lifted as ordinary values**, which would have let one survive into the
  destination and break the reference-freedom guarantee at the source. Both lifting paths now route through one
  `lift_constant` seam that rejects them, on the grounds that a reference stored as a constant belongs to no root.
- **Root handles now carry the identity of the environment that minted them**, so a handle from an unrelated discharge
  is reported instead of silently addressing whichever root occupies the same position. A `fork` keeps its parent's
  identity, because a fork is a snapshot of one logical environment and a structured rule has to pass the handles it
  already holds into the probe.
- **`fork` takes its destination as a parameter.** Isolating the root environment is only half of isolation: under a
  staging destination the work a probe binds also has to go somewhere abandonable. Making the destination explicit is
  what lets phase 2 supply a child destination without changing the signature.
- **`discharge_reference_free_operation` landed**, which is the exit criterion for the default reference-free replay rule and its
  conversion seam. It rejects a region-carrying application rather than replaying it, because how region boundaries
  widen is knowledge that belongs to the operation. Both test universes now use it for their reference-free arms.
- **`bind_preserved` and `derive` validate their destination reference values** against the reference type the handle
  will expose, using the policy's existing projection rather than a new policy method.
- **Selection validation dedups across the whole selection before any kind check**, so the documented precedence is
  actually the one it follows, and it resolves reference semantics only for the instructions actually named instead of
  for every instruction in the closure.
- **`try_into_full` reports the smallest retained coordinate** rather than the first one the unordered traversal
  happens to reach.
- **Coverage added** for operand-kind mismatches, binding through an operation rule with a reference operand,
  reference-typed constant rejection, the reference-free replay seam, driver replay positions, the empty-driver rejection, and
  the visit-once behaviour of site enumeration across a twice-attached region. The downstream integration test gained
  a **staging-destination** case that pins the rendered discharged program, which is the first proof that the staged
  half of the architecture works end to end.

**Known gaps recorded rather than fixed.** The trybuild compile-fail test that §2.1 asks for is deferred (there is no
`trybuild` harness in `ryft-core`; the downstream integration test proves public reachability positively instead). The
policy has no referent-type lift, so a root's *state* type cannot be validated against its referent. And the while
prototype's two acknowledged simplifications — probing into the same destination, and an iteration bound taken before
probing — are recorded in its `TODO(eaplatanios)` comment as phase-2 requirements.

**The uncommitted `differentiation/forward.rs` test additions.** Those tests were reviewed against
`.agents/unit-testing-guidelines.md`. Their coverage intent is sound and none of it is redundant with owner-module
tests, so nothing was discarded; the guideline violations that are independent of phase sequencing were fixed in
place (rustdoc on private unit-test helpers demoted to ordinary comments, and trailing commas corrected on the
argument lists the diff introduced). One violation was deliberately left for phase 3, which owns the transform
adapters: `test_custom_jvp_operation_rejects_unresolved_references_in_dormant_rules` and its `custom_vjp` counterpart
assert behaviour implemented in `differentiation/operations/custom_jvp.rs` and `custom_vjp.rs`, and the guidelines put
colocated coverage in the owning module, so they and the `ReferenceRuleDifferentiationDriver` fixture belong there.
Moving them now would have collided with concurrent owner edits in that directory.

**Entry note for phase 1.** The rule trait is open with no defaulted body, which is the price of the capability-only
context bound: a default reference-free replay body would need `C: Context` and `Self: Clone + Into<C::Operation>` on the trait
itself, which is exactly the shape `PartiallyEvaluatableOperation<C: Context>: Clone + Into<C::Operation>` uses to get
its fully defaulted rule. Keeping the approved bound therefore means every non-reference payload needs a real
implementation rather than an empty one. That is not extra design work, only mechanical breadth: it is the same
per-payload sweep the repository already performs for `impl_non_transposable_operation!` and
`impl_non_differentiable_operation!`, so phase 1 should open by adding an `impl_reference_free_dischargeable_operation!` macro beside
those that emits a one-line implementation delegating to `discharge_reference_free_operation`, and then applying it across the
operation tree. Sizing note for §6: phase 1's operand is roughly eighty payloads across the operations tree, which is
larger than that section's estimate implies, though the per-payload cost is one line.

**Verification.** `cargo test -p ryft-core --lib`: 1574 passed, 0 failed, 3 ignored (23 of them in
`programs::references::discharge`). `cargo test -p ryft-core --tests`: 6 + 6 + 2 + 1 passed across the four
integration binaries. `cargo test -p ryft-core --doc`: 5 passed. `cargo test -p ryft-xla --lib`: 542 passed.
`cargo test -p ryft-pjrt --lib`: 130 passed. `cargo test -p ryft-macros-tests`: 20 + 17 passed.
`cargo check --workspace --all-targets`: zero warnings and zero errors. `rustfmt --check` on every file this phase
touched: clean. `cargo doc -p ryft-core --no-deps`: no warnings from any file this phase touched.

One unrelated pre-existing flake was observed once and is not caused by this phase:
`telemetry::tests::test_live_array_count_tracks_array_construction_and_drop` in `ryft-xla` asserts on a process-global
live-array counter, so it fails intermittently under parallel execution; it passes in isolation and passed three
consecutive full-suite runs afterwards.

### Phase 1 — flat interpreter discharge (landed)

**What landed.** The complete flat reference language now discharges through per-operation rules, and both real
operation families (`ArrayIrOperation` and `XlaOperation`) participate through a generated dispatcher. The planner
remains production; the interpreter runs beside it in tests.

- **`impl_reference_free_dischargeable_operation!`** (`crates/ryft-core/src/macros.rs`), beside `impl_non_transposable_operation!`
  and accepting the same four invocation forms. It emits the shared reference-free replay rule by delegating to
  phase 0's `discharge_reference_free_operation`, which rejects an application that carries regions and an operand that is a
  live reference handle.
- **The five generic reference-primitive rules** (`programs/references/operations.rs`), one per payload. None of them
  names the referent type parameter: allocation reads its fresh root's reference type back out of its *own* inferred
  output type through `ReferenceDischargePolicy::project_reference_type`, and every access reads the type off the
  flowing handle. They are therefore universe-generic, which is what lets the same five rules serve the arrays
  universe, the in-crate list prototype, and the downstream register universe.
- **The arrays policy** `ArrayReferenceDischarge` (`arrays/reference_discharge.rs`), whose referent is `ArrayType` and
  whose alias is the composed `ArrayReferenceView`. Its three alias applications run the *same* `ArrayReferenceView`
  traversal the eager handles use, through a new `DestinationViewCarrier` that binds canonical slice, reshape, and
  update-slice operations into the destination. Staged and eager reference semantics therefore still cannot drift.
- **The two SSA view rules** for `reference_index` and `reference_slice` (`arrays/operations/references.rs`). Each
  validates and derives the composed referent type with exactly the eager handle's arithmetic, records the composed
  chain as the derived handle's authoritative alias, and binds nothing: the coordinates are materialized per access.
- **`Program::discharge_references_with_policy`** (`programs/references/discharge.rs`), the program-level entry. It
  traces into a fresh destination of the program's own universe, turns each reference-typed input into a state input
  at the same boundary position, replays the entry region through `discharge_region` (not `interpret_in_context`, so
  instruction coordinates are threaded), appends the final state of each *mutated* external root as a hidden output in
  entry-boundary order, and proves the result reference-free by assembling a `PartialReferenceDischargeResult` and
  converting it through `try_into_full`.
- **The `dispatch(discharge)` derive** (`ryft-macros`), with the composite-native delegation and a verbatim-replay
  fallback, plus stand-in coverage in `ryft-macros-tests` and the canonical derive documentation in
  `programs/operations.rs`.
- **The dual-run parity harness** (`arrays/reference_discharge.rs`), which discharges one source with both
  implementations and compares the rendered program, both boundaries, the public-output prefix, the external-state
  bindings (whose final-state indices are the hidden-output order), and the discharged program's effects.
- **Both phase-0 deferrals.** `ReferenceDischargePolicy::lift_referent_type` landed because the entry boundary needs
  it, and the trybuild compile-fail proof landed as two cases under `crates/ryft-core/tests/reference_discharge/`,
  driven from the existing downstream-surface integration test. They are two files rather than one because their
  rejections belong to different compiler passes and only the earlier one is reported when they share a file.

**Decisions this phase forced.**

1. **Accumulation is its own policy trait (`ReferenceAccumulationPolicy`).** Section 2.1 put `accumulate` on the one
   policy behind a function-level `where C::Value: Add` clause, on the premise that a staged `Tracer` implements `Add`
   by staging. That premise does not hold for Ryft's *production* reference universe. `ArrayIrOperations` states
   explicitly that homogeneous array capabilities such as `Add` are deliberately **not** composite members, and
   `AddOperation<ArrayIrType>` is a lift marker rather than an `Operation`, so no composite array-IR value — eager or
   staged — implements `Add`. A single trait-level requirement cannot serve both a value-capability universe and an
   operation-lifting universe, and a requirement the implementation cannot use is worse than none. Splitting
   accumulation into a super-trait-refining contract preserves the approved *semantics* — compile-time unavailability
   scoped to exactly `reference_add_update`, with a runtime `UnsupportedOperation` rejection as the distinct second
   case — and in fact strengthens the standing proof: the downstream register universe now declines accumulation by
   not implementing the trait at all, rather than by never being asked.
2. **The derive's fallback shrinks the sweep from roughly eighty payloads to twenty-five.** The phase-0 entry note
   sized the sweep as one implementation per operation payload. The generated dispatcher instead replays every
   projected, mixed, and generic-extension variant as the *whole enum* through `discharge_reference_free_operation`, which is
   both correct (the destination receives the identical operation) and free of new predicates. Only the
   composite-native payloads of a dispatching enum need an implementation.
3. **Region-carrying payloads get rejecting placeholders, not silence.** The seven region-carrying composite-native
   payloads in `ryft-core` plus `jit_call` and `shard_map` in `ryft-xla` take the same macro, whose rule rejects a
   region-carrying application by name. Each site carries a `TODO(eaplatanios)` naming phase 2 as the phase that
   replaces it with the operation's own widening rule.
4. **The arrays policy needs an operation-family seam for the canonical view constructors and the lifted addition.**
   It initially reused `ArrayReferenceDischargeOperation`, which already provides both; the audit round split the
   constructors into `ArrayReferenceViewOperation` instead, so that the interpreter states only what it uses and phase
   3 can delete the planner's classification half without touching it.
5. **`ArrayOperation` is deliberately not opted in.** Its eighty variants are reached through
   `ArrayIrOperation::Array`, which the fallback replays whole, so the homogeneous array family needs no dispatcher.

**Deviations from the plan, with rationale.**

- **The accumulation split** above is a deviation from §2.1's single-policy sketch. It is recorded there in the two
  traits' documentation as well.
- **The arrays operation-family seam was split after all.** Decision 4 above deferred it; the conventions audit
  correctly held it against the minimum-bounds rule, since the interpreter would otherwise depend on precisely the two
  planner methods phase 3 deletes. `ArrayReferenceViewOperation` now owns the three canonical view constructors and
  `ArrayReferenceDischargeOperation` inherits it, so phase 3 deletes the classification half and leaves the
  interpreter untouched.
- **One deliberate parity divergence: dead constants.** The planner copies every stored constant into the destination
  before replaying anything; the interpreter reaches the destination through the shared program replay path, which
  lifts only constants something still consumes. A source containing a dead constant therefore discharges to a program
  with one fewer atom. Dropping it is what every other transform already does and is strictly the better artifact, so
  the interpreter keeps that behavior; the divergence is pinned by
  `test_interpreter_discharge_omits_a_dead_constant_the_planner_retains` so that phase 3 confirms no pinned rendering
  moves. Nothing in the existing suite contains a dead constant.
- **Rejection-diagnostic parity is narrower than §5 implies for flat programs.** Most planner rejections are
  contract checks on its own `reference_discharge_rule` classification, which the interpreter has no analogue for
  because it has no classification to violate. The interpreter's own rejections — use-after-consume, a consumed
  external root, an oversized capture prefix — are covered by
  `test_interpreter_discharge_reports_environment_and_boundary_failures`.
- **Full discharge routes through the partial envelope.** `discharge_references_with_policy` returns
  `ReferenceDischargeResult` by proving reference freedom with `PartialReferenceDischargeResult::try_into_full`
  rather than asserting it, which is exactly the reuse phase 0 built the envelope pair for.
- **One unrelated repair.** `differentiation::types::tests::test_dense_array_coordinate_basis_stages_ordinary_primitives`
  failed at `HEAD`: commit `e02c16788` introduced a rectangular coordinate-basis path that stages `zero` before `one`
  while the test in the same file still expected the opposite order. The staging order of the two `select` constants
  is not a semantic contract, so the expectation was corrected. Without this the phase could not land on a green
  matrix.

**Audit round.** Two independent Opus auditors reviewed the complete phase-1 surface, one for correctness and one for
conventions. Every real finding was fixed; the ones that changed behavior rather than prose are worth recording:

- **One blocker: `freeze` through a derived view produced a silently ill-typed program.** `consume` yields the whole
  root by design and deliberately ignores the handle's alias, delegating the root-handles-only restriction to the
  standalone analysis and the authored-program lint — neither of which discharge runs. A builder-constructible program
  that froze a `reference_slice` view therefore discharged to a program whose output carried the *root's* type instead
  of the view's, where both the eager handles and the planner reject it. `consume` now enforces the invariant it
  relies upon, comparing the root's state type against the handle's lifted referent, which the new
  `lift_referent_type` makes a universe-generic check.
- **`reference_swap` and `reference_add_update` re-derive their own inference over the carriers they receive.** A
  program built through a `ProgramBuilder` already ran that inference, but a direct `Context::bind` into a discharge
  context does not, and both rules relate two operands to each other in a way the carriers alone cannot recover: a
  universe whose write mechanics only require the replacement to *fit* the selected coordinates would otherwise
  perform a silent partial write, which is exactly what the eager handles validate against. The allocation rule
  already derived its own output type; the read and freeze rules relate no operands and so stay lean.
- **The parity contract is now actually run over the flat suite.** `discharged_by_both` was reached from one test;
  every flat test in the existing suite now goes through it, including the exhaustive generated-program enumeration,
  which puts forty programs through both implementations per run.
- **The interpreter surface no longer depends on the trait phase 3 deletes.** The three canonical view constructors
  split out of `ArrayReferenceDischargeOperation` into `ArrayReferenceViewOperation`, which the planner keeps as a
  super-trait. The interpreter policy and its carrier now state only that narrower contract, and the accumulation
  policy adds only the lifted-addition conversion it actually binds.
- **Coverage added** for the two array view rules in their owning module (composition, the invalid-composition
  rejection, operand-kind and arity rejections, and the preserved-root rejection), the region-carrying phase boundary
  on the real `ArrayIrOperation` family, the freeze-through-view rejection end to end, `impl_reference_free_dischargeable_operation!`
  in the module that defines it, the program-level entry point over the *downstream* universe (which also proves it is
  universe-generic), and the allocation rule's projection-disagreement rejection.

**Known gaps recorded rather than fixed.** Region-carrying operations reject rather than discharge, which is phase 2's
scope, so the interpreter currently serves flat programs only. Preserved roots are rejected by `derive` because
partial discharge is phase 4; the view rules note where the preserved replay attaches. The planner and the interpreter
both remain live, which is the two-implementation window §7 calls out and phase 3 closes.

Three narrower gaps are recorded deliberately:

- **The planner's contract checks have no interpreter counterpart, by design.** `validate_discharge_support`'s
  `matches_primitive`, `positional_regions`, `positional_outputs`, and alias-shape checks exist because the planner
  must trust an arbitrary operation that *claims* a rewrite rule; §1 is explicit that those contracts disappear when
  each operation discharges itself. Their tests (`test_discharge_rejects_read_rule_with_non_canonical_boundary_types`
  and its four siblings) are therefore planner-only and are deleted with it at phase 3. The one behavior worth naming:
  a third-party family that declares a canonical read while inferring a different output type is rejected by the
  planner and simply produces the root's own value under the interpreter, which is the house answer (builder-time type
  checking, the result envelopes, and oracles) rather than a per-operation contract.
- **The dead-constant divergence needs a rendering sweep before cutover.** Phase 3 must check the `ryft-xla` stateful
  suites and every other pinned discharged rendering for a retained dead constant, because those pins move at cutover.
- **`ryft-core` now carries a `trybuild` harness**, which costs the downstream-surface integration test roughly
  fifteen seconds and pins two rustc diagnostics verbatim. Regenerate the snapshots with `TRYBUILD=overwrite` after a
  toolchain bump rather than editing them.

**Verification.** `cargo test -p ryft-core --lib`: 1585 passed, 0 failed, 3 ignored. `cargo test -p ryft-core --tests`:
1585 + 6 + 6 + 4 + 1 passed across the five integration binaries (the downstream-surface binary now also drives two
`trybuild` compile-fail cases). `cargo test -p ryft-core --doc`: 5 passed. `cargo test -p ryft-xla --lib`: 540 passed,
5 ignored. `cargo test -p ryft-pjrt --lib`: 130 passed. `cargo test -p ryft-macros`: 57 + 1 passed.
`cargo test -p ryft-macros-tests`: 21 + 17 passed. `cargo check --workspace --all-targets`: zero warnings and zero
errors. `rustfmt --check` on every file this phase touched: clean. `cargo doc -p ryft-core --no-deps`: no warnings from
any file this phase touched.

Phase-1 size, against §6's estimate: roughly 1,150 added lines across the tracked files plus about 500 in the two
untracked `programs/references` modules, split between production, tests, and documentation in about the proportion
§6 projected. The one estimate that was materially wrong is the phase-0 entry note's sizing of the payload sweep: the
generated dispatcher's verbatim-replay fallback means only a dispatching enum's composite-native payloads need an
implementation, so the sweep is eighteen one-line invocations rather than eighty.

### Phase 2 — structured interpreter discharge (landed)

**What landed.** Every region-carrying operation now discharges itself, and the nine rejecting placeholders are gone.
The planner remains production; the interpreter runs beside it, and the complete structured half of the existing
discharge suite now goes through both.

- **The transactional fork.** `ReferenceDischargeDriver::discharge_region_program` is the third driver service. It
  rebuilds one attached region against an *isolated* environment over a fresh destination of the same universe and
  returns the sealed `ReferenceRegionDischargeFork`: the rebuilt region program, the caller root each declared region
  output denotes, and the threaded roots the region actually mutated. It carries no values of any kind, which is what
  makes the isolation a type-level fact rather than a convention. The rule requests a boundary through
  `ReferenceRegionDischargeBoundary`, which describes the declared positions (`ReferenceRegionDischargeInput::Value`
  or `::State`) separately from the added state inputs and outputs and their insertion positions — separately because
  only the declared positions are replayed: an added input exists in the rebuilt boundary and in the caller's operand
  list, but the source region's body never named it.
- **The transitive closure summary.** `ReferenceDischargeContext::region_summary` reports, in caller-root terms, which
  roots a region closure accesses and which of them it mutates, plus the caller root each declared region output
  denotes. It is computed from generic hooks alone: operation-local `reference_semantics`, `input_region_provenance`,
  `reference_output_identity_input`, `output_region_provenance`, and recursive summaries of nested regions. Roots
  allocated inside the closure are absent by construction, because they cross no boundary.
- **The shared positional rewrite.** `discharge_positional_region_operation` serves the two structured shapes whose
  regions mirror the operand list after a constant leading offset and whose results are their regions' outputs: a
  condition (leading offset one, for its predicate) and a positional call (offset zero). It applies the read-only
  pruning — only roots some closure mutates gain an appended output — and gives every attached region one identical
  boundary, so a rebuilt condition's branches keep agreeing with each other.
- **The four structured rules.** `ConditionOperation` and `JitCallOperation` delegate to that rewrite in one line each.
  `WhileOperation` and `ScanOperation` own theirs, because a loop is boundary-symmetric rather than pruned: every
  threaded root occupies a carry position in the operand list, the output list, and both region boundaries, or in none
  of them. The `while` rule additionally enforces the asymmetry its own contract forces (the condition receives the
  entering state and publishes none, so a mutating condition is rejected) and the carry fixed point (a body that does
  not return a carry as the reference it entered with is rejected). The `scan` rule inserts its synthesized carries
  immediately after the declared carry prefix — in the operand list, in the body's inputs, and in both output
  boundaries — and grows the payload through `ScanOperation::with_added_carries`, which preserves length, direction,
  unroll factor, and captures.
- **Region-free region carriers replay verbatim.** `discharge_reference_free_operation` now copies a region-carrying
  application's regions into the destination when nothing in their closure touches a reference, and rejects the
  application with the planner's own `\`{name}\` carries reference state but has no reference discharge rule` message
  when anything does. That is exactly the planner's `Ordinary` behavior, so `linear_call`, `custom_jvp`, `custom_vjp`,
  `rematerialize`, and `shard_map` keep `impl_reference_free_dischargeable_operation!` and each carries a comment stating why
  threading state through it has no defined meaning rather than a `TODO`.
- **Root reference types moved into the environment.** A structured rule threading an inherited root holds only that
  root's handle, never a handle it could read a type off, so the environment now records each root's whole-root
  reference type and exposes it through `ReferenceDischargeContext::root_reference_type`.
- **The prototype universe became value-generic and gained a structured operation.** Its policy now reaches its alias
  mechanics through two new pure operations (`list.select`, `list.splice`) bound into the destination instead of
  through a pinned eager value, and the family gained a positional `list.call`. The driver services therefore have
  owner-module coverage in a universe that mentions no arrays, and the standing non-array proof now covers a staging
  destination as well as an eager one.

**Decisions this phase forced.**

1. **The nested-destination obligation belongs to the recursive driver and must never reach a rule.** Rebuilding a
   region needs this universe's operations to discharge into `ReferenceDischargeRegionDestination<C>` — a fresh
   `TracingContext<C::Constant, C::Operation>` — as well as into the live destination, and the two policy
   instantiations must agree on their referent type system. Stating that on the *rules* does not compile: the enum
   dispatcher's predicate for a structured payload would then demand that the whole enum discharge into the
   destination whose dischargeability is what the graph is trying to establish, which is a genuine `E0275` (confirmed
   on a reduced case before the architecture was chosen, not discovered late). Stating it on
   `RecursiveReferenceDischargeDriver`'s driver implementation breaks the cycle, because nothing reachable from a
   `ReferenceDischargeableOperation` implementation names it. This is the same placement `RecursiveBatchingPolicy` and
   `RecursivePartialEvaluationDriver` already use, and it is why the four structured rules carry plain bounds.
2. **A region fork mints a fresh environment rather than snapshotting the caller's.** The fork's environment holds
   exactly the roots its boundary names, each entering as an ordinary value at its own position, and its own
   environment identity means a caller handle cannot address a fork root and a fork handle cannot address a caller
   root — a leak in either direction is *reported*, not merely absent. One private table relates the two, and the fork
   reports its results in caller terms. This is strictly stronger than the phase-0 snapshot, so
   `ReferenceDischargeContext::fork` was deleted rather than left as an unused isolation primitive; the clone-sharing
   contract it also covered kept its test.
3. **The destination of a rebuilt region is a fresh root trace, not a nested one.** A rebuilt region is
   self-contained: its complete interface is its own boundary, so it must not close over a value of the destination it
   will be attached in. Being a root trace is also what makes the type a fixed point of its own construction, which is
   what keeps decision 1's obligation finite. `NestedTracingContext<C>` would not terminate.
4. **The prototype universe had to stop pinning its destination value.** Decision 1 makes every discharge destination
   support a staging child, so the phase-0 recorded simplification became untenable. Reworking the prototype to bind
   its alias mechanics through its operation family is what a real universe does anyway, and the arrays policy already
   did it.
5. **Consuming a caller root inside a region is rejected.** A consumed root has no successor state, so no symmetric
   boundary and no final-state output can describe what became of it. The summary service rejects it where it is
   discovered. This is stricter than the standalone analysis, which permits the access and leaves the caller holding
   state that is no longer live.
6. **A derived view cannot cross a structured boundary.** A state boundary carries whole-root values, so a handle
   exposing a narrower referent would silently widen to the root's own value as it crossed. The
   `ReferenceDischargeContext::operand_root` seam rejects it by name and says to derive the view inside the region,
   which is what the module documentation already required and what the analysis enforced ahead of the planner.
7. **Symmetry is a property of a loop's boundaries, not a claim that the loop wrote what it carried.** A loop
   returns a successor state for every root it carries, including roots its closure only read, so merging that state
   back with `set_discharged_state` — which records a mutation — would publish a hidden final-state output for a root
   the program never writes, and make its caller write an unchanged holder back. That is also a parity break, because
   the planner takes its mutated set from the analysis rather than from what crossed a boundary. The context therefore
   gained `thread_discharged_state`, which installs a carried-out state without recording a mutation, and
   `merge_discharged_state`, which picks between the two from the summary. All three structured rules merge that way,
   including the positional rewrite's declared reference outputs, which have the same property. `is_mutated` on a
   `ReferenceStateBinding` consequently keeps meaning "the program writes this root", not "the discharged state is a
   different atom from the entry value". Pinned by a read-only-loop parity test.
8. **The summary reports declared output roots, and `merged` keeps the receiver's.** A region that *returns* a root
   already publishes that root's final state at its own output position, so a rule must not publish it a second time —
   which is how the planner's `represented_output_roots` pruning is reproduced without an analysis. Merging is how one
   operation's several regions agree on one shared *state* boundary; declared output roots belong to one region's own
   boundary and are therefore not merged, and an operation whose regions must agree on them has that agreement checked
   against the rebuilt regions themselves.

**Deviations from the plan, with rationale.**

- **No fixed-point probing.** §2.1 and the phase-0 prototype reached the loop-widening set by probing the body until
  the mutated-root set stopped growing. With a real summary service the set is computed statically from the source
  closure before anything is rebuilt, exactly as the planner computes it, so the loop rules converge by construction.
  The isolation the probe language was protecting is unchanged and in fact stronger (decision 2): every structured
  region is rebuilt against an environment that commits nothing. The summary's prediction is still held to the replay,
  through `ReferenceRegionDischargeFork::validate_predicted_mutations`, which is the phase-0 prototype's commit check
  kept as a standing invariant rather than as a loop termination condition.
- **Five of the nine placeholders became verbatim replay rather than widening rules.** §4's phase 2 names "condition,
  while, scan, and call rules", and §5 requires exact parity with the planner, which classifies `linear_call`,
  `custom_jvp`, `custom_vjp`, `rematerialize`, and `shard_map` as `Ordinary` and rejects reference state in their
  closures. Giving them real widening rules would exceed the oracle and be untested by construction, so they keep
  planner behavior with the planner's own diagnostic. Each site now records the semantic reason rather than a phase
  marker: a dormant derivative rule region has no derivative for a mutation, rematerialization is sound only when
  recomputation is unobservable, and a manual-SPMD boundary does not define which shard owns a referent.
- **Rejection-diagnostic parity stays narrower than §5's literal reading, in the direction §5 already records.** The
  interpreter's structured rejections — a consumed caller root, a derived view crossing a boundary, a branch-local
  allocation escaping, an unpredicted mutation, a non-fixed-point carry — have no planner counterpart, because the
  planner delegates those to the standalone analysis it runs first. They are covered by their own tests.

**Audit round.** Two independent Opus auditors reviewed the complete phase-2 surface. One of them additionally built
a throwaway crate outside the workspace and ran eight further planner-versus-interpreter parity probes of its own
devising — a branch that both returns and mutates a reference, a branch that returns a read-only reference, the same
root passed as two operands, a read-only scan carry, a `while` nested inside a condition branch, a branch-local
allocation beside an external mutation, branches returning different roots, and a `while` body that swaps its carries.
All eight agreed with the planner exactly, and every malformed variant was rejected by both implementations. Every real
finding was fixed; the ones that changed behavior rather than prose are worth recording:

- **The summary silently dropped accesses it could not resolve.** A reference-typed atom the traversal never bound
  denotes a reference that entered the region other than through its boundary — today, a capture-scoped reference
  constant. The traversal ignored the access and let the failure surface much later, from inside a rebuilt region, as
  a lift-time complaint that no longer named the operation that performed the access. The summary now reports it where
  it is discovered, which is also what makes the recorded capture gap visible at its real boundary.
- **The loop rules' mutation cross-check was vacuous.** `while` and `scan` passed *every* accessed root as the
  published set, so `validate_predicted_mutations` could never fire for them — and their entry-boundary mutation flag
  comes from the summary rather than from the replay, which is exactly the direction the check exists to guard. Both
  now publish only the roots the summary calls mutated, which is what the positional rewrite already did and makes the
  three rules state one thing.
- **Nothing held the replay's declared output roots to the summary's.** The widening reads which outputs denote roots
  from the static summary and sizes the boundary from it, while the result mapping consumed the fork's. A family whose
  provenance hooks disagreed with its own rule could therefore lose an update silently.
  `ReferenceRegionDischargeFork::validate_predicted_output_roots` now holds one to the other, which also subsumes the
  weaker check that only compared an operation's several regions against each other.
- **`allows_reference_access_through_region_input` was ignored.** The summary now honors it for nested regions, and
  the `while` rule restates it for its own condition region, so a mutating `while` condition is reported against the
  contract it violates instead of as a widening that a rebuilt region contradicted.
- **`thread_discharged_state` was folded into `merge_discharged_state`** rather than published as a second entry point
  with no caller of its own.
- **Coverage and prose.** The zero-length scan test joined the parity harness, which was the last structured test the
  §9 claim of complete structured parity did not cover; the program-level entry point's rustdoc no longer claims the
  interpreter "serves flat programs only"; missing `# Errors` sections were added to the fork services, the two
  validators, and the rule trait; and the root rendering changed from `reference root 6.0`, which reads as a number in
  prose, to `reference root 6:0`.

Two findings were deliberately deferred with rationale rather than fixed. `ReferenceDischargeEnvironmentId` is public
and re-exported while its only out-of-module use is a compile-fail fixture; that is phase-0 surface, and §4's phase-7
deletion audit is where surface reduction belongs rather than a phase boundary. And the three structured rules were
left stating their threaded-root set slightly differently — `while` and `scan` derive it from accesses alone, where the
positional rewrite adds the declared output roots — because for a loop the identity hook makes the declared output
roots a subset of the carries, so the sets are provably equal; the second auditor examined this and reached the same
conclusion independently.

**Known gaps recorded rather than fixed.**

- **The synthesized-state path is correct but currently unreachable end to end, and that is the same capture gap.**
  A rule threads a root as an *added* boundary position exactly when a region closure reaches a root that is not one of
  its declared operands. In the interpreter the only source construct that could produce one is a reference-typed
  constant, which `lift_constant` rejects, so `entering` is empty for every universe today and only the planner
  exercises synthesized carries (through the capture-reference constants the `test_closed_program_discharge_*` tests
  pin). The machinery is nevertheless the planner's own `added_input_roots` behaviour and must not be dropped: cutting
  it would silently remove a capability those tests pin. Its boundary-insertion arithmetic therefore has direct
  coverage against the fork service
  (`test_reference_discharge_region_program_inserts_added_state_at_its_boundary_position`), and **phase 3's cutover is
  blocked until the interpreter path can bind region-closure roots** — through a capture scope on the discharge
  context, or by resolving reference-typed constants against an outer environment.
- **Capture-reference constants are a phase-3 prerequisite, and the gap is a typing gap rather than a runtime one.**
  A capture-lifted program names its caller's references through `CaptureReference` constants inside nested regions,
  which the planner resolves against the analysis's capture scopes. The interpreter has no capture scope: it threads
  roots through its environment and rejects a reference-typed constant outright. Worse, `discharge_references_with_policy`
  cannot even be *instantiated* for such a program, because the derived dispatcher for `ArrayIrOperation<A>` pins
  `Constant = ArrayIrValue<A>` while a capture-lifted program's constants are `CaptureReference<ArrayIrType>`. The five
  `test_closed_program_discharge_*` tests therefore stay planner-only. Phase 3 must resolve this before cutover, by
  either relaxing the dispatcher's constant pinning and giving the discharge context a capture scope, or resolving
  capture references into ordinary roots ahead of the interpreter.
- **The planner remains production.** Phase 3 closes the two-implementation window.

**Verification.** `cargo test -p ryft-core --lib`: 1595 passed, 0 failed, 3 ignored. `cargo test -p ryft-core
--tests`: 1595 + 6 + 6 + 4 + 1 passed across the five integration binaries. `cargo test -p ryft-core --doc`: 65 passed
and 16 ignored, plus 5 compile-fail cases. `cargo test -p ryft-xla --lib`: 541 passed, 5 ignored. `cargo test -p
ryft-pjrt --lib`: 130 passed. `cargo test -p ryft-macros`: 57 + 1 passed. `cargo test -p ryft-macros-tests`: 21 + 17
passed. `cargo check --workspace --all-targets`: zero warnings and zero errors. `rustfmt --check` on every file this
phase touched: clean. `cargo doc -p ryft-core --no-deps`: no warnings from any file this phase touched.

The `ryft-xla` `telemetry` flake recorded in phase 0 did not recur; it was also rerun single-threaded and passed. The
`stablehlo.broadcast_in_dim` diagnostic that `ryft-xla` prints during its run is MLIR output from a passing negative
lowering test, exactly as in the phase-1 baseline.

Phase-2 size, against §6's estimate: roughly 1,050 added lines in `programs/references/discharge.rs` (the fork and
summary services, the shared positional rewrite, the prototype universe's value-generic rework and its `list.call`),
about 300 across the three control-flow rules, about 70 in `ryft-xla`, and about 750 of tests, against §6's projected
~400/350/80 plus ~200/150/40. The overrun is concentrated in two places §6 did not anticipate: the prototype universe
had to become value-generic (decision 4), and the boundary vocabulary is richer than "boundary descriptors" implies
because declared and added positions must be described separately.

### Phase 3 — cutover (landed)

**What landed.** Production reference discharge is the interpreter. The centralized planner, its bespoke replay, and
the value-family-independent plan surface are gone, and the two prerequisites phase 2 recorded were discharged first,
as a contracts-first increment recorded in §2.1 before any of it was written.

*The capture-scope increment (blocker 1).*

- **`ReferenceCaptureScope<Constant>`** (`programs/references/discharge.rs`) pairs the roots one capture prefix binds
  with the constant-family seam that recognizes a capture. The seam is a function pointer rather than a
  [`CaptureConstant`] bound, because the interpreter deliberately serves constant families that are not
  capture-bearing — the in-crate list universe and the downstream register universe both decline it — and because it
  is the same higher-order seam `RegionRef::analyze_references_with_capture_indices` already uses, reduced to a
  `Copy`, allocation-free carrier. `Default` recognizes nothing, so every pre-existing instantiation is unchanged.
- **The scope rides on the context**, beside the root environment, and is fixed for the life of one context: a
  context replays exactly one region body, and a structured rule that needs another region forks a new one. It is
  installed after the boundary has minted its roots, through the private `under_captures`, which shares the
  environment. `lift_constant` resolves a reference-typed constant through it — to the *whole-root handle of the root
  that capture already binds*, never to a second root, exactly as the analysis reuses the entry root — and
  additionally holds the constant's declared reference type against the bound root's.
- **`summarize_region_closure` seeds reference-typed constant atoms from the scope**, so a capture-scoped access is
  summarized instead of reported as unresolvable. That single change is what makes a structured rule's `entering` set
  non-empty and finally reaches the synthesized state-carry paths phase 2 could build but not exercise (blocker 2).
- **Nested scopes are derived from the existing generic hook.** `nested_capture_scope` reads
  `Operation::region_capture_input_count`: a region declaring its own leading capture prefix establishes a fresh
  scope over the roots that prefix binds, and every other region inherits its parent's. `region_summary` therefore
  takes the owning operation and region index and derives the scope itself, so no rule reasons about captures, and
  `ReferenceRegionDischargeBoundary` gained `capture_input_count` so the region fork can do the same. The fork's
  scope names only *fork* roots — the fresh case reads them off its threaded declared inputs, the inherited case maps
  the caller's scope through the same private caller-to-fork table the boundary already uses — so the sealed-fork
  isolation contract is unchanged.
- **`Program::discharge_references_with_lifted_captures_and_policy`** is the capture-aware entry point, requiring
  `V: CaptureConstant`; it and `discharge_references_with_policy` share one private body that takes the seam as a
  parameter, mirroring the analysis's own pair.
- **A reference-free program is its own discharge** and is returned untouched by that shared body, which reproduces
  the deleted planner's identity fast path generically. It is not only cheaper on the two transform adapters that
  discharge unconditionally: re-tracing would renumber the program's atoms, drop its dead constants, and abandon the
  region transform cache its regions carry, all for a rewrite with nothing to rewrite.

*The cutover.*

- `ReferenceDischarge::discharge_references` for array-IR programs, `Program::discharge_references_with_lifted_captures`,
  and `ClosedProgram::discharge_references` are now one-line forwards to the interpreter under
  `ArrayReferenceDischarge`. Their operation bound changed from `ArrayReferenceDischargeOperation` to
  `ArrayReferenceViewOperation + ReferenceDischargeableOperation<TracingContext<V, O>, ArrayReferenceDischarge>`.
- **Deleted, with nothing left behind:** `validate_discharge_support` and its `PrimitiveReferenceContract` oracle,
  `discharge_with_analysis`, `StagedViewCarrier`, `stage_view_access`, `stage_reference_view_reconstruction`,
  `discharge_region`, `discharge_higher_order_instruction`, `root_referent_type`, `attached_root`,
  `discharged_instruction_inputs`, `mapped_value`, `map_source_output`, `analyzed_input_root`, `current_state`, and
  `verify_discharged_program` (all in `arrays/reference_discharge.rs`, ~800 lines); the whole plan surface
  `ReferenceDischargePlan` / `ReferenceRegionDischargePlan` / `ReferenceInstructionDischargePlan` /
  `ReferenceRegionDischargeLayout` with `plan_region`, `analyzed_root`, `attached_root`, and
  `ReferenceDischargeRule::is_structured` (~385 lines in `programs/references/discharge.rs`) together with their
  `pub(crate)` re-export chain; `ArrayReferenceDischargeOperation::with_added_reference_scan_carries` and its two
  real implementations plus two test stubs; and, on the analysis side, the now-unconsumed `region_summaries` field
  with `ReferenceAnalysis::region_summary` and `is_reference_free`. `arrays/reference_discharge.rs` went from 3,880
  lines to 2,545 and `programs/references/discharge.rs` from 5,467 to 5,242.
- **Surviving provenance-hook consumers, named.** No hook is planner-exclusive, so none was removed.
  `input_region_provenance` and `reference_output_identity_input` are consumed by
  `ReferenceDischargeContext::region_summary`'s `summarize_region_closure` and by
  `RegionRef::analyze_references_with_capture_indices`; `output_region_provenance` additionally by
  `forwarded_output_root` and by `RematerializationCandidate::resolve_producers` (which is what keeps
  `LinearCallOperation::output_region_provenance` load-bearing — rematerializing a linearized program). Differentiation
  provides these hooks but consumes none of them. `region_capture_input_count` gained a second consumer in
  `nested_capture_scope`. `ReferenceDischargeRule` and `ArrayReferenceDischargeOperation::reference_discharge_rule`
  also survive: their remaining consumer is `PreservedReferenceKernelOperation::validate_body` in `ryft-xla`, which
  *decides about* a reference program without rewriting it.
- **The standalone analysis keeps exactly two production consumers**, which is the inventory phase 6 starts from:
  the eager lifetime preflight (`ArrayIrValue::validate_eager_replay` calling `validate_reference_lifetimes`) and
  that same preserved-reference kernel validator (`analyze_array_references`). Discharge itself no longer runs it.
- **The parity window is closed.** `discharged_by_both` and the capture-lifted `closed_discharged_by_both` compared
  the interpreter against the planner; with the planner gone they would compare the interpreter with itself, so both
  were deleted and their callers now call `discharge_references` directly. The behavioral oracles §5 names — eager
  reference semantics, the exhaustive generated-program enumeration, and the pinned renderings — are untouched and
  remain the ground truth.

**Decisions this phase forced.**

1. **The derive's discharge dispatcher must not pin the parent context's constant type.** The generated
   `ReferenceDischargeableOperation` impl required `__ParentContext: Context<Constant = #program_constant_type, ..>`,
   which no rule body and no shared rule helper reads, and which made the dispatcher inapplicable to precisely the
   programs the cutover had to serve: a capture-lifted array-IR program carries `CaptureReference<ArrayIrType>`
   constants while `ArrayIrOperation<A>` declares `ArrayIrValue<A>`. Dropping that one predicate — from the discharge
   dispatcher only, leaving interpretation and partial evaluation pinned — is what let `discharge_references_with_policy`
   be instantiated for it at all. `XlaOperation<Constant>` never had the problem because it is parameterized by its
   constant, which is why the XLA half of the interpreter already worked.
2. **A capture constant yields the bound root's handle, not a new root.** Minting a second root would give the same
   caller reference two identities and two hidden final-state outputs. Reusing the entry root is what the analysis
   does at `ReferenceRoot::RegionInput { region: scope, input_index }`, and it is why capture-scoped references
   perturb neither the external-state binding order nor the hidden-output order.
3. **The nested-scope rule belongs to the summary and the boundary, not to the rules.** `region_summary` takes the
   operation and region index and derives the scope; the boundary carries only `capture_input_count`. A structured
   rule therefore never mentions captures, which is what keeps a third-party structured operation free of a companion
   capture declaration beyond the hook it already implements.
4. **The reference-free identity fast path is not reproduced.** The planner returned a reference-free program
   untouched; the interpreter re-traces it. That is the same program for every case in the suite (pinned by
   `test_reference_free_discharge_is_identity`) except for a dead constant, which is decision 5.
5. **The dead-constant divergence resolved in the interpreter's favor and moved no pin.** Every pinned discharged
   rendering in the workspace was swept before the cutover: sixteen program renderings in
   `arrays/reference_discharge.rs` were already asserted equal under both implementations by the parity harness, the
   two `ClosedProgram` renderings and the five `ryft-xla` StableHLO snapshots were read in full and contain no dead
   constant, and the `ryft-xla` suites passed unchanged after the cutover, which is the strongest evidence available
   that the production path's output is byte-identical. The divergence test itself became
   `test_reference_discharge_omits_a_dead_constant`, which now pins only the surviving behavior.

**Deviations from the plan, with rationale.**

- **§4 says "No new derive or dispatch work lands here."** The one-predicate relaxation in
  `generate_reference_dischargeable_operation` is derive work, and it is unavoidable: without it the interpreter
  cannot be *instantiated* for a capture-lifted array-IR program, which is the first of the two recorded blockers.
  It adds no dispatcher, no variant handling, and no new predicate — it removes one.
- **The analysis lost three members at cutover rather than at phase 6.** `region_summaries`, `region_summary`, and
  `is_reference_free` had exactly one consumer each, all of it planner code, so they became dead the moment it was
  deleted and the crate does not build warning-free with them. The transitive summarization they were built from
  survives unchanged as the input to `instruction_summaries`, which is public and still covered. Phase 6 keeps the
  rest of the reduction.
- **Two planner-only tests became one interpreter test and one deletion.**
  `test_discharge_rejects_invalid_program_before_producing_an_artifact` asserted the analysis's `UseAfterConsume`
  variant, which discharge no longer raises; the same program shape is already covered with the interpreter's own
  instruction-level diagnostic by `test_reference_discharge_reports_environment_and_boundary_failures`, so it was
  deleted rather than rewritten into a duplicate. The other twelve planner-only tests — the seven
  `validate_discharge_support` contract rejections with their `MalformedDischargeOperation` fixture, the plan-layout
  test, and the four non-array plan tests — were deleted with the machinery they tested, as §5 anticipated.

**Audit round.** Three independent Opus auditors reviewed the complete phase-3 surface — one for correctness, one for
conventions, and one consolidating both after the first two were slow to return. One of them additionally built probe
programs in an out-of-tree scratch crate and reproduced a real defect. Every finding that changed behavior rather than
prose is recorded here:

- **One real bug, found by probe and fixed: a capture-scoped root that a closure only *passes along* was never
  threaded.** The summary recorded an access only from `reference_semantics` and from nested closures, so a capture
  constant handed to a nested region that ignores it was *lifted* by the replay — which materializes every constant
  something consumes — while no root had been threaded for it. The rebuilt region then failed at `lift_constant` on a
  program the analysis accepts. The summary now follows the replay's own liveness rule: a capture-resolved constant
  that the region consumes records a read. Read-only pruning keeps the artifact minimal, so the root gains a state
  input and no state output. Pinned by
  `test_reference_discharge_threads_a_capture_scoped_root_a_nested_region_only_receives`.
- **`ReferenceRegionDischargeBoundary` no longer trusts the rule for the capture prefix.** Its constructor takes the
  operation and the region index and reads `region_capture_input_count` itself, because the summary derives the same
  fact from the same hook and a rule that disagreed with itself would have made the fork resolve a capture constant to
  a different root than the one the summary threaded — silent for a read-only access. The fork additionally rejects an
  added state input placed inside that prefix, which would renumber the captures the rebound operation still names.
- **The loop rules thread their returned roots too.** `while` and `scan` derived their threaded set from
  `summary.accessed()` alone. A capture root can be an output root without being accessed, and the rules would then
  have left it unthreaded and failed inside the rebuilt body instead of at the fixed-point check that is the real
  violation. All three structured rules now build the same threaded set, and `scan` gained the
  `validate_predicted_output_roots` call `while` and the positional rewrite already made.
- **The reference-free identity fast path was restored** (recorded above with the entry points). Without it the two
  transform adapters that discharge unconditionally — `batched_with_local_references` and
  `rematerialize_with_local_references` — re-traced every reference-free program, losing its dead constants and its
  region transform cache for no benefit.
- **`ArrayReferenceViewOperation` is no longer a super-trait of `ArrayReferenceDischargeOperation`.** The surviving
  consumer classifies operations and never constructs one, so the super-trait only forced a test fixture to implement
  three constructors it does not use. The two contracts are orthogonal and their documentation now says so.
- **Coverage added** for the two widening validators, which §5 asks to be tested directly and which nothing asserted:
  `test_reference_region_discharge_fork_holds_the_replay_to_the_widening_that_sized_it` produces an honest fork and
  holds it to deliberately wrong predictions, which is the shape a lying third-party family presents.
- **Naming and prose.** `under_captures` became `with_captures`, matching `ReferenceCaptureScope::with_roots` and
  `ScanOperation::with_captures`; `ReferenceCaptureScope::index` became `capture_index`, matching
  `CaptureConstant::capture_index`; four comments still describing two implementations or an identity artifact were
  rewritten; the summary's unresolved-reference diagnostic now says "neither through its boundary nor through its
  capture scope" instead of claiming captures cannot be resolved at all; the `ryft-xla` flat-discharge test gained the
  rendered program its name claimed; and `indoc` is imported rather than invoked qualified.

Two findings were deliberately deferred with rationale. `ReferenceDischargeContext::captures`,
`ReferenceCaptureScope::roots`, and `with_roots` are public with no out-of-module consumer today; that is the same
surface-reduction question phase 2 deferred for `ReferenceDischargeEnvironmentId`, and §4's phase-7 deletion audit is
where it belongs. And `discharge_region`, the inlining driver service, still inlines under the caller's scope rather
than deriving a nested one; every in-tree caller inlines a region that inherits, and its rustdoc now states the
restriction rather than silently carrying it.

**Known gaps recorded rather than fixed.** Discharge is now more permissive than the standalone analysis in four
places, each a *missed rejection* that produces a correct program and each verified as such: an ambiguous capture
scope cannot arise because every region is rebuilt per use site under its own path's scope; a capture constant in the
scope region itself resolves to the same root its lifted input already binds; an out-of-range capture index is
rejected with the generic reference-constant diagnostic rather than a precise one; and two capture constants naming
one root are unified into one environment slot instead of being reported as duplicate aliases. The eager lifetime
preflight and the authored-program lint still enforce the stricter contract on their own paths.

**Verification.** `cargo test -p ryft-core --lib`: 1585 passed, 0 failed, 3 ignored. `cargo test -p ryft-core
--tests`: 1585 + 6 + 6 + 4 + 1 passed across the five integration binaries. `cargo test -p ryft-core --doc`: 65 passed
and 16 ignored, plus 5 compile-fail cases. `cargo test -p ryft-xla --lib`: 541 passed, 5 ignored — including the five
pinned StableHLO renderings of production-discharged stateful programs, which did not move. `cargo test -p ryft-pjrt
--lib`: 130 passed. `cargo test -p ryft-macros`: 57 + 1 passed. `cargo test -p ryft-macros-tests`: 21 + 17 passed.
`cargo check --workspace --all-targets`: zero warnings and zero errors. `rustfmt --check` on every file this phase
touched: clean. `cargo doc -p ryft-core --no-deps` and `-p ryft-xla`: no warnings from any file this phase touched.

Phase-3 size, against §6's estimate of ~100/100/40 plus the planner deletion: roughly 700 added lines across
`programs/references/discharge.rs`, `arrays/reference_discharge.rs`, the three control-flow rules, `ryft-macros`, and
`ryft-xla`, against roughly 1,500 deleted. §6 sized this phase as a thin cutover because it did not anticipate the two
recorded prerequisites; the capture-scope increment is most of the addition, and the planner deletion came in larger
than the ~900 lines §6 projected for it.
