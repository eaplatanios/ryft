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
the three canonical view constructors. It was split out of the planner's classification trait so that the interpreter
states only the contract it uses, which let phase 3 delete the planner's structural half without touching the policy;
phase 7 deleted the classification trait itself, so only the constructor contract remains. The `C` bound placement is
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
    /// Ordinary destination value, carrying no reference and replayed as-is.
    Ordinary(C::Value),

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
that keeps the dispatch derives clear of trait-solver recursion); the reference-free replay rule's conversion from
`Self` into `C::Operation` uses the established conversion seam with an explicit `Self::Type`/`C::Type` relationship;
and `ReferenceDischargeTracer` proves out as a real `Value` (capabilities delegate on the ordinary carrier, error on
`Reference`, and the reference arm types as the exact handle type). The capability-only bound has one mechanical
consequence recorded with the phase-0 summary: the rule trait cannot carry a defaulted body, because a default
reference-free replay body would need `C: Context` and `Self: Clone + Into<C::Operation>` on the trait itself.
Reference-free replay is therefore the free function `discharge_reference_free_operation`, which
`impl_reference_free_dischargeable_operation!` delegates to, on the model of
`impl_non_transposable_operation!`.

**Capture scopes (phase-3 contract).** A capture-lifted program names its caller's references through constants
rather than through its boundary, so the interpreter needs one more piece of per-scope context: which root each
capture position binds. That is a property of the *scope* a region discharges under, not of any rule, so it lives
beside the root environment on the discharge context and is recomputed at every region boundary:

```rust
/// Roots the enclosing capture prefix binds, together with the constant-family seam that recognizes a capture.
struct ReferenceCaptureScope<Constant> {
    /// Seam reporting the capture position a constant names, or [`None`] when it is an ordinary constant.
    capture_index: fn(&Constant) -> Option<usize>,

    /// Root each capture position binds, or [`None`] when that position carries an ordinary value.
    roots: Vec<Option<ReferenceRootHandle>>,
}
```

The type is private: phase 7's deletion audit found no out-of-module consumer, as it did for
`ReferenceDischargeEnvironmentId`. The seam is a plain function pointer rather than a bound, because
`CaptureConstant` cannot be required of every constant family the interpreter serves — the non-array prototype universes
deliberately do not implement it — and it is the same higher-order seam
`RegionRef::analyze_references_with_capture_indices` already uses, reduced to a `Copy`, allocation-free carrier. The
default scope resolves nothing, so every existing instantiation keeps today's behavior: a reference-typed constant that
no scope binds is still rejected by `lift_constant`, with its message unchanged.

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
  discharge time, reporting against the environment root it reached rather than a source coordinate, with the
  structured-boundary rejections additionally naming the operation whose rule raised them.
- Deliberately kept ahead of JAX: read-only pruning (JAX returns a final value for every discharged reference) and
  the typed runtime failure semantics of the eager holders. The effects gates, the one shared view traversal, and the
  eager/staged parity oracles are load-bearing here without being deltas against JAX.

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

### 3.1 Outcome: SSA view operations are retained (decided, phase 5)

The prototype was built, measured against the criteria above, and deleted. The decision is option 2: **`reference_index`
and `reference_slice` stay SSA operations**, and the descriptor direction is closed rather than scheduled. The
load-bearing reason is not that descriptors delete little — they delete considerably more than this section
anticipated — but that everything they buy is reachable additively under the representation already in place, while
what they cost is a coupling constraint across every operation family in a type universe.

**Net deletion (larger than expected, and never weighed against additions).** Adopting descriptors would make the
whole alias category dead, not only the analysis half: roughly 207–218 production lines in
`programs/references/analysis.rs` (`ReferenceAlias` and its accessors, the `aliases` table, `ReferenceHandle::is_view`,
the `Alias` arm of output classification, the two region-boundary view rejections, and four error variants), the
complete 190-line production body of `arrays/reference_analysis.rs` with the roughly 468 lines of its tests that
are view-specific (the file's remaining ~1,565 test lines exercise the generic pass and would be re-homed rather
than deleted), roughly 369
lines of view operations and rules in `arrays/operations/references.rs`, and — the part this section did not
anticipate — about 130–150 lines in `programs/references/discharge.rs` (the `Alias` associated type and `root_alias`,
the handle's `alias` field, `ReferenceDischargeContext::derive` in full, and four separate "derived view" rejection
families), plus `ReferenceAliasKind` and the `Alias` variant of `ReferenceOutputSemantics` in `semantics.rs`, whose
only production declarer in the workspace is the shared static the two view operations share, and the three
production `ReferenceDischargeRule::Alias` arms with `ArrayReferenceOperation::reference_view_transform` and its
`XlaOperation` implementation. (Phase 7 deleted `ReferenceDischargeRule` outright, so those three arms are already
gone and the total is now an upper bound; `reference_view_transform` and `ReferenceAliasKind` survive and are still
consumed by the array analysis overlay.) That totals roughly 950 production lines. What survives either way is the
duplicate-alias family (four diagnostics, ~140 lines) — those detect two live handles reaching one root and read no
view metadata at all — together with about 1,554 lines of root identity, capture scopes, liveness, access summaries,
and non-view boundary rules. The criterion, however, is *net* deletion, and the additions were never measured: the
prototype's own 505 non-test lines cover a deliberately compressed surface (one payload with an access-kind
discriminant instead of five, and only the discharged arm of each rule), so a faithful implementation is somewhere in
the 400–900 range. Net deletion is therefore plausibly modest and possibly a wash. That is the honest reading, and it
is the one recorded.

**Discharge gains nothing.** This follows from §2.2 rather than from measurement: the phase-1 flowing alias already
made view resolution discharge-internal and policy-owned, so where the chain is written down cannot change what the
transform produces. The prototype's identical-program result is a consistency check on that claim, not evidence for
it — its policy delegates to `ReferenceDischargePolicy::read`/`replace` with an `ArrayReferenceView` prefix, so
agreement is guaranteed by construction.

**IR size and rendering legibility split.** On size the descriptor form is slightly better in the source program (N
access instructions against a chain of length *k* plus N), and the SSA form's sharing is nominal in the artifact,
because discharge re-materializes the chain at every access under both representations — pinned today by
`test_array_reference_discharge_policy_stages_composed_view_accesses`. On legibility the descriptor form is
measurably worse: the prototype renders `viewed_reference_read [view=[Index { axis: 0, index: 1 }]]`, a debug-formatted
transform list stuffed into an attribute, against named `reference_index`/`reference_slice` instructions whose derived
reference types the printer already shows.

**The traced-handle model is a trade, not a cost.** The first reading of this criterion overstated it and is
withdrawn. There is no single value-level capability surface to split: `arrays/operations/references.rs` already
carries three impl families over seven traits, `Output` is a defaulted generic parameter rather than an associated
type, and seven of the nineteen implementations already have `Output != Self`. Nor does the parity oracle rest on the
view capabilities — `test_array_ir_reference_program_matches_eager_execution` never calls them, and eager/staged view
agreement rests on the single shared `ArrayReferenceView` traversal, which the descriptor design keeps. The
capture-rejection reasoning was also wrong: `ArrayIrValue::validate_as_constant` rejects every reference value, root
and derived alike, and a capture is the sanctioned escape hatch that stages as a `CaptureReference` and bypasses that
check entirely. What genuinely remains is ergonomic: today the eager and tracer implementations agree that
`Output = Self`, and under descriptors the tracer implementation could not, so the seventeen call sites (all in one
file) would need annotation. And the clone-shared invalidation the criterion asks about is what the eager handles
already are — `ArrayReference` is a cloneable host wrapper over an `Arc`-shared holder with generations, freeze, and
alias-family invalidation — so descriptors would make the staged side pay dynamically what the eager side already
pays, rather than adding an unpaid obligation.

**Dynamic indexing is cheaper under descriptors but is not exclusive to them.** The prototype carries a dynamic index
as an operand of the access with no view metadata at all. The SSA route reaches the same capability additively,
because `ReferenceDischargePolicy` is parameterized by the destination `C`, so `type Alias` may mention `C::Value`:
`Parameter` is an empty marker, `Value` already implies it, every `P::Alias: PartialEq` site is already paired with
`C::Value: PartialEq`, the handle already carries a `C::Value` in its `preserved` field, and the recursive driver pins
only `Referent` across its two policy instantiations, so a caller alias and a fork alias may differ freely. Two
caveats are recorded rather than resolved, because neither is on this decision's critical path. The projection
equality a view rule would then state (`Alias = ArrayReferenceView<C::Value>`) sits inside the derive-generated
dispatch obligation graph, which is where this repository has met `E0275` before, so the follow-up that implements
dynamic indexing must confirm it before relying on it. And `ArrayReferenceView` is not only the policy's alias — it is
also the eager handle's view and the analysis's resolved view — so a value-carrying view needs a value-free variant
for the analysis and the preserved-reference kernel path, which have no values to resolve against.

**Backend extensibility is the decisive argument against descriptors.** `XlaOperation` flattens the core reference
variants and relates them to the core family through total bidirectional `From` conversions. Making the descriptor
part of an access operation's *type* forces every family in one type universe — core array IR, the XLA superset, and
any third-party superset — onto one descriptor algebra, or makes those conversions lossy. Under SSA views a backend
adds a view operation to its own family plus a discharge rule, and the core access operations are untouched. (The
prototype's other apparent cost here, a wider operation-family seam for the dynamic slicing operations, is not a
differentiator: that seam grows with dynamic indexing under either representation.)

**Two costs the prototype did not measure, both of which support retention.** It skipped the preserved arm of each
rule on the premise that replay is identical under both representations, and that premise is false: under SSA a
preserved viewed access replays a real `reference_index` into the destination and stores it as the handle's exact
preserved value, while under descriptors there is no view instruction and the backend must interpret the descriptor,
which restructures `ryft-xla`'s preserved-kernel eligibility contract and its lowering records. Partial discharge and
the preserved-kernel consumer are precisely the survivors §2.3 names, and neither was exercised. Second, this section
says migration happens "after Track A is stable", and phases 6 and 7 are still open; deciding now is nonetheless
right, because §4's phase-6 reduction scope is explicitly conditional on this decision, so leaving it open would
block the reduction rather than inform it.

**What the follow-up would be, if the question is ever reopened.** Not a representation migration but a targeted,
additive one: a dynamic view operation whose discharge rule materializes rather than composes, sized from the two
caveats recorded above. The one consumer that would genuinely benefit from descriptors —
`PreservedReferenceKernelOperation::validate_body`, the sole caller of `ArrayReferenceAnalysis::view`, which embeds
the resolved view into its lowering records — can be served instead by reading the view off the analysis it already
runs.

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
- [x] **Phase 4 — flat partial discharge.** Selection parameter, `Preserved` threading, mixed-result typing, and
      the `PartialReferenceDischargeResult` envelope with `try_into_full`; preserved roots at structured boundaries
      are rejected with an exact diagnostic; adapters stay on the full-discharge contract; kernel-pipeline shaped
      tests (discharge pipeline state, keep kernel references) and preserved-kernel integration as a consumer.
- [x] **Phase 4b — structured partial discharge.** Mixed structured carries: preserved reference-typed carries
      crossing condition/while/scan/call boundaries beside discharged state, with the boundary-rejection diagnostic
      from phase 4 lifted and the structured-rule and summary machinery extended accordingly. This phase owns the
      capability the plan advertises; partial discharge is not complete until it lands.
- [x] **Phase 5 — view-representation prototype and decision (Track B).** Build the generic-descriptor prototype,
      run the Section 3 comparison, record the decision in this plan, and only then migrate or close the question.
      Outcome: SSA views are retained (Section 3.1); the prototype is deleted and nothing is migrated.
- [x] **Phase 6 — trace-time prevention and lint reduction.** Traced-handle scoping and freeze
      generation/invalidation (shared state across clones; capture rejection); shrink the standalone analysis to
      what is actually consumed. The reduction scope is settled by Section 3.1: with SSA views retained, the alias
      and view machinery stays in the IR and in the analysis, because the preserved-reference kernel path consumes
      the resolved view; the reduction is therefore confined to members with no consumer. With the flowing
      reference value carrying the complete alias chain (Section 2.1), discharge needs no analysis-provided view
      resolution regardless. The surviving consumers (lint, eager preflight, summary service, site enumeration) are
      named in the phase summary.
- [x] **Phase 7 — stabilization.** Deletion-audit questions parked from earlier phases plus one owner-raised item,
      now answered: `ReferenceDischargeRule` and `ArrayReferenceDischargeOperation` are **deleted**, and
      `PreservedReferenceKernelOperation::validate_body` derives its eligibility classification directly from
      `Operation::reference_semantics` (the `Operation`-level declarative hooks themselves were already settled:
      they are the fact layer that the analysis, the discharge summaries, the kernel validator, and XLA lowering all
      consume, and the rule trait was the second declaration they made redundant). Documentation (JAX correspondence,
      prevention ladder, experimental markers), doctests, `ryft-macros` and `ryft-macros-tests` verification (the derive
      surface changes), full suites across `ryft-core`/`ryft-xla`/`ryft-pjrt`, deletion audit, and independent audit
      rounds to convergence per the house protocol.

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
all changelog files: excluded; no behavioral changes. Phase 7 revises the holder runtime's documentation only.

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
~3,200 production, ~3,050 test, and ~900 documentation lines written, against roughly 1,700 production lines deleted
(the planner and bespoke replay in phase 3, which came in at ~1,500 rather than the ~900 projected here; the analysis
reduction in phase 6; and the discharge-rule classification in phase 7, which is a net-deletion phase rather than the
~0/100/300 its entry below projects). The retained baseline excluding the
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
   context-free inner carrier is `ReferenceDischargeValue::{Ordinary, Reference}` (phase 7 renamed the first variant
   from `Pure`, which the repository now reserves for effect purity).
2. **The discharge context implements `Context` (decided, phase 0 validates).** Discharge is a single
   program-to-program interpretation, so it runs through `interpret_in_context` like batching and differentiation
   rather than through a bespoke replay like partial evaluation (whose wrapper shape is justified only because
   partitioning is not a single interpretation). The context's value is the context-stamped
   `ReferenceDischargeTracer` on the `BatchingTracer`/`DifferentiationTracer` precedent — not an eager-style
   contextless enum, because discharge values must dispatch through the live context that owns the root
   environment, and not the generic `Tracer<C>`, because a discharged handle has no destination atom to wrap.
   Capabilities delegate on the ordinary carrier and error on `Reference`, and the reference arm types as the exact
   handle type.
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
1585 + 6 + 6 + 4 + 1 passed across the lib target and the four integration binaries (the downstream-surface binary now also drives two
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
  or `::Root`, then named `::State`) separately from the added state inputs and outputs and their insertion positions — separately because
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
deletion audit is where surface reduction belongs rather than a phase boundary. (Phase 7 privatized it.) And the three
structured rules were left stating their threaded-root set slightly differently — `while` and `scan` derive it from
accesses alone, where the positional rewrite adds the declared output roots — because for a loop the identity hook makes
the declared output roots a subset of the carries, so the sets are provably equal; the second auditor examined this and
reached the same conclusion independently.

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
--tests`: 1595 + 6 + 6 + 4 + 1 passed across the lib target and the four integration binaries. `cargo test -p ryft-core --doc`: 65 passed
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
  *decides about* a reference program without rewriting it. (Superseded by phase 7: both were deleted, and that
  validator now derives the same classification from `Operation::reference_semantics`.)
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
where it belongs. (Phase 7 made the whole capture scope private.) And `discharge_region`, the inlining driver service,
still inlines under the caller's scope rather than deriving a nested one; every in-tree caller inlines a region that
inherits, and its rustdoc now states the restriction rather than silently carrying it.

**Known gaps recorded rather than fixed.** Discharge is now more permissive than the standalone analysis in four
places, each a *missed rejection* that produces a correct program and each verified as such: an ambiguous capture
scope cannot arise because every region is rebuilt per use site under its own path's scope; a capture constant in the
scope region itself resolves to the same root its lifted input already binds; an out-of-range capture index is
rejected with the generic reference-constant diagnostic rather than a precise one; and two capture constants naming
one root are unified into one environment slot instead of being reported as duplicate aliases. The eager lifetime
preflight and the authored-program lint still enforce the stricter contract on their own paths.

**Verification.** `cargo test -p ryft-core --lib`: 1585 passed, 0 failed, 3 ignored. `cargo test -p ryft-core
--tests`: 1585 + 6 + 6 + 4 + 1 passed across the lib target and the four integration binaries. `cargo test -p ryft-core --doc`: 65 passed
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

### Phase 4 — flat partial discharge (landed)

**What landed.** Partial discharge is the general rewrite and full discharge is its everything-selected case; the two
share one body. A caller names [`ReferenceDischargeSite`]s, every root the selection omits survives in the destination
as an ordinary reference, and every access to a surviving root replays verbatim.

- **The selection knob.** A private `ReferenceDischargeSelection` — either "everything" or a validated site set behind
  an `Rc<BTreeSet<..>>` — rides on `ReferenceDischargeContext` beside the root environment and the capture scope, and
  is shared unchanged by every clone and by every region fork. Selecting everything is deliberately a state of its own
  rather than a set listing every site: a program's sites are enumerated from its own arena while a selection is
  caller-supplied, and an allocation no site *can* name (one bound directly rather than replayed) must still be
  discharged. Rules ask exactly one question, `ReferenceDischargeContext::selects_allocation(instruction,
  output_index)`; whether an entry-boundary root was selected is decided once, by the program-level boundary threading,
  through the private `selects_external`.
- **`Program::partially_discharge_references_with_policy`**, the new program-level entry point, validates its selection
  against the program through phase 0's `validate_reference_discharge_sites` before anything is replayed and returns
  `PartialReferenceDischargeResult`. `discharge_references_with_capture_seam` — the shared body — now always assembles
  that envelope, and the two full entry points convert through `try_into_full`. The arrays universe exposes
  `Program::partially_discharge_references`.
- **`Preserved` threading.** An unselected external root keeps its reference-typed boundary position exactly as the
  source declared it and contributes no `ReferenceStateBinding`; an unselected allocation site is replayed and the
  root it binds is the destination reference that replay produced. The five generic primitive rules and the two array
  view rules each branch on `ReferenceDischargeReference::preserved()`: an access replays through the new shared
  `discharge_preserved_access`, a `freeze` additionally unbinds the root through the new
  `ReferenceDischargeContext::unbind_preserved`, and a view replays its own operation and stamps the bound output as
  the derived handle's exact preserved value, so later accesses consume that value instead of re-deriving the chain.
  Program outputs may now be preserved references.
- **The structured-boundary rejection** lives at the two seams every path funnels through:
  `ReferenceDischargeContext::operand_root`, which every structured rule already calls for its operands, and the region
  fork's `thread` closure, which covers capture-scoped entering roots. The fork additionally names a preserved
  region-local root at its publication check rather than reporting it as an unthreaded one.
- **Consumer integration.** `PreservedReferenceKernelOperation::validate_body` in `ryft-xla` is now exercised as a real
  consumer of the new entry point: a kernel body partially discharged with every parameter preserved still validates,
  with the same bindings and the same mock lowering it derives from the source, and discharging one parameter instead
  makes the validator reject the mixed boundary by name. That pair *is* the contract between the two vocabularies —
  the selection a pipeline hands to discharge is the complement of its kernels' parameters.

**Decisions this phase forced.**

1. **The primitive rules' destination bound moved from `C: Domain` to `C: Context<Operation: From<Self>>`.** Replaying
   an access verbatim requires the conversion seam into the destination's operation family, and no policy method or
   context service can supply it, because the operation is the rule's own. The derive needed no change: its dispatcher
   already pins `__ParentContext: Context<Type = T, Operation = Self>`, and a composite-native payload's `From` impl is
   generated beside it. Two in-crate test fixtures had to become real operation families to satisfy it, which is what a
   real universe already is.
2. **A preserved root's liveness is resolved against the environment at every use, including a program output.** The
   handle carries the destination value, but only the environment knows whether the root is still live, so
   `discharge_preserved_access` and the entry point's output mapping both consult `root_state` first. Without that, a
   program that froze a preserved root and then returned it published a stale reference.
3. **Consuming a preserved external root is accepted, where full discharge rejects it.** Full discharge rejects because
   a `ReferenceStateBinding` cannot describe a holder that no longer exists. A preserved root has no binding to
   describe: the payload keeps the consuming operation, and the caller hands its holder to that operation directly. The
   difference is pinned by a test that runs both entry points over the same program.
4. **The selection propagates into region forks.** A site names a source coordinate that means the same thing wherever
   the replay reaches it, so an unselected allocation inside a rebuilt region survives there exactly as it would have
   in the caller's own body. Coordinate agreement between `Program::reference_discharge_sites` and the replay is
   structural: both mint `InstructionId::new(region_id, index_in_region)` over the same unskipped region walk.
5. **The kernel-path integration stops at the body, and that boundary is recorded rather than papered over.**
   `PreservedReferenceKernelOperation` is not an `XlaOperation` variant, so no program containing a kernel is ever
   discharged; there is no outer pipeline program in the tree to partially discharge *around* a kernel. What exists
   today is the body, and the honest integration is the one that landed: the validator consuming a partially
   discharged body, plus the pinned rejection that states why a kernel parameter may not be selected. Embedding the
   kernel operation in a program would additionally require a discharge rule for a region-carrying operation whose
   closure keeps references, which is phase 4b's capability.

**Owner directives applied mid-phase.** Two arrived while the phase was in flight and are recorded here as such.

- **Renames.** `discharge_pure_operation` became `discharge_reference_free_operation` and
  `impl_pure_reference_discharge!` became `impl_reference_free_dischargeable_operation!`, across every call site, doc
  comment, and this plan's prose. "Pure" collides with the effects vocabulary (`Effects::is_pure`) while the path
  serves reference-free operations regardless of their effects, so both rustdoc blocks now state the actual
  precondition: an operation with ordered effects replays here unchanged, because replaying it reproduces those
  effects in the destination exactly as the source performed them.
- **Ordering.** The macro declaration moved before `impl_differentiable_operation!` in `macros.rs`; every operation's
  `ReferenceDischargeableOperation` implementation (or macro invocation) moved before that operation's batching,
  differentiation, and transposition implementations in its own module — eighteen modules, with each impl's anchor
  comment moving with it; and the dispatch derive now emits discharge before batching, differentiation, and
  transposition. All of it is pure code motion, verified by a green matrix. Two files the automated sweep could not
  classify were handled by hand: `linear_call.rs`, whose batching impl header wraps onto a `>` continuation line, and
  `compare.rs`, whose three macro invocations sit in one adjacent group.

**Deviations from the plan, with rationale.**

- **No partial sibling for the capture-lifted entry point.** §4's phase 4 says nothing about captures, and no consumer
  needs one; the shared body already takes the seam as a parameter, so adding it later is a ten-line public wrapper.
  Both partial entry points now say so, and the fork's preserved-root rejection carries a comment recording that it is
  unreachable until that sibling exists — every other path into it passes through `operand_root` first.
- **Preserved-rule coverage is split across three universes rather than colocated in one.** A preserved root needs a
  destination value that *is* a reference. The array universe's eager values can be (so its view rules are covered
  eagerly, against the eager reference oracle), the in-crate list universe's and the downstream register universe's
  cannot, so their preserved coverage runs against staging destinations and pins the recorded programs. The generic
  primitive rules in `programs/references/operations.rs` are covered colocated, against a staging destination its
  fixture gained for the purpose.
- **`reference_discharge_sites` still enumerates sites inside closures the replay only copies.** An allocation inside a
  dormant derivative rule region is enumerated and validates cleanly, but discharge rejects such a program whichever
  way the site is selected, because threading state through a `custom_jvp`, `custom_vjp`, `linear_call`,
  `rematerialize`, or `shard_map` closure has no defined meaning (§2.2). Letting the verbatim copy through when nothing
  in the closure was selected was considered and rejected: rematerializing a region that mutates a *live* reference is
  unsound for the same reason it is unsound for a discharged one. The enumeration documents the inert class instead of
  second-guessing region roles.

**Audit round.** Two independent Opus auditors reviewed the complete phase-4 surface, one for correctness and one for
conventions; the correctness auditor additionally drove the real `XlaOperation` family through the new entry point from
an out-of-tree probe crate. Every finding was addressed. The ones that changed behavior rather than prose:

- **One real defect: the output boundary was the single use of a preserved root that skipped the liveness check.** A
  program that froze a preserved root and then returned it produced a payload publishing a stale reference, where full
  discharge rejects the same shape. Fixed at the entry point's output mapping and pinned by
  `test_partial_reference_discharge_lets_a_program_consume_a_preserved_external_root`, which also pins the accepted
  half of decision 3.
- **A preserved region-local allocation escaping a fork was reported as an unthreaded root.** The fork now names it as
  preserved, which is the phase-4 diagnostic the plan asks for.
- **`discharge_preserved_access` no longer `unwrap()`s the handle invariant**, so a handle that reached it some other
  way is reported rather than panicking inside a fallible rewrite.
- **Two stale rustdoc claims from before phase 2** said the reference-free replay rejects *any* region-carrying
  application; it copies reference-free regions across and rejects only a closure that reaches a reference. Both
  copies (the macro and the derive) now say what the code does.
- **Prose, naming, and coverage.** `ReferenceDischargeSelection::of` became `from_sites` per the `from_*` convention;
  the omnibus preserved-access test split into a replay test that pins the recorded program and a separate
  `unbind_preserved` test; `selects_allocation`'s doc now explains why it is the only selection question a rule can
  ask; the fixture's destination aliases were regrouped; one over-long rustdoc link line and three trailing-comma
  violations were fixed.

**Known gaps recorded rather than fixed.** A preserved root cannot cross a structured boundary, which is phase 4b's
scope and is rejected by name at both seams. Partial discharge is not offered on the capture-lifted entry point. The
transform adapters stay on the full-discharge contract, as §4 requires. The `ryft-xla` kernel path consumes partial
discharge at the body level only, for the reason decision 5 records.

**Verification.** `cargo test -p ryft-core --lib`: 1596 passed, 0 failed, 3 ignored. `cargo test -p ryft-core --tests`:
1596 + 6 + 6 + 5 + 1 passed across the lib target and the four integration binaries, including the two `trybuild` compile-fail cases.
`cargo test -p ryft-core --doc`: 65 passed and 16 ignored, plus 5 compile-fail cases. `cargo test -p ryft-xla --lib`:
542 passed, 5 ignored. `cargo test -p ryft-pjrt --lib`: 130 passed. `cargo test -p ryft-macros`: 57 + 1 passed.
`cargo test -p ryft-macros-tests`: 21 + 17 passed. `cargo check --workspace --all-targets`: zero warnings and zero
errors. `cargo fmt --check` scoped to `ryft-core`, `ryft-macros`, `ryft-macros-tests`, and `ryft-xla`: clean.
`cargo doc -p ryft-core --no-deps` and `-p ryft-xla --no-deps`: no warnings from any file this phase touched.

Phase-4 size, against §6's estimate of ~250/300/60 plus ~100/150/30: roughly 720 added lines in
`programs/references/discharge.rs` and 315 in `programs/references/operations.rs`, about 130 across the arrays
modules, the downstream integration test, and `ryft-xla`, split between production, tests, and documentation in about
the proportion §6 projected. The owner-directed rename and reordering touched twenty further files without changing
their line counts materially. The one place the estimate was low is the test fixtures: making two of them real
operation families, which decision 1 forced, cost about 150 lines that §6 did not anticipate.

### Phase 4b — structured partial discharge (landed)

**What landed.** A preserved root now crosses a condition, loop, scan, or positional call boundary as the reference it
already is, at its own declared operand position, exactly as the source passed it. The phase-4 rejection is gone, and
the capability the plan advertises is complete: partial discharge threads discharged state and surviving references
side by side through every structured shape, and only the discharged half widens.

The insight that made this small is that a preserved carry needs *no* widening. Discharge's structured rewrites exist
to turn mutation into threaded state; a root that survives as a reference has no state to thread, so it simply occupies
the operand position the source already gave it. Everything the four rules needed was therefore one distinction, drawn
once:

- **`ReferenceDischargeContext::threaded_state_roots`** replaced `validate_threaded_roots`, which is deleted. It
  returns the roots a closure needs threaded *as state* — the union of the summary's accessed roots and its declared
  output roots, with preserved roots removed — and validates liveness on the way, propagating the environment's own
  reason rather than restating a cause it did not check. All four structured rewrites build their threaded set through
  it, so the three that previously duplicated the union now state one thing.
- **`ReferenceDischargeContext::operand_root`** no longer rejects a preserved root; it still rejects a derived view,
  which no boundary of either kind can carry.
- **`operand_state` became `operand_value`**, because what an operand contributes is now a discharged root's state, a
  preserved root's own reference, or an ordinary value. It resolves the kind from the environment rather than from the
  handle.
- **`ReferenceRegionDischargeInput::State` became `::Root`**, and the fork derives from the caller's environment
  whether that position carries entering state or a surviving reference. A preserved position binds through
  `fork.bind_preserved` over an input of the *reference* type; the caller-to-fork table relates preserved roots exactly
  as it relates discharged ones, so a region returning one still reports the caller root it denotes; a preserved region
  output publishes the handle's own reference; and `mutated_roots` counts only discharged roots, because a preserved
  root's writes replayed as the operations the source performed.
- **The rules merge state back only for threaded roots.** A preserved carry comes back out as the same reference and
  has nothing to merge, which is the one line each of the positional rewrite, `while`, and `scan` needed.

**Decisions this phase forced.**

1. **A preserved root's kind is read off the environment, never restated by the boundary.** The first cut gave the
   fork's `thread` closure a `preserved: bool` and cross-checked it. That is the shape phase 3's audit already rejected
   for the capture prefix — the boundary derives the fact rather than trusting the rule — so the flag went away, the
   two declared-position variants collapsed into `Root`, and a rule that synthesizes a preserved root onto a rebuilt
   boundary's *added* state is rejected by one explicit guard instead. Added state is state: a surviving reference
   reaches a region only where the source already passed it.
2. **The caller keeps its own handle across the boundary.** A structured operation whose region returns a preserved
   root produces a reference-typed output denoting that same root, and the rule reports the operand carrier rather than
   that output, exactly as it does for a discharged root. The two references are the same reference, so the choice is
   semantically free; keeping the caller's handle avoids a second destination value for one root and leaves the
   operation's own reference output unused, which is what the `while` and `scan` renderings show. The dead position is
   also forced — the declared output boundary belongs to the operation, and a loop's symmetry requires it — and a later
   full discharge of the same payload collapses it into an ordinary state carry, which an auditor verified end to end.
3. **Consuming a caller root inside a region stays rejected for preserved roots too.** Whether a region consumed a root
   can depend on which branch ran, and the caller's environment cannot represent "maybe consumed". The summary's
   diagnostic changed from "cannot thread ... through a region" to "cannot pass ... into a region", which is true of
   both kinds.
4. **A region-local preserved allocation still may not escape.** It is reported by the fork's existing publication
   check as a root the caller did not thread, which is the same answer discharged region-local allocations get.

**Deviations from the plan, with rationale.** None of substance. §4's phase 4b asks for the boundary rejection lifted
and the structured-rule and summary machinery extended; the summary itself needed no extension, because the
discharged/preserved distinction belongs to the environment rather than to the static closure analysis, and drawing it
in `threaded_state_roots` is what kept the four rules unchanged in shape.

**Audit round.** Two independent Opus auditors reviewed the complete phase-4b surface, one for correctness and one for
conventions. Decision 1 above was made mid-audit and answered two of the conventions auditor's questions before they
were reported. Every other finding was addressed; the ones that changed behavior rather than prose:

- **`threaded_state_roots` no longer swallows the environment's error.** It reported "accesses {root} without entering
  state" for a consumed root as well, asserting a cause it had not established.
- **`operand_value` no longer dispatches on the handle**, matching the seam every other consumption path now uses, and
  its liveness check has a visible consumer instead of being a discarded probe.
- **Three documentation statements that the increment had made false** — the entry point's `# Errors` clause still
  promising the phase-4 structured rejection, a "reported by name" promise anchored on an unreachable capture-scoped
  path, and the fork's inherited-capture comment, whose "a scope position the closure reaches is therefore threaded"
  premise no longer holds for a preserved root and would become a real defect the day a partial form of the
  capture-lifted entry point lands — were corrected, and the `while` and `scan` rule-header contracts gained the
  preserved-carry sentence they were missing.
- **One correction to the phase-4 record above.** It says the fork "now names it as preserved" when a region-local
  preserved allocation escapes. Phase 4b removed that arm: preservation is no longer *the* reason such a root cannot be
  published, so the fork reports the accurate generic reason again — the caller never threaded it (decision 4).
- **Coverage added for the two interactions the first pass left untested**: a region that *writes* a preserved root
  while only reading the discharged one beside it, which is where read-only pruning and preservation meet, and a
  preserved root crossing two nested boundaries, where it is bound as a preserved root of the outer fork and threaded
  again into the inner one.

**Known gaps recorded rather than fixed.** Partial discharge is still not offered on the capture-lifted entry point, so
a preserved root can only reach a region through an operand the source already passed. A region may not consume a
caller root of either kind, and a region-local preserved allocation may not escape (decisions 3 and 4). The eager
interpreter cannot carry a reference through a `while` at all — its masked predicate selection needs value semantics —
so a mixed loop is exactly as runnable as the source it came from and no more; the test pins that both report the same
limitation.

**Verification.** `cargo test -p ryft-core --lib`: 1602 passed, 0 failed, 3 ignored. `cargo test -p ryft-core --tests`:
1602 + 6 + 6 + 5 + 1 passed across the lib target and the four integration binaries. `cargo test -p ryft-core --doc`: 65 passed and 16
ignored, plus 5 compile-fail cases. `cargo test -p ryft-xla --lib`: 542 passed, 5 ignored. `cargo test -p ryft-pjrt
--lib`: 130 passed. `cargo test -p ryft-macros`: 57 + 1 passed. `cargo test -p ryft-macros-tests`: 21 + 17 passed.
`cargo check --workspace --all-targets`: zero warnings and zero errors. `cargo fmt --check` scoped to the four touched
crates: clean.

Phase-4b size, against §6's estimate of ~250/300/40: roughly 120 production lines across
`programs/references/discharge.rs`, `while.rs`, and `scan.rs` — a *net* reduction in the two loop rules, because
`threaded_state_roots` absorbed a block each — plus about 400 test lines and 60 of documentation. The estimate was high
because it assumed the summary machinery would have to grow; it did not.

### Phase 5 — view-representation prototype and decision (landed)

**What landed.** The Track B decision and its evidence, recorded in §3.1. Nothing was migrated, and the prototype that
produced the evidence was deleted at the decision, as §6 requires.

The prototype spanned the `programs -> arrays` boundary the phase exists to evaluate, in two `cfg(test)` modules:
`programs/references/view_descriptor_prototype.rs` (the universe-generic `ReferenceViewDescriptor<T>` contract, the
`ReferenceViewDescriptorPolicy<C, X>` replacement for the policy's alias mechanics, and one descriptor-carrying access
payload with its `Operation` and `ReferenceDischargeableOperation` implementations) and
`arrays/reference_view_descriptor_prototype.rs` (the `ArrayViewDescriptor` algebra — a static `ArrayReferenceView`
prefix plus an optional dynamically supplied index — its descriptor implementation, the policy implementation on
`ArrayReferenceDischarge`, and three tests). It measured three things: that both representations stage the identical
destination program for a static chain read twice, that a dynamic index works as an operand of the access with no view
metadata at all, and what a descriptor access's type inference and rendering look like. It came in at 505 non-test and
211 test lines, against §6's hard budget of ~500/200.

**The decision.** Retain SSA view operations. §3.1 records the outcome and the evidence criterion by criterion; the
short form is that descriptors would delete roughly 950 production lines of alias machinery against unmeasured
additions in the 400–900 range, buy a dynamic-indexing capability that is reachable additively under the current
representation, and cost a constraint that forces every operation family in one type universe onto a single descriptor
algebra. "Keep SSA views" was an explicitly acceptable conclusion from the outset, and it is the one the evidence
supports.

**Deviations from the plan, with rationale.**

- **The prototype is `cfg(test)`-only rather than production code that is later removed.** §6 sizes it as ~500/200
  production/test, which reads as production surface. Compiling it only for tests makes the deletion criterion
  mechanical (two module declarations and two files), keeps the shipped surface free of a design that was not adopted,
  and costs nothing in fidelity: the generic half still lives in `programs`, the array specialization still lives in
  `arrays`, and the compiler still enforced the dependency direction.
- **The access surface was compressed.** One payload with an access-kind discriminant stands in for one
  descriptor-parameterized payload per access mode, and only the discharged arm of each rule was implemented. Both
  compressions are recorded in the prototype's own module documentation and in §3.1, and the second one turned out to
  matter: the preserved arm is where the two representations differ most, so the comparison's largest unmeasured cost
  is named rather than papered over.
- **The dynamic index was narrowed** to a rank-one static selection consuming one rank-zero integer operand.
  Generalizing it needs one zero index per remaining axis, which is destination-value construction rather than a
  representation question.

**Audit round.** One independent Opus auditor reviewed the prototype and every intended finding, with a mandate to
challenge rather than confirm. It confirmed the conclusion and rejected two of the seven findings in their stated form;
both corrections are folded into §3.1 and are worth recording here because they changed the argument rather than its
prose:

- **The "decisive cost" was withdrawn.** The claim that descriptors would split one value-level capability surface was
  wrong on three counts: `arrays/operations/references.rs` already carries three impl families over seven traits,
  `Output` is a defaulted generic parameter rather than an associated type (seven of nineteen implementations already
  have `Output != Self`), and the eager/staged parity oracle never calls the view capabilities at all — parity rests on
  the shared `ArrayReferenceView` traversal, which descriptors keep. The companion capture-rejection argument was also
  wrong: `ArrayIrValue::validate_as_constant` rejects root and derived references alike, and a capture stages as a
  `CaptureReference` that bypasses the check entirely. What survives is an ergonomic cost at seventeen call sites in
  one file.
- **The deletion ledger was one-sided and undercounted.** The first count reached ~420 lines by looking only at the two
  analysis modules. The audit added `programs/references/discharge.rs` (the `Alias` associated type and `root_alias`,
  the handle's `alias` field, `ReferenceDischargeContext::derive` in full, and four derived-view rejection families),
  `semantics.rs` (`ReferenceAliasKind` and the `Alias` variant of `ReferenceOutputSemantics`, whose only production
  declarer is the static the two view operations share), and the cross-file registration surface
  (`ArrayReferenceOperation::reference_view_transform` with its `XlaOperation` implementation, and the three production
  `ReferenceDischargeRule::Alias` arms), reaching ~950. It also observed that the criterion is *net* deletion and that
  no additions column existed. Both corrections cut against the conclusion and are recorded as such.
- **The value-carrying-alias claim was verified** at the type level: `Parameter` is an empty marker, `Value` implies
  it, every `P::Alias: PartialEq` obligation is already paired with `C::Value: PartialEq`, the handle already carries a
  `C::Value`, and the recursive driver pins only `Referent` across its two policy instantiations. Two residuals are
  recorded in §3.1 rather than resolved, because the decision does not rest on them: the projection equality a view
  rule would state sits inside the derive-generated dispatch graph, and `ArrayReferenceView` is also the eager handle's
  and the analysis's view, so a value-carrying form needs a value-free variant for the analysis and the kernel path.
- **One argument was dropped as a non-differentiator.** The prototype needed a wider operation-family seam for the
  dynamic slicing operations, which reads as a descriptor cost but is not: that seam grows with dynamic indexing under
  either representation. The real extensibility argument is `XlaOperation`'s flattened reference variants and their
  total bidirectional `From` conversions.
- **Two §3 criteria had been skipped and were added:** rendering legibility, which the prototype measured and which is
  unflattering to descriptors, and §3's own "after Track A is stable" condition, which is honestly answered by §4's
  phase-6 scope being conditional on this decision.

**Known gaps recorded rather than fixed.** The comparison never exercised partial discharge or the preserved-reference
kernel path under descriptors, which §3.1 names as its largest unmeasured cost. The additions column is an estimate
rather than a measurement, and §3.1 says so. Both are acceptable because the conclusion is retention: an adopting
decision would have required measuring them.

**Verification.** With the prototype in the tree: `cargo test -p ryft-core --lib`: 1605 passed, 0 failed, 3 ignored.
`cargo test -p ryft-core --tests`: 1605 + 6 + 6 + 5 + 1 passed across the lib target and the four integration binaries.
`cargo test -p ryft-core --doc`: 65 passed and 16 ignored, plus 5 compile-fail cases. `cargo test -p ryft-xla --lib`:
542 passed, 5 ignored. `cargo test -p ryft-pjrt --lib`: 130 passed. `cargo test -p ryft-macros`: 57 + 1 passed. `cargo
test -p ryft-macros-tests`: 21 + 17 passed. `rustfmt --check` on both prototype files: clean. After the deletion the
same matrix returns to its phase-4b numbers, which is the check that the prototype left nothing behind.

### Phase 6 — trace-time prevention and lint reduction (landed)

**What landed.** A traced program now fails at the call that misuses a reference, rather than at discharge; `freeze`
consumes its handle; and the standalone analysis lost the members that survived phase 3's cutover without a consumer.

- **The trace-time rung.** `ReferenceLiveness` — renamed `ReferenceLifetimes` at phase 7 —
  (`programs/references/semantics.rs`) records, per region under
  construction, which alias family each derived reference atom belongs to and which operation consumed each family.
  `ProgramBuilder` holds one and exposes `add_staged_instruction`, which `StagingContext::stage_operation` now calls
  instead of `add_instruction`. Two misuses are reported at the staging call: accessing a reference whose alias family
  was already consumed, and consuming a handle whose view narrows the root the consumption would invalidate. The state
  needs no per-clone bookkeeping, because it is keyed by `AtomId` and every clone of one `Tracer` names the same atom
  — which is exactly the "shared state across clones" §4 asks for, obtained from the SSA identity rather than built
  beside it.
- **Alias families are transitive and include structured carries.** An identity edge onto a view still names part of a
  root, so narrowing is carried along the chain rather than read off the last edge. And a reference handed back out of
  a structured boundary declares nothing in its reference semantics — the forwarding is stated through
  `Operation::reference_output_identity_input` — so `record` consults that hook as well, which is what puts a `while`
  carry's result in its operand's family.
- **Capture rejection moved to where it happens.** `StagingContext::lift_constant` — renamed `checked_constant` at
  phase 7 — runs `Value::validate_as_constant`
  before staging a constant, so a value family that forbids constant storage — a mutable reference holder, most
  notably — is reported at the lift rather than at `build`. Region sealing still performs the same check over every
  stored constant and remains the backstop that covers non-tracing construction paths.
- **By-value `freeze`** (§8 decision 4). `FreezeReference::freeze` takes `self`, with all three implementations
  migrated. "Both surfaces" is satisfied at the value-level capability, which is the surface a user calls; the two
  inherent runtime methods `Reference::freeze` and `ArrayReference::freeze` deliberately keep `&self`, because
  consumption there is a property of the shared holder and a by-value signature would promise a linearity that
  cloning the handle always defeats. `ArrayReference::freeze` now says so.
- **The analysis reduction, which is small and honestly so.** `ReferenceAnalysis::alias` is gone: it had no
  production consumer, and `aliases()` already exposes the same data to the one consumer there is. Everything else
  the reduction reached is named by phase 7's own deletion audit, which is where the remaining unconsumed members
  were found.

**Surviving consumers, named** (the inventory §4 asks for). Production: the eager lifetime preflight
(`ArrayIrValue::validate_eager_replay` -> `RegionRef::validate_reference_lifetimes`) and the preserved-reference
kernel path (`PreservedReferenceKernelOperation::validate_body` -> `RegionRef::analyze_array_references`, the sole
consumer of `ArrayReferenceAnalysis::view` and, through the overlay, of `ReferenceAnalysis::aliases` and
`ReferenceAnalysis::root`); `accesses` and `external_roots` are consumed by that same validator. The four
`Program`-level `analyze_*` entry points have no in-tree production caller and are retained deliberately: they are the
authored-program lint §2.3 names as a surviving surface, and they are what a program built other than by tracing is
checked with. `instruction_summary` and `ReferenceTransitiveAccess`'s accessors are likewise retained: the
summarization behind them runs unconditionally and is load-bearing validation (`ForbiddenRegionInputAccess` and the
caller-root binding check), so the query is the only way to observe what that pass computed. Discharge consumes none
of it; its own `ReferenceRegionSummary` is independent. Site enumeration (`Program::reference_discharge_sites`) reads
only `Operation::reference_semantics` and was independent of the analysis from phase 0.

**Decisions this phase forced.**

1. **The check belongs to the staging seam, not to `add_instruction`.** Placing it in the builder's general append
   would make a deliberately malformed reference program unconstructible, and several tests build exactly those to
   assert the analysis's `UseAfterConsume`/`InvalidConsume` and discharge's own environment diagnostics. Confining it
   to `stage_operation` keeps the rungs of the ladder distinct: trace time reports what a tracing call did, the
   analysis reports what an authored program contains, and discharge reports what its environment observed.
2. **Trace time is deliberately more permissive than the analysis about *what* may be consumed.** The analysis
   restricts consumption to a root allocated in the consuming region; partial discharge deliberately accepts a program
   that consumes a surviving external root (phase-4 decision 3). Adopting the analysis's rule here would forbid a
   capability the transform supports, so the trace-time rung enforces only what the runtime and discharge semantics
   require: whole-family invalidation, and whole-root consumption.
3. **The interpreter clones its operand to freeze it, and that weakens nothing.** Interpretation replays an
   already-built instruction from a borrowed environment. A clone names the same root, so consuming it invalidates the
   same family; the linearity the capability enforces was never this layer's to enforce.

**Deviations from the plan, with rationale.**

- **`arrays/reference_views.rs` and the tracing handle modules, named as phase-6 owning files in §6, are untouched.**
  Both were named against the traced-handle model Track B option 1 would have forced. With SSA views retained
  (§3.1) a derived handle is an ordinary SSA value, and the eager handles already implement clone-shared
  generation and freeze invalidation over an `Arc`-shared holder. There was nothing left to build there.
- **The reduction is smaller than §6's "deletions reported below" implies.** Phase 3's cutover already deleted
  `region_summaries`, `region_summary`, and `is_reference_free` the moment the planner went, so what remained
  unconsumed was three items totalling roughly 40 production lines.

**Audit round.** One independent Opus auditor reviewed the complete phase-6 surface for correctness and conventions.
It confirmed the headline property — no false rejection, argued from the sealed-region import discipline, the
strictly-increasing atom identities that make the alias walk terminate, and the fact that the state can only add
edges — and found three real defects, all under-rejections, all fixed:

- **Transitive views escaped the consume rejection.** The check read the immediate alias edge, so a `root -> View ->
  Identity` chain consumed cleanly. Narrowing is now carried along the chain and resolved at insertion, which also
  turned the family walk into a single map lookup.
- **Structured carries were invisible.** `record` consulted only `reference_semantics`, so a reference forwarded out
  of a `while`, `scan`, or call boundary started a fresh family and a freeze of the root left the carry usable. The
  identity-forwarding hook is now consulted for every output.
- **The consume-a-view rejection was inverted relative to the analysis in one respect that had to stay** — the
  analysis rejects consuming *any* alias and any non-allocation root. Decision 2 above records why that was not
  adopted, and the trait documentation now states the divergence rather than leaving it implicit.

Also fixed from the audit: the fast path in `add_staged_instruction` was removed rather than optimized, because the
identity-forwarding probe must run for operations that declare no reference semantics at all, and the recorded
application is now read back off the appended instruction — a disjoint field borrow — so recording costs neither an
operation clone nor an operand-list clone; a gratuitous `.clone()` before a `freeze` whose handle was dead afterwards;
the interpreter's comment, which claimed a guarantee the direct-construction hatch does not provide; the
`semantics.rs` module documentation and the `ReferenceLiveness` claim that one builder builds one region (it
*constructs* one and imports others sealed); `add_staged_instruction`'s claim to be the only staging-shaped append,
which now names partial evaluation's residualization and reverse mode's transposition splice; and four over-long
lines that `rustfmt` cannot break because they are string literals. The stale rustdoc links left by
`region_root_for_source` and the tests that asserted it belong to phase 7's audit, which is where that query was
removed.

Two audit findings were deliberately not acted on. The trace-time rung reports `ProgramError::MalformedProgram` where
the analysis reports a structured `ReferenceAnalysisError`; the wording differs because the two rungs know different
things (the analysis names an instruction coordinate, which does not exist yet at the staging call), and
`MalformedProgram` is what every other builder-level rejection uses. And a nested-region false-rejection test — two
condition branches each consuming their own allocation — was not added: the property follows from region builders
being independent by construction, and the 1,605 `ryft-core` and 542 `ryft-xla` tests that trace structured reference
programs are the standing regression evidence. Both are recorded here rather than left silent.

**Known gaps recorded rather than fixed.** Consumption performed *inside* an attached region is invisible to the
enclosing builder's state, because the region is traced by its own builder; discharge and the analysis both reject
that shape, so the ladder still catches it, one rung later. `splice_program` and the other rebuild paths bypass the
check by design. And the identity-forwarded carry is pinned end to end for `while` only; `scan` and positional calls
share the same hook and the same code path.

**Verification.** `cargo test -p ryft-core --lib`: 1605 passed, 0 failed, 3 ignored. `cargo test -p ryft-core
--tests`: 1605 + 6 + 6 + 5 + 1 passed across the lib target and the four integration binaries. `cargo test -p ryft-core
--doc`: 65 passed and 16 ignored, plus 5 compile-fail cases. `cargo test -p ryft-xla --lib`: 542 passed, 5 ignored.
`cargo test -p ryft-pjrt --lib`: 130 passed. `cargo test -p ryft-macros`: 57 + 1 passed. `cargo test -p
ryft-macros-tests`: 21 + 17 passed. `cargo check --workspace --all-targets`: zero warnings and zero errors. `cargo fmt
--check` scoped to the four touched crates: clean.

Phase-6 size, against §6's estimate of ~250/300/60: roughly 150 production lines across `semantics.rs`,
`builders.rs`, `contexts.rs`, and `tracing.rs`, about 200 test lines, and about 90 of documentation, against roughly
40 production lines deleted. The estimate was high for the same reason the reduction is small: it assumed a traced
handle would need invalidation machinery of its own, and the SSA atom already is that machinery.

### Phase 7 — stabilization (landed)

**What landed.** The owner-raised question is answered by deletion, the parked surface questions are closed, the
documentation says what the code does, and the subsystem converged over three independent audit rounds.

*The owner-raised item: `ReferenceDischargeRule` is deleted.* Its one consumer,
`PreservedReferenceKernelOperation::validate_body`, now derives a private `KernelReferenceRole` from
`Operation::reference_semantics` through `kernel_reference_role`. The equivalence is exact rather than approximate,
and the argument is short: the validator already rejects every instruction that carries attached regions before the
classification loop runs, so the four structured rules were unreachable there, and the remaining seven map one-to-one
onto declared semantics — an allocation is a `NewRoot` output, a view is an `Alias` output with no access, and read,
replacement, accumulation, and consumption are each a single access of their own mode. What that removes is the last
operation-family declaration that duplicated `reference_semantics`, which is §1's thesis applied to the one place the
cutover left it standing: a family had to state its rule *and* its semantics and keep them agreeing, and the validator
did not even trust the rule — it re-derived the canonical primitive and cross-checked.
`ArrayReferenceDischargeOperation` went with it, since the rule was its only method, along with both implementations and
their mapping tests. One further check went too: the validator's "reports ordinary lowering for a reference-bearing
contract" rejection is unreachable now that the role is derived, because the analysis it runs first refuses to classify
exactly that instruction; the test that covered it now asserts the rung that does.

*The parked deletion-audit items.* `ReferenceDischargeEnvironmentId` is private (its only out-of-module appearance was
a compile-fail fixture, which now proves something stronger — the environment identity is not even nameable
downstream); `ReferenceCaptureScope` and `ReferenceDischargeContext::captures` are private; and
`ReferenceDischargeTracer::context`, `ReferenceAnalysis::region_root_for_source` with its backing table, and
`ReferenceAlias::input_index` with its field are gone, each having had no consumer at all. Two tests that asserted the
deleted query were rewritten onto `instruction_summary`, which observes the same caller-root substitution through the
artifact whose computation actually validates it.

*Documentation.* `programs/references/mod.rs` gained the two pieces §4 asks for: a JAX correspondence naming what maps
onto `jax._src.state` and what does not, and the four-rung prevention ladder — trace time, the eager runtime, the
authored-program lint, and discharge — with what each rung can see, why the rung below it still exists, and the one
place they deliberately disagree. Three doctests were added: partial discharge end to end, the compile-fail that pins
§8 decision 4's linearity promise beside the runtime failure a clone still produces, and the prevention ladder's own
rung-1 rejection.

**Decisions this phase forced.**

1. **Deriving beats re-homing.** §4 offered two answers for `ReferenceDischargeRule`: delete it, or re-home it as
   kernel vocabulary. Re-homing was feasible — `ryft-xla` could define the trait and implement it for both families —
   but it would have preserved a second declaration that a family must keep in sync with its semantics for no gain,
   because the validator already re-derives what it needs. Deleting is what the fact-layer principle in §1 requires.
2. **`ReferenceDischargeValue::Pure` became `::Ordinary`.** §8 decision 1 fixed the name, but the repository has since
   made "pure" mean effect purity and nothing else, and this variant means "not a reference handle" — the same
   collision that renamed `discharge_pure_operation` at phase 4. The variant's own documentation already said
   "ordinary destination value", so that is the name it took. The decision in §8 is superseded on this one point.
3. **Trace-time lifetimes are named for what they track.** `ReferenceLiveness` became `ReferenceLifetimes`, because
   "liveness" already means dataflow live-variable analysis in this crate (`ProgramLiveSets`, `kernel_data_liveness`)
   while the reference subsystem calls this concept lifetimes (`validate_reference_lifetimes`). Its `validate` and
   `record` now take the operation and derive the semantics themselves, rather than accepting both — which removed
   the parameter that let a caller describe one contract and record another, a mistake the first version of its own
   test had already made.

**Audit rounds.** The house three-auditor convergence protocol ran to a clean round, which took five. Round 1 put
three independent Opus auditors — correctness, conventions, and documentation/plan accuracy — over the complete
surface concurrently, and their findings were reconciled into one fix pass. The correctness auditor reported
**no behavioral defects**, having tried specifically to break the kernel-role equivalence, the unreachability argument
behind the deleted check, and the trace-time state; it confirmed the equivalence arm by arm and confirmed that
`checked_constant` changes no legitimate path, because capture lifting stages through the unchecked primitive and
every other staging context delegates to a parent `lift`. What the round changed:

- **The kernel role, the lifetime state, and the checked lift were all renamed** (decisions 2 and 3, plus
  `KernelReferenceShape` to `KernelReferenceRole`, since "shape" is array vocabulary in this repository, and
  `lift_constant` to `checked_constant`, which pairs with the unchecked `constant` and does not collide with a
  private free function of the same name in `discharge.rs`).
- **`record` no longer tracks non-reference atoms.** Loop carries identity-forward positionally regardless of type,
  so every array carry was inserting an alias edge. The builder now records only for an application that declares
  reference semantics or names a reference-typed value, which restores the empty-state fast path and makes the
  field's documentation true.
- **Nine documentation claims that the code does not support were corrected**, the sharpest being the ladder's claim
  that discharge reports "instruction-level coordinates" — its root handles are interpreter identities by design and
  its diagnostics name the operation and the root, not a source coordinate — and the JAX section's claim that four
  things are ahead of JAX, when two of them (the effects gates and the shared view traversal) are load-bearing here
  without being deltas. §2.3 carried the same overstatement and was corrected in lockstep.
- **Twelve plan sentences were updated**, including §2.1's description of a trait this phase deleted, §2.1's
  capture-scope sketch, §2.3's diagnostic claim, §3.1's ledger footnote, §6's exclusion and deletion totals, and the
  two parked-deferral paragraphs in the phase-2 and phase-3 records, which a reader would otherwise take at face
  value and go looking for public items that are no longer there.
- **Test coverage was strengthened where the deletions had weakened it**: both rewritten substitution tests regained
  a negative case, the alias-edge assertion compares the whole recorded vector instead of searching it, the kernel
  role test pins all eight diagnostic names and both alias kinds, and the traced rejections assert exact errors
  rather than `matches!` guards.

The protocol then iterated to a clean round, which took four more. Round 2 (correctness) returned **converged** after
verifying the fix pass introduced no regression — it additionally confirmed the compile-fail doctest fails for the
intended reason by reconstructing it in an out-of-tree crate, and confirmed the record-filter fix with a traced `while`
carrying an array and a reference side by side. Conventions and documentation took rounds 3 through 5, on a shrinking
list of mechanical items: two tautological "negative" assertions the round-1 fix pass had introduced (each following an
`assert_eq!` against a different root, so neither could fail) were replaced with assertions against the innermost
formal root, which is what makes the caller-side substitution non-vacuous; a test comment that contradicted its own
assertion was corrected; a reflow regression, seven short rustdoc paragraphs, seven stray trailing commas, and two
over-long lines in this record were fixed; and three more documentation claims were corrected, including one the
round-2 fix pass had changed to a *different* wrong claim — §2.3 scoped "naming the operation" to precisely the two
diagnostics that name only the root. Round 5 returned clean on every gate.

Findings deliberately not acted on, each recorded rather than silently dropped: two trace-time coverage holes that are
missed rejections only (a pure alias *of* an already-consumed family is reported at the next access rather than at the
alias, and `ConditionOperation` declares no identity forwarding — correctly, since its outputs come from two branches
that may return different roots, so no positional identity exists to declare); the four
`TODO(eaplatanios): Review this module.` markers in the arrays reference modules, which are owner review requests
rather than phase scaffolding; the pre-existing rustdoc on trait `impl` blocks in `ryft-xla`'s `ops.rs`, which predates
this work; and the asymmetry that `programs::references::operations` is public while its four sibling modules are not.

**Known gaps recorded rather than fixed.** `validate_body` has no in-tree production caller — the preserved-reference
kernel boundary is an experimental surface that defines and tests a contract rather than serving a pipeline, which
§2.3 already says — so the role derivation is exercised by tests and by the discharge suites rather than by a shipping
path. A failed `checked_constant` poisons the trace, which is right for every caller in the tree but makes `lift`
unconditionally trace-fatal.

**Verification.** `cargo test -p ryft-core --lib`: 1604 passed, 0 failed, 3 ignored. `cargo test -p ryft-core
--tests`: 1604 + 6 + 6 + 5 + 1 passed across the lib target and the four integration binaries, including the two
regenerated `trybuild` compile-fail snapshots. `cargo test -p ryft-core --doc`: 68 passed and 16 ignored, plus 6
compile-fail cases (three doctests and one compile-fail case added this phase). `cargo test -p ryft-xla --lib`: 543
passed, 5 ignored. `cargo test -p ryft-pjrt --lib`: 130 passed. `cargo test -p ryft-macros`: 57 + 1 passed. `cargo test
-p ryft-macros-tests`: 21 + 17 passed. `cargo check --workspace --all-targets`: zero warnings and zero errors. `cargo
fmt --check` scoped to the four touched crates: clean. `cargo doc -p ryft-core --no-deps`: zero warnings from any file
this phase touched.

Phase-7 size, against §6's estimate of ~0/100/300: roughly 90 production lines added (the kernel role derivation) and
about 420 deleted (`ReferenceDischargeRule` with its two family implementations and their tests,
`ArrayReferenceDischargeOperation`, the three unconsumed items, and the check the derivation made unreachable), plus
about 250 documentation lines and 150 test lines. Two further items were privatized rather than deleted. The estimate
was right that no new production *surface* lands; it did not anticipate that answering the owner-raised question would
make the phase a net deletion.

### Post-completion deletion record (2026-08-23, owner-directed)

After the plan completed, a deletion audit established that the retained static analysis stack had exactly one live
production edge (the eager replay preflight, which consumed only an entry-boundary fact) and that the preserved-
reference kernel validator — the other intended consumer — was reachable only from its own tests as the seam proof
for `plan-pallas.md`. The owner directed deletion of both, resolving the prevention ladder to three rungs (trace
time, the eager runtime, discharge):

- Deleted `crates/ryft-core/src/programs/references/analysis.rs` (the generic whole-closure `ReferenceAnalysis` with
  its seven exported types and 35-variant error enum), `crates/ryft-core/src/arrays/reference_analysis.rs` (the
  `ArrayReferenceAnalysis` view overlay), and `crates/ryft-xla/src/experimental/reference_kernels.rs` (the
  preserved-reference kernel validator mock). `ReferenceSource` had already been rehomed to `semantics.rs` and
  survives as the discharge and persistence boundary vocabulary.
- Replaced the eager replay preflight's whole-closure analysis with an inline entry-input scan in
  `ArrayIrType::validate_eager_replay` that preserves the external-reference rejection and its exact diagnostic; the
  `EagerReplayValidation` evidence contract is unchanged. (A follow-up owner-directed vocabulary pass then renamed
  this mechanism onto the codebase's canonical interpretation vocabulary — `validate_eager_interpretation`,
  `VALIDATES_EAGER_INTERPRETATION`, `EagerInterpretationValidation`, and "boundary validation" instead of
  "preflight" in prose — without changing behavior.)
- The generic discharge boundary tests' shared fixture, previously imported from the analysis test module, was
  rebuilt in `discharge.rs`'s own test module, trimmed to the four operations those tests use.
- The prevention-ladder documentation in `programs/references/mod.rs` and every doc cross-reference were rewritten;
  one eager test asserting the deleted whole-closure pre-validation of unselected condition branches was removed, and
  the preflight rejection test now asserts the scan's diagnostic.
- `plan-pallas.md` gained a restoration phase (its Phase 6) that rebuilds the analysis and kernel validation against
  the real kernel operation; its retained-pieces inventory records the deletion. Note the deleted files were never
  committed; approximate transcript-recovered copies were archived in the session scratchpad at deletion time.
- Verified: full `ryft-core` suites (1,547 lib tests, integration suites, 68 doctests, the 5-test downstream surface
  with both compile-fail fixtures), `ryft-xla` (529 lib tests), workspace-wide `cargo check --all-targets`, and
  `cargo fmt`, all clean with zero warnings. Net effect on the working tree relative to `main`: the reference rework
  now deletes more than it adds.
