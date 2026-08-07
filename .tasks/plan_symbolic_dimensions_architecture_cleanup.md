# First-class dimension architecture cleanup

## Status

Active; repository state was re-audited on 2026-07-29 after the original side-chat history was lost. Phases 0–2,
P3a–P3j, and the public broadcast consolidation are committed on `u/eaplatanios/dynamic-shapes`; the clean resumption
point for the current review was `0278adba4617a89fcf9db88c7c27836b3e39d845`. P3k's collective parity surface now
supports tiled and untiled shape semantics, validated participant groups, all-gather variance, and the
`pshuffle`/`pswapaxes` compositions. Phase 4 has completed the production XLA/composite cutover and verified explicit
dimension authority through condition, while, scan, eager branch execution, nested-region import, and repeated
alpha-equivalent program splicing. Arithmetic extents remain ordinary dimension SSA through inference, eager
execution, PE, batching, JVP/VJP, import, rendering, and native lowering. Phase 6's shape-changing collective
adjoints are implemented through ordinary linear-call residuals. Phase 7's compiled gateway, diagnostic assertions,
per-class effect-token lowering, explicit bounded physical ABI, and CUDA `PadToStatic` execution are complete. The
final current-JAX comparison fixtures remain an explicit P3k gate.

On 2026-08-01 the plan's end state was extended beyond the containment cleanup: after the cleanup closure gates
(Phases 10–11), Phases 12–14 take Ryft from input-derived (tier-2) dynamism to full data-dependent (tier-3) dynamism.
The semantic entry point already exists — P3d landed the checked `dimension_from_scalar` provenance gateway with
eager bounds-checked execution, partial evaluation, and mapped-batching rejection — so the remaining tier-3 work is
Phase 12 (close the semantics: effects-model coverage, cache-identity and transform coverage, fixtures), Phase 13
(bounded data-dependent compiled execution), and Phase 14 (ragged batching). The tier definitions and the
representation rationale are recorded at the end of the **Objective** section; the P6a `LinearCallOperation` residual
contract is deliberately reused unchanged as tier 3's differentiation mechanism, which is the strongest evidence that
the tier-2 architecture was built correctly.

This plan remains a containment and simplification follow-up to `.tasks/plan_first_class_dimension_programs.md`. It
preserves that plan's user-visible capabilities and its decision to represent runtime dimensions as ordinary SSA
values, but supersedes the following implementation compromises:

- one operation payload implementing materially different `Operation<ArrayType>` and
  `Operation<ArrayIrType>` contracts;
- a complete homogeneous `ArrayOperation` graph used as an implicit-shape transform staging language;
- `ArrayContextView` side channels that recover dimension operands from ambient dimensions or source arrays;
- replay-time conversion of implicit shape dependencies back into explicit dimension SSA operands;
- array/dimension-specific projection traits where a generic typed projection is sufficient; and
- operation-declared identity-source overrides where closure can derive producer versus forwarding behavior
  structurally.

The migration must not restore expression trees, witnesses, scopes, substitution, hidden shape environments, packed
shape tensors, host readback, or a second dimension program. Backwards compatibility is not a goal. Do not add
compatibility aliases or retain the old homogeneous shape-program surface after its consumers migrate.

The original archive/increment workflow is retained below as historical execution evidence, not as the current source
of truth. The immutable archive branch remains protected, but the mutable remainder and delivery ledger stopped being
maintained after P3c/P3h respectively and must not be used to overwrite or reconstruct the newer integration tree.
The 2026-07-28 repository audit in the **Review** section and the live integration worktree now determine resumption.

## Objective

Retain the power, diagnostics, transformation behavior, and backend efficiency of first-class dimension SSA while
making the implementation proportionate:

1. Every operation payload has one stable semantic signature.
2. Dynamic extents remain explicit operands from construction through every transform and lowering.
3. The heterogeneous array/dimension sum is contained at program storage and genuinely mixed operation boundaries.
4. Homogeneous rule bodies use generic, zero-state typed projections rather than a special-purpose replay domain.
5. Batching, differentiation, and partial evaluation express policies for value kinds instead of duplicating an
   array/dimension transform engine.
6. Region closure derives identity ownership and forwarding from type occurrences and graph structure.
7. The complete array-program operation classification has one source of truth.
8. Production code and generated code are measurably smaller than the implementation that this plan replaces.

This is an architectural cleanup, not a feature reduction. The final implementation must continue supporting:

- bounded dynamic reshape, broadcast, slicing, padding, gather, concatenate, reduce, constructors, RNG, collectives,
  and custom calls;
- first-class dimension arithmetic, comparisons, data gateways, and ordered requirements;
- nested condition, while, scan, custom derivatives, and rematerialization;
- eager interpretation, tracing, partial evaluation, batching, JVP, VJP, transposition, JIT, import, caching, and XLA
  lowering;
- caller/callee requirement composition and exact diagnostics; and
- replicated-only dense batching/sharding of dimension authority.

As of 2026-08-01 this plan also owns Ryft's dynamism end state. The dynamism tiers are: **tier 1** — static extents;
**tier 2** — input-derived extents (dimension inputs, `dimension_size` of array inputs, and dimension arithmetic over
those), which is the scope of Phases 0–11; and **tier 3** — data-dependent extents computed from array contents within
declared bounds. Phases 12–14 extend the finished cleanup to full tier 3. The dimensions-as-SSA-values representation
was chosen with this end state in mind: it is the architecture JAX's own dynamic-shapes effort
(`DShapedArray` avals containing jaxpr variables, `bint` bounded integers with padding rules) headed toward before
stalling on retrofit scale, and tier 3 is reachable here by relaxing provenance *policy* through one checked gateway
rather than by re-architecting types, transforms, or programs. The gateway itself (`dimension_from_scalar`) already
landed in P3d; Phases 12–14 complete the semantics, backend execution, and batching around it. Every tier preserves
the same invariants: rank stays static, types carry identities and bounds but never expressions or values, and the
graph remains the complete source of data dependencies.

## Architectural diagnosis

The broad semantic reach of first-class dimensions is legitimate. Program storage, type closure, transforms, region
interfaces, shape operations, and lowering all need to understand dimension values. The present implementation
nonetheless pays avoidable costs because it has two shape-operation contracts:

```text
homogeneous rule graph:
    array operation + shape metadata

stored array-program graph:
    array operation + explicit dimension SSA operands
```

`ArrayContextView::bind_replayed` bridges those contracts by inspecting operation payloads, discovering dynamic
variables, searching ambient dimension values, or synthesizing `dimension_size` readers from source arrays. This is a
hidden dependency environment. It is smaller than the deleted expression/witness machinery, but it violates the same
architectural invariant: the graph temporarily stops being the complete source of data dependencies.

The current operation trait still permits the same payload to implement multiple type-family contracts. The
post-Phase-7 inventory at `7da7d7f25`, recorded in `.tasks/plan_p8a_operation_contract_inventory.md`, supersedes the
2026-07-28 count:

- `ConcatenateOperation`, `CustomCallOperation`, `PadOperation`, and `RngBitGeneratorOperation` directly implement
  both `Operation<ArrayType>` and `Operation<ArrayIrType>`;
- `CompareOperation` has a generic homogeneous `Operation<T>` implementation plus its distinct composite dimension
  comparison contract;
- `ShardMapOperation<V>` implements both array and composite contracts;
- `SelectOperation` and `StopGradientOperation` each implement scalar and array contracts; and
- ten more payloads use a type-independent generic `impl<T> Operation<T>` and must become nominally typed before an
  associated operation type is representable: `PrintOperation`, `WhileOperation`, `RematerializeOperation`,
  `CustomJvpOperation`, `CustomVjpOperation`, `OneLikeOperation`, `ZeroLikeOperation`, `TagOperation`,
  `ConvertElementTypeOperation`, and `JitCallOperation` (`CompareOperation` belongs to both categories).

`BroadcastOperation`, `ReshapeOperation`, and `DimensionSizeOperation` have one canonical composite contract, while
remaining homogeneous behavior lives under explicit legacy payloads where still needed. Dynamic slice, gather,
reduce, and ordinary slice are intentionally array-only. Constructor mixed semantics must continue to live in the
composite variant arm rather than adding another dual-contract payload.

The archived inventory also included overlapping generic constructor contracts, which are a distinct and harder
case:

- `ZeroOperation<ArrayType>`;
- `OneOperation<ArrayType>`;
- the former specialized array fill operation; and
- `IotaOperation<ArrayType>`.

The current branch has removed those four archived array-program-specific implementations. `One`, fill, and `Iota`
appear only in the homogeneous array family. One temporary exception remains:
`ArrayIrOperation::Zero(ZeroOperation<ArrayIrType>)`, which lets generic differentiation machinery
materialize a composite array zero without explicit geometry. The final constructor design cannot be resolved merely
by retaining that generic escape hatch. The canonical destination is:

- operand-relative `zero_like`/`one_like` for transform-generated values whenever a source array exists;
- a homogeneous nullary constructor for zero, one, and iota only when its stored output type is identity-free,
  enforced by the blanket `Operation<T>` inference;
- a typed rank-zero `ConstantOperation` literal, followed by ordinary broadcast for every rank-positive fill; and
- for identity-bearing zero, one, and iota output types, a mixed `Operation<ArrayIrType>` contract owned by the
  corresponding flat `ArrayIrOperation` variant arm. The stored `ArrayType` is the complete output authority and
  the variant consumes one explicit dimension operand per dynamic axis in identity-validated axis order.

Fill intentionally has no mixed constructor variant. JAX implements scalar `lax.full` as dtype conversion plus
broadcast and implements array-valued `jax.numpy.full` through ordinary `broadcast_to`; Ryft follows that graph
directly. A rank-zero literal fill or caller-provided array SSA value feeds the existing homogeneous or mixed
broadcast operation, whose explicit dimension operands own dynamic output geometry.

No wrapper type exists and no payload carries two trait implementations: the mixed contract is owned by the
`ArrayIrOperation::DynamicZero` variant arm, which delegates rendering, identity renaming, and structural flags
to the payload's single `Operation<ArrayType>` implementation and calls the shared dynamic-constructor inference
helper directly. The `From<ZeroOperation<ArrayType>>` conversion routes canonically: reference-bearing dynamic output
types become `DynamicZero`, identity-free types become the homogeneous member family's zero, and the helper rejects
reference-free stored types so each zero has exactly one encoding. Static and dynamic construction must not be
distinguished by ambient operand recovery. `DynamicZero` is the unambiguous migration name while the temporary generic
composite `Zero` variant still exists. Phase 6 deletes that generic variant and then renames `DynamicZero` to `Zero`;
the final top-level name denotes the genuinely mixed `(Dimension...) -> Array` adapter, while identity-free zeros
remain nested under the homogeneous `Array` member family.

`DimensionSizeOperation` demonstrates why this is unsafe: its homogeneous contract returns a rank-zero integer array,
while its heterogeneous contract returns a first-class `DimensionType`. An operation's result kind must not depend on
the surrounding trait instantiation.

`ReshapeOperation::transpose_dimension_variables` is a second concrete containment failure. The corresponding values
are explicit dimension SSA operands, so it is not an expression-evaluation witness, but the payload field is a
differentiation-only residual manifest understood only by reshape and composite differentiation. Other transpose rules
will need the same primal-extent retention. Residual selection and threading belong to the differentiation transform,
not to individual primal operation payloads.

The original working-tree baseline was provisional because the parent refactor was uncommitted. Phase 0 subsequently
captured:

- tracked and untracked production/test line counts relative to `HEAD`;
- non-test line counts for `backends/array_programs/{mod,batching,differentiation}.rs`;
- occurrences and files containing `ArrayIrProjection`, `ArrayContextView`, `DimensionContextView`,
  `with_dimensions`, `with_source_array`, `bind_replayed`, `runtime_dimension_variables`, and
  `OutputIdentityRole`;
- operation-family variant counts and duplicate operation-contract implementations;
- clean-build and incremental `cargo check` time, peak `rustc` memory, generated macro token counts, and release binary
  size for the existing golden programs;
- graph instruction count, rendered IR size, allocations, compile time, and runtime for the existing static and dynamic
  golden programs; and
- exact eager, transform-time, compile-time, and runtime diagnostics.

The snapshot observed while drafting this plan contains 80 tracked `ryft-core` files with approximately 16.8k
insertions and 7.1k deletions, plus approximately 14.2k lines in new untracked core files. Composite-family identifiers
occur more than 1,400 times across 35 core files, and the mixed type-projection cursor is constructed at roughly 45
production/test sites. These are investigation inputs, not acceptance baselines; record reproducible counts from the
exact source state immediately before execution.

## Non-negotiable invariants

### One graph

- Runtime dimension arithmetic is ordinary SSA in the same program as array computation.
- Rank stays static.
- A dynamic array axis stores one dimension identity and its authoritative bounds, never an arithmetic expression.
- Mixed shape operations consume their runtime extents as explicit dimension operands. When an operation's output
  shape is fully described by those operands, exact dimension constants represent static axes and the output shape is
  derived rather than duplicated in the payload.
- Dimension values are never reconstructed from rendered strings, names, ambient maps, or expression metadata.

Leaf-only dimensions remain a deliberate trade, not an unexamined assumption. They keep type metadata decidable and
bounded, make identity and bounds the only type-level authority, and avoid restoring:

- polynomial/expression normalization;
- substitution and capture-avoidance;
- expression scopes and mixed-scope diagnostics;
- witnesses and expression evaluation environments;
- a second persistence/canonicalization format; and
- an incomplete algebra that still cannot represent conditional, loop-carried, or data-dependent extents.

The accepted cost is that reverse-mode rules cannot recover every primal extent from cotangent types. They must retain
the required first-class dimension SSA values as ordinary differentiation residuals. This is a transform concern and
must be handled once by differentiation machinery; it is not permission for operation-specific residual-variable
fields. A bounded product-of-leaves type expression is explicitly rejected because it would restore the parallel
expression language while solving only the algebraic subset of residual needs.

### One operation contract

- One concrete operation payload has one operand/result/region contract, and therefore one `Operation` trait
  implementation. When a payload with an existing homogeneous contract needs mixed composite semantics, those
  semantics are owned by the composite family's variant arm (the nominal adapter), not by a second trait
  implementation on the payload. This is what keeps the design representable under Phase 8's associated-type
  `Operation { type Type; }`, whose compile-fail goal is precisely that a payload cannot acquire two type contracts.
  Remaining transitional dual implementations (e.g., `CompareOperation`) stay in the Phase 0 inventory and must be
  resolved the same way before Phase 8 lands.
- `dimension_size` means `array -> dimension` in every context.
- `dimension_to_scalar` is the explicit `dimension -> scalar-array` gateway.
- `dimension_from_scalar` (landed in P3d) is the explicit checked `scalar-array -> dimension` gateway and the *only*
  operation that converts data into dimension authority. It declares a fresh `DimensionVariable` with caller-declared
  `DimensionBounds`, and eager execution checks every accepted integer payload against them. Phase 7 lowers its
  compiled bounds check on the `OrderedAssertion` chain and declares that effect in core so the check survives DCE and
  preserves deterministic same-class ordering.
- A shape-carrying operation is mixed even when a particular invocation happens to have only static dimensions.
- A static convenience API may omit dimension operands only when the payload's metadata proves there are none; it
  still binds the same mixed operation contract. Constructors are the documented exception: an identity-free
  constructor has one canonical encoding inside the homogeneous array member family, and the mixed dynamic
  constructor rejects reference-free stored types so equivalent zeros cannot acquire two enum representations.

### Contained heterogeneity

- `ArrayIrType` and `ArrayIrValue<A>` remain the single storage sum for an array program.
- Array-only payload inference and semantic rules receive `ArrayType` and typed array values.
- Dimension-only payload inference and semantic rules receive `DimensionType` and typed dimension values.
- Mixed payloads receive the storage sum only because their signatures genuinely cross kinds.
- Generic program machinery must not match on array/dimension variants.

### Explicit transformation dependencies

- Transform rules may forward, clone, or synthesize a dimension SSA operation explicitly.
- Transform rules may not recover an unlisted dimension operand from output type metadata or a source-array side
  channel.
- If a transform introduces a new dynamic axis, the transformed graph must contain the operation that produces its
  dimension value.
- Structural zeros should remain structural where possible. Materialized zeros/ones should prefer operand-relative
  `zero_like`/`one_like` operations rather than shape-metadata constructors that require dependency recovery.

### Stable diagnostics and effects

- Proven requirements erase, disproven requirements fail with exact typed diagnostics, and inconclusive requirements
  retain `Effect::OrderedAssertion`.
- No cleanup may weaken actor names, observed values, bounds, divisibility, equality, overflow, or underflow
  diagnostics.
- Assertion effect ordering, DCE survival, and partial-evaluation behavior remain unchanged.
- Backend lowering must finish the semantic distinction that the effect model already exposes. The current XLA
  lowerer has one `Option<ValueRef>` token per lowering scope and therefore serializes assertions and ordered I/O on
  the same chain, despite `Effect` documentation promising one chain per ordered class. Phase 7 must lower
  `OrderedAssertion` and `OrderedIo` through independent deterministic chains and thread the active chain set through
  condition, while, scan, rematerialization, and inlined calls.

## Target architecture

### Canonically typed operations

Prototype changing `Operation<T>` to an operation-owned associated type:

```rust,ignore
pub trait Operation: Clone {
    type Type: Type;

    fn name(&self) -> &'static str;

    fn infer_output_types(
        &self,
        input_types: &[Self::Type],
        region_interfaces: &[RegionInterface<Self::Type>],
    ) -> Result<Vec<Self::Type>, TypeError>;

    // Rendering, identity renaming, regions, effects, and related methods.
}
```

Programs then require `O: Operation<Type = V::Type>`. Generic payloads such as `ZeroOperation<T>` remain generic and
set `type Type = T`; the operation payload still has one contract for each concrete instantiation. Homogeneous wrapper
enums require every payload to use the same associated type. The heterogeneous outer dispatcher is a distinct
operation type whose associated type is `ArrayIrType`; it adapts inner array and dimension operation families.

This change is recommended because it makes the most important invariant compiler-enforced. It is not permission for
an unbounded mechanical sweep. Phase 8 prototypes the trait only after the semantic architecture is stable, covering
the derive macro, one generic operation, one homogeneous enum, and the heterogeneous vertical slice. Proceed only if:

- trait solving remains stable;
- generated bounds become no larger or clearer;
- the implementation produces a net reduction in generic bounds/adapter code;
- no pervasive wrapper type is needed merely to name an operation's type; and
- the full migration map demonstrates that the mechanical churn is bounded and reviewable.

If the associated-type prototype fails those gates, retain `Operation<T>` and introduce one small sealed
`OperationContract` trait implemented once per payload to record its canonical type. Do not continue allowing dual
semantic implementations merely to avoid the trait migration.

**ADOPTED (2026-08-02).** The full-scale worktree experiment (`experiment/p8-assoc-type` @ `50e86f964`; evidence in
its `EXPERIMENT_NOTES.md` and `EXPERIMENT_E0284_PROBE.rs`) passed all five gates: trait solving stable, derive output
slightly smaller with the homogeneous-family invariant compiler-enforced, bound spellings 623 → 295 with all 199
trait-disambiguation turbofishes in the experiment's scope (core + macros) deleted, no inference-only wrapper (the 3
universe-dispatch traits used at the time carried real per-universe algorithm bodies; all three were subsequently
removed — see the gate item in the Phase 8 checklist), and ~2,100-line ~92%-mechanical churn splittable along review
seams. Count scopes, to avoid ledger confusion: the experiment deleted 199 turbofishes (core + macros at the
experiment boundary); the live adoption deleted 242 (all crates at the adoption boundary); both counts are
trait-disambiguation turbofishes, distinct from payload-constructor marker spellings such as `AddOperation::<X>::new()`,
which remain and belong to the open projected-helper item (full census in the P8a plan's post-adoption note: 474
lines, ~250 currently necessary, ~212 removable today — dominated by 132 redundant
`ArrayIrOperation::<A>::from(...)` spellings). An earlier prototype's two
failure classes were diagnosed as (1) E0283 from per-instantiation `Operation` impls — eliminated by the blanket-impl
discipline (one `impl<T: Universe> Operation for FooOperation<T>` per payload family) — and (2) E0284 from
supertraits projecting `Self`'s operation type through a context built from `Self` — an independent current-solver
limitation handled by relaxing 4 projecting supertraits, with per-method `where Self: Operation<Type = C::Type>`
clauses restoring the enforcement and `TODO(eaplatanios)` markers to restore the strict supertraits once the
next-generation trait solver stabilizes. The fallback paragraph above is retained for history only. Execution
checklist: `.tasks/plan_p8a_operation_contract_inventory.md` Phase 4.

Adoption landed on the live tree (2026-08-02, Phase 4 review in the P8a plan): 84 files, +1,553/−1,063, all suites
green (1,112 core / 434 XLA / macros incl. new trybuild mismatch coverage), zero new annotations, turbofishes
242 → 0, `ryft-xla` (including the `XlaOperation` dispatcher) migrated with **no** solver errors at all. One
enforcement nuance: the method-level equality clause works on `BatchableOperation::batch`,
`DifferentiableOperation::jvp`, and `ZeroOperationProvider` (on `zero_operation_with_residuals`; placing it on the
`ResidualZeroProvider` supertrait chain instead exploded into E0284), but `InterpretableOperation::interpret` fell
back to plain relaxation — the composite eager dispatcher cannot discharge the clause under the current solver
(E0275/E0284 in every spelling), so its `TODO(eaplatanios)` covers both the supertrait and the method clause.

### Generic typed projection

Use standard `From` and borrowed `TryFrom` implementations for type lifting and projection. Retain one reusable value
projection contract because eager and symbolic values require different borrowed and owned representations. Value
projection must distinguish borrowing from ownership transfer:

```rust,ignore
pub trait ValueProjection<T: Type>: Value {
    type Projected: Value<Type = T>;
    type ProjectedRef<'v>: Typed<Type = T>
    where
        Self: 'v;

    fn from_projected(value: Self::Projected) -> Self;
    fn projected(&self) -> Result<Self::ProjectedRef<'_>, ProgramError>;
    fn into_projected(self) -> Result<Self::Projected, ProgramError>;
}
```

The exact borrowed-view bounds are a prototype decision, but returning an owned `A` from `&ArrayIrValue<A>` is
not allowed. The current reference `Array` stores `values: Vec<Scalar>`, so its `Clone` deep-copies the complete
payload. A projection API that calls `.cloned()` would copy every eager operand on every bind and contradict the
allocation gate.

Before implementing projected contexts, audit the ownership needs of interpretation and transforms and select one of
these evidence-backed designs:

1. borrowed projected inputs through a GAT/reference-view contract, with ownership transfer only through
   `into_projected`;
2. an immutable shared eager payload such as `Arc<[Scalar]>`, making reference-backend `Array::clone` constant-time
   and allocation-free; or
3. a combination in which read-only projection borrows and the already-immutable reference backend also uses shared
   payload storage because `Value: Clone` is pervasive outside projection.

Do not change `Array` storage merely to make the projection prototype compile. Treat shared payload storage as an
independent prerequisite: confirm that no production/reference operation depends on in-place mutation, measure clone
and kernel costs, and land it separately if selected. `Arc<Vec<Scalar>>` is not preferred over `Arc<[Scalar]>` unless
retaining vector capacity has a measured benefit.

Required projection behavior is fixed:

- eager `ArrayIrValue<A>` borrows `A`/`DimensionValue` for read-only projection and transfers ownership without
  copying when consumed;
- captures and tracers project to a checked typed view preserving the original SSA identity;
- partial tracers preserve known/unknown state without cloning concrete values unnecessarily;
- no eager projection deep-copies array payloads or allocates;
- wrong-kind errors use one canonical diagnostic; and
- the contracts work for another sum member without adding another sum-specific projection trait.

A generic zero-state `ProjectedContext<C, T>` adapts an outer context to homogeneous operations of member type `T`.
It may contain only the parent context or a borrow/reference to it. It must not contain dimensions, source arrays,
identity maps, replay substitutions, or other semantic state. Its bind path:

1. lifts typed operands into the parent storage family;
2. lifts the homogeneous inner operation through the outer operation dispatcher;
3. binds directly in the parent context; and
4. projects the results back to the member type.

There is no replay inspection and no reconstruction pass.

### Explicit differentiation residuals

Differentiation owns the primal values needed by a later transpose. Introduce or extend one transform-level residual
contract that can retain ordinary SSA values of any storage kind. For dimension-dependent array operations:

1. linearization identifies required primal dimension values from explicit operation operands or stages an explicit
   `dimension_size` reader while the primal array is available;
2. the values are appended to the linearization's ordinary residual structure;
3. the transpose receives typed dimension residuals explicitly; and
4. inverse shape operations consume those residuals directly.

The contract must cover reshape, concatenate, mean/reductions, slicing, padding, gather, and any future transpose whose
input geometry is not recoverable from its cotangent. It must not store differentiation-only variables in primal
operation payloads, infer residuals from cotangent type expressions, or add tangent/cotangent slots for dimensions.

Delete `ReshapeOperation::transpose_dimension_variables`, its rendering and identity-renaming behavior, and every
reshape-specific composite differentiation branch that exists only to populate or consume that field.

### Explicit dimension-operand signatures

The order of explicit dimension operands is part of each mixed operation's semantic signature. Replace the current
family of ad hoc `runtime_dimension_variables` methods and recovery loops with direct positional contracts. For
reshape and broadcast, the contract is deliberately uniform: one array followed by one dimension value per output
axis. Exact constants represent static axes, bounded non-exact values represent dynamic axes, and inference derives
the complete output shape from the operand types.

Do not introduce a parallel `DimensionOperandSchema` or another metadata language describing operands already present
in SSA. Operation inference is the authoritative declaration of operand count, kind, order, and result construction.
Eager interpretation, transforms, and lowering preserve and consume the same ordered edges.

Later mixed operations may need additional semantic segments, such as slice starts or pad widths. Extend the existing
typed mixed projection cursor only when repeated call-site code justifies a small helper. Keep such helpers structural:
they may project fixed or repeated member kinds and produce canonical count/kind diagnostics, but must not carry a
second copy of dimension identities, bounds, or output shapes.

### Operation families

The final production families are:

```text
ArrayPrimitiveOperation<A> Operation<Type = ArrayType>
DimensionOperation        Operation<Type = DimensionType>
ArrayIrOperation<A>  Operation<Type = ArrayIrType>
```

The names may be simplified after the migration, but the roles are fixed.

- `ArrayPrimitiveOperation` contains operations whose complete signature is array-only.
- `DimensionOperation` contains dimension-only operations.
- `ArrayIrOperation` is the sole stored dispatcher and public array-program operation family. It projects the two
  homogeneous member families and stores cross-kind and higher-order operations as direct flat variants.

Do not introduce a separate mixed-operation family solely to reduce forwarding match arms. Such a family describes
storage organization rather than a semantic operation contract, leaks nested variants into the public API, and adds a
second policy match in transforms and lowering. Reconsider generated forwarding only if it can preserve flat variants,
native eager member interpretation, and composite-operation validation without new public wrappers or special-purpose
derive modes.

Generic nullary constructors remain homogeneous only for output types with no identity *references*, enforced by a
shared position-aware rule in their blanket `Operation<T>` inference (definition-position identities, such as a
`DimensionType`'s own variable, are grounded by the constructed value itself and stay constructible). Array
construction with dynamic axes routes to a composite variant (`DynamicZero`) whose mixed contract is owned by the
composite family's variant arm over the payload's single `Operation<ArrayType>` implementation: the stored
`ArrayType` is the complete output authority (static axes keep stored extents, mixed static/dynamic shapes are
first-class) and each dynamic axis consumes one explicit dimension operand, validated by identity in axis order.
This operand contract is deliberately *narrower* than mixed reshape/broadcast, which derive every output axis from an
operand including exact constants: constructor static axes have no input geometry to relate to, so dynamic-only
operands minimize IR and correspond one-to-one with `stablehlo.set_dimension_size`.

Delete the complete homogeneous `backends::arrays::ArrayOperation` shape-program family. If the shorter name
`ArrayOperation` remains desirable, give it to the array-only primitive family after consumers migrate. Do not keep a
production homogeneous graph containing implicit-shape versions of reshape, broadcast, slice, or constructors.

Array-only test domains may define small local enums. They must not justify retaining a second production array
language.

### Mixed signatures

Keep one typed projection vocabulary for mixed operations. Improve the existing projection cursor only where measured
call-site repetition warrants it. The desired rule body reads like a signature declaration:

```rust,ignore
let [input] = inputs.arrays()?;
let shape = inputs.remaining_dimensions()?;
inputs.finish()?;
```

The vocabulary must support:

- fixed operands;
- homogeneous repeated segments;
- optional operands;
- several named segments;
- heterogeneous regions;
- borrowed projections without temporary vectors when practical; and
- one canonical count/kind diagnostic.

Prefer a small typed cursor extension over a new declarative or procedural schema language. Generate code only when it
deletes more handwritten code than it introduces. The operation declaration is the sole source for outer enum
variants, `From` conversions, type projection/lifting, and transform dispatch classification.

### Transform-owned value-kind policies

Do not add batching or differentiation concepts to `Type`. Each transform owns its policy:

- batching defines how a member kind represents replicated/mapped values;
- differentiation defines its primal/tangent/cotangent space;
- partial evaluation defines known-value representation and abstract transfer behavior;
- transposition defines whether values are linear, structural, or nondifferentiable.

For dimensions:

```text
batching:          replicated only
sharding:          replicated only
forward tangent:   no tangent
reverse cotangent: structural zero / absent
objective:         rejected
residual:          allowed
partial value:     known integer or unknown dimension SSA
```

This table is the tier-2 policy and is unchanged by Phase 12: a data-derived dimension whose gateway operand is
replicated behaves identically to every other replicated dimension. Phase 14 alone may relax `batching: replicated
only` to a ragged mapped representation owned by the batch carrier; no other row changes at any tier, and raggedness
must never appear on `Type`.

Generic outer transform dispatch projects a homogeneous operation and its values, invokes its existing rule through a
zero-state projected context, and lifts results. Mixed operations retain explicit rules because their semantics truly
cross kinds. The resulting design must not need a parallel complete array/dimension implementation of batching or
differentiation.

Batch extent authority must itself be explicit. If batching introduces or reads a dynamic mapped extent, the batching
state carries the corresponding dimension value, not only `Dimension` metadata. Broadcasting a replicated array to a
dynamic mapped size consumes that value directly.

### Structural identity authority

Keep the identity and complete-signature refinement hooks on `Type`; all Ryft types participate in programs, and these
hooks are the appropriate generic boundary. Do not add transform-specific behavior to `Type`.

Delete `OutputIdentityRole` if the structural prototype proves the following algorithm:

1. Formal input definition occurrences establish boundary identities.
2. Instruction result definition occurrences whose identity is already available from an operand are forwards/readers.
3. Other instruction result definition occurrences establish internal identities.
4. Reference occurrences must resolve to an available boundary or internal identity.
5. An internal identity has exactly one establishing definition.
6. Region-carrying results that reuse operand identities are forwarded structurally; results with fresh definition
   identities are producers.

This derives `dimension_size`, condition, while, and scan behavior without an operation override. Capture exact current
diagnostics before changing closure. If any valid operation cannot be represented by these rules, refine the
type-position model rather than restoring operation-specific output-index hooks.

### Module ownership

After behavior stabilizes, place concepts with their semantic owner:

- `types::dimensions`: `Dimension`, `DimensionVariable`, `DimensionBounds`, `DimensionType`, and `Shape`;
- `operations::dimensions`: constants, arithmetic, comparisons, gateways, requirements, and `dimension_size`;
- backend modules: concrete eager `DimensionValue` representation and XLA interpretation/lowering;
- a neutral public dimension API only if the P9 audit finds a capability not already provided by the first-class
  dimension value traits; the removed `RuntimeDimension`/`RuntimeShape` wrappers are not a mandatory destination; and
- array-program storage/dispatch: only the sum value, outer operation family, and generic projection implementations.

Do not perform module moves before the semantic deletions. Late moves keep reviewable diffs and avoid moving code that
will be removed.

### `TypeError` representation

Retain `TypeError` as an enum because `Custom` will carry production type-family errors such as `DimensionError`
through type/program boundaries with typed recovery. Projection failures do not need a separate structured variant:
the archived implementation only needs their rendered message and does not branch on a more specific error category.
Render them through `Invalid { message: String }` using one canonical expected/actual diagnostic.

Use the named field for clear destructuring while routing every construction through one canonical helper:

```rust,ignore
TypeError::invalid(message)
```

The helper owns message conversion and keeps construction sites independent of the variant's storage syntax;
`TypeError::Invalid { message }` remains available for the few tests and consumers that inspect the rendered payload.
Apply this as a separate mechanical cleanup so it does not obscure semantic dimension changes, and verify exact
rendered diagnostics. Do not convert `TypeError` back to a single struct with an optional source, which would erase
typed custom errors and encode another implicit union.

Do not add a reverse `TypeError::from_program` conversion. The archived call sites only need it because type and shape
helpers return the higher-level `ProgramError`; stringifying every non-type program error would erase its structure and
invert error ownership. During each operation migration, make pure type/shape helpers return `TypeError`,
`DimensionError`, or another owner-specific error, split helpers that mix program construction with inference, and
wrap `TypeError` in `ProgramError` only at outward program boundaries.

## Carries-over/deletes ledger

### Carries over

- `DimensionVariable` as identity and bounds authority;
- leaf-only `Dimension`, `DimensionType`, and `ArrayType` shape metadata;
- `ArrayIrType` and `ArrayIrValue<A>` as the storage sum;
- first-class dimension arithmetic and requirement operations;
- interval, congruence, order, equality, and constant abstract interpretation;
- `Effect::OrderedAssertion`;
- the bounded-tensor ABI: one public array leaf lowered to bound-shaped physical storage plus hidden replicated extent
  scalars, logical reconstruction with `set_dimension_size`, hidden output extents, and plugin `PadToStatic`
  legalization;
- existing StableHLO dynamic-shape operands, diagnostic runtime assertion lowering, and per-effect token separation;
- behavioral JAX parity tests;
- exact diagnostics;
- identity alpha-renaming and cache behavior; and
- public first-class dimension ergonomics without reviving expression-era `RuntimeDimension`/`RuntimeShape` wrappers
  unless a concrete missing capability is demonstrated.

### Deletes

- every dual semantic operation implementation listed in the diagnosis;
- the full homogeneous implicit-shape `ArrayOperation` program family;
- `ArrayContextView::with_dimensions`;
- `ArrayContextView::with_source_array`;
- `ArrayContextView::bind_replayed`;
- ambient dimension/source-array fields in context views;
- replay-time `runtime_dimension_variables` classification matches;
- temporary homogeneous programs used solely to recover or append dimension operands;
- `ReshapeOperation::transpose_dimension_variables` and every differentiation-only payload field serving the same
  purpose;
- reshape-specific residual collection/recovery replaced by transform-owned ordinary SSA residuals;
- array/dimension-specific projection traits replaced by generic member projection;
- eager projection paths that clone concrete array payloads;
- independent `runtime_dimension_variables` contracts and copied dimension identity/bounds/order validation loops;
- redundant operation-family variant lists and manual outer `From` boilerplate;
- `OutputIdentityRole` and its operation-specific implementations, subject to the structural prototype gate;
- P1c's temporary result-reference producer fallback after P3 supplies explicit result-dimension operands;
- duplicated transform rules whose only purpose is storage-kind projection/lifting;
- direct `TypeError::Invalid { message: ... }` construction after the canonical `TypeError::invalid(...)` migration;
  named-field destructuring remains supported; and
- compatibility shims for the retired homogeneous shape-program API.

Inference may derive result metadata from explicit operand types. What must disappear is independent per-consumer
collection, copied identity/order validation, duplicated payload shape metadata, and any helper that recovers operands
absent from the staged graph.

## Historical execution staging and review process

### Why this section exists

This section records how the original 142-path archive was made recoverable and how Phases S0–P3c were mined from it.
It is no longer the active execution protocol. The immutable archive still provides evidence and must never move, but:

- the integration branch has advanced to `21c7442fe`;
- the mutable remainder stopped at `12398a196` after P3c;
- the staging worktree remains on the obsolete P3d increment; and
- `.tasks/dimensions_cleanup_ledger.md` stops at P3h even though P3i is committed.

Trying to resume by reconciling the stale remainder would reintroduce superseded code over a much newer tree. From the
2026-07-28 audit onward, resume from the live integration branch, preserve the current dirty P3j work, update this plan
with each review unit, and use ordinary focused commits after owner review. The archive, remainder, and old increment
branches are read-only historical evidence. P11 replaces the former “empty mutable remainder” gate with a current-tree
residual audit plus confirmation that the immutable archive still points at its recorded commit.

The parent refactor exists as one uncommitted working tree containing 112 tracked changes and 29 nonignored untracked
paths, for 141 expanded status entries, plus this ignored plan. It spans unrelated refactors and several generations
of the dimension design. It cannot be reviewed, bisected, or safely resumed in that form, and it has no recovery point.
This section first preserves that tree exactly, then mines only the correct pieces into the target architecture as
small reviewed increments.

The rest of this section is intentionally preserved as the historical procedure that produced the committed
increments. Do not execute it against the current owner checkout.

### Roles, branches, and working areas

**Integration branch** — `u/eaplatanios/dynamic-shapes`, currently at `8105cfd26` and pushed to `origin`. Receives
reviewed increments one at a time and is the project's working base for this effort. `main` is not involved.
Intermediate increments are reviewed and committed directly here without pull requests. Never force-push or rebase it.

**Immutable archive branch** — `u/eaplatanios/archive/dimensions-wip-2026-07-24`. A single verbatim snapshot of the
original uncommitted tree, including this ignored plan. Push it once and never advance, rewrite, or delete it. Its
commit is immutable evidence, not a branch to clean up or merge.

**Mutable remainder branch** — `u/eaplatanios/wip/dimensions-remainder`. Starts at the immutable snapshot. After each
increment lands, merge the integration branch into it and reconcile the corresponding archived scope in favor of the
reviewed implementation or an explicit deletion. This branch answers what remains to be mined from the original work
without changing the archival evidence.

**Increment branches** — `u/eaplatanios/increment/<id>-<slug>`. Each starts from the current remote integration
branch, contains one review unit, and is pushed before owner handoff as a recoverable implementation record. No
intermediate pull request is opened.

**Owner checkout** — `/Users/eaplatanios/Development/Repositories/ryft-1`. Where the owner reviews, and where the
integration branch stays checked out. Source authoring and extraction never happen here. After an increment is pushed
and verified, the executor may stage its merge here for review using the exact protocol below. The owner either commits
that merge directly to `u/eaplatanios/dynamic-shapes` or requests changes.

**Staging worktree** — `../ryft-1-dimensions`. Where all extraction, authoring, and verification happens.
Executor-owned, and may be deleted and recreated at will.

The staging worktree must use its own Cargo target directory, because sharing the owner checkout's `target/` causes
build-lock contention and cache thrashing. Export
`CARGO_TARGET_DIR=/Users/eaplatanios/Development/Repositories/ryft-1-dimensions-target` in the staging worktree and
accept one cold build.

### Prohibited and sanctioned commands

Never run any of the following in the **owner checkout**, at any time, for any reason:

- `git stash` in any form, including `git stash create`;
- `git reset` on tracked files;
- `git checkout <path>` or `git restore <path>`;
- `git clean`;
- direct source authoring or mechanical rewrites after bootstrap;
- `cargo fmt --all` as part of a semantic increment; and
- any force push to the integration, archive, remainder, or increment branches.

The rationale is recorded in `AGENTS.md`: large refactors live uncommitted for hours, and these commands silently
destroy that work with no recovery path.

Path-level restore is sanctioned **only inside the staging worktree, and only after bootstrap has completed**, because
at that point every line of the original work is committed and pushed, so nothing is destroyable:

```bash
git restore --source=origin/u/eaplatanios/archive/dimensions-wip-2026-07-24 -- <explicit-paths>
```

The staging worktree must be clean before this command. The source must be the immutable archive or a reviewed
integration ref, and the targets must be the explicit paths recorded for the current increment. Never target the
worktree root, a directory broader than the increment, a glob, or an unresolved variable. Restore an entire path only
when its complete archived delta belongs to the current increment. Use `git restore --patch` or an explicitly reviewed
manual reconciliation when the path also contains later archived work. If the staging worktree holds uncommitted work,
commit and push it before restoring anything. The narrow exception added to `AGENTS.md` permits exactly this workflow
and does not authorize `checkout`, `reset`, `clean`, or `stash`.

### Bootstrap: make the current work indestructible

Run once, in order, in the owner checkout, with the owner present. Do not proceed past a failed verification.

- [x] Confirm the starting state exactly: the branch is `u/eaplatanios/dynamic-shapes`; `git rev-parse HEAD` reports
      `8105cfd26817ab728bb2799c889021f240345993`; and `git status --porcelain=v1 -uall` reports 112 tracked changes
      plus 29 nonignored untracked paths, for 141 entries. If any value differs, stop and record the new baseline
      rather than assuming the drafted counts remain valid.
- [x] Record the complete intended snapshot manifest before staging. `.tasks/` is ignored, so add this plan explicitly:

      ```bash
      {
        git diff --name-only HEAD
        git ls-files --others --exclude-standard
        printf '%s\n' .tasks/plan_symbolic_dimensions_architecture_cleanup.md
      } | sort -u > /tmp/ryft-dimensions-bootstrap-manifest.txt
      ```

- [x] Create the immutable archive branch and commit the tree verbatim:

      ```bash
      git switch -c u/eaplatanios/archive/dimensions-wip-2026-07-24
      git add -A
      git add -f .tasks/plan_symbolic_dimensions_architecture_cleanup.md
      git commit -m "Archive full symbolic-dimensions working tree"
      ```

- [x] Verify `git status --porcelain=v1 -uall` is empty. Build the committed manifest with
      `git diff-tree --no-commit-id --name-only -r HEAD | sort`, compare it byte-for-byte with
      `/tmp/ryft-dimensions-bootstrap-manifest.txt`, and record the commit and counts. Do not substitute
      `git show --stat`, whose summary is not an exact manifest check.
- [x] Push the immutable archive:

      ```bash
      git push -u origin u/eaplatanios/archive/dimensions-wip-2026-07-24
      ```

- [x] Create and push the mutable remainder branch from that exact archive commit:

      ```bash
      git switch -c u/eaplatanios/wip/dimensions-remainder
      git push -u origin u/eaplatanios/wip/dimensions-remainder
      ```

- [x] Return the owner checkout to `u/eaplatanios/dynamic-shapes`, then create the staging worktree on the remainder
      branch:

      ```bash
      git switch u/eaplatanios/dynamic-shapes
      git worktree add ../ryft-1-dimensions u/eaplatanios/wip/dimensions-remainder
      ```

- [x] From the staging worktree, run `S0`: restore this plan from the immutable archive, create
      `.tasks/dimensions_cleanup_ledger.md`, and apply only the narrow staging-worktree `git restore` exception to the
      integration version of `AGENTS.md` rather than restoring the archive's unrelated `AGENTS.md` edits. Force-add
      both ignored `.tasks` files and take the increment through the complete no-PR staged-review workflow. This
      rehearses recovery, verification, handoff, direct integration, and remainder reconciliation before code moves.
- [x] Gate: the owner checkout is clean on the integration branch; the immutable archive and mutable remainder exist
      on `origin`; the immutable archive manifest matches the captured manifest; the staging worktree compiles; and
      `S0` has landed with the plan and ledger tracked on `u/eaplatanios/dynamic-shapes`.

### Increment catalog

Increments land in this order. `S*` increments recover independent, already-correct work from the archive. `P*`
increments build the target dimension architecture deliberately. The monolithic archived dimension implementation is
never landed as `S6`; it is a reference and source of tests, not the architectural baseline.

| ID     | Scope                                                                     | Review method             |
| ------ | ------------------------------------------------------------------------- | ------------------------- |
| `S0`   | This plan, delivery ledger, and narrow `AGENTS.md` restore exception        | Line by line              |
| `B0`   | Repair the incomplete custom-derivatives module move                       | Line by line              |
| `B1`   | Rename the public array/data type modules                                  | Pattern and residual      |
| `S1`   | `RegionRef::with_id` helper and call-site cleanup                           | Line by line              |
| `S4`   | Structured `TypeError` core, then mechanical `invalid(...)` call sites       | Core plus sampled sites  |
| `S5a`  | Pure `Size` to `Dimension` rename and `Shape` module move, semantics intact | Pattern, residual, sample |
| `P0`   | Behavioral/evidence freeze and archive implementation disposition          | Evidence review           |
| `P1`   | Leaf identity, bounds, refinements, structural closure, alpha-equivalent cache identity | Line by line     |
| `P2a`  | Homogeneous dimension SSA values and checked arithmetic                     | Line by line              |
| `P2b`  | Ordered dimension requirements and partial-evaluation behavior              | Line by line              |
| `P2b.1` | Canonical dimension operation modules and capability APIs                  | Line by line              |
| `P2b.2` | Restore backend-neutral eager dimension interpretation                    | Line by line              |
| `P2c`  | Generic storage-sum type/value projection                                  | Line by line              |
| `P2d`  | Zero-state projected binding and third-member extensibility gate           | Line by line              |
| `P3.*` | Explicit extent operands, constructors, and one increment per mixed shape operation | Line by line       |
| `P4`   | Control flow, partial evaluation, import, and higher-order composition       | Line by line              |
| `P5`   | Batching and replicated dimension authority                               | Line by line              |
| `P6`   | Differentiation, transposition, and ordinary dimension residuals            | Line by line              |
| `P7`   | XLA lowering, bounded ABI, eager execution, and ordered assertions          | Line by line              |
| `P8`   | Late compiler-enforced `Operation::Type` migration                         | Line by line              |
| `P9`   | Final module and public API placement                                      | Line by line              |
| `P10`  | Persistence and measured performance closure                              | Evidence review           |
| `P11`  | Deletion, minimality, and empty-remainder proof                            | Final architecture review |

Each increment depends on the one above it. `B0` reconstructs the physical custom-derivatives move from the last
compiling pre-move implementation, updates every direct module path, and excludes the symbolic-dimension semantics
accidentally mixed into the original move commit. `B1` is a pure module/file rename from `types::array_types` and
`types::data_types` to `types::arrays` and `types::data`: update every in-repo reference directly, add no compatibility
re-exports, and preserve all public items and behavior. `S5a` renames `Size` to `Dimension` and moves `Shape` and
`StaticShape` into `types::dimensions` without changing representation or behavior.

`B0` and `S1` are refactors unrelated to dimensions that were present only because they were in flight concurrently.
They land first because they are bounded, reduce the archived remainder, and provide low-risk rehearsals of the
workflow.

`S1` is not a global rename of `Program::region_ref` or `ProgramBuilder::region_ref`. It introduces
`RegionRef::with_id` as the one way to select another root from an existing borrowed arena and replaces only
`RegionRef::new(existing_ref.regions(), id)` reconstruction sites. Initial arena entry through `Program` and
`ProgramBuilder` remains named `region_ref`.

`S4` has two review strata in one coherent increment: review the semantic enum core and typed conversions line by line,
then review the uniform `TypeError::invalid(...)` call-site rewrite by transformation command, empty constructor
residual search, and representative samples. `Custom` is semantic structure rather than mechanical noise; a later
dimension-owned conversion will carry `DimensionError` through it without adding dimension knowledge here. Wrong-kind
projection failures remain canonical `Invalid { message }` diagnostics because no production consumer needs to
recover separate expected/actual fields. S4 also owns the explicit `Result<_, TypeError>` annotations at elementwise
inference macro boundaries: the structured error's additional conversion paths make those annotations necessary to
avoid inference ambiguity.

`S5a` is strictly mechanical. Its dynamic variant must retain the pre-refactor `Option<usize>` semantics. Introducing
`DimensionVariable`, authoritative bounds, identity, or refinements belongs to `P1`; mixing those into `S5a` would
disguise a semantic foundation change as a rename.

Split any increment that exceeds roughly 800 substantive lines or crosses concerns that can compile independently.
Split `P3` per operation and `P6` per transform/residual seam. The review budget, not implementation convenience,
sets the boundary. The archived implementation may be consulted for behavior, tests, and proven backend details, but
never copied wholesale: every extracted path must be reconciled against this plan and stripped of known warts such as
`bind_replayed`, `transpose_dimension_variables`, dual contracts, and ad hoc `runtime_dimension_variables`.

### Per-increment workflow

Execute every increment, `S*` and `P*` alike, with these steps.

1. In the staging worktree, fetch `origin`, verify a clean status, and create
   `u/eaplatanios/increment/<id>-<slug>` from `origin/u/eaplatanios/dynamic-shapes`.
2. Add a ledger entry with status `in progress` before writing code. For `S0`, create the ledger and force-add both
   ignored `.tasks` files.
3. Author the change. For `S*`, restore only the explicit scoped paths from
   `origin/u/eaplatanios/archive/dimensions-wip-2026-07-24`, then remove content belonging to later increments. For
   `P*`, implement the target architecture directly, selectively porting only reviewed behavior and tests. Record
   whether each path is wholly owned by this increment or shares archived hunks with later increments.
4. Verify, scoped to the crates touched, with an explicit per-command timeout of 300 seconds unless the owner approves
   longer. At minimum: `cargo check -p <crate>`, then `cargo test -p <crate> --lib`, then the doctests for any crate
   whose public API moved. Add `ryft-macros` and `ryft-macros-tests` whenever a core trait consumed by derive macros
   changed. Add `ryft-xla` whenever lowering, domains, or operation families changed. `ryft-ndarray` no longer exists;
   do not include it.
5. Run `cargo fmt -p <crate>` for the touched crates only, then `git diff --check`.
6. Run the increment's residual search and paste its output into the handoff. An increment that claims a rename,
   deletion, or sweep is not complete until a targeted search proves no stale references remain, with every remaining
   match classified as migrated, intentionally retained with a stated reason, or out of scope.
7. Update the ledger entry to `ready for review` with verification evidence, residual search, deliberate deferrals,
   and review method; then commit and push the increment branch. Report the resulting source commit in the handoff, but
   do not try to write a commit's own SHA into itself.
8. Fetch immediately before handoff and require this check to succeed:

   ```bash
   git merge-base --is-ancestor \
     origin/u/eaplatanios/dynamic-shapes \
     origin/u/eaplatanios/increment/<id>-<slug>
   ```

   If integration advanced, merge it into the increment in the staging worktree, resolve and reverify there, and push
   before touching the owner checkout. Then, in the clean owner checkout on `u/eaplatanios/dynamic-shapes`, fetch
   `origin`, verify there is no existing merge state or local change, and stage the exact reviewed integration result:

   ```bash
   git merge --no-commit --no-ff origin/u/eaplatanios/increment/<id>-<slug>
   ```

   Do not create the merge commit. Present `git status`, `git diff --cached --stat`, the staged diff, and the handoff
   evidence to the owner. The owner reviews and commits the merge directly, or requests changes. There is no PR.
9. Apply requested edits in the staging worktree, rerun verification, commit, and push them to the increment branch.
   Never silently abort, reset, or overwrite the in-review staged merge. The owner must first commit or explicitly
   clear that review state; only after the owner confirms a clean checkout may the executor stage the updated merge.
10. After the owner commits and pushes the merge, fetch in the staging worktree, switch to
    `u/eaplatanios/wip/dimensions-remainder`, and run
    `git merge --no-commit --no-ff origin/u/eaplatanios/dynamic-shapes`. Resolve conflicts. For wholly owned paths, run
    `git restore --source=origin/u/eaplatanios/dynamic-shapes --staged --worktree -- <explicit-scoped-paths>`. For
    shared paths, reconcile only the reviewed hunks with `git restore --patch` or a manually reviewed patch, retaining
    and recording every later archived hunk. Commit the merge/reconciliation together and push the remainder branch.
    Never merge into or advance the immutable archive.
11. The next increment updates the preceding ledger entry to `landed` with its source commit, integration commit, and
    remainder reconciliation commit, then adds its own `in progress` entry. `P11` includes a final bookkeeping
    increment so the last substantive entry is also closed.

An `S*` slice may fail to compile on its own, because the original tree was never written as a sequence. When that
happens, do not weaken verification and do not hand over a red increment. Choose one of these and record the choice and
its reason in the ledger:

- widen the increment by the smallest set of additional paths that makes it compile, keeping the review method
  unchanged;
- author the small bridging change yourself so the slice stands alone, which is preferred when the bridge is a few
  lines and semantically obvious; or
- fold the slice into the next increment and mark its own entry `abandoned`, superseded by that increment.

Never reorder increments to dodge this without recording why, because the catalog's order is what keeps the extraction
tractable.

### Delivery ledger

Create `.tasks/dimensions_cleanup_ledger.md` and append one entry per increment. The ledger records delivery; the
checkboxes in this plan record design progress. Both must be kept current, and the remainder accounting below proves
that no archived work silently disappears.

```markdown
## <ID>: <short title>

- Status: in progress | ready for review | landed | abandoned
- Branch: u/eaplatanios/increment/<id>-<slug>
- Source commit: <pushed increment commit, filled by the next increment after landing>
- Integration commit: <direct merge sha, once landed>
- Remainder reconciliation commit: <sha, once reconciled>
- Immutable archive unchanged: yes | no
- Landed: <what actually changed, two or three sentences>
- Deferred: <what was intentionally left out, and to which increment>
- Verification: <exact commands and results, e.g. `cargo test -p ryft-core --lib` → 937 passed>
- Residual search: <command> → <output or "empty">
- Next action: <the single next thing to do, written so a new engineer can start from it>
```

Never delete a ledger entry. Mark abandoned work `abandoned` with the reason.

### Progress accounting

The remaining archived work is the diff between the remote integration and mutable remainder branches. It shrinks as
increments land and are reconciled. The immutable archive never moves and therefore remains a stable audit source:

```bash
git fetch origin
git diff --stat \
  origin/u/eaplatanios/dynamic-shapes...origin/u/eaplatanios/wip/dimensions-remainder \
  | tail -1
```

Record this result after every reconciliation. It need not shrink by a fixed number because fresh target-architecture
work may differ from the archive, but every remaining path must have a ledger disposition: pending extraction,
superseded by a named increment, or deliberately dropped with rationale. `P11` requires no unexplained remainder.

### Resumption protocol

Follow this exactly when picking up work started by someone else, or after any interruption. Determine state from the
repository, never from memory or from a previous session's claims.

1. Read `.tasks/dimensions_cleanup_ledger.md` bottom-up. The last entry's `Next action` is the starting point.
2. Read this plan's phase checkboxes to see which design items the ledger claims are satisfied. Where the two
   disagree, the repository wins: verify with a targeted search before trusting either.
3. Establish branch reality: `git fetch origin`, then `git branch -vv`, then
   `git log --oneline origin/u/eaplatanios/dynamic-shapes -15`. Confirm both archive and remainder branches exist on
   `origin`, and verify the immutable archive still points to its recorded bootstrap commit.
4. Detect in-flight increments with `git branch --list 'u/eaplatanios/increment/*'`, inspect the owner checkout for a
   staged merge, and inspect `git status --porcelain=v1 -uall` in the staging worktree.
   A non-empty status means someone stopped mid-increment. Commit it to a scratch branch before doing anything else;
   do not restore, reset, or clean over it.
5. Compute the remaining diff with the progress-accounting command above and compare it to the last ledger entry.
6. Re-run the last increment's verification commands yourself. Do not trust a recorded pass; a concurrent edit in
   another crate may have invalidated it.
7. If a crate you depend on is transiently broken by another agent's uncommitted edits, verify your own change in
   isolation — for example, a throwaway crate outside the workspace depending only on the crates you changed. Never
   stash, reset, or check out tracked paths to work around it.
8. Write a new ledger entry describing what you found before resuming, especially if it contradicts the previous
   entry.

### Rules for executors, including subagents

Subagents do not inherit this repository's conventions. Any delegated extraction, authoring, or verification task must
restate these constraints in its own prompt:

- author only in the staging worktree; the primary executor alone may stage the documented no-commit merge in the
  clean owner checkout for review;
- never run `git stash`, `git reset`, `git checkout <path>`, `git restore <path>` in the owner checkout, or `git clean`
  anywhere;
- never run `cargo fmt --all`, and never format crates outside the increment;
- never commit the integration merge or advance the immutable archive;
- never mark a plan item or ledger item complete because a test passed; record the implementation or deletion that
  satisfies it;
- if a Rust verification command causes `rustc` to be killed or to grow to extreme memory, stop rerunning broad checks
  and first reduce the generic surface the increment introduced; and
- if an increment turns out to require a design decision this plan does not answer, stop and escalate rather than
  choosing one silently.

### Resolved sequencing and contract decisions

- Build and simplify the semantic target architecture first. Move the associated-type `Operation::Type` enforcement
  late, after dual semantic contracts, implicit replay, and mixed constructor overlap have already been removed. The
  trait migration then enforces a proven shape instead of determining it.
- Keep the strict one-payload/one-contract invariant. Parameterize `SelectOperation`, `StopGradientOperation`, and
  `TestNullaryOperation` by their type so each concrete instantiation has one contract. Three localized type parameters
  are an acceptable cost; they are not evidence of pervasive wrapper churn.
- Never land the archived monolithic feature and clean it afterward. Mine correct tests and implementations into the
  phase that owns them, and record every archived path as landed, superseded, or dropped.

## Execution phases

Execute these specifications in order: `P0`, `P1`, `P2`, `P3.*`, `P4`, `P5`, `P6`, `P7`, `P8`, `P9`, `P10`, and
`P11`. The current resumption unit is the P3j boundary correction and shaped-zero gate. Split later work into
review-sized vertical slices recorded directly in this plan; do not revive the obsolete increment branches or treat
the stale delivery ledger as authoritative.

### Phase 0: freeze evidence and classify the migration

The first release-baseline build found one feature-gated S1 call site that normal core checks did not compile:
`tracing_v2::benchmarking` invoked `RegionRef::region_ref(...)` after that borrowed traversal API was renamed to
`RegionRef::with_id(...)`. P0 owns the one-line correction because the benchmark emitter cannot serve as baseline
evidence until it compiles. This is a compatibility correction to the already-reviewed S1 seam, not dimension
semantics; the P0 release build is its regression gate.

- [x] Record the exact source revision, toolchain, machine, and feature configuration. After bootstrap there is no
      dirty state to fingerprint: behavioral and code-size baselines come from the immutable archive commit. Also
      report the final net feature cost relative to the reviewed integration branch after `S5a`, but do not require the
      feature-bearing result to be smaller than the branch before the feature existed.
- [x] Create an archive-disposition table covering every archived changed path: independent extraction, behavioral/test
      source for a named `P*` increment, explicitly superseded design, or deliberate deletion. No path may remain
      unclassified merely because the archive happened to compile.
- [x] Capture the code-size, occurrence-count, operation-family, trait-implementation, compile-time, memory, generated
      code, graph-size, allocation, runtime, and diagnostics baselines listed above.
- [x] Separate production lines from unit/integration tests and generated code.
- [x] Inventory every `Operation<ArrayType>` and `Operation<ArrayIrType>` implementation and classify it as
      array-only, dimension-only, mixed, region-polymorphic, or erroneous dual contract.
- [x] Separately inventory blanket generic constructor implementations that overlap array-program-specific
      instantiations: zero, one, fill, and iota. Assign each static, dynamic, public-API, transform, and lowering use to
      the homogeneous constructor, operand-relative operation, or variant-owned mixed stored-type contract.
- [x] Inventory every use of the complete homogeneous `ArrayOperation` outside tests and assign its migration target.
- [x] Inventory every `ArrayContextView`/`DimensionContextView` construction and state why the caller needs a view.
- [x] For every `with_dimensions`, `with_source_array`, and `bind_replayed` path, record the explicit SSA dependency
      that must replace it.
- [x] Inventory every batching, differentiation, transposition, and partial-evaluation special case in
      `backends/array_programs`.
- [x] Inventory every transpose that needs primal dimension values unavailable from the cotangent and record the
      explicit SSA residuals it requires. Include reshape, concatenate, mean/reductions, slice, pad, and gather.
- [x] Inventory every independent `runtime_dimension_variables` collector and every copied dimension-operand
      validation loop; assign each operation a direct positional operand contract before the sweep.
- [x] Audit eager backend clone cost, beginning with reference `Array`'s `Vec<Scalar>` payload, and identify every
      current or proposed projection that clones an eager value.
- [x] Decide the projection ownership model—borrowed views, immutable shared eager payloads, or both—before the
      Phase 2 prototype. Record allocation/latency measurements and land any reference-array storage change separately.
- [x] Capture exact rendered IR and diagnostics for the existing static, direct-dynamic, derived-dynamic,
      control-flow, batching, differentiation, and assertion golden programs.
- [x] Add a code-ownership map for core types, operations, transforms, public API, reference eager values, and XLA.
- [x] Record a migration table with one row per affected operation and columns for canonical signature, eager rule,
      tracing, PE, batching, JVP, VJP, transpose, regions, lowering, tests, and old-code deletion.
- [x] Gate: do not begin the sweep until every dual-contract operation and every hidden reconstruction path has an
      explicit destination.

### Phase 1: introduce leaf identity and derive ownership structurally

P1 is split at compile-safe semantic boundaries so each increment remains reviewable:

- `P1a` introduces the local `DimensionBounds` and `DimensionVariable` foundations, including fresh identity,
  authoritative immutable bounds, diagnostics, equality, hashing, and focused tests. It does not change
  `Dimension::Dynamic` yet.
- `P1b` changes `Dimension::Dynamic` from `Option<usize>` to one `DimensionVariable` leaf and migrates shape/type
  consumers directly, without a compatibility variant or expression representation.
- `P1c` adds the minimal generic `Type::Identity`/`Type::Refinements` contract, implements structural closure,
  instantiation, renaming, and alpha-equivalent cache matching, then deletes `OutputIdentityRole`.

- [x] `P1a`: add validated inclusive-lower/exclusive-upper bounds with exact diagnostics.
- [x] `P1a`: add fresh identity semantics whose clone preserves identity and whose name is diagnostic-only.
- [x] `P1a`: keep bounds owned only by the identity and immutable after construction.
- [x] `P1a` gate: focused tests cover bounds, display, clone/fresh equality, hashing, and typed error recovery without
      changing existing `Dimension` behavior.

- [x] Replace the mechanically renamed `Dimension::Dynamic(Option<usize>)` with the reviewed leaf-only dynamic form:
      one `DimensionVariable` identity plus authoritative bounds, with no arithmetic expression in types.
- [x] Establish one source of truth for bounds. Types carry the authoritative bounds used for checking and compilation;
      public declaration helpers construct those types rather than maintaining an independently mutable copy.
- [x] Add only the generic `Type::Identity` and `Type::Refinements` hooks needed for program closure, alpha-renaming,
      instantiation, and alpha-equivalent cache matching. Do not put batching, differentiation, or dimension-specific
      behavior on `Type`.
- [x] Capture exact current identity closure, dominance, forwarding, import, and cache diagnostics.
- [x] Implement the structural producer/forwarder algorithm behind a focused internal prototype.
- [x] Cover repeated `dimension_size` readers, fresh dimension arithmetic results, shared outputs, condition forwarding,
      while carries, scan stacked outputs, captures, shared regions, and alpha-equivalent imports.
- [x] Delete `OutputIdentityRole` from the operation trait, derive macro, box delegation, builders, and operation
      payloads if all valid cases pass.
- [x] Ensure definition/reference positions remain owned by type families and boundary/internal classification remains
      owned by region closure.
- [x] Replace repeated linear membership scans only if profiling shows closure cost is material; do not require `Hash`
      on identities without a real consumer.
- [x] Remove avoidable temporary array/dimension refinement vectors with one-pass validation where it reduces
      allocations without obscuring diagnostics.
- [x] Gate: cache identity, permutation behavior, and exact diagnostics match or exceed the baseline with no
      operation-specific identity-source hook.

### Phase 2: introduce generic member projection and direct binding

Phase 2 is split at compile-safe boundaries to stay within the review budget. `P2a` introduces and verifies
homogeneous dimension SSA values and checked arithmetic without depending on the heterogeneous storage sum. `P2b`
adds ordered requirements, static proof behavior, and partial-evaluation placement. `P2c` introduces the storage sum
and generic borrowed/consuming member projection. `P2d` adds direct zero-state projected binding and completes the
vertical and third-member extensibility gates.

At the owner's request, `P2b.1` moves the now-stable arithmetic and requirement primitives into canonical
`operations::dimensions` submodules and introduces their user-facing capability traits before P2c. Arithmetic uses
one nominal payload and capability per primitive, with shared behavior centralized by
`ArithmeticDimensionOperation`; this flattens stored-program dispatch to the outer operation-family selection without
duplicating inference or identity plumbing. This advances only the primitive-operation ownership portion of Phase 9:
concrete host values and the reference backend's closed operation family remain under `backends`, while neutral
public dimension ergonomics remain a P9 capability/API audit. The absent expression-era
`RuntimeDimension`/`RuntimeShape` wrappers are not a required destination.

`P2b.2` is a narrow ownership correction to the final P2b.1 macro follow-up. Dimension operations follow the same
capability-based interpretation architecture as scalar and array operations: each operation owns a generic
`InterpretableOperation` implementation constrained by its value capability, while concrete backends own the
capability implementations. Operation-aware capability methods receive the nominal operation so eager values preserve
its fresh result identity. `DimensionValue` uses the same constant-only dispatch/rich execution-domain split as
`Scalar` and `Array`, keeping generic context-carrying implementations disjoint from its concrete eager
implementations. No generic operation generation names the reference backend's `DimensionValue`, and no parallel
checked-evaluation hook or backend-owned interpretation adapter is introduced.

- [x] P2a: introduce homogeneous `DimensionType`/`DimensionValue` SSA, generic constant reuse, checked bounded
      arithmetic, eager host execution, tracing, and ordinary partial evaluation without projection machinery.
- [x] P2b: introduce equality, less-than-or-equal, positive-divisibility, and explicit-bounds requirements as one
      homogeneous dimension-operation payload. Keep the payload tagged because every predicate has the same semantic
      contract and transformation behavior; distinct nominal operation types would only multiply dispatch and trait
      implementations.
- [x] P2b: classify each requirement from exact, shared-identity, and interval facts as proven, disproven, or
      inconclusive. Proven requirements are pure and erasable, disproven requirements return the owning
      `DimensionError`, and inconclusive requirements carry `Effect::OrderedAssertion`.
- [x] P2b: preserve source order and DCE survival for inconclusive requirements, fold all-known requirements on the
      known side, retain any unknown-side requirement exactly once in the residual program, and include named observed
      values in eager/known-side failures.
- [x] P2b gate: focused tests pin all proof outcomes, exact diagnostics, rendering, eager interpretation, simplification,
      partial-evaluation placement, and deterministic first-failure order. Graph-wide entailment across preceding
      requirements and nested regions remains assigned to P4.
- [x] P2b.1: move dimension primitives into canonical operation modules, replace tagged arithmetic with nine nominal
      payloads, and centralize their shared contract in `ArithmeticDimensionOperation`.
- [x] P2b.2: remove the concrete backend dependency from generic arithmetic operation generation, make operations own
      capability-constrained interpretation, and make backends own concrete capability implementations.
- [x] Introduce the array/dimension storage sum only at atom/region interfaces and genuinely mixed operations.
- [x] Integrate inconclusive requirements with the existing effects model as `Effect::OrderedAssertion`; specify
      ordering, DCE survival, known-side PE folding, runtime observation values, and diagnostic ownership before
      lowering.
- [x] Complete the Phase 0 projection-ownership decision before writing the generic projection trait.
- [x] Implement standard `From`/borrowed `TryFrom` type conversions and `ValueProjection<T>` for `ArrayType` and
      `DimensionType` members of the storage sum.
- [x] Provide distinct borrowed projection and consuming ownership-transfer paths; do not implement eager projection
      as `.cloned()` from a borrowed storage-sum value.
- [x] Implement eager, capture, tracer, partial-tracer, and differentiation-tracer projections. Composite batching
      projection remains assigned to P5 because the current `BatchingTracer` is intentionally array-only; introducing
      its heterogeneous batch representation here would prematurely encode P5's replicated-dimension policy.
- [x] Replace duplicated projected-value wrappers with one generic owned projected value and one borrowed projected
      view where the concrete eager value cannot be returned directly.
- [x] Introduce a zero-state `ProjectedContext<C, T>` that binds homogeneous inner operations directly into the outer
      graph.
- [x] Add one generic inner-operation lift contract implemented by the outer operation family.
- [x] Preserve SSA atom identity exactly through staged projections.
- [x] Preserve concrete eager values without boxing or heap allocation.
- [x] Add allocation and payload-size tests proving that projecting a large reference-backend array neither allocates
      nor copies its `Scalar` payload.
- [x] Pin canonical wrong-kind diagnostics as compile/runtime goldens. Wrong-count diagnostics belong to P3's mixed
      operation inference because P2c introduces no operand-count projection.
- [x] Add a compile-only toy third member kind to prove that another kind needs projection and policy
      implementations, not changes to generic `Program`, `Context`, capture, tracer, or projected-context machinery.
- [x] Gate: the projected context contains no semantic state other than its parent, and the vertical slice creates no
      implicit dimension dependency.

### Phase 3: establish canonical mixed operation signatures

Phase 3's canonical operation-signature slices are complete through P3k, but its integration gate is not closed.
The remaining unchecked items below are dependency-coupled continuations with explicit Phase 4–6 owners: production
shard-map reachability is required to finish collective parity, the composite XLA graph is required before deleting
the frozen homogeneous operation language, and dimension-aware transform residuals are required before deleting
dynamic zero/one materialization. Phase 4 therefore begins as a prerequisite to those Phase 3 exit gates rather than
as evidence that Phase 3 is complete. These numbered phases are dependency-ordered workstreams, not a strict
waterfall; every deferred checkbox names its owner so the open Phase 3 gate remains visible.

- [x] P3a: introduce the production array-program dispatcher.
- [x] P3b: make `DimensionSizeOperation` the sole canonical `array -> dimension` observation.
- [x] P3c: keep `DimensionToScalarOperation` as the sole explicit `dimension -> scalar-array` conversion.
- [x] P3d: add the checked `rank-0 integer array -> dimension` gateway.
- [x] P3e: reject a dedicated indexed `rank-1 integer array -> dimension` gateway. Extract vector elements with
      ordinary array indexing/slicing and scalarization, then cross the sole `DimensionFromScalarOperation` gateway.
      Add a general checked element-indexing primitive only if existing array operations cannot express a required
      dynamic case; do not encode indexing inside the dimension operation family.
- [x] P3f: extend the canonical `CompareOperation` with
      `(Dimension, Dimension) -> Array(Boolean scalar)`, using an output-parameterized `Compare<Output>` capability
      rather than a dimension-specific operation or comparison vocabulary. Preserve the two dimension operands as
      explicit SSA through eager execution, partial evaluation, replicated-only batching, JVP/import, and direct
      signed StableHLO comparison lowering. P2a and P2b already provide ordinary dimension SSA, constants, arithmetic,
      and requirements.
- [x] P3g Delivery A: retain cross-member primitives as flat `ArrayIrOperation` variants and specify one explicit
      dimension operand per reshape/broadcast output axis. Exact constants represent static axes and inference derives
      the output shape from operand types. The rejected `DimensionOperandSchema`, nested-family prototype, and
      projection-aware-derive analysis are recorded in the P3g plan; none remains in production.
- [x] P3g Deliveries B–D: migrate reshape, migrate broadcast, and close the combined transform/lowering vertical slice
      before freezing their legacy homogeneous contracts for consumer-by-consumer deletion. Delivery B landed at
      `1aeea5329`, Delivery C landed at `7aef33d93`, and Delivery D landed at `a4f2c833`. The combined acceptance
      program preserves dimension arithmetic through eager execution, partial evaluation, batching, JVP, import,
      direct StableHLO lowering, and PJRT compilation/execution. The exact legacy-consumer and deletion manifest is in
      `.tasks/plan_p3g_reshape_broadcast.md`.
- [x] P3g public broadcast API consolidation: make `Broadcast` the sole public program-construction capability, with
      exact and computed first-class extents accepted by the same `BroadcastOperation`; remove
      `BroadcastToDimensions`, the public packed-array `DynamicBroadcast` capability, and all old method names; retain
      one backend-only `backends::arrays::BroadcastKernel` contract for already-concrete eager output types. The frozen
      `LegacyBroadcastOperation` remains hidden only for the homogeneous `ArrayOperation` transform language and its
      XLA lowering/import consumers assigned to Phases 4–9. Phase 4 deleted the obsolete packed-array
      `DynamicBroadcastOperation` after proving the canonical explicit-dimension broadcast superseded it. Exact
      implementation and verification evidence is recorded here and in
      `.tasks/plan_broadcast_api_consolidation.md`.
- [x] P3h Delivery A: give the existing `ConcatenateOperation` a canonical mixed contract with a trailing explicit
      result-extent operand while retaining its unchanged homogeneous contract on the same axis-only payload. Mixed
      inference, eager validation, tracing, partial evaluation, identity instantiation, and import are complete.
- [x] P3h Delivery B: preserve the explicit result extent through batching and differentiation without
      result-dimension recovery. Composite batching aligns mapped array axes while requiring replicated extent
      authority. JVP reuses the same extent SSA value for primal and tangent concatenates; static transpose reuses the
      established slicing pullback and dynamic transpose names the Phase 6 residual requirement.
- [x] P3h Delivery C: lower statically proven explicit-extent concatenate directly and close its measured CPU
      execution and residual
      audits. The dependency correction and review-sized delivery are specified in
      `.tasks/plan_p3h_concatenate.md`.
- [x] P3i: finish the remaining shape-operation sweep. Reuse the existing custom-call, pad, and RNG payloads for the
      three genuinely mixed contracts; retain dynamic slice, JAX-compatible gather, ordinary slice, and reduce as
      array-only operations; and do not introduce the archived slice-scatter payload. The complete classification,
      migration, and deletion gates are specified in `.tasks/plan_p3i_remaining_shape_operations.md`.
- [x] Delete P1c's temporary result-reference producer fallback once every shape-producing operation carries its
      first-class result-dimension operands. After this point, a fresh output reference without an available operand or
      a definition-position occurrence is a closure error.
- [ ] Phase 4–9 deletion gate: delete each frozen homogeneous reshape/broadcast implementation and transform rule as
      its owning consumer migrates. Do not attempt the final zero-residual deletion before the composite public
      capability and transform domains replace the current homogeneous `Reshape`/`Broadcast` implementations.
- [x] Complete and remove the remaining-operation inventory through P3i's explicit mixed migration or array-only
      classification proof: custom call, dynamic slice, gather, pad, reduce, RNG bit generation, slice, and the
      archived slice-scatter proposal.
- [x] P3j dynamic-zero constructor contract. The wrapper-based Delivery A from
      `.tasks/plan_p3j_shaped_constructors.md` (Delivery A) was implemented and then replaced by the
      stored-type-authoritative design recorded in the diagnosis and target-architecture sections, corrected by a
      second review pass (see the constructor revision notes in the Review section). The landed implementation has
      the right variant-owned `DynamicZero(ZeroOperation<ArrayType>)` contract, canonical routing, eager rule,
      replicated batching rule, structural differentiation behavior, and direct bounded XLA lowering. The
      `known_extent` leak is gone; genuine cross-program instantiation, pairwise-distinct XLA operand/axis execution,
      retained specialization, macro integration, and the complete zero verification gate pass.
- [x] P3j boundary-refinement correction: remove `DimensionType::known_extent`,
      `DimensionType::with_known_extent`, all observed-extent logic from `ArrayIrValue::r#type`, and the
      corresponding equality/hash/display/renaming and `DimensionValue` checks. Do not replace them with a new
      `Value`, `Typed`, `Type`, or array-program-specific boundary-evidence abstraction.
- [x] Let a concrete output establish the first refinement for any identity already established by the formal input
      signature as well as for an identity defined internally. This is sound without inspecting runtime payloads:
      structural region closure already rejects an instruction result reference unless that instruction consumes or
      defines the identity, and each mixed eager rule validates or constructs its concrete output from those explicit
      operands. The allocation-free one-vector/input-split representation is implemented.
      `TypeRefinements::validate` is parameterized by the complete identity *slice* (`&[T::Identity]`) rather than the
      `TypeIdentitySignature` container; refinement validation needs authority membership but not the signature's
      input/internal partition.
- [x] Add focused negative tests proving that an already-established output identity without a consumed/defined edge
      fails closure, that a fresh output reference without a consumed/defined edge fails closure, and that two concrete
      outputs for one input-owned identity must agree. Positive non-exact eager/refinement coverage exists for reshape,
      broadcast, and `DynamicZero`.
- [x] Validate every concrete `DynamicZero` extent against the corresponding stored dynamic variable's bounds inside
      the eager rule before allocating the output. Program interpretation does not validate intermediate instruction
      result types, and `EagerContext::bind` does not run inference, so final-boundary refinement alone is insufficient
      when a malformed extent feeds a later instruction or the operation is bound directly. Preserve alpha-renamed
      input identities during program interpretation; the runtime check is the stored bounds plus operand kind/count,
      not literal variable-name equality.
- [x] Add a retained-staging/JIT cache test showing that two calls with the same `DimensionType` identity and bounds but
      different concrete extents reuse one specialization while producing correctly sized outputs. Pin that
      `DimensionType` equality, hashing, display, rendering, and persisted signatures are independent of the observed
      extent. `test_array_program_dynamic_zero_retained_jit_reuses_one_specialization` exercises two concrete extents,
      observes distinct output shapes, and proves one trace/lowering/compilation with one retained-cache hit.
- [x] Add a non-exact `DynamicZero` cross-program instantiation/import test. The test invokes
      `with_instantiated_type_identities`, verifies the renamed payload and atom types, executes the instantiated
      program, and separately covers alpha-renamed boundary interpretation. Keep the transpose test proving the array
      cotangent is ignored while every explicit extent operand receives a structural-zero cotangent.
- [x] Re-audit the pending generic fused-JVP zero-primal reuse and Jacobian validation reordering after the boundary
      correction. Retain each only if an independently named regression test requires it; otherwise remove it as
      constructor-driven global complexity. The UFCS-only `fill` and `iota` residue has been removed. If the Jacobian
      ordering remains temporarily necessary until Phase 6 removes dynamic nullary tangent materialization, document
      that dependency and keep the existing exact non-finite-coordinate diagnostics tests as its gate. The zero-primal
      reuse is independently covered by the shaped-zero JVP regression. The Jacobian reordering now names its Phase 6
      dependency and remains gated by the exact non-finite-coordinate diagnostics tests.
- [x] Strengthen the mixed static/dynamic XLA execution fixture with distinct static and runtime dynamic extents so its
      observed output shape catches axis/operand pairing mistakes rather than only an out-of-bounds operand index. All
      three axis extents are pairwise distinct (`4 x 2 x 3`), so swapping the two dynamic operands changes the logical
      output.
- [x] P3j dynamic-zero gate: `DimensionType` is exactly identity plus bounds; `Typed::r#type` is structural; dynamic
      zero has one
      canonical explicit-operand encoding; reference eager execution, PE, batching, JVP/transpose, import, direct XLA
      lowering, CPU PJRT execution, exact diagnostics, and retained specialization reuse all pass.
- [x] P3j dynamic-one slice: execute `.tasks/plan_p3j_dynamic_one.md` as the next isolated review unit. Mirror zero's
      stored-type-authoritative mixed contract and shared policies without touching fill or iota.
- [x] P3j full-parity fill slice: execute the reviewed `.tasks/plan_p3j_dynamic_fill.md` as the next isolated review
      unit. Match JAX's scalar-SSA `lax.full` and array-broadcasting `jax.numpy.full` behavior by representing every
      rank-positive fill as ordinary broadcast: a rank-zero or broadcast-compatible array SSA value followed by static
      or first-class-dynamic output extents. Materialize host scalar literals with `ConstantOperation`; do not add
      `FillOperation`, `DynamicFill`, hide the fill value in a mixed payload, or fold iota into this review unit.
- [x] P3j dynamic-iota slice: specify and execute the final constructor review unit after the full-parity fill slice is
      reviewed and committed.
- [x] Land `One`, full-parity fill, and `Iota` as separate complete vertical slices after corrected `DynamicZero`.
      One and iota use variant-owned mixed constructor contracts through the shared helper. Fill is deliberately
      different: JAX defines it as conversion plus broadcast, so rank-positive static and dynamic fill reuse the
      canonical constant and broadcast operations and their transforms. All three slices must be complete before the
      Phase 3 gate.
- [x] P3k canonical collectives: retain the existing all-gather, psum-scatter, and all-to-all payloads on one
      homogeneous semantic contract while giving their flat composite variants one positional extent operand per
      result axis. Public composite capabilities derive those operands with exact constants, `dimension_size`, and
      ordinary checked dimension arithmetic at the original operation boundary. Mixed inference, eager execution,
      PE, direct tiled and untiled static and bounded-dynamic batching, JVP, identity instantiation, import, rendering,
      and exact Phase 4/5/6 boundaries are recorded in `.tasks/plan_p3k_collective_dimensions.md`.
- [ ] Phase 4/P3k continuation — full JAX collective parity: add validated `axis_index_groups`, the
      all-gather variance policy if the canonical sharding model can represent it, and `pshuffle`/`pswapaxes` as
      compositions. Complete this alongside Phase 4 shard-map migration so the composite graph, group-aware native
      StableHLO lowering, and bounded-dynamic multi-device fixtures land as one vertical slice rather than adding
      another temporary homogeneous API. The public semantics, shared native lowerers, direct composite binder
      fixture, production composite shard-map reachability, and static two-device execution are complete. The
      checkbox remains open for group-aware production reachability, Phase 7 bounded-dynamic execution, and the final
      current-JAX behavioral/StableHLO comparisons.
- [ ] Phase 6 owner: route transform-generated zero/one values through structural zero or `zero_like`/`one_like`
      whenever an operand supplies geometry.
- [ ] Phase 6 owner: migrate transform consumers that stage `ZeroOperation<ArrayType>` with possibly-dynamic types
      (condition, scan, while, gather, scatter, slice, pad, and differentiation rules) to `zero_like`, structural
      zeros, or explicit extent residuals, with dynamic-shape acceptance tests per consumer. The zero reference guard
      is not complete until this lands.
- [ ] Cross-phase invariant: keep `DimensionType` strictly identity plus bounds throughout all later phases. Concrete
      extents are runtime values and output-refinement observations, never part of `Typed::r#type`, structural
      equality, hashing, rendering, persistence, or cache identity.
- [x] Complete the Phase 3 inventory sweep of shape-changing collectives and every other operation whose result
      metadata references first-class dimension operands. P3i records the remaining-operation classification and P3k
      records the collective classification.
- [x] Give each operation classified as mixed by the Phase 3 inventory one direct positional operand contract and
      migrate its inference, eager rule, transforms, and lowering to preserve those same SSA edges. Explicit later
      transform and production-XLA reachability gates remain assigned above.
- [ ] Phase 4–6 cleanup gate: delete copied dimension operand identity, bounds, and ordering validation after
      inference derives result metadata directly from operand types. Centralize only genuinely repeated count/kind
      projection.
- [ ] Phase 6 owner: replace shape-metadata zero/one materialization inside transforms with structural zero or
      `zero_like`/`one_like` wherever semantics allow.
- [x] Static and dynamic reshape/broadcast invocations bind the same canonical payload with exact constants or dynamic
      dimension SSA operands. Constructors intentionally use static homogeneous encoding versus the variant-owned
      dynamic stored-type contract described above.
- [x] Add a residual search proving no concrete payload in the completed Phase 3 inventory implements materially
      different operation type contracts. P3k's final residual audit records the zero-result search.
- [x] Generic constructors have no overlapping array-program-specific payload implementation. `DynamicZero` is a
      composite variant-arm contract, while the temporary `ZeroOperation<ArrayIrType>` is a different generic
      instantiation restricted to identity-free member types until Phase 6 deletes it.
- [x] No operation consumer independently calls an ad hoc
      `runtime_dimension_variables` contract.
- [ ] Phase 4–6 integration gate: every shape dependency in rendered IR is an operand edge or an explicit
      `dimension_size` instruction.

### Phase 4: remove implicit-shape replay and the parallel array language

Execute the XLA portion as one dependency-ordered migration, not as a second body representation:

1. Change the existing flat `XlaOperation` family, its program constants, and `XlaDomain` values to the composite
   `ArrayIrType`/`ArrayIrValue` contract. Keep the backend enum flat; adapt homogeneous array payloads
   through the canonical typed projection machinery rather than wrapping a stored `ArrayIrOperation`.
2. Retype `ShardMapOperation` and every attached higher-order region to that same composite family while keeping the
   public shard-map boundary array-only. Boundary projection must reject a dimension-valued argument or result with
   the canonical wrong-member diagnostic.
3. Trace the local manual body directly in the composite context so `dimension_size`, dimension arithmetic,
   requirements, and mixed shape-changing operations are ordinary attached-region SSA.
4. Route the attached body through the existing composite lowerer with the entered `CollectiveLoweringState`; then
   delete the homogeneous body replay and the transitional `Legacy*` collective capabilities from production tests.
5. Migrate eager compilation/execution and transform entry points by projecting array-valued public inputs/outputs at
   the boundary. No public XLA array API should expose `ArrayIrValue`; the storage sum is an internal program
   representation.
6. Only after all region and transform tests pass, delete duplicated homogeneous lowering dispatch and narrow or
   remove the old full `ArrayOperation` family according to the remaining consumer ledger.

- [x] P4a delete the obsolete packed integer-array `DynamicBroadcastOperation`/`LegacyDynamicBroadcast` path from the
      core operation and reference-backend families, XLA conversion/lowering, and tests. Canonical
      `BroadcastOperation` with one explicit first-class dimension operand per output axis is now the sole
      runtime-sized broadcast representation. This also removes the final dependency on P1c's fresh-result-reference
      closure fallback.
- [x] P4b production composite XLA cutover: execute
      `.tasks/plan_p4b_production_composite_xla.md` as one semantic review unit. Group homogeneous array and dimension
      payloads behind their canonical projected member families, keep mixed and XLA-owned higher-order operations as
      direct backend variants, and switch the existing program/domain/region/lowering cycle in place. Preserve
      array-only public APIs through projection; do not add a projection-aware derive mode, a stored
      `ArrayIrOperation`, a replay bridge, or a parallel production lowerer.
- [x] P4b1 projected-region foundation: add and verify lossless `Program` member-program unprojection specified by the
      P4b plan. This is a behavior-preserving, core-only prerequisite for importing public array-only regions into the
      production composite graph without instruction replay.
- [x] P4b2 projected capture-leaf classification: introduce `XlaArrayConstant = CaptureReference<ArrayType>` and
      migrate array-member payload, lowering, and fixture uses while keeping `XlaConstant` as the temporary production
      alias. This compile-safe naming boundary prevents the atomic composite constant flip from retyping homogeneous
      array payloads accidentally.
- [x] P4b3 projected capture delegation: make `ProjectedContext` register member runtime values in its composite
      parent's capture table exactly once and project the returned constant. Keep generic projected binding region-free;
      lift complete owned public bodies once before constructing direct composite higher-order drivers during the
      atomic P4b cutover so replay sharing and callee identity remain native.
- [x] P4b4 pre-cutover correctness remediation: execute
      `.tasks/plan_p4b4_pre_cutover_correctness.md` before the atomic composite XLA flip. Fix canonical broadcast
      identity elision, exact rank-positive literal lowering, and fill memory fidelity. Keep the complete enclosing
      lowering-state fix assigned to the production dispatcher in P4b rather than extending the standalone composite
      path.
- [x] P4b4a broadcast correctness: require an identity axis mapping for canonical traced no-op elimination, preserve
      the true identity zero-instruction path, reject dynamically typed concrete targets before eager materialization,
      and cover equal-shape axis permutation in eager and traced execution.
- [x] P4b4b exact rank-positive literal lowering: replace the lossy `f64` bridge with exact element-type-directed
      dense construction, keep one-element rank-positive literals shaped, preserve all materializable storage families
      including low-precision and complex payload bits, and verify representative literals through CPU execution.
- [x] P4b4c fill memory fidelity: construct fill's rank-zero literal in the requested output memory while preserving
      the canonical constant-plus-broadcast decomposition and broadcast's memory-mismatch validation.
- [x] P4b5 higher-order prerequisites: generalize `JitCallOperation` over its enclosing program type and give
      `ShardMapOperation` a composite operation contract that preserves its array-only public boundary with canonical
      member projection.
- [x] P4b6 projected compilation facade prototype: retain public `In`/`Out: Parameterized<ArrayType>`, make internal
      compilation artifacts own `In::To<ArrayIrType>`/`Out::To<ArrayIrType>`, and trace directly into the
      composite domain through checked array-member views. The static and declared-dynamic compile/execute probe
      passed without a new core hook, second domain, replay bridge, or temporary homogeneous top-level program.
- [x] P4b7 mixed control-flow prerequisite: add direct composite condition, while, and scan contracts before the
      production domain flip. Keep predicates Boolean arrays, define variant-aware state/carry/output rules, and do
      not make `ArrayIrType` an `ElementType` solely to route mixed regions through homogeneous contracts.
- [x] P4b8 eager gateway prerequisite and final cutover audit: give concrete XLA arrays checked host
      `dimension_size` and `dimension_from_scalar` capabilities, derive global extents from complete shard metadata,
      and record the domain-owned eager dispatch split required by the atomic cutover. Keep array members on cached
      compilation, dimension members on checked host arithmetic, and `dimension_to_scalar` placement in the active
      domain; do not attach backend state to `DimensionValue` or interpret array members recursively.
- [x] `ArrayContextView` and `DimensionContextView` no longer exist. The remaining `with_dimensions` occurrences are
      the unrelated `ReshapeParameters`/`ReshapeOperation` permutation builder and carry no ambient extent state.
- [x] `with_source_array` no longer exists.
- [x] `bind_replayed` and its operation-classification match no longer exist.
- [x] Ambient dimension and source-array context-view fields no longer exist.
- [x] Delete temporary homogeneous program construction used only to replay shape-carrying rules.
- [x] Narrow the homogeneous operation family to array-only primitives.
- [ ] Phase 5/6 owner: migrate public/reference `EagerContext<Array, ArrayOperation<Array>>` consumers to the canonical
      array-program domain where they need dynamic shape, mixed control-flow, or composite transform functionality.
      Retain the homogeneous reference backend and focused transform fixtures until their composite replacements land;
      they are comparison baselines, not unfinished production-XLA migration.
- [x] Replace production tests that rely on the complete homogeneous backend with canonical array-program tests;
      retain small local homogeneous enums only for focused generic tests.
- [x] Migrate XLA operation conversion and compilation entry points to the sole stored array-program operation family.
- [x] Carry explicit dimension operands through condition, while, scan, custom derivatives, rematerialization, region
      capture/import, and caller/callee requirement composition.
- [x] Make partial evaluation project known dimension integers and retain unknown dimension SSA without reconstruction;
      erase proven requirements, reject disproven requirements with exact diagnostics, and retain inconclusive ordered
      assertions.
- [x] Verify conditional and loop-carried extents, gateway compaction, region forwarding, and alpha-equivalent imports.
- [ ] Phase 8/9 owner: rename the narrowed primitive family only after the old full family is deleted and all residual
      references are classified.
- [x] Gate: targeted searches find no `with_dimensions`, `with_source_array`, `bind_replayed`, ambient replay
      environment, or full homogeneous implicit-shape graph.

### Phase 5: simplify batching around value-kind policy

P3c advances the smallest composite batching implementation needed to make the
`dimension_size -> dimension_to_scalar` vertical slice reachable. Treat the semantics it establishes as durable
(arrays may be mapped or replicated, while first-class dimensions are replicated-only), but do not assume that its
parallel `ArrayIrBatchingContext`, `ArrayIrBatchingTracer`, or exact `ArrayIrBatch` representation is
the final architecture. A projected-array wrapper alone cannot represent dimension members, mixed operations,
region recursion, or a first-class dynamic batching extent, and so merely adding a public `ProjectedArrayBatch` would
rename only part of the problem while introducing another carrier.

- [x] Before expanding the composite operation sweep, inventory the duplicated responsibilities across `ArrayBatch`,
      `BatchingContext`, `BatchingTracer`, `BatchableOperation`, and their P3c array-program counterparts. Classify
      each responsibility as value-kind-neutral, array-specific, or genuinely composite. The exact ledger and
      prototype gate are recorded in `.tasks/plan_p5a_batching_policy_prototype.md`.
- [x] Prototype a transform-owned batching-policy abstraction that can select the batch carrier and batching-extent
      representation for a parent context. The concrete shape is deliberately open, but evaluate a design equivalent
      in power to `BatchingPolicy<C> { type Batch; type Extent; ... }` before committing to the parallel composite
      context/tracer tower. Keep this policy in batching machinery; do not add batching hooks to [`Type`].
- [x] Exercise the prototype with one homogeneous array operation, one dimension operation, `dimension_size`,
      `dimension_to_scalar`, the promoted toy third member kind, and one nested-region operation. Prove that array
      members reuse ordinary `ArrayBatch` semantics, dimension members remain replicated-only, and genuinely mixed
      operations retain explicit rules.
- [x] Reuse [`ValueProjection`] for borrowed and consuming member access. Do not add a separate public
      `ProjectedArrayBatch` unless a concrete residual need remains after the policy prototype, and reject any design
      that clones or allocates eager array payloads merely to project a batch.
- [x] Prefer one generic batching context/tracer over the P3c parallel array-program context/tracer when the prototype
      removes more code than it adds, keeps existing array batching rules unchanged or mechanically adaptable, and
      has neutral trait-solver, compile-time, and allocation behavior. If the policy parameter spreads equivalent or
      greater ceremony through ordinary batching, retain a localized composite adapter, document the evidence, and
      reduce it to the smallest value-kind policy layer rather than forcing the abstraction.
- [x] Generalize the operation batching contract alongside the carrier, context, and driver so that one
      `BatchableOperation` can serve homogeneous and composite batching, then delete
      `ArrayIrBatchableOperation`. Do not generalize the operation trait in isolation: its inputs, outputs,
      active context, and recursive driver must all come from the same transform-owned batching policy. If the
      prototype gate rejects the generic policy because it adds more ceremony than it removes, retain the localized
      composite trait and record that evidence explicitly.
- [x] Gate: the final design has one canonical representation for each necessary batching concept, no wrapper that
      merely renames `ArrayIrBatch`, and no parallel context/tracer tower unless the rejected-policy evidence
      demonstrates that the localized duplication is the simpler implementation.
- [x] Promote the toy composite projection fixtures from the `contexts.rs` unit tests (`ProjectedMemberType`,
      `ProjectedMemberValue`, `ProjectedProgramType`, `ProjectedProgramValue`, `ProjectedProgramOperation`, and the
      `impl_projected_test_member!` macro) into a `pub(crate)` shared test fixture the moment this phase's generic
      dispatch tests become their second consumer. Do not duplicate a synthetic composite universe; the promoted
      fixture is also the vehicle for Phase 6's dispatch tests and the verification matrix's toy third-kind gate.
- [x] Represent a dynamic batching extent with its first-class dimension value, not metadata alone.
- [x] Make the generic outer dispatcher project array primitives, invoke their existing homogeneous batching rule
      through the zero-state context, and lift results. Test the generic projection/lift path against the promoted
      toy composite fixture in addition to the production array-program universe, so the dispatch machinery is proven
      member-kind-agnostic rather than array-specific.
- [x] Handle dimension-only operations with the replicated-only dimension batching policy.
- [x] Reject mapped dimension authority at the boundary with the existing typed diagnostic.
- [x] Generalize the public `Batch` and free `batch` entrypoints over the program type's selected batching policy.
      Composite invocation infers first-class mapped extent authority through `dimension_size`, checks additional or
      explicit extents with ordered requirements, and dynamically materializes requested output axes. Reuse the
      canonical move-or-broadcast helper in composite concatenate as the first operation-rule consumer.
- [x] Keep dedicated rules only for genuinely mixed shape-changing and region-carrying operations. Homogeneous array
      operations now use generated generic projection/delegation/lifting; the remaining explicit rules were audited
      and are mixed, region-carrying, RNG, or custom-call contracts.
- [x] Move mapped-state RNG batching's state-carry-free `scan` into the composite region contract and thread every dynamic
      bits-output extent through that body as replicated first-class shape authority. Delete P3i's exact
      `"requires Phase 5 composite scan-region support"` boundary only after the resulting program remains
      size-independent in the mapped-axis extent and preserves the existing replicated-state diagnostic.
- [x] Centralize explicit dynamic alignment/broadcasting so elementwise rules do not rediscover extents.
      Public output materialization, input-sharding normalization, and concatenate now share the first-class dynamic
      boundary helper. Keep this item open until the generic array-primitive dispatcher and all internal elementwise
      alignment stop using the localized static bridge.
- [x] Remove repeated outer-enum matches that only project/lift batches.
- [ ] Delete dimension/source-array reconstruction in dynamic slice, concatenate, reduce, collectives, RNG, and
      constructors. P5c/P5d close control flow, RNG, constructors, and collectives. The remaining production reads
      are the dynamic broadcast/alignment paths (`array_dimensions` and `DimensionSource::Value`), transform-boundary
      normalization/extent inference, and pad-mask construction in `backends/array_programs/batching.rs`; these still
      serve concatenate/reduce/slice-style projected rules and keep this checkbox open.
- [x] Verify nested `vmap`, mapped arrays with dynamic logical extents, replicated dimension residuals, control flow,
      and all mapped-authority rejection paths.
- [x] P5d — immediately after P5c3: enforce one canonical batching algorithm per semantic contract. Member rules
      reused through typed projection count once; direct composite rules remain necessary when explicit dimension
      operands or the threaded-extent region protocol change the contract. Unprojection now promotes condition,
      while, and scan directly and the nested compatibility arms are deleted. Composite collectives now carry every
      result extent and implement matching/different-axis batching directly without delegating to homogeneous rules.
      Homogeneous control-flow and collective rules remain only for the still-public reference array domain, with
      Phase 6 and Phase 8/9 deletion owners recorded in `.tasks/plan_p5d_batching_rule_consolidation.md`.
- [x] P5d provisional gate: the policy-generic elementwise rule is shared infrastructure and every residual
      homogeneous control-flow consumer has an owner. P5d originally treated homogeneous implicit extents and
      composite explicit extents as different collective contracts; the follow-up audit found that this criterion was
      too weak because both implementations repeat the same axis choreography. P5e supersedes that part of this gate.
- [x] P5e — execute `.tasks/plan_p5e_extent_polymorphic_collective_batching.md` immediately after P5d and before
      Phase 6. Preserve both `ArrayOperation` and `ArrayIrOperation`, but make each collective's batching
      semantics one extent-polymorphic algorithm reached through thin implicit-extent and explicit-extent adapters.
      Do not introduce a whole-program promotion pass, a parallel batching context/tracer, or another operation
      family.
- [x] Freeze committed P5d revision `ef42b00aa` as P5e's comparison boundary: its six changed Rust files contain 1,417
      insertions and 272 deletions relative to committed P5c, split into a net 765 production lines and 380 test lines;
      warm `ryft-core` check time is 2.54 seconds; the core/XLA/macro gates pass 1,033, 405 plus one ignored, and two
      groups of 17 tests respectively. Re-measure the same file set and commands after every prototype.
- [x] Make Ryft's fallible arithmetic capabilities provider-aware where the prototype requires it. `Mul::mul`,
      `Div::div`, and analogous capabilities select their type family's concrete stateless operation through the
      single generic `BinaryOperationFor<T>` contract rather than operation-specific provider traits or a hard-coded
      array operation. `DimensionType` continues selecting checked dimension operations. Add direct checked
      host-extent implementations where coherence permits; do not add `multiply_extents`, `divide_extents`, or a
      parallel extent arithmetic trait that restates Ryft's existing fallible capabilities.
- [x] Keep the private collective policy limited to representation boundaries: obtaining an arithmetic extent view
      when the public boundary extent is a composite value, creating exact constants in the owning domain, materializing
      reshape/broadcast/alignment with explicit extents, and staging requirements that cannot use the existing
      `DimensionRequirement` capability directly. First test whether `BatchingPolicy::Extent` can serve as the
      arithmetic extent unchanged; introduce at most one associated projected extent type only if the composite
      value's `ArrayIrType` wrapper makes that impossible. Do not put collective formulas or per-operation axis
      decisions on the policy.
- [x] Prototype all-gather end to end before touching the other collective rules. Its one shared matching-axis kernel
      must be generic over the policy's fallible extent operations and materialization hooks. The homogeneous adapter
      derives its complete logical result extents from `ArrayType`; the composite adapter validates and projects the
      explicit dimension operands. Non-matching-axis forwarding may retain two small encoding adapters, but the move,
      merge, reshape, variance, group, sharding, and diagnostic decisions must exist once.
- [x] Gate the prototype on lower production lines than the P5d all-gather implementation, no new source-array extent
      reconstruction, no worse diagnostics, stable trait solving, and unchanged static/dynamic/nested behavior. If
      the prototype needs pervasive wrappers, duplicated bounds, or operation-specific policy methods, stop and
      redesign rather than applying it to sum-scatter and all-to-all.
- [x] Apply the accepted pattern to sum-scatter and all-to-all. Share their reduction/split/swap/merge choreography,
      checked multiplication, exact division plus divisibility requirement, output-axis placement, and output
      sharding once across both operation families. Preserve tiled and untiled modes, matching and different named
      axes, replicated and mapped operands, bounded-dynamic extents, and the exact group/variance rejection policy.
- [x] Delete the superseded composite-only collective arithmetic, reshape, alignment, and matching-axis helpers and
      every independently maintained homogeneous/composite algorithm body. Retain only helpers that encode one
      genuinely shared policy boundary or are reused by non-collective operations. Run a source ledger showing the
      single owner of each collective's semantic decisions and every remaining family-specific adapter.
- [x] Audit the dimension-specific arithmetic capability aliases after the provider-aware fallible traits work. Delete
      `DimensionMul`, `DimensionDivFloor`, or analogous capability traits only when the corresponding Ryft `Mul`,
      `Div`, or other fallible trait has identical checked semantics, diagnostics, bounds inference, eager behavior,
      and staging coverage. Keep distinct operations such as saturating subtraction and requirement effects when no
      generic arithmetic capability expresses their semantics.
- [x] Re-run the complete P5d matrix: static and bounded-dynamic tiled/untiled collective batching, replicated and
      mapped inputs, matching and different named axes, nested `vmap`, rendering/import, PE/JVP, exact diagnostics,
      StableHLO lowering, CPU multi-device execution, full `ryft-core`, full `ryft-xla`, macro integration, formatting,
      diff hygiene, and the source gates for no `dimension_size` reconstruction inside collective batching.
- [x] P5e gate: the final tree supports both operation families with one collective batching algorithm per operation;
      introduces no redundant extent algebra; has a strictly negative production-line delta relative to the verified
      P5d boundary; and does not regress warm check time by more than 10% without an explained, measured reason. Record
      the exact deletion ledger and measurements before Phase 6 begins.
- [x] Gate: adding an array-only primitive with a standard batching rule requires no handwritten change in composite
      batching dispatch.

P5a adopted the transform-owned policy design and completed the shared-frame migration. `BatchingContext`,
`BatchingTracer`, `BatchingDriver`, and `BatchableOperation` are now policy-generic; the three parallel
`ArrayIrBatching*` context/tracer/trait concepts are deleted; and composite extents are parent-owned dimension SSA
values. The promoted three-member fixture, ordinary array and dimension rules, mixed gateways, dynamic reshape edge,
mapped-dimension diagnostic, recursive replay, rendering/import, macro integration, full `ryft-core`, and full
`ryft-xla` suites all pass. Clean repeated `ryft-core` check time remained effectively neutral (`12.17-12.26s`
baseline versus `12.28-12.31s` repeated current measurements; one `13.24s` current outlier), and warm incremental time
was `0.17s` versus `0.16s`. Full measurements and the deletion ledger are recorded in
`.tasks/plan_p5a_batching_policy_prototype.md`.

At the close of P5a, the remaining Phase 5 items were intentionally production work rather than P5a misses:
eliminate the localized `static_axis_extent` bridge, prove the generic projection/lift dispatcher, migrate composite
region operations and RNG scan, and run the complete nested/dynamic/JAX-parity matrix before the final Phase 5 gate.

P5b completed the first two items in `.tasks/plan_p5b_dynamic_batch_alignment.md`. Homogeneous array operations now
use one generated policy-generic projection/delegation/lifting path, and static and first-class dynamic batching share
one structural alignment algorithm with mode-selected broadcast materialization. The static bridge, duplicated
composite alignment, and per-primitive outer dispatch are deleted. Focused parity, import/render, macro, core, and XLA
tests pass. Composite region operations and RNG scan remain deliberately separate Phase 5 units.

P5c is complete and recorded in `.tasks/plan_p5c_composite_region_batching.md` as three separately reviewable vertical
slices: the explicit composite threaded-extent boundary plus condition, then while/scan, then mapped-state RNG. The
implementation preserves that separation even though P5c2 and P5c3 were executed together at the user's request.
Structural region batching carries the mapped extent explicitly as dimension SSA, and the source audit found no
region projection, source-array extent reconstruction, host concretization, parallel context/tracer, or XLA-local
copy of the batching semantics.

P5d completed the missing explicit-extent collective capability and correctly established that the elementwise
blanket is already one shared algorithm reached through typed projection. Its follow-up audit found that retaining
homogeneous and composite collective batching as independently maintained algorithms was not an acceptable cleanup:
although their operand encodings differ, they repeat the same axis choreography and produced a net 765-line
production increase. `.tasks/plan_p5d_batching_rule_consolidation.md` remains the capability baseline and records the
consumer ledger; P5e supersedes its contract-based duplication exception.

P5c and P5d are complete. Structural batching selects direct and composite threaded-extent boundary types through
`BatchingPolicy`, making the distinction static and deleting the interim runtime boundary enum. Condition, while, and
scan support mixed array/dimension values, explicit mapped-extent threading, nested batching, rendering/import, and
the existing XLA promotion/lowering/execution path. Composite scan accepts a first-class dynamic trip count, and
mapped-state RNG batching now decomposes to one state-carry-free composite scan while preserving its exact
replicated-state diagnostic. P5d canonicalizes unprojected control-flow carriers, completes direct explicit-extent
collective batching for static and bounded-dynamic graphs, and records the retained reference-domain rules and their
deletion owners. The full `ryft-core`, `ryft-xla`, and macro integration suites pass.

P5e is complete. Both operation families now reach exactly one matching-axis kernel per shape-changing collective,
with shared nonmatching-axis geometry and no source-array dimension reconstruction inside those kernels. A private
`CollectiveBatchingPolicy<C>` contains only the static-versus-projected extent, requirement, and materialization
boundary; it contains no collective formula or operation-specific axis decision. The composite adapters contain only
explicit-extent validation/projection/lifting, while the homogeneous adapters derive implicit extents and call the
same kernels.

The arithmetic consolidation completed at the same time: generic fallible `Add`, `Sub`, `Mul`, `Div`, and `Rem` use
one `BinaryOperationFor<T>` provider contract; checked `usize` and dimension SSA extents therefore use the same
collective algorithms. The five dimension capability aliases, five operation-specific provider traits, duplicated
tracer operator branches, and the composite-only multiply/divide/equality/alignment/reshape/sharding helpers are
deleted. The final delta from committed P5d is 1,016 insertions and 1,145 deletions: **-130 production lines, +1 test
line, and -129 lines overall**. A recompiling core check measured 2.77 seconds versus P5d's 2.54 seconds (9.1%, inside
the 10% guard), followed by a 0.11-second no-op check.

The final gates pass: 1,034 core tests; 405 XLA tests plus one ignored timing benchmark; both 17-test macro groups;
the static/bounded-dynamic, tiled/untiled, mapped/replicated, matching/nonmatching, nested, rendering/import, PE/JVP,
diagnostic, StableHLO, and CPU-execution fixtures; formatting; diff hygiene; and targeted single-owner/deletion
searches. The complete source ledger and verification record are in
`.tasks/plan_p5e_extent_polymorphic_collective_batching.md`. Phase 6 is now the next review unit.

### Phase 6: simplify differentiation and transposition

- [x] P6a — execute `.tasks/plan_p6a_differentiation_residual_architecture.md` as the first isolated Phase 6 review
      unit. Establish one differentiation-owned, storage-generic residual operand contract and prove it end to end on
      dynamic reshape before migrating other mixed operations or composite regions. The prototype must keep residual
      dimensions as ordinary SSA values, remove fake dimension tangent/cotangent slots from generated linear programs,
      and reject side tables, identity lookup, copied shape metadata, or reshape-specific residual fields.
- [x] P6b — execute `.tasks/plan_p6b_extent_residual_operation_sweep.md` as separately reviewable batching,
      mixed-shape, array-only, and collective slices. Extend the P6a linear-call residual contract across every
      remaining extent-sensitive derivative rule, delete superseded captured-factor payloads and Phase 6 rejections,
      and require full JAX parity plus Ryft's bounded-dynamic extensions before the P6b gate.
- [x] Express the dimension tangent/cotangent space once in differentiation-owned policy.
- [x] Introduce or extend one differentiation-owned residual structure capable of carrying ordinary array-program SSA
      values, including dimensions, without assigning them tangent/cotangent slots.
- [x] Make linearization rules declare required primal dimension residuals explicitly while those operands or source
      arrays are available.
- [x] Thread dimension residuals through nested regions, rematerialization, custom derivatives, JVP/VJP construction,
      import, and transpose exactly like other residual values.
- [x] Rewrite reshape transposition to consume ordinary dimension residual inputs.
- [x] `ReshapeOperation::transpose_dimension_variables` and every exact identifier occurrence are absent from the
      integration tree. Do not reintroduce an equivalent payload residual manifest while implementing the ordinary
      residual path above.
- [x] Audit concatenate, mean/reductions, slice, pad, and gather transposes and migrate every analogous extent need to
      the same residual contract.
- [x] P6c — execute `.tasks/plan_p6c_generic_outer_differentiation_dispatch.md` as the next isolated Phase 6 review
      unit. Make generic outer dispatch project/lift array-only JVP and transpose rules, let VJP reuse their existing
      composition, and exercise the shared toy composite fixture for member-kind-agnostic dispatch tests.
- [x] P6d — execute `.tasks/plan_p6d_composite_region_differentiation.md` as three review-sized vertical slices:
      condition, scan, and while. Reuse the existing control-flow algorithms over `ArrayIrType` regions, preserve
      dimensions as primal/residual SSA without differential slots, compact only time-varying dimension residuals
      through the checked scalar gateway when iteration storage requires it, and reach full JAX parity plus Ryft's
      bounded-dynamic extensions before beginning composite-zero deletion.
- [x] Preserve dimension values as ordinary structural residuals without tangent slots.
- [x] Keep explicit mixed rules only where primal dimension operands control array results or region interfaces.
- [x] Remove temporary homogeneous differentiation programs and dimension recovery.
- [x] Add a residual search proving no primal operation payload stores differentiation-only dimension variables or
      residual manifests.
- [x] Prefer structural zeros over materializing shaped zero arrays within the P6b residual carrier and migrated
      extent-sensitive rules. The complete composite-zero operation deletion remains the separately assigned unit
      below.
- [x] P6e — execute `.tasks/plan_p6e_composite_zero_deletion.md` as the next isolated Phase 6 unit. Delete the generic
      type-only composite zero, migrate every caller to structural, operand-relative, homogeneous, or explicit-extent
      materialization, and rename the sole mixed dynamic-array constructor from `DynamicZero` to `Zero`.
- [x] P6f — execute `.tasks/plan_p6f_region_zero_residuals.md` to complete the residual-zero protocol at direct JVP,
      split linearization, reusable derivative-callable, and scan-transpose boundaries. Materialize structural zeros
      from their primal results' explicit geometry; share one canonical operation/operand assembly hook across
      value-level and builder-level spending; project unused zero-space scan inputs out instead of constructing them;
      and retain condition partial evaluation's conservative whole-condition fallback for identity-bearing branch
      edges.
- [x] Inventory every production construction of `ZeroOperation<ArrayIrType>` and every composite
      `Zero<ArrayIrValue<_>>` materialization path, including the retained-linearization residual-zero sites in
      `differentiation/forward.rs`. Classify each as a structural zero that should remain unmaterialized, an
      operand-relative `zero_like`, an identity-free homogeneous array zero, or a genuinely dynamic zero whose
      explicit dimension SSA operands must already be available.
- [x] Migrate those callers so generic differentiation and transposition never request a zero from
      `ArrayIrType` alone. Preserve [`MaybeZero`] structurally for as long as possible; use `zero_like` when an
      array value supplies runtime geometry; stage `Array(ArrayOperation::Zero(ZeroOperation<ArrayType>))` only for
      identity-free array types; and stage the mixed constructor with explicit dimension operands when a dynamic array
      zero truly must be materialized. A dimension-member zero must be unrepresentable rather than constructed and
      rejected later by inference.
- [x] Delete `ArrayIrOperation::Zero(ZeroOperation<ArrayIrType>)`,
      `From<ZeroOperation<ArrayIrType>>`, and every dedicated inference, eager, batching, differentiation,
      rendering, identity-renaming, and lowering arm once the generic callers are gone. Do not replace them with
      another type-only composite constructor or a hidden extent-recovery path.
- [x] Rename `ArrayIrOperation::DynamicZero(ZeroOperation<ArrayType>)` to
      `ArrayIrOperation::Zero(ZeroOperation<ArrayType>)` after deleting the conflicting generic variant. Expand
      its rustdoc to explain that this top-level variant owns the mixed `(Dimension...) -> Array` signature because
      homogeneous `Array(...)` projection cannot accept dimension operands and structural types do not contain
      concrete runtime extents. Update all tests, diagnostics, rendering expectations, and documentation directly
      without a compatibility alias.
- [x] Add canonical-representation tests proving identity-free zeros use
      `Array(ArrayOperation::Zero(ZeroOperation<ArrayType>))`, identity-bearing zeros use the renamed top-level
      `Zero(ZeroOperation<ArrayType>)` with one explicit operand per dynamic axis, and no operation can encode a
      dimension-member zero. Add residual searches requiring zero source occurrences of
      `ZeroOperation<ArrayIrType>` and `DynamicZero`.
- [x] Preserve proven/disproven/residual requirement behavior and `OrderedAssertion` effects.
- [x] Verify nested JVP/VJP, linearization, transpose, rematerialization, custom derivatives, condition, while, and
      scan.
- [x] Add exact rendered-IR tests proving residual dimension atoms are explicit dataflow edges shared by the forward
      linearization and transpose, with no type expression or payload witness.
- [x] Gate: adding an array-only primitive with ordinary AD/PE rules requires no handwritten composite dispatcher
      case, the generic composite zero escape hatch is gone, and the only top-level zero variant is the explicit mixed
      constructor.

P6a completed at frozen boundary `236b05b1c`. `DifferentiableType::is_zero_space` now owns the distinction between a
typed public zero leaf and an omitted generated tangent/cotangent slot. `Linearization`, `Pushforward`, and `Pullback`
use compact live differential boundaries while preserving the public derivative tree. The new generic
`LinearCallOperation<T>` structurally associates `[linear operands..., residual operands...]` with attached forward
and transpose programs; partial evaluation remains the sole owner of residual first-use ordering and SSA
deduplication. It replaces `CustomVjpTangentOperation` rather than duplicating it, supports eager execution, PE,
nested JVP, import/identity renaming, transpose, custom VJP/rematerialization, XLA forward-region inlining, and a
promoted third storage member without a composite-kind match.

Composite dynamic reshape is the first migrated consumer. `[n, 4] -> [2, 2*n]` carries the explicit output extent and
the independently required source extent as two visible residual edges; repeated/permuted `[n, n]` deduplicates to one
residual; an already-explicit source extent avoids `dimension_size`; and zero extents, multiple runtime sizes,
pullback, nested JVP, import, and sharding are covered. No reshape payload metadata, identity lookup, type-to-value
recovery, or composite reshape Phase 6 rejection remains.

The source ledger is: differential classification in `DifferentiableType`; compact/public boundary handling in
`Linearization`/`Pushforward`/`Pullback`; residual ordering and deduplication in `PartialEvaluationContext`; association
and transpose delivery in `LinearCallOperation`; inverse geometry in the composite reshape JVP's attached regions; and
execution in eager region replay plus XLA forward-region inlining. The final delta is +478 production and +398
test/documentation Rust lines across core/XLA. The generic 523-line carrier replaces the specialized 477-line custom
VJP carrier and is shared by the remaining sweep. Controlled clean core check cost is 7.36s versus 6.92s (+6.4%);
no-op is 0.10s versus 0.11s. Final gates pass 1,065 core tests, 406 XLA tests plus one ignored benchmark, both 17-test
macro groups, 52 core doctests plus 16 intentional ignores, XLA doctests, formatting, and diff hygiene. Full evidence
is in `.tasks/plan_p6a_differentiation_residual_architecture.md`.

P6b is complete. Executable linear calls batch both attached regions, and broadcast, concatenate, pad,
slicing/gathering, reductions, and all shape-changing collectives retain their exact primal geometry as ordinary
residual SSA. The collective adjoints cover varying, invariant, and reduced all-gather, psum-scatter, and all-to-all
without a payload witness or migration-phase diagnostic. Composite derivative semantics now live in the private
`backends::array_programs::differentiation` capability module rather than the central operation-family module, while
exact-static broadcast, reshape, pad, and concatenate transpose through their canonical homogeneous rules. This
removes the remaining duplicate inverse geometry and the last composite static-extent reconstruction without adding
a payload trait or public policy.

The Phase 5 consolidation is net `-21` production Rust lines and four Rust lines overall. Across complete P6b, the
frozen `87065b914` baseline grows by 1,591 production and 1,238 Rust test/documentation lines for the new supported
semantics; the captured-primal payloads, custom residual wrapper, and repeated carrier assembly are deleted. Clean
`ryft-core` checks are unchanged at 14.321 seconds frozen versus 14.306 seconds final. Residual-count and exact-IR
goldens remain pinned, zero output cotangents stage no allocation, and the full core/XLA/macro/doctest gates pass.
Executable fixtures pass JAX 0.6.1 and 0.11.0 across the complete extent-sensitive matrix; Ryft deliberately exceeds
both for non-finite-safe padding-value cotangents. The complete ledger is in
`.tasks/plan_p6b_extent_residual_operation_sweep.md`. Generic outer differentiation dispatch is complete below;
composite-region differentiation and composite-zero deletion remain separate.

The post-review evaluation of `LinearCallOperation` additionally adopted JAX's `linear_call` transposition shape:
the carrier's operands and both region interfaces are residuals-first (`forward: (r, u) -> v`,
`transpose: (r, v̄) -> ū`, matching the `custom_vjp` backward convention verbatim), and transposing the executable
form *swaps* the attached regions into one staged linear call instead of inlining the transpose body. Transposition
is therefore involutive, pullback programs retain the linear-call boundary and its explicit residual edges, and the
future batching rule is swap-stable. The swap validates `cotangent(cotangent(u)) = u` for linear operand types at
staging time, which tangent types satisfy even where primal storage types do not. This deliberately changed the
staged tangent/pullback program shapes and their exact-IR goldens; the transpose-only `custom_vjp` form still
replays its user-written backward inline because that program carries no linearity contract of its own.

P6c is complete at frozen boundary `10e98313d`. Homogeneous batching, JVP, and transpose fallbacks now delegate to
storage-generic, transform-owned adapters built from the existing policy/projection and program-splicing machinery.
`BatchingPolicy::batch_axis` completes the policy's carrier-access contract, allowing
`batch_projected_operation<T, Q, C, P, O>` to replace the array-specific batching adapter without a separate
`BatchProjection` trait. Structural zeros remain types; batching axes and extents cross unchanged; known primal inputs
and live output cotangents are spliced in deterministic order; and VJP inherits the change through the existing
JVP-plus-transposition composition. Mixed, extent-sensitive, collective, opaque, and region-carrying rules remain
explicit. The promoted third member proves all three adapters require no member-kind branch, while production goldens
cover batching, mapped-condition selection, eager JVP, VJP, and direct transpose.

The array-program batching and differentiation owners delete 122 production lines while their generic transform
owners add 185, for an explicitly accepted one-time net `+63`; 332 test/documentation lines bring core to 134,196 Rust
lines from 133,801. There is no duplicate projection protocol, separate projection trait, projected-policy registry,
new production carrier, or parallel transform API. The three context-first projected-operation helpers are public and
re-exported from their owning facades so composite backends can reuse the single canonical implementations. Clean
`ryft-core` check time improves from 14.97 to 12.23 seconds and
no-op from 0.28 to 0.11 seconds. The full 1,089-test core suite, 407-test XLA suite plus one intentional ignore, macro
unit/integration/compile-fail suites,
doctests, formatting, source closure, and JAX extent/collective parity gates pass. Full evidence is in
`.tasks/plan_p6c_generic_outer_differentiation_dispatch.md`. P6d completion is recorded below; composite-zero deletion
remains afterward.

P6d is complete as one combined condition/scan/while review unit. All three operations now reuse their shared
partial-evaluation, JVP, and transpose algorithms for `ArrayIrType`, and the final abstraction pass removed the
duplicated homogeneous/composite bounded-while dispatcher. Dimensions remain primal-only boundary leaves; invariant
dimension residuals thread directly as scan carries, while only time-varying residuals use the visible checked scalar
gateway pair. Dynamic scan lengths, batched-predicate bounded loops, early exit, structural zeros, identity closure,
direct transpose, VJP, batching/rematerialization/custom-derivative nesting, import, StableHLO lowering, and CPU
execution are covered. The public compiled-gradient facade now stages VJP through the retained `jit_call`, reuses the
canonical projected-array gradient seed, and executes captured compiled functions without an XLA-local AD algorithm.
XLA otherwise owns only thin dispatch and consumes the shared core algorithms.

The retained delta is +941 production and +713 Rust test lines across core/XLA, plus a 96-line Python JAX parity
fixture. Representative residual counts are pinned at one, two, or three; the combined condition/scan/bounded-while
lowering preserves two primal and one pullback top-level `jit_call` instructions and measures 5,504/3,258 StableHLO
bytes after lowering their callees. A warmed
non-incremental core check took 7.50 seconds, an isolated no-op check reported 0.24 seconds, and a fresh isolated target
including dependencies took 14.52 seconds. Final gates pass 1,096 core tests, 409 XLA tests plus one intentional
ignore, 52 core doctests plus 16 intentional ignores, all seven focused composite regressions, all three JAX parity
fixtures, compact zero-space `jit_call` boundary coverage, compiled-gradient capture/execution coverage, formatting,
diff hygiene, and deletion/ownership searches.
No generated contract changed. The old
condition/scan/while composite diagnostics and backend algorithm copies are gone; no direct type-only composite-zero
construction was added. The pre-existing generic composite-zero representation/materialization references remain the
next isolated Phase 6 deletion unit exactly as listed below. Full evidence and the source ledger are in
`.tasks/plan_p6d_composite_region_differentiation.md`.

P6e is complete against frozen boundary `f2bc0dfa5`. The type-only composite
`ZeroOperation<ArrayIrType>` representation is gone, and the sole top-level zero variant is now the explicit
mixed `Zero(ZeroOperation<ArrayType>)` constructor whose dimension operands provide runtime geometry. Identity-free
array zeros remain canonical homogeneous `Array(ArrayOperation::Zero(...))` operations, while dimension-member zeros
are unrepresentable. Promotion, inference, eager interpretation, rendering, identity renaming, transforms, and XLA
lowering all preserve that distinction without a compatibility alias or hidden extent recovery.

`ZeroOperationProvider<T>` now captures the actual capability needed by generic differentiation. Homogeneous
operation families retain type-only zero construction, while composite families keep zeros structural until they can
use an exemplar, an identity-free type, or explicit dimension residuals. Projected JVP handles widened tangent element
types through conversion plus `zero_like`; disconnected dynamic pullbacks retain and deduplicate first-class extent
residuals, including repeated identities; and XLA call/shard-map transposition resolves dynamic zero extents from
known dimension inputs. Core owns the residual algorithm and XLA delegates to it.

Focused coverage pins static and dynamic canonical routing, dimension-member rejection, widened projected JVP,
program-level and nested disconnected pullbacks, repeated-identity residual sharing, XLA pullback execution, and
dynamic call cotangents. The final source delta and exact verification evidence are recorded in
`.tasks/plan_p6e_composite_zero_deletion.md`. Phase 7 backend execution and lowering was the next phase at this review
boundary and is now complete above.

P6f is complete against frozen boundary `6471cc9cc`. `ResidualZeroProvider` now has one canonical
`zero_operation_with_residuals` assembly hook; its value-level binding and builder-level insertion are shared defaults,
and XLA delegates assembly to core. Direct JVP captures geometry from the corresponding primal output before
materializing a structural zero. Both split linearization paths force residual-backed zeros into their tangent
programs, preserving the existing guard against arbitrary known/affine tangent outputs. Reusable pushforwards and
pullbacks retain only the dimension residuals required to rebuild omitted zero-space public boundary leaves.

Scan transpose is simpler: its omitted zero-space cotangent input is semantically unused, so the transposed body is
projected to the live boundary instead of receiving a fabricated zero. Condition partial evaluation retains the whole
condition when branch-edge types carry identities; this remains a deliberate non-optimization because no
opposite-branch geometry source exists. Focused eager/staged JVP, program/callable linearization, pushforward,
pullback, condition-fallback, and scan-projection fixtures pass, together with all 1,105 core tests, all 426 passing XLA
tests plus one intentional ignore, strict core doctests, and the three pinned JAX control-flow parity fixtures. The
complete source ledger and verification record are in `.tasks/plan_p6f_region_zero_residuals.md`.

### Phase 7: backend execution and lowering

- [x] Verify every mixed operation lowers explicit dimension operands directly with no reconstruction environment.
- [x] Verify eager XLA dimension arithmetic remains host integer computation with zero device dispatch/cache probes.
- [x] Verify the logical boundary remains unchanged while the physical executable ABI represents every bounded dynamic
      input with bound-shaped storage plus hidden replicated extents, reconstructs it with `set_dimension_size`, and
      returns hidden output extents used to recover concrete logical result shapes.
- [x] Lower every residual `DimensionRequirementOperation` to a runtime assertion that observes the concrete operand
      values and preserves its exact actor name, predicate, bounds/divisor, and observed-value diagnostic.
- [x] Replace each lowering scope's single shared `Option<ValueRef>` token with one deterministic token slot per
      ordered `Effect` class. Assertions advance only `OrderedAssertion`; prints advance only `OrderedIo`; unordered
      I/O does not acquire an ordered chain merely because another effect exists.
- [x] Thread the active ordered-effect token set through condition results, while/scan state, rematerialization,
      custom derivatives, and effectful inlined calls. Pure regions add no token state, and a region containing only
      one ordered class carries only that class.
- [x] Add structural MLIR tests for assertion→assertion, print→print, assertion interleaved with print, pure/effectful
      branches, loops, scan, rematerialization, and repeated inlined calls. Add CPU execution tests proving
      deterministic first-failure order within the assertion class and independence from ordered I/O.
- [x] Verify ordered runtime assertions preserve exact actor-named diagnostics and deterministic same-class order.
- [x] Restore a permanent CUDA-13 bounded-dynamic execution regression in the production composite XLA domain. One
      executable covers logical sizes 4 and 7, repeated execution, exact device values and logical shapes, and the
      absence of concrete-size recompilation.
- [x] Run the complete serialized local XLA suite, the focused CUDA bounded-dynamic two-size execution regression, and
      the complete serialized CUDA-13 feature suite, including the plugin's `PadToStatic` path.
- [x] While sweeping mixed-operation lowering above, record an initial per-operation padding-discipline inventory
      (padding-oblivious versus mask-required versus zero-padding-required under bound-shaped physical storage) as
      Phase 13 input. This is classification only — no behavior changes in this phase — but capturing it during the
      lowering sweep avoids a second complete pass over the same operations later.
- [x] Gate: backend behavior, diagnostics, and bounded physical storage match or exceed the archived golden evidence.

Phase 7 is complete. The exact lowering/eager/ABI inventories, padding classification, assertion architecture, and
command ledger are recorded in `.tasks/plan_p7_backend_execution_and_lowering.md`. The first focused GB10 execution
disproved the prior plugin-owned-boundary assumption: a compact logical size-4 input could not satisfy the executable's
bound-sized allocation. Ryft now owns an explicit physical ABI with bound-shaped data arguments, hidden replicated
input extents, hidden output extents, and concrete logical result reconstruction. The restored three-tier input
materialization path preserves at-bound reuse, clone-shared retained caching, uncached fallback, single-flight
publication, bounded LRU storage, and cached extent scalars. Schema-versioned compilation keys and persistent metadata
encode the physical mappings exactly.

The focused CUDA regression ran logical sizes 4 and 7 twice through one compiled executable with exact values and
logical shapes. On `spark-9460.local` (NVIDIA GB10, CUDA 13.0), the complete serialized CUDA-13 `ryft-xla` feature
suite passed 433 tests with 1 intentional ignore; the serialized local suite passed 432 tests with 1 intentional
ignore. Dynamic RNG retains its exact semantic rejection because upper-bound generation would advance its functional
state by the physical rather than logical element count; Phase 13 owns that padding-sensitive design. Effectful
shard-map bodies also reject exactly because `sdy.manual_computation` cannot thread StableHLO effect tokens across its
boundary; pure shard-map bodies are unchanged. The post-review correction pass moved every checked failure into the
core effect model, deleted the backend effect overlay, made division/remainder and concatenate checks scheduling-safe,
and carried assertion-runtime requirements through persistent cache restoration. Local verification passed 1,100
`ryft-core` unit tests plus its integration/compile-fail/doctest suites, 432 `ryft-xla` tests (1 ignored benchmark),
both macro integration/trybuild suites, formatting, diff hygiene, and residual searches; the exact ledger is in the
P7 plan.

P8a's post-Phase-7 operation-contract inventory is complete against `7da7d7f25`. It corrects the stale dual-contract
count, records the 65 explicit production impl heads and compilation baselines, and establishes the dependency order
in `.tasks/plan_p8a_operation_contract_inventory.md`. The next isolated code review parameterizes
`SelectOperation`, `StopGradientOperation`, and the macro's `TestNullaryOperation` fixture before any associated-type
trait change.

The first P8a prerequisite slice is complete. `SelectOperation<T>`, `StopGradientOperation<T>`, and the test-only
`TestNullaryOperation<T>` now encode their type universe in each zero-sized payload, without aliases or default type
arguments, and only their intended scalar/array instantiations implement `Operation`. Select's mathematical transform
rules remain one shared declaration, stop-gradient uses the generic transform generators, and all core, composite, and
XLA callers select the concrete contract through inference or an explicit family field type. Verification passed the
complete `ryft-core` library suite (1,108 tests) plus `ryft-core` and `ryft-xla` test-target checks. Later prerequisite
slices and the compile-fail gate stay pending as separate review units.

The second P8a prerequisite slice is complete. `OneLikeOperation<T>`, `ZeroLikeOperation<T>`, `PrintOperation<T>`,
`TagOperation<T>`, and `ConvertElementTypeOperation<T>` now make their type universe nominal, retain inferred
constructors, and implement only the matching `Operation<T>` contract. One-like, zero-like, print, and tag continue to
share one declarative differentiation/transposition rule through the extended generic elementwise forms; conversion
retains its specialized derivative behavior under explicit type-equality bounds. Scalar and array conversion inference
remain covered by separate explicitly typed payloads. The region/call payloads, genuinely mixed dual-contract payloads,
and compile-fail gate remain separate review units. Verification passed all 1,107 `ryft-core` library tests and the
`ryft-core` and `ryft-xla` test-target checks before the subsequent region/call edits began in the shared checkout. A
clean isolated snapshot against the advanced HEAD then revalidated all 1,107 core tests plus both macro integration and
trybuild suites for the exact leaf diff.

The third P8a prerequisite slice is complete. `WhileOperation<T>`, `RematerializeOperation<T>`,
`CustomJvpOperation<T>`, `CustomVjpOperation<T>`, and the XLA-owned `JitCallOperation<T>` now carry their type universe
nominally. Each concrete payload instantiation implements exactly one `Operation<T>` contract, while the existing
generic inference, interpretation, PE, batching, JVP, transposition, region, and lowering implementations remain
single shared rules. The array-to-composite while lift reconstructs only the payload's validated iteration-bound
metadata; attached regions continue to live on the instruction. JIT calls retain two intentional instantiations:
`JitCallOperation<ArrayType>` owns the reusable homogeneous batching contract, and
`JitCallOperation<ArrayIrType>` owns the executable composite call contract. No aliases, default type arguments,
compatibility implementations, duplicated semantic rules, or new projection layers were introduced. The genuinely
mixed/homogeneous dual payloads and compile-fail gate remain separate review units. Verification passed the complete
`ryft-core` suite (1,107 tests), the complete `ryft-xla` library suite (433 passed, 1 intentional benchmark ignore),
strict core doctests (52 passed, 16 intentionally ignored), both test-target checks, and both macro unit/integration
and trybuild suites.

The fourth P8a prerequisite slice resolves the most diagnostic genuinely mixed payload.
`CompareOperation<DataType>` and `CompareOperation<ArrayType>` now share the homogeneous comparison contract, while
`CompareOperation<ArrayIrType>` alone owns the mixed dimension-to-Boolean-array contract. The payload's type
parameter is inferred at ordinary bind and operation-family conversion sites; there is no alias, default type
argument, projection adapter, or duplicated inference algorithm. Focused core and XLA compare tests and both complete
test-target checks pass. The remaining random-bit-generator, custom-call, concatenate, pad, and shard-map payloads are
still pending as separate review units.

The fifth P8a prerequisite slice parameterizes `RngBitGeneratorOperation<T>`. Its homogeneous `ArrayType` instance
owns the static-shape kernel contract and scan-based array batching, while its `ArrayIrType` instance owns the
mixed explicit-output-extent contract and composite batching rule. Shared payload validation, identity renaming,
rendering, reference execution, and XLA lowering remain single implementations, and ordinary constructors infer the
type universe from the receiving context. Focused random, composite batching, and XLA eager/lowering tests pass, as do
all 1,112 core tests and all 433 passing XLA tests (with one intentional benchmark ignore); the custom-call,
concatenate, pad, and shard-map payloads remain pending.

The sixth P8a prerequisite slice parameterizes `CustomCallOperation<T>` and `PadOperation<T>`. Their `ArrayType`
instantiations own the homogeneous contracts, while their `ArrayIrType` instantiations own the mixed contracts
with explicit result extents. Each remains one public, type-indexed operation family rather than acquiring a nominal
`Dynamic*` adapter. Shared configuration, rendering, validation, and lowering remain single implementations;
conversion at a family boundary moves the existing owned metadata without allocating or copying it. Composite eager
execution, batching, differentiation, transposition, and XLA lowering select the appropriate typed instantiation.
Verification passed the focused core and XLA custom-call/padding tests, both test-target checks, all 1,112 core tests,
and all 433 passing XLA tests plus its one intentional benchmark ignore. Concatenate and shard-map remain pending.

The seventh P8a prerequisite slice parameterizes `ConcatenateOperation<T>` and removes shard map's obsolete
homogeneous contract. Concatenate's `ArrayType` and `ArrayIrType` instantiations now own the homogeneous and mixed
explicit-result-extent signatures, respectively, while sharing the same normalized axis and semantic implementations.
The later authoritative-declaration step still owns concatenate's conditional assertion-effect metadata, avoiding
premature duplication of operand-derived extent metadata in this prerequisite. `ShardMapOperation<V>` now retains only
its live composite contract: its homogeneous contract had no production consumer after the composite-body migration,
and its projected batching rule only returned an unreachable unsupported-operation error. The composite shard-map
contract continues to enforce the array-only public boundary. Verification passed both test-target checks, focused
concatenate and shard-map coverage, all 1,112 core tests, all 433 passing XLA tests plus its one intentional benchmark
ignore, and strict-warning rustdoc tests for both crates. All production dual-contract prerequisites are now resolved;
the compile-fail gate remains.

The eighth P8a prerequisite slice transiently verified every then-known former alternate contract with core and XLA trybuild
fixtures: all 16 core obligations plus the old mixed JIT-call and homogeneous shard-map contracts failed at the
intended `Operation<WrongType>` bound. Those enumerated snapshots were intentionally removed before the associated-
type prototype rather than retaining more than 300 lines of compiler-version-sensitive diagnostics for failures that
`Operation::Type` makes structurally unavoidable. Phase 8 keeps only durable compile-fail coverage for the universal
one-contract invariant and mismatched homogeneous derive families.

The first direct associated-type compile probe then exposed a missing macro-generated prerequisite and stopped at the
abort gate without retaining source changes. All 32 `define_elementwise_operation!` invocations still generate one
unit payload with both scalar and array `Operation<T>` contracts, which necessarily conflict under
`Operation { type Type; }`. The probe found no other production declarative macro generating multiple contracts, and
all mechanical prototype edits were reversed exactly. P8a therefore reopens the one-contract prerequisites for one
macro-centric elementwise slice before restarting the associated-type prototype.

The reopened elementwise prerequisite is complete. All 32 macro-generated families now use zero-sized markers indexed
by `T: Type`; scalar and array inference attach only to the `DataType` and `ArrayType` instantiations, respectively,
while interpretation, partial evaluation, batching, differentiation, and transposition remain shared generated
semantics. Capability provision selects the marker through `V::Type`, the five checked dimension providers use their
`DimensionType` marker instantiations, and scalar, array, composite, and XLA families store or lower the exact payload
contract they own. Reverse-mode cotangent accumulation uses one `AddOperationProvider` contract so homogeneous
families receive `AddOperation<T>` through a blanket conversion and composite families select their array member from
the cotangent type without inventing a composite elementwise contract.

The migration was reviewed in architectural layers below the roughly 800-line bound rather than leaving a partial set
of generated operations. Zero-sized/default representation, display and debug spellings, heterogeneous inference
fixtures, eager/tracing/PE/batching/AD/transposition behavior, and XLA lowering are all pinned. Verification passes all
1,112 core library tests, all 434 runnable XLA library tests plus one intentional ignore, all 17 macro integration and
compile-fail tests, 52 core doctests with 16 intentional ignores, XLA doctests, both complete test-target checks,
formatting, residual construction searches, and diff hygiene. Phase 8 may now restart the associated-type vertical
prototype.

A follow-up custom-call parity slice completes the portable contract without adding another operation family.
`ArrayType::layout` is the single source of truth for explicit FFI buffer layouts; flat array input/output aliases are
validated by `CustomCallOperation<T>` and lower to StableHLO output-operand aliases; and side-effecting calls now share
the hidden `Effect::OrderedIo` token chain with `PrintOperation` while pure calls remain token-free. The same lowering
supports homogeneous and bounded-dynamic composite calls, including aliased dynamic outputs. Exact StableHLO fixtures
cover complete layout lists, multi-result alias paths, hidden token layouts, and mixed result extents; CPU execution
passes for a non-default column-major call and an aliased side-effecting call. Verification passed all 1,112 core
tests, all 434 passing XLA tests plus its one intentional benchmark ignore, strict core doctests (52 passed and 16
intentionally ignored), XLA doctests, formatting, and diff hygiene.

### Phase 8: enforce contracts and consolidate operation declarations

- [x] Begin only after Phases 1 through 7 have removed implicit replay and overlapping mixed constructors. Capture the
      remaining dual/type-polymorphic payloads plus the implementor and bound inventory before changing the trait. The
      exact ledger, baseline, migration order, and abort criteria are recorded in
      `.tasks/plan_p8a_operation_contract_inventory.md` against `7da7d7f25`.
- [x] Parameterize `SelectOperation`, `StopGradientOperation`, and `TestNullaryOperation` by their operation type as
      the first isolated prerequisite. Each concrete payload instantiation now has exactly one contract, without a
      compatibility alias or default type argument.
- [x] Apply the proven pattern to the type-polymorphic leaf payloads (`OneLike`, `ZeroLike`, `Print`, `Tag`, and
      conversion), retaining constructor inference and one shared transform rule per semantic operation.
- [x] Apply the proven pattern to the remaining region/call payloads (`While`, rematerialization, custom JVP/VJP, and
      XLA JIT call). Each concrete payload instantiation now has exactly one contract.
- [x] Resolve all genuinely dual-contract payloads before adding `Operation::Type`. Each concrete
      payload instantiation must have exactly one contract without duplicating semantics.
  - [x] Parameterize `CompareOperation<T>` so its homogeneous and mixed contracts belong to distinct concrete payload
        instantiations while retaining shared semantics.
  - [x] Parameterize `RngBitGeneratorOperation<T>` so its homogeneous static-shape and mixed explicit-extent contracts
        belong to distinct concrete payload instantiations while retaining shared semantics.
  - [x] Parameterize `CustomCallOperation<T>` so its homogeneous and mixed explicit-result-extent contracts belong to
        distinct typed instantiations of one public operation family without duplicating foreign-kernel semantics.
  - [x] Parameterize `ConcatenateOperation<T>` so its homogeneous and mixed explicit-result-extent contracts belong
        to distinct typed instantiations while retaining shared semantics.
  - [x] Parameterize `PadOperation<T>` so its homogeneous and mixed explicit-result-extent contracts belong to
        distinct typed instantiations of one public operation family without duplicating padding semantics or rules.
  - [x] Retain only the canonical composite contract on `ShardMapOperation<V>` and delete its obsolete homogeneous
        contract and unreachable always-rejecting batching rule.
  - [x] Parameterize the 32 operation payloads generated by `define_elementwise_operation!` so scalar and array
        contracts belong to distinct typed instantiations. Migrate capability provision, homogeneous families,
        transforms, tests, and XLA consumers through shared macros in one bounded prerequisite; add no compatibility
        aliases, default type arguments, nominal adapters, or duplicated semantic rules.
- [x] Verify every migrated production payload's former alternate contract with transient pre-prototype compile-fail
      coverage, then remove those enumerated snapshots before the associated-type migration. Keep only the durable
      universal and homogeneous-enum failures in the prototype gate below.
- [x] Prototype `Operation` with an associated `Type` — executed as a full-scale worktree experiment rather than a
      bounded slice, since the trait change is compile-atomic (branch `experiment/p8-assoc-type` @ `50e86f964`;
      evidence in its `EXPERIMENT_NOTES.md` and `EXPERIMENT_E0284_PROBE.rs`), then adopted on the live tree
      (2026-08-02, 84 files, +1,553/−1,063 across `ryft-core`/`ryft-xla`/`ryft-macros`/`ryft-macros-tests`).
- [x] Update the derive macro so homogeneous enums prove that every payload has the same operation type
      (`#[derive(Operation)]` emits `type Type = <family type>` plus one `Payload: Operation<Type = <family type>>`
      predicate per member).
- [x] Exercise the design through inference, eager interpretation, tracing, PE, batching, JVP, VJP, transposition,
      rendering, region import (experiment), and XLA lowering (adoption; 434 XLA library tests).
- [x] Simplify the three projected-operation helper signatures and every call site after `Operation::Type` makes the
      member type recoverable from the projected operation: `jvp_projected_operation` and
      `transpose_projected_operation` must require no turbofish, while `batch_projected_operation` must drop its
      explicit member-type argument and either infer its projected batching policy from the final policy contracts or
      keep that one irreducible policy selection explicit. Do not retain inferred generic placeholders or introduce a
      marker/wrapper solely to relocate the same type annotation. Add compile-checked direct-call fixtures pinning the
      final syntax. Completed 2026-08-03: all three helpers now infer the projected member type from the operation
      argument and are called without helper turbofishes; batching initially selected its sole member policy through
      `BatchingPolicy::Projected` (subsequently generalized to the type-indexed `BatchingPolicyProjection<C, T>` in
      P8b); the duplicate direct homogeneous-array `AddOperation` conversions were deleted; and all 34 `ryft-xla`
      `AddOperation::<X>::new()` annotations were removed.
- [x] Add compile-fail tests proving one payload cannot acquire two semantic type contracts and a homogeneous enum
      cannot combine mismatched payload types (`error_multiple_operation_types` pins E0119;
      `error_mismatched_payload_type` pins the derive-boundary rejection).
- [x] Measure clean/incremental compile time, peak memory, macro output size, and trait-solver stability against
      Phase 0 (clean +3.1% time / RSS flat; no-change flat; incremental after touching `programs/operations.rs`
      +36%; derive output slightly smaller; zero residual solver ambiguity).
- [x] Produce a mechanical migration count for all crates, not only `ryft-core` (84 files: core 70, XLA 7,
      macros-tests 5, macros 2).
- [x] Gate: adopt the associated-type trait only if it enforces the already-established canonical signatures with no
      trait-solver regression, no wrapper layer beyond the approved localized type-indexed payloads, and a neutral or
      smaller final generic surface. **Passed and adopted** (bound spellings 623 → 295; turbofishes 242 → 0; the
      accepted trade is the 4 relaxed projecting supertraits recorded in the ADOPTED note above). Post-adoption
      simplification (2026-08-03): all three universe-dispatch traits were subsequently removed in favor of
      per-instantiation / generic-plus-composite `Operation` impls (`ElementwiseUniverse` via the owner's macro
      restructuring; `CompareUniverse` and `RngBitGeneratorUniverse` by owner decision after establishing they bought
      only direct-call inference ergonomics), at the cost of four explicit universe spellings in tests.
- [x] ~~Fallback gate: if rejected, implement the smallest sealed one-contract marker that prohibits dual semantics
      and document why the associated type failed.~~ Not triggered; the interim `OperationType` marker that had been
      staged in this direction was subsumed into the associated type during adoption.

- [x] Re-audit the Phase 8 declaration prerequisites after adopting associated-type `Operation`. The audit found that
      the previous single `member(T)` model conflated rule-bearing projected array operations with replicated,
      zero-differential dimension operations, and that `BatchingPolicy::Projected` cannot select by projected member
      type. The corrected migration and review-unit ledger lives in
      `.tasks/plan_p8b_array_program_operation_declaration.md`.
- [x] Replace `BatchingPolicy::Projected` with `BatchingPolicyProjection<C, T>`, add the replicated dimension-member
      policy, and route both homogeneous array and replicated dimension member operations through the common projected
      batching helper without changing their distinct mapped-axis semantics.
- [x] Execute `.tasks/plan_p8b_array_program_operation_declaration.md` as the next bounded Phase 8 sequence: first
      restore type-indexed batching-policy projection and move semantic rules to their payload owners, then extend the
      existing derive vertically, and only then migrate the production enum.
  - [x] Freeze the variant/conversion migration ledger and pre-migration source, expansion-size, and macro-test build
        baselines at `ef1cb48823436a298f625daa975155a819ad0000`.
  - [x] P8b1a: move linear-call batching beside its payload, introduce the narrow `LinearCallBatchingPolicy` extension
        point for universe-specific cotangent collapse, and replace the homogeneous/composite operation adapters with
        one generic `BatchableOperation` implementation while leaving direct delegation in the outer dispatcher.
  - [x] Continue P8b1 with the condition/while/scan and trivial structural rules.
    - [x] Relocate composite condition and while batching to their payload owners, leaving direct outer delegation.
    - [x] Relocate composite scan batching and close the P8b1b control-flow gate.
    - [x] Relocate the P8b1c trivial structural rules.
  - [x] Continue P8b2 with the extent-sensitive indexing rules.
    - [x] Add the generic projected-differentiation capability for member rules that require the composite context.
    - [x] Relocate the P8b2a static and dynamic read-slice rules.
    - [x] Relocate the P8b2b dynamic-update-slice and gather rules.
    - [x] Relocate the P8b2c reduction rules and close P8b2.
  - [x] Continue P8b3 with the shape-operation differentiation and transposition rules.
    - [x] Relocate the P8b3a concatenate rules.
    - [x] Relocate the P8b3b reshape rules.
    - [x] Relocate the P8b3c broadcast rules.
    - [x] Relocate the P8b3d pad rules and close P8b3.
  - [x] Continue P8b4 with the remaining mixed-operation transform rules.
    - [x] Relocate the P8b4a comparison, custom-call, and RNG rules.
    - [x] Relocate the P8b4b collective differentiation and transposition rules.
    - [x] Relocate the P8b4c collective batching rules.
    - [x] Relocate the P8b4d remaining mixed structural and batching rules, then close P8b4.
  - [x] Extend the derive parser with typed `#[ryft(projected(T))]` and `#[ryft(replicated(T))]` class markers and pin
        all invalid declaration diagnostics.
  - [x] Prove projected-member base dispatch through type inference, eager interpretation, PE, rendering, conversions,
        and a second projected member type without array-program-specific generated code.
  - [x] Add the minimal outer-family declaration needed by production-shaped composite enums: deriving
        `ArrayIrOperation<A>` must select `ArrayIrType` and `ArrayIrValue<A>` even though its stored
        generic `A` belongs to the projected `ArrayType` member. Do not add a phantom composite-value parameter or
        redesign the enum's public generic merely to satisfy derive inference.
  - [x] Generate and prove the complete projected-member and replicated-member vertical contracts before annotating
        `ArrayIrOperation`.
- [x] Establish one authoritative declaration of every array-program operation and its class.
- [x] Use the declared outer variants to generate inner lifts, direct `From` conversions, borrowed projections, and
      mechanical dispatch.
- [x] Make `#[derive(Operation)]` with `#[ryft(dispatch(batching, differentiation, transposition))]` the mechanical
      dispatch surface for `ArrayIrOperation`, matching the homogeneous families instead of introducing a second
      generator syntax. Before migration, move every semantic rule off the enum match and into its colocated payload
      implementation, and ensure each dual-contract variant carries a dedicated typed payload instantiation (or a
      semantic mixed payload when its stored metadata differs) so one concrete payload never carries two semantic
      contracts. Extend the derive with variant-level class markers that distinguish rule-bearing projected members
      from replicated structural members and emit the corresponding project-delegate-lift or structural arm by calling
      shared helpers rather than inlining boundary logic. Select a projected batching policy through the type-indexed
      `BatchingPolicyProjection<C, T>` relation instead of assuming one projected policy per outer policy, and select each
      top-level universe's entrypoint policy through `<PrimaryType as BatchableType>::Policy` instead of hard-coding
      `ArrayBatching<P>`. The class markers and typed mixed payloads are irreducible declaration content, not ceremony:
      which universe and transform contract a variant's payload speaks must be declared exactly once on the enum.
      Design decisions settled in review (2026-08-03):
      - The projected-member arm is **family-level**, not per-primitive: because `ArrayIrOperation` embeds the
        entire `ArrayOperation` family as one member variant, the derive generates one project-delegate-lift adapter
        at the family boundary and every current and future `ArrayOperation` batching rule flows through it. A new
        composite universe provides its projection vocabulary once (`ValueProjection`, `OperationProjection<ArrayType>`
        plus operation lifts, `BatchingPolicyProjection<C, ArrayType>` with a projected policy compatible with the
        outer extent representation, and its `BatchableType` entrypoint policy) and inherits the adapters.
      - `LinearCallOperation<ArrayIrType>` is classified **composite-native**, not a projected member: its
        regions carry composite boundaries with dimension residuals and batching threads the first-class mapped
        extent through them, so no member marker can generate its semantics. The shared semantic algorithm stays in
        `LinearCallOperation::batch_regions`, and one generic `BatchableOperation` implementation uses the narrow
        `LinearCallBatchingPolicy` extension point to collapse mapped cotangents. Ordinary arrays reduce directly;
        array programs project/reduce/lift. A downstream composite universe implements that policy capability once,
        without reimplementing the Ryft-owned operation or its region algorithm. This is an exceptional extension
        seam, not a template: if another operation needs the same hook, generalize the shared semantic capability;
        do not introduce one operation-specific policy trait per primitive.
      - Prototype sequencing: prove the generated member adapter on the macro third-member fixture and then the
        production `Array` variant; use `Concatenate` afterwards to prove the shared-array-core-plus-explicit-mixed-
        wrapper pattern (it is NOT a member-projection prototype — its composite contract consumes a trailing
        result-extent operand, validates and rejects mapped extents, and lifts the concatenation axis; the 2026-07-24
        "delegatable" audit finding predates that mixed contract). All genuinely mixed contracts (reshape, broadcast,
        concatenate, pad, custom call, RNG, collectives, gateways, and region operations) keep explicit wrappers
        where their signatures or region boundaries differ.
- [x] Make each mixed operation's inference contract the authoritative source for dimension operand positions,
      member kinds, ordering, and result metadata.
- [x] Extend the typed mixed projection vocabulary only for repeated fixed/optional/segmented patterns found in the
      Phase 0 inventory.
- [x] Centralize only structural projection needed to ensure eager interpretation, transforms, and lowering preserve
      the operand order declared by inference.
- [x] Delete redundant local variant lists, conversion macros, manual wrong-kind matches, and projection boilerplate.
- [x] Delete independent `runtime_dimension_variables` methods after their operations consume explicit operands.
- [x] Keep semantically meaningful operation rules handwritten and colocated with their payload.
- [x] Add compile-fail coverage for invalid generated operation declarations and runtime goldens for canonical
      projection diagnostics.
- [x] Run macro unit and integration tests and compare generated token counts/compile time with the baseline.
- [x] Gate: one new array-only primitive requires one family declaration and its semantic/backend rules; one new mixed
      operation declares its signature once and does not add projection ceremony to transforms.
- [x] Replace concatenation's conservative unconditional `OrderedAssertion` effect with a typed mixed payload that can
      own the operand-derived extent proof needed to classify the effect conditionally. This is a semantic effect-
      precision follow-up, not part of the derive's mechanical dispatch contract.
  - [x] Add a type-derived mixed-concatenation constructor and retain only the proof bit needed by `effects`; do not
        clone complete operand signatures into the payload or widen the generic `Operation` contract.
  - [x] Make inference validate that a payload classified as pure is still bound to a signature that proves the
        explicit result extent, while allowing an assertion-bearing payload to remain conservatively effectful after
        type refinement.
  - [x] Make eager execution retain its defensive extent check and make XLA lowering omit the assertion callback,
        assertion token, and extent-size IR exactly when the payload is proven pure.
  - [x] Update all mixed construction sites, add pure/effectful core and lowering regressions, run the core and XLA
        gates, and record the completed review unit below.

### Phase 9: module and public API cleanup

- [x] Standardize the composite array/dimension vocabulary on `ArrayIrType`, `ArrayIrValue`, and `ArrayIrOperation`,
      including the directly owned `ArrayIrTypeRefinements` companion, `ArrayIrBatch` carrier, `ArrayIrBatching`
      policy, and projection-allocation fixture. Keep the existing `types::arrays` and `backends::array_programs`
      module placement for this review unit, update every in-repo consumer without compatibility aliases, and reserve
      the shorter “array IR” terminology for this heterogeneous SSA representation rather than ordinary homogeneous
      programs over `ArrayType`.
- [x] Confirm the `S4` typed `Custom`/`DimensionError` recovery behavior and canonical invalid projection diagnostics
      remain intact;
      do not mix another error-representation migration into the module move.
- [x] Core dimension operation semantics are split from the eager host representation.
- [x] Dimension operation semantics live in `operations::dimensions`.
- [x] `DimensionValue`, its closed eager operation family, and concrete capability implementations remain under backend
      ownership.
- [x] Re-evaluate the historical `RuntimeDimension`/`RuntimeShape` item: neither identifier exists in the current
      tree. Confirm that the public first-class dimension capabilities cover the intended ergonomics; do not recreate
      wrapper types merely to satisfy the old module-move wording. If a neutral public alias/API is still needed, add
      only the smallest capability-based surface after the operation and transform families settle.
- [x] After Phase 8 has moved every operation-specific batching algorithm to its payload owner, replace the historical
      asymmetric module layout with a symmetric batching hierarchy:
  - [x] Make `BatchedProgram` the universe-neutral carrier contract, rename the reusable exact-source-boundary carrier
        to `BoundaryPreservingBatchedProgram`, require every `BatchingPolicy::BatchedProgram` to implement the contract,
        and consolidate output-axis alignment into one policy-generic `BatchingContext` implementation. Concrete
        carriers continue to validate their own boundary invariants and the generic contract never interprets or drops
        policy-owned bookkeeping values.
  - [x] Keep universe-neutral contracts and machinery in the `batching` module root (`BatchingPolicy`,
        `BatchingPolicyProjection`, `BatchableType`, contexts/tracers/drivers, the `BatchedProgram` trait and
        `BoundaryPreservingBatchedProgram` default carrier, transform entrypoints, and projected-operation helpers).
  - [x] Move the `ArrayType` specialization to `batching::arrays` (`ArrayBatch`, `ArrayBatching`,
        `ArrayBatchingPolicy`, `StaticArrayBatchingPolicy`, their policy/recursive/entrypoint implementations, and
        shared array-axis mechanics).
    - [x] Move `ArrayBatch`, batching-specific `ArrayType` normalization/unbatching, and shared mapped-axis sharding
          mechanics; keep `batching::ArrayBatch` as the intentional public batching facade and route generated and
          handwritten paths through it.
    - [x] Move `ArrayBatching`, `ArrayBatchingPolicy`, `StaticArrayBatchingPolicy`, and their policy, recursive, and
          entrypoint implementations after the carrier slice is reviewed.
    - [x] Move the array-specialization tests beside their owner and close the specialization-module checkbox.
  - [x] Move the `ArrayIrType` specialization to `batching::array_ir` (`ArrayIrBatch`,
        `ArrayIrBatching`, `ThreadedExtentBatchedProgram`, `DynamicArrayBatchingPolicy`,
        `ReplicatedDimensionBatchingPolicy`, their policy/projection/recursive/entrypoint implementations, and shared
        first-class-extent mechanics).
  - [x] Keep operation-specific `BatchableOperation` implementations and operation-owned policy extensions such as
        `LinearCallBatchingPolicy` beside their operation payloads rather than moving them into either specialization
        module.
  - [x] Gate: neither specialization module contains an outer operation dispatcher or operation-specific batching
        algorithm, and `backends` contains concrete eager values/operation families rather than transform policy types.
        Update every in-repo path directly and decide the intentional `batching` facade during this move without adding
        compatibility re-exports.
- [ ] After the batching hierarchy and operation ownership have settled, consolidate the reference array stack under
      one top-level `ryft_core::arrays` hierarchy as a separate, measured API and representation change:
  - [ ] Execute the Phase 9a representation and scalar-retirement sequence specified below. Storage must migrate
        before scalar deletion because the current `Array` uses `Vec<Scalar>` in production; keep existing public
        backend and type paths fixed throughout this sequence, apart from the private `arrays::addressing` module, so
        storage, literal lowering, and scalar retirement remain separate from the later hierarchy move.
    - [ ] Replace the reference `Array`'s element-wise `Vec<Scalar>` payload with validated immutable contiguous bytes,
          with payload-copy-free cloning and checked exact encodings for every `DataType`.
    - [ ] Route exact array literals and XLA constant lowering through logical traversal of the layout-aware byte
          representation without an `f64` round trip, including the sub-byte integer families.
    - [ ] Prove that the remaining standalone scalar program universe has no unique production role. Move useful
          `DataType`-universe tests to rank-zero arrays or narrowly scoped test fixtures, then delete `Scalar`,
          `ScalarOperation`, `ScalarTracingContext`, `backends::scalars`, and their public exports without a
          compatibility layer.
    - [ ] Gate the representation/retirement sequence with exact-bit, allocation, core, macro, XLA CPU, available CUDA,
          and before/after size and performance evidence.
  - [ ] Define and document the final public hierarchy before moving files. The expected canonical layout is
        `arrays::{data, dimensions, ir, layouts, memories, sharding}` with common array types re-exported from
        `arrays`; remove ambiguous glob exports and duplicate canonical paths rather than preserving the former
        `backends`, `types`, and `sharding` facades.
  - [ ] Move the current `backends::arrays` reference backend to `ryft_core::arrays`, and place the current dimension
        backend and heterogeneous array IR beneath the same hierarchy (expected submodules: `arrays::dimensions` and
        `arrays::ir`). Keep generic operation semantics in `operations`, generic program machinery in `programs`, and
        transform policy machinery in its transform modules; this hierarchy owns the complete array-language type
        vocabulary, concrete reference values, and their closed operation families.
  - [ ] Move every array-language-specific type out of the top-level `types` hierarchy and give it one canonical home
        under `ryft_core::arrays`:
    - [ ] Move `DataType`/`DataTypeError` under `arrays::data`; after the scalar backend is deleted, these describe array
          element data rather than an independent scalar execution universe.
    - [ ] Move `ArrayType`/`ArrayTypeRefinements` into the array hierarchy and expose them canonically as
          `ryft_core::arrays::ArrayType` and `ryft_core::arrays::ArrayTypeRefinements`.
    - [ ] Move `Dimension`, `DimensionBounds`, `DimensionType`, `DimensionVariable`, `Shape`, `StaticShape`, their
          errors/constants, and the concrete `DimensionValue`/`DimensionOperation` backend into
          `arrays::dimensions`, with the commonly used type vocabulary re-exported from `arrays`.
    - [ ] Move `ArrayIrType`/`ArrayIrTypeRefinements` beside `ArrayIrValue`/`ArrayIrOperation` in `arrays::ir`, while
          re-exporting the four canonical IR names from `arrays` for concise public signatures.
    - [ ] Move `Layout`, `StridedLayout`, `Tile`, `TileDimension`, `TiledLayout`, and `LayoutError` under
          `arrays::layouts`, and move `Memory` under `arrays::memories`; both are metadata of `ArrayType`, not generic
          program-type machinery.
    - [ ] Delete the top-level `types` module after its array-specific contents have moved. Do **not** move those
          concrete types into `programs::types`: that module already owns the correct backend-neutral layer
          (`Type`, `Typed`, `TypeError`, `TypeRefinements`, and signature traversal), which must remain independent of
          arrays and of any future value universe.
  - [ ] Move the current top-level `sharding` hierarchy to `arrays::sharding` (including meshes, shardings,
        visualizations, and `ShardingError`) after auditing that no non-array value universe uses it independently.
        Update all paths directly without a `ryft_core::sharding` compatibility module; named-axis transform machinery
        remains outside this hierarchy where it is genuinely universe-neutral.
  - [ ] Gate: the top-level hierarchy has one obvious public path for reference arrays, dimensions, and array IR; no
        scalar backend or per-element `Scalar` payload remains; all reference-backend semantics, transformations,
        exact-literal tests, core/XLA execution suites, and allocation/performance thresholds pass.
- [ ] Audit names after responsibilities settle; rename only where the final name is materially clearer.
- [ ] Known residual from the P4b audit: `ryft-xla/src/profile_guided.rs` names `ArrayIrValue<Array>` in the
      `where` clauses of two public functions (`interpret` and `profile_baseline`). Bounds only — no public value
      positions — but it is public-signature/rustdoc surface; tighten or accept explicitly during this cleanup.
- [ ] Update every in-repo use site directly without compatibility re-exports.
- [ ] Update rustdoc, examples, error links, and behavioral JAX fixtures.
- [ ] Close the foreign-call batching gap as an isolated transform/API review after the operation contracts settle.
      Follow JAX's current direction rather than copying the deprecated `ffi_call(vmap_method = ...)` parameter:
      prototype a general user-defined custom batching rule that can wrap foreign calls, and retain it only if one
      implementation composes through homogeneous and mixed programs, nested batching, JIT, AD, partial evaluation,
      ordered effects, and first-class dynamic output extents. Add small convenience rules only where they reproduce
      the useful `sequential`, `sequential_unrolled`, `expand_dims`, `broadcast_all`, and `legacy_vectorized`
      behaviors without introducing a second batching path.
- [ ] Run targeted searches for every old canonical path and classify all remaining matches.
- [ ] Gate: core language semantics no longer appear to be backend implementation details.

#### Phase 9a: contiguous array storage and scalar-backend retirement

Objective: replace the reference `Array` backend's `Vec<Scalar>` payload with one validated immutable contiguous byte
representation, route exact literals through that representation, and then delete the standalone `Scalar` value
universe and `ScalarOperation` family. The final implementation must reduce production code, preserve reference
semantics for every `DataType`, make `Array::clone` share rather than copy or allocate its element payload, and avoid
introducing a second public scalar backend under another name. `ArrayType` remains stored by value and may retain its
existing small metadata-clone cost.

Current-tree facts and dependency order:

- `Array` stores `values: Vec<Scalar>` and uses `Scalar` in production for storage, element conversion, arithmetic,
  comparisons, reductions, indexing, formatting, and construction. Scalar deletion therefore cannot be the first
  slice.
- `ScalarOperation` and `ScalarTracingContext` are primarily a second eager/staged test universe. Useful generic
  transform coverage must move to rank-zero arrays or narrowly scoped test fixtures before that universe is deleted.
- XLA literal lowering expands each `Array` into typed vectors by matching stored `Scalar` variants, adding avoidable
  per-element enum dispatch and leaving the sub-byte integer families unsupported.
- Keep the current public backend and type paths fixed through Phase 9a. The private `arrays::addressing` module is the
  deliberate first piece of the eventual array hierarchy because both reference storage and future physical-layout
  consumers share it; storage, literal lowering, scalar retirement, and the later public hierarchy move remain
  separate review units.

Target representation and API:

- Store `Array { type: ArrayType, bytes: Arc<Vec<u8>> }`. The type's static shape and physical layout determine the
  required storage span and byte length. The privately owned `Vec` is immutable after construction, avoids the extra
  allocation and payload copy required to convert a built `Vec<u8>` into `Arc<[u8]>`, and remains one contiguous
  physical allocation even when a layout introduces holes or tile padding. Clones share its immutable element storage;
  cloning the by-value `ArrayType` is intentionally permitted.
- Physically honor `ArrayType::layout()` in the reference `Array`, and therefore in the array member of the reference
  `ArrayIrValue<Array>` instantiation. A type without an explicit layout uses dense logical row-major storage. A
  strided layout determines byte strides and the base offset required by negative strides; a tiled layout determines
  physical ordering and padding. Portable literal construction is a conversion over logical coordinates through
  `ArrayAddressing`, not a second layout-independent storage representation retained inside `Array`.
- Use a documented portable little-endian encoding. `Token` and `Zero` carry no payload bytes. Boolean, sub-byte
  integers, and 4/6/8-bit floats occupy one validated byte per logical element; this favors constant-time reference
  indexing, and lowering packs Boolean and sub-byte integer elements only where the target literal format requires it.
  Wider integers and floats use exact bit patterns, and complex values interleave real and imaginary component
  encodings.
- Add checked physical-storage construction and immutable byte access. Validate the layout-derived storage span,
  Boolean encodings, sub-byte ranges, unused float bits, holes/padding, and payload-free types. Provide concise typed
  logical construction/decoding through one sealed array-element codec for Rust primitives, `half`, and `num_complex`;
  ambiguous low-precision and sub-byte families use the checked raw-bits path and are placed through layout-aware
  addressing.
- Remove `Array::new(ArrayType, Vec<Scalar>)` and `values() -> &[Scalar]` instead of retaining overloads. Preserve the
  `scalar`, `vector`, `matrix`, `from_f64s`, and `to_f64s` conveniences with byte-backed implementations, while exact
  literals use typed or raw-bit construction rather than `f64` conversion.
- The existing scalar element algebra may be reused only as an implementation migration source. If a transient private
  element enum materially keeps kernels smaller, move and narrow the implementation rather than duplicating it; it
  must never be stored in `Array`, implement `Type`/`Value`, own an operation family, or remain publicly exported.

Guardrails:

- No compatibility module, deprecated alias, dual storage representation, `Vec<Scalar>` fallback, per-element box, or
  unsafe typed-slice cast from byte storage.
- No public indexing capability, general array-view abstraction, or parallel canonical payload beside physical
  layout-aware storage. Add only the private addressing and traversal machinery required by reference operations and
  literal conversion.
- No silent lossy conversion through `f64`, especially for integers, signed zeros, NaN payloads, low-precision floats,
  and complex values.
- Do not materialize a complete secondary element vector in reference kernels. Build one output byte buffer with known
  capacity and freeze it into shared immutable storage once.
- Keep implementation slices near the established review-size limit. A mechanical test migration may exceed it only
  through deletions and direct substitutions, not new production abstractions.

Phase 9a0 — baseline and vertical prototype:

- [x] Record the clean starting revision, production/test line counts for `backends::{arrays,scalars}`, `size_of` for
      `Array` and `Scalar`, construction/clone byte and allocation costs for representative 4K-element arrays, and
      current core/XLA test counts.
- [x] Classify every production and test use of `Scalar`, `ScalarOperation`, and `ScalarTracingContext` as
      array-element semantics, reusable generic-transform coverage, scalar-universe-only coverage, or dead coverage.
- [x] Inventory every `DataType` encoding, current `Scalar` support gap, `Array` constructor/accessor call site, and XLA
      literal route; record exact expected byte lengths and validation rules.
- [x] Prototype one complete F32 vertical slice: checked typed construction/decoding, one unary kernel, one broadcasting
      binary kernel, equality/rendering, clone-allocation measurement, and exact XLA dense-literal construction.
- [x] Compare a narrow private transient-element implementation with direct typed byte dispatch for the prototype.
      Select the smaller option only if it adds no per-element heap allocation or public API surface.
- [x] Gate: `Array::clone` allocates or copies no element payload bytes; only the accepted `ArrayType` metadata clone
      may allocate. The prototype adds no unsafe cast, second stored representation, or more production code than its
      equivalent current F32 path.

Phase 9a1 — layout-aware byte storage and construction:

- [x] Before converting `Array`, add one private checked dense-buffer addressing contract derived from a static
      `ArrayType`: element byte width, logical element count, logical byte length, physical storage byte length,
      row-major element strides, flat and multi-index element offsets, and byte ranges for one element or a contiguous
      flat element range. Validate rank, bounds, multiplication/addition overflow, zero-rank arrays, empty arrays, and
      zero-byte `Token`/`Zero` elements when constructing the descriptor or validating each externally supplied index.
- [x] Add one allocation-free rectangular/strided logical-index range iterator for traversal patterns appearing in
      slice/update, transpose, broadcast, pad, concatenate, gather/scatter, reduce, and dot. Validate the complete
      start/size/stride specification once, then yield maximal contiguous flat element ranges (or single-element ranges
      when a traversal is non-contiguous) without per-element allocation or repeated bounds checks. Do not add a general
      view/slice type: specialized kernels may keep their own coordinate logic when the shared iterator would make them
      less clear.
- [x] Generalize `ArrayAddressing` to support every physical `ArrayType` layout used by reference arrays. A missing
      layout defaults to dense row-major addressing; explicit layouts must determine actual reference-array storage.
      Cover positive and negative byte strides, derived base offsets, holes, alias rejection, tiled layouts and padding,
      checked storage-span calculation, and physical-contiguity-aware range coalescing. Keep `ArrayType` as the sole
      stored source of truth and add exact logical-index-to-storage-range tests for every layout family before routing
      physical-buffer consumers through the abstraction.
- [x] Pin layout-aware reference storage: construct equal logical values under layout-free, positive/negative strided,
      permuted tiled, and padded tiled types; assert each type's expected physical bytes and storage span; and decode
      each back to the same logical values through `ArrayAddressing`. Portable literal conversion must traverse logical
      coordinates and produce the target format independently of the reference array's physical byte ordering.
- [x] Add byte-length/range validation and the sealed typed codec at the array/data ownership boundary.
- [x] Convert `Array` storage and migrate constructors, accessors, `Parameter`, equality, approximation, formatting,
      and the test-only malformed-type constructor in one complete slice.
- [x] Add exact encoding round trips for all supported primitives, low-precision raw bits, signed zero, infinities,
      representative NaNs/payloads, complex values, empty arrays, `Token`, and `Zero`.
- [x] Add I1/I2/I4/U1/U2/U4 reference-array construction and validation, closing the current `Scalar` storage gap
      without adding scalar enum variants.
- [x] Extend allocation tests to prove large-array clone performs no payload-sized allocation, while borrowed and
      consuming projection remain fully allocation-free after setup.
- [x] Gate: no `Vec<Scalar>` payload, `values()` accessor, or duplicate byte ownership remains.

Phase 9a2 — byte-backed reference kernels:

- [x] Migrate elementwise arithmetic, logical, comparison, complex, conversion, zero/one/fill/iota, and random kernels
      family by family, preserving integer wrapping and fallible errors.
  - [x] P9a2a: migrate `Not`, `And`, `Or`, and `Xor` directly over layout-aware Boolean and integer storage, including
        sub-byte masks and complete NumPy-style broadcasting, without temporary scalar or logical-byte payloads.
  - [x] P9a2b: migrate ThreeFry and Philox random-bit generation to typed, layout-aware state decoding and result
        construction for U8/U16/U32/U64 outputs, without routing state or generated bits through `Scalar`.
  - [x] P9a2e: add exhaustive sealed-element dispatch and a zero-intermediate-allocation broadcasted binary loop, then
        migrate exact equality and comparison to direct typed codecs. Preserve signed-zero, NaN, complex-ordering,
        payload-free, promotion, arbitrary-layout, and complete NumPy-style broadcasting semantics.
  - [x] P9a2k: migrate element-type conversion to direct typed codecs for every supported source and destination data
        type. Preserve Boolean, integer narrowing, sub-byte, low-precision, complex-to-real, fallible encoding,
        payload-free rejection, same-type bit preservation, and arbitrary-layout semantics without a temporary
        [`Scalar`] payload.
  - [x] P9a2l: migrate `Abs`, `Neg`, `Add`, `Sub`, `Mul`, `Div`, `Rem`, `Max`, and `Min` to direct typed codecs.
        Normalize mixed data types through the canonical conversion kernel, retain the one-output-buffer fast path for
        already-matching operands, preserve complete broadcasting, arbitrary layouts, integer wrapping and failure
        diagnostics, low-precision re-encoding, complex magnitude and stable division, and exact extremum selection.
  - [x] P9a2m: migrate `Sin`, `Cos`, `Atan2`, `Exp`, `Log`, `Sqrt`, `Rsqrt`, `Tanh`, `Logistic`, `Erf`, `Pow`, `Sign`,
        `Floor`, `Ceil`, and `Round` to direct typed codecs. Preserve each operation's exact real/complex dtype
        contract, low-precision re-encoding, IEEE special values and ties-to-even rounding, mixed-type promotion and
        complete broadcasting for binary operations, arbitrary physical layouts, and existing diagnostics.
  - [x] P9a2n: confirm complex construction/accessors already use direct typed codecs, then migrate zero, one,
        zero-like, one-like, fill, and iota construction to direct typed codecs. Make `Fill` generic over sealed typed
        host elements rather than accepting the scalar backend's value representation; delete the scalar-sequence
        encode/decode bridge, migrate its remaining tests to typed elements, and preserve payload-free, sub-byte,
        low-precision, complex, arbitrary-layout, dynamic-shape-rejection, and exact diagnostic semantics.
- [x] Migrate structural operations, including broadcast, reshape, transpose, slice/update, pad, concatenate,
      gather/scatter, reduce, sort, dot/attention, collectives, and control flow.
  - [x] P9a2c: migrate broadcast, transpose, reshape, static and dynamic slice/update, pad, and concatenate to direct
        layout-aware byte copies, retaining exact validation order and bulk-copying physically contiguous selections.
  - [x] P9a2d: migrate `Select`, Boolean concretization, and batched while-predicate reduction/masking to direct,
        layout-aware byte routing. Preserve branch encodings exactly and support complete NumPy-style broadcasting.
  - [x] P9a2f: migrate dynamic slice/update, gather, and scatter index decoding to direct typed integer storage,
        including signed and unsigned sub-byte indices and arbitrary layouts. Migrate gather payload routing and the
        eager sort index passenger to direct layout-aware construction/copying without payload-sized intermediates.
  - [x] P9a2g: migrate `to_f64s`, approximate equality, and sort key ranking to direct typed, layout-aware decoding.
        Support every ordered key type, including signed and unsigned sub-byte integers, while retaining stable IEEE
        total ordering for all floating-point formats and exact componentwise complex approximation.
  - [x] P9a2h: migrate scatter payload combining and reduction to direct typed, layout-aware kernels. Share one private
        arithmetic-element capability across both families; support every numeric codec, sub-byte modular arithmetic,
        low-precision floating-point formats, complex sum/mean, Boolean reductions, arbitrary physical layouts, and
        structural-zero payloads without a payload-sized `Scalar` intermediate.
  - [x] P9a2i: close the existing JAX extremum-semantics gap before moving to dot/attention and collectives. Match
        `lax.reduce_min`, `lax.reduce_max`, `lax.scatter_min`, and `lax.scatter_max` for Boolean extrema, IEEE-754 NaN
        and signed-zero behavior, lexicographic complex ordering, dtype-specific reduction identities, and empty-axis
        reductions. Keep reference and XLA execution identical, including complex identity lowering, and pin each
        behavior with exact type-rule, reference, StableHLO, and execution tests.
  - [x] P9a2j: migrate generalized dot's same-element-type contraction to a direct typed, layout-aware kernel with no
        operand-sized temporary payload. Preserve arbitrary batching and contracting dimensions, integer wrapping,
        low-precision and complex arithmetic, empty contractions, and arbitrary physical input layouts. Preserve
        preferred accumulation through its canonical operand-conversion composition followed by the same direct
        kernel. Delete the superseded flat-`Vec` evaluator; verify attention through its existing composition and
        record that collectives own no local reference-array payload kernel requiring migration.
- [x] Route raw-buffer access through the Phase 9a1 addressing contract and reuse its rectangular traversal where it
      removes today's duplicated row-major stride, odometer, block-copy, and block-replacement logic. Keep operation
      semantics in their kernels; the addressing layer owns only checked logical-index-to-byte-range mapping.
- [x] Preserve exact equality, numeric approximation, display, Boolean concretization, and indexing semantics directly
      over encoded bytes.
- [x] Gate: all reference-backend and transform tests pass; representative kernels add no allocation-count slope beyond
      output allocation; no production array kernel depends on `Scalar`; and unchecked byte-offset arithmetic is not
      duplicated across kernels.

Phase 9a3 — exact XLA literals:

- [x] Replace per-element `Scalar` matching and typed-vector reconstruction with logical traversal of layout-aware
      `Array` storage. Decode only when an MLIR typed helper requires it; pass raw physical bytes directly only when
      their layout already matches the literal API's required ordering.
- [x] Pack Boolean and sub-byte integers at the lowering boundary and cover I1/I2/I4/U1/U2/U4 constants.
- [x] Add exact-bit StableHLO and execution tests for low-precision floats, signed zero, preserved NaN payloads, wide
      integers, complex values, empty tensors, and sub-byte integers.
- [x] Measure literal construction, lowering allocations, StableHLO size, compile time, and runtime against Phase 9a0.
- [x] Gate: no literal round-trips through `f64`; exact values lower and execute identically on CPU, with CUDA coverage
      where supported by the existing backend matrix.

Phase 9a4 — retire the scalar program universe:

- [ ] Migrate useful `DataType`-universe transform tests to rank-zero arrays. Use a narrow test-only fixture only where
      a test genuinely verifies universe-neutral machinery and an array would obscure that contract.
  - [x] Migrate the foundational atom, builder, operation-formatting, program, region-graph, capture-region, and
        interpretation-replay fixtures. Make the shared test-only region operation use `ArrayType` so its complete
        consumer graph uses rank-zero arrays without introducing another test value universe.
  - [x] Migrate the generic context, tracing, partial-evaluation, batching, differentiation, custom-derivative, and
        rematerialization fixtures.
    - [x] Migrate generic context and tracing fixtures.
    - [x] Migrate partial-evaluation and batching fixtures.
    - [x] Migrate differentiation, custom-derivative, and rematerialization fixtures.
      - [x] Migrate elementwise differentiation, custom-derivative, and rematerialization fixtures.
      - [x] Migrate forward-mode differentiation fixtures.
      - [x] Migrate reverse-mode differentiation fixtures.
  - [x] Migrate remaining operation-local scalar fixtures or delete coverage already covered by array tests.
    - [x] Migrate constants, comparison, logical, conversion, and stop-gradient fixtures.
    - [x] Migrate control-flow fixtures.
    - [x] Migrate mathematical and complex-number fixtures.
      - [x] Migrate arithmetic, ordering, clamping, sign, and rounding fixtures.
      - [x] Migrate transcendental and special-function fixtures.
      - [x] Migrate complex construction and projection fixtures.
- [x] Replace scalar-mode gradient macro coverage with rank-zero array coverage and delete scalar-only macro branches.
- [ ] Delete `ScalarOperation`, `ScalarTracingContext`, scalar-domain capabilities/transforms, the backend module and
      exports, scalar-only doctests, and the obsolete scalar-domain compile-fail fixture.
  - [x] Migrate or delete the final scalar-backed capture and interpretation fixtures.
  - [ ] Migrate or delete the final scalar-backed program fixtures.
  - [ ] Migrate or delete the final scalar-backed macro, compilation, and benchmark fixtures, then remove the obsolete
        scalar-domain compile-fail fixture.
  - [ ] Delete the scalar backend module, exports, scalar-only doctests, production specializations, and stale
        documentation references.
- [ ] Delete or privatize-and-rename any surviving transient element helper according to the prototype decision. No
      item named `Scalar` and no standalone scalar `Value`/domain may remain in production code.
- [ ] Update testing guidance to name rank-zero `Array` as the scalar-semantics reference.
- [ ] Gate: targeted searches find no retired scalar identifier or path outside historical plans, and production/test
      line counts materially decrease.

Phase 9a5 — closure:

- [ ] Run full core, macro integration/compile-fail, XLA, affected doctest, allocation, CPU, and available CUDA suites
      with 300-second command timeouts.
- [ ] Re-run Phase 9a0 measurements. Clone remains payload-copy-free and constant-time in array size, and other
      regressions remain within the master plan's evidence-based thresholds or receive explicit approval.
- [ ] Review the complete diff for redundant codecs, duplicate conversions, compatibility residue, unnecessary trait
      surface, and avoidable allocation; simplify before closing the review unit.
- [ ] Record the completed review here and leave the module/type/sharding hierarchy move as the next isolated unit.

Abort if the representation requires unsafe unaligned typed views, per-element boxing, two authoritative payloads, or
cannot round-trip exact bits. Also stop if scalar-test migration reveals a genuinely independent non-array production
use case rather than deleting it by assertion.

### Phase 10: persistence and measured performance closure

- [ ] Verify cache keys remain alpha-invariant and distinguish semantically different dimension graphs.
- [ ] Re-run Phase 0 graph-size, allocation, compile-time, memory, executable-size, and runtime measurements.
- [ ] Gate: no performance regression exceeds the existing evidence-based thresholds without explicit approval.

### Phase 11: deletion and minimality gate

- [ ] Delete every item in the deletes ledger.
- [ ] Treat `u/eaplatanios/wip/dimensions-remainder` as retired historical state at `12398a196`; it stopped being
      reconciled after P3c and is not a valid representation of pending work. Prove final completeness from the current
      integration tree, this plan's residual searches, and the already-complete 142-path archive-disposition table.
      Do not merge the stale remainder or alter the immutable archive to make bookkeeping appear current.
- [ ] Verify `origin/u/eaplatanios/archive/dimensions-wip-2026-07-24` still points to the recorded bootstrap commit.
- [ ] Record a final current-tree review entry in this plan. The historical delivery ledger may be annotated as retired
      after P3h, but must not be backfilled with invented increment/remainder commits.
- [ ] Remove dead imports, helper traits, macros, tests, documentation, and allowances made obsolete by the cleanup.
- [ ] Run a repository-wide residual search for retired identifiers and classify every match.
- [ ] Compare production/test/generated line counts against Phase 0.
- [ ] Require a material reduction in the combined non-test source of the array-program projection, batching, and
      differentiation adapters. The target is at least 40%; if the result is smaller, stop for architectural review
      rather than declaring success from passing tests alone.
- [ ] Require the final total production line count to be lower than the Phase 0 baseline. Test additions are reported
      separately and may grow.
- [ ] Require zero hidden reconstruction paths and zero dual semantic operation contracts.
- [ ] Run `cargo fmt --all -- --check`, `git diff --check`, the core/macro/XLA focused suites, all doctests affected by
      moved public APIs, and the full workspace all-target suite serially.
- [ ] Review the final diff by subsystem and ask whether every remaining changed line is necessary for the target
      semantics.
- [ ] Gate: a staff-level review confirms simpler dependency direction, one source of truth, no compatibility layer,
      no redundant abstraction, and no unexplained bloat.

### Phase 12: close tier-3 semantics around the `dimension_from_scalar` gateway

Phases 12–14 extend the finished cleanup from input-derived (tier-2) to data-dependent (tier-3) dynamism. Begin only
after the Phase 11 gate: tier 3 is a provenance-policy relaxation over a *stable* architecture, and starting it over
moving transform rules would repeat the retrofit failure mode this phase order exists to avoid. The design bet being
cashed here is that tier 3 requires no new type-system, program, or transform machinery. The entry point already
exists: P3d landed `DimensionFromScalarOperation` as the sole checked `rank-0 integer array -> dimension` gateway,
with a declared fresh identity plus bounds, eager bounds-checked execution, partial-evaluation fold/residualize
behavior, and the mapped-batching rejection diagnostic. A gateway output types exactly like a derived arithmetic
dimension (fresh identity plus bounds), and the P6a `LinearCallOperation` residual contract threads dimension values
through differentiation without caring where they came from. This phase turns those properties from
believed-by-construction into verified-by-fixture and extends the effects-model decision completed in Phase 7 across
the remaining tier-3 transforms and fixtures. The
non-negotiable invariants are unchanged: rank stays static, no expression trees or witnesses, no side tables, no
ambient environments, and the graph remains the complete source of data dependencies.

- [x] Decide the gateway's effects model. `DimensionFromScalarOperation` declares `Effect::OrderedAssertion` in core,
      matching residual requirements and ensuring unknown bounds checks survive DCE with deterministic same-class
      ordering. Partial evaluation may still fold a known valid conversion or report a known invalid conversion before
      lowering. Generated scan/while pullbacks remain valid because transposition treats assertion checks as replayable
      primal preconditions rather than rejecting them as I/O.
- [ ] Add a retained-JIT cache-identity test proving one compiled specialization serves multiple runtime extents of a
      data-derived dimension: structural equality, hashing, and cache identity must not observe the data dependence.
- [ ] Verify gateway-defined variables need no boundary `TypeRefinements` entry: they are internal identities
      established by their producing instruction under the existing structural-closure rules. Cover closure, import,
      alpha-renaming, and repeated splicing.
- [ ] Differentiation: dimensions remain nondifferentiable, and the linear-call residual contract must carry
      gateway-defined dimensions through linearization, transposition, and nested JVP without modification. Add a
      rendered-IR fixture in which a data-derived extent is a visible residual edge of a `linear_call`.
- [ ] Batching: pin the existing tier-3 MVP policy with fixtures — a gateway whose scalar operand is replicated
      produces ordinary replicated dimension authority; a mapped operand keeps its exact typed rejection diagnostic,
      updated to name Phase 14 raggedness as the missing capability.
- [ ] Control flow: verify the existing carry-type equality checks reject shape-varying loop-carried state with exact
      diagnostics (a fresh per-iteration variable cannot instantiate the declared carry type). Bounds-widened
      loop-carried extents are an explicit non-goal; record the rejection fixture rather than designing widening.
- [ ] Confirm the Phase 8 authoritative operation declaration covers the gateway so it acquires generated dispatch,
      conversions, and classification like every other dimension operation.
- [ ] Update the `DimensionType` motivation rustdoc in `types/dimensions.rs` so the provenance story describes the
      tiers and names the gateway as the single data-to-dimension boundary, and update the `ArrayIrType`
      cross-reference if its wording changes.
- [ ] Add JAX comparison fixtures for eager and staged `n = count(mask); take(x, n)`-shaped programs that JAX rejects
      (`ConcretizationTypeError`) and Ryft accepts eagerly and stages symbolically. Compiled execution may reject with
      an exact "requires Phase 13 bounded data-dependent lowering" diagnostic until Phase 13 lands.
- [ ] Gate: tier-3 programs interpret eagerly end to end; staged tier-3 programs type-check, batch (replicated),
      differentiate, and partially evaluate; the bounds check provably survives DCE with deterministic ordering; every
      unsupported surface fails with an exact diagnostic rather than silently mis-executing; and no expression trees,
      side tables, or ambient environments were added.

### Phase 13: bounded data-dependent compiled execution

This is the dominant tier-3 cost and the piece most exposed to backend maturity: XLA's support for data-dependent
dynamism is uneven, which is part of why JAX's own effort stalled. The physical model is fixed — bound-shaped buffers
carrying smaller logical extents — but the encoding route is an explicit measured decision, not an assumption. Phase 7
already owns the gateway's own compiled lowering (its range check as an ordered assertion); this phase makes the
*rest of the operation set* correct over data-derived extents and validates the explicit bounded physical ABI through
the plugin's `PadToStatic` legalization.

- [ ] Decide the compiled route for operations consuming data-derived extents, on measured evidence: (a) XLA bounded
      dynamism through the existing bounded-input ABI, `set_dimension_size`, and `PadToStatic` machinery; or (b) fully
      static bound-shaped StableHLO plus explicit Ryft-generated masks. Prototype (a) first because the ABI already
      exists; record CPU and CUDA coverage evidence before committing, and fall back to (b) per backend where (a) is
      unsupported rather than globally.
- [ ] Require a finite upper bound at the gateway for any program that reaches compilation; reject unbounded
      data-derived dimensions at lowering with an exact diagnostic naming the variable and its bounds.
- [x] Confirm the Phase 7 gateway lowering composes with the Phase 12 effects-model coverage: the range check rides
      the per-class `OrderedAssertion` token chain with its exact diagnostic and deterministic same-class ordering.
- [ ] Complete the per-operation padding-discipline inventory started in Phase 7 and record it in this plan as the
      authoritative table: padding-oblivious (elementwise, reshape-within-bounds), mask-required (reductions,
      argmin/argmax, cumulative and windowed operations), or zero-padding-required (contractions, convolutions).
- [ ] Implement the padding rules for the supported operation matrix so padding garbage is unobservable in results.
      Every unclassified or unsupported operation must reject lowering of data-derived extents with an exact
      diagnostic naming the operation; silent truncation or garbage propagation is an abort criterion.
- [ ] Run CPU (and CUDA where backend support permits) eager/JIT parity for a data-dependent golden set including the
      Phase 12 fixtures, proving one compiled specialization serves multiple runtime extents.
- [ ] Gate: bounded data-dependent programs compile and execute correctly on supported backends, padding effects are
      unobservable in every supported operation's results, unsupported operations fail before execution with exact
      diagnostics, and the route decision is recorded with its measured evidence.

### Phase 14: ragged batching for data-dependent extents

The last and largest tier-3 unit: `vmap` over a data-derived extent yields per-batch-element extents, i.e. ragged
intermediates — the exact problem JAX's team named as the hard transform case, and the one place where djax built
substantial machinery (`RaggedAxis`). Ryft has two structural advantages to reuse: the recursive batching meta stack
composes nested axes already, and the relaxed-while-predicate work established the consumer-owned-masking pattern.

- [ ] Confirm and record the concrete motivating workload (e.g., ragged/variable-length batch items without API-level
      padding waste) before implementation so the supported operation surface is demand-shaped rather than
      speculative. If the owner explicitly approves deferral instead, record it here and re-scope the tier-3 exit
      criteria; do not defer silently.
- [ ] Extend `BatchingPolicy` with a ragged mapped-extent representation: a per-element extent vector (dimension SSA
      indexed along the batch axis) plus the declared bound as the packed physical extent, with masks owned by
      consumers. Raggedness lives on the batch carrier only; do not add it to `Type` and do not build a parallel
      batching context/tracer tower.
- [ ] Batching rule for the gateway: a mapped scalar operand now yields a ragged mapped dimension instead of the
      Phase 12 rejection; replicated behavior is unchanged.
- [ ] Ragged rules for the elementwise blanket, masked reductions, and the shape-carrying `linear_call` carrier
      (batch both attached regions with replicated residual extents and ragged linear operands, reusing the
      swap-stable P6 batching rule). Every operation without a ragged rule keeps an exact typed diagnostic.
- [ ] Prove nested `vmap` over ragged extents composes through the recursive batching meta stack.
- [ ] Control flow: ragged trip counts remain rejected with an exact diagnostic; record the fixture.
- [ ] Gate: the ragged surface covers the recorded workload end to end with static and dynamic tests, every
      unsupported path has an exact diagnostic, and no parallel batching tower or type-level raggedness was
      introduced.

## Verification matrix

- [ ] Static array-only primitives never inspect the heterogeneous storage sum.
- [ ] Dimension-only primitives never inspect array variants.
- [ ] Mixed inference covers fixed, repeated, optional, and segmented operands.
- [ ] Each mixed operation defines its explicit dimension operands once through its inference signature, with no
      parallel payload metadata or schema.
- [ ] `dimension_size` has exactly one result kind.
- [ ] Dimension-to-data conversion occurs only through explicit gateways.
- [ ] Shape operations retain all explicit dimension operands through tracing, import, PE, batching, AD, and lowering.
- [ ] No transform reconstructs a dimension operand from output metadata or an ambient source array.
- [ ] Eager projection adds no heap allocation for concrete array/dimension values.
- [ ] Eager projection neither deep-copies the reference `Array` payload nor relies on an undocumented cheap-clone
      assumption.
- [ ] Staged projection preserves SSA atom identity.
- [ ] Mapped dimension authority and sharded dimension authority remain rejected.
- [x] Dimension tangents/cotangents remain absent or structural zero.
- [x] Required primal dimension values travel as ordinary differentiation residual SSA values.
- [x] No primal operation payload contains `transpose_dimension_variables` or an equivalent residual manifest.
- [x] Dynamic batching alignment consumes an explicit dimension value.
- [ ] Requirement effects survive every transform and lower in deterministic order.
- [ ] Nested condition, while, scan, custom derivative, and rematerialization regions carry dimensions correctly.
- [ ] Repeated boundary readers do not become duplicate producers.
- [ ] Fresh internal dimensions have one producer and dominate every reference.
- [ ] Alpha-equivalent programs share cache identity; live permutations and different graphs do not.
- [ ] Exact diagnostics match the baseline.
- [x] Bounded dynamic ABI, CPU, and CUDA behavior match the baseline.
- [ ] Behavioral JAX parity and Ryft-exceeds-JAX cases remain intact.
- [x] Toy third-kind tests demonstrate that generic program/context/projection machinery is closed to modification.
- [ ] (Tier 3) Data-to-dimension conversion occurs only through the checked `dimension_from_scalar` gateway.
- [ ] (Tier 3) Data-derived dimensions never enter structural type identity or retained-compilation cache keys.
- [ ] (Tier 3) Gateway bounds checks retain `OrderedAssertion` ordering through every transform and lowering.
- [ ] (Tier 3) Data-derived extents ride the linear-call residual contract through differentiation unchanged.
- [ ] (Tier 3) Every operation without data-dependent lowering or ragged batching support fails with an exact
      diagnostic before execution; padding effects are unobservable on every supported path.

## Abort and reassessment criteria

Stop the current phase and revise this plan if any of the following occurs:

- the associated-type `Operation` prototype creates unstable trait solving, extreme `rustc` memory, or more wrapper
  code than it deletes;
- eager projection requires deep-cloning an array payload or assumes without proof that every backend value is cheap
  to clone;
- typed projection requires semantic state beyond the parent context;
- an array-only rule still needs ambient dimension lookup after its operation classification is corrected;
- removing the homogeneous shape language forces duplicate semantic implementations rather than direct outer binding;
- transform policy leaks batching/differentiation hooks back onto `Type`;
- mixed signature generation becomes a second general operation DSL with more code than direct inference contracts;
- structural identity ownership cannot represent a valid operation without weakening closure soundness;
- differentiation requires a new operation-specific dimension residual field after the generic residual migration;
- inference, transforms, eager interpretation, and lowering disagree about the operation's direct operand order;
- diagnostics regress to generic assertion/type errors;
- a phase increases production code after its temporary coexistence code should have been deleted;
- a broad Rust check causes extreme memory growth; reduce generic obligations before rerunning;
- any backend path restores shape expression evaluation, host readback, or reconstruction;
- the toy third-kind test still requires edits throughout generic program and transform machinery;
- a tier-3 phase needs changes to generic type-system, program, projection, or residual machinery beyond the single
  `dimension_from_scalar` gateway and transform-owned policy — that falsifies the bet Phase 12 exists to cash and
  requires a design review, not incremental patching; or
- any supported data-dependent lowering path lets padding garbage become observable or silently truncates logical
  extents instead of failing with an exact diagnostic.

## Exit criteria

The cleanup is complete only when:

1. Runtime dimensions remain ordinary SSA values in one program graph.
2. Each operation payload has one compiler-enforced semantic contract.
3. `dimension_size` always returns a dimension and the data gateway is distinct.
4. Dynamic zero, one, and iota construction binds the constructors' own mixed stored-type contracts; fill values are
   ordinary array SSA inputs to broadcast; transform-generated values use structural or operand-relative construction
   where possible.
5. Every shape-carrying operation consumes explicit dimension operands through one direct operation signature.
6. No ambient dimension/source-array replay environment exists.
7. No complete homogeneous implicit-shape array program exists in production.
8. Eager projection borrows or transfers ownership without copying payloads.
9. Heterogeneous storage matching is limited to projections, outer dispatch, and genuinely mixed rules.
10. Array-only/dimension-only rules operate over their homogeneous types.
11. Transform behavior is driven by transform-owned value-kind policy.
12. Primal dimension values needed by transpose are ordinary transform residuals, never operation payload witnesses.
13. Identity ownership is structural and has one source of truth.
14. Operation classification and mechanical dispatch have one declaration.
15. Public module placement reflects language semantics versus backend implementation.
16. Exact behavior, diagnostics, cache identity, ABI, CPU/CUDA execution, and performance gates pass.
17. Production code is materially smaller, with the special-purpose adapter modules reduced by at least 40%.
18. Adding a third nondifferentiable member kind does not require another core-wide architecture sweep.

Full tier-3 dynamism (Phases 12–14) is additionally complete only when:

19. Data-derived dimension authority exists through exactly one checked gateway with mandatory bounds and ordered
    assertion semantics, and remains unrepresentable everywhere else.
20. Tier-3 programs interpret eagerly, stage, batch, differentiate, and partially evaluate with unchanged
    type-system, program, and residual machinery — the gateway is the only new operation.
21. Bounded data-dependent programs compile and execute on supported backends with padding effects unobservable in
    every supported operation's results.
22. Ragged batching covers the recorded motivating workload (or its deferral is explicitly owner-approved and
    recorded), with exact diagnostics on every unsupported path.
23. No tier introduced expression trees, witnesses, scopes, substitution, side tables, or ambient environments.

## Review

Execution notes, per-phase summaries, measurements, superseded decisions, and verification evidence must be recorded
here as the plan is executed. Do not check an item solely because a test happens to pass; record the implementation or
deletion that satisfies it.

### Resumption audit: 2026-07-28

The repository, not the expired side-chat history, establishes the current boundary:

- `u/eaplatanios/dynamic-shapes` and its remote both point to
  `21c7442fee3e94eb422b440cc25b479691526df5`;
- the immutable archive still points to `770e77d001547c72150a44843c170ea6417ab41e`;
- the mutable remainder is stale at `12398a196d96a61088fb2d81000c18ce6fd26f40` and differs from integration by
  109 files, 34,626 insertions, and 19,263 deletions;
- the staging worktree is still parked on the obsolete P3d increment;
- the delivery ledger stops after P3h, while commits through P3i are present on integration; and
- the owner checkout has 13 modified paths: this plan plus 12 P3j prototype source/test files, for 987 source/plan
  insertions and 91 deletions before this audit's plan edits.

The committed architecture has completed Phases 0–2 and P3a–P3i. Repository searches confirm that
`ArrayContextView`, `DimensionContextView`, `with_source_array`, `bind_replayed`, `runtime_dimension_variables`,
`OutputIdentityRole`, and `transpose_dimension_variables` have zero occurrences. The remaining `with_dimensions`
identifier is only reshape permutation configuration. Phase 4 is nevertheless not complete:
`ReshapeDimensionExpression` has 80 occurrences, `LegacyReshapeOperation` 68,
`LegacyBroadcastOperation` 81, `DynamicBroadcastOperation` 51, and the complete homogeneous `ArrayOperation` family
remains a production transform/backend language. Phase 5's separate `ArrayIrBatchingContext`,
`ArrayIrBatchingTracer`, and `ArrayIrBatchableOperation` also remain, as do the composite differentiation
dispatcher and the five transitional dual-contract payloads listed in the diagnosis. `Operation` is still generic as
`Operation<T>`; Phases 8–11 have not begun.

The current unstaged P3j prototype was reviewed path by path:

- **Retain after correction:** the position-aware nullary constructor guard; the shared variant-owned dynamic
  constructor inference; canonical `ZeroOperation<ArrayType>` routing; the flat `DynamicZero` variant; eager
  materialization from compact dynamic-axis operands; replicated-only batching; structural-zero extent cotangents;
  direct upper-bound allocation plus `stablehlo.set_dimension_size`; the mixed static/dynamic axis pairing fix; and
  their focused tests.
- **Delete rather than generalize:** `DimensionType::known_extent`, `with_known_extent`, observed extents injected by
  `ArrayIrValue::r#type`, `ArrayIrTypeRefinements`' dependence on those type payloads, the extra
  `DimensionValue` validation, and all equality/hash/display/renaming tests for this field. These make one runtime
  value part of the structural type and would specialize caches per extent.
- **Re-evaluate before retaining:** the global fused-JVP “reuse a zero primal as its tangent” rule and the Jacobian
  coordinate-validation reorder. Both are plausible independently, but the current comments justify them as
  workarounds for constructor/type behavior. They need exact standalone regression tests after the boundary correction
  or they should be removed.
- **Remove as residue:** the UFCS-only `fill` and `iota` test changes. Their mixed vertical slices were deliberately
  deferred and the edits no longer resolve an ambiguity in the retained design.

The corrected boundary design uses machinery already present. `Region::type_identity_signature` proves that every
reference-position instruction result identity is consumed by that instruction or defined by it; otherwise closure
fails. `TypeRefinements::validate` can therefore allow a static output to establish the first concrete refinement for
an identity in the formal input identity set as well as an internally defined identity. The mixed eager operation
remains responsible for constructing or validating the concrete output from its explicit dimension operand. This
deletes the need for value-payload inspection, a boundary-refinement provider, and any new `Type`, `Typed`, or `Value`
surface. Store the closed input/internal identities so the complete authority slice is available without a new
per-interpretation concatenation allocation.

Current verification evidence:

- `cargo test -p ryft-core --lib` passed all 990 tests;
- `cargo test -p ryft-xla --lib` passed 408 tests with one intentional benchmark ignored;
- the dynamic-zero structural and CPU PJRT tests ran inside the XLA suite;
- `cargo fmt -p ryft-core -p ryft-xla -- --check` passed; and
- `git diff --check` passed.

This evidence proved that the first prototype was internally consistent, not that it was architecturally complete. At
that audit, the next review unit was strictly P3j boundary correction and shaped-zero closure:

1. delete the `known_extent` path and implement input/internal refinement authority using structural identity closure;
2. add dynamic eager reshape/broadcast/zero, conflicting-output, dangling-edge, and retained-specialization tests;
3. reassess and minimize the forward/Jacobian changes;
4. rerun core, macro, XLA, doctest, formatting, and residual gates; and
5. update this review with the final retained diff before beginning `One`, `Fill`, `Iota`, Phase 4, or Phase 6.

### P3j second review: 2026-07-28

The owner revised the prototype after the resumption audit. The current checkout contains this plan plus 13 modified
source/test paths (874 source insertions and 107 source deletions relative to `HEAD`). The original architectural
blocker is resolved:

- `known_extent` and `with_known_extent` have zero source occurrences;
- `DimensionType` is again exactly identity plus bounds, with exact extents derived only from singleton bounds;
- `ArrayIrValue::r#type` no longer incorporates runtime observations;
- `TypeIdentitySignature` stores one ordered identity vector with an input/internal split, so the complete authority
  set is available without concatenating or cloning vectors during interpretation;
- output refinement admits both input-owned and internally defined closed identities and rejects inconsistent repeated
  concrete observations; and
- the stale `fill`/`iota` changes are gone.

The variant-owned `DynamicZero` vertical slice remains directionally correct: canonical `From` routing distinguishes
identity-free homogeneous zeros from reference-bearing mixed zeros; inference requires one matching dimension operand
per dynamic axis; eager execution materializes the concrete shape; PE folds only fully known calls; batching rejects
mapped extent authority; differentiation preserves extent operands as nondifferentiable structure; and XLA allocates
the finite upper-bound buffer before applying one `stablehlo.set_dimension_size` per dynamic axis.

The second review found these remaining blockers:

1. `TypeRefinements::validate` now receives `&TypeIdentitySignature<T::Identity>`, but validation uses only
   `identities()`. Pass the identity slice directly so the generic type-refinement contract does not depend on the
   region-metadata container or its input/internal partition. This also restores the allocation-free `&[]` path for
   refinement-free types and removes repeated empty-signature construction in tests.
2. The eager `DynamicZero` rule consumes extents without checking them against the corresponding stored variable's
   bounds. `EagerContext::bind` does not run inference, and `Program::interpret_with` checks counts but not
   intermediate result types, so relying on final program-output refinement misses malformed direct calls and
   malformed constructor results consumed internally. Check operand kind/count and stored bounds before allocation;
   do not require literal identity equality because runtime program inputs may carry alpha-renamed identities.
3. `test_array_program_dimension_values_share_one_abstract_type` proves equality, hashing, and display only. It does
   not prove that retained dispatch, tracing, lowering, and compilation reuse one specialization while two different
   extents produce different logical output shapes. The named retained-specialization gate remains open.
4. The slice has no non-exact identity-instantiation/import acceptance test and no transpose acceptance test, despite
   implementing both renaming and structural-zero extent cotangents.
5. The mixed static/dynamic XLA fixture uses `2` for both its static middle axis and trailing runtime extent. It catches
   the historical out-of-range operand index, but distinct values would make output shape/length also prove correct
   axis pairing.
6. The generic fused-JVP zero-primal reuse is semantically sound for the only operations currently reporting
   `is_zero`, and it avoids materializing an invalid nullary dynamic zero at the fused output boundary. The Jacobian
   reordering preserves existing exact `NonFiniteCoordinateSpace` tests, but remains symptom-oriented duplication
   pending Phase 6's structural-zero cleanup. Keep neither merely because the broad suites pass: document the temporary
   dependency or replace it with the narrower structural fix.
7. Two comments in `interpretation.rs` still say only internally defined identities may establish output facts. Update
   them to describe the now-implemented complete closed boundary authority.

Verification for this second review:

- `cargo test -p ryft-core --lib`: 991 passed;
- `cargo test -p ryft-xla --lib`: 408 passed and one timing benchmark ignored;
- `cargo test -p ryft-macros`: 53 passed;
- `cargo test -p ryft-macros-tests`: operation tests passed, but the parameter trybuild suite failed because its
  expected `$N others` diagnostic omits already-committed composite `Parameter` implementors; this appears to be a
  stale snapshot rather than a P3j semantic failure, but the full gate is not green until it is deliberately resolved;
- `cargo test -p ryft-core --doc`: 58 passed and 16 ignored;
- `cargo fmt -p ryft-core -p ryft-xla -- --check`: passed; and
- `git diff --check`: passed.

P3j therefore advances from “architecturally incorrect” to “correct core design with bounded completion work.” Finish
the seven items above before marking the phase complete or beginning the `One`, `Fill`, and `Iota` slices.

### P3j third review: 2026-07-28

The owner addressed the material findings from the second review:

- `TypeRefinements::validate` now receives only `&[T::Identity]`, and interpretation passes the closed identity
  signature's borrowed identity slice;
- `TypeIdentitySignature::new` now accepts the complete ordered identity vector plus `input_count`; region closure
  uses that same vector as its dominance environment and transfers it directly into retained metadata, eliminating
  the cloned input-identity vector, the parallel internal-identity vector, and their duplicate identity cloning;
- the position-aware nullary identity-reference check is inlined in `ZeroOperation` while zero is its sole production
  caller. `One`, `Fill`, and `Iota` remain intentionally unguarded until their complete mixed vertical slices land;
  extract shared machinery only when that migration creates a second real caller;
- eager `DynamicZero` checks each concrete extent against the corresponding stored dynamic variable's bounds before
  allocating, while continuing to accept alpha-renamed interpreted inputs;
- a test-local retained JIT domain calls the same program with extents `3` and `4`, observes different logical output
  shapes, and proves one trace, one lowering, one compilation, one cache miss, and one cache hit;
- transposition has a focused regression proving that the output cotangent is ignored and the extent receives a
  structural-zero cotangent;
- the fused-JVP zero-primal reuse has an independent shaped-zero regression, while the retained Jacobian validation
  order now explicitly records its transitional Phase 6 dependency; and
- the stale interpretation comments now describe complete closed input/internal boundary authority.

No new implementation correctness defect was found in the revised core or XLA paths. Three acceptance details still
prevent closing P3j:

1. `test_array_program_dynamic_zero_alpha_renamed_instantiation` directly renames the operation and separately
   interprets a program with an alpha-renamed boundary value, but it does not invoke program instantiation,
   cross-program import, or `splice_program`. Add one real cross-program vertical test so the operation's renaming
   implementation is exercised through the production import machinery.
2. The XLA execution fixture now distinguishes its static middle axis (`2`) from its runtime extents, but both runtime
   extents are `3`. Make all three extents pairwise distinct so reversing the compact dynamic operands changes the
   observed shape and output length.
3. The full macro integration gate still fails only because
   `crates/ryft-macros-tests/tests/parameters/error_structs.stderr` has a stale compiler-generated implementor list.
   Refresh that snapshot deliberately after reviewing the actual diagnostic. Also remove the unrelated blank-line-only
   diff in `backends/dimensions.rs` before landing the increment.

Verification for this third review:

- `cargo test -p ryft-core --lib`: 994 passed;
- `cargo test -p ryft-xla --lib`: 408 passed and one timing benchmark ignored;
- `cargo test -p ryft-macros`: 53 passed;
- `cargo test -p ryft-macros-tests`: operation tests passed, while the parameter trybuild suite has the single stale
  snapshot failure described above;
- `cargo test -p ryft-core --doc`: 58 passed and 16 ignored;
- `cargo fmt -p ryft-core -p ryft-xla -- --check`: passed; and
- `git diff --check`: passed.

P3j is now a correct and nearly complete vertical slice. Close only the three bounded items above, rerun the same
gates, review the final source-only diff for minimality, and then begin the separate `One`, `Fill`, and `Iota` slices.

### P3j dynamic-one review: 2026-07-28

The corrected dynamic-zero unit was committed at `6d47d6996209b99a512f18471bc441e65c1d722b`. The next isolated unit,
specified and reviewed in `.tasks/plan_p3j_dynamic_one.md`, is complete and unstaged:

- identity-bearing ones use `DynamicOne(OneOperation<ArrayType>)` with one explicit dimension operand per dynamic
  output axis, while identity-free ones retain the unchanged homogeneous representation;
- zero and one share the position-aware nullary guard, eager concrete-shape materializer, replicated batching arm,
  transpose arm, and bounded constant lowering without introducing a shaped-constructor wrapper;
- dynamic one's JVP stages a dynamic zero tangent from the same extent SSA operands, and the generic all-zero fast path
  now preserves operation rules when reference-position identities make nullary tangent materialization impossible;
- inference, eager execution, tracing, PE, import/identity renaming, batching, JVP, transpose, StableHLO lowering, and
  CPU PJRT execution have focused tests; and
- the core, serial XLA, macro integration, doctest, formatting, and diff gates pass. The ordinary parallel XLA suite
  exposed one pre-existing live-array telemetry race; its isolated rerun and the complete serial suite pass.

The subsequent forward-mode review follow-up replaced callback-based type-identity visitation and the
`first_identity_reference` convenience method with one allocation-free borrowed `Type::identities` iterator. It also
removed the direct-context operation clone from the all-zero fast path by retaining only the reusable zero-output
indices, and added a regression proving DynamicZero reuses its shaped primal SSA value as its tangent. The complete
core, macro integration, doctest, and serial XLA gates remain green; exact results are recorded in the focused plan.

At that checkpoint, neither the then-existing fill operation nor `IotaOperation` changed; the now-completed dynamic
fill review followed as the next isolated unit.

### Plan revision: constructor contracts without a wrapper

The shaped-constructor wrapper (P3j Delivery A) was replaced after review by the stored-type-authoritative design:
constructor payloads keep their possibly-dynamic output `ArrayType`, and dynamic axes consume explicit
identity-validated dimension operands. This deleted `ShapedConstructorOperation`, `ArrayConstructorOperation`, and
the template-shape representation entirely, and resolved the template-shape canonicalization question by making it
unrepresentable. Jacobian forward and reverse entry points now validate input and output coordinate spaces before
linearization and pullback, preserving their precise `NonFiniteCoordinateSpace` diagnostics ahead of the new
constructor rule.

A second review pass corrected the first implementation of this design:

- the mixed contract moved from a second `impl Operation<ArrayIrType>` on `ZeroOperation<ArrayType>` to the
  `ArrayIrOperation::DynamicZero` variant arm, restoring one trait implementation per payload and Phase 8
  compatibility;
- the nullary guard became position-aware: it rejects only ungrounded identity *references* (dynamic array axes) and
  allows definition-position identities such as a `DimensionType`'s own variable;
- `From<ZeroOperation<ArrayType>>` routes canonically (reference-bearing types to `DynamicZero`, identity-free types
  to the homogeneous member family) and the dynamic-constructor inference rejects reference-free stored types, so
  each zero has one encoding;
- XLA lowering was fixed to treat the dimension operands as compact (one per dynamic axis, in axis order) instead of
  indexing them by physical axis number, with a mixed static/dynamic lowering-and-execution test;
- the intermediate `DimensionType::rename_identities` implementation revalidated a carried `known_extent` instead of
  copying it; the 2026-07-28 audit supersedes that field and deletes the entire path rather than polishing it further;
  and
- the increment was narrowed to zero only: the `One`/`Fill`/`Iota` guards and mixed contracts were reverted and land
  as complete vertical slices with their composite wiring, per review preference for reviewability.

Two review findings remain open as blocking follow-ups:

- `DimensionType::known_extent` participates in `Typed::r#type`, structural equality, and hashing, so concrete
  boundary extents leak into retained-JIT cache keys and specialize compilation per extent — the opposite of the
  feature's intent. The 2026-07-28 audit found that no replacement provider is necessary: structural closure already
  proves that each output reference identity is consumed or defined by its producing instruction, so complete-signature
  output validation may establish its first concrete fact for input-owned as well as internally defined identities.
  Keep `DimensionType` strictly identity plus bounds and add a retained-JIT test proving one specialization serves
  multiple extents.
- Transform consumers that stage `ZeroOperation<ArrayType>` (condition, scan, while, gather, scatter, slice, pad,
  and differentiation rules) can now hit the nullary reference guard when a dynamic `ArrayType` reaches them. The
  Jacobian reordering preserves precise diagnostics at two entry points but treats a symptom; Phase 6 owns migrating
  those consumers to `zero_like`, structural zeros, or explicit extent residuals, with dynamic-shape acceptance
  tests per consumer.

### Plan revision: projection ownership, constructors, residuals, and explicit operands

The pre-execution review identified four missing design decisions and this revision resolves them:

- projection now has distinct borrowed and consuming paths, and Phase 0 must decide whether the immutable reference
  `Array` payload also moves from `Vec<Scalar>` to measured shared storage before the prototype;
- generic zero/one/iota overlap is part of the dual-contract inventory, with operand-relative construction for
  transforms, homogeneous construction for static geometry, and one variant-owned mixed contract for dynamic
  geometry; rank-positive fill instead composes a scalar literal or caller value with the canonical broadcast
  operations for JAX parity;
- leaf-only dimensions remain explicit policy, while transpose-only primal extents move through one
  differentiation-owned ordinary SSA residual mechanism and `transpose_dimension_variables` is deleted; and
- each mixed operation owns one direct positional dimension-operand contract, and all consumers preserve those SSA
  edges without a parallel schema or payload copy of the shape.

The revision also retains the structured `TypeError` enum while selecting a named `Invalid { message: String }`
variant, with all construction routed through `TypeError::invalid(...)`, as a separately reviewable mechanical cleanup.

### Plan revision: execution staging and review process

The parent refactor's 141 expanded status entries plus this ignored plan are not reviewable, bisectable, or resumable
as one change. This revision resolves the staging and sequencing decisions:

- every created branch uses the `u/eaplatanios/` prefix;
- `u/eaplatanios/archive/dimensions-wip-2026-07-24` is a pushed, immutable, manifest-verified snapshot;
- `u/eaplatanios/wip/dimensions-remainder` is the separate mutable branch used to reconcile what remains;
- `u/eaplatanios/dynamic-shapes`, not `main`, is the sole integration and working base;
- intermediate increments use no pull requests: the executor pushes each recoverable increment branch and stages a
  no-commit merge on `u/eaplatanios/dynamic-shapes` for owner review and direct commit;
- the ignored plan is force-added to the archive and `S0`, preventing the recovery and delivery protocol from
  disappearing;
- the monolithic archived feature is not landed as `S6`; correct behavior and tests are mined into the ideal
  architecture phase by phase;
- `S5a` is a genuinely mechanical rename/module move, while identity-bearing dynamic dimensions land visibly in
  `P1`;
- semantic cleanup and the complete vertical architecture precede the associated-type `Operation` migration;
- `SelectOperation`, `StopGradientOperation`, and `TestNullaryOperation` become type-parameterized so the strict
  one-contract invariant remains universal;
- `AGENTS.md` contains a narrowly scoped restore exception only for clean dedicated staging worktrees, immutable or
  reviewed sources, and explicit documented paths;
- a delivery ledger at `.tasks/dimensions_cleanup_ledger.md` with a fixed entry template, kept distinct from this
  plan's design checkboxes;
- progress accounting uses fetched remote integration and remainder refs, while the immutable archive remains stable;
- a resumption protocol that derives state from the repository rather than from a previous session's claims; and
- final closure requires every archived path to be landed, superseded, or deliberately dropped, with no unexplained
  remainder.

### Execution: bootstrap archive and staging branches

The bootstrap preserved the original working tree before any extraction:

- confirmed `u/eaplatanios/dynamic-shapes` at `8105cfd26817ab728bb2799c889021f240345993`;
- recorded 112 tracked changes and 29 nonignored untracked paths, for 141 expanded status entries;
- recorded a 142-path snapshot manifest after explicitly including this ignored plan;
- committed the immutable archive as `770e77d001547c72150a44843c170ea6417ab41e`, with tree
  `4edb3eb201ab45e474c03614a33d580dab70bf67`;
- verified the pre-snapshot and committed manifests both had SHA-256
  `428782ca2768c9dfdb2f8260a2e806f8c5af0ac91ed5503df89037662dfab206`;
- pushed `u/eaplatanios/archive/dimensions-wip-2026-07-24` and
  `u/eaplatanios/wip/dimensions-remainder`;
- returned the owner checkout to clean `u/eaplatanios/dynamic-shapes`;
- created `/Users/eaplatanios/Development/Repositories/ryft-1-dimensions` on the remainder branch; and
- verified the archived baseline with isolated `cargo check -p ryft-core`, which completed successfully.

### Execution: S0 landed

The owner committed and pushed the staged S0 merge as `fbf43052ec04ea2822f3b3753883b68b0bb42c7e`. The mutable
remainder was reconciled and pushed as `979d9dd171ffab30944e80d4ab666614d96cf834`; the immutable archive still points
to `770e77d001547c72150a44843c170ea6417ab41e`. Both final bootstrap checkboxes are therefore complete.

### Execution: pre-S1 integration-baseline blocker

The focused S1 test exposed a failure before the compiler reached the S1 code. Integration commit
`8105cfd26817ab728bb2799c889021f240345993` already contains the physical move of
`tracing_v2/operations/custom_derivatives.rs` one level up, but its production use sites still import the deleted
`tracing_v2::operations` module. The moved file also includes 187 insertions from the later symbolic-dimension work,
referencing `OperationSymbols`, `SymbolSubstitution`, `Dimension`, and operation hooks that do not exist on
integration. The archived dirty tree compiled only because it supplied those later dependencies.

Do not widen S1 around this unrelated break. Preserve S1 on its increment branch, land `B0` as a separate baseline
repair reconstructed from the last compiling pre-move implementation plus direct path updates, then merge that repair
into S1 and resume its verification. `B0` supersedes the previously planned `S2`: the physical move was already
committed, but the commit neither updated its consumers nor separated later symbolic-dimension changes.

### Execution: B0 baseline repair

`B0` restores the last compiling pre-move `custom_derivatives.rs` implementation at its already-committed canonical
path, retains the root module documentation, and updates every production import and rustdoc link directly. It removes
the 187 later symbolic-dimension insertions that made the move depend on APIs absent from integration. Verification:

- `cargo check -p ryft-core` and `cargo check -p ryft-xla` passed;
- `cargo check -p ryft-xla` passed;
- `cargo test -p ryft-core --lib` passed all 911 tests;
- `cargo test -p ryft-core --doc` passed 43 tests with 13 ignored; and
- `cargo test -p ryft-xla --lib` passed 395 tests with 1 ignored.

The residual search found no `tracing_v2::operations` reference in Rust source under `crates/`.

### Execution: B1 type-module renames

- [x] Move `types/array_types.rs` to `types/arrays.rs` and change the public module path to `types::arrays`.
- [x] Move `types/data_types.rs` to `types/data.rs` and change the public module path to `types::data`.
- [x] Update all in-repo Rust paths and documentation links directly, without compatibility modules or re-exports.
- [x] Run formatting, affected crate checks, library tests, and doctests.
- [x] Record an empty residual search for the old module paths and stage the no-commit integration merge for review.

The rename exposed an existing root-facade collision: both `backends` and `types` now contain a public `arrays` module
and both are glob-re-exported by `ryft-core`. The B1 source branch explicitly selected `backends::arrays`, preserving
its existing root-facade meaning while exposing the type module at `types::arrays`; that disambiguation was omitted
during owner integration review. The landed tree therefore retains an `ambiguous_glob_reexports` warning. S1 does not
silently reverse that review outcome; the root-facade decision remains open for explicit owner direction or `P9`.

The B1 source branch completed verification without failures or code warnings:

- `cargo fmt --all -- --check`;
- `cargo check -p ryft-core`;
- `cargo check -p ryft-xla`;
- `cargo test -p ryft-core --lib` passed all 911 tests;
- `cargo test -p ryft-core --doc` passed 43 tests with 13 ignored; and
- `cargo test -p ryft-xla --lib` passed 395 tests with 1 ignored.

The exact old module-path and filename search is empty. The five remaining `data_types` matches are intentionally
retained local variable/parameter names describing collections of data types.

### Execution: S1 region arena ID selection

- [x] Add `RegionRef::with_id` as the canonical way to select another root from an existing borrowed arena.
- [x] Replace every in-repo reconstruction from `existing_ref.regions()` while retaining initial arena-entry
      `RegionRef::new` calls.
- [x] Cover successful ID replacement, arena preservation, rooted interfaces, and invalid identifiers in one focused
      test.
- [x] Run formatting, the focused test, the complete core library suite, and core doctests.
- [x] Push the verified increment and stage its no-commit integration merge for review.

`RegionRef::with_id` is public because `RegionRef` itself is a public borrowed arena view and downstream transformation
or backend implementations need the same metadata-preserving traversal seam as in-crate consumers. Keeping it
`pub(crate)` would leave downstream code reconstructing views through `RegionRef::new(existing.regions(), id)`, which
is precisely the duplicate arena-plumbing idiom S1 removes.

Verification completed:

- `cargo fmt --all -- --check`;
- the focused `test_region_ref_with_id` test passed;
- `cargo test -p ryft-core --lib` passed all 912 tests; and
- `cargo test -p ryft-core --doc` passed 43 tests with 13 ignored.

The compiler emitted the `arrays` ambiguous-glob warning already present on the reviewed B1 integration tree; S1
introduces no warning. The residual `RegionRef::new` calls are the two intentional initial arena-entry APIs in
`Program` and `ProgramBuilder`, the entry-region constructor, and the direct invalid-constructor test.

### Disposition: former S3 elementwise-macro increment

The former standalone S3 was based on a whole-file comparison that conflated three histories:

- the independent elementwise-macro restructure is already ancestral to the integration baseline;
- the remaining `TypeError::Invalid` rewrites and explicit elementwise inference result types belong to S4; and
- the remaining `Size` to `Dimension` changes belong to S5a.

There is therefore no valid standalone S3 patch. Restoring the archived `macros.rs` and
`differentiation/elementwise.rs` wholesale after S4 and S5a would merely duplicate those phases, while restoring them
beforehand would mix their prerequisites. S3 is removed from the increment catalog, its error-related hunks are
accepted by S4, and its dimension-related hunks remain deferred to S5a.

### Execution: S4 structured type errors

S4 keeps generic type machinery decoupled from dimensions while establishing the structured error surface needed by
the later heterogeneous program type:

- `TypeError::Invalid { message: String }` preserves the existing general diagnostics while allowing clear payload
  destructuring, and `TypeError::invalid(...)` centralizes all construction;
- `TypeError::Custom` carries typed, equality- and hash-preserving family errors through generic type APIs;
- heterogeneous projection failures use canonical `TypeError::Invalid` diagnostics because no consumer recovers their
  expected and actual variants;
- elementwise inference has explicit result types at its seven ambiguous macro boundaries.

The mechanical portion converted all 759 pre-existing `TypeError` construction and destructuring sites. Exact
diagnostic behavior is covered by the complete core and XLA suites, while focused tests cover invalid display and
custom downcasting, clone/equality, and hash behavior. All operation files touched by later dimension phases remain
shared paths: S4 owns only their error syntax.

Verification completed:

- `cargo fmt --all -- --check` and `git diff --check` passed;
- `cargo check -p ryft-core` passed;
- `cargo test -p ryft-core --lib` passed all 913 tests;
- `cargo test -p ryft-core --doc` passed 43 tests with 13 ignored;
- `cargo test -p ryft-macros -p ryft-macros-tests` passed all 53 macro unit tests and all 17 operation integration
  tests; one parameter compile-fail snapshot has an independently reproduced integration-baseline mismatch because
  the compiler's implementor list now includes `Axes`; and
- `cargo test -p ryft-xla --lib` passed 395 tests with 1 ignored.

The residual search contains no old `TypeError` struct construction, tuple-variant construction, or direct named-field
construction. The remaining `TypeError::Invalid { message }` matches are intentional destructuring patterns;
similarly named test fixtures and `DataTypeError` are unrelated.

### Execution: S5a dimension rename and module move

- [x] Rename the public `Size` descriptor to `Dimension` without changing its representation or semantics.
- [x] Move `Dimension`, `Shape`, `StaticShape`, and their 14 tests from `types::arrays` to
      `types::dimensions`.
- [x] Update every in-repo Rust use directly, without a compatibility alias or re-export.
- [x] Run formatting, diff checks, affected crate checks, focused and full library tests, doctests, and macro
      integration tests.
- [x] Classify every residual `Size` and old module-path match before staging the no-commit integration merge.

The semantic boundary remained intact: `Dimension` still has exactly the pre-S5a
`Static(usize) | Dynamic(Option<usize>)` representation. Identity-bearing dynamic dimensions, authoritative bounds,
and refinements remain wholly assigned to P1. Moving the shape types required only using their public
`dimensions()` accessor from `ArrayType`; one empty-axis test now passes `Axes::default()` because the module split
removed enough surrounding type context that rustc could no longer infer the element type of `[]`.

Verification completed:

- `cargo fmt -p ryft-core -p ryft-xla -p ryft -- --check` and `git diff --check` passed;
- `cargo check -p ryft-core`, `cargo check -p ryft-xla`, and `cargo check -p ryft` passed;
- the 14 focused `types::dimensions::tests` passed;
- `cargo test -p ryft-core --lib` passed all 913 tests;
- `cargo test -p ryft-core --doc` passed 43 tests with 13 ignored;
- `cargo test -p ryft-xla --lib` passed 395 tests with 1 ignored; and
- `cargo test -p ryft-macros -p ryft-macros-tests` passed all 53 macro unit tests and all 17 operation integration
  tests. The parameter compile-fail snapshot retains the independently reproduced S4 integration-baseline mismatch
  caused by rustc listing `Axes`.

At the S5a handoff, the only reported `Size` reference under `ryft-core`, `ryft-xla`, and `ryft` was the unrelated
`ryft_mlir::Size as MlirSize` import. That residual audit missed three low-level example expressions that the broad
rename had incorrectly changed to `Dimension`; S5b below corrects the omission. No old public core `Size` declaration,
core `Size` variant use, old core test name, or stale `types::arrays` path for the moved types remains.

### Execution: S5b MLIR size correction

- [x] Reproduce the integration failure by compiling the touched example target directly.
- [x] Restore only the three unrelated `ryft_mlir::Size` expressions.
- [x] Re-run the example check, scoped formatting, diff check, and exact residual classification.

S5a incorrectly renamed the low-level StableHLO example's `ryft_mlir::Size` uses. That type is unrelated to the
former `ryft_core::types::Size`; `cargo check -p ryft` did not compile the example target, so the original verification
missed the error. S5b restores those three expressions and adds
`cargo check -p ryft --example stable_hlo_matmul` as the direct regression check before P0.

The baseline command failed with six unresolved `Dimension` uses. After the correction, the same command passes.
`cargo fmt -p ryft -- --check` and `git diff --check` also pass. The four remaining exact `Size` matches are the three
restored example expressions and XLA lowering's `ryft_mlir::Size as MlirSize` import; all are intentional MLIR uses.

### Execution: P0 behavioral and architectural evidence freeze

P0 produces three review artifacts:

- `.tasks/dimensions_p0_evidence.md` freezes the exact revisions and environment; source, generated, compile, memory,
  graph, runtime-smoke, allocation, diagnostic, and proof baselines; all operation-family, context-view,
  reconstruction, collector, transform, residual, and eager-clone inventories; the projection ownership decision; and
  final code ownership.
- `.tasks/dimensions_operation_migration.md` gives every affected mixed, region-polymorphic, homogeneous, and
  constructor operation an explicit canonical signature and destination across eager execution, tracing, PE,
  batching, differentiation, regions, lowering, testing, and old-code deletion.
- `.tasks/dimensions_archive_disposition.md` classifies all 142 archived paths as independently extracted,
  behavioral input to a named phase, architecturally superseded, or deliberately deleted. A mechanical comparison
  found exactly 142 unique document rows with no missing or extra archive path.

The projection ownership decision is borrowed projection for read-only use plus consuming projection for ownership
transfer. P2 will not change reference `Array` storage to `Arc<Vec<Scalar>>` merely to compensate for an owned
projection API; it must first prove zero allocation and zero payload copying for a large eager array.

P0 also records two pre-existing deficiencies rather than hiding them: `shard_map_grad_inside` panics in both graph
benchmark revisions, and the archive has one ignored composite-while batching test because a batch-varying predicate
is not handled. P5/P6/P10 own their explicit resolution.

Verification completed:

- `cargo fmt --all -- --check` and `git diff --check` passed;
- `cargo check -p ryft-core --features benchmarking` passed, directly covering the one-line S1 benchmark call-site
  correction;
- the reviewed integration `cargo test -p ryft-core --lib` passed all 913 tests; and
- the immutable archive `cargo test -p ryft-core --lib` passed 1,035 tests with the one documented ignored batching
  gap.

The existing ambiguous `arrays` glob-re-export warning remains assigned to P9. P0 changes no dimension semantics.

### Execution: P1a dimension identity foundations

P1a adds only the compile-safe local foundations in `types::dimensions`:

- `DimensionBounds` validates inclusive-lower/exclusive-upper ranges and provides containment without duplicating
  identity metadata;
- `DimensionVariable` uses one shared immutable core, clone-preserving identity, pointer identity for equality and
  hashing, and a diagnostic-only name; and
- `DimensionError::InvalidBounds` derives `thiserror::Error` and travels through the generic `TypeError::Custom` path
  established in S4, so generic type machinery remains independent of dimensions.

`Dimension::Dynamic(Option<usize>)` remains unchanged until P1b. No compatibility variant, expression representation,
program hook, or backend behavior is introduced. Adding the typed `DimensionError -> TypeError` conversion exposed one
pre-existing unconstrained `Result<_, _>` in symbolic reshape inversion; P1a adds only the explicit
`Ok::<ReshapeDimensionExpression, TypeError>` annotation needed to select the intended conversion.

Verification completed:

- all 16 focused `types::dimensions` tests passed;
- `cargo check -p ryft-core` passed;
- `cargo test -p ryft-core --lib` passed all 915 tests;
- `cargo test -p ryft-core --doc` passed 43 tests with 13 ignored; and
- scoped formatting and `git diff --check` passed.

The known ambiguous `arrays` glob-re-export warning remains assigned to P9.

### Execution: P1b dynamic dimension leaves

P1b makes the P1a identity foundation the sole dynamic-axis representation:

- `Dimension::Dynamic` now owns one clone-preserving `DimensionVariable`, and all bounds consumers read the variable's
  authoritative immutable `DimensionBounds`;
- array inference preserves existing leaves through broadcast, reshape, transpose, reduction, dot, and unchanged
  full-extent slicing instead of reconstructing anonymous bounds;
- `Shape::is_refined_by` enforces repeated-leaf equality within a shape without allocating a refinement table;
- derived dynamic reshape, concatenate, pad, and strided full-extent slice results fail with exact diagnostics until
  P3 makes their result extents explicit SSA operands; and
- XLA lowering reads bounds from variables directly, while version-3 persistent signatures encode a typed shared
  variable table. Canonical compilation keys omit diagnostic names but retain shared-versus-independent variable
  relationships; executable metadata retains names and reconstructs each shared variable once.

No compatibility variant, expression representation, dimension witness, generic `Type` hook, or operation-specific
identity-source mechanism was added. Cross-type closure, boundary-wide refinement, alpha-renaming, cache identity,
and `OutputIdentityRole` deletion remain P1c work.

Verification completed:

- `cargo check -p ryft-core -p ryft-xla`, scoped formatting, and `git diff --check` passed;
- `cargo test -p ryft-core --lib` passed all 915 tests;
- `cargo test -p ryft-core --doc` passed 43 tests with 13 ignored;
- `cargo test -p ryft-xla --lib` passed 396 tests with one documented ignored benchmark;
- all 53 `ryft-macros` unit tests and all 17 operation macro-integration tests passed; and
- the remaining parameter trybuild mismatch reproduces unchanged on reviewed integration commit
  `7a2a0a39a96a3700c9855439faa8c2bfecece50c`: rustc includes `Axes` in a non-exhaustive implementor help list. P1b
  deliberately does not alter that inherited golden.

Targeted residual searches found no old `Dimension::Dynamic(None)`/`Dimension::Dynamic(Some(...))` construction, old
XLA version-2 type-schema identifier, invalid zero-bound lowering variant, or non-static `Dimension` `.copied()` use
under `crates`. The sole non-test/non-doc `DimensionVariable::new` outside `types::dimensions` is the version-3 XLA
signature decoder, which recreates validated shared variables from persistent metadata.

### Execution: P1c structural dimension identities

P1c adds the generic program-boundary machinery required by leaf identities without introducing dimension arithmetic
or an operation-specific producer classification:

- `Type::Identity` and `Type::Refinements` keep type-family identity occurrences and complete-boundary refinement facts
  generic; `DataType` uses zero-state implementations and `ArrayType` supplies dimension-variable behavior;
- structural region closure classifies formal-input and instruction-produced identities from definition/reference
  positions and graph dataflow, retaining the live input/internal partition needed by interpretation;
- simultaneous alpha-renaming covers swaps and permutations across atom types, constant and capture metadata, operation
  payloads, nested regions, and shared callees;
- region-carrying operations declare only their ordinary operand-to-region input mapping, allowing condition, while,
  scan, custom-derivative, and rematerialization regions to reuse the same generic instantiation path;
- complete-signature refinement establishes one concrete extent per shared input identity and validates related output
  occurrences, including identities structurally produced inside a region; and
- instantiation caches share disjoint alpha-equivalent calls but keep overlapping live-identity permutations distinct,
  using bidirectional structural instantiation rather than redundant canonical-signature fields on caches already keyed
  by one exact source callee or region root.

The prototype also showed that a separate generic canonical-identity-signature object had no production consumer:
core interning is restricted to one exact source callee or region root, while persistent XLA compilation keys already
own their typed canonical signature. P1c therefore does not add a dead canonical representation or
`TypeIdentity::CanonicalProperties` hook. Alpha-equivalent calls are compared structurally in both directions, which
retains bounds and repeated-identity relationships without storing parallel metadata.

The current array-only graph has no definition-position dimension values until P2 introduces `DimensionType`. P1c
therefore includes one explicit temporary closure rule: the first fresh result-reference occurrence may establish an
internal identity for a legacy shape-producing operation. P3 must delete that fallback after those operations consume
their result extents as first-class dimension operands; the rule is recorded in both the Phase 3 checklist and the
deletion ledger.

Verification and residual-audit results are recorded in the P1c cleanup-ledger entry at handoff.

### Execution: P2a dimension SSA foundations

P2a introduces the first homogeneous graph in which dimensions are ordinary SSA values. `DimensionType` owns one
definition-position `DimensionVariable`; `DimensionValue` carries a checked portable host extent; and the existing
generic `ConstantOperation` represents literals. Binary arithmetic is intentionally one
`DimensionArithmeticOperation` parameterized by a semantic `DimensionArithmetic` enum rather than nine parallel
payload types. Each application derives one fresh result variable with conservative bounds and performs checked eager
host arithmetic. The generic region-closure algorithm now recognizes definition-position constant types as immutable
internal SSA definitions, which lets the existing constant path carry dimensions without a special tracing rule while
preserving the prior unresolved-reference diagnostic.

Dimension input refinement is directional and bounds-based: a runtime or instantiated dimension with narrower bounds
may satisfy a broader declared operand even though it owns a different definition identity. Program instantiation
continues to derive explicit identity renamings, while ordinary interpretation can therefore accept fresh exact
dimension literals. The focused vertical slice passes through checked construction, inference, eager interpretation,
ordinary tracing, rendering, and known-side partial evaluation without introducing expression trees, witnesses,
projection wrappers, or special program machinery.

Ordered requirements, comparisons, gateways, `dimension_size`, heterogeneous storage, and projected binding remain
separate P2b–P2d/P3 increments. Verification and residual-audit results are recorded in the P2a cleanup-ledger entry at
handoff.

### Execution: P2b.2 capability-based dimension interpretation

The first P2b.2 implementation corrected the dependency direction but introduced a redundant checked-evaluation hook
instead of applying Ryft's established scalar/array architecture. It was staged but not committed by the owner and was
superseded in place:

- each nominal arithmetic operation owns `InterpretableOperation<C>` for domains whose values implement its
  capability;
- every arithmetic capability has an operation-aware `*_with` method plus the ergonomic method that constructs the
  operation from operand types;
- generic context-carrying values implement `*_with` by staging the supplied operation;
- `DimensionValue` implements each capability in `backends::dimensions`, validates the supplied operation against the
  eager operand types, performs checked host arithmetic, and materializes exactly the operation's fresh result type;
- `DimensionValue::DispatchDomain` becomes constant-only while `ExecutionDomain` retains the closed dimension
  operation family, matching `Scalar` and `Array`;
- `ArithmeticDimensionOperation::evaluate` and the backend arithmetic interpretation-adapter macro are deleted; and
- the requirement operation follows the same ownership rule: the operation owns interpretation, generic values stage
  its operation-aware capability, and `DimensionValue` owns concrete enforcement.

The concrete capability tests pin preservation of the supplied operation's fresh result identity and the exact
observed-value diagnostics for subtraction, floor division, and remainder. Generic tracing tests continue to pin
operation staging, while eager program and requirement tests now reach operation-owned interpretation through the
backend capability implementations.

Verification completed:

- `cargo fmt -p ryft-core -- --check` and `git diff --check` passed;
- `cargo check -p ryft-core -p ryft-xla` passed;
- focused dimension, backend-capability, and declarative-macro tests passed;
- `cargo test -p ryft-core --lib` passed all 946 tests;
- `cargo test -p ryft-core --doc` passed 53 tests with 15 ignored; and
- `cargo test -p ryft-xla --lib` passed 396 tests with one ignored.

Residual searches find no arithmetic `evaluate` hook, no backend-owned `InterpretableOperation` implementation, no
rich `DimensionValue::DispatchDomain`, and no production operation-to-backend dependency. The requirement operation's
predicate-specific `evaluate_extents` remains intentionally shared by eager enforcement and known-side reasoning; it
is not an arithmetic value-materialization side channel. The four pre-existing ambiguous dimension/math glob-re-export
warnings remain assigned to P9.

### Execution: P2c generic storage-sum projection

P2c introduces only the heterogeneous storage and projection seam:

- `ArrayIrType` and `ArrayIrValue<A>` are the sole array/dimension storage sums;
- standard `From`/borrowed `TryFrom` type conversions and the `ValueProjection<T>` contract provide lifting and
  distinct borrowed and consuming projection paths;
- eager array and dimension projection returns direct references or transfers the stored payload, with no clone,
  allocation, boxing, or type-erased dispatch;
- `ProjectedValue<T, V>` and `ProjectedValueRef<'v, T, V>` preserve the original capture, tracer, partial-tracer, or
  differentiation-tracer value while exposing its checked homogeneous member type; the borrowed view avoids cloning
  member metadata when it can be borrowed directly;
- reference `Typed` delegation makes `&A` a zero-state typed projection instead of requiring a borrowing wrapper;
- storage-sum identity renaming and boundary refinements delegate to the existing array/dimension contracts while
  enforcing canonical wrong-kind diagnostics; and
- no projected context, operation-family lift, mixed operation, gateway, `dimension_size`, or batching policy is
  introduced.

Composite batching projection remains P5 work because the current batching carrier is structurally array-only.
Defining a heterogeneous batch enum in P2c would make the storage increment choose the replicated-dimension and mapped
authority policies assigned to P5. P2d can therefore prototype direct projected binding and a third storage member
without pulling transformation policy forward.

Verification and residual-audit results are recorded in the P2c cleanup-ledger entry at handoff.

### Execution: P2d zero-state projected binding

P2d completes the generic member adapter without introducing the production array-program dispatcher or any mixed
shape operation:

- `OperationProjection<T>` associates a composite operation family with one homogeneous member family, while its
  required standard `From` implementation provides the sole lift into the outer dispatcher;
- `ProjectedContext<C, T>` stores only `C` plus a zero-sized type marker, delegates eagerness and resolution, and
  performs one project/lift/bind/project round trip with no program inspection or dependency reconstruction;
- nullary, unary, and binary projected binds reconstruct parent inputs in fixed-size stack arrays; only wider
  homogeneous operations allocate a temporary input vector; projecting the parent context's returned output vector
  still materializes the projected output vector required by the current `Context::bind` API;
- projected binding rejects both declared and attached regions because heterogeneous region signatures remain owned by
  the outer higher-order operations in P4;
- one blanket `Value for ProjectedValue<T, V>` supplies dispatch and execution domains for tracers, partial tracers,
  and differentiation tracers; no carrier-specific `Value` implementation is added;
- a three-member test family exercises eager binding, ordinary trace staging, exact SSA identity, constant/staged
  resolution, absence of implicit operands and regions, and unchanged generic support for tracer and transform
  carriers; and
- the context-size assertion pins that the type marker adds no runtime storage beyond the parent context.

Production `ArrayIrOperation`, mixed operand contracts, shape operations, higher-order region projection, batching,
and differentiation policies remain assigned to P3–P6. The existing reference-array allocation tests continue to prove
that borrowed and consuming eager value projection neither allocates nor copies payloads. Projected-context binding
temporarily reconstructs parent values from borrowed inputs because the generic `Context::bind` contract accepts a
slice; concrete eager member values continue to dispatch through their native contexts, while symbolic projected
values clone only their parent tracer/transform representation and preserve SSA identity. P10 retains the final
cross-context allocation and latency measurement once production outer dispatch is present.

Verification and residual-audit results are recorded in the P2d cleanup-ledger entry at handoff.

### Execution: P3a production array-program dispatcher

P3a introduced `ArrayIrOperation<A>` as the sole stored dispatcher for heterogeneous array/dimension programs.
Homogeneous array and dimension operations retain their native type contracts and pass through generic projection and
lifting. Genuinely mixed signatures receive explicit outer-family variants rather than a generic mixed bucket.
P3a added no mixed operation; verification and residual-audit results are recorded in the P3a cleanup-ledger entry.

### Execution: P3b canonical first-class dimension size

P3b introduced the sole `array -> dimension` `DimensionSizeOperation`. The operation records the selected declared
axis dimension, preserves dynamic identity structurally, produces exact static results, and works through eager
interpretation, tracing, import, partial evaluation, and the initial composite lowering path. The owner consolidated
host extent extraction and composite first-class results under one `DimensionSize<Output>` capability.
Verification and residual-audit results are recorded in the P3b cleanup-ledger entry.

### Execution: P3c explicit dimension-to-scalar conversion

P3c introduced the sole `dimension -> rank-0 i64 array` `DimensionToScalarOperation`. It also added the minimum
composite batching, structural differentiation, and StableHLO foundations needed to exercise the complete
`dimension_size -> dimension_to_scalar` vertical slice. Dimension arithmetic remains host-side in eager execution,
and the logical conversion lowers as scalar SSA identity. Verification and residual-audit results are recorded in the
P3c cleanup-ledger entry.

### Execution: P3d checked scalar-array gateway

P3d introduced the sole checked `rank-0 integer array -> dimension` `DimensionFromScalarOperation`. The operation owns
one declared result identity and bounds, reference eager execution checks all signed/unsigned integer payloads,
ordinary partial evaluation folds or residualizes the gateway, batching rejects mapped shape authority with the typed
diagnostic, and composite lowering explicitly defers checked runtime assertions to P7. The subsequent owner-reviewed
namespacing cleanup prefixed every dimension operation module and arithmetic capability method with `dimension_`
without changing standard operator syntax. Verification and residual-audit results are recorded in the P3d
cleanup-ledger entry.

### Execution: P3e vector-element composition

The initial P3e implementation introduced `DimensionFromVectorElementOperation`, but review found that it fused two
independent existing concepts: ordinary array element extraction and `DimensionFromScalarOperation`'s checked
data-to-dimension authority boundary. It also had no production consumer and no direct JAX or StableHLO counterpart,
yet required dedicated inference, eager, dispatch, batching, differentiation, lowering, and test wiring.

The dedicated operation and all of its cross-cutting machinery were therefore removed before landing. Statically
sized vectors compose `slice -> reshape-to-scalar -> dimension_from_scalar`; the scalar gateway documentation now
demonstrates that canonical path. Dynamically sized vectors must use ordinary checked indexing or an explicit
logical-length requirement before scalarization because `dynamic_slice` clamps out-of-range indices. If that use case
cannot be expressed cleanly after the mixed slicing migration, Ryft should add a general checked array-element
operation rather than another dimension-specific gateway. Verification and the residual audit are recorded in the
P3e cleanup-ledger entry.

### Execution: P3f first-class dimension comparison

P3f extended the existing `CompareOperation` rather than adding a dimension-specific comparison language. Its
output-parameterized capability supports both ordinary homogeneous comparisons and
`(Dimension, Dimension) -> Array(Boolean scalar)` in the flat array-program dispatcher. Dimension comparison now
preserves explicit operands through eager execution, partial evaluation, replicated-only batching, structural
differentiation, import, rendering, and direct signed StableHLO comparison lowering. Verification and the residual
audit are recorded in the P3f cleanup-ledger entry.

### Execution: P3g explicit reshape and broadcast dimensions

P3g made canonical reshape and broadcast mixed operations whose output extents are ordered first-class dimension
operands. Exact constants and dynamic extents use the same contract; inference derives the complete output shape and
the operation payload stores only non-shape semantics. Deliveries B and C established the individual reshape and
broadcast paths. Delivery D proves dimension arithmetic feeding both operations in one stored program across eager
execution, partial evaluation, batching, JVP, identity instantiation, import, direct StableHLO lowering, and static
PJRT execution.

The public API now exposes only `Broadcast`: exact constants, computed dimensions, right-aligned expansion, and
leading expansion all bind the same mixed operation. `backends::arrays::BroadcastKernel` is the backend-only eager
kernel over an already-concrete `ArrayType`; it cannot stage an operation or turn metadata into shape authority.
Phase 4 removed the packed integer-array `DynamicBroadcastOperation` representation and its backend capability,
transform rules, XLA lowering, and tests. The static-metadata `LegacyBroadcastOperation` remains hidden and frozen
until its homogeneous `ArrayOperation` transform and XLA consumers migrate to the composite graph. The detailed
consolidation and residual evidence is in `.tasks/plan_broadcast_api_consolidation.md` and the P4a review below.

### Execution: P3h Delivery A explicit concatenate result extent

P3h Delivery A extended the existing axis-only `ConcatenateOperation` with
`Operation<ArrayIrType>` rather than creating a redundant legacy payload. Its canonical mixed signature is
`(Array..., Dimension) -> Array`: inference derives the concatenated result axis exclusively from the final
dimension operand, rejects contradictory exact sums, and preserves dynamic result identities. Eager execution
validates the supplied extent against the checked observed input sum before calling the existing array kernel.

The dynamic acceptance program computes both input extents with `dimension_size`, combines them with
`dimension_add`, and passes the resulting ordinary SSA value directly to concatenate. Exact rendering, partial
evaluation, identity instantiation, and cross-program import preserve that edge without expressions, witnesses,
packed shape data, or source-array recovery. Batching/differentiation and lowering remain explicit Delivery B/C
rejections rather than receiving incorrect generic behavior. Verification and residual evidence are recorded in
`.tasks/plan_p3h_concatenate.md` and the cleanup ledger.

### Execution: P3h Delivery B concatenate transforms

P3h Delivery B preserves the explicit result extent through batching and differentiation. Composite batching rejects
mapped dimension authority with `BatchingError::MappedDimension`, aligns every mapped or replicated array operand on
one physical batch axis using the existing `ArrayBatch` machinery, shifts the logical concatenate axis when needed,
and stages the same mixed operation with the unchanged extent operand.

Forward differentiation stages primal and tangent concatenates against the same transformed extent SSA value and
materializes structural zero array tangents only through the existing projected array context. Static transposition
delegates to the established homogeneous slice-based pullback and gives the extent a structural-zero cotangent.
Dynamic concatenated axes retain the explicit Phase 6 dimension-residual boundary. The dynamic
`dimension_size -> dimension_add -> concatenate` acceptance program now passes eager execution, partial evaluation,
batching, JVP, identity instantiation, and import; direct lowering remains P3h Delivery C.

### Execution: P3h Delivery C concatenate lowering

P3h Delivery C lowers the canonical four-instruction
`dimension_size(left), dimension_size(right), dimension_add, concatenate` program directly when every concatenated
input extent and the explicit result extent are exact. Mixed inference proves their equality; lowering consumes the
trailing scalar as compile-time shape authority and emits one `stablehlo.concatenate` over only the physical arrays.
The CPU plugin compiles and executes the result successfully.

Dynamic concatenated axes retain the Phase 7 runtime-equality-assertion boundary with an exact diagnostic. Lowering
does not trust the explicit scalar, reconstruct dimensions from arrays, emit a packed shape operand, or perform host
readback. The golden exact program contains 4 instructions, renders to 223 bytes, and lowers to 379 bytes of
StableHLO; a warm local probe compiled in 28,795 microseconds and executed/synchronized/copied in 354 microseconds.
Verification and the classified homogeneous-consumer residual audit are recorded in
`.tasks/plan_p3h_concatenate.md` and the cleanup ledger.

### Execution: P3j Delivery A shaped-zero architecture gate (HISTORICAL — superseded)

This section records the wrapper-based Delivery A as executed. The wrapper design it describes was subsequently
replaced by the stored-type-authoritative `DynamicZero` design (variant-owned mixed contract, dynamic-only operands,
canonical `From` routing); see the constructor revision notes in the Review section. It is retained as the execution
record of the superseded increment only.


P3j Delivery A introduced the sole generic `ShapedConstructorOperation<C>` adapter and proved its zero specialization
as a flat `ArrayIrOperation::ShapedZero` variant. The operation consumes one explicit first-class dimension
operand per output axis, including exact constants, and derives its complete result shape from those operand types.
The wrapped homogeneous zero contributes only element type, expected rank, and placement metadata; no identity,
bounds, extent ordering, witness, or packed shape data is duplicated in the payload.

Eager execution resolves those values to one concrete static allocation. Complete-signature validation carries the
observed concrete extent as a private `DimensionType` boundary refinement, keeping variable identity and bounds
authoritative while avoiding a generic `Value` payload-inspection hook. Partial evaluation folds known extents and
residualizes unknown ones; batching requires replicated extent authority; transpose gives extents structural-zero
cotangents; and fused JVP reuses the same-typed zero primal for its zero tangent so no nullary dynamic zero is
invented.

Bounded XLA lowering materializes the upper-bound zero buffer and attaches each dynamic logical size with
`stablehlo.set_dimension_size` using the explicit SSA operand. Its test emits no `get_dimension_size` and compiles and
executes through CPU PJRT. Generic linearization and disconnected-input pullback still need transform-owned dimension
residuals, so deletion of the temporary `Zero(ZeroOperation<ArrayIrType>)` escape hatch remains P3j Delivery D
and may move to the first Phase 6 residual delivery rather than adding type-to-value recovery. Exact inventory,
measurements, verification, and residual evidence are recorded in `.tasks/plan_p3j_shaped_constructors.md`.

### Execution: P3j full-parity fill and dynamic iota

The constructor gate is complete. Fill deliberately does not add another composite constructor: current JAX source
defines `lax.full` as scalar conversion plus broadcast and `jax.numpy.full` as scalar `lax.full` or array
`broadcast_to`. Ryft now uses the same decomposition. `Fill` is convenience syntax that converts the host scalar,
binds a rank-zero `ConstantOperation`, and uses ordinary broadcast for rank-positive outputs. Scalar or
broadcast-compatible array SSA values with first-class extents bind the existing mixed `BroadcastOperation` through
`Broadcast`. There is no specialized fill operation, `DynamicFill`, captured runtime fill payload, copied
extent schema, or geometry recovery.

The general mixed-broadcast JVP materializes identity-bearing structural-zero tangents through one `DynamicZero` fed
by the original extent operands. Existing broadcast implementations continue to own live JVP, transposition,
batching, partial evaluation, import, and lowering. XLA broadcasts statically shaped inputs to finite physical bounds
and refines logical axes directly from the extent operands, avoiding unsupported dynamic-broadcast legalization
without observing input geometry.

Dynamic iota is the final variant-owned mixed constructor. Identity-free outputs remain homogeneous
`ArrayOperation::Iota`; identity-bearing outputs use `DynamicIota(IotaOperation<ArrayType>)` with compact dynamic
extent operands. Its fallible constructor accepts JAX's integer, floating-point, and complex numeric dtypes and
validates the axis before an operation can enter a program; inference reuses the shared constructor signature. Eager
execution, PE, batching, JVP, transpose, identity instantiation, and import preserve those edges. The shared
dynamic-constructor JVP explicitly includes `DynamicZero`, so a future zero-capable dtype with a widened,
non-zero-space tangent does not depend on primal-reuse eligibility. Lowering emits native `stablehlo.iota` followed by
the shared bounded
`set_dimension_size` refinement; a pairwise-distinct `4 x 2 x 3` complex fixture compiles and executes on CPU with no
`get_dimension_size`.

Final verification passed 1,002 core unit tests, 410 runnable XLA unit tests (one pre-existing ignored benchmark), all
macro integration tests, and 58 runnable core doctests (16 ignored examples), plus formatting and diff checks. Exact
implementation and review records are in `.tasks/plan_p3j_dynamic_fill.md` and
`.tasks/plan_p3j_dynamic_iota.md`.

### Execution: P3k canonical collectives

P3k moves the complete result geometry of all-gather, psum-scatter, and all-to-all into ordinary dimension SSA without
duplicating their semantic operation payloads. The payloads remain homogeneous `Operation<ArrayType>`
implementations; the flat `ArrayIrOperation` variant arms own the mixed positional contracts. Every result axis
has one trailing extent operand in axis order. Unchanged axes preserve their input identities, changed axes carry
ordinary arithmetic SSA, and exact extents are checked against participant-count multiplication or division during
inference and eager execution.

The public composite capabilities produce exact constants for static axes, trace `dimension_size` only for dynamic
source axes at the operation boundary, apply ordinary dimension multiplication or checked division, and bind the
collective against the complete result shape. Eager execution validates observed geometry and supports only the
binder-free one-participant identity. PE folds or residualizes the same explicit graph. JVP gives primal and live
tangent collectives the same extent SSA. P5d added direct matching-axis materialization and different-axis forwarding
for static and bounded-dynamic graphs; batching performs no source-array dimension reconstruction. Dynamic transpose
retains one exact Phase 6 residual error rather than rebuilding geometry from types.

A golden trace proves `dimension_size -> dimension_mul -> all_gather` survives rendering, alpha-renaming,
instantiation, and import as ordinary operand edges. The residual audit finds no result extent, identity, bound,
witness, or transform-residual metadata in any collective payload and no payload with dual operation-type contracts.
The direct composite XLA dispatcher accepts the same explicit manual-binder state used by production shard-map
lowering, and a fixture proves the canonical dimension SSA graph reaches the shared native collective lowerer without
reconstructing dimensions. Phase 4 subsequently migrated production attached-region storage and tracing to that
composite graph. At this review boundary, checked bounded-dynamic public execution remained Phase 7
boundary-materialization work; Phase 7 has now completed it.

The complete implementation ledger and remaining JAX-parity continuation are in
`.tasks/plan_p3k_collective_dimensions.md`.

### Execution: P4a packed dynamic-broadcast deletion and structural-closure completion

The first Phase 4 review unit deleted the obsolete homogeneous runtime broadcast representation rather than carrying
it through the production XLA migration. `DynamicBroadcastOperation` encoded output extents as a rank-one integer
array, duplicated the canonical mixed broadcast's inference, eager, batching, differentiation, and lowering behavior,
and was the only remaining operation whose result type introduced a fresh dynamic identity without an explicit
dimension producer edge. The operation, `LegacyDynamicBroadcast` capability, `ArrayOperation` and `XlaOperation`
variants, reference-backend implementation, XLA lowering, and dedicated tests are gone. StableHLO's
`dynamic_broadcast_in_dim` wrapper is unrelated backend IR and remains available. The surviving canonical composite
broadcast test now executes one compiled bounded-dynamic program at multiple extents, so deleting the packed
representation does not reduce runtime-varying-shape coverage.

With that producer removed, `Region::type_identity_signature` no longer treats a previously unseen
reference-position result identity as an internal definition. Every result reference must now forward an operand
identity or refer to a definition-position occurrence on a sibling result. The structural closure suite includes the
corresponding fresh-reference negative regression. The full core gate also exposed and corrected one stale expected
diagnostic so dynamic constructor messages consistently use the canonical `dimension<name ∈ bounds>` rendering.

Verification for this review unit:

- `cargo test -p ryft-core --lib`: 1,014 passed;
- `cargo test -p ryft-xla --lib`: 417 passed and one pre-existing benchmark ignored;
- `cargo test -p ryft-macros-tests`: both 17-test integration suites passed, including all compile-fail fixtures;
- targeted searches find no core/XLA occurrence of `DynamicBroadcastOperation`, `LegacyDynamicBroadcast`,
  `DYNAMIC_BROADCAST_OPERATION_NAME`, or an operation-family `DynamicBroadcast` variant; and
- `cargo fmt --all -- --check` and `git diff --check` passed.

### Execution: P4b1 lossless projected-region foundation

`Program::into_unprojected` is now the one structural bridge from an already-built homogeneous member program to its
unprojected graph. The consuming conversion lifts constants with `ValueProjection::from_projected`, variable types and
operations with their canonical `From` contracts, preserves public parameter structure and every atom,
instruction, and region edge, and then rebuilds the mapped arena once through `Program::new`. It does not interpret,
trace, or replay instructions and therefore cannot invent dependencies or change SSA identity.

The regression covers a dynamic array identity, a lifted constant, and one imported branch region attached to both
condition slots. The mapped graph retains exact shared-region topology, effects, type-identity signature, parameter
structures, operation variants, and constant values. The full core suite passed 1,015 tests; the XLA suite passed 417
tests with one pre-existing ignored benchmark; `cargo check -p ryft-xla --lib`, formatting, and diff hygiene passed.
The next P4b increment must consume this lift at the projected region-binding boundary; `ProjectedContext` still
rejects regions until that driver conversion is atomic.

### Execution: P4b2 projected array capture-leaf classification

`XlaArrayConstant` now names every capture reference whose semantic role is an `ArrayType` member payload.
`XlaConstant` deliberately remains a temporary alias until the atomic production-domain cutover, so current
`XlaProgram` and compilation artifacts remain unchanged while standalone composite lowering, array-member operation
payloads, `MlirLowerableValue`, and array-typed scan/shard-map markers are already insulated from the upcoming type
flip.

The dependency audit confirmed that persistent-cache V3 stores lowered StableHLO and the public physical array ABI,
not the internal Ryft program, so no schema bump is justified unless those persisted artifacts become incompatible.
It also assigns new composite higher-order batching and differentiation policies to P5/P6 while requiring P4b to
preserve current behavior and exact deferred diagnostics.

`cargo check -p ryft-xla --lib` passed without warnings. The complete XLA library suite passed 417 tests with one
timing-sensitive benchmark ignored; formatting and diff hygiene passed.

### Execution: P4b3 projected capture delegation

The region-driver audit established that generic driver unprojection is neither lossless nor minimal. Per-root
materialization would discard `ReplayRegionDriver`'s cross-application mappings and `CalleeRegionDriver`'s `Rc`
identity, while a generic cache spanning different value/operation families would add disproportionate machinery.
The atomic P4b cutover will instead lift complete owned public bodies once into ordinary composite drivers and keep
all subsequent replay natively composite. This review unit therefore leaves `ProjectedContext`'s region rejection in
place and adds only the independent capture-delegation prerequisite. One regression captures array and dimension
members through projected views of the same parent and verifies their ordered indices, exact projected types, and
single shared parent table. The core suite passed 1,016 tests; the XLA suite passed 417 tests with one ignored
timing-sensitive benchmark; `cargo check -p ryft-xla --lib`, formatting, and diff hygiene passed.

### Execution: P4b5–P4b7 production boundary prerequisites

P4b5 generalized the jitted-call region contract over its enclosing program type and gave shard-map a direct
composite contract with an array-only public boundary. P4b6 then proved in an isolated compile/execute prototype that
the public XLA facade can retain `In`/`Out: Parameterized<ArrayType>` while its internal compilation artifacts use the
canonical `In::To<ArrayIrType>`/`Out::To<ArrayIrType>` families. The trace enters the composite domain
directly and uses checked array-member views at the user closure; no second domain, top-level member-program replay, or
new core compilation hook is required.

P4b7 completed the remaining type-level region prerequisite. Condition accepts a scalar Boolean array predicate and
mixed branch boundaries. While accepts mixed state under a scalar predicate and rejects first-class dimension state
under a batched predicate because one extent cannot carry per-item masked values. Scan accepts array/dimension carries
and array-only stacks; stacked arrays may have dynamic inner axes tied to explicit carried identities. Focused
composite regressions passed, followed by all 1,019 core tests and all 421 runnable XLA tests. The exact contracts,
prototype evidence, and residual production-cutover work are recorded in
`.tasks/plan_p4b_production_composite_xla.md`.

### Execution: P4b8 concrete eager dimension gateways

The final production type-cycle probe validated the intended generic operation relationship—one composite constant
family whose array payload is derived through `ValueProjection<ArrayType>`—and found one missing runtime contract:
the public-facade prototype did not exercise eager mixed operation dispatch. The incomplete representation edit was
reversed. The production plan now specifies the nonrecursive split explicitly: compile array members, interpret
dimension members on the host, let the active XLA domain materialize dimension-to-scalar values, and insert concrete
dimension constants into eager mixed-operation SSA before lowering only array inputs.

As an independent prerequisite, concrete XLA arrays now recover exact global axis extents from complete shard
descriptors without device work and convert checked rank-zero integer arrays into `DimensionValue`s through one
explicit host readback. The focused CPU gateway regression, XLA library check, formatting, and diff hygiene passed.

### Execution: P4b atomic production composite XLA cutover

The XLA backend now has one stored graph and one production lowering path. `XlaDomain` uses
`ArrayIrType`, `ArrayIrValue<Array>`, the composite `XlaConstant`, and the flat composite `XlaOperation`;
all public array APIs preserve their existing `ArrayType`/`Array` contracts through checked projection. Complete
owned member regions are lifted structurally once, while replay and callee regions remain natively composite and
retain their identity-sharing semantics. No projection-aware derive mode, second production domain, replay bridge, or
stored `ArrayIrOperation` was introduced.

Eager execution separates host-owned dimension work from cached array kernels, and mixed eager operations specialize
their concrete dimension operands into internal scalar SSA constants. Production StableHLO lowering consumes
first-class dimension operands directly and recursively lowers every higher-order region through the same dispatcher
with shared capture, collective, nested-function, and ordered-effect state. The standalone composite driver and
duplicate test suite were deleted. The retained diff removes more code than it adds.

Production coverage executes dimension-size arithmetic followed by dynamic broadcast and reshape, pins eager cache
behavior, retains the static eager/JIT/reshard/sharding/cache/profile-guided suites, and keeps exact Phase 5/6
diagnostics where composite higher-order batching or reverse differentiation remains deliberately deferred.
At the P4b boundary, inconclusive compiled requirement assertions remained Phase 7 work, and public bounded-dynamic
shard-map execution plus plugin-specific `PadToStatic` validation remained the named P3k/P7 continuation.
Static/manual shard-map production reachability was already complete; Phase 7 has since closed its assertion and
plugin-validation work.

Verification passed all 1,019 core library tests, all 396 runnable XLA library tests (one timing benchmark ignored),
the benchmark-feature all-target compile gate, focused benchmark-feature tests, the compilation benchmark smoke run,
XLA doctests, formatting, and diff hygiene. The residual scan found one composite `XlaProgram` alias, no standalone
composite lowerer, no retired dynamic-broadcast path, no disabled `cfg(any())` fixtures, and no public array API
exposing `ArrayIrValue`. The complete implementation ledger and exact evidence are recorded in
`.tasks/plan_p4b_production_composite_xla.md`.

### Execution: P4c composite region and identity verification

Focused production fixtures now prove that condition branches, while state, and scan carries preserve first-class
dimension SSA and use that authority directly to shape downstream array results. A CPU eager condition fixture
executes both branch choices while sharing one compiled specialization.

Repeatedly splicing a gateway-producing region graph exposed one generic ownership defect: identities defined inside
the imported program were copied literally. `TypeIdentity::fresh` now gives program relocation the nominal primitive
it needs, and `ProgramBuilder::splice_program` instantiates boundary identities before alpha-renaming all internal
identities across the complete imported region arena. Fresh replacements are checked against all source and
destination identities and all replacements generated earlier in the same splice. The regressions pin both collision
rejection and distinct imported identities with the same diagnostic names and bounds, exact gateway-to-condition
operand edges, and matching nested-region interfaces.

At the P4c boundary, compiled `dimension_from_scalar` and public bounded-dynamic shard-map execution remained Phase 7
work because their range checks had to lower as ordered assertions and their physical boundary required the checked
`PadToStatic` path. P4c did not weaken those contracts or add an unchecked temporary lowering; Phase 7 has since
completed the gateway lowering and bounded physical boundary.

Verification passed all 1,020 core library tests, all 57 runnable core doctests (16 examples ignored), all 401
runnable XLA library tests (one timing benchmark ignored), both macro integration suites, the XLA all-target check,
formatting, and diff hygiene. The complete fixture and implementation ledger is recorded in
`.tasks/plan_p4c_composite_region_verification.md`.

### Plan revision: tier-3 data-dependent dynamism extension (2026-08-01)

The plan's end state was extended from the containment cleanup to full data-dependent (tier-3) dynamism. The tier
vocabulary (tier 1 static, tier 2 input-derived, tier 3 data-dependent within declared bounds) is now defined in the
**Objective** section, and Phases 12–14 were added after the cleanup closure gates: Phase 12 closes tier-3 semantics
around the existing P3d `dimension_from_scalar` gateway (effects-model coverage for DCE-surviving bounds checks,
cache-identity and closure coverage, linear-call residual and control-flow fixtures, JAX-rejection comparison
fixtures), Phase 13 owns bounded data-dependent compiled execution (measured route decision between XLA bounded
dynamism and static-bound-plus-masks, the authoritative per-operation padding-discipline table, exact rejection
diagnostics for unsupported operations), and Phase 14 owns ragged batching (carrier-owned per-element extents plus
consumer-owned masks, demand-shaped by a recorded motivating workload). Phase 7 gained one classification-only item
capturing the initial padding-discipline inventory during its existing lowering sweep, the operation-contract
invariants now name the gateway, the transform policy table records that only Phase 14 may relax replicated-only
dimension batching, and the verification matrix, abort criteria, and exit criteria gained tier-3 entries.

The revision was motivated by external evidence gathered while reviewing the P6 linear-call architecture: JAX's
stalled dynamic-shapes effort (`DShapedArray` avals containing jaxpr variables, `bint` bounded integers with padding
rules, and a residuals-first `call_transpose` annotation convention identical to `LinearCallOperation`'s swap
signature) validates dimensions-as-SSA-values as the tier-3-capable representation and locates the retrofit costs
Ryft avoids by being greenfield. The audit for this revision also found that the semantic entry point was further
along than the drafted phases assumed: P3d had already landed the gateway with eager checked execution, partial
evaluation, and the mapped-batching rejection, so Phase 12 was scoped as verification-and-closure rather than
introduction, and the gateway's own compiled lowering was assigned to Phase 7 as recorded by P4c and is now complete.
The `DimensionType` motivation rustdoc added earlier the same day overstated the provenance restriction ("never from
array data") and was corrected in the same session to name the gateway as the single checked exception.

### Phase 8 P8b4 closure: mixed-operation transform ownership (2026-08-03)

Completed P8b4c and P8b4d together. Explicit-extent collective batching now lives beside the all-gather,
sum-scatter, and all-to-all payloads and reuses the homogeneous matching-axis kernels without reconstructing dynamic
geometry from array metadata. The remaining mixed batching algorithms moved beside dynamic-shape slice, reshape,
broadcast, concatenate, and pad. `MemberBatchableOperation` records the exceptional contract for native member
payloads whose parent instruction has a mixed signature, parallel to the existing member JVP and transposition
capabilities.

The array-program batching dispatcher now performs only projected dispatch, direct payload dispatch, member dispatch,
and the previously documented shared zero/one/iota constructor classification that the later derive marker must
generate. The obsolete backend-local collective helpers and mixed-operation algorithms were deleted. All core tests,
integration and compile-fail tests, runnable doctests, and all runnable XLA library tests passed. The detailed
implementation and verification ledger is recorded in `.tasks/plan_p8b_array_program_operation_declaration.md`.

### Phase 9 early execution: universe-neutral batched-program carriers (2026-08-03)

`BatchedProgram` is now the universe-neutral contract for structural batching results. The former concrete
`BatchedProgram` became `BoundaryPreservingBatchedProgram`, the reusable carrier for policies whose transformed
boundary exactly matches the source region, while `ThreadedExtentBatchedProgram` implements the same contract without
weakening its leading extent-input/output forwarding checks. `BatchingPolicy::BatchedProgram` is constrained by the
new trait, so future program universes must expose semantic output-axis metadata and lossless access to their complete
policy-specific boundary.

The two concrete `BatchingContext::align_batched_program_outputs` implementations were replaced by one policy-generic
implementation in the core transform module. It compares and validates only semantic source-output axes and preserves
all bookkeeping values carried by the selected program representation. Array and array-program control-flow rules now
use the same core algorithm; their distinct boundary mechanics remain isolated in their concrete carriers and
policies. The remaining Phase 9 module relocation is intentionally still pending.

Verification passed `cargo check -p ryft-core --tests`, all 1,123 core library tests, all 52 runnable core doctests (16
examples ignored), both 17-test macro integration suites including their compile-fail fixtures, all 434 runnable XLA
library tests (one timing-sensitive benchmark ignored), formatting, and diff hygiene.

### Phase 8 complete operation-derive migration (2026-08-03)

Completed the remaining operation-derive work as one review unit. The derive now owns the mechanical contract for
projected, replicated, mixed, and composite-native variants, including inference, eager interpretation, PE, batching,
JVP, transposition, conversions, and operation-family projections. `ArrayIrOperation<A>` is the production proof:
its enum declaration is now the sole variant/class ledger and its previous outer dispatcher implementations have been
deleted. Semantic rules remain handwritten beside their payloads, while shared member contracts contain only the
repeated parent-boundary projection and delegation vocabulary.

This closes the Phase 8 derive gate with a net reduction in the review diff and a 4,083-line reduction in the three
production dispatcher surfaces. The larger macro fixture expansion is intentional test coverage for the complete
multi-universe contract; fresh compile time did not regress. The detailed carries-over/deletes ledger, metrics, and
verification evidence are recorded in `.tasks/plan_p8b_array_program_operation_declaration.md`. Conditional
concatenation-effect precision remained a separate semantic follow-up because the axis-only payload lacked the
operand-derived extent proof; the following review unit records its completion without conflating it with the derive
migration.

### Phase 8 concatenation effect-precision closure (2026-08-04)

Mixed concatenation now retains one operand-derived Boolean stating whether its complete construction signature proves
the explicit result extent. `from_input_types` validates that signature and is the canonical mixed constructor when
types are available. The generic homogeneous-to-mixed conversion remains necessary for operation projection and is
therefore deliberately conservative: without operand types it creates an assertion-bearing payload. Inference rejects
only the unsafe stale case in which a payload classified as pure is rebound to a signature that needs a runtime check;
an assertion-bearing payload may remain conservatively effectful after refinement.

Eager execution continues to compare the observed input-axis sum with the supplied extent defensively. XLA lowering
now omits the assertion callback, ordered token, and input-extent extraction for proven-static signatures, while
dynamic signatures retain the checked host callback. Batching reconstructs the mixed payload from its transformed
operand types so the effect classification remains precise. Core and lowering regressions pin both classifications and
the absence of assertion and dimension-size IR in the pure case. The review also corrected one pre-existing stale
all-gather expected diagnostic to the current canonical wording; collective behavior was unchanged.

Verification passed all 1,129 `ryft-core` library tests, all 53 runnable core doctests (16 ignored), and all 436
runnable `ryft-xla` library tests (one ignored). The first incremental core invocation encountered a Rust compiler
fingerprint ICE; rerunning with `CARGO_INCREMENTAL=0` passed without cleaning the workspace. Formatting and diff
hygiene passed. A follow-up audit documented and pinned that the proof bit intentionally participates in equality and
hashing, that the homogeneous instantiation always stores `false`, and that mixed-to-homogeneous-to-mixed conversion
soundly loses provenness. The only production mixed-to-homogeneous conversion is the terminal delegation into the
static homogeneous transposition rule; it never round-trips into a mixed program.

### Phase 9 entry audits: error recovery and public dimension ergonomics (2026-08-04)

The S4 error contract remains intact. `TypeError::Custom` stores an equality- and hash-preserving `CustomError`, the
dimension-owned `From<DimensionError> for TypeError` conversion uses that path, and callers recover the original typed
error with `TypeError::downcast_custom`. Array/dimension type and value projection failures remain the canonical
`TypeError::Invalid` messages `expected array type but got dimension type` and `expected dimension type but got array
type`; no production path branches on a redundant projection-specific error variant.

The historical `RuntimeDimension` and `RuntimeShape` wrappers are deliberately retired rather than recreated. Their
useful behavior is now supplied directly by ordinary first-class SSA values and capability traits: `DimensionSize`
produces a dimension member, fallible dimension arithmetic stages the corresponding dimension operations,
`DimensionRequirement` stages checked constraints, dimension/scalar gateways are explicit, and reshape, broadcast,
slice, constructors, collectives, and transforms consume those same values. Rank remains statically represented by
ordinary Rust parameter structure and array types, so a second shape wrapper would add neither authority nor safety;
it would only duplicate storage-sum projection and operation APIs. Focused typed-error, type/value-projection, and
public dimension-capability tests passed.

### Phase 9 array-specialization slice 1: carrier and axis mechanics (2026-08-04)

The `ArrayBatch` carrier, batching-specific `ArrayType` normalization and unbatching methods, and the shared mapped-axis
sharding join/normalization functions now live in `batching::arrays`. The batching root moved from `batching.rs` to
`batching/mod.rs`, placing it beside the specialization module while preserving `crate::batching` as the canonical
facade. Universe-neutral contracts and transform machinery remain in that root, while the array policies stay there
only until the next isolated review slice. The intentional `batching::ArrayBatch` re-export keeps callers on the
batching facade; only private shared mechanics name `batching::arrays` directly. The crate-root `ArrayBatch` export
remains the flat public facade. The moved implementation is otherwise unchanged, and root policy code now uses the
carrier's public accessors instead of relying on former same-module field visibility.

Verification passed `cargo check -p ryft-core --tests`, all 1,129 core library tests, all 53 runnable core doctests (16
ignored), all 57 macro unit tests, both macro integration suites including all compile-fail fixtures, and all 436
runnable XLA library tests (one ignored). Formatting, diff hygiene, unique-owner searches, and the old-path residual
search passed. The following review slice moves the array policy family; the array-specialization tests move last so
each extraction remains within the review budget.

### Phase 9 array IR vocabulary rename (2026-08-04)

The heterogeneous array/dimension SSA vocabulary is now named `ArrayIrType`, `ArrayIrValue`, and `ArrayIrOperation`;
its signature refinement state is `ArrayIrTypeRefinements`, its batching carrier is `ArrayIrBatch`, and its batching
policy is `ArrayIrBatching`. The existing `types::arrays` and `backends::array_programs` module placement remains
unchanged, no compatibility aliases were added, and the public crate-root exports use only the new names. Directly
corresponding helper, lowering, test, and allocation-fixture names now use “array IR” as well, while ordinary
homogeneous programs over `ArrayType` remain described as programs over arrays. The obsolete TODO proposing that the
heterogeneous dispatcher become `ArrayOperation` was deleted because the new name makes its distinct mixed-universe
role explicit.

The Phase 9 checklist now also records the subsequent, separately reviewable reference-backend consolidation: replace
`Array`'s `Vec<Scalar>` payload with validated immutable contiguous bytes; route exact XLA literals through that
canonical representation; retire the scalar backend after proving its remaining tests have suitable replacements; and
then move concrete arrays, dimensions, and the array IR under one top-level `ryft_core::arrays` hierarchy. The future
slice has explicit encoding, allocation, exact-bit, CPU/XLA execution, and performance gates and does not alter this
rename.

Verification passed the renamed two-test projection-allocation fixture, all 1,129 core library tests, all 53 runnable
core doctests (16 ignored), all 57 macro unit tests, both macro integration suites including every compile-fail fixture,
and all 436 runnable XLA library tests (one ignored). `cargo check -p ryft-core --tests`, formatting, diff hygiene, and
repository-wide residual searches for the six retired public identifiers and their corresponding lowering/inference
helper names passed. The follow-up batching rename additionally confirmed that the rejected historical
`ArrayIrBatchingContext`, `ArrayIrBatchingTracer`, `ArrayIrBatchableOperation`, and `ArrayIrProjection` concepts have no
live code definitions; they remain plan-history names only.

### Phase 9 planned array hierarchy refinement (2026-08-04)

The eventual top-level `ryft_core::arrays` hierarchy now owns not only the reference `Array`, dimension backend, and
array IR, but also their complete domain-specific type vocabulary: element data types, array and dimension types,
shapes, layouts, memories, and sharding. `ArrayIrType` moves beside `ArrayIrValue` and `ArrayIrOperation` in
`arrays::ir`; commonly used names are re-exported through `arrays` so public signatures do not require deep module
paths. The existing top-level `types` and `sharding` facades are deleted without compatibility modules once all
consumers have moved.

The generic type layer does not absorb these concrete types. `programs::types` already contains the correct
backend-neutral abstractions (`Type`, `Typed`, `TypeError`, `TypeRefinements`, and signature traversal), whereas moving
`DataType`, `ArrayType`, `DimensionType`, layouts, or memories there would couple core program machinery to one value
universe. Consequently, `types/mod.rs` disappears after its contents move under `arrays`; its contents are not copied
into `programs/types.rs`. The plan uses the intended name `arrays::sharding`—not `arrays::sharing`—and retains
universe-neutral named-axis and transform machinery outside that array-specific hierarchy.

### Phase 9 array-specialization slice 2: policy and program mechanics (2026-08-04)

The complete homogeneous-array policy family now lives beside `ArrayBatch` in `batching::arrays`:
`ArrayBatching`, `ArrayBatchingPolicy`, `StaticArrayBatchingPolicy`, and `DimensionSource`, together with their
`BatchingPolicy`, entrypoint, recursive, shared elementwise, and homogeneous region/program batching implementations.
The parent module retains the intentional `batching::{ArrayBatch, ArrayBatching, ArrayBatchingPolicy,
DimensionSource, StaticArrayBatchingPolicy}` facade, while the implementation owner is now unambiguously the array
specialization module. No compatibility module or alternate implementation path was introduced.

This was a source move plus import and module-documentation adjustment: operation-specific batching rules remain with
their payloads, universe-neutral contracts and transform machinery remain in `batching`, and the existing
array-specialization tests remain in the parent module for the next isolated review slice. Verification passed
`cargo check -p ryft-core --tests`, all 1,129 core library tests, all 53 runnable core doctests (16 ignored), all 57
macro unit tests, both macro integration suites including every compile-fail fixture, and all 436 runnable XLA library
tests (one ignored). Formatting, diff hygiene, and definition-owner searches passed; the moved implementations occur
only in `batching::arrays`, while the parent module contains only their intentional public re-exports.

### Phase 9 batching specialization closure (2026-08-04)

The array-specialization tests now live at the end of `batching::arrays`, ordered after the implementation surface
they exercise. Generic batching errors, axes, boundary carriers, contexts, and projected-operation tests remain in the
parent module. The extraction also corrected declaration order: `ArrayBatching` is declared with all of its policy,
entrypoint, and recursive implementations before `impl BatchableType for ArrayType`, followed by the shared
elementwise rule, homogeneous region/program helpers, and tests.

The complete heterogeneous specialization moved from `backends::array_programs::batching` to
`batching::array_ir`: `ArrayIrBatch`, `ArrayIrBatching`, `ThreadedExtentBatchedProgram`, the dynamic-array and
replicated-dimension projected policies, projection/value adapters, first-class extent helpers, entrypoint and
recursive policies, named-axis support, and their tests moved together. `backends::array_programs` now owns only the
eager array-IR value and operation family plus its differentiation capability module. All in-repo consumers use the
new canonical specialization path directly; `batching` and the crate root retain the intentional flat public facade,
and no compatibility module or old-path re-export remains.

The array-IR file is byte-for-byte identical at its new path, while the array test move changes only test ownership and
imports. Verification passed `cargo check -p ryft-core --tests`, all 1,129 core library tests, all 53 runnable core
doctests (16 ignored), all 57 macro unit tests, both macro integration suites including every compile-fail fixture, and
all 436 runnable XLA library tests (one ignored). Formatting and diff hygiene passed, the old module declaration and
all Rust paths to `backends::array_programs::batching` are absent, and the implementation definitions occur only in
`batching::{arrays, array_ir}`.

A follow-up minimality pass inlined `array_dimensions`, `normalize_array_input`, and `dimension_constant` into their
owning policy rules and deleted the three private helpers. The inlined code preserves the same explicit dimension-size
reads, sharding normalization, constant binding, output-count validation, and error propagation. The full 1,129-test
core library suite and `cargo check -p ryft-core --tests` pass after the simplification.

### Phase 9 batching hierarchy ownership closure (2026-08-04)

The final two operation-specific algorithms left in `batching::array_ir` now live beside their payload owners. The
dynamic `CollectiveBatchingPolicy` implementation moved next to the static policy and collective batching kernels in
`operations::collectives`; it reuses the array-IR specialization's existing first-class broadcast primitive rather
than duplicating that representation mechanic. The existing `impl_nullary_batchable_operation!` macro now supports
`@member<ArrayIrType, ArrayIrBatching>` for homogeneous-nullary member constructors whose mixed encoding consumes
replicated first-class dimension operands. The member expansion depends only on the named parent universe and its
generic `BatchingPolicy` contract; it does not name array types, values, batches, or policy implementations internally.
Its `@replicated` branch is likewise generic over every context matching the operation's native type and every batching
policy for that context. Zero, one, and iota request both forms beside their payload definitions, eliminating the local
constructor-specific macro without replacing it with an array-IR-specific shared branch. Malformed mapped member
operands remain rejected through a policy-neutral diagnostic; valid operation behavior and signatures are unchanged.

The closure audit found no outer operation dispatcher or payload-specific batching algorithm in either
`batching::arrays` or `batching::array_ir`. Their remaining operation bounds and bindings implement generic policy
mechanics: elementwise delegation, mapped-axis alignment, first-class extent materialization, recursive program
rewriting, and entrypoint adaptation. Backend modules retain concrete eager values and closed operation families; the
dimension backend's batching implementation remains beside the backend-owned `DimensionOperation` payload and does not
define a transform policy. The intended `batching` facade remains direct, and one stale rustdoc link now names that
canonical facade.

Verification passed `cargo check -p ryft-core --tests`, the focused 12 array-IR batching tests, 35 collective tests,
10 constant-family tests, all 1,129 core library tests, and all 53 runnable core doctests (16 ignored). Formatting,
diff hygiene, dispatcher/algorithm ownership searches, transform-policy definition searches, and the stale-path search
passed.

### Phase 9a representation dependency correction and execution plan (2026-08-04)

The current-tree audit found that the former checklist order could not be executed honestly: `Scalar` still has a
production role as `Array`'s stored payload and element algebra, so deleting the scalar backend before changing array
storage would require either breaking the reference backend or introducing a temporary replacement abstraction. Phase
9 now migrates `Array` to canonical immutable bytes and exact byte-backed XLA literals first, retires the standalone
scalar program universe second, and performs the public module hierarchy move only after both representation paths
have settled.

The detailed Phase 9a checklist records the representation, encoding, API, allocation, exact-bit, test-migration, and
performance contracts plus abort criteria. Execution has not started; Phase 9a0 is the next review unit.

### Phase 9a0 baseline and representation prototype (2026-08-04)

Phase 9a0 completed at clean starting revision `9af35b7a17e33dc88b1f212b96a067d9913d74cd` using Rust 1.93.1. Physical
source-line counts, including comments and blanks but separating every `#[cfg(test)]` block, are:

| Module | Non-test lines | Test-only lines | Total lines |
| --- | ---: | ---: | ---: |
| `backends::arrays` | 2,010 | 634 | 2,644 |
| `backends::scalars` | 2,327 | 701 | 3,028 |
| Combined | 4,337 | 1,335 | 5,672 |

On the baseline host, `size_of::<Array>()` is 208 bytes and `size_of::<Scalar>()` is 24 bytes. The temporary F32
prototype, storing `ArrayType` by value and an `Arc<Vec<u8>>`, was 192 bytes. A counting allocator measured a
4,096-element F32 vector as follows. The current construction measurement includes its required temporary `Vec<f32>`;
the prototype construction borrows the already-built F32 slice and therefore isolates byte-storage construction.

| Measurement | Current `Vec<Scalar>` | F32 byte prototype |
| --- | ---: | ---: |
| Retained element payload | 98,304 bytes | 16,384 bytes |
| Construction allocations | 3 | 3 |
| Construction allocated bytes | 114,704 | 16,440 |
| Clone allocations | 2 | 1 |
| Clone allocated bytes | 98,320 | 16 |

The prototype payload is six times smaller. Its sole clone allocation is the accepted 16-byte `ArrayType` metadata
clone; the 16-KiB element payload is shared without allocation or copying. A consuming `Vec<f32>` convenience API
would additionally allocate its 16-KiB input vector, so Phase 9a1 must not present the isolated construction figure as
an end-to-end constructor cost. `ArrayType` remains owned by value; shared ownership is reserved for the large element
payload.

The complete reference scan classified the scalar vocabulary as follows:

- **Array-element semantics:** `backends::{scalars,arrays}` owns the current element algebra and storage. XLA literal
  lowering decodes those elements. These semantics migrate to checked bytes and byte-backed kernels; they are not
  deleted with the scalar program universe.
- **Host-literal clients:** fill, attention, random, array IR scalar gateways, coordinate-basis construction, dot,
  `erf`, collectives, and reduction use `Scalar` only to spell rank-zero or broadcast host literals. They migrate to
  typed/rank-zero `Array` construction. No independent scalar execution requirement was found.
- **Scalar-universe-only infrastructure:** `ScalarOperation`, `ScalarTracingContext`, their exports, scalar benchmark
  support, macro scalar modes, and the scalar-domain compile-fail fixture exist to run standalone programs whose
  values are `Scalar`. Useful coverage moves to rank-zero arrays; the family is then deleted.
- **Reusable generic-transform coverage:** the scalar-heavy tests in tracing, programs/builders, partial evaluation,
  batching, differentiation, custom derivatives, rematerialization, control flow, and elementwise operations exercise
  generic machinery cheaply. Each must move to rank-zero arrays or an existing minimal test value before scalar
  retirement. Exact low-precision and complex element tests instead become array-codec/kernel tests.
- **Documentation and unrelated names:** remaining production mentions in select, program rendering, axes,
  dimensions, slicing, transposition, and XLA prose are examples or ordinary uses of the English word “scalar.”
  `ryft-pjrt`'s FFI attribute `Scalar` variant is an unrelated upstream concept and remains unchanged. The audit found
  no otherwise-dead production scalar path; scalar-only tests become redundant only after their replacement coverage
  lands.

The workspace contains 2,371 constructor mentions: 77 `Array::new`, 703 `Array::scalar`, 867 `Array::vector`, 298
`Array::matrix`, and 426 `Array::from_f64s`; 2,143 are under crate `src` trees and the rest are integration fixtures.
There are 230 `values()` accessor mentions across 36 core/XLA files. Phase 9a1 preserves the four ergonomic typed
constructors, replaces `new` with checked typed/raw-byte construction, and replaces each `values()` dependency with
borrowed bytes or typed single-element decoding. `from_f64s`/`to_f64s` remain explicitly lossy test conveniences and
must not become the exact literal route.

The complete portable little-endian element-encoding matrix is:

| Bytes per logical element | `DataType`s | Validation |
| ---: | --- | --- |
| 0 | `Token`, `Zero` | Entire payload must be empty for every static shape. |
| 1 | `Boolean` | Byte must be `0` or `1`. |
| 1 | `I1`, `I2`, `I4`, `U1`, `U2`, `U4` | High bits are zero; signed values use narrow two's-complement bits. |
| 1 | `I8`, `U8` | Every byte is valid. |
| 1 | `F4E2M1FN`, `F6E2M3FN`, `F6E3M2FN` | Unused high four or two bits must be zero. |
| 1 | Every `F8*` variant | Every byte is an exact format bit pattern. |
| 2 | `I16`, `U16`, `BF16`, `F16` | Exact little-endian integer or floating-point bits. |
| 4 | `I32`, `U32`, `F32` | Exact little-endian integer or floating-point bits. |
| 8 | `I64`, `U64`, `F64` | Exact little-endian integer or floating-point bits. |
| 8 | `C64` | Interleaved real then imaginary F32 little-endian bits. |
| 16 | `C128` | Interleaved real then imaginary F64 little-endian bits. |

Logical constructors check `element_count * byte_width` for overflow and accept exactly that many encoded logical
elements before placing them through `ArrayAddressing`; physical-storage constructors require the layout-derived span,
including any holes or tile padding. The current `Scalar` family covers all `DataType`s except `I1`, `I2`, `I4`, `U1`,
`U2`, and `U4`; byte storage closes precisely those six gaps. Lowering currently reconstructs typed vectors for
Boolean, ordinary integers, BF16/F16/F32/F64; reconstructs raw temporary vectors for low-precision floats and complex
values; has a separate scalar-splat path; and rejects token, zero, and all six sub-byte integer types. Phase 9a3
replaces those routes with layout-aware logical traversal, packing Boolean and narrow integers only at the MLIR
boundary. A big-endian host must normalize before calling MLIR's raw-buffer API rather than forwarding little-endian
element encodings unchanged.

The temporary vertical prototype covered checked F32 construction/decoding, unary negation, scalar broadcasting add,
numeric equality, signed-zero rendering, payload-sharing clone measurement, and exact raw MLIR construction including
a NaN payload. The raw dense-elements attribute preserved all source bytes exactly. A second prototype routed F32
negation through a stack-only private transient-element enum: it added no per-element allocation, but had identical
allocation counts to direct dispatch and added decode/encode wrappers and variant matches. Direct typed byte dispatch
is therefore the smaller F32 implementation and is selected for Phase 9a1. Phase 9a2 may adopt a narrowed private
transient element only if the full 34-type kernel matrix demonstrates a net production-code reduction; it may never be
stored, public, heap-allocated per element, or become another value/operation universe.

Both probes were removed after recording their evidence, leaving no prototype production or test scaffolding. The
Phase 9a0 gate passes: no unsafe cast, alternate stored representation, public abstraction, or payload-sized clone
allocation is required. Baseline verification passed all 1,129 core library tests, all 53 runnable core doctests (16
ignored), and all 436 runnable XLA library tests (one timing-sensitive benchmark ignored). Phase 9a1 is the next
isolated implementation unit.

### Phase 9a1 dense-buffer addressing refinement (2026-08-04)

The pre-implementation audit found repeated row-major stride, odometer, flat-index, block-copy, and block-replacement
logic across the current reference kernels. Phase 9a1 introduces one private checked descriptor for mapping logical
indices and logical element ranges to physical byte ranges, plus one prevalidated allocation-free iterator over maximal
physically contiguous runs in rectangular/strided selections. Phase 9a2 reuses these mechanics where they remove
duplication, while keeping operation-specific coordinate semantics in the operation kernels and prohibiting a public
view/indexing abstraction.

The plan makes physical layout part of reference-value semantics. `ArrayType::layout()` determines the byte placement
inside `Array`, including derived bases for negative strides, holes, and tile padding; a missing layout defaults to
dense row-major storage. StableHLO or other portable literals remain logical values: their conversion traverses logical
coordinates through `ArrayAddressing` and emits the ordering required by the destination format rather than imposing a
second layout-independent storage representation on reference arrays.

### Phase 9a1 dense-buffer addressing implementation (2026-08-04)

The first Phase 9a1 implementation slice added `ArrayAddressing`, `ArrayIndexRange`, and `ArrayIndexRanges` in the new
private `arrays::addressing` module. Addressing is the deliberate term because the contract maps logical indices to both
element offsets and byte ranges. `ArrayAddressing` stores only its validated `ArrayType`, keeping the type as the sole
source of truth instead of duplicating its shape, counts, or strides. Its first implementation covers the interim dense
row-major `Vec<Scalar>` backend: it validates static materializability, the portable byte width for all 34 `DataType`s,
logical element and byte lengths, physical storage byte length, overflow, multi-index bounds, and flat element ranges,
while representing rank-zero, empty, and zero-byte `Token`/`Zero` values without alternate storage rules. Arbitrary
physical-layout addressing remains the next required extension before byte-backed reference arrays are constructed.

The range iterator validates complete start/size/stride selections once, borrows all selection metadata, and stores
only scalar traversal state. It derives row-major strides during the same reverse pass that decodes each mixed-radix
ordinal, so neither the descriptor nor the iterator allocates a coordinate or stride vector. It emits maximal
contiguous runs: complete selections become one range, contiguous inner windows become one range per outer coordinate,
and genuinely strided inner selections become single-element ranges. Existing block-copy and block-replacement kernels
now consume those ranges, deleting their duplicated stride calculation, coordinate odometers, and element-by-element
writes while preserving their caller-validated contracts. Arbitrary-layout addressing, layout-aware byte storage, and
exact physical-byte/logical-round-trip tests remain the next Phase 9a1 slices.

Verification passed the three focused addressing/range tests, all 35 reference-array tests, all 1,132 core library
tests, `cargo check -p ryft-core --lib` without warnings, formatting, and diff hygiene. A normal Clippy pass reports no
diagnostics in the changed implementation; the repository-wide `-D warnings` gate remains blocked by 131 pre-existing
diagnostics elsewhere in `ryft-core`, beginning with `operations/math/dot.rs` and `backends/scalars.rs`.

### Phase 9a1 layout-aware addressing and codec boundary (2026-08-04)

The second Phase 9a1 slice generalized `ArrayAddressing` without adding cached shape or layout state. Missing layouts
remain dense row-major. Strided layouts now validate rank, derive a safe base for negative byte strides, include holes
in their checked storage span, and reject layouts whose ordered stride spans cannot prove that element byte ranges are
non-overlapping. Tiled layouts validate a complete minor-to-major permutation and implement XLA's suffix tiling,
padding, repeated tiling, and combined dimensions. Exact tests pin positive, negative, holed, permuted, padded, nested,
and combined address calculations, malformed layouts, alias rejection, and storage-span overflow.

`ArrayIndexRanges` now follows logical slice order while coalescing only consecutive logical elements whose physical
byte ranges are also consecutive in ascending address order. This preserves direct bulk-copy semantics for dense
runs and splits safely at holes, reversals, permutations, and tile boundaries. The iterator still allocates nothing
and validates the complete slice before iteration; because the number of physical runs is layout-dependent, it no
longer claims `ExactSizeIterator`.

The new sealed `ArrayElement` codec fixes one little-endian representation for Rust Boolean, integer, floating-point,
half-precision, and complex values. Crate-private typed and checked-raw entrypoints encode directly into final physical
storage without a dense intermediate allocation, validate logical and physical byte lengths, Boolean/sub-byte/4-bit/
6-bit encodings, zero-valued holes and tile padding, and payload-free types, and decode either typed values or portable
logical bytes independently of physical ordering. The entrypoints are intentionally established one checklist item
before their production `Array` consumer; the module-local dead-code allowance is removed by the immediately following
byte-storage migration rather than exposing a temporary public raw-storage API.

Layout-aware storage fixtures place the same six logical `U8` values into layout-free, positive and negative strided,
permuted tiled, and padded tiled storage, assert every physical byte and span, and decode every case back to the same
logical values and portable bytes. Focused addressing and codec tests pass, all 1,137 core library tests pass,
all 53 runnable core doctests pass with 16 intentional ignores, `cargo check -p ryft-core --lib` is warning-free, and
formatting and diff hygiene pass. The next Phase 9a1 slice converts `Array` from `Vec<Scalar>` to the checked shared byte
storage that now has a fully pinned ownership boundary.

A second follow-up slice (plan: `plan_sub_byte_and_low_precision_elements.md`, 2026-08-05) added checked element
newtypes to `arrays/elements.rs` (split out of `encoding.rs`): sub-byte integers `i1`/`i2`/`i4`/`u1`/`u2`/`u4` that
own the two's-complement-in-the-low-bits storage convention (fixing the trap where `-1i4` read back as `15`), and
conversion-only low-precision floats `f4e2m1fn` through `f8e8m0fnu` backed by one class-driven `FloatFormat` engine
with exact decodes and round-to-nearest-even encodes following ml_dtypes/XLA semantics (quiet canonical NaNs;
overflow to ±infinity for IEEE-style formats, canonical NaN for `fn`/`fnuz`/`fnu`, saturation for the NaN-less MX
formats; fallible only for NaN into MX formats and zero into `f8e8m0fnu`). Encodes were cross-validated bitwise
against an independent exact-rational model over every valid pattern and every adjacent midpoint. `Scalar`'s private
low-precision engine intentionally diverges (saturating overflow, signaling canonical NaN bits, an unguarded
huge-input nearest scan, and a `fnuz` negative-underflow NaN) and dies with the byte-storage migration; the
partially overlapping checklist item below still owes the `Array` construction/validation wiring. All 1,150 core
tests pass.

A review follow-up hardened three findings. `ArrayAddressing::new` now propagates static element counts that overflow
`usize` as construction errors instead of panicking through the element-count accessor, with an exact regression test.
A crate-private `ArrayAddressing::is_dense_row_major` predicate (missing layout, dense-equivalent strides, or a
descending tile-free minor-to-major permutation) gives `range` an O(1) bulk byte range and gives every codec
entrypoint a bulk chunked path instead of per-element addressing, with dense-equivalent strided and tiled layouts
added to the exact-byte round-trip fixtures. The one-byte-per-element storage of sub-byte data types (zero high bits,
unlike XLA's packed two-per-byte host representation) is now documented on `element_byte_width`; the Phase 9a3
lowering-boundary packing item already covers the required repack. The follow-up also renamed the private
`ArrayIndexRanges::range` to `element_range` and clarified the tiled-addressing helper and codec-entrypoint
docstrings. All 1,137 core library tests pass.

A subsequent API review replaced `ArrayIndexRanges`' three parallel start, size, and optional-stride slices with one
borrowed `[ArraySliceAxis]`, eliminating mismatched metadata lengths from the representation. `ArraySliceAxis` stores
one axis's start, selected-coordinate count, and stride, exposes those values through accessors, and converts from
`Range<usize>` for concise unit-stride call sites such as `(0..8).into()`. Reference slicing and block-update kernels
now construct the same descriptor consumed by addressing instead of passing parallel metadata through another layer.
All 90 focused array, addressing, codec, and reference-backend tests pass, and `cargo check -p ryft-core --lib`
succeeds; its eight warnings are confined to the concurrently developed, not-yet-consumed codec entrypoints.

### Phase 9a1 byte-backed `Array` storage (2026-08-05)

`Array` now owns exactly one shared immutable physical byte buffer alongside its `ArrayType`; cloning shares that
buffer without copying its payload. `Array::new` accepts and validates complete layout-aware physical storage,
`from_elements` and `from_logical_bytes` construct physical storage from typed or encoded logical row-major elements,
and `storage_bytes`, `elements`, and `logical_bytes` expose the corresponding checked views. The scalar, vector, and
matrix conveniences are typed through the sealed `ArrayElement` contract, while the explicitly lossy `from_f64s` and
`to_f64s` test conveniences remain. The old public `values()` accessor and `Vec<Scalar>` stored representation are
deleted, and the test-only malformed-type constructor now carries raw bytes without weakening production validation.

All core and XLA consumers were migrated in the same slice. Reference-array fixtures use typed decoding or exact
logical bytes, finite-difference tests perturb real and imaginary projections without inspecting scalar variants, and
XLA dense-literal lowering decodes typed elements or traverses logical bytes so physical reference layouts never leak
into portable literals. Existing equality, approximation, debug/display, and reference kernels temporarily decode
through one private transient scalar bridge; it is neither stored nor public, and Phase 9a2 removes it family by family
in favor of direct typed-byte dispatch as already required by that phase's gate.

Verification passes all 1,146 core library tests, all 53 runnable core doctests (16 intentional ignores), all 436
runnable XLA library tests (one intentional timing-sensitive ignore), `cargo check -p ryft-xla --tests`, formatting,
and diff hygiene. The next Phase 9a1 slice adds the exhaustive encoding-round-trip matrix before the dedicated
sub-byte construction and allocation gates.

### Phase 9a1 exhaustive encoding round trips (2026-08-05)

Four focused reference-array tests now pin exact physical and logical bytes across every byte-aligned primitive
family. Boolean and signed/unsigned integers cover boundary and representative bit patterns. BF16, F16, F32, and F64
cover positive and negative zero, both infinities, and retained NaN payloads. Every low-precision floating-point format
round-trips representative raw encodings through typed decoding and re-encoding. C64 and C128 preserve those same
component-level edge cases, including complex NaN payloads and infinities. Empty shaped arrays and positive-element-
count `Token`/`Zero` arrays prove that zero-byte storage is valid without introducing an alternate representation. The
dedicated following checklist item still owns construction and validation coverage for sub-byte integers.

The codec entrypoints remain in their defining `arrays::encoding` module rather than broadening the parent module's
public vocabulary for an internal consumer. Verification passes all 1,150 core library tests, the focused 36-test
reference-array suite, formatting, and diff hygiene.

### Phase 9a1 sub-byte reference-array construction (2026-08-05)

The reference `Array` boundary now has one exact integration test covering every signed and unsigned sub-byte integer
type: I1, I2, I4, U1, U2, and U4. Typed construction preserves sign-extended native values while emitting the specified
low-bit physical encodings. Complete physical storage and logical-byte construction decode to the same typed values,
and each data type rejects the first byte with a bit set above its declared width. This closes the former `Scalar`
storage gap entirely through the sealed `ArrayElement` codecs without adding any `Scalar` variants or another stored
representation.

The byte-backed `Array` implementation imports codec functions directly from `arrays::encoding`; those implementation
entrypoints are not re-exported from `arrays`. Verification passes the focused sub-byte integration test, all 1,151
core library tests, formatting, and diff hygiene. The allocation-proof item remains the next isolated Phase 9a1 slice.

### Phase 9a1 allocation and ownership gate (2026-08-05)

The dedicated allocation-test binary now records allocation requests, total requested bytes, and the largest request.
Cloning a one-element F32 array and a 4,096-element F32 array produces identical allocation statistics, and both clones
retain the exact physical-storage pointer of their source. The large clone's total and largest allocations are each
strictly smaller than its 16-KiB payload, proving that clone cost is metadata-only and independent of element count.
Borrowed array-member projection remains allocation-free over 1,000 iterations, while consuming projection performs
zero allocations and transfers the owned array directly.

The Phase 9a1 ownership audit confirms that `Array` stores only its `ArrayType` and one `Arc<Vec<u8>>`; it has no
`Vec<Scalar>` payload, public `values()` compatibility accessor, or second stored byte representation. The private
`scalar_values()` bridge materializes temporary kernel inputs and remains explicitly owned by Phase 9a2 rather than
stored state. All `ryft-core` test targets pass, including 1,151 library tests, the three allocation tests, the six
remaining integration tests, and the compile-fail contract; formatting and diff hygiene pass. Phase 9a1 is complete.

### Phase 9a2a logical and bitwise reference kernels (2026-08-05)

`Not`, `And`, `Or`, and `Xor` no longer decode through `Scalar` or allocate temporary logical input and output buffers.
The unary kernel traverses each logical element's physical byte range through `ArrayAddressing`, masks complements to
the declared Boolean or sub-byte width, and leaves layout holes and tile padding zero. The shared binary kernel reads
both physical inputs through their addressing descriptors, maps arbitrary NumPy-style broadcast coordinates into the
common output shape, and writes one zero-initialized layout-aware physical result. Full-width signed and unsigned
integers use the same bytewise truth tables, which are independent of signedness and host endianness.

The focused tests now cover Boolean truth tables, two-dimensional `(2, 1)` with `(1, 3)` broadcasting, multi-byte
integer complement, signed I2 masking, U4 bitwise combination, explicit strided storage with a preserved zero hole,
and the exact unsupported-floating-point diagnostic. All 1,151 `ryft-core` library tests and all 436 runnable
`ryft-xla` library tests pass (one intentional timing-sensitive ignore); formatting and diff hygiene pass. The broader
Phase 9a2 family checkbox remains open for the remaining arithmetic, comparison, complex, conversion, constructor, and
random kernels.

### Phase 9a2b random-bit reference kernel (2026-08-05)

The reference `RngBitGenerator` implementation now decodes ThreeFry and Philox states as checked `u64` elements and
constructs advanced states and U8/U16/U32/U64 bit arrays through the sealed typed codec. It no longer creates temporary
`Scalar` state or result vectors. The operation's existing counter advancement, integer narrowing, and algorithm output
remain unchanged, while direct backend calls now use the same canonical state-type diagnostic as operation inference.

Focused storage coverage exercises a negative-stride state, a positive-stride U16 result with physical holes, exact
logical values and physical bytes, U8 low-bit narrowing, and direct state-contract rejection. Existing ThreeFry and
Philox tests continue to cover U32/U64 word expansion, odd output counts, state advancement, dynamic output
materialization, and batching. All 1,152 `ryft-core` library tests and all 436 runnable `ryft-xla` library tests pass
(one intentional timing-sensitive ignore); formatting and diff hygiene pass. The broader Phase 9a2 family checkbox
remains open for the remaining arithmetic, comparison, complex, conversion, and constructor kernels.

### Phase 9a2c structural byte-copy reference kernels (2026-08-05)

Broadcast, transpose, reshape, static and dynamic slice/update, pad, and concatenate now move exact element encodings
between physical buffers through `ArrayAddressing`; none decodes the copied array payload through `Scalar`. Transpose,
reshape, and broadcast map logical output coordinates directly to addressed source and destination byte ranges. Slice
and update share two small block-copy methods that consume `ArrayIndexRanges`, bulk-copy physically contiguous runs
when the other side is dense, and fall back to addressed element copies only for arbitrary layouts. Copy-on-write
updates use `Arc::make_mut`, so an owned concatenation destination is filled in place while an update of a shared input
allocates exactly one independent output buffer. Pad initializes every logical output element from the rank-zero
padding encoding and then overlays addressed input elements. Concatenation no longer needs a temporary seed element or
an element-wise scalar buffer, including for `f8e8m0fnu`, which cannot represent zero.

Focused reference tests now pin reversed and holed strided inputs, exact output physical bytes, broadcast into a holed
layout, layout-clearing transpose/reshape/concatenate behavior, destination-layout-preserving updates, and exact U16
padding. The full core suite caught and the implementation corrected one validation-order regression: oversized
broadcast shapes must retain their established shape-element-count diagnostic before byte-span validation. All 1,152
`ryft-core` library tests and all 436 runnable `ryft-xla` library tests pass (one intentional timing-sensitive ignore);
formatting and diff hygiene pass. The structural parent remains open for gather/scatter, reduce, sort, dot/attention,
collectives, and control flow.

### Phase 9a2d Boolean and control-flow reference kernels (2026-08-05)

`Select`, scalar Boolean concretization, batched while-predicate reduction, and masked loop-state selection now read
and route encoded bytes directly through `ArrayAddressing`; none materializes temporary `Scalar` payloads. `Select`
converts branch elements only when data-type promotion requires it, preserves equal-typed encodings byte-for-byte,
and uses the shared broadcast-index mapping introduced by the logical kernels. This also closes an existing semantic
gap: reference selection now supports arbitrary right-aligned NumPy-style broadcasting rather than only equal element
counts and scalar expansion. Masked loop-state selection retains the congruent branch type and its physical layout,
including holes and negative strides.

Focused tests cover independently strided `(2, 1)` conditions, `(1, 3)` true branches, and `(2, 1)` false branches;
exact U16 result bytes; negative-stride while predicates; and holed negative-stride branch storage. All 38 focused
reference-array tests, all 1,152 core library tests, and all 436 runnable XLA library tests pass (one intentional
timing-sensitive ignore); formatting and diff hygiene pass. The structural parent remains open for gather/scatter,
reduce, sort, dot/attention, and collectives; the elementwise parent remains open for comparison, conversion,
arithmetic, complex, and constructor kernels.

### Phase 9a2e typed element dispatch, equality, and comparison (2026-08-05)

The reference backend now has one exhaustive runtime `DataType` dispatcher over the sealed Rust element codecs, plus
checked one-element encode/decode methods on the sealed `ArrayElement` contract and one generic broadcasted binary
loop. The loop decodes each addressed input element, applies a statically monomorphized typed function, and encodes
directly into the single output buffer; it allocates no input-sized temporary vectors or dynamic element
representation. A small private comparison capability records the only family distinction: real, integer, Boolean,
and low-precision values use partial ordering, while complex values support equality directions only.

Exact `Array` equality and `Compare` no longer decode through `Scalar`. Equality retains value rather than byte
semantics, so signed zeros compare equal and NaNs compare unequal. Comparison now supports signed and unsigned
sub-byte integers, arbitrary right-aligned broadcasting, layout-aware input reads and output writes, NaN unorderedness,
complex equality diagnostics, and the established vacuous behavior for empty payload-free arrays. Mixed-data-type
comparisons still delegate only their required promotion step to `ConvertElementType`; that dependency disappears with
the dedicated conversion slice rather than being duplicated here.

Focused tests pin F32 signed-zero and NaN equality, I2 `(2, 1)` by `(1, 3)` broadcasting, holed strided U16 inputs and
Boolean output bytes, low-precision NaN comparison, complex equality and ordered-comparison rejection, and empty versus
nonempty token behavior. All 38 focused reference-array tests, all 1,152 core library tests, and all 436 runnable XLA
library tests pass (one intentional timing-sensitive ignore); formatting and diff hygiene pass. The elementwise parent
remains open for conversion, arithmetic, complex, and constructor kernels.

### Phase 9a2f typed indexing and gather payload routing (2026-08-05)

Every reference indexing kernel now decodes index elements directly from `ArrayAddressing` ranges for all signed and
unsigned integer types, including I1/I2/I4/U1/U2/U4. Dynamic slice/update decode each rank-zero start without a
temporary scalar, while gather and scatter build one index addressing descriptor and reuse their coordinate buffers
across the complete traversal rather than allocating per query or materializing an index-value vector. The eager
sort/top-k index passenger likewise constructs its I32 elements directly through `Array::from_fn_elements`.

Gather now copies the selected operand encoding—or the element type's ordinary zero-constructor encoding for a
fill-or-drop query—straight into one layout-aware output buffer. It handles arbitrary operand and index layouts without
decoding the gathered payload or allocating an input-sized/result-sized secondary representation. Scatter's index side
is complete, but its payload combiner remains intentionally owned by the later typed arithmetic/reduction slice.
Promise-in-bounds and clipping gathers no longer construct an unused zero, so formats such as F8E8M0FNU that cannot
represent zero remain valid whenever no fill is semantically required.

Focused tests pin clamped signed sub-byte dynamic slicing, reversed U16 gather operands, reversed I4 gather/scatter
indices, fill-or-drop zero bytes, zero-free F8E8M0FNU promise-in-bounds gather, and the existing I64 indexing behavior.
All 1,156 core library tests and all 436 runnable XLA library tests pass (one intentional timing-sensitive ignore);
formatting and diff hygiene pass. The structural parent remains open for scatter payload combining, sort, reduce,
dot/attention, and collectives.

### Phase 9a2g direct observation and sort ranking (2026-08-05)

`Array::to_f64s`, `AbsDiffEq`, and eager sort key ranking now decode each logical element directly from its
`ArrayAddressing` range. One shared real-element decoder covers Booleans, every signed and unsigned integer width,
every low-precision floating-point format, BF16/F16, and F32/F64 without allocating a temporary dynamic payload.
Approximate equality retains exact fallback semantics for Boolean, integer, token, and structural-zero elements;
widens floating-point values exactly to F64; and compares both complex components after widening C64 components
before subtraction. Arbitrary physical layouts are traversed logically throughout.

Sort ranking now reads all ordered element types directly, including I1/I2/I4/U1/U2/U4 keys that the transitional
scalar bridge could not represent. Signed integers retain sign-biased two's-complement ranking, and every
floating-point format retains the established stable IEEE total ordering, including signed zero, infinities, and NaNs.
The now-unused `Scalar::total_order_rank` method was deleted rather than retained as dead compatibility machinery; no
new trait or public API was introduced.

Focused tests pin sub-byte conversion to F64, negative-stride I4 key sorting, and negative-stride F8E4M3FN approximate
equality. All 1,157 core library tests and all 436 runnable XLA library tests pass (one intentional timing-sensitive
ignore); formatting and diff hygiene pass. The structural parent remains open for scatter payload combining, reduce,
dot/attention, and collectives, while the elementwise parent remains open for conversion, arithmetic, complex, and
constructor kernels.

### Phase 9a2h direct scatter and reduction kernels (2026-08-05)

Scatter now performs its shared index/window traversal once and combines operand and update encodings directly in one
copy-on-write output payload. Overwrite preserves exact bytes, while add, multiply, minimum, and maximum decode only the
two elements being combined. Reduction likewise traverses arbitrary input layouts directly into one addressed output
buffer: sum and mean use typed arithmetic, minimum and maximum use the established ordered-element behavior, and any
and all use direct Boolean accumulation. Structural-zero arrays remain payload-free throughout.

A single private `ArrayArithmeticElement` capability centralizes zero, wrapping addition and multiplication, and mean
division for every numeric codec. It covers primitive and sub-byte integers, all low-precision floating-point formats,
BF16/F16/F32/F64, and C64/C128 without expanding the public array API. The abstract rules now reject unsupported
reduction and scatter combiner data types before execution. Focused tests pin reversed and holed layouts, sub-byte
wrapping, low-precision arithmetic, complex sum/mean, Boolean reduction, exact zero-free overwrite, and empty sum.

The current upstream JAX and StableHLO audit also exposed a pre-existing extrema gap that is deliberately recorded as
P9a2i rather than hidden inside this storage migration: Ryft does not yet implement JAX's Boolean reduction extrema,
IEEE floating-point maximum/minimum details, lexicographic complex extrema, or dtype-specific empty-reduction
identities consistently across reference and XLA execution. All 1,157 core library tests and all 436 runnable XLA
library tests pass (one intentional timing-sensitive ignore); formatting and diff hygiene pass. The structural parent
remains open for P9a2i, dot/attention, and collectives.

### Phase 9a2i JAX-compatible extrema (2026-08-05)

Reduction and scatter minimum/maximum now accept every Boolean and numeric element type, including complex values.
The reference backend shares one private typed extremum capability across both kernels: Booleans use conjunction and
disjunction; integers use their exact bounds; floating-point values propagate NaNs and order negative zero below
positive zero; and complex values compare `(real, imaginary)` lexicographically. Reduction initializes every output
with the exact dtype-specific identity, so empty reductions work uniformly for integers, Boolean values, finite-only
and infinite floating-point formats, and complex values without a first-element special case.

XLA lowering uses native StableHLO extrema for integers and Boolean values, JAX's explicit lexicographic
compare/select sequence for complex values, and explicit total-order comparison plus NaN propagation for real
floating-point values. The latter is intentional: CPU execution demonstrated that delegating directly to a backend
`maximum`/`minimum` instruction could lose NaNs and signed-zero ordering, while the explicit reducer preserves the
specified semantics across PJRT implementations. Complex and zero-free floating-point identities lower directly in
their declared dtype, and scatter reuses the same lowering helper as reduction.

Type-rule tests, reference tests, exact and structural StableHLO tests, and CPU execution tests pin Boolean extrema,
NaNs, signed zero, lexicographic complex values, dtype-specific identities, empty reductions, and both reduction and
scatter combiners. All 1,157 core library tests and all 438 runnable XLA library tests pass (one intentional ignore),
with XLA tests run serially to isolate the repository's concurrency-sensitive live-array telemetry assertion.
Formatting and diff hygiene pass. The structural parent remains open for dot/attention and collectives.

### Phase 9a2j direct generalized dot and structural closure (2026-08-05)

After any requested preferred-type promotion, the reference generalized-dot implementation now decodes each operand
element directly through its `ArrayAddressing` mapping and writes typed accumulators straight into one addressed
result buffer. Its logical index construction implements the full `[batching..., lhs_result..., rhs_result...]` output
order and arbitrary paired contracting axes. Arithmetic reuses the private typed zero/add/multiply capabilities
introduced for reduction and scatter, retaining modular sub-byte and integer behavior, per-step low-precision
rounding, complex accumulation, preferred accumulation-type semantics, and additive identities for empty
contractions. The contraction itself uses only rank-bounded temporary state rather than operand-sized payloads;
preferred accumulation continues to use the canonical element-conversion capability whose direct codec migration is
owned by the pending elementwise-conversion slice.

The obsolete public `dot_general_evaluate` flat-`Vec` helper and its private index walkers were deleted. Dot tests now
pin ordinary and batched contraction, positive and negative physical strides, narrow integer wrapping, complex
arithmetic, preferred accumulation types, and empty contracting dimensions. Attention requires no separate payload
kernel: its forward and backward reference implementations are compositions over dot and the other typed array
capabilities, and all attention composition tests pass through the new contraction. Collectives likewise have no
single-device reference-array payload evaluator; their array work is expressed by staging, transformation, and XLA
lowering, so there was no scalar payload bridge to replace.

All 1,157 core library tests and all 438 runnable XLA library tests pass (one intentional ignore), with XLA tests run
serially to isolate the concurrency-sensitive live-array telemetry assertion. Formatting and diff hygiene pass. The
structural migration parent is complete; the next review unit is the cross-family addressing/traversal cleanup.

### Phase 9a2 cross-family addressing and observation closure (2026-08-06)

`ArrayAddressing` now provides crate-private prevalidated multi-index byte mapping and row-major index advancement for
reference kernels. Reduction, generalized-dot operand reads, transpose, broadcast, pad, gather, and scatter delegate
their physical lookup and repeated index advancement to that contract instead of reconstructing flat row-major
indices or open-coding local odometers. The kernels retain only operation-specific coordinate construction. Gather,
scatter, and dynamic slicing share the same direct typed-index decoder over addressed logical multi-indices.

The existing static/dynamic slice and update paths already share `copy_block` and `replace_block`, both driven by
`ArrayIndexRanges`; the audit retained those two direction-specific methods because each has a distinct dense-side
bulk-copy fast path and extracting a generic byte-copy cursor would add more machinery than duplication removed.
Exact equality, approximate equality, display, Boolean concretization, and indexing were also audited and remain
direct typed-codec/addressing consumers with arbitrary-layout coverage. The focused 40-test reference-array suite,
all 1,157 core library tests, and all 438 runnable XLA library tests pass (one intentional ignore). The review unit is
net code-negative outside its focused addressing tests and adds no public API.

### Phase 9a2k direct element-type conversion (2026-08-06)

Reference-array conversion now decodes and encodes one addressed element at a time through two private typed
capabilities: source types select an exact signed-integer, unsigned-integer, real, or complex conversion category, and
destination types implement that category's conversion without a dynamic scalar value or payload-sized intermediate.
The implementation covers every pair of Boolean, signed and unsigned sub-byte and native integer, low-precision,
BF16/F16/F32/F64, and C64/C128 data types. Same-type conversion shares the original byte storage to preserve NaN
payloads and layout padding exactly; token and structural-zero rules retain their established diagnostics.

The exhaustive conversion test instantiates all 1,024 materialized source/destination pairs and separately pins
Boolean truth conversion, modular sub-byte narrowing, complex component handling, exact and fallible low-precision
encoding, negative-stride traversal, and same-type storage sharing. `Array::from_f64s` now uses the same destination
codec contract directly, removing another production constructor dependency on `Scalar`.

This migration exposed one latent block-quantization dependency on the obsolete scalar encoder's saturating
`f8e4m3fn` overflow. The portable block-quantization composition now clamps normalized elements explicitly to the
format's finite range before conversion, preserving the OCP MX saturation rule independently of StableHLO conversion
overflow policy and keeping reference and compiled execution aligned.

All 1,157 core library tests, 54 runnable core doctests (16 intentional ignores), and all 438 runnable XLA library
tests (one intentional timing-sensitive ignore) pass. Formatting and diff hygiene pass. The Phase 9a2 elementwise
parent remains open for arithmetic, complex, and constructor kernels.

### Phase 9a2l direct core arithmetic kernels (2026-08-06)

`Abs`, `Neg`, `Add`, `Sub`, `Mul`, `Div`, `Rem`, `Max`, and `Min` now decode and encode sealed element types directly.
The binary operations share one generated promotion-and-broadcast path: already-matching operands are borrowed and
the result is the only payload-sized allocation, while mixed element types reuse the canonical direct conversion
kernel for only the mismatched operands. The path supports complete NumPy-style broadcasting and arbitrary physical
input layouts.

Per-element capabilities preserve modular native and sub-byte integer arithmetic; exact integer division/remainder
errors; checked low-precision re-encoding; complex magnitude, arithmetic, and overflow-resistant division; and exact
NaN-payload and signed-zero extremum selection. `Array * f64` now constructs its typed rank-zero factor through the
direct conversion codec and delegates to the same multiplication kernel rather than reconstructing a scalar payload.

Focused tests pin mixed-type multidimensional broadcasting across reversed layouts, sub-byte wrapping, low-precision
operations, integer error diagnostics, stable large-complex division, and exact floating-point extremum encodings.
All 1,157 core library tests, 54 runnable core doctests (16 intentional ignores), and all 438 runnable XLA library tests
(one intentional timing-sensitive ignore) pass. Formatting and diff hygiene pass. The Phase 9a2 elementwise parent
remains open for transcendental/sign/rounding math, complex construction and accessors, and constructors.

### Phase 9a2m direct floating-point math kernels (2026-08-06)

`Sin`, `Cos`, `Atan2`, `Exp`, `Log`, `Sqrt`, `Rsqrt`, `Tanh`, `Logistic`, `Erf`, `Pow`, `Sign`, `Floor`, `Ceil`, and
`Round` now decode and encode sealed element types directly. Three private element contracts group operations by their
actual dtype families: real/complex floating-point math, real-only error and rounding math, and sign extraction. The
real implementations retain each format's established working precision and checked low-precision re-encoding; the
complex implementations retain the stable extreme-value sine/cosine and division formulas. `Erf` shares only the pure
double-precision evaluator with the scalar backend, never a scalar value or payload representation.

Unary kernels preserve arbitrary physical layouts with one result allocation. `Atan2` and `Pow` reuse the canonical
promotion kernel and complete NumPy-style typed broadcasting, allocating promoted operands only when their data types
differ. IEEE signed zeros and NaNs, ties-to-even rounding, complex analytic continuations, and existing invalid-type
diagnostics remain intact. Direct sign extraction additionally makes the operation executable for I1/I2/I4 arrays,
which the operation contract supported but the transitional scalar representation could not encode. The obsolete
payload-materializing `unary`, `binary`, and scalar-broadcast helpers are deleted.

Focused tests pin negative-stride unary traversal, mixed F32/F64 multidimensional `Atan2` and `Pow` broadcasting,
low-precision re-encoding, real rounding/error behavior, exact signed-zero and NaN sign encodings, and signed sub-byte
sign extraction. All 1,159 core library tests, 54 runnable core doctests (16 intentional ignores), and all 438 runnable
XLA library tests (one intentional timing-sensitive ignore) pass. Formatting and diff hygiene pass. The Phase 9a2
elementwise parent remains open only for complex construction/accessors and zero/one/fill/iota constructors.

### Phase 9a2n direct constructors and Phase 9a2 closure (2026-08-06)

The existing complex construction, conjugation, and real/imaginary accessor kernels were audited and already used the
shared typed element loops directly, so they required no migration. Zero, one, zero-like, one-like, and iota now
construct addressed output storage through sealed element dispatch without a scalar payload. This adds the sub-byte
integer behavior their operation contracts already admitted, preserves `f8e8m0fnu`'s no-zero zero-like semantics,
retains payload-free Token/Zero handling, and writes arbitrary physical layouts directly.

Eager and staged array fill now accept sealed typed host elements directly. Each implementation creates one rank-zero
typed `Array`, applies the canonical element conversion and memory placement, and uses ordinary broadcast for the
requested output; `fill.rs` no longer imports, matches, or otherwise depends on the scalar backend. The generic
transform forwarding is payload-agnostic, while the scalar backend retains its temporary scalar-domain behavior in
its own module. No scalar sequence or logical-byte intermediate is materialized. The obsolete `scalar_values`,
`from_scalar_values`, `from_scalar_element`, and `zero_element` bridge family was deleted, and its remaining tests now
construct typed elements directly.

Allocation regression tests compare one-element and 4,096-element unary, binary, fill, and iota executions. Allocation
counts stay constant and allocated-byte growth is exactly one output payload; no payload-sized intermediate remains.
The residual production-kernel audit finds no `Vec<Scalar>` bridge and no unchecked byte-offset arithmetic outside
`ArrayAddressing`; row-major arithmetic that remains computes logical broadcasting or contraction coordinates only.

All 1,158 core library tests, the five allocation regression tests, 54 runnable core doctests (16 intentional ignores),
and all 438 runnable XLA library tests (one intentional timing-sensitive ignore) pass. Formatting and diff hygiene
pass, closing the complete Phase 9a2 gate. Phase 9a3 exact XLA literals is the next isolated implementation unit.

### Phase 9a3 exact XLA literals (2026-08-06)

Concrete array literals now have one byte-oriented lowering path. Layout-free arrays lend their physical storage
directly to MLIR; explicitly laid-out arrays traverse `ArrayAddressing` once to produce logical row-major bytes; and
big-endian targets normalize each multi-byte scalar component before calling MLIR's native raw-buffer API. Boolean,
I1, and U1 alone use MLIR's typed Boolean constructor because it owns the required one-bit packing. I2/I4/U2/U4 use
the raw API's specified one-byte-per-element representation. StableHLO rejects `ui1`, so U1 uses the same physical
signless-i1 carrier as Boolean and I1 while Ryft's operation metadata continues to select unsigned semantics.

The old per-data-type typed-vector reconstruction, rank-zero `Scalar` reconstruction, and unused scalar-splat hook
were deleted. Existing layout-free byte-aligned constants therefore go from one payload-sized Rust allocation to
zero before entering MLIR: a 4,096-element F32 literal removes one 16,384-byte `Vec<f32>`. Explicit physical layouts
retain exactly one required logical-order byte buffer, while one-bit values retain MLIR's typed packing path because
its raw-buffer API cannot unambiguously represent every short non-splat bit sequence. Phase 9a0's borrowed-slice
construction result remains unchanged at three allocations and 16,440 allocated bytes because Phase 9a3 changes
lowering, not `Array` ownership. Existing StableHLO fixtures did not change, so their
module size and downstream compile/runtime input remain byte-for-byte stable; the expanded 16-literal CPU compile and
execution smoke completes in approximately 0.06 seconds on the baseline host. The production and test cleanup removes
86 net lines from `lowering.rs` despite adding complete sub-byte and layout coverage.

Exact tests now pin all low-precision raw encodings and their StableHLO renderings; BF16/F16/F32/F64 signed zero and
NaN payloads; U64 values beyond F64's exact range; interleaved C64/C128 payloads; empty tensors; explicit strided
physical storage; and all six signed/unsigned sub-byte integer families. CPU execution returns exact host bytes for
every supported standard, complex, empty, and sub-byte case. The pinned upstream CPU StableHLO-to-HLO converter aborts
when asked to execute the F4/F6/F8 formats, so those formats retain exact attribute and StableHLO coverage rather than
claiming unsupported CPU execution.

Local gates pass: all 438 runnable `ryft-xla` library tests (one timing-sensitive ignore), the five array allocation
guards, formatting, and diff hygiene. After the Phase 9a3 source was committed and pushed, revision
`65bc37eafcdf46822c47eee0ca51a28bfecb46d4` was cloned into an isolated DGX Spark checkout and verified with the
CUDA 13 feature. All six focused literal tests pass, including exact CPU execution, and the complete serialized
CUDA-feature library suite passes with 439 tests and one timing-sensitive ignore. This closes the Phase 9a3 gate;
Phase 9a4 scalar-program-universe retirement is the next isolated implementation unit.

### Phase 9a4 scalar program universe retirement (in progress, 2026-08-06)

The gradient-checking macro now has one array-valued implementation instead of parallel scalar and array selectors.
Rank-zero F64 and C128 arrays pin scalar-function coverage through the same reverse-mode and finite-difference path as
rank-positive arrays. The public selector and the scalar-only central-difference, complex-perturbation, and assertion
branches were deleted; existing array call sites now use the direct `check_gradient!(function, ...)` form. This first
Phase 9a4 slice removes 85 net lines and leaves no `@scalar` or `@array` gradient syntax. All four focused macro tests,
the complex absolute-value differentiation tests, and the error-function differentiation test pass. The complete
1,158-test core library suite and all 54 runnable doctests (16 intentional ignores) also pass. The reusable generic-
transform test migration remains the next Phase 9a4 unit.

The next migration slice moves the foundational program machinery to rank-zero arrays. `TestRegionOperation`, which
exists solely to isolate generic attached-region behavior, now operates on `ArrayType`; this moves its complete
builder, region-graph, capture-region, and interpretation-replay consumer graph to `Array` without creating a renamed
scalar test value. Atom and operation-formatting doctest coverage also use arrays, and the entire `programs::builders`
test module now exercises both construction and eager interpretation through `ArrayOperation<Array>`. Expected
program renderings change only by making the rank-zero shape explicit (`f64[]`).

This review unit changes 516 lines across eight source files with four net deletions. It removes every retired scalar
identifier from `programs::{atoms,builders,operations,regions}` and reduces the core residual audit from 408 matches
across 63 files to 365 matches across 59 files. All 121 program-layer tests, the five directly affected capture and
interpretation tests, the complete 1,158-test core library suite, and all 54 runnable doctests (16 intentional ignores)
pass. Generic transform and operation-local fixture migration remain open in the checklist above.

The context-and-tracing slice moves every generic context and tracing fixture to rank-zero `Array` values and
`ArrayType` signatures. The two test-only operations that isolate staging and invalid-output behavior now use
`ArrayType` directly, and rendered-program expectations expose rank-zero shapes as `f64[]`. This preserves the tests'
universe-neutral contracts while removing their dependency on the scalar backend.

This review unit changes 559 lines across two test modules with 51 net additions, caused by explicit rank-zero type and
value construction rather than new machinery. It removes all retired scalar identifiers from `contexts.rs` and
`tracing.rs`, reducing the core residual audit from 365 matches across 59 files to 289 matches across 57 files. All 39
focused context and tracing tests and the complete 1,158-test core library suite pass. Partial evaluation and batching
are the next bounded generic-transform fixture unit.

The partial-evaluation slice moves all seven generic partial-evaluation fixtures to rank-zero arrays, including eager
folding, staged known-side evaluation, residual replay, program partitioning, effect placement, poison propagation, and
constant recovery. Boolean concretization now uses `bool[]` values explicitly instead of relying on the retired scalar
backend's numeric truthiness, and the one value reused after becoming a residual input is cloned because `Array` is not
`Copy`. The generic batching fixtures were already fully array-backed, so their half of this checklist unit required no
source changes.

This review unit changes 371 lines in one test module with 15 net additions, all from explicit rank-zero type/value
construction and formatting. It removes every retired scalar identifier from `partial.rs`; batching remains clean; and
the core residual audit falls from 289 matches across 57 files to 260 matches across 56 files. All seven focused partial
evaluation tests and the complete 1,158-test core library suite pass. Differentiation, custom derivatives, and
rematerialization are the next generic-transform fixture unit.

The first differentiation slice migrates the generic elementwise rule fixtures and every custom-derivative and
rematerialization fixture to arrays. Rank-zero array tests now cover elementwise tangent/cotangent alignment,
zero-tangent propagation, zero-space outputs, custom-JVP affine-tangent rejection, token/zero boundaries, tagged
rematerialization, nested rematerialization, and nested second-order reverse mode. Four scalar-only tests—three repeated
custom-derivative cases and one unconstrained-policy rematerialization repetition—were deleted because existing array
tests pin the same behavior more completely.

This review unit changes 387 lines across three test modules with 109 net deletions. It removes all retired scalar
identifiers from the three modules and reduces the core residual audit from 260 matches across 56 files to 227 matches
across 53 files. All 77 focused tests and the complete 1,154-test core library suite pass. Forward-mode and reverse-mode
differentiation fixtures remain as separate review-sized substeps.

The forward-mode differentiation slice moves all six formerly scalar-backed JVP and linearization fixtures to rank-zero
arrays. Program construction, fused JVP rendering, direct linearization, pullback construction, eager and traced
transform entrypoints, nested transform composition, complex and half-precision arithmetic, symbolic-zero
canonicalization, and token/zero-space boundaries now run through `ArrayOperation<Array>`. The eager host-control-flow
cases use an explicit Boolean primal and structural-zero tangent, because array concretization intentionally accepts
only `bool[]` instead of the retired scalar backend's numeric truthiness.

This review unit changes 345 lines in one test module with 19 net additions, all from explicit rank-zero array types,
payload-free token/zero construction, ownership-preserving clones, and array payload assertions. It removes all 104
previously audited scalar-reference lines from `differentiation::forward`. All nine focused forward-mode tests and the
complete 1,154-test core library suite pass. Reverse-mode differentiation fixtures are the next bounded substep.

The reverse-mode differentiation slice moves the remaining scalar-backed transpose, VJP, gradient, auxiliary-output,
holomorphic, and nested-differentiation fixtures to rank-zero arrays. Its test-only malformed-transpose operation family
now uses `ArrayType`, so builder ownership, structural-zero materialization, known-producer replay, attached-region
replay, effect rejection, and transposition diagnostics are tested without retaining a second type universe. Complex,
E8M0, token, zero-space, traced, and non-`Copy` boundaries retain their prior coverage. As in forward mode, eager host
control flow now receives an explicit Boolean primal because array concretization accepts only `bool[]`.

This review unit changes 581 lines in one test module with 21 net additions, all from explicit rank-zero types, array
payload extraction, payload-free value construction, and ownership-preserving clones. It removes all 150 retired scalar
references from `differentiation::reverse`; the sole remaining `Scalar` substring belongs to the unrelated
`NonScalarGradientOutput` diagnostic variant. All 17 focused reverse-mode tests and the complete 1,154-test core library
suite pass. This closes the generic context, tracing, partial-evaluation, batching, differentiation, custom-derivative,
and rematerialization fixture migration. Remaining operation-local scalar fixtures are the next Phase 9a4 unit.

The first operation-local slice moves constants, comparisons, logical operations, element-type conversion, and
stop-gradient fixtures to rank-zero arrays. Repeated scalar assertions were deleted where rank-zero or rank-positive
array cases already covered the same contract. `check_operation_partial_evaluation!` now has one canonical array-backed
concise form rather than a scalar default or a transitional concise backend selector. Every concise scalar-backed caller
has migrated, including `select`, all 24 mathematical callers, the macro examples, and the macro's own tests. The
pre-existing explicit-case backend form remains available for genuinely different value families such as `ArrayIrValue`.

This review unit changes 534 lines across 38 Rust source files with 48 net additions, principally from making rank-zero
array construction explicit at the mathematical call sites. It removes 92 retired scalar references and reduces the
operation-local residual audit from 410 matches across 40 files to 318 matches across 28 files. All 53 focused partial-
evaluation tests, all 37 macro integration tests, the complete 1,154-test core library suite, and all 54 runnable
doctests (16 intentional ignores) pass. Control-flow fixtures are the next bounded operation-local slice.

The control-flow slice removes the final scalar-backed fixtures from `operations::control_flow`. `select` retains its
piecewise JVP and VJP scenarios over rank-zero arrays, while its duplicate `DataType` inference and scalar eager-value
assertions are deleted. The scalar staged-`while` JVP/linearization fixture is also deleted because the adjacent array
tests already pin the same numerical behavior, fused-loop structure, and closed-knownness linearization separately.
The `Select` example now uses honestly Boolean array storage, and the repository testing guidelines now name arrays and
the selector-free `check_gradient!` form as the canonical reference path.

This review unit changes 148 lines across two Rust source files with 112 net deletions, plus a 16-line testing-guideline
correction. It removes 26 retired scalar references and reduces the operation-local residual audit from 318 matches
across 28 files to 292 matches across 26 files. All 76 focused control-flow tests, the complete 1,153-test core library
suite, and all 54 runnable doctests (16 intentional ignores) pass. Mathematical and complex-number fixtures are the
next bounded operation-local slice.

The arithmetic, ordering, clamping, sign, and rounding slice moves all remaining scalar-backed fixtures in `add`,
`sub`, `mul`, `div`, `rem`, `neg`, `min`, `max`, `clamp`, `ceil`, `floor`, `round`, and `sign` to rank-zero arrays.
Mixed-precision promotion, low-precision tangent widening, smallest-positive tangents, signed remainder and integer
zero-divisor handling, NaN propagation, signed-zero ordering, ties-to-even rounding, and complex arithmetic retain
operation-local coverage on the canonical reference backend.

This review unit changes 292 lines across 13 Rust source files with 40 net deletions; the plan separately gains the
checklist split that bounds the remaining review units and this review record. It removes 111 retired scalar references
and reduces the operation-local residual audit from 292 matches across 26 files to 181 matches across 18 files.
Formatting and diff hygiene pass, and the complete 1,153-test core library suite passes. Transcendental and special-
function fixtures are the next bounded operation-local slice.

The transcendental and special-function slice moves all remaining scalar-backed fixtures in `abs`, `atan2`, `cos`,
`erf`, `exp`, `log`, `logistic`, `pow`, `rsqrt`, `sin`, `sqrt`, and `tanh` to rank-zero arrays. The migrated coverage
retains mixed and low precision, complex principal branches and holomorphic derivatives, overflow-resistant
differentiation, exact error-function regimes, signed special values, and extreme complex trigonometric results. The
two tests that inspect complex infinities decode the typed array payload explicitly; all other approximate checks use
the array backend's existing comparison contract directly.

This review unit changes 313 lines across 12 Rust source files with nine net additions from explicit typed rank-zero
expectations. It removes 139 retired scalar references and reduces the operation-local residual audit from 181 matches
across 18 files to 42 matches across six files. Formatting and diff hygiene pass, and the complete 1,153-test core
library suite passes. Complex construction and projection fixtures are the final mathematical/complex-number subunit.

The complex construction and projection slice moves the remaining conjugation, part extraction, complex construction,
JVP, and non-holomorphic gradient fixtures to rank-zero arrays. Duplicate scalar eager assertions for conjugation and
part extraction were deleted because the existing vector-array assertions cover the same contracts; the construction
assertion now uses rank-zero F32 arrays so concrete C64 materialization remains pinned. Structural-zero tangent
materialization and the bilinear complex-gradient convention retain direct operation-local coverage.

This review unit changes 101 lines in `operations::complex` with 39 net deletions. It removes all 35 scalar references
from that module and reduces the operation-local residual audit from 42 matches across six files to seven matches
across five files. Those seven references are production documentation for scalar values or the still-live scalar
`WhileOperation` specialization rather than test fixtures, so this closes the operation-local fixture checklist; the
following scalar-universe deletion item owns their removal or rewording. Formatting and diff hygiene pass, and the
complete 1,153-test core library suite passes.

The first scalar-backend dependency-removal slice moves the final capture and interpretation fixtures to rank-zero
arrays. Capture validation, pruning, lifting, nested-region discovery, duplicate output materialization, live-constant
lifting, mismatched parameter structures, and malformed operation arity all retain their previous coverage through
`ArrayType`, `ArrayOperation`, and `Array`. Rendered capture programs change only by making rank-zero shapes explicit,
and closures that previously copied `Scalar` values now clone the byte-sharing reference `Array` value.

This review unit changes 165 lines across `captures.rs` and `interpretation.rs` with one net deletion, plus the nested
dependency-removal checklist. It removes all 18 retired scalar-backend references from those modules and reduces the
exact backend-identifier audit from 117 matches across 13 files to 99 matches across 11 files. All 18 focused tests,
formatting, diff hygiene, and the complete 1,153-test core library suite pass. Program fixtures are the next dependency-
removal unit.
