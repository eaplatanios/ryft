# First-class dimension architecture cleanup

## Status

Active; repository state was re-audited on 2026-07-28 after the original side-chat history was lost. Phases 0–2 and
P3a–P3i are committed on `u/eaplatanios/dynamic-shapes` at `21c7442fee3e94eb422b440cc25b479691526df5`.
The owner checkout currently contains one unstaged P3j shaped-zero prototype plus this plan revision. That prototype is
nearly ready to land. The third implementation correctly removes `DimensionType::known_extent`, keeps concrete
extents out of structural types and cache identity, uses the region's complete closed identity authority
allocation-free, narrows `TypeRefinements` to the identity slice it needs, validates eager constructor bounds, and
proves retained-specialization reuse. Its constructor contract, eager materialization, batching policy,
differentiation behavior, transposition, and XLA lowering pass the core/XLA suites. The remaining P3j acceptance work
is narrow: exercise actual cross-program instantiation/import rather than only direct operation renaming, make both
runtime extents in the XLA axis-pairing fixture distinct from each other as well as from the static axis, deliberately
refresh the stale macro trybuild diagnostic, and remove an unrelated whitespace-only diff. Phase 4 has not begun as a
complete migration.

This plan remains a containment and simplification follow-up to `.tasks/plan_first_class_dimension_programs.md`. It
preserves that plan's user-visible capabilities and its decision to represent runtime dimensions as ordinary SSA
values, but supersedes the following implementation compromises:

- one operation payload implementing materially different `Operation<ArrayType>` and
  `Operation<ArrayProgramType>` contracts;
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

The current operation trait still permits the same payload to implement multiple type-family contracts, but the
original inventory is no longer current. As of the 2026-07-28 audit:

- `ConcatenateOperation`, `CustomCallOperation`, `PadOperation`, and `RngBitGeneratorOperation` directly implement
  both `Operation<ArrayType>` and `Operation<ArrayProgramType>`;
- `CompareOperation` has a generic homogeneous `Operation<T>` implementation plus its distinct composite dimension
  comparison contract;
- `BroadcastOperation`, `ReshapeOperation`, and `DimensionSizeOperation` now have one canonical composite contract,
  while their remaining homogeneous behavior lives under explicit legacy payloads where still needed; and
- dynamic slice, gather, reduce, and ordinary slice are intentionally array-only. The archived slice-scatter proposal
  was never introduced.

The remaining five dual-contract payloads are transitional and still block Phase 8's associated-type operation
contract. Constructor mixed semantics must continue to live in the composite variant arm rather than adding a sixth
dual-contract payload.

The archived inventory also included overlapping generic constructor contracts, which are a distinct and harder
case:

- `ZeroOperation<ArrayType>`;
- `OneOperation<ArrayType>`;
- `FillOperation<ArrayType, V>`; and
- `IotaOperation<ArrayType>`.

The current branch has removed those four archived array-program-specific implementations. `One`, `Fill`, and `Iota`
appear only in the homogeneous array family. One temporary exception remains:
`ArrayProgramOperation::Zero(ZeroOperation<ArrayProgramType>)`, which lets generic differentiation machinery
materialize a composite array zero without explicit geometry. The final constructor design cannot be resolved merely
by retaining that generic escape hatch. The canonical destination is:

- operand-relative `zero_like`/`one_like` for transform-generated values whenever a source array exists;
- a homogeneous nullary constructor only when its stored output type is identity-free, enforced by a generic
  identity-free rule in the blanket `Operation<T>` inference of zero, one, fill, and iota; and
- for identity-bearing output types, a mixed `Operation<ArrayProgramType>` contract owned by the corresponding flat
  `ArrayProgramOperation` variant arm over `ZeroOperation<ArrayType>`, `OneOperation<ArrayType>`,
  `FillOperation<ArrayType, V>`, or `IotaOperation<ArrayType>`. The stored `ArrayType` is the complete output
  authority and the variant consumes one explicit dimension operand per dynamic axis of its stored shape, in axis
  order, validated by identity — the same contract every mixed shape-carrying operation follows.

No wrapper type exists and no payload carries two trait implementations: the mixed contract is owned by the
`ArrayProgramOperation::DynamicZero` variant arm, which delegates rendering, identity renaming, and structural flags
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
- occurrences and files containing `ArrayProgramProjection`, `ArrayContextView`, `DimensionContextView`,
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
- A shape-carrying operation is mixed even when a particular invocation happens to have only static dimensions.
- A static convenience API may omit dimension operands only when the payload's metadata proves there are none; it
  still binds the same mixed operation contract. Constructors are the documented exception: an identity-free
  constructor has one canonical encoding inside the homogeneous array member family, and the mixed dynamic
  constructor rejects reference-free stored types so equivalent zeros cannot acquire two enum representations.

### Contained heterogeneity

- `ArrayProgramType` and `ArrayProgramValue<A>` remain the single storage sum for an array program.
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
operation type whose associated type is `ArrayProgramType`; it adapts inner array and dimension operation families.

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

The exact borrowed-view bounds are a prototype decision, but returning an owned `A` from `&ArrayProgramValue<A>` is
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

- eager `ArrayProgramValue<A>` borrows `A`/`DimensionValue` for read-only projection and transfers ownership without
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
ArrayProgramOperation<A>  Operation<Type = ArrayProgramType>
```

The names may be simplified after the migration, but the roles are fixed.

- `ArrayPrimitiveOperation` contains operations whose complete signature is array-only.
- `DimensionOperation` contains dimension-only operations.
- `ArrayProgramOperation` is the sole stored dispatcher and public array-program operation family. It projects the two
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
- `ArrayProgramType` and `ArrayProgramValue<A>` as the storage sum;
- first-class dimension arithmetic and requirement operations;
- interval, congruence, order, equality, and constant abstract interpretation;
- `Effect::OrderedAssertion`;
- the bounded-input ABI, hidden physical extents, and `PadToStatic`;
- existing StableHLO dynamic-shape operands; ordered runtime assertion lowering and per-effect token separation remain
  Phase 7 work;
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
- [x] Inventory every `Operation<ArrayType>` and `Operation<ArrayProgramType>` implementation and classify it as
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
- [x] P3g Delivery A: retain cross-member primitives as flat `ArrayProgramOperation` variants and specify one explicit
      dimension operand per reshape/broadcast output axis. Exact constants represent static axes and inference derives
      the output shape from operand types. The rejected `DimensionOperandSchema`, nested-family prototype, and
      projection-aware-derive analysis are recorded in the P3g plan; none remains in production.
- [x] P3g Deliveries B–D: migrate reshape, migrate broadcast, and close the combined transform/lowering vertical slice
      before freezing their legacy homogeneous contracts for consumer-by-consumer deletion. Delivery B landed at
      `1aeea5329`, Delivery C landed at `7aef33d93`, and Delivery D landed at `a4f2c833`. The combined acceptance
      program preserves dimension arithmetic through eager execution, partial evaluation, batching, JVP, import,
      direct StableHLO lowering, and PJRT compilation/execution. The exact legacy-consumer and deletion manifest is in
      `.tasks/plan_p3g_reshape_broadcast.md`.
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
- [ ] Delete P1c's temporary result-reference producer fallback once every shape-producing operation carries its
      first-class result-dimension operands. After this point, a fresh output reference without an available operand or
      a definition-position occurrence is a closure error.
- [ ] Delete each frozen homogeneous reshape/broadcast implementation and transform rule as its owning Phase 4–9
      consumer migrates. Do not attempt the final zero-residual deletion before the composite public capability and
      transform domains replace the current homogeneous `Reshape`/`Broadcast` implementations.
- [x] Complete and remove the remaining-operation inventory through P3i's explicit mixed migration or array-only
      classification proof: custom call, dynamic slice, gather, pad, reduce, RNG bit generation, slice, and the
      archived slice-scatter proposal.
- [ ] P3j: constructor contracts. The wrapper-based Delivery A from
      `.tasks/plan_p3j_shaped_constructors.md` (Delivery A) was implemented and then replaced by the
      stored-type-authoritative design recorded in the diagnosis and target-architecture sections, corrected by a
      second review pass (see the constructor revision notes in the Review section). The current unstaged prototype has
      the right variant-owned `DynamicZero(ZeroOperation<ArrayType>)` contract, canonical routing, eager rule,
      replicated batching rule, structural differentiation behavior, and direct bounded XLA lowering. The
      `known_extent` leak is now gone and the structural boundary path works, but the phase remains incomplete until
      actual cross-program import, the strengthened XLA operand-order fixture, and the full verification gates below
      pass. Do not mark P3j complete merely because the current 994 core and 408 XLA tests pass.
- [x] P3j boundary-refinement correction: remove `DimensionType::known_extent`,
      `DimensionType::with_known_extent`, all observed-extent logic from `ArrayProgramValue::r#type`, and the
      corresponding equality/hash/display/renaming and `DimensionValue` checks. Do not replace them with a new
      `Value`, `Typed`, `Type`, or array-program-specific boundary-evidence abstraction.
- [x] Let a concrete output establish the first refinement for any identity already established by the formal input
      signature as well as for an identity defined internally. This is sound without inspecting runtime payloads:
      structural region closure already rejects an instruction result reference unless that instruction consumes or
      defines the identity (or is temporarily classified as an internal definition by P1c's fresh-reference fallback),
      and each mixed eager rule validates or constructs its concrete output from those explicit operands. The
      allocation-free one-vector/input-split representation is implemented.
      `TypeRefinements::validate` is parameterized by the complete identity *slice* (`&[T::Identity]`) rather than the
      `TypeIdentitySignature` container; refinement validation needs authority membership but not the signature's
      input/internal partition.
- [x] Add focused negative tests proving that an already-established output identity without a consumed/defined edge
      fails closure and that two concrete outputs for one input-owned identity must agree. Preserve the temporary
      fresh-result-reference fallback until the explicit-operand migration item above deletes it; do not add a test
      that contradicts that documented transitional behavior. Positive non-exact eager/refinement coverage exists for
      reshape, broadcast, and `DynamicZero`.
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
- [ ] Add a non-exact `DynamicZero` cross-program instantiation/import test. Direct operation identity renaming,
      alpha-renamed boundary interpretation, and transposition are covered, but the current test named
      `test_array_program_dynamic_zero_alpha_renamed_instantiation` does not call the program instantiation or splice
      path. Keep the transpose test proving the array cotangent is ignored while every explicit extent operand receives
      a structural-zero cotangent.
- [x] Re-audit the pending generic fused-JVP zero-primal reuse and Jacobian validation reordering after the boundary
      correction. Retain each only if an independently named regression test requires it; otherwise remove it as
      constructor-driven global complexity. The UFCS-only `fill` and `iota` residue has been removed. If the Jacobian
      ordering remains temporarily necessary until Phase 6 removes dynamic nullary tangent materialization, document
      that dependency and keep the existing exact non-finite-coordinate diagnostics tests as its gate. The zero-primal
      reuse is independently covered by the shaped-zero JVP regression. The Jacobian reordering now names its Phase 6
      dependency and remains gated by the exact non-finite-coordinate diagnostics tests.
- [ ] Strengthen the mixed static/dynamic XLA execution fixture with distinct static and runtime dynamic extents so its
      observed output shape catches axis/operand pairing mistakes rather than only an out-of-bounds operand index. All
      three axis extents must be pairwise distinct; the current `3 x 2 x 3` fixture cannot detect a swap between the two
      dynamic operands.
- [ ] P3j gate: `DimensionType` is exactly identity plus bounds; `Typed::r#type` is structural; dynamic zero has one
      canonical explicit-operand encoding; reference eager execution, PE, batching, JVP/transpose, import, direct XLA
      lowering, CPU PJRT execution, exact diagnostics, and retained specialization reuse all pass.
- [ ] Land `One`, `Fill`, and `Iota` as complete vertical slices mirroring zero: the position-aware reference guard on
      their blanket inference, variant-owned mixed contracts through the shared helper, canonical `From` routing,
      composite eager rules, batching arms, lowering, and tests. Land them as separate review units after corrected
      `DynamicZero`; a slice may coincide with its first composite consumer, but all three must be complete before the
      Phase 3 gate. The first implementation of their guards and payload-level mixed contracts was deliberately
      reverted for reviewability; do not restrict their homogeneous contracts before the replacement is usable.
- [ ] Route transform-generated zero/one values through structural zero or `zero_like`/`one_like` whenever an operand
      supplies geometry.
- [ ] Migrate transform consumers that stage `ZeroOperation<ArrayType>` with possibly-dynamic types (condition, scan,
      while, gather, scatter, slice, pad, and differentiation rules) to `zero_like`, structural zeros, or Phase 6
      extent residuals, with dynamic-shape acceptance tests per consumer. The zero reference guard is not complete
      until this lands.
- [ ] Keep `DimensionType` strictly identity plus bounds throughout all later phases. Concrete extents are runtime
      values and output-refinement observations, never part of `Typed::r#type`, structural equality, hashing,
      rendering, persistence, or cache identity.
- [ ] Sweep shape-changing collectives and every other operation whose result metadata references first-class
      dimension operands.
- [ ] Give each mixed operation one direct positional operand contract and migrate its inference, eager rule,
      transforms, and lowering to preserve those same SSA edges.
- [ ] Delete copied dimension operand identity, bounds, and ordering validation after inference derives result metadata
      directly from operand types. Centralize only genuinely repeated count/kind projection.
- [ ] Replace shape-metadata zero/one materialization inside transforms with structural zero or `zero_like`/`one_like`
      wherever semantics allow.
- [x] Static and dynamic reshape/broadcast invocations bind the same canonical payload with exact constants or dynamic
      dimension SSA operands. Constructors intentionally use static homogeneous encoding versus the variant-owned
      dynamic stored-type contract described above.
- [ ] Add a residual search proving no concrete payload implements materially different operation type contracts.
- [x] Generic constructors have no overlapping array-program-specific payload implementation. `DynamicZero` is a
      composite variant-arm contract, while the temporary `ZeroOperation<ArrayProgramType>` is a different generic
      instantiation restricted to identity-free member types until Phase 6 deletes it.
- [x] No operation consumer independently calls an ad hoc
      `runtime_dimension_variables` contract.
- [ ] Gate: every shape dependency in rendered IR is an operand edge or an explicit `dimension_size` instruction.

### Phase 4: remove implicit-shape replay and the parallel array language

- [x] `ArrayContextView` and `DimensionContextView` no longer exist. The remaining `with_dimensions` occurrences are
      the unrelated `ReshapeParameters`/`ReshapeOperation` permutation builder and carry no ambient extent state.
- [x] `with_source_array` no longer exists.
- [x] `bind_replayed` and its operation-classification match no longer exist.
- [x] Ambient dimension and source-array context-view fields no longer exist.
- [ ] Delete temporary homogeneous program construction used only to replay shape-carrying rules.
- [ ] Narrow the homogeneous operation family to array-only primitives.
- [ ] Migrate public/reference `EagerContext<Array, ArrayOperation<Array>>` consumers to the canonical array-program
      domain where they need shape, control-flow, or transform functionality.
- [ ] Replace production tests that rely on the complete homogeneous backend with canonical array-program tests;
      retain small local homogeneous enums only for focused generic tests.
- [ ] Migrate XLA operation conversion and compilation entry points to the sole stored array-program operation family.
- [ ] Carry explicit dimension operands through condition, while, scan, custom derivatives, rematerialization, region
      capture/import, and caller/callee requirement composition.
- [ ] Make partial evaluation project known dimension integers and retain unknown dimension SSA without reconstruction;
      erase proven requirements, reject disproven requirements with exact diagnostics, and retain inconclusive ordered
      assertions.
- [ ] Verify conditional and loop-carried extents, gateway compaction, region forwarding, and alpha-equivalent imports.
- [ ] Rename the narrowed primitive family only after the old full family is deleted and all residual references are
      classified.
- [ ] Gate: targeted searches find no `with_dimensions`, `with_source_array`, `bind_replayed`, ambient replay
      environment, or full homogeneous implicit-shape graph.

### Phase 5: simplify batching around value-kind policy

P3c advances the smallest composite batching implementation needed to make the
`dimension_size -> dimension_to_scalar` vertical slice reachable. Treat the semantics it establishes as durable
(arrays may be mapped or replicated, while first-class dimensions are replicated-only), but do not assume that its
parallel `ArrayProgramBatchingContext`, `ArrayProgramBatchingTracer`, or exact `ArrayProgramBatch` representation is
the final architecture. A projected-array wrapper alone cannot represent dimension members, mixed operations,
region recursion, or a first-class dynamic batching extent, and so merely adding a public `ProjectedArrayBatch` would
rename only part of the problem while introducing another carrier.

- [ ] Before expanding the composite operation sweep, inventory the duplicated responsibilities across `ArrayBatch`,
      `BatchingContext`, `BatchingTracer`, `BatchableOperation`, and their P3c array-program counterparts. Classify
      each responsibility as value-kind-neutral, array-specific, or genuinely composite.
- [ ] Prototype a transform-owned batching-policy abstraction that can select the batch carrier and batching-extent
      representation for a parent context. The concrete shape is deliberately open, but evaluate a design equivalent
      in power to `BatchingPolicy<C> { type Batch; type AxisExtent; ... }` before committing to the parallel composite
      context/tracer tower. Keep this policy in batching machinery; do not add batching hooks to [`Type`].
- [ ] Exercise the prototype with one homogeneous array operation, one dimension operation, `dimension_size`,
      `dimension_to_scalar`, the promoted toy third member kind, and one nested-region operation. Prove that array
      members reuse ordinary `ArrayBatch` semantics, dimension members remain replicated-only, and genuinely mixed
      operations retain explicit rules.
- [ ] Reuse [`ValueProjection`] for borrowed and consuming member access. Do not add a separate public
      `ProjectedArrayBatch` unless a concrete residual need remains after the policy prototype, and reject any design
      that clones or allocates eager array payloads merely to project a batch.
- [ ] Prefer one generic batching context/tracer over the P3c parallel array-program context/tracer when the prototype
      removes more code than it adds, keeps existing array batching rules unchanged or mechanically adaptable, and
      has neutral trait-solver, compile-time, and allocation behavior. If the policy parameter spreads equivalent or
      greater ceremony through ordinary batching, retain a localized composite adapter, document the evidence, and
      reduce it to the smallest value-kind policy layer rather than forcing the abstraction.
- [ ] Generalize the operation batching contract alongside the carrier, context, and driver so that one
      `BatchableOperation` can serve homogeneous and composite batching, then delete
      `ArrayProgramBatchableOperation`. Do not generalize the operation trait in isolation: its inputs, outputs,
      active context, and recursive driver must all come from the same transform-owned batching policy. If the
      prototype gate rejects the generic policy because it adds more ceremony than it removes, retain the localized
      composite trait and record that evidence explicitly.
- [ ] Gate: the final design has one canonical representation for each necessary batching concept, no wrapper that
      merely renames `ArrayProgramBatch`, and no parallel context/tracer tower unless the rejected-policy evidence
      demonstrates that the localized duplication is the simpler implementation.
- [ ] Promote the toy composite projection fixtures from the `contexts.rs` unit tests (`ProjectedMemberType`,
      `ProjectedMemberValue`, `ProjectedProgramType`, `ProjectedProgramValue`, `ProjectedProgramOperation`, and the
      `impl_projected_test_member!` macro) into a `pub(crate)` shared test fixture the moment this phase's generic
      dispatch tests become their second consumer. Do not duplicate a synthetic composite universe; the promoted
      fixture is also the vehicle for Phase 6's dispatch tests and the verification matrix's toy third-kind gate.
- [ ] Represent a dynamic batching extent with its first-class dimension value, not metadata alone.
- [ ] Make the generic outer dispatcher project array primitives, invoke their existing homogeneous batching rule
      through the zero-state context, and lift results. Test the generic projection/lift path against the promoted
      toy composite fixture in addition to the production array-program universe, so the dispatch machinery is proven
      member-kind-agnostic rather than array-specific.
- [ ] Handle dimension-only operations with the replicated-only dimension batching policy.
- [ ] Reject mapped dimension authority at the boundary with the existing typed diagnostic.
- [ ] Keep dedicated rules only for genuinely mixed shape-changing and region-carrying operations.
- [ ] Move mapped-state RNG batching's carry-free `scan` into the composite region contract and thread every dynamic
      bits-output extent through that body as replicated first-class shape authority. Delete P3i's exact
      `"requires Phase 5 composite scan-region support"` boundary only after the resulting program remains
      size-independent in the mapped-axis extent and preserves the existing replicated-state diagnostic.
- [ ] Centralize explicit dynamic alignment/broadcasting so elementwise rules do not rediscover extents.
- [ ] Remove repeated outer-enum matches that only project/lift batches.
- [ ] Delete dimension/source-array reconstruction in dynamic slice, concatenate, reduce, collectives, RNG, and
      constructors.
- [ ] Verify nested `vmap`, mapped arrays with dynamic logical extents, replicated dimension residuals, control flow,
      and all mapped-authority rejection paths.
- [ ] Gate: adding an array-only primitive with a standard batching rule requires no handwritten change in composite
      batching dispatch.

### Phase 6: simplify differentiation and transposition

- [ ] Express the dimension tangent/cotangent space once in differentiation-owned policy.
- [ ] Introduce or extend one differentiation-owned residual structure capable of carrying ordinary array-program SSA
      values, including dimensions, without assigning them tangent/cotangent slots.
- [ ] Make linearization rules declare required primal dimension residuals explicitly while those operands or source
      arrays are available.
- [ ] Thread dimension residuals through nested regions, rematerialization, custom derivatives, JVP/VJP construction,
      import, and transpose exactly like other residual values.
- [ ] Rewrite reshape transposition to consume ordinary dimension residual inputs.
- [x] `ReshapeOperation::transpose_dimension_variables` and every exact identifier occurrence are absent from the
      integration tree. Do not reintroduce an equivalent payload residual manifest while implementing the ordinary
      residual path above.
- [ ] Audit concatenate, mean/reductions, slice, pad, and gather transposes and migrate every analogous extent need to
      the same residual contract.
- [ ] Make generic outer dispatch project/lift array-only JVP, VJP, and transpose rules, reusing the shared toy
      composite fixture promoted in Phase 5 for the member-kind-agnostic dispatch tests.
- [ ] Preserve dimension values as ordinary structural residuals without tangent slots.
- [ ] Keep explicit mixed rules only where primal dimension operands control array results or region interfaces.
- [ ] Remove temporary homogeneous differentiation programs and dimension recovery.
- [ ] Add a residual search proving no primal operation payload stores differentiation-only dimension variables or
      residual manifests.
- [ ] Prefer structural zeros over materializing shaped zero arrays.
- [ ] Inventory every production construction of `ZeroOperation<ArrayProgramType>` and every composite
      `Zero<ArrayProgramValue<_>>` materialization path, including the retained-linearization residual-zero sites in
      `differentiation/forward.rs`. Classify each as a structural zero that should remain unmaterialized, an
      operand-relative `zero_like`, an identity-free homogeneous array zero, or a genuinely dynamic zero whose
      explicit dimension SSA operands must already be available.
- [ ] Migrate those callers so generic differentiation and transposition never request a zero from
      `ArrayProgramType` alone. Preserve [`MaybeZero`] structurally for as long as possible; use `zero_like` when an
      array value supplies runtime geometry; stage `Array(ArrayOperation::Zero(ZeroOperation<ArrayType>))` only for
      identity-free array types; and stage the mixed constructor with explicit dimension operands when a dynamic array
      zero truly must be materialized. A dimension-member zero must be unrepresentable rather than constructed and
      rejected later by inference.
- [ ] Delete `ArrayProgramOperation::Zero(ZeroOperation<ArrayProgramType>)`,
      `From<ZeroOperation<ArrayProgramType>>`, and every dedicated inference, eager, batching, differentiation,
      rendering, identity-renaming, and lowering arm once the generic callers are gone. Do not replace them with
      another type-only composite constructor or a hidden extent-recovery path.
- [ ] Rename `ArrayProgramOperation::DynamicZero(ZeroOperation<ArrayType>)` to
      `ArrayProgramOperation::Zero(ZeroOperation<ArrayType>)` after deleting the conflicting generic variant. Expand
      its rustdoc to explain that this top-level variant owns the mixed `(Dimension...) -> Array` signature because
      homogeneous `Array(...)` projection cannot accept dimension operands and structural types do not contain
      concrete runtime extents. Update all tests, diagnostics, rendering expectations, and documentation directly
      without a compatibility alias.
- [ ] Add canonical-representation tests proving identity-free zeros use
      `Array(ArrayOperation::Zero(ZeroOperation<ArrayType>))`, identity-bearing zeros use the renamed top-level
      `Zero(ZeroOperation<ArrayType>)` with one explicit operand per dynamic axis, and no operation can encode a
      dimension-member zero. Add residual searches requiring zero source occurrences of
      `ZeroOperation<ArrayProgramType>` and `DynamicZero`.
- [ ] Preserve proven/disproven/residual requirement behavior and `OrderedAssertion` effects.
- [ ] Verify nested JVP/VJP, linearization, transpose, rematerialization, custom derivatives, condition, while, and
      scan.
- [ ] Add exact rendered-IR tests proving residual dimension atoms are explicit dataflow edges shared by the forward
      linearization and transpose, with no type expression or payload witness.
- [ ] Gate: adding an array-only primitive with ordinary AD/PE rules requires no handwritten composite dispatcher
      case, the generic composite zero escape hatch is gone, and the only top-level zero variant is the explicit mixed
      constructor.

### Phase 7: backend execution and lowering

- [ ] Verify every mixed operation lowers explicit dimension operands directly with no reconstruction environment.
- [ ] Verify eager XLA dimension arithmetic remains host integer computation with zero device dispatch/cache probes.
- [ ] Verify bounded-input ABI argument counts and `set_dimension_size` behavior are unchanged.
- [ ] Lower every residual `DimensionRequirementOperation` to a runtime assertion that observes the concrete operand
      values and preserves its exact actor name, predicate, bounds/divisor, and observed-value diagnostic.
- [ ] Replace each lowering scope's single shared `Option<ValueRef>` token with one deterministic token slot per
      ordered `Effect` class. Assertions advance only `OrderedAssertion`; prints advance only `OrderedIo`; unordered
      I/O does not acquire an ordered chain merely because another effect exists.
- [ ] Thread the active ordered-effect token set through condition results, while/scan state, rematerialization,
      custom derivatives, and effectful inlined calls. Pure regions add no token state, and a region containing only
      one ordered class carries only that class.
- [ ] Add structural MLIR tests for assertion→assertion, print→print, assertion interleaved with print, pure/effectful
      branches, loops, scan, rematerialization, and repeated inlined calls. Add CPU execution tests proving
      deterministic first-failure order within the assertion class and independence from ordered I/O.
- [ ] Verify ordered runtime assertions preserve exact actor-named diagnostics and deterministic same-class order.
- [ ] Run CPU and CUDA eager/JIT parity for the full dynamic operation matrix, including `PadToStatic`.
- [ ] Gate: backend behavior, diagnostics, and bounded physical storage match or exceed the archived golden evidence.

### Phase 8: enforce contracts and consolidate operation declarations

- [ ] Begin only after Phases 1 through 7 have removed dual semantic contracts, implicit replay, and overlapping mixed
      constructors. Capture the resulting implementor and bound inventory before changing the trait.
- [ ] Prototype `Operation` with an associated `Type` on a bounded vertical slice:
      `AddOperation`, `ZeroOperation<T>`, `ArrayPrimitiveOperation`, `DimensionArithmeticOperation`,
      `DimensionSizeOperation`, one mixed stored-type constructor contract, `ReshapeOperation`, and
      `ArrayProgramOperation`.
- [ ] Parameterize `SelectOperation`, `StopGradientOperation`, and `TestNullaryOperation` by their operation type so
      each concrete payload instantiation has exactly one associated contract.
- [ ] Update the derive macro in the prototype so homogeneous enums prove that every payload has the same operation
      type.
- [ ] Exercise the prototype through inference, eager interpretation, tracing, PE, batching, JVP, VJP, transposition,
      rendering, region import, and XLA lowering.
- [ ] Add compile-fail tests proving one payload cannot acquire two semantic type contracts and a homogeneous enum
      cannot combine mismatched payload types.
- [ ] Measure clean/incremental compile time, peak memory, macro output size, and trait-solver stability against
      Phase 0.
- [ ] Produce a mechanical migration count for all crates, not only `ryft-core`.
- [ ] Gate: adopt the associated-type trait only if it enforces the already-established canonical signatures with no
      trait-solver regression, no wrapper layer beyond the three approved localized type parameters, and a neutral or
      smaller final generic surface.
- [ ] Fallback gate: if rejected, implement the smallest sealed one-contract marker that prohibits dual semantics and
      document why the associated type failed. Do not leave the invariant conventional.

- [ ] Establish one authoritative declaration of every array-program operation and its class.
- [ ] Generate the outer variants, inner lifts, `From` conversions, and mechanical dispatch from that declaration.
- [ ] Make each mixed operation's inference contract the authoritative source for dimension operand positions,
      member kinds, ordering, and result metadata.
- [ ] Extend the typed mixed projection vocabulary only for repeated fixed/optional/segmented patterns found in the
      Phase 0 inventory.
- [ ] Centralize only structural projection needed to ensure eager interpretation, transforms, and lowering preserve
      the operand order declared by inference.
- [ ] Delete redundant local variant lists, conversion macros, manual wrong-kind matches, and projection boilerplate.
- [ ] Delete independent `runtime_dimension_variables` methods after their operations consume explicit operands.
- [ ] Keep semantically meaningful operation rules handwritten and colocated with their payload.
- [ ] Add compile-fail coverage for invalid generated operation declarations and runtime goldens for canonical
      projection diagnostics.
- [ ] Run macro unit and integration tests and compare generated token counts/compile time with the baseline.
- [ ] Gate: one new array-only primitive requires one family declaration and its semantic/backend rules; one new mixed
      operation declares its signature once and does not add projection ceremony to transforms.

### Phase 9: module and public API cleanup

- [ ] Confirm the `S4` typed `Custom`/`DimensionError` recovery behavior and canonical invalid projection diagnostics
      remain intact;
      do not mix another error-representation migration into the module move.
- [x] Core dimension operation semantics are split from the eager host representation.
- [x] Dimension operation semantics live in `operations::dimensions`.
- [x] `DimensionValue`, its closed eager operation family, and concrete capability implementations remain under backend
      ownership.
- [ ] Re-evaluate the historical `RuntimeDimension`/`RuntimeShape` item: neither identifier exists in the current
      tree. Confirm that the public first-class dimension capabilities cover the intended ergonomics; do not recreate
      wrapper types merely to satisfy the old module-move wording. If a neutral public alias/API is still needed, add
      only the smallest capability-based surface after the operation and transform families settle.
- [ ] Audit names after responsibilities settle; rename only where the final name is materially clearer.
- [ ] Update every in-repo use site directly without compatibility re-exports.
- [ ] Update rustdoc, examples, error links, and behavioral JAX fixtures.
- [ ] Run targeted searches for every old canonical path and classify all remaining matches.
- [ ] Gate: core language semantics no longer appear to be backend implementation details.

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
- [ ] Dimension tangents/cotangents remain absent or structural zero.
- [ ] Required primal dimension values travel as ordinary differentiation residual SSA values.
- [ ] No primal operation payload contains `transpose_dimension_variables` or an equivalent residual manifest.
- [ ] Dynamic batching alignment consumes an explicit dimension value.
- [ ] Requirement effects survive every transform and lower in deterministic order.
- [ ] Nested condition, while, scan, custom derivative, and rematerialization regions carry dimensions correctly.
- [ ] Repeated boundary readers do not become duplicate producers.
- [ ] Fresh internal dimensions have one producer and dominate every reference.
- [ ] Alpha-equivalent programs share cache identity; live permutations and different graphs do not.
- [ ] Exact diagnostics match the baseline.
- [ ] Bounded dynamic ABI, CPU, and CUDA behavior match the baseline.
- [ ] Behavioral JAX parity and Ryft-exceeds-JAX cases remain intact.
- [ ] Toy third-kind tests demonstrate that generic program/context/projection machinery is closed to modification.

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
- any backend path restores shape expression evaluation, host readback, or reconstruction; or
- the toy third-kind test still requires edits throughout generic program and transform machinery.

## Exit criteria

The cleanup is complete only when:

1. Runtime dimensions remain ordinary SSA values in one program graph.
2. Each operation payload has one compiler-enforced semantic contract.
3. `dimension_size` always returns a dimension and the data gateway is distinct.
4. Dynamic zero, one, fill, and iota construction binds the constructors' own mixed stored-type contracts;
   transform-generated
   values use structural or operand-relative construction where possible.
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
remains a production transform/backend language. Phase 5's separate `ArrayProgramBatchingContext`,
`ArrayProgramBatchingTracer`, and `ArrayProgramBatchableOperation` also remain, as do the composite differentiation
dispatcher and the five transitional dual-contract payloads listed in the diagnosis. `Operation` is still generic as
`Operation<T>`; Phases 8–11 have not begun.

The current unstaged P3j prototype was reviewed path by path:

- **Retain after correction:** the position-aware nullary constructor guard; the shared variant-owned dynamic
  constructor inference; canonical `ZeroOperation<ArrayType>` routing; the flat `DynamicZero` variant; eager
  materialization from compact dynamic-axis operands; replicated-only batching; structural-zero extent cotangents;
  direct upper-bound allocation plus `stablehlo.set_dimension_size`; the mixed static/dynamic axis pairing fix; and
  their focused tests.
- **Delete rather than generalize:** `DimensionType::known_extent`, `with_known_extent`, observed extents injected by
  `ArrayProgramValue::r#type`, `ArrayProgramTypeRefinements`' dependence on those type payloads, the extra
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
- `ArrayProgramValue::r#type` no longer incorporates runtime observations;
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

### Plan revision: constructor contracts without a wrapper

The shaped-constructor wrapper (P3j Delivery A) was replaced after review by the stored-type-authoritative design:
constructor payloads keep their possibly-dynamic output `ArrayType`, and dynamic axes consume explicit
identity-validated dimension operands. This deleted `ShapedConstructorOperation`, `ArrayConstructorOperation`, and
the template-shape representation entirely, and resolved the template-shape canonicalization question by making it
unrepresentable. Jacobian forward and reverse entry points now validate input and output coordinate spaces before
linearization and pullback, preserving their precise `NonFiniteCoordinateSpace` diagnostics ahead of the new
constructor rule.

A second review pass corrected the first implementation of this design:

- the mixed contract moved from a second `impl Operation<ArrayProgramType>` on `ZeroOperation<ArrayType>` to the
  `ArrayProgramOperation::DynamicZero` variant arm, restoring one trait implementation per payload and Phase 8
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
- generic zero/one/fill/iota overlap is part of the dual-contract inventory, with operand-relative construction for
  transforms, homogeneous construction for static geometry, and one variant-owned mixed contract for dynamic geometry;
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

- `ArrayProgramType` and `ArrayProgramValue<A>` are the sole array/dimension storage sums;
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

Production `ArrayProgramOperation`, mixed operand contracts, shape operations, higher-order region projection, batching,
and differentiation policies remain assigned to P3–P6. The existing reference-array allocation tests continue to prove
that borrowed and consuming eager value projection neither allocates nor copies payloads. Projected-context binding
temporarily reconstructs parent values from borrowed inputs because the generic `Context::bind` contract accepts a
slice; concrete eager member values continue to dispatch through their native contexts, while symbolic projected
values clone only their parent tracer/transform representation and preserve SSA identity. P10 retains the final
cross-context allocation and latency measurement once production outer dispatch is present.

Verification and residual-audit results are recorded in the P2d cleanup-ledger entry at handoff.

### Execution: P3a production array-program dispatcher

P3a introduced `ArrayProgramOperation<A>` as the sole stored dispatcher for heterogeneous array/dimension programs.
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

The old expression/homogeneous implementations remain deliberately isolated for the immediately following consumer
migration and deletion increment. Their exact symbol counts, owning files, `From` bounds, call-site counts, deletion
order, and required acceptance tests are recorded in `.tasks/plan_p3g_reshape_broadcast.md`; the Phase 3 homogeneous
deletion items remain open until that residual search reaches zero.

### Execution: P3h Delivery A explicit concatenate result extent

P3h Delivery A extended the existing axis-only `ConcatenateOperation` with
`Operation<ArrayProgramType>` rather than creating a redundant legacy payload. Its canonical mixed signature is
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
as a flat `ArrayProgramOperation::ShapedZero` variant. The operation consumes one explicit first-class dimension
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
residuals, so deletion of the temporary `Zero(ZeroOperation<ArrayProgramType>)` escape hatch remains P3j Delivery D
and may move to the first Phase 6 residual delivery rather than adding type-to-value recovery. Exact inventory,
measurements, verification, and residual evidence are recorded in `.tasks/plan_p3j_shaped_constructors.md`.
