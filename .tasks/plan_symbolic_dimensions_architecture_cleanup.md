# First-class dimension architecture cleanup

## Status

Approved for staged execution; bootstrap has not started. This plan is a containment and simplification follow-up to
`.tasks/plan_first_class_dimension_programs.md`. It preserves that plan's user-visible capabilities and its decision to
represent runtime dimensions as ordinary SSA values, but supersedes the following implementation compromises:

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

Before executing any phase, read **Execution staging and review process** below. That section defines the branch
topology, the increment catalog, the per-increment workflow, the delivery ledger, and the resumption protocol. No
work may land outside that process.

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

The current operation trait permits the same payload to implement multiple type-family contracts. The following
payloads currently implement both `Operation<ArrayType>` and `Operation<ArrayProgramType>`:

- `BroadcastOperation`;
- `ConcatenateOperation`;
- `CustomCallOperation`;
- `DimensionSizeOperation`;
- `DynamicSliceOperation`;
- `GatherOperation`;
- `PadOperation`;
- `ReduceOperation`;
- `ReshapeOperation`;
- `RngBitGeneratorOperation`;
- `SliceOperation`; and
- `SliceScatterOperation`.

The inventory must also include overlapping generic constructor contracts, which are a distinct and harder case:

- `ZeroOperation<ArrayType>`;
- `OneOperation<ArrayType>`;
- `FillOperation<ArrayType, V>`; and
- `IotaOperation<ArrayType>`.

Each constructor has a generic homogeneous `Operation<T>` implementation and an array-program-specific implementation
that adds explicit dynamic-dimension operands. These cannot be resolved by merely selecting one of two handwritten
implementations while retaining the blanket generic contract. The canonical destination is:

- operand-relative `zero_like`/`one_like` for transform-generated values whenever a source array exists;
- a homogeneous static constructor only when its output type has no dynamic axes; and
- one explicitly mixed shaped-constructor wrapper for zero, one, fill, and iota when dynamic extents are operands.

The wrapper is a distinct operation contract around the homogeneous constructor payload; the payload itself does not
implement a second type-family contract. Static and dynamic construction must not be distinguished by ambient operand
recovery.

`DimensionSizeOperation` demonstrates why this is unsafe: its homogeneous contract returns a rank-zero integer array,
while its heterogeneous contract returns a first-class `DimensionType`. An operation's result kind must not depend on
the surrounding trait instantiation.

`ReshapeOperation::transpose_dimension_variables` is a second concrete containment failure. The corresponding values
are explicit dimension SSA operands, so it is not an expression-evaluation witness, but the payload field is a
differentiation-only residual manifest understood only by reshape and composite differentiation. Other transpose rules
will need the same primal-extent retention. Residual selection and threading belong to the differentiation transform,
not to individual primal operation payloads.

The current working-tree baseline is provisional because the parent refactor is still uncommitted, but Phase 0 must
capture at least:

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
- Shape operations consume every independently computed dynamic extent as an explicit dimension operand.
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

- One concrete operation payload has one operand/result/region contract.
- `dimension_size` means `array -> dimension` in every context.
- `dimension_to_scalar` is the explicit `dimension -> scalar-array` gateway.
- A shape-carrying operation is mixed even when a particular invocation happens to have only static dimensions.
- A static convenience API may omit dimension operands only when the payload's metadata proves there are none; it
  still binds the same mixed operation contract.

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
- Assertion effect ordering, DCE survival, partial-evaluation behavior, and backend lowering remain unchanged.

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

Replace the bundled `ArrayProgramProjection` contract with reusable projection contracts parameterized by the member
type. Projection must distinguish borrowing from ownership transfer:

```rust,ignore
pub trait TypeProjection<T: Type>: Type {
    fn project(&self) -> Result<&T, TypeError>;
    fn lift(r#type: T) -> Self;
}

pub trait ValueProjection<T: Type>: Value
where
    Self::Type: TypeProjection<T>,
{
    type Projected: Value<Type = T>;
    type ProjectedRef<'a>: Typed<Type = T>
    where
        Self: 'a;

    fn project_ref(&self) -> Result<Self::ProjectedRef<'_>, ProgramError>;
    fn into_projected(self) -> Result<Self::Projected, ProgramError>;
    fn lift(value: Self::Projected) -> Self;
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

### Central dimension-operand schemas

The order and identity of explicit dimension operands are part of a mixed operation's semantic signature. Replace the
current family of ad hoc `runtime_dimension_variables` methods and copied validation loops with one mixed-operation
schema contract. Conceptually, it yields ordered segments such as:

```rust,ignore
DimensionOperandSchema {
    segments: [
        operation_derived("starts", start_dimensions),
        shape("sizes", size_dimensions),
    ],
}
```

The final schema representation must:

- identify the array-program operand position of every dimension value;
- name fixed, optional, repeated, and operation-derived segments;
- carry the expected `DimensionVariable` identity and bounds;
- validate count, kind, identity, bounds, and deterministic ordering once;
- expose borrowed typed slices/views to inference, eager interpretation, transforms, and lowering;
- support schemas whose expected variables depend on already-projected array input types, such as pad or reduce; and
- produce canonical operation/segment/index diagnostics.

Keep this contract on the mixed operation/schema layer, not on the generic `Operation` trait. Scalar and unrelated
operation families must not acquire dimension-specific methods. A local `runtime_dimension_variables` helper may
remain only as the implementation of the centralized schema builder; operation consumers may not call independent
collectors or repeat validation loops.

### Operation families

The final production families are:

```text
ArrayPrimitiveOperation<A>       Operation<Type = ArrayType>
DimensionOperation              Operation<Type = DimensionType>
MixedArrayDimensionOperation<A> Operation<Type = ArrayProgramType>
ArrayProgramOperation<A>        Operation<Type = ArrayProgramType>
```

The names may be simplified after the migration, but the roles are fixed.

- `ArrayPrimitiveOperation` contains operations whose complete signature is array-only.
- `DimensionOperation` contains dimension-only operations.
- `MixedArrayDimensionOperation` contains every operation with cross-kind operands/results and every higher-order
  operation whose regions may carry both kinds.
- `ArrayProgramOperation` is the sole stored dispatcher and the sole operation family of public array-program
  execution contexts.

Generic nullary constructors remain homogeneous only for types whose complete output geometry is metadata-only. Array
construction with dynamic axes uses one mixed shaped-constructor wrapper that owns the explicit dimension-operand
schema and delegates the element-generation semantics to `ZeroOperation<ArrayType>`, `OneOperation<ArrayType>`,
`FillOperation<ArrayType, V>`, or `IotaOperation<ArrayType>`. This avoids overlapping a blanket `Operation<T>`
implementation with an `ArrayProgramType` implementation for the same instantiated payload.

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

Prefer a small declarative schema macro or typed cursor extension over a new general procedural-macro language.
Generate code only when it deletes more handwritten code than it introduces. The operation declaration is the sole
source for outer enum variants, `From` conversions, type projection/lifting, and transform dispatch classification.

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
- a neutral public dimension API module: `RuntimeDimension` and `RuntimeShape`; and
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
- existing StableHLO dynamic-shape operands and assertion lowering;
- behavioral JAX parity tests;
- exact diagnostics;
- identity alpha-renaming, canonical signatures, and cache behavior; and
- public `RuntimeDimension`/`RuntimeShape` ergonomics, after decoupling them from backend-specific projection names.

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
- duplicated transform rules whose only purpose is storage-kind projection/lifting;
- direct `TypeError::Invalid { message: ... }` construction after the canonical `TypeError::invalid(...)` migration;
  named-field destructuring remains supported; and
- compatibility shims for the retired homogeneous shape-program API.

One centralized schema implementation may internally collect dynamic variables from `Shape` metadata. What must
disappear is independent per-consumer collection, copied validation, and any use of a schema to recover operands absent
from the staged graph.

## Execution staging and review process

### Why this section exists

The parent refactor exists as one uncommitted working tree containing 112 tracked changes and 29 nonignored untracked
paths, for 141 expanded status entries, plus this ignored plan. It spans unrelated refactors and several generations
of the dimension design. It cannot be reviewed, bisected, or safely resumed in that form, and it has no recovery point.
This section first preserves that tree exactly, then mines only the correct pieces into the target architecture as
small reviewed increments.

Read this section completely before running any command. It is written to be executed literally.

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
| `P1`   | Leaf identity, bounds, refinements, structural closure, canonical identity | Line by line              |
| `P2`   | Dimension SSA/value operations, storage sum, and projection vertical slice | Line by line              |
| `P3.*` | Central schemas, constructors, and one increment per mixed shape operation  | Line by line              |
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
`P11`. The increment catalog controls any further split within a phase.

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
      the homogeneous constructor, operand-relative operation, or mixed shaped-constructor wrapper.
- [x] Inventory every use of the complete homogeneous `ArrayOperation` outside tests and assign its migration target.
- [x] Inventory every `ArrayContextView`/`DimensionContextView` construction and state why the caller needs a view.
- [x] For every `with_dimensions`, `with_source_array`, and `bind_replayed` path, record the explicit SSA dependency
      that must replace it.
- [x] Inventory every batching, differentiation, transposition, and partial-evaluation special case in
      `backends/array_programs`.
- [x] Inventory every transpose that needs primal dimension values unavailable from the cotangent and record the
      explicit SSA residuals it requires. Include reshape, concatenate, mean/reductions, slice, pad, and gather.
- [x] Inventory every independent `runtime_dimension_variables` collector and every copied dimension-operand
      validation loop; design its centralized schema segment before the sweep.
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
  instantiation, renaming, and canonical identity, then deletes `OutputIdentityRole`.

- [x] `P1a`: add validated inclusive-lower/exclusive-upper bounds with exact diagnostics.
- [x] `P1a`: add fresh identity semantics whose clone preserves identity and whose name is diagnostic-only.
- [x] `P1a`: keep bounds owned only by the identity and immutable after construction.
- [x] `P1a` gate: focused tests cover bounds, display, clone/fresh equality, hashing, and typed error recovery without
      changing existing `Dimension` behavior.

- [ ] Replace the mechanically renamed `Dimension::Dynamic(Option<usize>)` with the reviewed leaf-only dynamic form:
      one `DimensionVariable` identity plus authoritative bounds, with no arithmetic expression in types.
- [ ] Establish one source of truth for bounds. Types carry the authoritative bounds used for checking and compilation;
      public declaration helpers construct those types rather than maintaining an independently mutable copy.
- [ ] Add only the generic `Type::Identity` and `Type::Refinements` hooks needed for program closure, alpha-renaming,
      instantiation, and canonical signatures. Do not put batching, differentiation, or dimension-specific behavior on
      `Type`.
- [ ] Capture exact current identity closure, dominance, forwarding, import, and cache diagnostics.
- [ ] Implement the structural producer/forwarder algorithm behind a focused internal prototype.
- [ ] Cover repeated `dimension_size` readers, fresh dimension arithmetic results, shared outputs, condition forwarding,
      while carries, scan stacked outputs, captures, shared regions, and alpha-equivalent imports.
- [ ] Delete `OutputIdentityRole` from the operation trait, derive macro, box delegation, builders, and operation
      payloads if all valid cases pass.
- [ ] Ensure definition/reference positions remain owned by type families and boundary/internal classification remains
      owned by region closure.
- [ ] Replace repeated linear membership scans only if profiling shows closure cost is material; do not require `Hash`
      on identities without a real consumer.
- [ ] Remove avoidable temporary array/dimension refinement vectors with one-pass validation where it reduces
      allocations without obscuring diagnostics.
- [ ] Gate: cache identity, permutation behavior, and exact diagnostics match or exceed the baseline with no
      operation-specific identity-source hook.

### Phase 2: introduce generic member projection and direct binding

- [ ] Introduce ordinary `DimensionType`/`DimensionValue` scalar SSA and the minimal dimension operation family for
      constants, arithmetic, comparisons, gateways, `dimension_size`, and requirements.
- [ ] Introduce the array/dimension storage sum only at atom/region interfaces and genuinely mixed operations.
- [ ] Integrate inconclusive requirements with the existing effects model as `Effect::OrderedAssertion`; specify
      ordering, DCE survival, known-side PE folding, runtime observation values, and diagnostic ownership before
      lowering.
- [ ] Complete the Phase 0 projection-ownership decision before writing the generic projection trait.
- [ ] Prototype `TypeProjection<T>` and `ValueProjection<T>` for `ArrayType` and `DimensionType` members of the storage
      sum.
- [ ] Provide distinct borrowed projection and consuming ownership-transfer paths; do not implement eager projection
      as `.cloned()` from a borrowed storage-sum value.
- [ ] Implement eager, capture, tracer, partial-tracer, batching-tracer, and differentiation-tracer projections.
- [ ] Replace duplicated projected-value wrappers with one generic checked projected value where the concrete eager
      value cannot be returned directly.
- [ ] Introduce a zero-state `ProjectedContext<C, T>` that binds homogeneous inner operations directly into the outer
      graph.
- [ ] Add one generic inner-operation lift contract implemented by the outer operation family.
- [ ] Preserve SSA atom identity exactly through staged projections.
- [ ] Preserve concrete eager values without boxing or heap allocation.
- [ ] Add allocation and payload-size tests proving that projecting a large reference-backend array neither allocates
      nor copies its `Scalar` payload.
- [ ] Pin canonical wrong-kind and wrong-count diagnostics as compile/runtime goldens.
- [ ] Add a compile-only toy third member kind to prove that another kind needs projection and policy
      implementations, not changes to generic `Program`, `Context`, capture, tracer, or projected-context machinery.
- [ ] Gate: the projected context contains no semantic state other than its parent, and the vertical slice creates no
      implicit dimension dependency.

### Phase 3: establish canonical mixed operation signatures

- [ ] Make `DimensionSizeOperation` exclusively `array -> dimension`.
- [ ] Keep `DimensionToScalarOperation` as the only `dimension -> scalar-array` conversion.
- [ ] Migrate reshape and broadcast first as the canonical mixed vertical slice.
- [ ] Delete their homogeneous operation implementations and every transform rule that depends on those contracts.
- [ ] Migrate the remaining dual-contract operations:
      concatenate, custom call, dynamic slice, gather, pad, reduce, RNG bit generation, slice, and slice scatter.
- [ ] Remove the array-program-specific `Operation<ArrayProgramType>` implementations from
      `ZeroOperation<ArrayType>`, `OneOperation<ArrayType>`, `FillOperation<ArrayType, V>`, and
      `IotaOperation<ArrayType>`.
- [ ] Route transform-generated zero/one values through structural zero or `zero_like`/`one_like` whenever an operand
      supplies geometry.
- [ ] Keep homogeneous nullary zero/one/fill/iota only for fully static array output types.
- [ ] Route dynamic zero/one/fill/iota through one mixed shaped-constructor wrapper whose only additional operands are
      the explicit dynamic extents required by its centralized schema.
- [ ] Sweep shape-changing collectives and every other operation whose result metadata references first-class
      dimension operands.
- [ ] Implement the centralized dimension-operand schema and migrate each mixed operation's inference, eager rule,
      transforms, and lowering to its typed views.
- [ ] Delete copied dimension operand identity, bounds, count, and ordering validation after each operation migrates.
- [ ] Replace shape-metadata zero/one materialization inside transforms with structural zero or `zero_like`/`one_like`
      wherever semantics allow.
- [ ] Ensure static invocations of canonical mixed shape operations bind the same payload with an empty
      explicit-dimension segment rather than a second contract. Constructors follow the explicit static-homogeneous
      versus dynamic-shaped-wrapper split above.
- [ ] Add a residual search proving no concrete payload implements materially different operation type contracts.
- [ ] Add a residual search proving generic constructors have no overlapping array-program-specific implementation.
- [ ] Add a residual search proving no operation consumer independently calls an ad hoc
      `runtime_dimension_variables` contract.
- [ ] Gate: every shape dependency in rendered IR is an operand edge or an explicit `dimension_size` instruction.

### Phase 4: remove implicit-shape replay and the parallel array language

- [ ] Rewrite each `ArrayContextView::with_dimensions` call to pass the required dimension values through the actual
      operation/transform input.
- [ ] Rewrite each `with_source_array` call to consume an existing explicit dimension value or stage an explicit
      `dimension_size` at the semantic point that needs it.
- [ ] Delete the operation-classification match in `bind_replayed`.
- [ ] Delete ambient dimension and source-array fields.
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

- [ ] Represent a dynamic batching extent with its first-class dimension value, not metadata alone.
- [ ] Make the generic outer dispatcher project array primitives, invoke their existing homogeneous batching rule
      through the zero-state context, and lift results.
- [ ] Handle dimension-only operations with the replicated-only dimension batching policy.
- [ ] Reject mapped dimension authority at the boundary with the existing typed diagnostic.
- [ ] Keep dedicated rules only for genuinely mixed shape-changing and region-carrying operations.
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
- [ ] Delete `ReshapeOperation::transpose_dimension_variables`, its builder/accessor, rendering, identity renaming,
      mixed inference segment, eager input handling, and composite differentiation special case.
- [ ] Audit concatenate, mean/reductions, slice, pad, and gather transposes and migrate every analogous extent need to
      the same residual contract.
- [ ] Make generic outer dispatch project/lift array-only JVP, VJP, and transpose rules.
- [ ] Preserve dimension values as ordinary structural residuals without tangent slots.
- [ ] Keep explicit mixed rules only where primal dimension operands control array results or region interfaces.
- [ ] Remove temporary homogeneous differentiation programs and dimension recovery.
- [ ] Add a residual search proving no primal operation payload stores differentiation-only dimension variables or
      residual manifests.
- [ ] Prefer structural zeros over materializing shaped zero arrays.
- [ ] Preserve proven/disproven/residual requirement behavior and `OrderedAssertion` effects.
- [ ] Verify nested JVP/VJP, linearization, transpose, rematerialization, custom derivatives, condition, while, and
      scan.
- [ ] Add exact rendered-IR tests proving residual dimension atoms are explicit dataflow edges shared by the forward
      linearization and transpose, with no type expression or payload witness.
- [ ] Gate: adding an array-only primitive with ordinary AD/PE rules requires no handwritten composite dispatcher case.

### Phase 7: backend execution and lowering

- [ ] Verify every mixed operation lowers explicit dimension operands directly with no reconstruction environment.
- [ ] Verify eager XLA dimension arithmetic remains host integer computation with zero device dispatch/cache probes.
- [ ] Verify bounded-input ABI argument counts and `set_dimension_size` behavior are unchanged.
- [ ] Verify ordered runtime assertions preserve exact actor-named diagnostics and deterministic order.
- [ ] Run CPU and CUDA eager/JIT parity for the full dynamic operation matrix, including `PadToStatic`.
- [ ] Gate: backend behavior, diagnostics, and bounded physical storage match or exceed the archived golden evidence.

### Phase 8: enforce contracts and consolidate operation declarations

- [ ] Begin only after Phases 1 through 7 have removed dual semantic contracts, implicit replay, and overlapping mixed
      constructors. Capture the resulting implementor and bound inventory before changing the trait.
- [ ] Prototype `Operation` with an associated `Type` on a bounded vertical slice:
      `AddOperation`, `ZeroOperation<T>`, `ArrayPrimitiveOperation`, `DimensionAddOperation`,
      `DimensionSizeOperation`, one mixed shaped-constructor wrapper, `ReshapeOperation`, and
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
- [ ] Make the centralized dimension-operand schema the authoritative source for dimension operand positions,
      segments, variables, bounds, and diagnostics.
- [ ] Extend the typed mixed projection vocabulary only for repeated fixed/optional/segmented patterns found in the
      Phase 0 inventory.
- [ ] Generate or centralize schema validation so inference, eager interpretation, transforms, and lowering cannot
      disagree about dimension operand order.
- [ ] Delete redundant local variant lists, conversion macros, manual wrong-kind matches, and projection boilerplate.
- [ ] Delete independent `runtime_dimension_variables` methods after their operation schemas migrate.
- [ ] Keep semantically meaningful operation rules handwritten and colocated with their payload.
- [ ] Add compile-fail coverage for malformed schemas and runtime goldens for canonical projection diagnostics.
- [ ] Run macro unit and integration tests and compare generated token counts/compile time with the baseline.
- [ ] Gate: one new array-only primitive requires one family declaration and its semantic/backend rules; one new mixed
      operation declares its signature once and does not add projection ceremony to transforms.

### Phase 9: module and public API cleanup

- [ ] Confirm the `S4` typed `Custom`/`DimensionError` recovery behavior and canonical invalid projection diagnostics
      remain intact;
      do not mix another error-representation migration into the module move.
- [ ] Split core dimension operations from the eager host representation currently colocated under `backends`.
- [ ] Move operation semantics to `operations::dimensions`.
- [ ] Keep concrete eager values and backend-specific behavior under backend ownership.
- [ ] Move `RuntimeDimension`/`RuntimeShape` to a neutral public module and replace backend-specific trait bounds with
      generic projection/binding capabilities.
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
- [ ] Reconcile every path on `u/eaplatanios/wip/dimensions-remainder` as landed, superseded, or deliberately dropped.
      Require an empty unexplained remainder and a complete archive-disposition ledger; do not alter the immutable
      archive to make this check pass.
- [ ] Verify `origin/u/eaplatanios/archive/dimensions-wip-2026-07-24` still points to the recorded bootstrap commit.
- [ ] Land a final bookkeeping increment that closes the last substantive ledger entry with its integration and
      remainder reconciliation commits.
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
- [ ] One centralized schema defines every explicit dimension operand's segment, position, identity, bounds, and
      diagnostic.
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
- mixed signature generation becomes a second general operation DSL with more code than the handwritten schemas;
- structural identity ownership cannot represent a valid operation without weakening closure soundness;
- differentiation requires a new operation-specific dimension residual field after the generic residual migration;
- inference, transforms, eager interpretation, and lowering derive different dimension operand orderings instead of
  consuming one schema;
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
4. Dynamic zero, one, fill, and iota construction has one mixed shaped-constructor contract; transform-generated
   values use structural or operand-relative construction where possible.
5. Every shape-carrying operation consumes explicit dimension operands described by one centralized schema.
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

### Plan revision: projection ownership, constructors, residuals, and schemas

The pre-execution review identified four missing design decisions and this revision resolves them:

- projection now has distinct borrowed and consuming paths, and Phase 0 must decide whether the immutable reference
  `Array` payload also moves from `Vec<Scalar>` to measured shared storage before the prototype;
- generic zero/one/fill/iota overlap is part of the dual-contract inventory, with operand-relative construction for
  transforms, homogeneous construction for static geometry, and one mixed wrapper for dynamic geometry;
- leaf-only dimensions remain explicit policy, while transpose-only primal extents move through one
  differentiation-owned ordinary SSA residual mechanism and `transpose_dimension_variables` is deleted; and
- one mixed-operation dimension-operand schema owns positions, segments, identities, bounds, ordering, validation,
  diagnostics, and consumer views across inference, eager execution, transforms, and lowering.

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
