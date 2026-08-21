# Ryft References: Architecture and Implementation Plan

**Status:** Phases 0 through 13 implemented and verified. The reference architecture and implementation plan is
complete; a production Pallas-style kernel language remains a separate future program.

**Research snapshot:** 2026-08-14

**Scope:** first-class mutable array references in `ryft-core`, functional reference execution through ordinary XLA,
external reference holders, and an abstraction boundary that can support a future Pallas-like kernel layer

**Supersedes for reference work:** `.tasks/plan_ref_type.md` and the reference-specific parts of
`.tasks/plan_first_class_references_lists_and_pallas.md`

## 1. Executive decision

Ryft should add references directly to the existing array IR value universe:

```rust
pub enum ArrayIrType {
    Array(ArrayType),
    Dimension(DimensionType),
    Reference(ReferenceType<ArrayType>),
}

pub enum ArrayIrValue<A: Value<Type = ArrayType>> {
    Array(A),
    Dimension(DimensionValue),
    Reference(ArrayReference<A>),
}
```

The new `ArrayIrType`/`ArrayIrValue` architecture has already solved the heterogeneous-program problem that the older
`AtomType<T>` and `ProgramType<T>` proposals were intended to solve. Ryft should not introduce another program type
sum, add a reference flag to `ArrayType`, or create a parallel XLA-only reference representation.

References are nevertheless much more than a new enum member. `ArrayIrType` provides storage and dispatch; it does not
provide mutable resource identity, aliasing, lifetime validation, effect ordering, control-flow state threading,
transform semantics, or an executable ABI. Those capabilities must be modeled explicitly.

The canonical pipeline is:

```text
trace rich Array IR
    -> validate reference types, roots, aliases, scopes, and uses
    -> select a lowering policy
       -> ordinary program: discharge references into immutable array SSA
          -> run generic transforms on reference-free IR
          -> lower array-only IR to StableHLO/XLA
       -> future kernel region: preserve references and views
          -> lower accesses to target memory operations
```

This ordering is a correctness rule. No generic differentiation, batching, partial-evaluation, rematerialization, or
ordinary StableHLO lowering path may silently receive unresolved references.

## 2. Goals and non-goals

### Goals

- Represent a mutable reference to one array as an explicit, distinct `ArrayIrType` member.
- Give eager and staged reference values stable resource identity and explicit invalidation semantics.
- Support scoped creation, reads, writes, swaps, additive updates, and consuming freeze operations.
- Keep references second-class and reject unsupported aliases and escapes before any mutation or lowering occurs.
- Make state visible to existing transforms through a conservative ordered-state effect.
- Provide a separate resource-access analysis that resolves operands, captures, allocations, and views to canonical
  reference roots.
- Discharge eligible reference programs in `ryft-core` into reference-free array programs plus deterministic external
  state metadata.
- Thread discharged state correctly through conditions, while loops, scans, nested calls, and captures.
- Compile local reference programs through the existing array-only StableHLO/XLA path.
- Support external and captured XLA references through hidden final-state outputs and input/output aliasing without
  making donation a semantic requirement.
- Preserve the same logical reference type, operations, views, and resource analysis for a future Pallas-like kernel
  lowering.
- Produce precise errors for every unsupported transform, alias, shape, lifetime, and backend combination.

### Non-goals for the first usable release

- References to dimensions, references, Lists, tuples, or arbitrary `ArrayIrType` values.
- Returning references from public computations or higher-order regions.
- Arbitrary aliases, alias merging, reference equality operations, or user-observable raw resource identifiers.
- Bounded-dynamic mutation, unbounded or nonreplicated dynamic state, multi-host state, and zero-space external
  references. Static replicated/sharded state and finite replicated bounded-dynamic read-only state are supported on
  fully addressable single-process meshes.
- Writes from a while-loop condition region.
- External reference mutation under automatic differentiation.
- Native tangent/cotangent references, gradient references, or a `vjp.with_refs` analogue.
- Mapped external references, replicated shared writes, race semantics, atomics, barriers, or asynchronous copies.
- General dynamic allocation or deallocation in ordinary XLA.
- A complete Pallas/kernel language. This plan preserves the necessary boundary but does not implement the kernel
  compiler, launch model, or scheduler.

## 3. External model: what to retain from JAX

JAX's current public `Ref` design provides the right semantic reference point:

- A `Ref` is distinct from an immutable array and supports explicit indexed access rather than ordinary array
  arithmetic.
- Local references can be internally stateful while the enclosing function remains pure to its caller.
- Reference arguments and captures make a function externally stateful.
- References are second-class: they cannot be returned, duplicated at higher-order boundaries, frozen outside their
  creation scope, or nested inside other references.
- Ordinary JAX/XLA compilation functionalizes state before normal lowering.
- Pallas preserves the same reference idea and lowers accesses as memory operations.

Current primary references:

- [JAX Ref guide](https://docs.jax.dev/en/latest/array_refs.html)
- [`jax.ref.new_ref`](https://docs.jax.dev/en/latest/_autosummary/jax.ref.new_ref.html)
- [JAX Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html)
- [XLA input/output aliasing](https://openxla.org/xla/aliasing)

Ryft should adopt the architecture, not copy every current API or internal implementation detail. In particular,
Ryft's existing transform, region, capture, dynamic-dimension, sharding, and executable-signature systems determine
where its contracts must live.

## 4. Ryft today

### Existing foundations to reuse

| Capability | Current ownership | Consequence for references |
|---|---|---|
| Mixed array IR type universe | `crates/ryft-core/src/arrays/types/ir.rs` | Add `Reference`; do not create `AtomType<T>`. |
| Mixed runtime value universe | `crates/ryft-core/src/arrays/ir.rs` | Add an identity-bearing `ArrayReference<A>`. |
| Checked member projection | `ValueProjection` and tracer projections | Keep ordinary array operations narrow and reject references unless explicitly read. |
| Composite operation family | `ArrayIrOperation` | Add cross-member reference operations as native mixed variants. |
| Composite XLA domain | `XlaDomain` and `XlaOperation` | Trace references without changing the public array-only facade initially. |
| First-class regions | condition, while, scan, calls, captures | Discharge can widen region interfaces and thread hidden state. |
| Effect summaries | `Effect`, `Effects`, and region sealing | A coarse ordered-state effect can make existing passes conservative immediately. |
| Transform caching/replay infrastructure | `Program`/`Region` transforms | Implement discharge as a structural replay transform, not backend pattern rewriting. |
| Array-only XLA executable ABI | `XlaExecutableSignature` | Keep StableHLO array-only and derive physical aliases after existing projections. |
| PJRT donation/copy protection | `ryft-xla` arrays and `ryft-pjrt` execution | Optimize physical reuse without relying on it for mutation semantics. |
| Function argument attributes | `ryft-mlir` `func` wrappers | Emit XLA entry input/output aliases without a new MLIR abstraction unless a gap is proven. |
| Mosaic GPU/TPU operations | `ryft-mlir` Mosaic dialects | Reuse later as target primitives for preserved kernel references. |

### Important current constraints

- `ArrayIrType` now contains `Array`, `Dimension`, and `Reference`; the completed compiler-guided exhaustive-match
  audit intentionally supports metadata-only paths and rejects unresolved references at transform/backend boundaries.
- `ArrayIrType::Identity` is `DimensionVariable`. Reference referents may contain those dimension identities, but
  runtime reference identity must not become a `Type::Identity`.
- `Effects` is a small `Copy` bitset with four global classes, including `OrderedState`. It cannot encode dynamic
  resource identities.
- `Operation::effects()` has no instruction operands and therefore cannot identify a concrete root by itself.
- Generic partial evaluation may move all-known work ahead of residual work; unresolved state operations cannot use
  its default rule.
- Generic reverse-mode transposition rejects most observable effects and must not receive unresolved mutation.
- `ArrayIrBatch` and `DifferentiableType for ArrayIrType` currently match member kinds exhaustively. Adding an arm to
  satisfy exhaustiveness must not accidentally promise mapped-reference or zero-tangent-reference semantics.
- The ordinary XLA boundary deliberately projects the composite IR back to arrays and dimensions. That remains the
  correct boundary after adding references.
- Captures are currently treated as non-donatable ordinary values. Captured references will need a distinct internal
  donation policy.

## 5. Semantic model

### 5.1 Reference type

Introduce a reusable `ReferenceType<T: Type>` wrapper in core, and specialize it to `ArrayType` in `ArrayIrType`:

```rust
pub struct ReferenceType<T: Type> {
    referent: T,
}

// Inside ArrayIrType:
Reference(ReferenceType<ArrayType>)
```

Requirements:

- `ReferenceType<T>` implements `Type` by delegating identity traversal/renaming and structural refinement to `T`,
  while keeping reference/non-reference kind separation in the enclosing composite universe (compatibility does not
  delegate — it is exact equality, below). Identity-renaming derivation must be explicit: the default
  `Type::derive_identity_renaming` establishes refinements and then returns an *empty* renaming
  (`programs/types.rs:117-123`), so `ReferenceType<T>::derive_identity_renaming` must unwrap complete reference
  signatures and delegate to `T::derive_identity_renaming`, making declared `ref<f32[n]>` against actual `ref<f32[m]>`
  produce `n -> m` rather than merely validating. The composite `ArrayIrType` arm instead calls
  `ArrayType::extend_identity_renaming` (`arrays/types/arrays.rs:404`) with the shared renaming and refinement
  accumulators.
- `ReferenceType<T>` needs a real `Refinements` adapter: `ReferenceTypeRefinements<T>` delegates whole-signature
  refinement to `T::Refinements` so cross-signature facts survive. A bare `type Refinements = ()` would be unsound
  for the standalone generic contract: `()` checks pairs and retains no shared-identity state
  (`programs/types.rs:237-255`), so declared inputs `ref<f32[n]>, ref<f32[n]>` would accept `ref<f32[3]>,
  ref<f32[4]>`. The composite path is unaffected and never consults the member's own `Refinements`:
  `ArrayIrTypeRefinements::visit_dynamic_to_static_refinements` (`arrays/types/ir.rs:230-252`) is the single
  member-dispatch site, and the `Reference` arm delegates the referent pair to the existing shared
  `arrays: ArrayTypeRefinements` accumulator across array and reference members alike.
- For `ReferenceType<ArrayType>`, the referent supplies data type, shape, layout, sharding, memory, and dimension
  identities.
- Do not duplicate `Memory` or other array metadata in `ReferenceType<ArrayType>`.
- Do not store resource identity, allocation identity, capture identity, or SSA identity in the type. This is
  load-bearing for dispatch: `ReferenceType` participates in the retained-JIT dispatch key
  (`XlaDispatchKeyKind::Exact` over `ArrayIrType` signatures), so its `Hash`/`Eq` must stay
  structural and holder-free or every holder mints its own specialization. Pin this with the same test shape as the
  existing `DimensionType` structural-hash test (`arrays/ir.rs:186-201`).
- Do not add a `ReferenceKind` until at least two implemented behaviors require it. A speculative flag bag would make
  compatibility, serialization, and transforms harder without adding semantics.
- `ReferenceType<T>` compatibility is exact equality, never a delegation to `T::is_compatible_with`: for `ArrayType`
  that relation is broadcastability (`arrays/types/arrays.rs:497`), and a reference handle cannot be implicitly
  broadcast or promoted. Structural refinement still delegates to the referent: `is_refined_by` holds only between
  two `ReferenceType<T>`s whose referents satisfy `T::is_refined_by`. An enclosing composite universe never treats a
  reference type as compatible with the referent itself.
- Reference/Reference identity traversal and refinement delegate through the referent, using the existing shared
  `ArrayTypeRefinements` logic. All cross-kind pairs fail.
- A reference handle is not a numeric scalar or complex value, even when it refers to a scalar or complex array.
  `Type::is_scalar()` and `Type::is_complex()` return `false`.

External XLA state mutation requires exact physical referent compatibility. Dynamic refinement may remain valid for
type checking, but external input/output aliasing is admitted only after physical shape, layout, sharding, and
dynamic-extent compatibility have been proven.

### 5.2 Runtime value and identity

The reusable `Reference<V: Value>` owns the `Arc`-shared root holder. Array IR wraps it in
`ArrayReference<A> { root, view }`, keeping root resource identity separate from the handle-local coordinate mapping.
Root identity follows the `DimensionVariable` precedent (`arrays/types/dimensions.rs:198-255`): `PartialEq` via
`Arc::ptr_eq`, `Hash` via `Arc::as_ptr`, and a diagnostic-only `Display`. `ReferenceId` is a handle derived from that
pointer (used for stable lock ordering and diagnostics), not a global counter — no counter scheme exists in
`ryft-core` outside cache statistics, and none should be introduced. `ArrayIrValue<A>` stores `ArrayReference<A>`.
Root `Reference` equality and hashing identify the shared resource; `ArrayReference` equality and hashing additionally
include the exact view. The synchronization primitive remains private, but the semantics are fixed:

- Cloning the internal value aliases the same resource; it does not copy the array contents.
- Equality and hashing use stable reference identity, never mutable contents. `Display` is deterministic and
  type-based (renderings back diagnostics and structural fingerprints), while `Debug` includes the runtime identity.
- No public program operation compares reference identities.
- A read returns an immutable snapshot. Later writes may reuse storage only when doing so cannot mutate any retained
  snapshot; otherwise eager execution or the backend must copy or use copy-on-write protection.
- `new_reference(value)` does not invalidate or make later mutations observable through the initializer `value`.
  Likewise, two distinct roots initialized from storage-sharing values remain logically independent. Physical reuse
  must copy-protect whenever either invariant would otherwise be violated.
- The declared referent type remains invariant for the reference's lifetime. Finite bounded-dynamic declarations may
  admit different concrete runtime refinements only on a backend path that preserves their extents; XLA currently
  admits that class for replicated read-only external state and rejects mutation.
- Frozen, failed, or otherwise invalid state produces an explicit error rather than panicking or returning stale data.
- The holder is an opaque `Parameter` leaf. This is free: `Parameter` is a bare marker trait (`parameters.rs:153`),
  so a leaf implementation exposes nothing and there is no traversal behavior to suppress.
- Identity renaming of an eager `ArrayIrValue::Reference` preserves the shared root while deriving exact bidirectional
  root-to-handle and handle-to-root mappings. A type-changing renaming is accepted only when both value-reconstruction
  directions are valid and inverse at the declared types; non-bijective mappings fail without changing the holder.
  Reference constants remain structurally rejected at region sealing because process-local holder identity cannot
  participate in deterministic program storage. The staged `XlaConstant` captured-reference path remains
  metadata-only and renames independently.
- Mutable external references are not literal constants and are not serialized as payload data. They enter through
  arguments or typed captures whose invocation supplies the holder. This is enforced structurally:
  `Value::validate_as_constant` rejects reference values at region sealing — the one boundary every region
  construction path crosses (program builders, `Program::new`, and region imports alike, nested regions included) —
  because a holder's process-local identity is deliberately absent from its deterministic rendering, and a stored
  reference constant would let two programs over distinct holders render (and therefore fingerprint) identically.

The staged resource identity is not the runtime `ReferenceId`. Within a program it is derived from the canonical SSA
root: an entry reference input, a captured reference, or the result of a local reference allocation. Invocation-time
identity validation connects external roots to actual holders.

Pending readiness, leases, cumulative errors, generations, and completion callbacks (Phase 10) need a backend-neutral
dependency abstraction, because `Value` exposes none of those concepts and backend fence types must not leak into
`ryft-core`: define a type-erased completion/dependency token in core alongside `Reference<V>` (clone, await, query,
register-completion-callback), which `ryft-xla` implements over `ExecutionFence`. Pending holder states and read
leases store that token directly; there is no `ryft-xla` side map keyed by `ReferenceId`.

`Reference<V>` should implement `Typed<Type = ReferenceType<V::Type>>` and the traits needed for storage/projection.
It should not automatically implement `Value` in the first slice: doing so would require a standalone
`Domain<Type = ReferenceType<V::Type>, Value = Reference<V>>` and would incorrectly imply a homogeneous reference
operation universe. `ArrayIrValue<A>` remains the actual `Value`; its `ValueProjection<ReferenceType<ArrayType>>`
projects to `ArrayReference<A>`. A standalone `Value` implementation can be added later only if an independent
reference domain has real use.

The generic carriers are reusable representation, not a promise that every `T` or `V` is a supported referent. Each
composite universe chooses its admitted specialization. Array IR admits only `ReferenceType<ArrayType>` and
`ArrayReference<A>`; it remains structurally unable to contain a reference to a dimension, another reference, or a
List.

### 5.3 Operations

Start with root and view operations:

| Operation | Logical signature | Semantics |
|---|---|---|
| `NewReferenceOperation` | `Array -> Reference` | Create a scoped root initialized from the array. |
| `ReferenceIndexOperation` | `Reference -> Reference` | Create a pure axis-removing view alias. |
| `ReferenceSliceOperation` | `Reference -> Reference` | Create a pure static unit-stride slice alias. |
| `ReferenceReadOperation` | `Reference -> Array` | Snapshot the current value. |
| `ReferenceSwapOperation` | `(Reference, Array) -> Array` | Return and replace the selected root or view. |
| `ReferenceAddUpdateOperation` | `(Reference, Array) -> ()` | Ordered accumulation into the selected root or view. |
| `FreezeReferenceOperation` | `Reference -> Array` | Return final state and invalidate the entire alias family. |

`swap` is the sole replacement primitive. `write`/`set` are binding-level sugar that stages a `swap` and discards the
result; they never appear as distinct IR operations. This keeps exactly one discharge rule, differentiation rule,
effect declaration, and lowering arm per replacement behavior and avoids the canonicalization question two primitives
would create (an unused-result `swap` is a `write`). JAX/Pallas precedent confirms a single primitive suffices even
under kernel lowering: an unused old value lowers to a plain store at the target level. Additive update remains
distinct because future kernel lowering and differentiation need to distinguish accumulation from replacement.

The access contract classifies `swap` as `Write` (JAX does the same for `swap_p`), and `Write` deliberately does not
assert the absence of a read: a used-result `swap` observes prior state through its result, so generic consumers must
treat every `Write` conservatively and never remove an earlier write based on the mode alone. Read-ness is
operation-specific knowledge rather than a generic result-liveness rule — kernel lowering of `swap` knows its own
old-value result and emits an exchange when it is live and a plain store when it is provably dead (§6.2). A
write-only or uninitialized kernel operand must never require readable previous contents, which the dead-result
store lowering guarantees (Phase 12).

These are composite-native `ArrayIrOperation` variants because their signatures cross array/reference kinds. Do not
create a projected homogeneous reference operation universe until a concrete family of reference-to-reference
operations makes it useful.

Replacement/update type legality is stricter than `ArrayType::is_compatible_with`:

- `write` and `swap` perform no implicit broadcasting or data-type promotion. The stored value must have the same
  instantiated referent type and the same exact runtime extents, layout, sharding, and memory.
- A finite bounded-dynamic input may refine a declared referent only for a backend path that preserves its runtime
  extents. XLA admits replicated external references read-only; bounded-dynamic mutation remains rejected because the
  backend does not preserve the aliased runtime extent.
- `add_update` may use ordinary array addition semantics internally only when the inferred addition result has exactly
  the current instantiated referent type. Any promoted or broadcast result that would change stored type is rejected.
- Reference-type compatibility/refinement is not a substitute for these operation-specific storage checks.

Array IR represents explicit reference views as an ordered root-relative mapping:

```text
ReferenceView {
    root,
    ordered transforms: [index, static unit-stride slice]
}
```

Only implemented and validated transforms may appear. A view preserves its base root, composes its mapping back to
that root, and never allocates a new resource. Indexed reads, writes, swaps, and accumulations then operate through the
view. Views are never freezable: `freeze` accepts only a root handle, and `freeze(view)` is rejected in the MVP —
whether it would return the slice or the complete root is ambiguous, and neither reading composes cleanly with
alias-family invalidation. The initial implementation deliberately rejects dynamic indices, non-unit strides, and
reshape/transpose/bitcast views until each has an explicit inverse/update proof. It reuses canonical slice, reshape,
and update-slice operations for discharge rather than inventing a second indexing language.

### 5.4 Purity, lifetime, and second-class restrictions

A program that allocates and uses only local references is externally pure after discharge. It may explicitly freeze a
root to recover its final array, or implicitly discard a still-live unescaped root at the end of its creation region. A
program that reads or mutates an argument or captured reference is externally stateful.

Initial static rules:

- References cannot be public program outputs.
- References cannot escape the complete program or a local allocation scope. Existing roots may be forwarded through
  condition results, fixed-point while/scan carries, and nested-call results only when every provenance path resolves
  to the same canonical root; scan sequence outputs remain invalid. Derived views do not cross attached-region
  boundaries in Phase 8 and must be recreated from a carried root inside the region. Discharge-generated hidden state
  carries and outputs are arrays, not references.
- References cannot be nested in references or Lists.
- Only explicit view operations introduce aliases.
- Every reference operand resolves to exactly one canonical root.
- A local root cannot escape its creation scope.
- `freeze` is valid only in the local root's creation scope, consumes the root, and invalidates all aliases and views.
- Scope exit implicitly discards any still-live, unescaped local root without producing an array result. Discharge
  drops the root's final current state. Interpreted execution needs no separate invalidation registry: static
  nonescape guarantees that releasing the region environment drops its final handles, making explicit invalidation
  unobservable. Directly eager references have no program-owned creation scope, so they are never implicitly
  invalidated: they live until an explicit `freeze` or until the last handle drops, and their second-class
  restrictions are enforced at program boundaries (arguments and captures), where validation exists.
- External or captured references cannot be frozen in the initial implementation.
- All uses after freeze are rejected, including uses through pre-existing views or nested captures.
- Allocations inside a condition branch, loop iteration, scan step, or nested call are allowed only if completely
  frozen or implicitly discarded within that invocation and unable to escape.

An outer root cannot be frozen from a nested branch, loop, scan, or call because that is not its creation scope. A root
created within such a nested region may be frozen there or implicitly discarded at that region's exit. Validation must
reject inconsistent lifecycle paths before discharge.

Initial invocation-time rules:

- The same external holder cannot be supplied through two public argument positions.
- The same holder cannot be both passed and captured.
- Two capture paths cannot resolve to the same mutable holder.
- Alias validation completes before interpretation, lowering, or runtime state extraction mutates anything.

Static SSA validation and invocation-time holder validation are distinct and both are required. Static validation
lands in Phase 2; holder-identity validation lands with the external runtime boundary in Phase 9 and must complete
before that runtime extracts or mutates holder state.

## 6. Effects and resource-access analysis

### 6.1 Coarse correctness effect

Add `Effect::OrderedState` to the existing `Effects` bitset. Every unresolved state operation reports it, including
reads, creation, writes, swaps, accumulations, and freeze. A view-construction operation that only derives alias
metadata may remain pure; every actual access through that view is ordered.

Reads are ordered in the conservative model because the unchanged reference handle does not encode a new SSA version;
their result depends on intervening writes. This global chain may serialize independent resources, but it immediately
keeps existing simplification, liveness, control-flow summaries, and rematerialization logic safe. Partial evaluation
is the exception: its default rule folds all-known effectful operations, so it needs its own dedicated gate (§9,
Phase 2) rather than inheriting safety from the ordered class. The cost is bounded: it only pessimizes pre-discharge simplification (for example, common-subexpression
elimination of adjacent reads), and every transform and lowering route discharges first, so no post-discharge pass
pays for it. Do not attempt to relax pre-discharge read ordering as an optimization.

Do not make `Effects` carry dynamic resource IDs. It should remain a small operation-level classification used by
existing passes.

Why a new class rather than reusing `OrderedIo`: the generic ordered guarantees (retention, no reordering,
conservative gates) would indeed fall out of any ordered class, but the class is the policy discriminator in three
places where state and I/O must diverge. Partial evaluation folds and executes all-known ordered-I/O operations on
the known side (correct for `print`, wrong for mutation), so its gate must distinguish the two. The post-discharge
invariant "no state effects remain" (§8.2) is only expressible as a cheap `contains(OrderedState)` check if state
has its own bit — a program that also prints legitimately keeps `OrderedIo` after discharge. And XLA lowering
threads one token chain per ordered class (`EffectTokens`), so a leaked reference operation sharing `OrderedIo`
would be silently token-threaded instead of failing the "references must be discharged before lowering" verifier.
A shared class would also serialize prints against reference accesses for no semantic reason, and the future
per-resource (input-indexed) refinement hangs specifically off the state class. This mirrors JAX, where `RefEffect`
is a distinct family from I/O effects so each transform can whitelist or reject it independently.

### 6.2 Precise semantic contract

Add a separate backend-neutral reference semantics/access contract. Its exact Rust API should be prototyped against
one allocation, one read, and one view before finalizing, but it must express at least:

```text
ReferenceOperationSemantics
    outputs:  NewRoot { output_index }
              | Alias { output_index, input_index, kind: Identity | View }
    accesses: input_index -> Read | Write | Accumulate | Consume
```

All indices are operation-local operand/result positions, never resource identifiers. Output classification is
mutually exclusive by construction (an output is a fresh root or an alias, never both), and `Alias` carries exactly
one `input_index` on purpose: it structurally encodes the one-canonical-root invariant (§5.4), so multi-source
aliases are unrepresentable rather than merely rejected. There is deliberately no `ReadWrite` mode (`swap` classifies
as `Write`, matching JAX's `swap_p`) and no `Freeze` mode (`freeze` classifies as `Consume`, a lifetime event that
also covers a future result-less `free_reference`). View operations attach their ordered coordinate-transform stacks
to the arrays-owned `ArrayReferenceOperation` contract; the generic `Alias { kind: View }` edge records root
provenance without making the generic program layer depend on array coordinate descriptors. `ReferenceAnalysis`
validates and stores the exact composed `ArrayReferenceView` once, and discharge consumes that artifact rather than
re-resolving view coordinates.

The program-level `ReferenceAnalysis` resolves these templates to region-relative canonical roots:

```text
RegionInput(region, input_index)
Allocation(instruction, output_index)
```

Entry-region `RegionInput` roots are additionally classified by `ReferenceSource` as captures or public inputs.
Nested invocations retain `ReferenceRegionInputBinding` substitutions from each formal region root to its canonical
caller root; the analysis does not rewrite the nested region's own access records into a different root namespace.

It must:

- propagate roots through views;
- substitute region inputs and captures at condition, loop, scan, and call boundaries;
- record nested-region substitutions so consumers can resolve accesses in caller context;
- preserve deterministic root order independent of hash-map iteration;
- distinguish ordinary ordered accumulation from future atomic/commutative accumulation;
- treat `Write` as an over-approximation that permits observing prior state through results: never infer
  absence-of-read from the mode, and never remove an earlier write based on it alone. Whether a specific write
  actually reads is operation-specific knowledge — the descriptor deliberately carries no access-to-result mapping,
  so kernel lowering of `swap` uses its own old-value result (a plain store when that result is provably dead, an
  exchange when it is live) rather than a generic result-liveness rule;
- report unresolved, multiply rooted, escaped, or invalidated operands precisely.

This analysis becomes the shared source of truth for static validation, discharge, invocation alias checks, eventual
per-resource scheduling, and Pallas/kernel lowering.

## 7. Reference validation

Implement reference validation as a dedicated array-IR analysis after generic `Program` construction. Avoid putting
reference-specific rules into the generic builder unless a rule is truly common to all future second-class kinds.

Validation is all-or-nothing and runs before replay. An invalid program must not produce a partially discharged
artifact.

The validator proves:

- each reference type wraps an array and preserves referent identity/refinement rules;
- each reference use resolves to one live root;
- every view composes to its root and respects bounds/type invariants;
- references do not occupy forbidden output, carry, sequence, constant, or data-structure positions;
- local roots do not escape their creation scope;
- freeze is local, unique, consuming, root-only (never through a view), and followed by no use;
- external roots are not frozen;
- region captures and inputs do not create forbidden static aliases;
- while conditions do not write, accumulate into, or consume roots that enter the condition evaluation, because the
  Boolean-only condition boundary cannot return their final state. A condition-local allocation that neither escapes
  nor survives the evaluation is externally pure and may remain supported. Phase 2's operation-owned access policy
  validates explicit and captured entering roots, and capture-aware discharge makes their read-only state flow
  explicit in Phase 5;
- transform/backend-specific restrictions are satisfied before the corresponding pipeline starts.

Diagnostics must name the operation, root source, scope, and violated rule. Generic “expected array” errors are
insufficient for lifecycle or alias failures.

## 8. Backend-neutral state discharge

### 8.1 Capture-lifting boundary

Discharge operates on an open `Program`, after `ClosedProgram::to_program_with_lifted_captures` has converted capture
references into leading program inputs. It does not mutate or fabricate concrete capture-table values.

The compilation/core-interpretation adapter must:

1. Validate the source `ClosedProgram` and lift its captures using the existing canonical mechanism.
2. Retain `capture_count` and classify each leading reference input as `ReferenceSource::Capture(index)`; later
   reference inputs are classified from public parameter positions. (`to_program_with_lifted_captures` records
   nothing itself; `XlaLoweredProgram::capture_count` is the existing precedent for carrying this count alongside a
   lifted program — mirror it.)
3. Run reference analysis and discharge on the lifted open program.
4. Return a binding recipe that maps each logical external-reference slot back to its original capture or public
   argument position.
5. At interpretation/execution, snapshot or transactionally extract the holder's current array according to the slot's
   access disposition, then install any hidden final state back into that same holder.

Nested closed calls are already normalized at staging time: `StagedFunction::call_with_flat_capture_references_in_context`
re-registers callee captures in the caller's table and attaches the callee's memoized lifted program
— verify rather than implement, and memoize discharge alongside that `OnceLock` cache rather than invalidating it.
One gap is real, however: `to_program_with_lifted_captures` rewrites
capture-referencing constants only in the entry region, while `CaptureReference`s inside attached regions are
preserved verbatim and resolved later against the hidden capture-argument prefix (`captures.rs:546-551`). The "no
reference-typed `CaptureReference` survives lifting" invariant therefore does not hold automatically. Phases 4 and 5
are implemented together, so they skip temporary reject-then-remove churn. Capture-aware `ReferenceAnalysis`
resolves an attached-region `CaptureReference` against its active lexical scope: ordinary structured regions inherit
the enclosing scope, while nested calls establish a callee scope rooted at their leading lifted inputs. Per-invocation
bindings then translate those formal roots into the caller namespace. Discharge consumes that exact artifact while
introducing the explicit immutable array carries required by each enclosing condition, loop, scan, or call boundary.
This is closure conversion during the one-shot discharge, not a separately materialized reference-bearing
intermediate program: it avoids making while captures into invalid reference results or misclassifying scan captures
as stacked sequences. Captured references used directly inside `condition`/`while`/`scan` bodies and nested calls
remain flagship use cases. The pipeline order is canonical and fixed: entry capture lifting, capture-aware
`ReferenceAnalysis` over the exact lifted source arena, then discharge using exactly that analysis artifact. The
resulting ordinary program may contain array inputs/captures but no rewritten capture table containing mutable array
contents, keeping `ClosedProgram`'s capture type/value invariant intact and giving core discharged interpretation and
XLA compilation one boundary contract.

### 8.2 Result contract

Discharge belongs in `ryft-core`. It returns a reference-free program and logical external-state metadata:

```rust
pub struct DischargedReferenceProgram<V, O> {
    program: Program<...>,
    public_output_count: usize,
    external_states: Vec<DischargedReferenceState>,
}

pub enum ReferenceSource {
    Capture { index: usize },
    PublicInput { index: usize },
}

pub struct DischargedReferenceState {
    source: ReferenceSource,
    discharged_input_index: usize,
    final_state_output_index: Option<usize>,
}
```

The implemented API keeps these fields private and exposes read-only accessors; the state's list position is its
logical slot, and `is_mutated()` derives the access disposition from the presence of a hidden final-state output.
The contract is:

- discharge consumes the validated `ReferenceAnalysis` artifact as its input; it never re-derives root resolution,
  alias structure, or region access summaries independently, so the validator and the rewrite cannot drift;
- all indices are logical flattened program indices, never PJRT/XLA physical indices;
- original public outputs retain exactly their structure and order;
- hidden final states for mutated external roots follow public outputs in canonical external-root order;
- local roots create no external state slots;
- read-only external roots retain an input slot but have no hidden final-state output;
- mutated external roots have exactly one hidden final-state output;
- referent types are not stored: the discharged program's input type at `discharged_input_index` is the referent
  type, and duplicating it would create a second source of truth;
- compilation and runtime metadata never contain process-local `ReferenceId` values;
- a successful result contains no reference types, operations, values, accesses, or `OrderedState` effects.

### 8.3 Straight-line rewrite

Maintain one current immutable array SSA value per root in program order:

```text
new_reference(x)      -> current[root] = x
read(root)            -> current[root]
swap(root, x)         -> old = current[root]; current[root] = x; result = old
add_update(root, x)   -> current[root] = add(current[root], x)
freeze(root)          -> result = current[root]; close root
```

Views translate reads and updates through canonical slice/gather/scatter/update operations while updating the same
root state. Preserve array data type, dimension identities, layout, sharding, memory, operation payloads, and all
non-state effects. Instructions currently carry no source-location field.

### 8.4 Regions and calls

The replay transform may change types, arities, captures, region inputs, and region outputs. `Program::map_operations`
is insufficient.

| Construct | Discharge rule |
|---|---|
| Condition | Pass the current canonical state tuple to both branches. Append the same state tuple to both branch outputs. A branch that does not write a root returns its incoming state unchanged. |
| While condition | Read the current array carries. Initially reject writes because the condition contract returns only the Boolean predicate and cannot expose a final state from the terminating evaluation. |
| While body | Add mutated roots as explicit loop carries in canonical order. Preserve incoming state on zero iterations. |
| Scan | Add mutated roots as scan carries, never reference-valued sequences. Keep per-step public outputs separate from final hidden state. |
| Nested call | Replace reference arguments/captures with array state inputs and hidden state outputs, then substitute the callee's root mapping into the caller. |
| Local allocation in a region | Discharge locally only when creation and all uses remain within one region invocation; explicit freeze returns state, while region exit implicitly discards a still-live unescaped state. |

Canonical state order derives from root order at the enclosing boundary, not branch or region encounter order. Nested
replay must compose this order transitively.

### 8.5 Discharge proof obligations

- Eager reference interpretation, discharged interpretation, and an explicit hand-written state-passing oracle are
  observationally equivalent.
- Zero-iteration loops and untaken branches preserve incoming state exactly.
- Different write counts in condition branches still produce identical state signatures.
- Multiple roots never exchange state slots.
- Invalid input fails before the destination builder contains a partial result.
- Discharge is deterministic.
- A reference-free source program produces an identity discharge artifact: its program is unchanged, its public output
  count is the source output count, and its external-state list is empty. Discharge itself is one-shot: it accepts a
  source program plus validated analysis and returns `DischargedReferenceProgram`; the wrapper is not a valid input to
  discharge because replaying its inner program would misclassify hidden final-state outputs as public outputs. Do not
  add an `already_discharged` bit to generic `Program`.
- Simplification after discharge may optimize the pure state chain; simplification before discharge retains every
  unresolved state access in order.

### 8.6 Error ownership

Keep reference failures with the layer that has enough evidence to diagnose them, and do not introduce overlapping
catch-all error types:

- `ReferenceError`, colocated with `Reference<V>`, owns eager holder and runtime-lifecycle failures such as frozen,
  invalid, or poisoned state and referent-type mismatches.
- `ReferenceAnalysisError`, colocated with `ReferenceAnalysis`, owns static root resolution, multiple roots, use after
  freeze, escape, forbidden boundary positions, invalid views, alias ambiguity, and unsupported region/capture forms.
  Expose it through `ProgramError::Custom` rather than adding reference-specific variants to generic `ProgramError`.
- Discharge adds no public `ReferenceDischargeError` initially. User-correctable legality failures occur during the
  prerequisite analysis; ordinary replay/build failures remain `ProgramError`, and disagreement between a validated
  analysis and its exact source program is an internal malformed-artifact invariant.
- Each generic transform owns its targeted surviving-reference error (`BatchingError`, `DifferentiationError`, and the
  corresponding partial-evaluation or rematerialization error path); there is no generic reference-transform error.
- `ryft-xla` adds a backend-owned unsupported-reference-ABI error for logical-to-physical state restrictions and a
  distinct undischarged-reference lowering error for the ordinary StableHLO boundary. Persistent corruption remains a
  persistent-executable error, and PJRT execution failures remain PJRT errors.

Core errors never contain physical buffer indices, PJRT values, StableHLO attributes, or backend fence types.

## 9. Transformation policy

Every public transform entry point must either discharge an eligible local-reference scope before invoking generic
logic or return a targeted reference-specific error.

| Transform | Initial policy |
|---|---|
| Simplification/liveness | Undischarged operations remain live and ordered through `OrderedState`. After discharge, use ordinary pure optimization. |
| Partial evaluation | Never execute, fold, or split an unresolved state chain. Discharge eligible externally pure local state first; reject any remaining reference-bearing partial evaluation in the MVP. Whole-chain residualization is a later explicit design. |
| Forward-mode AD | Support externally pure local refs only through trace -> validate -> discharge -> JVP. Reject external refs and surviving ref operations. |
| Reverse-mode AD | Discharge before linearization/transposition. Never treat a reference handle as a zero tangent and silently lose write-to-read dependencies. |
| Direct/eager AD APIs | Add an outer trace -> discharge -> replay route. Program-level discharge support does not make direct `DifferentiationContext` execution safe automatically. |
| Batching | Support local refs through discharge-before-batching. Reject external/captured refs, mapped refs, replicated shared writes, and ambiguous lane writes initially. |
| Rematerialization | Discharge local state first. Unresolved state is a non-recomputable boundary and may not be duplicated. Reject externally stateful rematerialization. |
| Custom derivatives | Reject refs in rule regions initially. Add semantics only with a dedicated design. |
| Compilation/lowering | Ordinary XLA always validates and discharges. A future kernel region may explicitly select preservation. |

If `DifferentiableType for ArrayIrType` requires an exhaustive representational arm, that arm must not advertise native
reference differentiation. A legality guard must prove that no `Reference` reaches generic AD, and tests must assert
targeted rejection if one does. The same principle applies to any representational `ArrayIrBatch` arm.

Advanced external-reference AD, including caller-provided gradient accumulator references analogous to JAX's current
`vjp.with_refs`, is a later design milestone after functionalized local-reference AD is correct.

Discharge does not expand the capabilities of the underlying pure transform. Local-reference AD inherits every
existing pure-program restriction. In particular, bounded while-loop reverse mode may be a positive conformance case,
while the currently unsupported unbounded while-loop transpose remains a targeted error until first-class List/tape or
another residual-storage strategy exists.

## 10. Public API staging

### Stage A: local references inside array-only functions

Keep the current public array-only JIT facade. An array tracer can create an internal composite reference value using
the same parent-binding pattern used by array-to-dimension operations (template: `DimensionSize` at
`operations/dimensions/dimension_size.rs:66-92`, including its `ProjectedValue` receiver form). Reference methods bind composite operations and
return arrays or no value. References cannot cross the public function boundary.

This delivers the core programming model without redesigning retained compiled-function argument types.

### Stage B: eager public holders and captured references

Expose a controlled reference holder with identity, invalidation, and sequencing semantics. Capturing it in a compiled
closure makes the call externally stateful. Captures remain typed side-table values, not literal program constants.

### Stage C: explicit external reference arguments

Add a heterogeneous compiled-function facade over `ArrayIrType`/`ArrayIrValue<Array>` or another typed boundary that
preserves existing array-only APIs. Do not weaken the current array projection merely to accept references.

Externally stateful calls return a completion-bearing wrapper conceptually like `ReferenceExecution<Output>`, even
when `Output` has no leaves. Naming and seam: the existing pure, output-only call remains unchanged — it is not
"synchronous" (static-shape XLA calls already enqueue work and return pending arrays, discarding only the explicit
fence carrier in `execute_compiled_async`); `call_statefully` is the blocking Phase 9 convenience; and
`call_statefully_async` returns the completion-bearing wrapper, keeping its public shape in Phase 10. Model the
stateful surface as an opt-in `StatefulCompilationDomain` capability rather than required methods on every
`CompilationDomain`: `CompilationCall` fixes `RuntimeOutput = Output::To<D::Value>`, so the stateful path is a new
request/method, not a reparameterization of the existing trait. Refactor executable
specialization acquisition so pure and stateful invocation share the dispatcher pipeline instead of duplicating it,
and make ordinary calls explicitly reject executables with non-empty external-state metadata — the output-only path
must never be the one that silently discards the only fence/error carrier for an externally stateful call.

No stage returns reference handles from compiled code.

## 11. Ordinary XLA lowering and execution

### 11.1 Lowering boundary

Run core discharge immediately before the current array-only entry conversion. Then verify that the program contains
only arrays, structural dimensions, and operations already supported by ordinary XLA lowering. Do not add direct
StableHLO lowering rules for reference operations.

Local references disappear entirely. External and captured references become ordinary array inputs, and mutated
external/captured references additionally produce hidden final array outputs.

### 11.2 Logical-to-physical ABI

Keep core discharge metadata logical. Extend `XlaExecutableSignature`, or add a tightly coupled companion, to derive:

```rust
pub struct XlaReferenceStateSignature {
    slot: usize,
    source: ReferenceSource,
    access: ExternalReferenceAccess,
    logical_input_index: usize,
    logical_output_index: Option<usize>,
}
```

Physical indices and referent types are deliberately not stored: `XlaExecutableSignature` already models zero-space
erasure and hidden dynamic-extent scalars through `input_mapping`/`output_mapping` and the generic
`project_inputs`/`project_outputs` projectors, and the referent type is
`XlaLoweredProgram::input_types[logical_input_index]`. Duplicating any of them creates a second source of truth.
Resolve physical indices through those accessors at use time, only after composing with:

- capture/public-input flattening;
- zero-space logical value erasure;
- hidden bounded-dynamic extent inputs;
- public, hidden state, and hidden dynamic-extent result ordering.

Never derive physical indices by adding counts or assuming one logical value maps to one physical value. This is
already the codebase's discipline: the donation path flattens captures plus public inputs and projects through the
signature — extend that construction rather than adding a parallel one.

Initially admit only these combinations, stated in terms of resolved mappings:

- read-only: `input_mapping[logical_input]` is present and there is no hidden final-state output;
- mutated: `input_mapping[logical_input]` and `output_mapping[logical_output]` are both present.

Public and executable output signatures are distinct concepts and must be modeled separately: the staged function's
public output types/structure on one side, and the executable's logical output list (public outputs followed by
hidden final-state outputs) on the other, split by a validated `public_output_count` carried through lowering,
execution, persistence, and executable replacement. Lowering validates the staged public signature and applies user
`out_shardings` only to that public prefix; each hidden state output inherits its paired input's effective sharding.

Reject any external reference whose physical input is erased, including zero-space references: the current holder ABI
requires one executable device buffer per logical state slot and defines no erased-state representation. A read-only
slot is never donated or aliased. A mutated slot has exactly one hidden output and one may-alias relation.

Entry lowering attaches `tf.aliasing_output = <physical output index>` to each mutated physical external-state input,
merged with existing argument attributes such as sharding. Read-only slots carry no alias. Aliases must be injective,
in range, and physically compatible in element type, shape, dynamic contract, layout, sharding, and memory
representation. Hidden extent scalars are never alias targets.

Use may-alias semantics. Input/output aliasing and donation are performance mechanisms, not the definition of
mutation. If an old snapshot shares a buffer, XLA/PJRT may copy-protect it; Ryft still installs the returned hidden
state as the holder's new value.

### 11.3 Runtime holder protocol

The runtime must define state transitions, concurrency, and failure behavior before exposing external references.

The protocol lands in two stages. The synchronous stage (Phase 9) awaits execution completion before installing state
and returning: the lease/reservation machinery of steps 8 and 10 below and the pending/generation failure semantics
collapse to a single install-or-poison decision under the held guards, and no pending states or read leases exist.
Waiting before returning does not by itself serialize anything —
two host threads can enter concurrently before either returns — so the synchronous stage holds per-holder guards,
acquired in stable `ReferenceId` order, for the entire execution including device completion. Holding host guards
across device execution is an accepted, explicit Phase 9 limitation; the asynchronous stage (Phase 10) replaces it
with pending states, read leases, and generation-safe chains without changing the public call shape. The full
protocol below is the asynchronous end state.

For each invocation:

1. Resolve public and captured logical roots to runtime holders.
2. Reject duplicate holders before snapshotting or extracting any state.
3. Acquire multiple holders in stable `ReferenceId` order.
4. For a read-only slot, clone/snapshot the current array and its dependency without taking ownership. For a mutated
   slot, Phase 10 likewise retains a copy-protecting holder snapshot through submission, so an immediate execute error
   leaves the holder ready. The synchronous Phase 9 implementation transactionally takes the current array while its
   long-held guard remains live.
5. Build ordinary physical arguments through `XlaExecutableSignature`.
6. Construct logical donation flags in flattened capture-plus-public-input order, then project them through
   `XlaExecutableSignature`: ordinary captures and dimensions are `false`; ordinary public arrays use the user's flag;
   reference inputs are `false` in Phase 10 because their holder-owned snapshots remain available across immediate
   submission failure; hidden extent carriers are `false`. Phase 9's earlier synchronous extraction path requested
   mutation donation subject to safe uniqueness downgrade.
7. Submit the copy-protected Phase 10 reference inputs to PJRT. Successful submission is the asynchronous protocol's
   irreversibility boundary; the transitional Phase 9 protocol conservatively marks handoff before the execute call
   because it may pass an extracted donatable holder value.
8. Immediately after successful PJRT submission, while the ordered holder guards remain held, atomically publish the
   execution fence as a read lease on every read-only holder and reserve a pending generation on every mutated
   holder; then release the guards. Nothing fallible precedes this step — today's output splitting constructs
   arrays, materializes dynamic extents, and validates ownership fallibly after obtaining the fence, so
   lease/reservation publication must come first or a failure there leaves active device reads unregistered.
9. Construct, split, and validate public results, hidden state results, and hidden dynamic extents.
10. Replace every mutated holder's reservation with its pending final value, carrying readiness events and
    generation/dependency information — or poison all mutated holders if any hidden state cannot be constructed or
    validated.
11. Reconstruct and expose only public outputs through the completion-bearing stateful-call wrapper.

Required failure semantics:

- Before the protocol's applicable irreversibility boundary, every extracted mutated state is restored.
- Passing prepared arguments to `PJRT_LoadedExecutable_Execute` remains the transitional Phase 9 irreversibility
  boundary because that implementation may pass extracted donatable holder storage. Phase 10 instead retains the
  holder's copy-protecting snapshot through submission, which safely downgrades reference-state donation when storage
  is shared. An immediate execute error without a fence therefore leaves every mutated and read-only holder ready and
  publishes neither generations nor leases. Ryft's safe `LoadedExecutable::execute` wrapper drops its retained input-
  buffer `Arc`s on this error path, whose memory-safety contract requires that an immediate error retain no
  asynchronous device access.
- After successful Phase 10 submission, Ryft does not roll back the logical mutation. If all hidden final states can be
  constructed, it installs them even when later public-output reconstruction or refinement fails. If any hidden state
  cannot be constructed or validated, it poisons every mutated holder in the invocation.
- A failure reported by the submitted execution fence poisons the complete cumulative mutation chain. An immediate
  execute error that returns no fence is not an asynchronous execution failure and leaves Phase 10 holders ready under
  the copy-protected policy above.
- Asynchronous execution failure poisons mutated holders with the execution error; later reads/writes fail until an
  explicit replacement/reset API exists. Read-only participants are not poisoned by this invocation's failure.
- The completion-bearing call reports launch, public reconstruction, read-only execution, and asynchronous errors even
  when the public output structure is empty.
- Once execution crosses the irreversibility boundary, dropping its completion handle does not cancel or roll back the
  mutation.
- A holder never exposes potentially donated stale storage as its current state.
- State installation for a Phase 9 multi-reference call is logically atomic: all mutated holders are installed after
  completion, all are restored before handoff, or all are poisoned after an unrecoverable post-handoff failure.
- Calls involving the same holder serialize under the ordered guards in Phase 9; read-only calls do not overlap.
  Independent holders may execute concurrently.
- Phase 10 replaces those long-held guards with pending states and generation-safe cumulative dependency/error chains.
  If call B consumes call A's pending result and A later fails, B cannot overwrite or hide that failure; the holder
  remains poisoned. The backend completion callback owns only its type-erased token; typed holder state reconciles the
  matching generation lazily on the next holder access, so no callback captures a non-`'static` array holder and an
  older completion can never mutate a newer generation. Read-only calls may overlap only after the
  holder tracks every outstanding read lease and mutations wait or dependency-chain those leases.

In the Phase 10 asynchronous protocol, do not hold a host mutex for device execution duration. Install pending
generation/event state and read leases, then let later accesses await or dependency-chain them. Phase 9 deliberately
holds ordered holder guards through synchronous device completion as its simpler transitional contract.

### 11.4 Compilation identity and persistence

Reference ABI metadata participates in:

- `XlaLoweredProgram`;
- `XlaCompiledProgram`;
- compilation and specialization keys;
- persistent executable schema/versioning;
- deserialize-time validation;
- executable replacement/profile compatibility;
- diagnostics and byte/arity accounting.

Persist canonical logical state slots and the `public_output_count` split, never runtime holder IDs; physical
mappings and referent types derive from the persisted signature and input types. On load, validate slot order,
index ranges, alias injectivity, public/hidden output counts, physical compatibility, and signature arity before an
executable can be invoked.

### 11.5 XLA support progression

1. Local fixed-shape references, which disappear before XLA lowering.
2. One fixed-shape unsharded external reference, synchronous protocol, with copy-protection fallback.
3. Multiple unique external references (synchronous).
4. Captured references (synchronous).
5. Persistent executable round trips and replacement compatibility.
6. Asynchronous sequencing: pending states, read leases, and overlapping-call semantics.
7. Sharded references with atomic whole-shard-set holder updates.
8. Finite replicated bounded-dynamic references for read-only state; mutation remains rejected until physical alias
   compatibility is proven.
9. A deliberate zero-space-reference policy.
10. Multi-device execution on fully addressable single-process meshes, with explicit multi-host rejection.

The resulting ABI admits the supported classes above only after their runtime checks pass. Zero-space, non-device,
unbounded or nonreplicated dynamic, input-bucketed, foreign/non-addressable-mesh, and multi-host state remain explicit
prelaunch rejections.

## 12. Pallas-ready preserved-reference path

The future kernel path must consume the same logical reference abstraction and choose preservation instead of
discharge inside an explicitly validated kernel region.

### Shared core contract

- `ReferenceType<ArrayType>` and `ArrayIrValue::Reference(ArrayReference<A>)` remain unchanged.
- Root identity, access modes, lifetime checks, and view composition come from the same `ReferenceAnalysis`.
- Ordinary and kernel lowering share read/write/swap/accumulate semantics.
- `ArrayType` remains the canonical source of element, shape, layout, sharding, and memory information.
- The canonical `Memory` vocabulary remains the source of physical array placement. The experimental kernel boundary
  uses `KernelAddressSpace` only as target-eligibility metadata for how an operand or future scratch allocation may be
  exposed; it is not stored in a reference and does not introduce a parallel reference-memory model.

### Kernel-owned concepts

The following do not belong in `ReferenceType<T>`:

- grid shape and program IDs;
- block specifications and launch dimensions;
- scheduling and software-pipeline metadata;
- barriers, semaphores, and asynchronous-copy topology;
- backend-specific address-space encodings.

They belong to a future higher-order kernel-call operation and its attached kernel region.

### Kernel roadmap

1. The experimental Phase 12 kernel-call boundary defines read-only, write-only, read/write, and scratch parameter
   contracts while keeping its outer ABI array-based; scratch bindings remain explicitly unsupported.
2. The Phase 12 validator preserves references only inside a validated standalone kernel body and ordinary XLA
   rejects that same unresolved body.
3. Add scoped uninitialized scratch allocation and non-escape validation. **Access-mode revisit trigger:** when
   uninitialized allocation (`empty_reference`-style scratch) lands, read-before-initialization becomes an error
   worth catching, and a `Write`-classified used-result `swap` reads uninitialized memory — at that point either a
   `ReadWrite` mode returns to `ReferenceAccessMode` or the definedness check becomes operation-aware
   (result-liveness-based). Until then `Write`-for-`swap` is safe because every reference is initialized at
   allocation. A result-less `free_reference` needs no new mode: it is `Consume` without a result.
4. Complete view metadata with offsets, strides, alignment, layout, address space, and access mode.
5. Distinguish ordered accumulation from atomic/commutative operations.
6. Add target-neutral synchronization operations.
7. Lower preserved accesses to Mosaic/Triton/other target memory operations.
8. Keep outer kernel-call inputs/outputs array-based, distinguishing two alias layers that must never be conflated:
   operation-local kernel/custom-call operand-result alias metadata (the `OutputOperandAlias` family) for kernel
   buffer reuse, and executable-entry `tf.aliasing_output`, which is reserved for external Ryft reference mutation.
   Kernel-internal preserved references never create `XlaReferenceStateSignature` slots or entry aliases.

The preserved-reference contract proves that the ordinary reference design contains no XLA-functionalization-specific
field or assumption that would force replacement by a future kernel layer. A production kernel compiler, launch model,
and scheduler remain separate work.

## 13. Relationship to future first-class Lists

A future `List` can also become an `ArrayIrType`/`ArrayIrValue` member. References and Lists therefore share:

- the composite member/projection machinery;
- region-aware replay capable of changing representation and arity;
- transform legality gates;
- backend-specific physicalization.

They do not share semantics. A reference is an identity-bearing effect capability over array storage with one invariant
declared type; a List is a persistent variable-cardinality computational value. Do not model a logical List as
`Reference<List<T>>` or make List operations stateful.

The List design should continue to reuse `Size` for logical length and derive packed capacity only during lowering;
it should not introduce a parallel public `ListCapacity`. That work is outside this reference plan, but reference
implementation choices must not close the `ArrayIrType::List` path or make the replay infrastructure
reference-specific when a generic boundary-widening primitive is genuinely reusable.

## 14. Detailed implementation plan

### Phase 0: Freeze contracts and prototype the seam

- [x] Confirm names for `ReferenceType<T>`, `Reference<V>`, `ReferenceId`, and the five whole-array operations
      (`swap` is the sole replacement primitive; `write`/`set` are binding-level sugar). Weigh the existing
      collisions when confirming: `TypeIdentityPosition::Reference` (`programs/identities.rs:44`) and
      `CaptureReference` (`captures.rs:96`) already use the word for unrelated concepts.
- [x] Prototype `tf.aliasing_output` emission on a toy module and confirm XLA accepts and honors it. No in-repo code
      currently emits this attribute, so attribute-value construction and XLA acceptance are the unproven halves of
      the §15 "no broad initial changes" claim. Merging is not at risk: `TypeAndAttributes.attributes` is a
      per-argument `HashMap` with multi-attribute round-trip tests, and sharding already attaches at the
      entry-lowering site to extend (`sdy.sharding` in physical index space).
- [x] Record the supported and unsupported matrix from this document in public-facing design documentation.
- [x] Prototype one `Reference` type projection, one `new_reference`, and one `read` through `ArrayIrOperation` without
      migrating unrelated operations.
- [x] Prototype the operation-level reference semantic descriptor against allocation, read, and an aliasing view.
- [x] Verify (already answered affirmatively by shipped code) that the composite operation derive expresses native
      mixed reference operations: unmarked composite-native `ArrayIrOperation` variants already ship, with
      `DimensionSize` (`operations/dimensions/dimension_size.rs:66-92`, including its `ProjectedValue` receiver form)
      as the template. Composite-native variants need no `#[ryft(members(...))]` change and no projected
      `ReferenceOperation` family.
- [x] Decide whether reference-free discharge is idempotent or explicitly one-shot.
- [x] Specify error types for lifecycle, root resolution, transform rejection, discharge, and backend ABI failures.

**Exit criterion:** the minimum type/operation/access contract is demonstrated without a parallel type universe or a
broad trait-solver migration.

### Phase 1: Add the array IR reference member

- [x] Add `crates/ryft-core/src/programs/references.rs` with generic `ReferenceType<T: Type>` and its
      `ReferenceTypeRefinements<T>` adapter over `T::Refinements` (a bare `()` loses cross-signature identity facts,
      §5.1), with exact-equality compatibility and referent-delegating refinement, re-exported through the programs
      facade and crate root. Note this is the first generic `Type` implementor in the tree; the nearest existing
      generic wrapper is the value-side `CaptureReference<T: Type>` (`captures.rs:96-184`), whose blanket
      `ValueProjection` shape is the pattern to follow.
- [x] Add `ArrayIrType::Reference` and checked `From`/`TryFrom` projections.
- [x] Extend display, identity traversal/renaming, compatibility, refinement, scalar/complex classification, and
      `ArrayIrTypeRefinements`. Sizing notes: `derive_identity_renaming` grows from a 2x2 to a 3x3 member match
      (`arrays/types/ir.rs:124-146`), and the refinement change is a one-arm addition to the single dispatch site
      `visit_dynamic_to_static_refinements` (`ir.rs:230-252`).
- [x] Route reference/referent dimension refinements through the existing shared `ArrayTypeRefinements` state.
- [x] Add generic `Reference<V: Value>` in the same core references module, with `Arc`-pointer identity mirroring
      `DimensionVariable` (`arrays/types/dimensions.rs:241-255`) and `ReferenceId` derived from that pointer.
- [x] Implement `Typed<Type = ReferenceType<V::Type>>`, identity-based traits, a bare `Parameter` leaf (marker trait
      only — nothing to suppress), and the initial holder access primitives without adding a standalone `Value`
      implementation. Freeze-driven invalidation remains Phase 3 work so Phase 1 does not add a temporary lifecycle
      state with no consuming operation.
- [x] Add `ArrayIrValue::Reference(ArrayReference<A>)` and
      `ValueProjection<ReferenceType<ArrayType>, Projected = ArrayReference<A>>`.
- [x] Implement identity-based equality/hash with deterministic type-based display (runtime identity stays on
      `Debug`, per the `Value` rendering contract) and alias-preserving internal clone behavior.
- [x] Update exports and documentation that currently describe Array IR as containing only arrays and dimensions.
- [x] Audit every exhaustive type/value match in core, XLA, tests, and derives. Each site must intentionally support,
      reject, or remain unreachable after a verified pipeline boundary. Measured magnitude: ~40 files, with
      `arrays/operations/control_flow.rs` (75 matches), `arrays/batching.rs` (37), `ryft-xla/.../ops.rs` (29),
      `arrays/operations/constants.rs` (23), and `ryft-xla/.../shard_map.rs` (22) as the top sites.
- [x] Handle `XlaConstant` explicitly: its `Captured(CaptureReference<ArrayIrType>)` variant can retain composite
      reference-typed capture metadata, while concrete `ArrayReference<Array>` holders remain excluded from
      literal/serialized constants. Do not add a `ValueProjection<ReferenceType<ArrayType>>` for `XlaConstant` until
      capture lifting consumes that projected form. By contrast, the tracer projections (`Tracer`, `PartialTracer`,
      `DifferentiationTracer`, and `CaptureReference`) are blanket implementations keyed on the seam-1 `TryFrom` and
      fall out for free.
- [x] Record that `BatchingPolicyProjection` is compile-time absent for references (it is implemented only for
      `ArrayType` and `DimensionType`), which prevents reference-specific operation rules from projecting through a
      `BatchingTracer`. Opaque replicated carriers do not use that projection, so every batching entry, structural
      constant replay, and output-materialization boundary also uses the checked policy path and rejects references.
- [x] Add tests for cross-kind failures, projection, dynamic referent identity refinement, aliasing clones, poisoned
      holder access, a dimension variable shared across array, dimension, and reference leaves, dynamic-to-dynamic
      identity-renaming derivation through references (declared `ref<f32[n]>` against actual `ref<f32[m]>` yields
      `n -> m`), repeated identities, mixed array/reference occurrences of one identity, and eager non-identity rename
      rejection. Alias-family invalidation tests land with the first real consuming `freeze` operation in Phase 3.

**Exit criterion:** references can be stored, typed, projected, and diagnosed without changing ordinary
array/dimension behavior or being accepted by numeric operations. Phase 8 subsequently added exact bidirectional
identity reconstruction for renamed eager handles (§5.2).

### Phase 2: Add effects, reference semantics, and validation

- [x] Add `Effect::OrderedState`: extend `Effect::ALL`, `Effect::bit`, and `Effect::is_ordered`, plus the exhaustive
      `EffectTokens::get`/`set` matches in XLA lowering.
      `OrderedState` gets no token slot: its `EffectTokens` arm is an error made unreachable by the discharge
      verifier, never a new token chain. No `Display` exists for effects, so there is no rendering to update. Add
      tests.
- [x] Add the operation-level reference semantics contract: mutually exclusive output classification
      (`NewRoot` | `Alias`) plus input accesses (`Read`/`Write`/`Accumulate`/`Consume`) in operation-local index space.
      `Alias` records identity versus view provenance; exact view transforms remain arrays-owned and are validated
      through `ArrayReferenceOperation`.
- [x] Complete the whole-array reference operation set as native `ArrayIrOperation` variants: `new_reference`, `read`,
      `swap`, additive update, and `freeze`.
- [x] Mirror the complete operation set in `XlaOperation` so XLA staging can carry it before discharge.
- [x] Give every currently implemented unresolved state access the coarse ordered-state effect; apply the same rule to
      each remaining operation as it lands.
- [x] Implement `ReferenceAnalysis` over entry inputs, captures, allocations, aliases, and nested regions.
- [x] Implement the dedicated static validator.
- [x] Keep invocation-time holder validation separate from static analysis and defer its implementation to Phase 9,
      where external holders first enter the runtime boundary.
- [x] Verify (existing behavior, regression tests only) that simplification retains effectful instructions with
      unused outputs in program order (`programs/programs.rs:639-700`, `:1204-1220`) and that region ops need no
      effect overrides because seal-time folds already aggregate nested-region effects (`programs/regions.rs:413-448`,
      `Operation::effects` defaults to `PURE` at `operations.rs:498-500`).
- [x] Verify (existing behavior, regression tests only) that rematerialization force-saves residual roots reaching
      non-pure instructions and hard-errors on replaying them (`tracing_v2/rematerialization.rs:1524-1533`,
      `:1598-1628`, `:1726-1735`).
- [x] Implement the partial-evaluation gate — this one is genuinely new work, not verification: the default
      `fold_or_residualize` contract executes an all-known effectful operation on the known side and imports a
      mixed/unknown operation into the residual program unchanged. Either branch can violate the unresolved-state
      contract by executing hidden state early or preserving it past the transform boundary. The "never execute, fold,
      residualize, or split an unresolved state chain" rule must therefore be enforced before the knownness branch;
      the existing per-region-operation purity gates (`condition.rs:301-308`, `scan.rs:1090-1103`, `while.rs:501`)
      are not sufficient. Independently, pure reference passthroughs and unused reference boundaries contain no
      stateful instruction, so partial evaluation must recursively reject reference-bearing types at root and inline
      replay, operation-input and attached-region sinks, and finalization before it seeds or emits residual state.
- [x] Add precise diagnostics for every second-class, alias, scope, freeze, root, and unsupported operation violation.

**Exit criterion:** every reference operation has both conservative effect visibility and a precise canonical root;
invalid programs fail before any mutation or replay.

### Phase 3: Implement eager and staged local references

- [x] Implement eager create, read, write, swap, additive update, and freeze.
- [x] Enforce the exact replacement/update storage rules on every write, swap, and additive update; do not use broad
      `ArrayType::is_compatible_with` as the mutation rule.
- [x] Make freeze invalidate the complete alias family.
- [x] Expose array-to-reference creation and reference capabilities through the composite parent-binding API.
- [x] Stage operations with exact inferred types, source locations, rendering, effects, and access semantics.
- [x] Support local references in straight-line programs.
- [x] Run complete reference analysis before eager program replay and reject external reference roots until the
      external-holder runtime protocol lands, so invalid or unsupported programs cannot mutate state before failing.
- [x] Add eager/staged equivalence tests and all lifecycle/type errors.
- [x] Verify that a retained read snapshot is unchanged by every later write/update path.
- [x] Verify that the retained initializer and two distinct roots initialized from storage-sharing values remain
      unchanged/independent under later mutation.
- [x] Test explicit freeze, implicit scope-exit discard, a never-frozen local root, nested-region local discard, and
      invalid branch/loop lifecycle paths.
- [x] Test that directly eager references are never implicitly invalidated: they remain valid until explicit `freeze`
      or last handle drop, with implicit scope-exit discard applying only to staged/interpreted region execution
      (§5.4).

**Exit criterion:** the whole-array reference language has one observable meaning in eager and staged execution.

### Phase 4: Implement straight-line discharge

- [x] Add a core discharge module and result metadata types.
- [x] Integrate discharge after `ClosedProgram::to_program_with_lifted_captures` and return the canonical
      capture/public-holder binding recipe without rewriting concrete capture tables.
- [x] Validate the complete program before constructing output.
- [x] Consume the validated `ReferenceAnalysis` artifact as the discharge input contract; do not re-resolve roots,
      aliases, or accesses inside the rewrite.
- [x] Resolve reference-typed `CaptureReference`s inside attached regions through the final Phase 5 capture-aware
      analysis/discharge route; do not add a temporary reject-then-remove validator.
- [x] Track one immutable current array per root.
- [x] Rewrite each whole-array operation according to the state-passing semantics.
- [x] Eliminate local create/freeze state and preserve mutated external state as hidden outputs.
- [x] Preserve non-state effects, operation payloads, identities, layout, sharding, and memory. Instructions currently
      carry no source-location field, so discharge must not invent one.
- [x] Verify that successful output contains no reference artifacts or ordered-state effect.
- [x] Verify the existing distinct-name default `Operation::render` output for reference operations, and give the
      discharge metadata — which is not an operation — deterministic `Debug`, serialization, and equality; add
      determinism tests. Renderings back the debug-assertions transform-cache determinism recheck
      (`programs/transforms.rs:591-618`) and rendered-program test assertions — they are not production cache keys.
- [x] Add exhaustive generated tests over short straight-line state programs against eager and hand-written oracles.

**Exit criterion:** straight-line local and external reference programs produce a deterministic reference-free core
program plus complete logical state metadata.

### Phase 5: Extend discharge through regions and calls

- [x] Thread canonical state through condition branches and joins.
- [x] Thread body-mutated state through while carries.
- [x] Allow while conditions to read current entering state and reject writes, accumulations, or consumption of that
      entering state; condition-local nonescaping state remains legal.
- [x] Thread scan-mutated state as carries, separate from per-step values.
- [x] Rewrite nested calls and substitute callee root mappings into callers.
- [x] Derive every nested-region and callee root substitution from the same `ReferenceAnalysis` summaries used by
      validation; never re-resolve locally.
- [x] Resolve attached-region reference captures against their lifted entry or nested-call lexical scope in
      `ReferenceAnalysis`, and have discharge add the explicit immutable array state inputs/outputs required by
      `condition`/`while`/`scan`/call boundaries. Keep analysis and discharge paired on the exact same lifted source
      arena (§8.1).
- [x] Handle captures and local region allocations without escape.
- [x] Preserve zero-iteration and untaken-branch state exactly.
- [x] Test one and multiple roots, different branch write sets/counts, nested regions, nested calls, and invalid
      escapes.
- [x] Add explicit small-control-flow equivalence cases covering both branches, zero and many loop iterations, and
      zero and nonzero scans.

**Exit criterion:** every supported higher-order reference program has an equivalent reference-free array program, and
all unsupported ownership/control-flow patterns fail before replay.

### Phase 6: Integrate generic transforms safely

- [x] Route staged local-reference simplification through the documented pre/post-discharge behavior.
- [x] Route externally pure local-reference partial evaluation through discharge and reject every remaining
      reference-bearing case. Do not use the generic default rule or claim whole-chain residualization in the MVP.
- [x] Add trace -> validate -> discharge -> replay support for forward and reverse AD entry points.
- [x] Add the corresponding route for direct/eager differentiation APIs or reject them explicitly until it exists.
- [x] Route local-reference batching through discharge; reject external/mapped/shared reference batching.
- [x] Route rematerialization through discharge; reject externally stateful rematerialization.
- [x] Reject references in custom derivative/rule regions.
- [x] Add guards proving no reference reaches generic AD representational rules. For batching, the projection is
      already compile-time absent (`BatchingPolicyProjection` covers only `ArrayType` and `DimensionType`,
      `arrays/batching.rs:2429-2493`), but opaque replicated carriers bypass member projection; checked batching entry,
      replay, and output boundaries remain required for prevention as well as error quality.
- [x] Test nested transform orderings: discharge/JVP/transpose, discharge/batch, discharge/remat, and transforms around
      condition/while/scan.

**Exit criterion:** every public transform has a documented successful path or targeted rejection; no reference case
succeeds by accidental structural or zero-space treatment.

Known limitation: operations without a dedicated discharge rule (`shard_map`, rematerialization, linear-call, and
custom-derivative carriers) conservatively reject reference state anywhere in their attached-region closures, even
state that is allocated, mutated, and consumed entirely inside the region. Supporting region-local references there
requires per-family rules and is deferred.

### Phase 7: Compile local references through XLA

- [x] Invoke core validation/discharge before the current array-only XLA boundary.
- [x] Carry discharge metadata into lowering even when no external states exist.
- [x] Add an explicit verifier rejecting surviving references before StableHLO construction.
- [x] Keep public array-only JIT input/output APIs unchanged.
- [x] Test static fixed-shape local refs through straight-line and control-flow programs.
- [x] Snapshot reference-free StableHLO and compare execution with eager/discharged core interpretation.

**Exit criterion:** local reference programs compile and run through ordinary XLA without adding a StableHLO reference
representation or changing the executable ABI.

### Phase 8: Add indexed views and transformations

Views land before the external runtime because they are the bread-and-butter programming model (accumulate-into-slice
and the scan patterns references exist to replace), they are pure core analysis/discharge work with no concurrency
risk, and they exercise the view-to-update discharge path that later phases reuse. In the initial slice, only root
handles cross attached-region boundaries; a derived view is recreated from the carried root inside each region. This
keeps region state signatures root-based while supporting loop/scan update patterns without a second view-carry ABI.

- [x] Add a handle-local identity/view mapping and the value-reconstruction contract needed when values cross between
      that handle type and the root-shared stored value (§5.2). The holder state and handle-local type are already
      separate; the missing mapping lets renamed or projected handles retain their root without misrepresenting the
      stored value's metadata.
- [x] Define the composable arrays-owned `ArrayReferenceView`; generic alias semantics records `kind: View`, while
      `ArrayReferenceOperation` supplies the exact transform to analysis without an ownership inversion.
- [x] Add indexing/slicing views and bounds/type validation.
- [x] Lower reads/writes/swaps/additive updates through canonical array slicing/reshape/update-slice operations.
- [x] Preserve base-root identity and composed access mappings.
- [x] Defer reshape/transpose/bitcast and non-unit-stride slices until each has explicit layout and inverse-update
      proofs; do not accept them through the Phase 8 static index/unit-stride slice surface.
- [x] Add base/view mutual-observation, composed-view, overlap, invalidation, and discharge equivalence tests.

**Exit criterion:** views provide enough backend-neutral address semantics for ordinary discharge and future kernel
lowering without creating independent resources.

### Phase 9: Add synchronous external and captured XLA references

The synchronous contract: every externally stateful call acquires per-holder guards in stable `ReferenceId` order,
holds them through device completion, then installs all hidden final states (or poisons all mutated holders) before
releasing and returning. Awaiting before returning does not serialize concurrent host threads; the held guards do.
No pending states, generations, or read leases exist yet (§11.3 staging).

- [x] Define and implement the eager/XLA holder state machine with ready, poisoned, and frozen/invalid states;
      pending states are deferred to Phase 10.
- [x] Hold per-holder guards in stable `ReferenceId` order for the entire execution, through device completion and
      state installation; do not rely on await-before-return for serialization. Holding host guards across device
      execution is the accepted Phase 9 limitation that Phase 10 removes.
- [x] Extend the executable-signature metadata with logical reference-state slots only; resolve physical indices and
      referent types through the existing `input_mapping`/`output_mapping`/`input_types` accessors at use time
      (§11.2), never storing them.
- [x] Separate public and executable output signatures end to end (§11.2): executable logical outputs are the public
      prefix followed by hidden state outputs, split by a validated `public_output_count` carried through lowering,
      execution, persistence, and replacement. Scope `validate_output_types` and user `out_shardings` to the public
      prefix, and give each hidden state output its paired input's effective sharding.
- [x] Record `ReadOnly` versus `Mutated` disposition; reject reference slots whose logical input is erased
      (`input_mapping` absent) in this phase.
- [x] Handle retained-JIT dispatch keys: `XlaDispatchKeyKind::Exact` hashes `ArrayIrType`s structurally and is
      holder-free by construction, but `BucketedDispatchSignature` alpha-normalizes only `ArrayType`s — reference
      slots must take the exact path or be explicitly rejected from bucketing.
- [x] Derive physical indices only after zero-space and dynamic-extent mappings are complete.
- [x] Emit and verify `tf.aliasing_output` on the correct entry argument.
- [x] Implement hidden final-state result splitting and holder installation; keep hidden results out of public
      reconstruction.
- [x] Construct donation flags in logical ABI order: ordinary capture/dimension/read-only reference `false`, ordinary
      public array from the user flag, and mutated reference internally `true` with uniqueness downgrade. Extend the
      existing flatten-then-project construction — the recipe already exists for ordinary inputs; do not add a
      parallel one.
- [x] Mark the PJRT handoff irreversibility boundary; complete every fallible preflight before extraction so
      pre-handoff failures leave holders ready, and await completion then install-or-poison after handoff.
- [x] Define pre-handoff, execute-call, and post-submission public-reconstruction failure behavior.
- [x] Introduce the stateful call surface as an opt-in `StatefulCompilationDomain` capability: `call_statefully`
      (blocking convenience) and `call_statefully_async` returning the completion-bearing wrapper — already-completed
      in this phase, so Phase 10 changes the implementation rather than the public shape. Share the executable
      specialization/dispatcher pipeline with pure calls, make ordinary calls reject executables with non-empty
      external-state metadata, and never lose execution errors on zero-output calls.
- [x] Add internal mutated-reference donation while preserving copy-protection fallback; never donate read-only slots.
- [x] Start with one static, unsharded, device-memory external reference; then multiple unique holders with stable
      lock order and all-installed-or-all-poisoned installation; then captured references with the
      reference-specific internal donation policy.
- [x] Add explicit external reference arguments through a heterogeneous boundary without breaking array-only APIs.
- [x] Reject public/capture duplicate identities before state extraction.
- [x] Add state ABI metadata to lowering, compiled programs, cache identity, persistence, and executable replacement;
      bump and validate the persistent executable schema as the complete V6 migration: `XlaPersistentKeyV6`,
      `XlaPersistentExecutableMetadataV6`, schema version 6, and magic `RYFTXLA6`.
- [x] Add capture, cache round-trip, corruption, replacement, and synchronous failure tests.

**Exit criterion:** consecutive compiled calls observe mutation under a fully synchronous state protocol; retained
snapshots remain valid; failures leave every holder restored or explicitly poisoned; semantics remain correct when
physical alias reuse does not occur; persisted/replaced executables cannot carry mismatched state ABIs.

### Phase 10: Add the asynchronous external runtime protocol

- [x] Add pending generation/event holder states: reserve generations at submission time under the held guards
      (§11.3 step 8) and replace reservations with pending final values after result construction (§11.3 step 10),
      poisoning all mutated holders when construction or validation fails.
- [x] Track read-only execution leases, published atomically with the mutated-holder generation reservations while
      the ordered guards are still held — immediately after successful submission and before any fallible result
      processing (§11.3 step 8) — and require later mutations to wait or dependency-chain them before donation.
- [x] Implement the copy-protected immediate-execute-error-without-fence policy: retain mutated holder snapshots
      through submission so every holder remains ready when PJRT returns no fence, publishing no synthetic read lease;
      after successful submission, asynchronous failures poison the complete cumulative mutation chain (§11.3).
- [x] Use generation-safe cumulative dependency/error state so a failure in an earlier pending mutation cannot be
      hidden by a later chained call. Reconcile typed holder generations lazily on holder access; backend callbacks
      update only the type-erased completion token and never capture a non-`'static` holder.
- [x] Serialize conflicting same-holder calls while allowing safe read overlap and independent-holder concurrency.
- [x] Make multi-holder pending installation logically atomic on the same submitted execution.
- [x] Define asynchronous-execution failure poisoning and dropped-completion semantics; dropping a completion handle
      after the irreversibility boundary neither cancels nor rolls back the mutation.
- [x] Introduce the type-erased completion/dependency token in `ryft-core` (§5.2) and implement it in `ryft-xla` over
      `ExecutionFence`; pending holder states and read leases store the token directly, with no backend side map
      keyed by `ReferenceId`.
- [x] Do not hold a host mutex for device execution duration; later accesses await or dependency-chain pending state.
- [x] Add concurrency, overlap, asynchronous failure, chained-failure, and stale-callback/generation tests.

**Exit criterion:** overlapping calls have defined sequencing and failure semantics; asynchronous failures poison
exactly the involved mutated holders; no stale callback or chained call can hide or overwrite an earlier failure.

### Phase 11: Expand XLA shape and distribution support

- [x] Admit static device-memory references with replicated or sharded distribution only when the logical sharding
      uses the selected compilation mesh, mutated input/final-state shardings are identical, and the runtime holder's
      physical `DeviceMesh` exactly matches the compiled device ordering. Update complete shard sets atomically.
- [x] Add finite bounded-dynamic read-only references with replicated sharding. Reject bounded-dynamic mutation before
      lowering because CPU backend execution does not preserve the required aliased runtime extent; retain the exact
      low-level alias rejection until backend compatibility is proven.
- [x] Reject zero-space and non-device-memory external state before lowering because each holder slot requires an
      executable device buffer.
- [x] Reject input-bound bucketing whenever discharged external state is present, including capture-only reference
      state that is absent from the public dispatch signature.
- [x] Admit fully addressable, single-process multi-device meshes. Reject foreign logical/physical meshes,
      non-addressable devices, and multi-host meshes before launch because holder identity and poisoning are
      process-local.
- [x] Add each class independently with lowering and runtime conformance tests.

**Exit criterion:** every admitted shape/sharding/memory class has a documented ABI, and unsupported classes fail during
validation/lowering rather than after launch.

### Phase 12: Establish the preserved-reference kernel contract

- [x] Add a mock/kernel eligibility validator using the same roots, views, and access summaries.
- [x] Define a higher-order kernel-call region boundary and operand access modes.
- [x] Prove ordinary XLA rejects preserved refs outside that boundary.
- [x] Define scratch/non-escape eligibility, address spaces, atomics, synchronization, and view alignment as separate
      kernel contracts, rejecting scratch bindings until uninitialized allocation semantics exist.
- [x] Lower `swap` by result liveness — exchange when the old value is live, plain store when it is provably dead
      (§5.3, §6.2) — so write-only and scratch operands never require readable previous contents.
- [x] Lower one conformance program both through ordinary discharge and through a preserved mock/Mosaic path.
- [x] Keep all grid/launch/backend concepts outside the core reference type.

**Exit criterion:** a future Pallas layer can preserve and lower the existing logical reference IR without replacing
the type, operation, identity, effect, or view model.

### Phase 13: Stabilize APIs and documentation

- [x] Document local purity versus external impurity.
- [x] Document second-class restrictions, freeze/invalidation, snapshot, sequencing, and failure semantics.
- [x] Document transform support and precise unsupported combinations.
- [x] Document XLA hidden-state and may-alias behavior without promising physical in-place reuse.
- [x] Add end-to-end local, external, captured, control-flow, AD, batching, and view examples for the supported subset.
- [x] Remove temporary scaffolding and compatibility layers; every such layer must have been marked at creation with a
      `TODO(eaplatanios)` naming the phase that deletes it, and this phase verifies by search that none remain.
- [x] Reassess which APIs should remain experimental until external AD and kernel semantics mature.

**Exit criterion:** the supported contract is comprehensible without reading implementation code and does not imply
support for deferred aliases or transforms, bounded-dynamic mutation, unbounded or nonreplicated dynamic state, or
kernel operations beyond the explicitly validated preserved-reference boundary.

## 15. Likely change surface

### `ryft-core`

- `src/programs/references.rs`: generic `ReferenceType<T>` with `ReferenceTypeRefinements<T>`, `Reference<V>` with
  `Arc`-pointer identity, and the derived `ReferenceId` handle.
- `src/compilation/function.rs` and `src/compilation/contexts.rs`: the opt-in `StatefulCompilationDomain` capability
  with `call_statefully`/`call_statefully_async`, sharing the specialization/dispatcher pipeline with the unchanged
  pure output-only call (§10 Stage C); the type-erased completion/dependency token lives beside `Reference<V>` in the
  core references module (§5.2).
- `src/arrays/types/ir.rs`: third member, projection, identities, refinements, classifications.
- `src/arrays/ir.rs`: third value member and `ValueProjection`.
- `src/arrays/operations/mod.rs`: mixed reference operation variants and capability membership.
- `src/arrays/operations/references.rs`: eager composite execution.
- `src/operations/references.rs`: public operations, inference, capabilities, rendering, effect/access semantics.
- `src/programs/effects.rs`: `OrderedState`.
- `src/programs/operations.rs` or a focused reference module: reusable semantic-access contract if it is genuinely
  operation-family-wide.
- `src/arrays/reference_analysis.rs`: roots, aliases, scopes, accesses, validation.
- `src/programs/values.rs`, `src/interpretation.rs`, `src/contexts.rs`, and `src/programs/regions.rs`: the generic
  value-family preflight hook, checked eager root/direct-bind replay, explicit prevalidated replay provenance, shared
  closure traversal, and canonical region sealing invariants needed to guarantee validation before mutation.
- `src/arrays/reference_discharge.rs`: region-aware replay and logical external-state metadata.
- `src/partial.rs`, `src/arrays/batching.rs`, differentiation modules, and rematerialization: routing/guards.
- condition, while, and scan implementations: only generic boundary-widening hooks proven necessary by discharge.
- tracing, captures, and exports: projections/tests and public integration.
- `ryft-macros`/`ryft-macros-tests`: only if existing composite derive support cannot express the new native variants.

Prefer cohesive reference modules over distributing lifecycle logic through generic program files. Conversely, if
discharge reveals a reusable region-replay/boundary-widening primitive that also benefits future Lists, put only that
generic mechanism in the program transform layer.

### `ryft-xla`

- `src/experimental/ops.rs`: staged reference operation variants before discharge.
- `src/experimental/domains.rs`: discharge invocation, holder resolution, donation policy, execution result splitting,
  persistence, cache/replacement validation, and errors.
- `src/experimental/lowering.rs`: logical-to-physical state signature and entry alias attributes.
- `src/jit.rs`: stateful call plumbing through the retained-JIT facade (§10 Stage C).
- composite lowering: targeted error for surviving references.
- `src/arrays.rs` or a focused reference module: only narrowly scoped state extraction/install support that cannot
  remain generic over `ArrayReference<A>`.

### `ryft-mlir`, `ryft-pjrt`, and `ryft-xla-sys`

No broad initial changes are expected:

- existing function argument attributes should carry `tf.aliasing_output` — but no in-repo code emits this attribute
  today, so the Phase 0 seam prototype must prove attach-and-merge with sharding attributes before Phase 9 depends
  on it;
- existing PJRT donation and copy-protection behavior should preserve correctness;
- pinned XLA already recognizes the entry alias attribute;
- Mosaic operation wrappers are future preserved-reference targets.

Add lower-layer APIs only after a concrete gap is demonstrated.

## 16. Test matrix

Use named families rather than an uncontrolled Cartesian product.

| Area | Positive cases | Negative/safety cases |
|---|---|---|
| Type/member | reference projection; dynamic referent refinement; shared dimension identities | every cross-kind projection/refinement; numeric use of a reference |
| Primitive ops | create/read/write/swap/add; interleaved roots | bad type; dynamic mutation; frozen use |
| Views | base/view and composed/overlapping observation | implicit/escaping view; bad transform |
| Effects/liveness | unused write retained; read/write order; I/O plus state | folding, DCE, duplication, speculation, rematerialization |
| Straight-line discharge | each primitive; one/two roots; local/external; read-only/mutated | unresolved root; partial replay on validation failure |
| Condition | branch write combinations; two roots; fixed-root forwarding | inconsistent/duplicate roots; escape |
| While | condition read; body write; zero/many; fixed carry | condition write; permutation; escape |
| Scan | zero/nonzero; step output; fixed carry | sequence output; permutation; alias |
| Nested call | argument/capture; two levels; observed state | capture alias; escape; bad provenance |
| Partial evaluation | discharged local refs with known/mixed inputs | trace-time external mutation; split unresolved state chain |
| Batching | discharged local function equals per-example loop | external/mapped/captured ref; replicated write; lane conflict |
| JVP/VJP | local read/overwrite/swap/accumulate; condition/bounded while/scan; pure oracle | external ref; tangent ref; custom-rule ref; silent zero derivative; unbounded while transpose |
| Rematerialization | discharged local result matches baseline | preserved mutation recomputed or duplicated |
| XLA lowering | local; static sharded; finite dynamic reads | surviving refs; unsupported storage/topology |
| XLA runtime | calls, captures, snapshots, leases, sharded updates | duplicates; failure, stale state, multi-host |
| Persistence | V6 dynamic-read/static-sharded round trips | old/corrupt schema; alias/type/sharding/mesh mismatch |
| Pallas contract | same root/view/access summary under preservation | preserved ref accepted by ordinary XLA; kernel metadata in core type |

Add property/equivalence tests for short generated straight-line programs and small conditions/loops/scans:

```text
eager reference interpretation
    == discharged core interpretation
    == explicit immutable state-passing oracle
```

This is the strongest practical check against ordering and hidden-state-threading bugs.

## 17. Verification plan

For each implementation phase:

- [x] Inspect `git diff` and classify every changed file by the phase's declared ownership.
- [x] Run targeted tests before broad crate tests.
- [x] Search all `ArrayIrType`/`ArrayIrValue` exhaustive matches after adding the variant and inspect every remaining
      array/dimension-only assumption.
- [x] Verify no old `AtomType`, `ProgramType`, or parallel reference-universe scaffolding was introduced.
- [x] Verify source rendering distinguishes all semantics-bearing reference/view metadata. Renderings back the
      debug-assertions transform-cache determinism recheck (`transforms.rs:591-618`) and rendered-program test
      assertions; production cache keys are argument-based (`ErasedTransformArguments`), StableHLO-text-based
      (`XlaPersistentKeyV6`), and dispatch-signature-based (`XlaDispatchKey`) — reference ABI metadata participates
      in the persistent key schema, not the rendering.
- [x] Verify invalid programs fail before mutation, replay, or backend lowering.
- [x] Verify ordinary StableHLO contains no reference types or operations.
- [x] Inspect translated/optimized HLO for exact input/output alias configuration when external refs land.
- [x] Run each potentially expensive command with a 300-second timeout.

Expected command progression:

- [x] `cargo fmt --check`
- [x] `cargo test -p ryft-core --lib`
- [x] `cargo test -p ryft-macros --lib` and `cargo test -p ryft-macros-tests` when derive contracts change
- [x] `cargo test -p ryft-xla --lib`
- [x] `cargo test -p ryft --lib`
- [x] `cargo test -p ryft-mlir --lib` when function attributes or Mosaic wrappers change (not applicable: unchanged)
- [x] `cargo test -p ryft-pjrt --lib` when execution/donation APIs change
- [x] `git diff --check`

## 18. Delivery milestones and effort split

### Milestone A: core local-reference vertical slice

Phases 0-4: type/value member, operations, effects, validation, eager/staged execution, and straight-line discharge.

Estimated ownership:

- `ryft-core`: 90%+
- other crates: exhaustive-match and staging mirrors only

### Milestone B: structured local references and transforms

Phases 5-7: region/call discharge, supported local transform routes, and local XLA execution.

Estimated ownership:

- `ryft-core`: 75-85%
- `ryft-xla`: 15-25%

### Milestone C: indexed views

Phase 8: view representation, composition, validation, and view discharge through canonical indexing operations.

Estimated ownership:

- `ryft-core`: ~95% (pure analysis/discharge work; no runtime or ABI changes)

### Milestone D: external/captured references

Phases 9-10: holder lifecycle, hidden state ABI, aliases, donation, caching, and persistence under the synchronous
protocol first, then the asynchronous protocol (pending states, read leases, concurrency, and failure chains).

Estimated ownership:

- `ryft-core`: 35-45%
- `ryft-xla`: 50-60%
- `ryft-mlir`/`ryft-pjrt`/`ryft-xla-sys`: 0-10%, only for demonstrated gaps

### Milestone E: expanded XLA support

Phase 11: sharding, dynamic extents, zero-space policy, and distribution.

Ownership depends on which representation constraints surface, but remains primarily core analysis/discharge plus XLA
ABI/runtime integration.

### Milestone F: Pallas-readiness checkpoint

Phase 12 proves preservation viability. A complete Pallas layer is a separate large program with substantially more
`ryft-xla` and `ryft-mlir` ownership.

For a static fixed-shape reference MVP including local semantics, structured discharge, one external holder, aliases,
captures, and basic legality, the overall estimate remains approximately:

- `ryft-core`: 65-75%
- `ryft-xla`: 25-35%
- lower-level crates: 0-5%

## 19. Risks and mitigations

### Treating the enum member as the feature

**Risk:** reference values compile through storage/dispatch but transforms reorder or lose mutation.

**Mitigation:** gate release on effects, root analysis, validation, and discharge equivalence—not enum exhaustiveness.

### Conflating type identity and resource identity

**Risk:** runtime aliases become embedded in types, destabilizing caching/refinement and conflicting with dynamic
dimension identities.

**Mitigation:** keep referent structure in `ReferenceType<T>`; derive roots from SSA/captures/holders in dedicated
analysis.

### Overengineering the effect system

**Risk:** dynamic resource effects force a broad redesign before an MVP exists.

**Mitigation:** add one coarse ordered-state class now and a separate precise access analysis. Optimize scheduling later.

### Unsound generic transforms

**Risk:** reference handles are treated like structural dimensions or zero tangent values.

**Mitigation:** enforce the canonical pipeline and targeted guards at every transform entry point.

### Region discharge complexity

**Risk:** hidden state order or capture substitution differs across branches and nested regions.

**Mitigation:** canonical root order, analysis-before-replay, all-or-nothing construction, and three-way equivalence
tests.

### Donation mistaken for semantics

**Risk:** a shared buffer or PJRT donation downgrade breaks observable mutation.

**Mitigation:** always consume/install the hidden final-state output; treat may-alias/donation only as physical reuse.

### Async failure leaves invalid state

**Risk:** a holder loses its old value after submission but its final buffer fails.

**Mitigation:** explicit pending/poisoned states, atomic multi-holder transition, and failure tests before public API
stabilization.

### Premature Pallas specialization

**Risk:** grid/address-space/kernel concerns pollute ordinary reference types and transforms.

**Mitigation:** share roots/views/accesses only; keep kernel launch and synchronization in a separate higher-order
operation/lowering policy.

### Accidental scope expansion into Lists

**Risk:** reference implementation attempts to solve dynamic collections and loop tapes simultaneously.

**Mitigation:** share only proven composite/replay mechanisms; keep List purity, `Size`, and physicalization in a
separate plan.

## 20. Review checkpoints

Do not begin the next checkpoint until the prior one is reviewed:

1. **Semantic review:** type/value identity, second-class rules, operations, effects, and unsupported cases.
2. **Core architecture review:** `ArrayIr` integration, access analysis, validation, and straight-line discharge.
3. **Control-flow review:** canonical state ordering and region/call replay.
4. **Transform review:** every public transform has a proven route or explicit rejection.
5. **Local XLA review:** reference-free StableHLO and unchanged public ABI.
6. **External runtime review:** holder ownership, failure, concurrency, hidden results, donation, and aliases.
7. **Persistence/distribution review:** cache identity, schema validation, sharding, and dynamic extents.
8. **Pallas-readiness review:** preserved-reference eligibility without kernel concerns in core types.

At each checkpoint, ask:

- Is there a smaller semantic surface that still proves the architecture?
- Can every supported behavior be explained without relying on physical in-place reuse?
- Can every reference operand be resolved to one root?
- Can every generic transform prove that references were discharged, or reject them explicitly?
- Would the same type/operation/view contract still make sense inside a kernel region?
- Are deferred behaviors rejected at the earliest responsible boundary?

## 21. Plan review record

- [x] Re-audited the current `ArrayIrType`, `ArrayIrValue`, `ArrayIrOperation`, effect, transform, XLA lowering,
      executable-signature, donation, capture, persistence, and Mosaic seams on 2026-08-14.
- [x] Rechecked current official JAX Ref, Pallas, and OpenXLA aliasing documentation on 2026-08-14.
- [x] Replaced the obsolete parallel `AtomType<T>`/`ProgramType<T>` direction with direct Array IR integration.
- [x] Separated coarse effects from precise resource analysis.
- [x] Separated core logical discharge metadata from XLA physical ABI indices.
- [x] Added explicit lifecycle, transform, control-flow, async failure, concurrency, persistence, and Pallas-readiness
      proof obligations.
- [x] Kept first-class Lists outside implementation scope while preserving the shared Array IR/replay path.
- [x] 2026-08-16 review pass 1: made `swap` the sole replacement primitive (`write`/`set` as binding sugar); bounded
      the read-ordering cost note; made discharge consume `ReferenceAnalysis` an explicit contract; added the Phase 0
      `tf.aliasing_output` seam prototype; moved views ahead of the external runtime; split the external runtime into
      a synchronous phase and an asynchronous phase; required `TODO(eaplatanios)` markers on scaffolding.
- [x] 2026-08-16 review pass 2 (code audit): downgraded to verification the items existing machinery already provides
      (simplification retention, region effect aggregation, rematerialization non-duplication, nested-call capture
      normalization, composite-derive mixed variants, MLIR multi-attribute merging, `Parameter` leaf opacity, effect
      rendering); kept the partial-evaluation gate as genuinely new work at `fold_or_residualize`; fixed the
      nested-region `CaptureReference` lifting gap with an MVP validator rule; switched `Reference` identity to the
      `DimensionVariable` `Arc`-pointer precedent; dropped stored physical indices and referent types from both state
      metadata structs in favor of existing accessors; recorded the compile-time absence of reference batching
      projections, the `XlaConstant` and bucketed-dispatch handling gaps, and the rendering-vs-cache-key distinction.
- [x] 2026-08-16 review pass 3 (external feedback): restored `ReferenceTypeRefinements<T>` for the standalone generic
      contract (bare `()` loses cross-signature identity facts) and made reference compatibility exact equality
      (referent compatibility is broadcastability); separated public from executable output signatures with a
      validated `public_output_count` and public-prefix-only `out_shardings`; replaced the incorrect
      await-serializes-calls claim with ordered per-holder guards held through completion in Phase 9 and
      submission-time lease publication in Phase 10; scoped implicit scope-exit invalidation to program execution
      (directly eager references live until freeze or last drop); made the MVP reject non-identity renames of eager
      references and scheduled the root-shared/handle-local split before views; moved recursive attached-region
      capture lifting into Phase 5 (the temporary Phase 4 rejection was superseded by the one-shot capture-aware
      Phase 4–5 implementation recorded below); chose a separate stateful call method as the
      compilation seam and added `ryft-core::compilation` to the change surface; added the dead-old-result
      `swap`-to-write-only canonicalization for preserved kernels; folded `source_parameter_index` into
      `ReferenceSource`; corrected V5-modify wording to a new `XlaPersistentKeyV6`; declared `OrderedState` token-free
      with a verifier-proven-unreachable lowering arm; distinguished operation rendering from discharge-metadata
      determinism; removed the stale composite-variant count.
- [x] 2026-08-16 review pass 4 (external feedback): required explicit `derive_identity_renaming` delegation through
      references (the default returns an empty renaming after validating, so `ref<f32[n]>` vs `ref<f32[m]>` must
      produce `n -> m`; composite arm uses `ArrayType::extend_identity_renaming`) and removed the contradictory
      compatibility-delegation wording; restructured the async protocol so lease publication and mutated-generation
      reservation happen atomically under the held guards immediately after submission, before all fallible result
      processing, with a copy-protected policy that leaves holders ready on immediate execute errors without a fence;
      type-erased core completion/dependency token (XLA implements it over `ExecutionFence`, no side maps); fixed
      the pipeline order to capture-normalization before `ReferenceAnalysis` before discharge; renamed the stateful
      call surface (`call_statefully`/`call_statefully_async` on an opt-in `StatefulCompilationDomain`, shared
      dispatcher, ordinary calls reject stateful executables) and widened the change surface to
      `compilation/contexts.rs` and `ryft-xla/src/jit.rs`; rejected `freeze(view)` (root-only freeze); separated
      kernel operand-result aliases from executable-entry aliases; named the complete V6 persistence migration; and
      scoped Phase 1's rename exit criterion to types and staged metadata.
- [x] 2026-08-16 review pass 5 (implementation feedback on the semantics contract): simplified
      `ReferenceOperationSemantics` to mutually exclusive output classification (`NewRoot` | `Alias`) plus input
      accesses, dropping the separate root/alias collections; dropped `ReadWrite` (`swap` classifies as `Write`,
      matching JAX's `swap_p`, with `Write` documented as not asserting absence-of-read — result liveness decides,
      and dead-store elimination must consult it); renamed `Freeze` to `Consume` (a lifetime event that also covers
      a future result-less `free_reference`); deleted the premature `ReferenceViewTransform` (transform stacks attach
      to the `Alias` arm in Phase 8; the alias edge alone is the Phase 0 view proof); documented that `Alias`'s
      single `input_index` structurally encodes the one-root invariant; inverted the dead-result `swap` rule into a
      result-liveness lowering decision (exchange vs. store); added per-operation examples and the
      indices-are-not-resource-IDs warning to the rustdoc; and recorded the uninitialized-scratch access-mode revisit
      trigger under the kernel phases.
- [x] 2026-08-16 review pass 6 (working-tree code review, 24 findings fixed): forwarded `reference_semantics` through
      `Box<O>`; made the eager-bind guard union driver region effects (region-carried unresolved state now rejected,
      with a `while`-body regression test); made `Reference`'s `Display` deterministic and type-based (identity stays
      on `Debug`); moved the token-slot classification onto `Effect::has_token_slot` and guarded the shard-map
      `to_mlir_module` entry; added the `From` coherence discriminators to the batching/differentiation rejection
      blankets and switched transposition to the shared `impl_non_transposable_operation!`; made
      `ReferenceOperationSemantics::new` enforce its documented invariants (panicking on duplicate classifications or
      accesses) and return shared `Cow<'static, _>` descriptors; recursed `has_only_static_array_types` into
      reference referents; extracted the shared rename-rejection helper with an identity-renaming fast path and
      documented the import/inline limitation above; made refinements delegation borrow-based; consolidated
      member-kind diagnostics behind `kind_name` helpers while keeping declared-variant compile gates; centralized
      attached-region count validation at eager binding instead of duplicating it in interpretation rules; added the
      missing rejection/eager tests and convention fixes
      (`$crate` paths, defining-path test imports, `Display`-via-`render`, rustdoc links); and marked
      `XlaReferenceConstant` with its owning-phase TODO.
- [x] 2026-08-16 review pass 7 (external feedback on the fixes, 8 findings applied): rejected concrete reference
      constants structurally (`Value::validate_as_constant`, enforced at `ProgramBuilder::build` and, since
      pass 8, at region sealing, with the
      rendering-contract rationale recorded in §5.2); guarded custom JVP/VJP rule regions against unresolved
      state before direct interpretation (with an end-to-end rejection test); added the compilation-path
      preflight at the start of `lower_xla_program` so external and dynamically shaped references get the targeted
      discharge diagnostic before boundary projection and partitioner selection (module-entry guards retained as
      defense, tested through the real lowering path); removed the premature commutativity/atomicity promises from
      `Accumulate` (ordered additive accumulation, linear in the update; atomic/commutative semantics stay future
      work) and the generic result-liveness promise from `Write` (read-ness is operation-specific; no
      access-to-result mapping until a generic analysis needs one); moved the token-slot classification out of core
      into an exhaustive local match in XLA's `token_threaded_effects` (backend representation decisions stay out of
      `ryft-core`; exhaustiveness keeps the forcing function); corrected `ReferenceOperationSemantics::new`'s arity
      claim to name the planned reference analysis as the validator; and fixed the stale plan statements (partial
      evaluation is not made safe by the ordered class alone; display is type-based with identity on `Debug`; rename
      rejection is type-change-keyed; the pass-5 record no longer claims the removed `ArraySliceAxis` descriptor or
      completed discharge work) plus the test import grouping and the vague temporal-residual TODO.
- [x] 2026-08-16 review pass 8 (external feedback, 8 findings applied): moved constant-storage validation to region
      sealing — the one boundary every construction path crosses (builders, `Program::new`, imports, nested regions) —
      removing the builder-only scan and testing the direct `Region`/`RegionArena` path; added the central
      differentiation state guard in `DifferentiationContext::bind` before the all-zero tangent fast path (covering
      computation and dormant rule regions; operation-local guards remain defense in depth; both the normal and
      stop-gradient zero-tangent paths are tested); replaced the effects-only XLA preflights with two shared scans,
      `contains_unresolved_state` and `contains_unresolved_references`, used at the compilation preflight and both
      module-lowering entries, with pure pass-through and forwarded-capture tests; added a separate dispatch-boundary
      reference check ahead of every mesh/option constraint; aligned eager binding with
      `Program::effects` by excluding dormant rule regions at that pass (subsequently superseded by pass 9's
      artifact-wide ordinary-XLA rejection policy); relaxed
      `reference_semantics` to `Cow<'_, _>` so future payload-borrowing operations need not clone; and fixed the
      remaining documentation (state may alternatively be handled by a state-aware backend; the import/inline rename
      consequence is unreachable now that constants are sealed out; the `XlaReferenceConstant` TODO moved out of
      public rustdoc). Known accepted gap, unchanged: the central partial-evaluation gate for `fold_or_residualize`
      remains Phase 2 work, so the branch must not be described as generally transform-safe until it lands —
      higher-order all-known partial evaluation can still execute hidden state, while mixed-input evaluation can
      import it into the residual program unchanged.
- [x] 2026-08-16 review pass 9 (external feedback, 6 findings applied): added
      `RegionRef::contains_effect_in_closure` — a closure-wide scan that descends into dormant rule regions at every
      nesting depth — and used it to guard the fused `RegionRef::jvp` replay at entry (covering `Program::jvp`,
      `jvp_shared`, and linearization, whose all-zero shortcut previously staged primals unguarded) and to deepen the
      `DifferentiationContext::bind` guard (nested rule regions were previously invisible through sealed effects),
      with direct `Program::jvp` and pure-program rule-region-hidden-state tests; made `BatchingContext::lift` apply
      the constant-storability contract so lifted reference holders cannot ride through batching as replicated
      batches; unified the XLA dormant-rule policy on artifact-wide rejection (eager binding now scans rule regions
      too, matching the ordinary-XLA unresolved-artifact checks, with an eager custom-JVP rule-region test; later phases
      may relax this with an executable-region analysis); added staged-boundary reference checks before sharding
      overrides in `stage` so overridden reference boundaries get the targeted diagnostic; removed the speculative
      `XlaReferenceConstant` alias and projection until capture lifting consumes them (keeping the rename-metadata
      coverage over plain captures); and fixed the remaining stale statements (identity is equality/hash-only with
      type-based `Display`; constant-storage docs and tests now name region sealing, with direct `Program::new`
      coverage; the multi-device dispatch claim is now actually asserted). The partial-evaluation gate remains the
      acknowledged Phase 2 gap.
- [x] 2026-08-17 review pass 10 (external feedback, 6 findings applied): closed the structural-batching passthrough
      by routing both `BatchingContext::lift` and `ArrayIrBatching::batch_region` constant replay through the checked
      `BatchingPolicy::batch` constructor (replacing the semantically inappropriate constant-storage check in `lift`;
      the infallible replicated carrier is no longer an entry boundary, and its docs plus the disproven
      "projection absence is structural prevention" claims were corrected); made `contains_effect_in_closure`
      iterative with an arena-indexed visited set (shared attachment DAGs were revisited once per path — worst-case
      exponential — and deep closures risked stack overflow), pinned by a 64-level shared-diamond rule-closure test;
      centralized `validate_region_count` on `Operation` and invoked it from the differentiation, batching, and
      partial-evaluation `bind` implementations so every operation-application boundary rejects mismatched region
      attachments; reordered all three XLA discharge boundaries to check unresolved references before generic state
      so reference operations get the dedicated diagnostic (the state error remains for future non-reference state);
      added focused staging-guard tests proving reference inputs with `in_shardings` and reference outputs with
      `out_shardings` are rejected before array projection; and fixed the stale statements (the fused-JVP guard
      comment no longer claims to cover linearization, which has its own guard; the partial-evaluation gap is now
      described over both unsafe branches — executing all-known state and residualizing/importing mixed-input state
      unchanged). The central partial-evaluation gate remains the acknowledged Phase 2 gap.
- [x] 2026-08-17 review pass 11 (final Phase 0/1 hardening): centralized structural constant and operation replay
      behind one checked `ArrayIrBatching::batch_region_values` helper shared by both `batch_region` and
      `batch_program`, eliminating the duplicated implementation that previously let the whole-program path drift;
      retained the independent output-materialization rejection for manually manufactured replicated reference
      carriers; moved XLA region-count validation ahead of the nullary identity fast path and made its control-flow
      reference regression a well-formed condition/body pair; strengthened operation-local custom-JVP/VJP guards over
      complete nested rule closures; made the all-role closure scan iterative and conservative for invalid attachment
      identifiers; disambiguated composite `Display` forwarding; and corrected the remaining plan/rustdoc/test-name
      drift. Full verification passed: 1,373 core tests plus 3 ignored, the macro integration suite, XLA all-target
      compilation, and 477 XLA tests plus 5 ignored. The central `fold_or_residualize` rejection remains deliberately
      pending in Phase 2 and is not part of the completed Phase 0/1 support claim.
- [x] 2026-08-17 Phase 2 implementation/review pass 12 (superseded in part by pass 14): completed the whole-array
      operation language with `swap`, additive update, and consuming `freeze`, together with exact types,
      ordered-state effects, operation-local access semantics, Array IR/XLA staging variants, and deterministic
      pre-discharge transform/backend rejection; added the program-relative `ReferenceAnalysis` artifact with canonical
      allocation/region-input roots, validated access records, exact nested-region bindings, caller-context root
      substitution, capture/public input classification, precise instruction-scoped diagnostics, second-class and
      lifetime enforcement, dormant-rule closure rejection, and per-invocation static alias rejection; initially added
      invocation-time holder uniqueness validation without placing holder identity in types or effects, but pass 14
      removed that unused pre-runtime surface and made Phase 9 its canonical owner; closed partial evaluation before
      knownness, control-flow probing, JIT-call partitioning, and shard-map partitioning; and pinned simplification
      ordering and rematerialization non-recomputation. Phase 2 remains intentionally non-executable: eager mutation
      starts in Phase 3 and ordinary XLA continues to reject every undischarge reference artifact.
- [x] 2026-08-17 Phase 3 implementation/review pass 13: implemented the fixed-shape local eager language over one
      shared ready/frozen holder, including snapshot reads, exact swaps, ordered additive updates, consuming freeze,
      complete alias-family invalidation, and failure atomicity; exposed all five composite reference capabilities
      while preserving one `swap` IR primitive for write sugar; reused canonical operation inference before every
      eager holder mutation and retained exact holder checks as defense in depth; added transactional whole-program
      eager preflight from the Phase 2 `ReferenceAnalysis`, rejecting every external root before replay (duplicate
      external-holder validation remains deferred to the Phase 9 runtime boundary), and closed the standalone
      `RegionRef` bypass with root preflight plus a private prevalidated nested replay path; validated implicit local
      discard without a scope registry, and covered repeated `while` invocation,
      invalid reference carries, snapshots, independent storage-sharing roots, dynamic-referent rejection, and direct
      eager/staged equivalence; completed the composite capability bundle and aligned condition/while/scan contracts;
      and strengthened ordinary-XLA eager rejection to catch pure reference-typed atoms throughout attached dormant
      closures as well as ordered state. Review pass 14 below records the subsequent combined Phase 2-3 audit and the
      final verification counts.
- [x] 2026-08-17 combined Phase 2-3 implementation/review pass 14: ran three fresh independent audits focused on
      correctness, repository conventions, and removing avoidable complexity; made partial evaluation report
      unresolved state before deferred-error erasure; moved the array-specific reference analysis into the `arrays`
      owner, removed parallel root state, and added a reference-free fast path; centralized region Single Static
      Assignment validation at sealing so every safe program construction path rejects orphan or cyclic providers;
      closed differentiation's intrinsic-state zero-tangent shortcut and eager higher-order direct-bind/standalone
      region preflight bypasses with explicit prevalidated replay provenance; reused one iterative region-closure walk;
      completed fixed-shape read/freeze checks, exact diagnostics, and local-reference scan coverage; and kept
      structural batching on one shared checked replay helper instead of adding an artificial test-only operation
      universe. Also deleted the speculative pre-Phase-4 discharge surface until a consumer exists (contextual root
      resolution, source-identity binding, holder-list invocation validation, external-root referent copies, and the
      public holder identity, which returned to crate-private); reduced handle origin tracking to the single
      allocation bit the freeze check reads; collapsed the five per-rule partial-evaluation state-gate preambles into
      one replayed-instruction gate next to `Context::bind`'s; deferred diagnostic name allocation to error paths; and
      applied the remaining convention fixes (defining-path imports, error-enum and shared-fixture placement, line
      wrapping, trailing commas, and bound ordering). All three final auditor passes reported no remaining actionable
      issue. Review pass 15 below supersedes this pass's final verification counts and records the later type-only
      partial-evaluation and replay-evidence hardening.
- [x] 2026-08-18 combined Phase 2-3 independent re-audit pass 15: repeated three strict independent audits over core
      reference semantics and eager lifecycle, transform/control-flow/replay safety, and XLA/macro/convention
      integration. Closed every remaining public partial-evaluation escape route: a type-recursive
      `Type::is_reference` contract now rejects pure passthroughs, unused or mismatched reference boundaries,
      attached dormant reference regions, direct residual emission, and finalization, while a read-only root preflight
      preserves the precise first intrinsic-state operation diagnostic before replay. Replaced forgeable replay booleans
      with opaque `PrevalidatedReplay` evidence, issued that evidence only after successful eager value-family
      validation, generalized the eager hook to borrowed rooted closures, and pinned both Program and RegionRef positive
      and negative paths. Corrected `ReferenceAnalysis` to its actual rooted-closure/source-arena contract and added an
      unreachable-sibling regression. Reconciled static root validation with Phase 9's deferred external-holder identity
      validation, removed stale API/docs/imports, and retained defense-in-depth gates only at independently public or
      directly callable transform sinks. Consolidated the ordered-state and region-effect test fixtures into the shared
      crate test module (one parameterized `Effectful(Effect)` region family plus one shared `TestOrderedStateOperation`
      replacing three local copies), removed the shard-map duplicate of the centralized partial-evaluation gate test in
      favor of the interned-callee `jit_call` variant, dropped the redundant operation-level dynamic-referent assertion
      from the capability tests, and applied the remaining convention fixes (module documentation headers, imported-name
      call sites, literal wrapping, and the `Display` import in `instructions.rs`). All three final auditors reported no
      actionable code issue. Final verification passed: formatting and `git diff --check`; 1,407 core tests plus 3
      ignored; the complete macro integration and compile-fail suites; XLA all-target compilation; 481 XLA tests plus 5
      ignored; and core documentation generation with 97 pre-existing warnings, none in the reference modules.
- [x] 2026-08-16 Phase 0 implementation/review pass 5: fixed the canonical generic and Array IR reference names;
      implemented the minimum production-safe type/value/projection plus `new_reference`/`read` vertical slice;
      added the operation-local reference semantics descriptor (initially carrying an `ArraySliceAxis` view mapping,
      since simplified in the later semantics refactor to `NewRoot`/`Alias` outputs plus accesses, with view
      transforms deferred to Phase 8);
      added conservative `OrderedState` handling and deterministic transform/backend rejections; integrated references
      with the fallible differentiation-type contract (`is_zero_space == false`, undefined tangent/cotangent errors)
      without a separate program preflight or fabricated differential representation; proved `tf.aliasing_output`
      attribute coexistence and donated CPU buffer reuse without adding a production XLA ABI; decided (not yet
      implemented — discharge itself remains Phase 4 work) that reference-free discharge will be a one-shot wrapper
      contract with reference-free identity output; assigned lifecycle, analysis, transform, and backend error
      ownership; and completed independent core and XLA simplification reviews.
- [x] 2026-08-16 Phase 1 implementation/review pass 6: completed the generic `ReferenceType<T>` refinement and
      whole-signature identity-renaming contract; completed holder-free structural type identity and identity-bearing
      `Reference<V>` semantics; integrated the third `ArrayIrType`/`ArrayIrValue` member across projections,
      classification, diagnostics, captures, control-flow boundaries, and documentation; supported reference-typed
      `XlaConstant` capture metadata without admitting concrete holder literals; made member batching projection
      structurally unavailable for references through the existing `BatchingPolicyProjection` contract (later
      found insufficient on its own, since infallible replicated carriers bypass projection — refer to review
      pass 10); added exact
      mixed array/dimension/reference identity, eager rename, holder poisoning, numeric rejection, and XLA dispatch
      tests; deliberately kept freeze-driven invalidation in Phase 3 rather than adding a temporary holder state; and
      completed full core, macro, and XLA test suites plus three independent simplification reviews, including their
      final fixes for borrowed capture projection, zero construction, batched-`while` reference diagnostics, and stale
      projection/member-kind documentation.
- [x] 2026-08-19 combined Phase 4–5 implementation/review pass 16: implemented one analysis-coupled, all-or-nothing
      reference-discharge pipeline that rewrites whole-array references into immutable array SSA state and emits
      deterministic logical metadata for external roots. Extended `ReferenceAnalysis` with deterministic transitive
      summaries, exact nested-call capture scopes, structured-output provenance, fixed-point loop-carry constraints,
      and operation-owned access policy; discharge consumes those facts without re-resolving roots and threads
      canonical state through conditions, while loops, scans, calls, and recursively captured regions. Kept
      reference-free programs unchanged, preserved ordinary payloads and non-state effects, rejected malformed public
      operation-family rules before constructing output, and verified every artifact is free of references and ordered
      state. Focused coverage includes every primitive, retained snapshots, external metadata ordering, exhaustive
      short straight-line sequences, branch joins, zero/many loop iterations, zero/nonzero scans, nested calls and
      captures, shared-region specialization, fixed-point identity, malformed provenance, and transactional failures.
      Full verification passed: formatting and `git diff --check`; 1,435 core tests plus 3 ignored; the complete macro
      integration and compile-fail suites; XLA all-target compilation; 485 XLA tests plus 5 ignored; and core
      documentation generation with 95 pre-existing warnings, none in the reference modules. Three final independent
      audits of correctness, conventions/API design, and simplification reported no remaining actionable findings.
- [x] 2026-08-19 combined Phase 2–5 independent re-audit pass 17: three fresh independent audits over the Phase 4–5
      discharge work plus the committed Phase 2–3 reference modules. No exploitable correctness defect; fixes landed
      for every finding. Production: reverted an accidental rematerialization policy regression by pinning the
      intended pass-through provenance classification instead of changing the framework; consolidated scan carry
      widening onto one canonical `ScanOperation::with_added_carries` so backend rebuilds cannot silently drop future
      scan fields; validated the positional-provenance/arity alignment the condition/call rewrites rely on; routed
      `jit_call` rendering through `OperationFormatter`; replaced the wrapper summary type, derived access enum,
      logical slot, and dead binding fields/accessors with the minimal analysis/discharge metadata surface; gated the
      discharge fast path on the analysis fast path; removed the per-nesting re-verification, duplicate layout arms,
      duplicated root resolution and error constructions, and the module-local reference-type helper in favor of
      `Type::is_reference`; named discharge rules in diagnostics; and moved the discharge tests after the
      implementation. Committed-code cleanups: deleted the dead per-replayed-instruction and duplicate entry
      partial-evaluation gates, reduced the reference transform-rejection macro's partial-evaluation arm to the trait
      default, collapsed identical lowering arms, adopted one shared `reachable_region_mask` closure walk at five
      sites, consolidated the replay-driver constructors, and removed stale review markers and the dead holder-id
      `Display`. Tests: split the five-operation omnibus into per-operation tests on the official inference macro;
      pinned discharged program shapes with rendered fixtures; added eager-versus-discharged equivalence for
      condition and scan plus a hand-written immutable-loop oracle for `while` (eager replay of reference-carrying
      `while` predicates is intentionally undefined); covered the mutated-capture synthesized-carry path, ambiguous
      shared capture scopes, nested-binding lookup, the XLA discharge-rule mapping and scan-metadata preservation;
      trimmed operation enumerations of operation-agnostic guards to representatives; and merged the duplicate
      seal-time effect-fold block while keeping the ordered-state pin. A second audit round over the fixed tree found
      no correctness defect and converged on residual polish, all applied: primitive discharge rules now validate
      against the canonical core operations through one object-safe contract oracle instead of hand-copied arity and
      type patterns; `ReferenceOperationSemantics::is_empty` replaced six spelled-out predicates; the derive macro's
      eleven identical forwarding-arm generators collapsed onto one helper; the reference transform-rejection macro
      takes the whole operation family in one invocation; the XLA reference scan delegates to the core closure
      helper; `region_summary` went crate-private; owner-module tests landed for `ScanOperation::with_added_carries`
      and `reachable_region_mask`; tautological reference-free assertions, triplicated carry pins, duplicated
      custom-derivative fixtures, and the remaining omnibus rejection tests were deduplicated or split; and this
      section's earlier discharge-metadata sketch was aligned with the shipped surface. Third and fourth audit rounds
      then converged: round three surfaced five residual polish items (a doc reflow, helper placement, one remaining
      spelled-out emptiness predicate, a canonical-inference rejection case proven to kill its target mutation, and
      owner-module `is_empty` assertions), all fixed and verified; round four reported no findings. Final
      verification passed: formatting and `git diff --check`; 1,473 core tests plus 3 ignored; the complete macro
      integration and compile-fail suites; 487 XLA tests plus 5 ignored; and zero compiler warnings across all
      targets. A colleague review then surfaced two correctness gaps and three cleanups, all fixed with regressions:
      `jit_call` lowering now takes its capture prefix from the operation payload instead of scanning capture-constant
      indices across the callee arena (which conflated nested calls' independent capture namespaces), discharge
      validation now pins the exact result-region arity and positional output provenance that the higher-order replay
      zips against (plus positional inputs for `while` and the scan carry prefix), the nested-binding query became the
      keyed public `region_root_for_source` with the flat binding list crate-private, the scan overflow expectation is
      target-width portable, and the single-input eager capability checks borrow their type instead of cloning it.
      A final verification audit confirmed all six fixes and surfaced three residual polish items, also fixed: the
      linearize-primal and partial-evaluation known-side derived calls now declare their true surviving capture-prefix
      lengths (with accurate invariant comments on the tangent/residual/transpose boundaries that genuinely have
      none), the stale `linear_jit_call` doc clause is gone, and the invocation-lossy binding record went
      crate-private with its dead accessors deleted. The audit also forwarded one pre-existing out-of-scope hazard
      (fresh-root traced region bodies silently recording captures into throwaway tables) as a separate tracked task.
      A follow-up colleague round then closed the remaining capture-scope and preflight gaps: mixed partial
      evaluation preserves the original `jit_call` boundary whenever the callee closure retains a capture constant
      (partitioning does not remap absolute capture indices; guarded through the new
      `RegionRef::contains_atom_in_closure` and pinned by a preserved-boundary regression, with the split test's
      fixture made capture-free), the discharge preflight accepts the dynamic-length scan's trailing runtime-length
      operand (pinned by an eager-equivalence regression), and the fresh-root capture-rejection additions received
      their convention polish (rustdoc on the counting trace helper, a `where`-clause bound, and line-width fixes).
      Post-fix verification: 1,476 core tests plus 3 ignored, 492 XLA tests plus 5 ignored, all macro suites,
      formatting, `git diff --check`, and zero warnings.
- [x] 2026-08-20 combined Phase 6–7 implementation/review pass 18: added explicit flat Array-IR adapters that
      discharge local whole-array references before partial evaluation and partitioning, JVP and linearization,
      structural batching, and rematerialization, while rejecting public/captured external holders before generic
      transform replay. Preserved the generic engines and their direct unresolved-state rejections; added focused
      condition, bounded-while, scan, and rematerialized primal/JVP/VJP composition coverage. Integrated capture-aware
      discharge into ordinary XLA lowering before array projection, retained a borrowed single-scan path for the
      reference-free majority, independently verified rewritten artifacts, and rejected nonempty external-state
      metadata without adding Phase 9 runtime/alias machinery. Fixed lifted capture namespaces in attached regions,
      preserved capture-count diagnostic precedence, kept the public JIT array ABI unchanged, and pinned exact
      straight-line and condition StableHLO plus compiled condition/while/static-scan/dynamic-scan/nested-call/public-
      JIT behavior. Three independent audit loops covered correctness, conventions/API/testing, and architecture/
      simplicity; every finding was fixed and all final passes reported clean. Final verification passed: 1,488 core
      tests plus 3 ignored; 501 XLA tests plus 5 ignored; 20 macro integration tests including the complete trybuild
      compile-fail suite; core and XLA all-target checks; core documentation with 95 pre-existing warnings and
      warning-free XLA documentation; formatting, stale-identifier and added-line-width audits; and
      `git diff --check`.

- [x] 2026-08-20 combined Phase 2–7 independent re-audit pass 19: three fresh independent audits over the complete
      phases 4–7 surface plus the committed phases 2–3 modules. No exploitable correctness defect; every finding was
      fixed. Production: the four transform adapters now delegate to one shared
      `Program::discharge_local_references` gate that owns both the external-root rejection and the hidden-output
      invariant previously enforced only at the XLA boundary; the re-added duplicate partial-evaluation entry gates
      were removed again with the replay-loop comment corrected; the tautological StableHLO reference-name scan was
      deleted; the dual lowering boundary forms are pinned by a debug assertion (the auditor's stronger
      single-ABI proposal was rejected — the unlifted-plus-captures form is the pre-existing public module contract);
      the independent verifier's semantics arm documents its unique core-bug catch; the rematerialization adapter's
      context-recovery diagnostic names its actual cause; and the fresh-root capture rationale now lives in two
      canonical places with cross-references elsewhere. The conservative rejection of region-local references inside
      `shard_map`/rematerialization/linear-call/custom-derivative carriers is now a documented limitation here and in
      the discharge module docs. The dropped changelog entries were restored in `ryft-core` (including the
      `DiscardedCaptures` breaking-change note) and `ryft-xla`, and the missing `ryft-macros` derive-forwarding entry
      was added — twice: the first restoration was lost again to a concurrent commit wave, and the second-round audit
      caught the regression, so verify these entries survive before cutting any release. Tests: the four external-root
      rejection tests were reduced to one owner-module gate test plus
      per-entry routing smokes; the duplicated condition fixture was hoisted into the shared crate test module; group
      comments, alias hoists, oracle literals, `pretty_assertions`, and the last two UFCS call sites were cleaned up;
      and a new pin records that `jit_call` residual candidates now classify through callee provenance to leaf
      producers. Final verification passed: 1,489 core tests plus 3 ignored; 502 XLA tests plus 5 ignored; all macro
      suites; zero compiler warnings across all targets; formatting and `git diff --check`.

- [x] 2026-08-20 combined Phase 8–9 implementation/review pass 20: added arrays-owned `ArrayReference` handles with
      composable static-index and positive unit-stride slice views over one generic root holder. Eager mutations are
      transactional, root mutations keep a direct no-allocation path, analysis preserves canonical root identity and
      rejects derived views at attached-region boundaries, and discharge lowers view access through canonical
      slice/reshape/update-slice SSA. Exact eager, discharge, condition-region, and XLA tests cover composed and
      overlapping views, indexed inverse reconstruction, invalidation, and root-only structured carries. Added the
      opt-in synchronous stateful compilation/JIT surface: logical external-state metadata and the public-output split
      flow through lowering, compiled artifacts, replacement/cache identity, and V6 persistence while physical indices
      remain derived from `XlaExecutableSignature`; entry aliases merge with sharding attributes; and ordinary calls
      reject stateful executables. Runtime transactions deduplicate and order root holders by `ReferenceId`, complete
      fallible preflight before extraction, keep every guard through the execution fence, donate only uniquely owned
      mutated state, install all hidden states before reconstructing public outputs, and otherwise leave holders ready
      before handoff or poison all extracted mutated holders afterward. Coverage includes public/captured/read-only and
      multiple holders, reverse logical/identity order, retained snapshots, uniqueness downgrade, zero-output errors,
      exact failure boundaries, non-root/effective-sharding rejection, physical alias mapping, restored captured V6
      execution, corruption, replacement, and retained-dispatch cache reuse. Three independent correctness,
      runtime/ABI, and conventions/simplicity audit loops fixed every finding and each final pass reported clean. Final
      verification passed: 1,502 core tests plus 3 ignored; 520 XLA tests plus 5 ignored; 57 macro unit tests; 20
      operation-macro integration tests and 17 parameter-macro tests including all trybuild cases; core and XLA
      all-target checks; core documentation with 95 pre-existing warnings and warning-free XLA documentation;
      formatting, stale-identifier, added-line-width, and `git diff --check` audits. Phase 10 asynchronous leases and
      generations plus Phase 11 shape and distribution support were deferred by this pass and are now in progress;
      the Phase 12 preserved-reference kernel contract remains planned.

- [x] 2026-08-20 combined Phase 2–9 re-audit fix pass 21: three independent correctness, conventions,
      and simplification audits over phases 8–9 (re-flagging 2–7) produced a consolidated fix inventory; per an
      explicit owner directive, changelog entries are excluded from this pass because the owner writes them manually,
      and Phase 10/11 development proceeds concurrently in the same files. Correctness wave: `ReferenceGuard::poison`
      is infallible so failure paths cannot trade the backend error for a guard-state error; the stateful transaction
      window poisons with its actual cause from extraction through installation (the tautological `accepts` pre-pass
      was deleted and `accepts` narrowed to `pub(crate)`; the structure was subsequently absorbed by the Phase 10
      reservation/generation rework, which preserves poison-with-cause); zero-public-output synchronous execution now
      blocks on the execution fence so asynchronous errors are not lost; `interpret_async` rejects stateful
      executables; compile-time cache resolution cross-checks restored/shared executables against the requesting
      lowering's invocation metadata (shared `incompatible_xla_invocation_field`), and replacement compatibility now
      includes `requires_assertion_handler`; the zero-space state diagnostic names the real condition and the dead
      input-side alias-injectivity check was replaced by an invariant comment. Simplification wave: view transforms
      normalize through one `ViewSelection` (starts/limits/squeezed-axis/output-shape) shared by type derivation,
      eager apply/replace, and reconstruction proof, with `validate_reconstruction` documented as the metadata
      round-trip guard that runs once per composition; `ArrayReference` caches its derived `ReferenceType` (borrowed
      `Typed::r#type`, incremental `with_transform` validation, no per-access re-fold);
      `with_validated_transform` was renamed `with_transform_unchecked`; dead `ArrayReference::{view, is_root}` and
      `Reference{Index,Slice}Operation` field accessors were deleted; the two view operations share one
      `infer_view_output_types`; the read/swap/add-update discharge arms share `stage_view_access`; the duplicate
      `XlaProgramTracer`/`XlaProgramValue` aliases were renamed away (91 sites); `ReferenceSource` dropped
      `#[non_exhaustive]` and its three unreachable catch-alls. Conventions wave: byte-identical discharge impl
      headers merged; `# Parameters` completed for `discharge_local_references` and
      `analyze_references_with_capture_indices`; misplaced `discharge_region` doc restored and
      `stage_reference_view_transform` documented; `ReferenceSource`/`ReferenceError`/`ReferenceOutputSemantics`/
      `ReferenceAliasKind`/`ArraySliceAxis`/`Effect` import normalization; stale public/hidden `output_types` docs and
      the `execute_compiled_async` stateless contract documented; batching "lanes" wording replaced and its reference
      probe uses `is_reference`; the rematerialization context recovery uses an invariant `unwrap` and its
      discharge-owned assertions were trimmed; backticks added for `usize` and holder-identity diagnostics; bound
      order and `impl<C>` naming fixed in the eager capability impls and `XlaOperation` reference impls. Deliberately
      kept: the `from_reference_*` named constructors (they document the canonical staged operation set and the
      nested enums have no matching `From` impls). Findings against the pre-Phase-10 execution plumbing
      (`prepare_state` naming, guard-vector shape, state-ABI validator unification, jit stateful dead API) were
      obsoleted by the concurrent Phase 10 rework and intentionally not applied. Test wave: `reference_views` gained
      its owner tests (transform rejections incl. the `usize`-overflow pin, composition algebra and hashing, the
      root-only `read` gate, cached-type consistency, and the derived-view rejection pinned through the live
      `ArrayReference::with_transform` path); the operation tests were split per operation with
      `check_operation_type_inference!` and `indoc!` fixtures; discharge gained a `View`-rule rejection case, a
      core-owned `Call`-rule end-to-end widening test, and the composed index-of-slice swap case; plus analysis and
      tracing comment/import touch-ups. Re-audit rounds: round one found that the staged discharge path still
      re-derived view addresses by hand, that `ViewSelection` carried dead payloads, that the superseded
      `ArrayReferenceView::with_transform` composer survived, and — critically — that the new compile cross-check
      compared boundary types with `ArrayType`'s dimension-variable *identity* equality, false-positively rejecting
      legitimate in-memory and persistent cache hits for bounded-dynamic boundaries. Fixes: staged discharge now
      stages through `ArrayReferenceViewTransform::selection` (one shared normalization with a merged
      `squeezed_output_shape` payload); the superseded composer was deleted; invocation-metadata boundary types are
      compared through the alpha-normalized persistent signature encoding (`canonical_boundary_types`, one "boundary
      types" arm, pinned by a fresh-variable compatibility regression test); the replacement-metadata test covers the
      assertion-handler field; the with-reference-state lowering doc no longer claims an unenforced ordering; the
      stateful dispatcher doc states the real lock-release point; and the `XlaOperation` constant parameter is
      uniformly named `Constant` (the `C`/`Capture` spellings are gone; `C` remains reserved for contexts). Round two
      reported no correctness defects and round three confirmed the residual mechanical fixes. Verification: 1,519
      core, 529 XLA, 17 + 20 macro integration tests; zero warnings; formatting, whitespace, orphan-identifier, and
      line-width audits clean. Changelogs were deliberately left untouched for the owner to write manually.

- [x] 2026-08-20 combined Phase 10–11 implementation/review pass 22: replaced the synchronous holder-lock protocol
      with a type-erased completion contract, cumulative pending generations, read leases, and atomic reservation and
      installation transitions. XLA publishes the complete dependency immediately after successful submission,
      prepares device inputs without holding holder mutexes, revalidates optimistic snapshots by generation, preserves
      ordinary donation, and leaves holders ready when execution returns no fence. Submitted failures poison the
      complete mutated group, dropped completion handles do not cancel work, chained failures remain observable, and
      typed holder state reconciles generation-safely on its next access. PJRT execution fences now provide exact
      all-event readiness, callbacks, dropped-handle keepalive, and ordered error propagation.

      Phase 11 admits static device-memory state with replicated or sharded distribution on fully addressable
      single-process meshes when logical and physical mesh identities match exactly; multi-shard mutation installs
      atomically. Finite replicated bounded-dynamic state is admitted read-only, while mutation remains rejected until
      backend alias compatibility is proven. Zero-space, non-device, unbounded dynamic, dynamic non-replicated,
      bucketing, asymmetric alias-sharding metadata, foreign/non-addressable mesh, and multi-host state all fail before
      launch. V6 persistence revalidates the same state ABI, round-trips bounded-dynamic read-only state, and preserves
      real two-device sharded mutation metadata when the backend exposes executable serialization. No changelog was
      modified, as explicitly requested.

      Independent Phase 10 correctness, Phase 11 ABI/runtime, and conventions/simplicity audit loops fixed every
      finding and each final pass reported clean. Final verification passed: 1,519 core tests plus 3 ignored; 128 PJRT
      tests; 529 XLA tests plus 5 ignored; 20 operation-macro integration tests and 17 parameter-macro tests including
      all trybuild cases; core, PJRT, and XLA all-target checks; documentation generation with 95 pre-existing core
      warnings, 3 pre-existing PJRT warnings, and warning-free XLA docs; formatting, added-line-width, stale-contract,
      and `git diff --check` audits.

- [x] 2026-08-20 combined Phase 12–13 implementation/review pass 23: added an explicitly experimental XLA-owned
      preserved-reference kernel boundary with an array outer ABI and a standalone reference body. Its validator
      consumes the canonical core root, view, access, and source analysis; distinguishes read-only, write-only, and
      read/write operands; records scratch/address-space/alignment/atomic/synchronization requirements only as kernel
      eligibility metadata; and rejects unsupported scratch and target contracts. A deterministic mock artifact
      preserves root-relative views and lowers live-result swap to exchange, dead-result swap to store, reads, and
      ordered accumulation. Complete primitive-contract validation prevents custom public operation families from
      lying about rule shape, accesses, semantics, types, or effects. Exact tests use the same external-root body for
      preserved execution, ordinary discharge, and the immutable oracle, while ordinary XLA rejects the unresolved
      body and the discharged sibling has no entry aliases or kernel state slots.

      Phase 13 stabilizes the documented contract: backend-neutral core reference semantics remain public, backend
      transaction SPI is hidden, and external XLA state plus preserved-kernel APIs are explicitly experimental. Public
      docs now cover local purity, caller-owned impurity, lifetime/freeze/snapshot rules, conditions/loops/scans,
      local AD and batching, captures, asynchronous leases/generations/failures, hidden final-state outputs, and the
      non-semantic nature of may-alias/donation. Phase-targeted reference TODOs and compatibility scaffolding were
      removed. No changelog was modified, as explicitly requested.

      Three independent correctness/architecture, API/docs, and conventions/simplicity audit streams iterated until
      every finding was fixed and the final frozen-tree passes reported no actionable findings. Verification passed:
      1,519 core tests plus 3 ignored; 537 XLA tests plus 5 ignored; 128 PJRT tests; 57 macro unit tests; 20 operation-
      macro integration tests and 17 parameter-macro tests including trybuild; 8 focused preserved-kernel tests; and
      the zero-test facade library. Core, XLA, and facade all-target checks passed. Core doctests passed 69 with 16
      ignored; XLA/facade doctests passed; documentation generation reported only 95 pre-existing core warnings and
      no XLA/facade warnings. Formatting, `git diff --check`, source-rendering, ordinary-reference rejection,
      stale/scaffolding/parallel-universe, added-line-width, and changelog-diff audits were clean.

- [ ] 2026-08-21 whole-plan independent re-audit fix pass 24 (in progress): three fresh independent Opus audits over
      the complete uncommitted feature (core reference semantics and the phase-10 holder state machine, the
      compilation/PJRT/XLA runtime stack, and the phase-12 kernels plus phase-13 docs with a cross-cutting sweep)
      produced roughly sixty verified findings; no changelog was touched, per the standing owner instruction.
      Correctness: the PJRT execution fence no longer destroys the completing event from inside its own native
      `OnReady` callback (retained events are released on a detached thread, and the already-terminal registration
      race no longer leaks the state cycle) and `is_ready` reports the callback-joined terminal result through a
      plain lock; reference analysis tracks consumed roots instead of exhaustively seeded live roots, so a root
      forwarded out of a depth-two nested region is usable where it lands; derived-view `swap`/`add_update` enforce
      the exact derived referent type (update-slice fit alone allowed silent partial writes); `ReferenceGuard::take`
      rejects extraction while a read lease is active; `Reference::validate_live` no longer blocks view derivation on
      pending completions or reservations; non-bijective handle renamings are reported in the caller's direction;
      the reference-free stateful async path carries the whole-execution fence as its completion (awaiting a
      reference-free `call_statefully_async` observes asynchronous errors, and the shared `call_stateless_request`
      body keeps the zero-output blocking rule); `PendingXlaReferenceReservations` poisoning is non-panicking so an
      unwind cannot abort with holders reserved; the post-snapshot argument projection propagates instead of
      unwrapping; and the V6 sharding-arity diagnostic no longer reuses the donation-arity message. Docs contradicted
      by code were corrected: donation is never applied to reference-state inputs (`tf.aliasing_output` is a
      non-semantic hint), post-installation failures do not poison, the discharge schematic now names the real
      `*_with_local_references` adapters (reverse mode goes through `Pullback`; there is deliberately no direct
      `vjp` adapter), the deleted fixed-shape restriction and the unenforced ordering claim are gone,
      `ReferenceOutput` and `RuleRegion` describe the forwarding exemption and closure scope accurately, the
      predecessor dependency is chained (never awaited), and `ExecutionFence::on_ready` documents inline delivery,
      exactly-once, and dropped-handle keepalive. Simplifications: `ReferenceGeneration` is used uniformly (no bare
      `u64` twins), `lock_ready` callers collapse to one `let`-`else`, the checked guard combinators became test-only
      (the unused lease variant was deleted) while `validate_*` plus `*_unchecked` remain the documented backend
      protocol, the dead `execute_compiled_async_with_state` hop and the per-execution clone-and-discard argument dry
      run were removed, read-only roots no longer receive dead synthesized condition/call state outputs, one shared
      carrier-generic view traversal now serves both the eager values and staged discharge (`ViewReadCarrier`/
      `ViewWriteCarrier` with the staged carrier over the program builder), the analysis binding relation collapsed
      to one map, the redundant per-instruction partial-evaluation gate and the vacuous kernel ABI cross-check plus
      its dead error variant and dead match arms were deleted, `batched_with_local_references` takes the batch-axis
      sharding like `Program::batched` (unnamed axis documented), and the five-fold adapter classification sentence
      and four routing-pin comments dedup into the discharge module doc. Conventions: `ReferenceError` sits after the
      imports, facade paths and test imports normalized, diagnostics backtick their identifiers, the kernels module
      doc line and `validate_xla_replacement_metadata` doc were added, stateful test naming/rustdoc style fixed, and
      the vacuous sharded V6 round-trip guard now asserts the exact unsupported-serialization reason. Tests: fifteen
      regression tests plus the first discharge module doctest pin the concurrency contract (pending await, lease
      waits, poison wakeups, stale generations, lease-blocked extraction), raw-handle view mutation typing, depth-two
      forwarding, `UnresolvedAlias`, rendered-root diagnostics, the scan synthesized-carry offset, read-only
      condition boundaries, kernel ordinary-instruction retention and the full forbidden-access matrix, and the
      reference-free async await path. Deliberately kept with rationale: the completion `on_ready` callback surface
      (a documented phase-10 deliverable), the `Taken` holder state (the honest placeholder while a value is
      extracted for donation), the cross-crate `ControlledCompletion` test double (sharing it would add public
      test-support surface), and the batching adapter's internal tracing context (the batching context is the
      canonical axis-metadata carrier). Verification so far: 1,535 core, 540 XLA, 128 PJRT, 20 + 17 macro tests, 70
      core doctests plus 16 ignored, zero warnings, formatting and whitespace clean. Remaining: convergence re-audit
      rounds over this pass's fixes.

All implementation and verification items in this reference plan are complete. The production Pallas-style kernel
language, launch model, and scheduler described by the roadmap remain a separate future program rather than unchecked
work in this plan.
