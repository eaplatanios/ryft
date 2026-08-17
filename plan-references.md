# Ryft References: Architecture and Implementation Plan

**Status:** Phases 0 and 1 implemented and verified; narrow later-phase prerequisites are prototyped, with remaining
phase work still planned

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
    Reference(Reference<A>),
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
- Dynamic-shape, sharded, multi-host, or zero-space external reference aliases in the first XLA slice.
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
| Mixed runtime value universe | `crates/ryft-core/src/arrays/ir.rs` | Add an identity-bearing `Reference<A>`. |
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
  (`XlaDispatchKeyKind::Exact` over `ArrayIrType` signatures, `domains.rs:1019`), so its `Hash`/`Eq` must stay
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

For the first XLA slice, mutation requires exact physical referent compatibility. Dynamic refinement may remain valid
for type checking, but external input/output aliasing is admitted only after physical shape, layout, sharding, and
dynamic-extent compatibility have been proven.

### 5.2 Runtime value and identity

Introduce a reusable `Reference<V: Value>` holding an `Arc`-shared holder for the current value, with identity
following the `DimensionVariable` precedent (`arrays/types/dimensions.rs:198-255`): `PartialEq` via `Arc::ptr_eq`,
`Hash` via `Arc::as_ptr`, and a diagnostic-only `Display`. `ReferenceId` is a handle derived from that pointer (used
for stable lock ordering and diagnostics), not a global counter — no counter scheme exists in `ryft-core` outside
cache statistics, and none should be introduced. `ArrayIrValue<A>` stores `Reference<A>`. The exact synchronization
primitive is an implementation choice, but the semantics are fixed:

- Cloning the internal value aliases the same resource; it does not copy the array contents.
- Equality and hashing use stable reference identity, never mutable contents. `Display` is deterministic and
  type-based (renderings back diagnostics and structural fingerprints), while `Debug` includes the runtime identity.
- No public program operation compares reference identities.
- A read returns an immutable snapshot. Later writes may reuse storage only when doing so cannot mutate any retained
  snapshot; otherwise eager execution or the backend must copy or use copy-on-write protection.
- `new_reference(value)` does not invalidate or make later mutations observable through the initializer `value`.
  Likewise, two distinct roots initialized from storage-sharing values remain logically independent. Physical reuse
  must copy-protect whenever either invariant would otherwise be violated.
- The declared referent type remains invariant for the reference's lifetime. A future dynamic-shape policy may permit
  different concrete runtime refinements within that declaration, but the fixed-shape MVP does not.
- Frozen, failed, or otherwise invalid state produces an explicit error rather than panicking or returning stale data.
- The holder is an opaque `Parameter` leaf. This is free: `Parameter` is a bare marker trait (`parameters.rs:153`),
  so a leaf implementation exposes nothing and there is no traversal behavior to suppress.
- Identity renaming of an eager `ArrayIrValue::Reference` rejects every renaming that would change the handle's type,
  via the shared rejection helper behind the default `Value::rename_type_identities` behavior (identity renamings and
  renamings that do not touch the handle's identities clone the handle; type-changing renamings error): renaming
  shared holder metadata would rename every alias, while minting a new holder
  would break resource identity, so neither is admissible. Because reference constants are structurally rejected at
  region sealing, no built program can store an eager reference constant, so this rename rejection can never be
  reached through program import/inlining — it governs directly held eager values only, while the staged
  `XlaConstant` captured-reference path renames successfully because it stores only metadata. Before views land
  (Phase 8), separate root-shared holder state (and `ReferenceId`) from the handle-local referent
  type and eventual view mapping; a renamed or projected handle then retains its root without mutating shared
  metadata, and both the rename restriction and the import limitation can be revisited.
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
projects to `Reference<A>`. A standalone `Value` implementation can be added later only if an independent reference
domain has real use.

The generic carriers are reusable representation, not a promise that every `T` or `V` is a supported referent. Each
composite universe chooses its admitted specialization. Array IR admits only `ReferenceType<ArrayType>` and
`Reference<A>`; it remains structurally unable to contain a reference to a dimension, another reference, or a List.

### 5.3 Operations

Start with whole-array operations:

| Operation | Logical signature | Semantics |
|---|---|---|
| `NewReferenceOperation` | `Array -> Reference` | Create a scoped root initialized from the array. |
| `ReferenceReadOperation` | `Reference -> Array` | Snapshot the current value. |
| `ReferenceSwapOperation` | `(Reference, Array) -> Array` | Return the old value and install the new value. |
| `ReferenceAddUpdateOperation` | `(Reference, Array) -> ()` | Ordered elementwise accumulation. |
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
  instantiated referent type and, in the fixed-shape MVP, the same exact runtime extents, layout, sharding, and memory.
- For a future dynamic referent, an input may refine the declared referent only when the holder and selected backend
  explicitly support changing concrete extents within that declaration. This is rejected until Phase 11.
- `add_update` may use ordinary array addition semantics internally only when the inferred addition result has exactly
  the current instantiated referent type. Any promoted or broadcast result that would change stored type is rejected.
- Reference-type compatibility/refinement is not a substitute for these operation-specific storage checks.

After whole-value discharge is proven, add explicit reference views:

```text
ReferenceView {
    root,
    ordered transforms: [index, slice, reshape, transpose, bitcast, ...]
}
```

Only implemented and validated transforms may appear. A view preserves its base root, composes its mapping back to
that root, and never allocates a new resource. Indexed reads, writes, swaps, and accumulations then operate through the
view. Views are never freezable: `freeze` accepts only a root handle, and `freeze(view)` is rejected in the MVP —
whether it would return the slice or the complete root is ambiguous, and neither reading composes cleanly with
alias-family invalidation. Start with indexing/slicing and reuse Ryft's canonical array indexing and gather/scatter
descriptors rather than inventing a second indexing language.

### 5.4 Purity, lifetime, and second-class restrictions

A program that allocates and uses only local references is externally pure after discharge. It may explicitly freeze a
root to recover its final array, or implicitly discard a still-live unescaped root at the end of its creation region. A
program that reads or mutates an argument or captured reference is externally stateful.

Initial static rules:

- References cannot be public program outputs.
- References cannot be condition branch results, while carries/results, scan sequences/carries/results, or nested-call
  public results. Discharge-generated hidden carries and outputs are arrays, not references.
- References cannot be nested in references or Lists.
- Only explicit view operations introduce aliases.
- Every reference operand resolves to exactly one canonical root.
- A local root cannot escape its creation scope.
- `freeze` is valid only in the local root's creation scope, consumes the root, and invalidates all aliases and views.
- Scope exit implicitly discards and invalidates any still-live, unescaped local root without producing an array
  result. This is a program-execution concept: discharge drops the root's final current state, and interpreted region
  execution invalidates the alias family before the region is released, because only there does an owner observe the
  scope exit. Directly eager references have no observable creation scope — handles are cloneable and `Arc` last-drop
  is not scope exit — so they are never implicitly invalidated: they live until an explicit `freeze` or until the
  last handle drops, and their second-class restrictions are enforced at program boundaries (arguments and captures),
  where validation exists.
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

Static SSA validation and invocation-time holder validation are distinct and both are required.

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
    outputs:  NewRoot { output_index } | Alias { output_index, input_index }
    accesses: input_index -> Read | Write | Accumulate | Consume
```

All indices are operation-local operand/result positions, never resource identifiers. Output classification is
mutually exclusive by construction (an output is a fresh root or an alias, never both), and `Alias` carries exactly
one `input_index` on purpose: it structurally encodes the one-canonical-root invariant (§5.4), so multi-source
aliases are unrepresentable rather than merely rejected. There is deliberately no `ReadWrite` mode (`swap` classifies
as `Write`, matching JAX's `swap_p`) and no `Freeze` mode (`freeze` classifies as `Consume`, a lifetime event that
also covers a future result-less `free_reference`). View operations attach their ordered coordinate-transform stacks
to the `Alias` case when the Phase 8 view representation lands; the alias edge alone is all root resolution needs
until then.

The program-level `ReferenceAnalysis` resolves these templates to canonical roots:

```text
EntryInput(index)
Capture(index)
Allocation(region, instruction, output)
```

It must:

- propagate roots through views;
- substitute region inputs and captures at condition, loop, scan, and call boundaries;
- summarize nested-region access in parent-root terms;
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
- while conditions do not write state in the supported subset;
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
   nothing itself; `XlaLoweredProgram::capture_count` at `domains.rs:1232` is the existing precedent for carrying
   this count alongside a lifted program — mirror it.)
3. Run reference analysis and discharge on the lifted open program.
4. Return a binding recipe that maps each logical external-reference slot back to its original capture or public
   argument position.
5. At interpretation/execution, snapshot or transactionally extract the holder's current array according to the slot's
   access disposition, then install any hidden final state back into that same holder.

Nested closed calls are already normalized at staging time: `StagedFunction::call_with_flat_capture_references_in_context`
re-registers callee captures in the caller's table and attaches the callee's memoized lifted program
(`compilation/function.rs:592-641`) — verify rather than implement, and memoize discharge alongside that `OnceLock`
cache rather than invalidating it. One gap is real, however: `to_program_with_lifted_captures` rewrites
capture-referencing constants only in the entry region, while `CaptureReference`s inside attached regions are
preserved verbatim and resolved later against the hidden capture-argument prefix (`captures.rs:546-551`). The "no
reference-typed `CaptureReference` survives lifting" invariant therefore does not hold automatically. Phase 4 closes
it temporarily in the validator: reject any reference-typed `CaptureReference` occurring inside an attached region
after lifting. Phase 5 then adds recursive attached-region capture lifting/substitution and removes that rejection —
captured references used directly inside `condition`/`while`/`scan` bodies are a flagship use case (the JAX analogue
is closed-over refs in loop bodies), not a non-goal. The pipeline order is canonical and fixed: recursive capture
normalization first (entry lifting plus, from Phase 5, attached-region lifting), then `ReferenceAnalysis` and
validation over the normalized program, then discharge of exactly that normalized program using exactly that analysis
artifact — any structural rewrite after analysis would leave the artifact stale against renumbered atoms and region
boundaries. The resulting ordinary
program may contain array inputs/captures but no rewritten capture table containing mutable array contents, keeping
`ClosedProgram`'s capture type/value invariant intact and giving core discharged interpretation and XLA compilation
one boundary contract.

### 8.2 Result contract

Discharge belongs in `ryft-core`. It returns a reference-free program and logical external-state metadata:

```rust
pub struct DischargedReferenceProgram<O> {
    program: Program<...>,
    public_output_count: usize,
    external_states: Vec<DischargedReferenceState>,
}

pub enum ReferenceSource {
    Capture(usize),
    PublicInput(usize),
}

pub struct DischargedReferenceState {
    slot: usize,
    source: ReferenceSource,
    discharged_input_index: usize,
    access: ExternalReferenceAccess,
    final_state_output_index: Option<usize>,
}

pub enum ExternalReferenceAccess {
    ReadOnly,
    Mutated,
}
```

Names and generic details remain subject to implementation review. The contract is:

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
root state. Preserve array data type, dimension identities, layout, sharding, memory, source locations, and all
non-state effects.

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
fence carrier, `domains.rs:2633-2639`); `call_statefully` is the blocking Phase 9 convenience; and
`call_statefully_async` returns the completion-bearing wrapper, keeping its public shape in Phase 10. Model the
stateful surface as an opt-in `StatefulCompilationDomain` capability rather than required methods on every
`CompilationDomain`: `CompilationCall` fixes `RuntimeOutput = Output::To<D::Value>` (`compilation/function.rs:927`),
so the stateful path is a new request/method, not a reparameterization of the existing trait. Refactor executable
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
`project_inputs`/`project_outputs` projectors (`lowering.rs:171-196`, `:327-334`), and the referent type is
`XlaLoweredProgram::input_types[logical_input_index]`. Duplicating any of them creates a second source of truth.
Resolve physical indices through those accessors at use time, only after composing with:

- capture/public-input flattening;
- zero-space logical value erasure;
- hidden bounded-dynamic extent inputs;
- public, hidden state, and hidden dynamic-extent result ordering.

Never derive physical indices by adding counts or assuming one logical value maps to one physical value. This is
already the codebase's discipline: the donation path flattens captures plus public inputs and projects through the
signature at `domains.rs:2165-2174` — extend that construction rather than adding a parallel one.

Initially admit only these combinations, stated in terms of resolved mappings:

- read-only: `input_mapping[logical_input]` is present and there is no hidden final-state output;
- mutated: `input_mapping[logical_input]` and `output_mapping[logical_output]` are both present.

Public and executable output signatures are distinct concepts and must be modeled separately: the staged function's
public output types/structure on one side, and the executable's logical output list (public outputs followed by
hidden final-state outputs) on the other, split by a validated `public_output_count` carried through lowering,
execution, persistence, and executable replacement. Today's lowering validates the staged public signature directly
against the complete lowered output list (`validate_output_types` at `domains.rs:2374`) and applies user
`out_shardings` to the complete list (`domains.rs:2460`); both must be rescoped to the public prefix, and each hidden
state output inherits its paired input's effective sharding.

Reject any external reference whose physical input is erased, including zero-space references, until Phase 11 defines
another representation. A read-only slot is never donated or aliased. A mutated slot has exactly one hidden output and
one may-alias relation.

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
   slot, transactionally take the current array and dependency.
5. Build ordinary physical arguments through `XlaExecutableSignature`.
6. Construct logical donation flags in flattened capture-plus-public-input order, then project them through
   `XlaExecutableSignature`: ordinary captures and dimensions are `false`; ordinary public arrays use the user's flag;
   read-only reference inputs are `false`; mutated reference inputs are internally `true` subject to safe uniqueness
   downgrade; hidden extent carriers are `false`.
7. Cross one explicit irreversibility boundary when donatable inputs are handed to the PJRT execute call.
8. Immediately after successful PJRT submission, while the ordered holder guards remain held, atomically publish the
   execution fence as a read lease on every read-only holder and reserve a pending generation on every mutated
   holder; then release the guards. Nothing fallible precedes this step — today's output splitting constructs
   arrays, materializes dynamic extents, and validates ownership fallibly after obtaining the fence
   (`domains.rs:2178-2192`), so lease/reservation publication must come first or a failure there leaves active
   device reads unregistered.
9. Construct, split, and validate public results, hidden state results, and hidden dynamic extents.
10. Replace every mutated holder's reservation with its pending final value, carrying readiness events and
    generation/dependency information — or poison all mutated holders if any hidden state cannot be constructed or
    validated.
11. Reconstruct and expose only public outputs through the completion-bearing stateful-call wrapper.

Required failure semantics:

- Before the PJRT irreversibility boundary, every extracted mutated state is restored.
- When `PJRT_LoadedExecutable_Execute` returns an immediate error without producing a fence, do not assume no device
  read was accepted unless the PJRT contract proves it: restore mutated states (the pre-boundary rule), and
  conservatively quarantine read-only participants behind a synthetic lease until the error path is proven
  access-free. Narrow this only when the PJRT contract guarantees non-acceptance.
- After that boundary, Ryft does not claim it can restore a potentially donated input. If all hidden final states can be
  constructed, it installs them even when later public-output reconstruction or refinement fails. If any hidden state
  cannot be constructed or validated, it poisons every mutated holder in the invocation.
- If future PJRT APIs can prove that a failed submission did not accept donation and return recoverable inputs, a
  narrower restore path may be added. Until then, execute-call failure after handoff conservatively poisons mutated
  holders.
- Asynchronous execution failure poisons mutated holders with the execution error; later reads/writes fail until an
  explicit replacement/reset API exists. Read-only participants are not poisoned by this invocation's failure.
- The completion-bearing call reports launch, public reconstruction, read-only execution, and asynchronous errors even
  when the public output structure is empty.
- Once execution crosses the irreversibility boundary, dropping its completion handle does not cancel or roll back the
  mutation.
- A holder never exposes potentially donated stale storage as its current state.
- State installation for a multi-reference call is logically atomic: all mutated holders become pending on the same
  submitted execution, all are restored before handoff, or all are poisoned after an unrecoverable post-handoff
  failure.
- Calls involving the same holder serialize through its state/dependency chain. Read-only calls may overlap only when
  the holder tracks all outstanding read leases; a mutation waits/dependency-chains them before donation. Independent
  holders may execute concurrently.
- Pending states use generation-safe cumulative dependency/error chains. If call B consumes call A's pending result
  and A later fails, B cannot overwrite or hide that failure; the holder remains poisoned. Older completion callbacks
  cannot mutate a newer generation.

Do not hold a host mutex for device execution duration. Install pending generation/event state and read leases, then
let later accesses await or dependency-chain them.

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

### 11.5 XLA support order

1. Local fixed-shape references, which disappear before XLA lowering.
2. One fixed-shape unsharded external reference, synchronous protocol, with copy-protection fallback.
3. Multiple unique external references (synchronous).
4. Captured references (synchronous).
5. Persistent executable round trips and replacement compatibility.
6. Asynchronous sequencing: pending states, read leases, and overlapping-call semantics.
7. Sharded references with atomic whole-shard-set holder updates.
8. Bounded-dynamic references after physical alias compatibility is proven.
9. A deliberate zero-space-reference policy.
10. Multi-device and multi-host failure propagation.

Each later class remains rejected at lowering until its ABI and runtime tests land.

## 12. Pallas-ready preserved-reference path

The future kernel path must consume the same logical reference abstraction and choose preservation instead of
discharge inside an explicitly validated kernel region.

### Shared core contract

- `ReferenceType<ArrayType>` and `ArrayIrValue::Reference(Reference<A>)` remain unchanged.
- Root identity, access modes, lifetime checks, and view composition come from the same `ReferenceAnalysis`.
- Ordinary and kernel lowering share read/write/swap/accumulate semantics.
- `ArrayType` remains the canonical source of element, shape, layout, sharding, and memory information.
- The canonical `Memory` vocabulary may later be generalized for target-neutral global, workgroup/shared, and
  private/local spaces; no parallel reference-memory enum is introduced.

### Kernel-owned concepts

The following do not belong in `ReferenceType<T>`:

- grid shape and program IDs;
- block specifications and launch dimensions;
- scheduling and software-pipeline metadata;
- barriers, semaphores, and asynchronous-copy topology;
- backend-specific address-space encodings.

They belong to a future higher-order kernel-call operation and its attached kernel region.

### Future kernel phases

1. Define a kernel-call boundary whose operands are read-only, write-only, read/write, and scratch references.
2. Define preserved-reference eligibility and prohibit references outside validated kernel regions.
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

Architecture acceptance does not require implementing these phases now. It requires proving that the ordinary
reference design contains no XLA-functionalization-specific field or assumption that would force replacement later.

## 13. Relationship to future first-class Lists

A future `List` can also become an `ArrayIrType`/`ArrayIrValue` member. References and Lists therefore share:

- the composite member/projection machinery;
- region-aware replay capable of changing representation and arity;
- transform legality gates;
- backend-specific physicalization.

They do not share semantics. A reference is an identity-bearing effect capability over fixed-shape storage. A List is
a persistent variable-cardinality computational value. Do not model a logical List as `Reference<List<T>>` or make List
operations stateful.

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
      per-argument `HashMap` with multi-attribute round-trip tests (`ryft-mlir/src/types.rs:120-126`,
      `operations/traits.rs:596-680`), and sharding already attaches at the entry-lowering site to extend
      (`sdy.sharding` in physical index space, `lowering.rs:4321-4342`).
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
- [x] Add `ArrayIrValue::Reference(Reference<A>)` and
      `ValueProjection<ReferenceType<ArrayType>, Projected = Reference<A>>`.
- [x] Implement identity-based equality/hash with deterministic type-based display (runtime identity stays on
      `Debug`, per the `Value` rendering contract) and alias-preserving internal clone behavior.
- [x] Update exports and documentation that currently describe Array IR as containing only arrays and dimensions.
- [x] Audit every exhaustive type/value match in core, XLA, tests, and derives. Each site must intentionally support,
      reject, or remain unreachable after a verified pipeline boundary. Measured magnitude: ~40 files, with
      `arrays/operations/control_flow.rs` (75 matches), `arrays/batching.rs` (37), `ryft-xla/.../ops.rs` (29),
      `arrays/operations/constants.rs` (23), and `ryft-xla/.../shard_map.rs` (22) as the top sites.
- [x] Handle `XlaConstant` explicitly: its `ValueProjection` impls are hand-written per member
      (`ryft-xla/src/experimental/ops.rs:145`, `:171`, `:200`), and its `Typed`/`Value`/`CaptureConstant` matches
      (`ops.rs:63-130`) deliberately support typed `CaptureReference<ReferenceType<ArrayType>>` metadata while
      excluding concrete `Reference<Array>` holders from literal/serialized constants. By contrast, the tracer
      projections (`Tracer`, `PartialTracer`, `DifferentiationTracer`, `CaptureReference`) are blanket impls keyed on
      the seam-1 `TryFrom` and fall out for free.
- [x] Record that batching projection is compile-time absent for references: `BatchingPolicyProjection` is
      implemented only for `ArrayType` and `DimensionType` (`arrays/batching.rs:2445`, `:2500`), so a
      reference cannot be projected through `BatchingTracer` at all — the plan's batching rejection is structural,
      not a runtime guard.
- [x] Add tests for cross-kind failures, projection, dynamic referent identity refinement, aliasing clones, poisoned
      holder access, a dimension variable shared across array, dimension, and reference leaves, dynamic-to-dynamic
      identity-renaming derivation through references (declared `ref<f32[n]>` against actual `ref<f32[m]>` yields
      `n -> m`), repeated identities, mixed array/reference occurrences of one identity, and eager non-identity rename
      rejection. Alias-family invalidation tests land with the first real consuming `freeze` operation in Phase 3.

**Exit criterion:** references can be stored, typed, projected, and diagnosed — with reference *types* and staged
metadata renameable, while eager handles reject non-identity renames until Phase 8 (§5.2) — without changing ordinary
array/dimension behavior or being accepted by numeric operations.

### Phase 2: Add effects, reference semantics, and validation

- [x] Add `Effect::OrderedState`: extend `Effect::ALL`, `Effect::bit`, and `Effect::is_ordered`, plus the exhaustive
      `EffectTokens::get`/`set` matches in XLA lowering.
      `OrderedState` gets no token slot: its `EffectTokens` arm is an error made unreachable by the discharge
      verifier, never a new token chain. No `Display` exists for effects, so there is no rendering to update. Add
      tests.
- [x] Add the operation-level reference semantics contract: mutually exclusive output classification
      (`NewRoot` | `Alias`) plus input accesses (`Read`/`Write`/`Accumulate`/`Consume`) in operation-local index
      space, with per-operation examples in the rustdoc; view transform stacks attach to the `Alias` arm in Phase 8.
- [ ] Complete the whole-array reference operation set as native `ArrayIrOperation` variants (`new_reference` and
      `read` are already present from Phase 0; `swap`, additive update, and `freeze` remain).
- [ ] Mirror the remaining operations in `XlaOperation` so XLA staging can carry them before discharge
      (`new_reference` and `read` are already mirrored).
- [x] Give every currently implemented unresolved state access the coarse ordered-state effect; apply the same rule to
      each remaining operation as it lands.
- [ ] Implement `ReferenceAnalysis` over entry inputs, captures, allocations, aliases, and nested regions.
- [ ] Implement the dedicated static validator.
- [ ] Implement invocation-time duplicate-holder validation separately.
- [ ] Verify (existing behavior, regression tests only) that simplification retains effectful instructions with
      unused outputs in program order (`programs/programs.rs:639-700`, `:1204-1220`) and that region ops need no
      effect overrides because seal-time folds already aggregate nested-region effects (`programs/regions.rs:413-448`,
      `Operation::effects` defaults to `PURE` at `operations.rs:498-500`).
- [ ] Verify (existing behavior, regression tests only) that rematerialization force-saves residual roots reaching
      non-pure instructions and hard-errors on replaying them (`tracing_v2/rematerialization.rs:1524-1533`,
      `:1598-1628`, `:1726-1735`).
- [ ] Implement the partial-evaluation gate — this one is genuinely new work, not verification: the default
      `fold_or_residualize` contract places all-known *effectful* operations on the known side and executes them
      (`partial.rs:863-890`), the opposite of what unresolved state needs. The "never execute, fold, or split an
      unresolved state chain" rule must be implemented at that level; the existing per-region-op purity gates
      (`condition.rs:301-308`, `scan.rs:1090-1103`, `while.rs:501`) are not sufficient.
- [ ] Add precise diagnostics for every second-class, alias, scope, freeze, root, and unsupported operation violation.

**Exit criterion:** every reference operation has both conservative effect visibility and a precise canonical root;
invalid programs fail before any mutation or replay.

### Phase 3: Implement eager and staged local references

- [ ] Implement eager create, read, write, swap, additive update, and freeze.
- [ ] Enforce the exact replacement/update storage rules on every write, swap, and additive update; do not use broad
      `ArrayType::is_compatible_with` as the mutation rule.
- [ ] Make freeze invalidate the complete alias family.
- [ ] Expose array-to-reference creation and reference capabilities through the composite parent-binding API.
- [ ] Stage operations with exact inferred types, source locations, rendering, effects, and access semantics.
- [ ] Support local references in straight-line programs.
- [ ] Add eager/staged equivalence tests and all lifecycle/type errors.
- [ ] Verify that a retained read snapshot is unchanged by every later write/update path.
- [ ] Verify that the retained initializer and two distinct roots initialized from storage-sharing values remain
      unchanged/independent under later mutation.
- [ ] Test explicit freeze, implicit scope-exit discard, a never-frozen local root, nested-region local discard, and
      invalid branch/loop lifecycle paths.
- [ ] Test that directly eager references are never implicitly invalidated: they remain valid until explicit `freeze`
      or last handle drop, with implicit scope-exit discard applying only to staged/interpreted region execution
      (§5.4).

**Exit criterion:** the whole-array reference language has one observable meaning in eager and staged execution.

### Phase 4: Implement straight-line discharge

- [ ] Add a core discharge module and result metadata types.
- [ ] Integrate discharge after `ClosedProgram::to_program_with_lifted_captures` and return the canonical
      capture/public-holder binding recipe without rewriting concrete capture tables.
- [ ] Validate the complete program before constructing output.
- [ ] Consume the validated `ReferenceAnalysis` artifact as the discharge input contract; do not re-resolve roots,
      aliases, or accesses inside the rewrite.
- [ ] Temporarily reject reference-typed `CaptureReference`s inside attached regions (entry-region lifting only,
      `captures.rs:546-551`); Phase 5 removes this via recursive attached-region lifting.
- [ ] Track one immutable current array per root.
- [ ] Rewrite each whole-array operation according to the state-passing semantics.
- [ ] Eliminate local create/freeze state and preserve mutated external state as hidden outputs.
- [ ] Preserve non-state effects, source locations, identities, layout, sharding, and memory.
- [ ] Verify that successful output contains no reference artifacts or ordered-state effect.
- [ ] Implement `Operation::render` for the reference and view operations per the fingerprint contract
      (`programs/operations.rs:521-539`), and give the discharge metadata — which is not an operation — deterministic
      `Debug`, serialization, and equality instead; add determinism tests. Renderings back the debug-assertions
      transform-cache determinism recheck (`programs/transforms.rs:591-618`) and rendered-program test assertions —
      they are not production cache keys.
- [ ] Add property tests over short generated straight-line state programs against eager and hand-written oracles.

**Exit criterion:** straight-line local and external reference programs produce a deterministic reference-free core
program plus complete logical state metadata.

### Phase 5: Extend discharge through regions and calls

- [ ] Thread canonical state through condition branches and joins.
- [ ] Thread body-mutated state through while carries.
- [ ] Allow while conditions to read current state and reject condition writes.
- [ ] Thread scan-mutated state as carries, separate from per-step values.
- [ ] Rewrite nested calls and substitute callee root mappings into callers.
- [ ] Derive every nested-region and callee root substitution from the same `ReferenceAnalysis` summaries used by
      validation; never re-resolve locally.
- [ ] Add recursive attached-region capture lifting/substitution so reference-typed captures work directly inside
      `condition`/`while`/`scan` bodies, and remove the temporary Phase 4 validator rejection. Normalization runs
      before `ReferenceAnalysis`, so analysis and discharge always see the same normalized program (§8.1).
- [ ] Handle captures and local region allocations without escape.
- [ ] Preserve zero-iteration and untaken-branch state exactly.
- [ ] Test one and multiple roots, different branch write sets/counts, nested regions, nested calls, and invalid escapes.
- [ ] Add generated small-control-flow equivalence tests where practical.

**Exit criterion:** every supported higher-order reference program has an equivalent reference-free array program, and
all unsupported ownership/control-flow patterns fail before replay.

### Phase 6: Integrate generic transforms safely

- [ ] Route staged local-reference simplification through the documented pre/post-discharge behavior.
- [ ] Route externally pure local-reference partial evaluation through discharge and reject every remaining
      reference-bearing case. Do not use the generic default rule or claim whole-chain residualization in the MVP.
- [ ] Add trace -> validate -> discharge -> replay support for forward and reverse AD entry points.
- [ ] Add the corresponding route for direct/eager differentiation APIs or reject them explicitly until it exists.
- [ ] Route local-reference batching through discharge; reject external/mapped/shared reference batching.
- [ ] Route rematerialization through discharge; reject externally stateful rematerialization.
- [ ] Reject references in custom derivative/rule regions.
- [ ] Add guards proving no reference reaches generic AD representational rules. For batching, the projection is
      already compile-time absent (`BatchingPolicyProjection` covers only `ArrayType` and `DimensionType`,
      `arrays/batching.rs:2429-2493`), so the guard work there is error quality, not prevention.
- [ ] Test nested transform orderings: discharge/JVP/transpose, discharge/batch, discharge/remat, and transforms around
      condition/while/scan.

**Exit criterion:** every public transform has a documented successful path or targeted rejection; no reference case
succeeds by accidental structural or zero-space treatment.

### Phase 7: Compile local references through XLA

- [ ] Invoke core validation/discharge before the current array-only XLA boundary.
- [ ] Carry discharge metadata into lowering even when no external states exist.
- [ ] Add an explicit verifier rejecting surviving references before StableHLO construction.
- [ ] Keep public array-only JIT input/output APIs unchanged.
- [ ] Test static fixed-shape local refs through straight-line and control-flow programs.
- [ ] Snapshot reference-free StableHLO and compare execution with eager/discharged core interpretation.

**Exit criterion:** local reference programs compile and run through ordinary XLA without adding a StableHLO reference
representation or changing the executable ABI.

### Phase 8: Add indexed views and transformations

Views land before the external runtime because they are the bread-and-butter programming model (accumulate-into-slice
and the scan patterns references exist to replace), they are pure core analysis/discharge work with no concurrency
risk, and they exercise the view-to-gather/scatter discharge path that later phases reuse.

- [ ] Separate root-shared holder state (and `ReferenceId`) from the handle-local referent type and view mapping
      (§5.2), so renamed or projected handles retain their root without mutating shared metadata.
- [ ] Define the composable reference-view representation, attaching the ordered transform stack to the semantics
      contract's `Alias` arm (which ships without transforms until this phase).
- [ ] Add indexing/slicing views and bounds/type validation.
- [ ] Lower reads/writes/swaps/additive updates through canonical array slicing/gather/scatter/update operations.
- [ ] Preserve base-root identity and composed access mappings.
- [ ] Add reshape/transpose/bitcast only with explicit layout, overlap, and alias proofs.
- [ ] Add base/view mutual-observation, composed-view, overlap, invalidation, and discharge equivalence tests.

**Exit criterion:** views provide enough backend-neutral address semantics for ordinary discharge and future kernel
lowering without creating independent resources.

### Phase 9: Add synchronous external and captured XLA references

The synchronous contract: every externally stateful call acquires per-holder guards in stable `ReferenceId` order,
holds them through device completion, then installs all hidden final states (or poisons all mutated holders) before
releasing and returning. Awaiting before returning does not serialize concurrent host threads; the held guards do.
No pending states, generations, or read leases exist yet (§11.3 staging).

- [ ] Define and implement the eager/XLA holder state machine with ready, poisoned, and frozen/invalid states;
      pending states are deferred to Phase 10.
- [ ] Hold per-holder guards in stable `ReferenceId` order for the entire execution, through device completion and
      state installation; do not rely on await-before-return for serialization. Holding host guards across device
      execution is the accepted Phase 9 limitation that Phase 10 removes.
- [ ] Extend the executable-signature metadata with logical reference-state slots only; resolve physical indices and
      referent types through the existing `input_mapping`/`output_mapping`/`input_types` accessors at use time
      (§11.2), never storing them.
- [ ] Separate public and executable output signatures end to end (§11.2): executable logical outputs are the public
      prefix followed by hidden state outputs, split by a validated `public_output_count` carried through lowering,
      execution, persistence, and replacement. Rescope `validate_output_types` to the public prefix (it currently
      compares against the complete lowered output list, `domains.rs:2374`), apply user `out_shardings` to the
      public prefix only (currently the complete list, `domains.rs:2460`), and give each hidden state output its
      paired input's effective sharding.
- [ ] Record `ReadOnly` versus `Mutated` disposition; reject reference slots whose logical input is erased
      (`input_mapping` absent) in this phase.
- [ ] Handle retained-JIT dispatch keys: exact keys (`XlaDispatchKeyKind::Exact`, `domains.rs:1019`) hash
      `ArrayIrType`s structurally and are holder-free by construction, but the bucketed path
      (`BucketedDispatchSignature`, `domains.rs:1025-1060`) alpha-normalizes only `ArrayType`s — reference slots must
      take the exact path or be explicitly rejected from bucketing.
- [ ] Derive physical indices only after zero-space and dynamic-extent mappings are complete.
- [ ] Emit and verify `tf.aliasing_output` on the correct entry argument.
- [ ] Implement hidden final-state result splitting and holder installation; keep hidden results out of public
      reconstruction.
- [ ] Construct donation flags in logical ABI order: ordinary capture/dimension/read-only reference `false`, ordinary
      public array from the user flag, and mutated reference internally `true` with uniqueness downgrade. Extend the
      existing flatten-then-project construction at `domains.rs:2165-2174` — the recipe already exists for ordinary
      inputs; do not add a parallel one.
- [ ] Mark the PJRT handoff irreversibility boundary; restore extracted states on failure before handoff, and await
      completion then install-or-poison after handoff.
- [ ] Define pre-handoff, execute-call, and post-submission public-reconstruction failure behavior.
- [ ] Introduce the stateful call surface as an opt-in `StatefulCompilationDomain` capability: `call_statefully`
      (blocking convenience) and `call_statefully_async` returning the completion-bearing wrapper — already-completed
      in this phase, so Phase 10 changes the implementation rather than the public shape. Share the executable
      specialization/dispatcher pipeline with pure calls, make ordinary calls reject executables with non-empty
      external-state metadata, and never lose execution errors on zero-output calls.
- [ ] Add internal mutated-reference donation while preserving copy-protection fallback; never donate read-only slots.
- [ ] Start with one static, unsharded, device-memory external reference; then multiple unique holders with stable
      lock order and all-installed-or-all-poisoned installation; then captured references with the
      reference-specific internal donation policy.
- [ ] Add explicit external reference arguments through a heterogeneous boundary without breaking array-only APIs.
- [ ] Reject public/capture duplicate identities before state extraction.
- [ ] Add state ABI metadata to lowering, compiled programs, cache identity, persistence, and executable replacement;
      bump and validate the persistent executable schema as the complete V6 migration: `XlaPersistentKeyV6`,
      `XlaPersistentExecutableMetadataV6`, schema version 6, and magic `RYFTXLA6` (mirroring the current V5 set at
      `domains.rs:1293`, `:1418`).
- [ ] Add capture, cache round-trip, corruption, replacement, and synchronous failure tests.

**Exit criterion:** consecutive compiled calls observe mutation under a fully synchronous state protocol; retained
snapshots remain valid; failures leave every holder restored or explicitly poisoned; semantics remain correct when
physical alias reuse does not occur; persisted/replaced executables cannot carry mismatched state ABIs.

### Phase 10: Add the asynchronous external runtime protocol

- [ ] Add pending generation/event holder states: reserve generations at submission time under the held guards
      (§11.3 step 8) and replace reservations with pending final values after result construction (§11.3 step 10),
      poisoning all mutated holders when construction or validation fails.
- [ ] Track read-only execution leases, published atomically with the mutated-holder generation reservations while
      the ordered guards are still held — immediately after successful submission and before any fallible result
      processing (§11.3 step 8) — and require later mutations to wait or dependency-chain them before donation.
- [ ] Define the immediate-execute-error-without-fence policy: restore mutated states and conservatively quarantine
      read-only participants behind a synthetic lease unless the PJRT contract proves no device access was accepted
      (§11.3 failure semantics).
- [ ] Use generation-safe cumulative dependency/error state so a failure in an earlier pending mutation cannot be
      hidden by a later chained call or stale callback.
- [ ] Serialize conflicting same-holder calls while allowing safe read overlap and independent-holder concurrency.
- [ ] Make multi-holder pending installation logically atomic on the same submitted execution.
- [ ] Define asynchronous-execution failure poisoning and dropped-completion semantics; dropping a completion handle
      after the irreversibility boundary neither cancels nor rolls back the mutation.
- [ ] Introduce the type-erased completion/dependency token in `ryft-core` (§5.2) and implement it in `ryft-xla` over
      `ExecutionFence`; pending holder states and read leases store the token directly, with no backend side map
      keyed by `ReferenceId`.
- [ ] Do not hold a host mutex for device execution duration; later accesses await or dependency-chain pending state.
- [ ] Add concurrency, overlap, asynchronous failure, chained-failure, and stale-callback/generation tests.

**Exit criterion:** overlapping calls have defined sequencing and failure semantics; asynchronous failures poison
exactly the involved mutated holders; no stale callback or chained call can hide or overwrite an earlier failure.

### Phase 11: Expand XLA shape and distribution support

- [ ] Add sharded references with identical input/final-state sharding and atomic holder updates across shards.
- [ ] Add bounded-dynamic references after alias shape/extent compatibility is proven.
- [ ] Define and implement or explicitly reject zero-space references.
- [ ] Validate memory placement and input-bound bucketing interactions.
- [ ] Add multi-device and multi-host sequencing/failure behavior.
- [ ] Add each class independently with lowering and runtime conformance tests.

**Exit criterion:** every admitted shape/sharding/memory class has a documented ABI, and unsupported classes fail during
validation/lowering rather than after launch.

### Phase 12: Establish the preserved-reference kernel contract

- [ ] Add a mock/kernel eligibility validator using the same roots, views, and access summaries.
- [ ] Define a higher-order kernel-call region boundary and operand access modes.
- [ ] Prove ordinary XLA rejects preserved refs outside that boundary.
- [ ] Define scratch allocation/non-escape, address spaces, atomics, synchronization, and view alignment as separate
      kernel contracts.
- [ ] Lower `swap` by result liveness — exchange when the old value is live, plain store when it is provably dead
      (§5.3, §6.2) — so write-only and scratch operands never require readable previous contents.
- [ ] Lower one conformance program both through ordinary discharge and through a preserved mock/Mosaic path.
- [ ] Keep all grid/launch/backend concepts outside the core reference type.

**Exit criterion:** a future Pallas layer can preserve and lower the existing logical reference IR without replacing
the type, operation, identity, effect, or view model.

### Phase 13: Stabilize APIs and documentation

- [ ] Document local purity versus external impurity.
- [ ] Document second-class restrictions, freeze/invalidation, snapshot, sequencing, and failure semantics.
- [ ] Document transform support and precise unsupported combinations.
- [ ] Document XLA hidden-state and may-alias behavior without promising physical in-place reuse.
- [ ] Add end-to-end local, external, captured, control-flow, AD, batching, and view examples for the supported subset.
- [ ] Remove temporary scaffolding and compatibility layers; every such layer must have been marked at creation with a
      `TODO(eaplatanios)` naming the phase that deletes it, and this phase verifies by search that none remain.
- [ ] Reassess which APIs should remain experimental until external AD and kernel semantics mature.

**Exit criterion:** the supported contract is comprehensible without reading implementation code and does not imply
support for deferred aliases, transforms, dynamic shapes, or kernel operations.

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
  remain generic over `Reference<A>`.

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
| Primitive ordering | create/read; write/read; swap old/new; ordered accumulates; interleaved roots | promoted/broadcast replacement; update result type change; dynamic extent change in MVP; use/view after freeze |
| Aliases/views | base/view mutual observation; composed and disjoint views share root | implicit alias; escaping view; unsupported transform/overlap |
| Effects/liveness | unused write retained; read/write order; I/O plus state | folding, DCE, duplication, speculation, rematerialization |
| Straight-line discharge | each primitive; one/two roots; local/external; read-only/mutated | unresolved root; partial replay on validation failure |
| Condition | then/else/both/neither writes; different write counts; two roots | reference branch result; mismatched root order |
| While | condition reads; body writes; zero/one/many iterations; nested condition | condition write; reference carry/result; escaping iteration allocation |
| Scan | zero/nonzero steps; state plus per-step output; nested condition | reference sequence/output; capture alias ambiguity |
| Nested call | argument/capture read/write; two-level call; caller observes state | argument-plus-capture alias; reference result; inconsistent summary |
| Partial evaluation | discharged local refs with known/mixed inputs | trace-time external mutation; split unresolved state chain |
| Batching | discharged local function equals per-example loop | external/mapped/captured ref; replicated write; lane conflict |
| JVP/VJP | local read/overwrite/swap/accumulate; condition/bounded while/scan; pure oracle | external ref; tangent ref; custom-rule ref; silent zero derivative; unbounded while transpose |
| Rematerialization | discharged local result matches baseline | preserved mutation recomputed or duplicated |
| XLA lowering | local refs; external state; aliases with zero-space index shifts | surviving ref; out-of-range/wrong alias; unsupported dynamic/sharded ref |
| XLA runtime | consecutive calls; retained initializer/read snapshots; distinct roots sharing initial storage; unique/shared buffer; captures; read lease before mutation | duplicate holder; pre/post-handoff failure; async failure; chained earlier failure; stale callback/generation; stale donated state |
| Persistence | new-schema round trip; replacement compatibility | old/corrupt schema; duplicate alias; type/sharding mismatch |
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

- [ ] Inspect `git diff` and classify every changed file by the phase's declared ownership.
- [ ] Run targeted tests before broad crate tests.
- [ ] Search all `ArrayIrType`/`ArrayIrValue` exhaustive matches after adding the variant and inspect every remaining
      array/dimension-only assumption.
- [ ] Verify no old `AtomType`, `ProgramType`, or parallel reference-universe scaffolding was introduced.
- [ ] Verify source rendering distinguishes all semantics-bearing reference/view metadata. Renderings back the
      debug-assertions transform-cache determinism recheck (`transforms.rs:591-618`) and rendered-program test
      assertions; production cache keys are argument-based (`ErasedTransformArguments`), StableHLO-text-based
      (`XlaPersistentKeyV5`), and dispatch-signature-based (`XlaDispatchKey`) — reference ABI metadata must enter
      the persistent key schema as a new `XlaPersistentKeyV6`, not the rendering.
- [ ] Verify invalid programs fail before mutation, replay, or backend lowering.
- [ ] Verify ordinary StableHLO contains no reference types or operations.
- [ ] Inspect translated/optimized HLO for exact input/output alias configuration when external refs land.
- [ ] Run each potentially expensive command with a 300-second timeout.

Expected command progression:

- [ ] `cargo fmt --check`
- [ ] `cargo test -p ryft-core --lib`
- [ ] `cargo test -p ryft-macros --lib` and `cargo test -p ryft-macros-tests` when derive contracts change
- [ ] `cargo test -p ryft-xla --lib`
- [ ] `cargo test -p ryft --lib`
- [ ] `cargo test -p ryft-mlir --lib` when function attributes or Mosaic wrappers change
- [ ] `cargo test -p ryft-pjrt --lib` when execution/donation APIs change
- [ ] `git diff --check`

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
      capture lifting into Phase 5 (Phase 4 rejection is temporary); chose a separate stateful call method as the
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
      processing, with a conservative quarantine policy for immediate execute errors without a fence; chose the
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
      unresolved state before direct interpretation (with an end-to-end rejection test); added the compilation-path
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
      `contains_unresolved_state` and `contains_unresolved_references`, used at dispatch, the
      compilation preflight, and both module-lowering entries, with pure pass-through and forwarded-capture tests;
      moved the dispatch-time reference rejection ahead of every mesh/option constraint; aligned eager binding with
      `Program::effects` by excluding dormant rule regions (differentiation owns rule-region state); relaxed
      `reference_semantics` to `Cow<'_, _>` so future payload-borrowing operations need not clone; and fixed the
      remaining documentation (state may alternatively be handled by a state-aware backend; the import/inline rename
      consequence is unreachable now that constants are sealed out; the `XlaReferenceConstant` TODO moved out of
      public rustdoc). Known accepted gap, unchanged: the central partial-evaluation gate for `fold_or_residualize`
      remains Phase 2 work, so the branch must not be described as generally transform-safe until it lands —
      higher-order all-known partial evaluation can still execute hidden state.
- [x] 2026-08-16 review pass 9 (external feedback, 6 findings applied): added
      `RegionRef::contains_effect_in_closure` — a closure-wide scan that descends into dormant rule regions at every
      nesting depth — and used it to guard the fused `RegionRef::jvp` replay at entry (covering `Program::jvp`,
      `jvp_shared`, and linearization, whose all-zero shortcut previously staged primals unguarded) and to deepen the
      `DifferentiationContext::bind` guard (nested rule regions were previously invisible through sealed effects),
      with direct `Program::jvp` and pure-program rule-region-hidden-state tests; made `BatchingContext::lift` apply
      the constant-storability contract so lifted reference holders cannot ride through batching as replicated
      batches; unified the XLA dormant-rule policy on artifact-wide rejection (eager binding now scans rule regions
      too, matching the ordinary-XLA unresolved-artifact checks, with an eager custom-JVP rule-region test; later phases
      may
      relax this with an executable-region analysis); added staged-boundary reference checks before sharding
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

Unchecked implementation items above remain future execution work. Completed items record code and verification that
landed with the corresponding phase summary.
