# Ryft References: Architecture and Implementation Plan

**Status:** proposed architecture; no implementation has begun

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

- `ArrayIrType` currently contains only `Array` and `Dimension`; adding a third member requires a broad but mostly
  compiler-guided exhaustive-match audit.
- `ArrayIrType::Identity` is `DimensionVariable`. Reference referents may contain those dimension identities, but
  runtime reference identity must not become a `Type::Identity`.
- `Effects` is a small `Copy` bitset with three global classes. It cannot encode dynamic resource identities.
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

- `ReferenceType<T>` implements `Type` by delegating identity traversal/renaming, compatibility, and refinement to
  `T`, while keeping reference/non-reference kind separation in the enclosing composite universe.
- A small generic `ReferenceTypeRefinements<T>` wrapper delegates complete-signature refinement to `T::Refinements` if
  the `Type` contract cannot reuse `T::Refinements` directly.
- For `ReferenceType<ArrayType>`, the referent supplies data type, shape, layout, sharding, memory, and dimension
  identities.
- Do not duplicate `Memory` or other array metadata in `ReferenceType<ArrayType>`.
- Do not store resource identity, allocation identity, capture identity, or SSA identity in the type.
- Do not add a `ReferenceKind` until at least two implemented behaviors require it. A speculative flag bag would make
  compatibility, serialization, and transforms harder without adding semantics.
- `ReferenceType<T>` is compatible with and refined by only another `ReferenceType<T>` whose referent satisfies the
  corresponding `T` relation. An enclosing composite universe never treats it as compatible with the referent itself.
- Reference/Reference identity traversal and refinement delegate through the referent, using the existing shared
  `ArrayTypeRefinements` logic. All cross-kind pairs fail.
- A reference handle is not a numeric scalar or complex value, even when it refers to a scalar or complex array.
  `Type::is_scalar()` and `Type::is_complex()` return `false`.

For the first XLA slice, mutation requires exact physical referent compatibility. Dynamic refinement may remain valid
for type checking, but external input/output aliasing is admitted only after physical shape, layout, sharding, and
dynamic-extent compatibility have been proven.

### 5.2 Runtime value and identity

Introduce a reusable `Reference<V: Value>` carrying a stable `ReferenceId` and an opaque shared holder for the current
value. `ArrayIrValue<A>` stores `Reference<A>`. The exact synchronization primitive is an implementation choice, but
the semantics are fixed:

- Cloning the internal value aliases the same resource; it does not copy the array contents.
- Equality, hashing, and display use stable reference identity, never mutable contents.
- No public program operation compares reference identities.
- A read returns an immutable snapshot. Later writes may reuse storage only when doing so cannot mutate any retained
  snapshot; otherwise eager execution or the backend must copy or use copy-on-write protection.
- `new_reference(value)` does not invalidate or make later mutations observable through the initializer `value`.
  Likewise, two distinct roots initialized from storage-sharing values remain logically independent. Physical reuse
  must copy-protect whenever either invariant would otherwise be violated.
- The declared referent type remains invariant for the reference's lifetime. A future dynamic-shape policy may permit
  different concrete runtime refinements within that declaration, but the fixed-shape MVP does not.
- Frozen, failed, or otherwise invalid state produces an explicit error rather than panicking or returning stale data.
- The holder is an opaque `Parameter` leaf. Parameter traversal exposes its identity-bearing handle, never its mutable
  array contents as ordinary child parameters.
- Reference identity-renaming support, called by `ArrayIrValue::rename_type_identities`, must keep holder contents and
  referent metadata consistent; a metadata-only rename that would violate that relationship is invalid.
- Mutable external references are not literal constants and are not serialized as payload data. They enter through
  arguments or typed captures whose invocation supplies the holder.

The staged resource identity is not the runtime `ReferenceId`. Within a program it is derived from the canonical SSA
root: an entry reference input, a captured reference, or the result of a local reference allocation. Invocation-time
identity validation connects external roots to actual holders.

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
| `ReferenceWriteOperation` | `(Reference, Array) -> ()` | Replace the current value. |
| `ReferenceSwapOperation` | `(Reference, Array) -> Array` | Return the old value and install the new value. |
| `ReferenceAddUpdateOperation` | `(Reference, Array) -> ()` | Ordered elementwise accumulation. |
| `FreezeReferenceOperation` | `Reference -> Array` | Return final state and invalidate the entire alias family. |

`set` may be public convenience syntax for `write` or discarded `swap`, but the IR should have whichever primitive set
makes effects, interpretation, and discharge clearest. Additive update remains distinct because future kernel lowering
and differentiation need to distinguish accumulation from replacement.

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
view. Start with indexing/slicing and reuse Ryft's canonical array indexing and gather/scatter descriptors rather than
inventing a second indexing language.

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
  result. Discharge drops its final current state; eager execution invalidates the alias family before the region is
  released.
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
keeps existing simplification, liveness, partial-evaluation gates, control-flow summaries, and rematerialization logic
safe.

Do not make `Effects` carry dynamic resource IDs. It should remain a small operation-level classification used by
existing passes.

### 6.2 Precise semantic contract

Add a separate backend-neutral reference semantics/access contract. Its exact Rust API should be prototyped against
one allocation, one read, and one view before finalizing, but it must express at least:

```text
ReferenceOperationSemantics
    root definitions: output -> NewRoot
    aliases: output -> AliasOf(input)
    accesses: input -> Read | Write | ReadWrite | Accumulate | Freeze
    view mapping: optional ordered transform stack
```

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
- freeze is local, unique, consuming, and followed by no use;
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
   reference inputs are classified from public parameter positions.
3. Run reference analysis and discharge on the lifted open program.
4. Return a binding recipe that maps each logical external-reference slot back to its original capture or public
   argument position.
5. At interpretation/execution, snapshot or transactionally extract the holder's current array according to the slot's
   access disposition, then install any hidden final state back into that same holder.

Nested closed calls must be normalized through the same capture-lifting rule before their bodies are discharged. The
resulting ordinary program may contain array inputs/captures, but no `CaptureReference` whose type is a reference and
no rewritten capture table containing mutable array contents. This keeps `ClosedProgram`'s capture type/value invariant
intact and gives core discharged interpretation and XLA compilation one boundary contract.

### 8.2 Result contract

Discharge belongs in `ryft-core`. It returns a reference-free program and logical external-state metadata:

```rust
pub struct DischargedReferenceProgram<O> {
    program: Program<...>,
    public_output_count: usize,
    external_states: Vec<DischargedReferenceState>,
}

pub struct DischargedReferenceState {
    slot: usize,
    source: ReferenceSource,
    source_parameter_index: usize,
    discharged_input_index: usize,
    access: ExternalReferenceAccess,
    final_state_output_index: Option<usize>,
    referent_type: ArrayType,
}

pub enum ExternalReferenceAccess {
    ReadOnly,
    Mutated,
}
```

Names and generic details remain subject to implementation review. The contract is:

- all indices are logical flattened program indices, never PJRT/XLA physical indices;
- original public outputs retain exactly their structure and order;
- hidden final states for mutated external roots follow public outputs in canonical external-root order;
- local roots create no external state slots;
- read-only external roots retain an input slot but have no hidden final-state output;
- mutated external roots have exactly one hidden final-state output;
- compilation and runtime metadata never contain process-local `ReferenceId` values;
- a successful result contains no reference types, operations, values, accesses, or `OrderedState` effects.

### 8.3 Straight-line rewrite

Maintain one current immutable array SSA value per root in program order:

```text
new_reference(x)      -> current[root] = x
read(root)            -> current[root]
write(root, x)        -> current[root] = x
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
- Reference-free input is either returned unchanged or handled by one documented idempotent path.
- Simplification after discharge may optimize the pure state chain; simplification before discharge retains every
  unresolved state access in order.

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
the same parent-binding pattern used by array-to-dimension operations. Reference methods bind composite operations and
return arrays or no value. References cannot cross the public function boundary.

This delivers the core programming model without redesigning retained compiled-function argument types.

### Stage B: eager public holders and captured references

Expose a controlled reference holder with identity, invalidation, and sequencing semantics. Capturing it in a compiled
closure makes the call externally stateful. Captures remain typed side-table values, not literal program constants.

### Stage C: explicit external reference arguments

Add a heterogeneous compiled-function facade over `ArrayIrType`/`ArrayIrValue<Array>` or another typed boundary that
preserves existing array-only APIs. Do not weaken the current array projection merely to accept references.

Externally stateful calls return a completion-bearing wrapper conceptually like `ReferenceExecution<Output>`, even
when `Output` has no leaves. A synchronous convenience method awaits it. The existing output-only execution path may
remain for ordinary pure calls, but it must not discard the only fence/error carrier for an externally stateful call.

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
    physical_input_index: Option<usize>,
    physical_output_index: Option<usize>,
    referent_type: ArrayType,
}
```

Physical indices are computed only after composing with:

- capture/public-input flattening;
- zero-space logical value erasure;
- hidden bounded-dynamic extent inputs;
- public, hidden state, and hidden dynamic-extent result ordering.

Never derive physical indices by adding counts or assuming one logical value maps to one physical value.

Initially admit only these combinations:

- `(physical_input = Some, physical_output = None, access = ReadOnly)`;
- `(physical_input = Some, physical_output = Some, access = Mutated)`.

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
8. Split public results, hidden state results, and hidden dynamic extents.
9. Install all mutated hidden states, with their readiness events and generation/dependency information, before
   reporting successful submission.
10. Register the execution fence as a read lease on every read-only holder so a later donating mutation waits or
    dependency-chains until the read is finished.
11. Reconstruct and expose only public outputs through the completion-bearing stateful-call wrapper.

Required failure semantics:

- Before the PJRT irreversibility boundary, every extracted mutated state is restored.
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

Persist canonical logical/physical mappings and referent types, never runtime holder IDs. On load, validate slot order,
index ranges, alias injectivity, public/hidden output counts, physical compatibility, and signature arity before an
executable can be invoked.

### 11.5 XLA support order

1. Local fixed-shape references, which disappear before XLA lowering.
2. One fixed-shape unsharded external reference.
3. Multiple unique external references.
4. Captured references.
5. Shared-snapshot copy protection and asynchronous sequencing.
6. Persistent executable round trips and replacement compatibility.
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
3. Add scoped uninitialized scratch allocation and non-escape validation.
4. Complete view metadata with offsets, strides, alignment, layout, address space, and access mode.
5. Distinguish ordered accumulation from atomic/commutative operations.
6. Add target-neutral synchronization operations.
7. Lower preserved accesses to Mosaic/Triton/other target memory operations.
8. Keep outer kernel-call inputs/outputs array-based with explicit alias metadata.

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

- [ ] Confirm names for `ReferenceType<T>`, `Reference<V>`, `ReferenceId`, and the six whole-array operations.
- [ ] Record the supported and unsupported matrix from this document in public-facing design documentation.
- [ ] Prototype one `Reference` type projection, one `new_reference`, and one `read` through `ArrayIrOperation` without
      migrating unrelated operations.
- [ ] Prototype the operation-level reference semantic descriptor against allocation, read, and an aliasing view.
- [ ] Confirm that the existing composite operation derive can express native mixed reference operations without a
      new projected `ReferenceOperation` family.
- [ ] Decide whether reference-free discharge is idempotent or explicitly one-shot.
- [ ] Specify error types for lifecycle, root resolution, transform rejection, discharge, and backend ABI failures.

**Exit criterion:** the minimum type/operation/access contract is demonstrated without a parallel type universe or a
broad trait-solver migration.

### Phase 1: Add the array IR reference member

- [ ] Add `crates/ryft-core/src/programs/references.rs` with generic `ReferenceType<T: Type>` and
      `ReferenceTypeRefinements<T>` as required by the `Type` contract, re-exported through the programs facade and
      crate root.
- [ ] Add `ArrayIrType::Reference` and checked `From`/`TryFrom` projections.
- [ ] Extend display, identity traversal/renaming, compatibility, refinement, scalar/complex classification, and
      `ArrayIrTypeRefinements`.
- [ ] Route reference/referent dimension refinements through the existing shared `ArrayTypeRefinements` state.
- [ ] Add generic `Reference<V: Value>` and `ReferenceId` in the same core references module.
- [ ] Implement `Typed<Type = ReferenceType<V::Type>>`, identity-based traits, opaque `Parameter` leaf behavior, and
      the storage/lifecycle primitives without adding a standalone `Value` implementation.
- [ ] Add `ArrayIrValue::Reference(Reference<A>)` and
      `ValueProjection<ReferenceType<ArrayType>, Projected = Reference<A>>`.
- [ ] Implement identity-based display/equality/hash and alias-preserving internal clone behavior.
- [ ] Update exports and documentation that currently describe Array IR as containing only arrays and dimensions.
- [ ] Audit every exhaustive type/value match in core, XLA, tests, and derives. Each site must intentionally support,
      reject, or remain unreachable after a verified pipeline boundary.
- [ ] Add tests for cross-kind failures, projection, dynamic referent identity refinement, aliasing clones, invalidation,
      and a dimension variable shared across array, dimension, and reference leaves.

**Exit criterion:** references can be stored, typed, projected, renamed, and diagnosed without changing ordinary
array/dimension behavior or being accepted by numeric operations.

### Phase 2: Add effects, reference semantics, and validation

- [ ] Add `Effect::OrderedState` and update effect iteration, ordering, rendering, and tests.
- [ ] Add the operation-level root-definition/alias/access/view semantic contract.
- [ ] Add whole-array reference operations as native `ArrayIrOperation` variants.
- [ ] Mirror them in `XlaOperation` so XLA staging can carry them before discharge.
- [ ] Give every unresolved state access the coarse ordered-state effect.
- [ ] Implement `ReferenceAnalysis` over entry inputs, captures, allocations, aliases, and nested regions.
- [ ] Implement the dedicated static validator.
- [ ] Implement invocation-time duplicate-holder validation separately.
- [ ] Ensure simplification retains unused writes and preserves read/write order.
- [ ] Ensure default partial evaluation and rematerialization cannot execute or duplicate unresolved state.
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

**Exit criterion:** the whole-array reference language has one observable meaning in eager and staged execution.

### Phase 4: Implement straight-line discharge

- [ ] Add a core discharge module and result metadata types.
- [ ] Integrate discharge after `ClosedProgram::to_program_with_lifted_captures` and return the canonical
      capture/public-holder binding recipe without rewriting concrete capture tables.
- [ ] Validate the complete program before constructing output.
- [ ] Track one immutable current array per root.
- [ ] Rewrite each whole-array operation according to the state-passing semantics.
- [ ] Eliminate local create/freeze state and preserve mutated external state as hidden outputs.
- [ ] Preserve non-state effects, source locations, identities, layout, sharding, and memory.
- [ ] Verify that successful output contains no reference artifacts or ordered-state effect.
- [ ] Add deterministic rendering/cache tests.
- [ ] Add property tests over short generated straight-line state programs against eager and hand-written oracles.

**Exit criterion:** straight-line local and external reference programs produce a deterministic reference-free core
program plus complete logical state metadata.

### Phase 5: Extend discharge through regions and calls

- [ ] Thread canonical state through condition branches and joins.
- [ ] Thread body-mutated state through while carries.
- [ ] Allow while conditions to read current state and reject condition writes.
- [ ] Thread scan-mutated state as carries, separate from per-step values.
- [ ] Rewrite nested calls and substitute callee root mappings into callers.
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
- [ ] Add guards proving no reference reaches generic AD/batching representational rules.
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

### Phase 8: Add one external XLA reference

- [ ] Define and implement the eager/XLA holder state machine, including ready, pending, poisoned, and frozen/invalid
      states as applicable.
- [ ] Extend logical and physical executable-signature metadata.
- [ ] Record `ReadOnly` versus `Mutated` disposition and represent physical input/output indices as optional mappings;
      reject erased physical reference inputs in this phase.
- [ ] Derive physical indices only after zero-space and dynamic-extent mappings are complete.
- [ ] Emit and verify `tf.aliasing_output` on the correct entry argument.
- [ ] Implement hidden final-state result splitting and holder installation.
- [ ] Construct donation flags in logical ABI order: ordinary capture/dimension/read-only reference `false`, ordinary
      public array from the user flag, and mutated reference internally `true` with uniqueness downgrade.
- [ ] Mark the PJRT handoff irreversibility boundary and implement restore-before-handoff versus
      install-or-poison-after-handoff behavior.
- [ ] Add a completion-bearing stateful-call wrapper so zero-output calls cannot lose execution errors.
- [ ] Add internal mutated-reference donation while preserving copy-protection fallback; never donate read-only slots.
- [ ] Define pre-handoff, execute-call, post-submission reconstruction, asynchronous-execution, and dropped-completion
      behavior.
- [ ] Keep hidden results out of public reconstruction.
- [ ] Start with one static, unsharded, device-memory external reference.

**Exit criterion:** consecutive compiled calls observe mutation; retained snapshots remain valid; failures leave the
holder restored or explicitly poisoned; semantics remain correct when physical alias reuse does not occur.

### Phase 9: Complete external/captured runtime integration

- [ ] Support multiple unique holders with stable lock order and logically atomic pending-state installation.
- [ ] Track read-only execution leases and require later mutations to wait or dependency-chain them before donation.
- [ ] Use generation-safe cumulative dependency/error state so a failure in an earlier pending mutation cannot be
      hidden by a later chained call or stale callback.
- [ ] Serialize conflicting same-holder calls while allowing safe read overlap and independent-holder concurrency.
- [ ] Support captured references with reference-specific internal donation policy.
- [ ] Add explicit external reference arguments through a heterogeneous boundary without breaking array-only APIs.
- [ ] Reject public/capture duplicate identities before state extraction.
- [ ] Add state ABI metadata to lowering, compiled programs, cache identity, persistence, and executable replacement.
- [ ] Bump and validate the persistent executable schema.
- [ ] Add concurrency, capture, cache round-trip, corruption, replacement, and failure tests.

**Exit criterion:** public and captured references have the same observable state semantics, concurrent use is defined,
and persisted/replaced executables cannot carry mismatched state ABIs.

### Phase 10: Add indexed views and transformations

- [ ] Define the composable reference-view representation.
- [ ] Add indexing/slicing views and bounds/type validation.
- [ ] Lower reads/writes/swaps/additive updates through canonical array slicing/gather/scatter/update operations.
- [ ] Preserve base-root identity and composed access mappings.
- [ ] Add reshape/transpose/bitcast only with explicit layout, overlap, and alias proofs.
- [ ] Add base/view mutual-observation, composed-view, overlap, invalidation, and discharge equivalence tests.

**Exit criterion:** views provide enough backend-neutral address semantics for ordinary discharge and future kernel
lowering without creating independent resources.

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
- [ ] Remove temporary scaffolding and compatibility layers.
- [ ] Reassess which APIs should remain experimental until external AD and kernel semantics mature.

**Exit criterion:** the supported contract is comprehensible without reading implementation code and does not imply
support for deferred aliases, transforms, dynamic shapes, or kernel operations.

## 15. Likely change surface

### `ryft-core`

- `src/programs/references.rs`: generic `ReferenceType<T>`, `ReferenceTypeRefinements<T>`, `Reference<V>`, and
  `ReferenceId`.
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
- composite lowering: targeted error for surviving references.
- `src/arrays.rs` or a focused reference module: only narrowly scoped state extraction/install support that cannot
  remain generic over `Reference<A>`.

### `ryft-mlir`, `ryft-pjrt`, and `ryft-xla-sys`

No broad initial changes are expected:

- existing function argument attributes should carry `tf.aliasing_output`;
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
- [ ] Verify source rendering distinguishes all semantics-bearing reference/view metadata because renderings are used
      as structural fingerprints.
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

### Milestone C: external/captured references

Phases 8-9: holder lifecycle, hidden state ABI, aliases, donation, async failure/concurrency, caching, and persistence.

Estimated ownership:

- `ryft-core`: 35-45%
- `ryft-xla`: 50-60%
- `ryft-mlir`/`ryft-pjrt`/`ryft-xla-sys`: 0-10%, only for demonstrated gaps

### Milestone D: views and expanded XLA support

Phases 10-11: indexing/view composition, sharding, dynamic extents, zero-space policy, and distribution.

Ownership depends on which representation constraints surface, but remains primarily core analysis/discharge plus XLA
ABI/runtime integration.

### Milestone E: Pallas-readiness checkpoint

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

No implementation items above are complete merely because this plan exists. Checkboxes in Phases 0-13 are execution
work and must be updated, with phase summaries, only when implementation begins.
