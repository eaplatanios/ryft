# Ryft IR Provenance Scopes Plan

## Status

- [ ] Await review and approval before implementation.
- [ ] Reconcile this plan with the completed `CoordinateBasisOperation` inlining before changing code.

## Objective

Add persistent, hierarchical, non-semantic provenance to Ryft IR instructions. The first producer will annotate the
ordinary primitive operations that construct a dense Jacobian coordinate basis with the scope name
`ryft.differentiation.coordinate_basis`.

The same mechanism must support future program visualization, readable diagnostic renderings, transform provenance,
and backend source locations without introducing marker operations, executable regions, or dependencies from compiler
correctness onto diagnostic annotations.

This plan includes complete propagation through Ryft program rebuilding and transformations, followed by lowering to
MLIR `NameLoc` and `FusedLoc` locations.

## Design Decisions

### Provenance is diagnostic, not semantic

Provenance must never change:

- operation type inference, effects, reference semantics, differentiation, batching, or interpretation;
- program inputs, outputs, atoms, instruction order, attached regions, or runtime results;
- the existing canonical `Program` rendering or semantic program fingerprint;
- whether an optimization or transformation is legal.

Compiler behavior must remain correct if all provenance is removed. Code that needs to recognize a coordinate basis for
correctness or optimization must match the ordinary primitive computation or use a semantic operation; it must not
inspect `ryft.differentiation.coordinate_basis`.

A direct consequence worth stating explicitly: `Unknown` is always a correct value. A transform that drops provenance
produces a worse diagnostic, never a wrong program. Full propagation coverage in step 5 is therefore a quality bar
rather than a correctness gate, and any propagation gap discovered later is a small follow-up rather than a bug with
blast radius.

### Provenance is stored on each instruction

Do not represent scopes with `scope_begin`/`scope_end` instructions or contiguous instruction ranges. Transformations
may delete, duplicate, move, or merge instructions, so range markers would become invalid.

Do not use `TagOperation`. It annotates a value through a real identity operation and already participates in semantic
transform behavior. A provenance scope instead describes the origin of an instruction without adding an SSA value or
data dependency.

Do not use an owning `Program` scope-ID arena in the initial implementation. Ryft frequently clones, imports, splices,
and independently seals regions. Program-global IDs would require remapping at every such boundary and would make a
borrowed `RegionRef` insufficient to copy a region faithfully.

Store provenance as a small immutable value backed by shared nodes. Cloning an instruction or region then preserves
provenance without remapping, while all instructions emitted under one scope can share the same allocation.

Proposed model:

```rust
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ProvenanceScope(Arc<str>);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Provenance(Option<Arc<ProvenanceNode>>);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ProvenanceNode {
    Scope {
        scope: ProvenanceScope,
        origin: Provenance,
    },
    Fused {
        origins: Arc<[Provenance]>,
    },
}
```

`Unknown` is the `None` case of the internal representation rather than a node variant. Every instruction carries a
provenance field and `Unknown` dominates in practice, so it must be allocation-free: `Option<Arc<ProvenanceNode>>` is
niche-optimized to pointer size, `Provenance::unknown()` allocates nothing, and no `LazyLock` singleton is needed. The
representation stays private behind the constructors and accessors below.

The public API should expose constructors and traversal methods rather than the node representation. In particular:

- `ProvenanceScope::new(name)` creates a named scope.
- `Provenance::unknown()` represents instructions with no recorded origin.
- `Provenance::scope(scope, origin)` attaches one named scope above an existing origin.
- `Provenance::fused(origins)` represents a generated instruction with multiple source origins.
- `Provenance::scope_path()` supports program renderers and visualizers without exposing storage internals. It returns
  the scope names from the outermost `Scope` node down to the first non-`Scope` node; for `Unknown` and for a `Fused`
  root it returns an empty path, and fused constituents are reached through `Provenance::origins()`.
- `Provenance::origins()` supports recursive visualization of fused origins.

Keep scope names as one stable, fully-qualified string in the first version. Do not introduce a parallel display name
or an open-ended attribute map until a concrete consumer requires it. A visualizer may recognize well-known names and
render friendly labels, while unknown user-defined names remain directly displayable. This keeps the initial API small
and maps exactly onto MLIR `NameLoc`.

Normalize fused provenance at construction:

- discard `Unknown` when another origin is present;
- flatten nested `Fused` nodes;
- remove structurally duplicate origins while preserving first-occurrence order;
- return `Unknown` for zero origins and the sole origin directly for one origin.

### Scope state belongs to contexts, not a global thread-local

Add scope handling to Ryft's `Context` abstraction. The active provenance must travel with cloned context handles just
like tracing and transform state; it must not depend on the current operating-system thread or a process-global
subscriber.

Proposed context API:

```rust
fn current_provenance(&self) -> Provenance;

fn with_provenance_scope<R, F: FnOnce() -> R>(&self, scope: ProvenanceScope, function: F) -> R;

fn with_provenance_origin<R, F: FnOnce() -> R>(&self, origin: Provenance, function: F) -> R;
```

`with_provenance_scope` is the producer-facing API. `with_provenance_origin` is the replay/transform API used to say
that newly generated instructions originated from an existing instruction. Keeping the two operations distinct avoids
conflating a logical scope with one or more source instructions.

The closure form is preferred to exposing manual push/pop calls. Its implementation must use an RAII restoration guard
so the previous state is restored on ordinary return, error return, and panic unwinding. The guard must not hold a
`RefCell` borrow while `function` runs.

Internally, active context state should keep an ordered scope stack separately from its source origin. Entering a scope
pushes one frame. Entering an origin temporarily combines it with any existing origin through normalized
`Provenance::fused`, while leaving ambient scopes in place. `current_provenance()` folds the scope stack over that
combined origin. Thus a top-level replay preserves its source exactly, an explicit scope remains outside generated
work, and a nested replay can retain both its enclosing and nested source origins without treating operand dependencies
as origins. For example, entering `outer` and then `inner` produces `Scope(outer, Scope(inner, origin))`, and
`scope_path()` returns `[outer, inner]`.

All three methods are required `Context` methods without default bodies, mirroring `is_eager`. A defaulted method
would let a wrapper context silently drop provenance by forgetting to delegate, and the only safety net would be a
manual audit; required methods make the compiler enumerate every context implementation instead. Terminal eager and
test-only contexts implement the explicit no-op behavior (`Unknown` plus running the closure directly). Every wrapper
context delegates to its parent unless it owns a new staging boundary. This includes projected, batching,
differentiation, partial-evaluation, reference-discharge, and other recursive transform contexts.

`TracingContext` and `NestedTracingContext` own active provenance state shared by their clones. Keep this state separate
from the `RefCell<ProgramBuilder>` so entering a scope does not hold a builder borrow while instructions are staged.
A newly created nested tracing context should seed its origin from its parent context's current provenance, then own an
independent scope state for the nested program.

### Backend locations and compilation caches

Extend the existing `Program::render` function with an explicit enum-valued rendering-mode argument. Semantic mode
must continue to omit provenance, while provenance mode emits it deterministically. `Display` must call semantic mode,
so existing canonical strings remain unchanged; do not introduce a separate `display_with_provenance` function or
display-wrapper API.

There is a necessary distinction between semantic cache identity and compiled-artifact identity:

- Semantic canonicalization and semantic comparison must ignore provenance.
- A cache containing lowered MLIR, HLO, or a compiled executable cannot reuse an artifact carrying different emitted
  provenance without returning stale names to visualizers and profilers.

Characterize the actual caches before designing new machinery. There are currently few relevant boundaries: the eager
per-operation compile cache, whose single-operation programs are staged outside any user scope and therefore
effectively always carry `Unknown` provenance, and the jit executable cache, whose identity is documented as deriving
from the complete lowered computation. The decisive question is whether that key derivation prints MLIR locations,
because MLIR's default printer omits locations unless debug-info printing is enabled:

- If the key already includes locations, provenance participates in cache identity automatically and no extra
  machinery is needed.
- If it does not, prefer switching that one key derivation to a location-inclusive rendering over introducing and
  threading a parallel provenance-only fingerprint through cache keys.
- Only if neither approach works should a separate, stable, deterministic provenance fingerprint be added to the
  affected keys.

In all cases, do not put provenance into operation rendering or the semantic canonical program representation.
Document that exact backend provenance can reduce compiled-artifact cache reuse. If this cost later matters, add an
explicit compilation option that strips provenance before lowering and consequently removes it from cache identity;
do not silently return incorrectly labeled cached artifacts.

## Implementation Plan

### 1. Add the core `Provenance` model

- [ ] Add a provenance module under `crates/ryft-core/src/programs` and re-export its intended public surface through
      the normal `programs` facade and downstream crate-root facade.
- [ ] Implement immutable `Unknown`, `Scope`, and normalized `Fused` provenance as described above, with the
      allocation-free `Option<Arc<ProvenanceNode>>` representation for `Unknown`.
- [ ] Add the small `ProvenanceScope` name wrapper. Keep construction infallible and preserve user-provided names
      verbatim; document that names should be non-empty and that the `ryft.` prefix is reserved for framework-owned
      scopes rather than adding validation machinery that has no correctness role.
- [ ] Define `COORDINATE_BASIS_PROVENANCE_NAME` as
      `"ryft.differentiation.coordinate_basis"` in the differentiation-owned module rather than the generic programs
      module.
- [ ] Add concise `Display` output suitable for annotated diagnostics, while keeping full recursive inspection
      available through accessors.
- [ ] Add exact unit tests for unknown, one scope, nested scopes, fusion normalization, duplicate removal, equality,
      hashing, and display.

### 2. Attach provenance during staging

- [ ] Add a `provenance: Provenance` field and accessor to `Instruction<O>`. Confirm that `Instruction<O>` continues
      to derive only `Clone` and `Debug`; adding the field must not introduce a `PartialEq` or `Hash` implementation
      that would make provenance observable to semantic equality.
- [ ] Keep an ergonomic constructor for manually created instructions with `Unknown` provenance, and add an explicit
      provenance-preserving construction path used by builders and rebuilds.
- [ ] Change destructive decomposition such as `Instruction::into_parts` so provenance cannot be accidentally omitted;
      use a named parts structure if a five-element tuple would be unclear.
- [ ] Extend `ProgramBuilder` with an instruction insertion path that accepts explicit provenance. Preserve the existing
      insertion API as the `Unknown` convenience path for direct/manual builders.
- [ ] Add active provenance state and the three context methods described above as required `Context` methods without
      default bodies, so every existing implementation must be updated before the crate compiles.
- [ ] Update `StagingContext::stage_operation` to snapshot `current_provenance()` and attach it to the emitted
      instruction.
- [ ] Implement shared, unwind-safe scope state in `TracingContext` and `NestedTracingContext`; ensure cloned contexts
      observe the same active scope and independent traces do not leak scopes into each other.
- [ ] Implement the required methods on every `Context` implementation, using the resulting compile errors as the
      enumeration. Transform/projected wrappers delegate; terminal eager and test-only contexts implement and document
      the explicit no-op behavior.
- [ ] Add tests for nested scope order, clone sharing, restoration after `Result::Err`, restoration during panic
      unwinding, independent tracing contexts, nested tracing, and ordinary eager execution remaining unaffected.

### 3. Add diagnostic rendering and visualizer-facing traversal

- [ ] Add a public enum with explicit variants rather than a boolean, provisionally:

      ```rust
      #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
      pub enum ProgramRenderingMode {
          Semantic,
          WithProvenance,
      }
      ```

- [ ] Change the existing renderer to
      `Program::render(&self, formatter, indentation, mode: ProgramRenderingMode)`. Do not add a second public render or
      display-wrapper function.
- [ ] Make `Display` call `Program::render(..., ProgramRenderingMode::Semantic)`, preserving all existing `to_string()`
      output and canonical structural strings byte-for-byte for equivalent programs.
- [ ] Thread the mode through the private recursive instruction/region rendering helpers and every nested-program
      rendering path. Because `OperationFormatter::program` can render program-valued operation metadata, extend the
      existing operation-rendering/formatter contract with the same enum as needed rather than silently reverting
      nested programs to semantic mode. Update derive-generated operation renderers and downstream implementations in
      the same change.
- [ ] Require every direct caller of `Program::render` and operation rendering to choose a mode explicitly. Canonical
      fingerprints and semantic operation fields use `Semantic`; visualization, provenance assertions, and diagnostic
      dumps use `WithProvenance`.
- [ ] Render provenance per instruction, not as assumed-contiguous begin/end blocks. The first textual form should use a
      stable prefix or adjacent comment containing the complete scope path and fused origins.
- [ ] Expose enough read-only traversal for a future visualizer to group instructions by common scope ancestors while
      retaining non-contiguous membership and fused origins.
- [ ] Add exact rendering tests proving that canonical output is unchanged, diagnostic output contains nested paths,
      nested program-valued operation metadata receives the selected mode, and fused origins are deterministic.

### 4. Annotate coordinate-basis construction

- [ ] Wait for the ongoing `CoordinateBasisOperation` inlining to land, then wrap only the ordinary primitive
      construction in `DenseDifferentiableType::coordinate_basis` with
      `ryft.differentiation.coordinate_basis`.
- [ ] Do not reintroduce a coordinate-basis operation, marker value, wrapper region, or special backend lowering.
- [ ] Ensure validation that fails before construction does not emit partially scoped instructions. If some validation
      necessarily stages shape computations, decide explicitly whether those computations belong inside the scope and
      pin that choice in tests.
- [ ] Verify forward Jacobian input bases and reverse Jacobian output cotangent bases receive the same scope.
- [ ] Add an exact staged-program test asserting that every primitive belonging to basis construction carries the
      coordinate-basis scope and adjacent user computation does not.
- [ ] Add nested batching/differentiation regressions, including a rank-greater-than-one basis and nonzero coordinate
      offset, proving that one-to-many transformed primitive sequences retain the scope without reviving the removed
      operation.

### 5. Preserve and propagate provenance through all transformations

- [ ] Audit every `Instruction::new`, `Instruction::into_parts`, direct `Instruction` destructure, region clone,
      program rebuild, and `ProgramBuilder::add_instruction_unchecked` call in `ryft-core`, macro-generated code, and
      downstream crates. This audit is bounded and compiler-driven, not open-ended: as of this writing there are
      roughly thirty `Instruction::new` call sites, concentrated in `crates/ryft-core/src/programs`, plus isolated
      sites in shard-map lowering, rematerialization, and reverse differentiation, and the `into_parts` arity change
      surfaces every destructure as a compile error.
- [ ] Preserve provenance verbatim for structural relocation and identity rebuilds, including:
      - type-identity renaming and operation-family projection/unprojection;
      - borrowed and owned region import;
      - callee interning;
      - program splicing;
      - borrowing and consuming simplification;
      - subgraph extraction and region/program restructuring.
- [ ] At each source-instruction replay boundary, run the operation's transform rule inside
      `with_provenance_origin(source_instruction.provenance().clone(), ...)`.
- [ ] Apply that policy to batching, forward differentiation/JVP, reverse differentiation/VJP, linearization,
      transposition, partial evaluation, reference discharge, rematerialization, and nested region/callee replay.
- [ ] Use the following explicit propagation rules:
      - one source to one generated instruction: preserve the source provenance;
      - one source to many generated instructions: attach the source provenance to every generated instruction;
      - unchanged instruction copied structurally: copy provenance exactly;
      - newly synthesized transform scaffolding with no source instruction: use the active transform scope or
        `Unknown`, never infer provenance from operand data dependencies;
      - multiple source instructions intentionally merged into one: use `Provenance::fused` at the merge site;
      - deleted instructions: delete their provenance with them.
- [ ] Do not infer fused provenance by walking input operands. Dataflow dependency is not the same as instruction
      origin; a pass that performs a real many-to-one rewrite must provide the origins explicitly.
- [ ] Ensure transform-cache reuse cannot return provenance belonging to a different source region. Provenance remains
      excluded from semantic transformation decisions, but identity-rebuild cache adoption must require provenance to
      have been preserved exactly when the cached transformed artifact itself carries provenance.
- [ ] Add focused tests for every propagation class above, plus nested combinations such as batching over a
      differentiated program, differentiation through a batched program, partial evaluation of scoped work, attached
      regions, shared callees, simplification, and an explicit fused-origin test helper.
- [ ] Add macro integration tests wherever operation/program derive output reconstructs instructions, and run
      `ryft-macros-tests` after any derive contract changes.

### 6. Lower provenance to MLIR locations

- [ ] Add a conversion from Ryft `Provenance` plus the caller-provided base `LocationRef` to MLIR locations:
      - `Unknown` uses the base location;
      - `Scope { name, origin }` becomes `NamedLocationRef(name, lower(origin, base))`;
      - `Fused { origins }` becomes `FusedLocationRef` over the recursively lowered origins, using unit metadata unless
        a concrete metadata payload is introduced later.
- [ ] Reuse the existing `ryft-mlir` named/fused/unknown location wrappers; do not create parallel MLIR bindings.
- [ ] Change the StableHLO replay loops to derive an instruction-specific location before constructing each plain or
      shard-map lowerer. Composite lowerings already emit all constituent MLIR operations through the lowerer's shared
      location, so every StableHLO operation generated from one Ryft instruction should inherit that instruction's
      provenance automatically.
- [ ] Apply the same logic to nested inline regions, condition/loop bodies, shared callees, and manual-computation
      bodies. Function/module scaffolding may retain its existing base location unless it has program-level provenance.
- [ ] Preserve an existing file/line base location as the innermost child of named scopes rather than replacing it.
- [ ] Verify that later MLIR transformations preserve or fuse locations according to MLIR conventions; do not add a
      Ryft-specific StableHLO attribute when standard locations suffice.
- [ ] Add exact MLIR tests for unknown, single, nested, and fused provenance, including
      `ryft.differentiation.coordinate_basis` on every StableHLO operation produced by its scoped Ryft primitives.
- [ ] Add end-to-end XLA tests demonstrating that nested batching/differentiation scopes survive compilation and are
      visible in dumped MLIR/HLO metadata without affecting numeric results.

### 7. Audit fingerprints, caches, serialization, and overhead

- [ ] Characterize the actual cache boundaries (the eager per-operation compile cache, the jit executable cache, and
      any transform, MLIR/module, or persistent compilation cache) and determine for each whether its key derivation
      already includes MLIR locations.
- [ ] Prove with tests that canonical program rendering and semantic fingerprints remain unchanged by provenance.
- [ ] Make provenance participate in every cache key whose cached artifact contains lowered provenance, preferring a
      location-inclusive module rendering in the existing key derivation. Add a separate deterministic provenance
      fingerprint (never pointer identity or hash-map iteration order) only where a location-inclusive key is not
      possible. Verify that two semantically equal programs with different scope names cannot receive each other's
      labeled executable.
- [ ] Verify that region imports and common scope sharing remain cheap: entering one scope should allocate one shared
      provenance node, while staging each instruction should ordinarily clone only an `Arc`.
- [ ] Add a focused construction benchmark if profiling infrastructure already provides an appropriate home. Do not add
      a new benchmark framework solely for this feature.
- [ ] Document the cache-reuse tradeoff and the future opt-out design: stripping provenance before lowering may improve
      compiled-artifact cache sharing, but only an explicit mode may do so.

### 8. Documentation and verification

- [ ] Document `Provenance`, the context scope API, instruction attachment semantics, transform propagation rules, and
      MLIR lowering behavior with links to the precise
      [MLIR `NameLoc`](https://mlir.llvm.org/docs/Dialects/Builtin/#nameloc) and
      [`FusedLoc`](https://mlir.llvm.org/docs/Dialects/Builtin/#fusedloc) documentation.
- [ ] Document framework namespace ownership: built-in names use `ryft.<subsystem>.<concept>`, beginning with
      `ryft.differentiation.coordinate_basis`; user scopes should use their own namespace.
- [ ] Explain that Rust `tracing` integration, if added later, is an optional telemetry bridge and never the persistent
      IR source of truth.
- [ ] Run targeted provenance, builder, program, tracing, batching, differentiation, partial-evaluation, and lowering
      tests with a 300-second timeout per command.
- [ ] Run `cargo test -p ryft-core --lib`, `cargo test -p ryft-macros-tests`, and `cargo test -p ryft-xla --lib`, each
      with the repository-required 300-second timeout.
- [ ] Run workspace formatting checks and targeted Clippy/check commands appropriate to the touched crates.
- [ ] Review the complete diff and targeted searches for lost provenance construction paths, accidental canonical
      rendering changes, the old `autodiff` namespace, and any reintroduced `CoordinateBasisOperation` reference.

## Why Not Use Rust `tracing` as the Implementation?

Rust [`tracing` spans](https://docs.rs/tracing/latest/tracing/struct.Span.html) are a useful API and terminology
reference, but they are not the right persistent representation for compiler IR provenance:

- A span exists for runtime diagnostics through the active subscriber. It may be disabled by filtering, whereas IR
  provenance must be recorded deterministically whenever a program is staged.
- A [`tracing_core::span::Id`](https://docs.rs/tracing-core/latest/tracing_core/span/struct.Id.html) identifies a span
  only within one subscriber. It cannot be serialized into a `Program`, cloned across independent compiler passes, or
  used as a stable program-local identity.
- The current span is ambient execution state. Ryft programs and regions outlive the call stack that built them and are
  later imported, replayed, differentiated, batched, simplified, fused, and lowered on potentially different threads.
- `tracing` does not define compiler-specific one-to-many or many-to-one provenance propagation. Ryft still needs the
  immutable `Scope`/`Fused` model and explicit pass policies described above.
- Entered span guards have thread/async constraints that are irrelevant overhead for a synchronous, context-owned IR
  annotation mechanism.
- Making `ryft-core` IR fidelity depend on a global telemetry subscriber would couple the core compiler model to an
  optional observability stack.

Use a `tracing::Span::in_scope`-like closure API and RAII restoration pattern, but implement it directly on Ryft
contexts. A future optional integration can enter a Rust `tracing` span whenever Ryft enters a provenance scope, or emit
events while lowering, so application telemetry and IR visualization share names. The `Provenance` stored on
instructions must remain authoritative even when no subscriber is installed.

## Non-Goals

- Reintroducing semantic coordinate-basis recognition after the operation is inlined.
- Treating provenance scopes as scheduling, fusion, optimization, or execution boundaries.
- Encoding arbitrary provenance attributes before a concrete visualizer/profiler requirement exists.
- Building the full `Program` visualizer in this change.
- Capturing Rust source file/line/call stacks in the first version. The representation and MLIR base-location chaining
  should leave room for that separate feature.
- Adding runtime profiler ranges; those may bridge to Rust `tracing` later but are distinct from persistent IR origin.

## Risks and Mitigations

- **Accidental provenance loss during rebuilds:** make destructive instruction decomposition include provenance, audit
  all constructors, and add transform-by-transform exact tests.
- **Incorrect stale labels from cache hits:** separate semantic identity from artifact provenance identity and ensure
  provenance participates in every lowered-artifact cache key, preferring a location-inclusive key rendering over a
  parallel provenance fingerprint.
- **Scope leakage across traces or errors:** use context-owned shared state plus an unwind-safe RAII restoration guard.
- **Silently non-delegating context wrappers:** make the three provenance methods required `Context` methods without
  defaults, so the compiler enumerates every implementation.
- **Transform correctness depending on names:** keep provenance inaccessible to operation semantics and state the
  non-semantic invariant in public documentation and tests.
- **Memory growth:** share immutable nodes with `Arc`, normalize fused origins, and avoid per-instruction scope-path
  vectors.
- **Misleading visual grouping after instruction motion:** annotate each instruction independently and never infer
  membership from contiguous ranges.
- **Cross-cutting API churn:** stage the work in the numbered order above and use compiler errors plus targeted searches
  to enumerate every instruction reconstruction and context wrapper.

## Review Record

Populate this section while implementing the approved plan.

- Step 1: pending.
- Step 2: pending.
- Step 3: pending.
- Step 4: pending.
- Step 5: pending.
- Step 6: pending.
- Step 7: pending.
- Step 8: pending.
