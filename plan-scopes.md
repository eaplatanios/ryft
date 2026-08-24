# Ryft IR Provenance Scopes Plan

## Status

- [ ] Await review and approval before implementation.
- [x] Reconcile this plan with the completed `CoordinateBasisOperation` inlining: the operation is gone from the
      repository and step 4 now states the completed prerequisite instead of waiting on it.

## Objective

Add persistent, hierarchical, non-semantic provenance to Ryft IR instructions. The first producer will annotate the
ordinary primitive operations that construct a dense Jacobian coordinate basis with the nested framework scopes
`ryft::differentiation::coordinate_basis`.

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
inspect the `ryft::differentiation::coordinate_basis` scopes.

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
- `Provenance::is_unknown()` returns whether the provenance is `Unknown`, giving renderers, lowerers, serialization
  boundaries, and tests a direct way to make that distinction without reconstructing it from traversal results.
- `Provenance::scope(scope, origin)` attaches one named scope above an existing origin.
- `Provenance::fused(origins)` represents a generated instruction with multiple source origins.
- `Provenance::as_scope()` returns the outermost scope together with the origin recorded below it for a `Scope` root,
  and `None` for `Unknown` and `Fused` roots. Walking it repeatedly recovers the complete outermost-first scope path
  and the provenance below the chain, so no separate path or below-chain accessors are needed, and traversal
  allocates nothing.
- `Provenance::as_fused()` returns the fused constituents for a `Fused` root and `None` otherwise.
- Together, `is_unknown()`, `as_scope()`, and `as_fused()` mirror the three provenance shapes one-to-one and let a
  visualizer traverse the complete tree without a node-view API. Returning `None` from the shape accessors instead of
  using `self` as a sentinel prevents accidental non-terminating traversal.

Keep scope names as single path segments and express namespacing structurally by nesting scopes: the framework
annotation is the nested path `ryft::differentiation::coordinate_basis` — three scope levels, not one dotted name.
This makes the scope tree and the namespace tree one mechanism (visualizers group framework work under `ryft` and
`differentiation` structurally, with no display-time name parsing), keeps names identifier-like so they render bare,
and lowers to a chain of nested MLIR `NameLoc`s in the JAX name-stack style. Do not introduce a parallel display name
or an open-ended attribute map until a concrete consumer requires it. A visualizer may recognize well-known scope
paths and render friendly labels, while unknown user-defined names remain directly displayable.

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

Internally, active context state keeps an ordered scope stack, an optional source origin, and a cached fully composed
`Provenance`. The cache is recomputed only when entering or leaving a scope or origin; `current_provenance()` and
instruction staging clone the cached `Arc` and never rebuild nodes. Note that because the nodes are immutable and the
outermost scope sits at the root, recomputing the cache at a scope transition rebuilds the chain above the change
point, which is O(scope depth) node allocations per transition. That is accepted: scope nesting is shallow in
practice, and the performance requirement is precisely that *composition occurs only at scope and origin transitions,
never per staged instruction*. Do not complicate the representation (for example, innermost-first storage with
reversal at traversal time) to shave transition cost unless profiling ever shows it matters.

Composition semantics are defined precisely as follows and must be pinned by tests before implementation:

- Entering a scope pushes one frame. Entering an origin records the origin together with the scope-stack depth at
  entry. The composed provenance folds only the frames pushed *after* the innermost origin entry over that origin;
  frames that were already active when the origin was entered are the transform's ambient context, not part of the
  instruction's origin. This makes the 1→1 propagation rule automatic: replaying a source instruction under an
  ambient scope that its provenance already records preserves the source provenance exactly, with no double wrap,
  while synthesized scaffolding staged with no origin still receives the ambient transform scope folded over
  `Unknown`.
- Entering an origin while another origin is active *fuses* the new origin with the provenance *composed at that
  moment*: the outer origin with all frames pushed after it already folded. It neither uses nor replaces the raw outer
  origin. Concretely, after entering origin `A`, then scope `S`, then origin `B`, the composition is
  `fused[Scope(S, A), B]`; fusing `B` with raw `A` would silently drop `S`. The new entry records the scope-stack
  depth at `B`'s entry, so only frames pushed after `B` fold over the fused node. Leaving `B` restores the previous
  origin, depth, and cached composition (here, back to `Scope(S, A)`). A nested replay thus retains both its enclosing
  and nested source origins without treating operand dependencies as origins.
- `Provenance::scope(scope, origin)` performs no deduplication or common-prefix factoring; normalization exists only
  in `Provenance::fused`. Visualizers may factor common scope prefixes at display time, but the stored representation
  stays purely structural.
- For example, with an origin entered (or seeded) first and scopes `outer` then `inner` entered afterwards, the
  composed provenance is `Scope(outer, Scope(inner, origin))` and walking `as_scope()` yields `outer` then `inner`
  before reaching `origin`.

All three methods are required `Context` methods without default bodies, mirroring `is_eager`. A defaulted method
would let a wrapper context silently drop provenance by forgetting to delegate, and the only safety net would be a
manual audit; required methods make the compiler enumerate every context implementation instead. Terminal eager and
test-only contexts implement the explicit no-op behavior (`Unknown` plus running the closure directly). Every wrapper
context delegates to its parent unless it owns a staging boundary — that is, unless it owns a `ProgramBuilder` that
instructions are emitted into. Delegating wrappers include projected, batching, differentiation, reference-discharge,
and other recursive transform contexts that stage through an inner context.

`PartialEvaluationContext` is a provenance-owning boundary, not a delegating wrapper. It owns the residual
`ProgramBuilder` and emits residual instructions into it directly, so delegating provenance reads to its known-side
parent (often a terminal eager context returning `Unknown`) would erase source provenance from every residual
program. It must own shared active-provenance state exactly like the tracing contexts, `Rc`-shared across clones
alongside its builder, seeded from the parent context's current provenance at construction.

`TracingContext` and `NestedTracingContext` own active provenance state shared by their clones. Keep this state separate
from the `RefCell<ProgramBuilder>` so entering a scope does not hold a builder borrow while instructions are staged.
A newly created nested tracing context should seed its origin from its parent context's current provenance at depth
zero of its own scope stack, then own an independent scope state for the nested program. Under the depth-based
composition rule above, later replaying the nested program's instructions into the parent under the same ambient
scope preserves each instruction's provenance exactly instead of wrapping or fusing the shared scope twice.

### Backend locations and compilation caches

Extend the existing `Program::render` function with an explicit enum-valued rendering-mode argument. Semantic mode
must continue to omit provenance, while provenance mode emits it deterministically. `Display` must call semantic mode,
so existing canonical strings remain unchanged. `Program::render` takes a `std::fmt::Formatter` that callers cannot
construct directly, so external `WithProvenance` consumers reach it through `std::fmt::from_fn`; a dedicated
`Program::display(mode)` adapter was added and then removed again (post-landing review decision) because its only
consumers were tests, which inline `std::fmt::from_fn` instead. Reintroduce the single mode-parameterized adapter
when the first production consumer (e.g., the visualizer) lands; never add mode-specific named functions such as
`display_with_provenance` or `to_string_with_mode`.

There is a necessary distinction between semantic cache identity and compiled-artifact identity:

- Semantic canonicalization and semantic comparison must ignore provenance.
- A cache containing lowered MLIR, HLO, or a compiled executable cannot reuse an artifact carrying different emitted
  provenance without returning stale names to visualizers and profilers.

This question is resolved against the current code rather than left open. The relevant boundaries are the eager
per-operation compile cache, whose single-operation programs are staged outside any user scope and therefore
effectively always carry `Unknown` provenance, and the jit executable cache, whose persistent key already embeds the
complete textual StableHLO module (`XlaPersistentKeyV6::stable_hlo`, built by `xla_compilation_key` in
`crates/ryft-xla/src/experimental/domains.rs`). However, lowering currently serializes the module with
`module.to_string()`, and `OperationPrintingFlags::default()` in `crates/ryft-mlir/src/operations/printing.rs` has
`enable_debug_information: false`, so MLIR locations do not enter the key today.

The fix is therefore a serialization change, not a parallel fingerprint: serialize scoped modules with
`enable_debug_information = true` and `pretty_print_debug_information = false` (the pretty debug form is documented
as unparsable and must not feed cache keys or reloadable artifacts). Enable this only when the program carries
non-`Unknown` provenance, so ordinary programs keep byte-identical StableHLO text, existing snapshots, and existing
cache keys. "Carries provenance" must be detected transitively, not by scanning top-level instructions, which would
miss attached regions, shared callees, and nested program-valued operation metadata: lowering accumulates a
`has_provenance` flag whenever it constructs an instruction-specific (non-base) location, including in every
recursive lowering path, and that flag selects the serialization mode after lowering completes. A separate
deterministic provenance fingerprint is a fallback to be added only if location-inclusive serialization proves
unworkable at some cache boundary.

In all cases, do not put an instruction's provenance into its operation's semantic `Operation::render` output or the
semantic canonical program representation. The contextual `Program` renderer owns the instruction-level suffix.
Nested program-valued metadata always renders semantically: the originally planned defaulted
`Operation::render_with_mode` (with derive-dispatcher and `Box<O>` forwarding, and a mode parameter on
`OperationFormatter::program`) was implemented and then removed by a post-landing review decision, because no
operation renders program-valued metadata and no use case is foreseen; reintroduce that exact design if one ever
appears. Document that
exact backend provenance can reduce compiled-artifact cache reuse. If this cost later matters, add an explicit
compilation option that strips provenance before lowering and consequently removes it from cache identity; do not
silently return incorrectly labeled cached artifacts.

## Implementation Plan

### 1. Add the core `Provenance` model

- [x] Add a provenance module under `crates/ryft-core/src/programs` and re-export its intended public surface through
      the normal `programs` facade and downstream crate-root facade.
- [x] Implement immutable `Unknown`, `Scope`, and normalized `Fused` provenance as described above, with the
      allocation-free `Option<Arc<ProvenanceNode>>` representation for `Unknown`.
- [x] Add the small `ProvenanceScope` name wrapper. Keep construction infallible and preserve user-provided names
      verbatim; document that names should be non-empty and that the `ryft.` prefix is reserved for framework-owned
      scopes rather than adding validation machinery that has no correctness role.
- [x] Deferred to step 4: the coordinate-basis scope-path definition is not added until its first consumer exists, so
      no unused global definition lands ahead of the coordinate-basis annotation work.
- [x] Add concise `Display` output suitable for annotated diagnostics, while keeping full recursive inspection
      available through accessors.
- [x] Add exact unit tests for unknown, one scope, nested scopes, fusion normalization, duplicate removal, equality,
      hashing, and display.

### 2. Attach provenance during staging

- [x] Add a `provenance: Provenance` field and accessor to `Instruction<O>`. Confirm that `Instruction<O>` continues
      to derive only `Clone` and `Debug`; adding the field must not introduce a `PartialEq` or `Hash` implementation
      that would make provenance observable to semantic equality.
- [x] Keep an ergonomic constructor for manually created instructions with `Unknown` provenance, and add an explicit
      provenance-preserving construction path used by builders and rebuilds.
- [x] Change destructive decomposition such as `Instruction::into_parts` so provenance cannot be accidentally omitted;
      use a named parts structure if a five-element tuple would be unclear.
- [x] Superseded: `ProgramBuilder` first gained a separate `add_instruction_with_provenance` beside the untouched
      `add_instruction`, and a post-landing review merged them into one
      `add_instruction(operation, regions, inputs, provenance: Option<Provenance>)`, with `None` recording unknown
      provenance at direct/manual call sites.
- [x] Add active provenance state and the three context methods described above as required `Context` methods without
      default bodies, so every existing implementation must be updated before the crate compiles.
- [x] Update `StagingContext::stage_operation` to snapshot `current_provenance()` and attach it to the emitted
      instruction.
- [x] Implement the cached-composition active state: recompute the composed provenance only on scope/origin entry and
      exit, and make `current_provenance()` and staging clone the cached `Arc`.
- [x] Implement shared, unwind-safe scope state in `TracingContext` and `NestedTracingContext`; ensure cloned contexts
      observe the same active scope and independent traces do not leak scopes into each other.
- [x] Give `PartialEvaluationContext` its own `Rc`-shared active-provenance state seeded from its parent, as described
      above; residual instructions emitted into its builder must snapshot that state, never the known-side parent's.
- [x] Implement the required methods on every `Context` implementation, using the resulting compile errors as the
      enumeration. Transform/projected wrappers delegate; terminal eager and test-only contexts implement and document
      the explicit no-op behavior.
- [x] Add tests for nested scope order, clone sharing, restoration after `Result::Err`, restoration during panic
      unwinding, independent tracing contexts, nested tracing, and ordinary eager execution remaining unaffected.
- [x] Add tests pinning the composition semantics: nested tracing seeded inside scope `outer` followed by replay
      inside `outer` produces no double wrap; `with_provenance_origin` nested inside another origin fuses both
      origins; the exact `enter A, enter S, enter B` sequence composes `fused[Scope(S, A), B]` and restores
      `Scope(S, A)` after leaving `B`; scopes entered before an origin do not fold over it while scopes entered after
      it do; and no automatic common-prefix factoring occurs in `Provenance::scope`.

### 3. Add diagnostic rendering and visualizer-facing traversal

- [x] Add a public enum with explicit variants rather than a boolean, provisionally:

      ```rust
      #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
      pub enum ProgramRenderingMode {
          Semantic,
          WithProvenance,
      }
      ```

- [x] Change the existing renderer to
      `Program::render(&self, formatter, indentation, mode: ProgramRenderingMode)`. Do not add a second public render
      function.
- [x] Superseded: `Program::display(mode)` was added as the lightweight `Display` adapter and later removed by a
      post-landing review decision because its only consumers were tests, which now inline `std::fmt::from_fn`
      around `Program::render`. The adapter returns when the first production `WithProvenance` consumer lands.
- [x] Make `Display` call `Program::render(..., ProgramRenderingMode::Semantic)`, preserving all existing `to_string()`
      output and canonical structural strings byte-for-byte for equivalent programs.
- [x] Superseded: the mode was threaded through the private recursive instruction/region rendering helpers, and the
      operation-level plumbing (a defaulted `Operation::render_with_mode` delegating to `render`, forwarding overrides
      on `Box<O>` and the derive-generated dispatchers, and a mode parameter on `OperationFormatter::program`) was
      implemented as planned and then removed again by a post-landing review decision: no operation renders
      program-valued metadata, so the entire surface was a forward-compatibility guard with no consumer. Nested
      program-valued metadata now always renders semantically (`OperationFormatter::program` documents this and the
      reintroduction condition). `Operation::render` remains semantic-only.
- [x] Require every program-rendering path to choose a mode explicitly through `Program::render`. Standalone calls to
      `Operation::render` are explicitly semantic-only and do not select a mode. Canonical fingerprints and semantic
      operation fields use `Semantic`; visualization, provenance assertions, and diagnostic dumps use
      `WithProvenance`.
- [x] Render provenance per instruction, not as assumed-contiguous begin/end blocks, using the following exact
      grammar, modeled on MLIR location syntax and appended to the instruction line:

      ```text
      suffix     ::= ""                                    // Unknown: no suffix at all.
                   | " ; " expression                      // Comment-style, LLVM-IR-like trailing annotation.
      expression ::= segment ("::" segment)*               // Outermost scope first.
      segment    ::= name                                  // One scope level.
                   | "fused[" expression ("," " " expression)* "]"
      name       ::= <bare identifier> | <Rust string literal>
      ```

      The suffix is inserted immediately before the instruction statement's final newline. For instructions whose
      rendering spans multiple lines (for example, operations with attached regions), that means after the final
      closing bracket on the statement's last line, never inside the nested body. No `unknown` token exists in the
      grammar: `Unknown` renders as the absence of a suffix, and fused normalization already guarantees `Unknown`
      never appears as a constituent. A `fused[...]` segment only ever terminates a chain, because a scope above a
      fused origin ends the scope chain. Names render bare when they match `[A-Za-z_][A-Za-z0-9_]*` and otherwise
      render quoted and escaped deterministically as Rust string literals (the `{:?}` / `escape_debug` form), so
      arbitrary name content — including `::`, brackets, quotes, and newlines — can never corrupt the grammar, and
      construction stays infallible with no name validation. A scope literally named `fused` stays unambiguous
      because the keyword form is always followed by `[`. The format is readable and unambiguous; round-tripping
      (parsing provenance back from renderings) is an explicit non-goal in the first version, but the grammar must
      not preclude adding it later.
- [x] Expose enough read-only traversal for a future visualizer to group instructions by common scope ancestors while
      retaining non-contiguous membership and fused origins.
- [x] Add exact rendering tests proving that canonical output is unchanged, diagnostic output contains nested paths,
      fused origins are deterministic, name escaping handles quotes and newlines, and one multiline instruction (an
      operation with an attached region) places the suffix after the final closing bracket. The
      mode-propagation-into-nested-metadata test was removed together with `Operation::render_with_mode`.

### 4. Annotate coordinate-basis construction

- [x] Prerequisite complete: the `CoordinateBasisOperation` inlining has landed — the operation no longer exists in
      the repository, and `DenseDifferentiableType::coordinate_basis`
      (`crates/ryft-core/src/differentiation/types.rs`) already stages ordinary primitives. Wrap only that primitive
      construction with the nested scopes `ryft::differentiation::coordinate_basis`.
- [x] Define the differentiation-owned coordinate-basis scope path — the nested `ryft`, `differentiation`, and
      `coordinate_basis` scopes, e.g., as a small constructor or constants colocated with
      `DenseDifferentiableType` — rather than in the generic programs module (deferred from step 1 so the
      definition is not added before its consumer exists).
- [x] Do not reintroduce a coordinate-basis operation, marker value, wrapper region, or special backend lowering.
- [x] Ensure validation that fails before construction does not emit partially scoped instructions. If some validation
      necessarily stages shape computations, decide explicitly whether those computations belong inside the scope and
      pin that choice in tests.
- [x] Verify forward Jacobian input bases and reverse Jacobian output cotangent bases receive the same scope.
- [x] Add an exact staged-program test asserting that every primitive belonging to basis construction carries the
      coordinate-basis scope and adjacent user computation does not.
- [x] Add nested batching/differentiation regressions, including a rank-greater-than-one basis and nonzero coordinate
      offset, proving that one-to-many transformed primitive sequences retain the scope without reviving the removed
      operation.

### 5. Preserve and propagate provenance through all transformations

- [x] Audit every `Instruction::new`, `Instruction::into_parts`, direct `Instruction` destructure, region clone,
      program rebuild, and `ProgramBuilder::add_instruction_unchecked` call in `ryft-core`, macro-generated code, and
      downstream crates. This audit is bounded and compiler-driven, not open-ended: as of this writing there are
      roughly thirty `Instruction::new` call sites, concentrated in `crates/ryft-core/src/programs`, plus isolated
      sites in shard-map lowering, rematerialization, and reverse differentiation, and the `into_parts` arity change
      surfaces every destructure as a compile error.
- [x] Preserve provenance verbatim for structural relocation and identity rebuilds, including:
      - type-identity renaming and operation-family projection/unprojection;
      - borrowed and owned region import;
      - callee interning;
      - program splicing;
      - borrowing and consuming simplification;
      - subgraph extraction and region/program restructuring.
- [x] At each source-instruction replay boundary, run the operation's transform rule inside
      `with_provenance_origin(source_instruction.provenance().clone(), ...)`.
- [x] Apply that policy to batching, forward differentiation/JVP, reverse differentiation/VJP, linearization,
      transposition, partial evaluation, reference discharge, rematerialization, and nested region/callee replay.
- [x] Use the following explicit propagation rules:
      - one source to one generated instruction: preserve the source provenance;
      - one source to many generated instructions: attach the source provenance to every generated instruction;
      - unchanged instruction copied structurally: copy provenance exactly;
      - newly synthesized transform scaffolding with no source instruction: use the active transform scope or
        `Unknown`, never infer provenance from operand data dependencies;
      - multiple source instructions intentionally merged into one: use `Provenance::fused` at the merge site;
      - deleted instructions: delete their provenance with them.
- [x] Do not infer fused provenance by walking input operands. Dataflow dependency is not the same as instruction
      origin; a pass that performs a real many-to-one rewrite must provide the origins explicitly.
- [x] Treat reverse-mode cotangent accumulation as an explicit many-to-one merge. The `accumulate` helper in
      `crates/ryft-core/src/differentiation/reverse.rs` stages an add combining cotangent contributions that
      originate from transposing *different* source instructions, so wrapping each transpose rule in
      `with_provenance_origin` is not sufficient to label that add. Track the provenance of each accumulated
      contribution alongside its atom in the adjoint table, and stage every accumulation add with `Provenance::fused`
      over the contributing provenances.
- [x] Ensure transform-cache reuse cannot return provenance belonging to a different source region. Provenance remains
      excluded from semantic transformation decisions, but identity-rebuild cache adoption must require provenance to
      have been preserved exactly when the cached transformed artifact itself carries provenance.
- [x] Add focused tests for every propagation class above, plus nested combinations such as batching over a
      differentiated program, differentiation through a batched program, partial evaluation of scoped work, attached
      regions, shared callees, simplification, and an explicit fused-origin test helper.
- [x] Add macro integration tests wherever operation/program derive output reconstructs instructions, and run
      `ryft-macros-tests` after any derive contract changes.

### 6. Lower provenance to MLIR locations

- [x] Add a conversion from Ryft `Provenance` plus the caller-provided base `LocationRef` to MLIR locations:
      - `Unknown` uses the base location;
      - `Scope { name, origin }` becomes `NamedLocationRef(name, lower(origin, base))`;
      - `Fused { origins }` becomes `FusedLocationRef` over the recursively lowered origins, with *no* metadata
        attribute. Unit metadata is an actual metadata payload and changes the printed location, so it must not be
        used as a stand-in for "none".
- [x] Reuse the existing `ryft-mlir` named/fused/unknown location wrappers; do not create parallel MLIR bindings.
      `Context::fused_location` currently requires a metadata attribute, so add an optional/no-metadata construction
      path to the existing wrapper (passing a null attribute handle through `mlirLocationFusedGet`), matching how
      `fused_metadata` already models absent metadata as `None`.
- [x] Change the StableHLO replay loops to derive an instruction-specific location before constructing each plain or
      shard-map lowerer. Composite lowerings already emit all constituent MLIR operations through the lowerer's shared
      location, so every StableHLO operation generated from one Ryft instruction should inherit that instruction's
      provenance automatically.
- [x] Apply the same logic to nested inline regions, condition/loop bodies, shared callees, and manual-computation
      bodies. Function/module scaffolding may retain its existing base location unless it has program-level provenance.
- [x] Preserve an existing file/line base location as the innermost child of named scopes rather than replacing it.
- [x] Verify that later MLIR transformations preserve or fuse locations according to MLIR conventions; do not add a
      Ryft-specific StableHLO attribute when standard locations suffice.
- [x] Add exact MLIR tests for unknown, single, nested, and fused provenance, including the nested
      `ryft::differentiation::coordinate_basis` `NameLoc` chain on every StableHLO operation produced by its scoped
      Ryft primitives.
- [x] Add end-to-end XLA tests demonstrating that nested batching/differentiation scopes survive compilation and are
      visible in dumped MLIR/HLO metadata without affecting numeric results.

### 7. Audit fingerprints, caches, serialization, and overhead

- [x] Confirm the cache-boundary inventory from the design section (the eager per-operation compile cache, the jit
      executable cache keyed through `XlaPersistentKeyV6::stable_hlo`, and any transform, MLIR/module, or persistent
      compilation cache added since) and check that no other boundary caches lowered artifacts.
- [ ] Prove with tests that canonical program rendering and semantic fingerprints remain unchanged by provenance.
- [x] Serialize modules lowered from programs carrying non-`Unknown` provenance with `enable_debug_information = true`
      and `pretty_print_debug_information = false`, so locations enter `XlaPersistentKeyV6::stable_hlo` and cache
      identity automatically; keep ordinary programs on the existing location-free serialization so their StableHLO
      text, snapshots, and cache keys remain byte-identical. Detect "carries provenance" via the lowering-accumulated
      `has_provenance` flag described in the design section (set whenever any recursive lowering path constructs an
      instruction-specific location), never via a top-level instruction scan. Add a separate deterministic provenance
      fingerprint (never pointer identity or hash-map iteration order) only if location-inclusive serialization proves
      unworkable at some boundary. Verify that two semantically equal programs with different scope names cannot
      receive each other's labeled executable, and that a provenance-free program's key is unchanged from before this
      feature.
- [x] Verify that region imports and common scope sharing remain cheap: composition work is O(scope depth) and happens
      only at scope/origin transitions, staging each instruction clones only the cached `Arc`, and no composition work
      happens per staged instruction.
- [x] Add a focused construction benchmark if profiling infrastructure already provides an appropriate home. Do not add
      a new benchmark framework solely for this feature.
- [x] Document the cache-reuse tradeoff and the future opt-out design: stripping provenance before lowering may improve
      compiled-artifact cache sharing, but only an explicit mode may do so.

### 8. Documentation and verification

- [x] Document `Provenance`, the context scope API, instruction attachment semantics, transform propagation rules, and
      MLIR lowering behavior with links to the precise
      [MLIR `NameLoc`](https://mlir.llvm.org/docs/Dialects/Builtin/#nameloc) and
      [`FusedLoc`](https://mlir.llvm.org/docs/Dialects/Builtin/#fusedloc) documentation.
- [x] Document framework namespace ownership: the root scope name `ryft` is reserved, and built-in scopes nest as
      `ryft::<subsystem>::<concept>`, beginning with `ryft::differentiation::coordinate_basis`; user scopes should
      use their own namespace.
- [x] Explain that Rust `tracing` integration, if added later, is an optional telemetry bridge and never the persistent
      IR source of truth.
- [x] Run targeted provenance, builder, program, tracing, batching, differentiation, partial-evaluation, and lowering
      tests with a 300-second timeout per command.
- [x] Run `cargo test -p ryft-core --lib`, `cargo test -p ryft-macros-tests`, and `cargo test -p ryft-xla --lib`, each
      with the repository-required 300-second timeout.
- [x] Run workspace formatting checks and targeted Clippy/check commands appropriate to the touched crates.
- [x] Review the complete diff and targeted searches for lost provenance construction paths, accidental canonical
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
- **Incorrect stale labels from cache hits:** separate semantic identity from artifact provenance identity; lowered
  provenance enters cache identity through location-inclusive module serialization (debug information enabled,
  pretty-printing disabled) feeding the existing `stable_hlo` key field, with a parallel provenance fingerprint only
  as a fallback.
- **Provenance erased at the partial-evaluation boundary:** `PartialEvaluationContext` owns its residual builder, so
  it owns active-provenance state like the tracing contexts instead of delegating reads to its known-side parent.
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

- Step 1: done. Added `crates/ryft-core/src/programs/provenance.rs` with `ProvenanceScope` (verbatim single-segment
  `Arc<str>` name, infallible construction, documented reserved `ryft` root scope) and `Provenance` over the private
  `Option<Arc<ProvenanceNode>>` representation (allocation-free `unknown()`, `is_unknown()`, structural `scope()`,
  normalized `fused()` with unknown-discarding/flattening/first-occurrence dedup/zero-and-one collapsing, and the
  shape-mirroring `as_scope()`/`as_fused()` accessors, which replaced an earlier
  `scope_path()`/`origin()`/`origins()` trio after review). `Display` renders `::`-separated scope chains with
  bare-or-quoted names (bare for `[A-Za-z_][A-Za-z0-9_]*`, Rust-string-literal escaping otherwise), terminal
  `fused[...]` segments, and a standalone `unknown` token; this revised the initially landed MLIR-style
  `"outer"("inner")` form after review, and namespacing became structural (nested scopes such as
  `ryft::differentiation::coordinate_basis`) instead of dotted names. Re-exported through the `programs` facade and
  the crate-root facade (the `ryft` crate glob-re-exports `ryft_core::*`, so no change was needed there). The
  coordinate-basis scope-path definition was deferred to step 4 so it is not added before its consumer exists.
  Verified with seven exact unit tests covering unknown, one scope, nested scopes (including scope-over-fused and no
  prefix factoring), fusion normalization, duplicate removal, equality, hashing, and display (bare, quoted/escaped,
  `::` chains, and terminal fused segments); `cargo test -p ryft-core --lib` passes and `cargo fmt -p ryft-core` is
  clean.
- Step 2: done. `Instruction<O>` gained a `provenance` field (still deriving only `Clone`/`Debug`), a `provenance()`
  accessor, a consuming `with_provenance` attach path, and a five-element `into_parts`. `ProgramBuilder` gained
  a provenance-accepting insertion path (post-landing review merged the initial two-method design into one
  `add_instruction` taking `provenance: Option<Provenance>`, migrating roughly 1,400 call sites to pass `None`).
  `ProvenanceState` (in `programs/provenance.rs`) implements the cached, depth-based composition with RAII
  restoration guards and no `RefCell` borrow held while closures run. The three provenance methods are required
  `Context` methods; all thirteen implementations were updated (terminal eager `EagerContext`/`XlaDomain` plus three
  test contexts are documented no-ops; `ProjectedContext`, `BatchingContext`, `DifferentiationContext`,
  `ReferenceDischargeContext`, and the test `ExplicitContext` delegate; `TracingContext`/`NestedTracingContext`/
  `PartialEvaluationContext` own `Rc`-shared state, the latter two seeded from their parent).
  `StagingContext::stage_operation` and the partial-evaluation residual emission snapshot `current_provenance()`
  before borrowing the builder. State-level tests pin the composition semantics (depth rule, `enter A, enter S,
  enter B` fusing to `fused[Scope(S, A), B]` with restoration, no prefix factoring, error/panic restoration), and
  tracing-level tests cover snapshotting, clone sharing, origin replay, independent traces, eager no-ops, and
  nested-trace seeding. Full `cargo test -p ryft-core --lib` passes and the whole workspace type-checks.
- Step 3: done. Added `ProgramRenderingMode` (in `programs/programs.rs` above `Program`, re-exported through the `programs` facade
  and crate root), changed `Program::render` to take the mode, made `Display` render `Semantic`, and added the
  a `Program::display(mode)` adapter that a post-landing review removed again (test-only consumers inline
  `std::fmt::from_fn` around `Program::render`; the adapter returns with the first production consumer). The instruction renderer emits the comment-style ` ; ...` suffix immediately before
  each statement's final newline (after the closing bracket for multiline instructions) and threads the mode through
  region recursion. The operation-level mode plumbing (`Operation::render_with_mode`, its `Box<O>` and
  derive-dispatcher forwards, the `ryft-macros-tests` stand-ins, and the mode parameter on
  `OperationFormatter::program`) was implemented and then removed by post-landing review: it had no production
  consumer, so nested program-valued metadata always renders semantically, with the reintroduction condition
  documented on `OperationFormatter::program`. Exact tests cover unchanged canonical output, nested scope paths,
  deterministic fused origins with escaped names, and the multiline suffix placement after the final closing
  bracket. `ryft-core` and `ryft-macros-tests` suites pass.
- Step 4: done (transform regressions folded into step 5's test wave). Added the private
  `with_coordinate_basis_provenance` helper colocated with `DenseDifferentiableType` in `differentiation/types.rs`,
  entering the nested `ryft` → `differentiation` → `coordinate_basis` scopes, and wrapped exactly the primitive
  value construction in `ArrayType::coordinate_basis` with it. All validation preceding construction is host-side
  and stages nothing, so no partially scoped instructions can appear; the trailing type check is also host-side.
  Forward Jacobian input bases and reverse Jacobian output cotangent bases both flow through this one method
  (`jacobian.rs` call sites), so both receive the same scopes. Extended the exact staged-program test to assert that
  all seven staged basis primitives carry the nested scopes while an adjacent user multiplication stays unknown. No
  coordinate-basis operation, marker value, wrapper region, or special lowering was reintroduced.
- Step 5: done. The two `interpret_with` replay boundaries in `interpretation.rs` bind every replayed instruction
  inside `with_provenance_origin(source.provenance())`, which the depth rule turns into automatic 1-to-1
  preservation (no ambient double wrap) and 1-to-many attribution; the same wrap was applied to the jvp loop (both
  dispatch paths), linearization, partial-evaluation `inline_region`, all three batching region replays, reference
  discharge, and the transposition rule loop (via a cloned context handle because the rule borrows the context
  mutably). Structural relocations preserve provenance verbatim: `map_operations`, `simplified`/`into_simplified`/
  `filtered`/`into_filtered` and their subgraph-extraction workers, `rename_type_identities` (covering
  `with_instantiated_type_identities` and `intern_callee`), `splice_program`, capture lifting, rematerialization's
  primal-slice and reconstruction relocations, shard-map rebuild/pruning, and the pullback's known-producer
  materialization. Reverse-mode cotangent accumulation tracks `(AtomId, Provenance)` per adjoint slot and stages
  every accumulation add with `Provenance::fused` over its contributions. Synthesized scaffolding (rematerialization
  storage operations, disconnected-input zeros) deliberately stays unknown rather than inferring origins from
  operands. Transform-cache adoption semantics are documented in `programs/transforms.rs` (provenance is outside the
  semantic recheck by design; identity rebuilds now preserve provenance so adopted artifacts stay consistent).
  Focused tests cover replay preservation (incl. the ambient no-double-wrap case), simplification/filtering, splice,
  jvp 1-to-many, partial-evaluation residuals, transposition with fused accumulation, batching 1-to-many, and a
  staged forward+reverse Jacobian regression (rank-2 basis, nonzero offset) proving the coordinate-basis scopes
  survive the internal jvp/batching pipeline while user computation stays unknown. `ryft-macros-tests` passes with
  the derive dispatcher forwarding `render_with_mode`.
- Step 6: done. `ryft-mlir`'s `Context::fused_location` now takes `Option<A>` metadata (a null `MlirAttribute`
  handle models "none", matching `named_location`'s null child and `fused_metadata`'s `None`), with a test pinning
  the metadata-free `loc(fused[...])` printed form. In `ryft-xla`, `CollectiveLoweringState::instruction_location`
  converts `Provenance` recursively onto a caller-provided base location (unknown → base; scope → nested `NameLoc`;
  fused → metadata-free `FusedLoc`) and accumulates the module-scoped `has_provenance` flag (an `Rc<Cell<bool>>`
  beside the channel allocator, so every recursive path shares it). The three replay closures that construct
  per-instruction lowerers derive instruction-specific locations, which nested inline regions, condition/loop/scan
  bodies, shared-callee bodies, and manual-computation bodies inherit through the lowerers' shared location;
  function/module scaffolding keeps the base location. An exact module test pins unknown/nested/fused location
  renderings, and an end-to-end CPU test proves the nested `ryft`→`differentiation`→`coordinate_basis` `NameLoc`
  chain survives lowering into `stable_hlo()` while numeric results are unchanged.
- Step 7: done (benchmark intentionally skipped: the workspace has no benchmark infrastructure to house one, and
  the plan forbids adding a framework for this feature alone). `serialize_lowered_module` prints modules with
  `enable_debug_information = true`, `pretty_print_debug_information = false`, and elision thresholds disabled
  exactly when the lowering-accumulated `has_provenance` flag is set, so locations enter
  `XlaPersistentKeyV6::stable_hlo` automatically while provenance-free modules keep byte-identical text (the entire
  pre-existing StableHLO snapshot suite passes unchanged, proving key stability). The end-to-end test verifies that
  two semantically equal programs with different scope names produce different `XlaCompilationKey`s and that a
  provenance-free program serializes with no `loc(` at all. Canonical-rendering invariance is pinned by the core
  rendering tests; the cached-composition overhead property is pinned by design (composition only at scope/origin
  transitions) and its tests. The cache-reuse tradeoff and the explicit strip-before-lowering opt-out design are
  documented on `serialize_lowered_module`, and transform-cache adoption semantics are documented in
  `programs/transforms.rs`.
- Step 8: done. Documentation lives with the code: `Provenance`/`ProvenanceScope`/`ProvenanceState` document the
  model, composition semantics (including the deliberate fused-model consequences for unknown nested origins and
  seeded boundaries), namespace ownership (reserved `ryft` root scope, `ryft::<subsystem>::<concept>`), the MLIR
  `NameLoc`/`FusedLoc` links, and the optional-telemetry-bridge status of any future Rust `tracing` integration;
  the `Context` methods document attachment and delegation semantics; `programs/transforms.rs` documents cache
  adoption; and `serialize_lowered_module` documents the cache-reuse tradeoff and strip-before-lowering opt-out.
  Verification: `cargo test -p ryft-core` (lib + integration + doctests), `cargo test -p ryft-macros-tests`,
  `cargo test -p ryft-xla --lib`, and `cargo test -p ryft-mlir --lib` all pass with 300-second timeouts;
  `cargo fmt --check` is clean on every touched crate; clippy reports only pre-existing warnings in the touched
  files; targeted searches confirm no `autodiff` namespace, no `CoordinateBasisOperation` reference, and no
  remaining production `Instruction` reconstruction that drops provenance. Three independent audit waves
  (conventions, correctness, simplicity) ran over the complete change; every finding was fixed or pinned as a
  documented, tested design decision, and a clean-verdict re-audit closed the loop. Changelogs were deliberately
  not touched per instruction. The plan file itself stays at the repository root where its author created it.
