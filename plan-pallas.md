# Ryft Pallas-Style Kernels: Architecture and Implementation Plan

Status: proposed. The reference architecture formerly recorded in `plan-references.md` is complete through its
preserved-reference boundary. This plan begins the separate program required to turn that boundary into a production
kernel language. No phase in this document is implemented merely because a lower-level wrapper or mock validator
already exists.

This is a source-sensitive plan. Mosaic, cuTile, Triton, and their runtime integrations are evolving systems, so
Phase 0 refreshes the first-release non-TPU inventory against the selected OpenXLA, JAX, CUDA, and cuTile revisions;
Phase 21 refreshes the TPU inventory immediately before TPU implementation; and Phase 22 refreshes the direct Triton
and ROCm inventories immediately before that optional post-v1 expansion. Links describe the design snapshot, not an
upstream stability promise.

## 1. Executive decisions

1. **Ryft owns a backend-neutral Pallas-style kernel language and semantic IR.** Mosaic GPU, Mosaic TPU, cuTile, and
   direct Triton are compiler backends. NVIDIA and AMD GPUs and Google TPUs are hardware targets. These layers must
   not be conflated. `ryft_core::kernels` owns the portable DSL and semantics. Separate `ryft-mosaic`,
   `ryft-cutile`, and `ryft-triton` crates own compiler adapters and exact extensions; `ryft-cuda` remains the shared
   producer-neutral CUDA launcher. `ryft-xla` integrates those adapters with outer XLA programs and PJRT execution.
2. **Finish the non-TPU lower layers first.** Phases 0-5 complete and prove the missing GPU/cuTile `ryft-xla-sys`,
   `ryft-mlir`, `ryft-cuda`, and `ryft-pjrt` foundations. No production change in `ryft-core` or `ryft-xla` may begin
   before the Phase 5 gate passes. Every TPU-specific inventory, wrapper, runtime, lowering, test, and qualification
   task is consolidated in dedicated Phase 21 so Phases 0-20 never require libtpu or TPU hardware. Phase 22 is a
   post-v1 optional backend expansion and may begin after Phase 20 independently of Phase 21.
3. **Extend what exists.** `ryft-xla-sys` already links the JAX Mosaic dialects, `ryft-mlir` already wraps substantial
   Mosaic GPU and TPU surfaces, and `ryft-pjrt` already compiles, executes, serializes, and reloads generic MLIR
   programs. The missing work is primarily pinned-surface parity, pass/compiler bridges, target capabilities, artifact
   metadata, and end-to-end conformance—not a parallel dialect or runtime universe.
4. **Keep portable semantics and exact target control side by side.** Portable operations may select an equivalent
   implementation or a documented fallback. Explicit target operations such as Mosaic GPU `tcgen05_mma` must either
   lower exactly on a compatible target or fail before compilation; they must never silently emulate.
5. **Mosaic is the primary production route.** Mosaic GPU is the explicit NVIDIA path and Mosaic TPU is the TPU path.
   cuTile support is a required roadmap deliverable but remains an optional install/runtime component for the portable
   tile subset. Direct Triton and ROCm support are optional post-v1 extensions in Phase 22. They consume the verified
   portable IR through pinned native interfaces; deprecated JAX Pallas-to-Triton compatibility is not a backend.
6. **Current and future hardware features are additive capabilities.** Storage type, scale encoding and geometry,
   accumulator type, layout, memory space, synchronization, and target instruction are modeled separately. This is
   required for Blackwell NVFP4 and for hardware capabilities that do not exist yet.
7. **Correctness precedes automatic performance.** A reference interpreter, precise verifier, deterministic lowering,
   and code-generation evidence precede scheduling heuristics and autotuning. Numerical equality alone does not prove
   that a tensor core, DMA engine, or asynchronous path was used.
8. **The public outer ABI remains functional and array-valued.** Kernel-internal references, scratch, barriers, and
   target resources never become ordinary program values. Buffer aliasing and donation are optimization metadata, not
   mutation semantics.

## 2. Goals and non-goals

### Goals

- A Rust-native kernel authoring surface with Pallas-style grids, block mappings, references, scratch, masked memory
  operations, scalar/tile computation, control flow, asynchronous transfers, synchronization, and target extensions.
- One canonical kernel IR with verifiable memory, shape, access, alias, initialization, and synchronization rules.
- A deterministic interpreter that serves as the semantic oracle on machines without accelerators.
- Production Mosaic GPU lowering for Hopper and newer NVIDIA GPUs, including explicit Hopper and Blackwell features.
- Production Mosaic TPU lowering for supported TPU generations, memory spaces, DMA, semaphores, vector work, and MXU
  operations.
- An optional cuTile backend for kernels expressible in cuTile's block/tile execution model.
- An optional direct Triton backend for the portable subset, compiling pinned TTIR without a JAX or Python runtime.
- Post-v1 ROCm execution on supported AMD GPUs through a single versioned artifact and launch contract.
- Direct integration with Ryft staging, compilation, PJRT execution, caching, persistence, profiling, and diagnostics.
- A capability and cache model that admits new architectures without changing portable kernel semantics.
- Explicit transform behavior for batching, differentiation, partial evaluation, rematerialization, and sharding.
- Reproducible correctness and performance qualification across compiler-only, GPU, TPU, and release CI tiers.

### Non-goals for the first production release

- Reimplementing the Mosaic, OpenXLA, CUDA, TPU, or cuTile compilers.
- Mirroring JAX's Python syntax or promising source compatibility with JAX Pallas.
- A lowest-common-denominator schedule language that hides meaningful GPU/TPU differences.
- Deprecated JAX Pallas-to-Triton compatibility or Triton as a primary first-release backend. Phase 22 adds an
  optional direct backend instead.
- AMD code generation in the first production release. Phase 22 adds optional post-v1 ROCm support; Intel, Metal, and
  CPU-native kernel code generation remain outside this plan. The interpreter remains the CPU correctness path.
- Arbitrary pointer arithmetic, host pointers, dynamic allocation, recursion, exceptions, or escaping kernel references.
- Unbounded dynamic grid or block dimensions. Resource planning requires a static extent or a finite symbolic bound.
- Automatic differentiation through arbitrary mutable kernel bodies in the first release.
- Automatic scheduling or autotuning before manual schedules are correct, inspectable, and reproducible.
- A new PJRT extension invented solely for Ryft kernels. Use standard PJRT program compilation, XLA FFI/custom calls,
  and existing plugin extensions unless an upstream, independently useful extension exists.
- Depending on Python at execution time. A build-time cuTile compiler tool may be optional; production artifacts must
  be self-contained and launched through a stable binary ABI.

## 3. Terminology and layer boundaries

- **Kernel language:** the user-facing Rust operations and higher-order kernel call.
- **Kernel IR:** the backend-neutral, verified representation produced by staging.
- **Portable operation:** an operation with backend-independent semantics and capability-gated lowerings.
- **Target operation:** an operation whose semantics name a backend or hardware contract explicitly.
- **Grid:** an arbitrary-rank, statically ranked logical launch space. Rank zero is one singleton program; any zero
  extent launches no programs. Backends flatten or map it into their physical launch dimensions with checked
  arithmetic.
- **Block mapping:** the map from a grid point and static parameters to an operand window.
- **Kernel reference:** a non-escaping capability over an operand window or scratch allocation.
- **Scratch:** kernel-local storage with explicit memory space, lifetime, alignment, and initialization state.
- **Mosaic GPU:** the JAX/OpenXLA MLIR-based NVIDIA kernel backend. Current Pallas documentation targets Hopper and
  newer devices.
- **Mosaic TPU:** the JAX/OpenXLA TPU kernel backend, using TPU memory spaces, vector operations, DMA, semaphores, and
  matrix units.
- **cuTile:** NVIDIA's tile programming system. Its tile execution space exposes block-level parallelism without
  per-thread control or explicit intra-block synchronization.
- **Direct Triton backend:** Ryft lowering from verified portable kernel IR to a pinned TTIR contract, followed by
  compilation through the existing PJRT Triton extension. It does not reuse JAX's Pallas lowering or require Python.
- **ROCm:** the AMD GPU target stack used in Phase 22, including the selected PJRT plugin, HIP runtime, HSACO artifact,
  architecture identifier, and launch ABI.
- **Capability:** a compiler-and-device fact used for legality, selection, caching, and diagnostics—not a semantic
  fallback promise.

### 3.1 Language, compiler adapters, and execution integration

The portable authoring path is:

```text
Rust kernel closure using ryft_core::kernels
    -> kernel region in the existing Ryft Program IR
    -> core verification, specialization, and portable canonicalization
    -> verified kernel + logical call contract
       -> ryft-mosaic: Mosaic GPU module or Mosaic TPU compiler payload
       -> ryft-cutile: cuTile compiler input -> versioned AOT CUDA artifact
       -> ryft-triton: typed TTIR -> pinned compiler -> NVIDIA or AMD artifact
    -> ryft-xla: adapter selection, StableHLO embedding, outer executable integration
    -> ryft-pjrt: compile/load/execute/fence
       -> existing Mosaic/plugin runtime, or shared platform artifact launcher
```

This is data flow, not a Cargo dependency graph. `ryft-xla` invokes the selected compiler adapter while lowering the
outer program; the adapter never calls back into or depends on `ryft-xla`. A portable kernel can be authored,
staged, verified, serialized, and interpreted without any compiler adapter or accelerator runtime installed.
Backend independence means stable semantics, not universal lowerability: each adapter admits an explicit subset.

Compiler and execution platform are separate choices. cuTile and Triton can both produce NVIDIA artifacts consumed
by `ryft-cuda`; Triton can also target AMD. Mosaic GPU and Mosaic TPU share a compiler family but have different
runtime contracts. Neither every backend artifact nor every compiled kernel is a cubin. Direct non-XLA kernel
execution is a possible later consumer of these boundaries, not an additional first-release API or completion gate.

### 3.2 Crate ownership and dependency rules

All new crate, module, and contract names in this section are planned APIs, not claims that they already exist.

| Owner | Owned contract |
|---|---|
| `ryft_core::kernels` | Portable DSL, kernel IR, verification, interpreter, logical signature, extension contract |
| `ryft_mosaic::kernels` | Mosaic GPU/TPU lowering, exact extensions, target simulation, compiler payloads |
| `ryft_cutile::kernels` | cuTile lowering/tool invocation, compiler options, subset checks, manifest interpretation |
| `ryft_triton::kernels` | Direct Triton lowering, typed TTIR, compiler invocation, target legality and metadata |
| `ryft-cuda` | Producer-neutral CUDA artifacts, physical launch ABI, driver loading, module cache, stream launch |
| `ryft-mlir` / `ryft-xla-sys` | Typed compiler IR, native compiler bridges, source pins and low-level build surfaces |
| `ryft-pjrt` | Raw plugin capabilities, compiler extensions, buffers, streams, FFI adapters and execution fences |
| `ryft_xla::kernels` | Outer-program integration, adapter selection, custom calls, executable persistence |
| `ryft` facade | Stable portable exports and explicitly enabled backend integration conveniences |

The allowed dependency direction is:

```text
ryft-xla      -> ryft-core, ryft-mlir, ryft-pjrt, ryft-xla-sys
              -> optional ryft-mosaic / ryft-cutile / ryft-triton
ryft-mosaic   -> ryft-core, ryft-mlir, required ryft-xla-sys compiler bridges
ryft-cutile   -> ryft-core, ryft-cuda
ryft-triton   -> ryft-core, ryft-pjrt's existing Triton compiler extension
              -> ryft-mlir only for required typed TTIR; ryft-cuda for NVIDIA artifacts
ryft-pjrt     -> ryft-xla-sys, optional ryft-cuda
ryft-mlir     -> ryft-xla-sys
ryft-cuda     -> driver-loading and artifact utilities only
ryft-core     -> existing backend-independent dependencies only
```

`ryft-core` must never depend on the adapter crates, `ryft-cuda`, `ryft-pjrt`, `ryft-mlir`, or `ryft-xla-sys`.
No compiler adapter depends on `ryft-xla`, and neither `ryft-cuda` nor `ryft-pjrt` depends on a compiler adapter.
In particular, adding PJRT-dependent compilation to `ryft-cuda` would reverse the existing PJRT-to-CUDA edge.
Keeping cuTile in `ryft-cutile` preserves the lightweight launcher without feature-dependent dependency cycles.
Backend independence does not require compiler adapters to avoid native XLA libraries: Mosaic may need
`ryft-xla-sys`, and the selected Triton route deliberately uses the PJRT compiler extension.

Create adapter crates when their implementation phase begins, not as empty scaffolding. Introduce `ryft-mosaic`
with GPU support in Phase 14, `ryft-cutile` in Phase 16, and `ryft-triton` in Phase 22. Add Mosaic TPU behind an
explicit feature only in Phase 21. Keep optional compiler dependencies, external tools, and accelerator test features
out of the core and default launcher builds. Verify isolated and combined feature graphs; Cargo feature unification
must not activate an unwanted compiler, TPU runtime, or a circular dependency.

### 3.3 Portable DSL and exact extensions

`ryft_core::kernels` owns typed builders and closure staging for kernel definitions, grids, block mappings,
references, scratch, scalar/tile operations, bounded control flow, portable synchronization, and schedule hints.
Reuse existing `Program`, array/type, reference/view, context, operation-capability, tracing, and transform machinery;
do not introduce a second instruction graph, reference system, or JIT lifecycle. Kernel-specific sugar stages the
canonical operations and has the same verifier and interpreter behavior as explicit construction.

A portable definition contains no backend selection, compiler options, CUDA launch dimensions, TTIR layouts, or
Mosaic resource enums. The caller binds it to a compiler choice and typed options through an execution integration.
The same definition and logical specialization must be reusable for multiple adapters without retracing a different
kernel body. Backend-independent semantic requirements stay in the definition; target legality stays in adapters.

The planned `KernelOperation<Extension>` boundary uses a core-defined typed extension contract. Portable kernels use
an empty extension family. Each adapter owns its exact operation types, authoring capabilities, launch/resource
contracts, and deterministic target simulation. Mosaic GPU extensions live in `ryft_mosaic::kernels::gpu`; TPU
extensions arrive under its TPU module in Phase 21. cuTile and direct Triton initially consume portable operations;
introduce exact extensions there only for a separately specified semantic need. Mosaic operations remain outside
the ordinary core array operation family; neither cuTile syntax nor TTIR becomes the common kernel IR.

An extension declares types, effects, reference accesses and aliases, resource creation/consumption, liveness,
race/synchronization rules, transform behavior, capability requirements, and semantic/schema identity. Core checks
these declarations through its canonical analyses; adapter validation additionally checks target-specific invariants.
Adapter implementations are trusted compiler code and require conformance tests; a trait implementation alone is
not proof of correct semantics. Unknown contracts are rejected rather than granted conservative-looking permissions.
Use existing operation traits for the obligations they already express and keep new extension bounds narrowly scoped.

Core must not seal the extension contract to an XLA-owned enum. Adapter operation families may remain sealed and
experimental. If heterogeneous dispatch requires a combined enum, the consuming integration owns it and composes
adapter types without redefining their semantics. Adding an adapter must not require editing a central core enum.
Arbitrary names, strings, byte payloads, or unchecked `Any` values cannot represent verified operations. Artifact
payload bytes are allowed only after semantic verification and with typed, versioned producer contracts.

Exact target operations never silently fall back to another backend or portable emulation. Deterministic target
simulation is a test/debug semantic model, not a production fallback. Portable interpretation works without loading
extension crates; interpreting an exact extension requires its supplied typed simulation or an explicit unsupported
result. Persisted extension bodies require the matching decoder, schema version, and revalidation on reload.

### 3.4 Proposed macro authoring surface

The primary authoring surface is a `#[kernel]` procedural macro re-exported from `ryft_core::kernels`. Explicit
builders remain available for programmatic construction and as the expansion target. The macro implementation lives
in the existing `ryft-macros` crate; it depends only on syntax tooling and emits calls to core APIs, avoiding a reverse
macro-to-core dependency. Macro integration tests live in `ryft-macros-tests`, with an isolated core-only consumer
proving that macro use does not require the backend-enabled `ryft` facade.

The following is a proposed API sketch, not an existing compilable example. Use the canonical `Array` vocabulary,
not a new `Matrix` type. The existing `Array` is not generic over its element type: dtype/rank constraints belong in
signature metadata and are checked against `ArrayType`. The attribute consumes the annotated function and generates
an implementation over symbolic values and canonical references; it does not execute the displayed body against
borrowed concrete host arrays or add kernel-only functions to the concrete `Array` implementation.

```rust
use ryft_core::arrays::Array;
use ryft_core::kernels::{kernel, zeros};

/// Computes matrix multiplication using 32 × 32 tiles.
#[kernel(requires = left.shape()[1] == right.shape()[0])]
fn matmul(
    #[input(data_type = F32, rank = 2)] left: &Array,
    #[input(data_type = F32, rank = 2)] right: &Array,
    #[output(
        data_type = F32,
        shape = [left.shape()[0], right.shape()[1]],
        tile = [32, 32],
        boundary = masked,
    )]
    output: &mut Array,
) {
    let [row, column] = output.tile_index();

    let left_tiles = left.tiles([32, 32]).pad(0.0);
    let right_tiles = right.tiles([32, 32]).pad(0.0);
    let mut accumulator = zeros::<f32>([32, 32]);

    for depth in 0..left.shape()[1].div_ceil(32) {
        let left_tile = left_tiles.load([row, depth]);
        let right_tile = right_tiles.load([depth, column]);
        accumulator += left_tile.dot(right_tile);
    }

    output.store(accumulator);
}
```

The generated definition must enforce these semantics:

- Input rank and dtype are validated before shape indexing. For this kernel, require equal reduction dimensions
  `left.shape()[1] == right.shape()[0]` through the explicit `requires` annotation, even though zero padding would
  otherwise hide a mismatch. The generated definition checks rank/dtype before evaluating shape constraints and
  output shape expressions. Tile shape compatibility alone is not evidence of equal full reduction dimensions.
- The output annotation declares a write-only result, with grid extents equal to the ceiling-divided output shape.
  Each program receives one disjoint logical output window. `output.tile_index()` supplies grid indices, not element
  offsets. Input `.tiles(...)` constructs reference-view metadata, and `.load([row, depth])` selects by tile index.
- `.pad(0.0)` defines out-of-bounds input values; `boundary = masked` discards output stores outside the full shape.
  No implicit unchecked access is introduced. Zero reduction depth produces zeros; zero output extents launch no
  programs and return an empty result. All extents are static or explicitly bounded under the existing shape rules.
- The accumulator is a pure local array-valued tracer with shape `[32, 32]`, not a reference or allocated host tile.
  Its reassignment produces loop-carried SSA values. `dot` uses a declared portable numerical contract and does not
  promise tensor-core instructions or silently choose reduced precision.
- The generated outer callable accepts the two input arrays and returns one array through the existing compilation
  domain. The annotated output parameter exists only inside the kernel region. It does not require host output
  allocation, expose mutable array storage, or change the functional outer ABI. Read-write operands require an
  explicit access declaration rather than being inferred from every `&mut` parameter.

`tiles`, `tile_index`, `load`, and `store` are proposed authoring conveniences over `ArrayReference`, reference views,
and canonical operations, not a second array/reference/partition IR. A lightweight wrapper may carry validated view
metadata if needed, but all effects, shape rules, and identity must remain in the existing owners. General grids,
block mappings, and scratch remain expressible explicitly; output-derived grid inference is the simple default and
must reject ambiguous multi-output mappings instead of guessing.

Tile sizes may later become static parameters using the same specialization machinery; the fixed sizes above keep
the example focused. Shape and output annotations are pure metadata expressions, never reads from runtime array
contents. Explicit backend selection and compiler options remain outside the annotated definition. Unsupported
constructs receive source diagnostics; no macro fallback bypasses core verification.

### 3.5 Macro expansion uses Ryft tracing

Pallas traces kernel functions with reference abstract values into JAX's existing jaxpr representation before backend
lowering; its implementation calls the ordinary tracing machinery in `_trace_kernel_to_jaxpr`.
See the [Pallas tracing implementation](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/pallas_call.py).
This is architectural context, not a new dependency or a promise of matching JAX syntax.

Ryft follows the same reuse principle with its own machinery. The existing
[`tracing` module](crates/ryft-core/src/tracing.rs) records operations through `Context::bind`, using `Tracer`,
`StagingContext`, and `ProgramBuilder`; `Trace::trace`/`TracingContext::trace` provide root tracing and
`NestedTracingContext::trace` provides nested regions. Kernel-specific work adds the operation family, permitted
reference boundary, metadata, and validation rules to these facilities. It does not create an independent tracer,
AST-to-kernel compiler, or instruction graph. Use the existing `ArrayIrType`/`ArrayIrValue` universe for regions
containing both arrays and `ReferenceType` values; an ordinary array-only tracer is insufficient for kernel references.
Generated structured signatures use the canonical `Parameterized` families, without separate flattening rules.

```text
#[kernel] function and metadata
    -> Rust macro expansion: generated typed staging function + definition/call metadata
    -> existing root or nested Ryft tracing with symbolic array/reference inputs
    -> Context::bind and canonical operation capabilities
    -> existing ProgramBuilder and nested Program regions
    -> reference/kernel verification and immutable semantic Program
    -> existing specialization/compilation-domain lifecycle
    -> selected compiler adapter -> execution integration
```

The macro and tracing have distinct responsibilities:

| Source construct | Macro responsibility | Existing semantic machinery |
|---|---|---|
| Input/output attributes | Generate types, access, grids/mappings | `ArrayType`, references, parameter trees |
| Arithmetic and tile calls | Generate typed calls and source spans | Tracers and `Context::bind` |
| `let mut`, assignment, `+=` | Identify changing values and carried state | SSA values and region parameters/results |
| Bounded `for` | Generate counted staged loop through established control-flow APIs | Nested tracing and loop regions |
| Value-dependent `if` | Generate condition and both branch closures | Nested tracing and conditional regions |
| Load/store and views | Generate canonical view/access operations | Reference effects, liveness, initialization |
| Staging failures | Propagate errors and source locations | Existing poison/finalization and `ProgramError` |

Ordinary Rust tracing cannot intercept language-level `if`/`for` or inspect a symbolic value as a host boolean. The
macro must translate supported control flow into staged APIs before tracing runs. A staged loop traces its body once
per traced structure, with explicit accumulator inputs/results; it does not unroll merely because a bound is static.
An explicit, bounded static-specialization/unrolling construct can be added separately. Data-dependent branches stage
both branches and validate their result types and effects, rather than selecting one at host trace time. Reuse the
existing condition/while/scan region machinery where it fits; specify any missing convenience in its current owner.

The initial subset includes typed function parameters, scalar literals, local bindings/reassignments, arithmetic,
recognized capability calls, tuple/array destructuring, bounded range loops, and structured conditionals. Reject
unbounded loops, arbitrary host I/O, allocation of dynamic host containers, recursion, escaping references, unsupported
`break`/`continue`/early-return forms, and arbitrary nested macros with precise diagnostics. Allow helper functions
only through documented traceable capability-based helpers or explicit core staging functions; a proc macro cannot
inspect arbitrary external function bodies or perform Rust name/type resolution itself. Do not build a second Rust
compiler to support unrestricted source syntax.

Rust checks the generated generic staging code; abstract shape/dtype/effect checks remain in core tracing and kernel
verification. Generated code must remain hygienic under renamed imports/crates and use the established macro path
conventions. Preserve Rust spans and expose expansion/IR inspection for diagnostics. Kernel authors need not write
`context` or `?` for every expression: generated fallible calls and existing tracer poison propagation report failures
at the generated staging/call boundary without panicking or swallowing errors.

Reuse `InterpretableOperation` for tracer-valued replay and the existing compilation-domain cache lifecycle. Batching,
partial evaluation, differentiation policy, and reference discharge operate on the resulting canonical `Program`;
macro syntax does not add new transform semantics or permission to differentiate mutable bodies. Kernel references
remain in the validated preserved-reference region. Traceable helper captures and foreign/escaped tracers obey the
same restrictions as explicit builder construction.

## 4. Current repository foundation

### Existing pieces to retain

- [`crates/ryft-xla-sys`](crates/ryft-xla-sys) currently pins OpenXLA
  `f16a4aeb435b2896ab96b605f004f982f6c97eb8` and JAX `a33ed614c58ee8a10d0b7536c50c2609c38500c1`,
  builds the JAX Mosaic dialect C
  API objects, carries custom C++/Rust bridges for Mosaic GPU and TPU, and archives the Mosaic GPU pass header.
- [`crates/ryft-mlir/src/dialects/mosaic/gpu`](crates/ryft-mlir/src/dialects/mosaic/gpu) already wraps GPU attributes,
  types, and 37 named operations, including barriers, asynchronous GMEM/SMEM/TMEM transfers, WGMMA, TMEM, and
  `tcgen05` operations.
- [`crates/ryft-mlir/src/dialects/mosaic/tpu`](crates/ryft-mlir/src/dialects/mosaic/tpu) already wraps TPU attributes,
  types, and 86 named operations spanning loads/stores, DMA, semaphores, vector transforms, reductions, MXU work,
  tracing, and device communication.
- `ryft-mlir` also has broad standard MLIR, GPU, NVGPU, NVVM, SCF, memref, and LLVM surfaces, plus builtin vector
  types. It does not yet have a typed Vector dialect module. NVVM already includes Blackwell `tcgen05` and
  block-scaled MMA operations.
- [`crates/ryft-pjrt`](crates/ryft-pjrt) already supplies generic MLIR/HLO program formats, compile/load/execute,
  asynchronous fences, serialization, topology, memory, stream, FFI, GPU custom-call, Triton, profiling, and
  executable-metadata wrappers.
- The reference architecture of `plan-references.md` proved that roots, views, accesses, liveness, and operation-local
  aliases can survive inside an explicitly validated kernel boundary, through a preserved-reference kernel mock
  (`crates/ryft-xla/src/experimental/reference_kernels.rs`) backed by a whole-closure static `ReferenceAnalysis`.
  After the interpreter-style discharge rework of `plan-reference-discharge.md`, that analysis stack and the mock were
  deliberately deleted (they were working-tree-only and were never committed), because discharge validates programs
  itself and their only remaining consumer was this plan. The restoration phase below rebuilds them against the real
  kernel operation instead of the mock.
- `ryft-core` already models `F4E2M1FN`, and `ryft-xla` already lowers and executes a portable block-scaled-dot path.
  That is useful groundwork, but it is not by itself an NVFP4 tensor-core kernel contract.
- [`crates/ryft-cuda`](crates/ryft-cuda) owns the producer-neutral CUDA artifact launcher, including dynamic driver
  loading, context-identity-aware bounded module caching, argument packing, and launch on a borrowed stream.
  `ryft-pjrt::cuda` owns only the PJRT-client and XLA FFI stream/buffer adapters. Experimental probes consume that
  launcher and must not introduce a second implementation.

### Gaps that must be measured, not assumed

- Exact parity between the pinned Mosaic TableGen/source surface and local C/Rust wrappers.
- Missing Mosaic registration, version, analysis, serialization, and runtime symbols that lack an upstream C API.
  Existing MLIR C APIs and `ryft-mlir` wrappers already own generic diagnostics, pass managers, cloning, and bytecode.
- Binary MLIR parsing. `ryft-mlir` can write bytecode but its current parsing conveniences are text-oriented; Mosaic
  modules and backend payloads must round-trip as bytes without a UTF-8/C-string detour.
- Typed Vector and Math dialects and the exact Complex, UB, Bufferization, and conversion-pass deltas used by the
  pinned Mosaic pipelines. Vector and Math C APIs/TableGen are linked; Complex and Bufferization TableGen files are
  archived without typed local facades; UB is only a transitive dialect dependency and its TableGen archive coverage
  must be decided in Phase 0.
- Typed `ryft-mlir` wrappers for every actually required missing attribute, type, pass, pipeline option, and verifier.
- CUDA-plugin linkage and retention for JAX's Mosaic GPU custom-call/runtime registration. Dialect wrappers alone do
  not prove that the `mosaic_gpu_v2` execution target is present in the shipped PJRT plugin.
- A proven path from hand-authored Mosaic GPU MLIR through the pinned compiler and PJRT plugin to execution, followed
  by the equivalent TPU proof only in Phase 21.
- Architecture and compiler capability discovery with exact prelaunch diagnostics.
- A versioned generic CUDA kernel artifact/calling-convention contract that can be cached and reloaded through
  existing PJRT/XLA
  machinery.
- An official cuTile integration seam. cuTile currently documents JIT launch and AOT export to cubin or versioned
  TileIR bytecode with the `cutile_python_v2` calling convention; it does not define a Ryft or PJRT extension.
- NVIDIA's documented `cuda.tile.jax.cutile_call` provides useful prior art for passing read-only arrays,
  input/output arrays, output placeholders, scalars, and static arguments through XLA FFI. Ryft may reuse that ABI
  model where it fits, but must not depend on JAX or Python at deployment time.
- The production kernel language, verifier, interpreter, scheduling contract, backend lowerings, transformations,
  debugging, profiling, autotuning, examples, and release qualification.
- A pinned, typed direct-Triton compiler contract and one production ROCm execution route. The existing PJRT Triton
  extension is useful groundwork, but it does not itself prove TTIR ownership, HSACO execution, or AMD qualification.

Phase 0 produces machine-checkable non-TPU source contracts that compare the pinned and local surfaces directly.
Phase 21 produces the separate TPU source contract, and Phase 22 produces the direct Triton and ROCm source contracts.
No phase may call its owned surface complete without updating the relevant executable contract and its human-reviewed
decision record.

## 5. Semantic model

### 5.1 Higher-order kernel call

A kernel call owns:

- one typed kernel body region;
- a grid and optional dimension names/semantics;
- one block mapping per array operand and result;
- static parameters and specialization constraints;
- operand access contracts and operation-local result/operand aliases;
- scratch specifications;
- portable schedule hints;
- backend-independent semantic requirements;
- source locations and a stable semantic fingerprint.

Compiler selection and target-specific compiler options belong to the execution integration's binding of a kernel,
not to the portable definition. A typed extension may add target launch/resource semantics to the body contract;
those semantic requirements cannot be erased or changed by choosing different compilation options.

The outer signature contains arrays, scalars, and static parameters. Inner array windows and scratch are references.
Read/write effects remain inside the kernel operation; updated outputs make mutation explicit at the outer SSA
boundary. Kernel-internal references cannot be returned, captured by ordinary regions, serialized as constants, or
stored in composite program values.

Kernel calls and their typed body regions remain part of Ryft's ordinary `Program` IR. Full and partial reference
discharge are strictly Program-to-Program transformations: references selected for discharge become explicit state in
the rewritten `Program`, while kernel references selected for preservation remain reference-typed values in that same
program. Backend-specific MLIR, source, binary, and executable artifacts arise only during later lowering and cannot
serve as alternate reference-discharge result types.

Read-only operands contain their entering values and produce no updated result. Read-write operands initially contain
their entering values and publish an updated result. A write-only result starts uninitialized: the verifier must prove
that every published element is definitely initialized on every successful completion path, or the caller must provide
an explicit initial fill. Ordered, race-free repeated writes are valid and publish the final value. Reading a
write-only result is invalid. Masked and partial-grid writes must prove coverage of every published element. An empty
grid therefore requires zero-sized write-only results or an explicit fill. Outputs become observable only after the
kernel completion succeeds; overlapping writes across grid programs follow the race rules in §5.6.

### 5.2 Grids, block mappings, and bounds

- Grids have statically known arbitrary rank and statically or symbolically bounded extents. Rank zero is one singleton
  program, a zero extent is an empty launch, and GPU/cuTile lowerings flatten or map logical dimensions into their
  physical one-to-three-dimensional launch limits with checked arithmetic.
- `program_id(axis)` and `num_programs(axis)` are pure scalar operations.
- A block mapping is a pure, separately validated function of grid IDs, scalar-prefetched values, and static
  parameters. It returns starts plus a statically known block shape.
- Block mappings cannot read mutable kernel references or depend on data loaded by the kernel body.
- Dynamic boundary tiles require explicit masks or padding semantics. Out-of-bounds accesses without a proven mask
  are errors, never backend-defined behavior.
- Index arithmetic is checked for overflow in the interpreter and is lowered with the same signedness and width on all
  backends.

### 5.3 Memory and references

The logical root/view/access model is reused from `plan-references.md`. Kernel-owned additions are:

- memory space and physical layout eligibility;
- minimum alignment and stride constraints;
- scratch lifetime and definite initialization;
- asynchronous-copy participation;
- atomicity and memory ordering;
- synchronization scope;
- alias and race validation across operands and grid programs.

Portable storage classes are limited to external operand windows, program-local scratch, and private values.
Backends map those classes to compatible target storage only when semantics and lifetime match. GPU GMEM/SMEM/TMEM
and TPU HBM/VMEM/SMEM are target layout/storage capabilities, not aliases in a misleading portable memory-space enum.
Barriers, semaphores, and async tokens are typed resources rather than memory spaces.

Scratch is uninitialized unless its constructor provides a value. The verifier rejects read-before-initialization,
partial initialization followed by whole-value reads, escape, overlapping incompatible aliases, and use after its
scope. Control-flow definite-initialization uses intersection at joins and a fixed point for loops.

### 5.4 Primitive operation families

The first portable family includes:

- scalar constants, arithmetic, comparisons, selection, casts, and bounded control flow;
- tile creation, reshape, transpose, broadcast, iota, slice, concatenate, and reductions;
- reference load, store, swap, and ordered accumulation with views, masks, and optional `other` values;
- sequentially consistent atomic read-modify-write operations over the portable scopes defined in §5.6, with
  commutative atomic accumulation carried by the dedicated access mode defined there;
- dot and block-scaled dot with explicit input, scale, accumulator, output, rounding, and saturation contracts;
- asynchronous copy start/wait through linear completion tokens and target-neutral pipeline stages;
- debug assertions and trace markers that can be compiled out by policy.

Operation legality is defined independently of backend availability. A portable operation may lower differently but
must preserve semantics. Target operation families are namespaced, capability-gated, and excluded from portable
fallback.

### 5.5 NVFP4 and future tensor-core formats

NVFP4 is modeled as a compound contract, not as a single scalar data type:

- E2M1 four-bit value storage and packing;
- E4M3 per-block scale storage;
- scale block geometry, currently including the 16-value Blackwell form;
- per-tensor FP32 scale, where omission semantically means an exact factor of `1.0`;
- accumulator and output types;
- operand layout/swizzle and memory spaces;
- rounding, saturation, NaN, and exceptional-value behavior;
- sparse or dense form and collective CTA mode;
- exact target instruction capability.

The portable `block_scaled_dot` path can select a compatible backend implementation. The explicit Mosaic GPU
`tcgen05` path exposes TMEM, scale transfer, ordering barriers, and collective MMA and fails on non-Blackwell targets.
Cache identity includes the complete format and architecture contract, so a kernel compiled for `sm_100` or a
particular feature extension cannot be reused on an incompatible device.

Native Blackwell NVFP4 eligibility is narrower than portable block scaling: it requires packed E2M1 values, E4M3
scales for each consecutive block of 16 values, the defined tensor-level FP32 scale, supported accumulator/output
types, exact scale encoding, and compatible operand layouts. When the semantic scale is omitted, native lowering must
materialize or encode the hardware-required exact `1.0` identity representation. Other scale geometries are valid
portable operations but cannot be labeled or tested as native NVFP4.

### 5.6 Synchronization and race freedom

- Ordered reference effects define compiler ordering but do not imply atomicity between programs. The converse also
  holds: cross-program accumulation is never expressed by relaxing the ordered `Accumulate` access mode. Commutative
  atomic accumulation is the semantics that the `Accumulate` documentation in
  `crates/ryft-core/src/programs/references/semantics.rs` explicitly excludes from the ordered mode, and Phase 9
  introduces it as the separate `AtomicAccumulate` mode: the program author promises order-independence, the
  implementation promises tear-free exactly-once application, and results may differ bitwise across runs for
  non-associative element types. Sequential
  contexts admit both accumulation modes, because serializing atomic accumulations is an admitted execution; a
  parallel grid region admits `AtomicAccumulate` and rejects ordered `Accumulate`, because an unspecified traversal
  order cannot honor program-ordered accumulation. Non-commutative atomics such as exchange and compare-and-swap are
  not accumulations and keep read/write-class reference semantics under the same sequentially consistent scope
  contract.
- Grid expansion creates independently scheduled program instances, each internally sequential; grid traversal order
  is not semantically observable. Tile lanes are values, not independently scheduled agents. Portable atomics are
  device-scoped and sequentially consistent: all admitted atomic operations participate in one total order consistent
  with each program instance's order. Conflicting atomic/non-atomic accesses remain data races unless ordered by a
  target synchronization contract. Core validation rejects invalid operations; backend selection rejects valid
  atomic dtype/operation combinations it cannot implement.
- Portable synchronization coordinates one program instance with named asynchronous engines through linear completion
  tokens. CTA, warpgroup, cluster, TPU-core, barrier, and semaphore participants exist only in typed target launch and
  operation contracts. Those contracts define agent membership, scope, uniform arrival, divergence legality, and
  target-specific simulation; the portable interpreter never guesses those execution agents.
- Async operations return linear completion tokens or update typed barriers; dropping or duplicating them is illegal.
- A static race analysis proves simple disjoint block mappings and supported synchronization. Programs outside the
  decidable proof subset are rejected; an unsafe race escape hatch remains outside the first stable API.
- GPU warpgroup/CTA/cluster synchronization and TPU semaphore/DMA synchronization lower separately.

### 5.7 Transform policy

- Transformations inside a kernel operate on pure scalar/tile computations when the transformed operations remain
  lowerable.
- `batch(kernel_call)` initially adds or fuses a grid dimension and rewrites block mappings. It rejects conflicting
  writes or target constraints that cannot be preserved.
- `jvp`/`vjp` initially treat kernel calls as opaque and require an explicit custom rule or a separately traced pure
  fallback. The compiler must not differentiate mutable kernel bodies implicitly.
- Partial evaluation may specialize static parameters and pure block mappings but cannot execute device effects.
- Rematerialization may duplicate pure value work but never memory effects, barriers, async operations, or atomics.
- Sharding composes outside the kernel call. Multi-device kernels require an explicit backend capability and launch
  contract rather than accidental replication.

## 6. Compilation, runtime, and artifact contracts

### 6.1 Pipeline

The core language, compiler adapter, and execution integration each own a validation boundary. The following pipeline
uses XLA as the first production execution integration without making it part of the portable authoring API.

1. Stage the outer call and kernel body.
2. Infer and validate types, block mappings, references, initialization, aliases, effects, synchronization, and races.
3. Canonicalize portable value work without erasing effect or liveness distinctions.
4. In `ryft-xla`, select an enabled compiler adapter from caller policy and validated compiler/device capabilities.
   Ask the adapter to admit the verified semantic contract before artifact lookup or compiler invocation.
5. Apply portable schedule decisions and target-specific legalization.
6. In the selected adapter crate, lower to Mosaic GPU MLIR, Mosaic TPU compiler payload, the supported cuTile subset,
   or Phase 22's supported direct-Triton subset.
7. The adapter verifies target IR and invokes its pinned compiler/tool, or produces a validated deferred compiler
   payload when final code generation occurs during outer XLA/PJRT compilation.
8. `ryft-xla` validates and embeds the adapter output through an XLA custom call with exact layouts, aliases,
   side effects, and metadata. Backend payload encoding remains owned by its adapter.
9. Compile/load/execute through existing PJRT APIs and return the ordinary Ryft execution fence.
10. Persist the semantic IR, target contract, compiler options, artifact, and compatibility metadata atomically.

### 6.2 Capability descriptor

The capability view composes core semantic requirements, adapter-normalized compiler/target facts, and execution
integration facts. Core defines only the semantic vocabulary; adapters own named target features and version schemas;
`ryft-pjrt` and platform runtimes expose raw facts; `ryft-xla` joins them for selection. The composed view records:

- backend and backend ABI version;
- platform, device kind, architecture, compute capability, and feature extensions;
- compiler, OpenXLA, JAX Mosaic, PJRT plugin, CUDA/toolkit, TPU, cuTile, Triton, ROCm, and HIP versions;
- address spaces, maximum grid/CTA/cluster shape, and scratch/resource limits;
- supported scalar, scale, accumulator, atomic, and matrix-operation combinations;
- layouts, swizzles, async copies, barriers, semaphores, tensor memory, and collective modes;
- dynamic-shape, multi-device, profiling, and AOT support;
- known restrictions that affect correctness, not merely performance.

Capability checks occur before artifact lookup and compilation. Diagnostics name the operation, requested contract,
backend, target, and missing feature.

### 6.3 Cache and AOT identity

Reuse the existing `ryft-core` compilation/specialization/cache lifecycle and its domain hooks. Kernel compilation
contributes semantic identity from core, compiler identity from the adapter, and embedding/execution identity from
`ryft-xla`; it does not introduce a parallel JIT dispatcher or independent competing executable cache. The key includes:

- normalized semantic kernel IR and source schema version;
- input/output types, block mappings, static arguments, and specialization constraints;
- access, alias, scratch, synchronization, and dynamic-bound contracts;
- backend, architecture, exact required features, and device-independent vs device-specific status;
- schedule decisions and compiler options;
- OpenXLA, JAX Mosaic, PJRT ABI/plugin, CUDA/toolkit, TPU compiler, cuTile, Triton, ROCm, and HIP versions;
- target artifact and calling-convention schema versions.

Deserialization revalidates all fields and the current capability descriptor. A cache miss is always safe; accepting an
incompatible artifact is not.

### 6.4 cuTile integration boundary

Phase 0 must choose from official interfaces only:

- the documented `cuda.tile.jax.cutile_call` FFI registration/configuration contract, promoted behind a Python-free
  runtime boundary if its implementation and redistribution terms support that use;
- AOT cubin export with `cutile_python_v2`, loaded/launched through an XLA-supported CUDA custom-kernel or FFI path;
- versioned TileIR bytecode export, but only if NVIDIA documents a stable non-Python compiler/runtime API suitable for
  redistribution; or
- a build-time isolated compiler subprocess that emits the AOT cubin and a content-hashed compatibility manifest.

The plan does not assume a stable TileIR C API. It does not embed Python in Ryft, expose the GIL to compilation, or
invent a cuTile PJRT extension. The selected tool is optional, version-pinned, time-bounded, sandboxable, and absent
from runtime deployments that consume precompiled artifacts.

Runtime ABI values are arrays, runtime scalars and dynamic extents, plus genuinely external runtime resources. Static
arguments, constants omitted by a backend ABI, layouts, schedules, scratch declarations, target resources, and
internal completion tokens live in compiler configuration and artifact metadata, not in the physical argument list.

### 6.5 Direct Triton and ROCm integration boundary

Phase 22 is a direct compiler backend, not compatibility with the deprecated JAX Pallas Triton backend. It lowers the
verified portable subset to a pinned TTIR contract and compiles through the existing PJRT Triton extension. If the
pinned native surface lacks typed TTIR construction, Ryft owns a minimal typed schema and deterministic serializer for
the admitted subset; arbitrary user-supplied TTIR strings are not a production API.

The resulting artifact uses exactly one production execution route per platform. NVIDIA reuses the established CUDA
artifact path. ROCm selects either an official XLA/PJRT custom-call route or one hardened HIP launcher for HSACO, based
on the Phase 22 source contract; it does not implement and maintain both. CUDA and ROCm launchers remain concrete and
separate until repeated implementation proves a smaller shared contract. `ryft-triton` owns compiler output and target
metadata, core owns semantic aliases/effects and logical signatures, platform runtimes own physical artifacts, and
`ryft-xla` owns the embedding and executable persistence envelope. No adapter imports an XLA-owned artifact schema.

### 6.6 Minimal compiler adapter contract

Phase 12 specifies the boundary from the Phase 5 native probes; Phase 13 wires an experimental contract into the
existing compilation domain; Phases 14 and 16 validate it against two real DSL-to-backend implementations before
stabilization. Prefer one small named compiler capability over a hierarchy of universal backend/driver traits.
Use existing compilation-domain contracts where they already fit. The following obligations describe the contract,
not a frozen Rust signature or a new executable abstraction:

- **Inputs:** an immutable verified kernel region and core logical signature, static/symbolic specialization facts,
  portable schedule, and adapter-owned typed target/options. Runtime pointers, streams, and buffers are not inputs
  to semantic lowering. Verification is invalidated by semantic mutations, including specialization or transforms;
  revalidate before handing the result to the adapter.
- **Admission:** report supported or a structured rejection with source operation, requested contract, missing
  capability, and owner. Distinguish semantic invalidity, unsupported target/subset, missing compiler installation,
  incompatible version, and compiler failure. Explicit selection never silently selects another compiler; automatic
  portable selection may try another admitted adapter before launch according to caller policy.
- **Output:** an associated concrete artifact or deferred compilation payload, physical argument mapping, resource
  and launch requirements, compiler identity, compatibility metadata, and diagnostics/source mapping. Associated
  types preserve CUDA, Mosaic, and Triton distinctions without exposing those types in core.
- **Identity and validation:** deterministically fingerprint every lowering-relevant option and version, validate
  decoded payloads before reuse, and provide the information the existing cache/AOT lifecycle needs. No global mutable
  backend registry, dynamic plugin ABI, or string-based operation dispatch is required for the initial implementation.
- **Execution integration:** `ryft-xla` implements bridges for supported adapter outputs, translating the core logical
  signature into custom-call operands/results and binding the adapter's physical mapping. Adapters must not allocate
  outer program buffers, submit outer executables, or create a competing public completion/fence type.

Final native compilation need not happen at the same step for every compiler. cuTile returns a ready CUDA artifact;
Mosaic may return versioned compiler input consumed by the plugin; Triton uses the selected PJRT compilation seam.
A compile-only adapter test must run without linking `ryft-xla`; required native compiler libraries or a PJRT plugin
are allowed and explicitly documented. This does not promise hardware-free compilation when the native seam needs a
client/device. The independent core interpreter tests require neither native compilers nor plugins.

### 6.7 Artifact, ABI, and persistence ownership

Keep four contracts separate and reference their canonical types instead of copying fields into parallel schemas:

| Contract | Owner | Contents |
|---|---|---|
| Logical kernel call | `ryft_core::kernels` | Signature, bounds, access, aliases, effects, semantic identity |
| Compiler output | Compiler adapter | Target input/output, options, source mapping, compiler/version compatibility |
| Physical launch | Platform runtime | CUDA artifact/argument ABI, or the selected plugin/ROCm execution contract |
| Outer execution | `ryft-xla` | StableHLO embedding, PJRT integration, executable cache/AOT envelope |

The adapter maps logical parameters to the physical ABI, including pointer/shape/stride expansion, omitted static
arguments, resource metadata, and entry symbols. `ryft-cuda` remains unaware of cuTile or Triton semantics and validates
only its producer-neutral artifact/launch contract. No adapter-specific schema is added to `ryft-pjrt` unless an
upstream API owns it. `ryft-xla` validates aliases, effects, layouts, required buffers, and execution compatibility
when embedding the output; the compiler adapter never needs the XLA envelope to represent its own result.

The AOT envelope records the semantic schema/fingerprint, adapter ID/schema/options/compiler identity, physical ABI,
target requirements, and execution/plugin compatibility as separately versioned components. Hashes cover payloads and
all legality/codegen-relevant metadata. Reload rejects unknown extension decoders, incompatible adapter schemas, or
mismatched physical argument maps before native loading. A runtime-only deployment may omit the compiler tool and
lowerer when the chosen artifact supports it, but must retain the required payload decoder, validator, launcher,
registration, and PJRT integration. Deferred Mosaic payloads may still require native compilation at load time.

No first-release work introduces a universal driver, buffer, stream, allocator, or fence abstraction solely for
kernels. CUDA execution stays in `ryft-cuda`, PJRT adaptation in `ryft-pjrt`, and the Phase 22 ROCm route has exactly
one owner selected by its native contract. Compiler independence is enforced even where execution still uses XLA.

## 7. Estimate methodology

Estimates count new or materially rewritten logical source lines:

- **Production:** Rust, C/C++, Bazel/build configuration, schemas, and non-test fixtures required at runtime/build time.
- **Tests:** unit/integration/property/golden/hardware tests and test-only infrastructure.
- **Docs:** rustdoc, examples, user guides, design records, and migration/support matrices.

Generated bindings, vendored upstream source, lockfiles, build outputs, and formatting-only changes are excluded.
Generated artifacts are reported separately. Deletions do not reduce the estimate. Ranges are deliberately broad:
approximately ±30% for in-repository work and up to ±50% for upstream/compiler, cuTile, Triton, and ROCm seams until
their source-contract phases are complete.

The table budgets the dynamic Rust CUDA launcher branch because it has the largest known Phase 4 implementation.
Phase 0 replaces the Phase 2/4 rows with the documented branch-specific estimates when it selects the official FFI,
built-in plugin, or dynamic launcher route; alternatives are never summed.

| Phase | Primary scope | Production | Tests | Docs |
|---:|---|---:|---:|---:|
| 0 | Pinned source contracts and seam decisions | 150-300 | 250-450 | 300-500 |
| 1 | Common `ryft-xla-sys` registration/archive surface | 250-600 | 350-700 | 100-200 |
| 2 | `ryft-xla-sys` Mosaic GPU runtime/compiler bridge | 500-1,100 | 700-1,300 | 180-300 |
| 3 | `ryft-mlir` compiler dialects, binary IR, and Mosaic GPU | 9,000-15,000 | 4,500-8,000 | 1,500-3,000 |
| 4 | `ryft-cuda` launcher and `ryft-pjrt` adapters | 1,800-3,000 | 1,800-3,000 | 350-600 |
| 5 | GPU/cuTile lower-layer vertical-slice gate | 0-250 | 1,800-3,000 | 350-600 |
| 6 | Restore reference analysis and kernel validation | 2,400-3,400 | 3,000-4,500 | 450-700 |
| 7 | Core kernel types and semantic IR | 1,400-2,200 | 1,200-1,800 | 350-550 |
| 8 | Grids, block mappings, and bounds | 1,200-1,900 | 1,100-1,700 | 300-500 |
| 9 | References, scratch, atomics, and sync | 1,600-2,600 | 1,500-2,400 | 400-650 |
| 10 | Macro DSL, tracing, builders, and kernel call | 2,800-4,500 | 2,800-4,400 | 650-1,100 |
| 11 | Interpreter, diagnostics, and debugging | 1,100-1,800 | 1,600-2,400 | 350-550 |
| 12 | Portable primitives and scheduling | 1,500-2,400 | 1,300-2,100 | 350-600 |
| 13 | Adapter contract and XLA integration | 1,500-2,400 | 1,200-1,900 | 350-550 |
| 14 | `ryft-mosaic` GPU baseline | 1,800-2,900 | 1,500-2,400 | 400-700 |
| 15 | Hopper/Blackwell Mosaic GPU | 2,300-3,800 | 2,000-3,300 | 500-850 |
| 16 | `ryft-cutile` adapter and portability gate | 1,800-3,000 | 1,500-2,500 | 450-750 |
| 17 | Transforms and composition | 1,400-2,300 | 1,600-2,600 | 400-650 |
| 18 | Profiling, autotuning, persistence, AOT | 1,500-2,500 | 1,400-2,300 | 400-650 |
| 19 | Distributed and asynchronous kernels | 1,200-2,000 | 1,200-2,100 | 350-600 |
| 20 | Non-TPU stabilization and production hardening | 900-1,500 | 2,000-3,200 | 1,200-1,900 |
| 21 | Consolidated Mosaic TPU support and qualification | 5,300-9,200 | 5,200-8,800 | 1,750-3,000 |
| 22 | `ryft-triton` adapter and ROCm execution | 3,200-5,400 | 3,000-5,000 | 700-1,200 |

The arithmetic totals and crate split are recorded in §13 after the detailed phases. Adapter extraction reallocates
existing lowering work rather than adding another implementation. The existing Phase 7/12/13/14/16/20/22 ranges now
include contract, manifest/feature wiring, and dependency/portability checks; each phase must re-estimate if that work
exceeds its range. No additional direct-execution framework or universal runtime is budgeted.

The macro authoring frontend is new work:
Phase 10 adds 1,200-2,000 production, 1,500-2,300 test, and 250-450 documentation lines beyond its former builder-only
budget. This covers syntax transformation, staged control flow, signature generation, and diagnostics; it excludes
unrestricted Rust compilation. These increments are included in the table and aggregate totals.

## 8. Detailed implementation phases

Completed checkboxes and implementation records below are historical evidence and remain intact. Any earlier record
assigning future Mosaic normalization or compiler policy to `ryft-xla` is superseded by §3/§6 and the revised future
phase owners: compiler policy belongs to adapters, while XLA embedding belongs to `ryft-xla`. This ownership revision
does not reopen or claim new completion of Phases 0-6. No adapter implementation bypasses the Phase 5 lower-layer gate.

### Phase 0: Freeze non-TPU lower-layer surfaces and prove integration seams

**Prerequisites:** `plan-references.md` complete.

**Owners:** `ryft-xla-sys`, `ryft-mlir`, `ryft-cuda`, `ryft-pjrt`, planning/tooling only. No `ryft-core` or
`ryft-xla` production edits.

- [x] Diff the pinned JAX Mosaic GPU TableGen, C API, pass, compiler, serde, and custom-call sources against all local
      C++, Rust FFI, and typed MLIR wrappers. Defer the equivalent TPU inventory to dedicated Phase 21.
- [x] Inventory the standard Vector, Math, Complex, UB, Bufferization, Arith, MemRef, SCF, GPU, NVGPU, NVVM, LLVM,
      and conversion surfaces actually traversed by the pinned Mosaic pipelines; distinguish missing dialect coverage
      from operations already wrapped elsewhere.
- [x] Add direct, machine-checkable contracts for operations, attributes, types, interfaces, passes, translations,
      compiler options, artifact formats, and unsupported items at the pinned source revisions.
- [x] Pin the OpenXLA/JAX pairing and minimum CUDA, driver, and Python/tool requirements for each non-TPU tier.
      Record cuTile's current operating-system, architecture, compute-capability, driver, toolkit, and Python support
      as snapshot-sensitive toolchain facts rather than permanent Ryft policy.
- [ ] Prototype one Mosaic GPU module through the pinned serialized-module plus linked `mosaic_gpu_v2` PJRT route.
      Wrap a standalone compiler entry point only if Phase 0 proves that the pinned upstream exposes one.
- [ ] Prototype the cuTile AOT seam: export one cubin with `cutile_python_v2`, inspect its signature, and prove one
      supported XLA/PJRT launch route. Record licensing, redistribution, platform, security, timeout, and
      crash-isolation constraints.
- [ ] Audit and prototype the official `cutile_call` registration/configuration contract first. If it cannot support
      AOT and Python-free deployment, choose exactly one generic cubin-launch architecture: promote the experimental
      dynamic Rust CUDA driver launcher into the existing GPU XLA-FFI/custom-call path, or link a built-in CUDA-plugin
      handler. Do not implement both.
- [x] Decide the artifact schemas and error ownership before adding public wrappers.
- [x] Record unsupported targets and the CI hardware matrix.

**Tests/docs:** exact source-parity test; symbol/link probe on every supported build platform; minimal compile probes;
exact decision record for the cuTile seam.

**Excludes:** language IR, public APIs, broad wrappers, Python runtime embedding, and all TPU inventory or prototyping.

**Estimate:** production 150-300; tests 250-450; docs 300-500.

**Exit criterion:** every non-TPU lower-layer deliverable maps to a pinned upstream symbol/source or an explicitly
owned Ryft schema, every required compiler dialect is classified, and the GPU and cuTile seams have executable
prototypes. No exit check loads libtpu or requires TPU access.

**Implementation status (2026-08-31):** the direct source contracts, decisions, isolated probes, and
NVIDIA workflow are implemented. The three executable-prototype items remain unchecked until that workflow passes on
a supported NVIDIA runner; macOS ARM64 cannot satisfy the hardware gate.

### Phase 1: Complete the common `ryft-xla-sys` registration and archive surface

**Prerequisites:** Phase 0.

**Owners:** `crates/ryft-xla-sys/src/c++`, `src/mlir`, `BUILD.bazel`, archive/export configuration.

- [x] Reuse the existing upstream MLIR C APIs and `ryft-mlir` wrappers for pass managers, cloning, diagnostics, module
      ownership, parsing, and bytecode rather than adding a parallel native ABI.
- [x] Add source-owned C functions only for pinned Mosaic registration, version, analysis, serialization, or compiler
      results that lack a usable upstream C API. Associate compiler-stage labels in the owning caller unless a native
      Mosaic result exposes them.
- [x] Add explicit owned-byte/result destructors and null/error contracts for every genuinely new native result.
- [x] Complete the required Vector/Math and exact Complex/UB/Bufferization dialect-handle, C bridge, TableGen archive,
      and registration surface identified by Phase 0; archive UB definitions when parity tooling requires them.
- [x] Export all required headers, libraries, symbols, and platform link dependencies in source and release archives.
- [x] Make feature gating exact: CPU builds expose parsing/verification where supported but fail target compilation
      deterministically; CUDA artifacts expose only linked capabilities.
- [x] Generate ABI size/offset/version assertions and derive the required symbol set directly from pinned sources.

**Tests/docs:** C and Rust ABI layout tests for new functions, null/error/ownership tests, archive-content tests,
symbol parity, and Linux/macOS/Windows link probes where supported.

**Excludes:** Mosaic-specific policy, TPU-specific bridges, and any kernel-language semantics.

**Estimate:** production 250-600; tests 350-700; docs 100-200.

**Exit criterion:** every lower-layer phase can access its pinned registrations, versions, analyses, serialization,
and required compiler-dialect symbols through one safe ownership model, with generic MLIR behavior still owned by the
existing MLIR C API and `ryft-mlir`.

**Implementation status (2026-08-31):** complete. Phase 0 proved that no Mosaic result ABI belongs here, so no result
structs or destructors were added. The new source-owned surface is limited to the three missing dialect handles; the
direct source-parity and required-symbol gates cover the existing ABI, while cross-platform execution is enforced by CI.

### Phase 2: Complete the `ryft-xla-sys` Mosaic GPU runtime/compiler bridge

**Prerequisites:** Phase 1.

**Owners:** Mosaic GPU C++ bridge, Rust FFI, Bazel dependencies, CUDA release artifacts.

- [x] Close the pinned GPU dialect attribute/type accessor gaps; do not create per-operation C builders when generic
      MLIR construction plus typed Rust verification is sufficient.
- [x] Bind GPU serde/pass registration and any documented version or runtime-configuration functions exposed by the
      pinned source. Keep generic pass construction and diagnostics in `ryft-mlir`.
- [x] Expose a standalone target/lowering compiler bridge only if the pinned upstream provides a supported callable
      interface. Otherwise expose and document only the serialized Mosaic-module contract consumed by
      `mosaic_gpu_v2`; Phase 14 owns StableHLO custom-call construction and backend-configuration policy.
- [x] Return compiled object/PTX/cubin data, entry symbol, launch metadata, required shared/TMEM resources, and compiler
      diagnostics through an owned artifact where the pinned native interface exposes those stages. Otherwise return
      the verified serialized Mosaic module consumed by the linked runtime. Wrap a configuration producer only if
      Phase 0 identifies an actual supported upstream API.
- [x] Link JAX's Mosaic GPU custom-call/runtime, pass, serde, and target dependencies into the CUDA PJRT plugin, retain
      static registration in release artifacts, and prove that `mosaic_gpu_v2` is present. Keep CPU, ROCm, and
      unsupported platforms dependency-clean.
- [x] If Phase 0 selects a built-in plugin launcher, add the generic cubin/PTX handler here. If Phase 0 selects the
      dynamic Rust launcher, expose only the minimal stream/context contract it needs and leave promotion to Phase 4.
- [x] Keep native/compiler target representations open-ended and ensure CUDA plugin builds do not cap the newest known
      SM version. PJRT exposes raw device architecture; Phase 13 normalizes Hopper/Blackwell capabilities.
- [x] Preserve the JAX visibility patch only if the pinned source still requires it; delete it when upstream exports
      the necessary targets.

**Tests/docs:** exact C ABI tests, invalid module/option diagnostics, deterministic artifact metadata, CUDA build/link
matrix, and pinned-source parity tests.

**Excludes:** Rust kernel lowering, scheduling heuristics, and direct CUDA launch outside the one generic artifact
handler selected in Phase 0.

**Estimate:** production 500-1,100; tests 700-1,300; docs 180-300 for plugin linkage or the official reusable FFI
route. A selected built-in generic artifact handler raises production to 1,500-2,600 and tests to 1,100-2,000 while
reducing Phase 4 as described there; exactly one branch contributes to aggregate actuals.

**Exit criterion:** a caller can verify and serialize a hand-authored Mosaic GPU module under the documented
`mosaic_gpu_v2` module contract. A standalone compiled artifact is required only when the pinned upstream exposes that
supported interface; custom-call construction, runtime compilation, and execution are proven later.

**Implementation status (2026-08-31):** complete. The direct pinned-source contract has zero Phase 2 gaps. The pinned
source exposes no supported standalone compiler/artifact ABI, so the implemented contract is MLIR bytecode consumed by
`mosaic_gpu_v2`. CUDA-only linkage and a registration-retention test are wired for Linux CUDA; execution remains gated
on the NVIDIA workflow. Portable native tests prove the Mosaic TypeIDs/serde contract, all Complex/UB bridges, all 20
required compiler-pass symbols, and byte-reproducible archives.

### Phase 3: Complete binary MLIR, compiler dialects, and typed Mosaic GPU support in `ryft-mlir`

**Prerequisites:** Phase 0 and the required dialect/archive items from Phase 1. This work proceeds in parallel with
Phase 2's CUDA runtime linkage.

**Owners:** `crates/ryft-mlir/src/dialects`, module/operation parsing, pass/pipeline facades, tests.

- [x] Add byte-slice module/operation parsing over `MlirStringRef`, preserving arbitrary MLIR bytecode and structured
      parse diagnostics while retaining text conveniences.
- [x] Add source-owned typed Vector and Math dialect modules with operation-specific builders, traits, passes, exact
      renderings, and invalid verifier tests.
- [x] Add the exact Complex, UB, Bufferization, and conversion operations/passes required by the pinned Mosaic
      pipelines. Audit existing Arith, MemRef, SCF, GPU, NVGPU, NVVM, and LLVM coverage and fill deltas only.
- [x] Reconcile all local operations, types, and attributes directly with the Phase 0 pinned sources; add only missing
      items.
- [x] Replace permissive generic operand/result/attribute lists with typed constructors where the upstream operation
      has a stable contract, while retaining an explicit raw escape hatch for forward-compatible parsing only.
- [x] Add typed pass registration, supported pipeline construction, target verification, and analysis wrappers.
      Preserve upstream textual pipeline syntax where MLIR exposes no typed pass constructor; validate and snapshot
      it rather than wrapping strings in a cosmetic second API.
- [x] Verify region counts, successor counts, variadic segments, memory spaces, layouts, barriers, async tokens,
      WGMMA, TMEM, `tcgen05`, and scale operands before native verification.
- [x] Add canonical parse/print/accessor support for every admitted type and attribute.
- [x] Enforce exact source parity while keeping reviewed public Rust APIs source-owned and documented.

**Tests/docs:** binary/text round trips; one focused construction/accessor/module-verification/complete-rendering test
for every new or changed concrete operation, attribute, and type in module order; invalid arity/type/attribute tests;
parser failures; pass-pipeline snapshots; target-gated verification; and pinned parity checks.

**Excludes:** kernel-language operations, automatic scheduling, and Mosaic TPU dialect completion.

**Estimate:** production 9,000-15,000; tests 4,500-8,000; docs 1,500-3,000. Most of the range is the missing typed
Vector/Math and compiler-dialect surface; generated metadata remains excluded.

**Exit criterion:** `ryft-mlir` can parse binary Mosaic IR and construct, inspect, verify, transform, and serialize
every standard/Mosaic GPU construct required by the planned backend through typed APIs wherever upstream exposes a
typed contract.

**Implementation status (2026-08-31):** complete at the pinned source surface: Math 46/46 operations, Vector 39/39
operations plus 4/4 attributes, UB 2/2 operations plus its poison attribute, Bufferization 7/7 operations, the pinned
Complex number attribute, exact compiler passes, and Mosaic GPU 37/37 operations. Safe builders validate every stable
contract represented by their Rust inputs; richer Mosaic dialect-interface/layout constraints remain the native MLIR
verifier's explicit boundary. All 37 Mosaic operation fixtures pass native module verification and exact rendering.
The fresh macOS native archive completed on 2026-09-02, and the full `ryft-mlir` runtime suite then ran for the first
time: it exposed 84 fixture and rendering bugs in the typed Math, Vector, UB, and Complex modules (mismatched
`func.func` types in test fixtures, default-valued attributes that are always materialized, the `in_bounds`
attribute type of `vector.transfer_read/write`, and printer-format drift), all fixed; the suite is now 3121/3121.

### Phase 4: Complete generic Mosaic and cuTile kernel prerequisites in `ryft-cuda` and `ryft-pjrt`

**Prerequisites:** Phase 0 and the exact stream/plugin/runtime seams selected from Phases 1-2. Raw PJRT work may
proceed in parallel with typed MLIR completion.

**Owners:** producer-neutral artifacts, driver loading, launches, and caching in `ryft-cuda`; existing PJRT program,
stream, FFI, GPU custom-call, executable-metadata, topology, and profiling modules own only their runtime adapters.

- [x] Add only raw plugin/device/topology/extension/version and metadata accessors proven missing by the vertical
      slices. Normalized Mosaic and kernel semantic capabilities belong to `ryft-xla` Phase 13.
- [x] Wrap Pallas-relevant upstream extensions still missing at the selected pin, such as raw-buffer or phase-compile
      support, only when Phase 0 demonstrates that the chosen runtime path needs them.
- [x] Extend generic `Program` formats only when Mosaic ingestion requires a format not representable as current MLIR
      or HLO; do not add `Program::Mosaic` as a label over ordinary MLIR.
- [x] Prove XLA FFI/custom-call access to the backend stream and device buffers needed by a precompiled cubin.
- [x] Promote and harden the existing experimental CUDA artifact launcher if selected in Phase 0: add module unload
      and lifetime management, context/device-keyed caches, arbitrary symbols and cubin/PTX bytes, complete CUDA error
      names/strings, typed argument descriptors, concurrency, and exact stream ordering.
- [x] If a generic launcher needs an owned descriptor, define only a backend-neutral `CudaKernelArtifact` containing
      cubin/PTX bytes, symbol, target architecture, launch dimensions, immutable parameter/ABI descriptors, resource
      requirements, and ABI schema/version. Per-execution buffers, scalars, and pointers belong to a separate launch
      call frame. cuTile constraints, versions, tuple flattening, and `cutile_python_v2` translation stay in Phase 16.
- [x] If the built-in plugin launcher was selected in Phase 2, wrap that one implementation instead and delete the
      experimental dynamic launcher when its tests have migrated.
- [x] Preserve event/fence ownership, callback, error, and dropped-handle behavior. Add cancellation or timeouts only
      when the selected upstream API actually exposes them.
- [x] Keep AOT executable serialization and reload on the existing `Executable`/`LoadedExecutable` path.

**Tests/docs:** extension availability, unsupported plugin errors, raw metadata, generic artifact round trips,
malformed/wrong-architecture cubins, concurrent context/device module caches, stream lifetime, fence failure, AOT
reload, and shared-buffer aliases. Test cancellation/timeouts only for selected APIs that provide them.

**Excludes:** a Mosaic-specific PJRT extension, a second executable type, CUDA scheduling semantics, and TPU plugin
work.

**Estimate:** production 1,800-3,000; tests 1,800-3,000; docs 350-600 for promotion of the dynamic Rust launcher, the
branch used by the aggregate table. Reusing an official FFI handler is roughly production 400-1,200 and tests
900-1,700; wrapping a Phase 2 built-in handler is roughly production 400-1,000 and tests 800-1,500. Phase 0 selects
one branch and recalculates totals without summing alternatives.

**Exit criterion:** Mosaic remains ordinary MLIR/custom-call compilation through PJRT, while generic AOT CUDA
artifacts use exactly one safe stream launcher. PJRT exposes the raw facts and ownership needed by both without
knowing Mosaic or cuTile semantic schemas.

**Implementation status (2026-09-01):** complete. Existing PJRT programs, executable persistence, FFI streams,
buffers, topology, versions, metadata, fences, and callbacks already covered the non-launcher bullets, so no duplicate
extensions or `Program::Mosaic` variant were added. `ryft-cuda` owns the one producer-neutral launcher, its typed
immutable artifacts, borrowed raw CUDA resources, per-call launch frames, bounded context/device caches, retryable
cleanup, and exact unsafe context-lifetime contracts without depending on PJRT or XLA. `ryft-pjrt::cuda` only adapts
CUDA clients and XLA FFI streams/buffers. The experimental duplicate was removed, and cuTile metadata now drives the
single launch path. Portable mock/lifecycle tests pass; actual cubin/cuTile execution remains gated on NVIDIA CI.

### Phase 5: GPU/cuTile lower-layer vertical-slice gate

**Prerequisites:** Phases 1-4. This gate blocks every later non-TPU production phase.

**Owners:** lower-layer integration tests, fixtures, and reusable artifact-inspection/qualification tooling only.

- [ ] Construct one vector-add and one tiled matmul Mosaic GPU module through typed `ryft-mlir`, verify it, run the
      documented serde/pass stages, compile through the selected `mosaic_gpu_v2` seam, execute it through CUDA PJRT,
      and reload the enclosing executable from AOT where supported. Exercise standalone lowering only when Phase 0
      found a supported interface.
- [ ] Launch the Phase 0 cuTile cubin through the selected standard XLA/PJRT route.
- [x] Round-trip binary Mosaic GPU modules without text conversion and assert the serde version and custom-call target
      expected by the runtime.
- [ ] Assert the strongest supported target evidence: Mosaic GPU target IR/PTX when exposed and cuTile artifact
      architecture.
- [ ] Pin exact unsupported CPU, architecture, plugin, version, memory, and missing-tool diagnostics.
- [ ] Run address/thread/error sanitizers where supported and prove repeated compilation/execution has no leaks.
- [ ] Prove CPU-only builds and archive symbols remain clean; build supported CUDA 12 and CUDA 13 plugin matrices on
      Linux x86_64/aarch64; and keep ROCm, Windows, and macOS free of CUDA/cuTile dependencies where applicable.
- [ ] Reject malformed bytecode/configuration, wrong-SM and wrong-calling-convention artifacts, and stress the
      context/device module cache concurrently.

**Tests/docs:** two end-to-end vertical slices, compiler-only fallbacks, hardware CI instructions, and artifact
inspection tooling.

**Excludes:** `ryft-core`/`ryft-xla` production code and every TPU-specific build, runtime, fixture, or hardware check.

**Estimate:** production 0-250; tests 1,800-3,000; docs 350-600. Production lines are limited to reusable artifact
inspection or qualification tools; fixtures remain test code.

**Exit criterion:** GPU and cuTile lower-layer paths are real, inspectable, repeatable, and safe. If either path fails,
revise its seam before designing the language around it. TPU remains entirely deferred to Phase 21.

**Implementation status (2026-09-02):** the slices, tooling, and CI are implemented; hardware execution remains
gated on the NVIDIA workflow. Typed `ryft-mlir` vector-add and tiled-matmul Mosaic GPU modules mirror the pinned JAX
host ABI, verify natively, serialize through the exact `mosaic_gpu-serde` pipeline (version 6), round-trip as bytecode
without text conversion, and are wrapped in the exact `custom_call @mosaic_gpu_v2` program; their CUDA execution, AOT
reload, PTX evidence, and upstream diagnostics are env-gated tests run by `pallas_gpu_seam_probes.yaml` on a
CUDA 12/13 matrix with PTX dumps and `compute-sanitizer` memcheck/leak-check. `ryft-cuda` now inspects cubin ELF
headers and checks device compute capability before loading, so wrong-SM artifacts fail with prelaunch diagnostics on
every host; wrong-calling-convention artifacts and concurrent cache stress are covered portably. CPU archives are
proven free of Mosaic/CUDA/cuTile symbols and Linux CUDA plugins prove `mosaic_gpu_v2` retention. The fresh macOS
native archive also allowed the full `ryft-mlir` suite to run for the first time, which surfaced and fixed 84 Phase 3
fixture and rendering bugs (now 3121/3121). aarch64 GPU runners are unavailable, so aarch64 stays build-only.

### Phase 6: Restore reference program analysis and kernel validation foundations

**Prerequisites:** Phase 5 and the completed `plan-reference-discharge.md` architecture already in the repository.
This is the first phase permitted to make production changes in `ryft-core` or `ryft-xla`; it must not proceed in
parallel with Phases 0-5.

**Owners:** `ryft-core` reference analysis; `ryft-xla` experimental kernel validation.

The interpreter-style discharge rework deleted the static reference-analysis stack — the generic whole-closure
`ReferenceAnalysis` (`crates/ryft-core/src/programs/references/analysis.rs`), the array-view overlay
`ArrayReferenceAnalysis` (`crates/ryft-core/src/arrays/references.rs`), and the preserved-reference kernel
validator mock (`crates/ryft-xla/src/experimental/reference_kernels.rs`) — because after that rework its only live
production consumer was an entry-boundary fact (now an inline scan in the eager replay preflight) and its remaining
purpose was this plan's kernel work. The deleted files were never committed, so restoration re-implements against the
contracts recorded in `plan-references.md` (and approximate transcript-recovered copies archived at deletion time)
rather than reverting a commit.

- [x] Rebuild the generic `ReferenceAnalysis`: root/alias/access resolution, capture scopes, region-input bindings,
      region-output forwarding, transitive instruction summaries, and lifetime/second-class boundary validation over
      complete region closures (condition, while, scan, and call-like operations).
- [x] Evaluate a shared source-reference traversal that can supply the generic `ReferenceAnalysis` while allowing
      reference discharge to project only the caller-allocation reachability, accesses, and output identities needed
      by `ReferenceDischargeRegionSummary`. Keep discharge-specific allocation identities, boundary planning, and
      validation in the discharge transform; do not generalize that narrower summary into the source analysis or make
      Phase 6 depend on a discharge refactor unless the shared traversal produces a clear reduction in duplication.
- [x] Rebuild the array-view overlay (`ArrayReferenceAnalysis`) deriving each validated root-relative
      `ArrayReferenceView` from the generic alias edges exactly once.
- [x] Rebuild kernel-body validation on that analysis — second-class boundaries, declared access modes, liveness, and
      per-root alias/view maps for lowering — targeting the real kernel operation of the phases below rather than the
      deleted mock.
- [x] Keep the three-rung prevention ladder (trace time, eager runtime, discharge) as the default for non-kernel
      paths; reconnect the eager replay preflight to whole-closure validation only if kernel work demonstrates the
      entry-boundary scan is insufficient.

**Tests/docs:** restore the analysis test corpus (roots/aliases/accesses, capture scopes, nested-region root
substitution, and error diagnostics) and the kernel-boundary validation suites; document the analysis as kernel-owned
validation infrastructure rather than a standing whole-program lint.

**Excludes:** kernel types themselves (next phase) and any change to discharge, which remains the authority for
staged reference rewriting.

**Estimate:** production 2,400-3,400; tests 3,000-4,500; docs 450-700.

**Exit criterion:** a region closure containing references can again be statically analyzed for roots, aliases,
accesses, and lifetimes, with the kernel validator consuming that analysis, and no mandatory lint reintroduced into
any non-kernel path.

**Implementation status (2026-09-02):** complete. `ReferenceAnalysis` (value-level roots, alias edges with
transitive narrowing, accesses, capture scopes, nested-region bindings, identity constraints, root-only boundaries,
region access policies, lifetime rules, and per-instruction transitive summaries) lives in
`crates/ryft-core/src/programs/references/analysis.rs`; `ArrayReferenceAnalysis` derives each view once from the
generic alias edges through the new `ArrayReferenceViewOperation::reference_view_transform` method; and
`crates/ryft-xla/src/experimental/reference_kernels.rs` validates a standalone kernel body against a
`KernelBoundaryContract` (declared read-only/write-only/read-write access, no reference outputs, no captured or
consumed operands, swap store-versus-exchange by result liveness) that Phase 7 attaches to the real kernel operation.
The shared-traversal evaluation kept `ReferenceDischargeRegionSummary` untouched because a shared loop would have
generalized that narrower allocation-space summary rather than removing duplication. The eager replay boundary scan
and the three-rung ladder are unchanged.

### Phase 7: Add backend-neutral kernel types and semantic IR

**Prerequisites:** Phases 5-6.

**Owners:** `ryft_core::kernels`, reusing existing core modules; no target dialect or runtime dependency.

- [ ] Promote the proven experimental boundary into a backend-neutral higher-order `KernelCallOperation` and region.
- [ ] Define grid, parameter, static-argument, access, alias, scratch, source-location, capability-requirement, and
      portable policy types without backend-specific fields. Keep backend binding/options outside the definition.
- [ ] Reuse `ArrayType`, `ReferenceType`, `ArrayReferenceView`, the Phase 6 restored `ReferenceAnalysis`, effects,
      identities, and parameter structures rather than copying them.
- [ ] Define kernel-specific effect/resource classes for memory, async operations, barriers, and atomics while keeping
      ordinary program effect semantics intact.
- [ ] Specify deterministic display, hashing, equality, identity renaming, refinement, serialization eligibility, and
      source schema version.
- [ ] Define the core-owned extension contract from §3.3 and parameterize the kernel operation family over it. Use
      an empty extension family for portable kernels; adapters supply concrete families in their owning phases.
      Reuse existing operation traits, reject opaque semantic payloads, and avoid a central backend enum in core.
- [ ] Define the logical signature and verified-body boundary from §6.6-§6.7. Keep physical launch types, native
      artifacts, and XLA envelopes outside core; semantic rewrites require renewed validation.
- [ ] Add dependency checks proving core contains no adapter/native compiler/runtime dependency, including through
      features. Serialization and interpretation of the portable family must work with core alone.

**Tests/docs:** exhaustive type/value/operation projections, malformed regions, deterministic render/hash, reference
root/view/access reuse, no target enum leakage, and compile-fail examples for escaping references.

**Excludes:** grids' indexing behavior, lowering, public convenience macros.

**Estimate:** production 1,400-2,200; tests 1,200-1,800; docs 350-550.

**Exit criterion:** a kernel is a first-class staged operation with a stable semantic identity but no backend or
runtime assumptions in its core types.

### Phase 8: Add grids, block mappings, indexing, and bounds

**Prerequisites:** Phase 7.

**Owners:** `ryft-core` kernel modules and array indexing utilities.

- [ ] Implement grid rank/extents, named dimension semantics, program IDs, number of programs, and scalar prefetch.
- [ ] Define block shapes and pure block mappings with static result shape and bounded starts.
- [ ] Add dynamic slice/index components, broadcasting advanced indices, masks, padding/`other`, and boundary policies.
- [ ] Prove mapping purity and reject reference/data-dependent mappings.
- [ ] Add checked index arithmetic, dynamic-bound refinement, and precise overlap/disjointness summaries.
- [ ] Define specialization of static parameters and dynamic grid bounds without specializing on incidental pointer
      alignment or runtime values.

**Tests/docs:** grid enumeration, mapping/oracle equivalence, singleton and zero-extent grids, logical-to-physical rank
mapping, overflow, masks, partial tiles, dynamic bounds, overlapping mappings, scalar prefetch, and stable
specialization keys.

**Excludes:** scratch, atomics, async scheduling, target layouts.

**Estimate:** production 1,200-1,900; tests 1,100-1,700; docs 300-500.

**Exit criterion:** every operand window and boundary behavior is statically described or explicitly masked before a
kernel body executes.

### Phase 9: Complete references, scratch, atomics, and synchronization

**Prerequisites:** Phases 7-8.

**Owners:** `ryft-core` kernel verifier and operations.

- [ ] Add scoped uninitialized/initialized scratch allocation with memory-space, shape, layout eligibility, alignment,
      lifetime, and non-escape validation.
- [ ] Implement path-sensitive definite initialization through conditions and fixed-point loops. Derive each transfer
      from the exact access mode together with its root-relative view, mask, and proven coverage; do not add a generic
      whole-reference initialization classifier to `ReferenceAccessMode`, because a partial or masked write does not
      necessarily initialize its complete root.
- [ ] Add masked load/store/swap, ordered accumulation, device-scoped sequentially consistent atomics, and async-copy
      tokens/waits with exact operation-local reference semantics. Define the typed extension contract used later by
      target barriers and semaphores without pretending they are portable operations.
- [ ] Introduce the generic `ReferenceAccessMode::AtomicAccumulate` mode and `reference_atomic_add_update` operation
      carrying the atomic/commutative semantics that the `Accumulate` documentation in
      `crates/ryft-core/src/programs/references/semantics.rs` explicitly excludes from the ordered mode (§5.6). The
      fixed exact-mode set forces every classifier, summary, display rendering,
      and liveness action phrase to handle the variant at compile time. Region policies admit it wherever ordered
      `Accumulate` is admitted and additionally through parallel grid inputs, where ordered `Accumulate` is rejected.
      Discharge reuses `ReferenceAccumulationPolicy`, because sequential replay is an admitted execution of the
      commutative contract. The update operand stays linear and therefore transposable. The first release keeps
      `EffectClass::OrderedState` so that only the kernel boundary exploits same-root commutation; generic transforms gain
      no reordering rights.
- [ ] Define the portable sequentially consistent atomic model and device scope. Reject invalid combinations
      in core; defer otherwise valid but unavailable dtype/scope combinations to backend capability selection.
- [ ] Add root/view overlap and race analysis for common affine block mappings.
- [ ] Preserve dead swap-result store optimization only when old contents and untouched root elements are provably
      unnecessary.
- [ ] Record in this plan that the preserved-reference boundary formerly captured in `plan-references.md` is
      superseded by this production contract. Its `reference_kernels.rs` mock was already deleted when the
      interpreter-style discharge landed, so no mock-removal coordination remains.

**Tests/docs:** all access modes; write-only full-output initialization and publication; ordered repeated stores;
empty-grid/nonempty-output rejection; uninitialized reads; partial writes; escape/use-after-scope; sibling views;
masked accesses; atomic contract tests, including ordered-versus-atomic accumulation admission through sequential and
parallel regions; token linearity; statically provable async ordering/races; and liveness.

**Excludes:** unsafe arbitrary races and target encodings.

**Estimate:** production 1,600-2,600; tests 1,500-2,400; docs 400-650.

**Exit criterion:** the verifier defines and enforces the decidable static memory, initialization, atomic, resource,
and synchronization contract, and rejects programs outside its proof subset. Dynamic interleaving, deadlock, and
execution conformance become executable acceptance criteria in Phase 11.

### Phase 10: Add the macro DSL, tracing integration, builders, and public kernel-call surface

**Prerequisites:** Phases 7-9.

**Owners:** `ryft-macros` procedural frontend, `ryft_core::kernels` authoring facade over existing core
compilation/tracing/operations, and `ryft-macros-tests` integration coverage; `ryft` portable re-exports after
stabilization. Macro expansion depends on core APIs only in emitted code, preserving the dependency direction.

- [ ] Add typed builders for kernels, grid/block specs, scratch specs, static parameters, and portable schedules.
      Keep typed compiler options in adapter crates and backend binding in the execution integration.
- [ ] Demonstrate one reusable portable definition staged without an adapter and prepare it for the Phase 11
      core-only interpreter gate. Record its logical specialization/fingerprint so Phase 16 can compile that same
      body through two adapters.
- [ ] Implement `#[kernel]`, shape requirements, and consumed input/output attributes from §3.4 in `ryft-macros`,
      re-exported through `ryft_core::kernels`. Use `Array` and canonical dtype/rank metadata rather than new matrix
      or typed-array types.
- [ ] Generate typed staging functions and functional outer-call metadata; infer unambiguous output grids, create
      disjoint output views, and enforce full-array shape constraints separately from tile-shape constraints.
- [ ] Translate the §3.5 supported Rust subset into capability calls and explicit staged control flow. Carry mutable
      locals through nested loop/branch parameters and reject unsupported host behavior or control flow.
- [ ] Trace generated and explicit closures through existing root/nested tracing, `StagingContext`, and
      `Context::bind`, preserving reference restrictions. Reuse `InterpretableOperation` for tracer-valued replay;
      no new tracing graph, direct AST-to-native lowering, or macro-specific operation semantics.
- [ ] Preserve fallible staging behavior, poison/finalization diagnostics, source spans, and hygienic emitted paths.
      Document supported helper/capture rules and expose expansion plus canonical IR for inspection.
- [ ] Admit the minimal existing scalar-index, `f32` zero/add/dot, and structured-control-flow capabilities needed
      to stage the §3.4 example through the kernel operation family. Reuse their canonical type/numerical contracts;
      Phase 11 interprets this subset and Phase 12 completes the wider portable operation/schedule coverage.
- [ ] Infer parameter trees, result trees, aliases, source locations, and static specialization constraints.
- [ ] Add user-facing load/store/indexing sugar only over the canonical operations; no second semantic path.
- [ ] Seal regions atomically after validation and preserve exact diagnostics through nested helper calls.
- [ ] Keep the surface experimental and out of broad crate-root exports until Phase 20.

**Tests/docs:** macro parse/expansion tests in `ryft-macros`; compile-pass/fail integration in `ryft-macros-tests`;
isolated core-only macro consumer; renamed-crate/import hygiene; parameter/capture restrictions; output/grid inference;
full reduction-dimension mismatch despite compatible tiles; zero/non-multiple dimensions; loop-carried accumulators;
value-dependent branches; unsupported syntax diagnostics; stable source spans; no panic on staging failure; and
semantic IR/effect equivalence between macro and explicit builder definitions. Read the repository's unit-testing
guidelines before implementing these tests.

**Excludes:** unrestricted Rust parsing/type resolution, arbitrary host callbacks or AST interpretation at runtime,
opaque macro semantics, implicit backend selection, and a second tracer/compiler/JIT implementation.

**Estimate:** production 2,800-4,500; tests 2,800-4,400; docs 650-1,100. Includes the macro frontend increment.

**Exit criterion:** the §3.4 matmul and supported control-flow examples stage through ordinary Ryft tracing into the
same normalized semantic IR/effects as explicit builders, with precise failures. Phase 11 proves interpreter results
and Phase 16 proves both backend lowerings using these same macro-authored definitions.

### Phase 11: Build the semantic interpreter and debugging model

**Prerequisites:** Phases 7-10.

**Owners:** `ryft-core` interpreter and test utilities; thin `ryft-xla` debug integration later.

- [ ] Execute grids, block mappings, references, scratch, masks, sequentially consistent atomics, async completion
      tokens, and bounded control flow deterministically on host arrays.
- [ ] Add configurable race, bounds, initialization, NaN/precision, and async-token diagnostics.
- [ ] Model concurrency with a deterministic scheduler capable of exploring small interleavings for litmus tests.
- [ ] Preserve source locations and render an execution trace with grid point, operation, reference root/view, and
      synchronization state.
- [ ] Add property generators for small well-typed kernels and shrink failing cases.

**Tests/docs:** operation semantics, random interpreter/oracle checks, race/OOB/token reports, deterministic traces,
and tutorial debugging workflows. Target barrier/deadlock simulation belongs to its backend phase.

**Excludes:** performance, exact device timing, emulator claims for undocumented hardware behavior.

**Estimate:** production 1,100-1,800; tests 1,600-2,400; docs 350-550.

**Exit criterion:** every portable kernel has an accelerator-independent executable specification suitable for
backend conformance testing.

### Phase 12: Complete portable value operations and scheduling contracts

**Prerequisites:** Phases 7-11.

**Owners:** `ryft-core` portable kernel operations and schedule metadata.

- [ ] Admit the scalar/tile arithmetic, reductions, shape operations, dot, and block-scaled-dot families required by
      representative elementwise, reduction, matmul, convolution, and attention kernels.
- [ ] Define portable layout constraints, pipeline stages, buffering depth, placement, and resource budgets as optional
      schedules. Target agent membership, collective mode, and synchronization topology are semantic fields in the
      typed target launch/operation contract and are never erased as schedule hints.
- [ ] Canonicalize and DCE pure work while retaining effects and all live target resources.
- [ ] Validate operation-specific precision, rounding, saturation, and accumulator behavior.
- [ ] Specify the minimal compiler capability in §6.6 using the Phase 5 GPU/cuTile native probes and existing
      compilation-domain hooks. Prototype only the necessary typed boundary; do not confuse hand-authored native
      probes with completed DSL lowerers. Validate the contract in Phases 14/16 before stabilizing it in Phase 20.
- [ ] Separate portable semantic requirements from adapter target/options and associated output types. Add no
      universal launcher, central target enum, or adapter registry to core. Phase 21 adds only proven TPU needs.

**Tests/docs:** exact inference, folding, liveness, schedule validation, numerical edge cases, block scaling, and
interpreter equivalence for representative kernels.

**Excludes:** automatic schedule search and target instruction selection promises.

**Estimate:** production 1,500-2,400; tests 1,300-2,100; docs 350-600.

**Exit criterion:** the portable subset can describe useful kernels and all result-preserving schedule hints can be
erased without changing results; target execution-agent and synchronization contracts remain intact.

### Phase 13: Connect compiler adapters to XLA dispatch, ABI, caching, and execution

**Prerequisites:** Phases 5-12.

**Owners:** `ryft_core::kernels` minimal compiler contract; `ryft_xla::kernels` experimental operation integration,
compilation-domain hooks, selection, embedding, and persistence. Concrete compiler implementations follow in their
adapter phases; shared native wrappers and platform artifacts retain their existing owners.

- [ ] Add the higher-order kernel call to the XLA operation family and preserve its body until backend selection.
- [ ] Implement the four ownership contracts in §6.7: core logical signature, adapter output, platform launch ABI,
      and XLA embedding/persistence envelope. Consume canonical types without duplicating metadata.
- [ ] Bind a portable definition to typed adapter options at the integration boundary; preserve body identity across
      compiler choices. Keep any heterogeneous adapter/extension enum here, not in core or a low-level runtime.
- [ ] Join adapter-normalized target/compiler facts with PJRT execution facts before selection or compilation.
      Keep core semantic capabilities independent of compiler-version and device-name enums.
- [ ] Add explicit enabled-adapter selection with exact user override, portable fallback policy, and target-operation
      rejection.
- [ ] Embed backend artifacts in StableHLO custom calls with operation-local aliases and side effects, never external
      reference-state entry aliases.
- [ ] Reuse existing core compilation-domain dispatch/cache hooks, PJRT fences, persistent executables, and
      replacement validation. Do not create a kernel-specific JIT lifecycle or duplicate the CUDA module cache.
- [ ] Test typed ready-artifact and deferred-payload integration with hand-authored fixtures. Confirm an adapter
      output can be constructed and validated without an XLA-owned type or dependency.
- [ ] Return host-asynchronous single-host execution through existing fences from the first backend; awaiting a fence
      must not require a second kernel-specific completion type.
- [ ] Key caches by §6.3 and make compiler crashes/timeouts/cancellation non-poisoning to unrelated calls.

**Tests/docs:** exact StableHLO ABI, zero external-state slots, capability failures, cache separation, persistence
corruption, dropped execution, replacement mismatch, and reference-kernel interaction.

**Excludes:** backend lowering details and public stable API.

**Estimate:** production 1,500-2,400; tests 1,200-1,900; docs 350-550.

**Exit criterion:** a verified kernel has deterministic ABI/configuration serialization, backend dispatch, cache
identity, persistence validation, and host-asynchronous fence plumbing, demonstrated with a mock or hand-authored
lower-layer artifact. Production lowering and device execution begin in Phase 14.

### Phase 14: Implement the Mosaic GPU baseline

**Prerequisites:** Phases 5-13.

**Owners:** new `ryft_mosaic::kernels::gpu` lowerer, extensions, options, and target simulation using `ryft-mlir`;
`ryft_xla::kernels` owns only selection and StableHLO/PJRT embedding.

- [ ] Create the GPU-only `ryft-mosaic` adapter with the dependency rules in §3.2. Compile/validate adapter outputs
      in tests without linking `ryft-xla`; enable no TPU compiler/runtime dependency in this phase.
- [ ] Lower grids to CUDA block/CTA launch semantics and portable references to GMEM/SMEM/register operations.
- [ ] Lower scalar/tile arithmetic, masks, layouts, slices, broadcasts, reductions, ordinary dot, and bounded control
      flow for the supported Hopper-or-newer baseline.
- [ ] Lower Mosaic GPU target barriers and basic asynchronous GMEM/SMEM transfers with declared agent membership and
      verified lifetimes.
- [ ] Implement deterministic target simulation for baseline CTA/barrier/async behavior and compare ordering,
      deadlock, and race outcomes with GPU execution.
- [ ] Produce adapter-owned launch/resource metadata and compiler payloads; in `ryft-xla`, derive and verify
      StableHLO custom-call aliases/effects from the core logical contract and adapter physical mapping.
- [ ] Add target legality for compute capability, CUDA/PTX/toolkit versions, shapes, layouts, and resource limits.
- [ ] Compare interpreter, immutable oracle, Mosaic MLIR, target IR/PTX, and device results.

**Tests/docs:** exact MLIR snapshots, verifier negatives, PTX feature assertions, GPU numerical suites, resource-limit
errors, and vector/reduction/matmul examples.

**Excludes:** WGMMA/TMEM/tcgen05, automatic pipelines, clusters, multi-GPU.

**Estimate:** production 1,800-2,900; tests 1,500-2,400; docs 400-700.

**Exit criterion:** representative portable kernels execute correctly through Mosaic GPU on the baseline supported
architecture with inspectable code generation.

### Phase 15: Add advanced Hopper and Blackwell Mosaic GPU support

**Prerequisites:** Phase 14.

**Owners:** `ryft_mosaic::kernels::gpu` target extensions, schedules, capability tables, and qualification tests;
`ryft-xla` consumes their typed output and execution requirements.

- [ ] Add Hopper WGMMA, TMA descriptors/transfers, warpgroup scheduling, barrier semantics, swizzles, and pipeline
      generation.
- [ ] Add Blackwell TMEM allocation/lifetime, `tcgen05` MMA/commit/wait, tensor-core-ordering barriers, SMEM/TMEM scale
      transfers, cluster and two-CTA collective modes where supported.
- [ ] Extend target simulation to warpgroup, cluster, TMEM, collective-MMA, and tensor-core-ordering resources.
- [ ] Lower portable block-scaled dot to a compatible optimized path and expose an exact Mosaic GPU `tcgen05`
      operation family for manual control.
- [ ] Implement the complete NVFP4 contract from §5.5, including scale packing/geometry and FP32 tensor scale.
- [ ] Add sparse block-scaled forms only after dense behavior and metadata are stable.
- [ ] Include architecture/feature/compiler choices in cache identity and reject missing features before native
      compilation.
- [ ] Add target-code inspection that proves WGMMA, TMA, TMEM, or `tcgen05` use; numerical parity is insufficient.

**Tests/docs:** Hopper and Blackwell MLIR/PTX snapshots, barrier-order litmus tests, TMEM lifetime failures, NVFP4
bit-pattern and tolerance tests, at least matmul and attention kernels, architecture mismatch, and scheduled benchmarks.

**Excludes:** pretending Blackwell-specific operations have portable emulation and requiring the newest hardware for
the baseline backend.

**Estimate:** production 2,300-3,800; tests 2,000-3,300; docs 500-850.

**Exit criterion:** portable and explicit advanced GPU kernels use the intended hardware instructions with correct
ordering and numerics, and fail exactly elsewhere.

### Phase 16: Implement the optional Ryft-to-cuTile compiler backend

**Prerequisites:** Phases 4-5, Phase 12, Phase 13, and a still-supported Phase 0 seam to start implementation.
Phase 14 must also pass before the two-compiler portability tests and this phase's exit gate can complete; cuTile
lowering may otherwise proceed in parallel with the Mosaic GPU baseline.

**Owners:** new `ryft_cutile::kernels` compiler adapter/tool driver; `ryft-cuda` shared artifact/launch runtime;
`ryft-pjrt` stream/FFI adaptation; `ryft_xla::kernels` embedding and selection.

- [ ] Create `ryft-cutile` with core and CUDA artifact dependencies, independent of `ryft-xla` and `ryft-pjrt`.
      Keep the compiler tool optional and absent from the launcher/runtime-only dependency path.
- [ ] Define the exact portable subset compatible with cuTile's block-level tile model, immutable local objects,
      global arrays, control flow, atomics, and no explicit intra-block synchronization.
- [ ] Translate verified kernel IR to deterministic cuTile source or an official compiler input; never translate
      arbitrary target-specific Mosaic operations.
- [ ] Compile in an isolated, cancellable, time-limited build-time process and export a `cutile_python_v2` cubin and
      manifest for the exact target GPU.
- [ ] Interpret the cuTile-owned manifest in `ryft-cutile` and produce the canonical `ryft-cuda` artifact plus a
      validated logical-to-physical mapping: pointer/shape/stride expansion, static-shape constraints, constants,
      tuples, entry symbols, alignment, no-alias requirements, grid, and compiler hints. `ryft-xla` consumes this
      output through the Phase 13 embedding contract; the adapter never imports the XLA persistence envelope.
- [ ] Execute through the one Phase 4 stream/custom-call launcher, retain compiler logs, and keep the runtime
      Python-free.
- [ ] Include cuTile/compiler/CUDA versions and target GPU in adapter compatibility metadata and cache/AOT identity.
- [ ] Compile the unchanged Phase 10 portable definitions through both Mosaic GPU and cuTile. Compare core semantic
      identity, oracle/interpreter results, outputs, aliases, and effects; keep backend options outside the body.
      Include the §3.4 macro-authored matmul, elementwise, masked boundary-tile, reduction, and supported dot cases
      plus an exact Mosaic rejection. Compare explicit-builder and macro definitions before adapter lowering.
- [ ] Validate the minimal adapter contract against both real lowerers. Remove provisional duplication, check isolated
      and combined Cargo feature graphs, and document unsupported subsets before transform/release qualification.
- [ ] Support portable FP4/block-scaled operations only when the selected cuTile version documents them; exact Mosaic
      operations remain rejected.

**Tests/docs:** generated-source snapshots, AOT manifest/calling convention, tool failure and timeout, compiler sandbox,
numerical parity, cubin target inspection, no-alias failures, and NVIDIA device execution.

**Excludes:** a second cubin loader/launcher, embedding Python, translating TileIR without a documented stable API,
explicit barrier-heavy kernels, and using cuTile as the portable IR.

**Estimate:** production 1,800-3,000; tests 1,500-2,500; docs 450-750. This range has the highest pre-Phase-0
uncertainty.

**Exit criterion:** supported portable kernels compile ahead of time and execute through cuTile with a versioned,
auditable, Python-free runtime artifact; unsupported kernels fail before tool invocation. The same portable definitions
also compile through Mosaic GPU without body changes, and the two adapters have no `ryft-xla` dependency.

### Phase 17: Add transforms, composition, and sharding

**Prerequisites:** Phase 13 plus the selected non-TPU backend: Phase 14 for Mosaic GPU or Phase 16 for cuTile. Phase 15
is required only for transforms over advanced GPU target extensions. TPU transform work remains in Phase 21.

**Owners:** `ryft-core` transform rules; compiler adapters own post-transform target legality; `ryft-xla` owns
outer-program composition and execution integration.

- [ ] Implement batching as a grid/block-mapping transform with write-conflict validation.
- [ ] Add explicit custom JVP/VJP rules and pure fallback differentiation; keep implicit mutable-body AD rejected.
- [ ] Specialize static arguments and block mappings through partial evaluation without executing effects.
- [ ] Preserve kernel calls as indivisible effectful operations under rematerialization unless an explicit pure rule is
      provided.
- [ ] Compose kernel calls with condition/while/scan/call, external references, shard maps, and device-memory transfers.
- [ ] Add sharding rules for local per-shard launches and reject unsupported automatic partitioning, collectives, and
      overlapping writes.

**Tests/docs:** batched grid equivalence, conflict negatives, custom AD versus pure oracle, remat non-duplication,
partial specialization, control-flow sequencing, shard-map execution, and exact unsupported diagnostics.

**Excludes:** general internal AD, automatic distributed kernel synthesis.

**Estimate:** production 1,400-2,300; tests 1,600-2,600; docs 400-650.

**Exit criterion:** every public transform and higher-order composition has one proven rule or one early exact
rejection, with no accidental effect duplication.

### Phase 18: Add profiling, autotuning, persistence, and AOT workflows

**Prerequisites:** Phase 13 plus the participating non-TPU backend: Phase 14 for Mosaic GPU or Phase 16 for cuTile.
Phase 15 metadata is required only when profiling or persisting advanced GPU extensions; TPU work stays in Phase 21.

**Owners:** existing core compilation lifecycle; compiler adapters own compiler reports and compatibility metadata;
`ryft-xla` owns profiling correlation, tuning integration, and executable persistence; `ryft-pjrt` changes only for
proven profiling gaps.

- [ ] Attach source-aware kernel metadata to PJRT/XLA profiling and backend compiler reports.
- [ ] Report compile stages, target IR, register/scratch/TMEM use, occupancy, spills, and launch timing where the
      backend exposes them.
- [ ] Define bounded, deterministic schedule-search spaces and an explicit tuning budget.
- [ ] Store measurements with device/compiler/environment fingerprints; reject stale results.
- [ ] Make tuning concurrency-safe, cancellable, reproducible, and isolated from the ordinary executable cache.
- [ ] Export/import complete AOT bundles with semantic IR, backend artifact, metadata, compatibility manifest, and
      optional fallback policy.
- [ ] Add golden performance thresholds only after stable baselines and variance controls exist.

**Tests/docs:** metadata correlation, cache corruption/version mismatch, concurrent tuning, timeout/cancellation,
deterministic search, AOT relocation/reload, target incompatibility, and benchmark methodology.

**Excludes:** unconstrained auto-scheduling, machine-learning cost models, cross-machine artifact acceptance without
validation.

**Estimate:** production 1,500-2,500; tests 1,400-2,300; docs 400-650.

**Exit criterion:** developers can inspect, tune, persist, and deploy kernels reproducibly without weakening semantic
or compatibility checks.

### Phase 19: Add distributed coordination and asynchronous cross-host transfers

**Prerequisites:** Phases 17-18 plus the relevant Phase 14/15 GPU contract. TPU distributed work stays in Phase 21.

**Owners:** `ryft-core` launch semantics, `ryft-xla` runtime, existing `ryft-pjrt` distributed/transfer APIs.

- [ ] Define process/device launch IDs, collective ordering, cross-host failure propagation, and artifact agreement.
- [ ] Thread asynchronous kernel completion into existing execution fences and external-reference generation/lease
      chains without backend side maps.
- [ ] Add asynchronous input/output transfers and remote buffers with explicit lifetime and cancellation.
- [ ] For aliased distributed outputs, add coordinator epochs with prepare, commit, and abort records. Publish only
      after every participant prepares. Use shadow output buffers until commit; an exclusive donated input may be
      reused only when no retained or external alias exists and uncertain completion poisons it until reconciliation.
      If that protocol is not supported, admit only functionally returned outputs without observable in-place state.
- [ ] Reject distributed launches on plugins lacking a trustworthy collective fence and ordering contract.
- [ ] Add topology-aware artifact selection without compiling one process against a different target contract.

**Tests/docs:** deterministic multi-process ordering, mismatched launch IDs/artifacts, cancellation, dropped handles,
partial host failure, collective failure, remote-buffer lifetime, and supported topology matrix.

**Excludes:** transparent distributed mutation or best-effort recovery that can expose divergent state.

**Estimate:** production 1,200-2,000; tests 1,200-2,100; docs 350-600.

**Exit criterion:** every admitted distributed launch has one completion and failure chain across all participants,
plus atomic publication or an explicit no-external-alias restriction; unsupported coordination is rejected before
submission.

### Phase 20: Stabilize non-TPU APIs, documentation, CI, and production quality

**Prerequisites:** Phases 14-19 for the selected non-TPU backends; Phase 19 may remain experimental if it is not
release-ready.

**Owners:** all touched crates and the `ryft` facade.

- [ ] Require the §3 dependency boundaries and §10 portability gates with both Mosaic GPU and cuTile, including
      backend-free core builds, adapter-only compilation tests, and no backend-specific fields in portable definitions.
- [ ] Publish macro-first authoring docs under `ryft_core::kernels`, including `Array` annotations, staged versus
      static control flow, supported helpers, shape/boundary validation, and generated diagnostics. Keep explicit
      builder docs available. Publish compiler/extension docs under each adapter and runtime binding docs under
      `ryft_xla::kernels`; show changing the compiler without changing a portable macro-authored kernel body.
- [ ] Choose the stable portable surface and keep Mosaic GPU, cuTile, raw artifact, and exact target operations
      explicitly experimental until their upstream ABIs stabilize.
- [ ] Remove mock boundaries, temporary bridges, parallel metadata, deprecated names, and compatibility shims; update
      all in-repo users directly.
- [ ] Publish a support matrix by backend, architecture, dtype, operation, memory space, transform, distribution, and
      toolchain version for the non-TPU release. Phase 21 adds TPU rows without rewriting existing contracts.
- [ ] Add end-to-end examples: elementwise, reduction, matmul, attention, masked partial tiles, scratch pipeline,
      Hopper WGMMA, Blackwell NVFP4 `tcgen05`, cuTile, custom AD, batching, sharding, AOT, and debugging.
- [ ] Establish upgrade tooling that directly compares pinned Mosaic GPU surfaces and forces an explicit decision for
      every addition, removal, or semantic change. Phase 21 adds the isolated TPU contract workflow.
- [ ] Run compiler fuzzing, malformed artifact tests, sanitizers, concurrency stress, long-run leak tests, and hardware
      qualification.
- [ ] Set compile-time, binary-size, runtime, numerical, and benchmark regression budgets.
- [ ] Require independent correctness, convention, security, and complexity audits with zero remaining findings.

**Tests/docs:** complete named-family matrix in §10, public doctests, examples, release qualification, migration and
troubleshooting guides, and exact verification record.

**Excludes:** declaring upstream experimental APIs stable by documentation alone.

**Estimate:** production 900-1,500; tests 2,000-3,200; docs 1,200-1,900.

**Exit criterion:** the supported contract is understandable without implementation knowledge, reproducibly qualified,
and contains no temporary or redundant architecture.

### Phase 21: Add and qualify complete Mosaic TPU support

**Prerequisites:** Phase 20. No earlier phase, milestone, CI tier, or release gate may require libtpu, TPU compiler
metadata, a TPU PJRT plugin, or TPU hardware.

**Owners:** `ryft_mosaic::kernels::tpu` lowering, extensions, simulation, compiler payloads, and capability policy;
TPU native work in `ryft-xla-sys`/`ryft-mlir`/`ryft-pjrt`; `ryft-xla` embedding/execution, plus TPU fixtures,
documentation, CI, and qualification. Portable `ryft-core` semantics are already complete and change only if a reviewed
backend-neutral defect is found.

- [ ] Freeze the pinned JAX Mosaic TPU, OpenXLA, libtpu/PJRT plugin, serde, custom-call, and hardware support matrix;
      add a direct TPU-specific parity contract and record every existing, partial, missing, or unsupported surface.
- [ ] Complete the open-source `ryft-xla-sys` TPU bridge: safe construction/accessors for `VectorLayoutAttr`,
      `TiledLayoutAttr`, and other required types; serde/pass registration; communication/custom-barrier analysis;
      documented bytecode/version functions; archive/export symbols; and exact ownership/error contracts.
- [ ] Treat libtpu as the closed-source runtime/compiler owner. Submit supported custom calls through PJRT and consume
      only documented diagnostics, metadata, serialization, and profiling hooks; do not bind private compiler internals.
- [ ] Reconcile the existing 86-operation `ryft-mlir` Mosaic TPU surface with the pinned contract. Complete required
      types, attributes, interfaces, operations, typed constructors/accessors, serde/pass APIs, layout inference,
      communication analysis, memory spaces, and canonical parsing/rendering.
- [ ] Add one focused construction/accessor/module-verification/complete-rendering test for every new or changed
      concrete TPU operation, attribute, and type, plus exact malformed DMA/semaphore/layout/MXU cases and serde/pass
      snapshots. Keep standard Vector/Arith/SCF/MemRef behavior in its owning dialect.
- [ ] Add only raw TPU plugin/device/topology/version/extension facts genuinely missing from `ryft-pjrt`. Keep Mosaic
      capability normalization, target extension schemas, compiler payloads, and compiler policy in `ryft-mosaic`;
      keep StableHLO embedding and executable persistence in `ryft-xla`.
- [ ] Pass the TPU lower-layer gate with hand-authored VMEM, HBM transfer, DMA/semaphore, and MXU modules. Verify and
      serialize open-source IR, construct the exact custom call, execute on TPU, and inspect the strongest supported
      compiler evidence. Serialize/reload executables only when the plugin advertises it; otherwise prove deterministic
      recompilation from the serialized program and assert the exact unsupported result.
- [ ] Add normalized TPU capabilities and lower portable grids, scalar prefetch, HBM windows, VMEM/SMEM work, scratch,
      scalar/vector/tile operations, MXU matmul, transfers, DMA, semaphores, and supported double-buffered pipelines.
- [ ] Implement deterministic TPU target simulation for DMA engines, semaphores, and declared core participants;
      compare ordering, deadlock, race, precision, layout, and numerical behavior with supported TPU execution.
- [ ] Emit adapter-owned TPU compiler configuration; let `ryft-xla` build typed `stablehlo.custom_call` attributes
      for aliases and side effects. Keep memory-space, source, communication, collective, and compiler metadata in
      their owning fields. Support deferred compilation without forcing the payload into a CUDA artifact type.
- [ ] Extend the same portable-definition conformance suite with admitted TPU cases. Prove the GPU-only adapter and
      default core builds still exclude TPU dependencies and that no core backend enum or XLA reverse edge was added.
- [ ] Add deeper pipelines, accumulator/reference semantics, semaphore topologies, indirect DMA, asynchronous
      completion, multi-TensorCore partitioning, supported collectives, remote transfers, and launch ordering.
- [ ] Add SparseCore or hardware PRNG only as separately capability-gated target extensions after proving their
      execution, resource, transform, and simulation models.
- [ ] Integrate TPU batching, custom AD policy, partial evaluation, rematerialization, control flow, sharding,
      profiling, tuning, persistence, AOT, multi-core distribution, and cross-host failure/publication semantics using
      the already-stabilized shared contracts. Reject every unsupported route before submission.
- [ ] Publish TPU-specific examples, support/compatibility matrices, upgrade tooling, compiler-only CI, and hardware
      qualification across supported generations. Run concurrency, leak, failure, and performance qualification, then
      require independent correctness, conventions, security, and simplicity audits with zero findings.

**Tests/docs:** pinned sys/MLIR parity; per-concrete-item wrapper tests; compiler-only serde/config/analysis snapshots;
real vector, DMA/semaphore, MXU, pipeline, multi-core, collective, transform, persistence, and failure tests on
supported TPUs; exact unsupported-plugin/serialization/generation/layout/resource diagnostics; tutorials; support
matrix; and a complete verification record. All pre-Phase-20 test commands must remain runnable without TPU hardware.

**Excludes:** private libtpu compiler bindings, TPU requirements in earlier phases, duplicating portable semantics,
implicit distributed state, unsupported replica lowering, and claiming inspection data the plugin does not expose.

**Estimate:** production 5,300-9,200; tests 5,200-8,800; docs 1,750-3,000. This consolidates the former sys/MLIR,
baseline, advanced, transform, profiling, distributed, and stabilization TPU estimates into one end phase.

**Exit criterion:** the complete admitted Mosaic TPU stack—from pinned native/typed foundations through execution,
transforms, persistence, distribution, documentation, and hardware qualification—is production-ready and independently
clean. Until then, the completed non-TPU language, Mosaic GPU, and cuTile work remains buildable, testable, and usable
without any TPU dependency.

### Phase 22: Add optional direct Triton and ROCm backend support

**Prerequisites:** Phase 20. This post-v1 phase may proceed independently of Phase 21 and must not change the portable
semantics stabilized by the first production release.

**Owners:** new `ryft_triton::kernels` lowering, compiler invocation, typed options, and artifact mapping; existing
Triton compiler extension and selected ROCm execution seam in `ryft-pjrt`; `ryft-cuda` NVIDIA artifacts/launching;
`ryft-xla` selection/embedding/execution; only proven native or typed-IR gaps in `ryft-xla-sys` and `ryft-mlir`;
Linux CUDA and ROCm CI/tooling.

- [ ] Freeze compatible Triton, TTIR, PJRT Triton extension, ROCm plugin, HIP, HSACO, driver, operating-system, and
      hardware revisions. Add direct source contracts for every native symbol, artifact field, supported NVIDIA
      compute capability, supported AMD `gfx` architecture, and known restriction.
- [ ] Create `ryft-triton` following §3.2. Use the existing PJRT Triton compiler extension directly without depending
      on `ryft-xla`; do not copy its C API or move compiler ownership into `ryft-cuda` or `ryft-pjrt`.
- [ ] Record explicitly that deprecated JAX Pallas-to-Triton lowering is unsupported. Do not import its Python
      lowering, compatibility surface, or runtime behavior into Ryft.
- [ ] Prove one structured TTIR module through the existing PJRT Triton extension to executable PTX on NVIDIA and
      HSACO on AMD. Inspect entry symbols, layouts, resource metadata, diagnostics, and target code before designing
      the production lowering.
- [ ] Select exactly one ROCm execution route for the resulting HSACO: an official XLA/PJRT custom-call route when the
      pinned plugin exposes one, or a hardened HIP artifact launcher otherwise. Do not maintain both routes.
- [ ] Keep CUDA and ROCm low-level launchers concrete and separate. Reuse the four ownership contracts in §6.7,
      composed capability/cache identity, and core alias/effect diagnostics without a universal artifact schema or
      vendor-neutral launcher layer.
- [ ] Define the portable subset admitted by direct Triton, including scalar/tile operations, layouts, reductions,
      masked memory, atomics, barriers, dot and block-scaled operations, and bounded grids. Reject unsupported
      synchronization, memory-space, alias, resource, and target-operation contracts before TTIR construction.
- [ ] Implement structured lowering to the pinned TTIR surface. Prefer typed native construction where supported;
      otherwise own a minimal typed Ryft schema and deterministic TTIR serializer for the admitted subset. Do not
      expose arbitrary TTIR text as a verified kernel body.
- [ ] Validate artifact entry points, argument ordering, buffer layouts, dynamic extents, aliases, side effects,
      scratch, launch geometry, architecture, PTX/HSACO format, and required resources before cache insertion or
      execution.
- [ ] Include Triton, TTIR schema, PJRT extension, CUDA or ROCm/HIP, architecture, compiler options, launch ABI, and
      artifact format versions in capabilities, cache identity, persistence manifests, and AOT compatibility checks.
- [ ] Integrate batching, custom AD policy, specialization, rematerialization, profiling, tuning, persistence,
      sharding, asynchronous completion, and distributed execution only where the portable contract already defines
      them. Reject every unsupported path before compilation.
- [ ] Add CUDA and ROCm compiler-only CI, supported NVIDIA and AMD hardware qualification, clean-room AOT deployment,
      concurrency and leak tests, malformed TTIR/artifact tests, and target-code inspection for PTX and HSACO.
- [ ] Run the unchanged portable-definition suite through direct Triton on each supported platform. Prove NVIDIA
      artifacts reuse the one CUDA launcher, adapter-only tests exclude `ryft-xla`, and core gains no Triton/AMD enum.
- [ ] Publish exact support matrices, installation requirements, backend-selection behavior, cache compatibility,
      failure diagnostics, and examples that compare portable semantics across the interpreter and supported backends.

**Tests/docs:** typed TTIR construction and snapshot tests; invalid portable-to-Triton lowering cases; existing PJRT
extension compile/error coverage; PTX and HSACO metadata inspection; CUDA and ROCm compile/launch/AOT reload tests;
cross-backend numerical, alias, effect, synchronization, transform, cache, failure, and performance qualification;
version and hardware matrices; direct-backend guide; and a complete verification record.

**Excludes:** the deprecated JAX Pallas Triton backend, a JAX or Python runtime dependency, arbitrary TTIR ingestion,
the full Triton language surface, treating TTIR as stable across unpinned revisions, Mosaic-specific target operations,
maintaining two ROCm launch paths, and a speculative common CUDA/ROCm launcher hierarchy.

**Estimate:** production 3,200-5,400; tests 3,000-5,000; docs 700-1,200. Re-estimate after the pinned compiler and
ROCm execution probes select exact native, serialization, and launch seams.

**Exit criterion:** supported portable kernels compile directly through Triton and execute through versioned NVIDIA
and ROCm artifact paths with deterministic caching, AOT reload, exact prelaunch diagnostics, target-code evidence, and
independent correctness, conventions, security, and simplicity audits reporting zero findings.

## 9. Likely change surface

### `ryft-xla-sys`

- Mosaic common/GPU/TPU C++ bridges and Rust FFI modules.
- Bazel dependencies, visibility patches, exported symbols, source archive manifests, and build feature gates.
- Compiler/pass/serde/artifact/version APIs proven by Phase 0 for GPU and Phase 21 for TPU.
- Possibly CUDA/XLA FFI stream or custom-kernel declarations required by the selected cubin launch seam.
- Only direct Triton or ROCm native symbols proven missing by Phase 22; no broad TTIR or HIP wrapper surface.

### `ryft-mlir`

- Mosaic GPU/TPU operations, attributes, types, passes, pipelines, and compiler facades.
- Standard dialect wrappers only where a concrete kernel lowering needs a missing operation or attribute.
- Direct source-parity contracts and source-owned typed wrappers/tests.
- A minimal typed TTIR construction surface only if the pinned native interface requires it.

### `ryft-pjrt`

- Existing program, execution, stream, FFI, GPU custom-call, metadata, topology, profiling, and distributed modules.
- Thin CUDA client and XLA FFI stream/buffer adapters over `ryft-cuda`; no duplicate CUDA driver or cache policy.
- No Mosaic-specific executable hierarchy and no cuTile-specific PJRT extension absent an upstream standard.
- The existing Triton extension plus exactly one selected ROCm artifact execution path; CUDA and ROCm launchers stay
  concrete until implementation evidence supports a smaller common contract.

### `ryft-cuda`

- Producer-neutral cubin/PTX artifacts, typed launch ABIs, borrowed CUDA streams/device pointers, dynamic driver
  loading, context/device-aware bounded module caching, deterministic cleanup, and driver diagnostics.
- No PJRT, XLA, CUDA toolkit, Mosaic, or cuTile dependency and no producer-specific artifact schema.

### `ryft-core`

- `ryft_core::kernels`: portable DSL and builders, typed kernel region/operations, grids, block mappings,
  references/scratch, effects, verifier, interpreter, logical call metadata, and extension/compiler contracts.
- Reuse existing array/reference/program/context/tracing/compilation/transform modules wherever they already own the
  concept. Backend-free staging, semantic serialization, interpretation, and logical specialization remain here.
- The one deliberate core-reference-vocabulary extension: `ReferenceAccessMode::AtomicAccumulate` and
  `reference_atomic_add_update` in `programs::references` (Phase 9, §5.6), rippling through exhaustive mode
  classifiers, summaries, and region policies that the exact-mode set audits at compile time.
- No CUDA/Triton/Mosaic/PJRT types, backend enum, physical artifact payload, native compiler dependency, or duplicate
  JIT/cache lifecycle. Compiler options and outputs remain adapter-owned associated types.

### `ryft-macros` and `ryft-macros-tests`

- `#[kernel]` frontend, consumed signature annotations, source-preserving syntax transformation, and generated typed
  calls into core tracing/capabilities; no dependency from the proc-macro crate back to core or compiler adapters.
- Macro expansion, compile-pass/fail, hygiene, diagnostics, and canonical-IR parity tests. Keep an isolated core-only
  consumer because the existing macro integration crate also depends on the broader `ryft` facade.

### `ryft-mosaic` (new in Phase 14; TPU added in Phase 21)

- `ryft_mosaic::kernels::gpu`: GPU lowering, exact extensions, schedules, legality, simulation, compiler payloads,
  source mapping, and compatibility identity using existing typed MLIR/native wrappers.
- `ryft_mosaic::kernels::tpu`: the equivalent TPU responsibilities, isolated behind an explicit Phase 21 feature.
- No dependency on `ryft-xla`, outer-program buffer allocation, custom-call site construction, or PJRT fence wrapper.
  Backend-specific payload encoding belongs here; XLA call-site construction belongs to the integration.

### `ryft-cutile` (new in Phase 16)

- Portable-subset checks, deterministic compiler input, optional isolated tool invocation, cuTile manifest decoding,
  typed compiler options, target/version validation, source diagnostics, and logical-to-physical argument mapping.
- Reuse `ryft-cuda` artifacts and launcher; keep producer-specific knowledge out of CUDA and PJRT wrappers.
- No `ryft-xla`/`ryft-pjrt` dependency, second kernel IR, second CUDA launcher, or deployment-time Python requirement.

### `ryft-triton` (new in Phase 22)

- Direct portable lowering, pinned typed TTIR, compiler invocation through the existing PJRT extension, target
  legality, typed configuration, diagnostics, compatibility metadata, and NVIDIA/AMD artifact mapping.
- Reuse canonical CUDA artifacts and the selected ROCm contract. Add native/typed wrappers only at their existing
  low-level owners and only where the pinned compiler contract proves a gap.
- No `ryft-xla` dependency, JAX Pallas compatibility, duplicate PJRT extension wrappers, or common GPU driver hierarchy.

### `ryft-xla`

- `ryft_xla::kernels`: experimental kernel-call integration, enabled-adapter selection, typed compiler bindings,
  StableHLO custom-call embedding, logical/physical ABI validation, and PJRT execution integration.
- Reuse existing compilation-domain lifecycle and persistence hooks. Own the outer executable/AOT envelope,
  profiling correlation, tuning integration, and execution capability composition.
- Optional dependencies on compiler adapters; any combined extension/dispatch enum exists only at this integration
  boundary. Compiler lowering, target extension semantics, and compiler payload schemas stay in adapter crates.
- Promote/remove the Phase 6 experimental kernel validator when the production core boundary supersedes it; keep no
  compatibility shim or duplicate semantic validation stack.

### `ryft`

- Stable portable re-exports and examples only in Phase 20; TPU-specific facade additions remain in Phase 21, and
  optional direct Triton/ROCm backend selection and examples remain in Phase 22.

## 10. Test matrix

Use named families rather than a Cartesian product. Add the following mandatory architecture gates:

- **Core alone (Phases 7-12):** build, stage, verify, serialize/reload, and interpret representative portable kernels
  in a consumer depending only on `ryft-core`. Assert the resolved dependency graph excludes every native compiler,
  adapter, GPU runtime, and PJRT crate; avoid workspace feature unification masking accidental dependencies.
- **Macro/tracing parity (Phases 10-11):** compare normalized IR and effects with explicit builder equivalents,
  ignoring source-span differences; test loop-carried values, staged branches, no implicit host unrolling, full-array
  shape validation, boundary masks, zero dimensions, reference escape, poisoned traces, and source diagnostics.
  Run macro unit and integration suites plus the isolated core-only consumer; no new tracer or AST executor is used.
- **Adapter alone (Phases 14/16/22):** compile or construct/validate the supported deferred output in a consumer of
  core plus one adapter, with no `ryft-xla`. Install only the native libraries/plugins required by that adapter's
  documented seam. Test missing optional compiler tools and malformed physical argument mappings.
- **Same definition, two compilers (Phase 16):** reuse identical verified portable body and logical specialization
  for Mosaic GPU and cuTile; compiler options may differ but semantic fingerprints, access/alias/effect contracts,
  oracle results, and supported numerical policies must agree. Include boundary tiles and supported reductions/dot.
  Core interpretation runs without adapters; target-code evidence remains specific to each backend.
- **Exact extension rejection (Phases 14-16):** an explicit Mosaic operation is rejected by cuTile before compiler
  invocation; a missing extension decoder or simulation produces an exact error, never implicit emulation.
- **Feature/dependency composition (Phase 20):** core-only, launcher-only, each compiler, and combined GPU compiler
  builds remain acyclic; disabling cuTile removes its tool requirement; GPU builds exclude TPU. Phases 21/22 extend
  these checks and the unchanged portable-definition suite for their admitted subsets without altering core syntax.

No hand-authored native probe or mock adapter satisfies the two-compiler DSL gate. Retain those tests for their
lower-layer contracts and add production authoring-to-adapter tests separately.

| Area | Positive cases | Negative/safety cases |
|---|---|---|
| Native ABI | symbols, sizes, ownership, diagnostics | null, bad version, double release, missing library |
| Mosaic MLIR | every admitted concrete attr/type/op; deterministic passes | malformed arity/type/region/layout |
| PJRT | compile/load/execute/fence/AOT/capability | plugin unsupported, cancellation, timeout, stale metadata |
| Grids/mappings | arbitrary rank, singleton/empty, static/bounded, scalar prefetch | flatten overflow, data map, OOB |
| References/scratch | roots, views, access modes, initialized scratch | escape, uninitialized read, alias conflict |
| Atomics/sync | operations/scopes; async/barrier pipelines | bad dtype/order, missing wait, deadlock, race |
| Interpreter | every portable primitive; deterministic traces | bounds/init/race diagnostics and shrinking |
| Portable math | scalar/tile/reduction/dot/block-scaled dot | precision/rounding/shape/scale mismatch |
| Mosaic GPU | baseline, WGMMA/TMA, TMEM/tcgen05/NVFP4 | architecture/resource/order/toolchain mismatch |
| Mosaic TPU | VMEM/SMEM, DMA/semaphore, MXU, multi-core | layout/memory/generation/communication mismatch |
| cuTile | supported subset, cubin AOT, stream launch | unsupported sync/alias, tool/version/crash/timeout |
| Direct Triton | typed TTIR, PTX/HSACO compile, portable subset | unsupported op/layout/resource, TTIR/ABI drift |
| ROCm | AMD architecture, HSACO launch/AOT, stream/fence | plugin/HIP mismatch, bad artifact/resource/stream |
| ABI/aliases | dynamic extents, static args, scratch, reuse | entry-alias confusion, alignment, ABI drift |
| Transforms | batching, custom AD, specialization, remat | write conflict, implicit AD, duplicated effect |
| Sharding | per-shard local launch, supported collectives | automatic unsupported partition, overlapping writes |
| Async/distributed | completion chains, transfers, launch IDs | partial failure, stale state, wrong artifact |
| Cache/AOT | cold/warm, serialize/reload, replacement | compiler/arch/feature/schema mismatch, corruption |
| Diagnostics | source spans, backend stage, capability details | compiler crash, malformed artifact, missing tool |
| Performance | stable representative baselines | spills, wrong instruction path, excessive compile time |

Every representative portable kernel should satisfy:

```text
immutable array oracle
    ~= kernel interpreter
    ~= Mosaic GPU result, when supported
    ~= Mosaic TPU result, when supported
    ~= cuTile result, when supported
    ~= direct Triton result on NVIDIA or AMD, when supported
```

Here `~=` means bit-exact equality for operations whose contract requires it, and the operation's declared tolerance,
rounding, saturation, signed-zero, and NaN-equivalence policy otherwise. Target-instruction evidence is a separate
requirement and cannot be inferred from semantic equivalence.

Backend tests additionally inspect target artifacts. For example, Blackwell qualification must show the expected
TMEM/`tcgen05` path; matching FP32 output after widening is not sufficient.

## 11. Verification tiers

### Tier A: hermetic and compiler-only presubmit

- `cargo fmt --all -- --check`
- `cargo check`/`cargo test` for every changed crate and all targets.
- Native ABI/symbol/layout and archive-content tests.
- MLIR parse/print/verifier/pass snapshots.
- Kernel interpreter, property, transform, cache, malformed artifact, and diagnostics tests.
- Isolated core/launcher/adapter dependency checks and typed extension/argument-map negative tests from §10.
- Macro unit and integration suites, compile-fail diagnostics, and macro-versus-builder IR/effect parity.
- GPU/PTX compiler tests that do not require hardware. Phase 22 adds TTIR and HSACO compiler-only coverage; TPU
  compiler-only coverage begins in Phase 21.
- Rustdoc with warnings reviewed, doctests, examples that have a CPU/interpreter route.
- `git diff --check`, 120-column added-line audit, direct pinned-source parity, and no stale scaffolding.

### Tier B: accelerator presubmit smoke

- One baseline Mosaic GPU kernel and one advanced feature appropriate to the available GPU.
- One cuTile AOT/launch kernel on a supported NVIDIA runner, sharing its portable definition with the Mosaic case.
- AOT reload, asynchronous completion, exact capability mismatch, and target artifact inspection.

### Tier C: scheduled hardware qualification

- Hopper and Blackwell matrices, including WGMMA/TMA and NVFP4/TMEM/`tcgen05`.
- Concurrency, stress, sanitizers where available, leak detection, distributed failures, and performance variance.

### Tier D: non-TPU release qualification

- Full crate/facade/docs/examples matrix.
- Exact supported platform/toolchain versions and AOT compatibility.
- Reproducible clean-room artifact build and deployment without Python.
- Upgrade source-contract diff, security/licensing review for optional cuTile tooling, and independent zero-findings
  audits.

### Tier E: dedicated TPU qualification

- Run only in Phase 21 after the non-TPU release matrix is already clean.
- Cover compiler-only TPU parity/configuration tests without hardware, then supported TPU generations with
  VMEM/SMEM, DMA/semaphores, MXU, pipelines, transforms, persistence, multi-core, and distribution.
- Repeat the full affected crate/facade/docs matrix with TPU features enabled and record exact libtpu/plugin versions.

### Tier F: optional direct Triton and ROCm qualification

- Run only in Phase 22 after the non-TPU release matrix is already clean.
- Cover typed TTIR and PTX/HSACO compilation without hardware, then supported NVIDIA and AMD generations with
  execution, transforms, cache/AOT reload, target-code inspection, concurrency, sanitizers where available, leaks,
  failure injection, distribution, and performance qualification.
- Repeat the full affected crate/facade/docs matrix with direct Triton and ROCm features enabled and record exact
  Triton, TTIR, PJRT plugin, CUDA, ROCm, HIP, driver, and architecture versions.

All expensive local commands should use the repository's 300-second default timeout unless a specific hardware test has
a reviewed longer bound.

## 12. Delivery milestones and dependency graph

### Milestone A: lower-layer compiler/runtime foundation

Phases 0-5. This is a hard non-TPU gate, not preparatory work that can be papered over later.

- `ryft-xla-sys`: missing Mosaic registration/serde/runtime ABI and pinned archive parity.
- `ryft-mlir`: complete typed Mosaic GPU and required standard compiler-dialect construction and pipelines.
- `ryft-cuda`: producer-neutral CUDA artifacts, driver loading, launches, and cache/lifecycle policy.
- `ryft-pjrt`: only demonstrated generic capability/metadata/runtime gaps.
- GPU and cuTile hand-authored vertical slices.

### Milestone B: portable language and semantic oracle

Phases 6-12, beginning only after Milestone A passes: restored reference analysis and `ryft_core::kernels` types,
grids, references, memory/synchronization safety, macro DSL over existing tracing, explicit builders, interpreter,
operations, manual schedules, and minimal adapter contract. Prove portable authoring in a core-only consumer without
native compiler dependencies.

### Milestone C: shared XLA runtime and Mosaic GPU baseline

Phases 13-14: typed compiler-output/embedding contracts, existing dispatch/cache integration, and production baseline
Mosaic GPU in `ryft-mosaic`. Prove adapter compilation without `ryft-xla` and execution through its XLA integration.

### Milestone D: latest hardware and optional backend breadth

Phases 15-16: Hopper/Blackwell in `ryft-mosaic` and cuTile in `ryft-cutile`, sharing the established CUDA launcher
where the artifact route applies. Pass the unchanged portable-definition gate across both compiler adapters.

### Milestone E: composition and production operations

Phases 17-20: transforms, profiling/autotuning/AOT, GPU distributed/asynchronous execution, stabilization, and the
non-TPU release.

### Milestone F: consolidated TPU support

Phase 21 alone owns the `ryft-mosaic` TPU adapter and every TPU-specific native, typed-IR, runtime, transform,
distribution, documentation, and qualification task. It begins only after Milestones A-E are independently complete.

### Milestone G: optional direct Triton and ROCm expansion

Phase 22 owns `ryft-triton`, the pinned TTIR contract, direct portable lowering, selected ROCm execution seam, CUDA/ROCm
qualification, documentation, and AOT compatibility. It begins only after Milestone E and may proceed independently
of Milestone F.

Safe parallelism after Phase 5:

- Phase 6 starts only after the Phase 5 lower-layer gate passes.
- Phases 7-9 may divide by semantic owner but must merge before staging/interpreter work completes.
- cuTile lowering may proceed beside Mosaic GPU after Phase 13; its Phase 16 completion additionally requires
  Phase 14 and the two-compiler portable-definition gate.
- Transform and profiling work should consume at least two working backends to avoid single-backend abstractions.
- TPU work never proceeds in parallel with Phases 0-20; that isolation is the purpose of dedicated Phase 21.
- Phase 22 may proceed beside Phase 21 after Phase 20, but neither milestone may change stabilized portable semantics
  or silently inherit the other's target-specific contracts.

## 13. Aggregate estimates

Summing the phase ranges in §7 gives approximately, including the Phase 10 macro frontend increment:

- **Production:** 44,600-74,050 logical lines.
- **Tests:** 42,500-69,850 logical lines.
- **Docs/examples:** 12,130-20,700 logical lines.
- **Total:** 99,230-164,600 logical lines, excluding generated/vendored code.

Expected production ownership:

| Crate/area | Indicative midpoint share | Main work |
|---|---:|---|
| `ryft-xla-sys` | 4% | missing registration/serde/runtime bridges and builds |
| `ryft-mlir` | 25% | typed compiler dialects, Mosaic parity, verification, and pipelines |
| `ryft-cuda` | 3% | producer-neutral CUDA artifact and launch runtime |
| `ryft-pjrt` | 5% | demonstrated generic gaps, adapters, Triton extension, selected ROCm execution path |
| `ryft-core` / `ryft-macros` | 22% | macro DSL/tracing, IR, verifier, interpreter, transforms, adapter contracts |
| `ryft-mosaic` | 20% | GPU/TPU compiler adapters, exact extensions, simulation, target policy |
| `ryft-cutile` | 5% | cuTile compiler adapter, tool isolation, manifests and physical mapping |
| `ryft-triton` | 6% | direct Triton compiler adapter, target policy and artifact mapping |
| `ryft-xla` | 8% | selection, embedding, existing cache integration, execution, tuning |
| facade/examples/tooling | 2% | stable exports, examples, manifests, qualification tools |

These midpoint shares are indicative ownership allocations, not independently measured crate forecasts. The former
39% XLA compiler/integration allocation is redistributed across the three compiler adapters and the narrower XLA
integration. The portable-core allocation includes the macro frontend; the Phase 10 increment is included in the
aggregate ranges above. These are planning ranges, not commitments. Phase 0 replaces
first-release non-TPU estimates using its pinned contract, Phase 21 does the same for TPU, and Phase 22 does the same
for direct Triton and ROCm before implementation.
Each phase records actual logical lines and explains a variance above 30% before the next phase begins.

## 14. Risks and mitigations

### Reimplementing an existing lower layer

**Risk:** new dialect builders, executable types, or compiler wrappers duplicate mature code.

**Mitigation:** machine-checkable pinned source contract, extend existing facades, and require a concrete missing
symbol or contract for every lower-layer addition.

### Designing the language around a Python implementation detail

**Risk:** cuTile or JAX Python becomes a runtime dependency or defines Ryft semantics accidentally.

**Mitigation:** backend-neutral semantics, official AOT boundary, isolated optional compiler tool, Python-free runtime,
and exact version/artifact manifests.

### Lowest-common-denominator portability

**Risk:** the common schedule hides hardware features or target enums leak into core.

**Mitigation:** portable semantics plus additive target extensions; erased schedule preserves results; explicit target
operations fail rather than emulate.

### Mistaking data type support for tensor-core support

**Risk:** an NVFP4 kernel widens or uses scalar code while tests only inspect outputs.

**Mitigation:** compound block-scaling contract, architecture capabilities, target IR/PTX inspection, and hardware
performance/codegen qualification.

### Unsound memory or synchronization semantics

**Risk:** masks, scratch, views, async copies, atomics, or barriers admit OOB access, races, or deadlocks.

**Mitigation:** canonical reference analysis, definite initialization, affine overlap proof, linear tokens,
deterministic interpreter, litmus/property tests, and early rejection outside the proof subset.

### Upstream churn

**Risk:** Mosaic, cuTile, Triton, TTIR, or ROCm changes break wrappers and cached artifacts silently.

**Mitigation:** paired source pins, direct source parity, ABI/schema versions, source-owned C boundary, exact cache
identity, and upgrade CI that requires decisions for every surface change.

### Building on deprecated Pallas Triton compatibility or unstable TTIR

**Risk:** a backend accidentally depends on JAX's deprecated Pallas-to-Triton path or treats version-specific TTIR as
a stable public interchange format.

**Mitigation:** lower directly from verified Ryft IR, pin the native Triton and TTIR contracts, keep the admitted
schema small and typed, snapshot target IR, version every artifact, and reject incompatible revisions before cache
lookup or compilation.

### Premature universal backend abstraction

**Risk:** a large trait hierarchy encodes guesses and makes target-specific work harder.

**Mitigation:** specify the small compiler capability from core semantics and native probes, then validate it with
real `ryft-mosaic` and `ryft-cutile` lowerers before stabilization. Keep adapter options/outputs concrete through
associated types and CUDA/ROCm launchers separate. Reuse existing compilation-domain/cache/fence contracts. TPU may
return deferred compiler input; do not force every compiler through an eagerly compiled GPU binary contract.

### Crate separation without dependency or semantic independence

**Risk:** moved modules still import XLA artifact schemas, embed backend options in core definitions, or create a
PJRT-to-CUDA-to-PJRT cycle. Parallel schemas or a core backend enum make future adapters require central rewrites.

**Mitigation:** enforce §3.2 dependency rules and §6.7 ownership in isolated consumer builds. Keep compiler families in
adapter crates, use associated concrete outputs, and compose them only in execution integrations. Require the same
portable definition to compile through Mosaic and cuTile; a directory move or hand-authored native probe is not proof.

### Autotuning before reproducibility

**Risk:** noisy measurements and stale device facts produce unstable behavior.

**Mitigation:** manual correct schedules first, versioned fingerprints, bounded deterministic search, explicit budgets,
and performance gates only after variance is controlled.

### Artifact security and compiler isolation

**Risk:** untrusted compiler tools, cubins, or cached metadata compromise builds or execution.

**Mitigation:** optional sandboxed compiler subprocess, time/memory limits, content-hashed bundles, strict schema
validation, trusted-source policy, and no implicit execution of foreign artifacts. Artifact signing is deferred unless
an untrusted distribution model defines trust roots, key rotation, and verification policy.

### Scope explosion

**Risk:** distributed execution, internal AD, new vendors, and auto-scheduling prevent a shippable baseline.

**Mitigation:** milestone gates, named non-goals, Mosaic GPU/cuTile production before TPU work, and Phase 22's explicit
post-v1 boundary for direct Triton and ROCm without weakening already shipped semantics.

## 15. Review checkpoints

Checkpoints 1-12 are sequential. After checkpoint 12 passes, checkpoints 13 and 14 are independent branches and may
proceed in parallel:

1. **Pinned-source review:** exact non-TPU lower-layer contract and official cuTile seam.
2. **Native ABI review:** ownership, errors, symbols, artifacts, and feature gates.
3. **Typed MLIR review:** parity, verification, pass pipelines, and no stringly production paths.
4. **Runtime review:** standard PJRT/XLA lifecycle, capabilities, streams, fences, and AOT.
5. **Lower-layer gate review:** real GPU/cuTile vertical slices and target-code evidence.
6. **Semantic review:** grids, block mappings, references, scratch, effects, atomics, synchronization, and races.
7. **Interpreter review:** deterministic oracle and diagnostic completeness.
8. **Adapter/ABI review:** acyclic crate graph, core logical contract, typed compiler outputs, platform ABI, XLA
   embedding, selection, aliases, existing cache integration, persistence, and failures.
9. **Mosaic GPU review:** baseline, Hopper, Blackwell, NVFP4, and exact instruction use.
10. **cuTile/portability review:** supported subset, tool isolation, AOT ABI, Python-free deployment, and the same
    core-authored kernel definitions compiled through independent Mosaic GPU and cuTile adapters.
11. **Transform/distribution review:** every non-TPU route proven or rejected; no duplicated effects.
12. **Non-TPU production review:** docs, examples, CI, security, upgrades, performance, and no temporary scaffolding.
13. **Mosaic TPU review:** native/typed foundations, memory, DMA, semaphores, MXU, transforms, pipelines,
    multi-core, distribution, documentation, and zero findings.
14. **Direct Triton/ROCm review:** pinned TTIR contract, portable subset, selected HSACO launch path, CUDA/AMD target
    evidence, cache/AOT compatibility, documentation, and zero findings.

At every checkpoint ask:

- Is this one source of truth, or did the phase introduce parallel metadata or runtime types?
- Is the behavior semantic, portable optimization, or exact target behavior—and is that distinction testable?
- Can unsupported input fail earlier with a more precise owner?
- Does the cache key include every fact that can change legality or generated code?
- Would a future architecture add a capability/target extension rather than force a core semantic rewrite?
- Does the test prove the intended engine/instruction was used, not merely that a value was computed?
- Can any abstraction, compatibility layer, or generated wrapper be deleted or narrowed?

## 16. Plan completion criteria

This plan is complete only when:

- all selected phases and their exit criteria are checked;
- the GPU/cuTile lower-layer gate preceded all core/XLA production implementation;
- `ryft_core::kernels` owns portable authoring, semantic IR, verification, interpretation, and logical contracts;
- core-only consumers stage, serialize/reload, and interpret without compiler/backend/runtime dependencies;
- the `Array`-based macro DSL uses existing tracing and canonical reference/operation semantics; generated control
  flow, diagnostics, and macro-versus-builder equivalence pass the Phase 10/11 gates, and macro-authored portable
  kernels pass the same two-adapter gate;
- adapter crates own lowering, typed options, exact extensions, target simulation, and compiler payloads, with no
  dependency on `ryft-xla`; adding an adapter requires no central core backend enum;
- the same portable definitions pass the production Mosaic GPU/cuTile gate with unchanged semantic identity;
- compiler output, platform launch ABI, and XLA embedding retain separate canonical owners; CUDA launchers and
  existing core compilation/cache/PJRT fence lifecycles are reused rather than duplicated;
- portable semantics have an interpreter and immutable oracle;
- Mosaic GPU and cuTile execute before Phase 21 without TPU dependencies, and the complete Mosaic TPU stack executes on
  supported TPU hardware only in Phase 21;
- Hopper/Blackwell and NVFP4 tests prove exact target instructions and ordering;
- cuTile support uses an official versioned AOT seam with no Python runtime dependency, while installation remains
  optional;
- when Phase 22 is selected, direct Triton uses a pinned typed TTIR contract rather than JAX Pallas compatibility,
  and supported AMD GPUs execute through exactly one qualified ROCm artifact path;
- every transform, dynamic/distributed class, backend, and hardware feature is documented as supported or rejected;
- AOT/cache compatibility and failure behavior are deterministic;
- public docs and examples explain semantics without relying on physical in-place reuse or implementation knowledge;
- no phase-owned TODO, mock boundary, deprecated alias, compatibility shim, parallel universe, or stale identifier
  remains;
- exact full verification counts and hardware/toolchain versions are recorded;
- independent correctness, convention, security, and simplicity auditors report zero findings.

## 17. Primary external references

Revalidate the GPU, cuTile, XLA, and PJRT sources in Phase 0. Revalidate the two Mosaic TPU sources only when Phase 21
begins, and revalidate the direct Triton and ROCm sources only when Phase 22 begins:

- [JAX Pallas overview](https://docs.jax.dev/en/latest/pallas/): language status and guide index.
- [Pallas quickstart and programming model](https://docs.jax.dev/en/latest/pallas/quickstart.html): grids,
  `BlockSpec`s, program IDs, and current GPU/TPU backend mapping.
- [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html): references, loads/stores, backend
  lowering, and transform model.
- [Pallas changelog](https://docs.jax.dev/en/latest/pallas/CHANGELOG.html): backend lifecycle, including the
  deprecation status of the JAX Pallas Triton backend.
- [Mosaic GPU reference](https://docs.jax.dev/en/latest/pallas/gpu/reference.html): GPU execution, WGMMA/TMEM,
  Blackwell `tcgen05`, barriers, and explicit control.
- [Mosaic GPU pipelining](https://docs.jax.dev/en/latest/pallas/gpu/pipelining.html): explicit pipeline model.
- [Mosaic TPU quickstart](https://docs.jax.dev/en/latest/pallas/tpu/quickstart.html): HBM/VMEM and pipeline entry.
- [Mosaic TPU details](https://docs.jax.dev/en/latest/pallas/tpu/details.html): supported operations, memory, shapes,
  layouts, and precision.
- [NVIDIA NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html):
  NVFP4 E2M1 values, per-16 E4M3 block scales, and the tensor-level FP32 scale.
- [OpenXLA custom calls and XLA FFI](https://openxla.org/xla/custom_call): typed custom-call signatures,
  registration, attributes, buffers, and execution contexts such as the GPU stream.
- [OpenXLA PJRT C API](https://github.com/openxla/xla/blob/main/xla/pjrt/c/README.md): plugin model, ABI
  versioning, integration resources, and the canonical C API entry points.
- [JAX installation](https://github.com/jax-ml/jax/blob/main/docs/installation.md): current ROCm plugin installation,
  supported platforms, and version constraints that Phase 22 must revalidate.
- [Triton project support matrix](https://github.com/triton-lang/triton/blob/main/README.md): current NVIDIA and AMD
  hardware, operating-system, and ROCm requirements that Phase 22 must pin rather than assume.
- [cuTile execution model](https://docs.nvidia.com/cuda/cutile-python/execution.html): block/tile model, execution
  spaces, synchronization restrictions, launch, and compiler hints.
- [cuTile compilation and export](https://docs.nvidia.com/cuda/cutile-python/compilation.html): cubin/TileIR export,
  signatures, constraints, and `cutile_python_v2` calling convention.
- [cuTile interoperability](https://docs.nvidia.com/cuda/cutile-python/interoperability.html): official JAX FFI
  integration and the boundary between compiled cuTile kernels and array frameworks.
- [`cuda.tile.jax.cutile_call`](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.jax.cutile_call.html):
  documented array, input/output, output-placeholder, scalar, and static-argument conventions.
- [cuTile quickstart](https://docs.nvidia.com/cuda/cutile-python/quickstart.html): current platform, driver, CUDA
  Toolkit, Python, and compute-capability requirements that Phase 0 must revalidate.
- [cuTile data model](https://docs.nvidia.com/cuda/cutile-python/data.html): arrays, tiles, scalar and low-precision
  types including `float4_e2m1fn`.

## 18. Plan review record

- [x] Audited the completed reference plan and preserved-reference mock boundary.
- [x] Audited current Mosaic GPU/TPU, standard GPU/NVVM, PJRT FFI/stream/custom-call, and block-scaled-dot surfaces.
- [x] Checked current official Pallas, Mosaic GPU, Mosaic TPU, cuTile, Triton, and ROCm documentation.
- [x] Made completion of non-TPU `ryft-xla-sys`, `ryft-mlir`, `ryft-cuda`, and `ryft-pjrt` work a hard gate before
      core/XLA production work.
- [x] Distinguished existing lower-layer coverage from partial and missing compiler/runtime work.
- [x] Kept cuTile behind an official AOT seam rather than inventing an MLIR dialect or PJRT extension.
- [x] Added separate Mosaic GPU baseline, Hopper/Blackwell, and cuTile phases, followed by one consolidated dedicated
      TPU phase that does not block the preceding roadmap.
- [x] Modeled NVFP4 as values, scales, geometry, accumulation, layout, and exact instruction capabilities.
- [x] Added phase-specific production, test, and documentation ranges with explicit methodology.
- [x] Reconciled arithmetic totals, path/link checks, and independent foundation, architecture, and conventions audits;
      all three original-plan auditors reported zero findings.
- [x] Consolidated every TPU-specific implementation and verification deliverable into dedicated Phase 21,
      renumbered all prior phases, and revalidated 21 complete phase templates, dependencies, estimates, paths, line
      widths, and diff hygiene without changing a changelog.
- [x] Added restoration Phase 6 after the interpreter-style discharge rework deleted the static reference-analysis
      stack and the `reference_kernels.rs` mock this plan previously named as retained pieces; renumbered later
      phases, reconciled the estimate table, aggregate totals, milestones, and every cross-reference, and rewrote the
      retained-pieces inventory to record the deletion.
- [x] Made Phases 0-5 strictly precede every `ryft-core` and `ryft-xla` production change by removing the former
      Phase 6 parallel-start exception and clarifying the lower-layer scope of Phase 0.
- [x] Added optional post-v1 Phase 22 for direct Triton and ROCm support, explicitly excluding deprecated JAX Pallas
      compatibility, selecting one ROCm execution route, and reconciling dependencies, estimates, CI, risks, review
      gates, completion criteria, and references.
- [x] Revised the architecture around `ryft_core::kernels`, separate `ryft-mosaic`/`ryft-cutile`/`ryft-triton`
      compiler adapters, the retained `ryft-cuda` launcher, and narrower `ryft-xla` execution integration. Specified
      acyclic dependencies, portable/extension contracts, ready/deferred compiler outputs, ABI/AOT ownership, and
      core-only/adapter-only/same-definition qualification. Updated future phase owners, milestones, risk reviews,
      and estimate allocations without marking new implementation complete or changing completed lower-layer evidence.
- [x] Added the proposed `Array`-based macro matmul and explicit macro-to-existing-tracing design. Assigned frontend
      work to `ryft-macros`, retained canonical array/reference/program ownership, specified staged control flow and
      signature/boundary checks, and expanded Phase 10 scope, budgets, macro integration tests, and portability gates.
