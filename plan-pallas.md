# Ryft Pallas-Style Kernels: Architecture and Implementation Plan

Status: proposed. The reference architecture in [`plan-references.md`](plan-references.md) is complete through its
preserved-reference boundary. This plan begins the separate program required to turn that boundary into a production
kernel language. No phase in this document is implemented merely because a lower-level wrapper or mock validator
already exists.

This is a source-sensitive plan. Mosaic and cuTile are evolving, experimental systems, so Phase 0 refreshes the
non-TPU upstream inventory against the selected OpenXLA, JAX, CUDA, and cuTile revisions; Phase 21 refreshes the TPU
inventory immediately before TPU implementation. Links describe the design snapshot, not an upstream stability
promise.

## 1. Executive decisions

1. **Ryft owns a backend-neutral Pallas-style kernel language and semantic IR.** Mosaic GPU, Mosaic TPU, and cuTile
   are compiler backends. NVIDIA GPUs and Google TPUs are hardware targets. These layers must not be conflated.
2. **Finish the non-TPU lower layers first.** Phases 0-5 complete and prove the missing GPU/cuTile
   `ryft-xla-sys`, `ryft-mlir`, and `ryft-pjrt` foundations. No production change in `ryft-core` or `ryft-xla` may
   begin before the Phase 5 gate passes. Every TPU-specific inventory, wrapper, runtime, lowering, test, and
   qualification task is consolidated in final Phase 21 so Phases 0-20 never require libtpu or TPU hardware.
3. **Extend what exists.** `ryft-xla-sys` already links the JAX Mosaic dialects, `ryft-mlir` already wraps substantial
   Mosaic GPU and TPU surfaces, and `ryft-pjrt` already compiles, executes, serializes, and reloads generic MLIR
   programs. The missing work is primarily pinned-surface parity, pass/compiler bridges, target capabilities, artifact
   metadata, and end-to-end conformance—not a parallel dialect or runtime universe.
4. **Keep portable semantics and exact target control side by side.** Portable operations may select an equivalent
   implementation or a documented fallback. Explicit target operations such as Mosaic GPU `tcgen05_mma` must either
   lower exactly on a compatible target or fail before compilation; they must never silently emulate.
5. **Mosaic is the primary production route.** Mosaic GPU is the explicit NVIDIA path and Mosaic TPU is the TPU path.
   cuTile support is a required roadmap deliverable but remains an optional install/runtime component for the portable
   tile subset. It uses only an official, versioned compiler/export seam, is not the common IR, and does not replace
   Mosaic GPU's low-level control.
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
- Direct integration with Ryft staging, compilation, PJRT execution, caching, persistence, profiling, and diagnostics.
- A capability and cache model that admits new architectures without changing portable kernel semantics.
- Explicit transform behavior for batching, differentiation, partial evaluation, rematerialization, and sharding.
- Reproducible correctness and performance qualification across compiler-only, GPU, TPU, and release CI tiers.

### Non-goals for the first production release

- Reimplementing the Mosaic, OpenXLA, CUDA, TPU, or cuTile compilers.
- Mirroring JAX's Python syntax or promising source compatibility with JAX Pallas.
- A lowest-common-denominator schedule language that hides meaningful GPU/TPU differences.
- Triton as a primary backend. A later backend may reuse the portable contract, but it is outside this plan.
- AMD, Intel, Metal, or CPU-native kernel code generation. The interpreter remains the CPU correctness path.
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
- **Capability:** a compiler-and-device fact used for legality, selection, caching, and diagnostics—not a semantic
  fallback promise.

The intended dependency direction is:

```text
Ryft user function
    -> ryft-core kernel call, types, operations, verifier, interpreter
    -> verified shared kernel IR
    -> ryft-xla dispatch, specialization, cache, and backend selection
       -> Mosaic GPU MLIR -> Mosaic GPU compiler -> XLA/PJRT execution on NVIDIA GPU
       -> Mosaic TPU MLIR -> TPU custom-call payload -> XLA/PJRT execution on TPU
       -> cuTile source/TileIR -> versioned AOT cubin -> XLA FFI/custom-call launch on NVIDIA GPU
```

Target extensions branch after verification of the shared boundary. Mosaic operations do not enter the core array
operation family, and cuTile syntax does not become the common kernel IR.

The core IR is parameterized as `KernelOperation<Extension>`. Portable operations are concrete core variants.
`ryft-xla` owns the sealed, typed experimental extension enum: Phase 13 introduces its GPU-capable form and Phase 21
adds TPU variants. Every extension operation must provide its complete type, effect, resource, liveness, race,
transform, and capability contract to the core verifier. Opaque names, strings, or byte payloads are artifact metadata
only and can never enter a verified kernel body or bypass semantic analysis.

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
- [`crates/experimental/src/jax/gpu_runtime.rs`](crates/experimental/src/jax/gpu_runtime.rs) already prototypes dynamic
  CUDA driver loading, module/function lookup, argument packing, and launch on an XLA-provided stream. The production
  plan promotes and hardens one generic CUDA artifact launcher from that experiment; it must not build a second
  launcher beside it.

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

Phase 0 produces a machine-checkable non-TPU manifest that classifies every item as **existing**, **partial**,
**missing**, or **intentionally unsupported**. Phase 21 produces the separate TPU manifest. No phase may call its
owned surface complete without updating the relevant manifest.

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
- optional backend selection and target-specific compiler parameters;
- source locations and a stable semantic fingerprint.

The outer signature contains arrays, scalars, and static parameters. Inner array windows and scratch are references.
Read/write effects remain inside the kernel operation; updated outputs make mutation explicit at the outer SSA
boundary. Kernel-internal references cannot be returned, captured by ordinary regions, serialized as constants, or
stored in composite program values.

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
- sequentially consistent atomic read-modify-write operations over the portable scopes defined in §5.6;
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

- Ordered reference effects define compiler ordering but do not imply atomicity between programs.
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

1. Stage the outer call and kernel body.
2. Infer and validate types, block mappings, references, initialization, aliases, effects, synchronization, and races.
3. Canonicalize portable value work without erasing effect or liveness distinctions.
4. Select a backend from user policy and exact capabilities.
5. Apply portable schedule decisions and target-specific legalization.
6. Lower to Mosaic GPU MLIR, Mosaic TPU MLIR/custom-call configuration, or the supported cuTile subset.
7. Verify the target IR and compile it through the pinned native compiler/tool.
8. Embed or reference the artifact through an XLA custom call with exact layouts, aliases, side effects, and metadata.
9. Compile/load/execute through existing PJRT APIs and return the ordinary Ryft execution fence.
10. Persist the semantic IR, target contract, compiler options, artifact, and compatibility metadata atomically.

### 6.2 Capability descriptor

The canonical descriptor records:

- backend and backend ABI version;
- platform, device kind, architecture, compute capability, and feature extensions;
- compiler, OpenXLA, JAX Mosaic, PJRT plugin, CUDA/toolkit, TPU, and cuTile versions;
- address spaces, maximum grid/CTA/cluster shape, and scratch/resource limits;
- supported scalar, scale, accumulator, atomic, and matrix-operation combinations;
- layouts, swizzles, async copies, barriers, semaphores, tensor memory, and collective modes;
- dynamic-shape, multi-device, profiling, and AOT support;
- known restrictions that affect correctness, not merely performance.

Capability checks occur before artifact lookup and compilation. Diagnostics name the operation, requested contract,
backend, target, and missing feature.

### 6.3 Cache and AOT identity

The key includes:

- normalized semantic kernel IR and source schema version;
- input/output types, block mappings, static arguments, and specialization constraints;
- access, alias, scratch, synchronization, and dynamic-bound contracts;
- backend, architecture, exact required features, and device-independent vs device-specific status;
- schedule decisions and compiler options;
- OpenXLA, JAX Mosaic, PJRT ABI/plugin, CUDA/toolkit, TPU compiler, and cuTile versions;
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

## 7. Estimate methodology

Estimates count new or materially rewritten logical source lines:

- **Production:** Rust, C/C++, Bazel/build configuration, schemas, and non-test fixtures required at runtime/build time.
- **Tests:** unit/integration/property/golden/hardware tests and test-only infrastructure.
- **Docs:** rustdoc, examples, user guides, design records, and migration/support matrices.

Generated bindings, vendored upstream source, lockfiles, build outputs, and formatting-only changes are excluded.
Generated manifests are reported separately. Deletions do not reduce the estimate. Ranges are deliberately broad:
approximately ±30% for in-repository work and up to ±50% for upstream/compiler and cuTile seams until Phase 0 is
complete.

The table budgets the dynamic Rust CUDA launcher branch because it has the largest known Phase 4 implementation.
Phase 0 replaces the Phase 2/4 rows with the documented branch-specific estimates when it selects the official FFI,
built-in plugin, or dynamic launcher route; alternatives are never summed.

| Phase | Primary scope | Production | Tests | Docs |
|---:|---|---:|---:|---:|
| 0 | Pinned inventory and seam decisions | 150-300 | 250-450 | 300-500 |
| 1 | Common `ryft-xla-sys` registration/archive surface | 250-600 | 350-700 | 100-200 |
| 2 | `ryft-xla-sys` Mosaic GPU runtime/compiler bridge | 500-1,100 | 700-1,300 | 180-300 |
| 3 | `ryft-mlir` compiler dialects, binary IR, and Mosaic GPU | 9,000-15,000 | 4,500-8,000 | 1,500-3,000 |
| 4 | `ryft-pjrt` Mosaic/cuTile kernel prerequisites | 1,800-3,000 | 1,800-3,000 | 350-600 |
| 5 | GPU/cuTile lower-layer vertical-slice gate | 0-250 | 1,800-3,000 | 350-600 |
| 6 | Restore reference analysis and kernel validation | 2,400-3,400 | 3,000-4,500 | 450-700 |
| 7 | Core kernel types and semantic IR | 1,400-2,200 | 1,200-1,800 | 350-550 |
| 8 | Grids, block mappings, and bounds | 1,200-1,900 | 1,100-1,700 | 300-500 |
| 9 | References, scratch, atomics, and sync | 1,600-2,600 | 1,500-2,400 | 400-650 |
| 10 | Staging, builder, and kernel call | 1,600-2,500 | 1,300-2,100 | 400-650 |
| 11 | Interpreter, diagnostics, and debugging | 1,100-1,800 | 1,600-2,400 | 350-550 |
| 12 | Portable primitives and scheduling | 1,500-2,400 | 1,300-2,100 | 350-600 |
| 13 | XLA dispatch, ABI, cache, and runtime | 1,500-2,400 | 1,200-1,900 | 350-550 |
| 14 | Mosaic GPU baseline | 1,800-2,900 | 1,500-2,400 | 400-700 |
| 15 | Hopper/Blackwell Mosaic GPU | 2,300-3,800 | 2,000-3,300 | 500-850 |
| 16 | cuTile backend | 1,800-3,000 | 1,500-2,500 | 450-750 |
| 17 | Transforms and composition | 1,400-2,300 | 1,600-2,600 | 400-650 |
| 18 | Profiling, autotuning, persistence, AOT | 1,500-2,500 | 1,400-2,300 | 400-650 |
| 19 | Distributed and asynchronous kernels | 1,200-2,000 | 1,200-2,100 | 350-600 |
| 20 | Non-TPU stabilization and production hardening | 900-1,500 | 2,000-3,200 | 1,200-1,900 |
| 21 | Consolidated Mosaic TPU support and qualification | 5,300-9,200 | 5,200-8,800 | 1,750-3,000 |

The arithmetic totals and crate split are recorded in §13 after the detailed phases.

## 8. Detailed implementation phases

### Phase 0: Freeze pinned surfaces and prove integration seams

**Prerequisites:** `plan-references.md` complete.

**Owners:** `ryft-xla-sys`, `ryft-mlir`, `ryft-pjrt`, planning/tooling only. No `ryft-core` or `ryft-xla`
production edits.

- [ ] Diff the pinned JAX Mosaic GPU TableGen, C API, pass, compiler, serde, and custom-call sources against all local
      C++, Rust FFI, and typed MLIR wrappers. Defer the equivalent TPU inventory to final Phase 21.
- [ ] Inventory the standard Vector, Math, Complex, UB, Bufferization, Arith, MemRef, SCF, GPU, NVGPU, NVVM, LLVM,
      and conversion surfaces actually traversed by the pinned Mosaic pipelines; distinguish missing dialect coverage
      from operations already wrapped elsewhere.
- [ ] Generate a checked manifest of operations, attributes, types, interfaces, passes, translations, compiler
      options, artifact formats, and unsupported items.
- [ ] Pin the OpenXLA/JAX pairing and minimum CUDA, driver, and Python/tool requirements for each non-TPU tier.
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
- [ ] Decide the artifact schemas and error ownership before adding public wrappers.
- [ ] Record unsupported targets and the CI hardware matrix.

**Tests/docs:** manifest-diff test; symbol/link probe on every supported build platform; minimal compile probes; exact
decision record for the cuTile seam.

**Excludes:** language IR, public APIs, broad wrappers, Python runtime embedding, and all TPU inventory or prototyping.

**Estimate:** production 150-300; tests 250-450; docs 300-500.

**Exit criterion:** every non-TPU lower-layer deliverable maps to a pinned upstream symbol/source or an explicitly
owned Ryft schema, every required compiler dialect is classified, and the GPU and cuTile seams have executable
prototypes. No exit check loads libtpu or requires TPU access.

### Phase 1: Complete the common `ryft-xla-sys` registration and archive surface

**Prerequisites:** Phase 0.

**Owners:** `crates/ryft-xla-sys/src/c++`, `src/mlir`, `BUILD.bazel`, archive/export configuration.

- [ ] Reuse the existing upstream MLIR C APIs and `ryft-mlir` wrappers for pass managers, cloning, diagnostics, module
      ownership, parsing, and bytecode rather than adding a parallel native ABI.
- [ ] Add source-owned C functions only for pinned Mosaic registration, version, analysis, serialization, or compiler
      results that lack a usable upstream C API. Associate compiler-stage labels in the owning caller unless a native
      Mosaic result exposes them.
- [ ] Add explicit owned-byte/result destructors and null/error contracts for every genuinely new native result.
- [ ] Complete the required Vector/Math and exact Complex/UB/Bufferization dialect-handle, C bridge, TableGen archive,
      and registration surface identified by Phase 0; archive UB definitions when parity tooling requires them.
- [ ] Export all required headers, libraries, symbols, and platform link dependencies in source and release archives.
- [ ] Make feature gating exact: CPU builds expose parsing/verification where supported but fail target compilation
      deterministically; CUDA artifacts expose only linked capabilities.
- [ ] Generate ABI size/offset/version assertions and a symbol allowlist from the manifest.

**Tests/docs:** C and Rust ABI layout tests for new functions, null/error/ownership tests, archive-content tests,
symbol parity, and Linux/macOS/Windows link probes where supported.

**Excludes:** Mosaic-specific policy, TPU-specific bridges, and any kernel-language semantics.

**Estimate:** production 250-600; tests 350-700; docs 100-200.

**Exit criterion:** every lower-layer phase can access its pinned registrations, versions, analyses, serialization,
and required compiler-dialect symbols through one safe ownership model, with generic MLIR behavior still owned by the
existing MLIR C API and `ryft-mlir`.

### Phase 2: Complete the `ryft-xla-sys` Mosaic GPU runtime/compiler bridge

**Prerequisites:** Phase 1.

**Owners:** Mosaic GPU C++ bridge, Rust FFI, Bazel dependencies, CUDA release artifacts.

- [ ] Close the pinned GPU dialect attribute/type accessor gaps; do not create per-operation C builders when generic
      MLIR construction plus typed Rust verification is sufficient.
- [ ] Bind GPU serde/pass registration and any documented version or runtime-configuration functions exposed by the
      pinned source. Keep generic pass construction and diagnostics in `ryft-mlir`.
- [ ] Expose a standalone target/lowering compiler bridge only if the pinned upstream provides a supported callable
      interface. Otherwise expose and document only the serialized Mosaic-module contract consumed by
      `mosaic_gpu_v2`; Phase 14 owns StableHLO custom-call construction and backend-configuration policy.
- [ ] Return compiled object/PTX/cubin data, entry symbol, launch metadata, required shared/TMEM resources, and compiler
      diagnostics through an owned artifact where the pinned native interface exposes those stages. Otherwise return
      the verified serialized Mosaic module consumed by the linked runtime. Wrap a configuration producer only if
      Phase 0 identifies an actual supported upstream API.
- [ ] Link JAX's Mosaic GPU custom-call/runtime, pass, serde, and target dependencies into the CUDA PJRT plugin, retain
      static registration in release artifacts, and prove that `mosaic_gpu_v2` is present. Keep CPU, ROCm, and
      unsupported platforms dependency-clean.
- [ ] If Phase 0 selects a built-in plugin launcher, add the generic cubin/PTX handler here. If Phase 0 selects the
      dynamic Rust launcher, expose only the minimal stream/context contract it needs and leave promotion to Phase 4.
- [ ] Keep native/compiler target representations open-ended and ensure CUDA plugin builds do not cap the newest known
      SM version. PJRT exposes raw device architecture; Phase 13 normalizes Hopper/Blackwell capabilities.
- [ ] Preserve the JAX visibility patch only if the pinned source still requires it; delete it when upstream exports
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

### Phase 3: Complete binary MLIR, compiler dialects, and typed Mosaic GPU support in `ryft-mlir`

**Prerequisites:** Phase 0 and the required dialect/archive items from Phase 1. This work proceeds in parallel with
Phase 2's CUDA runtime linkage.

**Owners:** `crates/ryft-mlir/src/dialects`, module/operation parsing, pass/pipeline facades, tests.

- [ ] Add byte-slice module/operation parsing over `MlirStringRef`, preserving arbitrary MLIR bytecode and structured
      parse diagnostics while retaining text conveniences.
- [ ] Add source-owned typed Vector and Math dialect modules with operation-specific builders, traits, passes, exact
      renderings, and invalid verifier tests.
- [ ] Add the exact Complex, UB, Bufferization, and conversion operations/passes required by the pinned Mosaic
      pipelines. Audit existing Arith, MemRef, SCF, GPU, NVGPU, NVVM, and LLVM coverage and fill deltas only.
- [ ] Reconcile all local operations, types, and attributes with the Phase 0 manifest; add only missing pinned items.
- [ ] Replace permissive generic operand/result/attribute lists with typed constructors where the upstream operation
      has a stable contract, while retaining an explicit raw escape hatch for forward-compatible parsing only.
- [ ] Add typed pass registration, supported pipeline construction, target verification, and analysis wrappers.
      Preserve upstream textual pipeline syntax where MLIR exposes no typed pass constructor; validate and snapshot
      it rather than wrapping strings in a cosmetic second API.
- [ ] Verify region counts, successor counts, variadic segments, memory spaces, layouts, barriers, async tokens,
      WGMMA, TMEM, `tcgen05`, and scale operands before native verification.
- [ ] Add canonical parse/print/accessor support for every admitted type and attribute.
- [ ] Generate the parity inventory but keep reviewed public Rust APIs source-owned and documented.

**Tests/docs:** binary/text round trips; one focused construction/accessor/module-verification/complete-rendering test
for every new or changed concrete operation, attribute, and type in module order; invalid arity/type/attribute tests;
parser failures; pass-pipeline snapshots; target-gated verification; and pinned parity checks.

**Excludes:** kernel-language operations, automatic scheduling, and Mosaic TPU dialect completion.

**Estimate:** production 9,000-15,000; tests 4,500-8,000; docs 1,500-3,000. Most of the range is the missing typed
Vector/Math and compiler-dialect surface; generated manifests remain excluded.

**Exit criterion:** `ryft-mlir` can parse binary Mosaic IR and construct, inspect, verify, transform, and serialize
every standard/Mosaic GPU construct required by the planned backend through typed APIs wherever upstream exposes a
typed contract.

### Phase 4: Complete generic Mosaic and cuTile kernel prerequisites in `ryft-pjrt`

**Prerequisites:** Phase 0 and the exact stream/plugin/runtime seams selected from Phases 1-2. Raw PJRT work may
proceed in parallel with typed MLIR completion.

**Owners:** existing PJRT program, stream, FFI, GPU custom-call, executable-metadata, topology, and profiling modules.

- [ ] Add only raw plugin/device/topology/extension/version and metadata accessors proven missing by the vertical
      slices. Normalized Mosaic and kernel semantic capabilities belong to `ryft-xla` Phase 13.
- [ ] Wrap Pallas-relevant upstream extensions still missing at the selected pin, such as raw-buffer or phase-compile
      support, only when Phase 0 demonstrates that the chosen runtime path needs them.
- [ ] Extend generic `Program` formats only when Mosaic ingestion requires a format not representable as current MLIR
      or HLO; do not add `Program::Mosaic` as a label over ordinary MLIR.
- [ ] Prove XLA FFI/custom-call access to the backend stream and device buffers needed by a precompiled cubin.
- [ ] Promote and harden the existing experimental CUDA artifact launcher if selected in Phase 0: add module unload
      and lifetime management, context/device-keyed caches, arbitrary symbols and cubin/PTX bytes, complete CUDA error
      names/strings, typed argument descriptors, concurrency, and exact stream ordering.
- [ ] If a generic launcher needs an owned descriptor, define only a backend-neutral `CudaKernelArtifact` containing
      cubin/PTX bytes, symbol, target architecture, launch dimensions, immutable parameter/ABI descriptors, resource
      requirements, and ABI schema/version. Per-execution buffers, scalars, and pointers belong to a separate launch
      call frame. cuTile constraints, versions, tuple flattening, and `cutile_python_v2` translation stay in Phase 16.
- [ ] If the built-in plugin launcher was selected in Phase 2, wrap that one implementation instead and delete the
      experimental dynamic launcher when its tests have migrated.
- [ ] Preserve event/fence ownership, callback, error, and dropped-handle behavior. Add cancellation or timeouts only
      when the selected upstream API actually exposes them.
- [ ] Keep AOT executable serialization and reload on the existing `Executable`/`LoadedExecutable` path.

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

### Phase 5: GPU/cuTile lower-layer vertical-slice gate

**Prerequisites:** Phases 1-4. This gate blocks every later non-TPU production phase.

**Owners:** lower-layer integration tests, fixtures, and reusable artifact-inspection/qualification tooling only.

- [ ] Construct one vector-add and one tiled matmul Mosaic GPU module through typed `ryft-mlir`, verify it, run the
      documented serde/pass stages, compile through the selected `mosaic_gpu_v2` seam, execute it through CUDA PJRT,
      and reload the enclosing executable from AOT where supported. Exercise standalone lowering only when Phase 0
      found a supported interface.
- [ ] Launch the Phase 0 cuTile cubin through the selected standard XLA/PJRT route.
- [ ] Round-trip binary Mosaic GPU modules without text conversion and assert the serde version and custom-call target
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

### Phase 6: Restore reference program analysis and kernel validation foundations

**Prerequisites:** none within this plan; requires only the completed `plan-reference-discharge.md` architecture
already in the repository, so it may proceed in parallel with Phases 0-5.

**Owners:** `ryft-core` reference analysis; `ryft-xla` experimental kernel validation.

The interpreter-style discharge rework deleted the static reference-analysis stack — the generic whole-closure
`ReferenceAnalysis` (`crates/ryft-core/src/programs/references/analysis.rs`), the array-view overlay
`ArrayReferenceAnalysis` (`crates/ryft-core/src/arrays/reference_analysis.rs`), and the preserved-reference kernel
validator mock (`crates/ryft-xla/src/experimental/reference_kernels.rs`) — because after that rework its only live
production consumer was an entry-boundary fact (now an inline scan in the eager replay preflight) and its remaining
purpose was this plan's kernel work. The deleted files were never committed, so restoration re-implements against the
contracts recorded in `plan-references.md` (and approximate transcript-recovered copies archived at deletion time)
rather than reverting a commit.

- [ ] Rebuild the generic `ReferenceAnalysis`: root/alias/access resolution, capture scopes, region-input bindings,
      region-output forwarding, transitive instruction summaries, and lifetime/second-class boundary validation over
      complete region closures (condition, while, scan, and call-like operations).
- [ ] Rebuild the array-view overlay (`ArrayReferenceAnalysis`) deriving each validated root-relative
      `ArrayReferenceView` from the generic alias edges exactly once.
- [ ] Rebuild kernel-body validation on that analysis — second-class boundaries, declared access modes, liveness, and
      per-root alias/view maps for lowering — targeting the real kernel operation of the phases below rather than the
      deleted mock.
- [ ] Keep the three-rung prevention ladder (trace time, eager runtime, discharge) as the default for non-kernel
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

### Phase 7: Add backend-neutral kernel types and semantic IR

**Prerequisites:** Phases 5-6.

**Owners:** new or existing kernel-owned modules in `ryft-core`; no target dialect dependency.

- [ ] Promote the proven experimental boundary into a backend-neutral higher-order `KernelCallOperation` and region.
- [ ] Define grid, parameter, static-argument, access, alias, scratch, source-location, capability-requirement, and
      compiler-policy types without backend-specific fields.
- [ ] Reuse `ArrayType`, `ReferenceType`, `ArrayReferenceView`, the Phase 6 restored `ReferenceAnalysis`, effects,
      identities, and parameter structures rather than copying them.
- [ ] Define kernel-specific effect/resource classes for memory, async operations, barriers, and atomics while keeping
      ordinary program effect semantics intact.
- [ ] Specify deterministic display, hashing, equality, identity renaming, refinement, serialization eligibility, and
      source schema version.
- [ ] Parameterize the kernel operation family over a typed extension contract. The initial portable API uses no
      extensions; Phase 13 supplies a sealed experimental `ryft-xla` enum whose variants declare full type, effect,
      resource, liveness, race, transform, capability, and deterministic target-simulation semantics. Reject opaque
      extension payloads in verified bodies.

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
- [ ] Implement path-sensitive definite initialization through conditions and fixed-point loops.
- [ ] Add masked load/store/swap, ordered accumulation, device-scoped sequentially consistent atomics, and async-copy
      tokens/waits with exact operation-local reference semantics. Define the typed extension contract used later by
      target barriers and semaphores without pretending they are portable operations.
- [ ] Define the portable sequentially consistent atomic model and device scope. Reject invalid combinations
      in core; defer otherwise valid but unavailable dtype/scope combinations to backend capability selection.
- [ ] Add root/view overlap and race analysis for common affine block mappings.
- [ ] Preserve dead swap-result store optimization only when old contents and untouched root elements are provably
      unnecessary.
- [ ] Mark the preserved-reference boundary work in `plan-references.md` as superseded by this production contract.
      Its `reference_kernels.rs` mock was already deleted when the interpreter-style discharge landed, so no
      mock-removal coordination remains.

**Tests/docs:** all access modes; write-only full-output initialization and publication; ordered repeated stores;
empty-grid/nonempty-output rejection; uninitialized reads; partial writes; escape/use-after-scope; sibling views;
masked accesses; atomic contract tests; token linearity; statically provable async ordering/races; and liveness.

**Excludes:** unsafe arbitrary races and target encodings.

**Estimate:** production 1,600-2,600; tests 1,500-2,400; docs 400-650.

**Exit criterion:** the verifier defines and enforces the decidable static memory, initialization, atomic, resource,
and synchronization contract, and rejects programs outside its proof subset. Dynamic interleaving, deadlock, and
execution conformance become executable acceptance criteria in Phase 11.

### Phase 10: Add staging, builders, and the public kernel-call surface

**Prerequisites:** Phases 7-9.

**Owners:** `ryft-core` compilation/tracing/operations plus the `ryft` facade after stabilization.

- [ ] Add typed builders for kernels, grid/block specs, scratch specs, static parameters, and compiler policies.
- [ ] Trace a Rust closure into the kernel body while keeping reference arguments second-class and non-capturable.
- [ ] Infer parameter trees, result trees, aliases, source locations, and static specialization constraints.
- [ ] Add user-facing load/store/indexing sugar only over the canonical operations; no second semantic path.
- [ ] Seal regions atomically after validation and preserve exact diagnostics through nested helper calls.
- [ ] Keep the surface experimental and out of broad crate-root exports until Phase 20.

**Tests/docs:** parameter structures, closures/captures, static arguments, malformed return trees, alias inference,
source spans, staged/eager parity, compile-fail restrictions, and concise kernel examples.

**Excludes:** decorators/macros that hide semantics, implicit host callbacks, backend selection heuristics.

**Estimate:** production 1,600-2,500; tests 1,300-2,100; docs 400-650.

**Exit criterion:** users can express and stage the supported language with typed, deterministic IR and diagnostics.

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
- [ ] Establish a backend-lowering trait only after the GPU and cuTile prototypes demonstrate its minimal common
      needs. Phase 21 must extend that contract only for concrete TPU requirements, not speculative universality.

**Tests/docs:** exact inference, folding, liveness, schedule validation, numerical edge cases, block scaling, and
interpreter equivalence for representative kernels.

**Excludes:** automatic schedule search and target instruction selection promises.

**Estimate:** production 1,500-2,400; tests 1,300-2,100; docs 350-600.

**Exit criterion:** the portable subset can describe useful kernels and all result-preserving schedule hints can be
erased without changing results; target execution-agent and synchronization contracts remain intact.

### Phase 13: Integrate kernel dispatch, ABI, caching, and execution in `ryft-xla`

**Prerequisites:** Phases 5-12.

**Owners:** `ryft-xla` experimental operation family, lowering, compilation domains, JIT facade, persistence.

- [ ] Add the higher-order kernel call to the XLA operation family and preserve its body until backend selection.
- [ ] Define one logical-to-physical runtime ABI for arrays, runtime scalars/dynamic extents, and external resources.
      Put static arguments, layouts, schedules, scratch declarations, internal tokens, aliases, resources, and compiler
      identity in one XLA-owned kernel artifact/configuration schema.
- [ ] Query normalized capabilities before selecting or compiling a backend.
- [ ] Add backend registration/selection with exact user override, portable fallback policy, and target-operation
      rejection.
- [ ] Embed backend artifacts in StableHLO custom calls with operation-local aliases and side effects, never external
      reference-state entry aliases.
- [ ] Reuse compilation dispatch, PJRT fences, persistent executables, and replacement validation.
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

**Owners:** `ryft-xla` Mosaic GPU lowerer using `ryft-mlir` typed APIs.

- [ ] Lower grids to CUDA block/CTA launch semantics and portable references to GMEM/SMEM/register operations.
- [ ] Lower scalar/tile arithmetic, masks, layouts, slices, broadcasts, reductions, ordinary dot, and bounded control
      flow for the supported Hopper-or-newer baseline.
- [ ] Lower Mosaic GPU target barriers and basic asynchronous GMEM/SMEM transfers with declared agent membership and
      verified lifetimes.
- [ ] Implement deterministic target simulation for baseline CTA/barrier/async behavior and compare ordering,
      deadlock, and race outcomes with GPU execution.
- [ ] Produce exact launch/resource metadata and StableHLO custom-call aliases.
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

**Owners:** `ryft-xla` Mosaic GPU target extensions, schedules, capability tables, qualification tests.

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

**Prerequisites:** Phases 4-5, Phase 12, Phase 13, and a still-supported Phase 0 seam.

**Owners:** isolated `ryft-xla` cuTile backend/tool driver plus existing PJRT/XLA custom-call launch path.

- [ ] Define the exact portable subset compatible with cuTile's block-level tile model, immutable local objects,
      global arrays, control flow, atomics, and no explicit intra-block synchronization.
- [ ] Translate verified kernel IR to deterministic cuTile source or an official compiler input; never translate
      arbitrary target-specific Mosaic operations.
- [ ] Compile in an isolated, cancellable, time-limited build-time process and export a `cutile_python_v2` cubin and
      manifest for the exact target GPU.
- [ ] Emit and validate a cuTile-owned manifest translated into the generic Phase 13 artifact schema:
      pointer/shape/stride argument expansion, static-shape
      constraints, constants, tuples, symbol mangling, alignment, no-alias requirements, grid, and compiler hints.
- [ ] Execute through the one Phase 4 stream/custom-call launcher, retain compiler logs, and keep the runtime
      Python-free.
- [ ] Include cuTile/compiler/CUDA versions and target GPU in cache/AOT identity.
- [ ] Support portable FP4/block-scaled operations only when the selected cuTile version documents them; exact Mosaic
      operations remain rejected.

**Tests/docs:** generated-source snapshots, AOT manifest/calling convention, tool failure and timeout, compiler sandbox,
numerical parity, cubin target inspection, no-alias failures, and NVIDIA device execution.

**Excludes:** a second cubin loader/launcher, embedding Python, translating TileIR without a documented stable API,
explicit barrier-heavy kernels, and using cuTile as the portable IR.

**Estimate:** production 1,800-3,000; tests 1,500-2,500; docs 450-750. This range has the highest pre-Phase-0
uncertainty.

**Exit criterion:** supported portable kernels compile ahead of time and execute through cuTile with a versioned,
auditable, Python-free runtime artifact; unsupported kernels fail before tool invocation.

### Phase 17: Add transforms, composition, and sharding

**Prerequisites:** Phase 13 plus the selected non-TPU backend: Phase 14 for Mosaic GPU or Phase 16 for cuTile. Phase 15
is required only for transforms over advanced GPU target extensions. TPU transform work remains in Phase 21.

**Owners:** `ryft-core` transform rules and `ryft-xla` backend legality.

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

**Owners:** `ryft-xla` compilation/persistence, `ryft-pjrt` profiling only where a demonstrated gap remains.

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

- [ ] Choose the stable portable surface and keep Mosaic GPU, cuTile, raw artifact, and exact target operations
      explicitly experimental until their upstream ABIs stabilize.
- [ ] Remove mock boundaries, temporary bridges, parallel metadata, deprecated names, and compatibility shims; update
      all in-repo users directly.
- [ ] Publish a support matrix by backend, architecture, dtype, operation, memory space, transform, distribution, and
      toolchain version for the non-TPU release. Phase 21 adds TPU rows without rewriting existing contracts.
- [ ] Add end-to-end examples: elementwise, reduction, matmul, attention, masked partial tiles, scratch pipeline,
      Hopper WGMMA, Blackwell NVFP4 `tcgen05`, cuTile, custom AD, batching, sharding, AOT, and debugging.
- [ ] Establish upgrade tooling that diffs pinned Mosaic GPU surfaces and forces an explicit decision for every
      addition, removal, or semantic change. Phase 21 adds the isolated TPU manifest workflow.
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

**Owners:** TPU-specific work across `ryft-xla-sys`, `ryft-mlir`, `ryft-pjrt`, and `ryft-xla`, plus TPU fixtures,
documentation, CI, and qualification. Portable `ryft-core` semantics are already complete and change only if a reviewed
backend-neutral defect is found.

- [ ] Freeze the pinned JAX Mosaic TPU, OpenXLA, libtpu/PJRT plugin, serde, custom-call, and hardware support matrix;
      generate the TPU-specific parity manifest and record every existing, partial, missing, or unsupported surface.
- [ ] Complete the open-source `ryft-xla-sys` TPU bridge: safe construction/accessors for `VectorLayoutAttr`,
      `TiledLayoutAttr`, and other required types; serde/pass registration; communication/custom-barrier analysis;
      documented bytecode/version functions; archive/export symbols; and exact ownership/error contracts.
- [ ] Treat libtpu as the closed-source runtime/compiler owner. Submit supported custom calls through PJRT and consume
      only documented diagnostics, metadata, serialization, and profiling hooks; do not bind private compiler internals.
- [ ] Reconcile the existing 86-operation `ryft-mlir` Mosaic TPU surface with the pinned manifest. Complete required
      types, attributes, interfaces, operations, typed constructors/accessors, serde/pass APIs, layout inference,
      communication analysis, memory spaces, and canonical parsing/rendering.
- [ ] Add one focused construction/accessor/module-verification/complete-rendering test for every new or changed
      concrete TPU operation, attribute, and type, plus exact malformed DMA/semaphore/layout/MXU cases and serde/pass
      snapshots. Keep standard Vector/Arith/SCF/MemRef behavior in its owning dialect.
- [ ] Add only raw TPU plugin/device/topology/version/extension facts genuinely missing from `ryft-pjrt`. Keep Mosaic
      capability normalization, backend configuration, semantic schemas, and compiler policy in `ryft-xla`.
- [ ] Pass the TPU lower-layer gate with hand-authored VMEM, HBM transfer, DMA/semaphore, and MXU modules. Verify and
      serialize open-source IR, construct the exact custom call, execute on TPU, and inspect the strongest supported
      compiler evidence. Serialize/reload executables only when the plugin advertises it; otherwise prove deterministic
      recompilation from the serialized program and assert the exact unsupported result.
- [ ] Add normalized TPU capabilities and lower portable grids, scalar prefetch, HBM windows, VMEM/SMEM work, scratch,
      scalar/vector/tile operations, MXU matmul, transfers, DMA, semaphores, and supported double-buffered pipelines.
- [ ] Implement deterministic TPU target simulation for DMA engines, semaphores, and declared core participants;
      compare ordering, deadlock, race, precision, layout, and numerical behavior with supported TPU execution.
- [ ] Emit the exact TPU backend configuration and typed `stablehlo.custom_call` attributes for aliases and side
      effects. Keep memory-space, source, communication, collective, and compiler metadata in their owning fields.
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

## 9. Likely change surface

### `ryft-xla-sys`

- Mosaic common/GPU/TPU C++ bridges and Rust FFI modules.
- Bazel dependencies, visibility patches, exported symbols, source archive manifests, and build feature gates.
- Compiler/pass/serde/artifact/version APIs proven by Phase 0 for GPU and Phase 21 for TPU.
- Possibly CUDA/XLA FFI stream or custom-kernel declarations required by the selected cubin launch seam.

### `ryft-mlir`

- Mosaic GPU/TPU operations, attributes, types, passes, pipelines, and compiler facades.
- Standard dialect wrappers only where a concrete kernel lowering needs a missing operation or attribute.
- Generated parity manifests and source-owned typed wrappers/tests.

### `ryft-pjrt`

- Existing program, execution, stream, FFI, GPU custom-call, metadata, topology, profiling, and distributed modules.
- No Mosaic-specific executable hierarchy and no cuTile-specific PJRT extension absent an upstream standard.

### `ryft-core`

- Kernel types/operations, grids, block mappings, references/scratch, effects, verifier, interpreter, builders, and
  transform rules.
- Existing reference, array, tracing, compilation, differentiation, batching, partial-evaluation, and program modules
  only where the kernel operation participates in their established contracts.

### `ryft-xla`

- Experimental kernel operation integration, lowering, backend selection, Mosaic GPU, Mosaic TPU, cuTile tool driver,
  dispatch/runtime, capability normalization, persistence, profiling, and public experimental facades.
- The `plan-references.md` kernel mock (`reference_kernels.rs`) has already been removed; the production kernel
  boundary is built fresh here.

### `ryft`

- Stable portable re-exports and examples only in Phase 20; TPU-specific facade additions remain in Phase 21.

## 10. Test matrix

Use named families rather than a Cartesian product.

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
- GPU/PTX compiler tests that do not require hardware. TPU compiler-only coverage begins in Phase 21.
- Rustdoc with warnings reviewed, doctests, examples that have a CPU/interpreter route.
- `git diff --check`, 120-column added-line audit, generated manifest parity, and no stale scaffolding.

### Tier B: accelerator presubmit smoke

- One baseline Mosaic GPU kernel and one advanced feature appropriate to the available GPU.
- One cuTile AOT/launch kernel on a supported NVIDIA runner.
- AOT reload, asynchronous completion, exact capability mismatch, and target artifact inspection.

### Tier C: scheduled hardware qualification

- Hopper and Blackwell matrices, including WGMMA/TMA and NVFP4/TMEM/`tcgen05`.
- Concurrency, stress, sanitizers where available, leak detection, distributed failures, and performance variance.

### Tier D: non-TPU release qualification

- Full crate/facade/docs/examples matrix.
- Exact supported platform/toolchain versions and AOT compatibility.
- Reproducible clean-room artifact build and deployment without Python.
- Upgrade-manifest diff, security/licensing review for optional cuTile tooling, and independent zero-findings audits.

### Tier E: final TPU qualification

- Run only in Phase 21 after the non-TPU release matrix is already clean.
- Cover compiler-only TPU parity/configuration tests without hardware, then supported TPU generations with
  VMEM/SMEM, DMA/semaphores, MXU, pipelines, transforms, persistence, multi-core, and distribution.
- Repeat the full affected crate/facade/docs matrix with TPU features enabled and record exact libtpu/plugin versions.

All expensive local commands should use the repository's 300-second default timeout unless a specific hardware test has
a reviewed longer bound.

## 12. Delivery milestones and dependency graph

### Milestone A: lower-layer compiler/runtime foundation

Phases 0-5. This is a hard non-TPU gate, not preparatory work that can be papered over later.

- `ryft-xla-sys`: missing Mosaic registration/serde/runtime ABI and pinned archive parity.
- `ryft-mlir`: complete typed Mosaic GPU and required standard compiler-dialect construction and pipelines.
- `ryft-pjrt`: only demonstrated generic capability/metadata/runtime gaps.
- GPU and cuTile hand-authored vertical slices.

### Milestone B: portable language and semantic oracle

Phases 6-12: restored reference analysis, core types, grids, references, memory/synchronization safety, staging,
interpreter, operations, and manual
schedules.

### Milestone C: shared XLA runtime and Mosaic GPU baseline

Phases 13-14: one dispatch/cache/ABI plus production baseline Mosaic GPU.

### Milestone D: latest hardware and optional backend breadth

Phases 15-16: Hopper/Blackwell and cuTile.

### Milestone E: composition and production operations

Phases 17-20: transforms, profiling/autotuning/AOT, GPU distributed/asynchronous execution, stabilization, and the
non-TPU release.

### Milestone F: consolidated TPU support

Phase 21 alone owns every TPU-specific native, typed-IR, runtime, lowering, transform, distribution, documentation,
and qualification task. It begins only after Milestones A-E are independently complete.

Safe parallelism after Phase 5:

- Phase 6 requires nothing from Phases 0-5 and may start immediately.
- Phases 7-9 may divide by semantic owner but must merge before staging/interpreter work completes.
- cuTile may proceed beside Mosaic backends after its Phase 0 seam and portable subset are fixed.
- Transform and profiling work should consume at least two working backends to avoid single-backend abstractions.
- TPU work never proceeds in parallel with Phases 0-20; that isolation is the purpose of final Phase 21.

## 13. Aggregate estimates

Summing the phase ranges in §7 gives approximately:

- **Production:** 40,200-66,650 logical lines.
- **Tests:** 38,000-62,550 logical lines.
- **Docs/examples:** 11,180-19,050 logical lines.
- **Total:** 89,380-148,250 logical lines, excluding generated/vendored code.

Expected production ownership:

| Crate/area | Indicative midpoint share | Main work |
|---|---:|---|
| `ryft-xla-sys` | 4% | missing registration/serde/runtime bridges and builds |
| `ryft-mlir` | 27% | typed compiler dialects, Mosaic parity, verification, and pipelines |
| `ryft-pjrt` | 5% | demonstrated generic capability/metadata/runtime gaps |
| `ryft-core` | 24% | language IR, verifier, interpreter, builder, transforms |
| `ryft-xla` | 38% | ABI, dispatch, three backends, caching, tuning, runtime |
| facade/examples/tooling | 2% | stable exports, examples, manifests, qualification tools |

These are planning ranges, not commitments. Phase 0 replaces non-TPU estimates using its pinned manifest, and Phase 21
does the same for TPU before implementation. Each phase records actual logical lines and explains a variance above 30%
before the next phase begins.

## 14. Risks and mitigations

### Reimplementing an existing lower layer

**Risk:** new dialect builders, executable types, or compiler wrappers duplicate mature code.

**Mitigation:** machine-checkable pinned manifest, extend existing facades, and require a concrete missing symbol or
contract for every lower-layer addition.

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

**Risk:** Mosaic or cuTile changes break wrappers and cached artifacts silently.

**Mitigation:** paired source pins, generated parity manifest, ABI/schema versions, source-owned C boundary, exact cache
identity, and upgrade CI that requires decisions for every surface change.

### Premature universal backend abstraction

**Risk:** a large trait hierarchy encodes guesses and makes target-specific work harder.

**Mitigation:** build the minimal shared contract from the interpreter plus working Mosaic GPU and cuTile prototypes;
keep backend lowerers concrete until real duplication appears. Final TPU work adapts to the proven contract.

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

**Mitigation:** milestone gates, named non-goals, Mosaic GPU/cuTile production before TPU work, and explicit deferral
that does not weaken already shipped semantics.

## 15. Review checkpoints

Do not begin a checkpoint until the prior one is independently audited:

1. **Pinned-source review:** exact non-TPU lower-layer manifest and official cuTile seam.
2. **Native ABI review:** ownership, errors, symbols, artifacts, and feature gates.
3. **Typed MLIR review:** parity, verification, pass pipelines, and no stringly production paths.
4. **Runtime review:** standard PJRT/XLA lifecycle, capabilities, streams, fences, and AOT.
5. **Lower-layer gate review:** real GPU/cuTile vertical slices and target-code evidence.
6. **Semantic review:** grids, block mappings, references, scratch, effects, atomics, synchronization, and races.
7. **Interpreter review:** deterministic oracle and diagnostic completeness.
8. **Shared ABI review:** backend selection, aliases, cache identity, persistence, and failures.
9. **Mosaic GPU review:** baseline, Hopper, Blackwell, NVFP4, and exact instruction use.
10. **cuTile review:** supported subset, tool isolation, AOT ABI, and Python-free deployment.
11. **Transform/distribution review:** every non-TPU route proven or rejected; no duplicated effects.
12. **Non-TPU production review:** docs, examples, CI, security, upgrades, performance, and no temporary scaffolding.
13. **Final Mosaic TPU review:** native/typed foundations, memory, DMA, semaphores, MXU, transforms, pipelines,
    multi-core, distribution, documentation, and zero findings.

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
- portable semantics have an interpreter and immutable oracle;
- Mosaic GPU and cuTile execute before Phase 21 without TPU dependencies, and the final Mosaic TPU stack executes on
  supported TPU hardware only in Phase 21;
- Hopper/Blackwell and NVFP4 tests prove exact target instructions and ordering;
- cuTile support uses an official versioned AOT seam with no Python runtime dependency, while installation remains
  optional;
- every transform, dynamic/distributed class, backend, and hardware feature is documented as supported or rejected;
- AOT/cache compatibility and failure behavior are deterministic;
- public docs and examples explain semantics without relying on physical in-place reuse or implementation knowledge;
- no phase-owned TODO, mock boundary, deprecated alias, compatibility shim, parallel universe, or stale identifier
  remains;
- exact full verification counts and hardware/toolchain versions are recorded;
- independent correctness, convention, security, and simplicity auditors report zero findings.

## 17. Primary external references

Revalidate the GPU, cuTile, XLA, and PJRT sources in Phase 0. Revalidate the two Mosaic TPU sources only when Phase 21
begins:

- [JAX Pallas overview](https://docs.jax.dev/en/latest/pallas/): language status and guide index.
- [Pallas quickstart and programming model](https://docs.jax.dev/en/latest/pallas/quickstart.html): grids,
  `BlockSpec`s, program IDs, and current GPU/TPU backend mapping.
- [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html): references, loads/stores, backend
  lowering, and transform model.
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
- [x] Checked current official Pallas, Mosaic GPU, Mosaic TPU, and cuTile documentation.
- [x] Made completion of non-TPU `ryft-xla-sys`, `ryft-mlir`, and `ryft-pjrt` work a hard gate before core/XLA
      production work.
- [x] Distinguished existing lower-layer coverage from partial and missing compiler/runtime work.
- [x] Kept cuTile behind an official AOT seam rather than inventing an MLIR dialect or PJRT extension.
- [x] Added separate Mosaic GPU baseline, Hopper/Blackwell, and cuTile phases, followed by one consolidated final TPU
      phase that does not block the preceding roadmap.
- [x] Modeled NVFP4 as values, scales, geometry, accumulation, layout, and exact instruction capabilities.
- [x] Added phase-specific production, test, and documentation ranges with explicit methodology.
- [x] Reconciled arithmetic totals, path/link checks, and independent foundation, architecture, and conventions audits;
      all three original-plan auditors reported zero findings.
- [x] Consolidated every TPU-specific implementation and verification deliverable into final Phase 21, renumbered all
      prior phases, and revalidated 21 complete phase templates, dependencies, estimates, paths, line widths, and diff
      hygiene without changing a changelog.
- [x] Added restoration Phase 6 after the interpreter-style discharge rework deleted the static reference-analysis
      stack and the `reference_kernels.rs` mock this plan previously named as retained pieces; renumbered later
      phases, reconciled the estimate table, aggregate totals, milestones, and every cross-reference, and rewrote the
      retained-pieces inventory to record the deletion.
