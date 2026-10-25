# Ryft Megakernels: Architecture and Execution Plan

**Status:** proposed extension; no implementation or performance result implied

**Review date:** 2026-10-24

**Companion:** [Ryft inference plan](plan-inference.md)

## 1. Decision and motivation

Retain XLA as the outer compilation and execution system. Make selected regions of a Ryft program eligible for
specialized implementations that own their internal tiling, synchronization, and scheduling. A region may become an
ordinary fused kernel, a bounded persistent kernel, or a device task graph, depending on hardware and workload evidence.

Megakernels are relevant because they can remove coarse operation boundaries and overlap computation, memory traffic,
and communication at finer granularity. XLA fusion and command-buffer replay already address some overhead, but they
are not equivalent to a persistent task scheduler. The opportunity is to compose these mechanisms through a measured
region boundary, while keeping one model implementation and the inference plan's resource protocol.

Ryft should make kernel authoring, verification, compilation, deployment, and comparison easier. It should not promise
that ordinary model code automatically becomes an efficient whole-model megakernel. Region selection and schedules are
initially explicit, with automatic outlining considered only after successful concrete implementations.

## 2. Terminology and tradeoffs

| Mechanism                   | Execution boundary                                        | Main opportunity                                                                         |
|-----------------------------|-----------------------------------------------------------|------------------------------------------------------------------------------------------|
| XLA fusion                  | Compiler-selected group of operations                     | Remove intermediate work and storage; generate efficient kernels.                        |
| Command-buffer/graph replay | Recorded sequence of device operations                    | Reduce repeated submission overhead while retaining operation boundaries.                |
| Bounded persistent kernel   | Workers remain active for a finite region                 | Pipeline tasks and reuse resources without repeated kernel boundaries.                   |
| Device task graph           | Explicit tasks and readiness dependencies within a kernel | Schedule work at tile granularity across workers, potentially overlapping communication. |
| Long-lived serving kernel   | Device workers span multiple externally admitted steps    | Requires a separate host/device control protocol; outside the initial scope.             |

A megakernel need not cover a full model or many generated tokens. One persistent decode-step implementation is already
a meaningful experiment. Persistence does not require cooperative launch in every design, but each design must prove
its own residency and progress assumptions.

The expected benefit depends on the bottleneck. Low-concurrency decode can suffer from execution gaps and weight-memory
stalls; larger workloads may benefit from overlapping complementary compute and communication. Costs include scheduler
workers, barriers, queue traffic, register pressure, shared-memory demand, instruction footprint, and lost opportunities
to use specialized library kernels. Global-memory traffic does not disappear simply because operations share a launch.

Compare against tuned captured execution. Fewer launches are diagnostic evidence, not an acceptance criterion.

## 3. Existing Ryft substrate and limits

These observations describe the current checkout; revalidate them before implementation, especially where concurrent
kernel work changes the support matrix.

| Existing owner | Useful foundation | Boundary |
|---|---|---|
| [Core kernels](crates/ryft-core/src/kernels/mod.rs) | `#[kernel]`, canonical Ryft programs, typed tiles and references, bounded control flow, interpretation, verification, scheduling and compilation contracts. | `VerifiedKernel` validates its supported model; it does not certify arbitrary device code or an unmodeled global scheduler. |
| [XLA kernel integration](crates/ryft-xla/src/kernels/mod.rs) | Embedding compiled kernels as custom calls, executable caching, AOT, and completion ownership. | The compiled region is opaque to general XLA optimization; alias/effect declarations must describe all external accesses. |
| [Mosaic GPU](crates/ryft-mosaic/src/kernels/gpu/mod.rs) | CTA-local and two-CTA cluster execution, TMA/WGMMA/TMEM capabilities, resource checks, and synchronization simulation. | Local/cluster synchronization does not establish arbitrary cross-SM task scheduling. |
| [Triton adapter](crates/ryft-triton/src/kernels/mod.rs) | Typed compilation of an admitted portable kernel subset. | Current admission is restricted; upstream Triton capabilities do not establish adapter support for persistent state, barriers, atomics, or scratch. |
| [CUDA artifacts and launch](crates/ryft-cuda/src/lib.rs) | Producer-neutral PTX/cubin artifacts, checked launch frames, borrowed streams, and module ownership. | Current launcher rejects graph capture and excludes cooperative launches and runtime cluster configuration. |

Reuse these owners. Extend existing modules when practical; do not create a parallel kernel language, artifact cache,
or generic device-runtime crate before a concrete implementation requires it. Follow the owning operation, native
wrapper, and compiler-maintenance skills when implementation reaches those contracts.

## 4. Region architecture

The intended composition is:

```text
Shared Ryft model/program
  -> explicitly selected semantic region
     -> ordinary Ryft/XLA implementation
     -> specialized kernel-region implementation
  -> common physical buffers, state leases, and completion protocol
```

The region's reference body defines inputs, outputs, state updates, and numerical semantics. Its selected implementation
may combine several existing operations without duplicating the model architecture. Preserve the canonical body for
reference execution, interpretation, and fallback. Initially bind implementations explicitly; avoid a broad
graph-pattern rewriter until region identity and legality have been demonstrated on real examples.

A provider must declare:

- model/operation semantics, numerical mode, valid shape ranges, and target capabilities;
- physical layouts, alignments, logical-length metadata, masks, and inactive-lane behavior;
- all state inputs/outputs, aliases, effects, scratch, and completion dependencies;
- task/worker resources, supported synchronization scopes, and termination assumptions;
- graph compatibility, workspace stability, artifact identity, and compiler/runtime requirements.

An implementation is selected during profile compilation or warmup. Logical request lengths and task metadata may vary
within bounded capacities; request identities and prefix contents do not create new compilation keys. Selection includes
ordinary XLA and smaller fused alternatives so a larger region can be disabled outside its measured envelope.

Ryft transformations remain centered on the semantic body. A serving implementation need not have a backward kernel
when training can use the ordinary differentiable implementation. Unsupported transformation combinations must fail
explicitly or select a supported implementation; differentiation through an opaque binary is not automatic.

XLA sees dependencies at the custom-call boundary. If an optimization needs tile-level producer/consumer overlap,
include both sides within the region. A separate custom call for each operation cannot generally expose that schedule.
Precompiled host-launched kernels also cannot simply be concatenated into a device task runtime; task implementations
need a compatible calling convention, resource model, and synchronization discipline.

## 5. Device execution and correctness contracts

Start with a finite task set and bounded execution. Specify task inputs/outputs, storage regions, readiness counters,
worker membership, and completion generations. A device dependency must establish memory visibility at the correct
CTA, cluster, device, or peer-device scope, not merely increment a counter. Readiness and scratch reuse must account
for every consumer, including asynchronous transfers.

Progress is a separate obligation from data-race safety. Resident workers must not wait for producers that cannot be
scheduled because waiting blocks occupy all resources. Use a justified residency bound, an appropriate cooperative
mechanism, or a work-queue protocol that guarantees runnable producers can execute. Reject unsupported launch modes
before submission. Do not infer GPU progress or memory ordering from Rust ownership or from an acyclic abstract graph.

Extend verification only for modeled semantics. Useful checks include initialization, disjoint writes, dependency
coverage, counter generations, queue bounds, barrier participation, and scratch lifetimes. Simulation and static checks
are evidence within that model; hardware stress and numerical tests remain necessary.

Expose model state and metadata as ordinary explicit buffers with validated aliases and effects. Hidden global queues,
raw-pointer mutation outside declared state, or undocumented cross-launch communication would defeat compiler ordering
and cancellation safety. Retain task metadata, scratch, modules, weights, and state leases until device completion.
Distinguish successful submission from successful completion and apply the inference plan's failure/quarantine rules.

Cancellation initially suppresses publication and further launches while the bounded region finishes. A timeout does
not prove device quiescence. Later stop/yield points require a defined consistent frontier; a stop flag alone cannot
safely interrupt arbitrary collectives or leave a partial state update publishable. Choose region duration so admission,
streaming, fairness, and weight publication remain responsive.

Multi-GPU regions are a later capability. They require a compatible device communication mechanism, peer-memory
visibility, rank participation, and coherent failure handling. A host NCCL call cannot simply execute inside a kernel.
Coordinate internal communication with surrounding XLA collectives and avoid incompatible ordering or resource usage.

## 6. Authoring and integration ergonomics

Build on the current kernel authoring APIs. A first user should be able to retain a normal reference body, attach one
specialized implementation, compile it once for a bounded profile, and obtain a numerical comparison and execution
trace. The design should make layouts, resource bounds, dependencies, and unsupported cases inspectable.

Separate semantic choices from tuning. Tile shapes, worker allocation, buffering depth, and static versus dynamic task
schedules may be tuned, but a tuning parameter cannot invent synchronization, relax numerical requirements, or change
state identity. Artifact/cache keys include all semantic and execution-relevant choices.

Support external CUDA implementations through the same deployment contract where useful. Generated or handwritten
artifacts are equally subject to ABI, ownership, numerical, and synchronization validation. Preserve an escape hatch
for hardware-specific implementations rather than requiring every experiment to pass through a new universal DSL.

For graph capture, preinitialize modules and bounded workspace before capture, retain referenced modules for the graph's
lifetime, and validate replay with changed metadata and supported addresses. The current launcher must be extended or
another qualified integration used; advertising FFI capture compatibility cannot bypass its rejection. Avoid allocation,
module eviction, or context-wide synchronization inside a captured launch path. Compare eager and captured results and
traces, including replay after profile and weight changes.

## 7. Staged execution plan

### Stage 0 — select a bottleneck and register the experiment

- [ ] Pin hardware, numerical policy, profile shapes, workload, compiler revision, and memory budget.
- [ ] Profile a captured XLA path and identify launch gaps, memory traffic, synchronization, or communication that a
      region could improve. Estimate the potential end-to-end benefit before building another scheduler.
- [ ] Select one bounded region with a reusable semantic body. Candidates include normalization/projection chains,
      small decoder subregions, or attention/postprocessing; MoE requires a separately selected model.
- [ ] Record unsupported modes and success/regression budgets before tuning.

Exit: a falsifiable performance hypothesis and a reference implementation. Synthetic cases establish mechanics;
full-model inference claims still require the Gemma acceptance surface in the inference plan.

### Stage 1 — physical integration proof

- [ ] Implement the smallest useful region with the existing kernel path or an external artifact provider.
- [ ] Prove XLA buffer aliases, state mutation, stream ordering, repeated launches, and completion-safe resource reuse.
- [ ] Qualify the required graph path, module lifetime, workspace, and replay behavior; expose unsupported launches.
- [ ] Test numerical agreement, inactive lanes, changed metadata, delayed completion, and cancellation after submission.

Exit: an executable bounded region that composes with XLA and has an explained time/memory breakdown. It need not use
persistent workers yet; establish deployment before introducing device scheduling complexity.

### Stage 2 — bounded persistent scheduling

- [ ] Add persistent execution only where Stage 0 identified a remaining opportunity. Compare static task schedules
      with a dynamic queue before selecting a general scheduler.
- [ ] Model task dependencies, memory ordering, worker resources, and progress. Add adversarial queue/barrier tests.
- [ ] Measure scheduler work, occupied SMs, queue traffic, synchronization, register pressure, and spill/scratch costs.
- [ ] Compare ordinary captured XLA, the smaller fused region, and the persistent implementation under equal semantics.

Exit: repeated measurements demonstrate an attributable benefit without violating correctness or serving budgets.
If persistence loses, retain a useful smaller fusion and record the limiting cost.

### Stage 3 — serving integration and larger regions

- [ ] Select qualified variants by warmed execution profile with automatic fallback outside the supported envelope.
- [ ] Integrate dynamic arrivals, mixed prefill/decode, speculation, cancellation, and weight churn incrementally.
- [ ] Expand to multiple layers or a full decode step only when dependency and resource analysis supports it.
- [ ] Demonstrate the registered full Gemma workload and SLO contract before claiming engine-level improvement.

Exit: improved end-to-end latency or goodput with stable memory and bounded fairness/cancellation behavior. This work
can follow the inference plan's physical proof and minimal engine; it does not delay their exit gates.

### Stage 4 — optional distributed task regions and automatic outlining

- [ ] Prototype fused compute/communication on a measured topology bottleneck with explicit communication ownership.
- [ ] Validate peer visibility, rank failure, collective ordering, and graph/profile transitions.
- [ ] Consider automatic region discovery only after several explicit regions demonstrate common legal boundaries.

Exit: the broader mechanism earns its complexity through reproducible gains. Neither a multi-GPU device runtime nor
a long-lived serving kernel is a prerequisite for successful bounded region support.

## 8. Measurement and acceptance

Use the inference plan's controlled numerical and quality-constrained tracks. Hold checkpoint, precision, sampling,
state layout, resource budget, and workload constant when attributing a mechanism's effect. Report any differences.

Compare ordinary uncaptured execution, captured XLA, a conventional fused region, and persistent execution as
applicable. Separate kernel quality from scheduling gains; include compilation, autotuning, capture, startup, and
resident memory. Benchmark low concurrency, saturated batches, varied context lengths, and mixed traffic. Do not assume
either latency or throughput is the only potential benefit.

Record full-step latency, SLO goodput, output delivery gaps, GPU idle time, HBM traffic, register spills, shared memory,
worker/scheduler utilization, synchronization stalls, communication overlap, and cancellation/drain delay. Charge all
work, including schedule construction, scratch initialization, and discarded speculative work. Avoid inferring byte
savings or occupancy from launch counts when hardware counters are unavailable.

Stage 0 sets a minimum worthwhile gain and a regression bound using measured baseline variance. Promotion requires
repeated-run uncertainty supporting the gain and an explained breakdown. The inference plan's competitive and leadership
criteria continue to govern engine claims; microbenchmark improvements cannot replace them.

## 9. Primary sources and evidence limits

- [Mirage persistent-kernel compiler and runtime](https://arxiv.org/abs/2512.22219)
- [Mirage implementation](https://github.com/mirage-project/mirage)
- [Hazy Research low-latency megakernel](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- [Hazy Research throughput-oriented tensor-parallel megakernel](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)
- [Hazy Research implementation](https://github.com/HazyResearch/Megakernels)
- [XLA custom-call and GPU-stream integration](https://openxla.org/xla/custom_call)
- [XLA command-buffer tuning](https://openxla.org/xla/command_buffer_tutorial)
- [CUDA graph execution model](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)

These sources establish mechanisms and research precedents, not expected Ryft speedups. No megakernel benchmark was
run for this plan. Pin implementation revisions and reproduce relevant behavior when selecting a provider.
