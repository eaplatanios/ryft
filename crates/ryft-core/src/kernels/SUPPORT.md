# Kernel support and qualification

The portable entry point is `ryft_core::kernels::kernel`; explicit `KernelDefinition::trace` builders produce the same
canonical program. Compiler admission is authoritative for a particular operation, dtype, shape, layout, schedule and
target. The table below is an implementation inventory, not a claim of universal lowering or a stable upstream ABI.
Mosaic GPU, cuTile, raw artifact interfaces and exact GPU operations remain experimental. TPU and direct Triton are
later phases; neither is a prerequisite for using the non-TPU authoring surface.

## Portable API contract

The non-TPU milestone preserves the portable authoring and semantic contract: `kernel` with `Array` annotations,
`KernelDefinition` tracing/verification/interpretation, logical grids and block mappings, explicit boundary and access
policies, schedules, and the documented transform rules. Compiler selection stays outside a portable definition;
changing adapters cannot change its semantic identity. Unsupported syntax or semantics must fail explicitly. Persisted
source uses its declared schema version rather than treating Rust implementation layout as a file format.

This stability boundary does not include upstream compiler IR, backend artifact layouts, exact GPU extension families,
or experimental XLA profiling/tuning/AOT/distributed integration. Those interfaces continue to validate pinned versions
and live capabilities. TPU and direct Triton can add adapters without changing the portable contract. No crate-root
facade or compatibility bridge is needed: the canonical authoring path remains `ryft_core::kernels`.

## Supported boundaries

| Area | Portable core | Mosaic GPU | CUDA cuTile |
|---|---|---|---|
| Authoring | Macro and explicit builders | Same portable definition; explicit `GpuOperation` extensions | Same portable definition; extensions rejected |
| Values | Canonical array/dimension types | Boolean, I32/U32, I64/U64, F32/F64 baseline; operation-specific half/FP8/packed types | Boolean, I32/U32, I64/U64, F16/BF16, F32/F64 |
| Shapes | Bounds, mappings, masked edges | Specialized dense row-major device arrays | Specialized dense row-major device arrays; zero-sized external arrays rejected |
| Numerics | Canonical dtype/rounding semantics | Scalar baseline or explicitly selected legal tensor-core form | Native operation admission; no implicit TF32 substitution |
| Memory | References, views, scratch, masked/async operations | Shared scratch, checked async copy/TMA and explicit TMEM protocols | Global windows and clipped gather/scatter; no scratch, atomics or async protocols |
| Control flow | Canonical condition/while/scan/call machinery | Uniform admitted body control flow and sequential grid axes | Admitted condition/while and sequential grid axes |
| Transforms | Static batching, scalar-prefetch specialization, explicit derivative contracts | Re-admit transformed body | Re-admit transformed body |
| Distribution | Canonical sharding metadata | Validated local shards; explicit admitted cluster operations are not distributed PJRT mutation | Validated local manual shards; no synthesized collectives |
| Runtime | No accelerator dependency | XLA-owned embedding and existing Mosaic runtime | XLA-owned embedding and existing CUDA launcher |

Whole-call composition with outer conditions, loops, scans, calls, memory transfers and rematerialization uses the
existing XLA operation family. This does not imply that every outer operation can occur inside an adapter's kernel
body. Direct mutable-body AD is rejected: provide the execution integration's custom JVP/VJP contract, or a pure
canonical fallback with no lost observable effects. Unmapped read-write batching and unsupported ragged/dynamic
batching fail before launch. Named manual axes must match the enclosing shard-map descriptors, not merely their names.
The host-staged distributed route described below remains experimental and does not synthesize native collectives.

## Targets and evidence

| Route | Exact scope | Qualification distinction |
|---|---|---|
| Mosaic baseline | Hopper-or-newer target admission, operation-dependent | GB10 `sm_121` native vector/reduction/matmul/attention and edge cases |
| Hopper WGMMA | CC 9.0, 128 threads, F16/BF16 operands and FP32 accumulation | Generated/serialized native source and assembled PTX; no Hopper device execution on Spark |
| Warp NVFP4 | CC 12.0/12.1, packed E2M1 and E4M3 block16 scales | CC 12.1 native dense/sparse fixtures and isolated invalid-metadata rejection |
| TCGEN/TMEM | Exact CC 10.0/10.1/11.0 admission, one/two CTA, explicit allocation/commit/wait/release | Datacenter PTX/assembler evidence is distinct from device execution; Spark cannot execute these forms |
| cuTile | Pinned tool accepts `sm_100`, `sm_103`, `sm_110`, `sm_120`, `sm_121` | Native qualification is `sm_121`; other target spellings do not imply device qualification |

Mosaic source compatibility follows the XLA and JAX commits in `crates/ryft-xla-sys/WORKSPACE`, plus native
serialization and ABI versions. XLA validates execution facts separately: in particular CC 10.1 with CUDA 12.9 differs
from CC 11.0 with CUDA 13.2/PTX 9.0. Do not infer compatibility from a broad architecture family name. cuTile currently
pins cuda-tile 1.5.0, tileiras 13.3.36, nvcc/nvvm 13.3.73 and bytecode 13.3; its XLA integration admits the pinned CUDA
13.2 PJRT platform. The CUDA README describes optional `cutile` versus `cutile-compiler` features and Python-free
reload.

## Existing runnable examples

Use existing owner fixtures rather than copying a second kernel implementation:

- `crates/experimental/src/kernels.rs`: macro vector addition, reduction, masked partial matmul, independent attention
  oracle, async scratch pipeline, TMA, cluster communication, packed NVFP4, cuTile and local shard-map execution.
- `crates/ryft-mosaic/src/kernels/gpu/lowering/tmem.rs`: one/two-CTA TMEM lifetime, MXFP8 and signed-scale NVFP4 source
  qualification. `lowering/mma.rs` owns WGMMA lowering tests; these are not GPU execution substitutes.
- `crates/ryft-xla/src/kernels/staging.rs`: explicit custom AD, pure fallback, batching, rematerialization and outer
  control-flow examples. `crates/experimental/src/kernels.rs` also exercises manifest and executable reload with the
  compiler disabled, proving deployment does not invoke Python.
- `crates/ryft-macros-tests/tests/test_kernels.rs`: runnable macro syntax, generated definition equivalence and exact
  compile-fail diagnostics. Core `interpretation.rs` and `scheduling.rs` own bounded tracing/interleaving examples.

Select the documented environment gates in `crates/experimental/README.md`. Run assertion and invalid-metadata
failures in separate processes because a device error may poison the CUDA context. A skipped prerequisite is not a
passed hardware gate. Compiler selection changes the binding, target and options, not a portable kernel's body.

## Profiling, tuning and AOT workflow

The execution integration owns these workflows in `ryft_xla::kernels`; core definitions do not acquire a runtime or
profiling dependency. The implemented workflows are qualified on GB10; broader release gates remain explicitly open.

1. Construct and verify the portable definition. Select one explicit compiler binding, target, options and live device
   mesh. Retain the semantic identity when comparing adapters or transformed definitions.
2. Inspect `KernelCompilationReport`: semantic/configuration digests and provenance correlate the existing lowering
   with typed XLA/PJRT analysis. Lowering duration includes adapter compilation; compilation duration includes the
   ordinary cache lookup. Neither is a device timestamp. Do not relabel reported memory totals as register counts,
   spills, occupancy or TMEM measurements.
3. Build a finite `KernelTuningRequest` with explicit candidate count, warmups, repetitions and time budget. Its
   environment identity must describe workload inputs, deterministic reset, contention policy and settings not already
   covered by compiler/device facts. Request identity construction is tool-free; `KernelTuningRunner::prepare` must use
   normal compiler admission and complete setup under the tuner's gate and cooperative deadline. Each execution must
   return the existing real completion; merely wrapping enqueue success in a ready handle violates the contract.
4. Choose `KernelTuner::load` or `run` explicitly. Samples measure host submission through completed execution, excluding
   preparation and warmup. Equal upper medians choose the earlier candidate. Cancellation stops further submissions but
   awaits in-flight work; incomplete measurements are not published. Share a tuner for the device and coordinate other
   processes externally. The auxiliary disk namespace contains measurements, not another executable cache.
5. Compile `KernelAotBundle` using the same definition, XLA domain and compiler options. The current bundle source codec
   admits portable, capture-free kernels with at least one ordinary input. Unsupported exact extensions fail before
   compilation; they may still use the existing native executable persistence API independently. The bundle includes
   source, lowering/report metadata and the complete existing executable envelope with backend artifacts.
6. Store `to_bytes()` output from a trusted producer. `from_bytes()` validates bounded sections, checksums and source;
   `load()` compares the expected binding and live execution facts before delegating to native deserialization and
   checking the stateless logical boundary and mesh. `LoadedKernel::call` returns the existing asynchronous execution
   handle: await it before reading results or ending the owning session. No Python compiler or implicit fallback runs
   during import. Recompiling the retained source after incompatibility is an explicit caller decision.

Mosaic and cuTile AOT reload examples have passed through the existing native experimental runner. Native tuning
qualification is tracked separately from the passing owner harness. Do not treat mock owner runners, transport-only
bundle fixtures or a successful compile as evidence of GPU timing or native execution.

## Distributed admission

Native kernel execution currently requires a fully addressable, single-process device mesh. Existing PJRT execution
fences join the participating local devices and preserve asynchronous failures. Existing `DistributedRuntime` artifact
exchange owns compilation round/process/key agreement; agreement alone is not an execution fence or atomic mutation
protocol. Cross-process participants and nonaddressable devices are rejected before resharding, donation and submission.

`ryft_xla::kernels::DistributedKernel` coordinates functional calls over an existing `DistributedRuntime` and a loaded
AOT kernel. Each participant uses one local device. Construct coordinators and invoke them in the same order on every
process; the runtime owns launch identity and coordinator sequence numbers. For each input, an explicit source-process
index selects whose value is copied. Different source rows support local calls, rings and broadcasts. Agreement checks
the same semantic/compiler contract and the ordered manifest of each participant's live execution facts.

Transfers are explicitly host-staged: existing PJRT downloads produce dense row-major bytes; bounded checksummed KV
chunks carry them between processes; PJRT uploads create fresh invocation buffers. Locally sourced inputs are also
copied. Options bound bytes, chunks, rounds and cooperative coordination deadlines. The existing trusted runtime store
retains records until shutdown and allocates received values before this layer can check their lengths. This route
therefore assumes cooperative participants; it is not a direct GPU transport or a hostile-peer allocation boundary.

`call_async` blocks during preflight and host staging, then returns the canonical `ReferenceExecution`. Awaiting its
completion drains native work and joins participant readiness before exposing outputs. There is no new worker: readiness
queries report only cached terminal state, and the next call first awaits the previous call. Cancellation prevents
further submission or publication when observed, but always drains submitted work. Dropping the coordinator and all
completion handles drains and reports abandonment; dropping only the returned handle leaves the coordinator's retained
completion alive. Transport failures and deadlines propagate through the same completion channel.

External references and observable I/O are rejected before submission. Functional outputs have fresh storage, so the
protocol cannot mutate caller-owned inputs. Successful readiness records are immutable. A network fault after readiness
may prevent delivery to one host after another receives its result; readiness is not an atomic all-host transaction.
Native cross-process meshes, aliased external-state publication and synthesized collectives remain unsupported.

## Qualification and regression policy

Start each potentially expensive command with `timeout 300`; coordinate Cargo so lock waits do not consume the budget.
Record the command, source tree and relevant artifact hashes, tool versions, device facts, feature graph and result.
If unrelated concurrent edits require an isolated baseline, record every substituted file; never label it current HEAD.

Reuse existing bounded adversarial coverage: core `kernels/tests.rs` enumerates and shrinks masked cases against an
independent scalar oracle; serialization owner tests reject malformed payloads; CUDA tests exercise driver stubs, cache
concurrency and failure cleanup; cuTile compiler tests retain failure/timeout/cancellation diagnostics. These are
bounded tests. The seeded process-isolated compiler driver in `crates/experimental/tools/kernel_stress.py` adds
replayable vector/matmul generation and malformed source-codec cases with independent scalar oracles. It saves each case
before invoking the real backend and bounds both each process and the campaign. `KERNEL_STRESS.md` describes seed/index
replay and the required native completion marker; a successful process that selected no test is rejected. The
experimental generated vector corpus additionally compiles and executes eight edge extents through each adapter against
an independent scalar oracle. Run existing native fixtures under Compute Sanitizer memcheck/racecheck/synccheck where
supported. `crates/experimental/tools/compare_sanitizer_leaks.py` compares baseline and workload reports; preserve its
exclusions and raw logs. GB10 qualification covers 8192 awaited AOT invocations per adapter under memcheck, with no
additional retention beyond the same six cuDNN initialization allocations in the ordinary PJRT control. Racecheck and
synccheck pass on the 128-invocation fixtures. This bounded evidence does not qualify other architectures or establish
an absolute zero-leak claim.

| Budget | Measurement and admission |
|---|---|
| Compilation | Wall time for clean and incremental builds with exact features; record compiler/tool versions and peak memory when available |
| Binary/artifact size | Exact byte counts and hashes for matching targets/features; distinguish source bytecode, PTX, cubin and executable envelopes |
| Runtime | Completed device execution, explicit warmup/repetition, stable device load and summary statistics; never measure enqueue time as execution |
| Numerical | Independent scalar/interpreter oracle and declared dtype/rounding; exact comparison where promised, justified tolerances otherwise |
| Resources | Report compiler-owned shared-memory bounds as bounds; hardware register/occupancy/counter measurements are unavailable unless actually collected |

A 300-second command limit is an operational safety limit, not a performance regression threshold. Use
`crates/experimental/tools/kernel_baseline.py` to merge independent process reports and compare upper medians against a
budget pinned to the exact baseline SHA256. Semantic, compiler, execution and methodology identities must agree;
malformed, undersampled and unstable runs fail qualification. The accompanying guide specifies warmup, raw sample
retention, provenance and variance review. Provisional policy is declared before measurement and is distinct from
historically calibrated golden thresholds; a same-source comparison demonstrates repeatability, not improvement. No
release-complete claim follows from this inventory while fuzz/stress, long-run, hardware or independent-review gates
remain unqualified. New profiling/AOT/tuning interfaces must retain unavailable fields explicitly and validate their
semantic, compiler and execution identities before reuse.

## Upgrades and migration

Use `crates/ryft-mosaic/tools/check_surface.py` to snapshot and compare clean, pinned upstream GPU surfaces. Every
addition, removal and byte change requires an exact before/after digest and a reviewer-written reason; comments and
implementation changes also require inspection. This is a conservative change detector, not a semantic-equivalence
proof. Re-run wrapper, serialization, target-code and native qualification after accepting source changes.

cuTile now lives in `ryft_cuda::kernels::cutile`; use CUDA features `cutile` for runtime metadata and `cutile-compiler`
for tooling. The shared CUDA artifact and launcher APIs are unchanged. XLA bindings remain in `ryft_xla::kernels`.
There are no compatibility re-export shims. Backend-free consumers continue importing only `ryft_core::kernels`.
