# Ryft Inference: Architecture and Execution Plan

**Status:** proposed architecture

**Research snapshot:** 2026-08-14

**Architecture review:** 2026-09-10; includes a targeted substrate and primary-source refresh

**Scope:** one inference system for agentic/RL rollouts, embedded use, and production online serving

## 1. Executive decision

Ryft should build an inference engine, but it should not build a Rust clone of vLLM and it should not try to turn one
large XLA `while` program into a serving runtime.

The proposed system is a **Rust-native dynamic inference runtime around a family of statically compiled Ryft/XLA
executables**. Rust owns requests, admission, scheduling, page and weight ownership, prefix identity, cancellation,
streaming, and distributed lifecycle. Ryft/XLA owns model computation, sharding, compilation, buffer planning, and
portable fallbacks. Specialized paged-attention, cache-update, sampling, quantized GEMM, MoE, and communication kernels
enter through capability-selected XLA FFI custom calls.

The central abstraction should be a general **paged sequence-state service**, not only a KV cache. It must eventually
hold transformer KV pages, recurrent/Mamba state, multimodal encoder state, speculative checkpoints, and other
model-specific persistent state.

This creates a credible differentiation thesis:

> One typed, compiler-backed execution system spanning model development, training, RL rollout, embedded inference,
> and production serving, with safe Rust resource state machines and workload-adaptive compiled execution profiles.

Treat this thesis as a set of measurable hypotheses. The first architectural gate is a physical execution proof with
real paged-attention kernels, persistent state, graph replay, and overlapped launches. Keep interfaces provisional until
that proof passes, then build a minimal continuous Gemma engine before completing the broader lifecycle framework.
State-of-the-art claims require both controlled numerical comparisons and quality-constrained comparisons against each
baseline's best production configuration, with an attributable win on the declared workload.

It is realistic to build something materially better on selected axes: RL/serving integration, correctness under
concurrency, typed extensibility, agentic prefix reuse, CPU-side latency stability, portable compilation, and selected
model/hardware/workload Pareto frontiers. It is not realistic to assume near-term universal throughput leadership over
vLLM, SGLang, or TensorRT-LLM. Rust is a strong enabler for the host runtime; it is not a substitute for paged state,
excellent GPU kernels, topology-aware communication, mature scheduling, model coverage, or production hardening.

## 2. Goals and non-goals

### Goals

- One model definition and parameter structure usable for training, ordinary inference, RL rollout, and serving.
- An embedded Rust API with no network server or Python process required.
- An optional production server with streaming OpenAI-compatible APIs and a lower-level binary/Rust protocol.
- Continuous batching, chunked prefill, prefix reuse, preemption, cancellation, and explicit SLO-aware scheduling.
- Efficient paged persistent state with safe fork, commit, rollback, pause, resume, eviction, and transfer.
- Aggregated and prefill/decode-disaggregated deployments selected from workload and topology evidence.
- Tensor, pipeline, expert, context, data, and attention parallelism without hand-written communication in each model.
- Capability-selected portable and platform-specific kernels with correctness references and reproducible autotuning.
- Versioned, deterministic, provenance-rich rollouts for synchronous and bounded-staleness agentic RL.
- Quantization, structured generation, speculative decoding, LoRA/adapters, multimodality, and hybrid sequence models.
- Production observability, overload behavior, graceful draining, hot model lifecycle, and failure containment.
- Gemma 4 31B as the first production-model target, with a reduced-shape architecture-equivalent configuration for
  routine correctness tests and the full published checkpoint as the milestone acceptance target.

### Non-goals for the first usable release

- Supporting every Hugging Face architecture, accelerator, quantization, and decoding feature.
- Gemma 4 vision input, MTP speculation, training, and peak 256K-context operation in the first engine milestone. Keep
  their interfaces representable, but validate the text decoder and ordinary autoregressive decoding first.
- Replacing Kubernetes, a cluster scheduler, or an RL training framework.
- Inventing all GPU kernels in Ryft before using proven vendor/community kernels behind stable interfaces.
- Using bounded-dynamic XLA values on the decode hot path where they introduce synchronization or multi-device limits.
- Making Rust or XLA an externally visible requirement of the serving protocol.
- Claiming performance leadership without reproducible SLO-constrained comparisons.

## 3. What the leading systems teach us

Headline benchmark numbers below are not cross-system rankings. They use different dates, models, hardware, request
distributions, feature sets, and SLOs. They are evidence for architectural mechanisms, not a basis for claiming one
universal winner.

| System | Ideas Ryft should adopt | Boundary or caution |
|---|---|---|
| vLLM | Paged KV allocation; continuous iteration-level scheduling; a unified computed-token/required-token model; chunked prefill; prefix caching; preemption; broad APIs and model coverage. Its V1 rewrite also demonstrates the cost of adding features without a coherent scheduler/cache core. | The original PagedAttention paper's 2–4x result is against 2023 systems. Python is still prominent in the control plane, though async scheduling and graphs hide much of that cost. |
| SGLang | RadixAttention; cache-aware scheduling; frontend/runtime co-design for agentic multi-call workloads; grammar compilation and deterministic-span fast forwarding; speculative decoding; overlap; RL refit support. | Cache locality must be balanced against fairness and deadlines. Its largest gains occur on workloads rich in reusable prefixes or structured generation, not universally. |
| SGLang-JAX | JIT-compiled model execution, Pallas kernels, sharding, and a dynamic serving scheduler demonstrate that compiler-backed serving already exists. | Its documented primary platform is TPU; CUDA validation is limited. Use it as an architectural comparison and a baseline where validated, not as evidence that Ryft's CUDA integration is already solved. |
| TensorRT-LLM | The deepest feature stack on NVIDIA: packed in-flight batching, paged/radix KV reuse, priority eviction, host/disk offload, graph buckets, overlap, P/D transfer, TP/PP/EP/attention-DP, low-bit paths, adapters, and many speculative modes. | NVIDIA-specific, integration-heavy, and constrained by a large feature-combination matrix. It proves that feature composition is as hard as individual features. |
| TokenSpeed | Local-SPMD placement annotations; an explicit typed scheduler FSM; strict KV ownership; a clean kernel registry; a first-class long-context agentic workload; aggressive host/device overlap. Its May 2026 preview reports workload-specific wins against TensorRT-LLM on B200. | It is new and explicitly preview-quality. Its execution layer remains mostly Python, and its performance claims are narrow and self-reported. It is a design peer to learn from, not yet a production baseline to assume. |
| NVIDIA Dynamo | Separate request, control, and state/event planes; Rust distributed runtime; KV-aware routing; topology-aware P/D planning; multi-tier state; transfer abstraction; draining and failure-aware orchestration. | Dynamo orchestrates vLLM/SGLang/TensorRT-LLM rather than replacing their model executors. It also shows that Rust alone is not a unique moat. |
| FlashInfer | A serving-specific kernel library with paged/ragged/mixed attention, multiple implementations per operation, JIT artifacts, sampling, quantized GEMM/MoE, standalone numerics, benchmarking, and automatic backend choice. | Ryft needs a stable kernel problem/trait interface and can initially wrap proven implementations; generic XLA fusion alone will not consistently match specialist kernels. |
| DeepSpeed-FastGen | Token-budget scheduling and Dynamic SplitFuse/chunked prefill as a way to combine long prefills with latency-sensitive decode. | Historical comparisons use older vLLM versions. The durable lesson is token-level resource scheduling. |
| MLC-LLM | Compiler-generated model libraries, JIT/AOT packaging, one portable runtime across server and local platforms, and programmable sub-request orchestration. | Strong portability precedent; less complete as a large distributed serving control plane. |
| TGI | Rust request router, explicit router/model-server split, streaming, batching, metrics, and a clean production API boundary. | The project is now in maintenance mode and recommends engines such as vLLM/SGLang for new deployments. |
| llama.cpp | Dependency-light embedding, simple server APIs, broad local hardware support, GGUF, aggressive weight/KV quantization, prompt caching, grammars, and CPU/GPU hybrid placement. | Optimize Ryft's ergonomics against it, but do not copy a local-first execution architecture for cluster serving. |

### Durable conclusions

1. **Scheduling is a resource allocation problem.** Schedule tokens, pages, graph profiles, collectives, adapters,
   deadlines, and predicted cost—not merely request counts or two rigid queues.
2. **State ownership is the heart of the engine.** A request lifecycle and page lifecycle must form one verifiable
   protocol. Cache reuse, preemption, speculative branches, tool pauses, weight changes, and transfer all depend on it.
3. **The hot host path must be small and predictable.** Batch formation, metadata preparation, and completion handling
   cannot repeatedly allocate large structures, run Python, synchronize devices, or rebuild grammars.
4. **Compiled execution must coexist with dynamic admission.** Compile bounded physical profiles; choose and launch one
   at each scheduler tick. Do not place online request admission inside a compiled autoregressive loop.
5. **Kernels are a subsystem, not incidental custom calls.** They need semantic APIs, capability declarations,
   numerics, benchmarking, artifacts, selection, cache identity, graph-safety metadata, and fallback behavior.
6. **Aggregated and disaggregated serving are workload choices.** P/D separation can improve SLO goodput, but transfer
   overhead makes it worse for some models, prompts, traffic levels, and fabrics.
7. **Agentic and RL workloads deserve first-class semantics.** Pause/resume, forks, long prefixes, tool latency,
   deterministic sampling, logprobs, partial rollouts, rapid weight refits, and version provenance are core operations.
8. **Performance is a Pareto surface.** Report TTFT, TPOT/ITL, E2E latency, goodput, tokens/GPU, memory, energy, and
   cost under an explicit trace and SLO—not one peak tokens/second number.

## 4. Ryft today: unusually strong substrate, missing inference runtime

The repository audit found that Ryft is much closer at the compiler/device layer than a greenfield inference project,
but has almost none of the serving control plane.

### Existing foundations to reuse

| Capability | Current evidence | Inference use |
|---|---|---|
| Typed, backend-neutral programs and transformations | `crates/ryft-core/src/contexts.rs`, `crates/ryft-core/src/programs/programs.rs`, `crates/ryft-core/src/parameters.rs` | Shared model/state code, structured weights, reference execution, JIT, vmap, autodiff, checkpoint mapping. |
| Broad inference primitives | `crates/ryft-core/src/operations`, `.tasks/plan_tier1_inference_primitives.md`, `.tasks/plan_inference_completeness.md` | Dot/quantized dot, control flow, RNG, sampling building blocks, dynamic slice/update, fused attention, custom calls, collectives. |
| Real decode expressibility | Tiny greedy/top-k/custom-attention compiled decode-loop tests in `crates/ryft-xla/src/jit.rs` | Excellent correctness fixture and useful fixed-horizon RL path; not a continuous server. |
| Retained and persistent compilation | `crates/ryft-core/src/compilation`, `crates/ryft-xla/src/experimental/domains.rs` | Precompiled profile families, single-flight compilation, AOT warmup, validated executable restore, distributed artifact exchange. |
| Static and bounded symbolic dimensions | `ryft-core` array dimensions plus `XlaInputBoundBucketing` | Prefill profile bucketing and model polymorphism, subject to the current dynamic-boundary restrictions. |
| Async PJRT execution | `crates/ryft-pjrt/src/events.rs`, `crates/ryft-pjrt/src/programs.rs`; `CompiledXlaFunction::interpret_async` | Enqueue model steps without waiting, chain pending buffers, attach completion fences. |
| Async external-reference execution | `crates/ryft-xla/src/jit.rs`; `call_statefully_async` | Existing read leases, pending mutation generations, and completion failure handling inform the serving protocol; they do not guarantee physical in-place state. |
| Device-buffer lifecycle and interop | `crates/ryft-pjrt/src/buffers.rs`, `crates/ryft-pjrt/src/transfers.rs` | Uninitialized buffers, donation, aliases, DMA/pinned staging, external device pointers, zero-copy interop, asynchronous transfers. |
| Stateful custom-kernel escape hatch | typed XLA FFI, GPU custom calls, Triton extension, Mosaic GPU bindings | Paged attention, cache writes, fused sampling, quantized kernels, MoE, custom communication. |
| Explicit custom-call aliasing/effects | `crates/ryft-core/src/operations/custom_call.rs` and XLA lowering | In-place serving state transitions with compiler-visible output/input aliases. |
| Dense fused attention | `crates/ryft-core/src/operations/attention`; `crates/ryft-xla/src/experimental/lowering/attention.rs` | Correct portable reference plus fast CUDA dense prefill/training path with GQA, masks, windows, bias, dropout, and backward. |
| Sharding and collectives | logical meshes, `shard_map`, all-reduce/gather/scatter/all-to-all/permute | Tensor and other SPMD parallel execution without model-specific communication code. |
| Multi-host substrate | PJRT distributed runtime, cross-host transfers, topology, stream extension | Worker-group startup, rank-local buffers, future P/D state transfers and artifact distribution. |
| Profiling and adaptive recompilation | PJRT profiler, executable analysis, `crates/ryft-xla/src/profile_guided.rs` | Observe real profiles, produce compatible optimized replacements, and atomically install them. |

### Important current constraints

- `Array` is a functional distributed value, not a durable mutable resource arena.
- Dense attention explicitly emits `is_paged_attention: false`; no block table or paged kernel exists.
- No model/checkpoint/tokenizer crate exists, and model plan files remain prospective.
- No request scheduler, page allocator, prefix index, sampler runtime, grammar engine, adapter manager, or server exists.
- XLA bounded-dynamic input/output boundaries can synchronize with the host, and retained bound bucketing is currently
  single-device because the dynamic path is incompatible with the Shardy path.
- `CompiledXlaFunction::batch` is currently unsupported. Serving batches must be assembled explicitly and dispatched to
  an already compiled profile; the design must not assume post-compilation vmap.
- Requested donation may be downgraded when a buffer has another shared owner. Donation can remain an ordinary tensor
  optimization, but it must not be the correctness mechanism for shared sequence state.
- The current external-reference ABI explicitly uses non-semantic may-alias hints and never donates reference-state
  inputs. It commits returned final values and dependency-chains mutations to the same reference. This preserves
  functional snapshots, but does not establish stable arena addresses, absence of pool-sized copies, or independence
  of disjoint pages. Its admitted mutable boundary is static state on a fully addressable single-process mesh.
- The high-level XLA execution wrapper does not yet expose the low-level launch IDs, incarnations, callbacks, execution
  contexts, device overrides, or multi-slice configuration needed for robust serving-oriented multi-host launches.
- Existing telemetry is compiler/array-oriented, not request/page/SLO-oriented.

### Architectural consequence

Do not overload `ryft-core::Array` or move a KV allocator into the mathematical IR. Add an inference-owned resource
layer. Its public types represent safe leases and state transitions; its XLA implementation uses stable PJRT buffers,
explicit FFI aliases, completion fences, and device-resident metadata internally.

## 5. Target system architecture

```text
                           ┌──────────────────────────────────────────┐
                           │ User surfaces                            │
                           │ Rust API · Python binding · OpenAI API   │
                           └───────────────────┬──────────────────────┘
                                               │ typed requests/streams
                    ┌──────────────────────────▼──────────────────────────┐
                    │ Inference engine (Rust)                             │
                    │ validation · tokenize · admission · lifecycle       │
                    │ scheduler · sampler/grammar · cancellation · output │
                    └──────────┬──────────────────────┬────────────────────┘
                               │ plans                 │ leases/transactions
                    ┌──────────▼─────────┐   ┌────────▼───────────────────┐
                    │ Execution worker   │   │ Paged sequence-state svc   │
                    │ profile selection  │   │ pools · tables · radix     │
                    │ async launch       │   │ COW · fork · rollback      │
                    │ weights/adapters   │   │ eviction · tiers · transfer│
                    └──────────┬─────────┘   └────────┬───────────────────┘
                               │ compiled calls        │ device buffers/tables
                    ┌──────────▼───────────────────────▼──────────────────┐
                    │ Ryft/XLA execution                                 │
                    │ prefill · decode · verify · draft · score · embed  │
                    │ static physical profiles · sharding · collectives  │
                    └──────────┬──────────────────────┬───────────────────┘
                               │ semantic kernel calls │ portable StableHLO
                    ┌──────────▼──────────────────────▼──────────────────┐
                    │ Kernel providers                                   │
                    │ vendor libs · FlashInfer · Triton · Mosaic · ref   │
                    └───────────────────┬────────────────────────────────┘
                                        │ PJRT / CUDA / ROCm / TPU / CPU
                    ┌───────────────────▼────────────────────────────────┐
                    │ Devices, memory spaces, streams, network           │
                    └────────────────────────────────────────────────────┘

       Distributed deployments add a separate router/control/state-event plane above worker groups.
```

### Planes and ownership

| Plane | Owns | Must not own |
|---|---|---|
| Request plane | Protocol normalization, tokenization, streaming, deadlines, quotas, backpressure | GPU page allocation or model graph details |
| Scheduling plane | Request state machine, token/resource budgets, profile choice, preemption policy | Physical kernel implementation |
| Sequence-state plane | Page lifecycle, prefix identity, state transactions, tiers, transfers | Request routing policy or text APIs |
| Execution plane | Resident weights, executable profiles, metadata upload, async launches, completion | Global service discovery or tenant policy |
| Kernel plane | Operation implementations, capability checks, artifacts, tuning, numerics | Request lifecycle or cache eviction |
| Distributed control plane | Worker membership, health, placement, routing, scaling, draining, failure recovery | Rank-local page allocation or token sampling |

This separation prevents the common failure mode where one scheduler class becomes the allocator, radix tree, transfer
manager, batch builder, model runner, and distributed router simultaneously.

## 6. Proposed crate boundaries

The following names describe logical ownership boundaries, not six crates to create immediately. Start with
`ryft-models` and an embedded `ryft-inference` crate whose core, XLA executor, and kernel integration are internal
modules. Extract the proposed crates only when optional dependencies, independent consumers, or provider distribution
justify the split. Keep internal Rust interfaces evolvable; version the external provider ABI and persisted artifact
formats that actually need independent compatibility.

### `ryft-models`

- Model configurations and `Parameterized` weight/state schemas.
- Reusable NN layers and architecture implementations.
- Stable parameter names and SafeTensors/Hugging Face import mapping.
- `prefill`, `decode`, `score`, `embed`, and optional training semantics expressed with Ryft operations.
- No scheduler, page allocator, HTTP types, or platform-specific kernel target names.

### One model implementation across execution modes

Keep configuration, parameter names, layer order, position rules, masking, normalization, and logits semantics in one
model implementation. Express attention and sequence-state access through small semantic capabilities that distinguish
full-sequence evaluation, state reads, and state advancement. Training and contiguous-reference contexts provide
differentiable or dense implementations; serving contexts lower the same model calls to paged or recurrent providers.
Serving-specific lowering may fuse operations and choose layouts without duplicating architecture logic.

Model code declares state-group requirements; the executor supplies validated bindings and leases. Model code never
allocates pages or chooses a kernel target. Prefill, mixed extension, decode, and score wrappers adapt inputs to this
shared implementation. Test reduced-shape full-sequence versus contiguous and paged execution under the numerical
contract, then validate the shared training/score path in the early rollout milestone. A serving-only provider need
not have a backward kernel when training uses the same semantic operation's differentiable implementation.

### `ryft-inference-core`

- Backend-neutral request, sequence, sampling, output, model-version, adapter, and SLO types.
- Authoritative request state machine and scheduler/resource traits.
- Logical sequence-state interfaces and transactional contracts.
- A deterministic pure-Rust conformance executor and virtual clock behind test-support features. They implement the
  production executor contracts and provide controllable costs, completions, failures, and state transitions.
- No `ryft_xla::Array` or HTTP types in policy-facing public APIs.

### `ryft-inference-xla`

- `CompiledModel`, resident weights, XLA sharding plans, and executable-profile families.
- Prefill/decode/verify/draft/score/embed program construction and persistent warmup.
- Async PJRT launch integration, device-resident metadata, fences, and error propagation.
- Kernel-provider registration and XLA custom-call lowering integration.
- XLA implementation of sequence-state buffers without exposing raw buffers to policy code.
- A tiny XLA conformance model that uses the production physical ABI, aliases, buffers, profiles, and completion path
  without requiring Gemma 4 kernels or weights.

### `ryft-kernels`

- Stable semantic operation requests and versioned FFI ABIs.
- Provider registry, capability declarations, artifact loading, selection, tuning, and fallback chain.
- Reference numerics and standalone correctness/performance harnesses.
- Initially wrap proven libraries/vendor calls; add Ryft-native Triton/Mosaic/CUDA/ROCm kernels selectively.

### `ryft-inference`

- Embedded engine joining tokenizer adapters, scheduler, state manager, executor, grammar/sampler, and output streams.
- Sync batch API plus async per-request streaming API.
- RL rollout facade and local/embedded mode.

### `ryft-serving`

- Optional service dependencies: OpenAI-compatible HTTP, gRPC/binary protocol, auth, quotas, health, metrics,
  readiness, graceful drain, model and adapter administration.
- Distributed router/controller may begin here but should remain separable from the embedded engine.

The page allocator can begin as an internal module of `ryft-inference-core`/`ryft-inference-xla`. Split a `ryft-state`
crate only when tiered storage or multiple executors genuinely require an independent dependency boundary.

## 7. Core domain model

### Request state machine

Every request has one authoritative state. Representative transitions are:

```text
admitted -> queued -> prefilling -> decoding -> completed
                         │             │
                         ├-> paused_tool <-┤
                         ├-> transferring -> decoding
                         ├-> preempted -> queued
                         └-> aborted/failed
```

State transitions must be explicit methods returning owned transition results. They update resource leases and emit
events atomically. Invalid transitions are type/domain errors, not ignored flags.

Each request records at least:

- request, tenant, trace, and optional conversation IDs;
- model, adapter, and immutable weight-version IDs;
- input tokens, generated tokens, required/completed/in-flight frontiers, accepted/published boundaries, and maximum
  output;
- sampling/grammar/speculation state and deterministic RNG stream;
- owned/shared sequence-state blocks and optional speculative checkpoint;
- deadline, priority, SLO class, enqueue time, and fairness accounting;
- chosen worker/profile/topology and transfer/refit state;
- per-token logprob and policy-version provenance when requested.

### Versioned model state

`WeightVersion` is monotonically increasing within a model lineage and identifies an immutable snapshot. A request
binds to one version. Bounded staleness permits an older pinned version within a declared lag limit; it does not permit
changing weights mid-sequence while retaining state produced by the old weights. Moving a continuation to a new version
requires rebuilding its state from tokens and recording the policy boundary. Sequence-state cache keys include the
model lineage and weight version; the default APIs make reuse across weight updates impossible.

A weight update is transactional:

1. prepare and validate a complete sharded snapshot;
2. transfer it using disk, host, IPC, collective, or RDMA provider;
3. make the snapshot executable-ready on every rank;
4. atomically publish the new version for new requests;
5. drain or retain old-version requests according to policy;
6. reclaim the old snapshot only after all requests and asynchronous launches release it.

Use double buffering first where the declared memory envelope supports it. Approximately 31 billion BF16 parameters
require 62 GB for one snapshot and 124 GB for two, before state, workspaces, activations, and transfer staging. These
are decimal estimates, to be replaced by the pinned checkpoint's measured allocation sizes. Budget old-version
state and paused requests as well as weights. If this does not fit, bring tensor parallelism forward or explicitly
measure a drain/offload/load policy and its update pause. Optimize to delta/LoRA or in-place refit only with explicit
correctness and memory evidence.

## 8. Paged sequence-state service

### Why it is not just `KvCache`

Modern workloads include MHA/GQA/MLA KV, sliding windows, recurrent state, Mamba/linear attention, multimodal encoder
outputs, speculative branches, and tool-paused sessions. The allocator and prefix index should be generic over a
`StateClass`, while model-specific kernels interpret the bytes/layout. A common resource protocol does not require
every state class to have token-addressed pages or cheap arbitrary rollback.

### State groups and capabilities

A sequence owns a bundle of state groups with explicit semantics:

- append-only history, bounded sliding-window history, overwritten recurrent state, or immutable encoder state;
- retention horizon, valid checkpoint boundaries, and dependencies on other groups, including shared KV producers;
- supported fork/snapshot/restore/replay operations and their memory, transfer, and recomputation costs;
- the token frontier represented by each retained checkpoint and the conditions for safe prefix publication.

Transformer KV can often truncate unpublished append-only writes. A recurrent accumulator instead needs a saved state,
intermediate verification states, or replay from a trusted checkpoint. Accepting three of eight speculative tokens must
produce the state after token three in every group; changing a page-table length alone cannot restore an overwritten
accumulator. Providers advertise which checkpoint strategies they support, and the scheduler budgets their actual cost.

Prefix eligibility is the latest common token boundary reconstructible across all required groups. Sliding-window
retention and recurrent checkpoints must not falsely advertise an arbitrary historical prefix. Test these contracts
early with append-only, sliding-window, and tiny recurrent conformance state; production Qwen kernels can remain later.

### Main types

- `StatePool`: one physical allocation class on one device/memory tier.
- `StatePageId`: stable logical identifier, never a raw pointer in scheduler code.
- `StateLayout`: model/layer group, allocation granularity (including tokens per page where applicable), dtype,
  dimensions, sharding, and memory-space identity.
- `SequenceTable`: mappings and checkpoint frontiers for the state groups of one sequence; token positions map to pages
  only for state classes that support that representation.
- `PageLease`: shared or exclusive ownership of allocated pages or fixed state slots tied to completion dependencies.
- `SequenceCheckpoint`: a coherent boundary across state groups, with an explicit snapshot/restore/replay strategy and
  cost for speculative decoding and branching agents.
- `PrefixKey`: hash over tokens plus model, weight version, adapter, position policy, state layout, and relevant model
  configuration.
- `StateTransaction`: a resource transition with separate reservation, submission, completion, publication, and
  reclamation states, covering append, fork, truncate, restore, transfer, and eviction.

### Required invariants

- A physical page is never reused while a compute or transfer fence may still access it.
- A shared page is immutable; partial-page continuation requires copy-on-write.
- Prefix publication happens only after the producing launch completes successfully.
- Aborting before submission releases reservations. Aborting after submission suppresses results and defers resource
  reclamation until all compute and transfer accesses have ceased; it cannot undo an in-flight write.
- Page-table updates become visible atomically at the scheduler/executor boundary.
- Each launch retains an immutable metadata snapshot or a uniquely leased metadata-buffer slot until its consumers
  finish. Host edits and buffer recycling cannot alter metadata observed by an outstanding launch.
- Weight/model/layout mismatches make a prefix ineligible, even when token hashes match.
- An asynchronous failure after possible writes poisons affected state unless it can be reconstructed from a trusted
  checkpoint. Unknown device quiescence requires quarantine or worker recovery, never immediate page reuse.
- Every allocation, reference, transfer, and eviction can be reconstructed from a deterministic event log in tests.

### Storage tiers

Start with HBM only. Add pinned host memory after eviction and transfer metrics exist, then remote DRAM/NVMe only for a
validated workload. The interface supports async `store`, `load`, `transfer`, and `cancel`, but policy decides whether
the expected reuse value exceeds the transfer and capacity cost.

Use a radix tree or block-hash tree only as an index. Keep it separate from physical allocation and tier I/O. This makes
the index replaceable and avoids the large coupled cache-manager classes now being split in mature engines.

## 9. Scheduler design

### Unified work model

Adopt vLLM V1's required/computed-token work model, extended with explicit state, cost, and asynchronous progress.
Track required work, reserved/submitted ranges with dependencies, successfully completed computation, and the accepted
and published boundaries. Prefill, decode, prefix hits, speculative verification, and deterministic grammar spans
advance these boundaries differently. Draft proposals are candidates, not accepted target-model state. Never schedule
a range twice because its submitted computation has not yet completed.

The scheduler produces a `StepPlan` containing:

- admitted requests and token ranges;
- selected execution phase/profile, per-sequence query lengths, and active slot count, including mixed prefill/decode;
- page reservations, block tables, slot mappings, lengths, masks, and state checkpoints;
- model/adapter/weight version and distributed worker group;
- sampling/grammar/speculation actions;
- expected resource usage, launch dependencies and sequence generations, and completion/publication/reclamation actions.

The successful plan lifecycle is `reserved -> submitted -> completed -> published`. Submission acceptance transfers
reservations into in-flight leases; it does not commit successful model state. Completion checks execution status and
required semantic/acceptance results before publication. Numerical reference comparisons belong in parity tests or
explicit diagnostics, not a per-token host readback. Publication retains ownership of committed state; reclamation
occurs separately after the last owner and fence release it. Cancelled or failed work bypasses publication and follows
the appropriate deferred reclamation or quarantine path. A rejected submission can roll back immediately only when
no device access was started. An uncertain or partially submitted launch follows the in-flight failure path.

Cancellation before submission releases reservations. After submission it suppresses publication and further work,
but retains leases through all dependent launches and a terminal fence proving access has ceased. Unpublished append
regions can then be discarded; overwritten state requires restore/replay or poisoning. Streaming outputs become
externally visible only after acceptance and successful completion; generation IDs reject stale callbacks after slot
reuse. Tests cover cancellation after enqueue, partial writes, out-of-order callbacks, and unknown device quiescence.

### Policy layers

1. **Capacity admission:** Can the complete or safely preemptible work fit in page, weight, adapter, graph, and transfer
   budgets without thrashing?
2. **SLO priority:** Deadline/slack, queue age, priority, tenant quota, and predicted TTFT/TPOT risk.
3. **Locality:** Reusable prefix/state, resident adapter/weights, and topology/transfer cost.
4. **Batch efficiency:** Compatible model/profile/kernel/grammar/speculation shapes and predicted execution time.
5. **Fairness:** Bounded starvation regardless of locality and long prompt/output behavior.

Policies implement a small trait over immutable scheduler snapshots. The transition engine and resource invariants are
not replaceable. Record every decision so a trace can be replayed deterministically.

### Policy and numerical configuration

- `Latency`: decode-biased, tight batch wait, SLO slack first.
- `Throughput`: fuller batches and larger chunks for offline generation/RL.
- `Agentic`: high value on prefix/session retention and pause/resume.
- Custom policies may adjust scoring, but not bypass ownership or version rules.

Numerical reproducibility is configured independently of scheduling policy. Replay, batch-invariant execution, and
on-policy provenance are distinct contracts described in section 12; a latency or throughput policy can request any
supported numerical contract. Do not force all on-policy work into a slower deterministic scheduler mode.

## 10. Execution profiles and XLA strategy

### Host-driven steps

Compile distinct semantic entry points:

- `prefill`: one or more prompt chunks, producing state updates and optional first-token logits;
- `decode`: one token per active sequence (or a small deterministic span);
- `extend`: mixed prompt chunks and decode tokens with per-sequence query lengths in a packed model forward pass;
- `verify`: validate speculative token chains/trees and produce accepted-frontier and state-restoration results;
- `draft`: generate candidate tokens for model-based speculation;
- `score`: logprobs/reward/reference-policy evaluation;
- `embed`/encoder stages for multimodal and encoder-decoder models.

The online scheduler selects compatible profiles for each step and may prepare subsequent steps while earlier ones
execute. Mixed `extend` shares packed dense computation and may dispatch attention separately for prefill and decode
subsets. Retain specialized homogeneous profiles where measurements favor them. Alternating prefill-only and
decode-only launches is an explicit alternative to benchmark, not a substitute for representing mixed execution.
Executables produce state updates; the host transaction protocol controls their successful publication.

A compiled multi-token loop remains useful for fixed homogeneous offline rollouts, benchmarks, and some local
workloads, but it is an optimization outside the scheduler's correctness boundary.

### Overlapped execution protocol

Build the initial executor to support bounded lookahead rather than awaiting every step before preparing the next:

1. Reserve ranges and metadata slots using completed state plus explicit dependencies on in-flight work.
2. Enqueue metadata transfer and model execution with device-side dependencies. Keep sampled next-token IDs on device
   and represent unavailable token values as future references when constructing dependent steps.
3. Prepare compatible subsequent work on the host while computation runs. Dependent launches may consume in-flight
   state through explicit device dependencies; do not publish uncompleted results or recycle still-leased storage.
4. Deliver token/status readback asynchronously to a completion reactor, then accept outputs, update frontiers,
   process stop conditions, and retire leases. Bound unnecessary lookahead after EOS or cancellation.

Specify launch ordering, sequence generations, ring-buffer reuse, and backpressure for metadata and output slots.
Stop/grammar/speculation decisions that need host results create explicit dependencies; they must not become hidden
global barriers. Fixed output capacity does not imply that valid output counts or acceptance lengths are host-known.
No implicit shape readback, full-logit materialization, or blocking host wait is allowed on the steady-state enqueue
path. Necessary token/status delivery remains asynchronous and is charged to end-to-end latency.

Measure host scheduling, metadata preparation, transfer, GPU execution, readback, and GPU idle gaps separately. Compare
lookahead enabled/disabled on the same traces and prove completion-safe reclamation in both modes. Merely returning
an execution future does not satisfy this gate.

### Static physical ABI

Each executable receives fixed-capacity physical buffers plus device-resident logical metadata:

- packed input token IDs or device future-token references, positions, and per-sequence query offsets/lengths;
- active slot mask and per-sequence lengths;
- page/block tables and append slot mappings;
- state pool buffers (explicitly aliased where mutated);
- resident weight/adaptor buffers or stable handles;
- RNG counters, grammar masks/state, and sampling parameters;
- optional speculative tree/acceptance metadata.

Inactive lanes have defined semantics and never mutate state or emit tokens. Output buffers have host-known fixed
capacity with device-resident valid counts, acceptance lengths, and status; these can be consumed by dependent device
work or read asynchronously. This avoids shape-dependent allocation and preserves graph/profile reuse without assuming
speculative output cardinality is known before execution. Compile profile keys include model architecture,
weight format, state layout, phase, token and sequence buckets, topology/sharding, kernel selections, speculation mode,
determinism mode, compiler identity, and relevant XLA flags.

Start with a small geometric profile family. Collect misses and padding waste, then tune bucket boundaries from traces.
Do not compile an unbounded Cartesian product.

### Command buffers/graphs

XLA already performs command-buffer/CUDA-graph extraction for supported thunks. Ryft must verify, per profile, that the
whole step—including custom calls, collectives, cache mutation, and sampler—is capture-safe. The kernel registry records
graph safety and stable workspace requirements. A profile that cannot capture remains correct but is not accepted for a
latency target until measured.

## 11. Kernel subsystem

### Semantic API

Kernel calls describe problems rather than implementations:

- paged prefill/append/decode/verify attention;
- MLA, sliding/sparse/hybrid attention;
- state gather/scatter/append/copy-on-write;
- dense and quantized GEMM; normalization/RoPE/residual fusion;
- MoE route, dispatch, grouped GEMM, combine, and load balancing;
- logits processing, grammar masking, top-k/top-p/min-p sampling, and speculative acceptance;
- fused collectives and collective epilogues.

### Provider contract

Each provider declares:

- platform and architecture range;
- dtype/quantization/layout/page-size support;
- head dimensions, GQA ratio, shape/profile ranges, and attention mode;
- deterministic and batch-invariant behavior;
- graph/capture safety and stable workspace needs;
- required runtime libraries and artifact ABI/version;
- numerical tolerance and reference implementation;
- cost observations or an autotuning key.

Selection occurs at profile compile/warmup time, not in the token hot path. The chosen provider and version are part of
executable cache identity. Override and deny-list mechanisms are required for rollout, debugging, and regressions.

Separate provider selection from per-batch kernel planning. A selected provider may still need ragged scheduling
metadata, split choices, or workspace initialization based on actual lengths. Specify that work, its host/device
placement, capture compatibility, and cost explicitly; compile-time selection must not conceal a synchronizing
`plan` call on every token. Validate reuse across length distributions and profile changes with the actual provider.

### Fallback order

1. validated platform-specialized implementation;
2. portable serving kernel (Triton/Mosaic or community provider);
3. correct StableHLO/Ryft composition for unsupported/small/test cases;
4. clear unsupported error when a semantic operation cannot be represented correctly.

Every kernel family ships with standalone generated cases, adversarial cases, reference comparison, profiler scopes,
and benchmark replay from captured serving shapes.

## 12. Sampling, grammars, and speculation

### Sampling

The production sampler supports greedy, temperature, top-k, top-p, min-p, typical sampling as needed, repetition and
presence/frequency penalties, bad/stop tokens, logprobs, and deterministic counter-based RNG. Use CPU implementations
and existing Ryft primitives as correctness references. The competitive path samples on device and returns token IDs,
requested statistics, and status asynchronously; it must not read full vocabulary logits back each step. Start with
compiled sampling primitives and add fused kernels where measurement justifies them.

Define the exact order of logits processors, normalization, sampling, stop handling, and grammar updates. Distinguish
raw model logprobs from behavior-distribution logprobs after temperature, penalties, truncation, and constraints; label
returned values and record the transforms used. RL consumers choose the probability semantics their objective needs.
Speculative sampling declares its acceptance/correction algorithm and whether it preserves the requested target
distribution; distribution-changing acceleration must be an explicit alternative policy.

### Numerical and provenance contracts

- **Replay reproducibility:** repeat a request with a recorded RNG stream and execution configuration on a declared
  runtime/hardware scope. Specify the RNG counter mapping for sequence branches, token positions, and proposal draws
  so cancellation and rejection do not silently perturb unrelated requests.
- **Batch invariance:** invariance to batch composition and supported profile choices requires validated kernels and
  reduction strategies. Advertise the supported matrix and performance cost separately from replay reproducibility.
- **Numerical agreement:** define tolerances for layers, logits, and both logprob forms across training, reference,
  contiguous, paged, and quantized paths. Require exact greedy tokens only on suitable declared fixtures; use error
  bounds and distribution tests where floating-point differences can change near-tied or sampled choices.
- **Policy provenance:** identify the immutable policy snapshot and actual behavior distribution that produced each
  accepted token. Synchronous on-policy execution uses version barriers; bounded-staleness asynchronous execution
  declares its lag and the trainer's acceptance/correction policy rather than assuming stale data is on-policy.
  On-policy correctness does not itself require bitwise identity, stable scheduling order, or batch-invariant execution.

Record weight version, RNG stream, sampling transforms, numerical mode, kernel/profile identity, and relevant compiler
identity. Do not promise bitwise equivalence across arbitrary providers, topologies, or compiler revisions. Training
and rollout parity gates validate the declared numerical and probability contract, not unexplained token equality.

### Structured output

Compile JSON Schema/regex/EBNF into cached automata. The request carries only automaton state. Token masks should move to
the device once CPU masking becomes visible. Deterministic spans may fast-forward multiple tokens, but the scheduler
accounts for their state/pages exactly like speculative work.

### Speculation

Use a protocol, not one draft-model flag:

- proposer: n-gram, prompt lookup, draft model, Medusa/EAGLE/MTP, or user tokens;
- proposal shape: chain or tree;
- verifier: target-model acceptance policy;
- state checkpoint: fork, commit accepted prefix, roll back remainder;
- adaptive controller: enablement and depth from acceptance, concurrency, and memory cost.

Speculation is allowed to turn itself off. It often helps low-concurrency latency and can hurt saturated throughput.

## 13. RL and agentic integration

### Rollout API

Expose a transport-free `RolloutEngine` over the same scheduler/executor:

- async single/multi-turn generation and fixed batch generation;
- pause/resume around tools without discarding useful state by default;
- partial rollouts and continuation IDs;
- forks for best-of-N/tree search using copy-on-write or the state group's declared snapshot/replay strategy;
- token IDs, masks, per-token logprobs, entropy/auxiliary outputs, finish reasons, and version provenance;
- explicit replay/batch-invariance settings and bounded-staleness asynchronous policy;
- abort, retract, checkpoint, and resume.

Agent loops remain outside the model executor. They can be Rust, Python, or remote clients and should not inject tracing
or tool latency into the GPU scheduler process.

### Colocated training and inference

Support two deployment modes:

- **Dedicated:** trainer and rollout/serving workers use separate devices; transfer versioned snapshots.
- **Colocated/time-shared:** inference can sleep/offload/release weights and state, then wake after training. Placement
  and role transitions are coordinated above the engine.

Define synchronous on-policy barriers first. Add bounded-staleness asynchronous training only after every token can be
attributed to a policy version and stale sequence state is rejected across refits.

The strongest Ryft opportunity is sharing model definitions, parameter trees, sharding, and numerical tests between the
trainer and rollout engine while retaining inference-specific compiled profiles and kernels.

### Weight churn and session retention

Frequent updates invalidate prefix state for new-version requests. Keeping an old version alive preserves its session
prefixes but consumes weight/state memory and can violate the allowed lag. Define a bounded retention policy for old
snapshots and paused sessions, including expiry, re-admission, and explicit reconstruction of a continuation on a new
version. Training integration must account for rollout tokens produced under each retained policy.

Measure update cadence, tool-pause duration, prefix reuse distance, concurrent versions, cache reconstruction, and
useful completed rollouts in the same experiment. Report time from trainer snapshot availability to accepted rollout
completion and end-to-end agent latency, including transfer, publication, cold-prefix rebuild, and drain costs. Compare
retention enabled/disabled under an identical memory and staleness budget. Prefix reuse and rapid refits are competing
resource choices whose combined benefit must be demonstrated.

## 14. Distributed architecture

### Worker groups

A worker group is a fault domain executing one SPMD model/profile over a fixed topology. It owns rank-local weights,
state pools, executables, and a local scheduler. PJRT/XLA handles the compiled collective computation inside the group.

The external control plane owns membership, discovery, health, placement, rolling updates, draining, and retries. Do not
use the existing ordered XLA artifact-exchange protocol as a service-discovery or fault-recovery protocol.

### Routing

Route with a scored combination of:

- model/adapter/weight readiness;
- reusable prefix/state location;
- queue and page pressure;
- predicted TTFT/TPOT slack;
- topology and transfer cost;
- tenant/fairness policy.

Publish block/state events asynchronously. The router's global index is advisory; the selected worker validates leases
and capacity before admission.

### Aggregated and P/D modes

Worker roles are capabilities: aggregated, prefill, decode, draft, score, or hybrid. An offline simulator/profiler and
online planner choose mode, parallelism, and pool ratios. P/D handoff uses an opaque versioned `StateSessionRef`; the
source owns it until the destination acknowledges attach, and every timeout/failure path has deterministic cleanup.

Begin with aggregated single-node and multi-GPU execution. Add disaggregation only after a measured transfer path and a
workload demonstrate better SLO goodput.

## 15. Production surface and observability

### Service behavior

- OpenAI-compatible chat/completions/responses where practical, plus embeddings and tokenization.
- Streaming with bounded queues and cancellation propagation.
- Admission control, deadlines, priorities, tenant quotas, rate limits, and load shedding.
- Health/liveness/readiness, graceful drain, rolling model/adapter update, and reproducible configuration snapshots.
- Tokenizer/chat-template adapters are CPU components and can scale separately from GPU workers.
- Python bindings wrap the Rust API; they do not become an internal execution plane.

### Metrics and traces

At minimum:

- TTFT, TPOT/ITL, E2E, queue time, goodput/SLO attainment, and tokens/GPU;
- prompt/generated/accepted/rejected speculative tokens;
- prefill/decode batch occupancy, token budget, padding, and profile hit/miss;
- page capacity/utilization, prefix hit, COW, eviction, preemption, offload, recall, and transfer;
- scheduler decision reason and predicted/actual step cost;
- kernel/provider/profile selection and launch duration;
- collective/network time, weight/refit duration, and version drain time;
- compilation/cache/PGO metrics and errors;
- cancellation, overload, deadline miss, retry, and failure counts;
- optional energy and cost accounting.

Use Prometheus-compatible metrics and OpenTelemetry traces. Correlate request, scheduler step, PJRT launch, kernel, and
transfer IDs. Provide a deterministic flight recorder that captures scheduler inputs/decisions without copying model
payloads.

## 16. XLA/PJRT work required

The underlying PJRT layer already implements many useful low-level capabilities. Reuse them, but treat serving arena
ownership, overlapped execution, and failure-safe state publication as new contracts that require physical proof.

### P0: physical feasibility and first competitive engine

- [ ] Prototype the smallest semantic kernel/provider ABI needed for the feasibility probe; stabilize only after the
      actual provider and state boundary pass. Version independently distributed providers and persisted artifacts.
- [ ] Add paged sequence-state pool/table metadata and aliased cache-write/paged-attention operations.
- [ ] Integrate at least one CUDA paged-attention implementation and one correctness fallback; define the ROCm path.
- [ ] Guarantee explicit exclusive state-buffer ownership. Detect rather than silently copy when an in-place serving
      contract cannot be honored.
- [ ] Validate ownership granularity with outstanding launches on disjoint pages of one pool; detect whole-buffer
      serialization, establish supported dependencies, and document any deliberate serialization and its cost.
- [ ] Add static physical prefill/decode/mixed profile construction with device-resident query lengths, active masks,
      page tables, and slot mappings.
- [ ] Keep the steady-state enqueue path free of shape readback, full-logit host materialization, and implicit waits;
      use asynchronous token/status delivery and device future-token dependencies.
- [ ] Verify whole-step command-buffer capture/replay and stable workspace behavior for chosen profiles.
- [ ] Compose awaitable/callback-driven completion fences, immutable per-launch metadata slots, sequence generations,
      and bounded cancellation points through the high-level executor. Retain leases until device access has ceased.
- [ ] Prove repeated calls and profile changes preserve arena addresses without pool-sized copies; test partial failure
      and cancellation without publication or early reuse of affected state.
- [ ] Surface rank-consistent launch IDs, device selection, and error handling before the first multi-GPU gate whenever
      the initial model or weight-update memory envelope requires tensor parallelism.

### P1: production scale

- [ ] Extend the proven local execution protocol with distributed execution contexts, worker incarnation IDs, and
      rank-consistent recovery; callbacks and completion-safe cancellation are already P0 prerequisites.
- [ ] Complete multi-slice execution configuration and rank-consistent error propagation.
- [ ] Build sharded hot weight replacement and adapter updates without recompiling unchanged executable profiles.
- [ ] Add fused/page-aware quantized KV, low-bit GEMM, MoE, sampling, and collective providers.
- [ ] Add layerwise/streaming state transfer with compute/communication overlap for P/D deployments.
- [ ] Extend or avoid multi-device bounded bucketing based on evidence; fixed physical profiles remain the default.
- [ ] Expose per-operation/profile tracing and scheduler-consumable cost observations.

### P2: differentiation

- [ ] Capability-parity plans for CUDA, ROCm, TPU, CPU, Metal/other PJRT targets.
- [ ] Profile-guided background retuning and compatible executable replacement using real serving traces.
- [ ] Automated kernel/profile regression isolation and safe rollback.
- [ ] Optional compiled-function batching if it simplifies homogeneous offline RL; do not make online serving depend on
      it.

## 17. First milestone and execution plan

Each stage has an exit gate. Do not advance because APIs exist; advance when correctness and performance evidence pass.

### First production-model decision

The first real-model target is the **Gemma 4 31B dense instruction model**. The first milestone covers its text decoder,
ordinary autoregressive prefill/decode, tokenization/chat template, BF16 numerical reference, paged state, continuous
batching, and embedded streaming API. Vision input, training, MTP-based speculation, peak 256K-context operation, and
production quantization remain representable but are not required for the first integrated engine.

Validation uses two sizes without creating two model implementations:

- a reduced-shape configuration derived from the 31B architecture, retaining GQA/QK-norm, partial RoPE, local/global
  attention, per-layer-input embeddings, KV-sharing relationships, normalization placement, and soft-capping semantics;
- the complete published Gemma 4 31B checkpoint as the milestone acceptance target.

The reduced configuration is for fast layer/model parity, generated cases, sanitizer runs, scheduler integration, and
CI. It is never used for performance claims. Full-checkpoint tests pin the exact checkpoint revision, tokenizer, chat
template, numerical mode, hardware topology, and reference implementation.

Qwen3.6 27B is the planned second production-model extensibility test. Its hybrid Gated DeltaNet/full-attention kernels
should be added only after Gemma passes the first milestone. Validate recurrent snapshot/restore/replay and hybrid
retention semantics with tiny conformance state before stabilizing contracts; Qwen then tests those same contracts at
production scale without requiring a scheduler or lifecycle fork.

### Conformance executors

Scheduler and state experiments can proceed alongside Gemma model bring-up. Build two deliberately production-shaped
conformance implementations behind the same `ModelExecutor`/profile/state contracts used by Gemma. They complement
the actual-provider feasibility probe; passing a synthetic executor cannot substitute for that probe.

1. **Pure-Rust conformance executor.** Fast deterministic execution with a virtual clock, configurable per-phase/token
   costs, controllable completion order, and fault/resource injection. It supports exhaustive state-machine and policy
   testing without a device.
2. **XLA conformance model.** A tiny compiled stateful model using the production static physical ABI, device buffers,
   page tables, aliases, profile selection, enqueue path, and completion fences. It validates the physical integration
   without requiring Gemma weights. Its state boundary must use the ownership and execution mechanism proven by the
   actual-provider probe, including the same completion and metadata lifecycle.

Begin with deterministic prefill/decode/mixed behavior, variable lengths, append-only pages, and a tiny overwritten
recurrent accumulator with bounded sliding-window history. Before contract stabilization, demonstrate coherent hybrid
checkpointing, accepting a prefix of speculative work, snapshot restoration or replay, delayed completion, and failure
after possible writes. Exercise cancellation after submission, metadata-slot reuse, device future-token dependencies,
and resource exhaustion. The Rust executor uses virtual time rather than wall-clock sleeps.

Extend both executors alongside each production feature with score/verify, prefix sharing and partial-page COW,
fork/commit/rollback, pause/resume, weight-dependent outputs, and version retention. Maintain deterministic event-log
reconstruction of every allocation, lease, transition, and reclamation. The entire eventual feature matrix is not a
prerequisite for the first real Gemma execution path.

The scheduler may not inspect a conformance-model type, install conformance-specific branches, bypass page leases, or
use a different completion path. Replacing a conformance executor with Gemma is executor/profile substitution only.

### Phase 0 — physical feasibility and benchmark laboratory

Deliverables:

- [ ] Pin the Gemma 4 31B checkpoint revision, reduced configuration, tokenizer/template, first NVIDIA target, initial
      context/batch envelope, numerical contract, and full memory budget, including two weight snapshots. Decide whether
      tensor parallelism is needed for the first generation or weight-update gate and schedule it accordingly.
- [ ] Register the primary workload, performance hypotheses, baseline configurations, quantitative acceptance targets,
      quality tolerances, and cost budgets from section 18 before tuning Ryft. Pin vLLM, SGLang, and TensorRT-LLM
      versions and supported configurations on the same hardware, documenting any unsupported baseline.
- [ ] Prototype minimal request, state-group, transaction, profile, and executor contracts without freezing them.
      Validate append-only, sliding-window, and overwritten recurrent conformance behavior and failure semantics.
- [ ] Build a physical probe using an actual candidate paged-attention CUDA provider with Gemma-compatible shapes,
      cache writes, device sampling, persistent state, fixed-capacity outputs, and real per-batch planning metadata.
- [ ] Prove repeated calls and profile changes preserve state addresses without pool-sized copies or hidden host waits.
      Validate graph capture/replay, workspace lifetime, future-token feedback, and bounded metadata-buffer reuse.
- [ ] Exercise outstanding launches touching disjoint pages in one pool and dependent launches touching shared state.
      Identify whole-buffer serialization; prove safe access and measure whether the design meets overlap budgets.
- [ ] Inject cancellation after submission, partial writes, asynchronous failure, and delayed completion; verify no
      premature publication or reclamation. Define quarantine/recovery when device quiescence cannot be established.
- [ ] Define Gemma numerical oracles against the official JAX implementation and an external serving engine. Build
      minimal trace replay and reporting for controlled and quality-constrained baseline comparisons.

Exit gate: the actual-provider probe proves the state/graph/overlap mechanism on the declared topology, or the design
is revised and retested before engine interfaces stabilize. Keep HLO/buffer-assignment evidence, address checks, device
traces, allocation/copy accounting, and fault-test results. A synthetic model or successful enqueue alone cannot pass.
Baseline fixtures and the performance contract must be reproducible; numeric targets cannot remain unspecified.

### Phase 1 — minimal continuous Gemma engine

After physical feasibility, model bring-up, minimal engine work, and provider integration can proceed concurrently
against the same provisional contracts. Stabilize only the interfaces exercised by the real-model path. Shared changes
receive joint review; do not build the complete future lifecycle framework before integrating Gemma.

Deliverables:

- [ ] Implement the Gemma configuration, `Parameterized` weights, checkpoint importer, tokenizer/chat template, and
      shared model semantics. Compare reduced-shape and full-checkpoint layers, logits, tokens, and logprobs under the
      declared numerical contract, using contiguous state as the initial reference.
- [ ] Keep weights as explicit resident versioned snapshots. Compile prefill/decode and representative mixed profiles;
      validate persistent reload and the same architecture code across contiguous and paged execution.
- [ ] Implement minimal admission, request FSM, required/in-flight/completed/accepted frontiers, batch construction,
      completion reactor, bounded streaming, cancellation, and backpressure.
- [ ] Implement HBM pools/tables, metadata slots, leases, and completion-safe reclamation using the proven state ABI.
      Integrate paged attention/cache append, device sampling, and a portable numerical fallback.
- [ ] Extend both conformance executors for this exact slice, with event-log reconstruction and deterministic faults.
- [ ] Implement tensor-parallel weights/state and rank-consistent launch/error handling here if Phase 0 requires it.
      Add full-step capture and overlap validation on that topology before declaring its execution path accepted.

Ordered integration gates within this phase:

1. **I1 — minimal engine semantics.** Admission, generation, completion, cancellation, and backpressure pass against
   Rust conformance with no leaks or invalid transitions, including submitted cancellation and failed writes.
2. **I2 — physical execution.** The same engine passes XLA conformance using the proven arena, metadata, and completion
   path; overlap traces show no unintended waits, copies, or early reuse.
3. **I3 — real model, contiguous state.** Reduced and full Gemma configurations share one implementation and satisfy
   the declared reference tolerances; exact-token fixtures and statistical sampling tests have explicit scope.
4. **I4 — real model, paged state.** Substitute paged execution without a scheduler fork; validate state updates and
   numerical agreement, cancellation, profile changes, and reclamation under repeated launches.
5. **I5 — minimal continuous Gemma engine.** Dynamic arrivals, bounded prompt chunks, decode, streaming, and device
   sampling run on the full checkpoint with stable memory and no per-token compilation. Measure homogeneous and
   mixed execution, lookahead enabled/disabled, GPU idle gaps, and end-to-end latency against pinned baselines.

Phase exit: a usable embedded Gemma engine with an explained performance breakdown. It may trail mature engines; record
each material gap against the budgets rather than describing correctness as competitive performance. Prefix retention,
advanced policy, and the complete lifecycle matrix are not prerequisites for this integration gate.

### Phase 2 — prefix reuse, mixed scheduling, and retention

Deliverables:

- [ ] Add prefix publication/indexing, partial-page COW, eviction, preemption/replay, and completion-safe reclamation.
- [ ] Tune chunked prefill and mixed `extend` execution against homogeneous profiles using token and state budgets.
      Add deadlines/slack, fairness/starvation bounds, and deterministic scheduler decision replay.
- [ ] Add session pause/resume, retention/expiry, and cost-aware admission for tool-paused and bursty workloads.
- [ ] Extend conformance coverage for each feature, including hybrid checkpoint validity and shared-prefix cancellation.
- [ ] Run controlled ablations for prefix retention, mixed execution, and lookahead, including cold-cache and hostile
      reuse distributions. Diagnose padding, launch, kernel, and state-memory costs against the registered budgets.

Phase exit: the first full Gemma workload/SLO envelope is reproducible against pinned baselines with bounded starvation,
stable memory, and attributable latency and goodput. State and request invariants remain shared across executors.

### Phase 3 — early weight-update and rollout cycle

Deliverables:

- [ ] Add snapshot prepare/validate/publish/drain/reclaim, version-aware cache invalidation, and bounded old-version
      retention. Use the Phase 0 memory policy and add tensor parallelism now if required for double buffering.
- [ ] Implement trainer-to-engine disk or colocated transfer first. Demonstrate a real synchronous generate/score/update
      cycle using shared model semantics; full training infrastructure is not part of the engine.
- [ ] Validate the reduced-shape differentiable training/score path, then full-checkpoint logits and raw/behavior
      logprobs under the selected contract. Updated snapshots must change outputs as expected without recompilation
      of unchanged profiles; a version-ID-only simulation does not pass the real update gate.
- [ ] Add token provenance, async agent-loop API, partial rollouts, replay/batch-invariance options, and explicit
      staleness admission. Reconstruct state when moving continuations to a newer policy.
- [ ] Benchmark update cadence jointly with session retention, tool pauses, cache rebuild, and useful rollout yield.
      Include end-to-end agent latency and snapshot-to-accepted-rollout time under fixed memory/staleness budgets.

Exit gate: real weight updates and Gemma rollouts repeat without process restart, stale-cache reuse, unbounded memory,
or unexplained probability mismatch. Demonstrate or reject the registered integration hypothesis after charging all
update, retention, and reconstruction costs. Run this before broad decoding, portability, or service expansion.

### Phase 4 — advanced lifecycle, decoding, and scale

Deliverables:

- [ ] Expand tensor-parallel Gemma performance/topology coverage; initial support has already shipped if earlier memory
      or SLO gates required it. Complete multi-rank failure handling before accepting each new topology.
- [ ] Add pipeline, context, and/or attention data parallelism only where the target envelope requires it; defer expert
      parallelism until an MoE model is selected.
- [ ] Add production fork/checkpoint/rollback for best-of-N/tree rollouts and richer speculative state strategies.
- [ ] Add sleep/offload/wake, colocated role arbitration, and collective/RDMA weight transfer where measured demand
      justifies them. Complete conformance coverage alongside each feature.
- [ ] Add grammar compilation/masking and deterministic-span fast forwarding.
- [ ] Add n-gram speculation, then enable Gemma's MTP proposer with adaptive depth/enablement.
- [ ] Add quantized weights/KV and adapter/LoRA registry based on measured demand.
- [ ] Bring up Qwen3.6 27B as the second model, including hybrid recurrent/attention state behind the existing
      `SequenceState` and scheduler contracts.
- [ ] Reach CUDA/ROCm provider parity for the supported feature slice or publish explicit capability degradation.

Exit gate: multi-rank failures fail coherently; Gemma performance scales predictably; Qwen requires no scheduler or
request-lifecycle fork; advanced features improve their target traces without unacceptable default-workload regressions.

### Phase 5 — service hardening

Deliverables:

- [ ] OpenAI-compatible service, lower-level protocol, auth/quota adapters, health/readiness, and graceful drain.
- [ ] Hot model/adapter load, rolling update, overload/load-shed policy, and operational runbooks.
- [ ] Production trace capture/replay, capacity estimator, and configuration advisor.
- [ ] Security review of model files, grammars, request limits, FFI plugins, and multi-tenant state isolation.
- [ ] Soak, chaos, restart, fragmentation, and long-context tests.

Exit gate: production SLOs, recovery, observability, and operations are demonstrated under representative failure and
burst tests, not only steady-state benchmarks.

### Phase 6 — distributed and disaggregated serving

Deliverables:

- [ ] Worker discovery/membership, KV-aware routing, replica placement, global state-event index, and autoscaling hooks.
- [ ] Aggregated-versus-P/D profiler/simulator and online planner.
- [ ] Versioned state-session handoff with measured transport providers and complete cleanup paths.
- [ ] Host/remote state tiers only where reuse-value models show benefit.
- [ ] Multi-node rolling upgrade, drain, retry, and partition behavior.

Exit gate: disaggregation beats aggregated serving on a declared workload/SLO/fabric after charging transfer and control
cost; otherwise aggregated mode remains the default.

## 18. Benchmark and acceptance strategy

### Primary envelope and performance hypotheses

The primary serving workload is full-checkpoint Gemma 4 31B text generation for multi-turn coding agents with long
prefixes, tool pauses, bursty resumptions, and concurrent new arrivals on one NVIDIA worker group. Measure fixed-version
serving first, then the same session distribution under weight churn. Short interactive chat and low-reuse traffic are
regression controls; saturated generation is a separate throughput surface. This choice does not imply that retention
or mixed execution will win before they are measured.

Before Phase 0 exits, commit an experiment manifest containing exact hardware/device count/interconnect, checkpoint
and tokenizer revisions, context/output-length distributions, arrival process, concurrency, pause/reuse distributions,
sampling and quality policy, TTFT/ITL/E2E SLOs, fairness bound, and memory limits. For RL include update cadence,
maximum policy lag, retained-version count, and trainer/transfer topology. Hardware and absolute latency values remain
unselected in this proposal; choose and record them before implementation tuning, not after seeing favorable results.

Register and test three mechanisms independently and together:

| Hypothesis | Mechanism and comparison | Success measure |
|---|---|---|
| Overlap reduces host-induced GPU idle time. | Device token feedback, metadata lookahead, and graph replay; compare the same engine with lookahead enabled/disabled. | Lower idle gaps and better SLO goodput or ITL without increased cancellation errors or unacceptable wasted work. |
| State-aware scheduling improves agent serving. | Prefix/session retention and mixed token execution; independently disable retention and mixed execution under the same memory budget. | More completed requests meeting all SLOs, bounded starvation, and improved end-to-end agent latency where reuse exists. |
| Shared model and weight lifecycle improves rollout turnaround. | Reuse model/score semantics and resident profile families; compare a mature rollout integration under the same training, memory, and lag constraints. | Lower snapshot-to-accepted-rollout time and higher useful rollout yield after charging transfer, retention, and prefix rebuild. |

### Quantitative targets and cost budgets

The following are initial engineering targets, not measured results. Ratify them against the pinned hardware and
baseline variance in Phase 0, recording any revision and rationale before tuning. A missed target requires an explained
gap and a new scoped experiment; it cannot silently become a performance claim.

- **Competitive milestone:** at least 95% of the best validated baseline's SLO goodput at the declared comparison
  point, with numerical/quality constraints satisfied and the same TTFT, ITL, and E2E limits.
- **Leadership milestone:** at least 20% higher SLO goodput, or 20% lower rollout turnaround, than the best validated
  baseline on the registered primary surface, with the alternative metric and quality/SLO guardrails reported.
  Require repeated-run uncertainty to support the improvement rather than one favorable sample.
- **Regression guardrail:** no more than 5% goodput or p99 latency regression on the registered control traces when
  enabling an optimization. Otherwise restrict it to the workload envelope where it helps and publish the tradeoff.
- **Enqueue/overlap budget:** host-induced GPU idle gaps should be no more than 5% of steady-state execution time.
  Report scheduler, metadata planning/upload, submission, and output handling separately, including their tail costs.
- **Padding budget:** target no more than 10% extra dense token-row computation over useful rows on the primary trace;
  report attention work separately. Measure sparse concurrency as well as full batches before adding profile buckets.
- **Compilation/copy budget:** zero per-token compilation, implicit shape synchronization, full-logit readback, or
  pool-sized copy protection in the competitive steady-state path. Account separately for intentional COW, checkpoint,
  and transfer copies and show that their volume follows touched state rather than total pool capacity.
- **Memory and lifecycle budgets:** the manifest sets byte limits for weights/retained versions, state, activations,
  captured profiles, provider workspaces, and metadata/transfer slots, plus maximum update pause and drain time. Sum
  simultaneous peak use with an explicit reserve; measure it on device. An LRU executable count alone is not a memory
  budget. Version retention and lookahead must remain within these limits under backpressure and delayed completion.

For each dominant profile, estimate compute work, weight/state bytes moved, and communication volume. Compare measured
time against applicable compute, bandwidth, and interconnect bounds, explaining overlap rather than summing incompatible
lower bounds. Attribute material gaps to kernels, layout, padding, launch/host idle, communication, or state movement.
Kernel microbenchmarks and a complete model-step trace must agree with the end-to-end diagnosis.

### Workload matrix

- pinned full-checkpoint Gemma 4 31B text generation as the first acceptance surface;
- short interactive chat;
- long prompt/short answer;
- long-context multi-turn coding agent with high prefix reuse;
- tool-paused conversations and bursty resumptions;
- structured JSON/tool calls;
- saturated offline generation;
- synchronous RL rollouts after frequent weight updates;
- asynchronous multi-turn agentic RL with bounded staleness;
- joint weight-churn/retention sweeps, including expiry, cold-prefix reconstruction, and old-version drain bursts;
- speculative-friendly and speculation-hostile traces;
- MoE and dense models; one multimodal trace after the text engine is stable.

### Metrics

- p50/p90/p99 TTFT, TPOT/ITL, E2E, and queue delay;
- SLO goodput and per-user decode-rate floor;
- prompt/output/total tokens per GPU-second and per dollar;
- HBM usage, page fragmentation, prefix hit, COW, eviction/recall, and transferred bytes;
- host CPU, scheduler time, launch overhead, graph/profile hit rate, and compilation misses;
- speculative acceptance/effective speedup;
- weight-update pause, transfer, publish, old-version drain, prefix rebuild, and snapshot-to-accepted-rollout time;
- useful completed rollouts per GPU-second, discarded/stale rollout fraction, and end-to-end agent completion latency;
- metadata-ring occupancy, GPU idle gaps, lookahead work discarded after stop/cancellation, and per-profile memory;
- numerical mismatch, determinism, and failure/retry rate;
- power/energy where hardware telemetry is reliable.

### Comparison tracks and rules

- **Controlled numerical track:** same checkpoint, numerical mode, tokenizer, sampling, maximum lengths, prompt/output
  distribution, and SLO. Attribute engine/scheduler/kernel effects under equivalent semantics.
- **Quality-constrained production track:** use each engine's best validated quantization, speculation, and kernel
  configuration under a common declared quality and probability-policy constraint, hardware/resource budget, and SLO.
  Publish quality tests, tolerances, output-length differences, and all configuration differences. Do not equate BF16
  parity with leadership over optimized production configurations or accept speedups caused by degraded outputs.
- Reduced-shape and conformance models are correctness/development tools and never support competitive performance
  claims; external-engine comparisons use the complete pinned Gemma 4 31B checkpoint.
- Warm and cold results are separate; compilation, model load, and prefix-cache warmup are reported.
- Compare optimized supported configurations, not deliberately weak defaults.
- Publish Pareto curves across concurrency/profile choices and all failure/unsupported cases.
- Use open-loop arrival traces for overload/queueing tests to avoid hiding stalls through client pacing; use separate
  closed-loop agent traces for tool dependencies. Count rejected, timed-out, and cancelled requests explicitly. Define
  SLO goodput as completions satisfying all registered request SLOs, not merely accepted load or average token rate.
- Repeat experiments with declared seeds and run durations, report uncertainty, and hold trace splits for final
  validation. Include cold/low-reuse controls and component ablations so benchmark-specific tuning is visible.
- Treat project-reported numbers as hypotheses until reproduced in this harness.

## 19. Risks and explicit mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| XLA static semantics fight dynamic state | Copies, profile explosion, host sync | Rust-owned preallocated pools; static physical ABIs; aliased custom calls; explicit profile families. |
| Contracts stabilize before physical proof | Expensive executor/state redesign | Phase 0 actual-provider probe; provisional internal interfaces; minimal real-model integration before framework expansion. |
| Donation silently becomes copy protection | Catastrophic KV bandwidth/memory regression | Exclusive serving buffers and hard validation; donation is optimization, never state correctness. |
| Whole-buffer ownership serializes independent pages | GPU bubbles despite async APIs | Probe outstanding disjoint-page launches; explicit access dependencies; measured overlap and ownership granularity. |
| Submission is mistaken for successful completion | Published corrupt state or early reuse after cancellation | Separate submitted/completed/published states; fences, quarantine, trusted restore/replay, and fault injection. |
| Token-page abstractions omit recurrent semantics | Invalid rollback or prefix reuse in hybrid models | State-group capabilities, retention/checkpoint boundaries and costs; early recurrent conformance. |
| Generic XLA kernels trail specialists | Poor decode/MoE performance | Kernel registry; vendor/community integration; Ryft-native kernels only where strategically valuable. |
| Too many profile combinations | Compile latency and memory blow-up | Geometric buckets, trace-driven profiles, bounded cache, AOT warmup, compatibility matrix. |
| Kernel/plugin ABI drift | Cache corruption or runtime failure | Versioned ABI, capability handshake, artifact identity, subprocess/startup validation, fallback/deny-list. |
| Prefix-locality policy starves requests | Tail-latency/SLO failures | Deadline/slack and aging bounds dominate locality after a configured threshold. |
| Weight update reuses stale state | Invalid on-policy rollouts or serving output | Weight version in every request/cache key; transactional publish; stale reuse unrepresentable by default. |
| Weight churn destroys the expected reuse advantage | Higher latency or memory use despite faster refits | Joint update/retention experiments; bounded versions and lag; charge reconstruction and drain costs. |
| BF16-only comparisons overstate competitiveness | Good reference parity but weak production economics | Controlled numerical and best quality-constrained comparison tracks with registered targets. |
| P/D transfer costs exceed benefit | Worse latency and utilization | Workload/topology simulator; aggregated default; evidence-gated disaggregation. |
| Broad compatibility delays a useful engine | Never reaches competitive quality | Narrow beachhead and explicit capability matrix; expand only after milestone gates. |
| Conformance model creates unrealistic shortcuts | Gemma integration forces a scheduler or state rewrite | Same executor/profile/state contracts, a real-XLA conformance path, prohibited type checks/special cases, and ordered substitution gates. |
| Rust creates false confidence | Safe host code but slow kernels/poor scheduling | Measure GPU, memory, launch, network, and queue behavior independently; compare Pareto frontiers. |
| External ecosystem moves faster | Permanent feature catch-up | Stable semantic provider APIs; reuse external kernels/protocol adapters; focus Ryft differentiation. |

## 20. Feasibility assessment

### Why optimism is justified

Ryft does not start where most inference-engine projects start. It already has a coherent typed program model,
structured parameters, transformations, symbolic dimensions, sharding, broad inference primitives, async PJRT,
buffer interop, explicit aliasing, persistent/distributed executable caches, custom FFI kernels, collectives, profiling,
and adaptive recompilation. The repository's tiny decode loop proves that model state, sampling, and a foreign attention
kernel can already be expressed through the public stack.

Rust is especially valuable for the part that remains: scheduler concurrency, page ownership, state transitions,
cancellation, versioning, transfer leases, low-overhead embedding, and operational reliability. Ryft's common compiler
and parameter architecture creates a real opportunity to eliminate the duplicated model/sharding/weight semantics that
RL systems currently bridge between trainers and separate Python inference engines.

That opportunity requires one architecture implementation with context-selected attention/state semantics and explicit
numerical tests across training, scoring, and serving. SGLang-JAX already combines JIT execution, specialized kernels,
and serving; compiler-backed execution itself is not unique. Ryft must demonstrate the benefit of its shared semantics
and resource protocol through the registered serving and rollout experiments.

### Why expectations must remain disciplined

The missing work is precisely the moat of mature inference systems: model import and coverage, paged state, highly tuned
attention/GEMM/MoE/sampling kernels, continuous scheduling, quantization, speculative/structured decoding, distributed
transfer, service APIs, failure handling, and years of workload tuning. XLA's static buffer planning is helpful for model
activations and graph replay, but it does not solve a shared dynamically growing cache. Rust cannot improve GPU memory
bandwidth, Tensor Core utilization, collective overlap, or prefix hit rate by itself. Dynamo already uses Rust in its
distributed runtime, and TokenSpeed uses C++ in its scheduler; the language is an implementation advantage, not a moat.

### Realistic target

The right ambition is:

1. match a mature engine on the pinned Gemma 4 31B checkpoint/hardware/feature/SLO slice;
2. demonstrate a registered, attributable win on that beachhead through overlapped execution, state-aware agent serving,
   or shared-model rollout integration; measure frequent updates and prefix retention jointly rather than assuming
   their benefits add together;
3. preserve architectural coherence while adding models, kernels, platforms, and deployment modes;
4. add Qwen3.6 27B without forking scheduler or lifecycle semantics, validating hybrid persistent state;
5. claim broader superiority only as the benchmark matrix earns it.

On that definition, building something **much better** is realistic. Building something immediately and universally
faster, more complete, and more production-proven is not.

## 21. Decisions to validate early

- [x] Select Gemma 4 31B dense instruction as the first production-model target and Qwen3.6 27B as the second-model
      extensibility test.
- [x] Use pure-Rust and real-XLA conformance executors to unblock engine work while preserving production contracts.
- [ ] Prove the physical state/provider/graph/overlap mechanism before stabilizing executor and state interfaces.
- [ ] Validate append-only, sliding-window, and overwritten recurrent state, including checkpoint/replay costs and
      coherent hybrid-prefix boundaries, in the initial conformance slice.
- [ ] Pin the exact Gemma checkpoint revision, tokenizer/template revision, NVIDIA hardware topology, BF16 context/batch
      envelope, complete peak memory budget, and serving/RL traces. Decide when tensor parallelism is required.
- [ ] Ratify quantitative leadership/competitive targets, control-trace guardrails, quality tolerances, and cost budgets
      before tuning; record baseline uncertainty and controlled versus production comparison configurations.
- [ ] Specify the reduced-shape Gemma 4 31B-derived configuration and parity fixtures.
- [ ] Decide whether the first paged-attention provider wraps FlashInfer/vendor code or is Ryft-native.
- [ ] Validate that XLA preserves the required state-pool aliases without hidden copy protection.
- [ ] Validate disjoint-page access with outstanding launches, immutable metadata slots, device token feedback, mixed
      execution, and asynchronous token/status delivery. Bound post-cancellation lookahead and reclamation latency.
- [ ] Prove full-step command-buffer capture with the chosen custom calls and collectives.
- [ ] Choose page size/layout using measured prefill, decode, prefix sharing, and fragmentation tradeoffs.
- [ ] Decide whether model weights are ordinary executable inputs, captures, or external handles per platform.
- [ ] Specify replay/batch-invariance scope, raw versus behavior logprobs, speculative distribution semantics, and
      allowable training/inference numerical differences independently of on-policy provenance.
- [ ] Define old-version retention/expiry and continuation reconstruction; benchmark weight churn and prefix reuse
      together under fixed memory/staleness limits.
- [ ] Establish the initial scheduler SLO objective and starvation bound.
- [ ] Define the kernel/provider security and distribution model before loading out-of-tree artifacts.

## 22. Primary sources

### Beachhead models

- [Gemma 4 overview](https://deepmind.google/models/gemma/gemma-4/)
- [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Google DeepMind Gemma JAX reference](https://github.com/google-deepmind/gemma)
- [Qwen3.6 27B model card](https://huggingface.co/Qwen/Qwen3.6-27B)
- [Qwen3.5/3.6 hybrid architecture documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5)

### Inference engines and runtimes

- [vLLM PagedAttention paper](https://arxiv.org/abs/2309.06180)
- [vLLM V1 guide](https://docs.vllm.ai/en/v0.10.0/usage/v1_guide.html)
- [vLLM architecture walkthrough](https://vllm-project.github.io/2025/09/05/anatomy-of-vllm.html)
- [vLLM weight transfer](https://docs.vllm.ai/en/stable/training/weight_transfer/)
- [vLLM hybrid state/cache management](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)
- [vLLM reproducibility and batch invariance](https://docs.vllm.ai/en/latest/usage/reproducibility/)
- [SGLang paper](https://papers.nips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf)
- [SGLang/RadixAttention launch article](https://www.lmsys.org/blog/2024-01-17-sglang/)
- [SGLang overlapped scheduler](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/)
- [SGLang-JAX architecture](https://github.com/sgl-project/sglang-jax/blob/main/docs/architecture/01-architecture-overview.md)
- [TensorRT-LLM architecture](https://nvidia.github.io/TensorRT-LLM/developer-guide/overview.html)
- [TensorRT-LLM KV cache system](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html)
- [TensorRT-LLM disaggregated serving](https://nvidia.github.io/TensorRT-LLM/features/disagg-serving.html)
- [TokenSpeed launch and architecture](https://lightseek.org/blog/lightseek-tokenspeed.html)
- [TokenSpeed repository](https://github.com/lightseekorg/tokenspeed)
- [NVIDIA Dynamo architecture](https://docs.nvidia.com/dynamo/dev/knowledge-base/overview)
- [NVIDIA Dynamo disaggregated serving](https://docs.nvidia.com/dynamo/latest/user-guides/disaggregated-serving)
- [FlashInfer paper](https://arxiv.org/abs/2501.01005)
- [FlashInfer attention APIs](https://docs.flashinfer.ai/api/attention.html)
- [DeepSpeed-FastGen paper](https://arxiv.org/abs/2401.08671)
- [MLC-LLM overview](https://llm.mlc.ai/)
- [MLC-LLM compilation flow](https://llm.mlc.ai/docs/compilation/compile_models.html)
- [TGI architecture](https://huggingface.co/docs/text-generation-inference/architecture)

### RL and agentic integration

- [verl engine workers](https://verl.readthedocs.io/en/latest/workers/engine_workers.html)
- [verl agentic RL architecture](https://github.com/verl-project/verl/blob/main/docs/start/agentic_rl.rst)
- [vLLM RLHF integration](https://docs.vllm.ai/en/v0.8.1/training/rlhf.html)
- [SGLang for RL/refit](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/sglang_for_rl.md)

### XLA and kernel substrate

- [XLA:GPU architecture](https://openxla.org/xla/gpu_architecture)
- [XLA command buffers](https://openxla.org/xla/hlo_to_thunks)
- [XLA FFI custom calls](https://openxla.org/xla/custom_call)
- [PJRT concepts and async execution](https://openxla.org/xla/pjrt/cpp_api_overview)
- [StableHLO dynamism](https://openxla.org/stablehlo/dynamism)
- [JAX Pallas kernel language](https://docs.jax.dev/en/latest/pallas/index.html)
