# Ryft Inference: Architecture and Execution Plan

**Status:** proposed architecture

**Research snapshot:** 2026-08-14

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

Names are provisional. Keep the initial number of crates small and split only when dependency or ownership boundaries
become real.

### `ryft-models`

- Model configurations and `Parameterized` weight/state schemas.
- Reusable NN layers and architecture implementations.
- Stable parameter names and SafeTensors/Hugging Face import mapping.
- `prefill`, `decode`, `score`, `embed`, and optional training semantics expressed with Ryft operations.
- No scheduler, page allocator, HTTP types, or platform-specific kernel target names.

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
- input tokens, generated tokens, computed-token frontier, and maximum output;
- sampling/grammar/speculation state and deterministic RNG stream;
- owned/shared sequence-state blocks and optional speculative checkpoint;
- deadline, priority, SLO class, enqueue time, and fairness accounting;
- chosen worker/profile/topology and transfer/refit state;
- per-token logprob and policy-version provenance when requested.

### Versioned model state

`WeightVersion` is monotonically increasing and immutable. A request binds to one version unless an explicit policy
permits bounded staleness. Sequence-state cache keys include the weight version; the default APIs make reuse across
weight updates impossible.

A weight update is transactional:

1. prepare and validate a complete sharded snapshot;
2. transfer it using disk, host, IPC, collective, or RDMA provider;
3. make the snapshot executable-ready on every rank;
4. atomically publish the new version for new requests;
5. drain or retain old-version requests according to policy;
6. reclaim the old snapshot only after all requests and asynchronous launches release it.

Use double buffering first. Optimize to delta/LoRA or in-place refit only with explicit correctness and memory evidence.

## 8. Paged sequence-state service

### Why it is not just `KvCache`

Modern workloads include MHA/GQA/MLA KV, sliding windows, recurrent state, Mamba/linear attention, multimodal encoder
outputs, speculative branches, and tool-paused sessions. The allocator and prefix index should be generic over a
`StateClass`, while model-specific kernels interpret the bytes/layout.

### Main types

- `StatePool`: one physical allocation class on one device/memory tier.
- `StatePageId`: stable logical identifier, never a raw pointer in scheduler code.
- `StateLayout`: model/layer range, page tokens, dtype, dimensions, sharding, and memory-space identity.
- `SequenceTable`: logical token/state position to physical page mapping for one sequence.
- `PageLease`: shared or exclusive ownership tied to completion dependencies.
- `SequenceCheckpoint`: cheap fork/rollback boundary for speculative decoding and branching agents.
- `PrefixKey`: hash over tokens plus model, weight version, adapter, position policy, state layout, and relevant model
  configuration.
- `StateTransaction`: reserve, append, fork, truncate, commit, abort, publish, transfer, or evict as one operation.

### Required invariants

- A physical page is never reused while a compute or transfer fence may still access it.
- A shared page is immutable; partial-page continuation requires copy-on-write.
- Prefix publication happens only after the producing launch completes successfully.
- Aborting any transaction deterministically releases its uncommitted pages.
- Page-table updates become visible atomically at the scheduler/executor boundary.
- Weight/model/layout mismatches make a prefix ineligible, even when token hashes match.
- Every allocation, reference, transfer, and eviction can be reconstructed from a deterministic event log in tests.

### Storage tiers

Start with HBM only. Add pinned host memory after eviction and transfer metrics exist, then remote DRAM/NVMe only for a
validated workload. The interface supports async `store`, `load`, `transfer`, and `cancel`, but policy decides whether
the expected reuse value exceeds the transfer and capacity cost.

Use a radix tree or block-hash tree only as an index. Keep it separate from physical allocation and tier I/O. This makes
the index replaceable and avoids the large coupled cache-manager classes now being split in mature engines.

## 9. Scheduler design

### Unified work model

Adopt vLLM V1's clean idea that a request has `required_tokens` and `computed_tokens`, extended with explicit state and
cost. Prefill, decode, prefix hits, speculative proposals, verification, and deterministic grammar spans all become
work that advances the computed frontier.

The scheduler produces a `StepPlan` containing:

- admitted requests and token ranges;
- selected execution phase/profile and active slot count;
- page reservations, block tables, slot mappings, lengths, masks, and state checkpoints;
- model/adapter/weight version and distributed worker group;
- sampling/grammar/speculation actions;
- expected resource usage and completion/rollback actions.

Planning and committing are separate. A plan reserves resources; launch success commits it; launch failure or
cancellation rolls it back.

### Policy layers

1. **Capacity admission:** Can the complete or safely preemptible work fit in page, weight, adapter, graph, and transfer
   budgets without thrashing?
2. **SLO priority:** Deadline/slack, queue age, priority, tenant quota, and predicted TTFT/TPOT risk.
3. **Locality:** Reusable prefix/state, resident adapter/weights, and topology/transfer cost.
4. **Batch efficiency:** Compatible model/profile/kernel/grammar/speculation shapes and predicted execution time.
5. **Fairness:** Bounded starvation regardless of locality and long prompt/output behavior.

Policies implement a small trait over immutable scheduler snapshots. The transition engine and resource invariants are
not replaceable. Record every decision so a trace can be replayed deterministically.

### Modes

- `Latency`: decode-biased, tight batch wait, SLO slack first.
- `Throughput`: fuller batches and larger chunks for offline generation/RL.
- `Deterministic`: batch-invariant kernels/sampling and stable ordering for on-policy RL/evaluation.
- `Agentic`: high value on prefix/session retention and pause/resume.
- Custom policies may adjust scoring, but not bypass ownership or version rules.

## 10. Execution profiles and XLA strategy

### Host-driven steps

Compile distinct semantic entry points:

- `prefill`: one or more prompt chunks, producing/committing state pages and optional first-token logits;
- `decode`: one token per active sequence (or a small deterministic span);
- `verify`: validate speculative token chains/trees and commit the accepted frontier;
- `draft`: generate candidate tokens for model-based speculation;
- `score`: logprobs/reward/reference-policy evaluation;
- `embed`/encoder stages for multimodal and encoder-decoder models.

The online scheduler selects one entry point each tick. A compiled multi-token loop remains useful for fixed homogeneous
offline rollouts, benchmarks, and some local workloads, but it is an optimization outside the scheduler's correctness
boundary.

### Static physical ABI

Each executable receives fixed-capacity physical buffers plus device-resident logical metadata:

- packed input token IDs and positions;
- active slot mask and per-sequence lengths;
- page/block tables and append slot mappings;
- state pool buffers (explicitly aliased where mutated);
- resident weight/adaptor buffers or stable handles;
- RNG counters, grammar masks/state, and sampling parameters;
- optional speculative tree/acceptance metadata.

Inactive lanes have defined semantics and never mutate state or emit tokens. Output cardinality is host-known. This
avoids dynamic-output readback and preserves graph/profile reuse. Compile profile keys include model architecture,
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
presence/frequency penalties, bad/stop tokens, logprobs, and deterministic counter-based RNG. Start with CPU orchestration
and the existing Ryft primitives for correctness. Add fused GPU sampling only after measurement.

Deterministic mode defines whether results are invariant to batch composition and profile choice. RL records RNG stream,
weight version, kernel mode, and logits/logprob provenance.

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
- forks for best-of-N/tree search with copy-on-write state;
- token IDs, masks, per-token logprobs, entropy/auxiliary outputs, finish reasons, and version provenance;
- deterministic mode and explicit bounded-staleness asynchronous mode;
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

The underlying PJRT layer already implements many capabilities that a normal project would list as missing. The work is
mainly exposing and composing them safely for serving.

### P0: first competitive engine

- [ ] Define a versioned semantic kernel registry and XLA FFI provider ABI.
- [ ] Add paged sequence-state pool/table metadata and aliased cache-write/paged-attention operations.
- [ ] Integrate at least one CUDA paged-attention implementation and one correctness fallback; define the ROCm path.
- [ ] Guarantee explicit exclusive state-buffer ownership. Detect rather than silently copy when an in-place serving
      contract cannot be honored.
- [ ] Add static physical prefill/decode profile construction with device-resident lengths, active masks, page tables,
      and slot mappings.
- [ ] Keep the decode hot path enqueue-only: no shape readback, host materialization, or implicit synchronization.
- [ ] Verify whole-step command-buffer capture/replay and stable workspace behavior for chosen profiles.

### P1: production scale

- [ ] Make shared execution fences awaitable/callback-driven and cheaply cloneable.
- [ ] Surface execution context, callbacks, monotonic launch IDs, incarnation IDs, device selection, and cancellation or
      bounded cancellation points through the high-level XLA executor.
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

Qwen3.6 27B is the planned second-model extensibility test. Its hybrid Gated DeltaNet/full-attention state should be
added only after Gemma passes the first milestone. Requiring that model to use the same scheduler and general
`SequenceState` contracts will test whether the design extends beyond ordinary transformer KV without making recurrent
state kernels part of the initial critical path.

### Conformance executors

Scheduler and state work must not wait for Gemma model bring-up. Build two deliberately production-shaped conformance
implementations behind the same `ModelExecutor`/profile/state contracts used by Gemma:

1. **Pure-Rust conformance executor.** Fast deterministic execution with a virtual clock, configurable per-phase/token
   costs, controllable completion order, and fault/resource injection. It supports exhaustive state-machine and policy
   testing without a device.
2. **XLA conformance model.** A tiny compiled stateful model using the production static physical ABI, device buffers,
   page tables, aliases, profile selection, enqueue path, and completion fences. It validates the physical integration
   without requiring Gemma weights or optimized attention kernels.

Both implementations must support deterministic prefill, decode, score, and verify behavior; variable prompt and
generation lengths; append-only pages; prefix sharing and partial-page COW; fork/commit/rollback; pause/resume;
weight-version-dependent outputs; cancellation; delayed/out-of-order completion; launch failure; and resource
exhaustion. The Rust executor uses virtual time rather than wall-clock sleeps.

The scheduler may not inspect a conformance-model type, install conformance-specific branches, bypass page leases, or
use a different completion path. Replacing a conformance executor with Gemma is executor/profile substitution only.

### Phase 0 — contracts and benchmark laboratory

Deliverables:

- [ ] Freeze request, output, sampling, version, state transaction, kernel provider, profile, executor, and
      scheduler-plan contracts in minimal Rust types.
- [ ] Define conformance behavior, virtual-time semantics, the state event log, and deterministic fault schedules.
- [ ] Define Gemma correctness oracles against the official JAX implementation and an external serving engine.
- [ ] Build a trace format and generator for synthetic, ShareGPT-like, long-context coding-agent, structured-output,
      burst, multi-turn/tool-pause, and RL rollout workloads.
- [ ] Build benchmark reporting for latency/throughput Pareto curves and SLO goodput.
- [ ] Pin the Gemma 4 31B checkpoint revision, reduced configuration, tokenizer/template, first NVIDIA target, initial
      context/batch envelope, BF16 reference, and memory budget.
- [ ] Pin baseline versions/configurations for vLLM, SGLang, and TensorRT-LLM on the same hardware.

Exit gate: contracts admit all required conformance behaviors and Gemma profiles; identical Gemma prompts/sampling
fixtures are comparable; load generators and metrics are trusted; no Ryft performance claim can bypass the harness.

### Phase 1 — parallel foundations

After Phase 0, three workstreams proceed concurrently. Their public boundaries are frozen together; discoveries that
change a shared contract are reviewed across all three rather than patched locally.

#### Track A — engine and Rust conformance executor

- [ ] Implement the request FSM, capacity admission, unified token scheduler, static-profile batch builder, completion
      reactor, cancellation, bounded streaming queues, and backpressure.
- [ ] Implement logical `StatePool`, `SequenceTable`, leases, transactions, prefix publication, COW, fork/rollback,
      preemption, pause/resume, eviction, and completion-aware reclamation.
- [ ] Implement continuous batching, chunked prefill, decode priority, deadlines, fairness/starvation bounds, and
      deterministic scheduler decision replay.
- [ ] Implement the pure-Rust conformance executor, virtual clock, cost controls, and deterministic fault injection.
- [ ] Implement sampling/stop/logprob semantics required by the initial Gemma API and engine-level metrics/traces.
- [ ] Add exhaustive/model-based tests that reconstruct every allocation, reference, transition, and reclamation from
      the state event log.

Track exit: mixed virtual-time traces exercise every lifecycle and failure path without leaks, invalid transitions, or
unbounded starvation; the engine contains no conformance-model special cases.

#### Track B — Gemma 4 31B model bring-up

- [ ] Create `ryft-models`; implement the Gemma 4 configuration, `Parameterized` weight schema, text layers, checkpoint
      import, tokenizer adapter, and chat template.
- [ ] Implement the reduced-shape 31B-derived configuration and compare layers, logits, greedy generation, sampled
      generation, and logprobs with the official JAX reference.
- [ ] Load the pinned full 31B checkpoint and establish BF16 text-decoder parity on the target hardware.
- [ ] Keep weights as explicit resident versioned snapshots rather than embedding large constants into executables.
- [ ] Compile separate static contiguous-cache prefill and decode entry points and validate persistent profile reload.
- [ ] Represent, but do not initially enable, multimodal inputs and Gemma's MTP draft path in stable model metadata.

Track exit: the reduced and full configurations use one implementation; the full checkpoint has deterministic end-to-end
parity; no per-token compilation occurs; contiguous-cache generation is stable and fully profiled.

#### Track C — XLA state ABI and serving kernels

- [ ] Define the versioned semantic kernel registry and XLA FFI provider ABI.
- [ ] Implement the XLA conformance model over fixed-capacity tokens, masks, lengths, page tables, slot mappings, state
      buffers, weight-version inputs, and host-known outputs.
- [ ] Expose explicit exclusive state-buffer ownership and fail rather than silently copy when serving aliases cannot be
      honored.
- [ ] Implement device-resident HBM pools/tables, block mappings, asynchronous fences, and completion-safe page reuse.
- [ ] Integrate paged Gemma prefill/append/decode attention and fused cache append with one CUDA provider and a portable
      correctness fallback.
- [ ] Add kernel numerics, adversarial cases, standalone benchmarks, artifact/profile identity, and command-buffer
      capture validation.

Track exit: the XLA conformance model exercises the real ABI without hot-path synchronization; alias and reclamation
stress tests pass; paged Gemma kernels match dense reference behavior and are competitive on the declared target shapes.

### Phase 2 — ordered integration gates

The tracks converge through these gates in order. Passing a later gate cannot waive an earlier invariant.

1. **I1 — engine semantics.** The complete embedded request lifecycle passes against the Rust conformance executor,
   including mixed arrivals, cancellation, preemption, forks, tool pauses, weight changes, and injected failures.
2. **I2 — physical execution.** The same unmodified engine passes against the XLA conformance model. Traces prove there
   are no hidden host synchronizations, state copies, early page reuse, or profile-dependent semantic changes.
3. **I3 — real model, contiguous state.** Replace the XLA conformance model with Gemma 4 while retaining the same engine
   contracts. Reduced and full-checkpoint prefill/decode match the JAX reference using contiguous state.
4. **I4 — real model, paged state.** Gemma single-request and prefix-sharing cases use the production paged ABI and
   kernels. Paged logits/tokens match the dense reference; COW, cancellation, and reclamation stress tests pass.
5. **I5 — continuous Gemma engine.** Enable mixed prefill/decode continuous batching, chunked prefill, prefix caching,
   streaming, production sampling, and SLO policy on the full checkpoint. Sustained traces show stable memory, bounded
   starvation, explainable tail latency, and no per-token compilation.

Phase exit: the first declared Gemma 4 31B workload/SLO envelope is reproducible against pinned vLLM/SGLang/TensorRT-LLM
baselines. The first release may trail them, but every gap must be attributable to scheduler, kernel, communication,
memory, or launch evidence rather than unexplained runtime behavior.

### Phase 3 — RL rollout and weight lifecycle

Deliverables:

- [ ] Add versioned snapshot prepare/commit/drain/reclaim and cache invalidation.
- [ ] Implement trainer-to-engine disk and colocated transfer first; then collective/RDMA transfer as needed.
- [ ] Add deterministic rollout mode, full token provenance, async agent loop API, partial rollouts, and tool pauses.
- [ ] Add state fork/checkpoint/rollback for best-of-N/tree rollouts.
- [ ] Add sleep/offload/wake and explicit colocated role arbitration.
- [ ] Verify training/inference logits and logprobs under the selected deterministic policy.

Exit gate: an on-policy RL loop repeatedly updates weights and generates Gemma rollouts without process restart,
stale-cache reuse, unbounded memory growth, or unexplained logprob mismatch.

### Phase 4 — multi-GPU and advanced decoding

Deliverables:

- [ ] Tensor-parallel Gemma execution and page/state sharding with rank-consistent scheduling and launch IDs.
- [ ] Add pipeline, context, and/or attention data parallelism only where the target envelope requires it; defer expert
      parallelism until an MoE model is selected.
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
- speculative-friendly and speculation-hostile traces;
- MoE and dense models; one multimodal trace after the text engine is stable.

### Metrics

- p50/p90/p99 TTFT, TPOT/ITL, E2E, and queue delay;
- SLO goodput and per-user decode-rate floor;
- prompt/output/total tokens per GPU-second and per dollar;
- HBM usage, page fragmentation, prefix hit, COW, eviction/recall, and transferred bytes;
- host CPU, scheduler time, launch overhead, graph/profile hit rate, and compilation misses;
- speculative acceptance/effective speedup;
- weight-update pause, transfer, publish, and old-version drain time;
- numerical mismatch, determinism, and failure/retry rate;
- power/energy where hardware telemetry is reliable.

### Comparison rules

- Same model checkpoint, numerical mode, tokenizer, sampling, maximum lengths, prompt/output distribution, and SLO.
- Reduced-shape and conformance models are correctness/development tools and never support competitive performance
  claims; external-engine comparisons use the complete pinned Gemma 4 31B checkpoint.
- Warm and cold results are separate; compilation, model load, and prefix-cache warmup are reported.
- Compare optimized supported configurations, not deliberately weak defaults.
- Publish Pareto curves across concurrency/profile choices and all failure/unsupported cases.
- Treat project-reported numbers as hypotheses until reproduced in this harness.

## 19. Risks and explicit mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| XLA static semantics fight dynamic state | Copies, profile explosion, host sync | Rust-owned preallocated pools; static physical ABIs; aliased custom calls; explicit profile families. |
| Donation silently becomes copy protection | Catastrophic KV bandwidth/memory regression | Exclusive serving buffers and hard validation; donation is optimization, never state correctness. |
| Generic XLA kernels trail specialists | Poor decode/MoE performance | Kernel registry; vendor/community integration; Ryft-native kernels only where strategically valuable. |
| Too many profile combinations | Compile latency and memory blow-up | Geometric buckets, trace-driven profiles, bounded cache, AOT warmup, compatibility matrix. |
| Kernel/plugin ABI drift | Cache corruption or runtime failure | Versioned ABI, capability handshake, artifact identity, subprocess/startup validation, fallback/deny-list. |
| Prefix-locality policy starves requests | Tail-latency/SLO failures | Deadline/slack and aging bounds dominate locality after a configured threshold. |
| Weight update reuses stale state | Invalid on-policy rollouts or serving output | Weight version in every request/cache key; transactional publish; stale reuse unrepresentable by default. |
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
2. demonstrate a clear win on that beachhead—most plausibly long-context agentic RL plus online serving with shared
   Gemma model semantics, frequent weight updates, deterministic provenance, and high session-prefix reuse;
3. preserve architectural coherence while adding models, kernels, platforms, and deployment modes;
4. add Qwen3.6 27B without forking scheduler or lifecycle semantics, validating hybrid persistent state;
5. claim broader superiority only as the benchmark matrix earns it.

On that definition, building something **much better** is realistic. Building something immediately and universally
faster, more complete, and more production-proven is not.

## 21. Decisions to validate early

- [x] Select Gemma 4 31B dense instruction as the first production-model target and Qwen3.6 27B as the second-model
      extensibility test.
- [x] Use pure-Rust and real-XLA conformance executors to unblock engine work while preserving production contracts.
- [ ] Pin the exact Gemma checkpoint revision, tokenizer/template revision, NVIDIA hardware topology, BF16 context/batch
      envelope, memory budget, and serving/RL traces.
- [ ] Specify the reduced-shape Gemma 4 31B-derived configuration and parity fixtures.
- [ ] Decide whether the first paged-attention provider wraps FlashInfer/vendor code or is Ryft-native.
- [ ] Validate that XLA preserves the required state-pool aliases without hidden copy protection.
- [ ] Prove full-step command-buffer capture with the chosen custom calls and collectives.
- [ ] Choose page size/layout using measured prefill, decode, prefix sharing, and fragmentation tradeoffs.
- [ ] Decide whether model weights are ordinary executable inputs, captures, or external handles per platform.
- [ ] Specify deterministic/batch-invariant rollout semantics and allowable training/inference numerical differences.
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
- [SGLang paper](https://papers.nips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf)
- [SGLang/RadixAttention launch article](https://www.lmsys.org/blog/2024-01-17-sglang/)
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
