# Ryft inference engine

**Design review, November 20, 2026.** The first milestone is an embedded engine that loads the released Inkling 2 Mini
checkpoint, takes text and image inputs, streams generated text, and returns the scores needed for rollout replay.
The companion inventory specifies the model math, artifact contract, differentiable reference and training kernels.
[Model inventory](plan-inkling-2-mini.md) · [First milestone](#17-first-milestone-and-execution-plan) · [Benchmark and acceptance](#18-benchmark-and-acceptance-strategy)

## 1. Architecture decision

Use one Rust scheduler around a small family of compiled Ryft/XLA model steps. Keep request admission, cancellation,
streaming, state ownership and policy publication on the host. Keep the weights, physical page tables, active row
metadata, expert routing and next-token feedback on device across steps. Compile profiles for bounded physical shapes
and pass actual lengths and routing decisions as data. Reuse Ryft's kernel infrastructure for accelerator-specific providers.

Start with the ordinary full-sequence model that the trainer can differentiate. The same layer graph drives scoring,
contiguous generation and paged execution. At a layer visit, the executor binds the attention and convolution history
for that visit and the parameters for that physical layer. This lets an optimized worker share the learned model with
the trainer while owning its own state layout. Numerical agreement between execution modes is an acceptance test.

First generate from the real checkpoint, then bring the same path under continuous admission with paged state.
In parallel, test the actual accelerator provider for buffer aliasing, workspace lifetime, graph replay and overlap.
These proofs decide which optimizations the engine can advertise. Measure the online service by completions meeting
latency objectives and the post-training system by useful rollouts accepted after each update.

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
- Inkling 2 Mini as the first production model, with reduced fixtures for CI and the published mixed-format
  checkpoint, a text prompt and an image-conditioned prompt required for milestone acceptance.

### Non-goals for the first usable release

- Supporting every Hugging Face architecture, accelerator, quantization, and decoding feature.
- Peak 1,048,576-token operation, DSpark and distributed serving in the first engine milestone. The training
  workstream supplies the shared model, importer and scorer from the start.
- Replacing Kubernetes, a cluster scheduler, or an RL training framework.
- Inventing all GPU kernels in Ryft before using proven vendor/community kernels behind stable interfaces.
- Using bounded-dynamic XLA values on the decode hot path where they introduce synchronization or multi-device limits.
- Making Rust or XLA an externally visible requirement of the serving protocol.
- Claiming performance leadership without reproducible SLO-constrained comparisons.

## 3. What the leading systems teach us

The initial survey dates to August 2026, with updates in September and October for SkySynth, ZML/LLMD, DeepInfer
and MAX. The November review adds Baseten, recent post-training results, and model-matched Mini comparisons.

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

### SkySynth and ZML/LLMD: implications for this plan

SkySynth is a synthesis workflow, not a released general-purpose inference runtime. Its reported 2.2x result concerns
Qwen3-4B on one L4 with heavily shared long prefixes. The inspected public example pins two 24,576-token roots with
32 branches each, warms the roots outside timing, and uses random token prompts. Its published evaluator and task
specification do not include the generated engine or sufficient baseline artifacts to reproduce the headline result.
Treat this as motivation for shared-prefix decode experiments, not a Ryft target or evidence of general superiority.

The important mechanism is grouped cascade attention: prefix caching avoids repeated prefill and duplicate storage,
whereas grouping queries can also avoid repeatedly reading shared prefix KV during decode. Hydragen explains this
mechanism; FlashInfer already exposes cascade primitives. Baselines must enable their applicable optimized paths.
SkySynth's reported evaluator failures also motivate correctness tests on the actual long-context optimized path and
independent checks of token accounting and first-token delivery.

ZML/LLMD is a close compiler-backed architectural peer. Its alpha advertises five accelerator targets, Gemma DFlash,
and direct remote model loading. These are useful directions, but neither platform coverage nor its up-to-10x per-user
DFlash claim establishes comparative SLO goodput. The launch tables lack a matched non-speculative comparison and a
complete workload/configuration manifest. Add it as an optional measured peer when a pinned artifact supports our
contract; retain the established baselines. The concrete additions below are parallel-draft capability, bounded model
loading, deployment artifacts, and startup/streaming measurements. They do not expand the initial hardware scope.
The inspected ZML source revision contains compiler and loading infrastructure, but no LLMD server or DFlash
implementation; its LLM CLI is not evidence for the server's scheduling or feature-combination correctness.

### DeepInfer: execution mechanisms and release boundary

The October 20 review pins DeepInfer to `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`. The release supports
DeepSeek-V4.1-Flash offline execution on NVIDIA SM100 and Ascend 950; the complete cache service, stage disaggregation,
and replica load balancing are not included. Its published prefill/decode curves use different deployments: four and
64 SM100 GPUs respectively. These are specialized execution measurements, not a matched serving comparison with Ryft.

The useful additions are confidence-aware verification budgets, explicit exact-versus-approximate state reconstruction,
time-budgeted scheduling, and topology-aware MoE experiments. Existing async execution, portable provider, numerical
contract, and state-service designs remain appropriate. DeepInfer's deployment guidance also motivates measuring cache
lookup/fetch costs independently and comparing locality routing with shared storage. Its companion `deepseek-recipe`
illustrates a separate protocol/conversation/token adapter boundary. Use those mechanisms where the measured Mini workload and deployment topology justify them.

### Baseten DeepSeek Ultrafast and the cost of a decode step

Baseten's November release specializes Rust and CUDA for DeepSeek-V4.1-Flash on Blackwell. Baseten reports 2,060
output tokens/s on eight B300s at 2.71 ms per speculative verification step and about 5.6 emitted tokens/step. The
decode test used one coding prompt with thinking disabled. A separate eight-GPU prefill processed 1M tokens in
11.7 seconds. The public server runs one generation job per rank group at a time, fixes temperature and top-p at
one, and rejects logprob requests. Use the release to study single-stream latency. Establish Mini throughput under
concurrent arrivals and rollout scoring independently. The source is published under the Elastic License 2.0, which
restricts providing its functionality as a managed service. Plan independent Ryft implementations.
[Benchmark and method](https://github.com/basetenlabs/deepseek-ultrafast/blob/76766a97308bfd1658fb055e835d0f92de737dc3/README.md)
· [Worker loop](https://github.com/basetenlabs/deepseek-ultrafast/blob/76766a97308bfd1658fb055e835d0f92de737dc3/crates/server/src/engine.rs#L205-L213)
· [Server contract](https://github.com/basetenlabs/deepseek-ultrafast/blob/76766a97308bfd1658fb055e835d0f92de737dc3/docs/SERVER.md)
· [License](https://github.com/basetenlabs/deepseek-ultrafast/blob/76766a97308bfd1658fb055e835d0f92de737dc3/LICENSE)

- **Launches.** Baseten's separate optimization report describes fusion, CUDA Programmatic Dependent Launch (PDL),
  and breaking variable-length prefill into capturable segments in SGLang. For each Mini profile, compare ordinary
  compiled execution, full fixed-shape decode replay, and prefill segments split where attention metadata changes.
  Try PDL only inside a provider that controls both kernels and their data dependency. Measure GPU idle time, launches,
  workspace, TTFT, and inter-token stalls. Keep the simplest path that improves end-to-end performance. When adding a
  drafter, report step time and verified tokens emitted per step separately from output rate and streaming burst size.
  [Baseten optimization report](https://www.baseten.co/blog/how-to-make-flash-models-even-faster-optimizing-deepseek-v41-flash-1017-fast/)
- **Resuming a conversation on another replica.** Ultrafast retains compressed DeepSeek state in host memory.
  Redis records ownership with restart epochs, and replicas fetch compatible snapshots over host-to-host RDMA.
  For Mini, measure the actual bytes for full-attention prefixes, local windows and convolution history. At readmission,
  compare owner queue delay, remote lookup and transfer plus device upload, and local replay. Validate prefix identity,
  state format, policy revision and ownership epoch before installing fetched state. Inject stale directory entries,
  partial transfers and departed owners. A cache miss must permit replay from the recorded inputs. Enable transfer
  only on measured wins after accounting for congestion and tail latency.
  [Cache protocol](https://github.com/basetenlabs/deepseek-ultrafast/blob/76766a97308bfd1658fb055e835d0f92de737dc3/docs/SERVER.md)

### A numerical contract for the rollout and learner

Google's November Tunix prototype compared the sampled action's logprob at identical weights across incremental
decode, serving prefill and the differentiated trainer forward. Its bitwise result covers Qwen3-8B on 64 v5p TPU
chips, with MoE support under development. The technique fixes accumulation, casts, cross-shard summation and the
scoring tail, then supplies backward rules for the shared forward. Baseten's Serverless Loops separately demonstrates
publishing a trained adapter to both a sampler and a production endpoint, with an end-to-end RL evaluation alongside
numerical tests. These sources motivate two implementation gates.
[Tunix alignment method and evaluation](https://developers.googleblog.com/unlocking-stable-on-policy-rl-on-tpu-with-bitwise-zero-traininference-mismatch/)
· [Serverless Loops](https://www.baseten.co/blog/introducing-serverless-loops/)

1. Replay the exact recorded action and preceding token IDs, image inputs and masks through all three paths at one
   immutable policy revision. Compare raw model and transformed behavior logprobs separately, including across mixed
   batches, prefill chunks and cache hits. Record the first divergent layer, loop visit and expert choice. Offer a
   bitwise mode only for a kernel and topology combination that passes byte comparisons. Report measured logprob error
   and performance for each other supported mode. Measure weight lag separately by replaying the same actions under
   the learner's current revision.
2. Make a trainer update produce a typed snapshot with an immutable base and adapter or full-weight revision. Stage
   and validate it on all participating workers before publishing. Test rollback after a partial install, old sessions
   continuing on their pinned policy, and a new session seeing the updated weights. Measure time from optimizer step
   to the first trainer-accepted rollout and useful action tokens per GPU-second under an explicit staleness bound.

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

## 4. What Ryft can run today

The current checkout has enough machinery to start a differentiable reference model and a contiguous decode path.
The first full Inkling run also needs a checkpoint importer, the released processor and token conventions, a model
implementation, and persistent sequence state. These are separate deliverables from tuning or production serving.

| Reusable today | Where it helps | Evidence |
|---|---|---|
| Typed, traversable parameter trees, reverse-mode AD and loss operations | Share weights between training, scoring and decoding. An MLP example trains with gradient descent. A synthetic F32 GPT benchmark passes weights as runtime inputs to a compiled forward pass. | [Parameter trees](crates/ryft-core/src/parameters.rs), [losses](crates/ryft-core/src/operations/softmax.rs), [MLP](crates/ryft/examples/mlp.rs), [transformer benchmark](crates/ryft/examples/transformer_benchmark.rs) |
| Reference attention, selected cuDNN forward/backward paths and sampling primitives | Build an oracle from dense math. A CPU fixture compiles a fixed decode loop and compares greedy, top-k and custom-call attention paths to a reference. The cuDNN configuration uses dense state, with one segment per row. | [Attention semantics](crates/ryft-core/src/operations/attention.rs), [cuDNN eligibility and ABI](crates/ryft-xla/src/operations/attention.rs), [decode fixture](crates/ryft-xla/tests/test_decode_loop.rs) |
| Rank-local array I/O, resharding and collectives | Load each process's checkpoint slices into its addressable shards using `Array::from_host_shards`. Await local uploads before publication. Reuse the mesh and ragged all-to-all contracts when qualifying expert parallelism. | [Host shard I/O](crates/ryft-xla/src/arrays/host.rs), [resharding](crates/ryft-xla/src/arrays/resharding.rs), [collectives](crates/ryft-core/src/operations/collectives/mod.rs), [distributed runtime](crates/ryft-xla/src/distributed.rs) |
| Retained compilation, executable persistence, distributed artifact exchange and profile-guided recompilation | Prewarm physical profiles. `CompiledFunction::batch` compiles a wrapper with an explicit static batch extent. `ExecutableFunction::execute_async` returns outputs and a whole-invocation fence. | [Compiled functions](crates/ryft-xla/src/compilation/functions.rs), [persistence](crates/ryft-xla/src/compilation/persistence.rs), [artifact exchange](crates/ryft-core/src/compilation/exchange.rs), [profile-guided replacement](crates/ryft-xla/src/compilation/profile_guided.rs) |
| Kernel authoring, host verification, explicit derivative rules, AOT artifacts and schedule tuning | Stage a verified kernel with a VJP or portable fallback. Select and qualify a provider for each model shape, precision and mode. Existing compiler adapters have different supported operation sets. | [Kernel contracts](crates/ryft-kernels/src/lib.rs), [XLA staging](crates/ryft-xla/src/kernels/staging.rs), [AOT](crates/ryft-xla/src/kernels/aot.rs), [tuning](crates/ryft-xla/src/kernels/tuning.rs) |
| PJRT buffers, transfers, completion events, custom-call aliases and runtime extensions | Pass native resources through an execution context and retain their owners until completion. Use these contracts to prototype paged attention and a sequence-state pool. | [Buffers](crates/ryft-pjrt/src/buffers.rs), [transfers](crates/ryft-pjrt/src/transfers.rs), [events](crates/ryft-pjrt/src/events.rs), [custom calls](crates/ryft-core/src/operations/custom_call.rs), [runtime extensions](crates/ryft-xla/src/sessions/runtimes.rs) |

### Constraints on a serving executor

- **Batch construction.** The host must map admitted requests to slots in a precompiled profile. Bounded-dynamic
  arrays can cause host readback or repadding. Retained input-bound bucketing currently requires a single device.
  Use static physical buffers plus device lengths and active masks for the multi-device hot path.
  [Bucketing options](crates/ryft-xla/src/compilation/options.rs) and [execution](crates/ryft-xla/src/execution.rs).
- **State ownership.** An external reference uses read leases and pending mutation generations. Mutations commit
  returned final values. The reference input is never donated and its output alias is a reuse hint. Ordinary array
  donation can also be downgraded when any shard buffer is shared. Static references support sharded multi-process
  meshes, with independent local commits and no cross-rank rollback. Bounded-dynamic reference inputs require
  read-only replicated state on a fully addressable single-process mesh. Prove arena address stability, copy volume
  and safe disjoint-page overlap with the chosen kernel before depending on them.
  [External reference ABI](crates/ryft-xla/src/compilation/functions.rs) and [execution](crates/ryft-xla/src/execution.rs).
- **Distributed launches.** The XLA execution wrapper forwards the extension-provided context and retains resources
  on the fence. It currently submits launch ID zero and defaults for device override and incarnations. The serving
  executor needs rank-consistent launches, error propagation and a single publication decision after all ranks are ready.
  [XLA submission](crates/ryft-xla/src/execution.rs) and [PJRT executable API](crates/ryft-pjrt/src/programs.rs).
- **Missing product surfaces.** The workspace has no released-model importer, Inkling renderer/image processor,
  resumable optimizer/training-checkpoint workflow, or transactional weight publisher. Continuous admission, paged
  attention and allocation, prefix accounting, request-local sampling and stopping, streaming and rollout provenance
  also require implementation. Adaptive replacement of a compiled executable concerns its code, independently of the
  model weights bound to an invocation. [Workspace](Cargo.toml) and [profile-guided compilation](crates/ryft-xla/src/compilation/profile_guided.rs).

Keep page leases, logical sequence versions and cancellation in the inference runtime. The executor binds a validated
weight snapshot and state leases to compiled Ryft programs, tracks the resulting fences, and publishes or discards
each step's logical state. The model and trainer can then use the same math with ordinary arrays.

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

Start with a model package and one embedded inference package. Keep scheduler, state and XLA executor as internal
modules until they have independent consumers. Extend the existing [kernel crate](crates/ryft-kernels/src/lib.rs)
and its [XLA integration](crates/ryft-xla/src/kernels) for serving providers. The names below describe ownership.
Version independently distributed provider ABIs and persisted formats.

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
  with controlled state transitions and failure injection.

### Serving providers using the existing `ryft-kernels`

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
requires replay of token IDs and immutable, resolvable media inputs at their original positions. Re-encode images if
encoder weights or preprocessing changed. Record the policy boundary. If inputs have expired, return an explicit
expired continuation. A request and its prefix cache bind to a model lineage and immutable weight version.

A weight update is transactional:

1. prepare and validate a complete sharded snapshot;
2. transfer it using disk, host, IPC, collective, or RDMA provider;
3. make the snapshot executable-ready on every rank;
4. atomically publish the new version for new requests;
5. drain or retain old-version requests according to policy;
6. reclaim the old snapshot only after all requests and asynchronous launches release it.

For Mini, the main mixed-format checkpoint is about 19.5 GB on disk. Two such images are about 39 GB before runtime
packing, workspaces, state and staging. At 2 bytes per parameter, a hypothetical 26.4B all-BF16 representation costs
about 52.8 GB per copy. Actual mixed-dtype resident bytes depend on the loader and providers. Budget the overlap of
old and new weights, both versions' live state, quantization/repacking, any graph-bound addresses, activations and
transfer buffers. Validate against measured peaks on each rank. Prefer atomic snapshot swapping when it fits, share
immutable base weights for adapters, and otherwise measure drain/offload/reload. A changed weight address may require
graph rebind or recapture even when the compiled program is reusable.

Make each update idempotent by lineage, revision and content digest. Stage against an expected parent and abort a
partial installation before admission sees it. Publish only after all participating ranks acknowledge the same
manifest. An uncertain rank stays out of service until recovery reconciles its epoch. Requests, graphs in flight and
paused sessions hold leases on the old snapshot until completion or explicit reconstruction.

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

### Inkling state layout

Index weights by physical layer and state by the logical visit in the fixed 48-visit schedule. The 40 local visits
retain bounded KV and the eight full visits retain their full prefixes. All 48 visits also require the appropriate
causal convolution tails. Keep loop messages and attention-residual block summaries in token-step or prefill-chunk
workspace with the lifetime required by the graph. They do not grow with the decoded history. Immutable image features
may have their own cache. Include pixel-content identity, preprocessing revision, insertion positions, and encoder
weight version when reusing them or any downstream prefix. Partition prefix reuse by the authorized sharing domain.

For head dimension 128, eight KV heads and BF16 state, each visit stores 4,096 KV bytes per retained token per sequence.
Ideal KV bytes at length L are 4,096 × (8L + 40 min(L, 1,024)). At L = 4,096 this is 288 MiB, and at the published
maximum it is 32 GiB + 160 MiB. Add conv tails, page rounding, provisional writes, tables, allocator reserve and COW.
A context-length claim alone cannot set the admission limit. A publishable prefix or transfer is one coherent
frontier across global pages, each local window and its matching tails. Forks can share immutable pages. Capture or
replay the bounded histories before their next overwrite. [Visit and cache semantics](https://vllm.ai/blog/2026-11-20-inkling2-mini)

### Reconstruction semantics

A reconstruction strategy declares whether it restores exact model state or changes the computation. Exact suffix
replay requires a valid starting checkpoint and all dependencies; a sliding-window size alone does not bound the
receptive field through multiple layers. DeepInfer's SWA bounded replay truncates attention within the replayed segment
and is explicitly approximate. For Mini, measure the cost of restoring each visit’s full prefix, local window and convolution history.

Keep exact restoration as the default. Any approximate strategy is an opt-in model capability with a named quality
policy, cache/provenance identity, and separate acceptance results. Reconstructed approximate state cannot enter an
exact-state cache or be represented as the unchanged training policy. Test long-context retrieval, repeated reuse,
logits/logprobs, and downstream task quality against full recomputation. Declare incompatible modes such as prompt
logprob scoring explicitly rather than returning partial or approximate scores under an exact contract.
Budget ring-buffer capacity for the window, maximum speculative overwrite/backtracking, and in-flight accesses;
preserve a completed checkpoint before overwrite.
Validate wraparound and partial acceptance at checkpoint boundaries, including delayed copies and cancellation.

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
- `PrefixKey` includes the authorized sharing domain, token IDs and positions, inserted media digests and spans,
  processor/renderer revisions, model, weight and adapter revisions, state layout and numerical configuration.
  Encoder and prefix-state hits are validated independently.
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

For remote tiers, measure metadata lookup, storage fetch, host staging, and device swap-in separately. A metadata hit
is not resident state. Compare local-affinity routing with a shared cache accessible to all workers under the same
capacity and bandwidth budget; shared access may reduce routing imbalance but adds network and storage failure modes.
Choose retention by measured reuse distance and avoided recompute cost, including storage write amplification and
weight-version invalidation. Do not adopt multi-day retention or a dedicated storage cluster without that evidence.
Layer-wise prefetch may overlap compute only when readiness fences and leased staging buffers prevent early use/reuse.

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

### Measured execution-cost budgets

Refine token/page limits with a calibrated execution-time estimate using query tokens, existing context, state format,
phase, and device topology. In Phase 2, compare fixed chunks with chunks chosen to fit a latency budget; retain hard
memory/token limits, prediction-error margins, and a conservative fallback when estimates drift. Report cost-model
error and retraining overhead, not only average chunk utilization. Deadline and fairness rules remain authoritative.

For synchronized worker groups, budget the slowest participating rank and preserve collective ordering even for empty
local work. Decode routing should consider both active sequence count and estimated attention bytes/work, rather than
assuming equal request counts produce equal work. Test heterogeneous context lengths and skewed arrivals before
replacing the simpler policy. Model- and hardware-specific fits are observations, not universal scheduling constants.

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
weight format and layer-visit schedule, state layout, phase, token and sequence buckets, topology/sharding, kernel selections, speculation mode,
determinism mode, compiler identity, and relevant XLA flags.

Start with a small geometric profile family. Collect misses and padding waste, then tune bucket boundaries from traces.
Do not compile an unbounded Cartesian product.

Record each optimized profile's validity assumptions and measured performance envelope: model/numerical semantics,
shared-prefix length, group occupancy, suffix lengths, and supported masks/layouts. Group membership, page addresses,
and logical lengths remain runtime metadata within bounded capacities; request identities and prefix contents do not
become compilation keys. Choose among warmed variants and ordinary paged fallback as traffic changes. Charge grouping,
provider planning, scratch space, and metadata upload to the optimization; reject per-prefix or per-token compilation.

### Profile selection for serving and rollouts

Use aggregated execution, one rank and ordinary autoregressive decode first. Warm a batch-one decode profile to favor
inter-token latency, larger packed profiles for throughput, and a teacher-forced score profile that gathers only the
requested action scores. Share weights and immutable compatible state across requests on the same revision. Give
interactive requests a deadline and rollout work a bounded token/time quota. Fill spare slots with rollout work only
when the predicted step meets the online deadline. With separate pools, measure the cost of replica duplication.
For bulk rollouts, compare a bounded device-resident multi-step loop against continuous steps, charging wasted work
at stop/tool boundaries and the delay before publication of the next policy.

### Command buffers/graphs

XLA already performs command-buffer/CUDA-graph extraction for supported thunks. Ryft must verify, per profile, that the
whole step—including custom calls, collectives, cache mutation, and sampler—is capture-safe. The kernel registry records
graph safety and stable workspace requirements. A profile that cannot capture remains correct but is not accepted for a
latency target until measured.

Where tensor parallelism is enabled, stress captured collectives under allocator pressure, weight refits, and buffer
retirement/reallocation. Stable virtual addresses alone do not establish that captured communication registrations or
mappings remain valid. Verify the actual backend's lifetime requirements and invalidate/rebuild captures when needed;
do not transplant another runtime's compiler flags without reproducing its failure case.

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

### Shared-prefix decode attention

In Phase 2, evaluate grouped cascade attention using a validated provider before implementing a native kernel. Start
with independent shared-prefix groups and private suffixes; add hierarchical groups only when traces justify them.
Compute attention for disjoint prefix/suffix partitions and merge their outputs using their log-sum-exp normalizers.
This preserves the mathematical softmax result; floating-point agreement still requires explicit tolerance tests.
Never reuse query-dependent attention outputs across requests or average partition outputs without normalization.

The scheduler supplies group-to-query mappings and partition/page boundaries from valid, version-compatible cache
entries. Locality remains subordinate to deadlines and bounded starvation. Validate positions, causal/window masks,
GQA, Mini's position-dependent logits, and the visible region for each logical visit. Merge the query-dependent
sink exactly once across all partitions and use global query/key positions when constructing relative bias. Test
against ordinary unpartitioned attention, including the transition out of the local window.
Retain ordinary paged attention for unsupported semantics or unfavorable occupancy/suffix lengths. Include empty and
fully masked partitions in correctness coverage. Measure KV traffic and complete step latency alongside serving
outcomes: fewer prefix reads do not guarantee a win when weight traffic or dense computation dominates.

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
  so cancellation and rejection do not silently perturb unrelated requests. Partition draws by sampling purpose
  (proposal, acceptance, correction, ordinary sampling), independently of physical batch rows or compaction.
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

Test reproducibility across changes in batch membership, sequence order, profile buckets, and speculative lengths.
Feature composition is explicit: DeepInfer's documented deterministic mode disables variable-length verification.
Ryft must either prove the combined contract with stable RNG draw identities or reject/fall back for that combination;
individual deterministic kernels and a request seed do not establish whole-engine reproducibility.

### Structured output

Compile JSON Schema/regex/EBNF into cached automata. The request carries only automaton state. Token masks should move to
the device once CPU masking becomes visible. Deterministic spans may fast-forward multiple tokens, but the scheduler
accounts for their state/pages exactly like speculative work.

### Speculation

Use a protocol, not one draft-model flag:

- proposer: n-gram, prompt lookup, draft model, Medusa/EAGLE/MTP, block diffusion such as DFlash, or user tokens;
- proposal shape: chain or tree;
- verifier: target-model acceptance policy;
- state checkpoint: fork, commit accepted prefix, roll back remainder;
- adaptive controller: enablement and depth from acceptance, concurrency, and memory cost.

Speculation is allowed to turn itself off. It often helps low-concurrency latency and can hurt saturated throughput.

Represent parallel block drafting without forcing it through an autoregressive draft loop. DFlash conditions a separate
drafter on target-model features and proposes a block in one forward pass. Provide optional, explicitly selected target
feature outputs, bounded feature buffers, and draft state in the existing execution/state contracts. Ordinary decoding
must not materialize these features. Feature buffers follow the same accepted-frontier, rollback, and completion-safe
reclamation rules as draft state. Pin draft/target checkpoint compatibility and invalidate stale features on weight
changes; an RL update may require disabling or recalibrating the drafter.

For Mini, use ordinary target decode as the control and qualify its supplied three-layer DSpark drafter after M1.
Capture the running partial block residual immediately before logical visits 5, 23 and 41 (zero-based) and the final
normalized hidden state at boundary 48. Rebuild or invalidate draft feature state on policy changes and fork/rollback.
Use block proposals and target verification with fixed budgets first. [Target tap locations](https://github.com/vllm-project/vllm/blob/4c616217b466d94c2dcce8fd2d0c346cc48a2bed/vllm/models/inkling2/nvidia/loop.py#L347-L397) · [Release integration](https://huggingface.co/blog/inkling2mini#speculative-decoding)

Prove the verifier's declared greedy or stochastic distribution contract, including grammar/logits processing; do not
infer stochastic exactness from greedy agreement or from a method's headline claim. Charge feature extraction,
draft weights/state, verification, rejected work, and reduced serving capacity to effective speedup. Adapt block size
and enablement using measured latency, acceptance, concurrency, and memory limits.

### Confidence-aware verification budgets

Extend adaptive speculation to choose per-request verification prefix lengths under a shared token/time budget.
Where the proposer supplies calibrated conditional acceptance estimates, use cumulative prefix survival and a measured
whole-step cost curve, including sampler time and context-dependent timing corrections, to estimate accepted tokens
per unit time. Preserve prefix closure: verifying a later proposal requires its predecessors. Confidence changes
resource allocation, never the target verifier's acceptance rule.
Prove that selection does not bias the target distribution: later draft outcomes/confidences must not improperly
influence whether earlier tokens are verified. Track the provenance and age of feedback used for budget decisions,
including overlapping steps; prefix closure alone is insufficient. Add statistical tests against fixed verification.
Without a suitable confidence signal, retain fixed-length or empirically tuned verification; do not require a new
confidence head in every model or confuse token probability with calibrated acceptance probability.

Use fixed-capacity verification profiles with device-resident logical lengths and mappings. The host may select a
capacity from completed historical statistics while device work finalizes current request lengths from ready signals;
explicit dependencies and sequence generations prevent stale feedback after slot reuse. Charge padding, ranking,
packing, and synchronization to the decision; keep mandatory target progress and fairness for low-confidence requests.
Compare adaptive and fixed verification independently of draft-model choice across low and high concurrency. Revalidate
calibration after weight updates, grammar changes, or workload shifts, and disable adaptation when it loses goodput.

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

A rollout carries an append-only transcript of token IDs, image bytes or immutable references and digests,
preprocessing revision and insertion positions, action mask, stop/truncation reason, old policy revision, numerical profile and per-action scores. Preserve tool/environment observations as inputs
with zero action weight. Retokenizing streamed text can change boundaries. Keep independent slots for behavior, raw
old-policy, current learner and optional reference-policy logprobs. The trainer owns rewards, advantages, objectives
and gradient steps. The engine returns replayable inputs to SFT/distillation scoring and sampled continuations for RL.

At unchanged weights replay the very same actions through incremental decode, batched score and the differentiated
trainer forward. Compare action-aligned logprobs and first divergence by visit, routing and numerical mode. Check
importance ratios against one for matching distributions before an update. After a real update, test expected changes,
old-session isolation and policy-lag limits. Budget colocation on simultaneous trainer and worker peaks, including
moments, activation checkpoints, old weights and live state. Explicitly fence sleep, release and wake transitions.

Qualify optional router replay with expert IDs per token **per logical visit**. For replay of top-k/top-p sampling,
retain the chosen support, ordering and transforms to reproduce the behavior-policy probability. A current-policy
importance ratio uses the current policy's own support and normalizer. An old sampled action outside the current
support has zero current probability. Newly admitted actions lack coverage under the old policy. Label a deliberately
fixed-support or forced-route diagnostic separately. Charge replay metadata to storage, memory and transfer budgets. Use
SkyRL's published Mini BF16 full-tuning, adapter and quantized-rollout configurations as direct comparison cases.
[SkyRL Mini integration](https://docs.skyrl.ai/docs/models/inkling2#inkling-2-mini)

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

### MoE topology and expert placement

Mini requires top-8 routing and grouped expert computation on the first worker. Evaluate attention data parallelism
with expert parallelism and phase-specific layouts once observed capacity, expert skew or throughput justifies another rank.
Sweep group size within the actual fabric; a larger expert group can improve expert batches while increasing dispatch,
synchronization, and failure-domain costs. Compare specialized fused dispatch/compute/combine providers with unfused
paths, including graph compatibility and simultaneous communication/workspace memory. Reuse semantic provider contracts.

Measure per-expert demand and slowest-rank/mean step time. Add redundant hot-expert placement and periodic rebalancing
only when skew justifies the extra weight memory and transfer cost. Version placement maps; fence weight moves and
publish a consistent map across ranks before new dispatches can use it. Benchmark routing drift, cold experts, and
rebalancing intervals including their stalls. On rank failure, recover the worker group coherently; changing EP size
requires a newly validated topology/profile and state handoff, not an in-place change to outstanding collectives.

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

Keep protocol conversion, a canonical conversation representation, model-specific prompt/token encoding, and streamed
output parsing separate from execution. Use the pinned Inkling renderer and image processor for the first model.
Version templates with the tokenizer; test tool/reasoning boundaries, stop sequences split across chunks, token usage,
and round-trip protocol fixtures. User text resembling control tokens must remain text unless explicitly encoded as
trusted structure. Rendering a schema in a prompt does not enforce it: structured decoding remains an engine contract.

### Model loading and deployment artifacts

Use a small model-source interface for local files and later Hugging Face/object-store range reads. Resolve immutable
checkpoint/tokenizer revisions before loading; verify tensor names, shapes, dtypes, and shard completeness before
publication. Bound host staging and concurrent reads, load required shards directly where supported, and make partial
loads cancellable and reclaimable. Remote access still transfers bytes; distinguish avoided intermediate copies from
network, host-to-device, resharding, and conversion costs. Use the weight-publication protocol specified in section 7.

Ship target-specific runtime/provider artifacts with pinned compiler, PJRT, kernel, and driver compatibility metadata.
Document compilation and warmup requirements separately from binary packaging; a self-contained server does not imply
all model profiles are precompiled. Readiness requires loaded weights and validated warmed profiles. Keep the initial
NVIDIA package small and reproducible; expand accelerator support through a model/feature/dtype/topology capability
matrix with numerical, mutation, capture, and completion tests on each claimed platform.

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

### P0 — physical feasibility and first competitive profiles

Sections 16 and 17 use different gates. Correct model output and safe state publication qualify an execution path for
M1. Graph replay, lookahead, and copy/idle budgets qualify a profile for competitive performance.

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

### P1 — production scale

- [ ] Extend the proven local execution protocol with distributed execution contexts, worker incarnation IDs, and
      rank-consistent recovery; callbacks and completion-safe cancellation are already P0 prerequisites.
- [ ] Complete multi-slice execution configuration and rank-consistent error propagation.
- [ ] Build sharded hot weight replacement and adapter updates without recompiling unchanged executable profiles.
- [ ] Optimize Mini’s mixed NVFP4/MXFP8 grouped expert path, then evaluate quantized KV, fused sampling and collectives.
- [ ] Add layerwise/streaming state transfer with compute/communication overlap for P/D deployments.
- [ ] Extend or avoid multi-device bounded bucketing based on evidence; fixed physical profiles remain the default.
- [ ] Expose per-operation/profile tracing and scheduler-consumable cost observations.

### P2 — additional platforms and tuning

- [ ] Capability-parity plans for CUDA, ROCm, TPU, CPU, Metal/other PJRT targets.
- [ ] Profile-guided background retuning and compatible executable replacement using real serving traces.
- [ ] Automated kernel/profile regression isolation and safe rollback.
- [ ] Evaluate the existing static compiled-function batching wrapper for homogeneous offline rollout profiles.

## 17. First milestone and execution plan

**M1 is end-to-end Inkling 2 Mini inference on the released checkpoint.** A caller supplies messages (including an
image), receives streamed tokens through the embedded Rust engine, and can score the generated continuation. The first
slice is single-request text generation. The M1 exit adds image input and continuous batching on one worker group.
Target ordinary autoregressive generation with the published expert quantization semantics. DSpark, HTTP, long-context
leadership, expert parallelism, and a complete RL system have their own gates.

Use the full checkpoint identified in the [model and training plan](plan-inkling-2-mini.md). Its importer, renderer,
fixed visit schedule, vision encoder, parameters and teacher-forced scorer are shared with the trainer. Reduced fixtures
shrink widths/expert counts and sequence lengths while preserving all three passes, the six-layer attention pattern,
independent histories, four message reads, both expert formats, and image insertion.

Use the pinned Transformers implementation for structure and compatible intermediate tensors and NeMo for BF16
training gradients. For M1 record teacher-forced and incremental target scores from a pinned release-numerics vLLM
or SGLang build, or a reference whose rounding has been validated against it. Their route-weight placement can differ
in floating paths. Freeze container digests, settings and fixtures. [Model reference contracts](plan-inkling-2-mini.md#artifact-import-and-training-numerics)

### Phase 0 — model bring-up and physical feasibility in parallel

- [ ] Start the model plan's differentiable reference and import work immediately. Run a reduced token-to-loss step
      and materialize a full-checkpoint first-token fixture. Stage the renderer, preprocessing code and small fixtures
      with the same revision as the weights.
- [ ] Start on one NVIDIA Blackwell worker in aggregated mode. Proposed first experiment is one B200, at most four
      sequences, 4,096 combined positions per sequence, including image tokens, with generation capped at 128. Budget
      pool capacity from the actual live groups and lengths. Record precise hardware, available memory, topology, and
      software before the first run. Expand beyond this envelope only with measured headroom.
- [ ] Register separate parity, interactive-serving, saturated and rollout traces and section 18's thresholds before
      tuning. Pin the supported vLLM and SGLang Mini recipes. Include unsupported precision/feature combinations in the
      report. A DeepSeek-only implementation supplies mechanism experiments.
- [ ] Exercise provisional contracts with a deterministic Rust executor, virtual clock, out-of-order completions and
      cancellation. A tiny compiled XLA model exercises real aliases, immutable per-launch metadata, fences and device
      token feedback. Both use the same admission and completion code as the full model.
- [ ] Probe an actual paged CUDA attention provider with Mini's 24:8 head layout, learned relative scores and sink,
      local and global visits, mixed lengths and conv-state updates. A composition can supply missing semantics during
      bring-up. Inspect HLO, buffer assignment and GPU traces across repeated launches and profile changes. Test
      disjoint pages and dependent uses, ring wrap, partial writes, cancellation after submission, and unknown device
      quiescence. Measure hidden copies, whole-pool serialization, graph compatibility and metadata-slot lifetime.

Start with functional contiguous buffers and serialized completion wherever necessary to obtain correct tokens.
Stabilizing a shared mutable arena requires a proven physical ownership and completion mechanism. Adding overlap,
graph replay or a performance claim requires the corresponding trace and fault evidence. These experiments can
continue while the reference model generates its first tokens.

### Phase 1 — full-checkpoint generation, then continuous serving

| Integration gate | Deliverable | Acceptance |
|---|---|---|
| I1. Full-sequence reference | One model definition, trainable reduced instance, full released weights, renderer and scorer. | Match layer taps, selected experts, loop boundaries and next-token scores against pinned fixtures. Import the tensors by physical layer exactly once. |
| I2. First generated text | Render a conversation, prefill contiguous state, decode, sample, detokenize and stop via the public embedded entry point. | Match incremental scores to full teacher forcing after every token on margin-separated greedy fixtures. Cross a chunk split and the local window. Distinguish the three core visits by deliberately changing a single visit in a negative test. |
| I3. Image-conditioned output | Preprocess a real image, encode it, map patch outputs to the rendered positions and run I2 on the combined prompt. | Match visual embeddings, rendered IDs/positions and scores to an independent implementation. Change the pixels with fixed placeholder/token IDs and check the expected change in scores. Stream a complete answer to a held-out prompt. |
| I4. Paged execution | Bind a state bundle to each sequence. Install the same model behind paged append/decode and fixed-capacity profiles. | Match I2/I3 across chunking, partial pages, local eviction, mixed prompt lengths and batches. Include convolution histories in checkpoint/cancellation tests. Suppress rejected or failed writes and reclaim after the last fence. |
| I5. Continuous stream and score | Admit a second request during decode, interleave bounded prefill and decode, stream completed tokens, cancel one request and score the other's exact token history. | Run real text and image fixtures on the full checkpoint. Show bounded memory, stop reasons, independent output-token counts, no per-token compile, and correct reuse of a cancelled slot. Measure TTFT, ITL, HBM, compute, state traffic and idle time. Record performance gaps. |

An instrumented or emulated computation qualifies for numerical validation only when it preserves the selected
activation/weight rounding and all nonlinearities. Record emulation, storage expansion and transfers. Native
mixed-format providers enter when they pass the same fixtures and win on the intended shapes. Capture/replay,
prefix reuse, DSpark, a sharded trainer and production service packaging each have separate acceptance criteria.
A BF16 dequantized pass is a diagnostic until release-format forward parity passes. M1 depends on the forward portion
of model phase T4, including a verified emulation if needed. T2's full tuning and T4's fast backward can proceed
independently. Run the first real generate/score/update cycle once I2 and a training checkpoint exist. It can progress alongside
I3–I5 and the serving optimization work below.

### Phase 2 — prefix reuse, mixed scheduling, and retention

Deliverables:

- [ ] Compare whole decode graphs, segmented prefill graphs and uncaptured execution. Try dependent launch for
      eligible linked kernels only. Ablate fusion and graph replay separately on low-latency and saturated workloads.
- [ ] Add prefix publication/indexing, partial-page COW, eviction, preemption/replay, and completion-safe reclamation.
- [ ] Evaluate grouped cascade decode attention with reusable runtime group metadata, ordinary paged fallback, and
      deadline/fairness bounds. Start with independent groups; gate hierarchy on measured benefit.
- [ ] Tune chunked prefill and mixed `extend` execution against homogeneous profiles using token and state budgets.
      Add deadlines/slack, fairness/starvation bounds, and deterministic scheduler decision replay. Compare calibrated
      time-budgeted chunks with fixed chunks under mixed context lengths, including prediction error and fallback.
- [ ] Add session pause/resume, retention/expiry, and cost-aware admission for tool-paused and bursty workloads.
- [ ] Extend conformance coverage for each feature, including hybrid checkpoint validity and shared-prefix cancellation.
      Validate ring wraparound, speculation headroom, and checkpoint-copy completion; keep approximate replay disabled
      unless a separate model-specific quality contract and cache identity have been accepted.
- [ ] Run controlled ablations for prefix retention, mixed execution, and lookahead, including cold-cache and hostile
      reuse distributions. Separately ablate cascade attention with prefix storage reuse held constant; vary group size,
      prefix/suffix lengths, and independent/nested roots. Charge grouping/planning costs and measure KV traffic.
      Diagnose padding, launch, kernel, and state-memory costs against the registered budgets.

Phase exit: the first full Mini workload/SLO envelope is reproducible against pinned baselines with bounded starvation,
stable memory, and attributable latency and goodput. State and request invariants remain shared across executors.

### Phase 3 — weight updates and rollout qualification

Start this work at I2 with the model plan’s first resumable update. Phase 2 retention optimizations can ship
independently. A complete post-training qualification includes the full-checkpoint round trip below.

Deliverables:

- [ ] Change image bytes with identical placeholder IDs and repeat a continuation after an encoder refit. Verify
      reconstruction from the pinned media and confirm old image features and prefix pages cannot satisfy the new key.
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

Exit gate: real weight updates and Mini rollouts repeat without process restart, stale-cache reuse, unbounded memory,
or unexplained probability mismatch. Demonstrate or reject the registered integration hypothesis after charging all
update, retention, and reconstruction costs. Run this before broad decoding, portability, or service expansion.

### Phase 4 — advanced lifecycle, decoding, and scale

Deliverables:

- [ ] Expand tensor-parallel Mini performance/topology coverage; initial support has already shipped if earlier memory
      or SLO gates required it. Complete multi-rank failure handling before accepting each new topology.
- [ ] Compare larger batches on one worker, independent replicas, TP and EP for actual expert occupancy and fabric.
      Add pipeline/context sharding when the memory or latency envelope warrants it.
- [ ] Add production fork/checkpoint/rollback for best-of-N/tree rollouts and richer speculative state strategies.
- [ ] Add sleep/offload/wake, colocated role arbitration, and collective/RDMA weight transfer where measured demand
      justifies them. Complete conformance coverage alongside each feature.
- [ ] Add grammar compilation/masking and deterministic-span fast forwarding.
- [ ] Compare Mini's supplied DSpark drafter with ordinary decoding at low and high concurrency, with explicit
      verification/rollback for KV, conv state and feature taps. Add adaptive depth only after a calibrated fixed-depth
      baseline preserves the declared sampling distribution.
- [ ] Expand the adapter registry and evaluate quantized KV independently of the released expert formats.

- [ ] Bring up Qwen3.6 27B as the second model, including hybrid recurrent/attention state behind the existing
      `SequenceState` and scheduler contracts.
- [ ] Benchmark phase-specific attention-DP/EP layouts, fused expert providers, packing and skew on Mini.
      Gate redundant expert placement/rebalancing on measured benefit and coherent map/weight publication.
- [ ] Reach CUDA/ROCm provider parity for the supported feature slice or publish explicit capability degradation.

Exit gate: multi-rank failures fail coherently; Mini performance scales predictably; Qwen requires no scheduler or
request-lifecycle fork; advanced features improve their target traces without unacceptable default-workload regressions.

### Phase 5 — service hardening

Deliverables:

- [ ] OpenAI-compatible service, lower-level protocol, auth/quota adapters, health/readiness, and graceful drain.
      Add canonical conversation/encoding/stream-parser fixtures and verify structured-output enforcement boundaries.
- [ ] Hot model/adapter load, rolling update, overload/load-shed policy, and operational runbooks.
- [ ] Production trace capture/replay, capacity estimator, and configuration advisor.
- [ ] Security review of model files, grammars, request limits, FFI plugins, and multi-tenant state isolation.
- [ ] Add bounded remote checkpoint loading and target-specific runtime packages using the same weight publication path.
      Measure cold/warm startup, host-memory peaks, cache hits, and failures before expanding storage/platform support.
- [ ] Soak, chaos, restart, fragmentation, and long-context tests.

Exit gate: production SLOs, recovery, observability, and operations are demonstrated under representative failure and
burst tests, not only steady-state benchmarks.

### Phase 6 — distributed and disaggregated serving

Deliverables:

- [ ] Worker discovery/membership, KV-aware routing, replica placement, global state-event index, and autoscaling hooks.
- [ ] Aggregated-versus-P/D profiler/simulator and online planner.
- [ ] Versioned state-session handoff with measured transport providers and complete cleanup paths.
- [ ] Host/remote state tiers only where reuse-value models show benefit. Compare cache-local and shared-storage routing,
      account for lookup/fetch/swap-in stalls and retention economics, and test partial-tier misses and storage failures.
- [ ] Balance decode request counts and attention work separately; size prefill/decode capacity from measured demand.
      Include spare-capacity cost and drain/migration tails in scaling and failure tests.
- [ ] Replay identical resumptions with worker affinity, epoch-checked host-state transfer, and local reconstruction.
      Log directory hit, full and windowed KV bytes, conv tails, network/device-copy latency, owner queue delay and
      retry cost. A destination validates and fences the entire bundle before advertising it.
- [ ] Multi-node rolling upgrade, drain, retry, and partition behavior.

Exit gate: disaggregation beats aggregated serving on a declared workload/SLO/fabric after charging transfer and control
cost; otherwise aggregated mode remains the default.

### Optional future extension — MAX backend evaluation

Keep MAX as a potential backend alongside XLA, not a prerequisite for the phases above. XLA remains the primary
implementation. The October 23 review inspected Modular revision `da0db31d422e19ad5952d8d106b6da1d6899cf0a`, following
the MAX 26.7 source release. Native embedding is plausible through its C API; production integration and a performance
advantage over Ryft/XLA remain unproven. A MAX Serve comparison can begin independently of any backend implementation.

Distinguish three scopes: benchmarking the complete MAX server, executing MAX-compiled model steps under Ryft's
scheduler, and compiling Ryft programs through a general MAX backend. Only the last preserves the shared model through
Ryft's own lowering path. The inspected compiler accepts MAX's MEF/MO/MGP formats, not a documented StableHLO/PJRT
interface. A general backend needs new operation lowering, buffer/runtime integration, and capability tests; shared
MLIR infrastructure does not provide interchangeability. Reuse the existing backend-neutral compilation and inference
contracts rather than introducing a second scheduler or duplicating model architecture code.

Native execution also needs a physical proof. The inspected C header exposes synchronous execution and graph
capture/replay, but not the asynchronous execution entry point mentioned by the API overview. A blocking call on a
worker thread can overlap host work without establishing device-side chaining or completion-fence parity. Device-pointer
borrowing does not by itself prove safe mutation, aliasing, or cross-runtime sharing. Compile/init separation exists,
but Python weight-reload support does not establish an equivalent native refit path.

Evaluate incrementally, without committing to broad backend coverage:

- [ ] Benchmark pinned MAX Serve if a release supports the full Mini workload and section 18's comparison contracts.
      Attribute results to the complete engine; do not infer compiler superiority from server-level throughput.
- [ ] Before adopting a distributable dependency, review the exact component licenses and native API stability.
      Newly released internals use the source-available MAX License; separately licensed Apache components retain
      their terms. Account for modified-source distribution, AI-assisted reimplementation, and applicable hosted-service
      branding requirements. The license explicitly permits analysis and interoperability work.
- [ ] If evidence justifies it, lower one small Ryft-generated model step and execute it through a native bridge.
      Prove persistent KV mutation, buffer lifetimes, error/completion handling, scheduling overlap, and graph replay
      under changing metadata before attempting full model coverage. Keep training/score operation support explicit.
- [ ] Demonstrate weight rebinding without recompilation, safe old-version retirement, artifact compatibility, and
      bounded compile/startup memory. Add sharding/collective proofs for each proposed multi-device deployment.
- [ ] Promote the experiment only after matched end-to-end measurements show a material benefit or required hardware
      coverage that justifies maintaining another lowering/runtime stack. Otherwise retain MAX as a comparison target.

MAX's serving lessons also merit experiments within the XLA implementation:

- Compare immediate mixed admission with bounded prefill coalescing/shared cadence when mixed steps lose graph replay;
  measure TTFT, decode latency, and fairness together. Do not assume MAX's capture limitation also applies to Ryft.
- Ablate graph replay and host/device overlap independently, and charge additional speculative-width captures to
  startup time and graph memory, not only steady-state token throughput.
- Evaluate compilation from weight metadata before full checkpoint transfer, with real-weight validation before
  readiness. Extend hybrid-cache tests so KV blocks and recurrent checkpoints remain coherent across tier eviction.

These experiments refine existing Phases 0–5; a second production backend is a separate future decision and does not
expand their exit gates. No MAX performance result has been reproduced as part of this architecture review.

## 18. Benchmark and acceptance strategy

### Reproducible Mini baseline

Reproduce the launch team's SGLang test if a B300 is available, with 1,024 input and 1,024 output tokens, concurrency
1, 4, 16 and 64, TP1 and both ordinary and DSpark decoding. Record the exact image, flags, page size, attention and
expert backend. Re-run the same cases for Ryft and a pinned Mini-capable vLLM build on the *same* device. Keep the
B200 M1 measurements in their own hardware row. Then add heterogeneous lengths spanning the local window, image
prefixes, tool waits, routing skew, open-loop arrivals and dynamic policy versions. Compare selected-action scores and
matched output quality. Include encoder and draft time and HBM. Report single-stream latency, SLO goodput and
throughput on identical generated-token accounting. [SGLang launch commands and results](https://huggingface.co/blog/inkling2mini#speculative-decoding)
and [vLLM Mini serving recipe](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini)

### Primary envelope and performance hypotheses

The primary serving workload is full-checkpoint Inkling 2 Mini text generation for multi-turn coding agents with long
prefixes, tool pauses, bursty resumptions, and concurrent new arrivals on one NVIDIA worker group. Measure fixed-version
serving first, then the same session distribution under weight churn. Short interactive chat and low-reuse traffic are
regression controls; saturated generation is a separate throughput surface. This choice does not imply that retention
or mixed execution will win before they are measured.

Before Phase 0 exits, commit an experiment manifest containing exact hardware/device count/interconnect, checkpoint
and tokenizer revisions, context/output-length distributions, arrival process, concurrency, pause/reuse distributions,
sampling and quality policy, TTFT/ITL/E2E SLOs, fairness bound, and memory limits. For RL include update cadence,
maximum policy lag, retained-version count, and trainer/transfer topology. Section 17 proposes a first single-worker test envelope. Record the actual hardware
and choose absolute latency bounds from the intended service before tuning.

Register and test four mechanisms independently and together:

| Hypothesis | Mechanism and comparison | Success measure |
|---|---|---|
| Overlap reduces host-induced GPU idle time. | Device token feedback, metadata lookahead, and graph replay; compare the same engine with lookahead enabled/disabled. | Lower idle gaps and better SLO goodput or ITL without increased cancellation errors or unacceptable wasted work. |
| State-aware scheduling improves agent serving. | Prefix/session retention and mixed token execution; independently disable retention and mixed execution under the same memory budget. | More completed requests meeting all SLOs, bounded starvation, and improved end-to-end agent latency where reuse exists. |
| Grouped prefix attention reduces decode state traffic. | Hold prefix storage reuse constant and toggle cascade attention; sweep group occupancy, prefix/suffix lengths, and unrelated traffic. | Lower KV bytes read and better full-step latency/SLO goodput after grouping overhead, with numerical and fairness contracts preserved. |
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

- pinned full-checkpoint Inkling 2 Mini text generation as the first acceptance surface;
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
- multimodal requests within the first milestone, plus dense and recurrent models for later extensibility checks.

### Metrics

- p50/p90/p99 TTFT, TPOT/ITL, E2E, and queue delay;
- SLO goodput and per-user decode-rate floor;
- prompt/output/total tokens per GPU-second and per dollar;
- HBM usage, page fragmentation, prefix hit, COW, eviction/recall, and transferred bytes;
- host CPU, scheduler time, launch overhead, graph/profile hit rate, and compilation misses;
- speculative acceptance/effective speedup, feature/draft/verify costs, and accepted tokens per target verification;
- streaming burst size and inter-burst stall percentiles in addition to per-token ITL; tokens delivered together must
  not turn near-zero within-burst intervals into a claim of uniformly low user-visible latency;
- artifact pull, checkpoint fetch, deserialization/conversion/resharding, compilation, warmup, and time-to-ready;
  report cold/warm caches, peak host/device memory, temporary storage, and bytes transferred separately;
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
  claims; external-engine comparisons use the complete pinned Inkling 2 Mini checkpoint.
- Warm and cold results are separate; compilation, model load, and prefix-cache warmup are reported.
- Compare optimized supported configurations, not deliberately weak defaults.
- Publish Pareto curves across concurrency/profile choices and all failure/unsupported cases.
- Use open-loop arrival traces for overload/queueing tests to avoid hiding stalls through client pacing; use separate
  closed-loop agent traces for tool dependencies. Count rejected, timed-out, and cancelled requests explicitly. Define
  SLO goodput as completions satisfying all registered request SLOs, not merely accepted load or average token rate.
- Repeat experiments with declared seeds and run durations, report uncertainty, and hold trace splits for final
  validation. Include cold/low-reuse controls and component ablations so benchmark-specific tuning is visible.
- Independently validate output counts; SSE chunks and server-reported usage alone are not token oracles. TTFT starts
  at request submission and ends at delivery of a real accepted output token. Count unfinished and failed work.
- A Qwen3-4B/L4 SkySynth-shaped reproduction is optional diagnostic work, separate from Mini acceptance. Random-token
  shared-prefix bursts complement held-out real agent traces; they cannot replace semantic or realistic workload tests.
- For Mini DSpark, compare no-speculation and optimized baselines across concurrency and block sizes under the same
  total device-memory budget. Measure streaming stalls and SLO goodput, not only acceptance or average tokens/user.
- Separate directly counted accepted-token throughput from capacity estimates derived from step rate and assumed
  acceptance. DeepInfer's pinned sweep/plot path applies a supplied acceptance factor; its benchmark filters expert
  load-statistics flush intervals. Preserve those modes as diagnostics, but include periodic maintenance, rebalancing,
  real acceptance/rejection, queueing, and all serving stages in end-to-end claims. Report the unfiltered time series.
- For phase-separated results, publish prefill and decode GPU counts, fabric, cached/uncached input work, numerical
  formats, and total deployment resources. Do not combine per-stage maxima into a claimed full-service rate.
- Test exact versus approximate reconstruction separately. Evaluate task quality and long-context dependence alongside
  speed, and never let approximate replay pass the controlled exact-state comparison by matching only short prompts.
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
| Conformance model creates unrealistic shortcuts | Mini integration forces a scheduler or state rewrite | Same executor/profile/state contracts, a real-XLA conformance path, prohibited type checks/special cases, and ordered substitution gates. |
| Rust creates false confidence | Safe host code but slow kernels/poor scheduling | Measure GPU, memory, launch, network, and queue behavior independently; compare Pareto frontiers. |
| External ecosystem moves faster | Permanent feature catch-up | Stable semantic provider APIs; reuse external kernels/protocol adapters; focus Ryft differentiation. |

## 20. Feasibility and sequencing

The shortest path to a useful engine runs through the full checkpoint, one aggregated worker, and one model shared
with the trainer. I2 exercises the renderer, quantization, loop schedule and incremental state on real tokens. I5
establishes the first operating envelope with streaming, cancellation and a measured memory budget. Correctness here
is valuable even before the optimized profiles reach competitive throughput.

The highest-risk performance questions are whether the executor preserves page-pool aliases and overlaps independent
work, whether specialized attention and grouped experts improve a complete step, and whether policy publication plus
prefix reconstruction leaves enough time and memory for useful rollouts. Measure each against a controlled fallback.
For production compare same-model SLO goodput on matched hardware. For post-training compare time to accepted
rollouts at fixed policy lag and quality. Bring up a second architecture after these contracts survive Mini.

## 21. Decisions to validate early

- [x] Select the fixed-loop Inkling 2 Mini checkpoint for M1. Retain Qwen3.6 27B as a later hybrid-state test.
- [x] Use pure-Rust and real-XLA conformance executors to unblock engine work while preserving production contracts.
- [ ] Prove the physical state/provider/graph/overlap mechanism before stabilizing executor and state interfaces.
- [ ] Validate append-only, sliding-window, and overwritten recurrent state, including checkpoint/replay costs and
      coherent hybrid-prefix boundaries, in the initial conformance slice.
- [ ] Freeze the model plan’s pinned Mini artifact, renderer and reference versions in the benchmark manifest,
      together with device topology, numerical mode, memory limits and serving/RL traces. Measure when TP is required.
- [ ] Ratify quantitative leadership/competitive targets, control-trace guardrails, quality tolerances, and cost budgets
      before tuning; record baseline uncertainty and controlled versus production comparison configurations.
- [ ] Generate Mini fixtures that preserve the fixed visit schedule, routing, two expert formats and image path.
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

### First milestone and subsequent models

- [Thinking Machines Lab’s Inkling 2 Mini release](https://thinkingmachines.ai/news/inkling-2-mini/)
- [Pinned model configuration and artifacts](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/ed40e7da625f594fc9c4990f5f66be24873c22d7)
- [Model inventory, training kernels and acceptance fixtures](plan-inkling-2-mini.md)

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

### Targeted SkySynth and ZML/LLMD research

- [SkySynth release and evaluator lessons](https://skydiscover-ai.github.io/blog-skysynth.html)
- [SkySynth pinned inference example](https://github.com/skydiscover-ai/skydiscover/tree/c30ca1907b0084c7ef2d2ed4b6398d181a3fad0c/skydiscover/synthesize/examples/inference-engine)
- [Hydragen: shared-prefix attention decomposition](https://arxiv.org/abs/2402.05099)
- [FlashInfer cascade attention](https://docs.flashinfer.ai/api/cascade.html)
- [ZML/LLMD product and platform overview](https://zml.ai/llmd/)
- [ZML/LLMD alpha announcement and performance tables](https://zml.ai/posts/llmd/)
- [ZML inspected source revision](https://github.com/zml/zml/tree/c6530f08d39d9c502bd2b3c7035d73bbaf7196c9)
- [ZML bounded, shard-aware loading](https://github.com/zml/zml/blob/c6530f08d39d9c502bd2b3c7035d73bbaf7196c9/zml/io.zig)
- [ZML backend compilation policy](https://github.com/zml/zml/blob/c6530f08d39d9c502bd2b3c7035d73bbaf7196c9/zml/Compiler.zig)
- [DFlash paper](https://arxiv.org/abs/2602.06036)
- [DFlash implementation and supported draft checkpoints](https://github.com/z-lab/dflash)

### DeepInfer release review (2026-10-20)

- [Pinned release scope](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/README.md)
- [Deployment design and limitations](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/doc/inference_deployment_qa.md)
- [Benchmark accounting](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/examples/benchmark.py)
- [Sweep configuration](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/examples/sweep/run_sweep.sh)
- [Acceptance-based throughput estimates](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/examples/sweep/plot_pareto.py)
- [DeepSeek Recipe protocol/encoding boundary](https://github.com/deepseek-ai/deepseek-recipe)
- [Verification budget and feedback semantics](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/mtp_varlen.hpp)
- [Device-side verification metadata](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/tilelang_ops/cuda/schedule_mtp_varlen.py)
- [Request-local random sampling](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/tilelang_ops/cuda/random_sampling.py)

### MAX future-backend review (2026-10-23)

- [MAX source release and licensing boundary](https://www.modular.com/blog/max-license)
- [MAX License](https://www.modular.com/legal/max-license)
- [Pinned 26.7 serving and compilation changes](https://github.com/modular/modular/blob/da0db31d422e19ad5952d8d106b6da1d6899cf0a/max/docs/site/releases/v26.7.md)
- [Native model execution API](https://github.com/modular/modular/blob/da0db31d422e19ad5952d8d106b6da1d6899cf0a/max/include/max/c/model.h)
- [Compiler input formats](https://github.com/modular/modular/blob/da0db31d422e19ad5952d8d106b6da1d6899cf0a/GraphCompiler/include/GraphCompiler/Driver/Common/DriverInputFormat.h)
- [Native tensor ownership and borrowing API](https://max.modular.com/api/c/tensor/)
