# Training and running Inkling 2 Mini in Ryft

**November 20, 2026.** Start with a differentiable implementation of the released, fixed-three-pass model. The first
training deliverable is a reproducible supervised fine-tuning (SFT) update. The first engine deliverable is generation
from the published checkpoint, first with text and then with an image. Both use the same parameter tree, visit schedule,
processor and scorer. The [inference plan](plan-inference.md#17-first-milestone-and-execution-plan) owns request
scheduling, paged state and the full-checkpoint engine acceptance test.

Ryft already has the math to build a small oracle and to differentiate the grouped expert projections. The largest
training gaps are a validated importer and optimizer, a memory-bounded attention forward **and backward** for Mini's
relative logits and sink, and dropless grouped expert forward, input-gradient and weight-gradient kernels. The
repeated core makes a subtle integration test possible: the three passes must contribute to **one** optimizer update
for a shared layer, while each pass must see **its own** history when generating the same sequence.

The [Gemma inventory](crates/ryft-core/gemma_4_plan.md) and [Muse inventory](crates/ryft-core/muse_glimmer_30b_plan.md)
are useful for the common operations. This inventory is based on the working checkout at base
`c62153b6b136b4bf0c579e764d17ff6ee1dc2f93`. Source coverage, published reference results and proposed Ryft
milestones are identified separately. The performance figures below come from the cited projects.

[Checkpoint and architecture](#released-model-and-checkpoint-contract) · [Ryft coverage and gaps](#ryft-coverage-and-training-kernels) · [Implementation and acceptance](#implementation-sequence-and-acceptance)

## Released model and checkpoint contract

Target `thinkingmachines/Inkling-2-Mini` at revision `ed40e7da625f594fc9c4990f5f66be24873c22d7` (Apache 2.0, November 20, 2026). It takes text and images and generates text, with a maximum context of 1,048,576 positions. The 26.4B parameters include 3.4B active unique parameters per token. Counting repeated uses gives 5.4B weight activations per token. The separate variable-loop research checkpoint has different training and runtime semantics. [Release](https://thinkingmachines.ai/news/inkling-2-mini/) · [Model card](https://thinkingmachines.ai/model-card/inkling-2-mini/) · [Pinned configuration][ink-config]

### Decoder, from token IDs to logits

| Component | Computation Ryft must reproduce |
| --- | --- |
| Embedding and LM head | Separate BF16 embedding and output matrices, each stored as `[201024, 2304]`. RMS-normalize the embeddings. At the output, a learned read over block residuals feeds the final RMSNorm and LM head. Project with FP32 accumulation, apply `30 × tanh(logit/30)`, and normalize the loss and sampling distribution over IDs 0–200057. The 966 padding rows must never receive probability. [Checkpoint header][ink-tensors] · [Training oracle][ink-oracle] |
| Physical weights and loop schedule | Run zero-based layers `0..6`, then `6..18` three times, then `18..24` (half-open ranges). Each later pass restores the prelude state and reads the previous pass's two middle-block outputs. The coda consumes the last pass. Each of the 12 shared layers owns one set of parameters and receives the sum of its three gradient contributions. A fixed schedule of 48 applications suffices. [Pinned configuration][ink-config] · [Loop reference][ink-loop] |
| Block attention residuals and cross-pass gates | Six physical layers per residual block. Make learned, per-token softmax reads over the completed and partial block states with RMS-normalized values. Subsequent passes compute separate previous-pass messages for Q, K, V and the expert input. Each controller normalizes its input and message, concatenates them, projects to rank 64 and produces two tanh gates. Its output is `(1 − forget) × input + gain × message`. Even a local layer can have three distinct Q/K/V input streams on a later pass. Keep residual/query/gate accumulation precision explicit. [Loop reference][ink-loop] · [Training oracle][ink-oracle] |
| Hybrid grouped-query attention | In each six-layer block the first five layers are causal local (1,024 positions including self) and the last is global. All have 24 Q heads, 8 KV heads, head width 128. The target uses learned **query-conditioned relative bias**. A 16-component projection mixes a learned table indexed by backward distances 0–1023. Global keys at greater distances still participate with zero relative bias. Q/K head norms are clipped at √512 and relative vectors at 8, with learned Q/relative scales. Global Q and relative states additionally scale by `1 + 0.1 × log(max(1, (position + 1)/8192))`. V uses gamma-free RMSNorm. Include a learned zero-value sink logit per query and head in the FP32 softmax denominator, then apply a sigmoid gate to the attention output before the output projection. Packed positions and masks must isolate documents across every pass. [Pinned configuration][ink-config] · [Training oracle][ink-oracle] |
| Four causal short convolutions per visit | Convolve projected K, projected V, attention output and expert output at width 4. Local K/V and both output convolutions use static depthwise taps plus a residual, scaled by `1/√5`. Global K/V additionally generate per-token, per-head taps from their respective input and use `1/3` for the residual-plus-filter result. Match tap orientation and history across padding, segment starts, prefill chunks and single-token updates. Backpropagate through generated taps and the input history. [Training oracle][ink-oracle] · [Loop reference][ink-loop] |
| Routing and expert projections | Each layer has 128 routed experts of width 1,152. Compute sigmoid routing in FP32, select 8 by score plus correction bias, then renormalize the selected **unbiased** scores to sum to 8. vLLM resolves ties by smaller expert ID. In the native path unpack interleaved gate/up channels from W13, form `(4 tanh(g/4)) × sigmoid(g) × (25 tanh(u/25))`, multiply by the route weight, and quantize that activation for W2. Sum expert contributions in FP32. Moving the route weight past quantization changes results. Compare expert IDs, scores, post-gate activations and combined outputs at precision boundaries. [Router][ink-router] · [Native activation][ink-act] · [Training oracle][ink-oracle] |

### Artifact import and training numerics

The main Safetensors index accounts for 19,525,870,752 tensor payload bytes. Import the named tensors and sidecars from the main shard. Keep the approximately 636 MB auxiliary `mtp.safetensors` separate. Configuration describes a 3-layer BF16 DSpark drafter with block size 8, rank-256 Markov head and target taps `[5,23,41,48]` in the upstream visit convention. Capture the running partial block residual before visits 5, 23 and 41 and the final normalized hidden state at boundary 48 only for a later target-verified speculative path. [Release files][ink-files] · [Tensor index][ink-index] · [Pinned configuration][ink-config] · [vLLM recipe](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini)

| Parameter group | Storage and computation |
| --- | --- |
| Expert W13 and W2, layers 0–19 | NVFP4 E2M1, packed low nibble first, nearest weight rounding, 1×16 blocks along contraction, E4M3 block scales and FP32 per-expert amax divided by an outer-scale denominator. Layer 0 W13 is `U8[128,2304,1152]`, with `F8_E4M3[128,2304,144]` scales. The activation contract is per-token amax and nearest NVFP4 in blocks of 16, with E4M3 scales selected by the four-over-six MAE rule. Cover zeros, saturation, nibble order and gate/up interleave with byte fixtures. [Precision manifest][ink-quant] · [Tensor header][ink-tensors] · [Checkpoint adapter][ink-adapter] |
| Expert W13 and W2, layers 20–23 | MXFP8 E4M3 weights, unsigned E8M0 scales. The manifest describes 32×32 weight quantization tiles. The file stores per-output-row scales, one byte per 32 input elements, so physical W13 has shape `[128,2304,2304]` and its scales have shape `[128,2304,72]`. Validate this layout independently from the weight quantizer. Activations use nearest E4M3 in blocks of 32 with UE8M0 scales. [Precision manifest][ink-quant] · [Tensor header][ink-tensors] · [Checkpoint adapter][ink-adapter] |
| Other tensors | Load BF16 and FP32 groups individually, honoring output accumulation and the distinction between on-disk, compute and accumulator types. Weight importing should stream whole experts or shards into their destination placement. Retain one copy of each physical parameter. When building a BF16 training oracle, decode in bounded chunks. Persist a conversion version and test the inverse mapping for trainable/exported weights. [Precision manifest][ink-quant] · [Checkpoint adapter][ink-adapter] |

Use two named numerical targets for training. **BF16 fine-tuning** initializes trainable weights by decoding the release. On four GB200s, NVIDIA reports 94 GiB/GPU for MedPix image-text SFT of the full language model with frozen vision and 44 GiB/GPU for rank-8 LoRA on that workload. A separate packed 128K Tulu 3 text SFT run with context parallelism used 116 GiB/GPU. These are comparison points for Ryft's training implementation. [NeMo recipes and measurements](https://docs.nvidia.com/nemo/automodel/nightly/model-coverage/vision-language-models/thinkingmachines/Inkling-2-Mini)

**Native mixed-precision training** uses the release's forward formats and activation quantizers, including both expert projections. A faithful slow forward can quantize then dequantize before matmul. For the accelerated backward, follow the lab's documented recipe for the model family: save quantized forward values, dequantize, transpose or regroup, and stochastically requantize to MXFP8 for input and weight gradients. Record RNG streams, specify scale-gradient handling and compare the expected update against a numerical reference. Retain higher-precision trainable master weights and optimizer state across steps. The Mini release establishes its forward recipe. Its files do not supply the original unquantized masters, optimizer state, dataset, or full training-run provenance. Ryft can fine-tune the released model and test a specified precision policy. Reproducing the original trajectory is a separate experiment. [Mini training description](https://thinkingmachines.ai/news/inkling-2-mini/) · [Thinking Machines' training numerics](https://thinkingmachines.ai/blog/end-to-end-4-bit-training/)

### Renderer and vision

Use the released tokenizer and message template across training and serving. Construct token IDs for role, content, reasoning effort (default 0.9), tool declarations, canonical tool calls/results, images and stop ID 200006. A message terminator and the end of assistant sampling are distinct tokens. Use the template's assistant-generation spans for supervised labels, intersect with valid next-token targets within each packed document, and preserve prompt/image positions as conditioning. Compare IDs, roles, loss weights, stop handling and multi-turn tool-call round trips with pinned fixtures. [Chat template][ink-chat] · [Configuration][ink-config]

For still images, match RGB conversion, alpha compositing, padding to the 40×40 grid, rescaling and channel normalization. One raster-order image token consumes each grid tile (`ceil(H/40) × ceil(W/40)` for the pinned processor), comprising four native 20×20 patches. The patch projection has 1,200 input channels after flattening RGB. Add the learned class token for each image, interpolate the learned 32×32 position grid bilinearly, and apply axial rotary positions in the **vision encoder**. Its 23 bidirectional layers have width 1,024, 16 heads, 4× MLPs, LayerNorm and learned residual scales. Attention stays within each image. Remove class tokens, merge 2×2 neighboring features, project 4,096 → 4,096 → 2,304, and replace the expanded `<|unused_200054|>` placeholders. Compare patch ordering, feature counts, projected values, mixed text/image logits and one trainable vision-gradient fixture. The published image-text SFT recipe freezes the tower, which gives an earlier practical milestone. [Processor][ink-pixels] · [Vision encoder][ink-vision] · [Processor configuration][ink-processor] · [Image-text training recipe](https://docs.nvidia.com/nemo/automodel/nightly/model-coverage/vision-language-models/thinkingmachines/Inkling-2-Mini)

### State and capacity implications

Index K/V and the four convolution histories by **logical visit**. There are 40 local and 8 global visits, despite only 24 distinct sets of layer weights. At a prefix hit, fork or preemption boundary, restore the local windows, the global prefix, each visit's last three raw convolution inputs and absolute positions consistently. Block residuals and the two-block cross-pass message are working activations for the tokens being evaluated; they are distinct from persistent history between tokens. [Loop reference][ink-loop] · [vLLM implementation](https://github.com/vllm-project/vllm/pull/69028)

For one unsharded sequence and BF16 K/V, each retained position per visit costs `2 × 8 × 128 × 2 = 4,096` bytes. Ideal KV payload at length `L` is `4096 × (8L + 40 × min(L,1024))` bytes, about 32 GiB global plus 160 MiB local at the context limit. Add page rounding, convolution state, weights, transient activations and allocator reserve when admitting requests. Training also needs saved activations or rematerialization, gradients, master weights and optimizer moments. Size from unique trainable parameters and retained activations per **visit**, then measure peak memory for the chosen sharding and packing. [Configuration][ink-config] · [Cache layout and serving recipe](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini)

[ink-config]: https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/ed40e7da625f594fc9c4990f5f66be24873c22d7/config.json
[ink-files]: https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/ed40e7da625f594fc9c4990f5f66be24873c22d7
[ink-index]: https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/ed40e7da625f594fc9c4990f5f66be24873c22d7/model.safetensors.index.json
[ink-tensors]: https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/ed40e7da625f594fc9c4990f5f66be24873c22d7/model-00001-of-00001.safetensors
[ink-quant]: https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/ed40e7da625f594fc9c4990f5f66be24873c22d7/hf_quant_config.json
[ink-chat]: https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/ed40e7da625f594fc9c4990f5f66be24873c22d7/chat_template.jinja
[ink-processor]: https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/ed40e7da625f594fc9c4990f5f66be24873c22d7/processor_config.json
[ink-loop]: https://github.com/huggingface/transformers/blob/6173a2f679ff4ed393afc499aefc569da7e68bc1/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py
[ink-oracle]: https://github.com/NVIDIA-NeMo/Automodel/blob/bdf1fff2e8e8ef3e4035308820495e593b7df089/tests/unit_tests/models/inkling2/reference.py
[ink-adapter]: https://github.com/NVIDIA-NeMo/Automodel/blob/bdf1fff2e8e8ef3e4035308820495e593b7df089/nemo_automodel/components/models/inkling2/state_dict_adapter.py
[ink-router]: https://github.com/vllm-project/vllm/blob/4c616217b466d94c2dcce8fd2d0c346cc48a2bed/vllm/models/inkling2/nvidia/ops/moe.py
[ink-act]: https://github.com/vllm-project/vllm/blob/4c616217b466d94c2dcce8fd2d0c346cc48a2bed/vllm/models/inkling2/nvidia/ops/activation.py
[ink-pixels]: https://github.com/huggingface/transformers/blob/6173a2f679ff4ed393afc499aefc569da7e68bc1/src/transformers/models/inkling2/image_processing_inkling2.py
[ink-vision]: https://github.com/huggingface/transformers/blob/6173a2f679ff4ed393afc499aefc569da7e68bc1/src/transformers/models/inkling2/modeling_inkling2.py

## Ryft coverage and training kernels

Two earlier gaps have already narrowed. Ryft now has einsum and compiled-function batching with a static extent.
A general convolution primitive is optional for the initial implementation. Four shifted slices suffice for the text
filters, and the patch projection is a dense dot after patch extraction. [Einsum](crates/ryft-core/src/operations/einsum.rs) · [Compiled batching](crates/ryft-xla/src/compilation/functions.rs)

The following inventory reflects the current checkout. “Available” means the operation, lowering or example exists in source. Hardware throughput and end-to-end Mini parity are acceptance tests for the proposed work.

| Component | Available in Ryft | Required for Inkling 2 Mini |
|---|---|---|
| Model and checkpoint | `Parameterized` and path-aware tree traversal, backend-generic array code, an eager/JIT transformer forward benchmark and a reverse-mode MLP training example. [Parameter trees](crates/ryft-core/src/parameters.rs), [transformer](crates/ryft/examples/transformer_benchmark.rs), [MLP](crates/ryft/examples/mlp.rs) | Add the physical parameter tree, a checked safetensors importer, model-specific precision/packing map, tokenizer/processor and training checkpoint export. Keep one leaf and one optimizer slot per physical parameter even when visited three times. |
| Embeddings, norms, projections, gates and logits | Gather/scatter, general dots with accumulation and precision choices, casts, FP32 elementwise math, reductions and softmax support a differentiated reference. [Indexing](crates/ryft-core/src/operations/manipulation/), [dots](crates/ryft-core/src/operations/dot.rs), [reductions](crates/ryft-core/src/operations/reductions.rs), [softmax](crates/ryft-core/src/operations/softmax.rs) | Implement normalized token embeddings, learned and unweighted RMS norms, headwise learned scale and L2 clipping, position-dependent log scale, sigmoid-gated output projection, and capped vocabulary logits. Compare values **and** gradients on zero, near-threshold and large inputs. Match the reference's cast sites. |
| Block attention residuals and loop messages | Dots, softmax, concatenation, casts, `scan`, reverse-mode AD and rematerialization policies. [Scan](crates/ryft-core/src/operations/control_flow/scan.rs), [rematerialization](crates/ryft-core/src/differentiation/rematerialization.rs) | Use short reductions over normalized block states per token. Keep the pre-loop state, the previous pass's eligible block states and its final partial state. Restart from the pre-loop state on later passes and route the four messages through the learned low-rank gates. An explicit fixed schedule is a good reference. Measure rematerializing per visit/block versus retaining these small residual banks, and accumulate every visit's gradient into the shared leaf. |
| Width-four short convolutions | Padding, slices, broadcast, multiply and add can express four taps with ordinary AD. [Array manipulation](crates/ryft-core/src/operations/manipulation/), [arithmetic](crates/ryft-core/src/operations/arithmetic.rs) | Implement static depthwise K and V on sliding layers, dynamic per-token/head K and V taps on full layers, and static output convolutions after attention and MoE. Q has no convolution. Reproduce tap orientation, residual scaling and segment resets in forwards and adjoints. Fuse four reads per channel and reduce tap/weight gradients across tokens. Chunked prefills and decode preserve four independent histories per **visit**. |
| Causal grouped-query attention | Named SDPA composites provide reference GQA, masking, additive bias, local windows and a differentiated training entry point. Eligible BF16/FP16 CUDA calls can select cuDNN fused forward and backward. The adapter broadcasts an explicit score mask/bias to the full score shape. [Core attention](crates/ryft-core/src/operations/attention.rs), [CUDA lowering and tests](crates/ryft-xla/src/operations/attention.rs) | Implement the learned query-conditioned relative-position contribution within the distance band and the learned zero-value sink per query/head. The sink participates in the softmax denominator even when every real key is masked. Stock SDPA lacks this input. Implement a dense small oracle, then local/global tiled providers consuming sequence offsets, positions, query-relative features, the bias table and sink logits. Never allocate a global sequence-square bias, mask or score tensor in the fast path. |
| Routing and token packing | Stable top-k via sort/iota/slice, cumulative operations and indexed gather/scatter. Keys and indices have distinct AD semantics. [Ordering and tie rule](crates/ryft-core/src/operations/sorting.rs), [prefix sums](crates/ryft-core/src/operations/cumulative.rs), [scatter](crates/ryft-core/src/operations/manipulation/scattering.rs) | Compute 128 sigmoid scores in FP32. The correction bias affects selection only. Gather unbiased scores, normalize the selected eight and multiply by eight. Implement histogram, prefix offsets, permutation and inverse map. Sum the already route-weighted expert outputs in FP32. Preserve all eight selected assignments for each valid token, even under a maximally skewed route. Define ties, mask padding and preserve the selected IDs for backward/replay. |
| Expert contractions and activation | Non-contracting-token `RaggedDot` provides grouped matmul and first-order input and weight adjoints. Its weight adjoint uses the contracting mode. Transposing a contracting- or batching-mode forward is unsupported. XLA defaults to a masked dense expansion and also exposes opt-in CHLO instruction lowering. [Group semantics and AD](crates/ryft-core/src/operations/dot.rs), [lowering choices](crates/ryft-xla/src/operations/dot.rs) | Native grouped gate/up and down projections over the actual assignments. Fuse the FP32 softcapped SwiGLU between them (use the raw gate in the sigmoid). Provide grouped input- and weight-gradient kernels, FP32 reduction where needed, gate gradients through the normalized selected scores and scatter-add for repeated token IDs. The acceptance fixture includes empty experts, tails, repeated tokens, all tokens at eight experts, and gradients to shared weights. |
| Expert and data parallelism | Named meshes, manual `shard_map`, all-gather, reduce-scatter, all-reduce and ragged all-to-all are present. Ragged exchange lowers to a typed XLA custom call with a gated CUDA test. The XLA CPU backend rejects that exchange. [Sharding](crates/ryft-core/src/operations/sharding/), [collectives](crates/ryft-core/src/operations/collectives/), [ragged XLA exchange](crates/ryft-xla/src/operations/collectives/parallel_ragged_all_to_all.rs) | Build dropless dispatch and inverse combine, exchange sizes/offsets, reverse exchanges and gradient ownership per physical expert. Compare one-rank and sharded updates under skew, zero-length transfers and uneven last microbatches. Track maximum rank time, padding and communication bytes. |
| Published low precision | Block quantize/dequantize, low-bit types/scales, `ScaledDot` for quantized inputs, differentiable `block_scaled_dot` for unquantized inputs, and eligible CUDA dense `__op$block_scaled_dot` lowering for NVFP4 and MXFP8. Raw `ScaledDot` has no derivative. [Core precision contracts](crates/ryft-core/src/operations/quantization.rs), [CUDA eligibility](crates/ryft-xla/src/operations/quantization.rs) | Decode the actual checkpoint layout, scales and tensor-specific formats. Compare dequantized tiles and outputs to the pinned reference before repacking. Make BF16 dequantized-weight fine-tuning and frozen-packed-base adapters explicit modes. Training with quantized forward, activation gradients and weight gradients needs its own gradient recipe, stochastic state, grouped provider and convergence qualification. A dense scaled dot alone covers none of the MoE packing. |
| Images | Dots, reshape, gather, reductions, elementwise special functions and dense non-causal attention can implement patch projection, coordinate lookup, vision blocks and patch merging. [Dense dots](crates/ryft-core/src/operations/dot.rs), [array manipulation](crates/ryft-core/src/operations/manipulation/), [attention](crates/ryft-core/src/operations/attention.rs) | Implement the checkpoint processor's grid padding, pixel normalization, patch ordering, interpolated position embeddings, per-image attention boundaries, merge/projector and ordered text-placeholder replacement. Establish pixel-to-feature and gradient parity. Packed variable-resolution batches need per-image offsets to prevent cross-image attention and bound padding. |
| Loss, updates and resume | Integer-label CE, log-softmax, tree arithmetic, reverse-mode differentiation, rematerialization, explicit RNG state and categorical sampling. [Losses](crates/ryft-core/src/operations/softmax.rs), [RNG](crates/ryft-core/src/operations/random.rs), [SGD training example](crates/ryft/examples/mlp.rs) | Implement shifted target construction, assistant/token masks, valid-token reduction across ranks, AdamW and durable model/optimizer/scaler/RNG/data state. The CE mask API selects classes. Apply the SFT supervision mask to per-token losses. For a 200,058-way head, benchmark chunked or vocabulary-parallel softcapped CE that computes the global max, denominator and target logit without retaining every token's logits. |
| Kernel integration | Host interpreter, kernel definitions and verifiers, feature-gated backend adapters, staged calls with explicit VJP/JVP or a pure fallback, artifact handling and autotuning. [Kernel authoring](crates/ryft-kernels/README.md), [staging](crates/ryft-xla/src/kernels/staging.rs), [provider interfaces](crates/ryft-xla/src/kernels/mod.rs), [autotuning](crates/ryft-xla/src/kernels/tuning.rs) | Attach each optimized forward/backward pair to a numerical composite with an explicit eligibility predicate. Capture the selected provider, lowered calls, device trace and scratch allocation per bucket. Keep the dense oracle bounded to small fixtures. A compiled custom call or a host-kernel test establishes neither device throughput nor correct adjoints. |

One existing precision API deserves particular care. Ryft's dense
`BlockScaledDotConfiguration::nvfp4_training` already implements a Transformer Engine-style recipe, including
current global scaling, stochastic rounding of output cotangents, a 16-wide Hadamard transform for the weight
gradient, and tiled right-hand scales. Its reverse rule saves the original floating inputs. Mini needs per-token
forward scaling with four-over-six activation-scale selection and the selected family's backward recipe reuses the
**quantized forward** values, transposed and requantized to MXFP8. Setting the dense operation's gradient format
alone does not implement those residuals or grouped contractions. Reuse the format types, RNG and quantization
machinery, and give Mini's packed residuals and rounding sites a dedicated test. [Quantization and backward rules](crates/ryft-core/src/operations/quantization.rs)

For tiled attention, include the sink once per query/head in the online softmax with a zero value contribution. Across split-K tiles or context-parallel KV shards, assign the sink to one partial or incorporate it once after merging the partial normalizers. Compare its forward and adjoint with an unsharded reference using unequal shard lengths. Save the output and row log-sum-exp. In backward, replay score tiles, sum dK/dV across each KV head's three query heads, differentiate the sink probability and propagate the in-band bias gradient into both the 16-channel query features and shared distance table. Gate VJPs on segments that touch at a pack boundary, local-window edges, distances 1,023/1,024, positions around the 8,192 log-scaling floor, and global distant keys that still attend with zero relative bias. Compare to the dense reference on randomized cotangents. [Reference attention and convolution](https://github.com/huggingface/transformers/blob/6173a2f679ff4ed393afc499aefc569da7e68bc1/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py)

For grouped matmul, start with the same packed rows, expert offsets and saved route map for gate/up, down and the reverse pass. Within each expert, `Y = X Wᵀ`, `dX = dY W`, and `dW = dYᵀ X`. Restore the original token order and accumulate the eight contributions. Track expert counts, active versus padded rows, bytes moved, temporary bytes, kernel launches and forward/backward time separately. Reject a candidate that evaluates a dense token-by-expert cube or materializes all experts' dequantized weights per step. Compare the instruction and custom grouped paths on the intended hardware before choosing a default.

## Implementation sequence and acceptance

Use the proposed `ryft-models` package for the checkpoint schema, model, rendering and composition of operations.
Keep optimizer and data-cursor state in the training driver. Give reusable numerical operations explicit semantics in
`ryft-core` when composition or a staged kernel is insufficient. Implement device selection and lowering in
`ryft-xla` and the appropriate kernel provider. The engine binds this model to its state pools. Optimizer state,
page ownership and provider-specific tensor layouts have different lifetimes and owners.

| Order | Work product | What it unlocks |
|---|---|---|
| T0 | Immutable release manifest, importer, renderer and reference fixtures | Training and inference share the same inputs and named parameters. |
| T1 | Reduced BF16 decoder, differentiable packed loss, resumable AdamW step | Model math and gradient oracle. Contiguous inference bring-up can start as soon as the forward and loader pass. Release-format parity also needs T4’s forward oracle. |
| T2 | Memory-efficient BF16 attention and expert forward/backward, bounded optimizer/checkpoint sharding | Practical full-checkpoint language-model SFT with measured training throughput. |
| T3 | ViT, image insertion, frozen-tower SFT and trainable image path | Image-conditioned generation for inference M1. Start the vision forward alongside T1. |
| T4 | Release-numerics forward and qualified mixed-precision backward, adapter option, versioned export | Serve the original checkpoint in its native forward mode, fine-tune efficiently and feed real policy updates to the rollout engine. Start the packed-format oracle in T0. |
| T5 | Generate, score, update, publish and replay | Measured compatibility between a changing training policy and the inference engine. |

### T0. Specify the model at the file boundary

- Resolve the pinned config, tokenizer, template, processor, quantization sidecar and tensor index. Record file digests,
  per-tensor names/shape/dtype/offset/size, physical layer and expert ID, packed channels, scales and destination
  placement. Validate bounds, overlaps, missing or duplicate tensors, padded vocabulary and every expected sidecar.
  List the optional drafter as a separate model. Fail an import with a tensor-level diagnostic before publication.
- Implement two views from that mapping. A bounded CPU decoder expands reference tiles and initializes BF16 trainable
  parameters. The deployed layout retains packed weights and scales, repacked once per snapshot and provider if needed.
  Decode a known NVFP4 and MXFP8 tile through each path and through the pinned external adapter. Compare signed values,
  scale axes, zeros and extrema, expert orientation, and dot outputs. Test W13 and W2 independently.
- Pin a supported Transformers revision for structure and the NeMo float reference for gradients. Capture a separate
  native vLLM or SGLang trace for release-format arithmetic. For BF16 T1/T2, follow NeMo's FP32 gated activation × route weight, cast to BF16, W2, then FP32 sum.
  The pinned Transformers floating implementation applies the route weight after W2. Add a multi-route fixture
  that distinguishes the forward rounding and gradient to the selected score. For release numerics, the native
  kernel applies the weight before the W2 input quantization. Record any intermediate BF16 cast the provider uses. Bind each fixture to its producer, dtype/casts, routing decisions and accumulator policy. A BF16
  dequantization reference and a native activation-quantized reference answer different parity questions.
- Produce compact fixtures for text-only, image+text, tool calls and multi-turn continuations, with IDs, feature spans,
  action weights and stops. Include two unrelated packed examples so leakage is detectable. Keep an immutable manifest
  and small non-proprietary test inputs in test data. Stream the full weights from the user's local artifact store.
  [Released processor and template][ink-files] · [Checkpoint adapter][ink-adapter] · [Native expert kernel][ink-act]

### T1. Train a reduced model end to end

Keep the 24-layer layout and 48 visits in the principal CI fixture. One practical synthetic configuration uses width 128,
six query/two KV heads, head width 32, expert width 64, 16 experts and top-8 routing. Keep both quantization groups,
two message blocks, all four controller inputs and separate K/V and output convolutions. Test released head dimensions,
position constants and tensor packing in separate production-shape kernel cases. Keep the logical vocabulary small in CI with an explicitly padded physical head and invalid rows to mask.
Use small sequences and a shorter
window in CI to cross local and global boundaries without allocating a large attention matrix.

1. Implement a pure full-sequence forward over the physical parameter tree. Emit diagnostic taps at embedding lookup,
   prelude, each pass, coda, selected experts, pre-cap logits and vocabulary masking. An explicit unroll is sufficient.
   In a reference test deliberately untie the three visits at identical weights, differentiate each, and compare the
   sum of the three per-visit gradients to the gradient of the shared leaf. Include the four controllers and previous
   pass residual banks in the differentiation graph.
2. Render shifted integer targets with a separate mask for supervised tokens. Reset positions, attention boundaries and
   all convolution taps at document boundaries. Divide the sum of supervised losses by the global supervised-token
   count across data ranks and microbatches. A globally empty update must leave the step, moments and weights unchanged.
   Exercise uneven packing, ignored image/prompt positions, all-masked rows and vocabulary padding.
3. Implement a pure AdamW update over unique trainable leaves, with bias correction, explicit decay mask, learning
   rate, gradient accumulation, global norm clipping and nonfinite-gradient handling. Store moments and reductions in
   FP32. Make the treatment of router selection bias explicit: load it as selection metadata and keep it fixed in the
   baseline. Any later balancing update or auxiliary loss changes the training recipe and needs its own reference.
4. Save enough to resume at an optimizer boundary: master and working parameters or their rebuild recipe, moments,
   step/schedule, precision/scaling state, RNG keys/counters, data position, sharding schema and artifact revisions.
   Stage shard files under a new generation, verify sizes and checksums across ranks, then commit one manifest.
   Interrupt writes and reload the last committed generation. Compare the next updates to an uninterrupted run.
5. Pass a fixed-batch forward/loss/gradient/update comparison with an independent implementation. Use directional
   derivatives on smooth FP32 subgraphs with a fixed routing margin, plus named tests for ties, clipping and
   saturation. Compare BF16 updates to the same reference rounded at the same sites. Overfit a small fixture and
   report loss trajectory, gradient norms and nonzero updates to a selected expert, router and a loop controller.

### T2. Make a full-checkpoint language-model update practical

Start with BF16 dequantized weights and a frozen tower. Fit a small sequence-length bucket on a single worker group,
then scale sequence length, microbatch and parallelism separately. I2 can use the released mixed-precision forward or a verified emulation before its fast training adjoints are ready. A
dequantized BF16 decoder is a separate bring-up diagnostic. For training, qualify the following paths in roughly this order.

| Candidate | Work to eliminate | Numerical and performance gate |
|---|---|---|
| Packed local/global attention | Dense score, bias and mask materialization and explicit GQA head replication | Implement the tiled adjoint in the inventory above. Keep one FP32 row normalizer per head and query. Profile peak allocated bytes versus sequence length. Local work scales with tokens × window. Full-layer arithmetic remains quadratic while score storage stays bounded by tiles and row statistics. |
| Dropless routing and BF16 grouped GEMMs | Per-expert launches or dense products over all experts for every token | Build one assignment map over eight slots per valid token, group by expert, gather contiguous activations, run W13 and W2, and inverse-combine. Write VJPs using the saved map. Verify all three GEMM directions with an uneven histogram. Measure the full route-to-combine time. |
| Fused short convolutions, residual reads and controllers | Re-reading the same normalized block banks and repeated small projections at 48 visits | Fuse the four reads where their state banks agree, batch the rank-64 projections and preserve each input's learned parameters. Test fused adjoints, saved inputs, cast sites, and rematerialization. Compare profiles at small-batch decode and packed SFT separately. [vLLM's fused Mini implementation](https://vllm.ai/blog/2026-11-20-inkling2-mini) |
| Vocabulary-projection/loss and optimizer | Materializing token-by-vocabulary logits and full optimizer temporaries | Tile the projection and stable softcapped logsumexp, gather target logits, exclude padded rows, and recompute tiles for the backward. Compare masked CE and gradients including across a vocabulary shard. Fuse leafwise updates only after measuring step time and peak bytes. |

For the first CUDA provider, benchmark a proven grouped BF16 GEMM implementation such as CUTLASS behind the
existing XLA kernel/FFI boundary. Feed the per-expert problem sizes and offsets directly from the device. Check the
backward layouts and alignment for weight gradients independently of forward eligibility. For attention, require
a score-tile callback or a tile kernel that can compute Mini's differentiable relative bias and sink. An adapter
that only accepts a dense bias tensor cannot satisfy the long-context memory gate. Use the same composite and
VJP test cases for a Ryft-native implementation and a wrapped library. The lab describes its own fused Blackwell
expert kernels as in-house technology, so the roadmap needs a separately available provider.
[CUTLASS grouped scheduler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html) · [Thinking Machines' kernel design](https://thinkingmachines.ai/blog/end-to-end-4-bit-training/)

For the first distributed test, partition large expert weights and optimizer state, preserve replicated small
FP32 router/controller state where appropriate, and reduce shared-parameter gradients exactly once. Compare an
FSDP-style parameter all-gather/gradient reduce-scatter path with expert parallelism on the same fabric. EP sends
top-8 assignments to owner ranks and returns results in original token/slot order. Run all participants through the
same collective order, including empty routes. Bound buffers from the true assignment count or use a proven
overflow/split path that preserves every assignment. Measure imbalance, communication, overlap and temporary packing.
Defer context parallelism until global attention and sequence activation peaks justify it. A context-sharded
implementation must exchange K/V tiles, preserve global positions and supply conv halos across shard boundaries.
If EP transfers dominate, benchmark sending the already-quantized dispatch inputs and their scales. Keep the combine in
BF16 first. The lab reports a separate experiment using a 32-dimensional random Hadamard transform and block-scaled
S2F6 for combine traffic. Treat that as a later communication codec with its own inverse, accuracy and bandwidth
tests. [Thinking Machines' distributed training experiments](https://thinkingmachines.ai/blog/end-to-end-4-bit-training/)

A conservative memory worksheet with P trainable parameters is 2P bytes of BF16 weights, 4P of FP32 master weights,
8P of FP32 Adam moments and 4P of FP32 gradient accumulation (about 475 GB decimal for P = 26.4B). Charge only
the groups actually trained and stored this way. Add staging/packing, rematerialization boundaries across the
48 visits, workspace, all-gathers, optimizer temporaries and image activations. Sharding changes per-rank peak.
Log allocated and reserved device bytes, host peak and communication for a complete update and restart. NeMo's
four-GB200 results above give a matched hardware/recipe comparison where that system is available.

**Exit.** Run at least one full-checkpoint SFT update with real tokenized data, save/reload and repeat. Report which
parameter groups changed and confirm layer/gradient/optimizer parity on the declared reference cases. Before claiming
efficiency, benchmark multiple warm steps after compilation against a pinned Mini trainer on the same topology,
packing, trainable groups, context, optimizer and precision. Report supervised tokens/s/GPU, processed tokens/s/GPU,
step-time percentiles, peak by rank, padding/skew, and forward/backward/update/collective breakdown. Keep a short-run
loss or held-out likelihood guardrail. Incrementally enable each kernel and keep a speedup only within its validated
shape range. [NeMo Mini recipes](https://docs.nvidia.com/nemo/automodel/nightly/model-coverage/vision-language-models/thinkingmachines/Inkling-2-Mini)

### T3. Add images to the same training step and decoder

Implement processor → patch projection → vision blocks → merge/projector → decoder embedding replacement. Validate
single images before packing multiple images. Test odd grid sizes, alpha handling, multiple resolutions, ordering
across two images, and mismatched placeholder counts. In image-text SFT freeze the vision encoder first, declare
whether the projector is trainable, and verify the optimizer respects that declaration. Then enable visual gradients
and compare selected patch-projection, vision attention, projector and decoder gradients to a small independent
reference. At inference, encode once per prompt or verified compatible cache entry. Include a held-out image pair with
identical text and distinct expected scores. The forward path feeds I3 in the inference plan.

### T4. Qualify release precision and trainable exports

- **Native forward.** Test an entire expert at its real dimensions in both formats, including activation scaling,
  W13 interleave, capped gate, route weight, W2 requantization and FP32 combination. Test all-zero, outlier and
  thin/empty expert batches. Capture provider eligibility, padding, scratch and HLO/device traces. Keep a slow
  quantize/dequantize oracle for unsupported devices. Declare the actual accumulation and rounding mode at dispatch.
- **Native training.** Implement quantizer and contraction VJPs with an explicit surrogate-gradient policy and
  cross-axis scale/requantization rules for input and weight gradients. Keep frozen metadata out of the optimizer.
  Derive stochastic-rounding draws from step, microbatch, physical tensor, logical visit and element coordinates.
  Rematerialization must reproduce the same draws. Test fixed-seed reproducibility and multi-seed rounding error
  and bias against the family recipe. Compare multi-step loss/quality, finite gradients, training memory and
  tokens/s to BF16 and emulated-precision baselines. A derivative through integer storage cannot establish this gate.
- **Adapter tuning.** Exercise ordinary BF16 LoRA first, then a frozen packed base with trainable higher-precision
  low-rank updates. Declare target modules, ranks, scale and the exact addition point relative to activation and
  weight quantization. A core adapter attached to a physical projection follows it across all visits. Compare
  merged/repacked export with the unmerged path before treating them as equivalent. Measure memory released by
  freezing the base and the time to publish an adapter. Full parameter tuning retains its own T2/T4 gates.
- **Export.** Name the trainable policy and the quantized serving realization separately. Repack from updated
  masters, refresh the corresponding scales, verify coverage and hash all changed weights/scales/adapter leaves.
  Confirm one-step scores under the exported policy on both trainer and serving worker. Preserve the last good
  snapshot across a failed conversion. [Thinking Machines' low-precision training recipe](https://thinkingmachines.ai/blog/end-to-end-4-bit-training/)

### T5. Close the post-training loop

For a frozen snapshot, generate action tokens, save the exact trace and compare selected-action logprobs with
teacher-forced scoring and the trainable forward. Include a tool observation and an image. Check the behavior
distribution after logit processing on the same support. Apply one real learner update, export/install it, repeat
generation, and verify that the old revision can still reproduce its saved trace. Charge re-quantization and
adapter/weight publication to time-to-next-usable-rollout. Exercise synchronous version barriers before introducing
staleness limits. SkyRL and Miles publish Mini integrations for comparison. [SkyRL](https://docs.skyrl.ai/docs/models/inkling2#inkling-2-mini) · [Miles](https://miles.radixark.com/docs/models/thinkingmachines/inkling-2-mini) · [Ryft rollout contract](plan-inference.md#13-rl-and-agentic-integration)

## What to record before declaring support

For every claimed mode record the checkpoint and reference revisions, precision and selected kernel per tensor group,
trainable leaves, sequence/batch/packing envelope, device topology and numerical tolerances. Compare the reference
against itself across supported configurations before freezing tolerances. Require exact integer IDs, masks, segment
maps, shapes, stop reasons and byte fixtures. Use explicit absolute/relative or norm tolerances for floating-point
values and gradients, record router margins for ambiguous ties, and use distribution tests for stochastic policies.
Publish the slow compositional, optimized single-device and sharded results separately. Long-context, image training,
native low-precision backward and speculative decoding earn their own results.
