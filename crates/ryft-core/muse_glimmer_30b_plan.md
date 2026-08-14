# Muse Glimmer 30B Training Support Plan

This document is the sibling of [gemma_4_plan.md](gemma_4_plan.md) for Meta's **Muse Glimmer
30B** (released 2026-08-10 by Meta Superintelligence Labs under Apache 2.0). It captures (1)
the model's architecture and the operation inventory required to train it end-to-end in
`ryft`, marked against what exists in `ryft-core` / `ryft-xla` today, (2) the implementation
plan, and (3) a target `ryft` implementation of the components that differ from Gemma 4.

Because Muse Glimmer is deliberately Gemma-like ("similar to Gemma 3 27B and Gemma 4 31B, but
with tweaks"), this plan leans on the Gemma 4 plan for everything shared and details only the
deltas. The headline result is stated up front: **the text model requires zero new IR
primitives** — every operation it needs already exists in `ryft-core` with StableHLO
lowerings in `ryft-xla`. The gaps are exactly the ones the Gemma 4 plan's revised phases
already track (optimizer module, model crate, mixed-precision policy), plus the same
vision-encoder convolution gap, and one new optional workstream (the DFlash speculative
drafter).

Sources consulted:

- [Muse Glimmer 30B model card (Hugging Face)](https://huggingface.co/meta-models/Muse-Glimmer-30B)
  and the [assistant variant](https://huggingface.co/meta-models/Muse-Glimmer-30B-assistant).
- [Muse Glimmer 30B Architecture Notes — Sebastian Raschka](https://sebastianraschka.com/blog/2026/muse-glimmer-30b-architecture-notes.html).
- [Hugging Face release blog](https://huggingface.co/blog/muse-glimmer).
- [NVIDIA NIM model card](https://build.nvidia.com/meta/muse-glimmer-30b/modelcard) and
  [API reference](https://docs.api.nvidia.com/nim/reference/meta-muse-glimmer-30b).
- [SGLang day-0 support post](https://www.lmsys.org/blog/2026-08-10-meta-muse-glimmer) and
  [vLLM recipe](https://recipes.vllm.ai/meta-models/Muse-Glimmer-30B).
- [VentureBeat](https://venturebeat.com/technology/meta-returns-to-open-source-with-muse-glimmer-an-apache-2-0-licensed-30b-parameter-ai-model-optimized-for-agents-available-now)
  and [Neowin](https://www.neowin.net/news/meta-releases-muse-glimmer-a-30b-open-agentic-ai-model-that-runs-locally-on-pcs/)
  release coverage.
- Referenced papers: perception encoder ([arXiv:2504.13181](https://arxiv.org/abs/2504.13181));
  DFlash drafter ([arXiv:2602.06036](https://arxiv.org/abs/2602.06036)).

---

## 1. Architecture Summary & Operation Inventory

### 1.1 Architecture summary

Dense causal decoder-only transformer plus a ViT perception encoder; ~29.6B total parameters;
vocabulary 202,048 (200,000 BPE + 2,048 special tokens); context 131,072+ combined
input/output tokens; BF16 training precision; distilled from the larger Muse Spark.

| Component | Value |
|---|---|
| Layers | 52, pattern `(SWA, SWA, SWA, Full)` × 13 |
| Hidden dimension | 6,656 |
| FFN | SwiGLU, intermediate dimension 19,968 |
| Attention | Gated GQA: 32 query heads / 2 KV heads (16:1), head dim 128 |
| Sliding window | 2,048 tokens on the SWA layers |
| Positional encoding | RoPE (θ = 500,000) on SWA layers **only**; the Full layers use **NoPE** (no positional encoding) |
| QK-norm | RMS normalization per query/key head, plus an extra query scale acting as an inverse softmax temperature |
| Norm placement | Gemma-style pre + post RMSNorm around each sub-block |
| Vision encoder | ViT-G/14, ~1.8B params, 50 layers, width 1536, patch size 14; 2×2 pixel-shuffle merge (4× token reduction) then projection into the decoder embedding space; up to 4,096 visual tokens per image |
| Drafter | DFlash block drafter: 5 layers, 32Q/8KV heads, predicts 16-token blocks verified in parallel by the main model |

The architecturally interesting deltas vs Gemma 4:

1. **Gated attention.** A sigmoid gate is applied to the attention output to control how much
   attention information enters the residual stream — for both the SWA and global layers.
   (The public sources do not pin down whether the gate is computed per-head or elementwise,
   nor whether it applies before or after the output projection; the Qwen3-style convention is
   `attn_out * sigmoid(x @ W_gate)` applied before `o_proj`, elementwise. **Verify against the
   HF `transformers` implementation during R3'** — see §5.)
2. **SwiGLU instead of GeGLU** — `silu(x @ W_gate) * (x @ W_up)` where
   `silu(x) = x * sigmoid(x)`.
3. **NoPE global layers.** Instead of Gemma 4's partial-RoPE-with-huge-θ global layers, the
   every-fourth full-attention layer simply applies no positional encoding, and the SWA layers
   carry full RoPE at θ = 500,000. (Same family of ideas as Llama-4's iRoPE.)
4. **Extreme GQA (32:2) + small window (2,048)** — the point of the design: per-token KV cache
   is ~52 KiB in BF16, an order of magnitude below peers.
5. **No per-layer-input embeddings, no logit softcap documented, single dense FFN** — the
   Gemma 4 PLE/softcap machinery drops out. (Softcap absence and embedding tying are
   unconfirmed in public sources; verify during R3'.)

### 1.2 Operation inventory (delta view against `ryft` today)

The full per-primitive tables live in [gemma_4_plan.md §1](gemma_4_plan.md); everything marked
✅ there applies here unchanged. This table covers each Muse Glimmer component and the
`ryft` operations it needs:

| Component | Operations needed | State | Notes |
|---|---|---|---|
| Token embedding (202K vocab) | `Gather`; scale via `Mul` | ✅ | identical to Gemma path; whether the embedding is tied to the LM head is unverified — both variants are expressible (`Transpose` + `Dot` if tied) |
| RMSNorm (pre+post, QK-norm) | `Mul`, `Reduce(Mean)`, `Rsqrt`, `ConvertElementType` | ✅ | same `rms_norm` helper as the Gemma plan §3.3 |
| Extra query scale (inverse temperature) | `Mul` by a constant | ✅ | trivial |
| RoPE (θ = 500,000, full head dim, SWA layers) | `Iota`, `Pow`, `Sin`, `Cos`, `Slice`, `Concatenate`, `Mul`, `Add`, `Sub` | ✅ | the Gemma `apply_rope` helper with `rotated_dim = head_dim` and no +inf padding; NoPE layers simply skip the call |
| Attention core (GQA 32:2, causal, sliding window 2,048) | fused `DotProductAttentionOperation` | ✅ | native fit: query `[B, T, 32, 128]` over KV `[B, S, 2, 128]` (kv_heads divides heads), `AttentionMask::Causal`, `with_sliding_window(2048)` on SWA layers, no window on Full layers; cuDNN FMHA lowering |
| Attention output gate | `Dot` (gate projection), `Logistic`, `Mul` | ✅ | applied outside the fused attention op, before the residual add |
| SwiGLU MLP | `Dot` ×3, `Logistic`, `Mul` | ✅ | `silu(x) = x * logistic(x)`; a two-line `silu` helper next to the Gemma plan's `gelu` |
| LM-head cross-entropy over 202K vocab | `Dot`, `Reduce(Max/Sum)`, `Exp`, `Log`, one-hot contraction | ✅ | same `logsumexp` composition as the Gemma plan §3.5 |
| Distillation loss (from Muse Spark logits) | `Exp`, `Log`, `Reduce`, `Mul`, `Sub` | ✅ | forward-KL on log-softmax outputs is a composition of existing ops; only relevant if reproducing Meta's distillation recipe rather than fine-tuning |
| Autodiff / vmap / jit / remat | `differentiate_at`, `batch`, `jit`, `rematerialize` | ✅ | unchanged from Gemma plan §1.9 |
| Collectives / sharding for 30B-scale training | `psum`/`all_gather`/`psum_scatter`/`reshard`/`shard_map` | ✅ | unchanged; 30B dense wants FSDP-style `psum_scatter` + `all_gather` which exist |
| Optimizer (AdamW + clip-by-global-norm) | parameter-tree pure functions | ❌ | **same gap as Gemma plan R1** — nothing model-specific |
| Mixed-precision policy (bf16/fp32) | `ConvertElementType`, `dot_with_accumulation_type` | ❌ | same gap as Gemma plan |
| Vision tower (ViT-G/14) | **`Convolution`** (patch embed) + attention + MLP + `Reshape`/`Transpose` (pixel shuffle) | ⚠️ | patch embedding needs the one missing primitive (`stablehlo.convolution` lowering — Gemma plan R5); everything after the patch embed exists. The 14×14/stride-14 patch conv is also expressible as `Reshape` + `Transpose` + `Dot` (space-to-depth then matmul), which removes the conv dependency entirely — see §2 |
| Pixel-shuffle 2×2 merge + projection | `Reshape`, `Transpose`, `Dot` | ✅ | pure data movement |
| KV cache (inference) | `DynamicSliceOperation`, `DynamicUpdateSliceOperation` | ✅ | exists; the 32:2 GQA + 2,048-window design is what makes it small |
| DFlash drafter (training + speculative decode) | standard transformer ops; verification loop = `While`/`Scan`, `ArgMax`, `Compare`, `Select` | ⚠️ | no new primitives expected, but the DFlash block-drafting objective needs the [paper](https://arxiv.org/abs/2602.06036) details; optional workstream (§2 M4) |

### 1.3 Summary checklist

- [x] Every text-model operation: embeddings, RMSNorm/QK-norm, RoPE-θ500K, NoPE, fused
  gated-GQA attention with sliding window, SwiGLU, cross-entropy, distillation-KL
- [x] All transforms: `differentiate_at`, `batch`, `jit`, `rematerialize`, collectives
- [x] KV-cache and speculative-verification ops for inference
- [ ] Optimizer module — **shared gap with Gemma plan R1**
- [ ] Mixed-precision policy — shared gap
- [ ] Model crate + NN helpers — shared gap (R2); Muse Glimmer adds `silu` and the attention
  gate to the helper set
- [ ] Vision tower: either a `ConvolutionOperation` (Gemma plan R5) **or** the conv-free
  space-to-depth + `Dot` patch embedding (§2 M2)
- [ ] DFlash drafter workstream (optional, inference acceleration)

---

## 2. Implementation Plan

Muse Glimmer shares R1 (optimizer), R2 (model crate + NN helpers), and the validation
methodology of R4 with the Gemma 4 plan — those are one-time investments that serve both
models. The Muse-specific work:

### M1 — Muse-specific NN helpers (small)

Add to `ryft-models::common::nn` beyond the Gemma helpers:

- `silu(x) = x * logistic(x)` (two lines; `Logistic` exists).
- `gated_attention_output(x, attention_out, gate_weights)` =
  `attention_out * logistic(x.dot(gate_weights))` — placement/granularity confirmed against
  the HF reference first (§5).
- The Gemma `apply_rope` helper is reused with `rotated_dim = head_dim`; NoPE is the absence
  of a call, not a helper.

### M2 — Model implementation (`crates/ryft-models/src/muse_glimmer/`)

Same file layout as the Gemma plan's Phase 8: `config.rs`, `params.rs`, `attention.rs`,
`mlp.rs`, `block.rs`, `forward.rs`, `loss.rs`, `train.rs`. §3 sketches the components that
differ from Gemma. Two structural simplifications vs Gemma 4: no per-layer-input embedding
stream, and only two norms of interest inside attention (QK-norm) beyond the pre/post block
norms.

For the vision tower, prefer the **conv-free patch embedding**: a stride-14 patch conv over
`[B, 3, H, W]` is exactly `Reshape` (space-to-depth into `[B, (H/14)*(W/14), 3*14*14]`) +
`Dot` with a `[588, 1536]` weight — mathematically identical, uses only existing primitives,
and removes the R5 convolution dependency from this model entirely. Implement the remaining
49 ViT layers with the same fused attention + MLP helpers (no sliding window, no causal mask —
`AttentionMask::None`). If Meta's training recipe freezes the perception encoder (it is frozen
at deployment; training status unverified), the tower only ever runs forward, which also
removes it from the autodiff path.

### M3 — Training step & validation

Identical shape to the Gemma plan's R3/R4: `differentiate_at(model).with_captures(batch)
.value_and_gradient(loss)` + R1's `clip_by_global_norm` + `adamw_step`, wrapped in
`rematerialize` per block with a dots-saveable policy. Validation oracles:

1. Forward-logit parity against the released checkpoint through the HF `transformers`
   implementation (bf16 tolerance `1e-3`), text-only first, then interleaved image+text.
2. One-optimizer-step parameter-delta parity against a PyTorch reference fine-tune step.
3. Short LoRA-style or full fine-tune loss-curve comparison — Apache 2.0 licensing makes this
   model unusually convenient as a permanent regression fixture for `ryft-models`.

### M4 — DFlash drafter (optional, decoupled)

The drafter is a separate 5-layer model that predicts 16-token blocks which the main model
verifies in parallel. Training it and running speculative decoding are inference-acceleration
work, not base-model-training work, so this phase is optional and decoupled:

- Drafter architecture: standard transformer ops only (already covered).
- Training objective: per [arXiv:2602.06036](https://arxiv.org/abs/2602.06036) — block-parallel
  draft prediction conditioned on the target model's hidden state. Details to be taken from
  the paper + released drafter weights; do not improvise the conditioning interface.
- Speculative decode loop: `While` + `DynamicUpdateSlice` (KV append) + `ArgMax`/`Compare`/
  `Select` for accept/reject — all existing primitives.

### M5 — Low-precision training (points at Gemma plan §4)

Everything in [gemma_4_plan.md §4](gemma_4_plan.md) applies verbatim — `BlockQuantize` +
`ScaledDot` with the `__op$block_scaled_dot` CUDA lowering and portable fallback are
model-agnostic. The Muse-specific per-role mapping:

- **NVFP4 on the MLP GEMMs** (`6656 × 19968`, the dominant FLOPs; contracting axes divisible
  by 16 ✓).
- **FP8 E4M3 on Q/O projections** (`6656 × 4096`); the K/V projections are so small under
  32:2 GQA (`6656 × 256`) that quantizing them is not worth the numerics risk — keep bf16.
- **LM head** (`6656 × 202048`): FP8-quantize the contraction like Gemma's unembedding; the
  202,048 vocab axis is the output axis, not the contracting axis, so no block-alignment
  concern.
- Norms, softmax (inside the fused attention op), gates, and the tiny KV projections stay
  bf16/fp32; optimizer state fp32 — same rules of thumb as Gemma plan §4.4.

---

## 3. Target `ryft` Implementation (deltas from the Gemma 4 plan)

Written in the current idiom (generic `A: ArrayOperations`, per
[crates/ryft/examples/mlp.rs](../ryft/examples/mlp.rs)); the same caveats as
[gemma_4_plan.md §3](gemma_4_plan.md) apply — helpers like `constant_like`/`slice_axis` are
R2/M1 conveniences, and the model crate does not exist yet. Components identical to the Gemma
plan (`rms_norm`, `apply_rope`, `logsumexp`, loss, `train_step`, driver) are not repeated.

### 3.1 Configuration

```rust
// crates/ryft-models/src/muse_glimmer/config.rs

/// Attention flavor of one layer in the repeating `(Swa, Swa, Swa, Full)` pattern.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// Sliding-window attention with RoPE.
    Swa,

    /// Full (global) attention with no positional encoding (NoPE).
    Full,
}

#[derive(Clone, Debug)]
pub struct MuseGlimmerConfig {
    pub layer_count: usize,            // 52
    pub embed_dim: usize,              // 6656
    pub query_head_count: usize,       // 32
    pub kv_head_count: usize,          // 2
    pub head_dim: usize,               // 128
    pub mlp_hidden_dim: usize,         // 19968
    pub vocab_size: usize,             // 202_048
    pub sliding_window_size: usize,    // 2048
    pub rope_base_frequency: f64,      // 500_000.0
    pub query_scale: f64,              // post-QK-norm inverse softmax temperature (value: HF config)
    pub rms_norm_epsilon: f64,         // verify against HF config
}

impl MuseGlimmerConfig {
    /// The released 30B configuration. Layer kinds follow `(Swa, Swa, Swa, Full)` × 13.
    pub fn glimmer_30b() -> Self { /* values above */ }

    pub fn layer_kind(&self, layer_index: usize) -> LayerKind {
        if (layer_index + 1) % 4 == 0 { LayerKind::Full } else { LayerKind::Swa }
    }
}
```

### 3.2 Parameter tree (attention delta only)

```rust
// crates/ryft-models/src/muse_glimmer/params.rs
#[derive(Clone, Debug, Parameterized)]
pub struct AttentionParams<P: Parameter> {
    /// Query projection, `[embed_dim, query_head_count * head_dim]` = `[6656, 4096]`.
    pub q_proj: P,

    /// Key projection, `[embed_dim, kv_head_count * head_dim]` = `[6656, 256]`.
    pub k_proj: P,

    /// Value projection, `[6656, 256]`.
    pub v_proj: P,

    /// Output-gate projection feeding the sigmoid gate, `[embed_dim, query_head_count *
    /// head_dim]` (elementwise gate; shrink to `[embed_dim, query_head_count]` if the HF
    /// reference gates per-head — verify, §5).
    pub gate_proj: P,

    /// Output projection, `[query_head_count * head_dim, embed_dim]` = `[4096, 6656]`.
    pub o_proj: P,

    /// Per-head QK-norm scales.
    pub query_norm: RmsNormScale<P>,
    pub key_norm: RmsNormScale<P>,
}
```

`BlockParams` mirrors the Gemma tree minus PLE and skip-scale (verify skip-scale absence
against the HF config), with `MlpParams { gate_proj, up_proj, down_proj }` unchanged in shape
(`[6656, 19968]` ×2 and `[19968, 6656]`).

### 3.3 Gated attention with the SWA/NoPE split

```rust
// crates/ryft-models/src/muse_glimmer/attention.rs
use ryft::*;

use crate::common::nn::{apply_rope, rms_norm, silu_gate};
use crate::muse_glimmer::{AttentionParams, LayerKind, MuseGlimmerConfig};

pub fn attention<A: ArrayOperations>(
    config: &MuseGlimmerConfig,
    kind: LayerKind,
    params: &AttentionParams<A>,
    x: &A,          // [batch, seq, embed_dim]
    positions: &A,  // [batch, seq]
) -> Result<A, ProgramError> {
    let (heads, kv_heads, head_dim) = (config.query_head_count, config.kv_head_count, config.head_dim);
    let queries = project(x, &params.q_proj, heads)?;    // [B, T, 32, 128]
    let keys = project(x, &params.k_proj, kv_heads)?;    // [B, T, 2, 128]
    let values = project(x, &params.v_proj, kv_heads)?;

    // QK-norm per head, then the extra query scale (inverse softmax temperature).
    let queries = rms_norm(&queries, Some(&params.query_norm.scale), config.rms_norm_epsilon)?;
    let keys = rms_norm(&keys, Some(&params.key_norm.scale), config.rms_norm_epsilon)?;
    let queries = queries * constant_like(&queries, config.query_scale)?;

    // Positional encoding: full RoPE on SWA layers, nothing (NoPE) on Full layers.
    let (queries, keys, sliding_window) = match kind {
        LayerKind::Swa => (
            apply_rope(&queries, positions, head_dim, config.rope_base_frequency)?,
            apply_rope(&keys, positions, head_dim, config.rope_base_frequency)?,
            Some(config.sliding_window_size),
        ),
        LayerKind::Full => (queries, keys, None),
    };

    // Fused attention: 32:2 GQA is native (kv_heads divides heads). The QK-norm query scale
    // above already set the logit temperature, so the kernel's own scale is 1.
    let context = queries.dot_product_attention(&keys, &values, /*scale=*/ 1.0, AttentionMask::Causal, sliding_window)?;

    // Output gate: sigmoid of a learned projection of the layer input, applied elementwise to
    // the attention output before the output projection (placement per HF reference — §5).
    let matmul = DotDimensionNumbers::new(vec![2], vec![0], vec![], vec![]);
    let flat_context = context.reshape(flat_heads_shape(x, heads * head_dim))?;
    let gate = x.dot(&params.gate_proj, &matmul).logistic()?;
    Ok((flat_context * gate).dot(&params.o_proj, &matmul))
}
```

### 3.4 SwiGLU MLP

```rust
// crates/ryft-models/src/muse_glimmer/mlp.rs
/// SwiGLU: `silu(x W_gate) * (x W_up) W_down`, with `silu(x) = x * logistic(x)`.
pub fn mlp<A: ArrayOperations>(params: &MlpParams<A>, x: &A) -> Result<A, ProgramError> {
    let matmul = DotDimensionNumbers::new(vec![2], vec![0], vec![], vec![]);
    let gate = x.dot(&params.gate_proj, &matmul);
    let gate = gate.clone() * gate.logistic()?;   // silu
    Ok((gate * x.dot(&params.up_proj, &matmul)).dot(&params.down_proj, &matmul))
}
```

Everything else — block wiring (pre/post RMSNorm + residual), forward, loss, `train_step`,
rematerialization, and the driver — is byte-for-byte the Gemma plan §3.4–§3.6 shape minus the
PLE injection and the final logit softcap (pending §5 verification of the latter's absence).

---

## 4. Low-Precision Training on Blackwell

Fully covered by [gemma_4_plan.md §4](gemma_4_plan.md); the primitives (`BlockQuantize`,
`ScaledDotOperation`, `__op$block_scaled_dot` lowering + portable fallback) are implemented
and model-agnostic. Muse-specific notes are in §2 M5: NVFP4 on the MLP GEMMs, FP8 on Q/O
projections and the LM-head contraction, bf16 for the tiny 32:2-GQA K/V projections, norms,
softmax, and gates. The remaining recipe extras (stochastic rounding, RHT, delayed-scaling
amax tracker) are tracked in the Gemma plan §4.2 and apply here identically.

---

## 5. Risks & Open Questions

- **Undocumented details to verify against the HF `transformers` implementation before M2**
  (public sources do not specify them; do not improvise):
  - Gate computation: input to the gate projection (layer input vs attention output),
    elementwise vs per-head, applied before vs after `o_proj`.
  - Embedding tying (tied like Gemma vs untied like Llama) and any embedding scaling.
  - Presence/absence of a final logit softcap (none documented — likely absent).
  - RMSNorm epsilon, zero-centered vs plain scale, exact pre/post placement, and whether a
    Gemma-style learnable residual skip-scale exists.
  - The `query_scale` value and whether the fused-attention kernel scale should then be 1.
- **Fused attention + gating interplay.** The gate lives outside the fused
  `DotProductAttentionOperation`, so no kernel change is needed — but confirm the cuDNN FMHA
  path is still profitable at a 2,048-token window with 32:2 GQA, or whether the unfused
  composition (also fully supported) wins at that shape.
- **Vision tower training status.** Frozen at deployment; if it is also frozen for
  fine-tuning (likely, given "distilled from Muse Spark"), the tower is forward-only and the
  conv-free patch embedding (§2 M2) makes it runnable today with zero new primitives.
- **Distillation recipe.** Reproducing Meta's Muse Spark distillation requires teacher logits
  we don't have; the practical target for `ryft` is fine-tuning the released checkpoint, where
  ordinary cross-entropy suffices.
- **DFlash drafter conditioning interface.** The drafter consumes target-model hidden state;
  the exact interface must come from [arXiv:2602.06036](https://arxiv.org/abs/2602.06036) and
  the released drafter weights — flagged as the M4 unknown.
- **Shared gaps with the Gemma plan** (not Muse-specific): optimizer module, mixed-precision
  policy, model crate, `CompiledXlaFunction::batch` stub (vmap-of-jit only).

---

## 6. References

### 6.1 Model documentation

- [Muse Glimmer 30B model card](https://huggingface.co/meta-models/Muse-Glimmer-30B) — the
  authoritative hyperparameters (52 layers, 6656 hidden, 32/2 heads @128, SwiGLU 19,968,
  vocab 202,048, RoPE θ=500K local-only, ViT-G/14 details, DFlash drafter specs).
- [Muse-Glimmer-30B-assistant](https://huggingface.co/meta-models/Muse-Glimmer-30B-assistant)
  — instruction-tuned variant.
- [Sebastian Raschka: Muse Glimmer 30B Architecture Notes](https://sebastianraschka.com/blog/2026/muse-glimmer-30b-architecture-notes.html)
  — gated attention, QK-norm + query-scale, `(SWA,SWA,SWA,Full)`/NoPE analysis, KV-cache
  comparison, Gemma 3/4 and Qwen3.6 comparisons.
- [Hugging Face release blog](https://huggingface.co/blog/muse-glimmer) — perception-encoder
  pixel-shuffle integration, fine-tuning notebook.
- [NVIDIA NIM model card](https://build.nvidia.com/meta/muse-glimmer-30b/modelcard) /
  [API reference](https://docs.api.nvidia.com/nim/reference/meta-muse-glimmer-30b) — frozen
  perception encoder at deployment, context/visual-token limits.
- [SGLang day-0 support](https://www.lmsys.org/blog/2026-08-10-meta-muse-glimmer),
  [vLLM recipe](https://recipes.vllm.ai/meta-models/Muse-Glimmer-30B),
  [unsloth GGUF](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF) — serving ecosystem.
- [The Kaitchup: Muse Glimmer analysis](https://kaitchup.substack.com/p/muse-glimmer-metas-30b-model-built)
  (paywalled beyond the overview).
- Release coverage: [VentureBeat](https://venturebeat.com/technology/meta-returns-to-open-source-with-muse-glimmer-an-apache-2-0-licensed-30b-parameter-ai-model-optimized-for-agents-available-now),
  [Neowin](https://www.neowin.net/news/meta-releases-muse-glimmer-a-30b-open-agentic-ai-model-that-runs-locally-on-pcs/),
  [NVIDIA technical blog](https://developer.nvidia.com/blog/run-local-agentic-ai-workflows-with-metas-muse-glimmer-on-nvidia/).

### 6.2 Papers

- [Perception encoder (arXiv:2504.13181)](https://arxiv.org/abs/2504.13181) — the ViT-G/14
  perception-encoder lineage.
- [DFlash (arXiv:2602.06036)](https://arxiv.org/abs/2602.06036) — the block-drafting
  speculative-decoding method behind the 16-token drafter.

### 6.3 In-repo references

- [gemma_4_plan.md](gemma_4_plan.md) — the companion plan this document delta-references for
  the full operation inventory (§1), phase plan (§2), shared model code (§3), and the
  NVFP4/FP8 stack (§4).
- [crates/ryft-core/src/operations/attention.rs](src/operations/attention.rs) — the fused
  `DotProductAttentionOperation` whose GQA + sliding-window support makes the Muse attention
  core a single call.
- [crates/ryft-core/src/operations/math/logistic.rs](src/operations/math/logistic.rs) — the
  `Logistic` primitive underlying both SwiGLU and the attention output gate.
- [crates/ryft/examples/mlp.rs](../ryft/examples/mlp.rs) — the training-loop idiom §3 follows.
