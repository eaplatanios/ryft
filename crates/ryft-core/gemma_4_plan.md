# Gemma 4 Training Support Plan

This document captures (1) the operation inventory needed to train Google's Gemma 4 family
end-to-end inside `ryft`, marking which primitives are already in `ryft-core` / `ryft-xla` and
which still need to be added, (2) the high-level implementation plan that would follow once the
missing primitives land, and (3) a target `ryft` model implementation written against that API
surface.

> **Revision status.** This plan was originally written against the earlier `tracing_v2`-centric
> architecture. The codebase has since been rebuilt around the typed `programs` IR,
> context-based eager dispatch, the builder-style differentiation API (`differentiate_at`),
> backend-neutral `compilation`/JIT with structural and disk caching, first-class dynamic
> dimensions, and a hugely expanded operation set under `crates/ryft-core/src/operations/` —
> and **nearly every primitive this plan listed as missing has landed**, including the fused
> `DotProductAttentionOperation` (with causal masking, sliding windows, GQA, and a cuDNN FMHA
> lowering), the complete `ScaledDotOperation` + `BlockQuantize` NVFP4/MX path from §4, device
> RNG (ThreeFry/Philox), collectives, `ScanOperation`, and JAX-parity rematerialization
> policies with offload support. The inventory tables and phase statuses below have been
> updated in place. The remaining work is concentrated in §2's revised phase list — the model
> crate itself, an optimizer module, a mixed-precision policy helper, and (for the multimodal
> path only) a convolution primitive — rather than in IR primitives.

The architecture reference is the published Gemma 4 family (E2B, E4B, 26B-A4B mixture-of-experts,
31B dense flagship). The salient features that drive the inventory are: grouped-query attention
with QK-norm and partial RoPE, four RMSNorms per block (with zero-centered learnable scale),
local–global sliding-window alternation, per-layer-input embeddings (PLE), KV-cache sharing,
tied input/output embedding with a final logit softcap of `30.0`, and (for the MoE variant) a
ragged top-k expert routing path alongside a dense branch.

Canonical reference implementations and model documentation consulted while compiling the
inventory:

- Gemma 4 launch and product documentation:
  [Gemma 4 on Google DeepMind](https://deepmind.google/models/gemma/gemma-4/) and the
  [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4).
- Google DeepMind JAX/Flax-Linen: [`google-deepmind/gemma`](https://github.com/google-deepmind/gemma),
  in particular `gemma/gm/nn/gemma4/{_modules.py,_transformer.py,_gemma4.py}`,
  `gemma/gm/nn/_layers.py`, and `gemma/gm/math/_positional_embeddings.py`.
- HuggingFace Transformers Gemma 3 module:
  [`transformers/src/transformers/models/gemma3`](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma3).
- PyTorch reference: [`google/gemma_pytorch`](https://github.com/google/gemma_pytorch).
- Flax NNX example walkthrough:
  [Flax NNX Gemma tutorial](https://flax.readthedocs.io/en/stable/examples/gemma.html).
- Gemma 3 technical report (architectural ancestor): [arXiv:2503.19786](https://arxiv.org/abs/2503.19786).
- [Gemma explained: what's new in Gemma 3](https://developers.googleblog.com/en/gemma-explained-whats-new-in-gemma-3/)
  (Google Developers blog).

A consolidated list of every source cited inline below is collected in §6 References.

---

## 1. Operation Inventory

Each row lists the JAX/StableHLO primitive that appears (directly or indirectly) in a Gemma 4
forward + backward + optimizer step, the role it plays, and its support state in `ryft`. "✅" means
the value-level trait, the staged operation, the differentiation rules, and the XLA lowering are
all in place today; "⚠️" means partially supported (typically: traced but not yet lowered, or
trait-level only); "❌" means missing entirely. The "Crate target" column indicates where the
missing piece must land (`ryft-core` for the IR primitive plus tracing/autodiff rules,
`ryft-xla` for the StableHLO lowering and PJRT execution glue, and `ryft-core+ryft-xla` when
both are needed).

### 1.1 Elementwise arithmetic & scalar math

All landed. Every op lives in `crates/ryft-core/src/operations/math/` following the
`XxxOperation` struct + `Xxx` capability trait pattern, with blanket impls that give the
reference `Array` backend and every transform tracer the same method surface, and StableHLO
lowerings in `crates/ryft-xla/src/experimental/lowering.rs`.

| Primitive (JAX) | Used by | State | Notes |
|---|---|---|---|
| `lax.add` / `sub` / `mul` / `div` / `neg` / `rem` | residual stream, masks, optimizer updates | ✅ | `AddOperation` … `RemOperation`; both `std::ops` sugar and fallible capability traits |
| scalar scaling | RoPE base-frequency multiply, `sqrt(d_model)` embedding multiply | ✅ | plain `Mul` against a filled/broadcast scalar (the old dedicated `ScaleOperation` is gone) |
| `lax.sin` / `cos` / `atan2` | RoPE rotation | ✅ | `SinOperation`, `CosOperation`, `Atan2Operation` |
| `lax.rsqrt` | RMSNorm `x * rsqrt(mean(x^2) + eps)` | ✅ | `RsqrtOperation`, `Rsqrt::rsqrt` |
| `lax.sqrt` | AdamW `sqrt(v_hat) + eps`, gradient global norm | ✅ | `SqrtOperation` |
| `lax.exp` / `log` | softmax, `logsumexp` for cross-entropy | ✅ | `ExpOperation`, `LogOperation` (no `log1p`/`expm1` in core; not needed for Gemma) |
| `lax.tanh` | final logit softcap (`30 * tanh(logits / 30)`), GELU tanh-approx | ✅ | `TanhOperation` |
| `lax.erf` | exact GELU (`0.5 * x * (1 + erf(x / sqrt(2)))`) | ✅ | `ErfOperation`, lowered via `chlo.erf` |
| `lax.logistic` | sigmoid (router gates in the MoE variant) | ✅ | `LogisticOperation` |
| `jax.nn.gelu` | GeGLU activation in MLP | ⚠️ | still a model-level composition on `erf` (or `tanh`); all ingredients exist |
| `lax.abs` / `sign` | gradient global norm, clip sentinels | ✅ | `AbsOperation`, `SignOperation` |
| `lax.pow` | `g^2`, learning-rate schedules | ✅ | `PowOperation` (single op; no separate `integer_pow`) |
| `lax.max` / `min` (binary) | gradient clipping `min(1, clip/norm)`, bias floors | ✅ | `MaxOperation`, `MinOperation` |
| `lax.clamp` | logit floor/ceil, MX quantization clamping | ✅ | `Clamp` capability trait composed from `Max`+`Min` (no dedicated primitive — intentional) |
| `lax.floor` / `ceil` / `round` | quantization recipes, schedules | ✅ | `FloorOperation`, `CeilOperation`, `RoundOperation` (round-to-nearest-even lowering) |
| `lax.convert_element_type` | bf16 ↔ fp32 casts between forward, accumulation, and optimizer state | ✅ | `ConvertElementTypeOperation` + `promote_element_type`. **No rounding-mode parameter yet** — relevant only to §4's stochastic-rounding item |
| complex ops | not needed by Gemma; listed for completeness | ✅ | `Complex`, `Conjugate`, `Real`, `Imaginary` |

### 1.2 Comparisons & boolean logic (mask construction)

All landed, exactly in the single-carrier shape the original plan recommended.

| Primitive (JAX) | Used by | State | Notes |
|---|---|---|---|
| `lax.eq`/`ne`/`lt`/`le`/`gt`/`ge` | padding masks, causal mask, sliding-window mask | ✅ | one `CompareOperation` with `ComparisonDirection::{Equal, NotEqual, LessThan, LessThanOrEqual, GreaterThan, GreaterThanOrEqual}` |
| `lax.bitwise_and`/`or`/`not`/`xor` | mask combination (causal AND sliding), padding NOT | ✅ | `AndOperation`, `OrOperation`, `NotOperation`, `XorOperation` in `operations/logical/` |
| `lax.select_n` (a.k.a. `where`) | applying mask to logits, padding loss-mask | ✅ | `SelectOperation` |

Note that hand-built attention masks are now needed only for exotic cases: the fused
`DotProductAttentionOperation` (§1.5) carries causal masking and sliding windows natively.

### 1.3 Reductions

Landed as the single-carrier `ReduceOperation` the original plan recommended, with `Mean` as a
bonus first-class kind (so RMSNorm's `mean(x^2)` needs no helper).

| Primitive (JAX) | Used by | State | Notes |
|---|---|---|---|
| `lax.reduce_sum` | softmax denominator, cross-entropy sum, grad norm | ✅ | `value.reduce(&axes, ReductionKind::Sum)`; also `reduce_with_output_sharding` |
| `mean` | RMSNorm `mean(x^2)` | ✅ | first-class `ReductionKind::Mean` |
| `lax.reduce_max` / `reduce_min` | softmax numerical stability | ✅ | `ReductionKind::Max` / `Min` |
| `any` / `all` | mask diagnostics | ✅ | `ReductionKind::Any` / `All` |
| `lax.reduce_prod` | (not used by Gemma 4) | ❌ | no `Prod` kind; not needed |
| `argmax` / `argmin` | sampling, top-1 accuracy | ✅ | `ArgMax::argmax(axis)` / `ArgMin::argmin(axis)` in `operations/sort.rs`, composed from `Sort` |
| cumulative ops (`cumsum`) | (not used by Gemma 4 training) | ❌ | absent; not needed |
| `logsumexp` / `softmax` (composite) | cross-entropy; standalone softmax | ⚠️ | still model-level compositions (`reduce_max` + `sub` + `exp` + `reduce` + `log`); the fused attention op embeds its own softmax, so only the LM-head loss needs the composition |

### 1.4 Shape & data movement

All landed (in `operations/manipulation/` and `operations/constants/`), plus dynamic-shape
variants the original plan never asked for (this branch adds first-class runtime dimensions —
`DimensionOperation`, `ArrayIrOperation` — with `DynamicReshapeOperation`,
`DynamicBroadcastOperation`, and `DynamicShapeSliceOperation`).

| Primitive (JAX) | Used by | State | Notes |
|---|---|---|---|
| `lax.reshape` | folding GQA group dim, flattening for `dot` | ✅ | `ReshapeOperation` (+ `DynamicReshapeOperation`) |
| `lax.transpose` | layout permutations | ✅ | `TransposeOperation` + `Permutation` |
| `lax.broadcast_in_dim` | scale/mask/position broadcasts | ✅ | `BroadcastOperation` (+ `DynamicBroadcastOperation`) |
| `lax.concatenate` | RoPE half re-join, cache concat | ✅ | `ConcatenateOperation` |
| `lax.slice` (static) | RoPE half split, KV-cache prefix slicing | ✅ | `SliceOperation` (+ `DynamicShapeSliceOperation`) |
| `lax.dynamic_slice` / `dynamic_update_slice` | KV-cache reads/writes at runtime offsets | ✅ | `DynamicSliceOperation`, `DynamicUpdateSliceOperation` (+ static `UpdateSliceOperation`) |
| `lax.gather` | token embedding lookup `table[input_ids]` | ✅ | `GatherOperation` with `GatherDimensionNumbers` and `GatherScatterMode::{PromiseInBounds, Clip, FillOrDrop}` |
| `lax.scatter` / scatter-add | gradient of `gather` (embedding bwd), MoE dispatch | ✅ | `ScatterOperation` with `ScatterReductionKind::{Overwrite, Add, Mul, Min, Max}` |
| `lax.pad` | RoPE timescale padding, sequence padding | ✅ | `PadOperation` |
| `lax.iota` | position indices, RoPE arange | ✅ | `IotaOperation` (context-side constructor; + `DynamicIota` for dynamic shapes) |
| `lax.sort` / `top_k` | sampling, MoE expert routing | ✅ | `SortOperation` (multi-key, `SortDirection`) + `TopK::top_k(k, axis)` |
| `lax.rev` (reverse) | (not used by Gemma 4 training) | ❌ | absent as a standalone op (`ScanOperation` has `with_reverse`); not needed |

### 1.5 Tensor contraction, matmul & fused attention

| Primitive (JAX) | Used by | State | Notes |
|---|---|---|---|
| `lax.dot_general` | Q/K/V projections, MLP, vocab head | ✅ | `DotOperation` with `DotDimensionNumbers`, plus `dot_with_accumulation_type` and `dot_with_output_sharding` |
| scaled (block-quantized) dot | FP8/NVFP4 matmuls (§4) | ✅ | `ScaledDotOperation` (`scaled_dot`), see §4 — implemented since the original plan |
| fused dot-product attention | the entire attention core | ✅ | `DotProductAttentionOperation` + `DotProductAttentionBackwardOperation`: query `[batch, q_seq, heads, head_dim]` over key/value `[batch, kv_seq, kv_heads, head_dim]` with `kv_heads` dividing `heads` (**native GQA**), `AttentionMask::{None, Causal}`, `with_sliding_window`, `with_dropout(p, seed)`, optional bias and per-example sequence lengths, and `differentiable_dot_product_attention{,_with_bias,_with_sequence_lengths}` wiring the custom VJP. Lowers to cuDNN FMHA custom calls (`__cudnn$fmha…`) on GPU. This *subsumes* the hand-built logits→mask→softmax→context pipeline the original plan sketched |
| `einsum` (string frontend) | ergonomics only | ❌ | still absent; `dot` + `reshape` + `transpose` cover everything, and the fused attention op removes the largest einsum consumer |
| `lax.conv_general_dilated` | **vision encoder only** (SigLIP patch embedding) | ❌ | no convolution operation anywhere in `ryft-core`/`ryft-xla` (the raw `ryft-mlir` StableHLO bindings have it, but nothing emits it). Text-only training does not need it; the multimodal path does |

### 1.6 Random number generation

Landed in `operations/random.rs`, in exactly the stateless-key shape the plan asked for.

| Primitive (JAX) | Used by | State | Notes |
|---|---|---|---|
| `rng_bit_generator` | the base primitive | ✅ | `RngBitGeneratorOperation` with `RandomAlgorithm::{ThreeFry, Philox}`; lowers to `stablehlo.rng_bit_generator`; host reference kernels (`threefry2x32`, `philox4x32`) keep the CPU backend bit-compatible |
| `jax.random.split` | reproducible streams per layer / per batch | ✅ | `Random::split_key(count) -> (advanced_state, fresh_states)`; keys are plain `u64` state arrays (no dedicated key type) |
| `jax.random.normal` / `uniform` | weight initialization, sampling | ✅ | `Random::normal(shape, data_type)` / `Random::uniform(...)`, each returning `(advanced_state, samples)` |
| `jax.random.categorical` | inference sampling | ✅ | `Random::categorical(logits, axis)` |
| `jax.random.bernoulli` | dropout mask | ⚠️ | composite on `uniform` + `compare`; moreover the fused attention op carries its own `with_dropout(p, seed)`, which is the only dropout Gemma training would use |
| truncated normal init | Flax-default weight init | ⚠️ | still a composition (uniform + erf-inverse or rejection); ordinary `normal` covers the practical need |

### 1.7 Control flow & rematerialization

| Primitive (JAX) | Used by | State | Notes |
|---|---|---|---|
| `lax.cond` | softcap toggle, MoE-vs-dense branch | ✅ | `ConditionOperation` |
| `lax.while_loop` | autoregressive decode, training loop | ✅ | `WhileOperation` with `WhilePredicate` and `with_iteration_bound` |
| `lax.scan` | layer stacking, remat-friendly loops | ✅ | `ScanOperation` with `carry_count`, `length`, `with_reverse`, `with_unroll`, `with_captures` |
| `lax.fori_loop` | optimizer step over parameter tree | ⚠️ | express via `scan`/`while`; a trivial helper if wanted |
| `jax.checkpoint` (remat) | activation checkpointing per block | ✅ | `rematerialize(body)` + `RematerializeOperation` with the full JAX policy family: `EverythingSaveable`, `NothingSaveable`, `DotsSaveable`, `DotsWithNoBatchDimsSaveable`, `SaveOnlyTheseNames`, `SaveAnyNamesButThese`, `SaveAnythingExceptTheseNames`, `SaveFromBothPolicies`, and `OffloadDotsWithNoBatchDims` with `ResidualStorage`/`MemoryTransferStorage` **offload support** — beyond what the plan asked for |
| `jax.custom_jvp` / `custom_vjp` | custom derivative rules (fused attention uses this) | ✅ | `custom_jvp(primal, jvp)` / `custom_vjp` in `differentiation::custom`, lowered as first-class program operations |

### 1.8 Parallelism & sharding

All landed. The collective IR primitives the plan asked for now exist as named-axis operations
in `operations/collectives.rs`, with StableHLO lowerings emitting `all_reduce`, `all_gather`,
`all_to_all`, `reduce_scatter`, `collective_permute`, and `partition_id`.

| Primitive (JAX) | Used by | State | Notes |
|---|---|---|---|
| mesh sharding annotations | data + tensor parallel training | ✅ | `ReshardOperation` (`Reshard`) and `ShardingConstraintOperation` (`ConstrainSharding`), `Sharding`, `DeviceMesh` |
| `shard_map` | MoE dispatch, custom collective regions | ✅ | `ShardMapOperation` (lowered via manual computations in `experimental/shard_map.rs`) |
| `lax.psum` / `pmean` / `pmax` | gradient sync inside `shard_map` | ✅ | `CollectiveOperation` with `CollectiveKind::{PSum, PMean, PMax}` (no `PMin`; not needed) |
| `lax.all_gather` | tensor-parallel gathers | ✅ | `AllGatherOperation`, tiled/untiled modes, `axis_index_groups` |
| reduce-scatter | ZeRO-style gradient sharding | ✅ | `PSumScatterOperation` |
| `lax.ppermute` | pipeline parallelism | ✅ | `PpermuteOperation` (+ `Pshuffle`, `PSwapAxes` conveniences) |
| `lax.all_to_all` | MoE expert exchange | ✅ | `AllToAllOperation` |

### 1.9 Autodiff & training transforms

The transform stack was rebuilt around a single builder entry point:
`differentiate_at(primal).with_captures(..).with_auxiliary_output().value_and_gradient(f)`
(likewise `.jvp`, `.linearize`, `.vjp`, `.gradient`, `.jacobian_forward`, `.jacobian_reverse`,
`.hessian`), with nondifferentiated runtime captures as a first-class concept — see
[crates/ryft/examples/mlp.rs](crates/ryft/examples/mlp.rs) for the canonical usage.

| Capability | Used by | State | Notes |
|---|---|---|---|
| Forward-mode JVP / linearize | per-op linearization | ✅ | `differentiate_at(..).jvp(..)` / `.linearize(..)` → `Pushforward` |
| Reverse-mode VJP | training | ✅ | `.vjp(..)` → `Pullback`; `Program::transpose()` underneath |
| `value_and_gradient`, `gradient` | optimizer step | ✅ | builder terminals; gradients come back as `Input::To<V>` parameter trees |
| Jacobian / Hessian | curvature-aware optimizers (optional) | ✅ | `.jacobian_forward` / `.jacobian_reverse` / `.hessian` |
| `vmap` | per-example losses | ✅ | `batch(function, input, in_axes, out_axes, axis)` in `batching.rs`; nests and composes with staging |
| Activation checkpointing (`remat`) | memory pressure at long context | ✅ | `rematerialize(body)` with the full policy family (§1.7) |
| JIT compile + execute | end-to-end JIT to PJRT | ✅ | backend-neutral `jit`/`stage_function` in `compilation/` (structural + disk caching); XLA side: `compile`/`stage`/`jitted` returning `CompiledXlaFunction`/`JittedXlaFunction`, with `.gradient(&domain)` and `.jvp(&domain)` on compiled functions |
| Batching a *compiled* function | vmap-of-jit | ⚠️ | `CompiledXlaFunction::batch` is a stub (`UnsupportedOperation`); direct `batch` of the uncompiled function works |
| Stateful optimizer (Adam/AdamW) | training | ❌ | **still missing** — no optimizer module anywhere in the workspace |
| Gradient clip-by-global-norm | training stability | ❌ | **still missing**; all ingredients (`reduce`, `sqrt`, `min`, `mul`) exist |
| Mixed-precision policy | bf16 activations / fp32 master weights | ❌ | **still missing**; `ConvertElementType` + `dot_with_accumulation_type` are the ingredients |

### 1.10 Summary checklist

- [x] `add`, `sub`, `mul`, `div`, `neg`, `rem`, `pow`, `sin`, `cos`, `atan2`
- [x] `rsqrt`, `sqrt`, `exp`, `log`, `tanh`, `erf`, `logistic`, `abs`, `sign`, `floor`, `ceil`, `round`, `max`, `min`, `clamp`
- [x] `convert_element_type`
- [x] `compare` (all six directions), `and`, `or`, `not`, `xor`, `select`
- [x] `reduce` (`Sum`, `Mean`, `Max`, `Min`, `Any`, `All`) with axis lists
- [x] `reshape`, `transpose`, `broadcast`, `concatenate`, `slice`, `dynamic_slice`, `dynamic_update_slice`, `pad`, `iota`
- [x] `gather`, `scatter` (with `Add`/`Mul`/`Min`/`Max` reductions), `sort`, `top_k`, `argmax`, `argmin`
- [x] `dot_general` (+ accumulation type and output sharding), `scaled_dot` (block-scaled, §4)
- [x] fused `dot_product_attention` (+ backward) with causal mask, sliding window, GQA, dropout, bias, sequence lengths, cuDNN FMHA lowering
- [x] Device-side RNG (`rng_bit_generator` with ThreeFry/Philox; `split_key`, `normal`, `uniform`, `categorical`)
- [x] `condition`, `while_loop`, `scan`
- [x] Collectives (`psum`/`pmean`/`pmax`, `all_gather`, `psum_scatter`, `ppermute`, `all_to_all`) + `shard_map`, `reshard`, sharding constraints
- [x] Activation checkpointing (`rematerialize` with JAX-parity policies and offload)
- [x] `custom_jvp` / `custom_vjp`, `stop_gradient`, `tag`, `print`, `custom_call`, `transfer_to_memory`
- [x] `jvp`, `linearize`, `vjp`, `gradient`, `value_and_gradient`, Jacobians/Hessians, `batch` (vmap), backend-neutral `jit`
- [x] First-class dynamic dimensions (`DimensionOperation`, `ArrayIrOperation`) — beyond the original plan
- [ ] **Optimizer module** (AdamW + clip-by-global-norm) — no optimizer code exists in the workspace
- [ ] **Mixed-precision policy** helper inserting casts at module boundaries
- [ ] **Model crate** (`RMSNorm`/RoPE/GeGLU helpers, Gemma 4 model, training harness) — §2 revised phases
- [ ] Convolution (`conv_general_dilated`) — needed only for the multimodal vision encoder
- [ ] `einsum` string frontend (optional ergonomics; not blocking)
- [ ] Batching of *compiled* XLA functions (`CompiledXlaFunction::batch` stub)
- [ ] §4 NVFP4 extras beyond the landed `scaled_dot`/`block_quantize`: stochastic rounding, RHT, delayed-scaling amax tracker

---

## 2. Implementation Plan

> **Status rollup.** The primitive-building phases of this plan are complete: Phases 1–5 and 7
> are ✅ done (every op landed with type rules, transform rules, and StableHLO lowerings), and
> Phase 6 is half done (rematerialization ✅ with the full JAX policy family; the optimizer
> module is still ❌ missing). What remains is the model-level work: Phase 0 (model crate),
> the optimizer half of Phase 6, and Phases 8–9. A revised phase list for the remaining work
> follows the original phases below.

The plan is organized so each phase produces something that can be run, tested, and benchmarked
end-to-end against a reference (JAX, PyTorch, or both). Operation hierarchy conventions apply
uniformly (per-op capability traits with blanket impls over `Value`, closed operation families
like `ArrayOperation`, StableHLO lowerings in `ryft-xla::experimental::lowering`). Each new
primitive follows the same five-step contract: type/abstract-eval rule → operation family
variant + capability trait → forward (JVP) rule → transposition (cotangent) rule → StableHLO
lowering. A primitive is not "done" until all five plus per-primitive unit tests are in.

### Phase 0 — Scaffolding the model crate — ❌ OPEN (now the critical path)

1. Add a new crate `crates/ryft-models` (depends on `ryft-core` and `ryft-xla`) to house the
   Gemma 4 implementation, configurations, and integration tests. Keeping models out of
   `ryft-core` preserves the rule that `ryft-core` only owns the IR and transforms.
2. Land a thin `ryft-models::common` module for primitives that are useful across models
   (RMSNorm, RoPE, GeGLU, GQA attention, AdamW, gradient clipping). Each helper is a value-level
   function over `Tracer<'_, D>` (or a generic `V: Traceable<ArrayType>` plus the trait bounds for
   the primitives it uses) so it stages cleanly under any backend.
3. Create `crates/ryft-models/examples/gemma_4_train.rs` as the end-to-end harness.

### Phase 1 — Elementwise math primitives — ✅ DONE

Land the unary float primitives that block everything else: `rsqrt`, `sqrt`, `exp`, `log`, `tanh`,
`erf`, `abs`. Each follows the established pattern of `XxxOperation`, `Xxx` value trait,
`SupportsXxx`, JVP rule, transpose rule (most are non-linear, so they appear in the linear program
only as captured-factor multiplies via `LeftDot` / `RightDot`-style helpers), and `stablehlo.xxx`
lowering. Then land `max`, `min`, `clamp` (binary plus the three-arg `clamp`) — these need a
value-level argmax-style differentiation rule (`grad` flows only through the selected operand).

Land `convert_element_type` with `differentiation` that casts the cotangent back to the source
dtype. This unlocks bf16 forward / fp32 master weights.

### Phase 2 — Comparisons, masks, and `select`-driven masking — ✅ DONE

Add a single `CompareOperation { direction: Eq | Ne | Lt | Le | Gt | Ge }` carrier with a
`SupportsCompare` trait and a tiny value-level surface (`a.eq(b)`, `a.lt(b)`, …). Boolean ops
(`and`, `or`, `not`) follow the same shape. Mask construction in attention (causal AND sliding) is
already lower-bound on `select` — once compares exist, the full mask flow drops out.

### Phase 3 — Reductions — ✅ DONE

Implement `ReduceOperation { kind: ReduceKind, axes: Vec<usize>, keepdims: bool }`. The JVP rule
is straightforward: `sum` is linear (its own pushforward), `max`/`min` are piecewise linear (route
the tangent through the arg-extremum index, with `select` over a recomputed argmax-ish mask). The
StableHLO lowering picks `stablehlo.reduce` with the appropriate scalar body (`add` for sum, `max`
for max, …).

With `reduce_sum` and `mul` in place, RMSNorm reduces to two value-level helpers (`mean(x*x)`, then
`x * rsqrt(var + eps) * (1 + scale)`). With `reduce_max`, `sub`, `exp`, `reduce_sum`, and `div`,
softmax becomes a six-line helper. Add these as value-level convenience functions in
`ryft-models::common::nn`, not as IR primitives, because their decomposition has good autodiff
properties already.

### Phase 4 — Shape ops & indexing — ✅ DONE

Land `ConcatenateOperation`, `SliceOperation` (static), and `PadOperation`. RoPE then drops out
(split into halves, rotate, concatenate). Then land `IotaOperation` so position vectors stop being
device transfers. Land `DynamicSliceOperation` and `DynamicUpdateSliceOperation` next, which
together support KV-cache reads/writes during decode and during training when we want to update
just the tail of an activation.

Land `GatherOperation` and `ScatterOperation`. `gather` is the embedding lookup for the input
table; `scatter-add` is its adjoint and is also reusable inside MoE dispatch later. We need an
explicit `scatter_kind: Replace | Add` knob — scatter-add must be commutative so that
`gather`'s VJP composes correctly.

Finally land `ArgMaxOperation`. It is not strictly required for training, but is needed for the
top-1 accuracy metric we will report and for greedy sampling in the inference test.

### Phase 5 — RNG & initialization — ✅ DONE

Pick a stateless PRNG (threefry-2x32 is the obvious choice, since it lowers cleanly to
`stablehlo.rng_bit_generator`). Add a `PrngKey` value type, a `split(key, n) -> Vec<PrngKey>`
helper, and `random_normal(key, shape, dtype) -> Array` / `random_uniform(key, shape, dtype, [lo,
hi])`. Truncated-normal initialization (Flax's default for Linen) can be built on top via
rejection sampling inside a `while_loop`, or as a `pjit`-time host helper for the first cut.

### Phase 6 — Activation checkpointing & optimizer — ⚠️ HALF DONE (remat ✅, optimizer ❌)

Add a `Checkpoint` transform that, when staged into the forward, marks the wrapped sub-program for
re-tracing on the backward pass. Concretely: during `linearize`, the inside of a checkpointed
region produces a tangent program that records *only* the symbolic inputs; during transpose, the
primal is re-evaluated. This is the standard JAX `remat` model.

Add an `optimizer` module: `AdamWState<P: Parameter>`, `adamw_step(params, grads, state, hyper) ->
(new_params, new_state)`, and `clip_by_global_norm(grads, max_norm) -> grads_clipped`. All of them
are value-level functions over `Parameterized` trees and stage into the same JIT graph as the
forward+backward.

### Phase 7 — Sharding & collectives — ✅ DONE

Add `AllReduceOperation`, `AllGatherOperation`, `ReduceScatterOperation`, and
`CollectivePermuteOperation` for use inside `shard_map` bodies. These are needed for tensor
parallelism (head-sharded GQA needs an `all_reduce` after the output projection) and for
MoE dispatch (`all_to_all` can be expressed as `reduce_scatter` + `all_gather` or as a dedicated
primitive). The `Mesh` and `Sharding` machinery already exists; this phase only adds the
IR primitives.

### Phase 8 — Gemma 4 model code & training harness — ❌ OPEN

With all primitives and the optimizer in place, the model is built bottom-up in
`crates/ryft-models/src/gemma_4/`:

1. `config.rs`: `Gemma4Config` with the four variant presets.
2. `params.rs`: `Gemma4Params` with `#[derive(Parameterized)]` covering embedder, per-layer
   PLE table, every block's RMSNorms / GQA projections / MLP / skip-scale, the final RMSNorm,
   and (for E2B/E4B) the multimodal projection.
3. `rope.rs`, `rmsnorm.rs`, `attention.rs`, `mlp.rs`, `block.rs`: pure traced functions over
   `Tracer<'_, D>` returning either activations or activations + a fresh KV-cache tile.
4. `forward.rs`: a `forward(params, tokens, positions) -> logits` traced function.
5. `loss.rs`: shifted cross-entropy with padding loss-mask; `loss(params, batch) -> scalar`.
6. `train.rs`: a `train_step(params, opt_state, batch) -> (params, opt_state, loss)` traced
   function combining `value_and_grad`, `clip_by_global_norm`, and `adamw_step`. JIT-compiled
   once via the existing XLA executable plumbing.
7. Integration tests against the JAX reference: load a published Gemma 4 E2B checkpoint, compare
   forward logits within `1e-3` (bf16 tolerance), then check that one optimizer step on a
   fixed batch reproduces JAX's parameter delta within `5e-4`.

### Phase 9 — Performance & validation — ❌ OPEN

1. Microbenchmarks per primitive (`cargo bench -p ryft-core`).
2. End-to-end throughput on a single GPU, then on a 2x2 mesh.
3. Loss-curve parity vs. the Flax reference on a small mixture for ≥1k steps.

### Revised remaining plan

With the primitives done, the remaining work re-scopes to five focused phases:

- **R1 — Optimizer module.** `AdamWState<P>` as a `Parameterized` tree, `adamw_step(params,
  gradients, state, hyper)`, `clip_by_global_norm(gradients, max_norm)`, and a cosine/linear
  learning-rate schedule helper. Pure functions over parameter trees; every ingredient
  (`reduce`, `sqrt`, `min`, elementwise ops, `map_parameters`) exists. Follow the shape of
  `gradient_descent_step` in [crates/ryft/examples/mlp.rs](crates/ryft/examples/mlp.rs).
- **R2 — Model crate + NN helpers.** Create `crates/ryft-models` with `common::nn` (RMSNorm,
  RoPE, GeGLU, `softmax`/`logsumexp`, one-hot/`take_along_axis` conveniences) written against
  the `ArrayOperations` bundle so they run eagerly on the reference backend, eagerly on XLA,
  and under every transform tracer unchanged.
- **R3 — Gemma 4 model + training step.** The §3 implementation: config, `Parameterized`
  parameter tree, blocks on the fused `dot_product_attention`, loss, `train_step` via
  `differentiate_at(..).with_captures(..).value_and_gradient(..)` + R1's optimizer, JIT-compiled
  through `ryft-xla`'s `jitted`. Wrap each block in `rematerialize(..)` with a
  `DotsWithNoBatchDimsSaveable`-style policy for long-context memory.
- **R4 — Validation.** Forward-logit parity against the DeepMind JAX reference on a published
  E2B checkpoint; one-optimizer-step parameter-delta parity; short-run loss-curve parity; then
  the §4.7 NVFP4 checks (the `scaled_dot` portable fallback and `__op$block_scaled_dot` CUDA
  path already exist, so this is validation work, not implementation).
- **R5 — Multimodal (optional).** A `ConvolutionOperation` (`stablehlo.convolution` lowering)
  for the SigLIP patch embedding is the only missing primitive; the rest of the vision tower is
  attention + MLP + norms, which all exist.

---

## 3. Target `ryft` Model Implementation

The code below is written in the **current** `ryft` idiom, modeled on
[crates/ryft/examples/mlp.rs](crates/ryft/examples/mlp.rs): model code is generic over a value
type `A: ArrayOperations` (the capability bundle in
[crates/ryft-core/src/arrays/operations/mod.rs](crates/ryft-core/src/arrays/operations/mod.rs)),
so the same functions run eagerly on the reference `Array` backend, eagerly op-by-op on
`ryft_xla::Array`, and under every transform tracer. Every IR primitive used below exists
today; what remains aspirational is the `ryft-models` crate itself — the `common::nn` helpers
(`rms_norm`, `apply_rope`, `geglu`, `logsumexp`) and the R1 optimizer module (`adamw_step`,
`clip_by_global_norm`). Scalar-constant plumbing (materializing `epsilon`, `1.0`, and RoPE
timescales through the dispatch domain's `Fill`/`Iota` constructors, or passing them as
nondifferentiated captures like `mean_scale` in the MLP example) is elided where it would
obscure the structure; exact signatures of the not-yet-written helpers are indicative.

The example targets the **Gemma 4 E2B** variant. Other variants drop in by changing the config.
A companion plan for Meta's Muse Glimmer 30B — which shares this plan's R1/R2 infrastructure
and delta-references its inventory and NVFP4 sections — lives in
[muse_glimmer_30b_plan.md](muse_glimmer_30b_plan.md).

### 3.1 Configuration

```rust
// crates/ryft-models/src/gemma_4/config.rs
use ryft_core::types::DataType;

/// Whether a given attention block applies local sliding-window attention or global attention.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AttentionKind {
    /// Local sliding-window attention with window `sliding_window_size`.
    Local,

    /// Global causal attention.
    Global,
}

/// Hyperparameters for one Gemma 4 variant.
#[derive(Clone, Debug)]
pub struct Gemma4Config {
    /// Total number of transformer blocks.
    pub layer_count: usize,

    /// Per-block attention pattern, of length `layer_count`. Repeats the canonical 4 local / 1
    /// global pattern for E2B and 5 local / 1 global for E4B.
    pub attention_pattern: Vec<AttentionKind>,

    /// Model (residual stream) dimension.
    pub embed_dim: usize,

    /// Number of query heads in GQA.
    pub query_head_count: usize,

    /// Number of key/value heads. Must divide `query_head_count`.
    pub kv_head_count: usize,

    /// Head dimension.
    pub head_dim: usize,

    /// Hidden dimension of the GeGLU MLP.
    pub mlp_hidden_dim: usize,

    /// Vocabulary size.
    pub vocab_size: usize,

    /// Per-layer-input embedding dimension (set to 0 to disable PLE).
    pub ple_dim: usize,

    /// Sliding window for `AttentionKind::Local`.
    pub sliding_window_size: usize,

    /// RoPE base frequency for `AttentionKind::Local` layers.
    pub local_rope_base_frequency: f32,

    /// RoPE base frequency for `AttentionKind::Global` layers.
    pub global_rope_base_frequency: f32,

    /// Fraction of head_dim that receives RoPE on global layers (1.0 on local layers).
    pub global_rope_proportion: f32,

    /// Final logit softcap value.
    pub final_logit_softcap: f32,

    /// Numerical epsilon for RMSNorm.
    pub rms_norm_epsilon: f32,

    /// Parameter dtype (typically bf16).
    pub parameter_dtype: DataType,

    /// Accumulation dtype (typically fp32 for RMSNorm, softmax, and optimizer state).
    pub accumulation_dtype: DataType,
}

impl Gemma4Config {
    /// Returns the Gemma 4 E2B preset.
    pub fn e2b() -> Self {
        let pattern: Vec<AttentionKind> = (0..35)
            .map(|index| if (index + 1) % 5 == 0 { AttentionKind::Global } else { AttentionKind::Local })
            .collect();
        Self {
            layer_count: 35,
            attention_pattern: pattern,
            embed_dim: 1536,
            query_head_count: 8,
            kv_head_count: 1,
            head_dim: 256,
            mlp_hidden_dim: 6144,
            vocab_size: 262_144,
            ple_dim: 256,
            sliding_window_size: 512,
            local_rope_base_frequency: 10_000.0,
            global_rope_base_frequency: 1_000_000.0,
            global_rope_proportion: 0.25,
            final_logit_softcap: 30.0,
            rms_norm_epsilon: 1e-6,
            parameter_dtype: DataType::BF16,
            accumulation_dtype: DataType::F32,
        }
    }
}
```

### 3.2 Parameter tree

```rust
// crates/ryft-models/src/gemma_4/params.rs
use ryft_core::parameters::Parameter;
use ryft_macros::Parameterized;

/// Optional RMSNorm scale (zero-centered: applied as `x * (1 + scale)`).
#[derive(Clone, Debug, Parameterized)]
pub struct RmsNormScale<P: Parameter> {
    /// Rank-1 learnable scale, shape `[features]`.
    pub scale: P,
}

#[derive(Clone, Debug, Parameterized)]
pub struct EmbedderParams<P: Parameter> {
    /// Input embedding table, shape `[vocab_size, embed_dim]`. Also used as the tied output
    /// projection.
    pub table: P,

    /// Per-layer-input table, shape `[vocab_size, layer_count, ple_dim]`. Empty (zero-size leaf)
    /// when PLE is disabled.
    pub per_layer_inputs: P,

    /// Projection from `embed_dim` to `layer_count * ple_dim`, shape `[embed_dim, layer_count *
    /// ple_dim]`.
    pub per_layer_projection: P,
}

#[derive(Clone, Debug, Parameterized)]
pub struct AttentionParams<P: Parameter> {
    /// Query projection, shape `[embed_dim, query_head_count * head_dim]` (kept rank-2 so the
    /// projection is a plain matmul `Dot`; the head axis is split out with one `Reshape`).
    pub q_proj: P,

    /// Key projection, shape `[embed_dim, kv_head_count * head_dim]`.
    pub k_proj: P,

    /// Value projection, shape `[embed_dim, kv_head_count * head_dim]`.
    pub v_proj: P,

    /// Output projection, shape `[query_head_count * head_dim, embed_dim]`.
    pub o_proj: P,

    /// QK-norm scale on queries.
    pub query_norm: RmsNormScale<P>,

    /// QK-norm scale on keys.
    pub key_norm: RmsNormScale<P>,
}

#[derive(Clone, Debug, Parameterized)]
pub struct MlpParams<P: Parameter> {
    /// GELU-branch (gate) projection, shape `[embed_dim, mlp_hidden_dim]`. (The reference
    /// implementation stacks gate and up into one `[2, embed_dim, mlp_hidden_dim]` einsum
    /// parameter; two separate matrices are numerically identical and keep the `Dot` calls
    /// rank-2.)
    pub gate_proj: P,

    /// Linear-branch (up) projection, shape `[embed_dim, mlp_hidden_dim]`.
    pub up_proj: P,

    /// Down projection, shape `[mlp_hidden_dim, embed_dim]`.
    pub down_proj: P,
}

#[derive(Clone, Debug, Parameterized)]
pub struct BlockParams<P: Parameter> {
    /// Pre-attention RMSNorm.
    pub pre_attention_norm: RmsNormScale<P>,

    /// Post-attention RMSNorm.
    pub post_attention_norm: RmsNormScale<P>,

    /// Pre-MLP RMSNorm.
    pub pre_mlp_norm: RmsNormScale<P>,

    /// Post-MLP RMSNorm.
    pub post_mlp_norm: RmsNormScale<P>,

    /// Learnable scalar applied to each residual branch.
    pub skip_scale: P,

    /// Attention parameters.
    pub attention: AttentionParams<P>,

    /// MLP parameters.
    pub mlp: MlpParams<P>,
}

#[derive(Clone, Debug, Parameterized)]
pub struct Gemma4Params<P: Parameter> {
    /// Token + PLE embedder.
    pub embedder: EmbedderParams<P>,

    /// One [`BlockParams`] per block, length `config.layer_count`.
    pub blocks: Vec<BlockParams<P>>,

    /// Final RMSNorm before the LM head.
    pub final_norm: RmsNormScale<P>,
}
```

### 3.3 Common neural-network helpers

```rust
// crates/ryft-models/src/common/nn.rs
//
// Helpers are generic over `A: ArrayOperations`, so they run eagerly on the reference `Array`
// backend, eagerly on `ryft_xla::Array`, and under every transform tracer without change. Two
// small conveniences are assumed and worth adding alongside them: `broadcast_like` (broadcast
// a reduced value back over the reduced axes) and `constant_like` (fill a scalar constant of a
// value's data type via the dispatch domain's `Fill`).

use ryft::*;

/// `x * rsqrt(mean(x^2, axis=-1) + epsilon) * (1 + scale)`, with the moment computed in fp32
/// regardless of the activation dtype (Gemma keeps norms in high precision), and the
/// zero-centered scale applied as `1 + scale`.
pub fn rms_norm<A: ArrayOperations>(x: &A, scale: Option<&A>, epsilon: f64) -> Result<A, ProgramError> {
    let input_type = x.r#type().into_owned();
    let last_axis = input_type.rank() - 1;
    let wide = x.convert_element_type(DataType::F32)?;
    let variance = (wide.clone() * wide.clone()).reduce(&[last_axis], ReductionKind::Mean);
    let inverse = (variance + constant_like(&variance, epsilon)?).rsqrt()?;
    let normalized = wide * broadcast_like(&inverse, &input_type)?;
    let normalized = match scale {
        Some(scale) => {
            let one_plus_scale = scale.clone() + constant_like(scale, 1.0)?;
            normalized * broadcast_like(&one_plus_scale, &input_type)?
        }
        None => normalized,
    };
    normalized.convert_element_type(input_type.data_type())
}

/// Exact GELU via `erf`: `0.5 * x * (1 + erf(x / sqrt(2)))`. The GeGLU MLP multiplies this
/// against the linear branch of the gating projection.
pub fn gelu<A: ArrayOperations>(x: &A) -> Result<A, ProgramError> {
    let scaled = x.clone() * constant_like(x, std::f64::consts::FRAC_1_SQRT_2)?;
    let one_plus_erf = scaled.erf()? + constant_like(x, 1.0)?;
    Ok(x.clone() * one_plus_erf * constant_like(x, 0.5)?)
}

/// Numerically stable `log(sum(exp(x)))` along the last axis. Only the LM-head cross-entropy
/// needs this composition — attention's softmax lives inside the fused
/// `dot_product_attention` primitive.
pub fn logsumexp<A: ArrayOperations>(x: &A) -> Result<A, ProgramError> {
    let last_axis = x.r#type().rank() - 1;
    let max = x.reduce(&[last_axis], ReductionKind::Max);
    let shifted = x.clone() - broadcast_like(&max, &x.r#type())?;
    Ok(shifted.exp()?.reduce(&[last_axis], ReductionKind::Sum).log()? + max)
}

/// RoPE rotating the leading `rotated_dim` head dimensions (split-halves convention) and
/// leaving the trailing `head_dim - rotated_dim` dimensions untouched. Gemma 4's partial RoPE
/// passes `rotated_dim = head_dim` on local layers and `head_dim / 4` on global layers.
pub fn apply_rope<A: ArrayOperations>(
    x: &A,          // [batch, seq, heads, head_dim]
    positions: &A,  // [batch, seq]
    rotated_dim: usize,
    base_frequency: f64,
) -> Result<A, ProgramError> {
    let head_dim = x.r#type().shape().dimension(-1).value().unwrap();
    let half = rotated_dim / 2;
    // timescale[i] = base_frequency ^ (2 i / head_dim) for i < half, then `Pad`ded with +inf
    // over the "nope" tail so cos = 1, sin = 0 leave the unrotated dimensions unchanged;
    // theta = positions / timescale, broadcast to [batch, seq, 1, head_dim / 2]. Built from
    // `Iota` + `Pow` + `Pad` + `Broadcast` + `Div`; spelled out in `rope_angles` (elided).
    let theta = rope_angles(positions, half, head_dim, base_frequency)?;
    let (cos, sin) = (theta.cos()?, theta.sin()?);
    let first = slice_last_axis(x, 0, head_dim / 2)?;
    let second = slice_last_axis(x, head_dim / 2, head_dim)?;
    let rotated_first = first.clone() * cos.clone() - second.clone() * sin.clone();
    let rotated_second = second * cos + first * sin;
    Concatenate::concatenate([&rotated_first, &rotated_second], 3)
}
```

### 3.4 Attention, MLP, and block

The attention core is now **one call to the fused `dot_product_attention` primitive**: it takes
queries `[batch, seq, heads, head_dim]` over keys/values `[batch, kv_seq, kv_heads, head_dim]`
with `kv_heads` dividing `heads` — Gemma's GQA layout natively, no group-axis reshapes — and
carries the causal mask and sliding window itself, so the hand-built logits → mask → softmax →
context pipeline from the original plan disappears entirely. On GPU it lowers to the cuDNN
FMHA kernels; `differentiable_dot_product_attention` wires the fused backward operation in as
a custom VJP.

```rust
// crates/ryft-models/src/gemma_4/attention.rs
use ryft::*;

use crate::common::nn::{apply_rope, rms_norm};
use crate::gemma_4::{AttentionKind, AttentionParams, Gemma4Config};

/// Attention forward over `[batch, seq, embed_dim]`. Returns activations of the same shape.
pub fn attention<A: ArrayOperations>(
    config: &Gemma4Config,
    layer_rope_base_frequency: f64,
    layer_rotated_dim: usize,
    sliding_window: Option<usize>,
    params: &AttentionParams<A>,
    x: &A,
    positions: &A,
) -> Result<A, ProgramError> {
    let batch = x.r#type().shape().dimension(0);
    let seq = x.r#type().shape().dimension(1);
    let (head_dim, heads, kv_heads) = (config.head_dim, config.query_head_count, config.kv_head_count);

    // Q/K/V projections: [B, T, D] @ [D, N * H] -> [B, T, N, H] (dot + reshape; no einsum needed).
    let project = |input: &A, weights: &A, head_count: usize| -> Result<A, ProgramError> {
        input
            .dot(weights, &DotDimensionNumbers::new(vec![2], vec![0], vec![], vec![]))
            .reshape(Shape::new(vec![batch, seq, Dimension::Static(head_count), Dimension::Static(head_dim)]))
    };
    let queries = project(x, &params.q_proj, heads)?;
    let keys = project(x, &params.k_proj, kv_heads)?;
    let values = project(x, &params.v_proj, kv_heads)?;

    // QK-norm (learned zero-centered scales), then partial RoPE.
    let queries = rms_norm(&queries, Some(&params.query_norm.scale), config.rms_norm_epsilon)?;
    let keys = rms_norm(&keys, Some(&params.key_norm.scale), config.rms_norm_epsilon)?;
    let queries = apply_rope(&queries, positions, layer_rotated_dim, layer_rope_base_frequency)?;
    let keys = apply_rope(&keys, positions, layer_rotated_dim, layer_rope_base_frequency)?;

    // Fused attention: native GQA ([B, T, N, H] over [B, S, K, H] with K dividing N), causal
    // masking, and the sliding window all live inside the one primitive.
    let context = queries.dot_product_attention(
        &keys,
        &values,
        /*scale=*/ (head_dim as f64).powf(-0.5),
        AttentionMask::Causal,
        sliding_window,
    )?;

    // Output projection: [B, T, N, H] -> [B, T, N * H] @ [N * H, D] -> [B, T, D].
    let context = context.reshape(Shape::new(vec![batch, seq, Dimension::Static(heads * head_dim)]))?;
    Ok(context.dot(&params.o_proj, &DotDimensionNumbers::new(vec![2], vec![0], vec![], vec![])))
}
```

```rust
// crates/ryft-models/src/gemma_4/mlp.rs
/// GeGLU MLP: gate/up projections, exact GELU on the gate branch, down projection.
pub fn mlp<A: ArrayOperations>(params: &MlpParams<A>, x: &A) -> Result<A, ProgramError> {
    let matmul = DotDimensionNumbers::new(vec![2], vec![0], vec![], vec![]);
    let gate = gelu(&x.dot(&params.gate_proj, &matmul))?;
    let linear = x.dot(&params.up_proj, &matmul);
    Ok((gate * linear).dot(&params.down_proj, &matmul))
}
```

```rust
// crates/ryft-models/src/gemma_4/block.rs
pub fn block<A: ArrayOperations>(
    config: &Gemma4Config,
    layer_index: usize,
    params: &BlockParams<A>,
    x: &A,
    positions: &A,
) -> Result<A, ProgramError> {
    let (base, rotated_dim, sliding_window) = match config.attention_pattern[layer_index] {
        AttentionKind::Local => {
            (config.local_rope_base_frequency, config.head_dim, Some(config.sliding_window_size))
        }
        AttentionKind::Global => (
            config.global_rope_base_frequency,
            (config.head_dim as f64 * config.global_rope_proportion) as usize,
            None,
        ),
    };
    let epsilon = config.rms_norm_epsilon;

    let attended = {
        let normed = rms_norm(x, Some(&params.pre_attention_norm.scale), epsilon)?;
        let out = attention(config, base, rotated_dim, sliding_window, &params.attention, &normed, positions)?;
        rms_norm(&out, Some(&params.post_attention_norm.scale), epsilon)?
    };
    let x = x.clone() + attended * broadcast_like(&params.skip_scale, &x.r#type())?;

    let mlp_out = {
        let normed = rms_norm(&x, Some(&params.pre_mlp_norm.scale), epsilon)?;
        rms_norm(&mlp(&params.mlp, &normed)?, Some(&params.post_mlp_norm.scale), epsilon)?
    };
    Ok(x.clone() + mlp_out * broadcast_like(&params.skip_scale, &x.r#type())?)
}
```

For long-context training, wrap each block in the rematerialization transform with a
dots-saveable policy, mirroring the JAX `jax.checkpoint(..., policy=dots_with_no_batch_dims_saveable)`
recipe:

```rust
let block_output = rematerialize(|(x, positions)| block(config, layer_index, params, &x, &positions))
    .with_policy(DotsWithNoBatchDimsSaveable)
    .apply((x, positions.clone()))?;
```

### 3.5 Forward, loss, and one training step

```rust
// crates/ryft-models/src/gemma_4/forward.rs
pub fn forward<A: ArrayOperations>(
    config: &Gemma4Config,
    params: &Gemma4Params<A>,
    tokens: &A,       // [batch, seq] of i32
    positions: &A,    // [batch, seq] of i32
) -> Result<A, ProgramError> {
    // Token embedding: `Gather` rows of the table, then scale by sqrt(embed_dim).
    let embeds = gather_rows(&params.embedder.table, tokens)?; // GatherOperation wrapper
    let mut hidden = embeds.clone() * constant_like(&embeds, (config.embed_dim as f64).sqrt())?;

    // Per-layer-input embeddings: [B, T, layer_count, ple_dim].
    let ple = gather_rows(&params.embedder.per_layer_inputs, tokens)?;

    for (layer_index, layer_params) in params.blocks.iter().enumerate() {
        hidden = block(config, layer_index, layer_params, &hidden, positions)?;
        // Per-layer-input injection: slice this layer's PLE stream ([B, T, 1, P], reshaped to
        // [B, T, P]), project to the residual width, add.
        let layer_ple = squeeze_axis(&slice_axis(&ple, 2, layer_index, layer_index + 1)?, 2)?;
        let projected = layer_ple.dot(
            &params.embedder.per_layer_projection,
            &DotDimensionNumbers::new(vec![2], vec![0], vec![], vec![]),
        );
        hidden = hidden + projected;
    }

    // Final RMSNorm + tied output projection + final logit softcap `cap * tanh(logits / cap)`.
    let normed = rms_norm(&hidden, Some(&params.final_norm.scale), config.rms_norm_epsilon)?;
    let logits = normed.dot(
        &params.embedder.table.transpose(&Permutation::new(vec![1, 0])?)?,
        &DotDimensionNumbers::new(vec![2], vec![0], vec![], vec![]),
    );
    let cap = constant_like(&logits, config.final_logit_softcap)?;
    Ok((logits / cap.clone())?.tanh()? * cap)
}
```

```rust
// crates/ryft-models/src/gemma_4/loss.rs
/// Shifted next-token cross-entropy with a padding loss-mask, computed as
/// `logsumexp(logits) - logits[target]` per token, then mask-averaged.
pub fn loss<A: ArrayOperations>(
    config: &Gemma4Config,
    params: &Gemma4Params<A>,
    tokens: &A,       // [batch, seq + 1] i32
    positions: &A,    // [batch, seq] i32
    loss_mask: &A,    // [batch, seq] in {0.0, 1.0}
) -> Result<A, ProgramError> {
    let seq = positions.r#type().shape().dimension(1).value().unwrap();
    let inputs = slice_axis(tokens, 1, 0, seq)?;
    let targets = slice_axis(tokens, 1, 1, seq + 1)?;
    let logits = forward(config, params, &inputs, positions)?;

    // Target logit via one-hot contraction (Iota + Compare + ConvertElementType + Mul + Reduce),
    // which keeps the composition simple; a GatherOperation take-along-axis is the alternative.
    let target_logit = take_along_last_axis(&logits, &targets)?;
    let per_token_loss = logsumexp(&logits)? - target_logit;
    let masked_loss = per_token_loss * loss_mask.clone();
    let total = masked_loss.reduce(&[0, 1], ReductionKind::Sum);
    Ok(total / loss_mask.reduce(&[0, 1], ReductionKind::Sum))
}
```

The training step follows the MLP example's pattern exactly: the model is the only active
(differentiated) argument, and the batch rides along as nondifferentiated runtime captures.
The bounds on `A` are the same ones `train` in
[crates/ryft/examples/mlp.rs](crates/ryft/examples/mlp.rs) states —
`A: ArrayOperations` with `A::ExecutionDomain: ReverseModeDifferentiate` and
`LinearizationTracer<A::ExecutionDomain>: ArrayOperations`.

```rust
// crates/ryft-models/src/gemma_4/train.rs
pub fn train_step<A: ArrayOperations>(
    config: &Gemma4Config,
    model: Gemma4Params<A>,
    optimizer_state: AdamWState<A>,   // R1 optimizer module
    hyper: &AdamWHyper,
    batch: &Batch<A>,
) -> Result<(Gemma4Params<A>, AdamWState<A>, A), ProgramError>
where
    A::ExecutionDomain: ReverseModeDifferentiate<Operation: From<OneOperation<ArrayType>>> + Zero<A>,
    LinearizationTracer<A::ExecutionDomain>: ArrayOperations,
{
    // 1. Loss and gradients: the model is the active argument; the batch is captured.
    let (loss_value, gradients) = differentiate_at(model.clone())
        .with_captures((batch.tokens.clone(), batch.positions.clone(), batch.loss_mask.clone()))
        .value_and_gradient(|model, (tokens, positions, loss_mask)| {
            loss(config, &model, &tokens, &positions, &loss_mask)
        })?;

    // 2. Clip-by-global-norm, then AdamW (both R1; pure functions over `Parameterized` trees,
    //    built with `map_parameters` / `into_parameters` like `gradient_descent_step` in the
    //    MLP example).
    let gradients = clip_by_global_norm(gradients, hyper.max_global_norm)?;
    let (model, optimizer_state) = adamw_step(model, gradients, optimizer_state, hyper)?;

    Ok((model, optimizer_state, loss_value))
}
```

### 3.6 End-to-end driver

Because `ryft-xla`'s `Array` executes eagerly with a shared per-lineage executable cache, the
training loop below already runs on-device without explicit JIT; wrapping the step with
`ryft_xla::jitted` (or backend-neutral `ryft::jit`) fuses it into one compiled program with
retained-cache dispatch.

```rust
// crates/ryft-models/examples/gemma_4_train.rs
use ryft::pjrt::{ClientOptions, load_cpu_plugin};
use ryft::xla::{Array, FromPjrt};
use ryft::{Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType};
use ryft_models::gemma_4::{AdamWHyper, AdamWState, Gemma4Config, Gemma4Params, train_step};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Gemma4Config::e2b();

    // PJRT client + mesh, following the mlp.rs xla_backend setup (CUDA plugin with CPU fallback).
    let (plugin, client_options) = load_xla_plugin()?;
    let client = plugin.client(client_options)?;
    let mesh = single_axis_mesh(&client)?; // LogicalMesh::new(vec![MeshAxis::new("data", n, MeshAxisType::Auto)?])?

    // Initialize parameters with the device RNG (`Random::split_key` + `Random::normal` per leaf).
    let mut model: Gemma4Params<Array> = initialize_gemma_4(&client, &mesh, &config, /*seed=*/ 0)?;
    let mut optimizer_state = AdamWState::zeros_like(&model)?;
    let hyper = AdamWHyper {
        learning_rate: 1e-4, weight_decay: 0.1, b1: 0.9, b2: 0.95, eps: 1e-8, max_global_norm: 1.0,
    };

    for batch in data_loader(&client, &mesh)? {
        let (new_model, new_state, loss) = train_step(&config, model, optimizer_state, &hyper, &batch)?;
        model = new_model;
        optimizer_state = new_state;
        println!("loss = {:?}", read_scalar(&loss)?);
    }

    Ok(())
}
```

---

## 4. NVFP4 and FP8 Training on Blackwell

Blackwell (GB200, B100, B200) adds first-class hardware for two new low-precision regimes that
are directly relevant to Gemma 4 training:

- **FP8** in both `E4M3` (forward) and `E5M2` (backward) variants. Hopper already supports FP8;
  Blackwell roughly doubles per-SM throughput and adds tighter integration with the
  [2nd-gen Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html).
  The accumulator stays in fp32 by default.
- **NVFP4**, NVIDIA's microscaled FP4 format described in the
  [OCP Microscaling Formats v1.0 spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
  and refined in
  [Pretraining LLMs with NVFP4 (arXiv:2509.25149)](https://arxiv.org/pdf/2509.25149) and
  [Four Over Six (arXiv:2512.02010)](https://arxiv.org/pdf/2512.02010). Each `F4E2M1FN` element
  is paired with a per-block scale stored as `F8E8M0FNU` (UE8M0 in the MXFP4 case) or
  `F8E4M3FN` (UE4M3 in the NVFP4 case), with block size 16 (NVFP4) or 32 (MXFP4). A coarse
  per-tensor scale is applied on top to avoid overflow.

The `tcgen05.mma.blockscaled` MMA family on Blackwell — documented in the
[CUTLASS Blackwell SM100 functionality guide](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)
and walked through in
[Colfax's sub-byte GEMM tutorial](https://research.colfax-intl.com/cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus/) —
consumes the scale operands inline with the data, which means the matmul primitive itself
changes shape on Blackwell: the GEMM op grows two extra "scale" operands. Everything else in
the model stays in `bf16`.

### 4.1 What already exists in `ryft`

> **Status: the core of this section is implemented.** Since this section was written, the
> exact primitive set it proposed has landed: `ScaledDotOperation`
> ([crates/ryft-core/src/operations/math/dot.rs](crates/ryft-core/src/operations/math/dot.rs))
> with interpretation, partial-evaluation, forward-differentiation (bilinear with scales held
> fixed), and batching rules; the `BlockQuantize` composition
> ([crates/ryft-core/src/operations/math/block_quantize.rs](crates/ryft-core/src/operations/math/block_quantize.rs))
> covering both the NVFP4 recipe (`f4e2m1fn` elements + `f8e4m3fn` scales, `max_abs / 6.0`)
> and the OCP MX recipe (`f8e8m0fnu` power-of-two scales with the spec-prescribed clamping);
> and the two-path XLA lowering (`lower_scaled_dot_to_mlir` in
> [crates/ryft-xla/src/experimental/lowering.rs](crates/ryft-xla/src/experimental/lowering.rs)):
> on a CUDA target with qualifying formats it emits the `__op$block_scaled_dot` custom call
> (XLA's cuDNN block-scaled-dot target — the same kernel `jax.nn.scaled_matmul` reaches), and
> everywhere else it falls back to a portable dequantize → upcast → `dot_general` expansion —
> which is exactly the "A100 fallback" §4.7 asked for. The fused attention path similarly
> lowers to cuDNN FMHA custom calls (`__cudnn$fmha…`). What remains open is listed in §4.2.

- `DataType::F4E2M1FN` (NVFP4 data), `DataType::F8E4M3FN` (FP8 forward), `DataType::F8E5M2`
  (FP8 backward), `DataType::F8E8M0FNU` (UE8M0 microscale), `DataType::F8E4M3` (UE4M3
  microscale) are all in the enum at
  [crates/ryft-core/src/arrays/types/data.rs:694](crates/ryft-core/src/arrays/types/data.rs:694)
  (plus the MX FP6 types `F6E2M3FN`/`F6E3M2FN` and sub-byte integers). These mirror the
  StableHLO type set — see the
  [StableHLO specification](https://openxla.org/stablehlo/spec), the
  [F8E4M3/F8E3M4 RFC](https://github.com/openxla/stablehlo/blob/main/rfcs/20240808-f8E4M3_f8E3M4.md),
  and the [Speccing StableHLO quantization](https://groups.google.com/a/openxla.org/g/openxla-discuss/c/iwE9is49SS4)
  thread for the upstream rationale. The promotion lattice intentionally excludes them, which is
  correct — they are conversion-only.
- `ArrayType` accepts any `DataType`, and with `ScaledDotOperation` + `BlockQuantize` the IR
  now also has the operations that produce and consume FP4/FP8 tensors.

### 4.2 Operation inventory (updated)

| Capability | Used by | State | Notes |
|---|---|---|---|
| block quantization `(values, scales)` from bf16/fp32 | producing FP8/NVFP4 weights & activations | ✅ | `BlockQuantize::block_quantize(block_size, element_type, scale_type)` — a pure composition of existing primitives (as this table originally recommended for `reduce_max_abs`, and better than the dedicated op it proposed: the recipe inherits its transform rules from its ingredients). NVFP4 recipe (`f8e4m3fn` scales, `max_abs / element_max`) and OCP MX recipe (`f8e8m0fnu` power-of-two scales with spec-prescribed clamping) both implemented |
| block-scaled dot | every quantized matmul | ✅ | `ScaledDotOperation` / `ScaledDot::scaled_dot(lhs_scales, rhs, rhs_scales, block_size, accumulation_type)` (+ `scaled_dot_with_global_scale`); rank-2 or batched rank-3 operands; forward-mode rule treats it as bilinear with the scales held fixed |
| dequantize | debugging, portable execution | ✅ | implemented inside the portable lowering fallback (broadcast-expand scales → multiply → convert → `dot_general`); no standalone op, none needed |
| per-block `reduce_max_abs` | amax inside quantization | ✅ | composite (`Abs` + `Reshape` + `Reduce`), inside `block_quantize` |
| FP4/FP8 ↔ bf16 casts | scaled-GEMM boundaries | ✅ | `ConvertElementTypeOperation` handles the narrow types |
| Stochastic rounding for backward casts | NVFP4 backward quality | ❌ | still missing — `ConvertElementType` has no rounding-mode parameter; needs a `RoundingMode` knob (RNG-driven, via `RngBitGenerator`) |
| Random Hadamard Transform (RHT) | TE-parity NVFP4 outlier smoothing | ❌ | still missing; composable from existing ops (block-diagonal `dot` with a fixed Hadamard matrix + sign flips) or a fused kernel later |
| Delayed-scaling amax history buffer | FP8 per-tensor scale tracker | ❌ | still missing; a pure-function pattern over `reduce`/`slice`/`dynamic_update_slice` — belongs in the R1 optimizer/scaling module, no new IR primitive |
| Fused row+column dual quantization | TE-parity training throughput | ❌ | forward and backward consume a tensor along different axes; TE produces both quantized copies in one fused kernel. `ryft` can start with two `block_quantize` calls and fuse later |

The original three-primitive proposal (`QuantizeScaled`, `DequantizeScaled`,
`ScaledDotGeneral`) was realized as **one** primitive (`ScaledDotOperation`) plus **one
composition** (`BlockQuantize`) — a strictly smaller IR surface than planned. The remaining
rows are training-recipe refinements, not blockers: bf16-master-weight NVFP4 training runs
with what exists today.

### 4.3 Lowering strategy

> **Status: implemented.** `lower_scaled_dot_to_mlir` in
> [crates/ryft-xla/src/experimental/lowering.rs](crates/ryft-xla/src/experimental/lowering.rs)
> implements the recommendation below: when the target platform is CUDA and both operand
> format pairs qualify for hardware block scaling, it emits a `stablehlo.custom_call` to
> **`__op$block_scaled_dot`** (XLA's cuDNN block-scaled-dot target — the kernel
> `jax.nn.scaled_matmul` reaches), handling operand reordering, physical padding, and
> dynamic-dimension restoration; on every other platform (or non-qualifying formats) it
> expands to the portable dequantize → upcast → `dot_general` fallback. The background below
> is retained for the design rationale, including the FP8 `gemm-rewriter` alternative that
> remains available if per-tensor-FP8 (rather than block-scaled) recipes are ever wanted.

There is **no dedicated `stablehlo.scaled_dot_general` op today** — neither in the current spec
nor as a finalized in-progress proposal. The two viable paths on Blackwell, both grounded in
shipped or RFC'd OpenXLA behavior, are described below. They differ in a way that is essential
to understand: only the FP8 path relies on XLA-side pattern matching; the microscaling path
does not, and that asymmetry shapes what `ryft-xla` has to emit.

#### 4.3.1 FP8: emit plain HLO and let `gemm-rewriter` pattern-match

Per [OpenXLA RFC #22 — FP8 in XLA](https://github.com/openxla/xla/discussions/22), the
user-visible IR for an FP8 GEMM is plain StableHLO: each operand is upcast to a wider type via
`stablehlo.convert`, multiplied by a scale, fed through `stablehlo.dot_general`, then the
output is scaled, its absolute-maximum is taken, and the result is cast back to FP8. XLA's
[`gemm-rewriter` pass](https://github.com/openxla/xla/blob/main/xla/service/gpu/transforms/gemm_rewriter.cc)
pattern-matches that exact 6-step subgraph and rewrites it into a single fused HLO custom call
whose target string is **`__cublas$lt$matmul$f8`**, with the canonical signature
`(A, B, a_scale, b_scale, d_scale) -> (D, d_amax)` (see also the
[Flax FP8 user guide](https://flax-linen.readthedocs.io/en/latest/guides/quantization/fp8_basics.html)
for the same shape from the frontend perspective, and the original implementation
[`tensorflow/tensorflow#58720`](https://github.com/tensorflow/tensorflow/pull/58720)).

Concretely, what the pattern matcher is looking for is roughly:

```text
%a_wide  = stablehlo.convert(%a_fp8)     // FP8 -> FP16/BF16
%b_wide  = stablehlo.convert(%b_fp8)
%a_scaled = stablehlo.multiply(%a_wide, %a_scale)   // scalar broadcast
%b_scaled = stablehlo.multiply(%b_wide, %b_scale)
%d_wide   = stablehlo.dot_general(%a_scaled, %b_scaled, ...)
%d_amax   = stablehlo.reduce(... max, abs ...)(%d_wide)
%d_scaled = stablehlo.multiply(%d_wide, %d_scale)
%d_fp8    = stablehlo.convert(%d_scaled)            // wider -> FP8
```

When that template appears in the right order with broadcast-compatible scales, it folds into
one `__cublas$lt$matmul$f8` call. The match is structural — XLA does not need any FP8-specific
attributes on the ops; the ops are just normal `convert`/`multiply`/`dot_general` whose operand
types happen to be `f8E4M3FN` or `f8E5M2`. If the pattern is broken — e.g., an extra elementwise
op between the multiply and the dot, or a transpose with non-canonical layout — the fusion
fails, the FP8 GEMM falls back to a wider-type dot, and performance collapses. The
[`jax-ml/jax#22313`](https://github.com/jax-ml/jax/issues/22313) and
[`jax-ml/jax#24051`](https://github.com/jax-ml/jax/issues/24051) issues are good cautionary
tales about how brittle this is in practice and should be required reading before relying on
the path. **For `ryft`, this means our `ScaledDotGeneralOperation` lowers to exactly that 6-op
template, in that exact order, so `gemm-rewriter` can find it.**

Blackwell-specific code paths (e.g., `tcgen05.mma`) are **not** XLA's concern in this path —
once XLA has emitted the `__cublas$lt$matmul$f8` custom call, dispatch to the right SM kernel
is cuBLASLt's job. On Blackwell SM100+, cuBLASLt internally selects
`tcgen05.mma.blockscaled`-family kernels described in the
[CUTLASS Blackwell SM100 docs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)
and the [Colfax sub-byte GEMM tutorial](https://research.colfax-intl.com/cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus/);
on Hopper it picks `wgmma`; on Ampere it falls back to the wider-type fallback. The XLA-emitted
IR is identical.

#### 4.3.2 Microscaling (NVFP4/MXFP4): mirror what `jax.nn.scaled_matmul` already does

There is no `gemm-rewriter`-style pattern match for microscaling. Two reasons: (i) per-block
scale broadcast does not factor cleanly through `stablehlo.multiply` (the scale lives at block
granularity, not per-element or per-tensor), so there's no recognizable template; and (ii) the
JAX side already settled on a custom-call-based design, which means the frontend emits the
scaled matmul as a single op and there's nothing to "pattern match" into.

Concretely, modern JAX exposes two ops in `jax.nn` that we should mirror:

- [`jax.nn.scaled_matmul(lhs, rhs, lhs_scales, rhs_scales, preferred_element_type=jnp.float32)`](https://docs.jax.dev/en/latest/_autosummary/jax.nn.scaled_matmul.html) —
  takes pre-quantized `lhs`/`rhs` (shape `(B, M, K)` / `(B, N, K)`) and explicit per-block
  `lhs_scales`/`rhs_scales` (shape `(B, M, K_a)` / `(B, N, K_b)` where `K_a = K / block_size`).
  Element types: `jnp.float8_e4m3fn`/`e5m2` with `jnp.float8_e8m0fnu` scales and block size 32
  (MXFP8), or `jnp.float4_e2m1fn` with `jnp.float8_e4m3fn` scales and block size 16 (NVFP4).
- [`jax.nn.scaled_dot_general(lhs, rhs, dimension_numbers, preferred_element_type, configs, implementation)`](https://docs.jax.dev/en/latest/_autosummary/jax.nn.scaled_dot_general.html) —
  takes BF16/FP32 inputs, quantizes internally, dispatches to `scaled_matmul`, and provides the
  backward pass automatically. Configs come from `jax.nn.get_scaled_dot_general_config('nvfp4',
  global_scale)` or `('mxfp8')`.

Both lower through
[`jax._src.cudnn.scaled_matmul_stablehlo`](https://github.com/jax-ml/jax/blob/main/jax/_src/cudnn/scaled_matmul_stablehlo.py)
(the `scaled_matmul_wrapper` helper) into a `stablehlo.custom_call` whose target is the
**cuDNN** scaled-matmul kernel — *not* cuBLASLt. cuDNN on Blackwell internally routes that to
`tcgen05.mma.blockscaled` (see the [CUTLASS Blackwell SM100 functionality guide](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)
for the kernel family). The same custom-call falls back to a software path (or fails) on
pre-Blackwell hardware unless the kernel ships a fallback. This means:

1. There is a *de facto* standard target string already in use — whatever
   `scaled_matmul_wrapper` emits. `ryft-xla` should emit the same one so that we inherit cuDNN's
   ongoing kernel improvements for free.
2. The earlier [OpenXLA RFC #18085](https://github.com/openxla/xla/discussions/18085) framing
   ("use a custom call until an HLO op exists") is already the deployed approach in JAX 25.x.
   The "future HLO op" mentioned in the RFC has not landed and there's no committed timeline,
   but the JAX-side abstraction (`jax.nn.scaled_matmul` / `scaled_dot_general`) is stable and
   widely used.

So the answer to "how does XLA know to use Blackwell features in the MX path?" is: it doesn't
discover them — the custom call we emit names the cuDNN backend kernel directly, and that
kernel is the thing that targets `tcgen05.mma.blockscaled` on Blackwell. There is no
pattern-match step on the XLA side. The dispatch decision happens *inside cuDNN*, based on the
device the call is running on.

For `ryft-xla`'s lowering, the practical recipe is therefore:

- **Phase 4.5a (recommended):** emit a `stablehlo.custom_call` to cuDNN's scaled-matmul kernel
  with the same target string and operand layout as `jax.nn.scaled_matmul`. This is the
  highest-leverage path: it inherits cuDNN's tile-size, layout, and SM-dispatch handling, and
  golden-tests against `jax.nn.scaled_matmul` give us a continuous oracle.
- **Phase 4.5b (optional, post-MVP):** for cases where cuDNN doesn't cover the needed tile
  shape, fall back to a Pallas:MGPU-style kernel (see §4.4) or to TE's CUTLASS-based kernels
  via a separate custom-call target.

For `quantize_scaled` and `dequantize_scaled`, the lowering is purely composite —
`stablehlo.reduce` + `stablehlo.divide` + `stablehlo.convert` + `stablehlo.broadcast_in_dim` —
so no custom call is needed in either path. cuDNN's fused quantize-then-GEMM kernels can be
opted into later as a peephole optimization on top of the FP8 pattern-match path; for MX, the
fusion would need to live inside the custom-call kernel itself.

#### 4.3.3 Why `ryft-core` wants a first-class scaled-dot operation

*(This argument won: the op landed as `ScaledDotOperation`.)* An IR-level scaled-dot primitive
in `ryft-core` is justified independently of either lowering: it preserves the scaled-matmul
shape through `ryft`'s autodiff and sharding transforms (both of which need to see the matmul
as one node, not as an expanded sequence that's harder to differentiate and harder to shard).
The lowering rule in `ryft-xla` is then the single place where the decision "qualifying CUDA
target → `__op$block_scaled_dot` custom call; anything else → portable dequantized expansion"
lives, and that rule is small enough that swapping it for a future upstream HLO op is a
one-day change.

#### 4.3.4 Reference: how efficient NVFP4 model implementations in JAX are built today

The JAX ecosystem has converged on a three-layer stack for NVFP4 training, and `ryft`'s design
should track it deliberately because each layer dictates what we plug into where.

**Layer A — JAX-native primitives (`jax.nn`, JAX 25.x+).** Two ops, both lowering through
[`jax._src.cudnn.scaled_matmul_stablehlo`](https://github.com/jax-ml/jax/blob/main/jax/_src/cudnn/scaled_matmul_stablehlo.py)
into a `stablehlo.custom_call` whose target is cuDNN's scaled-matmul kernel:

- [`jax.nn.scaled_matmul`](https://docs.jax.dev/en/latest/_autosummary/jax.nn.scaled_matmul.html)
  — low-level. Takes pre-quantized inputs and explicit per-block scales; supports MXFP8
  (block 32, `f8E8M0FNU` scales) and NVFP4 (block 16, `f8E4M3FN` scales). Compute dtype is
  fixed at fp32; user-customizable precision is not yet exposed.
- [`jax.nn.scaled_dot_general`](https://docs.jax.dev/en/latest/_autosummary/jax.nn.scaled_dot_general.html)
  — high-level wrapper that accepts BF16/FP32 inputs, quantizes internally, dispatches to
  `scaled_matmul`, and **provides the backward pass automatically**. Configurations come from
  `jax.nn.get_scaled_dot_general_config('nvfp4', global_scale)` or `('mxfp8')`.

This is the layer `ryft-xla` should target directly: emit the same cuDNN custom call, with the
same operand layout. Doing so gives us cuDNN's ongoing kernel and tile-size improvements for
free, and it lets us golden-test against `jax.nn.scaled_matmul` on identical inputs.

**Layer B — NVIDIA Transformer Engine for JAX
([`NVIDIA/TransformerEngine`](https://github.com/NVIDIA/TransformerEngine),
`transformer_engine.jax`).** This is what production training jobs actually use. TE provides
drop-in Flax modules and an `fp8_autocast` context manager. A representative usage looks like
this:

```python
import transformer_engine.jax as te
from transformer_engine.common.recipe import NVFP4BlockScaling

# Drop-in Flax replacements: te.flax.LayerNormDenseGeneral, te.flax.LayerNormMLP,
# te.flax.MultiHeadAttention, te.flax.TransformerLayer, te.flax.DenseGeneral, te.flax.LayerNorm.

recipe = NVFP4BlockScaling(...)
with te.fp8_autocast(enabled=True, fp8_recipe=recipe, mesh_resource=...):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
```

Per [TE's NVFP4 feature page](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html),
the NVFP4 recipe does substantially more than `jax.nn.scaled_matmul`:

- **Stochastic rounding** on the backward cast to keep the expected value unbiased.
- **Random Hadamard Transform (RHT)** on the column-wise quantization of inputs and gradients,
  to smooth outliers before quantization.
- **Rowwise + columnwise quantized copies** of every tensor (forward and backward read along
  different axes; both copies are produced in a single fused kernel).
- **Fused row-cast / RHT / transpose / col-cast kernels** (recent perf work).
- **Delayed-scaling amax history** for the FP8 paths (when using `Format.HYBRID` /
  `DelayedScaling` instead of `NVFP4BlockScaling`).

Crucially, TE does **not** dispatch through `jax.nn.scaled_matmul`. It registers its own XLA
custom call targets backed by hand-written CUTLASS/CUDA kernels, exposed through
`transformer_engine.jax.cpp_extensions` and primitive-bound via `jax.core.Primitive` plus
`jax.interpreters.xla` lowering rules. This is the only way to ship features like RHT and
fused stochastic-rounding casts at the moment — they aren't yet in cuDNN's public surface. The
[`karpathy/nanochat` FP8/NVFP4 discussion](https://github.com/karpathy/nanochat/discussions/382)
documents the practical 20–50% speedups people are seeing from this layer.

**Layer C — research and bespoke kernels.**

- **Pallas:MGPU.** JAX's GPU kernel DSL targets Blackwell directly; see the
  [Blackwell matmul tutorial](https://docs.jax.dev/en/latest/pallas/gpu/blackwell_matmul.html).
  One Pallas thread of execution is one CUDA warpgroup, and the kernel issues `tcgen05.mma`
  instructions itself. Used when neither `jax.nn.scaled_matmul` nor TE provides the right
  shape, or for prototyping novel scaling schemes.
- [`mit-han-lab/fouroversix`](https://github.com/mit-han-lab/fouroversix) — research code
  accompanying *Four Over Six* ([arXiv:2512.02010](https://arxiv.org/pdf/2512.02010)) and
  *Adaptive Block-Scaled Data Types*; refines NVFP4 quantization with adaptive block scaling.
- [`vuiseng9/fp4-training`](https://github.com/vuiseng9/fp4-training) — minimal PyTorch +
  cuBLASLt + Microxcaling reference, useful as a numerical oracle.

**What this means for `ryft`'s plan.**

1. **MVP target: Layer A.** `ScaledDotGeneralOperation` lowers to the same `stablehlo.custom_call`
   that `jax.nn.scaled_matmul` emits. This gets us correctness and competitive throughput
   immediately, with `jax.nn.scaled_matmul` itself as the golden oracle in §4.7.
2. **Production target: Layer B parity.** For the features that drive end-to-end training quality
   on NVFP4 — stochastic rounding, RHT, fused row+column quantization — we eventually need TE-
   equivalent kernels. Two ways to land them: (a) wrap TE's existing custom calls so `ryft-xla`
   can emit them with the same target strings (lowest cost; ties us to TE's release cadence), or
   (b) build our own Pallas:MGPU kernels for those fused casts. Decision can be deferred to
   Phase 8.5 once we have parity at Layer A.
3. **Research escape hatch: Layer C.** Pallas:MGPU is the contingency for any kernel we need
   that doesn't exist upstream. The doors here are pre-existing; we only need to plumb a Pallas
   call site into the `ryft-xla` lowering when the time comes.

#### 4.3.5 Kernel source strategy: where the GPU code we ship comes from

A natural follow-up question is whether `ryft` should write its own NVFP4 GEMM kernels — and,
if so, in what language. Two NVlabs Rust GPU projects are now relevant and at very different
abstraction levels: [`NVlabs/cuda-oxide`](https://github.com/NVlabs/cuda-oxide) (SIMT, v0.1.0
May 2026) and [`NVlabs/cutile-rs`](https://github.com/NVlabs/cutile-rs) (tile-based,
pre-alpha). Both are alpha-stage; neither lets us write a SOTA NVFP4 kernel today; but their
gaps are different shapes, and the *better long-term bet for `ryft` is cuTile Rust*.

##### 4.3.5.1 The two NVlabs Rust GPU projects, side by side

| Aspect | [cuda-oxide](https://github.com/NVlabs/cuda-oxide) | [cutile-rs](https://github.com/NVlabs/cutile-rs) |
|---|---|---|
| Abstraction | SIMT (thread-level, CUDA C++-like) | Tile-based (Triton/Pallas-style) |
| Pipeline | Rust → MIR → LLVM IR → PTX (custom `rustc` codegen) | Rust → MLIR → PTX/CUBIN (with caching) |
| MMA API | `wgmma_mma_*` (Hopper), `tcgen05_mma_f16` (Blackwell) | `ct.mma(x, y, acc)` — compiler picks tensor-core instr. |
| FP4 / FP8 / microscale dtypes in kernels | Not listed in the [MMA-accelerators page](https://nvlabs.github.io/cuda-oxide/advanced/matrix-multiply-accelerators.html) | [`f4e2m1fn`, `f8e4m3fn`, `f8e5m2`, `f8e8m0fnu` all listed](https://docs.nvidia.com/cuda/cutile-python/data.html) (via the parent cuTile data model) |
| Block-scaled MMA exposed in user API | No (would need inline PTX) | Not yet — `ct.mma` takes no scale operands; tracked as [cutile-python#47](https://github.com/NVIDIA/cutile-python/issues/47), open since Dec 2025 |
| Underlying hardware MMA support | CTAGen05 atoms in LLVM NVPTX backend | CUTLASS/CuTe MMA atoms — already block-scaled-capable |
| Architectures targeted | Hopper + Blackwell | Blackwell-only currently |
| Maturity | v0.1.0 alpha | Pre-alpha, "expect breaking API changes" |
| Production usage | None visible | [Hugging Face's "Grout" Qwen 3 inference engine](https://github.com/NVlabs/cutile-rs) |
| Best fit | Hand-tuned warpgroup kernels, novel primitives | High-perf GEMM, structured matmul kernels |

##### 4.3.5.2 Why cuTile Rust is the more interesting bet for `ryft`

Four reasons cuTile Rust is the more natural long-term target for `ryft`'s kernel layer than
cuda-oxide:

1. **The abstraction matches the problem.** Tile-based DSLs are *the* idiom for high-perf
   GEMM — that is what Triton, Pallas:MGPU, and CuTe-Python were designed for. cuTile's
   `ct.mma(x, y, acc)` *"automatically invokes Tensor Cores"* (per the
   [NVIDIA matmul blog](https://developer.nvidia.com/blog/how-to-write-high-performance-matrix-multiply-in-nvidia-cuda-tile/)),
   so the user does not hand-schedule warpgroups or manage tensor memory directly. That is the
   correct level of abstraction for a kernel crate inside a compiler stack like `ryft`.
2. **Blackwell-focused from day one.** The same blog states: *"cuTile is the next-generation
   GPU programming framework… While it only supports optimization for the Blackwell (compute
   capabilities 10.x and 12.x) architecture, support for more architectures will be provided
   in upcoming releases."* The whole stack is tuned for the hardware NVFP4 actually runs on.
3. **The dtypes are already in.** Per the
   [cuTile data-types reference](https://docs.nvidia.com/cuda/cutile-python/data.html),
   cuTile already supports `f8e4m3fn`, `f8e5m2`, `f8e8m0fnu`, and `f4e2m1fn` — the exact set
   we need for FP8, NVFP4, and MXFP4. cuda-oxide does not yet have FP4 at all.
4. **Production validation.** [Hugging Face's "Grout"](https://github.com/NVlabs/cutile-rs)
   (Qwen 3 inference engine) is built on cuTile Rust — at least one real ML workload is
   already using it, which is more than cuda-oxide can claim today.

##### 4.3.5.3 The honest gap in cuTile Rust today

The blocker is the same shape as cuda-oxide's, but smaller: `cuda.tile.mma(x, y, /, acc)`
[currently takes no scale operands](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.mma.html).
The community-filed
[`NVIDIA/cutile-python#47`](https://github.com/NVIDIA/cutile-python/issues/47) — *"Clarification
on FP8 Micro-block Scaling and FP4 Support Timeline"* — asks exactly that question; as of May
2026 it sits at status "triaged" with no public answer. The *types* are in, the *target
hardware* (Blackwell `tcgen05.mma.blockscaled`) is supported in the underlying CUTLASS/CuTe
layer (see the [Colfax hardware-supported block-scaling tutorial](https://research.colfax-intl.com/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/)),
the *DSL surface* for a `mma_block_scaled(x, y, lhs_scales, rhs_scales, acc)` (or equivalent)
is what is missing. This is a smaller and more localized gap than cuda-oxide's — the
plumbing in the layer below already exists; only the operator needs to be surfaced — but the
timeline is still publicly uncommitted. The cuTile Rust DSL also typically tracks the Python
parent, so the Rust-side gap is at least as wide.

The other practical caveat is maturity. cutile-rs is explicitly *pre-alpha* (cuda-oxide v0.1.0
is alpha), so swapping a release of either project under a production training run is going
to require a `Cargo.lock`-pinned version, a kernel-level regression suite, and probably some
local patches.

##### 4.3.5.4 The realistic ladder for `ryft`

The kernel-source choice should be a swappable lowering target behind `ScaledDotGeneralOperation` —
which is exactly what §4.3.3 already argues for. That gives us a smooth ramp from "ship
today" to "all-Rust eventually" without a model-code rewrite at any point.

| Option | What `ryft-xla` does | Maturity | Gets SOTA NVFP4 today? |
|---|---|---|---|
| A. Mirror `jax.nn.scaled_matmul` | Emit the cuDNN scaled-matmul custom call (same target as JAX) | Production | Yes — inherits cuDNN |
| B. Wrap Transformer Engine | Emit TE's registered XLA custom-call targets | Production | Yes; adds RHT + stochastic rounding |
| C. CUTLASS via FFI | C++ shim around [CUTLASS Blackwell narrow-precision GEMM examples](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html), built with `nvcc`, linked via [`bindgen`](https://crates.io/crates/bindgen) / [`cc`](https://crates.io/crates/cc) | Production | Yes; fully customizable |
| D. cuda-oxide with current intrinsics | Rust kernels using FP16 `tcgen05_mma_f16` | Alpha | No — falls back to FP16 |
| E. cuda-oxide + inline PTX | Rust + hand-written `tcgen05.mma.blockscaled` PTX | Alpha + experimental | Maybe; defeats safe-Rust value |
| F. cutile-rs with current API | Rust tile-based kernels using FP16/BF16 `ct.mma` | Pre-alpha | No — `ct.mma` doesn't take scales yet |
| G. cutile-rs future | Rust tile kernels with a future `mma_block_scaled`-style API | Weeks to months out (uncommitted) | Yes — tile DSL + Blackwell-tuned, smallest gap of the all-Rust options |
| H. cuda-oxide future | Rust SIMT kernels with a future `tcgen05_mma_blockscaled_*` intrinsic | 6–18 months out | Yes; lower abstraction than G |

**Recommended sequence.**

- **Phase 4.5 (MVP) — Option A.** Lower `ScaledDotGeneralOperation` to the same cuDNN custom
  call that `jax.nn.scaled_matmul` emits. Cheapest path to "we train Gemma 4 on NVFP4."
- **Phase 4.6 (production parity) — Option B or C.** For RHT, stochastic rounding, and fused
  row+column quantization — features cuDNN does not expose — either wrap TE's custom-call
  targets (Option B; ties us to TE's release cadence and licence) or land a thin
  `ryft-cuda-kernels` crate that wraps CUTLASS via FFI (Option C; more work, full control).
  CUTLASS via FFI is the well-trodden path — it is what TE itself does internally.
- **Phase 4.7 (Rust kernel-layer staging) — Option F.** Once cutile-rs stabilizes its
  pre-alpha API, port the non-scaled GEMMs in `ryft-cuda-kernels` (e.g. FP16/BF16 fallbacks,
  ancillary kernels) to cuTile Rust. This is the rehearsal step that lets us shake out the
  build, FFI, and custom-call integration in `ryft-xla` *before* we depend on it for the
  critical NVFP4 path.
- **Phase 4.8+ (long-term, all-Rust) — Option G.** When cuTile Rust exposes block-scaled MMA
  in its tile DSL, port the NVFP4 GEMMs from CUTLASS-via-FFI to cuTile Rust one kernel at a
  time. The XLA custom-call target the lowering points at can stay stable through this
  migration. If cuTile Rust stalls and cuda-oxide ships block-scaled MMA first, Option H is
  the fallback; the architecture in this section is identical for either.

**Why this design is robust to either NVlabs project stalling.** In the worst case where both
cuTile Rust and cuda-oxide get deprioritized, we still have a production-quality kernel crate
(Option C); the model code in §3 does not change. In the best case where cuTile Rust ships
block-scaled MMA in its DSL in months, the migration from C++ to Rust kernels is local to one
crate and one set of custom-call implementations. The investment in Options A/B/C is never
wasted, and the long-term toolchain-unification win remains on the table on either NVlabs
roadmap.

### 4.4 Training recipe

The numerically correct Gemma 4 + FP8/NVFP4 recipe, taken from the
[Transformer Engine FP8 primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html),
the [Flax FP8 guide](https://flax-linen.readthedocs.io/en/latest/guides/quantization/fp8_basics.html),
and the [NVFP4 pretraining paper (arXiv:2509.25149)](https://arxiv.org/pdf/2509.25149), and
adapted to Gemma's specific norm structure:

| Tensor | Forward dtype | Backward dtype | Storage dtype |
|---|---|---|---|
| Q/K/V projection weights | E4M3 (FP8) or NVFP4 | E5M2 (FP8) or NVFP4 | bf16 master |
| Q/K/V projection inputs (residual) | E4M3 or NVFP4 | E5M2 or NVFP4 | bf16 activation |
| Attention logits matmul (`Q @ K^T`) | scaled GEMM | scaled GEMM | — |
| Attention output (`attn @ V`) | scaled GEMM | scaled GEMM | — |
| Attention output projection | E4M3 / NVFP4 | E5M2 / NVFP4 | bf16 master |
| MLP gating + down projection weights | E4M3 / NVFP4 | E5M2 / NVFP4 | bf16 master |
| RMSNorm scales | bf16 | bf16 | bf16 |
| Embedding table (tied) | bf16 (or E4M3 with stochastic rounding) | bf16 | bf16 master |
| Softmax | bf16 / fp32 | bf16 / fp32 | — |
| RMSNorm `mean(x^2)` + `rsqrt` | fp32 accumulation | fp32 | — |
| Final logit softcap (`tanh`) | bf16 | bf16 | — |
| Optimizer state (Adam m, v) | fp32 | — | fp32 |

Rules of thumb:

- **Norms and softmax stay high precision.** Their dynamic range exceeds what FP8 can carry, and
  they are not the dominant cost. Cast back to bf16 (or fp32 for reductions) at their
  boundaries.
- **Embedding `gather` stays bf16.** Token-frequency skew makes amax estimation unstable for
  embedding tables; quantize only if a follow-up benchmark shows ≥5% memory win that you need.
- **Final unembedding can be FP8.** Tied weights mean the same table is read by `gather` and
  contracted by the unembedding `dot_general`; quantize for the contraction, not for the gather.
- **NVFP4 vs FP8.** Use NVFP4 for the MLP gating, up, and down projections (largest GEMMs,
  most tolerant of quantization noise). Use FP8 E4M3 for the attention projections, which carry
  smaller activations and need slightly tighter precision. Both run on `tcgen05.mma`.

### 4.5 API surface (as implemented)

The real API landed as two value-level capabilities that compose exactly as the aspirational
sketch hoped, minus the config struct (quantization regime is expressed directly by the
`(element_type, scale_type, block_size)` triple):

```rust
// crates/ryft-core/src/operations/math/block_quantize.rs (implemented)
pub trait BlockQuantize: Sized {
    /// Quantizes `self` into `(elements, scales)` per block of `block_size` trailing-dimension
    /// values. `f8e4m3fn` scales select the NVFP4 recipe; `f8e8m0fnu` scales select the OCP MX
    /// power-of-two recipe (with the spec-prescribed clamping).
    fn block_quantize(
        &self,
        block_size: usize,
        element_type: DataType,
        scale_type: DataType,
    ) -> Result<(Self, Self), ProgramError>;
}

// crates/ryft-core/src/operations/math/dot.rs (implemented)
pub trait ScaledDot: Sized {
    /// Block-scaled matrix product of `self` `[b?, m, k]` (scaled by `lhs_scales`
    /// `[b?, m, k / block_size]`) and `rhs` `[b?, n, k]` (scaled by `rhs_scales`), dequantizing
    /// both operands to `accumulation_type` and returning the `[b?, m, n]` product at that type.
    fn scaled_dot(
        &self,
        lhs_scales: &Self,
        rhs: &Self,
        rhs_scales: &Self,
        block_size: usize,
        accumulation_type: DataType,
    ) -> Result<Self, ProgramError>;
    // plus `scaled_dot_with_global_scale(..)` carrying NVFP4's coarse per-tensor scale.
}
```

A quantized matmul in the model is then a two-call composition (this is what the model crate's
`Policy`-driven `maybe_scaled_dot` helper wraps):

```rust
// NVFP4 MLP projection: quantize both sides on the fly, contract at fp32, downcast to bf16.
let (x_q, x_scales) = x.block_quantize(16, DataType::F4E2M1FN, DataType::F8E4M3FN)?;
let (w_q, w_scales) = weights.block_quantize(16, DataType::F4E2M1FN, DataType::F8E4M3FN)?;
let product = x_q
    .scaled_dot(&x_scales, &w_q, &w_scales, /*block_size=*/ 16, DataType::F32)?
    .convert_element_type(DataType::BF16)?;
```

Note the operand convention: `scaled_dot` contracts the **last** axis of both sides
(`[m, k] × [n, k]`), so weight matrices feed it in `[n, k]` layout rather than the `[k, n]`
layout plain `dot` uses — the model crate's helper owns that transpose. A `Policy` on the model
config still earns its keep by carrying the per-role `(element_type, scale_type, block_size)`
choices (attention projections FP8, MLP NVFP4, §4.4) so call sites stay clean.

### 4.6 Implementation phases (delta on top of §2)

- **Phase 1.5 — Conversion and quantization — ✅ mostly done.** FP4/FP8 conversions work
  through `ConvertElementTypeOperation`, and quantization landed as the `BlockQuantize`
  composition (round-trip and amax tests live beside it). Remaining from this phase:
  rounding-mode metadata on the conversion op (stochastic rounding for backward casts).
- **Phase 4.5 — Scaled GEMM lowering — ✅ done (custom-call path).** Landed as
  `ScaledDotOperation` with the `__op$block_scaled_dot` cuDNN custom call on CUDA and the
  portable dequantize fallback elsewhere, plus interpretation/PE/JVP/batching rules. The
  alternative FP8 `gemm-rewriter` pattern-match path
  ([RFC #22](https://github.com/openxla/xla/discussions/22),
  [`gemm_rewriter.cc`](https://github.com/openxla/xla/blob/main/xla/service/gpu/transforms/gemm_rewriter.cc),
  [tensorflow#58720](https://github.com/tensorflow/tensorflow/pull/58720)) was not taken and
  remains available if per-tensor FP8 (`__cublas$lt$matmul$f8`) recipes are wanted later.
  Still open from the original description: an explicit transpose rule producing
  `E5M2`-flavored backward GEMMs (today the backward of a quantized matmul re-quantizes
  through the same forward recipe).
- **Phase 6.5 — Scale tracker — ❌ open.** A small `optimizer::scaling` module: delayed-scaling
  amax history buffer (`history: [steps, num_tensors]`), `update_scale_from_amax(history,
  amax)` per training step, and a hook wiring per-tensor scales into matmul call sites via the
  model's `Policy`. Belongs with the R1 optimizer work.
- **Phase 8.5 — End-to-end FP8/NVFP4 Gemma 4 — ❌ open (validation, not implementation).**
  Re-run `train_step` with the `Policy` selecting `block_quantize` + `scaled_dot` at the
  §4.4-designated call sites. Validate against the bf16 baseline on a fixed batch: per-step
  parameter delta within `1e-3` (FP8) / `5e-3` (NVFP4); 1k-step loss curves statistically
  indistinguishable.

### 4.7 Verification plan

1. **Per-op numerics.** For `BlockQuantize` and `ScaledDot` (both implemented), build
   reference NumPy/PyTorch implementations and check ULP-level agreement on randomized inputs.
   The primary oracle for `ScaledDot` is
   [`jax.nn.scaled_matmul`](https://docs.jax.dev/en/latest/_autosummary/jax.nn.scaled_matmul.html)
   itself — we emit the same cuDNN custom call, so the outputs should be bit-identical on
   identical inputs. Secondary oracles: `torch.ops.aten._scaled_mm`, Transformer Engine's
   [`fp8_gemm` reference](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html),
   and the [`vuiseng9/fp4-training`](https://github.com/vuiseng9/fp4-training) reference
   cuBLASLt + MX implementation for the NVFP4 path specifically. For loss-curve parity, the
   target reference is the Transformer Engine NVFP4 recipe configured via
   [`NVFP4BlockScaling`](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html)
   on a fixed batch.
2. **One-step gradient parity.** With a frozen batch and frozen RNG, compare bf16 and
   FP8/NVFP4 gradient norms layer by layer. Acceptable spread: <1% rms divergence per layer.
3. **Short fine-tune.** Run Gemma 4 E2B for 1k steps on the standard fine-tuning mixture under
   bf16 and under FP8/NVFP4. The loss curves should agree within statistical noise (<0.01
   nats end-to-end).
4. **Throughput sanity check.** On a single B200, expect ≥1.6× throughput for FP8 and ≥2.5× for
   NVFP4 vs bf16 on the MLP GEMMs. If the numbers are far below this, the lowering is
   misconfigured (most likely missing the cuBLASLt epilogue fusion).
5. **Portable fallback — ✅ implemented.** The `ScaledDotOperation` lowering already falls back
   to a transparent dequantize → upcast → `dot_general` expansion on any non-CUDA target (or
   non-qualifying formats), which keeps model code portable for development on CPU, Hopper, or
   Ampere. Remaining nicety: a one-time log line when the fallback is taken.

### 4.8 Open questions specific to Blackwell

- **NVFP4 vs MXFP4.** NVIDIA's NVFP4 (16-element blocks, UE4M3 scale, plus per-tensor scale) and
  the [OCP MXFP4 standard](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
  (32-element blocks, UE8M0 scale) coexist on Blackwell. `BlockQuantize` already implements
  both recipes (selected by the scale type); default to NVFP4 because the reference recipes
  target it and the loss-curve evidence in
  [arXiv:2509.25149](https://arxiv.org/pdf/2509.25149) and
  [arXiv:2512.02010](https://arxiv.org/pdf/2512.02010) is stronger.
- **Block alignment.** NVFP4's 16-element blocks must be aligned to the contracting axis. For
  attention heads where `head_dim = 256` this is automatic; for the MLP's 6144-wide axis it is
  also fine. But the unembedding's vocab axis of 262 144 needs explicit padding logic if it is
  ever the contracting axis (it is the reduction axis in the loss gradient `softmax * label -
  one_hot`-style path), which it is not in our recipe — but document it.
- **Stochastic rounding for backward.** NVFP4 backward gradients benefit from stochastic
  rounding to avoid systematic bias accumulation. This requires the Phase 5 RNG primitive
  threaded into `ConvertElementType`; treat it as an optional `RoundingMode` field rather than a
  separate op.
- **Scaling regime: delayed vs JIT.** [Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
  defaults to delayed scaling (amax history → scale used next step). JIT scaling (compute amax
  inline, use this step's scale) is simpler but adds a synchronization point. Start with
  delayed; revisit if loss-curve parity becomes hard to achieve.
- **`tcgen05.mma` tile sizes.** Blackwell's MMA tile is `128×128` for FP8 and `256×128` for
  NVFP4 with strict layout requirements. The cuBLASLt path (and the FP8 pattern-match path via
  XLA's `gemm-rewriter`) hides this from us today. If a dedicated upstream scaled-matmul HLO op
  ever lands — RFC #18085 mentions it only as eventual future work, with no committed name or
  shape — its lowering will need to pad inputs to those tile boundaries. Plan to handle that in
  the lowering, not in the IR primitive.
- **Mixed FP8/NVFP4 in the same GEMM.** Blackwell supports mixed E4M3/NVFP4 operand pairs, but
  the numerics are not well studied yet for transformer training. Hold this until §4.7
  verification on uniform formats is solid.

---

## 5. Risks & Open Questions

- **The critical path moved from primitives to the model layer.** Everything blocking is now
  concentrated in the missing optimizer module (R1), the model crate (R2/R3), and validation
  (R4). The primitive-level risks the original list carried are resolved.
- **Mixed-precision policy boundary.** §3's `rms_norm` does its bf16↔fp32 casts explicitly via
  `ConvertElementType`, and `dot_with_accumulation_type` covers matmul accumulators — but there
  is no `Policy` wrapper yet that applies a consistent regime across the model. Still open;
  now purely model-crate work.
- **KV-cache sharing & PLE during training.** The reference implementation conditionally reuses K
  and V from a donor layer (`kv_shared_cache`). During training (no decode), this collapses to
  "skip the K/V projection on the consuming layer and read from the donor's pre-RoPE K/V". The
  parameter tree captures this only implicitly via shared `Vec<BlockParams>` indices; a tagged
  `BlockKind` enum may be clearer.
- **Fused attention coverage for Gemma-specific details.** `DotProductAttentionOperation`
  covers causal + sliding-window + GQA + bias, which is everything the dense text path needs.
  Two details to verify during R3: soft-capping *inside* attention is not needed for Gemma 4
  (only the final-logit softcap survives, outside attention — fine), and QK-norm happens
  before the fused op (fine, it is outside the kernel in the reference too). If a future
  variant needs per-attention logit transforms, the fallback is the unfused composition, which
  all exists.
- **MoE variant (26B-A4B).** The dense path is fully covered. For MoE, `top_k`, `scatter`, and
  `all_to_all` all exist now; what remains is model-level routing code and (likely) a ragged /
  grouped GEMM story for expert efficiency — flag as Phase 8b once the dense path is stable.
- **Vision encoder needs convolution.** The only missing IR primitive in the whole plan:
  `ConvolutionOperation` + `stablehlo.convolution` lowering for the SigLIP patch embedding
  (R5). Text-only training is unaffected.
- **`einsum` ergonomics.** §3 now avoids einsum entirely (rank-2 `dot` + `reshape`, fused
  attention). A string-spec frontend remains a nice-to-have, not a blocker.
- **Determinism of init under sharding.** The device RNG is stateless (ThreeFry/Philox) as
  required; sharding-aware key derivation per shard still needs a convention in the model
  crate. Training-time dropout, if ever enabled, rides the fused attention op's own
  `with_dropout(p, seed)`.
- **`erf` vs `tanh`-approx GELU.** Both primitives exist now; expose the choice as a config
  knob in the model crate. Default to exact (`erf`) GELU.
- **Batching a compiled function.** `CompiledXlaFunction::batch` is stubbed — deliberately, not
  as an oversight (see the design note at [crates/ryft-xla/src/jit.rs:1031](crates/ryft-xla/src/jit.rs:1031)).
  Derived compiled functions are retained through a structural transform cache, and on this
  branch the batch extent is a first-class runtime `DimensionVariable` (batched sealed regions
  carry the `[extent, inputs...] ↦ [extent, outputs...]` boundary). The open design decision is
  the cache-key split: the extent's static type/identity contract (plus axis name, mapped
  sharding, normalized input axes, and output policy) belongs in the structural transform key,
  while the extent *value* should be a runtime operand — so `f.batch(32)` and `f.batch(64)`
  share one retained artifact. A naive extent-specialized implementation (mirroring the ~70-line
  `.jvp()` shape around the core `batch` transform, one compilation per axis size) is a few
  days of work; the intended runtime-extent design is weeks, but overlaps the dynamic-shapes
  work already in flight. Independent gap either way: `shard_map`/`linear_shard_map` have no
  batching rules yet. Not on the training critical path (data batching is just the leading
  axis inside the traced loss); it matters for vmap-of-jit workflows (ensembles, multi-seed,
  per-example gradients).

---

## 6. References

A consolidated list of every external source cited in this document, grouped by topic.

### 6.1 Gemma model documentation and reference implementations

- [Gemma 4 — Google DeepMind](https://deepmind.google/models/gemma/gemma-4/) — product page and
  variant overview.
- [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4) — authoritative
  hyperparameters and intended-use documentation.
- [`google-deepmind/gemma`](https://github.com/google-deepmind/gemma) — the canonical JAX /
  Flax-Linen reference implementation, including the Gemma 4 modules under
  `gemma/gm/nn/gemma4/`.
- [`google/gemma_pytorch`](https://github.com/google/gemma_pytorch) — PyTorch reference
  implementation.
- [HuggingFace `transformers/models/gemma3`](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma3) —
  HuggingFace's Gemma 3 module; the architectural ancestor of Gemma 4.
- [Flax NNX Gemma tutorial](https://flax.readthedocs.io/en/stable/examples/gemma.html) — Flax
  NNX-flavored walkthrough.
- [Gemma 3 technical report (arXiv:2503.19786)](https://arxiv.org/abs/2503.19786) — the report
  that documents the architectural choices Gemma 4 inherits.
- [Gemma explained: what's new in Gemma 3](https://developers.googleblog.com/en/gemma-explained-whats-new-in-gemma-3/) —
  Google Developers blog post on the local–global pattern, QK-norm, and per-layer-input
  embeddings.

### 6.2 OpenXLA / StableHLO

- [StableHLO Specification](https://openxla.org/stablehlo/spec) — op-by-op normative spec
  consulted for every primitive lowering target in §1.
- [StableHLO RFC: `f8E4M3` and `f8E3M4`](https://github.com/openxla/stablehlo/blob/main/rfcs/20240808-f8E4M3_f8E3M4.md) —
  rationale for the FP8 dtype set that `ryft`'s `DataType` mirrors.
- [Speccing StableHLO quantization (OpenXLA discuss)](https://groups.google.com/a/openxla.org/g/openxla-discuss/c/iwE9is49SS4) —
  background on the deferred `QuantizedType` direction.
- [OpenXLA RFC #22 — FP8 in XLA](https://github.com/openxla/xla/discussions/22) — the
  pattern-match approach for FP8 GEMMs; the `gemm-rewriter` template described in §4.3.1.
- [OpenXLA RFC #18085 — Microscaling (MX) types in XLA](https://github.com/openxla/xla/discussions/18085) —
  the custom-call approach for NVFP4/MXFP4 scaled-matmul described in §4.3.2.
- [`xla/service/gpu/transforms/gemm_rewriter.cc`](https://github.com/openxla/xla/blob/main/xla/service/gpu/transforms/gemm_rewriter.cc) —
  the actual pass that emits `__cublas$lt$matmul$f8`.
- [`tensorflow/tensorflow#58720` — FP8 GEMMs in XLA](https://github.com/tensorflow/tensorflow/pull/58720) —
  the PR that landed the FP8 path.
- [`jax-ml/jax#22313` — FP8 fusion not fully working](https://github.com/jax-ml/jax/issues/22313)
  and [`#24051` — regression in JAX FP8 matmul fusion](https://github.com/jax-ml/jax/issues/24051) —
  cautionary tales on how brittle the FP8 pattern match is.

### 6.3 NVIDIA Blackwell and microscaling formats

- [OCP Microscaling Formats v1.0 specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) —
  the standard that defines MXFP4/MXFP6/MXFP8 element and scale encodings.
- [Pretraining LLMs with NVFP4 (arXiv:2509.25149)](https://arxiv.org/pdf/2509.25149) — NVIDIA's
  NVFP4 pretraining methodology and loss-curve evidence.
- [Four Over Six: more accurate NVFP4 quantization with adaptive block scaling
  (arXiv:2512.02010)](https://arxiv.org/pdf/2512.02010) — refinement of the NVFP4 scaling
  approach; companion code at [`mit-han-lab/fouroversix`](https://github.com/mit-han-lab/fouroversix).
- [Transformer Engine FP8 primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html) —
  reference recipe for FP8/NVFP4 training, including delayed-scaling amax history.
- [Flax FP8 user guide](https://flax-linen.readthedocs.io/en/latest/guides/quantization/fp8_basics.html) —
  frontend-side FP8 usage guide that exercises the XLA pattern-match path.
- [CUTLASS Blackwell SM100 functionality](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html) —
  documents `tcgen05.mma.blockscaled` and tile-size requirements.
- [Colfax: Sub-byte GEMM on Blackwell](https://research.colfax-intl.com/cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus/) —
  walkthrough of NVFP4 GEMM kernel construction.
- [cuBLAS 13.x documentation](https://docs.nvidia.com/cuda/cublas/) — cuBLASLt matmul-descriptor
  attributes for block-scaled GEMM.
- [`vuiseng9/fp4-training`](https://github.com/vuiseng9/fp4-training) — reference PyTorch +
  cuBLASLt + Microxcaling implementation useful as a numerical oracle for NVFP4 unit tests.
- [NVIDIA Blackwell: the impact of NVFP4 for LLM inference (Edge AI and Vision Alliance)](https://www.edge-ai-vision.com/2025/10/nvidia-blackwell-the-impact-of-nvfp4-for-llm-inference/) —
  high-level overview of NVFP4 deployment numbers.

### 6.4 JAX-native scaled matmul and the modern ecosystem

- [`jax.nn.scaled_matmul`](https://docs.jax.dev/en/latest/_autosummary/jax.nn.scaled_matmul.html) —
  low-level scaled-matmul op. Takes pre-quantized inputs and explicit per-block scales; supports
  MXFP8 and NVFP4 on Blackwell via cuDNN.
- [`jax.nn.scaled_dot_general`](https://docs.jax.dev/en/latest/_autosummary/jax.nn.scaled_dot_general.html) —
  high-level wrapper. Accepts BF16/FP32 inputs, quantizes internally, dispatches to
  `scaled_matmul`, and provides the backward pass. Configs via `jax.nn.get_scaled_dot_general_config`.
- [`jax._src.cudnn.scaled_matmul_stablehlo`](https://github.com/jax-ml/jax/blob/main/jax/_src/cudnn/scaled_matmul_stablehlo.py) —
  the actual cuDNN custom-call lowering used by both ops.
- [NVIDIA/TransformerEngine](https://github.com/NVIDIA/TransformerEngine) and its
  [JAX integration guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/te_jax_integration.html) —
  the production library used for FP8/NVFP4 training in JAX. Provides `transformer_engine.jax`
  Flax modules and the `fp8_autocast` context manager.
- [Transformer Engine NVFP4 feature page](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html) —
  details the NVFP4 recipe: stochastic rounding, Random Hadamard Transform, rowwise+columnwise
  quantization, fused row-cast/RHT/transpose/col-cast kernels.
- [Pallas:MGPU Blackwell matmul tutorial](https://docs.jax.dev/en/latest/pallas/gpu/blackwell_matmul.html) —
  walkthrough for hand-written `tcgen05.mma` kernels in JAX's Pallas DSL.
- [`mit-han-lab/fouroversix`](https://github.com/mit-han-lab/fouroversix) — research
  implementation for adaptive block-scaled NVFP4 quantization.
- [`karpathy/nanochat#382` — FP8/NVFP4 training with Transformer Engine](https://github.com/karpathy/nanochat/discussions/382) —
  community write-up documenting practical 20–50% speedups from TE.
- [JAX 25.10 release notes (NVIDIA)](https://docs.nvidia.com/deeplearning/frameworks/jax-release-notes/rel-25-10.html) —
  release notes covering the `scaled_matmul`/`scaled_dot_general` additions.

### 6.5 Rust GPU toolchain (kernel-source strategy)

**cuda-oxide (SIMT Rust → PTX):**

- [`NVlabs/cuda-oxide`](https://github.com/NVlabs/cuda-oxide) — NVIDIA's experimental Rust → PTX
  compiler backend; v0.1.0 released May 2026.
- [The cuda-oxide Book](https://nvlabs.github.io/cuda-oxide/index.html) — primary documentation.
- [cuda-oxide: Matrix-Multiply Accelerators](https://nvlabs.github.io/cuda-oxide/advanced/matrix-multiply-accelerators.html) —
  the current WGMMA / tcgen05 intrinsic surface (block-scaled MMA not yet exposed as of v0.1.0).
- [The Rust + GPU ecosystem (cuda-oxide appendix)](https://nvlabs.github.io/cuda-oxide/appendix/ecosystem.html) —
  how cuda-oxide relates to `cudarc` and other Rust GPU projects.
- [NVIDIA AI: Releases cuda-oxide (MarkTechPost, May 2026)](https://www.marktechpost.com/2026/05/09/nvidia-ai-just-released-cuda-oxide-an-experimental-rust-to-cuda-compiler-backend-that-compiles-simt-gpu-kernels-directly-to-ptx/)
  and [Phoronix announcement](https://www.phoronix.com/news/NVIDIA-CUDA-Oxide-0.1) — release
  context.

**cuTile Rust (tile-based Rust → MLIR → PTX/CUBIN):**

- [`NVlabs/cutile-rs`](https://github.com/NVlabs/cutile-rs) — NVlabs' tile-based Rust DSL for
  GPU kernels. Pre-alpha as of May 2026.
- [cuTile Rust documentation](https://nvlabs.github.io/cutile-rs/) — the primary reference,
  including persistent-GEMM benchmark numbers (96.4% of cuBLAS reported).
- [`crates/cutile`](https://crates.io/crates/cutile) — the crates.io listing.
- [cuTile Python documentation](https://docs.nvidia.com/cuda/cutile-python/) — the sibling
  Python DSL; shares the underlying tile IR and dtype model with cutile-rs.
- [cuTile data-types reference](https://docs.nvidia.com/cuda/cutile-python/data.html) —
  enumerates the supported floating-point types including `f4e2m1fn`, `f8e4m3fn`, `f8e5m2`,
  and `f8e8m0fnu`.
- [`cuda.tile.mma` reference](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.mma.html) —
  current `mma(x, y, /, acc)` signature; no scale operands yet.
- [`NVIDIA/cutile-python#47` — FP8 micro-block scaling & FP4 timeline](https://github.com/NVIDIA/cutile-python/issues/47) —
  open issue tracking when block-scaled MMA is surfaced in the DSL.
- [How to write high-performance matrix multiply in NVIDIA cuTile (NVIDIA blog)](https://developer.nvidia.com/blog/how-to-write-high-performance-matrix-multiply-in-nvidia-cuda-tile/) —
  reference matmul tutorial; explains that cuTile currently targets only Blackwell sm_100/sm_120.
- [CUDA Tile (NVIDIA Developer)](https://developer.nvidia.com/cuda/tile) — overview page
  positioning the cuTile programming model.
- [From CUDA to Rust: Scaling GPU Performance with Tile-Based Programming (UC Berkeley Sky
  seminar)](https://sky.cs.berkeley.edu/events/sky-seminar-melih-elibol-stephen-jones-nvidia-from-cuda-to-rust-scaling-gpu-performance-with-tile-based-programming/) —
  talk by NVIDIA's Melih Elibol & Stephen Jones positioning the cuTile Rust effort.

**Shared / supporting:**

- [`cudarc`](https://docs.rs/cudarc) — host-side Rust bindings to the CUDA driver API; the
  fallback host-launch path for cuda-oxide- or cuTile-Rust-produced PTX/CUBIN when not using
  their first-party host crates.
- [Inline PTX Assembly in CUDA (NVIDIA docs)](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html) —
  reference for hand-writing `tcgen05.mma.blockscaled` mnemonics if Option E is ever pursued.
- [CUTLASS Tutorial: Hardware-supported Block-scaling with Blackwell (Colfax)](https://research.colfax-intl.com/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/) —
  documents the CUTLASS/CuTe block-scaled MMA layer that cuTile builds on top of.

### 6.6 In-repo references

- [`crates/ryft-core/src/operations/`](crates/ryft-core/src/operations/) — the complete
  primitive set (`math/`, `manipulation/`, `compare.rs`, `logical/`, `random.rs`,
  `collectives.rs`, `control_flow/`, `sort.rs`, `attention.rs`, `sharding.rs`, `dimensions/`).
- [`crates/ryft-core/src/arrays/operations/mod.rs`](crates/ryft-core/src/arrays/operations/mod.rs) —
  the `ArrayOperations` capability bundle and the closed `ArrayOperation` /
  `ArrayIrOperation` / `DimensionOperation` families.
- [`crates/ryft-core/src/operations/attention.rs`](crates/ryft-core/src/operations/attention.rs) —
  the fused `DotProductAttentionOperation` (+ backward) with GQA, causal masking, sliding
  windows, dropout, bias, and sequence lengths.
- [`crates/ryft-core/src/operations/math/dot.rs`](crates/ryft-core/src/operations/math/dot.rs) —
  `DotOperation` and the block-scaled `ScaledDotOperation` (§4).
- [`crates/ryft-core/src/operations/math/block_quantize.rs`](crates/ryft-core/src/operations/math/block_quantize.rs) —
  the `BlockQuantize` NVFP4/MX quantization recipes (§4).
- [`crates/ryft-core/src/arrays/types/data.rs:694`](crates/ryft-core/src/arrays/types/data.rs:694) —
  the `DataType` enum entries for `F4E2M1FN`, `F8E4M3FN`, `F8E5M2`, `F8E8M0FNU`, and friends.
- [`crates/ryft-core/src/differentiation/`](crates/ryft-core/src/differentiation/) — the
  `differentiate_at` builder (jvp/linearize/vjp/value_and_gradient/jacobians/hessian).
- [`crates/ryft-core/src/tracing_v2/rematerialization.rs`](crates/ryft-core/src/tracing_v2/rematerialization.rs) —
  `rematerialize` and the JAX-parity checkpointing policy family.
- [`crates/ryft-core/src/compilation/`](crates/ryft-core/src/compilation/) — backend-neutral
  `jit`/`stage_function` with structural and disk caching.
- [`crates/ryft-xla/src/experimental/lowering.rs`](crates/ryft-xla/src/experimental/lowering.rs) —
  the StableHLO lowering rules, including `lower_scaled_dot_to_mlir`
  (`__op$block_scaled_dot` + portable fallback) and the cuDNN FMHA attention custom calls.
- [`crates/ryft-xla/src/jit.rs`](crates/ryft-xla/src/jit.rs) and
  [`crates/ryft-xla/src/eager.rs`](crates/ryft-xla/src/eager.rs) — XLA compile/stage/jitted
  surface and eager op-by-op dispatch with the shared executable cache.
- [`crates/ryft/examples/mlp.rs`](crates/ryft/examples/mlp.rs) — the canonical end-to-end
  training example whose idioms §3 follows (generic `A: ArrayOperations` model code,
  `differentiate_at(..).with_captures(..).value_and_gradient(..)`, dual core/XLA runners).
