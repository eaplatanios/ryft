# Gemma 4 Training Support Plan

This document captures (1) the operation inventory needed to train Google's Gemma 4 family
end-to-end inside `ryft`, marking which primitives are already in `ryft-core` / `ryft-xla` and
which still need to be added, (2) the high-level implementation plan that would follow once the
missing primitives land, and (3) an aspirational `ryft` model implementation written against
that future API surface. The aspirational code does **not** compile against today's tree — it is
a forward-looking design artifact.

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

| Primitive (JAX) | Used by | State | Crate target | Notes |
|---|---|---|---|---|
| `lax.add` | residual stream, mask combination, AdamW first moment, gradient accumulation | ✅ | — | `AddOperation`, `SupportsAdd` |
| `lax.sub` | softmax stabilization (`x - max`), KV-cache positional masks | ✅ | — | `SubOperation`, `SupportsSub` |
| `lax.mul` | RMSNorm scale, mask × logits, attention `scale_factor`, AdamW second moment | ✅ | — | `MulOperation`, `SupportsMul` |
| `lax.div` | softmax normalization (denominator), AdamW update | ✅ | — | `DivOperation`, `SupportsDiv` |
| `lax.neg` | causal mask construction, sign of grads | ✅ | — | `NegOperation`, `SupportsNeg` |
| `Scale` (captured factor) | RoPE base-frequency multiply, scale * sqrt(d_model) embedding multiply | ✅ | — | `ScaleOperation`, `SupportsScale` |
| `lax.sin` | RoPE rotation | ✅ | — | `SinOperation`, `SupportsSin` |
| `lax.cos` | RoPE rotation | ✅ | — | `CosOperation`, `SupportsCos` |
| `lax.rsqrt` | RMSNorm `x * rsqrt(mean(x^2) + eps)` | ❌ | `ryft-core` + `ryft-xla` | New `RsqrtOperation` + StableHLO `stablehlo.rsqrt` |
| `lax.sqrt` | AdamW `sqrt(v_hat) + eps`, gradient global norm | ❌ | `ryft-core` + `ryft-xla` | New `SqrtOperation` + `stablehlo.sqrt` |
| `lax.exp` | softmax (`exp(x - max)`) | ❌ | `ryft-core` + `ryft-xla` | `ExpOperation` + `stablehlo.exponential` |
| `lax.log` | `logsumexp` for cross-entropy | ❌ | `ryft-core` + `ryft-xla` | `LogOperation` + `stablehlo.log` |
| `lax.tanh` | final logit softcap (`30 * tanh(logits / 30)`), GELU tanh-approx | ❌ | `ryft-core` + `ryft-xla` | `TanhOperation` + `stablehlo.tanh` |
| `lax.erf` | exact GELU (`0.5 * x * (1 + erf(x / sqrt(2)))`) | ❌ | `ryft-core` + `ryft-xla` | `ErfOperation` + `stablehlo.erf`. Optional if tanh-approx is chosen instead |
| `jax.nn.gelu` | GeGLU activation in MLP | ❌ | `ryft-core` | Composite — implementable as a value-level helper on top of `erf` (or `tanh`) + scale + add + mul. Adding it as a fused first-class op is also reasonable |
| `lax.abs` | gradient global norm (`sqrt(sum(g^2))` — `g^2` works, but `abs` shows up in clip-by-norm sentinels) | ❌ | `ryft-core` + `ryft-xla` | `AbsOperation` + `stablehlo.abs` |
| `lax.integer_pow` / `x*x` | `g^2` for variance/grad norm | ⚠️ | — | Achievable today via `Mul`; a fused `SquareOperation` is optional |
| `lax.max` (binary) / `lax.min` | `min(1, clip / norm)` in gradient clipping, attention bias floor | ❌ | `ryft-core` + `ryft-xla` | `MaxOperation`, `MinOperation` + `stablehlo.maximum`, `stablehlo.minimum`. Worth introducing together |
| `lax.clamp` | optional logit floor/ceil pre-softcap | ❌ | `ryft-core` + `ryft-xla` | Composite via `max`+`min`; first-class `ClampOperation` mirrors `stablehlo.clamp` |
| `lax.convert_element_type` | bf16 ↔ fp32 casts between forward, accumulation, and optimizer state | ❌ | `ryft-core` + `ryft-xla` | `ConvertElementTypeOperation` + `stablehlo.convert`. Needs proper differentiation (identity on the matching-precision branch, otherwise composed cast) |

### 1.2 Comparisons & boolean logic (mask construction)

| Primitive (JAX) | Used by | State | Crate target | Notes |
|---|---|---|---|---|
| `lax.eq`, `lax.ne` | padding masks (`tokens != pad_id`) | ❌ | `ryft-core` + `ryft-xla` | `EqOperation`, `NeOperation` + `stablehlo.compare` (EQ/NE) |
| `lax.lt`, `lax.le`, `lax.gt`, `lax.ge` | causal mask (`q_pos >= k_pos`), sliding-window mask (`q_pos - k_pos < W`) | ❌ | `ryft-core` + `ryft-xla` | One `CompareOperation { direction }` carrier mapping to `stablehlo.compare` is cleaner than four ops |
| `lax.bitwise_and` / `lax.bitwise_or` / `lax.bitwise_not` | mask combination (causal AND sliding), padding NOT, etc. | ❌ | `ryft-core` + `ryft-xla` | `AndOperation`, `OrOperation`, `NotOperation` + `stablehlo.and`, `stablehlo.or`, `stablehlo.not`. Boolean overloads with sensible promotion |
| `lax.select_n` (a.k.a. `where`) | applying mask to logits (replace with large negative), padding loss-mask | ✅ | — | Already covered by `SelectOperation` |

### 1.3 Reductions

| Primitive (JAX) | Used by | State | Crate target | Notes |
|---|---|---|---|---|
| `lax.reduce_sum` | RMSNorm `mean(x^2)`, softmax denominator, cross-entropy sum, grad norm | ❌ | `ryft-core` + `ryft-xla` | New `ReduceOperation { kind: Sum, axes, keepdims }` + lowering to `stablehlo.reduce` with the `add` body |
| `lax.reduce_max` | softmax numerical stability (`max(x, axis=-1)`) | ❌ | `ryft-core` + `ryft-xla` | Same `ReduceOperation` carrier with `Max` kind |
| `lax.reduce_min` | sliding-window bounds clamp (rare) | ❌ | `ryft-core` + `ryft-xla` | `ReduceOperation` with `Min` |
| `lax.reduce_prod` | (not used by Gemma 4; optional) | ❌ | `ryft-core` + `ryft-xla` | Optional |
| `mean` (composite) | RMSNorm | ⚠️ | `ryft-core` | Once `reduce_sum` + `Scale` exist, `mean` is a value-level helper |
| `logsumexp` (composite) | cross-entropy denominator | ⚠️ | `ryft-core` | Helper on `reduce_max` + `sub` + `exp` + `reduce_sum` + `log` + `add` |

The recommended shape is a single `ReduceOperation` primitive that carries an axis list, a
`keepdims` flag, and a small `ReduceKind` enum (`Sum`, `Max`, `Min`, optionally `Prod`). This
matches how StableHLO's `reduce` op is parameterized and keeps the carrier compact. The JVP/transpose
rules differ per kind, so the trait surface should still expose typed entry points
(`SupportsReduceSum`, `SupportsReduceMax`).

### 1.4 Shape & data movement

| Primitive (JAX) | Used by | State | Crate target | Notes |
|---|---|---|---|---|
| `lax.reshape` | folding GQA group dim into head dim, flattening for `dot_general` | ✅ | — | `ReshapeOperation`, `SupportsReshape` |
| `lax.transpose` / `lax.permute` | `[B,T,N,H] ↔ [B,N,T,H]`, RoPE half axis prep | ✅ | — | `TransposeOperation`, `SupportsTranspose` |
| `lax.broadcast_in_dim` | RMSNorm scale broadcast, mask broadcast across heads, RoPE position broadcast | ✅ | — | `BroadcastInDimOperation`, `SupportsBroadcastInDim` |
| `lax.concatenate` | RoPE half re-join, optional cache concat for prefill+decode | ❌ | `ryft-core` + `ryft-xla` | New `ConcatenateOperation { axis }` + `stablehlo.concatenate` |
| `lax.slice` (static) | RoPE half split, KV-cache prefix slicing | ❌ | `ryft-core` + `ryft-xla` | `SliceOperation { start_indices, limit_indices, strides }` + `stablehlo.slice` |
| `lax.dynamic_slice` | KV-cache reads at runtime offset, autoregressive decode | ❌ | `ryft-core` + `ryft-xla` | `DynamicSliceOperation { slice_sizes }` + `stablehlo.dynamic_slice` |
| `lax.dynamic_update_slice` | KV-cache writes | ❌ | `ryft-core` + `ryft-xla` | `DynamicUpdateSliceOperation` + `stablehlo.dynamic_update_slice` |
| `lax.gather` | token embedding lookup `table[input_ids]`, optional advanced indexing | ❌ | `ryft-core` + `ryft-xla` | `GatherOperation { dimension_numbers, slice_sizes }` + `stablehlo.gather`. Differentiated via a matching `scatter-add` |
| `lax.scatter` / `scatter-add` | gradient of `gather` (embedding bwd), MoE expert dispatch | ❌ | `ryft-core` + `ryft-xla` | `ScatterOperation { dimension_numbers, update_computation }` + `stablehlo.scatter`. Needed both as a primitive and as the `gather` adjoint |
| `lax.pad` | RoPE timescale padding with `+inf`, optional sequence padding | ❌ | `ryft-core` + `ryft-xla` | `PadOperation { padding_value, low, high, interior }` + `stablehlo.pad` |
| `lax.iota` | position indices, RoPE arange | ❌ | `ryft-core` + `ryft-xla` | `IotaOperation { dimension, type_ }` + `stablehlo.iota` |
| `lax.argmax` | inference sampling, top-1 accuracy metric | ❌ | `ryft-core` + `ryft-xla` | Lowers to `stablehlo.reduce` with a comparison body; first-class `ArgMaxOperation` keeps the IR readable |
| `lax.bitcast_convert_type` | not strictly needed; mention only for completeness | — | — | — |

### 1.5 Tensor contraction & matmul

| Primitive (JAX) | Used by | State | Crate target | Notes |
|---|---|---|---|---|
| `lax.dot_general` | Q/K/V projections, attention logits, output projection, MLP gate/up/down, vocab head | ✅ | — | `DotOperation` with full `DotDimensionNumbers` |
| `LeftDot` / `RightDot` (linearized factors) | transposition of `dot_general` in reverse-mode | ✅ | — | Already in `tracing_v2::operations::dot` |
| `einsum` (Flax style) | not needed as a primitive; lowers to `dot_general` + `transpose` + `reshape` | ✅ | — | Build as a value-level helper at the model layer |

### 1.6 Random number generation

| Primitive (JAX) | Used by | State | Crate target | Notes |
|---|---|---|---|---|
| `jax.random.split` (PRNG key plumbing) | reproducible streams per layer / per batch | ❌ | `ryft-core` | Needs a deterministic PRNG abstraction. Mirror JAX's threefry/Philox via a `PrngOperation` family |
| `jax.random.normal` | weight initialization (truncated normal in Flax) | ❌ | `ryft-core` + `ryft-xla` | `RandomNormalOperation` + `stablehlo.rng` |
| `jax.random.uniform` | dropout sampling, alternative inits | ❌ | `ryft-core` + `ryft-xla` | `RandomUniformOperation` + `stablehlo.rng` |
| `jax.random.bernoulli` | dropout mask | ❌ | `ryft-core` + `ryft-xla` | Composite on top of `uniform` + `compare`; not training-critical for Gemma 4 since dropout is off in the reference recipe |

For first-class training support we need at least a `(key, shape, dtype) -> array` interface to
`stablehlo.rng_bit_generator` together with stateless `split`. Weight initialization could be
implemented host-side (NumPy → device transfer) as a stopgap, but bit-exact reproducibility under
sharded data parallelism wants device-side RNG.

### 1.7 Control flow

| Primitive (JAX) | Used by | State | Crate target | Notes |
|---|---|---|---|---|
| `lax.cond` | optional logit softcap toggle, optional MoE-vs-dense branch | ✅ | — | `ConditionOperation` |
| `lax.while_loop` | autoregressive decode loop, training-loop step | ✅ | — | `WhileOperation` |
| `lax.scan` | layer-wise activation checkpointing, optional unrolled-vs-rolled choice | ❌ | `ryft-core` + `ryft-xla` | Can be desugared to `while_loop` + `dynamic_slice`/`dynamic_update_slice`; first-class `ScanOperation` keeps remat ergonomic |
| `lax.fori_loop` | optimizer step over parameter tree | ⚠️ | — | Built on `while_loop`; can be a value-level helper |
| `jax.checkpoint` (remat) | activation checkpointing per block | ❌ | `ryft-core` | A transform, not a primitive. Plumb a `CheckpointOperation` wrapper or a tracing-time policy that re-stages the inner program on the backward pass |

### 1.8 Parallelism & sharding

| Primitive (JAX) | Used by | State | Crate target | Notes |
|---|---|---|---|---|
| `jax.pjit` / mesh sharding annotations | data-parallel + tensor-parallel training | ✅ | — | `WithShardingConstraintOperation`, `Sharding`, `DeviceMesh` |
| `shard_map` | MoE expert dispatch, custom collective regions | ✅ | — | `ShardMapOperation`, `LinearShardMapOperation` |
| `lax.psum` / `lax.all_gather` / `lax.ppermute` | inside `shard_map` bodies | ❌ | `ryft-core` + `ryft-xla` | Collective operations need IR primitives; today they exist only implicitly via `shard_map` lowering. Add `AllReduceOperation { kind }`, `AllGatherOperation { axis }`, `CollectivePermuteOperation`, mirroring StableHLO ops |

### 1.9 Autodiff & training transforms

| Capability | Used by | State | Notes |
|---|---|---|---|
| Forward-mode JVP (`jvp`) | per-op linearization | ✅ | `JvpContext`, `JvpTracer` |
| Reverse-mode VJP (`vjp`) | training | ✅ | `vjp` in `tracing_v2::linear::reverse` |
| `value_and_grad`, `grad` | optimizer step | ✅ | Same module |
| Jacobian / Hessian | curvature-aware optimizers (optional) | ✅ | `Jacobian`, `Hessian` |
| `vmap` | per-example losses, sharded data parallelism | ✅ | `tracing_v2::batching::vmap` |
| Activation checkpointing (`remat`) | memory pressure at long context | ❌ | Needs a `Checkpoint` transform that re-traces the wrapped program during transpose |
| `pjit` compile + execute | end-to-end JIT to PJRT | ✅ | XLA backend via `arrays_v0::execution` |
| Stateful optimizer (Adam/AdamW) | training | ❌ | A pure-function pattern over a `Parameterized` tree; not a primitive, but a `ryft-core` module worth adding alongside the new reductions |
| Gradient clip-by-global-norm | training stability | ❌ | Composes `reduce_sum` + `sqrt` + `min` + `mul`; lives in the optimizer module |
| Mixed-precision policy (`bf16` activations, `fp32` master weights) | training | ⚠️ | Today `ryft` exposes all the data types but has no policy wrapper; add a `Policy` that inserts `convert_element_type` at module boundaries |

### 1.10 Summary checklist

- [x] `add`, `sub`, `mul`, `div`, `neg`, `scale`, `sin`, `cos`
- [x] `dot_general` (with batching), `LeftDot`, `RightDot`
- [x] `reshape`, `transpose`, `broadcast_in_dim`, `select`
- [x] `condition`, `while_loop`
- [x] `shard_map`, `with_sharding_constraint`
- [x] `jvp`, `vjp`, `grad`, `value_and_grad`, `vmap`
- [ ] `rsqrt`, `sqrt`, `exp`, `log`, `tanh`, `erf`, `abs`, `max`, `min`, `clamp`
- [ ] `convert_element_type`
- [ ] `compare` (EQ/NE/LT/LE/GT/GE), `and`, `or`, `not`
- [ ] `reduce` (`Sum`, `Max`, `Min`) with axis list and `keepdims`
- [ ] `concatenate`, `slice`, `dynamic_slice`, `dynamic_update_slice`
- [ ] `gather`, `scatter` (including scatter-add for embedding gradients)
- [ ] `pad`, `iota`, `argmax`
- [ ] Device-side RNG (`split`, `normal`, `uniform`)
- [ ] Collective IR ops (`all_reduce`, `all_gather`, `collective_permute`) for use inside `shard_map`
- [ ] Activation checkpointing transform
- [ ] Optimizer module (AdamW + clip-by-global-norm) with mixed-precision policy

---

## 2. Implementation Plan

The plan is organized so each phase produces something that can be run, tested, and benchmarked
end-to-end against a reference (JAX, PyTorch, or both). The pre-existing `ryft` conventions for
operation hierarchy (`SupportsXxx` capability traits, value-level entry points on `Tracer`, fused
StableHLO lowerings in `ryft-xla::experimental::lowering`) apply uniformly. Each new primitive
follows the same five-step contract: type/abstract-eval rule → carrier variant + `Supports`
trait → forward (JVP) rule → transposition (cotangent) rule → StableHLO lowering. A primitive is
not "done" until all five plus per-primitive unit tests are in.

### Phase 0 — Scaffolding the model crate

1. Add a new crate `crates/ryft-models` (depends on `ryft-core` and `ryft-xla`) to house the
   Gemma 4 implementation, configurations, and integration tests. Keeping models out of
   `ryft-core` preserves the rule that `ryft-core` only owns the IR and transforms.
2. Land a thin `ryft-models::common` module for primitives that are useful across models
   (RMSNorm, RoPE, GeGLU, GQA attention, AdamW, gradient clipping). Each helper is a value-level
   function over `Tracer<'_, D>` (or a generic `V: Traceable<ArrayType>` plus the trait bounds for
   the primitives it uses) so it stages cleanly under any backend.
3. Create `crates/ryft-models/examples/gemma_4_train.rs` as the end-to-end harness.

### Phase 1 — Elementwise math primitives

Land the unary float primitives that block everything else: `rsqrt`, `sqrt`, `exp`, `log`, `tanh`,
`erf`, `abs`. Each follows the established pattern of `XxxOperation`, `Xxx` value trait,
`SupportsXxx`, JVP rule, transpose rule (most are non-linear, so they appear in the linear program
only as captured-factor multiplies via `LeftDot` / `RightDot`-style helpers), and `stablehlo.xxx`
lowering. Then land `max`, `min`, `clamp` (binary plus the three-arg `clamp`) — these need a
value-level argmax-style differentiation rule (`grad` flows only through the selected operand).

Land `convert_element_type` with `differentiation` that casts the cotangent back to the source
dtype. This unlocks bf16 forward / fp32 master weights.

### Phase 2 — Comparisons, masks, and `select`-driven masking

Add a single `CompareOperation { direction: Eq | Ne | Lt | Le | Gt | Ge }` carrier with a
`SupportsCompare` trait and a tiny value-level surface (`a.eq(b)`, `a.lt(b)`, …). Boolean ops
(`and`, `or`, `not`) follow the same shape. Mask construction in attention (causal AND sliding) is
already lower-bound on `select` — once compares exist, the full mask flow drops out.

### Phase 3 — Reductions

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

### Phase 4 — Shape ops & indexing

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

### Phase 5 — RNG & initialization

Pick a stateless PRNG (threefry-2x32 is the obvious choice, since it lowers cleanly to
`stablehlo.rng_bit_generator`). Add a `PrngKey` value type, a `split(key, n) -> Vec<PrngKey>`
helper, and `random_normal(key, shape, dtype) -> Array` / `random_uniform(key, shape, dtype, [lo,
hi])`. Truncated-normal initialization (Flax's default for Linen) can be built on top via
rejection sampling inside a `while_loop`, or as a `pjit`-time host helper for the first cut.

### Phase 6 — Activation checkpointing & optimizer

Add a `Checkpoint` transform that, when staged into the forward, marks the wrapped sub-program for
re-tracing on the backward pass. Concretely: during `linearize`, the inside of a checkpointed
region produces a tangent program that records *only* the symbolic inputs; during transpose, the
primal is re-evaluated. This is the standard JAX `remat` model.

Add an `optimizer` module: `AdamWState<P: Parameter>`, `adamw_step(params, grads, state, hyper) ->
(new_params, new_state)`, and `clip_by_global_norm(grads, max_norm) -> grads_clipped`. All of them
are value-level functions over `Parameterized` trees and stage into the same JIT graph as the
forward+backward.

### Phase 7 — Sharding & collectives

Add `AllReduceOperation`, `AllGatherOperation`, `ReduceScatterOperation`, and
`CollectivePermuteOperation` for use inside `shard_map` bodies. These are needed for tensor
parallelism (head-sharded GQA needs an `all_reduce` after the output projection) and for
MoE dispatch (`all_to_all` can be expressed as `reduce_scatter` + `all_gather` or as a dedicated
primitive). The `Mesh` and `Sharding` machinery already exists; this phase only adds the
IR primitives.

### Phase 8 — Gemma 4 model code & training harness

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

### Phase 9 — Performance & validation

1. Microbenchmarks per primitive (`cargo bench -p ryft-core`).
2. End-to-end throughput on a single GPU, then on a 2x2 mesh.
3. Loss-curve parity vs. the Flax reference on a small mixture for ≥1k steps.

---

## 3. Aspirational `ryft` Model Implementation

The code below is intentionally written against the **post-plan** API surface. It will not
compile today; primitives noted as missing in §1 are used directly. The intent is to make the
target ergonomics concrete so that, as each primitive lands, the public surface in
`ryft-models::common::nn` can be shaped to land at exactly this call-site density.

The example targets the **Gemma 4 E2B** variant. Other variants drop in by changing the config.

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
    /// Query projection, shape `[embed_dim, query_head_count, head_dim]`.
    pub q_proj: P,

    /// Key projection, shape `[embed_dim, kv_head_count, head_dim]`.
    pub k_proj: P,

    /// Value projection, shape `[embed_dim, kv_head_count, head_dim]`.
    pub v_proj: P,

    /// Output projection, shape `[query_head_count, head_dim, embed_dim]`.
    pub o_proj: P,

    /// QK-norm scale on queries.
    pub query_norm: RmsNormScale<P>,

    /// QK-norm scale on keys.
    pub key_norm: RmsNormScale<P>,
}

#[derive(Clone, Debug, Parameterized)]
pub struct MlpParams<P: Parameter> {
    /// Gating projection, shape `[2, embed_dim, mlp_hidden_dim]`. Index 0 is the GELU branch and
    /// index 1 is the linear branch.
    pub gating: P,

    /// Down projection, shape `[mlp_hidden_dim, embed_dim]`.
    pub down: P,
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
// Helper functions consumed by the Gemma 4 model. They are generic over any tracing domain whose
// operation type supports the relevant primitives. Once the missing primitives in §1 land,
// these helpers compose entirely out of value-level traits like `Add`, `Mul`, `Reshape`, etc.

use ryft_core::tracing::{Traceable, TracingError};
use ryft_core::tracing_v2::operations::{
    BroadcastInDim, Concatenate, Dot, DotDimensionNumbers, Gather, GatherDimensionNumbers, Iota,
    Pad, Reduce, ReduceKind, Reshape, Select, Slice, Transpose,
};
use ryft_core::types::{ArrayType, DataType, Shape};

/// `x * rsqrt(mean(x*x, axis=-1, keepdims=true) + epsilon) * (1 + scale)`.
pub fn rms_norm<V: Reduce + Mul + Rsqrt + Add + BroadcastInDim + Clone>(
    x: V,
    scale: Option<V>,
    epsilon: V,
) -> Result<V, TracingError> {
    let last_axis = x.r#type().rank() - 1;
    let square = x.clone() * x.clone();
    let variance = square.reduce(ReduceKind::Sum, vec![last_axis], /*keepdims=*/ true)?;
    let variance_mean = variance * V::scalar_like(&x, 1.0 / x.r#type().dimension(-1).value().unwrap() as f32)?;
    let inv = (variance_mean + epsilon.broadcast_like(&variance_mean)?).rsqrt();
    let normalized = x * inv.broadcast_like(&x)?;
    match scale {
        Some(scale) => {
            let one_plus_scale = scale + V::scalar_like(&normalized, 1.0)?;
            Ok(normalized * one_plus_scale.broadcast_like(&normalized)?)
        }
        None => Ok(normalized),
    }
}

/// GeGLU: `gelu(gate[..., 0, :]) * gate[..., 1, :]`, where `gate` is the result of the gating
/// einsum and has shape `[..., 2, hidden_dim]`.
pub fn geglu<V: Slice + Gelu + Mul>(gate: V) -> Result<V, TracingError> {
    let inner = gate.r#type().rank() - 2;
    let gelu_branch = gate.clone().slice_axis(inner, 0, 1)?.squeeze_axis(inner)?.gelu();
    let linear_branch = gate.slice_axis(inner, 1, 2)?.squeeze_axis(inner)?;
    Ok(gelu_branch * linear_branch)
}

/// Stable softmax along the last axis.
pub fn softmax_last<V: Reduce + Sub + Exp + Div + BroadcastInDim + Clone>(x: V) -> Result<V, TracingError> {
    let last = x.r#type().rank() - 1;
    let max = x.clone().reduce(ReduceKind::Max, vec![last], true)?;
    let shifted = x - max.broadcast_like(&x)?;
    let exp = shifted.exp();
    let sum = exp.clone().reduce(ReduceKind::Sum, vec![last], true)?;
    Ok(exp / sum.broadcast_like(&exp)?)
}

/// RoPE that rotates the leading `rotated_dim` head dimensions and leaves the trailing
/// `head_dim - rotated_dim` dimensions untouched.
pub fn apply_rope<V>(
    x: V,
    positions: V,
    head_dim: usize,
    rotated_dim: usize,
    base_frequency: f32,
) -> Result<V, TracingError>
where
    V: Iota + Scale + Sin + Cos + Slice + Concatenate + Mul + Sub + Add + Reshape + BroadcastInDim + Clone,
{
    assert!(rotated_dim % 2 == 0 && rotated_dim <= head_dim);
    let half = rotated_dim / 2;
    let exponents = V::iota(half, V::scalar_dtype())?
        * V::scalar_like(&x, 2.0 / head_dim as f32)?;
    let timescale = exponents.scale_constant(base_frequency, /*as_power_of=*/ true);
    // Pad the trailing `head_dim/2 - half` dims with `+inf` so cos=1, sin=0 leave them unchanged.
    let timescale = timescale.pad_high(0, head_dim / 2 - half, V::scalar_like(&x, f32::INFINITY)?)?;
    let theta = positions.unsqueeze(-1) / timescale.broadcast_like(&positions.unsqueeze(-1))?;
    let cos = theta.clone().cos();
    let sin = theta.sin();
    let first = x.clone().slice_axis(-1, 0, head_dim / 2)?;
    let second = x.slice_axis(-1, head_dim / 2, head_dim)?;
    let out_first = first.clone() * cos.broadcast_like(&first)? - second.clone() * sin.broadcast_like(&second)?;
    let out_second = second * cos.broadcast_like(&first)? + first * sin.broadcast_like(&first)?;
    out_first.concatenate(out_second, -1)
}
```

### 3.4 Attention, MLP, and block

```rust
// crates/ryft-models/src/gemma_4/attention.rs
use ryft_core::tracing_v2::operations::{Dot, DotDimensionNumbers};
use ryft_core::types::DataType;

use crate::common::nn::{apply_rope, rms_norm, softmax_last};
use crate::gemma_4::{AttentionKind, AttentionParams, Gemma4Config};

const NEG_INF: f32 = -2.3819763e38;

/// Attention forward over `[batch, seq_len, embed_dim]`. Returns activations of the same shape.
pub fn attention<V>(
    config: &Gemma4Config,
    layer_kind: AttentionKind,
    layer_rope_base_frequency: f32,
    layer_rotated_dim: usize,
    params: &AttentionParams<V>,
    x: V,
    positions: V,
    attention_mask: V,
) -> Result<V, TracingError>
where
    V: /* the same trait bouquet plus `Compare`, `And`, `Select`, etc. */ Clone,
{
    let batch = x.r#type().dimension(0).value().unwrap();
    let seq = x.r#type().dimension(1).value().unwrap();
    let head_dim = config.head_dim;
    let qh = config.query_head_count;
    let kvh = config.kv_head_count;
    let groups = qh / kvh;

    // Q/K/V projections via `dot_general` with reshape-folded head axes.
    let queries = x.clone().einsum_3d("BTD,DNH->BTNH", &params.q_proj)?;
    let keys = x.clone().einsum_3d("BTD,DKH->BTKH", &params.k_proj)?;
    let values = x.einsum_3d("BTD,DKH->BTKH", &params.v_proj)?;

    // QK-norm.
    let queries = rms_norm(queries, Some(params.query_norm.scale.clone()), epsilon_v(config))?;
    let keys = rms_norm(keys, Some(params.key_norm.scale.clone()), epsilon_v(config))?;

    // Partial RoPE.
    let queries = apply_rope(queries, positions.clone(), head_dim, layer_rotated_dim, layer_rope_base_frequency)?;
    let keys = apply_rope(keys, positions, head_dim, layer_rotated_dim, layer_rope_base_frequency)?;

    // Reshape Q to expose the GQA group axis: [B, T, K, G, H].
    let queries = queries.reshape(shape![batch, seq, kvh, groups, head_dim])?;

    // Logits via grouped `dot_general`: BTKGH,BSKH -> BTKGS.
    let logits = queries.dot_general(
        keys.clone(),
        &DotDimensionNumbers::new(
            /*lhs_contract=*/ vec![4],
            /*rhs_contract=*/ vec![3],
            /*lhs_batch=*/ vec![0, 2],
            /*rhs_batch=*/ vec![0, 2],
        ),
    )? * V::scalar_like(&queries, (head_dim as f32).rsqrt())?;

    // Mask: pre-built outside the attention block (causal AND, for local layers, sliding-window).
    let masked = V::select(attention_mask, logits, V::scalar_like(&logits, NEG_INF)?)?;
    let weights = softmax_last(masked)?;

    // Weighted sum: BTKGS,BSKH -> BTKGH then reshape to BTNH.
    let context = weights.dot_general(
        values,
        &DotDimensionNumbers::new(vec![4], vec![1], vec![0, 2], vec![0, 2]),
    )?;
    let context = context.reshape(shape![batch, seq, qh, head_dim])?;

    // Output projection.
    context.einsum_3d("BTNH,NHD->BTD", &params.o_proj)
}
```

```rust
// crates/ryft-models/src/gemma_4/mlp.rs
pub fn mlp<V>(params: &MlpParams<V>, x: V) -> Result<V, TracingError> {
    let gate = x.einsum_3d("...D,NHF->...NF", &params.gating)?; // [B, T, 2, hidden_dim]
    let activations = geglu(gate)?;                              // [B, T, hidden_dim]
    activations.einsum_2d("...H,HD->...D", &params.down)
}
```

```rust
// crates/ryft-models/src/gemma_4/block.rs
pub fn block<V>(
    config: &Gemma4Config,
    layer_index: usize,
    params: &BlockParams<V>,
    x: V,
    positions: V,
    attention_mask: V,
) -> Result<V, TracingError> {
    let kind = config.attention_pattern[layer_index];
    let (base, rotated_dim) = match kind {
        AttentionKind::Local => (config.local_rope_base_frequency, config.head_dim),
        AttentionKind::Global => (
            config.global_rope_base_frequency,
            (config.head_dim as f32 * config.global_rope_proportion) as usize,
        ),
    };

    let attended = {
        let normed = rms_norm(x.clone(), Some(params.pre_attention_norm.scale.clone()), epsilon_v(config))?;
        let attention_out = attention(
            config, kind, base, rotated_dim, &params.attention, normed, positions.clone(), attention_mask.clone(),
        )?;
        rms_norm(attention_out, Some(params.post_attention_norm.scale.clone()), epsilon_v(config))?
    };
    let x = x + attended * params.skip_scale.broadcast_like(&attended)?;

    let mlp_out = {
        let normed = rms_norm(x.clone(), Some(params.pre_mlp_norm.scale.clone()), epsilon_v(config))?;
        let mlp_out = mlp(&params.mlp, normed)?;
        rms_norm(mlp_out, Some(params.post_mlp_norm.scale.clone()), epsilon_v(config))?
    };
    Ok(x + mlp_out * params.skip_scale.broadcast_like(&mlp_out)?)
}
```

### 3.5 Forward, loss, and one training step

```rust
// crates/ryft-models/src/gemma_4/forward.rs
pub fn forward<V>(
    config: &Gemma4Config,
    params: &Gemma4Params<V>,
    tokens: V,        // [batch, seq] of int32
    positions: V,     // [batch, seq] of int32
) -> Result<V, TracingError> {
    // Token embedding: gather + scale by sqrt(embed_dim).
    let embeds = params.embedder.table.clone().gather_rows(tokens.clone())?;
    let mut hidden = embeds * V::scalar_like(&embeds, (config.embed_dim as f32).sqrt())?;

    // Per-layer-input embeddings: [B, T, layer_count, ple_dim].
    let ple = params.embedder.per_layer_inputs.clone().gather_rows(tokens.clone())?;

    // Build the attention mask once. Causal for global, causal AND sliding window for local.
    let mask = build_attention_mask(config, &positions);

    for (layer_index, layer_params) in params.blocks.iter().enumerate() {
        hidden = block(config, layer_index, layer_params, hidden, positions.clone(), mask.clone())?;
        // Per-layer-input gating injection: take `ple[..., layer_index, :]`, project, norm, add.
        let layer_ple = ple.clone().slice_axis(-2, layer_index, layer_index + 1)?.squeeze_axis(-2)?;
        let projected = layer_ple.einsum_2d("...P,PD->...D", &params.embedder.per_layer_projection)?;
        hidden = hidden + projected;
    }

    // Final RMSNorm + tied output projection + softcap.
    let normed = rms_norm(hidden, Some(params.final_norm.scale.clone()), epsilon_v(config))?;
    let logits = normed.einsum_2d("...D,VD->...V", &params.embedder.table)?;
    let cap = V::scalar_like(&logits, config.final_logit_softcap)?;
    Ok((logits / cap.broadcast_like(&logits)?).tanh() * cap.broadcast_like(&logits)?)
}
```

```rust
// crates/ryft-models/src/gemma_4/loss.rs
pub fn loss<V>(
    config: &Gemma4Config,
    params: &Gemma4Params<V>,
    tokens: V,           // [batch, seq+1] int32
    positions: V,        // [batch, seq] int32
    loss_mask: V,        // [batch, seq] in {0.0, 1.0}
) -> Result<V, TracingError> {
    let inputs = tokens.clone().slice_axis(1, 0, tokens.r#type().dimension(1).value().unwrap() - 1)?;
    let targets = tokens.slice_axis(1, 1, tokens.r#type().dimension(1).value().unwrap())?;
    let logits = forward(config, params, inputs, positions)?;

    // Stable cross-entropy via `logsumexp - target_logit`.
    let max = logits.clone().reduce(ReduceKind::Max, vec![-1], true)?;
    let shifted = logits.clone() - max.broadcast_like(&logits)?;
    let logsumexp = max.squeeze_axis(-1)? + shifted.exp().reduce(ReduceKind::Sum, vec![-1], false)?.log();
    let target_logit = logits.gather_along_last_axis(targets)?;
    let per_token_loss = logsumexp - target_logit;
    let masked_loss = per_token_loss * loss_mask.clone();
    let total = masked_loss.reduce(ReduceKind::Sum, vec![0, 1], false)?;
    let normalizer = loss_mask.reduce(ReduceKind::Sum, vec![0, 1], false)?;
    Ok(total / normalizer)
}
```

```rust
// crates/ryft-models/src/gemma_4/train.rs
use ryft_core::tracing_v2::linear::value_and_grad;

pub fn train_step<V>(
    config: &Gemma4Config,
    params: Gemma4Params<V>,
    optimizer_state: AdamWState<V>,
    hyper: AdamWHyper,
    batch: Batch<V>,
) -> Result<(Gemma4Params<V>, AdamWState<V>, V), TracingError> {
    // 1. value_and_grad: returns (scalar_loss, grad_tree_matching_params).
    let (loss_value, gradients) = value_and_grad(
        /*domain=*/ &xla_domain(),
        |params| loss(config, &params, batch.tokens.clone(), batch.positions.clone(), batch.loss_mask.clone()),
        params.clone(),
    )?;

    // 2. Clip-by-global-norm.
    let gradients = clip_by_global_norm(gradients, hyper.max_global_norm)?;

    // 3. AdamW step.
    let (new_params, new_state) = adamw_step(params, gradients, optimizer_state, hyper)?;

    Ok((new_params, new_state, loss_value))
}
```

### 3.6 End-to-end driver

```rust
// crates/ryft-models/examples/gemma_4_train.rs
use ryft_core::sharding::{DeviceMesh, MeshAxisType};
use ryft_core::tracing_v2::operations::ShardMap;
use ryft_models::gemma_4::{Gemma4Config, Gemma4Params, train_step, AdamWHyper, AdamWState};
use ryft_xla::experimental::XlaDomain;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Gemma4Config::e2b();
    let mesh = DeviceMesh::new(/* logical mesh: data×model */)?;
    let domain = XlaDomain::new(&mesh)?;

    // Initialize parameters via the device RNG (Phase 5 work).
    let mut params: Gemma4Params<ArrayValue> = initialize_gemma_4(&domain, &config, /*seed=*/ 0)?;
    let mut optimizer_state = AdamWState::zeros_like(&params);
    let hyper = AdamWHyper { learning_rate: 1e-4, weight_decay: 0.1, b1: 0.9, b2: 0.95, eps: 1e-8, max_global_norm: 1.0 };

    // JIT-compile train_step once.
    let compiled = domain.jit(|inputs| {
        let (params, opt_state, batch) = inputs;
        train_step(&config, params, opt_state, hyper.clone(), batch)
    })?;

    for batch in data_loader()? {
        let (new_params, new_state, loss) = compiled.run((params, optimizer_state, batch))?;
        params = new_params;
        optimizer_state = new_state;
        println!("loss = {}", loss.to_host_scalar::<f32>()?);
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

- `DataType::F4E2M1FN` (NVFP4 data), `DataType::F8E4M3FN` (FP8 forward), `DataType::F8E5M2`
  (FP8 backward), `DataType::F8E8M0FNU` (UE8M0 microscale), `DataType::F8E4M3` (UE4M3
  microscale) are all already in the enum at
  [crates/ryft-core/src/types/data.rs:759](crates/ryft-core/src/types/data.rs:759).
  These mirror the StableHLO type set — see the
  [StableHLO specification](https://openxla.org/stablehlo/spec), the
  [F8E4M3/F8E3M4 RFC](https://github.com/openxla/stablehlo/blob/main/rfcs/20240808-f8E4M3_f8E3M4.md),
  and the [Speccing StableHLO quantization](https://groups.google.com/a/openxla.org/g/openxla-discuss/c/iwE9is49SS4)
  thread for the upstream rationale. The promotion lattice intentionally excludes them, which is
  correct — they are conversion-only.
- `ArrayType` already accepts any `DataType`, so the IR can carry FP4/FP8 tensors today; what is
  missing is the operations that produce, consume, or compute with them.

### 4.2 Missing operation inventory

| Capability | Used by | State | Crate target | Notes |
|---|---|---|---|---|
| `quantize_scaled(x, block_size, scale_dtype, value_dtype)` | producing FP8/NVFP4 weights & activations from bf16 input | ❌ | `ryft-core` + `ryft-xla` | New `QuantizeScaledOperation` returns a `(values, scales)` pair. Block reduction computes per-block `amax`, scale = `amax / fmax(value_dtype)`, then casts to `value_dtype` (rounded), and casts the scale to `scale_dtype`. For per-tensor FP8, set `block_size = -1` (whole-tensor reduction). |
| `dequantize_scaled(values, scales, block_size)` | rare: recovery for debugging, mixed paths | ❌ | `ryft-core` + `ryft-xla` | New `DequantizeScaledOperation`; the inverse of the above. Lowers to broadcast + multiply + convert. |
| `scaled_dot_general(lhs, lhs_scales, rhs, rhs_scales, dimensions, accumulator_dtype)` | every matmul in the model | ❌ | `ryft-core` + `ryft-xla` | New `ScaledDotGeneralOperation`. Carrier: same `DotDimensionNumbers` plus a `ScaledDotConfig { lhs_block_size, rhs_block_size, accumulator_dtype }`. JVP and transposition rules treat it as a generalized linear op (linear in both data operands; the scales come in as auxiliary captured tensors), which mirrors how Transformer-Engine's `fp8_gemm` is exposed. |
| `reduce_max_abs` (per-block) | amax inside `quantize_scaled` | ❌ | `ryft-core` + `ryft-xla` | Once `abs` and `reduce_max` from Phase 1/3 land, this is a composite (`abs` + `reduce_max` over the block axis after reshape). A first-class fused variant pays off only if the lowering benefits from a single `stablehlo.reduce` — keep it composite by default. |
| `convert_element_type` for FP4/FP8 → bf16 and back | the boundary of every scaled GEMM | ⚠️ | `ryft-core` + `ryft-xla` | Same primitive as Phase 1, but the lowering must hand off to `stablehlo.convert` with the correct rounding mode (RTE for forward, stochastic rounding optional for backward grads). |
| Delayed-scaling amax history buffer | FP8 per-tensor scale tracker | ❌ | `ryft-models::optimizer` | A pure-function pattern: `(amax_history, current_amax) -> (new_amax_history, scale)`. No new IR primitive — needs `reduce_max`, `slice`, `dynamic_update_slice`, and `concatenate`, all of which appear in Phase 1/3/4 already. |
| Stochastic rounding for backward gradients | optional, improves NVFP4 backward quality | ❌ | `ryft-core` + `ryft-xla` | A flag on `convert_element_type`. Needs the Phase 5 RNG primitive. |

The shape of the new IR primitive is intentionally small — three additions
(`QuantizeScaled`, `DequantizeScaled`, `ScaledDotGeneral`) cover the entire FP8/NVFP4 surface
that Gemma 4 training touches. Activation casts at the input and output of each scaled GEMM, the
Transformer Engine-style amax tracker, and the mixed-precision boundary policy are all built on
top of those three plus existing primitives.

### 4.3 Lowering strategy

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

#### 4.3.3 Why `ryft-core` still wants a `ScaledDotGeneralOperation`

The IR-level `ScaledDotGeneralOperation` in `ryft-core` is justified independently of either
lowering: it preserves the scaled-matmul shape through `ryft`'s autodiff and sharding
transforms (both of which need to see the matmul as one node, not as an expanded sequence
that's harder to differentiate and harder to shard). The lowering rule in `ryft-xla` is then
the single place where the decision "FP8 → expand into the 6-op template; MX → emit a cuDNN
custom call" lives, and that rule is small enough that swapping it for a future upstream HLO
op is a one-day change.

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

### 4.5 API surface (aspirational)

The model code from §3 changes only at matmul call sites. The helper takes the same
`DotDimensionNumbers` plus a `ScaledDotConfig` describing the quantization regime; everything
else (residual, normalization, mask) stays identical.

```rust
// crates/ryft-models/src/common/nn.rs

use ryft_core::types::DataType;

/// Microscaling format for one operand of a scaled GEMM.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ScaledFormat {
    /// Per-tensor FP8 E4M3, scale dtype = `F32` (Hopper-style).
    Fp8E4m3PerTensor,

    /// Per-tensor FP8 E5M2 — used for backward operands.
    Fp8E5m2PerTensor,

    /// NVFP4: 16-element blocks, scale dtype = `F8E4M3FN` (UE4M3), plus per-tensor `F32` scale.
    Nvfp4Block16,

    /// MXFP4: 32-element blocks, scale dtype = `F8E8M0FNU` (UE8M0).
    Mxfp4Block32,
}

#[derive(Clone, Debug)]
pub struct ScaledDotConfig {
    /// Quantization of the LHS operand.
    pub lhs_format: ScaledFormat,

    /// Quantization of the RHS operand.
    pub rhs_format: ScaledFormat,

    /// Accumulator dtype. Almost always `DataType::F32` on Blackwell.
    pub accumulator_dtype: DataType,

    /// Output dtype. Typically `DataType::BF16` because the next consumer is a residual.
    pub output_dtype: DataType,
}

/// Scaled `dot_general`. Quantizes both operands inline (computing block-amax → scale → cast),
/// invokes `tcgen05.mma` via the configured lowering target, and downcasts the accumulator to
/// `output_dtype`. Differentiation rules treat the scaled GEMM as a linear op in both data
/// operands; the scales are captured at trace time and rebuilt per step in the backward pass.
pub fn scaled_dot_general<V>(
    lhs: V,
    rhs: V,
    dimensions: &DotDimensionNumbers,
    config: &ScaledDotConfig,
) -> Result<V, TracingError>
where
    V: QuantizeScaled + ScaledDotGeneral + ConvertElementType + Clone,
{
    let (lhs_values, lhs_scales) = lhs.quantize_scaled(config.lhs_format)?;
    let (rhs_values, rhs_scales) = rhs.quantize_scaled(config.rhs_format)?;
    V::scaled_dot_general(lhs_values, lhs_scales, rhs_values, rhs_scales, dimensions, config)
}
```

Inside the model, the only delta is replacing `einsum_3d` / `dot_general` calls with
`scaled_dot_general`:

```rust
// crates/ryft-models/src/gemma_4/attention.rs (delta)

let logits = scaled_dot_general(
    queries,
    keys,
    &DotDimensionNumbers::new(vec![4], vec![3], vec![0, 2], vec![0, 2]),
    &ScaledDotConfig {
        lhs_format: ScaledFormat::Fp8E4m3PerTensor,
        rhs_format: ScaledFormat::Fp8E4m3PerTensor,
        accumulator_dtype: DataType::F32,
        output_dtype: DataType::BF16,
    },
)? * V::scalar_like(&queries, (head_dim as f32).rsqrt())?;
```

```rust
// crates/ryft-models/src/gemma_4/mlp.rs (delta)

let gate = scaled_dot_general(
    x,
    &params.gating,
    &DotDimensionNumbers::new(vec![2], vec![1], vec![], vec![]),
    &ScaledDotConfig {
        lhs_format: ScaledFormat::Nvfp4Block16,
        rhs_format: ScaledFormat::Nvfp4Block16,
        accumulator_dtype: DataType::F32,
        output_dtype: DataType::BF16,
    },
)?;
```

A `Policy` struct on the model config (mentioned as a Phase 6 deliverable in §2) carries the
default `ScaledDotConfig` for each role (attention, MLP, unembedding) so the call sites stay
free of repeated boilerplate.

### 4.6 Implementation phases (delta on top of §2)

These slot in between existing phases without re-ordering them:

- **Phase 1.5 — Conversion and quantization.** Land `ConvertElementTypeOperation` (already in
  Phase 1) with explicit support for FP4/FP8 source/target plus rounding-mode metadata. Add
  `QuantizeScaledOperation` and `DequantizeScaledOperation` on top of the Phase 3 `Reduce`. Unit
  tests: round-trip `bf16 → NVFP4 → bf16` matches the Transformer Engine reference within
  one ULP per block; `amax` lines up bitwise with cuBLASLt's helper.
- **Phase 4.5 — Scaled GEMM lowering.** Introduce `ScaledDotGeneralOperation` plus
  `SupportsScaledDotGeneral`. Wire two lowering paths chosen per `ScaledFormat`: (a) for FP8,
  expand to `stablehlo.convert` + `stablehlo.multiply` + `stablehlo.dot_general` so XLA's
  [`gemm-rewriter`](https://github.com/openxla/xla/blob/main/xla/service/gpu/transforms/gemm_rewriter.cc)
  pattern-matches it into a `__cublas$lt$matmul$f8` custom call
  ([RFC #22](https://github.com/openxla/xla/discussions/22),
  [tensorflow#58720](https://github.com/tensorflow/tensorflow/pull/58720)); (b) for NVFP4/MXFP4,
  emit a `stablehlo.custom_call` to the cuBLASLt scaled-matmul entry point with the operands
  represented as `(elements, scales)` tuples
  ([RFC #18085](https://github.com/openxla/xla/discussions/18085)). Add a unit test that golden
  compares the FP8 lowering output against the [JAX FP8 fusion smoke
  test](https://github.com/jax-ml/jax/issues/22313) shape so we catch any rewrite-breaking IR
  drift early. Forward differentiation rule: linear in both data operands. Transpose rule:
  produces two scaled GEMMs against the cotangent, with `E5M2`-flavored scales for the backward
  operand pair.
- **Phase 6.5 — Scale tracker.** Build a small `optimizer::scaling` module: a delayed-scaling
  amax history buffer (`history: [steps, num_tensors]`), `update_scale_from_amax(history, amax)`
  per training step, and a hook into `train_step` that wires the per-tensor scales into the
  matmul call sites via the model's `Policy`.
- **Phase 8.5 — End-to-end FP8/NVFP4 Gemma 4.** Re-trace `train_step` with the scaled GEMM path
  selected by `Policy`. Validate against the bf16 baseline on a fixed batch: per-step parameter
  delta should match within `1e-3` (FP8) or `5e-3` (NVFP4), and the 1k-step loss curve should
  be statistically indistinguishable.

### 4.7 Verification plan

1. **Per-op numerics.** For `QuantizeScaled`, `DequantizeScaled`, and `ScaledDotGeneral`, build
   reference NumPy/PyTorch implementations and check ULP-level agreement on randomized inputs.
   The primary oracle for `ScaledDotGeneral` is
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
5. **A100 fallback.** When the device lacks FP8/NVFP4 hardware, the `ScaledDotGeneral` lowering
   should fall back to a `dequantize → bf16 dot_general → quantize` chain transparently and emit
   a warning. This keeps `ryft-models` portable for development on Hopper or Ampere.

### 4.8 Open questions specific to Blackwell

- **NVFP4 vs MXFP4.** NVIDIA's NVFP4 (16-element blocks, UE4M3 scale, plus per-tensor scale) and
  the [OCP MXFP4 standard](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
  (32-element blocks, UE8M0 scale) coexist on Blackwell. We need both in `ScaledFormat`; default
  to NVFP4 because the reference recipes target it and the loss-curve evidence in
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

- **Mixed-precision policy boundary.** The aspirational code does the bf16↔fp32 casts implicitly
  inside RMSNorm and softmax. The actual lowering needs to make those casts explicit so XLA does
  not silently downcast accumulators on GPU. Consider a `Policy` wrapper that traces the
  conversions into the IR.
- **KV-cache sharing & PLE during training.** The reference implementation conditionally reuses K
  and V from a donor layer (`kv_shared_cache`). During training (no decode), this collapses to
  "skip the K/V projection on the consuming layer and read from the donor's pre-RoPE K/V". The
  parameter tree captures this only implicitly via shared `Vec<BlockParams>` indices; a tagged
  `BlockKind` enum may be clearer.
- **MoE variant (26B-A4B).** The aspirational code targets the dense E2B/E4B path. The MoE
  variant needs `top_k`, ragged `gather`/`scatter` (Phase 4 already plans `scatter`, but `top_k`
  itself is a new IR primitive), and a `shard_map` body that does an `all_to_all` exchange before
  expert evaluation. Add this as a Phase 8b once the dense path is stable.
- **`einsum` ergonomics.** The aspirational code uses an `einsum_2d` / `einsum_3d` helper. The
  underlying primitive is already `dot_general`; the helper is a string-spec frontend. Decide
  whether to make this a compile-time parsed macro (`einsum!("BTKGH,BSKH->BTKGS", q, k)`) or a
  runtime-parsed builder.
- **Determinism of dropout / init under sharding.** Bit-exact reproducibility across mesh shapes
  requires the device RNG (Phase 5) to be stateless and sharding-aware. Punting to host-side
  initialization for the first cut is acceptable, but training-time dropout (if ever enabled) must
  be device-side.
- **`erf` vs `tanh`-approx GELU.** Gemma's reference uses exact GELU on TPU and tanh-approx on
  some GPUs. Plan to expose both with a config knob; the IR primitive of choice is `erf` because
  it lowers more directly.

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

- [`crates/ryft-core/src/types/data.rs:759`](crates/ryft-core/src/types/data.rs:759) —
  the `DataType` enum entries for `F4E2M1FN`, `F8E4M3FN`, `F8E5M2`, `F8E8M0FNU`, and friends.
- [`crates/ryft-core/src/tracing_v2/operations/`](crates/ryft-core/src/tracing_v2/operations/) —
  existing operation types used as the template for new primitives in §1.
- [`crates/ryft-xla/src/experimental/lowering.rs`](crates/ryft-xla/src/experimental/lowering.rs) —
  existing StableHLO lowering rules that the new ops in §1 will extend.
