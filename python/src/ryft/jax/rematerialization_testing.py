"""Rematerialization (i.e., gradient checkpointing) differential cases, with JAX builders and analytic references.

Each case differentiates one function wrapped in `jax.checkpoint` with a residual policy from
`jax.checkpoint_policies`, which the Rust emitter mirrors with `ryft_core::rematerialize` (refer to
`crates/ryft-xla/src/bin/differential_testing/rematerialization.rs`). The functions take `x: f32[2, 3]` and
`w: f32[3, 2]` (and, in the scan cases, stacked `layers: f32[2, 2, 2]`) and only multiply and add small integers, so
every value is exact in `f32` and both frameworks must agree exactly. Each case records its output, its Jacobian-vector
product, its input gradients, and the values that reverse mode differentiation saves. JAX reports the saved values
through the private `jax._src.ad_checkpoint.saved_residuals`, whose source descriptions differ from Ryft's, so only
their count, element counts, and bit widths are compared, sorted by element count and then by bit width.

The references compute the values analytically with NumPy, independently of both frameworks, and record the saved
values that JAX 0.12.0.1 reports. Both StableHLO modules lower the staged output and gradients, whose recomputation
must sit behind an optimization barrier.

This module intentionally does not import JAX at module load time.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ryft.jax.differential_testing import SCHEMA, DifferentialObservation


STABLEHLO_PATTERNS = ("stablehlo.optimization_barrier", "stablehlo.dot_general")
X = ((1.0, 2.0, -1.0), (0.0, 1.0, 2.0))
W = ((1.0, -1.0), (2.0, 0.0), (-1.0, 1.0))
X_TANGENT = ((1.0, 0.0, 0.0), (0.0, -1.0, 1.0))
W_TANGENT = ((0.0, 1.0), (1.0, 0.0), (0.0, 0.0))
NEGATED_X = ((-1.0, -2.0, 1.0), (0.0, -1.0, -2.0))
LAYERS = (((1.0, 0.0), (1.0, -1.0)), ((0.0, 1.0), (1.0, 1.0)))
LAYERS_TANGENT = (((0.0, 1.0), (0.0, 0.0)), ((1.0, 0.0), (0.0, -1.0)))
VECTOR_SEED = (1.0, -1.0)
MATRIX_SEED = ((1.0, -1.0), (2.0, 1.0))


@dataclass(frozen=True)
class RematerializationCase:
    """One rematerialized function together with its analytic reference and the residuals that JAX saves for it.

    `function` returns the checkpointed JAX function given the `jax` and `jax.numpy` modules. `reference` returns the
    output, the Jacobian-vector product, and the input gradients given `numpy` and the `float64` inputs, tangents, and
    output cotangent. `residuals` lists the `(element_count, bit_width)` pairs of the values that JAX 0.12.0.1 saves.
    A case with `negated_evaluation` is also evaluated at `-x`, under names that start with `false_branch_`.
    """

    case_id: str
    function: Callable[[Any, Any], Callable[..., Any]]
    reference: Callable[..., tuple[Any, Any, tuple[Any, ...]]]
    residuals: tuple[tuple[int, int], ...]
    input_names: tuple[str, ...] = ("x", "w")
    seed: Any = VECTOR_SEED
    negated_evaluation: bool = False


def residual_observations(residuals: Sequence[tuple[int, int]]) -> dict[str, tuple[tuple[float, ...], ...]]:
    """Encodes saved values as numeric observations of their count, element counts, and bit widths.

    # Parameters

      - `residuals`: `(element_count, bit_width)` pairs of the saved values, in any order.
    """

    facts = sorted((int(element_count), int(bit_width)) for element_count, bit_width in residuals)
    return {
        "residual_count": ((float(len(facts)),),),
        "residual_element_counts": (tuple(float(element_count) for element_count, _ in facts),),
        "residual_bit_widths": (tuple(float(bit_width) for _, bit_width in facts),),
    }


def jax_residuals(saved: Sequence[tuple[Any, str]]) -> tuple[tuple[int, int], ...]:
    """Returns the `(element_count, bit_width)` pairs of the `(aval, source)` pairs that `saved_residuals` returns."""

    facts = []
    for aval, _ in saved:
        element_count = 1
        for extent in aval.shape:
            element_count *= int(extent)
        facts.append((element_count, aval.dtype.itemsize * 8))
    return tuple(facts)


def _flatten(value: Any, numpy: Any) -> tuple[tuple[float, ...], ...]:
    """Returns the flattened values of one array as a single-execution observation."""

    return (tuple(float(element) for element in numpy.asarray(value).reshape(-1)),)


def _inputs(case: RematerializationCase, numpy: Any) -> tuple[tuple[Any, ...], tuple[Any, ...], Any]:
    """Returns the `float64` inputs, tangents, and output cotangent of `case`."""

    inputs = (X, W, LAYERS)[: len(case.input_names)]
    tangents = (X_TANGENT, W_TANGENT, LAYERS_TANGENT)[: len(case.input_names)]
    return (
        tuple(numpy.array(value, dtype=numpy.float64) for value in inputs),
        tuple(numpy.array(value, dtype=numpy.float64) for value in tangents),
        numpy.array(case.seed, dtype=numpy.float64),
    )


def _evaluations(case: RematerializationCase, inputs: tuple[Any, ...], numpy: Any) -> tuple[tuple[str, tuple], ...]:
    """Returns the name prefixes and inputs at which `case` is evaluated."""

    evaluations = (("", inputs),)
    if case.negated_evaluation:
        evaluations += (("false_branch_", (numpy.array(NEGATED_X, dtype=inputs[0].dtype), *inputs[1:])),)
    return evaluations


def rematerialization_reference(case: RematerializationCase) -> dict[str, tuple[tuple[float, ...], ...]]:
    """Returns the analytic observations of `case`, computed in `float64` with NumPy."""

    import numpy

    inputs, tangents, seed = _inputs(case, numpy)
    observations = residual_observations(case.residuals)
    for prefix, evaluation_inputs in _evaluations(case, inputs, numpy):
        output, tangent, gradients = case.reference(numpy, evaluation_inputs, tangents, seed)
        observations[f"{prefix}output"] = _flatten(output, numpy)
        observations[f"{prefix}jvp"] = _flatten(tangent, numpy)
        for name, gradient in zip(case.input_names, gradients, strict=True):
            observations[f"{prefix}{name}_gradient"] = _flatten(gradient, numpy)
    return observations


def build_rematerialization_jax(
    case: RematerializationCase, jax: Any, jax_numpy: Any, numpy: Any,
) -> DifferentialObservation:
    """Evaluates `case` with the pinned JAX and records its values, saved values, and staged gradient module."""

    from jax._src.ad_checkpoint import saved_residuals

    function = case.function(jax, jax_numpy)
    inputs, tangents, seed = _inputs(case, numpy)
    inputs = tuple(jax_numpy.asarray(value, dtype=jax_numpy.float32) for value in inputs)
    tangents = tuple(jax_numpy.asarray(value, dtype=jax_numpy.float32) for value in tangents)
    seed = jax_numpy.asarray(seed, dtype=jax_numpy.float32)
    observations = residual_observations(jax_residuals(saved_residuals(function, *inputs)))
    for prefix, evaluation_inputs in _evaluations(case, inputs, jax_numpy):
        output, tangent = jax.jvp(function, evaluation_inputs, tangents)
        _, pullback = jax.vjp(function, *evaluation_inputs)
        observations[f"{prefix}output"] = _flatten(output, numpy)
        observations[f"{prefix}jvp"] = _flatten(tangent, numpy)
        for name, gradient in zip(case.input_names, pullback(seed), strict=True):
            observations[f"{prefix}{name}_gradient"] = _flatten(gradient, numpy)

    def staged(*arguments: Any) -> tuple[Any, ...]:
        output, pullback = jax.vjp(function, *arguments[:-1])
        return (output, *pullback(arguments[-1]))

    return DifferentialObservation(
        schema=SCHEMA,
        case_id=case.case_id,
        observations=observations,
        stablehlo=str(jax.jit(staged).lower(*inputs, seed).compiler_ir("stablehlo")),
    )


def _policy_body(jax: Any, jax_numpy: Any) -> Callable[[Any, Any], Any]:
    """Returns `(x, w) ↦ b ⊙ b` for `y = x · w` named `dot`, `z = y ⊙ y` named `square`, and the rowwise batched dot
    product `b` of `z` and `y` named `batched`."""

    from jax.ad_checkpoint import checkpoint_name as name

    def body(x: Any, w: Any) -> Any:
        y = name(jax_numpy.dot(x, w), "dot")
        z = name(y * y, "square")
        b = name(jax_numpy.einsum("ij,ij->i", z, y), "batched")
        return b * b

    return body


def _policy_reference(numpy: Any, inputs: tuple[Any, ...], tangents: tuple[Any, ...], seed: Any) -> tuple:
    """Differentiates `b²` for `b = Σⱼ yᵢⱼ³` and `y = x · w` analytically."""

    (x, w), (x_tangent, w_tangent) = inputs, tangents
    y = x @ w
    y_tangent = x_tangent @ w + x @ w_tangent
    b = (y**3).sum(axis=1)
    y_cotangent = 3 * y**2 * (2 * b * seed)[:, None]
    return b**2, 2 * b * (3 * y**2 * y_tangent).sum(axis=1), (y_cotangent @ w.T, x.T @ y_cotangent)


def _fourth_power_reference(numpy: Any, inputs: tuple[Any, ...], tangents: tuple[Any, ...], seed: Any) -> tuple:
    """Differentiates `y⁴` for `y = x · w` analytically."""

    (x, w), (x_tangent, w_tangent) = inputs, tangents
    y = x @ w
    y_cotangent = 4 * y**3 * seed
    return y**4, 4 * y**3 * (x_tangent @ w + x @ w_tangent), (y_cotangent @ w.T, x.T @ y_cotangent)


def _condition_reference(numpy: Any, inputs: tuple[Any, ...], tangents: tuple[Any, ...], seed: Any) -> tuple:
    """Differentiates `(y · y)²` when `Σ x > 0` and `y³` otherwise, for `y = x · w`, analytically."""

    (x, w), (x_tangent, w_tangent) = inputs, tangents
    y = x @ w
    y_tangent = x_tangent @ w + x @ w_tangent
    if x.sum() > 0:
        t = y @ y
        t_cotangent = 2 * t * seed
        y_cotangent = t_cotangent @ y.T + y.T @ t_cotangent
        output, tangent = t**2, 2 * t * (y_tangent @ y + y @ y_tangent)
    else:
        y_cotangent = 3 * y**2 * seed
        output, tangent = y**3, 3 * y**2 * y_tangent
    return output, tangent, (y_cotangent @ w.T, x.T @ y_cotangent)


def _scan_reference(numpy: Any, inputs: tuple[Any, ...], tangents: tuple[Any, ...], seed: Any) -> tuple:
    """Differentiates the scan `hₖ₊₁ = (hₖ · layerₖ)²` from `h₀ = x · w` analytically."""

    (x, w, layers), (x_tangent, w_tangent, layers_tangent) = inputs, tangents
    hidden = [x @ w]
    hidden_tangent = x_tangent @ w + x @ w_tangent
    for layer, layer_tangent in zip(layers, layers_tangent, strict=True):
        t = hidden[-1] @ layer
        hidden_tangent = 2 * t * (hidden_tangent @ layer + hidden[-1] @ layer_tangent)
        hidden.append(t**2)
    hidden_cotangent = seed
    layers_cotangent = numpy.zeros_like(layers)
    for index in reversed(range(len(layers))):
        t_cotangent = 2 * (hidden[index] @ layers[index]) * hidden_cotangent
        layers_cotangent[index] = hidden[index].T @ t_cotangent
        hidden_cotangent = t_cotangent @ layers[index].T
    return hidden[-1], hidden_tangent, (hidden_cotangent @ w.T, x.T @ hidden_cotangent, layers_cotangent)


def _known_region_reference(
    numpy: Any, inputs: tuple[Any, ...], tangents: tuple[Any, ...], seed: Any, s: Any,
) -> tuple:
    """Differentiates `(x · w) ⊙ s` analytically for a matrix `s` that the cases compute from inputs without
    gradients, which therefore receive zero gradients."""

    x, w = inputs[:2]
    x_tangent, w_tangent = tangents[:2]
    y_cotangent = seed * s
    gradients = (y_cotangent @ w.T, x.T @ y_cotangent, *(numpy.zeros_like(value) for value in inputs[2:]))
    return (x @ w) * s, (x_tangent @ w + x @ w_tangent) * s, gradients


def _known_condition_reference(numpy: Any, inputs: tuple[Any, ...], tangents: tuple[Any, ...], seed: Any) -> tuple:
    """Differentiates `(x · w) ⊙ s` for `s = 3 wᵀ w` when `Σ x > 0` and `s = 6 wᵀ w` otherwise analytically."""

    x, w = inputs
    s = (3 if x.sum() > 0 else 6) * (w.T @ w)
    return _known_region_reference(numpy, inputs, tangents, seed, s)


def _known_scan_reference(numpy: Any, inputs: tuple[Any, ...], tangents: tuple[Any, ...], seed: Any) -> tuple:
    """Differentiates `(x · w) ⊙ s` for `s = 3 (Σₖ layerₖ + Σₖ layerₖ · layerₖ)` analytically."""

    layers = inputs[2]
    s = 3 * (layers.sum(axis=0) + sum(layer @ layer for layer in layers))
    return _known_region_reference(numpy, inputs, tangents, seed, s)


def _checkpointed_policy_body(policy: Callable[[Any], Any]) -> Callable[[Any, Any], Callable[..., Any]]:
    """Returns a `function` field that checkpoints `_policy_body` with the policy that `policy` selects."""

    def function(jax: Any, jax_numpy: Any) -> Callable[..., Any]:
        return jax.checkpoint(_policy_body(jax, jax_numpy), policy=policy(jax.checkpoint_policies))

    return function


def _call_function(jax: Any, jax_numpy: Any) -> Callable[..., Any]:
    """Returns `z ⊙ z` for `z = y ⊙ y` computed by a `jax.jit` function of `y = x · w`, under `dots_saveable`."""

    @jax.jit
    def call(x: Any, w: Any) -> Any:
        y = jax_numpy.dot(x, w)
        return y * y

    def body(x: Any, w: Any) -> Any:
        z = call(x, w)
        return z * z

    return jax.checkpoint(body, policy=jax.checkpoint_policies.dots_saveable)


def _condition_function(jax: Any, jax_numpy: Any) -> Callable[..., Any]:
    """Returns a `lax.cond` on `Σ x > 0` over `y = x · w` whose `true` branch alone computes a dot product, under
    `dots_saveable`."""

    def true_branch(y: Any) -> Any:
        t = jax_numpy.dot(y, y)
        return t * t

    def body(x: Any, w: Any) -> Any:
        return jax.lax.cond(jax_numpy.sum(x) > 0, true_branch, lambda y: y * y * y, jax_numpy.dot(x, w))

    return jax.checkpoint(body, policy=jax.checkpoint_policies.dots_saveable)


def _scan_function(jax: Any, jax_numpy: Any) -> Callable[..., Any]:
    """Returns a `lax.scan` of `h ↦ (h · layer)²` over `layers` from `x · w`, under `dots_saveable`."""

    def step(hidden: Any, layer: Any) -> tuple[Any, None]:
        t = jax_numpy.dot(hidden, layer)
        return t * t, None

    def body(x: Any, w: Any, layers: Any) -> Any:
        return jax.lax.scan(step, jax_numpy.dot(x, w), layers)[0]

    return jax.checkpoint(body, policy=jax.checkpoint_policies.dots_saveable)


def _known_condition_function(jax: Any, jax_numpy: Any) -> Callable[..., Any]:
    """Returns `(x · w) ⊙ Σₖ eₖ` under `dots_saveable`, where a `lax.cond` on `Σ x > 0` whose inputs carry no gradients
    expands the compact `wᵀ w` into `e = broadcast(wᵀ w)` of shape `[3, 2, 2]` or returns `e + e`."""

    def expand(w: Any) -> Any:
        return jax_numpy.broadcast_to(w.T @ w, (3, 2, 2))

    def double(w: Any) -> Any:
        e = expand(w)
        return e + e

    def body(x: Any, w: Any) -> Any:
        known_x, known_w = jax.lax.stop_gradient(x), jax.lax.stop_gradient(w)
        e = jax.lax.cond(jax_numpy.sum(known_x) > 0, expand, double, known_w)
        return jax_numpy.dot(x, w) * e.sum(axis=0)

    return jax.checkpoint(body, policy=jax.checkpoint_policies.dots_saveable)


def _known_scan_function(jax: Any, jax_numpy: Any) -> Callable[..., Any]:
    """Returns `(x · w) ⊙ Σₖ hₖ` under `dots_saveable`, where a `lax.scan` over `layers` whose inputs carry no gradients
    adds the expansion `broadcast(layer · layer)` of shape `[3, 2, 2]` to a carry that starts from
    `broadcast(Σ layers)`."""

    def step(hidden: Any, layer: Any) -> tuple[Any, None]:
        return hidden + jax_numpy.broadcast_to(layer @ layer, (3, 2, 2)), None

    def body(x: Any, w: Any, layers: Any) -> Any:
        known_layers = jax.lax.stop_gradient(layers)
        initial = jax_numpy.broadcast_to(known_layers.sum(axis=0), (3, 2, 2))
        hidden = jax.lax.scan(step, initial, known_layers)[0]
        return jax_numpy.dot(x, w) * hidden.sum(axis=0)

    return jax.checkpoint(body, policy=jax.checkpoint_policies.dots_saveable)


def _nested_function(jax: Any, jax_numpy: Any) -> Callable[..., Any]:
    """Returns `v ⊙ v` under `save_only_these_names("square")` for `v = y ⊙ y` named `square` and `y = x · w` named
    `dot`, computed by an inner checkpoint under `dots_saveable`."""

    from jax.ad_checkpoint import checkpoint_name as name
    policies = jax.checkpoint_policies

    def inner(x: Any, w: Any) -> Any:
        y = name(jax_numpy.dot(x, w), "dot")
        return name(y * y, "square")

    inner = jax.checkpoint(inner, policy=policies.dots_saveable)

    def outer(x: Any, w: Any) -> Any:
        v = inner(x, w)
        return v * v

    return jax.checkpoint(outer, policy=policies.save_only_these_names("square"))


REMATERIALIZATION_CASES = (
    RematerializationCase(
        "rematerialization_nothing_saveable",
        _checkpointed_policy_body(lambda policies: policies.nothing_saveable),
        _policy_reference,
        ((6, 32), (6, 32)),
    ),
    RematerializationCase(
        "rematerialization_everything_saveable",
        _checkpointed_policy_body(lambda policies: policies.everything_saveable),
        _policy_reference,
        ((2, 32), (4, 32), (4, 32), (6, 32), (6, 32)),
    ),
    RematerializationCase(
        "rematerialization_dots_saveable",
        _checkpointed_policy_body(lambda policies: policies.dots_saveable),
        _policy_reference,
        ((2, 32), (4, 32), (6, 32), (6, 32)),
    ),
    RematerializationCase(
        "rematerialization_dots_with_no_batch_dims_saveable",
        _checkpointed_policy_body(lambda policies: policies.dots_with_no_batch_dims_saveable),
        _policy_reference,
        ((4, 32), (6, 32), (6, 32)),
    ),
    RematerializationCase(
        "rematerialization_save_only_these_names",
        _checkpointed_policy_body(lambda policies: policies.save_only_these_names("square")),
        _policy_reference,
        ((4, 32), (6, 32), (6, 32)),
    ),
    RematerializationCase(
        "rematerialization_save_any_names_but_these",
        _checkpointed_policy_body(lambda policies: policies.save_any_names_but_these("square")),
        _policy_reference,
        ((2, 32), (4, 32), (6, 32), (6, 32)),
    ),
    RematerializationCase(
        "rematerialization_save_from_both_policies",
        _checkpointed_policy_body(
            lambda policies: policies.save_from_both_policies(
                policies.dots_with_no_batch_dims_saveable, policies.save_only_these_names("square"),
            )
        ),
        _policy_reference,
        ((4, 32), (4, 32), (6, 32), (6, 32)),
    ),
    RematerializationCase(
        "rematerialization_save_and_offload_only_these_names",
        _checkpointed_policy_body(
            lambda policies: policies.save_and_offload_only_these_names(
                names_which_can_be_saved=["dot"],
                names_which_can_be_offloaded=["square"],
                offload_src="device",
                offload_dst="pinned_host",
            )
        ),
        _policy_reference,
        ((4, 32), (4, 32), (6, 32), (6, 32)),
    ),
    RematerializationCase(
        "rematerialization_dots_saveable_call",
        _call_function,
        _fourth_power_reference,
        ((4, 32), (6, 32), (6, 32)),
        seed=MATRIX_SEED,
    ),
    RematerializationCase(
        "rematerialization_dots_saveable_condition",
        _condition_function,
        _condition_reference,
        ((4, 32), (4, 32), (6, 32), (6, 32)),
        seed=MATRIX_SEED,
        negated_evaluation=True,
    ),
    RematerializationCase(
        "rematerialization_dots_saveable_scan",
        _scan_function,
        _scan_reference,
        ((4, 32), (6, 32), (6, 32), (8, 32), (8, 32)),
        input_names=("x", "w", "layers"),
        seed=MATRIX_SEED,
    ),
    RematerializationCase(
        "rematerialization_dots_saveable_known_condition",
        _known_condition_function,
        _known_condition_reference,
        ((4, 32), (6, 32), (6, 32)),
        seed=MATRIX_SEED,
        negated_evaluation=True,
    ),
    RematerializationCase(
        "rematerialization_dots_saveable_known_scan",
        _known_scan_function,
        _known_scan_reference,
        ((6, 32), (6, 32), (8, 32), (8, 32)),
        input_names=("x", "w", "layers"),
        seed=MATRIX_SEED,
    ),
    RematerializationCase(
        "rematerialization_nested_policies",
        _nested_function,
        _fourth_power_reference,
        ((4, 32), (6, 32), (6, 32)),
        seed=MATRIX_SEED,
    ),
)


__all__ = [
    "REMATERIALIZATION_CASES",
    "RematerializationCase",
    "STABLEHLO_PATTERNS",
    "build_rematerialization_jax",
    "jax_residuals",
    "rematerialization_reference",
    "residual_observations",
]
