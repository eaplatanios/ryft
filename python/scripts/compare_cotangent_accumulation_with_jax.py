"""Record JAX cotangent demand and indexed buffer accumulation in the pinned environment.

Run with `python/.venv/bin/python python/scripts/compare_cotangent_accumulation_with_jax.py`.
The output distinguishes staged-program evidence from execution timing: this probe makes
no memory-traffic or performance claim. Buffer initialization is outside the sparse replay.
"""

import json
import platform

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np


def primitive_counts(program):
    """Count equations recursively, including nested call programs."""
    counts = {}

    def visit(value):
        if hasattr(value, "jaxpr"):
            visit(value.jaxpr)
        elif hasattr(value, "eqns"):
            for equation in value.eqns:
                name = equation.primitive.name
                counts[name] = counts.get(name, 0) + 1
                for parameter in equation.params.values():
                    visit(parameter)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    visit(program)
    return counts


def main():
    """Check numerical results and print versioned baseline programs as JSON."""
    result = {"python": platform.python_version(), "jax": jax.__version__, "jaxlib": jaxlib.__version__}
    matrix = jnp.ones((4, 4), dtype=jnp.float32)
    vector = jnp.arange(4, dtype=jnp.float32)
    seed = jnp.ones(4, dtype=jnp.float32)
    _, pullback = jax.vjp(lambda weights, data: weights @ data, matrix, vector)
    both = jax.make_jaxpr(pullback)(seed)
    result["both_gradients"] = {"program": str(both), "primitives": primitive_counts(both)}
    if not hasattr(pullback, "with_refs") or not hasattr(jax, "ad"):
        result["destination_api"] = "unavailable in the installed version"
        print(json.dumps(result, indent=2))
        return

    _, pullback = jax.vjp(lambda weights, data: weights @ data, matrix, vector)
    selected = pullback.with_refs(jax.ad.GradValue(), jax.ad.DontWant())
    matrix_gradient, _ = selected(seed)
    _, pullback = jax.vjp(lambda weights, data: weights @ data, matrix, vector)
    only_matrix = jax.make_jaxpr(pullback.with_refs(jax.ad.GradValue(), jax.ad.DontWant()))(seed)
    np.testing.assert_array_equal(matrix_gradient, np.broadcast_to(np.arange(4, dtype=np.float32), (4, 4)))
    result["only_matrix_gradient"] = {"program": str(only_matrix), "primitives": primitive_counts(only_matrix)}

    # A nonzero initial buffer distinguishes additive updates from overwriting gradients.
    values = jnp.arange(8, dtype=jnp.float32)
    buffer = jax.ref.new_ref(jnp.full_like(values, 2))

    @jax.jit
    def accumulate(destination, index):
        """Add one indexed-read pullback into existing caller storage."""
        _, backward = jax.vjp(lambda array, position: array[position], values, index)
        backward.with_refs(destination, jax.ad.DontWant())(jnp.float32(1))

    sparse = jax.make_jaxpr(accumulate)(buffer, jnp.int32(3))
    for index in (3, 5, 3):
        accumulate(buffer, jnp.int32(index))
    actual = jax.ref.get(buffer)
    expected = np.full(8, 2, dtype=np.float32)
    expected[3] += 2
    expected[5] += 1
    np.testing.assert_array_equal(actual, expected)
    result["indexed_accumulation"] = {
        "program": str(sparse),
        "primitives": primitive_counts(sparse),
        "initial_buffer": [2] * 8,
        "indices": [3, 5, 3],
        "final_buffer": np.asarray(actual).tolist(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
