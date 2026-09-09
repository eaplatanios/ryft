"""Compare JVP and repeated linearization of a square function with local reference state.

Run with the repository's pinned environment:
    python/.venv/bin/python python/scripts/compare_reference_linearization_with_jax.py

The matching Ryft regression is
`test_custom_jvp_linearization_keeps_tangent_state_fresh_and_hoists_coefficients`.
Results are reported rather than asserting a particular JAX failure, so the same
probe can also identify when a later JAX version fixes the discrepancy.
"""

import json
import platform

import jax
import jax.numpy as jnp
import jaxlib


def custom_square(mode):
    """Build equivalent square JVP rules with different tangent implementations."""
    @jax.custom_jvp
    def square(value):
        return value * value

    @square.defjvp
    def square_jvp(primals, tangents):
        value, = primals
        tangent, = tangents
        coefficient = value + value
        if mode == "pure":
            derivative = coefficient * tangent
        else:
            # A tangent-dependent initializer is a control: the mathematical value
            # is still zero for these finite inputs, but partial evaluation must stage it.
            initial = tangent * 0 if mode == "tangent_dependent_zero" else jnp.zeros_like(value)
            reference = jax.ref.new_ref(initial)
            jax.ref.addupdate(reference, (), coefficient * tangent)
            derivative = jax.ref.freeze(reference) if mode == "freeze" else jax.ref.get(reference)
        return value * value, derivative

    return square


def ordinary_square(value):
    """Supply an ordinary differentiation control without a custom rule."""
    return value * value


def ordinary_reference_square(value):
    """Check that references work under ordinary, non-custom differentiation."""
    reference = jax.ref.new_ref(value * value)
    return jax.ref.freeze(reference)


def measure(function):
    """Record direct JVPs, repeated pushforwards, their staged program, and a gradient."""
    value = jnp.array(3.0, dtype=jnp.float32)
    tangents = [jnp.array(tangent, dtype=value.dtype) for tangent in (2.0, 5.0, 2.0)]
    result = {}
    # Convert device results to Python scalars to await completion before recording them.
    direct = [jax.jvp(function, (value,), (tangent,)) for tangent in tangents]
    result["jvp_primals"] = [float(primal) for primal, _ in direct]
    result["jvp_tangents"] = [float(tangent) for _, tangent in direct]
    try:
        primal, pushforward = jax.linearize(function, value)
        result["linearize_primal"] = float(primal)
        result["pushforward_tangents"] = [float(pushforward(tangent)) for tangent in tangents]
        result["matches_expected"] = result["pushforward_tangents"] == [12.0, 30.0, 12.0]
        result["pushforward_jaxpr"] = str(jax.make_jaxpr(pushforward)(tangents[0]))
    except Exception as error:
        result["linearize_error"] = {"type": type(error).__name__, "message": str(error)}
    try:
        result["gradient"] = float(jax.grad(function)(value))
    except Exception as error:
        result["gradient_error"] = {"type": type(error).__name__, "message": str(error)}
    return result


def main():
    """Print versioned results for the reproducer and its controls as JSON."""
    functions = {
        "ordinary_pure": ordinary_square,
        "ordinary_reference": ordinary_reference_square,
        "custom_pure": custom_square("pure"),
        "custom_reference_read": custom_square("read"),
        "custom_reference_freeze": custom_square("freeze"),
        "custom_reference_tangent_dependent_zero": custom_square("tangent_dependent_zero"),
    }
    results = {name: measure(function) for name, function in functions.items()}
    print(json.dumps({
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "backend": jax.default_backend(),
        },
        "input": 3.0,
        "tangents": [2.0, 5.0, 2.0],
        "expected_primal": 9.0,
        "expected_tangents": [12.0, 30.0, 12.0],
        "cases": results,
    }, indent=2))


if __name__ == "__main__":
    main()
