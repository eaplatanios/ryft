"""Check scan/while carry convergence, repeated linearization, and reference ordering.

Run with python/.venv/bin/python python/scripts/compare_loop_carries_with_jax.py.
The direct partition probes intentionally use pinned JAX internals to match Ryft's
Program::partition tests; the execution and differentiation probes use public APIs.
"""

import json
import platform

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from jax._src import core
from jax._src.interpreters import partial_eval


def scan_chain(first, second, third, items):
    """Shift a value through three carry positions, recording the oldest value."""
    def body(carry, item):
        return (carry[1], carry[2], item), carry[0]
    carry, history = jax.lax.scan(body, (first, second, third), items)
    return *carry, history


def while_chain(counter, first, second, source):
    """Propagate an initially unknown source through two initially known carries."""
    return jax.lax.while_loop(
        lambda state: state[0] > 0,
        lambda state: (state[0] - 1, state[2], state[3], state[3]),
        (counter, first, second, source),
    )


def scan_reference(items, explicit_carry=False):
    """Reset then update each iteration; splitting resets from updates changes the result."""
    reference = jax.ref.new_ref(jnp.float32(2))
    def body(carry, item):
        current = carry if explicit_carry else reference
        jax.ref.set(current, (), jnp.float32(1))
        jax.ref.addupdate(current, (), item)
        return carry, None
    jax.lax.scan(body, reference if explicit_carry else (), items)
    return jax.ref.freeze(reference)


def while_reference(update, explicit_carry=False):
    """Accumulate three updates in a reference carried explicitly or captured by the body."""
    reference = jax.ref.new_ref(jnp.float32(2))
    if explicit_carry:
        def body(state):
            counter, current = state
            jax.ref.addupdate(current, (), update)
            return counter - 1, current
        jax.lax.while_loop(lambda state: state[0] > 0, body, (jnp.float32(3), reference))
    else:
        def body(counter):
            jax.ref.addupdate(reference, (), update)
            return counter - 1
        jax.lax.while_loop(lambda counter: counter > 0, body, jnp.float32(3))
    return jax.ref.freeze(reference)


def values(tree):
    """Await array results and convert their structure to JSON-compatible values."""
    return json.loads(json.dumps(jax.tree.map(lambda value: np.asarray(value).tolist(), tree)))


def record(function):
    """Record either a completed numerical result or a concrete exception."""
    try:
        return {"value": values(function())}
    except Exception as error:
        return {"error": {"type": type(error).__name__, "message": str(error)}}


def execution_case(function, arguments, tangent_sets, expected_primal, expected_tangents):
    """Compare eager/JIT execution, direct JVP, and repeated pushforwards with an oracle."""
    result = {
        "expected_primal": expected_primal,
        "expected_tangents": expected_tangents,
        "eager": record(lambda: function(*arguments)),
        "jit": record(lambda: jax.jit(function)(*arguments)),
        "jvp_tangents": record(lambda: [jax.jvp(function, arguments, tangents)[1] for tangents in tangent_sets]),
    }
    try:
        primal, pushforward = jax.linearize(function, *arguments)
        result["linearize_primal"] = values(primal)
        result["pushforward_tangents"] = values([pushforward(*tangents) for tangents in tangent_sets])
        result["matches_expected"] = (
            result["eager"] == {"value": expected_primal}
            and result["jit"] == {"value": expected_primal}
            and result["jvp_tangents"] == {"value": expected_tangents}
            and result["linearize_primal"] == expected_primal
            and result["pushforward_tangents"] == expected_tangents
        )
    except Exception as error:
        result["linearize_error"] = {"type": type(error).__name__, "message": str(error)}
        result["matches_expected"] = False
    return result


def partition_case(function, arguments, unknowns, expected, expected_unknowns):
    """Execute both JAX partial-evaluation halves and reassemble the original outputs."""
    source = jax.make_jaxpr(function)(*arguments)
    known, residual, output_unknowns, _ = partial_eval.partial_eval_jaxpr_nounits(source, unknowns, False)
    known_arguments = [argument for argument, unknown in zip(arguments, unknowns) if not unknown]
    unknown_arguments = [argument for argument, unknown in zip(arguments, unknowns) if unknown]
    known_outputs = core.eval_jaxpr(known.jaxpr, known.consts, *known_arguments)
    # Known results precede saved intermediate values, which become the residual program's leading inputs.
    known_count = output_unknowns.count(False)
    residual_outputs = core.eval_jaxpr(residual.jaxpr, residual.consts, *known_outputs[known_count:], *unknown_arguments)
    known_values, residual_values = iter(known_outputs[:known_count]), iter(residual_outputs)
    result = values([next(residual_values) if unknown else next(known_values) for unknown in output_unknowns])
    return {
        "input_unknowns": unknowns,
        "output_unknowns": output_unknowns,
        "reconstructed_outputs": result,
        "matches_expected": result == expected and output_unknowns == expected_unknowns,
        "known_jaxpr": str(known),
        "residual_jaxpr": str(residual),
    }


def main():
    """Print a versioned comparison with exact numerical oracles and explicit API rejections."""
    scalar = jnp.float32
    scan_arguments = (scalar(0), scalar(0), scalar(10), jnp.array([1., 2., 3., 4.], dtype=jnp.float32))
    first_scan_tangent = (scalar(0), scalar(0), scalar(2), jnp.ones(4, dtype=jnp.float32))
    second_scan_tangent = (scalar(0), scalar(0), scalar(5), jnp.array([2., 3., 4., 5.], dtype=jnp.float32))
    while_arguments = (scalar(3), scalar(1), scalar(1), scalar(7))
    while_tangents = [(scalar(0), scalar(0), scalar(0), scalar(tangent)) for tangent in (2, 5, 2)]
    cases = {
        "scan_carry_chain": execution_case(scan_chain, scan_arguments,
            [first_scan_tangent, second_scan_tangent, first_scan_tangent],
            [2., 3., 4., [0., 0., 10., 1.]],
            [[1., 1., 1., [0., 0., 2., 1.]], [3., 4., 5., [0., 0., 5., 2.]], [1., 1., 1., [0., 0., 2., 1.]]]),
        "while_carry_chain": execution_case(while_chain, while_arguments, while_tangents,
            [0., 7., 7., 7.], [[0., 2., 2., 2.], [0., 5., 5., 5.], [0., 2., 2., 2.]]),
        "scan_zero_iterations": execution_case(scan_chain, (*scan_arguments[:3], jnp.empty(0, dtype=jnp.float32)),
            [(scalar(0), scalar(0), scalar(2), jnp.empty(0, dtype=jnp.float32))],
            [0., 0., 10., []], [[0., 0., 2., []]]),
        "while_zero_iterations": execution_case(while_chain, (scalar(0), *while_arguments[1:]), while_tangents,
            [0., 1., 1., 7.], [[0., 0., 0., 2.], [0., 0., 0., 5.], [0., 0., 0., 2.]]),
        "scan_captured_reference": execution_case(scan_reference, (jnp.array([1., 2., 3.], dtype=jnp.float32),),
            [(jnp.array([2., 5., 7.], dtype=jnp.float32),), (jnp.ones(3, dtype=jnp.float32),), (jnp.array([2., 5., 7.], dtype=jnp.float32),)],
            4., [7., 1., 7.]),
        "while_captured_reference": execution_case(while_reference, (scalar(1),),
            [(scalar(2),), (scalar(5),), (scalar(2),)], 5., [6., 15., 6.]),
    }
    partitions = {
        "scan": partition_case(scan_chain, scan_arguments, [False, False, True, True],
            [2., 3., 4., [0., 0., 10., 1.]], [True, True, True, True]),
        "while": partition_case(while_chain, while_arguments, [False, False, False, True],
            [0., 7., 7., 7.], [False, True, True, True]),
    }
    print(json.dumps({
        "environment": {"python": platform.python_version(), "jax": jax.__version__, "jaxlib": jaxlib.__version__, "backend": jax.default_backend()},
        "execution": cases,
        "partial_evaluation": partitions,
        "explicit_reference_carry": {
            "scan": record(lambda: scan_reference(jnp.array([1., 2., 3.], dtype=jnp.float32), True)),
            "while": record(lambda: while_reference(scalar(1), True)),
        },
        "ordinary_unbounded_while_reverse_mode": record(lambda: jax.grad(lambda source: while_chain(scalar(3), scalar(1), scalar(1), source)[1])(scalar(7))),
    }, indent=2))
    if not all(case["matches_expected"] for case in (*cases.values(), *partitions.values())):
        raise SystemExit("a numerical or knownness comparison failed")


if __name__ == "__main__":
    main()
