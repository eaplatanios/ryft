"""Probe the pinned, internal JAX accumulator-taking high-level primitive interface.

Run with `python/.venv/bin/python python/scripts/compare_hijax_backward_rules_with_jax.py`.
This uses JAX 0.10.0's private VJPHiPrimitive, not upstream's evolving HiPrim API.
Python callback counters demonstrate tracing behavior, not performance measurements.
The indexed probe uses in-bounds indices and nonzero output seeds; it is not a
complete general-purpose primitive implementation.
"""

import json
import platform

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from jax._src import core, hijax
from jax._src.interpreters import ad

from compare_cotangent_accumulation_with_jax import primitive_counts


CALLS = []


class IndexedRead(hijax.VJPHiPrimitive):
    """One backward definition supporting ignored, value, and buffer cotangents."""

    def __init__(self):
        self.in_avals = (core.ShapedArray((8,), np.dtype('float32')),
                         core.ShapedArray((), np.dtype('int32')))
        self.out_aval = core.ShapedArray((), np.dtype('float32'))
        self.params = {}
        super().__init__()

    def expand(self, values, index):
        return values[index]

    def vjp_fwd(self, nonzeros, values, index):
        return self.expand(values, index), index

    def vjp_bwd(self, index, seed, values, position):
        CALLS.append(type(values).__name__)
        if isinstance(values, ad.NullAccum):
            return
        if isinstance(values, ad.RefAccum):
            values.inst().ref.addupdate(seed, idx=index)
        else:
            values.accum(jnp.zeros((8,), dtype=jnp.float32).at[index].add(seed))


class Square(hijax.VJPHiPrimitive):
    """A nonlinear rule whose backward consists of ordinary differentiable work."""

    def __init__(self):
        self.in_avals = (core.ShapedArray((), np.dtype('float32')),)
        self.out_aval = self.in_avals[0]
        self.params = {}
        super().__init__()

    def expand(self, value):
        return value * value

    def vjp_fwd(self, nonzeros, value):
        return self.expand(value), value

    def batch_dim_rule(self, axis_data, dimensions):
        return dimensions[0]

    def jvp(self, primals, tangents):
        return self.expand(primals[0]), 2 * primals[0] * tangents[0]

    def vjp_bwd(self, value, seed, accumulator):
        accumulator.accum(2 * value * seed)


def main():
    """Check each accumulator path and report generated IR plus higher-order results."""
    result = dict(python=platform.python_version(), jax=jax.__version__, jaxlib=jaxlib.__version__)
    indexed = IndexedRead()
    values = jnp.arange(8, dtype=jnp.float32)

    def returned(index):
        _, backward = jax.vjp(indexed, values, index)
        return backward.with_refs(jax.ad.GradValue(), jax.ad.DontWant())(jnp.float32(1))[0]

    returned_program = jax.make_jaxpr(returned)(jnp.int32(3))
    np.testing.assert_array_equal(returned(jnp.int32(3)), np.eye(8, dtype=np.float32)[3])
    result['returned'] = dict(program=str(returned_program), primitives=primitive_counts(returned_program))

    @jax.jit
    def accumulate(destination, index):
        _, backward = jax.vjp(indexed, values, index)
        backward.with_refs(destination, jax.ad.DontWant())(jnp.float32(1))

    buffer = jax.ref.new_ref(jnp.full_like(values, 2))
    buffer_program = jax.make_jaxpr(accumulate)(buffer, jnp.int32(3))
    before_execution = len(CALLS)
    for index in (3, 5, 3):
        accumulate(buffer, jnp.int32(index))
    actual = np.asarray(jax.ref.get(buffer))
    np.testing.assert_array_equal(actual, [2, 2, 2, 4, 2, 3, 2, 2])
    assert len(CALLS) == before_execution
    buffer_counts = primitive_counts(buffer_program)
    assert buffer_counts['addupdate'] == 1
    assert 'scatter-add' not in buffer_counts
    assert 'broadcast_in_dim' not in buffer_counts
    assert not any('accum' in name for name in buffer_counts)
    result['buffer'] = dict(program=str(buffer_program), primitives=primitive_counts(buffer_program),
                            final_buffer=actual.tolist(), callbacks_during_three_executions=len(CALLS)-before_execution)

    def ignored(index):
        _, backward = jax.vjp(indexed, values, index)
        return backward.with_refs(jax.ad.DontWant(), jax.ad.DontWant())(jnp.float32(1))

    ignored_program = jax.make_jaxpr(ignored)(jnp.int32(3))
    assert all(type(item).__name__ == 'DidntWant' for item in ignored(jnp.int32(3)))
    result['ignored'] = dict(program=str(ignored_program), primitives=primitive_counts(ignored_program))
    result['backward_callback_accumulator_classes'] = CALLS.copy()

    square = Square()
    first = jax.grad(square)(jnp.float32(3))
    second = jax.grad(jax.grad(square))(jnp.float32(3))
    np.testing.assert_array_equal(first, 6)
    np.testing.assert_array_equal(second, 2)
    result['higher_order'] = dict(first=float(first), second=float(second),
                                  program=str(jax.make_jaxpr(jax.grad(jax.grad(square)))(jnp.float32(3))))
    inputs = jnp.arange(4, dtype=jnp.float32)
    for name, function in (
        ('vmap_grad', jax.vmap(jax.grad(square))),
        ('grad_vmap', jax.grad(lambda array: jax.vmap(square)(array).sum())),
    ):
        try:
            actual = function(inputs)
            np.testing.assert_array_equal(actual, 2 * np.arange(4, dtype=np.float32))
            result[name] = dict(values=np.asarray(actual).tolist())
        except NotImplementedError as error:
            result[name] = dict(error_type=type(error).__name__, error=str(error))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
