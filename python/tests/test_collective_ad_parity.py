"""Executable JAX references for shape-changing collective adjoints.

Run this module with four host devices:

```
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  uv run python -m unittest discover -s tests -p test_collective_ad_parity.py
```
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np


HAS_FOUR_DEVICES = len(jax.devices()) >= 4


@unittest.skipUnless(HAS_FOUR_DEVICES, "collective parity requires four JAX devices")
class CollectiveAdParityTest(unittest.TestCase):
    """Pins the current JAX transpose semantics mirrored by Ryft's collective rules."""

    def test_varying_all_gather_transposes_to_sum_scatter(self) -> None:
        inputs = jnp.arange(8.0, dtype=jnp.float32).reshape(4, 2)
        gather = jax.pmap(
            lambda value: jax.lax.all_gather(value, "x", axis=0, tiled=True),
            axis_name="x",
        )

        outputs, pullback = jax.vjp(gather, inputs)
        output_cotangents = jnp.arange(outputs.size, dtype=outputs.dtype).reshape(outputs.shape)
        (input_cotangents,) = pullback(output_cotangents)
        expected = np.stack(
            [np.asarray(output_cotangents)[:, 2 * index : 2 * (index + 1)].sum(axis=0) for index in range(4)]
        )

        np.testing.assert_array_equal(input_cotangents, expected)

    def test_sum_scatter_transposes_to_varying_all_gather(self) -> None:
        inputs = jnp.arange(32.0, dtype=jnp.float32).reshape(4, 8)
        scatter = jax.pmap(
            lambda value: jax.lax.psum_scatter(value, "x", scatter_dimension=0, tiled=True),
            axis_name="x",
        )

        outputs, pullback = jax.vjp(scatter, inputs)
        output_cotangents = jnp.arange(outputs.size, dtype=outputs.dtype).reshape(outputs.shape)
        (input_cotangents,) = pullback(output_cotangents)
        expected = np.broadcast_to(np.asarray(output_cotangents).reshape(1, -1), inputs.shape)

        np.testing.assert_array_equal(input_cotangents, expected)

    def test_all_to_all_transpose_swaps_split_and_concatenation_axes(self) -> None:
        inputs = jnp.arange(32.0, dtype=jnp.float32).reshape(4, 2, 4)
        exchange = jax.pmap(
            lambda value: jax.lax.all_to_all(value, "x", split_axis=1, concat_axis=0, tiled=True),
            axis_name="x",
        )
        inverse_exchange = jax.pmap(
            lambda value: jax.lax.all_to_all(value, "x", split_axis=0, concat_axis=1, tiled=True),
            axis_name="x",
        )

        outputs, pullback = jax.vjp(exchange, inputs)
        output_cotangents = jnp.arange(outputs.size, dtype=outputs.dtype).reshape(outputs.shape)
        (input_cotangents,) = pullback(output_cotangents)

        np.testing.assert_array_equal(input_cotangents, inverse_exchange(output_cotangents))

    def test_invariant_all_gather_transpose_selects_the_local_chunk(self) -> None:
        from jax.sharding import AxisType, PartitionSpec as P

        # Use `shard_map` rather than `pmap`, whose JAX 0.12 pullback sums every device's chunk and therefore no longer
        # isolates this transpose. An invariant output has one cotangent, and each device selects its own chunk of it.
        mesh = jax.make_mesh((4,), ("x",), axis_types=(AxisType.Explicit,))
        with jax.set_mesh(mesh):
            inputs = jax.device_put(jnp.arange(8.0, dtype=jnp.float32).reshape(4, 2), P("x"))
            gather = jax.shard_map(
                lambda value: jax.lax.all_gather(value, "x", axis=0, tiled=True, to="invarying"),
                out_specs=P(),
            )

            outputs, pullback = jax.vjp(gather, inputs)
            output_cotangents = jnp.arange(outputs.size, dtype=outputs.dtype).reshape(outputs.shape)
            (input_cotangents,) = pullback(output_cotangents)

        np.testing.assert_array_equal(input_cotangents, output_cotangents)

    def test_reduced_all_gather_transpose_consumes_an_unreduced_cotangent(self) -> None:
        from jax.sharding import AxisType, Mesh, PartitionSpec as P

        mesh = Mesh(
            np.asarray(jax.devices()[:4]).reshape(2, 2),
            ("x", "y"),
            axis_types=(AxisType.Explicit, AxisType.Explicit),
        )
        with jax.set_mesh(mesh):
            inputs = jax.device_put(np.arange(32.0, dtype=np.float32).reshape(8, 4), P("x", "y"))

            @jax.jit
            @jax.shard_map(out_specs=P(reduced={"y"}))
            def loss(values):
                gathered = jax.lax.all_gather(values, "y", axis=1, tiled=True, to="reduced")
                return jax.lax.psum(jnp.sum(gathered), "x")

            input_cotangents = jax.jit(jax.grad(loss))(inputs)

        np.testing.assert_array_equal(input_cotangents, np.ones(inputs.shape, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
