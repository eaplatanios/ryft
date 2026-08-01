"""Executable JAX references for the extent-sensitive adjoints implemented by Ryft.

Run this module against the repository-pinned JAX environment:

```
uv run python -m unittest discover -s tests -p test_extent_sensitive_ad_parity.py
```
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np


class ExtentSensitiveAdParityTest(unittest.TestCase):
    """Pins JAX's transformation behavior for extent-sensitive array operations."""

    def assert_linear_transforms(
        self,
        function,
        primals: tuple[jax.Array, ...],
        tangents: tuple[jax.Array, ...],
        output_cotangent: jax.Array,
        expected_tangent: jax.Array,
        expected_cotangents: tuple[jax.Array, ...],
    ) -> None:
        """Checks all forward and reverse transformations of one linear function."""
        _, tangent = jax.jvp(function, primals, tangents)
        np.testing.assert_array_equal(tangent, expected_tangent)

        _, pushforward = jax.linearize(function, *primals)
        np.testing.assert_array_equal(pushforward(*tangents), expected_tangent)

        _, pullback = jax.vjp(function, *primals)
        for actual, expected in zip(pullback(output_cotangent), expected_cotangents, strict=True):
            np.testing.assert_array_equal(actual, expected)

        transpose = jax.linear_transpose(function, *primals)
        for actual, expected in zip(transpose(output_cotangent), expected_cotangents, strict=True):
            np.testing.assert_array_equal(actual, expected)

    def test_broadcast_and_concatenate(self) -> None:
        broadcast = lambda operand: jax.lax.broadcast_in_dim(operand, (3, 2), (1,))
        operand = jnp.array([1.0, 2.0], dtype=jnp.float32)
        operand_tangent = jnp.array([3.0, 4.0], dtype=jnp.float32)
        output_cotangent = jnp.arange(1.0, 7.0, dtype=jnp.float32).reshape(3, 2)
        self.assert_linear_transforms(
            broadcast,
            (operand,),
            (operand_tangent,),
            output_cotangent,
            jnp.broadcast_to(operand_tangent, (3, 2)),
            (jnp.array([9.0, 12.0], dtype=jnp.float32),),
        )
        batched = jax.vmap(broadcast)(jnp.stack([operand, operand + 2.0]))
        np.testing.assert_array_equal(batched.shape, (2, 3, 2))

        concatenate = lambda left, right: jnp.concatenate([left, right], axis=1)
        left = jnp.array([[1.0], [2.0]], dtype=jnp.float32)
        right = jnp.array([[3.0, 4.0], [5.0, 6.0]], dtype=jnp.float32)
        left_tangent = left + 6.0
        right_tangent = right + 6.0
        output_cotangent = jnp.arange(1.0, 7.0, dtype=jnp.float32).reshape(2, 3)
        self.assert_linear_transforms(
            concatenate,
            (left, right),
            (left_tangent, right_tangent),
            output_cotangent,
            jnp.concatenate([left_tangent, right_tangent], axis=1),
            (output_cotangent[:, :1], output_cotangent[:, 1:]),
        )

    def test_pad_differentiates_the_operand_and_padding_value(self) -> None:
        pad = lambda operand, padding: jax.lax.pad(operand, padding, ((1, 1, 0),))
        operand = jnp.array([2.0, 3.0], dtype=jnp.float32)
        padding = jnp.array(5.0, dtype=jnp.float32)
        output, pullback = jax.vjp(pad, operand, padding)
        np.testing.assert_array_equal(output, jnp.array([5.0, 2.0, 3.0, 5.0], dtype=jnp.float32))

        # JAX's padding-value reduction is contaminated by non-finite cotangents at operand positions. Ryft
        # deliberately exceeds this behavior by selecting padding positions before reducing and therefore returns 3.
        operand_cotangent, padding_cotangent = pullback(
            jnp.array([1.0, jnp.nan, jnp.nan, 2.0], dtype=jnp.float32)
        )
        np.testing.assert_array_equal(jnp.isnan(operand_cotangent), jnp.array([True, True]))
        self.assertTrue(bool(jnp.isnan(padding_cotangent)))

        batched = jax.vmap(pad, in_axes=(0, None))(jnp.stack([operand, operand + 2.0]), padding)
        np.testing.assert_array_equal(batched.shape, (2, 4))

    def test_slice_update_slice_and_gather(self) -> None:
        operand = jnp.arange(6.0, dtype=jnp.float32)
        slice_function = lambda values: jax.lax.slice(values, (1,), (6,), (2,))
        slice_cotangent = jnp.array([2.0, 3.0, 4.0], dtype=jnp.float32)
        _, slice_pullback = jax.vjp(slice_function, operand)
        np.testing.assert_array_equal(
            slice_pullback(slice_cotangent)[0],
            jnp.array([0.0, 2.0, 0.0, 3.0, 0.0, 4.0], dtype=jnp.float32),
        )

        dynamic_slice = lambda values: jax.lax.dynamic_slice(values, (1,), (3,))
        _, dynamic_slice_pullback = jax.vjp(dynamic_slice, operand)
        np.testing.assert_array_equal(
            dynamic_slice_pullback(jnp.array([2.0, 3.0, 4.0], dtype=jnp.float32))[0],
            jnp.array([0.0, 2.0, 3.0, 4.0, 0.0, 0.0], dtype=jnp.float32),
        )

        update = jnp.array([7.0, 8.0], dtype=jnp.float32)
        dynamic_update = lambda base, values: jax.lax.dynamic_update_slice(base, values, (2,))
        _, dynamic_update_pullback = jax.vjp(dynamic_update, operand, update)
        output_cotangent = jnp.arange(1.0, 7.0, dtype=jnp.float32)
        base_cotangent, update_cotangent = dynamic_update_pullback(output_cotangent)
        np.testing.assert_array_equal(base_cotangent, jnp.array([1.0, 2.0, 0.0, 0.0, 5.0, 6.0]))
        np.testing.assert_array_equal(update_cotangent, jnp.array([3.0, 4.0]))

        indices = jnp.array([1, 1, 3], dtype=jnp.int32)
        gather = lambda values: values[indices]
        _, gather_pullback = jax.vjp(gather, jnp.arange(4.0, dtype=jnp.float32))
        np.testing.assert_array_equal(
            gather_pullback(jnp.array([2.0, 3.0, 4.0], dtype=jnp.float32))[0],
            jnp.array([0.0, 5.0, 0.0, 4.0], dtype=jnp.float32),
        )

    def test_reductions(self) -> None:
        values = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
        np.testing.assert_array_equal(
            jax.grad(lambda operand: jnp.sum(operand))(values),
            jnp.ones_like(values),
        )
        np.testing.assert_array_equal(
            jax.grad(lambda operand: jnp.mean(operand))(values),
            jnp.full_like(values, 0.25),
        )

        ties = jnp.array([2.0, 2.0, 1.0], dtype=jnp.float32)
        np.testing.assert_array_equal(
            jax.grad(lambda operand: jnp.max(operand))(ties),
            jnp.array([0.5, 0.5, 0.0], dtype=jnp.float32),
        )
        np.testing.assert_array_equal(
            jax.grad(lambda operand: jnp.min(operand))(-ties),
            jnp.array([0.5, 0.5, 0.0], dtype=jnp.float32),
        )

        empty = jnp.empty((0,), dtype=jnp.float32)
        np.testing.assert_array_equal(jax.grad(lambda operand: jnp.sum(operand))(empty), empty)
        self.assertTrue(bool(jnp.isnan(jnp.mean(empty))))


if __name__ == "__main__":
    unittest.main()
