"""Executable JAX references for Ryft's condition, scan, and while differentiation rules.

Run this module against the repository-pinned JAX environment:

```
uv run --project python python -m unittest discover -s python/tests -p test_control_flow_ad_parity.py
```
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np


class ControlFlowAdParityTest(unittest.TestCase):
    """Pins the JAX control-flow transformation behavior implemented by Ryft."""

    def test_condition_selects_the_same_branch_for_primal_and_derivative(self) -> None:
        def condition(predicate, operand):
            return jax.lax.cond(predicate, lambda value: 2.0 * value, lambda value: 3.0 * value, operand)

        for predicate, factor in ((True, 2.0), (False, 3.0)):
            primal, tangent = jax.jvp(
                condition,
                (jnp.array(predicate), jnp.array(5.0)),
                (jnp.zeros((), dtype=jax.dtypes.float0), jnp.array(7.0)),
            )
            np.testing.assert_array_equal(primal, 5.0 * factor)
            np.testing.assert_array_equal(tangent, 7.0 * factor)
            np.testing.assert_array_equal(jax.grad(lambda value: condition(predicate, value))(5.0), factor)

        np.testing.assert_array_equal(
            jax.vmap(condition)(jnp.array([True, False]), jnp.array([5.0, 5.0])),
            jnp.array([10.0, 15.0]),
        )

    def test_scan_supports_carries_outputs_reverse_zero_length_and_derivatives(self) -> None:
        def body(carry, item):
            next_carry = carry * item
            return next_carry, next_carry

        def scan(carry, items, *, reverse=False):
            return jax.lax.scan(body, carry, items, reverse=reverse)

        carry = jnp.array(1.0)
        items = jnp.array([2.0, 3.0, 4.0])
        final_carry, outputs = scan(carry, items)
        np.testing.assert_array_equal(final_carry, 24.0)
        np.testing.assert_array_equal(outputs, jnp.array([2.0, 6.0, 24.0]))

        reverse_carry, reverse_outputs = scan(carry, items, reverse=True)
        np.testing.assert_array_equal(reverse_carry, 24.0)
        np.testing.assert_array_equal(reverse_outputs, jnp.array([24.0, 12.0, 4.0]))

        empty_carry, empty_outputs = scan(carry, jnp.empty((0,), dtype=items.dtype))
        np.testing.assert_array_equal(empty_carry, carry)
        self.assertEqual(empty_outputs.shape, (0,))

        (primal_carry, primal_outputs), (tangent_carry, tangent_outputs) = jax.jvp(
            scan,
            (carry, items),
            (jnp.array(5.0), jnp.array([0.5, 1.0, 1.5])),
        )
        np.testing.assert_array_equal(primal_carry, final_carry)
        np.testing.assert_array_equal(primal_outputs, outputs)
        np.testing.assert_array_equal(tangent_carry, 143.0)
        np.testing.assert_array_equal(tangent_outputs, jnp.array([10.5, 33.5, 143.0]))

        def loss(initial_carry, scanned_items):
            result_carry, result_outputs = scan(initial_carry, scanned_items)
            return result_carry + jnp.sum(result_outputs)

        carry_cotangent, item_cotangents = jax.grad(loss, argnums=(0, 1))(carry, items)
        np.testing.assert_array_equal(carry_cotangent, 56.0)
        np.testing.assert_array_equal(item_cotangents, jnp.array([28.0, 18.0, 12.0]))

    def test_while_supports_forward_mode_and_rejects_reverse_mode(self) -> None:
        def loop(operand):
            return jax.lax.while_loop(lambda value: value < 8.0, lambda value: 2.0 * value, operand)

        primal, tangent = jax.jvp(loop, (jnp.array(1.0),), (jnp.array(3.0),))
        np.testing.assert_array_equal(primal, 8.0)
        np.testing.assert_array_equal(tangent, 24.0)
        np.testing.assert_array_equal(jax.jit(loop)(jnp.array(1.0)), 8.0)
        np.testing.assert_array_equal(jax.vmap(loop)(jnp.array([1.0, 2.0, 4.0])), jnp.array([8.0, 8.0, 8.0]))

        with self.assertRaisesRegex(ValueError, "Reverse-mode differentiation does not work for lax.while_loop"):
            jax.grad(loop)(jnp.array(1.0))


if __name__ == "__main__":
    unittest.main()
