"""JAX references for bounded data-dependent shape behavior that Ryft supports symbolically.

Run this module against the repository-pinned JAX environment:

```
uv run python -m unittest discover -s tests -p test_data_dependent_shape_parity.py
```
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np


class DataDependentShapeParityTest(unittest.TestCase):
    """Pins JAX's staging boundary for data-dependent result extents."""

    def test_data_dependent_prefix_take_requires_concretization(self) -> None:
        """Pins the JAX boundary exceeded by Ryft's checked dimension gateway."""

        def prefix(values: jax.Array, mask: jax.Array) -> jax.Array:
            count = jnp.count_nonzero(mask)
            return jnp.take(values, jnp.arange(count))

        values = jnp.array([10.0, 20.0, 30.0, 40.0], dtype=jnp.float32)
        mask = jnp.array([True, False, True, False])
        np.testing.assert_array_equal(prefix(values, mask), jnp.array([10.0, 20.0], dtype=jnp.float32))

        # JAX can execute the concrete eager count, but staging requires `jnp.arange`'s result length to be static.
        # Ryft's paired `test_array_ir_data_dependent_prefix_slice` fixture stages the count as scalar SSA, crosses
        # the checked `dimension_from_scalar` gateway, and retains the resulting bounded extent symbolically.
        with self.assertRaises(jax.errors.ConcretizationTypeError):
            jax.make_jaxpr(prefix)(values, mask)


if __name__ == "__main__":
    unittest.main()
