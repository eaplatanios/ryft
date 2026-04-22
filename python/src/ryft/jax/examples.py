from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ryft.jax.extraction import ProgramCase


def differential(function: Callable[[jax.Array], jax.Array]) -> Callable[[jax.Array], jax.Array]:
    """Returns the forward-mode differential of a scalar JAX function."""

    return lambda x: jax.jvp(function, [x], [1.0])[1]


def sin_plus_cos_times_sin(x: jax.Array) -> jax.Array:
    """Returns `sin(x) + cos(x) * sin(x)`."""

    return jax.lax.sin(x) + jax.lax.cos(x) * jax.lax.sin(x)


def scalar_float32_abstract_value() -> jax.ShapeDtypeStruct:
    """Builds the scalar abstract value used by the inspection cases."""

    return jax.ShapeDtypeStruct(shape=(), dtype=np.dtype(np.float32))


def program_cases_by_name() -> dict[str, ProgramCase]:
    """Returns the named JAX inspection cases derived from the old ad hoc experiments."""

    scalar_abstract_value = scalar_float32_abstract_value()
    right_mul_transpose = jax.linear_transpose(lambda x: x * 4.2, scalar_abstract_value)

    return {
        "square": ProgramCase(
            name="square",
            description="Simple scalar multiplication rendered from `x * x`.",
            function=lambda x: x * x,
            example_args=(1.0,),
            abstract_args=(scalar_abstract_value,),
        ),
        "left_mul_4_2_transpose": ProgramCase(
            name="left_mul_4_2_transpose",
            description="Linear transpose of `4.2 * x`.",
            function=jax.linear_transpose(lambda x: 4.2 * x, scalar_abstract_value),
            example_args=(1.0,),
            abstract_args=(scalar_abstract_value,),
        ),
        "right_mul_4_2_transpose": ProgramCase(
            name="right_mul_4_2_transpose",
            description="Linear transpose of `x * 4.2`.",
            function=right_mul_transpose,
            example_args=(1.0,),
            abstract_args=(scalar_abstract_value,),
        ),
        "right_mul_4_2_transpose_differential": ProgramCase(
            name="right_mul_4_2_transpose_differential",
            description="Forward-mode differential of the transpose of `x * 4.2`.",
            function=differential(right_mul_transpose),
            example_args=(1.0,),
            abstract_args=(scalar_abstract_value,),
        ),
        "right_mul_4_2_transpose_second_differential": ProgramCase(
            name="right_mul_4_2_transpose_second_differential",
            description="Second forward-mode differential of the transpose of `x * 4.2`.",
            function=differential(differential(right_mul_transpose)),
            example_args=(1.0,),
            abstract_args=(scalar_abstract_value,),
        ),
        "sin_plus_cos_times_sin": ProgramCase(
            name="sin_plus_cos_times_sin",
            description="Mixed trigonometric scalar program.",
            function=sin_plus_cos_times_sin,
            example_args=(3.0,),
            abstract_args=(scalar_abstract_value,),
        ),
        "sin_plus_cos_times_sin_differential": ProgramCase(
            name="sin_plus_cos_times_sin_differential",
            description="Forward-mode differential of the mixed trigonometric scalar program.",
            function=differential(sin_plus_cos_times_sin),
            example_args=(3.0,),
            abstract_args=(scalar_abstract_value,),
        ),
    }
