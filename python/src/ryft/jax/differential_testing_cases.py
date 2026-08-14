"""Shared case registry and pinned-JAX emitters for differential testing.

JAX is imported only after `build_jax_observations` configures four host devices. Keep this module free of module-level
JAX imports so the parent comparison process can safely spawn it after running the Ryft emitter.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ryft.jax.differential_testing import (
    PINNED_JAX_VERSION,
    SCHEMA,
    DifferentialObservation,
    StableHloCollective,
    StagingObservation,
)


@dataclass(frozen=True)
class DifferentialCase:
    """One registered workload and the relationship expected between Ryft and JAX.

    `collectives` is the collective projection both frameworks must produce, in module order. `stablehlo_patterns`
    lists semantic operation or attribute spellings that must occur in both modules. An exact-parity case must declare
    at least one of these contracts, because comparing two unconstrained modules would pass vacuously. Capability
    cases that compare no module declare neither contract.
    """

    case_id: str
    relationship: str
    build_jax: Callable[[Any, Any, Any], DifferentialObservation]
    collectives: tuple[StableHloCollective, ...] = ()
    stablehlo_patterns: tuple[str, ...] = ()


def _configure_jax_devices() -> None:
    """Configures four host devices before the process imports JAX."""

    flag = "--xla_force_host_platform_device_count=4"
    existing = [
        value
        for value in os.environ.get("XLA_FLAGS", "").split()
        if not value.startswith("--xla_force_host_platform_device_count=")
    ]
    os.environ["XLA_FLAGS"] = " ".join((*existing, flag))


def _observation_values(array: Any, numpy: Any) -> tuple[tuple[float, ...], ...]:
    """Returns one flattened logical output vector per leading device axis."""

    values = numpy.asarray(array)
    return tuple(
        tuple(float(value) for value in device_values) for device_values in values.reshape(values.shape[0], -1)
    )


def _single_observation_values(array: Any, numpy: Any) -> tuple[tuple[float, ...], ...]:
    """Returns one flattened logical output vector for a non-collective execution."""

    return (tuple(float(value) for value in numpy.asarray(array).reshape(-1)),)


def _build_grouped_collectives(jax: Any, jax_numpy: Any, numpy: Any) -> DifferentialObservation:
    """Builds grouped tiled all-gather, sum-scatter, and all-to-all observations."""

    groups = [[0, 2], [3, 1]]

    def grouped(input_value: Any) -> tuple[Any, Any, Any]:
        return (
            jax.lax.all_gather(input_value, "x", axis=0, tiled=True, axis_index_groups=groups),
            jax.lax.psum_scatter(input_value, "x", scatter_dimension=0, tiled=True, axis_index_groups=groups),
            jax.lax.all_to_all(
                input_value,
                "x",
                split_axis=0,
                concat_axis=0,
                tiled=True,
                axis_index_groups=groups,
            ),
        )

    function = jax.pmap(grouped, axis_name="x")
    inputs = jax_numpy.arange(16.0, dtype=jax_numpy.float32).reshape(4, 4)
    all_gather, psum_scatter, all_to_all = function(inputs)
    return DifferentialObservation(
        schema=SCHEMA,
        case_id="grouped_shape_changing_collectives",
        observations={
            "all_gather": _observation_values(all_gather, numpy),
            "psum_scatter": _observation_values(psum_scatter, numpy),
            "all_to_all": _observation_values(all_to_all, numpy),
        },
        stablehlo=str(function.lower(inputs).compiler_ir("stablehlo")),
    )


def _build_pshuffle(jax: Any, jax_numpy: Any, numpy: Any) -> DifferentialObservation:
    """Builds `pshuffle` behavior and its `collective_permute` lowering."""

    function = jax.pmap(lambda input_value: jax.lax.pshuffle(input_value, "x", [2, 0, 3, 1]), axis_name="x")
    inputs = jax_numpy.arange(8.0, dtype=jax_numpy.float32).reshape(4, 2)
    output = function(inputs)
    return DifferentialObservation(
        schema=SCHEMA,
        case_id="pshuffle",
        observations={"output": _observation_values(output, numpy)},
        stablehlo=str(function.lower(inputs).compiler_ir("stablehlo")),
    )


def _build_pswapaxes(jax: Any, jax_numpy: Any, numpy: Any) -> DifferentialObservation:
    """Builds `pswapaxes` behavior and its untiled all-to-all lowering."""

    function = jax.pmap(lambda input_value: jax.lax.pswapaxes(input_value, "x", 0), axis_name="x")
    inputs = jax_numpy.arange(32.0, dtype=jax_numpy.float32).reshape(4, 4, 2)
    output = function(inputs)
    return DifferentialObservation(
        schema=SCHEMA,
        case_id="pswapaxes",
        observations={"output": _observation_values(output, numpy)},
        stablehlo=str(function.lower(inputs).compiler_ir("stablehlo")),
    )


def _build_data_dependent_prefix_take(jax: Any, jax_numpy: Any, numpy: Any) -> DifferentialObservation:
    """Builds eager prefix-take values and records JAX's traced-data staging rejection."""

    def prefix(values: Any, mask: Any) -> Any:
        count = jax_numpy.count_nonzero(mask)
        return jax_numpy.take(values, jax_numpy.arange(count))

    values = jax_numpy.array([10.0, 20.0, 30.0, 40.0], dtype=jax_numpy.float32)
    two_matches = prefix(values, jax_numpy.array([True, False, True, False]))
    zero_matches = prefix(values, jax_numpy.array([False, False, False, False]))
    try:
        jax.make_jaxpr(prefix)(values, jax_numpy.array([True, False, True, False]))
    except jax.errors.ConcretizationTypeError:
        staging = StagingObservation(status="rejected", category="concretization")
    else:
        staging = StagingObservation(status="supported", output_type="unexpected")
    return DifferentialObservation(
        schema=SCHEMA,
        case_id="data_dependent_prefix_take",
        observations={
            "two_matches": (tuple(float(value) for value in numpy.asarray(two_matches)),),
            "zero_matches": (tuple(float(value) for value in numpy.asarray(zero_matches)),),
        },
        staging=staging,
    )


def _build_scaled_dot_and_matmul(jax: Any, jax_numpy: Any, numpy: Any) -> DifferentialObservation:
    """Builds generalized scaled-dot values and the named-composite lowering used by scaled matmul."""

    lhs = jax_numpy.arange(1, 9, dtype=jax_numpy.float32).reshape(2, 4)
    rhs = jax_numpy.arange(1, 13, dtype=jax_numpy.float32).reshape(4, 3)
    lhs_scale = jax_numpy.array([[1.0, 2.0], [0.5, 1.0]], dtype=jax_numpy.float32)
    rhs_scale = jax_numpy.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=jax_numpy.float32)
    dimensions = (((1,), (0,)), ((), ()))

    def scaled_dot(lhs_value: Any, rhs_value: Any, lhs_scale_value: Any, rhs_scale_value: Any) -> Any:
        return jax.lax.scaled_dot(
            lhs_value,
            rhs_value,
            lhs_scale=lhs_scale_value,
            rhs_scale=rhs_scale_value,
            dimension_numbers=dimensions,
            preferred_element_type=jax_numpy.float32,
        )

    # `scaled_matmul` has no CPU lowering in pinned JAX. Its public rank-three contract is nevertheless exactly the
    # corresponding `scaled_dot` dimension arrangement, so evaluate that semantic composition after asking JAX to
    # validate the wrapper's abstract output signature.
    matmul_lhs = jax_numpy.ones((1, 1, 4), dtype=jax_numpy.float32)
    matmul_rhs = jax_numpy.ones((1, 2, 4), dtype=jax_numpy.float32)
    matmul_lhs_scale = jax_numpy.ones((1, 1, 2), dtype=jax_numpy.float32)
    matmul_rhs_scale = jax_numpy.ones((1, 2, 2), dtype=jax_numpy.float32)
    matmul_output = jax.lax.scaled_dot(
        matmul_lhs,
        matmul_rhs,
        lhs_scale=matmul_lhs_scale,
        rhs_scale=matmul_rhs_scale,
        dimension_numbers=(((2,), (2,)), ((0,), (0,))),
        preferred_element_type=jax_numpy.float32,
    )
    abstract_output = jax.eval_shape(
        jax.nn.scaled_matmul,
        matmul_lhs,
        matmul_rhs,
        matmul_lhs_scale,
        matmul_rhs_scale,
    )
    if abstract_output.shape != matmul_output.shape or abstract_output.dtype != matmul_output.dtype:
        raise AssertionError("scaled_matmul and its scaled_dot composition disagree on the abstract output")

    observations = {
        "both_scales": _single_observation_values(scaled_dot(lhs, rhs, lhs_scale, rhs_scale), numpy),
        "lhs_scale": _single_observation_values(
            jax.lax.scaled_dot(
                lhs,
                rhs,
                lhs_scale=lhs_scale,
                dimension_numbers=dimensions,
                preferred_element_type=jax_numpy.float32,
            ),
            numpy,
        ),
        "rhs_scale": _single_observation_values(
            jax.lax.scaled_dot(
                lhs,
                rhs,
                rhs_scale=rhs_scale,
                dimension_numbers=dimensions,
                preferred_element_type=jax_numpy.float32,
            ),
            numpy,
        ),
        "unscaled": _single_observation_values(
            jax.lax.scaled_dot(
                lhs,
                rhs,
                dimension_numbers=dimensions,
                preferred_element_type=jax_numpy.float32,
            ),
            numpy,
        ),
        "scaled_matmul": _single_observation_values(matmul_output, numpy),
    }
    return DifferentialObservation(
        schema=SCHEMA,
        case_id="scaled_dot_and_matmul",
        observations=observations,
        stablehlo=str(jax.jit(scaled_dot).lower(lhs, rhs, lhs_scale, rhs_scale).compiler_ir("stablehlo")),
    )


def _build_dot_product_attention(jax: Any, jax_numpy: Any, numpy: Any) -> DifferentialObservation:
    """Builds portable rank-three MQA attention with every structural option represented."""

    query = jax_numpy.zeros((2, 2, 1), dtype=jax_numpy.float32)
    key = jax_numpy.zeros((2, 1, 1), dtype=jax_numpy.float32)
    value = jax_numpy.array([[[3.0]], [[9.0]]], dtype=jax_numpy.float32)
    bias = jax_numpy.array(0.0, dtype=jax_numpy.float32)
    mask = jax_numpy.array([[True, False], [False, True]])
    query_sequence_lengths = jax_numpy.array([2], dtype=jax_numpy.int32)
    key_value_sequence_lengths = jax_numpy.array([2], dtype=jax_numpy.int32)

    def attention(
        query_value: Any,
        key_value: Any,
        value_value: Any,
        bias_value: Any,
        mask_value: Any,
        query_lengths_value: Any,
        key_value_lengths_value: Any,
    ) -> tuple[Any, Any]:
        return jax.nn.dot_product_attention(
            query_value,
            key_value,
            value_value,
            bias=bias_value,
            mask=mask_value,
            scale=1.0,
            is_causal=True,
            query_seq_lengths=query_lengths_value,
            key_value_seq_lengths=key_value_lengths_value,
            local_window_size=(1, 0),
            implementation="xla",
            return_residual=True,
        )

    output, residual = attention(
        query,
        key,
        value,
        bias,
        mask,
        query_sequence_lengths,
        key_value_sequence_lengths,
    )
    gqa_output = jax.nn.dot_product_attention(
        jax_numpy.zeros((1, 2, 4, 1), dtype=jax_numpy.float32),
        jax_numpy.zeros((1, 3, 2, 1), dtype=jax_numpy.float32),
        jax_numpy.array([[[[1.0], [10.0]], [[2.0], [20.0]], [[4.0], [40.0]]]], dtype=jax_numpy.float32),
        key_value_seq_lengths=key_value_sequence_lengths,
        local_window_size=(1, 1),
        implementation="xla",
    )
    return DifferentialObservation(
        schema=SCHEMA,
        case_id="dot_product_attention",
        observations={
            "output": _single_observation_values(output, numpy),
            "rank_four_gqa": _single_observation_values(gqa_output, numpy),
            "residual": _single_observation_values(residual, numpy),
        },
        stablehlo=str(
            jax.jit(attention)
            .lower(query, key, value, bias, mask, query_sequence_lengths, key_value_sequence_lengths)
            .compiler_ir("stablehlo")
        ),
    )


# Ordered participant groups both frameworks emit for the `[[0, 2], [3, 1]]` grouping of the grouped collectives.
_GROUPED_COLLECTIVE_GROUPS = ((0, 2), (3, 1))


DIFFERENTIAL_CASES = (
    DifferentialCase(
        "grouped_shape_changing_collectives",
        "parity",
        _build_grouped_collectives,
        (
            StableHloCollective("all_gather", _GROUPED_COLLECTIVE_GROUPS, (("all_gather_dim", 0),)),
            StableHloCollective("reduce_scatter", _GROUPED_COLLECTIVE_GROUPS, (("scatter_dimension", 0),)),
            StableHloCollective(
                "all_to_all",
                _GROUPED_COLLECTIVE_GROUPS,
                (("concat_dimension", 0), ("split_count", 2), ("split_dimension", 0)),
            ),
        ),
    ),
    DifferentialCase(
        "pshuffle",
        "parity",
        _build_pshuffle,
        (StableHloCollective("collective_permute", ((2, 0), (0, 1), (3, 2), (1, 3)), ()),),
    ),
    DifferentialCase(
        "pswapaxes",
        "parity",
        _build_pswapaxes,
        (
            StableHloCollective(
                "all_to_all",
                ((0, 1, 2, 3),),
                (("concat_dimension", 0), ("split_count", 4), ("split_dimension", 0)),
            ),
        ),
    ),
    DifferentialCase("data_dependent_prefix_take", "ryft_exceeds_jax", _build_data_dependent_prefix_take),
    DifferentialCase(
        "scaled_dot_and_matmul",
        "parity",
        _build_scaled_dot_and_matmul,
        stablehlo_patterns=('stablehlo.composite "xla.scaled_dot"', "dimension_numbers", "preferred_element_type"),
    ),
    DifferentialCase(
        "dot_product_attention",
        "parity",
        _build_dot_product_attention,
        stablehlo_patterns=("stablehlo.dot_general", "stablehlo.select", "stablehlo.exponential", "stablehlo.reduce"),
    ),
)


def build_jax_observations(case_ids: Sequence[str]) -> tuple[DifferentialObservation, ...]:
    """Builds selected observations against the exact repository-pinned JAX version.

    # Parameters

      - `case_ids`: Exact registry IDs to execute, in desired output order.
    """

    _configure_jax_devices()
    import jax
    import jax.numpy as jax_numpy
    import numpy

    if jax.__version__ != PINNED_JAX_VERSION:
        raise RuntimeError(f"differential harness requires jax=={PINNED_JAX_VERSION} but found {jax.__version__}")
    if len(jax.devices()) != 4:
        raise RuntimeError(
            f"differential harness requires exactly four JAX host devices but found {len(jax.devices())}"
        )
    by_id = {case.case_id: case for case in DIFFERENTIAL_CASES}
    return tuple(by_id[case_id].build_jax(jax, jax_numpy, numpy) for case_id in case_ids)


__all__ = ["DIFFERENTIAL_CASES", "DifferentialCase", "build_jax_observations"]
