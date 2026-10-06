"""Native single-GPU ragged workloads and independent host transfer references.

The shared manifest describes runtime metadata, never compile-time constants. These cases require CUDA execution;
the host reference computes expected values without implementing an accelerator fallback.
"""

from __future__ import annotations

import json
import math
from typing import Any

from ryft.jax.collective_testing import validate_collective_arrays
from ryft.jax.differential_testing import SCHEMA, DifferentialObservation, repo_root


MANIFEST_PATH = repo_root() / "crates/ryft-xla/src/bin/differential_testing/ragged_cases.json"


def ragged_descriptors() -> tuple[dict[str, Any], ...]:
    """Returns explicitly selected CUDA workloads shared with the Rust emitter."""

    return tuple(json.loads(MANIFEST_PATH.read_text()))


def ragged_inputs(descriptor: dict[str, Any]) -> tuple[Any, ...]:
    """Creates float32 data and exact int32 or uint64 routing metadata for one GPU participant."""

    import numpy

    if descriptor["participants"] != 1:
        raise ValueError("CUDA ragged references require exactly one participant")
    input_shape = (1, *descriptor["local_shape"])
    output_shape = (1, *descriptor["output_shape"])
    operand = numpy.arange(1, math.prod(input_shape) + 1, dtype=numpy.float32).reshape(input_shape)
    seed = numpy.arange(100, 100 + math.prod(output_shape), dtype=numpy.float32).reshape(output_shape)
    batch_size = descriptor["local_shape"][0] if descriptor["transform"] == "batch" else 1
    metadata_shape = (1, batch_size, -1) if descriptor["transform"] == "batch" else (1, -1)
    metadata_type = {"i32": numpy.int32, "u64": numpy.uint64}[descriptor["metadata_type"]]
    metadata = tuple(
        numpy.asarray(descriptor[name], dtype=metadata_type).reshape(metadata_shape)
        for name in ("input_offsets", "send_sizes", "output_offsets", "receive_sizes")
    )
    return operand, seed, *metadata


def _exchange(operand: Any, seed: Any, metadata: tuple[Any, ...], batched: bool) -> Any:
    """Copies explicit leading-axis intervals into seed values using ordinary host arrays."""

    result = seed.copy()
    if batched:
        for index in range(operand.shape[1]):
            result[:, index] = _exchange(
                operand[:, index], seed[:, index], tuple(value[:, index] for value in metadata), False,
            )
        return result
    input_offsets, send_sizes, output_offsets, receive_sizes = metadata
    occupied: set[int] = set()
    for input_offset, send_size, output_offset, receive_size in zip(
        input_offsets[0], send_sizes[0], output_offsets[0], receive_sizes[0], strict=True,
    ):
        source, size, destination, received = map(int, (input_offset, send_size, output_offset, receive_size))
        if size != received or size < 0 or source < 0 or destination < 0:
            raise ValueError("invalid ragged transfer metadata")
        if source + size > operand.shape[1] or destination + size > seed.shape[1]:
            raise ValueError("ragged transfer exceeds its data extent")
        rows = set(range(destination, destination + size))
        if occupied.intersection(rows):
            raise ValueError("ragged received regions overlap")
        occupied.update(rows)
        result[0, destination:destination + size] = operand[0, source:source + size]
    return result


def _reference_arrays(descriptor: dict[str, Any]) -> dict[str, Any]:
    """Evaluates primal, forward tangent, and exact basis-vector adjoints of the host transfer map."""

    import numpy

    operand, seed, *metadata = ragged_inputs(descriptor)
    batched = descriptor["transform"] == "batch"
    primal = _exchange(operand, seed, tuple(metadata), batched)
    outputs = {"primal": primal}
    if descriptor["transform"] == "jvp":
        operand_tangent = (numpy.arange(operand.size, dtype=numpy.float32) % 5 + 1).reshape(operand.shape)
        seed_tangent = (numpy.arange(seed.size, dtype=numpy.float32) % 3 + 1).reshape(seed.shape)
        outputs["tangent"] = _exchange(operand_tangent, seed_tangent, tuple(metadata), False)
    elif descriptor["transform"] == "vjp":
        cotangent = numpy.arange(1, primal.size + 1, dtype=numpy.float32).reshape(primal.shape)
        # Differentiate both linear data families through independent basis vectors. Metadata remain fixed residuals.
        for name, value in (("operand_cotangent", operand), ("seed_cotangent", seed)):
            gradient = numpy.zeros_like(value)
            for index in range(value.size):
                input_basis = numpy.zeros_like(operand)
                seed_basis = numpy.zeros_like(seed)
                (input_basis if name == "operand_cotangent" else seed_basis).flat[index] = 1
                gradient.flat[index] = numpy.sum(_exchange(input_basis, seed_basis, tuple(metadata), False) * cotangent)
            outputs[name] = gradient
    return outputs


def ragged_reference(descriptor: dict[str, Any]) -> dict[str, tuple[tuple[float, ...], ...]]:
    """Returns independent per-participant values for the shared CUDA workload."""

    return {
        name: tuple(tuple(float(element) for element in local.reshape(-1)) for local in value)
        for name, value in _reference_arrays(descriptor).items()
    }


def build_ragged_jax(descriptor: dict[str, Any], jax: Any, jax_numpy: Any, numpy: Any) -> DifferentialObservation:
    """Executes runtime ragged metadata through pinned JAX on an actual CUDA device."""

    devices = jax.devices("cuda")
    if not devices or any(device.platform != "gpu" for device in devices[:1]):
        raise RuntimeError("CUDA ragged workloads require an actual GPU device")
    host_inputs = ragged_inputs(descriptor)
    inputs = tuple(jax_numpy.asarray(value) for value in host_inputs)
    if numpy.dtype(inputs[2].dtype) != numpy.dtype(host_inputs[2].dtype):
        raise RuntimeError("CUDA ragged metadata dtype changed; uint64 workloads require jax_enable_x64")

    def operation(operand: Any, seed: Any, *metadata: Any) -> Any:
        """Binds the runtime routing metadata to the current GPU participant axis."""

        return jax.lax.ragged_all_to_all(operand, seed, *metadata, axis_name="x")

    if descriptor["transform"] == "batch":
        operation = jax.vmap(operation)
    function = jax.pmap(operation, axis_name="x", devices=devices[:1])
    transform = descriptor["transform"]
    if transform == "jvp":
        def observed(operand: Any, seed: Any, *metadata: Any) -> tuple[Any, Any]:
            """Differentiates both data operands while keeping the routing metadata fixed."""

            operand_tangent = (jax_numpy.arange(operand.size, dtype=jax_numpy.float32) % 5 + 1).reshape(operand.shape)
            seed_tangent = (jax_numpy.arange(seed.size, dtype=jax_numpy.float32) % 3 + 1).reshape(seed.shape)
            return jax.jvp(
                lambda operand, seed: function(operand, seed, *metadata),
                (operand, seed),
                (operand_tangent, seed_tangent),
            )
        primal, tangent = observed(*inputs)
        arrays = {"primal": primal, "tangent": tangent}
    elif transform == "vjp":
        def observed(operand: Any, seed: Any, *metadata: Any) -> tuple[Any, Any, Any]:
            """Returns the primal and both independently seeded data cotangents."""

            primal, pullback = jax.vjp(lambda operand, seed: function(operand, seed, *metadata), operand, seed)
            cotangent = jax_numpy.arange(1, primal.size + 1, dtype=jax_numpy.float32).reshape(primal.shape)
            operand_cotangent, seed_cotangent = pullback(cotangent)
            return primal, operand_cotangent, seed_cotangent
        primal, operand_cotangent, seed_cotangent = observed(*inputs)
        arrays = {"primal": primal, "operand_cotangent": operand_cotangent, "seed_cotangent": seed_cotangent}
    else:
        observed = function
        arrays = {"primal": function(*inputs)}
    expected = _reference_arrays(descriptor)
    validate_collective_arrays(descriptor["id"], arrays, {name: value.shape for name, value in expected.items()})
    return DifferentialObservation(
        SCHEMA,
        descriptor["id"],
        {
            name: tuple(tuple(float(element) for element in local.reshape(-1)) for local in numpy.asarray(value))
            for name, value in arrays.items()
        },
        stablehlo=str(jax.jit(observed).lower(*inputs).compiler_ir("stablehlo")),
    )


def validate_ragged_observation(descriptor: dict[str, Any], observation: DifferentialObservation) -> tuple[str, ...]:
    """Checks the executed ragged adjoint identity over the operand and the independent output seed."""

    import numpy

    if descriptor["transform"] != "vjp":
        return ()
    expected = ragged_reference(descriptor)
    if any(
        name not in observation.observations
        or tuple(map(len, observation.observations[name])) != tuple(map(len, values))
        for name, values in expected.items()
    ):
        return ("ragged adjoint identity requires all expected participant-local output shapes",)
    operand, seed, *_ = ragged_inputs(descriptor)
    primal = numpy.asarray(observation.observations["primal"], dtype=numpy.float64)
    cotangent = numpy.arange(1, primal.size + 1, dtype=numpy.float64).reshape(primal.shape)
    output_inner_product = float(numpy.sum(primal * cotangent))
    input_inner_product = sum(
        float(numpy.sum(
            value.reshape(-1) * numpy.asarray(observation.observations[name], dtype=numpy.float64).reshape(-1),
        ))
        for value, name in ((operand, "operand_cotangent"), (seed, "seed_cotangent"))
    )
    if input_inner_product != output_inner_product:
        return (
            f"ragged adjoint identity: output inner product {output_inner_product} differs from input inner product "
            f"{input_inner_product}",
        )
    return ()
