"""Two-process CUDA workloads with rank-dependent data and independent host references.

Workers execute actual distributed PJRT collectives after the launcher initializes JAX's distributed runtime. Each
worker supplies only its local input and copies only its addressable output; the launcher merges observations by rank.
"""

from __future__ import annotations

import json
import math
from typing import Any

from ryft.jax.collective_testing import collective_reference, validate_collective_arrays
from ryft.jax.differential_testing import SCHEMA, DifferentialObservation, StableHloCollective, repo_root


MANIFEST_PATH = repo_root() / "crates/ryft-xla/src/bin/differential_testing/distributed_collective_cases.json"


def distributed_descriptors() -> tuple[dict[str, Any], ...]:
    """Returns explicitly selected two-rank CUDA experiments shared with the Rust worker."""

    return tuple(json.loads(MANIFEST_PATH.read_text()))


def _data_arrays(descriptor: dict[str, Any]) -> tuple[Any, Any]:
    """Creates distinct float32 operand and output-seed values in rank order."""

    import numpy

    if descriptor["participants"] != 2:
        raise ValueError("distributed CUDA collective workloads require exactly two participants")
    input_shape = (2, *descriptor["local_shape"])
    output_shape = (2, *descriptor["output_shape"])
    operand = numpy.arange(1, math.prod(input_shape) + 1, dtype=numpy.float32).reshape(input_shape)
    seed = numpy.arange(100, 100 + math.prod(output_shape), dtype=numpy.float32).reshape(output_shape)
    return operand, seed


def distributed_inputs(descriptor: dict[str, Any], rank: int) -> tuple[Any, ...]:
    """Returns one rank's data and routing metadata with a size-one local mapped axis."""

    import numpy

    if rank not in (0, 1):
        raise ValueError("distributed CUDA collective rank must be zero or one")
    operand, seed = _data_arrays(descriptor)
    if descriptor["operation"] != "ragged_all_to_all":
        return (operand[rank:rank + 1],)
    metadata_type = {"i32": numpy.int32, "u64": numpy.uint64}[descriptor["metadata_type"]]
    metadata = tuple(
        numpy.asarray(descriptor[name][rank], dtype=metadata_type)[None, :]
        for name in ("input_offsets", "send_sizes", "output_offsets", "receive_sizes")
    )
    return operand[rank:rank + 1], seed[rank:rank + 1], *metadata


def distributed_reference(descriptor: dict[str, Any]) -> dict[str, tuple[tuple[float, ...], ...]]:
    """Evaluates all ranks on the host, including sender-owned ragged destination offsets."""

    if descriptor["operation"] != "ragged_all_to_all":
        operation = descriptor["operation"]
        translated = {**descriptor, "operation": "reduce" if operation == "sum" else operation, "tiled": True}
        if operation == "sum":
            translated["reduction"] = "sum"
        elif operation == "all_to_all":
            translated.update(split_axis=0, concatenation_axis=1)
        return collective_reference(translated)
    operand, result = _data_arrays(descriptor)
    metadata = [descriptor[name] for name in ("input_offsets", "send_sizes", "output_offsets", "receive_sizes")]
    input_offsets, send_sizes, output_offsets, receive_sizes = metadata
    lengths = {len(row) for values in metadata for row in values}
    if len(lengths) != 1 or not next(iter(lengths)) or next(iter(lengths)) % 2:
        raise ValueError("distributed ragged metadata lengths must agree and be divisible by two")
    updates_per_peer = next(iter(lengths)) // 2
    occupied: list[set[int]] = [set(), set()]
    for sender in range(2):
        for index, (source, size, destination) in enumerate(zip(
            input_offsets[sender], send_sizes[sender], output_offsets[sender], strict=True,
        )):
            receiver = index // updates_per_peer
            receive_index = sender * updates_per_peer + index % updates_per_peer
            if size != receive_sizes[receiver][receive_index] or min(source, size, destination) < 0:
                raise ValueError("distributed ragged send and receive metadata are inconsistent")
            if source + size > operand.shape[1] or destination + size > result.shape[1]:
                raise ValueError("distributed ragged transfer exceeds its data extent")
            rows = set(range(destination, destination + size))
            if occupied[receiver].intersection(rows):
                raise ValueError("distributed ragged received regions overlap")
            occupied[receiver].update(rows)
            result[receiver, destination:destination + size] = operand[sender, source:source + size]
    return {"primal": tuple(tuple(float(element) for element in local.reshape(-1)) for local in result)}


def distributed_contract(descriptor: dict[str, Any]) -> tuple[StableHloCollective, ...]:
    """Pins the global two-rank collective groups and operation-defining tensor axes."""

    operation = descriptor["operation"]
    groups = ((0, 1),)
    if operation == "sum":
        return (StableHloCollective("all_reduce", groups, ()),)
    if operation == "all_gather":
        return (StableHloCollective("all_gather", groups, (("all_gather_dim", 0),)),)
    if operation == "all_to_all":
        return (StableHloCollective(
            "all_to_all", groups, (("concat_dimension", 1), ("split_count", 2), ("split_dimension", 0)),
        ),)
    return ()


def build_distributed_jax(
    descriptor: dict[str, Any], rank: int, jax: Any, jax_numpy: Any, numpy: Any,
) -> DifferentialObservation:
    """Executes one collective over two initialized CUDA processes and copies the current rank's output."""

    if jax.process_count() != descriptor["participants"] or jax.process_index() != rank:
        raise RuntimeError("distributed CUDA collective worker has an inconsistent process topology")
    devices = jax.devices("cuda")
    if len(devices) != 2 or len(jax.local_devices(backend="cuda")) != 1:
        raise RuntimeError("distributed CUDA collective workers require one local GPU and two global GPUs")
    host_inputs = distributed_inputs(descriptor, rank)
    inputs = tuple(jax_numpy.asarray(value) for value in host_inputs)
    if any(numpy.dtype(value.dtype) != expected.dtype for value, expected in zip(inputs, host_inputs, strict=True)):
        raise RuntimeError("distributed CUDA collective input dtype changed; uint64 metadata require jax_enable_x64")

    def operation(operand: Any, *additional_inputs: Any) -> Any:
        """Keeps every data and routing input dynamic while binding the global participant axis."""

        operation_name = descriptor["operation"]
        if operation_name == "sum":
            return jax.lax.psum(operand, "x")
        if operation_name == "all_gather":
            return jax.lax.all_gather(operand, "x", axis=0, tiled=True)
        if operation_name == "all_to_all":
            return jax.lax.all_to_all(operand, "x", split_axis=0, concat_axis=1, tiled=True)
        if operation_name == "ragged_all_to_all":
            return jax.lax.ragged_all_to_all(operand, *additional_inputs, axis_name="x")
        raise ValueError(f"unknown distributed collective operation '{operation_name}'")

    # The device sequence is global and identical on both ranks, whereas each input's mapped axis is local size one.
    # JAX's multi-process pmap binds collectives over the complete device sequence and returns each process's shard.
    function = jax.pmap(operation, axis_name="x", devices=devices)
    primal = function(*inputs).block_until_ready()
    shards = primal.addressable_shards
    if len(shards) != 1:
        raise RuntimeError("distributed CUDA collective worker returned more than one addressable shard")
    local_output = shards[0].data
    validate_collective_arrays(
        descriptor["id"], {"primal": local_output}, {"primal": (1, *descriptor["output_shape"])},
    )
    values = numpy.asarray(local_output).reshape(-1)
    return DifferentialObservation(
        SCHEMA,
        descriptor["id"],
        {"primal": (tuple(float(element) for element in values),)},
        stablehlo=str(function.lower(*inputs).compiler_ir("stablehlo")),
    )


def build_distributed_case_jax(
    descriptor: dict[str, Any], jax: Any, jax_numpy: Any, numpy: Any,
) -> DifferentialObservation:
    """Adapts the shared registry callback to the initialized distributed worker's current rank."""

    return build_distributed_jax(descriptor, jax.process_index(), jax, jax_numpy, numpy)
