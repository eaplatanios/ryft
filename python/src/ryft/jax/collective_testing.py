"""Shared distributed workloads and independent host-array collective reference semantics.

Descriptors are also consumed by the Rust emitter. References explicitly move participant-local arrays without JAX,
XLA, or Ryft, so two backends making the same mistake cannot establish correctness by agreeing with one another.
"""

from __future__ import annotations

import json
import math
from typing import Any

from ryft.jax.differential_testing import SCHEMA, DifferentialObservation, StableHloCollective, repo_root


MANIFEST_PATH = repo_root() / "crates/ryft-xla/src/bin/differential_testing/collective_cases.json"


def collective_descriptors() -> tuple[dict[str, Any], ...]:
    """Returns the shared, ordered workload descriptors."""

    return tuple(json.loads(MANIFEST_PATH.read_text()))


def participant_groups(descriptor: dict[str, Any]) -> tuple[tuple[int, ...], ...]:
    """Expands axis-relative participant groups into flattened, row-major mesh device IDs."""

    import numpy

    shape = descriptor.get("mesh_shape", [descriptor["participants"]])
    axis = ("x", "y").index(descriptor.get("axis_name", "x"))
    groups = descriptor.get("groups", [list(range(shape[axis]))])
    devices = numpy.arange(descriptor["participants"]).reshape(shape)
    moved = numpy.moveaxis(devices, axis, -1).reshape(-1, shape[axis])
    return tuple(tuple(int(row[index]) for index in group) for row in moved for group in groups)


def input_arrays(descriptor: dict[str, Any]) -> Any:
    """Builds the same exact small float32 inputs as the Rust emitter."""

    import numpy

    shape = (descriptor["participants"], *descriptor["local_shape"])
    inputs = numpy.arange(1, math.prod(shape) + 1, dtype=numpy.float32).reshape(shape)
    if descriptor.get("replicated_input", False):
        inputs[:] = inputs[0]
    return inputs


def _primal_arrays(descriptor: dict[str, Any], inputs: Any) -> Any:
    """Evaluates collective semantics by indexing, reducing, and assembling ordinary host arrays."""

    import numpy

    operation = descriptor["operation"]
    if descriptor.get("transform") == "batch":
        unbatched = {**descriptor, "transform": "primal"}
        return numpy.stack([_primal_arrays(unbatched, inputs[:, index]) for index in range(inputs.shape[1])], axis=1)
    if operation == "vary":
        return inputs.copy()
    if operation == "axis_index":
        mesh_shape = descriptor.get("mesh_shape", [descriptor["participants"]])
        axis = ("x", "y").index(descriptor.get("axis_name", "x"))
        indices = numpy.asarray(
            [numpy.unravel_index(index, mesh_shape)[axis] for index in range(len(inputs))], dtype=numpy.float32
        )
        return numpy.broadcast_to(indices.reshape((len(inputs),) + (1,) * (inputs.ndim - 1)), inputs.shape).copy()
    if operation in ("permute", "shuffle"):
        output = numpy.zeros_like(inputs)
        pairs = descriptor.get("pairs")
        if operation == "shuffle":
            pairs = [(source, destination) for destination, source in enumerate(descriptor["permutation"])]
        for source, destination in pairs:
            output[destination] = inputs[source]
        return output
    output = [None] * len(inputs)
    for group in participant_groups(descriptor):
        values = [inputs[index] for index in group]
        tensor_axis = descriptor.get("tensor_axis", 0)
        if operation == "all_gather":
            combine = numpy.concatenate if descriptor.get("tiled", False) else numpy.stack
            gathered = combine(values, axis=tensor_axis)
            for participant in group:
                output[participant] = gathered.copy()
        elif operation == "reduce":
            reduction = {
                "sum": numpy.sum,
                "mean": numpy.mean,
                "min": numpy.min,
                "max": numpy.max,
                "product": numpy.prod,
            }[descriptor["reduction"]]
            reduced = reduction(numpy.stack(values), axis=0)
            for participant in group:
                output[participant] = reduced.copy()
        elif operation == "sum_scatter":
            reduced = numpy.sum(numpy.stack(values), axis=0)
            chunks = (
                numpy.split(reduced, len(group), axis=tensor_axis)
                if descriptor.get("tiled", False)
                else [numpy.take(reduced, index, axis=tensor_axis) for index in range(len(group))]
            )
            for participant, chunk in zip(group, chunks, strict=True):
                output[participant] = chunk
        elif operation in ("all_to_all", "swap_axes"):
            split_axis = descriptor.get("split_axis", tensor_axis)
            concatenation_axis = descriptor.get("concatenation_axis", tensor_axis)
            tiled = descriptor.get("tiled", False) if operation == "all_to_all" else False
            for recipient_index, participant in enumerate(group):
                chunks = [
                    numpy.split(value, len(group), axis=split_axis)[recipient_index]
                    if tiled
                    else numpy.take(value, recipient_index, axis=split_axis)
                    for value in values
                ]
                combine = numpy.concatenate if tiled else numpy.stack
                output[participant] = combine(chunks, axis=concatenation_axis)
        else:
            raise ValueError(f"unknown collective operation '{operation}'")
    return numpy.stack(output)


def _flatten_participants(values: Any) -> tuple[tuple[float, ...], ...]:
    """Flattens each participant's local output without discarding participant order."""

    return tuple(tuple(float(value) for value in local.reshape(-1)) for local in values)


def collective_reference(descriptor: dict[str, Any]) -> dict[str, tuple[tuple[float, ...], ...]]:
    """Returns independent primal and transform references for a shared workload.

    Reverse mode computes the exact transpose of the host-reference linear map on basis vectors. This deliberately
    avoids copying either framework's transpose rule, including its collective ordering or gradient scaling.
    """

    import numpy

    inputs = input_arrays(descriptor)
    primal = _primal_arrays(descriptor, inputs)
    observations = {"primal": _flatten_participants(primal)}
    transform = descriptor.get("transform", "primal")
    if transform == "jvp":
        tangent = (numpy.arange(inputs.size, dtype=numpy.float32) % 5 + 1).reshape(inputs.shape)
        observations["tangent"] = _flatten_participants(_primal_arrays(descriptor, tangent))
    elif transform == "vjp":
        seed = numpy.arange(1, primal.size + 1, dtype=numpy.float32).reshape(primal.shape)
        cotangent = numpy.zeros_like(inputs)
        for index in range(inputs.size):
            basis = numpy.zeros_like(inputs)
            basis.flat[index] = 1
            cotangent.flat[index] = numpy.sum(_primal_arrays(descriptor, basis) * seed)
        observations["cotangent"] = _flatten_participants(cotangent)
    return observations


def _jax_operation(descriptor: dict[str, Any], value: Any, jax: Any, jax_numpy: Any) -> Any:
    """Applies one descriptor inside an already bound manual participant axis."""

    operation = descriptor["operation"]
    axis_name = descriptor.get("axis_name", "x")
    groups = descriptor.get("groups")
    tensor_axis = descriptor.get("tensor_axis", 0)
    tiled = descriptor.get("tiled", False)
    if operation == "all_gather":
        return jax.lax.all_gather(value, axis_name, axis=tensor_axis, tiled=tiled, axis_index_groups=groups)
    if operation == "sum_scatter":
        return jax.lax.psum_scatter(
            value, axis_name, scatter_dimension=tensor_axis, tiled=tiled, axis_index_groups=groups
        )
    if operation in ("all_to_all", "swap_axes"):
        return jax.lax.all_to_all(
            value,
            axis_name,
            split_axis=descriptor.get("split_axis", tensor_axis),
            concat_axis=descriptor.get("concatenation_axis", tensor_axis),
            tiled=tiled if operation == "all_to_all" else False,
            axis_index_groups=groups,
        )
    if operation == "reduce":
        reduction = {
            "sum": jax.lax.psum, "mean": jax.lax.pmean, "min": jax.lax.pmin, "max": jax.lax.pmax,
        }[descriptor["reduction"]]
        return reduction(value, axis_name, axis_index_groups=groups)
    if operation == "permute":
        return jax.lax.ppermute(value, axis_name, descriptor["pairs"])
    if operation == "shuffle":
        return jax.lax.pshuffle(value, axis_name, descriptor["permutation"])
    if operation == "axis_index":
        return jax_numpy.full_like(value, jax.lax.axis_index(axis_name), dtype=jax_numpy.float32)
    if operation == "vary":
        return jax.lax.pcast(value, axis_name, to="varying")
    raise ValueError(f"unknown collective operation '{operation}'")


def validate_collective_arrays(
    case_id: str, outputs: dict[str, Any], expected_shapes: dict[str, tuple[int, ...]]
) -> None:
    """Validates shape and float32 element types of executed outputs before flattening."""

    import numpy

    for name, expected_shape in expected_shapes.items():
        actual_shape = tuple(outputs[name].shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"collective '{case_id}' {name} shape {actual_shape} differs from reference {expected_shape}"
            )
        actual_dtype = numpy.dtype(outputs[name].dtype)
        if actual_dtype != numpy.dtype(numpy.float32):
            raise ValueError(f"collective '{case_id}' {name} dtype {actual_dtype} differs from reference float32")


def validate_collective_shapes(descriptor: dict[str, Any], outputs: dict[str, Any]) -> None:
    """Validates executed tensor shapes before participant-local flattening can hide a reshape error."""

    inputs = input_arrays(descriptor)
    expected_primal_shape = _primal_arrays(descriptor, inputs).shape
    expected_shapes = {"primal": expected_primal_shape}
    transform = descriptor.get("transform", "primal")
    if transform == "jvp":
        expected_shapes["tangent"] = expected_primal_shape
    elif transform == "vjp":
        expected_shapes["cotangent"] = inputs.shape
    validate_collective_arrays(descriptor["id"], outputs, expected_shapes)


def build_collective_jax(
    descriptor: dict[str, Any], jax: Any, jax_numpy: Any, numpy: Any
) -> DifferentialObservation:
    """Executes the shared collective workload on pinned JAX's logical CPU devices."""

    inputs = jax_numpy.asarray(input_arrays(descriptor))
    operation_descriptor = descriptor
    # JAX has no public parallel product reduction. Gather values and reduce locally, preserving the same semantics.
    if descriptor["operation"] == "reduce" and descriptor["reduction"] == "product":
        def operation(value: Any) -> Any:
            gathered = jax.lax.all_gather(
                value, descriptor.get("axis_name", "x"), axis_index_groups=descriptor.get("groups"),
            )
            return jax_numpy.prod(gathered, axis=0)
    else:
        def operation(value: Any) -> Any:
            return _jax_operation(operation_descriptor, value, jax, jax_numpy)

    if descriptor.get("transform") == "batch":
        operation = jax.vmap(operation)
    mesh_shape = descriptor.get("mesh_shape", [descriptor["participants"]])
    if len(mesh_shape) == 1:
        function = jax.pmap(operation, axis_name="x", devices=jax.devices("cpu")[: descriptor["participants"]])
    else:
        names = ("x", "y")
        devices = numpy.asarray(jax.devices("cpu")[: descriptor["participants"]]).reshape(mesh_shape)
        mesh = jax.sharding.Mesh(devices, names)
        prefix = (1,) * len(mesh_shape)
        specification = jax.sharding.PartitionSpec(*names)

        def mapped(value: Any) -> Any:
            local = value.reshape(descriptor["local_shape"])
            result = operation(local)
            return result.reshape((*prefix, *result.shape))

        mapped_function = jax.jit(jax.shard_map(
            mapped, mesh=mesh, in_specs=specification, out_specs=specification, check_vma=False,
        ))

        def function(value: Any) -> Any:
            result = mapped_function(value.reshape((*mesh_shape, *descriptor["local_shape"])))
            return result.reshape((descriptor["participants"], *result.shape[len(mesh_shape):]))

    transform = descriptor.get("transform", "primal")
    if transform == "jvp":
        def observed(value: Any) -> tuple[Any, Any]:
            tangent = (jax_numpy.arange(value.size, dtype=jax_numpy.float32) % 5 + 1).reshape(value.shape)
            return jax.jvp(function, (value,), (tangent,))
        primal, tangent = observed(inputs)
        validate_collective_shapes(descriptor, {"primal": primal, "tangent": tangent})
        observations = {
            "primal": _flatten_participants(numpy.asarray(primal)),
            "tangent": _flatten_participants(numpy.asarray(tangent)),
        }
    elif transform == "vjp":
        def observed(value: Any) -> tuple[Any, Any]:
            primal, pullback = jax.vjp(function, value)
            seed = jax_numpy.arange(1, primal.size + 1, dtype=jax_numpy.float32).reshape(primal.shape)
            return primal, pullback(seed)[0]
        primal, cotangent = observed(inputs)
        validate_collective_shapes(descriptor, {"primal": primal, "cotangent": cotangent})
        observations = {
            "primal": _flatten_participants(numpy.asarray(primal)),
            "cotangent": _flatten_participants(numpy.asarray(cotangent)),
        }
    else:
        observed = function
        primal = function(inputs)
        validate_collective_shapes(descriptor, {"primal": primal})
        observations = {"primal": _flatten_participants(numpy.asarray(primal))}
    return DifferentialObservation(
        schema=SCHEMA,
        case_id=descriptor["id"],
        observations=observations,
        stablehlo=str(jax.jit(observed).lower(inputs).compiler_ir("stablehlo")),
    )


def collective_contract(descriptor: dict[str, Any]) -> tuple[StableHloCollective, ...]:
    """Returns exact primal collective attributes for cases with a shared direct lowering."""

    operation = descriptor["operation"]
    if descriptor.get("transform", "primal") != "primal" or descriptor["participants"] == 1:
        return ()
    groups = participant_groups(descriptor)
    tensor_axis = descriptor.get("tensor_axis", 0)
    if operation == "all_gather":
        return (StableHloCollective("all_gather", groups, (("all_gather_dim", tensor_axis),)),)
    if operation == "sum_scatter":
        return (StableHloCollective("reduce_scatter", groups, (("scatter_dimension", tensor_axis),)),)
    if operation in ("all_to_all", "swap_axes"):
        split_axis = descriptor.get("split_axis", tensor_axis)
        concatenation_axis = descriptor.get("concatenation_axis", tensor_axis)
        # Untiled all-to-all inserts a singleton at the concatenation axis before communication.
        if not descriptor.get("tiled", False) and split_axis != concatenation_axis:
            split_axis += int(concatenation_axis <= split_axis)
        return (StableHloCollective("all_to_all", groups, (
            ("concat_dimension", concatenation_axis), ("split_count", len(groups[0])), ("split_dimension", split_axis)
        )),)
    if operation == "reduce" and descriptor["reduction"] != "product":
        return (StableHloCollective("all_reduce", groups, ()),)
    if operation in ("permute", "shuffle"):
        pairs = descriptor.get("pairs")
        if operation == "shuffle":
            pairs = [(source, destination) for destination, source in enumerate(descriptor["permutation"])]
        return (StableHloCollective("collective_permute", tuple(tuple(pair) for pair in pairs), ()),)
    return ()


def validate_collective_observation(
    descriptor: dict[str, Any], observation: DifferentialObservation
) -> tuple[str, ...]:
    """Checks the adjoint identity against actual executed primal and reverse-mode outputs."""

    import numpy

    if descriptor.get("transform") != "vjp":
        return ()
    expected = collective_reference(descriptor)
    for name in ("primal", "cotangent"):
        actual_values = observation.observations.get(name)
        expected_values = expected[name]
        if actual_values is None or len(actual_values) != len(expected_values) or any(
            len(actual) != len(reference) for actual, reference in zip(actual_values, expected_values, strict=True)
        ):
            return (f"adjoint identity requires the expected participant-local {name} shapes",)
    primal = numpy.asarray(observation.observations["primal"], dtype=numpy.float64)
    cotangent = numpy.asarray(observation.observations["cotangent"], dtype=numpy.float64)
    seed = numpy.arange(1, primal.size + 1, dtype=numpy.float64).reshape(primal.shape)
    inputs = input_arrays(descriptor).astype(numpy.float64).reshape(cotangent.shape)
    output_inner_product = float(numpy.sum(primal * seed))
    input_inner_product = float(numpy.sum(inputs * cotangent))
    if output_inner_product != input_inner_product:
        return (
            f"adjoint identity: output inner product {output_inner_product} differs from input inner product "
            f"{input_inner_product}",
        )
    return ()


def legacy_collective_reference(case_id: str) -> dict[str, tuple[tuple[float, ...], ...]]:
    """Adds independent host references to the original three collective workload IDs."""

    if case_id == "grouped_shape_changing_collectives":
        descriptor = dict(participants=4, local_shape=[4], groups=[[0, 2], [3, 1]], tiled=True)
        inputs = input_arrays(descriptor) - 1
        return {
            "all_gather": _flatten_participants(_primal_arrays({**descriptor, "operation": "all_gather"}, inputs)),
            "psum_scatter": _flatten_participants(_primal_arrays({**descriptor, "operation": "sum_scatter"}, inputs)),
            "all_to_all": _flatten_participants(_primal_arrays({**descriptor, "operation": "all_to_all"}, inputs)),
        }
    if case_id == "pshuffle":
        descriptor = dict(operation="shuffle", participants=4, local_shape=[2], permutation=[2, 0, 3, 1])
    elif case_id == "pswapaxes":
        descriptor = dict(operation="swap_axes", participants=4, local_shape=[4, 2], tensor_axis=0)
    else:
        raise ValueError(f"unknown legacy collective case '{case_id}'")
    return {"output": _flatten_participants(_primal_arrays(descriptor, input_arrays(descriptor) - 1))}
