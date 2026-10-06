"""Executable behavioral and StableHLO differential testing between Ryft and pinned JAX.

The Rust side is emitted by `cargo run -p ryft-xla --features differential-testing --bin differential_testing`.
The JAX side runs in a fresh process so that its selected backend is configured before JAX is imported. Exact
parity cases compare named values and declared semantic StableHLO contracts. Collective cases additionally compare
each backend against an independent host reference. The bounded data-dependent case instead
encodes an explicit capability relation: eager values agree, Ryft stages the bounded result, and pinned JAX rejects
staging because the result extent depends on traced data.

This module intentionally does not import JAX at module load time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence


if TYPE_CHECKING:
    from ryft.jax.differential_testing_cases import DifferentialCase


SCHEMA = "ryft-jax-differential-v1"
PINNED_JAX_VERSION = "0.10.0"
BUILD_HINT = "cargo build -p ryft-xla --features differential-testing --bin differential_testing"
SUBPROCESS_TIMEOUT_SECONDS = 300
CUDA_SUITES = ("cuda-collectives", "cuda-distributed-collectives")


@dataclass(frozen=True)
class StagingObservation:
    """One framework's staging result for a differential case."""

    status: str
    output_type: str | None = None
    category: str | None = None


@dataclass(frozen=True)
class DifferentialObservation:
    """One framework's observations for one shared differential case."""

    schema: str
    case_id: str
    observations: dict[str, tuple[tuple[float, ...], ...]]
    staging: StagingObservation | None = None
    stablehlo: str | None = None


@dataclass(frozen=True)
class StableHloCollective:
    """Semantic StableHLO fields compared for one collective operation."""

    operation: str
    groups: tuple[tuple[int, ...], ...]
    axis_attributes: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class CaseComparison:
    """Comparison result for one Ryft/JAX case pair."""

    case_id: str
    differences: tuple[str, ...]

    def passed(self) -> bool:
        """Returns whether the case produced no differences."""

        return not self.differences


def repo_root() -> Path:
    """Returns the repository root containing the `python` and `crates` directories."""

    return Path(__file__).resolve().parents[4]


def _required_mapping(value: Any, path: str) -> Mapping[str, Any]:
    """Returns `value` as a mapping or raises a path-specific schema error."""

    if not isinstance(value, Mapping):
        raise ValueError(f"field '{path}' must be an object")
    return value


def _required_string(value: Any, path: str) -> str:
    """Returns `value` as a string or raises a path-specific schema error."""

    if not isinstance(value, str):
        raise ValueError(f"field '{path}' must be a string")
    return value


def _parse_observations(value: Any) -> dict[str, tuple[tuple[float, ...], ...]]:
    """Parses named floating-point observations from one record payload."""

    observations = _required_mapping(value, "observations")
    parsed: dict[str, tuple[tuple[float, ...], ...]] = {}
    for name, executions in observations.items():
        name = _required_string(name, "observations key")
        if not isinstance(executions, Sequence) or isinstance(executions, (str, bytes)):
            raise ValueError(f"field 'observations.{name}' must be an array")
        parsed_executions = []
        for execution_index, execution in enumerate(executions):
            if not isinstance(execution, Sequence) or isinstance(execution, (str, bytes)):
                raise ValueError(f"field 'observations.{name}[{execution_index}]' must be an array")
            values = []
            for value_index, element in enumerate(execution):
                if isinstance(element, bool) or not isinstance(element, (int, float)):
                    raise ValueError(
                        f"field 'observations.{name}[{execution_index}][{value_index}]' must be numeric"
                    )
                if not math.isfinite(element):
                    raise ValueError(
                        f"field 'observations.{name}[{execution_index}][{value_index}]' must be finite"
                    )
                values.append(float(element))
            parsed_executions.append(tuple(values))
        parsed[name] = tuple(parsed_executions)
    return parsed


def _parse_staging(value: Any) -> StagingObservation:
    """Parses one staging observation and validates its status-specific fields."""

    fields = _required_mapping(value, "staging")
    status = _required_string(fields.get("status"), "staging.status")
    output_type = fields.get("output_type")
    category = fields.get("category")
    if status == "supported":
        return StagingObservation(status=status, output_type=_required_string(output_type, "staging.output_type"))
    if status == "rejected":
        return StagingObservation(status=status, category=_required_string(category, "staging.category"))
    raise ValueError(f"field 'staging.status' has unsupported value '{status}'")


def parse_observation(value: Any) -> DifferentialObservation:
    """Parses and validates one versioned differential observation record."""

    fields = _required_mapping(value, "record")
    schema = _required_string(fields.get("schema"), "schema")
    if schema != SCHEMA:
        raise ValueError(f"unsupported differential observation schema '{schema}'")
    stablehlo = fields.get("stablehlo")
    if stablehlo is not None:
        stablehlo = _required_string(stablehlo, "stablehlo")
    staging = fields.get("staging")
    return DifferentialObservation(
        schema=schema,
        case_id=_required_string(fields.get("case_id"), "case_id"),
        observations=_parse_observations(fields.get("observations")),
        staging=None if staging is None else _parse_staging(staging),
        stablehlo=stablehlo,
    )


def observation_payload(observation: DifferentialObservation) -> dict[str, Any]:
    """Returns the JSON-serializable payload for one observation."""

    payload = asdict(observation)
    if observation.staging is None:
        payload.pop("staging")
    else:
        payload["staging"] = {name: value for name, value in payload["staging"].items() if value is not None}
    if observation.stablehlo is None:
        payload.pop("stablehlo")
    return payload


_COLLECTIVE_PATTERN = re.compile(
    r'"stablehlo\.(all_reduce|all_gather|reduce_scatter|all_to_all|collective_permute)"[^\n]*'
)
_DENSE_GROUPS_PATTERN = re.compile(
    r"(?:replica_groups|source_target_pairs) = dense<(\[\[.*?\]\]|-?\d+)> : tensor<(\d+)x(\d+)xi64>"
)
_AXIS_PATTERNS = {
    "all_gather_dim": re.compile(r"all_gather_dim = (\d+) : i64"),
    "scatter_dimension": re.compile(r"scatter_dimension = (\d+) : i64"),
    "concat_dimension": re.compile(r"concat_dimension = (\d+) : i64"),
    "split_count": re.compile(r"split_count = (\d+) : i64"),
    "split_dimension": re.compile(r"split_dimension = (\d+) : i64"),
}


def project_collective_stablehlo(module: str) -> tuple[StableHloCollective, ...]:
    """Projects collective semantics from a StableHLO module.

    The projection intentionally ignores SSA names, channel handles, tensor spellings, wrapper functions, and Shardy
    metadata. It retains the collective family, ordered replica/source-target groups, and operation-defining axis
    attributes. Independent numerical references also check reduction combiners and tensor layouts.

    # Parameters

      - `module`: Textual StableHLO module emitted by Ryft or JAX.
    """

    collectives = []
    for match in _COLLECTIVE_PATTERN.finditer(module):
        operation = match.group(1)
        operation_text = match.group(0)
        groups_match = _DENSE_GROUPS_PATTERN.search(operation_text)
        if groups_match is None:
            raise ValueError(f"stablehlo.{operation} is missing replica/source-target groups")
        groups_payload = json.loads(groups_match.group(1))
        if isinstance(groups_payload, int):
            groups = tuple(
                (groups_payload,) * int(groups_match.group(3)) for _ in range(int(groups_match.group(2)))
            )
        else:
            groups = tuple(tuple(int(member) for member in group) for group in groups_payload)
        axis_attributes = tuple(
            (name, int(axis_match.group(1)))
            for name, pattern in _AXIS_PATTERNS.items()
            if (axis_match := pattern.search(operation_text)) is not None
        )
        collectives.append(
            StableHloCollective(operation=operation, groups=groups, axis_attributes=axis_attributes)
        )
    return tuple(collectives)


def differential_cases() -> tuple[DifferentialCase, ...]:
    """Returns the shared case registry without importing JAX eagerly."""

    from ryft.jax.differential_testing_cases import DIFFERENTIAL_CASES

    return DIFFERENTIAL_CASES


def _compare_projection(
    side: str,
    module: str,
    collectives: tuple[StableHloCollective, ...],
) -> str | None:
    """Compares one framework's projected collectives against the registry expectation.

    Anchoring both frameworks to a declared expectation is what keeps the parity gate honest: two modules that no
    longer contain any recognizable collective project to the same empty tuple and would otherwise agree vacuously.

    # Parameters

      - `side`: Framework name used in the returned difference.
      - `module`: Textual StableHLO module emitted by that framework.
      - `collectives`: Expected projection declared by the registry entry.
    """

    projection = project_collective_stablehlo(module)
    missing = [
        operation
        for operation in dict.fromkeys(collective.operation for collective in collectives)
        if all(projected.operation != operation for projected in projection)
    ]
    if missing:
        return f"StableHLO collectives: {side} module is missing expected collective families {', '.join(missing)}"
    if projection != collectives:
        return f"StableHLO collectives: {side} {projection!r} != expected {collectives!r}"
    return None


def compare_case(
    relationship: str,
    collectives: tuple[StableHloCollective, ...],
    ryft: DifferentialObservation,
    jax: DifferentialObservation,
    stablehlo_patterns: tuple[str, ...] = (),
    reference: Mapping[str, tuple[tuple[float, ...], ...]] | None = None,
) -> CaseComparison:
    """Compares one Ryft/JAX record pair according to its declared capability relationship.

    # Parameters

      - `relationship`: Either `parity` or `ryft_exceeds_jax`.
      - `collectives`: Collective projection the registry entry expects from both frameworks.
      - `ryft`: Ryft-side observation.
      - `jax`: JAX-side observation.
      - `stablehlo_patterns`: Semantic operation or attribute spellings that both StableHLO modules must contain.
      - `reference`: Independent host-computed outputs checked separately against each framework.
    """

    differences = []
    if ryft.case_id != jax.case_id:
        differences.append(f"case ID: ryft '{ryft.case_id}' != jax '{jax.case_id}'")
    if ryft.observations != jax.observations:
        differences.append(f"observations: ryft {ryft.observations!r} != jax {jax.observations!r}")
    if reference is not None:
        for side, observation in (("ryft", ryft), ("jax", jax)):
            if observation.observations != reference:
                differences.append(
                    f"reference observations: {side} {observation.observations!r} != expected {reference!r}"
                )
    if relationship == "parity":
        if ryft.staging != jax.staging:
            differences.append(f"staging: ryft {ryft.staging!r} != jax {jax.staging!r}")
        if not collectives and not stablehlo_patterns and reference is None:
            differences.append("StableHLO: exact-parity case must declare a semantic contract")
        if ryft.stablehlo is None or jax.stablehlo is None:
            differences.append("StableHLO: exact-parity case requires modules from both frameworks")
        else:
            if collectives:
                differences.extend(
                    difference
                    for difference in (
                        _compare_projection("ryft", ryft.stablehlo, collectives),
                        _compare_projection("jax", jax.stablehlo, collectives),
                    )
                    if difference is not None
                )
            for side, module in (("Ryft", ryft.stablehlo), ("JAX", jax.stablehlo)):
                missing = tuple(pattern for pattern in stablehlo_patterns if pattern not in module)
                if missing:
                    differences.append(f"StableHLO: {side} module is missing semantic patterns {missing!r}")
    elif relationship == "ryft_exceeds_jax":
        if ryft.staging != StagingObservation(status="supported", output_type="f32[count]"):
            differences.append(f"Ryft staging: expected bounded symbolic support but got {ryft.staging!r}")
        if jax.staging != StagingObservation(status="rejected", category="concretization"):
            differences.append(f"JAX staging: expected concretization rejection but got {jax.staging!r}")
        if collectives or ryft.stablehlo is not None or jax.stablehlo is not None:
            differences.append("StableHLO: the staging-rejected capability case must not claim a module comparison")
    else:
        raise ValueError(f"unknown differential relationship '{relationship}'")
    return CaseComparison(case_id=ryft.case_id, differences=tuple(differences))


def _selected_cases(case_ids: Sequence[str], suite: str | None = None) -> tuple[DifferentialCase, ...]:
    """Returns registry entries selected by suite and exact case ID, preserving registry order."""

    cases = differential_cases()
    unknown = [case_id for case_id in case_ids if all(case.case_id != case_id for case in cases)]
    if unknown:
        raise ValueError(f"unknown differential-testing case '{unknown[0]}'")
    if suite is not None:
        cases = tuple(case for case in cases if case.suite == suite)
    if not case_ids:
        return tuple(case for case in cases if suite is not None or case.suite not in CUDA_SUITES)
    selected = set(case_ids)
    outside_suite = selected.difference(case.case_id for case in cases)
    if outside_suite:
        raise ValueError(f"case '{sorted(outside_suite)[0]}' is not in suite '{suite}'")
    cases = tuple(case for case in cases if case.case_id in selected)
    if any(case.suite in CUDA_SUITES for case in cases) and any(case.suite not in CUDA_SUITES for case in cases):
        raise ValueError("CPU and CUDA cases must run in separate invocations")
    if any(case.suite == "cuda-distributed-collectives" for case in cases) and any(
        case.suite != "cuda-distributed-collectives" for case in cases
    ):
        raise ValueError("single-process and distributed CUDA cases must run in separate invocations")
    return cases


@contextmanager
def _shared_gpu_environment():
    """Owns a private MPS daemon unless the caller supplies an existing MPS pipe directory.

    Both ranks need concurrent progress on the same GPU. Restrict each to one NCCL CTA and less than half the GPU's
    active threads, and avoid allocator preallocation. Only a daemon started here is shut down here.
    """

    environment = os.environ.copy()
    environment.update({
        "NCCL_MULTI_RANK_GPU_ENABLE": "1", "NCCL_NVLS_ENABLE": "0", "NCCL_MAX_CTAS": "1",
        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "45", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    })
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
    environment.setdefault("NCCL_DEBUG", "INFO")
    if environment.get("CUDA_MPS_PIPE_DIRECTORY"):
        yield environment
        return
    control = shutil.which("nvidia-cuda-mps-control")
    if control is None:
        raise RuntimeError("distributed CUDA suite requires `nvidia-cuda-mps-control` or an existing MPS pipe directory")
    with tempfile.TemporaryDirectory(prefix="ryft-cuda-mps-", delete=False) as directory:
        pipes, logs = Path(directory) / "pipes", Path(directory) / "logs"
        pipes.mkdir()
        logs.mkdir()
        environment.update({"CUDA_MPS_PIPE_DIRECTORY": str(pipes), "CUDA_MPS_LOG_DIRECTORY": str(logs)})
        failure = None
        try:
            try:
                subprocess.run([control, "-d"], env=environment, capture_output=True, text=True, check=True, timeout=10)
            except subprocess.SubprocessError as error:
                raise RuntimeError(f"private CUDA MPS startup failed: {error}") from error
            yield environment
        except BaseException as error:
            failure = error
            raise
        finally:
            # Even a timed-out `-d` invocation may have forked a daemon. Attempt private shutdown after startup
            # failures too; retain its directories when shutdown fails so the daemon remains addressable.
            try:
                subprocess.run(
                    [control], input="quit\n", env=environment, capture_output=True, text=True, check=True, timeout=10,
                )
            except (OSError, subprocess.SubprocessError) as error:
                message = f"private CUDA MPS shutdown failed: {error}; retained control directory: {directory}"
                if failure is None:
                    raise RuntimeError(message) from error
                raise RuntimeError(f"{failure}; {message}") from failure
            else:
                shutil.rmtree(directory)


def _run_distributed_emitters(
    command: Sequence[str], directory: Path, timeout: int, side: str, log_directory: Path | None = None,
) -> tuple[str, str]:
    """Runs both ranks concurrently, cancelling their process groups on any failure or timeout.

    File-backed output prevents a verbose worker from blocking on a full pipe while the other waits in a collective.
    The timeout covers the whole launch. An optional directory retains each rank's stdout/stderr on success as well.
    """

    with socket.socket() as endpoint:
        endpoint.bind(("127.0.0.1", 0))
        coordinator = f"127.0.0.1:{endpoint.getsockname()[1]}"
    with tempfile.TemporaryDirectory(prefix="ryft-distributed-") as temporary, ExitStack() as stack:
        logs = Path(temporary) if log_directory is None else log_directory.resolve()
        logs.mkdir(parents=True, exist_ok=True)
        outputs, errors, processes = [], [], []
        with _shared_gpu_environment() as environment:
            try:
                for rank in range(2):
                    output = stack.enter_context((logs / f"{side}-rank-{rank}.stdout.log").open("w+"))
                    error = stack.enter_context((logs / f"{side}-rank-{rank}.stderr.log").open("w+"))
                    outputs.append(output)
                    errors.append(error)
                    worker_command = [*command, "--distributed-worker", str(rank), "--coordinator", coordinator]
                    # NCCL's default INFO destination is stdout, which must remain a strict JSON channel.
                    worker_environment = {**environment, "NCCL_DEBUG_FILE": str(logs / f"{side}-rank-{rank}.nccl.log")}
                    processes.append(subprocess.Popen(
                        worker_command, cwd=directory, env=worker_environment, stdout=output, stderr=error,
                        text=True, start_new_session=os.name == "posix",
                    ))
                deadline = time.monotonic() + timeout
                while any(process.poll() is None for process in processes):
                    if any(process.poll() not in (None, 0) for process in processes):
                        raise RuntimeError("a worker exited unsuccessfully")
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"workers timed out after {timeout} seconds")
                    time.sleep(0.02)
                if any(process.returncode != 0 for process in processes):
                    raise RuntimeError("a worker exited unsuccessfully")
            except (OSError, RuntimeError) as failure:
                diagnostics = []
                for rank, error in enumerate(errors):
                    error.seek(0)
                    diagnostics.append(f"rank {rank}:\n{error.read()[-65536:]}")
                retained = "" if log_directory is None else f"\nworker logs: {logs}"
                raise RuntimeError(f"{side} distributed emitter failed: {failure}\n" + "\n".join(diagnostics) + retained) from failure
            finally:
                for process in processes:
                    if os.name == "posix":
                        # A failed worker can leave live children even after its own exit.
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    elif process.poll() is None:
                        process.kill()
                    process.wait()
            records = []
            for output in outputs:
                output.seek(0)
                records.append(output.read())
            return records[0], records[1]


def _merge_distributed_observations(
    payloads: Sequence[str], case_ids: Sequence[str], side: str,
) -> tuple[DifferentialObservation, ...]:
    """Combines one validated local observation per rank in deterministic global rank order."""

    if len(payloads) != 2:
        raise ValueError(f"{side} distributed emitter must produce exactly two rank payloads")
    ranks = []
    for payload in payloads:
        records = json.loads(payload)
        if not isinstance(records, list):
            raise ValueError(f"{side} distributed worker must return a JSON array")
        indexed = _record_map(tuple(parse_observation(record) for record in records), side)
        if set(indexed) != set(case_ids):
            raise ValueError(f"{side} distributed rank case set does not match the selected cases")
        ranks.append(indexed)
    merged = []
    for case_id in case_ids:
        first, second = (rank[case_id] for rank in ranks)
        if first.staging is not None or second.staging is not None or first.stablehlo != second.stablehlo:
            raise ValueError(f"{side} distributed ranks disagree on the module contract for `{case_id}`")
        if set(first.observations) != set(second.observations) or any(
            len(record.observations[name]) != 1 for record in (first, second) for name in record.observations
        ):
            raise ValueError(f"{side} distributed ranks must emit matching names and one local output each")
        merged.append(DifferentialObservation(
            SCHEMA, case_id,
            {name: first.observations[name] + second.observations[name] for name in first.observations},
            stablehlo=first.stablehlo,
        ))
    return tuple(merged)


def _run_emitter(command: Sequence[str], directory: Path, timeout: int) -> str:
    """Captures one emitter, terminating its process group if compilation or execution exceeds its deadline.

    A `cargo run` invocation can own a compiler or an executing test child. Killing the launcher alone would leave
    that child running after a timeout, so POSIX launches receive their own process group.
    """

    with subprocess.Popen(
        command, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=os.name == "posix",
    ) as process:
        try:
            output, errors = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    # The process group can exit between the timeout and cancellation.
                    pass
            else:
                process.kill()
            process.communicate()
            raise
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command, output=output, stderr=errors)
        return output


def collect_ryft_observations(
    root: Path,
    case_ids: Sequence[str],
    *,
    timeout: int = SUBPROCESS_TIMEOUT_SECONDS,
    binary: Path | None = None,
    log_directory: Path | None = None,
) -> tuple[DifferentialObservation, ...]:
    """Runs the Rust emitter and parses its selected records."""

    cases = _selected_cases(case_ids)
    features = (
        "differential-testing,cuda-13" if any(case.suite in CUDA_SUITES for case in cases)
        else "differential-testing"
    )
    command = [str(binary.resolve())] if binary is not None else [
        "cargo",
        "run",
        "--quiet",
        "-p",
        "ryft-xla",
        "--features",
        features,
        "--bin",
        "differential_testing",
        "--",
    ]
    distributed = any(case.suite == "cuda-distributed-collectives" for case in cases)
    if distributed and binary is None:
        # Build once before rendezvous; concurrent `cargo run` workers can serialize on Cargo's build lock.
        build = ["cargo", "build", "--quiet", "-p", "ryft-xla", "--features", features,
                 "--bin", "differential_testing", "--message-format=json"]
        try:
            artifacts = [json.loads(line) for line in _run_emitter(build, root, timeout).splitlines() if line.strip()]
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Ryft distributed emitter build failed:\n{error.stderr.strip()}") from None
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Ryft distributed emitter build timed out after {timeout} seconds; pre-build with CUDA") from None
        executables = [artifact["executable"] for artifact in artifacts if artifact.get("executable")]
        if len(executables) != 1:
            raise RuntimeError("Ryft distributed emitter build must produce exactly one executable")
        command = [executables[0]]
    for case_id in case_ids:
        command.extend(("--case", case_id))
    if distributed:
        payloads = _run_distributed_emitters(command, root, timeout, "ryft", log_directory)
        return _merge_distributed_observations(payloads, case_ids, "Ryft")
    try:
        output = _run_emitter(command, root, timeout)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Ryft differential emitter failed:\n{error.stderr.strip()}") from None
    except subprocess.TimeoutExpired:
        build_hint = BUILD_HINT.replace("--features differential-testing", f"--features {features}")
        raise RuntimeError(
            f"Ryft differential emitter timed out after {timeout} seconds. A cold compile of the "
            f"emitter can exceed that budget; pre-build it with '{build_hint}' and rerun."
        ) from None
    payload = json.loads(output)
    if not isinstance(payload, list):
        raise ValueError("Ryft differential emitter must return a JSON array")
    return tuple(parse_observation(record) for record in payload)


def collect_jax_observations(
    root: Path,
    case_ids: Sequence[str],
    *,
    timeout: int = SUBPROCESS_TIMEOUT_SECONDS,
    log_directory: Path | None = None,
) -> tuple[DifferentialObservation, ...]:
    """Runs the JAX emitters in a fresh process and parses their selected records."""

    command = [sys.executable, "-m", "ryft.jax.differential_testing", "--emit-jax"]
    for case_id in case_ids:
        command.extend(("--case", case_id))
    if any(case.suite == "cuda-distributed-collectives" for case in _selected_cases(case_ids)):
        payloads = _run_distributed_emitters(command, root / "python", timeout, "jax", log_directory)
        return _merge_distributed_observations(payloads, case_ids, "JAX")
    try:
        output = _run_emitter(command, root / "python", timeout)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"JAX differential emitter failed:\n{error.stderr.strip()}") from None
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"JAX differential emitter timed out after {timeout} seconds"
        ) from None
    payload = json.loads(output)
    if not isinstance(payload, list):
        raise ValueError("JAX differential emitter must return a JSON array")
    return tuple(parse_observation(record) for record in payload)


def _record_map(records: Sequence[DifferentialObservation], side: str) -> dict[str, DifferentialObservation]:
    """Indexes records by case ID while rejecting duplicates."""

    indexed = {}
    for record in records:
        if record.case_id in indexed:
            raise ValueError(f"{side} emitted duplicate case '{record.case_id}'")
        indexed[record.case_id] = record
    return indexed


def run_comparison(
    root: Path,
    case_ids: Sequence[str],
    *,
    timeout: int = SUBPROCESS_TIMEOUT_SECONDS,
    binary: Path | None = None,
    log_directory: Path | None = None,
) -> tuple[CaseComparison, ...]:
    """Runs both frameworks and returns comparisons for the selected registry entries."""

    cases = _selected_cases(case_ids)
    selected_ids = tuple(case.case_id for case in cases)
    ryft = _record_map(collect_ryft_observations(
        root, selected_ids, timeout=timeout, binary=binary, log_directory=log_directory,
    ), "Ryft")
    jax = _record_map(collect_jax_observations(root, selected_ids, timeout=timeout, log_directory=log_directory), "JAX")
    if set(ryft) != set(selected_ids):
        raise ValueError(f"Ryft case set {sorted(ryft)} does not match selected cases {sorted(selected_ids)}")
    if set(jax) != set(selected_ids):
        raise ValueError(f"JAX case set {sorted(jax)} does not match selected cases {sorted(selected_ids)}")
    comparisons = []
    for case in cases:
        comparison = compare_case(
            case.relationship,
            case.collectives,
            ryft[case.case_id],
            jax[case.case_id],
            case.stablehlo_patterns,
            None if case.reference is None else case.reference(),
        )
        differences = list(comparison.differences)
        if case.validate is not None:
            for side, observation in (("ryft", ryft[case.case_id]), ("jax", jax[case.case_id])):
                differences.extend(f"{side} {difference}" for difference in case.validate(observation))
        comparisons.append(CaseComparison(case_id=case.case_id, differences=tuple(differences)))
    return tuple(comparisons)


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments for the comparison and private JAX-emission modes."""

    parser = argparse.ArgumentParser(description="Differentially test Ryft behavior and StableHLO against pinned JAX.")
    parser.add_argument("--case", action="append", default=[], help="select one case ID; may be repeated")
    parser.add_argument("--suite", choices=("collectives", *CUDA_SUITES), help="select a correctness suite")
    parser.add_argument(
        "--timeout", type=int, default=SUBPROCESS_TIMEOUT_SECONDS,
        help="maximum seconds for each framework subprocess (default: 300)",
    )
    parser.add_argument("--ryft-binary", type=Path, help="run a prebuilt Rust emitter instead of cargo run")
    parser.add_argument("--worker-log-directory", type=Path, help="retain distributed stdout/stderr separately per rank")
    parser.add_argument("--distributed-worker", type=int, choices=(0, 1), help=argparse.SUPPRESS)
    parser.add_argument("--coordinator", help=argparse.SUPPRESS)
    parser.add_argument("--list", action="store_true", help="list case IDs without executing them")
    parser.add_argument("--emit-jax", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Runs the CLI and returns its process exit code."""

    parsed = parse_arguments(arguments)
    try:
        if parsed.timeout <= 0:
            raise ValueError("--timeout must be positive")
        cases = _selected_cases(parsed.case, parsed.suite)
        if (parsed.distributed_worker is None) != (parsed.coordinator is None):
            raise ValueError("distributed worker and coordinator must be specified together")
        if parsed.distributed_worker is not None and (
            not parsed.emit_jax or not cases or any(case.suite != "cuda-distributed-collectives" for case in cases)
        ):
            raise ValueError("distributed worker mode requires CUDA distributed JAX cases")
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    if parsed.list:
        if parsed.emit_jax:
            raise ValueError("--list cannot be combined with --emit-jax")
        for case in cases:
            print(case.case_id)
        return 0
    if parsed.emit_jax:
        from ryft.jax.differential_testing_cases import build_jax_observations

        records = build_jax_observations(
            tuple(case.case_id for case in cases), rank=parsed.distributed_worker, coordinator=parsed.coordinator,
        )
        print(json.dumps([observation_payload(record) for record in records], indent=2))
        return 0
    try:
        comparisons = run_comparison(
            repo_root(), tuple(case.case_id for case in cases), timeout=parsed.timeout, binary=parsed.ryft_binary,
            log_directory=parsed.worker_log_directory,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    for comparison in comparisons:
        if comparison.passed():
            print(f"PASS {comparison.case_id}")
        else:
            print(f"FAIL {comparison.case_id}")
            for difference in comparison.differences:
                print(f"  {difference}")
    return 0 if all(comparison.passed() for comparison in comparisons) else 1


if __name__ == "__main__":
    # Registry callbacks import this module by its canonical name. Use that same instance for comparisons so
    # dataclass contracts do not compare unequal merely because `python -m` defined a second copy in `__main__`.
    from ryft.jax.differential_testing import main as entry_point

    raise SystemExit(entry_point())


__all__ = [
    "CaseComparison",
    "DifferentialObservation",
    "PINNED_JAX_VERSION",
    "SCHEMA",
    "StableHloCollective",
    "StagingObservation",
    "collect_jax_observations",
    "collect_ryft_observations",
    "compare_case",
    "differential_cases",
    "main",
    "observation_payload",
    "parse_arguments",
    "parse_observation",
    "project_collective_stablehlo",
    "repo_root",
    "run_comparison",
]
