#!/usr/bin/env python3
"""Exports the Phase 0 cuTile vector-add fixture and records its launch contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from typing import Any


SYMBOL = "ryft_cutile_vector_add"
VECTOR_LENGTH = 1
TILE_SIZE = 1


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _array_constraint(compilation: Any, tile: Any, alias_group: str) -> Any:
    return compilation.ArrayConstraint(
        tile.int32,
        1,
        index_dtype=tile.int32,
        stride_lower_bound_incl=0,
        alias_groups=(alias_group,),
        may_alias_internally=False,
        stride_constant=(1,),
        shape_constant=(VECTOR_LENGTH,),
        stride_divisible_by=1,
        shape_divisible_by=1,
        base_addr_divisible_by=1,
    )


def _export_fixture(arguments: argparse.Namespace) -> dict[str, Any]:
    try:
        import cuda.tile as tile
        from cuda.tile import compilation
    except ImportError as error:
        raise RuntimeError(
            "cuTile Python is unavailable; install a supported `cuda-tile[tileiras]` environment"
        ) from error

    @tile.kernel
    def vector_add(lhs, rhs, output, tile_size: tile.Constant):
        block_id = tile.bid(0)
        lhs_tile = tile.load(lhs, index=(block_id,), shape=(tile_size,))
        rhs_tile = tile.load(rhs, index=(block_id,), shape=(tile_size,))
        tile.store(output, index=(block_id,), tile=lhs_tile + rhs_tile)

    calling_convention = compilation.CallingConvention.cutile_python_v2()
    signature = compilation.KernelSignature(
        [
            _array_constraint(compilation, tile, "lhs"),
            _array_constraint(compilation, tile, "rhs"),
            _array_constraint(compilation, tile, "output"),
            compilation.ConstantConstraint(TILE_SIZE),
        ],
        calling_convention,
        symbol=SYMBOL,
    )

    with tile.compiler_timeout(arguments.compiler_timeout_seconds):
        compilation.export_kernel(
            vector_add,
            [signature],
            arguments.output_cubin,
            gpu_code=arguments.gpu_code,
            output_format="cubin",
        )

        jax_contract_verified = False
        if arguments.verify_jax_contract:
            try:
                import jax
                import jax.numpy as jax_numpy
                from cuda.tile.jax import OutputPlaceholder, cutile_call
            except ImportError as error:
                raise RuntimeError("JAX cuTile interoperability is unavailable") from error

            @jax.jit
            def jax_vector_add(lhs, rhs):
                output = OutputPlaceholder(lhs.shape, lhs.dtype)
                return cutile_call((1,), vector_add, (lhs, rhs, output, TILE_SIZE))

            result = jax_vector_add(
                jax_numpy.asarray([7], dtype=jax_numpy.int32),
                jax_numpy.asarray([35], dtype=jax_numpy.int32),
            )
            result.block_until_ready()
            if result.tolist() != [42]:
                raise RuntimeError(f"JAX `cutile_call` returned {result.tolist()}, expected [42]")
            jax_contract_verified = True

    cubin = Path(arguments.output_cubin).read_bytes()
    return {
        "schema_version": 1,
        "artifact": {
            "format": "cubin",
            "path": str(Path(arguments.output_cubin).resolve()),
            "sha256": hashlib.sha256(cubin).hexdigest(),
            "size_bytes": len(cubin),
            "target_sm": arguments.gpu_code,
        },
        "kernel": {
            "symbol": SYMBOL,
            "calling_convention": "cutile_python_v2",
            "grid_dimensions": [1, 1, 1],
            "block_dimensions": [1, 1, 1],
            "shared_memory_bytes": 0,
            "parameters": [
                {
                    "name": name,
                    "constraint": "ArrayConstraint",
                    "dtype": "int32",
                    "ndim": 1,
                    "index_dtype": "int32",
                    "shape_constant": [VECTOR_LENGTH],
                    "stride_constant": [1],
                    "alias_groups": [name],
                    "abi": ["device_pointer", "shape_i32", "stride_i32"],
                }
                for name in ("lhs", "rhs", "output")
            ]
            + [
                {
                    "name": "tile_size",
                    "constraint": "ConstantConstraint",
                    "value": TILE_SIZE,
                    "abi": [],
                }
            ],
        },
        "verification": {
            "jax_cutile_call_contract": jax_contract_verified,
        },
        "toolchain": {
            "python": sys.version.split()[0],
            "cuda_tile": _package_version("cuda-tile"),
            "nvidia_cuda_tileiras": _package_version("nvidia-cuda-tileiras"),
            "jax": _package_version("jax"),
            "jaxlib": _package_version("jaxlib"),
        },
    }


def _worker(arguments: argparse.Namespace) -> int:
    metadata = _export_fixture(arguments)
    print(f"RYFT_CUTILE_METADATA={json.dumps(metadata, sort_keys=True)}")
    return 0


def _diagnostics_path(arguments: argparse.Namespace) -> Path:
    if arguments.diagnostics is not None:
        return arguments.diagnostics
    return arguments.output_metadata.with_suffix(".diagnostics.json")


def _orchestrate(arguments: argparse.Namespace) -> int:
    arguments.output_cubin.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path = _diagnostics_path(arguments)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=arguments.output_cubin.parent,
        prefix=f".{arguments.output_cubin.name}.",
    )
    os.close(file_descriptor)
    temporary_cubin = Path(temporary_name)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--output-cubin",
        str(temporary_cubin),
        "--output-metadata",
        str(arguments.output_metadata),
        "--gpu-code",
        arguments.gpu_code,
        "--compiler-timeout-seconds",
        str(arguments.compiler_timeout_seconds),
    ]
    if arguments.verify_jax_contract:
        command.append("--verify-jax-contract")

    diagnostics: dict[str, Any] = {
        "schema_version": 1,
        "command": command,
        "process_timeout_seconds": arguments.process_timeout_seconds,
    }
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=arguments.process_timeout_seconds,
        )
        diagnostics.update(
            {
                "status": "completed",
                "return_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
    except subprocess.TimeoutExpired as error:
        diagnostics.update(
            {
                "status": "timed_out",
                "stdout": error.stdout or "",
                "stderr": error.stderr or "",
            }
        )
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
        temporary_cubin.unlink(missing_ok=True)
        raise RuntimeError(
            f"cuTile export exceeded the {arguments.process_timeout_seconds}-second process timeout; "
            f"diagnostics: {diagnostics_path}"
        ) from error

    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    if completed.returncode != 0:
        temporary_cubin.unlink(missing_ok=True)
        raise RuntimeError(f"cuTile export failed; diagnostics: {diagnostics_path}")

    metadata_line = next(
        (line for line in reversed(completed.stdout.splitlines()) if line.startswith("RYFT_CUTILE_METADATA=")),
        None,
    )
    if metadata_line is None:
        temporary_cubin.unlink(missing_ok=True)
        raise RuntimeError(f"cuTile export produced no metadata; diagnostics: {diagnostics_path}")
    metadata = json.loads(metadata_line.removeprefix("RYFT_CUTILE_METADATA="))

    temporary_cubin.replace(arguments.output_cubin)
    metadata["artifact"]["path"] = str(arguments.output_cubin.resolve())
    arguments.output_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, sort_keys=True))
    return 0


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-cubin", required=True, type=Path)
    parser.add_argument("--output-metadata", required=True, type=Path)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--gpu-code", required=True, help="cuTile target, for example `sm_100`")
    parser.add_argument("--compiler-timeout-seconds", type=int, default=120)
    parser.add_argument("--process-timeout-seconds", type=int, default=150)
    parser.add_argument("--verify-jax-contract", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.compiler_timeout_seconds <= 0:
        parser.error("--compiler-timeout-seconds must be positive")
    if arguments.process_timeout_seconds <= arguments.compiler_timeout_seconds:
        parser.error("--process-timeout-seconds must exceed --compiler-timeout-seconds")
    return arguments


def main() -> int:
    arguments = _parse_arguments()
    try:
        return _worker(arguments) if arguments.worker else _orchestrate(arguments)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
