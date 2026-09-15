"""Pinned official cuTile AOT worker; the Rust adapter owns process and artifact lifetime."""

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path
import sys


def versions():
    """Checks the actual pinned installation and executable without initializing CUDA."""
    distributions = {
        "cuda_tile": importlib.metadata.distribution("cuda-tile"),
        "tileiras": importlib.metadata.distribution("nvidia-cuda-tileiras"),
        "nvcc": importlib.metadata.distribution("nvidia-cuda-nvcc"),
        "nvvm": importlib.metadata.distribution("nvidia-nvvm"),
    }
    result = {name: distribution.version for name, distribution in distributions.items()}
    result["python"] = sys.version.split()[0]
    distribution = distributions["tileiras"]
    binaries = [distribution.locate_file(path).resolve() for path in distribution.files or ()
                if str(path).endswith("/bin/tileiras")]
    if len(binaries) != 1 or not binaries[0].is_file():
        raise RuntimeError("pinned tileiras distribution must contain exactly one compiler binary")
    binary = binaries[0]
    namespace = importlib.import_module("nvidia.cu13")
    selected = Path(next(iter(namespace.__path__))) / "bin" / "tileiras"
    if selected.resolve() != binary:
        raise RuntimeError("nvidia.cu13 namespace does not select the pinned distribution compiler")
    completed = subprocess.run([str(binary), "--version"], check=True, capture_output=True, text=True, timeout=10)
    match = re.search(r"\bV(\d+\.\d+\.\d+)\b", completed.stdout)
    if match is None:
        raise RuntimeError("tileiras did not report a recognizable executable version")
    result["binary_version"] = match.group(1)
    # The pinned frontend prefers matching pip companions. A controlled fallback path cannot select a different
    # system compiler; CUDA_HOME must not redirect native companion lookup to an unrelated toolkit.
    os.environ["PATH"] = str(binary.parent) + os.pathsep + os.defpath
    os.environ.pop("CUDA_HOME", None)
    os.environ.pop("CUDA_PATH", None)
    return result


def main():
    """Exports one generated source module under its explicit versioned signature."""
    # Testing flags can remove arithmetic/token ordering and are not semantic compiler options. Strip every
    # frontend override before import, then use isolated adapter-owned temporary and cache locations.
    for name in list(os.environ):
        if name.startswith(("CUDA_TILE_", "EXPERIMENTAL_CUDA_TILE_")):
            del os.environ[name]
    os.environ["CUDA_TILE_TEMP_DIR"] = str(Path.cwd())
    os.environ["CUDA_TILE_CACHE_DIR"] = str(Path.cwd() / "cache")
    if sys.argv[1] == "probe":
        try:
            actual = versions()
        except importlib.metadata.PackageNotFoundError as error:
            print(json.dumps({"missing": str(error)}, sort_keys=True))
            return
        print(json.dumps(actual, sort_keys=True))
        return
    actual = versions()
    request = json.loads(Path(sys.argv[1]).read_text())
    if actual != request["versions"]:
        raise RuntimeError("compiler installation changed after admission")
    import cuda.tile as ct
    from cuda.tile import compilation
    expected_frontend = importlib.metadata.distribution("cuda-tile").locate_file("cuda/tile/__init__.py").resolve()
    if Path(ct.__file__).resolve() != expected_frontend:
        raise RuntimeError("imported cuTile frontend does not belong to the pinned distribution")

    directory = Path(sys.argv[1]).parent
    specification = importlib.util.spec_from_file_location("ryft_cutile_generated", directory / "kernel.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    constraints = [
        compilation.ArrayConstraint(
            getattr(ct, parameter["dtype"]),
            len(parameter["shape"]),
            index_dtype=ct.int32,
            stride_lower_bound_incl=0,
            alias_groups=("ryft",) if len(request["parameters"]) > 1 else (),
            may_alias_internally=False,
            stride_constant=tuple(parameter["strides"]),
            shape_constant=tuple(parameter["shape"]),
            stride_divisible_by=1,
            shape_divisible_by=1,
            base_addr_divisible_by=1,
        )
        for parameter in request["parameters"]
    ]
    signature = compilation.KernelSignature(
        constraints, compilation.CallingConvention.cutile_python_v2(), symbol=request["symbol"]
    )
    artifact = directory / "kernel.cubin"
    with ct.compiler_timeout(request["compiler_timeout_seconds"]):
        compilation.export_kernel(
            module.ryft_cutile_kernel,
            [signature],
            artifact,
            gpu_code=request["target"],
            output_format="cubin",
            bytecode_version="13.3",
        )
    if artifact.stat().st_size > request["maximum_artifact_bytes"]:
        raise RuntimeError("cubin exceeds the requested artifact size limit")
    data = artifact.read_bytes()
    manifest = {
        "schema": request["schema"],
        "semantic_key": request["semantic_key"],
        "configuration_key": request["configuration_key"],
        "versions": actual,
        "target": request["target"],
        "symbol": request["symbol"],
        "calling_convention": "cutile_python_v2",
        "parameters": request["parameters"],
        "grid": request["grid"],
        "block": [1, 1, 1],
        "shared_memory_bytes": 0,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
