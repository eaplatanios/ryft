#!/usr/bin/env python3
"""Validates the portable contents of a built `ryft-xla-sys` archive."""

from __future__ import annotations

import pathlib
import sys
import tarfile


REQUIRED_PATHS = {
    "include/jaxlib/mosaic/dialect/gpu/integrations/c/attributes.h",
    "include/jaxlib/mosaic/dialect/gpu/integrations/c/gpu_dialect.h",
    "include/jaxlib/mosaic/gpu/integrations/c/passes.h",
    "include/mlir-c/Dialect/Math.h",
    "include/mlir-c/Dialect/Complex.h",
    "include/mlir-c/Dialect/Vector.h",
    "include/src/c++/common.h",
    "include/src/c++/distributed.h",
    "include/src/c++/mlir/dialects/affine.h",
    "include/src/c++/mlir/dialects/arith.h",
    "include/src/c++/mlir/dialects/bufferization.h",
    "include/src/c++/mlir/dialects/builtin.h",
    "include/src/c++/mlir/dialects/complex.h",
    "include/src/c++/mlir/dialects/gpu.h",
    "include/src/c++/mlir/dialects/llvm.h",
    "include/src/c++/mlir/dialects/mosaic_gpu.h",
    "include/src/c++/mlir/dialects/mosaic_tpu.h",
    "include/src/c++/mlir/dialects/nvgpu.h",
    "include/src/c++/mlir/dialects/shape.h",
    "include/src/c++/mlir/dialects/sparse_tensor.h",
    "include/src/c++/mlir/dialects/transform.h",
    "include/src/c++/mlir/dialects/triton.h",
    "include/src/c++/mlir/dialects/ub.h",
    "include/src/c++/profiler.h",
    "include/xla/pjrt/c/pjrt_c_api_abi_version_extension.h",
    "include/xla/pjrt/c/pjrt_c_api_ffi_extension.h",
    "include/xla/pjrt/c/pjrt_c_api_layouts_extension.h",
    "include/xla/pjrt/c/pjrt_c_api_megascale_extension.h",
    "include/xla/pjrt/c/pjrt_c_api_memory_descriptions_extension.h",
    "include/xla/pjrt/c/pjrt_c_api_phase_compile_extension.h",
    "include/xla/pjrt/c/pjrt_c_api_profiler_extension.h",
    "include/xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h",
    "include/xla/pjrt/c/pjrt_c_api_stream_extension.h",
    "include/xla/pjrt/c/pjrt_c_api_triton_extension.h",
    "include/xla/pjrt/c/pjrt_c_api_xla_transform_extension.h",
    "include/xla/pjrt/extensions/host_allocator/host_allocator_extension.h",
    "include/xla/pjrt/extensions/host_memory_allocator/host_memory_allocator_extension.h",
    "td/jaxlib/mosaic/dialect/gpu/mosaic_gpu.td",
    "td/mlir/Dialect/Bufferization/IR/BufferizationOps.td",
    "td/mlir/Dialect/Complex/IR/ComplexOps.td",
    "td/mlir/Dialect/Math/IR/MathOps.td",
    "td/mlir/Dialect/UB/IR/UBOps.td",
    "td/mlir/Dialect/Vector/IR/VectorOps.td",
}

FORBIDDEN_PREFIXES = ("share/ryft-xla-sys/pallas/",)


def normalized_path(name: str) -> str:
    """Returns the canonical relative path for one tar member name."""
    return name.removeprefix("./")


def main() -> int:
    """Checks required paths, duplicates, and the single-library contract."""
    if len(sys.argv) != 2:
        raise SystemExit("usage: archive_contract_test.py <archive.tar.gz>")

    archive_path = pathlib.Path(sys.argv[1])
    with tarfile.open(archive_path, "r:gz") as archive:
        paths = [normalized_path(member.name) for member in archive.getmembers() if member.isfile()]

    duplicate_paths = sorted(path for path in set(paths) if paths.count(path) > 1)
    if duplicate_paths:
        raise AssertionError(f"duplicate archive paths: {duplicate_paths}")

    missing_paths = sorted(REQUIRED_PATHS.difference(paths))
    if missing_paths:
        raise AssertionError("missing required archive paths:\n" + "\n".join(missing_paths))

    forbidden_paths = sorted(path for path in paths if path.startswith(FORBIDDEN_PREFIXES))
    if forbidden_paths:
        raise AssertionError("source-only metadata present in native archive:\n" + "\n".join(forbidden_paths))

    libraries = sorted(
        path for path in paths if path.startswith("lib/") and path.endswith((".a", ".dylib", ".dll", ".lib", ".so"))
    )
    if len(libraries) != 1 or "ryft-xla-sys-static-library" not in libraries[0]:
        raise AssertionError(f"expected exactly the main static library, found: {libraries}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
