#!/usr/bin/env python3
"""Validates the portable contents of a built `ryft-xla-sys` archive."""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


REQUIRED_PATHS = {
    "include/jaxlib/mosaic/dialect/gpu/integrations/c/attributes.h",
    "include/jaxlib/mosaic/dialect/gpu/integrations/c/gpu_dialect.h",
    "include/jaxlib/mosaic/gpu/integrations/c/passes.h",
    "include/mlir-c/Dialect/Math.h",
    "include/mlir-c/Dialect/Complex.h",
    "include/mlir-c/Dialect/Vector.h",
    "include/mlir/Dialect/GPU/Transforms/NVVM/Passes.capi.h.inc",
    "include/mlir/Dialect/GPU/Transforms/ROCDL/Passes.capi.h.inc",
    "include/mlir/Dialect/GPU/Transforms/SPIRV/Passes.capi.h.inc",
    "include/mlir/Dialect/GPU/Transforms/XeVM/Passes.capi.h.inc",
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
    "include/src/c++/triton_compiler.h",
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
    "include/xla/pjrt/c/pjrt_c_api_xla_transform_extension.h",
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


def check_macos_platform_commands(library: pathlib.Path) -> None:
    """Requires every Mach-O archive member to declare its target platform."""
    output = subprocess.run(
        ["xcrun", "otool", "-l", str(library)], capture_output=True, text=True, check=True, timeout=300
    ).stdout
    members = re.split(r"(?m)^.+\((.+)\):\n", output)[1:]
    if not members:
        raise AssertionError("no Mach-O objects found in static library")
    missing_commands = [
        name
        for name, commands in zip(members[::2], members[1::2])
        if not re.search(r"(?m)^\s*cmd LC_(BUILD_VERSION|VERSION_MIN_MACOSX)$", commands)
    ]
    if missing_commands:
        raise AssertionError("objects missing macOS platform version commands:\n" + "\n".join(missing_commands))


@unittest.skipUnless(sys.platform == "darwin", "requires the macOS assembler and object tools")
class MacosPlatformCommandsTest(unittest.TestCase):
    """Exercises platform validation on real archives with mixed object metadata."""

    def test_check_macos_platform_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for name, directive in (
                ("modern", ".build_version macos, 11, 0"),
                ("legacy", ".macosx_version_min 10, 14"),
                ("missing", ""),
            ):
                source = root / f"{name}.s"
                source.write_text(f"{directive}\n.section __TEXT,__const\n.byte 1\n", encoding="utf-8")
                subprocess.run(
                    [
                        "xcrun", "clang", "-cc1as", "-triple", "x86_64-apple-macosx", "-filetype", "obj",
                        str(source), "-o", str(root / f"{name}.o"),
                    ],
                    check=True,
                    timeout=300,
                )

            library = root / "library.a"
            subprocess.run(
                ["xcrun", "ar", "rcs", str(library), str(root / "modern.o"), str(root / "legacy.o")],
                check=True,
                timeout=300,
            )
            check_macos_platform_commands(library)

            subprocess.run(
                ["xcrun", "ar", "r", str(library), str(root / "missing.o")], check=True, timeout=300
            )
            with self.assertRaises(AssertionError) as error:
                check_macos_platform_commands(library)
            self.assertEqual(str(error.exception), "objects missing macOS platform version commands:\nmissing.o")


class ArchiveContractTest(unittest.TestCase):
    """Validates the built archive supplied on the script's command line."""

    archive_path: pathlib.Path

    def test_archive_contents(self) -> None:
        with tarfile.open(self.archive_path, "r:gz") as archive:
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

        if sys.platform == "darwin":
            with tarfile.open(self.archive_path, "r:gz") as archive, tempfile.TemporaryDirectory() as directory:
                member = next(member for member in archive.getmembers() if normalized_path(member.name) == libraries[0])
                library = pathlib.Path(directory) / "library.a"
                with archive.extractfile(member) as source, library.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                check_macos_platform_commands(library)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: archive_contract_test.py <archive.tar.gz> [unittest options]")
    ArchiveContractTest.archive_path = pathlib.Path(sys.argv.pop(1))
    unittest.main()
