#!/usr/bin/env python3
"""Validates pinned non-TPU source contracts and the symbols in a built `ryft-xla-sys` archive.

The static archive is always the CPU archive: CUDA and cuTile runtimes ship only in the Linux CUDA PJRT plugins, so
the archive must define every source-derived symbol and none of the CUDA-only runtime symbols below.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from collections.abc import Iterable
from pathlib import Path
from unittest.mock import patch


MOSAIC_DIALECT = Path("jaxlib/mosaic/dialect/gpu/mosaic_gpu.td")
MOSAIC_C_API_DIRECTORY = Path("jaxlib/mosaic/dialect/gpu/integrations/c")
MOSAIC_RUNTIME_DIRECTORY = Path("jaxlib/mosaic/gpu")


class ContractError(RuntimeError):
    """Reports an invalid input source tree or unsupported Pallas contract drift."""


def read(root: Path, relative_path: Path | str) -> str:
    """Reads one required UTF-8 source file below `root`."""
    path = root / relative_path
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ContractError(f"failed to read required source `{path}`: {error}") from error


def normalize_name(name: str) -> str:
    """Normalizes Rust and TableGen type names for parity comparisons."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def extract_revision(text: str, name: str, source: str) -> str:
    """Extracts and validates one pinned hexadecimal revision."""
    match = re.search(rf"^\s*{re.escape(name)}\s*=\s*\"([0-9a-f]+)\"", text, re.MULTILINE)
    if match is None:
        raise ContractError(f"failed to extract `{name}` from `{source}`")
    revision = match.group(1)
    if len(revision) != 40:
        raise ContractError(f"expected `{name}` in `{source}` to contain a 40-character revision")
    return revision


def require_equal_sets(contract: str, expected: set[str], actual: set[str]) -> None:
    """Requires two named source surfaces to contain exactly the same values."""
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise ContractError(f"{contract} differs: {'; '.join(details)}")


def check_mosaic_surface(jax_root: Path, ryft_mlir_root: Path) -> None:
    """Validates exact pinned Mosaic GPU TableGen and typed Rust parity."""
    tablegen = read(jax_root, MOSAIC_DIALECT)
    operation_names = sorted(set(re.findall(r'Op<MosaicGPU_Dialect,\s*"([^"]+)"', tablegen)))
    type_names = sorted(set(re.findall(r'MosaicGPU_Type<"([^"]+)"', tablegen)))
    attribute_names = sorted(
        set(
            re.findall(r'AttrDef<MosaicGPU_Dialect,\s*"([^"]+)"', tablegen)
            + re.findall(r'MosaicGPU_Attr<"([^"]+)"', tablegen)
            + re.findall(r'I32EnumAttr<"([^"]+)"', tablegen)
        )
    )
    interface_names = sorted(set(re.findall(r'AttrInterface<"([^"]+)"', tablegen)))

    expected_counts = {"attributes": 17, "interfaces": 1, "operations": 43, "types": 3}
    actual_counts = {
        "attributes": len(attribute_names),
        "interfaces": len(interface_names),
        "operations": len(operation_names),
        "types": len(type_names),
    }
    if actual_counts != expected_counts:
        raise ContractError(f"pinned Mosaic GPU surface changed: expected {expected_counts}, found {actual_counts}")

    operations_path = Path("src/dialects/mosaic/gpu/operations.rs")
    attributes_path = Path("src/dialects/mosaic/gpu/attributes.rs")
    types_path = Path("src/dialects/mosaic/gpu/types.rs")
    local_operations = set(
        re.findall(r'"mosaic_gpu\.([a-z0-9_]+)"', read(ryft_mlir_root, operations_path))
    )
    local_attribute_text = read(ryft_mlir_root, attributes_path)
    local_attributes = {
        normalize_name(name.removesuffix("AttributeRef"))
        for name in re.findall(r"(?:pub struct|attribute_name\s*=)\s*([A-Za-z0-9_]+AttributeRef)", local_attribute_text)
    }
    local_type_text = read(ryft_mlir_root, types_path)
    local_types = {
        normalize_name(name.removesuffix("TypeRef"))
        for name in re.findall(r"pub struct\s+([A-Za-z0-9_]+TypeRef)", local_type_text)
    }

    normalized_attributes = {normalize_name(name) for name in attribute_names}
    normalized_interfaces = {normalize_name(name) for name in interface_names}
    local_attribute_wrappers = {
        normalize_name(name.removesuffix("AttributeRef"))
        for name in re.findall(r"pub struct\s+([A-Za-z0-9_]+AttributeRef)", local_attribute_text)
    }
    local_interface_wrappers = local_attribute_wrappers - normalized_attributes

    require_equal_sets("Mosaic GPU operations", set(operation_names), local_operations)
    require_equal_sets("Mosaic GPU types", {normalize_name(name) for name in type_names}, local_types)
    require_equal_sets("Mosaic GPU attributes", normalized_attributes, local_attributes - normalized_interfaces)
    require_equal_sets("Mosaic GPU attribute interfaces", normalized_interfaces, local_interface_wrappers)


def extract_c_functions(text: str, export_macro: str) -> list[str]:
    """Extracts C function names declared after one visibility macro."""
    pattern = rf"{re.escape(export_macro)}\s+(?:[A-Za-z_][\w:]*(?:\s*\*)?\s+)+([A-Za-z_]\w*)\s*\("
    return sorted(set(re.findall(pattern, text, re.MULTILINE)))


def extract_dialect_handle_symbols(text: str) -> list[str]:
    """Expands MLIR C dialect-registration declarations to symbol names."""
    namespaces = re.findall(r"MLIR_DECLARE_CAPI_DIALECT_REGISTRATION\([^,]+,\s*([a-z0-9_]+)\s*\)", text)
    return sorted({f"mlirGetDialectHandle__{namespace}__" for namespace in namespaces})


def discover_required_symbols(
    jax_root: Path,
    llvm_root: Path,
    ryft_xla_sys_root: Path,
) -> set[str]:
    """Validates pinned C API bindings and returns symbols promised by the native archive."""
    upstream_headers = [
        MOSAIC_C_API_DIRECTORY / "attributes.h",
        MOSAIC_C_API_DIRECTORY / "gpu_dialect.h",
        MOSAIC_RUNTIME_DIRECTORY / "integrations/c/passes.h",
    ]
    mosaic_symbols: set[str] = set()
    for header in upstream_headers:
        text = read(jax_root, header)
        for symbol in extract_c_functions(text, "MLIR_CAPI_EXPORTED") + extract_dialect_handle_symbols(text):
            mosaic_symbols.add(symbol)

    complex_header = Path("mlir/include/mlir-c/Dialect/Complex.h")
    complex_symbols = set(extract_c_functions(read(llvm_root, complex_header), "MLIR_CAPI_EXPORTED"))

    pass_definitions = {
        Path("mlir/include/mlir/Dialect/Arith/Transforms/Passes.td"): {
            "ArithExpandOpsPass",
        },
        Path("mlir/include/mlir/Dialect/LLVMIR/Transforms/Passes.td"): {
            "DIScopeForLLVMFuncOpPass",
        },
        Path("mlir/include/mlir/Dialect/Math/Transforms/Passes.td"): {
            "MathExpandOpsPass",
            "MathExtendToSupportedTypes",
            "MathSincosFusionPass",
            "MathUpliftToFMA",
        },
        Path("mlir/include/mlir/Dialect/MemRef/Transforms/Passes.td"): {
            "ExpandStridedMetadataPass",
        },
        Path("mlir/include/mlir/Dialect/Vector/Transforms/Passes.td"): {
            "LowerVectorMaskPass",
            "LowerVectorMultiReduction",
            "LowerVectorToFromElementsToShuffleTree",
        },
    }
    for tablegen_path, expected_names in pass_definitions.items():
        pass_names = set(re.findall(r"\bdef\s+([A-Za-z_]\w*)\s*:\s*Pass<", read(llvm_root, tablegen_path)))
        missing_names = expected_names - pass_names
        if missing_names:
            raise ContractError(
                f"failed to find pinned pass definitions in `{tablegen_path}`: {sorted(missing_names)}"
            )
    pass_symbols = {
        f"{prefix}{pass_name}"
        for expected_names in pass_definitions.values()
        for pass_name in expected_names
        for prefix in ("mlirCreate", "mlirRegister")
    }

    rust_paths = [
        Path("src/bindings.rs"),
        Path("src/mlir/dialects/complex.rs"),
        Path("src/mlir/dialects/mosaic/gpu.rs"),
    ]
    local_rust_symbols: dict[Path, set[str]] = {}
    for rust_path in rust_paths:
        local_rust = read(ryft_xla_sys_root, rust_path)
        local_rust_symbols[rust_path] = set(re.findall(r"pub fn\s+([A-Za-z_]\w*)\s*\(", local_rust)) | set(
            extract_dialect_handle_symbols(local_rust)
        )

    source_owned_mosaic_header = Path("src/c++/mlir/dialects/mosaic_gpu.h")
    source_owned_mosaic_symbols = set(
        extract_c_functions(read(ryft_xla_sys_root, source_owned_mosaic_header), "RYFT_XLA_SYS_EXPORT")
    )
    mosaic_rust_symbols = local_rust_symbols[Path("src/mlir/dialects/mosaic/gpu.rs")]
    complex_rust_symbols = local_rust_symbols[Path("src/mlir/dialects/complex.rs")]
    relevant_complex_rust_symbols = {
        symbol
        for symbol in complex_rust_symbols
        if symbol.startswith("mlirComplex") or symbol == "mlirAttributeIsAComplex"
    }
    require_equal_sets("Mosaic GPU C API bindings", mosaic_symbols | source_owned_mosaic_symbols, mosaic_rust_symbols)
    require_equal_sets("Complex C API bindings", complex_symbols, relevant_complex_rust_symbols)

    missing_pass_symbols = pass_symbols - local_rust_symbols[Path("src/bindings.rs")]
    if missing_pass_symbols:
        raise ContractError(f"compiler pass C API bindings are missing {sorted(missing_pass_symbols)}")

    required_symbols: set[str] = set()
    headers_root = ryft_xla_sys_root / "src/c++"
    for header in sorted(headers_root.rglob("*.h")):
        relative_path = header.relative_to(ryft_xla_sys_root)
        if "tpu" in relative_path.as_posix().lower():
            continue
        text = header.read_text(encoding="utf-8")
        for symbol in extract_c_functions(text, "RYFT_XLA_SYS_EXPORT") + extract_dialect_handle_symbols(text):
            required_symbols.add(symbol)
    required_symbols.update(mosaic_symbols | complex_symbols | pass_symbols)
    if any("tpu" in symbol.lower() for symbol in required_symbols):
        raise ContractError("TPU symbols leaked into the non-TPU native contract")
    return required_symbols


def check_versions_and_routes(jax_root: Path, ryft_xla_sys_root: Path, ryft_mlir_root: Path) -> None:
    """Validates Mosaic versions, targets, passes, formats, and unsupported native ABIs."""
    serde_path = MOSAIC_RUNTIME_DIRECTORY / "serde.cc"
    serde_text = read(jax_root, serde_path)
    serde_match = re.search(r"constexpr int kVersion\s*=\s*(\d+);", serde_text)
    if serde_match is None:
        raise ContractError(f"failed to extract Mosaic serde version from `{serde_path}`")

    custom_call_path = MOSAIC_RUNTIME_DIRECTORY / "custom_call.cc"
    custom_call_text = read(jax_root, custom_call_path)
    resource_match = re.search(r"kernel_proto\.set_version\((\d+)\);", custom_call_text)
    if resource_match is None:
        raise ContractError(f"failed to extract Mosaic resource version from `{custom_call_path}`")
    targets = sorted(set(re.findall(r'XLA_FFI_REGISTER_HANDLER\([^;]+?"([^"]+)"', custom_call_text, re.DOTALL)))
    if "mosaic_gpu_v2" not in targets:
        raise ContractError("pinned JAX source no longer registers `mosaic_gpu_v2`")

    pass_header = MOSAIC_RUNTIME_DIRECTORY / "passes.h"
    if not re.findall(r"void\s+register[A-Za-z0-9]+Pass\s*\(\);", read(jax_root, pass_header)):
        raise ContractError(f"failed to find Mosaic GPU pass registrations in `{pass_header}`")
    serde_header = MOSAIC_RUNTIME_DIRECTORY / "integrations/c/passes.h"
    serde_symbol = "mlirMosaicGpuRegisterSerdePass"
    if serde_symbol not in read(jax_root, serde_header):
        raise ContractError(f"failed to find `{serde_symbol}` in `{serde_header}`")
    local_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ryft_xla_sys_root / "src").rglob("*"))
        if path.is_file() and path.suffix in {".cc", ".h", ".rs"} and "tpu" not in path.as_posix().lower()
    )

    rust_path = Path("src/mlir/dialects/mosaic/gpu.rs")
    local_rust = read(ryft_xla_sys_root, rust_path)

    def require_constant(constant: str, value: str) -> None:
        """Requires one Rust constant to equal its pinned upstream value."""
        match = re.search(rf"pub const\s+{constant}\s*:\s*[^=]+\s*=\s*(?:\"([^\"]+)\"|(\d+))\s*;", local_rust)
        local_value = next((group for group in match.groups() if group is not None), None) if match else None
        if local_value != value:
            raise ContractError(f"`{constant}` is `{local_value}`, expected pinned value `{value}`")

    require_constant("MOSAIC_GPU_SERDE_VERSION", serde_match.group(1))
    require_constant("MOSAIC_GPU_RESOURCE_SCHEMA_VERSION", resource_match.group(1))
    require_constant("MOSAIC_GPU_FFI_TARGET", "mosaic_gpu_v2")
    if serde_symbol not in local_text:
        raise ContractError(f"local sources do not declare `{serde_symbol}`")

    read(jax_root, MOSAIC_RUNTIME_DIRECTORY / "mosaic_gpu.proto")
    modules_text = read(ryft_mlir_root, "src/modules.rs")
    operations_text = read(ryft_mlir_root, "src/operations/operation.rs")
    for function_name, source_text in (
        ("parse_module_from_bytes", modules_text),
        ("parse_operation_from_bytes", operations_text),
        ("bytecode", operations_text),
    ):
        if not re.search(rf"\bfn\s+{function_name}\b", source_text):
            raise ContractError(f"`ryft-mlir` is missing binary MLIR function `{function_name}`")

    for unsupported_symbol in ("MosaicGpuCompile", "MosaicGpuUnload", "MosaicGpuClearKernelCache"):
        if re.search(rf"(?:pub fn|RYFT_XLA_SYS_EXPORT[^;]*\b){unsupported_symbol}\s*\(", local_text):
            raise ContractError(f"unsupported wheel-only ABI `{unsupported_symbol}` was declared locally")


def check_standard_dialects(llvm_root: Path, ryft_xla_sys_root: Path) -> None:
    """Validates the five common compiler-dialect build and archive contracts."""
    build_path = Path("BUILD.bazel")
    build_text = read(ryft_xla_sys_root, build_path)
    archive_path = Path("bazel/archive.bzl")
    archive_text = read(ryft_xla_sys_root, archive_path)
    modules_path = Path("src/mlir/dialects.rs")
    modules_text = read(ryft_xla_sys_root, modules_path)
    dialects = {
        "bufferization": ("Bufferization", "BufferizationOpsTdFiles"),
        "complex": ("Complex", "ComplexOpsTdFiles"),
        "math": ("Math", "MathOpsTdFiles"),
        "ub": ("UB", "UBDialectTdFiles"),
        "vector": ("Vector", "VectorOpsTdFiles"),
    }
    for namespace, (class_name, tablegen_target) in sorted(dialects.items()):
        llvm_header = Path(f"mlir/include/mlir-c/Dialect/{class_name}.h")
        llvm_tablegen = Path(f"mlir/include/mlir/Dialect/{class_name}/IR/{class_name}Ops.td")
        if namespace == "ub":
            llvm_tablegen = Path("mlir/include/mlir/Dialect/UB/IR/UBOps.td")
        upstream_handle = (llvm_root / llvm_header).is_file()
        upstream_tablegen = (llvm_root / llvm_tablegen).is_file()
        local_header = Path(f"src/c++/mlir/dialects/{namespace}.h")
        local_source = Path(f"src/c++/mlir/dialects/{namespace}.cc")
        local_rust = Path(f"src/mlir/dialects/{namespace}.rs")
        expected_symbol = f"mlirGetDialectHandle__{namespace}__"
        local_header_text = (
            read(ryft_xla_sys_root, local_header) if (ryft_xla_sys_root / local_header).is_file() else ""
        )
        local_rust_text = read(ryft_xla_sys_root, local_rust) if (ryft_xla_sys_root / local_rust).is_file() else ""
        local_handle = all(
            (
                expected_symbol in extract_dialect_handle_symbols(local_header_text),
                expected_symbol in local_rust_text,
                local_source.as_posix() in build_text,
                local_header.as_posix() in build_text,
                f'@llvm-project//mlir:{class_name}Dialect' in build_text,
                f"pub mod {namespace};" in modules_text,
            )
        )
        if namespace in {"math", "vector"}:
            handle_available = upstream_handle and f"@llvm-project//mlir:CAPI{class_name}" in build_text
            archive_header = f"mlir-c/Dialect/{class_name}.h" in archive_text
        else:
            handle_available = upstream_handle and local_handle
            archive_header = local_header.as_posix() in archive_text
        tablegen_archived = tablegen_target in build_text
        failures = [
            name
            for name, satisfied in (
                ("C API handle", handle_available),
                ("upstream TableGen", upstream_tablegen),
                ("archived TableGen target", tablegen_archived),
                ("archived C API header", archive_header),
            )
            if not satisfied
        ]
        if failures:
            raise ContractError(f"`{namespace}` compiler dialect contract is missing {failures}")


def check_sources(args: argparse.Namespace) -> set[str]:
    """Validates the complete source contract and returns required archive symbols."""
    jax_root = args.jax_root.resolve()
    xla_root = args.xla_root.resolve()
    llvm_root = args.llvm_root.resolve()
    ryft_xla_sys_root = args.ryft_xla_sys_root.resolve()
    ryft_mlir_root = args.ryft_mlir_root.resolve()

    workspace_path = Path("WORKSPACE")
    workspace = read(ryft_xla_sys_root, workspace_path)
    extract_revision(workspace, "XLA_COMMIT", workspace_path.as_posix())
    extract_revision(workspace, "JAX_COMMIT", workspace_path.as_posix())
    llvm_workspace_path = Path("third_party/llvm/workspace.bzl")
    llvm_workspace = read(xla_root, llvm_workspace_path)
    extract_revision(llvm_workspace, "LLVM_COMMIT", llvm_workspace_path.as_posix())

    check_mosaic_surface(jax_root, ryft_mlir_root)
    required_symbols = discover_required_symbols(jax_root, llvm_root, ryft_xla_sys_root)
    check_versions_and_routes(jax_root, ryft_xla_sys_root, ryft_mlir_root)
    check_standard_dialects(llvm_root, ryft_xla_sys_root)
    return required_symbols


# C-linkage symbols that only the Linux CUDA PJRT plugins may define. `MosaicGpu*` covers the private JAX runtime ABI
# (`MosaicGpuCompile`, `MosaicGpuUnload`, `MosaicGpuClearKernelCache`), the `cu*` names are CUDA Driver API entry
# points used by cubin launchers, and `mosaic_gpu_v2` is the XLA FFI registration target. The MLIR C API bridge for the
# Mosaic GPU dialect (`mlirMosaicGpu*`, `mlirGetDialectHandle__mosaic_gpu__`) is portable and stays allowed.
FORBIDDEN_CPU_SYMBOL_PREFIXES = ("MosaicGpu",)
FORBIDDEN_CPU_SYMBOLS = frozenset(
    {
        "cuInit",
        "cuLaunchKernel",
        "cuModuleLoadData",
        "cuModuleLoadDataEx",
        "cuModuleGetFunction",
        "cuGetProcAddress",
        "cuGetProcAddress_v2",
    }
)
FORBIDDEN_CPU_SYMBOL_FRAGMENTS = ("cutile", "tileiras", "mosaic_gpu_v2")


def symbol_tool() -> tuple[str, list[str]]:
    """Returns an available symbol-inspection tool and its arguments."""
    llvm_nm = shutil.which("llvm-nm")
    if llvm_nm is not None:
        return llvm_nm, ["--defined-only", "--extern-only"]
    nm = shutil.which("nm")
    if nm is not None:
        return nm, ["-gU"] if platform.system() == "Darwin" else ["-g", "--defined-only"]
    raise RuntimeError("`llvm-nm` or `nm` is required to validate the archive symbol contract")


def missing_symbols(required_symbols: set[str], emitted_symbols: Iterable[str]) -> list[str]:
    """Returns required symbols that are absent, ignoring duplicate definitions."""
    return sorted(required_symbols - set(emitted_symbols))


def is_forbidden_cpu_symbol(symbol: str) -> bool:
    """Returns whether one emitted symbol belongs to the CUDA-only runtime surface.

    Mach-O prefixes C symbols with one underscore, so a single leading underscore is ignored before matching the
    C-linkage names; mangled C++ names (`_Z...`) never match the prefixes and are only subject to the fragment rules.
    MSVC string-literal symbols (`??_C@...`) are ignored because their contents do not define runtime entry points.
    """
    if symbol.startswith("??_C@"):
        return False
    name = symbol[1:] if symbol.startswith("_") and not symbol.startswith("_Z") else symbol
    if name in FORBIDDEN_CPU_SYMBOLS or name.startswith(FORBIDDEN_CPU_SYMBOL_PREFIXES):
        return True
    lowered = symbol.lower()
    return any(fragment in lowered for fragment in FORBIDDEN_CPU_SYMBOL_FRAGMENTS)


def forbidden_symbols(emitted_symbols: Iterable[str]) -> list[str]:
    """Returns emitted symbols that a CPU archive must not define."""
    return sorted({symbol for symbol in emitted_symbols if is_forbidden_cpu_symbol(symbol)})


def check_archive_symbols(archive_path: Path, required_symbols: set[str]) -> None:
    """Checks the built library against the validated source symbols and CPU-only restrictions."""
    tool, arguments = symbol_tool()

    with tempfile.TemporaryDirectory() as temporary_directory:
        with tarfile.open(archive_path, "r:gz") as archive:
            libraries = [
                member
                for member in archive.getmembers()
                if member.isfile() and member.name.removeprefix("./").startswith("lib/")
            ]
            if len(libraries) != 1:
                raise AssertionError(f"expected one archive library, found {[member.name for member in libraries]}")
            archive.extract(libraries[0], temporary_directory, filter="data")
            library_path = Path(temporary_directory) / libraries[0].name.removeprefix("./")

        completed = subprocess.run(
            [tool, *arguments, str(library_path)],
            check=True,
            capture_output=True,
            text=True,
        )

    emitted_symbols = set()
    for line in completed.stdout.splitlines():
        if not line.strip() or line.rstrip().endswith(":"):
            continue
        emitted_symbol = line.split()[-1]
        if emitted_symbol.startswith("_") and emitted_symbol[1:] in required_symbols:
            emitted_symbol = emitted_symbol[1:]
        emitted_symbols.add(emitted_symbol)

    missing = missing_symbols(required_symbols, emitted_symbols)
    if missing:
        raise AssertionError("missing required symbols:\n" + "\n".join(missing))
    forbidden = forbidden_symbols(emitted_symbols)
    if forbidden:
        raise AssertionError("CPU archive defines CUDA-only runtime symbols:\n" + "\n".join(forbidden))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    for option in ("jax", "xla", "llvm", "ryft-xla-sys", "ryft-mlir"):
        parser.add_argument(f"--{option}-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Validates the source contracts and their required symbols in the built CPU archive."""
    args = parse_args(argv)
    try:
        required_symbols = check_sources(args)
        check_archive_symbols(args.archive, required_symbols)
    except ContractError as error:
        print(f"native contract error: {error}", file=sys.stderr)
        return 2
    print(f"validated source and archive contracts with {len(required_symbols)} required symbols")
    return 0


OPERATIONS = [
    "arrive_dyn_expect_tx_supported",
    "assume_multiple",
    "async_store_smem",
    "get_cluster_ref",
    "mma",
    "vector_concat",
    "arrive",
    "arrive_expect_tx",
    "async_load",
    "async_load_tmem",
    "async_prefetch",
    "async_store",
    "async_store_scales_smem_to_tmem",
    "async_store_smem_to_tmem",
    "async_store_sparse_metadata_smem_to_tmem",
    "async_store_tmem",
    "broadcast_in_dim",
    "broadcasted_iota",
    "custom_primitive",
    "debug_print",
    "initialize_barrier",
    "layout_cast",
    "multimem_load_reduce",
    "optimization_barrier",
    "print_layout",
    "query_cluster_cancel",
    "reinterpret_cast",
    "return",
    "slice_smem",
    "slice_tmem",
    "tcgen05_commit_arrive",
    "tcgen05_mma",
    "tmem_alloc",
    "tmem_dealloc",
    "tmem_layout_cast",
    "tmem_relinquish_alloc_permit",
    "try_cluster_cancel",
    "vector_load",
    "vector_store",
    "wait",
    "warp_map",
    "wgmma",
    "with_transforms",
]

ATTRIBUTES = [
    "AtomicOpType",
    "CopyPartitioned",
    "CopyReplicated",
    "Dimension",
    "MultimemLoadReductionType",
    "OOBFillMode",
    "Replicated",
    "SwizzleTransform",
    "SwizzlingMode",
    "TMAReduction",
    "TileTransform",
    "TiledLayout",
    "Tmem",
    "SmemCluster",
    "TMEMLoadReduction",
    "WGSplatFragLayout",
    "WGStridedFragLayout",
]


class SourceFixture:
    """Creates small source roots that preserve the audited source patterns."""

    def __init__(self, root: Path):
        self.jax = root / "jax"
        self.xla = root / "xla"
        self.llvm = root / "llvm"
        self.ryft_xla_sys = root / "ryft-xla-sys"
        self.ryft_mlir = root / "ryft-mlir"
        self._create()

    def write(self, root: Path, path: str, text: str) -> None:
        """Writes one fixture source file."""
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")

    def _create(self) -> None:
        tablegen = [
            'def MosaicGPU_Barrier : MosaicGPU_Type<"Barrier", "barrier">;',
            'def MosaicGPU_B6x16P32 : MosaicGPU_Type<"B6x16P32", "b6x16p32">;',
            'def MosaicGPU_P2B6 : MosaicGPU_Type<"P2B6", "p2b6">;',
            'def MosaicGPU_CopyPartitionAttrInterface : AttrInterface<"CopyPartition">;',
        ]
        tablegen.extend(
            f'def MosaicGPU_{name}Op : Op<MosaicGPU_Dialect, "{name}", []>;' for name in OPERATIONS
        )
        tablegen.extend(
            f'def MosaicGPU_{name} : AttrDef<MosaicGPU_Dialect, "{name}", []>;' for name in ATTRIBUTES
        )
        self.write(self.jax, str(MOSAIC_DIALECT), "\n".join(tablegen))

        attributes_header = "\n".join(
            [
                "MLIR_CAPI_EXPORTED MlirTypeID mlirMosaicGpuBarrierTypeGetTypeID();",
                "MLIR_CAPI_EXPORTED bool mlirMosaicGpuIsATileTransformAttr(MlirAttribute attr);",
            ]
        )
        self.write(
            self.jax,
            "jaxlib/mosaic/dialect/gpu/integrations/c/attributes.h",
            attributes_header,
        )
        self.write(
            self.jax,
            "jaxlib/mosaic/dialect/gpu/integrations/c/gpu_dialect.h",
            (
                "MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(MosaicGPU, mosaic_gpu);\n"
                "MLIR_CAPI_EXPORTED void mlirDialectRegistryInsertMosaicGpuInlinerExtensions("
                "MlirDialectRegistry registry);\n"
            ),
        )
        self.write(self.jax, "jaxlib/mosaic/gpu/serde.cc", "constexpr int kVersion = 6;\n")
        self.write(
            self.jax,
            "jaxlib/mosaic/gpu/custom_call.cc",
            """
kernel_proto.set_version(1);
XLA_FFI_REGISTER_HANDLER(api, "mosaic_gpu_v2", "CUDA", handlers);
void** MosaicGpuCompile(const char* module, int size);
void MosaicGpuUnload(void** value);
void MosaicGpuClearKernelCache();
""",
        )
        self.write(
            self.jax,
            "jaxlib/mosaic/gpu/passes.h",
            "void registerConvertGpuToLLVMPass();\n",
        )
        self.write(
            self.jax,
            "jaxlib/mosaic/gpu/integrations/c/passes.h",
            "MLIR_CAPI_EXPORTED void mlirMosaicGpuRegisterSerdePass();\n",
        )
        self.write(self.jax, "jaxlib/mosaic/gpu/mosaic_gpu.proto", "message MosaicGpuKernelProto {}\n")

        self.write(
            self.xla,
            "third_party/llvm/workspace.bzl",
            'LLVM_COMMIT = "1a5376e062e3f3b99ffce25e24e53182202e06b9"\n',
        )
        dialects = {
            "Bufferization": "BufferizationOps",
            "Complex": "ComplexOps",
            "Math": "MathOps",
            "UB": "UBOps",
            "Vector": "VectorOps",
        }
        for dialect, tablegen in dialects.items():
            header = "/* C API */\n"
            if dialect == "Complex":
                header += (
                    "MLIR_CAPI_EXPORTED bool mlirAttributeIsAComplex(MlirAttribute attribute);\n"
                    "MLIR_CAPI_EXPORTED MlirAttribute mlirComplexAttrDoubleGet("
                    "MlirContext context, MlirType type, double real, double imaginary);\n"
                    "MLIR_CAPI_EXPORTED MlirAttribute mlirComplexAttrDoubleGetChecked("
                    "MlirLocation location, MlirType type, double real, double imaginary);\n"
                    "MLIR_CAPI_EXPORTED double mlirComplexAttrGetRealDouble(MlirAttribute attribute);\n"
                    "MLIR_CAPI_EXPORTED double mlirComplexAttrGetImagDouble(MlirAttribute attribute);\n"
                    "MLIR_CAPI_EXPORTED MlirTypeID mlirComplexAttrGetTypeID(void);\n"
                )
            self.write(self.llvm, f"mlir/include/mlir-c/Dialect/{dialect}.h", header)
            self.write(self.llvm, f"mlir/include/mlir/Dialect/{dialect}/IR/{tablegen}.td", "// TableGen\n")
        self.write(
            self.llvm,
            "mlir/include/mlir/Dialect/Arith/Transforms/Passes.td",
            'def ArithExpandOpsPass : Pass<"test">;\n',
        )
        self.write(
            self.llvm,
            "mlir/include/mlir/Dialect/LLVMIR/Transforms/Passes.td",
            'def DIScopeForLLVMFuncOpPass : Pass<"test">;\n',
        )
        self.write(
            self.llvm,
            "mlir/include/mlir/Dialect/Math/Transforms/Passes.td",
            "\n".join(
                f'def {name} : Pass<"test">;'
                for name in (
                    "MathExpandOpsPass",
                    "MathExtendToSupportedTypes",
                    "MathSincosFusionPass",
                    "MathUpliftToFMA",
                )
            ),
        )
        self.write(
            self.llvm,
            "mlir/include/mlir/Dialect/MemRef/Transforms/Passes.td",
            'def ExpandStridedMetadataPass : Pass<"test">;\n',
        )
        self.write(
            self.llvm,
            "mlir/include/mlir/Dialect/Vector/Transforms/Passes.td",
            "\n".join(
                f'def {name} : Pass<"test">;'
                for name in (
                    "LowerVectorMaskPass",
                    "LowerVectorMultiReduction",
                    "LowerVectorToFromElementsToShuffleTree",
                )
            ),
        )

        self.write(
            self.ryft_xla_sys,
            "WORKSPACE",
            """
XLA_COMMIT = "f16a4aeb435b2896ab96b605f004f982f6c97eb8"
JAX_COMMIT = "a33ed614c58ee8a10d0b7536c50c2609c38500c1"
""",
        )
        self.write(
            self.ryft_xla_sys,
            "BUILD.bazel",
            "\n".join(
                [
                    '"src/c++/mlir/dialects/bufferization.cc"',
                    '"src/c++/mlir/dialects/bufferization.h"',
                    '"src/c++/mlir/dialects/complex.cc"',
                    '"src/c++/mlir/dialects/complex.h"',
                    '"src/c++/mlir/dialects/ub.cc"',
                    '"src/c++/mlir/dialects/ub.h"',
                    "@llvm-project//mlir:BufferizationDialect",
                    "@llvm-project//mlir:ComplexDialect",
                    "@llvm-project//mlir:UBDialect",
                    "@llvm-project//mlir:CAPIMath",
                    "@llvm-project//mlir:CAPIVector",
                    "BufferizationOpsTdFiles",
                    "ComplexOpsTdFiles",
                    "MathOpsTdFiles",
                    "UBDialectTdFiles",
                    "VectorOpsTdFiles",
                ]
            ),
        )
        self.write(
            self.ryft_xla_sys,
            "bazel/archive.bzl",
            "\n".join(
                [
                    '"mlir-c/Dialect/Math.h"',
                    '"mlir-c/Dialect/Vector.h"',
                    '"src/c++/mlir/dialects/bufferization.h"',
                    '"src/c++/mlir/dialects/complex.h"',
                    '"src/c++/mlir/dialects/ub.h"',
                ]
            ),
        )
        self.write(
            self.ryft_xla_sys,
            "src/bindings.rs",
            "\n".join(
                f"pub fn {prefix}{name}();"
                for name in (
                    "ArithExpandOpsPass",
                    "DIScopeForLLVMFuncOpPass",
                    "MathExpandOpsPass",
                    "MathExtendToSupportedTypes",
                    "MathSincosFusionPass",
                    "MathUpliftToFMA",
                    "ExpandStridedMetadataPass",
                    "LowerVectorMaskPass",
                    "LowerVectorMultiReduction",
                    "LowerVectorToFromElementsToShuffleTree",
                )
                for prefix in ("mlirCreate", "mlirRegister")
            ),
        )
        self.write(
            self.ryft_xla_sys,
            "src/mlir/dialects.rs",
            "pub mod bufferization;\npub mod complex;\npub mod ub;\n",
        )
        self.write(
            self.ryft_xla_sys,
            "src/mlir/dialects/mosaic/gpu.rs",
            """
pub fn mlirGetDialectHandle__mosaic_gpu__();
pub fn mlirDialectRegistryInsertMosaicGpuInlinerExtensions(registry: MlirDialectRegistry);
pub fn mlirMosaicGpuBarrierTypeGetTypeID();
pub fn mlirMosaicGpuIsATileTransformAttr(attribute: MlirAttribute) -> bool;
pub fn mlirMosaicGpuEnumAttrGet(context: MlirContext) -> MlirAttribute;
pub fn mlirMosaicGpuRegisterSerdePass();
pub const MOSAIC_GPU_SERDE_VERSION: i32 = 6;
pub const MOSAIC_GPU_RESOURCE_SCHEMA_VERSION: i32 = 1;
pub const MOSAIC_GPU_FFI_TARGET: &str = "mosaic_gpu_v2";
""",
        )
        self.write(
            self.ryft_xla_sys,
            "src/c++/mlir/dialects/mosaic_gpu.h",
            "RYFT_XLA_SYS_EXPORT MlirAttribute mlirMosaicGpuEnumAttrGet(MlirContext context);\n",
        )
        for dialect in ("bufferization", "complex", "ub"):
            self.write(
                self.ryft_xla_sys,
                f"src/c++/mlir/dialects/{dialect}.h",
                f"MLIR_DECLARE_CAPI_DIALECT_REGISTRATION({dialect.title()}, {dialect});\n",
            )
            self.write(self.ryft_xla_sys, f"src/c++/mlir/dialects/{dialect}.cc", "// Definition\n")
            self.write(
                self.ryft_xla_sys,
                f"src/mlir/dialects/{dialect}.rs",
                f"pub fn mlirGetDialectHandle__{dialect}__();\n",
            )
        complex_rust = self.ryft_xla_sys / "src/mlir/dialects/complex.rs"
        complex_rust.write_text(
            complex_rust.read_text(encoding="utf-8")
            + (
                "pub fn mlirAttributeIsAComplex(attribute: MlirAttribute) -> bool;\n"
                "pub fn mlirComplexAttrDoubleGet(context: MlirContext, type_: MlirType, "
                "real: f64, imaginary: f64) -> MlirAttribute;\n"
                "pub fn mlirComplexAttrDoubleGetChecked(location: MlirLocation, type_: MlirType, "
                "real: f64, imaginary: f64) -> MlirAttribute;\n"
                "pub fn mlirComplexAttrGetRealDouble(attribute: MlirAttribute) -> f64;\n"
                "pub fn mlirComplexAttrGetImagDouble(attribute: MlirAttribute) -> f64;\n"
                "pub fn mlirComplexAttrGetTypeID() -> MlirTypeID;\n"
            ),
            encoding="utf-8",
        )
        self.write(
            self.ryft_xla_sys,
            "src/c++/profiler.h",
            "RYFT_XLA_SYS_EXPORT void ryftProfilerStart(void);\n",
        )

        self.write(
            self.ryft_mlir,
            "src/dialects/mosaic/gpu/operations.rs",
            "\n".join(f'OperationBuilder::new("mosaic_gpu.{name}", location);' for name in OPERATIONS),
        )
        self.write(
            self.ryft_mlir,
            "src/dialects/mosaic/gpu/attributes.rs",
            "\n".join(
                [f"pub struct {name}AttributeRef;" for name in ATTRIBUTES]
                + ["pub struct CopyPartitionAttributeRef;"]
            ),
        )
        self.write(self.ryft_mlir, "src/dialects/mosaic/gpu/types.rs", "pub struct BarrierTypeRef;\npub struct B6x16P32TypeRef;\npub struct P2B6TypeRef;\n")
        self.write(self.ryft_mlir, "src/modules.rs", "pub fn parse_module_from_bytes() {}\n")
        self.write(
            self.ryft_mlir,
            "src/operations/operation.rs",
            "pub fn parse_operation_from_bytes() {}\npub fn bytecode() {}\n",
        )

    def args(self) -> argparse.Namespace:
        """Returns contract-checker arguments for this fixture."""
        return argparse.Namespace(
            jax_root=self.jax,
            llvm_root=self.llvm,
            ryft_mlir_root=self.ryft_mlir,
            ryft_xla_sys_root=self.ryft_xla_sys,
            xla_root=self.xla,
        )


class ContractTests(unittest.TestCase):
    """Covers direct parity, symbol discovery, and drift failures."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.fixture = SourceFixture(Path(self.temporary_directory.name))

    def test_check_validates_contract_and_discovers_symbols(self) -> None:
        symbols = check_sources(self.fixture.args())
        self.assertIn("mlirGetDialectHandle__mosaic_gpu__", symbols)
        self.assertTrue(
            {
                "mlirAttributeIsAComplex",
                "mlirComplexAttrDoubleGet",
                "mlirComplexAttrDoubleGetChecked",
                "mlirComplexAttrGetImagDouble",
                "mlirComplexAttrGetRealDouble",
                "mlirComplexAttrGetTypeID",
            }.issubset(symbols)
        )
        self.assertIn("mlirMosaicGpuRegisterSerdePass", symbols)
        self.assertIn("mlirCreateMathExpandOpsPass", symbols)
        self.assertIn("mlirRegisterArithExpandOpsPass", symbols)
        self.assertIn("mlirRegisterDIScopeForLLVMFuncOpPass", symbols)
        self.assertIn("mlirRegisterExpandStridedMetadataPass", symbols)
        self.assertIn("mlirRegisterLowerVectorMaskPass", symbols)
        self.assertIn("ryftProfilerStart", symbols)
        self.assertFalse(any("tpu" in symbol.lower() for symbol in symbols))

    def test_cli_reports_invalid_pin(self) -> None:
        arguments = self.fixture.args()
        llvm_workspace = arguments.xla_root / "third_party/llvm/workspace.bzl"
        llvm_workspace.write_text('LLVM_COMMIT = "0000"\n', encoding="utf-8")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(main(self._argv(arguments)), 2)
        self.assertIn("40-character revision", stderr.getvalue())

    def test_same_count_surface_substitution_is_rejected(self) -> None:
        tablegen = self.fixture.jax / MOSAIC_DIALECT
        tablegen.write_text(
            tablegen.read_text(encoding="utf-8").replace(
                'Op<MosaicGPU_Dialect, "arrive", []>',
                'Op<MosaicGPU_Dialect, "replacement", []>',
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ContractError, "Mosaic GPU operations differs"):
            check_sources(self.fixture.args())

    def test_missing_native_binding_is_rejected(self) -> None:
        rust = self.fixture.ryft_xla_sys / "src/mlir/dialects/mosaic/gpu.rs"
        rust.write_text(
            rust.read_text(encoding="utf-8").replace("pub fn mlirMosaicGpuBarrierTypeGetTypeID();\n", ""),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ContractError, "Mosaic GPU C API bindings differs"):
            check_sources(self.fixture.args())

    def test_wheel_only_compiler_abi_is_rejected(self) -> None:
        local_header = self.fixture.ryft_xla_sys / "src/c++/profiler.h"
        local_header.write_text(
            local_header.read_text(encoding="utf-8")
            + "RYFT_XLA_SYS_EXPORT void MosaicGpuCompile(const char* module, int size);\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ContractError, "unsupported wheel-only ABI"):
            check_sources(self.fixture.args())

    def test_main_validates_sources_and_archive(self) -> None:
        """Runs the combined CLI over valid sources and a library's reported symbol table."""
        arguments = self.fixture.args()
        archive_path = arguments.ryft_xla_sys_root / "archive.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            member = tarfile.TarInfo("lib/library.a")
            member.size = len(b"library")
            archive.addfile(member, io.BytesIO(b"library"))
        symbols = check_sources(arguments)
        symbol_output = "\n".join(f"00000000 T {symbol}" for symbol in sorted(symbols))
        completed = subprocess.CompletedProcess(["nm"], 0, stdout=symbol_output, stderr="")
        stdout = io.StringIO()
        with (
            patch(f"{__name__}.symbol_tool", return_value=("nm", ["-g", "--defined-only"])),
            patch(f"{__name__}.subprocess.run", return_value=completed) as run,
            contextlib.redirect_stdout(stdout),
        ):
            self.assertEqual(main(self._argv(arguments)), 0)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][:3], ["nm", "-g", "--defined-only"])
        self.assertEqual(
            stdout.getvalue(),
            f"validated source and archive contracts with {len(symbols)} required symbols\n",
        )

    def test_main_rejects_missing_and_forbidden_archive_symbols(self) -> None:
        """Ensures the combined CLI still enforces both sides of the archive symbol contract."""
        arguments = self.fixture.args()
        archive_path = arguments.ryft_xla_sys_root / "archive.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.addfile(tarfile.TarInfo("lib/library.a"))
        symbols = check_sources(arguments)
        missing = "mlirMosaicGpuRegisterSerdePass"
        for emitted, message in (
            (symbols - {missing}, "missing required symbols"),
            (symbols | {"cuLaunchKernel"}, "CPU archive defines CUDA-only runtime symbols"),
        ):
            with self.subTest(message=message):
                completed = subprocess.CompletedProcess(
                    ["nm"], 0, stdout="\n".join(f"00000000 T {symbol}" for symbol in sorted(emitted)), stderr="",
                )
                with (
                    patch(f"{__name__}.symbol_tool", return_value=("nm", ["-g", "--defined-only"])),
                    patch(f"{__name__}.subprocess.run", return_value=completed),
                    self.assertRaisesRegex(AssertionError, message),
                ):
                    main(self._argv(arguments))

    @staticmethod
    def _argv(arguments: argparse.Namespace) -> list[str]:
        return [
            "--archive",
            str(arguments.ryft_xla_sys_root / "archive.tar.gz"),
            "--jax-root",
            str(arguments.jax_root),
            "--xla-root",
            str(arguments.xla_root),
            "--llvm-root",
            str(arguments.llvm_root),
            "--ryft-xla-sys-root",
            str(arguments.ryft_xla_sys_root),
            "--ryft-mlir-root",
            str(arguments.ryft_mlir_root),
        ]


class SymbolContractTests(unittest.TestCase):
    """Covers the presence and absence symbol contracts."""

    def test_duplicate_definitions_are_accepted(self) -> None:
        self.assertEqual(missing_symbols({"required"}, ["required", "required"]), [])

    def test_forbidden_symbols_classifier(self) -> None:
        # CUDA-only runtime symbols are rejected with and without the Mach-O underscore prefix.
        self.assertTrue(is_forbidden_cpu_symbol("MosaicGpuCompile"))
        self.assertTrue(is_forbidden_cpu_symbol("_MosaicGpuClearKernelCache"))
        self.assertTrue(is_forbidden_cpu_symbol("cuLaunchKernel"))
        self.assertTrue(is_forbidden_cpu_symbol("_cuModuleLoadDataEx"))
        self.assertTrue(is_forbidden_cpu_symbol("cuTileCompileKernel"))
        self.assertTrue(is_forbidden_cpu_symbol("_ZN8tileiras7compileEv"))
        self.assertTrue(is_forbidden_cpu_symbol("xla_ffi_register_mosaic_gpu_v2"))
        self.assertTrue(is_forbidden_cpu_symbol("?compile@tileiras@@YAXXZ"))

        # MSVC encodes string contents in symbols; this exact literal appeared in the Windows CPU archive.
        self.assertFalse(is_forbidden_cpu_symbol("??_C@_0O@GABHFABO@mosaic_gpu_v2?$AA@"))

        # The portable Mosaic GPU dialect bridge, serde pass, and unrelated CUDA-looking names stay allowed.
        self.assertFalse(is_forbidden_cpu_symbol("mlirMosaicGpuRegisterSerdePass"))
        self.assertFalse(is_forbidden_cpu_symbol("_mlirGetDialectHandle__mosaic_gpu__"))
        self.assertFalse(is_forbidden_cpu_symbol("_ZN10mosaic_gpu16MosaicGPUDialectD2Ev"))
        self.assertFalse(is_forbidden_cpu_symbol("mlirMosaicGpuTileTransformAttrGet"))
        self.assertFalse(is_forbidden_cpu_symbol("cuda_version_string"))
        self.assertEqual(
            forbidden_symbols(["_mlirModuleCreateEmpty", "_cuLaunchKernel", "MosaicGpuCompile", "_cuLaunchKernel"]),
            ["MosaicGpuCompile", "_cuLaunchKernel"],
        )


if __name__ == "__main__":
    raise SystemExit(main())
