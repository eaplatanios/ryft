"""Builds deterministic native release archives."""

load("@com_google_protobuf//bazel/common:proto_info.bzl", "ProtoInfo")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

HEADERS = [
    "jaxlib/mosaic/gpu/integrations/c/passes.h",
    "jaxlib/mosaic/dialect/gpu/integrations/c/attributes.h",
    "jaxlib/mosaic/dialect/gpu/integrations/c/gpu_dialect.h",
    "llvm/Config/llvm-config.h",
    "llvm-c/Core.h",
    "llvm-c/DataTypes.h",
    "llvm-c/Deprecated.h",
    "llvm-c/ErrorHandling.h",
    "llvm-c/ExternC.h",
    "llvm-c/Support.h",
    "llvm-c/Types.h",
    "llvm-c/Visibility.h",
    "mlir/Config/mlir-config.h",
    "mlir/Conversion/Passes.capi.h.inc",
    "mlir/Dialect/AMDGPU/Transforms/Passes.capi.h.inc",
    "mlir/Dialect/Arith/Transforms/Passes.capi.h.inc",
    "mlir/Dialect/Async/Passes.capi.h.inc",
    "mlir/Dialect/EmitC/Transforms/Passes.capi.h.inc",
    "mlir/Dialect/Func/Transforms/Passes.capi.h.inc",
    "mlir/Dialect/GPU/Transforms/Passes.capi.h.inc",
    "mlir/Dialect/Linalg/Passes.capi.h.inc",
    "mlir/Dialect/LLVMIR/Transforms/Passes.capi.h.inc",
    "mlir/Dialect/Math/Transforms/Passes.capi.h.inc",
    "mlir/Dialect/MemRef/Transforms/Passes.capi.h.inc",
    "mlir/Dialect/MLProgram/Transforms/Passes.capi.h.inc",
    "mlir/Dialect/NVGPU/Transforms/Passes.capi.h.inc",
    "mlir/Dialect/SCF/Transforms/Passes.capi.h.inc",
    "mlir/Dialect/Shape/Transforms/Passes.capi.h.inc",
    "mlir/Dialect/SparseTensor/Transforms/Passes.capi.h.inc",
    "mlir/Dialect/Tensor/Transforms/Passes.capi.h.inc",
    "mlir/Dialect/Transform/Transforms/Passes.capi.h.inc",
    "mlir/Dialect/Vector/Transforms/Passes.capi.h.inc",
    "mlir/Transforms/Transforms.capi.h.inc",
    "mlir-c/AffineExpr.h",
    "mlir-c/AffineMap.h",
    "mlir-c/BuiltinAttributes.h",
    "mlir-c/BuiltinTypes.h",
    "mlir-c/Conversion.h",
    "mlir-c/Debug.h",
    "mlir-c/Diagnostics.h",
    "mlir-c/ExecutionEngine.h",
    "mlir-c/IR.h",
    "mlir-c/IntegerSet.h",
    "mlir-c/Interfaces.h",
    "mlir-c/Pass.h",
    "mlir-c/RegisterEverything.h",
    "mlir-c/Rewrite.h",
    "mlir-c/Support.h",
    "mlir-c/Transforms.h",
    "mlir-c/Dialect/AMDGPU.h",
    "mlir-c/Dialect/Arith.h",
    "mlir-c/Dialect/Async.h",
    "mlir-c/Dialect/ControlFlow.h",
    "mlir-c/Dialect/Complex.h",
    "mlir-c/Dialect/EmitC.h",
    "mlir-c/Dialect/Func.h",
    "mlir-c/Dialect/GPU.h",
    "mlir-c/Dialect/Index.h",
    "mlir-c/Dialect/IRDL.h",
    "mlir-c/Dialect/Linalg.h",
    "mlir-c/Dialect/LLVM.h",
    "mlir-c/Dialect/Math.h",
    "mlir-c/Dialect/MemRef.h",
    "mlir-c/Dialect/MLProgram.h",
    "mlir-c/Dialect/NVGPU.h",
    "mlir-c/Dialect/NVVM.h",
    "mlir-c/Dialect/OpenMP.h",
    "mlir-c/Dialect/PDL.h",
    "mlir-c/Dialect/Quant.h",
    "mlir-c/Dialect/ROCDL.h",
    "mlir-c/Dialect/SCF.h",
    "mlir-c/Dialect/Shape.h",
    "mlir-c/Dialect/SMT.h",
    "mlir-c/Dialect/SparseTensor.h",
    "mlir-c/Dialect/SPIRV.h",
    "mlir-c/Dialect/Tensor.h",
    "mlir-c/Dialect/Transform.h",
    "mlir-c/Dialect/Transform/Interpreter.h",
    "mlir-c/Dialect/Vector.h",
    "mlir-c/Target/LLVMIR.h",
    "shardy/integrations/c/attributes.h",
    "shardy/integrations/c/dialect.h",
    "shardy/integrations/c/passes.h",
    "stablehlo/integrations/c/ChloAttributes.h",
    "stablehlo/integrations/c/ChloDialect.h",
    "stablehlo/integrations/c/StablehloAttributes.h",
    "stablehlo/integrations/c/StablehloDialect.h",
    "stablehlo/integrations/c/StablehloPasses.h",
    "stablehlo/integrations/c/StablehloDialectApi.h",
    "stablehlo/integrations/c/StablehloUnifiedApi.h",
    "stablehlo/integrations/c/StablehloTypes.h",
    "stablehlo/integrations/c/InterpreterDialect.h",
    "stablehlo/integrations/c/VhloDialect.h",
    "xla/backends/profiler/plugin/profiler_c_api.h",
    "xla/ffi/api/c_api.h",
    "xla/mlir_hlo/bindings/c/Attributes.h",
    "xla/mlir_hlo/bindings/c/Dialects.h",
    "xla/mlir_hlo/bindings/c/Passes.h",
    "xla/mlir_hlo/bindings/c/Types.h",
    "xla/pjrt/c/pjrt_c_api.h",
    "xla/pjrt/c/pjrt_c_api_abi_version_extension.h",
    "xla/pjrt/c/pjrt_c_api_ffi_extension.h",
    "xla/pjrt/c/pjrt_c_api_layouts_extension.h",
    "xla/pjrt/c/pjrt_c_api_megascale_extension.h",
    "xla/pjrt/c/pjrt_c_api_memory_descriptions_extension.h",
    "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h",
    "xla/pjrt/c/pjrt_c_api_profiler_extension.h",
    "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h",
    "xla/pjrt/c/pjrt_c_api_stream_extension.h",
    "xla/pjrt/c/pjrt_c_api_triton_extension.h",
    "xla/pjrt/c/pjrt_c_api_xla_transform_extension.h",
    "xla/pjrt/extensions/host_allocator/host_allocator_extension.h",
    "xla/pjrt/extensions/host_memory_allocator/host_memory_allocator_extension.h",
    "xla/service/custom_call_status.h",
    "xla/service/spmd/shardy/integrations/c/passes.h",
    "src/c++/common.h",
    "src/c++/distributed.h",
    "src/c++/mlir/dialects/affine.h",
    "src/c++/mlir/dialects/arith.h",
    "src/c++/mlir/dialects/bufferization.h",
    "src/c++/mlir/dialects/builtin.h",
    "src/c++/mlir/dialects/complex.h",
    "src/c++/mlir/dialects/gpu.h",
    "src/c++/mlir/dialects/llvm.h",
    "src/c++/mlir/dialects/mosaic_gpu.h",
    "src/c++/mlir/dialects/mosaic_tpu.h",
    "src/c++/mlir/dialects/nvgpu.h",
    "src/c++/mlir/dialects/shape.h",
    "src/c++/mlir/dialects/sparse_tensor.h",
    "src/c++/mlir/dialects/transform.h",
    "src/c++/mlir/dialects/triton.h",
    "src/c++/mlir/dialects/ub.h",
    "src/c++/profiler.h",
]

def _build_archive_impl(ctx):
    output = ctx.actions.declare_file(ctx.label.name + ".tar.gz")

    archive_files = []
    for dep in ctx.attr.deps:
        files = []
        if DefaultInfo in dep:
            files.extend(dep[DefaultInfo].files.to_list())
        if CcInfo in dep:
            files.extend(dep[CcInfo].compilation_context.headers.to_list())
        if ProtoInfo in dep:
            files.extend(dep[ProtoInfo].transitive_sources.to_list())
            files.extend(dep[ProtoInfo].transitive_descriptor_sets.to_list())
        for file in files:
            path = file.short_path

            # Strip Bazel path prefixes.
            if path.startswith("../"):
                path = path[len("../"):]
            if path.startswith("external/"):
                path = path[len("external/"):]
            for prefix in [
                "jax/",
                "llvm-project/llvm/include/",
                "llvm-project/mlir/include/",
                "shardy/",
                "stablehlo/",
                "xla/",
            ]:
                if path.startswith(prefix):
                    path = path[len(prefix):]

            is_linux_library = path.endswith(".so") or path.endswith(".a")
            is_macos_library = path.endswith(".dylib") or path.endswith(".a")
            is_windows_library = path.endswith(".dll") or path.endswith(".lib") or path.endswith(".def")
            is_library = is_linux_library or is_macos_library or is_windows_library
            is_header = path.endswith(".h") or path.endswith(".hpp") or path.endswith(".inc")
            is_proto = path.endswith(".proto") or path.endswith(".proto.bin")
            is_td = path.endswith(".td")

            # Filter out unnecessary header files.
            if is_header and path not in HEADERS:
                continue
            if "_virtual_includes" in path or "_virtual_imports" in path:
                continue
            if path.endswith(".def"):
                continue

            # Add archive path prefix.
            if is_library:
                # Libraries are admitted only from the explicit `library` attribute below. Some TableGen filegroups
                # expose incidental helper archives in their default outputs; those are not part of our link contract.
                continue
            elif is_header:
                path = "include/" + path
            elif is_proto:
                path = "proto/" + path
            elif is_td:
                path = "td/" + path
            else:
                # Skip any other files that may be present here.
                continue

            archive_files.append((file, path))

    for file in ctx.attr.library[DefaultInfo].files.to_list():
        path = file.basename
        is_library = (
            path.endswith(".so") or
            path.endswith(".a") or
            path.endswith(".dylib") or
            path.endswith(".dll") or
            path.endswith(".lib")
        )
        if not is_library:
            fail("archive library target produced a non-library file: {}".format(file.short_path))
        archive_files.append((file, "lib/" + path))

    archive_paths = {}
    unique_archive_files = []
    for file, path in archive_files:
        if path in archive_paths:
            if archive_paths[path].path == file.path:
                continue
            fail(
                "duplicate archive path {} from {} and {}".format(
                    path,
                    archive_paths[path].short_path,
                    file.short_path,
                ),
            )
        archive_paths[path] = file
        unique_archive_files.append((file, path))
    archive_files = unique_archive_files

    manifest_entries = []
    for file, path in archive_files:
        is_library = path.startswith("lib/")
        manifest_entries.append({
            "mode": 0o755 if is_library else 0o644,
            "path": path,
            "source": file.path,
        })

    manifest = ctx.actions.declare_file(ctx.label.name + ".manifest.json")
    ctx.actions.write(manifest, json.encode(manifest_entries))
    ctx.actions.run(
        executable = ctx.executable._archive_tool,
        inputs = depset([manifest] + [file[0] for file in archive_files]),
        outputs = [output],
        arguments = [manifest.path, output.path],
        mnemonic = "DeterministicArchive",
    )

    return [DefaultInfo(files = depset([output]))]

build_archive = rule(
    implementation = _build_archive_impl,
    attrs = {
        "_archive_tool": attr.label(
            default = Label("//:create-deterministic-archive"),
            executable = True,
            cfg = "exec",
        ),
        "deps": attr.label_list(providers = [[DefaultInfo], [CcInfo], [ProtoInfo]]),
        "library": attr.label(mandatory = True, providers = [DefaultInfo]),
    },
)

def _extract_headers_impl(ctx):
    cc_info = ctx.attr.library[CcInfo]
    headers = cc_info.compilation_context.headers.to_list()
    return DefaultInfo(files = depset(headers))

extract_headers = rule(
    implementation = _extract_headers_impl,
    attrs = {"library": attr.label(providers = [CcInfo])},
)
