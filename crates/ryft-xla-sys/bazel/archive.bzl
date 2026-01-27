load("@com_google_protobuf//bazel/common:proto_info.bzl", "ProtoInfo")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

HEADERS = [
    "jaxlib/mosaic/gpu/integrations/c/passes.h",
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
    "mlir/Dialect/Async/Passes.capi.h.inc",
    "mlir/Dialect/GPU/Transforms/Passes.capi.h.inc",
    "mlir/Dialect/Linalg/Passes.capi.h.inc",
    "mlir/Dialect/SparseTensor/Transforms/Passes.capi.h.inc",
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
    "xla/pjrt/c/pjrt_c_api_ffi_extension.h",
    "xla/pjrt/c/pjrt_c_api_layouts_extension.h",
    "xla/pjrt/c/pjrt_c_api_memory_descriptions_extension.h",
    "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h",
    "xla/pjrt/c/pjrt_c_api_profiler_extension.h",
    "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h",
    "xla/pjrt/c/pjrt_c_api_stream_extension.h",
    "xla/pjrt/c/pjrt_c_api_triton_extension.h",
    "xla/service/spmd/shardy/integrations/c/passes.h",
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
                path = "lib/" + path
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

    copy_commands = []
    for file, path in archive_files:
        copy_commands.append("""
            dst="{dst}"
            target_dir="$archive_dir/$(dirname "$dst")"
            mkdir -p "$target_dir"
            chmod 755 "$target_dir"

            cp "{src}" "$archive_dir/$dst"
            chmod 644 "$archive_dir/$dst"

            # Set permissions based on file type
            if [[ "$src" == *.so ]] || [[ "$src" == *.a ]] || [[ "$src" == *.dylib ]] || [[ "$src" == *.dll ]] || [[ "$src" == *.lib ]]; then
                # Shared libraries need execute permissions.
                chmod 755 "$archive_dir/$dst"
            else
                # Other files do not.
                chmod 644 "$archive_dir/$dst"
            fi
        """.format(
            src = file.path,
            dst = path,
        ))

    ctx.actions.run_shell(
        inputs = [file[0] for file in archive_files],
        outputs = [output],
        command = """
            set -e
            archive_dir=$(mktemp -d)
            chmod 755 "$archive_dir"

            {copy_commands}

            tar -C "$archive_dir" -czf "{output}" .
            rm -rf "$archive_dir"
        """.format(
            copy_commands = "\n".join(copy_commands),
            output = output.path,
        ),
    )

    return [DefaultInfo(files = depset([output]))]

build_archive = rule(
    implementation = _build_archive_impl,
    attrs = {"deps": attr.label_list(providers = [[DefaultInfo], [CcInfo], [ProtoInfo]])},
)

def _extract_headers_impl(ctx):
    cc_info = ctx.attr.library[CcInfo]
    headers = cc_info.compilation_context.headers.to_list()
    return DefaultInfo(files = depset(headers))

extract_headers = rule(
    implementation = _extract_headers_impl,
    attrs = {"library": attr.label(providers = [CcInfo])},
)
