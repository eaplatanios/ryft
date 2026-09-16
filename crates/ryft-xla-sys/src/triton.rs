//! Native Triton compilation through the linked XLA compiler. Output buffers are owned by the corresponding arguments.

#![allow(non_camel_case_types, non_snake_case)]

use crate::bindings::{MlirLogicalResult, MlirModule, MlirStringRef};

/// Arguments and owned results for synchronous compilation of a borrowed MLIR module.
#[repr(C)]
pub struct RYFT_XLA_Triton_Compile_Args {
    /// Borrowed input module, cloned before lowering.
    pub module: MlirModule,

    /// Borrowed platform name: `cuda` or `rocm`.
    pub platform: MlirStringRef,

    /// Borrowed CUDA capability or AMD architecture.
    pub architecture: MlirStringRef,

    /// Requested number of warps.
    pub warp_count: i32,

    /// Requested pipeline stage count.
    pub stage_count: i32,

    /// Maximum size of the returned artifact.
    pub maximum_artifact_bytes: usize,

    /// Maximum number of captured diagnostic bytes.
    pub maximum_diagnostic_bytes: usize,

    /// Owned artifact buffer.
    pub artifact: *mut u8,

    /// Length of the artifact buffer.
    pub artifact_size: usize,

    /// Owned entry name buffer.
    pub entry_name: *mut u8,

    /// Length of the entry name buffer.
    pub entry_name_size: usize,

    /// Owned compilation diagnostic buffer.
    pub diagnostics: *mut u8,

    /// Length of the diagnostic buffer.
    pub diagnostics_size: usize,

    /// Number of pointer arguments in the compiled entry.
    pub argument_count: i64,

    /// Number of warps selected by the compiler.
    pub actual_warp_count: i64,

    /// Number of threads in each warp or wavefront.
    pub threads_per_warp: i64,

    /// Dynamic shared memory required by the compiled entry.
    pub shared_memory_bytes: i64,
}

impl RYFT_XLA_Triton_Compile_Args {
    /// Creates compilation arguments with empty owned output buffers.
    pub fn new(
        module: MlirModule,
        platform: MlirStringRef,
        architecture: MlirStringRef,
        warp_count: i32,
        stage_count: i32,
        maximum_artifact_bytes: usize,
        maximum_diagnostic_bytes: usize,
    ) -> Self {
        Self {
            module,
            platform,
            architecture,
            warp_count,
            stage_count,
            maximum_artifact_bytes,
            maximum_diagnostic_bytes,
            artifact: std::ptr::null_mut(),
            artifact_size: 0,
            entry_name: std::ptr::null_mut(),
            entry_name_size: 0,
            diagnostics: std::ptr::null_mut(),
            diagnostics_size: 0,
            argument_count: 0,
            actual_warp_count: 0,
            threads_per_warp: 0,
            shared_memory_bytes: 0,
        }
    }
}

/// Linked source revisions and backend availability, including an owned effective assembler version.
#[repr(C)]
pub struct RYFT_XLA_Triton_Versions {
    /// Borrowed XLA revision with process lifetime.
    pub xla: MlirStringRef,

    /// Borrowed JAX revision with process lifetime.
    pub jax: MlirStringRef,

    /// Borrowed Triton revision with process lifetime.
    pub triton: MlirStringRef,

    /// Borrowed embedded ROCm device library revision.
    pub rocm_device_libs: MlirStringRef,

    /// Whether CUDA compilation is linked into the native archive.
    pub cuda_available: bool,

    /// Whether ROCm compilation is linked into the native archive.
    pub rocm_available: bool,

    /// CUDA toolkit version used to build the native archive, or zero.
    pub cuda_toolkit_version: i32,

    /// Owned effective CUDA assembler version buffer.
    pub assembler_version: *mut u8,

    /// Length of the assembler version buffer.
    pub assembler_version_size: usize,
}

unsafe extern "C" {
    /// Compiles a clone of the input module and returns success or failure with owned diagnostics.
    ///
    /// # Safety
    ///
    /// The arguments and their borrowed module and strings must be valid. The caller must hold the module's context
    /// exclusively throughout this call. Outputs must be empty on entry and destroyed after either success or failure.
    /// This native entry point supports compiler integration without serializing or transferring ownership of MLIR IR.
    pub fn RYFT_XLA_Triton_Compile(args: *mut RYFT_XLA_Triton_Compile_Args) -> MlirLogicalResult;

    /// Releases owned compilation output buffers and resets output fields.
    ///
    /// # Safety
    ///
    /// The argument pointer must be valid and its output fields must originate from this C API. Borrowed input handles
    /// remain owned by the caller. This function exposes the matching native allocator for FFI interoperability.
    pub fn RYFT_XLA_Triton_Compile_Args_Destroy(args: *mut RYFT_XLA_Triton_Compile_Args);

    /// Returns linked compiler versions without creating a GPU context.
    ///
    /// # Safety
    ///
    /// The returned assembler version must be released with [`RYFT_XLA_Triton_Versions_Destroy`]. The remaining strings
    /// borrow process-lifetime storage. This function exposes native build and toolchain identity to compiler adapters.
    pub fn RYFT_XLA_Triton_Get_Versions() -> RYFT_XLA_Triton_Versions;

    /// Releases the owned assembler version and resets all version fields.
    ///
    /// # Safety
    ///
    /// The argument must be a version result from [`RYFT_XLA_Triton_Get_Versions`] or a previously destroyed result.
    /// This function exposes the matching native allocator for FFI interoperability.
    pub fn RYFT_XLA_Triton_Versions_Destroy(versions: *mut RYFT_XLA_Triton_Versions);
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use pretty_assertions::assert_eq;

    use crate::bindings::{
        mlirContextCreate, mlirContextDestroy, mlirDialectHandleLoadDialect, mlirModuleCreateParse, mlirModuleDestroy,
        mlirModuleGetOperation, mlirOperationPrint, mlirOperationVerify,
    };
    use crate::mlir::dialects::triton::tt::mlirGetDialectHandle__tt__;

    use super::*;

    /// Borrows a test string for the duration of a native call.
    fn from_string(value: &str) -> MlirStringRef {
        MlirStringRef { data: value.as_ptr().cast(), length: value.len() }
    }

    /// Copies synchronous native printer fragments into caller-owned storage.
    unsafe extern "C" fn append_string(fragment: MlirStringRef, storage: *mut c_void) {
        let storage = unsafe { &mut *storage.cast::<Vec<u8>>() };
        storage.extend_from_slice(unsafe { std::slice::from_raw_parts(fragment.data.cast(), fragment.length) });
    }

    #[test]
    fn test_ryft_xla_triton_compile_args_new() {
        let module = MlirModule { ptr: std::ptr::null_mut() };
        let arguments =
            RYFT_XLA_Triton_Compile_Args::new(module, from_string("cuda"), from_string("8.0"), 4, 2, 1024, 512);
        assert!(arguments.module.ptr.is_null());
        assert_eq!((arguments.warp_count, arguments.stage_count), (4, 2));
        assert_eq!((arguments.maximum_artifact_bytes, arguments.maximum_diagnostic_bytes), (1024, 512));
        assert!(arguments.artifact.is_null());
        assert!(arguments.entry_name.is_null());
        assert!(arguments.diagnostics.is_null());
        assert_eq!((arguments.artifact_size, arguments.entry_name_size, arguments.diagnostics_size), (0, 0, 0));
        assert_eq!(arguments.argument_count, 0);
        assert_eq!(arguments.actual_warp_count, 0);
        assert_eq!(arguments.threads_per_warp, 0);
        assert_eq!(arguments.shared_memory_bytes, 0);
    }

    #[test]
    fn test_ryft_xla_triton_compile() {
        let mut versions = unsafe { RYFT_XLA_Triton_Get_Versions() };
        let context = unsafe { mlirContextCreate() };
        unsafe { mlirDialectHandleLoadDialect(mlirGetDialectHandle__tt__(), context) };
        let module = unsafe {
            mlirModuleCreateParse(
                context,
                from_string("module { tt.func public @ryft_kernel(%argument: !tt.ptr<f32>) { tt.return } }"),
            )
        };
        assert!(!module.ptr.is_null());
        let operation = unsafe { mlirModuleGetOperation(module) };
        let mut before = Vec::<u8>::new();
        unsafe { mlirOperationPrint(operation, Some(append_string), (&mut before as *mut Vec<u8>).cast()) };
        let (platform, architecture) = if versions.cuda_available { ("cuda", "8.0") } else { ("rocm", "gfx942") };
        let mut arguments = RYFT_XLA_Triton_Compile_Args::new(
            module,
            from_string(platform),
            from_string(architecture),
            4,
            2,
            1024 * 1024,
            1024 * 1024,
        );
        let result = unsafe { RYFT_XLA_Triton_Compile(&mut arguments) };
        if versions.rocm_available {
            assert_eq!(result.value, 1);
            assert_eq!(arguments.diagnostics_size, 0);
            assert!(arguments.artifact_size > 0);
            assert_eq!(arguments.argument_count, 1);
            assert_eq!(arguments.actual_warp_count, 4);
            assert_eq!(arguments.threads_per_warp, if versions.cuda_available { 32 } else { 64 });
            assert_eq!(arguments.shared_memory_bytes, 0);
            assert_eq!(
                unsafe { std::slice::from_raw_parts(arguments.entry_name, arguments.entry_name_size) },
                b"ryft_kernel",
            );
        } else {
            assert_eq!(result.value, 0);
            assert_eq!(
                unsafe { std::slice::from_raw_parts(arguments.diagnostics, arguments.diagnostics_size) },
                b"triton compilation is unavailable in this native archive",
            );
        }
        if versions.rocm_available {
            unsafe { RYFT_XLA_Triton_Compile_Args_Destroy(&mut arguments) };
            arguments.maximum_artifact_bytes = 1;
            assert_eq!(unsafe { RYFT_XLA_Triton_Compile(&mut arguments) }.value, 0);
            assert!(arguments.artifact.is_null());
            assert_eq!(arguments.artifact_size, 0);
            assert_eq!(
                unsafe { std::slice::from_raw_parts(arguments.diagnostics, arguments.diagnostics_size) },
                b"compiler artifact exceeds its byte limit",
            );
        }
        let mut after = Vec::<u8>::new();
        unsafe { mlirOperationPrint(operation, Some(append_string), (&mut after as *mut Vec<u8>).cast()) };
        assert_eq!(before, after);
        assert!(unsafe { mlirOperationVerify(operation) });
        unsafe {
            RYFT_XLA_Triton_Compile_Args_Destroy(&mut arguments);
            RYFT_XLA_Triton_Versions_Destroy(&mut versions);
            mlirModuleDestroy(module);
            mlirContextDestroy(context);
        }
    }

    #[test]
    fn test_ryft_xla_triton_compile_null_module_and_diagnostic_limit() {
        let mut versions = unsafe { RYFT_XLA_Triton_Get_Versions() };
        let mut arguments = RYFT_XLA_Triton_Compile_Args::new(
            MlirModule { ptr: std::ptr::null_mut() },
            from_string("rocm"),
            from_string("gfx942"),
            4,
            2,
            1024,
            1024,
        );
        assert_eq!(unsafe { RYFT_XLA_Triton_Compile(&mut arguments) }.value, 0);
        let expected: &[u8] = if versions.rocm_available {
            b"expected a non-null Triton module"
        } else {
            b"triton compilation is unavailable in this native archive"
        };
        assert_eq!(unsafe { std::slice::from_raw_parts(arguments.diagnostics, arguments.diagnostics_size) }, expected,);
        unsafe { RYFT_XLA_Triton_Compile_Args_Destroy(&mut arguments) };
        arguments.maximum_diagnostic_bytes = 4;
        assert_eq!(unsafe { RYFT_XLA_Triton_Compile(&mut arguments) }.value, 0);
        assert_eq!(arguments.diagnostics_size, 4);
        assert_eq!(unsafe { std::slice::from_raw_parts(arguments.diagnostics, 4) }, &expected[..4]);
        unsafe {
            RYFT_XLA_Triton_Compile_Args_Destroy(&mut arguments);
            RYFT_XLA_Triton_Versions_Destroy(&mut versions);
        }
    }

    #[test]
    fn test_ryft_xla_triton_compile_args_destroy() {
        let mut arguments = RYFT_XLA_Triton_Compile_Args::new(
            MlirModule { ptr: std::ptr::null_mut() },
            from_string("invalid"),
            from_string("8.0"),
            4,
            2,
            1024,
            1024,
        );
        assert_eq!(unsafe { RYFT_XLA_Triton_Compile(&mut arguments) }.value, 0);
        assert!(arguments.diagnostics_size > 0);
        unsafe { RYFT_XLA_Triton_Compile_Args_Destroy(&mut arguments) };
        assert!(arguments.artifact.is_null());
        assert!(arguments.entry_name.is_null());
        assert!(arguments.diagnostics.is_null());
        assert_eq!((arguments.artifact_size, arguments.entry_name_size, arguments.diagnostics_size), (0, 0, 0));
        unsafe { RYFT_XLA_Triton_Compile_Args_Destroy(&mut arguments) };
    }
    #[test]
    fn test_ryft_xla_triton_get_versions() {
        let mut versions = unsafe { RYFT_XLA_Triton_Get_Versions() };
        let revision = unsafe { std::slice::from_raw_parts(versions.xla.data.cast::<u8>(), versions.xla.length) };
        assert_eq!(revision, crate::XLA_COMMIT.as_bytes());
        assert_eq!(versions.jax.length, 40);
        assert_eq!(versions.triton.length, 40);
        assert_eq!(versions.rocm_device_libs.length, 40);
        assert!(versions.assembler_version_size > 0);
        assert_eq!(versions.cuda_toolkit_version > 0, versions.cuda_available);
        assert!(!versions.cuda_available || versions.rocm_available);
        unsafe { RYFT_XLA_Triton_Versions_Destroy(&mut versions) };
    }

    #[test]
    fn test_ryft_xla_triton_versions_destroy() {
        let mut versions = unsafe { RYFT_XLA_Triton_Get_Versions() };
        unsafe { RYFT_XLA_Triton_Versions_Destroy(&mut versions) };
        assert!(versions.assembler_version.is_null());
        assert_eq!(versions.assembler_version_size, 0);
        assert_eq!(versions.xla.length, 0);
        assert_eq!(versions.cuda_toolkit_version, 0);
        assert!(!versions.cuda_available);
        assert!(!versions.rocm_available);
        unsafe { RYFT_XLA_Triton_Versions_Destroy(&mut versions) };
    }
}
