use std::env;
use std::ffi::{CString, c_char, c_void};
use std::sync::OnceLock;

use libloading::{Library, Symbol};

use crate::jax::JaxPlatform;

const CUDA_SUCCESS: i32 = 0;
const HIP_SUCCESS: i32 = 0;
const HIPRTC_SUCCESS: i32 = 0;

const CUDA_ADD_KERNEL_PTX: &str = r#"
.version 6.0
.target sm_50
.address_size 64

.visible .entry add_i32(
    .param .u64 lhs,
    .param .u64 rhs,
    .param .u64 out
)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<4>;

    ld.param.u64 %rd1, [lhs];
    ld.param.u64 %rd2, [rhs];
    ld.param.u64 %rd3, [out];

    ld.global.s32 %r1, [%rd1];
    ld.global.s32 %r2, [%rd2];
    add.s32 %r3, %r1, %r2;
    st.global.s32 [%rd3], %r3;
    ret;
}
"#;

const ROCM_ADD_KERNEL_SOURCE: &str = r#"
extern "C" __global__ void add_i32(const int* lhs, const int* rhs, int* out) {
  out[0] = lhs[0] + rhs[0];
}
"#;

enum BackendRuntime<T> {
    Available(T),
    Unavailable(String),
    Failed(String),
}

impl<T> BackendRuntime<T> {
    fn unavailable<M: Into<String>>(message: M) -> Self {
        Self::Unavailable(message.into())
    }

    fn failed<M: Into<String>>(message: M) -> Self {
        Self::Failed(message.into())
    }
}

type CuInit = unsafe extern "C" fn(flags: u32) -> i32;
type CuCtxGetCurrent = unsafe extern "C" fn(ctx: *mut *mut c_void) -> i32;
type CuModuleLoadDataEx = unsafe extern "C" fn(
    module: *mut *mut c_void,
    image: *const c_void,
    num_options: u32,
    options: *mut u32,
    option_values: *mut *mut c_void,
) -> i32;
type CuModuleLoadData = unsafe extern "C" fn(module: *mut *mut c_void, image: *const c_void) -> i32;
type CuModuleGetFunction =
    unsafe extern "C" fn(function: *mut *mut c_void, module: *mut c_void, name: *const c_char) -> i32;
type CuLaunchKernel = unsafe extern "C" fn(
    function: *mut c_void,
    grid_dim_x: u32,
    grid_dim_y: u32,
    grid_dim_z: u32,
    block_dim_x: u32,
    block_dim_y: u32,
    block_dim_z: u32,
    shared_mem_bytes: u32,
    stream: *mut c_void,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> i32;

type HipModuleLoadData = unsafe extern "C" fn(module: *mut *mut c_void, image: *const c_void) -> i32;
type HipModuleGetFunction =
    unsafe extern "C" fn(function: *mut *mut c_void, module: *mut c_void, name: *const c_char) -> i32;
type HipModuleLaunchKernel = unsafe extern "C" fn(
    function: *mut c_void,
    grid_dim_x: u32,
    grid_dim_y: u32,
    grid_dim_z: u32,
    block_dim_x: u32,
    block_dim_y: u32,
    block_dim_z: u32,
    shared_mem_bytes: u32,
    stream: *mut c_void,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> i32;

type HiprtcCreateProgram = unsafe extern "C" fn(
    program: *mut *mut c_void,
    src: *const c_char,
    name: *const c_char,
    num_headers: i32,
    headers: *const *const c_char,
    include_names: *const *const c_char,
) -> i32;
type HiprtcCompileProgram =
    unsafe extern "C" fn(program: *mut c_void, num_options: i32, options: *const *const c_char) -> i32;
type HiprtcDestroyProgram = unsafe extern "C" fn(program: *mut *mut c_void) -> i32;
type HiprtcGetProgramLogSize = unsafe extern "C" fn(program: *mut c_void, log_size: *mut usize) -> i32;
type HiprtcGetProgramLog = unsafe extern "C" fn(program: *mut c_void, log: *mut c_char) -> i32;
type HiprtcGetCodeSize = unsafe extern "C" fn(program: *mut c_void, code_size: *mut usize) -> i32;
type HiprtcGetCode = unsafe extern "C" fn(program: *mut c_void, code: *mut c_char) -> i32;

struct CudaRuntime {
    launch_kernel: CuLaunchKernel,
    add_function: *mut c_void,
}

unsafe impl Send for CudaRuntime {}
unsafe impl Sync for CudaRuntime {}

struct RocmRuntime {
    launch_kernel: HipModuleLaunchKernel,
    add_function: *mut c_void,
    add_code_object: Vec<u8>,
}

unsafe impl Send for RocmRuntime {}
unsafe impl Sync for RocmRuntime {}

static CUDA_RUNTIME: OnceLock<BackendRuntime<CudaRuntime>> = OnceLock::new();
static ROCM_RUNTIME: OnceLock<BackendRuntime<RocmRuntime>> = OnceLock::new();

pub(crate) fn add_i32_artifact(platform: &JaxPlatform) -> Result<Vec<u8>, String> {
    match platform {
        JaxPlatform::Cuda => Ok(CUDA_ADD_KERNEL_PTX.as_bytes().to_vec()),
        JaxPlatform::Rocm => rocm_runtime().map(|runtime| runtime.add_code_object.clone()),
        _ => Err(format!("{platform:?} does not support add_i32 GPU artifacts")),
    }
}

pub(crate) fn launch_add_i32(
    stream: *mut c_void,
    lhs_device: *mut c_void,
    rhs_device: *mut c_void,
    out_device: *mut c_void,
) -> Result<(), String> {
    if stream.is_null() {
        return Err("null GPU stream pointer".to_string());
    }
    if lhs_device.is_null() || rhs_device.is_null() || out_device.is_null() {
        return Err("null GPU buffer pointer".to_string());
    }

    match cuda_runtime() {
        Ok(runtime) => return launch_cuda_add(runtime, stream, lhs_device, rhs_device, out_device),
        Err(cuda_error) => {
            if let Ok(runtime) = rocm_runtime() {
                return launch_rocm_add(runtime, stream, lhs_device, rhs_device, out_device);
            }
            return Err(format!("no supported GPU runtime available; CUDA error: {cuda_error}"));
        }
    }
}

fn launch_cuda_add(
    runtime: &CudaRuntime,
    stream: *mut c_void,
    lhs_device: *mut c_void,
    rhs_device: *mut c_void,
    out_device: *mut c_void,
) -> Result<(), String> {
    let mut lhs_argument = lhs_device;
    let mut rhs_argument = rhs_device;
    let mut out_argument = out_device;
    let mut kernel_params = [
        (&mut lhs_argument as *mut *mut c_void).cast::<c_void>(),
        (&mut rhs_argument as *mut *mut c_void).cast::<c_void>(),
        (&mut out_argument as *mut *mut c_void).cast::<c_void>(),
    ];

    let launch_status = unsafe {
        (runtime.launch_kernel)(
            runtime.add_function,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            stream,
            kernel_params.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    if launch_status != CUDA_SUCCESS {
        return Err(format!("cuLaunchKernel failed with error code {launch_status}"));
    }
    Ok(())
}

fn launch_rocm_add(
    runtime: &RocmRuntime,
    stream: *mut c_void,
    lhs_device: *mut c_void,
    rhs_device: *mut c_void,
    out_device: *mut c_void,
) -> Result<(), String> {
    let mut lhs_argument = lhs_device;
    let mut rhs_argument = rhs_device;
    let mut out_argument = out_device;
    let mut kernel_params = [
        (&mut lhs_argument as *mut *mut c_void).cast::<c_void>(),
        (&mut rhs_argument as *mut *mut c_void).cast::<c_void>(),
        (&mut out_argument as *mut *mut c_void).cast::<c_void>(),
    ];

    let launch_status = unsafe {
        (runtime.launch_kernel)(
            runtime.add_function,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            stream,
            kernel_params.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    if launch_status != HIP_SUCCESS {
        return Err(format!("hipModuleLaunchKernel failed with error code {launch_status}"));
    }
    Ok(())
}

fn cuda_runtime() -> Result<&'static CudaRuntime, String> {
    match CUDA_RUNTIME.get_or_init(initialize_cuda_runtime) {
        BackendRuntime::Available(runtime) => Ok(runtime),
        BackendRuntime::Unavailable(message) | BackendRuntime::Failed(message) => Err(message.clone()),
    }
}

fn rocm_runtime() -> Result<&'static RocmRuntime, String> {
    match ROCM_RUNTIME.get_or_init(initialize_rocm_runtime) {
        BackendRuntime::Available(runtime) => Ok(runtime),
        BackendRuntime::Unavailable(message) | BackendRuntime::Failed(message) => Err(message.clone()),
    }
}

fn initialize_cuda_runtime() -> BackendRuntime<CudaRuntime> {
    let Some(library) = load_library(&["libcuda.so.1", "libcuda.so"]) else {
        return BackendRuntime::unavailable("failed to load libcuda".to_string());
    };

    let cu_init = match resolve_symbol::<CuInit>(&library, &[b"cuInit\0"]) {
        Ok(symbol) => symbol,
        Err(error) => return BackendRuntime::failed(error),
    };
    let cu_ctx_get_current = match resolve_symbol::<CuCtxGetCurrent>(&library, &[b"cuCtxGetCurrent\0"]) {
        Ok(symbol) => symbol,
        Err(error) => return BackendRuntime::failed(error),
    };
    let cu_module_load_data_ex = resolve_symbol::<CuModuleLoadDataEx>(&library, &[b"cuModuleLoadDataEx\0"]);
    let cu_module_load_data = resolve_symbol::<CuModuleLoadData>(&library, &[b"cuModuleLoadData\0"]);
    let cu_module_get_function = match resolve_symbol::<CuModuleGetFunction>(&library, &[b"cuModuleGetFunction\0"]) {
        Ok(symbol) => symbol,
        Err(error) => return BackendRuntime::failed(error),
    };
    let cu_launch_kernel = match resolve_symbol::<CuLaunchKernel>(&library, &[b"cuLaunchKernel\0"]) {
        Ok(symbol) => symbol,
        Err(error) => return BackendRuntime::failed(error),
    };

    std::mem::forget(library);

    let init_status = unsafe { cu_init(0) };
    if init_status != CUDA_SUCCESS {
        return BackendRuntime::failed(format!("cuInit failed with error code {init_status}"));
    }

    let mut current_context: *mut c_void = std::ptr::null_mut();
    let ctx_status = unsafe { cu_ctx_get_current(&mut current_context as *mut _) };
    if ctx_status != CUDA_SUCCESS {
        return BackendRuntime::failed(format!("cuCtxGetCurrent failed with error code {ctx_status}"));
    }
    if current_context.is_null() {
        return BackendRuntime::failed("cuCtxGetCurrent returned a null CUDA context".to_string());
    }

    let mut module = std::ptr::null_mut();
    let ptx = CString::new(CUDA_ADD_KERNEL_PTX).expect("static PTX should not contain NUL bytes");
    let module_status = match cu_module_load_data_ex {
        Ok(load_data_ex) => unsafe {
            load_data_ex(&mut module as *mut _, ptx.as_ptr().cast(), 0, std::ptr::null_mut(), std::ptr::null_mut())
        },
        Err(_) => match cu_module_load_data {
            Ok(load_data) => unsafe { load_data(&mut module as *mut _, ptx.as_ptr().cast()) },
            Err(error) => return BackendRuntime::failed(error),
        },
    };
    if module_status != CUDA_SUCCESS {
        return BackendRuntime::failed(format!("cuModuleLoadDataEx failed with error code {module_status}"));
    }

    let mut add_function = std::ptr::null_mut();
    let kernel_name = CString::new("add_i32").expect("static kernel name should not contain NUL bytes");
    let function_status = unsafe { cu_module_get_function(&mut add_function as *mut _, module, kernel_name.as_ptr()) };
    if function_status != CUDA_SUCCESS {
        return BackendRuntime::failed(format!("cuModuleGetFunction failed with error code {function_status}"));
    }

    BackendRuntime::Available(CudaRuntime { launch_kernel: cu_launch_kernel, add_function })
}

fn initialize_rocm_runtime() -> BackendRuntime<RocmRuntime> {
    let Some(hip_library) = load_library(&["libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"]) else {
        return BackendRuntime::unavailable("failed to load libamdhip64".to_string());
    };
    let Some(hiprtc_library) = load_library(&["libhiprtc.so", "libhiprtc.so.7", "libhiprtc.so.6"]) else {
        return BackendRuntime::unavailable("failed to load libhiprtc".to_string());
    };

    let hip_module_load_data = match resolve_symbol::<HipModuleLoadData>(&hip_library, &[b"hipModuleLoadData\0"]) {
        Ok(symbol) => symbol,
        Err(error) => return BackendRuntime::failed(error),
    };
    let hip_module_get_function =
        match resolve_symbol::<HipModuleGetFunction>(&hip_library, &[b"hipModuleGetFunction\0"]) {
            Ok(symbol) => symbol,
            Err(error) => return BackendRuntime::failed(error),
        };
    let hip_module_launch_kernel =
        match resolve_symbol::<HipModuleLaunchKernel>(&hip_library, &[b"hipModuleLaunchKernel\0"]) {
            Ok(symbol) => symbol,
            Err(error) => return BackendRuntime::failed(error),
        };
    let hiprtc_create_program =
        match resolve_symbol::<HiprtcCreateProgram>(&hiprtc_library, &[b"hiprtcCreateProgram\0"]) {
            Ok(symbol) => symbol,
            Err(error) => return BackendRuntime::failed(error),
        };
    let hiprtc_compile_program =
        match resolve_symbol::<HiprtcCompileProgram>(&hiprtc_library, &[b"hiprtcCompileProgram\0"]) {
            Ok(symbol) => symbol,
            Err(error) => return BackendRuntime::failed(error),
        };
    let hiprtc_destroy_program =
        match resolve_symbol::<HiprtcDestroyProgram>(&hiprtc_library, &[b"hiprtcDestroyProgram\0"]) {
            Ok(symbol) => symbol,
            Err(error) => return BackendRuntime::failed(error),
        };
    let hiprtc_get_program_log_size =
        match resolve_symbol::<HiprtcGetProgramLogSize>(&hiprtc_library, &[b"hiprtcGetProgramLogSize\0"]) {
            Ok(symbol) => symbol,
            Err(error) => return BackendRuntime::failed(error),
        };
    let hiprtc_get_program_log =
        match resolve_symbol::<HiprtcGetProgramLog>(&hiprtc_library, &[b"hiprtcGetProgramLog\0"]) {
            Ok(symbol) => symbol,
            Err(error) => return BackendRuntime::failed(error),
        };
    let hiprtc_get_code_size = match resolve_symbol::<HiprtcGetCodeSize>(&hiprtc_library, &[b"hiprtcGetCodeSize\0"]) {
        Ok(symbol) => symbol,
        Err(error) => return BackendRuntime::failed(error),
    };
    let hiprtc_get_code = match resolve_symbol::<HiprtcGetCode>(&hiprtc_library, &[b"hiprtcGetCode\0"]) {
        Ok(symbol) => symbol,
        Err(error) => return BackendRuntime::failed(error),
    };

    std::mem::forget(hip_library);
    std::mem::forget(hiprtc_library);

    let add_code_object = match compile_rocm_add_code_object(
        hiprtc_create_program,
        hiprtc_compile_program,
        hiprtc_destroy_program,
        hiprtc_get_program_log_size,
        hiprtc_get_program_log,
        hiprtc_get_code_size,
        hiprtc_get_code,
    ) {
        Ok(code) => code,
        Err(error) => return BackendRuntime::failed(error),
    };

    let mut module = std::ptr::null_mut();
    let module_status = unsafe { hip_module_load_data(&mut module as *mut _, add_code_object.as_ptr().cast()) };
    if module_status != HIP_SUCCESS {
        return BackendRuntime::failed(format!("hipModuleLoadData failed with error code {module_status}"));
    }

    let mut add_function = std::ptr::null_mut();
    let kernel_name = CString::new("add_i32").expect("static kernel name should not contain NUL bytes");
    let function_status = unsafe { hip_module_get_function(&mut add_function as *mut _, module, kernel_name.as_ptr()) };
    if function_status != HIP_SUCCESS {
        return BackendRuntime::failed(format!("hipModuleGetFunction failed with error code {function_status}"));
    }

    BackendRuntime::Available(RocmRuntime { launch_kernel: hip_module_launch_kernel, add_function, add_code_object })
}

#[allow(clippy::too_many_arguments)]
fn compile_rocm_add_code_object(
    create_program: HiprtcCreateProgram,
    compile_program: HiprtcCompileProgram,
    destroy_program: HiprtcDestroyProgram,
    get_program_log_size: HiprtcGetProgramLogSize,
    get_program_log: HiprtcGetProgramLog,
    get_code_size: HiprtcGetCodeSize,
    get_code: HiprtcGetCode,
) -> Result<Vec<u8>, String> {
    let source = CString::new(ROCM_ADD_KERNEL_SOURCE).expect("static HIP kernel source should not contain NUL bytes");
    let program_name = CString::new("ryft_add_i32.hip").expect("static program name should not contain NUL bytes");

    let mut program = std::ptr::null_mut();
    let create_status = unsafe {
        create_program(
            &mut program as *mut _,
            source.as_ptr(),
            program_name.as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        )
    };
    if create_status != HIPRTC_SUCCESS {
        return Err(format!("hiprtcCreateProgram failed with error code {create_status}"));
    }

    let mut option_strings = Vec::<CString>::new();
    if let Ok(arch) = env::var("RYFT_PJRT_ROCM_ARCH") {
        option_strings.push(
            CString::new(format!("--gpu-architecture={arch}"))
                .map_err(|_| "invalid RYFT_PJRT_ROCM_ARCH value".to_string())?,
        );
    }
    let option_pointers = option_strings.iter().map(|option| option.as_ptr()).collect::<Vec<_>>();

    let compile_status = unsafe {
        compile_program(
            program,
            option_pointers.len() as i32,
            if option_pointers.is_empty() { std::ptr::null() } else { option_pointers.as_ptr() },
        )
    };

    if compile_status != HIPRTC_SUCCESS {
        let log = hiprtc_program_log(program, get_program_log_size, get_program_log)
            .unwrap_or_else(|_| "<failed to read hiprtc log>".to_string());
        let _ = unsafe { destroy_program(&mut program as *mut _) };
        return Err(format!("hiprtcCompileProgram failed with error code {compile_status}; log: {log}"));
    }

    let mut code_size = 0usize;
    let size_status = unsafe { get_code_size(program, &mut code_size as *mut _) };
    if size_status != HIPRTC_SUCCESS {
        let _ = unsafe { destroy_program(&mut program as *mut _) };
        return Err(format!("hiprtcGetCodeSize failed with error code {size_status}"));
    }

    let mut code = vec![0u8; code_size];
    let code_status = unsafe { get_code(program, code.as_mut_ptr().cast::<c_char>()) };
    if code_status != HIPRTC_SUCCESS {
        let _ = unsafe { destroy_program(&mut program as *mut _) };
        return Err(format!("hiprtcGetCode failed with error code {code_status}"));
    }

    let destroy_status = unsafe { destroy_program(&mut program as *mut _) };
    if destroy_status != HIPRTC_SUCCESS {
        return Err(format!("hiprtcDestroyProgram failed with error code {destroy_status}"));
    }

    Ok(code)
}

fn hiprtc_program_log(
    program: *mut c_void,
    get_program_log_size: HiprtcGetProgramLogSize,
    get_program_log: HiprtcGetProgramLog,
) -> Result<String, String> {
    let mut log_size = 0usize;
    let size_status = unsafe { get_program_log_size(program, &mut log_size as *mut _) };
    if size_status != HIPRTC_SUCCESS {
        return Err(format!("hiprtcGetProgramLogSize failed with error code {size_status}"));
    }
    if log_size == 0 {
        return Ok("".to_string());
    }

    let mut log = vec![0u8; log_size];
    let log_status = unsafe { get_program_log(program, log.as_mut_ptr().cast::<c_char>()) };
    if log_status != HIPRTC_SUCCESS {
        return Err(format!("hiprtcGetProgramLog failed with error code {log_status}"));
    }

    Ok(String::from_utf8_lossy(log.as_slice()).trim_matches('\0').to_string())
}

fn load_library(candidates: &[&str]) -> Option<Library> {
    for &candidate in candidates {
        if let Ok(library) = unsafe { Library::new(candidate) } {
            return Some(library);
        }
    }
    None
}

fn resolve_symbol<T: Copy>(library: &Library, names: &[&[u8]]) -> Result<T, String> {
    for &name in names {
        let symbol: Result<Symbol<'_, T>, _> = unsafe { library.get(name) };
        if let Ok(symbol) = symbol {
            return Ok(*symbol);
        }
    }
    let names = names
        .iter()
        .map(|name| String::from_utf8_lossy(name).trim_end_matches('\0').to_string())
        .collect::<Vec<_>>()
        .join(", ");
    Err(format!("failed to resolve one of symbols: {names}"))
}
