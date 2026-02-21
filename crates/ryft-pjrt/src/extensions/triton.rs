use crate::{Api, Client, Device, Error, Plugin, Value, invoke_pjrt_api_error_fn, slice_from_c_api};

/// The PJRT Triton extension provides capabilities for compiling [Triton](https://github.com/triton-lang/triton)
/// kernels through PJRT [`Plugin`]s. The extension is both optional for PJRT [`Plugin`]s and _experimental_,
/// meaning that incompatible changes may be introduced at any time, including changes that break
/// _Application Binary Interface (ABI)_ compatibility.
#[derive(Copy, Clone)]
pub struct TritonExtension {
    /// Handle that represents this [`TritonExtension`] in the PJRT C API.
    handle: *const ffi::PJRT_Triton_Extension,

    /// Underlying PJRT [`Api`].
    api: Api,
}

impl TritonExtension {
    /// Constructs a new [`TritonExtension`] from the provided
    /// [`PJRT_Extension_Base`](crate::ffi::PJRT_Extension_Base) handle if the type of that PJRT
    /// extension matches the PJRT Triton extension type.
    pub(crate) unsafe fn from_c_api(handle: *const crate::ffi::PJRT_Extension_Base, api: Api) -> Option<Self> {
        unsafe {
            if !handle.is_null() && (*handle).extension_type == crate::ffi::PJRT_Extension_Type_Triton {
                Some(Self { handle: handle as *const _, api })
            } else {
                None
            }
        }
    }

    /// Returns the [`PJRT_Triton_Extension`](ffi::PJRT_Triton_Extension) that corresponds
    /// to this [`TritonExtension`] and which can be passed to functions in the PJRT C API.
    pub(crate) unsafe fn to_c_api(&self) -> *const ffi::PJRT_Triton_Extension {
        self.handle
    }

    /// Returns the underlying PJRT [`Api`].
    pub(crate) fn api(&self) -> Api {
        self.api
    }
}

unsafe impl Send for TritonExtension {}
unsafe impl Sync for TritonExtension {}

impl Device<'_> {
    /// Compiles the provided [Triton](https://github.com/triton-lang/triton) kernel module, for the hardware
    /// architecture of this [`Device`]. This function expects that this [`Device`] is a GPU device and exposes
    /// a `"compute_capability"` attribute whose value is already in the format expected by
    /// [`Client::compile_triton_kernel`] (i.e., a `"major.minor"` string for CUDA, like `"8.9"`, or a GFX version
    /// string for ROCm like `"gfx90a"`).
    ///
    /// Refer to the documentation of [`Client::compile_triton_kernel`]
    /// for more information on Triton kernel compilation.
    pub fn compile_triton_kernel<M: AsRef<[u8]>>(
        &self,
        module: M,
        warp_count: i32,
        cta_count: i32,
        stage_count: i32,
    ) -> Result<TritonKernel, Error> {
        match self.attribute("compute_capability")? {
            Value::String(architecture) => {
                self.api().compile_triton_kernel(module, architecture, warp_count, cta_count, stage_count)
            }
            value => Err(Error::invalid_argument(format!(
                "expected a string `compute_capability` attribute but got {value} instead"
            ))),
        }
    }
}

impl Client<'_> {
    /// Attempts to load the [`TritonExtension`] from this [`Client`] and returns [`Error::Unimplemented`]
    /// if it is not provided by the underlying [`Plugin`].
    pub fn triton_extension(&self) -> Result<TritonExtension, Error> {
        self.api().triton_extension()
    }

    /// Compiles a [Triton](https://github.com/triton-lang/triton) kernel using the provided
    /// [Triton Intermediate Representation (TTIR)](https://triton-lang.org/main/dialects/triton_ir.html)
    /// expressed as serialized MLIR bytecode. The compilation pipeline lowers the TTIR through backend-specific
    /// stages (e.g., TTIR → TTGIR → LLVM IR → PTX for NVIDIA, or TTIR → TTGIR → LLVM IR → HSACO for AMD) and returns
    /// a [`TritonKernel`] that contains the resulting backend assembly alongside launch metadata.
    ///
    /// Note that the PJRT Triton extension is **experimental** and only available on GPU [`Plugin`]s. CPU, TPU,
    /// and calling this function on other backends will result in it returning an [`Error::Unimplemented`].
    ///
    /// # Parameters
    ///
    ///   - `module`: MLIR bytecode containing a Triton TTIR module. The MLIR context used for parsing is pre-loaded
    ///     with the Triton dialect (and backend-specific Triton GPU dialects), so the module may reference Triton
    ///     operations such as `tt.func`, `tt.load`, `tt.store`, `tt.get_program_id`, etc. as well as standard
    ///     MLIR dialects like `arith` and `math`. Note that Triton IR is **not** guaranteed to be stable across
    ///     releases and so modules that were valid for one version of the underlying PJRT [`Plugin`] may fail to
    ///     parse on a newer version.
    ///   - `architecture`: Target GPU architecture string whose format is backend-specific:
    ///     - **NVIDIA (CUDA):** A `"major.minor"` compute-capability string (e.g., `"8.0"` for Ampere A100, `"9.0"`
    ///       for Hopper H100). Note that this is **not** specified in the `sm_XX` format; the PJRT [`TritonExtension`]
    ///       specifically expects the dot-separated representation.
    ///     - **AMD (ROCm):** A GCN architecture name such as `"gfx90a"` (MI200) or `"gfx942"` (MI300). Feature
    ///       flags may optionally be appended (e.g., `"gfx90a:sramecc+:xnack-"`).
    ///   - `warp_count`: Number of warps (each consisting of 32 threads on NVIDIA hardware, for example) that
    ///     will execute the resulting kernel within a single _Cooperative Thread Array (CTA)_. This value controls
    ///     intra-block parallelism and register pressure. Typical values are powers of two (e.g., `1`, `2`, `4`,
    ///     or `8`).
    ///   - `cta_count`: Number of _Cooperative Thread Arrays (CTAs)_ in a cooperative cluster. This enables
    ///     distributed shared memory across clustered CTAs (which were first introduced for NVIDIA GPUs in the Hopper
    ///     architecture). You must use the value `1` on hardware that does not support clustered CTAs or when
    ///     clustering is not needed.
    ///   - `stage_count`: Number of software-pipelining stages for memory accesses. Higher values can hide memory
    ///     latency by overlapping loads with computation at the cost of increased register pressure. Typical values
    ///     range from `1` to `4`.
    pub fn compile_triton_kernel<M: AsRef<[u8]>, A: AsRef<str>>(
        &self,
        module: M,
        architecture: A,
        warp_count: i32,
        cta_count: i32,
        stage_count: i32,
    ) -> Result<TritonKernel, Error> {
        self.api().compile_triton_kernel(module, architecture, warp_count, cta_count, stage_count)
    }
}

impl Plugin {
    /// Attempts to load the [`TritonExtension`] from this [`Plugin`] and returns [`Error::Unimplemented`]
    /// if it is not provided by this [`Plugin`].
    pub fn triton_extension(&self) -> Result<TritonExtension, Error> {
        self.api().triton_extension()
    }

    /// Compiles the provided [Triton](https://github.com/triton-lang/triton) kernel module. Refer to the documentation of
    /// [`Client::compile_triton_kernel`] for more information.
    pub fn compile_triton_kernel<M: AsRef<[u8]>, A: AsRef<str>>(
        &self,
        module: M,
        architecture: A,
        warp_count: i32,
        cta_count: i32,
        stage_count: i32,
    ) -> Result<TritonKernel, Error> {
        self.api().compile_triton_kernel(module, architecture, warp_count, cta_count, stage_count)
    }
}

impl Api {
    /// Attempts to load the [`TritonExtension`] from this [`Api`] and returns [`Error::Unimplemented`]
    /// if it is not provided by the underlying [`Plugin`].
    pub(crate) fn triton_extension(&self) -> Result<TritonExtension, Error> {
        unsafe {
            let mut extension = (*self.to_c_api()).extension_start;
            while !extension.is_null() {
                let triton_extension = TritonExtension::from_c_api(extension, *self);
                if let Some(triton_extension) = triton_extension {
                    return Ok(triton_extension);
                }
                extension = (*extension).next;
            }
            Err(Error::unimplemented("the Triton extension is not provided by the PJRT plugin"))
        }
    }

    /// Compiles the provided [Triton](https://github.com/triton-lang/triton) kernel module. Refer to the documentation
    /// of [`Client::compile_triton_kernel`] for more information.
    pub(crate) fn compile_triton_kernel<M: AsRef<[u8]>, A: AsRef<str>>(
        &self,
        module: M,
        architecture: A,
        warp_count: i32,
        cta_count: i32,
        stage_count: i32,
    ) -> Result<TritonKernel, Error> {
        use ffi::PJRT_Triton_Compile_Args;
        let extension = self.triton_extension()?;
        let module = module.as_ref();
        let architecture = architecture.as_ref().as_bytes();
        invoke_pjrt_api_error_fn!(
            @unchecked extension,
            PJRT_Triton_Compile,
            {
                module = module.as_ptr() as *const _,
                module_size = module.len(),
                arch_name = architecture.as_ptr() as *const _,
                arch_name_size = architecture.len(),
                num_warps = warp_count,
                num_ctas = cta_count,
                num_stages = stage_count,
            },
            {
                out_asm,
                out_asm_size,
                out_smem_bytes,
                out_cluster_dim_x,
                out_cluster_dim_y,
                out_cluster_dim_z,
            },
        )
        .map(
            |(out_asm, out_asm_size, out_smem_bytes, out_cluster_dim_x, out_cluster_dim_y, out_cluster_dim_z)| {
                TritonKernel {
                    assembly: unsafe { slice_from_c_api(out_asm as *const u8, out_asm_size) }.to_vec(),
                    shared_memory_bytes: out_smem_bytes,
                    thread_block_cluster_dimensions: [out_cluster_dim_x, out_cluster_dim_y, out_cluster_dim_z],
                }
            },
        )
    }
}

/// Represents a compiled Triton kernel (i.e., a Triton kernel that was compiled via [`Client::compile_triton_kernel`]
/// or [`Plugin::compile_triton_kernel`]).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TritonKernel {
    /// Assembly code for the kernel generated by Triton.
    pub assembly: Vec<u8>,

    /// Shared-memory requirement of this kernel as a number of bytes.
    pub shared_memory_bytes: i64,

    /// [Thread block cluster](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-clusters)
    /// shape for this kernel. Thread block clusters are a feature that was first introduced in the Hopper architecture
    /// (i.e., they are an `sm_90+` feature for NVIDIA GPUs) that groups multiple _Cooperative Thread Arrays (CTAs)_ so
    /// that they can cooperatively access each other's shared memory via _Distributed Shared Memory (DSMEM)_. Each
    /// element specifies how many CTAs sit along the corresponding axis of the cluster. For example, `[2, 1, 1]` means
    /// each cluster contains two CTAs arranged along the `x` axis. On pre-Hopper hardware, or when the kernel was
    /// compiled with `cta_count = 1`, all dimension sizes will be `0` indicating that clustering is not applicable.
    pub thread_block_cluster_dimensions: [i32; 3],
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use crate::errors::ffi::PJRT_Error;
    use crate::ffi::PJRT_Extension_Base;

    pub const PJRT_API_TRITON_EXTENSION_VERSION: usize = 1;

    #[repr(C)]
    pub struct PJRT_Triton_Compile_Args {
        pub struct_size: usize,
        pub module: *const std::ffi::c_char,
        pub module_size: usize,
        pub arch_name: *const std::ffi::c_char,
        pub arch_name_size: usize,
        pub num_warps: i32,
        pub num_ctas: i32,
        pub num_stages: i32,
        pub out_asm: *const std::ffi::c_char,
        pub out_asm_size: usize,
        pub out_smem_bytes: i64,
        pub out_cluster_dim_x: i32,
        pub out_cluster_dim_y: i32,
        pub out_cluster_dim_z: i32,
    }

    impl PJRT_Triton_Compile_Args {
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            module: *const std::ffi::c_char,
            module_size: usize,
            arch_name: *const std::ffi::c_char,
            arch_name_size: usize,
            num_warps: i32,
            num_ctas: i32,
            num_stages: i32,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                module,
                module_size,
                arch_name,
                arch_name_size,
                num_warps,
                num_ctas,
                num_stages,
                out_asm: std::ptr::null(),
                out_asm_size: 0,
                out_smem_bytes: 0,
                out_cluster_dim_x: 0,
                out_cluster_dim_y: 0,
                out_cluster_dim_z: 0,
            }
        }
    }

    pub type PJRT_Triton_Compile = unsafe extern "C" fn(args: *mut PJRT_Triton_Compile_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Triton_Extension {
        pub base: PJRT_Extension_Base,
        pub PJRT_Triton_Compile: Option<PJRT_Triton_Compile>,
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;

    use crate::Error;
    use crate::tests::{TestPlatform, test_for_each_platform};

    #[test]
    fn test_triton_extension() {
        test_for_each_platform!(|plugin, client, platform| {
            match platform {
                TestPlatform::Cuda12 | TestPlatform::Cuda13 | TestPlatform::Rocm7 => {
                    assert!(plugin.triton_extension().is_ok());
                    assert!(client.triton_extension().is_ok());
                }
                _ => {
                    assert!(matches!(plugin.triton_extension(), Err(Error::Unimplemented { .. })));
                    assert!(matches!(client.triton_extension(), Err(Error::Unimplemented { .. })));
                }
            }
        });
    }

    #[test]
    fn test_client_compile_triton_kernel() {
        test_for_each_platform!(|plugin, client, platform| {
            match platform {
                TestPlatform::Cuda12 | TestPlatform::Cuda13 | TestPlatform::Rocm7 => {
                    let device = client.addressable_devices().unwrap().remove(0);

                    // Test using a minimal Triton TTIR vector-add module (same as the one used by XLA's own
                    // [`triton_test.cc`](https://github.com/openxla/xla/blob/main/xla/pjrt/triton_test.cc) suite).
                    let module = indoc! {"
                        module {
                            tt.func public @add_kernel(
                                %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                %arg3: i32 {tt.divisibility = 16 : i32}) {
                                %c1024_i32 = arith.constant 1024 : i32
                                %0 = tt.get_program_id x : i32
                                %1 = arith.muli %0, %c1024_i32 : i32
                                %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
                                %3 = tt.splat %1 : i32 -> tensor<1024xi32>
                                %4 = arith.addi %3, %2 : tensor<1024xi32>
                                %5 = tt.splat %arg3 : i32 -> tensor<1024xi32>
                                %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32>
                                %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
                                %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
                                %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>>
                                %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
                                %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
                                %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<f32>>
                                %13 = arith.addf %9, %12 : tensor<1024xf32>
                                %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
                                %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
                                tt.store %15, %13, %6 : tensor<1024x!tt.ptr<f32>>
                                tt.return
                            }
                        }
                    "};
                    let kernel = device.compile_triton_kernel(module.as_bytes(), 4, 1, 2).unwrap();
                    assert!(!kernel.assembly.is_empty());
                    assert!(kernel.thread_block_cluster_dimensions.iter().all(|&d| d >= 0));

                    // Test using an invalid module.
                    let module = indoc! {"
                        module {
                            tt.func public @add_kernel(
                                %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                %arg3: i32 {tt.divisibility = 16 : i32}) {
                                %c1024_i32 = arith.constant 1024 : i32
                                %0 = tt.get_program_id x : i64
                                %1 = arith.muli %0, %c1024_i32 : i32
                                %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
                                %3 = tt.splat %1 : i32 -> tensor<1024xi32>
                                %4 = arith.addi %3, %2 : tensor<1024xi32>
                                %5 = tt.splat %arg3 : i32 -> tensor<1024xi32>
                                %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32>
                                %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
                                %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
                                %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>>
                                %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
                                %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
                                %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<f32>>
                                %13 = arith.addf %9, %12 : tensor<1024xf32>
                                %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
                                %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
                                tt.store %15, %13, %6 : tensor<1024x!tt.ptr<f32>>
                                tt.return
                            }
                        }
                    "};
                    assert!(matches!(
                        device.compile_triton_kernel(module.as_bytes(), 4, 1, 2),
                        Err(Error::InvalidArgument { message, .. }) if message == "Failed to parse Triton module",
                    ));
                }
                _ => {
                    assert!(matches!(plugin.triton_extension(), Err(Error::Unimplemented { .. })));
                    assert!(matches!(client.triton_extension(), Err(Error::Unimplemented { .. })));
                }
            }
        });
    }
}
