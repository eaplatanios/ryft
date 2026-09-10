# **Ryft CUDA:** CUDA Kernel Loading & Execution

This crate provides Rust APIs for loading CUDA cubin and PTX artifacts and launching kernels through the CUDA Driver
API. It represents kernel signatures and launch dimensions explicitly, packs typed arguments, and caches loaded modules
by CUDA context, device, image, and symbol. Artifacts can come from any producer that supplies the compiled code and
its corresponding metadata.

`ryft-cuda` has no PJRT, XLA, or CUDA toolkit build dependency. Kernel execution requires a CUDA 12.0-or-newer driver;
artifact construction does not require a CUDA installation. Frameworks supply the CUDA contexts, streams, and device
allocations. The [`ryft-pjrt`](../ryft-pjrt) crate provides adapters for XLA FFI execution contexts and buffers.

## Introduction

The following is an example for how you can describe a small PTX kernel that adds one to an unsigned integer and writes
it to a device pointer. It packages the kernel with its argument signature and launch dimensions, ready for execution:

```rust
use ryft_cuda::{
    CudaArtifactFormat, CudaKernelAbi, CudaKernelArtifact, CudaKernelLaunchDimensions,
    CudaKernelParameterType, CudaScalarType, Error,
};

fn main() -> Result<(), Error> {
    let ptx = br#"
        .version 8.0
        .target sm_80
        .address_size 64

        .visible .entry add_one(.param .u64 output, .param .u32 input) {
            .reg .u64 address;
            .reg .u32 value;
            ld.param.u64 address, [output];
            ld.param.u32 value, [input];
            add.u32 value, value, 1;
            st.global.u32 [address], value;
            ret;
        }
    "#;
    let abi = CudaKernelAbi::new(
        "example.add_one",
        1,
        [CudaKernelParameterType::DevicePointer, CudaKernelParameterType::Scalar(CudaScalarType::U32)],
    )?;
    let artifact = CudaKernelArtifact::new(
        CudaArtifactFormat::Ptx,
        ptx.to_vec(),
        "add_one",
        "compute_80",
        CudaKernelLaunchDimensions::new([1; 3], [1; 3], 0)?,
        abi,
    )?;
    println!("Kernel: {}", artifact.symbol());
    Ok(())
}
```

At a high level, a typical workflow for working with `ryft-cuda` looks as follows:

1. **Describe a Kernel:** Construct a `CudaKernelArtifact` from cubin or PTX bytes, the entrypoint symbol, target
   architecture, argument ABI, and grid, block, and dynamic shared-memory requirements. Use `with_launch_dimensions`
   to change those dimensions without copying or rehashing the compiled image.
2. **Create a Launcher:** Create a `CudaKernelLauncher` for the required `CudaVersion`. The launcher loads the driver
   dynamically and reuses cached modules and functions across launches. It configures each function's shared memory
   opt-in allowance once, so later launches can request different amounts within the device limit.
3. **Supply Arguments & Launch:** Borrow an existing non-default stream and device allocations using `CudaStream` and
   `CudaDevicePointer`, then build a `CudaKernelLaunch` with device pointers and typed `CudaScalarValue` arguments.
   Call the launcher's unsafe `launch` function with the artifact and arguments.
4. **Wait & Clean Up:** Synchronize through the framework that owns the stream before accessing results or releasing
   allocations. Call `clear_context` before retiring a context, or `shutdown` before destroying all context owners.
   The [crate documentation](src/lib.rs) includes an integration example showing enqueue, synchronization, and shutdown.

### Execution & Ownership

Artifact construction validates metadata and, for cubins, the ELF64 CUDA header. It does not establish executable code
safety or verify the kernel's actual signature. Ordinary cubins support the same major and a greater or equal minor
compute capability. Ordinary PTX targets support forward Just-In-Time (JIT) compilation. Restricted architecture and
family targets are delegated to CUDA. Refer to [NVIDIA's official compatibility documentation](
https://docs.nvidia.com/cuda/archive/13.0.0/blackwell-compatibility-guide/index.html) for more information.

Framework integrations must provide an accurate ABI, valid memory extents, access permissions and aliasing, and correct
stream ordering. A successful launch means work was enqueued and allocations and external resources must remain valid
until GPU completion. Supported arguments are non-null device pointers and the scalar types in `CudaScalarType`. Default
streams, optional null pointers, by-value aggregates, cooperative launches, and runtime cluster configuration are not
supported.

Graph capture is also unsupported. A capturing launch stream is rejected before the cache changes. Callers must exclude
capture on other streams in contexts synchronized during eviction or cleanup, including destructor cleanup, and must
not retain captured graph references to modules owned by the launcher. These requirements apply across host threads.

Explicit cleanup returns errors and retains resources that need another attempt. Destructor failures are available
through `Error::take_cleanup_errors`; context destruction ultimately releases any remaining modules. Cache mutations
and synchronous eviction share one mutex. Cache partitions have separate resource budgets and pending cleanup retries,
but slow module operations can delay other partitions.

## Verification

Portable unit tests exercise the real driver adapter through function-pointer stubs, the bootstrap resolver, artifact
validation, every scalar storage representation, and deterministic cache concurrency/failure scenarios:

```sh
cargo test -p ryft-cuda
cargo clippy -p ryft-cuda --all-targets --all-features -- -D warnings
cargo +nightly fmt -p ryft-cuda --check
```

On Linux, the ordinary unit suite also exercises real CUDA execution when a CUDA 12-or-newer driver and an
Ampere-or-newer NVIDIA GPU (compute capability 8.0+) are available. Missing prerequisites skip those cases with a
message. Unexpected driver errors and incorrect results fail. The cubin case additionally needs `nvcc` and skips if
its default path `/usr/local/cuda/bin/nvcc` is absent. `CUDA_NVCC` can override that path; an invalid explicit override
fails. PTX cases do not require the toolkit. These tests verify output/completion, larger shared-memory launches,
cache reuse, context cleanup, and preservation of graph capture when a launch is rejected.

To select the real-driver cases and show prerequisite messages:

```sh
timeout 300 cargo test -p ryft-cuda --lib _gpu -- --nocapture
```

#### License

<sup>
Licensed under either <a href="../../LICENSE-APACHE">Apache License, Version 2.0</a>
or <a href="../../LICENSE-MIT">MIT license</a> at your option.
</sup>

<br>

<sub>
Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this crate by you,
as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
</sub>
