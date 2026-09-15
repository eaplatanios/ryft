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

## Optional cuTile Compiler

`ryft_cuda::kernels::cutile` compiles verified portable `ryft_core::kernels` definitions through NVIDIA's official
cuTile Python AOT interface. This optional module owns source generation, compiler admission, tool isolation, and
the adapter-authored manifest. It reuses this crate's producer-neutral artifacts and launcher without depending on
XLA, PJRT, or MLIR.

Enable `cutile` for the target, immutable output, and validated manifest decoder; it adds optional `ryft-core`,
`serde`, and `serde_json` dependencies. Enable `cutile-compiler` where ahead-of-time compilation runs; it includes
`cutile` and adds the compiler and `tempfile` dependency. Both features are disabled by default, preserving the
existing lightweight CUDA APIs. Python is not needed for manifest validation, CUDA loading, invocation, or executable
reload. XLA selection and `CuTileEmbedding` remain owned by `ryft-xla`.

### Installation and identity

Use an explicit Python executable with `cuda-tile==1.5.0`, `nvidia-cuda-tileiras==13.3.36`,
`nvidia-cuda-nvcc==13.3.73`, and `nvidia-nvvm==13.3.73`. The compiler checks
installed versions before invoking the exporter and the worker rechecks them before compilation. Configuration
identity includes the source schema, bundled worker hash, pinned versions, explicit executable path, target and
schedule. Cache identity construction does not start Python. Process timeouts and cancellation are control policies,
not numerical or scheduling choices. The worker verifies the actual distribution-owned compiler executable version
and namespace selection, selects bytecode version `13.3`, clears frontend environment overrides (including test flags
that can remove token ordering), and uses isolated temporary/cache directories. It does not silently fall back to a
system CUDA toolkit.

Supported target spellings are `sm_100`, `sm_103`, `sm_110`, `sm_120`, and `sm_121`, as accepted by the pinned tool.
Acceptance by the compiler does not establish driver compatibility or hardware execution qualification. The execution
integration must establish the actual device and CUDA runtime facts before dispatch.

### Artifact and ABI

The worker uses explicit `ArrayConstraint` values and `CallingConvention.cutile_python_v2()`. It never derives
constraints from example arrays or addresses. Physical arguments are a pointer followed by every signed I32 shape
component and then every signed I32 element stride. Static shapes and strides remain in this ABI. Constant values
inside the generated body do not become runtime arguments. Logical scalar arrays retain their rank-zero canonical
types but use a physical one-element array constraint `[1]`, stride `[1]`: the public AOT partition-view contract does
not support rank-zero array parameters. The lowerer loads/stores this element and reshapes private scalar values.

`Argument::Array(index)` refers to the canonical kernel parameter order. The execution integration maps read-only
parameters to inputs and writable parameters to results, retaining its ordinary input liveness and alias policy.
Zero-sized external arrays are rejected before compilation because the shared runtime does not admit null CUDA
pointers. An empty logical grid over nonempty arrays remains a valid no-op.
When two or more parameters exist, array constraints share an alias group so repeated read-only inputs remain legal.
A single parameter has an empty alias-group list because the exporter rejects redundant singleton alias groups.
Dense row-major storage has no internal element aliases; address alignment is conservatively one byte. Launch block
dimensions are `(1,1,1)` under CUDA Tile's launch convention, not a promise about the compiler's physical thread
allocation.

`CompiledKernel::from_manifest` checks the exact verified body identity, full logical parameter types, physical ABI,
producer versions and configuration, derived launch grid, cubin size/hash, and canonical ELF architecture. The
manifest is a Ryft adapter format, not an upstream cuTile schema. These checks detect mismatches and corruption;
executing artifacts still requires trusting their producer.

### Compiler process

Compilation runs in a fresh temporary directory and Unix process group. A caller-owned atomic cancellation signal,
a wall-clock deadline, and bounded diagnostic capture terminate the whole compiler group. Partial stdout and stderr
remain in errors. A failed invocation publishes no artifact and cannot poison an independent compilation. Successful
output retains diagnostics for inspection. The `cutile` feature does not impose a host operating-system requirement;
only the current compiler process implementation requires Unix.

### Portable subset

Source admission is operation-driven. Static dense arrays, bounded dimensions and reference windows use the same
canonical definition as other adapters. Arithmetic, shape operations, reductions and dot preserve their declared
numerical policies. Floating-point dot must not silently substitute reduced-precision TF32. Explicit masks govern
edge-window memory accesses. Unsupported operations, memory layouts, synchronization, target extensions and numerical
modes are rejected before the compiler process starts; no implicit backend fallback occurs.

The initial concrete subset is:

| Area | Admitted operations | Explicit limits |
|---|---|---|
| Values | Boolean, I32/U32, I64/U64, F16/BF16, F32/F64 | Static dense device arrays; manual-only local sharding |
| Arithmetic | Add/subtract/multiply/negate, floating divide, exp, min/max, compare/select | Native dtype contract |
| Shape | Reshape, broadcast, transpose, element conversion | Checked physical tile shapes and conversion mode |
| Reduction | Sum and maximum | Explicit neutral padding, admitted axes only |
| Dot | Ordinary matrix multiplication | Supported canonical dimensions and declared accumulation |
| References | Read, write, swap, bounded tile load | Global windows with explicit clipped gather/scatter masks |
| Control | Static grids, bounded while, condition | Reference ownership preserved; parallel carry assignment |

Explicit scratch, async copy, wait/barrier protocols, atomics, target extensions and block-scaled dot are not admitted.
Full manual-only sharding metadata remains on logical types; the execution integration must bind every named manual
axis in an enclosing `shard_map`. Automatic or explicit partitioning and unconstrained placement are rejected.
A scratch-byte schedule bound is rejected because this compiler cannot certify cuTile's physical scratch allocation.
The implementation's source owner tests define the detailed shape restrictions.
Native qualification compares unchanged definitions against the core interpreter and Mosaic GPU, rather than using
example-name recognition or hard-coded generated kernels.

Checked indexing retains native device assertions. XLA integration carries their ordered effect tokens outside the
CUDA argument list and reports failures when execution completes. The negative assertion test runs in a separate
process because a failed device assertion can invalidate its CUDA context.

DGX Spark qualification covers `sm_121`: partial vector tiles, scalar reduction, partial matrix multiplication,
FP32 precision, independent batches, read-write aliases, repeated input buffers, and a local manual shard map.
Serialized kernels reload in a fresh session with compiler cancellation enabled. Other accepted architectures need
their own device qualification.

Sources: [official compilation and export](https://docs.nvidia.com/cuda/cutile-python/compilation.html),
[CUDA Tile launch contract](https://github.com/NVIDIA/cuda-tile), and
[cuTile data model](https://docs.nvidia.com/cuda/cutile-python/data.html).

## Verification

Portable unit tests exercise the real driver adapter through function-pointer stubs, the bootstrap resolver, artifact
validation, every scalar storage representation, and deterministic cache concurrency/failure scenarios:

```sh
cargo test -p ryft-cuda
cargo test -p ryft-cuda --features cutile-compiler --lib kernels::cutile
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
