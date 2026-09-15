# ryft-triton

Direct Triton compilation of verified Ryft kernels. The adapter constructs typed TTIR through `ryft-mlir` and invokes
an explicitly selected native compiler executable. It has no dependency on `ryft-xla`, Python, or JAX Pallas lowering.
NVIDIA artifacts use the existing `ryft-cuda` launcher; AMD artifacts use the concrete `ryft-rocm` HIP launcher.

## Supported contract

| Target | Compiler qualification | Hardware qualification |
| --- | --- | --- |
| CUDA `8.0` | PTX entry, pointer ABI, resources and native compiler tests | Not executed on SM80 |
| CUDA `12.1` | PTX entry, pointer ABI, resources and native compiler tests | DGX Spark GB10, driver 580.173.02 |
| ROCm `gfx908`, `gfx90a`, `gfx942` | Actual vector/dot HSACO and kernel descriptor inspection | No AMD machine available |

Source revisions are XLA `eb6b90ed013f511eca088c52f541f3c0819f919e`, JAX
`a7606f995e1a92707cbeb257e487fa53e7abe84b`, and Triton `a77e7c793abc0d0c923a9afb275058e2fe57a198` with the pinned XLA
patches. CUDA compilation requires the recorded CUDA 13.2 toolkit contract and assembler 13.0.88, producing PTX 9.0.
The runtime integration requires the pinned PJRT FFI contract. ROCm uses HIP 7.13 and the native compiler's embedded
ROCm device-bitcode revision `53996464fa8d94b182ac4aaa7dc3a109ab524f45`; see `ryft-rocm` for its exact ABI contract.
A full native build with CUDA configuration disabled has not been qualified.

The initial portable subset uses static, unsharded, dense row-major F32 device parameters, one global pointer per
logical parameter, and at most four axes per tile. AMD kernels admit one through 64 physical parameters. Immutable intermediates may also use I64 and Boolean values.
The lowerer supports masked loads/stores, indexed and tiled references, arithmetic, comparisons, selection,
broadcasting, transpose, power-of-two reshape, FP32 sum reduction, IEEE FP32 matrix multiplication, static parallel
grids and bounded control flow. Physical power-of-two padding is masked before reductions and contractions.

Zero-element physical parameters, read-write aliases, unspecialized scalar prefetch, sequential grid axes, dynamic
array extents, unsupported sharding,
atomics, barriers, asynchronous copies, scratch-memory operations, block-scaled dot and backend extensions are
rejected. CUDA retains checked-dimension assertions; AMD assertion hostcalls are rejected because their runtime
service has not been qualified. Batched indexed references use the existing portable transform. Custom derivative,
rematerialization, tuning and distributed composition remain subject to the same canonical verification and target
admission; selecting Triton does not add a new transform policy or silently select another backend.

## Compiler installation and selection

Build the optional `//:triton-compiler` target following the `ryft-xla-sys` README. This target is excluded from the
default native archive. Supply its absolute path to `Compiler::new`; construction itself does not load a GPU runtime.
Native version probes and compilation run in bounded, cancellable child processes with an explicit environment,
finite artifact/diagnostic limits and a default 300-second deadline. Missing tools report compiler unavailability;
unsupported portable semantics report admission errors; native failures retain bounded stdout and stderr.

```rust,ignore
use std::path::PathBuf;

use ryft_core::kernels::{KernelSchedule, VerifiedKernel};
use ryft_triton::kernels::{Compiler, Options, Target};

let compiler = Compiler::new(PathBuf::from("/opt/ryft/bin/triton-compiler"))?;
let target = Target::Cuda { major: 12, minor: 1 };
let options = Options::default();
let schedule = KernelSchedule::default();
let verified = VerifiedKernel::new(&definition, 1024)?;
let compiled = verified.compile(&compiler, &target, &options, &schedule)?;
```

For XLA, enable the `triton` feature and pass these choices with `TritonEmbedding` to
`XlaKernelCompilerBinding::new`, then select the binding through `XlaOptions::with_kernel_compiler`. The root `ryft`
crate forwards this feature. Compiler and target selection are explicit. `ryft_core::kernels` has no Triton or AMD
operation family, and the macro-authored portable definition is unchanged.

## Persistence and deployment

Compiler identity includes the executable SHA-256, validated source/toolchain versions, target architecture, warp
count, pipeline stages and scratch budget. Timeouts and output-capture limits do not change emitted-code identity.
Concrete artifacts validate entry symbols, exact pointer arguments, target formats and resources before embedding.
XLA adds its runtime/device facts and embedding ABI to the binding and AOT compatibility checks.

Save the bytes returned by `KernelCompiler::configuration_key` alongside the `KernelAotBundle`. At deployment,
`Compiler::from_configuration` validates those recorded bytes without opening or probing any compiler executable.
Use it to reconstruct the same target/options/schedule binding for `KernelAotBundle::load`. It cannot compile new
kernels. The deployment still requires the matching PJRT plugin and CUDA or HIP runtime. Persisted artifacts contain
native executable code; their digests detect corruption and are not a producer-authentication mechanism.

## Verification and examples

`crates/experimental/src/kernels.rs` contains shared macro-authored vector addition, sum and matrix multiplication
examples, with independent numerical oracles. The Triton tests use those same definitions, including partial tiles,
IEEE input precision and batching, and check fresh-session executable restoration and compiler-free AOT loading.
On a qualified CUDA 13 machine, set `RYFT_TRITON_COMPILER` to the native executable and run:

```sh
RYFT_PJRT_RUN_TRITON_KERNELS=1 cargo test -p ryft-experimental --lib \
  --features triton,cuda-13 kernels::triton:: -- --nocapture --test-threads=1
```

`cargo test -p ryft-triton --lib` checks typed lowering, admission, compiler identity, process limits and malformed
products without a GPU. `cargo test -p ryft-rocm --lib` checks actual HSACO metadata and injected native ownership
failures. These compiler and ownership checks do not establish AMD hardware execution correctness.

The production Rust-to-HSACO path can also be checked on Linux without an AMD device:

```sh
RYFT_RUN_TRITON_COMPILER_TESTS=1 cargo test -p ryft-experimental --lib \
  --features triton kernels::triton::test_compiler_on_rocm -- --nocapture
```

This requires the same `RYFT_TRITON_COMPILER` path. It compiles the shared vector definition for all three AMD targets
and a whole-array IEEE dot for `gfx942`, then validates the concrete artifact through `ryft-rocm`. The separate
`kernels::tuning::test_triton_on_cuda` test uses the existing tuner with explicit one- and two-stage schedules and
completed numerical samples; timings include host readback and do not imply cross-backend performance rankings.
