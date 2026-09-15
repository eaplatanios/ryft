# Ryft ROCm

`ryft-rocm` owns concrete HSACO artifacts and the HIP launcher used by Ryft's direct Triton adapter. It does not
compile kernels, depend on XLA, allocate outer program buffers, or create streams. The embedding runtime supplies a
primary-context stream and pointer arguments and retains them through its existing asynchronous completion fence.

The driver contract is HIP **7.13** on 64-bit Linux. Runtime major/minor are checked before versioned device properties
or module APIs are used. Missing libraries, incompatible versions, graph capture, unsupported targets, and invalid
launch geometry produce explicit errors. AMD hardware execution remains unqualified; deterministic native-call
injection tests establish ownership and validation behavior, not hardware correctness.

The launcher retains modules in a bounded cache across all device contexts. Eviction synchronizes the owning context
before unloading a module and releasing its primary-context retain. Explicit shutdown follows the same ordering and
is retryable after a native cleanup failure. Destructor failures are exposed through `Error::take_cleanup_errors`;
resources that cannot safely be released remain retained rather than invalidating pending work.

No stream in a context being evicted or shut down may participate in graph capture, and no graph may refer to its
cached modules. The caller owns memory bounds, permissions, alias legality, stream ordering and completion. These
obligations form the safety contract of the borrowed-handle constructors and launch/shutdown functions.

## Native source contract

Headers were read from AMD's configured
[`therock-dist-linux-gfx908-7.13.0.tar.gz`](https://repo.amd.com/rocm/tarball/therock-dist-linux-gfx908-7.13.0.tar.gz)
archive. The inspected headers identify HIP `7.13.99004`, source revision `3309c6114a`. Only the archive prefix needed
for the headers was downloaded; its full distribution checksum was not verified. The exact header hashes are:

- `hip_runtime_api.h`: `2b10f6e53712d7d6fdc09ad74f1b8eb7feb7bbb0a71598d60ee408d19446df07`
- `hip_version.h`: `1b328b56af5a6779c8d0c1bdbd5deaa060e81aea574d28a25630bbdead9348e4`

The effective properties symbol is `hipGetDevicePropertiesR0600`. Its structure size, alignment and architecture-field
layout were checked by a native Linux/aarch64 C probe and Linux/x86_64 compile-time assertions. No HIP runtime was
loaded by these layout checks.

## Artifact admission and fixtures

Artifacts must be little-endian ELF64 AMD HSA code objects v5, with one global entry function and its 64-byte `.kd`
descriptor. The bounded reader checks ELF table ranges, MessagePack metadata, pointer argument offsets and sizes,
wavefront and workgroup requirements, and descriptor resource fields and relative entry address. Hidden arguments,
dynamic stack, scalar arguments and unsupported metadata layouts are rejected before module loading. The current
base architecture set is `gfx908`, `gfx90a`, and `gfx942`; explicit XNACK and SRAM ECC requirements must match
the runtime device. Architecture admission is a compiler contract, not a claim of device execution qualification.

The `src/fixtures` vector and IEEE matrix multiplication HSACO files were generated for `gfx942` by the optional
`ryft-xla-sys` compiler worker with XLA `eb6b90ed013f511eca088c52f541f3c0819f919e`, JAX
`a7606f995e1a92707cbeb257e487fa53e7abe84b`, and Triton `a77e7c793abc0d0c923a9afb275058e2fe57a198`.
Both have three global pointers, a 256-thread workgroup and zero static LDS. Matrix multiplication additionally
requires 4096 bytes of dynamic LDS, supplied by the compiler's launch metadata. The native target spelling
`amdgcn-unknown-amdhsa-amdgiz-gfx942` comes from the pinned XLA AMDGPU target triple.

Run deterministic ownership and artifact validation with `cargo test -p ryft-rocm --lib`. These tests require no HIP
runtime or AMD device; the fixture reader uses actual compiler products and native ownership tests use injected
function pointers whose handles never refer to device memory.
