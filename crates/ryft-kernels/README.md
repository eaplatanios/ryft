# **Ryft Kernels:** Work-in-Progress Portable Kernel APIs

> [!WARNING]
> `ryft` is currently a work in progress and is evolving very actively. APIs and module or crate boundaries may change.

`ryft-kernels` hosts Ryft's backend-independent kernel subsystem: the `#[kernel]` authoring macro, the kernel
operation family, kernel definitions and verification, host interpretation for debugging, serialization, transforms,
and the `KernelCompiler` interface that native compiler adapters implement. Kernel bodies are ordinary
[`ryft-core`](../ryft-core) programs built from canonical array, dimension, reference, and control-flow operations.

This crate has no native compiler, MLIR, PJRT, CUDA, ROCm, Mosaic, Triton, or XLA dependency. Native compilation and
execution belong to the adapter crates and the XLA integration. The `ryft` crate re-exports this crate as
`ryft::kernels`.

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
