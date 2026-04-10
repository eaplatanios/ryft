# **Ryft XLA:** XLA Backend for Ryft

> [!WARNING]
> `ryft` is currently a work in progress and is evolving very actively. APIs and module or crate boundaries may change.

This crate provides the XLA-specific backend functionality for Ryft. It builds on [`ryft-core`](../ryft-core) for core
abstractions, [`ryft-mlir`](../ryft-mlir) for MLIR bindings, and [`ryft-pjrt`](../ryft-pjrt) for the PJRT runtime
interface. `ryft-xla` is where traced Ryft programs are lowered to [StableHLO](https://openxla.org/stablehlo),
partitioned via [Shardy](https://github.com/openxla/shardy), and executed through PJRT.

Note that this crate forwards the following feature flags directly to `ryft-pjrt`: `cuda-12`, `cuda-13`, `rocm-7`,
`tpu`, `neuron`, and `metal`. For information on what those flags represent and guidance on how to use them, refer to
[`crates/ryft-xla-sys/README.md`](../ryft-xla-sys/README.md).

## Coming Soon

`ryft-xla` is expected to grow into the main home for XLA-specific functionality.

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
