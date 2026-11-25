# **Ryft Experiments:** Work-in-Progress End-to-End Experiments

> [!WARNING]
> `ryft` is currently a work in progress and is evolving very actively. APIs and module or crate boundaries may change.

`ryft-experiments` puts the pieces of Ryft together into runnable experiments. It combines the models of
[`ryft-models`](../ryft-models) and the optimizers of [`ryft-optimizers`](../ryft-optimizers) with the XLA backend of
[`ryft-xla`](../ryft-xla) to compile and run training steps on the PJRT plugin of the selected platform, and saves
and loads training state with [`ryft-checkpoints`](../ryft-checkpoints).

These features change what it includes:

- `safetensors` (enabled by default) stores the arrays of checkpoints in the safetensors file format and builds the
  `inkling2_checkpoint` binary.
- `zarr` (disabled by default) stores the arrays of checkpoints as Zarr version 3 arrays.
- `cuda-12`, `cuda-13`, `rocm-10`, `tpu`, `neuron`, `metal`, and `mps` select the platform of the XLA backend and are
  forwarded to `ryft-xla`.

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
