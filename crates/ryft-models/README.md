# **Ryft Models:** Work-in-Progress Model Definitions and Importers

> [!WARNING]
> `ryft` is currently a work in progress and is evolving very actively. APIs and module or crate boundaries may change.

`ryft-models` hosts model definitions written over the backend-independent array capabilities of
[`ryft-core`](../ryft-core), together with the components that surround a model, namely importers of released
checkpoints, tokenizers, and conversation renderers.

This crate depends on `ryft-core` and [`ryft-checkpoints`](../ryft-checkpoints) only among the Ryft crates. It has no
XLA, PJRT, or accelerator dependency, so its models run on any backend whose values implement the array capabilities
of `ryft-core`.

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
