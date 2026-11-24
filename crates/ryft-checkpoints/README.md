# **Ryft Checkpoints:** Modular and Customizable Checkpointing Support for Ryft

`ryft-checkpoints` saves, restores, and imports the parameter structures of [`ryft-core`](../ryft-core). It manages
sequences of atomically saved step checkpoints with the architecture of [Orbax](https://orbax.readthedocs.io/)
(i.e., save decision and preservation policies, asynchronous and multi-process saving, and partial restore), reads
and writes the safetensors file format (behind the default `safetensors` feature), pins released checkpoints with
manifests, decodes packed low precision weights, and defines how model code maps a foreign checkpoint to its
parameters.

By default, this crate depends on `ryft-core` only among the Ryft crates. It works with host arrays and has no XLA,
PJRT, or accelerator dependency. Checkpoints live on the local file system or, through the `FileSystem` abstraction,
in other storage. These features change what it includes:

- `safetensors` (enabled by default) stores the arrays of checkpoints in the safetensors file format. Without it, the
  storage format is whatever implementation of `ArrayFormat` the caller provides.
- `xla` adds a dependency on [`ryft-xla`](../ryft-xla), a leaf handler that restores checkpoints directly into the
  buffers of XLA devices, with a sharding per leaf, and reads them back for saving, and a process barrier over the
  coordination service of a distributed XLA runtime. The platform features of `ryft-xla` (i.e., `cuda-12`, `cuda-13`,
  `rocm-10`, `tpu`, `neuron`, `metal`, and `mps`) are forwarded under the same names and enable `xla`.
- `object-store` (disabled by default) adds a dependency on [`object_store`](https://docs.rs/object_store) and a file
  system that stores checkpoints in S3, GCS, Azure Blob Storage, or any other object store, finalizing them with commit
  files because object stores cannot rename atomically. Cloud backends are enabled through the `aws`, `gcp`, and `azure`
  features of `object_store` in the depending crate.
- `zarr` (disabled by default) stores the arrays of checkpoints as uncompressed [Zarr](https://zarr.dev/) version 3
  arrays, one group per item and one array per leaf, which is the layout that TensorStore writes for Orbax without
  [OCDBT](https://google.github.io/tensorstore/kvstore/ocdbt/index.html).

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
