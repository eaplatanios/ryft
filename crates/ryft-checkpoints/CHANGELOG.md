# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/).

<!-- next-header -->
## [Unreleased] - Release Date

### Added

- Initial release with safetensors reading and writing, release manifests, importers, and host codecs for packed
  NVFP4 and MXFP8 arrays, which moved here from `ryft-models`.
- Saving and loading shaped after the Orbax v1 API: the free functions `save`, `save_async`, `save_checkpointables`,
  `load`, `load_checkpointable`, `metadata`, `checkpointables_metadata`, and `delete`, which take an explicit
  `Context` (`contexts`) with `ParameterizedOptions` (partial loading, renaming, and data type conversion),
  `ArrayOptions` (the array format, chosen at runtime as an `Arc<dyn ArrayFormat>`), `MultiprocessingOptions`, and
  the atomicity mode. A checkpoint is a directory of named checkpointables, and `save` writes the single
  checkpointable `state`. `save_async` returns an `AsyncResponse` (`synchronization`). Loading is strict unless
  partial loading is enabled, so a checkpoint that holds arrays that the abstract state does not name is rejected.
- Checkpointable handlers (`handlers`): the `CheckpointableHandler`, `CheckpointableSaver`, and `CheckpointableLoader`
  traits, `ParameterizedHandler` for `Parameterized` structures with pluggable `LeafHandler`s, `JsonHandler`, and
  `StatefulHandler` for `StatefulCheckpointable` objects, bound to their checkpointables by `Checkpointables`.
- A `training` module with `Checkpointer`, `CheckpointerOptions`, `CheckpointMetadata`, and `RootMetadata`, which
  keeps the step checkpoints of a training run, saves synchronously or asynchronously with at most one save in
  flight, and decides when to save and which checkpoints to keep with the policies of
  `training::save_decision_policies` (including preemption signals and the full continuous checkpointing policy) and
  `training::preservation_policies`.
- Step directory naming, atomic finalization, and deletion (`paths`), with `Atomicity::Rename` and
  `Atomicity::CommitFile` finalization modes, and step and parameterized metadata (`metadata`) with the JSON keys of
  Orbax.
- A `hugging_face` example that imports the arrays of a Hugging Face safetensors checkpoint and keeps step
  checkpoints of a training run that starts from them.
- Storage abstraction (`filesystems`) with `FileSystem`, `CheckpointPath`, and `LocalFileSystem`, threaded through
  every checkpoint file access, and an `object-store` feature with `ObjectStoreFileSystem`, which stores checkpoints in
  object stores.
- Multi-process saving with barriers and a primary process (`multiprocessing`).
- Metrics that are not finite are stored by name in the step metadata (i.e., `NaN`, `Infinity`, and `-Infinity`),
  and are worse than every other value for `BestN`.
- `LocalFileSystem` synchronizes every file that it writes to durable storage, so that the files of a checkpoint are
  on disk before the checkpoint is finalized. On Apple platforms it uses a plain `fsync` instead of `F_FULLFSYNC`, as
  TensorStore does.
- `zarr` feature with `ZarrFormat` and `ZarrCheckpoint`, which store the arrays of a checkpointable as uncompressed
  Zarr version 3 arrays in one group, optionally chunked along the leading dimension, and written on several threads
  (`ZarrFormat::with_write_concurrency`).
- `xla` feature with `XlaLeafHandler`, which loads the leaves of a checkpoint directly into the buffers of XLA
  devices, with a sharding per leaf, and reads them back for saving, and `XlaProcessBarrier`, which synchronizes the
  processes of a distributed XLA runtime that save one checkpoint together. The platform features of `ryft-xla` are
  forwarded under the same names.
