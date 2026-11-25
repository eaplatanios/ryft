//! Checkpoints for Ryft parameter structures.
//!
//! By default the crate depends on `ryft-core` only among the Ryft crates and works with host
//! [`Array`](ryft_core::Array)s, so it is independent of the backend that computes with the checkpointed values. Its
//! saving and loading follow the v1 API of [Orbax](https://orbax.readthedocs.io/en/latest/), the checkpointing library
//! of JAX, with [`Parameterized`](ryft_core::Parameterized) structures in place of the `PyTree`s of Orbax.
//!
//!   - [`contexts`] configures saving and loading ([`Context`], [`ParameterizedOptions`], [`ArrayOptions`]).
//!   - [`saving`] saves checkpoints atomically, synchronously or on a background thread ([`save()`], [`save_async`],
//!     [`save_checkpointables`], [`SaveOptions`], [`Checkpointables`]).
//!   - [`loading`] loads checkpoints and reads their metadata ([`load()`], [`load_checkpointable`], [`metadata()`],
//!     [`checkpointables_metadata`]).
//!   - [`synchronization`] waits for operations that run on background threads ([`AsyncResponse`]).
//!   - [`handlers`] saves and loads the checkpointables of a checkpoint ([`CheckpointableHandler`],
//!     [`CheckpointableSaver`], [`CheckpointableLoader`], [`StatefulCheckpointable`], [`ParameterizedHandler`],
//!     [`LeafHandler`], [`ArrayLeafHandler`], [`JsonHandler`], [`StatefulHandler`]).
//!   - [`training`] keeps the step checkpoints of a training run in one root directory, deciding when to save and
//!     which checkpoints to keep ([`training::Checkpointer`], [`training::CheckpointerOptions`],
//!     [`training::CheckpointMetadata`], [`training::RootMetadata`], and the policies of
//!     [`training::save_decision_policies`] and [`training::preservation_policies`]).
//!   - [`metadata`](mod@metadata) describes checkpoints and saved parameter structures ([`CheckpointMetadata`],
//!     [`StepMetadata`], [`ParameterizedMetadata`], [`LeafMetadata`]).
//!   - [`paths`] names step directories, finalizes checkpoint directories atomically, and deletes checkpoints
//!     ([`StepNameFormat`], [`Atomicity`], [`delete`]).
//!   - [`arrays`] describes named arrays on durable storage and reads them ([`Checkpoint`], [`ArrayDescriptor`],
//!     [`ArraySpecification`]).
//!   - [`filesystems`] abstracts the storage that checkpoints live in ([`FileSystem`], [`CheckpointPath`],
//!     [`LocalFileSystem`], and, with the `object-store` feature, `ObjectStoreFileSystem`).
//!   - [`formats`] abstracts the storage format of the arrays of one checkpointable ([`ArrayFormat`]).
//!   - `safetensors` reads and writes the safetensors file format (`SafetensorsCheckpoint`, `SafetensorsFormat`,
//!     `write_safetensors`, `stream_safetensors`). It requires the `safetensors` feature, which is enabled by
//!     default.
//!   - `zarr` stores the arrays of a checkpointable as a Zarr version 3 group (`ZarrFormat`, `ZarrCheckpoint`). It
//!     requires the `zarr` feature.
//!   - `xla` loads leaves directly into the buffers of XLA devices and reads them back for saving (`XlaLeafHandler`),
//!     and synchronizes the processes of a distributed XLA runtime that save one checkpoint together
//!     (`XlaProcessBarrier`). It requires the `xla` feature, which adds a dependency on `ryft-xla`. The platform
//!     features of `ryft-xla` (e.g., `cuda-13` and `mps`) are forwarded under the same names and enable `xla`.
//!   - [`multiprocessing`] coordinates the processes that save one checkpoint together ([`MultiprocessingOptions`],
//!     [`ProcessBarrier`]).
//!   - [`codecs`] decodes and encodes the packed block-scaled formats that released checkpoints store (i.e., NVFP4
//!     and MXFP8).
//!   - [`manifests`] pins the files and arrays of released checkpoints ([`Manifest`]).
//!   - [`importers`] defines how model code maps a checkpoint that another system wrote to its parameters
//!     ([`Importer`], [`import_arrays`]).
//!
//! # Saving and Loading
//!
//! [`save()`] saves a parameter structure of host arrays as one checkpoint, and [`load()`] loads it into the structure
//! that an abstract structure of [`ArrayType`](ryft_core::ArrayType)s describes. A [`training::Checkpointer`] saves
//! one such checkpoint per step and keeps the checkpoints that its preservation policy selects:
//!
//! ```rust
//! # #[cfg(feature = "safetensors")]
//! # fn main() -> Result<(), ryft_checkpoints::CheckpointError> {
//! use ryft_checkpoints::training::{Checkpointer, CheckpointerOptions, LatestN};
//! use ryft_checkpoints::{Context, SaveOptions, load, save};
//! use ryft_core::{Array, ArrayType, DataType};
//!
//! let directory = tempfile::tempdir().unwrap();
//! let context = Context::default();
//! let state = (Array::vector(vec![1.0f32, 2.0])?, vec![Array::scalar(3u8)?]);
//! let abstract_state = (ArrayType::new_static(DataType::F32, [2]), vec![ArrayType::scalar(DataType::U8)]);
//! save(directory.path().join("checkpoint"), &state, &SaveOptions::new(), &context)?;
//! assert_eq!(load(directory.path().join("checkpoint"), abstract_state.clone(), &context)?, state);
//!
//! let options = CheckpointerOptions::new().with_preservation_policy(LatestN::new(2));
//! let mut checkpointer = Checkpointer::new(directory.path().join("run"), context, options)?;
//! for step in 1..=3 {
//!     checkpointer.save(step, &state, &SaveOptions::new())?;
//! }
//! assert_eq!(checkpointer.checkpoints().iter().map(|checkpoint| checkpoint.step()).collect::<Vec<_>>(), [2, 3]);
//! assert_eq!(checkpointer.load(None, abstract_state)?, state);
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "safetensors"))]
//! # fn main() {}
//! ```
//!
//! A checkpoint is a directory of named checkpointables. [`save()`] writes the single checkpointable `state`, and
//! [`save_checkpointables`] writes any number of them, each with the [`CheckpointableSaver`] that [`Checkpointables`]
//! binds it to (e.g., a [`JsonHandler`] for a training configuration next to the parameters).
//!
//! # Storage and Atomicity
//!
//! Every function that takes a checkpoint location accepts a [`CheckpointPath`], which pairs a path with the
//! [`FileSystem`] that it belongs to, the counterpart of the `etils.epath.Path` of Orbax. Local paths (e.g., [`Path`]s,
//! [`PathBuf`]s, and strings) convert into checkpoint paths of the [`LocalFileSystem`]. With the `object-store`
//! feature, `ObjectStoreFileSystem` stores checkpoints in S3, GCS, Azure Blob Storage, or any other store of the
//! [`object_store`](https://docs.rs/object_store) crate.
//!
//! A checkpoint is readable only once it is complete. On file systems that rename directories atomically, it is
//! written into a temporary directory that is renamed to its final name ([`Atomicity::Rename`]). On object stores, it
//! is written in place and completed by a commit file ([`Atomicity::CommitFile`]). Checkpoints of both modes are
//! always readable, and the mode of saving can be overridden with [`Context::with_atomicity`].
//!
//! The [`LocalFileSystem`] synchronizes every file that it writes to durable storage, as the file driver of
//! TensorStore does for the arrays of Orbax by default, so the files of a checkpoint are on disk before the checkpoint
//! is finalized. Formats that write many small files pay for this once per file (e.g., `ZarrFormat` writes two files
//! per array, which it does on several threads to share the cost), while `SafetensorsFormat` writes one file per
//! checkpointable.
//!
//! # Differences From Orbax
//!
//! The following parts of the Orbax v1 API are not mirrored:
//!
//!   - A context is never activated through ambient state, and there are no handler registries. Every function takes
//!     its [`Context`] explicitly, and handlers are chosen statically.
//!   - `load_checkpointables` does not return a dictionary. [`load_checkpointable`] loads one named checkpointable
//!     with the loader and the output type that the caller selects.
//!   - There is no `save_checkpointables_async`, `load_async`, `delete_async`, or cancellation of saves in flight.
//!   - Partial saving (i.e., the `partial` module of Orbax) and `PLACEHOLDER` leaves are not supported.
//!   - There is no `CheckpointLayout`. Foreign safetensors files are read through `SafetensorsCheckpoint` and the
//!     [`importers`] instead.
//!   - The option groups `AsyncOptions`, `MemoryOptions`, `FileOptions`, `PathwaysOptions`, and `DeletionOptions`
//!     and the protobuf handler have no counterparts.
//!   - Checkpoints have no `orbax.checkpoint` indicator file, because there is no older Ryft layout to tell apart, and
//!     metrics are stored in the step metadata rather than as a checkpointable of their own.
//!   - The `EveryNStepsClosest`, `CustomSteps`, and `LatestDuration` preservation policies and a [`training::BestN`]
//!     over a metric function are not provided.
//!
//! Earlier limits also remain. Loading cannot pad or truncate arrays whose shapes differ from the abstract state.
//! Leaves that are split across processes (i.e., where each process holds only some shards) are not supported, the
//! save decision of the primary process is not broadcast to the other processes, and a multi-process save that failed
//! cannot be retried (refer to [`multiprocessing`]). For storage, the `zarr` feature writes the Zarr version 3 layout
//! that TensorStore writes for Orbax without OCDBT, but OCDBT itself, compressing Zarr codecs (e.g., `zstd`), and
//! sharded Zarr arrays (i.e., the `sharding_indexed` codec) are not supported.
//!
//! [`Path`]: std::path::Path
//! [`PathBuf`]: std::path::PathBuf

pub mod arrays;
pub mod codecs;
pub mod contexts;
pub mod errors;
pub mod filesystems;
pub mod formats;
pub mod handlers;
pub mod importers;
pub mod loading;
pub mod manifests;
pub mod metadata;
pub mod multiprocessing;
pub mod paths;
pub mod saving;
pub mod synchronization;
pub mod training;

#[cfg(feature = "safetensors")]
pub mod safetensors;

#[cfg(feature = "xla")]
pub mod xla;

#[cfg(feature = "zarr")]
pub mod zarr;

pub use arrays::{ArrayDescriptor, ArraySpecification, Checkpoint};
pub use codecs::{CodecError, Nvfp4OuterScale, Nvfp4ScaleSelection};
pub use contexts::{ArrayOptions, Context, ParameterizedOptions};
pub use errors::CheckpointError;
#[cfg(feature = "object-store")]
pub use filesystems::ObjectStoreFileSystem;
pub use filesystems::{CheckpointPath, FileReader, FileSystem, FileWriter, LocalFileSystem};
pub use formats::ArrayFormat;
pub use handlers::{
    ArrayLeafHandler, CheckpointableHandler, CheckpointableLoader, CheckpointableSaver, JsonHandler, LeafHandler,
    ParameterizedHandler, StatefulCheckpointable, StatefulHandler,
};
pub use importers::{Importer, import_arrays};
pub use loading::{checkpointables_metadata, load, load_checkpointable, metadata};
pub use manifests::{ArrayRecord, FileRecord, Manifest};
pub use metadata::{CheckpointMetadata, LeafMetadata, ParameterizedMetadata, StepMetadata};
pub use multiprocessing::{LocalProcessBarrier, MultiprocessingOptions, ProcessBarrier};
pub use paths::{Atomicity, StepNameFormat, delete};
#[cfg(feature = "safetensors")]
pub use safetensors::{SafetensorsCheckpoint, SafetensorsFormat, stream_safetensors, write_safetensors};
pub use saving::{Checkpointables, STATE_CHECKPOINTABLE_NAME, SaveOptions, save, save_async, save_checkpointables};
pub use synchronization::AsyncResponse;
#[cfg(feature = "xla")]
pub use xla::{XlaLeafHandler, XlaProcessBarrier};
#[cfg(feature = "zarr")]
pub use zarr::{ZarrCheckpoint, ZarrFormat};

// TODO(eaplatanios): Review this module.
