//! Checkpointable handlers, which save checkpointables into and load them from their directories of a checkpoint,
//! mirroring the `CheckpointableHandler` protocol of the Orbax v1 API.
//!
//! A checkpoint holds named checkpointables, each in the subdirectory named after it (refer to the
//! [`saving`](crate::saving) module). A handler owns the storage format of one kind of checkpointable and nothing
//! else. It writes a checkpointable into a directory that it is handed and reads it back from such a directory, while
//! the functions of [`saving`](crate::saving) and [`loading`](crate::loading) own everything around that (i.e.,
//! temporary directories, atomic finalization, and step metadata). The handler contract is split into three traits:
//!
//!   - [`CheckpointableHandler`] identifies a handler by name, which the step metadata records for every
//!     checkpointable, and gives it a hook that runs once the checkpoint directory has its final name.
//!   - [`CheckpointableSaver`] saves checkpointables of one type.
//!   - [`CheckpointableLoader`] loads checkpointables given an abstract checkpointable of one type.
//!
//! Orbax resolves the handler of every checkpointable at runtime from a registry, by the type of the checkpointable
//! or by its name, and has one `CheckpointableHandler` protocol with `save` and `load`. In Rust, handlers for
//! different checkpointable types are different implementations, so the checkpointable and the abstract
//! checkpointable are type parameters of the traits, the compiler selects the implementation, and the caller binds
//! every checkpointable to its handler (refer to [`Checkpointables`](crate::saving::Checkpointables)). A single handler
//! type may implement several savers and loaders (e.g., [`ParameterizedHandler`] saves every [`Parameterized`] type
//! and loads into every abstract structure).
//!
//! The handlers of this module mirror those of Orbax:
//!
//!   | Handler                  | Orbax counterpart               | Stored as                                 |
//!   |--------------------------|---------------------------------|-------------------------------------------|
//!   | [`ParameterizedHandler`] | `PyTreeHandler`                 | arrays of an [`ArrayFormat`] and metadata |
//!   | [`JsonHandler`]          | `JsonHandler`                   | one pretty printed JSON file              |
//!   | [`StatefulHandler`]      | `StatefulCheckpointableHandler` | whatever the checkpointable writes        |
//!
//! [`Parameterized`]: ryft_core::Parameterized
//! [`ArrayFormat`]: crate::formats::ArrayFormat

use crate::errors::CheckpointError;
use crate::filesystems::CheckpointPath;
use crate::multiprocessing::MultiprocessingOptions;

pub mod json;
pub mod parameterized;
pub mod stateful;

pub use json::JsonHandler;
pub use parameterized::{ArrayLeafHandler, LeafHandler, ParameterizedHandler};
pub use stateful::StatefulHandler;

// TODO(eaplatanios): Review this module.

/// Common contract of all checkpointable handlers, which identifies a handler and lets it act once a checkpoint was
/// finalized. Saving and loading are provided by [`CheckpointableSaver`] and [`CheckpointableLoader`].
pub trait CheckpointableHandler {
    /// Returns the name that identifies the kind of this handler in step metadata, which mirrors the `typestr` of
    /// Orbax handlers.
    fn name(&self) -> String;

    /// Called once after the checkpoint that this handler saved into was finalized, with the final directory of its
    /// checkpointable as `directory`. The default implementation does nothing.
    ///
    /// # Errors
    ///
    /// Returns a [`CheckpointError`] if finalization fails, in which case the checkpoint stays finalized.
    fn finalize(&self, directory: &CheckpointPath) -> Result<(), CheckpointError> {
        let _ = directory;
        Ok(())
    }
}

/// [`CheckpointableHandler`] that saves checkpointables of type `T`.
pub trait CheckpointableSaver<T: ?Sized>: CheckpointableHandler {
    /// Saves `checkpointable` into `directory`, which exists and is empty. The directory belongs to a checkpoint that
    /// is finalized once all of its checkpointables were saved, which may rename it, so implementations must not
    /// record its path.
    ///
    /// Every process that saves the checkpoint calls this function with the same directory and its own
    /// `multiprocessing` options. The handlers of this crate assume that every process holds the whole checkpointable,
    /// so they write only in the primary process (i.e., when [`MultiprocessingOptions::is_primary`] holds). Refer to
    /// the [`multiprocessing`](crate::multiprocessing) module for the phases of a save across processes.
    ///
    /// # Errors
    ///
    /// Returns a [`CheckpointError`] if `checkpointable` cannot be converted or written.
    fn save(
        &self,
        directory: &CheckpointPath,
        checkpointable: &T,
        multiprocessing: &MultiprocessingOptions,
    ) -> Result<(), CheckpointError>;
}

/// [`CheckpointableHandler`] that loads checkpointables given an abstract checkpointable of type `A`, which describes
/// what to load (e.g., the structure and types of a parameter structure, or nothing for self describing
/// checkpointables).
pub trait CheckpointableLoader<A>: CheckpointableHandler {
    /// Type of the loaded checkpointable.
    type Output;

    /// Loads a checkpointable from `directory`, its directory in a finalized checkpoint, as described by
    /// `abstract_checkpointable`.
    ///
    /// # Errors
    ///
    /// Returns a [`CheckpointError`] if the stored checkpointable cannot be read or does not match
    /// `abstract_checkpointable`.
    fn load(&self, directory: &CheckpointPath, abstract_checkpointable: A) -> Result<Self::Output, CheckpointError>;
}

/// Object that saves itself into a directory and loads itself back from one, which mirrors the
/// `StatefulCheckpointable` protocol of Orbax. A [`StatefulHandler`] saves and loads such objects, so that types that
/// own their storage format (e.g., the state of a data loader) need no handler of their own.
pub trait StatefulCheckpointable {
    /// Saves this object into `directory`, which exists and is empty. Like [`CheckpointableSaver::save`], this must
    /// not record the path of `directory`.
    ///
    /// # Errors
    ///
    /// Returns a [`CheckpointError`] if this object cannot be written.
    fn save(&self, directory: &CheckpointPath) -> Result<(), CheckpointError>;

    /// Replaces the state of this object with the one that [`Self::save`] saved into `directory`.
    ///
    /// # Errors
    ///
    /// Returns a [`CheckpointError`] if the stored state cannot be read.
    fn load(&mut self, directory: &CheckpointPath) -> Result<(), CheckpointError>;
}
