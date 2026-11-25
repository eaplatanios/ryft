//! Saving of checkpoints, mirroring the `save`, `save_async`, and `save_checkpointables` functions of the Orbax v1
//! API.
//!
//! A checkpoint is a directory that holds named checkpointables, each in the subdirectory named after it and written
//! by the [`CheckpointableSaver`] that it is bound to in [`Checkpointables`], next to the [`StepMetadata`] of the whole
//! checkpoint. [`save`] saves a parameter structure as the single checkpointable [`STATE_CHECKPOINTABLE_NAME`] with
//! the [`ParameterizedHandler`](crate::handlers::ParameterizedHandler) of its [`Context`], and
//! [`save_checkpointables`] saves any number of checkpointables. Both save atomically, so that a checkpoint directory
//! is only ever read once it holds a complete checkpoint, even when the process dies while saving. How that is
//! achieved depends on the [`Atomicity`] mode of the context, which defaults to the mode that the file system of the
//! checkpoint supports (refer to the [`paths`](crate::paths#atomicity) module for both modes).
//!
//! ```text
//!   save_checkpointables(path, checkpointables, options, context)
//!     │
//!     ├─▶ remove `path` when `overwrite`, or fail when it exists
//!     ├─▶ create the working directory and one subdirectory per checkpointable
//!     │     (`<path>.ryft-checkpoint-tmp` for `Rename`, `path` itself for `CommitFile`)
//!     ├─▶ `CheckpointableSaver::save` into every subdirectory ┄┄▶ remove the working directory and return the error
//!     ├─▶ `StepMetadata::write`, with the commit timestamp for `Rename` only
//!     ├─▶ rename the working directory to `path` (`Rename`) or write `commit_success.txt` (`CommitFile`)
//!     └─▶ `CheckpointableHandler::finalize` of every checkpointable
//! ```
//!
//! [`save_async`] runs the same steps on a background thread and returns an [`AsyncResponse`]. Every call is
//! independent, so callers that save to the same path must wait for one save before they start the next. A
//! [`training::Checkpointer`](crate::training::Checkpointer) keeps at most one save in flight instead.

use std::collections::{BTreeMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::path::is_separator;
use std::time::{SystemTime, UNIX_EPOCH};

use ryft_core::Parameterized;

use crate::contexts::Context;
use crate::errors::CheckpointError;
use crate::filesystems::CheckpointPath;
use crate::handlers::{CheckpointableSaver, LeafHandler};
use crate::metadata::{STEP_METADATA_FILE_NAME, StepMetadata};
use crate::multiprocessing::MultiprocessingOptions;
use crate::paths::{Atomicity, COMMIT_SUCCESS_FILE, temporary_path};
use crate::synchronization::AsyncResponse;

// TODO(eaplatanios): Review this module.

/// Name of the checkpointable that [`save`] saves and [`load`](crate::loading::load) loads, which is the default
/// `checkpointable_name` of Orbax.
pub const STATE_CHECKPOINTABLE_NAME: &str = "state";

/// Options of one save, which mirror the keyword arguments of the saving functions of Orbax.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SaveOptions {
    /// Whether an existing checkpoint at the destination is replaced.
    overwrite: bool,

    /// Whether a [`training::Checkpointer`](crate::training::Checkpointer) saves regardless of its save decision
    /// policy.
    force: bool,

    /// Metrics to record in the [`StepMetadata`] of the checkpoint, if any.
    metrics: Option<BTreeMap<String, f64>>,

    /// Free-form custom metadata to record in the [`StepMetadata`] of the checkpoint.
    custom_metadata: serde_json::Value,
}

impl SaveOptions {
    /// Creates new [`SaveOptions`] that neither overwrite nor force and that record neither metrics nor custom
    /// metadata.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns these options with the provided `overwrite` flag, which replaces an existing checkpoint at the
    /// destination when it is set.
    #[inline]
    pub fn with_overwrite(mut self, overwrite: bool) -> Self {
        self.overwrite = overwrite;
        self
    }

    /// Returns these options with the provided `force` flag, which makes a
    /// [`training::Checkpointer`](crate::training::Checkpointer) save regardless of its save decision policy when it
    /// is set. The saving functions of this module ignore it.
    #[inline]
    pub fn with_force(mut self, force: bool) -> Self {
        self.force = force;
        self
    }

    /// Returns these options with the provided metrics.
    #[inline]
    pub fn with_metrics(mut self, metrics: BTreeMap<String, f64>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Returns these options with the provided free-form custom metadata.
    #[inline]
    pub fn with_custom_metadata(mut self, custom_metadata: serde_json::Value) -> Self {
        self.custom_metadata = custom_metadata;
        self
    }

    /// Returns whether an existing checkpoint at the destination is replaced.
    #[inline]
    pub fn overwrite(&self) -> bool {
        self.overwrite
    }

    /// Returns whether a [`training::Checkpointer`](crate::training::Checkpointer) saves regardless of its save
    /// decision policy.
    #[inline]
    pub fn force(&self) -> bool {
        self.force
    }

    /// Returns the metrics to record, if any.
    #[inline]
    pub fn metrics(&self) -> Option<&BTreeMap<String, f64>> {
        self.metrics.as_ref()
    }

    /// Returns the free-form custom metadata to record.
    #[inline]
    pub fn custom_metadata(&self) -> &serde_json::Value {
        &self.custom_metadata
    }
}

/// Named checkpointables of one save, each bound to the [`CheckpointableSaver`] that saves it, which is the statically
/// typed counterpart of the dictionary of checkpointables and the handler registry of Orbax.
///
/// Names are validated when the checkpointables are saved. A name must be a single non-empty path component other
/// than `.` and `..`, must not be a reserved file name of checkpoints (i.e., [`STEP_METADATA_FILE_NAME`] and
/// [`COMMIT_SUCCESS_FILE`]), and must be unique within one save.
#[derive(Default)]
pub struct Checkpointables<'o> {
    /// Name of every checkpointable together with the checkpointable bound to its saver, in insertion order.
    entries: Vec<(String, Box<dyn 'o + BoundCheckpointable>)>,
}

impl<'o> Checkpointables<'o> {
    /// Creates new empty [`Checkpointables`].
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns these checkpointables with `checkpointable` saved by `handler` under the name `name`.
    #[inline]
    pub fn with<N: Into<String>, H: CheckpointableSaver<T>, T: ?Sized>(
        mut self,
        name: N,
        handler: &'o H,
        checkpointable: &'o T,
    ) -> Self {
        self.entries.push((name.into(), Box::new(HandledCheckpointable { handler, checkpointable })));
        self
    }

    /// Returns the names of the checkpointables, in insertion order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(name, _)| name.as_str())
    }

    /// Returns the name of the handler of every checkpointable, keyed by the name of the checkpointable, which is what
    /// the [`StepMetadata`] of the checkpoint records.
    fn handlers(&self) -> BTreeMap<String, String> {
        self.entries
            .iter()
            .map(|(name, checkpointable)| (name.clone(), checkpointable.handler_name()))
            .collect()
    }

    /// Validates the names of these checkpointables, as the [type documentation](Self) describes.
    fn validate(&self) -> Result<(), CheckpointError> {
        let mut names = HashSet::with_capacity(self.entries.len());
        for (name, _) in &self.entries {
            validate_checkpointable_name(name)?;
            if !names.insert(name) {
                return Err(CheckpointError::InvalidArgument {
                    message: format!("checkpointable `{name}` is saved more than once"),
                });
            }
        }
        Ok(())
    }
}

impl Debug for Checkpointables<'_> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.debug_map().entries(self.handlers()).finish()
    }
}

/// Saves the parameter structure `state` atomically as the checkpoint at `path`, which mirrors `save` of Orbax. The
/// structure is the single checkpointable [`STATE_CHECKPOINTABLE_NAME`], which the
/// [`Context::parameterized_handler`] of `context` saves. Refer to [`save_checkpointables`] for the rules of saving.
///
/// # Errors
///
/// Returns the errors of [`save_checkpointables`].
pub fn save<P: Into<CheckpointPath>, L: LeafHandler, S: Parameterized<L::Leaf>>(
    path: P,
    state: &S,
    options: &SaveOptions,
    context: &Context<L>,
) -> Result<(), CheckpointError> {
    let handler = context.parameterized_handler();
    save_checkpointables(
        path,
        &Checkpointables::new().with(STATE_CHECKPOINTABLE_NAME, &handler, state),
        options,
        context,
    )
}

/// Starts saving the parameter structure `state` as the checkpoint at `path` with [`save`] on a background thread,
/// which mirrors `save_async` of Orbax, and returns the [`AsyncResponse`] of the save. The state and a clone of
/// `context` move into the background thread.
///
/// # Errors
///
/// Returns [`CheckpointError::Io`] if the background thread cannot be started. Errors of the save itself are returned
/// by [`AsyncResponse::result`].
pub fn save_async<P: Into<CheckpointPath>, L, S>(
    path: P,
    state: S,
    options: SaveOptions,
    context: &Context<L>,
) -> Result<AsyncResponse<()>, CheckpointError>
where
    L: 'static + Clone + Send + Sync + LeafHandler,
    S: 'static + Send + Parameterized<L::Leaf>,
{
    let path = path.into();
    let context = context.clone();
    AsyncResponse::spawn("ryft-checkpoint-save", &path.to_string(), move || save(path, &state, &options, &context))
}

/// Saves `checkpointables` atomically as the checkpoint at `path`, which mirrors `save_checkpointables` of Orbax.
/// Refer to the [module documentation](self) for the steps of a save.
///
/// The destination must not exist unless `options` ask to overwrite it, in which case the existing directory is
/// removed first. In [`Atomicity::Rename`] mode, a stale temporary directory of an interrupted save to the same path is
/// replaced. The working directory of a failed save is removed before the error is returned. With several processes,
/// the primary process prepares and finalizes the directory, every process saves its part, and the processes
/// synchronize between these phases (refer to the [`multiprocessing`](crate::multiprocessing) module).
///
/// # Errors
///
/// Returns [`CheckpointError::InvalidArgument`] if a name of `checkpointables` is not valid (refer to
/// [`Checkpointables`]), [`CheckpointError::ExistingCheckpoint`] if `path` exists and is not overwritten, the errors
/// of the handlers, [`CheckpointError::Io`] if a file or directory cannot be written, renamed, or removed, and
/// [`CheckpointError::Barrier`] if the processes fail to synchronize.
pub fn save_checkpointables<P: Into<CheckpointPath>, L: LeafHandler>(
    path: P,
    checkpointables: &Checkpointables<'_>,
    options: &SaveOptions,
    context: &Context<L>,
) -> Result<(), CheckpointError> {
    checkpointables.validate()?;
    let directory = path.into();
    let multiprocessing = context.multiprocessing_options();
    let primary = multiprocessing.is_primary();
    let atomicity = context.atomicity().unwrap_or_else(|| Atomicity::for_filesystem(directory.filesystem().as_ref()));
    let working = match atomicity {
        Atomicity::Rename => temporary_path(&directory),
        Atomicity::CommitFile => directory.clone(),
    };
    multiprocessing.synchronize(&format!("save_start:{directory}"))?;
    if primary {
        if directory.exists()? {
            if !options.overwrite() {
                return Err(CheckpointError::ExistingCheckpoint { path: directory.to_string() });
            }
            directory.remove()?;
        }
        if atomicity == Atomicity::Rename && working.exists()? {
            working.remove()?;
        }
        working.create_directory()?;
        if let Err(error) = checkpointables.names().try_for_each(|name| working.join(name).create_directory()) {
            working.remove().ok();
            return Err(error);
        }
    }
    multiprocessing.synchronize(&format!("directory_created:{directory}"))?;

    let initialization_timestamp_nanoseconds = timestamp_nanoseconds();
    let written = checkpointables
        .entries
        .iter()
        .try_for_each(|(name, checkpointable)| checkpointable.save(&working.join(name), multiprocessing))
        .and_then(|()| multiprocessing.synchronize(&format!("write_complete:{directory}")));
    if !primary {
        written?;
        return multiprocessing.synchronize(&format!("commit_complete:{directory}"));
    }
    let result = written.and_then(|()| {
        let metadata = StepMetadata::new(checkpointables.handlers(), initialization_timestamp_nanoseconds)
            .with_metrics(options.metrics().cloned())
            .with_custom_metadata(options.custom_metadata().clone());
        match atomicity {
            Atomicity::Rename => {
                metadata.with_commit_timestamp_nanoseconds(timestamp_nanoseconds()).write(&working)?;
                working.rename(&directory)
            }
            Atomicity::CommitFile => {
                metadata.write(&working)?;
                working.join(COMMIT_SUCCESS_FILE).write(&[])
            }
        }
    });
    if let Err(error) = result {
        // The error of the save is the one that callers need. A working directory that cannot be removed here is
        // removed by the next save that overwrites the same path or by `cleanup_temporary_paths`, and it is never
        // mistaken for a checkpoint.
        working.remove().ok();
        return Err(error);
    }
    // The other processes are released before the handlers run their hooks, so that a failing hook does not leave
    // them waiting at the last barrier.
    multiprocessing.synchronize(&format!("commit_complete:{directory}"))?;
    checkpointables
        .entries
        .iter()
        .try_for_each(|(name, checkpointable)| checkpointable.finalize(&directory.join(name)))
}

/// Checkpointable of [`Checkpointables`] bound to its saver, with the types of both erased.
trait BoundCheckpointable {
    /// Returns the name of the handler, as
    /// [`CheckpointableHandler::name`](crate::handlers::CheckpointableHandler::name) does.
    fn handler_name(&self) -> String;

    /// Saves the checkpointable into `directory`, as [`CheckpointableSaver::save`] does.
    fn save(&self, directory: &CheckpointPath, multiprocessing: &MultiprocessingOptions)
    -> Result<(), CheckpointError>;

    /// Finalizes the checkpointable in `directory`, as
    /// [`CheckpointableHandler::finalize`](crate::handlers::CheckpointableHandler::finalize) does.
    fn finalize(&self, directory: &CheckpointPath) -> Result<(), CheckpointError>;
}

/// [`BoundCheckpointable`] of a checkpointable of type `T` and its saver of type `H`.
struct HandledCheckpointable<'o, H, T: ?Sized> {
    /// Saver of the checkpointable.
    handler: &'o H,

    /// Checkpointable to save.
    checkpointable: &'o T,
}

impl<H: CheckpointableSaver<T>, T: ?Sized> BoundCheckpointable for HandledCheckpointable<'_, H, T> {
    fn handler_name(&self) -> String {
        self.handler.name()
    }

    fn save(
        &self,
        directory: &CheckpointPath,
        multiprocessing: &MultiprocessingOptions,
    ) -> Result<(), CheckpointError> {
        self.handler.save(directory, self.checkpointable, multiprocessing)
    }

    fn finalize(&self, directory: &CheckpointPath) -> Result<(), CheckpointError> {
        self.handler.finalize(directory)
    }
}

/// Returns an error if `name` cannot name a checkpointable, as [`Checkpointables`] describes.
pub(crate) fn validate_checkpointable_name(name: &str) -> Result<(), CheckpointError> {
    if name.is_empty() || name == "." || name == ".." || name.chars().any(is_separator) {
        return Err(CheckpointError::InvalidArgument {
            message: format!("`{name}` is not a valid checkpointable name, which must be a single path component"),
        });
    }
    if name == STEP_METADATA_FILE_NAME || name == COMMIT_SUCCESS_FILE {
        return Err(CheckpointError::InvalidArgument {
            message: format!("`{name}` is a reserved file name of checkpoints and cannot name a checkpointable"),
        });
    }
    Ok(())
}

/// Returns the current time in nanoseconds since the Unix epoch, saturating at the limits of `u64`.
pub(crate) fn timestamp_nanoseconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use pretty_assertions::assert_eq;
    use ryft_core::{Array, ArrayType, DataType};
    use serde_json::json;

    use crate::contexts::tests::test_context;
    use crate::handlers::{CheckpointableHandler, JsonHandler};
    use crate::loading::{load, load_checkpointable};
    use crate::multiprocessing::{LocalProcessBarrier, ProcessBarrier};
    use crate::paths::{TEMPORARY_DIRECTORY_SUFFIX, is_checkpoint_finalized};

    use super::*;

    /// Handler whose saves fail after writing a file, which shows whether failed saves leave anything behind.
    struct FailingHandler;

    impl CheckpointableHandler for FailingHandler {
        fn name(&self) -> String {
            "FailingHandler".to_string()
        }
    }

    impl CheckpointableSaver<u32> for FailingHandler {
        fn save(
            &self,
            directory: &CheckpointPath,
            checkpointable: &u32,
            multiprocessing: &MultiprocessingOptions,
        ) -> Result<(), CheckpointError> {
            let _ = multiprocessing;
            directory.join("partial").write(b"").unwrap();
            Err(CheckpointError::InvalidArgument { message: format!("cannot save {checkpointable}") })
        }
    }

    /// Handler of numbers that records, for every call of its finalization hook, the directory that it was called
    /// with and whether the checkpoint was finalized by then, and whose hook fails for the number zero.
    #[derive(Default)]
    struct FinalizingHandler {
        finalized: Mutex<Vec<(CheckpointPath, bool)>>,
    }

    impl CheckpointableHandler for FinalizingHandler {
        fn name(&self) -> String {
            "FinalizingHandler".to_string()
        }

        fn finalize(&self, directory: &CheckpointPath) -> Result<(), CheckpointError> {
            let finalized = is_checkpoint_finalized(&directory.parent().unwrap())?;
            self.finalized.lock().unwrap().push((directory.clone(), finalized));
            match directory.join("number").read()?.as_slice() {
                b"0" => Err(CheckpointError::InvalidArgument { message: "cannot finalize 0".to_string() }),
                _ => Ok(()),
            }
        }
    }

    impl CheckpointableSaver<u32> for FinalizingHandler {
        fn save(
            &self,
            directory: &CheckpointPath,
            checkpointable: &u32,
            multiprocessing: &MultiprocessingOptions,
        ) -> Result<(), CheckpointError> {
            let _ = multiprocessing;
            directory.join("number").write(checkpointable.to_string().as_bytes())
        }
    }

    /// Returns `true` if `path` is a finalized checkpoint on the local file system.
    fn is_finalized(path: &Path) -> bool {
        is_checkpoint_finalized(&CheckpointPath::from(path)).unwrap()
    }

    /// Returns the local path of the temporary directory of the checkpoint `path`.
    fn temporary(path: &Path) -> std::path::PathBuf {
        temporary_path(&CheckpointPath::from(path)).path().to_path_buf()
    }

    /// Saves `value` as the JSON checkpointable `config` of the checkpoint at `path`.
    fn save_json<P: Into<CheckpointPath>>(
        path: P,
        value: &serde_json::Value,
        options: &SaveOptions,
        context: &Context,
    ) -> Result<(), CheckpointError> {
        save_checkpointables(
            path,
            &Checkpointables::new().with("config", &JsonHandler::default(), value),
            options,
            context,
        )
    }

    /// Loads the JSON checkpointable `config` of the checkpoint at `path`.
    fn load_json<P: Into<CheckpointPath>>(path: P) -> Result<serde_json::Value, CheckpointError> {
        load_checkpointable(path, "config", &JsonHandler::default(), ())
    }

    #[test]
    fn test_save_options() {
        let options = SaveOptions::new()
            .with_overwrite(true)
            .with_force(true)
            .with_metrics(BTreeMap::from([("loss".to_string(), 1.0)]))
            .with_custom_metadata(json!("note"));
        assert!(options.overwrite());
        assert!(options.force());
        assert_eq!(options.metrics(), Some(&BTreeMap::from([("loss".to_string(), 1.0)])));
        assert_eq!(options.custom_metadata(), &json!("note"));
        assert_eq!(
            SaveOptions::default(),
            SaveOptions { overwrite: false, force: false, metrics: None, custom_metadata: serde_json::Value::Null },
        );
    }

    #[test]
    fn test_checkpointables() {
        let handler = JsonHandler::default();
        let (first, second) = (json!(1), json!(2));
        let checkpointables = Checkpointables::new().with("first", &handler, &first).with("second", &handler, &second);
        assert_eq!(checkpointables.names().collect::<Vec<_>>(), vec!["first", "second"]);
        assert_eq!(format!("{checkpointables:?}"), r#"{"first": "JsonHandler", "second": "JsonHandler"}"#);
        assert_eq!(checkpointables.validate(), Ok(()));

        // Names must be unique single path components that are not reserved file names.
        let duplicate = Checkpointables::new().with("same", &handler, &first).with("same", &handler, &second);
        assert_eq!(
            duplicate.validate(),
            Err(CheckpointError::InvalidArgument {
                message: "checkpointable `same` is saved more than once".to_string()
            }),
        );
        for name in ["", ".", "..", "a/b"] {
            assert_eq!(
                Checkpointables::new().with(name, &handler, &first).validate(),
                Err(CheckpointError::InvalidArgument {
                    message: format!(
                        "`{name}` is not a valid checkpointable name, which must be a single path component",
                    ),
                }),
            );
        }
        for name in [STEP_METADATA_FILE_NAME, COMMIT_SUCCESS_FILE] {
            assert_eq!(
                Checkpointables::new().with(name, &handler, &first).validate(),
                Err(CheckpointError::InvalidArgument {
                    message: format!(
                        "`{name}` is a reserved file name of checkpoints and cannot name a checkpointable",
                    ),
                }),
            );
        }
    }

    #[test]
    fn test_save() {
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().join("1");
        let context = test_context();
        let state = (Array::vector(vec![1.0f32, 2.0]).unwrap(), vec![Array::scalar(3u8).unwrap()]);
        assert_eq!(save(&directory, &state, &SaveOptions::new(), &context), Ok(()));
        assert!(is_finalized(&directory));
        assert!(directory.join(STATE_CHECKPOINTABLE_NAME).is_dir());

        // The step metadata records the handler of the state.
        let metadata = StepMetadata::read(&CheckpointPath::from(directory.as_path())).unwrap();
        assert_eq!(
            metadata.checkpointable_handlers(),
            &BTreeMap::from([("state".to_string(), "ParameterizedHandler".to_string())]),
        );
        let abstract_state = (ArrayType::new_static(DataType::F32, [2]), vec![ArrayType::scalar(DataType::U8)]);
        assert_eq!(load(&directory, abstract_state, &context), Ok(state));
    }

    #[test]
    fn test_save_async() {
        let root = tempfile::tempdir().unwrap();
        let context = test_context();
        let state = vec![Array::scalar(3i64).unwrap()];
        let mut response = save_async(root.path().join("1"), state.clone(), SaveOptions::new(), &context).unwrap();
        assert_eq!(response.result(None), Ok(()));
        assert!(is_finalized(&root.path().join("1")));
        assert_eq!(load(root.path().join("1"), vec![ArrayType::scalar(DataType::I64)], &context), Ok(state.clone()));

        // Errors of the background save are returned by the response, and a dropped response waits for its save.
        let mut existing = save_async(root.path().join("1"), state.clone(), SaveOptions::new(), &context).unwrap();
        assert_eq!(
            existing.result(None),
            Err(CheckpointError::ExistingCheckpoint { path: root.path().join("1").display().to_string() }),
        );
        drop(save_async(root.path().join("2"), state, SaveOptions::new(), &context).unwrap());
        assert!(is_finalized(&root.path().join("2")));
    }

    #[test]
    fn test_save_checkpointables() {
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().join("1");
        let context = test_context();
        let options = SaveOptions::new()
            .with_metrics(BTreeMap::from([("accuracy".to_string(), 0.75)]))
            .with_custom_metadata(json!({"epoch": 2}));
        let handler = JsonHandler::default();
        let config = json!([1, 2]);
        let state = vec![Array::scalar(1u8).unwrap()];
        let parameterized = context.parameterized_handler();
        let checkpointables =
            Checkpointables::new().with("config", &handler, &config).with("state", &parameterized, &state);
        assert_eq!(save_checkpointables(&directory, &checkpointables, &options, &context), Ok(()));
        assert!(is_finalized(&directory));
        assert!(!temporary(&directory).exists());
        assert!(!directory.join(COMMIT_SUCCESS_FILE).exists());
        assert_eq!(load_json(&directory), Ok(json!([1, 2])));
        assert_eq!(load(&directory, vec![ArrayType::scalar(DataType::U8)], &context), Ok(state.clone()));

        // The step metadata records the handler of every checkpointable, the options, and ordered timestamps.
        let metadata = StepMetadata::read(&CheckpointPath::from(directory.as_path())).unwrap();
        assert_eq!(
            metadata.checkpointable_handlers(),
            &BTreeMap::from([
                ("config".to_string(), "JsonHandler".to_string()),
                ("state".to_string(), "ParameterizedHandler".to_string()),
            ]),
        );
        assert_eq!(metadata.metrics(), Some(&BTreeMap::from([("accuracy".to_string(), 0.75)])));
        assert_eq!(metadata.custom_metadata(), &json!({"epoch": 2}));
        assert!(metadata.initialization_timestamp_nanoseconds() > 0);
        assert!(metadata.commit_timestamp_nanoseconds().unwrap() >= metadata.initialization_timestamp_nanoseconds());

        // Invalid names are rejected before anything is written.
        let invalid = Checkpointables::new().with(STEP_METADATA_FILE_NAME, &handler, &config);
        assert!(matches!(
            save_checkpointables(root.path().join("2"), &invalid, &options, &context),
            Err(CheckpointError::InvalidArgument { .. }),
        ));
        assert!(!root.path().join("2").exists());
    }

    #[test]
    fn test_save_checkpointables_finalize() {
        // The hook of every handler runs once, with the final directory of its checkpointable, after the checkpoint
        // was finalized.
        let root = tempfile::tempdir().unwrap();
        let (context, handler) = (test_context(), FinalizingHandler::default());
        let checkpointables = Checkpointables::new().with("first", &handler, &1u32).with("second", &handler, &2u32);
        assert_eq!(
            save_checkpointables(root.path().join("1"), &checkpointables, &SaveOptions::new(), &context),
            Ok(())
        );
        let directory = CheckpointPath::from(root.path().join("1"));
        assert_eq!(
            handler.finalized.lock().unwrap().clone(),
            vec![(directory.join("first"), true), (directory.join("second"), true)],
        );

        // A failing hook fails the save, and the checkpoint keeps its final name.
        let checkpointables = Checkpointables::new().with("zero", &handler, &0u32);
        assert_eq!(
            save_checkpointables(root.path().join("2"), &checkpointables, &SaveOptions::new(), &context),
            Err(CheckpointError::InvalidArgument { message: "cannot finalize 0".to_string() }),
        );
        assert!(is_finalized(&root.path().join("2")));
    }

    #[test]
    fn test_save_checkpointables_commit_file() {
        // In commit file mode, the checkpoint is written in place, marked by the commit file, and records no commit
        // timestamp. Loading does not depend on the mode.
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().join("1");
        let context = test_context().with_atomicity(Atomicity::CommitFile);
        assert_eq!(save_json(&directory, &json!("committed"), &SaveOptions::new(), &context), Ok(()));
        assert!(is_finalized(&directory));
        assert!(directory.join(COMMIT_SUCCESS_FILE).is_file());
        assert!(!temporary(&directory).exists());
        assert_eq!(
            StepMetadata::read(&CheckpointPath::from(directory.as_path()))
                .unwrap()
                .commit_timestamp_nanoseconds(),
            None,
        );
        assert_eq!(load_json(&directory), Ok(json!("committed")));

        // Overwriting saves replace existing checkpoints, and failed saves remove the final directory that they wrote
        // into.
        assert!(matches!(
            save_json(&directory, &json!(2), &SaveOptions::new(), &context),
            Err(CheckpointError::ExistingCheckpoint { .. }),
        ));
        let overwrite = SaveOptions::new().with_overwrite(true);
        assert_eq!(save_json(&directory, &json!("overwritten"), &overwrite, &context), Ok(()));
        assert_eq!(load_json(&directory), Ok(json!("overwritten")));
        assert_eq!(
            save_checkpointables(
                root.path().join("3"),
                &Checkpointables::new().with("number", &FailingHandler, &7),
                &SaveOptions::new(),
                &context,
            ),
            Err(CheckpointError::InvalidArgument { message: "cannot save 7".to_string() }),
        );
        assert!(!root.path().join("3").exists());
        assert_eq!(
            save_checkpointables(
                &directory,
                &Checkpointables::new().with("number", &FailingHandler, &8),
                &overwrite,
                &context,
            ),
            Err(CheckpointError::InvalidArgument { message: "cannot save 8".to_string() }),
        );
        assert!(!directory.exists());
    }

    #[test]
    fn test_save_checkpointables_multiprocessing() {
        // Three threads play the role of processes that save one checkpoint together, in both atomicity modes. Only
        // the primary process writes, so exactly one set of files exists.
        for atomicity in [Atomicity::Rename, Atomicity::CommitFile] {
            let root = tempfile::tempdir().unwrap();
            let barrier: Arc<dyn ProcessBarrier> = Arc::new(LocalProcessBarrier::new(3));
            let threads = (0..3)
                .map(|process_index| {
                    let directory = root.path().join("1");
                    let multiprocessing = MultiprocessingOptions::new(process_index, 3)
                        .unwrap()
                        .with_barrier(barrier.clone())
                        .with_barrier_timeout(Some(Duration::from_secs(60)));
                    let context =
                        test_context().with_atomicity(atomicity).with_multiprocessing_options(multiprocessing);
                    std::thread::spawn(move || {
                        save_json(&directory, &json!(process_index), &SaveOptions::new(), &context)
                    })
                })
                .collect::<Vec<_>>();
            for thread in threads {
                assert_eq!(thread.join().unwrap(), Ok(()));
            }
            let mut names = fs::read_dir(root.path().join("1"))
                .unwrap()
                .map(|entry| entry.unwrap().file_name().into_string().unwrap())
                .collect::<Vec<_>>();
            names.sort();
            let mut expected = vec![STEP_METADATA_FILE_NAME.to_string(), "config".to_string()];
            if atomicity == Atomicity::CommitFile {
                expected.insert(1, COMMIT_SUCCESS_FILE.to_string());
            }
            assert_eq!(names, expected);
            assert_eq!(fs::read_dir(root.path().join("1").join("config")).unwrap().count(), 1);
            assert_eq!(fs::read_dir(root.path()).unwrap().count(), 1);
            assert_eq!(load_json(root.path().join("1")), Ok(json!(0)));
        }
    }

    #[test]
    fn test_save_checkpointables_multiprocessing_timeout() {
        // When only two of three processes take part, the save fails at the first barrier instead of waiting forever.
        let root = tempfile::tempdir().unwrap();
        let barrier: Arc<dyn ProcessBarrier> = Arc::new(LocalProcessBarrier::new(3));
        let threads = (0..2)
            .map(|process_index| {
                let directory = root.path().join("1");
                let multiprocessing = MultiprocessingOptions::new(process_index, 3)
                    .unwrap()
                    .with_barrier(barrier.clone())
                    .with_barrier_timeout(Some(Duration::from_millis(50)));
                let context = test_context().with_multiprocessing_options(multiprocessing);
                std::thread::spawn(move || save_json(&directory, &json!(1), &SaveOptions::new(), &context))
            })
            .collect::<Vec<_>>();
        for thread in threads {
            assert!(matches!(
                thread.join().unwrap(),
                Err(CheckpointError::Barrier { key, .. }) if key.starts_with("save_start:"),
            ));
        }
        assert!(!root.path().join("1").exists());
    }

    #[test]
    fn test_save_checkpointables_multiprocessing_without_barrier() {
        // Several processes that share a primary process need a barrier, which the first phase reports.
        let root = tempfile::tempdir().unwrap();
        let context = test_context().with_multiprocessing_options(MultiprocessingOptions::new(0, 2).unwrap());
        assert_eq!(
            save_json(root.path().join("1"), &json!(1), &SaveOptions::new(), &context),
            Err(CheckpointError::InvalidArgument {
                message: "2 processes that share a primary process need a barrier to save a checkpoint together"
                    .to_string(),
            }),
        );
        assert!(!root.path().join("1").exists());
    }

    #[test]
    fn test_save_checkpointables_existing() {
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().join("1");
        let context = test_context();
        save_json(&directory, &json!(1), &SaveOptions::new(), &context).unwrap();

        // An existing checkpoint is kept unless the save overwrites it, in which case it is replaced.
        assert_eq!(
            save_json(&directory, &json!(2), &SaveOptions::new(), &context),
            Err(CheckpointError::ExistingCheckpoint { path: directory.display().to_string() }),
        );
        assert_eq!(load_json(&directory), Ok(json!(1)));
        assert_eq!(save_json(&directory, &json!(2), &SaveOptions::new().with_overwrite(true), &context), Ok(()));
        assert_eq!(load_json(&directory), Ok(json!(2)));

        // A stale temporary directory of an interrupted save is replaced.
        let stale = root.path().join(format!("2{TEMPORARY_DIRECTORY_SUFFIX}"));
        fs::create_dir(&stale).unwrap();
        fs::write(stale.join("stale"), "").unwrap();
        assert_eq!(save_json(root.path().join("2"), &json!(3), &SaveOptions::new(), &context), Ok(()));
        assert!(!stale.exists());
        assert!(!root.path().join("2").join("stale").exists());
    }

    #[test]
    fn test_save_checkpointables_failure() {
        // A failing handler leaves neither the final nor the temporary directory behind.
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().join("1");
        assert_eq!(
            save_checkpointables(
                &directory,
                &Checkpointables::new().with("number", &FailingHandler, &5),
                &SaveOptions::new(),
                &test_context(),
            ),
            Err(CheckpointError::InvalidArgument { message: "cannot save 5".to_string() }),
        );
        assert!(!directory.exists());
        assert!(!temporary(&directory).exists());
    }
}
