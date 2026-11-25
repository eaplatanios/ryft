//! Checkpoints of training runs, mirroring the `training` module of the Orbax v1 API.
//!
//! A [`Checkpointer`] owns a root directory with one finalized checkpoint per saved step, named by a
//! [`StepNameFormat`], and optionally the [`RootMetadata`] of the whole run. It saves every step atomically with
//! [`save_checkpointables`] and the [`Context`] that it owns, decides which steps to save with a
//! [`SaveDecisionPolicy`], and decides which checkpoints to keep with a [`PreservationPolicy`]. It remembers the
//! [`CheckpointMetadata`] of every checkpoint (i.e., its step, path, timestamps, custom metadata, and metrics), so
//! that policies, [`Checkpointer::latest`], and [`Checkpointer::checkpoints`] do not touch the file system.
//! [`Checkpointer::reload`] scans the root directory again after other processes changed it.
//!
//! The root directory is a [`CheckpointPath`], so it may live on any [`FileSystem`](crate::filesystems::FileSystem)
//! (e.g., an object store), and steps are finalized in the [`Atomicity`](crate::paths::Atomicity) mode that the
//! context selects or, by default, that the file system supports. On file systems without real directories, an empty
//! root directory does not exist, and creating it does nothing.
//!
//! [`Checkpointer::save_async`] saves on a background thread, like the asynchronous checkpointing of Orbax. At most
//! one save is in flight. Its step is visible immediately through [`Checkpointer::checkpoints`] and
//! [`Checkpointer::latest`]. The functions that save, load, read the metadata of a step, delete, or reload wait for
//! it first, and its error surfaces once from [`Checkpointer::wait`], [`Checkpointer::check_for_errors`], or the next
//! function that waits, which also removes its step again. Several processes save together when the context
//! carries [`MultiprocessingOptions`](crate::multiprocessing::MultiprocessingOptions) (refer to the
//! [`multiprocessing`](crate::multiprocessing) module), and a [`PreemptionSignal`] reports an upcoming preemption to
//! the save decision policy.
//!
//! The following parts of Orbax are not mirrored. Saves in flight are never cancelled. Metrics are stored in the step
//! metadata rather than as a checkpointable of their own. Loading cannot pad or truncate arrays whose shapes differ
//! from the abstract state. Leaves that are split across processes are not supported.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::thread::{Builder, JoinHandle};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use ryft_core::{ArrayType, Parameterized, ParameterizedFamily};
use serde::{Deserialize, Serialize};

use crate::contexts::Context;
use crate::errors::CheckpointError;
use crate::filesystems::CheckpointPath;
use crate::handlers::{ArrayLeafHandler, CheckpointableLoader, LeafHandler};
use crate::loading::{load, load_checkpointable, state_metadata};
use crate::metadata::{ParameterizedMetadata, StepMetadata, read_json, write_json};
use crate::paths::{StepNameFormat, cleanup_temporary_paths, delete, remove_checkpoint};
use crate::saving::{
    Checkpointables, STATE_CHECKPOINTABLE_NAME, SaveOptions, save, save_checkpointables, timestamp_nanoseconds,
};
use crate::synchronization::panic_message;

pub mod preservation_policies;
pub mod save_decision_policies;

pub use preservation_policies::{
    AnyPreservationPolicy, BestN, EveryNSeconds, EveryNSteps, LatestN, MetricMode, PreservationPolicy, PreserveAll,
};
pub use save_decision_policies::{
    AnySavePolicy, ContinuousCheckpointingPolicy, DecisionContext, FixedIntervalPolicy, InitialSavePolicy,
    PreemptionCheckpointingPolicy, PreemptionSignal, SaveDecisionPolicy, SpecificStepsPolicy,
};

// TODO(eaplatanios): Review this module.

/// Name of the subdirectory of a root directory that holds its [`RootMetadata`]. Step names never parse from this
/// name, so the directory is never mistaken for a step.
pub const ROOT_METADATA_DIRECTORY_NAME: &str = "metadata";

/// Name of the file in the [`ROOT_METADATA_DIRECTORY_NAME`] directory that holds the [`RootMetadata`].
pub const ROOT_METADATA_FILE_NAME: &str = "_ROOT_METADATA";

/// Metadata of one checkpoint of a training run, which mirrors the `CheckpointMetadata` of the training API of Orbax
/// and is what policies decide on. The metadata `M` is that of what was asked for (e.g., the
/// [`ParameterizedMetadata`] of the `state` checkpointable), and nothing for the checkpoints that a [`Checkpointer`]
/// lists.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckpointMetadata<M = ()> {
    /// Step of the checkpoint.
    step: u64,

    /// Path of the checkpoint.
    path: CheckpointPath,

    /// Metadata of what was asked for.
    metadata: M,

    /// Time at which saving started, in nanoseconds since the Unix epoch.
    initialization_timestamp_nanoseconds: u64,

    /// Time at which the checkpoint was committed, in nanoseconds since the Unix epoch, if it is known.
    commit_timestamp_nanoseconds: Option<u64>,

    /// Free-form custom metadata.
    custom_metadata: serde_json::Value,

    /// Metrics of the checkpoint, if any.
    metrics: Option<BTreeMap<String, f64>>,
}

impl CheckpointMetadata {
    /// Creates a new [`CheckpointMetadata`] without commit time, custom metadata (i.e., with
    /// [`serde_json::Value::Null`] custom metadata), and metrics.
    ///
    /// # Parameters
    ///
    ///   - `step`: Step of the checkpoint.
    ///   - `path`: Path of the checkpoint.
    ///   - `initialization_timestamp_nanoseconds`: Time at which saving started, in nanoseconds since the Unix epoch.
    #[inline]
    pub fn new(step: u64, path: CheckpointPath, initialization_timestamp_nanoseconds: u64) -> Self {
        Self {
            step,
            path,
            metadata: (),
            initialization_timestamp_nanoseconds,
            commit_timestamp_nanoseconds: None,
            custom_metadata: serde_json::Value::Null,
            metrics: None,
        }
    }

    /// Creates the [`CheckpointMetadata`] of the checkpoint of `step` at `path` from its `step_metadata`.
    fn from_step_metadata(step: u64, path: CheckpointPath, step_metadata: &StepMetadata) -> Self {
        Self {
            step,
            path,
            metadata: (),
            initialization_timestamp_nanoseconds: step_metadata.initialization_timestamp_nanoseconds(),
            commit_timestamp_nanoseconds: step_metadata.commit_timestamp_nanoseconds(),
            custom_metadata: step_metadata.custom_metadata().clone(),
            metrics: step_metadata.metrics().cloned(),
        }
    }
}

impl<M> CheckpointMetadata<M> {
    /// Returns this metadata with the provided commit time, in nanoseconds since the Unix epoch.
    #[inline]
    pub fn with_commit_timestamp_nanoseconds(mut self, commit_timestamp_nanoseconds: u64) -> Self {
        self.commit_timestamp_nanoseconds = Some(commit_timestamp_nanoseconds);
        self
    }

    /// Returns this metadata with the provided free-form custom metadata.
    #[inline]
    pub fn with_custom_metadata(mut self, custom_metadata: serde_json::Value) -> Self {
        self.custom_metadata = custom_metadata;
        self
    }

    /// Returns this metadata with the provided metrics.
    #[inline]
    pub fn with_metrics(mut self, metrics: Option<BTreeMap<String, f64>>) -> Self {
        self.metrics = metrics;
        self
    }

    /// Returns this metadata with `metadata` as the metadata of what was asked for.
    #[inline]
    pub fn with_metadata<N>(self, metadata: N) -> CheckpointMetadata<N> {
        CheckpointMetadata {
            step: self.step,
            path: self.path,
            metadata,
            initialization_timestamp_nanoseconds: self.initialization_timestamp_nanoseconds,
            commit_timestamp_nanoseconds: self.commit_timestamp_nanoseconds,
            custom_metadata: self.custom_metadata,
            metrics: self.metrics,
        }
    }

    /// Returns the step of the checkpoint.
    #[inline]
    pub fn step(&self) -> u64 {
        self.step
    }

    /// Returns the path of the checkpoint.
    #[inline]
    pub fn path(&self) -> &CheckpointPath {
        &self.path
    }

    /// Returns the metadata of what was asked for.
    #[inline]
    pub fn metadata(&self) -> &M {
        &self.metadata
    }

    /// Returns the time at which saving started, in nanoseconds since the Unix epoch.
    #[inline]
    pub fn initialization_timestamp_nanoseconds(&self) -> u64 {
        self.initialization_timestamp_nanoseconds
    }

    /// Returns the time at which the checkpoint was committed, in nanoseconds since the Unix epoch, if it is known,
    /// which it is for checkpoints that were finalized by renaming.
    #[inline]
    pub fn commit_timestamp_nanoseconds(&self) -> Option<u64> {
        self.commit_timestamp_nanoseconds
    }

    /// Returns the free-form custom metadata.
    #[inline]
    pub fn custom_metadata(&self) -> &serde_json::Value {
        &self.custom_metadata
    }

    /// Returns the metrics of the checkpoint, if any.
    #[inline]
    pub fn metrics(&self) -> Option<&BTreeMap<String, f64>> {
        self.metrics.as_ref()
    }

    /// Returns the value of the metric named `metric`, if the checkpoint has it.
    #[inline]
    pub fn metric(&self, metric: &str) -> Option<f64> {
        self.metrics.as_ref().and_then(|metrics| metrics.get(metric).copied())
    }

    /// Returns the time of the checkpoint, which is its commit time when that is known and the time at which saving
    /// started otherwise (e.g., for checkpoints that were finalized by a commit file, or for a step that a save is
    /// considered for).
    #[inline]
    pub fn time(&self) -> SystemTime {
        let nanoseconds = self.commit_timestamp_nanoseconds.unwrap_or(self.initialization_timestamp_nanoseconds);
        UNIX_EPOCH + Duration::from_nanos(nanoseconds)
    }
}

/// Custom metadata of the whole collection of checkpoints of a training run, stored in the [`ROOT_METADATA_FILE_NAME`]
/// file of the [`ROOT_METADATA_DIRECTORY_NAME`] subdirectory of its root directory, which mirrors
/// `metadata/_ROOT_METADATA` in Orbax.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RootMetadata {
    /// Free-form custom metadata.
    #[serde(default)]
    custom_metadata: serde_json::Value,
}

impl RootMetadata {
    /// Creates a new [`RootMetadata`] with the provided free-form custom metadata.
    #[inline]
    pub fn new(custom_metadata: serde_json::Value) -> Self {
        Self { custom_metadata }
    }

    /// Returns the free-form custom metadata.
    #[inline]
    pub fn custom_metadata(&self) -> &serde_json::Value {
        &self.custom_metadata
    }

    /// Writes this metadata into the root directory `root`, creating the [`ROOT_METADATA_DIRECTORY_NAME`]
    /// subdirectory if necessary and replacing an existing file.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::Io`] if the directory cannot be created or the file cannot be written.
    pub fn write(&self, root: &CheckpointPath) -> Result<(), CheckpointError> {
        root.join(ROOT_METADATA_DIRECTORY_NAME).create_directory()?;
        write_json(&Self::path(root), self)
    }

    /// Reads the metadata of the root directory `root`.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::Io`] if the file cannot be read, and [`CheckpointError::InvalidFile`] if it does not
    /// hold valid root metadata.
    pub fn read(root: &CheckpointPath) -> Result<Self, CheckpointError> {
        read_json(&Self::path(root))
    }

    /// Returns the path of the file that holds the root metadata of the root directory `root`.
    #[inline]
    pub fn path(root: &CheckpointPath) -> CheckpointPath {
        root.join(ROOT_METADATA_DIRECTORY_NAME).join(ROOT_METADATA_FILE_NAME)
    }
}

/// Options of a [`Checkpointer`], which mirror the keyword arguments of the `Checkpointer` of the training API of
/// Orbax. How checkpoints are stored, finalized, and shared between processes is configured by the [`Context`] of the
/// checkpointer instead.
pub struct CheckpointerOptions {
    /// Format of the names of step directories.
    step_name_format: StepNameFormat,

    /// Policy that decides which steps to save.
    save_decision_policy: Box<dyn SaveDecisionPolicy>,

    /// Policy that decides which checkpoints to keep.
    preservation_policy: Box<dyn PreservationPolicy>,

    /// Custom metadata of the whole run to write as [`RootMetadata`] when the root directory has none, if any.
    custom_metadata: Option<serde_json::Value>,

    /// Whether to remove temporary directories of interrupted saves when the checkpointer is created.
    cleanup_temporary_directories: bool,

    /// Flag that reports an upcoming preemption to the save decision policy, if any.
    preemption_signal: Option<PreemptionSignal>,
}

impl CheckpointerOptions {
    /// Creates new [`CheckpointerOptions`] that name steps by their plain step numbers, save as often as the
    /// [`ContinuousCheckpointingPolicy`] allows, keep every checkpoint, write no root metadata, and leave temporary
    /// directories alone.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns these options with the provided format of the names of step directories.
    #[inline]
    pub fn with_step_name_format(mut self, step_name_format: StepNameFormat) -> Self {
        self.step_name_format = step_name_format;
        self
    }

    /// Returns these options with the provided policy that decides which steps to save.
    #[inline]
    pub fn with_save_decision_policy<P: 'static + SaveDecisionPolicy>(mut self, policy: P) -> Self {
        self.save_decision_policy = Box::new(policy);
        self
    }

    /// Returns these options with the provided policy that decides which checkpoints to keep.
    #[inline]
    pub fn with_preservation_policy<P: 'static + PreservationPolicy>(mut self, policy: P) -> Self {
        self.preservation_policy = Box::new(policy);
        self
    }

    /// Returns these options with custom metadata of the whole run, which the checkpointer writes as [`RootMetadata`]
    /// when the root directory has none yet.
    #[inline]
    pub fn with_custom_metadata(mut self, custom_metadata: serde_json::Value) -> Self {
        self.custom_metadata = Some(custom_metadata);
        self
    }

    /// Returns these options with the provided flag that says whether to remove temporary directories of interrupted
    /// saves when the checkpointer is created.
    #[inline]
    pub fn with_cleanup_temporary_directories(mut self, cleanup_temporary_directories: bool) -> Self {
        self.cleanup_temporary_directories = cleanup_temporary_directories;
        self
    }

    /// Returns these options with `preemption_signal` reported to the save decision policy as
    /// [`DecisionContext::reached_preemption`]. The configured policy stays the sole authority, so a policy that
    /// should save on preemption must include a [`PreemptionCheckpointingPolicy`] (e.g., through an
    /// [`AnySavePolicy`]).
    #[inline]
    pub fn with_preemption_signal(mut self, preemption_signal: PreemptionSignal) -> Self {
        self.preemption_signal = Some(preemption_signal);
        self
    }

    /// Returns the format of the names of step directories.
    #[inline]
    pub fn step_name_format(&self) -> &StepNameFormat {
        &self.step_name_format
    }

    /// Returns the policy that decides which steps to save.
    #[inline]
    pub fn save_decision_policy(&self) -> &dyn SaveDecisionPolicy {
        self.save_decision_policy.as_ref()
    }

    /// Returns the policy that decides which checkpoints to keep.
    #[inline]
    pub fn preservation_policy(&self) -> &dyn PreservationPolicy {
        self.preservation_policy.as_ref()
    }

    /// Returns the custom metadata of the whole run that the checkpointer writes when the root directory has none, if
    /// any.
    #[inline]
    pub fn custom_metadata(&self) -> Option<&serde_json::Value> {
        self.custom_metadata.as_ref()
    }

    /// Returns whether temporary directories of interrupted saves are removed when the checkpointer is created.
    #[inline]
    pub fn cleanup_temporary_directories(&self) -> bool {
        self.cleanup_temporary_directories
    }

    /// Returns the flag that reports an upcoming preemption, if any.
    #[inline]
    pub fn preemption_signal(&self) -> Option<&PreemptionSignal> {
        self.preemption_signal.as_ref()
    }
}

impl Default for CheckpointerOptions {
    fn default() -> Self {
        Self {
            step_name_format: StepNameFormat::new(),
            save_decision_policy: Box::new(ContinuousCheckpointingPolicy::new(None)),
            preservation_policy: Box::new(PreserveAll),
            custom_metadata: None,
            cleanup_temporary_directories: false,
            preemption_signal: None,
        }
    }
}

/// Sequence of step checkpoints of a training run in one root directory, which mirrors the `Checkpointer` of the
/// training API of Orbax. Refer to the [module documentation](self) for its responsibilities, asynchronous saving,
/// and what it does not mirror. Dropping a checkpointer waits for its save in flight, and an error of that save is
/// then lost unless [`Self::wait`] or [`Self::close`] was called first.
pub struct Checkpointer<L: LeafHandler = ArrayLeafHandler> {
    /// Root directory of the checkpoints.
    directory: CheckpointPath,

    /// Context of saving and loading, which is shared with the thread of an asynchronous save.
    context: Arc<Context<L>>,

    /// Options of this checkpointer, which are shared with the thread of an asynchronous save.
    options: Arc<CheckpointerOptions>,

    /// Metadata of every checkpoint, ordered by step, including the step of an asynchronous save in flight.
    checkpoints: Vec<CheckpointMetadata>,

    /// Asynchronous save in flight, if any, until it is joined.
    pending: Option<PendingSave>,
}

impl<L: LeafHandler> Checkpointer<L> {
    /// Creates a new [`Checkpointer`] for the root directory `directory` that saves and loads with `context`. The root
    /// directory is created when it is missing, temporary directories of interrupted saves are removed when `options`
    /// ask for it, the custom metadata of `options` is written as [`RootMetadata`] when the root directory has none,
    /// and the existing finalized checkpoints are loaded. Only the primary process removes temporary directories and
    /// writes root metadata.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::Io`] if the root directory cannot be created, listed, or cleaned up, and the errors
    /// of [`StepMetadata::read`] for a finalized checkpoint whose step metadata is invalid.
    pub fn new<P: Into<CheckpointPath>>(
        directory: P,
        context: Context<L>,
        options: CheckpointerOptions,
    ) -> Result<Self, CheckpointError> {
        let directory = directory.into();
        if !directory.is_directory()? {
            directory.create_directory()?;
        }
        if context.multiprocessing_options().is_primary() {
            if options.cleanup_temporary_directories() {
                cleanup_temporary_paths(&directory, options.step_name_format())?;
            }
            if let Some(custom_metadata) = options.custom_metadata()
                && !RootMetadata::path(&directory).exists()?
            {
                RootMetadata::new(custom_metadata.clone()).write(&directory)?;
            }
        }
        let mut checkpointer = Self {
            directory,
            context: Arc::new(context),
            options: Arc::new(options),
            checkpoints: Vec::new(),
            pending: None,
        };
        checkpointer.reload()?;
        Ok(checkpointer)
    }

    /// Returns the root directory of the checkpoints.
    #[inline]
    pub fn directory(&self) -> &CheckpointPath {
        &self.directory
    }

    /// Returns the context of saving and loading.
    #[inline]
    pub fn context(&self) -> &Context<L> {
        &self.context
    }

    /// Returns the options of this checkpointer.
    #[inline]
    pub fn options(&self) -> &CheckpointerOptions {
        &self.options
    }

    /// Returns the metadata of all checkpoints, in increasing order of their steps, including the step of an
    /// asynchronous save in flight.
    #[inline]
    pub fn checkpoints(&self) -> &[CheckpointMetadata] {
        &self.checkpoints
    }

    /// Returns the metadata of the checkpoint with the largest step, including the step of an asynchronous save in
    /// flight, if there is any checkpoint.
    #[inline]
    pub fn latest(&self) -> Option<&CheckpointMetadata> {
        self.checkpoints.last()
    }

    /// Returns `true` if the [`PreemptionSignal`] of the options has been notified.
    pub fn reached_preemption(&self) -> bool {
        self.options.preemption_signal().is_some_and(PreemptionSignal::is_notified)
    }

    /// Returns `true` if an asynchronous save is still running.
    pub fn is_saving_in_progress(&self) -> bool {
        self.pending.as_ref().is_some_and(|pending| !pending.thread.is_finished())
    }

    /// Returns `true` if [`Self::save`] would save `step` without forcing it, which is when `step` is larger than the
    /// latest step and the save decision policy agrees in the current [`DecisionContext`].
    pub fn should_save(&self, step: u64) -> bool {
        self.accepts(step, false)
    }

    /// Saves the parameter structure `state` as the checkpoint of `step` with [`Self::save_checkpointables`], as the
    /// single checkpointable [`STATE_CHECKPOINTABLE_NAME`] that the [`Context::parameterized_handler`] of the context
    /// saves. Returns whether the step was saved.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::save_checkpointables`].
    pub fn save<S: Parameterized<L::Leaf>>(
        &mut self,
        step: u64,
        state: &S,
        options: &SaveOptions,
    ) -> Result<bool, CheckpointError> {
        let context = self.context.clone();
        let handler = context.parameterized_handler();
        self.save_checkpointables(
            step,
            &Checkpointables::new().with(STATE_CHECKPOINTABLE_NAME, &handler, state),
            options,
        )
    }

    /// Saves `checkpointables` as the checkpoint of `step` and then deletes every older checkpoint that the
    /// preservation policy does not keep. An asynchronous save in flight is waited for first. Nothing is saved when
    /// `step` is not larger than the latest step, or when the save decision policy declines and `options` do not force
    /// the save. An existing directory of `step` (e.g., an unfinalized directory that another process left behind) is
    /// replaced only when `options` ask to overwrite it. With several processes, only the primary process deletes
    /// checkpoints. Returns whether the step was saved.
    ///
    /// # Errors
    ///
    /// Returns the error of the asynchronous save in flight, if it failed, [`CheckpointError::ExistingCheckpoint`] if
    /// the directory of `step` exists and is not overwritten, the errors of [`save_checkpointables`], and
    /// [`CheckpointError::Io`] if a checkpoint that is not preserved cannot be deleted.
    pub fn save_checkpointables(
        &mut self,
        step: u64,
        checkpointables: &Checkpointables<'_>,
        options: &SaveOptions,
    ) -> Result<bool, CheckpointError> {
        self.wait()?;
        if !self.accepts(step, options.force()) {
            return Ok(false);
        }
        let path = self.step_path(step);
        if !options.overwrite() && path.exists()? {
            return Err(CheckpointError::ExistingCheckpoint { path: path.to_string() });
        }
        // A save can fail after its checkpoint was finalized (e.g., at the last barrier or while older checkpoints
        // are deleted), so after a failure the checkpoints are read back from storage. The error of the save is the
        // one that callers need, also when the root directory cannot be read.
        let primary = self.context.multiprocessing_options().is_primary();
        let saved = save_checkpointables(&path, checkpointables, options, &self.context)
            .and_then(|()| complete_save(self.checkpoints.clone(), step, &self.directory, &self.options, primary));
        match saved {
            Ok(checkpoints) => {
                self.checkpoints = checkpoints;
                Ok(true)
            }
            Err(error) => {
                self.reload().ok();
                Err(error)
            }
        }
    }

    /// Starts saving the parameter structure `state` as the checkpoint of `step` on a background thread, which
    /// mirrors `save_async` of the training API of Orbax, and returns whether the save started without waiting for it.
    /// The rules of [`Self::save_checkpointables`] decide whether to save, with the [`DecisionContext`] reporting the
    /// save in flight, if any. When the save goes ahead, the save in flight is waited for first, because at most one
    /// save runs at a time. The new step is recorded immediately, so [`Self::checkpoints`] and [`Self::latest`] include
    /// it, and the background thread deletes the checkpoints that the preservation policy does not keep once the save
    /// is finalized. Errors of the background save surface from the next function that waits for it, which also
    /// removes the step again.
    ///
    /// # Errors
    ///
    /// Returns the error of the asynchronous save in flight, if it failed, [`CheckpointError::ExistingCheckpoint`] if
    /// the directory of `step` exists and is not overwritten, and [`CheckpointError::Io`] if the background thread
    /// cannot be started.
    pub fn save_async<S>(&mut self, step: u64, state: S, options: SaveOptions) -> Result<bool, CheckpointError>
    where
        L: 'static + Send + Sync,
        S: 'static + Send + Parameterized<L::Leaf>,
    {
        // A save that has already finished is joined first, so that its error surfaces here and its step no longer
        // counts as being in flight.
        self.check_for_errors()?;
        if !self.accepts(step, options.force()) {
            return Ok(false);
        }
        self.wait()?;
        if self.latest().is_some_and(|latest| step <= latest.step()) {
            return Ok(false);
        }
        let path = self.step_path(step);
        if !options.overwrite() && path.exists()? {
            return Err(CheckpointError::ExistingCheckpoint { path: path.to_string() });
        }

        // The step is recorded optimistically with the time, the custom metadata, and the metrics of the save, and the
        // entry is replaced by the one read back from the step metadata once the save is joined.
        let previous = self.checkpoints.clone();
        self.checkpoints.push(
            CheckpointMetadata::new(step, path.clone(), timestamp_nanoseconds())
                .with_custom_metadata(options.custom_metadata().clone())
                .with_metrics(options.metrics().cloned()),
        );
        let checkpoints = self.checkpoints.clone();
        let context = self.context.clone();
        let checkpointer_options = self.options.clone();
        let directory = self.directory.clone();
        let thread = Builder::new()
            .name("ryft-checkpointer-save".to_string())
            .spawn(move || {
                save(&path, &state, &options, &context)?;
                let primary = context.multiprocessing_options().is_primary();
                complete_save(checkpoints, step, &directory, &checkpointer_options, primary)
            })
            .map_err(|error| CheckpointError::Io { path: self.directory.to_string(), message: error.to_string() });
        match thread {
            Ok(thread) => {
                self.pending = Some(PendingSave { step, previous, thread });
                Ok(true)
            }
            Err(error) => {
                self.checkpoints = previous;
                Err(error)
            }
        }
    }

    /// Waits for the asynchronous save in flight, if any, which mirrors `wait` of the training API of Orbax. When it
    /// succeeded, the checkpoints reflect the saved checkpoint and the checkpoints that it deleted. When it failed, its
    /// error is returned, once, and the checkpoints are read back from the root directory, because a save can fail
    /// after its checkpoint was finalized (e.g., while it deletes older checkpoints). If the root directory cannot be
    /// read either, the checkpoints are restored to what they were before the save started.
    ///
    /// # Errors
    ///
    /// Returns the error of the asynchronous save, and [`CheckpointError::Panic`] if its thread panicked.
    pub fn wait(&mut self) -> Result<(), CheckpointError> {
        let Some(pending) = self.pending.take() else {
            return Ok(());
        };
        match pending
            .thread
            .join()
            .unwrap_or_else(|payload| Err(CheckpointError::Panic { message: panic_message(payload) }))
        {
            Ok(checkpoints) => {
                self.checkpoints = checkpoints;
                Ok(())
            }
            Err(error) => {
                self.checkpoints = pending.previous;
                self.reload().ok();
                Err(error)
            }
        }
    }

    /// Returns the error of the asynchronous save in flight if it has finished and failed, without waiting for it,
    /// and otherwise behaves like [`Self::wait`] for a finished save.
    ///
    /// # Errors
    ///
    /// Returns the error of the finished asynchronous save, and [`CheckpointError::Panic`] if its thread panicked.
    pub fn check_for_errors(&mut self) -> Result<(), CheckpointError> {
        if self.pending.as_ref().is_some_and(|pending| pending.thread.is_finished()) { self.wait() } else { Ok(()) }
    }

    /// Loads the parameter structure of the checkpoint of `step`, or of the latest checkpoint when `step` is [`None`],
    /// into the structure that `abstract_state` describes with [`load`], after waiting for an asynchronous save in
    /// flight.
    ///
    /// # Errors
    ///
    /// Returns the error of the asynchronous save in flight, if it failed, [`CheckpointError::MissingCheckpoint`] if
    /// there is no checkpoint or if `step` has no checkpoint, and the errors of [`load`].
    pub fn load<S: Parameterized<ArrayType>>(
        &mut self,
        step: Option<u64>,
        abstract_state: S,
    ) -> Result<S::To<L::Leaf>, CheckpointError>
    where
        S::Family: ParameterizedFamily<L::Leaf>,
    {
        let path = self.checkpoint_path(step)?;
        load(path, abstract_state, &self.context)
    }

    /// Loads the checkpointable named `name` of the checkpoint of `step`, or of the latest checkpoint when `step` is
    /// [`None`], with [`load_checkpointable`], after waiting for an asynchronous save in flight.
    ///
    /// # Errors
    ///
    /// Returns the error of the asynchronous save in flight, if it failed, [`CheckpointError::MissingCheckpoint`] if
    /// there is no checkpoint or if `step` has no checkpoint, and the errors of [`load_checkpointable`].
    pub fn load_checkpointable<A, H: CheckpointableLoader<A>>(
        &mut self,
        step: Option<u64>,
        name: &str,
        handler: &H,
        abstract_checkpointable: A,
    ) -> Result<H::Output, CheckpointError> {
        let path = self.checkpoint_path(step)?;
        load_checkpointable(path, name, handler, abstract_checkpointable)
    }

    /// Reads the [`CheckpointMetadata`] of the checkpoint of `step`, with the [`ParameterizedMetadata`] of its
    /// [`STATE_CHECKPOINTABLE_NAME`] checkpointable, after waiting for an asynchronous save in flight. A checkpoint
    /// that [`Self::save_checkpointables`] saved without that checkpointable is described by
    /// [`Self::checkpointables_metadata`] instead.
    ///
    /// # Errors
    ///
    /// Returns the error of the asynchronous save in flight, if it failed, [`CheckpointError::MissingCheckpoint`] if
    /// `step` has no checkpoint or its checkpoint has no [`STATE_CHECKPOINTABLE_NAME`] checkpointable, and the errors
    /// of [`StepMetadata::read`] and [`ParameterizedMetadata::read`].
    pub fn metadata(&mut self, step: u64) -> Result<CheckpointMetadata<ParameterizedMetadata>, CheckpointError> {
        let path = self.checkpoint_path(Some(step))?;
        let step_metadata = StepMetadata::read(&path)?;
        let metadata = state_metadata(&path, &step_metadata)?;
        Ok(CheckpointMetadata::from_step_metadata(step, path, &step_metadata).with_metadata(metadata))
    }

    /// Reads the [`CheckpointMetadata`] of the checkpoint of `step`, with the name of the handler of every
    /// checkpointable keyed by the name of the checkpointable, after waiting for an asynchronous save in flight.
    ///
    /// # Errors
    ///
    /// Returns the error of the asynchronous save in flight, if it failed, [`CheckpointError::MissingCheckpoint`] if
    /// `step` has no checkpoint, and the errors of [`StepMetadata::read`].
    pub fn checkpointables_metadata(
        &mut self,
        step: u64,
    ) -> Result<CheckpointMetadata<BTreeMap<String, String>>, CheckpointError> {
        let path = self.checkpoint_path(Some(step))?;
        let step_metadata = StepMetadata::read(&path)?;
        let handlers = step_metadata.checkpointable_handlers().clone();
        Ok(CheckpointMetadata::from_step_metadata(step, path, &step_metadata).with_metadata(handlers))
    }

    /// Reads the [`RootMetadata`] of the root directory, or returns [`None`] if it has none.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`RootMetadata::read`] if the root metadata file exists but cannot be read.
    pub fn root_metadata(&self) -> Result<Option<RootMetadata>, CheckpointError> {
        if RootMetadata::path(&self.directory).exists()? {
            RootMetadata::read(&self.directory).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Deletes the checkpoint of `step` with [`delete`], after waiting for an asynchronous save in flight.
    ///
    /// # Errors
    ///
    /// Returns the error of the asynchronous save in flight, if it failed, [`CheckpointError::MissingCheckpoint`] if
    /// `step` has no checkpoint, and the errors of [`delete`].
    pub fn delete(&mut self, step: u64) -> Result<(), CheckpointError> {
        let path = self.checkpoint_path(Some(step))?;
        delete(path, &self.context)?;
        self.checkpoints.retain(|checkpoint| checkpoint.step() != step);
        Ok(())
    }

    /// Scans the root directory again for finalized checkpoints and loads their step metadata, which picks up changes
    /// that other processes made, after waiting for an asynchronous save in flight.
    ///
    /// # Errors
    ///
    /// Returns the error of the asynchronous save in flight, if it failed, [`CheckpointError::Io`] if the root
    /// directory cannot be listed, and the errors of [`StepMetadata::read`] for a finalized checkpoint whose step
    /// metadata is invalid.
    pub fn reload(&mut self) -> Result<(), CheckpointError> {
        self.wait()?;
        self.checkpoints = self
            .options
            .step_name_format()
            .find_all(&self.directory)?
            .into_iter()
            .map(|(step, path)| {
                let step_metadata = StepMetadata::read(&path)?;
                Ok(CheckpointMetadata::from_step_metadata(step, path, &step_metadata))
            })
            .collect::<Result<_, CheckpointError>>()?;
        Ok(())
    }

    /// Waits for the asynchronous save in flight, if any, and closes this checkpointer.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::wait`].
    pub fn close(mut self) -> Result<(), CheckpointError> {
        self.wait()
    }

    /// Returns the [`DecisionContext`] of the next save.
    fn decision_context(&self) -> DecisionContext {
        // A save whose thread has finished is no longer in flight, even when it has not been joined yet.
        let in_progress = self.pending.as_ref().filter(|pending| !pending.thread.is_finished());
        DecisionContext::new()
            .with_in_progress_step(in_progress.map(|pending| pending.step))
            .with_reached_preemption(self.reached_preemption())
    }

    /// Returns `true` if a save of `step` goes ahead, which is when `step` is larger than the latest step and the save
    /// is forced or the save decision policy agrees.
    fn accepts(&self, step: u64, force: bool) -> bool {
        self.latest().is_none_or(|latest| step > latest.step())
            && (force
                || self.options.save_decision_policy().should_save(
                    &CheckpointMetadata::new(step, self.step_path(step), timestamp_nanoseconds()),
                    &self.checkpoints,
                    &self.decision_context(),
                ))
    }

    /// Waits for an asynchronous save in flight and returns the path of the checkpoint of `step`, or of the latest
    /// checkpoint when `step` is [`None`].
    fn checkpoint_path(&mut self, step: Option<u64>) -> Result<CheckpointPath, CheckpointError> {
        self.wait()?;
        let checkpoint = match step {
            Some(step) => self
                .checkpoints
                .binary_search_by_key(&step, CheckpointMetadata::step)
                .ok()
                .map(|index| &self.checkpoints[index])
                .ok_or_else(|| CheckpointError::MissingCheckpoint {
                    message: format!("step {step} has no checkpoint in `{}`", self.directory),
                })?,
            None => self.latest().ok_or_else(|| CheckpointError::MissingCheckpoint {
                message: format!("`{}` holds no checkpoint", self.directory),
            })?,
        };
        Ok(checkpoint.path().clone())
    }

    /// Returns the path of the directory of `step`.
    fn step_path(&self, step: u64) -> CheckpointPath {
        self.directory.join(self.options.step_name_format().build_name(step))
    }
}

impl<L: LeafHandler> Drop for Checkpointer<L> {
    fn drop(&mut self) {
        // Joining keeps the save from being cut short when the process exits. Its result has no caller to go to.
        if let Some(pending) = self.pending.take() {
            pending.thread.join().ok();
        }
    }
}

/// Asynchronous save of a [`Checkpointer`] that has not been joined yet.
struct PendingSave {
    /// Step that is being saved.
    step: u64,

    /// Checkpoints from before the save started, which are restored when it fails.
    previous: Vec<CheckpointMetadata>,

    /// Background thread, which returns the checkpoints after the save and its deletions.
    thread: JoinHandle<Result<Vec<CheckpointMetadata>, CheckpointError>>,
}

/// Completes the save of `step`, whose checkpoint was just finalized, and returns the updated `checkpoints`. The entry
/// of `step` is replaced by the one read back from its step metadata, and every older checkpoint that the preservation
/// policy does not keep is removed from the list and, in the `primary` process, deleted.
fn complete_save(
    mut checkpoints: Vec<CheckpointMetadata>,
    step: u64,
    directory: &CheckpointPath,
    options: &CheckpointerOptions,
    primary: bool,
) -> Result<Vec<CheckpointMetadata>, CheckpointError> {
    let path = directory.join(options.step_name_format().build_name(step));
    let step_metadata = StepMetadata::read(&path)?;
    checkpoints.retain(|checkpoint| checkpoint.step() != step);
    let index = checkpoints.partition_point(|checkpoint| checkpoint.step() < step);
    checkpoints.insert(index, CheckpointMetadata::from_step_metadata(step, path, &step_metadata));

    // The checkpoint that was just saved is always kept, even when the policy would not keep it.
    let preserve = options.preservation_policy().should_preserve(&checkpoints);
    let deleted = checkpoints
        .iter()
        .zip(preserve)
        .filter(|(checkpoint, keep)| !keep && checkpoint.step() != step)
        .map(|(checkpoint, _)| checkpoint.path().clone())
        .collect::<Vec<_>>();
    for deleted_path in &deleted {
        if primary {
            remove_checkpoint(deleted_path)?;
        }
        checkpoints.retain(|checkpoint| checkpoint.path() != deleted_path);
    }
    Ok(checkpoints)
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::sync::{Condvar, Mutex};
    use std::{fs, thread};

    use pretty_assertions::assert_eq;
    use ryft_core::{Array, DataType};
    use serde_json::json;

    use crate::arrays::{ArraySpecification, Checkpoint};
    use crate::contexts::tests::test_context;
    #[cfg(feature = "object-store")]
    use crate::filesystems::objects::tests::test_object_store_root;
    use crate::formats::ArrayFormat;
    use crate::formats::tests::TestFormat;
    use crate::handlers::JsonHandler;
    use crate::metadata::{LeafMetadata, STEP_METADATA_FILE_NAME};
    use crate::multiprocessing::{LocalProcessBarrier, MultiprocessingOptions, ProcessBarrier};
    #[cfg(feature = "object-store")]
    use crate::paths::COMMIT_SUCCESS_FILE;
    use crate::paths::is_checkpoint_finalized;

    use super::*;

    /// Returns the metadata of a checkpoint of `step` saved `seconds` after the Unix epoch, with a `loss` metric if
    /// one is provided, which the tests of the policies decide on.
    pub(super) fn checkpoint(step: u64, seconds: u64, loss: Option<f64>) -> CheckpointMetadata {
        CheckpointMetadata::new(step, CheckpointPath::local(step.to_string()), seconds * 1_000_000_000)
            .with_metrics(loss.map(|loss| BTreeMap::from([("loss".to_string(), loss)])))
    }

    /// Returns save options that record a `loss` metric.
    fn with_loss(loss: f64) -> SaveOptions {
        SaveOptions::new().with_metrics(BTreeMap::from([("loss".to_string(), loss)]))
    }

    /// Returns the state that the tests save for `number`, which is one `i64` scalar.
    fn number(number: i64) -> Array {
        Array::scalar(number).unwrap()
    }

    /// Returns the abstract state of [`number`].
    fn abstract_number() -> ArrayType {
        ArrayType::scalar(DataType::I64)
    }

    /// [`ArrayFormat`] that stores numbers like the [`TestFormat`], fails to save negative numbers, and blocks every
    /// save while it is held.
    #[derive(Default)]
    struct NumberFormat {
        held: Mutex<bool>,
        released: Condvar,
    }

    impl NumberFormat {
        /// Makes every save wait until [`Self::release`] is called.
        fn hold(&self) {
            *self.held.lock().unwrap() = true;
        }

        /// Lets every waiting and future save proceed.
        fn release(&self) {
            *self.held.lock().unwrap() = false;
            self.released.notify_all();
        }
    }

    impl ArrayFormat for NumberFormat {
        fn name(&self) -> &'static str {
            "number"
        }

        fn write<'a>(
            &self,
            directory: &CheckpointPath,
            specifications: &[ArraySpecification],
            produce: &mut dyn FnMut(usize) -> Result<Cow<'a, Array>, CheckpointError>,
        ) -> Result<(), CheckpointError> {
            let mut held = self.held.lock().unwrap();
            while *held {
                held = self.released.wait(held).unwrap();
            }
            drop(held);
            TestFormat.write(directory, specifications, &mut |index| {
                let array = produce(index)?;
                match array.elements::<i64>()?.as_slice() {
                    [number] if *number < 0 => {
                        Err(CheckpointError::InvalidArgument { message: format!("cannot save {number}") })
                    }
                    _ => Ok(array),
                }
            })
        }

        fn open(&self, directory: &CheckpointPath) -> Result<Box<dyn Checkpoint>, CheckpointError> {
            TestFormat.open(directory)
        }
    }

    /// Returns a [`NumberFormat`] and a context that stores arrays in it.
    fn number_context() -> (Arc<NumberFormat>, Context) {
        let format = Arc::new(NumberFormat::default());
        (format.clone(), Context::new(ArrayLeafHandler, format))
    }

    /// [`ProcessBarrier`] of a single process that fails the barrier with one key and passes every other one, which
    /// fails a save after its checkpoint was finalized when the key is that of the last phase of the save.
    #[derive(Debug)]
    struct FailingBarrier {
        key: String,
    }

    impl ProcessBarrier for FailingBarrier {
        fn wait(&self, key: &str, timeout: Option<Duration>) -> Result<(), CheckpointError> {
            let _ = timeout;
            if key == self.key {
                return Err(CheckpointError::Barrier { key: key.to_string(), message: "failed".to_string() });
            }
            Ok(())
        }
    }

    /// Save decision policy that records every context that it sees and declines while a save is in progress.
    #[derive(Clone, Default)]
    struct RecordingPolicy {
        contexts: Arc<Mutex<Vec<DecisionContext>>>,
    }

    impl SaveDecisionPolicy for RecordingPolicy {
        fn should_save(
            &self,
            step: &CheckpointMetadata,
            previous: &[CheckpointMetadata],
            context: &DecisionContext,
        ) -> bool {
            let _ = (step, previous);
            self.contexts.lock().unwrap().push(*context);
            !context.is_saving_in_progress()
        }
    }

    /// Returns the steps of the checkpoints of `checkpointer`.
    fn steps<L: LeafHandler>(checkpointer: &Checkpointer<L>) -> Vec<u64> {
        checkpointer.checkpoints().iter().map(CheckpointMetadata::step).collect()
    }

    #[test]
    fn test_checkpoint_metadata() {
        let metadata = CheckpointMetadata::new(3, CheckpointPath::local("root/3"), 10_000_000_000)
            .with_custom_metadata(json!("note"))
            .with_metrics(Some(BTreeMap::from([("loss".to_string(), 0.5)])));
        assert_eq!(metadata.step(), 3);
        assert_eq!(metadata.path(), &CheckpointPath::local("root/3"));
        assert_eq!(metadata.metadata(), &());
        assert_eq!(metadata.initialization_timestamp_nanoseconds(), 10_000_000_000);
        assert_eq!(metadata.commit_timestamp_nanoseconds(), None);
        assert_eq!(metadata.custom_metadata(), &json!("note"));
        assert_eq!(metadata.metrics(), Some(&BTreeMap::from([("loss".to_string(), 0.5)])));
        assert_eq!(metadata.metric("loss"), Some(0.5));
        assert_eq!(metadata.metric("accuracy"), None);

        // The time of a checkpoint is its commit time when that is known and the start of its save otherwise.
        assert_eq!(metadata.time(), UNIX_EPOCH + Duration::from_secs(10));
        let committed = metadata.with_commit_timestamp_nanoseconds(20_000_000_000).with_metadata("state");
        assert_eq!(committed.time(), UNIX_EPOCH + Duration::from_secs(20));
        assert_eq!(committed.metadata(), &"state");
        assert_eq!(committed.step(), 3);
        assert_eq!(checkpoint(3, 10, None).metric("loss"), None);
    }

    #[test]
    fn test_root_metadata_write() {
        let directory = tempfile::tempdir().unwrap();
        let root = CheckpointPath::from(directory.path());
        let metadata = RootMetadata::new(json!({"model": "tiny"}));
        assert_eq!(metadata.custom_metadata(), &json!({"model": "tiny"}));
        assert_eq!(metadata.write(&root), Ok(()));
        assert_eq!(RootMetadata::path(&root), root.join("metadata").join("_ROOT_METADATA"));
        assert!(directory.path().join("metadata").join("_ROOT_METADATA").is_file());
        assert_eq!(RootMetadata::read(&root), Ok(metadata));
    }

    #[test]
    fn test_checkpointer_options() {
        let options = CheckpointerOptions::new();
        assert_eq!(options.step_name_format(), &StepNameFormat::new());
        assert!(!options.cleanup_temporary_directories());
        assert_eq!(options.custom_metadata(), None);
        assert!(options.preemption_signal().is_none());
        let (step, context) = (checkpoint(3, 0, None), DecisionContext::new());
        assert!(options.save_decision_policy().should_save(&step, &[], &context));
        assert_eq!(options.preservation_policy().should_preserve(&[checkpoint(1, 0, None)]), vec![true]);

        let options = CheckpointerOptions::new()
            .with_step_name_format(StepNameFormat::new().with_prefix("step"))
            .with_save_decision_policy(FixedIntervalPolicy::new(2).unwrap())
            .with_preservation_policy(LatestN::new(0))
            .with_custom_metadata(json!(1))
            .with_cleanup_temporary_directories(true)
            .with_preemption_signal(PreemptionSignal::new());
        assert_eq!(options.step_name_format(), &StepNameFormat::new().with_prefix("step"));
        assert!(options.cleanup_temporary_directories());
        assert_eq!(options.custom_metadata(), Some(&json!(1)));
        assert!(options.preemption_signal().is_some());
        assert!(!options.save_decision_policy().should_save(&step, &[], &context));
        assert_eq!(options.preservation_policy().should_preserve(&[checkpoint(1, 0, None)]), vec![false]);
    }

    #[test]
    fn test_checkpointer_new() {
        // An existing root directory is scanned for finalized checkpoints with their metrics.
        let root = tempfile::tempdir().unwrap();
        let context = test_context();
        save(root.path().join("4"), &number(4), &with_loss(0.5), &context).unwrap();
        save(root.path().join("2"), &number(2), &SaveOptions::new(), &context).unwrap();
        fs::create_dir(root.path().join("6")).unwrap();
        fs::create_dir(root.path().join("8.ryft-checkpoint-tmp")).unwrap();
        let checkpointer = Checkpointer::new(root.path(), test_context(), CheckpointerOptions::new()).unwrap();
        assert_eq!(checkpointer.directory(), &CheckpointPath::from(root.path()));
        assert_eq!(checkpointer.context().array_options().format().name(), "test");
        assert_eq!(steps(&checkpointer), vec![2, 4]);
        assert_eq!(checkpointer.latest().map(CheckpointMetadata::step), Some(4));
        assert_eq!(checkpointer.latest().unwrap().path(), &CheckpointPath::from(root.path().join("4")));
        assert_eq!(checkpointer.checkpoints()[1].metrics(), Some(&BTreeMap::from([("loss".to_string(), 0.5)])));
        assert_eq!(checkpointer.checkpoints()[0].metrics(), None);
        assert!(root.path().join("8.ryft-checkpoint-tmp").exists());

        // Temporary directories are removed on request.
        let options = CheckpointerOptions::new().with_cleanup_temporary_directories(true);
        Checkpointer::new(root.path(), test_context(), options).unwrap();
        assert!(!root.path().join("8.ryft-checkpoint-tmp").exists());
        assert!(!root.path().join("6").exists());
    }

    #[test]
    fn test_checkpointer_new_missing_directory() {
        let root = tempfile::tempdir().unwrap();
        let missing = root.path().join("nested").join("checkpoints");
        let checkpointer = Checkpointer::new(&missing, test_context(), CheckpointerOptions::new()).unwrap();
        assert!(missing.is_dir());
        assert_eq!(checkpointer.checkpoints(), &[]);
        assert_eq!(checkpointer.latest(), None);
    }

    #[test]
    fn test_checkpointer_should_save() {
        let root = tempfile::tempdir().unwrap();
        let options = CheckpointerOptions::new().with_save_decision_policy(FixedIntervalPolicy::new(5).unwrap());
        let mut checkpointer = Checkpointer::new(root.path(), test_context(), options).unwrap();
        assert!(checkpointer.should_save(5));
        assert!(!checkpointer.should_save(6));
        checkpointer.save(10, &number(10), &SaveOptions::new()).unwrap();
        assert!(!checkpointer.should_save(5));
        assert!(!checkpointer.should_save(10));
        assert!(checkpointer.should_save(15));
    }

    #[test]
    fn test_checkpointer_save() {
        let root = tempfile::tempdir().unwrap();
        let options = CheckpointerOptions::new()
            .with_step_name_format(StepNameFormat::new().with_fixed_length(3))
            .with_save_decision_policy(FixedIntervalPolicy::new(2).unwrap());
        let mut checkpointer = Checkpointer::new(root.path(), test_context(), options).unwrap();
        assert_eq!(checkpointer.save(2, &number(2), &SaveOptions::new()), Ok(true));
        assert_eq!(is_checkpoint_finalized(&CheckpointPath::from(root.path().join("002"))), Ok(true));

        // Steps must increase, the policy may decline, and forcing bypasses the policy.
        let force = SaveOptions::new().with_force(true);
        assert_eq!(checkpointer.save(2, &number(2), &force), Ok(false));
        assert_eq!(checkpointer.save(1, &number(1), &force), Ok(false));
        assert_eq!(checkpointer.save(3, &number(3), &SaveOptions::new()), Ok(false));
        assert!(!root.path().join("003").exists());
        assert_eq!(checkpointer.save(3, &number(3), &force), Ok(true));
        assert_eq!(steps(&checkpointer), vec![2, 3]);
        assert_eq!(checkpointer.load(Some(3), abstract_number()), Ok(number(3)));

        // A directory that is in the way is replaced only when the save overwrites it, which forcing does not do.
        fs::create_dir(root.path().join("004")).unwrap();
        assert_eq!(
            checkpointer.save(4, &number(4), &force),
            Err(CheckpointError::ExistingCheckpoint { path: root.path().join("004").display().to_string() }),
        );
        assert_eq!(steps(&checkpointer), vec![2, 3]);
        assert_eq!(checkpointer.save(4, &number(4), &force.with_overwrite(true)), Ok(true));
        assert_eq!(steps(&checkpointer), vec![2, 3, 4]);
    }

    #[test]
    fn test_checkpointer_save_failure() {
        // A save that fails after its checkpoint was finalized keeps the step, which is read back from storage, so
        // the step is not saved again and preservation accounts for it.
        let root = tempfile::tempdir().unwrap();
        let barrier = FailingBarrier { key: format!("commit_complete:{}", root.path().join("2").display()) };
        let multiprocessing = MultiprocessingOptions::default().with_barrier(Arc::new(barrier));
        let (_, context) = number_context();
        let context = context.with_multiprocessing_options(multiprocessing);
        let options = CheckpointerOptions::new().with_preservation_policy(LatestN::new(1));
        let mut checkpointer = Checkpointer::new(root.path(), context, options).unwrap();
        assert_eq!(checkpointer.save(1, &number(1), &SaveOptions::new()), Ok(true));
        assert!(matches!(
            checkpointer.save(2, &number(2), &SaveOptions::new()),
            Err(CheckpointError::Barrier { message, .. }) if message == "failed",
        ));
        assert_eq!(steps(&checkpointer), vec![1, 2]);
        assert!(!checkpointer.should_save(2));
        assert_eq!(checkpointer.save(3, &number(3), &SaveOptions::new()), Ok(true));
        assert_eq!(steps(&checkpointer), vec![3]);

        // A save that fails before its checkpoint is finalized leaves no step behind.
        assert!(matches!(
            checkpointer.save(4, &number(-4), &SaveOptions::new()),
            Err(CheckpointError::InvalidArgument { .. }),
        ));
        assert_eq!(steps(&checkpointer), vec![3]);
        assert!(!root.path().join("4").exists());
    }

    #[test]
    fn test_checkpointer_save_preservation() {
        // The best checkpoint by loss and the latest one survive, and the one just saved is never deleted.
        let root = tempfile::tempdir().unwrap();
        let options = CheckpointerOptions::new().with_preservation_policy(BestN::new("loss", MetricMode::Minimize, 1));
        let mut checkpointer = Checkpointer::new(root.path(), test_context(), options).unwrap();
        assert_eq!(checkpointer.save(1, &number(1), &with_loss(0.5)), Ok(true));
        assert_eq!(checkpointer.save(2, &number(2), &with_loss(0.25)), Ok(true));
        assert_eq!(steps(&checkpointer), vec![2]);
        assert_eq!(checkpointer.save(3, &number(3), &with_loss(0.75)), Ok(true));
        assert_eq!(steps(&checkpointer), vec![2, 3]);
        assert_eq!(checkpointer.save(4, &number(4), &with_loss(f64::NAN)), Ok(true));
        assert_eq!(steps(&checkpointer), vec![2, 4]);
        assert!(!root.path().join("1").exists());
        assert!(!root.path().join("3").exists());
        assert!(root.path().join("2").exists());
    }

    #[test]
    fn test_checkpointer_save_checkpointables() {
        let root = tempfile::tempdir().unwrap();
        let mut checkpointer = Checkpointer::new(root.path(), test_context(), CheckpointerOptions::new()).unwrap();
        let context = test_context();
        let (json_handler, parameterized_handler) = (JsonHandler::default(), context.parameterized_handler());
        let (config, state) = (json!({"layers": 2}), number(1));
        let checkpointables = Checkpointables::new().with("config", &json_handler, &config);
        let checkpointables = checkpointables.with(STATE_CHECKPOINTABLE_NAME, &parameterized_handler, &state);
        assert_eq!(checkpointer.save_checkpointables(1, &checkpointables, &SaveOptions::new()), Ok(true));
        assert_eq!(checkpointer.load_checkpointable(None, "config", &json_handler, ()), Ok(config.clone()));
        assert_eq!(checkpointer.load(Some(1), abstract_number()), Ok(state.clone()));
        assert!(matches!(
            checkpointer.load_checkpointable(Some(1), "missing", &JsonHandler::default(), ()),
            Err(CheckpointError::MissingCheckpoint { .. }),
        ));
    }

    #[test]
    fn test_checkpointer_load() {
        let root = tempfile::tempdir().unwrap();
        let mut checkpointer = Checkpointer::new(root.path(), test_context(), CheckpointerOptions::new()).unwrap();
        assert_eq!(
            checkpointer.load(None, abstract_number()),
            Err(CheckpointError::MissingCheckpoint {
                message: format!("`{}` holds no checkpoint", root.path().display()),
            }),
        );
        checkpointer.save(1, &number(1), &SaveOptions::new()).unwrap();
        checkpointer.save(2, &number(2), &SaveOptions::new()).unwrap();
        assert_eq!(checkpointer.load(None, abstract_number()), Ok(number(2)));
        assert_eq!(checkpointer.load(Some(1), abstract_number()), Ok(number(1)));
        assert_eq!(
            checkpointer.load(Some(3), abstract_number()),
            Err(CheckpointError::MissingCheckpoint {
                message: format!("step 3 has no checkpoint in `{}`", root.path().display()),
            }),
        );
    }

    #[test]
    fn test_checkpointer_metadata() {
        let root = tempfile::tempdir().unwrap();
        let mut checkpointer = Checkpointer::new(root.path(), test_context(), CheckpointerOptions::new()).unwrap();
        checkpointer.save(1, &number(1), &with_loss(0.5).with_custom_metadata(json!("note"))).unwrap();
        let metadata = checkpointer.metadata(1).unwrap();
        assert_eq!(metadata.step(), 1);
        assert_eq!(metadata.metadata(), &ParameterizedMetadata::new("test", vec![LeafMetadata::new("$", "I64", [])]));
        assert_eq!(metadata.metrics(), Some(&BTreeMap::from([("loss".to_string(), 0.5)])));
        assert_eq!(metadata.custom_metadata(), &json!("note"));
        assert!(matches!(checkpointer.metadata(2), Err(CheckpointError::MissingCheckpoint { .. })));

        // A checkpoint without the `state` checkpointable has no parameterized metadata.
        let (handler, config) = (JsonHandler::default(), json!({"layers": 2}));
        let checkpointables = Checkpointables::new().with("config", &handler, &config);
        checkpointer.save_checkpointables(2, &checkpointables, &SaveOptions::new()).unwrap();
        assert!(matches!(
            checkpointer.metadata(2),
            Err(CheckpointError::MissingCheckpoint { message }) if message.contains("no checkpointable named `state`"),
        ));
    }

    #[test]
    fn test_checkpointer_checkpointables_metadata() {
        let root = tempfile::tempdir().unwrap();
        let mut checkpointer = Checkpointer::new(root.path(), test_context(), CheckpointerOptions::new()).unwrap();
        checkpointer.save(1, &number(1), &SaveOptions::new()).unwrap();
        assert_eq!(
            checkpointer.checkpointables_metadata(1).map(|metadata| metadata.metadata().clone()),
            Ok(BTreeMap::from([("state".to_string(), "ParameterizedHandler".to_string())])),
        );
        assert!(matches!(checkpointer.checkpointables_metadata(2), Err(CheckpointError::MissingCheckpoint { .. })));
    }

    #[test]
    fn test_checkpointer_root_metadata() {
        let root = tempfile::tempdir().unwrap();
        let checkpointer = Checkpointer::new(root.path(), test_context(), CheckpointerOptions::new()).unwrap();
        assert_eq!(checkpointer.root_metadata(), Ok(None));

        // Root metadata is written once and never replaced by later checkpointers.
        let options = CheckpointerOptions::new().with_custom_metadata(json!({"model": "a"}));
        let mut checkpointer = Checkpointer::new(root.path(), test_context(), options).unwrap();
        assert_eq!(checkpointer.root_metadata(), Ok(Some(RootMetadata::new(json!({"model": "a"})))));
        let options = CheckpointerOptions::new().with_custom_metadata(json!({"model": "b"}));
        let other = Checkpointer::new(root.path(), test_context(), options).unwrap();
        assert_eq!(other.root_metadata(), Ok(Some(RootMetadata::new(json!({"model": "a"})))));

        // The root metadata directory is never mistaken for a step.
        checkpointer.save(1, &number(1), &SaveOptions::new()).unwrap();
        checkpointer.reload().unwrap();
        assert_eq!(steps(&checkpointer), vec![1]);
    }

    #[test]
    fn test_checkpointer_delete() {
        let root = tempfile::tempdir().unwrap();
        let mut checkpointer = Checkpointer::new(root.path(), test_context(), CheckpointerOptions::new()).unwrap();
        checkpointer.save(1, &number(1), &SaveOptions::new()).unwrap();
        checkpointer.save(2, &number(2), &SaveOptions::new()).unwrap();
        assert_eq!(checkpointer.delete(1), Ok(()));
        assert_eq!(steps(&checkpointer), vec![2]);
        assert!(!root.path().join("1").exists());
        assert_eq!(
            checkpointer.delete(1),
            Err(CheckpointError::MissingCheckpoint {
                message: format!("step 1 has no checkpoint in `{}`", root.path().display()),
            }),
        );
    }

    #[test]
    fn test_checkpointer_reload() {
        let root = tempfile::tempdir().unwrap();
        let mut checkpointer = Checkpointer::new(root.path(), test_context(), CheckpointerOptions::new()).unwrap();
        checkpointer.save(1, &number(1), &SaveOptions::new()).unwrap();

        // Checkpoints that another process adds or removes appear after reloading.
        save(root.path().join("5"), &number(5), &with_loss(0.1), &test_context()).unwrap();
        fs::remove_file(root.path().join("1").join(STEP_METADATA_FILE_NAME)).unwrap();
        assert_eq!(steps(&checkpointer), vec![1]);
        assert_eq!(checkpointer.reload(), Ok(()));
        assert_eq!(steps(&checkpointer), vec![5]);
        assert_eq!(checkpointer.latest().unwrap().metric("loss"), Some(0.1));
    }

    #[cfg(feature = "object-store")]
    #[test]
    fn test_checkpointer_object_store() {
        // Steps in an object store are finalized with commit files, and the preservation policy deletes their objects.
        let root = test_object_store_root();
        let options = || {
            CheckpointerOptions::new()
                .with_preservation_policy(LatestN::new(2))
                .with_custom_metadata(json!("remote"))
        };
        let mut checkpointer = Checkpointer::new(root.clone(), test_context(), options()).unwrap();
        for step in 1..=3 {
            assert_eq!(checkpointer.save(step, &number(step as i64), &with_loss(step as f64)), Ok(true));
        }
        assert_eq!(steps(&checkpointer), vec![2, 3]);
        assert_eq!(root.list(), Ok(vec!["2".to_string(), "3".to_string(), "metadata".to_string()]));
        assert_eq!(root.join("3").join(COMMIT_SUCCESS_FILE).exists(), Ok(true));
        assert_eq!(checkpointer.metadata(3).unwrap().commit_timestamp_nanoseconds(), None);
        assert_eq!(checkpointer.latest().unwrap().metric("loss"), Some(3.0));
        assert_eq!(checkpointer.load(None, abstract_number()), Ok(number(3)));
        assert_eq!(checkpointer.load(Some(2), abstract_number()), Ok(number(2)));
        assert_eq!(checkpointer.root_metadata(), Ok(Some(RootMetadata::new(json!("remote")))));

        // A save that was interrupted before its commit file is ignored, blocks its step, and is cleaned up.
        let interrupted = root.join("4");
        interrupted.join(STATE_CHECKPOINTABLE_NAME).join("partial").write(b"").unwrap();
        StepMetadata::new(BTreeMap::new(), 1).write(&interrupted).unwrap();
        assert_eq!(checkpointer.reload(), Ok(()));
        assert_eq!(steps(&checkpointer), vec![2, 3]);
        assert_eq!(
            checkpointer.save(4, &number(4), &SaveOptions::new()),
            Err(CheckpointError::ExistingCheckpoint { path: interrupted.to_string() }),
        );
        let mut checkpointer =
            Checkpointer::new(root.clone(), test_context(), options().with_cleanup_temporary_directories(true))
                .unwrap();
        assert_eq!(interrupted.exists(), Ok(false));
        assert_eq!(checkpointer.save(4, &number(4), &SaveOptions::new()), Ok(true));
        assert_eq!(steps(&checkpointer), vec![3, 4]);

        // Deleting removes every object of a step.
        assert_eq!(checkpointer.delete(3), Ok(()));
        assert_eq!(steps(&checkpointer), vec![4]);
        assert_eq!(root.list(), Ok(vec!["4".to_string(), "metadata".to_string()]));
    }

    #[test]
    fn test_checkpointer_save_async() {
        let root = tempfile::tempdir().unwrap();
        // Every step is saved, also while a save is in flight, which the default policy would skip.
        let options = CheckpointerOptions::new().with_save_decision_policy(FixedIntervalPolicy::new(1).unwrap());
        let (format, context) = number_context();
        let mut checkpointer = Checkpointer::new(root.path(), context, options).unwrap();

        // The step is visible while it is saved, and its entry matches its step metadata once the save is joined.
        format.hold();
        assert_eq!(checkpointer.save_async(1, number(1), with_loss(0.5)), Ok(true));
        assert_eq!(steps(&checkpointer), vec![1]);
        assert_eq!(checkpointer.latest().unwrap().metric("loss"), Some(0.5));
        assert!(checkpointer.is_saving_in_progress());
        assert_eq!(checkpointer.check_for_errors(), Ok(()));
        format.release();
        assert_eq!(checkpointer.wait(), Ok(()));
        assert!(!checkpointer.is_saving_in_progress());
        let path = CheckpointPath::from(root.path().join("1"));
        let step_metadata = StepMetadata::read(&path).unwrap();
        assert_eq!(checkpointer.checkpoints(), &[CheckpointMetadata::from_step_metadata(1, path, &step_metadata)]);
        assert_eq!(checkpointer.latest().unwrap().metric("loss"), Some(0.5));
        assert_eq!(checkpointer.load(None, abstract_number()), Ok(number(1)));

        // A second asynchronous save and a synchronous save both wait for the save in flight.
        assert_eq!(checkpointer.save_async(2, number(2), SaveOptions::new()), Ok(true));
        assert_eq!(checkpointer.save_async(3, number(3), SaveOptions::new()), Ok(true));
        assert_eq!(is_checkpoint_finalized(&CheckpointPath::from(root.path().join("2"))), Ok(true));
        assert_eq!(checkpointer.save(4, &number(4), &SaveOptions::new()), Ok(true));
        assert_eq!(is_checkpoint_finalized(&CheckpointPath::from(root.path().join("3"))), Ok(true));
        assert_eq!(steps(&checkpointer), vec![1, 2, 3, 4]);
        assert_eq!(checkpointer.close(), Ok(()));
    }

    #[test]
    fn test_checkpointer_save_async_failure() {
        // A failing background save is rolled back, and its error is returned exactly once.
        let root = tempfile::tempdir().unwrap();
        // Every step is saved, also while a save is in flight, which the default policy would skip. The barrier fails
        // the last phase of the save of step 5, after its checkpoint was finalized.
        let options = CheckpointerOptions::new().with_save_decision_policy(FixedIntervalPolicy::new(1).unwrap());
        let barrier = FailingBarrier { key: format!("commit_complete:{}", root.path().join("5").display()) };
        let multiprocessing = MultiprocessingOptions::default().with_barrier(Arc::new(barrier));
        let (_, context) = number_context();
        let context = context.with_multiprocessing_options(multiprocessing);
        let mut checkpointer = Checkpointer::new(root.path(), context, options).unwrap();
        assert_eq!(checkpointer.save_async(1, number(1), SaveOptions::new()), Ok(true));
        assert_eq!(checkpointer.save_async(2, number(-2), SaveOptions::new()), Ok(true));
        assert_eq!(steps(&checkpointer), vec![1, 2]);
        assert_eq!(
            checkpointer.wait(),
            Err(CheckpointError::InvalidArgument { message: "cannot save -2".to_string() }),
        );
        assert_eq!(steps(&checkpointer), vec![1]);
        assert_eq!(checkpointer.wait(), Ok(()));
        assert!(!root.path().join("2").exists());

        // The error of an unchecked failure is returned by the next asynchronous save, which then does not start.
        assert_eq!(checkpointer.save_async(3, number(-3), SaveOptions::new()), Ok(true));
        assert_eq!(
            checkpointer.save_async(4, number(4), SaveOptions::new()),
            Err(CheckpointError::InvalidArgument { message: "cannot save -3".to_string() }),
        );
        assert_eq!(steps(&checkpointer), vec![1]);
        assert_eq!(checkpointer.check_for_errors(), Ok(()));

        // A save that fails after its checkpoint was finalized keeps the step, which is read back from storage.
        assert_eq!(checkpointer.save_async(5, number(5), SaveOptions::new()), Ok(true));
        assert!(matches!(checkpointer.wait(), Err(CheckpointError::Barrier { message, .. }) if message == "failed"));
        assert_eq!(steps(&checkpointer), vec![1, 5]);
        assert_eq!(is_checkpoint_finalized(&CheckpointPath::from(root.path().join("5"))), Ok(true));
    }

    #[test]
    fn test_checkpointer_save_async_with_default_options() {
        // The default policy declines while a save is running, and saves again once it has finished, without the
        // caller having to wait for it explicitly.
        let root = tempfile::tempdir().unwrap();
        let (format, context) = number_context();
        let mut checkpointer = Checkpointer::new(root.path(), context, CheckpointerOptions::new()).unwrap();
        format.hold();
        assert_eq!(checkpointer.save_async(1, number(1), SaveOptions::new()), Ok(true));
        assert_eq!(checkpointer.save_async(2, number(2), SaveOptions::new()), Ok(false));
        format.release();
        while checkpointer.is_saving_in_progress() {
            thread::yield_now();
        }
        assert_eq!(checkpointer.save_async(3, number(3), SaveOptions::new()), Ok(true));
        assert_eq!(checkpointer.wait(), Ok(()));
        assert_eq!(steps(&checkpointer), vec![1, 3]);
    }

    #[test]
    fn test_checkpointer_save_async_preservation() {
        let root = tempfile::tempdir().unwrap();
        let options = CheckpointerOptions::new()
            .with_save_decision_policy(FixedIntervalPolicy::new(1).unwrap())
            .with_preservation_policy(LatestN::new(1));
        let (_, context) = number_context();
        let mut checkpointer = Checkpointer::new(root.path(), context, options).unwrap();
        assert_eq!(checkpointer.save_async(1, number(1), SaveOptions::new()), Ok(true));
        assert_eq!(checkpointer.save_async(2, number(2), SaveOptions::new()), Ok(true));
        assert_eq!(steps(&checkpointer), vec![1, 2]);
        assert_eq!(checkpointer.wait(), Ok(()));
        assert_eq!(steps(&checkpointer), vec![2]);
        assert!(!root.path().join("1").exists());
    }

    #[test]
    fn test_checkpointer_decision_context() {
        // The policy sees the asynchronous save in flight until it is joined.
        let root = tempfile::tempdir().unwrap();
        let policy = RecordingPolicy::default();
        let options = CheckpointerOptions::new().with_save_decision_policy(policy.clone());
        let (format, context) = number_context();
        let mut checkpointer = Checkpointer::new(root.path(), context, options).unwrap();
        format.hold();
        assert_eq!(checkpointer.save_async(1, number(1), SaveOptions::new()), Ok(true));
        assert_eq!(checkpointer.save_async(2, number(2), SaveOptions::new()), Ok(false));
        format.release();
        assert_eq!(checkpointer.wait(), Ok(()));
        assert_eq!(checkpointer.save_async(3, number(3), SaveOptions::new()), Ok(true));
        assert_eq!(checkpointer.close(), Ok(()));
        assert_eq!(
            policy.contexts.lock().unwrap().clone(),
            vec![DecisionContext::new(), DecisionContext::new().with_in_progress_step(Some(1)), DecisionContext::new()],
        );
    }

    #[test]
    fn test_checkpointer_reached_preemption() {
        let root = tempfile::tempdir().unwrap();
        let signal = PreemptionSignal::new();
        let options = CheckpointerOptions::new()
            .with_save_decision_policy(AnySavePolicy::new(vec![
                Box::new(FixedIntervalPolicy::new(100).unwrap()),
                Box::new(PreemptionCheckpointingPolicy),
            ]))
            .with_preemption_signal(signal.clone());
        let mut checkpointer = Checkpointer::new(root.path(), test_context(), options).unwrap();
        assert!(!checkpointer.reached_preemption());
        assert_eq!(checkpointer.save(1, &number(1), &SaveOptions::new()), Ok(false));
        signal.notify();
        assert!(checkpointer.reached_preemption());
        assert!(checkpointer.should_save(2));
        assert_eq!(checkpointer.save(2, &number(2), &SaveOptions::new()), Ok(true));
        assert_eq!(steps(&checkpointer), vec![2]);
    }

    #[test]
    fn test_checkpointer_multiprocessing() {
        // Three threads play the role of processes that save the same steps, and only the primary process deletes.
        let root = tempfile::tempdir().unwrap();
        let barrier: Arc<dyn ProcessBarrier> = Arc::new(LocalProcessBarrier::new(3));
        let threads = (0..3)
            .map(|process_index| {
                let directory = root.path().to_path_buf();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    let multiprocessing = MultiprocessingOptions::new(process_index, 3)
                        .unwrap()
                        .with_barrier(barrier)
                        .with_barrier_timeout(Some(Duration::from_secs(60)));
                    let context = test_context().with_multiprocessing_options(multiprocessing);
                    let options = CheckpointerOptions::new().with_preservation_policy(LatestN::new(2));
                    let mut checkpointer = Checkpointer::new(directory, context, options)?;
                    for step in 1..=3 {
                        checkpointer.save(step, &number(step as i64), &SaveOptions::new())?;
                    }
                    Ok::<_, CheckpointError>(steps(&checkpointer))
                })
            })
            .collect::<Vec<_>>();
        for thread in threads {
            assert_eq!(thread.join().unwrap(), Ok(vec![2, 3]));
        }
        let mut names = fs::read_dir(root.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect::<Vec<_>>();
        names.sort();
        assert_eq!(names, vec!["2".to_string(), "3".to_string()]);
    }
}
