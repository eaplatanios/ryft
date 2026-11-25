//! Naming of step directories, finalization of checkpoint directories, and deletion of checkpoints, mirroring
//! `path/step.py` and `path/atomicity.py` of Orbax and the `delete` function of its v1 API.
//!
//! The checkpoints of a training run live in one root directory that holds one directory per saved step. A
//! [`StepNameFormat`] names these directories (e.g., `42`, `00000042`, or `checkpoint_42`) and recovers the step from
//! a name, accepting only names that it would build itself, so that other entries of the root directory (e.g., the
//! root metadata directory) are never mistaken for steps.
//!
//! # Atomicity
//!
//! A checkpoint must never be read while it is incomplete, which an [`Atomicity`] mode ensures in one of two ways:
//!
//!   - [`Atomicity::Rename`] (Orbax `AtomicRenameTemporaryPath`) writes a checkpoint into a temporary sibling directory
//!     whose name is the final name followed by [`TEMPORARY_DIRECTORY_SUFFIX`] (see [`temporary_path`]), including its
//!     [`StepMetadata`] with a commit timestamp, and then renames that directory to its
//!     final name in one atomic step. It is the default on file systems that rename atomically (e.g., local disks).
//!   - [`Atomicity::CommitFile`] (Orbax `CommitFileTemporaryPath`) writes a checkpoint directly into its final
//!     directory, including step metadata without a commit timestamp, and marks its completion by writing the empty
//!     [`COMMIT_SUCCESS_FILE`] as its last step. It is the default on file systems that cannot rename atomically
//!     (e.g., object stores).
//!
//! Whether a directory is a finalized checkpoint does not depend on the mode that is configured for saving, so
//! checkpoints of both modes are always readable (see [`is_checkpoint_finalized`]). A finalized directory has no
//! temporary suffix and contains the step metadata file, and it contains the commit file or its step metadata records a
//! commit timestamp. The second alternative identifies checkpoints that were finalized by renaming, which never carry
//! a commit file. Saving in commit file mode records the commit timestamp nowhere, because a step metadata file with
//! a commit timestamp but without the commit file would otherwise be indistinguishable from a renamed checkpoint after
//! a save that was interrupted right before its commit file was written.
//!
//! Interrupted saves leave temporary directories of rename mode and step directories that are not finalized behind.
//! [`temporary_paths`] lists both kinds and [`cleanup_temporary_paths`] removes them. [`delete`] and
//! [`remove_checkpoint`] remove a checkpoint such that an interrupted removal never leaves a checkpoint behind that
//! reads as finalized.

use std::path::{Component, PathBuf};

use crate::contexts::Context;
use crate::errors::CheckpointError;
use crate::filesystems::{CheckpointPath, FileSystem};
use crate::handlers::LeafHandler;
use crate::metadata::{STEP_METADATA_FILE_NAME, StepMetadata};

// TODO(eaplatanios): Review this module.

/// Suffix that is appended to the final name of a checkpoint directory to name the temporary directory that the
/// checkpoint is written into in [`Atomicity::Rename`] mode before it is finalized.
pub const TEMPORARY_DIRECTORY_SUFFIX: &str = ".ryft-checkpoint-tmp";

/// Name of the empty file whose presence marks a checkpoint that was saved in [`Atomicity::CommitFile`] mode as
/// finalized.
pub const COMMIT_SUCCESS_FILE: &str = "commit_success.txt";

/// Way in which saving makes a checkpoint visible only once it is complete. Refer to the
/// [module documentation](self#atomicity) for both modes and what each treats as finalized.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Atomicity {
    /// Write into a temporary sibling directory and rename it to its final name, which mirrors the
    /// `AtomicRenameTemporaryPath` of Orbax.
    Rename,

    /// Write into the final directory and write the empty [`COMMIT_SUCCESS_FILE`] last, which mirrors the
    /// `CommitFileTemporaryPath` of Orbax.
    CommitFile,
}

impl Atomicity {
    /// Returns the default mode for `filesystem`, which is [`Atomicity::Rename`] when it renames atomically and
    /// [`Atomicity::CommitFile`] otherwise.
    #[inline]
    pub fn for_filesystem(filesystem: &dyn FileSystem) -> Self {
        if filesystem.supports_atomic_rename() { Self::Rename } else { Self::CommitFile }
    }
}

/// Format of the names of step directories, which is `<prefix>_<step>` when there is a prefix and `<step>` otherwise,
/// where `<step>` is the decimal step number, padded with leading zeros to the fixed length when there is one. Steps
/// whose decimal representation is longer than the fixed length are not truncated.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct StepNameFormat {
    /// Prefix of every step name, which is separated from the step number by an underscore.
    prefix: Option<String>,

    /// Number of digits that step numbers are padded to with leading zeros.
    fixed_length: Option<usize>,
}

impl StepNameFormat {
    /// Creates a new [`StepNameFormat`] without prefix and without padding, which names step 42 `42`.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns this format with the provided prefix (e.g., `checkpoint` names step 42 `checkpoint_42`).
    #[inline]
    pub fn with_prefix<P: Into<String>>(mut self, prefix: P) -> Self {
        self.prefix = Some(prefix.into());
        self
    }

    /// Returns this format with step numbers padded to `fixed_length` digits with leading zeros (e.g., `8` names step
    /// 42 `00000042`).
    #[inline]
    pub fn with_fixed_length(mut self, fixed_length: usize) -> Self {
        self.fixed_length = Some(fixed_length);
        self
    }

    /// Returns the prefix of every step name, if any.
    #[inline]
    pub fn prefix(&self) -> Option<&str> {
        self.prefix.as_deref()
    }

    /// Returns the number of digits that step numbers are padded to, if any.
    #[inline]
    pub fn fixed_length(&self) -> Option<usize> {
        self.fixed_length
    }

    /// Returns the name of the directory of `step`.
    pub fn build_name(&self, step: u64) -> String {
        let step = format!("{step:0width$}", width = self.fixed_length.unwrap_or(0));
        match &self.prefix {
            Some(prefix) => format!("{prefix}_{step}"),
            None => step,
        }
    }

    /// Returns the step whose directory is named `name`, or [`None`] if `name` is not exactly the name that
    /// [`Self::build_name`] builds for some step (e.g., because it has a different prefix, padding, or suffix).
    pub fn parse_name(&self, name: &str) -> Option<u64> {
        let digits = match &self.prefix {
            Some(prefix) => name.strip_prefix(prefix.as_str())?.strip_prefix('_')?,
            None => name,
        };
        if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
            return None;
        }
        let step = digits.parse::<u64>().ok()?;
        (self.build_name(step) == name).then_some(step)
    }

    /// Returns the step and path of every finalized checkpoint in `root`, sorted by step. Entries whose names this
    /// format does not parse, temporary directories, and directories that are not finalized are skipped. A missing
    /// root holds no checkpoints on file systems without real directories.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::Io`] if `root` cannot be listed or an entry cannot be inspected.
    pub fn find_all(&self, root: &CheckpointPath) -> Result<Vec<(u64, CheckpointPath)>, CheckpointError> {
        let mut steps = Vec::new();
        for name in root.list()? {
            if let Some(step) = self.parse_name(&name) {
                let path = root.join(&name);
                if is_checkpoint_finalized(&path)? {
                    steps.push((step, path));
                }
            }
        }
        steps.sort_by_key(|(step, _)| *step);
        Ok(steps)
    }
}

/// Returns the path of the temporary directory that the checkpoint whose final directory is `final_path` is written
/// into in [`Atomicity::Rename`] mode, which is the sibling of `final_path` whose name ends with
/// [`TEMPORARY_DIRECTORY_SUFFIX`].
pub fn temporary_path(final_path: &CheckpointPath) -> CheckpointPath {
    // Normal components only are kept, so that a trailing separator or `.` does not hide the last name.
    let mut path = final_path
        .path()
        .components()
        .filter(|component| !matches!(component, Component::CurDir))
        .collect::<PathBuf>();
    let name =
        format!("{}{TEMPORARY_DIRECTORY_SUFFIX}", path.file_name().and_then(|name| name.to_str()).unwrap_or_default());
    path.set_file_name(name);
    CheckpointPath::new(final_path.filesystem().clone(), path)
}

/// Returns `true` if the name of `path` ends with [`TEMPORARY_DIRECTORY_SUFFIX`].
pub fn is_temporary_path(path: &CheckpointPath) -> bool {
    path.file_name().is_some_and(|name| name.ends_with(TEMPORARY_DIRECTORY_SUFFIX))
}

/// Returns `true` if `path` is a finalized checkpoint, which is a directory whose name does not end with
/// [`TEMPORARY_DIRECTORY_SUFFIX`], which contains the [`STEP_METADATA_FILE_NAME`] file, and which either contains the
/// [`COMMIT_SUCCESS_FILE`] or has step metadata that records a commit timestamp. This holds for checkpoints of both
/// [`Atomicity`] modes, as the [module documentation](self#atomicity) explains. Step metadata that is not valid (e.g.,
/// because a save was interrupted while writing it) makes a directory without a commit file unfinalized.
///
/// # Errors
///
/// Returns [`CheckpointError::Io`] if the file system cannot be queried.
pub fn is_checkpoint_finalized(path: &CheckpointPath) -> Result<bool, CheckpointError> {
    if is_temporary_path(path) || !path.is_directory()? || !path.join(STEP_METADATA_FILE_NAME).exists()? {
        return Ok(false);
    }
    if path.join(COMMIT_SUCCESS_FILE).exists()? {
        return Ok(true);
    }
    match StepMetadata::read(path) {
        Ok(metadata) => Ok(metadata.commit_timestamp_nanoseconds().is_some()),
        Err(CheckpointError::InvalidFile { .. }) => Ok(false),
        Err(error) => Err(error),
    }
}

/// Removes the finalized checkpoint at `path`. The files that mark it as finalized (i.e., the [`COMMIT_SUCCESS_FILE`]
/// and the [`STEP_METADATA_FILE_NAME`] file) are removed first, so that a removal that is interrupted leaves an
/// unfinalized directory, which [`temporary_paths`] lists, and never a checkpoint that reads as finalized while some
/// of its arrays are gone.
///
/// # Errors
///
/// Returns [`CheckpointError::Io`] if the checkpoint cannot be removed.
pub fn remove_checkpoint(path: &CheckpointPath) -> Result<(), CheckpointError> {
    for marker in [COMMIT_SUCCESS_FILE, STEP_METADATA_FILE_NAME] {
        let marker = path.join(marker);
        if marker.exists()? {
            marker.remove()?;
        }
    }
    // File systems without real directories have nothing left to remove when the markers were the only files.
    if path.exists()? { path.remove() } else { Ok(()) }
}

/// Deletes the checkpoint at `path` with [`remove_checkpoint`], which mirrors `delete` of Orbax. Only the primary
/// process of `context` removes it, so that the processes that save checkpoints together can all call this function,
/// and the other processes return immediately.
///
/// # Errors
///
/// Returns [`CheckpointError::MissingCheckpoint`] if `path` does not exist, and the errors of [`remove_checkpoint`].
pub fn delete<P: Into<CheckpointPath>, L: LeafHandler>(path: P, context: &Context<L>) -> Result<(), CheckpointError> {
    if !context.multiprocessing_options().is_primary() {
        return Ok(());
    }
    let path = path.into();
    if !path.exists()? {
        return Err(CheckpointError::MissingCheckpoint { message: format!("`{path}` does not exist") });
    }
    remove_checkpoint(&path)
}

/// Returns the paths of everything in `root` that interrupted saves left behind, sorted, which are the temporary
/// directories of [`Atomicity::Rename`] mode and the directories whose names `step_name_format` parses but which are
/// not finalized checkpoints (e.g., those of [`Atomicity::CommitFile`] mode without a commit file).
///
/// # Errors
///
/// Returns [`CheckpointError::Io`] if `root` cannot be listed or an entry cannot be inspected.
pub fn temporary_paths(
    root: &CheckpointPath,
    step_name_format: &StepNameFormat,
) -> Result<Vec<CheckpointPath>, CheckpointError> {
    let mut paths = Vec::new();
    for name in root.list()? {
        let path = root.join(&name);
        let leftover = if is_temporary_path(&path) {
            path.is_directory()?
        } else {
            step_name_format.parse_name(&name).is_some() && path.is_directory()? && !is_checkpoint_finalized(&path)?
        };
        if leftover {
            paths.push(path);
        }
    }
    Ok(paths)
}

/// Removes everything in `root` that [`temporary_paths`] lists.
///
/// # Errors
///
/// Returns the errors of [`temporary_paths`], and [`CheckpointError::Io`] if a directory cannot be removed.
pub fn cleanup_temporary_paths(
    root: &CheckpointPath,
    step_name_format: &StepNameFormat,
) -> Result<(), CheckpointError> {
    for path in temporary_paths(root, step_name_format)? {
        path.remove()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::contexts::tests::test_context;
    use crate::multiprocessing::MultiprocessingOptions;

    use super::*;

    /// Creates a directory named `name` in `root` with step metadata that records a commit timestamp if `committed`
    /// is `true`, and with the commit file if `commit_file` is `true`.
    fn create_checkpoint(root: &CheckpointPath, name: &str, committed: bool, commit_file: bool) -> CheckpointPath {
        let path = root.join(name);
        path.create_directory().unwrap();
        let metadata = StepMetadata::new(Default::default(), 1);
        let metadata = if committed { metadata.with_commit_timestamp_nanoseconds(2) } else { metadata };
        metadata.write(&path).unwrap();
        if commit_file {
            path.join(COMMIT_SUCCESS_FILE).write(&[]).unwrap();
        }
        path
    }

    #[test]
    fn test_atomicity_for_filesystem() {
        assert_eq!(Atomicity::for_filesystem(&crate::filesystems::LocalFileSystem), Atomicity::Rename);
    }

    #[test]
    fn test_step_name_format_build_name() {
        assert_eq!(StepNameFormat::new().build_name(42), "42");
        assert_eq!(StepNameFormat::new().with_fixed_length(5).build_name(42), "00042");
        assert_eq!(StepNameFormat::new().with_fixed_length(2).build_name(1234), "1234");
        assert_eq!(StepNameFormat::new().with_prefix("checkpoint").build_name(0), "checkpoint_0");
        assert_eq!(StepNameFormat::new().with_prefix("step").with_fixed_length(3).build_name(7), "step_007");
        assert_eq!(StepNameFormat::new().with_prefix("step").prefix(), Some("step"));
        assert_eq!(StepNameFormat::new().with_fixed_length(3).fixed_length(), Some(3));
    }

    #[test]
    fn test_step_name_format_parse_name() {
        let plain = StepNameFormat::new();
        assert_eq!(plain.parse_name("42"), Some(42));
        assert_eq!(plain.parse_name("0"), Some(0));
        assert_eq!(plain.parse_name("042"), None);
        assert_eq!(plain.parse_name("+42"), None);
        assert_eq!(plain.parse_name(""), None);
        assert_eq!(plain.parse_name("metadata"), None);
        assert_eq!(plain.parse_name("42.ryft-checkpoint-tmp"), None);
        assert_eq!(plain.parse_name("99999999999999999999999"), None);

        let padded = StepNameFormat::new().with_fixed_length(3);
        assert_eq!(padded.parse_name("007"), Some(7));
        assert_eq!(padded.parse_name("1234"), Some(1234));
        assert_eq!(padded.parse_name("7"), None);
        assert_eq!(padded.parse_name("0007"), None);

        let prefixed = StepNameFormat::new().with_prefix("step").with_fixed_length(2);
        assert_eq!(prefixed.parse_name("step_05"), Some(5));
        assert_eq!(prefixed.parse_name("step05"), None);
        assert_eq!(prefixed.parse_name("05"), None);
        assert_eq!(prefixed.parse_name("other_05"), None);
        assert_eq!(prefixed.parse_name("step_"), None);
    }

    #[test]
    fn test_step_name_format_find_all() {
        let directory = tempfile::tempdir().unwrap();
        let root = CheckpointPath::from(directory.path());
        let format = StepNameFormat::new();
        let renamed = create_checkpoint(&root, "10", true, false);
        let committed = create_checkpoint(&root, "2", false, true);

        // Temporary, unfinalized, unparsable, and root metadata directories are skipped, as are plain files.
        create_checkpoint(&root, "3.ryft-checkpoint-tmp", true, false);
        create_checkpoint(&root, "4", false, false);
        root.join("5").create_directory().unwrap();
        create_checkpoint(&root, "007", true, true);
        create_checkpoint(&root, "metadata", true, true);
        root.join("6").write(b"").unwrap();
        assert_eq!(format.find_all(&root), Ok(vec![(2, committed), (10, renamed)]));

        let missing = root.join("missing");
        assert!(matches!(
            format.find_all(&missing),
            Err(CheckpointError::Io { path, .. }) if path == missing.to_string(),
        ));
    }

    #[test]
    fn test_temporary_path() {
        assert_eq!(
            temporary_path(&CheckpointPath::local("root/7")),
            CheckpointPath::local("root/7.ryft-checkpoint-tmp")
        );
        assert_eq!(
            temporary_path(&CheckpointPath::local("root/7/")),
            CheckpointPath::local("root/7.ryft-checkpoint-tmp")
        );
    }

    #[test]
    fn test_is_temporary_path() {
        assert!(is_temporary_path(&CheckpointPath::local("root/7.ryft-checkpoint-tmp")));
        assert!(!is_temporary_path(&CheckpointPath::local("root/7")));
        assert!(!is_temporary_path(&CheckpointPath::local("root/7.ryft-checkpoint-tmp/item")));
    }

    #[test]
    fn test_is_checkpoint_finalized() {
        let directory = tempfile::tempdir().unwrap();
        let root = CheckpointPath::from(directory.path());

        // Checkpoints finalized by renaming record a commit timestamp, and those finalized by a commit file carry it.
        assert_eq!(is_checkpoint_finalized(&create_checkpoint(&root, "1", true, false)), Ok(true));
        assert_eq!(is_checkpoint_finalized(&create_checkpoint(&root, "2", false, true)), Ok(true));
        assert_eq!(is_checkpoint_finalized(&create_checkpoint(&root, "3", true, true)), Ok(true));

        // A commit file mode save that was interrupted before its commit file, temporary directories, missing
        // directories, directories without step metadata, and invalid step metadata are not finalized.
        assert_eq!(is_checkpoint_finalized(&create_checkpoint(&root, "4", false, false)), Ok(false));
        assert_eq!(is_checkpoint_finalized(&create_checkpoint(&root, "5.ryft-checkpoint-tmp", true, true)), Ok(false));
        assert_eq!(is_checkpoint_finalized(&root.join("6")), Ok(false));
        root.join("7").create_directory().unwrap();
        assert_eq!(is_checkpoint_finalized(&root.join("7")), Ok(false));
        root.join("7").join(STEP_METADATA_FILE_NAME).write(b"{").unwrap();
        assert_eq!(is_checkpoint_finalized(&root.join("7")), Ok(false));
        root.join("7").join(COMMIT_SUCCESS_FILE).write(b"").unwrap();
        assert_eq!(is_checkpoint_finalized(&root.join("7")), Ok(true));
    }

    #[test]
    fn test_temporary_paths() {
        let directory = tempfile::tempdir().unwrap();
        let root = CheckpointPath::from(directory.path());
        let format = StepNameFormat::new();
        create_checkpoint(&root, "1", true, false);
        let renaming = create_checkpoint(&root, "1.ryft-checkpoint-tmp", true, false);
        let uncommitted = create_checkpoint(&root, "2", false, false);
        let empty = root.join("3");
        empty.create_directory().unwrap();
        root.join("4.ryft-checkpoint-tmp").write(b"").unwrap();
        create_checkpoint(&root, "metadata", false, false);
        assert_eq!(temporary_paths(&root, &format), Ok(vec![renaming, uncommitted, empty]));
    }

    #[test]
    fn test_cleanup_temporary_paths() {
        let directory = tempfile::tempdir().unwrap();
        let root = CheckpointPath::from(directory.path());
        let format = StepNameFormat::new();
        let finalized = create_checkpoint(&root, "1", false, true);
        create_checkpoint(&root, "2.ryft-checkpoint-tmp", true, false);
        create_checkpoint(&root, "3", false, false);
        assert_eq!(cleanup_temporary_paths(&root, &format), Ok(()));
        assert_eq!(temporary_paths(&root, &format), Ok(Vec::new()));
        assert_eq!(root.list(), Ok(vec!["1".to_string()]));
        assert_eq!(finalized.is_directory(), Ok(true));
    }

    #[test]
    fn test_remove_checkpoint() {
        let directory = tempfile::tempdir().unwrap();
        let root = CheckpointPath::from(directory.path());
        for (name, committed, commit_file) in [("1", true, false), ("2", false, true), ("3", true, true)] {
            let checkpoint = create_checkpoint(&root, name, committed, commit_file);
            checkpoint.join("arrays").write(b"data").unwrap();
            assert_eq!(remove_checkpoint(&checkpoint), Ok(()));
            assert_eq!(checkpoint.exists(), Ok(false));
        }

        // A directory that lost its markers already (e.g., in a removal that was interrupted) is removed as well.
        root.join("4").create_directory().unwrap();
        root.join("4").join("arrays").write(b"data").unwrap();
        assert_eq!(remove_checkpoint(&root.join("4")), Ok(()));
        assert_eq!(root.list(), Ok(Vec::new()));
    }

    #[test]
    fn test_delete() {
        let directory = tempfile::tempdir().unwrap();
        let root = CheckpointPath::from(directory.path());
        let checkpoint = create_checkpoint(&root, "1", true, false);
        checkpoint.join("state").create_directory().unwrap();

        // Processes other than the primary one leave the checkpoint to the primary process.
        let secondary = test_context().with_multiprocessing_options(MultiprocessingOptions::new(1, 2).unwrap());
        assert_eq!(delete(checkpoint.clone(), &secondary), Ok(()));
        assert_eq!(is_checkpoint_finalized(&checkpoint), Ok(true));
        assert_eq!(delete(checkpoint.clone(), &test_context()), Ok(()));
        assert_eq!(checkpoint.exists(), Ok(false));
        assert_eq!(
            delete(checkpoint.clone(), &test_context()),
            Err(CheckpointError::MissingCheckpoint { message: format!("`{checkpoint}` does not exist") }),
        );
    }
}
