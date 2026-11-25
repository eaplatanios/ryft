//! Storage that checkpoints live in, mirroring how Orbax reaches every storage system through `etils.epath.Path`.
//!
//! Every file access of saving, loading, listing, finalizing, and deleting checkpoints goes through a
//! [`FileSystem`], so that checkpoints can live on a local disk as well as in an object store. A [`CheckpointPath`]
//! pairs a path with the file system that it belongs to, and the APIs of this crate take checkpoint paths instead of
//! plain paths. Local paths convert into checkpoint paths of the shared [`LocalFileSystem`], so callers that work
//! with local directories pass [`Path`]s, [`PathBuf`]s, or strings wherever a function takes
//! `impl Into<CheckpointPath>`.
//!
//! File systems differ in two ways that checkpointing cares about:
//!
//!   - **Directories.** Local file systems have real directories. Object stores only have objects whose keys contain
//!     `/` separators, so a directory exists exactly when an object exists below it, and creating a directory does
//!     nothing.
//!   - **Atomic renames.** Local file systems rename a directory in one atomic step, which is how
//!     [`save_checkpointables`](crate::saving::save_checkpointables) finalizes checkpoints by default. Object stores
//!     rename by copying and deleting every object, so checkpoints there are finalized with a commit file instead
//!     (see [`Atomicity`](crate::paths::Atomicity)). [`FileSystem::supports_atomic_rename`] selects the default.
//!
//! Large files (e.g., the arrays of a parameter structure) are written through a [`FileWriter`], which an object store
//! uploads when it is finished, and read through a [`FileReader`], which reads byte ranges on demand instead of
//! loading the whole file.
//!
//! The crate provides [`LocalFileSystem`] and, with the `object-store` feature, `ObjectStoreFileSystem`, which stores
//! checkpoints in any store of the [`object_store`](https://docs.rs/object_store) crate (e.g., S3, GCS, or Azure Blob
//! Storage).

use std::fmt::{Debug, Display, Formatter};
use std::io::{Error as IoError, ErrorKind, Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};

use crate::errors::CheckpointError;

pub mod local;
#[cfg(feature = "object-store")]
pub mod objects;

pub use local::LocalFileSystem;
#[cfg(feature = "object-store")]
pub use objects::ObjectStoreFileSystem;

// TODO(eaplatanios): Review this module.

/// Storage system that checkpoints live in. Paths are interpreted by each implementation (e.g., relative to a key
/// prefix of an object store), and every failure is reported as a [`CheckpointError::Io`] that names the path.
pub trait FileSystem: Debug + Send + Sync {
    /// Returns `true` if [`Self::rename`] moves a directory in one atomic step, which makes renaming a temporary
    /// directory a safe way to finalize a checkpoint.
    fn supports_atomic_rename(&self) -> bool;

    /// Returns `true` if a file or a directory exists at `path`.
    fn exists(&self, path: &Path) -> Result<bool, CheckpointError>;

    /// Returns `true` if a directory exists at `path`. On file systems without real directories, a directory exists
    /// when a file exists below it.
    fn is_directory(&self, path: &Path) -> Result<bool, CheckpointError>;

    /// Returns the names of the immediate children (i.e., files and directories) of the directory `path`, sorted.
    fn list(&self, path: &Path) -> Result<Vec<String>, CheckpointError>;

    /// Creates the directory `path` and its missing ancestors, which does nothing on file systems without real
    /// directories.
    fn create_directory(&self, path: &Path) -> Result<(), CheckpointError>;

    /// Removes the file or the whole directory tree at `path`, failing if nothing exists there.
    fn remove(&self, path: &Path) -> Result<(), CheckpointError>;

    /// Moves the file or directory tree at `from` to `to`, which must not exist. Only file systems whose
    /// [`Self::supports_atomic_rename`] returns `true` do this atomically.
    fn rename(&self, from: &Path, to: &Path) -> Result<(), CheckpointError>;

    /// Reads the whole file at `path`.
    fn read(&self, path: &Path) -> Result<Vec<u8>, CheckpointError>;

    /// Reads the bytes `range` of the file at `path`, which must lie within the file.
    fn read_range(&self, path: &Path, range: Range<u64>) -> Result<Vec<u8>, CheckpointError>;

    /// Returns the size of the file at `path` in bytes.
    fn size(&self, path: &Path) -> Result<u64, CheckpointError>;

    /// Writes `bytes` as the whole contents of the file at `path`, replacing an existing file. The parent directory
    /// must exist on file systems with real directories.
    fn write(&self, path: &Path, bytes: &[u8]) -> Result<(), CheckpointError>;

    /// Returns a writer that streams the contents of a new file at `path`, replacing an existing file. The file is
    /// complete only once [`FileWriter::finish`] returns successfully.
    fn writer(&self, path: &Path) -> Result<Box<dyn FileWriter>, CheckpointError>;
}

/// Writer of one file of a [`FileSystem`], whose contents are complete only once [`Self::finish`] returns
/// successfully. Dropping a writer without finishing it may leave a partial file behind or no file at all, depending
/// on the file system.
pub trait FileWriter: Send + Write {
    /// Completes the file (e.g., flushes buffered bytes or uploads an object).
    fn finish(self: Box<Self>) -> Result<(), CheckpointError>;
}

/// Path of a file or directory together with the [`FileSystem`] that it belongs to, which mirrors the
/// `etils.epath.Path` that Orbax passes around. Cloning is cheap, because the file system is shared. Two checkpoint
/// paths are equal when they belong to the same file system instance and have equal paths.
#[derive(Clone)]
pub struct CheckpointPath {
    /// File system that the path belongs to.
    filesystem: Arc<dyn FileSystem>,

    /// Path within the file system.
    path: PathBuf,
}

impl CheckpointPath {
    /// Creates a new [`CheckpointPath`] for `path` within `filesystem`.
    #[inline]
    pub fn new<P: Into<PathBuf>>(filesystem: Arc<dyn FileSystem>, path: P) -> Self {
        Self { filesystem, path: path.into() }
    }

    /// Creates a new [`CheckpointPath`] for `path` on the local file system, which all local checkpoint paths share.
    #[inline]
    pub fn local<P: Into<PathBuf>>(path: P) -> Self {
        Self::new(LOCAL_FILE_SYSTEM.clone(), path)
    }

    /// Returns the path within the file system.
    #[inline]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the file system that the path belongs to.
    #[inline]
    pub fn filesystem(&self) -> &Arc<dyn FileSystem> {
        &self.filesystem
    }

    /// Returns the path of `name` within this directory, on the same file system.
    #[inline]
    pub fn join<P: AsRef<Path>>(&self, name: P) -> Self {
        Self { filesystem: self.filesystem.clone(), path: self.path.join(name) }
    }

    /// Returns the last component of this path, or [`None`] if it has none or if it is not valid Unicode.
    #[inline]
    pub fn file_name(&self) -> Option<&str> {
        self.path.file_name().and_then(|name| name.to_str())
    }

    /// Returns the parent of this path, on the same file system, or [`None`] if it has none.
    #[inline]
    pub fn parent(&self) -> Option<Self> {
        self.path
            .parent()
            .map(|parent| Self { filesystem: self.filesystem.clone(), path: parent.to_path_buf() })
    }

    /// Returns `true` if a file or a directory exists at this path (refer to [`FileSystem::exists`]).
    #[inline]
    pub fn exists(&self) -> Result<bool, CheckpointError> {
        self.filesystem.exists(&self.path)
    }

    /// Returns `true` if a directory exists at this path (refer to [`FileSystem::is_directory`]).
    #[inline]
    pub fn is_directory(&self) -> Result<bool, CheckpointError> {
        self.filesystem.is_directory(&self.path)
    }

    /// Returns the sorted names of the immediate children of this directory (refer to [`FileSystem::list`]).
    #[inline]
    pub fn list(&self) -> Result<Vec<String>, CheckpointError> {
        self.filesystem.list(&self.path)
    }

    /// Creates this directory and its missing ancestors (refer to [`FileSystem::create_directory`]).
    #[inline]
    pub fn create_directory(&self) -> Result<(), CheckpointError> {
        self.filesystem.create_directory(&self.path)
    }

    /// Removes the file or the whole directory tree at this path (refer to [`FileSystem::remove`]).
    #[inline]
    pub fn remove(&self) -> Result<(), CheckpointError> {
        self.filesystem.remove(&self.path)
    }

    /// Moves the file or directory tree at this path to `to` (refer to [`FileSystem::rename`]).
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::InvalidArgument`] if `to` belongs to another file system, and the errors of
    /// [`FileSystem::rename`].
    pub fn rename(&self, to: &CheckpointPath) -> Result<(), CheckpointError> {
        if !self.has_same_filesystem(to) {
            return Err(CheckpointError::InvalidArgument {
                message: format!("cannot rename `{self}` to `{to}` on another file system"),
            });
        }
        self.filesystem.rename(&self.path, &to.path)
    }

    /// Reads the whole file at this path (refer to [`FileSystem::read`]).
    #[inline]
    pub fn read(&self) -> Result<Vec<u8>, CheckpointError> {
        self.filesystem.read(&self.path)
    }

    /// Reads the bytes `range` of the file at this path (refer to [`FileSystem::read_range`]).
    #[inline]
    pub fn read_range(&self, range: Range<u64>) -> Result<Vec<u8>, CheckpointError> {
        self.filesystem.read_range(&self.path, range)
    }

    /// Returns the size of the file at this path in bytes (refer to [`FileSystem::size`]).
    #[inline]
    pub fn size(&self) -> Result<u64, CheckpointError> {
        self.filesystem.size(&self.path)
    }

    /// Writes `bytes` as the whole contents of the file at this path (refer to [`FileSystem::write`]).
    #[inline]
    pub fn write(&self, bytes: &[u8]) -> Result<(), CheckpointError> {
        self.filesystem.write(&self.path, bytes)
    }

    /// Returns a writer that streams the contents of a new file at this path (refer to [`FileSystem::writer`]).
    #[inline]
    pub fn writer(&self) -> Result<Box<dyn FileWriter>, CheckpointError> {
        self.filesystem.writer(&self.path)
    }

    /// Returns `true` if `other` belongs to the same file system instance as this path.
    fn has_same_filesystem(&self, other: &CheckpointPath) -> bool {
        std::ptr::addr_eq(Arc::as_ptr(&self.filesystem), Arc::as_ptr(&other.filesystem))
    }
}

impl PartialEq for CheckpointPath {
    fn eq(&self, other: &Self) -> bool {
        self.has_same_filesystem(other) && self.path == other.path
    }
}

impl Eq for CheckpointPath {}

impl Debug for CheckpointPath {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CheckpointPath")
            .field("filesystem", &self.filesystem)
            .field("path", &self.path)
            .finish()
    }
}

impl Display for CheckpointPath {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.path.display())
    }
}

impl From<&Path> for CheckpointPath {
    fn from(path: &Path) -> Self {
        Self::local(path)
    }
}

impl From<PathBuf> for CheckpointPath {
    fn from(path: PathBuf) -> Self {
        Self::local(path)
    }
}

impl From<&PathBuf> for CheckpointPath {
    fn from(path: &PathBuf) -> Self {
        Self::local(path.as_path())
    }
}

impl From<&str> for CheckpointPath {
    fn from(path: &str) -> Self {
        Self::local(path)
    }
}

impl From<&CheckpointPath> for CheckpointPath {
    fn from(path: &CheckpointPath) -> Self {
        path.clone()
    }
}

/// Reader of one file of a [`FileSystem`] that reads byte ranges on demand, so that seeking within a large file and
/// reading parts of it (e.g., one array of a safetensors file) never loads the whole file.
#[derive(Clone, Debug)]
pub struct FileReader {
    /// Path of the file.
    path: CheckpointPath,

    /// Size of the file in bytes.
    size: u64,

    /// Offset of the next byte to read.
    position: u64,
}

impl FileReader {
    /// Opens the file at `path` for reading, starting at its first byte.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`FileSystem::size`].
    pub fn open(path: CheckpointPath) -> Result<Self, CheckpointError> {
        let size = path.size()?;
        Ok(Self { path, size, position: 0 })
    }

    /// Returns the path of the file.
    #[inline]
    pub fn path(&self) -> &CheckpointPath {
        &self.path
    }

    /// Returns the size of the file in bytes.
    #[inline]
    pub fn size(&self) -> u64 {
        self.size
    }
}

impl Read for FileReader {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        // Every call reads at most the requested number of bytes with one range read, so `read_exact` of a whole
        // array issues a single request.
        if buffer.is_empty() || self.position >= self.size {
            return Ok(0);
        }
        let end = self.size.min(self.position + buffer.len() as u64);
        let bytes = self.path.read_range(self.position..end).map_err(IoError::other)?;
        buffer[..bytes.len()].copy_from_slice(&bytes);
        self.position += bytes.len() as u64;
        Ok(bytes.len())
    }
}

impl Seek for FileReader {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        let position = match position {
            SeekFrom::Start(offset) => Some(offset),
            SeekFrom::End(offset) => self.size.checked_add_signed(offset),
            SeekFrom::Current(offset) => self.position.checked_add_signed(offset),
        };
        self.position = position.ok_or_else(|| {
            IoError::new(ErrorKind::InvalidInput, "cannot seek before the start of the file or past `u64::MAX`")
        })?;
        Ok(self.position)
    }
}

/// Local file system that every local [`CheckpointPath`] shares, so that local paths compare equal across
/// conversions.
static LOCAL_FILE_SYSTEM: LazyLock<Arc<dyn FileSystem>> = LazyLock::new(|| Arc::new(LocalFileSystem));

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_checkpoint_path_new() {
        let filesystem: Arc<dyn FileSystem> = Arc::new(LocalFileSystem);
        let path = CheckpointPath::new(filesystem.clone(), "root/1");
        assert_eq!(path.path(), Path::new("root/1"));
        assert!(Arc::ptr_eq(path.filesystem(), &filesystem));

        // Paths of distinct file system instances differ even when their paths are equal, while local paths share
        // one instance.
        assert_ne!(path, CheckpointPath::local("root/1"));
        assert_eq!(CheckpointPath::from("root/1"), CheckpointPath::local(PathBuf::from("root/1")));
        assert_eq!(CheckpointPath::from(Path::new("root/1")), CheckpointPath::from(&PathBuf::from("root/1")));
        assert_eq!(CheckpointPath::from(&path), path);
        assert_eq!(path.to_string(), "root/1");
    }

    #[test]
    fn test_checkpoint_path_join() {
        let path = CheckpointPath::local("root").join("1").join("item");
        assert_eq!(path, CheckpointPath::local("root/1/item"));
        assert_eq!(path.file_name(), Some("item"));
        assert_eq!(path.parent(), Some(CheckpointPath::local("root/1")));
        assert_eq!(CheckpointPath::local("").parent(), None);
    }

    #[test]
    fn test_checkpoint_path_rename() {
        let directory = tempfile::tempdir().unwrap();
        let from = CheckpointPath::from(directory.path()).join("a");
        let to = CheckpointPath::from(directory.path()).join("b");
        from.write(b"contents").unwrap();
        assert_eq!(from.rename(&to), Ok(()));
        assert_eq!(to.read(), Ok(b"contents".to_vec()));

        // Renaming across file system instances is rejected.
        let other = CheckpointPath::new(Arc::new(LocalFileSystem), directory.path().join("c"));
        assert_eq!(
            to.rename(&other),
            Err(CheckpointError::InvalidArgument {
                message: format!("cannot rename `{to}` to `{other}` on another file system"),
            }),
        );
    }

    #[test]
    fn test_file_reader() {
        let directory = tempfile::tempdir().unwrap();
        let path = CheckpointPath::from(directory.path()).join("file");
        path.write(b"0123456789").unwrap();
        let mut reader = FileReader::open(path.clone()).unwrap();
        assert_eq!(reader.path(), &path);
        assert_eq!(reader.size(), 10);

        let mut buffer = [0u8; 4];
        assert_eq!(reader.seek(SeekFrom::Start(3)).unwrap(), 3);
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!(&buffer, b"3456");
        assert_eq!(reader.seek(SeekFrom::End(-2)).unwrap(), 8);
        let mut rest = Vec::new();
        reader.read_to_end(&mut rest).unwrap();
        assert_eq!(rest, b"89".to_vec());
        assert_eq!(reader.seek(SeekFrom::Current(-10)).unwrap(), 0);
        assert!(reader.seek(SeekFrom::Current(-1)).is_err());
    }
}
