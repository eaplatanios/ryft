//! Object stores, which store checkpoints through the [`object_store`] crate and are available with the
//! `object-store` feature.
//!
//! [`ObjectStoreFileSystem`] adapts any [`ObjectStore`] (e.g., Amazon S3, Google Cloud Storage, or Azure Blob Storage,
//! which downstream crates enable through the `aws`, `gcp`, and `azure` features of `object_store`) to the
//! [`FileSystem`] contract of this crate. Every path becomes an object key below an optional key prefix, with one key
//! segment per path component. Object stores have no directories, so a directory exists when an object exists below
//! it, creating a directory does nothing, and removing or renaming a directory removes or copies every object below
//! it. Renames are therefore not atomic, and checkpoints in object stores are finalized with commit files (see
//! [`Atomicity::CommitFile`](crate::paths::Atomicity::CommitFile)).
//!
//! The `object_store` API is asynchronous, while the API of this crate is synchronous. Every file system owns a
//! private single threaded [Tokio](https://tokio.rs) runtime that it runs each operation on to completion. A call made
//! from within the asynchronous context of another Tokio runtime (where blocking that runtime would panic) runs the
//! operation on a scoped helper thread instead, so the file system can be used from anywhere, but callers inside an
//! asynchronous context should still prefer to move checkpointing to a blocking task, because the call blocks its
//! thread until the operation completes.

use std::fmt::{Debug, Formatter};
use std::future::Future;
use std::io::Write;
use std::ops::Range;
use std::path::{Component, Path};
use std::sync::Arc;

use object_store::path::Path as ObjectPath;
use object_store::{Error as ObjectStoreError, ObjectStore, ObjectStoreExt, PutPayload};
use tokio::runtime::{Builder, Handle, Runtime};

use crate::errors::CheckpointError;
use crate::filesystems::{FileSystem, FileWriter};

// TODO(eaplatanios): Review this module.

/// [`FileSystem`] over an [`ObjectStore`], which stores every file as one object. Refer to the
/// [module documentation](self) for how paths map to keys, what is not atomic, and how the asynchronous object store
/// is driven.
pub struct ObjectStoreFileSystem {
    /// Object store that holds the objects.
    store: Arc<dyn ObjectStore>,

    /// Key prefix of every object, which is the root key when there is no prefix.
    prefix: ObjectPath,

    /// Runtime that drives the asynchronous operations of the object store.
    runtime: Arc<Runtime>,
}

impl ObjectStoreFileSystem {
    /// Creates a new [`ObjectStoreFileSystem`] over `store` without a key prefix.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::Io`] if the runtime that drives the object store cannot be created.
    pub fn new(store: Arc<dyn ObjectStore>) -> Result<Self, CheckpointError> {
        let runtime = Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| CheckpointError::Io { path: store.to_string(), message: error.to_string() })?;
        Ok(Self { store, prefix: ObjectPath::ROOT, runtime: Arc::new(runtime) })
    }

    /// Returns this file system with every key below `prefix` (e.g., `team/checkpoints`).
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::InvalidArgument`] if `prefix` is not a valid object key (e.g., because it has
    /// empty segments).
    pub fn with_prefix(mut self, prefix: &str) -> Result<Self, CheckpointError> {
        self.prefix = ObjectPath::parse(prefix).map_err(|error| CheckpointError::InvalidArgument {
            message: format!("`{prefix}` is not a valid object key prefix: {error}"),
        })?;
        Ok(self)
    }

    /// Returns the object store that holds the objects.
    #[inline]
    pub fn store(&self) -> &Arc<dyn ObjectStore> {
        &self.store
    }

    /// Returns the key prefix of every object, which is empty when there is no prefix.
    #[inline]
    pub fn prefix(&self) -> &str {
        self.prefix.as_ref()
    }

    /// Returns the key of the object at `path`, which appends one key segment per normal path component to the
    /// prefix. Root and current directory components are ignored.
    fn key(&self, path: &Path) -> Result<ObjectPath, CheckpointError> {
        let mut key = self.prefix.clone();
        for component in path.components() {
            match component {
                Component::Normal(name) => {
                    let name = name.to_str().ok_or_else(|| self.error(path, "path is not valid Unicode"))?;
                    key = key.join(name);
                }
                Component::RootDir | Component::CurDir => {}
                Component::Prefix(_) | Component::ParentDir => {
                    return Err(self.error(path, "path has a component that object keys cannot represent"));
                }
            }
        }
        Ok(key)
    }

    /// Returns the keys of every object strictly below the directory key `key`, found by walking the listing of every
    /// directory level.
    fn keys_below(&self, key: &ObjectPath) -> Result<Vec<ObjectPath>, ObjectStoreError> {
        let mut keys = Vec::new();
        let mut pending = vec![key.clone()];
        while let Some(directory) = pending.pop() {
            let prefix = (!directory.is_root()).then_some(&directory);
            let listing = self.block_on(self.store.list_with_delimiter(prefix))?;
            keys.extend(listing.objects.into_iter().map(|object| object.location));
            pending.extend(listing.common_prefixes);
        }
        Ok(keys)
    }

    /// Runs `future` to completion on the runtime of this file system. Within the asynchronous context of another
    /// Tokio runtime, the runtime runs on a scoped helper thread, because blocking that context would panic.
    fn block_on<F: Send + Future<Output: Send>>(&self, future: F) -> F::Output {
        if Handle::try_current().is_ok() {
            std::thread::scope(|scope| {
                scope.spawn(|| self.runtime.block_on(future)).join().expect("an object store operation panicked")
            })
        } else {
            self.runtime.block_on(future)
        }
    }

    /// Returns the [`CheckpointError::Io`] for a failed access of `path` with the provided message.
    fn error<M: ToString>(&self, path: &Path, message: M) -> CheckpointError {
        CheckpointError::Io { path: path.display().to_string(), message: message.to_string() }
    }
}

impl Debug for ObjectStoreFileSystem {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ObjectStoreFileSystem")
            .field("store", &self.store)
            .field("prefix", &self.prefix)
            .finish()
    }
}

impl FileSystem for ObjectStoreFileSystem {
    #[inline]
    fn supports_atomic_rename(&self) -> bool {
        false
    }

    fn exists(&self, path: &Path) -> Result<bool, CheckpointError> {
        let key = self.key(path)?;
        if key.is_root() {
            return Ok(true);
        }
        match self.block_on(self.store.head(&key)) {
            Ok(_) => Ok(true),
            Err(ObjectStoreError::NotFound { .. }) => self.is_directory(path),
            Err(error) => Err(self.error(path, error)),
        }
    }

    fn is_directory(&self, path: &Path) -> Result<bool, CheckpointError> {
        let key = self.key(path)?;
        if key.is_root() {
            return Ok(true);
        }
        let listing =
            self.block_on(self.store.list_with_delimiter(Some(&key))).map_err(|error| self.error(path, error))?;
        Ok(!listing.objects.is_empty() || !listing.common_prefixes.is_empty())
    }

    fn list(&self, path: &Path) -> Result<Vec<String>, CheckpointError> {
        let key = self.key(path)?;
        let prefix = (!key.is_root()).then_some(&key);
        let listing = self.block_on(self.store.list_with_delimiter(prefix)).map_err(|error| self.error(path, error))?;
        let mut names = listing
            .common_prefixes
            .iter()
            .chain(listing.objects.iter().map(|object| &object.location))
            .filter_map(|location| location.filename().map(str::to_string))
            .collect::<Vec<_>>();
        names.sort();
        names.dedup();
        Ok(names)
    }

    fn create_directory(&self, path: &Path) -> Result<(), CheckpointError> {
        // Directories are implied by the keys of the objects below them, but the path must still map to a key.
        self.key(path).map(|_| ())
    }

    fn remove(&self, path: &Path) -> Result<(), CheckpointError> {
        let key = self.key(path)?;
        let mut keys = self.keys_below(&key).map_err(|error| self.error(path, error))?;
        if !key.is_root() {
            match self.block_on(self.store.head(&key)) {
                Ok(_) => keys.push(key),
                Err(ObjectStoreError::NotFound { .. }) => {}
                Err(error) => return Err(self.error(path, error)),
            }
        }
        if keys.is_empty() {
            return Err(self.error(path, "no such file or directory"));
        }
        for key in keys {
            self.block_on(self.store.delete(&key)).map_err(|error| self.error(path, error))?;
        }
        Ok(())
    }

    fn rename(&self, from: &Path, to: &Path) -> Result<(), CheckpointError> {
        // Every object is copied before any object is deleted, so an interrupted rename leaves the source complete.
        let from_key = self.key(from)?;
        let to_key = self.key(to)?;
        if self.exists(to)? {
            return Err(self.error(to, "the destination already exists"));
        }
        let mut moves = Vec::new();
        match self.block_on(self.store.head(&from_key)) {
            Ok(_) => moves.push((from_key.clone(), to_key.clone())),
            Err(ObjectStoreError::NotFound { .. }) => {}
            Err(error) => return Err(self.error(from, error)),
        }
        for key in self.keys_below(&from_key).map_err(|error| self.error(from, error))? {
            let suffix = key.prefix_match(&from_key).unwrap().collect::<Vec<_>>();
            moves.push((key.clone(), suffix.into_iter().fold(to_key.clone(), |key, part| key.join(part))));
        }
        if moves.is_empty() {
            return Err(self.error(from, "no such file or directory"));
        }
        for (source, destination) in &moves {
            self.block_on(self.store.copy(source, destination)).map_err(|error| self.error(to, error))?;
        }
        for (source, _) in &moves {
            self.block_on(self.store.delete(source)).map_err(|error| self.error(from, error))?;
        }
        Ok(())
    }

    fn read(&self, path: &Path) -> Result<Vec<u8>, CheckpointError> {
        let key = self.key(path)?;
        let bytes = self
            .block_on(async { self.store.get(&key).await?.bytes().await })
            .map_err(|error| self.error(path, error))?;
        Ok(bytes.to_vec())
    }

    fn read_range(&self, path: &Path, range: Range<u64>) -> Result<Vec<u8>, CheckpointError> {
        let key = self.key(path)?;
        if range.start >= range.end {
            return Ok(Vec::new());
        }
        let bytes = self.block_on(self.store.get_range(&key, range)).map_err(|error| self.error(path, error))?;
        Ok(bytes.to_vec())
    }

    fn size(&self, path: &Path) -> Result<u64, CheckpointError> {
        let key = self.key(path)?;
        Ok(self.block_on(self.store.head(&key)).map_err(|error| self.error(path, error))?.size)
    }

    fn write(&self, path: &Path, bytes: &[u8]) -> Result<(), CheckpointError> {
        let key = self.key(path)?;
        self.block_on(self.store.put(&key, PutPayload::from(bytes.to_vec())))
            .map_err(|error| self.error(path, error))?;
        Ok(())
    }

    fn writer(&self, path: &Path) -> Result<Box<dyn FileWriter>, CheckpointError> {
        let key = self.key(path)?;
        Ok(Box::new(ObjectWriter {
            filesystem: ObjectStoreFileSystem {
                store: self.store.clone(),
                prefix: self.prefix.clone(),
                runtime: self.runtime.clone(),
            },
            path: path.to_path_buf(),
            key,
            bytes: Vec::new(),
        }))
    }
}

/// [`FileWriter`] of one object, which buffers the whole object in memory and uploads it with one request when it is
/// finished. Streaming multipart uploads are not used yet, so writing a file needs as much host memory as the file
/// is large.
struct ObjectWriter {
    /// File system that uploads the object.
    filesystem: ObjectStoreFileSystem,

    /// Path of the file, which is used in diagnostics.
    path: std::path::PathBuf,

    /// Key of the object.
    key: ObjectPath,

    /// Contents of the object written so far.
    bytes: Vec<u8>,
}

impl Write for ObjectWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl FileWriter for ObjectWriter {
    fn finish(self: Box<Self>) -> Result<(), CheckpointError> {
        let Self { filesystem, path, key, bytes } = *self;
        filesystem
            .block_on(filesystem.store.put(&key, PutPayload::from(bytes)))
            .map_err(|error| filesystem.error(&path, error))?;
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use object_store::memory::InMemory;
    use pretty_assertions::assert_eq;

    use crate::filesystems::CheckpointPath;

    use super::*;

    /// Returns the root of an empty in-memory object store whose keys are below the prefix `checkpoints`.
    pub(crate) fn test_object_store_root() -> CheckpointPath {
        let filesystem =
            ObjectStoreFileSystem::new(Arc::new(InMemory::new())).unwrap().with_prefix("checkpoints").unwrap();
        CheckpointPath::new(Arc::new(filesystem), "run")
    }

    #[test]
    fn test_object_store_file_system_new() {
        let filesystem = ObjectStoreFileSystem::new(Arc::new(InMemory::new())).unwrap();
        assert_eq!(filesystem.prefix(), "");
        assert!(!filesystem.supports_atomic_rename());
        let filesystem = filesystem.with_prefix("a/b").unwrap();
        assert_eq!(filesystem.prefix(), "a/b");
        assert_eq!(filesystem.store().to_string(), "InMemory");
        assert!(matches!(
            ObjectStoreFileSystem::new(Arc::new(InMemory::new())).unwrap().with_prefix("a//b"),
            Err(CheckpointError::InvalidArgument { .. }),
        ));
    }

    #[test]
    fn test_object_store_file_system() {
        let root = test_object_store_root();

        // Directories exist exactly when objects exist below them, and creating them does nothing.
        assert_eq!(root.join("1").create_directory(), Ok(()));
        assert_eq!(root.join("1").exists(), Ok(false));
        assert_eq!(root.join("1").join("item").join("file").write(b"0123"), Ok(()));
        assert_eq!(root.join("1").join("other").write(b"x"), Ok(()));
        assert_eq!(root.join("1").exists(), Ok(true));
        assert_eq!(root.join("1").is_directory(), Ok(true));
        assert_eq!(root.join("1").join("other").is_directory(), Ok(false));
        assert_eq!(root.join("1").join("other").exists(), Ok(true));
        assert_eq!(root.join("1").list(), Ok(vec!["item".to_string(), "other".to_string()]));
        assert_eq!(root.list(), Ok(vec!["1".to_string()]));

        // Files are read whole or by byte range, and streamed writes are uploaded once they are finished.
        let file = root.join("1").join("item").join("file");
        assert_eq!(file.read(), Ok(b"0123".to_vec()));
        assert_eq!(file.read_range(1..3), Ok(b"12".to_vec()));
        assert_eq!(file.size(), Ok(4));
        let mut writer = root.join("streamed").writer().unwrap();
        writer.write_all(b"abc").unwrap();
        assert_eq!(root.join("streamed").exists(), Ok(false));
        assert_eq!(writer.finish(), Ok(()));
        assert_eq!(root.join("streamed").read(), Ok(b"abc".to_vec()));

        // Renames copy whole trees, and removal deletes files and trees but fails for missing paths.
        assert_eq!(root.join("1").rename(&root.join("2")), Ok(()));
        assert_eq!(root.list(), Ok(vec!["2".to_string(), "streamed".to_string()]));
        assert_eq!(root.join("2").join("item").join("file").read(), Ok(b"0123".to_vec()));
        assert_eq!(root.join("2").remove(), Ok(()));
        assert_eq!(root.join("streamed").remove(), Ok(()));
        assert_eq!(root.list(), Ok(Vec::new()));
        assert_eq!(
            root.join("2").remove(),
            Err(CheckpointError::Io { path: "run/2".to_string(), message: "no such file or directory".to_string() }),
        );
        assert!(matches!(root.join("..").read(), Err(CheckpointError::Io { .. })));
    }

    #[test]
    fn test_object_store_file_system_within_runtime() {
        // Operations work from within the asynchronous context of another runtime.
        let root = test_object_store_root();
        let runtime = Builder::new_current_thread().build().unwrap();
        runtime.block_on(async {
            root.join("file").write(b"inside").unwrap();
            assert_eq!(root.join("file").read(), Ok(b"inside".to_vec()));
        });
    }
}
