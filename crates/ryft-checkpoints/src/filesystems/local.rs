//! Local file system, which stores checkpoints with [`std::fs`].
//!
//! Every file is synchronized to durable storage before its write returns, so the files of a checkpoint are on disk
//! before the checkpoint is finalized and a machine that loses power right afterward cannot leave a finalized
//! checkpoint whose files are empty or truncated. This is what Orbax gets for its arrays from the file driver of
//! TensorStore, whose `file_io_sync` option is enabled by default. Directories are not synchronized, as in Orbax, so
//! the rename that finalizes a checkpoint can itself be lost with the power, which leaves the complete temporary
//! directory behind instead of the checkpoint.
//!
//! On Apple platforms files are synchronized with a plain `fsync`, as TensorStore does, and not with the
//! `F_FULLFSYNC` request that [`File::sync_all`] makes there. A plain `fsync` hands the data to the drive without
//! waiting for the drive to empty its own cache, so it protects against a crash of the process or of the operating
//! system but not fully against a loss of power, and it is far cheaper for formats that write many files. On other
//! platforms `fsync` waits for the drive already, and [`File::sync_all`] is used.

use std::fs::{self, File};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::ops::Range;
#[cfg(target_vendor = "apple")]
use std::os::fd::AsRawFd;
use std::path::Path;

use crate::errors::CheckpointError;
use crate::filesystems::{FileSystem, FileWriter};

// TODO(eaplatanios): Review this module.

/// [`FileSystem`] of the local machine, which has real directories, renames them atomically, and synchronizes every
/// file that it writes to durable storage (refer to the [module documentation](self)).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct LocalFileSystem;

impl FileSystem for LocalFileSystem {
    #[inline]
    fn supports_atomic_rename(&self) -> bool {
        true
    }

    fn exists(&self, path: &Path) -> Result<bool, CheckpointError> {
        path.try_exists().map_err(|error| CheckpointError::from_io_error(path, error))
    }

    fn is_directory(&self, path: &Path) -> Result<bool, CheckpointError> {
        match fs::metadata(path) {
            Ok(metadata) => Ok(metadata.is_dir()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(CheckpointError::from_io_error(path, error)),
        }
    }

    fn list(&self, path: &Path) -> Result<Vec<String>, CheckpointError> {
        let mut names = Vec::new();
        for entry in fs::read_dir(path).map_err(|error| CheckpointError::from_io_error(path, error))? {
            let entry = entry.map_err(|error| CheckpointError::from_io_error(path, error))?;
            // Names that are not valid Unicode cannot be checkpoint names, so they are skipped.
            if let Some(name) = entry.file_name().to_str() {
                names.push(name.to_string());
            }
        }
        names.sort();
        Ok(names)
    }

    fn create_directory(&self, path: &Path) -> Result<(), CheckpointError> {
        fs::create_dir_all(path).map_err(|error| CheckpointError::from_io_error(path, error))
    }

    fn remove(&self, path: &Path) -> Result<(), CheckpointError> {
        let result = if self.is_directory(path)? { fs::remove_dir_all(path) } else { fs::remove_file(path) };
        result.map_err(|error| CheckpointError::from_io_error(path, error))
    }

    fn rename(&self, from: &Path, to: &Path) -> Result<(), CheckpointError> {
        fs::rename(from, to).map_err(|error| CheckpointError::from_io_error(to, error))
    }

    fn read(&self, path: &Path) -> Result<Vec<u8>, CheckpointError> {
        fs::read(path).map_err(|error| CheckpointError::from_io_error(path, error))
    }

    fn read_range(&self, path: &Path, range: Range<u64>) -> Result<Vec<u8>, CheckpointError> {
        let io_error = |error| CheckpointError::from_io_error(path, error);
        let mut file = File::open(path).map_err(io_error)?;
        file.seek(SeekFrom::Start(range.start)).map_err(io_error)?;
        let mut bytes = vec![0u8; range.end.saturating_sub(range.start) as usize];
        file.read_exact(&mut bytes).map_err(io_error)?;
        Ok(bytes)
    }

    fn size(&self, path: &Path) -> Result<u64, CheckpointError> {
        Ok(fs::metadata(path).map_err(|error| CheckpointError::from_io_error(path, error))?.len())
    }

    fn write(&self, path: &Path, bytes: &[u8]) -> Result<(), CheckpointError> {
        let io_error = |error| CheckpointError::from_io_error(path, error);
        let mut file = File::create(path).map_err(io_error)?;
        file.write_all(bytes).map_err(io_error)?;
        synchronize(&file).map_err(io_error)
    }

    fn writer(&self, path: &Path) -> Result<Box<dyn FileWriter>, CheckpointError> {
        let file = File::create(path).map_err(|error| CheckpointError::from_io_error(path, error))?;
        Ok(Box::new(LocalFileWriter { path: path.display().to_string(), writer: BufWriter::new(file) }))
    }
}

/// [`FileWriter`] of a local file, which buffers writes, and flushes and synchronizes them to durable storage when
/// it is finished.
struct LocalFileWriter {
    /// Path of the file, which is used in diagnostics.
    path: String,

    /// Buffered writer of the file.
    writer: BufWriter<File>,
}

impl Write for LocalFileWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.writer.write(bytes)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

impl FileWriter for LocalFileWriter {
    fn finish(mut self: Box<Self>) -> Result<(), CheckpointError> {
        self.writer
            .flush()
            .and_then(|()| synchronize(self.writer.get_ref()))
            .map_err(|error| CheckpointError::Io { path: self.path.clone(), message: error.to_string() })
    }
}

/// Synchronizes the contents and the metadata of `file` to durable storage, as the
/// [module documentation](self) describes.
#[cfg(not(target_vendor = "apple"))]
fn synchronize(file: &File) -> std::io::Result<()> {
    file.sync_all()
}

/// Synchronizes the contents and the metadata of `file` to durable storage with a plain `fsync`, as the
/// [module documentation](self) describes.
#[cfg(target_vendor = "apple")]
fn synchronize(file: &File) -> std::io::Result<()> {
    loop {
        // SAFETY: `file` is open for the duration of the call, so its descriptor is valid, and `fsync` neither
        // retains the descriptor nor accesses memory of this process.
        if unsafe { libc::fsync(file.as_raw_fd()) } == 0 {
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_local_file_system() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let filesystem = LocalFileSystem;
        assert!(filesystem.supports_atomic_rename());

        // Directories are real and created recursively, and files are listed with directories by name.
        let nested = root.join("a").join("b");
        assert_eq!(filesystem.create_directory(&nested), Ok(()));
        assert_eq!(filesystem.is_directory(&nested), Ok(true));
        assert_eq!(filesystem.write(&root.join("a").join("file"), b"0123"), Ok(()));
        assert_eq!(filesystem.list(&root.join("a")), Ok(vec!["b".to_string(), "file".to_string()]));
        assert_eq!(filesystem.exists(&root.join("a").join("file")), Ok(true));
        assert_eq!(filesystem.is_directory(&root.join("a").join("file")), Ok(false));
        assert_eq!(filesystem.exists(&root.join("missing")), Ok(false));
        assert_eq!(filesystem.is_directory(&root.join("missing")), Ok(false));

        // Files are read whole or by byte range, and streamed writes are complete once they are finished.
        assert_eq!(filesystem.read(&root.join("a").join("file")), Ok(b"0123".to_vec()));
        assert_eq!(filesystem.read_range(&root.join("a").join("file"), 1..3), Ok(b"12".to_vec()));
        assert_eq!(filesystem.size(&root.join("a").join("file")), Ok(4));
        let mut writer = filesystem.writer(&root.join("streamed")).unwrap();
        writer.write_all(b"abc").unwrap();
        assert_eq!(writer.finish(), Ok(()));
        assert_eq!(filesystem.read(&root.join("streamed")), Ok(b"abc".to_vec()));

        // Renames move whole trees, and removal deletes files and trees but fails for missing paths.
        assert_eq!(filesystem.rename(&root.join("a"), &root.join("c")), Ok(()));
        assert_eq!(filesystem.list(root), Ok(vec!["c".to_string(), "streamed".to_string()]));
        assert_eq!(filesystem.remove(&root.join("streamed")), Ok(()));
        assert_eq!(filesystem.remove(&root.join("c")), Ok(()));
        assert_eq!(filesystem.list(root), Ok(Vec::new()));
        assert!(matches!(
            filesystem.remove(&root.join("c")),
            Err(CheckpointError::Io { path, .. }) if path == root.join("c").display().to_string(),
        ));
    }
}
