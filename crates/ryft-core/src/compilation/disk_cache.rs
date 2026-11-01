//! Disk-backed persistent compilation cache.
//!
//! [`DiskCache`] persists serialized compiled programs across processes, similar to JAX's persistent compilation
//! cache. When attached to a [`CompilationContext`](super::CompilationContext), it serves as a second tier below the
//! in-memory LRU cache. On a memory miss, the context first checks the disk for a serialized executable that matches
//! the persistent cache key and deserializes it if found. Otherwise, it compiles the program and writes the serialized
//! executable back to disk.
//!
//! Entries are stored gzip-compressed as `<directory>/<sha256>.executable`. A backend opts into this tier by providing
//! stable canonical bytes for the complete persistent cache key, and the filename is the SHA-256 digest of those
//! bytes. The compressed entry is a versioned envelope that repeats the key digest and records the payload length and
//! SHA-256 checksum, all of which are validated before backend deserialization. Backends may also store auxiliary
//! payloads (e.g., tuning results) in the same directory as `<directory>/<sha256>.metadata` entries that use the same
//! envelope.
//!
//! Writes are atomic because they use the standard pattern of writing a temporary file and renaming it. Filesystem and
//! validation failures are returned to [`CompilationContext`](super::CompilationContext), which records each failure
//! and degrades it to a cache miss.
//!
//! When constructed via [`DiskCache::with_capacity`] or with the [`DiskCache::MAX_BYTES_ENV_VAR`] environment
//! variable set, the cache evicts the least recently written entries after each write to keep its footprint on disk
//! under the configured cap. Caches opened via [`DiskCache::open`] have no cap.

// TODO(eaplatanios): Review this module.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use sha2::{Digest, Sha256};

/// Magic bytes that open every entry envelope.
const ENTRY_MAGIC: &[u8; 8] = b"RYFTCACH";

/// Version of the entry envelope format.
const ENTRY_FORMAT_VERSION: u32 = 1;

/// Length of a SHA-256 digest, in bytes.
const SHA256_LENGTH: usize = 32;

/// Length of the entry envelope header (i.e., magic, version, key digest, payload length, and payload checksum).
const ENTRY_HEADER_LENGTH: usize = ENTRY_MAGIC.len() + 4 + SHA256_LENGTH + 8 + SHA256_LENGTH;

/// Maximum size of an auxiliary payload, in bytes.
const MAXIMUM_AUXILIARY_ENTRY_SIZE: usize = 64 * 1024 * 1024;

/// Filename extension of serialized executable entries.
const EXECUTABLE_EXTENSION: &str = "executable";

/// Filename extension of auxiliary entries.
const AUXILIARY_EXTENSION: &str = "metadata";

/// Maximum number of temporary filenames that a single write tries before giving up.
const MAXIMUM_TEMPORARY_FILE_ATTEMPTS: usize = 256;

/// Default minimum compilation duration required before a newly compiled program is persisted.
const DEFAULT_MINIMUM_COMPILE_DURATION: Duration = Duration::from_secs(1);

/// Default minimum serialized payload size required before a newly compiled program is persisted.
const DEFAULT_MINIMUM_ENTRY_SIZE: usize = 0;

/// Directory-backed cache of serialized compiled programs.
///
/// Wire one into a [`CompilationContext`](super::CompilationContext) via
/// [`CompilationContext::with_disk_cache`](super::CompilationContext::with_disk_cache),
/// [`CompilationContext::with_configured_disk_cache`](super::CompilationContext::with_configured_disk_cache), or
/// [`CompilationContext::with_disk_cache_from_env`](super::CompilationContext::with_disk_cache_from_env).
pub struct DiskCache {
    /// Filesystem directory holding the cached entries.
    directory: PathBuf,

    /// Candidate suffix for temporary filenames. Exclusive file creation resolves collisions with independently opened
    /// caches and with stale files left by an earlier process that had the same process identifier.
    write_counter: AtomicU64,

    /// Optional cap on the size of the cached entries on disk, in bytes. [`None`] disables eviction entirely. When set,
    /// the cache evicts the least recently written entries after each successful write to keep the total under the
    /// cap, always preserving the entry that was just written.
    max_bytes: Option<u64>,

    /// Minimum compilation duration required before a newly compiled program is persisted.
    minimum_compile_duration: Duration,

    /// Minimum serialized backend payload size required before it is persisted.
    minimum_entry_size: usize,
}

impl DiskCache {
    /// Environment variable that [`Self::from_env`] reads for the cache directory.
    pub const ENV_VAR: &str = "RYFT_COMPILATION_CACHE_DIR";

    /// Environment variable that [`Self::from_env`] reads for the optional cap on the size of the cache on disk, in
    /// bytes. It must hold an unsigned 64-bit integer when set, and eviction is disabled when it is absent.
    pub const MAX_BYTES_ENV_VAR: &str = "RYFT_COMPILATION_CACHE_MAX_BYTES";

    /// Opens a [`DiskCache`] rooted at `directory`, creating the directory and any missing parents if necessary.
    /// Returns an error if `directory` already exists as a non-directory or cannot be created.
    ///
    /// The resulting cache has no size cap. Use [`Self::with_capacity`] to bound its footprint on disk.
    pub fn open(directory: impl Into<PathBuf>) -> std::io::Result<Self> {
        Self::open_with_max_bytes(directory, None)
    }

    /// Opens a [`DiskCache`] rooted at `directory` whose entries are evicted, least recently written first, after each
    /// write that brings the size of the cache on disk above `max_bytes`.
    ///
    /// The most recently written entry is always preserved, even if it alone exceeds `max_bytes`, so that callers
    /// always observe their own writes on the immediately following read.
    pub fn with_capacity(directory: impl Into<PathBuf>, max_bytes: u64) -> std::io::Result<Self> {
        Self::open_with_max_bytes(directory, Some(max_bytes))
    }

    /// Opens the [`DiskCache`] configured by the [`RYFT_COMPILATION_CACHE_DIR`](Self::ENV_VAR) environment variable.
    /// Returns `Ok(Some(cache))` when the variable is set and points to a directory that can be read or created, and
    /// `Ok(None)` when it is unset. Invalid capacity configuration and filesystem failures are returned explicitly.
    ///
    /// When [`RYFT_COMPILATION_CACHE_MAX_BYTES`](Self::MAX_BYTES_ENV_VAR) is also set, the resulting cache is capped
    /// as if it were opened via [`Self::with_capacity`].
    pub fn from_env() -> std::io::Result<Option<Self>> {
        let Some(directory) = std::env::var_os(Self::ENV_VAR) else {
            return Ok(None);
        };
        let max_bytes = match std::env::var(Self::MAX_BYTES_ENV_VAR) {
            Ok(value) => Some(value.parse::<u64>().map_err(|error| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("invalid `{}` value `{value}`: {error}", Self::MAX_BYTES_ENV_VAR),
                )
            })?),
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("invalid `{}` value: {error}", Self::MAX_BYTES_ENV_VAR),
                ));
            }
        };
        Self::open_with_max_bytes(directory, max_bytes).map(Some)
    }

    /// Opens a [`DiskCache`] rooted at `directory` with an optional size cap and the default write thresholds.
    fn open_with_max_bytes(directory: impl Into<PathBuf>, max_bytes: Option<u64>) -> std::io::Result<Self> {
        let directory = directory.into();
        fs::create_dir_all(&directory)?;
        Ok(Self {
            directory,
            write_counter: AtomicU64::new(0),
            max_bytes,
            minimum_compile_duration: DEFAULT_MINIMUM_COMPILE_DURATION,
            minimum_entry_size: DEFAULT_MINIMUM_ENTRY_SIZE,
        })
    }

    /// Configures the minimum compilation duration and serialized payload size required before a newly compiled
    /// program is written. Reads are unaffected, so an existing validated entry is always eligible for reuse.
    #[inline]
    pub fn with_write_thresholds(mut self, minimum_compile_duration: Duration, minimum_entry_size: usize) -> Self {
        self.minimum_compile_duration = minimum_compile_duration;
        self.minimum_entry_size = minimum_entry_size;
        self
    }

    /// Returns the directory backing this cache.
    #[inline]
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Returns the configured size cap, if any.
    #[inline]
    pub fn max_bytes(&self) -> Option<u64> {
        self.max_bytes
    }

    /// Returns the total size, in bytes, of every executable and auxiliary entry in the cache directory, which is
    /// zero for a missing or empty directory.
    pub fn cache_size_bytes(&self) -> std::io::Result<u64> {
        let entries = match fs::read_dir(&self.directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(error) => return Err(error),
        };
        let mut total_bytes = 0u64;
        for entry in entries {
            let entry = entry?;
            if Self::is_entry_path(&entry.path()) {
                total_bytes = total_bytes.saturating_add(entry.metadata()?.len());
            }
        }
        Ok(total_bytes)
    }

    /// Reads a checksummed auxiliary payload that a backend associated with `key` in `namespace`.
    ///
    /// Auxiliary entries share the directory of this cache but use a different filename extension than executables.
    /// Reads are bounded to 64 MiB of uncompressed payload so that corrupted compressed inputs cannot grow memory
    /// without limit. Missing entries return `Ok(None)`, while validation and I/O failures are returned explicitly.
    pub fn get_auxiliary(&self, namespace: &str, key: &[u8]) -> std::io::Result<Option<Vec<u8>>> {
        let digest = Self::auxiliary_digest(namespace, key);
        Self::read_entry(&self.auxiliary_entry_path(&digest), &digest, Some(MAXIMUM_AUXILIARY_ENTRY_SIZE))
    }

    /// Atomically writes a checksummed auxiliary payload that a backend associates with `key` in `namespace`.
    ///
    /// Payloads larger than 64 MiB are rejected. Readers observe either the prior complete entry or the new complete
    /// entry across crashes and concurrent processes.
    pub fn put_auxiliary(&self, namespace: &str, key: &[u8], data: &[u8]) -> std::io::Result<()> {
        if data.len() > MAXIMUM_AUXILIARY_ENTRY_SIZE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "persistent auxiliary payload exceeds the 64 MiB limit",
            ));
        }
        let digest = Self::auxiliary_digest(namespace, key);
        self.write_entry(&self.auxiliary_entry_path(&digest), &digest, AUXILIARY_EXTENSION, data)
    }

    /// Reads and validates the cached serialized payload for `digest`. A missing entry is a cache miss, while
    /// filesystem, decompression, envelope, and checksum failures are returned explicitly.
    pub(crate) fn get(&self, digest: &CacheDigest) -> std::io::Result<Option<Vec<u8>>> {
        Self::read_entry(&self.entry_path(digest), digest, None)
    }

    /// Atomically writes `data` to the cache under `digest`, so that a partial write or process crash can never leave
    /// a half-written entry visible to other readers. After a successful write, evicts the least recently written
    /// entries when a size cap is configured.
    ///
    /// Returns any write, synchronization, rename, cleanup, or eviction failure explicitly. The compilation context
    /// records such failures before degrading to an in-memory result.
    pub(crate) fn put(&self, digest: &CacheDigest, data: &[u8]) -> std::io::Result<()> {
        self.write_entry(&self.entry_path(digest), digest, EXECUTABLE_EXTENSION, data)
    }

    /// Returns whether a compilation that took `compile_duration` may be persisted, which is checked before backend
    /// serialization so that cheap compilations are never serialized.
    #[inline]
    pub(crate) fn should_serialize(&self, compile_duration: Duration) -> bool {
        compile_duration >= self.minimum_compile_duration
    }

    /// Returns whether a newly compiled program that took `compile_duration` and serialized to `entry_size` bytes
    /// meets the write policy of this cache.
    #[inline]
    pub(crate) fn should_persist(&self, compile_duration: Duration, entry_size: usize) -> bool {
        self.should_serialize(compile_duration) && entry_size >= self.minimum_entry_size
    }

    /// Returns the filesystem path of the executable entry for `digest`.
    fn entry_path(&self, digest: &CacheDigest) -> PathBuf {
        self.directory.join(format!("{}.{EXECUTABLE_EXTENSION}", digest.as_hex()))
    }

    /// Returns the filesystem path of the auxiliary entry for `digest`.
    fn auxiliary_entry_path(&self, digest: &CacheDigest) -> PathBuf {
        self.directory.join(format!("{}.{AUXILIARY_EXTENSION}", digest.as_hex()))
    }

    /// Returns whether `path` names an executable or auxiliary entry, as opposed to a temporary or unrelated file.
    fn is_entry_path(path: &Path) -> bool {
        path.extension()
            .is_some_and(|extension| extension == EXECUTABLE_EXTENSION || extension == AUXILIARY_EXTENSION)
    }

    /// Returns the digest that addresses the auxiliary entry for `key` in `namespace`. The digest is domain separated
    /// from executable digests and hashes both components independently, so that no pair of namespaces and keys
    /// can collide by concatenation.
    fn auxiliary_digest(namespace: &str, key: &[u8]) -> CacheDigest {
        let mut hasher = Sha256::new();
        hasher.update(b"RYFT-AUXILIARY\0");
        hasher.update(Sha256::digest(namespace.as_bytes()));
        hasher.update(Sha256::digest(key));
        CacheDigest { bytes: hasher.finalize().into() }
    }

    /// Reads and validates the entry at `path` for `digest`, returning `Ok(None)` when it does not exist.
    ///
    /// # Parameters
    ///
    ///   - `path`: Filesystem path of the entry.
    ///   - `digest`: Digest that the entry envelope must repeat.
    ///   - `maximum_payload_size`: Optional bound on the uncompressed payload size, which also bounds the memory that
    ///     decompression may allocate.
    fn read_entry(
        path: &Path,
        digest: &CacheDigest,
        maximum_payload_size: Option<usize>,
    ) -> std::io::Result<Option<Vec<u8>>> {
        let file = match File::open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        let mut envelope = Vec::new();
        match maximum_payload_size {
            None => {
                GzDecoder::new(file).read_to_end(&mut envelope)?;
            }
            Some(maximum_payload_size) => {
                let maximum_envelope_size = ENTRY_HEADER_LENGTH + maximum_payload_size;
                GzDecoder::new(file).take(maximum_envelope_size as u64 + 1).read_to_end(&mut envelope)?;
                if envelope.len() > maximum_envelope_size {
                    return Err(Self::invalid_data(format!(
                        "persistent cache payload exceeds the {} MiB limit",
                        maximum_payload_size / (1024 * 1024),
                    )));
                }
            }
        }
        Self::decode_envelope(digest, envelope.as_slice()).map(Some)
    }

    /// Atomically writes `data` as the entry at `path` for `digest` and then evicts older entries when a size cap is
    /// configured.
    ///
    /// The entry is first written gzip-compressed to an exclusively created temporary file, synchronized, and then
    /// renamed over `path`, so that readers observe either the prior complete entry or the new complete entry. The
    /// temporary file is removed when any step before the rename fails.
    fn write_entry(&self, path: &Path, digest: &CacheDigest, extension: &str, data: &[u8]) -> std::io::Result<()> {
        let (temporary_path, temporary_file) = self.temporary_entry(digest, extension)?;
        let write_result = (|| -> std::io::Result<()> {
            // Fast compression bounds the host overhead added after an already expensive compilation.
            let mut encoder = GzEncoder::new(temporary_file, Compression::fast());
            Self::write_envelope(&mut encoder, digest, data)?;
            encoder.finish()?.sync_all()
        })();
        if let Err(error) = write_result.and_then(|()| fs::rename(&temporary_path, path)) {
            return Self::remove_temporary_entry(&temporary_path, error);
        }
        if let Some(max_bytes) = self.max_bytes {
            self.evict_to_fit(path, max_bytes)?;
        }
        Ok(())
    }

    /// Exclusively creates a temporary file for an entry without truncating the active write of another cache instance.
    /// Existing files, including symbolic links and files abandoned by an earlier process with the same process
    /// identifier, are skipped, while all other creation errors are returned without deleting anything.
    fn temporary_entry(&self, digest: &CacheDigest, extension: &str) -> std::io::Result<(PathBuf, File)> {
        for _ in 0..MAXIMUM_TEMPORARY_FILE_ATTEMPTS {
            let counter = self.write_counter.fetch_add(1, Ordering::Relaxed);
            let path = self.directory.join(format!("{}.{extension}.tmp.{}.{counter}", digest.as_hex(), process::id()));
            match File::options().write(true).create_new(true).open(&path) {
                Ok(file) => return Ok((path, file)),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!("could not reserve a temporary cache file after {MAXIMUM_TEMPORARY_FILE_ATTEMPTS} attempts"),
        ))
    }

    /// Removes the temporary file at `path` after a failed write and returns `error`, extended with the cleanup failure
    /// if the file could not be removed.
    fn remove_temporary_entry(path: &Path, error: std::io::Error) -> std::io::Result<()> {
        match fs::remove_file(path) {
            Ok(()) => Err(error),
            Err(cleanup_error) if cleanup_error.kind() == std::io::ErrorKind::NotFound => Err(error),
            Err(cleanup_error) => Err(std::io::Error::new(
                error.kind(),
                format!("{error}, and failed to clean up the temporary cache entry: {cleanup_error}"),
            )),
        }
    }

    /// Evicts the least recently written entries until the size of the cache on disk is at most `max_bytes`, never
    /// evicting the entry at `keep_path`. Returns the number of evicted entries.
    fn evict_to_fit(&self, keep_path: &Path, max_bytes: u64) -> std::io::Result<usize> {
        let mut total_bytes = 0u64;
        let mut candidates = Vec::new();
        for entry in fs::read_dir(&self.directory)? {
            let entry = entry?;
            let path = entry.path();
            if !Self::is_entry_path(&path) {
                continue;
            }
            let metadata = entry.metadata()?;
            total_bytes = total_bytes.saturating_add(metadata.len());
            if path != keep_path {
                candidates.push((path, metadata.modified()?, metadata.len()));
            }
        }
        if total_bytes <= max_bytes {
            return Ok(0);
        }
        candidates.sort_by_key(|(_, modified, _)| *modified);
        let mut bytes_to_evict = total_bytes - max_bytes;
        let mut evicted = 0usize;
        for (path, _, size) in candidates {
            if bytes_to_evict == 0 {
                break;
            }
            match fs::remove_file(&path) {
                Ok(()) => {
                    evicted += 1;
                    bytes_to_evict = bytes_to_evict.saturating_sub(size);
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => return Err(error),
            }
        }
        Ok(evicted)
    }

    /// Returns the uncompressed envelope that stores `payload` under `digest`.
    #[cfg(test)]
    fn encode_envelope(digest: &CacheDigest, payload: &[u8]) -> Vec<u8> {
        let mut envelope = Vec::with_capacity(ENTRY_HEADER_LENGTH + payload.len());
        Self::write_envelope(&mut envelope, digest, payload).unwrap();
        envelope
    }

    /// Writes the uncompressed envelope that stores `payload` under `digest` to `writer`.
    fn write_envelope(writer: &mut impl Write, digest: &CacheDigest, payload: &[u8]) -> std::io::Result<()> {
        let payload_checksum = Sha256::digest(payload);
        let payload_length = u64::try_from(payload.len())
            .map_err(|_| Self::invalid_data("persistent cache payload length does not fit in `u64`"))?;
        writer.write_all(ENTRY_MAGIC)?;
        writer.write_all(&ENTRY_FORMAT_VERSION.to_le_bytes())?;
        writer.write_all(digest.as_bytes())?;
        writer.write_all(&payload_length.to_le_bytes())?;
        writer.write_all(payload_checksum.as_slice())?;
        writer.write_all(payload)
    }

    /// Validates the uncompressed `envelope` of an entry stored under `digest` and returns its payload.
    fn decode_envelope(digest: &CacheDigest, envelope: &[u8]) -> std::io::Result<Vec<u8>> {
        if envelope.len() < ENTRY_HEADER_LENGTH {
            return Err(Self::invalid_data("persistent cache entry is shorter than its header"));
        }
        if &envelope[..ENTRY_MAGIC.len()] != ENTRY_MAGIC {
            return Err(Self::invalid_data("persistent cache entry has an invalid magic value"));
        }

        let mut offset = ENTRY_MAGIC.len();
        let version = u32::from_le_bytes(envelope[offset..offset + 4].try_into().unwrap());
        offset += 4;
        if version != ENTRY_FORMAT_VERSION {
            return Err(Self::invalid_data(format!(
                "persistent cache entry uses unsupported format version {version}",
            )));
        }

        let stored_digest = &envelope[offset..offset + SHA256_LENGTH];
        offset += SHA256_LENGTH;
        if stored_digest != digest.as_bytes() {
            return Err(Self::invalid_data("persistent cache entry key digest does not match its requested key"));
        }

        let payload_length = u64::from_le_bytes(envelope[offset..offset + 8].try_into().unwrap());
        offset += 8;
        let payload_length = usize::try_from(payload_length)
            .map_err(|_| Self::invalid_data("persistent cache payload length does not fit in memory"))?;
        let stored_checksum = &envelope[offset..offset + SHA256_LENGTH];
        offset += SHA256_LENGTH;
        let payload = &envelope[offset..];
        if payload.len() != payload_length {
            return Err(Self::invalid_data(format!(
                "persistent cache payload declares {payload_length} bytes but contains {}",
                payload.len(),
            )));
        }
        if Sha256::digest(payload).as_slice() != stored_checksum {
            return Err(Self::invalid_data("persistent cache payload checksum does not match"));
        }
        Ok(payload.to_vec())
    }

    /// Returns an [`std::io::ErrorKind::InvalidData`] error carrying `message`.
    #[inline]
    fn invalid_data(message: impl Into<String>) -> std::io::Error {
        std::io::Error::new(std::io::ErrorKind::InvalidData, message.into())
    }
}

/// SHA-256 digest of a stable persistent cache key provided by a compilation domain.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct CacheDigest {
    /// Raw digest bytes.
    bytes: [u8; SHA256_LENGTH],
}

impl CacheDigest {
    /// Hashes the stable canonical persistent key bytes provided by a compilation domain.
    pub(crate) fn from_bytes(key: &[u8]) -> Self {
        Self { bytes: Sha256::digest(key).into() }
    }

    /// Returns the raw digest bytes.
    #[inline]
    fn as_bytes(&self) -> &[u8; SHA256_LENGTH] {
        &self.bytes
    }

    /// Returns the lowercase hexadecimal encoding of this digest, which is suitable for filenames.
    fn as_hex(&self) -> String {
        self.bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

#[cfg(test)]
mod tests {
    use std::fs::FileTimes;
    use std::sync::Barrier;
    use std::time::SystemTime;

    use super::*;

    #[test]
    fn test_disk_cache_open() {
        let directory = tempfile::tempdir().unwrap();
        let nested = directory.path().join("nested").join("cache");
        let cache = DiskCache::open(&nested).unwrap();
        assert_eq!(cache.directory(), nested.as_path());
        assert_eq!(cache.max_bytes(), None);
        assert_eq!(cache.cache_size_bytes().unwrap(), 0);

        // A path that already exists as a file cannot back a cache.
        let file = directory.path().join("file");
        fs::write(&file, b"not a directory").unwrap();
        assert!(DiskCache::open(&file).is_err());
    }

    #[test]
    fn test_disk_cache_with_capacity() {
        let directory = tempfile::tempdir().unwrap();
        let cache = DiskCache::with_capacity(directory.path(), 64).unwrap();
        assert_eq!(cache.max_bytes(), Some(64));

        // The most recently written entry is preserved even when it alone exceeds the cap.
        let digest = CacheDigest::from_bytes(b"oversized");
        let payload = vec![0u8; 16 * 1024];
        cache.put(&digest, &payload).unwrap();
        assert_eq!(cache.get(&digest).unwrap(), Some(payload));
    }

    #[test]
    fn test_disk_cache_with_write_thresholds() {
        let directory = tempfile::tempdir().unwrap();
        let cache = DiskCache::open(directory.path()).unwrap().with_write_thresholds(Duration::from_secs(2), 128);

        assert!(!cache.should_serialize(Duration::from_secs(1)));
        assert!(cache.should_serialize(Duration::from_secs(2)));
        assert!(!cache.should_persist(Duration::from_secs(1), 256));
        assert!(!cache.should_persist(Duration::from_secs(3), 64));
        assert!(cache.should_persist(Duration::from_secs(2), 128));
    }

    #[test]
    fn test_disk_cache_get_auxiliary() {
        let directory = tempfile::tempdir().unwrap();
        DiskCache::open(directory.path())
            .unwrap()
            .put_auxiliary("adaptive-profile", b"baseline-and-policy", b"aggregated-profile")
            .unwrap();

        let reopened = DiskCache::open(directory.path()).unwrap();
        assert_eq!(
            reopened.get_auxiliary("adaptive-profile", b"baseline-and-policy").unwrap(),
            Some(b"aggregated-profile".to_vec()),
        );
        assert_eq!(reopened.get_auxiliary("different-namespace", b"baseline-and-policy").unwrap(), None);
    }

    #[test]
    fn test_disk_cache_get_auxiliary_rejects_corruption() {
        let directory = tempfile::tempdir().unwrap();
        let cache = DiskCache::open(directory.path()).unwrap();
        let key = b"baseline-and-policy";
        cache.put_auxiliary("adaptive-profile", key, b"aggregated-profile").unwrap();
        let path = cache.auxiliary_entry_path(&DiskCache::auxiliary_digest("adaptive-profile", key));
        let mut bytes = fs::read(&path).unwrap();
        *bytes.last_mut().unwrap() ^= 0xff;
        fs::write(path, bytes).unwrap();

        // Corrupting the gzip trailer makes decompression fail before the envelope is validated.
        let error = cache.get_auxiliary("adaptive-profile", key).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn test_disk_cache_put_auxiliary() {
        let directory = tempfile::tempdir().unwrap();
        let cache = DiskCache::open(directory.path()).unwrap();
        let error = cache.put_auxiliary("adaptive-profile", b"key", &vec![0u8; MAXIMUM_AUXILIARY_ENTRY_SIZE + 1]);
        let error = error.unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(error.to_string(), "persistent auxiliary payload exceeds the 64 MiB limit");
        assert_eq!(cache.cache_size_bytes().unwrap(), 0);
    }

    #[test]
    fn test_disk_cache_put_auxiliary_evicts_to_capacity() {
        let directory = tempfile::tempdir().unwrap();
        let uncapped = DiskCache::open(directory.path()).unwrap();
        uncapped.put_auxiliary("adaptive-profile", b"first", &[1; 256]).unwrap();
        let first_path = uncapped.auxiliary_entry_path(&DiskCache::auxiliary_digest("adaptive-profile", b"first"));
        let first_size = fs::metadata(&first_path).unwrap().len();

        let cache = DiskCache::with_capacity(directory.path(), first_size).unwrap();
        cache.put_auxiliary("adaptive-profile", b"second", &[2; 256]).unwrap();

        assert_eq!(cache.get_auxiliary("adaptive-profile", b"first").unwrap(), None);
        assert_eq!(cache.get_auxiliary("adaptive-profile", b"second").unwrap(), Some(vec![2; 256]));
        let second_path = cache.auxiliary_entry_path(&DiskCache::auxiliary_digest("adaptive-profile", b"second"));
        assert_eq!(cache.cache_size_bytes().unwrap(), fs::metadata(second_path).unwrap().len());
    }

    #[test]
    fn test_disk_cache_get() {
        let directory = tempfile::tempdir().unwrap();
        let cache = DiskCache::open(directory.path()).unwrap();
        let digest = CacheDigest::from_bytes(b"stable-key");
        let data = b"hello compiled bytes".to_vec();

        assert_eq!(cache.get(&digest).unwrap(), None);
        cache.put(&digest, &data).unwrap();
        assert_eq!(cache.get(&digest).unwrap(), Some(data));
        assert_eq!(cache.get(&CacheDigest::from_bytes(b"missing")).unwrap(), None);
    }

    #[test]
    fn test_disk_cache_put_compresses_entries() {
        let directory = tempfile::tempdir().unwrap();
        let cache = DiskCache::open(directory.path()).unwrap();
        let digest = CacheDigest::from_bytes(b"compressible");
        let payload = vec![0u8; 64 * 1024];
        cache.put(&digest, &payload).unwrap();

        let size_on_disk = fs::metadata(cache.entry_path(&digest)).unwrap().len();
        assert!(
            size_on_disk < payload.len() as u64,
            "expected gzip to compress 64 KiB of zeros below the raw size, got {size_on_disk} bytes",
        );
        assert_eq!(cache.cache_size_bytes().unwrap(), size_on_disk);
        assert_eq!(cache.get(&digest).unwrap(), Some(payload));
    }

    #[test]
    fn test_disk_cache_put_independent_instances() {
        let directory = tempfile::tempdir().unwrap();
        let first = DiskCache::open(directory.path()).unwrap();
        let second = DiskCache::open(directory.path()).unwrap();
        let digest = CacheDigest::from_bytes(b"same-key");
        let barrier = Barrier::new(2);
        let first_payload = vec![17u8; 32 * 1024];
        let second_payload = vec![93u8; 48 * 1024];
        std::thread::scope(|scope| {
            let left = scope.spawn(|| {
                barrier.wait();
                first.put(&digest, &first_payload).unwrap();
                first.put_auxiliary("tuning", b"same-key", &first_payload).unwrap();
            });
            let right = scope.spawn(|| {
                barrier.wait();
                second.put(&digest, &second_payload).unwrap();
                second.put_auxiliary("tuning", b"same-key", &second_payload).unwrap();
            });
            left.join().unwrap();
            right.join().unwrap();
        });
        let executable = first.get(&digest).unwrap().unwrap();
        let auxiliary = second.get_auxiliary("tuning", b"same-key").unwrap().unwrap();
        assert!(executable == first_payload || executable == second_payload);
        assert!(auxiliary == first_payload || auxiliary == second_payload);

        // Atomic publication leaves only the two complete envelope files, with no temporary write debris.
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 2);
    }

    #[test]
    fn test_disk_cache_temporary_entry_independent_instances() {
        let directory = tempfile::tempdir().unwrap();
        let first = DiskCache::open(directory.path()).unwrap();
        let second = DiskCache::open(directory.path()).unwrap();
        let digest = CacheDigest::from_bytes(b"same-key");
        let barrier = Barrier::new(2);

        // Both caches begin with counter zero. Holding both files open together prevents either writer from reusing a
        // name after the other publishes or cleans up, which deterministically exercises independent instances.
        let paths = std::thread::scope(|scope| {
            let left = scope.spawn(|| {
                let (path, mut file) = first.temporary_entry(&digest, AUXILIARY_EXTENSION).unwrap();
                barrier.wait();
                file.write_all(b"first writer").unwrap();
                path
            });
            let right = scope.spawn(|| {
                let (path, mut file) = second.temporary_entry(&digest, AUXILIARY_EXTENSION).unwrap();
                barrier.wait();
                file.write_all(b"second writer").unwrap();
                path
            });
            (left.join().unwrap(), right.join().unwrap())
        });
        assert_ne!(paths.0, paths.1);
        assert_eq!(fs::read(&paths.0).unwrap(), b"first writer");
        assert_eq!(fs::read(&paths.1).unwrap(), b"second writer");

        // A reopened instance must also skip abandoned files instead of truncating their contents.
        let reopened = DiskCache::open(directory.path()).unwrap();
        let (path, file) = reopened.temporary_entry(&digest, AUXILIARY_EXTENSION).unwrap();
        assert_ne!(path, paths.0);
        assert_ne!(path, paths.1);
        drop(file);
        assert_eq!(fs::read(&paths.0).unwrap(), b"first writer");
        assert_eq!(fs::read(&paths.1).unwrap(), b"second writer");
    }

    #[test]
    fn test_disk_cache_evict_to_fit() {
        let directory = tempfile::tempdir().unwrap();
        let cache = DiskCache::open(directory.path()).unwrap();
        let payloads: Vec<Vec<u8>> =
            (0..3).map(|entry| (0..256).map(|byte| (entry * 31 + byte) as u8).collect()).collect();
        let digests: Vec<CacheDigest> = (0..3).map(|index| CacheDigest::from_bytes(&[index])).collect();
        for index in 0..3 {
            cache.put(&digests[index], &payloads[index]).unwrap();
            let file = File::options().write(true).open(cache.entry_path(&digests[index])).unwrap();
            let modified = SystemTime::UNIX_EPOCH + Duration::from_secs(index as u64 + 1);
            file.set_times(FileTimes::new().set_modified(modified)).unwrap();
        }

        let sizes = digests
            .iter()
            .map(|digest| fs::metadata(cache.entry_path(digest)).unwrap().len())
            .collect::<Vec<_>>();
        let capacity = sizes[1] + sizes[2];
        assert_eq!(cache.evict_to_fit(&cache.entry_path(&digests[2]), capacity).unwrap(), 1);
        assert_eq!(cache.get(&digests[0]).unwrap(), None);
        assert_eq!(cache.get(&digests[1]).unwrap(), Some(payloads[1].clone()));
        assert_eq!(cache.get(&digests[2]).unwrap(), Some(payloads[2].clone()));
        assert_eq!(cache.cache_size_bytes().unwrap(), capacity);
    }

    #[test]
    fn test_disk_cache_decode_envelope() {
        let digest = CacheDigest::from_bytes(b"key");
        let envelope = DiskCache::encode_envelope(&digest, b"compiled");
        assert_eq!(envelope.len(), ENTRY_HEADER_LENGTH + b"compiled".len());
        assert_eq!(DiskCache::decode_envelope(&digest, &envelope).unwrap(), b"compiled");
    }

    #[test]
    fn test_disk_cache_decode_envelope_rejects_malformed_envelopes() {
        let digest = CacheDigest::from_bytes(b"key");
        let envelope = DiskCache::encode_envelope(&digest, b"compiled");

        let error = DiskCache::decode_envelope(&digest, &envelope[..ENTRY_HEADER_LENGTH - 1]).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), "persistent cache entry is shorter than its header");

        let mut invalid_magic = envelope.clone();
        invalid_magic[0] ^= 0xff;
        let error = DiskCache::decode_envelope(&digest, &invalid_magic).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), "persistent cache entry has an invalid magic value");

        let mut unsupported_version = envelope.clone();
        unsupported_version[ENTRY_MAGIC.len()..ENTRY_MAGIC.len() + 4].copy_from_slice(&2u32.to_le_bytes());
        let error = DiskCache::decode_envelope(&digest, &unsupported_version).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), "persistent cache entry uses unsupported format version 2");

        let other_key = DiskCache::encode_envelope(&CacheDigest::from_bytes(b"other"), b"compiled");
        let error = DiskCache::decode_envelope(&digest, &other_key).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), "persistent cache entry key digest does not match its requested key");

        let error = DiskCache::decode_envelope(&digest, &envelope[..envelope.len() - 1]).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), "persistent cache payload declares 8 bytes but contains 7");

        let mut corrupted_payload = envelope.clone();
        *corrupted_payload.last_mut().unwrap() ^= 0xff;
        let error = DiskCache::decode_envelope(&digest, &corrupted_payload).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), "persistent cache payload checksum does not match");
    }

    #[test]
    fn test_cache_digest_from_bytes() {
        let digest = CacheDigest::from_bytes(b"abc");
        assert_eq!(digest.as_hex(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(digest, CacheDigest::from_bytes(b"abc"));
        assert_ne!(digest, CacheDigest::from_bytes(b"abd"));
    }
}
