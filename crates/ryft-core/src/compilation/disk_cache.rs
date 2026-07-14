//! Disk-backed persistent compile cache.
//!
//! [`DiskCache`] is the `ryft` analogue of JAX's `JAX_COMPILATION_CACHE_DIR`. When attached to a
//! [`CompilationContext`](super::CompilationContext) it serves as a second-tier cache below the
//! in-memory LRU: on cache miss the context first checks the disk for a serialized executable
//! matching the structural cache key, deserializes if found, and otherwise compiles fresh and
//! writes the serialized executable back to disk.
//!
//! Entries are stored gzip-compressed as `<dir>/<sha256>.executable`. A backend opts into this
//! tier by providing stable canonical bytes for the complete persistent cache key. The filename
//! is the SHA-256 digest of those bytes. The compressed entry is a versioned envelope that repeats
//! the key digest and records the payload length and SHA-256 checksum, all of which are validated
//! before backend deserialization.
//!
//! Writes are atomic via the standard temp-file-plus-rename pattern. Filesystem and validation
//! failures are returned to [`CompilationContext`](super::CompilationContext), which records the
//! failure and degrades it to a cache miss without suppressing the fallible operation.
//!
//! When constructed via [`DiskCache::with_capacity`] or with the
//! [`DiskCache::MAX_BYTES_ENV_VAR`] environment variable, the cache evicts oldest entries by
//! file modification time after each write to keep the on-disk footprint under the configured
//! cap. Caches opened via [`DiskCache::open`] have no cap by default.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use sha2::{Digest, Sha256};

const ENTRY_MAGIC: &[u8; 8] = b"RYFTCACH";
const ENTRY_FORMAT_VERSION: u32 = 1;
const SHA256_LENGTH: usize = 32;
const ENTRY_HEADER_LENGTH: usize = ENTRY_MAGIC.len() + 4 + SHA256_LENGTH + 8 + SHA256_LENGTH;
const MAXIMUM_AUXILIARY_ENTRY_SIZE: usize = 64 * 1024 * 1024;
const DEFAULT_MINIMUM_COMPILE_DURATION: Duration = Duration::from_secs(1);
const DEFAULT_MINIMUM_ENTRY_SIZE: usize = 0;

/// Directory-backed cache of serialized compiled programs.
///
/// Wire one into a [`CompilationContext`](super::CompilationContext) via
/// [`CompilationContext::with_disk_cache`](super::CompilationContext::with_disk_cache) or
/// [`CompilationContext::with_disk_cache_from_env`](super::CompilationContext::with_disk_cache_from_env).
pub struct DiskCache {
    /// Filesystem directory holding the cached entries.
    directory: PathBuf,

    /// Counter used to make per-write temp file names unique inside one process, in addition to
    /// the PID. Pairing PID and an atomic counter eliminates the chance of two threads racing on
    /// the same temp path before rename.
    write_counter: AtomicU64,

    /// Optional cap on the on-disk footprint of cached entries, in bytes. `None` disables
    /// eviction entirely. When set, the cache evicts oldest entries by file modification time
    /// after each successful write to keep the total under the cap. Newly-written entries are
    /// preserved (they have the most recent mtime).
    max_bytes: Option<u64>,

    /// Minimum producer duration required before a newly compiled program is persisted.
    minimum_compile_duration: Duration,

    /// Minimum serialized backend payload size required before it is persisted.
    minimum_entry_size: usize,
}

impl DiskCache {
    /// Opens a [`DiskCache`] rooted at `directory`, creating the directory and any missing
    /// parents if necessary. Returns an error if `directory` already exists as a non-directory
    /// or cannot be created.
    ///
    /// The resulting cache has no size cap. Use [`Self::with_capacity`] to bound the on-disk
    /// footprint.
    pub fn open(directory: impl Into<PathBuf>) -> std::io::Result<Self> {
        Self::open_inner(directory, None)
    }

    /// Opens a [`DiskCache`] rooted at `directory` with an explicit byte-cap. Entries beyond
    /// `max_bytes` are evicted by file modification time after each write (oldest first).
    ///
    /// The most recently written entry is always preserved, even if it alone exceeds
    /// `max_bytes`, so callers always observe their own writes on the immediately following
    /// read.
    pub fn with_capacity(directory: impl Into<PathBuf>, max_bytes: u64) -> std::io::Result<Self> {
        Self::open_inner(directory, Some(max_bytes))
    }

    /// Configures the minimum producer duration and serialized payload size required before a
    /// newly compiled program is written. Reads are unaffected: an existing validated entry is
    /// always eligible for reuse.
    #[inline]
    pub fn with_write_thresholds(mut self, minimum_compile_duration: Duration, minimum_entry_size: usize) -> Self {
        self.minimum_compile_duration = minimum_compile_duration;
        self.minimum_entry_size = minimum_entry_size;
        self
    }

    /// Reads the [`RYFT_COMPILATION_CACHE_DIR`](Self::ENV_VAR) environment variable. Returns
    /// `Ok(Some(cache))` when the variable is set and points to a directory we can read or
    /// create, and `Ok(None)` when it is unset. Invalid capacity configuration and filesystem
    /// failures are returned explicitly.
    ///
    /// When [`RYFT_COMPILATION_CACHE_MAX_BYTES`](Self::MAX_BYTES_ENV_VAR) is also set to an
    /// integer, the resulting cache is capped via [`Self::with_capacity`].
    pub fn from_env() -> std::io::Result<Option<Self>> {
        let Some(directory) = std::env::var_os(Self::ENV_VAR) else {
            return Ok(None);
        };
        let max_bytes = match std::env::var(Self::MAX_BYTES_ENV_VAR) {
            Ok(raw) => Some(raw.parse::<u64>().map_err(|error| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("invalid {} value `{raw}`: {error}", Self::MAX_BYTES_ENV_VAR),
                )
            })?),
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("invalid {} value: {error}", Self::MAX_BYTES_ENV_VAR),
                ));
            }
        };
        Self::open_inner(directory, max_bytes).map(Some)
    }

    /// Environment variable that [`Self::from_env`] reads for the cache directory.
    pub const ENV_VAR: &str = "RYFT_COMPILATION_CACHE_DIR";

    /// Environment variable that [`Self::from_env`] reads for the optional on-disk byte cap.
    /// Parsed as an unsigned 64-bit integer. Unparseable or absent values disable eviction.
    pub const MAX_BYTES_ENV_VAR: &str = "RYFT_COMPILATION_CACHE_MAX_BYTES";

    fn open_inner(directory: impl Into<PathBuf>, max_bytes: Option<u64>) -> std::io::Result<Self> {
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

    /// Returns the filesystem path to the entry for `digest`, suitable for [`fs::read`] /
    /// [`fs::write`].
    fn entry_path(&self, digest: &CacheDigest) -> PathBuf {
        self.directory.join(format!("{}.executable", digest.as_hex()))
    }

    fn auxiliary_entry_path(&self, digest: &CacheDigest) -> PathBuf {
        self.directory.join(format!("{}.metadata", digest.as_hex()))
    }

    fn auxiliary_digest(namespace: &str, key: &[u8]) -> CacheDigest {
        let namespace_digest = Sha256::digest(namespace.as_bytes());
        let key_digest = Sha256::digest(key);
        let mut hasher = Sha256::new();
        hasher.update(b"RYFT-AUXILIARY\0");
        hasher.update(namespace_digest);
        hasher.update(key_digest);
        CacheDigest { bytes: hasher.finalize().into() }
    }

    /// Reads a checksummed backend-owned auxiliary payload associated with `key`.
    ///
    /// Auxiliary entries share this cache's directory but use a separate filename extension from executables. Reads
    /// are bounded to 64 MiB of uncompressed envelope data so corrupted compressed inputs cannot grow memory without
    /// limit. Missing entries return `Ok(None)`; validation and I/O failures remain explicit.
    pub fn get_auxiliary(&self, namespace: &str, key: &[u8]) -> std::io::Result<Option<Vec<u8>>> {
        let digest = Self::auxiliary_digest(namespace, key);
        let path = self.auxiliary_entry_path(&digest);
        let file = match File::open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        let decoder = GzDecoder::new(file);
        let mut envelope = Vec::new();
        decoder
            .take((ENTRY_HEADER_LENGTH + MAXIMUM_AUXILIARY_ENTRY_SIZE + 1) as u64)
            .read_to_end(&mut envelope)?;
        if envelope.len() > ENTRY_HEADER_LENGTH + MAXIMUM_AUXILIARY_ENTRY_SIZE {
            return Err(Self::invalid_data("persistent auxiliary payload exceeds the 64 MiB limit"));
        }
        Self::decode_envelope(&digest, envelope.as_slice()).map(Some)
    }

    /// Atomically writes a checksummed backend-owned auxiliary payload associated with `key`.
    ///
    /// Payloads larger than 64 MiB are rejected. The temp-file-plus-rename protocol gives readers either the prior
    /// complete entry or the new complete entry across crashes and concurrent processes.
    pub fn put_auxiliary(&self, namespace: &str, key: &[u8], data: &[u8]) -> std::io::Result<()> {
        if data.len() > MAXIMUM_AUXILIARY_ENTRY_SIZE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "persistent auxiliary payload exceeds the 64 MiB limit",
            ));
        }
        let digest = Self::auxiliary_digest(namespace, key);
        let final_path = self.auxiliary_entry_path(&digest);
        let counter = self.write_counter.fetch_add(1, Ordering::Relaxed);
        let temp_path = self.directory.join(format!("{}.metadata.tmp.{}.{}", digest.as_hex(), process::id(), counter,));
        let write_result = (|| -> std::io::Result<()> {
            let temporary_file = File::create(&temp_path)?;
            let mut encoder = GzEncoder::new(temporary_file, Compression::fast());
            Self::write_envelope(&mut encoder, &digest, data)?;
            let temporary_file = encoder.finish()?;
            temporary_file.sync_all()
        })();
        if let Err(error) = write_result {
            return Self::remove_temporary_entry(&temp_path, error);
        }
        if let Err(error) = fs::rename(&temp_path, &final_path) {
            return Self::remove_temporary_entry(&temp_path, error);
        }
        if let Some(max_bytes) = self.max_bytes {
            self.evict_to_fit(&final_path, max_bytes)?;
        }
        Ok(())
    }

    /// Reads and validates the cached serialized payload for `digest`. A missing entry is a cache
    /// miss; filesystem, decompression, envelope, and checksum failures are returned explicitly.
    pub(crate) fn get(&self, digest: &CacheDigest) -> std::io::Result<Option<Vec<u8>>> {
        let path = self.entry_path(digest);
        let file = match File::open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        let mut decoder = GzDecoder::new(file);
        let mut envelope = Vec::new();
        decoder.read_to_end(&mut envelope)?;
        Self::decode_envelope(digest, envelope.as_slice()).map(Some)
    }

    /// Writes `data` to the cache under `digest`. Compresses with gzip, then uses the
    /// temp-file-plus-rename pattern so a partial write or process crash can never leave a
    /// half-written entry visible to other readers. After a successful write, evicts older
    /// entries by file modification time when a `max_bytes` cap is configured.
    ///
    /// Returns any write, synchronization, rename, cleanup, or eviction failure explicitly. The
    /// compilation context records such failures before degrading to an in-memory-only result.
    pub(crate) fn put(&self, digest: &CacheDigest, data: &[u8]) -> std::io::Result<()> {
        let final_path = self.entry_path(digest);
        let counter = self.write_counter.fetch_add(1, Ordering::Relaxed);
        let temp_path =
            self.directory.join(format!("{}.executable.tmp.{}.{}", digest.as_hex(), process::id(), counter));
        let write_result = (|| -> std::io::Result<()> {
            // Fast compression bounds host-side overhead after an already-expensive compile.
            let tmp_file = File::create(&temp_path)?;
            let mut encoder = GzEncoder::new(tmp_file, Compression::fast());
            Self::write_envelope(&mut encoder, digest, data)?;
            let tmp_file = encoder.finish()?;
            tmp_file.sync_all()?;
            Ok(())
        })();
        if let Err(error) = write_result {
            return Self::remove_temporary_entry(&temp_path, error);
        }
        if let Err(error) = fs::rename(&temp_path, &final_path) {
            return Self::remove_temporary_entry(&temp_path, error);
        }
        if let Some(max_bytes) = self.max_bytes {
            self.evict_to_fit(&final_path, max_bytes)?;
        }
        Ok(())
    }

    fn remove_temporary_entry(path: &Path, error: std::io::Error) -> std::io::Result<()> {
        match fs::remove_file(path) {
            Ok(()) => Err(error),
            Err(cleanup_error) if cleanup_error.kind() == std::io::ErrorKind::NotFound => Err(error),
            Err(cleanup_error) => Err(std::io::Error::new(
                error.kind(),
                format!("{error}; failed to clean up temporary cache entry: {cleanup_error}"),
            )),
        }
    }

    /// Returns whether a newly compiled serialized payload meets this cache's write policy.
    #[inline]
    pub(crate) fn should_persist(&self, compile_duration: Duration, entry_size: usize) -> bool {
        compile_duration >= self.minimum_compile_duration && entry_size >= self.minimum_entry_size
    }

    /// Returns whether producer latency meets the write policy before backend serialization.
    #[inline]
    pub(crate) fn should_serialize(&self, compile_duration: Duration) -> bool {
        compile_duration >= self.minimum_compile_duration
    }

    #[cfg(test)]
    fn encode_envelope(digest: &CacheDigest, payload: &[u8]) -> Vec<u8> {
        let mut envelope = Vec::with_capacity(ENTRY_HEADER_LENGTH + payload.len());
        Self::write_envelope(&mut envelope, digest, payload).expect("writing an envelope to memory cannot fail");
        envelope
    }

    fn write_envelope(writer: &mut impl Write, digest: &CacheDigest, payload: &[u8]) -> std::io::Result<()> {
        let payload_checksum = Sha256::digest(payload);
        let payload_length = u64::try_from(payload.len())
            .map_err(|_| Self::invalid_data("persistent cache payload length does not fit in u64"))?;
        writer.write_all(ENTRY_MAGIC)?;
        writer.write_all(&ENTRY_FORMAT_VERSION.to_le_bytes())?;
        writer.write_all(digest.as_bytes())?;
        writer.write_all(&payload_length.to_le_bytes())?;
        writer.write_all(payload_checksum.as_slice())?;
        writer.write_all(payload)
    }

    fn decode_envelope(digest: &CacheDigest, envelope: &[u8]) -> std::io::Result<Vec<u8>> {
        if envelope.len() < ENTRY_HEADER_LENGTH {
            return Err(Self::invalid_data("persistent cache entry is shorter than its header"));
        }
        if &envelope[..ENTRY_MAGIC.len()] != ENTRY_MAGIC {
            return Err(Self::invalid_data("persistent cache entry has an invalid magic value"));
        }

        let mut offset = ENTRY_MAGIC.len();
        let version = u32::from_le_bytes(envelope[offset..offset + 4].try_into().expect("version slice has length 4"));
        offset += 4;
        if version != ENTRY_FORMAT_VERSION {
            return Err(Self::invalid_data(format!(
                "persistent cache entry uses unsupported format version {version}"
            )));
        }

        let stored_digest = &envelope[offset..offset + SHA256_LENGTH];
        offset += SHA256_LENGTH;
        if stored_digest != digest.as_bytes() {
            return Err(Self::invalid_data("persistent cache entry key digest does not match its requested key"));
        }

        let payload_length =
            u64::from_le_bytes(envelope[offset..offset + 8].try_into().expect("payload length slice has length 8"));
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

    #[inline]
    fn invalid_data(message: impl Into<String>) -> std::io::Error {
        std::io::Error::new(std::io::ErrorKind::InvalidData, message.into())
    }

    /// Returns the directory backing this cache. Mostly useful for tests.
    #[inline]
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Returns the configured size cap, if any.
    #[inline]
    pub fn max_bytes(&self) -> Option<u64> {
        self.max_bytes
    }

    /// Sums the byte size of every `.executable` and `.metadata` entry in the cache directory. Returns `Ok(0)`
    /// on a missing or empty directory.
    pub fn cache_size_bytes(&self) -> std::io::Result<u64> {
        let mut total: u64 = 0;
        let entries = match fs::read_dir(&self.directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(error) => return Err(error),
        };
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|extension| extension == "executable" || extension == "metadata") {
                total = total.saturating_add(entry.metadata()?.len());
            }
        }
        Ok(total)
    }

    /// Evicts oldest entries by file modification time until the on-disk footprint is at or
    /// below `max_bytes`. Never evicts the entry at `keep_path`. Returns the number of
    /// entries deleted.
    fn evict_to_fit(&self, keep_path: &Path, max_bytes: u64) -> std::io::Result<usize> {
        let mut total_bytes: u64 = 0;
        let mut candidates: Vec<(PathBuf, SystemTime, u64)> = Vec::new();
        for entry in fs::read_dir(&self.directory)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_none_or(|extension| extension != "executable" && extension != "metadata") {
                continue;
            }
            let metadata = entry.metadata()?;
            let size = metadata.len();
            total_bytes = total_bytes.saturating_add(size);
            if path == keep_path {
                continue;
            }
            let modified = metadata.modified()?;
            candidates.push((path, modified, size));
        }
        if total_bytes <= max_bytes {
            return Ok(0);
        }
        // Oldest first.
        candidates.sort_by_key(|(_, modified, _)| *modified);
        let mut bytes_to_evict = total_bytes - max_bytes;
        let mut evicted = 0usize;
        for (path, _, size) in candidates {
            if bytes_to_evict == 0 {
                break;
            }
            match fs::remove_file(&path) {
                Ok(_) => {
                    evicted += 1;
                    bytes_to_evict = bytes_to_evict.saturating_sub(size);
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => return Err(error),
            }
        }
        Ok(evicted)
    }
}

/// SHA-256 digest of a domain-provided stable persistent cache key.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct CacheDigest {
    bytes: [u8; SHA256_LENGTH],
}

impl CacheDigest {
    /// Hashes stable canonical persistent-key bytes supplied by a compilation domain.
    pub(crate) fn from_bytes(key: &[u8]) -> Self {
        Self { bytes: Sha256::digest(key).into() }
    }

    /// Returns the hex-encoded digest suitable for filename construction.
    pub(crate) fn as_hex(&self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut result = String::with_capacity(SHA256_LENGTH * 2);
        for byte in self.bytes {
            result.push(HEX[(byte >> 4) as usize] as char);
            result.push(HEX[(byte & 0x0f) as usize] as char);
        }
        result
    }

    #[inline]
    fn as_bytes(&self) -> &[u8; SHA256_LENGTH] {
        &self.bytes
    }
}

#[cfg(test)]
mod tests {
    use std::fs::FileTimes;

    use super::*;

    #[test]
    fn test_disk_cache_put_then_get_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DiskCache::open(dir.path()).unwrap();
        let digest = CacheDigest::from_bytes(b"stable-key");
        let data = b"hello compiled bytes".to_vec();

        assert!(cache.get(&digest).unwrap().is_none(), "empty cache should miss");
        cache.put(&digest, &data).unwrap();
        assert_eq!(cache.get(&digest).unwrap().as_deref(), Some(data.as_slice()));
    }

    #[test]
    fn test_disk_cache_digest_is_sha256_of_stable_key_bytes() {
        let digest = CacheDigest::from_bytes(b"abc");
        assert_eq!(digest.as_hex(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    #[test]
    fn test_disk_cache_get_returns_none_for_unwritten_key() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DiskCache::open(dir.path()).unwrap();
        assert!(cache.get(&CacheDigest::from_bytes(b"missing")).unwrap().is_none());
    }

    #[test]
    fn test_disk_cache_envelope_rejects_wrong_key() {
        let first = CacheDigest::from_bytes(b"first");
        let second = CacheDigest::from_bytes(b"second");
        let envelope = DiskCache::encode_envelope(&first, b"compiled");

        let error = DiskCache::decode_envelope(&second, &envelope).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("key digest"));
    }

    #[test]
    fn test_disk_cache_envelope_rejects_unsupported_version() {
        let digest = CacheDigest::from_bytes(b"key");
        let mut envelope = DiskCache::encode_envelope(&digest, b"compiled");
        envelope[ENTRY_MAGIC.len()..ENTRY_MAGIC.len() + 4].copy_from_slice(&2u32.to_le_bytes());

        let error = DiskCache::decode_envelope(&digest, &envelope).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("unsupported format version 2"));
    }

    #[test]
    fn test_disk_cache_envelope_rejects_corrupted_payload() {
        let digest = CacheDigest::from_bytes(b"key");
        let mut envelope = DiskCache::encode_envelope(&digest, b"compiled");
        *envelope.last_mut().unwrap() ^= 0xff;

        let error = DiskCache::decode_envelope(&digest, &envelope).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("checksum"));
    }

    #[test]
    fn test_disk_cache_compresses_entries() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DiskCache::open(dir.path()).unwrap();
        let digest = CacheDigest::from_bytes(b"compressible");
        let payload = vec![0u8; 64 * 1024];
        cache.put(&digest, &payload).unwrap();

        let on_disk_size = std::fs::metadata(cache.entry_path(&digest)).unwrap().len();
        assert!(
            on_disk_size < payload.len() as u64,
            "expected gzip to compress 64 KiB of zeros below the raw size, got {on_disk_size} bytes",
        );
        assert_eq!(cache.get(&digest).unwrap().as_deref(), Some(payload.as_slice()));
    }

    #[test]
    fn test_disk_cache_evicts_oldest_entries_deterministically() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DiskCache::open(dir.path()).unwrap();
        let payloads: Vec<Vec<u8>> = (0..3).map(|i| (0..256).map(|j| (i * 31 + j) as u8).collect()).collect();
        let digests: Vec<CacheDigest> = (0..3).map(|index| CacheDigest::from_bytes(&[index])).collect();

        for index in 0..3 {
            cache.put(&digests[index], &payloads[index]).unwrap();
            let file = File::options().write(true).open(cache.entry_path(&digests[index])).unwrap();
            let modified = SystemTime::UNIX_EPOCH + Duration::from_secs(index as u64 + 1);
            file.set_times(FileTimes::new().set_modified(modified)).unwrap();
        }

        let sizes = digests
            .iter()
            .map(|digest| std::fs::metadata(cache.entry_path(digest)).unwrap().len())
            .collect::<Vec<_>>();
        let capacity = sizes[1] + sizes[2];
        assert_eq!(cache.evict_to_fit(&cache.entry_path(&digests[2]), capacity).unwrap(), 1);
        assert!(cache.get(&digests[0]).unwrap().is_none());
        assert_eq!(cache.get(&digests[1]).unwrap().as_deref(), Some(payloads[1].as_slice()));
        assert_eq!(cache.get(&digests[2]).unwrap().as_deref(), Some(payloads[2].as_slice()));
        assert!(cache.cache_size_bytes().unwrap() <= capacity);
    }

    #[test]
    fn test_disk_cache_with_capacity_keeps_most_recent_even_when_alone_exceeds_cap() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DiskCache::with_capacity(dir.path(), 64).unwrap();
        let digest = CacheDigest::from_bytes(b"oversized");
        let payload = vec![0u8; 16 * 1024];
        cache.put(&digest, &payload).unwrap();
        assert_eq!(cache.get(&digest).unwrap().as_deref(), Some(payload.as_slice()));
    }

    #[test]
    fn test_disk_cache_write_thresholds_require_both_limits() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DiskCache::open(dir.path()).unwrap().with_write_thresholds(Duration::from_secs(2), 128);

        assert!(!cache.should_persist(Duration::from_secs(1), 256));
        assert!(!cache.should_persist(Duration::from_secs(3), 64));
        assert!(cache.should_persist(Duration::from_secs(2), 128));
    }

    #[test]
    fn test_disk_cache_auxiliary_payload_survives_reopen() {
        let directory = tempfile::tempdir().unwrap();
        DiskCache::open(directory.path())
            .unwrap()
            .put_auxiliary("adaptive-profile", b"baseline-and-policy", b"aggregated-profile")
            .unwrap();

        let reopened = DiskCache::open(directory.path()).unwrap();
        assert_eq!(
            reopened.get_auxiliary("adaptive-profile", b"baseline-and-policy").unwrap().as_deref(),
            Some(b"aggregated-profile".as_slice()),
        );
        assert!(reopened.get_auxiliary("different-namespace", b"baseline-and-policy").unwrap().is_none());
    }

    #[test]
    fn test_disk_cache_auxiliary_payload_rejects_corruption() {
        let directory = tempfile::tempdir().unwrap();
        let cache = DiskCache::open(directory.path()).unwrap();
        let key = b"baseline-and-policy";
        cache.put_auxiliary("adaptive-profile", key, b"aggregated-profile").unwrap();
        let digest = DiskCache::auxiliary_digest("adaptive-profile", key);
        let path = cache.auxiliary_entry_path(&digest);
        let mut bytes = std::fs::read(&path).unwrap();
        *bytes.last_mut().unwrap() ^= 0xff;
        std::fs::write(path, bytes).unwrap();

        assert!(cache.get_auxiliary("adaptive-profile", key).is_err());
    }

    #[test]
    fn test_disk_cache_capacity_counts_auxiliary_entries() {
        let directory = tempfile::tempdir().unwrap();
        let uncapped = DiskCache::open(directory.path()).unwrap();
        uncapped.put_auxiliary("adaptive-profile", b"first", &[1; 256]).unwrap();
        let first_path = uncapped.auxiliary_entry_path(&DiskCache::auxiliary_digest("adaptive-profile", b"first"));
        let first_size = std::fs::metadata(&first_path).unwrap().len();

        let cache = DiskCache::with_capacity(directory.path(), first_size).unwrap();
        cache.put_auxiliary("adaptive-profile", b"second", &[2; 256]).unwrap();

        assert!(cache.get_auxiliary("adaptive-profile", b"first").unwrap().is_none());
        assert_eq!(cache.get_auxiliary("adaptive-profile", b"second").unwrap().as_deref(), Some([2; 256].as_slice()));
        let second_path = cache.auxiliary_entry_path(&DiskCache::auxiliary_digest("adaptive-profile", b"second"));
        assert_eq!(cache.cache_size_bytes().unwrap(), std::fs::metadata(second_path).unwrap().len());
    }
}
