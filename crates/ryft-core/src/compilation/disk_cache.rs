//! Disk-backed persistent compile cache.
//!
//! [`DiskCache`] is the `ryft` analogue of JAX's `JAX_COMPILATION_CACHE_DIR`. When attached to a
//! [`CompilationContext`](super::CompilationContext) it serves as a second-tier cache below the
//! in-memory LRU: on cache miss the context first checks the disk for a serialized executable
//! matching the structural cache key, deserializes if found, and otherwise compiles fresh and
//! writes the serialized executable back to disk.
//!
//! Entries are stored gzip-compressed as `<dir>/<hex-digest>.executable`, where the digest is a
//! pseudo-random fingerprint derived from the structural cache key produced by
//! [`CompilationDomain::compilation_key`](super::CompilationDomain::compilation_key). Platform
//! scoping (e.g. so a CPU executable can't be loaded against a GPU client) is the domain's
//! responsibility: the domain includes platform identity in its
//! [`CompilationKey`](super::CompilationDomain::CompilationKey) and validates platform
//! compatibility inside
//! [`CompilationDomain::deserialize_program`](super::CompilationDomain::deserialize_program).
//!
//! Writes are atomic via the standard temp-file-plus-rename pattern. Read errors and domain
//! deserialization failures are treated as cache misses, so the cache never blocks compilation
//! when a stored entry can't be loaded.
//!
//! When constructed via [`DiskCache::with_capacity`] or with the
//! [`DiskCache::MAX_BYTES_ENV_VAR`] environment variable, the cache evicts oldest entries by
//! file modification time after each write to keep the on-disk footprint under the configured
//! cap. Caches opened via [`DiskCache::open`] have no cap by default.

use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;

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

    /// Reads the [`RYFT_COMPILATION_CACHE_DIR`](Self::ENV_VAR) environment variable. Returns
    /// `Some(cache)` when the variable is set and points to a directory we can read or create;
    /// returns `None` when the variable is unset or opening the directory fails.
    ///
    /// When [`RYFT_COMPILATION_CACHE_MAX_BYTES`](Self::MAX_BYTES_ENV_VAR) is also set to a
    /// non-zero integer, the resulting cache is capped via [`Self::with_capacity`].
    pub fn from_env() -> Option<Self> {
        let dir = std::env::var_os(Self::ENV_VAR)?;
        let max_bytes = std::env::var(Self::MAX_BYTES_ENV_VAR).ok().and_then(|raw| raw.parse::<u64>().ok());
        Self::open_inner(dir, max_bytes).ok()
    }

    /// Environment variable that [`Self::from_env`] reads for the cache directory.
    pub const ENV_VAR: &'static str = "RYFT_COMPILATION_CACHE_DIR";

    /// Environment variable that [`Self::from_env`] reads for the optional on-disk byte cap.
    /// Parsed as an unsigned 64-bit integer. Unparseable or absent values disable eviction.
    pub const MAX_BYTES_ENV_VAR: &'static str = "RYFT_COMPILATION_CACHE_MAX_BYTES";

    fn open_inner(directory: impl Into<PathBuf>, max_bytes: Option<u64>) -> std::io::Result<Self> {
        let directory = directory.into();
        fs::create_dir_all(&directory)?;
        Ok(Self { directory, write_counter: AtomicU64::new(0), max_bytes })
    }

    /// Returns the filesystem path to the entry for `digest`, suitable for [`fs::read`] /
    /// [`fs::write`].
    fn entry_path(&self, digest: &CacheDigest) -> PathBuf {
        self.directory.join(format!("{}.executable", digest.as_hex()))
    }

    /// Reads the cached serialized bytes for `digest`, returning `None` on any error
    /// (missing file, permission denied, malformed contents, decompression failure). Read errors
    /// are intentionally non-fatal — they just degrade to a cache miss.
    pub(crate) fn get(&self, digest: &CacheDigest) -> Option<Vec<u8>> {
        let path = self.entry_path(digest);
        let file = File::open(&path).ok()?;
        let mut decoder = GzDecoder::new(file);
        let mut buffer = Vec::new();
        decoder.read_to_end(&mut buffer).ok()?;
        Some(buffer)
    }

    /// Writes `data` to the cache under `digest`. Compresses with gzip, then uses the
    /// temp-file-plus-rename pattern so a partial write or process crash can never leave a
    /// half-written entry visible to other readers. After a successful write, evicts older
    /// entries by file modification time when a `max_bytes` cap is configured.
    ///
    /// Returns an error only if the underlying filesystem op fails irrecoverably; the caller can
    /// ignore the error since we degrade to no caching when persistence fails.
    pub(crate) fn put(&self, digest: &CacheDigest, data: &[u8]) -> std::io::Result<()> {
        let final_path = self.entry_path(digest);
        let counter = self.write_counter.fetch_add(1, Ordering::Relaxed);
        let temp_path =
            self.directory.join(format!("{}.executable.tmp.{}.{}", digest.as_hex(), process::id(), counter));
        {
            // `Compression::fast()` trades ratio for CPU; compile-cache writes happen on every
            // miss and we'd rather not stall the compile pipeline for marginal extra savings.
            let tmp_file = File::create(&temp_path)?;
            let mut encoder = GzEncoder::new(tmp_file, Compression::fast());
            encoder.write_all(data)?;
            let tmp_file = encoder.finish()?;
            tmp_file.sync_all()?;
        }
        if let Err(error) = fs::rename(&temp_path, &final_path) {
            // Best-effort cleanup; if the temp file is gone (e.g., racing rename) ignore.
            let _ = fs::remove_file(&temp_path);
            return Err(error);
        }
        if let Some(max_bytes) = self.max_bytes {
            // Best-effort eviction: if listing or unlinking fails we leave the directory at its
            // current size and let the next write try again. Don't surface the error since it
            // doesn't invalidate the entry we just wrote.
            let _ = self.evict_to_fit(digest, max_bytes);
        }
        Ok(())
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

    /// Sums the byte size of every `.executable` entry in the cache directory. Returns `Ok(0)`
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
            if path.extension().is_some_and(|ext| ext == "executable") {
                total = total.saturating_add(entry.metadata()?.len());
            }
        }
        Ok(total)
    }

    /// Evicts oldest entries by file modification time until the on-disk footprint is at or
    /// below `max_bytes`. Never evicts the entry under `keep_digest`. Returns the number of
    /// entries deleted.
    fn evict_to_fit(&self, keep_digest: &CacheDigest, max_bytes: u64) -> std::io::Result<usize> {
        let keep_filename = format!("{}.executable", keep_digest.as_hex());
        let mut total_bytes: u64 = 0;
        let mut candidates: Vec<(PathBuf, SystemTime, u64)> = Vec::new();
        for entry in fs::read_dir(&self.directory)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_none_or(|ext| ext != "executable") {
                continue;
            }
            let metadata = entry.metadata()?;
            let size = metadata.len();
            total_bytes = total_bytes.saturating_add(size);
            if path.file_name().is_some_and(|name| name == keep_filename.as_str()) {
                continue;
            }
            let modified = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
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

/// Digest used to key disk-cache entries. The wrapped bytes are a hex-rendered filename-safe
/// representation of the domain's `u64` cache key, expanded into ~192 bits of pseudo-entropy to
/// keep accidental filename collisions vanishingly rare.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CacheDigest {
    /// Pre-rendered hex string. Keeping this as a `String` avoids re-rendering on each filename
    /// lookup; the digest is small (48 hex chars for the three-chunk SipHash digest).
    hex: String,
}

impl CacheDigest {
    /// Builds a digest from any `Hash`-implementing cache key. Platform-identity scoping (so
    /// that a cached CPU artifact doesn't accidentally serve a GPU client) is the domain's
    /// responsibility — the domain includes platform identity in its
    /// [`CompilationKey`](super::CompilationDomain::CompilationKey), and validates platform
    /// compatibility inside
    /// [`CompilationDomain::deserialize_program`](super::CompilationDomain::deserialize_program).
    ///
    /// We don't pull in a cryptographic-hash crate; SHA-1 is overkill for collision resistance
    /// at this scale anyway. Use Rust's standard `DefaultHasher` (SipHash) repeatedly with
    /// different seed bytes to derive a ~192-bit-equivalent digest — enough entropy to avoid
    /// accidental collisions across the lifetime of a single cache directory while staying
    /// dependency-free.
    pub(crate) fn from_key<K: Hash + ?Sized>(cache_key: &K) -> Self {
        use std::collections::hash_map::DefaultHasher;

        // Three independent hashes seeded with different domain-separation strings together
        // yield ~192 bits of pseudo-entropy. More than enough for our use case.
        let mut chunks = Vec::with_capacity(3);
        for seed in ["ryft.disk_cache.v1.chunk0", "ryft.disk_cache.v1.chunk1", "ryft.disk_cache.v1.chunk2"] {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            cache_key.hash(&mut hasher);
            chunks.push(hasher.finish());
        }
        let mut hex = String::with_capacity(chunks.len() * 16);
        for chunk in chunks {
            hex.push_str(&format!("{:016x}", chunk));
        }
        Self { hex }
    }

    /// Returns the hex-encoded digest as a string slice suitable for filename construction.
    #[inline]
    pub(crate) fn as_hex(&self) -> &str {
        &self.hex
    }
}

#[cfg(test)]
mod tests {
    use std::thread::sleep;
    use std::time::Duration;

    use super::*;

    #[test]
    fn test_disk_cache_put_then_get_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DiskCache::open(dir.path()).unwrap();
        let digest = CacheDigest::from_key(&42u64);
        let data = b"hello compiled bytes".to_vec();

        assert!(cache.get(&digest).is_none(), "empty cache should miss");
        cache.put(&digest, &data).unwrap();
        assert_eq!(cache.get(&digest).as_deref(), Some(data.as_slice()));
    }

    #[test]
    fn test_disk_cache_digest_distinguishes_keys() {
        let a = CacheDigest::from_key(&42u64);
        let b = CacheDigest::from_key(&43u64);
        let c = CacheDigest::from_key(&u64::MAX);
        assert_ne!(a.as_hex(), b.as_hex());
        assert_ne!(a.as_hex(), c.as_hex());
        assert_ne!(b.as_hex(), c.as_hex());
    }

    #[test]
    fn test_disk_cache_get_returns_none_for_unwritten_key() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DiskCache::open(dir.path()).unwrap();
        assert!(cache.get(&CacheDigest::from_key(&0u64)).is_none());
    }

    #[test]
    fn test_disk_cache_compresses_entries() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DiskCache::open(dir.path()).unwrap();
        let digest = CacheDigest::from_key(&7u64);
        // 64 KiB of zeros — a worst case for incompressible data, best case for gzip.
        let payload = vec![0u8; 64 * 1024];
        cache.put(&digest, &payload).unwrap();

        let on_disk_size = std::fs::metadata(cache.entry_path(&digest)).unwrap().len();
        assert!(
            on_disk_size < payload.len() as u64,
            "expected gzip to compress 64 KiB of zeros below the raw size, got {on_disk_size} bytes",
        );
        // Round-trip: decompression must recover the original bytes byte-for-byte.
        assert_eq!(cache.get(&digest).as_deref(), Some(payload.as_slice()));
    }

    #[test]
    fn test_disk_cache_evicts_oldest_entries_when_over_capacity() {
        let dir = tempfile::tempdir().unwrap();
        // Each entry compresses to well under 1 KiB; capping at 600 bytes forces eviction after
        // the third entry lands.
        let cache = DiskCache::with_capacity(dir.path(), 600).unwrap();
        // Distinct, non-trivial payloads so each entry has a measurable compressed size.
        let payloads: Vec<Vec<u8>> = (0..3).map(|i| (0..256).map(|j| (i * 31 + j) as u8).collect()).collect();
        let digests: Vec<CacheDigest> = (0..3).map(|i| CacheDigest::from_key(&(i as u64))).collect();

        // Write the entries with mtime gaps wide enough to be measurable on any common
        // filesystem (HFS+, APFS, ext4 all support millisecond mtime resolution).
        for index in 0..3 {
            cache.put(&digests[index], &payloads[index]).unwrap();
            // Sleep between writes to ensure mtimes order strictly: most filesystems track
            // mtime at second or sub-second resolution, but `fs::rename` may collapse two
            // adjacent writes into the same tick. 100 ms is conservative.
            if index < 2 {
                sleep(Duration::from_millis(100));
            }
        }

        // Newest entry must still be readable.
        assert_eq!(cache.get(&digests[2]).as_deref(), Some(payloads[2].as_slice()));
        // Oldest entry must have been evicted to keep us under the byte cap.
        assert!(cache.get(&digests[0]).is_none(), "oldest entry should be evicted after a write that exceeds the cap");
        // Total on-disk size is at or below the cap.
        let total = cache.cache_size_bytes().unwrap();
        assert!(total <= 600, "cache size {total} exceeds 600-byte cap");
    }

    #[test]
    fn test_disk_cache_with_capacity_keeps_most_recent_even_when_alone_exceeds_cap() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DiskCache::with_capacity(dir.path(), 64).unwrap();
        let digest = CacheDigest::from_key(&13u64);
        // 16 KiB of zeros compresses to a few dozen bytes, but even at minimum still likely >
        // 64 bytes — confirming we never evict the entry we just wrote.
        let payload = vec![0u8; 16 * 1024];
        cache.put(&digest, &payload).unwrap();
        assert_eq!(cache.get(&digest).as_deref(), Some(payload.as_slice()));
    }
}
