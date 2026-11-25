//! Manifests, which pin the files and arrays of a released checkpoint.
//!
//! A [`Manifest`] names a release by its repository and revision and records two things about it. The first is the
//! list of its files with their sizes and, where known, their SHA-256 digests. The second is the complete array table
//! of its checkpoint file (i.e., the name, data type, shape, and byte range of every array). Model code is written
//! against one such table, so [`Manifest::validate_checkpoint`] compares the table of a [`Checkpoint`] with the
//! recorded one before any array contents are read, and [`Manifest::validate_file`] checks a downloaded file against
//! its recorded size and digest. Manifests serialize to JSON with one record per line, which keeps them reviewable
//! when they are checked into a repository next to the code that depends on them.

use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::arrays::{ArrayDescriptor, Checkpoint, data_type_from_tag, data_type_tag};
use crate::errors::CheckpointError;

// TODO(eaplatanios): Review this module.

/// Record of one file of a release, consisting of its path relative to the root of the release, its size, and its
/// SHA-256 digest when one is known.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileRecord {
    /// Path of the file relative to the root of the release.
    path: String,

    /// Size of the file in bytes.
    byte_count: u64,

    /// Lowercase hexadecimal SHA-256 digest of the file contents, if one is known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    sha256: Option<String>,
}

impl FileRecord {
    /// Creates a new [`FileRecord`] without a digest.
    #[inline]
    pub fn new<P: Into<String>>(path: P, byte_count: u64) -> Self {
        Self { path: path.into(), byte_count, sha256: None }
    }

    /// Returns this [`FileRecord`] with the provided lowercase hexadecimal SHA-256 digest.
    #[inline]
    pub fn with_sha256<D: Into<String>>(mut self, sha256: D) -> Self {
        self.sha256 = Some(sha256.into());
        self
    }

    /// Returns the path of the file relative to the root of the release.
    #[inline]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Returns the size of the file in bytes.
    #[inline]
    pub fn byte_count(&self) -> u64 {
        self.byte_count
    }

    /// Returns the lowercase hexadecimal SHA-256 digest of the file contents, if one is known.
    #[inline]
    pub fn sha256(&self) -> Option<&str> {
        self.sha256.as_deref()
    }
}

/// Serialized form of an [`ArrayDescriptor`] in a [`Manifest`], which spells the data type as a safetensors tag.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArrayRecord {
    /// Name of the array.
    name: String,

    /// Safetensors data type tag of the array (e.g., `BF16`).
    data_type: String,

    /// Extents of the dimensions of the array.
    shape: Vec<usize>,

    /// Start and end of the byte range of the array contents, relative to the start of the data section.
    byte_range: [u64; 2],
}

impl ArrayRecord {
    /// Creates the [`ArrayRecord`] of `descriptor`.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::InvalidManifest`] if the data type of `descriptor` has no safetensors tag.
    pub fn from_descriptor(descriptor: &ArrayDescriptor) -> Result<Self, CheckpointError> {
        let data_type = data_type_tag(descriptor.data_type()).ok_or_else(|| CheckpointError::InvalidManifest {
            message: format!(
                "array `{}` has data type `{}`, which has no safetensors data type tag",
                descriptor.name(),
                descriptor.data_type(),
            ),
        })?;
        let byte_range = descriptor.byte_range();
        Ok(Self {
            name: descriptor.name().to_string(),
            data_type: data_type.to_string(),
            shape: descriptor.shape().to_vec(),
            byte_range: [byte_range.start, byte_range.end],
        })
    }

    /// Returns the [`ArrayDescriptor`] that this record describes.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::InvalidManifest`] if the data type tag of this record is not supported.
    pub fn to_descriptor(&self) -> Result<ArrayDescriptor, CheckpointError> {
        let data_type = data_type_from_tag(&self.data_type).ok_or_else(|| CheckpointError::InvalidManifest {
            message: format!("array `{}` has unsupported data type tag `{}`", self.name, self.data_type),
        })?;
        Ok(ArrayDescriptor::new(
            self.name.clone(),
            data_type,
            self.shape.clone(),
            self.byte_range[0]..self.byte_range[1],
        ))
    }
}

/// Pinned description of a released checkpoint, consisting of the repository and revision that identify the release,
/// the [`FileRecord`]s of its files, and the array table of its checkpoint file. Refer to the
/// [module documentation](self) for how manifests are used.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Manifest {
    /// Name of the repository that hosts the release (e.g., an organization and model name).
    repository: String,

    /// Immutable revision of the release (e.g., a commit hash).
    revision: String,

    /// Records of the files of the release.
    files: Vec<FileRecord>,

    /// Descriptors of the arrays of the checkpoint file, in the order of their byte ranges.
    arrays: Vec<ArrayDescriptor>,
}

impl Manifest {
    /// Creates a new [`Manifest`] with no files and no arrays.
    #[inline]
    pub fn new<N: Into<String>, R: Into<String>>(repository: N, revision: R) -> Self {
        Self { repository: repository.into(), revision: revision.into(), files: Vec::new(), arrays: Vec::new() }
    }

    /// Creates a [`Manifest`] with no files whose array table is that of `checkpoint`.
    pub fn from_checkpoint<N: Into<String>, R: Into<String>, C: Checkpoint>(
        repository: N,
        revision: R,
        checkpoint: &C,
    ) -> Self {
        Self::new(repository, revision).with_arrays(checkpoint.arrays().to_vec())
    }

    /// Parses a [`Manifest`] from the JSON form that [`Self::to_json`] produces.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::InvalidManifest`] if `json` is not a valid manifest or if it names a file or a
    /// array more than once.
    pub fn from_json(json: &str) -> Result<Self, CheckpointError> {
        let serialized: SerializedManifest = serde_json::from_str(json)
            .map_err(|error| CheckpointError::InvalidManifest { message: error.to_string() })?;
        let arrays = serialized.arrays.iter().map(ArrayRecord::to_descriptor).collect::<Result<Vec<_>, _>>()?;
        let manifest =
            Self { repository: serialized.repository, revision: serialized.revision, files: serialized.files, arrays };

        let mut paths = HashSet::new();
        for file in &manifest.files {
            if !paths.insert(file.path()) {
                return Err(CheckpointError::InvalidManifest {
                    message: format!("file `{}` is recorded more than once", file.path()),
                });
            }
        }
        let mut names = HashSet::new();
        for array in &manifest.arrays {
            if !names.insert(array.name()) {
                return Err(CheckpointError::InvalidManifest {
                    message: format!("array `{}` is recorded more than once", array.name()),
                });
            }
        }
        Ok(manifest)
    }

    /// Returns this [`Manifest`] with the provided file records.
    #[inline]
    pub fn with_files(mut self, files: Vec<FileRecord>) -> Self {
        self.files = files;
        self
    }

    /// Returns this [`Manifest`] with the provided array table.
    #[inline]
    pub fn with_arrays(mut self, arrays: Vec<ArrayDescriptor>) -> Self {
        self.arrays = arrays;
        self
    }

    /// Returns the name of the repository that hosts the release.
    #[inline]
    pub fn repository(&self) -> &str {
        &self.repository
    }

    /// Returns the immutable revision of the release.
    #[inline]
    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// Returns the records of the files of the release.
    #[inline]
    pub fn files(&self) -> &[FileRecord] {
        &self.files
    }

    /// Returns the record of the file at `path` relative to the root of the release, if there is one.
    #[inline]
    pub fn file(&self, path: &str) -> Option<&FileRecord> {
        self.files.iter().find(|file| file.path() == path)
    }

    /// Returns the descriptors of the arrays of the checkpoint file, in the order of their byte ranges.
    #[inline]
    pub fn arrays(&self) -> &[ArrayDescriptor] {
        &self.arrays
    }

    /// Serializes this manifest to JSON with one file record and one array record per line.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::InvalidManifest`] if the data type of an array has no safetensors tag.
    pub fn to_json(&self) -> Result<String, CheckpointError> {
        // Every value below is a string, a number, or a list of those, so serializing them cannot fail.
        let mut json = String::from("{\n");
        json.push_str(&format!("  \"repository\": {},\n", serde_json::to_string(&self.repository).unwrap()));
        json.push_str(&format!("  \"revision\": {},\n", serde_json::to_string(&self.revision).unwrap()));
        json.push_str("  \"files\": [");
        for (index, file) in self.files.iter().enumerate() {
            json.push_str(if index == 0 { "\n    " } else { ",\n    " });
            json.push_str(&serde_json::to_string(file).unwrap());
        }
        json.push_str(if self.files.is_empty() { "],\n" } else { "\n  ],\n" });
        json.push_str("  \"arrays\": [");
        for (index, array) in self.arrays.iter().enumerate() {
            json.push_str(if index == 0 { "\n    " } else { ",\n    " });
            json.push_str(&serde_json::to_string(&ArrayRecord::from_descriptor(array)?).unwrap());
        }
        json.push_str(if self.arrays.is_empty() { "]\n" } else { "\n  ]\n" });
        json.push_str("}\n");
        Ok(json)
    }

    /// Checks that `checkpoint` holds exactly the arrays that this manifest records, with the same data types,
    /// shapes, and byte ranges.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::ManifestMismatch`] that names the first recorded array that `checkpoint` lacks or
    /// describes differently, or otherwise the first array of `checkpoint` that this manifest does not record.
    pub fn validate_checkpoint<C: Checkpoint>(&self, checkpoint: &C) -> Result<(), CheckpointError> {
        for expected in &self.arrays {
            match checkpoint.array(expected.name()) {
                None => {
                    return Err(CheckpointError::ManifestMismatch {
                        message: format!("array `{}` is missing from the checkpoint", expected.name()),
                    });
                }
                Some(actual) if actual != expected => {
                    return Err(CheckpointError::ManifestMismatch {
                        message: format!(
                            "array `{}` is recorded as `{}` {:?} at bytes {}..{} but the checkpoint holds `{}` {:?} \
                             at bytes {}..{}",
                            expected.name(),
                            expected.data_type(),
                            expected.shape(),
                            expected.byte_range().start,
                            expected.byte_range().end,
                            actual.data_type(),
                            actual.shape(),
                            actual.byte_range().start,
                            actual.byte_range().end,
                        ),
                    });
                }
                Some(_) => {}
            }
        }

        // Every recorded array is present, so equal counts imply that the checkpoint holds no other array.
        if checkpoint.arrays().len() != self.arrays.len() {
            let recorded = self.arrays.iter().map(ArrayDescriptor::name).collect::<HashSet<_>>();
            let unexpected = checkpoint
                .arrays()
                .iter()
                .find(|array| !recorded.contains(array.name()))
                .map(ArrayDescriptor::name)
                .unwrap_or_default();
            return Err(CheckpointError::ManifestMismatch {
                message: format!("array `{unexpected}` of the checkpoint is not recorded in the manifest"),
            });
        }
        Ok(())
    }

    /// Checks that the file at `path` on the local file system has the size, and the SHA-256 digest if one is
    /// recorded, of the file that this manifest records at `release_path`. Checking a digest reads the whole file.
    ///
    /// # Parameters
    ///
    ///   - `release_path`: Path of the file relative to the root of the release, which selects the [`FileRecord`].
    ///   - `path`: Location of the downloaded file on the local file system.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::InvalidManifest`] if this manifest records no file at `release_path`,
    /// [`CheckpointError::Io`] if the file cannot be read, and [`CheckpointError::ManifestMismatch`] if its size or
    /// digest differs from the recorded one.
    pub fn validate_file<P: AsRef<Path>>(&self, release_path: &str, path: P) -> Result<(), CheckpointError> {
        let record = self.file(release_path).ok_or_else(|| CheckpointError::InvalidManifest {
            message: format!("file `{release_path}` is not recorded in the manifest"),
        })?;
        let display_path = path.as_ref().display().to_string();
        let io_error =
            |error: std::io::Error| CheckpointError::Io { path: display_path.clone(), message: error.to_string() };
        let mut file = File::open(path.as_ref()).map_err(io_error)?;
        let byte_count = file.metadata().map_err(io_error)?.len();
        if byte_count != record.byte_count() {
            return Err(CheckpointError::ManifestMismatch {
                message: format!(
                    "file `{release_path}` is recorded with {} bytes but `{display_path}` holds {byte_count} bytes",
                    record.byte_count(),
                ),
            });
        }

        if let Some(expected) = record.sha256() {
            let mut hasher = Sha256::new();
            let mut buffer = vec![0u8; 1 << 20];
            loop {
                let count = file.read(&mut buffer).map_err(io_error)?;
                if count == 0 {
                    break;
                }
                hasher.update(&buffer[..count]);
            }
            let actual = hasher.finalize().iter().map(|byte| format!("{byte:02x}")).collect::<String>();
            if actual != expected {
                return Err(CheckpointError::ManifestMismatch {
                    message: format!(
                        "file `{release_path}` is recorded with SHA-256 digest `{expected}` but `{display_path}` has \
                         digest `{actual}`",
                    ),
                });
            }
        }
        Ok(())
    }
}

/// Serialized form of a [`Manifest`].
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SerializedManifest {
    /// Name of the repository that hosts the release.
    repository: String,

    /// Immutable revision of the release.
    revision: String,

    /// Records of the files of the release.
    files: Vec<FileRecord>,

    /// Records of the arrays of the checkpoint file.
    arrays: Vec<ArrayRecord>,
}

#[cfg(test)]
mod tests {
    use indoc::{formatdoc, indoc};
    use pretty_assertions::assert_eq;
    use ryft_core::DataType;

    #[cfg(feature = "safetensors")]
    use crate::arrays::tests::test_checkpoint;

    use super::*;

    /// SHA-256 digest of the bytes `hello`.
    const HELLO_SHA256: &str = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";

    /// Returns a manifest with two files and the array table of `test_checkpoint`.
    fn test_manifest() -> Manifest {
        Manifest::new("organization/model", "0123abcd")
            .with_files(vec![
                FileRecord::new("config.json", 3),
                FileRecord::new("model.safetensors", 5).with_sha256(HELLO_SHA256),
            ])
            .with_arrays(vec![
                ArrayDescriptor::new("matrix", DataType::BF16, [2, 2], 0..8),
                ArrayDescriptor::new("stack", DataType::F32, [3, 1, 2], 8..32),
                ArrayDescriptor::new("scalar", DataType::U8, [], 32..33),
            ])
    }

    #[test]
    fn test_file_record() {
        let record = FileRecord::new("model.safetensors", 42);
        assert_eq!(record.path(), "model.safetensors");
        assert_eq!(record.byte_count(), 42);
        assert_eq!(record.sha256(), None);
        assert_eq!(record.with_sha256("ab").sha256(), Some("ab"));
    }

    #[test]
    fn test_array_record() {
        let descriptor = ArrayDescriptor::new("weights", DataType::F8E4M3FN, [2, 3], 4..10);
        let record = ArrayRecord::from_descriptor(&descriptor).unwrap();
        assert_eq!(
            serde_json::to_string(&record).unwrap(),
            r#"{"name":"weights","data_type":"F8_E4M3","shape":[2,3],"byte_range":[4,10]}"#,
        );
        assert_eq!(record.to_descriptor(), Ok(descriptor));

        // Data types without a safetensors tag cannot be recorded, and unknown tags cannot be resolved.
        assert_eq!(
            ArrayRecord::from_descriptor(&ArrayDescriptor::new("narrow", DataType::F4E2M1FN, [1], 0..1)),
            Err(CheckpointError::InvalidManifest {
                message: "array `narrow` has data type `f4e2m1fn`, which has no safetensors data type tag".to_string(),
            }),
        );
        let unknown: ArrayRecord =
            serde_json::from_str(r#"{"name":"a","data_type":"F4","shape":[],"byte_range":[0,1]}"#).unwrap();
        assert_eq!(
            unknown.to_descriptor(),
            Err(CheckpointError::InvalidManifest {
                message: "array `a` has unsupported data type tag `F4`".to_string(),
            }),
        );
    }

    #[test]
    fn test_manifest_new() {
        let manifest = Manifest::new("organization/model", "0123abcd");
        assert_eq!(manifest.repository(), "organization/model");
        assert_eq!(manifest.revision(), "0123abcd");
        assert_eq!(manifest.files(), &[]);
        assert_eq!(manifest.arrays(), &[]);
        assert_eq!(
            manifest.to_json().unwrap(),
            indoc! {r#"
                {
                  "repository": "organization/model",
                  "revision": "0123abcd",
                  "files": [],
                  "arrays": []
                }
            "#},
        );
    }

    #[cfg(feature = "safetensors")]
    #[test]
    fn test_manifest_from_checkpoint() {
        let checkpoint = test_checkpoint();
        let manifest = Manifest::from_checkpoint("organization/model", "0123abcd", &checkpoint);
        assert_eq!(manifest.arrays(), checkpoint.arrays());
        assert_eq!(manifest.files(), &[]);
    }

    #[test]
    fn test_manifest_from_json() {
        let manifest = test_manifest();
        assert_eq!(Manifest::from_json(&manifest.to_json().unwrap()), Ok(manifest));

        assert!(matches!(
            Manifest::from_json("{}"),
            Err(CheckpointError::InvalidManifest { message }) if message.starts_with("missing field `repository`"),
        ));
        let repeated_file = concat!(
            r#"{"repository":"r","revision":"v","files":[{"path":"a","byte_count":1},"#,
            r#"{"path":"a","byte_count":2}],"arrays":[]}"#,
        );
        assert_eq!(
            Manifest::from_json(repeated_file),
            Err(CheckpointError::InvalidManifest { message: "file `a` is recorded more than once".to_string() }),
        );
        let repeated_array = concat!(
            r#"{"repository":"r","revision":"v","files":[],"arrays":["#,
            r#"{"name":"a","data_type":"U8","shape":[1],"byte_range":[0,1]},"#,
            r#"{"name":"a","data_type":"U8","shape":[1],"byte_range":[1,2]}]}"#,
        );
        assert_eq!(
            Manifest::from_json(repeated_array),
            Err(CheckpointError::InvalidManifest { message: "array `a` is recorded more than once".to_string() }),
        );
    }

    #[test]
    fn test_manifest_with_files() {
        let manifest = test_manifest();
        assert_eq!(manifest.files().len(), 2);
        assert_eq!(manifest.file("config.json"), Some(&FileRecord::new("config.json", 3)));
        assert_eq!(manifest.file("missing.json"), None);
    }

    #[test]
    fn test_manifest_with_arrays() {
        let arrays = vec![ArrayDescriptor::new("a", DataType::U8, [1], 0..1)];
        assert_eq!(Manifest::new("r", "v").with_arrays(arrays.clone()).arrays(), arrays.as_slice());
    }

    #[test]
    fn test_manifest_to_json() {
        assert_eq!(
            test_manifest().to_json().unwrap(),
            formatdoc! {r#"
                {{
                  "repository": "organization/model",
                  "revision": "0123abcd",
                  "files": [
                    {{"path":"config.json","byte_count":3}},
                    {{"path":"model.safetensors","byte_count":5,"sha256":"{HELLO_SHA256}"}}
                  ],
                  "arrays": [
                    {{"name":"matrix","data_type":"BF16","shape":[2,2],"byte_range":[0,8]}},
                    {{"name":"stack","data_type":"F32","shape":[3,1,2],"byte_range":[8,32]}},
                    {{"name":"scalar","data_type":"U8","shape":[],"byte_range":[32,33]}}
                  ]
                }}
            "#},
        );
    }

    #[cfg(feature = "safetensors")]
    #[test]
    fn test_manifest_validate_checkpoint() {
        let checkpoint = test_checkpoint();
        let manifest = test_manifest();
        assert_eq!(manifest.validate_checkpoint(&checkpoint), Ok(()));

        // A recorded array that the checkpoint lacks, an array that it describes differently, and an array that the
        // manifest does not record are each reported by name.
        let mut arrays = manifest.arrays().to_vec();
        arrays.push(ArrayDescriptor::new("extra", DataType::U8, [1], 33..34));
        assert_eq!(
            manifest.clone().with_arrays(arrays).validate_checkpoint(&checkpoint),
            Err(CheckpointError::ManifestMismatch {
                message: "array `extra` is missing from the checkpoint".to_string(),
            }),
        );
        let mut arrays = manifest.arrays().to_vec();
        arrays[1] = ArrayDescriptor::new("stack", DataType::F32, [3, 2, 1], 8..32);
        assert_eq!(
            manifest.clone().with_arrays(arrays).validate_checkpoint(&checkpoint),
            Err(CheckpointError::ManifestMismatch {
                message: "array `stack` is recorded as `f32` [3, 2, 1] at bytes 8..32 but the checkpoint holds `f32` \
                          [3, 1, 2] at bytes 8..32"
                    .to_string(),
            }),
        );
        let arrays = manifest.arrays()[..2].to_vec();
        assert_eq!(
            manifest.clone().with_arrays(arrays).validate_checkpoint(&checkpoint),
            Err(CheckpointError::ManifestMismatch {
                message: "array `scalar` of the checkpoint is not recorded in the manifest".to_string(),
            }),
        );
    }

    #[test]
    fn test_manifest_validate_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.safetensors");
        std::fs::write(&path, b"hello").unwrap();
        let manifest = test_manifest();
        assert_eq!(manifest.validate_file("model.safetensors", &path), Ok(()));

        // Unrecorded files, size mismatches, and digest mismatches are reported separately.
        assert_eq!(
            manifest.validate_file("other.bin", &path),
            Err(CheckpointError::InvalidManifest {
                message: "file `other.bin` is not recorded in the manifest".to_string(),
            }),
        );
        assert_eq!(
            manifest.validate_file("config.json", &path),
            Err(CheckpointError::ManifestMismatch {
                message: format!("file `config.json` is recorded with 3 bytes but `{}` holds 5 bytes", path.display(),),
            }),
        );
        std::fs::write(&path, b"world").unwrap();
        assert_eq!(
            manifest.validate_file("model.safetensors", &path),
            Err(CheckpointError::ManifestMismatch {
                message: format!(
                    "file `model.safetensors` is recorded with SHA-256 digest `{HELLO_SHA256}` but `{}` has digest \
                     `486ea46224d1bb4fb680f34f7c9ad96a8f24ec88be73ea8e5a6c65260e9cb8a7`",
                    path.display(),
                ),
            }),
        );
        let missing = directory.path().join("missing.safetensors");
        assert!(matches!(
            manifest.validate_file("model.safetensors", &missing),
            Err(CheckpointError::Io { path, .. }) if path == missing.display().to_string(),
        ));
    }
}
