//! Metadata files that describe checkpoints, mirroring the metadata files of Orbax, and the metadata that loading
//! reports.
//!
//! A checkpoint directory carries two kinds of metadata, each stored as one JSON file:
//!
//!   - [`StepMetadata`] describes one finalized checkpoint and is stored in the [`STEP_METADATA_FILE_NAME`] file of its
//!     directory, which mirrors the `_CHECKPOINT_METADATA` file of Orbax. It records which handler saved every
//!     checkpointable, the metrics that the checkpoint was saved with, when saving started and when it was committed,
//!     and free-form custom metadata. Its presence is what marks a checkpoint directory as finalized (see
//!     [`is_checkpoint_finalized`](crate::paths::is_checkpoint_finalized)), because
//!     [`save_checkpointables`](crate::saving::save_checkpointables) writes it only after every checkpointable was
//!     saved.
//!   - [`ParameterizedMetadata`] describes the leaves of one saved parameter structure and is stored in the
//!     [`PARAMETERIZED_METADATA_FILE_NAME`] file of its checkpointable directory, which mirrors the `_METADATA` file
//!     that the Orbax `PyTreeHandler` writes. It lets a reader learn what a checkpoint holds without opening its
//!     arrays.
//!
//! The JSON keys of [`StepMetadata`] are those of Orbax (e.g., `item_handlers` and `init_timestamp_nsecs`), so that
//! tools that inspect Orbax checkpoints can read the step metadata of Ryft checkpoints as well.
//!
//! A [`CheckpointMetadata`] is what [`metadata`](crate::loading::metadata) and
//! [`checkpointables_metadata`](crate::loading::checkpointables_metadata) return, which mirrors the
//! `CheckpointMetadata` of the Orbax v1 API. It pairs the path, the timestamps, and the custom metadata of a checkpoint
//! with the metadata of what was asked for. The collection of checkpoints of a training run has metadata of its own,
//! which [`training::RootMetadata`](crate::training::RootMetadata) describes.

use std::collections::BTreeMap;

use serde::de::{DeserializeOwned, Error as DeserializationError};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::errors::CheckpointError;
use crate::filesystems::CheckpointPath;

// TODO(eaplatanios): Review this module.

/// Name of the file in a step directory that holds its [`StepMetadata`].
pub const STEP_METADATA_FILE_NAME: &str = "_CHECKPOINT_METADATA";

/// Name of the file in a checkpointable directory that holds the [`ParameterizedMetadata`] of a saved parameter
/// structure.
pub const PARAMETERIZED_METADATA_FILE_NAME: &str = "_METADATA";

/// Metadata of one checkpoint as loading reports it, which mirrors the `CheckpointMetadata` of the Orbax v1 API. The
/// metadata `M` is that of what was asked for (e.g., the [`ParameterizedMetadata`] of the `state` checkpointable).
#[derive(Clone, Debug, PartialEq)]
pub struct CheckpointMetadata<M> {
    /// Path of the checkpoint.
    path: CheckpointPath,

    /// Metadata of what was asked for.
    metadata: M,

    /// Time at which saving started, in nanoseconds since the Unix epoch.
    initialization_timestamp_nanoseconds: u64,

    /// Time at which the checkpoint was committed, in nanoseconds since the Unix epoch, if its step metadata records
    /// it.
    commit_timestamp_nanoseconds: Option<u64>,

    /// Free-form custom metadata.
    custom_metadata: serde_json::Value,

    /// Metrics that the checkpoint was saved with, if any.
    metrics: Option<BTreeMap<String, f64>>,
}

impl<M> CheckpointMetadata<M> {
    /// Creates a new [`CheckpointMetadata`] for the checkpoint at `path` with the timestamps, the custom metadata,
    /// and the metrics that its `step_metadata` records.
    ///
    /// # Parameters
    ///
    ///   - `path`: Path of the checkpoint.
    ///   - `metadata`: Metadata of what was asked for.
    ///   - `step_metadata`: Step metadata of the checkpoint.
    pub fn new(path: CheckpointPath, metadata: M, step_metadata: &StepMetadata) -> Self {
        Self {
            path,
            metadata,
            initialization_timestamp_nanoseconds: step_metadata.initialization_timestamp_nanoseconds(),
            commit_timestamp_nanoseconds: step_metadata.commit_timestamp_nanoseconds(),
            custom_metadata: step_metadata.custom_metadata().clone(),
            metrics: step_metadata.metrics().cloned(),
        }
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

    /// Returns the time at which the checkpoint was committed, in nanoseconds since the Unix epoch, if its step
    /// metadata records it, which only checkpoints that were finalized by renaming do.
    #[inline]
    pub fn commit_timestamp_nanoseconds(&self) -> Option<u64> {
        self.commit_timestamp_nanoseconds
    }

    /// Returns the free-form custom metadata.
    #[inline]
    pub fn custom_metadata(&self) -> &serde_json::Value {
        &self.custom_metadata
    }

    /// Returns the metrics that the checkpoint was saved with, if any.
    #[inline]
    pub fn metrics(&self) -> Option<&BTreeMap<String, f64>> {
        self.metrics.as_ref()
    }

    /// Returns the metadata of what was asked for, consuming this metadata.
    #[inline]
    pub fn into_metadata(self) -> M {
        self.metadata
    }
}

/// Metadata of one finalized checkpoint, stored in the [`STEP_METADATA_FILE_NAME`] file of its directory.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StepMetadata {
    /// Name of the handler of every checkpointable, keyed by the name of the checkpointable, which Orbax stores under
    /// the key `item_handlers`.
    #[serde(rename = "item_handlers")]
    checkpointable_handlers: BTreeMap<String, String>,

    /// Metrics that the checkpoint was saved with, if any. Values that are not finite are stored by name (refer to
    /// [`StoredMetric`]).
    #[serde(default, serialize_with = "serialize_metrics", deserialize_with = "deserialize_metrics")]
    metrics: Option<BTreeMap<String, f64>>,

    /// Time at which saving started, in nanoseconds since the Unix epoch.
    #[serde(rename = "init_timestamp_nsecs")]
    initialization_timestamp_nanoseconds: u64,

    /// Time at which the checkpoint was committed, in nanoseconds since the Unix epoch, if it was committed.
    #[serde(default, rename = "commit_timestamp_nsecs")]
    commit_timestamp_nanoseconds: Option<u64>,

    /// Free-form custom metadata.
    #[serde(default)]
    custom_metadata: serde_json::Value,
}

impl StepMetadata {
    /// Creates a new [`StepMetadata`] without metrics, commit timestamp, and custom metadata (i.e., with
    /// [`serde_json::Value::Null`] custom metadata).
    ///
    /// # Parameters
    ///
    ///   - `checkpointable_handlers`: Name of the handler of every checkpointable, keyed by the name of the
    ///     checkpointable.
    ///   - `initialization_timestamp_nanoseconds`: Time at which saving started, in nanoseconds since the Unix epoch.
    #[inline]
    pub fn new(checkpointable_handlers: BTreeMap<String, String>, initialization_timestamp_nanoseconds: u64) -> Self {
        Self {
            checkpointable_handlers,
            metrics: None,
            initialization_timestamp_nanoseconds,
            commit_timestamp_nanoseconds: None,
            custom_metadata: serde_json::Value::Null,
        }
    }

    /// Returns this metadata with the provided metrics.
    #[inline]
    pub fn with_metrics(mut self, metrics: Option<BTreeMap<String, f64>>) -> Self {
        self.metrics = metrics;
        self
    }

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

    /// Returns the name of the handler of every checkpointable, keyed by the name of the checkpointable.
    #[inline]
    pub fn checkpointable_handlers(&self) -> &BTreeMap<String, String> {
        &self.checkpointable_handlers
    }

    /// Returns the metrics that the checkpoint was saved with, if any.
    #[inline]
    pub fn metrics(&self) -> Option<&BTreeMap<String, f64>> {
        self.metrics.as_ref()
    }

    /// Returns the time at which saving started, in nanoseconds since the Unix epoch.
    #[inline]
    pub fn initialization_timestamp_nanoseconds(&self) -> u64 {
        self.initialization_timestamp_nanoseconds
    }

    /// Returns the time at which the checkpoint was committed, in nanoseconds since the Unix epoch, if it was.
    #[inline]
    pub fn commit_timestamp_nanoseconds(&self) -> Option<u64> {
        self.commit_timestamp_nanoseconds
    }

    /// Returns the free-form custom metadata.
    #[inline]
    pub fn custom_metadata(&self) -> &serde_json::Value {
        &self.custom_metadata
    }

    /// Writes this metadata to the [`STEP_METADATA_FILE_NAME`] file of `directory`, replacing an existing file.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::Io`] if the file cannot be written.
    pub fn write(&self, directory: &CheckpointPath) -> Result<(), CheckpointError> {
        write_json(&directory.join(STEP_METADATA_FILE_NAME), self)
    }

    /// Reads the metadata from the [`STEP_METADATA_FILE_NAME`] file of `directory`.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::Io`] if the file cannot be read, and [`CheckpointError::InvalidFile`] if it does not
    /// hold valid step metadata.
    pub fn read(directory: &CheckpointPath) -> Result<Self, CheckpointError> {
        read_json(&directory.join(STEP_METADATA_FILE_NAME))
    }
}

/// Value of a metric in the JSON of [`StepMetadata`]. JSON has no numbers that are not finite, and a diverged training
/// run must still be able to save a checkpoint with the metrics that it has, so such values are stored as the names
/// that Python, and therefore Orbax, writes for them (i.e., `NaN`, `Infinity`, and `-Infinity`).
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum StoredMetric {
    Number(f64),
    Name(String),
}

/// Serializes the metrics of a [`StepMetadata`] with their values as [`StoredMetric`]s.
fn serialize_metrics<S: Serializer>(metrics: &Option<BTreeMap<String, f64>>, serializer: S) -> Result<S::Ok, S::Error> {
    let stored = metrics.as_ref().map(|metrics| {
        metrics
            .iter()
            .map(|(name, value)| {
                let value = match *value {
                    value if value.is_nan() => StoredMetric::Name("NaN".to_string()),
                    f64::INFINITY => StoredMetric::Name("Infinity".to_string()),
                    f64::NEG_INFINITY => StoredMetric::Name("-Infinity".to_string()),
                    value => StoredMetric::Number(value),
                };
                (name.as_str(), value)
            })
            .collect::<BTreeMap<_, _>>()
    });
    stored.serialize(serializer)
}

/// Deserializes the metrics of a [`StepMetadata`] from values that are [`StoredMetric`]s.
fn deserialize_metrics<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<BTreeMap<String, f64>>, D::Error> {
    let Some(stored) = Option::<BTreeMap<String, StoredMetric>>::deserialize(deserializer)? else {
        return Ok(None);
    };
    let mut metrics = BTreeMap::new();
    for (name, value) in stored {
        let value = match value {
            StoredMetric::Number(value) => value,
            StoredMetric::Name(value) => match value.as_str() {
                "NaN" => f64::NAN,
                "Infinity" => f64::INFINITY,
                "-Infinity" => f64::NEG_INFINITY,
                _ => return Err(D::Error::custom(format!("metric `{name}` has the invalid value `{value}`"))),
            },
        };
        metrics.insert(name, value);
    }
    Ok(Some(metrics))
}

/// Metadata of one leaf of a saved parameter structure.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeafMetadata {
    /// [`ParameterPath`](ryft_core::ParameterPath) of the leaf, rendered with its `Display` implementation (e.g.,
    /// `$.layers[2].weight`), which is also the name of the array that stores the leaf.
    path: String,

    /// Safetensors data type tag of the stored leaf (e.g., `BF16`).
    data_type: String,

    /// Extents of the dimensions of the stored leaf.
    shape: Vec<usize>,
}

impl LeafMetadata {
    /// Creates a new [`LeafMetadata`].
    ///
    /// # Parameters
    ///
    ///   - `path`: Rendered [`ParameterPath`](ryft_core::ParameterPath) of the leaf.
    ///   - `data_type`: Safetensors data type tag of the stored leaf (e.g., `BF16`).
    ///   - `shape`: Extents of the dimensions of the stored leaf.
    #[inline]
    pub fn new<P: Into<String>, D: Into<String>, S: Into<Vec<usize>>>(path: P, data_type: D, shape: S) -> Self {
        Self { path: path.into(), data_type: data_type.into(), shape: shape.into() }
    }

    /// Returns the rendered [`ParameterPath`](ryft_core::ParameterPath) of the leaf.
    #[inline]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Returns the safetensors data type tag of the stored leaf.
    #[inline]
    pub fn data_type(&self) -> &str {
        &self.data_type
    }

    /// Returns the extents of the dimensions of the stored leaf.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

/// Metadata of one saved parameter structure, stored in the [`PARAMETERIZED_METADATA_FILE_NAME`] file of its
/// checkpointable directory, which mirrors the `PyTreeMetadata` of Orbax.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterizedMetadata {
    /// Name of the [`ArrayFormat`](crate::formats::ArrayFormat) that stores the leaves (e.g., `safetensors`).
    format: String,

    /// Metadata of every leaf, in the order of [`Parameterized::named_parameters`](ryft_core::Parameterized).
    leaves: Vec<LeafMetadata>,

    /// Free-form custom metadata.
    #[serde(default)]
    custom_metadata: serde_json::Value,
}

impl ParameterizedMetadata {
    /// Creates a new [`ParameterizedMetadata`] for leaves that the format named `format` stores, without custom
    /// metadata (i.e., with [`serde_json::Value::Null`] custom metadata).
    #[inline]
    pub fn new<F: Into<String>>(format: F, leaves: Vec<LeafMetadata>) -> Self {
        Self { format: format.into(), leaves, custom_metadata: serde_json::Value::Null }
    }

    /// Returns this metadata with the provided free-form custom metadata.
    #[inline]
    pub fn with_custom_metadata(mut self, custom_metadata: serde_json::Value) -> Self {
        self.custom_metadata = custom_metadata;
        self
    }

    /// Returns the name of the [`ArrayFormat`](crate::formats::ArrayFormat) that stores the leaves.
    #[inline]
    pub fn format(&self) -> &str {
        &self.format
    }

    /// Returns the metadata of every leaf, in the order in which the leaves were saved.
    #[inline]
    pub fn leaves(&self) -> &[LeafMetadata] {
        &self.leaves
    }

    /// Returns the free-form custom metadata.
    #[inline]
    pub fn custom_metadata(&self) -> &serde_json::Value {
        &self.custom_metadata
    }

    /// Writes this metadata to the [`PARAMETERIZED_METADATA_FILE_NAME`] file of `directory`, replacing an existing
    /// file.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::Io`] if the file cannot be written.
    pub fn write(&self, directory: &CheckpointPath) -> Result<(), CheckpointError> {
        write_json(&directory.join(PARAMETERIZED_METADATA_FILE_NAME), self)
    }

    /// Reads the metadata from the [`PARAMETERIZED_METADATA_FILE_NAME`] file of `directory`.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::Io`] if the file cannot be read, and [`CheckpointError::InvalidFile`] if it does not
    /// hold valid parameterized metadata.
    pub fn read(directory: &CheckpointPath) -> Result<Self, CheckpointError> {
        read_json(&directory.join(PARAMETERIZED_METADATA_FILE_NAME))
    }
}

/// Writes `value` as pretty JSON to the file at `path`, replacing an existing file.
pub(crate) fn write_json<T: Serialize + ?Sized>(path: &CheckpointPath, value: &T) -> Result<(), CheckpointError> {
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| CheckpointError::InvalidFile { path: path.to_string(), message: error.to_string() })?;
    path.write(&bytes)
}

/// Reads the JSON file at `path` as a `T`.
pub(crate) fn read_json<T: DeserializeOwned>(path: &CheckpointPath) -> Result<T, CheckpointError> {
    serde_json::from_slice(&path.read()?)
        .map_err(|error| CheckpointError::InvalidFile { path: path.to_string(), message: error.to_string() })
}

#[cfg(test)]
mod tests {
    use std::fs;

    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::*;

    #[test]
    fn test_checkpoint_metadata() {
        let step_metadata = StepMetadata::new(BTreeMap::new(), 10)
            .with_commit_timestamp_nanoseconds(20)
            .with_custom_metadata(json!("note"));
        let metadata = CheckpointMetadata::new(CheckpointPath::local("root/1"), 3, &step_metadata);
        assert_eq!(metadata.path(), &CheckpointPath::local("root/1"));
        assert_eq!(metadata.metadata(), &3);
        assert_eq!(metadata.initialization_timestamp_nanoseconds(), 10);
        assert_eq!(metadata.commit_timestamp_nanoseconds(), Some(20));
        assert_eq!(metadata.custom_metadata(), &json!("note"));
        assert_eq!(metadata.into_metadata(), 3);
    }

    #[test]
    fn test_step_metadata() {
        let metadata = StepMetadata::new(BTreeMap::from([("state".to_string(), "Handler".to_string())]), 10)
            .with_metrics(Some(BTreeMap::from([("loss".to_string(), 0.5)])))
            .with_commit_timestamp_nanoseconds(20)
            .with_custom_metadata(json!({"epoch": 3}));
        assert_eq!(metadata.checkpointable_handlers(), &BTreeMap::from([("state".to_string(), "Handler".to_string())]));
        assert_eq!(metadata.metrics(), Some(&BTreeMap::from([("loss".to_string(), 0.5)])));
        assert_eq!(metadata.initialization_timestamp_nanoseconds(), 10);
        assert_eq!(metadata.commit_timestamp_nanoseconds(), Some(20));
        assert_eq!(metadata.custom_metadata(), &json!({"epoch": 3}));

        // The JSON keys are those of Orbax.
        assert_eq!(
            serde_json::to_value(&metadata).unwrap(),
            json!({
                "item_handlers": {"state": "Handler"},
                "metrics": {"loss": 0.5},
                "init_timestamp_nsecs": 10,
                "commit_timestamp_nsecs": 20,
                "custom_metadata": {"epoch": 3},
            }),
        );
    }

    #[test]
    fn test_step_metadata_write() {
        let directory = tempfile::tempdir().unwrap();
        let directory = CheckpointPath::from(directory.path());
        let metadata = StepMetadata::new(BTreeMap::from([("a".to_string(), "Handler".to_string())]), 1);
        assert_eq!(metadata.write(&directory), Ok(()));
        assert_eq!(StepMetadata::read(&directory), Ok(metadata));
    }

    #[test]
    fn test_step_metadata_read() {
        // Optional fields may be missing, as in files that other tools wrote.
        let temporary = tempfile::tempdir().unwrap();
        let directory = CheckpointPath::from(temporary.path());
        let path = temporary.path().join(STEP_METADATA_FILE_NAME);
        fs::write(&path, r#"{"item_handlers": {}, "init_timestamp_nsecs": 5}"#).unwrap();
        assert_eq!(StepMetadata::read(&directory), Ok(StepMetadata::new(BTreeMap::new(), 5)));

        // Metrics that are not finite are stored by name, so that the checkpoint of a diverged step stays readable.
        let metrics = BTreeMap::from([
            ("loss".to_string(), f64::NAN),
            ("norm".to_string(), f64::INFINITY),
            ("score".to_string(), f64::NEG_INFINITY),
            ("step".to_string(), 2.0),
        ]);
        let metadata = StepMetadata::new(BTreeMap::new(), 5).with_metrics(Some(metrics));
        assert_eq!(metadata.write(&directory), Ok(()));
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&fs::read_to_string(&path).unwrap()).unwrap()["metrics"],
            json!({"loss": "NaN", "norm": "Infinity", "score": "-Infinity", "step": 2.0}),
        );
        let metrics = StepMetadata::read(&directory).unwrap().metrics().cloned().unwrap();
        assert!(metrics["loss"].is_nan());
        assert_eq!((metrics["norm"], metrics["score"], metrics["step"]), (f64::INFINITY, f64::NEG_INFINITY, 2.0));
        fs::write(&path, r#"{"item_handlers": {}, "metrics": {"loss": "large"}, "init_timestamp_nsecs": 5}"#).unwrap();
        assert!(matches!(
            StepMetadata::read(&directory),
            Err(CheckpointError::InvalidFile { message, .. })
                if message.starts_with("metric `loss` has the invalid value `large`"),
        ));

        // Invalid and missing files are reported with their paths.
        fs::write(&path, "{}").unwrap();
        assert!(matches!(
            StepMetadata::read(&directory),
            Err(CheckpointError::InvalidFile { path: invalid, message })
                if invalid == path.display().to_string() && message.starts_with("missing field `item_handlers`"),
        ));
        fs::remove_file(&path).unwrap();
        assert!(matches!(
            StepMetadata::read(&directory),
            Err(CheckpointError::Io { path: missing, .. }) if missing == path.display().to_string(),
        ));
    }

    #[test]
    fn test_leaf_metadata() {
        let metadata = LeafMetadata::new("$.0", "F32", [2, 3]);
        assert_eq!(metadata.path(), "$.0");
        assert_eq!(metadata.data_type(), "F32");
        assert_eq!(metadata.shape(), &[2, 3]);
    }

    #[test]
    fn test_parameterized_metadata_write() {
        let directory = tempfile::tempdir().unwrap();
        let metadata = ParameterizedMetadata::new(
            "raw",
            vec![LeafMetadata::new("$.0", "F32", [2]), LeafMetadata::new("$.1", "U8", [])],
        )
        .with_custom_metadata(json!("note"));
        assert_eq!(metadata.format(), "raw");
        assert_eq!(metadata.leaves(), &[LeafMetadata::new("$.0", "F32", [2]), LeafMetadata::new("$.1", "U8", [])]);
        assert_eq!(metadata.custom_metadata(), &json!("note"));
        let directory = CheckpointPath::from(directory.path());
        assert_eq!(metadata.write(&directory), Ok(()));
        assert_eq!(ParameterizedMetadata::read(&directory), Ok(metadata));
    }
}
