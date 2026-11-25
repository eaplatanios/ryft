//! Handler for JSON checkpointables, mirroring the `JsonHandler` of Orbax.
//!
//! A [`JsonHandler`] stores one [`serde_json::Value`] as a pretty printed JSON file in the checkpointable directory,
//! which suits small self describing checkpointables such as training configurations, data iterator states, or step
//! counters.

use crate::errors::CheckpointError;
use crate::filesystems::CheckpointPath;
use crate::handlers::{CheckpointableHandler, CheckpointableLoader, CheckpointableSaver};
use crate::metadata::{read_json, write_json};
use crate::multiprocessing::MultiprocessingOptions;

// TODO(eaplatanios): Review this module.

/// [`CheckpointableHandler`] that stores one [`serde_json::Value`] as a pretty printed JSON file, which mirrors the
/// `JsonHandler` of Orbax. Loading takes no abstract checkpointable (i.e., `()`) because JSON describes itself.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct JsonHandler {
    /// Name of the JSON file in the checkpointable directory.
    filename: String,
}

impl JsonHandler {
    /// Name of the JSON file that [`Self::new`] uses when no name is provided.
    pub const DEFAULT_FILENAME: &str = "metadata";

    /// Creates a new [`JsonHandler`] that stores its checkpointable in the file named `filename` of the
    /// checkpointable directory, or in [`Self::DEFAULT_FILENAME`] when `filename` is [`None`].
    #[inline]
    pub fn new(filename: Option<&str>) -> Self {
        Self { filename: filename.unwrap_or(Self::DEFAULT_FILENAME).to_string() }
    }

    /// Returns the name of the JSON file in the checkpointable directory.
    #[inline]
    pub fn filename(&self) -> &str {
        &self.filename
    }
}

impl Default for JsonHandler {
    fn default() -> Self {
        Self::new(None)
    }
}

impl CheckpointableHandler for JsonHandler {
    #[inline]
    fn name(&self) -> String {
        "JsonHandler".to_string()
    }
}

impl CheckpointableSaver<serde_json::Value> for JsonHandler {
    fn save(
        &self,
        directory: &CheckpointPath,
        checkpointable: &serde_json::Value,
        multiprocessing: &MultiprocessingOptions,
    ) -> Result<(), CheckpointError> {
        // Every process holds the whole value, so the primary process writes it.
        if !multiprocessing.is_primary() {
            return Ok(());
        }
        write_json(&directory.join(&self.filename), checkpointable)
    }
}

impl CheckpointableLoader<()> for JsonHandler {
    type Output = serde_json::Value;

    fn load(&self, directory: &CheckpointPath, abstract_checkpointable: ()) -> Result<Self::Output, CheckpointError> {
        let () = abstract_checkpointable;
        read_json(&directory.join(&self.filename))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use serde_json::json;

    #[cfg(feature = "object-store")]
    use crate::filesystems::objects::tests::test_object_store_root;

    use super::*;

    #[test]
    fn test_json_handler_new() {
        assert_eq!(JsonHandler::new(None).filename(), "metadata");
        assert_eq!(JsonHandler::new(Some("config.json")).filename(), "config.json");
        assert_eq!(JsonHandler::default(), JsonHandler::new(None));
        assert_eq!(JsonHandler::default().name(), "JsonHandler");
    }

    #[test]
    fn test_json_handler_save() {
        let temporary = tempfile::tempdir().unwrap();
        let directory = CheckpointPath::from(temporary.path());
        let handler = JsonHandler::new(Some("state.json"));
        let checkpointable = json!({"step": 3, "names": ["a", "b"]});
        assert_eq!(handler.save(&directory, &checkpointable, &MultiprocessingOptions::default()), Ok(()));
        assert_eq!(
            std::fs::read_to_string(temporary.path().join("state.json")).unwrap(),
            serde_json::to_string_pretty(&checkpointable).unwrap(),
        );
        assert_eq!(handler.load(&directory, ()), Ok(checkpointable));

        // Processes other than the primary one write nothing.
        let secondary = MultiprocessingOptions::new(1, 2).unwrap();
        assert_eq!(JsonHandler::new(Some("other.json")).save(&directory, &json!(1), &secondary), Ok(()));
        assert!(!temporary.path().join("other.json").exists());
    }

    #[test]
    fn test_json_handler_load() {
        let temporary = tempfile::tempdir().unwrap();
        let directory = CheckpointPath::from(temporary.path());
        let handler = JsonHandler::default();
        let path = temporary.path().join("metadata");
        assert!(matches!(
            handler.load(&directory, ()),
            Err(CheckpointError::Io { path: missing, .. }) if missing == path.display().to_string(),
        ));
        std::fs::write(&path, "{").unwrap();
        assert!(matches!(
            handler.load(&directory, ()),
            Err(CheckpointError::InvalidFile { path: invalid, .. }) if invalid == path.display().to_string(),
        ));
    }

    #[cfg(feature = "object-store")]
    #[test]
    fn test_json_handler_object_store() {
        let directory = test_object_store_root().join("checkpointable");
        let handler = JsonHandler::default();
        let checkpointable = json!({"remote": true});
        assert_eq!(handler.save(&directory, &checkpointable, &MultiprocessingOptions::default()), Ok(()));
        assert_eq!(directory.list(), Ok(vec!["metadata".to_string()]));
        assert_eq!(handler.load(&directory, ()), Ok(checkpointable));
    }
}
