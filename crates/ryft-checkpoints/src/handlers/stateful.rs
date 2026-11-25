//! Handler for checkpointables that save and load themselves, mirroring the `StatefulCheckpointableHandler` of Orbax.
//!
//! A [`StatefulHandler`] delegates to the [`StatefulCheckpointable`] implementation of its checkpointable, which owns
//! its storage format. Loading replaces the state of an existing object in place, so the abstract checkpointable of a
//! load is a mutable reference to that object, and the load returns nothing.

use crate::errors::CheckpointError;
use crate::filesystems::CheckpointPath;
use crate::handlers::{CheckpointableHandler, CheckpointableLoader, CheckpointableSaver, StatefulCheckpointable};
use crate::multiprocessing::MultiprocessingOptions;

// TODO(eaplatanios): Review this module.

/// [`CheckpointableHandler`] for [`StatefulCheckpointable`]s, which mirrors the `StatefulCheckpointableHandler` of
/// Orbax. Like every handler of this crate, it saves in the primary process only.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct StatefulHandler;

impl CheckpointableHandler for StatefulHandler {
    #[inline]
    fn name(&self) -> String {
        "StatefulHandler".to_string()
    }
}

impl<T: ?Sized + StatefulCheckpointable> CheckpointableSaver<T> for StatefulHandler {
    fn save(
        &self,
        directory: &CheckpointPath,
        checkpointable: &T,
        multiprocessing: &MultiprocessingOptions,
    ) -> Result<(), CheckpointError> {
        if !multiprocessing.is_primary() {
            return Ok(());
        }
        checkpointable.save(directory)
    }
}

impl<'o, T: ?Sized + StatefulCheckpointable> CheckpointableLoader<&'o mut T> for StatefulHandler {
    type Output = ();

    fn load(&self, directory: &CheckpointPath, abstract_checkpointable: &'o mut T) -> Result<(), CheckpointError> {
        abstract_checkpointable.load(directory)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    /// Counter that stores its value as decimal text.
    #[derive(Debug, PartialEq)]
    struct Counter(u64);

    impl StatefulCheckpointable for Counter {
        fn save(&self, directory: &CheckpointPath) -> Result<(), CheckpointError> {
            directory.join("counter").write(self.0.to_string().as_bytes())
        }

        fn load(&mut self, directory: &CheckpointPath) -> Result<(), CheckpointError> {
            let bytes = directory.join("counter").read()?;
            let text = String::from_utf8_lossy(&bytes);
            self.0 = text.parse().map_err(|_| CheckpointError::InvalidFile {
                path: directory.join("counter").to_string(),
                message: format!("`{text}` is not a counter"),
            })?;
            Ok(())
        }
    }

    #[test]
    fn test_stateful_handler_save() {
        let temporary = tempfile::tempdir().unwrap();
        let directory = CheckpointPath::from(temporary.path());
        assert_eq!(StatefulHandler.name(), "StatefulHandler");
        assert_eq!(StatefulHandler.save(&directory, &Counter(7), &MultiprocessingOptions::default()), Ok(()));
        assert_eq!(std::fs::read_to_string(temporary.path().join("counter")).unwrap(), "7");

        // Processes other than the primary one write nothing.
        let other = CheckpointPath::from(temporary.path().join("other"));
        other.create_directory().unwrap();
        assert_eq!(StatefulHandler.save(&other, &Counter(8), &MultiprocessingOptions::new(1, 2).unwrap()), Ok(()));
        assert_eq!(other.list(), Ok(Vec::new()));
    }

    #[test]
    fn test_stateful_handler_load() {
        let temporary = tempfile::tempdir().unwrap();
        let directory = CheckpointPath::from(temporary.path());
        StatefulHandler.save(&directory, &Counter(7), &MultiprocessingOptions::default()).unwrap();
        let mut counter = Counter(0);
        assert_eq!(StatefulHandler.load(&directory, &mut counter), Ok(()));
        assert_eq!(counter, Counter(7));

        // Errors of the checkpointable are returned unchanged.
        std::fs::write(temporary.path().join("counter"), "seven").unwrap();
        assert_eq!(
            StatefulHandler.load(&directory, &mut counter),
            Err(CheckpointError::InvalidFile {
                path: directory.join("counter").to_string(),
                message: "`seven` is not a counter".to_string(),
            }),
        );
    }
}
