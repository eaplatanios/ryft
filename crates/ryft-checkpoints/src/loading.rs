//! Loading of checkpoints, mirroring the `load`, `load_checkpointables`, `metadata`, and `checkpointables_metadata`
//! functions of the Orbax v1 API.
//!
//! Loading accepts finalized checkpoints only (see [`is_checkpoint_finalized`]), regardless of the [`Atomicity`] mode
//! that they were saved in. [`load`] loads the parameter structure of the
//! [`STATE_CHECKPOINTABLE_NAME`] checkpointable into the structure that an abstract structure of [`ArrayType`]s
//! describes, with the [`ParameterizedHandler`](crate::handlers::ParameterizedHandler) of a [`Context`].
//! [`load_checkpointable`] loads any one checkpointable with the [`CheckpointableLoader`] that the caller provides.
//! Orbax loads several checkpointables at once into a dictionary of values of any type. In Rust, every checkpointable
//! has its own output type, so checkpointables are loaded one at a time, which reads nothing of the others.
//!
//! [`metadata`] and [`checkpointables_metadata`] read what a checkpoint holds without loading it.
//!
//! [`Atomicity`]: crate::paths::Atomicity

use std::collections::BTreeMap;

use ryft_core::{ArrayType, Parameterized, ParameterizedFamily};

use crate::contexts::Context;
use crate::errors::CheckpointError;
use crate::filesystems::CheckpointPath;
use crate::handlers::{CheckpointableLoader, LeafHandler};
use crate::metadata::{CheckpointMetadata, ParameterizedMetadata, StepMetadata};
use crate::paths::is_checkpoint_finalized;
use crate::saving::{STATE_CHECKPOINTABLE_NAME, validate_checkpointable_name};

// TODO(eaplatanios): Review this module.

/// Loads the parameter structure that [`save`](crate::saving::save) saved as the checkpoint at `path` into the
/// structure that `abstract_state` describes, which mirrors `load` of Orbax. The structure is loaded from the
/// [`STATE_CHECKPOINTABLE_NAME`] checkpointable with the [`Context::parameterized_handler`] of `context`, whose
/// [`ParameterizedOptions`](crate::contexts::ParameterizedOptions) decide whether the load is partial, how leaves
/// are named, and whether data types are converted.
///
/// # Errors
///
/// Returns the errors of [`load_checkpointable`].
pub fn load<P: Into<CheckpointPath>, L: LeafHandler, S: Parameterized<ArrayType>>(
    path: P,
    abstract_state: S,
    context: &Context<L>,
) -> Result<S::To<L::Leaf>, CheckpointError>
where
    S::Family: ParameterizedFamily<L::Leaf>,
{
    load_checkpointable(path, STATE_CHECKPOINTABLE_NAME, &context.parameterized_handler(), abstract_state)
}

/// Loads the checkpointable named `name` of the finalized checkpoint at `path` with `handler`, as
/// `abstract_checkpointable` describes, which mirrors `load_checkpointables` of Orbax for a single checkpointable.
///
/// # Errors
///
/// Returns [`CheckpointError::InvalidArgument`] if `name` cannot name a checkpointable,
/// [`CheckpointError::UnfinalizedCheckpoint`] if `path` is not a finalized checkpoint (e.g., because it does not exist
/// or because its save was interrupted), [`CheckpointError::MissingCheckpoint`] if the checkpoint has no
/// checkpointable named `name`, the errors of [`is_checkpoint_finalized`] and [`StepMetadata::read`], and the errors
/// of the handler.
pub fn load_checkpointable<P: Into<CheckpointPath>, A, H: CheckpointableLoader<A>>(
    path: P,
    name: &str,
    handler: &H,
    abstract_checkpointable: A,
) -> Result<H::Output, CheckpointError> {
    validate_checkpointable_name(name)?;
    let path = path.into();
    let step_metadata = read_step_metadata(&path)?;
    if !step_metadata.checkpointable_handlers().contains_key(name) {
        return Err(CheckpointError::MissingCheckpoint {
            message: format!("checkpoint `{path}` has no checkpointable named `{name}`"),
        });
    }
    handler.load(&path.join(name), abstract_checkpointable)
}

/// Reads the [`ParameterizedMetadata`] of the parameter structure that [`save`](crate::saving::save) saved as the
/// finalized checkpoint at `path`, which mirrors `metadata` of Orbax. A checkpoint that
/// [`save_checkpointables`](crate::saving::save_checkpointables) saved without the [`STATE_CHECKPOINTABLE_NAME`]
/// checkpointable is described by [`checkpointables_metadata`] instead.
///
/// # Errors
///
/// Returns [`CheckpointError::UnfinalizedCheckpoint`] if `path` is not a finalized checkpoint,
/// [`CheckpointError::MissingCheckpoint`] if it has no [`STATE_CHECKPOINTABLE_NAME`] checkpointable, the errors of
/// [`is_checkpoint_finalized`] and [`StepMetadata::read`], and the errors of [`ParameterizedMetadata::read`].
pub fn metadata<P: Into<CheckpointPath>>(
    path: P,
) -> Result<CheckpointMetadata<ParameterizedMetadata>, CheckpointError> {
    let path = path.into();
    let step_metadata = read_step_metadata(&path)?;
    let metadata = state_metadata(&path, &step_metadata)?;
    Ok(CheckpointMetadata::new(path, metadata, &step_metadata))
}

/// Reads the name of the handler of every checkpointable of the finalized checkpoint at `path`, keyed by the name of
/// the checkpointable, which mirrors `checkpointables_metadata` of Orbax for what the step metadata records.
///
/// # Errors
///
/// Returns [`CheckpointError::UnfinalizedCheckpoint`] if `path` is not a finalized checkpoint, and the errors of
/// [`is_checkpoint_finalized`] and [`StepMetadata::read`].
pub fn checkpointables_metadata<P: Into<CheckpointPath>>(
    path: P,
) -> Result<CheckpointMetadata<BTreeMap<String, String>>, CheckpointError> {
    let path = path.into();
    let step_metadata = read_step_metadata(&path)?;
    let handlers = step_metadata.checkpointable_handlers().clone();
    Ok(CheckpointMetadata::new(path, handlers, &step_metadata))
}

/// Reads the [`ParameterizedMetadata`] of the [`STATE_CHECKPOINTABLE_NAME`] checkpointable of the checkpoint at
/// `path`, whose step metadata is `step_metadata`.
///
/// # Errors
///
/// Returns [`CheckpointError::MissingCheckpoint`] if the checkpoint has no such checkpointable, and the errors of
/// [`ParameterizedMetadata::read`].
pub(crate) fn state_metadata(
    path: &CheckpointPath,
    step_metadata: &StepMetadata,
) -> Result<ParameterizedMetadata, CheckpointError> {
    if !step_metadata.checkpointable_handlers().contains_key(STATE_CHECKPOINTABLE_NAME) {
        return Err(CheckpointError::MissingCheckpoint {
            message: format!("checkpoint `{path}` has no checkpointable named `{STATE_CHECKPOINTABLE_NAME}`"),
        });
    }
    ParameterizedMetadata::read(&path.join(STATE_CHECKPOINTABLE_NAME))
}

/// Reads the [`StepMetadata`] of the finalized checkpoint at `path`.
fn read_step_metadata(path: &CheckpointPath) -> Result<StepMetadata, CheckpointError> {
    if !is_checkpoint_finalized(path)? {
        return Err(CheckpointError::UnfinalizedCheckpoint { path: path.to_string() });
    }
    StepMetadata::read(path)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use pretty_assertions::assert_eq;
    use ryft_core::{Array, DataType};
    use serde_json::json;

    use crate::contexts::ParameterizedOptions;
    use crate::contexts::tests::test_context;
    use crate::handlers::JsonHandler;
    use crate::metadata::{LeafMetadata, STEP_METADATA_FILE_NAME};
    use crate::paths::COMMIT_SUCCESS_FILE;
    use crate::saving::{Checkpointables, SaveOptions, save, save_checkpointables};

    use super::*;

    #[test]
    fn test_load() {
        let root = tempfile::tempdir().unwrap();
        let context = test_context();
        let state = (Array::vector(vec![1.0f32, 2.0]).unwrap(), Array::scalar(3u8).unwrap());
        save(root.path().join("1"), &state, &SaveOptions::new(), &context).unwrap();
        let abstract_state = (ArrayType::new_static(DataType::F32, [2]), ArrayType::scalar(DataType::U8));
        assert_eq!(load(root.path().join("1"), abstract_state.clone(), &context), Ok(state.clone()));

        // The options of the context apply, so a part of the state loads only partially.
        let first = (ArrayType::new_static(DataType::F32, [2]),);
        assert!(matches!(
            load(root.path().join("1"), first.clone(), &context),
            Err(CheckpointError::InvalidArgument { .. }),
        ));
        let partial = context.with_parameterized_options(ParameterizedOptions::new().with_partial_load(true));
        assert_eq!(load(root.path().join("1"), first, &partial), Ok((state.0,)));
        assert_eq!(
            load(root.path().join("2"), abstract_state, &partial),
            Err(CheckpointError::UnfinalizedCheckpoint { path: root.path().join("2").display().to_string() }),
        );
    }

    #[test]
    fn test_load_checkpointable() {
        let root = tempfile::tempdir().unwrap();
        let handler = JsonHandler::default();

        // A missing directory, a directory without step metadata, and a directory whose step metadata records no
        // commit timestamp and which has no commit file are not finalized checkpoints.
        let missing = root.path().join("missing");
        assert_eq!(
            load_checkpointable(&missing, "config", &handler, ()),
            Err(CheckpointError::UnfinalizedCheckpoint { path: missing.display().to_string() }),
        );
        let unfinalized = root.path().join("1");
        fs::create_dir_all(unfinalized.join("config")).unwrap();
        fs::write(unfinalized.join("config").join("metadata"), "1").unwrap();
        assert_eq!(
            load_checkpointable(&unfinalized, "config", &handler, ()),
            Err(CheckpointError::UnfinalizedCheckpoint { path: unfinalized.display().to_string() }),
        );
        fs::write(
            unfinalized.join(STEP_METADATA_FILE_NAME),
            r#"{"item_handlers": {"config": "JsonHandler"}, "init_timestamp_nsecs": 0}"#,
        )
        .unwrap();
        assert_eq!(
            load_checkpointable(&unfinalized, "config", &handler, ()),
            Err(CheckpointError::UnfinalizedCheckpoint { path: unfinalized.display().to_string() }),
        );
        fs::write(unfinalized.join(COMMIT_SUCCESS_FILE), "").unwrap();
        assert_eq!(load_checkpointable(&unfinalized, "config", &handler, ()), Ok(json!(1)));

        // Names that the checkpoint does not record and names that cannot name a checkpointable are rejected.
        assert_eq!(
            load_checkpointable(&unfinalized, "state", &handler, ()),
            Err(CheckpointError::MissingCheckpoint {
                message: format!("checkpoint `{}` has no checkpointable named `state`", unfinalized.display()),
            }),
        );
        assert!(matches!(
            load_checkpointable(&unfinalized, "config/metadata", &handler, ()),
            Err(CheckpointError::InvalidArgument { .. }),
        ));
    }

    #[test]
    fn test_metadata() {
        let root = tempfile::tempdir().unwrap();
        let context = test_context();
        let options = SaveOptions::new()
            .with_custom_metadata(json!("note"))
            .with_metrics(BTreeMap::from([("loss".to_string(), 0.5)]));
        save(root.path().join("1"), &vec![Array::scalar(3u8).unwrap()], &options, &context).unwrap();
        let checkpoint = metadata(root.path().join("1")).unwrap();
        assert_eq!(checkpoint.path(), &CheckpointPath::from(root.path().join("1")));
        assert_eq!(
            checkpoint.metadata(),
            &ParameterizedMetadata::new("test", vec![LeafMetadata::new("$[0]", "U8", [])]),
        );
        assert_eq!(checkpoint.custom_metadata(), &json!("note"));
        assert_eq!(checkpoint.metrics(), Some(&BTreeMap::from([("loss".to_string(), 0.5)])));
        assert!(checkpoint.commit_timestamp_nanoseconds() >= Some(checkpoint.initialization_timestamp_nanoseconds()));
        assert!(matches!(metadata(root.path().join("2")), Err(CheckpointError::UnfinalizedCheckpoint { .. })));

        // A checkpoint without the `state` checkpointable has no parameterized metadata.
        let (handler, config) = (JsonHandler::default(), json!({"layers": 2}));
        let checkpointables = Checkpointables::new().with("config", &handler, &config);
        save_checkpointables(root.path().join("3"), &checkpointables, &SaveOptions::new(), &context).unwrap();
        assert!(matches!(
            metadata(root.path().join("3")),
            Err(CheckpointError::MissingCheckpoint { message }) if message.contains("no checkpointable named `state`"),
        ));
    }

    #[test]
    fn test_checkpointables_metadata() {
        let root = tempfile::tempdir().unwrap();
        let handler = JsonHandler::default();
        let config = json!({"layers": 2});
        let checkpointables = Checkpointables::new().with("config", &handler, &config);
        save_checkpointables(root.path().join("1"), &checkpointables, &SaveOptions::new(), &test_context()).unwrap();
        assert_eq!(
            checkpointables_metadata(root.path().join("1")).map(CheckpointMetadata::into_metadata),
            Ok(BTreeMap::from([("config".to_string(), "JsonHandler".to_string())])),
        );
    }
}
