//! Handler for parameter structures, mirroring the `PyTreeHandler` of Orbax.
//!
//! A [`ParameterizedHandler`] saves every [`Parameterized`] structure by converting its leaves to host [`Array`]s with
//! a [`LeafHandler`] (the counterpart of the Orbax `TypeHandler`) and writing them with an [`ArrayFormat`], one array
//! per leaf, named by the [`ParameterPath`] of the leaf rendered with its `Display` implementation (e.g.,
//! `$.layers[2].weight`). Next to the arrays it writes the [`ParameterizedMetadata`] of the structure, which records
//! the format and the path, data type, and shape of every leaf. The handler borrows its leaf handler and its
//! [`ParameterizedOptions`] from a [`Context`](crate::contexts::Context), which creates it with
//! [`Context::parameterized_handler`](crate::contexts::Context::parameterized_handler).
//!
//! Loading is driven by an abstract structure (i.e., the "abstract state" of Orbax) that has the structure of the
//! loaded checkpointable and an [`ArrayType`] in place of every leaf. Its leaf types say exactly what every loaded leaf
//! must look like, so loading rejects stored arrays whose types differ. The [`ParameterizedOptions`] decide whether
//! stored arrays that the abstract structure does not name are an error (the default) or ignored, from which stored
//! arrays the leaves are loaded, and whether data types are converted.
//!
//! Both directions stream. Saving computes the types of all leaves first, so that the format can write its header,
//! and then converts and writes one leaf at a time. Loading reads and converts one leaf at a time. At most one
//! converted leaf is therefore alive in addition to the structure itself.

use std::borrow::Cow;
use std::collections::HashSet;
use std::sync::Arc;

use ryft_core::{
    Array, ArrayType, ConvertElementType, Parameter, ParameterPath, Parameterized, ParameterizedFamily, Typed,
};

use crate::arrays::{ArraySpecification, data_type_tag};
use crate::contexts::ParameterizedOptions;
use crate::errors::CheckpointError;
use crate::filesystems::CheckpointPath;
use crate::formats::ArrayFormat;
use crate::handlers::{CheckpointableHandler, CheckpointableLoader, CheckpointableSaver};
use crate::metadata::{LeafMetadata, ParameterizedMetadata};
use crate::multiprocessing::MultiprocessingOptions;

// TODO(eaplatanios): Review this module.

/// Conversion between the leaves of a parameter structure and the host [`Array`]s that store them, which mirrors the
/// `TypeHandler` of Orbax. A leaf handler lets [`ParameterizedHandler`] save structures whose leaves are not host
/// arrays (e.g., arrays that live on an accelerator), by copying every leaf to the host when it is saved and placing
/// every loaded array where the leaf type needs it.
pub trait LeafHandler {
    /// Type of the leaves that this handler converts.
    type Leaf: Parameter;

    /// Returns the type of the host array that `leaf` is stored as, without converting `leaf`.
    ///
    /// # Errors
    ///
    /// Returns a [`CheckpointError`] if `leaf` cannot be stored as a host array.
    fn r#type(&self, path: &ParameterPath, leaf: &Self::Leaf) -> Result<ArrayType, CheckpointError>;

    /// Returns the host array that stores `leaf`, which must have the type that this handler reports for `leaf`.
    ///
    /// # Errors
    ///
    /// Returns a [`CheckpointError`] if `leaf` cannot be converted.
    fn serialize<'l>(&self, path: &ParameterPath, leaf: &'l Self::Leaf) -> Result<Cow<'l, Array>, CheckpointError>;

    /// Returns the leaf that `array` stores.
    ///
    /// # Errors
    ///
    /// Returns a [`CheckpointError`] if `array` cannot be converted.
    fn deserialize(&self, path: &ParameterPath, array: Array) -> Result<Self::Leaf, CheckpointError>;
}

/// [`LeafHandler`] for structures whose leaves are host [`Array`]s, for which both conversions are the identity.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ArrayLeafHandler;

impl LeafHandler for ArrayLeafHandler {
    type Leaf = Array;

    #[inline]
    fn r#type(&self, path: &ParameterPath, leaf: &Array) -> Result<ArrayType, CheckpointError> {
        let _ = path;
        Ok(leaf.r#type().into_owned())
    }

    #[inline]
    fn serialize<'l>(&self, path: &ParameterPath, leaf: &'l Array) -> Result<Cow<'l, Array>, CheckpointError> {
        let _ = path;
        Ok(Cow::Borrowed(leaf))
    }

    #[inline]
    fn deserialize(&self, path: &ParameterPath, array: Array) -> Result<Array, CheckpointError> {
        let _ = path;
        Ok(array)
    }
}

/// [`CheckpointableHandler`] for parameter structures, which mirrors the `PyTreeHandler` of Orbax. It converts leaves
/// with a borrowed [`LeafHandler`], stores them with an [`ArrayFormat`], and loads them as borrowed
/// [`ParameterizedOptions`] say. Refer to the [module documentation](self) for the stored layout and the loading
/// semantics.
#[derive(Clone)]
pub struct ParameterizedHandler<'o, L: LeafHandler> {
    /// Conversion between leaves and host arrays.
    leaf_handler: &'o L,

    /// Storage format of the arrays.
    format: Arc<dyn ArrayFormat>,

    /// Options of loading.
    options: &'o ParameterizedOptions,
}

impl<'o, L: LeafHandler> ParameterizedHandler<'o, L> {
    /// Creates a new [`ParameterizedHandler`].
    ///
    /// # Parameters
    ///
    ///   - `leaf_handler`: Conversion between leaves and host arrays.
    ///   - `format`: Storage format of the arrays.
    ///   - `options`: Options of loading.
    #[inline]
    pub fn new(leaf_handler: &'o L, format: Arc<dyn ArrayFormat>, options: &'o ParameterizedOptions) -> Self {
        Self { leaf_handler, format, options }
    }

    /// Returns the conversion between leaves and host arrays.
    #[inline]
    pub fn leaf_handler(&self) -> &'o L {
        self.leaf_handler
    }

    /// Returns the storage format of the arrays.
    #[inline]
    pub fn format(&self) -> &Arc<dyn ArrayFormat> {
        &self.format
    }

    /// Returns the options of loading.
    #[inline]
    pub fn options(&self) -> &'o ParameterizedOptions {
        self.options
    }
}

impl<L: LeafHandler> CheckpointableHandler for ParameterizedHandler<'_, L> {
    #[inline]
    fn name(&self) -> String {
        "ParameterizedHandler".to_string()
    }
}

impl<L: LeafHandler, T: Parameterized<L::Leaf>> CheckpointableSaver<T> for ParameterizedHandler<'_, L> {
    fn save(
        &self,
        directory: &CheckpointPath,
        checkpointable: &T,
        multiprocessing: &MultiprocessingOptions,
    ) -> Result<(), CheckpointError> {
        // Every process holds every leaf completely, so the primary process writes all of them.
        if !multiprocessing.is_primary() {
            return Ok(());
        }

        // The types of all leaves are computed before any leaf is converted, so that the format can write its header
        // first and then receive the converted leaves one at a time.
        let leaves = checkpointable.named_parameters().collect::<Vec<_>>();
        let mut specifications = Vec::with_capacity(leaves.len());
        let mut leaf_metadata = Vec::with_capacity(leaves.len());
        for (path, leaf) in &leaves {
            let name = path.to_string();
            let r#type = self.leaf_handler.r#type(path, leaf)?;
            let invalid = |message: String| CheckpointError::InvalidArray { name: name.clone(), message };
            let data_type = r#type.data_type();
            let tag = data_type_tag(data_type)
                .ok_or_else(|| invalid(format!("data type `{data_type}` has no data type tag")))?;
            let shape = r#type
                .static_shape()
                .ok_or_else(|| invalid(format!("type `{type}` does not have a static shape")))?
                .dimensions()
                .to_vec();
            leaf_metadata.push(LeafMetadata::new(name.clone(), tag, shape));
            specifications.push(ArraySpecification::new(name, r#type));
        }
        self.format.write(directory, &specifications, &mut |index| {
            let (path, leaf) = &leaves[index];
            self.leaf_handler.serialize(path, leaf)
        })?;
        ParameterizedMetadata::new(self.format.name(), leaf_metadata).write(directory)
    }
}

impl<L: LeafHandler, S: Parameterized<ArrayType>> CheckpointableLoader<S> for ParameterizedHandler<'_, L>
where
    S::Family: ParameterizedFamily<L::Leaf>,
{
    type Output = S::To<L::Leaf>;

    fn load(&self, directory: &CheckpointPath, abstract_checkpointable: S) -> Result<Self::Output, CheckpointError> {
        let checkpoint = self.format.open(directory)?;

        // A strict load first checks that the abstract structure names every stored array, so that a checkpoint of
        // another structure is never loaded silently in part.
        if !self.options.partial_load() {
            let names = abstract_checkpointable
                .named_parameters()
                .map(|(path, _)| self.options.array_name(&path))
                .collect::<HashSet<_>>();
            let unnamed = checkpoint
                .arrays()
                .iter()
                .filter(|descriptor| !names.contains(descriptor.name()))
                .map(|descriptor| format!("`{}`", descriptor.name()))
                .collect::<Vec<_>>();
            if !unnamed.is_empty() {
                return Err(CheckpointError::InvalidArgument {
                    message: format!(
                        "the abstract state does not name the stored arrays {}, which only partial loading ignores",
                        unnamed.join(", "),
                    ),
                });
            }
        }

        // Every leaf is checked against its stored descriptor before its contents are read, and stored arrays that
        // the abstract structure does not name are never read.
        abstract_checkpointable.try_map_named_parameters(|path, r#type| {
            let name = self.options.array_name(path);
            let descriptor = checkpoint.array(&name).ok_or_else(|| CheckpointError::InvalidArray {
                name: name.clone(),
                message: format!("leaf `{path}` is loaded from this array, which the checkpoint does not contain"),
            })?;
            let stored_type = descriptor.array_type();
            let compatible = if self.options.convert_data_types() {
                ArrayType::new_static(r#type.data_type(), descriptor.shape().to_vec()) == r#type
            } else {
                stored_type == r#type
            };
            if !compatible {
                return Err(CheckpointError::InvalidArray {
                    message: format!(
                        "leaf `{path}` must have type `{type}` but the stored array has type `{stored_type}`",
                    ),
                    name,
                });
            }
            let mut array = checkpoint.read(&name)?;
            if stored_type.data_type() != r#type.data_type() {
                array = array.convert_element_type(r#type.data_type())?;
            }
            self.leaf_handler.deserialize(path, array)
        })
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "safetensors")]
    use std::collections::BTreeMap;

    use pretty_assertions::assert_eq;
    use ryft_core::DataType;

    #[cfg(feature = "safetensors")]
    use crate::arrays::Checkpoint;
    #[cfg(all(feature = "safetensors", feature = "object-store"))]
    use crate::filesystems::objects::tests::test_object_store_root;
    use crate::formats::tests::TestFormat;
    #[cfg(feature = "safetensors")]
    use crate::metadata::PARAMETERIZED_METADATA_FILE_NAME;
    #[cfg(feature = "safetensors")]
    use crate::safetensors::{SafetensorsCheckpoint, SafetensorsFormat, write_safetensors};

    use super::*;

    /// Returns the structure that most tests save, which nests a vector of arrays in a tuple.
    fn test_state() -> (Array, Vec<Array>) {
        (
            Array::vector(vec![1.5f32, -2.0]).unwrap(),
            vec![Array::scalar(7u8).unwrap(), Array::matrix(1, 2, vec![3i32, 4]).unwrap()],
        )
    }

    /// Returns the abstract structure of [`test_state`].
    fn test_abstract_state() -> (ArrayType, Vec<ArrayType>) {
        (
            ArrayType::new_static(DataType::F32, [2]),
            vec![ArrayType::scalar(DataType::U8), ArrayType::new_static(DataType::I32, [1, 2])],
        )
    }

    /// Saves [`test_state`] into `directory` with the [`TestFormat`].
    fn save_test_state(directory: &CheckpointPath) {
        let options = ParameterizedOptions::new();
        let handler = ParameterizedHandler::new(&ArrayLeafHandler, Arc::new(TestFormat), &options);
        handler.save(directory, &test_state(), &MultiprocessingOptions::default()).unwrap();
    }

    #[test]
    fn test_array_leaf_handler() {
        let path = ParameterPath::root();
        let array = Array::vector(vec![1u8, 2]).unwrap();
        assert_eq!(ArrayLeafHandler.r#type(&path, &array), Ok(ArrayType::new_static(DataType::U8, [2])));
        assert_eq!(ArrayLeafHandler.serialize(&path, &array), Ok(Cow::Borrowed(&array)));
        assert_eq!(ArrayLeafHandler.deserialize(&path, array.clone()), Ok(array));
    }

    #[test]
    fn test_parameterized_handler_new() {
        let options = ParameterizedOptions::new().with_partial_load(true);
        let handler = ParameterizedHandler::new(&ArrayLeafHandler, Arc::new(TestFormat), &options);
        assert_eq!(handler.name(), "ParameterizedHandler");
        assert_eq!(handler.leaf_handler(), &ArrayLeafHandler);
        assert_eq!(handler.format().name(), "test");
        assert!(handler.options().partial_load());
    }

    #[test]
    fn test_parameterized_handler_save() {
        let temporary = tempfile::tempdir().unwrap();
        let directory = CheckpointPath::from(temporary.path());
        save_test_state(&directory);

        // The arrays are named by the parameter paths of their leaves, and the metadata records the format.
        let checkpoint = TestFormat.open(&directory).unwrap();
        assert_eq!(
            checkpoint.arrays().iter().map(|descriptor| descriptor.name().to_string()).collect::<Vec<_>>(),
            vec!["$.0".to_string(), "$.1[0]".to_string(), "$.1[1]".to_string()],
        );
        assert_eq!(
            ParameterizedMetadata::read(&directory),
            Ok(ParameterizedMetadata::new(
                "test",
                vec![
                    LeafMetadata::new("$.0", "F32", [2]),
                    LeafMetadata::new("$.1[0]", "U8", []),
                    LeafMetadata::new("$.1[1]", "I32", [1, 2]),
                ],
            )),
        );

        // Processes other than the primary one write nothing.
        let other = CheckpointPath::from(temporary.path().join("other"));
        other.create_directory().unwrap();
        let options = ParameterizedOptions::new();
        let handler = ParameterizedHandler::new(&ArrayLeafHandler, Arc::new(TestFormat), &options);
        assert_eq!(handler.save(&other, &test_state(), &MultiprocessingOptions::new(1, 2).unwrap()), Ok(()));
        assert_eq!(other.list(), Ok(Vec::new()));
    }

    #[cfg(feature = "safetensors")]
    #[test]
    fn test_parameterized_handler_save_safetensors() {
        let temporary = tempfile::tempdir().unwrap();
        let directory = CheckpointPath::from(temporary.path());
        let options = ParameterizedOptions::new();
        let handler = ParameterizedHandler::new(&ArrayLeafHandler, Arc::new(SafetensorsFormat), &options);
        let state = test_state();
        assert_eq!(handler.save(&directory, &state, &MultiprocessingOptions::default()), Ok(()));

        // The file is byte for byte the one that `write_safetensors` writes for the same named arrays.
        let checkpoint = SafetensorsCheckpoint::open(temporary.path().join(SafetensorsFormat::FILE_NAME)).unwrap();
        assert_eq!(
            checkpoint.arrays().iter().map(|descriptor| descriptor.name().to_string()).collect::<Vec<_>>(),
            vec!["$.0".to_string(), "$.1[0]".to_string(), "$.1[1]".to_string()],
        );
        let named = vec![
            ("$.0".to_string(), state.0.clone()),
            ("$.1[0]".to_string(), state.1[0].clone()),
            ("$.1[1]".to_string(), state.1[1].clone()),
        ];
        let mut expected = Vec::new();
        write_safetensors(&mut expected, &named, &BTreeMap::new()).unwrap();
        assert_eq!(std::fs::read(temporary.path().join(SafetensorsFormat::FILE_NAME)).unwrap(), expected);
        assert!(temporary.path().join(PARAMETERIZED_METADATA_FILE_NAME).is_file());
        assert_eq!(
            ParameterizedMetadata::read(&directory).map(|metadata| metadata.format().to_string()),
            Ok("safetensors".to_string()),
        );
        assert_eq!(handler.load(&directory, test_abstract_state()), Ok(state));
    }

    #[test]
    fn test_parameterized_handler_save_invalid_leaf() {
        // Leaves whose data types have no tag are rejected before anything is written.
        let temporary = tempfile::tempdir().unwrap();
        let directory = CheckpointPath::from(temporary.path());
        let narrow = Array::from_elements(
            ArrayType::new_static(DataType::F4E2M1FN, [2]),
            &[ryft_core::f4e2m1fn::from_f32(1.0).unwrap(); 2],
        )
        .unwrap();
        let options = ParameterizedOptions::new();
        let handler = ParameterizedHandler::new(&ArrayLeafHandler, Arc::new(TestFormat), &options);
        assert_eq!(
            handler.save(&directory, &vec![narrow], &MultiprocessingOptions::default()),
            Err(CheckpointError::InvalidArray {
                name: "$[0]".to_string(),
                message: "data type `f4e2m1fn` has no data type tag".to_string(),
            }),
        );
        assert_eq!(directory.list(), Ok(Vec::new()));
    }

    #[test]
    fn test_parameterized_handler_load() {
        let temporary = tempfile::tempdir().unwrap();
        let directory = CheckpointPath::from(temporary.path());
        save_test_state(&directory);
        let strict = ParameterizedOptions::new();
        let handler = ParameterizedHandler::new(&ArrayLeafHandler, Arc::new(TestFormat), &strict);
        assert_eq!(handler.load(&directory, test_abstract_state()), Ok(test_state()));

        // Stored arrays that the abstract state does not name are an error unless loading is partial.
        let first = (ArrayType::new_static(DataType::F32, [2]),);
        assert_eq!(
            handler.load(&directory, first.clone()),
            Err(CheckpointError::InvalidArgument {
                message: "the abstract state does not name the stored arrays `$.1[0]`, `$.1[1]`, which only partial \
                          loading ignores"
                    .to_string(),
            }),
        );
        let partial = ParameterizedOptions::new().with_partial_load(true);
        let handler = ParameterizedHandler::new(&ArrayLeafHandler, Arc::new(TestFormat), &partial);
        assert_eq!(handler.load(&directory, first), Ok((test_state().0,)));
    }

    #[test]
    fn test_parameterized_handler_load_mismatch() {
        let temporary = tempfile::tempdir().unwrap();
        let directory = CheckpointPath::from(temporary.path());
        save_test_state(&directory);
        let options = ParameterizedOptions::new();
        let handler = ParameterizedHandler::new(&ArrayLeafHandler, Arc::new(TestFormat), &options);

        // A leaf that the checkpoint does not store and a leaf whose stored type differs are both reported by path.
        let (first, mut rest) = test_abstract_state();
        rest.push(ArrayType::scalar(DataType::F32));
        assert_eq!(
            handler.load(&directory, (first, rest)),
            Err(CheckpointError::InvalidArray {
                name: "$.1[2]".to_string(),
                message: "leaf `$.1[2]` is loaded from this array, which the checkpoint does not contain".to_string(),
            }),
        );
        assert_eq!(
            handler.load(&directory, (ArrayType::new_static(DataType::F32, [3]), test_abstract_state().1)),
            Err(CheckpointError::InvalidArray {
                name: "$.0".to_string(),
                message: "leaf `$.0` must have type `f32[3]` but the stored array has type `f32[2]`".to_string(),
            }),
        );
    }

    #[test]
    fn test_parameterized_handler_load_names() {
        let temporary = tempfile::tempdir().unwrap();
        let directory = CheckpointPath::from(temporary.path());
        save_test_state(&directory);
        let state = test_state();
        let load = |options: ParameterizedOptions, abstract_state| {
            ParameterizedHandler::new(&ArrayLeafHandler, Arc::new(TestFormat), &options)
                .load(&directory, abstract_state)
        };

        // A prefix loads one substructure of the saved pair, which ignores the rest of it only when loading is
        // partial.
        let first = ArrayType::new_static(DataType::F32, [2]);
        assert_eq!(load(ParameterizedOptions::new().with_partial_load(true).with_prefix("$.0"), first), Ok(state.0));
        let second = ParameterizedOptions::new().with_partial_load(true).with_prefix("$.1");
        assert_eq!(
            ParameterizedHandler::new(&ArrayLeafHandler, Arc::new(TestFormat), &second)
                .load(&directory, test_abstract_state().1),
            Ok(state.1.clone()),
        );
        assert!(matches!(
            load(ParameterizedOptions::new().with_prefix("$.0"), ArrayType::new_static(DataType::F32, [2])),
            Err(CheckpointError::InvalidArgument { .. }),
        ));

        // Names map every leaf of the abstract state to the stored array that it is loaded from.
        let renamed = ParameterizedOptions::new()
            .with_partial_load(true)
            .with_names(|path| if path.to_string() == "$.0" { "$.1[1]".to_string() } else { "$.1[0]".to_string() });
        let swapped = (ArrayType::new_static(DataType::I32, [1, 2]), ArrayType::scalar(DataType::U8));
        assert_eq!(
            ParameterizedHandler::new(&ArrayLeafHandler, Arc::new(TestFormat), &renamed).load(&directory, swapped),
            Ok((state.1[1].clone(), state.1[0].clone())),
        );
        assert_eq!(
            load(
                ParameterizedOptions::new().with_partial_load(true).with_prefix("$.9"),
                ArrayType::new_static(DataType::F32, [2]),
            ),
            Err(CheckpointError::InvalidArray {
                name: "$.9".to_string(),
                message: "leaf `$` is loaded from this array, which the checkpoint does not contain".to_string(),
            }),
        );
    }

    #[test]
    fn test_parameterized_handler_load_data_type_conversion() {
        let temporary = tempfile::tempdir().unwrap();
        let directory = CheckpointPath::from(temporary.path());
        save_test_state(&directory);
        let load = |options: ParameterizedOptions, abstract_state| {
            let options = options.with_partial_load(true).with_prefix("$.0");
            ParameterizedHandler::new(&ArrayLeafHandler, Arc::new(TestFormat), &options)
                .load(&directory, abstract_state)
        };

        // Data types are converted only on request, while shapes must always match.
        assert_eq!(
            load(ParameterizedOptions::new(), ArrayType::new_static(DataType::BF16, [2])),
            Err(CheckpointError::InvalidArray {
                name: "$.0".to_string(),
                message: "leaf `$` must have type `bf16[2]` but the stored array has type `f32[2]`".to_string(),
            }),
        );
        let converting = ParameterizedOptions::new().with_data_type_conversion(true);
        assert_eq!(
            load(converting.clone(), ArrayType::new_static(DataType::BF16, [2])),
            Ok(test_state().0.convert_element_type(DataType::BF16).unwrap()),
        );
        assert_eq!(
            load(converting, ArrayType::new_static(DataType::BF16, [1, 2])),
            Err(CheckpointError::InvalidArray {
                name: "$.0".to_string(),
                message: "leaf `$` must have type `bf16[1, 2]` but the stored array has type `f32[2]`".to_string(),
            }),
        );
    }

    #[test]
    fn test_parameterized_handler_custom_leaf_handler() {
        // Leaves that are not arrays are converted by their leaf handler in both directions.
        #[derive(Clone, Debug, PartialEq)]
        struct Values(Vec<f32>);

        impl Parameter for Values {}

        struct ValuesLeafHandler;

        impl LeafHandler for ValuesLeafHandler {
            type Leaf = Values;

            fn r#type(&self, path: &ParameterPath, leaf: &Values) -> Result<ArrayType, CheckpointError> {
                let _ = path;
                Ok(ArrayType::new_static(DataType::F32, [leaf.0.len()]))
            }

            fn serialize<'l>(&self, path: &ParameterPath, leaf: &'l Values) -> Result<Cow<'l, Array>, CheckpointError> {
                let _ = path;
                Ok(Cow::Owned(Array::vector(leaf.0.clone())?))
            }

            fn deserialize(&self, path: &ParameterPath, array: Array) -> Result<Values, CheckpointError> {
                let _ = path;
                Ok(Values(array.elements::<f32>()?))
            }
        }

        let temporary = tempfile::tempdir().unwrap();
        let directory = CheckpointPath::from(temporary.path());
        let options = ParameterizedOptions::new();
        let handler = ParameterizedHandler::new(&ValuesLeafHandler, Arc::new(TestFormat), &options);
        let state = (Values(vec![1.0, 2.0]), vec![Values(vec![3.0])]);
        assert_eq!(handler.save(&directory, &state, &MultiprocessingOptions::default()), Ok(()));
        assert_eq!(
            handler.load(
                &directory,
                (ArrayType::new_static(DataType::F32, [2]), vec![ArrayType::new_static(DataType::F32, [1])]),
            ),
            Ok(state),
        );
    }

    #[cfg(all(feature = "safetensors", feature = "object-store"))]
    #[test]
    fn test_parameterized_handler_object_store() {
        // Arrays are streamed into one object and read back through byte ranges.
        let directory = test_object_store_root().join("state");
        let options = ParameterizedOptions::new();
        let handler = ParameterizedHandler::new(&ArrayLeafHandler, Arc::new(SafetensorsFormat), &options);
        assert_eq!(handler.save(&directory, &test_state(), &MultiprocessingOptions::default()), Ok(()));
        assert_eq!(
            directory.list(),
            Ok(vec![PARAMETERIZED_METADATA_FILE_NAME.to_string(), SafetensorsFormat::FILE_NAME.to_string()]),
        );
        assert_eq!(handler.load(&directory, test_abstract_state()), Ok(test_state()));
        assert_eq!(
            ParameterizedMetadata::read(&directory).map(|metadata| metadata.format().to_string()),
            Ok("safetensors".to_string()),
        );
    }
}
