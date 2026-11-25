//! Storage formats of the arrays of one checkpointable, which is the choice that Orbax makes between its OCDBT and
//! Zarr layouts of TensorStore.
//!
//! An [`ArrayFormat`] writes a sequence of named host [`Array`]s into a checkpointable directory and opens what it
//! wrote as a [`Checkpoint`], which then reads the arrays back on demand. The
//! [`ParameterizedHandler`](crate::handlers::ParameterizedHandler) stores its arrays with the format of the
//! [`ArrayOptions`](crate::contexts::ArrayOptions) of its [`Context`](crate::contexts::Context), so the conversion of
//! parameter structures to arrays is shared by every format while each format only decides how the arrays are laid
//! out on storage. The trait is object safe, and contexts hold their format as an `Arc<dyn ArrayFormat>`, so the
//! format is chosen at runtime, as the `use_zarr3` and `use_ocdbt` options of Orbax choose it. Formats are written in
//! a streaming fashion. The writer learns the names and types of all arrays first (e.g., to write a header) and then
//! asks for the arrays one at a time, so that at most one converted leaf needs to be alive at any time.
//!
//! The crate provides `SafetensorsFormat`, which stores all arrays of a checkpointable in one safetensors file, when
//! the `safetensors` feature is enabled, which it is by default, and `ZarrFormat`, which stores them as one Zarr
//! version 3 group with one Zarr array per array (i.e., the layout of TensorStore without OCDBT), when the `zarr`
//! feature is enabled.

use std::borrow::Cow;

use ryft_core::Array;

use crate::arrays::{ArraySpecification, Checkpoint};
use crate::errors::CheckpointError;
use crate::filesystems::CheckpointPath;

// TODO(eaplatanios): Review this module.

/// Storage format of the arrays of one checkpointable. Refer to the [module documentation](self) for how formats are
/// used. Formats are [`Send`] and [`Sync`], so that a [`Context`](crate::contexts::Context) that holds one can save
/// on a background thread.
pub trait ArrayFormat: Send + Sync {
    /// Returns the name of this format, which the [`ParameterizedMetadata`](crate::metadata::ParameterizedMetadata)
    /// of saved parameter structures records.
    fn name(&self) -> &'static str;

    /// Writes the arrays that `specifications` describe into `directory`, in order. Implementations may use the
    /// names and types of all arrays before they produce any of them, and then call `produce` with the index of every
    /// entry of `specifications` exactly once and in order, dropping each produced array before producing the next.
    ///
    /// # Parameters
    ///
    ///   - `directory`: Existing checkpointable directory to write into.
    ///   - `specifications`: Name and type of every array, in order.
    ///   - `produce`: Function that returns the array of the entry of `specifications` with the provided index.
    ///
    /// # Errors
    ///
    /// Returns the first error of `produce`, [`CheckpointError::InvalidArray`] if a specification cannot be stored in
    /// this format or a produced array does not have its specified type, and [`CheckpointError::Io`] if writing fails.
    fn write<'a>(
        &self,
        directory: &CheckpointPath,
        specifications: &[ArraySpecification],
        produce: &mut dyn FnMut(usize) -> Result<Cow<'a, Array>, CheckpointError>,
    ) -> Result<(), CheckpointError>;

    /// Opens the arrays that [`Self::write`] wrote into `directory`.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::Io`] if the stored arrays cannot be read, and the validation errors of the format if
    /// they are not stored in it.
    fn open(&self, directory: &CheckpointPath) -> Result<Box<dyn Checkpoint>, CheckpointError>;
}

#[cfg(test)]
pub(crate) mod tests {
    use std::ops::Range;

    use pretty_assertions::assert_eq;
    use ryft_core::{DataType, Typed};
    use serde::{Deserialize, Serialize};

    use crate::arrays::{ArrayDescriptor, data_type_from_tag, data_type_tag};
    use crate::metadata::{read_json, write_json};

    use super::*;

    /// [`ArrayFormat`] that stores every array in a file of its own and lists the arrays in [`Self::INDEX_FILE_NAME`],
    /// which exercises the format abstraction in every feature configuration.
    #[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
    pub(crate) struct TestFormat;

    impl TestFormat {
        /// Name of the file that lists the stored arrays.
        pub(crate) const INDEX_FILE_NAME: &str = "arrays.json";
    }

    impl ArrayFormat for TestFormat {
        fn name(&self) -> &'static str {
            "test"
        }

        fn write<'a>(
            &self,
            directory: &CheckpointPath,
            specifications: &[ArraySpecification],
            produce: &mut dyn FnMut(usize) -> Result<Cow<'a, Array>, CheckpointError>,
        ) -> Result<(), CheckpointError> {
            let mut entries = Vec::with_capacity(specifications.len());
            for (index, specification) in specifications.iter().enumerate() {
                let array = produce(index)?;
                let bytes = array.logical_bytes();
                directory.join(format!("array_{index}")).write(&bytes)?;
                let r#type = array.r#type();
                entries.push(TestIndexEntry {
                    name: specification.name().to_string(),
                    data_type: data_type_tag(r#type.data_type()).unwrap().to_string(),
                    shape: r#type.static_shape().unwrap().dimensions().to_vec(),
                    byte_count: bytes.len() as u64,
                });
            }
            write_json(&directory.join(Self::INDEX_FILE_NAME), &entries)
        }

        fn open(&self, directory: &CheckpointPath) -> Result<Box<dyn Checkpoint>, CheckpointError> {
            let entries: Vec<TestIndexEntry> = read_json(&directory.join(Self::INDEX_FILE_NAME))?;
            let mut descriptors = Vec::with_capacity(entries.len());
            let mut start = 0;
            for entry in entries {
                let data_type = data_type_from_tag(&entry.data_type).unwrap();
                let end = start + entry.byte_count;
                descriptors.push(ArrayDescriptor::new(entry.name, data_type, entry.shape, start..end));
                start = end;
            }
            Ok(Box::new(TestCheckpoint { directory: directory.clone(), descriptors }))
        }
    }

    /// Entry of the index file of a [`TestFormat`].
    #[derive(Serialize, Deserialize)]
    struct TestIndexEntry {
        /// Name of the array.
        name: String,

        /// Data type tag of the array.
        data_type: String,

        /// Shape of the array.
        shape: Vec<usize>,

        /// Number of bytes of the stored array.
        byte_count: u64,
    }

    /// Arrays that a [`TestFormat`] stored in one directory.
    struct TestCheckpoint {
        /// Directory that holds the arrays.
        directory: CheckpointPath,

        /// Descriptors of the arrays, whose byte ranges follow each other in the listed order.
        descriptors: Vec<ArrayDescriptor>,
    }

    impl Checkpoint for TestCheckpoint {
        fn arrays(&self) -> &[ArrayDescriptor] {
            &self.descriptors
        }

        fn array(&self, name: &str) -> Option<&ArrayDescriptor> {
            self.descriptors.iter().find(|descriptor| descriptor.name() == name)
        }

        fn read_bytes(&self, name: &str, byte_range: Range<u64>) -> Result<Vec<u8>, CheckpointError> {
            let index = self
                .descriptors
                .iter()
                .position(|descriptor| descriptor.name() == name)
                .ok_or_else(|| CheckpointError::UnknownArray { name: name.to_string() })?;
            let bytes = self.directory.join(format!("array_{index}")).read()?;
            Ok(bytes[byte_range.start as usize..byte_range.end as usize].to_vec())
        }
    }

    #[test]
    fn test_array_format_write() {
        // Formats are object safe, so the format of a context is chosen at runtime.
        let temporary = tempfile::tempdir().unwrap();
        let directory = CheckpointPath::from(temporary.path());
        let format: Box<dyn ArrayFormat> = Box::new(TestFormat);
        let arrays = [Array::vector(vec![1.5f32, -2.0]).unwrap(), Array::scalar(7u8).unwrap()];
        let specifications = vec![
            ArraySpecification::new("first", arrays[0].r#type().into_owned()),
            ArraySpecification::new("second", arrays[1].r#type().into_owned()),
        ];
        assert_eq!(format.name(), "test");
        assert_eq!(format.write(&directory, &specifications, &mut |index| Ok(Cow::Borrowed(&arrays[index]))), Ok(()));
        let checkpoint = format.open(&directory).unwrap();
        assert_eq!(
            checkpoint.arrays().iter().map(|descriptor| descriptor.name().to_string()).collect::<Vec<_>>(),
            vec!["first".to_string(), "second".to_string()],
        );
        assert_eq!(checkpoint.read("first"), Ok(arrays[0].clone()));
        assert_eq!(checkpoint.read("second"), Ok(arrays[1].clone()));
        assert_eq!(checkpoint.array("second").map(ArrayDescriptor::data_type), Some(DataType::U8));
    }
}
