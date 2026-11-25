//! Named arrays on durable storage and read access to them.
//!
//! A [`Checkpoint`] exposes the arrays of one collection through [`ArrayDescriptor`]s and reads their contents on
//! demand, either completely or as a range of leading axis entries, so that large arrays (e.g., the stacked expert
//! projections of a mixture of experts layer) can be streamed to their destination without holding a second copy in
//! host memory. `SafetensorsCheckpoint` (with the `safetensors` feature) implements it for the
//! safetensors file format. An [`ArraySpecification`] names an array together with the type that a consumer expects
//! it to have, and parameter structures of specifications describe what importers and handlers read.

use std::ops::Range;

use ryft_core::{Array, ArrayType, DataType, Parameter};

use crate::errors::CheckpointError;

// TODO(eaplatanios): Review this module.

/// Description of one array of a [`Checkpoint`], consisting of its name, its element [`DataType`], its shape, and the
/// byte range that its contents occupy in the data section of the checkpoint file that stores it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ArrayDescriptor {
    /// Name of the array, which is unique within its checkpoint.
    name: String,

    /// Data type of the elements of the array.
    data_type: DataType,

    /// Extents of the dimensions of the array.
    shape: Vec<usize>,

    /// Byte range of the array contents, relative to the start of the data section of its file.
    byte_range: Range<u64>,
}

impl ArrayDescriptor {
    /// Creates a new [`ArrayDescriptor`]. [`Checkpoint`] implementations validate that `byte_range` holds exactly the
    /// elements that `data_type` and `shape` describe.
    #[inline]
    pub fn new<N: Into<String>, S: Into<Vec<usize>>>(
        name: N,
        data_type: DataType,
        shape: S,
        byte_range: Range<u64>,
    ) -> Self {
        Self { name: name.into(), data_type, shape: shape.into(), byte_range }
    }

    /// Returns the name of this array.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the data type of the elements of this array.
    #[inline]
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Returns the extents of the dimensions of this array.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the byte range of the contents of this array, relative to the start of the data section of its file.
    #[inline]
    pub fn byte_range(&self) -> Range<u64> {
        self.byte_range.clone()
    }

    /// Returns the number of elements of this array, or [`None`] if that number overflows `usize`.
    #[inline]
    pub fn element_count(&self) -> Option<usize> {
        self.shape.iter().try_fold(1usize, |count, &extent| count.checked_mul(extent))
    }

    /// Returns the number of bytes that the contents of this array occupy.
    #[inline]
    pub fn byte_count(&self) -> u64 {
        self.byte_range.end - self.byte_range.start
    }

    /// Returns the static [`ArrayType`] of the host [`Array`] that holds the contents of this descriptor.
    #[inline]
    pub fn array_type(&self) -> ArrayType {
        ArrayType::new_static(self.data_type, self.shape.clone())
    }
}

/// Leaf of a parameter structure that describes where the leaf comes from and what it must look like once imported.
/// A structure of specifications is the single description of the parameters of a model that importers, exporters,
/// and initializers share. It names the checkpoint array of every leaf and states the [`ArrayType`] of the imported
/// leaf, which may differ from the stored type of the array when an importer decodes or converts the array.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ArraySpecification {
    /// Name of the checkpoint array that the leaf is imported from.
    name: String,

    /// Type of the imported leaf.
    r#type: ArrayType,
}

impl ArraySpecification {
    /// Creates a new [`ArraySpecification`] for a leaf of type `type` that is imported from the array named `name`.
    #[inline]
    pub fn new<N: Into<String>>(name: N, r#type: ArrayType) -> Self {
        Self { name: name.into(), r#type }
    }

    /// Returns the name of the checkpoint array that the leaf is imported from.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the type of the imported leaf.
    #[inline]
    pub fn r#type(&self) -> &ArrayType {
        &self.r#type
    }
}

impl Parameter for ArraySpecification {}

/// Named collection of arrays on durable storage. Implementations validate the layout of their storage when they are
/// constructed, so the descriptors that [`Self::arrays`] returns are consistent with each other and with the size of
/// the underlying storage, and reading a described array can only fail because the storage itself fails.
pub trait Checkpoint {
    /// Returns the descriptors of all arrays of this checkpoint in the order of their byte ranges.
    fn arrays(&self) -> &[ArrayDescriptor];

    /// Returns the descriptor of the array named `name`, or [`None`] if this checkpoint has no such array.
    fn array(&self, name: &str) -> Option<&ArrayDescriptor>;

    /// Reads `byte_range` of the contents of the array named `name`, where the range is relative to the start of the
    /// array contents. The bytes are returned exactly as they are stored (i.e., as little-endian elements in row-major
    /// order).
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::UnknownArray`] if this checkpoint has no array named `name`,
    /// [`CheckpointError::InvalidArray`] if `byte_range` does not lie within the array, and
    /// [`CheckpointError::Io`] if the underlying storage cannot be read.
    fn read_bytes(&self, name: &str, byte_range: Range<u64>) -> Result<Vec<u8>, CheckpointError>;

    /// Reads the complete array named `name` into a host [`Array`].
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::read_bytes`], and [`CheckpointError::Program`] if the stored bytes are not a valid
    /// encoding of the data type of the array.
    fn read(&self, name: &str) -> Result<Array, CheckpointError> {
        let descriptor = self.array(name).ok_or_else(|| CheckpointError::UnknownArray { name: name.to_string() })?;
        let bytes = self.read_bytes(name, 0..descriptor.byte_count())?;
        Ok(Array::new(descriptor.array_type(), bytes_to_native_endian(bytes, descriptor.data_type()))?)
    }

    /// Reads the entries `range` along the leading axis of the array named `name` into a host [`Array`] whose leading
    /// extent is the length of `range` and whose remaining extents are those of the array. This is how stacked
    /// arrays (e.g., one projection per expert) are streamed in bounded pieces.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::read`], and [`CheckpointError::InvalidArray`] if the array has rank zero or if
    /// `range` does not lie within its leading extent.
    fn read_leading(&self, name: &str, range: Range<usize>) -> Result<Array, CheckpointError> {
        let descriptor = self.array(name).ok_or_else(|| CheckpointError::UnknownArray { name: name.to_string() })?;
        let Some((&leading_extent, trailing_shape)) = descriptor.shape().split_first() else {
            return Err(CheckpointError::InvalidArray {
                name: name.to_string(),
                message: "a rank-0 array has no leading axis to read a range of".to_string(),
            });
        };
        if range.start > range.end || range.end > leading_extent {
            return Err(CheckpointError::InvalidArray {
                name: name.to_string(),
                message: format!(
                    "leading axis range {}..{} is not within the leading extent {leading_extent}",
                    range.start, range.end,
                ),
            });
        }

        // The descriptor was validated when the checkpoint was opened, so the byte count is an exact multiple of the
        // leading extent whenever that extent is not zero.
        let entry_byte_count = if leading_extent == 0 { 0 } else { descriptor.byte_count() / leading_extent as u64 };
        let bytes =
            self.read_bytes(name, range.start as u64 * entry_byte_count..range.end as u64 * entry_byte_count)?;
        let mut shape = Vec::with_capacity(descriptor.shape().len());
        shape.push(range.end - range.start);
        shape.extend_from_slice(trailing_shape);
        let array_type = ArrayType::new_static(descriptor.data_type(), shape);
        Ok(Array::new(array_type, bytes_to_native_endian(bytes, descriptor.data_type()))?)
    }
}

/// Converts stored little-endian element bytes into the native-endian element bytes of host [`Array`]s, which is the
/// identity on little-endian hosts.
fn bytes_to_native_endian(mut bytes: Vec<u8>, data_type: DataType) -> Vec<u8> {
    if cfg!(target_endian = "big") {
        // Complex elements consist of two components that are swapped separately.
        let width = data_type.bit_width().div_ceil(8) / if data_type.is_complex() { 2 } else { 1 };
        if width > 1 {
            bytes.chunks_exact_mut(width).for_each(<[u8]>::reverse);
        }
    }
    bytes
}

/// Returns the [`DataType`] that the data type tag `tag` denotes, or [`None`] for tags that have no counterpart among
/// the data types of Ryft. Data type tags are the names that this crate stores data types under in manifests and
/// metadata files, which are the data type tags of the safetensors format, so they are available without the
/// `safetensors` feature.
pub(crate) fn data_type_from_tag(tag: &str) -> Option<DataType> {
    Some(match tag {
        "BOOL" => DataType::Boolean,
        "I8" => DataType::I8,
        "I16" => DataType::I16,
        "I32" => DataType::I32,
        "I64" => DataType::I64,
        "U8" => DataType::U8,
        "U16" => DataType::U16,
        "U32" => DataType::U32,
        "U64" => DataType::U64,
        "F8_E4M3" => DataType::F8E4M3FN,
        "F8_E5M2" => DataType::F8E5M2,
        "BF16" => DataType::BF16,
        "F16" => DataType::F16,
        "F32" => DataType::F32,
        "F64" => DataType::F64,
        "C64" => DataType::C64,
        _ => return None,
    })
}

/// Returns the data type tag of `data_type`, or [`None`] for data types that have no tag (e.g., sub-byte data types).
/// This is the inverse of [`data_type_from_tag`].
pub(crate) fn data_type_tag(data_type: DataType) -> Option<&'static str> {
    Some(match data_type {
        DataType::Boolean => "BOOL",
        DataType::I8 => "I8",
        DataType::I16 => "I16",
        DataType::I32 => "I32",
        DataType::I64 => "I64",
        DataType::U8 => "U8",
        DataType::U16 => "U16",
        DataType::U32 => "U32",
        DataType::U64 => "U64",
        DataType::F8E4M3FN => "F8_E4M3",
        DataType::F8E5M2 => "F8_E5M2",
        DataType::BF16 => "BF16",
        DataType::F16 => "F16",
        DataType::F32 => "F32",
        DataType::F64 => "F64",
        DataType::C64 => "C64",
        _ => return None,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    #[cfg(feature = "safetensors")]
    use std::collections::BTreeMap;
    #[cfg(feature = "safetensors")]
    use std::io::Cursor;

    use pretty_assertions::assert_eq;
    #[cfg(feature = "safetensors")]
    use ryft_core::bf16;

    #[cfg(feature = "safetensors")]
    use crate::safetensors::{SafetensorsCheckpoint, write_safetensors};

    use super::*;

    /// Returns an in-memory safetensors checkpoint with a `bf16` matrix, an `f32` rank-3 array, and a `u8` scalar.
    #[cfg(feature = "safetensors")]
    pub(crate) fn test_checkpoint() -> SafetensorsCheckpoint<Cursor<Vec<u8>>> {
        let arrays = vec![
            (
                "matrix".to_string(),
                Array::from_elements(
                    ArrayType::new_static(DataType::BF16, [2, 2]),
                    &[1.0f32, 2.0, 3.0, 4.0].map(bf16::from_f32),
                )
                .unwrap(),
            ),
            (
                "stack".to_string(),
                Array::from_elements(
                    ArrayType::new_static(DataType::F32, [3, 1, 2]),
                    &[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0],
                )
                .unwrap(),
            ),
            ("scalar".to_string(), Array::scalar(7u8).unwrap()),
        ];
        let mut bytes = Vec::new();
        write_safetensors(&mut bytes, &arrays, &BTreeMap::new()).unwrap();
        SafetensorsCheckpoint::from_reader(Cursor::new(bytes), "memory").unwrap()
    }

    #[test]
    fn test_array_descriptor() {
        let descriptor = ArrayDescriptor::new("weights", DataType::BF16, [2, 3], 8..20);
        assert_eq!(descriptor.name(), "weights");
        assert_eq!(descriptor.data_type(), DataType::BF16);
        assert_eq!(descriptor.shape(), &[2, 3]);
        assert_eq!(descriptor.byte_range(), 8..20);
        assert_eq!(descriptor.element_count(), Some(6));
        assert_eq!(descriptor.byte_count(), 12);
        assert_eq!(descriptor.array_type(), ArrayType::new_static(DataType::BF16, [2, 3]));

        // Rank-0 arrays hold one element, and element counts that overflow are reported rather than wrapped.
        assert_eq!(ArrayDescriptor::new("scalar", DataType::F32, [], 0..4).element_count(), Some(1));
        assert_eq!(ArrayDescriptor::new("huge", DataType::U8, [usize::MAX, 2], 0..0).element_count(), None);
    }

    #[test]
    fn test_array_specification() {
        let specification = ArraySpecification::new("weights", ArrayType::new_static(DataType::F32, [2]));
        assert_eq!(specification.name(), "weights");
        assert_eq!(specification.r#type(), &ArrayType::new_static(DataType::F32, [2]));
    }

    #[cfg(feature = "safetensors")]
    #[test]
    fn test_checkpoint_read() {
        let checkpoint = test_checkpoint();
        assert_eq!(
            checkpoint.read("stack"),
            Ok(Array::from_elements(
                ArrayType::new_static(DataType::F32, [3, 1, 2]),
                &[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]
            )
            .unwrap()),
        );
        assert_eq!(checkpoint.read("scalar"), Ok(Array::scalar(7u8).unwrap()));
        assert_eq!(checkpoint.read("missing"), Err(CheckpointError::UnknownArray { name: "missing".to_string() }));
    }

    #[cfg(feature = "safetensors")]
    #[test]
    fn test_checkpoint_read_leading() {
        let checkpoint = test_checkpoint();
        assert_eq!(
            checkpoint.read_leading("stack", 1..3),
            Ok(Array::from_elements(ArrayType::new_static(DataType::F32, [2, 1, 2]), &[2.0f32, 3.0, 4.0, 5.0])
                .unwrap()),
        );
        assert_eq!(
            checkpoint.read_leading("stack", 2..2),
            Ok(Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0, 1, 2]), &[]).unwrap()),
        );

        // Ranges outside the leading extent, rank-0 arrays, and unknown arrays are rejected.
        assert_eq!(
            checkpoint.read_leading("stack", 2..4),
            Err(CheckpointError::InvalidArray {
                name: "stack".to_string(),
                message: "leading axis range 2..4 is not within the leading extent 3".to_string(),
            }),
        );
        assert_eq!(
            checkpoint.read_leading("scalar", 0..1),
            Err(CheckpointError::InvalidArray {
                name: "scalar".to_string(),
                message: "a rank-0 array has no leading axis to read a range of".to_string(),
            }),
        );
        assert_eq!(
            checkpoint.read_leading("missing", 0..1),
            Err(CheckpointError::UnknownArray { name: "missing".to_string() }),
        );
    }
}
