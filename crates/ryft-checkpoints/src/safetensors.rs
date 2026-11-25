//! Reading and writing of the [safetensors](https://huggingface.co/docs/safetensors/index) file format.
//!
//! A safetensors file starts with the length of its header as an unsigned 64-bit little-endian integer, followed by a
//! JSON header of that length and then by the data section. The header maps every array name to its data type tag,
//! its shape, and the byte range of its contents in the data section, and it may carry free-form string metadata under
//! the reserved `__metadata__` key. [`SafetensorsCheckpoint`] parses and validates the header once, when it is created,
//! and then reads array contents on demand. Validation rejects duplicate array names, unsupported data type tags,
//! byte ranges whose size disagrees with the data type and shape of their array, and data sections that the byte
//! ranges do not cover exactly once (i.e., overlapping ranges, gaps between ranges, and ranges past the end of the
//! file), which is the layout that the reference implementation requires.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Formatter;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::path::Path;
use std::sync::Mutex;

use ryft_core::{Array, Typed};
use serde::Deserialize;
use serde::de::{Error as DeserializationError, MapAccess, Visitor};

use crate::arrays::{ArrayDescriptor, ArraySpecification, Checkpoint, data_type_from_tag, data_type_tag};
use crate::errors::CheckpointError;
use crate::filesystems::{CheckpointPath, FileReader};
use crate::formats::ArrayFormat;

// TODO(eaplatanios): Review this module.

/// Reserved header key that holds the free-form string metadata of a safetensors file.
const METADATA_KEY: &str = "__metadata__";

/// Largest header size that [`SafetensorsCheckpoint`] accepts, which matches the limit of the reference
/// implementation (i.e., 100 MB).
const MAXIMUM_HEADER_BYTE_COUNT: u64 = 100_000_000;

/// [`Checkpoint`] stored in one safetensors file. The header is parsed and validated when the checkpoint is created,
/// and array contents are read from the underlying reader on demand, so opening a checkpoint is cheap regardless of
/// its size. Reads are serialized through an internal lock, so a checkpoint can be shared across threads when its
/// reader can be sent between them.
#[derive(Debug)]
pub struct SafetensorsCheckpoint<R> {
    /// Reader over the complete file, shared by all reads.
    reader: Mutex<R>,

    /// Name of the underlying storage (e.g., a file path) that is used in diagnostics.
    source: String,

    /// Offset of the data section from the start of the file.
    data_offset: u64,

    /// Descriptors of all arrays, in the order of their byte ranges.
    arrays: Vec<ArrayDescriptor>,

    /// Index into [`Self::arrays`] of every array name.
    indices: HashMap<String, usize>,

    /// Free-form string metadata of the file.
    metadata: BTreeMap<String, String>,
}

impl SafetensorsCheckpoint<File> {
    /// Opens the safetensors file at `path` and validates its header and layout.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::Io`] if the file cannot be opened or read, and the errors of
    /// [`Self::from_reader`] if its contents are not a valid safetensors file.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, CheckpointError> {
        let source = path.as_ref().display().to_string();
        let file = File::open(path.as_ref())
            .map_err(|error| CheckpointError::Io { path: source.clone(), message: error.to_string() })?;
        Self::from_reader(file, source)
    }
}

impl SafetensorsCheckpoint<FileReader> {
    /// Opens the safetensors file at `path` on its file system and validates its header and layout. Array contents
    /// are read on demand with one byte range read per array, so the file is never loaded as a whole, which matters
    /// for files in object stores.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::Io`] if the file cannot be read, and the errors of [`Self::from_reader`] if its
    /// contents are not a valid safetensors file.
    pub fn from_checkpoint_path(path: &CheckpointPath) -> Result<Self, CheckpointError> {
        Self::from_reader(FileReader::open(path.clone())?, path.to_string())
    }
}

impl<R: Read + Seek> SafetensorsCheckpoint<R> {
    /// Creates a [`SafetensorsCheckpoint`] over `reader`, which must cover one complete safetensors file, and
    /// validates its header and layout.
    ///
    /// # Parameters
    ///
    ///   - `reader`: Reader over the complete file.
    ///   - `source`: Name of the underlying storage (e.g., a file path) that is used in diagnostics.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::Io`] if `reader` fails, [`CheckpointError::InvalidHeader`] if the header is
    /// truncated, too large, not valid JSON of the expected form, or names an array more than once, and
    /// [`CheckpointError::InvalidArray`] if an array has an unsupported data type tag, a byte range whose size
    /// disagrees with its data type and shape, or a byte range that overlaps another array, leaves a gap in the data
    /// section, or extends past the end of the file.
    pub fn from_reader<S: Into<String>>(mut reader: R, source: S) -> Result<Self, CheckpointError> {
        let source = source.into();
        let io_error = |error: std::io::Error| CheckpointError::Io { path: source.clone(), message: error.to_string() };
        let file_byte_count = reader.seek(SeekFrom::End(0)).map_err(io_error)?;
        if file_byte_count < 8 {
            return Err(CheckpointError::InvalidHeader {
                message: format!(
                    "file holds {file_byte_count} bytes, which is fewer than the 8 bytes of the header size"
                ),
            });
        }

        let mut header_byte_count = [0u8; 8];
        reader.seek(SeekFrom::Start(0)).map_err(io_error)?;
        reader.read_exact(&mut header_byte_count).map_err(io_error)?;
        let header_byte_count = u64::from_le_bytes(header_byte_count);
        if header_byte_count > MAXIMUM_HEADER_BYTE_COUNT {
            return Err(CheckpointError::InvalidHeader {
                message: format!(
                    "header size {header_byte_count} exceeds the maximum supported size {MAXIMUM_HEADER_BYTE_COUNT}",
                ),
            });
        }
        if header_byte_count > file_byte_count - 8 {
            return Err(CheckpointError::InvalidHeader {
                message: format!(
                    "header size {header_byte_count} exceeds the {} bytes that follow the header size",
                    file_byte_count - 8,
                ),
            });
        }

        let mut header_bytes = vec![0u8; header_byte_count as usize];
        reader.read_exact(&mut header_bytes).map_err(io_error)?;
        let header: Header = serde_json::from_slice(&header_bytes)
            .map_err(|error| CheckpointError::InvalidHeader { message: error.to_string() })?;
        let data_offset = 8 + header_byte_count;
        let data_byte_count = file_byte_count - data_offset;

        let mut arrays = header
            .arrays
            .into_iter()
            .map(|(name, entry)| {
                let invalid = |message: String| CheckpointError::InvalidArray { name: name.clone(), message };
                let data_type = data_type_from_tag(&entry.dtype)
                    .ok_or_else(|| invalid(format!("unsupported data type tag `{}`", entry.dtype)))?;
                let [start, end] = entry.data_offsets;
                if start > end {
                    return Err(invalid(format!("byte range {start}..{end} ends before it starts")));
                }
                let descriptor = ArrayDescriptor::new(name.clone(), data_type, entry.shape, start..end);
                let expected_byte_count = descriptor
                    .element_count()
                    .and_then(|count| (count as u64).checked_mul((data_type.bit_width() / 8) as u64))
                    .ok_or_else(|| invalid("element count overflows".to_string()))?;
                if descriptor.byte_count() != expected_byte_count {
                    return Err(invalid(format!(
                        "byte range {start}..{end} holds {} bytes but `{data_type}` elements of shape {:?} occupy \
                         {expected_byte_count} bytes",
                        descriptor.byte_count(),
                        descriptor.shape(),
                    )));
                }
                Ok(descriptor)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // The byte ranges must cover the data section exactly once. Sorting by range start (and then by range end, so
        // that empty arrays precede the array that starts at the same offset) reduces that to checking that every
        // range starts where the previous one ended.
        arrays.sort_by(|left, right| {
            (left.byte_range().start, left.byte_range().end, left.name()).cmp(&(
                right.byte_range().start,
                right.byte_range().end,
                right.name(),
            ))
        });
        let mut covered_byte_count = 0u64;
        for array in &arrays {
            let Range { start, end } = array.byte_range();
            if start != covered_byte_count {
                return Err(CheckpointError::InvalidArray {
                    name: array.name().to_string(),
                    message: format!(
                        "byte range {start}..{end} {} the preceding array, which ends at {covered_byte_count}",
                        if start < covered_byte_count { "overlaps" } else { "leaves a gap after" },
                    ),
                });
            }
            covered_byte_count = end;
        }
        if covered_byte_count != data_byte_count {
            return Err(CheckpointError::InvalidHeader {
                message: format!(
                    "arrays cover {covered_byte_count} bytes but the data section holds {data_byte_count} bytes",
                ),
            });
        }

        let indices = arrays.iter().enumerate().map(|(index, array)| (array.name().to_string(), index)).collect();
        Ok(Self { reader: Mutex::new(reader), source, data_offset, arrays, indices, metadata: header.metadata })
    }

    /// Returns the name of the underlying storage (e.g., a file path) that is used in diagnostics.
    #[inline]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Returns the offset of the data section from the start of the file (i.e., the size of the header plus the 8
    /// bytes that encode that size).
    #[inline]
    pub fn data_offset(&self) -> u64 {
        self.data_offset
    }

    /// Returns the free-form string metadata of the file, which is empty when the header has none.
    #[inline]
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }
}

impl<R: Read + Seek> Checkpoint for SafetensorsCheckpoint<R> {
    #[inline]
    fn arrays(&self) -> &[ArrayDescriptor] {
        &self.arrays
    }

    #[inline]
    fn array(&self, name: &str) -> Option<&ArrayDescriptor> {
        self.indices.get(name).map(|&index| &self.arrays[index])
    }

    fn read_bytes(&self, name: &str, byte_range: Range<u64>) -> Result<Vec<u8>, CheckpointError> {
        let descriptor = self.array(name).ok_or_else(|| CheckpointError::UnknownArray { name: name.to_string() })?;
        if byte_range.start > byte_range.end || byte_range.end > descriptor.byte_count() {
            return Err(CheckpointError::InvalidArray {
                name: name.to_string(),
                message: format!(
                    "byte range {}..{} is not within the {} bytes of the array",
                    byte_range.start,
                    byte_range.end,
                    descriptor.byte_count(),
                ),
            });
        }

        let io_error =
            |error: std::io::Error| CheckpointError::Io { path: self.source.clone(), message: error.to_string() };
        let mut bytes = vec![0u8; (byte_range.end - byte_range.start) as usize];
        let mut reader = self.reader.lock().expect("the checkpoint reader lock was poisoned");
        reader
            .seek(SeekFrom::Start(self.data_offset + descriptor.byte_range().start + byte_range.start))
            .map_err(io_error)?;
        reader.read_exact(&mut bytes).map_err(io_error)?;
        Ok(bytes)
    }
}

/// Writes `arrays` to `writer` as one safetensors file, in the provided order and with the provided free-form string
/// `metadata`. The header is padded with spaces so that the data section starts at a multiple of 8 bytes, as the
/// reference implementation does.
///
/// # Errors
///
/// Returns [`CheckpointError::InvalidArray`] if two arrays share a name, if a name is the reserved `__metadata__`
/// key, or if the data type of an array has no safetensors tag (e.g., a sub-byte data type), and
/// [`CheckpointError::Io`] if `writer` fails.
pub fn write_safetensors<W: Write>(
    writer: W,
    arrays: &[(String, Array)],
    metadata: &BTreeMap<String, String>,
) -> Result<(), CheckpointError> {
    let specifications = arrays
        .iter()
        .map(|(name, array)| ArraySpecification::new(name.clone(), array.r#type().into_owned()))
        .collect::<Vec<_>>();
    stream_safetensors(writer, &specifications, metadata, |index| Ok(Cow::Borrowed(&arrays[index].1)))
}

/// Writes one safetensors file to `writer` whose arrays are produced one at a time, so that at most one array
/// produced by `produce` is alive at any time. The header is built from the names and types of `specifications`
/// alone and written first. Then `produce` is called with the index of every entry of `specifications` in order, and
/// the array that it returns is checked against the type of that entry, written, and dropped before the next entry is
/// produced. The header is padded with spaces so that the data section starts at a multiple of 8 bytes, as the
/// reference implementation does, and the output is identical to that of [`write_safetensors`] for the same arrays.
///
/// # Parameters
///
///   - `writer`: Destination of the file contents.
///   - `specifications`: Name and type of every array, in the order of the data section.
///   - `metadata`: Free-form string metadata of the file.
///   - `produce`: Function that returns the array of the entry of `specifications` with the provided index.
///
/// # Errors
///
/// Returns [`CheckpointError::InvalidArray`] if two arrays share a name, if a name is the reserved `__metadata__`
/// key, if the data type of an array has no safetensors tag (e.g., a sub-byte data type), if a specified type has no
/// static shape, or if a produced array does not have the type of its specification, the first error of `produce`,
/// and [`CheckpointError::Io`] if `writer` fails. Every specification is validated before anything is written, while
/// errors of `produce` and mismatching produced arrays leave a partially written file behind.
pub fn stream_safetensors<'a, W: Write, F: FnMut(usize) -> Result<Cow<'a, Array>, CheckpointError>>(
    mut writer: W,
    specifications: &[ArraySpecification],
    metadata: &BTreeMap<String, String>,
    mut produce: F,
) -> Result<(), CheckpointError> {
    let mut header = serde_json::Map::new();
    if !metadata.is_empty() {
        header.insert(METADATA_KEY.to_string(), serde_json::json!(metadata));
    }

    let mut offset = 0u64;
    for specification in specifications {
        let (name, array_type) = (specification.name(), specification.r#type());
        let invalid = |message: String| CheckpointError::InvalidArray { name: name.to_string(), message };
        let data_type = array_type.data_type();
        let tag = data_type_tag(data_type)
            .ok_or_else(|| invalid(format!("data type `{data_type}` has no safetensors data type tag")))?;
        if name == METADATA_KEY || header.contains_key(name) {
            return Err(invalid("array name is reserved or used more than once".to_string()));
        }
        let shape = array_type
            .static_shape()
            .ok_or_else(|| invalid(format!("type `{array_type}` does not have a static shape")))?
            .dimensions()
            .to_vec();
        let byte_count = shape
            .iter()
            .try_fold((data_type.bit_width() / 8) as u64, |count, &extent| count.checked_mul(extent as u64))
            .ok_or_else(|| invalid("byte count overflows".to_string()))?;
        let end = offset + byte_count;
        header.insert(
            name.to_string(),
            serde_json::json!({ "dtype": tag, "shape": shape, "data_offsets": [offset, end] }),
        );
        offset = end;
    }

    let mut header_bytes = serde_json::to_vec(&header).expect("a JSON map of strings and numbers always serializes");
    header_bytes.resize(header_bytes.len().next_multiple_of(8), b' ');
    let io_error =
        |error: std::io::Error| CheckpointError::Io { path: "writer".to_string(), message: error.to_string() };
    writer.write_all(&(header_bytes.len() as u64).to_le_bytes()).map_err(io_error)?;
    writer.write_all(&header_bytes).map_err(io_error)?;
    for (index, specification) in specifications.iter().enumerate() {
        let (name, array_type) = (specification.name(), specification.r#type());
        let array = produce(index)?;
        if array.r#type().as_ref() != array_type {
            return Err(CheckpointError::InvalidArray {
                name: name.to_string(),
                message: format!(
                    "the produced array has type `{}` but its specification has type `{array_type}`",
                    array.r#type(),
                ),
            });
        }
        let data_type = array_type.data_type();
        let mut bytes = array.logical_bytes();
        if cfg!(target_endian = "big") {
            let width = data_type.bit_width().div_ceil(8) / if data_type.is_complex() { 2 } else { 1 };
            if width > 1 {
                bytes.chunks_exact_mut(width).for_each(<[u8]>::reverse);
            }
        }
        writer.write_all(&bytes).map_err(io_error)?;
    }
    writer.flush().map_err(io_error)
}

/// [`ArrayFormat`] that stores all arrays of a checkpointable in one safetensors file named [`Self::FILE_NAME`], which
/// is written with [`stream_safetensors`] through a [`FileWriter`](crate::filesystems::FileWriter) and read with
/// [`SafetensorsCheckpoint::from_checkpoint_path`], on any file system.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct SafetensorsFormat;

impl SafetensorsFormat {
    /// Name of the safetensors file in a checkpointable directory.
    pub const FILE_NAME: &str = "parameters.safetensors";
}

impl ArrayFormat for SafetensorsFormat {
    #[inline]
    fn name(&self) -> &'static str {
        "safetensors"
    }

    fn write<'a>(
        &self,
        directory: &CheckpointPath,
        specifications: &[ArraySpecification],
        produce: &mut dyn FnMut(usize) -> Result<Cow<'a, Array>, CheckpointError>,
    ) -> Result<(), CheckpointError> {
        let mut writer = directory.join(Self::FILE_NAME).writer()?;
        stream_safetensors(&mut writer, specifications, &BTreeMap::new(), produce)?;
        writer.finish()
    }

    fn open(&self, directory: &CheckpointPath) -> Result<Box<dyn Checkpoint>, CheckpointError> {
        Ok(Box::new(SafetensorsCheckpoint::from_checkpoint_path(&directory.join(Self::FILE_NAME))?))
    }
}

/// Parsed safetensors header, which keeps the array entries in the order in which the header lists them.
struct Header {
    /// Free-form string metadata stored under the reserved `__metadata__` key.
    metadata: BTreeMap<String, String>,

    /// Array entries by name.
    arrays: Vec<(String, HeaderEntry)>,
}

impl<'de> Deserialize<'de> for Header {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // A JSON object may repeat a key, and ordinary map deserialization silently keeps the last value. Visiting
        // the entries directly lets a repeated array name be reported instead.
        struct HeaderVisitor;

        impl<'de> Visitor<'de> for HeaderVisitor {
            type Value = Header;

            fn expecting(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a JSON object that maps array names to their descriptions")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Header, A::Error> {
                let mut metadata = None;
                let mut arrays = Vec::new();
                let mut names = HashSet::new();
                while let Some(name) = map.next_key::<String>()? {
                    if !names.insert(name.clone()) {
                        return Err(A::Error::custom(format!("array name `{name}` appears more than once")));
                    }
                    if name == METADATA_KEY {
                        metadata = map.next_value::<Option<BTreeMap<String, String>>>()?;
                    } else {
                        arrays.push((name, map.next_value::<HeaderEntry>()?));
                    }
                }
                Ok(Header { metadata: metadata.unwrap_or_default(), arrays })
            }
        }

        deserializer.deserialize_map(HeaderVisitor)
    }
}

/// Description of one array in a safetensors header.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HeaderEntry {
    /// Data type tag (e.g., `BF16`).
    dtype: String,

    /// Extents of the dimensions of the array.
    shape: Vec<usize>,

    /// Start and end of the byte range of the array contents, relative to the start of the data section.
    data_offsets: [u64; 2],
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use pretty_assertions::assert_eq;
    use ryft_core::{ArrayType, DataType, f4e2m1fn};

    use crate::arrays::tests::test_checkpoint;

    use super::*;

    /// Returns the bytes of a safetensors file with the provided JSON header and data section.
    fn file_bytes(header: &str, data: &[u8]) -> Vec<u8> {
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(data);
        bytes
    }

    #[test]
    fn test_safetensors_checkpoint_open() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.safetensors");
        let arrays = vec![("weights".to_string(), Array::vector(vec![1.5f32, -2.0]).unwrap())];
        write_safetensors(File::create(&path).unwrap(), &arrays, &BTreeMap::new()).unwrap();
        let checkpoint = SafetensorsCheckpoint::open(&path).unwrap();
        assert_eq!(checkpoint.source(), path.display().to_string());
        assert_eq!(checkpoint.read("weights"), Ok(Array::vector(vec![1.5f32, -2.0]).unwrap()));

        // A missing file reports the path that could not be opened.
        let missing = directory.path().join("missing.safetensors");
        assert!(matches!(
            SafetensorsCheckpoint::open(&missing),
            Err(CheckpointError::Io { path, .. }) if path == missing.display().to_string(),
        ));
    }

    #[test]
    fn test_safetensors_checkpoint_from_checkpoint_path() {
        let directory = tempfile::tempdir().unwrap();
        let path = CheckpointPath::from(directory.path()).join("model.safetensors");
        let arrays = vec![("weights".to_string(), Array::vector(vec![1.5f32, -2.0]).unwrap())];
        let mut bytes = Vec::new();
        write_safetensors(&mut bytes, &arrays, &BTreeMap::new()).unwrap();
        path.write(&bytes).unwrap();
        let checkpoint = SafetensorsCheckpoint::from_checkpoint_path(&path).unwrap();
        assert_eq!(checkpoint.source(), path.to_string());
        assert_eq!(checkpoint.read("weights"), Ok(Array::vector(vec![1.5f32, -2.0]).unwrap()));
        assert!(matches!(
            SafetensorsCheckpoint::from_checkpoint_path(&path.join("missing")),
            Err(CheckpointError::Io { .. }),
        ));
    }

    #[test]
    fn test_safetensors_checkpoint_from_reader() {
        let header = concat!(
            r#"{"b":{"dtype":"F32","shape":[1],"data_offsets":[2,6]},"#,
            r#""a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"__metadata__":{"format":"pt"}}"#,
        );
        let checkpoint =
            SafetensorsCheckpoint::from_reader(Cursor::new(file_bytes(header, &[1, 2, 0, 0, 128, 63])), "memory")
                .unwrap();
        assert_eq!(checkpoint.source(), "memory");
        assert_eq!(checkpoint.data_offset(), 8 + header.len() as u64);
        assert_eq!(checkpoint.metadata(), &BTreeMap::from([("format".to_string(), "pt".to_string())]));

        // Descriptors are ordered by byte range rather than by header order.
        assert_eq!(
            checkpoint.arrays(),
            &[ArrayDescriptor::new("a", DataType::U8, [2], 0..2), ArrayDescriptor::new("b", DataType::F32, [1], 2..6),],
        );
        assert_eq!(checkpoint.array("b"), Some(&ArrayDescriptor::new("b", DataType::F32, [1], 2..6)));
        assert_eq!(checkpoint.array("c"), None);
        assert_eq!(checkpoint.read("a"), Ok(Array::vector(vec![1u8, 2]).unwrap()));
        assert_eq!(checkpoint.read("b"), Ok(Array::vector(vec![1.0f32]).unwrap()));
    }

    #[test]
    fn test_safetensors_checkpoint_from_reader_invalid_header() {
        let open = |bytes: Vec<u8>| SafetensorsCheckpoint::from_reader(Cursor::new(bytes), "memory").map(|_| ());
        assert_eq!(
            open(vec![0; 7]),
            Err(CheckpointError::InvalidHeader {
                message: "file holds 7 bytes, which is fewer than the 8 bytes of the header size".to_string(),
            }),
        );
        assert_eq!(
            open(200_000_000u64.to_le_bytes().to_vec()),
            Err(CheckpointError::InvalidHeader {
                message: "header size 200000000 exceeds the maximum supported size 100000000".to_string(),
            }),
        );
        assert_eq!(
            open(file_bytes("{}", &[])[..9].to_vec()),
            Err(CheckpointError::InvalidHeader {
                message: "header size 2 exceeds the 1 bytes that follow the header size".to_string(),
            }),
        );
        assert!(matches!(
            open(file_bytes("[]", &[])),
            Err(CheckpointError::InvalidHeader { message })
                if message.starts_with("invalid type: sequence, expected a JSON object that maps array names"),
        ));
        assert!(matches!(
            open(file_bytes(
                concat!(
                    r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"#,
                    r#""a":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#,
                ),
                &[0, 0],
            )),
            Err(CheckpointError::InvalidHeader { message })
                if message.starts_with("array name `a` appears more than once"),
        ));
        assert!(matches!(
            open(file_bytes(r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1],"extra":1}}"#, &[0])),
            Err(CheckpointError::InvalidHeader { message }) if message.starts_with("unknown field `extra`"),
        ));
        assert_eq!(
            open(file_bytes(r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#, &[0, 0])),
            Err(CheckpointError::InvalidHeader {
                message: "arrays cover 1 bytes but the data section holds 2 bytes".to_string(),
            }),
        );
    }

    #[test]
    fn test_safetensors_checkpoint_from_reader_invalid_array() {
        let open = |header: &str, data: &[u8]| {
            SafetensorsCheckpoint::from_reader(Cursor::new(file_bytes(header, data)), "memory").map(|_| ())
        };
        let invalid =
            |message: &str| Err(CheckpointError::InvalidArray { name: "a".to_string(), message: message.to_string() });
        assert_eq!(
            open(r#"{"a":{"dtype":"F4","shape":[1],"data_offsets":[0,1]}}"#, &[0]),
            invalid("unsupported data type tag `F4`"),
        );
        assert_eq!(
            open(r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[1,0]}}"#, &[0]),
            invalid("byte range 1..0 ends before it starts"),
        );
        assert_eq!(
            open(r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#, &[0; 4]),
            invalid("byte range 0..4 holds 4 bytes but `f32` elements of shape [2] occupy 8 bytes"),
        );
        assert_eq!(
            open(
                concat!(
                    r#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[1,3]},"#,
                    r#""b":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#,
                ),
                &[0; 3],
            ),
            invalid("byte range 1..3 overlaps the preceding array, which ends at 2"),
        );
        assert_eq!(
            open(
                concat!(
                    r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[2,3]},"#,
                    r#""b":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
                ),
                &[0; 3],
            ),
            invalid("byte range 2..3 leaves a gap after the preceding array, which ends at 1"),
        );
    }

    #[test]
    fn test_safetensors_checkpoint_read_bytes() {
        let checkpoint = test_checkpoint();
        assert_eq!(checkpoint.read_bytes("stack", 4..12), Ok([1.0f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat()));
        assert_eq!(checkpoint.read_bytes("stack", 24..24), Ok(Vec::new()));
        assert_eq!(
            checkpoint.read_bytes("stack", 20..28),
            Err(CheckpointError::InvalidArray {
                name: "stack".to_string(),
                message: "byte range 20..28 is not within the 24 bytes of the array".to_string(),
            }),
        );
        assert_eq!(
            checkpoint.read_bytes("missing", 0..1),
            Err(CheckpointError::UnknownArray { name: "missing".to_string() }),
        );
    }

    #[test]
    fn test_write_safetensors() {
        let arrays = vec![
            ("b".to_string(), Array::vector(vec![1u8, 2, 3]).unwrap()),
            ("a".to_string(), Array::scalar(2.0f32).unwrap()),
        ];
        let metadata = BTreeMap::from([("format".to_string(), "pt".to_string())]);
        let mut bytes = Vec::new();
        write_safetensors(&mut bytes, &arrays, &metadata).unwrap();

        // The header lists its keys in sorted order and is padded with spaces to a multiple of 8 bytes, while the
        // data section keeps the order in which the arrays were provided.
        let header = concat!(
            r#"{"__metadata__":{"format":"pt"},"a":{"data_offsets":[3,7],"dtype":"F32","shape":[]},"#,
            r#""b":{"data_offsets":[0,3],"dtype":"U8","shape":[3]}}"#,
        );
        let padded_header_byte_count = header.len().next_multiple_of(8);
        assert_eq!(&bytes[..8], &(padded_header_byte_count as u64).to_le_bytes());
        assert_eq!(std::str::from_utf8(&bytes[8..8 + header.len()]), Ok(header));
        assert!(bytes[8 + header.len()..8 + padded_header_byte_count].iter().all(|&byte| byte == b' '));
        assert_eq!(&bytes[8 + padded_header_byte_count..], &[1, 2, 3, 0, 0, 0, 64]);

        let checkpoint = SafetensorsCheckpoint::from_reader(Cursor::new(bytes), "memory").unwrap();
        assert_eq!(checkpoint.metadata(), &metadata);
        assert_eq!(checkpoint.read("a"), Ok(arrays[1].1.clone()));
        assert_eq!(checkpoint.read("b"), Ok(arrays[0].1.clone()));
    }

    #[test]
    fn test_write_safetensors_invalid_array() {
        let write = |arrays: &[(String, Array)]| write_safetensors(Vec::new(), arrays, &BTreeMap::new());
        let scalar = Array::scalar(1u8).unwrap();
        assert_eq!(
            write(&[("a".to_string(), scalar.clone()), ("a".to_string(), scalar.clone())]),
            Err(CheckpointError::InvalidArray {
                name: "a".to_string(),
                message: "array name is reserved or used more than once".to_string(),
            }),
        );
        assert_eq!(
            write(&[("__metadata__".to_string(), scalar)]),
            Err(CheckpointError::InvalidArray {
                name: "__metadata__".to_string(),
                message: "array name is reserved or used more than once".to_string(),
            }),
        );
        let narrow =
            Array::from_elements(ArrayType::new_static(DataType::F4E2M1FN, [1]), &[f4e2m1fn::from_f32(1.0).unwrap()])
                .unwrap();
        assert_eq!(
            write(&[("narrow".to_string(), narrow)]),
            Err(CheckpointError::InvalidArray {
                name: "narrow".to_string(),
                message: "data type `f4e2m1fn` has no safetensors data type tag".to_string(),
            }),
        );
    }

    #[test]
    fn test_stream_safetensors() {
        let arrays = vec![
            ("b".to_string(), Array::vector(vec![1u8, 2, 3]).unwrap()),
            ("a".to_string(), Array::scalar(2.0f32).unwrap()),
        ];
        let entries = arrays
            .iter()
            .map(|(name, array)| ArraySpecification::new(name.clone(), array.r#type().into_owned()))
            .collect::<Vec<_>>();
        let mut expected = Vec::new();
        write_safetensors(&mut expected, &arrays, &BTreeMap::new()).unwrap();

        // Every entry is produced exactly once and in order, and the bytes equal those of `write_safetensors`.
        let mut produced = Vec::new();
        let mut bytes = Vec::new();
        assert_eq!(
            stream_safetensors(&mut bytes, &entries, &BTreeMap::new(), |index| {
                produced.push(index);
                Ok(Cow::Owned(arrays[index].1.clone()))
            }),
            Ok(()),
        );
        assert_eq!(produced, vec![0, 1]);
        assert_eq!(bytes, expected);

        // Produced arrays must have their declared types, and errors of the producing function are returned unchanged.
        assert_eq!(
            stream_safetensors(Vec::new(), &entries, &BTreeMap::new(), |_| Ok(Cow::Borrowed(&arrays[1].1))),
            Err(CheckpointError::InvalidArray {
                name: "b".to_string(),
                message: "the produced array has type `f32[]` but its specification has type `u8[3]`".to_string(),
            }),
        );
        assert_eq!(
            stream_safetensors(Vec::new(), &entries, &BTreeMap::new(), |_| {
                Err(CheckpointError::Import { message: "unavailable".to_string() })
            }),
            Err(CheckpointError::Import { message: "unavailable".to_string() }),
        );
    }
}
