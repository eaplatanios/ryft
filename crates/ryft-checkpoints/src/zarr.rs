//! The [Zarr version 3](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html) array format, which is
//! available with the `zarr` feature.
//!
//! [`ZarrFormat`] stores the arrays of one checkpointable as one Zarr v3 group rooted at the checkpointable directory,
//! with one Zarr v3 array per array of the checkpointable. This is the layout that TensorStore writes when Orbax is
//! configured with `use_zarr3` and without OCDBT (i.e., one directory per leaf), so the arrays of a checkpoint are
//! readable by any Zarr v3 implementation. The group and every array have a `zarr.json` metadata document:
//!
//!   - The group document records the names of the arrays in the order in which they were written under the
//!     attribute `ryft.arrays`, which is the order in which [`ZarrCheckpoint`] lists them.
//!   - Every array document has the `regular` chunk grid, the `default` chunk key encoding with the `/` separator, the
//!     fill value that the specification prescribes for zero of its data type, and the `bytes` codec with
//!     little-endian byte order as its only codec, so chunks are not compressed. A compressing codec (e.g., `zstd`)
//!     would be a further entry in the same codec pipeline. The original array name is recorded under the attribute
//!     `ryft.name`, and `dimension_names` are omitted.
//!
//! # Data Types
//!
//! Data types map to the core Zarr data types (i.e., `bool`, `int8` to `int64`, `uint8` to `uint64`, `float16`,
//! `float32`, `float64`, `complex64`, and `complex128`), and `bf16` maps to the registered extension data type
//! `bfloat16`. Data types without a Zarr name that have a data type tag in this crate (e.g., `f8e4m3fn`) are stored as
//! the raw bits data type `r<bits>`, sized by the host storage width of one element (i.e., the bit width rounded up to
//! whole bytes, which is how host arrays store elements), and the tag is recorded under the attribute
//! `ryft.data_type`, so that reading restores the exact data type. The fill value of a raw bits data type is written
//! as one zero per byte. The specification states that its length equals `<N>`, which reads as the number of bits,
//! but every integer of the array describes one byte, so this format writes one integer per byte.
//!
//! # Chunks
//!
//! By default every array is one chunk. [`ZarrFormat::with_chunk_byte_size`] splits arrays along their leading
//! dimension into chunks of at most that many bytes (and at least one leading entry), which mirrors the
//! `chunk_byte_size` of the `SaveArgs` of Orbax, and leaves the other dimensions unchunked. Because the regular chunk
//! grid has a fixed chunk shape, the last chunk along the leading dimension is padded with the fill value up to the
//! full chunk shape, and reading ignores the padding. Chunk keys are `c/<i>/<j>/...` and `c` alone for rank-0 arrays.
//! Arrays without elements write no chunks, and the extents of their chunk shape are at least one, as the
//! specification requires. This writer writes every chunk, so [`ZarrCheckpoint`] treats a missing chunk as a damaged
//! checkpoint and fails, instead of reading the fill value as the specification permits. It reads arrays whose chunks
//! span every dimension except possibly the leading one, which covers everything this format writes.
//!
//! # Concurrent Writes
//!
//! Every file of a group is written and made durable on its own, and for checkpoints with many arrays that cost
//! (e.g., one `fsync` per file on a local disk or one request per object in an object store) dominates a save. The
//! format therefore writes files on up to [`ZarrFormat::write_concurrency`] threads, as TensorStore does for Orbax.
//! Arrays are still produced one at a time and in order. Files of more than [`CONCURRENT_WRITE_BYTE_LIMIT`] bytes are
//! written by the thread that produces the arrays, so the bytes that wait to be written are bounded by that limit
//! times the number of threads, and large arrays are in memory one at a time. How much concurrency helps depends on
//! how the storage makes files durable. It helps most where concurrent synchronizations are committed together (e.g.,
//! by the journal of ext4).
//!
//! # Names
//!
//! Every array is stored in the child node named by its array name. Array names here are parameter paths such as
//! `$.layers[0].weight`, which are valid Zarr node names (i.e., non-empty, without `/`, not only periods, without
//! the reserved `__` prefix, and not `zarr.json`). Names that are not valid are encoded reversibly with percent
//! encoding (i.e., `%` followed by two uppercase hexadecimal digits per byte). The encoding escapes `%` and `/`
//! everywhere, the leading `_` of a `__` prefix, and every `.` of a name that consists of periods or that is
//! `zarr.json`. Empty names are rejected. Readers find arrays through the names recorded in the attributes, so they
//! never need to decode node names.
//!
//! # Byte Ranges
//!
//! The [`ArrayDescriptor`]s of a [`ZarrCheckpoint`] carry virtual byte ranges, because the arrays do not share one
//! file. The range of every array starts where the range of the previous array ends, in the listed order, so the
//! ranges describe the sizes of the arrays and a consistent layout of the whole checkpointable.

use std::borrow::Cow;
use std::ops::Range;
use std::sync::Mutex;
use std::sync::mpsc::sync_channel;
use std::thread::{self, Builder};

use ryft_core::{Array, DataType, Typed};
use serde_json::{Value, json};

use crate::arrays::{ArrayDescriptor, ArraySpecification, Checkpoint, data_type_from_tag, data_type_tag};
use crate::errors::CheckpointError;
use crate::filesystems::CheckpointPath;
use crate::formats::ArrayFormat;

// TODO(eaplatanios): Review this module.

/// Name of the metadata document of every Zarr node.
pub const ZARR_METADATA_FILE_NAME: &str = "zarr.json";

/// Largest number of bytes of a file that [`ZarrFormat`] hands to its writer threads. Larger files are written by the
/// thread that produces the arrays, which bounds the bytes that wait to be written.
pub const CONCURRENT_WRITE_BYTE_LIMIT: usize = 8 << 20;

/// Number of threads on which a [`ZarrFormat`] writes files by default. Writer threads mostly wait for storage, so
/// the number is not tied to the number of processor cores.
pub const DEFAULT_WRITE_CONCURRENCY: usize = 32;

/// [`ArrayFormat`] that stores the arrays of a checkpointable as a Zarr v3 group. Refer to the
/// [module documentation](self) for the layout, data types, chunking, concurrent writes, and names.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ZarrFormat {
    /// Largest number of bytes of one chunk, if arrays are chunked.
    chunk_byte_size: Option<u64>,

    /// Largest number of threads that write files at the same time.
    write_concurrency: usize,
}

impl ZarrFormat {
    /// Creates a new [`ZarrFormat`] that stores every array as one chunk and writes files on up to
    /// [`DEFAULT_WRITE_CONCURRENCY`] threads.
    #[inline]
    pub fn new() -> Self {
        Self { chunk_byte_size: None, write_concurrency: DEFAULT_WRITE_CONCURRENCY }
    }

    /// Returns this format with arrays split along their leading dimension into chunks of at most `chunk_byte_size`
    /// bytes and at least one leading entry, or stored as one chunk when it is [`None`].
    #[inline]
    pub fn with_chunk_byte_size(mut self, chunk_byte_size: Option<u64>) -> Self {
        self.chunk_byte_size = chunk_byte_size;
        self
    }

    /// Returns this format with files written on up to `write_concurrency` threads at the same time. With a
    /// concurrency of one (or zero, which behaves like one), every file is written by the calling thread.
    #[inline]
    pub fn with_write_concurrency(mut self, write_concurrency: usize) -> Self {
        self.write_concurrency = write_concurrency.max(1);
        self
    }

    /// Returns the largest number of bytes of one chunk, if arrays are chunked.
    #[inline]
    pub fn chunk_byte_size(&self) -> Option<u64> {
        self.chunk_byte_size
    }

    /// Returns the largest number of threads that write files at the same time.
    #[inline]
    pub fn write_concurrency(&self) -> usize {
        self.write_concurrency
    }

    /// Returns the chunk shape of an array of `shape` whose elements occupy `element_byte_count` bytes each.
    fn chunk_shape(&self, shape: &[usize], element_byte_count: usize) -> Vec<usize> {
        let mut chunk_shape = shape.iter().map(|extent| (*extent).max(1)).collect::<Vec<_>>();
        if let (Some(chunk_byte_size), Some(leading)) = (self.chunk_byte_size, chunk_shape.first_mut()) {
            let entry_byte_count = shape[1..].iter().product::<usize>() as u64 * element_byte_count as u64;
            let entries = chunk_byte_size.checked_div(entry_byte_count).map_or(u64::MAX, |entries| entries.max(1));
            *leading = (*leading as u64).min(entries) as usize;
        }
        chunk_shape
    }
}

impl Default for ZarrFormat {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl ArrayFormat for ZarrFormat {
    #[inline]
    fn name(&self) -> &'static str {
        "zarr"
    }

    fn write<'a>(
        &self,
        directory: &CheckpointPath,
        specifications: &[ArraySpecification],
        produce: &mut dyn FnMut(usize) -> Result<Cow<'a, Array>, CheckpointError>,
    ) -> Result<(), CheckpointError> {
        // Every specification is validated before anything is written.
        let mut layouts = Vec::with_capacity(specifications.len());
        for specification in specifications {
            let name = specification.name();
            let invalid = |message: String| CheckpointError::InvalidArray { name: name.to_string(), message };
            let r#type = specification.r#type();
            let data_type = ZarrDataType::from_data_type(r#type.data_type()).map_err(invalid)?;
            let shape = r#type
                .static_shape()
                .ok_or_else(|| invalid(format!("type `{type}` does not have a static shape")))?
                .dimensions()
                .to_vec();
            let node_name = encode_node_name(name).map_err(invalid)?;
            // Node names are directory names, so names that differ only in case would share one directory on file
            // systems that ignore case (e.g., the default ones of macOS and Windows).
            let folded_name = node_name.to_lowercase();
            if layouts.iter().any(|layout: &ZarrArrayLayout| layout.node_name.to_lowercase() == folded_name) {
                return Err(invalid(
                    "array name is used more than once, or differs from another array name only in case".to_string(),
                ));
            }
            let chunk_shape = self.chunk_shape(&shape, data_type.element_byte_count);
            layouts.push(ZarrArrayLayout { data_type, node_name, shape, chunk_shape });
        }

        directory.create_directory()?;
        let names = specifications.iter().map(ArraySpecification::name).collect::<Vec<_>>();
        let group = json!({"zarr_format": 3, "node_type": "group", "attributes": {"ryft": {"arrays": names}}});
        directory.join(ZARR_METADATA_FILE_NAME).write(&document_bytes(&group))?;

        // A file is written by the calling thread when there is nothing to write concurrently with.
        let worker_count = self.write_concurrency.min(specifications.len());
        if worker_count <= 1 {
            return write_arrays(directory, specifications, &layouts, produce, &mut |path, bytes| path.write(&bytes));
        }

        // The channel has no buffer, so the files that wait to be written are the ones that the writer threads hold
        // and the one that is being handed over. Writer threads keep receiving after a failure, so that handing a
        // file over never blocks forever, and the first failure is returned once the calling thread notices it.
        let (sender, receiver) = sync_channel::<(CheckpointPath, Vec<u8>)>(0);
        let receiver = Mutex::new(receiver);
        let failure = Mutex::new(None::<CheckpointError>);
        let written = thread::scope(|scope| {
            for _ in 0..worker_count {
                let worker = Builder::new().name("ryft-checkpoint-zarr-write".to_string()).spawn_scoped(scope, || {
                    loop {
                        let file = receiver.lock().map(|receiver| receiver.recv());
                        let Ok(Ok((path, bytes))) = file else {
                            break;
                        };
                        if let Err(error) = path.write(&bytes)
                            && let Ok(mut failure) = failure.lock()
                        {
                            failure.get_or_insert(error);
                        }
                    }
                });
                if let Err(error) = worker {
                    // Dropping the sender lets the writer threads that did start finish.
                    drop(sender);
                    return Err(CheckpointError::Io { path: directory.to_string(), message: error.to_string() });
                }
            }
            let written = write_arrays(directory, specifications, &layouts, produce, &mut |path, bytes| {
                if let Some(error) = failure.lock().ok().and_then(|mut failure| failure.take()) {
                    return Err(error);
                }
                if bytes.len() > CONCURRENT_WRITE_BYTE_LIMIT {
                    return path.write(&bytes);
                }
                sender.send((path, bytes)).map_err(|_| CheckpointError::Io {
                    path: directory.to_string(),
                    message: "the threads that write Zarr files have stopped".to_string(),
                })
            });
            drop(sender);
            written
        });
        written?;
        failure.into_inner().ok().flatten().map_or(Ok(()), Err)
    }

    fn open(&self, directory: &CheckpointPath) -> Result<Box<dyn Checkpoint>, CheckpointError> {
        Ok(Box::new(ZarrCheckpoint::open(directory)?))
    }
}

/// [`Checkpoint`] over a Zarr v3 group that [`ZarrFormat`] wrote, which works on every
/// [`FileSystem`](crate::filesystems::FileSystem) and reads only the chunks that a read needs. Refer to the
/// [module documentation](self) for the virtual byte ranges of its descriptors.
#[derive(Clone, Debug)]
pub struct ZarrCheckpoint {
    /// Root directory of the group.
    directory: CheckpointPath,

    /// Descriptors of all arrays, in the order of the group attributes.
    arrays: Vec<ArrayDescriptor>,

    /// Storage of every array, in the same order.
    storages: Vec<ZarrArrayStorage>,
}

impl ZarrCheckpoint {
    /// Opens the Zarr v3 group rooted at `directory` and validates the metadata of all of its arrays. Arrays are
    /// listed in the order that the group attribute `ryft.arrays` records, or, when the group has no such attribute
    /// (e.g., because another tool wrote it), in the order of their node names.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::Io`] if a metadata document cannot be read, and [`CheckpointError::InvalidFile`] if
    /// one is malformed or uses a feature that this reader does not support (e.g., a compressing codec).
    pub fn open(directory: &CheckpointPath) -> Result<Self, CheckpointError> {
        let group_path = directory.join(ZARR_METADATA_FILE_NAME);
        let group = read_document(&group_path)?;
        let invalid = |message: String| CheckpointError::InvalidFile { path: group_path.to_string(), message };
        if group.get("zarr_format") != Some(&json!(3)) || group.get("node_type") != Some(&json!("group")) {
            return Err(invalid("the document does not describe a Zarr v3 group".to_string()));
        }
        let names = match group.pointer("/attributes/ryft/arrays") {
            Some(Value::Array(names)) => names
                .iter()
                .map(|name| name.as_str().map(str::to_string))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| invalid("attribute `ryft.arrays` must hold strings".to_string()))?,
            Some(_) => return Err(invalid("attribute `ryft.arrays` must be an array".to_string())),
            None => directory
                .list()?
                .into_iter()
                .filter(|name| name != ZARR_METADATA_FILE_NAME)
                .map(|name| decode_node_name(&name))
                .collect(),
        };

        let mut arrays = Vec::with_capacity(names.len());
        let mut storages = Vec::with_capacity(names.len());
        let mut offset = 0u64;
        for name in names {
            let node = directory.join(encode_node_name(&name).map_err(invalid)?);
            let storage = ZarrArrayStorage::open(node)?;
            let byte_count = storage.byte_count();
            arrays.push(ArrayDescriptor::new(
                name,
                storage.data_type,
                storage.shape.clone(),
                offset..offset + byte_count,
            ));
            storages.push(storage);
            offset += byte_count;
        }
        Ok(Self { directory: directory.clone(), arrays, storages })
    }

    /// Returns the root directory of the group.
    #[inline]
    pub fn directory(&self) -> &CheckpointPath {
        &self.directory
    }
}

impl Checkpoint for ZarrCheckpoint {
    #[inline]
    fn arrays(&self) -> &[ArrayDescriptor] {
        &self.arrays
    }

    fn array(&self, name: &str) -> Option<&ArrayDescriptor> {
        self.arrays.iter().find(|descriptor| descriptor.name() == name)
    }

    fn read_bytes(&self, name: &str, byte_range: Range<u64>) -> Result<Vec<u8>, CheckpointError> {
        let index = self
            .arrays
            .iter()
            .position(|descriptor| descriptor.name() == name)
            .ok_or_else(|| CheckpointError::UnknownArray { name: name.to_string() })?;
        let byte_count = self.arrays[index].byte_count();
        if byte_range.start > byte_range.end || byte_range.end > byte_count {
            return Err(CheckpointError::InvalidArray {
                name: name.to_string(),
                message: format!(
                    "byte range {}..{} is not within the {byte_count} bytes of the array",
                    byte_range.start, byte_range.end,
                ),
            });
        }
        self.storages[index].read(byte_range)
    }
}

/// Layout of one array in a Zarr group that [`ZarrFormat`] writes.
struct ZarrArrayLayout {
    /// Zarr data type of the array.
    data_type: ZarrDataType,

    /// Name of the child node that stores the array.
    node_name: String,

    /// Shape of the array.
    shape: Vec<usize>,

    /// Shape of every chunk of the array.
    chunk_shape: Vec<usize>,
}

/// Writes the metadata documents and the chunks of the arrays that `specifications` describe into the existing group
/// `directory`, producing the arrays one at a time and in order. Directories are created here, and every file is
/// handed to `write` together with its contents, which either writes it immediately or passes it to a writer thread.
fn write_arrays<'a>(
    directory: &CheckpointPath,
    specifications: &[ArraySpecification],
    layouts: &[ZarrArrayLayout],
    produce: &mut dyn FnMut(usize) -> Result<Cow<'a, Array>, CheckpointError>,
    write: &mut dyn FnMut(CheckpointPath, Vec<u8>) -> Result<(), CheckpointError>,
) -> Result<(), CheckpointError> {
    for (index, (specification, layout)) in specifications.iter().zip(layouts).enumerate() {
        let ZarrArrayLayout { data_type, node_name, shape, chunk_shape } = layout;
        let node = directory.join(node_name);
        node.create_directory()?;
        let document = array_document(specification.name(), data_type, shape, chunk_shape);
        write(node.join(ZARR_METADATA_FILE_NAME), document_bytes(&document))?;

        let array = produce(index)?;
        if array.r#type().as_ref() != specification.r#type() {
            return Err(CheckpointError::InvalidArray {
                name: specification.name().to_string(),
                message: format!(
                    "the produced array has type `{}` but its specification has type `{}`",
                    array.r#type(),
                    specification.r#type(),
                ),
            });
        }
        let bytes = little_endian_bytes(&array, data_type.data_type);
        drop(array);
        if shape.contains(&0) {
            continue;
        }
        if shape.is_empty() {
            write(node.join("c"), bytes)?;
            continue;
        }
        let entry_byte_count = bytes.len() / shape[0];
        let chunk_byte_count = chunk_shape[0] * entry_byte_count;
        let chunk_path = |chunk: usize| -> Result<CheckpointPath, CheckpointError> {
            let chunk_path = node.join(format!("c/{chunk}{}", "/0".repeat(shape.len() - 1)));
            if let Some(parent) = chunk_path.parent() {
                parent.create_directory()?;
            }
            Ok(chunk_path)
        };
        if bytes.len() == chunk_byte_count {
            // An array that is one whole chunk is written without copying its bytes again.
            write(chunk_path(0)?, bytes)?;
            continue;
        }
        for (chunk, contents) in bytes.chunks(chunk_byte_count).enumerate() {
            let mut contents = contents.to_vec();
            // The fill value of every data type that this format writes is encoded as zero bytes.
            contents.resize(chunk_byte_count, 0);
            write(chunk_path(chunk)?, contents)?;
        }
    }
    Ok(())
}

/// Zarr data type of one array, together with the Ryft data type that it stores.
#[derive(Clone, Debug, PartialEq)]
struct ZarrDataType {
    /// Ryft data type of the elements.
    data_type: DataType,

    /// Zarr data type name (e.g., `float32` or `r8`).
    name: String,

    /// Data type tag of the crate that is recorded for raw bits data types.
    tag: Option<&'static str>,

    /// Number of bytes that one element occupies.
    element_byte_count: usize,
}

impl ZarrDataType {
    /// Returns the Zarr data type that stores `data_type`, or an error message if there is none.
    fn from_data_type(data_type: DataType) -> Result<Self, String> {
        let element_byte_count = data_type.bit_width().div_ceil(8);
        let name = match data_type {
            DataType::Boolean => "bool",
            DataType::I8 => "int8",
            DataType::I16 => "int16",
            DataType::I32 => "int32",
            DataType::I64 => "int64",
            DataType::U8 => "uint8",
            DataType::U16 => "uint16",
            DataType::U32 => "uint32",
            DataType::U64 => "uint64",
            DataType::F16 => "float16",
            DataType::BF16 => "bfloat16",
            DataType::F32 => "float32",
            DataType::F64 => "float64",
            DataType::C64 => "complex64",
            DataType::C128 => "complex128",
            _ => {
                let tag = data_type_tag(data_type)
                    .filter(|_| element_byte_count > 0)
                    .ok_or_else(|| format!("data type `{data_type}` has neither a Zarr name nor a data type tag"))?;
                return Ok(Self {
                    data_type,
                    name: format!("r{}", element_byte_count * 8),
                    tag: Some(tag),
                    element_byte_count,
                });
            }
        };
        Ok(Self { data_type, name: name.to_string(), tag: None, element_byte_count })
    }

    /// Returns the Zarr data type named `name` whose raw bits data type, if any, stores the data type tagged `tag`.
    fn from_name(name: &str, tag: Option<&str>) -> Result<Self, String> {
        if let Some(bits) = name.strip_prefix('r') {
            if bits.parse::<usize>().is_err() {
                return Err(format!("unsupported data type `{name}`"));
            }
            let tag = tag.ok_or_else(|| format!("raw data type `{name}` needs the attribute `ryft.data_type`"))?;
            let data_type = data_type_from_tag(tag).ok_or_else(|| format!("unsupported data type tag `{tag}`"))?;
            let zarr_data_type = Self::from_data_type(data_type)?;
            if zarr_data_type.name != name {
                return Err(format!("raw data type `{name}` does not match the data type tag `{tag}`"));
            }
            return Ok(zarr_data_type);
        }
        let data_type = match name {
            "bool" => DataType::Boolean,
            "int8" => DataType::I8,
            "int16" => DataType::I16,
            "int32" => DataType::I32,
            "int64" => DataType::I64,
            "uint8" => DataType::U8,
            "uint16" => DataType::U16,
            "uint32" => DataType::U32,
            "uint64" => DataType::U64,
            "float16" => DataType::F16,
            "bfloat16" => DataType::BF16,
            "float32" => DataType::F32,
            "float64" => DataType::F64,
            "complex64" => DataType::C64,
            "complex128" => DataType::C128,
            _ => return Err(format!("unsupported data type `{name}`")),
        };
        Self::from_data_type(data_type)
    }

    /// Returns the JSON fill value that represents zero of this data type, following the permitted fill values of the
    /// specification.
    fn fill_value(&self) -> Value {
        match self.data_type {
            DataType::Boolean => json!(false),
            DataType::F16 | DataType::BF16 | DataType::F32 | DataType::F64 => json!(0.0),
            DataType::C64 | DataType::C128 => json!([0.0, 0.0]),
            _ if self.tag.is_some() => json!(vec![0; self.element_byte_count]),
            _ => json!(0),
        }
    }
}

/// Location and chunking of one stored array.
#[derive(Clone, Debug)]
struct ZarrArrayStorage {
    /// Directory of the array node.
    node: CheckpointPath,

    /// Ryft data type of the elements.
    data_type: DataType,

    /// Number of bytes that one element occupies.
    element_byte_count: usize,

    /// Extents of the dimensions of the array.
    shape: Vec<usize>,

    /// Number of leading entries of one chunk.
    chunk_entries: usize,

    /// Separator of the chunk keys.
    separator: char,
}

impl ZarrArrayStorage {
    /// Reads and validates the metadata of the array node `node`.
    fn open(node: CheckpointPath) -> Result<Self, CheckpointError> {
        let path = node.join(ZARR_METADATA_FILE_NAME);
        let document = read_document(&path)?;
        let invalid = |message: String| CheckpointError::InvalidFile { path: path.to_string(), message };
        if document.get("zarr_format") != Some(&json!(3)) || document.get("node_type") != Some(&json!("array")) {
            return Err(invalid("the document does not describe a Zarr v3 array".to_string()));
        }
        let extents = |value: Option<&Value>, field: &str| {
            value
                .and_then(Value::as_array)
                .and_then(|extents| {
                    extents.iter().map(|extent| extent.as_u64().map(|extent| extent as usize)).collect()
                })
                .ok_or_else(|| invalid(format!("field `{field}` must be an array of non-negative integers")))
        };
        let shape: Vec<usize> = extents(document.get("shape"), "shape")?;
        let data_type_name = document
            .get("data_type")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("field `data_type` must be a string".to_string()))?;
        let tag = document.pointer("/attributes/ryft/data_type").and_then(Value::as_str);
        let data_type = ZarrDataType::from_name(data_type_name, tag).map_err(invalid)?;
        if document.get("fill_value").is_none() {
            return Err(invalid("field `fill_value` is missing".to_string()));
        }

        // The regular chunk grid must span every dimension except possibly the leading one.
        if document.pointer("/chunk_grid/name") != Some(&json!("regular")) {
            return Err(invalid("only the `regular` chunk grid is supported".to_string()));
        }
        let chunk_shape: Vec<usize> =
            extents(document.pointer("/chunk_grid/configuration/chunk_shape"), "chunk_shape")?;
        if chunk_shape.len() != shape.len() || chunk_shape.contains(&0) {
            return Err(invalid("the chunk shape must have one positive extent per dimension".to_string()));
        }
        let element_count = shape.iter().product::<usize>();
        if element_count > 0 && chunk_shape.iter().zip(&shape).skip(1).any(|(chunk, extent)| chunk != extent) {
            return Err(invalid("only chunks that span every dimension but the leading one are supported".to_string()));
        }

        let separator = match document.pointer("/chunk_key_encoding/name") {
            Some(name) if name == "default" => match document.pointer("/chunk_key_encoding/configuration/separator") {
                None => '/',
                Some(separator) if separator == "/" => '/',
                Some(separator) if separator == "." => '.',
                Some(separator) => return Err(invalid(format!("unsupported chunk key separator {separator}"))),
            },
            _ => return Err(invalid("only the `default` chunk key encoding is supported".to_string())),
        };

        // The `bytes` codec with little-endian byte order must be the only codec, and its byte order may only be
        // omitted for single byte data types.
        let codecs = document
            .get("codecs")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("field `codecs` must be an array".to_string()))?;
        for codec in codecs {
            let name = codec.get("name").and_then(Value::as_str).unwrap_or_default();
            if name != "bytes" {
                return Err(invalid(format!("unsupported codec `{name}`")));
            }
            match codec.pointer("/configuration/endian") {
                Some(endian) if endian == "little" => {}
                None if data_type.element_byte_count == 1 => {}
                _ => return Err(invalid("the `bytes` codec must use little-endian byte order".to_string())),
            }
        }
        if codecs.len() != 1 {
            return Err(invalid("exactly one `bytes` codec is supported".to_string()));
        }
        if document
            .get("storage_transformers")
            .and_then(Value::as_array)
            .is_some_and(|transformers| !transformers.is_empty())
        {
            return Err(invalid("storage transformers are not supported".to_string()));
        }

        Ok(Self {
            node,
            data_type: data_type.data_type,
            element_byte_count: data_type.element_byte_count,
            chunk_entries: chunk_shape.first().copied().unwrap_or(1),
            shape,
            separator,
        })
    }

    /// Returns the number of bytes of the array contents.
    fn byte_count(&self) -> u64 {
        self.shape.iter().product::<usize>() as u64 * self.element_byte_count as u64
    }

    /// Reads `byte_range` of the array contents from the chunks that intersect it.
    fn read(&self, byte_range: Range<u64>) -> Result<Vec<u8>, CheckpointError> {
        if byte_range.start == byte_range.end {
            return Ok(Vec::new());
        }
        if self.shape.is_empty() {
            return self.node.join("c").read_range(byte_range);
        }
        let entry_byte_count = self.byte_count() / self.shape[0] as u64;
        let chunk_byte_count = self.chunk_entries as u64 * entry_byte_count;
        let mut bytes = Vec::with_capacity((byte_range.end - byte_range.start) as usize);
        for chunk in byte_range.start / chunk_byte_count..=(byte_range.end - 1) / chunk_byte_count {
            let chunk_start = chunk * chunk_byte_count;
            let start = byte_range.start.max(chunk_start) - chunk_start;
            let end = byte_range.end.min(chunk_start + chunk_byte_count) - chunk_start;
            let mut key = format!("c{}{chunk}", self.separator);
            for _ in 1..self.shape.len() {
                key.push(self.separator);
                key.push('0');
            }
            bytes.extend(self.node.join(&key).read_range(start..end)?);
        }
        Ok(bytes)
    }
}

/// Returns the metadata document of an array.
fn array_document(name: &str, data_type: &ZarrDataType, shape: &[usize], chunk_shape: &[usize]) -> Value {
    let mut attributes = json!({"ryft": {"name": name}});
    if let Some(tag) = data_type.tag {
        attributes["ryft"]["data_type"] = json!(tag);
    }
    json!({
        "zarr_format": 3,
        "node_type": "array",
        "shape": shape,
        "data_type": data_type.name,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": chunk_shape}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": data_type.fill_value(),
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "attributes": attributes,
    })
}

/// Returns the logical bytes of `array` with every element component in little-endian byte order.
fn little_endian_bytes(array: &Array, data_type: DataType) -> Vec<u8> {
    let mut bytes = array.logical_bytes();
    if cfg!(target_endian = "big") {
        // Complex elements consist of two components that are swapped separately.
        let width = data_type.bit_width().div_ceil(8) / if data_type.is_complex() { 2 } else { 1 };
        if width > 1 {
            bytes.chunks_exact_mut(width).for_each(<[u8]>::reverse);
        }
    }
    bytes
}

/// Returns `document` as pretty JSON.
fn document_bytes(document: &Value) -> Vec<u8> {
    serde_json::to_vec_pretty(document).expect("a JSON value always serializes")
}

/// Reads the JSON document at `path`.
fn read_document(path: &CheckpointPath) -> Result<Value, CheckpointError> {
    serde_json::from_slice(&path.read()?)
        .map_err(|error| CheckpointError::InvalidFile { path: path.to_string(), message: error.to_string() })
}

/// Returns the Zarr node name of the array named `name`, which is `name` itself when it is a valid node name and its
/// percent encoding otherwise, as the [module documentation](self#names) describes.
fn encode_node_name(name: &str) -> Result<String, String> {
    if name.is_empty() {
        return Err("array names must not be empty".to_string());
    }
    let only_periods = name.bytes().all(|byte| byte == b'.');
    let reserved = name == ZARR_METADATA_FILE_NAME;
    let mut encoded = String::with_capacity(name.len());
    for (index, character) in name.char_indices() {
        let escape = match character {
            '%' | '/' => true,
            '.' => only_periods || reserved,
            '_' => index == 0 && name.starts_with("__"),
            _ => false,
        };
        if escape {
            encoded.push_str(&format!("%{:02X}", character as u32));
        } else {
            encoded.push(character);
        }
    }
    Ok(encoded)
}

/// Returns the array name that the Zarr node name `name` encodes, which reverses [`encode_node_name`].
fn decode_node_name(name: &str) -> String {
    let bytes = name.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        let escaped = bytes
            .get(index + 1..index + 3)
            .filter(|_| bytes[index] == b'%')
            .and_then(|digits| std::str::from_utf8(digits).ok())
            .and_then(|digits| u8::from_str_radix(digits, 16).ok());
        match escaped {
            Some(byte) => {
                decoded.push(byte);
                index += 3;
            }
            None => {
                decoded.push(bytes[index]);
                index += 1;
            }
        }
    }
    String::from_utf8_lossy(&decoded).into_owned()
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::thread::ThreadId;

    use pretty_assertions::assert_eq;
    use ryft_core::{ArrayType, Complex, bf16, f4e2m1fn, f8e4m3fn, f16};

    use crate::contexts::{Context, ParameterizedOptions};
    use crate::filesystems::{FileSystem, FileWriter, LocalFileSystem};
    use crate::handlers::{ArrayLeafHandler, CheckpointableLoader, CheckpointableSaver, ParameterizedHandler};
    use crate::loading::load;
    use crate::metadata::ParameterizedMetadata;
    use crate::multiprocessing::MultiprocessingOptions;
    use crate::saving::SaveOptions;
    use crate::training::{Checkpointer, CheckpointerOptions, LatestN};

    use super::*;

    /// Writes `arrays` with `format` into `directory`.
    fn write(format: ZarrFormat, directory: &CheckpointPath, arrays: &[(&str, Array)]) -> Result<(), CheckpointError> {
        let specifications = arrays
            .iter()
            .map(|(name, array)| ArraySpecification::new(*name, array.r#type().into_owned()))
            .collect::<Vec<_>>();
        format.write(directory, &specifications, &mut |index| Ok(Cow::Borrowed(&arrays[index].1)))
    }

    /// Returns the parsed JSON document at `path`.
    fn document(path: &Path) -> Value {
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
    }

    /// Local file system that records the byte ranges that are read and the thread of every whole file write, and that
    /// fails to write the files whose paths end with `failing_write`, if any.
    #[derive(Debug, Default)]
    struct RecordingFileSystem {
        reads: Mutex<Vec<(String, Range<u64>)>>,
        writes: Mutex<Vec<(PathBuf, ThreadId)>>,
        failing_write: Option<&'static str>,
    }

    impl FileSystem for RecordingFileSystem {
        fn supports_atomic_rename(&self) -> bool {
            true
        }

        fn exists(&self, path: &Path) -> Result<bool, CheckpointError> {
            LocalFileSystem.exists(path)
        }

        fn is_directory(&self, path: &Path) -> Result<bool, CheckpointError> {
            LocalFileSystem.is_directory(path)
        }

        fn list(&self, path: &Path) -> Result<Vec<String>, CheckpointError> {
            LocalFileSystem.list(path)
        }

        fn create_directory(&self, path: &Path) -> Result<(), CheckpointError> {
            LocalFileSystem.create_directory(path)
        }

        fn remove(&self, path: &Path) -> Result<(), CheckpointError> {
            LocalFileSystem.remove(path)
        }

        fn rename(&self, from: &Path, to: &Path) -> Result<(), CheckpointError> {
            LocalFileSystem.rename(from, to)
        }

        fn read(&self, path: &Path) -> Result<Vec<u8>, CheckpointError> {
            LocalFileSystem.read(path)
        }

        fn read_range(&self, path: &Path, range: Range<u64>) -> Result<Vec<u8>, CheckpointError> {
            let name = path.components().rev().take(4).collect::<Vec<_>>();
            let name = name.iter().rev().map(|part| part.as_os_str().to_string_lossy()).collect::<Vec<_>>();
            self.reads.lock().unwrap().push((name.join("/"), range.clone()));
            LocalFileSystem.read_range(path, range)
        }

        fn size(&self, path: &Path) -> Result<u64, CheckpointError> {
            LocalFileSystem.size(path)
        }

        fn write(&self, path: &Path, bytes: &[u8]) -> Result<(), CheckpointError> {
            if self.failing_write.is_some_and(|failing_write| path.ends_with(failing_write)) {
                return Err(CheckpointError::Io { path: path.display().to_string(), message: "full".to_string() });
            }
            self.writes.lock().unwrap().push((path.to_path_buf(), thread::current().id()));
            LocalFileSystem.write(path, bytes)
        }

        fn writer(&self, path: &Path) -> Result<Box<dyn FileWriter>, CheckpointError> {
            LocalFileSystem.writer(path)
        }
    }

    #[test]
    fn test_zarr_format_new() {
        assert_eq!(ZarrFormat::new(), ZarrFormat::default());
        assert_eq!(ZarrFormat::new().chunk_byte_size(), None);
        assert_eq!(ZarrFormat::new().with_chunk_byte_size(Some(8)).chunk_byte_size(), Some(8));
        assert_eq!(ZarrFormat::new().write_concurrency(), DEFAULT_WRITE_CONCURRENCY);
        assert_eq!(ZarrFormat::new().with_write_concurrency(4).write_concurrency(), 4);
        assert_eq!(ZarrFormat::new().with_write_concurrency(0).write_concurrency(), 1);
        assert_eq!(ZarrFormat::new().name(), "zarr");
    }

    #[test]
    fn test_zarr_format_write() {
        let temporary = tempfile::tempdir().unwrap();
        let directory = CheckpointPath::from(temporary.path());
        let matrix = Array::matrix(2, 3, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let scalar = Array::scalar(7i64).unwrap();
        assert_eq!(write(ZarrFormat::new(), &directory, &[("$.weights", matrix), ("$.step", scalar)]), Ok(()));

        assert_eq!(
            document(&temporary.path().join("zarr.json")),
            json!({
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {"ryft": {"arrays": ["$.weights", "$.step"]}},
            }),
        );
        assert_eq!(
            document(&temporary.path().join("$.weights").join("zarr.json")),
            json!({
                "zarr_format": 3,
                "node_type": "array",
                "shape": [2, 3],
                "data_type": "float32",
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2, 3]}},
                "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
                "fill_value": 0.0,
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
                "attributes": {"ryft": {"name": "$.weights"}},
            }),
        );
        assert_eq!(
            document(&temporary.path().join("$.step").join("zarr.json")),
            json!({
                "zarr_format": 3,
                "node_type": "array",
                "shape": [],
                "data_type": "int64",
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": []}},
                "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
                "fill_value": 0,
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
                "attributes": {"ryft": {"name": "$.step"}},
            }),
        );

        // One chunk holds a whole array, and a rank-0 array is stored under the key `c`.
        assert_eq!(
            std::fs::read(temporary.path().join("$.weights").join("c").join("0").join("0")).unwrap(),
            [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0].iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>(),
        );
        assert_eq!(std::fs::read(temporary.path().join("$.step").join("c")).unwrap(), 7i64.to_le_bytes().to_vec());
    }

    #[test]
    fn test_zarr_format_write_chunks() {
        // Chunks of 8 bytes hold one row of three `u16` values each, so a final chunk would be padded, and chunks of
        // 13 bytes hold two rows, so the last one of three rows is padded with zeros.
        let temporary = tempfile::tempdir().unwrap();
        let directory = CheckpointPath::from(temporary.path());
        let rows = Array::matrix(3, 3, vec![1u16, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        assert_eq!(
            write(ZarrFormat::new().with_chunk_byte_size(Some(13)), &directory, &[("rows", rows.clone())]),
            Ok(())
        );
        let node = temporary.path().join("rows");
        assert_eq!(
            document(&node.join("zarr.json")).pointer("/chunk_grid/configuration/chunk_shape"),
            Some(&json!([2, 3])),
        );
        let bytes = |values: &[u16]| values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
        assert_eq!(std::fs::read(node.join("c").join("0").join("0")).unwrap(), bytes(&[1, 2, 3, 4, 5, 6]));
        assert_eq!(std::fs::read(node.join("c").join("1").join("0")).unwrap(), bytes(&[7, 8, 9, 0, 0, 0]));
        assert!(!node.join("c").join("2").exists());

        // Reading ignores the padding, and chunks smaller than one row still hold one row.
        let checkpoint = ZarrCheckpoint::open(&directory).unwrap();
        assert_eq!(checkpoint.read("rows"), Ok(rows.clone()));
        let temporary = tempfile::tempdir().unwrap();
        let directory = CheckpointPath::from(temporary.path());
        assert_eq!(
            write(ZarrFormat::new().with_chunk_byte_size(Some(1)), &directory, &[("rows", rows.clone())]),
            Ok(())
        );
        assert!(temporary.path().join("rows").join("c").join("2").join("0").is_file());
        assert_eq!(ZarrCheckpoint::open(&directory).unwrap().read("rows"), Ok(rows));
    }

    #[test]
    fn test_zarr_format_write_concurrently() {
        let mut arrays = (0..12)
            .map(|index| (format!("a{index}"), Array::vector(vec![index as f32; 3]).unwrap()))
            .collect::<Vec<_>>();
        arrays.push(("large".to_string(), Array::vector(vec![7u8; CONCURRENT_WRITE_BYTE_LIMIT + 1]).unwrap()));
        let arrays = arrays.iter().map(|(name, array)| (name.as_str(), array.clone())).collect::<Vec<_>>();

        // Writer threads write every file except the group document, which is written first, and files that exceed
        // the limit, which the calling thread writes so that they never wait in memory.
        let temporary = tempfile::tempdir().unwrap();
        let filesystem = Arc::new(RecordingFileSystem::default());
        let directory = CheckpointPath::new(filesystem.clone(), temporary.path().join("concurrent"));
        assert_eq!(write(ZarrFormat::new().with_write_concurrency(4), &directory, &arrays), Ok(()));
        let writes = filesystem.writes.lock().unwrap().clone();
        assert_eq!(writes.len(), 2 * arrays.len() + 1);
        let root = temporary.path().join("concurrent");
        let mut calling_thread_writes = writes
            .iter()
            .filter(|(_, thread)| *thread == thread::current().id())
            .map(|(path, _)| path.strip_prefix(&root).unwrap().to_path_buf())
            .collect::<Vec<_>>();
        calling_thread_writes.sort();
        assert_eq!(calling_thread_writes, vec![PathBuf::from("large/c/0"), PathBuf::from("zarr.json")]);
        let checkpoint = ZarrCheckpoint::open(&directory).unwrap();
        for (name, array) in &arrays {
            assert_eq!(checkpoint.read(name).as_ref(), Ok(array));
        }

        // With a concurrency of one, the calling thread writes every file, and the files are the same.
        let filesystem = Arc::new(RecordingFileSystem::default());
        let directory = CheckpointPath::new(filesystem.clone(), temporary.path().join("sequential"));
        assert_eq!(write(ZarrFormat::new().with_write_concurrency(1), &directory, &arrays), Ok(()));
        let sequential_writes = filesystem.writes.lock().unwrap().clone();
        assert!(sequential_writes.iter().all(|(_, thread)| *thread == thread::current().id()));
        for (path, _) in &sequential_writes {
            let path = path.strip_prefix(temporary.path().join("sequential")).unwrap();
            assert_eq!(std::fs::read(root.join(path)).unwrap(), std::fs::read(directory.path().join(path)).unwrap());
        }

        // A file that a writer thread fails to write fails the whole write with its error.
        let filesystem = Arc::new(RecordingFileSystem { failing_write: Some("a3/c/0"), ..Default::default() });
        let directory = CheckpointPath::new(filesystem, temporary.path().join("failing"));
        assert_eq!(
            write(ZarrFormat::new().with_write_concurrency(4), &directory, &arrays),
            Err(CheckpointError::Io {
                path: temporary.path().join("failing").join("a3/c/0").display().to_string(),
                message: "full".to_string(),
            }),
        );
    }

    #[test]
    fn test_zarr_format_write_empty() {
        // Arrays without elements write no chunks and have a positive chunk shape.
        let temporary = tempfile::tempdir().unwrap();
        let directory = CheckpointPath::from(temporary.path());
        let empty = Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0, 4]), &[]).unwrap();
        assert_eq!(
            write(ZarrFormat::new().with_chunk_byte_size(Some(8)), &directory, &[("empty", empty.clone())]),
            Ok(())
        );
        let node = temporary.path().join("empty");
        assert_eq!(
            document(&node.join("zarr.json")).pointer("/chunk_grid/configuration/chunk_shape"),
            Some(&json!([1, 4])),
        );
        assert!(!node.join("c").exists());
        let checkpoint = ZarrCheckpoint::open(&directory).unwrap();
        assert_eq!(checkpoint.arrays(), &[ArrayDescriptor::new("empty", DataType::F32, [0, 4], 0..0)]);
        assert_eq!(checkpoint.read("empty"), Ok(empty));
    }

    #[test]
    fn test_zarr_format_write_invalid() {
        let temporary = tempfile::tempdir().unwrap();
        let directory = CheckpointPath::from(temporary.path());
        let scalar = Array::scalar(1u8).unwrap();
        assert_eq!(
            write(ZarrFormat::new(), &directory, &[("", scalar.clone())]),
            Err(CheckpointError::InvalidArray {
                name: String::new(),
                message: "array names must not be empty".to_string()
            }),
        );
        assert_eq!(
            write(ZarrFormat::new(), &directory, &[("a", scalar.clone()), ("a", scalar.clone())]),
            Err(CheckpointError::InvalidArray {
                name: "a".to_string(),
                message: "array name is used more than once, or differs from another array name only in case"
                    .to_string(),
            }),
        );
        assert_eq!(
            write(ZarrFormat::new(), &directory, &[("W", scalar.clone()), ("w", scalar.clone())]),
            Err(CheckpointError::InvalidArray {
                name: "w".to_string(),
                message: "array name is used more than once, or differs from another array name only in case"
                    .to_string(),
            }),
        );
        let narrow =
            Array::from_elements(ArrayType::new_static(DataType::F4E2M1FN, [1]), &[f4e2m1fn::from_f32(1.0).unwrap()])
                .unwrap();
        assert_eq!(
            write(ZarrFormat::new(), &directory, &[("narrow", narrow)]),
            Err(CheckpointError::InvalidArray {
                name: "narrow".to_string(),
                message: "data type `f4e2m1fn` has neither a Zarr name nor a data type tag".to_string(),
            }),
        );
    }

    #[test]
    fn test_encode_node_name() {
        assert_eq!(encode_node_name("$.layers[0].weight"), Ok("$.layers[0].weight".to_string()));
        assert_eq!(encode_node_name("$[\"a/b%\"]"), Ok("$[\"a%2Fb%25\"]".to_string()));
        assert_eq!(encode_node_name("__x"), Ok("%5F_x".to_string()));
        assert_eq!(encode_node_name(".."), Ok("%2E%2E".to_string()));
        assert_eq!(encode_node_name("zarr.json"), Ok("zarr%2Ejson".to_string()));
        assert_eq!(encode_node_name(""), Err("array names must not be empty".to_string()));
        for name in ["$.layers[0].weight", "$[\"a/b%\"]", "__x", "..", "zarr.json", "%zz"] {
            assert_eq!(decode_node_name(&encode_node_name(name).unwrap()), name);
        }
    }

    #[test]
    fn test_zarr_checkpoint_open() {
        let temporary = tempfile::tempdir().unwrap();
        let directory = CheckpointPath::from(temporary.path());
        let first = Array::vector(vec![1u8, 2, 3]).unwrap();
        let second = Array::scalar(4.0f64).unwrap();
        write(ZarrFormat::new(), &directory, &[("b", first), ("a/x", second)]).unwrap();

        // Arrays are listed in the written order with consecutive virtual byte ranges.
        let checkpoint = ZarrCheckpoint::open(&directory).unwrap();
        assert_eq!(checkpoint.directory(), &directory);
        assert_eq!(
            checkpoint.arrays(),
            &[
                ArrayDescriptor::new("b", DataType::U8, [3], 0..3),
                ArrayDescriptor::new("a/x", DataType::F64, [], 3..11)
            ],
        );
        assert_eq!(checkpoint.read("a/x"), Ok(Array::scalar(4.0f64).unwrap()));
        assert_eq!(checkpoint.read_bytes("b", 1..3), Ok(vec![2, 3]));
        assert_eq!(
            checkpoint.read_bytes("b", 2..4),
            Err(CheckpointError::InvalidArray {
                name: "b".to_string(),
                message: "byte range 2..4 is not within the 3 bytes of the array".to_string(),
            }),
        );
        assert_eq!(checkpoint.read("c"), Err(CheckpointError::UnknownArray { name: "c".to_string() }));
    }

    #[test]
    fn test_zarr_checkpoint_open_invalid() {
        let temporary = tempfile::tempdir().unwrap();
        let directory = CheckpointPath::from(temporary.path());
        write(ZarrFormat::new(), &directory, &[("a", Array::vector(vec![1u8, 2]).unwrap())]).unwrap();
        let metadata = temporary.path().join("a").join("zarr.json");
        let original = document(&metadata);

        // A malformed document and a codec that this reader did not write are rejected with the document path.
        std::fs::write(&metadata, "{").unwrap();
        assert!(matches!(
            ZarrCheckpoint::open(&directory),
            Err(CheckpointError::InvalidFile { path, .. }) if path == metadata.display().to_string(),
        ));
        let mut compressed = original.clone();
        compressed["codecs"] = json!([{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "zstd"}]);
        std::fs::write(&metadata, serde_json::to_vec(&compressed).unwrap()).unwrap();
        assert_eq!(
            ZarrCheckpoint::open(&directory).map(|_| ()),
            Err(CheckpointError::InvalidFile {
                path: metadata.display().to_string(),
                message: "unsupported codec `zstd`".to_string(),
            }),
        );

        // A missing chunk fails the read that needs it.
        std::fs::write(&metadata, serde_json::to_vec(&original).unwrap()).unwrap();
        let chunk = temporary.path().join("a").join("c").join("0");
        std::fs::remove_file(&chunk).unwrap();
        let checkpoint = ZarrCheckpoint::open(&directory).unwrap();
        assert!(matches!(
            checkpoint.read("a"),
            Err(CheckpointError::Io { path, .. }) if path == chunk.display().to_string(),
        ));
    }

    #[test]
    fn test_zarr_checkpoint_read_leading() {
        // A file system that records every range read shows that only the needed chunk is read.
        let temporary = tempfile::tempdir().unwrap();
        let filesystem = Arc::new(RecordingFileSystem::default());
        let directory = CheckpointPath::new(filesystem.clone(), temporary.path());
        let stack = Array::from_elements(
            ArrayType::new_static(DataType::F32, [4, 2]),
            &[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        )
        .unwrap();
        write(ZarrFormat::new().with_chunk_byte_size(Some(8)), &directory, &[("stack", stack)]).unwrap();
        let checkpoint = ZarrCheckpoint::open(&directory).unwrap();
        assert_eq!(
            checkpoint.read_leading("stack", 2..3),
            Ok(Array::from_elements(ArrayType::new_static(DataType::F32, [1, 2]), &[4.0f32, 5.0]).unwrap()),
        );
        assert_eq!(filesystem.reads.lock().unwrap().clone(), vec![("stack/c/2/0".to_string(), 0..8)]);

        // A range that spans two chunks reads exactly the needed part of each.
        filesystem.reads.lock().unwrap().clear();
        assert_eq!(checkpoint.read_bytes("stack", 12..20).map(|bytes| bytes.len()), Ok(8));
        assert_eq!(
            filesystem.reads.lock().unwrap().clone(),
            vec![("stack/c/1/0".to_string(), 4..8), ("stack/c/2/0".to_string(), 0..4)],
        );
    }

    #[test]
    fn test_zarr_format_parameterized_handler() {
        // Every data type family round trips, including a narrow floating-point type stored as raw bits, arrays
        // without elements, and rank-0 arrays.
        let narrow = Array::from_elements(
            ArrayType::new_static(DataType::F8E4M3FN, [2]),
            &[f8e4m3fn::from_f32(1.5).unwrap(), f8e4m3fn::from_f32(-2.0).unwrap()],
        )
        .unwrap();
        let state = vec![
            Array::vector(vec![true, false]).unwrap(),
            Array::vector(vec![-1i8, 2]).unwrap(),
            Array::vector(vec![-3i16, 4]).unwrap(),
            Array::vector(vec![-5i32, 6]).unwrap(),
            Array::vector(vec![-7i64, 8]).unwrap(),
            Array::vector(vec![1u8, 2]).unwrap(),
            Array::vector(vec![3u16, 4]).unwrap(),
            Array::vector(vec![5u32, 6]).unwrap(),
            Array::vector(vec![7u64, 8]).unwrap(),
            Array::vector(vec![f16::from_f32(1.5)]).unwrap(),
            Array::vector(vec![bf16::from_f32(-2.5)]).unwrap(),
            Array::matrix(2, 2, vec![1.0f32, 2.0, 3.0, 4.0]).unwrap(),
            Array::vector(vec![1.0f64, -1.0]).unwrap(),
            Array::vector(vec![Complex::new(1.0f32, -2.0)]).unwrap(),
            narrow,
            Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [3, 0]), &[]).unwrap(),
            Array::scalar(9u32).unwrap(),
        ];
        let abstract_state = state.iter().map(|array| array.r#type().into_owned()).collect::<Vec<_>>();
        let temporary = tempfile::tempdir().unwrap();
        let directory = CheckpointPath::from(temporary.path());
        let options = ParameterizedOptions::new();
        let handler = ParameterizedHandler::new(
            &ArrayLeafHandler,
            Arc::new(ZarrFormat::new().with_chunk_byte_size(Some(4))),
            &options,
        );
        assert_eq!(handler.save(&directory, &state, &MultiprocessingOptions::default()), Ok(()));
        assert_eq!(handler.load(&directory, abstract_state), Ok(state));
        assert_eq!(
            ParameterizedMetadata::read(&directory).map(|metadata| metadata.format().to_string()),
            Ok("zarr".to_string()),
        );
        assert_eq!(
            document(&temporary.path().join("$[14]").join("zarr.json")).pointer("/attributes/ryft/data_type"),
            Some(&json!("F8_E4M3")),
        );
        assert_eq!(document(&temporary.path().join("$[14]").join("zarr.json"))["data_type"], json!("r8"));
        assert_eq!(document(&temporary.path().join("$[14]").join("zarr.json"))["fill_value"], json!([0]));
        assert_eq!(document(&temporary.path().join("$[10]").join("zarr.json"))["data_type"], json!("bfloat16"));
        assert_eq!(document(&temporary.path().join("$[13]").join("zarr.json"))["fill_value"], json!([0.0, 0.0]));
        assert_eq!(document(&temporary.path().join("$[0]").join("zarr.json"))["fill_value"], json!(false));
    }

    #[test]
    fn test_zarr_format_checkpointer() {
        let temporary = tempfile::tempdir().unwrap();
        let context = Context::new(ArrayLeafHandler, Arc::new(ZarrFormat::new()));
        let options = CheckpointerOptions::new().with_preservation_policy(LatestN::new(1));
        let mut checkpointer = Checkpointer::new(temporary.path(), context.clone(), options).unwrap();
        for step in 1..=2u8 {
            let state = (Array::vector(vec![step, step]).unwrap(),);
            assert_eq!(checkpointer.save(u64::from(step), &state, &SaveOptions::new()), Ok(true));
        }
        assert_eq!(checkpointer.checkpoints().iter().map(|checkpoint| checkpoint.step()).collect::<Vec<_>>(), vec![2]);
        assert_eq!(
            checkpointer.load(None, (ArrayType::new_static(DataType::U8, [2]),)),
            Ok((Array::vector(vec![2u8, 2]).unwrap(),)),
        );
        assert_eq!(
            load(temporary.path().join("2"), (ArrayType::new_static(DataType::U8, [2]),), &context),
            Ok((Array::vector(vec![2u8, 2]).unwrap(),)),
        );
    }

    #[cfg(feature = "object-store")]
    #[test]
    fn test_zarr_format_object_store() {
        let directory = crate::filesystems::objects::tests::test_object_store_root().join("state");
        let options = ParameterizedOptions::new();
        let handler = ParameterizedHandler::new(
            &ArrayLeafHandler,
            Arc::new(ZarrFormat::new().with_chunk_byte_size(Some(8))),
            &options,
        );
        let state =
            (Array::matrix(3, 2, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(), vec![Array::scalar(1i32).unwrap()]);
        assert_eq!(handler.save(&directory, &state, &MultiprocessingOptions::default()), Ok(()));
        let abstract_state = (ArrayType::new_static(DataType::F32, [3, 2]), vec![ArrayType::scalar(DataType::I32)]);
        assert_eq!(handler.load(&directory, abstract_state), Ok(state));
    }
}
