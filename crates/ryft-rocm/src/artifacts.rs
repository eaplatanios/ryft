//! Immutable pointer-only AMD HSA code objects with checked native ABI and launch metadata.

use std::sync::Arc;

use sha2::{Digest, Sha256};

use crate::Error;

/// Nonzero grid and block dimensions plus additional dynamic LDS requested by the caller.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RocmKernelLaunchDimensions {
    /// Number of thread blocks along each axis.
    grid: [u32; 3],

    /// Threads in each block dimension.
    block: [u32; 3],

    /// Dynamic shared memory added to the native kernel's static LDS allocation.
    dynamic_shared_memory_bytes: u32,
}

impl RocmKernelLaunchDimensions {
    /// Validates nonzero geometry, at most 1024 threads, and HIP's per-axis 32-bit work-item bound.
    pub fn new(grid: [u32; 3], block: [u32; 3], dynamic_shared_memory_bytes: u32) -> Result<Self, Error> {
        let threads = block.iter().try_fold(1_u32, |total, value| total.checked_mul(*value));
        if grid.contains(&0)
            || block.contains(&0)
            || threads.is_none_or(|threads| threads > 1024)
            || grid.iter().zip(block).any(|(grid, block)| grid.checked_mul(block).is_none())
        {
            return Err(Error::invalid_argument("invalid rocm grid or thread-block dimensions"));
        }
        Ok(Self { grid, block, dynamic_shared_memory_bytes })
    }

    /// Returns the number of thread blocks along each axis.
    pub fn grid(self) -> [u32; 3] {
        self.grid
    }

    /// Returns the thread-block dimensions.
    pub fn block(self) -> [u32; 3] {
        self.block
    }

    /// Returns additional dynamic shared memory in bytes.
    pub fn dynamic_shared_memory_bytes(self) -> u32 {
        self.dynamic_shared_memory_bytes
    }
}

/// A bounded AMD HSA code object with one compiler-established pointer-only kernel entry.
///
/// Admission checks ELF ranges, architecture, exported code and descriptor symbols, and AMDGPU metadata. It accepts
/// the pinned code-object v5 representation and metadata version 1.2, ordinary global-pointer arguments, no hidden
/// arguments or dynamic stack, and 64-thread wavefronts on the listed compute architectures. The compiler remains
/// responsible for pointer access semantics and required block dimensions when no exact workgroup size is encoded.
#[derive(Clone, Debug)]
pub struct RocmKernelArtifact {
    /// Immutable native image shared by launch variants.
    image: Arc<[u8]>,

    /// Exported function selected for launch.
    entry_name: String,

    /// Exact base AMD architecture recorded in the ELF header.
    target: String,

    /// Number of ordinary native pointer arguments.
    parameter_count: usize,

    /// Geometry and additional dynamic LDS for each invocation.
    launch_dimensions: RocmKernelLaunchDimensions,

    /// Image and selected symbol identity, independent of per-invocation geometry.
    content_hash: [u8; 32],

    /// ELF architecture and feature requirements.
    flags: u32,

    /// Native workgroup upper bound encoded in the metadata.
    maximum_threads: u32,

    /// Exact native workgroup dimensions when the compiler supplies them.
    required_block: Option<[u32; 3]>,
}

impl RocmKernelArtifact {
    /// Validates the image, exported entry and pointer ABI before it can enter a native module cache.
    pub fn new(
        image: Arc<[u8]>,
        entry_name: impl Into<String>,
        target: impl Into<String>,
        parameter_count: usize,
        launch_dimensions: RocmKernelLaunchDimensions,
    ) -> Result<Self, Error> {
        let entry_name = entry_name.into();
        let target = target.into();
        if image.len() > 64 * 1024 * 1024
            || parameter_count == 0
            || parameter_count > 64
            || !entry_name.as_bytes().first().is_some_and(|byte| byte.is_ascii_alphabetic() || *byte == b'_')
            || !entry_name.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            return Err(Error::invalid_argument("invalid rocm artifact size, entry name or argument count"));
        }
        let elf = Elf { bytes: &image };
        if elf.range(0, 9)? != b"\x7fELF\x02\x01\x01\x40\x03"
            || elf.number(16, 2)? != 3
            || elf.number(18, 2)? != 224
            || elf.number(20, 4)? != 1
            || elf.number(52, 2)? != 64
        {
            return Err(Error::invalid_argument("expected an AMD HSA little-endian ELF64 code object v5"));
        }
        let flags = elf.number(48, 4)? as u32;
        let architecture = match flags & 0xff {
            0x30 => "gfx908",
            0x3f => "gfx90a",
            0x4c => "gfx942",
            _ => return Err(Error::invalid_argument("unsupported rocm ELF architecture")),
        };
        if target != architecture || flags & !0xfff != 0 {
            return Err(Error::invalid_argument("rocm target does not match the ELF architecture or feature flags"));
        }
        let program_offset = elf.usize(32, 8)?;
        let program_count = elf.usize(56, 2)?;
        let section_offset = elf.usize(40, 8)?;
        let section_count = elf.usize(60, 2)?;
        if program_count == 0
            || program_count > 128
            || elf.number(54, 2)? != 56
            || section_count == 0
            || section_count > 4096
            || elf.number(58, 2)? != 64
        {
            return Err(Error::invalid_argument("unsupported rocm ELF table dimensions"));
        }
        elf.range(program_offset, program_count * 56)?;
        elf.range(section_offset, section_count * 64)?;
        for index in 0..program_count {
            let header = program_offset + index * 56;
            let size = elf.usize(header + 32, 8)?;
            if elf.usize(header + 40, 8)? < size {
                return Err(Error::invalid_argument("invalid rocm ELF segment size"));
            }
            elf.range(elf.usize(header + 8, 8)?, size)?;
        }
        let mut function_address = None;
        let mut descriptor = None;
        let descriptor_name = format!("{entry_name}.kd");
        let mut metadata = None;
        for index in 0..section_count {
            let section = section_offset + index * 64;
            let kind = elf.number(section + 4, 4)?;
            let offset = elf.usize(section + 24, 8)?;
            let length = elf.usize(section + 32, 8)?;
            if kind != 8 {
                elf.range(offset, length)?;
            }
            if kind == 2 || kind == 11 {
                let linked = elf.usize(section + 40, 4)?;
                if linked >= section_count || elf.number(section + 56, 8)? != 24 || length % 24 != 0 {
                    return Err(Error::invalid_argument("invalid rocm ELF symbol table"));
                }
                let strings = section_offset + linked * 64;
                if elf.number(strings + 4, 4)? != 3 {
                    return Err(Error::invalid_argument("invalid rocm ELF string table"));
                }
                let names = elf.range(elf.usize(strings + 24, 8)?, elf.usize(strings + 32, 8)?)?;
                for symbol in (offset..offset + length).step_by(24) {
                    let name_offset = elf.usize(symbol, 4)?;
                    let suffix = names
                        .get(name_offset..)
                        .ok_or_else(|| Error::invalid_argument("invalid rocm ELF symbol name"))?;
                    let end = suffix
                        .iter()
                        .position(|byte| *byte == 0)
                        .ok_or_else(|| Error::invalid_argument("unterminated rocm ELF symbol name"))?;
                    let name = &suffix[..end];
                    if name != entry_name.as_bytes() && name != descriptor_name.as_bytes() {
                        continue;
                    }
                    let information = elf.number(symbol + 4, 1)?;
                    let owner = elf.usize(symbol + 6, 2)?;
                    if information >> 4 != 1 || owner == 0 || owner >= section_count {
                        return Err(Error::invalid_argument("rocm entry symbols must be global definitions"));
                    }
                    let owner = section_offset + owner * 64;
                    let address = elf.number(symbol + 8, 8)?;
                    let size = elf.number(symbol + 16, 8)?;
                    let start = elf.number(owner + 16, 8)?;
                    let limit = start
                        .checked_add(elf.number(owner + 32, 8)?)
                        .ok_or_else(|| Error::invalid_argument("rocm ELF symbol range overflow"))?;
                    if elf.number(owner + 4, 4)? == 8
                        || size == 0
                        || address < start
                        || address.checked_add(size).is_none_or(|end| end > limit)
                    {
                        return Err(Error::invalid_argument("rocm entry symbol exceeds its section"));
                    }
                    if name == entry_name.as_bytes() {
                        if information & 15 != 2 || elf.number(owner + 8, 8)? & 4 == 0 {
                            return Err(Error::invalid_argument("invalid rocm entry function symbol"));
                        }
                        if function_address.replace(address).is_some_and(|previous| previous != address) {
                            return Err(Error::invalid_argument("conflicting rocm entry symbols"));
                        }
                    } else {
                        if information & 15 != 1 || size != 64 {
                            return Err(Error::invalid_argument("invalid rocm kernel descriptor symbol"));
                        }
                        let file_offset = elf
                            .number(owner + 24, 8)?
                            .checked_add(address - start)
                            .and_then(|offset| usize::try_from(offset).ok())
                            .ok_or_else(|| Error::invalid_argument("rocm descriptor range overflow"))?;
                        let current = (address, file_offset);
                        if descriptor.replace(current).is_some_and(|previous| previous != current) {
                            return Err(Error::invalid_argument("conflicting rocm descriptor symbols"));
                        }
                    }
                }
            } else if kind == 7 {
                let mut position = offset;
                while position < offset + length {
                    let name_size = elf.usize(position, 4)?;
                    let value_size = elf.usize(position + 4, 4)?;
                    let note_kind = elf.number(position + 8, 4)?;
                    let name = elf.range(position + 12, name_size)?;
                    let value_offset = position
                        .checked_add(12)
                        .and_then(|value| name_size.checked_add(3).and_then(|size| value.checked_add(size & !3)))
                        .ok_or_else(|| Error::invalid_argument("rocm ELF note range overflow"))?;
                    let next = value_offset
                        .checked_add(
                            value_size
                                .checked_add(3)
                                .ok_or_else(|| Error::invalid_argument("rocm ELF note range overflow"))?
                                & !3,
                        )
                        .ok_or_else(|| Error::invalid_argument("rocm ELF note range overflow"))?;
                    if next > offset + length {
                        return Err(Error::invalid_argument("rocm ELF note exceeds its section"));
                    }
                    if name == b"AMDGPU\0" && note_kind == 32 {
                        if metadata.is_some() || value_size > 64 * 1024 {
                            return Err(Error::invalid_argument("invalid rocm metadata count or size"));
                        }
                        let mut decoder =
                            MetadataDecoder { bytes: elf.range(value_offset, value_size)?, position: 0, nodes: 0 };
                        let value = decoder.value(0)?;
                        if decoder.position != decoder.bytes.len() {
                            return Err(Error::invalid_argument("trailing rocm metadata bytes"));
                        }
                        metadata = Some(value);
                    }
                    position = next;
                }
            }
        }
        let (Some(function_address), Some((descriptor_address, descriptor_offset))) = (function_address, descriptor)
        else {
            return Err(Error::invalid_argument("rocm entry function or kernel descriptor is missing"));
        };
        let metadata = metadata.ok_or_else(|| Error::invalid_argument("rocm AMDGPU metadata is missing"))?;
        if metadata.field("amdhsa.version")? != &Metadata::Array(vec![Metadata::Number(1), Metadata::Number(2)]) {
            return Err(Error::invalid_argument("unsupported rocm metadata version"));
        }
        if metadata.field("amdhsa.target")?.text()? != format!("amdgcn-unknown-amdhsa-amdgiz-{target}") {
            return Err(Error::invalid_argument("rocm metadata target does not match the ELF architecture"));
        }
        let kernels = metadata.field("amdhsa.kernels")?.array()?;
        if kernels.len() != 1 {
            return Err(Error::invalid_argument("expected one rocm metadata kernel"));
        }
        let kernel = &kernels[0];
        if kernel.field(".name")?.text()? != entry_name
            || kernel.field(".symbol")?.text()? != descriptor_name
            || kernel.field(".kernarg_segment_size")?.number()? != parameter_count as u64 * 8
            || kernel.field(".kernarg_segment_align")?.number()? != 8
            || kernel.field(".wavefront_size")?.number()? != 64
            || kernel.field(".uses_dynamic_stack")? != &Metadata::Boolean(false)
        {
            return Err(Error::invalid_argument("unsupported rocm kernel argument or execution ABI"));
        }
        // LLVM's backwards-compatible AMDHSA descriptor is the native loader's authority. Check its resource and
        // entry fields against metadata rather than trusting a note that could describe a different descriptor.
        let entry_offset = elf.number(descriptor_offset + 16, 8)? as i64;
        if descriptor_address.checked_add_signed(entry_offset) != Some(function_address)
            || elf.number(descriptor_offset, 4)? != kernel.field(".group_segment_fixed_size")?.number()?
            || elf.number(descriptor_offset + 4, 4)? != kernel.field(".private_segment_fixed_size")?.number()?
            || elf.number(descriptor_offset + 8, 4)? != parameter_count as u64 * 8
            || elf.number(descriptor_offset + 56, 2)? & ((1 << 10) | (1 << 11)) != 0
        {
            return Err(Error::invalid_argument("rocm kernel descriptor contradicts its metadata"));
        }
        let arguments = kernel.field(".args")?.array()?;
        if arguments.len() != parameter_count {
            return Err(Error::invalid_argument("unsupported rocm hidden or extra arguments"));
        }
        for (index, argument) in arguments.iter().enumerate() {
            if argument.field(".offset")?.number()? != index as u64 * 8
                || argument.field(".size")?.number()? != 8
                || argument.field(".address_space")?.text()? != "global"
                || argument.field(".value_kind")?.text()? != "global_buffer"
            {
                return Err(Error::invalid_argument("unsupported rocm pointer argument layout"));
            }
        }
        let maximum_threads = u32::try_from(kernel.field(".max_flat_workgroup_size")?.number()?)
            .map_err(|_| Error::invalid_argument("invalid rocm native workgroup limit"))?;
        if maximum_threads == 0 || maximum_threads > 1024 {
            return Err(Error::invalid_argument("invalid rocm native workgroup limit"));
        }
        let required_block = match kernel.optional_field(".reqd_workgroup_size")? {
            Some(value) => {
                let values = value.array()?;
                if values.len() != 3 {
                    return Err(Error::invalid_argument("invalid rocm required workgroup size"));
                }
                Some([
                    u32::try_from(values[0].number()?)
                        .map_err(|_| Error::invalid_argument("invalid rocm required workgroup size"))?,
                    u32::try_from(values[1].number()?)
                        .map_err(|_| Error::invalid_argument("invalid rocm required workgroup size"))?,
                    u32::try_from(values[2].number()?)
                        .map_err(|_| Error::invalid_argument("invalid rocm required workgroup size"))?,
                ])
            }
            None => None,
        };
        let mut hash = Sha256::new();
        hash.update(&image);
        hash.update(entry_name.as_bytes());
        let artifact = Self {
            image,
            entry_name,
            target,
            parameter_count,
            launch_dimensions,
            content_hash: hash.finalize().into(),
            flags,
            maximum_threads,
            required_block,
        };
        artifact.validate_dimensions(launch_dimensions)?;
        Ok(artifact)
    }

    /// Returns the immutable HSACO image.
    pub fn image(&self) -> &[u8] {
        &self.image
    }

    /// Returns the selected exported function name.
    pub fn entry_name(&self) -> &str {
        &self.entry_name
    }

    /// Returns the ELF's base AMD architecture.
    pub fn target(&self) -> &str {
        &self.target
    }

    /// Returns the number of compiler-established pointer arguments.
    pub fn parameter_count(&self) -> usize {
        self.parameter_count
    }

    /// Returns invocation geometry and dynamic shared memory.
    pub fn launch_dimensions(&self) -> RocmKernelLaunchDimensions {
        self.launch_dimensions
    }

    /// Returns the image-and-symbol digest used for native module reuse.
    pub fn content_hash(&self) -> [u8; 32] {
        self.content_hash
    }

    /// Changes invocation resources while sharing the image and its cached digest.
    pub fn with_launch_dimensions(&self, dimensions: RocmKernelLaunchDimensions) -> Result<Self, Error> {
        self.validate_dimensions(dimensions)?;
        Ok(Self { launch_dimensions: dimensions, ..self.clone() })
    }

    /// Validates actual device architecture and ELF feature requirements before cache insertion.
    pub(crate) fn validate_device(&self, architecture: &str) -> Result<(), Error> {
        let mut fields = architecture.split(':');
        if fields.next() != Some(&self.target) {
            return Err(Error::invalid_argument("rocm device architecture does not match the artifact"));
        }
        let features: Vec<&str> = fields.collect();
        for (name, shift) in [("xnack", 8), ("sramecc", 10)] {
            let value = (self.flags >> shift) & 3;
            if value >= 2 {
                let expected = format!("{name}{}", if value == 2 { '-' } else { '+' });
                if !features.iter().any(|feature| *feature == expected) {
                    return Err(Error::invalid_argument(format!("rocm device does not satisfy `{expected}`")));
                }
            }
        }
        Ok(())
    }

    /// Checks the geometry against compiler-recorded workgroup constraints.
    fn validate_dimensions(&self, dimensions: RocmKernelLaunchDimensions) -> Result<(), Error> {
        if dimensions.block().into_iter().product::<u32>() > self.maximum_threads
            || self.required_block.is_some_and(|required| required != dimensions.block())
        {
            return Err(Error::invalid_argument("rocm launch does not satisfy native workgroup requirements"));
        }
        Ok(())
    }
}

/// Checked integer and range access into one ELF image.
struct Elf<'a> {
    /// Complete immutable file contents.
    bytes: &'a [u8],
}

impl<'a> Elf<'a> {
    /// Returns a bounded file range with overflow checked before indexing.
    fn range(&self, offset: usize, length: usize) -> Result<&'a [u8], Error> {
        self.bytes
            .get(offset..offset.checked_add(length).ok_or_else(|| Error::invalid_argument("rocm ELF range overflow"))?)
            .ok_or_else(|| Error::invalid_argument("truncated rocm ELF range"))
    }

    /// Decodes one little-endian integer field of at most eight bytes.
    fn number(&self, offset: usize, length: usize) -> Result<u64, Error> {
        let mut result = [0; 8];
        result[..length].copy_from_slice(self.range(offset, length)?);
        Ok(u64::from_le_bytes(result))
    }

    /// Converts an ELF file offset or length to the host's checked indexing type.
    fn usize(&self, offset: usize, length: usize) -> Result<usize, Error> {
        usize::try_from(self.number(offset, length)?)
            .map_err(|_| Error::invalid_argument("rocm ELF offset is not representable"))
    }
}

/// Bounded MessagePack values used by AMDGPU metadata; arbitrary compiler IR is never accepted here.
#[derive(Debug, PartialEq)]
enum Metadata<'a> {
    Number(u64),
    Text(&'a str),
    Boolean(bool),
    Array(Vec<Self>),
    Map(Vec<(&'a str, Self)>),
}

impl<'a> Metadata<'a> {
    /// Looks up a mandatory named metadata field.
    fn field(&self, name: &str) -> Result<&Self, Error> {
        self.optional_field(name)?
            .ok_or_else(|| Error::invalid_argument(format!("missing rocm metadata field `{name}`")))
    }

    /// Looks up an optional named metadata field.
    fn optional_field(&self, name: &str) -> Result<Option<&Self>, Error> {
        match self {
            Self::Map(fields) => Ok(fields.iter().find(|(key, _)| *key == name).map(|(_, value)| value)),
            _ => Err(Error::invalid_argument("expected a rocm metadata map")),
        }
    }

    /// Extracts an unsigned numeric metadata value.
    fn number(&self) -> Result<u64, Error> {
        match self {
            Self::Number(value) => Ok(*value),
            _ => Err(Error::invalid_argument("expected a rocm metadata integer")),
        }
    }

    /// Extracts a borrowed metadata string.
    fn text(&self) -> Result<&'a str, Error> {
        match self {
            Self::Text(value) => Ok(value),
            _ => Err(Error::invalid_argument("expected a rocm metadata string")),
        }
    }

    /// Extracts a metadata sequence.
    fn array(&self) -> Result<&[Self], Error> {
        match self {
            Self::Array(value) => Ok(value),
            _ => Err(Error::invalid_argument("expected a rocm metadata array")),
        }
    }
}

/// Allocation- and depth-bounded decoder for the pinned AMDGPU MessagePack surface.
struct MetadataDecoder<'a> {
    /// Complete bounded note payload.
    bytes: &'a [u8],

    /// Next byte to consume.
    position: usize,

    /// Total values consumed across nested containers.
    nodes: usize,
}

impl<'a> MetadataDecoder<'a> {
    /// Reads an exact number of bytes without integer overflow or allocation.
    fn bytes(&mut self, length: usize) -> Result<&'a [u8], Error> {
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| Error::invalid_argument("rocm metadata length overflow"))?;
        let bytes = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| Error::invalid_argument("truncated rocm metadata"))?;
        self.position = end;
        Ok(bytes)
    }

    /// Decodes one big-endian unsigned MessagePack integer.
    fn number(&mut self, length: usize) -> Result<u64, Error> {
        let mut value = [0; 8];
        value[8 - length..].copy_from_slice(self.bytes(length)?);
        Ok(u64::from_be_bytes(value))
    }

    /// Decodes one value under fixed nesting, node and container limits.
    fn value(&mut self, depth: usize) -> Result<Metadata<'a>, Error> {
        self.nodes += 1;
        if depth > 16 || self.nodes > 4096 {
            return Err(Error::invalid_argument("rocm metadata exceeds structural limits"));
        }
        let tag = self.bytes(1)?[0];
        let (kind, length) = match tag {
            0..=0x7f => return Ok(Metadata::Number(u64::from(tag))),
            0xc2 | 0xc3 => return Ok(Metadata::Boolean(tag == 0xc3)),
            0xcc..=0xcf => return Ok(Metadata::Number(self.number(1 << (tag - 0xcc))?)),
            0xa0..=0xbf => (0, usize::from(tag & 31)),
            0xd9..=0xdb => (0, self.number(1 << (tag - 0xd9))? as usize),
            0x90..=0x9f => (1, usize::from(tag & 15)),
            0xdc | 0xdd => (1, self.number(if tag == 0xdc { 2 } else { 4 })? as usize),
            0x80..=0x8f => (2, usize::from(tag & 15)),
            0xde | 0xdf => (2, self.number(if tag == 0xde { 2 } else { 4 })? as usize),
            _ => return Err(Error::invalid_argument("unsupported rocm metadata encoding")),
        };
        if kind == 0 {
            return Ok(Metadata::Text(
                std::str::from_utf8(self.bytes(length)?)
                    .map_err(|_| Error::invalid_argument("rocm metadata is not UTF-8"))?,
            ));
        }
        if length > 128 || length > self.bytes.len() - self.position {
            return Err(Error::invalid_argument("rocm metadata container exceeds limits"));
        }
        if kind == 1 {
            return (0..length).map(|_| self.value(depth + 1)).collect::<Result<Vec<_>, _>>().map(Metadata::Array);
        }
        let mut fields = Vec::with_capacity(length);
        for _ in 0..length {
            let key = self.value(depth + 1)?.text()?;
            if fields.iter().any(|(name, _)| *name == key) {
                return Err(Error::invalid_argument("duplicate rocm metadata key"));
            }
            fields.push((key, self.value(depth + 1)?));
        }
        Ok(Metadata::Map(fields))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    /// Constructs the compiler-produced vector artifact using its established native launch ABI.
    fn vector(image: Arc<[u8]>, entry: &str, target: &str, arguments: usize) -> Result<RocmKernelArtifact, Error> {
        RocmKernelArtifact::new(
            image,
            entry,
            target,
            arguments,
            RocmKernelLaunchDimensions::new([1; 3], [256, 1, 1], 0).unwrap(),
        )
    }

    #[test]
    fn test_rocm_kernel_launch_dimensions_new() {
        assert_eq!(
            RocmKernelLaunchDimensions::new([2, 3, 1], [256, 1, 1], 4096),
            Ok(RocmKernelLaunchDimensions { grid: [2, 3, 1], block: [256, 1, 1], dynamic_shared_memory_bytes: 4096 }),
        );
        for (grid, block) in [
            ([0, 1, 1], [1; 3]),
            ([1; 3], [0, 1, 1]),
            ([1; 3], [1025, 1, 1]),
            ([u32::MAX, 1, 1], [2, 1, 1]),
            ([1; 3], [u32::MAX; 3]),
        ] {
            assert_eq!(
                RocmKernelLaunchDimensions::new(grid, block, 0),
                Err(Error::invalid_argument("invalid rocm grid or thread-block dimensions"))
            );
        }
    }

    #[test]
    fn test_rocm_kernel_launch_dimensions_grid() {
        let dimensions = RocmKernelLaunchDimensions::new([2, 3, 1], [256, 1, 1], 4096).unwrap();
        assert_eq!(dimensions.grid(), [2, 3, 1]);
    }

    #[test]
    fn test_rocm_kernel_launch_dimensions_block() {
        let dimensions = RocmKernelLaunchDimensions::new([2, 3, 1], [256, 1, 1], 4096).unwrap();
        assert_eq!(dimensions.block(), [256, 1, 1]);
    }

    #[test]
    fn test_rocm_kernel_launch_dimensions_dynamic_shared_memory_bytes() {
        let dimensions = RocmKernelLaunchDimensions::new([2, 3, 1], [256, 1, 1], 4096).unwrap();
        assert_eq!(dimensions.dynamic_shared_memory_bytes(), 4096);
    }

    #[test]
    fn test_rocm_kernel_artifact_new() {
        let image: Arc<[u8]> = Arc::from(include_bytes!("fixtures/vector-gfx942.hsaco").as_slice());
        vector(image, "ryft_kernel", "gfx942", 3).unwrap();
        let dot = RocmKernelArtifact::new(
            Arc::from(include_bytes!("fixtures/dot-gfx942.hsaco").as_slice()),
            "ryft_kernel",
            "gfx942",
            3,
            RocmKernelLaunchDimensions::new([1; 3], [256, 1, 1], 4096).unwrap(),
        )
        .unwrap();
        assert_eq!(dot.launch_dimensions().dynamic_shared_memory_bytes(), 4096);
    }

    #[test]
    fn test_rocm_kernel_artifact_new_invalid_header() {
        let mut image = include_bytes!("fixtures/vector-gfx942.hsaco").to_vec();
        image[0] = 0;
        assert!(matches!(
            vector(image.into(), "ryft_kernel", "gfx942", 3),
            Err(Error::InvalidArgument { message }) if {
                message == "expected an AMD HSA little-endian ELF64 code object v5"
            },
        ));
        let mut image = include_bytes!("fixtures/vector-gfx942.hsaco").to_vec();
        image[40..48].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(
            vector(image.into(), "ryft_kernel", "gfx942", 3),
            Err(Error::InvalidArgument { message }) if message == "rocm ELF range overflow",
        ));
        assert!(matches!(
            vector(Arc::from([]), "ryft_kernel", "gfx942", 3),
            Err(Error::InvalidArgument { message }) if message == "truncated rocm ELF range",
        ));
    }

    #[test]
    fn test_rocm_kernel_artifact_new_invalid_entry_and_arguments() {
        let image: Arc<[u8]> = Arc::from(include_bytes!("fixtures/vector-gfx942.hsaco").as_slice());
        assert!(matches!(
            vector(Arc::clone(&image), "missing", "gfx942", 3),
            Err(Error::InvalidArgument { message }) if message == "rocm entry function or kernel descriptor is missing",
        ));
        assert!(matches!(
            vector(Arc::clone(&image), "ryft_kernel", "gfx942", 2),
            Err(Error::InvalidArgument { message }) if message == "unsupported rocm kernel argument or execution ABI",
        ));
        assert!(matches!(
            vector(Arc::clone(&image), "ryft_kernel", "gfx908", 3),
            Err(Error::InvalidArgument { message }) if {
                message == "rocm target does not match the ELF architecture or feature flags"
            },
        ));
        assert!(matches!(
            vector(image, "invalid\0name", "gfx942", 3),
            Err(Error::InvalidArgument { message }) if {
                message == "invalid rocm artifact size, entry name or argument count"
            },
        ));
    }

    #[test]
    fn test_rocm_kernel_artifact_new_descriptor_mismatch() {
        let image = include_bytes!("fixtures/vector-gfx942.hsaco");
        // Locate the descriptor through the exported symbol, independently of its metadata note.
        let elf = Elf { bytes: image };
        let sections = elf.usize(40, 8).unwrap();
        let count = elf.usize(60, 2).unwrap();
        let mut descriptor = None;
        for index in 0..count {
            let section = sections + index * 64;
            if elf.number(section + 4, 4).unwrap() != 11 {
                continue;
            }
            let strings = sections + elf.usize(section + 40, 4).unwrap() * 64;
            let names = elf.range(elf.usize(strings + 24, 8).unwrap(), elf.usize(strings + 32, 8).unwrap()).unwrap();
            let offset = elf.usize(section + 24, 8).unwrap();
            let length = elf.usize(section + 32, 8).unwrap();
            for symbol in (offset..offset + length).step_by(24) {
                let name = elf.usize(symbol, 4).unwrap();
                if names[name..].starts_with(b"ryft_kernel.kd\0") {
                    let owner = sections + elf.usize(symbol + 6, 2).unwrap() * 64;
                    descriptor = Some(
                        elf.usize(owner + 24, 8).unwrap() + elf.usize(symbol + 8, 8).unwrap()
                            - elf.usize(owner + 16, 8).unwrap(),
                    );
                }
            }
        }
        let descriptor = descriptor.unwrap();
        for field in [0, 4, 8, 16, 57] {
            let mut changed = image.to_vec();
            changed[descriptor + field] ^= if field == 57 { 8 } else { 1 };
            assert!(matches!(
                vector(changed.into(), "ryft_kernel", "gfx942", 3),
                Err(Error::InvalidArgument { message }) if message == "rocm kernel descriptor contradicts its metadata"
            ));
        }
    }

    #[test]
    fn test_rocm_kernel_artifact_new_dynamic_stack() {
        let mut image = include_bytes!("fixtures/vector-gfx942.hsaco").to_vec();
        let marker = b".uses_dynamic_stack\xc2";
        let position = image.windows(marker.len()).position(|window| window == marker).unwrap() + marker.len() - 1;
        image[position] = 0xc3;
        assert!(matches!(
            vector(image.into(), "ryft_kernel", "gfx942", 3),
            Err(Error::InvalidArgument { message }) if message == "unsupported rocm kernel argument or execution ABI",
        ));
    }

    #[test]
    fn test_rocm_kernel_artifact_image() {
        let artifact =
            vector(Arc::from(include_bytes!("fixtures/vector-gfx942.hsaco").as_slice()), "ryft_kernel", "gfx942", 3)
                .unwrap();
        assert_eq!(artifact.image(), include_bytes!("fixtures/vector-gfx942.hsaco").as_slice());
    }

    #[test]
    fn test_rocm_kernel_artifact_entry_name() {
        let artifact =
            vector(Arc::from(include_bytes!("fixtures/vector-gfx942.hsaco").as_slice()), "ryft_kernel", "gfx942", 3)
                .unwrap();
        assert_eq!(artifact.entry_name(), "ryft_kernel");
    }

    #[test]
    fn test_rocm_kernel_artifact_target() {
        let artifact =
            vector(Arc::from(include_bytes!("fixtures/vector-gfx942.hsaco").as_slice()), "ryft_kernel", "gfx942", 3)
                .unwrap();
        assert_eq!(artifact.target(), "gfx942");
    }

    #[test]
    fn test_rocm_kernel_artifact_parameter_count() {
        let artifact =
            vector(Arc::from(include_bytes!("fixtures/vector-gfx942.hsaco").as_slice()), "ryft_kernel", "gfx942", 3)
                .unwrap();
        assert_eq!(artifact.parameter_count(), 3);
    }

    #[test]
    fn test_rocm_kernel_artifact_launch_dimensions() {
        let artifact =
            vector(Arc::from(include_bytes!("fixtures/vector-gfx942.hsaco").as_slice()), "ryft_kernel", "gfx942", 3)
                .unwrap();
        assert_eq!(artifact.launch_dimensions(), RocmKernelLaunchDimensions::new([1; 3], [256, 1, 1], 0).unwrap());
    }

    #[test]
    fn test_rocm_kernel_artifact_content_hash() {
        let image: Arc<[u8]> = Arc::from(include_bytes!("fixtures/vector-gfx942.hsaco").as_slice());
        let first = vector(Arc::clone(&image), "ryft_kernel", "gfx942", 3).unwrap();
        let second = vector(image, "ryft_kernel", "gfx942", 3).unwrap();
        assert_eq!(first.content_hash(), second.content_hash());
        let dot = vector(Arc::from(include_bytes!("fixtures/dot-gfx942.hsaco").as_slice()), "ryft_kernel", "gfx942", 3)
            .unwrap();
        assert_ne!(first.content_hash(), dot.content_hash());
    }

    #[test]
    fn test_rocm_kernel_artifact_with_launch_dimensions() {
        let artifact =
            vector(Arc::from(include_bytes!("fixtures/vector-gfx942.hsaco").as_slice()), "ryft_kernel", "gfx942", 3)
                .unwrap();
        let changed = artifact
            .with_launch_dimensions(RocmKernelLaunchDimensions::new([4, 1, 1], [256, 1, 1], 0).unwrap())
            .unwrap();
        assert_eq!(changed.launch_dimensions().grid(), [4, 1, 1]);
        assert_eq!(changed.content_hash(), artifact.content_hash());
        assert!(Arc::ptr_eq(&changed.image, &artifact.image));
        assert!(matches!(
            artifact.with_launch_dimensions(
                RocmKernelLaunchDimensions::new([1; 3], [512, 1, 1], 0).unwrap(),
            ),
            Err(Error::InvalidArgument { message }) if {
                message == "rocm launch does not satisfy native workgroup requirements"
            },
        ));
    }

    #[test]
    fn test_rocm_kernel_artifact_validate_device() {
        let mut image = include_bytes!("fixtures/vector-gfx942.hsaco").to_vec();
        let artifact = vector(Arc::from(image.clone()), "ryft_kernel", "gfx942", 3).unwrap();
        assert_eq!(artifact.validate_device("gfx942:sramecc+:xnack-"), Ok(()));
        assert_eq!(
            artifact.validate_device("gfx908"),
            Err(Error::invalid_argument("rocm device architecture does not match the artifact"))
        );
        image[48..52].copy_from_slice(&0x94c_u32.to_le_bytes());
        let artifact = vector(image.into(), "ryft_kernel", "gfx942", 3).unwrap();
        assert_eq!(artifact.validate_device("gfx942:sramecc-:xnack-"), Ok(()));
        assert_eq!(
            artifact.validate_device("gfx942:sramecc+:xnack-"),
            Err(Error::invalid_argument("rocm device does not satisfy `sramecc-`"))
        );
    }

    #[test]
    fn test_metadata_decoder_value() {
        let mut decoder = MetadataDecoder { bytes: &[0x81, 0xa1, b'a', 0x92, 1, 0xc2], position: 0, nodes: 0 };
        assert_eq!(
            decoder.value(0),
            Ok(Metadata::Map(vec![("a", Metadata::Array(vec![Metadata::Number(1), Metadata::Boolean(false)]))]))
        );
        let mut decoder = MetadataDecoder { bytes: &[0x82, 0xa1, b'a', 1, 0xa1, b'a', 2], position: 0, nodes: 0 };
        assert_eq!(decoder.value(0), Err(Error::invalid_argument("duplicate rocm metadata key")));
        let mut decoder = MetadataDecoder { bytes: &[0xdd, 255, 255, 255, 255], position: 0, nodes: 0 };
        assert_eq!(decoder.value(0), Err(Error::invalid_argument("rocm metadata container exceeds limits")));
        let mut decoder = MetadataDecoder { bytes: &[0x91; 18], position: 0, nodes: 0 };
        assert_eq!(decoder.value(0), Err(Error::invalid_argument("rocm metadata exceeds structural limits")));
    }
}
