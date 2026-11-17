//! Contains checked, format independent serialization of portable kernel source.
//!
//! This module turns a [`KernelDefinition`] into a versioned transport document and back, so that a kernel can be
//! stored, cached, or sent to another process and reconstructed there exactly. [`KernelDefinition::to_source`]
//! encodes a definition as an opaque [`KernelSource`] document, and [`KernelDefinition::from_source`] validates and
//! reconstructs one. Both report a typed [`KernelSerializationError`]. The [`Serialize`] and [`Deserialize`]
//! implementations of [`KernelDefinition`] are thin wrappers over these functions that convert the error into a
//! `serde` error message, so a [`KernelSource`] can be written in any `serde` format and decoded separately when the
//! caller needs the typed error.
//!
//! # Document Model
//!
//! A document consists of private wire records that describe the kernel call metadata (i.e., the grid, the
//! parameters with their access intents and block mapping programs, the coordinate types, and the prefetched scalar
//! types) and the exact body program. The records carry no execution behavior. They only reconstruct ordinary
//! programs through the existing canonical constructors and are never interpreted. The document follows a few
//! structural rules.
//!
//!   - Each nominal [`DimensionVariable`] is named exactly once, with its bounds, in a canonical identity table that
//!     the call metadata, the block mappings, and every region of the body share. Dimensions refer to that table by
//!     index, so a symbolic identity is never converted to diagnostic text, and decoding reconstructs exactly one
//!     fresh nominal variable for each identity instead of manufacturing a second one. A symbolic dimension keeps its
//!     identity even when its bounds admit a single extent.
//!   - Each program is a flat region arena with an entry index. Instructions refer to atoms and attached regions by
//!     index, and a region edge may only reference an earlier arena entry, so shared region edges are preserved
//!     without expanding the region graph into a tree.
//!   - Literals are stored as their exact logical bytes together with their array types, which preserves the data
//!     bits, including signed zeros and NaN payloads. Reference storage is never serialized.
//!   - Diagnostic provenance is a flat arena of nodes in child before parent order, whose edges only reference
//!     earlier nodes. Decoding rebuilds it iteratively and checks its nesting depth.
//!   - Every reference access of an instruction records its canonical transform path. The encoder derives these
//!     descriptors from [`validated_reference_access_descriptors`], and the decoder reapplies them, cross-checks that
//!     they cover exactly the access inputs of the operation, and validates them again. Folded reference paths retain
//!     their canonical trailing bindings in the ordered input list of the instruction.
//!
//! # Encodable Operations
//!
//! The schema covers local array types, exact constants, elementwise arithmetic, generalized dot, shape manipulation,
//! bounded control flow, dimension arithmetic, canonical reference updates, and portable kernel memory operations.
//!
//!   - The array subset is zero, zero like, one, one like, constant, iota, fill like, iota like, minimum, maximum,
//!     clamp, sign, negate, absolute value, add, subtract, saturating subtract, multiply, divide, remainder, power,
//!     sqrt, rsqrt, sin, cos, tan, tanh, atan2, exp, expm1, log, ln1p, log add exp, logistic, erf, floor, ceil,
//!     round, Boolean not, and, or, and xor, complex, real, imaginary, conjugate, reduce, argmax, argmin, cumulative,
//!     sort, dot, compare, select, transpose, reverse, reshape, broadcast, broadcast like, pad, concatenate, slice,
//!     update slice, dynamic slice, dynamic update slice, convert element type, reduce precision, transfer to
//!     memory, condition, while, stop gradient, tag, and cast. Every result accuracy and dot precision is encoded
//!     exactly, and the default ones keep their compact records.
//!   - The array IR forms of zero, one, and iota with explicit runtime extents, and of reshape, broadcast, pad,
//!     concatenate, and cast are supported. Reshape, pad, and concatenate are reconstructed from the input types of
//!     their instructions, so they are supported only when that reconstruction preserves their cached metadata.
//!   - Dimension arithmetic supports constants, add, subtract, saturating subtract, multiply, floor division,
//!     remainder, maximum, minimum, and power when canonical reconstruction preserves its complete cached metadata.
//!     Dimension sizes, dimension comparisons, scalar conversions in both directions, conditions, and while loops
//!     with optional iteration bounds are also supported.
//!   - Reference allocation, read, write, swap, freeze, add update, and atomic add update are supported, as are the
//!     kernel memory operations scratch, tile load, masked load, masked store, masked swap, window mask, async copy,
//!     and wait.
//!
//! Distributed metadata (i.e., array shardings, unreduced and reduced axes, and requested reduce, dot, reshape, and
//! broadcast output shardings), dynamically shaped arrays with explicit layouts, nested kernel calls, kernel
//! extensions, and all other operation payloads report explicit eligibility errors as
//! [`KernelSerializationError::Unsupported`] naming the rejected feature. A valid in-memory definition that exceeds
//! the record or literal budgets below is reported the same way, because it is valid but cannot be transported.
//!
//! # Semantic Keys
//!
//! [`KernelDefinition::semantic_key`] is derived from the same records. The encoder runs in a key mode that projects
//! out what is not semantic (i.e., diagnostic provenance, which becomes unknown, nominal dimension names, which become
//! empty, and grid axis names, which become absent) and serializes the resulting document as compact JSON. The records
//! contain no maps and no floating-point fields, so that text is a deterministic function of the document, with struct
//! fields in declaration order. Key mode is a superset of transport mode. It ignores the record and literal budgets,
//! and it encodes everything that transport rejects as unsupported through key-only records, which only serialize.
//! [`KernelDefinition::from_source`] therefore never accepts them. Nested kernel calls, kernel extensions, custom
//! calls, attached composites, and refined dimension arithmetic have exact structural key-only records, and named
//! composites are keyed by their exact composite keys. The retained rule state of named composites, the remaining
//! operations, and distributed type metadata are transitionally keyed by their [`Debug`](std::fmt::Debug) renderings,
//! after renaming the nominal identities of the document to canonical names. Each such fallback is marked with a
//! `TODO` that names what the wire form must cover to replace it.
//!
//! # Untrusted Input
//!
//! Decoding treats a document as untrusted input and checks it in layers before any canonical object exists.
//!
//!   - Every sequence is a bounded collection that ignores untrusted size hints for allocation and rejects sequences
//!     with more than one million records. Literal byte sequences are bounded by the literal budget instead.
//!   - All literal storage of a document together may consume at most 64 MiB after applying layouts, and a literal
//!     must contain exactly the number of logical bytes that its type requires.
//!   - Every identity, atom, region, and provenance index is checked before it is used, and provenance nesting is
//!     limited to 128 levels.
//!   - Every dimension identity, type, literal, and operation is reconstructed through its canonical constructor.
//!     Sealing each program then revalidates the region DAG and its interfaces, and
//!     [`KernelDefinition::new`] revalidates the kernel boundary. Executable initialization and adapter admission
//!     remain separate requirements, just as for an explicitly built definition.
//!
//! # Schema Version
//!
//! Every document records its schema version, which is independent of the encoding version of
//! [`KernelDefinition::semantic_key`], although every key contains the schema version of the records it was derived
//! from. Deserializing a [`KernelSource`] decodes only the structure of the document,
//! and [`KernelDefinition::from_source`] rejects any version other than the current one with
//! [`KernelSerializationError::Schema`]. Any change to the produced records requires a new schema version.

use std::cell::Cell;
use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;
use std::num::NonZeroUsize;

use ryft_core::{
    AbsOperation, Accuracy, AddOperation, AndOperation, ArgMaxOperation, ArgMinOperation, ArithmeticDimensionOperation,
    Array, ArrayAddressing, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayReferenceTransform,
    ArrayReferenceTransformIndex, ArraySliceAxis, ArrayType, Atan2Operation, Atom, AtomId, BroadcastLikeOperation,
    BroadcastOperation, COMPOSITE_OPERATION_NAME, COMPOSITE_TRANSPOSE_OPERATION_NAME, CastOperation, CeilOperation,
    ClampOperation, CompareOperation, ComparisonDirection, ComplexOperation, CompositeAttribute, CompositeJvpRule,
    CompositeOperation, ConcatenateOperation, ConditionOperation, ConjugateOperation, ConstantOperation,
    ConvertElementTypeOperation, CosOperation, CumulativeKind, CumulativeOperation, CustomCallArrayAttribute,
    CustomCallAttribute, CustomCallBatching, CustomCallOperation, CustomCallUnrolling, CustomRuleSource, DataType,
    DifferentiableType, Dimension, DimensionAddOperation, DimensionBounds, DimensionDivOperation, DimensionError,
    DimensionFromScalarOperation, DimensionMaxOperation, DimensionMinOperation, DimensionMulOperation,
    DimensionOperation, DimensionPowOperation, DimensionRemOperation, DimensionSizeOperation, DimensionSubOperation,
    DimensionToScalarOperation, DimensionType, DimensionValue, DimensionVariable, DivOperation, DotAlgorithm,
    DotAlgorithmPreset, DotDimensionNumbers, DotInputPrecision, DotOperation, DotPrecision, DynamicBroadcastOperation,
    DynamicReshapeOperation, DynamicSliceOperation, DynamicUpdateSliceOperation, EagerContext, EffectClass,
    ErfOperation, ExpM1Operation, ExpOperation, FillLikeOperation, FlatProgram, FloorOperation, ImaginaryOperation,
    Instruction, IotaLikeOperation, IotaOperation, Layout, Ln1pOperation, LogAddExpOperation, LogOperation,
    LogisticOperation, MAX_DIMENSION_EXTENT, MaxOperation, Memory, MinOperation, MulOperation, NegOperation,
    NotOperation, OneLikeOperation, OneOperation, Operation, OrOperation, PadOperation, Parameter, Placeholder,
    PowOperation, Program, ProgramError, Provenance, ProvenanceScope, REMATERIALIZE_OPERATION_NAME, RealOperation,
    ReducePrecisionOperation, ReduceOperation, ReductionKind, ReferenceAccessOperation, ReferenceAddUpdateOperation,
    ReferenceAtomicAddUpdateOperation, ReferenceFreezeOperation, ReferenceNewOperation, ReferenceReadOperation,
    ReferenceSwapOperation, ReferenceType, ReferenceWriteOperation, Region, RegionId, RemOperation, ReshapeOperation,
    ReverseOperation, RoundOperation, RsqrtOperation, SaturatingSubOperation, SelectOperation, Shape, SignOperation,
    SinOperation, SliceOperation, SortDirection, SortOperation, SortOrdering, SqrtOperation, StopGradientOperation,
    StridedLayout, SubOperation, TagOperation, TanOperation, TanhOperation, Tile, TileDimension, TiledLayout,
    Tolerance, TransferToMemoryOperation, TransposeOperation, TypeError, TypeIdentityRenaming, Typed,
    UpdateSliceOperation, WhileOperation, XorOperation, ZeroLikeOperation, ZeroOperation,
    validated_reference_access_descriptors,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

use crate::calls::{KERNEL_KEY_VERSION, KernelCallOperation, KernelDefinition, KernelError, KernelParameter};
use crate::grids::{Grid, GridDimension, GridExecution};
use crate::indexing::TileLoadOperation;
use crate::mappings::{BlockMapping, BoundaryPolicy};
use crate::memory::{
    AsyncCopyOperation, MaskedLoadOperation, MaskedStoreOperation, MaskedSwapOperation, ScratchOperation,
    WaitOperation, WindowMaskOperation,
};
use crate::operations::{KernelExtension, KernelOperation, NoKernelExtension};
use crate::validation::KernelParameterAccess;

/// Errors admitting or reconstructing a portable serialized kernel.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum KernelSerializationError {
    /// The [`KernelSource`] document was written with a schema version that this crate does not decode.
    #[error("unsupported kernel source schema version {version}")]
    Schema {
        /// Schema version recorded in the rejected document.
        version: u32,
    },

    /// The [`KernelDefinition`] is valid but uses a feature that the transport schema cannot represent (e.g., a
    /// distributed array type or an operation outside the portable subset).
    #[error("kernel source serialization does not support {feature}")]
    Unsupported {
        /// Description of the unsupported feature, usually naming the rejected operation. Code fragments such as
        /// operation names are surrounded by backticks.
        feature: String,
    },

    /// The [`KernelSource`] document is structurally well formed but does not describe a valid kernel (e.g., an index
    /// is out of bounds or a literal has the wrong number of bytes).
    #[error("invalid serialized kernel source: {message}")]
    Invalid {
        /// Description of the malformed part of the document.
        message: String,
    },

    /// Computing or reconstructing canonical metadata (e.g., literal addressing, reference access descriptors, or a
    /// sealed program) failed.
    #[error(transparent)]
    Program(#[from] ProgramError),

    /// Reconstructing the kernel call metadata or validating the kernel boundary failed.
    #[error(transparent)]
    Kernel(#[from] KernelError),
}

/// Versioned transport document of one [`KernelDefinition`].
///
/// A source is produced by [`KernelDefinition::to_source`] and consumed by [`KernelDefinition::from_source`]. Its
/// [`Serialize`] and [`Deserialize`] implementations delegate to the complete private wire document, so serializing a
/// source produces exactly the same data as serializing the corresponding definition, in any `serde` format.
/// Deserializing a source performs only the bounded structural decoding of that document, i.e., it checks record
/// shapes and collection size budgets. It does not check the schema version, resolve indices, or reconstruct any
/// program. That validation happens in [`KernelDefinition::from_source`], which reports a typed
/// [`KernelSerializationError`] for every rejected document.
#[derive(Serialize, Deserialize)]
#[serde(transparent)]
pub struct KernelSource {
    /// Complete transport document.
    document: WireDefinition,
}

impl std::fmt::Debug for KernelSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The wire records are private, so the debug rendering exposes only the schema version.
        formatter
            .debug_struct("KernelSource")
            .field("version", &self.document.version)
            .finish_non_exhaustive()
    }
}

impl KernelDefinition {
    /// Encodes this definition as a versioned [`KernelSource`] document.
    ///
    /// The document preserves the call metadata, the exact body program, its shared region edges, literal data bits,
    /// and diagnostic provenance, and it names each nominal dimension identity once. The [`Serialize`] implementation
    /// of [`KernelDefinition`] serializes exactly this document.
    ///
    /// # Errors
    ///
    /// Returns [`KernelSerializationError::Unsupported`] when the definition uses a feature outside the transport
    /// schema (e.g., distributed array metadata, an operation outside the portable subset, a nested kernel call, or
    /// more records or literal bytes than the document budget admits), and [`KernelSerializationError::Program`] when
    /// canonical metadata such as literal addressing or reference access descriptors cannot be computed.
    pub fn to_source(&self) -> Result<KernelSource, KernelSerializationError> {
        Ok(KernelSource { document: Encoder::new(EncodingMode::Transport).definition(self)? })
    }

    /// Reconstructs and validates the definition described by a [`KernelSource`] document.
    ///
    /// Decoding checks the schema version first, then reconstructs every dimension identity, type, literal, and
    /// operation through its canonical constructor, and finally revalidates the region graph and the kernel boundary
    /// exactly as [`KernelDefinition::new`] does for an explicitly built definition. The [`Deserialize`]
    /// implementation of [`KernelDefinition`] deserializes a [`KernelSource`] and then calls this function.
    ///
    /// # Errors
    ///
    /// Returns [`KernelSerializationError::Schema`] when the document has a different schema version,
    /// [`KernelSerializationError::Invalid`] when it is internally inconsistent (e.g., an index is out of bounds or a
    /// literal exceeds its allocation budget), [`KernelSerializationError::Unsupported`] when it describes a payload
    /// that the transport schema does not admit, and [`KernelSerializationError::Program`] or
    /// [`KernelSerializationError::Kernel`] when canonical reconstruction or kernel validation fails.
    pub fn from_source(source: KernelSource) -> Result<Self, KernelSerializationError> {
        source.document.decode()
    }
}

impl Serialize for KernelDefinition {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.to_source().map_err(serde::ser::Error::custom)?.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for KernelDefinition {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::from_source(KernelSource::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

/// Encodes the [`KernelDefinition::semantic_key`] of `definition`, which is the header `kernel key
/// {KERNEL_KEY_VERSION}` and a line break, followed by the compact JSON serialization of the key mode wire document of
/// the definition (refer to the module documentation for the projection and the key-only records of key mode).
///
/// # Errors
///
/// Returns a [`TypeError`] when the body contains a reference constant, an extension without an exact semantic
/// identity, or an operation whose identity is local to the process (i.e., `rematerialize` operations, `composite`
/// calls with retained rules that are not named composites, and `composite_transpose` carriers).
pub(crate) fn encode_semantic_key<Extension: KernelExtension>(
    definition: &KernelDefinition<Extension>,
) -> Result<String, TypeError> {
    let document = Encoder::new(EncodingMode::Key).definition(definition).map_err(|error| match error {
        KernelSerializationError::Program(ProgramError::Type(error)) => error,
        error => TypeError::invalid(error.to_string()),
    })?;
    let document = serde_json::to_string(&document).map_err(|error| TypeError::invalid(error.to_string()))?;
    Ok(format!("kernel key {KERNEL_KEY_VERSION}\n{document}"))
}

/// Version of the encoding of [`KernelSource`] documents, which is independent of [`KERNEL_KEY_VERSION`] (i.e., the
/// version of the encoding of semantic keys). Any change to the produced records requires a new version.
const KERNEL_SOURCE_VERSION: u32 = 1;

/// Maximum total physical bytes allocated for decoded literals, including layout padding.
const MAXIMUM_LITERAL_BYTES: usize = 64 * 1024 * 1024;

/// Maximum number of records in one transport collection.
const MAXIMUM_RECORDS: usize = 1_000_000;

/// Maximum nesting depth of diagnostic provenance, which bounds the recursion of the encoder and the depth of every
/// provenance tree that the decoder reconstructs.
const MAXIMUM_PROVENANCE_DEPTH: usize = 128;

/// Versioned transport envelope. All records are private and carry no execution behavior.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireDefinition {
    /// Schema version that the document was written with.
    version: u32,

    /// Canonical identity table of every nominal dimension variable, which [`WireDimension::Dynamic`] indexes.
    #[serde(deserialize_with = "bounded_records")]
    variables: Vec<WireVariable>,

    /// Kernel call metadata.
    call: WireCall,

    /// Kernel body program.
    body: WireProgram,
}

impl WireDefinition {
    /// Reconstructs canonical identities before any payload or program references them.
    fn decode(self) -> Result<KernelDefinition, KernelSerializationError> {
        if self.version != KERNEL_SOURCE_VERSION {
            return Err(KernelSerializationError::Schema { version: self.version });
        }
        validate_decoded_records(self.variables.len())?;
        let variables = self
            .variables
            .into_iter()
            .map(|variable| {
                let bounds = DimensionBounds::new(variable.lower, variable.upper).map_err(ProgramError::from)?;
                Ok(DimensionVariable::new(variable.name, bounds))
            })
            .collect::<Result<Vec<_>, KernelSerializationError>>()?;
        let decoder = Decoder::new(variables);
        let call = decoder.call(self.call)?;
        let body = decoder.program(self.body, Ok)?;
        Ok(KernelDefinition::new(call, body)?)
    }
}

/// Name and bounds of one nominal dimension identity.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireVariable {
    /// Diagnostic name of the variable.
    name: String,

    /// Inclusive lower bound of the variable.
    lower: usize,

    /// Exclusive upper bound of the variable, or `None` when it is unbounded.
    upper: Option<usize>,
}

/// Canonical dimension occurrence.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum WireDimension {
    /// Static extent.
    Static(usize),

    /// Symbolic extent, holding the index of its identity in [`WireDefinition::variables`].
    Dynamic(usize),
}

/// Canonical array type descriptor with distributed metadata explicitly excluded.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireArrayType {
    /// Data type of the array elements.
    data_type: WireDataType,

    /// Extent of each dimension, in dimension order.
    #[serde(deserialize_with = "bounded_records")]
    shape: Vec<WireDimension>,

    /// Explicit layout, or `None` for the default layout.
    layout: Option<WireLayout>,

    /// Memory placement.
    memory: WireMemory,

    /// Key-only rendering of the distributed metadata (i.e., the sharding and the unreduced and reduced axes), which
    /// only key mode records. Transport documents never contain it, and deserialization rejects it as an unknown
    /// field.
    #[serde(skip_serializing_if = "Option::is_none", skip_deserializing)]
    distribution: Option<String>,
}

/// Stable explicit data type tags, independent of Rust discriminants and diagnostic rendering.
#[derive(Copy, Clone, Serialize, Deserialize)]
enum WireDataType {
    /// Tag of [`DataType::Token`].
    Token,

    /// Tag of [`DataType::Zero`].
    Zero,

    /// Tag of [`DataType::Boolean`].
    Boolean,

    /// Tag of [`DataType::I1`].
    I1,

    /// Tag of [`DataType::I2`].
    I2,

    /// Tag of [`DataType::I4`].
    I4,

    /// Tag of [`DataType::I8`].
    I8,

    /// Tag of [`DataType::I16`].
    I16,

    /// Tag of [`DataType::I32`].
    I32,

    /// Tag of [`DataType::I64`].
    I64,

    /// Tag of [`DataType::U1`].
    U1,

    /// Tag of [`DataType::U2`].
    U2,

    /// Tag of [`DataType::U4`].
    U4,

    /// Tag of [`DataType::U8`].
    U8,

    /// Tag of [`DataType::U16`].
    U16,

    /// Tag of [`DataType::U32`].
    U32,

    /// Tag of [`DataType::U64`].
    U64,

    /// Tag of [`DataType::F4E2M1FN`].
    F4E2M1FN,

    /// Tag of [`DataType::F6E2M3FN`].
    F6E2M3FN,

    /// Tag of [`DataType::F6E3M2FN`].
    F6E3M2FN,

    /// Tag of [`DataType::F8E3M4`].
    F8E3M4,

    /// Tag of [`DataType::F8E4M3`].
    F8E4M3,

    /// Tag of [`DataType::F8E4M3FN`].
    F8E4M3FN,

    /// Tag of [`DataType::F8E4M3FNUZ`].
    F8E4M3FNUZ,

    /// Tag of [`DataType::F8E4M3B11FNUZ`].
    F8E4M3B11FNUZ,

    /// Tag of [`DataType::F8E5M2`].
    F8E5M2,

    /// Tag of [`DataType::F8E5M2FNUZ`].
    F8E5M2FNUZ,

    /// Tag of [`DataType::F8E8M0FNU`].
    F8E8M0FNU,

    /// Tag of [`DataType::BF16`].
    BF16,

    /// Tag of [`DataType::F16`].
    F16,

    /// Tag of [`DataType::F32`].
    F32,

    /// Tag of [`DataType::F64`].
    F64,

    /// Tag of [`DataType::C64`].
    C64,

    /// Tag of [`DataType::C128`].
    C128,
}

impl From<DataType> for WireDataType {
    fn from(value: DataType) -> Self {
        match value {
            DataType::Token => Self::Token,
            DataType::Zero => Self::Zero,
            DataType::Boolean => Self::Boolean,
            DataType::I1 => Self::I1,
            DataType::I2 => Self::I2,
            DataType::I4 => Self::I4,
            DataType::I8 => Self::I8,
            DataType::I16 => Self::I16,
            DataType::I32 => Self::I32,
            DataType::I64 => Self::I64,
            DataType::U1 => Self::U1,
            DataType::U2 => Self::U2,
            DataType::U4 => Self::U4,
            DataType::U8 => Self::U8,
            DataType::U16 => Self::U16,
            DataType::U32 => Self::U32,
            DataType::U64 => Self::U64,
            DataType::F4E2M1FN => Self::F4E2M1FN,
            DataType::F6E2M3FN => Self::F6E2M3FN,
            DataType::F6E3M2FN => Self::F6E3M2FN,
            DataType::F8E3M4 => Self::F8E3M4,
            DataType::F8E4M3 => Self::F8E4M3,
            DataType::F8E4M3FN => Self::F8E4M3FN,
            DataType::F8E4M3FNUZ => Self::F8E4M3FNUZ,
            DataType::F8E4M3B11FNUZ => Self::F8E4M3B11FNUZ,
            DataType::F8E5M2 => Self::F8E5M2,
            DataType::F8E5M2FNUZ => Self::F8E5M2FNUZ,
            DataType::F8E8M0FNU => Self::F8E8M0FNU,
            DataType::BF16 => Self::BF16,
            DataType::F16 => Self::F16,
            DataType::F32 => Self::F32,
            DataType::F64 => Self::F64,
            DataType::C64 => Self::C64,
            DataType::C128 => Self::C128,
        }
    }
}

impl From<WireDataType> for DataType {
    fn from(value: WireDataType) -> Self {
        match value {
            WireDataType::Token => Self::Token,
            WireDataType::Zero => Self::Zero,
            WireDataType::Boolean => Self::Boolean,
            WireDataType::I1 => Self::I1,
            WireDataType::I2 => Self::I2,
            WireDataType::I4 => Self::I4,
            WireDataType::I8 => Self::I8,
            WireDataType::I16 => Self::I16,
            WireDataType::I32 => Self::I32,
            WireDataType::I64 => Self::I64,
            WireDataType::U1 => Self::U1,
            WireDataType::U2 => Self::U2,
            WireDataType::U4 => Self::U4,
            WireDataType::U8 => Self::U8,
            WireDataType::U16 => Self::U16,
            WireDataType::U32 => Self::U32,
            WireDataType::U64 => Self::U64,
            WireDataType::F4E2M1FN => Self::F4E2M1FN,
            WireDataType::F6E2M3FN => Self::F6E2M3FN,
            WireDataType::F6E3M2FN => Self::F6E3M2FN,
            WireDataType::F8E3M4 => Self::F8E3M4,
            WireDataType::F8E4M3 => Self::F8E4M3,
            WireDataType::F8E4M3FN => Self::F8E4M3FN,
            WireDataType::F8E4M3FNUZ => Self::F8E4M3FNUZ,
            WireDataType::F8E4M3B11FNUZ => Self::F8E4M3B11FNUZ,
            WireDataType::F8E5M2 => Self::F8E5M2,
            WireDataType::F8E5M2FNUZ => Self::F8E5M2FNUZ,
            WireDataType::F8E8M0FNU => Self::F8E8M0FNU,
            WireDataType::BF16 => Self::BF16,
            WireDataType::F16 => Self::F16,
            WireDataType::F32 => Self::F32,
            WireDataType::F64 => Self::F64,
            WireDataType::C64 => Self::C64,
            WireDataType::C128 => Self::C128,
        }
    }
}

/// Lossless local layout metadata.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum WireLayout {
    /// Strided layout, holding the stride of each dimension in elements.
    Strided(#[serde(deserialize_with = "bounded_records")] Vec<isize>),

    /// Tiled layout.
    Tiled {
        /// Dimension indices ordered from the most minor to the most major dimension.
        #[serde(deserialize_with = "bounded_records")]
        minor_to_major: Vec<usize>,

        /// Tile dimensions of each tiling level, where `None` denotes a combined tile dimension.
        #[serde(deserialize_with = "bounded_tiles")]
        tiles: Vec<Vec<Option<usize>>>,
    },
}

impl WireLayout {
    /// Encodes lossless layout metadata.
    fn encode(layout: &Layout) -> Self {
        match layout {
            Layout::Strided(layout) => Self::Strided(layout.strides().to_vec()),
            Layout::Tiled(layout) => Self::Tiled {
                minor_to_major: layout.minor_to_major().to_vec(),
                tiles: layout
                    .tiles()
                    .iter()
                    .map(|tile| {
                        tile.dimensions()
                            .iter()
                            .map(|dimension| match dimension {
                                TileDimension::Sized(size) => Some(*size),
                                TileDimension::Combined => None,
                            })
                            .collect()
                    })
                    .collect(),
            },
        }
    }

    /// Reconstructs layout metadata, which the type that carries it validates.
    fn decode(self) -> Layout {
        match self {
            Self::Strided(strides) => Layout::Strided(StridedLayout::new(strides)),
            Self::Tiled { minor_to_major, tiles } => Layout::Tiled(TiledLayout::new(
                minor_to_major,
                tiles
                    .into_iter()
                    .map(|tile| {
                        Tile::new(
                            tile.into_iter()
                                .map(|dimension| match dimension {
                                    Some(size) => TileDimension::Sized(size),
                                    None => TileDimension::Combined,
                                })
                                .collect(),
                        )
                    })
                    .collect(),
            )),
        }
    }
}

/// Local memory placement.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum WireMemory {
    /// Device memory.
    Device,

    /// Host memory.
    Host {
        /// Whether the host memory is pinned.
        pinned: bool,
    },
}

impl From<Memory> for WireMemory {
    fn from(value: Memory) -> Self {
        match value {
            Memory::Device => Self::Device,
            Memory::Host { pinned } => Self::Host { pinned },
        }
    }
}

impl From<WireMemory> for Memory {
    fn from(value: WireMemory) -> Self {
        match value {
            WireMemory::Device => Self::Device,
            WireMemory::Host { pinned } => Self::Host { pinned },
        }
    }
}

/// Canonical value type occurrence.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum WireType {
    /// Array type.
    Array(WireArrayType),

    /// First-class dimension type.
    Dimension(WireDimension),

    /// Reference type, holding the type of its referent.
    Reference(WireArrayType),
}

/// Exact literal payload. Reference storage is never serialized.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum WireValue {
    /// Array literal.
    Array {
        /// Type of the literal.
        r#type: WireArrayType,

        /// Exact logical bytes of the literal elements.
        #[serde(deserialize_with = "bounded_bytes")]
        bytes: Vec<u8>,
    },

    /// Dimension literal.
    Dimension {
        /// Static extent of the literal.
        extent: usize,
    },
}

/// Region-local atom record.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum WireAtom {
    /// Constant atom, holding its literal value.
    Constant(WireValue),

    /// Variable atom, holding its type.
    Variable(WireType),
}

/// Diagnostic provenance retained as a flat arena whose edges reference earlier nodes.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireProvenance {
    /// Provenance nodes in child before parent order.
    #[serde(deserialize_with = "bounded_records")]
    nodes: Vec<WireProvenanceNode>,

    /// Index of the root node in [`WireProvenance::nodes`].
    root: usize,
}

impl WireProvenance {
    /// Encodes diagnostic nodes in child before parent order.
    fn encode(provenance: &Provenance, depth: usize) -> Result<Self, KernelSerializationError> {
        let mut nodes = Vec::new();
        let root = Self::encode_node(provenance, depth, &mut nodes)?;
        Ok(Self { nodes, root })
    }

    /// Bounds recursion over already constructed diagnostic source metadata.
    fn encode_node(
        provenance: &Provenance,
        depth: usize,
        nodes: &mut Vec<WireProvenanceNode>,
    ) -> Result<usize, KernelSerializationError> {
        if depth > MAXIMUM_PROVENANCE_DEPTH {
            return Err(unsupported(format!("provenance nesting deeper than {MAXIMUM_PROVENANCE_DEPTH}")));
        }
        validate_encoded_records(nodes.len() + 1)?;
        let node = if let Some((scope, origin)) = provenance.as_scope() {
            WireProvenanceNode::Scope {
                name: scope.name().to_owned(),
                origin: Self::encode_node(origin, depth + 1, nodes)?,
            }
        } else if let Some(origins) = provenance.as_fused() {
            WireProvenanceNode::Fused(
                origins.iter().map(|origin| Self::encode_node(origin, depth + 1, nodes)).collect::<Result<_, _>>()?,
            )
        } else {
            WireProvenanceNode::Unknown
        };
        validate_encoded_records(nodes.len() + 1)?;
        let index = nodes.len();
        nodes.push(node);
        Ok(index)
    }

    /// Reconstructs checked provenance iteratively, without recursive deserialization or forward edges. Every node may
    /// be referenced at most once, by a later node or as the root, which is the tree form that [`Self::encode`] emits.
    /// A document that shares nodes would describe a provenance whose expansion can be exponentially larger than the
    /// document, and comparing or hashing such a provenance walks that expansion, so sharing is rejected to keep the
    /// work of decoding proportional to the size of the document.
    fn decode(self) -> Result<Provenance, KernelSerializationError> {
        let mut nodes: Vec<Option<(Provenance, usize)>> = Vec::new();
        for node in self.nodes {
            let mut take = |index: usize| {
                nodes
                    .get_mut(index)
                    .ok_or_else(|| invalid("provenance edge must reference an earlier node"))?
                    .take()
                    .ok_or_else(|| invalid(format!("provenance node {index} is referenced more than once")))
            };
            let (provenance, depth) = match node {
                WireProvenanceNode::Unknown => (Provenance::unknown(), 0),
                WireProvenanceNode::Scope { name, origin } => {
                    let (origin, depth) = take(origin)?;
                    (Provenance::scope(ProvenanceScope::new(name), origin), depth + 1)
                }
                WireProvenanceNode::Fused(origins) => {
                    let origins = origins.into_iter().map(&mut take).collect::<Result<Vec<_>, _>>()?;
                    let depth = origins.iter().map(|(_, depth)| *depth).max().unwrap_or(0) + 1;
                    (Provenance::fused(origins.into_iter().map(|(origin, _)| origin)), depth)
                }
            };
            if depth > MAXIMUM_PROVENANCE_DEPTH {
                return Err(invalid(format!("provenance nesting exceeds {MAXIMUM_PROVENANCE_DEPTH}")));
            }
            nodes.push(Some((provenance, depth)));
        }
        nodes
            .get_mut(self.root)
            .ok_or_else(|| invalid("provenance root is out of bounds"))?
            .take()
            .map(|(provenance, _)| provenance)
            .ok_or_else(|| invalid(format!("provenance node {} is referenced more than once", self.root)))
    }
}

/// One diagnostic provenance node. This transport has no recursively nested records.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum WireProvenanceNode {
    /// Unknown provenance.
    Unknown,

    /// Named scope around another provenance.
    Scope {
        /// Name of the scope.
        name: String,

        /// Index of the scoped provenance node, which precedes this node in [`WireProvenance::nodes`].
        origin: usize,
    },

    /// Fusion of several provenances, holding the indices of their nodes, which precede this node in
    /// [`WireProvenance::nodes`].
    Fused(#[serde(deserialize_with = "bounded_records")] Vec<usize>),
}

/// Canonical path of one reference input. Its bindings remain in the ordered input list of the instruction.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireReferenceAccessDescriptor {
    /// Index of the accessed reference in the inputs of the instruction.
    input_index: usize,

    /// Selections applied to the reference, in application order.
    #[serde(deserialize_with = "bounded_records")]
    transforms: Vec<WireReferenceTransform>,
}

/// One canonical selection, encoded independently of an operation's number of base inputs.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum WireReferenceTransform {
    /// Selection of one static index along an axis.
    Index {
        /// Selected axis.
        axis: usize,

        /// Static index along the selected axis.
        index: usize,
    },

    /// Selection of one index along an axis, which a trailing instruction input binds at runtime.
    DynamicIndex {
        /// Selected axis.
        axis: usize,
    },

    /// Static strided slice.
    Slice {
        /// Sliced axes.
        #[serde(deserialize_with = "bounded_records")]
        axes: Vec<WireSliceAxis>,
    },
}

impl WireReferenceTransform {
    /// Encodes one checked canonical selection with bounded axis records.
    fn encode(transform: &ArrayReferenceTransform) -> Result<Self, KernelSerializationError> {
        Ok(match transform {
            ArrayReferenceTransform::Index { axis, index: ArrayReferenceTransformIndex::Static(index) } => {
                Self::Index { axis: *axis, index: *index }
            }
            ArrayReferenceTransform::Index { axis, index: ArrayReferenceTransformIndex::Dynamic } => {
                Self::DynamicIndex { axis: *axis }
            }
            ArrayReferenceTransform::Slice { axes } => {
                validate_encoded_records(axes.len())?;
                Self::Slice {
                    axes: axes
                        .iter()
                        .map(|axis| WireSliceAxis { start: axis.start(), size: axis.size(), stride: axis.stride() })
                        .collect(),
                }
            }
        })
    }

    /// Reconstructs a selection whose indices, extents, and binding types are checked when the program is sealed.
    fn decode(self) -> ArrayReferenceTransform {
        match self {
            Self::Index { axis, index } => {
                ArrayReferenceTransform::Index { axis, index: ArrayReferenceTransformIndex::Static(index) }
            }
            Self::DynamicIndex { axis } => {
                ArrayReferenceTransform::Index { axis, index: ArrayReferenceTransformIndex::Dynamic }
            }
            Self::Slice { axes } => ArrayReferenceTransform::Slice {
                axes: axes.into_iter().map(|axis| ArraySliceAxis::new(axis.start, axis.size, axis.stride)).collect(),
            },
        }
    }
}

/// One static slice axis. Validation remains owned by the canonical reference transform implementation.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireSliceAxis {
    /// First selected index.
    start: usize,

    /// Number of selected indices.
    size: usize,

    /// Distance between consecutive selected indices.
    stride: usize,
}

/// One instruction, whose region indices retain shared region edges without expanding the region graph into a tree.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireInstruction {
    /// Operation payload.
    operation: WireOperation,

    /// Access path of every reference access input, in increasing input order.
    #[serde(deserialize_with = "bounded_records")]
    reference_transforms: Vec<WireReferenceAccessDescriptor>,

    /// Indices of the input atoms in [`WireRegion::atoms`].
    #[serde(deserialize_with = "bounded_records")]
    inputs: Vec<usize>,

    /// Indices of the output atoms in [`WireRegion::atoms`].
    #[serde(deserialize_with = "bounded_records")]
    outputs: Vec<usize>,

    /// Indices of the attached regions in [`WireProgram::regions`], each of which precedes the containing region.
    #[serde(deserialize_with = "bounded_records")]
    regions: Vec<usize>,

    /// Diagnostic provenance.
    provenance: WireProvenance,
}

/// One canonical region, sealed after all its children during decoding.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRegion {
    /// Atoms of the region, in arena order.
    #[serde(deserialize_with = "bounded_records")]
    atoms: Vec<WireAtom>,

    /// Indices of the region inputs in [`WireRegion::atoms`].
    #[serde(deserialize_with = "bounded_records")]
    inputs: Vec<usize>,

    /// Indices of the region outputs in [`WireRegion::atoms`].
    #[serde(deserialize_with = "bounded_records")]
    outputs: Vec<usize>,

    /// Instructions of the region, in program order.
    #[serde(deserialize_with = "bounded_records")]
    instructions: Vec<WireInstruction>,
}

/// Program with flat parameters and its shared region arena.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireProgram {
    /// Region arena, in which every region follows the regions that it attaches.
    #[serde(deserialize_with = "bounded_records")]
    regions: Vec<WireRegion>,

    /// Index of the entry region in [`WireProgram::regions`].
    entry: usize,
}

/// Full call metadata, including coordinate identities retained by specialization.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCall {
    /// Grid dimensions, in grid order.
    #[serde(deserialize_with = "bounded_records")]
    grid: Vec<WireGridDimension>,

    /// Kernel parameters, in parameter order.
    #[serde(deserialize_with = "bounded_records")]
    parameters: Vec<WireParameter>,

    /// Dimension types of the grid coordinates.
    #[serde(deserialize_with = "bounded_records")]
    coordinates: Vec<WireDimension>,

    /// Types of the scalar prefetched inputs.
    #[serde(deserialize_with = "bounded_records")]
    prefetch: Vec<WireArrayType>,
}

/// One named grid dimension.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireGridDimension {
    /// Number of programs along the dimension.
    extent: WireDimension,

    /// Optional diagnostic name of the dimension.
    name: Option<String>,

    /// Execution order of the programs along the dimension.
    execution: WireGridExecution,
}

/// Explicit grid execution vocabulary.
#[derive(Copy, Clone, Serialize, Deserialize)]
enum WireGridExecution {
    /// Programs at different coordinates may execute concurrently.
    Parallel,

    /// Coordinates execute in increasing order within each fixed tuple of other coordinates.
    Sequential,
}

impl From<GridExecution> for WireGridExecution {
    fn from(value: GridExecution) -> Self {
        match value {
            GridExecution::Parallel => Self::Parallel,
            GridExecution::Sequential => Self::Sequential,
        }
    }
}

impl From<WireGridExecution> for GridExecution {
    fn from(value: WireGridExecution) -> Self {
        match value {
            WireGridExecution::Parallel => Self::Parallel,
            WireGridExecution::Sequential => Self::Sequential,
        }
    }
}

/// Canonical access intent and block mapping.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireParameter {
    /// Type of the full parameter array.
    r#type: WireArrayType,

    /// Access intent of the kernel body.
    access: WireAccess,

    /// Block mapping program from grid coordinates to block starts.
    mapping: WireProgram,

    /// Static shape of the block that each program accesses.
    #[serde(deserialize_with = "bounded_records")]
    block_shape: Vec<usize>,

    /// Required handling of a block that crosses the array boundary.
    boundary_policy: WireBoundaryPolicy,
}

/// Explicit boundary access vocabulary.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum WireAccess {
    /// The parameter contains its entering value and publishes nothing.
    ReadOnly,

    /// The parameter starts uninitialized and publishes its final value.
    WriteOnly,

    /// The parameter contains its entering value and publishes its final value.
    ReadWrite,
}

/// Explicit block boundary policy vocabulary.
#[derive(Copy, Clone, Serialize, Deserialize)]
enum WireBoundaryPolicy {
    /// Every selected coordinate must belong to the full array.
    InBounds,

    /// Consumers must explicitly mask invalid coordinates before memory access.
    Masked,
}

impl From<BoundaryPolicy> for WireBoundaryPolicy {
    fn from(value: BoundaryPolicy) -> Self {
        match value {
            BoundaryPolicy::InBounds => Self::InBounds,
            BoundaryPolicy::Masked => Self::Masked,
        }
    }
}

impl From<WireBoundaryPolicy> for BoundaryPolicy {
    fn from(value: WireBoundaryPolicy) -> Self {
        match value {
            WireBoundaryPolicy::InBounds => Self::InBounds,
            WireBoundaryPolicy::Masked => Self::Masked,
        }
    }
}

/// Eligible operation payloads. These records only reconstruct the existing canonical operation implementations.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum WireOperation {
    /// Array [`ConstantOperation`], holding its literal value.
    ArrayConstant(WireValue),

    /// Dimension [`ConstantOperation`], holding its literal value.
    DimensionConstant(WireValue),

    /// Array [`ZeroOperation`], holding its output type.
    Zero(WireArrayType),

    /// [`ZeroOperation`] of the array IR family with explicit runtime extents, holding its output type.
    MixedZero(WireArrayType),

    /// Array [`CompareOperation`], holding its comparison direction.
    Compare(WireComparison),

    /// [`CompareOperation`] of two dimensions, holding its comparison direction.
    DimensionCompare(WireComparison),

    /// [`DotOperation`] with the default precision and no requested output sharding.
    Dot {
        /// Left-hand side contracting, right-hand side contracting, left-hand side batching, and right-hand side
        /// batching dimensions, in that order.
        #[serde(deserialize_with = "bounded_dot_dimensions")]
        dimensions: [Vec<usize>; 4],

        /// Optional accumulation data type.
        accumulation: Option<WireDataType>,
    },

    /// [`WhileOperation`].
    While {
        /// Optional static upper bound on the number of iterations.
        bound: Option<usize>,
    },

    /// [`ConditionOperation`].
    Condition,

    /// [`DimensionFromScalarOperation`], holding its output dimension.
    DimensionFromScalar(WireDimension),

    /// [`DimensionToScalarOperation`].
    DimensionToScalar,

    /// [`TileLoadOperation`].
    TileLoad {
        /// Static shape of the loaded tile.
        #[serde(deserialize_with = "bounded_records")]
        block_shape: Vec<usize>,

        /// Required handling of a tile that crosses the array boundary.
        boundary_policy: WireBoundaryPolicy,
    },

    /// [`ScratchOperation`].
    Scratch {
        /// Type of the allocated scratch referent.
        referent: WireArrayType,

        /// Minimum native allocation alignment in bytes.
        alignment: usize,
    },

    /// [`MaskedLoadOperation`].
    MaskedLoad,

    /// [`MaskedStoreOperation`].
    MaskedStore,

    /// [`MaskedSwapOperation`].
    MaskedSwap,

    /// [`AsyncCopyOperation`].
    AsyncCopy,

    /// [`WaitOperation`].
    Wait,

    /// Array [`AddOperation`].
    Add,

    /// Array [`SubOperation`].
    Sub,

    /// Array [`MulOperation`].
    Mul,

    /// Array [`DivOperation`].
    Div,

    /// Array [`RemOperation`].
    Rem,

    /// Array [`NegOperation`].
    Neg,

    /// Array [`AbsOperation`].
    Abs,

    /// Array [`MinOperation`].
    Min,

    /// Array [`MaxOperation`].
    Max,

    /// Array [`ClampOperation`].
    Clamp,

    /// Array [`ExpOperation`] with the default accuracy.
    Exp,

    /// Array [`LogOperation`] with the default accuracy.
    Log,

    /// Array [`SqrtOperation`] with the default accuracy.
    Sqrt,

    /// Array [`RsqrtOperation`] with the default accuracy.
    Rsqrt,

    /// Array [`TanhOperation`] with the default accuracy.
    Tanh,

    /// Array [`NotOperation`].
    Not,

    /// Array [`AndOperation`].
    And,

    /// Array [`OrOperation`].
    Or,

    /// Array [`XorOperation`].
    Xor,

    /// Array [`SelectOperation`].
    Select,

    /// [`ReferenceNewOperation`].
    ReferenceNew,

    /// [`ReferenceReadOperation`], whose access path is recorded in [`WireInstruction::reference_transforms`].
    ReferenceRead,

    /// [`ReferenceWriteOperation`], whose access path is recorded in [`WireInstruction::reference_transforms`].
    ReferenceWrite,

    /// [`ReferenceSwapOperation`], whose access path is recorded in [`WireInstruction::reference_transforms`].
    ReferenceSwap,

    /// [`ReferenceFreezeOperation`].
    ReferenceFreeze,

    /// [`ReferenceAddUpdateOperation`], whose access path is recorded in [`WireInstruction::reference_transforms`].
    ReferenceAddUpdate,

    /// [`ReferenceAtomicAddUpdateOperation`], whose access path is recorded in
    /// [`WireInstruction::reference_transforms`].
    ReferenceAtomicAddUpdate,

    /// [`DimensionAddOperation`].
    DimensionAdd {
        /// Left-hand side input dimension.
        left: WireDimension,

        /// Right-hand side input dimension.
        right: WireDimension,
    },

    /// [`DimensionSubOperation`].
    DimensionSub {
        /// Left-hand side input dimension.
        left: WireDimension,

        /// Right-hand side input dimension.
        right: WireDimension,
    },

    /// [`DimensionMulOperation`].
    DimensionMul {
        /// Left-hand side input dimension.
        left: WireDimension,

        /// Right-hand side input dimension.
        right: WireDimension,
    },

    /// [`DimensionDivOperation`].
    DimensionDiv {
        /// Left-hand side input dimension.
        left: WireDimension,

        /// Right-hand side input dimension.
        right: WireDimension,
    },

    /// [`DimensionRemOperation`].
    DimensionRem {
        /// Left-hand side input dimension.
        left: WireDimension,

        /// Right-hand side input dimension.
        right: WireDimension,
    },

    /// Array [`SaturatingSubOperation`].
    SaturatingSub,

    /// Dimension [`SaturatingSubOperation`].
    SaturatingSubDimension,

    /// [`ReduceOperation`] with no requested output sharding.
    Reduce {
        /// Reduced axes.
        #[serde(deserialize_with = "bounded_records")]
        axes: Vec<usize>,

        /// Reduction kind.
        kind: WireReductionKind,
    },

    /// [`WindowMaskOperation`].
    WindowMask,

    /// Array [`SignOperation`].
    Sign,

    /// Array [`PowOperation`].
    Pow,

    /// Array [`SinOperation`] with the default accuracy.
    Sin,

    /// Array [`CosOperation`] with the default accuracy.
    Cos,

    /// Array [`TanOperation`] with the default accuracy.
    Tan,

    /// Array [`Atan2Operation`].
    Atan2,

    /// Array [`ExpM1Operation`] with the default accuracy.
    ExpM1,

    /// Array [`Ln1pOperation`] with the default accuracy.
    Ln1p,

    /// Array [`LogAddExpOperation`].
    LogAddExp,

    /// Array [`LogisticOperation`] with the default accuracy.
    Logistic,

    /// Array [`ErfOperation`].
    Erf,

    /// Array [`FloorOperation`].
    Floor,

    /// Array [`CeilOperation`].
    Ceil,

    /// Array [`RoundOperation`].
    Round,

    /// Array [`ComplexOperation`].
    Complex,

    /// Array [`RealOperation`].
    Real,

    /// Array [`ImaginaryOperation`].
    Imaginary,

    /// Array [`ConjugateOperation`].
    Conjugate,

    /// Array unary operation with a non-default result accuracy. Operations with the default accuracy use their own
    /// records, so each operation has exactly one encoding.
    AccurateUnary {
        /// Applied function.
        function: WireAccurateFunction,

        /// Non-default result accuracy.
        accuracy: WireAccuracy,
    },

    /// [`DotOperation`] with a non-default precision and no requested output sharding. Calls with the default
    /// precision use [`WireOperation::Dot`], so each call has exactly one encoding.
    DotWithPrecision {
        /// Left-hand side contracting, right-hand side contracting, left-hand side batching, and right-hand side
        /// batching dimensions, in that order.
        #[serde(deserialize_with = "bounded_dot_dimensions")]
        dimensions: [Vec<usize>; 4],

        /// Optional accumulation data type.
        accumulation: Option<WireDataType>,

        /// Non-default precision.
        precision: WireDotPrecision,
    },

    /// Array [`ZeroLikeOperation`].
    ZeroLike,

    /// Array [`OneOperation`], holding its output type.
    One(WireArrayType),

    /// Array [`OneLikeOperation`].
    OneLike,

    /// Array [`IotaOperation`].
    Iota {
        /// Output type.
        r#type: WireArrayType,

        /// Dimension along which the output counts.
        dimension: usize,
    },

    /// Array [`FillLikeOperation`], holding its rank-0 fill value.
    FillLike(WireValue),

    /// Array [`IotaLikeOperation`].
    IotaLike {
        /// Dimension along which the output counts.
        dimension: usize,

        /// Output data type.
        data_type: WireDataType,
    },

    /// [`OneOperation`] of the array IR family with explicit runtime extents, holding its output type.
    MixedOne(WireArrayType),

    /// [`IotaOperation`] of the array IR family with explicit runtime extents.
    MixedIota {
        /// Output type.
        r#type: WireArrayType,

        /// Dimension along which the output counts.
        dimension: usize,
    },

    /// Array [`ArgMaxOperation`].
    ArgMax {
        /// Reduced axis.
        axis: usize,

        /// Data type of the returned indices.
        index_data_type: WireDataType,
    },

    /// Array [`ArgMinOperation`].
    ArgMin {
        /// Reduced axis.
        axis: usize,

        /// Data type of the returned indices.
        index_data_type: WireDataType,
    },

    /// Array [`CumulativeOperation`].
    Cumulative {
        /// Accumulated axis.
        axis: usize,

        /// Accumulation kind.
        kind: WireCumulativeKind,

        /// Whether the accumulation runs from the last index to the first one.
        reverse: bool,
    },

    /// Array [`SortOperation`].
    Sort {
        /// Sorted axis.
        axis: usize,

        /// Number of leading inputs that form the lexicographic sort key.
        key_count: NonZeroUsize,

        /// Sort direction.
        direction: WireSortDirection,

        /// Ordering of floating-point keys.
        ordering: WireSortOrdering,
    },

    /// Array [`TransposeOperation`], holding its permutation axes.
    Transpose(#[serde(deserialize_with = "bounded_records")] Vec<i128>),

    /// Array [`ReverseOperation`], holding its reversed axes.
    Reverse(#[serde(deserialize_with = "bounded_records")] Vec<i128>),

    /// Array [`ReshapeOperation`] with no requested output sharding.
    Reshape {
        /// Declared input shape.
        #[serde(deserialize_with = "bounded_records")]
        input_shape: Vec<WireDimension>,

        /// Output shape.
        #[serde(deserialize_with = "bounded_records")]
        output_shape: Vec<WireDimension>,
    },

    /// [`DynamicReshapeOperation`] of the array IR family with no requested output sharding, whose runtime assertion
    /// metadata is reconstructed from the input types of its instruction.
    MixedReshape,

    /// Array [`BroadcastOperation`].
    Broadcast {
        /// Output type.
        output_type: WireArrayType,

        /// Output axis of each input axis.
        #[serde(deserialize_with = "bounded_records")]
        output_axes: Vec<usize>,
    },

    /// Array [`BroadcastLikeOperation`], holding the output axis of each input axis.
    BroadcastLike(#[serde(deserialize_with = "bounded_records")] Vec<usize>),

    /// [`DynamicBroadcastOperation`] of the array IR family with no requested output sharding.
    MixedBroadcast {
        /// Output axis of each input axis.
        #[serde(deserialize_with = "bounded_records")]
        output_axes: Vec<usize>,

        /// Requested output layout, or `None` for the default layout.
        output_layout: Option<WireLayout>,
    },

    /// Array [`PadOperation`].
    Pad {
        /// Padding before the first element of each axis, which may be negative.
        #[serde(deserialize_with = "bounded_records")]
        low: Vec<i64>,

        /// Padding after the last element of each axis, which may be negative.
        #[serde(deserialize_with = "bounded_records")]
        high: Vec<i64>,

        /// Padding between consecutive elements of each axis.
        #[serde(deserialize_with = "bounded_records")]
        interior: Vec<usize>,
    },

    /// [`PadOperation`] of the array IR family, whose runtime assertion metadata is reconstructed from the input types
    /// of its instruction.
    MixedPad {
        /// Padding before the first element of each axis, which may be negative.
        #[serde(deserialize_with = "bounded_records")]
        low: Vec<i64>,

        /// Padding after the last element of each axis, which may be negative.
        #[serde(deserialize_with = "bounded_records")]
        high: Vec<i64>,

        /// Padding between consecutive elements of each axis.
        #[serde(deserialize_with = "bounded_records")]
        interior: Vec<usize>,
    },

    /// Array [`ConcatenateOperation`], whose rank is reconstructed from the first input type of its instruction.
    Concatenate {
        /// Concatenated axis.
        axis: usize,
    },

    /// [`ConcatenateOperation`] of the array IR family, whose runtime assertion metadata is reconstructed from the
    /// input types of its instruction.
    MixedConcatenate {
        /// Concatenated axis.
        axis: usize,
    },

    /// Array [`SliceOperation`].
    Slice {
        /// First selected index of each axis.
        #[serde(deserialize_with = "bounded_records")]
        start_indices: Vec<usize>,

        /// Exclusive limit of each axis.
        #[serde(deserialize_with = "bounded_records")]
        limits: Vec<WireDimension>,

        /// Distance between consecutive selected indices of each axis.
        #[serde(deserialize_with = "bounded_records")]
        strides: Vec<usize>,
    },

    /// Array [`UpdateSliceOperation`], holding the first updated index of each axis.
    UpdateSlice(#[serde(deserialize_with = "bounded_records")] Vec<usize>),

    /// Array [`DynamicSliceOperation`].
    DynamicSlice {
        /// Size of the slice along each axis.
        #[serde(deserialize_with = "bounded_records")]
        sizes: Vec<usize>,

        /// Whether negative start indices count from the end of their axis.
        allow_negative_indices: bool,
    },

    /// Array [`DynamicUpdateSliceOperation`].
    DynamicUpdateSlice {
        /// Whether negative start indices count from the end of their axis.
        allow_negative_indices: bool,
    },

    /// Array [`ConvertElementTypeOperation`].
    ConvertElementType {
        /// Output data type.
        data_type: WireDataType,

        /// Whether the conversion reinterprets the element bits instead of converting their values.
        bitcast: bool,
    },

    /// Array [`ReducePrecisionOperation`].
    ReducePrecision {
        /// Number of exponent bits of the emulated format.
        exponent_bits: u32,

        /// Number of mantissa bits of the emulated format.
        mantissa_bits: u32,
    },

    /// Array [`TransferToMemoryOperation`], holding its destination memory.
    TransferToMemory(WireMemory),

    /// Array [`ConditionOperation`].
    ArrayCondition,

    /// Array [`WhileOperation`].
    ArrayWhile {
        /// Optional static upper bound on the number of iterations.
        bound: Option<usize>,
    },

    /// Array [`StopGradientOperation`].
    StopGradient,

    /// Array [`TagOperation`], holding its tag key.
    Tag(String),

    /// Array [`CastOperation`].
    Cast {
        /// Declared output type.
        output_type: WireArrayType,

        /// Declared input types.
        #[serde(deserialize_with = "bounded_records")]
        input_types: Vec<WireArrayType>,

        /// Whether the cast checks its declaration at runtime.
        checked: bool,
    },

    /// [`CastOperation`] of the array IR family.
    MixedCast {
        /// Declared output type.
        output_type: WireType,

        /// Declared input types.
        #[serde(deserialize_with = "bounded_records")]
        input_types: Vec<WireType>,

        /// Whether the cast checks its declaration at runtime.
        checked: bool,
    },

    /// [`DimensionMaxOperation`].
    DimensionMax {
        /// Left-hand side input dimension.
        left: WireDimension,

        /// Right-hand side input dimension.
        right: WireDimension,
    },

    /// [`DimensionMinOperation`].
    DimensionMin {
        /// Left-hand side input dimension.
        left: WireDimension,

        /// Right-hand side input dimension.
        right: WireDimension,
    },

    /// [`DimensionPowOperation`].
    DimensionPow {
        /// Left-hand side input dimension.
        left: WireDimension,

        /// Right-hand side input dimension.
        right: WireDimension,
    },

    /// [`DimensionSizeOperation`], whose selected dimension is reconstructed from the input type of its instruction.
    DimensionSize {
        /// Selected axis.
        axis: usize,
    },

    /// Key-only record, which only key mode emits and deserialization never accepts.
    #[serde(skip_deserializing)]
    KeyOnly(WireKeyOnlyOperation),
}

/// Explicit comparison predicate.
#[derive(Copy, Clone, Serialize, Deserialize)]
enum WireComparison {
    /// Tag of [`ComparisonDirection::Equal`].
    Equal,

    /// Tag of [`ComparisonDirection::NotEqual`].
    NotEqual,

    /// Tag of [`ComparisonDirection::LessThan`].
    LessThan,

    /// Tag of [`ComparisonDirection::LessThanOrEqual`].
    LessThanOrEqual,

    /// Tag of [`ComparisonDirection::GreaterThan`].
    GreaterThan,

    /// Tag of [`ComparisonDirection::GreaterThanOrEqual`].
    GreaterThanOrEqual,
}

impl From<ComparisonDirection> for WireComparison {
    fn from(value: ComparisonDirection) -> Self {
        match value {
            ComparisonDirection::Equal => Self::Equal,
            ComparisonDirection::NotEqual => Self::NotEqual,
            ComparisonDirection::LessThan => Self::LessThan,
            ComparisonDirection::LessThanOrEqual => Self::LessThanOrEqual,
            ComparisonDirection::GreaterThan => Self::GreaterThan,
            ComparisonDirection::GreaterThanOrEqual => Self::GreaterThanOrEqual,
        }
    }
}

impl From<WireComparison> for ComparisonDirection {
    fn from(value: WireComparison) -> Self {
        match value {
            WireComparison::Equal => Self::Equal,
            WireComparison::NotEqual => Self::NotEqual,
            WireComparison::LessThan => Self::LessThan,
            WireComparison::LessThanOrEqual => Self::LessThanOrEqual,
            WireComparison::GreaterThan => Self::GreaterThan,
            WireComparison::GreaterThanOrEqual => Self::GreaterThanOrEqual,
        }
    }
}

/// Explicit reduction kind.
#[derive(Copy, Clone, Serialize, Deserialize)]
enum WireReductionKind {
    /// Tag of [`ReductionKind::Sum`].
    Sum,

    /// Tag of [`ReductionKind::Product`].
    Product,

    /// Tag of [`ReductionKind::Mean`].
    Mean,

    /// Tag of [`ReductionKind::LogSumExp`].
    LogSumExp,

    /// Tag of [`ReductionKind::Max`].
    Max,

    /// Tag of [`ReductionKind::Min`].
    Min,

    /// Tag of [`ReductionKind::Any`].
    Any,

    /// Tag of [`ReductionKind::All`].
    All,
}

impl From<ReductionKind> for WireReductionKind {
    fn from(value: ReductionKind) -> Self {
        match value {
            ReductionKind::Sum => Self::Sum,
            ReductionKind::Product => Self::Product,
            ReductionKind::Mean => Self::Mean,
            ReductionKind::LogSumExp => Self::LogSumExp,
            ReductionKind::Max => Self::Max,
            ReductionKind::Min => Self::Min,
            ReductionKind::Any => Self::Any,
            ReductionKind::All => Self::All,
        }
    }
}

impl From<WireReductionKind> for ReductionKind {
    fn from(value: WireReductionKind) -> Self {
        match value {
            WireReductionKind::Sum => Self::Sum,
            WireReductionKind::Product => Self::Product,
            WireReductionKind::Mean => Self::Mean,
            WireReductionKind::LogSumExp => Self::LogSumExp,
            WireReductionKind::Max => Self::Max,
            WireReductionKind::Min => Self::Min,
            WireReductionKind::Any => Self::Any,
            WireReductionKind::All => Self::All,
        }
    }
}

/// Explicit unary function with a result accuracy.
#[derive(Copy, Clone, Serialize, Deserialize)]
enum WireAccurateFunction {
    /// Tag of [`ExpOperation`].
    Exp,

    /// Tag of [`ExpM1Operation`].
    ExpM1,

    /// Tag of [`LogOperation`].
    Log,

    /// Tag of [`Ln1pOperation`].
    Ln1p,

    /// Tag of [`LogisticOperation`].
    Logistic,

    /// Tag of [`SinOperation`].
    Sin,

    /// Tag of [`CosOperation`].
    Cos,

    /// Tag of [`TanOperation`].
    Tan,

    /// Tag of [`TanhOperation`].
    Tanh,

    /// Tag of [`SqrtOperation`].
    Sqrt,

    /// Tag of [`RsqrtOperation`].
    Rsqrt,
}

/// Explicit non-default result accuracy. The default accuracy has no tag, because it selects the compact record of
/// each operation.
#[derive(Copy, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum WireAccuracy {
    /// Tag of [`Accuracy::Highest`].
    Highest,

    /// [`Accuracy::Tolerance`], whose floating-point tolerances are encoded by their IEEE bits.
    Tolerance {
        /// IEEE bits of the absolute error tolerance.
        absolute: u64,

        /// IEEE bits of the relative error tolerance.
        relative: u64,

        /// Error tolerance in units in the last place of the exact result.
        units_of_least_precision: usize,
    },
}

impl WireAccuracy {
    /// Selects `record` for the default accuracy and otherwise an [`WireOperation::AccurateUnary`] record of
    /// `function`, so that each operation has exactly one encoding.
    fn encode(record: WireOperation, function: WireAccurateFunction, accuracy: Accuracy) -> WireOperation {
        let accuracy = match accuracy {
            Accuracy::Default => return record,
            Accuracy::Highest => Self::Highest,
            Accuracy::Tolerance(tolerance) => Self::Tolerance {
                absolute: tolerance.absolute().to_bits(),
                relative: tolerance.relative().to_bits(),
                units_of_least_precision: tolerance.units_of_least_precision(),
            },
        };
        WireOperation::AccurateUnary { function, accuracy }
    }

    /// Reconstructs an accuracy through the checked [`Tolerance`] constructor.
    fn decode(self) -> Result<Accuracy, KernelSerializationError> {
        Ok(match self {
            Self::Highest => Accuracy::Highest,
            Self::Tolerance { absolute, relative, units_of_least_precision } => Accuracy::Tolerance(Tolerance::new(
                f64::from_bits(absolute),
                f64::from_bits(relative),
                units_of_least_precision,
            )?),
        })
    }
}

/// Explicit dot precision.
#[derive(Copy, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum WireDotPrecision {
    /// [`DotPrecision::Inputs`].
    Inputs {
        /// Precision of the left-hand side input.
        left: WireDotInputPrecision,

        /// Precision of the right-hand side input.
        right: WireDotInputPrecision,
    },

    /// [`DotPrecision::Algorithm`].
    Algorithm {
        /// Precision type of the left-hand side input.
        left_precision_type: WireDataType,

        /// Precision type of the right-hand side input.
        right_precision_type: WireDataType,

        /// Accumulation type.
        accumulation_type: WireDataType,

        /// Number of components of the left-hand side input.
        left_component_count: usize,

        /// Number of components of the right-hand side input.
        right_component_count: usize,

        /// Number of primitive operations.
        primitive_operation_count: usize,

        /// Whether the accumulation may be imprecise.
        allow_imprecise_accumulation: bool,
    },

    /// [`DotPrecision::Preset`].
    Preset(WireDotAlgorithmPreset),
}

impl From<DotPrecision> for WireDotPrecision {
    fn from(value: DotPrecision) -> Self {
        match value {
            DotPrecision::Inputs { lhs, rhs } => Self::Inputs { left: lhs.into(), right: rhs.into() },
            DotPrecision::Algorithm(algorithm) => Self::Algorithm {
                left_precision_type: algorithm.lhs_precision_type().into(),
                right_precision_type: algorithm.rhs_precision_type().into(),
                accumulation_type: algorithm.accumulation_type().into(),
                left_component_count: algorithm.lhs_component_count(),
                right_component_count: algorithm.rhs_component_count(),
                primitive_operation_count: algorithm.primitive_operation_count(),
                allow_imprecise_accumulation: algorithm.allow_imprecise_accumulation(),
            },
            DotPrecision::Preset(preset) => Self::Preset(preset.into()),
        }
    }
}

impl From<WireDotPrecision> for DotPrecision {
    fn from(value: WireDotPrecision) -> Self {
        match value {
            WireDotPrecision::Inputs { left, right } => Self::Inputs { lhs: left.into(), rhs: right.into() },
            WireDotPrecision::Algorithm {
                left_precision_type,
                right_precision_type,
                accumulation_type,
                left_component_count,
                right_component_count,
                primitive_operation_count,
                allow_imprecise_accumulation,
            } => Self::Algorithm(
                DotAlgorithm::new(left_precision_type.into(), right_precision_type.into(), accumulation_type.into())
                    .with_component_counts(left_component_count, right_component_count)
                    .with_primitive_operation_count(primitive_operation_count)
                    .with_imprecise_accumulation(allow_imprecise_accumulation),
            ),
            WireDotPrecision::Preset(preset) => Self::Preset(preset.into()),
        }
    }
}

/// Explicit dot input precision.
#[derive(Copy, Clone, Serialize, Deserialize)]
enum WireDotInputPrecision {
    /// Tag of [`DotInputPrecision::Default`].
    Default,

    /// Tag of [`DotInputPrecision::High`].
    High,

    /// Tag of [`DotInputPrecision::Highest`].
    Highest,
}

impl From<DotInputPrecision> for WireDotInputPrecision {
    fn from(value: DotInputPrecision) -> Self {
        match value {
            DotInputPrecision::Default => Self::Default,
            DotInputPrecision::High => Self::High,
            DotInputPrecision::Highest => Self::Highest,
        }
    }
}

impl From<WireDotInputPrecision> for DotInputPrecision {
    fn from(value: WireDotInputPrecision) -> Self {
        match value {
            WireDotInputPrecision::Default => Self::Default,
            WireDotInputPrecision::High => Self::High,
            WireDotInputPrecision::Highest => Self::Highest,
        }
    }
}

/// Explicit dot algorithm preset.
#[derive(Copy, Clone, Serialize, Deserialize)]
enum WireDotAlgorithmPreset {
    /// Tag of [`DotAlgorithmPreset::Any_F8_Any_F8_F32`].
    AnyF8AnyF8F32,

    /// Tag of [`DotAlgorithmPreset::Any_F8_Any_F8_F32_Fast_Accumulation`].
    AnyF8AnyF8F32FastAccumulation,

    /// Tag of [`DotAlgorithmPreset::Any_F8_Any_F8_Any`].
    AnyF8AnyF8Any,

    /// Tag of [`DotAlgorithmPreset::Any_F8_Any_F8_Any_Fast_Accumulation`].
    AnyF8AnyF8AnyFastAccumulation,

    /// Tag of [`DotAlgorithmPreset::F16_F16_F16`].
    F16F16F16,

    /// Tag of [`DotAlgorithmPreset::F16_F16_F32`].
    F16F16F32,

    /// Tag of [`DotAlgorithmPreset::BF16_BF16_BF16`].
    BF16BF16BF16,

    /// Tag of [`DotAlgorithmPreset::BF16_BF16_F32`].
    BF16BF16F32,

    /// Tag of [`DotAlgorithmPreset::BF16_BF16_F32_X3`].
    BF16BF16F32X3,

    /// Tag of [`DotAlgorithmPreset::BF16_BF16_F32_X6`].
    BF16BF16F32X6,

    /// Tag of [`DotAlgorithmPreset::BF16_BF16_F32_X9`].
    BF16BF16F32X9,

    /// Tag of [`DotAlgorithmPreset::TF32_TF32_F32`].
    TF32TF32F32,

    /// Tag of [`DotAlgorithmPreset::TF32_TF32_F32_X3`].
    TF32TF32F32X3,

    /// Tag of [`DotAlgorithmPreset::F32_F32_F32`].
    F32F32F32,

    /// Tag of [`DotAlgorithmPreset::F64_F64_F64`].
    F64F64F64,
}

impl From<DotAlgorithmPreset> for WireDotAlgorithmPreset {
    fn from(value: DotAlgorithmPreset) -> Self {
        match value {
            DotAlgorithmPreset::Any_F8_Any_F8_F32 => Self::AnyF8AnyF8F32,
            DotAlgorithmPreset::Any_F8_Any_F8_F32_Fast_Accumulation => Self::AnyF8AnyF8F32FastAccumulation,
            DotAlgorithmPreset::Any_F8_Any_F8_Any => Self::AnyF8AnyF8Any,
            DotAlgorithmPreset::Any_F8_Any_F8_Any_Fast_Accumulation => Self::AnyF8AnyF8AnyFastAccumulation,
            DotAlgorithmPreset::F16_F16_F16 => Self::F16F16F16,
            DotAlgorithmPreset::F16_F16_F32 => Self::F16F16F32,
            DotAlgorithmPreset::BF16_BF16_BF16 => Self::BF16BF16BF16,
            DotAlgorithmPreset::BF16_BF16_F32 => Self::BF16BF16F32,
            DotAlgorithmPreset::BF16_BF16_F32_X3 => Self::BF16BF16F32X3,
            DotAlgorithmPreset::BF16_BF16_F32_X6 => Self::BF16BF16F32X6,
            DotAlgorithmPreset::BF16_BF16_F32_X9 => Self::BF16BF16F32X9,
            DotAlgorithmPreset::TF32_TF32_F32 => Self::TF32TF32F32,
            DotAlgorithmPreset::TF32_TF32_F32_X3 => Self::TF32TF32F32X3,
            DotAlgorithmPreset::F32_F32_F32 => Self::F32F32F32,
            DotAlgorithmPreset::F64_F64_F64 => Self::F64F64F64,
        }
    }
}

impl From<WireDotAlgorithmPreset> for DotAlgorithmPreset {
    fn from(value: WireDotAlgorithmPreset) -> Self {
        match value {
            WireDotAlgorithmPreset::AnyF8AnyF8F32 => Self::Any_F8_Any_F8_F32,
            WireDotAlgorithmPreset::AnyF8AnyF8F32FastAccumulation => Self::Any_F8_Any_F8_F32_Fast_Accumulation,
            WireDotAlgorithmPreset::AnyF8AnyF8Any => Self::Any_F8_Any_F8_Any,
            WireDotAlgorithmPreset::AnyF8AnyF8AnyFastAccumulation => Self::Any_F8_Any_F8_Any_Fast_Accumulation,
            WireDotAlgorithmPreset::F16F16F16 => Self::F16_F16_F16,
            WireDotAlgorithmPreset::F16F16F32 => Self::F16_F16_F32,
            WireDotAlgorithmPreset::BF16BF16BF16 => Self::BF16_BF16_BF16,
            WireDotAlgorithmPreset::BF16BF16F32 => Self::BF16_BF16_F32,
            WireDotAlgorithmPreset::BF16BF16F32X3 => Self::BF16_BF16_F32_X3,
            WireDotAlgorithmPreset::BF16BF16F32X6 => Self::BF16_BF16_F32_X6,
            WireDotAlgorithmPreset::BF16BF16F32X9 => Self::BF16_BF16_F32_X9,
            WireDotAlgorithmPreset::TF32TF32F32 => Self::TF32_TF32_F32,
            WireDotAlgorithmPreset::TF32TF32F32X3 => Self::TF32_TF32_F32_X3,
            WireDotAlgorithmPreset::F32F32F32 => Self::F32_F32_F32,
            WireDotAlgorithmPreset::F64F64F64 => Self::F64_F64_F64,
        }
    }
}

/// Explicit cumulative accumulation kind.
#[derive(Copy, Clone, Serialize, Deserialize)]
enum WireCumulativeKind {
    /// Tag of [`CumulativeKind::Sum`].
    Sum,

    /// Tag of [`CumulativeKind::Product`].
    Product,

    /// Tag of [`CumulativeKind::LogSumExp`].
    LogSumExp,

    /// Tag of [`CumulativeKind::Max`].
    Max,

    /// Tag of [`CumulativeKind::Min`].
    Min,
}

impl From<CumulativeKind> for WireCumulativeKind {
    fn from(value: CumulativeKind) -> Self {
        match value {
            CumulativeKind::Sum => Self::Sum,
            CumulativeKind::Product => Self::Product,
            CumulativeKind::LogSumExp => Self::LogSumExp,
            CumulativeKind::Max => Self::Max,
            CumulativeKind::Min => Self::Min,
        }
    }
}

impl From<WireCumulativeKind> for CumulativeKind {
    fn from(value: WireCumulativeKind) -> Self {
        match value {
            WireCumulativeKind::Sum => Self::Sum,
            WireCumulativeKind::Product => Self::Product,
            WireCumulativeKind::LogSumExp => Self::LogSumExp,
            WireCumulativeKind::Max => Self::Max,
            WireCumulativeKind::Min => Self::Min,
        }
    }
}

/// Explicit sort direction.
#[derive(Copy, Clone, Serialize, Deserialize)]
enum WireSortDirection {
    /// Tag of [`SortDirection::Ascending`].
    Ascending,

    /// Tag of [`SortDirection::Descending`].
    Descending,
}

impl From<SortDirection> for WireSortDirection {
    fn from(value: SortDirection) -> Self {
        match value {
            SortDirection::Ascending => Self::Ascending,
            SortDirection::Descending => Self::Descending,
        }
    }
}

impl From<WireSortDirection> for SortDirection {
    fn from(value: WireSortDirection) -> Self {
        match value {
            WireSortDirection::Ascending => Self::Ascending,
            WireSortDirection::Descending => Self::Descending,
        }
    }
}

/// Explicit ordering of floating-point sort keys.
#[derive(Copy, Clone, Serialize, Deserialize)]
enum WireSortOrdering {
    /// Tag of [`SortOrdering::Canonical`].
    Canonical,

    /// Tag of [`SortOrdering::Total`].
    Total,
}

impl From<SortOrdering> for WireSortOrdering {
    fn from(value: SortOrdering) -> Self {
        match value {
            SortOrdering::Canonical => Self::Canonical,
            SortOrdering::Total => Self::Total,
        }
    }
}

impl From<WireSortOrdering> for SortOrdering {
    fn from(value: WireSortOrdering) -> Self {
        match value {
            WireSortOrdering::Canonical => Self::Canonical,
            WireSortOrdering::Total => Self::Total,
        }
    }
}

/// Key-only operation record for a payload that the transport schema does not cover. These records only serialize,
/// so no document that contains one can be deserialized.
#[derive(Serialize)]
enum WireKeyOnlyOperation {
    /// Nested [`KernelCallOperation`], holding its call metadata. Its body is an attached region of its instruction.
    Call(WireCall),

    /// Kernel extension, holding its exact versioned [`KernelExtension::semantic_key`] payload.
    Extension(Vec<u8>),

    /// Dimension arithmetic whose cached inference metadata canonical reconstruction does not preserve (e.g., after
    /// a refinement). The cached output name is diagnostic and therefore excluded.
    DimensionArithmetic {
        /// Name of the operation.
        operation: &'static str,

        /// Declared left-hand side input dimension.
        left: WireDimension,

        /// Declared right-hand side input dimension.
        right: WireDimension,

        /// Inclusive lower bound of the declared output.
        output_lower: usize,

        /// Exclusive upper bound of the declared output, or `None` when it is unbounded.
        output_upper: Option<usize>,

        /// Effect classes of the operation, in canonical order.
        effects: Vec<WireEffectClass>,
    },

    /// [`CustomCallOperation`].
    CustomCall {
        /// Name of the called target.
        target: String,

        /// Named attributes, in declaration order.
        attributes: Vec<(String, WireCustomCallAttribute)>,

        /// Required layout of each input, where `None` leaves the layout unconstrained.
        input_layouts: Vec<Option<WireLayout>>,

        /// Input and output index of each alias.
        input_output_aliases: Vec<(usize, usize)>,

        /// Output types.
        output_types: Vec<WireArrayType>,

        /// Effect class, or `None` for a pure call.
        effect_class: Option<WireEffectClass>,

        /// Batching rule.
        batching: WireCustomCallBatching,
    },

    /// [`CompositeOperation`] whose rules are attached regions of its instruction.
    AttachedComposite {
        /// Number of leading inputs that are not differentiated.
        non_differentiated_count: usize,

        /// Forward rule of the composite.
        jvp_rule: WireCompositeJvpRule,

        /// Whether a reverse mode rule is attached.
        has_vjp_rule: bool,
    },

    /// [`CompositeOperation`] that calls a named composite definition.
    NamedComposite {
        /// Transitional [`Debug`](std::fmt::Debug) rendering of the operation, which covers its retained rule state.
        operation: String,

        /// Key of the named composite definition, which identifies its rules across processes.
        key: WireCompositeKey,
    },

    /// Transitional [`Debug`](std::fmt::Debug) rendering of an operation that has no wire record, after renaming the
    /// nominal identities of the document to canonical names.
    Debug(String),
}

/// Explicit effect class.
#[derive(Copy, Clone, Serialize)]
enum WireEffectClass {
    /// Tag of [`EffectClass::OrderedAssertion`].
    OrderedAssertion,

    /// Tag of [`EffectClass::OrderedIo`].
    OrderedIo,

    /// Tag of [`EffectClass::DeviceOrderedIo`].
    DeviceOrderedIo,

    /// Tag of [`EffectClass::UnorderedIo`].
    UnorderedIo,

    /// Tag of [`EffectClass::OrderedState`].
    OrderedState,
}

impl From<EffectClass> for WireEffectClass {
    fn from(value: EffectClass) -> Self {
        match value {
            EffectClass::OrderedAssertion => Self::OrderedAssertion,
            EffectClass::OrderedIo => Self::OrderedIo,
            EffectClass::DeviceOrderedIo => Self::DeviceOrderedIo,
            EffectClass::UnorderedIo => Self::UnorderedIo,
            EffectClass::OrderedState => Self::OrderedState,
        }
    }
}

/// Exact custom call attribute, whose floating-point values are encoded by their IEEE bits.
#[derive(Serialize)]
enum WireCustomCallAttribute {
    /// [`CustomCallAttribute::String`].
    String(String),

    /// [`CustomCallAttribute::Bytes`].
    Bytes(Vec<u8>),

    /// [`CustomCallAttribute::Boolean`].
    Boolean(bool),

    /// [`CustomCallAttribute::I8`].
    I8(i8),

    /// [`CustomCallAttribute::I16`].
    I16(i16),

    /// [`CustomCallAttribute::I32`].
    I32(i32),

    /// [`CustomCallAttribute::I64`].
    I64(i64),

    /// [`CustomCallAttribute::U8`].
    U8(u8),

    /// [`CustomCallAttribute::U16`].
    U16(u16),

    /// [`CustomCallAttribute::U32`].
    U32(u32),

    /// [`CustomCallAttribute::U64`].
    U64(u64),

    /// [`CustomCallAttribute::F32`], holding its IEEE bits.
    F32(u32),

    /// [`CustomCallAttribute::F64`], holding its IEEE bits.
    F64(u64),

    /// [`CustomCallAttribute::Array`].
    Array(WireCustomCallArrayAttribute),

    /// [`CustomCallAttribute::Dictionary`], holding its named attributes in declaration order.
    Dictionary(Vec<(String, WireCustomCallAttribute)>),
}

impl From<&CustomCallAttribute> for WireCustomCallAttribute {
    fn from(value: &CustomCallAttribute) -> Self {
        match value {
            CustomCallAttribute::String(value) => Self::String(value.clone()),
            CustomCallAttribute::Bytes(value) => Self::Bytes(value.clone()),
            CustomCallAttribute::Boolean(value) => Self::Boolean(*value),
            CustomCallAttribute::I8(value) => Self::I8(*value),
            CustomCallAttribute::I16(value) => Self::I16(*value),
            CustomCallAttribute::I32(value) => Self::I32(*value),
            CustomCallAttribute::I64(value) => Self::I64(*value),
            CustomCallAttribute::U8(value) => Self::U8(*value),
            CustomCallAttribute::U16(value) => Self::U16(*value),
            CustomCallAttribute::U32(value) => Self::U32(*value),
            CustomCallAttribute::U64(value) => Self::U64(*value),
            CustomCallAttribute::F32(value) => Self::F32(value.to_bits()),
            CustomCallAttribute::F64(value) => Self::F64(value.to_bits()),
            CustomCallAttribute::Array(value) => Self::Array(value.into()),
            CustomCallAttribute::Dictionary(attributes) => Self::Dictionary(
                attributes.iter().map(|(name, value)| (name.clone(), value.into())).collect(),
            ),
        }
    }
}

/// Exact custom call array attribute, whose floating-point values are encoded by their IEEE bits.
#[derive(Serialize)]
enum WireCustomCallArrayAttribute {
    /// [`CustomCallArrayAttribute::I8`].
    I8(Vec<i8>),

    /// [`CustomCallArrayAttribute::I16`].
    I16(Vec<i16>),

    /// [`CustomCallArrayAttribute::I32`].
    I32(Vec<i32>),

    /// [`CustomCallArrayAttribute::I64`].
    I64(Vec<i64>),

    /// [`CustomCallArrayAttribute::U8`].
    U8(Vec<u8>),

    /// [`CustomCallArrayAttribute::U16`].
    U16(Vec<u16>),

    /// [`CustomCallArrayAttribute::U32`].
    U32(Vec<u32>),

    /// [`CustomCallArrayAttribute::U64`].
    U64(Vec<u64>),

    /// [`CustomCallArrayAttribute::F32`], holding the IEEE bits of its values.
    F32(Vec<u32>),

    /// [`CustomCallArrayAttribute::F64`], holding the IEEE bits of its values.
    F64(Vec<u64>),
}

impl From<&CustomCallArrayAttribute> for WireCustomCallArrayAttribute {
    fn from(value: &CustomCallArrayAttribute) -> Self {
        match value {
            CustomCallArrayAttribute::I8(values) => Self::I8(values.clone()),
            CustomCallArrayAttribute::I16(values) => Self::I16(values.clone()),
            CustomCallArrayAttribute::I32(values) => Self::I32(values.clone()),
            CustomCallArrayAttribute::I64(values) => Self::I64(values.clone()),
            CustomCallArrayAttribute::U8(values) => Self::U8(values.clone()),
            CustomCallArrayAttribute::U16(values) => Self::U16(values.clone()),
            CustomCallArrayAttribute::U32(values) => Self::U32(values.clone()),
            CustomCallArrayAttribute::U64(values) => Self::U64(values.clone()),
            CustomCallArrayAttribute::F32(values) => Self::F32(values.iter().map(|value| value.to_bits()).collect()),
            CustomCallArrayAttribute::F64(values) => Self::F64(values.iter().map(|value| value.to_bits()).collect()),
        }
    }
}

/// Explicit custom call batching rule.
#[derive(Copy, Clone, Serialize)]
enum WireCustomCallBatching {
    /// Tag of [`CustomCallBatching::Rejected`].
    Rejected,

    /// [`CustomCallBatching::Sequential`].
    Sequential {
        /// Unrolling of the sequential loop.
        unroll: WireCustomCallUnrolling,
    },

    /// Tag of [`CustomCallBatching::BroadcastAll`].
    BroadcastAll,

    /// Tag of [`CustomCallBatching::ExpandDimensions`].
    ExpandDimensions,

    /// Tag of [`CustomCallBatching::Vectorized`].
    Vectorized,
}

impl From<CustomCallBatching> for WireCustomCallBatching {
    fn from(value: CustomCallBatching) -> Self {
        match value {
            CustomCallBatching::Rejected => Self::Rejected,
            CustomCallBatching::Sequential { unroll } => Self::Sequential { unroll: unroll.into() },
            CustomCallBatching::BroadcastAll => Self::BroadcastAll,
            CustomCallBatching::ExpandDimensions => Self::ExpandDimensions,
            CustomCallBatching::Vectorized => Self::Vectorized,
        }
    }
}

/// Explicit unrolling of a sequentially batched custom call.
#[derive(Copy, Clone, Serialize)]
enum WireCustomCallUnrolling {
    /// Tag of [`CustomCallUnrolling::Disabled`].
    Disabled,

    /// [`CustomCallUnrolling::Factor`], holding its unrolling factor.
    Factor(usize),

    /// Tag of [`CustomCallUnrolling::Full`].
    Full,
}

impl From<CustomCallUnrolling> for WireCustomCallUnrolling {
    fn from(value: CustomCallUnrolling) -> Self {
        match value {
            CustomCallUnrolling::Disabled => Self::Disabled,
            CustomCallUnrolling::Factor(factor) => Self::Factor(factor),
            CustomCallUnrolling::Full => Self::Full,
        }
    }
}

/// Explicit forward rule of a composite with attached rules.
#[derive(Copy, Clone, Serialize)]
enum WireCompositeJvpRule {
    /// Tag of [`CompositeJvpRule::Absent`].
    Absent,

    /// Tag of [`CompositeJvpRule::Explicit`].
    Explicit,

    /// Tag of [`CompositeJvpRule::Derived`].
    Derived,
}

impl From<CompositeJvpRule> for WireCompositeJvpRule {
    fn from(value: CompositeJvpRule) -> Self {
        match value {
            CompositeJvpRule::Absent => Self::Absent,
            CompositeJvpRule::Explicit => Self::Explicit,
            CompositeJvpRule::Derived => Self::Derived,
        }
    }
}

/// Exact key of a named composite definition (refer to [`CompositeKey`](ryft_core::CompositeKey)).
#[derive(Serialize)]
struct WireCompositeKey {
    /// Name of the definition.
    name: String,

    /// Version of the definition.
    version: u32,

    /// Named attributes, in declaration order.
    attributes: Vec<(String, WireCompositeAttribute)>,
}

/// Exact named composite attribute, whose floating-point values are encoded by their IEEE bits.
#[derive(Serialize)]
enum WireCompositeAttribute {
    /// [`CompositeAttribute::Boolean`].
    Boolean(bool),

    /// [`CompositeAttribute::Integer`].
    Integer(i128),

    /// [`CompositeAttribute::Float`], holding its IEEE bits.
    Float(u64),

    /// [`CompositeAttribute::DataType`].
    DataType(WireDataType),

    /// [`CompositeAttribute::Sequence`].
    Sequence(Vec<WireCompositeAttribute>),

    /// [`CompositeAttribute::Optional`].
    Optional(Option<Box<WireCompositeAttribute>>),
}

impl From<&CompositeAttribute> for WireCompositeAttribute {
    fn from(value: &CompositeAttribute) -> Self {
        match value {
            CompositeAttribute::Boolean(value) => Self::Boolean(*value),
            CompositeAttribute::Integer(value) => Self::Integer(*value),
            CompositeAttribute::Float(value) => Self::Float(value.to_bits()),
            CompositeAttribute::DataType(value) => Self::DataType((*value).into()),
            CompositeAttribute::Sequence(values) => Self::Sequence(values.iter().map(Into::into).collect()),
            CompositeAttribute::Optional(value) => {
                Self::Optional(value.as_ref().map(|value| Box::new(value.as_ref().into())))
            }
        }
    }
}

/// Purpose of an encoding, which decides how the [`Encoder`] treats what the transport schema does not cover.
#[derive(Copy, Clone, PartialEq, Eq)]
enum EncodingMode {
    /// Encoding of a [`KernelSource`] document, which rejects every feature outside the transport schema and every
    /// definition that exceeds the record or literal budgets.
    Transport,

    /// Encoding of a [`KernelDefinition::semantic_key`], which projects out diagnostic provenance and names, ignores
    /// the budgets, and encodes the features outside the transport schema through key-only records.
    Key,
}

/// Encoding state of one document, holding the canonical identity table shared by the call metadata and every region.
struct Encoder {
    /// Purpose of this encoding.
    mode: EncodingMode,

    /// Identity table of the document, in registration order.
    variables: Vec<WireVariable>,

    /// Index of each registered identity in [`Encoder::variables`].
    identities: HashMap<DimensionVariable, usize>,

    /// Physical storage bytes of all literals encoded so far, which the literal budget bounds.
    literal_bytes: usize,
}

impl Encoder {
    /// Creates an encoder with an empty identity table.
    fn new(mode: EncodingMode) -> Self {
        Self { mode, variables: Vec::new(), identities: HashMap::new(), literal_bytes: 0 }
    }

    /// Encodes a complete document, registering the identities of the call metadata before those of the body.
    fn definition<Extension: KernelExtension>(
        mut self,
        definition: &KernelDefinition<Extension>,
    ) -> Result<WireDefinition, KernelSerializationError> {
        let call = self.call(definition.operation())?;
        let body = self.program(definition.body(), |operation| Ok(operation.clone()))?;
        self.validate_records(self.variables.len())?;
        Ok(WireDefinition { version: KERNEL_SOURCE_VERSION, variables: self.variables, call, body })
    }

    /// Applies the record budget in transport mode. Key mode encodes every valid definition.
    fn validate_records(&self, count: usize) -> Result<(), KernelSerializationError> {
        match self.mode {
            EncodingMode::Transport => validate_encoded_records(count),
            EncodingMode::Key => Ok(()),
        }
    }

    /// Registers a nominal identity once across the whole document. Key mode projects out its diagnostic name.
    fn identity(&mut self, variable: &DimensionVariable) -> usize {
        *self.identities.entry(variable.clone()).or_insert_with(|| {
            let index = self.variables.len();
            self.variables.push(WireVariable {
                name: match self.mode {
                    EncodingMode::Transport => variable.name().to_owned(),
                    EncodingMode::Key => String::new(),
                },
                lower: variable.bounds().lower(),
                upper: variable.bounds().upper(),
            });
            index
        })
    }

    /// Encodes a dimension without converting its nominal identity to diagnostic text.
    fn dimension(&mut self, dimension: &Dimension) -> WireDimension {
        match dimension {
            Dimension::Static(extent) => WireDimension::Static(*extent),
            Dimension::Dynamic(variable) => WireDimension::Dynamic(self.identity(variable)),
        }
    }

    /// Encodes local array metadata. Transport mode rejects distributed ownership and explicit layouts of
    /// dynamically shaped arrays, while key mode encodes both.
    fn array_type(&mut self, r#type: &ArrayType) -> Result<WireArrayType, KernelSerializationError> {
        let distribution = if r#type.sharding().is_some()
            || !r#type.unreduced_axes().is_empty()
            || !r#type.reduced_axes().is_empty()
        {
            match self.mode {
                EncodingMode::Transport => return Err(unsupported("distributed array type")),
                // TODO(eaplatanios): Replace this fallback with a wire record once the wire form covers distributed
                // array metadata.
                EncodingMode::Key => {
                    Some(format!("{:?}", (r#type.sharding(), r#type.unreduced_axes(), r#type.reduced_axes())))
                }
            }
        } else {
            None
        };
        if self.mode == EncodingMode::Transport
            && r#type.layout().is_some()
            && r#type.shape().dimensions().iter().any(|dimension| matches!(dimension, Dimension::Dynamic(_)))
        {
            return Err(unsupported("dynamically shaped array with explicit layout"));
        }
        Ok(WireArrayType {
            data_type: r#type.data_type().into(),
            shape: self.shape(r#type.shape()),
            layout: r#type.layout().map(WireLayout::encode),
            memory: r#type.memory().into(),
            distribution,
        })
    }

    /// Encodes a shape without converting the nominal identities of its dimensions to diagnostic text.
    fn shape(&mut self, shape: &Shape) -> Vec<WireDimension> {
        shape.dimensions().iter().map(|dimension| self.dimension(dimension)).collect()
    }

    /// Encodes a member type in the canonical array universe.
    fn r#type(&mut self, r#type: &ArrayIrType) -> Result<WireType, KernelSerializationError> {
        Ok(match r#type {
            ArrayIrType::Array(r#type) => WireType::Array(self.array_type(r#type)?),
            ArrayIrType::Dimension(r#type) => WireType::Dimension(self.dimension(r#type.dimension())),
            ArrayIrType::Reference(r#type) => WireType::Reference(self.array_type(r#type.referent())?),
        })
    }

    /// Preserves literal data bits, including signed zero and NaN payloads. Only transport mode applies the literal
    /// budget, and key mode rejects reference literals with the error of [`KernelDefinition::semantic_key`].
    fn value(&mut self, value: &ArrayIrValue<Array>) -> Result<WireValue, KernelSerializationError> {
        Ok(match value {
            ArrayIrValue::Array(value) => {
                if self.mode == EncodingMode::Transport {
                    let addressing = ArrayAddressing::new(value.r#type().into_owned())?;
                    self.literal_bytes = self
                        .literal_bytes
                        .checked_add(addressing.storage_byte_len())
                        .ok_or_else(|| unsupported("literal storage exceeds source allocation budget"))?;
                    if self.literal_bytes > MAXIMUM_LITERAL_BYTES
                        || addressing.element_count() > MAXIMUM_LITERAL_BYTES
                    {
                        return Err(unsupported("literal storage exceeds source allocation budget"));
                    }
                }
                WireValue::Array { r#type: self.array_type(&value.r#type())?, bytes: value.logical_bytes() }
            }
            ArrayIrValue::Dimension(value) => WireValue::Dimension { extent: value.extent() },
            ArrayIrValue::Reference(_) => {
                return Err(match self.mode {
                    EncodingMode::Transport => unsupported("reference literal"),
                    EncodingMode::Key => {
                        semantic_key_error("reference constants have no persistent kernel semantic identity")
                    }
                });
            }
        })
    }

    /// Encodes a flat canonical program while preserving arena order and shared edges. Key mode projects out the
    /// diagnostic provenance of every instruction, which it records as unknown.
    fn program<O: Operation<Type = ArrayIrType>, Extension: KernelExtension, F>(
        &mut self,
        program: &FlatProgram<EagerContext<ArrayIrValue<Array>, O>>,
        lift: F,
    ) -> Result<WireProgram, KernelSerializationError>
    where
        F: Fn(&O) -> Result<KernelOperation<Extension>, KernelSerializationError>,
    {
        self.validate_records(program.regions().len())?;
        let regions = program
            .regions()
            .iter()
            .map(|region| {
                self.validate_records(region.atoms().len())?;
                self.validate_records(region.instructions().len())?;
                let atoms = region
                    .atoms()
                    .iter()
                    .map(|atom| {
                        Ok(match atom {
                            Atom::Variable(r#type) => WireAtom::Variable(self.r#type(r#type)?),
                            Atom::Constant(value) => WireAtom::Constant(self.value(value)?),
                        })
                    })
                    .collect::<Result<_, KernelSerializationError>>()?;
                let instructions = region
                    .instructions()
                    .iter()
                    .map(|instruction| {
                        let operation = lift(instruction.operation())?;
                        let mut reference_transforms = Vec::new();
                        let descriptors =
                            validated_reference_access_descriptors(&operation, instruction.inputs().len())?;
                        for (input_index, descriptor) in descriptors.iter().enumerate() {
                            let Some(descriptor) = descriptor else { continue };
                            self.validate_records(descriptor.transforms().len())?;
                            reference_transforms.push(WireReferenceAccessDescriptor {
                                input_index,
                                transforms: descriptor
                                    .transforms()
                                    .iter()
                                    .map(WireReferenceTransform::encode)
                                    .collect::<Result<_, _>>()?,
                            });
                        }
                        let input_types = instruction
                            .inputs()
                            .iter()
                            .map(|id| region.atoms()[id.index()].r#type().into_owned())
                            .collect::<Vec<_>>();
                        Ok(WireInstruction {
                            operation: self.operation(&operation, &input_types)?,
                            reference_transforms,
                            inputs: instruction.inputs().iter().map(|id| id.index()).collect(),
                            outputs: instruction.outputs().iter().map(|id| id.index()).collect(),
                            regions: instruction.regions().iter().map(|id| id.index()).collect(),
                            provenance: match self.mode {
                                EncodingMode::Transport => WireProvenance::encode(instruction.provenance(), 0)?,
                                EncodingMode::Key => {
                                    WireProvenance { nodes: vec![WireProvenanceNode::Unknown], root: 0 }
                                }
                            },
                        })
                    })
                    .collect::<Result<_, KernelSerializationError>>()?;
                Ok(WireRegion {
                    atoms,
                    instructions,
                    inputs: region.input_ids().iter().map(|id| id.index()).collect(),
                    outputs: region.output_ids().iter().map(|id| id.index()).collect(),
                })
            })
            .collect::<Result<_, KernelSerializationError>>()?;
        Ok(WireProgram { regions, entry: program.entry().index() })
    }

    /// Encodes checked call metadata using the same identity table as its body. Key mode projects out the diagnostic
    /// names of the grid axes.
    fn call(&mut self, call: &KernelCallOperation) -> Result<WireCall, KernelSerializationError> {
        let grid = call
            .grid()
            .dimensions()
            .iter()
            .map(|dimension| WireGridDimension {
                extent: self.dimension(dimension.extent()),
                name: match self.mode {
                    EncodingMode::Transport => dimension.name().map(str::to_owned),
                    EncodingMode::Key => None,
                },
                execution: dimension.execution().into(),
            })
            .collect();
        let parameters = call
            .parameters()
            .iter()
            .map(|parameter| {
                Ok(WireParameter {
                    r#type: self.array_type(&parameter.r#type())?,
                    access: match parameter.access() {
                        KernelParameterAccess::ReadOnly => WireAccess::ReadOnly,
                        KernelParameterAccess::WriteOnly => WireAccess::WriteOnly,
                        KernelParameterAccess::ReadWrite => WireAccess::ReadWrite,
                    },
                    mapping: self.program(parameter.mapping().program(), |operation| {
                        Ok(KernelOperation::<NoKernelExtension>::Portable(operation.clone()))
                    })?,
                    block_shape: parameter.mapping().block_shape().to_vec(),
                    boundary_policy: parameter.mapping().boundary_policy().into(),
                })
            })
            .collect::<Result<_, KernelSerializationError>>()?;
        let coordinates = call.coordinate_types().iter().map(|r#type| self.dimension(r#type.dimension())).collect();
        let prefetch = call.prefetch_types().iter().map(|r#type| self.array_type(r#type)).collect::<Result<_, _>>()?;
        Ok(WireCall { grid, parameters, coordinates, prefetch })
    }

    /// Converts eligible payloads structurally, without parsing any diagnostic rendering. `input_types` holds the
    /// types of the instruction inputs, from which a few records are canonically reconstructed.
    fn operation<Extension: KernelExtension>(
        &mut self,
        operation: &KernelOperation<Extension>,
        input_types: &[ArrayIrType],
    ) -> Result<WireOperation, KernelSerializationError> {
        Ok(match operation {
            KernelOperation::Portable(operation) => self.portable_operation(operation, input_types)?,
            KernelOperation::Call(operation) => match self.mode {
                EncodingMode::Transport => return Err(unsupported("nested kernel call")),
                EncodingMode::Key => WireOperation::KeyOnly(WireKeyOnlyOperation::Call(self.call(operation)?)),
            },
            KernelOperation::Scratch(operation) => WireOperation::Scratch {
                referent: self.array_type(operation.referent())?,
                alignment: operation.alignment(),
            },
            KernelOperation::TileLoad(operation) => WireOperation::TileLoad {
                block_shape: operation.block_shape().to_vec(),
                boundary_policy: operation.boundary_policy().into(),
            },
            KernelOperation::AsyncCopy(_) => WireOperation::AsyncCopy,
            KernelOperation::Wait(_) => WireOperation::Wait,
            KernelOperation::MaskedLoad(_) => WireOperation::MaskedLoad,
            KernelOperation::MaskedStore(_) => WireOperation::MaskedStore,
            KernelOperation::MaskedSwap(_) => WireOperation::MaskedSwap,
            KernelOperation::WindowMask(_) => WireOperation::WindowMask,
            KernelOperation::Extension(operation) => match self.mode {
                EncodingMode::Transport => {
                    return Err(unsupported(format!("kernel extension `{}`", operation.name())));
                }
                EncodingMode::Key => WireOperation::KeyOnly(WireKeyOnlyOperation::Extension(
                    operation.semantic_key().map_err(ProgramError::from)?,
                )),
            },
        })
    }

    /// Converts a payload of the canonical array IR family, falling back to `portable` itself in key mode when no
    /// wire record covers it.
    fn portable_operation(
        &mut self,
        portable: &ArrayIrOperation<Array>,
        input_types: &[ArrayIrType],
    ) -> Result<WireOperation, KernelSerializationError> {
        Ok(match portable {
            ArrayIrOperation::Zero(operation) => WireOperation::MixedZero(self.array_type(operation.r#type())?),
            ArrayIrOperation::One(operation) => WireOperation::MixedOne(self.array_type(operation.r#type())?),
            ArrayIrOperation::Iota(operation) => WireOperation::MixedIota {
                r#type: self.array_type(operation.r#type())?,
                dimension: operation.dimension(),
            },
            ArrayIrOperation::Array(operation) => self.array_operation(portable, operation)?,
            ArrayIrOperation::Dimension(operation) => match operation {
                DimensionOperation::Constant(operation) => {
                    WireOperation::DimensionConstant(self.value(&operation.value().clone().into())?)
                }
                DimensionOperation::Max(operation) => self.dimension_arithmetic(
                    operation,
                    DimensionMaxOperation::new(operation.left_type(), operation.right_type()),
                    |left, right| WireOperation::DimensionMax { left, right },
                )?,
                DimensionOperation::Min(operation) => self.dimension_arithmetic(
                    operation,
                    DimensionMinOperation::new(operation.left_type(), operation.right_type()),
                    |left, right| WireOperation::DimensionMin { left, right },
                )?,
                DimensionOperation::Add(operation) => self.dimension_arithmetic(
                    operation,
                    DimensionAddOperation::new(operation.left_type(), operation.right_type()),
                    |left, right| WireOperation::DimensionAdd { left, right },
                )?,
                DimensionOperation::Sub(operation) => self.dimension_arithmetic(
                    operation,
                    DimensionSubOperation::new(operation.left_type(), operation.right_type()),
                    |left, right| WireOperation::DimensionSub { left, right },
                )?,
                DimensionOperation::SaturatingSub(_) => WireOperation::SaturatingSubDimension,
                DimensionOperation::Mul(operation) => self.dimension_arithmetic(
                    operation,
                    DimensionMulOperation::new(operation.left_type(), operation.right_type()),
                    |left, right| WireOperation::DimensionMul { left, right },
                )?,
                DimensionOperation::Div(operation) => self.dimension_arithmetic(
                    operation,
                    DimensionDivOperation::new(operation.left_type(), operation.right_type()),
                    |left, right| WireOperation::DimensionDiv { left, right },
                )?,
                DimensionOperation::Rem(operation) => self.dimension_arithmetic(
                    operation,
                    DimensionRemOperation::new(operation.left_type(), operation.right_type()),
                    |left, right| WireOperation::DimensionRem { left, right },
                )?,
                DimensionOperation::Pow(operation) => self.dimension_arithmetic(
                    operation,
                    DimensionPowOperation::new(operation.left_type(), operation.right_type()),
                    |left, right| WireOperation::DimensionPow { left, right },
                )?,
            },
            ArrayIrOperation::DimensionSize(operation) => {
                let preserved = match input_types.first() {
                    Some(ArrayIrType::Array(r#type)) => DimensionSizeOperation::new(r#type, operation.axis())
                        .is_ok_and(|reconstructed| reconstructed == *operation),
                    _ => false,
                };
                self.preserved(portable, preserved, WireOperation::DimensionSize { axis: operation.axis() })?
            }
            ArrayIrOperation::DimensionFromScalar(operation) => {
                WireOperation::DimensionFromScalar(self.dimension(operation.output_type().dimension()))
            }
            ArrayIrOperation::DimensionToScalar(_) => WireOperation::DimensionToScalar,
            ArrayIrOperation::ReferenceNew(_) => WireOperation::ReferenceNew,
            ArrayIrOperation::ReferenceRead(_) => WireOperation::ReferenceRead,
            ArrayIrOperation::ReferenceWrite(_) => WireOperation::ReferenceWrite,
            ArrayIrOperation::ReferenceSwap(_) => WireOperation::ReferenceSwap,
            ArrayIrOperation::ReferenceAddUpdate(_) => WireOperation::ReferenceAddUpdate,
            ArrayIrOperation::ReferenceAtomicAddUpdate(_) => WireOperation::ReferenceAtomicAddUpdate,
            ArrayIrOperation::ReferenceFreeze(_) => WireOperation::ReferenceFreeze,
            ArrayIrOperation::ParallelAllGather(_)
            | ArrayIrOperation::ParallelSumScatter(_)
            | ArrayIrOperation::ParallelAllToAll(_)
            | ArrayIrOperation::ParallelRaggedAllToAll(_) => {
                // TODO(eaplatanios): Replace this fallback with a wire record once the wire form covers collective
                // operations.
                self.fallback(portable, format!("`{}`", portable.name()))?
            }
            ArrayIrOperation::Compare(operation) => WireOperation::DimensionCompare(operation.direction().into()),
            ArrayIrOperation::Reshape(operation) => {
                if operation.output_sharding().is_some() {
                    // TODO(eaplatanios): Replace this fallback with a wire record once the wire form covers requested
                    // output shardings.
                    return self.fallback(portable, format!("`{}` output sharding", portable.name()));
                }
                let preserved = DynamicReshapeOperation::new()
                    .with_input_types(input_types)
                    .is_ok_and(|reconstructed| reconstructed == *operation);
                self.preserved(portable, preserved, WireOperation::MixedReshape)?
            }
            ArrayIrOperation::Broadcast(operation) => {
                if operation.output_sharding().is_some() {
                    // TODO(eaplatanios): Replace this fallback with a wire record once the wire form covers requested
                    // output shardings.
                    return self.fallback(portable, format!("`{}` output sharding", portable.name()));
                }
                WireOperation::MixedBroadcast {
                    output_axes: operation.output_axes().to_vec(),
                    output_layout: operation.output_layout().map(WireLayout::encode),
                }
            }
            ArrayIrOperation::Pad(operation) => {
                let low = operation.edge_padding_low().to_vec();
                let high = operation.edge_padding_high().to_vec();
                let interior = operation.interior_padding().to_vec();
                let preserved = PadOperation::<ArrayIrType>::new(low.clone(), high.clone(), interior.clone())
                    .ok()
                    .and_then(|reconstructed| reconstructed.with_input_types(input_types).ok())
                    .is_some_and(|reconstructed| reconstructed == *operation);
                self.preserved(portable, preserved, WireOperation::MixedPad { low, high, interior })?
            }
            ArrayIrOperation::Concatenate(operation) => {
                let preserved = ConcatenateOperation::<ArrayIrType>::new(operation.axis(), input_types)
                    .is_ok_and(|reconstructed| reconstructed == *operation);
                self.preserved(portable, preserved, WireOperation::MixedConcatenate { axis: operation.axis() })?
            }
            ArrayIrOperation::DynamicSlice(_) => {
                // TODO(eaplatanios): Replace this fallback with a wire record once the wire form covers array IR
                // dynamic slices.
                self.fallback(portable, format!("`{}`", portable.name()))?
            }
            ArrayIrOperation::ShardMap(_) => {
                // TODO(eaplatanios): Replace this fallback with a wire record once the wire form covers shard maps.
                self.fallback(portable, format!("`{}`", portable.name()))?
            }
            ArrayIrOperation::Condition(_) => WireOperation::Condition,
            ArrayIrOperation::While(operation) => WireOperation::While { bound: operation.iteration_bound() },
            ArrayIrOperation::Scan(_) => {
                // TODO(eaplatanios): Replace this fallback with a wire record once the wire form covers scans.
                self.fallback(portable, format!("`{}`", portable.name()))?
            }
            ArrayIrOperation::Rematerialize(_) => return Err(self.rematerialize_error()),
            ArrayIrOperation::LinearCall(_)
            | ArrayIrOperation::RngBitGenerator(_)
            | ArrayIrOperation::Assert(_)
            | ArrayIrOperation::Guard(_) => {
                // TODO(eaplatanios): Replace this fallback with a wire record once the wire form covers linear calls,
                // random bit generation, assertions, and guards.
                self.fallback(portable, format!("`{}`", portable.name()))?
            }
            ArrayIrOperation::Cast(operation) => WireOperation::MixedCast {
                output_type: self.r#type(operation.output_type())?,
                input_types: operation
                    .input_types()
                    .iter()
                    .map(|r#type| self.r#type(r#type))
                    .collect::<Result<_, _>>()?,
                checked: operation.is_checked(),
            },
            ArrayIrOperation::CustomCall(operation) => self.custom_call(portable, operation)?,
            ArrayIrOperation::Composite(operation) => self.composite(portable, operation)?,
            ArrayIrOperation::CompositeTranspose(operation) => {
                return Err(self.composite_transpose_error(operation.rules().is_some()));
            }
            ArrayIrOperation::LiftedComposite(operation) => self.composite(portable, operation)?,
            ArrayIrOperation::LiftedCompositeTranspose(_) => return Err(self.composite_transpose_error(true)),
        })
    }

    /// Converts a payload of the array member family. `portable` is the array IR operation that wraps `operation`.
    fn array_operation(
        &mut self,
        portable: &ArrayIrOperation<Array>,
        operation: &ArrayOperation<Array>,
    ) -> Result<WireOperation, KernelSerializationError> {
        Ok(match operation {
            ArrayOperation::Zero(operation) => WireOperation::Zero(self.array_type(operation.r#type())?),
            ArrayOperation::ZeroLike(_) => WireOperation::ZeroLike,
            ArrayOperation::One(operation) => WireOperation::One(self.array_type(operation.r#type())?),
            ArrayOperation::OneLike(_) => WireOperation::OneLike,
            ArrayOperation::Constant(operation) => {
                WireOperation::ArrayConstant(self.value(&operation.value().clone().into())?)
            }
            ArrayOperation::Iota(operation) => WireOperation::Iota {
                r#type: self.array_type(operation.r#type())?,
                dimension: operation.dimension(),
            },
            ArrayOperation::FillLike(operation) => {
                WireOperation::FillLike(self.value(&operation.value().clone().into())?)
            }
            ArrayOperation::IotaLike(operation) => WireOperation::IotaLike {
                dimension: operation.dimension(),
                data_type: operation.data_type().into(),
            },
            ArrayOperation::Min(_) => WireOperation::Min,
            ArrayOperation::Max(_) => WireOperation::Max,
            ArrayOperation::Clamp(_) => WireOperation::Clamp,
            ArrayOperation::Sign(_) => WireOperation::Sign,
            ArrayOperation::Neg(_) => WireOperation::Neg,
            ArrayOperation::Abs(_) => WireOperation::Abs,
            ArrayOperation::Add(_) => WireOperation::Add,
            ArrayOperation::Sub(_) => WireOperation::Sub,
            ArrayOperation::SaturatingSub(_) => WireOperation::SaturatingSub,
            ArrayOperation::Mul(_) => WireOperation::Mul,
            ArrayOperation::Div(_) => WireOperation::Div,
            ArrayOperation::Rem(_) => WireOperation::Rem,
            ArrayOperation::Pow(_) => WireOperation::Pow,
            ArrayOperation::Sqrt(operation) => {
                WireAccuracy::encode(WireOperation::Sqrt, WireAccurateFunction::Sqrt, operation.accuracy())
            }
            ArrayOperation::Rsqrt(operation) => {
                WireAccuracy::encode(WireOperation::Rsqrt, WireAccurateFunction::Rsqrt, operation.accuracy())
            }
            ArrayOperation::Sin(operation) => {
                WireAccuracy::encode(WireOperation::Sin, WireAccurateFunction::Sin, operation.accuracy())
            }
            ArrayOperation::Cos(operation) => {
                WireAccuracy::encode(WireOperation::Cos, WireAccurateFunction::Cos, operation.accuracy())
            }
            ArrayOperation::Tan(operation) => {
                WireAccuracy::encode(WireOperation::Tan, WireAccurateFunction::Tan, operation.accuracy())
            }
            ArrayOperation::Tanh(operation) => {
                WireAccuracy::encode(WireOperation::Tanh, WireAccurateFunction::Tanh, operation.accuracy())
            }
            ArrayOperation::Atan2(_) => WireOperation::Atan2,
            ArrayOperation::Exp(operation) => {
                WireAccuracy::encode(WireOperation::Exp, WireAccurateFunction::Exp, operation.accuracy())
            }
            ArrayOperation::ExpM1(operation) => {
                WireAccuracy::encode(WireOperation::ExpM1, WireAccurateFunction::ExpM1, operation.accuracy())
            }
            ArrayOperation::Log(operation) => {
                WireAccuracy::encode(WireOperation::Log, WireAccurateFunction::Log, operation.accuracy())
            }
            ArrayOperation::Ln1p(operation) => {
                WireAccuracy::encode(WireOperation::Ln1p, WireAccurateFunction::Ln1p, operation.accuracy())
            }
            ArrayOperation::LogAddExp(_) => WireOperation::LogAddExp,
            ArrayOperation::Logistic(operation) => {
                WireAccuracy::encode(WireOperation::Logistic, WireAccurateFunction::Logistic, operation.accuracy())
            }
            ArrayOperation::Erf(_) => WireOperation::Erf,
            ArrayOperation::Floor(_) => WireOperation::Floor,
            ArrayOperation::Ceil(_) => WireOperation::Ceil,
            ArrayOperation::Round(_) => WireOperation::Round,
            ArrayOperation::Not(_) => WireOperation::Not,
            ArrayOperation::And(_) => WireOperation::And,
            ArrayOperation::Or(_) => WireOperation::Or,
            ArrayOperation::Xor(_) => WireOperation::Xor,
            ArrayOperation::Complex(_) => WireOperation::Complex,
            ArrayOperation::Real(_) => WireOperation::Real,
            ArrayOperation::Imaginary(_) => WireOperation::Imaginary,
            ArrayOperation::Conjugate(_) => WireOperation::Conjugate,
            ArrayOperation::Reduce(operation) => {
                if operation.output_sharding().is_some() {
                    // TODO(eaplatanios): Replace this fallback with a wire record once the wire form covers requested
                    // output shardings.
                    return self.fallback(portable, format!("`{}` output sharding", portable.name()));
                }
                WireOperation::Reduce { axes: operation.axes().to_vec(), kind: operation.kind().into() }
            }
            ArrayOperation::ArgMax(operation) => WireOperation::ArgMax {
                axis: operation.axis(),
                index_data_type: operation.index_data_type().into(),
            },
            ArrayOperation::ArgMin(operation) => WireOperation::ArgMin {
                axis: operation.axis(),
                index_data_type: operation.index_data_type().into(),
            },
            ArrayOperation::Cumulative(operation) => WireOperation::Cumulative {
                axis: operation.axis(),
                kind: operation.kind().into(),
                reverse: operation.reverse(),
            },
            ArrayOperation::Sort(operation) => WireOperation::Sort {
                axis: operation.axis(),
                key_count: operation.key_count(),
                direction: operation.direction().into(),
                ordering: operation.ordering().into(),
            },
            ArrayOperation::Dot(operation) => {
                if operation.output_sharding().is_some() {
                    // TODO(eaplatanios): Replace this fallback with a wire record once the wire form covers requested
                    // output shardings.
                    return self.fallback(portable, format!("`{}` output sharding", portable.name()));
                }
                let dimensions = operation.dimensions();
                let dimensions = [
                    dimensions.lhs_contracting_dimensions().to_vec(),
                    dimensions.rhs_contracting_dimensions().to_vec(),
                    dimensions.lhs_batching_dimensions().to_vec(),
                    dimensions.rhs_batching_dimensions().to_vec(),
                ];
                let accumulation = operation.accumulation_type().map(Into::into);
                if operation.precision().is_default() {
                    WireOperation::Dot { dimensions, accumulation }
                } else {
                    WireOperation::DotWithPrecision {
                        dimensions,
                        accumulation,
                        precision: operation.precision().into(),
                    }
                }
            }
            ArrayOperation::RaggedDot(_) | ArrayOperation::ScaledDot(_) => {
                // TODO(eaplatanios): Replace this fallback with a wire record once the wire form covers ragged and
                // scaled dots.
                self.fallback(portable, format!("`{}`", portable.name()))?
            }
            ArrayOperation::AxisIndex(_)
            | ArrayOperation::ParallelReduce(_)
            | ArrayOperation::ParallelVary(_)
            | ArrayOperation::ParallelCast(_)
            | ArrayOperation::ParallelAllGather(_)
            | ArrayOperation::ParallelSumScatter(_)
            | ArrayOperation::ParallelPermute(_)
            | ArrayOperation::ParallelBroadcast(_)
            | ArrayOperation::ParallelAllToAll(_)
            | ArrayOperation::ParallelRaggedAllToAll(_) => {
                // TODO(eaplatanios): Replace this fallback with a wire record once the wire form covers collective
                // operations.
                self.fallback(portable, format!("`{}`", portable.name()))?
            }
            ArrayOperation::Compare(operation) => WireOperation::Compare(operation.direction().into()),
            ArrayOperation::Select(_) => WireOperation::Select,
            ArrayOperation::Transpose(operation) => {
                WireOperation::Transpose(operation.permutation().iter().map(|axis| axis.value()).collect())
            }
            ArrayOperation::Reverse(operation) => {
                WireOperation::Reverse(operation.axes().iter().map(|axis| axis.value()).collect())
            }
            ArrayOperation::Reshape(operation) => {
                if operation.output_sharding().is_some() {
                    // TODO(eaplatanios): Replace this fallback with a wire record once the wire form covers requested
                    // output shardings.
                    return self.fallback(portable, format!("`{}` output sharding", portable.name()));
                }
                WireOperation::Reshape {
                    input_shape: self.shape(operation.input_shape()),
                    output_shape: self.shape(operation.output_shape()),
                }
            }
            ArrayOperation::Broadcast(operation) => WireOperation::Broadcast {
                output_type: self.array_type(operation.output_type())?,
                output_axes: operation.output_axes().to_vec(),
            },
            ArrayOperation::BroadcastLike(operation) => WireOperation::BroadcastLike(operation.output_axes().to_vec()),
            ArrayOperation::Pad(operation) => WireOperation::Pad {
                low: operation.edge_padding_low().to_vec(),
                high: operation.edge_padding_high().to_vec(),
                interior: operation.interior_padding().to_vec(),
            },
            ArrayOperation::Concatenate(operation) => WireOperation::Concatenate { axis: operation.axis() },
            ArrayOperation::Gather(_) | ArrayOperation::Scatter(_) => {
                // TODO(eaplatanios): Replace this fallback with a wire record once the wire form covers gathers and
                // scatters.
                self.fallback(portable, format!("`{}`", portable.name()))?
            }
            ArrayOperation::Slice(operation) => WireOperation::Slice {
                start_indices: operation.start_indices().to_vec(),
                limits: operation.limits().iter().map(|limit| self.dimension(limit)).collect(),
                strides: operation.strides().to_vec(),
            },
            ArrayOperation::UpdateSlice(operation) => WireOperation::UpdateSlice(operation.start_indices().to_vec()),
            ArrayOperation::DynamicSlice(operation) => {
                let preserved = DynamicSliceOperation::new(operation.sizes().to_vec())
                    .with_allow_negative_indices(operation.allows_negative_indices())
                    == *operation;
                let record = WireOperation::DynamicSlice {
                    sizes: operation.sizes().to_vec(),
                    allow_negative_indices: operation.allows_negative_indices(),
                };
                self.preserved(portable, preserved, record)?
            }
            ArrayOperation::DynamicUpdateSlice(operation) => {
                WireOperation::DynamicUpdateSlice { allow_negative_indices: operation.allows_negative_indices() }
            }
            ArrayOperation::ConvertElementType(operation) => WireOperation::ConvertElementType {
                data_type: operation.data_type().into(),
                bitcast: operation.bitcast(),
            },
            ArrayOperation::ReducePrecision(operation) => WireOperation::ReducePrecision {
                exponent_bits: operation.exponent_bits(),
                mantissa_bits: operation.mantissa_bits(),
            },
            ArrayOperation::TransferToMemory(operation) => {
                WireOperation::TransferToMemory(operation.destination().into())
            }
            ArrayOperation::Reshard(_) | ArrayOperation::ConstrainSharding(_) => {
                // TODO(eaplatanios): Replace this fallback with a wire record once the wire form covers sharding
                // changes and constraints.
                self.fallback(portable, format!("`{}`", portable.name()))?
            }
            ArrayOperation::Condition(_) => WireOperation::ArrayCondition,
            ArrayOperation::While(operation) => WireOperation::ArrayWhile { bound: operation.iteration_bound() },
            ArrayOperation::Scan(_) => {
                // TODO(eaplatanios): Replace this fallback with a wire record once the wire form covers scans.
                self.fallback(portable, format!("`{}`", portable.name()))?
            }
            ArrayOperation::StopGradient(_) => WireOperation::StopGradient,
            ArrayOperation::Tag(operation) => WireOperation::Tag(operation.key().to_owned()),
            ArrayOperation::Rematerialize(_) => return Err(self.rematerialize_error()),
            ArrayOperation::LinearCall(_)
            | ArrayOperation::RngBitGenerator(_)
            | ArrayOperation::Dropout(_)
            | ArrayOperation::Print(_)
            | ArrayOperation::Assert(_)
            | ArrayOperation::Guard(_) => {
                // TODO(eaplatanios): Replace this fallback with a wire record once the wire form covers linear calls,
                // random bit generation, dropout, printing, assertions, and guards.
                self.fallback(portable, format!("`{}`", portable.name()))?
            }
            ArrayOperation::Cast(operation) => WireOperation::Cast {
                output_type: self.array_type(operation.output_type())?,
                input_types: operation
                    .input_types()
                    .iter()
                    .map(|r#type| self.array_type(r#type))
                    .collect::<Result<_, _>>()?,
                checked: operation.is_checked(),
            },
            ArrayOperation::CustomCall(operation) => self.custom_call(portable, operation)?,
            ArrayOperation::Composite(operation) => self.composite(portable, operation)?,
            ArrayOperation::CompositeTranspose(operation) => {
                return Err(self.composite_transpose_error(operation.rules().is_some()));
            }
        })
    }

    /// Encodes binary dimension arithmetic, whose record holds only its declared input dimensions. When canonical
    /// reconstruction from those dimensions (i.e., `reconstructed`) does not preserve its cached inference metadata,
    /// transport mode rejects it and key mode records that metadata explicitly, without its diagnostic output name.
    fn dimension_arithmetic<D: ArithmeticDimensionOperation + PartialEq>(
        &mut self,
        operation: &D,
        reconstructed: Result<D, DimensionError>,
        record: fn(WireDimension, WireDimension) -> WireOperation,
    ) -> Result<WireOperation, KernelSerializationError> {
        let left = self.dimension(operation.left_type().dimension());
        let right = self.dimension(operation.right_type().dimension());

        // Refinement may retain cached assertion effects and diagnostic names from wider inputs.
        if reconstructed.is_ok_and(|reconstructed| reconstructed == *operation) {
            return Ok(record(left, right));
        }
        match self.mode {
            EncodingMode::Transport => Err(unsupported(format!("`{}` cached inference metadata", operation.name()))),
            EncodingMode::Key => {
                let bounds = operation.output_bounds();
                Ok(WireOperation::KeyOnly(WireKeyOnlyOperation::DimensionArithmetic {
                    operation: operation.name(),
                    left,
                    right,
                    output_lower: bounds.lower(),
                    output_upper: bounds.upper(),
                    effects: operation.effects().classes().into_iter().map(Into::into).collect(),
                }))
            }
        }
    }

    /// Encodes a custom call through an exact key-only record in key mode, and rejects it in transport mode.
    fn custom_call(
        &mut self,
        portable: &ArrayIrOperation<Array>,
        operation: &CustomCallOperation,
    ) -> Result<WireOperation, KernelSerializationError> {
        if self.mode == EncodingMode::Transport {
            return Err(unsupported(format!("`{}`", portable.name())));
        }
        Ok(WireOperation::KeyOnly(WireKeyOnlyOperation::CustomCall {
            target: operation.target_name().to_owned(),
            attributes: operation.attributes().iter().map(|(name, value)| (name.clone(), value.into())).collect(),
            input_layouts: operation
                .input_layouts()
                .iter()
                .map(|layout| layout.as_ref().map(WireLayout::encode))
                .collect(),
            input_output_aliases: operation
                .input_output_aliases()
                .iter()
                .map(|alias| (alias.input_index(), alias.output_index()))
                .collect(),
            output_types: operation
                .output_types()
                .iter()
                .map(|r#type| self.array_type(r#type))
                .collect::<Result<_, _>>()?,
            effect_class: operation.effect_class().map(Into::into),
            batching: operation.batching().into(),
        }))
    }

    /// Encodes a `composite` call through a key-only record in key mode, and rejects it in transport mode. A call
    /// with attached rules is identified by its regions and rule flags. A call with retained rules is identified
    /// across processes only by the key of a named definition (refer to [`NamedComposite`](ryft_core::NamedComposite)),
    /// so calls of other definitions are rejected, because the identity of their rules is local to the process.
    fn composite<V: Typed + Parameter, O, S: CustomRuleSource<V, O>>(
        &mut self,
        portable: &ArrayIrOperation<Array>,
        operation: &CompositeOperation<V, O, S>,
    ) -> Result<WireOperation, KernelSerializationError>
    where
        V::Type: DifferentiableType + Eq + Hash,
    {
        if self.mode == EncodingMode::Transport {
            return Err(unsupported(format!("`{}`", portable.name())));
        }
        let Some(rules) = operation.rules() else {
            return Ok(WireOperation::KeyOnly(WireKeyOnlyOperation::AttachedComposite {
                non_differentiated_count: operation.non_differentiated_count(),
                jvp_rule: operation.jvp_rule().into(),
                has_vjp_rule: operation.has_vjp_rule(),
            }));
        };
        let Some(key) = rules.key() else {
            return Err(retained_composite_error(COMPOSITE_OPERATION_NAME));
        };
        let key = WireCompositeKey {
            name: key.name().to_owned(),
            version: key.version(),
            attributes: key.attributes().iter().map(|(name, value)| (name.to_string(), value.into())).collect(),
        };

        // TODO(eaplatanios): Replace this fallback with a wire record once the wire form covers the retained rule
        // state of named composites.
        Ok(WireOperation::KeyOnly(WireKeyOnlyOperation::NamedComposite { operation: self.debug(portable)?, key }))
    }

    /// Returns the error that rejects `rematerialize` operations, whose residual policies have an identity that is
    /// local to the process.
    fn rematerialize_error(&self) -> KernelSerializationError {
        match self.mode {
            EncodingMode::Transport => unsupported(format!("`{REMATERIALIZE_OPERATION_NAME}`")),
            EncodingMode::Key => semantic_key_error(format!(
                "`{REMATERIALIZE_OPERATION_NAME}` operations have no kernel semantic descriptor encoding, because the \
                 identity of their residual policies is local to the process",
            )),
        }
    }

    /// Returns the error that rejects `composite_transpose` carriers, which are unresolved reverse mode obligations
    /// whose `retained` rules additionally have an identity that is local to the process.
    fn composite_transpose_error(&self, retained: bool) -> KernelSerializationError {
        match self.mode {
            EncodingMode::Transport => unsupported(format!("`{COMPOSITE_TRANSPOSE_OPERATION_NAME}`")),
            EncodingMode::Key if retained => retained_composite_error(COMPOSITE_TRANSPOSE_OPERATION_NAME),
            EncodingMode::Key => semantic_key_error(format!(
                "`{COMPOSITE_TRANSPOSE_OPERATION_NAME}` operations have no kernel semantic descriptor encoding",
            )),
        }
    }

    /// Returns `record` when canonical reconstruction preserves the operation exactly (i.e., when `preserved` is
    /// `true`), and otherwise treats the cached inference metadata of `portable` as a feature without a wire record.
    fn preserved(
        &self,
        portable: &ArrayIrOperation<Array>,
        preserved: bool,
        record: WireOperation,
    ) -> Result<WireOperation, KernelSerializationError> {
        if preserved {
            return Ok(record);
        }

        // TODO(eaplatanios): Replace this fallback with a wire record once the wire form covers cached inference
        // metadata that canonical reconstruction does not preserve.
        self.fallback(portable, format!("`{}` cached inference metadata", portable.name()))
    }

    /// Rejects an operation without a wire record in transport mode as the unsupported `feature`, and records the
    /// transitional [`Debug`](std::fmt::Debug) rendering of `portable` in key mode. Callers mark each use with a
    /// `TODO` naming what the wire form must cover to replace it.
    fn fallback(
        &self,
        portable: &ArrayIrOperation<Array>,
        feature: String,
    ) -> Result<WireOperation, KernelSerializationError> {
        match self.mode {
            EncodingMode::Transport => Err(unsupported(feature)),
            EncodingMode::Key => Ok(WireOperation::KeyOnly(WireKeyOnlyOperation::Debug(self.debug(portable)?))),
        }
    }

    /// Renders the [`Debug`](std::fmt::Debug) representation of `portable` after renaming every identity registered
    /// so far to a canonical name derived from its index in the identity table, so that the rendering does not depend
    /// on nominal names. Every identity of the input and output types of an instruction is registered before its
    /// operation, and only an identity that appears solely in the payload keeps its nominal name.
    fn debug(&self, portable: &ArrayIrOperation<Array>) -> Result<String, KernelSerializationError> {
        // The renaming is a set of independent replacements, so the iteration order of the identity table is
        // irrelevant.
        let mut renaming = TypeIdentityRenaming::new();
        for (variable, index) in &self.identities {
            renaming
                .insert(variable.clone(), DimensionVariable::new(format!("dimension_{index}"), variable.bounds()))
                .map_err(ProgramError::from)?;
        }
        Ok(format!("{:?}", portable.rename_type_identities(&renaming).map_err(ProgramError::from)?))
    }
}

/// Decoding state of one document, holding the canonical identities already reconstructed from checked bounds.
struct Decoder {
    /// Reconstructed identity table, which [`WireDimension::Dynamic`] indexes.
    variables: Vec<DimensionVariable>,

    /// Physical storage bytes of all literals decoded so far, which the literal budget bounds. It is a [`Cell`]
    /// because the decoding functions borrow the decoder immutably.
    literal_bytes: Cell<usize>,
}

impl Decoder {
    /// Starts the physical literal allocation budget of one document.
    fn new(variables: Vec<DimensionVariable>) -> Self {
        Self { variables, literal_bytes: Cell::new(0) }
    }

    /// Resolves a checked identity index without manufacturing a second nominal variable.
    fn identity(&self, index: usize) -> Result<DimensionType, KernelSerializationError> {
        self.variables
            .get(index)
            .cloned()
            .map(DimensionType::from)
            .ok_or_else(|| invalid(format!("dimension identity {index} is out of bounds")))
    }

    /// Validates static extents before passing them to type constructors. A symbolic dimension keeps its identity even
    /// when its bounds admit a single extent, so the decoded dimension is exactly the encoded one.
    fn dimension(&self, dimension: WireDimension) -> Result<Dimension, KernelSerializationError> {
        Ok(self.dimension_type(dimension)?.dimension().clone())
    }

    /// Reconstructs a first-class dimension type, keeping a symbolic type's identity even when its bounds admit a
    /// single extent.
    fn dimension_type(&self, dimension: WireDimension) -> Result<DimensionType, KernelSerializationError> {
        match dimension {
            WireDimension::Static(extent) if extent <= MAX_DIMENSION_EXTENT => {
                Ok(DimensionType::from(Dimension::Static(extent)))
            }
            WireDimension::Static(extent) => Err(invalid(format!("dimension extent {extent} exceeds backend width"))),
            WireDimension::Dynamic(index) => self.identity(index),
        }
    }

    /// Reconstructs the dimensions of a shape, keeping the identities of its symbolic dimensions.
    fn shape(&self, wire: Vec<WireDimension>) -> Result<Vec<Dimension>, KernelSerializationError> {
        validate_decoded_records(wire.len())?;
        wire.into_iter().map(|dimension| self.dimension(dimension)).collect()
    }

    /// Reconstructs canonical local type metadata. Literals subsequently check their physical allocation size.
    fn array_type(&self, wire: WireArrayType) -> Result<ArrayType, KernelSerializationError> {
        if wire.distribution.is_some() {
            return Err(invalid("array type has key-only distributed metadata"));
        }
        let shape = Shape::new(self.shape(wire.shape)?);
        let layout = wire.layout.map(WireLayout::decode);
        let r#type =
            ArrayType::new(wire.data_type.into(), shape).with_layout(layout).with_memory(Memory::from(wire.memory));
        if r#type.shape().dimensions().iter().all(|dimension| matches!(dimension, Dimension::Static(_))) {
            ArrayAddressing::new(r#type.clone())?;
        } else if r#type.layout().is_some() {
            return Err(unsupported("dynamically shaped array with explicit layout"));
        }
        Ok(r#type)
    }

    /// Reconstructs a canonical member type.
    fn r#type(&self, wire: WireType) -> Result<ArrayIrType, KernelSerializationError> {
        Ok(match wire {
            WireType::Array(r#type) => self.array_type(r#type)?.into(),
            WireType::Dimension(dimension) => self.dimension_type(dimension)?.into(),
            WireType::Reference(r#type) => ReferenceType::new(self.array_type(r#type)?).into(),
        })
    }

    /// Checks exact logical byte count and bounded physical storage before allocating a literal array.
    fn value(&self, wire: WireValue) -> Result<ArrayIrValue<Array>, KernelSerializationError> {
        Ok(match wire {
            WireValue::Array { r#type, bytes } => {
                let r#type = self.array_type(r#type)?;
                let addressing = ArrayAddressing::new(r#type.clone())?;
                if addressing.logical_byte_len() != bytes.len() {
                    return Err(invalid(format!(
                        "literal expects {} bytes but received {}",
                        addressing.logical_byte_len(),
                        bytes.len(),
                    )));
                }
                let allocated = self
                    .literal_bytes
                    .get()
                    .checked_add(addressing.storage_byte_len())
                    .ok_or_else(|| invalid("literal storage size overflows"))?;
                if allocated > MAXIMUM_LITERAL_BYTES || addressing.element_count() > MAXIMUM_LITERAL_BYTES {
                    return Err(invalid(format!("literal storage exceeds {MAXIMUM_LITERAL_BYTES} bytes")));
                }
                self.literal_bytes.set(allocated);
                Array::from_logical_bytes(r#type, &bytes)?.into()
            }
            WireValue::Dimension { extent } => DimensionValue::constant(extent).map_err(ProgramError::from)?.into(),
        })
    }

    /// Reconstructs only explicitly eligible operation payloads through their canonical constructors. `input_types`
    /// holds the types of the instruction inputs, from which a few records are canonically reconstructed.
    fn operation(
        &self,
        wire: WireOperation,
        input_types: &[ArrayIrType],
    ) -> Result<KernelOperation, KernelSerializationError> {
        Ok(match wire {
            WireOperation::ArrayConstant(value) => {
                let ArrayIrValue::Array(value) = self.value(value)? else {
                    return Err(invalid("array constant has a dimension payload"));
                };
                ArrayIrOperation::Array(ArrayOperation::Constant(ConstantOperation::new(value))).into()
            }
            WireOperation::DimensionConstant(value) => {
                let ArrayIrValue::Dimension(value) = self.value(value)? else {
                    return Err(invalid("dimension constant has an array payload"));
                };
                ArrayIrOperation::Dimension(DimensionOperation::Constant(ConstantOperation::new(value))).into()
            }
            WireOperation::Zero(r#type) => {
                ArrayIrOperation::Array(ArrayOperation::Zero(ZeroOperation::new(self.array_type(r#type)?))).into()
            }
            WireOperation::MixedZero(r#type) => {
                ArrayIrOperation::Zero(ZeroOperation::new(self.array_type(r#type)?)).into()
            }
            WireOperation::Compare(direction) => {
                ArrayIrOperation::Array(ArrayOperation::Compare(CompareOperation::new(direction.into()))).into()
            }
            WireOperation::DimensionCompare(direction) => {
                ArrayIrOperation::Compare(CompareOperation::new(direction.into())).into()
            }
            WireOperation::Dot { dimensions, accumulation } => {
                let [left_contracting, right_contracting, left_batching, right_batching] = dimensions;
                ArrayIrOperation::Array(ArrayOperation::Dot(
                    DotOperation::new(DotDimensionNumbers::new(
                        left_contracting,
                        right_contracting,
                        left_batching,
                        right_batching,
                    ))
                    .with_accumulation_type(accumulation.map(DataType::from)),
                ))
                .into()
            }
            WireOperation::While { bound } => {
                ArrayIrOperation::While(WhileOperation::new().with_iteration_bound(bound)?).into()
            }
            WireOperation::Condition => ArrayIrOperation::Condition(ConditionOperation::new()).into(),
            WireOperation::DimensionFromScalar(dimension) => ArrayIrOperation::DimensionFromScalar(
                DimensionFromScalarOperation::new(self.dimension_type(dimension)?),
            )
            .into(),
            WireOperation::DimensionToScalar => ArrayIrOperation::DimensionToScalar(DimensionToScalarOperation).into(),
            WireOperation::TileLoad { block_shape, boundary_policy } => {
                TileLoadOperation::new(block_shape, boundary_policy.into()).map_err(ProgramError::from)?.into()
            }
            WireOperation::Scratch { referent, alignment } => {
                ScratchOperation::new(self.array_type(referent)?, alignment).map_err(ProgramError::custom)?.into()
            }
            WireOperation::MaskedLoad => MaskedLoadOperation::new().into(),
            WireOperation::MaskedStore => MaskedStoreOperation::new().into(),
            WireOperation::MaskedSwap => MaskedSwapOperation::new().into(),
            WireOperation::AsyncCopy => AsyncCopyOperation::new().into(),
            WireOperation::Wait => WaitOperation.into(),
            WireOperation::Add => ArrayIrOperation::Array(ArrayOperation::Add(AddOperation::new())).into(),
            WireOperation::Sub => ArrayIrOperation::Array(ArrayOperation::Sub(SubOperation::new())).into(),
            WireOperation::Mul => ArrayIrOperation::Array(ArrayOperation::Mul(MulOperation::new())).into(),
            WireOperation::Div => ArrayIrOperation::Array(ArrayOperation::Div(DivOperation::new())).into(),
            WireOperation::Rem => ArrayIrOperation::Array(ArrayOperation::Rem(RemOperation::new())).into(),
            WireOperation::Neg => ArrayIrOperation::Array(ArrayOperation::Neg(NegOperation::new())).into(),
            WireOperation::Abs => ArrayIrOperation::Array(ArrayOperation::Abs(AbsOperation::new())).into(),
            WireOperation::Min => ArrayIrOperation::Array(ArrayOperation::Min(MinOperation::new())).into(),
            WireOperation::Max => ArrayIrOperation::Array(ArrayOperation::Max(MaxOperation::new())).into(),
            WireOperation::Clamp => ArrayIrOperation::Array(ArrayOperation::Clamp(ClampOperation::new())).into(),
            WireOperation::Exp => ArrayIrOperation::Array(ArrayOperation::Exp(ExpOperation::new())).into(),
            WireOperation::Log => ArrayIrOperation::Array(ArrayOperation::Log(LogOperation::new())).into(),
            WireOperation::Sqrt => ArrayIrOperation::Array(ArrayOperation::Sqrt(SqrtOperation::new())).into(),
            WireOperation::Rsqrt => ArrayIrOperation::Array(ArrayOperation::Rsqrt(RsqrtOperation::new())).into(),
            WireOperation::Tanh => ArrayIrOperation::Array(ArrayOperation::Tanh(TanhOperation::new())).into(),
            WireOperation::Not => ArrayIrOperation::Array(ArrayOperation::Not(NotOperation::new())).into(),
            WireOperation::And => ArrayIrOperation::Array(ArrayOperation::And(AndOperation::new())).into(),
            WireOperation::Or => ArrayIrOperation::Array(ArrayOperation::Or(OrOperation::new())).into(),
            WireOperation::Xor => ArrayIrOperation::Array(ArrayOperation::Xor(XorOperation::new())).into(),
            WireOperation::Select => ArrayIrOperation::Array(ArrayOperation::Select(SelectOperation::new())).into(),
            WireOperation::ReferenceNew => ArrayIrOperation::ReferenceNew(ReferenceNewOperation::new()).into(),
            WireOperation::ReferenceRead => ArrayIrOperation::ReferenceRead(ReferenceReadOperation::new()).into(),
            WireOperation::ReferenceWrite => ArrayIrOperation::ReferenceWrite(ReferenceWriteOperation::new()).into(),
            WireOperation::ReferenceSwap => ArrayIrOperation::ReferenceSwap(ReferenceSwapOperation::new()).into(),
            WireOperation::ReferenceFreeze => ArrayIrOperation::ReferenceFreeze(ReferenceFreezeOperation::new()).into(),
            WireOperation::ReferenceAddUpdate => {
                ArrayIrOperation::ReferenceAddUpdate(ReferenceAddUpdateOperation::new()).into()
            }
            WireOperation::ReferenceAtomicAddUpdate => {
                ArrayIrOperation::ReferenceAtomicAddUpdate(ReferenceAtomicAddUpdateOperation::new()).into()
            }
            WireOperation::DimensionAdd { left, right } => ArrayIrOperation::Dimension(DimensionOperation::Add(
                DimensionAddOperation::new(&self.dimension_type(left)?, &self.dimension_type(right)?)
                    .map_err(ProgramError::from)?,
            ))
            .into(),
            WireOperation::DimensionSub { left, right } => ArrayIrOperation::Dimension(DimensionOperation::Sub(
                DimensionSubOperation::new(&self.dimension_type(left)?, &self.dimension_type(right)?)
                    .map_err(ProgramError::from)?,
            ))
            .into(),
            WireOperation::DimensionMul { left, right } => ArrayIrOperation::Dimension(DimensionOperation::Mul(
                DimensionMulOperation::new(&self.dimension_type(left)?, &self.dimension_type(right)?)
                    .map_err(ProgramError::from)?,
            ))
            .into(),
            WireOperation::DimensionDiv { left, right } => ArrayIrOperation::Dimension(DimensionOperation::Div(
                DimensionDivOperation::new(&self.dimension_type(left)?, &self.dimension_type(right)?)
                    .map_err(ProgramError::from)?,
            ))
            .into(),
            WireOperation::DimensionRem { left, right } => ArrayIrOperation::Dimension(DimensionOperation::Rem(
                DimensionRemOperation::new(&self.dimension_type(left)?, &self.dimension_type(right)?)
                    .map_err(ProgramError::from)?,
            ))
            .into(),
            WireOperation::SaturatingSub => {
                ArrayIrOperation::Array(ArrayOperation::SaturatingSub(SaturatingSubOperation::new())).into()
            }
            WireOperation::SaturatingSubDimension => {
                ArrayIrOperation::Dimension(DimensionOperation::SaturatingSub(SaturatingSubOperation::new())).into()
            }
            WireOperation::Reduce { axes, kind } => {
                ArrayIrOperation::Array(ArrayOperation::Reduce(ReduceOperation::new(axes, kind.into()))).into()
            }
            WireOperation::WindowMask => WindowMaskOperation.into(),
            WireOperation::Sign => ArrayOperation::Sign(SignOperation::new()).into(),
            WireOperation::Pow => ArrayOperation::Pow(PowOperation::new()).into(),
            WireOperation::Sin => ArrayOperation::Sin(SinOperation::new()).into(),
            WireOperation::Cos => ArrayOperation::Cos(CosOperation::new()).into(),
            WireOperation::Tan => ArrayOperation::Tan(TanOperation::new()).into(),
            WireOperation::Atan2 => ArrayOperation::Atan2(Atan2Operation::new()).into(),
            WireOperation::ExpM1 => ArrayOperation::ExpM1(ExpM1Operation::new()).into(),
            WireOperation::Ln1p => ArrayOperation::Ln1p(Ln1pOperation::new()).into(),
            WireOperation::LogAddExp => ArrayOperation::LogAddExp(LogAddExpOperation::new()).into(),
            WireOperation::Logistic => ArrayOperation::Logistic(LogisticOperation::new()).into(),
            WireOperation::Erf => ArrayOperation::Erf(ErfOperation::new()).into(),
            WireOperation::Floor => ArrayOperation::Floor(FloorOperation::new()).into(),
            WireOperation::Ceil => ArrayOperation::Ceil(CeilOperation::new()).into(),
            WireOperation::Round => ArrayOperation::Round(RoundOperation::new()).into(),
            WireOperation::Complex => ArrayOperation::Complex(ComplexOperation::new()).into(),
            WireOperation::Real => ArrayOperation::Real(RealOperation::new()).into(),
            WireOperation::Imaginary => ArrayOperation::Imaginary(ImaginaryOperation::new()).into(),
            WireOperation::Conjugate => ArrayOperation::Conjugate(ConjugateOperation::new()).into(),
            WireOperation::AccurateUnary { function, accuracy } => {
                let accuracy = accuracy.decode()?;
                match function {
                    WireAccurateFunction::Exp => ArrayOperation::Exp(ExpOperation::new().with_accuracy(accuracy)),
                    WireAccurateFunction::ExpM1 => {
                        ArrayOperation::ExpM1(ExpM1Operation::new().with_accuracy(accuracy))
                    }
                    WireAccurateFunction::Log => ArrayOperation::Log(LogOperation::new().with_accuracy(accuracy)),
                    WireAccurateFunction::Ln1p => ArrayOperation::Ln1p(Ln1pOperation::new().with_accuracy(accuracy)),
                    WireAccurateFunction::Logistic => {
                        ArrayOperation::Logistic(LogisticOperation::new().with_accuracy(accuracy))
                    }
                    WireAccurateFunction::Sin => ArrayOperation::Sin(SinOperation::new().with_accuracy(accuracy)),
                    WireAccurateFunction::Cos => ArrayOperation::Cos(CosOperation::new().with_accuracy(accuracy)),
                    WireAccurateFunction::Tan => ArrayOperation::Tan(TanOperation::new().with_accuracy(accuracy)),
                    WireAccurateFunction::Tanh => ArrayOperation::Tanh(TanhOperation::new().with_accuracy(accuracy)),
                    WireAccurateFunction::Sqrt => ArrayOperation::Sqrt(SqrtOperation::new().with_accuracy(accuracy)),
                    WireAccurateFunction::Rsqrt => {
                        ArrayOperation::Rsqrt(RsqrtOperation::new().with_accuracy(accuracy))
                    }
                }
                .into()
            }
            WireOperation::DotWithPrecision { dimensions, accumulation, precision } => {
                let precision = DotPrecision::from(precision);
                if precision.is_default() {
                    return Err(invalid("`dot` precision record has the default precision"));
                }
                let [left_contracting, right_contracting, left_batching, right_batching] = dimensions;
                ArrayOperation::Dot(
                    DotOperation::new(DotDimensionNumbers::new(
                        left_contracting,
                        right_contracting,
                        left_batching,
                        right_batching,
                    ))
                    .with_accumulation_type(accumulation.map(DataType::from))
                    .with_precision(precision),
                )
                .into()
            }
            WireOperation::ZeroLike => ArrayOperation::ZeroLike(ZeroLikeOperation::new()).into(),
            WireOperation::One(r#type) => ArrayOperation::One(OneOperation::new(self.array_type(r#type)?)).into(),
            WireOperation::OneLike => ArrayOperation::OneLike(OneLikeOperation::new()).into(),
            WireOperation::Iota { r#type, dimension } => ArrayOperation::Iota(
                IotaOperation::new(self.array_type(r#type)?, dimension).map_err(ProgramError::from)?,
            )
            .into(),
            WireOperation::FillLike(value) => {
                let ArrayIrValue::Array(value) = self.value(value)? else {
                    return Err(invalid("fill like operation has a dimension payload"));
                };
                ArrayOperation::FillLike(FillLikeOperation::new(value).map_err(ProgramError::from)?).into()
            }
            WireOperation::IotaLike { dimension, data_type } => ArrayOperation::IotaLike(
                IotaLikeOperation::new(dimension, data_type.into()).map_err(ProgramError::from)?,
            )
            .into(),
            WireOperation::MixedOne(r#type) => {
                ArrayIrOperation::One(OneOperation::new(self.array_type(r#type)?)).into()
            }
            WireOperation::MixedIota { r#type, dimension } => ArrayIrOperation::Iota(
                IotaOperation::new(self.array_type(r#type)?, dimension).map_err(ProgramError::from)?,
            )
            .into(),
            WireOperation::ArgMax { axis, index_data_type } => {
                ArrayOperation::ArgMax(ArgMaxOperation::new(axis, index_data_type.into())).into()
            }
            WireOperation::ArgMin { axis, index_data_type } => {
                ArrayOperation::ArgMin(ArgMinOperation::new(axis, index_data_type.into())).into()
            }
            WireOperation::Cumulative { axis, kind, reverse } => {
                ArrayOperation::Cumulative(CumulativeOperation::new(axis, kind.into()).with_reverse(reverse)).into()
            }
            WireOperation::Sort { axis, key_count, direction, ordering } => ArrayOperation::Sort(
                SortOperation::new(axis, direction.into()).with_key_count(key_count).with_ordering(ordering.into()),
            )
            .into(),
            WireOperation::Transpose(axes) => ArrayOperation::Transpose(TransposeOperation::new(axes)).into(),
            WireOperation::Reverse(axes) => ArrayOperation::Reverse(ReverseOperation::new(axes)).into(),
            WireOperation::Reshape { input_shape, output_shape } => ArrayOperation::Reshape(ReshapeOperation::new(
                Shape::new(self.shape(input_shape)?),
                Shape::new(self.shape(output_shape)?),
            ))
            .into(),
            WireOperation::MixedReshape => ArrayIrOperation::Reshape(
                DynamicReshapeOperation::new().with_input_types(input_types).map_err(ProgramError::from)?,
            )
            .into(),
            WireOperation::Broadcast { output_type, output_axes } => {
                ArrayOperation::Broadcast(BroadcastOperation::new(self.array_type(output_type)?, output_axes)).into()
            }
            WireOperation::BroadcastLike(output_axes) => {
                ArrayOperation::BroadcastLike(BroadcastLikeOperation::new(output_axes)).into()
            }
            WireOperation::MixedBroadcast { output_axes, output_layout } => ArrayIrOperation::Broadcast(
                DynamicBroadcastOperation::new(output_axes).with_output_layout(output_layout.map(WireLayout::decode)),
            )
            .into(),
            WireOperation::Pad { low, high, interior } => {
                ArrayOperation::Pad(PadOperation::new(low, high, interior)?).into()
            }
            WireOperation::MixedPad { low, high, interior } => ArrayIrOperation::Pad(
                PadOperation::new(low, high, interior)?.with_input_types(input_types).map_err(ProgramError::from)?,
            )
            .into(),
            WireOperation::Concatenate { axis } => {
                let Some(ArrayIrType::Array(r#type)) = input_types.first() else {
                    return Err(invalid("`concatenate` record requires a leading array input"));
                };
                ArrayOperation::Concatenate(
                    ConcatenateOperation::<ArrayType>::new(axis, r#type.rank()).map_err(ProgramError::from)?,
                )
                .into()
            }
            WireOperation::MixedConcatenate { axis } => ArrayIrOperation::Concatenate(
                ConcatenateOperation::<ArrayIrType>::new(axis, input_types).map_err(ProgramError::from)?,
            )
            .into(),
            WireOperation::Slice { start_indices, limits, strides } => ArrayOperation::Slice(
                SliceOperation::new(start_indices, self.shape(limits)?).with_strides(strides)?,
            )
            .into(),
            WireOperation::UpdateSlice(start_indices) => {
                ArrayOperation::UpdateSlice(UpdateSliceOperation::new(start_indices)).into()
            }
            WireOperation::DynamicSlice { sizes, allow_negative_indices } => ArrayOperation::DynamicSlice(
                DynamicSliceOperation::new(sizes).with_allow_negative_indices(allow_negative_indices),
            )
            .into(),
            WireOperation::DynamicUpdateSlice { allow_negative_indices } => ArrayOperation::DynamicUpdateSlice(
                DynamicUpdateSliceOperation::new().with_allow_negative_indices(allow_negative_indices),
            )
            .into(),
            WireOperation::ConvertElementType { data_type, bitcast } => {
                ArrayOperation::ConvertElementType(ConvertElementTypeOperation::new(data_type.into(), bitcast)).into()
            }
            WireOperation::ReducePrecision { exponent_bits, mantissa_bits } => {
                ArrayOperation::ReducePrecision(ReducePrecisionOperation::new(exponent_bits, mantissa_bits)).into()
            }
            WireOperation::TransferToMemory(memory) => {
                ArrayOperation::TransferToMemory(TransferToMemoryOperation::new(memory.into())).into()
            }
            WireOperation::ArrayCondition => ArrayOperation::Condition(ConditionOperation::new()).into(),
            WireOperation::ArrayWhile { bound } => {
                ArrayOperation::While(WhileOperation::new().with_iteration_bound(bound)?).into()
            }
            WireOperation::StopGradient => ArrayOperation::StopGradient(StopGradientOperation::new()).into(),
            WireOperation::Tag(key) => ArrayOperation::Tag(TagOperation::new(key)).into(),
            WireOperation::Cast { output_type, input_types, checked } => {
                let output_type = self.array_type(output_type)?;
                let input_types =
                    input_types.into_iter().map(|r#type| self.array_type(r#type)).collect::<Result<_, _>>()?;
                let operation = if checked {
                    CastOperation::checked(output_type, input_types)
                } else {
                    CastOperation::unchecked(output_type, input_types)
                };
                ArrayOperation::Cast(operation.map_err(ProgramError::from)?).into()
            }
            WireOperation::MixedCast { output_type, input_types, checked } => {
                let output_type = self.r#type(output_type)?;
                let input_types =
                    input_types.into_iter().map(|r#type| self.r#type(r#type)).collect::<Result<_, _>>()?;
                let operation = if checked {
                    CastOperation::checked(output_type, input_types)
                } else {
                    CastOperation::unchecked(output_type, input_types)
                };
                ArrayIrOperation::Cast(operation.map_err(ProgramError::from)?).into()
            }
            WireOperation::DimensionMax { left, right } => ArrayIrOperation::Dimension(DimensionOperation::Max(
                DimensionMaxOperation::new(&self.dimension_type(left)?, &self.dimension_type(right)?)
                    .map_err(ProgramError::from)?,
            ))
            .into(),
            WireOperation::DimensionMin { left, right } => ArrayIrOperation::Dimension(DimensionOperation::Min(
                DimensionMinOperation::new(&self.dimension_type(left)?, &self.dimension_type(right)?)
                    .map_err(ProgramError::from)?,
            ))
            .into(),
            WireOperation::DimensionPow { left, right } => ArrayIrOperation::Dimension(DimensionOperation::Pow(
                DimensionPowOperation::new(&self.dimension_type(left)?, &self.dimension_type(right)?)
                    .map_err(ProgramError::from)?,
            ))
            .into(),
            WireOperation::DimensionSize { axis } => {
                let Some(ArrayIrType::Array(r#type)) = input_types.first() else {
                    return Err(invalid("`dimension_size` record requires an array input"));
                };
                ArrayIrOperation::DimensionSize(DimensionSizeOperation::new(r#type, axis).map_err(ProgramError::from)?)
                    .into()
            }
            WireOperation::KeyOnly(_) => return Err(invalid("key-only records have no kernel source encoding")),
        })
    }

    /// Validates indices before constructing the canonical arena, whose sealing checks SSA and region interfaces.
    fn program<O: Operation<Type = ArrayIrType>, F>(
        &self,
        wire: WireProgram,
        project: F,
    ) -> Result<FlatProgram<EagerContext<ArrayIrValue<Array>, O>>, KernelSerializationError>
    where
        F: Fn(KernelOperation) -> Result<O, KernelSerializationError>,
    {
        validate_decoded_records(wire.regions.len())?;
        if wire.entry >= wire.regions.len() {
            return Err(invalid("entry region is out of bounds"));
        }
        let mut regions = Vec::with_capacity(wire.regions.len());
        for (region_index, region) in wire.regions.into_iter().enumerate() {
            validate_decoded_records(region.atoms.len())?;
            validate_decoded_records(region.instructions.len())?;
            let atom_count = region.atoms.len();
            let validate_atoms = |indices: &[usize]| -> Result<(), KernelSerializationError> {
                if indices.iter().any(|&index| index >= atom_count) {
                    return Err(invalid("atom index is out of bounds"));
                }
                Ok(())
            };
            validate_atoms(&region.inputs)?;
            validate_atoms(&region.outputs)?;
            for instruction in &region.instructions {
                validate_atoms(&instruction.inputs)?;
                validate_atoms(&instruction.outputs)?;
                if instruction.regions.iter().any(|&index| index >= region_index) {
                    return Err(invalid("region edge must reference an earlier arena entry"));
                }
            }
            let atoms = region
                .atoms
                .into_iter()
                .map(|atom| {
                    Ok(match atom {
                        WireAtom::Constant(value) => Atom::Constant(self.value(value)?),
                        WireAtom::Variable(r#type) => Atom::Variable(self.r#type(r#type)?),
                    })
                })
                .collect::<Result<Vec<Atom<ArrayIrValue<Array>>>, KernelSerializationError>>()?;
            let instructions = region
                .instructions
                .into_iter()
                .map(|instruction| {
                    let input_types = instruction
                        .inputs
                        .iter()
                        .map(|&index| atoms[index].r#type().into_owned())
                        .collect::<Vec<_>>();
                    let mut operation = self.operation(instruction.operation, &input_types)?;
                    let mut accesses = operation.effects().accesses().map(|(input, _)| input).collect::<Vec<_>>();
                    accesses.sort_unstable();
                    if instruction.reference_transforms.iter().map(|path| path.input_index).collect::<Vec<_>>()
                        != accesses
                    {
                        return Err(invalid(
                            "reference transform descriptors do not match the operation's access inputs",
                        ));
                    }
                    for path in instruction.reference_transforms {
                        operation = operation.with_reference_access_transforms(
                            path.input_index,
                            path.transforms.into_iter().map(WireReferenceTransform::decode).collect(),
                        )?;
                    }
                    validated_reference_access_descriptors(&operation, instruction.inputs.len()).map_err(|error| {
                        match error {
                            ProgramError::MalformedProgram(message) => invalid(message),
                            error => error.into(),
                        }
                    })?;
                    Ok(Instruction::new(
                        project(operation)?,
                        instruction.inputs.into_iter().map(AtomId::new).collect(),
                        instruction.outputs.into_iter().map(AtomId::new).collect(),
                        instruction.regions.into_iter().map(RegionId::new).collect(),
                    )
                    .with_provenance(instruction.provenance.decode()?))
                })
                .collect::<Result<_, KernelSerializationError>>()?;
            regions.push(Region::new(
                atoms,
                region.inputs.into_iter().map(AtomId::new).collect(),
                region.outputs.into_iter().map(AtomId::new).collect(),
                instructions,
            ));
        }
        let input_count = regions[wire.entry].input_ids().len();
        let output_count = regions[wire.entry].output_ids().len();
        Ok(Program::new(
            vec![Placeholder; input_count],
            vec![Placeholder; output_count],
            regions,
            RegionId::new(wire.entry),
        )?)
    }

    /// Rebuilds mappings and call metadata before its exact body is validated.
    fn call(&self, wire: WireCall) -> Result<KernelCallOperation, KernelSerializationError> {
        validate_decoded_records(wire.grid.len())?;
        validate_decoded_records(wire.parameters.len())?;
        let grid = Grid::new(
            wire.grid
                .into_iter()
                .map(|dimension| {
                    let mut result = GridDimension::new(self.dimension(dimension.extent)?, dimension.execution.into());
                    if let Some(name) = dimension.name {
                        result = result.with_name(name).map_err(KernelError::from)?;
                    }
                    Ok(result)
                })
                .collect::<Result<_, KernelSerializationError>>()?,
        )
        .map_err(KernelError::from)?;
        let parameters = wire
            .parameters
            .into_iter()
            .map(|parameter| {
                let r#type = self.array_type(parameter.r#type)?;
                let access = match parameter.access {
                    WireAccess::ReadOnly => KernelParameterAccess::ReadOnly,
                    WireAccess::WriteOnly => KernelParameterAccess::WriteOnly,
                    WireAccess::ReadWrite => KernelParameterAccess::ReadWrite,
                };
                let program = self.program(parameter.mapping, |operation| match operation {
                    KernelOperation::Portable(operation) => Ok(operation),
                    operation => Err(unsupported(format!("mapping operation `{}`", operation.name()))),
                })?;
                let mapping = BlockMapping::new(program, parameter.block_shape, parameter.boundary_policy.into())
                    .map_err(KernelError::from)?;
                Ok(KernelParameter::new(r#type, access, mapping)?)
            })
            .collect::<Result<_, KernelSerializationError>>()?;
        let prefetch = wire.prefetch.into_iter().map(|r#type| self.array_type(r#type)).collect::<Result<_, _>>()?;
        let coordinates = wire
            .coordinates
            .into_iter()
            .map(|coordinate| self.dimension_type(coordinate))
            .collect::<Result<_, _>>()?;
        Ok(KernelCallOperation::from_parts(grid, parameters, prefetch, coordinates)?)
    }
}

/// Deserializes a collection of at most `LIMIT` entries without trusting an untrusted sequence size hint for
/// allocation.
struct BoundedCollection<T, const LIMIT: usize>(
    /// Deserialized entries.
    Vec<T>,
);

impl<'de, T: Deserialize<'de>, const LIMIT: usize> Deserialize<'de> for BoundedCollection<T, LIMIT> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        /// Sequence visitor shared by transport records and literal bytes.
        struct Visitor<T, const LIMIT: usize>(
            /// Marker of the entry type.
            PhantomData<T>,
        );

        impl<'de, T: Deserialize<'de>, const LIMIT: usize> serde::de::Visitor<'de> for Visitor<T, LIMIT> {
            type Value = BoundedCollection<T, LIMIT>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "a sequence with at most {LIMIT} entries")
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
                if sequence.size_hint().is_some_and(|size| size > LIMIT) {
                    return Err(serde::de::Error::custom(format!("kernel source sequence exceeds {LIMIT} entries")));
                }
                let mut values = Vec::new();
                while let Some(value) = sequence.next_element()? {
                    if values.len() == LIMIT {
                        return Err(serde::de::Error::custom(format!(
                            "kernel source sequence exceeds {LIMIT} entries",
                        )));
                    }
                    values.push(value);
                }
                Ok(BoundedCollection(values))
            }
        }

        deserializer.deserialize_seq(Visitor::<T, LIMIT>(PhantomData))
    }
}

/// Validates collection sizes while decoding, before constructing canonical arena metadata. A document that exceeds
/// the record budget is malformed.
fn validate_decoded_records(count: usize) -> Result<(), KernelSerializationError> {
    if count > MAXIMUM_RECORDS { Err(invalid(format!("record count exceeds {MAXIMUM_RECORDS}"))) } else { Ok(()) }
}

/// Validates collection sizes while encoding. An in-memory definition that exceeds the record budget is valid but
/// cannot be transported, so it is reported as unsupported rather than malformed.
fn validate_encoded_records(count: usize) -> Result<(), KernelSerializationError> {
    if count > MAXIMUM_RECORDS {
        Err(unsupported(format!("record count exceeding {MAXIMUM_RECORDS}")))
    } else {
        Ok(())
    }
}

/// Creates an explicit diagnostic for an unsupported payload.
fn unsupported(feature: impl Into<String>) -> KernelSerializationError {
    KernelSerializationError::Unsupported { feature: feature.into() }
}

/// Creates a diagnostic for a malformed source.
fn invalid(message: impl Into<String>) -> KernelSerializationError {
    KernelSerializationError::Invalid { message: message.into() }
}

/// Creates the [`KernelDefinition::semantic_key`] error for a definition that has no semantic identity, which
/// [`encode_semantic_key`] reports as the wrapped [`TypeError`].
fn semantic_key_error(message: impl Into<String>) -> KernelSerializationError {
    ProgramError::from(TypeError::invalid(message)).into()
}

/// Creates the [`KernelDefinition::semantic_key`] error for the composite operation named `name` with retained rules,
/// whose identity is local to the process.
fn retained_composite_error(name: &str) -> KernelSerializationError {
    semantic_key_error(format!(
        "`{name}` operations with retained rules have no kernel semantic descriptor encoding, because the identity of \
         their rules is local to the process",
    ))
}

/// Applies the source record budget to one sequence before canonical reconstruction.
fn bounded_records<'de, D: Deserializer<'de>, T: Deserialize<'de>>(deserializer: D) -> Result<Vec<T>, D::Error> {
    Ok(BoundedCollection::<T, MAXIMUM_RECORDS>::deserialize(deserializer)?.0)
}

/// Applies the literal byte budget while decoding its transport sequence.
fn bounded_bytes<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
    Ok(BoundedCollection::<u8, MAXIMUM_LITERAL_BYTES>::deserialize(deserializer)?.0)
}

/// Bounds both levels of nested tile metadata before constructing canonical layouts.
fn bounded_tiles<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<Vec<Option<usize>>>, D::Error> {
    let tiles = BoundedCollection::<BoundedCollection<Option<usize>, MAXIMUM_RECORDS>, MAXIMUM_RECORDS>::deserialize(
        deserializer,
    )?;
    Ok(tiles.0.into_iter().map(|tile| tile.0).collect())
}

/// Bounds each of the four fixed sequences of dot dimension numbers.
fn bounded_dot_dimensions<'de, D: Deserializer<'de>>(deserializer: D) -> Result<[Vec<usize>; 4], D::Error> {
    Ok(<[BoundedCollection<usize, MAXIMUM_RECORDS>; 4]>::deserialize(deserializer)?.map(|dimensions| dimensions.0))
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        Context, LogicalMesh, MeshAxis, MeshAxisType, PrintOperation, ProgramBuilder, Sharding, TypedProgram,
    };
    use ryft_macros::kernel;

    use crate::authoring::whole_array_parameter;
    use crate::interpretation::DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS;

    use super::*;

    #[kernel(crate = "crate", requires = left.shape()[1] == right.shape()[0])]
    fn tiled_matmul(
        #[input(data_type = F32, rank = 2)] left: &Array,
        #[input(data_type = F32, rank = 2)] right: &Array,
        #[output(data_type = F32, shape = [left.shape()[0], right.shape()[1]], tile = [2, 2], boundary = masked)]
        output: &mut Array,
    ) {
        let [row, column] = output.tile_index();
        let left_tiles = left.tiles([2, 2]).pad(0.0);
        let right_tiles = right.tiles([2, 2]).pad(0.0);
        let mut accumulator = zeros::<f32>([2, 2]);
        for depth in 0..left.shape()[1].div_ceil(2) {
            let left_tile = left_tiles.load([row, depth]);
            let right_tile = right_tiles.load([depth, column]);
            accumulator += left_tile.dot(right_tile);
        }
        output.store(accumulator);
    }

    /// Roundtrips a definition through the public serde contract and preserves its exact semantic identity.
    fn roundtrip(definition: &KernelDefinition) -> KernelDefinition {
        let encoded = serde_json::to_vec(definition).unwrap();
        let decoded: KernelDefinition = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded.semantic_key().unwrap(), definition.semantic_key().unwrap());
        assert_eq!(serde_json::to_vec(&decoded).unwrap(), encoded);
        decoded
    }

    /// Constructs a write-only scalar kernel preserving the supplied literal bits.
    fn literal_definition(value: Array) -> KernelDefinition {
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![whole_array_parameter(value.r#type().into_owned(), KernelParameterAccess::WriteOnly).unwrap()],
        )
        .unwrap();
        KernelDefinition::trace(operation, |(references, _)| {
            let context = references[0].context();
            let value = context
                .bind(ArrayIrOperation::Array(ArrayOperation::Constant(ConstantOperation::new(value))), vec![], &[])?
                .remove(0);
            context.bind(ReferenceWriteOperation::new(), vec![], &[references[0].clone(), value])?;
            Ok(())
        })
        .unwrap()
    }

    /// Builds a read-only body whose one access carries the supplied selection and optional scalar binding.
    fn viewed_read_definition(transform: ArrayReferenceTransform, binding: Option<i64>) -> KernelDefinition {
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(ArrayType::new_static(DataType::I32, [2, 2]), KernelParameterAccess::ReadOnly)
                    .unwrap(),
            ],
        )
        .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let root = builder.add_input(call.body_input_types()[0].clone());
        let mut inputs = vec![root];
        if let Some(binding) = binding {
            inputs.push(builder.add_constant(ArrayIrValue::Array(Array::scalar(binding).unwrap())));
        }
        builder
            .add_instruction(ReferenceReadOperation::new().with_transforms(vec![transform]), vec![], inputs, None)
            .unwrap();
        KernelDefinition::new(call, builder.build(vec![], vec![Placeholder], vec![]).unwrap()).unwrap()
    }

    /// Builds a kernel that reads one whole array of each input type, applies `operation` to those arrays followed by
    /// one dimension constant per extent, and writes its only result to a whole array of the output type.
    fn operation_definition(
        operation: impl Into<KernelOperation>,
        input_types: Vec<ArrayType>,
        extents: Vec<usize>,
        output_type: ArrayType,
    ) -> KernelDefinition {
        let mut parameters = input_types
            .into_iter()
            .map(|r#type| whole_array_parameter(r#type, KernelParameterAccess::ReadOnly).unwrap())
            .collect::<Vec<_>>();
        parameters.push(whole_array_parameter(output_type, KernelParameterAccess::WriteOnly).unwrap());
        let call = KernelCallOperation::new(Grid::new(vec![]).unwrap(), parameters).unwrap();
        let operation = operation.into();
        KernelDefinition::trace(call, |(references, _)| {
            let context = references[0].context();
            let (output, inputs) = references.split_last().unwrap();
            let mut values = Vec::new();
            for input in inputs {
                let value = context.bind(ReferenceReadOperation::new(), vec![], std::slice::from_ref(input))?;
                values.extend(value);
            }
            for extent in extents {
                values.push(context.lift(ArrayIrValue::Dimension(DimensionValue::constant(extent)?))?);
            }
            let result = context.bind(operation, vec![], &values)?.remove(0);
            context.bind(ReferenceWriteOperation::new(), vec![], &[output.clone(), result])?;
            Ok(())
        })
        .unwrap()
    }

    /// Asserts that serializing `definition` round trips its exact operations, which also checks its semantic key.
    fn assert_operations_roundtrip(definition: &KernelDefinition) {
        let decoded = roundtrip(definition);
        assert_eq!(
            decoded.body().instructions().iter().map(Instruction::operation).collect::<Vec<_>>(),
            definition.body().instructions().iter().map(Instruction::operation).collect::<Vec<_>>(),
        );
        assert_eq!(decoded.body().to_string(), definition.body().to_string());
    }

    #[test]
    fn test_kernel_serialization_error_display() {
        assert_eq!(
            KernelSerializationError::Schema { version: 999 }.to_string(),
            "unsupported kernel source schema version 999",
        );
        assert_eq!(
            KernelSerializationError::Unsupported { feature: "`sin`".to_owned() }.to_string(),
            "kernel source serialization does not support `sin`",
        );
        assert_eq!(
            KernelSerializationError::Invalid { message: "entry region is out of bounds".to_owned() }.to_string(),
            "invalid serialized kernel source: entry region is out of bounds",
        );
        assert_eq!(
            KernelSerializationError::Program(ProgramError::MalformedProgram("missing entry region".to_owned()))
                .to_string(),
            "encountered malformed program: missing entry region",
        );
        assert_eq!(
            KernelSerializationError::Kernel(KernelError::ParameterRank { array_rank: 2, block_rank: 1 }).to_string(),
            "kernel parameter has array rank 2 but block rank 1",
        );
    }

    #[test]
    fn test_kernel_definition_to_source() {
        let definition = literal_definition(Array::scalar(-0.0f32).unwrap());
        let source = definition.to_source().unwrap();
        assert_eq!(format!("{source:?}"), format!("KernelSource {{ version: {KERNEL_SOURCE_VERSION}, .. }}"));

        // A source serializes to exactly the same document as the definition that produced it.
        assert_eq!(serde_json::to_vec(&source).unwrap(), serde_json::to_vec(&definition).unwrap());
    }

    #[test]
    fn test_kernel_definition_to_source_document() {
        // This test pins the produced records. A deliberate change to them requires a `KERNEL_SOURCE_VERSION` bump.
        let definition = literal_definition(Array::scalar(-0.0f32).unwrap());
        assert_eq!(
            serde_json::to_string_pretty(&definition.to_source().unwrap()).unwrap(),
            indoc! {r#"
            {
              "version": 1,
              "variables": [],
              "call": {
                "grid": [],
                "parameters": [
                  {
                    "type": {
                      "data_type": "F32",
                      "shape": [],
                      "layout": null,
                      "memory": "Device"
                    },
                    "access": "WriteOnly",
                    "mapping": {
                      "regions": [
                        {
                          "atoms": [],
                          "inputs": [],
                          "outputs": [],
                          "instructions": []
                        }
                      ],
                      "entry": 0
                    },
                    "block_shape": [],
                    "boundary_policy": "InBounds"
                  }
                ],
                "coordinates": [],
                "prefetch": []
              },
              "body": {
                "regions": [
                  {
                    "atoms": [
                      {
                        "Variable": {
                          "Reference": {
                            "data_type": "F32",
                            "shape": [],
                            "layout": null,
                            "memory": "Device"
                          }
                        }
                      },
                      {
                        "Variable": {
                          "Array": {
                            "data_type": "F32",
                            "shape": [],
                            "layout": null,
                            "memory": "Device"
                          }
                        }
                      }
                    ],
                    "inputs": [
                      0
                    ],
                    "outputs": [],
                    "instructions": [
                      {
                        "operation": {
                          "ArrayConstant": {
                            "Array": {
                              "type": {
                                "data_type": "F32",
                                "shape": [],
                                "layout": null,
                                "memory": "Device"
                              },
                              "bytes": [
                                0,
                                0,
                                0,
                                128
                              ]
                            }
                          }
                        },
                        "reference_transforms": [],
                        "inputs": [],
                        "outputs": [
                          1
                        ],
                        "regions": [],
                        "provenance": {
                          "nodes": [
                            "Unknown"
                          ],
                          "root": 0
                        }
                      },
                      {
                        "operation": "ReferenceWrite",
                        "reference_transforms": [
                          {
                            "input_index": 0,
                            "transforms": []
                          }
                        ],
                        "inputs": [
                          0,
                          1
                        ],
                        "outputs": [],
                        "regions": [],
                        "provenance": {
                          "nodes": [
                            "Unknown"
                          ],
                          "root": 0
                        }
                      }
                    ]
                  }
                ],
                "entry": 0
              }
            }"#},
        );
    }

    #[test]
    fn test_kernel_definition_to_source_unsupported_operation() {
        let r#type = ArrayType::scalar(DataType::F32);
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(r#type.clone(), KernelParameterAccess::ReadOnly).unwrap(),
                whole_array_parameter(r#type, KernelParameterAccess::WriteOnly).unwrap(),
            ],
        )
        .unwrap();
        let definition = KernelDefinition::trace(operation, |(references, _)| {
            let context = references[0].context();
            let input = context.bind(ReferenceReadOperation::new(), vec![], &[references[0].clone()])?.remove(0);
            let output = context
                .bind(ArrayIrOperation::Array(ArrayOperation::Print(PrintOperation::new("value"))), vec![], &[input])?
                .remove(0);
            context.bind(ReferenceWriteOperation::new(), vec![], &[references[1].clone(), output])?;
            Ok(())
        })
        .unwrap();
        assert_eq!(
            definition.to_source().unwrap_err(),
            KernelSerializationError::Unsupported { feature: "`print`".to_owned() },
        );

        // The `serde` implementation reports the same error as a custom error message.
        assert_eq!(
            serde_json::to_vec(&definition).unwrap_err().to_string(),
            "kernel source serialization does not support `print`",
        );
    }

    #[test]
    fn test_kernel_definition_from_source() {
        let definition = literal_definition(Array::scalar(-0.0f32).unwrap());
        let decoded = KernelDefinition::from_source(definition.to_source().unwrap()).unwrap();
        assert_eq!(decoded.semantic_key().unwrap(), definition.semantic_key().unwrap());
        assert_eq!(serde_json::to_vec(&decoded).unwrap(), serde_json::to_vec(&definition).unwrap());
    }

    #[test]
    fn test_kernel_definition_from_source_schema() {
        let definition = literal_definition(Array::scalar(1.0f32).unwrap());
        let mut wire = serde_json::to_value(definition).unwrap();
        wire["version"] = serde_json::json!(999);

        // Deserializing a source decodes only its structure, so the version is checked when the definition is built.
        let source = serde_json::from_value::<KernelSource>(wire).unwrap();
        assert_eq!(
            KernelDefinition::from_source(source).unwrap_err(),
            KernelSerializationError::Schema { version: 999 },
        );
    }

    #[test]
    fn test_kernel_definition_from_source_invalid_indices() {
        let definition = literal_definition(Array::scalar(1.0f32).unwrap());
        let mut wire = serde_json::to_value(definition).unwrap();
        wire["body"]["entry"] = serde_json::json!(999);
        let source = serde_json::from_value::<KernelSource>(wire).unwrap();
        assert_eq!(
            KernelDefinition::from_source(source).unwrap_err(),
            KernelSerializationError::Invalid { message: "entry region is out of bounds".to_owned() },
        );
    }

    #[test]
    fn test_kernel_definition_from_source_invalid_coordinate_identity() {
        let variable = DimensionVariable::new("programs", DimensionBounds::non_negative(Some(2)).unwrap());
        let grid = Grid::new(vec![GridDimension::new(Dimension::Dynamic(variable), GridExecution::Parallel)]).unwrap();
        let definition: KernelDefinition =
            KernelDefinition::trace(KernelCallOperation::new(grid, vec![]).unwrap(), |_| Ok(())).unwrap();
        let mut wire = serde_json::to_value(definition).unwrap();
        wire["call"]["coordinates"][0] = serde_json::json!({ "Dynamic": 999 });
        let source = serde_json::from_value::<KernelSource>(wire).unwrap();
        assert_eq!(
            KernelDefinition::from_source(source).unwrap_err(),
            KernelSerializationError::Invalid { message: "dimension identity 999 is out of bounds".to_owned() },
        );
    }

    #[test]
    fn test_kernel_definition_from_source_invalid_region_edge() {
        let definition = literal_definition(Array::scalar(1.0f32).unwrap());
        let mut wire = serde_json::to_value(definition).unwrap();
        wire["body"]["regions"][0]["instructions"][0]["regions"] = serde_json::json!([0]);
        let source = serde_json::from_value::<KernelSource>(wire).unwrap();
        assert_eq!(
            KernelDefinition::from_source(source).unwrap_err(),
            KernelSerializationError::Invalid {
                message: "region edge must reference an earlier arena entry".to_owned(),
            },
        );
    }

    #[test]
    fn test_kernel_definition_from_source_invalid_reference_binding_count() {
        let definition = viewed_read_definition(
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
            Some(1),
        );
        let mut wire = serde_json::to_value(&definition).unwrap();
        wire["body"]["regions"][0]["instructions"][0]["inputs"] = serde_json::json!([0]);
        let source = serde_json::from_value::<KernelSource>(wire).unwrap();
        assert_eq!(
            KernelDefinition::from_source(source).unwrap_err(),
            KernelSerializationError::Invalid {
                message: "operation `reference_read` reference transforms require 2 inputs but the instruction has 1"
                    .to_owned(),
            },
        );
    }

    #[test]
    fn test_kernel_definition_from_source_invalid_reference_descriptor() {
        let definition = viewed_read_definition(
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
            None,
        );
        let mut wire = serde_json::to_value(&definition).unwrap();
        wire["body"]["regions"][0]["instructions"][0]["reference_transforms"][0]["input_index"] = serde_json::json!(1);
        let source = serde_json::from_value::<KernelSource>(wire).unwrap();
        assert_eq!(
            KernelDefinition::from_source(source).unwrap_err(),
            KernelSerializationError::Invalid {
                message: "reference transform descriptors do not match the operation's access inputs".to_owned(),
            },
        );
    }

    #[test]
    fn test_kernel_definition_from_source_key_only_records() {
        // Key mode encodes a print through a key-only operation record, which deserialization never accepts.
        let r#type = ArrayType::scalar(DataType::F32);
        let operation = ArrayOperation::Print(PrintOperation::new("value"));
        let definition = operation_definition(operation, vec![r#type.clone()], vec![], r#type);
        let document = Encoder::new(EncodingMode::Key).definition(&definition).unwrap();
        let error = serde_json::from_value::<KernelSource>(serde_json::to_value(&document).unwrap()).unwrap_err();
        assert!(error.to_string().starts_with("unknown variant `KeyOnly`"));

        // Key-only distributed type metadata is rejected as an unknown field.
        let wire = WireArrayType {
            data_type: WireDataType::F32,
            shape: vec![],
            layout: None,
            memory: WireMemory::Device,
            distribution: Some("sharded".to_owned()),
        };
        let error = serde_json::from_value::<WireArrayType>(serde_json::to_value(&wire).unwrap()).err().unwrap();
        assert_eq!(
            error.to_string(),
            "unknown field `distribution`, expected one of `data_type`, `shape`, `layout`, `memory`",
        );
    }

    #[test]
    fn test_kernel_definition_serialize() {
        let value = Array::scalar(-0.0f32).unwrap();
        let definition = literal_definition(value.clone());
        let decoded = roundtrip(&definition);
        let output = decoded.interpret(vec![], DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS).unwrap();
        assert_eq!(output[0].logical_bytes(), value.logical_bytes());
    }

    #[test]
    fn test_kernel_definition_serialize_saturating_sub() {
        let r#type = ArrayType::new_static(DataType::I8, [4]);
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(r#type.clone(), KernelParameterAccess::ReadOnly).unwrap(),
                whole_array_parameter(r#type.clone(), KernelParameterAccess::ReadOnly).unwrap(),
                whole_array_parameter(r#type.clone(), KernelParameterAccess::WriteOnly).unwrap(),
            ],
        )
        .unwrap();
        let definition = KernelDefinition::trace(operation, |(references, _)| {
            let context = references[0].context();
            let left = context.bind(ReferenceReadOperation::new(), vec![], &[references[0].clone()])?.remove(0);
            let right = context.bind(ReferenceReadOperation::new(), vec![], &[references[1].clone()])?.remove(0);
            let difference = context
                .bind(
                    ArrayIrOperation::Array(ArrayOperation::SaturatingSub(SaturatingSubOperation::new())),
                    vec![],
                    &[left, right],
                )?
                .remove(0);
            context.bind(ReferenceWriteOperation::new(), vec![], &[references[2].clone(), difference])?;
            Ok(())
        })
        .unwrap();
        let decoded = roundtrip(&definition);
        let left = Array::from_elements(r#type.clone(), &[i8::MIN, i8::MAX, 3, -3]).unwrap();
        let right = Array::from_elements(r#type.clone(), &[1i8, -1, 7, -7]).unwrap();
        let expected = Array::from_elements(r#type, &[i8::MIN, i8::MAX, -4, 4]).unwrap();
        assert_eq!(decoded.interpret(vec![left, right], 1), Ok(vec![expected]));
    }

    #[test]
    fn test_kernel_definition_serialize_saturating_sub_mapping() {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let coordinate =
            builder.add_input(DimensionType::new("coordinate", DimensionBounds::non_negative(Some(4)).unwrap()).into());
        let one = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(1).unwrap()));
        let difference = builder
            .add_instruction(
                ArrayIrOperation::Dimension(DimensionOperation::SaturatingSub(SaturatingSubOperation::new())),
                vec![],
                vec![coordinate, one],
                None,
            )
            .unwrap()[0];
        let mapping = BlockMapping::new(
            builder.build(vec![difference], vec![Placeholder], vec![Placeholder]).unwrap(),
            vec![1],
            BoundaryPolicy::InBounds,
        )
        .unwrap();
        let parameter =
            KernelParameter::new(ArrayType::new_static(DataType::I32, [4]), KernelParameterAccess::ReadOnly, mapping)
                .unwrap();
        let call = KernelCallOperation::new(
            Grid::new(vec![GridDimension::new(Dimension::Static(4), GridExecution::Parallel)]).unwrap(),
            vec![parameter],
        )
        .unwrap();
        let decoded = roundtrip(&KernelDefinition::trace(call, |_| Ok(())).unwrap());
        let mapping = decoded.operation().parameters()[0].mapping();
        for (coordinate, expected) in [(0, 0), (1, 0), (3, 2)] {
            let ArrayIrType::Dimension(input_type) = mapping.program().input_types()[0].clone() else { unreachable!() };
            let input = DimensionValue::new(input_type, coordinate).unwrap();
            assert_eq!(mapping.evaluate(&[input], &[4]).unwrap().starts(), &[expected]);
        }
    }

    #[test]
    fn test_kernel_definition_serialize_reduce() {
        #[kernel(crate = "crate")]
        fn sum(
            #[input(data_type = F32, rank = 1)] input: &Array,
            #[output(data_type = F32, shape = [])] output: &mut Array,
        ) {
            output.store(input.load().sum([0]));
        }

        let input = Array::vector(vec![1.0f32, -2.0, 4.0]).unwrap();
        let definition = sum::definition(&input.r#type()).unwrap();
        let decoded = roundtrip(&definition);
        assert_eq!(decoded.interpret(vec![input], 1), Ok(vec![Array::scalar(3.0f32).unwrap()]));
    }

    #[test]
    fn test_kernel_definition_serialize_reference_transforms() {
        let transform =
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 1, 1), ArraySliceAxis::new(0, 2, 1)] };
        let definition = viewed_read_definition(transform.clone(), None);
        let decoded = roundtrip(&definition);
        let operation = decoded.body().entry_region().instructions()[0].operation();
        assert_eq!(operation.reference_access_descriptor(0).unwrap().transforms(), &[transform]);
    }

    #[test]
    fn test_kernel_definition_serialize_dynamic_reference_transforms() {
        let transform = ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic };
        let definition = viewed_read_definition(transform.clone(), Some(1));
        let decoded = roundtrip(&definition);
        let instruction = &decoded.body().entry_region().instructions()[0];
        assert_eq!(instruction.operation().reference_access_descriptor(0).unwrap().transforms(), &[transform]);
        assert_eq!(instruction.operation().reference_access_descriptor(0).unwrap().bindings(), 1..2);
        assert_eq!(instruction.inputs().len(), 2);
    }

    #[test]
    fn test_kernel_definition_serialize_shared_region_reference_transforms() {
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(ArrayType::new_static(DataType::I32, [2, 2]), KernelParameterAccess::ReadOnly)
                    .unwrap(),
            ],
        )
        .unwrap();
        let mut branch = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = branch.add_input(call.parameters()[0].body_type());
        let index = branch.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::I64)));
        branch
            .add_instruction(
                ReferenceReadOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                    axis: 0,
                    index: ArrayReferenceTransformIndex::Dynamic,
                }]),
                vec![],
                vec![reference, index],
                None,
            )
            .unwrap();
        let branch = branch
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = builder.add_input(call.parameters()[0].body_type());
        let index = builder.add_constant(ArrayIrValue::Array(Array::scalar(1i64).unwrap()));
        let predicate = builder.add_constant(ArrayIrValue::Array(Array::scalar(true).unwrap()));
        let branch = builder.import_region(branch.entry_region_ref());
        builder
            .add_instruction(
                ArrayIrOperation::Condition(ConditionOperation::new()),
                vec![branch, branch],
                vec![predicate, reference, index],
                None,
            )
            .unwrap();
        let body = builder.build(vec![], vec![Placeholder], vec![]).unwrap();
        let definition = KernelDefinition::new(call, body).unwrap();
        let decoded = roundtrip(&definition);
        let attached = decoded.body().instructions()[0].regions();
        assert_eq!(attached.len(), 2);
        assert_eq!(attached[0], attached[1]);
        let branch = decoded.body().region_ref(attached[0]).unwrap();
        let instruction = &branch.instructions()[0];
        assert_eq!(instruction.inputs()[1], branch.input_ids()[1]);
        assert_eq!(instruction.operation().reference_access_descriptor(0).unwrap().bindings(), 1..2);
    }

    #[test]
    fn test_kernel_definition_serialize_independent_async_copy_transforms() {
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(ArrayType::new_static(DataType::I32, [2, 2]), KernelParameterAccess::ReadOnly)
                    .unwrap(),
                whole_array_parameter(ArrayType::new_static(DataType::I32, [3, 2]), KernelParameterAccess::ReadWrite)
                    .unwrap(),
            ],
        )
        .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let roots = call.body_input_types().into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let source_index = builder.add_constant(ArrayIrValue::Array(Array::scalar(1i64).unwrap()));
        let destination_index = builder.add_constant(ArrayIrValue::Array(Array::scalar(2i64).unwrap()));
        let operation = AsyncCopyOperation::new()
            .with_source_transforms(vec![ArrayReferenceTransform::Index {
                axis: 0,
                index: ArrayReferenceTransformIndex::Dynamic,
            }])
            .with_destination_transforms(vec![ArrayReferenceTransform::Index {
                axis: 0,
                index: ArrayReferenceTransformIndex::Dynamic,
            }]);
        let token = builder
            .add_instruction(operation, vec![], vec![roots[0], roots[1], source_index, destination_index], None)
            .unwrap()[0];
        builder.add_instruction(WaitOperation, vec![], vec![token], None).unwrap();
        let definition =
            KernelDefinition::new(call, builder.build(vec![], vec![Placeholder; 2], vec![]).unwrap()).unwrap();
        let decoded = roundtrip(&definition);
        let instruction = &decoded.body().entry_region().instructions()[0];
        assert_eq!(instruction.operation().reference_access_descriptor(0).unwrap().bindings(), 2..3);
        assert_eq!(instruction.operation().reference_access_descriptor(1).unwrap().bindings(), 3..4);
    }

    #[test]
    fn test_kernel_definition_serialize_tiled_control_flow() {
        let left =
            Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3]), &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let right = Array::from_elements(
            ArrayType::new_static(DataType::F32, [3, 3]),
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let definition = tiled_matmul::definition(&left.r#type(), &right.r#type()).unwrap();
        let decoded = roundtrip(&definition);
        assert_eq!(
            decoded.interpret(vec![left, right], DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS).unwrap()[0]
                .elements::<f32>()
                .unwrap(),
            vec![30.0, 36.0, 42.0, 66.0, 81.0, 96.0],
        );
        assert_eq!(decoded.body().regions().len(), definition.body().regions().len());
        assert_eq!(
            decoded.body().entry_region().instructions()[0].provenance(),
            definition.body().entry_region().instructions()[0].provenance(),
        );
    }

    #[test]
    fn test_kernel_definition_serialize_grid_execution_and_boundary_policy() {
        let grid = Grid::new(vec![GridDimension::new(Dimension::Static(2), GridExecution::Sequential)]).unwrap();
        let definition: KernelDefinition =
            KernelDefinition::trace(KernelCallOperation::new(grid, vec![]).unwrap(), |_| Ok(())).unwrap();
        let wire = serde_json::to_value(&definition).unwrap();
        assert_eq!(wire["call"]["grid"][0]["execution"], "Sequential");
        let decoded = roundtrip(&definition);
        assert_eq!(decoded.operation().grid().dimensions()[0].execution(), GridExecution::Sequential);

        let left = ArrayType::new_static(DataType::F32, [2, 3]);
        let right = ArrayType::new_static(DataType::F32, [3, 3]);
        let definition = tiled_matmul::definition(&left, &right).unwrap();
        let wire = serde_json::to_value(&definition).unwrap();
        assert_eq!(wire["call"]["parameters"][0]["boundary_policy"], "InBounds");
        assert_eq!(wire["call"]["parameters"][2]["boundary_policy"], "Masked");
        let decoded = roundtrip(&definition);
        assert_eq!(decoded.operation().parameters()[0].mapping().boundary_policy(), BoundaryPolicy::InBounds);
        assert_eq!(decoded.operation().parameters()[2].mapping().boundary_policy(), BoundaryPolicy::Masked);
    }

    #[test]
    fn test_kernel_definition_serialize_in_bounds_arithmetic() {
        #[kernel(crate = "crate")]
        fn affine(
            #[input(data_type = F32, rank = 1)] input: &Array,
            #[output(data_type = F32, shape = [input.shape()[0]], tile = [2], boundary = in_bounds)]
            output: &mut Array,
        ) {
            let [block] = output.tile_index();
            let tiles = input.tiles([2]).pad(0.0);
            output.store((tiles.load([block]) - 1.0f32) * 3.0f32 / 2.0f32);
        }

        let input = Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let definition = affine::definition(&input.r#type()).unwrap();
        let decoded = roundtrip(&definition);
        assert_eq!(
            decoded.interpret(vec![input], DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS),
            Ok(vec![Array::vector(vec![0.0f32, 1.5, 3.0, 4.5]).unwrap()]),
        );
    }

    #[test]
    fn test_kernel_definition_serialize_portable_memory() {
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(r#type.clone(), KernelParameterAccess::ReadOnly).unwrap(),
                whole_array_parameter(r#type.clone(), KernelParameterAccess::WriteOnly).unwrap(),
            ],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
            let context = references[0].context();
            let scratch = context.bind(ScratchOperation::new(r#type.clone(), 4).unwrap(), vec![], &[])?.remove(0);
            let token = context
                .bind(AsyncCopyOperation::new(), vec![], &[references[0].clone(), scratch.clone()])?
                .remove(0);
            context.bind(WaitOperation, vec![], &[token])?;
            let value = context.bind(ReferenceReadOperation::new(), vec![], &[scratch])?.remove(0);
            context.bind(ReferenceWriteOperation::new(), vec![], &[references[1].clone(), value])?;
            Ok(())
        })
        .unwrap();
        let decoded = roundtrip(&definition);
        let value = Array::from_elements(r#type, &[2.0f32, 3.0]).unwrap();
        assert_eq!(decoded.interpret(vec![value.clone()], 1), Ok(vec![value]));
    }

    #[test]
    fn test_kernel_definition_serialize_masked_atomic_memory() {
        let r#type = ArrayType::scalar(DataType::F32);
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![whole_array_parameter(r#type, KernelParameterAccess::ReadWrite).unwrap()],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
            let context = references[0].context();
            let mask = context
                .bind(
                    ArrayIrOperation::Array(ArrayOperation::Constant(ConstantOperation::new(
                        Array::scalar(true).unwrap(),
                    ))),
                    vec![],
                    &[],
                )?
                .remove(0);
            let other = context
                .bind(
                    ArrayIrOperation::Array(ArrayOperation::Constant(ConstantOperation::new(
                        Array::scalar(0.0f32).unwrap(),
                    ))),
                    vec![],
                    &[],
                )?
                .remove(0);
            let value = context
                .bind(MaskedLoadOperation::new(), vec![], &[references[0].clone(), mask.clone(), other.clone()])?
                .remove(0);
            context.bind(ReferenceAtomicAddUpdateOperation::new(), vec![], &[references[0].clone(), value.clone()])?;
            let old = context
                .bind(MaskedSwapOperation::new(), vec![], &[references[0].clone(), value, mask.clone(), other])?
                .remove(0);
            context.bind(MaskedStoreOperation::new(), vec![], &[references[0].clone(), old, mask])?;
            Ok(())
        })
        .unwrap();
        let decoded = roundtrip(&definition);
        assert_eq!(
            decoded.interpret(vec![Array::scalar(2.0f32).unwrap()], 1),
            Ok(vec![Array::scalar(4.0f32).unwrap()]),
        );
    }

    #[test]
    fn test_kernel_definition_serialize_window_mask() {
        let r#type = ArrayType::new_static(DataType::I32, [2]);
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![whole_array_parameter(r#type, KernelParameterAccess::WriteOnly).unwrap()],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
            let context = references[0].context();
            let value = context.lift(ArrayIrValue::Array(Array::vector(vec![5i32, 7])?))?;
            let mask = context.bind(WindowMaskOperation, vec![], &[references[0].clone()])?.remove(0);
            context.bind(MaskedStoreOperation::new(), vec![], &[references[0].clone(), value, mask])?;
            Ok(())
        })
        .unwrap();
        let decoded = roundtrip(&definition);
        assert_eq!(decoded.body().to_string(), definition.body().to_string());
        assert_eq!(decoded.interpret(vec![], 1), Ok(vec![Array::vector(vec![5i32, 7]).unwrap()]));
    }

    #[test]
    fn test_kernel_definition_serialize_elementwise_operations() {
        let real = ArrayType::new_static(DataType::F32, [2]);
        let complex = ArrayType::new_static(DataType::C64, [2]);
        let unary = |operation: ArrayOperation<Array>| (operation, vec![real.clone()], real.clone());
        let binary = |operation: ArrayOperation<Array>| (operation, vec![real.clone(), real.clone()], real.clone());
        for (operation, input_types, output_type) in [
            unary(ArrayOperation::Sign(SignOperation::new())),
            binary(ArrayOperation::Pow(PowOperation::new())),
            unary(ArrayOperation::Sin(SinOperation::new())),
            unary(ArrayOperation::Cos(CosOperation::new())),
            unary(ArrayOperation::Tan(TanOperation::new())),
            binary(ArrayOperation::Atan2(Atan2Operation::new())),
            unary(ArrayOperation::ExpM1(ExpM1Operation::new())),
            unary(ArrayOperation::Ln1p(Ln1pOperation::new())),
            binary(ArrayOperation::LogAddExp(LogAddExpOperation::new())),
            unary(ArrayOperation::Logistic(LogisticOperation::new())),
            unary(ArrayOperation::Erf(ErfOperation::new())),
            unary(ArrayOperation::Floor(FloorOperation::new())),
            unary(ArrayOperation::Ceil(CeilOperation::new())),
            unary(ArrayOperation::Round(RoundOperation::new())),
            (ArrayOperation::Complex(ComplexOperation::new()), vec![real.clone(), real.clone()], complex.clone()),
            (ArrayOperation::Real(RealOperation::new()), vec![complex.clone()], real.clone()),
            (ArrayOperation::Imaginary(ImaginaryOperation::new()), vec![complex.clone()], real.clone()),
            (ArrayOperation::Conjugate(ConjugateOperation::new()), vec![complex.clone()], complex.clone()),
            unary(ArrayOperation::StopGradient(StopGradientOperation::new())),
            unary(ArrayOperation::Tag(TagOperation::new("activation"))),
        ] {
            assert_operations_roundtrip(&operation_definition(operation, input_types, vec![], output_type));
        }
    }

    #[test]
    fn test_kernel_definition_serialize_accuracies() {
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let tolerance = Accuracy::Tolerance(Tolerance::new(-0.0, 1e-3, 2).unwrap());
        for accuracy in [Accuracy::Highest, tolerance] {
            for operation in [
                ArrayOperation::Exp(ExpOperation::new().with_accuracy(accuracy)),
                ArrayOperation::ExpM1(ExpM1Operation::new().with_accuracy(accuracy)),
                ArrayOperation::Log(LogOperation::new().with_accuracy(accuracy)),
                ArrayOperation::Ln1p(Ln1pOperation::new().with_accuracy(accuracy)),
                ArrayOperation::Logistic(LogisticOperation::new().with_accuracy(accuracy)),
                ArrayOperation::Sin(SinOperation::new().with_accuracy(accuracy)),
                ArrayOperation::Cos(CosOperation::new().with_accuracy(accuracy)),
                ArrayOperation::Tan(TanOperation::new().with_accuracy(accuracy)),
                ArrayOperation::Tanh(TanhOperation::new().with_accuracy(accuracy)),
                ArrayOperation::Sqrt(SqrtOperation::new().with_accuracy(accuracy)),
                ArrayOperation::Rsqrt(RsqrtOperation::new().with_accuracy(accuracy)),
            ] {
                let definition = operation_definition(operation, vec![r#type.clone()], vec![], r#type.clone());
                assert_operations_roundtrip(&definition);
            }
        }

        // A tolerance keeps the IEEE bits of its floating-point tolerances, so a negative zero survives.
        let operation = ArrayOperation::Exp(ExpOperation::new().with_accuracy(tolerance));
        let definition = operation_definition(operation, vec![r#type.clone()], vec![], r#type);
        let wire = serde_json::to_value(&definition).unwrap();
        assert_eq!(
            wire["body"]["regions"][0]["instructions"][1]["operation"],
            serde_json::json!({
                "AccurateUnary": {
                    "function": "Exp",
                    "accuracy": {
                        "Tolerance": {
                            "absolute": 0x8000_0000_0000_0000u64,
                            "relative": 1e-3f64.to_bits(),
                            "units_of_least_precision": 2,
                        },
                    },
                },
            }),
        );
    }

    #[test]
    fn test_kernel_definition_serialize_dot_precisions() {
        let left = ArrayType::new_static(DataType::F32, [2, 3]);
        let right = ArrayType::new_static(DataType::F32, [3, 2]);
        let output = ArrayType::new_static(DataType::F32, [2, 2]);
        for precision in [
            DotPrecision::Inputs { lhs: DotInputPrecision::High, rhs: DotInputPrecision::Default },
            DotPrecision::Algorithm(
                DotAlgorithm::new(DataType::BF16, DataType::BF16, DataType::F32)
                    .with_component_counts(1, 2)
                    .with_primitive_operation_count(3)
                    .with_imprecise_accumulation(true),
            ),
            DotPrecision::Preset(DotAlgorithmPreset::F32_F32_F32),
        ] {
            let operation = ArrayOperation::Dot(DotOperation::matmul().with_precision(precision));
            let definition = operation_definition(operation, vec![left.clone(), right.clone()], vec![], output.clone());
            assert_operations_roundtrip(&definition);
        }
    }

    #[test]
    fn test_kernel_definition_serialize_constant_operations() {
        let r#type = ArrayType::new_static(DataType::F32, [2, 3]);
        let indices = ArrayType::new_static(DataType::I32, [2, 3]);
        for (operation, input_types, output_type) in [
            (ArrayOperation::ZeroLike(ZeroLikeOperation::new()), vec![r#type.clone()], r#type.clone()),
            (ArrayOperation::One(OneOperation::new(r#type.clone())), vec![], r#type.clone()),
            (ArrayOperation::OneLike(OneLikeOperation::new()), vec![r#type.clone()], r#type.clone()),
            (ArrayOperation::Iota(IotaOperation::new(indices.clone(), 1).unwrap()), vec![], indices.clone()),
            (
                ArrayOperation::FillLike(FillLikeOperation::new(Array::scalar(-0.0f32).unwrap()).unwrap()),
                vec![r#type.clone()],
                r#type.clone(),
            ),
            (
                ArrayOperation::IotaLike(IotaLikeOperation::new(0, DataType::I32).unwrap()),
                vec![r#type.clone()],
                indices.clone(),
            ),
        ] {
            assert_operations_roundtrip(&operation_definition(operation, input_types, vec![], output_type));
        }
    }

    #[test]
    fn test_kernel_definition_serialize_index_operations() {
        let r#type = ArrayType::new_static(DataType::F32, [2, 3]);
        let indices = ArrayType::new_static(DataType::I32, [2]);
        for (operation, output_type) in [
            (ArrayOperation::ArgMax(ArgMaxOperation::new(1, DataType::I32)), indices.clone()),
            (ArrayOperation::ArgMin(ArgMinOperation::new(1, DataType::I32)), indices),
            (
                ArrayOperation::Cumulative(CumulativeOperation::new(1, CumulativeKind::LogSumExp).with_reverse(true)),
                r#type.clone(),
            ),
            (
                ArrayOperation::Sort(
                    SortOperation::new(0, SortDirection::Descending).with_ordering(SortOrdering::Total),
                ),
                r#type.clone(),
            ),
        ] {
            assert_operations_roundtrip(&operation_definition(operation, vec![r#type.clone()], vec![], output_type));
        }
    }

    #[test]
    fn test_kernel_definition_serialize_shape_operations() {
        let matrix = ArrayType::new_static(DataType::F32, [2, 3]);
        let vector = ArrayType::new_static(DataType::F32, [4]);
        let pair = ArrayType::new_static(DataType::F32, [2]);
        let index = ArrayType::scalar(DataType::I32);
        for (operation, input_types, output_type) in [
            (
                ArrayOperation::Transpose(TransposeOperation::new([1, 0])),
                vec![matrix.clone()],
                ArrayType::new_static(DataType::F32, [3, 2]),
            ),
            (ArrayOperation::Reverse(ReverseOperation::new([-1])), vec![matrix.clone()], matrix.clone()),
            (
                ArrayOperation::Reshape(ReshapeOperation::new([2, 3], [6])),
                vec![matrix.clone()],
                ArrayType::new_static(DataType::F32, [6]),
            ),
            (
                ArrayOperation::Broadcast(BroadcastOperation::new(matrix.clone(), vec![1])),
                vec![ArrayType::new_static(DataType::F32, [3])],
                matrix.clone(),
            ),
            (
                ArrayOperation::BroadcastLike(BroadcastLikeOperation::new(vec![1])),
                vec![ArrayType::new_static(DataType::F32, [3]), matrix.clone()],
                matrix.clone(),
            ),
            (
                ArrayOperation::Pad(PadOperation::new(vec![1], vec![-1], vec![1]).unwrap()),
                vec![pair.clone(), ArrayType::scalar(DataType::F32)],
                ArrayType::new_static(DataType::F32, [3]),
            ),
            (
                ArrayOperation::Concatenate(ConcatenateOperation::<ArrayType>::new(0, 1).unwrap()),
                vec![pair.clone(), pair.clone()],
                vector.clone(),
            ),
            (
                ArrayOperation::Slice(
                    SliceOperation::new(vec![0, 1], vec![Dimension::Static(2), Dimension::Static(3)])
                        .with_strides(vec![1, 2])
                        .unwrap(),
                ),
                vec![matrix.clone()],
                ArrayType::new_static(DataType::F32, [2, 1]),
            ),
            (
                ArrayOperation::UpdateSlice(UpdateSliceOperation::new(vec![1])),
                vec![vector.clone(), pair.clone()],
                vector.clone(),
            ),
            (
                ArrayOperation::DynamicSlice(DynamicSliceOperation::new(vec![2]).with_allow_negative_indices(false)),
                vec![vector.clone(), index.clone()],
                pair.clone(),
            ),
            (
                ArrayOperation::DynamicUpdateSlice(
                    DynamicUpdateSliceOperation::new().with_allow_negative_indices(false),
                ),
                vec![vector.clone(), pair.clone(), index],
                vector.clone(),
            ),
        ] {
            assert_operations_roundtrip(&operation_definition(operation, input_types, vec![], output_type));
        }
    }

    #[test]
    fn test_kernel_definition_serialize_conversion_operations() {
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        for (operation, output_type) in [
            (
                ArrayOperation::ConvertElementType(ConvertElementTypeOperation::new(DataType::I32, true)),
                ArrayType::new_static(DataType::I32, [2]),
            ),
            (ArrayOperation::ReducePrecision(ReducePrecisionOperation::new(5, 10)), r#type.clone()),
            (
                ArrayOperation::Cast(CastOperation::unchecked(r#type.clone(), vec![r#type.clone()]).unwrap()),
                r#type.clone(),
            ),
            (
                ArrayOperation::TransferToMemory(TransferToMemoryOperation::new(Memory::Host { pinned: true })),
                r#type.clone().with_memory(Memory::Host { pinned: true }),
            ),
        ] {
            assert_operations_roundtrip(&operation_definition(operation, vec![r#type.clone()], vec![], output_type));
        }
    }

    #[test]
    fn test_kernel_definition_serialize_array_ir_operations() {
        let matrix = ArrayType::new_static(DataType::F32, [2, 3]);
        let pair = ArrayType::new_static(DataType::F32, [2]);
        let ir = |r#type: &ArrayType| ArrayIrType::Array(r#type.clone());
        let extent = |extent: usize| ArrayIrType::from(DimensionValue::constant(extent).unwrap().r#type().into_owned());
        for (operation, input_types, extents, output_type) in [
            (ArrayIrOperation::One(OneOperation::new(matrix.clone())), vec![], vec![], matrix.clone()),
            (
                ArrayIrOperation::Iota(IotaOperation::new(matrix.clone(), 0).unwrap()),
                vec![],
                vec![],
                matrix.clone(),
            ),
            (
                ArrayIrOperation::Reshape(
                    DynamicReshapeOperation::new().with_input_types(&[ir(&matrix), extent(6)]).unwrap(),
                ),
                vec![matrix.clone()],
                vec![6],
                ArrayType::new_static(DataType::F32, [6]),
            ),
            (
                ArrayIrOperation::Broadcast(DynamicBroadcastOperation::new(vec![0])),
                vec![pair.clone()],
                vec![2, 3],
                matrix.clone(),
            ),
            (
                ArrayIrOperation::Pad(
                    PadOperation::new(vec![1], vec![0], vec![0])
                        .unwrap()
                        .with_input_types(&[ir(&pair), ir(&ArrayType::scalar(DataType::F32)), extent(3)])
                        .unwrap(),
                ),
                vec![pair.clone(), ArrayType::scalar(DataType::F32)],
                vec![3],
                ArrayType::new_static(DataType::F32, [3]),
            ),
            (
                ArrayIrOperation::Concatenate(
                    ConcatenateOperation::<ArrayIrType>::new(0, &[ir(&pair), ir(&pair), extent(4)]).unwrap(),
                ),
                vec![pair.clone(), pair.clone()],
                vec![4],
                ArrayType::new_static(DataType::F32, [4]),
            ),
            (
                ArrayIrOperation::Cast(CastOperation::checked(ir(&pair), vec![ir(&pair)]).unwrap()),
                vec![pair.clone()],
                vec![],
                pair.clone(),
            ),
        ] {
            assert_operations_roundtrip(&operation_definition(
                KernelOperation::Portable(operation),
                input_types,
                extents,
                output_type,
            ));
        }
    }

    #[test]
    fn test_kernel_definition_serialize_dimension_operations() {
        let input_type = ArrayType::new_static(DataType::F32, [3]);
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![whole_array_parameter(input_type.clone(), KernelParameterAccess::ReadOnly).unwrap()],
        )
        .unwrap();
        let size = DimensionSizeOperation::new(&input_type, 0).unwrap();
        let two = DimensionValue::constant(2).unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = builder.add_input(call.body_input_types()[0].clone());
        let constant = builder.add_constant(ArrayIrValue::Dimension(two.clone()));
        let value = builder.add_instruction(ReferenceReadOperation::new(), vec![], vec![reference], None).unwrap()[0];
        let extent = builder
            .add_instruction(ArrayIrOperation::DimensionSize(size.clone()), vec![], vec![value], None)
            .unwrap()[0];
        for operation in [
            ArrayIrOperation::from(DimensionMaxOperation::new(size.output_type(), two.r#type().as_ref()).unwrap()),
            ArrayIrOperation::from(DimensionMinOperation::new(size.output_type(), two.r#type().as_ref()).unwrap()),
            ArrayIrOperation::from(DimensionPowOperation::new(size.output_type(), two.r#type().as_ref()).unwrap()),
        ] {
            builder.add_instruction(operation, vec![], vec![extent, constant], None).unwrap();
        }
        let definition =
            KernelDefinition::new(call, builder.build(vec![], vec![Placeholder], vec![]).unwrap()).unwrap();
        assert_operations_roundtrip(&definition);
    }

    #[test]
    fn test_kernel_definition_serialize_array_control_flow() {
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(r#type.clone(), KernelParameterAccess::ReadOnly).unwrap(),
                whole_array_parameter(r#type.clone(), KernelParameterAccess::WriteOnly).unwrap(),
            ],
        )
        .unwrap();
        let mut branch = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let input = branch.add_input(ArrayIrType::Array(r#type.clone()));
        let branch = branch
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![input],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let input = builder.add_input(call.body_input_types()[0].clone());
        let output = builder.add_input(call.body_input_types()[1].clone());
        let value = builder.add_instruction(ReferenceReadOperation::new(), vec![], vec![input], None).unwrap()[0];
        let predicate = builder.add_constant(ArrayIrValue::Array(Array::scalar(true).unwrap()));
        let branch = builder.import_region(branch.entry_region_ref());
        let result = builder
            .add_instruction(
                ArrayOperation::Condition(ConditionOperation::new()),
                vec![branch, branch],
                vec![predicate, value],
                None,
            )
            .unwrap()[0];
        builder.add_instruction(ReferenceWriteOperation::new(), vec![], vec![output, result], None).unwrap();
        let definition =
            KernelDefinition::new(call, builder.build(vec![], vec![Placeholder; 2], vec![]).unwrap()).unwrap();
        assert_operations_roundtrip(&definition);
    }

    #[test]
    fn test_kernel_definition_serialize_singleton_dimension_bounds() {
        let extent = DimensionVariable::new("programs", DimensionBounds::new(2, Some(3)).unwrap());
        let grid = Grid::new(vec![GridDimension::new(Dimension::Dynamic(extent), GridExecution::Parallel)]).unwrap();
        let definition: KernelDefinition =
            KernelDefinition::trace(KernelCallOperation::new(grid, vec![]).unwrap(), |_| Ok(())).unwrap();
        let decoded = roundtrip(&definition);

        // Decoding creates a fresh nominal identity, so the grid extent is compared by its name and bounds.
        let Dimension::Dynamic(variable) = decoded.operation().grid().dimensions()[0].extent() else {
            panic!("expected a dynamic grid extent");
        };
        assert_eq!(variable.name(), "programs");
        assert_eq!(variable.bounds(), DimensionBounds::new(2, Some(3)).unwrap());
    }

    #[test]
    fn test_kernel_definition_deserialize() {
        let extent = DimensionVariable::new("programs", DimensionBounds::non_negative(Some(9)).unwrap());
        let grid = Grid::new(vec![GridDimension::new(Dimension::Dynamic(extent), GridExecution::Parallel)]).unwrap();
        let definition: KernelDefinition =
            KernelDefinition::trace(KernelCallOperation::new(grid, vec![]).unwrap(), |_| Ok(()))
                .unwrap()
                .specialize_grid(&[2])
                .unwrap();
        let decoded = roundtrip(&definition);
        assert_eq!(decoded.operation().coordinate_types()[0].bounds(), DimensionBounds::non_negative(Some(8)).unwrap());
        assert_eq!(decoded.interpret(vec![], 2), Ok(vec![]));
    }

    #[test]
    fn test_kernel_definition_deserialize_schema() {
        let definition = literal_definition(Array::scalar(1.0f32).unwrap());
        let mut wire: serde_json::Value = serde_json::to_value(definition).unwrap();
        wire["version"] = serde_json::json!(999);
        assert_eq!(
            serde_json::from_value::<KernelDefinition>(wire).unwrap_err().to_string(),
            "unsupported kernel source schema version 999",
        );
    }

    #[test]
    fn test_encoder_value() {
        let r#type = ArrayType::new_static(DataType::F32, [2]).with_layout(Some(StridedLayout::new(vec![-4]).into()));
        let bytes = [0x00, 0x00, 0x00, 0x80, 0x45, 0x23, 0xc1, 0x7f];
        let value = Array::from_logical_bytes(r#type, &bytes).unwrap();
        let mut encoder = Encoder::new(EncodingMode::Transport);
        let wire = encoder.value(&value.clone().into()).unwrap();
        let decoded = Decoder::new(vec![]).value(wire).unwrap();
        let ArrayIrValue::Array(decoded) = decoded else { panic!("expected array") };
        assert_eq!(decoded.r#type(), value.r#type());
        assert_eq!(decoded.logical_bytes(), bytes);
    }

    #[test]
    fn test_encoder_operation_unsupported() {
        let operation = KernelOperation::<NoKernelExtension>::Portable(ArrayIrOperation::Array(ArrayOperation::Print(
            PrintOperation::new("value"),
        )));
        assert_eq!(
            Encoder::new(EncodingMode::Transport).operation(&operation, &[]).err(),
            Some(KernelSerializationError::Unsupported { feature: "`print`".to_owned() }),
        );

        // Requested reduction output shardings are distributed metadata, which the wire records exclude.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let operation = KernelOperation::<NoKernelExtension>::Portable(ArrayIrOperation::Array(
            ArrayOperation::Reduce(
                ReduceOperation::new(vec![0], ReductionKind::Sum)
                    .with_output_sharding(Sharding::replicated(mesh, 0))
                    .unwrap(),
            ),
        ));
        assert!(matches!(
            Encoder::new(EncodingMode::Transport).operation(&operation, &[]),
            Err(KernelSerializationError::Unsupported { feature }) if feature == "`reduce` output sharding",
        ));

        // Key mode records the same payloads through key-only records instead.
        assert!(matches!(
            Encoder::new(EncodingMode::Key).operation(&operation, &[]),
            Ok(WireOperation::KeyOnly(WireKeyOnlyOperation::Debug(_))),
        ));
    }

    #[test]
    fn test_encoder_operation_refined_dimension_metadata() {
        let bounds = DimensionBounds::non_negative(Some(4)).unwrap();
        let left = DimensionType::new("left", bounds);
        let right = DimensionType::new("right", bounds);
        let operation = DimensionMulOperation::new(&left, &right).unwrap();
        let mut renaming = TypeIdentityRenaming::new();
        renaming
            .insert(left.variable().unwrap().clone(), DimensionVariable::new("renamed", bounds))
            .unwrap();
        let operation = operation.rename_type_identities(&renaming).unwrap();
        let operation = KernelOperation::<NoKernelExtension>::Portable(ArrayIrOperation::Dimension(
            DimensionOperation::Mul(operation),
        ));
        assert_eq!(
            Encoder::new(EncodingMode::Transport).operation(&operation, &[]).err(),
            Some(KernelSerializationError::Unsupported {
                feature: "`dimension_mul` cached inference metadata".to_owned(),
            }),
        );

        // Key mode records the cached metadata explicitly, without the diagnostic output name.
        assert!(matches!(
            Encoder::new(EncodingMode::Key).operation(&operation, &[]),
            Ok(WireOperation::KeyOnly(WireKeyOnlyOperation::DimensionArithmetic {
                operation: "dimension_mul",
                left: WireDimension::Dynamic(0),
                right: WireDimension::Dynamic(1),
                output_lower: 0,
                output_upper: Some(10),
                ..
            })),
        ));
    }

    #[test]
    fn test_decoder_value_invalid_literal_size() {
        let wire = WireValue::Array {
            r#type: WireArrayType {
                data_type: WireDataType::F32,
                shape: vec![WireDimension::Static(usize::MAX)],
                layout: None,
                memory: WireMemory::Device,
                distribution: None,
            },
            bytes: vec![],
        };
        assert_eq!(
            Decoder::new(vec![]).value(wire).unwrap_err(),
            KernelSerializationError::Invalid {
                message: format!("dimension extent {} exceeds backend width", usize::MAX),
            },
        );
        let wire = WireValue::Array {
            r#type: WireArrayType {
                data_type: WireDataType::F32,
                shape: vec![],
                layout: None,
                memory: WireMemory::Device,
                distribution: None,
            },
            bytes: vec![0],
        };
        assert_eq!(
            Decoder::new(vec![]).value(wire).unwrap_err(),
            KernelSerializationError::Invalid { message: "literal expects 4 bytes but received 1".to_owned() },
        );
    }

    #[test]
    fn test_decoder_value_literal_storage_budget() {
        let wire = WireValue::Array {
            r#type: WireArrayType {
                data_type: WireDataType::U8,
                shape: vec![WireDimension::Static(2)],
                layout: Some(WireLayout::Strided(vec![MAXIMUM_LITERAL_BYTES as isize])),
                memory: WireMemory::Device,
                distribution: None,
            },
            bytes: vec![1, 2],
        };
        assert_eq!(
            Decoder::new(vec![]).value(wire).unwrap_err(),
            KernelSerializationError::Invalid {
                message: format!("literal storage exceeds {MAXIMUM_LITERAL_BYTES} bytes"),
            },
        );
    }

    #[test]
    fn test_wire_provenance_decode() {
        // The encoder emits a tree, in which every node is referenced exactly once, and decoding restores it.
        let provenance = Provenance::fused([
            Provenance::scope(ProvenanceScope::new("left"), Provenance::unknown()),
            Provenance::scope(ProvenanceScope::new("right"), Provenance::unknown()),
        ]);
        assert_eq!(WireProvenance::encode(&provenance, 0).unwrap().decode(), Ok(provenance));

        // A document that shares one node between two parents is rejected, because its expansion can be
        // exponentially larger than the document.
        let wire = WireProvenance {
            nodes: vec![
                WireProvenanceNode::Unknown,
                WireProvenanceNode::Scope { name: "left".to_owned(), origin: 0 },
                WireProvenanceNode::Scope { name: "right".to_owned(), origin: 0 },
                WireProvenanceNode::Fused(vec![1, 2]),
            ],
            root: 3,
        };
        assert_eq!(
            wire.decode(),
            Err(KernelSerializationError::Invalid {
                message: "provenance node 0 is referenced more than once".to_owned(),
            }),
        );

        // An edge may only reference an earlier node.
        let wire = WireProvenance {
            nodes: vec![WireProvenanceNode::Scope { name: "scope".to_owned(), origin: 1 }, WireProvenanceNode::Unknown],
            root: 0,
        };
        assert_eq!(
            wire.decode(),
            Err(KernelSerializationError::Invalid {
                message: "provenance edge must reference an earlier node".to_owned(),
            }),
        );
    }

    #[test]
    fn test_bounded_collection_deserialize() {
        let sequence = serde::de::value::SeqDeserializer::<_, serde::de::value::Error>::new([1u8, 2, 3].into_iter());
        let result = BoundedCollection::<u8, 2>::deserialize(sequence);
        assert_eq!(result.err().unwrap().to_string(), "kernel source sequence exceeds 2 entries");
    }

    #[test]
    fn test_validate_encoded_records() {
        assert!(matches!(validate_encoded_records(MAXIMUM_RECORDS), Ok(())));
        assert!(matches!(
            validate_encoded_records(MAXIMUM_RECORDS + 1),
            Err(KernelSerializationError::Unsupported { feature }) if feature == "record count exceeding 1000000",
        ));
    }
}
