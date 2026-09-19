//! Checked, format-independent serialization of portable kernel source.
//!
//! This schema covers local array types, exact constants, elementwise arithmetic, generalized dot, bounded control
//! flow, dimension arithmetic, canonical reference updates and portable kernel memory operations. Distributed types,
//! nested kernel calls and other operation payloads report explicit eligibility errors. Transport records reconstruct
//! ordinary programs; they are never interpreted. Decoding revalidates the region DAG and kernel boundary. Executable
//! initialization and adapter admission remain separate requirements, just as for an explicitly built definition.
//!
//! The array subset is constant, zero, add/subtract/multiply/divide/remainder, negate/absolute/minimum/maximum,
//! exp/log/sqrt/rsqrt/tanh, Boolean not/and/or/xor, select, compare and dot. Dimension arithmetic supports add,
//! subtract, multiply, floor division and remainder when canonical reconstruction preserves its complete cached
//! metadata; scalar conversions, conditions and while loops are also supported. Reference allocation/read/write/swap,
//! freeze/add-update/atomic-add-update, scratch, tile load, masked load/store/swap and async copy/wait are supported.
//! Reference views, other operation payloads, distributed metadata and dynamically shaped explicit layouts report
//! eligibility errors. Sequence sizes are bounded before allocation from size hints; all literal storage together
//! may consume at most 64 MiB after applying layouts. Diagnostic provenance uses a checked flat arena.

use std::cell::Cell;
use std::collections::HashMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

use crate::arrays::{
    Array, ArrayAddressing, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, DataType,
    Dimension, DimensionBounds, DimensionOperation, DimensionType, DimensionValue, DimensionVariable, Layout,
    MAX_DIMENSION_EXTENT, Memory, Shape, StridedLayout, Tile, TileDimension, TiledLayout,
};
use crate::contexts::EagerContext;
use crate::kernels::calls::{KernelCallOperation, KernelDefinition, KernelError, KernelParameter};
use crate::kernels::grids::{Grid, GridDimension, GridExecution};
use crate::kernels::indexing::TileLoadOperation;
use crate::kernels::mappings::{BlockMapping, BoundaryPolicy};
use crate::kernels::memory::{
    AsyncCopyOperation, MaskedLoadOperation, MaskedStoreOperation, MaskedSwapOperation, ScratchOperation, WaitOperation,
};
use crate::kernels::operations::KernelOperation;
use crate::kernels::validation::KernelParameterAccess;
use crate::operations::{
    AbsOperation, AddOperation, AndOperation, CompareOperation, ComparisonDirection, ConditionOperation,
    ConstantOperation, DimensionAddOperation, DimensionDivFloorOperation, DimensionFromScalarOperation,
    DimensionMulOperation, DimensionRemOperation, DimensionSubOperation, DimensionToScalarOperation, DivOperation,
    DotDimensionNumbers, DotOperation, ExpOperation, LogOperation, MaxOperation, MinOperation, MulOperation,
    NegOperation, NotOperation, OrOperation, ReferenceAddUpdateOperation, ReferenceAtomicAddUpdateOperation,
    ReferenceFreezeOperation, ReferenceNewOperation, ReferenceReadOperation, ReferenceSwapOperation,
    ReferenceWriteOperation, RemOperation, RsqrtOperation, SelectOperation, SqrtOperation, SubOperation, TanhOperation,
    WhileOperation, XorOperation, ZeroOperation,
};
use crate::parameters::Placeholder;
use crate::programs::{
    Atom, AtomId, FlatProgram, Instruction, Operation, Program, ProgramError, Provenance, ProvenanceScope,
    ReferenceType, Region, RegionId, Typed,
};

/// Errors admitting or reconstructing a portable serialized kernel.
#[derive(Debug, Error)]
pub enum KernelSerializationError {
    #[error("unsupported kernel source schema version {version}")]
    Schema { version: u32 },

    #[error("kernel source serialization does not support `{feature}`")]
    Unsupported { feature: String },

    #[error("invalid serialized kernel source: {message}")]
    Invalid { message: String },

    #[error(transparent)]
    Program(#[from] ProgramError),

    #[error(transparent)]
    Kernel(#[from] KernelError),
}

/// Independent transport schema version; it is not the semantic-key encoding version.
const SOURCE_SCHEMA_VERSION: u32 = 1;

/// Maximum total physical bytes allocated for decoded literals, including layout padding.
const MAXIMUM_LITERAL_BYTES: usize = 64 * 1024 * 1024;

/// Maximum number of records in one transport collection.
const MAXIMUM_RECORDS: usize = 1_000_000;

/// A canonical identity table shared by metadata and every region in the document.
#[derive(Default)]
struct Encoder {
    variables: Vec<WireVariable>,
    identities: HashMap<DimensionVariable, usize>,
    literal_bytes: usize,
}

/// Canonical identities already reconstructed from checked bounds.
struct Decoder {
    variables: Vec<DimensionVariable>,
    literal_bytes: Cell<usize>,
}

/// Versioned transport envelope; all records are private and carry no execution behavior.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireDefinition {
    version: u32,
    #[serde(deserialize_with = "bounded_records")]
    variables: Vec<WireVariable>,
    call: WireCall,
    body: WireProgram,
}

/// Name and bounds of one nominal dimension identity.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireVariable {
    name: String,
    lower: usize,
    upper: Option<usize>,
}

/// Canonical dimension occurrence.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum WireDimension {
    Static(usize),
    Dynamic(usize),
}

/// Canonical array type descriptor with distributed metadata explicitly excluded.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireArrayType {
    data_type: WireDataType,
    #[serde(deserialize_with = "bounded_records")]
    shape: Vec<WireDimension>,
    layout: Option<WireLayout>,
    memory: WireMemory,
}

/// Lossless local layout metadata.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum WireLayout {
    Strided(#[serde(deserialize_with = "bounded_records")] Vec<isize>),
    Tiled {
        #[serde(deserialize_with = "bounded_records")]
        minor_to_major: Vec<usize>,
        #[serde(deserialize_with = "bounded_tiles")]
        tiles: Vec<Vec<Option<usize>>>,
    },
}

/// Local memory placement.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum WireMemory {
    Device,
    Host { pinned: bool },
}

/// Canonical value type occurrence.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum WireType {
    Array(WireArrayType),
    Dimension(usize),
    Reference(WireArrayType),
}

/// Exact literal payload; reference storage is never serialized.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum WireValue {
    Array {
        r#type: WireArrayType,
        #[serde(deserialize_with = "bounded_bytes")]
        bytes: Vec<u8>,
    },
    Dimension {
        identity: usize,
        extent: usize,
    },
}

/// Region-local atom record.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum WireAtom {
    Constant(WireValue),
    Variable(WireType),
}

/// Diagnostic provenance retained as a flat arena with earlier-node edges.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireProvenance {
    #[serde(deserialize_with = "bounded_records")]
    nodes: Vec<WireProvenanceNode>,
    root: usize,
}

/// One diagnostic provenance node; this transport has no recursively nested records.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum WireProvenanceNode {
    Unknown,
    Scope { name: String, origin: usize },
    Fused(#[serde(deserialize_with = "bounded_records")] Vec<usize>),
}

/// Instruction indices retain shared region edges without expanding the region graph into a tree.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireInstruction {
    operation: WireOperation,
    #[serde(deserialize_with = "bounded_records")]
    inputs: Vec<usize>,
    #[serde(deserialize_with = "bounded_records")]
    outputs: Vec<usize>,
    #[serde(deserialize_with = "bounded_records")]
    regions: Vec<usize>,
    provenance: WireProvenance,
}

/// One canonical region, sealed after all its children during decoding.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRegion {
    #[serde(deserialize_with = "bounded_records")]
    atoms: Vec<WireAtom>,
    #[serde(deserialize_with = "bounded_records")]
    inputs: Vec<usize>,
    #[serde(deserialize_with = "bounded_records")]
    outputs: Vec<usize>,
    #[serde(deserialize_with = "bounded_records")]
    instructions: Vec<WireInstruction>,
}

/// Flat-parameter program and its shared arena.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireProgram {
    #[serde(deserialize_with = "bounded_records")]
    regions: Vec<WireRegion>,
    entry: usize,
}

/// Full call metadata, including coordinate identities retained by specialization.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCall {
    #[serde(deserialize_with = "bounded_records")]
    grid: Vec<WireGridDimension>,
    #[serde(deserialize_with = "bounded_records")]
    parameters: Vec<WireParameter>,
    #[serde(deserialize_with = "bounded_records")]
    coordinates: Vec<usize>,
    #[serde(deserialize_with = "bounded_records")]
    prefetch: Vec<WireArrayType>,
}

/// One named grid dimension.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireGridDimension {
    extent: WireDimension,
    name: Option<String>,
    sequential: bool,
}

/// Canonical access intent and block mapping.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireParameter {
    r#type: WireArrayType,
    access: WireAccess,
    mapping: WireProgram,
    #[serde(deserialize_with = "bounded_records")]
    block_shape: Vec<usize>,
    masked: bool,
}

/// Explicit boundary access vocabulary.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum WireAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

/// Stable explicit data-type tags, independent of Rust discriminants and diagnostic rendering.
#[derive(Copy, Clone, Serialize, Deserialize)]
enum WireDataType {
    Token,
    Zero,
    Boolean,
    I1,
    I2,
    I4,
    I8,
    I16,
    I32,
    I64,
    U1,
    U2,
    U4,
    U8,
    U16,
    U32,
    U64,
    F4E2M1FN,
    F6E2M3FN,
    F6E3M2FN,
    F8E3M4,
    F8E4M3,
    F8E4M3FN,
    F8E4M3FNUZ,
    F8E4M3B11FNUZ,
    F8E5M2,
    F8E5M2FNUZ,
    F8E8M0FNU,
    BF16,
    F16,
    F32,
    F64,
    C64,
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

/// Eligible operation payloads. These records only reconstruct the existing canonical operation implementations.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum WireOperation {
    ArrayConstant(WireValue),
    DimensionConstant(WireValue),
    Zero(WireArrayType),
    MixedZero(WireArrayType),
    Compare(WireComparison),
    DimensionCompare(WireComparison),
    Dot {
        #[serde(deserialize_with = "bounded_dot_dimensions")]
        dimensions: [Vec<usize>; 4],
        accumulation: Option<WireDataType>,
    },
    While {
        bound: Option<usize>,
    },
    Condition,
    DimensionFromScalar(usize),
    DimensionToScalar,
    TileLoad {
        #[serde(deserialize_with = "bounded_records")]
        block_shape: Vec<usize>,
        masked: bool,
    },
    Scratch {
        referent: WireArrayType,
        alignment: usize,
    },
    MaskedLoad,
    MaskedStore,
    MaskedSwap,
    AsyncCopy,
    Wait,
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Neg,
    Abs,
    Min,
    Max,
    Exp,
    Log,
    Sqrt,
    Rsqrt,
    Tanh,
    Not,
    And,
    Or,
    Xor,
    Select,
    ReferenceNew,
    ReferenceRead,
    ReferenceWrite,
    ReferenceSwap,
    ReferenceFreeze,
    ReferenceAddUpdate,
    ReferenceAtomicAddUpdate,
    DimensionAdd {
        left: usize,
        right: usize,
    },
    DimensionSub {
        left: usize,
        right: usize,
    },
    DimensionMul {
        left: usize,
        right: usize,
    },
    DimensionDivFloor {
        left: usize,
        right: usize,
    },
    DimensionRem {
        left: usize,
        right: usize,
    },
}

/// Explicit comparison predicate.
#[derive(Copy, Clone, Serialize, Deserialize)]
enum WireComparison {
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
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

impl Serialize for KernelDefinition {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut encoder = Encoder::default();
        let call = encoder.call(self.operation()).map_err(serde::ser::Error::custom)?;
        let body =
            encoder.program(self.body(), |operation| Ok(operation.clone())).map_err(serde::ser::Error::custom)?;
        check_records(encoder.variables.len()).map_err(serde::ser::Error::custom)?;
        WireDefinition { version: SOURCE_SCHEMA_VERSION, variables: encoder.variables, call, body }
            .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for KernelDefinition {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = WireDefinition::deserialize(deserializer)?;
        wire.decode().map_err(serde::de::Error::custom)
    }
}

impl WireDefinition {
    /// Reconstructs canonical identities before any payload or program references them.
    fn decode(self) -> Result<KernelDefinition, KernelSerializationError> {
        if self.version != SOURCE_SCHEMA_VERSION {
            return Err(KernelSerializationError::Schema { version: self.version });
        }
        check_records(self.variables.len())?;
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

impl Encoder {
    /// Registers a nominal identity once across the whole document.
    fn identity(&mut self, variable: &DimensionVariable) -> usize {
        *self.identities.entry(variable.clone()).or_insert_with(|| {
            let index = self.variables.len();
            self.variables.push(WireVariable {
                name: variable.name().to_owned(),
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

    /// Encodes local array metadata and rejects distributed ownership.
    fn array_type(&mut self, r#type: &ArrayType) -> Result<WireArrayType, KernelSerializationError> {
        if r#type.sharding().is_some() || !r#type.unreduced_axes().is_empty() || !r#type.reduced_axes().is_empty() {
            return Err(unsupported("distributed array type"));
        }
        if r#type.layout().is_some()
            && r#type.shape().dimensions().iter().any(|dimension| matches!(dimension, Dimension::Dynamic(_)))
        {
            return Err(unsupported("dynamically shaped array with explicit layout"));
        }
        let layout = r#type.layout().map(|layout| match layout {
            Layout::Strided(layout) => WireLayout::Strided(layout.strides().to_vec()),
            Layout::Tiled(layout) => WireLayout::Tiled {
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
        });
        Ok(WireArrayType {
            data_type: r#type.data_type().into(),
            shape: r#type.shape().dimensions().iter().map(|dimension| self.dimension(dimension)).collect(),
            layout,
            memory: match r#type.memory() {
                Memory::Device => WireMemory::Device,
                Memory::Host { pinned } => WireMemory::Host { pinned },
            },
        })
    }

    /// Encodes a member type in the canonical array universe.
    fn r#type(&mut self, r#type: &ArrayIrType) -> Result<WireType, KernelSerializationError> {
        Ok(match r#type {
            ArrayIrType::Array(r#type) => WireType::Array(self.array_type(r#type)?),
            ArrayIrType::Dimension(r#type) => WireType::Dimension(self.identity(r#type.variable())),
            ArrayIrType::Reference(r#type) => WireType::Reference(self.array_type(r#type.referent())?),
        })
    }

    /// Preserves literal data bits, including signed zero and NaN payloads.
    fn value(&mut self, value: &ArrayIrValue<Array>) -> Result<WireValue, KernelSerializationError> {
        Ok(match value {
            ArrayIrValue::Array(value) => {
                let addressing = ArrayAddressing::new(value.r#type().into_owned())?;
                self.literal_bytes = self
                    .literal_bytes
                    .checked_add(addressing.storage_byte_len())
                    .ok_or_else(|| invalid("literal storage size overflows"))?;
                if self.literal_bytes > MAXIMUM_LITERAL_BYTES || addressing.element_count() > MAXIMUM_LITERAL_BYTES {
                    return Err(unsupported("literal storage exceeds source allocation budget"));
                }
                WireValue::Array { r#type: self.array_type(&value.r#type())?, bytes: value.logical_bytes() }
            }
            ArrayIrValue::Dimension(value) => {
                WireValue::Dimension { identity: self.identity(value.r#type().variable()), extent: value.extent() }
            }
            ArrayIrValue::Reference(_) => return Err(unsupported("reference literal")),
        })
    }

    /// Encodes a flat canonical program while preserving arena order and shared edges.
    fn program<O: Operation<Type = ArrayIrType>, F: Fn(&O) -> Result<KernelOperation, KernelSerializationError>>(
        &mut self,
        program: &FlatProgram<EagerContext<ArrayIrValue<Array>, O>>,
        lift: F,
    ) -> Result<WireProgram, KernelSerializationError> {
        check_records(program.regions().len())?;
        let regions = program
            .regions()
            .iter()
            .map(|region| {
                check_records(region.atoms().len())?;
                check_records(region.instructions().len())?;
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
                        Ok(WireInstruction {
                            operation: self.operation(&lift(instruction.operation())?)?,
                            inputs: instruction.inputs().iter().map(|id| id.index()).collect(),
                            outputs: instruction.outputs().iter().map(|id| id.index()).collect(),
                            regions: instruction.regions().iter().map(|id| id.index()).collect(),
                            provenance: WireProvenance::encode(instruction.provenance(), 0)?,
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

    /// Encodes checked call metadata using the same identity table as its body.
    fn call(&mut self, call: &KernelCallOperation) -> Result<WireCall, KernelSerializationError> {
        let grid = call
            .grid()
            .dimensions()
            .iter()
            .map(|dimension| WireGridDimension {
                extent: self.dimension(dimension.extent()),
                name: dimension.name().map(str::to_owned),
                sequential: dimension.execution() == GridExecution::Sequential,
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
                    mapping: self.program(parameter.mapping().program(), |operation| Ok(operation.clone().into()))?,
                    block_shape: parameter.mapping().block_shape().to_vec(),
                    masked: parameter.mapping().boundary_policy() == BoundaryPolicy::Masked,
                })
            })
            .collect::<Result<_, KernelSerializationError>>()?;
        let coordinates = call.coordinate_types().iter().map(|r#type| self.identity(r#type.variable())).collect();
        let prefetch = call.prefetch_types().iter().map(|r#type| self.array_type(r#type)).collect::<Result<_, _>>()?;
        Ok(WireCall { grid, parameters, coordinates, prefetch })
    }

    /// Converts eligible payloads structurally; no diagnostic rendering is parsed.
    fn operation(&mut self, operation: &KernelOperation) -> Result<WireOperation, KernelSerializationError> {
        Ok(match operation {
            KernelOperation::Portable(operation) => match operation {
                ArrayIrOperation::Array(operation) => match operation {
                    ArrayOperation::Constant(operation) => {
                        WireOperation::ArrayConstant(self.value(&operation.value().clone().into())?)
                    }
                    ArrayOperation::Zero(operation) => WireOperation::Zero(self.array_type(operation.r#type())?),
                    ArrayOperation::Compare(operation) => WireOperation::Compare(operation.direction().into()),
                    ArrayOperation::Dot(operation) => {
                        if operation.output_sharding().is_some() {
                            return Err(unsupported("dot output sharding"));
                        }
                        let dimensions = operation.dimensions();
                        WireOperation::Dot {
                            dimensions: [
                                dimensions.lhs_contracting_dimensions().to_vec(),
                                dimensions.rhs_contracting_dimensions().to_vec(),
                                dimensions.lhs_batching_dimensions().to_vec(),
                                dimensions.rhs_batching_dimensions().to_vec(),
                            ],
                            accumulation: operation.accumulation_type().map(Into::into),
                        }
                    }
                    ArrayOperation::Add(_) => WireOperation::Add,
                    ArrayOperation::Sub(_) => WireOperation::Sub,
                    ArrayOperation::Mul(_) => WireOperation::Mul,
                    ArrayOperation::Div(_) => WireOperation::Div,
                    ArrayOperation::Rem(_) => WireOperation::Rem,
                    ArrayOperation::Neg(_) => WireOperation::Neg,
                    ArrayOperation::Abs(_) => WireOperation::Abs,
                    ArrayOperation::Min(_) => WireOperation::Min,
                    ArrayOperation::Max(_) => WireOperation::Max,
                    ArrayOperation::Exp(_) => WireOperation::Exp,
                    ArrayOperation::Log(_) => WireOperation::Log,
                    ArrayOperation::Sqrt(_) => WireOperation::Sqrt,
                    ArrayOperation::Rsqrt(_) => WireOperation::Rsqrt,
                    ArrayOperation::Tanh(_) => WireOperation::Tanh,
                    ArrayOperation::Not(_) => WireOperation::Not,
                    ArrayOperation::And(_) => WireOperation::And,
                    ArrayOperation::Or(_) => WireOperation::Or,
                    ArrayOperation::Xor(_) => WireOperation::Xor,
                    ArrayOperation::Select(_) => WireOperation::Select,
                    operation => return Err(unsupported(operation.name())),
                },
                ArrayIrOperation::Dimension(operation) => match operation {
                    DimensionOperation::Constant(operation) => {
                        WireOperation::DimensionConstant(self.value(&operation.value().clone().into())?)
                    }
                    DimensionOperation::Add(operation) => {
                        // Refinement may retain cached assertion effects and diagnostic names from wider operands.
                        let reconstructed =
                            DimensionAddOperation::new(operation.left_type(), operation.right_type())
                                .map_err(|_| unsupported(format!("{} cached inference metadata", operation.name())))?;
                        if *operation != reconstructed {
                            return Err(unsupported(format!("{} cached inference metadata", operation.name())));
                        }
                        WireOperation::DimensionAdd {
                            left: self.identity(operation.left_type().variable()),
                            right: self.identity(operation.right_type().variable()),
                        }
                    }
                    DimensionOperation::Sub(operation) => {
                        // Refinement may retain cached assertion effects and diagnostic names from wider operands.
                        let reconstructed =
                            DimensionSubOperation::new(operation.left_type(), operation.right_type())
                                .map_err(|_| unsupported(format!("{} cached inference metadata", operation.name())))?;
                        if *operation != reconstructed {
                            return Err(unsupported(format!("{} cached inference metadata", operation.name())));
                        }
                        WireOperation::DimensionSub {
                            left: self.identity(operation.left_type().variable()),
                            right: self.identity(operation.right_type().variable()),
                        }
                    }
                    DimensionOperation::Mul(operation) => {
                        // Refinement may retain cached assertion effects and diagnostic names from wider operands.
                        let reconstructed =
                            DimensionMulOperation::new(operation.left_type(), operation.right_type())
                                .map_err(|_| unsupported(format!("{} cached inference metadata", operation.name())))?;
                        if *operation != reconstructed {
                            return Err(unsupported(format!("{} cached inference metadata", operation.name())));
                        }
                        WireOperation::DimensionMul {
                            left: self.identity(operation.left_type().variable()),
                            right: self.identity(operation.right_type().variable()),
                        }
                    }
                    DimensionOperation::DivFloor(operation) => {
                        // Refinement may retain cached assertion effects and diagnostic names from wider operands.
                        let reconstructed =
                            DimensionDivFloorOperation::new(operation.left_type(), operation.right_type())
                                .map_err(|_| unsupported(format!("{} cached inference metadata", operation.name())))?;
                        if *operation != reconstructed {
                            return Err(unsupported(format!("{} cached inference metadata", operation.name())));
                        }
                        WireOperation::DimensionDivFloor {
                            left: self.identity(operation.left_type().variable()),
                            right: self.identity(operation.right_type().variable()),
                        }
                    }
                    DimensionOperation::Rem(operation) => {
                        // Refinement may retain cached assertion effects and diagnostic names from wider operands.
                        let reconstructed =
                            DimensionRemOperation::new(operation.left_type(), operation.right_type())
                                .map_err(|_| unsupported(format!("{} cached inference metadata", operation.name())))?;
                        if *operation != reconstructed {
                            return Err(unsupported(format!("{} cached inference metadata", operation.name())));
                        }
                        WireOperation::DimensionRem {
                            left: self.identity(operation.left_type().variable()),
                            right: self.identity(operation.right_type().variable()),
                        }
                    }
                    operation => return Err(unsupported(operation.name())),
                },
                ArrayIrOperation::Zero(operation) => WireOperation::MixedZero(self.array_type(operation.r#type())?),
                ArrayIrOperation::Compare(operation) => WireOperation::DimensionCompare(operation.direction().into()),
                ArrayIrOperation::While(operation) => WireOperation::While { bound: operation.iteration_bound() },
                ArrayIrOperation::Condition(_) => WireOperation::Condition,
                ArrayIrOperation::DimensionFromScalar(operation) => {
                    WireOperation::DimensionFromScalar(self.identity(operation.result_type().variable()))
                }
                ArrayIrOperation::DimensionToScalar(_) => WireOperation::DimensionToScalar,
                ArrayIrOperation::ReferenceNew(_) => WireOperation::ReferenceNew,
                ArrayIrOperation::ReferenceRead(_) => WireOperation::ReferenceRead,
                ArrayIrOperation::ReferenceWrite(_) => WireOperation::ReferenceWrite,
                ArrayIrOperation::ReferenceSwap(_) => WireOperation::ReferenceSwap,
                ArrayIrOperation::ReferenceFreeze(_) => WireOperation::ReferenceFreeze,
                ArrayIrOperation::ReferenceAddUpdate(_) => WireOperation::ReferenceAddUpdate,
                ArrayIrOperation::ReferenceAtomicAddUpdate(_) => WireOperation::ReferenceAtomicAddUpdate,
                operation => return Err(unsupported(operation.name())),
            },
            KernelOperation::TileLoad(operation) => WireOperation::TileLoad {
                block_shape: operation.block_shape().to_vec(),
                masked: operation.boundary_policy() == BoundaryPolicy::Masked,
            },
            KernelOperation::Scratch(operation) => WireOperation::Scratch {
                referent: self.array_type(operation.referent())?,
                alignment: operation.alignment(),
            },
            KernelOperation::MaskedLoad(_) => WireOperation::MaskedLoad,
            KernelOperation::MaskedStore(_) => WireOperation::MaskedStore,
            KernelOperation::MaskedSwap(_) => WireOperation::MaskedSwap,
            KernelOperation::AsyncCopy(_) => WireOperation::AsyncCopy,
            KernelOperation::Wait(_) => WireOperation::Wait,
            KernelOperation::Call(_) => return Err(unsupported("nested kernel call")),
            KernelOperation::Extension(extension) => match *extension {},
        })
    }
}

impl Decoder {
    /// Starts one document-wide physical literal allocation budget.
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

    /// Validates static extents before passing them to type constructors.
    fn dimension(&self, dimension: WireDimension) -> Result<Dimension, KernelSerializationError> {
        match dimension {
            WireDimension::Static(extent) if extent <= MAX_DIMENSION_EXTENT => Ok(Dimension::Static(extent)),
            WireDimension::Static(extent) => Err(invalid(format!("dimension extent {extent} exceeds backend width"))),
            WireDimension::Dynamic(index) => Ok(self.identity(index)?.to_dimension()),
        }
    }

    /// Reconstructs canonical local type metadata; literals subsequently check physical allocation size.
    fn array_type(&self, wire: WireArrayType) -> Result<ArrayType, KernelSerializationError> {
        check_records(wire.shape.len())?;
        let shape =
            Shape::new(wire.shape.into_iter().map(|dimension| self.dimension(dimension)).collect::<Result<_, _>>()?);
        let layout = wire.layout.map(|layout| match layout {
            WireLayout::Strided(strides) => Layout::Strided(StridedLayout::new(strides)),
            WireLayout::Tiled { minor_to_major, tiles } => Layout::Tiled(TiledLayout::new(
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
        });
        let memory = match wire.memory {
            WireMemory::Device => Memory::Device,
            WireMemory::Host { pinned } => Memory::Host { pinned },
        };
        let r#type = ArrayType::new(wire.data_type.into(), shape).with_layout(layout).with_memory(memory);
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
            WireType::Dimension(index) => self.identity(index)?.into(),
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
                        bytes.len()
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
            WireValue::Dimension { identity, extent } => {
                DimensionValue::new(self.identity(identity)?, extent).map_err(ProgramError::from)?.into()
            }
        })
    }

    /// Reconstructs only explicitly eligible operation payloads through their canonical constructors.
    fn operation(&self, wire: WireOperation) -> Result<KernelOperation, KernelSerializationError> {
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
            WireOperation::DimensionFromScalar(index) => ArrayIrOperation::DimensionFromScalar(
                DimensionFromScalarOperation::new(self.identity(index)?.variable().clone()),
            )
            .into(),
            WireOperation::DimensionToScalar => ArrayIrOperation::DimensionToScalar(DimensionToScalarOperation).into(),
            WireOperation::Add => ArrayIrOperation::Array(ArrayOperation::Add(AddOperation::new())).into(),
            WireOperation::Sub => ArrayIrOperation::Array(ArrayOperation::Sub(SubOperation::new())).into(),
            WireOperation::Mul => ArrayIrOperation::Array(ArrayOperation::Mul(MulOperation::new())).into(),
            WireOperation::Div => ArrayIrOperation::Array(ArrayOperation::Div(DivOperation::new())).into(),
            WireOperation::Rem => ArrayIrOperation::Array(ArrayOperation::Rem(RemOperation::new())).into(),
            WireOperation::Neg => ArrayIrOperation::Array(ArrayOperation::Neg(NegOperation::new())).into(),
            WireOperation::Abs => ArrayIrOperation::Array(ArrayOperation::Abs(AbsOperation::new())).into(),
            WireOperation::Min => ArrayIrOperation::Array(ArrayOperation::Min(MinOperation::new())).into(),
            WireOperation::Max => ArrayIrOperation::Array(ArrayOperation::Max(MaxOperation::new())).into(),
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
                DimensionAddOperation::new(&self.identity(left)?, &self.identity(right)?)
                    .map_err(ProgramError::from)?,
            ))
            .into(),
            WireOperation::DimensionSub { left, right } => ArrayIrOperation::Dimension(DimensionOperation::Sub(
                DimensionSubOperation::new(&self.identity(left)?, &self.identity(right)?)
                    .map_err(ProgramError::from)?,
            ))
            .into(),
            WireOperation::DimensionMul { left, right } => ArrayIrOperation::Dimension(DimensionOperation::Mul(
                DimensionMulOperation::new(&self.identity(left)?, &self.identity(right)?)
                    .map_err(ProgramError::from)?,
            ))
            .into(),
            WireOperation::DimensionDivFloor { left, right } => {
                ArrayIrOperation::Dimension(DimensionOperation::DivFloor(
                    DimensionDivFloorOperation::new(&self.identity(left)?, &self.identity(right)?)
                        .map_err(ProgramError::from)?,
                ))
                .into()
            }
            WireOperation::DimensionRem { left, right } => ArrayIrOperation::Dimension(DimensionOperation::Rem(
                DimensionRemOperation::new(&self.identity(left)?, &self.identity(right)?)
                    .map_err(ProgramError::from)?,
            ))
            .into(),
            WireOperation::TileLoad { block_shape, masked } => {
                TileLoadOperation::new(block_shape, boundary(masked)).map_err(ProgramError::from)?.into()
            }
            WireOperation::Scratch { referent, alignment } => {
                ScratchOperation::new(self.array_type(referent)?, alignment).map_err(ProgramError::custom)?.into()
            }
            WireOperation::MaskedLoad => MaskedLoadOperation.into(),
            WireOperation::MaskedStore => MaskedStoreOperation.into(),
            WireOperation::MaskedSwap => MaskedSwapOperation.into(),
            WireOperation::AsyncCopy => AsyncCopyOperation.into(),
            WireOperation::Wait => WaitOperation.into(),
        })
    }
    /// Validates indices before constructing the canonical arena, whose sealing checks SSA and region interfaces.
    fn program<O: Operation<Type = ArrayIrType>, F: Fn(KernelOperation) -> Result<O, KernelSerializationError>>(
        &self,
        wire: WireProgram,
        project: F,
    ) -> Result<FlatProgram<EagerContext<ArrayIrValue<Array>, O>>, KernelSerializationError> {
        check_records(wire.regions.len())?;
        if wire.entry >= wire.regions.len() {
            return Err(invalid("entry region is out of bounds"));
        }
        let mut regions = Vec::with_capacity(wire.regions.len());
        for (region_index, region) in wire.regions.into_iter().enumerate() {
            check_records(region.atoms.len())?;
            check_records(region.instructions.len())?;
            let atom_count = region.atoms.len();
            let check_atoms = |indices: &[usize]| -> Result<(), KernelSerializationError> {
                if indices.iter().any(|&index| index >= atom_count) {
                    return Err(invalid("atom index is out of bounds"));
                }
                Ok(())
            };
            check_atoms(&region.inputs)?;
            check_atoms(&region.outputs)?;
            for instruction in &region.instructions {
                check_atoms(&instruction.inputs)?;
                check_atoms(&instruction.outputs)?;
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
                .collect::<Result<_, KernelSerializationError>>()?;
            let instructions = region
                .instructions
                .into_iter()
                .map(|instruction| {
                    Ok(Instruction::new(
                        project(self.operation(instruction.operation)?)?,
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
        check_records(wire.grid.len())?;
        check_records(wire.parameters.len())?;
        let grid = Grid::new(
            wire.grid
                .into_iter()
                .map(|dimension| {
                    let mut result = GridDimension::new(
                        self.dimension(dimension.extent)?,
                        if dimension.sequential { GridExecution::Sequential } else { GridExecution::Parallel },
                    );
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
                    operation => Err(unsupported(format!("mapping operation {}", operation.name()))),
                })?;
                let mapping = BlockMapping::new(program, parameter.block_shape, boundary(parameter.masked))
                    .map_err(KernelError::from)?;
                Ok(KernelParameter::new(r#type, access, mapping)?)
            })
            .collect::<Result<_, KernelSerializationError>>()?;
        let prefetch = wire.prefetch.into_iter().map(|r#type| self.array_type(r#type)).collect::<Result<_, _>>()?;
        let coordinates = wire.coordinates.into_iter().map(|index| self.identity(index)).collect::<Result<_, _>>()?;
        Ok(KernelCallOperation::from_parts(grid, parameters, prefetch, coordinates)?)
    }
}

impl WireProvenance {
    /// Encodes diagnostic nodes in child-before-parent order.
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
        if depth > 128 {
            return Err(unsupported("provenance nesting deeper than 128"));
        }
        check_records(nodes.len() + 1)?;
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
        check_records(nodes.len() + 1)?;
        let index = nodes.len();
        nodes.push(node);
        Ok(index)
    }

    /// Reconstructs checked provenance iteratively, without recursive deserialization or forward edges.
    fn decode(self) -> Result<Provenance, KernelSerializationError> {
        let mut nodes: Vec<(Provenance, usize)> = Vec::new();
        for node in self.nodes {
            let get = |index: usize| {
                nodes.get(index).ok_or_else(|| invalid("provenance edge must reference an earlier node"))
            };
            let (provenance, depth) = match node {
                WireProvenanceNode::Unknown => (Provenance::unknown(), 0),
                WireProvenanceNode::Scope { name, origin } => {
                    let (origin, depth) = get(origin)?;
                    (Provenance::scope(ProvenanceScope::new(name), origin.clone()), depth + 1)
                }
                WireProvenanceNode::Fused(origins) => {
                    let origins = origins.into_iter().map(get).collect::<Result<Vec<_>, _>>()?;
                    let depth = origins.iter().map(|(_, depth)| *depth).max().unwrap_or(0) + 1;
                    (Provenance::fused(origins.into_iter().map(|(origin, _)| origin.clone())), depth)
                }
            };
            if depth > 128 {
                return Err(invalid("provenance nesting exceeds 128"));
            }
            nodes.push((provenance, depth));
        }
        nodes
            .get(self.root)
            .map(|(provenance, _)| provenance.clone())
            .ok_or_else(|| invalid("provenance root is out of bounds"))
    }
}

/// Maps the transport boundary flag to the canonical policy.
fn boundary(masked: bool) -> BoundaryPolicy {
    if masked { BoundaryPolicy::Masked } else { BoundaryPolicy::InBounds }
}

/// Checks collection sizes before constructing canonical arena metadata.
fn check_records(count: usize) -> Result<(), KernelSerializationError> {
    if count > MAXIMUM_RECORDS { Err(invalid(format!("record count exceeds {MAXIMUM_RECORDS}"))) } else { Ok(()) }
}

/// Creates an explicit unsupported-payload diagnostic.
fn unsupported(feature: impl Into<String>) -> KernelSerializationError {
    KernelSerializationError::Unsupported { feature: feature.into() }
}

/// Creates a malformed-source diagnostic.
fn invalid(message: impl Into<String>) -> KernelSerializationError {
    KernelSerializationError::Invalid { message: message.into() }
}

/// Deserializes a collection without trusting an untrusted sequence size hint for allocation.
struct BoundedCollection<T, const LIMIT: usize>(Vec<T>);

impl<'de, T: Deserialize<'de>, const LIMIT: usize> Deserialize<'de> for BoundedCollection<T, LIMIT> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        /// Sequence visitor shared by transport records and literal bytes.
        struct Visitor<T, const LIMIT: usize>(std::marker::PhantomData<T>);

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
                            "kernel source sequence exceeds {LIMIT} entries"
                        )));
                    }
                    values.push(value);
                }
                Ok(BoundedCollection(values))
            }
        }

        deserializer.deserialize_seq(Visitor::<T, LIMIT>(std::marker::PhantomData))
    }
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

/// Bounds each fixed dot dimension-number sequence.
fn bounded_dot_dimensions<'de, D: Deserializer<'de>>(deserializer: D) -> Result<[Vec<usize>; 4], D::Error> {
    Ok(<[BoundedCollection<usize, MAXIMUM_RECORDS>; 4]>::deserialize(deserializer)?.map(|dimensions| dimensions.0))
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::contexts::Context;
    use crate::kernels::authoring::whole_array_parameter;
    use crate::kernels::interpretation::DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS;
    use crate::operations::SinOperation;
    use crate::programs::TypeIdentityRenaming;

    use super::*;

    #[crate::kernels::kernel(crate = "crate", requires = left.shape()[1] == right.shape()[0])]
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

    #[test]
    fn test_kernel_definition_serialize() {
        let value = Array::scalar(-0.0f32).unwrap();
        let definition = literal_definition(value.clone());
        let decoded = roundtrip(&definition);
        let output = decoded.interpret(vec![], DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS).unwrap();
        assert_eq!(output[0].logical_bytes(), value.logical_bytes());
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
    fn test_kernel_definition_serialize_exact_literal_bits_and_layout() {
        let r#type = ArrayType::new_static(DataType::F32, [2]).with_layout(Some(StridedLayout::new(vec![-4]).into()));
        let bytes = [0x00, 0x00, 0x00, 0x80, 0x45, 0x23, 0xc1, 0x7f];
        let value = Array::from_logical_bytes(r#type, &bytes).unwrap();
        let mut encoder = Encoder::default();
        let wire = encoder.value(&value.clone().into()).unwrap();
        let decoded = Decoder::new(vec![]).value(wire).unwrap();
        let ArrayIrValue::Array(decoded) = decoded else { panic!("expected array") };
        assert_eq!(decoded.r#type(), value.r#type());
        assert_eq!(decoded.logical_bytes(), bytes);
    }

    #[test]
    fn test_kernel_definition_serialize_unsupported_operation() {
        let operation = KernelOperation::Portable(ArrayIrOperation::Array(ArrayOperation::Sin(SinOperation::new())));
        assert_eq!(
            Encoder::default().operation(&operation).err().unwrap().to_string(),
            "kernel source serialization does not support `sin`",
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
            let token = context.bind(AsyncCopyOperation, vec![], &[references[0].clone(), scratch.clone()])?.remove(0);
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
                .bind(MaskedLoadOperation, vec![], &[references[0].clone(), mask.clone(), other.clone()])?
                .remove(0);
            context.bind(ReferenceAtomicAddUpdateOperation::new(), vec![], &[references[0].clone(), value.clone()])?;
            let old = context
                .bind(MaskedSwapOperation, vec![], &[references[0].clone(), value, mask.clone(), other])?
                .remove(0);
            context.bind(MaskedStoreOperation, vec![], &[references[0].clone(), old, mask])?;
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
    fn test_kernel_definition_serialize_refined_dimension_metadata() {
        let bounds = DimensionBounds::non_negative(Some(4)).unwrap();
        let left = DimensionType::new("left", bounds);
        let right = DimensionType::new("right", bounds);
        let operation = DimensionMulOperation::new(&left, &right).unwrap();
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(left.variable().clone(), DimensionVariable::new("renamed", bounds)).unwrap();
        let operation = operation.rename_type_identities(&renaming).unwrap();
        let operation = KernelOperation::Portable(ArrayIrOperation::Dimension(DimensionOperation::Mul(operation)));
        assert_eq!(
            Encoder::default().operation(&operation).err().unwrap().to_string(),
            "kernel source serialization does not support `dimension_mul cached inference metadata`",
        );
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
    fn test_kernel_definition_deserialize_invalid_indices() {
        let definition = literal_definition(Array::scalar(1.0f32).unwrap());
        let mut wire: serde_json::Value = serde_json::to_value(definition).unwrap();
        wire["body"]["entry"] = serde_json::json!(999);
        assert_eq!(
            serde_json::from_value::<KernelDefinition>(wire).unwrap_err().to_string(),
            "invalid serialized kernel source: entry region is out of bounds",
        );
    }

    #[test]
    fn test_kernel_definition_deserialize_invalid_literal_size() {
        let wire = WireValue::Array {
            r#type: WireArrayType {
                data_type: WireDataType::F32,
                shape: vec![WireDimension::Static(usize::MAX)],
                layout: None,
                memory: WireMemory::Device,
            },
            bytes: vec![],
        };
        assert_eq!(
            Decoder::new(vec![]).value(wire).unwrap_err().to_string(),
            format!("invalid serialized kernel source: dimension extent {} exceeds backend width", usize::MAX),
        );
        let wire = WireValue::Array {
            r#type: WireArrayType {
                data_type: WireDataType::F32,
                shape: vec![],
                layout: None,
                memory: WireMemory::Device,
            },
            bytes: vec![0],
        };
        assert_eq!(
            Decoder::new(vec![]).value(wire).unwrap_err().to_string(),
            "invalid serialized kernel source: literal expects 4 bytes but received 1",
        );
    }

    #[test]
    fn test_kernel_definition_deserialize_invalid_coordinate_identity() {
        let variable = DimensionVariable::new("programs", DimensionBounds::non_negative(Some(2)).unwrap());
        let grid = Grid::new(vec![GridDimension::new(Dimension::Dynamic(variable), GridExecution::Parallel)]).unwrap();
        let definition: KernelDefinition =
            KernelDefinition::trace(KernelCallOperation::new(grid, vec![]).unwrap(), |_| Ok(())).unwrap();
        let mut wire: serde_json::Value = serde_json::to_value(definition).unwrap();
        wire["call"]["coordinates"][0] = serde_json::json!(999);
        assert_eq!(
            serde_json::from_value::<KernelDefinition>(wire).unwrap_err().to_string(),
            "invalid serialized kernel source: dimension identity 999 is out of bounds",
        );
    }

    #[test]
    fn test_kernel_definition_deserialize_invalid_region_edge() {
        let definition = literal_definition(Array::scalar(1.0f32).unwrap());
        let mut wire: serde_json::Value = serde_json::to_value(definition).unwrap();
        wire["body"]["regions"][0]["instructions"][0]["regions"] = serde_json::json!([0]);
        assert_eq!(
            serde_json::from_value::<KernelDefinition>(wire).unwrap_err().to_string(),
            "invalid serialized kernel source: region edge must reference an earlier arena entry",
        );
    }

    #[test]
    fn test_kernel_definition_deserialize_literal_storage_budget() {
        let wire = WireValue::Array {
            r#type: WireArrayType {
                data_type: WireDataType::U8,
                shape: vec![WireDimension::Static(2)],
                layout: Some(WireLayout::Strided(vec![MAXIMUM_LITERAL_BYTES as isize])),
                memory: WireMemory::Device,
            },
            bytes: vec![1, 2],
        };
        assert_eq!(
            Decoder::new(vec![]).value(wire).unwrap_err().to_string(),
            format!("invalid serialized kernel source: literal storage exceeds {MAXIMUM_LITERAL_BYTES} bytes"),
        );
    }

    #[test]
    fn test_bounded_collection_deserialize() {
        let sequence = serde::de::value::SeqDeserializer::<_, serde::de::value::Error>::new([1u8, 2, 3].into_iter());
        let result = BoundedCollection::<u8, 2>::deserialize(sequence);
        assert_eq!(result.err().unwrap().to_string(), "kernel source sequence exceeds 2 entries");
    }
}
