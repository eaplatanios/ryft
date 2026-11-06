use std::sync::{Arc, LazyLock, OnceLock};
use std::time::Duration;

use prost::Message;
use ryft_core::{
    ArrayType, DataType, Device, DeviceMesh, Dimension as CoreDimension, DimensionBounds, DimensionVariable,
    ExternalReferenceBinding, Layout, LogicalMesh, Memory, MeshAxis, MeshAxisType, ReferenceSource, Shape, Sharding,
    ShardingDimension, StridedLayout, Tile, TileDimension, TiledLayout, UnreducedKind,
};
use ryft_pjrt::protos::CompilationOptions;
use ryft_pjrt::{BufferType, LoadOptions};
use serde::{Deserialize, Serialize};

use crate::Error;
use crate::domains::Domain;
use crate::domains::compilation::{CompilationError, CompilationKey, canonical_compilation_options_bytes};
use crate::domains::execution::{ExecutionPlan, flatten_device_assignment};
use crate::domains::programs::{CompiledProgram, LoweredProgram};
use crate::domains::runtimes::{RuntimeRequirement, builtin_requirement, normalize_runtime_requirements};
use crate::experimental::lowering::signature::ExecutableSignature;

// TODO(eaplatanios): Review this module.

/// Fixed header identifying the V9 executable envelope.
const PERSISTENT_EXECUTABLE_MAGIC: &[u8; 8] = b"RYFTXLA9";

/// Metadata schema emitted by executable serialization.
const PERSISTENT_EXECUTABLE_SCHEMA_VERSION: u32 = 9;

/// Required features for reference states, dynamic extents, effects, and runtime contracts.
const PERSISTENT_EXECUTABLE_FEATURE_FLAGS: u64 = 15;

/// Schema used by canonical compilation keys.
pub(super) const PERSISTENT_KEY_SCHEMA_VERSION: u32 = 9;

/// Build identity shared by lowering, cache keys, and executable compatibility validation.
pub(super) static COMPILER_IDENTITY: LazyLock<String> = LazyLock::new(|| {
    format!(
        "ryft-xla/{}/openxla/{}/jax/{}",
        env!("CARGO_PKG_VERSION"),
        ryft_xla_sys::XLA_COMMIT,
        ryft_xla_sys::JAX_COMMIT,
    )
});

/// Stable cache-key payload for the V9 executable and external-reference ABI.
#[derive(Serialize)]
struct PersistentKeyV9<'o> {
    /// Version of this persistent record schema.
    schema_version: u32,

    /// Exact textual StableHLO/Shardy module submitted to PJRT.
    stable_hlo: &'o str,

    /// Canonical encoding of the complete PJRT compilation options.
    compilation_options: &'o [u8],

    /// Logical array types with shared dynamic dimension variables.
    signature: PersistentCanonicalArraySignatureV3,

    /// Physical executable argument position for each logical input, or `None` for erased arrays.
    input_mapping: Vec<Option<u64>>,

    /// Physical executable result position for each logical output, or `None` for erased arrays.
    output_mapping: Vec<Option<u64>>,

    /// Physical extent arguments for bounded dynamic input dimensions.
    input_dimensions: Vec<PersistentBoundaryDimensionV5>,

    /// Physical extent results for bounded dynamic output dimensions.
    output_dimensions: Vec<PersistentBoundaryDimensionV5>,

    /// Number of leading logical outputs in the public result.
    output_count: u64,

    /// External reference binding recipes without live reference handles.
    reference_states: Vec<PersistentReferenceStateV6>,

    /// Whether the executable requires the host runtime assertion handler.
    requires_assertion_handler: bool,

    /// Canonical runtime contracts resolved against the invoking session.
    runtime_requirements: Vec<RuntimeRequirement>,

    /// Whether a native ordered-I/O slot is present after the array and extent slots.
    ordered_io: bool,

    /// Whether completion includes unordered I/O without a boundary slot.
    unordered_io: bool,

    /// Whether every ordered-I/O occurrence is `DeviceOrderedIo`, permitting per-device execution.
    device_ordered_io: bool,

    /// Version of the hidden effect calling convention.
    effect_abi_version: u32,

    /// Donation flags for public logical inputs, excluding captures.
    donation_flags: &'o [bool],

    /// Number of leading logical inputs supplied by captures.
    capture_count: u64,

    /// Shardings of all physical executable arguments in argument order.
    expected_argument_shardings: Vec<PersistentShardingV1>,

    /// Concrete compilation and execution mesh.
    mesh: PersistentDeviceMeshV1,

    /// PJRT device kinds in the concrete mesh order.
    device_kinds: &'o [String],

    /// Exact execution facts for selected kernels, absent for ordinary executables.
    kernel_execution_facts: Option<&'o [u8]>,

    /// PJRT platform name that must match the loading client.
    platform_name: &'o str,

    /// PJRT platform version that must match the loading client.
    platform_version: &'o str,

    /// Exact Ryft, OpenXLA, and JAX build identity required when loading.
    compiler_identity: &'o str,

    /// Process-level XLA flags required when loading.
    xla_flags: &'o str,
}

/// Stable V9 metadata envelope stored before serialized PJRT executable bytes.
#[derive(Serialize, Deserialize)]
struct PersistentExecutableMetadataV9 {
    /// Version of this persistent record schema.
    schema_version: u32,

    /// Required executable transport features for this schema.
    feature_flags: u64,

    /// Protobuf encoding of the complete PJRT compilation options.
    compilation_options: Vec<u8>,

    /// Logical array types with shared dynamic dimension variables.
    signature: PersistentArraySignatureV3,

    /// Physical executable argument position for each logical input, or `None` for erased arrays.
    input_mapping: Vec<Option<u64>>,

    /// Physical executable result position for each logical output, or `None` for erased arrays.
    output_mapping: Vec<Option<u64>>,

    /// Physical extent arguments for bounded dynamic input dimensions.
    input_dimensions: Vec<PersistentBoundaryDimensionV5>,

    /// Physical extent results for bounded dynamic output dimensions.
    output_dimensions: Vec<PersistentBoundaryDimensionV5>,

    /// Number of leading logical outputs in the public result.
    output_count: u64,

    /// External reference binding recipes without live reference handles.
    reference_states: Vec<PersistentReferenceStateV6>,

    /// Whether the executable requires the host runtime assertion handler.
    requires_assertion_handler: bool,

    /// Canonical runtime contracts resolved against the invoking session.
    runtime_requirements: Vec<RuntimeRequirement>,

    /// Whether a native ordered-I/O slot is present after the array and extent slots.
    ordered_io: bool,

    /// Whether completion includes unordered I/O without a boundary slot.
    unordered_io: bool,

    /// Whether every ordered-I/O occurrence is `DeviceOrderedIo`, permitting per-device execution.
    device_ordered_io: bool,

    /// Version of the hidden effect calling convention.
    effect_abi_version: u32,

    /// Donation flags for public logical inputs, excluding captures.
    donation_flags: Vec<bool>,

    /// Number of leading logical inputs supplied by captures.
    capture_count: u64,

    /// Shardings of all physical executable arguments in argument order.
    expected_argument_shardings: Vec<PersistentShardingV1>,

    /// Concrete compilation and execution mesh.
    mesh: PersistentDeviceMeshV1,

    /// PJRT device kinds in the concrete mesh order.
    device_kinds: Vec<String>,

    /// Exact live kernel execution facts, revalidated before loading native executable bytes.
    kernel_execution_facts: Option<Vec<u8>>,

    /// Number of executable replicas in the device assignment.
    replica_count: u64,

    /// Number of computation partitions per executable replica.
    partition_count: u64,

    /// Flattened device IDs in replica-major, then partition-major order.
    device_assignment: Vec<u64>,

    /// PJRT platform name that must match the loading client.
    platform_name: String,

    /// PJRT platform version that must match the loading client.
    platform_version: String,

    /// Exact Ryft, OpenXLA, and JAX build identity required when loading.
    compiler_identity: String,

    /// Process-level XLA flags required when loading.
    xla_flags: String,

    /// Original PJRT compilation duration in nanoseconds, saturating at the maximum `u64` value.
    compilation_duration_nanoseconds: Option<u64>,
}

/// Persisted source descriptor for one logical external reference state. It contains no runtime reference handle.
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PersistentReferenceSourceV6 {
    /// Runtime capture position.
    Capture { index: u64 },

    /// Runtime public-input position.
    PublicInput { index: u64 },
}

/// Persisted logical reference-state binding recipe.
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct PersistentReferenceStateV6 {
    /// Capture or public-input source of the external reference.
    source: PersistentReferenceSourceV6,

    /// Redundant flat position retained solely for V6 wire compatibility and corruption detection.
    logical_input_index: u64,

    /// Hidden logical result containing the final state, or `None` for a read-only reference.
    logical_output_index: Option<u64>,
}

/// Persisted logical-axis-to-hidden-physical-slot mapping for one bounded dynamic dimension.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct PersistentBoundaryDimensionV5 {
    /// Logical flattened input or output index containing the dynamic axis.
    logical_index: u64,

    /// Axis within the logical array.
    axis: u64,

    /// Hidden physical argument or result index carrying the runtime extent.
    physical_index: u64,
}

/// V3 logical boundary types with a shared dimension-variable table. Decoding recreates each variable once and
/// preserves its diagnostic name.
#[derive(Serialize, Deserialize)]
struct PersistentArraySignatureV3 {
    /// Shared dynamic variables numbered by first occurrence across inputs and outputs.
    variables: Vec<PersistentDimensionVariableV3>,

    /// Logical input array types, including captures.
    input_types: Vec<PersistentArrayTypeV3>,

    /// Logical output array types, including hidden final reference states.
    output_types: Vec<PersistentArrayTypeV3>,
}

impl PersistentArraySignatureV3 {
    /// Encodes input and output types with shared dynamic dimension variables numbered by first occurrence.
    fn encode(input_types: &[ArrayType], output_types: &[ArrayType]) -> Self {
        let mut variables = Vec::<DimensionVariable>::new();
        let mut encode_type = |r#type: &ArrayType| {
            let shape = r#type
                .shape()
                .dimensions()
                .iter()
                .map(|dimension| match dimension {
                    CoreDimension::Static(value) => PersistentDimensionV3::Static(*value as u64),
                    CoreDimension::Dynamic(variable) => {
                        let index = variables.iter().position(|existing| existing == variable).unwrap_or_else(|| {
                            variables.push(variable.clone());
                            variables.len() - 1
                        });
                        PersistentDimensionV3::Dynamic { variable: index as u64 }
                    }
                })
                .collect();
            PersistentArrayTypeV3 {
                data_type: encode_data_type(r#type.data_type()),
                shape,
                layout: r#type.layout().map(PersistentLayoutV1::from),
                sharding: r#type.sharding().map(PersistentShardingV1::from),
                memory: match r#type.memory() {
                    Memory::Device => PersistentMemoryV1::Device,
                    Memory::Host { pinned } => PersistentMemoryV1::Host(pinned),
                },
            }
        };
        let input_types = input_types.iter().map(&mut encode_type).collect();
        let output_types = output_types.iter().map(&mut encode_type).collect();
        let variables = variables
            .into_iter()
            .map(|variable| PersistentDimensionVariableV3 {
                name: variable.name().to_string(),
                lower: variable.bounds().lower() as u64,
                upper: variable.bounds().upper().map(|upper| upper as u64),
            })
            .collect();
        Self { variables, input_types, output_types }
    }

    /// Removes diagnostic-only variable names from the stable compilation-key representation.
    fn into_canonical(self) -> PersistentCanonicalArraySignatureV3 {
        PersistentCanonicalArraySignatureV3 {
            variables: self
                .variables
                .into_iter()
                .map(|variable| PersistentCanonicalDimensionVariableV3 { lower: variable.lower, upper: variable.upper })
                .collect(),
            input_types: self.input_types,
            output_types: self.output_types,
        }
    }

    /// Decodes input and output types while recreating each shared dynamic dimension variable exactly once.
    fn decode(self) -> Result<(Vec<ArrayType>, Vec<ArrayType>), Error> {
        let variables = self
            .variables
            .into_iter()
            .map(|variable| {
                let bounds = DimensionBounds::new(
                    usize::try_from(variable.lower).map_err(|_| {
                        Error::Compilation(CompilationError::InvalidPersistentExecutable {
                            reason: format!("value {} does not fit in usize", variable.lower),
                        })
                    })?,
                    variable
                        .upper
                        .map(|value| {
                            usize::try_from(value).map_err(|_| {
                                Error::Compilation(CompilationError::InvalidPersistentExecutable {
                                    reason: format!("value {value} does not fit in usize"),
                                })
                            })
                        })
                        .transpose()?,
                )
                .map_err(|error| {
                    Error::Compilation(CompilationError::InvalidPersistentExecutable { reason: error.to_string() })
                })?;
                Ok(DimensionVariable::new(variable.name, bounds))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let decode_type = |r#type: PersistentArrayTypeV3| -> Result<ArrayType, Error> {
            let shape = Shape::new(
                r#type
                    .shape
                    .into_iter()
                    .map(|dimension| match dimension {
                        PersistentDimensionV3::Static(value) => usize::try_from(value)
                            .map_err(|_| {
                                Error::Compilation(CompilationError::InvalidPersistentExecutable {
                                    reason: format!("value {value} does not fit in usize"),
                                })
                            })
                            .map(CoreDimension::Static),
                        PersistentDimensionV3::Dynamic { variable } => variables
                            .get(usize::try_from(variable).map_err(|_| {
                                Error::Compilation(CompilationError::InvalidPersistentExecutable {
                                    reason: format!("value {variable} does not fit in usize"),
                                })
                            })?)
                            .cloned()
                            .map(CoreDimension::Dynamic)
                            .ok_or_else(|| {
                                Error::Compilation(CompilationError::InvalidPersistentExecutable {
                                    reason: "array type references an unknown dimension variable".into(),
                                })
                            }),
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            );
            let memory = match r#type.memory {
                PersistentMemoryV1::Device => Memory::Device,
                PersistentMemoryV1::Host(pinned) => Memory::Host { pinned },
            };
            ArrayType::new(decode_data_type(r#type.data_type)?, shape)
                .with_layout(r#type.layout.map(Layout::try_from).transpose()?)
                .with_sharding(r#type.sharding.map(Sharding::try_from).transpose()?)
                .map(|r#type| r#type.with_memory(memory))
                .map_err(|error| {
                    Error::Compilation(CompilationError::InvalidPersistentExecutable { reason: error.to_string() })
                })
        };
        let input_types = self.input_types.into_iter().map(&decode_type).collect::<Result<Vec<_>, _>>()?;
        let output_types = self.output_types.into_iter().map(decode_type).collect::<Result<Vec<_>, _>>()?;
        Ok((input_types, output_types))
    }
}

/// V3 cache identity for logical boundary types. Variable names are omitted and variables are numbered by first
/// occurrence, preserving their sharing and bounds.
#[derive(Serialize)]
struct PersistentCanonicalArraySignatureV3 {
    /// Shared dynamic variables numbered by first occurrence across inputs and outputs.
    variables: Vec<PersistentCanonicalDimensionVariableV3>,

    /// Logical input array types, including captures.
    input_types: Vec<PersistentArrayTypeV3>,

    /// Logical output array types, including hidden final reference states.
    output_types: Vec<PersistentArrayTypeV3>,
}

/// Named dynamic dimension and inclusive bounds in the V3 executable signature.
#[derive(Serialize, Deserialize)]
struct PersistentDimensionVariableV3 {
    /// Diagnostic name restored with this dynamic variable.
    name: String,

    /// Inclusive lower bound on the runtime extent.
    lower: u64,

    /// Inclusive upper bound on the runtime extent, or `None` for an unbounded dimension.
    upper: Option<u64>,
}

/// Dynamic dimension bounds in a canonical V3 key, with diagnostic names omitted.
#[derive(Serialize)]
struct PersistentCanonicalDimensionVariableV3 {
    /// Inclusive lower bound on the runtime extent.
    lower: u64,

    /// Inclusive upper bound on the runtime extent, or `None` for an unbounded dimension.
    upper: Option<u64>,
}

/// Stable V3 array type record, using explicit tags rather than Rust enum discriminants.
#[derive(Serialize, Deserialize)]
struct PersistentArrayTypeV3 {
    /// Stable numeric element-type tag defined by `encode_data_type`.
    data_type: u8,

    /// Logical dimensions in array axis order.
    shape: Vec<PersistentDimensionV3>,

    /// Explicit physical layout, or `None` when unspecified.
    layout: Option<PersistentLayoutV1>,

    /// Logical sharding, or `None` when unspecified.
    sharding: Option<PersistentShardingV1>,

    /// Memory placement of the array.
    memory: PersistentMemoryV1,
}

/// Static extent or index into the shared V3 dynamic dimension table.
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
enum PersistentDimensionV3 {
    /// Concrete dimension extent.
    Static(u64),

    /// Index into the signature's shared dynamic dimension table.
    Dynamic { variable: u64 },
}

/// Stable V1 encoding of tiled and strided physical layouts.
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum PersistentLayoutV1 {
    /// Minor-to-major axis order and tile dimensions from outermost to innermost tile.
    Tiled { minor_to_major: Vec<u64>, tiles: Vec<Vec<PersistentTileDimensionV1>> },

    /// Signed byte strides in logical axis order.
    Strided { strides: Vec<i64> },
}

impl From<&Layout> for PersistentLayoutV1 {
    fn from(value: &Layout) -> Self {
        match value {
            Layout::Tiled(layout) => Self::Tiled {
                minor_to_major: layout.minor_to_major().iter().map(|dimension| *dimension as u64).collect(),
                tiles: layout
                    .tiles()
                    .iter()
                    .map(|tile| {
                        tile.dimensions()
                            .iter()
                            .map(|dimension| match dimension {
                                TileDimension::Sized(size) => PersistentTileDimensionV1::Sized(*size as u64),
                                TileDimension::Combined => PersistentTileDimensionV1::Combined,
                            })
                            .collect()
                    })
                    .collect(),
            },
            Layout::Strided(layout) => {
                Self::Strided { strides: layout.strides().iter().map(|stride| *stride as i64).collect() }
            }
        }
    }
}

impl TryFrom<PersistentLayoutV1> for Layout {
    type Error = Error;

    fn try_from(value: PersistentLayoutV1) -> Result<Self, Self::Error> {
        Ok(match value {
            PersistentLayoutV1::Tiled { minor_to_major, tiles } => TiledLayout::new(
                minor_to_major
                    .into_iter()
                    .map(|value| {
                        usize::try_from(value).map_err(|_| {
                            Error::Compilation(CompilationError::InvalidPersistentExecutable {
                                reason: format!("value {value} does not fit in usize"),
                            })
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                tiles
                    .into_iter()
                    .map(|tile| {
                        tile.into_iter()
                            .map(|dimension| match dimension {
                                PersistentTileDimensionV1::Sized(size) => usize::try_from(size)
                                    .map_err(|_| {
                                        Error::Compilation(CompilationError::InvalidPersistentExecutable {
                                            reason: format!("value {size} does not fit in usize"),
                                        })
                                    })
                                    .map(TileDimension::Sized),
                                PersistentTileDimensionV1::Combined => Ok(TileDimension::Combined),
                            })
                            .collect::<Result<Vec<_>, _>>()
                            .map(Tile::new)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
            .into(),
            PersistentLayoutV1::Strided { strides } => StridedLayout::new(
                strides
                    .into_iter()
                    .map(|stride| {
                        isize::try_from(stride).map_err(|_| {
                            Error::Compilation(CompilationError::InvalidPersistentExecutable {
                                reason: "layout stride does not fit in isize".into(),
                            })
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
            .into(),
        })
    }
}

/// Stable V1 encoding of a tile extent or a combined tile dimension.
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
enum PersistentTileDimensionV1 {
    /// Fixed extent of one tile dimension.
    Sized(u64),

    /// Combines this dimension with the next more minor logical dimension.
    Combined,
}

/// Stable V1 encoding of device or host memory placement.
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", content = "pinned", rename_all = "snake_case")]
enum PersistentMemoryV1 {
    /// Default device memory.
    Device,

    /// Host memory, with a flag indicating whether it is pinned.
    Host(bool),
}

/// Stable V1 logical sharding record. Conversions retain the mesh, dimension assignments, and reduction state.
#[derive(Serialize, Deserialize)]
struct PersistentShardingV1 {
    /// Logical mesh used by the sharding.
    mesh: PersistentLogicalMeshV1,

    /// Sharding of each array dimension in axis order.
    dimensions: Vec<PersistentShardingDimensionV1>,

    /// Mesh axes whose partial values await reduction.
    unreduced_axes: Vec<String>,

    /// Reduction required to combine partial values. A sum is omitted so earlier payloads still decode and cache
    /// keys for shardings using sums remain unchanged.
    #[serde(default, skip_serializing_if = "PersistentUnreducedKindV1::is_sum")]
    unreduced_kind: PersistentUnreducedKindV1,

    /// Mesh axes along which the value has already been reduced.
    reduced_axes: Vec<String>,

    /// Manual mesh axes across which the value can vary.
    varying_manual_axes: Vec<String>,
}

impl From<&Sharding> for PersistentShardingV1 {
    fn from(value: &Sharding) -> Self {
        Self {
            mesh: PersistentLogicalMeshV1::from(value.mesh()),
            dimensions: value
                .dimensions()
                .iter()
                .map(|dimension| match dimension {
                    ShardingDimension::Replicated => PersistentShardingDimensionV1::Replicated,
                    ShardingDimension::Sharded(axes) => PersistentShardingDimensionV1::Sharded(axes.clone()),
                    ShardingDimension::Unconstrained => PersistentShardingDimensionV1::Unconstrained,
                })
                .collect(),
            unreduced_axes: value.unreduced_axes().iter().cloned().collect(),
            unreduced_kind: value.unreduced_kind().into(),
            reduced_axes: value.reduced_axes().iter().cloned().collect(),
            varying_manual_axes: value.varying_manual_axes().iter().cloned().collect(),
        }
    }
}

impl TryFrom<PersistentShardingV1> for Sharding {
    type Error = Error;

    fn try_from(value: PersistentShardingV1) -> Result<Self, Self::Error> {
        Sharding::new(
            LogicalMesh::try_from(value.mesh)?,
            value
                .dimensions
                .into_iter()
                .map(|dimension| match dimension {
                    PersistentShardingDimensionV1::Replicated => ShardingDimension::Replicated,
                    PersistentShardingDimensionV1::Sharded(axes) => ShardingDimension::Sharded(axes),
                    PersistentShardingDimensionV1::Unconstrained => ShardingDimension::Unconstrained,
                })
                .collect(),
        )
        .and_then(|sharding| sharding.with_unreduced_axes(value.unreduced_axes))
        .and_then(|sharding| sharding.with_unreduced_kind(value.unreduced_kind.into()))
        .and_then(|sharding| sharding.with_reduced_axes(value.reduced_axes))
        .and_then(|sharding| sharding.with_varying_manual_axes(value.varying_manual_axes))
        .map_err(|error| {
            Error::Compilation(CompilationError::InvalidPersistentExecutable { reason: error.to_string() })
        })
    }
}

/// Stable reduction tag for partial values. An omitted tag decodes as a sum for compatibility with earlier V1 records.
#[derive(Copy, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum PersistentUnreducedKindV1 {
    /// Partial values combined by addition. This is the default for older records.
    #[default]
    Sum,

    /// Partial values combined by a maximum.
    Max,

    /// Partial values combined by a minimum.
    Min,
}

impl PersistentUnreducedKindV1 {
    /// Returns whether the reduction tag can be omitted without changing its V1 meaning.
    fn is_sum(&self) -> bool {
        *self == Self::Sum
    }
}

impl From<UnreducedKind> for PersistentUnreducedKindV1 {
    fn from(value: UnreducedKind) -> Self {
        match value {
            UnreducedKind::Sum => Self::Sum,
            UnreducedKind::Max => Self::Max,
            UnreducedKind::Min => Self::Min,
        }
    }
}

impl From<PersistentUnreducedKindV1> for UnreducedKind {
    fn from(value: PersistentUnreducedKindV1) -> Self {
        match value {
            PersistentUnreducedKindV1::Sum => Self::Sum,
            PersistentUnreducedKindV1::Max => Self::Max,
            PersistentUnreducedKindV1::Min => Self::Min,
        }
    }
}

/// Stable V1 sharding assignment for one array dimension.
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", content = "axes", rename_all = "snake_case")]
enum PersistentShardingDimensionV1 {
    /// Array dimension partitioned over no mesh axes.
    Replicated,

    /// Array dimension partitioned over the named mesh axes, in order.
    Sharded(Vec<String>),

    /// Array dimension left unconstrained for the partitioner.
    Unconstrained,
}

/// Ordered logical mesh axes in the V1 transport format.
#[derive(Serialize, Deserialize)]
struct PersistentLogicalMeshV1 {
    /// Mesh axes in their declared order.
    axes: Vec<PersistentMeshAxisV1>,
}

impl From<&LogicalMesh> for PersistentLogicalMeshV1 {
    fn from(value: &LogicalMesh) -> Self {
        Self {
            axes: value
                .axes()
                .iter()
                .map(|axis| PersistentMeshAxisV1 {
                    name: axis.name().into(),
                    size: axis.size() as u64,
                    r#type: match axis.r#type() {
                        MeshAxisType::Auto => 0,
                        MeshAxisType::Explicit => 1,
                        MeshAxisType::Manual => 2,
                    },
                })
                .collect(),
        }
    }
}

impl TryFrom<PersistentLogicalMeshV1> for LogicalMesh {
    type Error = Error;

    fn try_from(value: PersistentLogicalMeshV1) -> Result<Self, Self::Error> {
        LogicalMesh::new(
            value
                .axes
                .into_iter()
                .map(|axis| {
                    let r#type = match axis.r#type {
                        0 => MeshAxisType::Auto,
                        1 => MeshAxisType::Explicit,
                        2 => MeshAxisType::Manual,
                        value => {
                            return Err(Error::Compilation(CompilationError::InvalidPersistentExecutable {
                                reason: format!("unknown mesh axis type {value}"),
                            }));
                        }
                    };
                    MeshAxis::new(
                        axis.name,
                        usize::try_from(axis.size).map_err(|_| {
                            Error::Compilation(CompilationError::InvalidPersistentExecutable {
                                reason: format!("value {} does not fit in usize", axis.size),
                            })
                        })?,
                        r#type,
                    )
                    .map_err(|error| {
                        Error::Compilation(CompilationError::InvalidPersistentExecutable { reason: error.to_string() })
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
        .map_err(|error| {
            Error::Compilation(CompilationError::InvalidPersistentExecutable { reason: error.to_string() })
        })
    }
}

/// Named mesh axis and its stable partitioning-kind tag.
#[derive(Serialize, Deserialize)]
struct PersistentMeshAxisV1 {
    /// Name used to identify this mesh axis in shardings.
    name: String,

    /// Number of devices along the axis.
    size: u64,

    /// Stable axis-kind tag for automatic, explicit, or manual partitioning.
    r#type: u8,
}

/// Logical mesh and ordered concrete device identities in the V1 transport format.
#[derive(Serialize, Deserialize)]
struct PersistentDeviceMeshV1 {
    /// Logical shape and axis kinds of the mesh.
    logical_mesh: PersistentLogicalMeshV1,

    /// Devices in the concrete mesh order.
    devices: Vec<PersistentDeviceV1>,
}

impl From<&DeviceMesh> for PersistentDeviceMeshV1 {
    fn from(value: &DeviceMesh) -> Self {
        Self {
            logical_mesh: PersistentLogicalMeshV1::from(value.logical_mesh()),
            devices: value
                .devices()
                .iter()
                .map(|device| PersistentDeviceV1 {
                    id: device.id() as u64,
                    process_index: device.process_index() as u64,
                })
                .collect(),
        }
    }
}

impl TryFrom<PersistentDeviceMeshV1> for DeviceMesh {
    type Error = Error;

    fn try_from(value: PersistentDeviceMeshV1) -> Result<Self, Self::Error> {
        DeviceMesh::new(
            LogicalMesh::try_from(value.logical_mesh)?,
            value
                .devices
                .into_iter()
                .map(|device| {
                    Ok(Device::new(
                        usize::try_from(device.id).map_err(|_| {
                            Error::Compilation(CompilationError::InvalidPersistentExecutable {
                                reason: format!("value {} does not fit in usize", device.id),
                            })
                        })?,
                        usize::try_from(device.process_index).map_err(|_| {
                            Error::Compilation(CompilationError::InvalidPersistentExecutable {
                                reason: format!("value {} does not fit in usize", device.process_index),
                            })
                        })?,
                    ))
                })
                .collect::<Result<Vec<_>, Error>>()?,
        )
        .map_err(|error| {
            Error::Compilation(CompilationError::InvalidPersistentExecutable { reason: error.to_string() })
        })
    }
}

/// Device identity used to reconstruct a concrete mesh without retaining live PJRT handles.
#[derive(Serialize, Deserialize)]
struct PersistentDeviceV1 {
    /// Global PJRT device identifier.
    id: u64,

    /// Index of the process that owns this device.
    process_index: u64,
}

/// Converts an in-memory executable signature mapping to its stable persistent representation.
fn persistent_mapping(mapping: &[Option<usize>]) -> Result<Vec<Option<u64>>, Error> {
    mapping
        .iter()
        .map(|index| {
            index
                .map(|index| {
                    u64::try_from(index).map_err(|_| {
                        Error::Compilation(CompilationError::InvalidPersistentExecutable {
                            reason: format!("physical signature index {index} does not fit in u64"),
                        })
                    })
                })
                .transpose()
        })
        .collect()
}

/// Encodes logical reference-state slots in the stable V6 shape without runtime reference handles.
///
/// V6 redundantly stores each flat logical input position. Deriving that coordinate from the canonical source keeps
/// valid serialized bytes unchanged without reintroducing the redundant field into [`ExternalReferenceBinding`].
fn persistent_reference_states(
    states: &[ExternalReferenceBinding],
    capture_count: usize,
) -> Result<Vec<PersistentReferenceStateV6>, Error> {
    states
        .iter()
        .map(|state| {
            let logical_input_index = state.source().flat_input_index(capture_count)?;
            let source = match state.source() {
                ReferenceSource::Capture { index } => PersistentReferenceSourceV6::Capture { index: index as u64 },
                ReferenceSource::Input { index } => PersistentReferenceSourceV6::PublicInput { index: index as u64 },
            };
            Ok(PersistentReferenceStateV6 {
                source,
                logical_input_index: logical_input_index as u64,
                logical_output_index: state.output_index().map(|index| index as u64),
            })
        })
        .collect()
}

/// Decodes logical reference-state slots without runtime reference handles and validates the redundant V6 flat input
/// coordinate.
fn decode_persistent_reference_states(
    states: Vec<PersistentReferenceStateV6>,
    capture_count: usize,
) -> Result<Vec<ExternalReferenceBinding>, Error> {
    states
        .into_iter()
        .map(|state| {
            let source = match state.source {
                PersistentReferenceSourceV6::Capture { index } => ReferenceSource::Capture {
                    index: usize::try_from(index).map_err(|_| {
                        Error::Compilation(CompilationError::InvalidPersistentExecutable {
                            reason: format!("value {index} does not fit in usize"),
                        })
                    })?,
                },
                PersistentReferenceSourceV6::PublicInput { index } => ReferenceSource::Input {
                    index: usize::try_from(index).map_err(|_| {
                        Error::Compilation(CompilationError::InvalidPersistentExecutable {
                            reason: format!("value {index} does not fit in usize"),
                        })
                    })?,
                },
            };
            let logical_input_index = usize::try_from(state.logical_input_index).map_err(|_| {
                Error::Compilation(CompilationError::InvalidPersistentExecutable {
                    reason: format!("value {} does not fit in usize", state.logical_input_index),
                })
            })?;
            if source.flat_input_index(capture_count).map_err(|error| {
                Error::Compilation(CompilationError::InvalidPersistentExecutable { reason: error.to_string() })
            })? != logical_input_index
            {
                return Err(Error::Compilation(CompilationError::InvalidPersistentExecutable {
                    reason: "reference-state source does not match its logical input".into(),
                }));
            }
            Ok(ExternalReferenceBinding::new(
                source,
                state
                    .logical_output_index
                    .map(|value| {
                        usize::try_from(value).map_err(|_| {
                            Error::Compilation(CompilationError::InvalidPersistentExecutable {
                                reason: format!("value {value} does not fit in usize"),
                            })
                        })
                    })
                    .transpose()?,
            ))
        })
        .collect()
}

/// Encodes the hidden bounded-input extent slots of one executable signature.
fn persistent_input_dimensions(signature: &ExecutableSignature) -> Vec<PersistentBoundaryDimensionV5> {
    signature
        .input_dimensions()
        .iter()
        .map(|dimension| PersistentBoundaryDimensionV5 {
            logical_index: dimension.logical_input_index() as u64,
            axis: dimension.axis() as u64,
            physical_index: dimension.physical_input_index() as u64,
        })
        .collect()
}

/// Encodes the hidden bounded-output extent slots of one executable signature.
fn persistent_output_dimensions(signature: &ExecutableSignature) -> Vec<PersistentBoundaryDimensionV5> {
    signature
        .output_dimensions()
        .iter()
        .map(|dimension| PersistentBoundaryDimensionV5 {
            logical_index: dimension.logical_output_index() as u64,
            axis: dimension.axis() as u64,
            physical_index: dimension.physical_output_index() as u64,
        })
        .collect()
}

/// Encodes `data_type` as a stable one-byte code for persistent executables. These codes are persisted, so new data
/// types must be appended with fresh codes and existing codes must never be renumbered.
fn encode_data_type(data_type: DataType) -> u8 {
    match data_type {
        DataType::Token => 0,
        DataType::Boolean => 1,
        DataType::I1 => 2,
        DataType::I2 => 3,
        DataType::I4 => 4,
        DataType::I8 => 5,
        DataType::I16 => 6,
        DataType::I32 => 7,
        DataType::I64 => 8,
        DataType::U1 => 9,
        DataType::U2 => 10,
        DataType::U4 => 11,
        DataType::U8 => 12,
        DataType::U16 => 13,
        DataType::U32 => 14,
        DataType::U64 => 15,
        DataType::F4E2M1FN => 16,
        DataType::F8E3M4 => 17,
        DataType::F8E4M3 => 18,
        DataType::F8E4M3FN => 19,
        DataType::F8E4M3FNUZ => 20,
        DataType::F8E4M3B11FNUZ => 21,
        DataType::F8E5M2 => 22,
        DataType::F8E5M2FNUZ => 23,
        DataType::F8E8M0FNU => 24,
        DataType::BF16 => 25,
        DataType::F16 => 26,
        DataType::F32 => 27,
        DataType::F64 => 28,
        DataType::C64 => 29,
        DataType::C128 => 30,
        DataType::F6E2M3FN => 31,
        DataType::F6E3M2FN => 32,
        DataType::Zero => 33,
    }
}

/// Decodes an explicit persistent element-type tag and rejects tags unknown to this schema.
fn decode_data_type(value: u8) -> Result<DataType, Error> {
    Ok(match value {
        0 => DataType::Token,
        1 => DataType::Boolean,
        2 => DataType::I1,
        3 => DataType::I2,
        4 => DataType::I4,
        5 => DataType::I8,
        6 => DataType::I16,
        7 => DataType::I32,
        8 => DataType::I64,
        9 => DataType::U1,
        10 => DataType::U2,
        11 => DataType::U4,
        12 => DataType::U8,
        13 => DataType::U16,
        14 => DataType::U32,
        15 => DataType::U64,
        16 => DataType::F4E2M1FN,
        17 => DataType::F8E3M4,
        18 => DataType::F8E4M3,
        19 => DataType::F8E4M3FN,
        20 => DataType::F8E4M3FNUZ,
        21 => DataType::F8E4M3B11FNUZ,
        22 => DataType::F8E5M2,
        23 => DataType::F8E5M2FNUZ,
        24 => DataType::F8E8M0FNU,
        25 => DataType::BF16,
        26 => DataType::F16,
        27 => DataType::F32,
        28 => DataType::F64,
        29 => DataType::C64,
        30 => DataType::C128,
        31 => DataType::F6E2M3FN,
        32 => DataType::F6E3M2FN,
        33 => DataType::Zero,
        value => {
            return Err(Error::Compilation(CompilationError::InvalidPersistentExecutable {
                reason: format!("unknown data type tag {value}"),
            }));
        }
    })
}

/// Encodes one boundary's input and output types using the persistent signature's canonical dimension-variable
/// numbering.
///
/// [`ArrayType`] equality is dimension-variable identity equality, and two structurally identical lowerings (or one
/// lowering and its cache-restored executable) never share variable identities, so boundary compatibility must be
/// compared through the same identity-free canonical form the compilation cache key uses.
pub(super) fn canonical_boundary_types(input_types: &[ArrayType], output_types: &[ArrayType]) -> Vec<u8> {
    // Serializing the plain persistent data model cannot fail.
    serde_json::to_vec(&PersistentArraySignatureV3::encode(input_types, output_types).into_canonical()).unwrap()
}

impl LoweredProgram {
    /// Encodes the complete compilation contract as a deterministic cache key.
    pub(super) fn compilation_key(&self) -> Result<CompilationKey, Error> {
        let compilation_options = canonical_compilation_options_bytes(&self.compilation_options);
        let key = PersistentKeyV9 {
            schema_version: PERSISTENT_KEY_SCHEMA_VERSION,
            stable_hlo: &self.stable_hlo,
            compilation_options: compilation_options.as_slice(),
            signature: PersistentArraySignatureV3::encode(&self.input_types, &self.output_types).into_canonical(),
            input_mapping: persistent_mapping(self.signature.input_mapping())?,
            output_mapping: persistent_mapping(self.signature.output_mapping())?,
            input_dimensions: persistent_input_dimensions(&self.signature),
            output_dimensions: persistent_output_dimensions(&self.signature),
            output_count: self.output_count as u64,
            reference_states: persistent_reference_states(&self.reference_states, self.capture_count)?,
            requires_assertion_handler: self.requires_assertion_handler,
            runtime_requirements: self.runtime_requirements.to_vec(),
            ordered_io: self.signature.has_ordered_io(),
            unordered_io: self.signature.has_unordered_io(),
            device_ordered_io: self.signature.has_device_ordered_io(),
            effect_abi_version: 1,
            donation_flags: &self.donation_flags,
            capture_count: self.capture_count as u64,
            expected_argument_shardings: self
                .expected_argument_shardings
                .iter()
                .map(PersistentShardingV1::from)
                .collect(),
            mesh: PersistentDeviceMeshV1::from(&self.mesh),
            device_kinds: &self.device_kinds,
            kernel_execution_facts: self.kernel_execution_facts.as_deref(),
            platform_name: &self.platform_name,
            platform_version: &self.platform_version,
            compiler_identity: &self.compiler_identity,
            xla_flags: &self.xla_flags,
        };
        let canonical_bytes = serde_json::to_vec(&key).map_err(|error| {
            Error::Compilation(CompilationError::InvalidPersistentExecutable {
                reason: format!("failed to encode persistent key: {error}"),
            })
        })?;
        Ok(CompilationKey { canonical_bytes: canonical_bytes.into() })
    }
}

impl CompiledProgram<'_> {
    /// Serializes the executable and its versioned invocation contract, or returns `None` when PJRT does not support
    /// executable serialization. Persistent cache directories must be trusted sources of executable code.
    pub(super) fn serialize(&self) -> Result<Option<Vec<u8>>, Error> {
        let executable = match self.executable.executable() {
            Ok(executable) => executable,
            Err(ryft_pjrt::Error::Unimplemented { .. }) => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let serialized = match executable.serialize() {
            Ok(serialized) => serialized,
            Err(ryft_pjrt::Error::Unimplemented { .. }) => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let device_assignment = self.executable.device_assignment()?;
        let metadata = PersistentExecutableMetadataV9 {
            schema_version: PERSISTENT_EXECUTABLE_SCHEMA_VERSION,
            feature_flags: PERSISTENT_EXECUTABLE_FEATURE_FLAGS,
            compilation_options: self.compilation_options.encode_to_vec(),
            signature: PersistentArraySignatureV3::encode(&self.input_types, &self.output_types),
            input_mapping: persistent_mapping(self.signature.input_mapping())?,
            output_mapping: persistent_mapping(self.signature.output_mapping())?,
            input_dimensions: persistent_input_dimensions(&self.signature),
            output_dimensions: persistent_output_dimensions(&self.signature),
            output_count: self.output_count as u64,
            reference_states: persistent_reference_states(&self.reference_states, self.capture_count)?,
            requires_assertion_handler: self.requires_assertion_handler,
            runtime_requirements: self.runtime_requirements.to_vec(),
            ordered_io: self.signature.has_ordered_io(),
            unordered_io: self.signature.has_unordered_io(),
            device_ordered_io: self.signature.has_device_ordered_io(),
            effect_abi_version: 1,
            donation_flags: self.donation_flags.to_vec(),
            capture_count: self.capture_count as u64,
            expected_argument_shardings: self
                .expected_argument_shardings
                .iter()
                .map(PersistentShardingV1::from)
                .collect(),
            mesh: PersistentDeviceMeshV1::from(&self.mesh),
            device_kinds: self.device_kinds.to_vec(),
            kernel_execution_facts: self.kernel_execution_facts.clone(),
            replica_count: device_assignment.replica_count() as u64,
            partition_count: device_assignment.computation_count() as u64,
            device_assignment: flatten_device_assignment(&device_assignment)?,
            platform_name: self.platform_name.to_string(),
            platform_version: self.platform_version.to_string(),
            compiler_identity: self.compiler_identity.to_string(),
            xla_flags: self.xla_flags.to_string(),
            compilation_duration_nanoseconds: self
                .compilation_duration
                .map(|duration| duration.as_nanos())
                .map(|nanoseconds| u64::try_from(nanoseconds).unwrap_or(u64::MAX)),
        };
        let metadata = serde_json::to_vec(&metadata).map_err(|error| {
            Error::Compilation(CompilationError::InvalidPersistentExecutable {
                reason: format!("failed to encode metadata: {error}"),
            })
        })?;
        let mut bytes = Vec::with_capacity(
            PERSISTENT_EXECUTABLE_MAGIC.len() + size_of::<u64>() + metadata.len() + serialized.data().len(),
        );
        bytes.extend_from_slice(PERSISTENT_EXECUTABLE_MAGIC);
        bytes.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        bytes.extend_from_slice(metadata.as_slice());
        bytes.extend_from_slice(serialized.data());
        Ok(Some(bytes))
    }
}

impl<'c> Domain<'c> {
    /// Validates and loads a versioned executable envelope for this domain. Unsupported schemas, target configurations,
    /// or PJRT loading return `None`. Malformed supported metadata returns an error.
    ///
    /// The envelope, target, runtime contracts, and signature mapping logical values to physical executable slots are
    /// validated before loading the native PJRT executable bytes. Persistent cache directories must be trusted sources
    /// of executable code.
    pub(super) fn deserialize_compiled_program(&self, bytes: &[u8]) -> Result<Option<CompiledProgram<'c>>, Error> {
        let header_size = PERSISTENT_EXECUTABLE_MAGIC.len() + size_of::<u64>();
        if bytes.len() < header_size {
            return Err(Error::Compilation(CompilationError::InvalidPersistentExecutable {
                reason: "missing persistent executable header".into(),
            }));
        }
        if &bytes[..PERSISTENT_EXECUTABLE_MAGIC.len()] != PERSISTENT_EXECUTABLE_MAGIC {
            return Ok(None);
        }
        let metadata_size = u64::from_le_bytes(
            bytes[PERSISTENT_EXECUTABLE_MAGIC.len()..header_size]
                .try_into()
                .expect("metadata size has an exact fixed width"),
        );
        let metadata_size = usize::try_from(metadata_size).map_err(|_| {
            Error::Compilation(CompilationError::InvalidPersistentExecutable {
                reason: format!("value {metadata_size} does not fit in usize"),
            })
        })?;
        let metadata_end = header_size
            .checked_add(metadata_size)
            .filter(|metadata_end| *metadata_end <= bytes.len())
            .ok_or_else(|| {
            Error::Compilation(CompilationError::InvalidPersistentExecutable {
                reason: "persistent executable metadata is truncated".into(),
            })
        })?;
        let metadata: PersistentExecutableMetadataV9 = serde_json::from_slice(&bytes[header_size..metadata_end])
            .map_err(|error| {
                Error::Compilation(CompilationError::InvalidPersistentExecutable {
                    reason: format!("failed to decode metadata: {error}"),
                })
            })?;
        if metadata.schema_version != PERSISTENT_EXECUTABLE_SCHEMA_VERSION
            || metadata.feature_flags != PERSISTENT_EXECUTABLE_FEATURE_FLAGS
        {
            return Ok(None);
        }
        let target = self.target()?;
        if metadata.platform_name != target.platform_name()
            || metadata.platform_version != target.platform_version()
            || metadata.compiler_identity != COMPILER_IDENTITY.as_str()
            || metadata.xla_flags != std::env::var("XLA_FLAGS").unwrap_or_default()
        {
            return Ok(None);
        }

        let mesh = DeviceMesh::try_from(metadata.mesh)?;
        let runtime_requirements = normalize_runtime_requirements(metadata.runtime_requirements)?;
        if let Some(expected) = &metadata.kernel_execution_facts {
            let actual = crate::kernels::KernelExecutionFacts::from_target(target, &mesh)?.configuration_key()?;
            if expected != &actual {
                return Ok(None);
            }
        }
        let capture_count = usize::try_from(metadata.capture_count).map_err(|_| {
            Error::Compilation(CompilationError::InvalidPersistentExecutable {
                reason: format!("value {} does not fit in usize", metadata.capture_count),
            })
        })?;
        let output_count = usize::try_from(metadata.output_count).map_err(|_| {
            Error::Compilation(CompilationError::InvalidPersistentExecutable {
                reason: format!("value {} does not fit in usize", metadata.output_count),
            })
        })?;
        let reference_states = decode_persistent_reference_states(metadata.reference_states, capture_count)?;
        if !reference_states.is_empty() && !target.is_participating_mesh(&mesh) {
            return Err(Error::Compilation(CompilationError::InvalidPersistentExecutable {
                reason: "reference-state mesh does not include an addressable device of this process".into(),
            }));
        }
        if metadata.device_kinds != target.mesh_device_kinds(&mesh)? {
            return Ok(None);
        }
        if mesh
            .devices()
            .iter()
            .any(|device| target.device(device.id()).is_none_or(|live| live.device() != *device))
        {
            return Ok(None);
        }
        let (input_types, output_types) = metadata.signature.decode()?;
        if output_count > output_types.len() {
            return Err(Error::Compilation(CompilationError::InvalidPersistentExecutable {
                reason: "reference-state metadata has an invalid output count".into(),
            }));
        }
        if capture_count > input_types.len() || metadata.donation_flags.len() != input_types.len() - capture_count {
            return Err(Error::Compilation(CompilationError::InvalidPersistentExecutable {
                reason: "capture or donation metadata has an invalid arity".into(),
            }));
        }
        let reference_state_input_indices = reference_states
            .iter()
            .map(|state| {
                state.source().flat_input_index(capture_count).map_err(|error| {
                    Error::Compilation(CompilationError::InvalidPersistentExecutable { reason: error.to_string() })
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if reference_state_input_indices.windows(2).any(|indices| indices[0] >= indices[1]) {
            return Err(Error::Compilation(CompilationError::InvalidPersistentExecutable {
                reason: "reference states are not in canonical logical input order".into(),
            }));
        }
        if !reference_states
            .iter()
            .filter_map(ExternalReferenceBinding::output_index)
            .eq(output_count..output_types.len())
        {
            return Err(Error::Compilation(CompilationError::InvalidPersistentExecutable {
                reason: "reference-state metadata does not cover the hidden output suffix".into(),
            }));
        }
        for (state, &expected_input_index) in reference_states.iter().zip(&reference_state_input_indices) {
            if expected_input_index >= input_types.len() {
                return Err(Error::Compilation(CompilationError::InvalidPersistentExecutable {
                    reason: "reference-state source does not match its logical input".into(),
                }));
            }
            let input_type = &input_types[expected_input_index];
            if input_type.data_type().is_zero() {
                return Err(Error::Compilation(CompilationError::InvalidPersistentExecutable {
                    reason: "reference-state input has a zero-space type".into(),
                }));
            }
            if input_type.memory() != Memory::Device {
                return Err(Error::Compilation(CompilationError::InvalidPersistentExecutable {
                    reason: "reference-state input is not in device memory".into(),
                }));
            }
            if input_type
                .shape()
                .dimensions()
                .iter()
                .any(|dimension| dimension.value().is_none() && dimension.upper_bound().is_none())
            {
                return Err(Error::Compilation(CompilationError::InvalidPersistentExecutable {
                    reason: "reference-state input has an unbounded dynamic dimension".into(),
                }));
            }
            if input_type.static_shape().is_none() && state.output_index().is_some() {
                return Err(Error::Compilation(CompilationError::InvalidPersistentExecutable {
                    reason: format!(
                        "mutated bounded-dynamic reference-state input {expected_input_index} is unsupported because \
                     backend alias compatibility has not been verified",
                    ),
                }));
            }
            if input_type.static_shape().is_none()
                && input_type.sharding().is_some_and(|sharding| !sharding.is_replicated())
            {
                return Err(Error::Compilation(CompilationError::InvalidPersistentExecutable {
                    reason: format!(
                        "bounded-dynamic reference-state input {expected_input_index} logical sharding is not replicated",
                    ),
                }));
            }
        }
        if metadata.effect_abi_version != 1 {
            return Ok(None);
        }
        if metadata.device_ordered_io && !metadata.ordered_io {
            return Err(Error::Compilation(CompilationError::InvalidPersistentExecutable {
                reason: "device-ordered I/O metadata requires an ordered boundary slot".into(),
            }));
        }
        for (required, identifier) in
            [(metadata.requires_assertion_handler, "ryft.assert"), (metadata.ordered_io, "ryft.ordered-io")]
        {
            if required && !runtime_requirements.contains(&builtin_requirement(identifier)) {
                return Err(Error::Compilation(CompilationError::InvalidPersistentExecutable {
                    reason: format!(
                        "effect metadata requires runtime extension `{identifier}` with ABI version 1 and empty configuration",
                    ),
                }));
            }
        }
        let signature = ExecutableSignature::new(input_types.as_slice(), output_types.as_slice()).with_effects(
            metadata.ordered_io,
            metadata.unordered_io,
            metadata.requires_assertion_handler,
            metadata.device_ordered_io,
        );
        if persistent_mapping(signature.input_mapping())? != metadata.input_mapping
            || persistent_mapping(signature.output_mapping())? != metadata.output_mapping
            || persistent_input_dimensions(&signature) != metadata.input_dimensions
            || persistent_output_dimensions(&signature) != metadata.output_dimensions
        {
            return Err(Error::Compilation(CompilationError::InvalidPersistentExecutable {
                reason: "executable signature does not match logical input and output types".into(),
            }));
        }
        if input_types
            .iter()
            .chain(output_types.iter())
            .filter_map(ArrayType::sharding)
            .any(|sharding| !sharding.mesh().has_same_topology(mesh.logical_mesh()))
        {
            return Err(Error::Compilation(CompilationError::InvalidPersistentExecutable {
                reason: "input or output sharding mesh does not match executable mesh".into(),
            }));
        }
        let expected_argument_shardings = metadata
            .expected_argument_shardings
            .into_iter()
            .map(Sharding::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        if expected_argument_shardings
            .iter()
            .any(|sharding| !sharding.mesh().has_same_topology(mesh.logical_mesh()))
        {
            return Err(Error::Compilation(CompilationError::InvalidPersistentExecutable {
                reason: "argument sharding mesh does not match executable mesh".into(),
            }));
        }
        let physical_input_count = signature.array_input_count();
        if expected_argument_shardings.len() != physical_input_count {
            return Err(Error::Compilation(CompilationError::InvalidPersistentExecutable {
                reason: "expected argument shardings do not match the physical input count".into(),
            }));
        }
        for (state, &expected_input_index) in reference_states.iter().zip(&reference_state_input_indices) {
            let physical_input = signature.input_mapping()[expected_input_index].ok_or_else(|| {
                Error::Compilation(CompilationError::InvalidPersistentExecutable {
                    reason: "reference-state input is erased from the executable boundary".into(),
                })
            })?;
            let input_type = &input_types[expected_input_index];
            let expected_sharding = &expected_argument_shardings[physical_input];
            if input_type.static_shape().is_none() && !expected_sharding.is_replicated() {
                return Err(Error::Compilation(CompilationError::InvalidPersistentExecutable {
                    reason: format!(
                        "bounded-dynamic reference-state input {expected_input_index} expected argument sharding is not \
                     replicated",
                    ),
                }));
            }
            if expected_sharding.rank() != input_type.rank()
                || input_type.sharding().is_some_and(|sharding| sharding != expected_sharding)
            {
                return Err(Error::Compilation(CompilationError::InvalidPersistentExecutable {
                    reason: format!(
                        "reference-state input {expected_input_index} logical sharding does not match its expected \
                     argument sharding",
                    ),
                }));
            }
            if let Some(logical_output_index) = state.output_index() {
                signature.output_mapping()[logical_output_index].ok_or_else(|| {
                    Error::Compilation(CompilationError::InvalidPersistentExecutable {
                        reason: "reference-state output is erased from the executable boundary".into(),
                    })
                })?;
                let output_type = &output_types[logical_output_index];
                if output_type.sharding().is_some_and(|sharding| sharding != expected_sharding) {
                    return Err(Error::Compilation(CompilationError::InvalidPersistentExecutable {
                        reason: format!(
                            "reference-state output {logical_output_index} logical sharding does not match state input \
                         {expected_input_index}",
                        ),
                    }));
                }
                if input_type != output_type {
                    return Err(Error::Compilation(CompilationError::InvalidPersistentExecutable {
                        reason: "reference-state output is not type-compatible with its input".into(),
                    }));
                }
            }
        }
        let replica_count = usize::try_from(metadata.replica_count).map_err(|_| {
            Error::Compilation(CompilationError::InvalidPersistentExecutable {
                reason: format!("value {} does not fit in usize", metadata.replica_count),
            })
        })?;
        let partition_count = usize::try_from(metadata.partition_count).map_err(|_| {
            Error::Compilation(CompilationError::InvalidPersistentExecutable {
                reason: format!("value {} does not fit in usize", metadata.partition_count),
            })
        })?;
        if replica_count.checked_mul(partition_count) != Some(mesh.device_count())
            || metadata.device_assignment.len() != mesh.device_count()
        {
            return Err(Error::Compilation(CompilationError::InvalidPersistentExecutable {
                reason: "device assignment does not match executable mesh".into(),
            }));
        }
        let donation_flags = metadata.donation_flags;
        let compilation_options =
            CompilationOptions::decode(metadata.compilation_options.as_slice()).map_err(|error| {
                Error::Compilation(CompilationError::InvalidPersistentExecutable {
                    reason: format!("failed to decode compilation options: {error}"),
                })
            })?;
        // Register before deserialization because loading an executable may resolve its custom-call target. The
        // persisted requirements are authoritative on cache hits, where the source program is no longer available.
        self.ensure_runtime_requirements(&runtime_requirements)?;
        let executable = self.client().deserialize_and_load_executable(
            &bytes[metadata_end..],
            Some(&compilation_options),
            &LoadOptions::default(),
        )?;
        let output_types_physical = executable.executable()?.output_element_types()?;
        if output_types_physical.len() != signature.physical_output_count()
            || signature
                .ordered_io_output_index()
                .is_some_and(|index| output_types_physical[index] != BufferType::Token)
        {
            return Err(Error::Compilation(CompilationError::InvalidPersistentExecutable {
                reason: "effect signature does not match executable physical outputs".into(),
            }));
        }
        let device_assignment = executable.device_assignment()?;
        if device_assignment.replica_count() != replica_count
            || device_assignment.computation_count() != partition_count
            || flatten_device_assignment(&device_assignment)? != metadata.device_assignment
        {
            return Ok(None);
        }
        let compilation_duration = metadata.compilation_duration_nanoseconds.map(Duration::from_nanos);
        let plan = ExecutionPlan::new(
            &executable,
            &input_types,
            &output_types,
            &signature,
            &donation_flags,
            capture_count,
            &mesh,
        )?;
        Ok(Some(CompiledProgram {
            executable: Arc::new(executable),
            input_types: input_types.into(),
            output_types: output_types.into(),
            output_count,
            reference_states: reference_states.into(),
            signature,
            requires_assertion_handler: metadata.requires_assertion_handler,
            runtime_requirements: runtime_requirements.into(),
            donation_flags: donation_flags.into(),
            capture_count,
            expected_argument_shardings: expected_argument_shardings.into(),
            mesh,
            compilation_options,
            platform_name: metadata.platform_name.into(),
            platform_version: metadata.platform_version.into(),
            device_kinds: metadata.device_kinds.into(),
            kernel_execution_facts: metadata.kernel_execution_facts,
            compiler_identity: metadata.compiler_identity.into(),
            xla_flags: metadata.xla_flags.into(),
            compilation_duration,
            analysis: Arc::new(OnceLock::new()),
            plan: Arc::new(plan),
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, AnalyzableCompilationDomain, ArrayIrType, ArrayReference, ArrayType, CompilationCacheDomain,
        CompilationContext, CompilationDomain, DataType, Dimension as CoreDimension, DimensionBounds,
        DimensionVariable, ExternalReferenceBinding, Memory, Placeholder, ProgramError, ReferenceAddUpdateOperation,
        ReferenceReadOperation, ReferenceSource, ReferenceType, Shape, Sharding, ShardingDimension,
        StagedFunction as CoreStagedFunction, StatefulCompilationDomain, Typed, ValueProjection,
    };
    use ryft_pjrt::protos::CompilationOptions;
    use ryft_pjrt::{ClientOptions, CpuClientOptions, load_cpu_plugin};

    use crate::domains::compilation::CompilationError;
    use crate::domains::options::Options;
    use crate::domains::sessions::Session;
    use crate::domains::tests::*;
    use crate::experimental::ops::{Constant, FlatProgram, ProgramBuilder};
    use crate::tests::values_to_bytes;
    use crate::{Array, Error, Value};

    use super::*;

    #[test]
    fn test_persistent_executable_round_trip_preserves_invocation_and_analysis() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let capture = Array::from_host_buffer(
            &domain,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[10.0, 20.0, 30.0, 40.0]).as_slice(),
        )
        .unwrap();
        fn add_capture<'c>(
            _: Vec<Constant>,
            captures: Vec<ryft_core::compilation::CompilationTracer<Domain<'c>>>,
            input: ryft_core::compilation::CompilationTracer<Domain<'c>>,
        ) -> Result<ryft_core::compilation::CompilationTracer<Domain<'c>>, Error> {
            let capture = ValueProjection::<ArrayType>::into_projected(captures.into_iter().next().unwrap())
                .map_err(ProgramError::from)?;
            let input = ValueProjection::<ArrayType>::into_projected(input).map_err(ProgramError::from)?;
            Ok(ValueProjection::<ArrayType>::from_projected(capture + input))
        }
        let staged: CoreStagedFunction<Domain<'_>, ArrayIrType, ArrayIrType> =
            CoreStagedFunction::<Domain<'_>, ArrayIrType, ArrayIrType>::stage(
                add_capture,
                vec![Value::Array(capture.clone())],
                ArrayIrType::Array(input_type.clone()),
                Options::new(mesh.clone()).with_donate(true),
            )
            .unwrap();
        let compiled = staged.lower(&domain).unwrap().compile(&domain).unwrap();
        let compiled_analysis = domain.analyze(compiled.executable()).unwrap();
        assert_eq!(domain.analyze(compiled.executable()).unwrap(), compiled_analysis);

        let bytes = domain.serialize_program(compiled.compiled_program()).unwrap().unwrap();
        let restored = domain.deserialize_program(bytes.as_slice()).unwrap().unwrap();
        assert_eq!(restored.input_types, compiled.compiled_program().input_types);
        assert_eq!(restored.output_types, compiled.compiled_program().output_types);
        assert_eq!(restored.signature, compiled.compiled_program().signature);
        assert_eq!(restored.runtime_requirements(), compiled.compiled_program().runtime_requirements());
        assert_eq!(restored.donation_flags, compiled.compiled_program().donation_flags);
        assert_eq!(restored.capture_count, compiled.compiled_program().capture_count);
        assert_eq!(restored.expected_argument_shardings, compiled.compiled_program().expected_argument_shardings);
        assert_eq!(restored.mesh, compiled.compiled_program().mesh);

        let header_size = PERSISTENT_EXECUTABLE_MAGIC.len() + size_of::<u64>();
        let metadata_size =
            u64::from_le_bytes(bytes[PERSISTENT_EXECUTABLE_MAGIC.len()..header_size].try_into().unwrap()) as usize;
        let metadata_end = header_size + metadata_size;
        // Reload validates the live facts before handing any native executable bytes to PJRT.
        let encode = |metadata: PersistentExecutableMetadataV9| {
            let metadata = serde_json::to_vec(&metadata).unwrap();
            let mut encoded = PERSISTENT_EXECUTABLE_MAGIC.to_vec();
            encoded.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
            encoded.extend_from_slice(&metadata);
            encoded.extend_from_slice(&bytes[metadata_end..]);
            encoded
        };
        let mut matching: PersistentExecutableMetadataV9 =
            serde_json::from_slice(&bytes[header_size..metadata_end]).unwrap();
        assert_eq!(matching.kernel_execution_facts, None);
        let facts = crate::kernels::KernelExecutionFacts::from_target(domain.target().unwrap(), &mesh).unwrap();
        matching.kernel_execution_facts = Some(facts.configuration_key().unwrap());
        let matching = domain.deserialize_program(&encode(matching)).unwrap().unwrap();
        assert_eq!(matching.kernel_execution_facts, Some(facts.configuration_key().unwrap()));
        let mut mismatch: PersistentExecutableMetadataV9 =
            serde_json::from_slice(&bytes[header_size..metadata_end]).unwrap();
        let mut changed_facts = facts;
        changed_facts.pjrt_version.minor += 1;
        mismatch.kernel_execution_facts = Some(changed_facts.configuration_key().unwrap());
        assert!(domain.deserialize_program(&encode(mismatch)).unwrap().is_none());
        let mut corrupt: PersistentExecutableMetadataV9 =
            serde_json::from_slice(&bytes[header_size..metadata_end]).unwrap();
        corrupt.kernel_execution_facts = Some(vec![255]);
        assert!(domain.deserialize_program(&encode(corrupt)).unwrap().is_none());
        let baseline: serde_json::Value = serde_json::from_slice(&bytes[header_size..metadata_end]).unwrap();
        assert_eq!(baseline["runtime_requirements"], serde_json::json!([]));
        let mut missing: PersistentExecutableMetadataV9 = serde_json::from_value(baseline).unwrap();
        missing.runtime_requirements = vec![crate::domains::runtimes::builtin_requirement("test.missing-runtime")];
        assert!(matches!(
            domain.deserialize_program(&encode(missing)),
            Err(Error::Runtime(crate::domains::runtimes::RuntimeError::MissingProvider { identifier }))
                if identifier == "test.missing-runtime"
        ));
        let mut explicit: PersistentExecutableMetadataV9 =
            serde_json::from_slice(&bytes[header_size..metadata_end]).unwrap();
        explicit.runtime_requirements =
            vec![builtin_requirement("ryft.assert"), builtin_requirement("ryft.ordered-io")];
        let explicit = domain.deserialize_program(&encode(explicit)).unwrap().unwrap();
        assert_eq!(
            explicit.runtime_requirements(),
            &[builtin_requirement("ryft.assert"), builtin_requirement("ryft.ordered-io")],
        );
        assert!(!explicit.requires_assertion_handler);
        assert!(!explicit.signature.has_ordered_io());

        let mut invalid_signature_metadata: PersistentExecutableMetadataV9 =
            serde_json::from_slice(&bytes[header_size..metadata_end]).unwrap();
        invalid_signature_metadata.input_mapping[0] = None;
        let invalid_signature_metadata = serde_json::to_vec(&invalid_signature_metadata).unwrap();
        let mut invalid_signature = PERSISTENT_EXECUTABLE_MAGIC.to_vec();
        invalid_signature.extend_from_slice(&(invalid_signature_metadata.len() as u64).to_le_bytes());
        invalid_signature.extend_from_slice(invalid_signature_metadata.as_slice());
        invalid_signature.extend_from_slice(&bytes[metadata_end..]);
        assert!(matches!(
            domain.deserialize_program(invalid_signature.as_slice()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { .. })),
        ));

        let mut incompatible_metadata: PersistentExecutableMetadataV9 =
            serde_json::from_slice(&bytes[header_size..metadata_end]).unwrap();
        incompatible_metadata.platform_version.push_str("-incompatible");
        let incompatible_metadata = serde_json::to_vec(&incompatible_metadata).unwrap();
        let mut incompatible = PERSISTENT_EXECUTABLE_MAGIC.to_vec();
        incompatible.extend_from_slice(&(incompatible_metadata.len() as u64).to_le_bytes());
        incompatible.extend_from_slice(incompatible_metadata.as_slice());
        incompatible.extend_from_slice(&bytes[metadata_end..]);
        assert!(domain.deserialize_program(incompatible.as_slice()).unwrap().is_none());

        let input = Array::from_host_buffer(
            &domain,
            input_type,
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        // A restored program owns the complete declaration contract, including the hidden capture prefix.
        assert!(matches!(
            domain.execute(&restored, vec![Value::Array(capture.clone())]),
            Err(Error::Core(ryft_core::Error::Program(ProgramError::InvalidInputCount { expected: 2, actual: 1 }))),
        ));
        let wrong_type = f64_vector(&domain, &mesh, &[1.0, 2.0, 3.0, 4.0]);
        for index in 0..2 {
            let expected = format!(
                "runtime input type {} does not refine declared type {}",
                wrong_type.r#type(),
                restored.input_types[index],
            );
            let mut inputs = vec![Value::Array(capture.clone()), Value::Array(input.clone())];
            inputs[index] = Value::Array(wrong_type.clone());
            assert!(matches!(
                domain.execute(&restored, inputs),
                Err(Error::Core(ryft_core::Error::Program(ProgramError::InvalidArgument { message })))
                    if message == expected,
            ));
        }
        let outputs = domain.execute(&restored, vec![Value::Array(capture), Value::Array(input)]).unwrap();
        assert_eq!(read_f32s(&client, program_array(&outputs[0])), vec![11.0, 22.0, 33.0, 44.0]);

        let analysis = restored.analyze().unwrap();
        assert_eq!(restored.analyze().unwrap(), analysis);
        assert!(analysis.to_json().unwrap().contains("\"properties\""));
        assert!(!analysis.to_string().is_empty());
    }

    #[test]
    fn test_persistent_executable_requires_runtime_contracts_for_effect_metadata() {
        use ryft_core::{AssertOperation, PrintOperation};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let mut builder = ProgramBuilder::new();
        let predicate = builder.add_input(replicated_scalar_type(&mesh, DataType::Boolean).into());
        let value = builder.add_input(replicated_scalar_type(&mesh, DataType::F32).into());
        builder
            .add_instruction(AssertOperation::<ArrayIrType>::new("test predicate"), Vec::new(), vec![predicate], None)
            .unwrap();
        builder
            .add_instruction(PrintOperation::<ArrayType>::new("test value"), Vec::new(), vec![value], None)
            .unwrap();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let lowered = domain.lower_program(&program, 0, &Options::new(mesh)).unwrap();
        let compiled = domain.compile_program(&lowered).unwrap();
        let bytes = domain.serialize_program(&compiled).unwrap().unwrap();
        let header_size = PERSISTENT_EXECUTABLE_MAGIC.len() + size_of::<u64>();
        let metadata_size = u64::from_le_bytes(bytes[8..header_size].try_into().unwrap()) as usize;
        let metadata_end = header_size + metadata_size;
        let baseline: PersistentExecutableMetadataV9 =
            serde_json::from_slice(&bytes[header_size..metadata_end]).unwrap();
        assert!(baseline.requires_assertion_handler);
        assert!(baseline.ordered_io);
        assert!(domain.deserialize_program(&bytes).unwrap().is_some());

        for identifier in ["ryft.assert", "ryft.ordered-io"] {
            for replacement in [None, Some((2, Vec::new())), Some((1, vec![1]))] {
                let mut metadata: PersistentExecutableMetadataV9 =
                    serde_json::from_slice(&bytes[header_size..metadata_end]).unwrap();
                metadata.runtime_requirements.retain(|requirement| requirement.identifier() != identifier);
                if let Some((version, configuration)) = replacement {
                    metadata
                        .runtime_requirements
                        .push(RuntimeRequirement::new(identifier, version, configuration).unwrap());
                }
                let metadata = serde_json::to_vec(&metadata).unwrap();
                let mut corrupt = PERSISTENT_EXECUTABLE_MAGIC.to_vec();
                corrupt.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
                corrupt.extend_from_slice(&metadata);
                corrupt.extend_from_slice(&bytes[metadata_end..]);

                assert!(matches!(
                    domain.deserialize_program(&corrupt),
                    Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                        if reason == format!("effect metadata requires runtime extension `{identifier}` with ABI version 1 and empty configuration"),
                ));
            }
        }
    }

    #[test]
    fn test_disk_cache_restores_into_a_fresh_compilation_context() {
        use ryft_core::{DiskCache, Sin};
        use tempfile::tempdir;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let directory = tempdir().unwrap();

        let first_domain = Session::with_disk_cache(
            &client,
            DiskCache::open(directory.path()).unwrap().with_write_thresholds(Duration::ZERO, 0),
        )
        .domain();
        let staged = crate::stage::<_, ArrayType, ArrayType>(
            |x| x.sin().unwrap(),
            input_type.clone(),
            Options::new(mesh.clone()),
        )
        .unwrap()
        .into_inner();
        let first: ryft_core::compilation::CompiledFunction<Domain<'_>, ArrayIrType, ArrayIrType> =
            staged.lower(&first_domain).unwrap().compile(&first_domain).unwrap();
        assert_eq!(first_domain.compilation_context().statistics().compilations, 1);
        drop(first);
        drop(first_domain);

        let second_domain = Session::with_disk_cache(
            &client,
            DiskCache::open(directory.path()).unwrap().with_write_thresholds(Duration::ZERO, 0),
        )
        .domain();
        let staged = crate::stage::<_, ArrayType, ArrayType>(|x| x.sin().unwrap(), input_type, Options::new(mesh))
            .unwrap()
            .into_inner();
        let _restored: ryft_core::compilation::CompiledFunction<Domain<'_>, ArrayIrType, ArrayIrType> =
            staged.lower(&second_domain).unwrap().compile(&second_domain).unwrap();
        let statistics = second_domain.compilation_context().statistics();
        assert_eq!(statistics.persistent_hits, 1);
        assert_eq!(statistics.compilations, 0, "restoration must not invoke backend compilation");
    }

    #[test]
    fn test_compile_rejects_an_exchanged_executable_with_a_foreign_invocation_contract() {
        use ryft_core::{
            CompilationArtifactExchange, CompilationArtifactExchangeError, CompilationArtifactExchangePolicy,
            CompilationCacheDomain, Sin,
        };

        // Follower-side exchange that answers every key with one fixed artifact. This is the reachable stand-in for
        // a colliding or corrupted persistent entry: the disk tier keys entries by the digest of the canonical key
        // bytes and re-validates that digest inside the stored envelope, so a planted disk entry can only ever be
        // rejected as corrupt, never resolved as a foreign executable.
        struct CollidingArtifactExchange {
            /// Serialized executable handed back for every requested key.
            artifact: Vec<u8>,
        }

        impl CompilationArtifactExchange for CollidingArtifactExchange {
            fn process_index(&self) -> usize {
                1
            }

            fn process_count(&self) -> usize {
                2
            }

            fn publish(&self, _key: &[u8], _artifact: &[u8]) -> Result<(), CompilationArtifactExchangeError> {
                Ok(())
            }

            fn receive(
                &self,
                _key: &[u8],
                _timeout: Duration,
            ) -> Result<Option<Vec<u8>>, CompilationArtifactExchangeError> {
                Ok(Some(self.artifact.clone()))
            }
        }

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();

        // The foreign executable keeps the requesting lowering's boundary types so that it loads and passes every
        // core-level output-type check, and differs only in the donation contract the runtime call path trusts.
        let producer = Session::new(&client).domain();
        let staged = crate::stage::<_, ArrayType, ArrayType>(
            |x| x.sin().unwrap(),
            input_type.clone(),
            Options::new(mesh.clone()).with_donate(true),
        )
        .unwrap()
        .into_inner();
        let foreign: ryft_core::compilation::CompiledFunction<Domain<'_>, ArrayIrType, ArrayIrType> =
            staged.lower(&producer).unwrap().compile(&producer).unwrap();
        let artifact = producer.serialize_program(foreign.compiled_program()).unwrap().unwrap();

        let consumer = |policy| {
            let exchange: Arc<dyn CompilationArtifactExchange> =
                Arc::new(CollidingArtifactExchange { artifact: artifact.clone() });
            Session::with_compilation_context(
                &client,
                CompilationContext::new().with_artifact_exchange(exchange, policy),
            )
            .domain()
        };
        let staged = crate::stage::<_, ArrayType, ArrayType>(|x| x.sin().unwrap(), input_type, Options::new(mesh))
            .unwrap()
            .into_inner();

        // The received executable loads cleanly and reports matching boundary types, so only the validation against
        // the requesting lowering catches the foreign invocation contract. The executable is then an incompatible
        // miss, so a follower that may fall back compiles its own executable.
        let fallback_consumer =
            consumer(CompilationArtifactExchangePolicy::PreferSharing { timeout: Duration::from_secs(5) });
        let lowered = staged.lower(&fallback_consumer).unwrap();
        let compiled = lowered.compile(&fallback_consumer).unwrap();
        assert!(
            fallback_consumer
                .validate_cached_program(lowered.lowered_program(), compiled.compiled_program())
                .is_ok()
        );
        let statistics = fallback_consumer.compilation_context().statistics();
        assert_eq!(statistics.exchange_hits, 0);
        assert_eq!(statistics.exchange_fallbacks, 1);
        assert_eq!(statistics.compilations, 1);

        // A follower that requires sharing fails instead.
        let required_consumer =
            consumer(CompilationArtifactExchangePolicy::RequireSharing { timeout: Duration::from_secs(5) });
        assert!(matches!(
            staged.lower(&required_consumer).unwrap().compile(&required_consumer),
            Err(Error::Core(ryft_core::Error::Program(ProgramError::InvalidArgument { message })))
                if message == "compilation artifact exchange requirement was not satisfied: received compilation \
                               artifact is incompatible: incompatible XLA executable: cached executable has \
                               incompatible donation flags",
        ));
        assert_eq!(required_consumer.compilation_context().statistics().compilations, 0);
    }

    #[test]
    fn test_persistent_executable_rejects_malformed_and_incompatible_metadata() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let domain = Session::new(&client).domain();

        assert!(matches!(
            domain.deserialize_program(b"truncated"),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { .. })),
        ));
        let mut legacy = b"RYFTXLA8".to_vec();
        legacy.extend_from_slice(&0u64.to_le_bytes());
        assert!(domain.deserialize_program(legacy.as_slice()).unwrap().is_none());

        let metadata = PersistentExecutableMetadataV9 {
            schema_version: PERSISTENT_EXECUTABLE_SCHEMA_VERSION + 1,
            feature_flags: 0,
            compilation_options: CompilationOptions::default().encode_to_vec(),
            signature: PersistentArraySignatureV3::encode(&[], &[]),
            input_mapping: Vec::new(),
            output_mapping: Vec::new(),
            input_dimensions: Vec::new(),
            output_dimensions: Vec::new(),
            output_count: 0,
            reference_states: Vec::new(),
            requires_assertion_handler: false,
            runtime_requirements: Vec::new(),
            ordered_io: false,
            unordered_io: false,
            device_ordered_io: false,
            effect_abi_version: 1,
            donation_flags: Vec::new(),
            capture_count: 0,
            expected_argument_shardings: Vec::new(),
            mesh: PersistentDeviceMeshV1 {
                logical_mesh: PersistentLogicalMeshV1 { axes: Vec::new() },
                devices: Vec::new(),
            },
            device_kinds: Vec::new(),
            kernel_execution_facts: None,
            replica_count: 0,
            partition_count: 0,
            device_assignment: Vec::new(),
            platform_name: client.platform_name().unwrap().into_owned(),
            platform_version: client.platform_version().unwrap().into_owned(),
            compiler_identity: COMPILER_IDENTITY.to_string(),
            xla_flags: std::env::var("XLA_FLAGS").unwrap_or_default(),
            compilation_duration_nanoseconds: None,
        };
        let metadata = serde_json::to_vec(&metadata).unwrap();
        let mut bytes = PERSISTENT_EXECUTABLE_MAGIC.to_vec();
        bytes.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        bytes.extend_from_slice(metadata.as_slice());
        assert!(domain.deserialize_program(bytes.as_slice()).unwrap().is_none());
    }

    #[test]
    fn test_persistent_v6_round_trips_and_validates_external_reference_state() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);

        let mut builder = ProgramBuilder::new();
        let reference = builder.add_input(ArrayIrType::Reference(ReferenceType::new(scalar_type.clone())));
        let read_only = builder.add_input(ArrayIrType::Reference(ReferenceType::new(scalar_type)));
        let read_only_output =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![read_only], None).unwrap()[0];
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, read_only_output], None)
            .unwrap();
        let left =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let output = builder
            .add_instruction(AddOperation::new(), Vec::new(), vec![left, read_only_output], None)
            .unwrap()[0];
        let program: FlatProgram = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let lowered = domain.lower_program(&program, 1, &Options::new(mesh.clone())).unwrap();
        let compiled = domain.compile_program(&lowered).unwrap();
        let bytes = compiled.serialize().unwrap().unwrap();
        let restored = domain.deserialize_compiled_program(bytes.as_slice()).unwrap().unwrap();
        assert_eq!(compiled.output_count, 1);
        assert_eq!(compiled.reference_states.len(), 2);
        assert_eq!(compiled.reference_states[0].source(), ReferenceSource::Capture { index: 0 });
        assert_eq!(compiled.reference_states[0].source().flat_input_index(compiled.capture_count).unwrap(), 0);
        assert_eq!(compiled.reference_states[0].output_index(), Some(1));
        assert_eq!(compiled.reference_states[1].source(), ReferenceSource::Input { index: 0 });
        assert_eq!(compiled.reference_states[1].source().flat_input_index(compiled.capture_count).unwrap(), 1);
        assert_eq!(compiled.reference_states[1].output_index(), None);
        assert_eq!(restored.output_count, compiled.output_count);
        assert_eq!(restored.reference_states, compiled.reference_states);
        let outputs = domain
            .execute_program(&restored, vec![f32_scalar(&domain, &mesh, 1.0), f32_scalar(&domain, &mesh, 2.0)])
            .unwrap();
        assert_eq!(read_f32s(&client, &outputs[0]), vec![5.0]);
        assert_eq!(read_f32s(&client, &outputs[1]), vec![3.0]);

        let header_size = PERSISTENT_EXECUTABLE_MAGIC.len() + size_of::<u64>();
        let metadata_size =
            u64::from_le_bytes(bytes[PERSISTENT_EXECUTABLE_MAGIC.len()..header_size].try_into().unwrap()) as usize;
        let metadata_end = header_size + metadata_size;
        let corrupt = |metadata: PersistentExecutableMetadataV9| {
            let metadata = serde_json::to_vec(&metadata).unwrap();
            let mut corrupted = PERSISTENT_EXECUTABLE_MAGIC.to_vec();
            corrupted.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
            corrupted.extend_from_slice(metadata.as_slice());
            corrupted.extend_from_slice(&bytes[metadata_end..]);
            corrupted
        };
        let mut metadata: PersistentExecutableMetadataV9 =
            serde_json::from_slice(&bytes[header_size..metadata_end]).unwrap();
        assert_eq!(metadata.reference_states[0].source, PersistentReferenceSourceV6::Capture { index: 0 });
        assert_eq!(metadata.reference_states[0].logical_input_index, 0);
        assert_eq!(metadata.reference_states[1].source, PersistentReferenceSourceV6::PublicInput { index: 0 });
        assert_eq!(metadata.reference_states[1].logical_input_index, 1);
        metadata.reference_states[0].logical_input_index = 1;
        let corrupted = corrupt(metadata);
        assert!(matches!(
            domain.deserialize_compiled_program(corrupted.as_slice()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "reference-state source does not match its logical input",
        ));

        let mut metadata: PersistentExecutableMetadataV9 =
            serde_json::from_slice(&bytes[header_size..metadata_end]).unwrap();
        metadata.reference_states[0].source = PersistentReferenceSourceV6::PublicInput { index: 1 };
        let corrupted = corrupt(metadata);
        assert!(matches!(
            domain.deserialize_compiled_program(corrupted.as_slice()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "reference-state source does not match its logical input",
        ));

        // An output count beyond the decoded complete output arity cannot name a hidden final-state suffix at all.
        let mut invalid_outputs: PersistentExecutableMetadataV9 =
            serde_json::from_slice(&bytes[header_size..metadata_end]).unwrap();
        invalid_outputs.output_count = 3;
        let corrupted = corrupt(invalid_outputs);
        assert!(matches!(
            domain.deserialize_compiled_program(corrupted.as_slice()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "reference-state metadata has an invalid output count",
        ));

        // Dropping one expected argument sharding leaves the persisted list shorter than the physical boundary, so
        // no reference-state input could be checked against its own expected sharding.
        let mut truncated_expected_shardings: PersistentExecutableMetadataV9 =
            serde_json::from_slice(&bytes[header_size..metadata_end]).unwrap();
        assert_eq!(truncated_expected_shardings.expected_argument_shardings.len(), 2);
        truncated_expected_shardings.expected_argument_shardings.pop();
        let corrupted = corrupt(truncated_expected_shardings);
        assert!(matches!(
            domain.deserialize_compiled_program(corrupted.as_slice()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "expected argument shardings do not match the physical input count",
        ));

        let mut reordered: PersistentExecutableMetadataV9 =
            serde_json::from_slice(&bytes[header_size..metadata_end]).unwrap();
        reordered.reference_states.swap(0, 1);
        let corrupted = corrupt(reordered);
        assert!(matches!(
            domain.deserialize_compiled_program(corrupted.as_slice()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "reference states are not in canonical logical input order",
        ));

        let mut foreign_process_mesh: PersistentExecutableMetadataV9 =
            serde_json::from_slice(&bytes[header_size..metadata_end]).unwrap();
        foreign_process_mesh.mesh.devices[0].process_index += 1;
        let corrupted = corrupt(foreign_process_mesh);
        assert!(matches!(
            domain.deserialize_compiled_program(corrupted.as_slice()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "reference-state mesh does not include an addressable device of this process",
        ));

        let mut mismatched_expected_sharding: PersistentExecutableMetadataV9 =
            serde_json::from_slice(&bytes[header_size..metadata_end]).unwrap();
        mismatched_expected_sharding.expected_argument_shardings[0].unreduced_axes.push("x".to_string());
        let corrupted = corrupt(mismatched_expected_sharding);
        assert!(matches!(
            domain.deserialize_compiled_program(corrupted.as_slice()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "reference-state input 0 logical sharding does not match its expected argument sharding",
        ));

        let mut mismatched_input: PersistentExecutableMetadataV9 =
            serde_json::from_slice(&bytes[header_size..metadata_end]).unwrap();
        mismatched_input.signature.input_types[0]
            .sharding
            .as_mut()
            .unwrap()
            .unreduced_axes
            .push("x".to_string());
        mismatched_input.signature.output_types[1]
            .sharding
            .as_mut()
            .unwrap()
            .unreduced_axes
            .push("x".to_string());
        let corrupted = corrupt(mismatched_input);
        assert!(matches!(
            domain.deserialize_compiled_program(corrupted.as_slice()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "reference-state input 0 logical sharding does not match its expected argument sharding",
        ));

        let mut mismatched_output: PersistentExecutableMetadataV9 =
            serde_json::from_slice(&bytes[header_size..metadata_end]).unwrap();
        mismatched_output.signature.output_types[1]
            .sharding
            .as_mut()
            .unwrap()
            .unreduced_axes
            .push("x".to_string());
        let corrupted = corrupt(mismatched_output);
        assert!(matches!(
            domain.deserialize_compiled_program(corrupted.as_slice()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "reference-state output 1 logical sharding does not match state input 0",
        ));

        let mut zero_space: PersistentExecutableMetadataV9 =
            serde_json::from_slice(&bytes[header_size..metadata_end]).unwrap();
        zero_space.signature.input_types[0].data_type = encode_data_type(DataType::Zero);
        let corrupted = corrupt(zero_space);
        assert!(matches!(
            domain.deserialize_compiled_program(corrupted.as_slice()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "reference-state input has a zero-space type",
        ));

        let mut host_memory: PersistentExecutableMetadataV9 =
            serde_json::from_slice(&bytes[header_size..metadata_end]).unwrap();
        host_memory.signature.input_types[0].memory = PersistentMemoryV1::Host(true);
        let corrupted = corrupt(host_memory);
        assert!(matches!(
            domain.deserialize_compiled_program(corrupted.as_slice()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "reference-state input is not in device memory",
        ));

        let mut unbounded_dynamic: PersistentExecutableMetadataV9 =
            serde_json::from_slice(&bytes[header_size..metadata_end]).unwrap();
        unbounded_dynamic.signature.variables.push(PersistentDimensionVariableV3 {
            name: "unbounded".to_string(),
            lower: 0,
            upper: None,
        });
        unbounded_dynamic.signature.input_types[1].shape = vec![PersistentDimensionV3::Dynamic { variable: 0 }];
        unbounded_dynamic.signature.input_types[1].sharding.as_mut().unwrap().dimensions =
            vec![PersistentShardingDimensionV1::Replicated];
        let corrupted = corrupt(unbounded_dynamic);
        assert!(matches!(
            domain.deserialize_compiled_program(corrupted.as_slice()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "reference-state input has an unbounded dynamic dimension",
        ));

        let mut mutated_dynamic: PersistentExecutableMetadataV9 =
            serde_json::from_slice(&bytes[header_size..metadata_end]).unwrap();
        mutated_dynamic.signature.variables.push(PersistentDimensionVariableV3 {
            name: "bounded_mutation".to_string(),
            lower: 0,
            upper: Some(5),
        });
        mutated_dynamic.signature.input_types[0].shape = vec![PersistentDimensionV3::Dynamic { variable: 0 }];
        mutated_dynamic.signature.input_types[0].sharding.as_mut().unwrap().dimensions =
            vec![PersistentShardingDimensionV1::Replicated];
        let corrupted = corrupt(mutated_dynamic);
        assert!(matches!(
            domain.deserialize_compiled_program(corrupted.as_slice()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "mutated bounded-dynamic reference-state input 0 is unsupported because backend alias \
                              compatibility has not been verified",
        ));

        let mut sharded_dynamic: PersistentExecutableMetadataV9 =
            serde_json::from_slice(&bytes[header_size..metadata_end]).unwrap();
        sharded_dynamic.signature.variables.push(PersistentDimensionVariableV3 {
            name: "bounded_sharded".to_string(),
            lower: 0,
            upper: Some(5),
        });
        sharded_dynamic.signature.input_types[1].shape = vec![PersistentDimensionV3::Dynamic { variable: 0 }];
        sharded_dynamic.signature.input_types[1].sharding.as_mut().unwrap().dimensions =
            vec![PersistentShardingDimensionV1::Sharded(vec!["x".to_string()])];
        let corrupted = corrupt(sharded_dynamic);
        assert!(matches!(
            domain.deserialize_compiled_program(corrupted.as_slice()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "bounded-dynamic reference-state input 1 logical sharding is not replicated",
        ));
    }

    #[test]
    fn test_persistent_v6_round_trips_bounded_dynamic_read_only_reference_state() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let extent = DimensionVariable::new("state_extent", DimensionBounds::new(1, Some(5)).unwrap());
        let state_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Dynamic(extent)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let lowered = domain
            .lower_program(&external_reference_read_program(state_type.clone()), 0, &Options::new(mesh.clone()))
            .unwrap();
        let compiled = domain.compile_program(&lowered).unwrap();
        let bytes = compiled.serialize().unwrap().unwrap();
        let restored = domain.deserialize_compiled_program(bytes.as_slice()).unwrap().unwrap();

        assert_eq!(restored.input_types[0].data_type(), state_type.data_type());
        assert_eq!(restored.input_types[0].shape().to_string(), state_type.shape().to_string());
        assert_eq!(restored.input_types[0].sharding(), state_type.sharding());
        assert_eq!(restored.input_types[0].memory(), state_type.memory());
        assert_eq!(restored.output_count, 1);
        assert_eq!(
            restored.reference_states.as_ref(),
            &[ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, None)],
        );

        // Runtime holders have exact extents. The restored declarations retain the bounds against which they refine.
        let array = Array::from_host_buffer(
            &domain,
            replicated_vector_type(&mesh, 3),
            mesh,
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0]).as_slice(),
        )
        .unwrap();
        let holder = ArrayReference::new(array);
        let reference = Value::Reference(crate::Reference::new(holder, domain.clone()));
        let outputs = domain.execute_statefully(&restored, vec![reference]).r#await().unwrap();
        assert_eq!(read_f32s(&client, program_array(&outputs[0])), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_persistent_v6_preserves_static_sharded_reference_metadata_and_round_trips_when_supported() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 2);
        let domain = Session::new(&client).domain();
        let state_sharding =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let state_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(state_sharding.clone())
            .unwrap();
        let mut builder = ProgramBuilder::new();
        let reference = builder.add_input(ArrayIrType::Reference(ReferenceType::new(state_type.clone())));
        let update = builder.add_input(ArrayIrType::Array(state_type.clone()));
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let output =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program: FlatProgram = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let lowered = domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = domain.compile_program(&lowered).unwrap();
        assert_eq!(compiled.input_types.as_ref(), &[state_type.clone(), state_type.clone()]);
        assert_eq!(compiled.output_types.as_ref(), &[state_type.clone(), state_type.clone()]);
        assert_eq!(compiled.expected_argument_shardings.as_ref(), &[state_sharding.clone(), state_sharding]);
        assert_eq!(compiled.mesh, mesh);
        assert_eq!(
            compiled.reference_states.as_ref(),
            &[ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, Some(1))],
        );
        match compiled.serialize().unwrap() {
            Some(bytes) => {
                let restored = domain.deserialize_compiled_program(bytes.as_slice()).unwrap().unwrap();
                assert_eq!(restored.input_types, compiled.input_types);
                assert_eq!(restored.output_types, compiled.output_types);
                assert_eq!(restored.reference_states, compiled.reference_states);
                assert_eq!(restored.expected_argument_shardings, compiled.expected_argument_shardings);
                assert_eq!(restored.mesh, compiled.mesh);
            }
            None => {
                // The CPU plugin does not implement executable serialization for this two-device program, so `None`
                // is the documented optional-capability answer rather than a silently skipped round trip. Either the
                // executable query or its serialization may be the unimplemented step.
                let unsupported = match compiled.executable().executable() {
                    Ok(executable) => executable.serialize().err(),
                    Err(error) => Some(error),
                };
                assert!(matches!(unsupported, Some(ryft_pjrt::Error::Unimplemented { .. })));
            }
        }
    }

    #[test]
    fn test_persistent_sharding_unreduced_kinds() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sum = Sharding::replicated(mesh, 1).with_unreduced_axes(["x"]).unwrap();
        let encode = |sharding: &Sharding| serde_json::to_string(&PersistentShardingV1::from(sharding)).unwrap();
        let decode =
            |payload: &str| Sharding::try_from(serde_json::from_str::<PersistentShardingV1>(payload).unwrap()).unwrap();

        // A sum is omitted from the payload, so the payloads (and cache keys) of sums are unchanged, while other kinds
        // round trip through their names.
        assert!(!encode(&sum).contains("unreduced_kind"));
        assert_eq!(decode(&encode(&sum)), sum);
        for kind in [UnreducedKind::Max, UnreducedKind::Min] {
            let sharding = sum.clone().with_unreduced_kind(kind).unwrap();
            assert!(encode(&sharding).contains(&format!("\"unreduced_kind\":\"{kind}\"")));
            assert_eq!(decode(&encode(&sharding)), sharding);
        }

        // A kind without unreduced axes is malformed.
        let malformed = encode(&Sharding::replicated(sum.mesh().clone(), 1))
            .replace("\"reduced_axes\"", "\"unreduced_kind\":\"max\",\"reduced_axes\"");
        assert!(Sharding::try_from(serde_json::from_str::<PersistentShardingV1>(&malformed).unwrap()).is_err());
    }

    #[test]
    fn test_persistent_data_type_codes() {
        // The persistent executable encoding assigns each data type a stable one-byte code, and so this test pins
        // every assignment: new data types must be appended with fresh codes and existing codes must never be
        // renumbered, because persisted executables would otherwise decode to the wrong types.
        let data_types = [
            (DataType::Token, 0),
            (DataType::Boolean, 1),
            (DataType::I1, 2),
            (DataType::I2, 3),
            (DataType::I4, 4),
            (DataType::I8, 5),
            (DataType::I16, 6),
            (DataType::I32, 7),
            (DataType::I64, 8),
            (DataType::U1, 9),
            (DataType::U2, 10),
            (DataType::U4, 11),
            (DataType::U8, 12),
            (DataType::U16, 13),
            (DataType::U32, 14),
            (DataType::U64, 15),
            (DataType::F4E2M1FN, 16),
            (DataType::F8E3M4, 17),
            (DataType::F8E4M3, 18),
            (DataType::F8E4M3FN, 19),
            (DataType::F8E4M3FNUZ, 20),
            (DataType::F8E4M3B11FNUZ, 21),
            (DataType::F8E5M2, 22),
            (DataType::F8E5M2FNUZ, 23),
            (DataType::F8E8M0FNU, 24),
            (DataType::BF16, 25),
            (DataType::F16, 26),
            (DataType::F32, 27),
            (DataType::F64, 28),
            (DataType::C64, 29),
            (DataType::C128, 30),
            (DataType::F6E2M3FN, 31),
            (DataType::F6E3M2FN, 32),
            (DataType::Zero, 33),
        ];
        for (data_type, code) in data_types {
            assert_eq!(encode_data_type(data_type), code);
            assert_eq!(decode_data_type(code).unwrap(), data_type);
        }
        assert!(decode_data_type(34).is_err());
    }

    #[test]
    fn test_persistent_array_type_round_trips_zero_space_metadata() {
        let batch = DimensionVariable::new("batch", DimensionBounds::non_negative(Some(4)).unwrap());
        let array_type = ArrayType::new(
            DataType::Zero,
            Shape::new(vec![CoreDimension::Static(2), CoreDimension::Dynamic(batch.clone())]),
        )
        .with_memory(Memory::Host { pinned: true });
        let output_type =
            ArrayType::new(DataType::Zero, Shape::new(vec![CoreDimension::Dynamic(batch), CoreDimension::Static(2)]));

        let (restored_inputs, restored_outputs) =
            PersistentArraySignatureV3::encode(std::slice::from_ref(&array_type), std::slice::from_ref(&output_type))
                .decode()
                .unwrap();

        assert_eq!(restored_inputs[0].data_type(), array_type.data_type());
        assert_eq!(restored_inputs[0].memory(), array_type.memory());
        assert_eq!(restored_inputs[0].dimension(0), CoreDimension::Static(2));
        assert_eq!(restored_inputs[0].dimension(1).variable().unwrap().name(), "batch");
        assert_eq!(
            restored_inputs[0].dimension(1).variable().unwrap().bounds(),
            DimensionBounds::non_negative(Some(4)).unwrap(),
        );
        assert_eq!(restored_outputs[0].data_type(), output_type.data_type());
        assert_eq!(restored_inputs[0].dimension(1), restored_outputs[0].dimension(0));
    }

    #[test]
    fn test_persistent_array_signature_canonicalizes_names_but_preserves_variable_relationships() {
        let bounds = DimensionBounds::non_negative(Some(4)).unwrap();
        let batch = DimensionVariable::new("batch", bounds);
        let renamed = DimensionVariable::new("renamed", bounds);
        let shared = ArrayType::new(
            DataType::F32,
            Shape::new(vec![CoreDimension::Dynamic(batch.clone()), CoreDimension::Dynamic(batch)]),
        );
        let shared_renamed = ArrayType::new(
            DataType::F32,
            Shape::new(vec![CoreDimension::Dynamic(renamed.clone()), CoreDimension::Dynamic(renamed)]),
        );
        let first = DimensionVariable::new("first", bounds);
        let second = DimensionVariable::new("second", bounds);
        let independent = ArrayType::new(
            DataType::F32,
            Shape::new(vec![CoreDimension::Dynamic(first), CoreDimension::Dynamic(second)]),
        );

        let canonical = |r#type: &ArrayType| canonical_boundary_types(std::slice::from_ref(r#type), &[]);
        assert_eq!(canonical(&shared), canonical(&shared_renamed));
        assert_ne!(canonical(&shared), canonical(&independent));
    }
}
