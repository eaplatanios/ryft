//! Persistent serialization of compiled XLA executables together with their invocation contracts.
//!
//! `CompiledProgram::serialize` writes a versioned envelope that consists of a fixed magic header, the little-endian
//! length of a JSON metadata record, that record, and the serialized PJRT executable bytes. The metadata records
//! everything that executions need besides the native executable (i.e., the logical signature with its shared dynamic
//! dimension variables, the mapping between logical values and physical executable slots, external reference states,
//! effects, runtime requirements, shardings, the device mesh and assignment, and the identity of the target and the
//! compiler) in stable explicit encodings rather than Rust enum discriminants.
//!
//! `Session::deserialize_compiled_program` validates that metadata against the loading session before it loads the
//! native bytes. Envelopes from other schemas, targets, compilers, or device assignments load as `None` so that
//! callers can recompile, while malformed metadata of a supported schema is an error. Persistent cache directories
//! must be trusted sources of executable code.
//!
//! `PersistentArraySignature::into_canonical` also provides the representation of logical boundary types that
//! compilation keys use, which omits the diagnostic names of dimension variables.

use std::time::Duration;

use prost::Message;
use ryft_core::{
    ArrayType, DataType, Device, DeviceMesh, Dimension, DimensionBounds, DimensionVariable, ExternalReferenceBinding,
    Layout, LogicalMesh, Memory, MeshAxis, MeshAxisType, ReferenceSource, Shape, Sharding, ShardingDimension,
    StridedLayout, Tile, TileDimension, TiledLayout, UnreducedKind,
};
use ryft_pjrt::protos::CompilationOptions;
use ryft_pjrt::{BufferType, LoadOptions};
use serde::{Deserialize, Serialize};

use crate::compilation::CompilationError;
use crate::compilation::keys::COMPILER_IDENTITY;
use crate::compilation::programs::CompiledProgram;
use crate::execution::{ExecutionPlan, flatten_device_assignment};
use crate::lowering::signature::ExecutableSignature;
use crate::{Error, RuntimeRequirement, Session, normalize_runtime_requirements};

// TODO(eaplatanios): Review this module.

/// Fixed header identifying the executable envelope.
const PERSISTENT_EXECUTABLE_MAGIC: &[u8; 8] = b"RYFTXL01";

/// Metadata schema emitted by executable serialization.
const PERSISTENT_EXECUTABLE_SCHEMA_VERSION: u32 = 1;

/// Required features for reference states, dynamic extents, effects, and runtime contracts.
const PERSISTENT_EXECUTABLE_FEATURE_FLAGS: u64 = 15;

/// Persisted binding recipe for one logical reference state.
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct PersistentReferenceState {
    /// Source of the external reference among the captures or the public inputs.
    source: PersistentReferenceSource,

    /// Redundant flat position retained solely for corruption detection.
    logical_input_index: u64,

    /// Hidden logical result containing the final state, or `None` for a read-only reference.
    logical_output_index: Option<u64>,
}

impl PersistentReferenceState {
    /// Encodes logical reference state slots in their stable shape without runtime reference handles.
    ///
    /// The record redundantly stores each flat logical input position. Deriving that coordinate from the canonical
    /// source keeps valid serialized bytes unchanged without reintroducing the redundant field into
    /// [`ExternalReferenceBinding`].
    pub(super) fn encode(states: &[ExternalReferenceBinding], capture_count: usize) -> Result<Vec<Self>, Error> {
        states
            .iter()
            .map(|state| {
                let logical_input_index = state.source().flat_input_index(capture_count)?;
                let source = match state.source() {
                    ReferenceSource::Capture { index } => PersistentReferenceSource::Capture { index: index as u64 },
                    ReferenceSource::Input { index } => PersistentReferenceSource::PublicInput { index: index as u64 },
                };
                Ok(Self {
                    source,
                    logical_input_index: logical_input_index as u64,
                    logical_output_index: state.output_index().map(|index| index as u64),
                })
            })
            .collect()
    }

    /// Decodes logical reference state slots without runtime reference handles and validates the redundant flat input
    /// coordinate.
    fn decode(states: Vec<Self>, capture_count: usize) -> Result<Vec<ExternalReferenceBinding>, Error> {
        states
            .into_iter()
            .map(|state| {
                let source = match state.source {
                    PersistentReferenceSource::Capture { index } => ReferenceSource::Capture {
                        index: usize::try_from(index).map_err(|_| {
                            Error::Compilation(CompilationError::InvalidPersistentExecutable {
                                reason: format!("value {index} does not fit in `usize`"),
                            })
                        })?,
                    },
                    PersistentReferenceSource::PublicInput { index } => ReferenceSource::Input {
                        index: usize::try_from(index).map_err(|_| {
                            Error::Compilation(CompilationError::InvalidPersistentExecutable {
                                reason: format!("value {index} does not fit in `usize`"),
                            })
                        })?,
                    },
                };
                let logical_input_index = usize::try_from(state.logical_input_index).map_err(|_| {
                    Error::Compilation(CompilationError::InvalidPersistentExecutable {
                        reason: format!("value {} does not fit in `usize`", state.logical_input_index),
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
                                    reason: format!("value {value} does not fit in `usize`"),
                                })
                            })
                        })
                        .transpose()?,
                ))
            })
            .collect()
    }
}

/// Persisted mapping from a logical axis to a hidden physical slot for one bounded dynamic dimension.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct PersistentBoundaryDimension {
    /// Logical flattened input or output index containing the dynamic axis.
    logical_index: u64,

    /// Axis within the logical array.
    axis: u64,

    /// Hidden physical argument or result index carrying the runtime extent.
    physical_index: u64,
}

impl PersistentBoundaryDimension {
    /// Encodes the hidden extent slots of the bounded dynamic inputs of one executable signature.
    pub(super) fn from_input_dimensions(signature: &ExecutableSignature) -> Vec<Self> {
        signature
            .input_dimensions()
            .iter()
            .map(|dimension| Self {
                logical_index: dimension.logical_input_index() as u64,
                axis: dimension.axis() as u64,
                physical_index: dimension.physical_input_index() as u64,
            })
            .collect()
    }

    /// Encodes the hidden extent slots of the bounded dynamic outputs of one executable signature.
    pub(super) fn from_output_dimensions(signature: &ExecutableSignature) -> Vec<Self> {
        signature
            .output_dimensions()
            .iter()
            .map(|dimension| Self {
                logical_index: dimension.logical_output_index() as u64,
                axis: dimension.axis() as u64,
                physical_index: dimension.physical_output_index() as u64,
            })
            .collect()
    }
}

/// Logical boundary types with a shared table of dimension variables. Decoding recreates each variable once and
/// preserves its diagnostic name.
#[derive(Serialize, Deserialize)]
pub(super) struct PersistentArraySignature {
    /// Shared dynamic variables numbered by first occurrence across inputs and outputs.
    variables: Vec<PersistentDimensionVariable>,

    /// Logical input array types, including captures.
    input_types: Vec<PersistentArrayType>,

    /// Logical output array types, including hidden final reference states.
    output_types: Vec<PersistentArrayType>,
}

impl PersistentArraySignature {
    /// Encodes input and output types with shared dynamic dimension variables numbered by first occurrence.
    pub(super) fn encode(input_types: &[ArrayType], output_types: &[ArrayType]) -> Self {
        let mut variables = Vec::<DimensionVariable>::new();
        let mut encode_type = |r#type: &ArrayType| {
            let shape = r#type
                .shape()
                .dimensions()
                .iter()
                .map(|dimension| match dimension {
                    Dimension::Static(value) => PersistentDimension::Static(*value as u64),
                    Dimension::Dynamic(variable) => {
                        let index = variables.iter().position(|existing| existing == variable).unwrap_or_else(|| {
                            variables.push(variable.clone());
                            variables.len() - 1
                        });
                        PersistentDimension::Dynamic { variable: index as u64 }
                    }
                })
                .collect();
            PersistentArrayType {
                data_type: encode_data_type(r#type.data_type()),
                shape,
                layout: r#type.layout().map(PersistentLayout::from),
                sharding: r#type.sharding().map(PersistentSharding::from),
                memory: match r#type.memory() {
                    Memory::Device => PersistentMemory::Device,
                    Memory::Host { pinned } => PersistentMemory::Host(pinned),
                },
            }
        };
        let input_types = input_types.iter().map(&mut encode_type).collect();
        let output_types = output_types.iter().map(&mut encode_type).collect();
        let variables = variables
            .into_iter()
            .map(|variable| PersistentDimensionVariable {
                name: variable.name().to_string(),
                lower: variable.bounds().lower() as u64,
                upper: variable.bounds().upper().map(|upper| upper as u64),
            })
            .collect();
        Self { variables, input_types, output_types }
    }

    /// Removes variable names, which only serve diagnostics, from the stable representation that compilation keys use.
    pub(super) fn into_canonical(self) -> PersistentCanonicalArraySignature {
        PersistentCanonicalArraySignature {
            variables: self
                .variables
                .into_iter()
                .map(|variable| PersistentCanonicalDimensionVariable { lower: variable.lower, upper: variable.upper })
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
                            reason: format!("value {} does not fit in `usize`", variable.lower),
                        })
                    })?,
                    variable
                        .upper
                        .map(|value| {
                            usize::try_from(value).map_err(|_| {
                                Error::Compilation(CompilationError::InvalidPersistentExecutable {
                                    reason: format!("value {value} does not fit in `usize`"),
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
        let decode_type = |r#type: PersistentArrayType| -> Result<ArrayType, Error> {
            let shape = Shape::new(
                r#type
                    .shape
                    .into_iter()
                    .map(|dimension| match dimension {
                        PersistentDimension::Static(value) => usize::try_from(value)
                            .map_err(|_| {
                                Error::Compilation(CompilationError::InvalidPersistentExecutable {
                                    reason: format!("value {value} does not fit in `usize`"),
                                })
                            })
                            .map(Dimension::Static),
                        PersistentDimension::Dynamic { variable } => variables
                            .get(usize::try_from(variable).map_err(|_| {
                                Error::Compilation(CompilationError::InvalidPersistentExecutable {
                                    reason: format!("value {variable} does not fit in `usize`"),
                                })
                            })?)
                            .cloned()
                            .map(Dimension::Dynamic)
                            .ok_or_else(|| {
                                Error::Compilation(CompilationError::InvalidPersistentExecutable {
                                    reason: "array type references an unknown dimension variable".into(),
                                })
                            }),
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            );
            let memory = match r#type.memory {
                PersistentMemory::Device => Memory::Device,
                PersistentMemory::Host(pinned) => Memory::Host { pinned },
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

/// Cache identity for logical boundary types. Variable names are omitted and variables are numbered by first
/// occurrence, preserving their sharing and bounds.
#[derive(Serialize)]
pub(super) struct PersistentCanonicalArraySignature {
    /// Shared dynamic variables numbered by first occurrence across inputs and outputs.
    variables: Vec<PersistentCanonicalDimensionVariable>,

    /// Logical input array types, including captures.
    input_types: Vec<PersistentArrayType>,

    /// Logical output array types, including hidden final reference states.
    output_types: Vec<PersistentArrayType>,
}

/// Stable logical sharding record. Conversions retain the mesh, dimension assignments, and reduction state.
#[derive(Serialize, Deserialize)]
pub(super) struct PersistentSharding {
    /// Logical mesh used by the sharding.
    mesh: PersistentLogicalMesh,

    /// Sharding of each array dimension in axis order.
    dimensions: Vec<PersistentShardingDimension>,

    /// Mesh axes whose partial values await reduction.
    unreduced_axes: Vec<String>,

    /// Reduction required to combine partial values. A sum is omitted so earlier payloads still decode and cache
    /// keys for shardings using sums remain unchanged.
    #[serde(default, skip_serializing_if = "PersistentUnreducedKind::is_sum")]
    unreduced_kind: PersistentUnreducedKind,

    /// Mesh axes along which the value has already been reduced.
    reduced_axes: Vec<String>,

    /// Manual mesh axes across which the value can vary.
    varying_manual_axes: Vec<String>,
}

impl From<&Sharding> for PersistentSharding {
    fn from(value: &Sharding) -> Self {
        Self {
            mesh: PersistentLogicalMesh::from(value.mesh()),
            dimensions: value
                .dimensions()
                .iter()
                .map(|dimension| match dimension {
                    ShardingDimension::Replicated => PersistentShardingDimension::Replicated,
                    ShardingDimension::Sharded(axes) => PersistentShardingDimension::Sharded(axes.clone()),
                    ShardingDimension::Unconstrained => PersistentShardingDimension::Unconstrained,
                })
                .collect(),
            unreduced_axes: value.unreduced_axes().iter().cloned().collect(),
            unreduced_kind: value.unreduced_kind().into(),
            reduced_axes: value.reduced_axes().iter().cloned().collect(),
            varying_manual_axes: value.varying_manual_axes().iter().cloned().collect(),
        }
    }
}

impl TryFrom<PersistentSharding> for Sharding {
    type Error = Error;

    fn try_from(value: PersistentSharding) -> Result<Self, Self::Error> {
        Sharding::new(
            LogicalMesh::try_from(value.mesh)?,
            value
                .dimensions
                .into_iter()
                .map(|dimension| match dimension {
                    PersistentShardingDimension::Replicated => ShardingDimension::Replicated,
                    PersistentShardingDimension::Sharded(axes) => ShardingDimension::Sharded(axes),
                    PersistentShardingDimension::Unconstrained => ShardingDimension::Unconstrained,
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

/// Logical mesh and ordered concrete device identities in the transport format.
#[derive(Serialize, Deserialize)]
pub(super) struct PersistentDeviceMesh {
    /// Logical shape and axis kinds of the mesh.
    logical_mesh: PersistentLogicalMesh,

    /// Devices in the concrete mesh order.
    devices: Vec<PersistentDevice>,
}

impl From<&DeviceMesh> for PersistentDeviceMesh {
    fn from(value: &DeviceMesh) -> Self {
        Self {
            logical_mesh: PersistentLogicalMesh::from(value.logical_mesh()),
            devices: value
                .devices()
                .iter()
                .map(|device| PersistentDevice { id: device.id() as u64, process_index: device.process_index() as u64 })
                .collect(),
        }
    }
}

impl TryFrom<PersistentDeviceMesh> for DeviceMesh {
    type Error = Error;

    fn try_from(value: PersistentDeviceMesh) -> Result<Self, Self::Error> {
        DeviceMesh::new(
            LogicalMesh::try_from(value.logical_mesh)?,
            value
                .devices
                .into_iter()
                .map(|device| {
                    Ok(Device::new(
                        usize::try_from(device.id).map_err(|_| {
                            Error::Compilation(CompilationError::InvalidPersistentExecutable {
                                reason: format!("value {} does not fit in `usize`", device.id),
                            })
                        })?,
                        usize::try_from(device.process_index).map_err(|_| {
                            Error::Compilation(CompilationError::InvalidPersistentExecutable {
                                reason: format!("value {} does not fit in `usize`", device.process_index),
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

/// Converts an in-memory executable signature mapping to its stable persistent representation.
pub(super) fn persistent_mapping(mapping: &[Option<usize>]) -> Result<Vec<Option<u64>>, Error> {
    mapping
        .iter()
        .map(|index| {
            index
                .map(|index| {
                    u64::try_from(index).map_err(|_| {
                        Error::Compilation(CompilationError::InvalidPersistentExecutable {
                            reason: format!("physical signature index {index} does not fit in `u64`"),
                        })
                    })
                })
                .transpose()
        })
        .collect()
}

impl CompiledProgram {
    /// Serializes the executable and its versioned invocation contract, or returns `None` when PJRT does not support
    /// executable serialization. Persistent cache directories must be trusted sources of executable code.
    pub(crate) fn serialize(&self) -> Result<Option<Vec<u8>>, Error> {
        let executable = match self.executable().executable() {
            Ok(executable) => executable,
            Err(ryft_pjrt::Error::Unimplemented { .. }) => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let serialized = match executable.serialize() {
            Ok(serialized) => serialized,
            Err(ryft_pjrt::Error::Unimplemented { .. }) => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let device_assignment = self.executable().device_assignment()?;
        let metadata = PersistentExecutableMetadata {
            schema_version: PERSISTENT_EXECUTABLE_SCHEMA_VERSION,
            feature_flags: PERSISTENT_EXECUTABLE_FEATURE_FLAGS,
            compilation_options: self.compilation_options().encode_to_vec(),
            signature: PersistentArraySignature::encode(self.input_types(), self.all_output_types()),
            input_mapping: persistent_mapping(self.signature().input_mapping())?,
            output_mapping: persistent_mapping(self.signature().output_mapping())?,
            input_dimensions: PersistentBoundaryDimension::from_input_dimensions(self.signature()),
            output_dimensions: PersistentBoundaryDimension::from_output_dimensions(self.signature()),
            output_count: self.output_count() as u64,
            reference_states: PersistentReferenceState::encode(self.reference_states(), self.capture_count())?,
            requires_assertion_handler: self.requires_assertion_handler(),
            runtime_requirements: self.runtime_requirements().to_vec(),
            ordered_io: self.signature().has_ordered_io(),
            unordered_io: self.signature().has_unordered_io(),
            device_ordered_io: self.signature().has_device_ordered_io(),
            effect_abi_version: 1,
            donation_flags: self.donation_flags().to_vec(),
            capture_count: self.capture_count() as u64,
            expected_argument_shardings: self
                .expected_argument_shardings()
                .iter()
                .map(PersistentSharding::from)
                .collect(),
            output_shardings: self.output_shardings().iter().map(PersistentSharding::from).collect(),
            mesh: PersistentDeviceMesh::from(self.mesh()),
            device_kinds: self.device_kinds().to_vec(),
            kernel_execution_facts: self.kernel_execution_facts().map(<[u8]>::to_vec),
            replica_count: device_assignment.replica_count() as u64,
            partition_count: device_assignment.computation_count() as u64,
            device_assignment: flatten_device_assignment(&device_assignment)?,
            platform_name: self.platform_name().to_string(),
            platform_version: self.platform_version().to_string(),
            compiler_identity: self.compiler_identity().to_string(),
            xla_flags: self.xla_flags().to_string(),
            compilation_duration_nanoseconds: self
                .compilation_duration()
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

impl Session {
    /// Validates and loads a versioned executable envelope for this session. Unsupported schemas, target
    /// configurations, or PJRT loading return `None`. Malformed supported metadata returns an error.
    ///
    /// The envelope, target, runtime contracts, and signature mapping logical values to physical executable slots are
    /// validated before loading the native PJRT executable bytes. Persistent cache directories must be trusted sources
    /// of executable code.
    pub(crate) fn deserialize_compiled_program(&self, bytes: &[u8]) -> Result<Option<CompiledProgram>, Error> {
        let header_size = PERSISTENT_EXECUTABLE_MAGIC.len() + size_of::<u64>();
        if bytes.len() < header_size {
            return Err(Error::Compilation(CompilationError::InvalidPersistentExecutable {
                reason: "missing persistent executable header".into(),
            }));
        }
        if &bytes[..PERSISTENT_EXECUTABLE_MAGIC.len()] != PERSISTENT_EXECUTABLE_MAGIC {
            return Ok(None);
        }
        let metadata_size =
            u64::from_le_bytes(bytes[PERSISTENT_EXECUTABLE_MAGIC.len()..header_size].try_into().unwrap());
        let metadata_size = usize::try_from(metadata_size).map_err(|_| {
            Error::Compilation(CompilationError::InvalidPersistentExecutable {
                reason: format!("value {metadata_size} does not fit in `usize`"),
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
        let metadata: PersistentExecutableMetadata = serde_json::from_slice(&bytes[header_size..metadata_end])
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
                reason: format!("value {} does not fit in `usize`", metadata.capture_count),
            })
        })?;
        let output_count = usize::try_from(metadata.output_count).map_err(|_| {
            Error::Compilation(CompilationError::InvalidPersistentExecutable {
                reason: format!("value {} does not fit in `usize`", metadata.output_count),
            })
        })?;
        let reference_states = PersistentReferenceState::decode(metadata.reference_states, capture_count)?;
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
                        "bounded-dynamic reference-state input {expected_input_index} logical sharding is not \
                         replicated",
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
            if required && !runtime_requirements.contains(&RuntimeRequirement::builtin(identifier)) {
                return Err(Error::Compilation(CompilationError::InvalidPersistentExecutable {
                    reason: format!("effect metadata requires runtime extension `{identifier}` with ABI version 1"),
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
            || PersistentBoundaryDimension::from_input_dimensions(&signature) != metadata.input_dimensions
            || PersistentBoundaryDimension::from_output_dimensions(&signature) != metadata.output_dimensions
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
        let output_shardings =
            metadata.output_shardings.into_iter().map(Sharding::try_from).collect::<Result<Vec<_>, _>>()?;
        if output_shardings.len() != output_types.len()
            || output_types.iter().zip(&output_shardings).any(|(output_type, sharding)| {
                !sharding.mesh().has_same_topology(mesh.logical_mesh())
                    || output_type
                        .sharding()
                        .is_some_and(|type_sharding| *type_sharding != sharding.without_auto_axes())
            })
        {
            return Err(Error::Compilation(CompilationError::InvalidPersistentExecutable {
                reason: "output shardings do not match the output types or the executable mesh".into(),
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
                        "bounded-dynamic reference-state input {expected_input_index} expected argument sharding \
                         is not replicated",
                    ),
                }));
            }
            if expected_sharding.rank() != input_type.rank()
                || input_type.sharding().is_some_and(|sharding| *sharding != expected_sharding.without_auto_axes())
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
                if output_type.sharding().is_some_and(|sharding| *sharding != expected_sharding.without_auto_axes()) {
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
                reason: format!("value {} does not fit in `usize`", metadata.replica_count),
            })
        })?;
        let partition_count = usize::try_from(metadata.partition_count).map_err(|_| {
            Error::Compilation(CompilationError::InvalidPersistentExecutable {
                reason: format!("value {} does not fit in `usize`", metadata.partition_count),
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
            &output_shardings,
            &signature,
            &donation_flags,
            capture_count,
            &mesh,
        )?;
        Ok(Some(CompiledProgram::from_persisted_parts(
            executable,
            input_types.into(),
            output_types.into(),
            output_count,
            reference_states.into(),
            signature,
            runtime_requirements.into(),
            metadata.requires_assertion_handler,
            donation_flags.into(),
            capture_count,
            expected_argument_shardings.into(),
            output_shardings.into(),
            mesh,
            compilation_options,
            metadata.platform_name.into(),
            metadata.platform_version.into(),
            metadata.device_kinds.into(),
            metadata.kernel_execution_facts,
            metadata.compiler_identity.into(),
            metadata.xla_flags.into(),
            compilation_duration,
            plan,
        )))
    }
}

/// Stable metadata envelope stored before serialized PJRT executable bytes.
#[derive(Serialize, Deserialize)]
struct PersistentExecutableMetadata {
    /// Version of this persistent record schema.
    schema_version: u32,

    /// Required executable transport features for this schema.
    feature_flags: u64,

    /// Protobuf encoding of the complete PJRT compilation options.
    compilation_options: Vec<u8>,

    /// Logical array types with shared dynamic dimension variables.
    signature: PersistentArraySignature,

    /// Physical executable argument position for each logical input, or `None` for erased arrays.
    input_mapping: Vec<Option<u64>>,

    /// Physical executable result position for each logical output, or `None` for erased arrays.
    output_mapping: Vec<Option<u64>>,

    /// Physical extent arguments for bounded dynamic input dimensions.
    input_dimensions: Vec<PersistentBoundaryDimension>,

    /// Physical extent results for bounded dynamic output dimensions.
    output_dimensions: Vec<PersistentBoundaryDimension>,

    /// Number of leading logical outputs in the public result.
    output_count: u64,

    /// External reference binding recipes without live reference handles.
    reference_states: Vec<PersistentReferenceState>,

    /// Whether the executable requires the host runtime assertion handler.
    requires_assertion_handler: bool,

    /// Canonical runtime contracts resolved against the invoking session.
    runtime_requirements: Vec<RuntimeRequirement>,

    /// Whether a native ordered I/O slot is present after the array and extent slots.
    ordered_io: bool,

    /// Whether completion includes unordered I/O without a boundary slot.
    unordered_io: bool,

    /// Whether every ordered I/O occurrence is `DeviceOrderedIo`, permitting per-device execution.
    device_ordered_io: bool,

    /// Version of the hidden effect calling convention.
    effect_abi_version: u32,

    /// Donation flags for public logical inputs, excluding captures.
    donation_flags: Vec<bool>,

    /// Number of leading logical inputs supplied by captures.
    capture_count: u64,

    /// Shardings of all physical executable arguments in argument order.
    expected_argument_shardings: Vec<PersistentSharding>,

    /// Runtime placements of all logical outputs, including the auto mesh axes that output types never record.
    output_shardings: Vec<PersistentSharding>,

    /// Concrete compilation and execution mesh.
    mesh: PersistentDeviceMesh,

    /// PJRT device kinds in the concrete mesh order.
    device_kinds: Vec<String>,

    /// Exact live kernel execution facts, revalidated before loading native executable bytes.
    kernel_execution_facts: Option<Vec<u8>>,

    /// Number of executable replicas in the device assignment.
    replica_count: u64,

    /// Number of computation partitions per executable replica.
    partition_count: u64,

    /// Flattened device IDs ordered by replica first and by partition second.
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
enum PersistentReferenceSource {
    /// Runtime capture position.
    Capture { index: u64 },

    /// Runtime position among the public inputs.
    PublicInput { index: u64 },
}

/// Named dynamic dimension in the executable signature, together with its inclusive lower and exclusive upper bound.
#[derive(Serialize, Deserialize)]
struct PersistentDimensionVariable {
    /// Diagnostic name restored with this dynamic variable.
    name: String,

    /// Inclusive lower bound on the runtime extent.
    lower: u64,

    /// Exclusive upper bound on the runtime extent, or `None` for an unbounded dimension.
    upper: Option<u64>,
}

/// Dynamic dimension bounds in a canonical key, with diagnostic names omitted.
#[derive(Serialize)]
struct PersistentCanonicalDimensionVariable {
    /// Inclusive lower bound on the runtime extent.
    lower: u64,

    /// Exclusive upper bound on the runtime extent, or `None` for an unbounded dimension.
    upper: Option<u64>,
}

/// Stable array type record, using explicit tags rather than Rust enum discriminants.
#[derive(Serialize, Deserialize)]
struct PersistentArrayType {
    /// Stable numeric element type tag defined by [`encode_data_type`].
    data_type: u8,

    /// Logical dimensions in array axis order.
    shape: Vec<PersistentDimension>,

    /// Explicit physical layout, or `None` when unspecified.
    layout: Option<PersistentLayout>,

    /// Logical sharding, or `None` when unspecified.
    sharding: Option<PersistentSharding>,

    /// Memory placement of the array.
    memory: PersistentMemory,
}

/// Static extent or index into the shared dynamic dimension table.
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
enum PersistentDimension {
    /// Concrete dimension extent.
    Static(u64),

    /// Index into the signature's shared dynamic dimension table.
    Dynamic { variable: u64 },
}

/// Stable encoding of tiled and strided physical layouts.
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum PersistentLayout {
    /// Minor-to-major axis order and tile dimensions from outermost to innermost tile.
    Tiled { minor_to_major: Vec<u64>, tiles: Vec<Vec<PersistentTileDimension>> },

    /// Signed byte strides in logical axis order.
    Strided { strides: Vec<i64> },
}

impl From<&Layout> for PersistentLayout {
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
                                TileDimension::Sized(size) => PersistentTileDimension::Sized(*size as u64),
                                TileDimension::Combined => PersistentTileDimension::Combined,
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

impl TryFrom<PersistentLayout> for Layout {
    type Error = Error;

    fn try_from(value: PersistentLayout) -> Result<Self, Self::Error> {
        Ok(match value {
            PersistentLayout::Tiled { minor_to_major, tiles } => TiledLayout::new(
                minor_to_major
                    .into_iter()
                    .map(|value| {
                        usize::try_from(value).map_err(|_| {
                            Error::Compilation(CompilationError::InvalidPersistentExecutable {
                                reason: format!("value {value} does not fit in `usize`"),
                            })
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                tiles
                    .into_iter()
                    .map(|tile| {
                        tile.into_iter()
                            .map(|dimension| match dimension {
                                PersistentTileDimension::Sized(size) => usize::try_from(size)
                                    .map_err(|_| {
                                        Error::Compilation(CompilationError::InvalidPersistentExecutable {
                                            reason: format!("value {size} does not fit in `usize`"),
                                        })
                                    })
                                    .map(TileDimension::Sized),
                                PersistentTileDimension::Combined => Ok(TileDimension::Combined),
                            })
                            .collect::<Result<Vec<_>, _>>()
                            .map(Tile::new)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
            .into(),
            PersistentLayout::Strided { strides } => StridedLayout::new(
                strides
                    .into_iter()
                    .map(|stride| {
                        isize::try_from(stride).map_err(|_| {
                            Error::Compilation(CompilationError::InvalidPersistentExecutable {
                                reason: "layout stride does not fit in `isize`".into(),
                            })
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
            .into(),
        })
    }
}

/// Stable encoding of a tile extent or a combined tile dimension.
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
enum PersistentTileDimension {
    /// Fixed extent of one tile dimension.
    Sized(u64),

    /// Combines this dimension with the next more minor logical dimension.
    Combined,
}

/// Stable encoding of device or host memory placement.
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", content = "pinned", rename_all = "snake_case")]
enum PersistentMemory {
    /// Default device memory.
    Device,

    /// Host memory, with a flag indicating whether it is pinned.
    Host(bool),
}

/// Stable reduction tag for partial values. An omitted tag decodes as a sum so that cache keys for shardings using
/// sums stay compact.
#[derive(Copy, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum PersistentUnreducedKind {
    /// Partial values combined by addition. This is the default for older records.
    #[default]
    Sum,

    /// Partial values combined by a maximum.
    Max,

    /// Partial values combined by a minimum.
    Min,
}

impl PersistentUnreducedKind {
    /// Returns whether the reduction tag can be omitted without changing its meaning.
    fn is_sum(&self) -> bool {
        *self == Self::Sum
    }
}

impl From<UnreducedKind> for PersistentUnreducedKind {
    fn from(value: UnreducedKind) -> Self {
        match value {
            UnreducedKind::Sum => Self::Sum,
            UnreducedKind::Max => Self::Max,
            UnreducedKind::Min => Self::Min,
        }
    }
}

impl From<PersistentUnreducedKind> for UnreducedKind {
    fn from(value: PersistentUnreducedKind) -> Self {
        match value {
            PersistentUnreducedKind::Sum => Self::Sum,
            PersistentUnreducedKind::Max => Self::Max,
            PersistentUnreducedKind::Min => Self::Min,
        }
    }
}

/// Stable sharding assignment for one array dimension.
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", content = "axes", rename_all = "snake_case")]
enum PersistentShardingDimension {
    /// Array dimension partitioned over no mesh axes.
    Replicated,

    /// Array dimension partitioned over the named mesh axes, in order.
    Sharded(Vec<String>),

    /// Array dimension left unconstrained for the partitioner.
    Unconstrained,
}

/// Ordered logical mesh axes in the transport format.
#[derive(Serialize, Deserialize)]
struct PersistentLogicalMesh {
    /// Mesh axes in their declared order.
    axes: Vec<PersistentMeshAxis>,
}

impl From<&LogicalMesh> for PersistentLogicalMesh {
    fn from(value: &LogicalMesh) -> Self {
        Self {
            axes: value
                .axes()
                .iter()
                .map(|axis| PersistentMeshAxis {
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

impl TryFrom<PersistentLogicalMesh> for LogicalMesh {
    type Error = Error;

    fn try_from(value: PersistentLogicalMesh) -> Result<Self, Self::Error> {
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
                                reason: format!("value {} does not fit in `usize`", axis.size),
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

/// Named mesh axis and the stable tag of its partitioning kind.
#[derive(Serialize, Deserialize)]
struct PersistentMeshAxis {
    /// Name used to identify this mesh axis in shardings.
    name: String,

    /// Number of devices along the axis.
    size: u64,

    /// Stable tag of the axis kind, which selects automatic, explicit, or manual partitioning.
    r#type: u8,
}

/// Device identity used to reconstruct a concrete mesh without retaining live PJRT handles.
#[derive(Serialize, Deserialize)]
struct PersistentDevice {
    /// Global PJRT device identifier.
    id: u64,

    /// Index of the process that owns this device.
    process_index: u64,
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

/// Decodes an explicit persistent element type tag and rejects tags unknown to this schema.
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, AnalyzableCompilationDomain, ArrayReference, ArrayType, AssertOperation, CompilationCacheDomain,
        CompilationDomain, CompilationTracer, CompiledFunction, DataType, Dimension, DimensionBounds,
        DimensionVariable, DiskCache, Error as CoreError, ExternalReferenceBinding, Memory, Placeholder,
        PrintOperation, ProgramError, ReferenceAddUpdateOperation, ReferenceReadOperation, ReferenceSource,
        ReferenceType, Shape, Sharding, ShardingDimension, Sin, StagedFunction, StatefulCompilationDomain, Typed,
        ValueProjection,
    };
    use ryft_pjrt::protos::CompilationOptions;

    use tempfile::tempdir;

    use crate::compilation::CompilationError;
    use crate::compilation::options::Options;
    use crate::compilation::tests::external_reference_read_program;
    use crate::sessions::tests::{
        domain_mesh, f32_scalar, f64_vector, program_array, read_first_shard, replicated_scalar_type,
        replicated_vector_type,
    };
    use crate::tests::{cpu_client, values_to_bytes};
    use crate::{
        Array, Constant, Error, FlatProgram, FromPjrt, ProgramBuilder, Session, Type, Value, with_captured_prints,
    };

    use super::*;

    /// Splits a serialized executable envelope into its encoded metadata and its native executable bytes.
    fn envelope_sections(bytes: &[u8]) -> (&[u8], &[u8]) {
        let header_size = PERSISTENT_EXECUTABLE_MAGIC.len() + size_of::<u64>();
        let metadata_size =
            u64::from_le_bytes(bytes[PERSISTENT_EXECUTABLE_MAGIC.len()..header_size].try_into().unwrap());
        let metadata_end = header_size + metadata_size as usize;
        (&bytes[header_size..metadata_end], &bytes[metadata_end..])
    }

    /// Decodes the metadata of a serialized executable envelope.
    fn decode_envelope(bytes: &[u8]) -> (PersistentExecutableMetadata, &[u8]) {
        let (metadata, executable) = envelope_sections(bytes);
        (serde_json::from_slice(metadata).unwrap(), executable)
    }

    /// Encodes a serialized executable envelope from a metadata record and native executable bytes.
    fn encode_envelope(metadata: &PersistentExecutableMetadata, executable: &[u8]) -> Vec<u8> {
        let metadata = serde_json::to_vec(metadata).unwrap();
        let mut bytes = PERSISTENT_EXECUTABLE_MAGIC.to_vec();
        bytes.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&metadata);
        bytes.extend_from_slice(executable);
        bytes
    }

    #[test]
    fn test_persistent_array_type_round_trips_zero_space_metadata() {
        let batch = DimensionVariable::new("batch", DimensionBounds::non_negative(Some(4)).unwrap());
        let array_type =
            ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(2), Dimension::Dynamic(batch.clone())]))
                .with_memory(Memory::Host { pinned: true });
        let output_type =
            ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(2)]));

        let (restored_inputs, restored_outputs) =
            PersistentArraySignature::encode(std::slice::from_ref(&array_type), std::slice::from_ref(&output_type))
                .decode()
                .unwrap();

        assert_eq!(restored_inputs[0].data_type(), array_type.data_type());
        assert_eq!(restored_inputs[0].memory(), array_type.memory());
        assert_eq!(restored_inputs[0].dimension(0), Dimension::Static(2));
        assert_eq!(restored_inputs[0].dimension(1).variable().unwrap().name(), "batch");
        assert_eq!(
            restored_inputs[0].dimension(1).variable().unwrap().bounds(),
            DimensionBounds::non_negative(Some(4)).unwrap(),
        );
        assert_eq!(restored_outputs[0].data_type(), output_type.data_type());
        assert_eq!(restored_inputs[0].dimension(1), restored_outputs[0].dimension(0));
    }

    #[test]
    fn test_persistent_sharding_round_trips_placement_over_auto_axes() {
        // Persisted output shardings are runtime placements, which may name auto mesh axes that types never record, so
        // their encoding preserves both the placement and the axis types of the mesh.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let sharding =
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])]).unwrap();
        let payload = serde_json::to_string(&PersistentSharding::from(&sharding)).unwrap();
        let restored = Sharding::try_from(serde_json::from_str::<PersistentSharding>(&payload).unwrap()).unwrap();
        assert_eq!(restored, sharding);
    }

    #[test]
    fn test_persistent_sharding_unreduced_kinds() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sum = Sharding::replicated(mesh, 1).with_unreduced_axes(["x"]).unwrap();
        let encode = |sharding: &Sharding| serde_json::to_string(&PersistentSharding::from(sharding)).unwrap();
        let decode =
            |payload: &str| Sharding::try_from(serde_json::from_str::<PersistentSharding>(payload).unwrap()).unwrap();

        // A sum is omitted from the payload, so the payloads and cache keys of sums are unchanged, while other kinds
        // round trip through their names.
        assert_eq!(
            encode(&sum),
            r#"{"mesh":{"axes":[{"name":"x","size":2,"type":1}]},"dimensions":[{"kind":"replicated"}],"unreduced_axes":["x"],"reduced_axes":[],"varying_manual_axes":[]}"#,
        );
        assert_eq!(decode(&encode(&sum)), sum);
        for kind in [UnreducedKind::Max, UnreducedKind::Min] {
            let sharding = sum.clone().with_unreduced_kind(kind).unwrap();
            assert_eq!(
                encode(&sharding),
                format!(
                    r#"{{"mesh":{{"axes":[{{"name":"x","size":2,"type":1}}]}},"dimensions":[{{"kind":"replicated"}}],"unreduced_axes":["x"],"unreduced_kind":"{kind}","reduced_axes":[],"varying_manual_axes":[]}}"#,
                ),
            );
            assert_eq!(decode(&encode(&sharding)), sharding);
        }

        // A kind without unreduced axes is malformed.
        let malformed = encode(&Sharding::replicated(sum.mesh().clone(), 1))
            .replace("\"reduced_axes\"", "\"unreduced_kind\":\"max\",\"reduced_axes\"");
        assert!(matches!(
            Sharding::try_from(serde_json::from_str::<PersistentSharding>(&malformed).unwrap()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "unreduced kind `max` requires at least one unreduced mesh axis",
        ));
    }

    #[test]
    fn test_compiled_program_serialize() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let capture = Array::from_host_buffer(
            &session,
            input_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&[10.0, 20.0, 30.0, 40.0]).as_slice(),
        )
        .unwrap();
        fn add_capture(
            _: Vec<Constant>,
            captures: Vec<CompilationTracer<Session>>,
            input: CompilationTracer<Session>,
        ) -> Result<CompilationTracer<Session>, Error> {
            let capture = ValueProjection::<ArrayType>::into_projected(captures.into_iter().next().unwrap())
                .map_err(ProgramError::from)?;
            let input = ValueProjection::<ArrayType>::into_projected(input).map_err(ProgramError::from)?;
            Ok(ValueProjection::<ArrayType>::from_projected(capture + input))
        }
        let staged: StagedFunction<Session, Type, Type> = StagedFunction::<Session, Type, Type>::stage(
            add_capture,
            vec![Value::Array(capture.clone())],
            Type::Array(input_type.clone()),
            Options::new(mesh.clone()).with_donate(true),
        )
        .unwrap();
        let compiled = staged.lower(&session).unwrap().compile(&session).unwrap();
        let compiled_analysis = session.analyze(compiled.executable()).unwrap();
        assert_eq!(session.analyze(compiled.executable()).unwrap(), compiled_analysis);

        let bytes = session.serialize_program(compiled.compiled_program()).unwrap().unwrap();
        let restored = session.deserialize_program(bytes.as_slice()).unwrap().unwrap();
        assert_eq!(restored.input_types(), compiled.compiled_program().input_types());
        assert_eq!(restored.all_output_types(), compiled.compiled_program().all_output_types());
        assert_eq!(restored.signature(), compiled.compiled_program().signature());
        assert_eq!(restored.runtime_requirements(), compiled.compiled_program().runtime_requirements());
        assert_eq!(restored.donation_flags(), compiled.compiled_program().donation_flags());
        assert_eq!(restored.capture_count(), compiled.compiled_program().capture_count());
        assert_eq!(restored.expected_argument_shardings(), compiled.compiled_program().expected_argument_shardings());
        assert_eq!(restored.mesh(), compiled.compiled_program().mesh());

        // Reload validates the live facts before handing any native executable bytes to PJRT.
        let (_, executable) = decode_envelope(&bytes);
        let encode = |metadata: PersistentExecutableMetadata| encode_envelope(&metadata, executable);
        let mut matching = decode_envelope(&bytes).0;
        assert_eq!(matching.kernel_execution_facts, None);
        let facts = crate::kernels::KernelExecutionFacts::from_target(session.target().unwrap(), &mesh).unwrap();
        matching.kernel_execution_facts = Some(facts.configuration_key().unwrap());
        let matching = session.deserialize_program(&encode(matching)).unwrap().unwrap();
        assert_eq!(matching.kernel_execution_facts(), Some(facts.configuration_key().unwrap().as_slice()));
        let mut mismatch = decode_envelope(&bytes).0;
        // The key is a JSON array whose fifth element is the minor version of the PJRT C API, so incrementing that
        // element gives the well-formed key of facts that differ from the live ones.
        let mut changed_key: serde_json::Value = serde_json::from_slice(&facts.configuration_key().unwrap()).unwrap();
        changed_key[4] = serde_json::json!(facts.pjrt_version().minor + 1);
        mismatch.kernel_execution_facts = Some(serde_json::to_vec(&changed_key).unwrap());
        assert!(session.deserialize_program(&encode(mismatch)).unwrap().is_none());
        let mut corrupt = decode_envelope(&bytes).0;
        corrupt.kernel_execution_facts = Some(vec![255]);
        assert!(session.deserialize_program(&encode(corrupt)).unwrap().is_none());
        let baseline = serde_json::to_value(decode_envelope(&bytes).0).unwrap();
        assert_eq!(baseline["runtime_requirements"], serde_json::json!([]));
        let mut missing: PersistentExecutableMetadata = serde_json::from_value(baseline).unwrap();
        missing.runtime_requirements = vec![RuntimeRequirement::builtin("test.missing-runtime")];
        assert!(matches!(
            session.deserialize_program(&encode(missing)),
            Err(Error::Runtime(crate::sessions::runtimes::RuntimeError::MissingProvider { identifier }))
                if identifier == "test.missing-runtime"
        ));
        let mut explicit = decode_envelope(&bytes).0;
        explicit.runtime_requirements =
            vec![RuntimeRequirement::builtin("ryft.assert"), RuntimeRequirement::builtin("ryft.ordered-io")];
        let explicit = session.deserialize_program(&encode(explicit)).unwrap().unwrap();
        assert_eq!(
            explicit.runtime_requirements(),
            &[RuntimeRequirement::builtin("ryft.assert"), RuntimeRequirement::builtin("ryft.ordered-io")],
        );
        assert!(!explicit.requires_assertion_handler());
        assert!(!explicit.signature().has_ordered_io());

        let mut invalid_signature_metadata = decode_envelope(&bytes).0;
        invalid_signature_metadata.input_mapping[0] = None;
        assert!(matches!(
            session.deserialize_program(&encode(invalid_signature_metadata)),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "executable signature does not match logical input and output types",
        ));

        let mut incompatible_metadata = decode_envelope(&bytes).0;
        incompatible_metadata.platform_version.push_str("-incompatible");
        assert!(session.deserialize_program(&encode(incompatible_metadata)).unwrap().is_none());

        let input = Array::from_host_buffer(
            &session,
            input_type,
            mesh.clone(),
            None,
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        // A restored program owns the complete declaration contract, including the hidden capture prefix.
        assert!(matches!(
            session.execute(&restored, vec![Value::Array(capture.clone())]),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
            ),
        ));
        let wrong_type = f64_vector(&session, &mesh, &[1.0, 2.0, 3.0, 4.0]);
        for index in 0..2 {
            let expected = format!(
                "runtime input type `{}` does not refine declared type `{}`",
                wrong_type.r#type(),
                restored.input_types()[index],
            );
            let mut inputs = vec![Value::Array(capture.clone()), Value::Array(input.clone())];
            inputs[index] = Value::Array(wrong_type.clone());
            assert!(matches!(
                session.execute(&restored, inputs),
                Err(Error::Core(core_error)) if matches!(
                    core_error.as_ref(),
                    CoreError::Program(ProgramError::InvalidArgument { message })
                        if *message == expected,
                ),
            ));
        }
        let outputs = session.execute(&restored, vec![Value::Array(capture), Value::Array(input)]).unwrap();
        assert_eq!(read_first_shard::<f32>(client, program_array(&outputs[0])), vec![11.0, 22.0, 33.0, 44.0]);

        let analysis = restored.analyze().unwrap();
        assert_eq!(restored.analyze().unwrap(), analysis);
        assert!(analysis.to_json().unwrap().contains("\"properties\""));
        assert!(!analysis.to_string().is_empty());
    }

    #[test]
    fn test_compiled_program_serialize_round_trips_external_reference_state() {
        let client = cpu_client(1);
        // The corruptions below add unreduced placement to persisted shardings, which types and persisted reference
        // state checks record over explicit mesh axes only.
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap()],
        )
        .unwrap();
        let session = Session::builder().client(client).build().unwrap();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);

        let mut builder = ProgramBuilder::new();
        let reference = builder.add_input(Type::Reference(ReferenceType::new(scalar_type.clone())));
        let read_only = builder.add_input(Type::Reference(ReferenceType::new(scalar_type)));
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
        let lowered = session.lower_program(&program, 1, &Options::new(mesh.clone())).unwrap();
        let compiled = session.compile_program(&lowered).unwrap();
        let bytes = compiled.serialize().unwrap().unwrap();
        let restored = session.deserialize_compiled_program(bytes.as_slice()).unwrap().unwrap();
        assert_eq!(compiled.output_count(), 1);
        assert_eq!(compiled.reference_states().len(), 2);
        assert_eq!(compiled.reference_states()[0].source(), ReferenceSource::Capture { index: 0 });
        assert_eq!(compiled.reference_states()[0].source().flat_input_index(compiled.capture_count()).unwrap(), 0);
        assert_eq!(compiled.reference_states()[0].output_index(), Some(1));
        assert_eq!(compiled.reference_states()[1].source(), ReferenceSource::Input { index: 0 });
        assert_eq!(compiled.reference_states()[1].source().flat_input_index(compiled.capture_count()).unwrap(), 1);
        assert_eq!(compiled.reference_states()[1].output_index(), None);
        assert_eq!(restored.output_count(), compiled.output_count());
        assert_eq!(restored.reference_states(), compiled.reference_states());
        let outputs = session
            .execute_program(&restored, vec![f32_scalar(&session, &mesh, 1.0), f32_scalar(&session, &mesh, 2.0)])
            .unwrap();
        assert_eq!(read_first_shard::<f32>(client, &outputs[0]), vec![5.0]);
        assert_eq!(read_first_shard::<f32>(client, &outputs[1]), vec![3.0]);

        let (_, executable) = decode_envelope(&bytes);
        let corrupt = |metadata: PersistentExecutableMetadata| encode_envelope(&metadata, executable);
        let mut metadata = decode_envelope(&bytes).0;
        assert_eq!(metadata.reference_states[0].source, PersistentReferenceSource::Capture { index: 0 });
        assert_eq!(metadata.reference_states[0].logical_input_index, 0);
        assert_eq!(metadata.reference_states[1].source, PersistentReferenceSource::PublicInput { index: 0 });
        assert_eq!(metadata.reference_states[1].logical_input_index, 1);
        metadata.reference_states[0].logical_input_index = 1;
        let corrupted = corrupt(metadata);
        assert!(matches!(
            session.deserialize_compiled_program(corrupted.as_slice()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "reference-state source does not match its logical input",
        ));

        let mut metadata = decode_envelope(&bytes).0;
        metadata.reference_states[0].source = PersistentReferenceSource::PublicInput { index: 1 };
        let corrupted = corrupt(metadata);
        assert!(matches!(
            session.deserialize_compiled_program(corrupted.as_slice()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "reference-state source does not match its logical input",
        ));

        // An output count beyond the decoded complete output arity cannot name a hidden suffix of final states.
        let mut invalid_outputs = decode_envelope(&bytes).0;
        invalid_outputs.output_count = 3;
        let corrupted = corrupt(invalid_outputs);
        assert!(matches!(
            session.deserialize_compiled_program(corrupted.as_slice()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "reference-state metadata has an invalid output count",
        ));

        // Dropping one expected argument sharding leaves the persisted list shorter than the physical boundary, so
        // no reference state input could be checked against its own expected sharding.
        let mut truncated_expected_shardings = decode_envelope(&bytes).0;
        assert_eq!(truncated_expected_shardings.expected_argument_shardings.len(), 2);
        truncated_expected_shardings.expected_argument_shardings.pop();
        let corrupted = corrupt(truncated_expected_shardings);
        assert!(matches!(
            session.deserialize_compiled_program(corrupted.as_slice()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "expected argument shardings do not match the physical input count",
        ));

        let mut reordered = decode_envelope(&bytes).0;
        reordered.reference_states.swap(0, 1);
        let corrupted = corrupt(reordered);
        assert!(matches!(
            session.deserialize_compiled_program(corrupted.as_slice()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "reference states are not in canonical logical input order",
        ));

        let mut foreign_process_mesh = decode_envelope(&bytes).0;
        foreign_process_mesh.mesh.devices[0].process_index += 1;
        let corrupted = corrupt(foreign_process_mesh);
        assert!(matches!(
            session.deserialize_compiled_program(corrupted.as_slice()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "reference-state mesh does not include an addressable device of this process",
        ));

        let mut mismatched_expected_sharding = decode_envelope(&bytes).0;
        mismatched_expected_sharding.expected_argument_shardings[0].unreduced_axes.push("x".to_string());
        let corrupted = corrupt(mismatched_expected_sharding);
        assert!(matches!(
            session.deserialize_compiled_program(corrupted.as_slice()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "reference-state input 0 logical sharding does not match its expected argument sharding",
        ));

        let mut mismatched_input = decode_envelope(&bytes).0;
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
        mismatched_input.output_shardings[1].unreduced_axes.push("x".to_string());
        let corrupted = corrupt(mismatched_input);
        assert!(matches!(
            session.deserialize_compiled_program(corrupted.as_slice()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "reference-state input 0 logical sharding does not match its expected argument sharding",
        ));

        let mut mismatched_output = decode_envelope(&bytes).0;
        mismatched_output.signature.output_types[1]
            .sharding
            .as_mut()
            .unwrap()
            .unreduced_axes
            .push("x".to_string());
        mismatched_output.output_shardings[1].unreduced_axes.push("x".to_string());
        let corrupted = corrupt(mismatched_output);
        assert!(matches!(
            session.deserialize_compiled_program(corrupted.as_slice()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "reference-state output 1 logical sharding does not match state input 0",
        ));

        // The runtime placement of every output must agree with its type over the mesh axes that are not auto.
        let mut mismatched_output_sharding = decode_envelope(&bytes).0;
        mismatched_output_sharding.output_shardings[1].unreduced_axes.push("x".to_string());
        let corrupted = corrupt(mismatched_output_sharding);
        assert!(matches!(
            session.deserialize_compiled_program(corrupted.as_slice()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "output shardings do not match the output types or the executable mesh",
        ));

        let mut zero_space = decode_envelope(&bytes).0;
        zero_space.signature.input_types[0].data_type = encode_data_type(DataType::Zero);
        let corrupted = corrupt(zero_space);
        assert!(matches!(
            session.deserialize_compiled_program(corrupted.as_slice()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "reference-state input has a zero-space type",
        ));

        let mut host_memory = decode_envelope(&bytes).0;
        host_memory.signature.input_types[0].memory = PersistentMemory::Host(true);
        let corrupted = corrupt(host_memory);
        assert!(matches!(
            session.deserialize_compiled_program(corrupted.as_slice()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "reference-state input is not in device memory",
        ));

        let mut unbounded_dynamic = decode_envelope(&bytes).0;
        unbounded_dynamic.signature.variables.push(PersistentDimensionVariable {
            name: "unbounded".to_string(),
            lower: 0,
            upper: None,
        });
        unbounded_dynamic.signature.input_types[1].shape = vec![PersistentDimension::Dynamic { variable: 0 }];
        unbounded_dynamic.signature.input_types[1].sharding.as_mut().unwrap().dimensions =
            vec![PersistentShardingDimension::Replicated];
        let corrupted = corrupt(unbounded_dynamic);
        assert!(matches!(
            session.deserialize_compiled_program(corrupted.as_slice()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "reference-state input has an unbounded dynamic dimension",
        ));

        let mut mutated_dynamic = decode_envelope(&bytes).0;
        mutated_dynamic.signature.variables.push(PersistentDimensionVariable {
            name: "bounded_mutation".to_string(),
            lower: 0,
            upper: Some(5),
        });
        mutated_dynamic.signature.input_types[0].shape = vec![PersistentDimension::Dynamic { variable: 0 }];
        mutated_dynamic.signature.input_types[0].sharding.as_mut().unwrap().dimensions =
            vec![PersistentShardingDimension::Replicated];
        let corrupted = corrupt(mutated_dynamic);
        assert!(matches!(
            session.deserialize_compiled_program(corrupted.as_slice()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "mutated bounded-dynamic reference-state input 0 is unsupported because backend alias \
                              compatibility has not been verified",
        ));

        let mut sharded_dynamic = decode_envelope(&bytes).0;
        sharded_dynamic.signature.variables.push(PersistentDimensionVariable {
            name: "bounded_sharded".to_string(),
            lower: 0,
            upper: Some(5),
        });
        sharded_dynamic.signature.input_types[1].shape = vec![PersistentDimension::Dynamic { variable: 0 }];
        sharded_dynamic.signature.input_types[1].sharding.as_mut().unwrap().dimensions =
            vec![PersistentShardingDimension::Sharded(vec!["x".to_string()])];
        let corrupted = corrupt(sharded_dynamic);
        assert!(matches!(
            session.deserialize_compiled_program(corrupted.as_slice()),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "bounded-dynamic reference-state input 1 logical sharding is not replicated",
        ));
    }

    #[test]
    fn test_compiled_program_serialize_round_trips_bounded_dynamic_read_only_reference_state() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let extent = DimensionVariable::new("state_extent", DimensionBounds::new(1, Some(5)).unwrap());
        let state_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let lowered = session
            .lower_program(&external_reference_read_program(state_type.clone()), 0, &Options::new(mesh.clone()))
            .unwrap();
        let compiled = session.compile_program(&lowered).unwrap();
        let bytes = compiled.serialize().unwrap().unwrap();
        let restored = session.deserialize_compiled_program(bytes.as_slice()).unwrap().unwrap();

        assert_eq!(restored.input_types()[0].data_type(), state_type.data_type());
        assert_eq!(restored.input_types()[0].shape().to_string(), state_type.shape().to_string());
        assert_eq!(restored.input_types()[0].sharding(), state_type.sharding());
        assert_eq!(restored.input_types()[0].memory(), state_type.memory());
        assert_eq!(restored.output_count(), 1);
        assert_eq!(
            restored.reference_states(),
            &[ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, None)],
        );

        // Runtime holders have exact extents. The restored declarations retain the bounds against which they refine.
        let array = Array::from_host_buffer(
            &session,
            replicated_vector_type(&mesh, 3),
            mesh,
            None,
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0]).as_slice(),
        )
        .unwrap();
        let holder = ArrayReference::new(array);
        let reference = Value::Reference(crate::Reference::new(holder, session.clone()));
        let outputs = session.execute_statefully(&restored, vec![reference]).r#await().unwrap();
        assert_eq!(read_first_shard::<f32>(client, program_array(&outputs[0])), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_compiled_program_serialize_preserves_static_sharded_reference_metadata() {
        let client = cpu_client(2);
        // Types record placement over explicit mesh axes only, so the statically sharded state uses an explicit axis.
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap(),
            client
                .addressable_devices()
                .unwrap()
                .into_iter()
                .map(|device| Device::from_pjrt(device).unwrap())
                .collect(),
        )
        .unwrap();
        let session = Session::builder().client(client).build().unwrap();
        let state_sharding =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let state_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(state_sharding.clone())
            .unwrap();
        let mut builder = ProgramBuilder::new();
        let reference = builder.add_input(Type::Reference(ReferenceType::new(state_type.clone())));
        let update = builder.add_input(Type::Array(state_type.clone()));
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let output =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program: FlatProgram = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = session.compile_program(&lowered).unwrap();
        assert_eq!(compiled.input_types(), &[state_type.clone(), state_type.clone()]);
        assert_eq!(compiled.all_output_types(), &[state_type.clone(), state_type.clone()]);
        assert_eq!(compiled.expected_argument_shardings(), &[state_sharding.clone(), state_sharding]);
        assert_eq!(compiled.mesh(), &mesh);
        assert_eq!(
            compiled.reference_states(),
            &[ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, Some(1))],
        );
        match compiled.serialize().unwrap() {
            Some(bytes) => {
                let restored = session.deserialize_compiled_program(bytes.as_slice()).unwrap().unwrap();
                assert_eq!(restored.input_types(), compiled.input_types());
                assert_eq!(restored.all_output_types(), compiled.all_output_types());
                assert_eq!(restored.reference_states(), compiled.reference_states());
                assert_eq!(restored.expected_argument_shardings(), compiled.expected_argument_shardings());
                assert_eq!(restored.mesh(), compiled.mesh());
            }
            None => {
                // The CPU plugin does not implement executable serialization for this two-device program, so `None`
                // is the documented answer for an optional capability rather than a silently skipped round trip.
                // Either the executable query or its serialization may be the unimplemented step.
                let unsupported = match compiled.executable().executable() {
                    Ok(executable) => executable.serialize().err(),
                    Err(error) => Some(error),
                };
                assert!(matches!(unsupported, Some(ryft_pjrt::Error::Unimplemented { .. })));
            }
        }
    }

    #[test]
    fn test_session_deserialize_compiled_program_requires_runtime_contracts_for_effect_metadata() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let mut builder = ProgramBuilder::new();
        let predicate = builder.add_input(replicated_scalar_type(&mesh, DataType::Boolean).into());
        let value = builder.add_input(replicated_scalar_type(&mesh, DataType::F32).into());
        builder
            .add_instruction(AssertOperation::<Type>::new("test predicate"), Vec::new(), vec![predicate], None)
            .unwrap();
        builder
            .add_instruction(PrintOperation::<ArrayType>::new("test value"), Vec::new(), vec![value], None)
            .unwrap();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let lowered = session.lower_program(&program, 0, &Options::new(mesh)).unwrap();
        let compiled = session.compile_program(&lowered).unwrap();
        let bytes = session.serialize_program(&compiled).unwrap().unwrap();
        let (metadata, executable) = envelope_sections(&bytes);
        let baseline: PersistentExecutableMetadata = serde_json::from_slice(metadata).unwrap();
        assert!(baseline.requires_assertion_handler);
        assert!(baseline.ordered_io);
        let restored = session.deserialize_program(&bytes).unwrap().unwrap();
        assert_eq!(restored.runtime_requirements(), baseline.runtime_requirements);

        // Only the identifier and ABI version of each requirement are written into the envelope.
        let metadata: serde_json::Value = serde_json::from_slice(metadata).unwrap();
        assert_eq!(
            metadata["runtime_requirements"],
            serde_json::json!([
                {"identifier": "ryft.assert", "abi_version": 1},
                {"identifier": "ryft.ordered-io", "abi_version": 1},
                {"identifier": "ryft.print", "abi_version": 1},
            ]),
        );

        // The effect ABI requires a compatible runtime contract for each declared handler.
        for identifier in ["ryft.assert", "ryft.ordered-io"] {
            for replacement in [None, Some(2)] {
                let mut metadata = decode_envelope(&bytes).0;
                metadata.runtime_requirements.retain(|requirement| requirement.identifier() != identifier);
                if let Some(version) = replacement {
                    metadata.runtime_requirements.push(RuntimeRequirement::new(identifier, version).unwrap());
                }
                assert!(matches!(
                    session.deserialize_program(&encode_envelope(&metadata, executable)),
                    Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                        if reason == format!(
                            "effect metadata requires runtime extension `{identifier}` with ABI version 1",
                        ),
                ));
            }
        }
    }

    #[test]
    fn test_session_deserialize_compiled_program_validates_ordered_io_signature() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let input = f64_vector(&session, &mesh, &[3.0]);
        let mut builder = ProgramBuilder::new();
        let argument = builder.add_input(input.r#type().into_owned().into());
        builder
            .add_instruction(PrintOperation::<ArrayType>::new("ordered"), Vec::new(), vec![argument], None)
            .unwrap();
        let program = builder.build::<Vec<Constant>, Vec<Constant>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let lowered = session.lower_program(&program, 0, &Options::new(mesh)).unwrap();
        let compiled = session.compile_program(&lowered).unwrap();
        assert!(compiled.signature().has_ordered_io());
        let bytes = compiled.serialize().unwrap().unwrap();

        let restored = session.deserialize_compiled_program(&bytes).unwrap().unwrap();
        assert_eq!(restored.signature(), compiled.signature());
        let (outputs, prints) = with_captured_prints(|| {
            let outputs = session.execute_program(&restored, vec![input]).unwrap();
            session.synchronize().unwrap();
            outputs
        });
        assert!(outputs.is_empty());
        assert_eq!(prints, vec!["ordered: [3.0]"]);

        let (mut metadata, executable) = decode_envelope(&bytes);
        metadata.ordered_io = false;
        let corrupted = encode_envelope(&metadata, executable);
        assert!(matches!(
            session.deserialize_compiled_program(&corrupted),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "effect signature does not match executable physical outputs",
        ));
    }

    #[test]
    fn test_session_deserialize_compiled_program_restores_into_a_fresh_compilation_cache() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let directory = tempdir().unwrap();

        let disk_cache = || DiskCache::open(directory.path()).unwrap().with_write_thresholds(Duration::ZERO, 0);
        let first_session = Session::builder().client(client).disk_cache(disk_cache()).build().unwrap();
        let staged = crate::stage::<_, ArrayType, ArrayType>(
            |x| x.sin().unwrap(),
            input_type.clone(),
            Options::new(mesh.clone()),
        )
        .unwrap()
        .into_inner();
        let first: CompiledFunction<Session, Type, Type> =
            staged.lower(&first_session).unwrap().compile(&first_session).unwrap();
        assert_eq!(first_session.compilation_cache().statistics().compilations, 1);
        drop(first);
        drop(first_session);

        let second_session = Session::builder().client(client).disk_cache(disk_cache()).build().unwrap();
        let staged = crate::stage::<_, ArrayType, ArrayType>(|x| x.sin().unwrap(), input_type, Options::new(mesh))
            .unwrap()
            .into_inner();
        let _restored: CompiledFunction<Session, Type, Type> =
            staged.lower(&second_session).unwrap().compile(&second_session).unwrap();
        let statistics = second_session.compilation_cache().statistics();
        assert_eq!(statistics.persistent_hits, 1);
        assert_eq!(statistics.compilations, 0, "restoration must not invoke backend compilation");
    }

    #[test]
    fn test_session_deserialize_compiled_program_rejects_malformed_and_incompatible_metadata() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();

        assert!(matches!(
            session.deserialize_program(b"truncated"),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "missing persistent executable header",
        ));
        let mut legacy = b"RYFTXLA8".to_vec();
        legacy.extend_from_slice(&0u64.to_le_bytes());
        assert!(session.deserialize_program(legacy.as_slice()).unwrap().is_none());

        let metadata = PersistentExecutableMetadata {
            schema_version: PERSISTENT_EXECUTABLE_SCHEMA_VERSION + 1,
            feature_flags: 0,
            compilation_options: CompilationOptions::default().encode_to_vec(),
            signature: PersistentArraySignature::encode(&[], &[]),
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
            output_shardings: Vec::new(),
            mesh: PersistentDeviceMesh {
                logical_mesh: PersistentLogicalMesh { axes: Vec::new() },
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
        assert!(session.deserialize_program(&encode_envelope(&metadata, &[])).unwrap().is_none());
    }

    #[test]
    fn test_persistent_data_type_codes() {
        // The persistent executable encoding assigns each data type a stable one-byte code, and so this test pins
        // every assignment. New data types must be appended with fresh codes and existing codes must never be
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
        assert!(matches!(
            decode_data_type(34),
            Err(Error::Compilation(CompilationError::InvalidPersistentExecutable { reason }))
                if reason == "unknown data type tag 34",
        ));
    }
}
