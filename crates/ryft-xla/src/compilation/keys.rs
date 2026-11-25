//! Dispatch and cache keys of the XLA [`Session`](crate::Session).
//!
//! Three keys identify work at different levels. A [`DispatchKey`] selects a retained specialization of a dispatcher
//! from the types and placements of its inputs. A [`CompilationKey`] is the complete canonical compilation contract
//! of a [`LoweredProgram`], so equal keys mean that compiled artifacts are interchangeable, both in the compilation
//! cache of a session and in a persistent cache. An `EagerDispatchKey` maps one eager operation application directly
//! to its compiled program.

use std::borrow::Cow;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, LazyLock};

use prost::Message;
use ryft_core::{
    ArrayType, DeviceMesh, Dimension, DimensionBounds, DimensionValue, DimensionVariable, DispatchInputs, ProgramError,
    RegionKey, Shape, Sharding, Typed,
};
use ryft_pjrt::protos::{AutoTuneReferenceKey, AutoTuneResultKey, CompilationOptions};
use serde::Serialize;

use crate::compilation::CompilationError;
use crate::compilation::domains::{apply_input_shardings, argument_type};
use crate::compilation::options::{InputBoundBucketing, Options};
use crate::compilation::persistence::{
    PersistentArraySignature, PersistentBoundaryDimension, PersistentCanonicalArraySignature, PersistentDeviceMesh,
    PersistentReferenceState, PersistentSharding, persistent_mapping,
};
use crate::compilation::programs::LoweredProgram;
use crate::{Constant, Error, Operation, RuntimeRequirement, Type, Value};

// TODO(eaplatanios): Review this module.

/// Schema used by canonical compilation keys.
const PERSISTENT_KEY_SCHEMA_VERSION: u32 = 1;

/// Build identity shared by lowering, cache keys, and executable compatibility validation.
pub(super) static COMPILER_IDENTITY: LazyLock<String> = LazyLock::new(|| {
    format!(
        "ryft-xla/{}/openxla/{}/jax/{}",
        env!("CARGO_PKG_VERSION"),
        ryft_xla_sys::XLA_COMMIT,
        ryft_xla_sys::JAX_COMMIT,
    )
});

/// Opaque key of a retained `jit` specialization, derived by [`Session`](crate::Session).
///
/// Exact dispatch preserves nominal [`DimensionVariable`] identity and shares its type slice with the effective staged
/// signature, so keying a call costs no deep type clone. Dispatch with [`InputBoundBucketing`] instead uses a
/// structural signature over physical bounds with dimension variables numbered by first occurrence, so calls routed to
/// the same bucket share one staged and compiled specialization. Bucketed keys exclude diagnostic names, which survive
/// only in the effective staged types.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DispatchKey {
    /// Signature component of this key.
    kind: DispatchKeyKind,

    /// Runtime placements of the public inputs of an executable specialization whose inputs are placed over auto mesh
    /// axes, which their types never record and which the input placements of its [`Options`] carry, or [`None`] when
    /// no input is.
    placements: Option<Arc<[Option<Sharding>]>>,
}

impl DispatchKey {
    /// Creates a [`DispatchKey`] without runtime placements.
    fn new(kind: DispatchKeyKind) -> Self {
        Self { kind, placements: None }
    }
}

/// Exact process-local compilation key for an XLA lowering.
///
/// This key carries the complete StableHLO computation and the executable contract retained by [`LoweredProgram`].
/// Equal keys share a native executable. Each invocation retains its own device assignment and dimension identities.
///
/// Dynamic dimensions are numbered by first occurrence, so diagnostic variable names do not affect identity.
/// Compilation options, shardings, the outputs that executions commit, target facts, runtime requirements, and effect
/// metadata remain part of the key.
/// This complete compilation identity is distinct from a [`DispatchKey`], which selects a retained specialization.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CompilationKey {
    /// Complete canonical compilation contract, compared and hashed without reducing it to a fingerprint.
    canonical_bytes: Arc<[u8]>,
}

impl CompilationKey {
    /// Returns the complete canonical compilation contract, compared and hashed without reducing it to a fingerprint.
    #[inline]
    pub(super) fn canonical_bytes(&self) -> &[u8] {
        &self.canonical_bytes
    }
}

impl LoweredProgram {
    /// Encodes the complete compilation contract as a deterministic cache key.
    pub(super) fn compilation_key(&self) -> Result<CompilationKey, Error> {
        let compilation_options = canonical_compilation_options_bytes(self.compilation_options());
        let key = PersistentKey {
            schema_version: PERSISTENT_KEY_SCHEMA_VERSION,
            stable_hlo: self.stable_hlo(),
            compilation_options: compilation_options.as_slice(),
            signature: PersistentArraySignature::encode(self.input_types(), self.all_output_types()).into_canonical(),
            input_mapping: persistent_mapping(self.signature().input_mapping())?,
            unread_inputs: self.signature().unread_inputs(),
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
            donation_flags: self.donation_flags(),
            capture_count: self.capture_count() as u64,
            expected_argument_shardings: self
                .expected_argument_shardings()
                .iter()
                .map(PersistentSharding::from)
                .collect(),
            output_shardings: self.output_shardings().iter().map(PersistentSharding::from).collect(),
            committed_outputs: self.committed_outputs(),
            mesh: PersistentDeviceMesh::from(self.mesh()),
            device_kinds: self.device_kinds(),
            kernel_execution_facts: self.kernel_execution_facts(),
            platform_name: self.platform_name(),
            platform_version: self.platform_version(),
            compiler_identity: self.compiler_identity(),
            xla_flags: self.xla_flags(),
        };
        let canonical_bytes = serde_json::to_vec(&key).map_err(|error| {
            Error::Compilation(CompilationError::InvalidPersistentExecutable {
                reason: format!("failed to encode persistent key: {error}"),
            })
        })?;
        Ok(CompilationKey { canonical_bytes: canonical_bytes.into() })
    }
}

/// Key of the eager dispatch cache of a [`Session`](crate::Session), which maps one eager operation application
/// directly to its compiled program so that a repeated application skips tracing, lowering, and the construction of
/// compilation keys, like the per-primitive `(primitive, params)` cache that JAX keeps in front of its dispatch path.
///
/// The key must capture everything that the eager lowering reads besides the session itself, whose client, and thus
/// [`Target`](crate::Target), is fixed. That is the operation including all of its attributes, the contents of every
/// attached region (e.g., of `condition`, `while`, `scan`, `jit_call`, or `shard_map` operations), the physical type of
/// every array input including its sharding, the extent of every dimension input, because eager binding bakes dimension
/// inputs into the traced program as constants, and the device mesh that the lowering compiles against. The default
/// compilation options of the session are immutable, so they are not part of the key. Attached regions are keyed by
/// their contents, as [`RegionKey`] describes, so applications whose closures are traced afresh on every call still hit
/// the cache when they trace to the same regions. Instruction provenance is not part of a [`RegionKey`], so
/// applications that differ only in provenance share one compiled program, whose debug locations are those of the first
/// application.
///
/// Keys are compared and hashed through [`EagerDispatchKeyView`], which they share with the borrowed
/// [`EagerDispatchQuery`] that eager dispatch uses to look up hits. Cache hits skip tracing and lowering while
/// execution still uses the invoking session's effect scope and runtime resources.
#[derive(Clone)]
pub(crate) struct EagerDispatchKey {
    /// Applied operation, compared and hashed with all of its attributes.
    operation: Operation,

    /// Keys of the regions attached to the application, in the order that the operation defines.
    regions: Arc<[RegionKey<Constant, Operation>]>,

    /// Inputs of the application, in order.
    inputs: Arc<[EagerDispatchInput]>,

    /// Mesh that eager lowering compiles against when it is not derived from the operation alone. It is the mesh of
    /// the first array input if there is one, and otherwise the mesh of the session, if it has one. It is [`None`]
    /// only for an operation without array inputs in a session without a mesh, whose mesh is then determined by the
    /// operation and the session's client.
    mesh: Option<DeviceMesh>,
}

impl EagerDispatchKey {
    /// Creates the owned key of one eager operation application.
    pub(crate) fn new(
        operation: Operation,
        regions: Arc<[RegionKey<Constant, Operation>]>,
        inputs: Arc<[EagerDispatchInput]>,
        mesh: Option<DeviceMesh>,
    ) -> Self {
        Self { operation, regions, inputs, mesh }
    }
}

/// One input of an [`EagerDispatchKey`].
#[derive(Clone)]
pub(crate) enum EagerDispatchInput {
    /// Array input, keyed by its physical [`ArrayType`], including its sharding, and by its runtime placement, which
    /// also covers the auto mesh axes that its type never records.
    Array(ArrayType, Sharding),

    /// Dimension input, keyed by its extent, which is how [`DimensionValue`]s compare.
    Dimension(DimensionValue),
}

/// Borrowed form of one [`EagerDispatchInput`], which owned keys and borrowed queries compare and hash alike.
#[derive(PartialEq, Eq, Hash)]
pub(crate) enum EagerDispatchInputRef<'o> {
    /// Array input, keyed by its physical [`ArrayType`], including its sharding, and by its runtime placement.
    Array(Cow<'o, ArrayType>, &'o Sharding),

    /// Dimension input, keyed by its extent, which is how [`DimensionValue`]s compare.
    Dimension(&'o DimensionValue),
}

impl EagerDispatchInputRef<'_> {
    /// Returns the owned [`EagerDispatchInput`] that this reference borrows from or describes.
    pub(crate) fn into_owned(self) -> EagerDispatchInput {
        match self {
            Self::Array(r#type, sharding) => EagerDispatchInput::Array(r#type.into_owned(), sharding.clone()),
            Self::Dimension(dimension) => EagerDispatchInput::Dimension(dimension.clone()),
        }
    }
}

/// Common view of an owned [`EagerDispatchKey`] and a borrowed [`EagerDispatchQuery`]. Both forms are compared
/// and hashed through `dyn EagerDispatchKeyView`, so that equal keys and queries hash alike and an owned key can be
/// [borrowed](std::borrow::Borrow) as a view for lookups in the eager dispatch cache.
pub(crate) trait EagerDispatchKeyView {
    /// Returns the applied operation.
    fn operation(&self) -> &Operation;

    /// Returns the keys of the regions attached to the application, in the order that the operation defines.
    fn regions(&self) -> &[RegionKey<Constant, Operation>];

    /// Returns the number of inputs of the application.
    fn input_count(&self) -> usize;

    /// Returns the input at `index`, which must be smaller than [`Self::input_count`].
    fn input(&self, index: usize) -> EagerDispatchInputRef<'_>;

    /// Returns the mesh that the eager lowering compiles against, if it is not derived from the operation alone.
    fn mesh(&self) -> Option<&DeviceMesh>;
}

impl EagerDispatchKeyView for EagerDispatchKey {
    fn operation(&self) -> &Operation {
        &self.operation
    }

    fn regions(&self) -> &[RegionKey<Constant, Operation>] {
        &self.regions
    }

    fn input_count(&self) -> usize {
        self.inputs.len()
    }

    fn input(&self, index: usize) -> EagerDispatchInputRef<'_> {
        match &self.inputs[index] {
            EagerDispatchInput::Array(r#type, sharding) => {
                EagerDispatchInputRef::Array(Cow::Borrowed(r#type), sharding)
            }
            EagerDispatchInput::Dimension(dimension) => EagerDispatchInputRef::Dimension(dimension),
        }
    }

    fn mesh(&self) -> Option<&DeviceMesh> {
        self.mesh.as_ref()
    }
}

impl PartialEq for dyn EagerDispatchKeyView + '_ {
    fn eq(&self, other: &Self) -> bool {
        self.operation() == other.operation()
            && self.regions() == other.regions()
            && self.input_count() == other.input_count()
            && (0..self.input_count()).all(|index| self.input(index) == other.input(index))
            && self.mesh() == other.mesh()
    }
}

impl Eq for dyn EagerDispatchKeyView + '_ {}

impl Hash for dyn EagerDispatchKeyView + '_ {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.operation().hash(state);
        self.regions().hash(state);
        self.input_count().hash(state);
        (0..self.input_count()).for_each(|index| self.input(index).hash(state));
        self.mesh().hash(state);
    }
}

impl PartialEq for EagerDispatchKey {
    fn eq(&self, other: &Self) -> bool {
        (self as &dyn EagerDispatchKeyView) == (other as &dyn EagerDispatchKeyView)
    }
}

impl Eq for EagerDispatchKey {}

impl Hash for EagerDispatchKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self as &dyn EagerDispatchKeyView).hash(state);
    }
}

impl<'o> std::borrow::Borrow<dyn EagerDispatchKeyView + 'o> for EagerDispatchKey {
    fn borrow(&self) -> &(dyn EagerDispatchKeyView + 'o) {
        self
    }
}

/// Borrowed [`EagerDispatchKey`] over the arguments of one eager application, used to look up cache hits without
/// building an owned key.
pub(crate) struct EagerDispatchQuery<'o> {
    /// Applied operation.
    operation: &'o Operation,

    /// Keys of the regions attached to the application, in the order that the operation defines.
    regions: &'o [RegionKey<Constant, Operation>],

    /// Inputs of the application, which contain no references.
    inputs: &'o [Value],

    /// Mesh that the eager lowering compiles against, as documented on [`EagerDispatchKey`].
    mesh: Option<&'o DeviceMesh>,
}

impl<'o> EagerDispatchQuery<'o> {
    /// Creates a query that borrows the arguments of one eager operation application.
    pub(crate) fn new(
        operation: &'o Operation,
        regions: &'o [RegionKey<Constant, Operation>],
        inputs: &'o [Value],
        mesh: Option<&'o DeviceMesh>,
    ) -> Self {
        Self { operation, regions, inputs, mesh }
    }
}

impl EagerDispatchKeyView for EagerDispatchQuery<'_> {
    fn operation(&self) -> &Operation {
        self.operation
    }

    fn regions(&self) -> &[RegionKey<Constant, Operation>] {
        self.regions
    }

    fn input_count(&self) -> usize {
        self.inputs.len()
    }

    fn input(&self, index: usize) -> EagerDispatchInputRef<'_> {
        match &self.inputs[index] {
            Value::Array(array) => EagerDispatchInputRef::Array(array.r#type(), array.sharding()),
            Value::Dimension(dimension) => EagerDispatchInputRef::Dimension(dimension.value()),
            Value::Reference(_) => unreachable!("reference inputs are rejected before the dispatch cache lookup"),
        }
    }

    fn mesh(&self) -> Option<&DeviceMesh> {
        self.mesh
    }
}

/// Derives the retained dispatch key, the effective staged input types, and any options specialized on input
/// placements for one dispatch request. This function implements
/// [`CompilationDomain::dispatch_signature`](ryft_core::CompilationDomain::dispatch_signature) for
/// [`Session`](crate::Session).
pub(super) fn dispatch_signature(
    inputs: DispatchInputs<'_, Value>,
    options: &Options,
) -> Result<(DispatchKey, Arc<[Type]>, Option<Options>), Error> {
    let mut inputs = match inputs {
        DispatchInputs::Call(mut inputs) => {
            // Available runtime values inform the staged types. The enclosing call determines actual placement.
            for (input_type, input) in &mut inputs {
                if let (Type::Array(array_type), Some(Value::Array(array))) = (&*input_type, *input)
                    && let Some(relocated_type) = argument_type(array, array_type)
                {
                    *input_type = Type::Array(relocated_type);
                }
            }
            let (key, input_types) = normalize_dispatch_signature(inputs, options, None)?;
            return Ok((key, input_types, None));
        }
        DispatchInputs::Values(inputs) => inputs,
    };

    // A relocatable argument is specialized for the same type as in a staged call. The executable signature for
    // other arguments records runtime placements that name auto mesh axes, which the input types omit.
    let mut placements = None::<Vec<Option<Sharding>>>;
    for (index, (input_type, input)) in inputs.iter_mut().enumerate() {
        let placement = match (*input, &*input_type) {
            (Value::Array(array), Type::Array(array_type)) => match argument_type(array, array_type) {
                Some(relocated_type) => {
                    *input_type = Type::Array(relocated_type);
                    None
                }
                None => array.sharding().references_auto_axis().then(|| array.sharding().clone()),
            },
            (Value::Array(array), _) => array.sharding().references_auto_axis().then(|| array.sharding().clone()),
            _ => None,
        };
        match (placement, &mut placements) {
            (Some(placement), placements) => placements.get_or_insert_with(|| vec![None; index]).push(Some(placement)),
            (None, Some(placements)) => placements.push(None),
            (None, None) => {}
        }
    }
    let (key, input_types) = normalize_dispatch_signature(inputs, options, options.input_bound_bucketing())?;
    let Some(placements) = placements.filter(|_| options.input_shardings().is_none()) else {
        return Ok((key, input_types, None));
    };
    let placements: Arc<[_]> = placements.into();
    let options = options.clone().with_input_placements(Arc::clone(&placements));
    Ok((DispatchKey { placements: Some(placements), ..key }, input_types, Some(options)))
}

/// Encodes one boundary's input and output types using the canonical numbering of dimension variables of the
/// persistent signature.
///
/// [`ArrayType`] equality compares dimension variable identities, and neither two structurally identical lowerings
/// nor one lowering and the executable restored for it from a cache share variable identities. Boundary compatibility
/// must therefore be compared through the same canonical form without identities that the compilation cache key uses.
pub(super) fn canonical_boundary_types(input_types: &[ArrayType], output_types: &[ArrayType]) -> Vec<u8> {
    // Serializing the plain persistent data model cannot fail.
    serde_json::to_vec(&PersistentArraySignature::encode(input_types, output_types).into_canonical()).unwrap()
}

/// Internal representation of [`DispatchKey`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum DispatchKeyKind {
    /// Exact runtime abstract signature used when bucketing is disabled, shared with the effective staged types.
    Exact(Arc<[Type]>),

    /// Bounded signature using canonical variable indices when bucketing is enabled.
    Bucketed(BucketedDispatchSignature),
}

/// Bounded input signature with dimension variables numbered by first occurrence, used as the key of retained dispatch.
///
/// Dimension variables are numbered by first occurrence across the whole input signature under
/// [`DimensionVariable`] identity, and only their bounds are retained. Two bucketed signatures are therefore equal
/// exactly when they agree on every input's data type, layout, sharding, memory space, rank, static extent of each
/// axis, and pattern of variable sharing across axes with matching bounds, regardless of how the variables are named.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct BucketedDispatchSignature {
    /// `(lower, upper)` bounds of each distinct dimension variable, in the order of their first occurrence.
    variables: Vec<(usize, Option<usize>)>,

    /// Bucketed input array types with their shapes rewritten into patterns of variable indices.
    input_types: Vec<BucketedArrayTypeKey>,
}

/// One bucketed input array type split into a carrier without a shape and a shape pattern of canonical variable
/// indices.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct BucketedArrayTypeKey {
    /// Array type with an empty shape, retaining data type, layout, sharding, and memory identity.
    carrier: ArrayType,

    /// Static extent or canonical variable index of each axis, which also fix the array's rank.
    dimensions: Vec<BucketedDimensionKey>,
}

/// One static extent or canonical variable index of a [`BucketedArrayTypeKey`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum BucketedDimensionKey {
    /// Statically known extent. Bucketing rewrites every static input axis into a bounded dynamic axis, so this
    /// variant is unreachable for signatures produced today. It is retained so that the normalization stays total over
    /// [`Dimension`].
    Static(usize),

    /// Index of the axis' dimension variable in [`BucketedDispatchSignature::variables`].
    Variable(usize),
}

/// Stable cache key payload covering the complete executable and the ABI of external references.
#[derive(Serialize)]
struct PersistentKey<'o> {
    /// Version of this persistent record schema.
    schema_version: u32,

    /// Exact textual StableHLO and Shardy module submitted to PJRT.
    stable_hlo: &'o str,

    /// Canonical encoding of the complete PJRT compilation options.
    compilation_options: &'o [u8],

    /// Logical array types with shared dynamic dimension variables.
    signature: PersistentCanonicalArraySignature,

    /// Physical executable argument position for each logical input, or `None` for erased arrays.
    input_mapping: Vec<Option<u64>>,

    /// Whether each logical input is erased because the executable never reads it, or empty when every input is read,
    /// in which case it is omitted, so that the key of such a program is what it was before unread inputs were
    /// recorded.
    #[serde(skip_serializing_if = "<[bool]>::is_empty")]
    unread_inputs: &'o [bool],

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
    donation_flags: &'o [bool],

    /// Number of leading logical inputs supplied by captures.
    capture_count: u64,

    /// Shardings of all physical executable arguments in argument order.
    expected_argument_shardings: Vec<PersistentSharding>,

    /// Runtime placements of all logical outputs, including the auto mesh axes that output types never record.
    output_shardings: Vec<PersistentSharding>,

    /// Whether executions commit each logical output whatever the commitment of their inputs, which lowerings of the
    /// same module can differ in (e.g., the identity and a placement onto the memory that a device array is already
    /// in, which lowers to nothing but commits the result).
    committed_outputs: &'o [bool],

    /// Concrete compilation and execution mesh.
    mesh: PersistentDeviceMesh,

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

/// Encodes compilation options deterministically even though Prost represents protobuf maps as randomly seeded
/// `HashMap`s. Map entries are removed from a clone before normal protobuf encoding and appended as records that are
/// delimited by lengths, qualified by paths, and sorted by their encoded keys and values.
pub(crate) fn canonical_compilation_options_bytes(options: &CompilationOptions) -> Vec<u8> {
    /// Appends one sorted protobuf map record with an explicit path and length delimiters.
    fn append_record(output: &mut Vec<u8>, path: &[u8], mut entries: Vec<(Vec<u8>, Vec<u8>)>) {
        entries.sort_unstable();
        output.extend_from_slice(&(path.len() as u64).to_le_bytes());
        output.extend_from_slice(path);
        output.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        for (key, value) in entries {
            output.extend_from_slice(&(key.len() as u64).to_le_bytes());
            output.extend_from_slice(key.as_slice());
            output.extend_from_slice(&(value.len() as u64).to_le_bytes());
            output.extend_from_slice(value.as_slice());
        }
    }

    /// Removes tuning knobs from a protobuf algorithm key and appends their sorted canonical record.
    fn drain_algorithm_map(output: &mut Vec<u8>, path: &[u8], algorithm: &mut ryft_pjrt::protos::AutoTuneAlgorithmKey) {
        append_record(
            output,
            path,
            std::mem::take(&mut algorithm.tuning_knobs)
                .into_iter()
                .map(|(key, value)| (key.to_le_bytes().to_vec(), value.to_le_bytes().to_vec()))
                .collect(),
        );
    }

    let mut options = options.clone();
    let mut maps = Vec::new();
    append_record(
        &mut maps,
        b"compilation.environment_option_overrides",
        std::mem::take(&mut options.environment_option_overrides)
            .into_iter()
            .map(|(key, value)| (key.into_bytes(), value.encode_to_vec()))
            .collect(),
    );
    if let Some(debug_options) =
        options.executable_build_options.as_mut().and_then(|options| options.debug_options.as_mut())
    {
        append_record(
            &mut maps,
            b"executable.debug.analytical_latency_estimator",
            std::mem::take(&mut debug_options.xla_gpu_analytical_latency_estimator_options)
                .into_iter()
                .map(|(key, value)| (key.into_bytes(), value.into_bytes()))
                .collect(),
        );
        append_record(
            &mut maps,
            b"executable.debug.experimental_cost_model_gemm_tiling",
            std::mem::take(&mut debug_options.xla_gpu_experimental_cost_model_gemm_tiling_options)
                .into_iter()
                .map(|(key, value)| (key.into_bytes(), value.into_bytes()))
                .collect(),
        );
        append_record(
            &mut maps,
            b"executable.debug.backend_extra_options",
            std::mem::take(&mut debug_options.xla_backend_extra_options)
                .into_iter()
                .map(|(key, value)| (key.into_bytes(), value.into_bytes()))
                .collect(),
        );
    }
    if let Some(target) = options.target_config.as_mut() {
        if let Some(device) = target.gpu_device_information.as_mut() {
            for (name, description) in [
                (b"target.device.scalar_rates".as_slice(), device.scalar_unit_description.as_mut()),
                (b"target.device.matrix_rates".as_slice(), device.matrix_unit_description.as_mut()),
            ] {
                if let Some(description) = description {
                    append_record(
                        &mut maps,
                        name,
                        std::mem::take(&mut description.rate_information)
                            .into_iter()
                            .map(|(key, value)| (key.to_le_bytes().to_vec(), value.encode_to_vec()))
                            .collect(),
                    );
                }
            }
        }
        if let Some(autotune_results) = target.autotune_results.as_mut() {
            for (index, entry) in autotune_results.results.iter_mut().enumerate() {
                let Some(result) = entry.result.as_mut() else {
                    continue;
                };
                if let Some(AutoTuneResultKey::Algorithm(algorithm)) = result.key.as_mut() {
                    drain_algorithm_map(&mut maps, format!("target.autotune.{index}.algorithm").as_bytes(), algorithm);
                }
                if let Some(AutoTuneReferenceKey::ReferenceAlgorithm(algorithm)) =
                    result.failure.as_mut().and_then(|failure| failure.key.as_mut())
                {
                    drain_algorithm_map(
                        &mut maps,
                        format!("target.autotune.{index}.reference_algorithm").as_bytes(),
                        algorithm,
                    );
                }
            }
        }
    }

    let protobuf = options.encode_to_vec();
    let mut bytes = b"RYFT_XLA_COMPILATION_OPTIONS_V1".to_vec();
    bytes.extend_from_slice(&(protobuf.len() as u64).to_le_bytes());
    bytes.extend_from_slice(protobuf.as_slice());
    bytes.extend_from_slice(maps.as_slice());
    bytes
}

/// Applies input sharding declarations and optional executable bucketing, without inspecting runtime values.
fn normalize_dispatch_signature<T>(
    mut inputs: Vec<(Type, T)>,
    options: &Options,
    bucketing: Option<InputBoundBucketing>,
) -> Result<(DispatchKey, Arc<[Type]>), Error> {
    apply_input_shardings(&mut inputs, options, |(r#type, _)| r#type)?;
    let bucketed = bucketing
        .map(|bucketing| bucket_dispatch_signature(inputs.iter_mut().map(|(r#type, _)| r#type), options, bucketing))
        .transpose()?;
    // The projection retains the exact length, so collecting allocates the shared type slice directly.
    let input_types: Arc<[_]> = inputs.into_iter().map(|(r#type, _)| r#type).collect();
    let kind = bucketed.map_or_else(|| DispatchKeyKind::Exact(input_types.clone()), DispatchKeyKind::Bucketed);
    Ok((DispatchKey::new(kind), input_types))
}

/// Buckets the effective input types in place and numbers dimension variables by first occurrence across the inputs.
fn bucket_dispatch_signature<'t>(
    input_types: impl ExactSizeIterator<Item = &'t mut Type>,
    options: &Options,
    bucketing: InputBoundBucketing,
) -> Result<BucketedDispatchSignature, Error> {
    if options.mesh().devices().len() > 1 {
        return Err(Error::Compilation(CompilationError::InvalidOptions {
            reason: "input-bound bucketing is unsupported on multi-device meshes because bucketed dynamic \
                     signatures require disabling the Shardy partitioner"
                .to_string(),
        }));
    }
    let mut variables = Vec::<DimensionVariable>::new();
    let mut bucketed_input_types = Vec::with_capacity(input_types.len());
    for (input_index, r#type) in input_types.enumerate() {
        let array_type = match r#type {
            Type::Array(array_type) => array_type,
            Type::Dimension(_) => {
                return Err(Error::Compilation(CompilationError::InvalidOptions {
                    reason: format!(
                        "input-bound bucketing only supports array inputs, but input {input_index} has first-class \
                         dimension type `{type}`",
                    ),
                }));
            }
            Type::Reference(_) => {
                return Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi {
                    reason: format!("external reference input {input_index} does not support input-bound bucketing"),
                }));
            }
        };
        let dimensions = array_type
            .shape()
            .dimensions()
            .iter()
            .enumerate()
            .map(|(axis, dimension)| match dimension {
                Dimension::Static(extent) => {
                    let bound = match bucketing {
                        InputBoundBucketing::PowerOfTwo => extent.checked_next_power_of_two(),
                    }
                    .and_then(|bound| bound.checked_add(1))
                    .ok_or_else(|| ProgramError::InvalidArgument {
                        message: format!(
                            "cannot select a finite input-bound bucket for input {input_index} axis {axis} with \
                             extent {extent}",
                        ),
                    })?;
                    let bounds = DimensionBounds::new(0, Some(bound))?;
                    Ok(Dimension::Dynamic(DimensionVariable::new(format!("input_{input_index}_axis_{axis}"), bounds)))
                }
                Dimension::Dynamic(variable) => Ok(Dimension::Dynamic(variable.clone())),
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;
        *array_type = array_type.clone().with_shape(Shape::new(dimensions));
        bucketed_input_types.push(BucketedArrayTypeKey {
            carrier: array_type.clone().with_shape(Shape::new(Vec::new())),
            dimensions: array_type
                .shape()
                .dimensions()
                .iter()
                .map(|dimension| match dimension {
                    Dimension::Static(extent) => BucketedDimensionKey::Static(*extent),
                    Dimension::Dynamic(variable) => {
                        let index = variables.iter().position(|existing| existing == variable).unwrap_or_else(|| {
                            variables.push(variable.clone());
                            variables.len() - 1
                        });
                        BucketedDimensionKey::Variable(index)
                    }
                })
                .collect(),
        });
    }
    let variables = variables.iter().map(|variable| (variable.bounds().lower(), variable.bounds().upper())).collect();
    Ok(BucketedDispatchSignature { variables, input_types: bucketed_input_types })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, ArrayType, BatchAxis, BatchAxisSpecification, CompilationCacheDomain, CompilationDomain,
        CompilationTracer, Compile, CompiledFunction, CompiledFunctionDispatcher, Context, CustomCallOperation,
        DataType, Device, Dimension as CoreDimension, DimensionBounds, DimensionFromScalar, DimensionType,
        DimensionValue, DimensionVariable, DynamicBroadcastOperation, LogicalMesh, Memory, MeshAxis, MeshAxisType,
        PartialEvaluationContext, PartialEvaluationValue, PartialTracer, Place, Placeholder, ProgramBuilder,
        ReduceOperation, ReductionKind, ReferenceType, Shape, Sharding, ShardingDimension, Sin, StaticShape,
        ValueProjection, batch_in_parent, differentiate_in_parent_at,
    };
    use ryft_pjrt::protos::{CompilationOptions, DebugOptions, ExecutableCompilationOptions, OptionOverride};
    #[cfg(feature = "cuda-13")]
    use ryft_pjrt::{ClientOptions, GpuClientOptions, GpuMemoryAllocator, GpuPlatform};

    #[cfg(feature = "cuda-13")]
    use crate::Platform;
    use crate::compilation::CompilationError;
    use crate::compilation::options::Options;
    use crate::compilation::tests::external_reference_read_program;
    #[cfg(feature = "cuda-13")]
    use crate::sessions::tests::replicated_vector_type;
    use crate::sessions::tests::{domain_mesh, f32_vector, program_array, read_first_shard, replicated_scalar_type};
    use crate::tests::{cpu_client, execution_client, hash_of, values_to_bytes};
    use crate::{
        Array, CompileTracer, Constant, CustomCallOperationExtension, Dimension, Error, FromPjrt, Operation,
        RuntimeRequirement, Session, Type, Value, compile, jit,
    };

    use super::*;

    /// Traces the sum of `value` broadcast to the data-derived `extent` that `size` selects, from the inputs
    /// `[size, value]`.
    fn trace_data_derived(
        extent: DimensionVariable,
        inputs: Vec<CompilationTracer<Session>>,
    ) -> Result<CompilationTracer<Session>, Error> {
        let [size, value] = inputs.as_slice() else {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
        };
        let dimension = size.to_dimension(extent)?;
        let broadcast = size
            .context()
            .bind(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), &[value.clone(), dimension])?
            .remove(0);
        Ok(size
            .context()
            .bind(ReduceOperation::new(vec![0], ReductionKind::Sum), Vec::new(), &[broadcast])?
            .remove(0))
    }

    #[test]
    fn test_dispatch_signature_relocates_uncommitted_arguments() {
        // A `jit` dispatcher and a function compiled ahead of time for the type that the dispatcher specializes for
        // agree, at a direct execution and at an eager call, for every argument and every mesh of the function, which
        // is named, so an eager call runs on it as an execution does. An uncommitted argument that records nothing
        // beyond replication moves there (the function compiled ahead of time is compiled for its type without a
        // sharding). A committed anonymous argument is admitted on the same device. The mesh of the function still
        // rejects incompatible devices and a split over another logical mesh. Each observation includes the runtime
        // sharding of the result, because an eager call and an
        // execution of one dispatcher must view an uncommitted argument alike (refer to `argument_type`): before, an
        // eager call specialized for its type on the canonical mesh of its device, so it rejected a placement onto a
        // bare device that every other entry point accepted.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let vector_type = ArrayType::new_static(DataType::F32, [2]);
        let bytes = values_to_bytes(&[1f32, 2f32]);
        let mesh = |name: &str, axis_type: MeshAxisType, devices: Vec<Device>| {
            let logical_mesh = LogicalMesh::new(vec![MeshAxis::new(name, devices.len(), axis_type).unwrap()]).unwrap();
            DeviceMesh::new(logical_mesh, devices).unwrap()
        };
        let split = |axis_type: MeshAxisType| {
            let mesh = mesh("x", axis_type, devices.clone());
            let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
            Array::from_host_buffer(&session, vector_type.clone(), mesh, Some(sharding), bytes.clone()).unwrap()
        };
        let uncommitted = Array::from_host_buffer(&session, vector_type.clone(), None, None, bytes.clone()).unwrap();
        let one_device_split = |committed: bool| {
            let mesh = mesh("x", MeshAxisType::Auto, vec![devices[0]]);
            let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
            let array = Array::from_host_buffer(&session, vector_type.clone(), mesh, Some(sharding), bytes.clone());
            array.unwrap().with_committed(committed)
        };
        let arguments = [
            ("uncommitted", uncommitted.clone()),
            ("committed here", uncommitted.to_placement(&session, devices[0]).unwrap()),
            ("committed elsewhere", uncommitted.to_placement(&session, devices[1]).unwrap()),
            ("split along an auto axis", split(MeshAxisType::Auto)),
            ("split along an explicit axis", split(MeshAxisType::Explicit)),
            ("split along an auto axis of one device", one_device_split(true)),
            ("uncommitted split along an auto axis of one device", one_device_split(false)),
        ];
        let meshes = [
            ("named here", mesh("q", MeshAxisType::Explicit, vec![devices[0]])),
            ("named elsewhere", mesh("q", MeshAxisType::Explicit, vec![devices[1]])),
            ("two devices", mesh("x", MeshAxisType::Auto, devices.clone())),
            ("one device", mesh("x", MeshAxisType::Auto, vec![devices[0]])),
        ];
        let observe = |result: Result<Array, Error>| {
            result
                .map(|array| {
                    let devices = array.mesh().devices().iter().map(|device| device.id()).collect::<Vec<_>>();
                    let values = array.copy_to_host().unwrap().elements::<f32>().unwrap();
                    (devices, array.is_committed(), values, array.sharding().to_string())
                })
                .map_err(|error| error.to_string())
        };
        let mut uncommitted_rows = Vec::new();
        let mut split_rows = Vec::new();
        let mut committed_rows = Vec::new();
        let mut rejected_rows = Vec::new();
        for (mesh_name, mesh) in &meshes {
            let dispatcher = jit::<_, (), ArrayType, ArrayType>(|(), x| Ok(x), &session, mesh.clone());
            for (argument_name, argument) in &arguments {
                let aot_type =
                    if argument.is_committed() { argument.r#type().into_owned() } else { vector_type.clone() };
                let execution = observe(dispatcher.execute((), argument.clone()));
                let call = observe(dispatcher.call((), argument.clone()).map_err(Error::from));
                match compile::<_, ArrayType, ArrayType>(|x| x, aot_type, &session, mesh.clone()) {
                    Ok(compiled) => {
                        let aot_execution = observe(compiled.executable().execute(&session, argument.clone()));
                        let aot_call = observe(compiled.call(argument.clone()).map_err(Error::from));
                        // A dispatcher keeps the placement of a committed split along an auto axis, which a function
                        // compiled for its type replicates (refer to the register entry "A call of a `jit` dispatcher
                        // keeps its inputs where they are"), so the runtime shardings are compared below instead.
                        let placed = |observation: &Result<_, String>| {
                            observation.clone().map(|(devices, committed, values, _)| (devices, committed, values))
                        };
                        assert_eq!(
                            (placed(&execution), placed(&call)),
                            (placed(&aot_execution), placed(&aot_call)),
                            "{mesh_name}, {argument_name}",
                        );
                    }
                    // The mesh of the function rejects the type of a committed argument or of a split over another
                    // logical mesh, ahead of time and at a direct execution of the dispatcher alike, and an eager call
                    // of the dispatcher rejects it too, because it runs on that mesh, with the error of its devices or
                    // of its sharding, which the dispatcher keeps (refer to the register entry "A call of a `jit`
                    // dispatcher keeps its inputs where they are").
                    Err(error) => {
                        assert_eq!(execution, Err(error.to_string()), "{mesh_name}, {argument_name}");
                    }
                }
                if call.is_err() {
                    rejected_rows.push((*mesh_name, *argument_name, call.clone()));
                } else if argument.is_committed() && argument.mesh().logical_mesh() == &LogicalMesh::single_device() {
                    committed_rows.push((*mesh_name, execution.clone(), call.clone()));
                }
                if !argument.is_committed() {
                    uncommitted_rows.push((*mesh_name, *argument_name, execution, call));
                } else if *argument_name == "split along an auto axis of one device" {
                    split_rows.push((*mesh_name, execution, call));
                }
            }
        }

        // A placement of an uncommitted argument onto the bare device of a mesh of one device commits the result there,
        // at every entry point alike.
        let mut placement_rows = Vec::new();
        for (mesh_name, mesh) in &meshes[..2] {
            let device = mesh.devices()[0];
            let context = session.with_device_mesh(mesh.clone());
            let dispatcher = jit::<_, (), ArrayType, ArrayType>(
                move |(), x: CompileTracer| Ok(x.place(&device, None)?),
                &session,
                mesh.clone(),
            );
            let compiled = compile::<_, ArrayType, ArrayType>(
                move |x: CompileTracer| x.place(&device, None).unwrap(),
                vector_type.clone(),
                &session,
                mesh.clone(),
            )
            .unwrap();
            let observations = [
                observe(dispatcher.execute((), uncommitted.clone())),
                observe(dispatcher.call((), uncommitted.clone()).map_err(Error::from)),
                observe(dispatcher.call_in_context(&context, (), uncommitted.clone()).map_err(Error::from)),
                observe(compiled.executable().execute(&session, uncommitted.clone())),
                observe(compiled.call(uncommitted.clone()).map_err(Error::from)),
            ];
            assert!(
                observations.iter().all(|observation| *observation == observations[0]),
                "{mesh_name}: {observations:#?}",
            );
            placement_rows.push((*mesh_name, observations[0].clone()));
        }
        let values = vec![1f32, 2f32];
        let placed = |device: usize| Ok((vec![device], true, values.clone(), "{mesh<['q'=1:explicit]>, [{}]}".into()));
        assert_eq!(placement_rows, vec![("named here", placed(0)), ("named elsewhere", placed(1))]);

        // An execution and an eager call place an uncommitted argument on the named mesh of the function and commit the
        // result there, as JAX does under `set_mesh`. A split of an uncommitted argument along an auto axis is
        // relocatable, so neither keeps it, whereas both keep the split of a committed one.
        let on = |devices: Vec<usize>, committed: bool, sharding: &str| {
            Ok::<_, String>((devices, committed, values.clone(), sharding.to_string()))
        };
        let canonical_split = on(vec![0], true, "{mesh<['x'=1:auto]>, [{}]}");
        let named = |device: usize| on(vec![device], true, "{mesh<['q'=1:explicit]>, [{}]}");
        assert_eq!(
            committed_rows,
            vec![
                ("named here", named(0), named(0)),
                ("named elsewhere", named(1), named(1)),
                ("one device", canonical_split.clone(), canonical_split.clone()),
            ],
        );
        let one_device_split = "uncommitted split along an auto axis of one device";
        let both_devices = on(vec![0, 1], true, "{mesh<['x'=2:auto]>, [{}]}");
        assert_eq!(
            uncommitted_rows,
            vec![
                ("named here", "uncommitted", named(0), named(0)),
                ("named here", one_device_split, named(0), named(0)),
                ("named elsewhere", "uncommitted", named(1), named(1)),
                ("named elsewhere", one_device_split, named(1), named(1)),
                ("two devices", "uncommitted", both_devices.clone(), both_devices.clone()),
                ("two devices", one_device_split, both_devices.clone(), both_devices),
                ("one device", "uncommitted", canonical_split.clone(), canonical_split.clone()),
                ("one device", one_device_split, canonical_split.clone(), canonical_split),
            ],
        );
        let kept_split = on(vec![0], true, "{mesh<['x'=1:auto]>, [{'x'}]}");
        assert_eq!(split_rows.last(), Some(&("one device", kept_split.clone(), kept_split)));
        let committed_to = |actual: &str, required: &str| {
            Err::<(Vec<usize>, bool, Vec<f32>, String), _>(format!(
                "public argument #0 is committed to devices {actual}, but the executable requires {required}"
            ))
        };
        let sharded_as = |actual: &str, required: &str| {
            Err::<(Vec<usize>, bool, Vec<f32>, String), _>(format!(
                "array type sharding `{actual}` does not match the placement `{required}` over the non-auto mesh axes"
            ))
        };
        let (q, x) = ("{mesh<['q'=1:explicit]>, [{}]}", "{mesh<['x'=1:auto]>, [{}]}");
        let (auto_split, explicit_split, committed_split) =
            ("split along an auto axis", "split along an explicit axis", "split along an auto axis of one device");
        assert_eq!(
            rejected_rows,
            vec![
                ("named here", "committed elsewhere", committed_to("[1]", "[0]")),
                ("named here", auto_split, committed_to("[0, 1]", "[0]")),
                ("named here", explicit_split, committed_to("[0, 1]", "[0]")),
                ("named here", committed_split, sharded_as(x, q)),
                ("named elsewhere", "committed here", committed_to("[0]", "[1]")),
                ("named elsewhere", auto_split, committed_to("[0, 1]", "[1]")),
                ("named elsewhere", explicit_split, committed_to("[0, 1]", "[1]")),
                ("named elsewhere", committed_split, committed_to("[0]", "[1]")),
                ("two devices", "committed here", committed_to("[0]", "[0, 1]")),
                ("two devices", "committed elsewhere", committed_to("[1]", "[0, 1]")),
                ("two devices", committed_split, committed_to("[0]", "[0, 1]")),
                ("one device", "committed elsewhere", committed_to("[1]", "[0]")),
                ("one device", auto_split, committed_to("[0, 1]", "[0]")),
                ("one device", explicit_split, committed_to("[0, 1]", "[0]")),
            ],
        );
    }

    #[test]
    fn test_dispatch_signature_relocates_uncommitted_arguments_under_transforms() {
        // A call of a `jit` dispatcher under JVP, batching, or partial evaluation with a known input sees the runtime
        // value that the transform wraps (i.e., the primal of a dual, the packed value of a batch, or a known value,
        // refer to `ImportingContext::as_source_value`), so it specializes an uncommitted argument for its type without
        // a sharding and places it on the named mesh of one device, as `.call` does. Before, each of these specialized
        // for the canonical upload mesh and rejected the argument with a mesh mismatch, both on the device where the
        // argument lives and on another one.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let values = vec![1f32, 2f32];
        let vector_type = ArrayType::new_static(DataType::F32, [2]);
        let uncommitted = Array::from_host_buffer(&session, vector_type, None, None, values_to_bytes(&values)).unwrap();
        let observe = |result: Result<Array, String>| {
            result.map(|array| {
                let devices = array.mesh().devices().iter().map(|device| device.id()).collect::<Vec<_>>();
                let values = array.copy_to_host().unwrap().elements::<f32>().unwrap();
                (devices, array.is_committed(), values, array.sharding().to_string())
            })
        };
        let program_error = |error: Error| ProgramError::InvalidArgument { message: error.to_string() };
        for index in [0, 1] {
            let device = Device::from_pjrt(&client.addressable_devices().unwrap()[index]).unwrap();
            let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("q", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
            let mesh = DeviceMesh::new(logical_mesh, vec![device]).unwrap();
            let place = move |(), x: CompileTracer| -> Result<CompileTracer, Error> { Ok(x.place(&device, None)?) };
            let dispatcher = jit::<_, (), ArrayType, ArrayType>(place, &session, mesh);
            let call = dispatcher.call((), uncommitted.clone()).map_err(|error| error.to_string());
            let (primal, tangent) = match differentiate_in_parent_at(uncommitted.clone())
                .jvp(uncommitted.clone(), |x| dispatcher.call((), x).map_err(program_error))
            {
                Ok((primal, tangent)) => (Ok(primal), Ok(tangent)),
                Err(error) => (Err(error.to_string()), Err(error.to_string())),
            };
            let batched = batch_in_parent(
                |x| dispatcher.call((), x).map_err(program_error),
                uncommitted.clone(),
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxisSpecification::default(),
            )
            .map_err(|error| error.to_string());
            let context = PartialEvaluationContext::new(session.clone());
            let known = PartialEvaluationValue::known(Value::Array(uncommitted.clone()));
            let known = ValueProjection::<ArrayType>::into_projected(PartialTracer::new(context.clone(), known));
            let partial = dispatcher
                .call_in_context(&context, (), known.unwrap())
                .map_err(|error| error.to_string())
                .and_then(|output| output.into_value().into_value().map_err(|error| error.to_string()))
                .and_then(|output| match output.as_known() {
                    Some(Value::Array(array)) => Ok(array.clone()),
                    _ => Err("expected a known array".to_string()),
                });
            let expected = Ok((vec![index], true, values.clone(), "{mesh<['q'=1:explicit]>, [{}]}".to_string()));
            assert_eq!(observe(call), expected.clone());
            assert_eq!(
                [primal, tangent, batched, partial].map(observe),
                [expected.clone(), expected.clone(), expected.clone(), expected],
                "device {index}",
            );
        }
    }

    #[test]
    fn test_dispatch_signature_keeps_mixed_call_inputs_aligned() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let uncommitted = f32_vector(&session, &mesh, &[1.0, 2.0]).with_committed(false);
        let declared = uncommitted.r#type().into_owned();
        let normalized = declared.clone().with_sharding(None).unwrap();
        assert_ne!(declared, normalized);

        // Only the inputs with concrete values lose their incidental placement.
        let expected = [declared.clone(), normalized.clone(), declared.clone(), normalized, declared.clone()];
        let dispatcher = jit::<_, (), Vec<ArrayType>, ArrayType>(
            move |(), inputs: Vec<CompileTracer>| {
                assert_eq!(inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(), expected);
                Ok(inputs[1].clone())
            },
            &session,
            mesh,
        );
        let context = PartialEvaluationContext::new(session.clone());
        let unknown = |index| context.unknown_input(Type::Array(declared.clone()), index);
        let known = || PartialEvaluationValue::known(Value::Array(uncommitted.clone()));
        let inputs = [unknown(0), known(), unknown(1), known(), unknown(2)]
            .into_iter()
            .map(|value| {
                ValueProjection::<ArrayType>::into_projected(PartialTracer::new(context.clone(), value)).unwrap()
            })
            .collect::<Vec<_>>();
        dispatcher.call_in_context(&context, (), inputs).unwrap();
    }

    #[test]
    fn test_dispatch_signature_validates_declared_sharding_before_call_normalization() {
        let client = cpu_client(1);
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh, vec![device]).unwrap();
        let session = Session::builder().client(client).build().unwrap();
        let values = [1.0f32, 2.0];
        // The declaration names a sharding that the concrete input satisfies. A relocatable argument loses that
        // sharding when staged, so validating the normalized type would reject this valid input.
        let array = f32_vector(&session, &mesh, &values).with_committed(false);
        let declared = array.r#type().into_owned();
        assert_eq!(declared.sharding(), Some(array.sharding()));
        let normalized = Type::Array(declared.clone().with_sharding(None).unwrap());
        let dispatcher: CompiledFunctionDispatcher<Session, (), Type, Type, _> = session.jit_with_options(
            move |(), input: CompilationTracer<Session>| {
                assert_eq!(input.r#type().as_ref(), &normalized);
                Ok(input)
            },
            Options::new(mesh.clone()),
        );
        let declared_type = declared.clone();
        let dispatcher = dispatcher.with_staged_input_types(Some(Type::Array(declared))).unwrap();
        let input = Value::Array(array);
        let called = dispatcher.call((), input.clone()).unwrap();
        let executed = dispatcher.execute((), input).unwrap();
        let called = program_array(&called);
        let executed = program_array(&executed);
        assert_eq!(called.mesh(), executed.mesh());
        assert_eq!(called.sharding(), executed.sharding());
        assert_eq!(called.is_committed(), executed.is_committed());
        assert_eq!(called.copy_to_host().unwrap().elements::<f32>().unwrap(), values);
        assert_eq!(executed.copy_to_host().unwrap().elements::<f32>().unwrap(), values);

        let other_mesh = domain_mesh(client, "other", 1);
        let incompatible = f32_vector(&session, &other_mesh, &values).with_committed(false);
        let message = format!(
            "compilation input `$` has type `{}`, which does not refine its declared staged type `{}`",
            Type::Array(incompatible.r#type().into_owned()),
            Type::Array(declared_type),
        );
        let incompatible = Value::Array(incompatible);
        let call_error = dispatcher.call((), incompatible.clone()).unwrap_err();
        let execute_error = dispatcher.execute((), incompatible).unwrap_err();
        assert_eq!(call_error.to_string(), message);
        assert_eq!(execute_error.to_string(), message);
    }

    #[test]
    fn test_lowered_program_compilation_key() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let staged = crate::stage::<_, ArrayType, ArrayType>(|x| x.sin().unwrap(), input_type, Options::new(mesh))
            .unwrap()
            .into_inner();
        let compiled: CompiledFunction<Session, Type, Type> =
            staged.lower(&session).unwrap().compile(&session).unwrap();
        let key = session.compilation_key(compiled.lowered().lowered_program()).unwrap();
        let independent_session = Session::builder().client(client).build().unwrap();
        let repeated = independent_session.compilation_key(compiled.lowered().lowered_program()).unwrap();

        assert_eq!(key, repeated);
        assert_eq!(session.persistent_cache_key(&key), Some(key.canonical_bytes.to_vec()));
        let decoded: serde_json::Value = serde_json::from_slice(&key.canonical_bytes).unwrap();
        assert_eq!(decoded["schema_version"], PERSISTENT_KEY_SCHEMA_VERSION);
        assert_eq!(decoded["runtime_requirements"], serde_json::json!([]));
        assert_eq!(decoded["compiler_identity"], COMPILER_IDENTITY.as_str());
        assert!(decoded["compiler_identity"].as_str().unwrap().contains(ryft_xla_sys::XLA_COMMIT));
        assert!(decoded["compiler_identity"].as_str().unwrap().contains(ryft_xla_sys::JAX_COMMIT));
        assert_eq!(decoded["platform_name"], client.platform_name().unwrap().as_ref());
        assert!(decoded["compilation_options"].as_array().is_some_and(|bytes| !bytes.is_empty()));
        assert!(decoded["stable_hlo"].as_str().is_some_and(|module| module.contains("stablehlo.sine")));
    }

    #[test]
    fn test_lowered_program_compilation_key_includes_runtime_requirements() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let array_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let lower = |requirements| {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(array_type.clone().into());
            let operation = CustomCallOperation::new("ryft.test.runtime", vec![array_type.clone()])
                .with_runtime_requirements(requirements)
                .unwrap();
            let output = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
            let program = builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap();
            session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap()
        };
        let requirement = RuntimeRequirement::new("test.runtime", 1).unwrap();
        let first = lower(vec![requirement.clone()]);
        let duplicate = lower(vec![requirement.clone(), requirement]);
        let changed_abi = lower(vec![RuntimeRequirement::new("test.runtime", 2).unwrap()]);
        let changed_identifier = lower(vec![RuntimeRequirement::new("test.other", 1).unwrap()]);

        assert_eq!(first.stable_hlo(), changed_abi.stable_hlo());
        assert_eq!(first.stable_hlo(), changed_identifier.stable_hlo());
        assert_eq!(session.compilation_key(&first).unwrap(), session.compilation_key(&duplicate).unwrap());
        assert_ne!(session.compilation_key(&first).unwrap(), session.compilation_key(&changed_abi).unwrap());
        assert_ne!(session.compilation_key(&first).unwrap(), session.compilation_key(&changed_identifier).unwrap());
    }

    #[test]
    fn test_lowered_program_compilation_key_includes_committed_outputs() {
        let client = cpu_client(2);
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![device]).unwrap();
        let vector_type = ArrayType::new_static(DataType::F32, [4]);
        let memory_placement = |x: CompileTracer| x.place(&Memory::Device, None).unwrap();

        // A placement that only names the memory where a device array already is lowers to nothing, so the function
        // that requests it lowers to the module of the identity, but it commits every result of that function.
        let session = Session::builder().client(client).build().unwrap();
        let lower = |function: fn(CompileTracer) -> CompileTracer| {
            let staged =
                crate::stage::<_, ArrayType, ArrayType>(function, vector_type.clone(), Options::new(mesh.clone()))
                    .unwrap()
                    .into_inner();
            staged.lower(&session).unwrap().lowered_program().clone()
        };
        let placed = lower(memory_placement);
        let identity = lower(|x| x);
        assert_eq!(placed.stable_hlo(), identity.stable_hlo());
        assert_eq!((placed.committed_outputs(), identity.committed_outputs()), (&[true][..], &[false][..]));
        assert!(session.compilation_key(&placed).unwrap() != session.compilation_key(&identity).unwrap());

        // An eager call therefore commits the results of each function as its direct execution does, whichever of the
        // two functions the compilation cache of the session sees first.
        for placed_first in [false, true] {
            let session = Session::builder().client(client).build().unwrap();
            let compile =
                |function: fn(CompileTracer) -> CompileTracer| -> crate::CompiledFunction<ArrayType, ArrayType> {
                    crate::compile(function, vector_type.clone(), &session, mesh.clone()).unwrap()
                };
            let input = Array::from_host_buffer(&session, vector_type.clone(), None, None, values_to_bytes(&[1f32; 4]))
                .unwrap();
            let functions = if placed_first {
                [(true, compile(memory_placement)), (false, compile(|x| x))]
            } else {
                [(false, compile(|x| x)), (true, compile(memory_placement))]
            };
            for (committed, function) in functions {
                assert_eq!(function.executable().execute(&session, input.clone()).unwrap().is_committed(), committed);
                assert_eq!(function.call(input.clone()).unwrap().is_committed(), committed);
            }
        }
    }

    #[test]
    fn test_eager_dispatch_query_compares_and_hashes_like_its_key() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let operation = Operation::Array(AddOperation::new().into());
        let inputs = [
            Value::Array(f32_vector(&session, &mesh, &[1.0, 2.0])),
            Value::Dimension(Dimension::new(DimensionValue::constant(2).unwrap(), session.clone())),
        ];
        let query = EagerDispatchQuery { operation: &operation, regions: &[], inputs: &inputs, mesh: Some(&mesh) };
        let key = EagerDispatchKey {
            operation: operation.clone(),
            inputs: (0..inputs.len()).map(|index| query.input(index).into_owned()).collect(),
            mesh: Some(mesh.clone()),
            regions: Arc::new([]),
        };
        let query_hash = hash_of(&query as &dyn EagerDispatchKeyView);

        // Owned keys borrow as views that equal, and hash like, the queries that they were built from.
        let borrowed_key: &dyn EagerDispatchKeyView = std::borrow::Borrow::borrow(&key);
        assert!(borrowed_key == &query as &dyn EagerDispatchKeyView);
        assert_eq!(hash_of(borrowed_key), query_hash);
        assert_eq!(hash_of(&key), query_hash);

        // Every component is compared by value, with dimensions compared by extent.
        let other_inputs = [
            inputs[0].clone(),
            Value::Dimension(Dimension::new(DimensionValue::constant(3).unwrap(), session.clone())),
        ];
        let other_inputs_query = EagerDispatchQuery { inputs: &other_inputs, ..query };
        assert!(borrowed_key != &other_inputs_query as &dyn EagerDispatchKeyView);
        let meshless_query = EagerDispatchQuery { mesh: None, ..query };
        assert!(borrowed_key != &meshless_query as &dyn EagerDispatchKeyView);
    }

    #[test]
    fn test_session_dispatch_signature() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let input = Value::Array(
            Array::from_host_buffer(
                &session,
                ArrayType::new_static(DataType::F32, [3]),
                mesh.clone(),
                None,
                values_to_bytes(&[1f32, 2.0, 3.0]),
            )
            .unwrap(),
        );
        let input_types = vec![input.r#type().into_owned()];
        let options = Options::new(mesh).with_input_bound_bucketing(InputBoundBucketing::PowerOfTwo);

        let (key, staged_types, specialized_options) =
            Session::dispatch_signature(DispatchInputs::Call(vec![(input_types[0].clone(), None)]), &options).unwrap();
        assert_eq!(key, DispatchKey::new(DispatchKeyKind::Exact(input_types.clone().into())));
        assert_eq!(staged_types.as_ref(), input_types.as_slice());
        assert_eq!(specialized_options, None);

        let call_signature =
            Session::dispatch_signature(DispatchInputs::Call(vec![(input_types[0].clone(), Some(&input))]), &options)
                .unwrap();
        assert_eq!(call_signature, (key, staged_types, specialized_options));

        // A missing value leaves the first input's declaration in place. Only the second input is relocatable.
        let array = program_array(&input);
        let declared = array.r#type().into_owned();
        let normalized = declared.clone().with_sharding(None).unwrap();
        assert_ne!(normalized, declared);
        let relocated = Value::Array(array.clone().with_committed(false));
        let (_, staged_types, _) = Session::dispatch_signature(
            DispatchInputs::Call(vec![
                (Type::Array(declared.clone()), None),
                (Type::Array(declared.clone()), Some(&relocated)),
            ]),
            &options,
        )
        .unwrap();
        assert_eq!(staged_types.as_ref(), &[Type::Array(declared), Type::Array(normalized)]);

        let (key, executable_types, specialized_options) =
            Session::dispatch_signature(DispatchInputs::Values(vec![(input_types[0].clone(), &input)]), &options)
                .unwrap();
        assert_eq!(
            key,
            normalize_dispatch_signature(vec![(input_types[0].clone(), ())], &options, options.input_bound_bucketing())
                .unwrap()
                .0,
        );
        let [Type::Array(array_type)] = executable_types.as_ref() else {
            panic!("the executable signature must contain one array");
        };
        let [CoreDimension::Dynamic(extent)] = array_type.shape().dimensions() else {
            panic!("runtime values must select input-bound bucketing");
        };
        assert_eq!(extent.name(), "input_0_axis_0");
        assert_eq!(extent.bounds(), DimensionBounds::new(0, Some(5)).unwrap());
        assert_eq!(specialized_options, None);
    }

    #[test]
    fn test_session_dispatch_signature_runtime_placement() {
        let client = cpu_client(2);
        let mesh = domain_mesh(client, "x", 2);
        let session = Session::builder().client(client).build().unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let tiled = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated_input = Value::Array(
            Array::from_host_buffer(
                &session,
                ArrayType::new_static(DataType::F32, [4]),
                mesh.clone(),
                Some(replicated.clone()),
                values_to_bytes(&[1f32, 2.0, 3.0, 4.0]),
            )
            .unwrap(),
        );
        let tiled_input = Value::Array(
            Array::from_host_buffer(
                &session,
                ArrayType::new_static(DataType::F32, [4]),
                mesh.clone(),
                Some(tiled.clone()),
                values_to_bytes(&[1f32, 2.0, 3.0, 4.0]),
            )
            .unwrap(),
        );
        assert_eq!(replicated_input.r#type(), tiled_input.r#type());
        let input_type = replicated_input.r#type().into_owned();
        let options = Options::new(mesh);
        let (replicated_key, replicated_types, replicated_options) = Session::dispatch_signature(
            DispatchInputs::Values(vec![(input_type.clone(), &replicated_input)]),
            &options,
        )
        .unwrap();
        let (tiled_key, tiled_types, tiled_options) =
            Session::dispatch_signature(DispatchInputs::Values(vec![(input_type.clone(), &tiled_input)]), &options)
                .unwrap();
        assert_ne!(replicated_key, tiled_key);
        assert_eq!(replicated_types, tiled_types);
        assert_eq!(replicated_options, None);
        assert_eq!(tiled_options, Some(options.clone().with_input_placements(vec![Some(tiled.clone())].into())),);

        let (_, _, mixed_options) = Session::dispatch_signature(
            DispatchInputs::Values(vec![
                (input_type.clone(), &replicated_input),
                (input_type.clone(), &tiled_input),
                (input_type.clone(), &replicated_input),
            ]),
            &options,
        )
        .unwrap();
        assert_eq!(mixed_options, Some(options.clone().with_input_placements(vec![None, Some(tiled), None].into())));

        let call_signature =
            Session::dispatch_signature(DispatchInputs::Call(vec![(input_type.clone(), Some(&tiled_input))]), &options)
                .unwrap();
        assert_eq!(call_signature, (replicated_key.clone(), replicated_types, None));

        // A declared sharding determines placement instead of the actual placement of either input.
        let options = options.with_input_shardings(vec![replicated]);
        let replicated_signature = Session::dispatch_signature(
            DispatchInputs::Values(vec![(input_type.clone(), &replicated_input)]),
            &options,
        )
        .unwrap();
        let tiled_signature =
            Session::dispatch_signature(DispatchInputs::Values(vec![(input_type, &tiled_input)]), &options).unwrap();
        assert_eq!(replicated_signature, tiled_signature);
        assert_eq!(replicated_signature.0, replicated_key);
        assert_eq!(replicated_signature.2, None);

        let unsharded = program_array(&tiled_input).r#type().into_owned().with_sharding(None).unwrap();
        let call_signature = Session::dispatch_signature(
            DispatchInputs::Call(vec![(Type::Array(unsharded), Some(&tiled_input))]),
            &options,
        )
        .unwrap();
        assert_eq!(call_signature, tiled_signature);
    }

    #[test]
    fn test_session_dispatch_signature_reuses_one_specialization_for_data_derived_extents() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let scalar_i64 = replicated_scalar_type(&mesh, DataType::I64);
        let scalar_f32 = replicated_scalar_type(&mesh, DataType::F32);
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let function: CompiledFunctionDispatcher<Session, (), Vec<Type>, Type, _> = session
            .jit_with_options(move |(), inputs| trace_data_derived(extent.clone(), inputs), Options::new(mesh.clone()));

        let call = |size: i64| {
            let size = Array::from_host_buffer(
                &session,
                scalar_i64.clone(),
                mesh.clone(),
                None,
                size.to_ne_bytes().as_slice(),
            )
            .unwrap();
            let value = Array::from_host_buffer(
                &session,
                scalar_f32.clone(),
                mesh.clone(),
                None,
                2.0f32.to_ne_bytes().as_slice(),
            )
            .unwrap();
            function.execute((), vec![Value::Array(size), Value::Array(value)]).unwrap()
        };

        assert_eq!(read_first_shard::<f32>(client, program_array(&call(2))), vec![4.0]);
        assert_eq!(read_first_shard::<f32>(client, program_array(&call(4))), vec![8.0]);
        let statistics = function.statistics();
        assert_eq!(statistics.dispatch_misses, 1);
        assert_eq!(statistics.dispatch_hits, 1);
        assert_eq!(statistics.traces, 1);
        assert_eq!(statistics.lowerings, 1);
        assert_eq!(statistics.compilation_requests, 1);
        assert_eq!(function.executable_specialization_count(), 1);
        assert_eq!(session.cached_program_count(), 1);
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_session_dispatch_signature_reuses_one_specialization_for_bounded_dynamic_programs_on_cuda() {
        let client = Platform::Cuda
            .client(Some(ClientOptions::GPU(GpuClientOptions {
                platform: Some(GpuPlatform::CUDA),
                allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                ..Default::default()
            })))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::builder().client(&client).build().unwrap().with_device_mesh(mesh.clone());
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(8)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![extent.into()]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let staged = crate::stage::<_, ArrayType, ArrayType>(
            |input| input.clone() + input,
            dynamic_type,
            Options::new(mesh.clone()),
        )
        .unwrap()
        .into_inner();
        let compiled: CompiledFunction<Session, Type, Type> =
            staged.lower(&session).unwrap().compile(&session).unwrap();
        assert_eq!(session.cached_program_count(), 1);

        for size in [4usize, 7] {
            let input_type = replicated_vector_type(&mesh, size);
            let values = (0..size).map(|value| value as f32).collect::<Vec<_>>();
            let input = Array::from_host_buffer(
                &session,
                input_type,
                mesh.clone(),
                None,
                values_to_bytes(values.as_slice()).as_slice(),
            )
            .unwrap();
            let mut programs_after_first_launch = None;
            for _ in 0..2 {
                let output = compiled.executable().execute(&session, Value::Array(input.clone())).unwrap();
                let Value::Array(output) = output else {
                    panic!("array-only compiled function returned a first-class dimension");
                };
                output.block_until_ready().unwrap();
                assert_eq!(output.shape().as_slice(), &[size]);
                assert_eq!(
                    read_first_shard::<f32>(&client, &output),
                    values.iter().map(|value| value * 2.0).collect::<Vec<_>>()
                );
                // Materializing the input compiles a padding program for each concrete size. The repeat launch
                // reuses both that padding program and the already loaded bounded executable.
                let program_count = session.cached_program_count();
                let expected_count = programs_after_first_launch.get_or_insert(program_count);
                assert_eq!(program_count, *expected_count);
            }
        }

        // The same plugin route also accepts an extent born from device data inside the program. The gateway's
        // ordered bounds assertion and the reduction's dynamic masking both survive compilation, and the scalar input
        // values select different logical extents without producing another retained specialization.
        let scalar_i64 = replicated_scalar_type(&mesh, DataType::I64);
        let scalar_f32 = replicated_scalar_type(&mesh, DataType::F32);
        let extent = DimensionVariable::new("data_extent", DimensionBounds::new(1, Some(8)).unwrap());
        let function: CompiledFunctionDispatcher<Session, (), Vec<Type>, Type, _> = session
            .jit_with_options(move |(), inputs| trace_data_derived(extent.clone(), inputs), Options::new(mesh.clone()));
        let call = |size: i64| {
            let size = Array::from_host_buffer(
                &session,
                scalar_i64.clone(),
                mesh.clone(),
                None,
                size.to_ne_bytes().as_slice(),
            )
            .unwrap();
            let value = Array::from_host_buffer(
                &session,
                scalar_f32.clone(),
                mesh.clone(),
                None,
                2.0f32.to_ne_bytes().as_slice(),
            )
            .unwrap();
            function.execute((), vec![Value::Array(size), Value::Array(value)])
        };
        assert_eq!(read_first_shard::<f32>(&client, program_array(&call(4).unwrap())), vec![8.0]);
        assert_eq!(read_first_shard::<f32>(&client, program_array(&call(7).unwrap())), vec![14.0]);
        let invalid = call(8).unwrap();
        let error = program_array(&invalid).block_until_ready().unwrap_err();
        assert!(
            error.to_string().contains(
                "`dimension_from_scalar` failed: input dimension `data_extent` = 8 is outside its declared bounds \
                 [1, 8)",
            ),
            "{error}",
        );
        assert_eq!(function.executable_specialization_count(), 1);
        assert_eq!(function.statistics().dispatch_misses, 1);
        assert_eq!(function.statistics().dispatch_hits, 2);
    }

    #[test]
    fn test_canonical_boundary_types() {
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

    #[test]
    fn test_canonical_compilation_options_bytes() {
        let mut left = CompilationOptions::default();
        left.environment_option_overrides.insert("z".into(), OptionOverride::default());
        left.environment_option_overrides.insert("a".into(), OptionOverride::default());
        left.executable_build_options = Some(ExecutableCompilationOptions {
            debug_options: Some(DebugOptions {
                xla_backend_extra_options: [("z".into(), "2".into()), ("a".into(), "1".into())].into_iter().collect(),
                ..DebugOptions::default()
            }),
            ..ExecutableCompilationOptions::default()
        });
        let mut right = CompilationOptions::default();
        right.environment_option_overrides.insert("a".into(), OptionOverride::default());
        right.environment_option_overrides.insert("z".into(), OptionOverride::default());
        right.executable_build_options = Some(ExecutableCompilationOptions {
            debug_options: Some(DebugOptions {
                xla_backend_extra_options: [("a".into(), "1".into()), ("z".into(), "2".into())].into_iter().collect(),
                ..DebugOptions::default()
            }),
            ..ExecutableCompilationOptions::default()
        });

        assert_eq!(left, right);
        assert_eq!(canonical_compilation_options_bytes(&left), canonical_compilation_options_bytes(&right));
    }

    #[test]
    fn test_normalize_dispatch_signature_reuses_power_of_two_specializations() {
        fn trace(input: CompilationTracer<Session>) -> Result<CompilationTracer<Session>, Error> {
            let context = input.context().clone();
            Ok(context.bind(AddOperation::new(), Vec::new(), &[input.clone(), input])?.remove(0))
        }

        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let function: CompiledFunctionDispatcher<Session, (), Type, Type, _> = session.jit_with_options(
            |(), input| trace(input),
            Options::new(mesh.clone()).with_input_bound_bucketing(InputBoundBucketing::PowerOfTwo),
        );
        let call = |values: &[f32]| {
            let input = f32_vector(&session, &mesh, values);
            function.execute((), Value::Array(input)).unwrap()
        };

        let three = call(&[1.0, 2.0, 3.0]);
        assert_eq!(program_array(&three).shape(), StaticShape::new(vec![3]));
        assert_eq!(read_first_shard::<f32>(client, program_array(&three)), vec![2.0, 4.0, 6.0]);
        let four = call(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(program_array(&four).shape(), StaticShape::new(vec![4]));
        assert_eq!(read_first_shard::<f32>(client, program_array(&four)), vec![2.0, 4.0, 6.0, 8.0]);
        let five = call(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(program_array(&five).shape(), StaticShape::new(vec![5]));
        assert_eq!(read_first_shard::<f32>(client, program_array(&five)), vec![2.0, 4.0, 6.0, 8.0, 10.0]);

        let statistics = function.statistics();
        assert_eq!(statistics.dispatch_hits, 1);
        assert_eq!(statistics.dispatch_misses, 2);
        assert_eq!(statistics.traces, 2);
        assert_eq!(statistics.lowerings, 2);
        assert_eq!(statistics.compilation_requests, 2);
        assert_eq!(function.executable_specialization_count(), 2);

        // Besides the two specializations, the session caches one padding program for each input shape that is below
        // its bound, which here are the shapes `[3]` and `[5]`.
        assert_eq!(session.cached_program_count(), 4);
    }

    #[test]
    fn test_normalize_dispatch_signature_rejects_unsupported_signatures() {
        let single_device_client = cpu_client(1);
        let single_device_mesh = domain_mesh(single_device_client, "x", 1);
        let single_device_session = Session::builder()
            .client(single_device_client)
            .build()
            .unwrap()
            .with_device_mesh(single_device_mesh.clone());
        let bucketing = Some(InputBoundBucketing::PowerOfTwo);
        let bucketed_options =
            Options::new(single_device_mesh.clone()).with_input_bound_bucketing(InputBoundBucketing::PowerOfTwo);
        let dimension_type = DimensionType::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        assert!(matches!(
            normalize_dispatch_signature(vec![(dimension_type.clone().into(), ())], &bucketed_options, bucketing),
            Err(Error::Compilation(CompilationError::InvalidOptions { reason }))
                if reason == format!(
                    "input-bound bucketing only supports array inputs, but input 0 has first-class dimension type \
                     `{dimension_type}`",
                ),
        ));
        // Exact dispatch admits structural reference types for the stateful call surface that callers opt into,
        // while bucketed dispatch rejects them because holder state is not part of the array-only bucketing model.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let reference_input_types: Arc<[Type]> = vec![Type::Reference(reference_type.clone())].into();
        assert_eq!(
            normalize_dispatch_signature(
                vec![(Type::Reference(reference_type.clone()), ())],
                &Options::new(single_device_mesh.clone()),
                None
            )
            .unwrap(),
            (DispatchKey::new(DispatchKeyKind::Exact(reference_input_types.clone())), reference_input_types,),
        );
        assert!(matches!(
            normalize_dispatch_signature(
                vec![(Type::Reference(reference_type.clone()), ())],
                &bucketed_options,
                bucketing,
            ),
            Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason }))
                if reason == "external reference input 0 does not support input-bound bucketing",
        ));
        assert!(matches!(
            single_device_session.lower_program(
                &external_reference_read_program(ArrayType::scalar(DataType::F32)),
                1,
                &Options::new(single_device_mesh.clone())
                    .with_input_bound_bucketing(InputBoundBucketing::PowerOfTwo),
            ),
            Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason }))
                if reason == "external reference inputs do not support input-bound bucketing",
        ));

        let multi_device_client = cpu_client(2);
        let multi_device_options = Options::new(domain_mesh(multi_device_client, "x", 2))
            .with_input_bound_bucketing(InputBoundBucketing::PowerOfTwo);
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(3)]));
        assert!(matches!(
            normalize_dispatch_signature(vec![(vector_type.into(), ())], &multi_device_options, bucketing),
            Err(Error::Compilation(CompilationError::InvalidOptions { reason }))
                if reason == "input-bound bucketing is unsupported on multi-device meshes because bucketed dynamic \
                              signatures require disabling the Shardy partitioner",
        ));

        // Multi-device bucketing is rejected before inspecting individual input kinds.
        assert!(matches!(
            normalize_dispatch_signature(
                vec![(Type::Reference(reference_type.clone()), ())],
                &multi_device_options,
                bucketing,
            ),
            Err(Error::Compilation(CompilationError::InvalidOptions { reason }))
                if reason == "input-bound bucketing is unsupported on multi-device meshes because bucketed dynamic \
                              signatures require disabling the Shardy partitioner",
        ));
    }

    #[test]
    fn test_normalize_dispatch_signature_numbers_variables_by_first_occurrence() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let options = Options::new(mesh).with_input_bound_bucketing(InputBoundBucketing::PowerOfTwo);
        let bounds = DimensionBounds::new(0, Some(5)).unwrap();
        let vector_type =
            |dimension: CoreDimension| Type::from(ArrayType::new(DataType::F32, Shape::new(vec![dimension])));
        let key = |input_types: Vec<Type>| {
            let inputs = input_types.into_iter().map(|r#type| (r#type, ())).collect();
            normalize_dispatch_signature(inputs, &options, options.input_bound_bucketing()).unwrap().0
        };

        // Two runtime signatures landing in the same power-of-two bucket share one key, and the effective staged
        // types carry the generated variable name of each axis, which the key itself excludes.
        let (three_key, three_types) = normalize_dispatch_signature(
            vec![(vector_type(CoreDimension::Static(3)), ())],
            &options,
            options.input_bound_bucketing(),
        )
        .unwrap();
        assert_eq!(three_key, key(vec![vector_type(CoreDimension::Static(4))]));
        assert_ne!(three_key, key(vec![vector_type(CoreDimension::Static(5))]));
        let [Type::Array(three_type)] = three_types.as_ref() else {
            panic!("bucketing must stage exactly one array input type");
        };
        let [CoreDimension::Dynamic(variable)] = three_type.shape().dimensions() else {
            panic!("bucketing must stage a rank-one bounded dynamic axis");
        };
        assert_eq!(variable.name(), "input_0_axis_0");
        assert_eq!(variable.bounds(), bounds);

        // Signatures requested for staging preserve the caller's types without executable dispatch bucketing.
        let (_, staged_types, _) = Session::dispatch_signature(
            DispatchInputs::Call(vec![(vector_type(CoreDimension::Static(3)), None)]),
            &options,
        )
        .unwrap();
        assert_eq!(staged_types.as_ref(), &[vector_type(CoreDimension::Static(3))]);

        // Dimension variables are keyed by their bounds and by the pattern in which inputs share them, numbered by
        // first occurrence, not by their names, so renaming a variable preserves the key while resharing it across
        // inputs does not.
        let first = DimensionVariable::new("first", bounds);
        let second = DimensionVariable::new("second", bounds);
        let renamed = DimensionVariable::new("renamed", bounds);
        assert_eq!(
            key(vec![vector_type(first.clone().into()), vector_type(first.clone().into())]),
            key(vec![vector_type(renamed.clone().into()), vector_type(renamed.into())]),
        );
        assert_ne!(
            key(vec![vector_type(first.clone().into()), vector_type(first.clone().into())]),
            key(vec![vector_type(first.clone().into()), vector_type(second.into())]),
        );

        // Everything that the carrier without a shape retains still discriminates, which is the element data type and
        // here also the rank.
        assert_ne!(
            key(vec![vector_type(first.clone().into())]),
            key(vec![ArrayType::new(DataType::F64, Shape::new(vec![first.clone().into()])).into()]),
        );
        assert_ne!(
            key(vec![vector_type(first.clone().into())]),
            key(vec![ArrayType::new(DataType::F32, Shape::new(vec![first.clone().into(), first.into()])).into()]),
        );
    }
}
