//! Dispatch and cache keys of the XLA [`Domain`](crate::Domain).
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
use serde::Serialize;

use ryft_core::{
    ArrayIrType, ArrayType, DeviceMesh, Dimension as CoreDimension, DimensionBounds, DimensionValue, DimensionVariable,
    DispatchInputs, ProgramError, RegionKey, Shape, Sharding, Typed,
};
use ryft_pjrt::protos::CompilationOptions;

use crate::compilation::CompilationError;
use crate::compilation::domains::apply_input_shardings;
use crate::compilation::options::{InputBoundBucketing, Options};
use crate::compilation::persistence::{
    PersistentArraySignature, PersistentBoundaryDimension, PersistentCanonicalArraySignature, PersistentDeviceMesh,
    PersistentReferenceState, PersistentSharding, persistent_input_dimensions, persistent_mapping,
    persistent_output_dimensions, persistent_reference_states,
};
use crate::compilation::programs::LoweredProgram;
use crate::domains::RuntimeRequirement;
use crate::{Constant, Error, Operation, Value};

// TODO(eaplatanios): Review this module.

/// Build identity shared by lowering, cache keys, and executable compatibility validation.
pub(super) static COMPILER_IDENTITY: LazyLock<String> = LazyLock::new(|| {
    format!(
        "ryft-xla/{}/openxla/{}/jax/{}",
        env!("CARGO_PKG_VERSION"),
        ryft_xla_sys::XLA_COMMIT,
        ryft_xla_sys::JAX_COMMIT,
    )
});

/// Opaque retained-JIT specialization key derived by [`Domain`](crate::Domain).
///
/// Exact dispatch preserves nominal [`DimensionVariable`] identity and shares its type slice with the effective staged
/// signature, so keying a call costs no deep type clone. Dispatch with input-bound bucketing instead uses a structural
/// physical-bound signature with dimension variables numbered by first occurrence, so calls routed to the same bucket
/// share one staged and compiled specialization. Bucketed keys exclude diagnostic names, which survive only in the
/// effective staged types.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DispatchKey {
    /// Signature component of this key.
    kind: DispatchKeyKind,

    /// Runtime placements of the public inputs of an executable specialization whose inputs are placed over auto mesh
    /// axes, which their types never record (refer to [`Options::input_placements`]), or [`None`] when no input is.
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
/// Unlike the former call-site fingerprint, this key carries the complete StableHLO computation and every
/// execution-relevant field retained by [`LoweredProgram`]. Equality therefore means that cached compiled
/// artifacts are interchangeable, even when unrelated call sites happen to share the same input signature.
///
/// Dynamic dimensions are numbered by first occurrence, so diagnostic variable names do not affect identity.
/// Compilation options, shardings, target facts, runtime requirements, and effect metadata remain part of the key.
/// This complete compilation identity is distinct from a [`DispatchKey`], which selects a retained specialization.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CompilationKey {
    /// Complete canonical compilation contract, compared and hashed without reducing it to a fingerprint.
    pub(super) canonical_bytes: Arc<[u8]>,
}

impl LoweredProgram {
    /// Encodes the complete compilation contract as a deterministic cache key.
    pub(super) fn compilation_key(&self) -> Result<CompilationKey, Error> {
        let compilation_options = canonical_compilation_options_bytes(&self.compilation_options);
        let key = PersistentKey {
            schema_version: PERSISTENT_KEY_SCHEMA_VERSION,
            stable_hlo: &self.stable_hlo,
            compilation_options: compilation_options.as_slice(),
            signature: PersistentArraySignature::encode(&self.input_types, &self.output_types).into_canonical(),
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
                .map(PersistentSharding::from)
                .collect(),
            output_shardings: self.output_shardings.iter().map(PersistentSharding::from).collect(),
            mesh: PersistentDeviceMesh::from(&self.mesh),
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

/// Key of the eager dispatch cache of a [`Session`](crate::Session), which maps one eager operation application
/// directly to its compiled program so that a repeated application skips tracing, lowering, and compilation-key
/// construction, like JAX's per-primitive `(primitive, params)` cache in front of its dispatch path.
///
/// The key must capture everything that the eager lowering reads besides the session itself (whose client, and thus
/// [`Target`](crate::Target), is fixed): the operation including all of its attributes, the contents of every attached region
/// (for operations such as `condition`, `while`, `scan`, `jit_call`, or `shard_map`), the physical type of every array
/// input (shardings included), the extent of every dimension input (because eager binding bakes dimension inputs into
/// the traced program as constants), the device mesh that the lowering compiles against, and the domain's compilation
/// options. Attached regions are keyed by their contents (refer to [`RegionKey`]), so applications whose closures are
/// traced afresh on every call still hit the cache when they trace to the same regions. Instruction provenance is not
/// part of a [`RegionKey`], so applications that differ only in provenance share one compiled program, whose debug
/// locations are those of the first application.
///
/// Keys are compared and hashed through [`EagerDispatchKeyView`], which they share with the borrowed
/// [`EagerDispatchQuery`] that eager dispatch uses to look up hits. Cache hits skip tracing and lowering while
/// execution still uses the invoking domain's effect scope and runtime resources.
#[derive(Clone)]
pub(crate) struct EagerDispatchKey {
    /// Applied operation, compared and hashed with all of its attributes.
    operation: Operation,

    /// Keys of the regions attached to the application, in operation-defined order.
    regions: Arc<[RegionKey<Constant, Operation>]>,

    /// Inputs of the application, in order.
    inputs: Arc<[EagerDispatchInput]>,

    /// Mesh that eager lowering compiles against when it is not derived from the operation alone: the domain mesh if
    /// it has one, and otherwise the mesh of the first array input. It is [`None`] only for an input-free operation of
    /// a domain without a mesh, whose mesh is then determined by the operation and the session's client.
    mesh: Option<DeviceMesh>,

    /// Compilation options of the dispatching domain, compared and hashed by the address of their shared allocation.
    /// Domains created through [`Session::domain`](crate::Session::domain) share one default allocation, so they
    /// share eager dispatch cache entries, while each
    /// [`Domain::with_compilation_options`](crate::Domain::with_compilation_options) call creates a distinct identity.
    /// Options that are equal but separately allocated therefore only miss each other's entries.
    compilation_options: Arc<CompilationOptions>,
}

impl EagerDispatchKey {
    /// Creates the owned key of one eager operation application.
    pub(crate) fn new(
        operation: Operation,
        regions: Arc<[RegionKey<Constant, Operation>]>,
        inputs: Arc<[EagerDispatchInput]>,
        mesh: Option<DeviceMesh>,
        compilation_options: Arc<CompilationOptions>,
    ) -> Self {
        Self { operation, regions, inputs, mesh, compilation_options }
    }
}

/// One input of an [`EagerDispatchKey`].
#[derive(Clone)]
pub(crate) enum EagerDispatchInput {
    /// Array input, keyed by its physical [`ArrayType`] (including its sharding) and by its runtime placement, which
    /// also covers the auto mesh axes that its type never records.
    Array(ArrayType, Sharding),

    /// Dimension input, keyed by its extent, which is how [`DimensionValue`]s compare.
    Dimension(DimensionValue),
}

/// Borrowed form of one [`EagerDispatchInput`], which owned keys and borrowed queries compare and hash alike.
#[derive(PartialEq, Eq, Hash)]
pub(crate) enum EagerDispatchInputRef<'o> {
    /// Array input, keyed by its physical [`ArrayType`] (including its sharding) and by its runtime placement.
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

    /// Returns the keys of the regions attached to the application, in operation-defined order.
    fn regions(&self) -> &[RegionKey<Constant, Operation>];

    /// Returns the number of inputs of the application.
    fn input_count(&self) -> usize;

    /// Returns the input at `index`, which must be smaller than [`Self::input_count`].
    fn input(&self, index: usize) -> EagerDispatchInputRef<'_>;

    /// Returns the mesh that the eager lowering compiles against, if it is not derived from the operation alone.
    fn mesh(&self) -> Option<&DeviceMesh>;

    /// Returns the compilation options of the dispatching domain, which are compared by identity.
    fn compilation_options(&self) -> &Arc<CompilationOptions>;
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

    fn compilation_options(&self) -> &Arc<CompilationOptions> {
        &self.compilation_options
    }
}

impl PartialEq for dyn EagerDispatchKeyView + '_ {
    fn eq(&self, other: &Self) -> bool {
        self.operation() == other.operation()
            && self.regions() == other.regions()
            && self.input_count() == other.input_count()
            && (0..self.input_count()).all(|index| self.input(index) == other.input(index))
            && self.mesh() == other.mesh()
            && Arc::ptr_eq(self.compilation_options(), other.compilation_options())
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
        Arc::as_ptr(self.compilation_options()).hash(state);
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
pub(crate) struct EagerDispatchQuery<'o, 'c> {
    /// Applied operation.
    operation: &'o Operation,

    /// Keys of the regions attached to the application, in operation-defined order.
    regions: &'o [RegionKey<Constant, Operation>],

    /// Inputs of the application, which contain no references.
    inputs: &'o [Value<'c>],

    /// Mesh that the eager lowering compiles against (refer to [`EagerDispatchKey`] for more information).
    mesh: Option<&'o DeviceMesh>,

    /// Compilation options of the dispatching domain.
    compilation_options: &'o Arc<CompilationOptions>,
}

impl<'o, 'c> EagerDispatchQuery<'o, 'c> {
    /// Creates a query that borrows the arguments of one eager operation application.
    pub(crate) fn new(
        operation: &'o Operation,
        regions: &'o [RegionKey<Constant, Operation>],
        inputs: &'o [Value<'c>],
        mesh: Option<&'o DeviceMesh>,
        compilation_options: &'o Arc<CompilationOptions>,
    ) -> Self {
        Self { operation, regions, inputs, mesh, compilation_options }
    }
}

impl EagerDispatchKeyView for EagerDispatchQuery<'_, '_> {
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
            Value::Reference(_) => unreachable!("reference inputs are rejected at the `eager_bind` entry guard"),
        }
    }

    fn mesh(&self) -> Option<&DeviceMesh> {
        self.mesh
    }

    fn compilation_options(&self) -> &Arc<CompilationOptions> {
        self.compilation_options
    }
}

/// Derives the retained dispatch key, the effective staged input types, and any options specialized on input
/// placements for one dispatch request. This function implements
/// [`CompilationDomain::dispatch_signature`](ryft_core::CompilationDomain::dispatch_signature) for
/// [`Domain`](crate::Domain).
pub(super) fn dispatch_signature<'c>(
    inputs: DispatchInputs<'_, '_, Value<'c>>,
    options: &Options,
) -> Result<(DispatchKey, Arc<[ArrayIrType]>, Option<Options>), Error> {
    let (input_types, values) = match inputs {
        DispatchInputs::Types(input_types) => (input_types, None),
        DispatchInputs::Values { values, types } => (types, Some(values)),
    };
    let input_count = input_types.len();
    // Only independent executable requests apply bucketing. Staged calls retain their caller's signature.
    let bucketing = values.as_ref().and(options.input_bound_bucketing);
    let (key, input_types) = normalize_dispatch_signature(input_types, options, bucketing)?;
    let Some(inputs) = values.filter(|_| options.input_shardings.is_none()) else {
        return Ok((key, input_types, None));
    };

    // Only placements that name auto mesh axes are missing from the input types, and the placement vector is only
    // allocated once such an input appears, so that ordinary calls pay for no allocation.
    let mut placements = None::<Vec<Option<Sharding>>>;
    for (index, input) in inputs.enumerate() {
        match input {
            Value::Array(array) if array.sharding().references_auto_axis() => {
                placements.get_or_insert_with(|| vec![None; index]).push(Some(array.sharding().clone()))
            }
            _ => {
                if let Some(placements) = &mut placements {
                    placements.push(None);
                }
            }
        }
    }
    let Some(mut placements) = placements else {
        return Ok((key, input_types, None));
    };
    placements.resize(input_count, None);
    let placements: Arc<[_]> = placements.into();
    let mut options = options.clone();
    options.input_placements = Some(Arc::clone(&placements));
    Ok((DispatchKey { placements: Some(placements), ..key }, input_types, Some(options)))
}

/// Encodes compilation options deterministically even though Prost represents protobuf maps as randomly seeded
/// `HashMap`s. Map entries are removed from a clone before normal protobuf encoding and appended as length-delimited,
/// path-qualified records sorted by their encoded keys and values.
pub(super) fn canonical_compilation_options_bytes(options: &CompilationOptions) -> Vec<u8> {
    use ryft_pjrt::protos::{AutoTuneReferenceKey, AutoTuneResultKey};

    /// Appends one sorted protobuf-map record with an explicit path and length delimiters.
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

/// Encodes one boundary's input and output types using the persistent signature's canonical dimension-variable
/// numbering.
///
/// [`ArrayType`] equality is dimension-variable identity equality, and two structurally identical lowerings (or one
/// lowering and its cache-restored executable) never share variable identities, so boundary compatibility must be
/// compared through the same identity-free canonical form the compilation cache key uses.
pub(super) fn canonical_boundary_types(input_types: &[ArrayType], output_types: &[ArrayType]) -> Vec<u8> {
    // Serializing the plain persistent data model cannot fail.
    serde_json::to_vec(&PersistentArraySignature::encode(input_types, output_types).into_canonical()).unwrap()
}

/// Internal representation of [`DispatchKey`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum DispatchKeyKind {
    /// Exact runtime abstract signature used when bucketing is disabled, shared with the effective staged types.
    Exact(Arc<[ArrayIrType]>),

    /// Bounded signature using canonical variable indices when bucketing is enabled.
    Bucketed(BucketedDispatchSignature),
}

/// Bounded input signature with dimension variables numbered by first occurrence, used as the retained-dispatch key.
///
/// Dimension variables are numbered by first occurrence across the whole input signature under
/// [`DimensionVariable`] identity, and only their bounds are retained. Two bucketed signatures are therefore equal
/// exactly when they agree on every input's data type, layout, sharding, memory space, rank, per-axis static extents,
/// and per-axis variable-sharing pattern with matching bounds, regardless of how the variables are named.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct BucketedDispatchSignature {
    /// `(lower, upper)` bounds of each distinct dimension variable, in first-occurrence order.
    variables: Vec<(usize, Option<usize>)>,

    /// Bucketed input array types with their shapes rewritten into variable-index patterns.
    input_types: Vec<BucketedArrayTypeKey>,
}

/// One bucketed input array type split into a shape-free carrier and a shape pattern using canonical variable indices.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct BucketedArrayTypeKey {
    /// Array type with an empty shape, retaining data type, layout, sharding, and memory identity.
    carrier: ArrayType,

    /// Per-axis static extents or canonical variable indices, which also fix the array's rank.
    dimensions: Vec<BucketedDimensionKey>,
}

/// One static extent or canonical variable index of a [`BucketedArrayTypeKey`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum BucketedDimensionKey {
    /// Statically known extent. Bucketing rewrites every static input axis into a bounded dynamic axis, so this
    /// variant is unreachable for signatures produced today; it is retained so the normalization stays total over
    /// [`CoreDimension`](ryft_core::Dimension).
    Static(usize),

    /// Index of the axis' dimension variable in [`BucketedDispatchSignature::variables`].
    Variable(usize),
}

impl BucketedDispatchSignature {
    /// Builds a dispatch key from the already bucketed `input_types`, numbering dimension variables by first
    /// occurrence.
    fn new(input_types: &[ArrayType]) -> Self {
        let mut variables = Vec::<DimensionVariable>::new();
        let input_types = input_types
            .iter()
            .map(|array_type| BucketedArrayTypeKey {
                carrier: array_type.clone().with_shape(Shape::new(Vec::new())),
                dimensions: array_type
                    .shape()
                    .dimensions()
                    .iter()
                    .map(|dimension| match dimension {
                        CoreDimension::Static(extent) => BucketedDimensionKey::Static(*extent),
                        CoreDimension::Dynamic(variable) => {
                            let index =
                                variables.iter().position(|existing| existing == variable).unwrap_or_else(|| {
                                    variables.push(variable.clone());
                                    variables.len() - 1
                                });
                            BucketedDimensionKey::Variable(index)
                        }
                    })
                    .collect(),
            })
            .collect();
        let variables =
            variables.iter().map(|variable| (variable.bounds().lower(), variable.bounds().upper())).collect();
        Self { variables, input_types }
    }
}

/// Schema used by canonical compilation keys.
const PERSISTENT_KEY_SCHEMA_VERSION: u32 = 1;

/// Stable cache-key payload covering the complete executable and external-reference ABI.
#[derive(Serialize)]
struct PersistentKey<'o> {
    /// Version of this persistent record schema.
    schema_version: u32,

    /// Exact textual StableHLO/Shardy module submitted to PJRT.
    stable_hlo: &'o str,

    /// Canonical encoding of the complete PJRT compilation options.
    compilation_options: &'o [u8],

    /// Logical array types with shared dynamic dimension variables.
    signature: PersistentCanonicalArraySignature,

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
    expected_argument_shardings: Vec<PersistentSharding>,

    /// Runtime placements of all logical outputs, including the auto mesh axes that output types never record.
    output_shardings: Vec<PersistentSharding>,

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

/// Applies input sharding declarations and optional executable bucketing, without inspecting runtime values.
fn normalize_dispatch_signature(
    input_types: Vec<ArrayIrType>,
    options: &Options,
    bucketing: Option<InputBoundBucketing>,
) -> Result<(DispatchKey, Arc<[ArrayIrType]>), Error> {
    let input_types = apply_input_shardings(input_types, options)?;
    let Some(bucketing) = bucketing else {
        let input_types: Arc<[ArrayIrType]> = input_types.into();
        return Ok((DispatchKey::new(DispatchKeyKind::Exact(input_types.clone())), input_types));
    };
    if options.mesh.devices().len() > 1 {
        return Err(Error::Compilation(CompilationError::InvalidOptions {
            reason: "input-bound bucketing is unsupported on multi-device meshes because bucketed dynamic \
                     signatures require disabling the Shardy partitioner"
                .to_string(),
        }));
    }
    let mut bucketed_array_types = Vec::with_capacity(input_types.len());
    for (input_index, r#type) in input_types.iter().enumerate() {
        let array_type = match r#type {
            ArrayIrType::Array(array_type) => array_type,
            ArrayIrType::Dimension(_) => {
                return Err(Error::Compilation(CompilationError::InvalidOptions {
                    reason: format!(
                        "input-bound bucketing only supports array inputs, but input {} has first-class dimension \
                         type {}",
                        input_index, r#type,
                    ),
                }));
            }
            ArrayIrType::Reference(_) => {
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
                CoreDimension::Static(extent) => {
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
                    Ok(CoreDimension::Dynamic(DimensionVariable::new(
                        format!("input_{input_index}_axis_{axis}"),
                        bounds,
                    )))
                }
                CoreDimension::Dynamic(variable) => Ok(CoreDimension::Dynamic(variable.clone())),
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;
        bucketed_array_types.push(array_type.clone().with_shape(Shape::new(dimensions)));
    }
    let key = BucketedDispatchSignature::new(bucketed_array_types.as_slice());
    let input_types = bucketed_array_types.into_iter().map(ArrayIrType::Array).collect::<Vec<_>>();
    Ok((DispatchKey::new(DispatchKeyKind::Bucketed(key)), input_types.into()))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, ArrayIrType, ArrayType, CompilationCacheDomain, CompilationDomain, CompilationTracer, Compile,
        CompiledFunctionDispatcher, Context, CustomCallOperation, DataType, Dimension as CoreDimension,
        DimensionBounds, DimensionType, DimensionValue, DimensionVariable, Placeholder, ReferenceType, Shape, Sharding,
        ShardingDimension, StaticShape,
    };
    use ryft_pjrt::protos::CompilationOptions;
    use ryft_pjrt::{ClientOptions, CpuClientOptions, load_cpu_plugin};

    use crate::compilation::CompilationError;
    use crate::compilation::options::Options;
    use crate::domains::tests::{domain_mesh, external_reference_read_program, f32_vector, program_array, read_f32s};
    use crate::operations::ProgramBuilder;
    use crate::tests::{execution_client, hash_of, values_to_bytes};
    use crate::{Array, Constant, Dimension, Domain, Error, Operation, Session, Value};

    use super::*;

    #[test]
    fn test_domain_dispatch_signature() {
        let client = execution_client();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let input = Value::Array(
            Array::from_host_buffer(
                &domain,
                ArrayType::new_static(DataType::F32, [3]),
                mesh.clone(),
                values_to_bytes(&[1f32, 2.0, 3.0]),
            )
            .unwrap(),
        );
        let input_types = vec![input.r#type().into_owned()];
        let options = Options::new(mesh).with_input_bound_bucketing(InputBoundBucketing::PowerOfTwo);

        let (key, staged_types, specialized_options) =
            Domain::dispatch_signature(DispatchInputs::Types(input_types.clone()), &options).unwrap();
        assert_eq!(key, DispatchKey::new(DispatchKeyKind::Exact(input_types.clone().into())));
        assert_eq!(staged_types.as_ref(), input_types.as_slice());
        assert_eq!(specialized_options, None);

        let (key, executable_types, specialized_options) = Domain::dispatch_signature(
            DispatchInputs::Values { values: &mut std::iter::once(&input), types: input_types.clone() },
            &options,
        )
        .unwrap();
        assert_eq!(key, normalize_dispatch_signature(input_types, &options, options.input_bound_bucketing).unwrap().0);
        let [ArrayIrType::Array(array_type)] = executable_types.as_ref() else {
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
    fn test_domain_dispatch_signature_runtime_placement() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 2);
        let domain = Session::new(&client).domain();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let tiled = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated_input = Value::Array(
            Array::from_host_buffer_with_sharding(
                &domain,
                ArrayType::new_static(DataType::F32, [4]),
                replicated.clone(),
                mesh.clone(),
                values_to_bytes(&[1f32, 2.0, 3.0, 4.0]),
            )
            .unwrap(),
        );
        let tiled_input = Value::Array(
            Array::from_host_buffer_with_sharding(
                &domain,
                ArrayType::new_static(DataType::F32, [4]),
                tiled.clone(),
                mesh.clone(),
                values_to_bytes(&[1f32, 2.0, 3.0, 4.0]),
            )
            .unwrap(),
        );
        assert_eq!(replicated_input.r#type(), tiled_input.r#type());
        let input_types = vec![replicated_input.r#type().into_owned()];
        let options = Options::new(mesh);
        let (replicated_key, replicated_types, replicated_options) = Domain::dispatch_signature(
            DispatchInputs::Values { values: &mut std::iter::once(&replicated_input), types: input_types.clone() },
            &options,
        )
        .unwrap();
        let (tiled_key, tiled_types, tiled_options) = Domain::dispatch_signature(
            DispatchInputs::Values { values: &mut std::iter::once(&tiled_input), types: input_types.clone() },
            &options,
        )
        .unwrap();
        assert_ne!(replicated_key, tiled_key);
        assert_eq!(replicated_types, tiled_types);
        assert_eq!(replicated_options, None);
        assert_eq!(
            tiled_options,
            Some(Options { input_placements: Some(vec![Some(tiled)].into()), ..options.clone() }),
        );

        // A declared sharding determines placement instead of the actual placement of either input.
        let options = options.with_input_shardings(vec![replicated]);
        let replicated_signature = Domain::dispatch_signature(
            DispatchInputs::Values { values: &mut std::iter::once(&replicated_input), types: input_types.clone() },
            &options,
        )
        .unwrap();
        let tiled_signature = Domain::dispatch_signature(
            DispatchInputs::Values { values: &mut std::iter::once(&tiled_input), types: input_types },
            &options,
        )
        .unwrap();
        assert_eq!(replicated_signature, tiled_signature);
        assert_eq!(replicated_signature.0, replicated_key);
        assert_eq!(replicated_signature.2, None);
    }

    #[test]
    fn test_input_bound_bucketing_reuses_power_of_two_specializations() {
        fn trace<'c>(input: CompilationTracer<Domain<'c>>) -> Result<CompilationTracer<Domain<'c>>, Error> {
            let context = input.context().clone();
            Ok(context.bind(AddOperation::new(), Vec::new(), &[input.clone(), input])?.remove(0))
        }

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let function: CompiledFunctionDispatcher<Domain<'_>, (), ArrayIrType, ArrayIrType, _> = domain
            .jit_with_options(
                |(), input| trace(input),
                Options::new(mesh.clone()).with_input_bound_bucketing(InputBoundBucketing::PowerOfTwo),
            );
        let call = |values: &[f32]| {
            let input = f32_vector(&domain, &mesh, values);
            function.execute((), Value::Array(input)).unwrap()
        };

        let three = call(&[1.0, 2.0, 3.0]);
        assert_eq!(program_array(&three).shape(), StaticShape::new(vec![3]));
        assert_eq!(read_f32s(&client, program_array(&three)), vec![2.0, 4.0, 6.0]);
        let four = call(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(program_array(&four).shape(), StaticShape::new(vec![4]));
        assert_eq!(read_f32s(&client, program_array(&four)), vec![2.0, 4.0, 6.0, 8.0]);
        let five = call(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(program_array(&five).shape(), StaticShape::new(vec![5]));
        assert_eq!(read_f32s(&client, program_array(&five)), vec![2.0, 4.0, 6.0, 8.0, 10.0]);

        let statistics = function.statistics();
        assert_eq!(statistics.dispatch_hits, 1);
        assert_eq!(statistics.dispatch_misses, 2);
        assert_eq!(statistics.traces, 2);
        assert_eq!(statistics.lowerings, 2);
        assert_eq!(statistics.compilation_requests, 2);
        assert_eq!(function.executable_specialization_count(), 2);
        assert_eq!(domain.program_count(), 2);
    }

    #[test]
    fn test_input_bound_bucketing_rejects_unsupported_signatures() {
        let plugin = load_cpu_plugin().unwrap();
        let single_device_client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let single_device_mesh = domain_mesh(&single_device_client, "x", 1);
        let single_device_domain = Session::new(&single_device_client).domain().with_mesh(single_device_mesh.clone());
        let dimension_type = DimensionType::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        assert!(matches!(
            normalize_dispatch_signature(vec![dimension_type.clone().into()], &Options::new(single_device_mesh.clone())
                    .with_input_bound_bucketing(InputBoundBucketing::PowerOfTwo), Some(InputBoundBucketing::PowerOfTwo)),
            Err(Error::Compilation(CompilationError::InvalidOptions { reason }))
                if reason == format!(
                    "input-bound bucketing only supports array inputs, but input 0 has first-class dimension type {}",
                    dimension_type,
                ),
        ));
        // Exact dispatch admits structural reference types for the opt-in stateful call surface, while bucketed
        // dispatch rejects them because holder state is not part of the array-only bucketing model.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let reference_input_types: Arc<[ArrayIrType]> = vec![ArrayIrType::Reference(reference_type.clone())].into();
        assert_eq!(
            normalize_dispatch_signature(
                reference_input_types.to_vec(),
                &Options::new(single_device_mesh.clone()),
                None
            )
            .unwrap(),
            (DispatchKey::new(DispatchKeyKind::Exact(reference_input_types.clone())), reference_input_types,),
        );
        assert!(matches!(
            normalize_dispatch_signature(vec![ArrayIrType::Reference(reference_type.clone())], &Options::new(single_device_mesh.clone())
                    .with_input_bound_bucketing(InputBoundBucketing::PowerOfTwo), Some(InputBoundBucketing::PowerOfTwo)),
            Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason }))
                if reason == "external reference input 0 does not support input-bound bucketing",
        ));
        assert!(matches!(
            single_device_domain.lower_program(
                &external_reference_read_program(ArrayType::scalar(DataType::F32)),
                1,
                &Options::new(single_device_mesh.clone())
                    .with_input_bound_bucketing(InputBoundBucketing::PowerOfTwo),
            ),
            Err(Error::Compilation(CompilationError::UnsupportedReferenceAbi { reason }))
                if reason == "external reference inputs do not support input-bound bucketing",
        ));

        let multi_device_client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let multi_device_mesh = domain_mesh(&multi_device_client, "x", 2);
        assert!(matches!(
            normalize_dispatch_signature(vec![ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(3)])).into()], &Options::new(multi_device_mesh).with_input_bound_bucketing(InputBoundBucketing::PowerOfTwo), Some(InputBoundBucketing::PowerOfTwo)),
            Err(Error::Compilation(CompilationError::InvalidOptions { reason }))
                if reason == "input-bound bucketing is unsupported on multi-device meshes because bucketed dynamic \
                    signatures require disabling the Shardy partitioner",
        ));

        // Multi-device bucketing is rejected before inspecting individual input kinds.
        assert!(matches!(
            normalize_dispatch_signature(vec![ArrayIrType::Reference(reference_type.clone())], &Options::new(domain_mesh(&multi_device_client, "x", 2))
                    .with_input_bound_bucketing(InputBoundBucketing::PowerOfTwo), Some(InputBoundBucketing::PowerOfTwo)),
            Err(Error::Compilation(CompilationError::InvalidOptions { reason }))
                if reason == "input-bound bucketing is unsupported on multi-device meshes because bucketed dynamic \
                    signatures require disabling the Shardy partitioner",
        ));
    }

    #[test]
    fn test_input_bound_bucketing_alpha_normalizes_dispatch_keys() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let options = Options::new(mesh).with_input_bound_bucketing(InputBoundBucketing::PowerOfTwo);
        let bounds = DimensionBounds::new(0, Some(5)).unwrap();
        let vector_type =
            |dimension: CoreDimension| ArrayIrType::from(ArrayType::new(DataType::F32, Shape::new(vec![dimension])));
        let key = |input_types: Vec<ArrayIrType>| {
            normalize_dispatch_signature(input_types, &options, options.input_bound_bucketing).unwrap().0
        };

        // Two runtime signatures landing in the same power-of-two bucket share one key, and the effective staged
        // types carry the generated per-axis variable names that the key itself excludes.
        let (three_key, three_types) = normalize_dispatch_signature(
            vec![vector_type(CoreDimension::Static(3))],
            &options,
            options.input_bound_bucketing,
        )
        .unwrap();
        assert_eq!(three_key, key(vec![vector_type(CoreDimension::Static(4))]));
        assert_ne!(three_key, key(vec![vector_type(CoreDimension::Static(5))]));
        let [ArrayIrType::Array(three_type)] = three_types.as_ref() else {
            panic!("bucketing must stage exactly one array input type");
        };
        let [CoreDimension::Dynamic(variable)] = three_type.shape().dimensions() else {
            panic!("bucketing must stage a rank-one bounded dynamic axis");
        };
        assert_eq!(variable.name(), "input_0_axis_0");
        assert_eq!(variable.bounds(), bounds);

        // Signatures requested for staging preserve the caller's types without executable dispatch bucketing.
        let (_, staged_types, _) =
            Domain::dispatch_signature(DispatchInputs::Types(vec![vector_type(CoreDimension::Static(3))]), &options)
                .unwrap();
        assert_eq!(staged_types.as_ref(), &[vector_type(CoreDimension::Static(3))]);

        // Dimension variables are keyed by their bounds and first-occurrence sharing pattern, not by their names, so
        // renaming a variable preserves the key while resharing it across inputs does not.
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

        // Everything the shape-free carrier retains still discriminates: element data type, and here the rank too.
        assert_ne!(
            key(vec![vector_type(first.clone().into())]),
            key(vec![ArrayType::new(DataType::F64, Shape::new(vec![first.clone().into()])).into()]),
        );
        assert_ne!(
            key(vec![vector_type(first.clone().into())]),
            key(vec![ArrayType::new(DataType::F32, Shape::new(vec![first.clone().into(), first.into()])).into()]),
        );
    }

    #[test]
    fn test_compilation_key_is_canonical_and_stable() {
        use ryft_core::Sin;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let staged = crate::stage::<_, ArrayType, ArrayType>(|x| x.sin().unwrap(), input_type, Options::new(mesh))
            .unwrap()
            .into_inner();
        let compiled: ryft_core::compilation::CompiledFunction<Domain<'_>, ArrayIrType, ArrayIrType> =
            staged.lower(&domain).unwrap().compile(&domain).unwrap();
        let key = domain.compilation_key(compiled.lowered().lowered_program()).unwrap();
        let independent_domain = Session::new(&client).domain();
        let repeated = independent_domain.compilation_key(compiled.lowered().lowered_program()).unwrap();

        assert_eq!(key, repeated);
        assert_eq!(domain.persistent_cache_key(&key), Some(key.canonical_bytes.to_vec()));
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
    fn test_compilation_key_includes_runtime_abi_and_configuration() {
        use crate::domains::{RuntimeRequirement, with_runtime_requirements};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain().with_mesh(mesh.clone());
        let array_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let lower = |requirements| {
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(array_type.clone().into());
            let operation = with_runtime_requirements(
                CustomCallOperation::new("ryft.test.runtime", vec![array_type.clone()]),
                requirements,
            )
            .unwrap();
            let output = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
            let program = builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap();
            domain.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap()
        };
        let requirement =
            |version, configuration| RuntimeRequirement::new("test.runtime", version, configuration).unwrap();
        let first = lower(vec![requirement(1, vec![7])]);
        let duplicate = lower(vec![requirement(1, vec![7]), requirement(1, vec![7])]);
        let changed_abi = lower(vec![requirement(2, vec![7])]);
        let changed_configuration = lower(vec![requirement(1, vec![8])]);

        assert_eq!(first.stable_hlo(), changed_abi.stable_hlo());
        assert_eq!(first.stable_hlo(), changed_configuration.stable_hlo());
        assert_eq!(domain.compilation_key(&first).unwrap(), domain.compilation_key(&duplicate).unwrap());
        assert_ne!(domain.compilation_key(&first).unwrap(), domain.compilation_key(&changed_abi).unwrap());
        assert_ne!(domain.compilation_key(&first).unwrap(), domain.compilation_key(&changed_configuration).unwrap());
    }

    #[test]
    fn test_eager_dispatch_query_compares_and_hashes_like_its_key() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = Session::new(&client).domain();
        let operation = Operation::Array(AddOperation::new().into());
        let inputs = [
            Value::Array(f32_vector(&domain, &mesh, &[1.0, 2.0])),
            Value::Dimension(Dimension::new(DimensionValue::constant(2).unwrap(), domain.clone())),
        ];
        let compilation_options = Arc::new(domain.compilation_options().clone());
        let query = EagerDispatchQuery {
            operation: &operation,
            regions: &[],
            inputs: &inputs,
            mesh: Some(&mesh),
            compilation_options: &compilation_options,
        };
        let key = EagerDispatchKey {
            operation: operation.clone(),
            inputs: (0..inputs.len()).map(|index| query.input(index).into_owned()).collect(),
            mesh: Some(mesh.clone()),
            compilation_options: Arc::clone(&compilation_options),
            regions: Arc::new([]),
        };
        let query_hash = hash_of(&query as &dyn EagerDispatchKeyView);

        // Owned keys borrow as views that equal, and hash like, the queries that they were built from.
        let borrowed_key: &dyn EagerDispatchKeyView = std::borrow::Borrow::borrow(&key);
        assert!(borrowed_key == &query as &dyn EagerDispatchKeyView);
        assert_eq!(hash_of(borrowed_key), query_hash);
        assert_eq!(hash_of(&key), query_hash);

        // Compilation options are compared by identity, and every other component by value (dimensions by extent).
        let other_options = Arc::new(domain.compilation_options().clone());
        let other_options_query = EagerDispatchQuery { compilation_options: &other_options, ..query };
        assert!(borrowed_key != &other_options_query as &dyn EagerDispatchKeyView);
        let other_inputs =
            [inputs[0].clone(), Value::Dimension(Dimension::new(DimensionValue::constant(3).unwrap(), domain.clone()))];
        let other_inputs_query = EagerDispatchQuery { inputs: &other_inputs, ..query };
        assert!(borrowed_key != &other_inputs_query as &dyn EagerDispatchKeyView);
        let meshless_query = EagerDispatchQuery { mesh: None, ..query };
        assert!(borrowed_key != &meshless_query as &dyn EagerDispatchKeyView);
    }

    #[test]
    fn test_canonical_compilation_options_sorts_protobuf_maps() {
        use ryft_pjrt::protos::{DebugOptions, ExecutableCompilationOptions, OptionOverride};

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
