use std::collections::BTreeSet;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock};

use ryft_pjrt::protos::CompilationOptions;
use ryft_pjrt::{Buffer, Client, LoadedExecutable, Program as PjrtProgram};

use ryft_core::compilation::{CompilationContext, CompilationDomain, FunctionFingerprint};
use ryft_core::contexts::Context;
use ryft_core::domains::Domain;
use ryft_core::operations::Operation;
use ryft_core::operations::constants::{ONE_OPERATION_NAME, ZERO_OPERATION_NAME};
use ryft_core::parameters::Parameterized;
use ryft_core::programs::{ProgramError, Value};
use ryft_core::sharding::{DeviceMesh, Sharding};
use ryft_core::tracing::DomainTracer;
use ryft_core::tracing_v2::{DifferentiationContext, DifferentiationError};
use ryft_core::types::{ArrayType, DataType, TypeError, Typed};

use super::ops::{LinearXlaOperation, XlaConstant, XlaOperation, XlaProgram};
use super::shard_map::ShardMapTraceError;
use crate::arrays_v0::ArrayError;
use crate::{Array, Error, ToPjrt};

use crate::arrays_v0::{ShardDescriptor, ShardLayout};
use ryft_core::sharding::DeviceId;
use ryft_core::types::Shape;

/// Error type returned by [`XlaDomain`] orchestration helpers.
#[derive(Debug, thiserror::Error)]
pub enum XlaDomainError {
    /// Error surfaced while lowering a traced XLA program to StableHLO/Shardy MLIR.
    #[error("{0}")]
    Lowering(#[from] ShardMapTraceError),

    /// Error surfaced while materializing or marshalling [`Array`] values.
    #[error("{0}")]
    Array(#[from] ArrayError),

    /// Error surfaced by crate-level XLA helpers.
    #[error("{0}")]
    Xla(#[from] Error),

    /// Error surfaced by the underlying PJRT runtime.
    #[error("{0}")]
    Pjrt(#[from] ryft_pjrt::Error),

    /// Error surfaced by the core tracing pipeline (e.g. a builder error during trace).
    #[error("{0}")]
    Tracing(#[from] ProgramError),

    /// Error surfaced by reverse-mode differentiation (e.g. a non-scalar gradient output).
    #[error("{0}")]
    Differentiation(#[from] DifferentiationError),

    /// Error surfaced when [`compile_with_options`](crate::jit::compile_with_options) is given options
    /// that do not match the traced function's arity or shape — for example a
    /// `donate_argnums` index outside the flat input range, or an `in_shardings` length that
    /// doesn't match the number of flat inputs.
    #[error("invalid compilation options: {reason}")]
    InvalidCompilationOptions { reason: String },
}

/// Stateful backend that materializes, lowers, compiles, and executes traced XLA programs
/// against a live PJRT [`Client`].
///
/// An [`XlaDomain`] bundles four pieces of context:
///
/// - a PJRT [`Client`] used to upload `zero`/`one` shards and to compile and execute programs,
/// - an optional concrete [`DeviceMesh`] used by test-only constant-materialization helpers,
/// - default [`CompilationOptions`] that the compile path forwards to PJRT, and
/// - an internal [`CompilationContext`] that memoizes compiled programs across calls, shared
///   across [`Clone`] of this domain via an [`Arc`].
///
/// The cache lives directly on the domain because the domain is the unit of execution from a
/// user's perspective. Calls that reuse the same [`XlaDomain`] also reuse its cache for repeat
/// compilations. Domain clones share the same underlying cache, so handing a cloned domain to a
/// long-lived [`CompiledXlaFunction`](crate::CompiledXlaFunction)
/// does not duplicate cached compilations.
///
/// The same domain type covers both staged tracing and concrete execution. Nested traced code can borrow
/// [`XlaDomain::token`] when it needs a clientless domain for static staging instead of defining a separate token type.
pub struct XlaDomain<'c> {
    /// PJRT client used by this domain.
    client: Option<&'c Client<'c>>,

    /// Concrete device mesh used by test-only constant-materialization helpers.
    mesh: Option<DeviceMesh>,

    /// Default compilation options forwarded to [`Client::compile`].
    compilation_options: CompilationOptions,

    /// Process-local cache of compiled programs, shared across domain clones via [`Arc`].
    cache: Arc<CompilationContext<XlaDomain<'c>>>,

    /// Phantom marker tying the domain lifetime to the concrete PJRT-backed array value type.
    marker: PhantomData<fn() -> Array<'c>>,
}

/// Tracer shape used while staging XLA programs directly from types.
pub(crate) type XlaTracer<'domain, 'context> = DomainTracer<'domain, XlaDomain<'context>>;

impl<'c> Clone for XlaDomain<'c> {
    fn clone(&self) -> Self {
        Self {
            client: self.client,
            mesh: self.mesh.clone(),
            compilation_options: self.compilation_options.clone(),
            cache: Arc::clone(&self.cache),
            marker: PhantomData,
        }
    }
}

impl<'c> XlaDomain<'c> {
    /// Creates a new [`XlaDomain`] from a PJRT [`Client`] with default [`CompilationOptions`]
    /// and an empty compile cache.
    #[inline]
    pub fn new(client: &'c Client<'c>) -> Self {
        Self::with_compilation_options(client, CompilationOptions::default())
    }

    /// Creates a new [`XlaDomain`] with an explicit [`CompilationOptions`] template.
    #[inline]
    pub fn with_compilation_options(client: &'c Client<'c>, compilation_options: CompilationOptions) -> Self {
        Self {
            client: Some(client),
            mesh: None,
            compilation_options,
            cache: Arc::new(CompilationContext::new()),
            marker: PhantomData,
        }
    }

    /// Creates a new [`XlaDomain`] with an explicit in-memory cache capacity. `capacity` must be
    /// greater than zero; values of zero are silently clamped to one entry.
    #[inline]
    pub fn with_cache_capacity(client: &'c Client<'c>, capacity: usize) -> Self {
        Self {
            client: Some(client),
            mesh: None,
            compilation_options: CompilationOptions::default(),
            cache: Arc::new(CompilationContext::with_capacity(capacity)),
            marker: PhantomData,
        }
    }

    /// Creates a new [`XlaDomain`] whose compile cache also writes through to a
    /// [`DiskCache`](ryft_core::compilation::DiskCache) rooted at `directory`. Returns an
    /// [`std::io::Error`] only when the directory itself can't be opened or created.
    #[inline]
    pub fn with_disk_cache(client: &'c Client<'c>, directory: impl Into<PathBuf>) -> std::io::Result<Self> {
        let cache = CompilationContext::new().with_disk_cache(directory)?;
        Ok(Self {
            client: Some(client),
            mesh: None,
            compilation_options: CompilationOptions::default(),
            cache: Arc::new(cache),
            marker: PhantomData,
        })
    }

    /// Creates a new [`XlaDomain`] whose compile cache also writes through to a
    /// [`DiskCache`](ryft_core::compilation::DiskCache) configured via the
    /// [`DiskCache::ENV_VAR`](ryft_core::compilation::DiskCache::ENV_VAR) environment variable,
    /// if it is set. Falls back to an in-memory-only cache when the variable is absent or
    /// unparseable.
    #[inline]
    pub fn with_disk_cache_from_env(client: &'c Client<'c>) -> Self {
        Self {
            client: Some(client),
            mesh: None,
            compilation_options: CompilationOptions::default(),
            cache: Arc::new(CompilationContext::new().with_disk_cache_from_env()),
            marker: PhantomData,
        }
    }

    /// Returns the singleton tracing-only domain token that carries the XLA staged operation
    /// universe but no PJRT execution context. The token's cache is empty and unused.
    ///
    /// This token is sufficient for nested transforms over already-traced XLA values because
    /// those paths only need the backend's operation types; they never materialize concrete
    /// arrays via eager [`Context::bind`] of the nullary identity operations.
    #[inline]
    pub fn token() -> &'static Self {
        static TOKEN: LazyLock<XlaDomain<'static>> = LazyLock::new(|| XlaDomain {
            client: None,
            mesh: None,
            compilation_options: CompilationOptions::default(),
            cache: Arc::new(CompilationContext::new()),
            marker: PhantomData,
        });
        &TOKEN
    }

    /// Returns the PJRT [`Client`] this domain was constructed with.
    #[inline]
    pub fn client(&self) -> &'c Client<'c> {
        self.client.expect("execution XlaDomain should always carry a client")
    }

    /// Returns the test-only [`DeviceMesh`] this domain resolves shard placement against.
    #[inline]
    pub fn mesh(&self) -> &DeviceMesh {
        self.mesh.as_ref().expect("XlaDomain::mesh was called on a domain constructed without a mesh")
    }

    /// Returns the base [`CompilationOptions`] template that the compile path forwards to PJRT.
    #[inline]
    pub fn compilation_options(&self) -> &CompilationOptions {
        &self.compilation_options
    }

    /// Returns the number of compiled programs currently cached in the in-memory tier.
    #[inline]
    pub fn cache_size(&self) -> usize {
        self.cache.cache_size()
    }

    /// Removes every entry from the in-memory cache. Mirrors JAX's
    /// `clear_in_memory_compilation_cache()`.
    #[inline]
    pub fn clear_cache(&self) {
        self.cache.clear_cache();
    }

    /// Test-only constructor that attaches a concrete [`DeviceMesh`] to this domain. Used by
    /// XLA-internal unit tests that exercise the [`Self::constant`] materialization helper.
    #[cfg(test)]
    #[inline]
    pub(crate) fn with_mesh(client: &'c Client<'c>, mesh: DeviceMesh) -> Self {
        Self {
            client: Some(client),
            mesh: Some(mesh),
            compilation_options: CompilationOptions::default(),
            cache: Arc::new(CompilationContext::new()),
            marker: PhantomData,
        }
    }
}

impl<'c> Domain for XlaDomain<'c> {
    type Type = ArrayType;
    type Value = Array<'c>;
    type Constant = XlaConstant;
    type Operation = XlaOperation;
}

impl<'c> Context for XlaDomain<'c> {
    fn lift(&self, constant: XlaConstant) -> Result<Array<'c>, ProgramError> {
        Err(TypeError {
            message: format!("xla captured constant {constant} requires a captured program capture table"),
        }
        .into())
    }

    /// XLA has no host interpreter for arbitrary operations, so eager [`bind`](Context::bind) supports only the
    /// nullary additive/multiplicative identities, which it materializes through the runtime client (the same path the
    /// removed `zero`/`one` methods used). Any other operation is rejected.
    fn bind(&self, operation: Self::Operation, inputs: &[Self::Value]) -> Result<Vec<Self::Value>, ProgramError> {
        let (identity, array_type) = eager_identity_operation(&operation, inputs.len())?;
        validate_identity_synthesis(identity, &array_type)?;
        let kind = if identity == ZERO_OPERATION_NAME { ConstantKind::Zero } else { ConstantKind::One };
        let value = self.constant(&array_type, kind).map_err(|error| TypeError { message: error.to_string() })?;
        Ok(vec![value])
    }
}

impl<'c> DifferentiationContext for XlaDomain<'c> {
    type Tangent = ArrayType;
    type LinearOperation<V: Value<ArrayType>, F: Value<ArrayType>> = LinearXlaOperation<V, XlaConstant, F>;

    fn zero_tangent(&self, array_type: &ArrayType) -> Result<Self::Tangent, ProgramError> {
        xla_identity_metadata(ZERO_OPERATION_NAME, array_type)
    }
}

/// Stateless linear [`Domain`] for XLA tangent and cotangent programs over abstract
/// [`ArrayType`] leaves.
#[derive(Copy, Clone, Debug, Default)]
pub struct LinearXlaDomain;

impl LinearXlaDomain {
    /// Returns a fresh zero-sized linear-XLA-domain instance.
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

impl Domain for LinearXlaDomain {
    type Type = ArrayType;
    type Value = ArrayType;
    type Constant = ArrayType;
    type Operation = LinearXlaOperation<ArrayType>;
}

impl Context for LinearXlaDomain {
    fn lift(&self, constant: ArrayType) -> Result<ArrayType, ProgramError> {
        Ok(constant)
    }

    /// Mirrors [`XlaDomain`]'s eager [`bind`](Context::bind): only the nullary identity operations are supported,
    /// and they resolve to the normalized identity metadata (this linear domain's "values" are [`ArrayType`]s).
    fn bind(&self, operation: Self::Operation, inputs: &[Self::Value]) -> Result<Vec<Self::Value>, ProgramError> {
        let (identity, array_type) = eager_identity_operation(&operation, inputs.len())?;
        Ok(vec![xla_identity_metadata(identity, &array_type)?])
    }
}

/// Validates that `operation` is a nullary additive/multiplicative identity ([`ZERO_OPERATION_NAME`] /
/// [`ONE_OPERATION_NAME`]) taking no inputs, and returns its name together with the single [`ArrayType`] it produces.
/// XLA cannot eagerly interpret arbitrary operations, so [`Context::bind`] only materializes these identities;
/// every other operation (or one given inputs) is rejected here.
fn eager_identity_operation<O: Operation<ArrayType>>(
    operation: &O,
    input_count: usize,
) -> Result<(&'static str, ArrayType), ProgramError> {
    let identity = operation.name();
    if input_count != 0 {
        return Err(TypeError {
            message: format!(
                "xla domain eagerly binds only nullary identity operations, but `{identity}` received \
                {input_count} input(s)",
            ),
        }
        .into());
    }
    if identity != ZERO_OPERATION_NAME && identity != ONE_OPERATION_NAME {
        return Err(TypeError {
            message: format!(
                "xla domain can only eagerly bind the `{ZERO_OPERATION_NAME}` and `{ONE_OPERATION_NAME}` identity \
                operations, but got `{identity}`",
            ),
        }
        .into());
    }
    let mut output_types = operation.infer_output_types(&[])?;
    if output_types.len() != 1 {
        return Err(TypeError {
            message: format!(
                "xla identity operation `{identity}` must produce exactly one output but produced {}",
                output_types.len(),
            ),
        }
        .into());
    }
    Ok((identity, output_types.pop().expect("output count checked above")))
}

fn validate_identity_synthesis(identity: &'static str, array_type: &ArrayType) -> Result<(), ProgramError> {
    match array_type.data_type() {
        DataType::Token | DataType::C64 | DataType::C128 => Err(TypeError {
            message: format!(
                "xla domain cannot synthesize {identity} value for element type {}",
                array_type.data_type()
            ),
        }
        .into()),
        _ => Ok(()),
    }
}

/// Returns the metadata value for an XLA uniform identity value of `array_type`.
fn xla_identity_metadata(identity: &'static str, array_type: &ArrayType) -> Result<ArrayType, ProgramError> {
    validate_identity_synthesis(identity, array_type)?;
    Ok(normalize_uniform_xla_array_type(array_type.clone()))
}

/// Clears sharding state that cannot vary for an XLA uniform identity value.
fn normalize_uniform_xla_array_type(array_type: ArrayType) -> ArrayType {
    let Some(sharding) = array_type.sharding().cloned() else {
        return array_type;
    };
    let sharding = Sharding::with_manual_axes(
        sharding.mesh().clone(),
        sharding.dimensions().to_vec(),
        sharding.unreduced_axes().clone(),
        sharding.reduced_axes().clone(),
        BTreeSet::<String>::new(),
    )
    .expect("normalized uniform XLA array type should preserve valid sharding metadata");
    ArrayType::new(array_type.data_type(), array_type.shape().clone())
        .with_layout(array_type.layout().cloned())
        .with_sharding(sharding)
        .expect("normalized uniform XLA array type should preserve rank-compatible sharding")
}

impl<'c> XlaDomain<'c> {
    /// Materializes a concrete [`Array`] whose addressable shards are filled with a constant.
    fn constant(&self, array_type: &ArrayType, kind: ConstantKind) -> Result<Array<'c>, XlaDomainError> {
        let client = self.client.ok_or_else(|| XlaDomainError::InvalidCompilationOptions {
            reason: "xla runtime constants require a PJRT client".to_string(),
        })?;
        let mesh = self.mesh.as_ref().ok_or_else(|| XlaDomainError::InvalidCompilationOptions {
            reason: "xla runtime constants require a concrete device mesh".to_string(),
        })?;
        static_dimensions_or_panic(array_type);
        let effective_type = match array_type.sharding() {
            Some(_) => array_type.clone(),
            None => array_type.replicated(mesh).map_err(ArrayError::from)?,
        };
        let addressable_ids = addressable_device_ids(client, mesh)?;
        let element_size_in_bytes = array_type.data_type().to_pjrt().element_size_in_bytes()?;

        let mut addressable_buffers = Vec::with_capacity(addressable_ids.len());
        for shard in shards_for_type(&effective_type, mesh)? {
            let shard_device = shard.device();
            let shard_device_id = shard_device.id();
            if !addressable_ids.contains(&shard_device_id) {
                continue;
            }
            let shard_shape = shard.shape();
            let element_count =
                Shape::from(&shard_shape).element_count().map_err(Error::from)?.expect("shard shapes are static");
            let bytes = constant_bytes(array_type.data_type(), kind, element_count, element_size_in_bytes);
            let dimensions = shard_shape.as_slice().iter().map(|&dimension| dimension as u64).collect::<Vec<_>>();
            let device = self
                .client
                .expect("checked above")
                .addressable_devices()?
                .into_iter()
                .find(|device| device.id().map(|id| id == shard_device_id).unwrap_or(false))
                .ok_or(Error::NonAddressableDevice {
                    device_id: shard_device_id,
                    process_index: shard_device.process_index(),
                })?;
            let buffer = client.buffer(
                bytes.as_slice(),
                array_type.data_type().to_pjrt(),
                dimensions.as_slice(),
                None,
                device,
                None,
            )?;
            addressable_buffers.push(buffer);
        }

        Array::from_addressable_buffers(effective_type, mesh.clone(), addressable_buffers).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// Constant materialization
// ---------------------------------------------------------------------------

/// Kind of constant value materialized by [`XlaDomain::constant`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ConstantKind {
    /// Additive identity.
    Zero,

    /// Multiplicative identity.
    One,
}

/// Returns the static dimensions encoded by `array_type`, panicking if any dimension is dynamic.
///
/// Tests use this helper when constructing static-only values and treat dynamic shapes as programmer error.
fn static_dimensions_or_panic(array_type: &ArrayType) -> Vec<usize> {
    array_type
        .static_shape()
        .unwrap_or_else(|| panic!("XlaDomain requires static ArrayType shapes, but got {}", array_type.shape()))
        .dimensions()
        .to_vec()
}

/// Returns a dense row-major host buffer encoding `element_count` copies of `kind` for
/// `data_type`.
///
/// Booleans are encoded as one byte per element (`0` / `1`). Integers and floating-point numbers
/// are encoded in native-endian byte order matching
/// [`ryft_pjrt::Client::buffer`](ryft_pjrt::Client::buffer)'s expectations. Complex numbers are
/// encoded as a `(real, imaginary)` pair of native-endian floats.
fn constant_bytes(data_type: DataType, kind: ConstantKind, element_count: usize, element_size: usize) -> Vec<u8> {
    match kind {
        ConstantKind::Zero => vec![0u8; element_count * element_size],
        ConstantKind::One => {
            let pattern = one_pattern_bytes(data_type);
            debug_assert_eq!(pattern.len(), element_size);
            let mut bytes = Vec::with_capacity(element_count * element_size);
            for _ in 0..element_count {
                bytes.extend_from_slice(&pattern);
            }
            bytes
        }
    }
}

/// Returns the native-endian byte pattern for a single `1`-valued element of `data_type`.
fn one_pattern_bytes(data_type: DataType) -> Vec<u8> {
    match data_type {
        DataType::Boolean => vec![1u8],
        DataType::I8 => 1i8.to_ne_bytes().to_vec(),
        DataType::U8 => 1u8.to_ne_bytes().to_vec(),
        DataType::I16 => 1i16.to_ne_bytes().to_vec(),
        DataType::U16 => 1u16.to_ne_bytes().to_vec(),
        DataType::I32 => 1i32.to_ne_bytes().to_vec(),
        DataType::U32 => 1u32.to_ne_bytes().to_vec(),
        DataType::I64 => 1i64.to_ne_bytes().to_vec(),
        DataType::U64 => 1u64.to_ne_bytes().to_vec(),
        DataType::BF16 => half::bf16::ONE.to_bits().to_ne_bytes().to_vec(),
        DataType::F16 => half::f16::ONE.to_bits().to_ne_bytes().to_vec(),
        DataType::F32 => 1.0f32.to_ne_bytes().to_vec(),
        DataType::F64 => 1.0f64.to_ne_bytes().to_vec(),
        DataType::C64 => {
            let mut bytes = Vec::with_capacity(8);
            bytes.extend_from_slice(&1.0f32.to_ne_bytes());
            bytes.extend_from_slice(&0.0f32.to_ne_bytes());
            bytes
        }
        DataType::C128 => {
            let mut bytes = Vec::with_capacity(16);
            bytes.extend_from_slice(&1.0f64.to_ne_bytes());
            bytes.extend_from_slice(&0.0f64.to_ne_bytes());
            bytes
        }
        // 8-bit floating-point types do not have a canonical Rust representation; encoding `1.0`
        // as a raw byte pattern would depend on the exact FP8 variant.
        DataType::F8E3M4
        | DataType::F8E4M3
        | DataType::F8E4M3FN
        | DataType::F8E4M3FNUZ
        | DataType::F8E4M3B11FNUZ
        | DataType::F8E5M2
        | DataType::F8E5M2FNUZ
        | DataType::F8E8M0FNU
        | DataType::Token
        | DataType::I1
        | DataType::I2
        | DataType::I4
        | DataType::U1
        | DataType::U2
        | DataType::U4
        | DataType::F4E2M1FN => {
            panic!("XlaDomain::one does not support element type {data_type}")
        }
    }
}

/// Returns the addressable device IDs for `client`, filtered to devices that are both addressable by the client and
/// present in the mesh.
fn addressable_device_ids(client: &Client<'_>, mesh: &DeviceMesh) -> Result<Vec<DeviceId>, XlaDomainError> {
    let mut addressable = Vec::new();
    for device in client.addressable_devices()? {
        let device_id = device.id()?;
        if mesh.devices().iter().any(|device| device.id() == device_id) {
            addressable.push(device_id);
        }
    }
    Ok(addressable)
}

/// Returns the shard descriptors implied by `array_type` and `mesh`.
fn shards_for_type(array_type: &ArrayType, mesh: &DeviceMesh) -> Result<Vec<ShardDescriptor>, ArrayError> {
    let sharding = array_type.sharding().ok_or(Error::MissingSharding)?;
    let global_shape =
        array_type.static_shape().ok_or_else(|| Error::DynamicShape { shape: array_type.shape().clone() })?;
    let (descriptors, _) = ShardLayout::new(&global_shape, mesh, sharding)?.into_parts();
    Ok(descriptors)
}

// ---------------------------------------------------------------------------
// CompilationDomain implementation
// ---------------------------------------------------------------------------

/// Backend-specific per-call options for the [`CompilationDomain`] implementation on
/// [`XlaDomain`]. Carries the device mesh, optional sharding overrides for inputs/outputs,
/// and the donation flags computed at jit time.
///
/// `donation_flags` is a flat `Vec<bool>` matching the flat input arity; the core jit pipeline
/// constructs it from its universal `donate_argnums` field before invoking
/// [`CompilationDomain::compile`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct XlaOptions {
    /// Concrete device mesh the compiled program runs against.
    pub mesh: DeviceMesh,

    /// Optional override for per-input shardings.
    pub in_shardings: Option<Vec<Sharding>>,

    /// Optional override for per-output shardings.
    pub out_shardings: Option<Vec<Sharding>>,

    /// Flat per-input donation flags. Length must match the program's flat input arity.
    pub donation_flags: Vec<bool>,
}

impl XlaOptions {
    /// Constructs default options for `mesh` with no shardings overrides and no donation.
    #[inline]
    pub fn new(mesh: DeviceMesh) -> Self {
        Self { mesh, in_shardings: None, out_shardings: None, donation_flags: Vec::new() }
    }

    /// Sets the per-input sharding overrides that the SPMD partitioner reads at lowering time.
    /// Length must equal the flat input arity once the program is traced; mismatches surface as
    /// [`XlaDomainError::InvalidCompilationOptions`] at compile time.
    #[inline]
    pub fn with_in_shardings(mut self, in_shardings: Vec<Sharding>) -> Self {
        self.in_shardings = Some(in_shardings);
        self
    }

    /// Sets the per-output sharding overrides for SPMD partitioning. Length must equal the flat
    /// output arity once the program is traced.
    #[inline]
    pub fn with_out_shardings(mut self, out_shardings: Vec<Sharding>) -> Self {
        self.out_shardings = Some(out_shardings);
        self
    }

    /// Sets the per-input donation flags from a [`Parameterized`] tree of `bool`s whose leaf
    /// shape matches the function's flat input layout.
    ///
    /// Each `true` leaf marks the corresponding input as donatable: the executor may reuse its
    /// device buffer for an output, leaving the caller's runtime value in a donated state after
    /// the call returns. The tree is flattened into [`Self::donation_flags`]; the resulting
    /// vector length is validated against the function's flat input arity at compile time.
    ///
    /// Typical leaf-shaped inputs lower to a single `bool` (e.g. `with_donate(true)` for a
    /// single-argument closure), while nested tuple / struct inputs accept the matching nested
    /// tuple / struct of `bool`s.
    #[inline]
    pub fn with_donate<P: Parameterized<bool>>(mut self, donate: P) -> Self {
        self.donation_flags = donate.into_parameters().collect();
        self
    }
}

impl std::hash::Hash for XlaOptions {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // [`DeviceMesh`] does not derive [`Hash`]; hash its (logical mesh, device order) pair
        // manually instead. Everything else here derives `Hash` already.
        self.mesh.logical_mesh().hash(state);
        self.mesh.devices().hash(state);
        self.in_shardings.hash(state);
        self.out_shardings.hash(state);
        self.donation_flags.hash(state);
    }
}

/// Structural compilation key for [`XlaDomain`]. Implements the
/// [`CompilationKey`](CompilationDomain::CompilationKey) associated type so the cache can use
/// `Eq` on the structured fields to eliminate silent hash collisions.
///
/// Two compilations whose `XlaCompilationKey`s compare equal are guaranteed to produce the
/// same compiled artifact (modulo non-deterministic XLA passes, which we treat as one
/// equivalence class). Two compilations whose keys differ get distinct cache entries.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct XlaCompilationKey {
    /// Identity of the user function being compiled (source location or primitive name).
    pub function: FunctionFingerprint,

    /// Flat input types in trace order. Determines the executable's expected input layouts.
    pub inputs: Vec<ArrayType>,

    /// Per-call options bundle: mesh, sharding overrides, donation flags.
    pub options: XlaOptions,

    /// `Debug` rendering of the domain's base [`PjrtCompilationOptions`](ryft_pjrt::protos::CompilationOptions).
    /// Stored as a string because the protobuf-generated type doesn't derive `Hash`/`Eq`;
    /// `Debug` is stable enough for cache-key purposes.
    pub base_options_debug: String,

    /// PJRT platform name reported by the domain's client, if available. Distinguishes CPU
    /// from GPU from TPU artifacts so a disk cache shared across machines doesn't accidentally
    /// serve a wrong-platform executable.
    pub platform_name: Option<String>,

    /// PJRT platform version reported by the domain's client, if available. Mirrors
    /// [`Self::platform_name`].
    pub platform_version: Option<String>,
}

/// XLA's compiled-program type returned by [`CompilationDomain::compile`]. Carries the loaded
/// PJRT executable plus every piece of per-call state [`CompilationDomain::execute`] needs.
#[derive(Clone)]
pub struct XlaCompiledProgram<'c> {
    /// Compiled PJRT executable. Shared via `std::sync::Arc` so multiple
    /// [`CompiledXlaFunction`](ryft_core::compilation::CompiledXlaFunction)s can share the same
    /// underlying compilation.
    executable: std::sync::Arc<LoadedExecutable<'c>>,

    /// Flat output [`ArrayType`]s in executor-output order. Used by `execute` to reassemble
    /// per-device PJRT buffers into distributed [`Array`] values.
    output_types: std::sync::Arc<[ArrayType]>,

    /// Flat per-input donation flags forwarded to PJRT at execute time.
    donation_flags: std::sync::Arc<[bool]>,

    /// Number of hidden captured-value arguments prepended to the executable signature.
    capture_count: usize,

    /// Expected per-argument shardings captured at compile time. Hidden captures come first,
    /// followed by user inputs. [`CompilationDomain::execute`] uses these to silently reshard
    /// mismatched runtime inputs at the call boundary, mirroring JAX's implicit reshard.
    expected_argument_shardings: std::sync::Arc<[Sharding]>,

    /// Mesh the compiled program runs against. Cloned from the
    /// [`XlaOptions`](XlaOptions::mesh) used at compile time.
    mesh: DeviceMesh,
}

impl<'c> XlaCompiledProgram<'c> {
    /// Returns a reference to the loaded PJRT executable.
    #[inline]
    pub fn executable(&self) -> &LoadedExecutable<'c> {
        &self.executable
    }

    /// Returns the flat output types in executor-output order.
    #[inline]
    pub fn output_types(&self) -> &[ArrayType] {
        &self.output_types
    }

    /// Returns the mesh the compiled program runs against.
    #[inline]
    pub fn mesh(&self) -> &DeviceMesh {
        &self.mesh
    }
}

impl<'c> XlaDomain<'c> {
    /// Compiles one XLA program with hidden captured-value arguments.
    pub(crate) fn compile_program_with_captures<Input, Output>(
        &self,
        program: &XlaProgram<Input, Output>,
        capture_types: &[ArrayType],
        options: &XlaOptions,
    ) -> Result<XlaCompiledProgram<'c>, XlaDomainError>
    where
        Input: Parameterized<XlaConstant>,
        Output: Parameterized<XlaConstant>,
    {
        // Walk input atoms to read each input's `ArrayType`, then apply the optional
        // `in_shardings` override to rewrite the sharding metadata used during lowering.
        let input_types_vec: Vec<ArrayType> = program
            .input_ids()
            .iter()
            .map(|atom_id| program.atoms()[atom_id.index()].r#type().into_owned())
            .collect();
        let input_types_vec = apply_signature_shardings(input_types_vec, options.in_shardings.as_deref(), "in")?;

        // Walk output atoms similarly and apply `out_shardings` override.
        let output_types_vec: Vec<ArrayType> = program
            .output_ids()
            .iter()
            .map(|atom_id| program.atoms()[atom_id.index()].r#type().into_owned())
            .collect();
        let output_types_vec = apply_signature_shardings(output_types_vec, options.out_shardings.as_deref(), "out")?;

        // Validate donation arity. The core jit constructs `donation_flags` from
        // `donate_argnums`; here we just verify the length matches the program's input arity.
        if !options.donation_flags.is_empty() && options.donation_flags.len() != input_types_vec.len() {
            return Err(XlaDomainError::InvalidCompilationOptions {
                reason: format!(
                    "donation_flags has {} entries but the program has {} flat input(s)",
                    options.donation_flags.len(),
                    input_types_vec.len(),
                ),
            });
        }

        // Extract per-argument and per-result shardings for the SPMD partitioner.
        let result_shardings: Option<Vec<Sharding>> =
            output_types_vec.iter().map(|array_type| array_type.sharding().cloned()).collect::<Option<Vec<_>>>();
        let capture_shardings: Vec<Sharding> = capture_types
            .iter()
            .map(|array_type| {
                array_type.sharding().cloned().unwrap_or_else(|| {
                    Sharding::replicated(options.mesh.logical_mesh().clone(), array_type.shape().rank())
                })
            })
            .collect();
        let argument_shardings = capture_shardings
            .iter()
            .cloned()
            .chain(input_types_vec.iter().map(|array_type| {
                array_type.sharding().cloned().unwrap_or_else(|| {
                    Sharding::replicated(options.mesh.logical_mesh().clone(), array_type.shape().rank())
                })
            }))
            .collect::<Vec<_>>();

        // Derive SPMD compilation options from the mesh size, mirroring `jit_compilation_options`.
        let compilation_options = jit_compilation_options(&self.compilation_options, options.mesh.devices().len());

        // Lower → MLIR text via the existing pipeline.
        let mlir_module = crate::experimental::lowering::to_mlir_module_for_program(
            program,
            capture_types,
            // The lowering helper takes `&Input` typed as `Parameterized<ArrayType>` for the
            // global input/output type trees. We pass the flat input/output type Vecs since
            // they implement `Parameterized<ArrayType>` (the trivial leaf-only family).
            &input_types_vec,
            &output_types_vec,
            "main",
            Some(argument_shardings.as_slice()),
            result_shardings.as_deref(),
        )
        .map_err(|error| XlaDomainError::Lowering(error.into()))?;

        // Compile MLIR via PJRT.
        let pjrt_program = PjrtProgram::Mlir { bytecode: mlir_module.into_bytes() };
        let executable = self.client().compile(&pjrt_program, &compilation_options)?;

        Ok(XlaCompiledProgram {
            executable: std::sync::Arc::new(executable),
            output_types: output_types_vec.into(),
            donation_flags: options.donation_flags.clone().into(),
            capture_count: capture_types.len(),
            expected_argument_shardings: argument_shardings.into(),
            mesh: options.mesh.clone(),
        })
    }

    /// Executes one compiled program with captured runtime values prepended as hidden arguments.
    pub(crate) fn execute_with_captures(
        &self,
        program: &XlaCompiledProgram<'c>,
        captures: &[Array<'c>],
        inputs: Vec<Array<'c>>,
    ) -> Result<Vec<Array<'c>>, XlaDomainError> {
        if captures.len() != program.capture_count {
            return Err(XlaDomainError::InvalidCompilationOptions {
                reason: format!(
                    "compiled program expects {} capture(s) but got {}",
                    program.capture_count,
                    captures.len(),
                ),
            });
        }
        let arguments = captures.iter().cloned().chain(inputs).collect::<Vec<_>>();
        let resharded_arguments =
            reshard_inputs_if_needed(self, &program.mesh, &program.expected_argument_shardings, arguments)?;
        let donation_flags = if program.donation_flags.is_empty() {
            vec![false; resharded_arguments.len()]
        } else {
            std::iter::repeat(false)
                .take(program.capture_count)
                .chain(program.donation_flags.iter().copied())
                .collect::<Vec<_>>()
        };
        execute_pjrt(
            &program.executable,
            &program.mesh,
            resharded_arguments,
            donation_flags.as_slice(),
            &program.output_types,
        )
    }
}

impl<'c> CompilationDomain for XlaDomain<'c> {
    type CompiledProgram = XlaCompiledProgram<'c>;
    type Options = XlaOptions;
    type Error = XlaDomainError;
    type CompilationKey = XlaCompilationKey;

    #[inline]
    fn cache(&self) -> Option<&CompilationContext<Self>> {
        Some(&self.cache)
    }

    fn compilation_key(
        &self,
        function: &FunctionFingerprint,
        inputs: &[ArrayType],
        options: &XlaOptions,
    ) -> XlaCompilationKey {
        // Materialize platform identity once at key-construction time. The PJRT client API
        // returns owned `String`s, so storing them in the key avoids re-querying on every
        // cache lookup. Missing platform info (e.g. on the clientless static-staging token) yields `None`,
        // which still distinguishes "no platform" from any concrete platform.
        let (platform_name, platform_version) = match self.client {
            Some(client) => (
                client.platform_name().ok().map(|name| name.into_owned()),
                client.platform_version().ok().map(|version| version.into_owned()),
            ),
            None => (None, None),
        };
        XlaCompilationKey {
            function: function.clone(),
            inputs: inputs.to_vec(),
            options: options.clone(),
            base_options_debug: format!("{:?}", &self.compilation_options),
            platform_name,
            platform_version,
        }
    }

    fn compile<Input, Output>(
        &self,
        program: &XlaProgram<Input, Output>,
        options: &XlaOptions,
    ) -> Result<XlaCompiledProgram<'c>, XlaDomainError>
    where
        Input: Parameterized<XlaConstant>,
        Output: Parameterized<XlaConstant>,
    {
        self.compile_program_with_captures(program, &[], options)
    }

    fn execute(
        &self,
        program: &XlaCompiledProgram<'c>,
        inputs: Vec<Array<'c>>,
    ) -> Result<Vec<Array<'c>>, XlaDomainError> {
        self.execute_with_captures(program, &[], inputs)
    }

    fn serialize_program(&self, program: &XlaCompiledProgram<'c>) -> Result<Vec<u8>, XlaDomainError> {
        // PJRT executables serialize directly to bytes. Note: this loses the metadata that
        // `XlaCompiledProgram` carries alongside (output_types, donation_flags, etc.) — a
        // production-grade disk cache would prepend a header. For now we just round-trip the
        // executable bytes; deserialize_program returns "unsupported" to fall back to a
        // fresh compile on disk-cache hit.
        let exec = program.executable.executable()?;
        let bytes = exec.serialize()?;
        Ok(bytes.data().to_vec())
    }

    fn deserialize_program(&self, _bytes: &[u8]) -> Result<XlaCompiledProgram<'c>, XlaDomainError> {
        // Disk-cache deserialization not yet wired up — would need a header carrying
        // output_types, donation_flags, expected_argument_shardings, and mesh metadata so the
        // full `XlaCompiledProgram` shape can be reconstructed. For now, always treat disk
        // cache hits as misses so the cache falls back to re-compile.
        Err(XlaDomainError::InvalidCompilationOptions {
            reason: "XlaDomain::deserialize_program is not yet implemented; disk cache will fall back to re-compile"
                .to_string(),
        })
    }
}

/// Applies an optional per-leaf sharding override to a flat list of [`ArrayType`]s. Returns
/// the inputs unchanged when `shardings` is `None`. Errors on arity mismatch.
fn apply_signature_shardings(
    mut types: Vec<ArrayType>,
    shardings: Option<&[Sharding]>,
    kind: &'static str,
) -> Result<Vec<ArrayType>, XlaDomainError> {
    let Some(shardings) = shardings else {
        return Ok(types);
    };
    if shardings.len() != types.len() {
        return Err(XlaDomainError::InvalidCompilationOptions {
            reason: format!(
                "{kind}_shardings has {} entries but the function has {} flat {kind}put(s)",
                shardings.len(),
                types.len(),
            ),
        });
    }
    for (array_type, sharding) in types.iter_mut().zip(shardings) {
        *array_type = ArrayType::new(array_type.data_type(), array_type.shape().clone())
            .with_layout(array_type.layout().cloned())
            .with_sharding(sharding.clone())
            .map_err(|error| XlaDomainError::Array(error.into()))?;
    }
    Ok(types)
}

/// Overlays SPMD partitioning fields on the base [`CompilationOptions`] template. Mirrors
/// `crate::jit::jit_compilation_options`.
fn jit_compilation_options(base: &CompilationOptions, partition_count: usize) -> CompilationOptions {
    use ryft_pjrt::protos::ExecutableCompilationOptions;
    let mut options = base.clone();
    let exec_options = options.executable_build_options.get_or_insert_with(ExecutableCompilationOptions::default);
    if exec_options.device_ordinal == 0 {
        exec_options.device_ordinal = -1;
    }
    exec_options.replica_count = 1;
    exec_options.partition_count = partition_count as i64;
    exec_options.use_spmd_partitioning = true;
    exec_options.use_shardy_partitioner = true;
    options
}

/// Reshards `inputs` to match `expected_shardings` at the call boundary. Inputs that already
/// match skip the reshard entirely; the implicit-reshard path is the cold path.
fn reshard_inputs_if_needed<'c>(
    domain: &XlaDomain<'c>,
    mesh: &DeviceMesh,
    expected_shardings: &[Sharding],
    inputs: Vec<Array<'c>>,
) -> Result<Vec<Array<'c>>, XlaDomainError> {
    let needs_reshard = inputs.iter().zip(expected_shardings).any(|(array, expected)| array.sharding() != expected);
    if !needs_reshard {
        return Ok(inputs);
    }
    inputs
        .into_iter()
        .zip(expected_shardings)
        .map(|(array, expected)| {
            if array.sharding() == expected {
                Ok(array)
            } else {
                crate::arrays_v0::compiled_reshard::reshard(&array, domain, mesh, expected)
                    .map_err(XlaDomainError::Array)
            }
        })
        .collect()
}

/// Executes a compiled PJRT executable against `mesh` and reassembles per-device output
/// buffers back into distributed [`Array`] values. Mirrors `XlaDomain::execute_with_donation`.
fn execute_pjrt<'c>(
    executable: &LoadedExecutable<'c>,
    mesh: &DeviceMesh,
    inputs: Vec<Array<'c>>,
    donation_flags: &[bool],
    output_types: &[ArrayType],
) -> Result<Vec<Array<'c>>, XlaDomainError> {
    let addressable_device_ids = executable
        .addressable_devices()?
        .iter()
        .map(|device| device.id().map_err(XlaDomainError::from))
        .collect::<Result<Vec<_>, _>>()?;
    let arguments =
        Array::into_execute_arguments_with_donation(inputs, addressable_device_ids.as_slice(), donation_flags)?;
    let device_outputs =
        executable.execute(arguments.as_execution_device_inputs(), 0, None, Some(file!()), None, None)?;

    let output_count = output_types.len();
    for outputs in &device_outputs {
        if outputs.outputs.len() != output_count {
            return Err(XlaDomainError::Pjrt(ryft_pjrt::Error::invalid_argument(format!(
                "expected {output_count} output(s) per device, but got {}",
                outputs.outputs.len(),
            ))));
        }
    }

    let mut per_output_buffers: Vec<Vec<Buffer<'c>>> =
        (0..output_count).map(|_| Vec::with_capacity(addressable_device_ids.len())).collect();
    for device_output in device_outputs {
        for (output_index, buffer) in device_output.outputs.into_iter().enumerate() {
            per_output_buffers[output_index].push(buffer);
        }
    }

    let mut outputs = Vec::with_capacity(output_count);
    for (output_index, addressable_buffers) in per_output_buffers.into_iter().enumerate() {
        let output_type = output_types[output_index].clone();
        let resolved_type = match output_type.sharding() {
            Some(_) => output_type,
            None => output_type.replicated(mesh).map_err(ArrayError::from)?,
        };
        outputs.push(Array::from_addressable_buffers(resolved_type, mesh.clone(), addressable_buffers)?);
    }
    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::Sharding;
    use ryft_core::sharding::{Device, LogicalMesh, MeshAxis, MeshAxisType, ShardingDimension};
    use ryft_core::types::{Shape, Size, StaticShape};
    use ryft_pjrt::{ClientOptions, CpuClientOptions, load_cpu_plugin};

    use crate::FromPjrt;
    use crate::tests::values_from_bytes;

    use super::*;

    fn cpu_domain_mesh(client: &Client<'_>, axis: &str, axis_size: usize) -> DeviceMesh {
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new(axis, axis_size, MeshAxisType::Auto).unwrap()]).unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .into_iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        DeviceMesh::new(logical_mesh, devices).unwrap()
    }

    #[test]
    fn test_domain_zero_defaults_missing_sharding_to_replicated() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = cpu_domain_mesh(&client, "x", 2);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());

        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3), Size::Static(2)]));
        let array = domain.constant(&array_type, ConstantKind::Zero).unwrap();

        assert_eq!(array.shape(), StaticShape::new(vec![3, 2]));
        assert_eq!(array.shards().len(), 2);
        assert_eq!(array.addressable_shards().count(), 2);
        for shard in array.addressable_shards() {
            let buffer = shard.buffer().unwrap();
            let host_bytes = buffer.copy_to_host(None).unwrap().r#await().unwrap();
            let values = values_from_bytes::<f32>(host_bytes.as_slice());
            assert_eq!(values, vec![0.0; 6]);
        }
    }

    #[test]
    fn test_domain_one_fills_sharded_array_with_ones() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = cpu_domain_mesh(&client, "x", 2);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let array_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)])).with_sharding(sharding).unwrap();
        let domain = XlaDomain::with_mesh(&client, mesh);

        let array = domain.constant(&array_type, ConstantKind::One).unwrap();

        assert_eq!(array.shape(), StaticShape::new(vec![4]));
        assert_eq!(array.shards().len(), 2);
        assert_eq!(array.addressable_shards().count(), 2);
        for shard in array.addressable_shards() {
            assert_eq!(shard.shape(), StaticShape::new(vec![2]));
            let buffer = shard.buffer().unwrap();
            let host_bytes = buffer.copy_to_host(None).unwrap().r#await().unwrap();
            let values = values_from_bytes::<f32>(host_bytes.as_slice());
            assert_eq!(values, vec![1.0, 1.0]);
        }
    }

    #[test]
    fn test_domain_identity_synthesis_rejects_unsupported_constant_type() {
        use ryft_core::operations::constants::SupportsOne;
        let array_type = ArrayType::scalar(DataType::C64);

        assert!(matches!(
            XlaDomain::token().bind(SupportsOne::one_operation(array_type.clone()), &[]),
            Err(ProgramError::Type(error))
                if error.message == "xla domain cannot synthesize one value for element type c64"
        ));
    }

    #[test]
    fn test_linear_domain_one_metadata_normalizes_varying_manual_axes() {
        use ryft_core::operations::constants::SupportsOne;
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::with_manual_axes(
            mesh,
            vec![ShardingDimension::replicated()],
            Vec::<String>::new(),
            Vec::<String>::new(),
            ["x".to_string()],
        )
        .unwrap();
        let array_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)])).with_sharding(sharding).unwrap();

        let one_type = LinearXlaDomain::new()
            .bind(SupportsOne::one_operation(array_type.clone()), &[])
            .unwrap()
            .pop()
            .unwrap();

        assert_eq!(one_type.shape(), array_type.shape());
        assert_eq!(
            one_type.sharding().expect("one metadata should preserve sharding metadata").varying_manual_axes(),
            &BTreeSet::<String>::new(),
        );
    }

    #[test]
    fn test_domain_accessors_return_constructor_arguments() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());

        assert_eq!(domain.mesh(), &mesh);
        assert_eq!(domain.compilation_options(), &CompilationOptions::default());
    }

    #[test]
    fn test_compilation_domain_impl_round_trips_through_core_pipeline() {
        use crate::tests::{values_from_bytes, values_to_bytes};
        use ryft_core::compilation::{CompilationOptions as CoreCompilationOptions, compile_with_options};
        use ryft_core::operations::trigonometric::Sin;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_domain_mesh(&client, "x", 1);
        let engine = XlaDomain::with_mesh(&client, mesh.clone());

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let options = CoreCompilationOptions::<XlaDomain<'_>>::new(XlaOptions::new(mesh.clone()));
        let compiled: ryft_core::compilation::CompiledFunction<'_, XlaDomain<'_>, ArrayType, ArrayType> =
            compile_with_options(&engine, |x| x.sin(), input_type.clone(), options).unwrap();

        // Round-trip a small input through the new CompilationDomain-driven pipeline.
        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();
        let array = compiled.call(source).unwrap();

        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let shard_bytes = array
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        let observed = values_from_bytes::<f32>(shard_bytes.as_slice());
        for (got, &input) in observed.iter().zip(values.iter()) {
            assert!((got - input.sin()).abs() < 1e-5, "got {got}, expected ~{}", input.sin());
        }

        // Repeat invocations at the same source line should share one cache entry.
        let cache_size_before = engine.cache_size();
        for _ in 0..3 {
            let _: ryft_core::compilation::CompiledFunction<'_, XlaDomain<'_>, ArrayType, ArrayType> =
                compile_with_options(
                    &engine,
                    |x| x.sin(),
                    input_type.clone(),
                    CoreCompilationOptions::<XlaDomain<'_>>::new(XlaOptions::new(mesh.clone())),
                )
                .unwrap();
        }
        assert_eq!(
            engine.cache_size(),
            cache_size_before + 1,
            "three repeat jit calls at the same source line should populate exactly one new cache entry",
        );
    }
}
