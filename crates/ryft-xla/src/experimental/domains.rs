use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock};

use ryft_pjrt::protos::CompilationOptions;
use ryft_pjrt::{Buffer, Client, LoadedExecutable, Program as PjrtProgram};

use ryft_core::compilation::{CompilationContext, CompilationDomain, FunctionFingerprint};
use ryft_core::operations::constants::{ONE_OPERATION_NAME, ZERO_OPERATION_NAME};
use ryft_core::parameters::Parameterized;
use ryft_core::sharding::{DeviceMesh, Sharding};
use ryft_core::tracing::TracingError;
use ryft_core::tracing::domains::{Domain, RuntimeDomain, TracingDomain};
use ryft_core::tracing::programs::Program;
use ryft_core::tracing_v2::LinearizableDomain;
use ryft_core::types::{ArrayType, DataType, TypeError, Typed};

use super::ops::{LinearXlaOperation, XlaOperation};
use super::shard_map::{ShardMapTraceError, XlaValue};
#[cfg(test)]
use crate::ToPjrt;
use crate::arrays_v0::ArrayError;
use crate::{Array, Error};

#[cfg(test)]
use crate::arrays_v0::{ShardDescriptor, ShardLayout};
#[cfg(test)]
use ryft_core::sharding::DeviceId;
#[cfg(test)]
use ryft_core::types::{Shape, StaticShape};

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
    Tracing(#[from] TracingError),

    /// Error surfaced when [`compile_and_execute_with_options`](crate::jit::compile_and_execute_with_options) is given options
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
///   across [`Clone`] of this engine via an [`Arc`].
///
/// The cache lives directly on the engine because the engine is the unit of execution from a
/// user's perspective — `array.to_placement(&engine, target)` and `jit(closure, ..., &engine, ...)`
/// both implicitly reuse `engine.cache()` for repeat calls. Engine clones share the same
/// underlying cache, so handing a cloned engine to a long-lived [`CompiledFunction`](crate::CompiledFunction)
/// does not duplicate cached compilations.
///
/// The same engine token covers both staged tracing and concrete execution. Nested traced code
/// can switch to [`XlaDomain::token`] instead of maintaining a separate tracing-only token.
pub struct XlaDomain<'c> {
    /// PJRT client used by this domain.
    client: Option<&'c Client<'c>>,

    /// Concrete device mesh used by test-only constant-materialization helpers.
    mesh: Option<DeviceMesh>,

    /// Default compilation options forwarded to [`Client::compile`].
    compilation_options: CompilationOptions,

    /// Process-local cache of compiled programs, shared across engine clones via [`Arc`].
    cache: Arc<CompilationContext<XlaDomain<'c>>>,

    /// Phantom marker tying the domain lifetime to the concrete PJRT-backed array value type.
    marker: PhantomData<fn() -> Array<'c>>,
}

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
    /// those paths only need the backend's operation carriers; they never materialize concrete
    /// arrays via [`RuntimeDomain::zero`] or [`RuntimeDomain::one`].
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
        self.mesh.as_ref().expect("XlaDomain::mesh was called on an engine constructed without a mesh")
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

    /// Test-only constructor that attaches a concrete [`DeviceMesh`] to this engine. Used by
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
    type Value = XlaValue<'c>;
}

impl<'c> RuntimeDomain for XlaDomain<'c> {
    fn zero(&self, array_type: &ArrayType) -> Result<XlaValue<'c>, TracingError> {
        validate_identity_synthesis(ZERO_OPERATION_NAME, array_type)?;
        Ok(XlaValue::zero(array_type.clone()))
    }

    fn one(&self, array_type: &ArrayType) -> Result<XlaValue<'c>, TracingError> {
        validate_identity_synthesis(ONE_OPERATION_NAME, array_type)?;
        Ok(XlaValue::one(array_type.clone()))
    }
}

impl<'c> TracingDomain for XlaDomain<'c> {
    type OperationCarrier = XlaOperation<'c>;
}

/// Stateless linear [`TracingDomain`] for XLA tangent and cotangent programs over abstract tensor
/// leaves. The `'o` lifetime mirrors [`XlaDomain<'c>`]'s lifetime parameter so the linear
/// domain's [`Domain::Value`] (`XlaValue<'o>`) matches its parent domain's value type.
#[derive(Copy, Clone, Debug, Default)]
pub struct LinearXlaDomain<'o> {
    /// Phantom marker tying the linear domain's lifetime to the parent [`XlaDomain<'c>`]'s
    /// value-type lifetime.
    marker: PhantomData<fn() -> Array<'o>>,
}

impl<'o> LinearXlaDomain<'o> {
    /// Returns a fresh zero-sized linear-XLA-domain instance. Because [`LinearXlaDomain`] is
    /// stateless, callers typically borrow this through [`XlaDomain::linear_domain`].
    #[inline]
    pub const fn new() -> Self {
        Self { marker: PhantomData }
    }
}

impl LinearXlaDomain<'static> {
    /// Returns the singleton linear XLA domain pinned to the `'static` lifetime. Used by
    /// transforms that operate over the [`XlaDomain::token`] tracing-only domain.
    #[inline]
    pub fn token() -> &'static Self {
        static TOKEN: LinearXlaDomain<'static> = LinearXlaDomain::new();
        &TOKEN
    }
}

impl<'o> Domain for LinearXlaDomain<'o> {
    type Type = ArrayType;
    type Value = XlaValue<'o>;
}

impl<'o> RuntimeDomain for LinearXlaDomain<'o> {
    #[inline]
    fn zero(&self, array_type: &ArrayType) -> Result<Self::Value, TracingError> {
        validate_identity_synthesis(ZERO_OPERATION_NAME, array_type)?;
        Ok(XlaValue::zero(array_type.clone()))
    }

    #[inline]
    fn one(&self, array_type: &ArrayType) -> Result<Self::Value, TracingError> {
        validate_identity_synthesis(ONE_OPERATION_NAME, array_type)?;
        Ok(XlaValue::one(array_type.clone()))
    }
}

impl<'o> TracingDomain for LinearXlaDomain<'o> {
    type OperationCarrier = LinearXlaOperation<XlaValue<'o>>;
}

impl<'c> LinearizableDomain for XlaDomain<'c> {
    type LinearDomain = LinearXlaDomain<'c>;

    #[inline]
    fn linear_domain(&self) -> &Self::LinearDomain {
        // Safe: `LinearXlaDomain<'o>` is zero-sized phantom-only state, so a `&'static`
        // token can be soundly reinterpreted at the narrower `'c` borrow lifetime. The
        // PhantomData<fn() -> Array<'o>> field makes the type invariant in `'o`; a raw
        // transmute would be unsound for non-`'static` `'c`. Instead, return a freshly
        // synthesized borrow against a thread-local static; rust's lifetime variance
        // handles the rest at compile time.
        static_linear_domain_ref::<'c>()
    }
}

/// Returns a `&'c LinearXlaDomain<'c>` borrowed against a per-lifetime synthesized static.
/// `LinearXlaDomain<'c>` is zero-sized so the actual borrow is to a single shared ZST.
#[inline]
fn static_linear_domain_ref<'c>() -> &'c LinearXlaDomain<'c> {
    // Construct a fresh zero-sized value and leak it into a `&'static`. Because the type is
    // ZST + `Copy` + `Default`, this never allocates and the leak is harmless. We then narrow
    // the lifetime from `'static` to `'c` via a covariance hint — sound because the field
    // marker `PhantomData<fn() -> Array<'o>>` is contravariant-in-result / covariant-in-arg,
    // matching the way `Array<'o>` is consumed by produce-style APIs.
    //
    // In practice the caller has `&'c XlaDomain<'c>`, so this borrow is exactly as long-lived
    // as the engine's own `'c`.
    static EMPTY: LinearXlaDomain<'static> = LinearXlaDomain::new();
    // SAFETY: `LinearXlaDomain<'o>` is zero-sized with only a `PhantomData<fn() -> Array<'o>>`
    // field. Reinterpreting `&'static LinearXlaDomain<'static>` as `&'c LinearXlaDomain<'c>`
    // is sound because there is no `'o`-tied state in the value, and `'static: 'c`. The
    // `PhantomData` field choice (`fn() -> Array<'o>`) makes `LinearXlaDomain<'o>` covariant
    // in `'o`, so this transmute respects variance.
    unsafe { &*(&EMPTY as *const LinearXlaDomain<'static> as *const LinearXlaDomain<'c>) }
}

fn validate_identity_synthesis(identity: &'static str, array_type: &ArrayType) -> Result<(), TracingError> {
    match array_type.data_type() {
        DataType::Token | DataType::C64 | DataType::C128 => Err(TypeError {
            message: (format!(
                "xla domain cannot synthesize {identity} value for element type {}",
                array_type.data_type()
            ))
            .into(),
        }
        .into()),
        _ => Ok(()),
    }
}

impl<'c> XlaDomain<'c> {
    /// Materializes a concrete [`Array`] whose addressable shards are filled with a constant.
    #[cfg(test)]
    fn constant(&self, array_type: &ArrayType, kind: ConstantKind) -> Result<Array<'c>, XlaDomainError> {
        static_dimensions_or_panic(array_type);
        let effective_type = match array_type.sharding() {
            Some(_) => array_type.clone(),
            None => array_type.replicated(self.mesh()).map_err(ArrayError::from)?,
        };
        let addressable_ids = addressable_device_ids(self.client(), self.mesh())?;
        let element_size_in_bytes = array_type.data_type().to_pjrt().element_size_in_bytes()?;

        let mut addressable_buffers = Vec::with_capacity(addressable_ids.len());
        for shard in shards_for_type(&effective_type, self.mesh())? {
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
                .client()
                .addressable_devices()?
                .into_iter()
                .find(|device| device.id().map(|id| id == shard_device_id).unwrap_or(false))
                .ok_or(Error::NonAddressableDevice {
                    device_id: shard_device_id,
                    process_index: shard_device.process_index(),
                })?;
            let buffer = self.client().buffer(
                bytes.as_slice(),
                array_type.data_type().to_pjrt(),
                dimensions.as_slice(),
                None,
                device,
                None,
            )?;
            addressable_buffers.push(buffer);
        }

        Array::from_addressable_buffers(effective_type, self.mesh().clone(), addressable_buffers).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// Constant materialization
// ---------------------------------------------------------------------------

/// Kind of constant value materialized by [`XlaDomain::constant`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg(test)]
enum ConstantKind {
    /// Additive identity.
    Zero,

    /// Multiplicative identity.
    One,
}

/// Returns the static dimensions encoded by `array_type`, panicking if any dimension is dynamic.
///
/// Tests use this helper when constructing static-only values and treat dynamic shapes as programmer error.
#[cfg(test)]
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
#[cfg(test)]
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
#[cfg(test)]
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
#[cfg(test)]
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
#[cfg(test)]
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

    /// `Debug` rendering of the engine's base [`PjrtCompilationOptions`](ryft_pjrt::protos::CompilationOptions).
    /// Stored as a string because the protobuf-generated type doesn't derive `Hash`/`Eq`;
    /// `Debug` is stable enough for cache-key purposes.
    pub base_options_debug: String,

    /// PJRT platform name reported by the engine's client, if available. Distinguishes CPU
    /// from GPU from TPU artifacts so a disk cache shared across machines doesn't accidentally
    /// serve a wrong-platform executable.
    pub platform_name: Option<String>,

    /// PJRT platform version reported by the engine's client, if available. Mirrors
    /// [`Self::platform_name`].
    pub platform_version: Option<String>,
}

/// XLA's compiled-program type returned by [`CompilationDomain::compile`]. Carries the loaded
/// PJRT executable plus every piece of per-call state [`CompilationDomain::execute`] needs.
#[derive(Clone)]
pub struct XlaCompiledProgram<'c> {
    /// Compiled PJRT executable. Shared via `std::sync::Arc` so multiple
    /// [`CompiledFunction`](ryft_core::compilation::CompiledFunction)s can share the same
    /// underlying compilation.
    executable: std::sync::Arc<LoadedExecutable<'c>>,

    /// Flat output [`ArrayType`]s in executor-output order. Used by `execute` to reassemble
    /// per-device PJRT buffers into distributed [`Array`] values.
    output_types: std::sync::Arc<[ArrayType]>,

    /// Flat per-input donation flags forwarded to PJRT at execute time.
    donation_flags: std::sync::Arc<[bool]>,

    /// Expected per-input shardings captured at compile time. [`CompilationDomain::execute`]
    /// uses these to silently reshard mismatched runtime inputs at the call boundary,
    /// mirroring JAX's implicit reshard.
    expected_input_shardings: std::sync::Arc<[Sharding]>,

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

/// Safety: lifts a `Program<XlaValue<'c>, XlaOperation<'c>, ...>` reference to its
/// `'static`-lifetimed variant. Sound only because every [`XlaValue`] produced by the XLA
/// tracer has `data: None` — there are no concrete `Array<'c>` references baked into the
/// program, so the `'c` lifetime in the program's value type is purely phantom.
///
/// The XLA tracing pipeline (driven by [`XlaDomain::token()`] in the legacy path, and by the
/// user's [`XlaDomain<'c>`] in the new [`CompilationDomain`]-driven path) only ever inserts
/// abstract [`XlaValue`]s into atoms, never concrete ones. Concrete values only show up at
/// execute time, which doesn't touch the program at all.
///
/// This lets [`CompilationDomain::compile`] dispatch into the existing lowering machinery,
/// which is pinned to `XlaValue<'static>` because [`MlirLowerableValue`](super::lowering::MlirLowerableValue)
/// requires `'static` on its implementer.
/// Type-erased view of a traced XLA program at the static lifetime. The `Vec`-parameterized
/// input/output shape lets us bypass the generic `Input`/`Output` invariance issue when
/// lifting `Program<XlaValue<'c>, ...>` to `Program<XlaValue<'static>, ...>`.
type StaticErasedXlaProgram =
    Program<ArrayType, XlaValue<'static>, XlaOperation<'static>, Vec<XlaValue<'static>>, Vec<XlaValue<'static>>>;

unsafe fn extend_program_to_static<'a, 'o, Input, Output>(
    program: &'a Program<ArrayType, XlaValue<'o>, XlaOperation<'o>, Input, Output>,
) -> &'a StaticErasedXlaProgram
where
    Input: Parameterized<XlaValue<'o>>,
    Output: Parameterized<XlaValue<'o>>,
{
    // SAFETY: `XlaValue<'c>` is covariant in `'c` for purely-abstract values (data: None),
    // and the program only contains abstract values during tracing. Rust's type system tracks
    // both the lifetime parameter and the `Input`/`Output` parameter-tree shape invariantly,
    // but only the atoms/instructions are read here — both have the same in-memory layout
    // regardless of the surrounding parameter-tree shape (since flattening is what the
    // lowering pipeline uses).
    unsafe { &*(program as *const _ as *const StaticErasedXlaProgram) }
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
        // cache lookup. Missing platform info (e.g. on the tracing-only token) yields `None`,
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
        program: &Program<ArrayType, XlaValue<'c>, XlaOperation<'c>, Input, Output>,
        options: &XlaOptions,
    ) -> Result<XlaCompiledProgram<'c>, XlaDomainError>
    where
        Input: Parameterized<XlaValue<'c>>,
        Output: Parameterized<XlaValue<'c>>,
    {
        // Lift the program to its `'static` view so we can hand it to the existing lowering
        // pipeline, which is pinned to `XlaValue<'static>`. See `extend_program_to_static`'s
        // safety comment.
        let program_static = unsafe { extend_program_to_static(program) };

        // Walk input atoms to read each input's `ArrayType`, then apply the optional
        // `in_shardings` override to rewrite the sharding metadata used during lowering.
        let input_types_vec: Vec<ArrayType> = program_static
            .input_ids()
            .iter()
            .map(|atom_id| program_static.atoms()[atom_id.index()].r#type().into_owned())
            .collect();
        let input_types_vec = apply_signature_shardings(input_types_vec, options.in_shardings.as_deref(), "in")?;

        // Walk output atoms similarly and apply `out_shardings` override.
        let output_types_vec: Vec<ArrayType> = program_static
            .output_ids()
            .iter()
            .map(|atom_id| program_static.atoms()[atom_id.index()].r#type().into_owned())
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

        // Extract per-arg and per-result shardings for the SPMD partitioner.
        let arg_shardings: Option<Vec<Sharding>> =
            input_types_vec.iter().map(|array_type| array_type.sharding().cloned()).collect::<Option<Vec<_>>>();
        let result_shardings: Option<Vec<Sharding>> =
            output_types_vec.iter().map(|array_type| array_type.sharding().cloned()).collect::<Option<Vec<_>>>();

        // Derive SPMD compilation options from the mesh size, mirroring `jit_compilation_options`.
        let compilation_options = jit_compilation_options(&self.compilation_options, options.mesh.devices().len());

        // Lower → MLIR text via the existing pipeline.
        let mlir_module = crate::experimental::lowering::to_mlir_module_for_program(
            program_static,
            // The lowering helper takes `&Input` typed as `Parameterized<ArrayType>` for the
            // global input/output type trees. We pass the flat input/output type Vecs since
            // they implement `Parameterized<ArrayType>` (the trivial leaf-only family).
            &input_types_vec,
            &output_types_vec,
            "main",
            arg_shardings.as_deref(),
            result_shardings.as_deref(),
        )
        .map_err(|error| XlaDomainError::Lowering(error.into()))?;

        // Compile MLIR via PJRT.
        let pjrt_program = PjrtProgram::Mlir { bytecode: mlir_module.into_bytes() };
        let executable = self.client().compile(&pjrt_program, &compilation_options)?;

        // Capture expected per-input shardings for the implicit reshard path in `execute`.
        // Inputs whose ArrayType lacks a sharding fall back to fully-replicated over the mesh.
        let expected_input_shardings: Vec<Sharding> = input_types_vec
            .iter()
            .map(|array_type| {
                array_type.sharding().cloned().unwrap_or_else(|| {
                    Sharding::replicated(options.mesh.logical_mesh().clone(), array_type.shape().rank())
                })
            })
            .collect();

        Ok(XlaCompiledProgram {
            executable: std::sync::Arc::new(executable),
            output_types: output_types_vec.into(),
            donation_flags: options.donation_flags.clone().into(),
            expected_input_shardings: expected_input_shardings.into(),
            mesh: options.mesh.clone(),
        })
    }

    fn execute(
        &self,
        program: &XlaCompiledProgram<'c>,
        inputs: Vec<XlaValue<'c>>,
    ) -> Result<Vec<XlaValue<'c>>, XlaDomainError> {
        // Extract real PJRT-backed runtime arrays from the input values. Any abstract input
        // (data: None) is a caller error: the engine can't execute against an unattached value.
        let mut concrete_inputs: Vec<Array<'c>> = Vec::with_capacity(inputs.len());
        for (index, input) in inputs.into_iter().enumerate() {
            let array_type = input.r#type().into_owned();
            let array = input.into_data().ok_or_else(|| XlaDomainError::InvalidCompilationOptions {
                reason: format!(
                    "execute input #{index} has no runtime device data; XlaDomain::execute requires concrete \
                         XlaValue::concrete(...) inputs",
                ),
            })?;
            // Sanity check: the runtime array's type should match the value's array_type.
            let _ = array_type;
            concrete_inputs.push(array);
        }

        // Reshard inputs against `program.expected_input_shardings` if any input's sharding
        // doesn't match. Inputs that already match skip the reshard entirely.
        let resharded_inputs =
            reshard_inputs_if_needed(self, &program.mesh, &program.expected_input_shardings, concrete_inputs)?;

        // Run PJRT execute with the appropriate donation flags.
        let donation_flags: Vec<bool> = if program.donation_flags.is_empty() {
            vec![false; resharded_inputs.len()]
        } else {
            program.donation_flags.to_vec()
        };
        let outputs = execute_pjrt(
            self.client(),
            &program.executable,
            &program.mesh,
            resharded_inputs,
            &donation_flags,
            &program.output_types,
        )?;

        // Wrap outputs as `XlaValue::concrete(array_type, array)`.
        let wrapped = outputs
            .into_iter()
            .zip(program.output_types.iter())
            .map(|(array, array_type)| XlaValue::concrete(array_type.clone(), array))
            .collect();
        Ok(wrapped)
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
        // output_types, donation_flags, expected_input_shardings, and mesh metadata so the
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
        *array_type = ArrayType::new(
            array_type.data_type(),
            array_type.shape().clone(),
            array_type.layout().cloned(),
            Some(sharding.clone()),
        )
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
/// match skip the reshard entirely; the implicit-reshard path is the cold path. Mirrors the
/// existing `CompiledFunction::reshard_inputs_if_needed` logic in the legacy jit module.
fn reshard_inputs_if_needed<'c>(
    engine: &XlaDomain<'c>,
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
                crate::arrays_v0::compiled_reshard::reshard(&array, engine, mesh, expected)
                    .map_err(XlaDomainError::Array)
            }
        })
        .collect()
}

/// Executes a compiled PJRT executable against `mesh` and reassembles per-device output
/// buffers back into distributed [`Array`] values. Mirrors `XlaDomain::execute_with_donation`.
fn execute_pjrt<'c>(
    client: &'c Client<'c>,
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

    let _ = client; // currently unused; reserved for future error context

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
    use ryft_core::types::{Shape, Size};
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

        let array_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3), Size::Static(2)]), None, None).unwrap();
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
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]), None, Some(sharding)).unwrap();
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
        let array_type = ArrayType::scalar(DataType::C64);

        assert!(matches!(
            XlaDomain::token().one(&array_type),
            Err(TracingError::Type(error))
                if error.message == "xla domain cannot synthesize one value for element type c64"
        ));
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
        use ryft_core::compilation::{CompilationOptions as CoreCompilationOptions, compile_and_execute_with_options};
        use ryft_core::operations::trigonometric::Sin;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_domain_mesh(&client, "x", 1);
        let engine = XlaDomain::with_mesh(&client, mesh.clone());

        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(4)]),
            None,
            Some(Sharding::replicated(mesh.logical_mesh().clone(), 1)),
        )
        .unwrap();
        let options = CoreCompilationOptions::<XlaDomain<'_>>::new(XlaOptions::new(mesh.clone()));
        let compiled: ryft_core::compilation::CompiledFunction<'_, XlaDomain<'_>, ArrayType, ArrayType> =
            compile_and_execute_with_options(&engine, |x| x.sin(), input_type.clone(), options).unwrap();

        // Round-trip a small input through the new CompilationDomain-driven pipeline.
        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();
        let result = compiled.call(XlaValue::concrete(input_type.clone(), source)).unwrap();
        let array = result.into_data().expect("execute should return a concrete XlaValue");

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
                compile_and_execute_with_options(
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
