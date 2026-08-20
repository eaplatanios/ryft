use std::cell::RefCell;
use std::collections::{BTreeMap, HashSet};
use std::fmt::{Display, Formatter};
use std::hash::Hash;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, LazyLock, OnceLock};
use std::time::{Duration, Instant};

use prost::Message;
use ryft_core::macros::check_count;
use ryft_core::operations::sort::SortDirection;
use ryft_core::{
    AnalyzableCompilationDomain, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, BatchingError,
    BindingRegionDriver, CallRequest, CompilationCacheDomain, CompilationContext, CompilationDomain, CompileRequest,
    Constant, ConstantOperation, Context, DataType, Device, DeviceId, DeviceMesh, DifferentiationError, Dimension,
    DimensionBounds, DimensionFromScalar, DimensionOperation, DimensionSize, DimensionType, DimensionValue,
    DimensionVariable, DischargedReferenceProgram, DiskCache, Domain, DomainTracer, EagerContext, Effect,
    InterpretableOperation, InterpretationDriver, Layout, LogicalMesh, LoweringRequest, Memory, MeshAxis, MeshAxisType,
    ONE_OPERATION_NAME, Operation, Parameterized, Placeholder, ProgramError, ReductionKind, ScatterReductionKind,
    Shape, Sharding, ShardingDimension, StageRequest, StagedFunction, StaticShape, StridedLayout, Tile, TileDimension,
    TiledLayout, Type, TypeError, TypeRefinements, Typed, ValueProjection, ZERO_OPERATION_NAME, Zero,
    ZeroOperationProvider,
};
#[cfg(test)]
use ryft_core::{Array as CpuArray, ProjectedContext};
use ryft_pjrt::protos::CompilationOptions;
use ryft_pjrt::{Buffer, Client, Execution, LoadOptions, LoadedExecutable, Program as PjrtProgram, Value as PjrtValue};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::lowering::{XlaExecutableSignature, contains_unresolved_references, contains_unresolved_state};
use super::ops::{FlatXlaProgram, JitCallOperation, XlaConstant, XlaOperation, XlaProgramBuilder};
use super::shard_map::ShardMapTraceError;
use crate::arrays::ArrayTypeExtension;
use crate::arrays_v0::host::{DenseArrayHostCopy, begin_materialize_dense_array_bytes, materialize_dense_array_bytes};
use crate::arrays_v0::{
    BoundedMaterializationKey, BoundedMaterializationProbe, BoundedMaterializationProducer,
    BoundedMaterializationWaiter,
};
use crate::experimental::operations::ShardMapOperation;
use crate::{Array, ArrayError, Error, FromPjrt, ShardDescriptor, ShardLayout, ToPjrt};

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

    /// Error surfaced by batching (e.g. an unsupported operation inside a batched `jit_call` boundary).
    #[error("{0}")]
    Batching(#[from] BatchingError),

    /// Error surfaced when [`compile_with_options`](crate::jit::compile_with_options) is given options
    /// that do not match the traced function's arity or shape — for example a
    /// `donate_argnums` index outside the flat input range, or an `in_shardings` length that
    /// doesn't match the number of flat inputs.
    #[error("invalid compilation options: {reason}")]
    InvalidCompilationOptions { reason: String },

    /// The runtime domain does not own the PJRT executable it was asked to invoke or install.
    #[error("xla executable belongs to a different PJRT client")]
    ExecutableClientMismatch,

    /// Error surfaced while encoding or decoding persistent compilation metadata.
    #[error("invalid persistent XLA executable: {reason}")]
    InvalidPersistentExecutable { reason: String },

    /// Error surfaced while normalizing backend analysis data.
    #[error("invalid XLA compilation analysis: {reason}")]
    InvalidCompilationAnalysis { reason: String },

    /// Runtime assertion host callbacks currently require host-accessible scalar buffers.
    #[error("compiled runtime assertions are not supported on XLA platform `{platform}`")]
    UnsupportedRuntimeAssertionPlatform { platform: String },

    /// Reference discharge produced external state that the ordinary array-only XLA invocation boundary cannot
    /// supply or install.
    #[error("unsupported XLA reference ABI: {reason}")]
    UnsupportedReferenceAbi { reason: String },
}

/// Stateful backend that materializes, lowers, compiles, and executes traced XLA programs
/// against a live PJRT [`Client`].
///
/// An [`XlaDomain`] bundles four pieces of context:
///
/// - a PJRT [`Client`] used to upload `zero`/`one` shards and to compile and execute programs (including the
///   per-operation programs behind eager [`Context::bind`] dispatch),
/// - an optional concrete [`DeviceMesh`] that eager binds prefer when deriving their execution mesh and that the
///   constant-materialization fast path requires,
/// - an immutable, shared default [`CompilationOptions`] template that the compile path forwards to PJRT, and
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

    /// Concrete device mesh that eager binds prefer when deriving their execution mesh and that the
    /// constant-materialization fast path requires.
    mesh: Option<DeviceMesh>,

    /// Default compilation options forwarded to [`Client::compile`]. Shared because the template is immutable and
    /// [`XlaDomain`] values are cloned into every transform tracer that executes through this domain.
    compilation_options: Arc<CompilationOptions>,

    /// Process-local cache of compiled programs, shared across domain clones via [`Arc`].
    cache: Arc<CompilationContext<XlaDomain<'c>>>,

    /// Phantom marker tying the domain lifetime to the concrete PJRT-backed array value type.
    marker: PhantomData<fn() -> Array<'c>>,
}

/// Tracer shape used while staging XLA programs directly from types.
pub(crate) type XlaTracer<'context> = DomainTracer<XlaDomain<'context>>;

impl<'c> Clone for XlaDomain<'c> {
    fn clone(&self) -> Self {
        Self {
            client: self.client,
            mesh: self.mesh.clone(),
            compilation_options: Arc::clone(&self.compilation_options),
            cache: Arc::clone(&self.cache),
            marker: PhantomData,
        }
    }
}

impl<'c> XlaDomain<'c> {
    /// Returns the shared default [`CompilationOptions`] template.
    #[inline]
    fn default_compilation_options() -> Arc<CompilationOptions> {
        static DEFAULT: LazyLock<Arc<CompilationOptions>> = LazyLock::new(|| Arc::new(CompilationOptions::default()));
        Arc::clone(&DEFAULT)
    }

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
            compilation_options: Arc::new(compilation_options),
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
            compilation_options: Self::default_compilation_options(),
            cache: Arc::new(CompilationContext::with_capacity(capacity)),
            marker: PhantomData,
        }
    }

    /// Creates a new [`XlaDomain`] whose compile cache also writes through to a
    /// [`DiskCache`] rooted at `directory`. Returns an
    /// [`std::io::Error`] only when the directory itself can't be opened or created.
    #[inline]
    pub fn with_disk_cache(client: &'c Client<'c>, directory: impl Into<PathBuf>) -> std::io::Result<Self> {
        let cache = CompilationContext::new().with_disk_cache(directory)?;
        Ok(Self {
            client: Some(client),
            mesh: None,
            compilation_options: Self::default_compilation_options(),
            cache: Arc::new(cache),
            marker: PhantomData,
        })
    }

    /// Creates a new [`XlaDomain`] using an already configured persistent [`DiskCache`]. This is the constructor to
    /// use when callers need explicit capacity or write thresholds rather than [`Self::with_disk_cache`]'s defaults.
    #[inline]
    pub fn with_configured_disk_cache(client: &'c Client<'c>, disk_cache: DiskCache) -> Self {
        Self {
            client: Some(client),
            mesh: None,
            compilation_options: Self::default_compilation_options(),
            cache: Arc::new(CompilationContext::new().with_configured_disk_cache(disk_cache)),
            marker: PhantomData,
        }
    }

    /// Creates a new [`XlaDomain`] whose compile cache also writes through to a
    /// [`DiskCache`] configured via the
    /// [`DiskCache::ENV_VAR`](ryft_core::compilation::DiskCache::ENV_VAR) environment variable,
    /// if it is set. An absent variable produces an in-memory-only cache; an invalid cache directory returns the
    /// corresponding I/O error.
    #[inline]
    pub fn with_disk_cache_from_env(client: &'c Client<'c>) -> std::io::Result<Self> {
        Ok(Self {
            client: Some(client),
            mesh: None,
            compilation_options: Self::default_compilation_options(),
            cache: Arc::new(CompilationContext::new().with_disk_cache_from_env()?),
            marker: PhantomData,
        })
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
            compilation_options: XlaDomain::default_compilation_options(),
            cache: Arc::new(CompilationContext::new()),
            marker: PhantomData,
        });
        &TOKEN
    }

    /// Creates a new [`XlaDomain`] that shares an existing compile cache instead of starting with an empty one.
    ///
    /// This is the constructor behind [`Array::execution_domain`](ryft_core::programs::Value::execution_domain)
    /// recovery: eager outputs carry the compile cache of the domain that produced them, so recovered domains keep
    /// hitting the same cache instead of recompiling every repeated operation signature.
    #[inline]
    pub(crate) fn with_shared_cache(client: &'c Client<'c>, cache: Arc<CompilationContext<XlaDomain<'c>>>) -> Self {
        Self {
            client: Some(client),
            mesh: None,
            compilation_options: Self::default_compilation_options(),
            cache,
            marker: PhantomData,
        }
    }

    /// Creates a new clientless [`XlaDomain`] equivalent to [`Self::token`] but owned by value. Like the token, the
    /// returned domain carries the XLA staged operation universe but no PJRT execution context, so eager
    /// [`Context::bind`] calls on it fail with the existing "requires a PJRT client" errors. This is the
    /// [`Array::execution_domain`](ryft_core::programs::Value::execution_domain) fallback for arrays that carry no
    /// attached client (e.g., arrays assembled through [`Array::from_addressable_buffers`] with a `None` client and
    /// no subsequent [`Array::with_client`] call).
    #[inline]
    pub(crate) fn clientless() -> Self {
        Self {
            client: None,
            mesh: None,
            compilation_options: Self::default_compilation_options(),
            cache: Arc::new(CompilationContext::new()),
            marker: PhantomData,
        }
    }

    /// Returns the PJRT [`Client`] this domain was constructed with, or [`Error::MissingClient`] for token and
    /// clientless domains.
    #[inline]
    pub fn client(&self) -> Result<&'c Client<'c>, Error> {
        self.client.ok_or(Error::MissingClient)
    }

    /// Returns the [`DeviceMesh`] this domain resolves shard placement against, or [`Error::MissingMesh`] when the
    /// domain was constructed without a mesh; eager binds derive their mesh from the inputs in that case.
    #[inline]
    pub fn mesh(&self) -> Result<&DeviceMesh, Error> {
        self.mesh.as_ref().ok_or(Error::MissingMesh)
    }

    /// Returns the base [`CompilationOptions`] template that the compile path forwards to PJRT.
    #[inline]
    pub fn compilation_options(&self) -> &CompilationOptions {
        self.compilation_options.as_ref()
    }

    /// Returns the number of compiled programs currently cached in the in-memory tier.
    #[inline]
    pub fn cache_size(&self) -> usize {
        self.cache.cache_size()
    }

    /// Returns this domain's shared compilation context.
    #[inline]
    pub fn compilation_context(&self) -> &CompilationContext<Self> {
        &self.cache
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
            compilation_options: Self::default_compilation_options(),
            cache: Arc::new(CompilationContext::new()),
            marker: PhantomData,
        }
    }
}

impl<'c> Domain for XlaDomain<'c> {
    type Type = ArrayIrType;
    type Value = ArrayIrValue<Array<'c>>;
    type Constant = XlaConstant;
    type Operation = XlaOperation;
}

impl<'c> Context for XlaDomain<'c> {
    /// An immediate [`XlaConstant::Dimension`] extent is a checked host integer and materializes directly. A
    /// [`XlaConstant::Captured`] payload is a symbolic index into a compiled function's capture table carrying only a
    /// type and no data, so there is nothing to materialize without the surrounding capture table and lifting it is
    /// always rejected.
    fn lift(&self, constant: XlaConstant) -> Result<ArrayIrValue<Array<'c>>, ProgramError> {
        match constant {
            XlaConstant::Captured(constant) => Err(TypeError::invalid(format!(
                "xla captured constant {constant} requires a captured program capture table"
            ))
            .into()),
            XlaConstant::Dimension(value) => Ok(ArrayIrValue::Dimension(value)),
        }
    }

    /// Eagerly executes `operation` on concrete input [`Array`]s, mirroring JAX's op-by-op dispatch: the operation
    /// is traced into a single-instruction program over the inputs' physical [`ArrayType`]s (shardings included),
    /// compiled through this domain's compile cache, and executed on this domain's PJRT client via the crate-private
    /// `eager_bind` path. The nullary additive/multiplicative identities keep a fast path that materializes the constant
    /// directly through the runtime client without compiling a program.
    fn bind<P, D: BindingRegionDriver<Self::Constant, Self::Operation>>(
        &self,
        operation: P,
        driver: D,
        inputs: &[Self::Value],
    ) -> Result<Vec<Self::Value>, ProgramError>
    where
        P: Into<Self::Operation>,
    {
        let operation = operation.into();
        operation.validate_region_count(driver.region_count())?;
        let name = operation.name();
        if inputs.is_empty() && (name == ZERO_OPERATION_NAME || name == ONE_OPERATION_NAME) {
            let array_type = eager_identity_output_type(&operation)?;
            validate_identity_synthesis(name, &array_type)?;
            // The direct constant-materialization fast path needs a concrete device mesh; mesh-less domains fall
            // through to the compiled eager path below, which derives a default execution mesh instead.
            if self.mesh.is_some() {
                let kind = if name == ZERO_OPERATION_NAME { ConstantKind::Zero } else { ConstantKind::One };
                let value = self.constant(&array_type, kind).map_err(|error| TypeError::invalid(error.to_string()))?;
                return Ok(vec![ArrayIrValue::Array(value)]);
            }
        }
        self.eager_bind(operation, driver, inputs)
    }

    /// A client-backed domain executes every bound operation for real and its concrete [`Array`]s support host
    /// readback through [`Concretizable<bool>`](ryft_core::Concretizable) and
    /// [`WhilePredicate`](ryft_core::operations::control_flow::WhilePredicate), so strategies that fold
    /// data-dependent work through host-visible values — the eager data-dependent `while` rules and
    /// concretizable-`while` unrolling — apply. Clientless domains (the static staging [`token`](Self::token) and
    /// domains recovered from arrays without an attached client) cannot execute operations and stay non-eager.
    fn is_eager(&self) -> bool {
        self.client.is_some()
    }
}

/// Context capability that materializes additive-identity [`Array`]s. Transform machinery over this domain (e.g.,
/// the batching rules of nullary constant operations and the accumulator seeding of recursive higher-order rules)
/// synthesizes constants through the active context's type-driven [`Zero`], [`One`](ryft_core::One),
/// [`Fill`](ryft_core::Fill), and [`Iota`](ryft_core::Iota) leaves.
/// The binds below take the constant-materialization fast path on domains constructed with a concrete mesh and the
/// compiled eager dispatch path (over a derived default mesh) otherwise. Dynamic array zeros are intentionally not
/// available through this type-only capability because they require explicit first-class extent operands.
impl<'c> Zero<ArrayIrValue<Array<'c>>> for XlaDomain<'c> {
    fn zero(&self, r#type: &ArrayIrType) -> Result<ArrayIrValue<Array<'c>>, ProgramError> {
        let mut outputs = self.bind(XlaOperation::zero_operation(r#type.clone())?, Vec::new(), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Materialization delegates to [`Context::lift`] and therefore shares its semantics: an immediate
/// [`XlaConstant::Dimension`] extent materializes directly as a host-side dimension value, while a
/// [`XlaConstant::Captured`] payload carries only a type and no data and is always rejected outside a surrounding
/// capture table. The implementation exists because interpretation- and batching-capable operation families require a
/// [`Constant`] leaf on their contexts; programs whose constants were compiled into capture tables never take the
/// captured path.
impl<'c> Constant<ArrayIrValue<Array<'c>>, XlaConstant> for XlaDomain<'c> {
    fn constant(&self, value: XlaConstant) -> Result<ArrayIrValue<Array<'c>>, ProgramError> {
        self.lift(value)
    }
}

/// Eager interpretation of a staged jitted call over concrete [`Array`]s. Top-level flat programs containing
/// `jit_call` instructions (for example the pullbacks produced by eager `vjp`/`grad`) replay through
/// [`Program::interpret_in_context`](ryft_core::Program::interpret_in_context), whose bind channel hands the callee
/// region to [`Context::bind`] so the call is compiled whole through this domain's dispatch cache and executed on its
/// PJRT client — mirroring JAX dispatching a jitted function called from eager code straight to the compiled
/// executable. This rule covers the remaining path — a `jit_call` nested inside another region being interpreted —
/// by re-entering the active interpreter on the callee region, which dispatches the callee's operations one by one
/// through this same domain.
impl<'c> InterpretableOperation<XlaDomain<'c>> for JitCallOperation<ArrayIrType> {
    fn interpret<D: InterpretationDriver<XlaDomain<'c>>>(
        &self,
        context: &XlaDomain<'c>,
        driver: &D,
        inputs: &[ArrayIrValue<Array<'c>>],
    ) -> Result<Vec<ArrayIrValue<Array<'c>>>, ProgramError> {
        driver.interpret_region(context, 0, inputs.to_vec())
    }
}

/// Eager interpretation of a `shard_map` instruction is rejected: eager shard-map execution flows through
/// [`Context::bind`]'s region channel instead ([`Program::interpret_in_context`](ryft_core::Program::interpret_in_context)
/// materializes the attached body region and [`XlaDomain::bind`] SPMD-compiles the manual computation whole), and the
/// [`InterpretationDriver`] this rule receives can replay the body region but cannot materialize it
/// for a whole rebind. This implementation exists to satisfy the operation family's interpretation bound on
/// [`XlaDomain`]-valued replays, which never dispatch it in practice.
// TODO(eaplatanios): [regions] Deferred-behavior rejection: if a nested eager replay ever needs to execute a
//  `shard_map` through this rule, extend `InterpretationDriver` with a whole-rebind request instead of
//  interpreting the local body over global values (phase 7 or later of
//  `.tasks/plan_first_class_program_regions.md`).
impl<'c> InterpretableOperation<XlaDomain<'c>> for ShardMapOperation<XlaConstant> {
    fn interpret<D: InterpretationDriver<XlaDomain<'c>>>(
        &self,
        _context: &XlaDomain<'c>,
        _driver: &D,
        _inputs: &[ArrayIrValue<Array<'c>>],
    ) -> Result<Vec<ArrayIrValue<Array<'c>>>, ProgramError> {
        Err(ProgramError::UnsupportedOperation {
            message: "eager shard_map replay must bind through a client-backed domain context".to_string(),
        })
    }
}

/// Returns the single [`ArrayType`] produced by a nullary additive/multiplicative identity operation
/// ([`ZERO_OPERATION_NAME`] / [`ONE_OPERATION_NAME`]). The identity fast path in [`Context::bind`] materializes these
/// constants directly through the runtime client instead of compiling a program.
fn eager_identity_output_type<O: Operation<Type = ArrayIrType>>(operation: &O) -> Result<ArrayType, ProgramError> {
    let mut output_types = operation.infer_output_types(&[], &[])?;
    if output_types.len() != 1 {
        return Err(TypeError::invalid(format!(
            "xla identity operation `{}` must produce exactly one output but produced {}",
            operation.name(),
            output_types.len(),
        ))
        .into());
    }
    let output_type = output_types.pop().expect("output count checked above");
    <&ArrayType>::try_from(&output_type).cloned().map_err(Into::into)
}

fn validate_identity_synthesis(identity: &'static str, array_type: &ArrayType) -> Result<(), ProgramError> {
    match array_type.data_type() {
        DataType::Token | DataType::Zero if identity == ONE_OPERATION_NAME => Err(TypeError::invalid(format!(
            "xla domain cannot synthesize {identity} value for element type {}",
            array_type.data_type()
        ))
        .into()),
        DataType::Token => Err(TypeError::invalid(format!(
            "xla domain cannot synthesize {identity} value for element type {}",
            array_type.data_type()
        ))
        .into()),
        _ => Ok(()),
    }
}

impl<'c> XlaDomain<'c> {
    /// Eagerly executes one staged operation on concrete input [`Array`]s — the JAX-style op-by-op dispatch path
    /// behind [`Context::bind`].
    ///
    /// The operation is traced into a single-instruction flat program over the inputs' physical [`ArrayType`]s
    /// (shardings included), compiled through this domain's compile cache, and executed on this domain's PJRT client.
    /// The cache key contains the complete lowered computation, its effective compilation options, and the derived
    /// mesh, so repeated eager binds of the same lowering reuse one compiled executable without conflating distinct
    /// computations. Higher-order operations (`condition` / `while` / `scan` / `jit_call` / `shard_map`) receive their
    /// nested programs as attached regions and flow through this same path — the compiler handles the control flow, so
    /// no host interpreter loops are needed.
    fn eager_bind<D: BindingRegionDriver<XlaConstant, XlaOperation>>(
        &self,
        operation: XlaOperation,
        driver: D,
        inputs: &[ArrayIrValue<Array<'c>>],
    ) -> Result<Vec<ArrayIrValue<Array<'c>>>, ProgramError> {
        if inputs.iter().any(|input| matches!(input, ArrayIrValue::Reference(_))) {
            return Err(ProgramError::UnsupportedOperation {
                message: "references must be discharged before XLA eager execution".to_string(),
            });
        }
        // The intrinsic operation effects alone would miss region-carried state, while effect scans alone would miss
        // a pure reference-typed pass-through or capture. The borrowed closure scan covers dormant rule regions too
        // without cloning their program graphs. Although those rules do not execute in this eager bind, ordinary XLA
        // rejects unresolved references artifact-wide, so accepting one here would only defer failure to compilation
        // with a less targeted diagnostic.
        if operation.effects().contains(Effect::OrderedState)
            || driver.regions().any(|region| {
                region.contains_effect_in_closure(Effect::OrderedState)
                    || region.contains_atom_type_in_closure(Type::is_reference)
            })
        {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("`{}` must be discharged before XLA eager execution", operation.name()),
            });
        }
        if let XlaOperation::Dimension(operation) = &operation {
            if driver.regions().count() != 0 {
                return Err(TypeError::invalid("dimension operations do not accept attached regions").into());
            }
            let inputs = inputs
                .iter()
                .map(|input| {
                    <ArrayIrValue<Array<'c>> as ryft_core::ValueProjection<DimensionType>>::projected(input).cloned()
                })
                .collect::<Result<Vec<_>, _>>()?;
            let outputs = EagerContext::<DimensionValue, DimensionOperation<DimensionValue>>::new().bind(
                operation.clone(),
                Vec::new(),
                inputs.as_slice(),
            )?;
            return Ok(outputs.into_iter().map(ArrayIrValue::Dimension).collect());
        }
        if let XlaOperation::DimensionSize(operation) = &operation {
            if driver.regions().count() != 0 {
                return Err(TypeError::invalid("dimension_size does not accept attached regions").into());
            }
            check_count!("input", inputs, 1, ProgramError);
            operation.infer_output_types(&[inputs[0].r#type().into_owned()], &[])?;
            let array = <ArrayIrValue<Array<'c>> as ryft_core::ValueProjection<ArrayType>>::projected(&inputs[0])?;
            let extent = array.dimension_size(operation.axis())?;
            return Ok(vec![ArrayIrValue::Dimension(DimensionValue::new(operation.result_type().clone(), extent)?)]);
        }
        if let XlaOperation::DimensionFromScalar(operation) = &operation {
            if driver.regions().count() != 0 {
                return Err(TypeError::invalid("dimension_from_scalar does not accept attached regions").into());
            }
            check_count!("input", inputs, 1, ProgramError);
            operation.infer_output_types(&[inputs[0].r#type().into_owned()], &[])?;
            let array = <ArrayIrValue<Array<'c>> as ryft_core::ValueProjection<ArrayType>>::projected(&inputs[0])?;
            return Ok(vec![ArrayIrValue::Dimension(array.to_dimension(operation.result_type().variable().clone())?)]);
        }

        let Some(client) = self.client else {
            return Err(ProgramError::InvalidArgument {
                message: format!(
                    "xla domain cannot eagerly execute operation `{}` without a PJRT client",
                    operation.name(),
                ),
            });
        };
        if matches!(&operation, XlaOperation::DimensionToScalar(_)) {
            if driver.regions().count() != 0 {
                return Err(TypeError::invalid("dimension_to_scalar does not accept attached regions").into());
            }
            check_count!("input", inputs, 1, ProgramError);
            let dimension =
                <ArrayIrValue<Array<'c>> as ryft_core::ValueProjection<DimensionType>>::projected(&inputs[0])?;
            let extent = i64::try_from(dimension.extent()).unwrap();
            let output_type = ArrayType::scalar(DataType::I64);
            let mesh = self.eager_mesh(client, &[], std::slice::from_ref(&output_type))?;
            let output_type = output_type
                .replicated(&mesh)
                .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
            let output = Array::from_host_buffer(client, output_type, mesh, extent.to_ne_bytes().as_slice())
                .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?
                .with_compilation_cache(Arc::clone(&self.cache));
            return Ok(vec![ArrayIrValue::Array(output)]);
        }

        let mut array_inputs = Vec::new();

        // Trace the single-instruction program over the inputs' physical types, shardings included, attaching the
        // provided region bodies to that instruction.
        let region_input_types = vec![None; driver.regions().count()];
        let builder = Rc::new(RefCell::new(XlaProgramBuilder::new()));
        let region_ids = driver.import_into(&builder, &region_input_types)?;
        let output_atoms = {
            let mut builder = builder.borrow_mut();
            let input_atoms = inputs
                .iter()
                .map(|input| match input {
                    ArrayIrValue::Array(array) => {
                        array_inputs.push(array.clone());
                        Ok(builder.add_input(ArrayIrType::Array(array.r#type().into_owned())))
                    }
                    ArrayIrValue::Dimension(dimension) => {
                        let operation = DimensionOperation::Constant(ConstantOperation::new(dimension.clone()));
                        Ok(builder.add_instruction(XlaOperation::Dimension(operation), Vec::new(), Vec::new())?[0])
                    }
                    ArrayIrValue::Reference(_) => {
                        unreachable!("reference inputs are rejected at the `eager_bind` entry guard")
                    }
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            builder.add_instruction(operation, region_ids, input_atoms)?.to_vec()
        };
        self.validate_eager_placement(client, array_inputs.as_slice())?;
        let output_count = output_atoms.len();
        let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        let program: FlatXlaProgram =
            builder.build(output_atoms, vec![Placeholder; array_inputs.len()], vec![Placeholder; output_count])?;

        // Derive the mesh after tracing so that input-free operations can fall back to their inferred output
        // shardings, then compile through the domain's cache (a repeated eager operation is a cache hit) and
        // execute via PJRT.
        let output_types = program
            .output_types()
            .iter()
            .map(<&ArrayType>::try_from)
            .map(|result| result.cloned())
            .collect::<Result<Vec<_>, _>>()?;
        let mesh = self.eager_mesh(client, array_inputs.as_slice(), output_types.as_slice())?;
        let options = XlaOptions::new(mesh);
        let lowered = self
            .lower_xla_program(&program, 0, &options)
            .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
        let cache_key = self
            .compilation_key(&lowered)
            .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
        let compiled = self
            .cache
            .get_or_compile(self, cache_key, || self.compile_xla_program(&lowered))
            .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
        let outputs = self
            .execute_xla_program(&compiled, array_inputs)
            .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;

        // Execution already attached this domain's client to every output, so attaching the compile cache is all
        // that is left for chained eager operations and transforms over the outputs to recover a context that keeps
        // executing on the same client and keeps hitting the same compile cache.
        Ok(outputs
            .into_iter()
            .map(|output| ArrayIrValue::Array(output.with_compilation_cache(Arc::clone(&self.cache))))
            .collect())
    }

    /// Validates that every input lives on this domain's PJRT client and that all inputs share one device placement,
    /// mirroring JAX's "received incompatible devices for jitted computation" error for the eager path. Inputs that
    /// carry an attached client (see [`Array::client`]) are checked by client identity, which also rejects
    /// same-device-id arrays owned by a *different* client; inputs with no attached client fall back to membership of
    /// every shard device in the executing client's device set as the placement proxy.
    fn validate_eager_placement(&self, client: &'c Client<'c>, inputs: &[Array<'c>]) -> Result<(), ProgramError> {
        if inputs.is_empty() {
            return Ok(());
        }
        let invalid_argument = |message: String| ProgramError::InvalidArgument { message };
        let client_device_ids = client
            .devices()
            .map_err(|error| invalid_argument(error.to_string()))?
            .iter()
            .map(|device| device.id())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| invalid_argument(error.to_string()))?;
        for (index, input) in inputs.iter().enumerate() {
            if let Some(input_client) = input.client() {
                // Client identity is exact: `Array::with_client` already validated that this client owns every
                // addressable shard buffer, so no per-shard device check is needed.
                if !std::ptr::eq(input_client, client) {
                    return Err(invalid_argument(format!(
                        "received incompatible devices for eager xla execution: input #{index} is owned by a \
                         different PJRT client than this domain's client",
                    )));
                }
                continue;
            }
            for shard in input.shards() {
                let device_id = shard.device().id();
                if !client_device_ids.contains(&device_id) {
                    return Err(invalid_argument(format!(
                        "received incompatible devices for eager xla execution: input #{index} is placed on device \
                         {device_id}, which does not belong to this domain's PJRT client",
                    )));
                }
            }
        }
        let first_device_ids = inputs[0].mesh().devices().iter().map(Device::id).collect::<Vec<_>>();
        for (index, input) in inputs.iter().enumerate().skip(1) {
            let device_ids = input.mesh().devices().iter().map(Device::id).collect::<Vec<_>>();
            if device_ids != first_device_ids {
                return Err(invalid_argument(format!(
                    "received incompatible devices for eager xla execution: input #{index} is placed on devices \
                     {device_ids:?} but input #0 is placed on devices {first_device_ids:?}",
                )));
            }
        }
        Ok(())
    }

    /// Returns the concrete [`DeviceMesh`] eager execution compiles against: the domain's own mesh when one is
    /// attached, otherwise the mesh implied by the inputs' shard placement, otherwise the logical mesh of the first
    /// declared output [`Sharding`] assembled over the client's addressable devices (for input-free operations with
    /// sharded outputs, such as the nullary sharded constants that transform machinery synthesizes through
    /// [`Zero`] / [`One`] / [`Fill`] / [`Iota`] on mesh-less recovered domains), otherwise a single-device mesh over
    /// the client's first addressable device (for input-free operations over unsharded data).
    fn eager_mesh(
        &self,
        client: &'c Client<'c>,
        inputs: &[Array<'c>],
        output_types: &[ArrayType],
    ) -> Result<DeviceMesh, ProgramError> {
        if let Some(mesh) = self.mesh.as_ref() {
            return Ok(mesh.clone());
        }
        if let Some(input) = inputs.first() {
            return Ok(input.mesh());
        }
        let invalid_argument = |message: String| ProgramError::InvalidArgument { message };
        let devices = client.addressable_devices().map_err(|error| invalid_argument(error.to_string()))?;
        if let Some(logical_mesh) =
            output_types.iter().find_map(|r#type| r#type.sharding().map(|sharding| sharding.mesh().clone()))
        {
            // An input-free operation with sharded outputs has no concrete placement to inherit, so materialize its
            // declared logical mesh over the client's addressable devices in enumeration order.
            let device_count = logical_mesh.device_count();
            if devices.len() < device_count {
                return Err(invalid_argument(format!(
                    "eager xla execution of an input-free operation sharded over mesh {logical_mesh:?} requires \
                     {device_count} addressable device(s) but the client only has {}",
                    devices.len(),
                )));
            }
            let devices = devices
                .iter()
                .take(device_count)
                .map(|device| Device::from_pjrt(device).map_err(|error| invalid_argument(error.to_string())))
                .collect::<Result<Vec<_>, _>>()?;
            return DeviceMesh::new(logical_mesh, devices).map_err(|error| invalid_argument(error.to_string()));
        }
        let device = devices
            .first()
            .ok_or_else(|| invalid_argument("eager xla execution requires at least one addressable device".into()))?;
        let device = Device::from_pjrt(device).map_err(|error| invalid_argument(error.to_string()))?;
        let axis = MeshAxis::new("x", 1, MeshAxisType::Auto).map_err(|error| invalid_argument(error.to_string()))?;
        let logical_mesh = LogicalMesh::new(vec![axis]).map_err(|error| invalid_argument(error.to_string()))?;
        DeviceMesh::new(logical_mesh, vec![device]).map_err(|error| invalid_argument(error.to_string()))
    }

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
        if array_type.data_type().is_zero() {
            return Ok(Array::from_zero_space(client, effective_type, mesh.clone())?
                .with_compilation_cache(Arc::clone(&self.cache)));
        }
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

        // Attach this domain's client and compile cache so that chained eager operations and transforms over the
        // materialized constant recover a context that keeps the same client and compile cache.
        Ok(Array::from_canonical_addressable_buffers(client, effective_type, mesh.clone(), addressable_buffers)?
            .with_compilation_cache(Arc::clone(&self.cache)))
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
        // Low-precision (4-, 6-, and 8-bit) floating-point types do not have a canonical Rust representation;
        // encoding `1.0` as a raw byte pattern would depend on the exact variant.
        DataType::F8E3M4
        | DataType::F8E4M3
        | DataType::F8E4M3FN
        | DataType::F8E4M3FNUZ
        | DataType::F8E4M3B11FNUZ
        | DataType::F8E5M2
        | DataType::F8E5M2FNUZ
        | DataType::F8E8M0FNU
        | DataType::Token
        | DataType::Zero
        | DataType::I1
        | DataType::I2
        | DataType::I4
        | DataType::U1
        | DataType::U2
        | DataType::U4
        | DataType::F4E2M1FN
        | DataType::F6E2M3FN
        | DataType::F6E3M2FN => {
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

/// Opaque XLA feedback-directed optimization profile used for profile-guided latency estimation.
///
/// XLA defines the binary payload consumed by `ExecutableCompilationOptions::fdo_profile`; Ryft intentionally does
/// not reinterpret it. Profiles are compiler- and program-specific. Supplying one changes exact, persistent, and
/// distributed compilation identity because its bytes are encoded in the canonical compilation options.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct XlaFeedbackDirectedProfile {
    bytes: Arc<[u8]>,
    version: i64,
}

impl XlaFeedbackDirectedProfile {
    /// Creates a profile with XLA profile version zero.
    #[inline]
    pub fn new(bytes: impl Into<Vec<u8>>) -> Self {
        Self { bytes: bytes.into().into(), version: 0 }
    }

    /// Loads opaque profile bytes from `path`.
    pub fn from_file(path: impl AsRef<Path>) -> std::io::Result<Self> {
        std::fs::read(path).map(Self::new)
    }

    /// Sets the XLA compilation profile version associated with these bytes.
    #[inline]
    pub fn with_version(mut self, version: i64) -> Self {
        self.version = version;
        self
    }

    /// Returns the opaque profile bytes passed to XLA.
    #[inline]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns the XLA compilation profile version.
    #[inline]
    pub fn version(&self) -> i64 {
        self.version
    }

    /// Returns a stable SHA-256 digest suitable for diagnostics and provenance.
    #[inline]
    pub fn digest(&self) -> [u8; 32] {
        Sha256::digest(self.bytes()).into()
    }

    /// Writes the opaque profile bytes to `path`.
    #[inline]
    pub fn write_to_file(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.bytes())
    }
}

/// Host-side policy for grouping concrete input extents into bounded retained-JIT specializations.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum XlaInputBoundBucketing {
    /// Rounds each host-observed positive axis extent up to the next power of two. The resulting dynamic type admits
    /// extents through that bucket, bounding padding overhead below two times the logical extent while retaining only
    /// logarithmically many compiled specializations.
    PowerOfTwo,
}

/// Opaque retained-JIT specialization key derived by [`XlaDomain`].
///
/// Exact dispatch preserves nominal [`DimensionVariable`] identity and shares its type slice with the effective staged
/// signature, so keying a call costs no deep type clone. Dispatch with input-bound bucketing instead uses a structural,
/// alpha-normalized physical-bound signature so calls routed to the same bucket share one staged and compiled
/// specialization. Bucketed keys index dimension variables by first occurrence and exclude their diagnostic names,
/// which survive only in the effective staged types.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct XlaDispatchKey(XlaDispatchKeyKind);

/// Internal representation of [`XlaDispatchKey`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum XlaDispatchKeyKind {
    /// Exact runtime abstract signature used when bucketing is disabled, shared with the effective staged types.
    Exact(Arc<[ArrayIrType]>),

    /// Alpha-normalized bounded signature used when bucketing is enabled.
    Bucketed(BucketedDispatchSignature),
}

/// Alpha-normalized bounded input signature used as the bucketed retained-dispatch key.
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

/// One bucketed input array type split into a shape-free carrier and its alpha-normalized shape pattern.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct BucketedArrayTypeKey {
    /// Array type with an empty shape, retaining data type, layout, sharding, and memory identity.
    carrier: ArrayType,

    /// Per-axis alpha-normalized dimensions, which also fix the array's rank.
    dimensions: Vec<BucketedDimensionKey>,
}

/// One alpha-normalized axis of a [`BucketedArrayTypeKey`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum BucketedDimensionKey {
    /// Statically known extent. Bucketing rewrites every static input axis into a bounded dynamic axis, so this
    /// variant is unreachable for signatures produced today; it is retained so the normalization stays total over
    /// [`Dimension`].
    Static(usize),

    /// Index of the axis' dimension variable in [`BucketedDispatchSignature::variables`].
    Variable(usize),
}

impl BucketedDispatchSignature {
    /// Alpha-normalizes the already bucketed `input_types` into a name-free dispatch key.
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
                        Dimension::Static(extent) => BucketedDimensionKey::Static(*extent),
                        Dimension::Dynamic(variable) => {
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

/// Backend-specific per-call options for the [`CompilationDomain`] implementation on
/// [`XlaDomain`]. Carries the device mesh, optional sharding overrides for inputs/outputs,
/// optional per-input donation flags, and retained-dispatch policies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct XlaOptions {
    /// Concrete device mesh the compiled program runs against.
    pub mesh: DeviceMesh,

    /// Optional override for per-input shardings.
    pub in_shardings: Option<Vec<Sharding>>,

    /// Optional override for per-output shardings.
    pub out_shardings: Option<Vec<Sharding>>,

    /// Optional flat per-input donation flags. `None` donates nothing; a present vector must match the program's
    /// flat public input arity exactly, which lowering validates.
    pub donation_flags: Option<Vec<bool>>,

    /// Optional opaque feedback-directed optimization profile forwarded to XLA.
    pub feedback_directed_profile: Option<XlaFeedbackDirectedProfile>,

    /// Optional host-side bound bucketing policy for retained-JIT input signatures.
    pub input_bound_bucketing: Option<XlaInputBoundBucketing>,
}

impl XlaOptions {
    /// Constructs default options for `mesh` with no shardings overrides and no donation.
    #[inline]
    pub fn new(mesh: DeviceMesh) -> Self {
        Self {
            mesh,
            in_shardings: None,
            out_shardings: None,
            donation_flags: None,
            feedback_directed_profile: None,
            input_bound_bucketing: None,
        }
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
    /// vector length is validated against the function's flat public input arity at lowering time.
    ///
    /// Typical leaf-shaped inputs lower to a single `bool` (e.g. `with_donate(true)` for a
    /// single-argument closure), while nested tuple / struct inputs accept the matching nested
    /// tuple / struct of `bool`s.
    #[inline]
    pub fn with_donate<P: Parameterized<bool>>(mut self, donate: P) -> Self {
        self.donation_flags = Some(donate.into_parameters().collect());
        self
    }

    /// Supplies an opaque feedback-directed optimization profile for manual profile-guided recompilation.
    ///
    /// The profile must have been produced for compatible StableHLO, topology, and compiler settings. XLA performs
    /// backend-specific validation according to its debug options; Ryft includes the exact bytes in every cache key.
    #[inline]
    pub fn with_feedback_directed_profile(mut self, profile: XlaFeedbackDirectedProfile) -> Self {
        self.feedback_directed_profile = Some(profile);
        self
    }

    /// Enables host-side retained-JIT input-bound bucketing.
    ///
    /// Static runtime array axes become bounded-dynamic axes in the staged signature, with bounds selected by
    /// `bucketing`. The observed extent remains entirely host-side, participates in dispatch identity only through its
    /// bucket, and is transported through the existing bounded-input ABI. Data-derived extents created inside the
    /// program are deliberately unaffected because observing them here would require a device-to-host synchronization
    /// and a split compilation boundary. Bucketing currently accepts only array inputs on a single-device mesh:
    /// bucketed signatures are dynamic and therefore require disabling the Shardy partitioner, which Ryft does not do
    /// implicitly for a multi-device request.
    #[inline]
    pub fn with_input_bound_bucketing(mut self, bucketing: XlaInputBoundBucketing) -> Self {
        self.input_bound_bucketing = Some(bucketing);
        self
    }
}

impl std::hash::Hash for XlaOptions {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // `DeviceMesh` does not derive `Hash`; hash its (logical mesh, device order) pair
        // manually instead. Everything else here derives `Hash` already.
        self.mesh.logical_mesh().hash(state);
        self.mesh.devices().hash(state);
        self.in_shardings.hash(state);
        self.out_shardings.hash(state);
        self.donation_flags.hash(state);
        self.feedback_directed_profile.hash(state);
        self.input_bound_bucketing.hash(state);
    }
}

/// StableHLO lowering and call-boundary metadata produced before PJRT compilation.
#[derive(Clone, Debug)]
pub struct XlaLoweredProgram {
    /// Textual StableHLO/Shardy module accepted by PJRT.
    stable_hlo: Arc<str>,

    /// Effective PJRT compilation options, including SPMD partition count.
    compilation_options: CompilationOptions,

    /// Effective logical output types after applying output-sharding overrides.
    output_types: Arc<[ArrayType]>,

    /// Effective logical input types, including captures first.
    input_types: Arc<[ArrayType]>,

    /// Logical-to-physical StableHLO/PJRT boundary mapping.
    signature: XlaExecutableSignature,

    /// Whether execution requires Ryft's host runtime assertion handler.
    requires_assertion_handler: bool,

    /// Donation declarations for public inputs. Captures are always non-donatable.
    donation_flags: Arc<[bool]>,

    /// Number of leading logical inputs supplied by the staged capture table.
    capture_count: usize,

    /// Effective sharding expected for every physical executable argument, including materialized captures first.
    expected_argument_shardings: Arc<[Sharding]>,

    /// Concrete mesh against which the module was lowered.
    mesh: DeviceMesh,

    /// PJRT platform name of the client against which this program was lowered.
    platform_name: Arc<str>,

    /// PJRT platform version of the client against which this program was lowered.
    platform_version: Arc<str>,

    /// Device kinds in the exact mesh order used by this lowering.
    device_kinds: Arc<[String]>,

    /// Exact Ryft, OpenXLA, and JAX build identity used by this lowering.
    compiler_identity: Arc<str>,

    /// Process-level XLA flags in effect when this program was lowered.
    xla_flags: Arc<str>,
}

impl XlaLoweredProgram {
    /// Returns the textual StableHLO/Shardy module.
    #[inline]
    pub fn stable_hlo(&self) -> &str {
        &self.stable_hlo
    }

    /// Returns the effective logical flat output types, including zero-space leaves erased from the executable ABI.
    #[inline]
    pub fn output_types(&self) -> &[ArrayType] {
        &self.output_types
    }

    /// Returns the device mesh this lowering targets.
    #[inline]
    pub fn mesh(&self) -> &DeviceMesh {
        &self.mesh
    }

    /// Returns this lowering with an opaque feedback-directed profile installed in its exact PJRT compilation
    /// options.
    ///
    /// The StableHLO and runtime signature are unchanged. Because canonical compilation identity includes the
    /// complete PJRT options, the returned lowering receives a distinct in-memory, persistent, and distributed cache
    /// key from the baseline executable.
    pub fn with_feedback_directed_profile(mut self, profile: XlaFeedbackDirectedProfile) -> Self {
        use ryft_pjrt::protos::ExecutableCompilationOptions;

        self.compilation_options
            .executable_build_options
            .get_or_insert_with(ExecutableCompilationOptions::default)
            .fdo_profile = profile.bytes().to_vec();
        self.compilation_options.profile_version = profile.version();
        self
    }
}

const XLA_PERSISTENT_EXECUTABLE_MAGIC: &[u8; 8] = b"RYFTXLA5";
const XLA_PERSISTENT_EXECUTABLE_SCHEMA_VERSION: u32 = 5;
const XLA_PERSISTENT_EXECUTABLE_FEATURE_FLAGS: u64 = 2;
const XLA_PERSISTENT_KEY_SCHEMA_VERSION: u32 = 5;
static XLA_COMPILER_IDENTITY: LazyLock<String> = LazyLock::new(|| {
    format!(
        "ryft-xla/{}/openxla/{}/jax/{}",
        env!("CARGO_PKG_VERSION"),
        ryft_xla_sys::XLA_COMMIT,
        ryft_xla_sys::JAX_COMMIT,
    )
});

/// Typed value returned for a plugin-specific PJRT cost-analysis property.
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum XlaAnalysisValue {
    /// Boolean property.
    Boolean(bool),
    /// Signed integer property.
    Integer(i64),
    /// Signed integer-list property.
    IntegerList(Vec<i64>),
    /// Floating-point property.
    Float(f64),
    /// String property.
    String(String),
}

/// Normalized memory requirements reported by PJRT for one XLA executable.
#[derive(Copy, Clone, Debug, Serialize, PartialEq, Eq)]
pub struct XlaMemoryAnalysis {
    pub device_generated_code_size_in_bytes: usize,
    pub device_input_size_in_bytes: usize,
    pub device_output_size_in_bytes: usize,
    pub device_alias_size_in_bytes: usize,
    pub device_temporary_size_in_bytes: usize,
    pub device_peak_memory_in_bytes: usize,
    pub device_total_memory_in_bytes: usize,
    pub device_total_allocation_bytes: usize,
    pub device_indefinite_allocations: usize,
    pub device_peak_unpadded_heap_bytes: usize,
    pub host_generated_code_size_in_bytes: usize,
    pub host_input_size_in_bytes: usize,
    pub host_output_size_in_bytes: usize,
    pub host_alias_size_in_bytes: usize,
    pub host_temporary_size_in_bytes: usize,
}

/// Backend-neutral-facing, typed analysis of one compiled XLA executable.
///
/// PJRT plugins expose different property sets. Common properties are normalized into optional fields and every raw
/// property is retained in [`Self::properties`]. Unsupported optional queries remain `None`.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct XlaCompilationAnalysis {
    pub floating_point_operations: Option<f64>,
    pub transcendental_operations: Option<f64>,
    pub bytes_accessed: Option<f64>,
    pub memory: Option<XlaMemoryAnalysis>,
    pub generated_code_size_in_bytes: Option<usize>,
    pub replica_count: Option<usize>,
    pub partition_count: Option<usize>,
    pub executable_fingerprint: Option<String>,
    pub compilation_duration: Option<Duration>,
    pub properties: BTreeMap<String, XlaAnalysisValue>,
}

impl XlaCompilationAnalysis {
    /// Returns a deterministic JSON representation suitable for machine-readable diagnostics.
    pub fn to_json(&self) -> Result<String, XlaDomainError> {
        serde_json::to_string(self)
            .map_err(|error| XlaDomainError::InvalidCompilationAnalysis { reason: error.to_string() })
    }
}

impl Display for XlaCompilationAnalysis {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "XLA compilation analysis [flops={}, bytes accessed={}, generated code={} bytes, replicas={}, partitions={}]",
            self.floating_point_operations
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unsupported".into()),
            self.bytes_accessed.map(|value| value.to_string()).unwrap_or_else(|| "unsupported".into()),
            self.generated_code_size_in_bytes
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unsupported".into()),
            self.replica_count.map(|value| value.to_string()).unwrap_or_else(|| "unsupported".into()),
            self.partition_count.map(|value| value.to_string()).unwrap_or_else(|| "unsupported".into()),
        )
    }
}

/// Optimized compiler program returned on explicit inspection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct XlaOptimizedProgram {
    /// PJRT program format (`mlir`, `hlo`, or `hlo_with_config`).
    pub format: &'static str,
    /// Backend-owned optimized program bytes.
    pub bytes: Vec<u8>,
}

#[derive(Serialize)]
struct XlaPersistentKeyV5<'a> {
    schema_version: u32,
    stable_hlo: &'a str,
    compilation_options: &'a [u8],
    signature: PersistentCanonicalArraySignatureV3,
    input_mapping: Vec<Option<u64>>,
    output_mapping: Vec<Option<u64>>,
    input_dimensions: Vec<PersistentBoundaryDimensionV5>,
    output_dimensions: Vec<PersistentBoundaryDimensionV5>,
    requires_assertion_handler: bool,
    donation_flags: &'a [bool],
    capture_count: u64,
    expected_argument_shardings: Vec<PersistentShardingV1>,
    mesh: PersistentDeviceMeshV1,
    device_kinds: &'a [String],
    platform_name: &'a str,
    platform_version: &'a str,
    compiler_identity: &'a str,
    xla_flags: &'a str,
}

#[derive(Serialize, Deserialize)]
struct XlaPersistentExecutableMetadataV5 {
    schema_version: u32,
    feature_flags: u64,
    compilation_options: Vec<u8>,
    signature: PersistentArraySignatureV3,
    input_mapping: Vec<Option<u64>>,
    output_mapping: Vec<Option<u64>>,
    input_dimensions: Vec<PersistentBoundaryDimensionV5>,
    output_dimensions: Vec<PersistentBoundaryDimensionV5>,
    requires_assertion_handler: bool,
    donation_flags: Vec<bool>,
    capture_count: u64,
    expected_argument_shardings: Vec<PersistentShardingV1>,
    mesh: PersistentDeviceMeshV1,
    device_kinds: Vec<String>,
    replica_count: u64,
    partition_count: u64,
    device_assignment: Vec<u64>,
    platform_name: String,
    platform_version: String,
    compiler_identity: String,
    xla_flags: String,
    compilation_duration_nanoseconds: Option<u64>,
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

#[derive(Serialize, Deserialize)]
struct PersistentArraySignatureV3 {
    variables: Vec<PersistentDimensionVariableV3>,
    input_types: Vec<PersistentArrayTypeV3>,
    output_types: Vec<PersistentArrayTypeV3>,
}

#[derive(Serialize)]
struct PersistentCanonicalArraySignatureV3 {
    variables: Vec<PersistentCanonicalDimensionVariableV3>,
    input_types: Vec<PersistentArrayTypeV3>,
    output_types: Vec<PersistentArrayTypeV3>,
}

#[derive(Serialize, Deserialize)]
struct PersistentDimensionVariableV3 {
    name: String,
    lower: u64,
    upper: Option<u64>,
}

#[derive(Serialize)]
struct PersistentCanonicalDimensionVariableV3 {
    lower: u64,
    upper: Option<u64>,
}

#[derive(Serialize, Deserialize)]
struct PersistentArrayTypeV3 {
    data_type: u8,
    shape: Vec<PersistentDimensionV3>,
    layout: Option<PersistentLayoutV1>,
    sharding: Option<PersistentShardingV1>,
    memory: PersistentMemoryV1,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
enum PersistentDimensionV3 {
    Static(u64),
    Dynamic { variable: u64 },
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum PersistentLayoutV1 {
    Tiled { minor_to_major: Vec<u64>, tiles: Vec<Vec<PersistentTileDimensionV1>> },
    Strided { strides: Vec<i64> },
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
enum PersistentTileDimensionV1 {
    Sized(u64),
    Combined,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", content = "pinned", rename_all = "snake_case")]
enum PersistentMemoryV1 {
    Device,
    Host(bool),
}

#[derive(Serialize, Deserialize)]
struct PersistentShardingV1 {
    mesh: PersistentLogicalMeshV1,
    dimensions: Vec<PersistentShardingDimensionV1>,
    unreduced_axes: Vec<String>,
    reduced_axes: Vec<String>,
    varying_manual_axes: Vec<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", content = "axes", rename_all = "snake_case")]
enum PersistentShardingDimensionV1 {
    Replicated,
    Sharded(Vec<String>),
    Unconstrained,
}

#[derive(Serialize, Deserialize)]
struct PersistentLogicalMeshV1 {
    axes: Vec<PersistentMeshAxisV1>,
}

#[derive(Serialize, Deserialize)]
struct PersistentMeshAxisV1 {
    name: String,
    size: u64,
    r#type: u8,
}

#[derive(Serialize, Deserialize)]
struct PersistentDeviceMeshV1 {
    logical_mesh: PersistentLogicalMeshV1,
    devices: Vec<PersistentDeviceV1>,
}

#[derive(Serialize, Deserialize)]
struct PersistentDeviceV1 {
    id: u64,
    process_index: u64,
}

impl PersistentArraySignatureV3 {
    /// Encodes input and output types while alpha-normalizing their shared dynamic dimension variables by first
    /// occurrence.
    fn encode(input_types: &[ArrayType], output_types: &[ArrayType]) -> Self {
        let mut variables = Vec::<DimensionVariable>::new();
        let mut encode_type = |r#type: &ArrayType| {
            let shape = r#type
                .shape()
                .dimensions()
                .iter()
                .map(|dimension| match dimension {
                    Dimension::Static(value) => PersistentDimensionV3::Static(*value as u64),
                    Dimension::Dynamic(variable) => {
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
    fn decode(self) -> Result<(Vec<ArrayType>, Vec<ArrayType>), XlaDomainError> {
        let variables = self
            .variables
            .into_iter()
            .map(|variable| {
                let bounds = DimensionBounds::new(
                    checked_usize(variable.lower)?,
                    variable.upper.map(checked_usize).transpose()?,
                )
                .map_err(|error| persistent_error(error.to_string()))?;
                Ok(DimensionVariable::new(variable.name, bounds))
            })
            .collect::<Result<Vec<_>, XlaDomainError>>()?;
        let decode_type = |r#type: PersistentArrayTypeV3| -> Result<ArrayType, XlaDomainError> {
            let shape = Shape::new(
                r#type
                    .shape
                    .into_iter()
                    .map(|dimension| match dimension {
                        PersistentDimensionV3::Static(value) => checked_usize(value).map(Dimension::Static),
                        PersistentDimensionV3::Dynamic { variable } => variables
                            .get(checked_usize(variable)?)
                            .cloned()
                            .map(Dimension::Dynamic)
                            .ok_or_else(|| persistent_error("array type references an unknown dimension variable")),
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
                .map_err(|error| persistent_error(error.to_string()))
        };
        let input_types = self.input_types.into_iter().map(&decode_type).collect::<Result<Vec<_>, _>>()?;
        let output_types = self.output_types.into_iter().map(decode_type).collect::<Result<Vec<_>, _>>()?;
        Ok((input_types, output_types))
    }
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
    type Error = XlaDomainError;

    fn try_from(value: PersistentLayoutV1) -> Result<Self, Self::Error> {
        Ok(match value {
            PersistentLayoutV1::Tiled { minor_to_major, tiles } => TiledLayout::new(
                minor_to_major.into_iter().map(checked_usize).collect::<Result<Vec<_>, _>>()?,
                tiles
                    .into_iter()
                    .map(|tile| {
                        tile.into_iter()
                            .map(|dimension| match dimension {
                                PersistentTileDimensionV1::Sized(size) => checked_usize(size).map(TileDimension::Sized),
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
                        isize::try_from(stride).map_err(|_| persistent_error("layout stride does not fit in isize"))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
            .into(),
        })
    }
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
            reduced_axes: value.reduced_axes().iter().cloned().collect(),
            varying_manual_axes: value.varying_manual_axes().iter().cloned().collect(),
        }
    }
}

impl TryFrom<PersistentShardingV1> for Sharding {
    type Error = XlaDomainError;

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
        .and_then(|sharding| sharding.with_reduced_axes(value.reduced_axes))
        .and_then(|sharding| sharding.with_varying_manual_axes(value.varying_manual_axes))
        .map_err(|error| persistent_error(error.to_string()))
    }
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
    type Error = XlaDomainError;

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
                        value => return Err(persistent_error(format!("unknown mesh axis type {value}"))),
                    };
                    MeshAxis::new(axis.name, checked_usize(axis.size)?, r#type)
                        .map_err(|error| persistent_error(error.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
        .map_err(|error| persistent_error(error.to_string()))
    }
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
    type Error = XlaDomainError;

    fn try_from(value: PersistentDeviceMeshV1) -> Result<Self, Self::Error> {
        DeviceMesh::new(
            LogicalMesh::try_from(value.logical_mesh)?,
            value
                .devices
                .into_iter()
                .map(|device| Ok(Device::new(checked_usize(device.id)?, checked_usize(device.process_index)?)))
                .collect::<Result<Vec<_>, XlaDomainError>>()?,
        )
        .map_err(|error| persistent_error(error.to_string()))
    }
}

fn persistent_error(reason: impl Into<String>) -> XlaDomainError {
    XlaDomainError::InvalidPersistentExecutable { reason: reason.into() }
}

fn checked_usize(value: u64) -> Result<usize, XlaDomainError> {
    usize::try_from(value).map_err(|_| persistent_error(format!("value {value} does not fit in usize")))
}

/// Converts an in-memory executable signature mapping to its stable persistent representation.
fn persistent_mapping(mapping: &[Option<usize>]) -> Result<Vec<Option<u64>>, XlaDomainError> {
    mapping
        .iter()
        .map(|index| {
            index
                .map(|index| {
                    u64::try_from(index)
                        .map_err(|_| persistent_error(format!("physical signature index {index} does not fit in u64")))
                })
                .transpose()
        })
        .collect()
}

/// Encodes the hidden bounded-input extent slots of one executable signature.
fn persistent_input_dimensions(signature: &XlaExecutableSignature) -> Vec<PersistentBoundaryDimensionV5> {
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
fn persistent_output_dimensions(signature: &XlaExecutableSignature) -> Vec<PersistentBoundaryDimensionV5> {
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

fn decode_data_type(value: u8) -> Result<DataType, XlaDomainError> {
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
        value => return Err(persistent_error(format!("unknown data type tag {value}"))),
    })
}

/// Exact process-local compilation key for an XLA lowering.
///
/// Unlike the former call-site fingerprint, this key carries the complete StableHLO computation and every
/// execution-relevant field retained by [`XlaLoweredProgram`]. Equality therefore means that cached compiled
/// artifacts are interchangeable, even when unrelated call sites happen to share the same input signature.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct XlaCompilationKey {
    canonical_bytes: Arc<[u8]>,
}

/// Loaded PJRT executable plus the metadata required to invoke it safely.
#[derive(Clone)]
pub struct XlaCompiledProgram<'c> {
    executable: Arc<LoadedExecutable<'c>>,
    input_types: Arc<[ArrayType]>,
    output_types: Arc<[ArrayType]>,
    signature: XlaExecutableSignature,
    requires_assertion_handler: bool,
    donation_flags: Arc<[bool]>,
    capture_count: usize,
    expected_argument_shardings: Arc<[Sharding]>,
    mesh: DeviceMesh,
    compilation_options: CompilationOptions,
    platform_name: Arc<str>,
    platform_version: Arc<str>,
    device_kinds: Arc<[String]>,
    compiler_identity: Arc<str>,
    xla_flags: Arc<str>,
    compilation_duration: Option<Duration>,
    analysis: Arc<OnceLock<Result<XlaCompilationAnalysis, String>>>,
}

#[derive(Copy, Clone)]
struct XlaInvocationMetadata<'a> {
    input_types: &'a [ArrayType],
    output_types: &'a [ArrayType],
    signature: &'a XlaExecutableSignature,
    donation_flags: &'a [bool],
    capture_count: usize,
    expected_argument_shardings: &'a [Sharding],
    mesh: &'a DeviceMesh,
}

impl<'a, 'c> From<&'a XlaCompiledProgram<'c>> for XlaInvocationMetadata<'a> {
    fn from(program: &'a XlaCompiledProgram<'c>) -> Self {
        Self {
            input_types: &program.input_types,
            output_types: &program.output_types,
            signature: &program.signature,
            donation_flags: &program.donation_flags,
            capture_count: program.capture_count,
            expected_argument_shardings: &program.expected_argument_shardings,
            mesh: &program.mesh,
        }
    }
}

fn validate_xla_replacement_metadata(
    current: XlaInvocationMetadata<'_>,
    replacement: XlaInvocationMetadata<'_>,
) -> Result<(), XlaDomainError> {
    let incompatible_field = if current.input_types != replacement.input_types {
        Some("input types")
    } else if current.output_types != replacement.output_types {
        Some("output types")
    } else if current.signature != replacement.signature {
        Some("executable signature")
    } else if current.donation_flags != replacement.donation_flags {
        Some("donation flags")
    } else if current.capture_count != replacement.capture_count {
        Some("capture count")
    } else if current.expected_argument_shardings != replacement.expected_argument_shardings {
        Some("argument shardings")
    } else if current.mesh != replacement.mesh {
        Some("device mesh")
    } else {
        None
    };
    match incompatible_field {
        Some(field) => Err(XlaDomainError::InvalidCompilationOptions {
            reason: format!("replacement executable has incompatible {field}"),
        }),
        None => Ok(()),
    }
}

impl<'c> XlaCompiledProgram<'c> {
    /// Returns the loaded PJRT executable.
    #[inline]
    pub fn executable(&self) -> &LoadedExecutable<'c> {
        &self.executable
    }

    /// Returns the logical flat output types in user-visible order, including reconstructed zero-space leaves.
    #[inline]
    pub fn output_types(&self) -> &[ArrayType] {
        &self.output_types
    }

    /// Returns the mesh the compiled program runs against.
    #[inline]
    pub fn mesh(&self) -> &DeviceMesh {
        &self.mesh
    }

    /// Returns the backend's optimized program only when explicitly requested.
    pub fn optimized_program(&self) -> Result<XlaOptimizedProgram, XlaDomainError> {
        match self.executable.executable()?.optimized_program()? {
            PjrtProgram::Mlir { bytecode } => Ok(XlaOptimizedProgram { format: "mlir", bytes: bytecode }),
            PjrtProgram::Hlo { proto } => Ok(XlaOptimizedProgram { format: "hlo", bytes: proto }),
            PjrtProgram::HloWithConfig { proto } => Ok(XlaOptimizedProgram { format: "hlo_with_config", bytes: proto }),
        }
    }
}

impl<'c> XlaDomain<'c> {
    /// Ensures that process-local runtime support required by one compiled program is available.
    fn ensure_runtime_requirements(
        &self,
        requires_assertion_handler: bool,
        platform_name: &str,
    ) -> Result<(), XlaDomainError> {
        if !requires_assertion_handler {
            return Ok(());
        }
        let supported = platform_name.eq_ignore_ascii_case("cpu")
            || (cfg!(any(feature = "cuda-12", feature = "cuda-13")) && platform_name.eq_ignore_ascii_case("cuda"));
        if !supported {
            return Err(XlaDomainError::UnsupportedRuntimeAssertionPlatform { platform: platform_name.to_string() });
        }
        super::assertions::ensure_assertion_handler_registered(self.client()?)?;
        Ok(())
    }

    /// Validates that this domain's PJRT client owns `program`.
    fn validate_xla_program_owner(&self, program: &XlaCompiledProgram<'c>) -> Result<(), XlaDomainError> {
        if program.executable.is_owned_by(self.client()?) {
            Ok(())
        } else {
            Err(XlaDomainError::ExecutableClientMismatch)
        }
    }

    /// Enqueues `program` and returns its possibly still pending flat outputs together with a whole-execution fence.
    ///
    /// For programs without bounded-dynamic boundaries, this call only enqueues device work: the returned arrays and
    /// fence resolve asynchronously. Bounded-dynamic boundaries weaken that guarantee in two ways: below-bound inputs
    /// perform blocking device-to-host copies and bound-shaped uploads before the launch, and each bounded-dynamic
    /// output blocks on the device-to-host readback of its hidden extent scalar because the logical output type cannot
    /// be constructed without the runtime extent.
    pub(crate) fn execute_compiled_async(
        &self,
        program: &XlaCompiledProgram<'c>,
        inputs: Vec<Array<'c>>,
    ) -> Result<Execution<Vec<Array<'c>>>, XlaDomainError> {
        self.validate_xla_program_owner(program)?;
        self.ensure_runtime_requirements(program.requires_assertion_handler, &program.platform_name)?;
        if inputs.len() != program.input_types.len() {
            return Err(XlaDomainError::InvalidCompilationOptions {
                reason: format!(
                    "compiled program expects {} flat argument(s), including {} capture(s), but got {}",
                    program.input_types.len(),
                    program.capture_count,
                    inputs.len(),
                ),
            });
        }
        let actual_input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let refinement_input_types = program
            .input_types
            .iter()
            .zip(actual_input_types.iter())
            .map(|(declared, actual)| {
                // Sharding is normalized separately at the executable boundary. Preserve every other observed type
                // component so refinement still rejects a wrong data type, layout, memory space, rank, or extent.
                actual.clone().with_sharding(declared.sharding().cloned()).map_err(ArrayError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let input_refinements =
            <ArrayType as Type>::Refinements::establish(program.input_types.iter(), refinement_input_types.iter())
                .map_err(ProgramError::from)?;
        let physical_input_types = program.signature.physical_input_types(&program.input_types);
        let inputs = materialize_bounded_dynamic_inputs(
            self.client()?,
            &program.signature,
            &program.input_types,
            actual_input_types.as_slice(),
            inputs,
        )?
        .inputs;
        let inputs = materialize_zero_space_carriers(self.client()?, physical_input_types.as_slice(), inputs)?;
        let inputs = reshard_inputs_if_needed(self, &program.mesh, &program.expected_argument_shardings, inputs)?;
        let logical_donation_flags = std::iter::repeat_n(false, program.capture_count)
            .chain(program.donation_flags.iter().copied())
            .collect::<Vec<_>>();
        let mut donation_flags = program.signature.project_inputs(logical_donation_flags.as_slice());
        donation_flags.extend(std::iter::repeat_n(false, program.signature.input_dimensions().len()));
        for (donation, input_type) in donation_flags.iter_mut().zip(physical_input_types.iter()) {
            if input_type.data_type().is_zero() {
                *donation = false;
            }
        }
        let physical_output_count =
            program.signature.output_mapping().iter().flatten().count() + program.signature.output_dimensions().len();
        let execution =
            execute_pjrt_buffers(&program.executable, inputs, donation_flags.as_slice(), physical_output_count)?;
        let (physical_outputs, fence) = execution.into_parts();
        let mut physical_outputs = physical_outputs.into_iter().map(Some).collect::<Vec<_>>();
        let scalar_type = ArrayType::scalar(DataType::I64).replicated(&program.mesh).map_err(ArrayError::from)?;
        let mut output_extents = BTreeMap::new();
        for output_dimension in program.signature.output_dimensions() {
            let scalar = Array::from_canonical_addressable_buffers(
                self.client()?,
                scalar_type.clone(),
                program.mesh.clone(),
                physical_outputs[output_dimension.physical_output_index()].take().unwrap(),
            )?
            .with_execution_fence(fence.clone());
            let bytes = materialize_dense_array_bytes(&scalar)?;
            let extent = bytes
                .get(..size_of::<i64>())
                .map(|bytes| i64::from_ne_bytes(bytes.try_into().unwrap()))
                .and_then(|extent| usize::try_from(extent).ok())
                .ok_or_else(|| ProgramError::InvalidArgument {
                    message: format!(
                        "runtime extent for output {} axis {} is negative, missing, or exceeds usize",
                        output_dimension.logical_output_index(),
                        output_dimension.axis(),
                    ),
                })?;
            output_extents.insert((output_dimension.logical_output_index(), output_dimension.axis()), extent);
        }

        let mut outputs = Vec::with_capacity(program.output_types.len());
        for (logical_index, (mapping, output_type)) in
            program.signature.output_mapping().iter().zip(program.output_types.iter()).enumerate()
        {
            match mapping {
                Some(physical_index) => {
                    let dimensions = output_type
                        .shape()
                        .dimensions()
                        .iter()
                        .enumerate()
                        .map(|(axis, dimension)| {
                            dimension
                                .value()
                                .or_else(|| output_extents.get(&(logical_index, axis)).copied())
                                .map_or_else(
                                    || {
                                        Err(ProgramError::MalformedProgram(format!(
                                            "executable omitted runtime extent for output {logical_index} axis {axis}",
                                        )))
                                    },
                                    |extent| Ok(Dimension::Static(extent)),
                                )
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let output_type = output_type.clone().with_shape(Shape::new(dimensions));
                    let output_type = match output_type.sharding() {
                        Some(_) => output_type,
                        None => output_type.replicated(&program.mesh).map_err(ArrayError::from)?,
                    };
                    outputs.push(
                        Array::from_canonical_addressable_buffers(
                            self.client()?,
                            output_type,
                            program.mesh.clone(),
                            physical_outputs[*physical_index].take().unwrap(),
                        )?
                        .with_execution_fence(fence.clone()),
                    );
                }
                None => outputs.push(
                    Array::from_zero_space(self.client()?, output_type.clone(), program.mesh.clone())?
                        .with_execution_fence(fence.clone()),
                ),
            }
        }
        if physical_outputs.iter().any(Option::is_some) {
            return Err(
                ProgramError::MalformedProgram("executable returned an unclaimed physical output".to_string()).into()
            );
        }
        let mut closed_identities = Vec::new();
        for (_, identity) in program.input_types.iter().chain(program.output_types.iter()).flat_map(Type::identities) {
            if !closed_identities.contains(identity) {
                closed_identities.push(identity.clone());
            }
        }
        input_refinements
            .validate(
                program.output_types.iter(),
                outputs.iter().map(|output| output.r#type().into_owned()),
                closed_identities.as_slice(),
            )
            .map_err(ProgramError::from)?;
        Ok(Execution::new(outputs, fence))
    }
}

impl<'c> CompilationDomain for XlaDomain<'c> {
    type DispatchKey = XlaDispatchKey;
    type LoweredProgram = XlaLoweredProgram;
    type CompiledProgram = XlaCompiledProgram<'c>;
    type Options = XlaOptions;
    type Error = XlaDomainError;

    fn dispatch_signature(
        &self,
        input_types: Vec<ArrayIrType>,
        options: &Self::Options,
    ) -> Result<(Self::DispatchKey, Arc<[ArrayIrType]>), Self::Error> {
        // Reference inputs are rejected before any mesh- or option-specific constraint so they always receive the
        // targeted discharge diagnostic instead of an unrelated bucketing or mesh error.
        if let Some(reference_input) = input_types.iter().find(|r#type| r#type.is_reference()) {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "references must be discharged before XLA compilation, but dispatch input has type \
                    `{reference_input}`",
                ),
            }
            .into());
        }
        let Some(bucketing) = options.input_bound_bucketing else {
            let input_types: Arc<[ArrayIrType]> = input_types.into();
            return Ok((XlaDispatchKey(XlaDispatchKeyKind::Exact(input_types.clone())), input_types));
        };
        if options.mesh.devices().len() > 1 {
            return Err(XlaDomainError::InvalidCompilationOptions {
                reason: "input-bound bucketing is unsupported on multi-device meshes because bucketed dynamic \
                         signatures require disabling the Shardy partitioner"
                    .to_string(),
            });
        }
        let mut bucketed_array_types = Vec::with_capacity(input_types.len());
        for (input_index, r#type) in input_types.iter().enumerate() {
            let array_type = match r#type {
                ArrayIrType::Array(array_type) => array_type,
                ArrayIrType::Dimension(_) => {
                    return Err(XlaDomainError::InvalidCompilationOptions {
                        reason: format!(
                            "input-bound bucketing only supports array inputs, but input {} has first-class dimension \
                             type {}",
                            input_index, r#type,
                        ),
                    });
                }
                ArrayIrType::Reference(_) => {
                    unreachable!("reference inputs are rejected at the `dispatch_signature` entry guard")
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
                            XlaInputBoundBucketing::PowerOfTwo => extent.checked_next_power_of_two(),
                        }
                        .and_then(|bound| bound.checked_add(1))
                        .ok_or_else(|| ProgramError::InvalidArgument {
                            message: format!(
                                "cannot select a finite input-bound bucket for input {input_index} axis {axis} with \
                                 extent {extent}",
                            ),
                        })?;
                        let bounds = DimensionBounds::new(0, Some(bound))?;
                        Ok(Dimension::Dynamic(DimensionVariable::new(
                            format!("input_{input_index}_axis_{axis}"),
                            bounds,
                        )))
                    }
                    Dimension::Dynamic(variable) => Ok(Dimension::Dynamic(variable.clone())),
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            bucketed_array_types.push(array_type.clone().with_shape(Shape::new(dimensions)));
        }
        let key = BucketedDispatchSignature::new(bucketed_array_types.as_slice());
        let input_types = bucketed_array_types.into_iter().map(ArrayIrType::Array).collect::<Vec<_>>();
        Ok((XlaDispatchKey(XlaDispatchKeyKind::Bucketed(key)), input_types.into()))
    }

    fn stage<Request>(
        &self,
        mut request: Request,
    ) -> Result<StagedFunction<Self, Request::Input, Request::Output>, Self::Error>
    where
        Request: StageRequest<Self>,
    {
        // Reference-typed boundary leaves must receive the targeted discharge diagnostic before any sharding
        // override is applied: the override path projects leaves to arrays and would otherwise fail with a generic
        // `expected array type` error.
        if let Some(reference) = request.input_types().parameters().find(|r#type| r#type.is_reference()) {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "references must be discharged before XLA compilation, but staged input has type `{reference}`",
                ),
            }
            .into());
        }
        if let Some(in_shardings) = request.options().in_shardings.clone() {
            let input_types = apply_signature_shardings(
                request.input_types().parameters().cloned().collect(),
                Some(in_shardings.as_slice()),
                "in",
            )?;
            request.replace_input_types(input_types)?;
        }
        request.trace(|options, output_types| {
            if let Some(reference) = output_types.iter().find(|r#type| r#type.is_reference()) {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "references must be discharged before XLA compilation, but staged output has type \
                        `{reference}`",
                    ),
                }
                .into());
            }
            apply_signature_shardings(output_types, options.out_shardings.as_deref(), "out")
        })
    }

    fn lower<Request>(
        &self,
        staged: Request,
    ) -> Result<ryft_core::compilation::LoweredFunction<Self, Request::Input, Request::Output>, Self::Error>
    where
        Request: LoweringRequest<Self>,
    {
        let lifted_program = staged.lifted_program()?;
        let program = self.lower_xla_program(
            &lifted_program,
            staged.staged().source_program().captures().len(),
            staged.staged().options(),
        )?;
        let output_types = program.output_types().to_vec();
        validate_output_types(staged.staged().output_types(), &output_types)?;
        Ok(staged.into_lowered(program, output_types.into_iter().map(Into::into).collect()))
    }

    fn compile<Request>(
        &self,
        lowered: Request,
    ) -> Result<ryft_core::compilation::CompiledFunction<Self, Request::Input, Request::Output>, Self::Error>
    where
        Request: CompileRequest<Self>,
    {
        self.cache.compile_request(
            self,
            lowered,
            |program| self.compile_xla_program(program),
            |program| program.output_types().iter().cloned().map(Into::into).collect(),
        )
    }

    fn call<Request>(&self, request: Request) -> Result<Request::RuntimeOutput, Self::Error>
    where
        Request: CallRequest<Self>,
    {
        let executable = request.executable().clone();
        if request.inputs().len() != executable.input_types().len() {
            return Err(ProgramError::InvalidInputCount {
                expected: executable.input_types().len(),
                actual: request.inputs().len(),
            }
            .into());
        }
        for (declared, actual) in executable.input_types().iter().zip(request.inputs().iter().map(Typed::r#type)) {
            validate_xla_input_type(
                <&ArrayType>::try_from(declared).map_err(ProgramError::from)?,
                <&ArrayType>::try_from(actual.as_ref()).map_err(ProgramError::from)?,
            )?;
        }
        let output_types = executable.compiled_program().output_types().to_vec();
        let arguments = request
            .into_arguments()
            .into_iter()
            .map(ValueProjection::<ArrayType>::into_projected)
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProgramError::from)?;
        let outputs = self.execute_xla_program(executable.compiled_program(), arguments)?;
        validate_runtime_outputs(&output_types, &outputs)?;
        Request::reconstruct(&executable, outputs.into_iter().map(ArrayIrValue::Array).collect())
    }
}

impl<'c> XlaDomain<'c> {
    fn lower_xla_program(
        &self,
        program: &FlatXlaProgram,
        capture_count: usize,
        options: &XlaOptions,
    ) -> Result<XlaLoweredProgram, XlaDomainError> {
        if capture_count > program.input_count() {
            return Err(XlaDomainError::InvalidCompilationOptions {
                reason: format!(
                    "capture_count is {capture_count}, but the program has only {} flat input(s)",
                    program.input_count(),
                ),
            });
        }
        // Avoid cloning the dominant reference-free program path. Its discharge artifact is the identity with an
        // empty external-state list, so the same metadata can cross the lowering boundary by borrow.
        if !contains_unresolved_references(program) {
            return self.lower_verified_reference_free_xla_program(program, capture_count, options);
        }
        // Discharge the exact lifted program so attached-region capture references resolve against the same lexical
        // capture prefixes used during staging. Programs that actually contain references are cloned because
        // discharge consumes and rewrites them; the reference-free identity path above remains borrowed.
        let discharged = program.clone().discharge_references_with_lifted_captures(capture_count)?;
        self.lower_discharged_xla_program(&discharged, capture_count, options)
    }

    /// Lowers one validated reference-discharge artifact through the ordinary array-only XLA path.
    fn lower_discharged_xla_program(
        &self,
        discharged: &DischargedReferenceProgram<XlaConstant, XlaOperation>,
        capture_count: usize,
        options: &XlaOptions,
    ) -> Result<XlaLoweredProgram, XlaDomainError> {
        if !discharged.external_states().is_empty() {
            return Err(XlaDomainError::UnsupportedReferenceAbi {
                reason: "external reference state requires a stateful XLA invocation boundary".to_string(),
            });
        }
        if discharged.public_output_count() != discharged.program().output_count() {
            return Err(ProgramError::MalformedProgram(format!(
                "local XLA reference discharge retained {} total outputs for {} public outputs",
                discharged.program().output_count(),
                discharged.public_output_count(),
            ))
            .into());
        }
        // The backend verifier remains independent from core discharge and runs before boundary projection,
        // partitioner selection, or StableHLO construction. The reference scan runs first so a malformed artifact
        // containing both a reference and ordered state receives the more precise reference diagnostic.
        if contains_unresolved_references(discharged.program()) {
            return Err(ProgramError::UnsupportedOperation {
                message: "references must be discharged before XLA compilation".to_string(),
            }
            .into());
        }
        self.lower_verified_reference_free_xla_program(discharged.program(), capture_count, options)
    }

    /// Lowers one program after an independent verifier has established that it contains no references.
    fn lower_verified_reference_free_xla_program(
        &self,
        program: &FlatXlaProgram,
        capture_count: usize,
        options: &XlaOptions,
    ) -> Result<XlaLoweredProgram, XlaDomainError> {
        if contains_unresolved_state(program) {
            return Err(ProgramError::UnsupportedOperation {
                message: "state must be discharged before XLA compilation".to_string(),
            }
            .into());
        }
        validate_data_dependent_compilation(program)?;
        let input_types = program.input_types();
        let (capture_types, public_input_types) = input_types.split_at(capture_count);
        if let Some(donation_flags) = &options.donation_flags
            && donation_flags.len() != public_input_types.len()
        {
            return Err(XlaDomainError::InvalidCompilationOptions {
                reason: format!(
                    "donation_flags has {} entries but the program has {} flat public input(s)",
                    donation_flags.len(),
                    public_input_types.len(),
                ),
            });
        }
        let mut donation_flags =
            options.donation_flags.clone().unwrap_or_else(|| vec![false; public_input_types.len()]);

        let effective_capture_types = capture_types
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProgramError::from)?;
        let effective_public_input_types = public_input_types
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProgramError::from)?;
        let program_output_types =
            apply_signature_shardings(program.output_types().to_vec(), options.out_shardings.as_deref(), "out")?;
        let effective_input_types = effective_capture_types
            .iter()
            .cloned()
            .chain(effective_public_input_types.iter().cloned())
            .collect::<Vec<_>>();
        let output_types = program_output_types
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProgramError::from)?;
        let logical_argument_shardings = effective_input_types
            .iter()
            .map(|array_type| {
                array_type.sharding().cloned().unwrap_or_else(|| {
                    Sharding::replicated(options.mesh.logical_mesh().clone(), array_type.shape().rank())
                })
            })
            .collect::<Vec<_>>();
        let result_shardings =
            output_types.iter().map(|array_type| array_type.sharding().cloned()).collect::<Option<Vec<_>>>();
        // Shardy rejects bounded-dynamic tensors anywhere in the module, not only at the computation boundary, so a
        // program whose boundary is static but whose interior derives a dynamic extent through a gateway operation
        // still has to take the replica path. This is why the whole region arena is inspected instead of just the
        // boundary types, which it already subsumes.
        let use_shardy_partitioner = has_only_static_array_types(program);
        // A `shard_map` body only ever has static boundary types, so it can coexist with dynamic tensors staged
        // elsewhere in the same program. Its `sdy.manual_computation` lowering is meaningless without Shardy on any
        // device count, and bounded-dynamic tensors force Shardy off for the whole module.
        if !use_shardy_partitioner && contains_shard_map(program) {
            return Err(XlaDomainError::InvalidCompilationOptions {
                reason: "bounded-dynamic programs cannot contain shard_map regions because sdy.manual_computation \
                         requires the Shardy partitioner"
                    .to_string(),
            });
        }
        if !use_shardy_partitioner && options.mesh.devices().len() > 1 {
            if logical_argument_shardings
                .iter()
                .chain(output_types.iter().filter_map(ArrayType::sharding))
                .any(|sharding| !sharding.is_replicated())
            {
                return Err(XlaDomainError::InvalidCompilationOptions {
                    reason: "bounded-dynamic multi-device programs currently require fully replicated shardings \
                             because Shardy rejects dynamic tensors"
                        .to_string(),
                });
            }
        }
        let compilation_options = jit_compilation_options(
            self.compilation_options.as_ref(),
            options.mesh.devices().len(),
            use_shardy_partitioner,
            options.feedback_directed_profile.as_ref(),
        );
        // The target platform gates platform-specific lowerings such as fused attention; a failed
        // platform query degrades to the portable lowerings instead of failing compilation.
        let target_platform =
            self.client.and_then(|client| client.platform_name().ok()).map(|platform| platform.into_owned());
        // Shardy currently rejects bounded-dynamic tensors. Fully replicated dynamic programs need no partitioning,
        // so lower them without Shardy metadata and execute one replica per mesh device instead.
        let lowered_argument_shardings = use_shardy_partitioner.then_some(logical_argument_shardings.as_slice());
        let lowered_result_shardings = if use_shardy_partitioner { result_shardings.as_deref() } else { None };
        let lowered_module = crate::experimental::lowering::lower_mlir_module_for_program(
            program,
            effective_capture_types.as_slice(),
            &effective_public_input_types,
            &output_types,
            "main",
            lowered_argument_shardings,
            lowered_result_shardings,
            target_platform.as_deref(),
        )
        .map_err(|error| XlaDomainError::Lowering(error.into()))?;
        let (stable_hlo, signature, requires_assertion_handler) = lowered_module.into_parts();
        for (mapping, donation) in signature.input_mapping()[capture_count..].iter().zip(donation_flags.iter_mut()) {
            if mapping.is_none() {
                *donation = false;
            }
        }
        let expected_argument_shardings = signature.physical_input_shardings(logical_argument_shardings.as_slice());
        let client = self.client()?;

        Ok(XlaLoweredProgram {
            stable_hlo: stable_hlo.into(),
            compilation_options,
            input_types: effective_input_types.into(),
            output_types: output_types.into(),
            signature,
            requires_assertion_handler,
            donation_flags: donation_flags.into(),
            capture_count,
            expected_argument_shardings: expected_argument_shardings.into(),
            mesh: options.mesh.clone(),
            platform_name: client.platform_name()?.into_owned().into(),
            platform_version: client.platform_version()?.into_owned().into(),
            device_kinds: ordered_device_kinds(client, &options.mesh)?.into(),
            compiler_identity: XLA_COMPILER_IDENTITY.as_str().into(),
            xla_flags: std::env::var("XLA_FLAGS").unwrap_or_default().into(),
        })
    }

    fn xla_compilation_key(program: &XlaLoweredProgram) -> Result<XlaCompilationKey, XlaDomainError> {
        let compilation_options = canonical_compilation_options_bytes(&program.compilation_options);
        let key = XlaPersistentKeyV5 {
            schema_version: XLA_PERSISTENT_KEY_SCHEMA_VERSION,
            stable_hlo: &program.stable_hlo,
            compilation_options: compilation_options.as_slice(),
            signature: PersistentArraySignatureV3::encode(&program.input_types, &program.output_types).into_canonical(),
            input_mapping: persistent_mapping(program.signature.input_mapping())?,
            output_mapping: persistent_mapping(program.signature.output_mapping())?,
            input_dimensions: persistent_input_dimensions(&program.signature),
            output_dimensions: persistent_output_dimensions(&program.signature),
            requires_assertion_handler: program.requires_assertion_handler,
            donation_flags: &program.donation_flags,
            capture_count: program.capture_count as u64,
            expected_argument_shardings: program
                .expected_argument_shardings
                .iter()
                .map(PersistentShardingV1::from)
                .collect(),
            mesh: PersistentDeviceMeshV1::from(&program.mesh),
            device_kinds: &program.device_kinds,
            platform_name: &program.platform_name,
            platform_version: &program.platform_version,
            compiler_identity: &program.compiler_identity,
            xla_flags: &program.xla_flags,
        };
        let canonical_bytes = serde_json::to_vec(&key)
            .map_err(|error| persistent_error(format!("failed to encode persistent key: {error}")))?;
        Ok(XlaCompilationKey { canonical_bytes: canonical_bytes.into() })
    }

    pub(crate) fn compile_xla_program(
        &self,
        program: &XlaLoweredProgram,
    ) -> Result<XlaCompiledProgram<'c>, XlaDomainError> {
        self.ensure_runtime_requirements(program.requires_assertion_handler, &program.platform_name)?;
        let pjrt_program = PjrtProgram::Mlir { bytecode: program.stable_hlo.as_bytes().to_vec() };
        let compilation_start = Instant::now();
        let executable = self.client()?.compile(&pjrt_program, &program.compilation_options)?;
        let compilation_duration = compilation_start.elapsed();
        Ok(XlaCompiledProgram {
            executable: Arc::new(executable),
            input_types: Arc::clone(&program.input_types),
            output_types: Arc::clone(&program.output_types),
            signature: program.signature.clone(),
            requires_assertion_handler: program.requires_assertion_handler,
            donation_flags: Arc::clone(&program.donation_flags),
            capture_count: program.capture_count,
            expected_argument_shardings: Arc::clone(&program.expected_argument_shardings),
            mesh: program.mesh.clone(),
            compilation_options: program.compilation_options.clone(),
            platform_name: Arc::clone(&program.platform_name),
            platform_version: Arc::clone(&program.platform_version),
            device_kinds: Arc::clone(&program.device_kinds),
            compiler_identity: Arc::clone(&program.compiler_identity),
            xla_flags: Arc::clone(&program.xla_flags),
            compilation_duration: Some(compilation_duration),
            analysis: Arc::new(OnceLock::new()),
        })
    }

    pub(crate) fn validate_xla_replacement(
        &self,
        current: &XlaCompiledProgram<'c>,
        replacement: &XlaCompiledProgram<'c>,
    ) -> Result<(), XlaDomainError> {
        self.validate_xla_program_owner(current)?;
        self.validate_xla_program_owner(replacement)?;
        validate_xla_replacement_metadata(current.into(), replacement.into())
    }

    fn execute_xla_program(
        &self,
        program: &XlaCompiledProgram<'c>,
        inputs: Vec<Array<'c>>,
    ) -> Result<Vec<Array<'c>>, XlaDomainError> {
        self.execute_compiled_async(program, inputs).map(Execution::into_output)
    }

    /// Returns the canonical, versioned identity of this XLA compilation. Persistent cache directories are trusted
    /// executable sources and must not be writable by untrusted users.
    #[inline]
    fn xla_persistent_cache_key(&self, key: &XlaCompilationKey) -> Option<Vec<u8>> {
        Some(key.canonical_bytes.to_vec())
    }

    fn serialize_xla_program(&self, program: &XlaCompiledProgram<'c>) -> Result<Option<Vec<u8>>, XlaDomainError> {
        let executable = match program.executable.executable() {
            Ok(executable) => executable,
            Err(ryft_pjrt::Error::Unimplemented { .. }) => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let serialized = match executable.serialize() {
            Ok(serialized) => serialized,
            Err(ryft_pjrt::Error::Unimplemented { .. }) => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let device_assignment = program.executable.device_assignment()?;
        let metadata = XlaPersistentExecutableMetadataV5 {
            schema_version: XLA_PERSISTENT_EXECUTABLE_SCHEMA_VERSION,
            feature_flags: XLA_PERSISTENT_EXECUTABLE_FEATURE_FLAGS,
            compilation_options: program.compilation_options.encode_to_vec(),
            signature: PersistentArraySignatureV3::encode(&program.input_types, &program.output_types),
            input_mapping: persistent_mapping(program.signature.input_mapping())?,
            output_mapping: persistent_mapping(program.signature.output_mapping())?,
            input_dimensions: persistent_input_dimensions(&program.signature),
            output_dimensions: persistent_output_dimensions(&program.signature),
            requires_assertion_handler: program.requires_assertion_handler,
            donation_flags: program.donation_flags.to_vec(),
            capture_count: program.capture_count as u64,
            expected_argument_shardings: program
                .expected_argument_shardings
                .iter()
                .map(PersistentShardingV1::from)
                .collect(),
            mesh: PersistentDeviceMeshV1::from(&program.mesh),
            device_kinds: program.device_kinds.to_vec(),
            replica_count: device_assignment.replica_count() as u64,
            partition_count: device_assignment.computation_count() as u64,
            device_assignment: flatten_device_assignment(&device_assignment)?,
            platform_name: program.platform_name.to_string(),
            platform_version: program.platform_version.to_string(),
            compiler_identity: program.compiler_identity.to_string(),
            xla_flags: program.xla_flags.to_string(),
            compilation_duration_nanoseconds: program
                .compilation_duration
                .map(|duration| duration.as_nanos())
                .map(|nanoseconds| u64::try_from(nanoseconds).unwrap_or(u64::MAX)),
        };
        let metadata = serde_json::to_vec(&metadata)
            .map_err(|error| persistent_error(format!("failed to encode metadata: {error}")))?;
        let mut bytes = Vec::with_capacity(
            XLA_PERSISTENT_EXECUTABLE_MAGIC.len() + size_of::<u64>() + metadata.len() + serialized.data().len(),
        );
        bytes.extend_from_slice(XLA_PERSISTENT_EXECUTABLE_MAGIC);
        bytes.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        bytes.extend_from_slice(metadata.as_slice());
        bytes.extend_from_slice(serialized.data());
        Ok(Some(bytes))
    }

    fn deserialize_xla_program(&self, bytes: &[u8]) -> Result<Option<XlaCompiledProgram<'c>>, XlaDomainError> {
        let header_size = XLA_PERSISTENT_EXECUTABLE_MAGIC.len() + size_of::<u64>();
        if bytes.len() < header_size {
            return Err(persistent_error("missing persistent executable header"));
        }
        if &bytes[..XLA_PERSISTENT_EXECUTABLE_MAGIC.len()] != XLA_PERSISTENT_EXECUTABLE_MAGIC {
            return Ok(None);
        }
        let metadata_size = u64::from_le_bytes(
            bytes[XLA_PERSISTENT_EXECUTABLE_MAGIC.len()..header_size]
                .try_into()
                .expect("metadata size has an exact fixed width"),
        );
        let metadata_size = checked_usize(metadata_size)?;
        let metadata_end = header_size
            .checked_add(metadata_size)
            .filter(|metadata_end| *metadata_end <= bytes.len())
            .ok_or_else(|| persistent_error("persistent executable metadata is truncated"))?;
        let metadata: XlaPersistentExecutableMetadataV5 = serde_json::from_slice(&bytes[header_size..metadata_end])
            .map_err(|error| persistent_error(format!("failed to decode metadata: {error}")))?;
        if metadata.schema_version != XLA_PERSISTENT_EXECUTABLE_SCHEMA_VERSION
            || metadata.feature_flags != XLA_PERSISTENT_EXECUTABLE_FEATURE_FLAGS
        {
            return Ok(None);
        }
        if metadata.platform_name != self.client()?.platform_name()?.as_ref()
            || metadata.platform_version != self.client()?.platform_version()?.as_ref()
            || metadata.compiler_identity != XLA_COMPILER_IDENTITY.as_str()
            || metadata.xla_flags != std::env::var("XLA_FLAGS").unwrap_or_default()
        {
            return Ok(None);
        }

        let mesh = DeviceMesh::try_from(metadata.mesh)?;
        if metadata.device_kinds != ordered_device_kinds(self.client()?, &mesh)? {
            return Ok(None);
        }
        let live_devices =
            self.client()?.devices()?.into_iter().map(Device::from_pjrt).collect::<Result<Vec<_>, _>>()?;
        if mesh.devices().iter().any(|device| !live_devices.contains(device)) {
            return Ok(None);
        }
        let (input_types, output_types) = metadata.signature.decode()?;
        let signature = XlaExecutableSignature::new(input_types.as_slice(), output_types.as_slice());
        if persistent_mapping(signature.input_mapping())? != metadata.input_mapping
            || persistent_mapping(signature.output_mapping())? != metadata.output_mapping
            || persistent_input_dimensions(&signature) != metadata.input_dimensions
            || persistent_output_dimensions(&signature) != metadata.output_dimensions
        {
            return Err(persistent_error("executable signature does not match logical input and output types"));
        }
        if input_types
            .iter()
            .chain(output_types.iter())
            .filter_map(ArrayType::sharding)
            .any(|sharding| sharding.mesh() != mesh.logical_mesh())
        {
            return Err(persistent_error("input or output sharding mesh does not match executable mesh"));
        }
        let expected_argument_shardings = metadata
            .expected_argument_shardings
            .into_iter()
            .map(Sharding::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        if expected_argument_shardings.iter().any(|sharding| sharding.mesh() != mesh.logical_mesh()) {
            return Err(persistent_error("argument sharding mesh does not match executable mesh"));
        }
        let capture_count = checked_usize(metadata.capture_count)?;
        let replica_count = checked_usize(metadata.replica_count)?;
        let partition_count = checked_usize(metadata.partition_count)?;
        if replica_count.checked_mul(partition_count) != Some(mesh.device_count())
            || metadata.device_assignment.len() != mesh.device_count()
        {
            return Err(persistent_error("device assignment does not match executable mesh"));
        }
        let physical_input_count = signature.physical_input_count();
        if capture_count > input_types.len()
            || expected_argument_shardings.len() != physical_input_count
            || metadata.donation_flags.len() != input_types.len() - capture_count
        {
            return Err(persistent_error("capture or donation metadata has an invalid arity"));
        }
        let donation_flags = metadata.donation_flags;
        let compilation_options = CompilationOptions::decode(metadata.compilation_options.as_slice())
            .map_err(|error| persistent_error(format!("failed to decode compilation options: {error}")))?;
        // Register before deserialization because loading an executable may resolve its custom-call target. The
        // persisted flag is authoritative on cache hits, where the source program is no longer available to inspect.
        self.ensure_runtime_requirements(metadata.requires_assertion_handler, metadata.platform_name.as_str())?;
        let executable = self.client()?.deserialize_and_load_executable(
            &bytes[metadata_end..],
            Some(&compilation_options),
            &LoadOptions::default(),
        )?;
        let device_assignment = executable.device_assignment()?;
        if device_assignment.replica_count() != replica_count
            || device_assignment.computation_count() != partition_count
            || flatten_device_assignment(&device_assignment)? != metadata.device_assignment
        {
            return Ok(None);
        }
        let compilation_duration = metadata.compilation_duration_nanoseconds.map(Duration::from_nanos);
        Ok(Some(XlaCompiledProgram {
            executable: Arc::new(executable),
            input_types: input_types.into(),
            output_types: output_types.into(),
            signature,
            requires_assertion_handler: metadata.requires_assertion_handler,
            donation_flags: donation_flags.into(),
            capture_count,
            expected_argument_shardings: expected_argument_shardings.into(),
            mesh,
            compilation_options,
            platform_name: metadata.platform_name.into(),
            platform_version: metadata.platform_version.into(),
            device_kinds: metadata.device_kinds.into(),
            compiler_identity: metadata.compiler_identity.into(),
            xla_flags: metadata.xla_flags.into(),
            compilation_duration,
            analysis: Arc::new(OnceLock::new()),
        }))
    }
}

impl<'c> CompilationCacheDomain for XlaDomain<'c> {
    type CacheKey = XlaCompilationKey;

    #[inline]
    fn compilation_key(&self, program: &Self::LoweredProgram) -> Result<Self::CacheKey, Self::Error> {
        Self::xla_compilation_key(program)
    }

    #[inline]
    fn persistent_cache_key(&self, key: &Self::CacheKey) -> Option<Vec<u8>> {
        self.xla_persistent_cache_key(key)
    }

    #[inline]
    fn serialize_program(&self, program: &Self::CompiledProgram) -> Result<Option<Vec<u8>>, Self::Error> {
        self.serialize_xla_program(program)
    }

    #[inline]
    fn deserialize_program(&self, bytes: &[u8]) -> Result<Option<Self::CompiledProgram>, Self::Error> {
        self.deserialize_xla_program(bytes)
    }
}

impl<'c> AnalyzableCompilationDomain for XlaDomain<'c> {
    type Analysis = XlaCompilationAnalysis;

    fn analyze<Input: Parameterized<ArrayIrType>, Output: Parameterized<ArrayIrType>>(
        &self,
        executable_function: &ryft_core::compilation::ExecutableFunction<Self, Input, Output>,
    ) -> Result<Self::Analysis, Self::Error> {
        let program = executable_function.compiled_program();
        program
            .analysis
            .get_or_init(|| analyze_xla_program(program).map_err(|error| error.to_string()))
            .clone()
            .map_err(|reason| XlaDomainError::InvalidCompilationAnalysis { reason })
    }
}

fn flatten_device_assignment(assignment: &ryft_pjrt::DeviceAssignment) -> Result<Vec<u64>, XlaDomainError> {
    let mut devices = Vec::with_capacity(assignment.replica_count() * assignment.computation_count());
    for replica in 0..assignment.replica_count() {
        for partition in 0..assignment.computation_count() {
            devices.push(assignment.device_id(replica, partition)? as u64);
        }
    }
    Ok(devices)
}

fn ordered_device_kinds(client: &Client<'_>, mesh: &DeviceMesh) -> Result<Vec<String>, XlaDomainError> {
    let devices = client.devices()?;
    mesh.devices()
        .iter()
        .map(|mesh_device| {
            let device =
                devices.iter().find(|device| device.id().is_ok_and(|id| id == mesh_device.id())).ok_or_else(|| {
                    persistent_error(format!("device {} is not visible to the live client", mesh_device.id()))
                })?;
            device.kind().map(|kind| kind.into_owned()).map_err(Into::into)
        })
        .collect()
}

pub(crate) fn validate_xla_input_type(declared: &ArrayType, actual: &ArrayType) -> Result<(), XlaDomainError> {
    let declared_without_sharding =
        declared.clone().with_sharding(None).map_err(|error| XlaDomainError::Array(error.into()))?;
    let actual_without_sharding =
        actual.clone().with_sharding(None).map_err(|error| XlaDomainError::Array(error.into()))?;
    if declared_without_sharding.is_refined_by(&actual_without_sharding) {
        Ok(())
    } else {
        Err(ProgramError::InvalidArgument {
            message: format!("runtime input type {actual} does not refine declared type {declared}"),
        }
        .into())
    }
}

fn validate_output_types(declared: &[ArrayIrType], actual: &[ArrayType]) -> Result<(), XlaDomainError> {
    if declared.len() != actual.len() {
        return Err(ProgramError::InvalidOutputCount { expected: declared.len(), actual: actual.len() }.into());
    }
    for (declared, actual) in declared.iter().zip(actual) {
        let declared = <&ArrayType>::try_from(declared).map_err(ProgramError::from)?;
        if !declared.is_refined_by(actual) {
            return Err(ProgramError::InvalidArgument {
                message: format!("backend output type {actual} does not refine declared type {declared}"),
            }
            .into());
        }
    }
    Ok(())
}

pub(crate) fn validate_runtime_outputs(declared: &[ArrayType], outputs: &[Array<'_>]) -> Result<(), XlaDomainError> {
    if declared.len() != outputs.len() {
        return Err(ProgramError::InvalidOutputCount { expected: declared.len(), actual: outputs.len() }.into());
    }
    for (declared, actual) in declared.iter().zip(outputs.iter().map(Typed::r#type)) {
        if !declared.is_refined_by(actual.as_ref()) {
            return Err(ProgramError::InvalidArgument {
                message: format!("runtime output type {actual} does not refine declared type {declared}"),
            }
            .into());
        }
    }
    Ok(())
}

fn analyze_xla_program(program: &XlaCompiledProgram<'_>) -> Result<XlaCompilationAnalysis, XlaDomainError> {
    let executable = program.executable.executable()?;
    let properties = match executable.cost_analysis() {
        Ok(properties) => properties
            .iter()
            .map(|(name, value)| (name.clone(), XlaAnalysisValue::from(value)))
            .collect::<BTreeMap<_, _>>(),
        Err(ryft_pjrt::Error::Unimplemented { .. } | ryft_pjrt::Error::Unavailable { .. }) => BTreeMap::new(),
        Err(error) => return Err(error.into()),
    };
    let memory = match executable.memory_statistics() {
        Ok(statistics) => Some(XlaMemoryAnalysis {
            device_generated_code_size_in_bytes: statistics.device_generated_code_size_in_bytes,
            device_input_size_in_bytes: statistics.device_input_size_in_bytes,
            device_output_size_in_bytes: statistics.device_output_size_in_bytes,
            device_alias_size_in_bytes: statistics.device_alias_size_in_bytes,
            device_temporary_size_in_bytes: statistics.device_temporary_size_in_bytes,
            device_peak_memory_in_bytes: statistics.device_peak_memory_in_bytes,
            device_total_memory_in_bytes: statistics.device_total_memory_in_bytes,
            device_total_allocation_bytes: statistics.device_total_allocation_bytes,
            device_indefinite_allocations: statistics.device_indefinite_allocations,
            device_peak_unpadded_heap_bytes: statistics.device_peak_unpadded_heap_bytes,
            host_generated_code_size_in_bytes: statistics.host_generated_code_size_in_bytes,
            host_input_size_in_bytes: statistics.host_input_size_in_bytes,
            host_output_size_in_bytes: statistics.host_output_size_in_bytes,
            host_alias_size_in_bytes: statistics.host_alias_size_in_bytes,
            host_temporary_size_in_bytes: statistics.host_temporary_size_in_bytes,
        }),
        Err(ryft_pjrt::Error::Unimplemented { .. } | ryft_pjrt::Error::Unavailable { .. }) => None,
        Err(error) => return Err(error.into()),
    };
    Ok(XlaCompilationAnalysis {
        floating_point_operations: numeric_analysis_property(&properties, &["flops", "floating_point_operations"])?,
        transcendental_operations: numeric_analysis_property(
            &properties,
            &["transcendentals", "transcendental_operations"],
        )?,
        bytes_accessed: numeric_analysis_property(&properties, &["bytes accessed", "bytes_accessed"])?,
        memory,
        generated_code_size_in_bytes: optional_pjrt_analysis(executable.generated_code_size_in_bytes())?,
        replica_count: optional_pjrt_analysis(executable.replica_count())?,
        partition_count: optional_pjrt_analysis(executable.computation_count())?,
        executable_fingerprint: optional_pjrt_analysis(executable.fingerprint().map(|value| value.into_owned()))?,
        compilation_duration: program.compilation_duration,
        properties,
    })
}

impl From<&PjrtValue> for XlaAnalysisValue {
    fn from(value: &PjrtValue) -> Self {
        match value {
            PjrtValue::Bool(value) => Self::Boolean(*value),
            PjrtValue::I64(value) => Self::Integer(*value),
            PjrtValue::I64List(value) => Self::IntegerList(value.clone()),
            PjrtValue::F32(value) => Self::Float(*value as f64),
            PjrtValue::String(value) => Self::String(value.clone()),
        }
    }
}

fn numeric_analysis_property(
    properties: &BTreeMap<String, XlaAnalysisValue>,
    names: &[&str],
) -> Result<Option<f64>, XlaDomainError> {
    let Some((name, value)) = names.iter().find_map(|name| properties.get(*name).map(|value| (*name, value))) else {
        return Ok(None);
    };
    match value {
        XlaAnalysisValue::Integer(value) => Ok(Some(*value as f64)),
        XlaAnalysisValue::Float(value) => Ok(Some(*value)),
        _ => Err(XlaDomainError::InvalidCompilationAnalysis { reason: format!("property `{name}` is not numeric") }),
    }
}

fn optional_pjrt_analysis<T>(result: Result<T, ryft_pjrt::Error>) -> Result<Option<T>, XlaDomainError> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(ryft_pjrt::Error::Unimplemented { .. } | ryft_pjrt::Error::Unavailable { .. }) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

/// Applies an optional per-leaf sharding override to a public composite signature. When overrides are provided, every
/// overridden leaf must be an array: first-class dimensions are internal SSA values, and references must be discharged
/// into explicit array state before either kind can cross the PJRT ABI. Note that without overrides the signature
/// passes through unchecked here; non-array boundary leaves are rejected by the later array-only conversions on the
/// compile path.
fn apply_signature_shardings(
    mut types: Vec<ArrayIrType>,
    shardings: Option<&[Sharding]>,
    kind: &'static str,
) -> Result<Vec<ArrayIrType>, XlaDomainError> {
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
    for (r#type, sharding) in types.iter_mut().zip(shardings) {
        let array_type = <&ArrayType>::try_from(&*r#type).map_err(ProgramError::from)?;
        *r#type = ArrayType::new(array_type.data_type(), array_type.shape().clone())
            .with_layout(array_type.layout().cloned())
            .with_sharding(sharding.clone())
            .map_err(|error| XlaDomainError::Array(error.into()))?
            .into();
    }
    Ok(types)
}

/// Returns whether every array type that occurs anywhere in `program` has a static shape. The walk covers the whole
/// region arena, and therefore every region boundary, constant, and instruction result, recursively through attached
/// regions. First-class dimension-typed atoms are ignored because they lower to ordinary scalars; only dynamically
/// shaped *array* types make the lowered module unacceptable to the Shardy partitioner.
fn has_only_static_array_types(program: &FlatXlaProgram) -> bool {
    program.regions().iter().all(|region| {
        region.atoms().iter().all(|atom| match atom.r#type().as_ref() {
            ArrayIrType::Array(array_type) => array_type.static_shape().is_some(),
            ArrayIrType::Dimension(_) => true,
            // The predicate answers its own question rather than relying on the lowering-time reference rejection
            // firing first: a dynamically shaped referent counts as dynamic so that partitioner selection and its
            // guards stay correct even for programs whose only dynamic type is a reference referent.
            ArrayIrType::Reference(reference_type) => reference_type.referent().static_shape().is_some(),
        })
    })
}

/// Returns whether `program` stages a `shard_map` anywhere, including inside nested regions. Traced `shard_map`
/// boundaries are required to be static by [`ShardMap::trace`](crate::experimental::shard_map::ShardMap::trace), so a
/// `shard_map` can still appear in a program that is bounded-dynamic elsewhere, which this predicate detects.
fn contains_shard_map(program: &FlatXlaProgram) -> bool {
    program.regions().iter().any(|region| {
        region
            .instructions()
            .iter()
            .any(|instruction| matches!(instruction.operation(), XlaOperation::ShardMap(_)))
    })
}

/// Padding contract used when admitting one staged operation to data-dependent bounded-dynamic lowering.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum DataDependentPaddingDiscipline {
    /// StableHLO propagates the operation's logical dynamic dimensions without inspecting physical padding lanes.
    Propagated,

    /// Ryft delegates identity- or sentinel-masking of padding lanes to XLA's bounded-dynamic legalizer.
    XlaMasked,

    /// Ryft delegates zero-padding of contraction lanes to XLA's bounded-dynamic legalizer.
    XlaZeroPadded,

    /// Ryft explicitly materializes bounded physical operands with operation-specific padding values or masks and
    /// restores dynamic result metadata after the physical computation.
    RyftMasked,

    /// Ryft lacks sufficient semantic or execution evidence to admit this operation safely.
    Unsupported { reason: &'static str },
}

/// Returns the XLA masking contract for one reduction kind. The exhaustive match makes adding a reduction kind a
/// compile-time update to the data-dependent inventory instead of silently inheriting another kind's padding rule.
fn reduction_data_dependent_padding_discipline(kind: ReductionKind) -> DataDependentPaddingDiscipline {
    use DataDependentPaddingDiscipline::{Unsupported, XlaMasked};

    match kind {
        ReductionKind::Sum | ReductionKind::Max | ReductionKind::Min | ReductionKind::Any | ReductionKind::All => {
            XlaMasked
        }
        ReductionKind::Mean => {
            Unsupported { reason: "mean over a dynamically sized reduction axis is not supported by the XLA lowering" }
        }
    }
}

/// Returns the XLA masking contract for one sort direction. Keeping the match exhaustive makes sort-direction
/// additions compile-forced inventory changes because ascending and descending sorts require opposite sentinels.
fn sort_data_dependent_padding_discipline(direction: SortDirection) -> DataDependentPaddingDiscipline {
    use DataDependentPaddingDiscipline::XlaMasked;

    match direction {
        SortDirection::Ascending | SortDirection::Descending => XlaMasked,
    }
}

/// Rejects every scatter combiner until its padding behavior has kind-specific execution evidence. The exhaustive
/// match deliberately forces new combiners through the same review rather than admitting them as ordinary shape-
/// preserving operations.
fn scatter_data_dependent_padding_discipline(kind: ScatterReductionKind) -> DataDependentPaddingDiscipline {
    use DataDependentPaddingDiscipline::Unsupported;

    match kind {
        ScatterReductionKind::Overwrite => Unsupported {
            reason: "overwrite-scatter padding semantics have not been verified for data-derived dimensions",
        },
        ScatterReductionKind::Add => {
            Unsupported { reason: "add-scatter padding semantics have not been verified for data-derived dimensions" }
        }
        ScatterReductionKind::Mul => Unsupported {
            reason: "multiply-scatter padding semantics have not been verified for data-derived dimensions",
        },
        ScatterReductionKind::Min => Unsupported {
            reason: "minimum-scatter padding semantics have not been verified for data-derived dimensions",
        },
        ScatterReductionKind::Max => Unsupported {
            reason: "maximum-scatter padding semantics have not been verified for data-derived dimensions",
        },
    }
}

/// Classifies every homogeneous array operation by the padding discipline required for data-derived dimensions.
/// Keeping this match exhaustive makes a newly added operation a compile-time inventory update rather than silently
/// assuming that arbitrary padding is safe.
fn array_data_dependent_padding_discipline(
    operation: &ArrayOperation<crate::experimental::ops::XlaArrayConstant>,
) -> DataDependentPaddingDiscipline {
    use DataDependentPaddingDiscipline::{Propagated, RyftMasked, Unsupported, XlaZeroPadded};

    match operation {
        ArrayOperation::Reduce(operation) => reduction_data_dependent_padding_discipline(operation.kind()),
        ArrayOperation::Sort(operation) => sort_data_dependent_padding_discipline(operation.direction()),
        ArrayOperation::Scatter(operation) => scatter_data_dependent_padding_discipline(operation.kind()),
        ArrayOperation::Dot(_) => XlaZeroPadded,
        ArrayOperation::ScaledDot(_) => RyftMasked,
        ArrayOperation::DotProductAttention(_) | ArrayOperation::DotProductAttentionBackward(_) => RyftMasked,
        ArrayOperation::RngBitGenerator(_) => Unsupported {
            reason: "random generation advances state according to physical rather than logical element count",
        },
        ArrayOperation::CustomCall(_) => Unsupported { reason: "custom-call physical-padding semantics are opaque" },
        ArrayOperation::Print(_) => Unsupported { reason: "print exposes the physical representation of its operand" },
        ArrayOperation::Zero(_)
        | ArrayOperation::ZeroLike(_)
        | ArrayOperation::One(_)
        | ArrayOperation::OneLike(_)
        | ArrayOperation::Constant(_)
        | ArrayOperation::Iota(_)
        | ArrayOperation::CoordinateBasis(_)
        | ArrayOperation::Abs(_)
        | ArrayOperation::Neg(_)
        | ArrayOperation::Add(_)
        | ArrayOperation::Sub(_)
        | ArrayOperation::Mul(_)
        | ArrayOperation::Div(_)
        | ArrayOperation::Sin(_)
        | ArrayOperation::Cos(_)
        | ArrayOperation::Atan2(_)
        | ArrayOperation::Exp(_)
        | ArrayOperation::Log(_)
        | ArrayOperation::Sqrt(_)
        | ArrayOperation::Rsqrt(_)
        | ArrayOperation::Tanh(_)
        | ArrayOperation::Logistic(_)
        | ArrayOperation::Erf(_)
        | ArrayOperation::Pow(_)
        | ArrayOperation::Sign(_)
        | ArrayOperation::Floor(_)
        | ArrayOperation::Ceil(_)
        | ArrayOperation::Round(_)
        | ArrayOperation::Max(_)
        | ArrayOperation::Min(_)
        | ArrayOperation::Rem(_)
        | ArrayOperation::Not(_)
        | ArrayOperation::And(_)
        | ArrayOperation::Or(_)
        | ArrayOperation::Xor(_)
        | ArrayOperation::Complex(_)
        | ArrayOperation::Conjugate(_)
        | ArrayOperation::Real(_)
        | ArrayOperation::Imaginary(_)
        | ArrayOperation::Collective(_)
        | ArrayOperation::AllGather(_)
        | ArrayOperation::PSumScatter(_)
        | ArrayOperation::Ppermute(_)
        | ArrayOperation::AllToAll(_)
        | ArrayOperation::AxisIndex(_)
        | ArrayOperation::Transpose(_)
        | ArrayOperation::Reshape(_)
        | ArrayOperation::Broadcast(_)
        | ArrayOperation::Pad(_)
        | ArrayOperation::Concatenate(_)
        | ArrayOperation::Gather(_)
        | ArrayOperation::Slice(_)
        | ArrayOperation::UpdateSlice(_)
        | ArrayOperation::DynamicSlice(_)
        | ArrayOperation::DynamicUpdateSlice(_)
        | ArrayOperation::Compare(_)
        | ArrayOperation::Select(_)
        | ArrayOperation::Condition(_)
        | ArrayOperation::While(_)
        | ArrayOperation::Scan(_)
        | ArrayOperation::ConvertElementType(_)
        | ArrayOperation::TransferToMemory(_)
        | ArrayOperation::Reshard(_)
        | ArrayOperation::ShardingConstraint(_)
        | ArrayOperation::StopGradient(_)
        | ArrayOperation::Tag(_)
        | ArrayOperation::Rematerialize(_)
        | ArrayOperation::CustomJvp(_)
        | ArrayOperation::CustomVjp(_)
        | ArrayOperation::LinearCall(_) => Propagated,
    }
}

/// Classifies every XLA operation by the bounded-dynamic padding discipline it requires.
fn data_dependent_padding_discipline(operation: &XlaOperation) -> DataDependentPaddingDiscipline {
    use DataDependentPaddingDiscipline::{Propagated, Unsupported};

    match operation {
        XlaOperation::Array(operation) => array_data_dependent_padding_discipline(operation),
        XlaOperation::RngBitGenerator(_) => Unsupported {
            reason: "random generation advances state according to physical rather than logical element count",
        },
        XlaOperation::CustomCall(_) => Unsupported { reason: "custom-call physical-padding semantics are opaque" },
        XlaOperation::NewReference(_)
        | XlaOperation::ReferenceRead(_)
        | XlaOperation::ReferenceSwap(_)
        | XlaOperation::ReferenceAddUpdate(_)
        | XlaOperation::FreezeReference(_) => {
            Unsupported { reason: "references must be discharged before bounded-dynamic XLA validation" }
        }
        XlaOperation::ShardMap(_) => {
            Unsupported { reason: "shard-map lowering requires Shardy, which does not support bounded-dynamic tensors" }
        }
        XlaOperation::Zero(_)
        | XlaOperation::DynamicOne(_)
        | XlaOperation::DynamicIota(_)
        | XlaOperation::Dimension(_)
        | XlaOperation::Compare(_)
        | XlaOperation::DimensionSize(_)
        | XlaOperation::DimensionFromScalar(_)
        | XlaOperation::DimensionToScalar(_)
        | XlaOperation::Reshape(_)
        | XlaOperation::Broadcast(_)
        | XlaOperation::Concatenate(_)
        | XlaOperation::Pad(_)
        | XlaOperation::DynamicShapeSlice(_)
        | XlaOperation::AllGather(_)
        | XlaOperation::PSumScatter(_)
        | XlaOperation::AllToAll(_)
        | XlaOperation::Condition(_)
        | XlaOperation::While(_)
        | XlaOperation::Scan(_)
        | XlaOperation::CustomJvp(_)
        | XlaOperation::CustomVjp(_)
        | XlaOperation::LinearCall(_)
        | XlaOperation::Rematerialize(_)
        | XlaOperation::JitCall(_) => Propagated,
    }
}

/// Returns whether `r#type` carries an identity derived from `dimension_from_scalar`.
fn references_data_derived_dimension(r#type: &ArrayIrType, variables: &HashSet<DimensionVariable>) -> bool {
    r#type.identities().any(|(_, identity)| variables.contains(identity))
}

/// Validates the bounded-dynamic compilation contract for dimensions produced by `dimension_from_scalar`.
///
/// `XlaMasked` and `XlaZeroPadded` are explicit delegation decisions: the bounded-dynamic XLA legalizer owns those
/// transformations. `RyftMasked` operations instead materialize their physical kernel inputs and restore logical
/// extents in Ryft's lowering. Only configurations with execution evidence are admitted; unbounded gateways and
/// unsupported consumers are rejected before MLIR construction.
fn validate_data_dependent_compilation(program: &FlatXlaProgram) -> Result<(), XlaDomainError> {
    let mut variables = HashSet::new();
    for region in program.regions() {
        for instruction in region.instructions() {
            if let XlaOperation::DimensionFromScalar(operation) = instruction.operation() {
                let variable = operation.result_type().variable();
                if variable.bounds().upper().is_none() {
                    return Err(ProgramError::InvalidArgument {
                        message: format!(
                            "data-derived dimension `{variable}` with bounds {} needs a finite upper bound for XLA \
                             compilation",
                            variable.bounds(),
                        ),
                    }
                    .into());
                }
                variables.insert(variable.clone());
            }
        }
    }
    if variables.is_empty() {
        return Ok(());
    }

    // Dimension arithmetic produces fresh identities, so close the data-derived set transitively over dimension-typed
    // instruction outputs before classifying the array operations that consume those identities.
    loop {
        let count = variables.len();
        for region in program.regions() {
            for instruction in region.instructions() {
                if !instruction.inputs().iter().any(|input| {
                    references_data_derived_dimension(region.atoms()[input.index()].r#type().as_ref(), &variables)
                }) {
                    continue;
                }
                for output in instruction.outputs() {
                    if let ArrayIrType::Dimension(r#type) = region.atoms()[output.index()].r#type().as_ref() {
                        variables.insert(r#type.variable().clone());
                    }
                }
            }
        }
        if variables.len() == count {
            break;
        }
    }

    for region in program.regions() {
        for instruction in region.instructions() {
            let consumes_data_derived_extent = instruction.inputs().iter().chain(instruction.outputs()).any(|atom| {
                references_data_derived_dimension(region.atoms()[atom.index()].r#type().as_ref(), &variables)
            });
            if consumes_data_derived_extent {
                if let DataDependentPaddingDiscipline::Unsupported { reason } =
                    data_dependent_padding_discipline(instruction.operation())
                {
                    return Err(ProgramError::UnsupportedOperation {
                        message: format!(
                            "operation `{}` cannot consume data-derived dimensions during XLA lowering because \
                             {reason}",
                            instruction.operation().name(),
                        ),
                    }
                    .into());
                }
            }
        }
    }
    Ok(())
}

/// Overlays the selected multi-device execution mode on the base [`CompilationOptions`] template. Static programs
/// use Shardy SPMD partitioning, while bounded-dynamic programs use one fully replicated executable replica per
/// device because Shardy does not yet accept dynamic tensors.
fn jit_compilation_options(
    base: &CompilationOptions,
    device_count: usize,
    use_shardy_partitioner: bool,
    feedback_directed_profile: Option<&XlaFeedbackDirectedProfile>,
) -> CompilationOptions {
    use ryft_pjrt::protos::ExecutableCompilationOptions;
    let mut options = base.clone();
    let exec_options = options.executable_build_options.get_or_insert_with(ExecutableCompilationOptions::default);
    if exec_options.device_ordinal == 0 {
        exec_options.device_ordinal = -1;
    }
    exec_options.replica_count = if use_shardy_partitioner { 1 } else { device_count as i64 };
    exec_options.partition_count = if use_shardy_partitioner { device_count as i64 } else { 1 };
    exec_options.use_spmd_partitioning = use_shardy_partitioner;
    exec_options.use_shardy_partitioner = use_shardy_partitioner;
    if let Some(profile) = feedback_directed_profile {
        exec_options.fdo_profile = profile.bytes().to_vec();
        options.profile_version = profile.version();
    }
    options
}

/// Encodes compilation options deterministically even though Prost represents protobuf maps as randomly seeded
/// `HashMap`s. Map entries are removed from a clone before normal protobuf encoding and appended as length-delimited,
/// path-qualified records sorted by their encoded keys and values.
fn canonical_compilation_options_bytes(options: &CompilationOptions) -> Vec<u8> {
    use ryft_pjrt::protos::{AutoTuneReferenceKey, AutoTuneResultKey};

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

/// Host copy that has been issued for one bounded input requiring padding.
struct PendingBoundedInputHostCopy<'c> {
    /// Logical input index used to recover the runtime array metadata.
    logical_input_index: usize,

    /// Physical executable input slot to populate.
    physical_input_index: usize,

    /// Static executable type defining the padding target.
    physical_type: ArrayType,

    /// Concrete logical shape copied from the runtime input.
    actual_shape: StaticShape,

    /// Static executable-bound shape used for physical storage.
    physical_shape: StaticShape,

    /// Already-issued host copy for the logical input.
    host_copy: DenseArrayHostCopy,

    /// Single-flight reservation that retains success.
    producer: BoundedMaterializationProducer<'c>,
}

/// Bound-shaped upload issued for one padded input but not yet published to its cache or executable slot.
struct PendingBoundedInputPublication<'c> {
    /// Physical executable input slot to populate after readiness succeeds.
    physical_input_index: usize,

    /// Uploaded physical array whose asynchronous transfer must become ready before publication.
    physical: Array<'c>,

    /// Single-flight reservation that retains ready success.
    producer: BoundedMaterializationProducer<'c>,
}

/// Cache waiter retained until all producer copies in this call have completed.
struct WaitingBoundedInputMaterialization<'c> {
    /// Logical input index used to retry production if the prior producer fails.
    logical_input_index: usize,

    /// Physical executable input slot to populate.
    physical_input_index: usize,

    /// Static executable type defining the cache key and padding target.
    physical_type: ArrayType,

    /// Concrete logical shape copied from the runtime input on a retry.
    actual_shape: StaticShape,

    /// Static executable-bound shape used for physical storage on a retry.
    physical_shape: StaticShape,

    /// Non-spinning single-flight wait handle.
    waiter: BoundedMaterializationWaiter<'c>,
}

/// Deterministic work and transport report for one bounded-input materialization boundary.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct BoundedInputMaterializationReport {
    /// Inputs that reused original device buffers because the runtime shape equaled the bound.
    at_bound_reuses: usize,

    /// Ready retained materializations reused without padding work or transport.
    cache_hits: usize,

    /// Missing cache entries whose successful production was retained.
    retained_misses: usize,

    /// Device-to-host shard copies issued by padding work.
    device_to_host_shard_copies: usize,

    /// Full-global host merge buffers allocated after shard copies.
    host_merge_buffer_allocations: usize,

    /// O(bound) host padding payloads allocated.
    host_padding_payload_allocations: usize,

    /// Bound-shaped host-to-device shard uploads.
    host_to_device_shard_uploads: usize,

    /// Hidden logical-extent scalar host-to-device uploads.
    extent_scalar_uploads: usize,

    /// Sum of logical payload bytes for bounded inputs.
    actual_bytes: usize,

    /// Sum of physical-bound payload bytes for bounded inputs.
    bound_bytes: usize,
}

/// Physical inputs and their deterministic bounded-materialization work report.
struct MaterializedBoundedInputs<'c> {
    /// Physical executable inputs, including hidden extent scalars.
    inputs: Vec<Array<'c>>,

    /// Path-local work and transport counts.
    #[cfg_attr(not(test), allow(dead_code))]
    report: BoundedInputMaterializationReport,
}

/// Selects whether a bounded input can reuse its device buffer or requires host padding.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum BoundedInputPacking {
    /// The runtime shape already equals the executable's physical bound.
    Reuse,

    /// At least one runtime extent is smaller than its executable bound.
    Pad,
}

/// Determines the bounded-input packing tier from concrete runtime and executable shapes.
fn bounded_input_packing(actual_shape: &StaticShape, physical_shape: &StaticShape) -> BoundedInputPacking {
    if actual_shape == physical_shape { BoundedInputPacking::Reuse } else { BoundedInputPacking::Pad }
}

/// Completes one already-issued host copy and issues its bound-shaped upload without publishing it.
fn upload_bounded_input_host_copy<'c>(
    client: &'c Client<'c>,
    input: &Array<'c>,
    pending: PendingBoundedInputHostCopy<'c>,
    report: &mut BoundedInputMaterializationReport,
) -> Result<PendingBoundedInputPublication<'c>, XlaDomainError> {
    report.device_to_host_shard_copies += pending.host_copy.shard_copy_count();
    let (source_bytes, allocated_merge_buffer) = pending.host_copy.finish_with_measurements()?;
    report.host_merge_buffer_allocations += usize::from(allocated_merge_buffer);
    let (padded_bytes, padding_payload_allocations) = pad_dense_array_bytes_with_target_allocation_count(
        source_bytes.as_slice(),
        pending.actual_shape.as_slice(),
        pending.physical_shape.as_slice(),
        pending.physical_type.data_type(),
    )?;
    report.host_padding_payload_allocations += padding_payload_allocations;
    let physical = Array::from_host_buffer(client, pending.physical_type, input.mesh(), padded_bytes)?;
    report.host_to_device_shard_uploads += physical.shards().iter().filter(|shard| shard.buffer().is_some()).count();
    Ok(PendingBoundedInputPublication {
        physical_input_index: pending.physical_input_index,
        physical,
        producer: pending.producer,
    })
}

/// Publishes one batch only after every asynchronously uploaded physical array passes `check_readiness`.
fn publish_bounded_input_uploads<'c>(
    publications: Vec<PendingBoundedInputPublication<'c>>,
    physical_inputs: &mut [Array<'c>],
    check_readiness: &mut impl FnMut(&[PendingBoundedInputPublication<'c>]) -> Result<(), XlaDomainError>,
) -> Result<(), XlaDomainError> {
    // Readiness is checked for the complete batch before any producer publishes. An error therefore drops every
    // reservation in the batch, caches no failed upload, and leaves all concurrent H2D work issued.
    check_readiness(publications.as_slice())?;
    for publication in publications {
        physical_inputs[publication.physical_input_index] = publication.producer.complete(publication.physical);
    }
    Ok(())
}

/// Pads bounded-dynamic inputs to static executable storage and appends their logical extents.
///
/// Dispatch has two tiers in this boundary: (1) an actual shape equal to the physical bound reuses the source array;
/// and (2) a smaller value uses a clone-shared retained materialization, with one single-flight producer per
/// structural `(ZeroOriginV1, physical ArrayType)` key. Below-bound zero-space values are deferred to
/// [`materialize_zero_space_carriers`], which synthesizes their bound-shaped carriers directly, and below-bound values
/// with non-addressable shards are rejected because padding must read every shard's bytes on this process. Every
/// producer D2H copy is issued during the first scan. The resulting H2D uploads are also all issued before any upload
/// readiness wait, and retained values are published only after the complete batch succeeds. Cold work is O(bound)
/// bytes plus O(shard-segment-count) overlap bookkeeping; ready hits perform no pad-path allocation or transport. A
/// failed asynchronous upload stores no cache entry and a later call retries the ordinary cold path.
fn materialize_bounded_dynamic_inputs<'c>(
    client: &'c Client<'c>,
    signature: &XlaExecutableSignature,
    declared_types: &[ArrayType],
    actual_types: &[ArrayType],
    inputs: Vec<Array<'c>>,
) -> Result<MaterializedBoundedInputs<'c>, XlaDomainError> {
    materialize_bounded_dynamic_inputs_with_readiness(
        client,
        signature,
        declared_types,
        actual_types,
        inputs,
        |publications| {
            for publication in publications {
                publication.physical.block_until_ready()?;
            }
            Ok(())
        },
    )
}

/// Implements bounded-input materialization with an injectable batch readiness check for deterministic failure tests.
fn materialize_bounded_dynamic_inputs_with_readiness<'c>(
    client: &'c Client<'c>,
    signature: &XlaExecutableSignature,
    declared_types: &[ArrayType],
    actual_types: &[ArrayType],
    inputs: Vec<Array<'c>>,
    mut check_readiness: impl FnMut(&[PendingBoundedInputPublication<'c>]) -> Result<(), XlaDomainError>,
) -> Result<MaterializedBoundedInputs<'c>, XlaDomainError> {
    assert_eq!(declared_types.len(), inputs.len());
    assert_eq!(actual_types.len(), inputs.len());
    let physical_types = signature.physical_input_types(declared_types);
    let mut physical_inputs = signature.project_inputs(inputs.as_slice());
    let mut pending_host_copies = Vec::new();
    let mut waiting_materializations = Vec::new();
    let mut report = BoundedInputMaterializationReport::default();
    for logical_input_index in 0..inputs.len() {
        if !signature
            .input_dimensions()
            .iter()
            .any(|input_dimension| input_dimension.logical_input_index() == logical_input_index)
        {
            continue;
        }
        let physical_input_index = signature.input_mapping()[logical_input_index].unwrap();
        let physical_type = physical_types[physical_input_index].clone();
        let physical_shape = physical_type.static_shape().ok_or_else(|| ProgramError::InvalidArgument {
            message: format!("bounded-dynamic executable input {logical_input_index} has no static upper shape"),
        })?;
        let actual_shape =
            actual_types[logical_input_index].static_shape().ok_or_else(|| ProgramError::InvalidArgument {
                message: format!("runtime executable input {logical_input_index} must have a static shape"),
            })?;
        report.actual_bytes += actual_types[logical_input_index].size_in_bytes()?;
        report.bound_bytes += physical_type.size_in_bytes()?;
        if bounded_input_packing(&actual_shape, &physical_shape) == BoundedInputPacking::Reuse {
            report.at_bound_reuses += 1;
            continue;
        }

        // A below-bound zero-space input carries no payload bytes to pad or cache: its dense host copy is empty, so
        // the retained-cache and byte-pad tiers below must not run. The zero-space carrier materialization that
        // follows this boundary synthesizes its bound-shaped all-false predicate directly instead.
        if physical_type.data_type().is_zero() {
            continue;
        }

        let input = &inputs[logical_input_index];
        if input.supports_bounded_materialization_cache() {
            match input.probe_bounded_materialization(BoundedMaterializationKey::new(physical_type.clone())) {
                BoundedMaterializationProbe::Hit(physical) => {
                    report.cache_hits += 1;
                    physical_inputs[physical_input_index] = physical;
                }
                BoundedMaterializationProbe::Produce(producer) => {
                    report.retained_misses += 1;
                    pending_host_copies.push(PendingBoundedInputHostCopy {
                        logical_input_index,
                        physical_input_index,
                        physical_type,
                        actual_shape,
                        physical_shape,
                        host_copy: begin_materialize_dense_array_bytes(input)?,
                        producer,
                    });
                }
                BoundedMaterializationProbe::Wait(waiter) => {
                    waiting_materializations.push(WaitingBoundedInputMaterialization {
                        logical_input_index,
                        physical_input_index,
                        physical_type,
                        actual_shape,
                        physical_shape,
                        waiter,
                    });
                }
            }
        } else {
            // Padding reads every shard's bytes on this process, so an input with a non-addressable shard cannot be
            // materialized here at all; reject it explicitly instead of failing deep inside the host copy.
            return Err(ProgramError::InvalidArgument {
                message: format!(
                    "bounded-dynamic executable input {logical_input_index} is below its bound but has \
                     non-addressable shards, so it cannot be padded on this process",
                ),
            }
            .into());
        }
    }

    let mut pending_publications = Vec::with_capacity(pending_host_copies.len());
    for pending in pending_host_copies {
        let input = &inputs[pending.logical_input_index];
        pending_publications.push(upload_bounded_input_host_copy(client, input, pending, &mut report)?);
    }
    publish_bounded_input_uploads(pending_publications, &mut physical_inputs, &mut check_readiness)?;

    for waiting in waiting_materializations {
        let WaitingBoundedInputMaterialization {
            logical_input_index,
            physical_input_index,
            physical_type,
            actual_shape,
            physical_shape,
            waiter,
        } = waiting;
        match waiter.resolve() {
            Ok(physical) => {
                report.cache_hits += 1;
                physical_inputs[physical_input_index] = physical;
            }
            Err(producer) => {
                // The prior producer failed and removed its reservation. This waiter is now the retry producer. Its
                // copy necessarily starts after that failure; the ordinary all-miss path above stays fully concurrent.
                report.retained_misses += 1;
                let input = &inputs[logical_input_index];
                let pending = PendingBoundedInputHostCopy {
                    logical_input_index,
                    physical_input_index,
                    physical_type,
                    actual_shape,
                    physical_shape,
                    host_copy: begin_materialize_dense_array_bytes(input)?,
                    producer,
                };
                let publication = upload_bounded_input_host_copy(client, input, pending, &mut report)?;
                publish_bounded_input_uploads(vec![publication], &mut physical_inputs, &mut check_readiness)?;
            }
        }
    }

    for input_dimension in signature.input_dimensions() {
        let input = &inputs[input_dimension.logical_input_index()];
        let extent = actual_types[input_dimension.logical_input_index()].shape().dimensions()[input_dimension.axis()]
            .value()
            .unwrap();
        let extent = i32::try_from(extent).map_err(|_| ProgramError::InvalidArgument {
            message: format!("runtime dimension extent {extent} does not fit in a StableHLO i32 scalar"),
        })?;
        let (extent_scalar, uploaded) = input.logical_extent_scalar(client, extent)?;
        report.extent_scalar_uploads += usize::from(uploaded);
        physical_inputs.push(extent_scalar);
    }
    assert_eq!(physical_inputs.len(), signature.physical_input_count());
    Ok(MaterializedBoundedInputs { inputs: physical_inputs, report })
}

/// Zero-pads one dense row-major array into a larger shape of the same rank.
#[cfg(test)]
fn pad_dense_array_bytes(
    source: &[u8],
    source_shape: &[usize],
    target_shape: &[usize],
    data_type: DataType,
) -> Result<Vec<u8>, XlaDomainError> {
    pad_dense_array_bytes_with_target_allocation_count(source, source_shape, target_shape, data_type)
        .map(|(bytes, _)| bytes)
}

/// Zero-pads one dense row-major array and reports the number of target payload allocations.
///
/// The count deliberately covers the O(bound) payload allocation, which dominates this path. Small O(rank) stride
/// vectors are implementation metadata and are not included.
fn pad_dense_array_bytes_with_target_allocation_count(
    source: &[u8],
    source_shape: &[usize],
    target_shape: &[usize],
    data_type: DataType,
) -> Result<(Vec<u8>, usize), XlaDomainError> {
    if source_shape.len() != target_shape.len()
        || source_shape.iter().zip(target_shape).any(|(source, target)| source > target)
    {
        return Err(ProgramError::InvalidArgument {
            message: format!(
                "cannot pad runtime shape {:?} into bounded physical shape {:?}",
                source_shape, target_shape,
            ),
        }
        .into());
    }
    let element_size = data_type.to_pjrt().element_size_in_bytes()?;
    let mut source_strides = vec![1usize; source_shape.len()];
    let mut target_strides = vec![1usize; target_shape.len()];
    for axis in (0..source_shape.len().saturating_sub(1)).rev() {
        source_strides[axis] = source_strides[axis + 1].checked_mul(source_shape[axis + 1]).ok_or_else(|| {
            ProgramError::InvalidArgument { message: "runtime input shape strides exceed usize".to_string() }
        })?;
        target_strides[axis] = target_strides[axis + 1].checked_mul(target_shape[axis + 1]).ok_or_else(|| {
            ProgramError::InvalidArgument { message: "bounded physical shape strides exceed usize".to_string() }
        })?;
    }
    let target_elements = target_shape.iter().try_fold(1usize, |count, dimension| {
        count.checked_mul(*dimension).ok_or_else(|| ProgramError::InvalidArgument {
            message: "bounded physical input element count exceeds usize".to_string(),
        })
    })?;
    let mut target = vec![
        0u8;
        target_elements.checked_mul(element_size).ok_or_else(|| ProgramError::InvalidArgument {
            message: "bounded physical input byte count exceeds usize".to_string(),
        })?
    ];
    if source_shape.is_empty() {
        target.copy_from_slice(source);
        let allocation_count = usize::from(!target.is_empty());
        return Ok((target, allocation_count));
    }
    copy_dense_array_into_padding(
        source,
        &mut target,
        source_shape,
        source_strides.as_slice(),
        target_strides.as_slice(),
        element_size,
        0,
        0,
        0,
    );
    let allocation_count = usize::from(!target.is_empty());
    Ok((target, allocation_count))
}

/// Recursively copies one dense row-major source into the origin-aligned region of a padded target.
#[allow(clippy::too_many_arguments)]
fn copy_dense_array_into_padding(
    source: &[u8],
    target: &mut [u8],
    source_shape: &[usize],
    source_strides: &[usize],
    target_strides: &[usize],
    element_size: usize,
    axis: usize,
    source_offset: usize,
    target_offset: usize,
) {
    if axis + 1 == source_shape.len() {
        let byte_count = source_shape[axis] * element_size;
        let source_offset = source_offset * element_size;
        let target_offset = target_offset * element_size;
        target[target_offset..target_offset + byte_count]
            .copy_from_slice(&source[source_offset..source_offset + byte_count]);
        return;
    }
    for index in 0..source_shape[axis] {
        copy_dense_array_into_padding(
            source,
            target,
            source_shape,
            source_strides,
            target_strides,
            element_size,
            axis + 1,
            source_offset + index * source_strides[axis],
            target_offset + index * target_strides[axis],
        );
    }
}

/// Materializes private false-predicate carriers for zero-space inputs that remain in the physical executable
/// signature, currently because their declared type contains dynamic dimensions. Static zero-space inputs are erased
/// before this point and never allocate a carrier.
///
/// Each carrier is allocated at the input's *physical* (bound-shaped) type from `physical_input_types`, not at the
/// value's runtime type: the compiled module declares the bound-shaped `i1` argument, and the hidden extent scalar
/// transports the logical size, so a below-bound zero-space input must still hand PJRT a bound-shaped buffer.
fn materialize_zero_space_carriers<'c>(
    client: &'c Client<'c>,
    physical_input_types: &[ArrayType],
    inputs: Vec<Array<'c>>,
) -> Result<Vec<Array<'c>>, XlaDomainError> {
    assert_eq!(physical_input_types.len(), inputs.len());
    physical_input_types
        .iter()
        .zip(inputs)
        .map(|(physical_type, input)| {
            if !physical_type.data_type().is_zero() {
                return Ok(input);
            }
            input.block_until_ready()?;
            let carrier_type = physical_type.clone().with_data_type(DataType::Boolean);
            let element_count = carrier_type
                .element_count()
                .map_err(Error::from)?
                .expect("physical bounded input types should only have static shapes");
            Ok(Array::from_host_buffer(client, carrier_type, input.mesh(), vec![0u8; element_count])?)
        })
        .collect()
}

/// Executes one PJRT executable and transposes device-major results into one buffer vector per physical output.
fn execute_pjrt_buffers<'c>(
    executable: &LoadedExecutable<'c>,
    inputs: Vec<Array<'c>>,
    donation_flags: &[bool],
    output_count: usize,
) -> Result<Execution<Vec<Vec<Buffer<'c>>>>, XlaDomainError> {
    let addressable_device_ids = executable
        .addressable_devices()?
        .iter()
        .map(|device| device.id().map_err(XlaDomainError::from))
        .collect::<Result<Vec<_>, _>>()?;
    let arguments =
        Array::into_execute_arguments_with_donation(inputs, addressable_device_ids.as_slice(), donation_flags)?;
    let (device_outputs, fence) = executable
        .execute(arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)?
        .into_parts();

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
    Ok(Execution::new(per_output_buffers, fence))
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda-13")]
    use std::time::Instant;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use ryft_core::operations::attention::{
        AttentionConfiguration, AttentionImplementation, AttentionOperandSignature,
        DotProductAttentionBackwardOperation, DotProductAttentionOperation,
    };
    use ryft_core::operations::custom_call::CustomCallOperation;
    use ryft_core::operations::random::{RandomAlgorithm, RngBitGeneratorOperation};
    use ryft_core::operations::sort::{SortDirection, SortOperation};
    use ryft_core::{
        AddOperation, AndOperation, ArrayOperation, Atan2Operation, CalleeRegionDriver, CaptureReference,
        CompareOperation, ComparisonDirection, CompilationTracer, CompiledFunctionDispatcher, ConditionOperation,
        ConstantOperation, ConvertElementTypeOperation, CustomJvpOperation, Dimension, DimensionAddOperation,
        DimensionDivFloorOperation, DimensionFromScalarOperation, DimensionRemOperation, DimensionRequirementOperation,
        DimensionSizeOperation, DimensionSubOperation, DimensionToScalarOperation, DivOperation, DotDimensionNumbers,
        DotOperation, DynamicBroadcastOperation, DynamicReshapeOperation, DynamicShapeSliceOperation, Fill,
        FreezeReference, FreezeReferenceOperation, IotaOperation, MulOperation, NegOperation, NewReference,
        NewReferenceOperation, OneOperation, PrintOperation, ReduceOperation, ReductionKind, Reference,
        ReferenceAddUpdate, ReferenceAddUpdateOperation, ReferenceRead, ReferenceReadOperation, ReferenceSwapOperation,
        ReferenceType, ScaledDotOperation, ScanOperation, ScatterDimensionNumbers, ScatterOperation, SelectOperation,
        Sharding, ShardingDimension, StaticShape, SubOperation, WhileOperation, ZeroOperation, try_jit_with_options,
    };
    use ryft_pjrt::{ClientOptions, CpuClientOptions, load_cpu_plugin};
    #[cfg(feature = "cuda-13")]
    use ryft_pjrt::{GpuClientOptions, GpuMemoryAllocator, GpuPlatform, load_cuda_13_plugin};

    use crate::experimental::shard_map::ShardMap;
    use crate::tests::{values_from_bytes, values_to_bytes};

    use super::*;

    fn attention_operation(
        scale: f64,
        causal: bool,
        residual: bool,
        dropout: Option<(f64, u64)>,
        signature: AttentionOperandSignature,
    ) -> DotProductAttentionOperation {
        DotProductAttentionOperation::new(
            AttentionConfiguration::new()
                .with_scale(scale)
                .with_causal(causal)
                .with_residual(residual)
                .with_dropout(dropout)
                .with_implementation(if dropout.is_some() {
                    AttentionImplementation::Fused
                } else {
                    AttentionImplementation::Automatic
                }),
            signature,
        )
    }

    fn attention_backward_operation(
        scale: f64,
        causal: bool,
        dropout: Option<(f64, u64)>,
        signature: AttentionOperandSignature,
    ) -> DotProductAttentionBackwardOperation {
        DotProductAttentionBackwardOperation::new(
            AttentionConfiguration::new()
                .with_scale(scale)
                .with_causal(causal)
                .with_dropout(dropout)
                .with_implementation(if dropout.is_some() {
                    AttentionImplementation::Fused
                } else {
                    AttentionImplementation::Automatic
                }),
            signature,
        )
    }

    fn domain_mesh(client: &Client<'_>, axis: &str, axis_size: usize) -> DeviceMesh {
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new(axis, axis_size, MeshAxisType::Auto).unwrap()]).unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .into_iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        DeviceMesh::new(logical_mesh, devices).unwrap()
    }

    fn array_domain<'c>(client: &'c Client<'c>) -> ProjectedContext<XlaDomain<'c>, ArrayType> {
        ProjectedContext::new(XlaDomain::new(client))
    }

    fn program_array<'a, 'c>(value: &'a ArrayIrValue<Array<'c>>) -> &'a Array<'c> {
        let ArrayIrValue::Array(array) = value else {
            panic!("expected an array IR value");
        };
        array
    }

    fn replicated_vector_type(mesh: &DeviceMesh, size: usize) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(size)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap()
    }

    fn replicated_scalar_type(mesh: &DeviceMesh, data_type: DataType) -> ArrayType {
        ArrayType::scalar(data_type)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap()
    }

    fn assert_reference_free_stable_hlo(lowered: &XlaLoweredProgram) {
        for unresolved in [
            "new_reference",
            "reference_read",
            "reference_swap",
            "reference_add_update",
            "freeze_reference",
            "ordered_state",
        ] {
            assert!(!lowered.stable_hlo().contains(unresolved), "{}", lowered.stable_hlo());
        }
    }

    fn f32_vector<'c>(client: &'c Client<'c>, mesh: &DeviceMesh, values: &[f32]) -> Array<'c> {
        let r#type = replicated_vector_type(mesh, values.len());
        Array::from_host_buffer(client, r#type, mesh.clone(), values_to_bytes::<f32>(values).as_slice()).unwrap()
    }

    fn f32_scalar<'c>(client: &'c Client<'c>, mesh: &DeviceMesh, value: f32) -> Array<'c> {
        let r#type = replicated_scalar_type(mesh, DataType::F32);
        Array::from_host_buffer(client, r#type, mesh.clone(), value.to_ne_bytes().as_slice()).unwrap()
    }

    fn f64_vector<'c>(client: &'c Client<'c>, mesh: &DeviceMesh, values: &[f64]) -> Array<'c> {
        let r#type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(values.len())]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        Array::from_host_buffer(client, r#type, mesh.clone(), values_to_bytes::<f64>(values).as_slice()).unwrap()
    }

    fn boolean_vector<'c>(client: &'c Client<'c>, mesh: &DeviceMesh, values: &[bool]) -> Array<'c> {
        let r#type = ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(values.len())]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let values = values.iter().copied().map(u8::from).collect::<Vec<_>>();
        Array::from_host_buffer(client, r#type, mesh.clone(), values.as_slice()).unwrap()
    }

    fn boolean_scalar<'c>(client: &'c Client<'c>, mesh: &DeviceMesh, value: bool) -> Array<'c> {
        let r#type = replicated_scalar_type(mesh, DataType::Boolean);
        Array::from_host_buffer(client, r#type, mesh.clone(), &[u8::from(value)]).unwrap()
    }

    fn read_f32s(client: &Client<'_>, array: &Array<'_>) -> Vec<f32> {
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let bytes = array
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        values_from_bytes::<f32>(bytes.as_slice())
    }

    fn read_bf16s(client: &Client<'_>, array: &Array<'_>) -> Vec<f32> {
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let bytes = array
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        values_from_bytes::<u16>(bytes.as_slice())
            .into_iter()
            .map(|bits| half::bf16::from_bits(bits).to_f32())
            .collect()
    }

    fn read_f64s(client: &Client<'_>, array: &Array<'_>) -> Vec<f64> {
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let bytes = array
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        values_from_bytes::<f64>(bytes.as_slice())
    }

    fn read_u64s(client: &Client<'_>, array: &Array<'_>) -> Vec<u64> {
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let bytes = array
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        values_from_bytes::<u64>(bytes.as_slice())
    }

    fn read_i64s(client: &Client<'_>, array: &Array<'_>) -> Vec<i64> {
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let bytes = array
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        values_from_bytes::<i64>(bytes.as_slice())
    }

    fn read_booleans(client: &Client<'_>, array: &Array<'_>) -> Vec<bool> {
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        array
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap()
            .into_iter()
            .map(|value| value != 0)
            .collect()
    }

    /// Builds the kind-sensitive bounded-dynamic padding fixture shared by CPU and CUDA execution tests.
    fn data_derived_padding_fixture(size_type: ArrayType, extent: DimensionVariable) -> FlatXlaProgram {
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent.clone())]));
        let index_type = dynamic_type.clone().with_data_type(DataType::U64);

        // The iota and its negation make physical suffix lanes numerically distinguishable from every admitted live
        // prefix. This catches both reduction identities and the opposite sentinels needed by ascending and descending
        // sort. The boolean reductions cover their own false/true identities, and dot covers zero-padded contraction.
        let mut builder = XlaProgramBuilder::new();
        let size = builder.add_input(size_type.into());
        let dimension =
            builder.add_instruction(DimensionFromScalarOperation::new(extent), Vec::new(), vec![size]).unwrap()[0];
        let iota = builder
            .add_instruction(IotaOperation::new(dynamic_type, 0).unwrap(), Vec::new(), vec![dimension])
            .unwrap()[0];
        let indices = builder
            .add_instruction(IotaOperation::new(index_type, 0).unwrap(), Vec::new(), vec![dimension])
            .unwrap()[0];
        let negative = builder.add_instruction(NegOperation::new(), Vec::new(), vec![iota]).unwrap()[0];
        let sorted_ascending = builder
            .add_instruction(SortOperation::new(0, SortDirection::Ascending), Vec::new(), vec![negative])
            .unwrap()[0];
        let sorted_descending = builder
            .add_instruction(SortOperation::new(0, SortDirection::Descending), Vec::new(), vec![iota])
            .unwrap()[0];
        let argmin_indices = builder
            .add_instruction(SortOperation::new(0, SortDirection::Ascending), Vec::new(), vec![negative, indices])
            .unwrap()[1];
        let argmax_indices = builder
            .add_instruction(SortOperation::new(0, SortDirection::Descending), Vec::new(), vec![iota, indices])
            .unwrap()[1];
        let sum = builder
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Sum), Vec::new(), vec![iota])
            .unwrap()[0];
        let maximum = builder
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Max), Vec::new(), vec![iota])
            .unwrap()[0];
        let minimum = builder
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Min), Vec::new(), vec![negative])
            .unwrap()[0];
        let any_input = builder
            .add_instruction(
                ArrayOperation::Compare(CompareOperation::<ArrayType>::new(ComparisonDirection::LessThan)),
                Vec::new(),
                vec![negative, iota],
            )
            .unwrap()[0];
        let any = builder
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Any), Vec::new(), vec![any_input])
            .unwrap()[0];
        let all_input = builder
            .add_instruction(
                ArrayOperation::Compare(CompareOperation::<ArrayType>::new(ComparisonDirection::GreaterThanOrEqual)),
                Vec::new(),
                vec![iota, negative],
            )
            .unwrap()[0];
        let all = builder
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::All), Vec::new(), vec![all_input])
            .unwrap()[0];
        let dot = builder
            .add_instruction(DotOperation::new(DotDimensionNumbers::inner_product()), Vec::new(), vec![iota, iota])
            .unwrap()[0];
        builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![
                    sorted_ascending,
                    sorted_descending,
                    argmin_indices,
                    argmax_indices,
                    sum,
                    maximum,
                    minimum,
                    any,
                    all,
                    dot,
                ],
                vec![Placeholder],
                vec![
                    Placeholder,
                    Placeholder,
                    Placeholder,
                    Placeholder,
                    Placeholder,
                    Placeholder,
                    Placeholder,
                    Placeholder,
                    Placeholder,
                    Placeholder,
                ],
            )
            .unwrap()
    }

    /// Builds the bounded-dynamic scaled-dot and attention proof fixture. The attention values vary along the
    /// key/value sequence axis, so including any physical suffix lane changes every valid output row.
    fn data_derived_scaled_dot_attention_fixture(
        size_type: ArrayType,
        extent: DimensionVariable,
        scaled_element_type: DataType,
        scaled_scale_type: DataType,
    ) -> FlatXlaProgram {
        let scaled_block_size = match scaled_element_type {
            DataType::F8E4M3FN | DataType::F8E5M2 => 32,
            _ => 16,
        };
        let attention_batch = DimensionVariable::new("attention_batch", extent.bounds().clone());
        let scaled_lhs_type = ArrayType::new(
            scaled_element_type,
            Shape::new(vec![Dimension::Dynamic(extent.clone()), Dimension::Static(scaled_block_size)]),
        );
        let scaled_lhs_scale_type = ArrayType::new(
            scaled_scale_type,
            Shape::new(vec![Dimension::Dynamic(extent.clone()), Dimension::Static(1)]),
        );
        let scaled_rhs_type = ArrayType::new(
            scaled_element_type,
            Shape::new(vec![Dimension::Static(1), Dimension::Static(scaled_block_size)]),
        );
        let scaled_rhs_scale_type =
            ArrayType::new(scaled_scale_type, Shape::new(vec![Dimension::Static(1), Dimension::Static(1)]));
        let attention_type = ArrayType::new(
            DataType::BF16,
            Shape::new(vec![
                Dimension::Dynamic(attention_batch.clone()),
                Dimension::Dynamic(extent.clone()),
                Dimension::Static(1),
                Dimension::Static(8),
            ]),
        );

        let mut builder = XlaProgramBuilder::new();
        let size = builder.add_input(size_type.into());
        let dimension =
            builder.add_instruction(DimensionFromScalarOperation::new(extent), Vec::new(), vec![size]).unwrap()[0];
        let batch_dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(attention_batch), Vec::new(), vec![size])
            .unwrap()[0];

        // `scaled_dot`: the runtime row extent is non-contracting. Both dynamic element/scale operands share it,
        // while the static right operand fixes one output column and the static block geometry fixes `k = 16`.
        let scaled_lhs =
            builder.add_instruction(OneOperation::new(scaled_lhs_type), Vec::new(), vec![dimension]).unwrap()[0];
        let scaled_lhs_scales = builder
            .add_instruction(OneOperation::new(scaled_lhs_scale_type), Vec::new(), vec![dimension])
            .unwrap()[0];
        let scaled_rhs =
            builder.add_instruction(OneOperation::new(scaled_rhs_type), Vec::new(), Vec::new()).unwrap()[0];
        let scaled_rhs_scales =
            builder.add_instruction(OneOperation::new(scaled_rhs_scale_type), Vec::new(), Vec::new()).unwrap()[0];
        let scaled_dot = builder
            .add_instruction(
                ScaledDotOperation::new(
                    DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new()),
                    DataType::F32,
                    true,
                    true,
                ),
                Vec::new(),
                vec![scaled_lhs, scaled_rhs, scaled_lhs_scales, scaled_rhs_scales],
            )
            .unwrap()[0];

        // Attention: equal query/key scores make the live softmax uniform, while values equal their sequence index.
        // Explicit lengths of `extent - 1` exercise the supported further-restriction contract: visible query rows
        // return the mean `(extent - 2) / 2`, while the final logical row and all physical suffix lanes stay zero.
        // Backward has exact `dQ = 0`, `dK[j] = 8 * (j - mean)`, and `dV = 1` on visible rows and zeros elsewhere.
        let query = builder
            .add_instruction(OneOperation::new(attention_type.clone()), Vec::new(), vec![batch_dimension, dimension])
            .unwrap()[0];
        let key = builder
            .add_instruction(OneOperation::new(attention_type.clone()), Vec::new(), vec![batch_dimension, dimension])
            .unwrap()[0];
        let value = builder
            .add_instruction(
                IotaOperation::new(attention_type.clone(), 1).unwrap(),
                Vec::new(),
                vec![batch_dimension, dimension],
            )
            .unwrap()[0];
        let output_cotangent = builder
            .add_instruction(OneOperation::new(attention_type), Vec::new(), vec![batch_dimension, dimension])
            .unwrap()[0];
        let logical_length =
            builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![dimension]).unwrap()[0];
        let logical_length = builder
            .add_instruction(ConvertElementTypeOperation::new(DataType::I32), Vec::new(), vec![logical_length])
            .unwrap()[0];
        let one = CpuArray::from_elements(ArrayType::scalar(DataType::I32), &[1_i32]).unwrap();
        let one = builder.add_instruction(ConstantOperation::new(one), Vec::new(), Vec::new()).unwrap()[0];
        let length = builder.add_instruction(SubOperation::new(), Vec::new(), vec![logical_length, one]).unwrap()[0];
        let lengths = builder
            .add_instruction(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), vec![length, batch_dimension])
            .unwrap()[0];
        let forward = builder
            .add_instruction(
                attention_operation(1.0, false, true, None, AttentionOperandSignature::new(false, false, true, true)),
                Vec::new(),
                vec![query, key, value, lengths, lengths],
            )
            .unwrap()
            .to_vec();
        let backward = builder
            .add_instruction(
                attention_backward_operation(
                    1.0,
                    false,
                    None,
                    AttentionOperandSignature::new(false, false, true, true),
                ),
                Vec::new(),
                vec![query, key, value, lengths, lengths, forward[0], forward[1], output_cotangent],
            )
            .unwrap()
            .to_vec();
        builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![scaled_dot, forward[0], backward[0], backward[1], backward[2]],
                vec![Placeholder],
                vec![Placeholder; 5],
            )
            .unwrap()
    }

    #[test]
    fn test_feedback_directed_profile_round_trips_and_changes_compilation_options() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("profile.pb");
        let profile = XlaFeedbackDirectedProfile::new(vec![1, 2, 3, 4]).with_version(7);
        profile.write_to_file(&path).unwrap();
        let restored = XlaFeedbackDirectedProfile::from_file(&path).unwrap().with_version(7);
        assert_eq!(restored, profile);
        assert_eq!(restored.digest(), profile.digest());

        let baseline = jit_compilation_options(&CompilationOptions::default(), 2, true, None);
        let profiled = jit_compilation_options(&CompilationOptions::default(), 2, true, Some(&profile));
        assert!(baseline.executable_build_options.unwrap().fdo_profile.is_empty());
        assert_eq!(profiled.executable_build_options.unwrap().fdo_profile, vec![1, 2, 3, 4]);
        assert_eq!(profiled.profile_version, 7);
    }

    #[test]
    fn test_replacement_metadata_rejects_changed_invocation_contract() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let signature = XlaExecutableSignature::new(&[], &[]);
        let current = XlaInvocationMetadata {
            input_types: &[],
            output_types: &[],
            signature: &signature,
            donation_flags: &[false],
            capture_count: 0,
            expected_argument_shardings: &[],
            mesh: &mesh,
        };
        let replacement = XlaInvocationMetadata { donation_flags: &[true], ..current };

        assert!(matches!(
            validate_xla_replacement_metadata(current, replacement),
            Err(XlaDomainError::InvalidCompilationOptions { reason }) if reason.contains("donation flags"),
        ));
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
    fn test_domain_zero_defaults_missing_sharding_to_replicated() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = domain_mesh(&client, "x", 2);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());

        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)]));
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
    fn test_domain_zero_constructs_a_bufferless_logical_zero_space_array() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh);
        let array_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));

        let array = domain.constant(&array_type, ConstantKind::Zero).unwrap();

        assert_eq!(array.data_type(), DataType::Zero);
        assert!(array.addressable_shards().next().unwrap().buffer().is_none());
    }

    #[test]
    fn test_domain_one_fills_sharded_array_with_ones() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = domain_mesh(&client, "x", 2);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(sharding)
            .unwrap();
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
        use ryft_core::OneOperation;
        let array_type = ArrayType::scalar(DataType::Token);

        assert!(matches!(
            XlaDomain::token().bind(OneOperation::new(array_type.clone()), Vec::new(), &[]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "xla domain cannot synthesize one value for element type token"
        ));
    }

    #[test]
    fn test_domain_identity_fast_path_rejects_attached_regions() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh);
        let array_type = ArrayType::scalar(DataType::F32);
        let operations = [
            XlaOperation::zero_operation(ArrayIrType::Array(array_type.clone())).unwrap(),
            OneOperation::new(array_type).into(),
        ];

        for operation in operations {
            let name = operation.name();
            let attached_region = XlaProgramBuilder::new()
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(Vec::new(), Vec::new(), Vec::new())
                .unwrap();
            assert_eq!(
                domain.bind(operation, vec![attached_region], &[]),
                Err(ProgramError::MalformedProgram(format!(
                    "operation `{name}` declares no region slots but 1 regions were attached",
                ))),
            );
        }
    }

    #[test]
    fn test_domain_accessors_return_constructor_arguments() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let cloned = domain.clone();

        assert_eq!(domain.mesh().unwrap(), &mesh);
        assert_eq!(domain.compilation_options(), &CompilationOptions::default());
        assert!(Arc::ptr_eq(&domain.compilation_options, &cloned.compilation_options));
        assert!(size_of::<XlaDomain<'_>>() < size_of::<CompilationOptions>());
    }

    #[test]
    fn test_eager_dimension_dispatch_runs_on_the_host_without_compilation() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh);
        let left = DimensionValue::constant(2).unwrap();
        let right = DimensionValue::constant(3).unwrap();
        let add = DimensionAddOperation::new(left.r#type().as_ref(), right.r#type().as_ref()).unwrap();

        let output = domain
            .bind(
                XlaOperation::Dimension(DimensionOperation::Add(add)),
                Vec::new(),
                &[ArrayIrValue::Dimension(left), ArrayIrValue::Dimension(right)],
            )
            .unwrap()
            .remove(0);
        let ArrayIrValue::Dimension(output) = output else {
            panic!("dimension arithmetic must produce a dimension value");
        };
        assert_eq!(output.extent(), 5);
        assert_eq!(domain.cache_size(), 0);

        let scalar = domain
            .bind(DimensionToScalarOperation, Vec::new(), &[ArrayIrValue::Dimension(output.clone())])
            .unwrap()
            .remove(0);
        let ArrayIrValue::Array(scalar) = scalar else {
            panic!("dimension_to_scalar must produce an array");
        };
        assert_eq!(read_i64s(&client, &scalar), vec![5]);
        assert_eq!(domain.cache_size(), 0);

        let from_scalar = domain
            .bind(
                ryft_core::DimensionFromScalarOperation::new(output.r#type().variable().clone()),
                Vec::new(),
                &[ArrayIrValue::Array(scalar)],
            )
            .unwrap()
            .remove(0);
        let ArrayIrValue::Dimension(from_scalar) = from_scalar else {
            panic!("dimension_from_scalar must produce a dimension value");
        };
        assert_eq!(from_scalar.extent(), 5);
        assert_eq!(domain.cache_size(), 0);
    }

    #[test]
    fn test_compiled_dimension_from_scalar_reports_observed_bounds_failure_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let input_type = replicated_scalar_type(&mesh, DataType::I64);
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());

        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type.clone().into());
        let dimension =
            builder.add_instruction(DimensionFromScalarOperation::new(extent), Vec::new(), vec![input]).unwrap()[0];
        let output = builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![dimension]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let lowered = domain.lower_xla_program(&program, 0, &XlaOptions::new(mesh.clone())).unwrap();
        assert_eq!(lowered.stable_hlo().matches("@ryft.assert").count(), 1, "{}", lowered.stable_hlo());
        let compiled = domain.compile_xla_program(&lowered).unwrap();
        assert!(compiled.requires_assertion_handler);
        // Some PJRT plugins do not serialize executables containing host callbacks. When serialization is available,
        // exercise the cache-hit path and prove its runtime-feature metadata survives restoration.
        let compiled = match domain.serialize_program(&compiled).unwrap() {
            Some(bytes) => domain.deserialize_program(bytes.as_slice()).unwrap().unwrap(),
            None => compiled,
        };
        assert!(compiled.requires_assertion_handler);

        let valid =
            Array::from_host_buffer(&client, input_type.clone(), mesh.clone(), 4_i64.to_ne_bytes().as_slice()).unwrap();
        let valid = domain.execute_xla_program(&compiled, vec![valid]).unwrap();
        assert_eq!(read_i64s(&client, &valid[0]), vec![4]);

        let invalid = Array::from_host_buffer(&client, input_type, mesh, 9_i64.to_ne_bytes().as_slice()).unwrap();
        let error = domain.execute_xla_program(&compiled, vec![invalid]).unwrap_err();
        assert!(
            error.to_string().contains(
                "`dimension_from_scalar` failed: input dimension `extent` = 9 is outside its declared bounds [1, 9)"
            ),
            "{error}",
        );
    }

    #[test]
    fn test_compiled_dimension_requirements_report_the_first_same_class_failure_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let input_type = replicated_scalar_type(&mesh, DataType::I64);
        let bounds = DimensionBounds::new(0, Some(10)).unwrap();
        let first_type = DimensionType::new(DimensionVariable::new("first", bounds));
        let second_type = DimensionType::new(DimensionVariable::new("second", bounds));
        let third_type = DimensionType::new(DimensionVariable::new("third", bounds));

        let mut builder = XlaProgramBuilder::new();
        let first_input = builder.add_input(input_type.clone().into());
        let second_input = builder.add_input(input_type.clone().into());
        let third_input = builder.add_input(input_type.clone().into());
        let first = builder
            .add_instruction(
                DimensionFromScalarOperation::new(first_type.variable().clone()),
                Vec::new(),
                vec![first_input],
            )
            .unwrap()[0];
        let second = builder
            .add_instruction(
                DimensionFromScalarOperation::new(second_type.variable().clone()),
                Vec::new(),
                vec![second_input],
            )
            .unwrap()[0];
        let third = builder
            .add_instruction(
                DimensionFromScalarOperation::new(third_type.variable().clone()),
                Vec::new(),
                vec![third_input],
            )
            .unwrap()[0];
        builder
            .add_instruction(
                DimensionRequirementOperation::less_than_or_equal(&first_type, &second_type),
                Vec::new(),
                vec![first, second],
            )
            .unwrap();
        builder
            .add_instruction(
                DimensionRequirementOperation::less_than_or_equal(&second_type, &third_type),
                Vec::new(),
                vec![second, third],
            )
            .unwrap();
        let output = builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![third]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let lowered = domain.lower_xla_program(&program, 0, &XlaOptions::new(mesh.clone())).unwrap();
        assert_eq!(lowered.stable_hlo().matches("@ryft.assert").count(), 5, "{}", lowered.stable_hlo());
        let compiled = domain.compile_xla_program(&lowered).unwrap();
        let input = |value: i64| {
            Array::from_host_buffer(&client, input_type.clone(), mesh.clone(), value.to_ne_bytes().as_slice()).unwrap()
        };

        let error = domain.execute_xla_program(&compiled, vec![input(4), input(3), input(2)]).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("`dimension_require_less_than_or_equal` failed: first <= second; observed first=4, second=3"),
            "{error}",
        );
        assert!(!error.to_string().contains("second <= third"), "{error}");
    }

    #[test]
    fn test_compiled_dimension_requirement_predicates_preserve_diagnostics_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let input_type = replicated_scalar_type(&mesh, DataType::I64);
        let bounds = DimensionBounds::new(0, Some(20)).unwrap();
        let left_type = DimensionType::new(DimensionVariable::new("left", bounds));
        let right_type = DimensionType::new(DimensionVariable::new("right", bounds));
        let check = |operation: DimensionRequirementOperation, left_value: i64, right_value: i64, expected: &str| {
            let mut builder = XlaProgramBuilder::new();
            let left_input = builder.add_input(input_type.clone().into());
            let right_input = builder.add_input(input_type.clone().into());
            let left = builder
                .add_instruction(
                    DimensionFromScalarOperation::new(left_type.variable().clone()),
                    Vec::new(),
                    vec![left_input],
                )
                .unwrap()[0];
            let right = builder
                .add_instruction(
                    DimensionFromScalarOperation::new(right_type.variable().clone()),
                    Vec::new(),
                    vec![right_input],
                )
                .unwrap()[0];
            builder.add_instruction(operation, Vec::new(), vec![left, right]).unwrap();
            let output = builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![right]).unwrap()[0];
            let program = builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![output],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder],
                )
                .unwrap();
            let lowered = domain.lower_xla_program(&program, 0, &XlaOptions::new(mesh.clone())).unwrap();
            let compiled = domain.compile_xla_program(&lowered).unwrap();
            let input = |value: i64| {
                Array::from_host_buffer(&client, input_type.clone(), mesh.clone(), value.to_ne_bytes().as_slice())
                    .unwrap()
            };

            let error = domain.execute_xla_program(&compiled, vec![input(left_value), input(right_value)]).unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        };

        check(
            DimensionRequirementOperation::equal(&left_type, &right_type),
            3,
            4,
            "`dimension_require_equal` failed: left == right; observed left=3, right=4",
        );
        check(
            DimensionRequirementOperation::divisible_by(&left_type, &right_type),
            7,
            3,
            "`dimension_require_divisible_by` failed: left % right == 0; observed left=7, right=3",
        );
        check(
            DimensionRequirementOperation::divisible_by(&left_type, &right_type),
            7,
            0,
            "`dimension_require_divisible_by` failed: right > 0 for divisibility; observed left=7, right=0",
        );
    }

    #[test]
    fn test_compiled_dimension_arithmetic_preserves_checked_host_diagnostics_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let input_type = replicated_scalar_type(&mesh, DataType::I64);
        let bounds = DimensionBounds::new(0, Some(10)).unwrap();
        let left_type = DimensionType::new(DimensionVariable::new("left", bounds));
        let right_type = DimensionType::new(DimensionVariable::new("right", bounds));
        let check = |operation: DimensionOperation<DimensionValue>,
                     valid_operands: (i64, i64),
                     expected_output: i64,
                     invalid_operands: (i64, i64),
                     expected_error: &str| {
            let mut builder = XlaProgramBuilder::new();
            let left_input = builder.add_input(input_type.clone().into());
            let right_input = builder.add_input(input_type.clone().into());
            let left = builder
                .add_instruction(
                    DimensionFromScalarOperation::new(left_type.variable().clone()),
                    Vec::new(),
                    vec![left_input],
                )
                .unwrap()[0];
            let right = builder
                .add_instruction(
                    DimensionFromScalarOperation::new(right_type.variable().clone()),
                    Vec::new(),
                    vec![right_input],
                )
                .unwrap()[0];
            let result = builder.add_instruction(operation, Vec::new(), vec![left, right]).unwrap()[0];
            let output = builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![result]).unwrap()[0];
            let program = builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![output],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder],
                )
                .unwrap();
            let lowered = domain.lower_xla_program(&program, 0, &XlaOptions::new(mesh.clone())).unwrap();
            assert_eq!(lowered.stable_hlo().matches("@ryft.assert").count(), 3, "{}", lowered.stable_hlo());
            let compiled = domain.compile_xla_program(&lowered).unwrap();
            let input = |value: i64| {
                Array::from_host_buffer(&client, input_type.clone(), mesh.clone(), value.to_ne_bytes().as_slice())
                    .unwrap()
            };

            let output = domain
                .execute_xla_program(&compiled, vec![input(valid_operands.0), input(valid_operands.1)])
                .unwrap();
            assert_eq!(read_i64s(&client, &output[0]), vec![expected_output]);
            let error = domain
                .execute_xla_program(&compiled, vec![input(invalid_operands.0), input(invalid_operands.1)])
                .unwrap_err();
            assert!(error.to_string().contains(expected_error), "{error}");
        };

        check(
            DimensionOperation::Sub(DimensionSubOperation::new(&left_type, &right_type).unwrap()),
            (5, 2),
            3,
            (2, 5),
            "left >= right; observed left=2, right=5",
        );
        check(
            DimensionOperation::DivFloor(DimensionDivFloorOperation::new(&left_type, &right_type).unwrap()),
            (7, 3),
            2,
            (7, 0),
            "right > 0; observed left=7, right=0",
        );
        check(
            DimensionOperation::Rem(DimensionRemOperation::new(&left_type, &right_type).unwrap()),
            (7, 3),
            1,
            (7, 0),
            "right > 0; observed left=7, right=0",
        );
    }

    #[test]
    fn test_compiled_dimension_assertion_does_not_merge_ordered_io_chains_on_cpu() {
        use crate::experimental::debugging::{ensure_print_handler_registered, with_captured_prints};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        ensure_print_handler_registered(&client).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let input_type = replicated_scalar_type(&mesh, DataType::I64);
        let bounds = DimensionBounds::new(0, Some(10)).unwrap();
        let left_type = DimensionType::new(DimensionVariable::new("left", bounds));
        let right_type = DimensionType::new(DimensionVariable::new("right", bounds));

        let mut builder = XlaProgramBuilder::new();
        let left_input = builder.add_input(input_type.clone().into());
        let right_input = builder.add_input(input_type.clone().into());
        builder.add_instruction(PrintOperation::new("left"), Vec::new(), vec![left_input]).unwrap();
        let left = builder
            .add_instruction(
                DimensionFromScalarOperation::new(left_type.variable().clone()),
                Vec::new(),
                vec![left_input],
            )
            .unwrap()[0];
        let right = builder
            .add_instruction(
                DimensionFromScalarOperation::new(right_type.variable().clone()),
                Vec::new(),
                vec![right_input],
            )
            .unwrap()[0];
        builder
            .add_instruction(
                DimensionRequirementOperation::less_than_or_equal(&left_type, &right_type),
                Vec::new(),
                vec![left, right],
            )
            .unwrap();
        let output = builder.add_instruction(PrintOperation::new("right"), Vec::new(), vec![right_input]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let lowered = domain.lower_xla_program(&program, 0, &XlaOptions::new(mesh.clone())).unwrap();
        assert_eq!(lowered.stable_hlo().matches("stablehlo.after_all").count(), 2, "{}", lowered.stable_hlo());
        let compiled = domain.compile_xla_program(&lowered).unwrap();
        let input = |value: i64| {
            Array::from_host_buffer(&client, input_type.clone(), mesh.clone(), value.to_ne_bytes().as_slice()).unwrap()
        };

        let (output, lines) = with_captured_prints(|| {
            let output = domain.execute_xla_program(&compiled, vec![input(3), input(4)]).unwrap();
            assert_eq!(read_i64s(&client, &output[0]), vec![4]);
            output
        });
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("left: "), "{:?}", lines);
        assert!(lines[1].starts_with("right: "), "{:?}", lines);
        assert_eq!(read_i64s(&client, &output[0]), vec![4]);
    }

    #[test]
    fn test_eager_mixed_shape_operations_specialize_dimensions_and_share_the_array_kernel_cache() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let input = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let extent = domain
            .bind(
                DimensionSizeOperation::new(input.r#type().as_ref(), 0).unwrap(),
                Vec::new(),
                &[ArrayIrValue::Array(input.clone())],
            )
            .unwrap()
            .remove(0);
        let ArrayIrValue::Dimension(extent) = extent else {
            panic!("dimension_size must produce a dimension value");
        };
        assert_eq!(extent.extent(), 6);
        assert_eq!(domain.cache_size(), 0);

        let two = DimensionValue::constant(2).unwrap();
        let division = DimensionDivFloorOperation::new(extent.r#type().as_ref(), two.r#type().as_ref()).unwrap();
        let three = domain
            .bind(
                XlaOperation::Dimension(DimensionOperation::DivFloor(division)),
                Vec::new(),
                &[ArrayIrValue::Dimension(extent), ArrayIrValue::Dimension(two.clone())],
            )
            .unwrap()
            .remove(0);
        let ArrayIrValue::Dimension(three) = three else {
            panic!("dimension division must produce a dimension value");
        };
        assert_eq!(three.extent(), 3);
        assert_eq!(domain.cache_size(), 0);

        let reshape_inputs =
            [ArrayIrValue::Array(input), ArrayIrValue::Dimension(two.clone()), ArrayIrValue::Dimension(three.clone())];
        let reshaped = domain.bind(DynamicReshapeOperation::new(), Vec::new(), &reshape_inputs).unwrap().remove(0);
        assert_eq!(program_array(&reshaped).shape().as_slice(), &[2, 3]);
        assert_eq!(domain.cache_size(), 1);

        let four = DimensionValue::constant(4).unwrap();
        let broadcast_inputs =
            [reshaped, ArrayIrValue::Dimension(four), ArrayIrValue::Dimension(two), ArrayIrValue::Dimension(three)];
        let broadcast = domain
            .bind(DynamicBroadcastOperation::new(vec![1, 2]), Vec::new(), &broadcast_inputs)
            .unwrap()
            .remove(0);
        assert_eq!(program_array(&broadcast).shape().as_slice(), &[4, 2, 3]);
        assert_eq!(read_f32s(&client, program_array(&broadcast)), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].repeat(4));
        assert_eq!(domain.cache_size(), 2);

        let repeated = domain
            .bind(DynamicBroadcastOperation::new(vec![1, 2]), Vec::new(), &broadcast_inputs)
            .unwrap()
            .remove(0);
        assert_eq!(program_array(&repeated).shape().as_slice(), &[4, 2, 3]);
        assert_eq!(domain.cache_size(), 2);
    }

    #[test]
    fn test_production_composite_lowering_executes_dynamic_dimension_arithmetic_broadcast_and_reshape() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(8)).unwrap());
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![extent.into()]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let build_program = |vector_type: &ArrayType, increment: usize| {
            let mut builder = XlaProgramBuilder::new();
            let vector = builder.add_input(vector_type.clone().into());
            let scalar = builder.add_input(scalar_type.clone().into());
            let size_operation = DimensionSizeOperation::new(vector_type, 0).unwrap();
            let size = builder.add_instruction(size_operation.clone(), Vec::new(), vec![vector]).unwrap()[0];
            let increment_value = DimensionValue::constant(increment).unwrap();
            let increment = builder
                .add_instruction(
                    XlaOperation::Dimension(DimensionOperation::Constant(ConstantOperation::new(
                        increment_value.clone(),
                    ))),
                    Vec::new(),
                    Vec::new(),
                )
                .unwrap()[0];
            let add =
                DimensionAddOperation::new(size_operation.result_type(), increment_value.r#type().as_ref()).unwrap();
            let output_extent = builder
                .add_instruction(
                    XlaOperation::Dimension(DimensionOperation::Add(add)),
                    Vec::new(),
                    vec![size, increment],
                )
                .unwrap()[0];
            let broadcast = builder
                .add_instruction(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), vec![scalar, output_extent])
                .unwrap()[0];
            let reshaped = builder
                .add_instruction(DynamicReshapeOperation::new(), Vec::new(), vec![broadcast, output_extent])
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![reshaped],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };

        let dynamic_program = build_program(&vector_type, 1);
        let dynamic_lowering = domain.lower_xla_program(&dynamic_program, 0, &XlaOptions::new(mesh.clone())).unwrap();
        assert!(dynamic_lowering.stable_hlo().contains("stablehlo.get_dimension_size"));
        assert!(dynamic_lowering.stable_hlo().contains("stablehlo.add"));
        assert!(dynamic_lowering.stable_hlo().contains("stablehlo.broadcast_in_dim"));
        assert!(dynamic_lowering.stable_hlo().contains("stablehlo.set_dimension_size"));
        assert!(dynamic_lowering.stable_hlo().contains("stablehlo.dynamic_reshape"));

        // Persistent compilation identity canonicalizes diagnostic-only variable names while retaining dimension-SSA
        // semantics. Rebuilding the same graph over a fresh nominal identity therefore shares its compilation key,
        // while changing the dimension arithmetic does not.
        let renamed_extent = DimensionVariable::new("renamed_extent", DimensionBounds::new(1, Some(8)).unwrap());
        let renamed_type = ArrayType::new(DataType::F32, Shape::new(vec![renamed_extent.into()]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let renamed_program = build_program(&renamed_type, 1);
        let renamed_lowering = domain.lower_xla_program(&renamed_program, 0, &XlaOptions::new(mesh.clone())).unwrap();
        let different_program = build_program(&vector_type, 2);
        let different_lowering =
            domain.lower_xla_program(&different_program, 0, &XlaOptions::new(mesh.clone())).unwrap();
        let dynamic_key = domain.compilation_key(&dynamic_lowering).unwrap();
        assert_eq!(dynamic_key, domain.compilation_key(&renamed_lowering).unwrap());
        assert_ne!(dynamic_key, domain.compilation_key(&different_lowering).unwrap());

        // CPU PJRT does not provide the bounded-input `PadToStatic` custom call. Execute the same first-class
        // dimension graph with a static input axis; this changes only dimension-size lowering from a runtime read to
        // its exact scalar constant and still exercises dimension SSA arithmetic and both mixed shape operations.
        let static_vector_type = replicated_vector_type(&mesh, 3);
        let static_program = build_program(&static_vector_type, 1);
        let static_lowering = domain.lower_xla_program(&static_program, 0, &XlaOptions::new(mesh.clone())).unwrap();
        let compiled = domain.compile_xla_program(&static_lowering).unwrap();
        let vector = Array::from_host_buffer(
            &client,
            static_vector_type,
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0]),
        )
        .unwrap();
        let scalar = f32_scalar(&client, &mesh, 7.0);
        let outputs = domain.execute_xla_program(&compiled, vec![vector, scalar]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].shape().as_slice(), &[4]);
        assert_eq!(read_f32s(&client, &outputs[0]), vec![7.0; 4]);
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_cuda_13_plugin_executes_bounded_dynamic_program_at_two_sizes() {
        fn trace_data_derived<'c>(
            extent: DimensionVariable,
            inputs: Vec<CompilationTracer<XlaDomain<'c>>>,
        ) -> Result<CompilationTracer<XlaDomain<'c>>, XlaDomainError> {
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

        let plugin = load_cuda_13_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::GPU(GpuClientOptions {
                platform: Some(GpuPlatform::CUDA),
                allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                ..Default::default()
            }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(8)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![extent.into()]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let staged = crate::jit::stage::<_, ArrayType, ArrayType>(
            |input| input.clone() + input,
            dynamic_type,
            &domain,
            XlaOptions::new(mesh.clone()),
        )
        .unwrap()
        .into_inner();
        let compiled: ryft_core::compilation::CompiledFunction<XlaDomain<'_>, ArrayIrType, ArrayIrType> =
            domain.compile(domain.lower(staged).unwrap()).unwrap();
        assert_eq!(domain.cache_size(), 1);

        for size in [4usize, 7] {
            let input_type = replicated_vector_type(&mesh, size);
            let values = (0..size).map(|value| value as f32).collect::<Vec<_>>();
            let input = Array::from_host_buffer(
                &client,
                input_type,
                mesh.clone(),
                values_to_bytes(values.as_slice()).as_slice(),
            )
            .unwrap();
            for _ in 0..2 {
                // Both launches use the same bounded executable. The second also proves that replay does not
                // accidentally specialize or recompile the program for the concrete logical size.
                let output = ryft_core::compilation::call_function(
                    &domain,
                    compiled.executable_function(),
                    ArrayIrValue::Array(input.clone()),
                )
                .unwrap();
                let ArrayIrValue::Array(output) = output else {
                    panic!("array-only compiled function returned a first-class dimension");
                };
                output.block_until_ready().unwrap();
                assert_eq!(output.shape().as_slice(), &[size]);
                assert_eq!(read_f32s(&client, &output), values.iter().map(|value| value * 2.0).collect::<Vec<_>>());
                assert_eq!(domain.cache_size(), 1, "executing a loaded executable must not trigger recompilation");
            }
        }

        // The same plugin route also accepts an extent born from device data inside the program. The gateway's
        // ordered bounds assertion and the reduction's dynamic masking both survive compilation, and the scalar input
        // values select different logical extents without producing another retained specialization.
        let scalar_i64 = replicated_scalar_type(&mesh, DataType::I64);
        let scalar_f32 = replicated_scalar_type(&mesh, DataType::F32);
        let extent = DimensionVariable::new("data_extent", DimensionBounds::new(1, Some(8)).unwrap());
        let function: CompiledFunctionDispatcher<XlaDomain<'_>, _, (), Vec<ArrayIrType>, ArrayIrType> =
            try_jit_with_options(
                &domain,
                move |(), inputs| trace_data_derived(extent.clone(), inputs),
                XlaOptions::new(mesh.clone()),
            );
        let call = |size: i64| {
            let size =
                Array::from_host_buffer(&client, scalar_i64.clone(), mesh.clone(), size.to_ne_bytes().as_slice())
                    .unwrap();
            let value =
                Array::from_host_buffer(&client, scalar_f32.clone(), mesh.clone(), 2.0_f32.to_ne_bytes().as_slice())
                    .unwrap();
            function.call((), vec![ArrayIrValue::Array(size), ArrayIrValue::Array(value)])
        };
        assert_eq!(read_f32s(&client, program_array(&call(4).unwrap())), vec![8.0]);
        assert_eq!(read_f32s(&client, program_array(&call(7).unwrap())), vec![14.0]);
        let invalid = call(8).unwrap();
        let error = program_array(&invalid).block_until_ready().unwrap_err();
        assert!(
            error.to_string().contains(
                "`dimension_from_scalar` failed: input dimension `data_extent` = 8 is outside its declared bounds [1, 8)"
            ),
            "{error}",
        );
        assert_eq!(function.specialization_count(), 1);
        assert_eq!(function.statistics().dispatch_misses, 1);
        assert_eq!(function.statistics().dispatch_hits, 2);
    }

    #[test]
    fn test_production_composite_lowering_executes_dynamic_shape_slice() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let input_type = replicated_vector_type(&mesh, 6);
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(input_type.clone().into());
        let start = builder
            .add_instruction(
                XlaOperation::Dimension(DimensionOperation::Constant(ConstantOperation::new(
                    DimensionValue::constant(1).unwrap(),
                ))),
                Vec::new(),
                Vec::new(),
            )
            .unwrap()[0];
        let size = builder
            .add_instruction(
                XlaOperation::Dimension(DimensionOperation::Constant(ConstantOperation::new(
                    DimensionValue::constant(3).unwrap(),
                ))),
                Vec::new(),
                Vec::new(),
            )
            .unwrap()[0];
        let output = builder
            .add_instruction(
                XlaOperation::DynamicShapeSlice(DynamicShapeSliceOperation::new(1).with_strides(vec![2]).unwrap()),
                Vec::new(),
                vec![input, start, size],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let lowering = domain.lower_xla_program(&program, 0, &XlaOptions::new(mesh.clone())).unwrap();
        assert!(lowering.stable_hlo().contains("stablehlo.dynamic_slice"));
        assert!(lowering.stable_hlo().contains("stablehlo.slice"));
        assert!(!lowering.stable_hlo().contains("stablehlo.real_dynamic_slice"));
        let compiled = domain.compile_xla_program(&lowering).unwrap();
        let input =
            Array::from_host_buffer(&client, input_type, mesh, values_to_bytes::<f32>(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
                .unwrap();
        let outputs = domain.execute_xla_program(&compiled, vec![input]).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].shape().as_slice(), &[3]);
        assert_eq!(read_f32s(&client, &outputs[0]), vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_compilation_domain_impl_round_trips_through_core_pipeline() {
        use crate::tests::{values_from_bytes, values_to_bytes};
        use ryft_core::Sin;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let engine = XlaDomain::with_mesh(&client, mesh.clone());

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let options = XlaOptions::new(mesh.clone());
        let staged =
            crate::jit::stage::<_, ArrayType, ArrayType>(|x| x.sin().unwrap(), input_type.clone(), &engine, options)
                .unwrap()
                .into_inner();
        let compiled: ryft_core::compilation::CompiledFunction<XlaDomain<'_>, ArrayIrType, ArrayIrType> =
            engine.compile(engine.lower(staged).unwrap()).unwrap();

        // Round-trip a small input through the new CompilationDomain-driven pipeline.
        let values = [0.0f32, 0.5, 1.0, 1.5];
        let source = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();
        let array =
            ryft_core::compilation::call_function(&engine, compiled.executable_function(), ArrayIrValue::Array(source))
                .unwrap();
        let ArrayIrValue::Array(array) = array else {
            panic!("array-only compiled function returned a first-class dimension");
        };
        array.block_until_ready().unwrap();

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

        // Equivalent lowerings share the entry populated by the first compilation, independent of source location.
        let cache_size_before = engine.cache_size();
        for _ in 0..3 {
            let staged = crate::jit::stage::<_, ArrayType, ArrayType>(
                |x| x.sin().unwrap(),
                input_type.clone(),
                &engine,
                XlaOptions::new(mesh.clone()),
            )
            .unwrap()
            .into_inner();
            let _: ryft_core::compilation::CompiledFunction<XlaDomain<'_>, ArrayIrType, ArrayIrType> =
                engine.compile(engine.lower(staged).unwrap()).unwrap();
        }
        assert_eq!(
            engine.cache_size(),
            cache_size_before,
            "equivalent repeat compilations should reuse the existing cache entry",
        );
    }

    #[test]
    fn test_compiled_zero_space_identity_preserves_logical_type_and_canonical_value() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let input_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let staged = crate::jit::stage::<_, ArrayType, ArrayType>(
            |input| input,
            input_type.clone(),
            &domain,
            XlaOptions::new(mesh.clone()),
        )
        .unwrap()
        .into_inner();
        let compiled: ryft_core::compilation::CompiledFunction<XlaDomain<'_>, ArrayIrType, ArrayIrType> =
            domain.compile(domain.lower(staged).unwrap()).unwrap();
        assert_eq!(compiled.compiled_program().output_types(), std::slice::from_ref(&input_type));

        let donated_staged: StagedFunction<XlaDomain<'_>, ArrayIrType, ArrayIrType> =
            crate::jit::stage::<_, ArrayType, ArrayType>(
                |input| input,
                input_type.clone(),
                &domain,
                XlaOptions { donation_flags: Some(vec![true]), ..XlaOptions::new(mesh.clone()) },
            )
            .unwrap()
            .into_inner();
        let donated_lowered = domain.lower(donated_staged).unwrap();
        assert_eq!(donated_lowered.lowered_program().donation_flags.as_ref(), &[false]);
        assert_eq!(
            domain.compilation_key(compiled.lowered().lowered_program()).unwrap(),
            domain.compilation_key(donated_lowered.lowered_program()).unwrap(),
        );

        // Boolean values retain physical `i1` arguments/results while the zero-space identity has an empty physical
        // signature. Their retained logical metadata also keeps their compilation identities distinct.
        let boolean_type = input_type.clone().with_data_type(DataType::Boolean);
        let boolean_staged: StagedFunction<XlaDomain<'_>, ArrayIrType, ArrayIrType> =
            crate::jit::stage::<_, ArrayType, ArrayType>(
                |input| input,
                boolean_type,
                &domain,
                XlaOptions::new(mesh.clone()),
            )
            .unwrap()
            .into_inner();
        let boolean_lowered = domain.lower(boolean_staged).unwrap();
        assert_ne!(
            domain.compilation_key(compiled.lowered().lowered_program()).unwrap(),
            domain.compilation_key(boolean_lowered.lowered_program()).unwrap(),
        );

        let input = Array::from_host_buffer(&client, input_type, mesh, []).unwrap();
        let output =
            ryft_core::compilation::call_function(&domain, compiled.executable_function(), ArrayIrValue::Array(input))
                .unwrap();
        let ArrayIrValue::Array(output) = output else {
            panic!("array-only compiled function returned a first-class dimension");
        };
        assert_eq!(output.data_type(), DataType::Zero);
        assert!(output.addressable_shards().next().unwrap().buffer().is_none());
    }

    #[test]
    fn test_compiled_mixed_zero_space_signature_projects_and_reconstructs_logical_values() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let value_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]))
            .with_sharding(sharding.clone())
            .unwrap();
        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]))
            .with_sharding(sharding)
            .unwrap();
        let options = XlaOptions { donation_flags: Some(vec![false, true]), ..XlaOptions::new(mesh.clone()) };
        let staged = crate::jit::stage::<_, (ArrayType, ArrayType), (ArrayType, ArrayType)>(
            |(value, zero)| (zero, value),
            (value_type.clone(), zero_type.clone()),
            &domain,
            options,
        )
        .unwrap()
        .into_inner();
        let lowered = domain.lower(staged).unwrap();
        assert!(lowered.lowered_program().stable_hlo().contains("func.func @main(%arg0: tensor<3xf32>"));
        let compiled: ryft_core::compilation::CompiledFunction<
            XlaDomain<'_>,
            (ArrayIrType, ArrayIrType),
            (ArrayIrType, ArrayIrType),
        > = domain.compile(lowered).unwrap();
        let value = Array::from_host_buffer(
            &client,
            value_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0]),
        )
        .unwrap();
        let zero = Array::from_host_buffer(&client, zero_type.clone(), mesh, []).unwrap();

        let (zero_output, value_output) = ryft_core::compilation::call_function(
            &domain,
            compiled.executable_function(),
            (ArrayIrValue::Array(value), ArrayIrValue::Array(zero)),
        )
        .unwrap();
        let ArrayIrValue::Array(zero_output) = zero_output else {
            panic!("array-only compiled function returned a first-class dimension");
        };
        let ArrayIrValue::Array(value_output) = value_output else {
            panic!("array-only compiled function returned a first-class dimension");
        };

        assert_eq!(zero_output.r#type().as_ref(), &zero_type);
        assert!(zero_output.addressable_shards().next().unwrap().buffer().is_none());
        assert_eq!(value_output.r#type().as_ref(), &value_type);
        let value_bytes = value_output
            .addressable_shards()
            .next()
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        assert_eq!(values_from_bytes::<f32>(value_bytes.as_slice()), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_bounded_dynamic_input_padding_is_dense_row_major() {
        let source = values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]);

        let padded = pad_dense_array_bytes(source.as_slice(), &[2, 2], &[3, 4], DataType::F32).unwrap();

        assert_eq!(
            values_from_bytes::<f32>(padded.as_slice()),
            vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
    }

    #[test]
    fn test_bounded_dynamic_input_packing_cost_guard() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let declared_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap()).into()]),
        )
        .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
        .unwrap();
        let signature = XlaExecutableSignature::new(std::slice::from_ref(&declared_type), &[]);

        // An input already at its bound takes the reuse tier: the physical payload preserves storage identity and the
        // only new value is its hidden logical-extent scalar.
        let exact_type = replicated_vector_type(&mesh, 8);
        let exact = Array::from_host_buffer(
            &client,
            exact_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[0.0; 8]).as_slice(),
        )
        .unwrap();
        assert_eq!(
            bounded_input_packing(&exact_type.static_shape().unwrap(), &StaticShape::new(vec![8])),
            BoundedInputPacking::Reuse,
        );
        let exact_storage = exact.clone();
        let at_bound = materialize_bounded_dynamic_inputs(
            &client,
            &signature,
            std::slice::from_ref(&declared_type),
            std::slice::from_ref(&exact_type),
            vec![exact],
        )
        .unwrap();
        assert_eq!(at_bound.inputs.len(), 2);
        assert_eq!(at_bound.inputs[0], exact_storage);
        assert_eq!(
            at_bound.report,
            BoundedInputMaterializationReport {
                at_bound_reuses: 1,
                extent_scalar_uploads: 1,
                actual_bytes: 32,
                bound_bytes: 32,
                ..Default::default()
            },
        );

        // A smaller single-shard input pays the cold padding cost exactly once and retains the physical buffer. The
        // second call reuses identical PJRT buffer Arcs and reports zero pad-path allocation and transport. Explicit
        // path-local counts avoid process-global allocator noise from PJRT and Rust's parallel test runner.
        let smaller_type = replicated_vector_type(&mesh, 4);
        assert_eq!(
            bounded_input_packing(&smaller_type.static_shape().unwrap(), &StaticShape::new(vec![8])),
            BoundedInputPacking::Pad,
        );
        let smaller = Array::from_host_buffer(
            &client,
            smaller_type.clone(),
            mesh,
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let cold = materialize_bounded_dynamic_inputs(
            &client,
            &signature,
            std::slice::from_ref(&declared_type),
            std::slice::from_ref(&smaller_type),
            vec![smaller.clone()],
        )
        .unwrap();
        assert_eq!(
            cold.report,
            BoundedInputMaterializationReport {
                retained_misses: 1,
                device_to_host_shard_copies: 1,
                host_padding_payload_allocations: 1,
                host_to_device_shard_uploads: 1,
                extent_scalar_uploads: 1,
                actual_bytes: 16,
                bound_bytes: 32,
                ..Default::default()
            },
        );
        let retained_storage = cold.inputs[0].clone();
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let donation_arguments =
            Array::into_execute_arguments_with_donation(vec![cold.inputs[0].clone()], &[device_id], &[true]).unwrap();
        assert!(!donation_arguments.inputs_by_device()[0][0].donatable);
        let warm =
            materialize_bounded_dynamic_inputs(&client, &signature, &[declared_type], &[smaller_type], vec![smaller])
                .unwrap();
        assert_eq!(warm.inputs[0], retained_storage);
        assert_eq!(
            warm.report,
            BoundedInputMaterializationReport {
                cache_hits: 1,
                extent_scalar_uploads: 0,
                actual_bytes: 16,
                bound_bytes: 32,
                ..Default::default()
            },
        );
    }

    #[test]
    fn test_independent_bounded_materialization_misses_share_one_issue_phase() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let declared_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap()).into()]),
        )
        .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
        .unwrap();
        let actual_type = replicated_vector_type(&mesh, 4);
        let left = Array::from_host_buffer(
            &client,
            actual_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let right = Array::from_host_buffer(
            &client,
            actual_type.clone(),
            mesh,
            values_to_bytes::<f32>(&[5.0, 6.0, 7.0, 8.0]).as_slice(),
        )
        .unwrap();
        let declared_types = [declared_type.clone(), declared_type];
        let actual_types = [actual_type.clone(), actual_type];
        let signature = XlaExecutableSignature::new(&declared_types, &[]);

        // The materializer's first scan reserves both cache entries and issues both D2H copies before its completion
        // loop awaits either one. The path-local report guards the two independent cold jobs and the warm zero-work
        // result without relying on scheduler timing.
        let cold = materialize_bounded_dynamic_inputs(
            &client,
            &signature,
            &declared_types,
            &actual_types,
            vec![left.clone(), right.clone()],
        )
        .unwrap();
        assert_eq!(
            cold.report,
            BoundedInputMaterializationReport {
                retained_misses: 2,
                device_to_host_shard_copies: 2,
                host_padding_payload_allocations: 2,
                host_to_device_shard_uploads: 2,
                extent_scalar_uploads: 2,
                actual_bytes: 32,
                bound_bytes: 64,
                ..Default::default()
            },
        );
        let retained = [cold.inputs[0].clone(), cold.inputs[1].clone()];

        let warm =
            materialize_bounded_dynamic_inputs(&client, &signature, &declared_types, &actual_types, vec![left, right])
                .unwrap();
        assert_eq!(warm.inputs[0], retained[0]);
        assert_eq!(warm.inputs[1], retained[1]);
        assert_eq!(
            warm.report,
            BoundedInputMaterializationReport {
                cache_hits: 2,
                actual_bytes: 32,
                bound_bytes: 64,
                ..Default::default()
            },
        );
    }

    #[test]
    fn test_failed_bounded_upload_batch_is_not_published_and_retries() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let declared_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap()).into()]),
        )
        .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
        .unwrap();
        let actual_type = replicated_vector_type(&mesh, 4);
        let left = Array::from_host_buffer(
            &client,
            actual_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let right = Array::from_host_buffer(
            &client,
            actual_type.clone(),
            mesh,
            values_to_bytes::<f32>(&[5.0, 6.0, 7.0, 8.0]).as_slice(),
        )
        .unwrap();
        let declared_types = [declared_type.clone(), declared_type];
        let actual_types = [actual_type.clone(), actual_type];
        let signature = XlaExecutableSignature::new(&declared_types, &[]);
        let mut observed_publications = 0usize;
        let mut observed_uploads = 0usize;

        // The injected failure runs only after both physical arrays and their H2D uploads exist. Returning the error
        // before publication drops both producer reservations, so neither failed result can become a warm cache hit.
        let error = match materialize_bounded_dynamic_inputs_with_readiness(
            &client,
            &signature,
            &declared_types,
            &actual_types,
            vec![left.clone(), right.clone()],
            |publications| {
                observed_publications = publications.len();
                observed_uploads = publications
                    .iter()
                    .flat_map(|publication| publication.physical.shards())
                    .filter(|shard| shard.buffer().is_some())
                    .count();
                Err(ProgramError::InvalidArgument { message: "injected bounded upload readiness failure".to_string() }
                    .into())
            },
        ) {
            Ok(_) => panic!("injected upload readiness failure must reject materialization"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("injected bounded upload readiness failure"));
        assert_eq!(observed_publications, 2);
        assert_eq!(observed_uploads, 2);

        let retry = materialize_bounded_dynamic_inputs(
            &client,
            &signature,
            &declared_types,
            &actual_types,
            vec![left.clone(), right.clone()],
        )
        .unwrap();
        assert_eq!(retry.report.retained_misses, 2);
        assert_eq!(retry.report.cache_hits, 0);
        assert_eq!(retry.report.device_to_host_shard_copies, 2);
        assert_eq!(retry.report.host_to_device_shard_uploads, 2);
        let retained = [retry.inputs[0].clone(), retry.inputs[1].clone()];

        let warm =
            materialize_bounded_dynamic_inputs(&client, &signature, &declared_types, &actual_types, vec![left, right])
                .unwrap();
        assert_eq!(warm.inputs[0], retained[0]);
        assert_eq!(warm.inputs[1], retained[1]);
        assert_eq!(warm.report.cache_hits, 2);
        assert_eq!(warm.report.device_to_host_shard_copies, 0);
        assert_eq!(warm.report.host_to_device_shard_uploads, 0);
    }

    #[test]
    fn test_sharded_bounded_materialization_reuses_every_device_buffer() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) })).unwrap();
        let mesh = domain_mesh(&client, "x", 4);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let declared_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap()).into()]),
        )
        .with_sharding(sharding.clone())
        .unwrap();
        let actual_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(sharding)
            .unwrap();
        let source = Array::from_host_buffer(
            &client,
            actual_type.clone(),
            mesh,
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let signature = XlaExecutableSignature::new(std::slice::from_ref(&declared_type), &[]);

        let cold = materialize_bounded_dynamic_inputs(
            &client,
            &signature,
            std::slice::from_ref(&declared_type),
            std::slice::from_ref(&actual_type),
            vec![source.clone()],
        )
        .unwrap();
        assert_eq!(cold.report.device_to_host_shard_copies, 4);
        assert_eq!(cold.report.host_merge_buffer_allocations, 1);
        assert_eq!(cold.report.host_padding_payload_allocations, 1);
        assert_eq!(cold.report.host_to_device_shard_uploads, 4);
        let retained = cold.inputs[0].clone();

        let warm =
            materialize_bounded_dynamic_inputs(&client, &signature, &[declared_type], &[actual_type], vec![source])
                .unwrap();
        assert_eq!(warm.inputs[0], retained);
        assert_eq!(warm.report.cache_hits, 1);
        assert_eq!(warm.report.device_to_host_shard_copies, 0);
        assert_eq!(warm.report.host_to_device_shard_uploads, 0);
    }

    #[test]
    fn test_bounded_dynamic_program_executes_on_multiple_cpu_devices() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) })).unwrap();
        let mesh = domain_mesh(&client, "x", 4);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 4);
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(4)).unwrap());
        let columns = DimensionVariable::new("columns", DimensionBounds::new(2, Some(5)).unwrap());
        let depth = DimensionVariable::new("depth", DimensionBounds::new(4, Some(7)).unwrap());
        let declared_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![
                Dimension::Static(4),
                Dimension::Dynamic(rows),
                Dimension::Dynamic(columns),
                Dimension::Dynamic(depth),
            ]),
        )
        .with_sharding(sharding.clone())
        .unwrap();
        let staged = crate::jit::stage::<_, ArrayType, ArrayType>(
            |input| input.clone() + input,
            declared_type,
            &domain,
            XlaOptions::new(mesh.clone()),
        )
        .unwrap()
        .into_inner();
        let compiled: ryft_core::compilation::CompiledFunction<XlaDomain<'_>, ArrayIrType, ArrayIrType> =
            domain.compile(domain.lower(staged).unwrap()).unwrap();
        let executable_options =
            compiled.compiled_program().compilation_options.executable_build_options.as_ref().unwrap();
        assert_eq!(executable_options.replica_count, 4);
        assert_eq!(executable_options.partition_count, 1);
        assert!(!executable_options.use_spmd_partitioning);
        assert!(!executable_options.use_shardy_partitioner);

        for shape in [[4usize, 2, 3, 5], [4, 3, 4, 6]] {
            let actual_type =
                ArrayType::new(DataType::F32, Shape::new(shape.into_iter().map(Dimension::Static).collect()))
                    .with_sharding(sharding.clone())
                    .unwrap();
            let values = (0..shape.iter().product()).map(|value| value as f32).collect::<Vec<_>>();
            let input = Array::from_host_buffer(
                &client,
                actual_type,
                mesh.clone(),
                values_to_bytes(values.as_slice()).as_slice(),
            )
            .unwrap();
            let output = ryft_core::compilation::call_function(
                &domain,
                compiled.executable_function(),
                ArrayIrValue::Array(input),
            )
            .unwrap();
            let ArrayIrValue::Array(output) = output else {
                panic!("array-only compiled function returned a first-class dimension");
            };
            output.block_until_ready().unwrap();

            assert_eq!(output.shape().as_slice(), shape.as_slice());
            assert_eq!(read_f32s(&client, &output), values.iter().map(|value| value * 2.0).collect::<Vec<_>>());
        }
        assert_eq!(domain.cache_size(), 1);

        // Dynamic tensor partitioning remains an explicit backend limitation rather than silently changing a user's
        // requested placement to replication.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(8)).unwrap());
        let sharded_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4), Dimension::Dynamic(extent)]))
                .with_sharding(
                    Sharding::new(
                        mesh.logical_mesh().clone(),
                        vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
                    )
                    .unwrap(),
                )
                .unwrap();
        let staged = crate::jit::stage::<_, ArrayType, ArrayType>(
            |input| input.clone() + input,
            sharded_type,
            &domain,
            XlaOptions::new(mesh),
        )
        .unwrap()
        .into_inner();
        let Err(error) = domain.lower(staged) else {
            panic!("bounded-dynamic sharded program unexpectedly lowered");
        };
        assert_eq!(
            error.to_string(),
            "invalid compilation options: bounded-dynamic multi-device programs currently require fully replicated \
             shardings because Shardy rejects dynamic tensors",
        );
    }

    #[test]
    fn test_internal_dynamic_tensor_behind_a_static_boundary_takes_the_replica_path() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) })).unwrap();
        let mesh = domain_mesh(&client, "x", 4);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let size_type = replicated_scalar_type(&mesh, DataType::I64);
        let value_type = replicated_scalar_type(&mesh, DataType::F32);

        // The boundary of this program is fully static, but its interior derives a dynamic extent from a runtime
        // scalar, broadcasts to it, and reduces back to a scalar.
        let mut builder = XlaProgramBuilder::new();
        let size = builder.add_input(size_type.clone().into());
        let value = builder.add_input(value_type.clone().into());
        let dimension =
            builder.add_instruction(DimensionFromScalarOperation::new(extent), Vec::new(), vec![size]).unwrap()[0];
        let broadcast = builder
            .add_instruction(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), vec![value, dimension])
            .unwrap()[0];
        let output = builder
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Sum), Vec::new(), vec![broadcast])
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let lowered = domain.lower_xla_program(&program, 0, &XlaOptions::new(mesh.clone())).unwrap();
        let executable_options = lowered.compilation_options.executable_build_options.as_ref().unwrap();
        assert_eq!(executable_options.replica_count, 4);
        assert_eq!(executable_options.partition_count, 1);
        assert!(!executable_options.use_spmd_partitioning);
        assert!(!executable_options.use_shardy_partitioner);
        assert!(!lowered.stable_hlo().contains("sdy."), "{}", lowered.stable_hlo());

        let compiled = domain.compile_xla_program(&lowered).unwrap();
        let size =
            Array::from_host_buffer(&client, size_type.clone(), mesh.clone(), 3_i64.to_ne_bytes().as_slice()).unwrap();
        let value = Array::from_host_buffer(
            &client,
            value_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[2.0]).as_slice(),
        )
        .unwrap();
        let outputs = domain.execute_xla_program(&compiled, vec![size, value]).unwrap();
        assert_eq!(read_f32s(&client, &outputs[0]), vec![6.0]);

        // The replica path is a hard requirement rather than a preference: annotating the very same program for
        // Shardy and compiling it with the Shardy partitioner enabled fails on the internal dynamic tensor, even
        // though every boundary type is static.
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let annotated = crate::experimental::lowering::lower_mlir_module_for_program(
            &program,
            &[],
            &vec![size_type, value_type.clone()],
            &vec![value_type],
            "main",
            Some(&[replicated.clone(), replicated.clone()]),
            Some(std::slice::from_ref(&replicated)),
            None,
        )
        .unwrap();
        let mut shardy_lowered = lowered.clone();
        shardy_lowered.stable_hlo = annotated.into_parts().0.into();
        shardy_lowered.compilation_options =
            jit_compilation_options(domain.compilation_options.as_ref(), 4, true, None);
        let Err(error) = domain.compile_xla_program(&shardy_lowered) else {
            panic!("Shardy unexpectedly accepted a module containing an internal dynamic tensor");
        };
        assert!(error.to_string().contains("only supports ranked tensors with a static shape"), "{error}");
    }

    #[test]
    fn test_data_derived_extent_retained_jit_reuses_one_specialization() {
        fn trace<'c>(
            extent: DimensionVariable,
            inputs: Vec<CompilationTracer<XlaDomain<'c>>>,
        ) -> Result<CompilationTracer<XlaDomain<'c>>, XlaDomainError> {
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

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let scalar_i64 = replicated_scalar_type(&mesh, DataType::I64);
        let scalar_f32 = replicated_scalar_type(&mesh, DataType::F32);
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let function: CompiledFunctionDispatcher<XlaDomain<'_>, _, (), Vec<ArrayIrType>, ArrayIrType> =
            try_jit_with_options(
                &domain,
                move |(), inputs| trace(extent.clone(), inputs),
                XlaOptions::new(mesh.clone()),
            );

        let call = |size: i64| {
            let size =
                Array::from_host_buffer(&client, scalar_i64.clone(), mesh.clone(), size.to_ne_bytes().as_slice())
                    .unwrap();
            let value =
                Array::from_host_buffer(&client, scalar_f32.clone(), mesh.clone(), 2.0_f32.to_ne_bytes().as_slice())
                    .unwrap();
            function.call((), vec![ArrayIrValue::Array(size), ArrayIrValue::Array(value)]).unwrap()
        };

        assert_eq!(read_f32s(&client, program_array(&call(2))), vec![4.0]);
        assert_eq!(read_f32s(&client, program_array(&call(4))), vec![8.0]);
        let statistics = function.statistics();
        assert_eq!(statistics.dispatch_misses, 1);
        assert_eq!(statistics.dispatch_hits, 1);
        assert_eq!(statistics.traces, 1);
        assert_eq!(statistics.lowerings, 1);
        assert_eq!(statistics.compilation_requests, 1);
        assert_eq!(function.specialization_count(), 1);
        assert_eq!(domain.cache_size(), 1);
    }

    #[test]
    fn test_data_dependent_padding_inventory_is_kind_and_kernel_specific() {
        use DataDependentPaddingDiscipline::{RyftMasked, Unsupported, XlaMasked};

        // Every reference operation shares one collapsed match arm, so one representative pins that arm.
        assert_eq!(
            data_dependent_padding_discipline(&XlaOperation::NewReference(NewReferenceOperation)),
            Unsupported { reason: "references must be discharged before bounded-dynamic XLA validation" },
        );

        for kind in [ReductionKind::Sum, ReductionKind::Max, ReductionKind::Min, ReductionKind::Any, ReductionKind::All]
        {
            assert_eq!(reduction_data_dependent_padding_discipline(kind), XlaMasked);
        }
        assert_eq!(
            reduction_data_dependent_padding_discipline(ReductionKind::Mean),
            Unsupported { reason: "mean over a dynamically sized reduction axis is not supported by the XLA lowering" },
        );
        for direction in [SortDirection::Ascending, SortDirection::Descending] {
            assert_eq!(sort_data_dependent_padding_discipline(direction), XlaMasked);
        }
        for kind in [
            ScatterReductionKind::Overwrite,
            ScatterReductionKind::Add,
            ScatterReductionKind::Mul,
            ScatterReductionKind::Min,
            ScatterReductionKind::Max,
        ] {
            assert!(matches!(
                scatter_data_dependent_padding_discipline(kind),
                Unsupported { reason } if reason.contains("scatter padding semantics have not been verified"),
            ));
        }

        assert_eq!(
            array_data_dependent_padding_discipline(&ArrayOperation::ScaledDot(ScaledDotOperation::new(
                DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new()),
                DataType::F32,
                true,
                true,
            ))),
            RyftMasked,
        );
        for operation in [
            ArrayOperation::DotProductAttention(attention_operation(
                1.0,
                false,
                false,
                None,
                AttentionOperandSignature::default(),
            )),
            ArrayOperation::DotProductAttentionBackward(attention_backward_operation(
                1.0,
                false,
                None,
                AttentionOperandSignature::default(),
            )),
        ] {
            assert_eq!(array_data_dependent_padding_discipline(&operation), RyftMasked);
        }
    }

    #[test]
    fn test_data_derived_padding_disciplines_execute_without_observable_padding() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let size_type = replicated_scalar_type(&mesh, DataType::I64);
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let program = data_derived_padding_fixture(size_type.clone(), extent);

        let lowered = domain.lower_xla_program(&program, 0, &XlaOptions::new(mesh.clone())).unwrap();
        let compiled = domain.compile_xla_program(&lowered).unwrap();
        let execute = |size: i64| {
            let input =
                Array::from_host_buffer(&client, size_type.clone(), mesh.clone(), size.to_ne_bytes().as_slice())
                    .unwrap();
            let eager = program
                .interpret_in_context(&domain, vec![ArrayIrValue::Array(input.clone())])
                .unwrap()
                .into_iter()
                .map(|value| match value {
                    ArrayIrValue::Array(array) => array,
                    ArrayIrValue::Dimension(_) => panic!("padding fixture returned a first-class dimension"),
                    ArrayIrValue::Reference(_) => panic!("padding fixture returned a reference"),
                })
                .collect::<Vec<_>>();
            let compiled = domain.execute_xla_program(&compiled, vec![input]).unwrap();
            assert_eq!(compiled.len(), eager.len());
            for (compiled, eager) in compiled.iter().zip(&eager) {
                assert_eq!(compiled.shape(), eager.shape());
                assert_eq!(compiled.data_type(), eager.data_type());
                let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
                let compiled_bytes = compiled
                    .device_shard(device_id)
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .copy_to_host(None)
                    .unwrap()
                    .r#await()
                    .unwrap();
                let eager_bytes = eager
                    .device_shard(device_id)
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .copy_to_host(None)
                    .unwrap()
                    .r#await()
                    .unwrap();
                assert_eq!(compiled_bytes, eager_bytes);
            }
            compiled
        };

        let small = execute(2);
        assert_eq!(small[0].shape(), StaticShape::new(vec![2]));
        assert_eq!(read_f32s(&client, &small[0]), vec![-1.0, 0.0]);
        assert_eq!(read_f32s(&client, &small[1]), vec![1.0, 0.0]);
        assert_eq!(read_u64s(&client, &small[2]), vec![1, 0]);
        assert_eq!(read_u64s(&client, &small[3]), vec![1, 0]);
        assert_eq!(read_f32s(&client, &small[4]), vec![1.0]);
        assert_eq!(read_f32s(&client, &small[5]), vec![1.0]);
        assert_eq!(read_f32s(&client, &small[6]), vec![-1.0]);
        assert_eq!(read_booleans(&client, &small[7]), vec![true]);
        assert_eq!(read_booleans(&client, &small[8]), vec![true]);
        assert_eq!(read_f32s(&client, &small[9]), vec![1.0]);

        let at_bound = execute(4);
        assert_eq!(at_bound[0].shape(), StaticShape::new(vec![4]));
        assert_eq!(read_f32s(&client, &at_bound[0]), vec![-3.0, -2.0, -1.0, 0.0]);
        assert_eq!(read_f32s(&client, &at_bound[1]), vec![3.0, 2.0, 1.0, 0.0]);
        assert_eq!(read_u64s(&client, &at_bound[2]), vec![3, 2, 1, 0]);
        assert_eq!(read_u64s(&client, &at_bound[3]), vec![3, 2, 1, 0]);
        assert_eq!(read_f32s(&client, &at_bound[4]), vec![6.0]);
        assert_eq!(read_f32s(&client, &at_bound[5]), vec![3.0]);
        assert_eq!(read_f32s(&client, &at_bound[6]), vec![-3.0]);
        assert_eq!(read_booleans(&client, &at_bound[7]), vec![true]);
        assert_eq!(read_booleans(&client, &at_bound[8]), vec![true]);
        assert_eq!(read_f32s(&client, &at_bound[9]), vec![14.0]);
    }

    #[test]
    fn test_data_derived_scaled_dot_and_attention_execute_without_observable_padding() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let size_type = replicated_scalar_type(&mesh, DataType::I64);
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let program =
            data_derived_scaled_dot_attention_fixture(size_type.clone(), extent, DataType::F32, DataType::F32);
        let lowered = domain.lower_xla_program(&program, 0, &XlaOptions::new(mesh.clone())).unwrap();
        assert!(lowered.stable_hlo().contains("stablehlo.composite \"xla.scaled_dot\""));
        assert!(!lowered.stable_hlo().contains("__cudnn$fmha"));
        let compiled = domain.compile_xla_program(&lowered).unwrap();
        let execute = |size: i64| {
            let input =
                Array::from_host_buffer(&client, size_type.clone(), mesh.clone(), size.to_ne_bytes().as_slice())
                    .unwrap();
            let eager = program
                .interpret_in_context(&domain, vec![ArrayIrValue::Array(input.clone())])
                .unwrap()
                .into_iter()
                .map(|value| match value {
                    ArrayIrValue::Array(array) => array,
                    ArrayIrValue::Dimension(_) => panic!("attention fixture returned a first-class dimension"),
                    ArrayIrValue::Reference(_) => panic!("attention fixture returned a reference"),
                })
                .collect::<Vec<_>>();
            let compiled = domain.execute_xla_program(&compiled, vec![input]).unwrap();
            assert_eq!(compiled.len(), eager.len());
            for (compiled, eager) in compiled.iter().zip(&eager) {
                assert_eq!(compiled.shape(), eager.shape());
                assert_eq!(compiled.data_type(), eager.data_type());
                let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
                assert_eq!(
                    compiled
                        .device_shard(device_id)
                        .unwrap()
                        .buffer()
                        .unwrap()
                        .copy_to_host(None)
                        .unwrap()
                        .r#await()
                        .unwrap(),
                    eager
                        .device_shard(device_id)
                        .unwrap()
                        .buffer()
                        .unwrap()
                        .copy_to_host(None)
                        .unwrap()
                        .r#await()
                        .unwrap(),
                );
            }
            compiled
        };

        let small = execute(2);
        assert_eq!(small[0].shape(), StaticShape::new(vec![2, 1]));
        assert_eq!(read_f32s(&client, &small[0]), vec![16.0; 2]);
        assert_eq!(small[1].shape(), StaticShape::new(vec![2, 2, 1, 8]));
        assert_eq!(read_bf16s(&client, &small[1]), vec![0.0; 32]);
        assert_eq!(read_bf16s(&client, &small[2]), vec![0.0; 32]);
        assert_eq!(read_bf16s(&client, &small[3]), vec![0.0; 32]);
        assert_eq!(
            read_bf16s(&client, &small[4]),
            [1.0, 0.0].into_iter().cycle().take(4).flat_map(|value| [value; 8]).collect::<Vec<_>>(),
        );

        let at_bound = execute(4);
        assert_eq!(at_bound[0].shape(), StaticShape::new(vec![4, 1]));
        assert_eq!(read_f32s(&client, &at_bound[0]), vec![16.0; 4]);
        assert_eq!(at_bound[1].shape(), StaticShape::new(vec![4, 4, 1, 8]));
        assert_eq!(
            read_bf16s(&client, &at_bound[1]),
            [1.0, 1.0, 1.0, 0.0].into_iter().cycle().take(16).flat_map(|value| [value; 8]).collect::<Vec<_>>(),
        );
        assert_eq!(read_bf16s(&client, &at_bound[2]), vec![0.0; 128]);
        assert_eq!(
            read_bf16s(&client, &at_bound[3]),
            [-7.78125, 0.0, 7.78125, 0.0]
                .into_iter()
                .cycle()
                .take(16)
                .flat_map(|value| [value; 8])
                .collect::<Vec<_>>(),
        );
        assert_eq!(
            read_bf16s(&client, &at_bound[4]),
            [0.97265625, 0.97265625, 0.97265625, 0.0]
                .into_iter()
                .cycle()
                .take(16)
                .flat_map(|value| [value; 8])
                .collect::<Vec<_>>(),
        );

        let zero = Array::from_host_buffer(&client, size_type, mesh, 0_i64.to_ne_bytes().as_slice()).unwrap();
        let error = domain.execute_xla_program(&compiled, vec![zero]).unwrap_err();
        assert!(
            error.to_string().contains(
                "`dimension_from_scalar` failed: input dimension `extent` = 0 is outside its declared bounds [1, 5)"
            ),
            "{error}",
        );
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_cuda_data_derived_padding_disciplines_execute_without_observable_padding() {
        let plugin = load_cuda_13_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::GPU(GpuClientOptions {
                platform: Some(GpuPlatform::CUDA),
                allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                ..Default::default()
            }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let size_type = replicated_scalar_type(&mesh, DataType::I64);
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let program = data_derived_padding_fixture(size_type.clone(), extent);
        let lowered = domain.lower_xla_program(&program, 0, &XlaOptions::new(mesh.clone())).unwrap();
        let compiled = domain.compile_xla_program(&lowered).unwrap();
        let execute = |size: i64| {
            let input =
                Array::from_host_buffer(&client, size_type.clone(), mesh.clone(), size.to_ne_bytes().as_slice())
                    .unwrap();
            domain.execute_xla_program(&compiled, vec![input]).unwrap()
        };

        let small = execute(2);
        assert_eq!(read_f32s(&client, &small[0]), vec![-1.0, 0.0]);
        assert_eq!(read_f32s(&client, &small[1]), vec![1.0, 0.0]);
        assert_eq!(read_u64s(&client, &small[2]), vec![1, 0]);
        assert_eq!(read_u64s(&client, &small[3]), vec![1, 0]);
        assert_eq!(read_f32s(&client, &small[4]), vec![1.0]);
        assert_eq!(read_f32s(&client, &small[5]), vec![1.0]);
        assert_eq!(read_f32s(&client, &small[6]), vec![-1.0]);
        assert_eq!(read_booleans(&client, &small[7]), vec![true]);
        assert_eq!(read_booleans(&client, &small[8]), vec![true]);
        assert_eq!(read_f32s(&client, &small[9]), vec![1.0]);

        let at_bound = execute(4);
        assert_eq!(read_f32s(&client, &at_bound[0]), vec![-3.0, -2.0, -1.0, 0.0]);
        assert_eq!(read_f32s(&client, &at_bound[1]), vec![3.0, 2.0, 1.0, 0.0]);
        assert_eq!(read_u64s(&client, &at_bound[2]), vec![3, 2, 1, 0]);
        assert_eq!(read_u64s(&client, &at_bound[3]), vec![3, 2, 1, 0]);
        assert_eq!(read_f32s(&client, &at_bound[4]), vec![6.0]);
        assert_eq!(read_f32s(&client, &at_bound[5]), vec![3.0]);
        assert_eq!(read_f32s(&client, &at_bound[6]), vec![-3.0]);
        assert_eq!(read_booleans(&client, &at_bound[7]), vec![true]);
        assert_eq!(read_booleans(&client, &at_bound[8]), vec![true]);
        assert_eq!(read_f32s(&client, &at_bound[9]), vec![14.0]);
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_cuda_data_derived_scaled_dot_and_attention_execute_fused_without_observable_padding() {
        let plugin = load_cuda_13_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::GPU(GpuClientOptions {
                platform: Some(GpuPlatform::CUDA),
                allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                ..Default::default()
            }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        for (scaled_element_type, scaled_scale_type) in [
            (DataType::F4E2M1FN, DataType::F8E4M3FN),
            (DataType::F8E4M3FN, DataType::F8E8M0FNU),
            (DataType::F8E5M2, DataType::F8E8M0FNU),
        ] {
            let scaled_dot_value = match scaled_element_type {
                DataType::F8E4M3FN | DataType::F8E5M2 => 32.0,
                _ => 16.0,
            };
            let size_type = replicated_scalar_type(&mesh, DataType::I64);
            let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
            let program = data_derived_scaled_dot_attention_fixture(
                size_type.clone(),
                extent,
                scaled_element_type,
                scaled_scale_type,
            );
            let lowered = domain.lower_xla_program(&program, 0, &XlaOptions::new(mesh.clone())).unwrap();
            assert!(lowered.stable_hlo().contains("stablehlo.composite \"xla.scaled_dot\""));
            assert!(lowered.stable_hlo().contains("__cudnn$fmhaSoftmax"));
            assert!(lowered.stable_hlo().contains("__cudnn$fmhaSoftmaxBackward"));
            let compiled = domain.compile_xla_program(&lowered).unwrap();
            let execute = |size: i64| {
                let input =
                    Array::from_host_buffer(&client, size_type.clone(), mesh.clone(), size.to_ne_bytes().as_slice())
                        .unwrap();
                domain.execute_xla_program(&compiled, vec![input]).unwrap()
            };

            let small = execute(2);
            assert_eq!(small[0].shape(), StaticShape::new(vec![2, 1]));
            assert_eq!(read_f32s(&client, &small[0]), vec![scaled_dot_value; 2]);
            assert_eq!(small[1].shape(), StaticShape::new(vec![2, 2, 1, 8]));
            assert_eq!(read_bf16s(&client, &small[1]), vec![0.0; 32]);
            assert_eq!(read_bf16s(&client, &small[2]), vec![0.0; 32]);
            assert_eq!(read_bf16s(&client, &small[3]), vec![0.0; 32]);
            assert_eq!(
                read_bf16s(&client, &small[4]),
                [1.0, 0.0].into_iter().cycle().take(4).flat_map(|value| [value; 8]).collect::<Vec<_>>(),
            );

            let at_bound = execute(4);
            assert_eq!(at_bound[0].shape(), StaticShape::new(vec![4, 1]));
            assert_eq!(read_f32s(&client, &at_bound[0]), vec![scaled_dot_value; 4]);
            assert_eq!(at_bound[1].shape(), StaticShape::new(vec![4, 4, 1, 8]));
            assert_eq!(
                read_bf16s(&client, &at_bound[1]),
                [1.0, 1.0, 1.0, 0.0].into_iter().cycle().take(16).flat_map(|value| [value; 8]).collect::<Vec<_>>(),
            );
            assert_eq!(read_bf16s(&client, &at_bound[2]), vec![0.0; 128]);
            let actual_query_gradient = read_bf16s(&client, &at_bound[3]);
            let expected_query_gradient =
                [-8.0, 0.0, 8.0, 0.0].into_iter().cycle().take(16).flat_map(|value| [value; 8]).collect::<Vec<_>>();
            // cuDNN's fused BF16 backward-attention kernel is not bit-identical to the unfused reference
            // composition, so compare its gradients with an explicit reduced-precision tolerance.
            actual_query_gradient.iter().zip(&expected_query_gradient).for_each(|(actual, expected)| {
                assert!((actual - expected).abs() <= 0.25, "expected {expected} within 0.25, but found {actual}");
            });
            let actual_value_gradient = read_bf16s(&client, &at_bound[4]);
            let expected_value_gradient =
                [1.0, 1.0, 1.0, 0.0].into_iter().cycle().take(16).flat_map(|value| [value; 8]).collect::<Vec<_>>();
            actual_value_gradient.iter().zip(&expected_value_gradient).for_each(|(actual, expected)| {
                assert!((actual - expected).abs() <= 0.25, "expected {expected} within 0.25, but found {actual}");
            });

            let zero =
                Array::from_host_buffer(&client, size_type, mesh.clone(), 0_i64.to_ne_bytes().as_slice()).unwrap();
            let error = domain.execute_xla_program(&compiled, vec![zero]).unwrap_err();
            assert!(
                error.to_string().contains(
                    "`dimension_from_scalar` failed: input dimension `extent` = 0 is outside its declared bounds [1, 5)"
                ),
                "{error}",
            );
        }
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "hardware throughput benchmark; run explicitly on a CUDA system"]
    fn bench_cuda_fused_attention_throughput() {
        let plugin = load_cuda_13_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::GPU(GpuClientOptions {
                platform: Some(GpuPlatform::CUDA),
                allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                ..Default::default()
            }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let [batch, sequence, heads, head_dimension] = [4, 2048, 32, 128];
        let r#type = ArrayType::new(
            DataType::BF16,
            Shape::new(vec![
                Dimension::Static(batch),
                Dimension::Static(sequence),
                Dimension::Static(heads),
                Dimension::Static(head_dimension),
            ]),
        )
        .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 4))
        .unwrap();
        let mut builder = XlaProgramBuilder::new();
        let query = builder.add_input(r#type.clone().into());
        let key = builder.add_input(r#type.clone().into());
        let value = builder.add_input(r#type.clone().into());
        let output = builder
            .add_instruction(
                attention_operation(
                    1.0 / (head_dimension as f64).sqrt(),
                    false,
                    false,
                    None,
                    AttentionOperandSignature::default(),
                ),
                Vec::new(),
                vec![query, key, value],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let lowered = domain.lower_xla_program(&program, 0, &XlaOptions::new(mesh.clone())).unwrap();
        assert!(lowered.stable_hlo().contains("__cudnn$fmhaSoftmax"));
        let compilation_start = Instant::now();
        let compiled = domain.compile_xla_program(&lowered).unwrap();
        let compilation_duration = compilation_start.elapsed();

        let one = 0x3f80_u16.to_ne_bytes();
        let mut bytes = vec![0_u8; batch * sequence * heads * head_dimension * size_of::<u16>()];
        bytes.chunks_exact_mut(size_of::<u16>()).for_each(|element| element.copy_from_slice(&one));
        let input = Array::from_host_buffer(&client, r#type, mesh, bytes.as_slice()).unwrap();
        input.block_until_ready().unwrap();
        let execute = || {
            let outputs = domain.execute_xla_program(&compiled, vec![input.clone(), input.clone(), input.clone()])?;
            outputs[0].block_until_ready()?;
            Ok::<_, XlaDomainError>(())
        };
        // The Spark idles at its lowest power state, so warm the fused kernel long enough for clocks to stabilize
        // before collecting synchronized samples.
        (0..20).for_each(|_| execute().unwrap());
        let mut samples = (0..20)
            .map(|_| {
                let start = Instant::now();
                execute().unwrap();
                start.elapsed().as_nanos()
            })
            .collect::<Vec<_>>();
        samples.sort_unstable();
        let median_nanoseconds = samples[samples.len() / 2];
        let floating_point_operations =
            4.0 * batch as f64 * heads as f64 * sequence as f64 * sequence as f64 * head_dimension as f64;
        let throughput_teraflops = floating_point_operations / (median_nanoseconds as f64 * 1e3);

        eprintln!(
            "fused attention b{batch} h{heads} s{sequence} d{head_dimension}: compile={compilation_duration:?}, median={:.3} ms, throughput={throughput_teraflops:.2} TFLOP/s",
            median_nanoseconds as f64 / 1e6,
        );
    }

    #[test]
    fn test_data_derived_prefix_slice_executes_at_multiple_extents() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let input_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let mask_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(input_sharding.clone())
            .unwrap();
        let values_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(input_sharding)
            .unwrap();
        let count_variable = DimensionVariable::new("count", DimensionBounds::new(0, Some(5)).unwrap());

        // This is the compiled counterpart of the Phase 4 prefix-take fixture: the mask determines a device-side
        // count, the checked gateway turns that count into a bounded dimension, and the slice returns that many
        // leading values without a host readback.
        let mut builder = XlaProgramBuilder::new();
        let mask = builder.add_input(mask_type.into());
        let values = builder.add_input(values_type.into());
        let mask = builder
            .add_instruction(ConvertElementTypeOperation::<ArrayType>::new(DataType::I64), Vec::new(), vec![mask])
            .unwrap()[0];
        let count = builder
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Sum), Vec::new(), vec![mask])
            .unwrap()[0];
        let count = builder
            .add_instruction(DimensionFromScalarOperation::new(count_variable), Vec::new(), vec![count])
            .unwrap()[0];
        let start = builder.add_constant(XlaConstant::Dimension(DimensionValue::constant(0).unwrap()));
        let output = builder
            .add_instruction(DynamicShapeSliceOperation::new(1), Vec::new(), vec![values, start, count])
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let lowered = domain.lower_xla_program(&program, 0, &XlaOptions::new(mesh.clone())).unwrap();
        let compiled = domain.compile_xla_program(&lowered).unwrap();
        let values = || f32_vector(&client, &mesh, &[10.0, 20.0, 30.0, 40.0]);
        let execute = |mask: &[bool]| {
            domain.execute_xla_program(&compiled, vec![boolean_vector(&client, &mesh, mask), values()]).unwrap()
        };

        let two = execute(&[true, false, true, false]);
        assert_eq!(two[0].shape(), StaticShape::new(vec![2]));
        assert_eq!(read_f32s(&client, &two[0]), vec![10.0, 20.0]);
        let four = execute(&[true, true, true, true]);
        assert_eq!(four[0].shape(), StaticShape::new(vec![4]));
        assert_eq!(read_f32s(&client, &four[0]), vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_data_derived_dynamic_shape_slice_checks_runtime_input_and_result_extents() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let size_type = replicated_scalar_type(&mesh, DataType::I64);
        let value_type = replicated_scalar_type(&mesh, DataType::F32);
        let input_extent = DimensionVariable::new("input_extent", DimensionBounds::new(1, Some(5)).unwrap());
        let result_extent = DimensionVariable::new("result_extent", DimensionBounds::new(0, Some(5)).unwrap());

        // The result extent is independent of the input extent at the type level. Lowering uses their finite physical
        // bounds for one specialization and preserves the eager `result_extent <= input_extent` contract with an
        // ordered runtime assertion.
        let mut builder = XlaProgramBuilder::new();
        let input_size = builder.add_input(size_type.clone().into());
        let result_size = builder.add_input(size_type.clone().into());
        let value = builder.add_input(value_type.clone().into());
        let input_dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(input_extent), Vec::new(), vec![input_size])
            .unwrap()[0];
        let result_dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(result_extent), Vec::new(), vec![result_size])
            .unwrap()[0];
        let input = builder
            .add_instruction(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), vec![value, input_dimension])
            .unwrap()[0];
        let start = builder.add_constant(XlaConstant::Dimension(DimensionValue::constant(0).unwrap()));
        let output = builder
            .add_instruction(DynamicShapeSliceOperation::new(1), Vec::new(), vec![input, start, result_dimension])
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let lowered = domain.lower_xla_program(&program, 0, &XlaOptions::new(mesh.clone())).unwrap();
        assert!(lowered.stable_hlo().contains("@ryft.assert"));
        let compiled = domain.compile_xla_program(&lowered).unwrap();
        let scalar =
            |r#type: ArrayType, bytes: &[u8]| Array::from_host_buffer(&client, r#type, mesh.clone(), bytes).unwrap();
        let inputs = |input_size: i64, result_size: i64| {
            vec![
                scalar(size_type.clone(), input_size.to_ne_bytes().as_slice()),
                scalar(size_type.clone(), result_size.to_ne_bytes().as_slice()),
                scalar(value_type.clone(), 7.0_f32.to_ne_bytes().as_slice()),
            ]
        };

        let eager = program
            .interpret_in_context(&domain, inputs(4, 2).into_iter().map(ArrayIrValue::Array).collect::<Vec<_>>())
            .unwrap();
        let compiled_valid = domain.execute_xla_program(&compiled, inputs(4, 2)).unwrap();
        assert_eq!(read_f32s(&client, program_array(&eager[0])), vec![7.0, 7.0]);
        assert_eq!(read_f32s(&client, &compiled_valid[0]), read_f32s(&client, program_array(&eager[0])));

        let error = match domain.execute_xla_program(&compiled, inputs(2, 4)) {
            Ok(invalid) => invalid[0].block_until_ready().unwrap_err().to_string(),
            Err(error) => error.to_string(),
        };
        assert_eq!(error, "`dynamic_shape_slice` limit 4 exceeds input axis 0 extent 2");
    }

    #[test]
    fn test_data_derived_compilation_rejects_unbounded_and_opaque_consumers() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let size_type = replicated_scalar_type(&mesh, DataType::I64);

        let mut builder = XlaProgramBuilder::new();
        let size = builder.add_input(size_type.clone().into());
        let extent = DimensionVariable::new("unbounded", DimensionBounds::unbounded());
        let dimension =
            builder.add_instruction(DimensionFromScalarOperation::new(extent), Vec::new(), vec![size]).unwrap()[0];
        let output = builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![dimension]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            domain.lower_xla_program(&program, 0, &XlaOptions::new(mesh.clone())),
            Err(XlaDomainError::Tracing(ProgramError::InvalidArgument { message }))
                if message == "data-derived dimension `unbounded` with bounds [0, ∞) needs a finite upper bound \
                    for XLA compilation",
        ));

        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent.clone())]));
        let mut builder = XlaProgramBuilder::new();
        let size = builder.add_input(size_type.clone().into());
        let dimension =
            builder.add_instruction(DimensionFromScalarOperation::new(extent), Vec::new(), vec![size]).unwrap()[0];
        let input = builder
            .add_instruction(IotaOperation::new(dynamic_type.clone(), 0).unwrap(), Vec::new(), vec![dimension])
            .unwrap()[0];
        let output = builder
            .add_instruction(
                CustomCallOperation::new("ryft.test.opaque_dynamic", vec![dynamic_type]),
                Vec::new(),
                vec![input, dimension],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            domain.lower_xla_program(&program, 0, &XlaOptions::new(mesh.clone())),
            Err(XlaDomainError::Tracing(ProgramError::UnsupportedOperation { message }))
                if message == "operation `custom_call` cannot consume data-derived dimensions during XLA lowering \
                    because custom-call physical-padding semantics are opaque",
        ));

        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent.clone())]));
        let mut builder = XlaProgramBuilder::new();
        let size = builder.add_input(size_type.clone().into());
        let dimension =
            builder.add_instruction(DimensionFromScalarOperation::new(extent), Vec::new(), vec![size]).unwrap()[0];
        let input = builder
            .add_instruction(IotaOperation::new(dynamic_type, 0).unwrap(), Vec::new(), vec![dimension])
            .unwrap()[0];
        let output = builder
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Mean), Vec::new(), vec![input])
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            domain.lower_xla_program(&program, 0, &XlaOptions::new(mesh.clone())),
            Err(XlaDomainError::Tracing(ProgramError::UnsupportedOperation { message }))
                if message == "operation `reduce_mean` cannot consume data-derived dimensions during XLA lowering \
                    because mean over a dynamically sized reduction axis is not supported by the XLA lowering",
        ));

        // `print` renders whatever the physical buffer holds, so the padded suffix of a bounded-dynamic operand would
        // become observable.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent.clone())]));
        let mut builder = XlaProgramBuilder::new();
        let size = builder.add_input(size_type.clone().into());
        let dimension =
            builder.add_instruction(DimensionFromScalarOperation::new(extent), Vec::new(), vec![size]).unwrap()[0];
        let input = builder
            .add_instruction(IotaOperation::new(dynamic_type, 0).unwrap(), Vec::new(), vec![dimension])
            .unwrap()[0];
        let output =
            builder.add_instruction(PrintOperation::<ArrayType>::new("value"), Vec::new(), vec![input]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            domain.lower_xla_program(&program, 0, &XlaOptions::new(mesh.clone())),
            Err(XlaDomainError::Tracing(ProgramError::UnsupportedOperation { message }))
                if message == "operation `print` cannot consume data-derived dimensions during XLA lowering because \
                    print exposes the physical representation of its operand",
        ));

        // Bit generation over a bounded-dynamic result would advance the functional generator state by the physical
        // upper-bound element count rather than the logical one.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let bits_type = ArrayType::new(DataType::U32, Shape::new(vec![Dimension::Dynamic(extent.clone())]));
        let mut builder = XlaProgramBuilder::new();
        let size = builder.add_input(size_type.clone().into());
        let state = builder.add_input(RandomAlgorithm::ThreeFry.state_type().into());
        let dimension =
            builder.add_instruction(DimensionFromScalarOperation::new(extent), Vec::new(), vec![size]).unwrap()[0];
        let output = builder
            .add_instruction(
                RngBitGeneratorOperation::<ArrayIrType>::new(RandomAlgorithm::ThreeFry, bits_type),
                Vec::new(),
                vec![state, dimension],
            )
            .unwrap()[1];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert!(matches!(
            domain.lower_xla_program(&program, 0, &XlaOptions::new(mesh.clone())),
            Err(XlaDomainError::Tracing(ProgramError::UnsupportedOperation { message }))
                if message == "operation `rng_bit_generator` cannot consume data-derived dimensions during XLA \
                    lowering because random generation advances state according to physical rather than logical \
                    element count",
        ));

        // Every scatter combiner is unverified against physical padding, so a data-derived operand extent is rejected
        // per kind rather than per operation name.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent.clone())]));
        let indices_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2), Dimension::Static(1)]));
        let updates_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]));
        let mut builder = XlaProgramBuilder::new();
        let size = builder.add_input(size_type.into());
        let indices = builder.add_input(indices_type.into());
        let updates = builder.add_input(updates_type.into());
        let dimension =
            builder.add_instruction(DimensionFromScalarOperation::new(extent), Vec::new(), vec![size]).unwrap()[0];
        let operand = builder
            .add_instruction(IotaOperation::new(dynamic_type, 0).unwrap(), Vec::new(), vec![dimension])
            .unwrap()[0];
        let output = builder
            .add_instruction(
                ScatterOperation::new(
                    ScatterDimensionNumbers::new(Vec::new(), vec![0], vec![0]),
                    ScatterReductionKind::Add,
                ),
                Vec::new(),
                vec![operand, indices, updates],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert!(matches!(
            domain.lower_xla_program(&program, 0, &XlaOptions::new(mesh)),
            Err(XlaDomainError::Tracing(ProgramError::UnsupportedOperation { message }))
                if message == "operation `scatter` cannot consume data-derived dimensions during XLA lowering because \
                    add-scatter padding semantics have not been verified for data-derived dimensions",
        ));
    }

    #[test]
    fn test_input_bound_bucketing_reuses_power_of_two_specializations() {
        fn trace<'c>(
            input: CompilationTracer<XlaDomain<'c>>,
        ) -> Result<CompilationTracer<XlaDomain<'c>>, XlaDomainError> {
            let context = input.context().clone();
            Ok(context.bind(AddOperation::new(), Vec::new(), &[input.clone(), input])?.remove(0))
        }

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let function: CompiledFunctionDispatcher<XlaDomain<'_>, _, (), ArrayIrType, ArrayIrType> = try_jit_with_options(
            &domain,
            |(), input| trace(input),
            XlaOptions::new(mesh.clone()).with_input_bound_bucketing(XlaInputBoundBucketing::PowerOfTwo),
        );
        let call = |values: &[f32]| {
            let input = f32_vector(&client, &mesh, values);
            function.call((), ArrayIrValue::Array(input)).unwrap()
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
        assert_eq!(function.specialization_count(), 2);
        assert_eq!(domain.cache_size(), 2);
    }

    #[test]
    fn test_input_bound_bucketing_rejects_unsupported_signatures() {
        let plugin = load_cpu_plugin().unwrap();
        let single_device_client =
            plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let single_device_mesh = domain_mesh(&single_device_client, "x", 1);
        let single_device_domain = XlaDomain::with_mesh(&single_device_client, single_device_mesh.clone());
        let dimension_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        assert!(matches!(
            single_device_domain.dispatch_signature(
                vec![dimension_type.clone().into()],
                &XlaOptions::new(single_device_mesh.clone())
                    .with_input_bound_bucketing(XlaInputBoundBucketing::PowerOfTwo),
            ),
            Err(XlaDomainError::InvalidCompilationOptions { reason })
                if reason == format!(
                    "input-bound bucketing only supports array inputs, but input 0 has first-class dimension type {}",
                    dimension_type,
                ),
        ));
        // Reference inputs are rejected before any mesh- or option-specific constraint, so the same targeted
        // diagnostic fires with bucketing options, without them, and on multi-device meshes whose bucketing
        // rejection would otherwise come first.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        for options in [
            XlaOptions::new(single_device_mesh.clone()),
            XlaOptions::new(single_device_mesh.clone()).with_input_bound_bucketing(XlaInputBoundBucketing::PowerOfTwo),
        ] {
            assert!(matches!(
                single_device_domain.dispatch_signature(vec![ArrayIrType::Reference(reference_type.clone())], &options),
                Err(XlaDomainError::Tracing(ProgramError::UnsupportedOperation { message }))
                    if message == format!(
                        "references must be discharged before XLA compilation, but dispatch input has type \
                        `{reference_type}`",
                    ),
            ));
        }

        let multi_device_client =
            plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let multi_device_mesh = domain_mesh(&multi_device_client, "x", 2);
        let multi_device_domain = XlaDomain::with_mesh(&multi_device_client, multi_device_mesh.clone());
        assert!(matches!(
            multi_device_domain.dispatch_signature(
                vec![ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])).into()],
                &XlaOptions::new(multi_device_mesh)
                    .with_input_bound_bucketing(XlaInputBoundBucketing::PowerOfTwo),
            ),
            Err(XlaDomainError::InvalidCompilationOptions { reason })
                if reason == "input-bound bucketing is unsupported on multi-device meshes because bucketed dynamic \
                    signatures require disabling the Shardy partitioner",
        ));

        // A reference input on the same multi-device bucketing options still gets the targeted discharge diagnostic:
        // the reference entry guard precedes the multi-device bucketing rejection asserted just above.
        assert!(matches!(
            multi_device_domain.dispatch_signature(
                vec![ArrayIrType::Reference(reference_type.clone())],
                &XlaOptions::new(domain_mesh(&multi_device_client, "x", 2))
                    .with_input_bound_bucketing(XlaInputBoundBucketing::PowerOfTwo),
            ),
            Err(XlaDomainError::Tracing(ProgramError::UnsupportedOperation { message }))
                if message == format!(
                    "references must be discharged before XLA compilation, but dispatch input has type \
                    `{reference_type}`",
                ),
        ));
    }

    #[test]
    fn test_input_bound_bucketing_alpha_normalizes_dispatch_keys() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let options = XlaOptions::new(mesh).with_input_bound_bucketing(XlaInputBoundBucketing::PowerOfTwo);
        let bounds = DimensionBounds::new(0, Some(5)).unwrap();
        let vector_type =
            |dimension: Dimension| ArrayIrType::from(ArrayType::new(DataType::F32, Shape::new(vec![dimension])));
        let key = |input_types: Vec<ArrayIrType>| domain.dispatch_signature(input_types, &options).unwrap().0;

        // Two runtime signatures landing in the same power-of-two bucket share one key, and the effective staged
        // types carry the generated per-axis variable names that the key itself excludes.
        let (three_key, three_types) =
            domain.dispatch_signature(vec![vector_type(Dimension::Static(3))], &options).unwrap();
        assert_eq!(three_key, key(vec![vector_type(Dimension::Static(4))]));
        assert_ne!(three_key, key(vec![vector_type(Dimension::Static(5))]));
        let [ArrayIrType::Array(three_type)] = three_types.as_ref() else {
            panic!("bucketing must stage exactly one array input type");
        };
        let [Dimension::Dynamic(variable)] = three_type.shape().dimensions() else {
            panic!("bucketing must stage a rank-one bounded dynamic axis");
        };
        assert_eq!(variable.name(), "input_0_axis_0");
        assert_eq!(variable.bounds(), bounds);

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
    fn test_bounded_dynamic_programs_reject_shard_map_regions_before_device_count_matters() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Manual).unwrap()]).unwrap();
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let size_type = replicated_scalar_type(&mesh, DataType::I64);
        let value_type = replicated_scalar_type(&mesh, DataType::F32);
        let global_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let local_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)]));

        // A `shard_map` body always has static boundary types, so it can coexist with dynamic tensors that are staged
        // beside it. Here the gateway makes the program bounded-dynamic while the `shard_map` stays fully static.
        let mut body_builder = XlaProgramBuilder::new();
        let local = body_builder.add_input(local_type.into());
        let doubled = body_builder.add_instruction(AddOperation::new(), Vec::new(), vec![local, local]).unwrap()[0];
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![doubled], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let manual_sharding = Sharding::new(manual_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let shard_map =
            ShardMap::new(manual_mesh, vec![manual_sharding.clone()], vec![manual_sharding], Vec::new(), true).unwrap();

        let mut builder = XlaProgramBuilder::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let size = builder.add_input(size_type.into());
        let value = builder.add_input(value_type.into());
        let global = builder.add_input(global_type.clone().into());
        let dimension =
            builder.add_instruction(DimensionFromScalarOperation::new(extent), Vec::new(), vec![size]).unwrap()[0];
        let dynamic = builder
            .add_instruction(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), vec![value, dimension])
            .unwrap()[0];
        let mapped = builder
            .add_instruction(
                XlaOperation::ShardMap(Box::new(ShardMapOperation::from_boundary(
                    shard_map,
                    vec![global_type.clone()],
                    vec![global_type],
                ))),
                vec![body_region],
                vec![global],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![mapped, dynamic],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();

        let Err(error) = domain.lower_xla_program(&program, 0, &XlaOptions::new(mesh)) else {
            panic!("bounded-dynamic program containing a shard_map unexpectedly lowered");
        };
        assert_eq!(
            error.to_string(),
            "invalid compilation options: bounded-dynamic programs cannot contain shard_map regions because \
             sdy.manual_computation requires the Shardy partitioner",
        );
    }

    #[test]
    fn test_concurrent_bounded_materialization_is_single_flight() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let declared_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![DimensionVariable::new("extent", DimensionBounds::new(0, Some(9)).unwrap()).into()]),
        )
        .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
        .unwrap();
        let actual_type = replicated_vector_type(&mesh, 4);
        let source = Array::from_host_buffer(
            &client,
            actual_type.clone(),
            mesh,
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let signature = XlaExecutableSignature::new(std::slice::from_ref(&declared_type), &[]);

        let (left, right) = std::thread::scope(|scope| {
            let left_source = source.clone();
            let left_declared_type = declared_type.clone();
            let left_actual_type = actual_type.clone();
            let left = scope.spawn(|| {
                materialize_bounded_dynamic_inputs(
                    &client,
                    &signature,
                    &[left_declared_type],
                    &[left_actual_type],
                    vec![left_source],
                )
                .unwrap()
            });
            let right = scope.spawn(|| {
                materialize_bounded_dynamic_inputs(
                    &client,
                    &signature,
                    std::slice::from_ref(&declared_type),
                    std::slice::from_ref(&actual_type),
                    vec![source],
                )
                .unwrap()
            });
            (left.join().unwrap(), right.join().unwrap())
        });

        assert_eq!(left.inputs[0], right.inputs[0]);
        let retained_misses = left.report.retained_misses + right.report.retained_misses;
        let cache_hits = left.report.cache_hits + right.report.cache_hits;
        assert_eq!(retained_misses, 1);
        assert_eq!(cache_hits, 1);
        assert_eq!(left.report.device_to_host_shard_copies + right.report.device_to_host_shard_copies, 1);
        assert_eq!(left.report.host_to_device_shard_uploads + right.report.host_to_device_shard_uploads, 1);
    }

    #[test]
    fn test_materialize_zero_space_carriers_allocate_the_bounded_physical_shape() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        // The compiled module declares the bound-shaped `[3]` argument, so a below-bound `[2]` runtime value must
        // still produce a `[3]` carrier; the hidden extent scalar transports the logical size.
        let physical_zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]))
            .with_sharding(sharding.clone())
            .unwrap();
        let runtime_zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(2)]))
            .with_sharding(sharding)
            .unwrap();
        let zero = Array::from_host_buffer(&client, runtime_zero_type, mesh, []).unwrap();

        let carriers = materialize_zero_space_carriers(&client, &[physical_zero_type], vec![zero]).unwrap();

        assert_eq!(carriers.len(), 1);
        assert_eq!(carriers[0].data_type(), DataType::Boolean);
        assert_eq!(carriers[0].shape().as_slice(), &[3]);
        let carrier_bytes = carriers[0]
            .addressable_shards()
            .next()
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        assert_eq!(carrier_bytes.as_slice(), &[0, 0, 0]);
    }

    #[test]
    fn test_below_bound_zero_space_inputs_skip_the_padding_tiers_and_carry_the_bounded_shape() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let declared_type = ArrayType::new(
            DataType::Zero,
            Shape::new(vec![DimensionVariable::new("extent", DimensionBounds::non_negative(Some(4)).unwrap()).into()]),
        )
        .with_sharding(sharding.clone())
        .unwrap();
        let signature = XlaExecutableSignature::new(std::slice::from_ref(&declared_type), &[]);
        let actual_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(2)]))
            .with_sharding(sharding)
            .unwrap();
        let zero = Array::from_host_buffer(&client, actual_type.clone(), mesh, []).unwrap();

        // The below-bound zero-space input must bypass the retained-cache and byte-pad tiers (its dense host copy is
        // empty) and rely on the carrier materialization to synthesize the bound-shaped predicate argument.
        let materialized = materialize_bounded_dynamic_inputs(
            &client,
            &signature,
            std::slice::from_ref(&declared_type),
            std::slice::from_ref(&actual_type),
            vec![zero],
        )
        .unwrap();
        assert_eq!(materialized.inputs.len(), 2);
        assert_eq!(
            materialized.report,
            BoundedInputMaterializationReport { extent_scalar_uploads: 1, ..Default::default() },
        );

        let physical_types = signature.physical_input_types(std::slice::from_ref(&declared_type));
        let carriers = materialize_zero_space_carriers(&client, &physical_types, materialized.inputs).unwrap();
        assert_eq!(carriers[0].data_type(), DataType::Boolean);
        assert_eq!(carriers[0].shape().as_slice(), &[3]);
        assert_eq!(carriers[1].data_type(), DataType::I32);
    }

    #[test]
    fn test_xla_compilation_key_is_canonical_and_stable() {
        use ryft_core::Sin;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let staged = crate::jit::stage::<_, ArrayType, ArrayType>(
            |x| x.sin().unwrap(),
            input_type,
            &domain,
            XlaOptions::new(mesh),
        )
        .unwrap()
        .into_inner();
        let compiled: ryft_core::compilation::CompiledFunction<XlaDomain<'_>, ArrayIrType, ArrayIrType> =
            domain.compile(domain.lower(staged).unwrap()).unwrap();
        let key = domain.compilation_key(compiled.lowered().lowered_program()).unwrap();
        let clientless_domain = XlaDomain::clientless();
        let repeated = clientless_domain.compilation_key(compiled.lowered().lowered_program()).unwrap();

        assert_eq!(key, repeated);
        assert_eq!(domain.persistent_cache_key(&key), Some(key.canonical_bytes.to_vec()));
        let decoded: serde_json::Value = serde_json::from_slice(&key.canonical_bytes).unwrap();
        assert_eq!(decoded["schema_version"], XLA_PERSISTENT_KEY_SCHEMA_VERSION);
        assert_eq!(decoded["compiler_identity"], XLA_COMPILER_IDENTITY.as_str());
        assert!(decoded["compiler_identity"].as_str().unwrap().contains(ryft_xla_sys::XLA_COMMIT));
        assert!(decoded["compiler_identity"].as_str().unwrap().contains(ryft_xla_sys::JAX_COMMIT));
        assert_eq!(decoded["platform_name"], client.platform_name().unwrap().as_ref());
        assert!(decoded["compilation_options"].as_array().is_some_and(|bytes| !bytes.is_empty()));
        assert!(decoded["stable_hlo"].as_str().is_some_and(|module| module.contains("stablehlo.sine")));
    }

    #[test]
    fn test_xla_persistent_executable_round_trip_preserves_invocation_and_analysis() {
        use ryft_core::StagedFunction;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let capture = Array::from_host_buffer(
            &client,
            input_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[10.0, 20.0, 30.0, 40.0]).as_slice(),
        )
        .unwrap();
        fn add_capture<'c>(
            _: Vec<XlaConstant>,
            captures: Vec<ryft_core::compilation::CompilationTracer<XlaDomain<'c>>>,
            input: ryft_core::compilation::CompilationTracer<XlaDomain<'c>>,
        ) -> Result<ryft_core::compilation::CompilationTracer<XlaDomain<'c>>, XlaDomainError> {
            let capture = ValueProjection::<ArrayType>::into_projected(captures.into_iter().next().unwrap())
                .map_err(ProgramError::from)?;
            let input = ValueProjection::<ArrayType>::into_projected(input).map_err(ProgramError::from)?;
            Ok(ValueProjection::<ArrayType>::from_projected(capture + input))
        }
        let staged: StagedFunction<XlaDomain<'_>, ArrayIrType, ArrayIrType> = domain
            .stage(
                ryft_core::compilation::CompilationStagingRequest::<XlaDomain<'_>, _, ArrayIrType, ArrayIrType>::new(
                    add_capture,
                    vec![ArrayIrValue::Array(capture.clone())],
                    ArrayIrType::Array(input_type.clone()),
                    XlaOptions::new(mesh.clone()).with_donate(true),
                ),
            )
            .unwrap();
        let compiled = domain.compile(domain.lower(staged).unwrap()).unwrap();
        let compiled_analysis = domain.analyze(compiled.executable_function()).unwrap();
        assert_eq!(domain.analyze(compiled.executable_function()).unwrap(), compiled_analysis);

        let bytes = domain.serialize_program(compiled.compiled_program()).unwrap().unwrap();
        let restored = domain.deserialize_program(bytes.as_slice()).unwrap().unwrap();
        assert_eq!(restored.input_types, compiled.compiled_program().input_types);
        assert_eq!(restored.output_types, compiled.compiled_program().output_types);
        assert_eq!(restored.signature, compiled.compiled_program().signature);
        assert_eq!(restored.donation_flags, compiled.compiled_program().donation_flags);
        assert_eq!(restored.capture_count, compiled.compiled_program().capture_count);
        assert_eq!(restored.expected_argument_shardings, compiled.compiled_program().expected_argument_shardings);
        assert_eq!(restored.mesh, compiled.compiled_program().mesh);

        let header_size = XLA_PERSISTENT_EXECUTABLE_MAGIC.len() + size_of::<u64>();
        let metadata_size =
            u64::from_le_bytes(bytes[XLA_PERSISTENT_EXECUTABLE_MAGIC.len()..header_size].try_into().unwrap()) as usize;
        let metadata_end = header_size + metadata_size;
        let mut invalid_signature_metadata: XlaPersistentExecutableMetadataV5 =
            serde_json::from_slice(&bytes[header_size..metadata_end]).unwrap();
        invalid_signature_metadata.input_mapping[0] = None;
        let invalid_signature_metadata = serde_json::to_vec(&invalid_signature_metadata).unwrap();
        let mut invalid_signature = XLA_PERSISTENT_EXECUTABLE_MAGIC.to_vec();
        invalid_signature.extend_from_slice(&(invalid_signature_metadata.len() as u64).to_le_bytes());
        invalid_signature.extend_from_slice(invalid_signature_metadata.as_slice());
        invalid_signature.extend_from_slice(&bytes[metadata_end..]);
        assert!(matches!(
            domain.deserialize_program(invalid_signature.as_slice()),
            Err(XlaDomainError::InvalidPersistentExecutable { .. }),
        ));

        let mut incompatible_metadata: XlaPersistentExecutableMetadataV5 =
            serde_json::from_slice(&bytes[header_size..metadata_end]).unwrap();
        incompatible_metadata.platform_version.push_str("-incompatible");
        let incompatible_metadata = serde_json::to_vec(&incompatible_metadata).unwrap();
        let mut incompatible = XLA_PERSISTENT_EXECUTABLE_MAGIC.to_vec();
        incompatible.extend_from_slice(&(incompatible_metadata.len() as u64).to_le_bytes());
        incompatible.extend_from_slice(incompatible_metadata.as_slice());
        incompatible.extend_from_slice(&bytes[metadata_end..]);
        assert!(domain.deserialize_program(incompatible.as_slice()).unwrap().is_none());

        let input = Array::from_host_buffer(
            &client,
            input_type,
            mesh,
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let outputs = domain.execute_xla_program(&restored, vec![capture, input]).unwrap();
        assert_eq!(read_f32s(&client, &outputs[0]), vec![11.0, 22.0, 33.0, 44.0]);

        let analysis = analyze_xla_program(&restored).unwrap();
        assert_eq!(analyze_xla_program(&restored).unwrap(), analysis);
        assert!(analysis.to_json().unwrap().contains("\"properties\""));
        assert!(!analysis.to_string().is_empty());
    }

    #[test]
    fn test_xla_disk_cache_restores_into_a_fresh_compilation_context() {
        use ryft_core::{DiskCache, Sin};
        use tempfile::tempdir;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let directory = tempdir().unwrap();

        let first_domain = XlaDomain::with_configured_disk_cache(
            &client,
            DiskCache::open(directory.path()).unwrap().with_write_thresholds(Duration::ZERO, 0),
        );
        let staged = crate::jit::stage::<_, ArrayType, ArrayType>(
            |x| x.sin().unwrap(),
            input_type.clone(),
            &first_domain,
            XlaOptions::new(mesh.clone()),
        )
        .unwrap()
        .into_inner();
        let first: ryft_core::compilation::CompiledFunction<XlaDomain<'_>, ArrayIrType, ArrayIrType> =
            first_domain.compile(first_domain.lower(staged).unwrap()).unwrap();
        assert_eq!(first_domain.cache.statistics().compilations, 1);
        drop(first);
        drop(first_domain);

        let second_domain = XlaDomain::with_configured_disk_cache(
            &client,
            DiskCache::open(directory.path()).unwrap().with_write_thresholds(Duration::ZERO, 0),
        );
        let staged = crate::jit::stage::<_, ArrayType, ArrayType>(
            |x| x.sin().unwrap(),
            input_type,
            &second_domain,
            XlaOptions::new(mesh),
        )
        .unwrap()
        .into_inner();
        let _restored: ryft_core::compilation::CompiledFunction<XlaDomain<'_>, ArrayIrType, ArrayIrType> =
            second_domain.compile(second_domain.lower(staged).unwrap()).unwrap();
        let statistics = second_domain.cache.statistics();
        assert_eq!(statistics.persistent_hits, 1);
        assert_eq!(statistics.compilations, 0, "restoration must not invoke backend compilation");
    }

    #[test]
    fn test_xla_persistent_executable_rejects_malformed_and_incompatible_metadata() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let domain = XlaDomain::new(&client);

        assert!(matches!(
            domain.deserialize_program(b"truncated"),
            Err(XlaDomainError::InvalidPersistentExecutable { .. }),
        ));
        let mut legacy = b"RYFTXLA1".to_vec();
        legacy.extend_from_slice(&0u64.to_le_bytes());
        assert!(domain.deserialize_program(legacy.as_slice()).unwrap().is_none());

        let metadata = XlaPersistentExecutableMetadataV5 {
            schema_version: XLA_PERSISTENT_EXECUTABLE_SCHEMA_VERSION + 1,
            feature_flags: 0,
            compilation_options: CompilationOptions::default().encode_to_vec(),
            signature: PersistentArraySignatureV3::encode(&[], &[]),
            input_mapping: Vec::new(),
            output_mapping: Vec::new(),
            input_dimensions: Vec::new(),
            output_dimensions: Vec::new(),
            requires_assertion_handler: false,
            donation_flags: Vec::new(),
            capture_count: 0,
            expected_argument_shardings: Vec::new(),
            mesh: PersistentDeviceMeshV1 {
                logical_mesh: PersistentLogicalMeshV1 { axes: Vec::new() },
                devices: Vec::new(),
            },
            device_kinds: Vec::new(),
            replica_count: 0,
            partition_count: 0,
            device_assignment: Vec::new(),
            platform_name: client.platform_name().unwrap().into_owned(),
            platform_version: client.platform_version().unwrap().into_owned(),
            compiler_identity: XLA_COMPILER_IDENTITY.to_string(),
            xla_flags: std::env::var("XLA_FLAGS").unwrap_or_default(),
            compilation_duration_nanoseconds: None,
        };
        let metadata = serde_json::to_vec(&metadata).unwrap();
        let mut bytes = XLA_PERSISTENT_EXECUTABLE_MAGIC.to_vec();
        bytes.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        bytes.extend_from_slice(metadata.as_slice());
        assert!(domain.deserialize_program(bytes.as_slice()).unwrap().is_none());
    }

    #[test]
    fn test_eager_bind_executes_binary_operation() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = array_domain(&client);

        let left = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0, 4.0]);
        let right = f32_vector(&client, &mesh, &[10.0, 20.0, 30.0, 40.0]);
        let outputs = domain.bind(AddOperation::new(), Vec::new(), &[left, right]).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(read_f32s(&client, &outputs[0]), vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_eager_bind_rejects_unresolved_references_before_special_cases_or_tracing() {
        // The guard is operation-agnostic: it keys off unresolved reference state rather than the specific reference
        // operation, so one representative operation pins the diagnostic.
        assert_eq!(
            XlaDomain::token().bind(XlaOperation::NewReference(NewReferenceOperation), Vec::new(), &[]),
            Err(ProgramError::UnsupportedOperation {
                message: "`new_reference` must be discharged before XLA eager execution".to_string(),
            }),
        );

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);
        let reference = ArrayIrValue::Reference(Reference::new(f32_vector(&client, &mesh, &[1.0])));
        assert_eq!(
            domain.bind(AddOperation::new(), Vec::new(), &[reference]),
            Err(ProgramError::UnsupportedOperation {
                message: "references must be discharged before XLA eager execution".to_string(),
            }),
        );

        // Region-carried state must be rejected too: control-flow operations report pure intrinsic effects, so the
        // guard has to union the driver's region-program effects to see a reference operation inside a loop body.
        let condition = {
            let mut builder = XlaProgramBuilder::new();
            let predicate = builder
                .add_instruction(ZeroOperation::new(ArrayType::scalar(DataType::Boolean)), Vec::new(), Vec::new())
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![predicate], Vec::new(), vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = XlaProgramBuilder::new();
            let state = builder
                .add_instruction(ZeroOperation::new(ArrayType::scalar(DataType::F32)), Vec::new(), Vec::new())
                .unwrap()[0];
            let reference = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![state]).unwrap()[0];
            builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap();
            builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(Vec::new(), Vec::new(), Vec::new()).unwrap()
        };
        assert_eq!(
            XlaDomain::token().bind(XlaOperation::While(WhileOperation::new()), vec![condition, body], &[]),
            Err(ProgramError::UnsupportedOperation {
                message: "`while` must be discharged before XLA eager execution".to_string(),
            }),
        );

        // Reference state confined to a dormant custom-derivative rule region is rejected too: ordinary XLA rejects
        // unresolved reference artifacts program-wide, so eager binding matches that policy instead of deferring the
        // failure to compilation with a less targeted diagnostic.
        let primal = {
            let mut builder = XlaProgramBuilder::new();
            let state = builder.add_input(ArrayType::scalar(DataType::F32).into());
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![state], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let rule = {
            let mut builder = XlaProgramBuilder::new();
            let state = builder.add_input(ArrayType::scalar(DataType::F32).into());
            let state_tangent = builder.add_input(ArrayType::scalar(DataType::F32).into());
            let reference = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![state]).unwrap()[0];
            let output = builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![output, state_tangent],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        assert_eq!(
            XlaDomain::token().bind(
                XlaOperation::CustomJvp(CustomJvpOperation::new()),
                vec![primal.clone(), rule],
                &[],
            ),
            Err(ProgramError::UnsupportedOperation {
                message: "`custom_jvp` must be discharged before XLA eager execution".to_string(),
            }),
        );

        // A pure reference-typed capture in the dormant rule has no ordered-state effect, so only the complete-arena
        // type scan can keep eager and compiled XLA rejection policy aligned.
        let pure_reference_rule = {
            let mut builder = XlaProgramBuilder::new();
            let state = builder.add_input(ArrayType::scalar(DataType::F32).into());
            let state_tangent = builder.add_input(ArrayType::scalar(DataType::F32).into());
            builder.add_constant(XlaConstant::Captured(CaptureReference::new(
                0,
                ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32))),
            )));
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![state, state_tangent],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        assert_eq!(
            XlaDomain::token().bind(
                XlaOperation::CustomJvp(CustomJvpOperation::new()),
                vec![primal, pure_reference_rule],
                &[],
            ),
            Err(ProgramError::UnsupportedOperation {
                message: "`custom_jvp` must be discharged before XLA eager execution".to_string(),
            }),
        );
    }

    #[test]
    fn test_stage_rejects_reference_boundaries_before_sharding_projection() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);
        let scalar_sharding = Sharding::new(mesh.logical_mesh().clone(), Vec::new()).unwrap();
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));

        // A reference input with `in_shardings` receives the targeted discharge diagnostic before the sharding
        // override projects boundary leaves to arrays.
        fn forward<'c>(
            _: Vec<XlaConstant>,
            _: Vec<CompilationTracer<XlaDomain<'c>>>,
            input: CompilationTracer<XlaDomain<'c>>,
        ) -> Result<CompilationTracer<XlaDomain<'c>>, XlaDomainError> {
            Ok(input)
        }
        assert!(matches!(
            domain
                .stage(
                    ryft_core::compilation::CompilationStagingRequest::<XlaDomain<'_>, _, ArrayIrType, ArrayIrType>::new(
                        forward,
                        Vec::new(),
                        reference_type.clone(),
                        XlaOptions::new(mesh.clone()).with_in_shardings(vec![scalar_sharding.clone()]),
                    ),
                )
                .map(|_| ()),
            Err(XlaDomainError::Tracing(ProgramError::UnsupportedOperation { message }))
                if message == format!(
                    "references must be discharged before XLA compilation, but staged input has type \
                    `{reference_type}`",
                ),
        ));

        // A reference output with `out_shardings` is likewise rejected before the output override runs.
        fn allocate<'c>(
            _: Vec<XlaConstant>,
            _: Vec<CompilationTracer<XlaDomain<'c>>>,
            input: CompilationTracer<XlaDomain<'c>>,
        ) -> Result<CompilationTracer<XlaDomain<'c>>, XlaDomainError> {
            Ok(input.new_reference()?)
        }
        assert!(matches!(
            domain
                .stage(
                    ryft_core::compilation::CompilationStagingRequest::<XlaDomain<'_>, _, ArrayIrType, ArrayIrType>::new(
                        allocate,
                        Vec::new(),
                        ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
                        XlaOptions::new(mesh.clone()).with_out_shardings(vec![scalar_sharding]),
                    ),
                )
                .map(|_| ()),
            Err(XlaDomainError::Tracing(ProgramError::UnsupportedOperation { message }))
                if message == format!(
                    "references must be discharged before XLA compilation, but staged output has type \
                    `{reference_type}`",
                ),
        ));
    }

    #[test]
    fn test_xla_lowering_discharges_and_executes_local_reference_state() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);

        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(scalar_type.clone().into());
        let reference = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![input]).unwrap()[0];
        let snapshot = builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
        builder.add_instruction(ReferenceAddUpdateOperation, Vec::new(), vec![reference, input]).unwrap();
        let final_value = builder.add_instruction(FreezeReferenceOperation, Vec::new(), vec![reference]).unwrap()[0];
        let program: FlatXlaProgram = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![snapshot, final_value],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();

        let lowered = domain.lower_xla_program(&program, 0, &XlaOptions::new(mesh.clone())).unwrap();
        assert_eq!(lowered.output_types(), &[scalar_type.clone(), scalar_type]);
        let expected_signature = concat!(
            "  func.func @main(%arg0: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}) -> ",
            "(tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, ",
            "tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}) {",
        );
        let expected_stable_hlo = indoc! {r#"
            module {
              sdy.mesh @mesh = <["x"=1]>
            @SIGNATURE@
                %0 = stablehlo.add %arg0, %arg0 : tensor<f32>
                return %arg0, %0 : tensor<f32>, tensor<f32>
              }
            }
        "#}
        .replace("@SIGNATURE@", expected_signature);
        assert_eq!(lowered.stable_hlo(), expected_stable_hlo,);
        let compiled = domain.compile_xla_program(&lowered).unwrap();
        let outputs = domain.execute_xla_program(&compiled, vec![f32_scalar(&client, &mesh, 3.0)]).unwrap();
        assert_eq!(read_f32s(&client, &outputs[0]), vec![3.0]);
        assert_eq!(read_f32s(&client, &outputs[1]), vec![6.0]);

        let eager_input = ArrayIrValue::Array(CpuArray::scalar(3.0f32));
        let eager_reference = eager_input.new_reference().unwrap();
        let eager_snapshot = eager_reference.read().unwrap();
        eager_reference.add_update(&eager_input).unwrap();
        let eager_final = eager_reference.freeze().unwrap();
        assert_eq!(eager_snapshot, ArrayIrValue::Array(CpuArray::scalar(3.0f32)));
        assert_eq!(eager_final, ArrayIrValue::Array(CpuArray::scalar(6.0f32)));
    }

    #[test]
    fn test_xla_lowering_preserves_lifted_capture_scope_inside_condition_regions() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let predicate_type = replicated_scalar_type(&mesh, DataType::Boolean);

        let branch = || {
            let mut builder = XlaProgramBuilder::new();
            let capture =
                builder.add_constant(XlaConstant::Captured(CaptureReference::new(0, scalar_type.clone().into())));
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![capture], Vec::new(), vec![Placeholder])
                .unwrap()
        };
        let true_branch = branch();
        let false_branch = branch();
        let mut builder = XlaProgramBuilder::new();
        builder.add_input(scalar_type.clone().into());
        let predicate = builder.add_input(predicate_type.into());
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_region(false_branch.entry_region_ref());
        let output = builder
            .add_instruction(ConditionOperation::new(), vec![true_region, false_region], vec![predicate])
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let lowered = domain.lower_xla_program(&program, 1, &XlaOptions::new(mesh.clone())).unwrap();
        let compiled = domain.compile_xla_program(&lowered).unwrap();
        let outputs = domain
            .execute_xla_program(&compiled, vec![f32_scalar(&client, &mesh, 7.0), boolean_scalar(&client, &mesh, true)])
            .unwrap();
        assert_eq!(read_f32s(&client, &outputs[0]), vec![7.0]);
    }

    #[test]
    fn test_xla_lowering_validates_capture_count_before_reference_discharge() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(scalar_type.into());
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();

        assert!(matches!(
            domain.lower_xla_program(&program, 2, &XlaOptions::new(mesh)),
            Err(XlaDomainError::InvalidCompilationOptions { reason })
                if reason == "capture_count is 2, but the program has only 1 flat input(s)",
        ));
    }

    #[test]
    fn test_xla_lowering_discharges_local_reference_state_through_condition() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());

        let true_branch = {
            let mut builder = XlaProgramBuilder::new();
            let reference = builder.add_input(reference_type.clone().into());
            let replacement = builder.add_input(scalar_type.clone().into());
            let snapshot =
                builder.add_instruction(ReferenceSwapOperation, Vec::new(), vec![reference, replacement]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let false_branch = {
            let mut builder = XlaProgramBuilder::new();
            let reference = builder.add_input(reference_type.into());
            builder.add_input(scalar_type.clone().into());
            let snapshot = builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };

        let mut builder = XlaProgramBuilder::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(replicated_scalar_type(&mesh, DataType::Boolean).into());
        let initial = builder.add_input(scalar_type.clone().into());
        let replacement = builder.add_input(scalar_type.clone().into());
        let reference = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![initial]).unwrap()[0];
        let snapshot = builder
            .add_instruction(
                ConditionOperation::<XlaConstant>::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference, replacement],
            )
            .unwrap()[0];
        let final_value = builder.add_instruction(FreezeReferenceOperation, Vec::new(), vec![reference]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![snapshot, final_value],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();

        let lowered = domain.lower_xla_program(&program, 0, &XlaOptions::new(mesh.clone())).unwrap();
        assert_eq!(lowered.output_types(), &[scalar_type.clone(), scalar_type]);
        let expected_signature = concat!(
            "  func.func @main(%arg0: tensor<i1> {sdy.sharding = #sdy.sharding<@mesh, []>}, ",
            "%arg1: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, ",
            "%arg2: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}) -> ",
            "(tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}, ",
            "tensor<f32> {sdy.sharding = #sdy.sharding<@mesh, []>}) {",
        );
        let expected_stable_hlo = indoc! {r#"
            module {
              sdy.mesh @mesh = <["x"=1]>
            @SIGNATURE@
                %0:2 = "stablehlo.if"(%arg0) ({
                  stablehlo.return %arg1, %arg2 : tensor<f32>, tensor<f32>
                }, {
                  stablehlo.return %arg1, %arg1 : tensor<f32>, tensor<f32>
                }) : (tensor<i1>) -> (tensor<f32>, tensor<f32>)
                return %0#0, %0#1 : tensor<f32>, tensor<f32>
              }
            }
        "#}
        .replace("@SIGNATURE@", expected_signature);
        assert_eq!(lowered.stable_hlo(), expected_stable_hlo,);
        let compiled = domain.compile_xla_program(&lowered).unwrap();

        let true_outputs = domain
            .execute_xla_program(
                &compiled,
                vec![
                    boolean_scalar(&client, &mesh, true),
                    f32_scalar(&client, &mesh, 10.0),
                    f32_scalar(&client, &mesh, 7.0),
                ],
            )
            .unwrap();
        assert_eq!(read_f32s(&client, &true_outputs[0]), vec![10.0]);
        assert_eq!(read_f32s(&client, &true_outputs[1]), vec![7.0]);

        let false_outputs = domain
            .execute_xla_program(
                &compiled,
                vec![
                    boolean_scalar(&client, &mesh, false),
                    f32_scalar(&client, &mesh, 10.0),
                    f32_scalar(&client, &mesh, 7.0),
                ],
            )
            .unwrap();
        assert_eq!(read_f32s(&client, &false_outputs[0]), vec![10.0]);
        assert_eq!(read_f32s(&client, &false_outputs[1]), vec![10.0]);
    }

    #[test]
    fn test_xla_lowering_discharges_local_reference_state_through_while() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());

        let condition = {
            let mut builder = XlaProgramBuilder::new();
            let reference = builder.add_input(reference_type.clone().into());
            builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap();
            let predicate = CpuArray::from_elements(replicated_scalar_type(&mesh, DataType::Boolean), &[true]).unwrap();
            let predicate =
                builder.add_instruction(ConstantOperation::new(predicate), Vec::new(), Vec::new()).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![predicate], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = XlaProgramBuilder::new();
            let reference = builder.add_input(reference_type.into());
            let update =
                builder.add_instruction(OneOperation::new(scalar_type.clone()), Vec::new(), Vec::new()).unwrap()[0];
            builder.add_instruction(ReferenceAddUpdateOperation, Vec::new(), vec![reference, update]).unwrap();
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![reference], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };

        let mut builder = XlaProgramBuilder::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(scalar_type.clone().into());
        let reference = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![initial]).unwrap()[0];
        let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(3).unwrap();
        let reference = builder.add_instruction(operation, vec![condition, body], vec![reference]).unwrap()[0];
        let output = builder.add_instruction(FreezeReferenceOperation, Vec::new(), vec![reference]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let lowered = domain.lower_xla_program(&program, 0, &XlaOptions::new(mesh.clone())).unwrap();
        assert_eq!(lowered.stable_hlo().matches("stablehlo.while").count(), 1, "{}", lowered.stable_hlo());
        assert_reference_free_stable_hlo(&lowered);
        let compiled = domain.compile_xla_program(&lowered).unwrap();
        let outputs = domain.execute_xla_program(&compiled, vec![f32_scalar(&client, &mesh, 2.0)]).unwrap();

        let immutable_oracle = (0..3).fold(2.0f32, |state, _| state + 1.0);
        assert_eq!(read_f32s(&client, &outputs[0]), vec![immutable_oracle]);
    }

    #[test]
    fn test_xla_lowering_discharges_local_reference_state_through_static_scan() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());

        let body = {
            let mut builder = XlaProgramBuilder::new();
            let reference = builder.add_input(reference_type.into());
            let update =
                builder.add_instruction(OneOperation::new(scalar_type.clone()), Vec::new(), Vec::new()).unwrap()[0];
            builder.add_instruction(ReferenceAddUpdateOperation, Vec::new(), vec![reference, update]).unwrap();
            let value = builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![reference, value],
                    vec![Placeholder],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };

        let mut builder = XlaProgramBuilder::new();
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(scalar_type.into());
        let reference = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![initial]).unwrap()[0];
        let scan_outputs = builder
            .add_instruction(ScanOperation::<XlaConstant>::new(1, 3), vec![body], vec![reference])
            .unwrap()
            .to_vec();
        let final_value =
            builder.add_instruction(FreezeReferenceOperation, Vec::new(), vec![scan_outputs[0]]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![final_value, scan_outputs[1]],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();

        let lowered = domain.lower_xla_program(&program, 0, &XlaOptions::new(mesh.clone())).unwrap();
        assert_eq!(lowered.stable_hlo().matches("stablehlo.while").count(), 1, "{}", lowered.stable_hlo());
        assert_reference_free_stable_hlo(&lowered);
        let compiled = domain.compile_xla_program(&lowered).unwrap();
        let outputs = domain.execute_xla_program(&compiled, vec![f32_scalar(&client, &mesh, 2.0)]).unwrap();

        let mut immutable_state = 2.0f32;
        let mut immutable_steps = Vec::new();
        for _ in 0..3 {
            immutable_state += 1.0;
            immutable_steps.push(immutable_state);
        }
        assert_eq!(read_f32s(&client, &outputs[0]), vec![immutable_state]);
        assert_eq!(read_f32s(&client, &outputs[1]), immutable_steps);
    }

    #[test]
    fn test_xla_lowering_discharges_local_reference_state_through_dynamic_scan() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let length_variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(9)).unwrap());
        let length = DimensionValue::new(DimensionType::new(length_variable.clone()), 3).unwrap();

        let body = {
            let mut builder = XlaProgramBuilder::new();
            let reference = builder.add_input(reference_type.into());
            let update =
                builder.add_instruction(OneOperation::new(scalar_type.clone()), Vec::new(), Vec::new()).unwrap()[0];
            builder.add_instruction(ReferenceAddUpdateOperation, Vec::new(), vec![reference, update]).unwrap();
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![reference], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };

        let mut builder = XlaProgramBuilder::new();
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(scalar_type.into());
        let reference = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![initial]).unwrap()[0];
        let runtime_length = builder.add_constant(XlaConstant::Dimension(length.clone()));
        let reference = builder
            .add_instruction(
                ScanOperation::<XlaConstant>::new(1, Dimension::Dynamic(length_variable.clone())),
                vec![body],
                vec![reference, runtime_length],
            )
            .unwrap()[0];
        let output = builder.add_instruction(FreezeReferenceOperation, Vec::new(), vec![reference]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let scan_instruction = &program.entry_region_ref().instructions()[1];
        let XlaOperation::Scan(scan) = scan_instruction.operation() else {
            panic!("expected dynamic scan operation");
        };
        assert_eq!(scan.carry_count(), 1);
        assert_eq!(scan.length(), &Dimension::Dynamic(length_variable));
        let runtime_length = *scan_instruction.inputs().last().unwrap();
        assert_eq!(
            program.entry_region_ref().atoms()[runtime_length.index()].as_constant(),
            Some(&XlaConstant::Dimension(length)),
        );
        let lowered = domain.lower_xla_program(&program, 0, &XlaOptions::new(mesh.clone())).unwrap();
        assert_eq!(lowered.stable_hlo().matches("stablehlo.while").count(), 1, "{}", lowered.stable_hlo());
        assert_reference_free_stable_hlo(&lowered);
        let compiled = domain.compile_xla_program(&lowered).unwrap();
        let outputs = domain.execute_xla_program(&compiled, vec![f32_scalar(&client, &mesh, 2.0)]).unwrap();

        let immutable_oracle = (0..3).fold(2.0f32, |state, _| state + 1.0);
        assert_eq!(read_f32s(&client, &outputs[0]), vec![immutable_oracle]);
    }

    #[test]
    fn test_xla_lowering_discharges_local_reference_state_through_nested_jit_call() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());

        let callee = {
            let mut builder = XlaProgramBuilder::new();
            let reference = builder.add_input(reference_type.into());
            let update = builder.add_input(scalar_type.clone().into());
            builder.add_instruction(ReferenceAddUpdateOperation, Vec::new(), vec![reference, update]).unwrap();
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![reference], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };

        let mut builder = XlaProgramBuilder::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let initial = builder.add_input(scalar_type.clone().into());
        let update = builder.add_input(scalar_type.into());
        let reference = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![initial]).unwrap()[0];
        let reference = builder
            .add_instruction(XlaOperation::JitCall(JitCallOperation::new(0)), vec![callee], vec![reference, update])
            .unwrap()[0];
        let output = builder.add_instruction(FreezeReferenceOperation, Vec::new(), vec![reference]).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        assert!(matches!(
            program.entry_region_ref().instructions()[1].operation(),
            XlaOperation::JitCall(operation) if operation.capture_count() == 0,
        ));
        let lowered = domain.lower_xla_program(&program, 0, &XlaOptions::new(mesh.clone())).unwrap();
        assert_eq!(lowered.stable_hlo().matches("stablehlo.add").count(), 1, "{}", lowered.stable_hlo());
        assert_reference_free_stable_hlo(&lowered);
        let compiled = domain.compile_xla_program(&lowered).unwrap();
        let outputs = domain
            .execute_xla_program(&compiled, vec![f32_scalar(&client, &mesh, 2.0), f32_scalar(&client, &mesh, 4.0)])
            .unwrap();

        let immutable_oracle = 2.0f32 + 4.0;
        assert_eq!(read_f32s(&client, &outputs[0]), vec![immutable_oracle]);
    }

    #[test]
    fn test_xla_lowering_rejects_external_reference_state_after_discharge() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);

        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let output = builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![input]).unwrap()[0];
        let program: FlatXlaProgram = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            domain.lower_xla_program(&program, 0, &XlaOptions::new(mesh)),
            Err(XlaDomainError::UnsupportedReferenceAbi { reason })
                if reason == "external reference state requires a stateful XLA invocation boundary",
        ));
    }

    #[test]
    fn test_eager_bind_executes_promoted_and_broadcast_elementwise_operations() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = array_domain(&client);

        let scalar = f32_scalar(&client, &mesh, 2.0);
        let vector = f64_vector(&client, &mesh, &[1.0, 2.0, 4.0, 8.0]);

        let divide = domain.bind(DivOperation::new(), Vec::new(), &[scalar.clone(), vector.clone()]).unwrap();
        assert_eq!(read_f64s(&client, &divide[0]), vec![2.0, 1.0, 0.5, 0.25]);

        let atan2 = domain.bind(Atan2Operation::new(), Vec::new(), &[vector.clone(), scalar.clone()]).unwrap();
        for (actual, expected) in read_f64s(&client, &atan2[0]).into_iter().zip([
            1.0f64.atan2(2.0),
            2.0f64.atan2(2.0),
            4.0f64.atan2(2.0),
            8.0f64.atan2(2.0),
        ]) {
            assert!((actual - expected).abs() < 1e-12, "got {actual}, expected {expected}");
        }

        let compare = domain
            .bind(CompareOperation::new(ComparisonDirection::LessThan), Vec::new(), &[scalar.clone(), vector.clone()])
            .unwrap();
        assert_eq!(read_booleans(&client, &compare[0]), vec![false, false, true, true]);

        let select = domain
            .bind(SelectOperation::new(), Vec::new(), &[boolean_scalar(&client, &mesh, true), scalar, vector])
            .unwrap();
        assert_eq!(read_f64s(&client, &select[0]), vec![2.0, 2.0, 2.0, 2.0]);

        let boolean_vector = boolean_vector(&client, &mesh, &[true, false, true, false]);
        let and = domain
            .bind(AndOperation::new(), Vec::new(), &[boolean_scalar(&client, &mesh, true), boolean_vector])
            .unwrap();
        assert_eq!(read_booleans(&client, &and[0]), vec![true, false, true, false]);
    }

    #[test]
    fn test_eager_bind_executes_unary_operation() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = array_domain(&client);

        let input = f32_vector(&client, &mesh, &[1.0, -2.0, 3.5, 0.0]);
        let outputs = domain.bind(NegOperation::new(), Vec::new(), &[input]).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(read_f32s(&client, &outputs[0]), vec![-1.0, 2.0, -3.5, 0.0]);
    }

    #[test]
    fn test_eager_fill_materializes_scalar_literal_then_broadcasts_over_a_default_mesh() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let domain = array_domain(&client);

        for memory in [Memory::Device, Memory::Host { pinned: true }, Memory::Host { pinned: false }] {
            let r#type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])).with_memory(memory);
            let output = domain.fill(&r#type, 2.5f64).unwrap();

            assert_eq!(output.r#type().memory(), memory);
            assert_eq!(output.shape(), StaticShape::new(vec![3]));
            assert_eq!(read_f32s(&client, &output), vec![2.5, 2.5, 2.5]);
        }

        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]));
        let value = num_complex::Complex::new(1.5f64, -2.0f64);
        let output = domain.fill(&r#type, value).unwrap();
        assert_eq!(read_f32s(&client, &output), vec![1.5, 1.5]);

        // A complex fill value lowers as two real part splats composed through `stablehlo.complex`, and a `c64`
        // buffer's bytes are the interleaved `f32` real and imaginary parts of its elements.
        let r#type = ArrayType::new(DataType::C64, Shape::new(vec![Dimension::Static(2)]));
        let value = num_complex::Complex::new(1.5f32, -2.0f32);
        let output = domain.fill(&r#type, value).unwrap();

        assert_eq!(output.shape(), StaticShape::new(vec![2]));
        assert_eq!(read_f32s(&client, &output), vec![1.5, -2.0, 1.5, -2.0]);

        // Integer literals remain exact through conversion and scalar constant lowering, including values that
        // cannot be represented exactly by the floating-point intermediary used for floating-point constants.
        let r#type = ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(2)]));
        let value = u64::MAX - 1;
        let output = domain.fill(&r#type, value).unwrap();
        assert_eq!(read_u64s(&client, &output), vec![value, value]);
    }

    #[test]
    fn test_eager_bind_reuses_cached_executable_for_repeated_operations() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = array_domain(&client);
        assert_eq!(domain.parent().cache_size(), 0);

        let left = f32_vector(&client, &mesh, &[1.0, 2.0]);
        let right = f32_vector(&client, &mesh, &[3.0, 4.0]);
        let first = domain.bind(AddOperation::new(), Vec::new(), &[left.clone(), right.clone()]).unwrap();
        assert_eq!(domain.parent().cache_size(), 1);

        let second = domain.bind(AddOperation::new(), Vec::new(), &[left, right]).unwrap();
        assert_eq!(domain.parent().cache_size(), 1, "a repeated eager operation must be a compile-cache hit");
        assert_eq!(read_f32s(&client, &first[0]), read_f32s(&client, &second[0]));

        // A different input signature compiles (and caches) a distinct executable.
        let wider_left = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0]);
        let wider_right = f32_vector(&client, &mesh, &[4.0, 5.0, 6.0]);
        domain.bind(AddOperation::new(), Vec::new(), &[wider_left, wider_right]).unwrap();
        assert_eq!(domain.parent().cache_size(), 2);
    }

    #[test]
    fn test_eager_bind_rejects_inputs_placed_on_a_foreign_device() {
        let plugin = load_cpu_plugin().unwrap();
        let domain_client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let foreign_client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let foreign_mesh = domain_mesh(&foreign_client, "x", 2);
        let domain = array_domain(&domain_client);

        // An input that carries an attached client is rejected by client identity.
        let input = f32_vector(&foreign_client, &foreign_mesh, &[1.0, 2.0]);
        assert!(matches!(
            domain.bind(NegOperation::new(), Vec::new(), &[input.clone()]),
            Err(ProgramError::InvalidArgument { message })
                if message == "received incompatible devices for eager xla execution: input #0 is owned by a \
                    different PJRT client than this domain's client",
        ));

        // An input with no attached client falls back to the device-set membership check.
        let mut clientless_input = input;
        clientless_input.detach_client_for_tests();
        assert!(matches!(
            domain.bind(NegOperation::new(), Vec::new(), &[clientless_input]),
            Err(ProgramError::InvalidArgument { message })
                if message == "received incompatible devices for eager xla execution: input #0 is placed on device \
                    1, which does not belong to this domain's PJRT client",
        ));
    }

    #[test]
    fn test_eager_bind_executes_condition_with_concrete_predicate() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);
        let vector_type = replicated_vector_type(&mesh, 4);

        let doubled = {
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(vector_type.clone().into());
            let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let squared = {
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(vector_type.clone().into());
            let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, input]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let operation = XlaOperation::Condition(ConditionOperation::new());

        let input = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0, 4.0]);
        let true_outputs = domain
            .bind(
                operation.clone(),
                [doubled.clone(), squared.clone()],
                &[ArrayIrValue::Array(boolean_scalar(&client, &mesh, true)), ArrayIrValue::Array(input.clone())],
            )
            .unwrap();
        assert_eq!(read_f32s(&client, program_array(&true_outputs[0])), vec![2.0, 4.0, 6.0, 8.0]);

        let false_outputs = domain
            .bind(
                operation,
                [doubled, squared],
                &[ArrayIrValue::Array(boolean_scalar(&client, &mesh, false)), ArrayIrValue::Array(input)],
            )
            .unwrap();
        assert_eq!(read_f32s(&client, program_array(&false_outputs[0])), vec![1.0, 4.0, 9.0, 16.0]);
        assert_eq!(domain.cache_size(), 1, "both predicate values must share one compiled executable");
    }

    #[test]
    fn test_eager_bind_condition_branches_consume_forwarded_dimension_authority() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);
        let extent = DimensionValue::constant(3).unwrap();
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);

        let branch = |negate: bool| {
            let mut builder = XlaProgramBuilder::new();
            let extent = builder.add_input(extent.r#type().into_owned().into());
            let scalar = builder.add_input(scalar_type.clone().into());
            let scalar = if negate {
                builder.add_instruction(NegOperation::new(), Vec::new(), vec![scalar]).unwrap()[0]
            } else {
                scalar
            };
            let output = builder
                .add_instruction(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), vec![scalar, extent])
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![output],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let operation = XlaOperation::Condition(ConditionOperation::new());
        let scalar = f32_scalar(&client, &mesh, 2.0);

        let true_outputs = domain
            .bind(
                operation.clone(),
                [branch(false), branch(true)],
                &[
                    ArrayIrValue::Array(boolean_scalar(&client, &mesh, true)),
                    ArrayIrValue::Dimension(extent.clone()),
                    ArrayIrValue::Array(scalar.clone()),
                ],
            )
            .unwrap();
        assert_eq!(program_array(&true_outputs[0]).shape(), StaticShape::new(vec![3]));
        assert_eq!(read_f32s(&client, program_array(&true_outputs[0])), vec![2.0; 3]);

        let false_outputs = domain
            .bind(
                operation,
                [branch(false), branch(true)],
                &[
                    ArrayIrValue::Array(boolean_scalar(&client, &mesh, false)),
                    ArrayIrValue::Dimension(extent),
                    ArrayIrValue::Array(scalar),
                ],
            )
            .unwrap();
        assert_eq!(read_f32s(&client, program_array(&false_outputs[0])), vec![-2.0; 3]);
        assert_eq!(domain.cache_size(), 1, "both branch selections must share one compiled executable");
    }

    #[test]
    fn test_eager_bind_executes_bounded_while() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);

        // Loop `state = state + 1` while `state < 3`, starting from `0`.
        let condition = {
            let mut builder = XlaProgramBuilder::new();
            let state = builder.add_input(scalar_type.clone().into());
            let literal = CpuArray::from_elements(scalar_type.clone(), &[3.0f32]).unwrap();
            let limit = builder.add_instruction(ConstantOperation::new(literal), Vec::new(), vec![]).unwrap()[0];
            let predicate = builder
                .add_instruction(
                    XlaOperation::Array(ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::LessThan))),
                    Vec::new(),
                    vec![state, limit],
                )
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![predicate],
                    vec![Placeholder; 1],
                    vec![Placeholder; 1],
                )
                .unwrap()
        };
        let body = {
            let mut builder = XlaProgramBuilder::new();
            let state = builder.add_input(scalar_type.clone().into());
            let one = builder.add_instruction(OneOperation::new(scalar_type.clone()), Vec::new(), vec![]).unwrap()[0];
            let next = builder.add_instruction(AddOperation::new(), Vec::new(), vec![state, one]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![next], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let operation = XlaOperation::While(WhileOperation::new());

        let outputs = domain
            .bind(operation, vec![condition, body], &[ArrayIrValue::Array(f32_scalar(&client, &mesh, 0.0))])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(read_f32s(&client, program_array(&outputs[0])), vec![3.0]);
    }

    #[test]
    fn test_eager_bind_executes_elementwise_operation_on_sharded_inputs() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = domain_mesh(&client, "x", 2);
        let domain = array_domain(&client);

        // A vector sharded over a 2-device mesh executes eagerly through per-operation SPMD compilation: each device
        // adds its own 2-element shard.
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(sharding.clone())
            .unwrap();
        let input = Array::from_host_buffer(
            &client,
            input_type,
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let outputs = domain.bind(AddOperation::new(), Vec::new(), &[input.clone(), input]).unwrap();

        assert_eq!(outputs.len(), 1);
        let output = &outputs[0];
        // The input sharding propagates to the output type, so the result stays sharded over the same mesh axis.
        assert_eq!(output.sharding(), &sharding);
        assert_eq!(output.shape(), StaticShape::new(vec![4]));
        assert_eq!(output.shards().len(), 2);
        let shard_values = output
            .addressable_shards()
            .map(|shard| {
                let bytes = shard.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
                values_from_bytes::<f32>(bytes.as_slice())
            })
            .collect::<Vec<_>>();
        assert_eq!(shard_values, vec![vec![2.0, 4.0], vec![6.0, 8.0]]);
    }

    #[test]
    fn test_eager_bind_executes_scan_with_per_step_outputs() {
        use ryft_core::ScanOperation;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);

        // Carry-only scan body `carry -> (carry + 1, carry + 1)`: the first output is the next carry and the second
        // is the per-step stacked output, so scanning 4 steps from `0` yields the cumulative sums `[1, 2, 3, 4]`.
        let body = {
            let mut builder = XlaProgramBuilder::new();
            let carry = builder.add_input(scalar_type.clone().into());
            let one = builder.add_instruction(OneOperation::new(scalar_type.clone()), Vec::new(), vec![]).unwrap()[0];
            let next = builder.add_instruction(AddOperation::new(), Vec::new(), vec![carry, one]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![next, next],
                    vec![Placeholder; 1],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let scan = ScanOperation::<XlaConstant>::new(1, 4);

        let outputs = domain
            .bind(XlaOperation::Scan(scan), [body], &[ArrayIrValue::Array(f32_scalar(&client, &mesh, 0.0))])
            .unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(read_f32s(&client, program_array(&outputs[0])), vec![4.0]);
        assert_eq!(program_array(&outputs[1]).shape(), StaticShape::new(vec![4]));
        assert_eq!(read_f32s(&client, program_array(&outputs[1])), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_eager_bind_executes_scan_with_dynamic_scalar_ssa_length() {
        use ryft_core::ScanOperation;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);
        let length = DimensionValue::new(
            DimensionType::new(DimensionVariable::new("length", DimensionBounds::new(0, Some(9)).unwrap())),
            3,
        )
        .unwrap();

        let body = {
            let mut builder = XlaProgramBuilder::new();
            let carry = builder.add_input(scalar_type.clone().into());
            let one = builder.add_instruction(OneOperation::new(scalar_type.clone()), Vec::new(), vec![]).unwrap()[0];
            let next = builder.add_instruction(AddOperation::new(), Vec::new(), vec![carry, one]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![next], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let scan = ScanOperation::<XlaConstant>::new(1, length.r#type().to_dimension());
        let outputs = domain
            .bind(
                XlaOperation::Scan(scan),
                [body],
                &[ArrayIrValue::Array(f32_scalar(&client, &mesh, 0.0)), ArrayIrValue::Dimension(length)],
            )
            .unwrap();

        assert_eq!(read_f32s(&client, program_array(&outputs[0])), vec![3.0]);
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_eager_bind_executes_scan_with_stacked_inputs() {
        use ryft_core::ScanOperation;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);
        let scalar_type = ArrayType::scalar(DataType::F32);

        // Cumulative-sum scan body `(carry, x) -> (carry + x, carry + x)` over the stacked input `[1, 2, 3, 4]`
        // starting from carry `0`: the final carry is the total `10` and the stacked per-step outputs are the
        // running sums `[1, 3, 6, 10]`. The body's metadata-free declared types are refined by the concrete input
        // types, which carry normalized shardings, so the scan binds eagerly despite the metadata mismatch.
        let body = {
            let mut builder = XlaProgramBuilder::new();
            let carry = builder.add_input(scalar_type.clone().into());
            let x = builder.add_input(scalar_type.clone().into());
            let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![carry, x]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![sum, sum], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let scan = ScanOperation::<XlaConstant>::new(1, 4);

        let carry = f32_scalar(&client, &mesh, 0.0);
        let xs = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0, 4.0]);
        let outputs = domain
            .bind(XlaOperation::Scan(scan), vec![body], &[ArrayIrValue::Array(carry), ArrayIrValue::Array(xs)])
            .unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(read_f32s(&client, program_array(&outputs[0])), vec![10.0]);
        assert_eq!(program_array(&outputs[1]).shape(), StaticShape::new(vec![4]));
        assert_eq!(read_f32s(&client, program_array(&outputs[1])), vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_eager_bind_executes_scan_with_sharded_stacked_inputs() {
        use ryft_core::ScanOperation;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = domain_mesh(&client, "x", 2);
        let domain = XlaDomain::new(&client);
        let scalar_type = ArrayType::scalar(DataType::F32);

        // The same cumulative-sum scan as above, but with the stacked input sharded over the scanned (leading) axis
        // of a 2-device mesh: per-operation SPMD compilation handles the cross-shard slicing, and the inferred scan
        // output types leave shardings unspecified, so the outputs come back replicated over the mesh.
        let body = {
            let mut builder = XlaProgramBuilder::new();
            let carry = builder.add_input(scalar_type.clone().into());
            let x = builder.add_input(scalar_type.clone().into());
            let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![carry, x]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![sum, sum], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let scan = ScanOperation::<XlaConstant>::new(1, 4);

        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let xs_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(sharding)
            .unwrap();
        let xs = Array::from_host_buffer(
            &client,
            xs_type,
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let carry = f32_scalar(&client, &mesh, 0.0);
        let outputs = domain
            .bind(XlaOperation::Scan(scan), vec![body], &[ArrayIrValue::Array(carry), ArrayIrValue::Array(xs)])
            .unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(read_f32s(&client, program_array(&outputs[0])), vec![10.0]);
        assert_eq!(program_array(&outputs[1]).shape(), StaticShape::new(vec![4]));
        assert_eq!(program_array(&outputs[1]).sharding(), &Sharding::replicated(mesh.logical_mesh().clone(), 1),);
        assert_eq!(read_f32s(&client, program_array(&outputs[1])), vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_eager_bind_executes_jit_call_and_reuses_cached_executable() {
        use std::sync::Arc;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);
        let vector_type = replicated_vector_type(&mesh, 4);

        // A staged jitted callee `x -> x * x` bound eagerly on concrete arrays dispatches straight through the
        // compiled per-operation path, mirroring JAX calling a jitted function from eager code.
        let callee = {
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(vector_type.clone().into());
            let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, input]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let operation = XlaOperation::JitCall(JitCallOperation::new(0));
        let callee = Arc::new(callee);

        let input = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0, 4.0]);
        let first = domain
            .bind(operation.clone(), CalleeRegionDriver::new(&[callee.clone()]), &[ArrayIrValue::Array(input.clone())])
            .unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(read_f32s(&client, program_array(&first[0])), vec![1.0, 4.0, 9.0, 16.0]);
        assert_eq!(domain.cache_size(), 1);

        // A repeated eager `jit_call` at the same input signature is a dispatch-cache hit.
        let second = domain.bind(operation, CalleeRegionDriver::new(&[callee]), &[ArrayIrValue::Array(input)]).unwrap();
        assert_eq!(read_f32s(&client, program_array(&second[0])), vec![1.0, 4.0, 9.0, 16.0]);
        assert_eq!(domain.cache_size(), 1, "a repeated eager jit_call must be a compile-cache hit");
    }

    #[test]
    fn test_eager_bind_executes_shard_map_over_sharded_inputs() {
        use crate::experimental::shard_map::{FlatTracedShardMap, ShardMap};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .into_iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices).unwrap();
        let domain = XlaDomain::new(&client);

        // Manual shard-map body `local -> local + local` over `f32[4]` sharded across the 2-device mesh: each device
        // doubles its own 2-element shard inside the manual region.
        let sharding = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));
        let local_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]));
        let body_program = {
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(local_type.clone().into());
            let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let body = FlatTracedShardMap::from_parts(
            ShardMap::from_shardings(
                logical_mesh,
                vec![sharding.clone()],
                vec![sharding.clone()],
                vec!["x".to_string()],
                true,
            ),
            vec![global_type.clone()],
            vec![local_type.clone()],
            vec![global_type],
            vec![local_type],
            body_program,
        );
        let (operation, body_region) = ShardMapOperation::from_body(body);
        let operation = XlaOperation::ShardMap(Box::new(operation));

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(sharding.clone())
            .unwrap();
        let input = Array::from_host_buffer(
            &client,
            input_type,
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let outputs = domain.bind(operation, vec![body_region], &[ArrayIrValue::Array(input)]).unwrap();

        assert_eq!(outputs.len(), 1);
        let output = program_array(&outputs[0]);
        assert_eq!(output.sharding(), &sharding);
        assert_eq!(output.shards().len(), 2);
        let shard_values = output
            .addressable_shards()
            .map(|shard| {
                let bytes = shard.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
                values_from_bytes::<f32>(bytes.as_slice())
            })
            .collect::<Vec<_>>();
        assert_eq!(shard_values, vec![vec![2.0, 4.0], vec![6.0, 8.0]]);
    }

    #[test]
    fn test_eager_bind_rejects_collective_outside_a_mapping_context() {
        use ryft_core::{CollectiveKind, CollectiveOperation};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = array_domain(&client);

        // A collective on a concrete array outside any `batch` / `shard_map` binder has no axis to resolve against,
        // mirroring JAX's "unbound axis name" error for a top-level `psum`. The value-level `Collective` capability
        // is not even implemented for `Array` (its dispatch domain carries no named-axis environment), so this binds
        // the operation directly and asserts the axis-resolution failure surfaced at compile time.
        let input = f32_vector(&client, &mesh, &[1.0, 2.0]);
        assert!(matches!(
            domain.bind(CollectiveOperation::new("i".to_string(), CollectiveKind::PSum), Vec::new(), &[input]),
            Err(ProgramError::InvalidArgument { message })
                if message == "collective over axis `i` can only be lowered inside a shard_map manual region",
        ));
    }

    #[test]
    fn test_eager_bind_executes_print_effect() {
        use ryft_core::PrintOperation;

        use crate::experimental::debugging::{ensure_print_handler_registered, with_captured_prints};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = array_domain(&client);
        assert_eq!(ensure_print_handler_registered(&client), Ok(()));

        // The effectful `print` rides the compiled per-operation program as a token-threaded `@ryft.print` custom
        // call: eagerly binding it fires the host callback once and passes the payload through unchanged.
        let r#type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let input =
            Array::from_host_buffer(&client, r#type, mesh.clone(), values_to_bytes::<f64>(&[1.5, 2.5]).as_slice())
                .unwrap();
        let (outputs, lines) =
            with_captured_prints(|| domain.bind(PrintOperation::new("x"), Vec::new(), &[input]).unwrap());

        assert_eq!(lines, vec!["x: [1.5, 2.5]".to_string()]);
        assert_eq!(outputs.len(), 1);
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let bytes = outputs[0]
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        assert_eq!(values_from_bytes::<f64>(bytes.as_slice()), vec![1.5, 2.5]);
    }

    #[test]
    fn test_eager_bind_surfaces_shape_mismatch_as_type_error() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let domain = array_domain(&client);

        // Mismatched operand shapes fail at bind time through type inference on the traced single-instruction
        // program — never reaching PJRT compilation or execution.
        let left = f32_vector(&client, &mesh, &[1.0, 2.0]);
        let right = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0]);
        assert!(matches!(
            domain.bind(AddOperation::new(), Vec::new(), &[left, right]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`add` input types are not broadcast-compatible",
        ));
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
        let array_type =
            ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(2), Dimension::Dynamic(batch.clone())]))
                .with_memory(Memory::Host { pinned: true });
        let output_type =
            ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(2)]));

        let (restored_inputs, restored_outputs) =
            PersistentArraySignatureV3::encode(std::slice::from_ref(&array_type), std::slice::from_ref(&output_type))
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
    fn test_persistent_array_signature_canonicalizes_names_but_preserves_variable_relationships() {
        let bounds = DimensionBounds::non_negative(Some(4)).unwrap();
        let batch = DimensionVariable::new("batch", bounds);
        let renamed = DimensionVariable::new("renamed", bounds);
        let shared = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Dynamic(batch)]),
        );
        let shared_renamed = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(renamed.clone()), Dimension::Dynamic(renamed)]),
        );
        let first = DimensionVariable::new("first", bounds);
        let second = DimensionVariable::new("second", bounds);
        let independent =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(first), Dimension::Dynamic(second)]));

        let canonical = |r#type: &ArrayType| {
            serde_json::to_vec(&PersistentArraySignatureV3::encode(std::slice::from_ref(r#type), &[]).into_canonical())
                .unwrap()
        };
        assert_eq!(canonical(&shared), canonical(&shared_renamed));
        assert_ne!(canonical(&shared), canonical(&independent));
    }
}
