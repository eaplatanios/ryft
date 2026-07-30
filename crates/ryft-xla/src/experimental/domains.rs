use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::hash::Hash;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, LazyLock, OnceLock};
use std::time::{Duration, Instant};

use prost::Message;
use ryft_pjrt::protos::CompilationOptions;
use ryft_pjrt::{Buffer, Client, Execution, LoadOptions, LoadedExecutable, Program as PjrtProgram, Value as PjrtValue};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use ryft_core::InterpretationDriver;
use ryft_core::backends::arrays::Array as ReferenceArray;
use ryft_core::backends::scalars::Scalar;
use ryft_core::batching::BatchingError;
use ryft_core::compilation::{
    AnalyzableCompilationDomain, CallRequest, CompilationCacheDomain, CompilationContext, CompilationDomain,
    CompileRequest, DiskCache, LoweringRequest, StageRequest, StagedFunction,
};
use ryft_core::contexts::{Context, Domain};
use ryft_core::differentiation::DifferentiationError;
use ryft_core::interpretation::InterpretableOperation;
use ryft_core::macros::check_count;
use ryft_core::operations::constants::{
    Constant, ConstantOperation, Fill, Iota, IotaOperation, ONE_OPERATION_NAME, One, OneOperation, ZERO_OPERATION_NAME,
    Zero, ZeroOperation,
};
use ryft_core::operations::manipulation::{ConvertElementType, LegacyBroadcast};
use ryft_core::parameters::{Parameterized, Placeholder};
use ryft_core::programs::ProgramError;
use ryft_core::programs::operations::Operation;
use ryft_core::programs::regions::BindingRegionDriver;
use ryft_core::programs::types::{Type, TypeError, Typed};
use ryft_core::sharding::{
    Device, DeviceId, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension,
};
use ryft_core::tracing::DomainTracer;
use ryft_core::types::dimensions::{DimensionBounds, DimensionVariable};
use ryft_core::types::{
    ArrayType, DataType, Dimension, Layout, Memory, Shape, StridedLayout, Tile, TileDimension, TiledLayout,
};

use super::lowering::XlaExecutableSignature;
use super::operations::ShardMapOperation;
use super::ops::{FlatXlaProgram, JitCallOperation, XlaArrayConstant, XlaConstant, XlaOperation, XlaProgramBuilder};
use super::shard_map::ShardMapTraceError;
use crate::arrays_v0::{ArrayError, ShardDescriptor, ShardLayout};
use crate::{Array, Error, FromPjrt, ToPjrt};

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
    type Type = ArrayType;
    type Value = Array<'c>;
    type Constant = XlaConstant;
    type Operation = XlaOperation;
}

impl<'c> Context for XlaDomain<'c> {
    /// [`XlaConstant`] is a [`CaptureReference`](ryft_core::captures::CaptureReference) — a symbolic index into a
    /// compiled function's capture table carrying only a type and no data — so there is nothing to materialize
    /// without the surrounding capture table and lifting is always rejected.
    fn lift(&self, constant: XlaConstant) -> Result<Array<'c>, ProgramError> {
        Err(TypeError::invalid(format!("xla captured constant {constant} requires a captured program capture table"))
            .into())
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
        let name = operation.name();
        if inputs.is_empty() && (name == ZERO_OPERATION_NAME || name == ONE_OPERATION_NAME) {
            let array_type = eager_identity_output_type(&operation)?;
            validate_identity_synthesis(name, &array_type)?;
            // The direct constant-materialization fast path needs a concrete device mesh; mesh-less domains fall
            // through to the compiled eager path below, which derives a default execution mesh instead.
            if self.mesh.is_some() {
                let kind = if name == ZERO_OPERATION_NAME { ConstantKind::Zero } else { ConstantKind::One };
                let value = self.constant(&array_type, kind).map_err(|error| TypeError::invalid(error.to_string()))?;
                return Ok(vec![value]);
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
/// synthesizes constants through the active context's type-driven [`Zero`] / [`One`] / [`Fill`] / [`Iota`] leaves.
/// The binds below take the constant-materialization fast path on domains constructed with a concrete mesh and the
/// compiled eager dispatch path (over a derived default mesh) otherwise.
impl<'c> Zero<Array<'c>> for XlaDomain<'c> {
    fn zero(&self, r#type: &ArrayType) -> Result<Array<'c>, ProgramError> {
        let mut outputs = self.bind(ZeroOperation::new(r#type.clone()), Vec::new(), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Refer to the documentation of this domain's [`Zero`] implementation for more information.
impl<'c> One<Array<'c>> for XlaDomain<'c> {
    fn one(&self, r#type: &ArrayType) -> Result<Array<'c>, ProgramError> {
        let mut outputs = self.bind(OneOperation::new(r#type.clone()), Vec::new(), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Refer to the documentation of this domain's [`Zero`] implementation for more information.
impl<'c> Fill<Scalar, Array<'c>> for XlaDomain<'c> {
    fn fill(&self, r#type: &ArrayType, value: Scalar) -> Result<Array<'c>, ProgramError> {
        let value = value.convert_element_type(r#type.data_type())?;
        let literal =
            ReferenceArray::new(ArrayType::scalar(r#type.data_type()).with_memory(r#type.memory()), vec![value])?;
        let mut outputs = self.bind(ConstantOperation::new(literal), Vec::new(), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        outputs.remove(0).legacy_broadcast(r#type.clone(), &[])
    }
}

/// Refer to the documentation of this domain's [`Zero`] implementation for more information.
impl<'c> Iota<Array<'c>> for XlaDomain<'c> {
    fn iota(&self, r#type: &ArrayType, dimension: usize) -> Result<Array<'c>, ProgramError> {
        let mut outputs = self.bind(IotaOperation::new(r#type.clone(), dimension)?, Vec::new(), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// [`XlaConstant`] is a [`CaptureReference`](ryft_core::captures::CaptureReference) carrying only a type and no
/// data, so — exactly like [`Context::lift`], to which this delegates — captured-constant materialization is always
/// rejected outside a surrounding capture table. The implementation exists because interpretation- and
/// batching-capable operation families require a [`Constant`] leaf on their contexts; programs whose constants were
/// compiled into capture tables never take this path.
impl<'c> Constant<Array<'c>, XlaConstant> for XlaDomain<'c> {
    fn constant(&self, value: XlaConstant) -> Result<Array<'c>, ProgramError> {
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
impl<'c> InterpretableOperation<XlaDomain<'c>> for JitCallOperation {
    fn interpret<D: InterpretationDriver<XlaDomain<'c>>>(
        &self,
        context: &XlaDomain<'c>,
        driver: &D,
        inputs: &[Array<'c>],
    ) -> Result<Vec<Array<'c>>, ProgramError> {
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
impl<'c> InterpretableOperation<XlaDomain<'c>> for ShardMapOperation<XlaArrayConstant> {
    fn interpret<D: InterpretationDriver<XlaDomain<'c>>>(
        &self,
        _context: &XlaDomain<'c>,
        _driver: &D,
        _inputs: &[Array<'c>],
    ) -> Result<Vec<Array<'c>>, ProgramError> {
        Err(ProgramError::UnsupportedOperation {
            message: "eager shard_map replay must bind through a client-backed domain context".to_string(),
        })
    }
}

/// Returns the single [`ArrayType`] produced by a nullary additive/multiplicative identity operation
/// ([`ZERO_OPERATION_NAME`] / [`ONE_OPERATION_NAME`]). The identity fast path in [`Context::bind`] materializes these
/// constants directly through the runtime client instead of compiling a program.
fn eager_identity_output_type<O: Operation<ArrayType>>(operation: &O) -> Result<ArrayType, ProgramError> {
    let mut output_types = operation.infer_output_types(&[], &[])?;
    if output_types.len() != 1 {
        return Err(TypeError::invalid(format!(
            "xla identity operation `{}` must produce exactly one output but produced {}",
            operation.name(),
            output_types.len(),
        ))
        .into());
    }
    Ok(output_types.pop().expect("output count checked above"))
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
        inputs: &[Array<'c>],
    ) -> Result<Vec<Array<'c>>, ProgramError> {
        let Some(client) = self.client else {
            return Err(ProgramError::InvalidArgument {
                message: format!(
                    "xla domain cannot eagerly execute operation `{}` without a PJRT client",
                    operation.name(),
                ),
            });
        };
        self.validate_eager_placement(client, inputs)?;

        // Trace the single-instruction program over the inputs' physical types, shardings included, attaching the
        // provided region bodies to that instruction.
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let region_input_types = vec![None; driver.regions().count()];
        let builder = Rc::new(RefCell::new(XlaProgramBuilder::new()));
        let region_ids = driver.import_into(&builder, &region_input_types)?;
        let output_atoms = {
            let mut builder = builder.borrow_mut();
            let input_atoms = input_types.iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
            builder.add_instruction(operation, region_ids, input_atoms)?.to_vec()
        };
        let output_count = output_atoms.len();
        let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        let program: FlatXlaProgram =
            builder.build(output_atoms, vec![Placeholder; input_types.len()], vec![Placeholder; output_count])?;

        // Derive the mesh after tracing so that input-free operations can fall back to their inferred output
        // shardings, then compile through the domain's cache (a repeated eager operation is a cache hit) and
        // execute via PJRT.
        let mesh = self.eager_mesh(client, inputs, program.output_types().as_slice())?;
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
            .execute_xla_program(&compiled, inputs.to_vec())
            .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;

        // Execution already attached this domain's client to every output, so attaching the compile cache is all
        // that is left for chained eager operations and transforms over the outputs to recover a context that keeps
        // executing on the same client and keeps hitting the same compile cache.
        Ok(outputs.into_iter().map(|output| output.with_compilation_cache(Arc::clone(&self.cache))).collect())
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

/// Backend-specific per-call options for the [`CompilationDomain`] implementation on
/// [`XlaDomain`]. Carries the device mesh, optional sharding overrides for inputs/outputs,
/// and optional per-input donation flags.
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
}

impl XlaOptions {
    /// Constructs default options for `mesh` with no shardings overrides and no donation.
    #[inline]
    pub fn new(mesh: DeviceMesh) -> Self {
        Self { mesh, in_shardings: None, out_shardings: None, donation_flags: None, feedback_directed_profile: None }
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

const XLA_PERSISTENT_EXECUTABLE_MAGIC: &[u8; 8] = b"RYFTXLA2";
const XLA_PERSISTENT_EXECUTABLE_SCHEMA_VERSION: u32 = 3;
const XLA_PERSISTENT_EXECUTABLE_FEATURE_FLAGS: u64 = 1;
const XLA_PERSISTENT_KEY_SCHEMA_VERSION: u32 = 3;
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
struct XlaPersistentKeyV3<'a> {
    schema_version: u32,
    stable_hlo: &'a str,
    compilation_options: &'a [u8],
    signature: PersistentCanonicalArraySignatureV3,
    input_mapping: Vec<Option<u64>>,
    output_mapping: Vec<Option<u64>>,
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
struct XlaPersistentExecutableMetadataV3 {
    schema_version: u32,
    feature_flags: u64,
    compilation_options: Vec<u8>,
    signature: PersistentArraySignatureV3,
    input_mapping: Vec<Option<u64>>,
    output_mapping: Vec<Option<u64>>,
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
    /// Validates that this domain's PJRT client owns `program`.
    fn validate_xla_program_owner(&self, program: &XlaCompiledProgram<'c>) -> Result<(), XlaDomainError> {
        if program.executable.is_owned_by(self.client()?) {
            Ok(())
        } else {
            Err(XlaDomainError::ExecutableClientMismatch)
        }
    }

    /// Enqueues `program` and returns its possibly still pending flat outputs together with a whole-execution fence.
    pub(crate) fn execute_compiled_async(
        &self,
        program: &XlaCompiledProgram<'c>,
        inputs: Vec<Array<'c>>,
    ) -> Result<Execution<Vec<Array<'c>>>, XlaDomainError> {
        self.validate_xla_program_owner(program)?;
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
        let inputs = program.signature.project_inputs(inputs.as_slice());
        let physical_input_types = program.signature.project_inputs(&program.input_types);
        let inputs = materialize_zero_space_carriers(self.client()?, physical_input_types.as_slice(), inputs)?;
        let inputs = reshard_inputs_if_needed(self, &program.mesh, &program.expected_argument_shardings, inputs)?;
        let logical_donation_flags = std::iter::repeat_n(false, program.capture_count)
            .chain(program.donation_flags.iter().copied())
            .collect::<Vec<_>>();
        let mut donation_flags = program.signature.project_inputs(logical_donation_flags.as_slice());
        for (donation, input_type) in donation_flags.iter_mut().zip(physical_input_types) {
            if input_type.data_type().is_zero() {
                *donation = false;
            }
        }
        let physical_output_types = program.signature.project_outputs(&program.output_types);
        let execution = execute_pjrt(
            self.client()?,
            &program.executable,
            &program.mesh,
            inputs,
            donation_flags.as_slice(),
            physical_output_types.as_slice(),
        )?;
        let (physical_outputs, fence) = execution.into_parts();
        let mut physical_outputs = physical_outputs.into_iter();
        let mut outputs = Vec::with_capacity(program.output_types.len());
        for (mapping, output_type) in program.signature.output_mapping().iter().zip(program.output_types.iter()) {
            match mapping {
                Some(_) => outputs.push(physical_outputs.next().unwrap()),
                None => outputs.push(
                    Array::from_zero_space(self.client()?, output_type.clone(), program.mesh.clone())?
                        .with_execution_fence(fence.clone()),
                ),
            }
        }
        assert!(physical_outputs.next().is_none());
        Ok(Execution::new(outputs, fence))
    }
}

impl<'c> CompilationDomain for XlaDomain<'c> {
    type LoweredProgram = XlaLoweredProgram;
    type CompiledProgram = XlaCompiledProgram<'c>;
    type Options = XlaOptions;
    type Error = XlaDomainError;
    fn stage<Request>(
        &self,
        mut request: Request,
    ) -> Result<StagedFunction<Self, Request::Input, Request::Output>, Self::Error>
    where
        Request: StageRequest<Self>,
    {
        if let Some(in_shardings) = request.options().in_shardings.clone() {
            let input_types = apply_signature_shardings(
                request.input_types().parameters().cloned().collect(),
                Some(in_shardings.as_slice()),
                "in",
            )?;
            request.replace_input_types(input_types)?;
        }
        request.trace(|options, output_types| {
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
        Ok(staged.into_lowered(program, output_types))
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
            |program| program.output_types().to_vec(),
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
            validate_xla_input_type(declared, actual.as_ref())?;
        }
        let output_types = executable.output_types().to_vec();
        let outputs = self.execute_xla_program(executable.compiled_program(), request.into_arguments())?;
        validate_runtime_outputs(&output_types, &outputs)?;
        Request::reconstruct(&executable, outputs)
    }
}

impl<'c> XlaDomain<'c> {
    fn lower_xla_program(
        &self,
        program: &FlatXlaProgram,
        capture_count: usize,
        options: &XlaOptions,
    ) -> Result<XlaLoweredProgram, XlaDomainError> {
        let input_types = program.input_types();
        if capture_count > input_types.len() {
            return Err(XlaDomainError::InvalidCompilationOptions {
                reason: format!(
                    "capture_count is {capture_count}, but the program has only {} flat input(s)",
                    input_types.len(),
                ),
            });
        }
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

        let effective_input_types =
            capture_types.iter().cloned().chain(public_input_types.iter().cloned()).collect::<Vec<_>>();
        let output_types = apply_signature_shardings(program.output_types(), options.out_shardings.as_deref(), "out")?;
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
        let compilation_options = jit_compilation_options(
            self.compilation_options.as_ref(),
            options.mesh.devices().len(),
            options.feedback_directed_profile.as_ref(),
        );
        // The target platform gates platform-specific lowerings (e.g., the block-scaled dot fast path); a failed
        // platform query degrades to the portable lowerings instead of failing compilation.
        let target_platform =
            self.client.and_then(|client| client.platform_name().ok()).map(|platform| platform.into_owned());
        let lowered_module = crate::experimental::lowering::lower_mlir_module_for_program(
            program,
            &[],
            &effective_input_types,
            &output_types,
            "main",
            Some(logical_argument_shardings.as_slice()),
            result_shardings.as_deref(),
            target_platform.as_deref(),
        )
        .map_err(|error| XlaDomainError::Lowering(error.into()))?;
        let (stable_hlo, signature) = lowered_module.into_parts();
        for (mapping, donation) in signature.input_mapping()[capture_count..].iter().zip(donation_flags.iter_mut()) {
            if mapping.is_none() {
                *donation = false;
            }
        }
        let expected_argument_shardings = signature.project_inputs(logical_argument_shardings.as_slice());
        let client = self.client()?;

        Ok(XlaLoweredProgram {
            stable_hlo: stable_hlo.into(),
            compilation_options,
            input_types: effective_input_types.into(),
            output_types: output_types.into(),
            signature,
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
        let key = XlaPersistentKeyV3 {
            schema_version: XLA_PERSISTENT_KEY_SCHEMA_VERSION,
            stable_hlo: &program.stable_hlo,
            compilation_options: compilation_options.as_slice(),
            signature: PersistentArraySignatureV3::encode(&program.input_types, &program.output_types).into_canonical(),
            input_mapping: persistent_mapping(program.signature.input_mapping())?,
            output_mapping: persistent_mapping(program.signature.output_mapping())?,
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
        let pjrt_program = PjrtProgram::Mlir { bytecode: program.stable_hlo.as_bytes().to_vec() };
        let compilation_start = Instant::now();
        let executable = self.client()?.compile(&pjrt_program, &program.compilation_options)?;
        let compilation_duration = compilation_start.elapsed();
        Ok(XlaCompiledProgram {
            executable: Arc::new(executable),
            input_types: Arc::clone(&program.input_types),
            output_types: Arc::clone(&program.output_types),
            signature: program.signature.clone(),
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
        let metadata = XlaPersistentExecutableMetadataV3 {
            schema_version: XLA_PERSISTENT_EXECUTABLE_SCHEMA_VERSION,
            feature_flags: XLA_PERSISTENT_EXECUTABLE_FEATURE_FLAGS,
            compilation_options: program.compilation_options.encode_to_vec(),
            signature: PersistentArraySignatureV3::encode(&program.input_types, &program.output_types),
            input_mapping: persistent_mapping(program.signature.input_mapping())?,
            output_mapping: persistent_mapping(program.signature.output_mapping())?,
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
        let metadata: XlaPersistentExecutableMetadataV3 = serde_json::from_slice(&bytes[header_size..metadata_end])
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
        let physical_input_count = signature.input_mapping().iter().filter(|index| index.is_some()).count();
        if capture_count > input_types.len()
            || expected_argument_shardings.len() != physical_input_count
            || metadata.donation_flags.len() != input_types.len() - capture_count
        {
            return Err(persistent_error("capture or donation metadata has an invalid arity"));
        }
        let donation_flags = metadata.donation_flags;
        let compilation_options = CompilationOptions::decode(metadata.compilation_options.as_slice())
            .map_err(|error| persistent_error(format!("failed to decode compilation options: {error}")))?;
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

    fn analyze<Input: Parameterized<ArrayType>, Output: Parameterized<ArrayType>>(
        &self,
        executable_program: &ryft_core::compilation::ExecutableProgram<Self, Input, Output>,
    ) -> Result<Self::Analysis, Self::Error> {
        let program = executable_program.compiled_program();
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

fn validate_output_types(declared: &[ArrayType], actual: &[ArrayType]) -> Result<(), XlaDomainError> {
    if declared.len() != actual.len() {
        return Err(ProgramError::InvalidOutputCount { expected: declared.len(), actual: actual.len() }.into());
    }
    for (declared, actual) in declared.iter().zip(actual) {
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
        _ => Err(XlaDomainError::InvalidCompilationAnalysis { reason: format!("property '{name}' is not numeric") }),
    }
}

fn optional_pjrt_analysis<T>(result: Result<T, ryft_pjrt::Error>) -> Result<Option<T>, XlaDomainError> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(ryft_pjrt::Error::Unimplemented { .. } | ryft_pjrt::Error::Unavailable { .. }) => Ok(None),
        Err(error) => Err(error.into()),
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

/// Overlays SPMD partitioning fields on the base [`CompilationOptions`] template.
fn jit_compilation_options(
    base: &CompilationOptions,
    partition_count: usize,
    feedback_directed_profile: Option<&XlaFeedbackDirectedProfile>,
) -> CompilationOptions {
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

/// Materializes private false-predicate carriers for zero-space inputs that remain in the physical executable
/// signature, currently because their declared type contains dynamic dimensions. Static zero-space inputs are erased
/// before this point and never allocate a carrier.
fn materialize_zero_space_carriers<'c>(
    client: &'c Client<'c>,
    input_types: &[ArrayType],
    inputs: Vec<Array<'c>>,
) -> Result<Vec<Array<'c>>, XlaDomainError> {
    assert_eq!(input_types.len(), inputs.len());
    input_types
        .iter()
        .zip(inputs)
        .map(|(input_type, input)| {
            if !input_type.data_type().is_zero() {
                return Ok(input);
            }
            input.block_until_ready()?;
            let carrier_type = input.r#type().into_owned().with_data_type(DataType::Boolean);
            let element_count = carrier_type
                .element_count()
                .map_err(Error::from)?
                .expect("runtime arrays should only have static shapes");
            Ok(Array::from_host_buffer(client, carrier_type, input.mesh(), vec![0u8; element_count])?)
        })
        .collect()
}

/// Executes a compiled PJRT executable against `mesh` and reassembles per-device output buffers into distributed
/// [`Array`] values carrying `client`, so eager execution and free transforms over the outputs can recover their
/// execution domain. Mirrors `XlaDomain::execute_with_donation`.
fn execute_pjrt<'c>(
    client: &'c Client<'c>,
    executable: &LoadedExecutable<'c>,
    mesh: &DeviceMesh,
    inputs: Vec<Array<'c>>,
    donation_flags: &[bool],
    output_types: &[ArrayType],
) -> Result<Execution<Vec<Array<'c>>>, XlaDomainError> {
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
        outputs.push(
            Array::from_canonical_addressable_buffers(client, resolved_type, mesh.clone(), addressable_buffers)?
                .with_execution_fence(fence.clone()),
        );
    }
    Ok(Execution::new(outputs, fence))
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::Sharding;
    use ryft_core::compilation::stage_function;
    use ryft_core::operations::compare::{CompareOperation, ComparisonDirection};
    use ryft_core::operations::constants::{ConstantOperation, OneOperation};
    use ryft_core::operations::control_flow::{ConditionOperation, SelectOperation, WhileOperation};
    use ryft_core::operations::logical::AndOperation;
    use ryft_core::operations::math::{AddOperation, Atan2Operation, DivOperation, MulOperation, NegOperation};
    use ryft_core::programs::regions::CalleeRegionDriver;
    use ryft_core::sharding::ShardingDimension;
    use ryft_core::types::{Dimension, StaticShape};
    use ryft_pjrt::{ClientOptions, CpuClientOptions, load_cpu_plugin};

    use crate::tests::{values_from_bytes, values_to_bytes};

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

    #[test]
    fn test_feedback_directed_profile_round_trips_and_changes_compilation_options() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("profile.pb");
        let profile = XlaFeedbackDirectedProfile::new(vec![1, 2, 3, 4]).with_version(7);
        profile.write_to_file(&path).unwrap();
        let restored = XlaFeedbackDirectedProfile::from_file(&path).unwrap().with_version(7);
        assert_eq!(restored, profile);
        assert_eq!(restored.digest(), profile.digest());

        let baseline = jit_compilation_options(&CompilationOptions::default(), 2, None);
        let profiled = jit_compilation_options(&CompilationOptions::default(), 2, Some(&profile));
        assert!(baseline.executable_build_options.unwrap().fdo_profile.is_empty());
        assert_eq!(profiled.executable_build_options.unwrap().fdo_profile, vec![1, 2, 3, 4]);
        assert_eq!(profiled.profile_version, 7);
    }

    #[test]
    fn test_replacement_metadata_rejects_changed_invocation_contract() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_domain_mesh(&client, "x", 1);
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
        let mesh = cpu_domain_mesh(&client, "x", 2);
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
        let mesh = cpu_domain_mesh(&client, "x", 1);
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
        let mesh = cpu_domain_mesh(&client, "x", 2);
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
        use ryft_core::operations::constants::OneOperation;
        let array_type = ArrayType::scalar(DataType::Token);

        assert!(matches!(
            XlaDomain::token().bind(OneOperation::new(array_type.clone()), Vec::new(), &[]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "xla domain cannot synthesize one value for element type token"
        ));
    }

    #[test]
    fn test_domain_accessors_return_constructor_arguments() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let cloned = domain.clone();

        assert_eq!(domain.mesh().unwrap(), &mesh);
        assert_eq!(domain.compilation_options(), &CompilationOptions::default());
        assert!(Arc::ptr_eq(&domain.compilation_options, &cloned.compilation_options));
        assert!(size_of::<XlaDomain<'_>>() < size_of::<CompilationOptions>());
    }

    #[test]
    fn test_compilation_domain_impl_round_trips_through_core_pipeline() {
        use crate::tests::{values_from_bytes, values_to_bytes};
        use ryft_core::operations::math::Sin;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_domain_mesh(&client, "x", 1);
        let engine = XlaDomain::with_mesh(&client, mesh.clone());

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let options = XlaOptions::new(mesh.clone());
        let staged = stage_function(&engine, |x| x.sin().unwrap(), input_type.clone(), options).unwrap();
        let compiled: ryft_core::compilation::CompiledFunction<XlaDomain<'_>, ArrayType, ArrayType> =
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
        let array = ryft_core::compilation::call_function(&engine, compiled.executable_program(), source).unwrap();
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
            let staged =
                stage_function(&engine, |x| x.sin().unwrap(), input_type.clone(), XlaOptions::new(mesh.clone()))
                    .unwrap();
            let _: ryft_core::compilation::CompiledFunction<XlaDomain<'_>, ArrayType, ArrayType> =
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
        let mesh = cpu_domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let input_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let staged = stage_function(&domain, |input| input, input_type.clone(), XlaOptions::new(mesh.clone())).unwrap();
        let compiled: ryft_core::compilation::CompiledFunction<XlaDomain<'_>, ArrayType, ArrayType> =
            domain.compile(domain.lower(staged).unwrap()).unwrap();
        assert_eq!(compiled.compiled_program().output_types(), std::slice::from_ref(&input_type));

        let donated_staged: StagedFunction<XlaDomain<'_>, ArrayType, ArrayType> = stage_function(
            &domain,
            |input| input,
            input_type.clone(),
            XlaOptions { donation_flags: Some(vec![true]), ..XlaOptions::new(mesh.clone()) },
        )
        .unwrap();
        let donated_lowered = domain.lower(donated_staged).unwrap();
        assert_eq!(donated_lowered.lowered_program().donation_flags.as_ref(), &[false]);
        assert_eq!(
            domain.compilation_key(compiled.lowered().lowered_program()).unwrap(),
            domain.compilation_key(donated_lowered.lowered_program()).unwrap(),
        );

        // Boolean values retain physical `i1` arguments/results while the zero-space identity has an empty physical
        // signature. Their retained logical metadata also keeps their compilation identities distinct.
        let boolean_type = input_type.clone().with_data_type(DataType::Boolean);
        let boolean_staged: StagedFunction<XlaDomain<'_>, ArrayType, ArrayType> =
            stage_function(&domain, |input| input, boolean_type, XlaOptions::new(mesh.clone())).unwrap();
        let boolean_lowered = domain.lower(boolean_staged).unwrap();
        assert_ne!(
            domain.compilation_key(compiled.lowered().lowered_program()).unwrap(),
            domain.compilation_key(boolean_lowered.lowered_program()).unwrap(),
        );

        let input = Array::from_host_buffer(&client, input_type, mesh, []).unwrap();
        let output = ryft_core::compilation::call_function(&domain, compiled.executable_program(), input).unwrap();
        assert_eq!(output.data_type(), DataType::Zero);
        assert!(output.addressable_shards().next().unwrap().buffer().is_none());
    }

    #[test]
    fn test_compiled_mixed_zero_space_signature_projects_and_reconstructs_logical_values() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let value_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]))
            .with_sharding(sharding.clone())
            .unwrap();
        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]))
            .with_sharding(sharding)
            .unwrap();
        let options = XlaOptions { donation_flags: Some(vec![false, true]), ..XlaOptions::new(mesh.clone()) };
        let staged =
            stage_function(&domain, |(value, zero)| (zero, value), (value_type.clone(), zero_type.clone()), options)
                .unwrap();
        let lowered = domain.lower(staged).unwrap();
        assert!(lowered.lowered_program().stable_hlo().contains("func.func @main(%arg0: tensor<3xf32>"));
        let compiled: ryft_core::compilation::CompiledFunction<
            XlaDomain<'_>,
            (ArrayType, ArrayType),
            (ArrayType, ArrayType),
        > = domain.compile(lowered).unwrap();
        let value = Array::from_host_buffer(
            &client,
            value_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0]),
        )
        .unwrap();
        let zero = Array::from_host_buffer(&client, zero_type.clone(), mesh, []).unwrap();

        let (zero_output, value_output) =
            ryft_core::compilation::call_function(&domain, compiled.executable_program(), (value, zero)).unwrap();

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
    fn test_materialize_zero_space_carriers_preserves_the_concrete_runtime_shape() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_domain_mesh(&client, "x", 1);
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let dynamic_zero_type = ArrayType::new(
            DataType::Zero,
            Shape::new(vec![
                DimensionVariable::new("dynamic_zero", DimensionBounds::non_negative(Some(3)).unwrap()).into(),
            ]),
        )
        .with_sharding(sharding.clone())
        .unwrap();
        let runtime_zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]))
            .with_sharding(sharding)
            .unwrap();
        let zero = Array::from_host_buffer(&client, runtime_zero_type, mesh, []).unwrap();

        let carriers = materialize_zero_space_carriers(&client, &[dynamic_zero_type], vec![zero]).unwrap();

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
    fn test_xla_compilation_key_is_canonical_and_stable() {
        use ryft_core::operations::math::Sin;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_domain_mesh(&client, "x", 1);
        let domain = XlaDomain::with_mesh(&client, mesh.clone());
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let staged = stage_function(&domain, |x| x.sin().unwrap(), input_type, XlaOptions::new(mesh)).unwrap();
        let compiled: ryft_core::compilation::CompiledFunction<XlaDomain<'_>, ArrayType, ArrayType> =
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
        use ryft_core::compilation::StagedFunction;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_domain_mesh(&client, "x", 1);
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
            Ok(captures.into_iter().next().unwrap() + input)
        }
        let staged: StagedFunction<XlaDomain<'_>, ArrayType, ArrayType> = domain
            .stage(ryft_core::compilation::CompilationStagingRequest::<XlaDomain<'_>, _, ArrayType, ArrayType>::new(
                add_capture,
                vec![capture.clone()],
                input_type.clone(),
                XlaOptions::new(mesh.clone()).with_donate(true),
            ))
            .unwrap();
        let compiled = domain.compile(domain.lower(staged).unwrap()).unwrap();
        let compiled_analysis = domain.analyze(compiled.executable_program()).unwrap();
        assert_eq!(domain.analyze(compiled.executable_program()).unwrap(), compiled_analysis);

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
        let mut invalid_signature_metadata: XlaPersistentExecutableMetadataV3 =
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

        let mut incompatible_metadata: XlaPersistentExecutableMetadataV3 =
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
        use ryft_core::compilation::DiskCache;
        use ryft_core::operations::math::Sin;
        use tempfile::tempdir;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_domain_mesh(&client, "x", 1);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let directory = tempdir().unwrap();

        let first_domain = XlaDomain::with_configured_disk_cache(
            &client,
            DiskCache::open(directory.path()).unwrap().with_write_thresholds(Duration::ZERO, 0),
        );
        let staged =
            stage_function(&first_domain, |x| x.sin().unwrap(), input_type.clone(), XlaOptions::new(mesh.clone()))
                .unwrap();
        let first: ryft_core::compilation::CompiledFunction<XlaDomain<'_>, ArrayType, ArrayType> =
            first_domain.compile(first_domain.lower(staged).unwrap()).unwrap();
        assert_eq!(first_domain.cache.statistics().compilations, 1);
        drop(first);
        drop(first_domain);

        let second_domain = XlaDomain::with_configured_disk_cache(
            &client,
            DiskCache::open(directory.path()).unwrap().with_write_thresholds(Duration::ZERO, 0),
        );
        let staged = stage_function(&second_domain, |x| x.sin().unwrap(), input_type, XlaOptions::new(mesh)).unwrap();
        let _restored: ryft_core::compilation::CompiledFunction<XlaDomain<'_>, ArrayType, ArrayType> =
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

        let metadata = XlaPersistentExecutableMetadataV3 {
            schema_version: XLA_PERSISTENT_EXECUTABLE_SCHEMA_VERSION + 1,
            feature_flags: 0,
            compilation_options: CompilationOptions::default().encode_to_vec(),
            signature: PersistentArraySignatureV3::encode(&[], &[]),
            input_mapping: Vec::new(),
            output_mapping: Vec::new(),
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
        let mesh = cpu_domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);

        let left = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0, 4.0]);
        let right = f32_vector(&client, &mesh, &[10.0, 20.0, 30.0, 40.0]);
        let outputs = domain.bind(AddOperation, Vec::new(), &[left, right]).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(read_f32s(&client, &outputs[0]), vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_eager_bind_executes_promoted_and_broadcast_elementwise_operations() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);

        let scalar = f32_scalar(&client, &mesh, 2.0);
        let vector = f64_vector(&client, &mesh, &[1.0, 2.0, 4.0, 8.0]);

        let divide = domain.bind(DivOperation, Vec::new(), &[scalar.clone(), vector.clone()]).unwrap();
        assert_eq!(read_f64s(&client, &divide[0]), vec![2.0, 1.0, 0.5, 0.25]);

        let atan2 = domain.bind(Atan2Operation, Vec::new(), &[vector.clone(), scalar.clone()]).unwrap();
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
            .bind(SelectOperation, Vec::new(), &[boolean_scalar(&client, &mesh, true), scalar, vector])
            .unwrap();
        assert_eq!(read_f64s(&client, &select[0]), vec![2.0, 2.0, 2.0, 2.0]);

        let boolean_vector = boolean_vector(&client, &mesh, &[true, false, true, false]);
        let and = domain
            .bind(AndOperation, Vec::new(), &[boolean_scalar(&client, &mesh, true), boolean_vector])
            .unwrap();
        assert_eq!(read_booleans(&client, &and[0]), vec![true, false, true, false]);
    }

    #[test]
    fn test_eager_bind_executes_unary_operation() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);

        let input = f32_vector(&client, &mesh, &[1.0, -2.0, 3.5, 0.0]);
        let outputs = domain.bind(NegOperation, Vec::new(), &[input]).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(read_f32s(&client, &outputs[0]), vec![-1.0, 2.0, -3.5, 0.0]);
    }

    #[test]
    fn test_eager_fill_materializes_scalar_literal_then_broadcasts_over_a_default_mesh() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let domain = XlaDomain::new(&client);

        for memory in [Memory::Device, Memory::Host { pinned: true }, Memory::Host { pinned: false }] {
            let r#type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])).with_memory(memory);
            let output = domain.fill(&r#type, Scalar::from(2.5f64)).unwrap();

            assert_eq!(output.r#type().memory(), memory);
            assert_eq!(output.shape(), StaticShape::new(vec![3]));
            assert_eq!(read_f32s(&client, &output), vec![2.5, 2.5, 2.5]);
        }

        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]));
        let value = Scalar::from(num_complex::Complex::new(1.5f64, -2.0f64));
        let output = domain.fill(&r#type, value).unwrap();
        assert_eq!(read_f32s(&client, &output), vec![1.5, 1.5]);

        // A complex fill value lowers as two real part splats composed through `stablehlo.complex`, and a `c64`
        // buffer's bytes are the interleaved `f32` real and imaginary parts of its elements.
        let r#type = ArrayType::new(DataType::C64, Shape::new(vec![Dimension::Static(2)]));
        let value = Scalar::from(num_complex::Complex::new(1.5f32, -2.0f32));
        let output = domain.fill(&r#type, value).unwrap();

        assert_eq!(output.shape(), StaticShape::new(vec![2]));
        assert_eq!(read_f32s(&client, &output), vec![1.5, -2.0, 1.5, -2.0]);

        // Integer literals remain exact through conversion and scalar constant lowering, including values that
        // cannot be represented exactly by the floating-point intermediary used for floating-point constants.
        let r#type = ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(2)]));
        let value = u64::MAX - 1;
        let output = domain.fill(&r#type, Scalar::U64(value)).unwrap();
        assert_eq!(read_u64s(&client, &output), vec![value, value]);
    }

    #[test]
    fn test_eager_bind_reuses_cached_executable_for_repeated_operations() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);
        assert_eq!(domain.cache_size(), 0);

        let left = f32_vector(&client, &mesh, &[1.0, 2.0]);
        let right = f32_vector(&client, &mesh, &[3.0, 4.0]);
        let first = domain.bind(AddOperation, Vec::new(), &[left.clone(), right.clone()]).unwrap();
        assert_eq!(domain.cache_size(), 1);

        let second = domain.bind(AddOperation, Vec::new(), &[left, right]).unwrap();
        assert_eq!(domain.cache_size(), 1, "a repeated eager operation must be a compile-cache hit");
        assert_eq!(read_f32s(&client, &first[0]), read_f32s(&client, &second[0]));

        // A different input signature compiles (and caches) a distinct executable.
        let wider_left = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0]);
        let wider_right = f32_vector(&client, &mesh, &[4.0, 5.0, 6.0]);
        domain.bind(AddOperation, Vec::new(), &[wider_left, wider_right]).unwrap();
        assert_eq!(domain.cache_size(), 2);
    }

    #[test]
    fn test_eager_bind_rejects_inputs_placed_on_a_foreign_device() {
        let plugin = load_cpu_plugin().unwrap();
        let domain_client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let foreign_client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let foreign_mesh = cpu_domain_mesh(&foreign_client, "x", 2);
        let domain = XlaDomain::new(&domain_client);

        // An input that carries an attached client is rejected by client identity.
        let input = f32_vector(&foreign_client, &foreign_mesh, &[1.0, 2.0]);
        assert!(matches!(
            domain.bind(NegOperation, Vec::new(), &[input.clone()]),
            Err(ProgramError::InvalidArgument { message })
                if message == "received incompatible devices for eager xla execution: input #0 is owned by a \
                    different PJRT client than this domain's client",
        ));

        // An input with no attached client falls back to the device-set membership check.
        let mut clientless_input = input;
        clientless_input.detach_client_for_tests();
        assert!(matches!(
            domain.bind(NegOperation, Vec::new(), &[clientless_input]),
            Err(ProgramError::InvalidArgument { message })
                if message == "received incompatible devices for eager xla execution: input #0 is placed on device \
                    1, which does not belong to this domain's PJRT client",
        ));
    }

    #[test]
    fn test_eager_bind_executes_condition_with_concrete_predicate() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);
        let vector_type = replicated_vector_type(&mesh, 4);

        let doubled = {
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(vector_type.clone());
            let output = builder.add_instruction(AddOperation, Vec::new(), vec![input, input]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let squared = {
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(vector_type.clone());
            let output = builder.add_instruction(MulOperation, Vec::new(), vec![input, input]).unwrap()[0];
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
                &[boolean_scalar(&client, &mesh, true), input.clone()],
            )
            .unwrap();
        assert_eq!(read_f32s(&client, &true_outputs[0]), vec![2.0, 4.0, 6.0, 8.0]);

        let false_outputs =
            domain.bind(operation, [doubled, squared], &[boolean_scalar(&client, &mesh, false), input]).unwrap();
        assert_eq!(read_f32s(&client, &false_outputs[0]), vec![1.0, 4.0, 9.0, 16.0]);
        assert_eq!(domain.cache_size(), 1, "both predicate values must share one compiled executable");
    }

    #[test]
    fn test_eager_bind_executes_bounded_while() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);

        // Loop `state = state + 1` while `state < 3`, starting from `0`.
        let condition = {
            let mut builder = XlaProgramBuilder::new();
            let state = builder.add_input(scalar_type.clone());
            let literal = ReferenceArray::new(scalar_type.clone(), vec![Scalar::from(3.0f32)]).unwrap();
            let limit = builder.add_instruction(ConstantOperation::new(literal), Vec::new(), vec![]).unwrap()[0];
            let predicate = builder
                .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), Vec::new(), vec![state, limit])
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
            let state = builder.add_input(scalar_type.clone());
            let one = builder.add_instruction(OneOperation::new(scalar_type.clone()), Vec::new(), vec![]).unwrap()[0];
            let next = builder.add_instruction(AddOperation, Vec::new(), vec![state, one]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![next], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let operation = XlaOperation::While(WhileOperation::new());

        let outputs = domain.bind(operation, vec![condition, body], &[f32_scalar(&client, &mesh, 0.0)]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(read_f32s(&client, &outputs[0]), vec![3.0]);
    }

    #[test]
    fn test_eager_bind_executes_elementwise_operation_on_sharded_inputs() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = cpu_domain_mesh(&client, "x", 2);
        let domain = XlaDomain::new(&client);

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
        let outputs = domain.bind(AddOperation, Vec::new(), &[input.clone(), input]).unwrap();

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
        use ryft_core::operations::control_flow::ScanOperation;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);
        let scalar_type = replicated_scalar_type(&mesh, DataType::F32);

        // Carry-only scan body `carry -> (carry + 1, carry + 1)`: the first output is the next carry and the second
        // is the per-step stacked output, so scanning 4 steps from `0` yields the cumulative sums `[1, 2, 3, 4]`.
        let body = {
            let mut builder = XlaProgramBuilder::new();
            let carry = builder.add_input(scalar_type.clone());
            let one = builder.add_instruction(OneOperation::new(scalar_type.clone()), Vec::new(), vec![]).unwrap()[0];
            let next = builder.add_instruction(AddOperation, Vec::new(), vec![carry, one]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![next, next],
                    vec![Placeholder; 1],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let scan = ScanOperation::<XlaArrayConstant>::new(1, 4);

        let outputs = domain.bind(XlaOperation::Scan(scan), [body], &[f32_scalar(&client, &mesh, 0.0)]).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(read_f32s(&client, &outputs[0]), vec![4.0]);
        assert_eq!(outputs[1].shape(), StaticShape::new(vec![4]));
        assert_eq!(read_f32s(&client, &outputs[1]), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_eager_bind_executes_scan_with_stacked_inputs() {
        use ryft_core::operations::control_flow::ScanOperation;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);
        let scalar_type = ArrayType::scalar(DataType::F32);

        // Cumulative-sum scan body `(carry, x) -> (carry + x, carry + x)` over the stacked input `[1, 2, 3, 4]`
        // starting from carry `0`: the final carry is the total `10` and the stacked per-step outputs are the
        // running sums `[1, 3, 6, 10]`. The body's metadata-free declared types are refined by the concrete input
        // types, which carry normalized shardings, so the scan binds eagerly despite the metadata mismatch.
        let body = {
            let mut builder = XlaProgramBuilder::new();
            let carry = builder.add_input(scalar_type.clone());
            let x = builder.add_input(scalar_type.clone());
            let sum = builder.add_instruction(AddOperation, Vec::new(), vec![carry, x]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![sum, sum], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let scan = ScanOperation::<XlaArrayConstant>::new(1, 4);

        let carry = f32_scalar(&client, &mesh, 0.0);
        let xs = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0, 4.0]);
        let outputs = domain.bind(XlaOperation::Scan(scan), vec![body], &[carry, xs]).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(read_f32s(&client, &outputs[0]), vec![10.0]);
        assert_eq!(outputs[1].shape(), StaticShape::new(vec![4]));
        assert_eq!(read_f32s(&client, &outputs[1]), vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_eager_bind_executes_scan_with_sharded_stacked_inputs() {
        use ryft_core::operations::control_flow::ScanOperation;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = cpu_domain_mesh(&client, "x", 2);
        let domain = XlaDomain::new(&client);
        let scalar_type = ArrayType::scalar(DataType::F32);

        // The same cumulative-sum scan as above, but with the stacked input sharded over the scanned (leading) axis
        // of a 2-device mesh: per-operation SPMD compilation handles the cross-shard slicing, and the inferred scan
        // output types leave shardings unspecified, so the outputs come back replicated over the mesh.
        let body = {
            let mut builder = XlaProgramBuilder::new();
            let carry = builder.add_input(scalar_type.clone());
            let x = builder.add_input(scalar_type.clone());
            let sum = builder.add_instruction(AddOperation, Vec::new(), vec![carry, x]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![sum, sum], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let scan = ScanOperation::<XlaArrayConstant>::new(1, 4);

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
        let outputs = domain.bind(XlaOperation::Scan(scan), vec![body], &[carry, xs]).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(read_f32s(&client, &outputs[0]), vec![10.0]);
        assert_eq!(outputs[1].shape(), StaticShape::new(vec![4]));
        assert_eq!(outputs[1].sharding(), &Sharding::replicated(mesh.logical_mesh().clone(), 1));
        assert_eq!(read_f32s(&client, &outputs[1]), vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_eager_bind_executes_jit_call_and_reuses_cached_executable() {
        use std::rc::Rc;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);
        let vector_type = replicated_vector_type(&mesh, 4);

        // A staged jitted callee `x -> x * x` bound eagerly on concrete arrays dispatches straight through the
        // compiled per-operation path, mirroring JAX calling a jitted function from eager code.
        let callee = {
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(vector_type.clone());
            let output = builder.add_instruction(MulOperation, Vec::new(), vec![input, input]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let operation = XlaOperation::JitCall(JitCallOperation::new());
        let callee = Rc::new(callee);

        let input = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0, 4.0]);
        let first = domain
            .bind(operation.clone(), CalleeRegionDriver::new(&[callee.clone()]), &[input.clone()])
            .unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(read_f32s(&client, &first[0]), vec![1.0, 4.0, 9.0, 16.0]);
        assert_eq!(domain.cache_size(), 1);

        // A repeated eager `jit_call` at the same input signature is a dispatch-cache hit.
        let second = domain.bind(operation, CalleeRegionDriver::new(&[callee]), &[input]).unwrap();
        assert_eq!(read_f32s(&client, &second[0]), vec![1.0, 4.0, 9.0, 16.0]);
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
            let input = builder.add_input(local_type.clone());
            let output = builder.add_instruction(AddOperation, Vec::new(), vec![input, input]).unwrap()[0];
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
        let outputs = domain.bind(operation, vec![body_region], &[input]).unwrap();

        assert_eq!(outputs.len(), 1);
        let output = &outputs[0];
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
        use ryft_core::operations::collectives::{CollectiveKind, CollectiveOperation};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);

        // A collective on a concrete array outside any `batch` / `shard_map` binder has no axis to resolve against,
        // mirroring JAX's "unbound axis name" error for a top-level `psum`. The value-level `Collective` capability
        // is not even implemented for `Array` (its dispatch domain carries no named-axis environment), so this binds
        // the operation directly and asserts the axis-resolution failure surfaced at compile time.
        let input = f32_vector(&client, &mesh, &[1.0, 2.0]);
        assert!(matches!(
            domain.bind(CollectiveOperation::new("i".to_string(), CollectiveKind::PSum), Vec::new(), &[input]),
            Err(ProgramError::InvalidArgument { message })
                if message == "collective over axis 'i' can only be lowered inside a shard_map manual region",
        ));
    }

    #[test]
    fn test_eager_bind_executes_print_effect() {
        use ryft_core::operations::debugging::PrintOperation;

        use crate::experimental::debugging::{ensure_print_handler_registered, with_captured_prints};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);
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
        let mesh = cpu_domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);

        // Mismatched operand shapes fail at bind time through type inference on the traced single-instruction
        // program — never reaching PJRT compilation or execution.
        let left = f32_vector(&client, &mesh, &[1.0, 2.0]);
        let right = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0]);
        assert!(matches!(
            domain.bind(AddOperation, Vec::new(), &[left, right]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "'add' input types are not broadcast-compatible",
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
