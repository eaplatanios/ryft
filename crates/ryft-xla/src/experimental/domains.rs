use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock};

use ryft_pjrt::protos::CompilationOptions;
use ryft_pjrt::{Buffer, Client, LoadedExecutable, Program as PjrtProgram};

use ryft_core::compilation::{CompilationContext, CompilationDomain, FunctionFingerprint};
use ryft_core::contexts::{Context, Domain};
use ryft_core::differentiation::DifferentiationError;
use ryft_core::interpretation::InterpretableOperation;
use ryft_core::macros::check_count;
use ryft_core::operations::Operation;
use ryft_core::operations::arithmetic::{AddOperation, MulOperation};
use ryft_core::operations::compare::{CompareOperation, ComparisonDirection};
use ryft_core::operations::constants::{
    Constant, Fill, FillOperation, Iota, IotaOperation, ONE_OPERATION_NAME, One, OneOperation, ZERO_OPERATION_NAME,
    Zero, ZeroOperation,
};
use ryft_core::operations::control_flow::SelectOperation;
use ryft_core::parameters::{Parameterized, Placeholder};
use ryft_core::programs::ProgramError;
use ryft_core::scalars::Scalar;
use ryft_core::sharding::{Device, DeviceId, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, Sharding};
use ryft_core::tracing::DomainTracer;
use ryft_core::tracing_v2::CoordinateBasis;
use ryft_core::types::{ArrayType, DataType, Shape, TypeError, Typed};

use super::operations::ShardMapOperation;
use super::ops::{FlatXlaProgram, JitCallOperation, XlaConstant, XlaOperation, XlaProgram, XlaProgramBuilder};
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
/// - a PJRT [`Client`] used to upload `zero`/`one` shards and to compile and execute programs (including the
///   per-operation programs behind eager [`Context::bind`] dispatch),
/// - an optional concrete [`DeviceMesh`] that eager binds prefer when deriving their execution mesh and that the
///   constant-materialization fast path requires,
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

    /// Concrete device mesh that eager binds prefer when deriving their execution mesh and that the
    /// constant-materialization fast path requires.
    mesh: Option<DeviceMesh>,

    /// Default compilation options forwarded to [`Client::compile`].
    compilation_options: CompilationOptions,

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
            compilation_options: CompilationOptions::default(),
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
            compilation_options: CompilationOptions::default(),
            cache: Arc::new(CompilationContext::new()),
            marker: PhantomData,
        }
    }

    /// Returns the PJRT [`Client`] this domain was constructed with.
    #[inline]
    pub fn client(&self) -> &'c Client<'c> {
        self.client.expect("execution XlaDomain should always carry a client")
    }

    /// Returns the [`DeviceMesh`] this domain resolves shard placement against. Panics when the domain was
    /// constructed without a mesh; eager binds derive their mesh from the inputs in that case.
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
    /// [`XlaConstant`] is a [`CaptureReference`](ryft_core::compilation::CaptureReference) — a symbolic index into a
    /// compiled function's capture table carrying only a type and no data — so there is nothing to materialize
    /// without the surrounding capture table and lifting is always rejected.
    fn lift(&self, constant: XlaConstant) -> Result<Array<'c>, ProgramError> {
        Err(TypeError {
            message: format!("xla captured constant {constant} requires a captured program capture table"),
        }
        .into())
    }

    /// Eagerly executes `operation` on concrete input [`Array`]s, mirroring JAX's op-by-op dispatch: the operation
    /// is traced into a single-instruction program over the inputs' physical [`ArrayType`]s (shardings included),
    /// compiled through this domain's compile cache, and executed on this domain's PJRT client via
    /// [`Self::eager_bind`]. The nullary additive/multiplicative identities keep a fast path that materializes the
    /// constant directly through the runtime client without compiling a program.
    fn bind<P: Into<Self::Operation>>(
        &self,
        operation: P,
        inputs: &[Self::Value],
    ) -> Result<Vec<Self::Value>, ProgramError> {
        let operation = operation.into();
        let name = operation.name();
        if inputs.is_empty() && (name == ZERO_OPERATION_NAME || name == ONE_OPERATION_NAME) {
            let array_type = eager_identity_output_type(&operation)?;
            validate_identity_synthesis(name, &array_type)?;
            // The direct constant-materialization fast path needs a concrete device mesh; mesh-less domains fall
            // through to the compiled eager path below, which derives a default execution mesh instead.
            if self.mesh.is_some() {
                let kind = if name == ZERO_OPERATION_NAME { ConstantKind::Zero } else { ConstantKind::One };
                let value =
                    self.constant(&array_type, kind).map_err(|error| TypeError { message: error.to_string() })?;
                return Ok(vec![value]);
            }
        }
        self.eager_bind(operation, inputs)
    }

    /// A client-backed domain executes every bound operation for real and its concrete [`Array`]s support host
    /// readback through [`BooleanLike`](ryft_core::operations::BooleanLike) and
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
        let mut outputs = self.bind(ZeroOperation::new(r#type.clone()), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Refer to the documentation of this domain's [`Zero`] implementation for more information.
impl<'c> One<Array<'c>> for XlaDomain<'c> {
    fn one(&self, r#type: &ArrayType) -> Result<Array<'c>, ProgramError> {
        let mut outputs = self.bind(OneOperation::new(r#type.clone()), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Refer to the documentation of this domain's [`Zero`] implementation for more information.
impl<'c> Fill<Scalar, Array<'c>> for XlaDomain<'c> {
    fn fill(&self, r#type: &ArrayType, value: Scalar) -> Result<Array<'c>, ProgramError> {
        let mut outputs = self.bind(FillOperation::new(r#type.clone(), value), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Refer to the documentation of this domain's [`Zero`] implementation for more information.
impl<'c> Iota<Array<'c>> for XlaDomain<'c> {
    fn iota(&self, r#type: &ArrayType, dimension: usize) -> Result<Array<'c>, ProgramError> {
        let mut outputs = self.bind(IotaOperation::new(r#type.clone(), dimension), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Synthesizes one packed standard-basis leaf entirely through a single compiled XLA program. The program constructs
/// the global row-major coordinate of every leaf element from integer iotas, compares it with the leading basis-row
/// iota plus `coordinate_offset`, and selects a typed one or zero. XLA can fuse that graph into one device kernel;
/// no one-hot buffers are built on the host and no derivative payload is copied back to the host.
impl<'c> CoordinateBasis<Array<'c>> for XlaDomain<'c> {
    fn coordinate_basis(
        &self,
        leaf_type: &ArrayType,
        coordinate_offset: usize,
        basis_size: usize,
    ) -> Result<Array<'c>, ProgramError> {
        let Some(client) = self.client else {
            return Err(ProgramError::InvalidArgument {
                message: "xla domain cannot synthesize a coordinate basis without a PJRT client".into(),
            });
        };
        if leaf_type.data_type() == DataType::Token {
            return Err(TypeError { message: "coordinate basis does not support token arrays".into() }.into());
        }
        let leaf_dimensions = leaf_type
            .shape()
            .dimensions()
            .iter()
            .map(|size| match size {
                ryft_core::types::Size::Static(size) => Ok(*size),
                ryft_core::types::Size::Dynamic(_) => Err(TypeError {
                    message: format!("coordinate basis requires a fully static leaf type but got {leaf_type}"),
                }),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let basis_type = leaf_type.with_inserted_dimension(0, ryft_core::types::Size::Static(basis_size))?;
        let index_type = basis_type.clone().with_data_type(DataType::U64);
        let offset = u64::try_from(coordinate_offset).map_err(|_| ProgramError::InvalidArgument {
            message: format!("coordinate offset {coordinate_offset} does not fit in u64"),
        })?;

        let mut builder = XlaProgramBuilder::new();
        let basis_index = builder.add_instruction(IotaOperation::new(index_type.clone(), 0), Vec::new())?[0];

        // Compute each leaf element's row-major flat coordinate in the same physical `[basis] ++ leaf_shape` tensor.
        // All arithmetic stays in u64 so large coordinate spaces retain exact indices.
        let mut flat_coordinate = None;
        let mut stride = 1u64;
        for (leaf_axis, &dimension_size) in leaf_dimensions.iter().enumerate().rev() {
            let coordinate =
                builder.add_instruction(IotaOperation::new(index_type.clone(), leaf_axis + 1), Vec::new())?[0];
            let coordinate = if stride == 1 {
                coordinate
            } else {
                let stride_value = builder
                    .add_instruction(FillOperation::new(index_type.clone(), Scalar::U64(stride)), Vec::new())?[0];
                builder.add_instruction(MulOperation, vec![coordinate, stride_value])?[0]
            };
            flat_coordinate = Some(match flat_coordinate {
                Some(accumulated) => builder.add_instruction(AddOperation, vec![accumulated, coordinate])?[0],
                None => coordinate,
            });
            let dimension_size = u64::try_from(dimension_size).map_err(|_| ProgramError::InvalidArgument {
                message: format!("leaf dimension {dimension_size} does not fit in u64"),
            })?;
            stride = stride.checked_mul(dimension_size).ok_or_else(|| ProgramError::InvalidArgument {
                message: format!("coordinate count overflows u64 for leaf type {leaf_type}"),
            })?;
        }
        let mut flat_coordinate = match flat_coordinate {
            Some(flat_coordinate) => flat_coordinate,
            None => builder.add_instruction(FillOperation::new(index_type.clone(), Scalar::U64(0)), Vec::new())?[0],
        };
        if offset != 0 {
            let offset_value =
                builder.add_instruction(FillOperation::new(index_type.clone(), Scalar::U64(offset)), Vec::new())?[0];
            flat_coordinate = builder.add_instruction(AddOperation, vec![flat_coordinate, offset_value])?[0];
        }

        let selected = builder
            .add_instruction(CompareOperation::new(ComparisonDirection::Equal), vec![basis_index, flat_coordinate])?[0];
        let one = builder.add_instruction(OneOperation::new(basis_type.clone()), Vec::new())?[0];
        let zero = builder.add_instruction(ZeroOperation::new(basis_type.clone()), Vec::new())?[0];
        let output = builder.add_instruction(SelectOperation, vec![selected, one, zero])?[0];
        let program: FlatXlaProgram = builder.build(vec![output], Vec::new(), vec![Placeholder])?;

        // Cache by the complete basis declaration; execution has no runtime inputs. The output type carries the
        // inserted replicated basis axis and the leaf's original sharding on its remaining axes.
        let mut hasher = DefaultHasher::new();
        leaf_type.hash(&mut hasher);
        coordinate_offset.hash(&mut hasher);
        basis_size.hash(&mut hasher);
        let fingerprint = FunctionFingerprint::Composite {
            base: Box::new(FunctionFingerprint::Primitive("ryft.coordinate_basis")),
            extra: hasher.finish(),
        };
        let mesh = self.eager_mesh(client, &[], program.output_types().as_slice())?;
        let options = XlaOptions::new(mesh);
        let cache_key = self.compilation_key(&fingerprint, &[], &options);
        let compiled = self
            .cache
            .get_or_compile(self, cache_key, || self.compile(&program, &options))
            .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
        let mut outputs = self
            .execute(&compiled, Vec::new())
            .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0).with_compilation_cache(Arc::clone(&self.cache)))
    }
}

/// [`XlaConstant`] is a [`CaptureReference`](ryft_core::compilation::CaptureReference) carrying only a type and no
/// data, so — exactly like [`Context::lift`], to which this delegates — captured-constant materialization is always
/// rejected outside a surrounding capture table. The implementation exists because interpretation- and
/// batching-capable operation families require a [`Constant`] leaf on their contexts; programs whose constants were
/// compiled into capture tables never take this path.
impl<'c> Constant<Array<'c>, XlaConstant> for XlaDomain<'c> {
    fn constant(&self, value: XlaConstant) -> Result<Array<'c>, ProgramError> {
        self.lift(value)
    }
}

/// Eager interpretation of a staged jitted call over concrete [`Array`]s: the call is bound whole through
/// [`Context::bind`] — compiled through this domain's dispatch cache and executed on its PJRT client — mirroring JAX
/// dispatching a jitted function called from eager code straight to the compiled executable. This is what lets flat
/// programs containing `jit_call` instructions (for example the pullbacks produced by eager `vjp`/`grad`) replay
/// through a client-backed [`XlaDomain`].
impl<'c> InterpretableOperation<Array<'c>, XlaDomain<'c>> for JitCallOperation {
    fn interpret(&self, context: &XlaDomain<'c>, inputs: &[Array<'c>]) -> Result<Vec<Array<'c>>, ProgramError> {
        context.bind(self.clone(), inputs)
    }
}

/// Eager interpretation of a captured-body shard map over concrete [`Array`]s: the operation is bound whole through
/// [`Context::bind`], so the manual sharding region is SPMD-compiled and executed over the inputs' mesh instead of
/// being inlined. Refer to the documentation of the [`JitCallOperation`] implementation above for how this powers
/// eager program replay.
impl<'c> InterpretableOperation<Array<'c>, XlaDomain<'c>> for ShardMapOperation<XlaConstant> {
    fn interpret(&self, context: &XlaDomain<'c>, inputs: &[Array<'c>]) -> Result<Vec<Array<'c>>, ProgramError> {
        context.bind(self.clone(), inputs)
    }
}

/// Returns the single [`ArrayType`] produced by a nullary additive/multiplicative identity operation
/// ([`ZERO_OPERATION_NAME`] / [`ONE_OPERATION_NAME`]). The identity fast path in [`Context::bind`] materializes these
/// constants directly through the runtime client instead of compiling a program.
fn eager_identity_output_type<O: Operation<ArrayType>>(operation: &O) -> Result<ArrayType, ProgramError> {
    let mut output_types = operation.infer_output_types(&[])?;
    if output_types.len() != 1 {
        return Err(TypeError {
            message: format!(
                "xla identity operation `{}` must produce exactly one output but produced {}",
                operation.name(),
                output_types.len(),
            ),
        }
        .into());
    }
    Ok(output_types.pop().expect("output count checked above"))
}

/// Returns the process-local compile-cache fingerprint for one eagerly bound operation.
///
/// The fingerprint hashes the operation's `Debug` rendering because derived `Debug` output includes every semantic
/// field — nested `condition` / `while` / `scan` bodies, `jit_call` callee programs, and literal payloads — whereas
/// the canonical rendered form summarizes call-like payloads by arity only, which would alias distinct callees. Input
/// types (with shardings), mesh, and compilation options are carried by the rest of the compilation key, so this
/// fingerprint only needs to identify the operation itself.
fn eager_operation_fingerprint(operation: &XlaOperation) -> FunctionFingerprint {
    let mut hasher = DefaultHasher::new();
    format!("{operation:?}").hash(&mut hasher);
    FunctionFingerprint::Composite {
        base: Box::new(FunctionFingerprint::Primitive("ryft.eager_bind")),
        extra: hasher.finish(),
    }
}

fn validate_identity_synthesis(identity: &'static str, array_type: &ArrayType) -> Result<(), ProgramError> {
    match array_type.data_type() {
        DataType::Token => Err(TypeError {
            message: format!(
                "xla domain cannot synthesize {identity} value for element type {}",
                array_type.data_type()
            ),
        }
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
    /// The cache key combines the operation's structural fingerprint with the input types and the derived mesh, so
    /// repeated eager binds of the same operation signature reuse one compiled executable. Higher-order operations
    /// (`condition` / `while` / `scan` / `jit_call` / `shard_map`) carry their nested programs as payloads and flow
    /// through this same path — the compiler handles the control flow, so no host interpreter loops are needed.
    fn eager_bind(&self, operation: XlaOperation, inputs: &[Array<'c>]) -> Result<Vec<Array<'c>>, ProgramError> {
        let Some(client) = self.client else {
            return Err(ProgramError::InvalidArgument {
                message: format!(
                    "xla domain cannot eagerly execute operation `{}` without a PJRT client",
                    operation.name(),
                ),
            });
        };
        self.validate_eager_placement(client, inputs)?;

        // Trace the single-instruction program over the inputs' physical types, shardings included.
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let fingerprint = eager_operation_fingerprint(&operation);
        let mut builder = XlaProgramBuilder::new();
        let input_atoms = input_types.iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let output_atoms = builder.add_instruction(operation, input_atoms)?.to_vec();
        let output_count = output_atoms.len();
        let program: FlatXlaProgram =
            builder.build(output_atoms, vec![Placeholder; input_types.len()], vec![Placeholder; output_count])?;

        // Derive the mesh after tracing so that input-free operations can fall back to their inferred output
        // shardings, then compile through the domain's cache (a repeated eager operation is a cache hit) and
        // execute via PJRT.
        let mesh = self.eager_mesh(client, inputs, program.output_types().as_slice())?;
        let options = XlaOptions::new(mesh);
        let cache_key = self.compilation_key(&fingerprint, input_types.as_slice(), &options);
        let compiled = self
            .cache
            .get_or_compile(self, cache_key, || self.compile(&program, &options))
            .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
        let outputs = self
            .execute(&compiled, inputs.to_vec())
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
        Ok(Array::from_addressable_buffers(client, effective_type, mesh.clone(), addressable_buffers)?
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
            self.client,
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
/// buffers back into distributed [`Array`] values carrying `client` (when one is provided), so that eager execution
/// and free transforms over the outputs can recover their execution domain. Mirrors
/// `XlaDomain::execute_with_donation`.
fn execute_pjrt<'c>(
    client: Option<&'c Client<'c>>,
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
        outputs.push(Array::from_addressable_buffers(client, resolved_type, mesh.clone(), addressable_buffers)?);
    }
    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::Sharding;
    use ryft_core::operations::arithmetic::{AddOperation, MulOperation, NegOperation};
    use ryft_core::operations::compare::{CompareOperation, ComparisonDirection};
    use ryft_core::operations::constants::{FillOperation, OneOperation};
    use ryft_core::operations::control_flow::{ConditionOperation, WhileOperation};
    use ryft_core::sharding::ShardingDimension;
    use ryft_core::types::{Size, StaticShape};
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
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(size)]))
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
        use ryft_core::operations::constants::OneOperation;
        let array_type = ArrayType::scalar(DataType::Token);

        assert!(matches!(
            XlaDomain::token().bind(OneOperation::new(array_type.clone()), &[]),
            Err(ProgramError::Type(error))
                if error.message == "xla domain cannot synthesize one value for element type token"
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
            compile_with_options(&engine, |x| x.sin().unwrap(), input_type.clone(), options).unwrap();

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
                    |x| x.sin().unwrap(),
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

    #[test]
    fn test_eager_bind_executes_binary_operation() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);

        let left = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0, 4.0]);
        let right = f32_vector(&client, &mesh, &[10.0, 20.0, 30.0, 40.0]);
        let outputs = domain.bind(AddOperation, &[left, right]).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(read_f32s(&client, &outputs[0]), vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_eager_bind_executes_unary_operation() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_domain_mesh(&client, "x", 1);
        let domain = XlaDomain::new(&client);

        let input = f32_vector(&client, &mesh, &[1.0, -2.0, 3.5, 0.0]);
        let outputs = domain.bind(NegOperation, &[input]).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(read_f32s(&client, &outputs[0]), vec![-1.0, 2.0, -3.5, 0.0]);
    }

    #[test]
    fn test_eager_bind_materializes_nullary_fill_over_a_default_mesh() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let domain = XlaDomain::new(&client);

        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3)]));
        let outputs = domain.bind(FillOperation::new(r#type, Scalar::from(2.5f64)), &[]).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].shape(), StaticShape::new(vec![3]));
        assert_eq!(read_f32s(&client, &outputs[0]), vec![2.5, 2.5, 2.5]);

        // A complex fill value lowers as two real part splats composed through `stablehlo.complex`, and a `c64`
        // buffer's bytes are the interleaved `f32` real and imaginary parts of its elements.
        let r#type = ArrayType::new(DataType::C64, Shape::new(vec![Size::Static(2)]));
        let value = Scalar::from(num_complex::Complex::new(1.5f32, -2.0f32));
        let outputs = domain.bind(FillOperation::new(r#type, value), &[]).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].shape(), StaticShape::new(vec![2]));
        assert_eq!(read_f32s(&client, &outputs[0]), vec![1.5, -2.0, 1.5, -2.0]);
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
        let first = domain.bind(AddOperation, &[left.clone(), right.clone()]).unwrap();
        assert_eq!(domain.cache_size(), 1);

        let second = domain.bind(AddOperation, &[left, right]).unwrap();
        assert_eq!(domain.cache_size(), 1, "a repeated eager operation must be a compile-cache hit");
        assert_eq!(read_f32s(&client, &first[0]), read_f32s(&client, &second[0]));

        // A different input signature compiles (and caches) a distinct executable.
        let wider_left = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0]);
        let wider_right = f32_vector(&client, &mesh, &[4.0, 5.0, 6.0]);
        domain.bind(AddOperation, &[wider_left, wider_right]).unwrap();
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
            domain.bind(NegOperation, &[input.clone()]),
            Err(ProgramError::InvalidArgument { message })
                if message == "received incompatible devices for eager xla execution: input #0 is owned by a \
                    different PJRT client than this domain's client",
        ));

        // An input with no attached client falls back to the device-set membership check.
        let mut clientless_input = input;
        clientless_input.detach_client_for_tests();
        assert!(matches!(
            domain.bind(NegOperation, &[clientless_input]),
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
            let output = builder.add_instruction(AddOperation, vec![input, input]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let squared = {
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(vector_type.clone());
            let output = builder.add_instruction(MulOperation, vec![input, input]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let operation = XlaOperation::Condition(Box::new(ConditionOperation::new(doubled, squared).unwrap()));

        let input = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0, 4.0]);
        let true_outputs =
            domain.bind(operation.clone(), &[boolean_scalar(&client, &mesh, true), input.clone()]).unwrap();
        assert_eq!(read_f32s(&client, &true_outputs[0]), vec![2.0, 4.0, 6.0, 8.0]);

        let false_outputs = domain.bind(operation, &[boolean_scalar(&client, &mesh, false), input]).unwrap();
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
            let limit = builder
                .add_instruction(FillOperation::new(scalar_type.clone(), Scalar::from(3.0f64)), vec![])
                .unwrap()[0];
            let predicate = builder
                .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), vec![state, limit])
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
            let one = builder.add_instruction(OneOperation::new(scalar_type.clone()), vec![]).unwrap()[0];
            let next = builder.add_instruction(AddOperation, vec![state, one]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![next], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let operation = XlaOperation::While(Box::new(WhileOperation::new(condition, body).unwrap()));

        let outputs = domain.bind(operation, &[f32_scalar(&client, &mesh, 0.0)]).unwrap();
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
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]))
            .with_sharding(sharding.clone())
            .unwrap();
        let input = Array::from_host_buffer(
            &client,
            input_type,
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let outputs = domain.bind(AddOperation, &[input.clone(), input]).unwrap();

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
            let one = builder.add_instruction(OneOperation::new(scalar_type.clone()), vec![]).unwrap()[0];
            let next = builder.add_instruction(AddOperation, vec![carry, one]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![next, next],
                    vec![Placeholder; 1],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let scan = ScanOperation::<XlaConstant, XlaOperation>::new(body, 1, 4).unwrap();

        let outputs = domain.bind(XlaOperation::Scan(Box::new(scan)), &[f32_scalar(&client, &mesh, 0.0)]).unwrap();
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
            let sum = builder.add_instruction(AddOperation, vec![carry, x]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![sum, sum], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let scan = ScanOperation::<XlaConstant, XlaOperation>::new(body, 1, 4).unwrap();

        let carry = f32_scalar(&client, &mesh, 0.0);
        let xs = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0, 4.0]);
        let outputs = domain.bind(XlaOperation::Scan(Box::new(scan)), &[carry, xs]).unwrap();
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
            let sum = builder.add_instruction(AddOperation, vec![carry, x]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![sum, sum], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let scan = ScanOperation::<XlaConstant, XlaOperation>::new(body, 1, 4).unwrap();

        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let xs_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)])).with_sharding(sharding).unwrap();
        let xs = Array::from_host_buffer(
            &client,
            xs_type,
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let carry = f32_scalar(&client, &mesh, 0.0);
        let outputs = domain.bind(XlaOperation::Scan(Box::new(scan)), &[carry, xs]).unwrap();
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
            let output = builder.add_instruction(MulOperation, vec![input, input]).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };
        let operation = XlaOperation::JitCall(Box::new(JitCallOperation::new(Rc::new(callee))));

        let input = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0, 4.0]);
        let first = domain.bind(operation.clone(), &[input.clone()]).unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(read_f32s(&client, &first[0]), vec![1.0, 4.0, 9.0, 16.0]);
        assert_eq!(domain.cache_size(), 1);

        // A repeated eager `jit_call` at the same input signature is a dispatch-cache hit.
        let second = domain.bind(operation, &[input]).unwrap();
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
        let global_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]));
        let local_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)]));
        let body_program = {
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(local_type.clone());
            let output = builder.add_instruction(AddOperation, vec![input, input]).unwrap()[0];
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
        let operation = XlaOperation::ShardMap(Box::new(ShardMapOperation::new(body)));

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]))
            .with_sharding(sharding.clone())
            .unwrap();
        let input = Array::from_host_buffer(
            &client,
            input_type,
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let outputs = domain.bind(operation, &[input]).unwrap();

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
        use ryft_core::tracing_v2::operations::{CollectiveKind, CollectiveOperation};

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
            domain.bind(CollectiveOperation::new("i".to_string(), CollectiveKind::PSum), &[input]),
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
        let r#type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]))
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let input =
            Array::from_host_buffer(&client, r#type, mesh.clone(), values_to_bytes::<f64>(&[1.5, 2.5]).as_slice())
                .unwrap();
        let (outputs, lines) = with_captured_prints(|| domain.bind(PrintOperation::new("x"), &[input]).unwrap());

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
            domain.bind(AddOperation, &[left, right]),
            Err(ProgramError::Type(error)) if error.message == "'add' input types are not broadcast-compatible",
        ));
    }
}
