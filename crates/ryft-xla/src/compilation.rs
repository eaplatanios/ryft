use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::panic::Location;
use std::sync::{Arc, Mutex};

use ryft_core::sharding::DeviceMesh;
use ryft_core::types::ArrayType;
use ryft_pjrt::protos::CompilationOptions;
use ryft_pjrt::{Client, LoadedExecutable, Program};

/// Thin wrapper around a PJRT [`Client`] that adds a process-local cache of compiled
/// [`LoadedExecutable`]s plus a customizable base [`CompilationOptions`] template.
///
/// Construct one [`CompilationContext`] per `Client` at program start and reuse it across calls
/// to [`Array::to_placement`](crate::Array::to_placement),
/// [`Array::to_device`](crate::Array::to_device), and [`device_put`](crate::arrays_v0::device_put).
///
/// The cache is keyed by a **structural signature** the caller provides up front — for example,
/// `(input ArrayType, src Sharding, dst Sharding, DeviceMesh)` for a reshard call — mixed with a
/// stable hash of the [`CompilationOptions`]. Repeat calls that hash to the same key short-circuit
/// without running the trace + lower work that would otherwise produce the MLIR text; only
/// cache-miss calls invoke the supplied closure to materialize MLIR and feed it to
/// [`Client::compile`].
///
/// This mirrors how JAX's compile cache is keyed on abstract value signatures: on a warm call,
/// neither tracing nor lowering runs.
pub struct CompilationContext<'c> {
    /// PJRT client wrapped by this context.
    client: &'c Client<'c>,

    /// Base [`CompilationOptions`] template. Callers that want non-default options (e.g. a
    /// specific matrix-unit precision) construct the context via
    /// [`CompilationContext::with_options`]. Reshard callers overlay mesh-derived
    /// `partition_count` / SPMD flags on top of this template before compiling.
    base_options: CompilationOptions,

    /// Compile-cache keyed by `(structural-signature hash, options-debug hash)`.
    executables: Mutex<HashMap<u64, Arc<LoadedExecutable<'c>>>>,
}

impl<'c> CompilationContext<'c> {
    /// Creates a [`CompilationContext`] wrapping the provided PJRT [`Client`] with the default
    /// [`CompilationOptions`] template.
    #[inline]
    pub fn new(client: &'c Client<'c>) -> Self {
        Self::with_options(client, CompilationOptions::default())
    }

    /// Creates a [`CompilationContext`] with an explicit [`CompilationOptions`] template.
    ///
    /// Reshard callers can override compilation-time knobs (e.g. matrix-unit operand precision,
    /// custom environment options) by constructing the context with the desired
    /// [`CompilationOptions`]; the reshard machinery then overlays the mesh-derived SPMD fields
    /// on top of this template per call.
    #[inline]
    pub fn with_options(client: &'c Client<'c>, options: CompilationOptions) -> Self {
        Self { client, base_options: options, executables: Mutex::new(HashMap::new()) }
    }

    /// Returns the PJRT [`Client`] wrapped by this context.
    #[inline]
    pub fn client(&self) -> &'c Client<'c> {
        self.client
    }

    /// Returns the base [`CompilationOptions`] template this context was constructed with.
    #[inline]
    pub fn base_options(&self) -> &CompilationOptions {
        &self.base_options
    }

    /// Returns the number of compiled [`LoadedExecutable`]s currently cached.
    ///
    /// Mostly useful for telemetry and tests that need to confirm that repeated compilations of
    /// the same structural signature reuse the cached executable instead of recompiling.
    #[inline]
    pub fn cache_size(&self) -> usize {
        self.executables.lock().expect("compile cache mutex should not be poisoned").len()
    }

    /// Returns a cached executable for `key` if present, otherwise invokes `produce_mlir` to
    /// materialize the MLIR text, compiles it via PJRT, caches the result under `(key, options)`,
    /// and returns it.
    ///
    /// `key` is the structural cache signature — typically a tuple of input/output types, sharding
    /// annotations, mesh device order, and any other data that uniquely identifies the MLIR text
    /// that `produce_mlir` would emit. The closure runs only on cache miss, so repeat calls with
    /// the same `(key, options)` pay no tracing / lowering cost.
    ///
    /// # Parameters
    ///
    ///   - `key`: Structural signature. Must implement [`Hash`].
    ///   - `options`: Compile options. Mixed into the cache key via their `Debug` representation.
    ///   - `produce_mlir`: Closure that materializes the MLIR text on cache miss.
    pub(crate) fn get_or_compile<K, F, E>(
        &self,
        key: &K,
        options: &CompilationOptions,
        produce_mlir: F,
    ) -> Result<Arc<LoadedExecutable<'c>>, E>
    where
        K: Hash,
        F: FnOnce() -> Result<String, E>,
        E: From<ryft_pjrt::Error>,
    {
        let cache_key = hash_signature(key, options);
        {
            let cache = self.executables.lock().expect("compile cache mutex should not be poisoned");
            if let Some(executable) = cache.get(&cache_key) {
                return Ok(executable.clone());
            }
        }
        let mlir_text = produce_mlir()?;
        let program = Program::Mlir { bytecode: mlir_text.into_bytes() };
        let executable = Arc::new(self.client.compile(&program, options)?);
        let mut cache = self.executables.lock().expect("compile cache mutex should not be poisoned");
        Ok(cache.entry(cache_key).or_insert(executable).clone())
    }
}

fn hash_signature<K: Hash>(key: &K, options: &CompilationOptions) -> u64 {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    // `CompilationOptions` is a `prost::Message` and does not derive `Hash`. Its `Debug` impl is
    // stable enough for cache-key purposes and avoids pulling `prost` into ryft-xla.
    format!("{options:?}").hash(&mut hasher);
    hasher.finish()
}

/// Identifier for the function or primitive being compiled.
///
/// Two compilations with the same [`FunctionFingerprint`], the same [`CompilationKey::input_types`],
/// and the same destination mesh produce the same executable and share a cache entry. This
/// mirrors how JAX's compile cache combines a function fingerprint with abstract input value
/// signatures.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum FunctionFingerprint {
    /// A `ryft` built-in primitive identified by a static name. Reserved for internal use cases
    /// like the compiled-reshard path.
    Primitive(&'static str),

    /// A user function identified by the source location of its outer entry point (for example
    /// the call site of a future `jit`). Construct via [`FunctionFingerprint::from_caller`].
    ///
    /// JAX uses the Python function's identity plus closure-captured cells as the fingerprint;
    /// Rust's closures don't expose an equivalent stable identity, so `ryft` uses the call-site
    /// location as a best-effort proxy. Callers that capture state in their closures should
    /// embed the captured values themselves into the fingerprint (see
    /// [`FunctionFingerprint::Composite`]).
    SourceLocation { file: &'static str, line: u32, column: u32 },

    /// A composite fingerprint: a base fingerprint mixed with an opaque 64-bit hash of any
    /// additional state (e.g. captured constants) that uniquely identifies a function instance.
    /// Use this when the call site alone is not enough to distinguish two logical functions.
    Composite { base: Box<FunctionFingerprint>, extra: u64 },
}

impl FunctionFingerprint {
    /// Constructs a [`FunctionFingerprint::SourceLocation`] from the call site that invokes this
    /// function. The call-site location is captured at compile time via `#[track_caller]` and is
    /// stable as long as the call site doesn't move.
    #[inline]
    #[track_caller]
    pub fn from_caller() -> Self {
        let location = Location::caller();
        Self::SourceLocation { file: location.file(), line: location.line(), column: location.column() }
    }
}

/// Generic cache key for the compiled-executable cache.
///
/// Captures the four pieces of information that, together with the [`CompilationOptions`], make
/// one compilation distinct from another:
///
///   1. [`Self::fingerprint`] — what function is being compiled.
///   2. [`Self::input_types`] — abstract input value signatures (shape, dtype, sharding).
///   3. [`Self::output_types`] — abstract output value signatures.
///   4. [`Self::mesh`] — concrete device topology the program will run on.
///
/// The [`compiled_reshard`](crate::arrays_v0::compiled_reshard) module uses this key with
/// [`FunctionFingerprint::Primitive("compiled_reshard.identity")`](FunctionFingerprint::Primitive).
/// A future user-facing `jit` would use [`FunctionFingerprint::SourceLocation`] (or
/// [`FunctionFingerprint::from_caller`]) and the same `CompilationKey` shape.
pub struct CompilationKey<'a> {
    /// Identifier for the function/primitive being compiled.
    pub fingerprint: FunctionFingerprint,

    /// Abstract input value types — shape, dtype, and sharding metadata for each program input.
    pub input_types: &'a [ArrayType],

    /// Abstract output value types.
    pub output_types: &'a [ArrayType],

    /// Concrete device topology. Different device orderings of the same logical mesh produce
    /// different executables, so the full mesh is part of the key.
    pub mesh: &'a DeviceMesh,
}

impl<'a> Hash for CompilationKey<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.fingerprint.hash(state);
        self.input_types.hash(state);
        self.output_types.hash(state);
        // `DeviceMesh` does not derive `Hash`; hash its (logical mesh, device order) tuple
        // manually instead. `LogicalMesh` and `Device` both implement `Hash`.
        self.mesh.logical_mesh().hash(state);
        self.mesh.devices().hash(state);
    }
}
