use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use ryft_pjrt::protos::CompilationOptions;
use ryft_pjrt::{Client, Error, LoadedExecutable, Program};

/// Thin wrapper around a PJRT [`Client`] that adds a process-local cache of compiled
/// [`LoadedExecutable`]s plus a customizable base [`CompilationOptions`] template.
///
/// Construct one [`CompilationContext`] per `Client` at program start and reuse it across calls
/// to [`Array::to_placement`](crate::Array::to_placement),
/// [`Array::to_device`](crate::Array::to_device), and [`device_put`](crate::arrays_v0::device_put).
/// The cache stores `Arc<LoadedExecutable<'c>>` so repeated compilations of the same MLIR text
/// and compile options hand back the previously compiled executable without paying the PJRT
/// compile cost again.
///
/// The cache key is a 64-bit hash of the MLIR bytecode mixed with the `Debug` representation of
/// the [`CompilationOptions`], so different option sets get independent cache entries.
pub struct CompilationContext<'c> {
    /// PJRT client wrapped by this context.
    client: &'c Client<'c>,

    /// Base [`CompilationOptions`] template. Callers that want non-default options (e.g. a
    /// specific matrix-unit precision) construct the context via
    /// [`CompilationContext::with_options`]. Reshard callers overlay mesh-derived
    /// `partition_count` / SPMD flags on top of this template before compiling.
    base_options: CompilationOptions,

    /// Compile-cache mapping `(mlir hash, options hash)` to its cached [`LoadedExecutable`].
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
    /// the same MLIR text reuse the cached executable instead of recompiling.
    #[inline]
    pub fn cache_size(&self) -> usize {
        self.executables.lock().expect("compile cache mutex should not be poisoned").len()
    }

    /// Compiles `mlir_text` against `options` if the combination is not already cached, otherwise
    /// returns the cached executable.
    ///
    /// The cache key is a 64-bit hash of `(mlir_text bytes, format!("{:?}", options) bytes)`.
    /// Concurrent compilations of distinct keys are serialized by the internal mutex.
    pub(crate) fn compile(
        &self,
        mlir_text: &str,
        options: &CompilationOptions,
    ) -> Result<Arc<LoadedExecutable<'c>>, Error> {
        let key = hash_key(mlir_text, options);
        let mut cache = self.executables.lock().expect("compile cache mutex should not be poisoned");
        if let Some(executable) = cache.get(&key) {
            return Ok(executable.clone());
        }
        let program = Program::Mlir { bytecode: mlir_text.as_bytes().to_vec() };
        let executable = Arc::new(self.client.compile(&program, options)?);
        cache.insert(key, executable.clone());
        Ok(executable)
    }
}

fn hash_key(mlir_text: &str, options: &CompilationOptions) -> u64 {
    let mut hasher = DefaultHasher::new();
    mlir_text.as_bytes().hash(&mut hasher);
    // `CompilationOptions` is a `prost::Message` and does not derive `Hash`. Its `Debug` impl is
    // stable enough for cache-key purposes and avoids pulling `prost` into ryft-xla.
    format!("{options:?}").as_bytes().hash(&mut hasher);
    hasher.finish()
}
