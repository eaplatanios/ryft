use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use ryft_pjrt::protos::CompilationOptions;
use ryft_pjrt::{Client, Error, LoadedExecutable, Program};

/// Thin wrapper around a PJRT [`Client`] that adds a process-local cache of compiled
/// [`LoadedExecutable`]s.
///
/// Construct one [`CompilationContext`] per `Client` at program start and reuse it across calls
/// to [`Array::to_placement`](crate::Array::to_placement),
/// [`Array::to_device`](crate::Array::to_device), and [`device_put`](crate::arrays_v0::device_put).
/// The cache stores `Arc<LoadedExecutable<'c>>` so repeated compilations of the same MLIR text
/// hand back the previously compiled executable without paying the PJRT compile cost again.
///
/// The cache is keyed by MLIR bytecode; [`CompilationOptions`] are intentionally not part of the
/// key, so a single [`CompilationContext`] should be used with one consistent set of options. If
/// callers need to compile the same MLIR with different options, they should construct separate
/// contexts.
pub struct CompilationContext<'c> {
    /// PJRT client wrapped by this context.
    client: &'c Client<'c>,

    /// Compile-cache mapping MLIR bytecode hash to its cached [`LoadedExecutable`].
    executables: Mutex<HashMap<u64, Arc<LoadedExecutable<'c>>>>,
}

impl<'c> CompilationContext<'c> {
    /// Creates a [`CompilationContext`] wrapping the provided PJRT [`Client`].
    #[inline]
    pub fn new(client: &'c Client<'c>) -> Self {
        Self { client, executables: Mutex::new(HashMap::new()) }
    }

    /// Returns the PJRT [`Client`] wrapped by this context.
    #[inline]
    pub fn client(&self) -> &'c Client<'c> {
        self.client
    }

    /// Returns the number of compiled [`LoadedExecutable`]s currently cached.
    ///
    /// Mostly useful for telemetry and tests that need to confirm that repeated compilations of
    /// the same MLIR text reuse the cached executable instead of recompiling.
    #[inline]
    pub fn cache_size(&self) -> usize {
        self.executables.lock().expect("compile cache mutex should not be poisoned").len()
    }

    /// Compiles `mlir_text` if it is not already cached, or returns the cached executable
    /// otherwise.
    ///
    /// The cache key is a 64-bit hash of `mlir_text` bytes. Concurrent compilations of distinct
    /// MLIR texts are serialized by the internal mutex.
    pub(crate) fn compile(
        &self,
        mlir_text: &str,
        options: &CompilationOptions,
    ) -> Result<Arc<LoadedExecutable<'c>>, Error> {
        let key = hash_mlir(mlir_text);
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

fn hash_mlir(mlir_text: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    mlir_text.as_bytes().hash(&mut hasher);
    hasher.finish()
}
