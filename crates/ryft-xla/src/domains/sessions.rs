//! Session ownership of XLA clients, caches, runtime resources, and completions.

use std::sync::{Arc, Mutex, OnceLock};

use ryft_core::{ArrayType, CompilationContext, DeviceMesh, DiskCache, SpecializationCache};
use ryft_pjrt::{Client, ExecutionFence};

use crate::domains::Domain;
use crate::domains::eager::{EAGER_DISPATCH_CACHE_CAPACITY, EagerDispatchKey};
use crate::domains::effects::EffectScope;
use crate::domains::programs::CompiledProgram;
use crate::domains::target::Target;
use crate::kernels::CudaKernelRuntime;
#[cfg(feature = "rocm")]
use crate::kernels::RocmKernelRuntime;
use crate::{Array, Error};

/// Shared compilation and effect resources for execution on one externally owned PJRT [`Client`].
///
/// A session is the handle through which `ryft-xla` executes on a client: every [`Domain`] belongs to exactly one
/// session, and every [`Array`] belongs to exactly one domain. Create one session per client and derive every domain
/// from it (through [`Self::domain`], [`Domain::with_compilation_options`], or [`Domain::fork_effect_scope`]) so
/// that all arrays and domains on that client share one compilation cache, one default effect scope, and one set of
/// kernel runtimes, much like JAX's process-wide caches. Separate sessions remain independent, even on the same
/// client, so creating a second session is an explicit request for an isolated cache and effect ordering. Cached
/// executables retain neither sessions nor arrays, so the ownership graph has no cache/session cycle. Call
/// [`Self::effects_barrier`] before dropping the client when its effects must be observed; destruction adds no implicit
/// barrier and backend behavior with unfinished device work is plugin-specific.
///
/// # Compilation Caching
///
/// Each session owns one [`CompilationContext`], which its constructors configure. [`Self::new`] keeps compiled
/// programs only in memory, and [`Self::with_cache_capacity`] bounds that in-memory tier. [`Self::with_disk_cache`]
/// adds a persistent [`DiskCache`] so that later processes restart warm, and [`Self::with_disk_cache_from_env`] adds
/// the one that the [`DiskCache::ENV_VAR`] environment variable configures, which sessions never read otherwise.
/// Multi-process programs share compiled artifacts through a context with an exchange from
/// [`DistributedRuntime::compilation_artifact_exchange`](crate::DistributedRuntime::compilation_artifact_exchange),
/// which they pass to [`Self::with_compilation_context`]. Refer to [`CompilationContext`] for a configuration
/// example.
///
/// # Example
///
/// ```no_run
/// use ryft_core::{ArrayType, DataType, DeviceMesh, Print};
/// use ryft_pjrt::Client;
/// use ryft_xla::Session;
///
/// fn print_values<'c>(client: &'c Client<'c>, mesh: DeviceMesh) -> Result<(), Box<dyn std::error::Error>> {
///     let session = Session::new(client);
///     let value = session.array(ArrayType::scalar(DataType::F64), mesh, 1.0f64.to_ne_bytes())?;
///     value.print("value");
///     session.effects_barrier()?;
///     Ok(())
/// }
/// ```
///
/// A session cannot outlive its client:
/// ```compile_fail
/// use ryft_xla::Session;
/// fn escaped(client: ryft_pjrt::Client<'_>) {
///     let session = Session::new(&client);
///     drop(client);
///     session.effects_barrier().unwrap();
/// }
/// ```
pub struct Session<'c> {
    /// Externally owned client, which outlives cached executables and effect carriers.
    client: &'c Client<'c>,

    /// Compile target of [`Self::client`], read lazily on first use and retained (including a failed read) for the
    /// lifetime of the session.
    target: OnceLock<Result<Target, ryft_pjrt::Error>>,

    /// Cached artifacts contain no runtime execution association.
    cache: CompilationContext<'c, Domain<'c>>,

    /// Compiled programs of eager operation applications, keyed in front of tracing and lowering. Refer to the
    /// documentation of [`EagerDispatchKey`] for information on what the key captures.
    pub(super) eager_dispatch_cache: SpecializationCache<EagerDispatchKey, Arc<CompiledProgram<'c>>>,
    /// Default ordering and completion scope shared by domains and arrays created through this session.
    pub(super) default_effect_scope: Arc<EffectScope<'c>>,

    /// Stable allocation borrowed by CUDA FFI invocations; destroyed after all retained completions.
    pub(super) cuda_kernel_runtime: Mutex<Option<Box<CudaKernelRuntime>>>,

    /// Existing whole-execution fences protecting borrowed CUDA contexts even after outputs are dropped.
    pub(super) cuda_kernel_completions: Mutex<Vec<ExecutionFence>>,

    /// Separate HIP resources borrowed by ROCm execution contexts.
    #[cfg(feature = "rocm")]
    pub(super) rocm_kernel_runtime: Mutex<Option<Box<RocmKernelRuntime>>>,

    /// Whole-program completions retaining HIP user data and loaded modules after dropped outputs.
    #[cfg(feature = "rocm")]
    pub(super) rocm_kernel_completions: Mutex<Vec<ExecutionFence>>,
}

impl Drop for Session<'_> {
    fn drop(&mut self) {
        let completions = self.cuda_kernel_completions.get_mut().expect("cuda completion owner mutex poisoned");
        for completion in completions.drain(..) {
            match completion.block_until_ready() {
                // An execution error is terminal too. It belongs to the caller's existing fence/error channel;
                // cleanup only needs to ensure no queued handler can still access the borrowed session resources.
                Ok(()) | Err(_) => {}
            }
        }
        // Drop uses the CUDA owner's existing cleanup/error retention policy while our client borrow is alive.
        drop(self.cuda_kernel_runtime.get_mut().expect("cuda runtime owner mutex poisoned").take());
        #[cfg(feature = "rocm")]
        {
            let completions = self.rocm_kernel_completions.get_mut().expect("rocm completion owner mutex poisoned");
            for completion in completions.drain(..) {
                match completion.block_until_ready() {
                    Ok(()) | Err(_) => {}
                }
            }
            // Keep the borrowed PJRT client alive while HIP releases its synchronized module cache.
            drop(self.rocm_kernel_runtime.get_mut().expect("rocm runtime owner mutex poisoned").take());
        }
    }
}

impl<'c> Session<'c> {
    /// Creates an independent session with an empty in-memory compilation cache and a fresh default effect scope.
    pub fn new(client: &'c Client<'c>) -> Arc<Self> {
        Self::with_compilation_context(client, CompilationContext::new())
    }

    /// Creates a session with an explicitly configured compilation cache.
    pub fn with_compilation_context(client: &'c Client<'c>, cache: CompilationContext<'c, Domain<'c>>) -> Arc<Self> {
        Arc::new(Self {
            client,
            target: OnceLock::new(),
            cache,
            eager_dispatch_cache: SpecializationCache::new(EAGER_DISPATCH_CACHE_CAPACITY),
            default_effect_scope: Arc::new(EffectScope::default()),
            cuda_kernel_runtime: Mutex::new(None),
            cuda_kernel_completions: Mutex::new(Vec::new()),
            #[cfg(feature = "rocm")]
            rocm_kernel_runtime: Mutex::new(None),
            #[cfg(feature = "rocm")]
            rocm_kernel_completions: Mutex::new(Vec::new()),
        })
    }

    /// Creates a session whose in-memory compilation cache retains at most `capacity` compiled programs. A `capacity`
    /// of zero is clamped to one entry.
    pub fn with_cache_capacity(client: &'c Client<'c>, capacity: usize) -> Arc<Self> {
        Self::with_compilation_context(client, CompilationContext::with_capacity(capacity))
    }

    /// Creates a session whose compilation cache also writes through to `disk_cache`, with its size cap and write
    /// thresholds.
    pub fn with_disk_cache(client: &'c Client<'c>, disk_cache: DiskCache) -> Arc<Self> {
        Self::with_compilation_context(client, CompilationContext::new().with_disk_cache(disk_cache))
    }

    /// Creates a session whose compilation cache also writes through to a [`DiskCache`] configured via the
    /// [`DiskCache::ENV_VAR`](ryft_core::compilation::DiskCache::ENV_VAR) environment variable, if it is set. An absent
    /// variable produces an in-memory-only cache.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] when the configured cache directory is invalid.
    pub fn with_disk_cache_from_env(client: &'c Client<'c>) -> Result<Arc<Self>, Error> {
        Ok(Self::with_compilation_context(client, CompilationContext::new().with_disk_cache_from_env()?))
    }

    /// Returns the externally owned PJRT client.
    pub fn client(&self) -> &'c Client<'c> {
        self.client
    }

    /// Returns the [`Target`] of this session's client, reading it from the client on first use.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Pjrt`] if querying the platform, plugin, or device facts from the client failed. The
    /// failure is retained, so every later call returns the same error.
    pub fn target(&self) -> Result<&Target, Error> {
        self.target
            .get_or_init(|| Target::from_client(self.client))
            .as_ref()
            .map_err(|error| error.clone().into())
    }

    /// Returns the shared compilation cache.
    pub fn compilation_context(&self) -> &CompilationContext<'c, Domain<'c>> {
        &self.cache
    }

    /// Creates a domain sharing this session's cache and default scope, with the default
    /// [`CompilationOptions`](ryft_pjrt::protos::CompilationOptions) and no domain mesh.
    pub fn domain(self: &Arc<Self>) -> Domain<'c> {
        Domain::from_session_configuration(Arc::clone(self), None, Domain::default_compilation_options())
    }

    /// Uploads host bytes into an array that belongs to [`Self::domain`]. Refer to the documentation of
    /// [`Array::from_host_buffer`] for more information.
    pub fn array<B: AsRef<[u8]>>(
        self: &Arc<Self>,
        r#type: ArrayType,
        mesh: DeviceMesh,
        buffer: B,
    ) -> Result<Array<'c>, Error> {
        Array::from_host_buffer(&self.domain(), r#type, mesh, buffer)
    }

    /// Waits for the default scope's previously reserved effects, including unordered I/O and local assertions.
    /// Later submissions and forked scopes are excluded. Failures are retained until explicitly acknowledged.
    pub fn effects_barrier(&self) -> Result<(), Error> {
        self.default_effect_scope.barrier()
    }

    /// Acknowledges failures reported by a completed barrier once their affected ordering chain is quiescent.
    /// This never waits, repairs reference state, or changes errors retained by existing arrays/executions.
    pub fn acknowledge_effect_errors(&self) -> Result<(), Error> {
        self.default_effect_scope.acknowledge()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    #[cfg(feature = "cuda-13")]
    use pretty_assertions::assert_eq;
    #[cfg(feature = "cuda-13")]
    use ryft_core::{DataType, EffectClass, TracingContext, Value as CoreValue};
    use ryft_pjrt::{ClientOptions, CpuClientOptions, load_cpu_plugin};
    #[cfg(feature = "cuda-13")]
    use ryft_pjrt::{GpuClientOptions, GpuMemoryAllocator, GpuPlatform, load_cuda_13_plugin};

    #[cfg(feature = "cuda-13")]
    use crate::domains::options::Options;
    #[cfg(feature = "cuda-13")]
    use crate::domains::tests::*;
    use crate::experimental::lowering::signature::ExecutableSignature;
    #[cfg(feature = "cuda-13")]
    use crate::experimental::ops::{Constant, Operation};
    #[cfg(feature = "cuda-13")]
    use crate::tests::values_from_bytes;

    use super::*;

    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_session_cuda_kernel_execution_and_persistence() {
        use std::convert::Infallible;

        use ryft_core::{Add, OneLike, ProjectedValue, Tracer as CoreTracer};
        use ryft_cuda::{
            CudaArtifactFormat, CudaKernelAbi, CudaKernelArtifact, CudaKernelLaunchDimensions, CudaKernelParameterType,
        };
        use ryft_kernels::{KernelCompilationError, KernelCompiler, KernelDefinition, KernelSchedule, VerifiedKernel};

        use crate::kernels::{
            CUDA_KERNEL_CUSTOM_CALL_TARGET, CudaKernelEmbedding, KernelCompilerBinding, stage_kernel,
        };

        let plugin = load_cuda_13_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::GPU(GpuClientOptions {
                platform: Some(GpuPlatform::CUDA),
                allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                ..Default::default()
            }))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let session = Session::new(&client);
        let domain = session.domain();
        let definition: KernelDefinition =
            KernelDefinition::trace(crate::kernels::tests::definition().operation().clone(), |(references, _)| {
                let value = ProjectedValue::new(references[0].read()?, ArrayType::scalar(DataType::I32));
                references[0].write(value.add(&value.one_like()?)?.value())?;
                Ok(())
            })
            .unwrap();
        // The observable scalar increment must execute through the handler. Keeping inputs live verifies that XLA
        // preserves functional array semantics while supplying an initialized aliased writable result.
        let artifact = CudaKernelArtifact::new(
            CudaArtifactFormat::Ptx,
            b".version 7.0\n.target sm_80\n.address_size 64\n.visible .entry increment(.param .u64 data) {\n\
              .reg .b64 pointer;\n.reg .b32 value;\nld.param.u64 pointer, [data];\nld.global.u32 value, [pointer];\n\
              add.u32 value, value, 1;\nst.global.u32 [pointer], value;\nret;\n}\n"
                .to_vec(),
            "increment",
            "compute_80",
            CudaKernelLaunchDimensions::new([1; 3], [1; 3], 0).unwrap(),
            CudaKernelAbi::new("ryft.kernel.pointer", 1, vec![CudaKernelParameterType::DevicePointer]).unwrap(),
        )
        .unwrap();
        /// Hand-authored artifact fixture admitted only for the exact verified scalar increment body.
        struct IncrementCompiler {
            artifact: CudaKernelArtifact,
            semantic: String,
        }
        /// PTX fixture target requiring the actual CUDA platform, FFI ABI, and supported device architecture.
        struct IncrementTarget;
        impl crate::kernels::KernelTarget for IncrementTarget {
            fn admit_execution(
                &self,
                facts: &crate::kernels::KernelExecutionFacts,
            ) -> Result<(), crate::kernels::KernelEmbeddingError> {
                let supported_devices = !facts.devices.is_empty()
                    && facts.devices.iter().all(|device| match device.attributes.get("compute_capability") {
                        Some(ryft_pjrt::Value::String(capability)) => capability
                            .split('.')
                            .next()
                            .and_then(|major| major.parse::<usize>().ok())
                            .is_some_and(|major| major >= 8),
                        _ => false,
                    });
                if !facts.platform_name.eq_ignore_ascii_case("cuda")
                    || !facts.has_ffi_extension
                    || facts.pjrt_version.major != ryft_pjrt::VERSION.major
                    || facts.pjrt_version.minor < ryft_pjrt::VERSION.minor
                    || !supported_devices
                {
                    return Err(crate::kernels::KernelEmbeddingError::Invalid {
                        message:
                            "increment fixture requires CUDA compute capability 8.0 or newer and compatible PJRT FFI"
                                .to_owned(),
                    });
                }
                Ok(())
            }
        }
        impl KernelCompiler for IncrementCompiler {
            type Target = IncrementTarget;
            type Options = ();
            type Output = CudaKernelArtifact;
            type Error = Infallible;

            fn admit(
                &self,
                kernel: &VerifiedKernel<'_>,
                _target: &IncrementTarget,
                _options: &(),
                _schedule: &KernelSchedule,
            ) -> Result<(), KernelCompilationError<Infallible>> {
                if kernel.definition().semantic_key().unwrap() != self.semantic {
                    return Err(KernelCompilationError::Incompatible {
                        message: "increment fixture body changed".to_owned(),
                    });
                }
                Ok(())
            }
            fn configuration_key(
                &self,
                _target: &IncrementTarget,
                _options: &(),
                _schedule: &KernelSchedule,
            ) -> Result<Vec<u8>, KernelCompilationError<Infallible>> {
                let mut key = b"increment-fixture-v1;compute_80;pointer-abi-v1;grid1;block1;shared0".to_vec();
                key.extend_from_slice(self.artifact.bytes());
                Ok(key)
            }
            fn compile(
                &self,
                _kernel: &VerifiedKernel<'_>,
                _target: &IncrementTarget,
                _options: &(),
                _schedule: &KernelSchedule,
            ) -> Result<CudaKernelArtifact, KernelCompilationError<Infallible>> {
                Ok(self.artifact.clone())
            }
        }
        let compiler = KernelCompilerBinding::new(
            IncrementCompiler { artifact, semantic: definition.semantic_key().unwrap() },
            IncrementTarget,
            (),
            KernelSchedule::default(),
            CudaKernelEmbedding::new(CUDA_KERNEL_CUSTOM_CALL_TARGET.to_owned(), vec![0]),
            1,
        )
        .unwrap();
        let scalar = ArrayType::scalar(DataType::I32);
        let (_, program) = TracingContext::<Constant, Operation>::trace(
            |inputs: Vec<CoreTracer<TracingContext<Constant, Operation>>>| {
                stage_kernel(inputs[0].context(), &definition, &inputs)
            },
            definition.operation().input_types(),
        )
        .unwrap();
        assert!(program.effects().classes().contains(EffectClass::OrderedState));
        let options = Options::new(mesh.clone()).with_kernel_compiler(compiler);
        let lowered = domain.lower_program(&program, 0, &options).unwrap();
        assert!(lowered.signature.requires_cuda_kernel_runtime());
        assert!(lowered.kernel_execution_facts.is_some());
        assert!(!lowered.signature.has_effects());
        let compiled = domain.compile_program(&lowered).unwrap();
        let input = session.array(scalar, mesh, 42_i32.to_ne_bytes()).unwrap();
        let outputs = domain.execute_program(&compiled, vec![input.clone()]).unwrap();
        let bytes = domain.serialize_compiled_program(&compiled).unwrap().unwrap();
        let restored_session = Session::new(&client);
        let restored_domain = restored_session.domain();
        let restored = restored_domain.deserialize_compiled_program(&bytes).unwrap().unwrap();
        assert!(restored.signature.requires_cuda_kernel_runtime());
        assert!(restored_session.cuda_kernel_runtime.lock().unwrap().is_some());
        let restored_outputs = restored_domain.execute_program(&restored, vec![outputs[0].clone()]).unwrap();
        let device = client.addressable_devices().unwrap()[0].id().unwrap();
        for (array, expected) in [(&input, 42), (&outputs[0], 43), (&restored_outputs[0], 44)] {
            let bytes =
                array.device_shard(device).unwrap().buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(values_from_bytes::<i32>(&bytes), vec![expected]);
        }
        drop(outputs);
        drop(restored_outputs);
        // Both sessions retain their native launchers until the final queued execution completes, even though
        // neither this call's outputs nor the session's public handles are retained by the caller.
        drop(domain.execute_program(&compiled, vec![input.clone()]).unwrap());
        drop(restored_domain.execute_program(&restored, vec![input.clone()]).unwrap());
        drop(input);
        drop(restored_domain);
        drop(restored_session);
        drop(domain);
        drop(session);
    }
    #[test]
    fn test_session_drop_retains_cuda_completion_after_dropped_outputs() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions::default())).unwrap();
        let session = Session::new(&client);
        let (first, first_promise) = client.event(()).unwrap();
        let (second, second_promise) = client.event(()).unwrap();
        let fence = ExecutionFence::new(vec![first, second]);
        session.cuda_kernel_completions.lock().unwrap().push(fence.clone());
        drop(fence);
        std::thread::scope(|threads| {
            let (started, waiting) = std::sync::mpsc::channel();
            let (released, completed) = std::sync::mpsc::channel();
            threads.spawn(move || {
                started.send(()).unwrap();
                drop(session);
                released.send(()).unwrap();
            });
            waiting.recv_timeout(Duration::from_secs(5)).unwrap();
            assert!(matches!(
                completed.recv_timeout(Duration::from_millis(100)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            ));
            first_promise.set(Some(ryft_pjrt::Error::aborted("terminal test failure"))).unwrap();
            assert!(matches!(
                completed.recv_timeout(Duration::from_millis(100)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            ));
            second_promise.set(None).unwrap();
            completed.recv_timeout(Duration::from_secs(5)).unwrap();
        });
    }
    #[cfg(feature = "rocm")]
    #[test]
    fn test_session_drop_retains_rocm_completion_after_dropped_outputs() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions::default())).unwrap();
        let session = Session::new(&client);
        let (first, first_promise) = client.event(()).unwrap();
        let (second, second_promise) = client.event(()).unwrap();
        let fence = ExecutionFence::new(vec![first, second]);
        session.rocm_kernel_completions.lock().unwrap().push(fence.clone());
        drop(fence);
        std::thread::scope(|threads| {
            let (started, waiting) = std::sync::mpsc::channel();
            let (released, completed) = std::sync::mpsc::channel();
            threads.spawn(move || {
                started.send(()).unwrap();
                drop(session);
                released.send(()).unwrap();
            });
            waiting.recv_timeout(Duration::from_secs(5)).unwrap();
            assert!(matches!(
                completed.recv_timeout(Duration::from_millis(100)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            ));
            first_promise.set(Some(ryft_pjrt::Error::aborted("terminal test failure"))).unwrap();
            assert!(matches!(
                completed.recv_timeout(Duration::from_millis(100)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            ));
            second_promise.set(None).unwrap();
            completed.recv_timeout(Duration::from_secs(5)).unwrap();
        });
    }
    #[test]
    fn test_domain_cuda_runtime_rejects_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions::default())).unwrap();
        let session = Session::new(&client);
        let signature = ExecutableSignature::new(&[], &[]).with_cuda_kernel_runtime(true);
        assert!(!signature.has_effects());
        assert!(matches!(
            session.domain().ensure_runtime_requirements(&signature, "cpu"),
            Err(error)
                if matches!(error.downcast_custom::<crate::kernels::KernelEmbeddingError>(),
                    Some(crate::kernels::KernelEmbeddingError::Invalid { message })
                        if message == "cuda artifact calls require a cuda platform"),
        ));
        assert!(session.cuda_kernel_runtime.lock().unwrap().is_none());
    }
    #[test]
    fn test_domain_rocm_runtime_rejects_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions::default())).unwrap();
        let session = Session::new(&client);
        let signature = ExecutableSignature::new(&[], &[]).with_rocm_kernel_runtime(true);
        assert!(!signature.has_effects());
        assert!(matches!(
            session.domain().ensure_runtime_requirements(&signature, "cpu"),
            Err(error)
                if matches!(error.downcast_custom::<crate::kernels::KernelEmbeddingError>(),
                    Some(crate::kernels::KernelEmbeddingError::Invalid { message })
                        if message == "rocm artifact calls require a rocm platform"),
        ));
        #[cfg(not(feature = "rocm"))]
        assert!(matches!(
            session.domain().ensure_runtime_requirements(&signature, "rocm"),
            Err(error)
                if matches!(error.downcast_custom::<crate::kernels::KernelEmbeddingError>(),
                    Some(crate::kernels::KernelEmbeddingError::Invalid { message })
                        if message == "rocm artifact execution requires the `rocm` feature"),
        ));
        #[cfg(feature = "rocm")]
        assert!(session.rocm_kernel_runtime.lock().unwrap().is_none());
    }
    #[test]
    fn test_session_domain() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions::default())).unwrap();
        let session = Session::new(&client);
        let retained = Arc::downgrade(&session);
        let domain = session.domain();
        let another = session.domain();
        let fork = domain.fork_effect_scope();
        assert!(std::ptr::eq(session.client(), &client));
        assert!(std::ptr::eq(domain.compilation_context(), session.compilation_context()));
        assert!(std::ptr::eq(fork.compilation_context(), session.compilation_context()));
        assert!(Arc::ptr_eq(&domain.effect_scope, &another.effect_scope));
        assert!(!Arc::ptr_eq(&domain.effect_scope, &fork.effect_scope));
        let pending = domain.effect_scope.reserve(None).unwrap();
        assert!(fork.effects_barrier().is_ok());
        drop(pending);
        assert!(session.effects_barrier().is_ok());
        drop(session);
        assert!(retained.upgrade().is_some());
        drop(domain);
        drop(another);
        drop(fork);
        assert!(retained.upgrade().is_none());
    }
}
