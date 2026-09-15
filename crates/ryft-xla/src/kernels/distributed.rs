//! Experimental distributed kernel execution and native participant admission.
//!
//! Native kernels currently execute only on a fixed, fully addressable, single-process device mesh. Multiple local
//! devices share the existing PJRT [`ExecutionFence`](ryft_pjrt::ExecutionFence); its terminal result joins every
//! participating device, including asynchronous failures. The check in this module runs at the common XLA submission
//! boundary before input resharding, donation, or device submission, including restored executables and stateful calls.
//!
//! Compilation agreement already belongs to [`DistributedRuntime`](crate::DistributedRuntime). Its
//! [`compilation_artifact_exchange`](crate::DistributedRuntime::compilation_artifact_exchange) compares ordered rounds,
//! process counts, launch identity, and exact persistent compilation keys before exchanging checksummed artifacts.
//! Kernel compiler, target, plugin, and topology identities participate in those keys. Artifact agreement does not
//! establish collective completion and cannot authorize a cross-process launch.
//!
//! [`DistributedKernel`] provides an explicit host-staged functional route over the existing distributed runtime.
//! It coordinates per-process local kernels, PJRT downloads/uploads and bounded KV chunks. It does not submit a native
//! multi-process mesh. CPU needs no cross-host PJRT extension for this route; CUDA/ROCm extension presence does not
//! change its transport or imply native collective qualification. External references and observable I/O are rejected.
//!
//! [`DistributedKernel::call_async`] returns the existing [`ReferenceExecution`](ryft_core::ReferenceExecution).
//! Preflight and host transfers block before submission; its completion retains the existing native fence and runtime.
//! Awaiting completes the distributed readiness barrier. No background worker advances that barrier, and readiness
//! queries only report cached terminal results. Blocking [`DistributedKernel::call`] delegates to this same path.
//!
//! The runtime service and participants must be trusted and cooperative. Published/reassembled chunks are bounded,
//! but the existing KV API allocates returned values before this layer can inspect their lengths; this is not a
//! hostile-peer allocation firewall. KV records remain scoped to the runtime launch until its service is destroyed.
//! Cross-process in-place reference publication remains unsupported: all outputs are functional and private until
//! successful completion. All-ready means native completion, not atomic delivery to every host after a network fault.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, TryLockError};
use std::time::{Duration, Instant};

use ryft_core::{
    ArrayAddressing, Device, DeviceMesh, ReferenceCompletion, ReferenceCompletionBackend, ReferenceExecution, Typed,
};
use ryft_pjrt::{Client, Execution, ExecutionFence, KeyValueStore};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::experimental::XlaDomainError;
use crate::kernels::KernelEmbeddingError;
use crate::kernels::aot::{KernelAotError, LoadedKernel};
use crate::{Array, DistributedRuntime, FromPjrt};

/// Failure in host-coordinated kernel execution. Submitted device work is drained before returning an error.
#[derive(Debug, thiserror::Error)]
pub enum DistributedKernelError {
    /// The call or its peer manifest violates a bounded coordination contract.
    #[error("invalid distributed kernel call: {message}")]
    Invalid {
        /// Failed invariant.
        message: String,
    },

    /// Cancellation was observed before this participant published successful completion.
    #[error("distributed kernel call cancelled")]
    Cancelled,

    /// A coordination deadline expired. This never implies cancellation of device work.
    #[error("distributed kernel coordination deadline exceeded")]
    Deadline,

    /// A participant reported a terminal failure.
    #[error("distributed kernel participant {process} failed: {message}")]
    Participant {
        /// Initialization-owned participant index.
        process: usize,
        /// Bounded diagnostic reported by that participant.
        message: String,
    },

    /// Reporting a local failure also failed because the coordination service was unavailable.
    #[error("distributed kernel failed ({failure}); reporting the failure also failed ({reporting})")]
    Reporting {
        /// Original local failure.
        failure: String,
        /// Failure of its coordination publication.
        reporting: ryft_pjrt::Error,
    },

    /// A pending distributed completion failed through the canonical completion channel.
    #[error("distributed kernel completion failed: {message}")]
    Completion {
        /// Stable terminal diagnostic from the existing completion backend.
        message: Arc<str>,
    },

    /// The existing PJRT transfer or coordination primitive failed.
    #[error(transparent)]
    Pjrt(#[from] ryft_pjrt::Error),

    /// The existing AOT invocation failed.
    #[error(transparent)]
    Aot(#[from] KernelAotError),

    /// The existing XLA array or session operation failed.
    #[error(transparent)]
    Xla(#[from] XlaDomainError),

    /// Bounded coordination metadata was malformed.
    #[error(transparent)]
    Serialization(#[from] serde_json::Error),

    /// Live local participant admission failed.
    #[error(transparent)]
    Embedding(#[from] KernelEmbeddingError),
}

impl DistributedKernelError {
    /// Constructs an exact coordination-owned contract diagnostic.
    fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid { message: message.into() }
    }
}

/// Rejects participant sets without a qualified collective completion contract before any native submission.
pub(crate) fn validate_kernel_participants(client: &Client<'_>, mesh: &DeviceMesh) -> Result<(), KernelEmbeddingError> {
    let process = client.process_index()?;
    if mesh.devices().iter().any(|device| device.process_index() != process) {
        return Err(KernelEmbeddingError::Invalid {
            message: "cross-process kernel execution requires a qualified collective ordering and completion contract"
                .to_owned(),
        });
    }
    let addressable = client.addressable_devices()?.iter().map(Device::from_pjrt).collect::<Result<Vec<_>, _>>()?;
    if mesh.devices().iter().any(|device| !addressable.contains(device)) {
        return Err(KernelEmbeddingError::Invalid {
            message: "kernel execution requires every mesh device to be addressable by the submitting client"
                .to_owned(),
        });
    }
    Ok(())
}

/// Explicit host-staged transport limits. Device copies use PJRT; inter-process bytes use the runtime's KV store.
///
/// Deadlines and cancellation are cooperative between native operations. A native fence is always drained even when
/// that exceeds the coordination deadline. The coordinator retains chunks until the distributed runtime shuts down;
/// the round limit bounds retention for this coordinator, and callers control the number of coordinators they create.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistributedKernelOptions {
    /// Maximum bytes published by each participant in one round.
    maximum_transfer_bytes: usize,
    /// Maximum bytes in one KV chunk before the existing store's hexadecimal transport encoding.
    chunk_bytes: usize,
    /// Coordination deadline, excluding no native work; expiration cannot interrupt a native wait.
    timeout: Duration,
    /// Maximum rounds whose retained input chunks this coordinator can publish.
    maximum_rounds: u64,
}

impl DistributedKernelOptions {
    /// Creates bounded host transport options. Byte budgets are positive and cannot exceed 64 MiB per participant.
    pub fn new(maximum_transfer_bytes: usize, timeout: Duration) -> Result<Self, DistributedKernelError> {
        if maximum_transfer_bytes == 0 || maximum_transfer_bytes > 64 * 1024 * 1024 || timeout.is_zero() {
            return Err(DistributedKernelError::Invalid {
                message: "invalid host transport byte budget or timeout".into(),
            });
        }
        Ok(Self {
            maximum_transfer_bytes,
            chunk_bytes: maximum_transfer_bytes.min(1024 * 1024),
            timeout,
            maximum_rounds: 1024,
        })
    }

    /// Returns the per-participant byte budget.
    pub fn maximum_transfer_bytes(&self) -> usize {
        self.maximum_transfer_bytes
    }

    /// Returns the native-wait-aware coordination deadline.
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Sets a positive chunk size no larger than one MiB or the participant byte budget.
    pub fn with_chunk_bytes(mut self, bytes: usize) -> Result<Self, DistributedKernelError> {
        if bytes == 0 || bytes > self.maximum_transfer_bytes.min(1024 * 1024) {
            return Err(DistributedKernelError::Invalid { message: "invalid host transport chunk size".into() });
        }
        self.chunk_bytes = bytes;
        Ok(self)
    }

    /// Sets the positive per-coordinator round limit, bounding retained KV payloads over its lifetime.
    pub fn with_maximum_rounds(mut self, rounds: u64) -> Result<Self, DistributedKernelError> {
        if rounds == 0 || rounds > 1024 {
            return Err(DistributedKernelError::Invalid { message: "round limit must be between one and 1024".into() });
        }
        self.maximum_rounds = rounds;
        Ok(self)
    }
}

/// One participant's exact preflight record, constructed from its loaded executable and live local topology.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct KernelPreflight {
    /// Initialization-owned sender index.
    process: usize,
    /// Declared runtime participant count.
    processes: usize,
    /// Exact canonical source and compiler-binding identity.
    identity: Vec<u8>,
    /// Locally validated plugin and canonical device topology identity.
    execution: Vec<u8>,
    /// Canonical logical byte counts in input order.
    sizes: Vec<usize>,
    /// Source-process owner for each local argument index.
    sources: Vec<usize>,
    /// Exact bounded transport and deadline policy.
    options: DistributedKernelOptions,
}

/// A completed input download and its bounded KV chunk checksum.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct KernelTransfer {
    /// Exact canonical logical payload length.
    size: usize,
    /// SHA-256 of every ordered payload byte.
    checksum: [u8; 32],
}

/// A local native terminal state; an absent error means the full execution fence completed successfully.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct KernelTerminal {
    /// Absent only after successful whole-execution completion.
    failure: Option<String>,
}

/// Functional distributed execution through explicit host staging, with one ordered call at a time.
///
/// Every participant constructs coordinators and calls them in the same order. For argument `i`, `sources[i]` selects
/// the process supplying that argument's value; all processes supply their complete local input vector. Source rows may
/// differ, supporting local data parallel calls, rings and broadcasts without an additional transfer graph. Every
/// participant uses the same semantic kernel and compiler configuration; its actual local topology is independently
/// validated and included in the collectively acknowledged preflight manifest.
///
/// Setup blocks the host for existing PJRT downloads and uploads; `call_async` then returns a pending native execution.
/// Inputs are copied into fresh invocation buffers, including locally sourced inputs, so functional read-write aliases
/// cannot mutate caller-owned inputs. External reference slots and observable I/O are forbidden. Kernel assertions
/// retain their native failure semantics. Native multi-process meshes and PJRT collective support are not implied.
///
/// Outputs remain private until every participant advertises a successful native fence. This proves completion, not
/// atomic delivery to every host: a network failure after readiness may prevent one process receiving a result already
/// visible elsewhere. Cancellation cannot revoke a returned immutable output or abort submitted device work.
/// The runtime, loaded kernel and their existing client/session resources must outlive this coordinator.
pub struct DistributedKernel<'r, 'c> {
    /// Existing loaded kernel; its client/session lifetime is retained by the caller.
    kernel: &'r LoadedKernel<'c>,
    /// Shared coordination ownership, also retained by pending native completions.
    coordination: Arc<KernelCoordination>,
    /// Checked next round within this reserved coordinator.
    round: u64,
    /// Canonical logical byte counts for each argument.
    sizes: Vec<usize>,
    /// Previous pending call; later setup cannot overtake its completion.
    previous: Option<ReferenceCompletion>,
}

/// Metadata and existing runtime ownership shared with pending completion; no worker or registry is created.
struct KernelCoordination {
    /// Existing runtime owner retained through pending completion.
    runtime: DistributedRuntime,
    /// Launch identity and unique ordered coordinator reservation.
    prefix: Vec<u8>,
    /// Initialization-owned local participant index.
    process: usize,
    /// Initialization-owned participant count.
    processes: usize,
    /// Checked bounded transport policy.
    options: DistributedKernelOptions,
}

impl<'r, 'c> DistributedKernel<'r, 'c> {
    /// Creates one coordinator over a loaded stateless kernel and an existing distributed runtime.
    /// The initial transport supports one fully addressable device per participant and static array boundaries.
    pub fn new(
        runtime: &DistributedRuntime,
        kernel: &'r LoadedKernel<'c>,
        options: DistributedKernelOptions,
    ) -> Result<Self, DistributedKernelError> {
        let options = DistributedKernelOptions::new(options.maximum_transfer_bytes, options.timeout)?
            .with_chunk_bytes(options.chunk_bytes)?
            .with_maximum_rounds(options.maximum_rounds)?;
        kernel.validate_distributed_effects()?;
        let (domain, inputs, mesh, identity, execution) = kernel.distributed_parts();
        let client = domain.client().map_err(XlaDomainError::from)?;
        validate_kernel_participants(client, mesh)?;
        if mesh.devices().len() != 1 || inputs.is_empty() || inputs.len() > 64 {
            return Err(DistributedKernelError::invalid(
                "host transport requires one local device and one to 64 array inputs",
            ));
        }
        if identity.len() > 64 * 1024 || execution.len() > 64 * 1024 {
            return Err(DistributedKernelError::invalid("kernel identity exceeds the coordination metadata budget"));
        }
        let actual = crate::kernels::XlaKernelExecutionFacts::from_client(client, mesh)?.configuration_key()?;
        if actual != execution {
            return Err(DistributedKernelError::invalid("loaded kernel live execution identity changed"));
        }
        let sizes = inputs
            .iter()
            .map(|r#type| {
                if r#type.static_shape().is_none() {
                    return Err(DistributedKernelError::invalid("host transport requires static input shapes"));
                }
                let addressing = ArrayAddressing::new(r#type.clone())
                    .map_err(|error| DistributedKernelError::invalid(error.to_string()))?;
                let bytes = addressing.logical_byte_len();
                if bytes == 0 || !addressing.is_dense_row_major() {
                    return Err(DistributedKernelError::invalid(
                        "host transport requires nonempty dense row-major input storage",
                    ));
                }
                Ok(bytes)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let total = sizes
            .iter()
            .try_fold(0usize, |total, size| total.checked_add(*size))
            .ok_or_else(|| DistributedKernelError::invalid("input byte count overflow"))?;
        if total > options.maximum_transfer_bytes {
            return Err(DistributedKernelError::invalid("inputs exceed the host transport byte budget"));
        }
        let chunks = sizes.iter().map(|size| size.div_ceil(options.chunk_bytes)).sum::<usize>();
        if chunks > 4096 {
            return Err(DistributedKernelError::invalid("input chunk count exceeds the host transport budget"));
        }
        let (launch, coordinator, process, processes) = runtime.reserve_kernel_coordinator()?;
        if processes == 0 || processes > 256 || process >= processes {
            return Err(DistributedKernelError::invalid("invalid distributed runtime participant coordinates"));
        }
        let mut prefix = b"ryft/kernel-host/v1/".to_vec();
        prefix.extend_from_slice(&launch);
        prefix.extend_from_slice(&coordinator.to_le_bytes());
        Ok(Self {
            kernel,
            coordination: Arc::new(KernelCoordination {
                runtime: runtime.clone(),
                prefix,
                process: process as usize,
                processes: processes as usize,
                options,
            }),
            round: 0,
            sizes,
            previous: None,
        })
    }

    /// Starts a call after bounded host preflight and transfers, returning a pending native completion.
    /// Outputs remain private inside the existing `ReferenceExecution` until all participants complete successfully.
    /// A later call first awaits this call. Dropping the coordinator and all completion handles drains native work
    /// and reports abandonment; dropping only the returned handle does not cancel work retained by the coordinator.
    pub fn call_async(
        &mut self,
        inputs: Vec<Array<'c>>,
        sources: &[usize],
        cancelled: Arc<AtomicBool>,
    ) -> Result<ReferenceExecution<Vec<Array<'c>>, DistributedKernelError>, DistributedKernelError> {
        if let Some(previous) = self.previous.take() {
            previous.r#await().map_err(|message| DistributedKernelError::Completion { message })?;
        }
        if self.round >= self.coordination.options.maximum_rounds {
            return Err(DistributedKernelError::invalid("distributed kernel round limit exceeded"));
        }
        let round = self.round;
        self.round = self
            .round
            .checked_add(1)
            .ok_or_else(|| DistributedKernelError::invalid("kernel round counter overflow"))?;
        let deadline = Instant::now()
            .checked_add(self.coordination.options.timeout)
            .ok_or_else(|| DistributedKernelError::invalid("coordination deadline overflow"))?;
        let execution = match self.run(round, inputs, sources, &cancelled, deadline) {
            Ok(execution) => execution,
            Err(error) => return Err(self.coordination.report_failure(round, error)),
        };
        let outputs = execution.output().clone();
        let completion = ReferenceCompletion::new(KernelCompletion {
            coordination: Arc::clone(&self.coordination),
            round,
            deadline,
            fence: execution.fence().clone(),
            cancelled,
            terminal: Mutex::new(None),
        });
        self.previous = Some(completion.clone());
        Ok(ReferenceExecution::pending(Ok(outputs), completion, |message| DistributedKernelError::Completion {
            message,
        }))
    }

    /// Waits for the existing pending call surface and returns only globally ready functional outputs.
    pub fn call(
        &mut self,
        inputs: Vec<Array<'c>>,
        sources: &[usize],
        cancelled: Arc<AtomicBool>,
    ) -> Result<Vec<Array<'c>>, DistributedKernelError> {
        self.call_async(inputs, sources, cancelled)?.r#await()
    }

    /// Performs ordered agreement, bounded host transfers and the ordinary completion-bearing invocation.
    fn run(
        &self,
        round: u64,
        inputs: Vec<Array<'c>>,
        sources: &[usize],
        cancelled: &AtomicBool,
        deadline: Instant,
    ) -> Result<Execution<Vec<Array<'c>>>, DistributedKernelError> {
        self.coordination.check(cancelled, deadline)?;
        let (domain, types, mesh, identity, execution) = self.kernel.distributed_parts();
        if inputs.len() != types.len()
            || sources.len() != types.len()
            || sources.iter().any(|&source| source >= self.coordination.processes)
        {
            return Err(DistributedKernelError::invalid(
                "input count or source-process routing differs from the kernel boundary",
            ));
        }
        for (input, r#type) in inputs.iter().zip(types) {
            if input.r#type().data_type() != r#type.data_type()
                || input.r#type().shape() != r#type.shape()
                || input.mesh() != *mesh
                || input.addressable_shards().count() != 1
                || !ArrayAddressing::new(input.r#type().into_owned())
                    .map_err(|error| DistributedKernelError::invalid(error.to_string()))?
                    .is_dense_row_major()
            {
                return Err(DistributedKernelError::invalid(
                    "local input type or placement differs from the loaded kernel boundary",
                ));
            }
        }
        let manifest = KernelPreflight {
            process: self.coordination.process,
            processes: self.coordination.processes,
            identity: identity.to_vec(),
            execution: execution.to_vec(),
            sizes: self.sizes.clone(),
            sources: sources.to_vec(),
            options: self.coordination.options.clone(),
        };
        self.coordination.put_json(round, self.coordination.process, "preflight", &manifest)?;
        let mut agreement = Sha256::new();
        for process in 0..self.coordination.processes {
            let bytes = self.coordination.wait(round, process, "preflight", cancelled, deadline)?;
            let peer: KernelPreflight = serde_json::from_slice(&bytes)?;
            if peer.process != process
                || peer.processes != self.coordination.processes
                || peer.identity != identity
                || peer.sizes != self.sizes
                || peer.options != self.coordination.options
                || peer.sources.len() != types.len()
                || peer.sources.iter().any(|&source| source >= self.coordination.processes)
                || peer.execution.is_empty()
            {
                return Err(DistributedKernelError::invalid(
                    "participant kernel, options or routing preflight mismatch",
                ));
            }
            agreement.update((bytes.len() as u64).to_le_bytes());
            agreement.update(bytes);
        }
        let agreement = agreement.finalize();
        self.coordination
            .runtime
            .key_value_store()
            .put(&self.coordination.key(round, self.coordination.process, "agreement"), agreement.as_slice())?;
        for process in 0..self.coordination.processes {
            if self.coordination.wait(round, process, "agreement", cancelled, deadline)? != agreement.as_slice() {
                return Err(DistributedKernelError::invalid("participant topology agreement mismatch"));
            }
        }
        for (index, input) in inputs.iter().enumerate() {
            self.coordination.check(cancelled, deadline)?;
            input.block_until_ready().map_err(XlaDomainError::from)?;
            let shard = input.addressable_shards().next().unwrap();
            let buffer =
                shard.buffer().ok_or_else(|| DistributedKernelError::invalid("input has no addressable buffer"))?;
            let bytes =
                buffer.copy_to_host(Some(ryft_pjrt::Layout::dense_major_to_minor(types[index].rank())))?.r#await()?;
            if bytes.len() != self.sizes[index] {
                return Err(DistributedKernelError::invalid(
                    "downloaded input byte count differs from its canonical type",
                ));
            }
            self.coordination.check(cancelled, deadline)?;
            for (chunk, bytes) in bytes.chunks(self.coordination.options.chunk_bytes).enumerate() {
                self.coordination.runtime.key_value_store().put(
                    &self.coordination.key(round, self.coordination.process, &format!("input/{index}/{chunk}")),
                    bytes,
                )?;
            }
            self.coordination.put_json(
                round,
                self.coordination.process,
                &format!("input/{index}/manifest"),
                &KernelTransfer { size: bytes.len(), checksum: Sha256::digest(&bytes).into() },
            )?;
        }
        let client = domain.client().map_err(XlaDomainError::from)?;
        let mut arguments = Vec::with_capacity(types.len());
        for (index, (&process, r#type)) in sources.iter().zip(types).enumerate() {
            let manifest: KernelTransfer = serde_json::from_slice(&self.coordination.wait(
                round,
                process,
                &format!("input/{index}/manifest"),
                cancelled,
                deadline,
            )?)?;
            if manifest.size != self.sizes[index] {
                return Err(DistributedKernelError::invalid("received input size differs from its canonical type"));
            }
            let mut bytes = Vec::with_capacity(manifest.size);
            for chunk in 0..manifest.size.div_ceil(self.coordination.options.chunk_bytes) {
                let value =
                    self.coordination.wait(round, process, &format!("input/{index}/{chunk}"), cancelled, deadline)?;
                let expected = (manifest.size - bytes.len()).min(self.coordination.options.chunk_bytes);
                if value.len() != expected {
                    return Err(DistributedKernelError::invalid("received input chunk has the wrong length"));
                }
                bytes.extend_from_slice(&value);
            }
            if Sha256::digest(&bytes).as_slice() != manifest.checksum {
                return Err(DistributedKernelError::invalid("received input checksum mismatch"));
            }
            let value =
                Array::from_host_buffer(client, r#type.clone(), mesh.clone(), bytes).map_err(XlaDomainError::from)?;
            value.block_until_ready().map_err(XlaDomainError::from)?;
            arguments.push(value);
        }
        self.coordination.check(cancelled, deadline)?;
        Ok(self.kernel.call(arguments)?)
    }
}

impl KernelCoordination {
    /// Publishes one failure before readiness, preserving the original error unless reporting itself fails.
    fn report_failure(&self, round: u64, error: DistributedKernelError) -> DistributedKernelError {
        let message = error.to_string().chars().take(2048).collect::<String>();
        let terminal = KernelTerminal { failure: Some(message) };
        match self.put_json(round, self.process, "terminal", &terminal) {
            Ok(()) => error,
            Err(DistributedKernelError::Pjrt(reporting)) => {
                DistributedKernelError::Reporting { failure: error.to_string(), reporting }
            }
            Err(reporting) => reporting,
        }
    }

    /// Encodes a fixed, launch-scoped rendezvous key without a process-local registry.
    fn key(&self, round: u64, process: usize, suffix: &str) -> Vec<u8> {
        let mut key = self.prefix.clone();
        key.extend_from_slice(format!("/{round:016x}/{process:08x}/{suffix}").as_bytes());
        key
    }

    /// Publishes one bounded metadata record through the existing distributed store.
    fn put_json(
        &self,
        round: u64,
        process: usize,
        suffix: &str,
        value: &impl Serialize,
    ) -> Result<(), DistributedKernelError> {
        let bytes = serde_json::to_vec(value)?;
        if bytes.len() > 512 * 1024 {
            return Err(DistributedKernelError::invalid("coordination metadata exceeds its size budget"));
        }
        Ok(self.runtime.key_value_store().put(&self.key(round, process, suffix), &bytes)?)
    }

    /// Waits with bounded polling while observing every participant's published failure.
    fn wait(
        &self,
        round: u64,
        process: usize,
        suffix: &str,
        cancelled: &AtomicBool,
        deadline: Instant,
    ) -> Result<Vec<u8>, DistributedKernelError> {
        loop {
            self.check(cancelled, deadline)?;
            for participant in 0..self.processes {
                match self.runtime.key_value_store().try_get(&self.key(round, participant, "terminal")) {
                    Ok(bytes) => {
                        if bytes.len() > 16 * 1024 {
                            return Err(DistributedKernelError::invalid("terminal metadata exceeds its size budget"));
                        }
                        let terminal: KernelTerminal = serde_json::from_slice(&bytes)?;
                        if let Some(message) = terminal.failure {
                            return Err(DistributedKernelError::Participant { process: participant, message });
                        }
                    }
                    Err(ryft_pjrt::Error::NotFound { .. }) => (),
                    Err(error) => return Err(error.into()),
                }
            }
            match self.runtime.key_value_store().try_get(&self.key(round, process, suffix)) {
                Ok(bytes) => {
                    if bytes.len() > (512 * 1024).max(self.options.chunk_bytes) {
                        return Err(DistributedKernelError::invalid(
                            "received coordination value exceeds its size budget",
                        ));
                    }
                    return Ok(bytes);
                }
                Err(ryft_pjrt::Error::NotFound { .. }) => std::thread::sleep(Duration::from_millis(2)),
                Err(error) => return Err(error.into()),
            }
        }
    }

    /// Checks cooperative cancellation and the coordination deadline between native operations.
    fn check(&self, cancelled: &AtomicBool, deadline: Instant) -> Result<(), DistributedKernelError> {
        if cancelled.load(Ordering::Acquire) {
            return Err(DistributedKernelError::Cancelled);
        }
        if Instant::now() >= deadline {
            return Err(DistributedKernelError::Deadline);
        }
        Ok(())
    }
}

/// Existing native completion plus owned host coordination; outputs remain in `ReferenceExecution`.
struct KernelCompletion {
    /// Coordination ownership retained independently of the public output handle.
    coordination: Arc<KernelCoordination>,
    /// Ordered call index within the coordinator.
    round: u64,
    /// Cooperative coordination deadline.
    deadline: Instant,
    /// Existing whole-native-execution fence with its resource ownership.
    fence: ExecutionFence,
    /// Shared cooperative cancellation request.
    cancelled: Arc<AtomicBool>,
    /// Cached immutable outcome, serialized across completion observers.
    terminal: Mutex<Option<Result<(), Arc<str>>>>,
}

impl KernelCompletion {
    /// Drains the native fence, then publishes local readiness and observes the common terminal barrier.
    fn finish(&self) -> Result<(), DistributedKernelError> {
        if let Err(error) = self.fence.block_until_ready() {
            return Err(self.coordination.report_failure(self.round, error.into()));
        }
        if let Err(error) = self.coordination.check(&self.cancelled, self.deadline) {
            return Err(self.coordination.report_failure(self.round, error));
        }
        // A lost put acknowledgement is ambiguous: after this attempt, never overwrite readiness with failure.
        self.coordination.put_json(
            self.round,
            self.coordination.process,
            "terminal",
            &KernelTerminal { failure: None },
        )?;
        let committed = AtomicBool::new(false);
        for process in 0..self.coordination.processes {
            let terminal: KernelTerminal = serde_json::from_slice(&self.coordination.wait(
                self.round,
                process,
                "terminal",
                &committed,
                self.deadline,
            )?)?;
            if let Some(message) = terminal.failure {
                return Err(DistributedKernelError::Participant { process, message });
            }
        }
        Ok(())
    }
}

impl ReferenceCompletionBackend for KernelCompletion {
    fn r#await(&self) -> Result<(), Arc<str>> {
        let mut terminal = self.terminal.lock().expect("distributed completion mutex poisoned");
        if let Some(result) = &*terminal {
            return result.clone();
        }
        let result = self.finish().map_err(|error| Arc::<str>::from(error.to_string()));
        *terminal = Some(result.clone());
        result
    }

    fn is_ready(&self) -> Result<bool, Arc<str>> {
        // Never make synchronous network requests from the nonblocking completion query.
        match self.terminal.try_lock() {
            Ok(terminal) => terminal.as_ref().map(|result| result.clone().map(|()| true)).unwrap_or(Ok(false)),
            Err(TryLockError::WouldBlock) => Ok(false),
            Err(TryLockError::Poisoned(_)) => panic!("distributed completion mutex poisoned"),
        }
    }
}

impl Drop for KernelCompletion {
    fn drop(&mut self) {
        if self.terminal.get_mut().expect("distributed completion mutex poisoned").is_none() {
            // Last-handle abandonment drains work and notifies peers; native buffers are never released early.
            let error = match self.fence.block_until_ready() {
                Ok(()) => DistributedKernelError::Cancelled,
                Err(error) => error.into(),
            };
            let error = self.coordination.report_failure(self.round, error);
            *self.terminal.get_mut().unwrap() = Some(Err(Arc::from(error.to_string())));
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::{LogicalMesh, MeshAxis, MeshAxisType};
    use ryft_pjrt::{ClientOptions, CpuClientOptions, load_cpu_plugin};

    use crate::tests::execution_client;

    use super::*;

    /// Produces a real CPU-loaded scalar identity through the existing AOT persistence fixture.
    fn loaded<'c>(
        client: &'c ryft_pjrt::Client<'c>,
        domain: &crate::XlaDomain<'c>,
        mesh: &DeviceMesh,
        compiler_options: u32,
    ) -> LoadedKernel<'c> {
        let compiler = crate::kernels::staging::tests::binding(compiler_options);
        crate::kernels::aot::tests::executable_bundle(client, domain, mesh, ryft_core::DataType::I32, false, &compiler)
            .load(domain, &compiler, mesh)
            .unwrap()
    }

    /// Downloads one completed scalar result for independent host assertions.
    fn scalar(array: &Array<'_>) -> i32 {
        array.block_until_ready().unwrap();
        let bytes = array
            .addressable_shards()
            .next()
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        i32::from_ne_bytes(bytes.as_slice().try_into().unwrap())
    }

    /// Runs real child processes with required sockets, retaining both child statuses for the qualification log.
    fn processes(mode: &str) {
        use std::process::{Command, Stdio};
        use std::time::{Duration, Instant};
        let listener =
            std::net::TcpListener::bind("127.0.0.1:0").expect("distributed kernel tests require loopback sockets");
        let address = listener.local_addr().unwrap().to_string();
        drop(listener);
        let mut children = (0..2)
            .map(|process| {
                Command::new(std::env::current_exe().unwrap())
                    .args(["--exact", "kernels::distributed::tests::test_distributed_kernel_process", "--nocapture"])
                    .env("RYFT_KERNEL_PROCESS", process.to_string())
                    .env("RYFT_KERNEL_ADDRESS", &address)
                    .env("RYFT_KERNEL_MODE", mode)
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let deadline = Instant::now() + Duration::from_secs(30);
        while children.iter_mut().any(|child| child.try_wait().unwrap().is_none()) {
            if Instant::now() >= deadline {
                for child in &mut children {
                    if child.try_wait().unwrap().is_none() {
                        child.kill().unwrap();
                    }
                }
                panic!("distributed kernel child timeout");
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        for child in children {
            let process = child.id();
            let output = child.wait_with_output().unwrap();
            eprintln!("distributed kernel {mode} child {process}: {}", output.status);
            assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        }
    }

    #[test]
    fn test_distributed_kernel_options_new() {
        let options = DistributedKernelOptions::new(16, Duration::from_secs(2))
            .unwrap()
            .with_chunk_bytes(2)
            .unwrap()
            .with_maximum_rounds(3)
            .unwrap();
        assert_eq!(options.maximum_transfer_bytes(), 16);
        assert_eq!(options.timeout(), Duration::from_secs(2));
        assert!(matches!(
                DistributedKernelOptions::new(0, options.timeout()),
                Err(DistributedKernelError::Invalid { message })
            if message == "invalid host transport byte budget or timeout"));
        assert!(matches!(options.clone().with_chunk_bytes(0), Err(DistributedKernelError::Invalid { message })
            if message == "invalid host transport chunk size"));
        assert!(matches!(options.with_maximum_rounds(1025), Err(DistributedKernelError::Invalid { message })
            if message == "round limit must be between one and 1024"));
    }

    #[test]
    fn test_distributed_kernel_call_async() {
        processes("success");
    }

    #[test]
    fn test_distributed_kernel_call_cancellation() {
        processes("cancel");
    }

    #[test]
    fn test_distributed_kernel_call_cancellation_after_dispatch() {
        processes("cancel_dispatched");
    }

    #[test]
    fn test_distributed_kernel_call_configuration_mismatch() {
        processes("configuration");
    }

    #[test]
    fn test_distributed_kernel_call_invalid_routing() {
        processes("routing");
    }

    #[test]
    fn test_distributed_kernel_call_order_mismatch() {
        processes("order");
    }

    #[test]
    fn test_distributed_kernel_call_dropped_completion() {
        processes("drop");
    }

    #[test]
    fn test_distributed_kernel_completion_readiness_is_immutable() {
        processes("late_failure");
    }

    #[test]
    fn test_distributed_kernel_process() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let Ok(process) = std::env::var("RYFT_KERNEL_PROCESS") else {
            return;
        };
        let process = process.parse::<usize>().unwrap();
        let address = std::env::var("RYFT_KERNEL_ADDRESS").unwrap();
        let mode = std::env::var("RYFT_KERNEL_MODE").unwrap();
        let plugin = load_cpu_plugin().unwrap();
        let runtime = DistributedRuntime::initialize(&plugin, &address, 2, process as u32).unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![]).unwrap(),
            vec![Device::from_pjrt(client.addressable_devices().unwrap().remove(0)).unwrap()],
        )
        .unwrap();
        let domain = crate::XlaDomain::with_mesh(&client, mesh.clone());
        let compiler_options = if mode == "configuration" { 1 + process as u32 } else { 1 };
        let kernel = loaded(&client, &domain, &mesh, compiler_options);
        let options = DistributedKernelOptions::new(16, Duration::from_secs(5)).unwrap().with_chunk_bytes(2).unwrap();
        let mut coordinator = DistributedKernel::new(&runtime, &kernel, options.clone()).unwrap();
        let cancelled = Arc::new(AtomicBool::new(false));
        let input = Array::from_host_buffer(
            &client,
            ryft_core::ArrayType::scalar(ryft_core::DataType::I32),
            mesh.clone(),
            (10 + process as i32).to_ne_bytes(),
        )
        .unwrap();
        match mode.as_str() {
            "success" => {
                let pending =
                    coordinator.call_async(vec![input.clone()], &[1 - process], Arc::clone(&cancelled)).unwrap();
                assert!(!coordinator.previous.as_ref().unwrap().is_ready().unwrap());
                let output = pending.r#await().unwrap();
                assert_eq!(scalar(&output[0]), 11 - process as i32);
                assert_eq!(scalar(&input), 10 + process as i32);
                let output = coordinator.call(vec![input.clone()], &[process], Arc::clone(&cancelled)).unwrap();
                assert_eq!(scalar(&output[0]), 10 + process as i32);
                let mut recreated = DistributedKernel::new(&runtime, &kernel, options.clone()).unwrap();
                assert_ne!(recreated.coordination.prefix, coordinator.coordination.prefix);
                let output = recreated.call(vec![input.clone()], &[1 - process], Arc::clone(&cancelled)).unwrap();
                assert_eq!(scalar(&output[0]), 11 - process as i32);
                // Constructor validation also applies to metadata decoded without its public constructors.
                let mut malformed = options.clone();
                malformed.chunk_bytes = 0;
                assert!(matches!(
                        DistributedKernel::new(&runtime, &kernel, malformed),
                        Err(DistributedKernelError::Invalid { message })
                    if message == "invalid host transport chunk size"));
            }
            "cancel" => {
                if process == 0 {
                    cancelled.store(true, Ordering::Release);
                }
                let result = coordinator.call(vec![input.clone()], &[1 - process], Arc::clone(&cancelled));
                if process == 0 {
                    assert!(matches!(result, Err(DistributedKernelError::Cancelled)));
                } else {
                    assert!(matches!(result, Err(DistributedKernelError::Participant { process: 0, .. })));
                }
            }
            "cancel_dispatched" => {
                let pending =
                    coordinator.call_async(vec![input.clone()], &[1 - process], Arc::clone(&cancelled)).unwrap();
                runtime
                    .key_value_store()
                    .put(format!("test-dispatched-{process}").as_bytes(), b"submitted")
                    .unwrap();
                assert_eq!(
                    runtime
                        .key_value_store()
                        .get(format!("test-dispatched-{}", 1 - process).as_bytes(), Duration::from_secs(5))
                        .unwrap(),
                    b"submitted"
                );
                // Both native calls have been submitted, but neither completion has published readiness.
                assert!(matches!(
                    runtime.key_value_store().try_get(&coordinator.coordination.key(0, process, "terminal")),
                    Err(ryft_pjrt::Error::NotFound { .. })
                ));
                if process == 0 {
                    cancelled.store(true, Ordering::Release);
                    runtime.key_value_store().put(b"test-cancel-requested", b"cancelled").unwrap();
                }
                assert_eq!(
                    runtime.key_value_store().get(b"test-cancel-requested", Duration::from_secs(5)).unwrap(),
                    b"cancelled"
                );
                let expected = if process == 0 {
                    "distributed kernel call cancelled"
                } else {
                    "distributed kernel participant 0 failed: distributed kernel call cancelled"
                };
                assert!(matches!(
                    pending.r#await(),
                    Err(DistributedKernelError::Completion { message }) if message.as_ref() == expected,
                ));
                let terminal = runtime
                    .key_value_store()
                    .get(&coordinator.coordination.key(0, 0, "terminal"), Duration::from_secs(5))
                    .unwrap();
                assert_eq!(
                    serde_json::from_slice::<KernelTerminal>(&terminal).unwrap().failure.as_deref(),
                    Some("distributed kernel call cancelled")
                );
            }
            "configuration" => {
                let expected = "participant kernel, options or routing preflight mismatch";
                let result = coordinator.call(vec![input.clone()], &[1 - process], Arc::clone(&cancelled));
                match result {
                    Err(DistributedKernelError::Invalid { message }) => assert_eq!(message, expected),
                    Err(DistributedKernelError::Participant { process: participant, message }) => {
                        assert_eq!(participant, 1 - process);
                        assert_eq!(message, format!("invalid distributed kernel call: {expected}"));
                    }
                    result => panic!("unexpected configuration result: {result:?}"),
                }
                assert!(coordinator.previous.is_none());
                let mut identities = Vec::new();
                for participant in 0..2 {
                    let preflight = runtime
                        .key_value_store()
                        .get(&coordinator.coordination.key(0, participant, "preflight"), Duration::from_secs(5))
                        .unwrap();
                    let preflight: KernelPreflight = serde_json::from_slice(&preflight).unwrap();
                    identities.push(serde_json::from_slice::<serde_json::Value>(&preflight.identity).unwrap());
                    assert!(matches!(
                        runtime.key_value_store().try_get(&coordinator.coordination.key(
                            0,
                            participant,
                            "input/0/manifest"
                        )),
                        Err(ryft_pjrt::Error::NotFound { .. })
                    ));
                }
                // The actual loaded source semantics agree; only the validated compiler-binding identities differ.
                assert_eq!(identities[0][0], identities[1][0]);
                assert_ne!(identities[0][1], identities[1][1]);
            }
            "routing" => {
                let source = if process == 0 { 2 } else { 0 };
                let result = coordinator.call(vec![input.clone()], &[source], Arc::clone(&cancelled));
                if process == 0 {
                    assert!(matches!(result, Err(DistributedKernelError::Invalid { message })
                    if message == "input count or source-process routing differs from the kernel boundary"));
                } else {
                    assert!(matches!(result, Err(DistributedKernelError::Participant { process: 0, .. })));
                }
            }
            "order" => {
                if process == 0 {
                    runtime.reserve_kernel_coordinator().unwrap();
                }
                let mut next = DistributedKernel::new(&runtime, &kernel, options).unwrap();
                assert!(matches!(
                    next.call(vec![input.clone()], &[process], Arc::clone(&cancelled)),
                    Err(DistributedKernelError::Deadline)
                ));
            }
            "late_failure" => {
                let pending =
                    coordinator.call_async(vec![input.clone()], &[1 - process], Arc::clone(&cancelled)).unwrap();
                if process == 1 {
                    let ready = runtime
                        .key_value_store()
                        .get(&coordinator.coordination.key(0, 0, "terminal"), Duration::from_secs(5))
                        .unwrap();
                    assert!(serde_json::from_slice::<KernelTerminal>(&ready).unwrap().failure.is_none());
                    cancelled.store(true, Ordering::Release);
                }
                assert!(matches!(pending.r#await(), Err(DistributedKernelError::Completion { .. })));
                let ready = runtime
                    .key_value_store()
                    .get(&coordinator.coordination.key(0, 0, "terminal"), Duration::from_secs(5))
                    .unwrap();
                assert!(serde_json::from_slice::<KernelTerminal>(&ready).unwrap().failure.is_none());
            }
            "drop" => {
                let pending =
                    coordinator.call_async(vec![input.clone()], &[1 - process], Arc::clone(&cancelled)).unwrap();
                if process == 0 {
                    drop(pending);
                    drop(coordinator);
                } else {
                    assert!(matches!(pending.r#await(), Err(DistributedKernelError::Completion { .. })));
                }
            }
            _ => panic!("unexpected process mode"),
        }
        assert_eq!(scalar(&input), 10 + process as i32);
        runtime
            .key_value_store()
            .put(format!("test-kernel-finished-{process}").as_bytes(), b"done")
            .unwrap();
        assert_eq!(
            runtime
                .key_value_store()
                .get(format!("test-kernel-finished-{}", 1 - process).as_bytes(), Duration::from_secs(10))
                .unwrap(),
            b"done"
        );
    }

    #[test]
    fn test_validate_kernel_participants() {
        let client = execution_client();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(Device::from_pjrt)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("device", devices.len(), MeshAxisType::Auto).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        assert!(validate_kernel_participants(&client, &mesh).is_ok());
    }

    #[test]
    fn test_validate_kernel_participants_multiple_local_devices() {
        let client = load_cpu_plugin()
            .unwrap()
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(Device::from_pjrt)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(devices.len(), 2);
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("device", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        assert!(validate_kernel_participants(&client, &mesh).is_ok());
    }

    #[test]
    fn test_validate_kernel_participants_remote_process() {
        let client = execution_client();
        let device = client.addressable_devices().unwrap().remove(0);
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![]).unwrap(),
            vec![Device::new(device.id().unwrap(), client.process_index().unwrap() + 1)],
        )
        .unwrap();
        let expected =
            "cross-process kernel execution requires a qualified collective ordering and completion contract";
        assert!(matches!(validate_kernel_participants(&client, &mesh),
            Err(KernelEmbeddingError::Invalid { message }) if message == expected));
    }

    #[test]
    fn test_validate_kernel_participants_nonaddressable_device() {
        let client = execution_client();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![]).unwrap(),
            vec![Device::new(usize::MAX, client.process_index().unwrap())],
        )
        .unwrap();
        let expected = "kernel execution requires every mesh device to be addressable by the submitting client";
        assert!(matches!(validate_kernel_participants(&client, &mesh),
            Err(KernelEmbeddingError::Invalid { message }) if message == expected));
    }
}
