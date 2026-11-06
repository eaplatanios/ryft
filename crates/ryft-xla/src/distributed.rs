//! High-level distributed-runtime setup for `ryft-xla`.
//!
//! [`DistributedRuntime`] is the `ryft` analogue of `jax.distributed.initialize`: it composes
//! [`ryft_pjrt::Plugin::distributed_runtime_service`],
//! [`ryft_pjrt::Plugin::distributed_runtime_client`], and
//! [`ryft_pjrt::DistributedKeyValueStore`] into a single entry point. The handle owns the
//! coordinator service (on the elected coordinator node) and the per-process distributed runtime
//! client; subsequent [`ryft_pjrt::Client`] construction routes through the bundled key-value
//! store so PJRT cross-host primitives have the rendezvous surface they need.
//!
//! ## Usage
//!
//! Each participating process calls [`DistributedRuntime::initialize`] with identical
//! `coordinator_address` and `num_nodes` values but a unique `node_id` (0 designates the
//! coordinator). The call blocks until every node has joined. After that, use
//! [`DistributedRuntime::create_client`] to mint a PJRT [`ryft_pjrt::Client`] that participates
//! in cross-host transfers and SPMD compilation.
//!
//! Initialization also establishes a coordinator-generated launch identity. Compilation-artifact rendezvous keys are
//! scoped to that identity, so a later job cannot observe terminal records or chunks from an earlier launch.
//!
//! Drop order matters: keep the [`DistributedRuntime`] alive at least as long as the
//! [`ryft_pjrt::Client`] it produced, because the client borrows the runtime's key-value store.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ryft_core::{CompilationArtifactExchange, CompilationArtifactExchangeError};
use ryft_pjrt::{
    Client, ClientOptions, DistributedKeyValueStore, DistributedRuntimeClientOptions, DistributedRuntimeService,
    DistributedRuntimeServiceOptions, KeyValueStore, Plugin,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const DEFAULT_ARTIFACT_CHUNK_SIZE: usize = 4 * 1024 * 1024;
const DEFAULT_MAX_ARTIFACT_SIZE: usize = 2 * 1024 * 1024 * 1024;
const DEFAULT_MAX_COMPATIBILITY_KEY_SIZE: usize = 64 * 1024 * 1024;
const LAUNCH_ID_WAIT_TIMEOUT: Duration = Duration::from_secs(60);
const ARTIFACT_EXCHANGE_SCHEMA_VERSION: u32 = 3;
const LAUNCH_ID_KEY: &[u8] = b"ryft/distributed-launch/v1/id";

#[derive(Serialize, Deserialize)]
struct CompilationArtifactManifest {
    schema_version: u32,
    size: u64,
    chunk_count: u64,
    checksum: [u8; 32],
    failure: Option<String>,
}

/// Announcement that the leader publishes during compatibility validation for one persistent compilation key.
#[derive(Serialize, Deserialize)]
struct CompilationCompatibilityManifest {
    /// Exchange protocol version of the leader.
    schema_version: u32,

    /// Number of processes that the leader expects to participate in the exchange.
    process_count: u64,
}

struct DistributedRuntimeState {
    key_value_store: DistributedKeyValueStore,
    launch_id: [u8; 32],
    kernel_sequence: AtomicU64,
    process_index: u32,
    process_count: u32,
    _service: Option<DistributedRuntimeService>,
}

struct DistributedCompilationArtifactExchange {
    runtime: Arc<DistributedRuntimeState>,
    process_index: usize,
    process_count: usize,
    chunk_size: usize,
    maximum_artifact_size: usize,
    maximum_compatibility_key_size: usize,
}

impl DistributedCompilationArtifactExchange {
    fn key(&self, key: &[u8], suffix: &[u8]) -> Vec<u8> {
        let digest = Sha256::digest(key);
        let mut exchange_key = b"ryft/compiled-artifact/v2/".to_vec();
        exchange_key.extend_from_slice(&self.runtime.launch_id);
        exchange_key.push(b'/');
        exchange_key.extend_from_slice(&digest);
        exchange_key.push(b'/');
        exchange_key.extend_from_slice(suffix);
        exchange_key
    }

    fn chunk_key(&self, key: &[u8], index: usize) -> Vec<u8> {
        self.key(key, format!("chunk/{index:016x}").as_bytes())
    }

    /// Stores `value` under `key` unless a value is already stored there, in which case the stored value is kept.
    ///
    /// The coordination store never overwrites values, and the leader legitimately writes the same record more than
    /// once, for example when it compiles a program again after evicting it from its in-memory cache.
    fn put_if_absent(&self, key: &[u8], value: &[u8]) -> Result<(), CompilationArtifactExchangeError> {
        match self.runtime.key_value_store.put(key, value) {
            Ok(()) | Err(ryft_pjrt::Error::AlreadyExists { .. }) => Ok(()),
            Err(error) => Err(Self::failed(error.to_string())),
        }
    }

    fn failed(message: impl Into<String>) -> CompilationArtifactExchangeError {
        CompilationArtifactExchangeError::Failed { message: message.into() }
    }
}

impl CompilationArtifactExchange for DistributedCompilationArtifactExchange {
    fn process_index(&self) -> usize {
        self.process_index
    }

    fn process_count(&self) -> usize {
        self.process_count
    }

    fn validate_compatibility(&self, key: &[u8], timeout: Duration) -> Result<(), CompilationArtifactExchangeError> {
        // Address compatibility validation by the compilation key because each process validates only compilations
        // that miss its own in-memory cache. The leader announces every key that it is about to compile without
        // waiting for anyone, and each follower waits for the announcement of its own key.
        if self.process_count == 0 || self.process_index >= self.process_count {
            return Err(CompilationArtifactExchangeError::Incompatible {
                message: "exchange process coordinates are invalid".to_string(),
            });
        }
        if key.len() > self.maximum_compatibility_key_size {
            return Err(CompilationArtifactExchangeError::Incompatible {
                message: format!(
                    "persistent compilation key size {} exceeds configured compatibility validation maximum {}",
                    key.len(),
                    self.maximum_compatibility_key_size,
                ),
            });
        }

        if self.process_index == 0 {
            let manifest = CompilationCompatibilityManifest {
                schema_version: ARTIFACT_EXCHANGE_SCHEMA_VERSION,
                process_count: self.process_count as u64,
            };
            let manifest = serde_json::to_vec(&manifest).map_err(|error| {
                Self::failed(format!("failed to encode compilation compatibility manifest: {error}"))
            })?;
            // The key is written before the manifest so that a follower that observes the manifest also finds the key.
            self.put_if_absent(&self.key(key, b"preflight/key"), key)?;
            return self.put_if_absent(&self.key(key, b"preflight/manifest"), manifest.as_slice());
        }

        let start = Instant::now();
        let get = |suffix: &[u8]| -> Result<Vec<u8>, CompilationArtifactExchangeError> {
            let remaining = timeout.checked_sub(start.elapsed()).ok_or(CompilationArtifactExchangeError::TimedOut)?;
            self.runtime.key_value_store.get(&self.key(key, suffix), remaining).map_err(|error| match error {
                ryft_pjrt::Error::NotFound { .. } | ryft_pjrt::Error::DeadlineExceeded { .. } => {
                    CompilationArtifactExchangeError::TimedOut
                }
                error => Self::failed(error.to_string()),
            })
        };
        let manifest = get(b"preflight/manifest")?;
        let manifest: CompilationCompatibilityManifest = serde_json::from_slice(manifest.as_slice())
            .map_err(|error| Self::failed(format!("failed to decode compilation compatibility manifest: {error}")))?;
        if manifest.schema_version != ARTIFACT_EXCHANGE_SCHEMA_VERSION {
            return Err(CompilationArtifactExchangeError::Incompatible {
                message: format!(
                    "process 0 uses compilation artifact exchange schema {}, but this process uses schema {}",
                    manifest.schema_version, ARTIFACT_EXCHANGE_SCHEMA_VERSION,
                ),
            });
        }
        if manifest.process_count != self.process_count as u64 {
            return Err(CompilationArtifactExchangeError::Incompatible {
                message: format!(
                    "process 0 expects {} processes, but this process expects {}",
                    manifest.process_count, self.process_count,
                ),
            });
        }
        if get(b"preflight/key")?.as_slice() != key {
            return Err(CompilationArtifactExchangeError::Incompatible {
                message: "process 0 announced a different persistent compilation key with the same digest".to_string(),
            });
        }
        Ok(())
    }

    fn publish(&self, key: &[u8], artifact: &[u8]) -> Result<(), CompilationArtifactExchangeError> {
        if artifact.len() > self.maximum_artifact_size {
            return Err(Self::failed(format!(
                "compiled artifact size {} exceeds configured maximum {}",
                artifact.len(),
                self.maximum_artifact_size,
            )));
        }
        // Every artifact published for one key is interchangeable, so a key whose artifact is already published (e.g.,
        // because the leader compiled it again after evicting it) needs no new publication. A terminal failure that
        // was already published for the key cannot be replaced, because the store never overwrites values.
        match self.runtime.key_value_store.try_get(&self.key(key, b"manifest")) {
            Ok(manifest) => {
                let manifest: CompilationArtifactManifest = serde_json::from_slice(manifest.as_slice())
                    .map_err(|error| Self::failed(format!("failed to decode artifact manifest: {error}")))?;
                return match manifest.failure {
                    None => Ok(()),
                    Some(_) => Err(Self::failed("a terminal failure was already published for this compilation key")),
                };
            }
            Err(ryft_pjrt::Error::NotFound { .. }) => {}
            Err(error) => return Err(Self::failed(error.to_string())),
        }
        for (index, chunk) in artifact.chunks(self.chunk_size).enumerate() {
            self.runtime
                .key_value_store
                .put(&self.chunk_key(key, index), chunk)
                .map_err(|error| Self::failed(error.to_string()))?;
        }
        let manifest = CompilationArtifactManifest {
            schema_version: ARTIFACT_EXCHANGE_SCHEMA_VERSION,
            size: artifact.len() as u64,
            chunk_count: artifact.len().div_ceil(self.chunk_size) as u64,
            checksum: Sha256::digest(artifact).into(),
            failure: None,
        };
        let manifest = serde_json::to_vec(&manifest)
            .map_err(|error| Self::failed(format!("failed to encode artifact manifest: {error}")))?;
        self.runtime
            .key_value_store
            .put(&self.key(key, b"manifest"), manifest.as_slice())
            .map_err(|error| Self::failed(error.to_string()))
    }

    fn publish_failure(&self, key: &[u8], message: &str) -> Result<(), CompilationArtifactExchangeError> {
        let manifest = CompilationArtifactManifest {
            schema_version: ARTIFACT_EXCHANGE_SCHEMA_VERSION,
            size: 0,
            chunk_count: 0,
            checksum: Sha256::digest(b"").into(),
            failure: Some(message.to_string()),
        };
        let manifest = serde_json::to_vec(&manifest)
            .map_err(|error| Self::failed(format!("failed to encode artifact failure: {error}")))?;
        // An artifact or failure that was already published for this key stays authoritative.
        self.put_if_absent(&self.key(key, b"manifest"), manifest.as_slice())
    }

    fn receive(&self, key: &[u8], timeout: Duration) -> Result<Option<Vec<u8>>, CompilationArtifactExchangeError> {
        let start = Instant::now();
        let manifest = match self.runtime.key_value_store.get(&self.key(key, b"manifest"), timeout) {
            Ok(manifest) => manifest,
            Err(ryft_pjrt::Error::NotFound { .. } | ryft_pjrt::Error::DeadlineExceeded { .. }) => return Ok(None),
            Err(error) => return Err(Self::failed(error.to_string())),
        };
        let manifest: CompilationArtifactManifest = serde_json::from_slice(manifest.as_slice())
            .map_err(|error| Self::failed(format!("failed to decode artifact manifest: {error}")))?;
        if manifest.schema_version != ARTIFACT_EXCHANGE_SCHEMA_VERSION {
            return Err(Self::failed(format!(
                "unsupported compiled artifact exchange schema {}",
                manifest.schema_version,
            )));
        }
        if let Some(message) = manifest.failure {
            return Err(Self::failed(format!("compilation producer failed: {message}")));
        }
        let size =
            usize::try_from(manifest.size).map_err(|_| Self::failed("compiled artifact size overflows usize"))?;
        let chunk_count = usize::try_from(manifest.chunk_count)
            .map_err(|_| Self::failed("compiled artifact chunk count overflows usize"))?;
        if size > self.maximum_artifact_size || chunk_count != size.div_ceil(self.chunk_size) {
            return Err(Self::failed("compiled artifact manifest exceeds configured bounds"));
        }
        let mut artifact = Vec::with_capacity(size);
        for index in 0..chunk_count {
            let remaining = timeout.checked_sub(start.elapsed()).ok_or(CompilationArtifactExchangeError::TimedOut)?;
            let chunk =
                self.runtime.key_value_store.get(&self.chunk_key(key, index), remaining).map_err(
                    |error| match error {
                        ryft_pjrt::Error::NotFound { .. } | ryft_pjrt::Error::DeadlineExceeded { .. } => {
                            CompilationArtifactExchangeError::TimedOut
                        }
                        error => Self::failed(error.to_string()),
                    },
                )?;
            if chunk.len() > self.chunk_size || artifact.len().saturating_add(chunk.len()) > size {
                return Err(Self::failed("compiled artifact chunk violates manifest bounds"));
            }
            artifact.extend_from_slice(chunk.as_slice());
        }
        if artifact.len() != size || <[u8; 32]>::from(Sha256::digest(artifact.as_slice())) != manifest.checksum {
            return Err(Self::failed("compiled artifact checksum mismatch"));
        }
        Ok(Some(artifact))
    }
}

/// One-process handle for a distributed `ryft-xla` job.
///
/// Construct via [`Self::initialize`]. The handle owns:
///
///   * a per-process [`DistributedRuntimeClient`](ryft_pjrt::DistributedRuntimeClient) (wrapped in a [`DistributedKeyValueStore`]) that
///     speaks to the coordinator service over the configured address, and
///   * on the coordinator node only, the [`DistributedRuntimeService`] that hosts the
///     coordination state for every participant.
///
/// PJRT [`ryft_pjrt::Client`]s minted via [`Self::create_client`] borrow the handle's key-value
/// store, so the handle must outlive every such client.
#[derive(Clone)]
pub struct DistributedRuntime {
    state: Arc<DistributedRuntimeState>,
}

impl DistributedRuntime {
    /// Initializes a [`DistributedRuntime`] for a single participating process. Mirrors
    /// `jax.distributed.initialize`.
    ///
    /// Every participating process in a job must call this with identical `coordinator_address`
    /// and `num_nodes`, and a unique `node_id` in `0..num_nodes`. Node 0 is the elected
    /// coordinator: it hosts the [`DistributedRuntimeService`] in addition to its own
    /// [`DistributedRuntimeClient`](ryft_pjrt::DistributedRuntimeClient). The call blocks (subject to the configured
    /// `initialization_timeout`) until all nodes have connected.
    ///
    /// # Parameters
    ///
    ///   - `plugin`: PJRT plugin used to spin up the service / client.
    ///   - `coordinator_address`: TCP address every node connects to (for example
    ///     `"10.0.0.1:24680"`). Must be reachable from every participant.
    ///   - `num_nodes`: Total participant count across the job.
    ///   - `node_id`: Unique participant ID in `0..num_nodes`. Node 0 is the coordinator.
    ///
    /// # Notes on tuning
    ///
    /// This entry point uses the default [`DistributedRuntimeServiceOptions`] and
    /// [`DistributedRuntimeClientOptions`] (plus the requested `num_nodes` / `node_id`). When
    /// non-default heartbeat or RPC timeouts matter, use
    /// [`Self::initialize_with_options`] instead.
    pub fn initialize(
        plugin: &Plugin,
        coordinator_address: &str,
        num_nodes: u32,
        node_id: u32,
    ) -> Result<Self, ryft_pjrt::Error> {
        let service_options = DistributedRuntimeServiceOptions { num_nodes, ..Default::default() };
        // The default client `missed_heartbeat_callback` panics, which causes orderly shutdowns
        // to abort the process when the polling RPC observes the service tearing down before
        // the client's own poll loop has wound up. Swap in a silent callback: callers that want
        // panic-on-heartbeat-miss can install their own via `Self::initialize_with_options`.
        let client_options = DistributedRuntimeClientOptions {
            node_id,
            missed_heartbeat_callback: Some(Box::new(|_| {})),
            ..Default::default()
        };
        Self::initialize_with_options(plugin, coordinator_address, node_id, service_options, client_options)
    }

    /// Same as [`Self::initialize`] but accepts explicit [`DistributedRuntimeServiceOptions`]
    /// and [`DistributedRuntimeClientOptions`]. Use this when the defaults' heartbeat / RPC /
    /// shutdown timeouts don't fit your job's profile.
    ///
    /// `service_options.num_nodes` and `client_options.node_id` are honored as provided — pass
    /// them in consistently with `node_id`.
    pub fn initialize_with_options(
        plugin: &Plugin,
        coordinator_address: &str,
        node_id: u32,
        service_options: DistributedRuntimeServiceOptions,
        client_options: DistributedRuntimeClientOptions,
    ) -> Result<Self, ryft_pjrt::Error> {
        if node_id != client_options.node_id || service_options.num_nodes == 0 || node_id >= service_options.num_nodes {
            return Err(ryft_pjrt::Error::invalid_argument("distributed runtime participant coordinates disagree"));
        }
        // Coordinator (node 0) hosts the service so workers have something to dial into.
        let process_count = service_options.num_nodes;
        let service = if node_id == 0 {
            Some(plugin.distributed_runtime_service(coordinator_address, service_options)?)
        } else {
            None
        };
        let client = plugin.distributed_runtime_client(coordinator_address, client_options)?;
        client.connect()?;
        let key_value_store = DistributedKeyValueStore::new(client);
        let launch_id = if node_id == 0 {
            let mut identity = Sha256::new();
            identity.update(coordinator_address.as_bytes());
            identity.update(std::process::id().to_le_bytes());
            identity.update(SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos().to_le_bytes());
            let launch_id: [u8; 32] = identity.finalize().into();
            key_value_store.put(LAUNCH_ID_KEY, launch_id.as_slice())?;
            launch_id
        } else {
            key_value_store
                .get(LAUNCH_ID_KEY, LAUNCH_ID_WAIT_TIMEOUT)?
                .try_into()
                .map_err(|_| ryft_pjrt::Error::invalid_argument("distributed launch identity has the wrong length"))?
        };
        Ok(Self {
            state: Arc::new(DistributedRuntimeState {
                key_value_store,
                launch_id,
                kernel_sequence: AtomicU64::new(0),
                process_index: node_id,
                process_count,
                _service: service,
            }),
        })
    }

    /// Returns the [`DistributedKeyValueStore`] backed by this runtime. Useful for callers that
    /// want to mint a [`ryft_pjrt::Client`] manually via
    /// [`ryft_pjrt::Plugin::client_with_key_value_store`].
    #[inline]
    pub fn key_value_store(&self) -> &DistributedKeyValueStore {
        &self.state.key_value_store
    }

    /// Reserves one ordered kernel coordinator namespace within the existing distributed launch.
    pub(crate) fn reserve_kernel_coordinator(&self) -> Result<([u8; 32], u64, u32, u32), ryft_pjrt::Error> {
        let sequence = self
            .state
            .kernel_sequence
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |value| value.checked_add(1))
            .map_err(|_| ryft_pjrt::Error::invalid_argument("kernel coordinator sequence overflow"))?;
        Ok((self.state.launch_id, sequence, self.state.process_index, self.state.process_count))
    }

    /// Creates a chunked, checksummed compilation-artifact exchange over this runtime's coordination store.
    ///
    /// The returned exchange keeps both the distributed client and coordinator service alive. A manifest is published
    /// only after every bounded chunk, so followers never accept a partially published artifact. Every record is
    /// addressed by the persistent compilation key, so processes may compile programs in different orders, skip
    /// programs that their in-memory caches already hold, or compile programs that no other process compiles (e.g.,
    /// eager operations that only process zero runs).
    ///
    /// Before process zero compiles a program, [`CompilationArtifactExchange::validate_compatibility`] announces the
    /// persistent compilation key and the process count without waiting for any other process. A follower that misses
    /// its in-memory cache waits for that announcement before it waits for the artifact, and rejects a process count
    /// that disagrees with its own. The XLA persistent key includes platform and plugin versions, compiler identity,
    /// and logical topology, so a follower whose compatibility properties differ from those of process zero never
    /// observes an announcement for its key. Its compatibility validation then times out, and the configured exchange
    /// policy decides whether it compiles locally.
    pub fn compilation_artifact_exchange(
        &self,
        process_index: usize,
        process_count: usize,
    ) -> Arc<dyn CompilationArtifactExchange> {
        Arc::new(DistributedCompilationArtifactExchange {
            runtime: Arc::clone(&self.state),
            process_index,
            process_count,
            chunk_size: DEFAULT_ARTIFACT_CHUNK_SIZE,
            maximum_artifact_size: DEFAULT_MAX_ARTIFACT_SIZE,
            maximum_compatibility_key_size: DEFAULT_MAX_COMPATIBILITY_KEY_SIZE,
        })
    }

    /// Mints a PJRT [`Client`] bound to this runtime's key-value store. The returned client
    /// shares the runtime's lifetime: it remains valid as long as `self` is alive and is
    /// invalidated when the runtime is dropped.
    pub fn create_client<'s>(
        &'s self,
        plugin: &Plugin,
        options: ClientOptions,
    ) -> Result<Client<'s>, ryft_pjrt::Error> {
        plugin.client_with_key_value_store(options, &self.state.key_value_store)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda-13")]
    use std::fs::File;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::process::{Command, Stdio};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use std::time::{Duration, Instant};

    #[cfg(feature = "cuda-13")]
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    #[cfg(feature = "cuda-13")]
    use ryft_core::ParallelBroadcast;
    use ryft_core::{
        ArrayIrType, ArrayReference, ArrayType, BatchAxis, CompilationArtifactExchange,
        CompilationArtifactExchangeError, DataType, Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType,
        ParallelReduce, ReductionKind, ReferenceType, ShardMapOptions, ShardMapTracer, Sharding, ShardingDimension,
        UnreducedKind, ValueProjection, batch, shard_map,
    };
    #[cfg(feature = "cuda-13")]
    use ryft_pjrt::{
        BufferType, ExecutionDeviceInputs, ExecutionInput, GpuClientOptions, GpuMemoryAllocator, GpuPlatform,
        Program as PjrtProgram, load_cuda_13_plugin,
    };
    use ryft_pjrt::{
        Client, ClientOptions, CpuClientOptions, DistributedRuntimeClientOptions, DistributedRuntimeServiceOptions,
        KeyValueStore, load_cpu_plugin,
    };
    use sha2::{Digest, Sha256};

    use crate::arrays_v0::host::materialize_dense_array_bytes;
    use crate::compilation::compile_statefully;
    use crate::compilation::tests::doubling_reduced_reference_program;
    #[cfg(feature = "cuda-13")]
    use crate::experimental::lowering::tests::test_spmd_compilation_options;
    use crate::tests::{values_from_bytes, values_to_bytes};
    use crate::{Array, ArrayError, Domain, Error, FromPjrt, Options, Reference, Session, Value};

    use super::{
        ARTIFACT_EXCHANGE_SCHEMA_VERSION, CompilationArtifactManifest, DistributedCompilationArtifactExchange,
        DistributedRuntime,
    };

    const PROCESS_TEST_ROLE: &str = "RYFT_DISTRIBUTED_EXCHANGE_TEST_ROLE";
    const PROCESS_TEST_ADDRESS: &str = "RYFT_DISTRIBUTED_EXCHANGE_TEST_ADDRESS";
    const PROCESS_TEST_DIRECTORY: &str = "RYFT_DISTRIBUTED_EXCHANGE_TEST_DIRECTORY";
    const PROCESS_TEST_ACKNOWLEDGEMENT: &[u8] = b"ryft/test/artifact-consumed";
    #[cfg(feature = "cuda-13")]
    const MULTI_HOST_TEST_ROLE: &str = "RYFT_DISTRIBUTED_MULTI_HOST_TEST_ROLE";
    #[cfg(feature = "cuda-13")]
    const MULTI_HOST_TEST_ADDRESS: &str = "RYFT_DISTRIBUTED_MULTI_HOST_TEST_ADDRESS";

    /// Returns a `127.0.0.1` URL with an OS-assigned free port, or `None` if local-port binding
    /// is denied in this environment.
    fn loopback_address() -> Option<String> {
        let listener = TcpListener::bind("127.0.0.1:0").ok()?;
        Some(format!("127.0.0.1:{}", listener.local_addr().unwrap().port()))
    }

    /// Returns the elements of the buffer of every shard of `array` that the current process addresses, in mesh order.
    fn read_addressable_f32s(array: &Array<'_>) -> Vec<Vec<f32>> {
        array
            .addressable_shards()
            .map(|shard| {
                let bytes = shard.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
                values_from_bytes::<f32>(bytes.as_slice())
            })
            .collect()
    }

    /// Checks `shard_map` pipelines over a mesh `x = 2` of the two global devices of `client`, reading only the shards
    /// that the current process addresses, so that the same checks hold for one process that addresses both devices
    /// and for two processes that address one device each (as JAX's multi-controller model requires). Every process of
    /// a job must call this function, because each scenario executes collectives over both devices.
    fn check_shard_map_multi_host_scenarios<'c>(client: &'c Client<'c>) {
        let process_index = client.process_index().unwrap();
        let devices = client
            .devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(devices.len(), 2);
        // Mesh positions of the devices that this process addresses select the expected per-device values below.
        let local_positions = devices
            .iter()
            .enumerate()
            .filter(|(_, device)| device.process_index() == process_index)
            .map(|(position, _)| position)
            .collect::<Vec<_>>();
        let domain = Session::new(client).domain();
        let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(manual_mesh.clone(), devices.clone()).unwrap();
        let sharded = Sharding::new(manual_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(manual_mesh.clone(), 1);

        // Each process uploads only its own tiles of `[1, 2, 3, 4]` and reads them back.
        let input_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap();
        let input =
            Array::from_host_buffer(&domain, input_type, mesh.clone(), values_to_bytes(&[1f32, 2., 3., 4.])).unwrap();
        let tiles = [vec![1.0, 2.0], vec![3.0, 4.0]];
        assert_eq!(
            read_addressable_f32s(&input),
            local_positions.iter().map(|&position| tiles[position].clone()).collect::<Vec<_>>(),
        );

        // Dense host materialization of a sharded array needs every tile, so it succeeds only in a process that addresses
        // both devices and otherwise names a shard of the other process before issuing any copy, as JAX rejects fetching
        // arrays whose value spans non-addressable devices.
        if local_positions.len() == 2 {
            let materialized = materialize_dense_array_bytes(&input).unwrap();
            assert_eq!(values_from_bytes::<f32>(materialized.as_slice()), vec![1.0, 2.0, 3.0, 4.0]);
        } else {
            let remote_position = 1 - local_positions[0];
            assert!(matches!(
                materialize_dense_array_bytes(&input),
                Err(Error::Array(error)) if matches!(
                    *error,
                    ArrayError::MissingShardBuffer { shard_index, device_id }
                        if shard_index == remote_position && device_id == devices[remote_position].id(),
                ),
            ));
        }

        // An unchecked map returns each tile as its candidate for an output that is unreduced along `x`, and a checked
        // map completes it with an all-reduce of the matching kind, so every device holds the sum `[4, 6]` (or the
        // maximum `[3, 4]`).
        let complete = |kind: UnreducedKind, reduction: ReductionKind| {
            let unreduced = replicated.clone().with_unreduced_axes(["x"]).unwrap().with_unreduced_kind(kind).unwrap();
            let candidates = shard_map(
                |x| Ok(x),
                input.clone(),
                manual_mesh.clone(),
                sharded.clone(),
                unreduced.clone(),
                ShardMapOptions::new().with_check_manual_variation(false),
            )
            .unwrap();
            assert_eq!(candidates.sharding(), &unreduced);
            shard_map(
                |candidates: ShardMapTracer<Domain<'_>>| Ok(candidates.parallel_reduce(reduction, "x")?),
                candidates,
                manual_mesh.clone(),
                unreduced,
                replicated.clone(),
                ShardMapOptions::new(),
            )
            .unwrap()
        };
        let sum = complete(UnreducedKind::Sum, ReductionKind::Sum);
        assert_eq!(read_addressable_f32s(&sum), vec![vec![4.0, 6.0]; local_positions.len()]);
        let maximum = complete(UnreducedKind::Max, ReductionKind::Max);
        assert_eq!(read_addressable_f32s(&maximum), vec![vec![3.0, 4.0]; local_positions.len()]);

        // A replicated array spans every process, but each process holds its whole value, so dense host reads succeed
        // in every process, as JAX reads a value that the current process holds in full.
        let materialized = materialize_dense_array_bytes(&sum).unwrap();
        assert_eq!(values_from_bytes::<f32>(materialized.as_slice()), vec![4.0, 6.0]);

        // Batching a map along its only manual axis for a caller over an `Explicit` copy of the mesh places one item on
        // each device, so each device holds its own item doubled.
        let caller_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let explicit_mesh = DeviceMesh::new(caller_mesh.clone(), devices.clone()).unwrap();
        let items_type = ArrayType::new_static(DataType::F32, [2, 3])
            .with_sharding(
                Sharding::new(caller_mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        let staged = crate::stage::<_, ArrayType, ArrayType>(
            |input| {
                let output = batch(
                    |item| {
                        let item = ValueProjection::<ArrayType>::into_projected(item)?;
                        let output = shard_map(
                            |x| Ok(x.clone() + x),
                            item,
                            manual_mesh.clone(),
                            replicated.clone(),
                            replicated.clone(),
                            ShardMapOptions::new(),
                        )?;
                        Ok(output.into_value())
                    },
                    input.into_value(),
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )
                .unwrap();
                ValueProjection::<ArrayType>::into_projected(output).unwrap()
            },
            items_type.clone(),
            Options::new(explicit_mesh.clone()),
        )
        .unwrap()
        .into_inner();
        let compiled = staged.lower(&domain).unwrap().compile(&domain).unwrap();
        let items =
            Array::from_host_buffer(&domain, items_type, explicit_mesh, values_to_bytes(&[1f32, 2., 3., 4., 5., 6.]))
                .unwrap();
        let Value::Array(doubled) = compiled.executable().execute(&domain, Value::Array(items)).unwrap() else {
            panic!("the compiled function returned a dimension");
        };
        let doubled_items = [vec![2.0, 4.0, 6.0], vec![8.0, 10.0, 12.0]];
        assert_eq!(
            read_addressable_f32s(&doubled),
            local_positions.iter().map(|&position| doubled_items[position].clone()).collect::<Vec<_>>(),
        );

        // An unchecked stateful map doubles every device's copy of a reference that is reduced along `x` and writes it
        // back, so each process reads `[2, 4]` from its own shards of the referent afterwards. Over a mesh that spans
        // processes, each process calls with its own reference, binds and commits only its addressable shards, and,
        // because the referent is replicated, can read the whole final state densely.
        let (program, referent_type) = doubling_reduced_reference_program(&manual_mesh);
        let compiled = compile_statefully::<_, Vec<ArrayIrType>, Vec<ArrayIrType>>(
            |inputs| {
                let context = inputs[0].context().clone();
                program.interpret_in_context(&context, inputs)?;
                Ok(Vec::new())
            },
            vec![ArrayIrType::Reference(ReferenceType::new(referent_type.clone()))],
            &domain,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let reference = ArrayReference::new(
            Array::from_host_buffer(&domain, referent_type, mesh, values_to_bytes(&[1f32, 2.0]).as_slice()).unwrap(),
        );
        compiled
            .execute_statefully_and_wait(
                &domain,
                vec![Value::Reference(Reference::new(reference.clone(), domain.clone()))],
            )
            .unwrap();
        assert_eq!(read_addressable_f32s(&reference.read().unwrap()), vec![vec![2.0, 4.0]; local_positions.len()]);
        let materialized = materialize_dense_array_bytes(&reference.read().unwrap()).unwrap();
        assert_eq!(values_from_bytes::<f32>(materialized.as_slice()), vec![2.0, 4.0]);
    }

    #[test]
    fn test_initialize_with_options_participant_identity() {
        let plugin = load_cpu_plugin().unwrap();
        assert!(matches!(DistributedRuntime::initialize_with_options(
            &plugin, "unused.invalid:1", 0,
            DistributedRuntimeServiceOptions { num_nodes: 2, ..Default::default() },
            DistributedRuntimeClientOptions { node_id: 1, ..Default::default() },
        ), Err(ryft_pjrt::Error::InvalidArgument { message, .. })
            if message == "distributed runtime participant coordinates disagree"));
    }

    #[test]
    fn test_initialize_single_node_runtime_returns_usable_client() {
        let Some(address) = loopback_address() else {
            return;
        };
        let plugin = load_cpu_plugin().unwrap();
        // num_nodes=1 / node_id=0: same process is both coordinator and worker. `connect()`
        // returns as soon as the coordinator observes its lone participant.
        let runtime = DistributedRuntime::initialize(&plugin, &address, 1, 0).unwrap();

        // Verify the key-value store is operational by round-tripping a write.
        runtime.key_value_store().put(b"hello", b"world").unwrap();
        let observed = runtime.key_value_store().try_get(b"hello").unwrap();
        assert_eq!(observed, b"world");

        // Verify we can mint a PJRT client through the runtime. The client borrows the
        // runtime's KV store, so it must be dropped before the runtime falls out of scope.
        let client = runtime
            .create_client(
                &plugin,
                ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }),
            )
            .unwrap();
        assert!(!client.addressable_devices().unwrap().is_empty());
    }

    #[test]
    fn test_compilation_artifact_exchange_chunks_and_validates_payloads() {
        let Some(address) = loopback_address() else {
            return;
        };
        let plugin = load_cpu_plugin().unwrap();
        let runtime = DistributedRuntime::initialize(&plugin, &address, 1, 0).unwrap();
        let exchange = DistributedCompilationArtifactExchange {
            runtime: Arc::clone(&runtime.state),
            process_index: 0,
            process_count: 2,
            chunk_size: 3,
            maximum_artifact_size: 16,
            maximum_compatibility_key_size: 16,
        };

        exchange.publish(b"program", b"abcdefgh").unwrap();
        assert_eq!(exchange.receive(b"program", Duration::from_secs(1)).unwrap(), Some(b"abcdefgh".to_vec()));
        exchange.publish_failure(b"failed-program", "backend compilation failed").unwrap();
        assert!(matches!(
            exchange.receive(b"failed-program", Duration::from_secs(1)),
            Err(CompilationArtifactExchangeError::Failed { message })
                if message.contains("backend compilation failed"),
        ));
        assert!(exchange.publish(b"too-large", &[0; 17]).is_err());
    }

    #[test]
    fn test_compilation_artifact_exchange_validate_compatibility_and_single_producer_path() {
        let Some(address) = loopback_address() else {
            return;
        };
        let plugin = load_cpu_plugin().unwrap();
        let runtime = DistributedRuntime::initialize(&plugin, &address, 1, 0).unwrap();
        let leader = runtime.compilation_artifact_exchange(0, 2);
        let follower = runtime.compilation_artifact_exchange(1, 2);
        let producer_calls = Arc::new(AtomicUsize::new(0));

        thread::scope(|scope| {
            let producer_calls = Arc::clone(&producer_calls);
            let leader = Arc::clone(&leader);
            let producer = scope.spawn(move || {
                leader.validate_compatibility(b"complete-xla-persistent-key", Duration::from_secs(1)).unwrap();
                producer_calls.fetch_add(1, Ordering::Relaxed);
                leader.publish(b"complete-xla-persistent-key", b"serialized-executable").unwrap();
            });
            let follower = Arc::clone(&follower);
            let consumer = scope.spawn(move || {
                follower.validate_compatibility(b"complete-xla-persistent-key", Duration::from_secs(1)).unwrap();
                follower.receive(b"complete-xla-persistent-key", Duration::from_secs(1)).unwrap().unwrap()
            });

            producer.join().unwrap();
            assert_eq!(consumer.join().unwrap(), b"serialized-executable");
        });
        assert_eq!(producer_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_compilation_artifact_exchange_validate_compatibility_times_out_on_key_disagreement() {
        let Some(address) = loopback_address() else {
            return;
        };
        let plugin = load_cpu_plugin().unwrap();
        let runtime = DistributedRuntime::initialize(&plugin, &address, 1, 0).unwrap();
        let leader = runtime.compilation_artifact_exchange(0, 2);
        let follower = runtime.compilation_artifact_exchange(1, 2);

        // The leader never waits, while a follower whose key differs never observes an announcement for its key.
        assert_eq!(leader.validate_compatibility(b"first-key", Duration::from_secs(1)), Ok(()));
        assert_eq!(
            follower.validate_compatibility(b"second-key", Duration::from_millis(20)),
            Err(CompilationArtifactExchangeError::TimedOut),
        );
    }

    #[test]
    fn test_compilation_artifact_exchange_validate_compatibility_is_independent_of_call_order() {
        let Some(address) = loopback_address() else {
            return;
        };
        let plugin = load_cpu_plugin().unwrap();
        let runtime = DistributedRuntime::initialize(&plugin, &address, 1, 0).unwrap();
        let leader = runtime.compilation_artifact_exchange(0, 2);
        let follower = runtime.compilation_artifact_exchange(1, 2);

        // The leader compiles programs that the follower skips (e.g., because its in-memory cache already holds them),
        // compiles one program twice (e.g., after evicting it), and the follower validates compatibility in a different
        // order.
        for key in [b"first-key".as_slice(), b"second-key", b"third-key", b"first-key"] {
            assert_eq!(leader.validate_compatibility(key, Duration::from_secs(1)), Ok(()));
        }
        assert_eq!(follower.validate_compatibility(b"third-key", Duration::from_secs(1)), Ok(()));
        assert_eq!(follower.validate_compatibility(b"first-key", Duration::from_secs(1)), Ok(()));

        // Publishing an already published artifact again is a no-op that keeps the first artifact.
        leader.publish(b"first-key", b"first-executable").unwrap();
        leader.publish(b"first-key", b"recompiled-executable").unwrap();
        let received = follower.receive(b"first-key", Duration::from_secs(1)).unwrap();
        assert_eq!(received, Some(b"first-executable".to_vec()));

        // A published terminal failure stays authoritative for its key.
        leader.publish_failure(b"second-key", "backend compilation failed").unwrap();
        leader.publish_failure(b"second-key", "backend compilation failed again").unwrap();
        assert_eq!(
            leader.publish(b"second-key", b"second-executable"),
            Err(CompilationArtifactExchangeError::Failed {
                message: "a terminal failure was already published for this compilation key".to_string(),
            }),
        );
        assert_eq!(
            follower.receive(b"second-key", Duration::from_secs(1)),
            Err(CompilationArtifactExchangeError::Failed {
                message: "compilation producer failed: backend compilation failed".to_string(),
            }),
        );
    }

    #[test]
    fn test_compilation_artifact_exchange_validate_compatibility_rejects_process_count_disagreement() {
        let Some(address) = loopback_address() else {
            return;
        };
        let plugin = load_cpu_plugin().unwrap();
        let runtime = DistributedRuntime::initialize(&plugin, &address, 1, 0).unwrap();
        let leader = runtime.compilation_artifact_exchange(0, 2);
        let follower = runtime.compilation_artifact_exchange(1, 3);

        assert_eq!(leader.validate_compatibility(b"key", Duration::from_secs(1)), Ok(()));
        assert_eq!(
            follower.validate_compatibility(b"key", Duration::from_secs(1)),
            Err(CompilationArtifactExchangeError::Incompatible {
                message: "process 0 expects 2 processes, but this process expects 3".to_string(),
            }),
        );
    }

    #[test]
    fn test_compilation_artifact_receive_times_out_after_producer_disappears() {
        let Some(address) = loopback_address() else {
            return;
        };
        let plugin = load_cpu_plugin().unwrap();
        let runtime = DistributedRuntime::initialize(&plugin, &address, 1, 0).unwrap();
        let leader = runtime.compilation_artifact_exchange(0, 2);
        let follower = runtime.compilation_artifact_exchange(1, 2);

        thread::scope(|scope| {
            let leader = scope.spawn(move || leader.validate_compatibility(b"key", Duration::from_secs(1)));
            let follower = scope.spawn(move || {
                follower.validate_compatibility(b"key", Duration::from_secs(1)).unwrap();
                follower.receive(b"key", Duration::from_millis(20))
            });
            leader.join().unwrap().unwrap();
            assert_eq!(follower.join().unwrap().unwrap(), None);
        });
    }

    #[test]
    fn test_compilation_artifact_exchange_rejects_corrupt_and_reordered_chunks() {
        let Some(address) = loopback_address() else {
            return;
        };
        let plugin = load_cpu_plugin().unwrap();
        let runtime = DistributedRuntime::initialize(&plugin, &address, 1, 0).unwrap();
        let exchange = DistributedCompilationArtifactExchange {
            runtime: Arc::clone(&runtime.state),
            process_index: 0,
            process_count: 2,
            chunk_size: 3,
            maximum_artifact_size: 16,
            maximum_compatibility_key_size: 16,
        };

        let publish_invalid = |key: &[u8], first: &[u8], second: &[u8]| {
            runtime.key_value_store().put(&exchange.chunk_key(key, 0), first).unwrap();
            runtime.key_value_store().put(&exchange.chunk_key(key, 1), second).unwrap();
            let manifest = serde_json::to_vec(&CompilationArtifactManifest {
                schema_version: ARTIFACT_EXCHANGE_SCHEMA_VERSION,
                size: 6,
                chunk_count: 2,
                checksum: Sha256::digest(b"abcdef").into(),
                failure: None,
            })
            .unwrap();
            runtime.key_value_store().put(&exchange.key(key, b"manifest"), manifest.as_slice()).unwrap();
        };

        publish_invalid(b"corrupt", b"xbc", b"def");
        assert!(exchange.receive(b"corrupt", Duration::from_secs(1)).is_err());

        publish_invalid(b"reordered", b"def", b"abc");
        assert!(exchange.receive(b"reordered", Duration::from_secs(1)).is_err());
    }

    #[test]
    fn test_compilation_artifact_exchange_process_helper() {
        let Ok(role) = std::env::var(PROCESS_TEST_ROLE) else {
            return;
        };
        let process_index = role.parse::<usize>().unwrap();
        let address = std::env::var(PROCESS_TEST_ADDRESS).unwrap();
        let directory = std::path::PathBuf::from(std::env::var(PROCESS_TEST_DIRECTORY).unwrap());
        let plugin = load_cpu_plugin().unwrap();
        let runtime = DistributedRuntime::initialize(&plugin, &address, 2, process_index as u32).unwrap();
        let exchange = runtime.compilation_artifact_exchange(process_index, 2);
        exchange.validate_compatibility(b"multi-process-key", Duration::from_secs(5)).unwrap();

        if process_index == 0 {
            let mut producers =
                std::fs::OpenOptions::new().create(true).append(true).open(directory.join("producers")).unwrap();
            producers.write_all(b"producer\n").unwrap();
            exchange.publish(b"multi-process-key", b"serialized-executable").unwrap();
            runtime.key_value_store().get(PROCESS_TEST_ACKNOWLEDGEMENT, Duration::from_secs(5)).unwrap();
        } else {
            assert_eq!(
                exchange.receive(b"multi-process-key", Duration::from_secs(5)).unwrap().unwrap(),
                b"serialized-executable",
            );
            runtime.key_value_store().put(PROCESS_TEST_ACKNOWLEDGEMENT, b"ready").unwrap();
        }
        std::fs::write(directory.join(format!("process-{process_index}")), b"complete").unwrap();
    }

    #[test]
    fn test_compilation_artifact_exchange_coordinates_two_processes() {
        let Some(address) = loopback_address() else {
            return;
        };
        let directory = tempfile::tempdir().unwrap();
        let executable = std::env::current_exe().unwrap();
        let mut children = (0..2)
            .map(|process_index| {
                Command::new(&executable)
                    .args([
                        "--exact",
                        "distributed::tests::test_compilation_artifact_exchange_process_helper",
                        "--nocapture",
                        "--test-threads=1",
                    ])
                    .env(PROCESS_TEST_ROLE, process_index.to_string())
                    .env(PROCESS_TEST_ADDRESS, &address)
                    .env(PROCESS_TEST_DIRECTORY, directory.path())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if children.iter_mut().all(|child| child.try_wait().unwrap().is_some()) {
                break;
            }
            if Instant::now() >= deadline {
                for child in &mut children {
                    if child.try_wait().unwrap().is_none() {
                        child.kill().expect("distributed test child should be killable after timeout");
                    }
                }
                panic!("distributed artifact exchange process test timed out");
            }
            thread::sleep(Duration::from_millis(10));
        }
        for mut child in children {
            let status = child.wait().unwrap();
            let mut stderr = String::new();
            child.stderr.take().unwrap().read_to_string(&mut stderr).unwrap();
            assert!(status.success(), "distributed test child failed: {stderr}");
        }

        assert_eq!(std::fs::read_to_string(directory.path().join("producers")).unwrap(), "producer\n");
        assert_eq!(std::fs::read(directory.path().join("process-0")).unwrap(), b"complete");
        assert_eq!(std::fs::read(directory.path().join("process-1")).unwrap(), b"complete");
    }

    #[test]
    fn test_shard_map_multi_host_scenarios_on_one_process_with_two_cpu_devices() {
        // The single-process baseline of the multi-host scenarios, which reads every shard because this process
        // addresses both devices. The client is minted through a one-node runtime, as in a distributed job.
        let Some(address) = loopback_address() else {
            return;
        };
        let plugin = load_cpu_plugin().unwrap();
        let runtime = DistributedRuntime::initialize(&plugin, &address, 1, 0).unwrap();
        let client = runtime
            .create_client(
                &plugin,
                ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }),
            )
            .unwrap();
        check_shard_map_multi_host_scenarios(&client);
    }

    /// Joins the two-process CUDA job that launched this process as rank `process_index` (refer to
    /// [`run_cuda_rank_pair`]), runs `check` with a client whose two global devices are the ranks' shares of CUDA device
    /// 0, and then shuts the job down in order.
    #[cfg(feature = "cuda-13")]
    fn run_as_cuda_rank(process_index: usize, check: impl FnOnce(&Client<'_>)) {
        let address = std::env::var(MULTI_HOST_TEST_ADDRESS).unwrap();
        let timeout = Duration::from_secs(60);
        let plugin = load_cuda_13_plugin().unwrap();
        let runtime = DistributedRuntime::initialize_with_options(
            &plugin,
            &address,
            process_index as u32,
            DistributedRuntimeServiceOptions {
                num_nodes: 2,
                heartbeat_timeout: timeout,
                cluster_register_timeout: timeout,
                shutdown_timeout: timeout,
            },
            DistributedRuntimeClientOptions {
                node_id: process_index as u32,
                rpc_timeout: timeout,
                initialization_timeout: timeout,
                shutdown_timeout: timeout,
                heartbeat_timeout: timeout,
                missed_heartbeat_callback: Some(Box::new(|error| {
                    if let Some(error) = error {
                        eprintln!("distributed heartbeat failed: {error}");
                    }
                })),
                ..Default::default()
            },
        )
        .unwrap();

        // The client borrows the runtime's key-value store, so every array, executable, and the client are dropped
        // before the coordination client shuts down, while rank zero keeps the coordination service alive.
        {
            let client = runtime
                .create_client(
                    &plugin,
                    ClientOptions::GPU(GpuClientOptions {
                        platform: Some(GpuPlatform::CUDA),
                        visible_devices: Some(vec![0]),
                        node_id: Some(process_index),
                        node_count: Some(2),
                        allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                        collective_memory_size: Some(32 * 1024 * 1024),
                        abort_collectives_on_failure: true,
                        ..Default::default()
                    }),
                )
                .unwrap();
            assert_eq!(client.process_index().unwrap(), process_index);
            assert_eq!(client.devices().unwrap().len(), 2);
            assert_eq!(client.addressable_devices().unwrap().len(), 1);
            check(&client);
        }
        let completed = |rank: usize| format!("ryft/test/cuda-ranks/completed/{rank}");
        runtime.key_value_store().put(completed(process_index).as_bytes(), b"ready").unwrap();
        for rank in 0..2 {
            assert_eq!(runtime.key_value_store().get(completed(rank).as_bytes(), timeout).unwrap(), b"ready");
        }
        runtime.key_value_store().client().shutdown().unwrap();
    }

    /// Runs the test `helper` (a path within this test binary) as the two ranks of a CUDA job that share CUDA device 0,
    /// asserts that both ranks pass, and returns their standard outputs, or returns `None` when local-port binding is
    /// denied in this environment.
    ///
    /// The CPU plugin cannot host such jobs, because a CPU client that joins a distributed runtime sees only its own
    /// devices (e.g., process 1 sees just device 2048), so it has no cross-process topology for collectives. Instead,
    /// both ranks share CUDA device 0 through NCCL's multiple ranks per GPU, with CUDA MPS for concurrent progress, which
    /// mirrors the `cuda-distributed-collectives` differential suite (refer to `python/README.md`).
    #[cfg(feature = "cuda-13")]
    fn run_cuda_rank_pair(helper: &str) -> Option<[String; 2]> {
        /// Private MPS control daemon, which is shut down on drop.
        struct MpsDaemon {
            environment: Vec<(&'static str, String)>,
        }

        impl Drop for MpsDaemon {
            fn drop(&mut self) {
                // Panicking here could abort a test that is already unwinding, so a failed shutdown is only reported.
                let shutdown = Command::new("nvidia-cuda-mps-control")
                    .envs(self.environment.iter().map(|(name, value)| (*name, value)))
                    .stdin(Stdio::piped())
                    .spawn()
                    .and_then(|mut control| {
                        control.stdin.take().unwrap().write_all(b"quit\n")?;
                        control.wait()
                    });
                if !matches!(&shutdown, Ok(status) if status.success()) {
                    eprintln!("private CUDA MPS daemon shutdown failed: {shutdown:?}");
                }
            }
        }

        let address = loopback_address()?;
        let directory = tempfile::tempdir().unwrap();
        let mut environment = vec![
            ("NCCL_MULTI_RANK_GPU_ENABLE", "1".to_string()),
            ("NCCL_NVLS_ENABLE", "0".to_string()),
            ("NCCL_MAX_CTAS", "1".to_string()),
            ("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "45".to_string()),
            ("CUDA_VISIBLE_DEVICES", std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "0".to_string())),
        ];
        // A caller-owned MPS daemon is used as is. Otherwise, this test owns a private daemon for the two ranks.
        let _daemon = if std::env::var_os("CUDA_MPS_PIPE_DIRECTORY").is_some() {
            None
        } else {
            let pipes = directory.path().join("mps-pipes");
            let logs = directory.path().join("mps-logs");
            std::fs::create_dir(&pipes).unwrap();
            std::fs::create_dir(&logs).unwrap();
            let daemon_environment = vec![
                ("CUDA_MPS_PIPE_DIRECTORY", pipes.display().to_string()),
                ("CUDA_MPS_LOG_DIRECTORY", logs.display().to_string()),
                ("CUDA_VISIBLE_DEVICES", environment[4].1.clone()),
            ];
            let status = Command::new("nvidia-cuda-mps-control")
                .arg("-d")
                .envs(daemon_environment.iter().map(|(name, value)| (*name, value)))
                .status()
                .unwrap();
            assert!(status.success(), "private CUDA MPS daemon failed to start: {status}");
            environment.extend(daemon_environment[..2].iter().cloned());
            Some(MpsDaemon { environment: daemon_environment })
        };

        // Each rank writes its output to files rather than pipes, so verbose native logs cannot block it.
        let executable = std::env::current_exe().unwrap();
        let mut children = (0..2)
            .map(|rank| {
                Command::new(&executable)
                    .args(["--exact", helper, "--nocapture", "--test-threads=1"])
                    .envs(environment.iter().map(|(name, value)| (*name, value)))
                    .env(MULTI_HOST_TEST_ROLE, rank.to_string())
                    .env(MULTI_HOST_TEST_ADDRESS, &address)
                    .stdout(File::create(directory.path().join(format!("rank-{rank}.stdout"))).unwrap())
                    .stderr(File::create(directory.path().join(format!("rank-{rank}.stderr"))).unwrap())
                    .spawn()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let deadline = Instant::now() + Duration::from_secs(300);
        let mut statuses = [None, None];
        while statuses.iter().any(Option::is_none) && Instant::now() < deadline {
            for (status, child) in statuses.iter_mut().zip(&mut children) {
                if status.is_none() {
                    *status = child.try_wait().unwrap();
                }
            }
            // A failed rank leaves its peer waiting on collectives or coordination, so it ends the whole job.
            if statuses.iter().flatten().any(|status| !status.success()) {
                break;
            }
            thread::sleep(Duration::from_millis(20));
        }
        for (status, child) in statuses.iter_mut().zip(&mut children) {
            if status.is_none() {
                child.kill().unwrap();
                *status = Some(child.wait().unwrap());
            }
        }
        let outputs = [0, 1].map(|rank| {
            let status = statuses[rank].unwrap();
            let stderr = std::fs::read_to_string(directory.path().join(format!("rank-{rank}.stderr"))).unwrap();
            let stdout = std::fs::read_to_string(directory.path().join(format!("rank-{rank}.stdout"))).unwrap();
            assert!(status.success(), "rank {rank} failed with {status}\nstdout:\n{stdout}\nstderr:\n{stderr}");
            assert!(stdout.contains("1 passed"), "rank {rank} did not run `{helper}`:\n{stdout}");
            stdout
        });
        Some(outputs)
    }

    /// Worker of `test_shard_map_multi_host_scenarios_on_two_cuda_processes`, which returns immediately unless that
    /// test launched it as one of its two ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_shard_map_multi_host_scenarios_cuda_process_helper() {
        let Ok(role) = std::env::var(MULTI_HOST_TEST_ROLE) else {
            return;
        };
        run_as_cuda_rank(role.parse().unwrap(), |client| check_shard_map_multi_host_scenarios(client));
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_shard_map_multi_host_scenarios_on_two_cuda_processes() {
        // The multi-host scenarios run in two processes that address one device each.
        run_cuda_rank_pair("distributed::tests::test_shard_map_multi_host_scenarios_cuda_process_helper");
    }

    /// Worker of `test_gpu_only_collectives_on_two_cuda_processes`, which returns immediately unless that test launched
    /// it as one of its two ranks. It compiles and executes each probe module with this rank's tile of `[1, 2, 3, 4]`
    /// and prints one `probe:` line per module with its outputs or the first line of its compilation or execution
    /// error.
    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_gpu_only_collectives_cuda_process_helper() {
        // JAX's CUDA lowerings of `shard_map` over a mesh `x = 2` whose input `f32[4]` is sharded along `x`, produced
        // by `.tasks/shard_map_jax_parity_gaps/m8_lower.py` with JAX 0.12.1.dev20261103 over jaxlib 0.12.0.1 and kept
        // verbatim. They cover `pbroadcast` (`collective_broadcast`), the asynchronous `*-start`/`*-done` custom calls
        // whose configuration is in the `async_collective_config` frontend attribute, and token-threaded `send`/`recv`.
        // The send/recv modules run last, because a mismatched point-to-point transfer would hang both ranks.
        let send_recv = r##"
module @jit_send_recv attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=2]> {stablehlo.mesh = {axes = [{name = "x", size = 2 : i64}]}}
  func.func public @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<4xf32> {jax.result_info = "result", sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
      %2 = stablehlo.create_token : !stablehlo.token
      %3 = "stablehlo.send"(%arg1, %2) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<[[0, 1], [1, 0]]> : tensor<2x2xi64>}> : (tensor<2xf32>, !stablehlo.token) -> !stablehlo.token
      %4:2 = "stablehlo.recv"(%3) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<[[0, 1], [1, 0]]> : tensor<2x2xi64>}> : (!stablehlo.token) -> (tensor<2xf32>, !stablehlo.token)
      sdy.return %4#0 : tensor<2xf32>
    } : (tensor<4xf32>) -> tensor<4xf32>
    %1 = sdy.sharding_constraint %0 <@mesh, [{"x"}]> : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}
"##;
        // XLA's HLO verifier rejects the cyclic source-target pairs of the swap above, so a variant whose only pair is
        // `0 -> 1` (the module's only change) checks whether point-to-point transfers work at all.
        let cyclic_pairs = "dense<[[0, 1], [1, 0]]> : tensor<2x2xi64>";
        assert_eq!(send_recv.matches(cyclic_pairs).count(), 2);
        let acyclic_send_recv = send_recv.replace(cyclic_pairs, "dense<[[0, 1]]> : tensor<1x2xi64>");
        let modules = [
            (
                "pbroadcast",
                r##"
module @jit__lambda attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=2]> {stablehlo.mesh = {axes = [{name = "x", size = 2 : i64}]}}
  func.func public @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<4xf32> {jax.result_info = "result", sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
      %2 = "stablehlo.collective_broadcast"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[1, 0]]> : tensor<1x2xi64>}> : (tensor<2xf32>) -> tensor<2xf32>
      sdy.return %2 : tensor<2xf32>
    } : (tensor<4xf32>) -> tensor<4xf32>
    %1 = sdy.sharding_constraint %0 <@mesh, [{"x"}]> : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}
"##,
            ),
            (
                "ppermute_start_done",
                r##"
module @jit__lambda attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=2]> {stablehlo.mesh = {axes = [{name = "x", size = 2 : i64}]}}
  func.func public @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<4xf32> {jax.result_info = "result", sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
      %2 = stablehlo.custom_call @"collective-permute-start"(%arg1) {backend_config = "", mhlo.frontend_attributes = {async_collective_config = "{\22permutation\22: [[0, 1], [1, 0]], \22channel_id\22: 1}"}} : (tensor<2xf32>) -> tensor<2xf32>
      %3 = stablehlo.custom_call @"collective-permute-done"(%2) {backend_config = ""} : (tensor<2xf32>) -> tensor<2xf32>
      sdy.return %3 : tensor<2xf32>
    } : (tensor<4xf32>) -> tensor<4xf32>
    %1 = sdy.sharding_constraint %0 <@mesh, [{"x"}]> : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}
"##,
            ),
            (
                "all_gather_start_done",
                r##"
module @jit__lambda attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=2]> {stablehlo.mesh = {axes = [{name = "x", size = 2 : i64}]}}
  func.func public @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {jax.result_info = "result", sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
      %2 = stablehlo.custom_call @"all-gather-start"(%arg1) {backend_config = "", mhlo.frontend_attributes = {async_collective_config = "{\22all_gather_dimension\22: 0, \22replica_groups\22: [[0, 1]], \22use_global_device_ids\22: true, \22channel_id\22: 1}"}} : (tensor<2xf32>) -> tensor<4xf32>
      %3 = stablehlo.custom_call @"all-gather-done"(%2) {backend_config = ""} : (tensor<4xf32>) -> tensor<4xf32>
      sdy.return %3 : tensor<4xf32>
    } : (tensor<4xf32>) -> tensor<8xf32>
    %1 = sdy.sharding_constraint %0 <@mesh, [{"x"}]> : tensor<8xf32>
    return %1 : tensor<8xf32>
  }
}
"##,
            ),
            (
                "psum_scatter_start_done",
                r##"
module @jit__lambda attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=2]> {stablehlo.mesh = {axes = [{name = "x", size = 2 : i64}]}}
  func.func private @add_float32_reducer(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func public @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<2xf32> {jax.result_info = "result", sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
      %2 = stablehlo.custom_call @"reduce-scatter-start"(%arg1) {backend_config = "", called_computations = [@add_float32_reducer], mhlo.frontend_attributes = {async_collective_config = "{\22scatter_dimension\22: 0, \22replica_groups\22: [[0, 1]], \22tiled\22: true, \22use_global_device_ids\22: true, \22channel_id\22: 1}"}} : (tensor<2xf32>) -> tensor<1xf32>
      %3 = stablehlo.custom_call @"reduce-scatter-done"(%2) {backend_config = ""} : (tensor<1xf32>) -> tensor<1xf32>
      sdy.return %3 : tensor<1xf32>
    } : (tensor<4xf32>) -> tensor<2xf32>
    %1 = sdy.sharding_constraint %0 <@mesh, [{"x"}]> : tensor<2xf32>
    return %1 : tensor<2xf32>
  }
}
"##,
            ),
            (
                "all_to_all_start_done",
                r##"
module @jit__lambda attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=2]> {stablehlo.mesh = {axes = [{name = "x", size = 2 : i64}]}}
  func.func public @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<4xf32> {jax.result_info = "result", sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
      %2 = stablehlo.custom_call @"all-to-all-start"(%arg1) {backend_config = "", mhlo.frontend_attributes = {async_collective_config = "{\22split_dimension\22: 0, \22concat_dimension\22: 0, \22split_count\22: 2, \22replica_groups\22: [[0, 1]], \22channel_id\22: 1}"}} : (tensor<2xf32>) -> tensor<2xf32>
      %3 = stablehlo.custom_call @"all-to-all-done"(%2) {backend_config = ""} : (tensor<2xf32>) -> tensor<2xf32>
      sdy.return %3 : tensor<2xf32>
    } : (tensor<4xf32>) -> tensor<4xf32>
    %1 = sdy.sharding_constraint %0 <@mesh, [{"x"}]> : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}
"##,
            ),
            ("psend_precv", send_recv),
            ("psend_precv_acyclic", acyclic_send_recv.as_str()),
        ];
        let Ok(role) = std::env::var(MULTI_HOST_TEST_ROLE) else {
            return;
        };
        run_as_cuda_rank(role.parse().unwrap(), |client| {
            let rank = client.process_index().unwrap();
            let device = client.addressable_devices().unwrap().remove(0);
            for (name, module) in modules {
                let compiled = client.compile(
                    &PjrtProgram::Mlir { bytecode: module.trim_start().as_bytes().to_vec() },
                    &test_spmd_compilation_options(2),
                );
                let executable = match compiled {
                    Ok(executable) => executable,
                    Err(error) => {
                        // XLA's verifier messages end with the offending instruction, whose generated names change
                        // between XLA versions, so the probe keeps only the stable part of the first line.
                        let message = error.to_string();
                        let message = message.lines().next().unwrap_or("").split(", but found").next().unwrap();
                        println!("probe: rank {rank} {name}: compilation failed: {message}");
                        continue;
                    }
                };
                let tile = values_to_bytes(&[2.0 * rank as f32 + 1.0, 2.0 * rank as f32 + 2.0]);
                let buffer =
                    client.buffer(tile.as_slice(), BufferType::F32, [2u64], None, device.clone(), None).unwrap();
                let inputs = vec![ExecutionInput { buffer: Arc::new(buffer), donatable: false }];
                let executed = executable
                    .execute(
                        vec![ExecutionDeviceInputs { inputs: &inputs, ..Default::default() }],
                        Vec::new(),
                        0,
                        None,
                        None,
                        None,
                        None,
                        &[],
                    )
                    .and_then(|execution| execution.block_until_ready())
                    .and_then(|mut outputs| outputs.remove(0).outputs.remove(0).copy_to_host(None)?.r#await());
                match executed {
                    Ok(bytes) => {
                        println!("probe: rank {rank} {name}: {:?}", values_from_bytes::<f32>(bytes.as_slice()))
                    }
                    Err(error) => {
                        println!(
                            "probe: rank {rank} {name}: execution failed: {}",
                            error.to_string().lines().next().unwrap_or("")
                        )
                    }
                }
            }
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_gpu_only_collectives_on_two_cuda_processes() {
        // Pins what Ryft's pinned XLA supports of JAX's GPU-only collectives across two processes, as each rank observes
        // it. Following JAX's semantics, `pbroadcast` from source 1 gives both ranks `[3, 4]`, the swapping `ppermute`
        // gives `[3, 4]` and `[1, 2]`, the tiled all-gather gives both ranks `[1, 2, 3, 4]`, the tiled sum scatter of the
        // sum `[4, 6]` gives `[4]` and `[6]`, and the all-to-all gives `[1, 3]` and `[2, 4]`. XLA rejects the swapping
        // send/recv, whose pairs form a cycle, while the acyclic variant delivers `[1, 2]` to rank 1 and leaves the
        // received value of rank 0, which no pair targets, zero.
        let Some(outputs) = run_cuda_rank_pair("distributed::tests::test_gpu_only_collectives_cuda_process_helper")
        else {
            return;
        };
        // libtest prints `test <name> ... ` without a line break before the helper runs, so the first probe line of each
        // rank follows that prefix.
        let probes = outputs
            .iter()
            .flat_map(|output| output.lines().filter_map(|line| line.split_once("probe: ").map(|(_, probe)| probe)))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            probes,
            indoc! {"
                rank 0 pbroadcast: [3.0, 4.0]
                rank 0 ppermute_start_done: [3.0, 4.0]
                rank 0 all_gather_start_done: [1.0, 2.0, 3.0, 4.0]
                rank 0 psum_scatter_start_done: [4.0]
                rank 0 all_to_all_start_done: [1.0, 3.0]
                rank 0 psend_precv: compilation failed: during context [hlo verifier]: Expected send and recv \
                    instructions to have non-cyclical source-target pairs
                rank 0 psend_precv_acyclic: [0.0, 0.0]
                rank 1 pbroadcast: [3.0, 4.0]
                rank 1 ppermute_start_done: [1.0, 2.0]
                rank 1 all_gather_start_done: [1.0, 2.0, 3.0, 4.0]
                rank 1 psum_scatter_start_done: [6.0]
                rank 1 all_to_all_start_done: [2.0, 4.0]
                rank 1 psend_precv: compilation failed: during context [hlo verifier]: Expected send and recv \
                    instructions to have non-cyclical source-target pairs
                rank 1 psend_precv_acyclic: [1.0, 2.0]"},
        );
    }

    /// Worker of `test_parallel_broadcast_on_two_cuda_processes`, which returns immediately unless that test launched
    /// it as one of its two ranks. It stages a `shard_map` of `parallel_broadcast` over a mesh `x = 2` whose input
    /// `f32[4]` is sharded along `x`, lowers it for this CUDA client through Ryft, executes it with this rank's tile of
    /// `[1, 2, 3, 4]`, and prints a `probe:` line with this rank's output shard and the custom call targets of the
    /// lowered module.
    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_parallel_broadcast_cuda_process_helper() {
        let Ok(role) = std::env::var(MULTI_HOST_TEST_ROLE) else {
            return;
        };
        run_as_cuda_rank(role.parse().unwrap(), |client| {
            let rank = client.process_index().unwrap();
            let devices = client
                .devices()
                .unwrap()
                .iter()
                .map(|device| Device::from_pjrt(device).unwrap())
                .collect::<Vec<_>>();
            let domain = Session::new(client).domain();
            let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
            let mesh = DeviceMesh::new(manual_mesh.clone(), devices).unwrap();
            let sharded = Sharding::new(manual_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
            let input_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap();
            let input = Array::from_host_buffer(
                &domain,
                input_type.clone(),
                mesh.clone(),
                values_to_bytes(&[1f32, 2., 3., 4.]),
            )
            .unwrap();

            // Stages, lowers, compiles, and executes the `shard_map` of `$body` with the output sharding
            // `$output_sharding`, and prints its probe line.
            macro_rules! probe {
                ($name:literal, $output_sharding:expr, $body:expr $(,)?) => {{
                    let output_sharding = $output_sharding;
                    let staged = crate::stage::<_, ArrayType, ArrayType>(
                        |x| {
                            shard_map(
                                $body,
                                x,
                                manual_mesh.clone(),
                                sharded.clone(),
                                output_sharding.clone(),
                                ShardMapOptions::new(),
                            )
                            .unwrap()
                        },
                        input_type.clone(),
                        Options::new(mesh.clone()),
                    )
                    .unwrap()
                    .into_inner();
                    let lowered = staged.lower(&domain).unwrap();
                    let targets = lowered
                        .lowered_program()
                        .stable_hlo()
                        .split("stablehlo.custom_call @")
                        .skip(1)
                        .map(|call| call.split('(').next().unwrap().trim_matches('"').to_string())
                        .collect::<Vec<_>>();
                    let compiled = lowered.compile(&domain).unwrap();
                    let output = compiled.executable().execute(&domain, Value::Array(input.clone())).unwrap();
                    let Value::Array(output) = output else {
                        panic!("the compiled function returned a dimension");
                    };
                    println!("probe: rank {rank} {}: {:?} {:?}", $name, read_addressable_f32s(&output)[0], targets);
                }};
            }

            probe!("parallel_broadcast", sharded.clone(), |x| Ok(x.parallel_broadcast("x", 1)?));
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_parallel_broadcast_on_two_cuda_processes() {
        // Ryft's CUDA lowering of `parallel_broadcast` (i.e., `collective_broadcast`) executes across two processes
        // with JAX's values (refer to `test_gpu_only_collectives_on_two_cuda_processes`).
        let Some(outputs) = run_cuda_rank_pair("distributed::tests::test_parallel_broadcast_cuda_process_helper")
        else {
            return;
        };
        let probes = outputs
            .iter()
            .flat_map(|output| output.lines().filter_map(|line| line.split_once("probe: ").map(|(_, probe)| probe)))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            probes,
            indoc! {"
                rank 0 parallel_broadcast: [3.0, 4.0] []
                rank 1 parallel_broadcast: [3.0, 4.0] []"},
        );
    }
}
