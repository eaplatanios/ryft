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

use ryft_core::compilation::{CompilationArtifactExchange, CompilationExchangeError};
use ryft_pjrt::{
    Client, ClientOptions, DistributedKeyValueStore, DistributedRuntimeClientOptions, DistributedRuntimeService,
    DistributedRuntimeServiceOptions, KeyValueStore, Plugin,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const DEFAULT_ARTIFACT_CHUNK_SIZE: usize = 4 * 1024 * 1024;
const DEFAULT_MAX_ARTIFACT_SIZE: usize = 2 * 1024 * 1024 * 1024;
const DEFAULT_MAX_PREFLIGHT_KEY_SIZE: usize = 64 * 1024 * 1024;
const LAUNCH_ID_WAIT_TIMEOUT: Duration = Duration::from_secs(60);
const ARTIFACT_EXCHANGE_SCHEMA_VERSION: u32 = 2;
const LAUNCH_ID_KEY: &[u8] = b"ryft/distributed-launch/v1/id";

#[derive(Serialize, Deserialize)]
struct CompilationArtifactManifest {
    schema_version: u32,
    size: u64,
    chunk_count: u64,
    checksum: [u8; 32],
    failure: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct CompilationPreflightManifest {
    schema_version: u32,
    sequence: u64,
    process_index: u64,
    process_count: u64,
    key_size: u64,
    key_checksum: [u8; 32],
}

struct DistributedRuntimeState {
    key_value_store: DistributedKeyValueStore,
    launch_id: [u8; 32],
    _service: Option<DistributedRuntimeService>,
}

struct DistributedCompilationArtifactExchange {
    runtime: Arc<DistributedRuntimeState>,
    process_index: usize,
    process_count: usize,
    chunk_size: usize,
    maximum_artifact_size: usize,
    maximum_preflight_key_size: usize,
    preflight_sequence: AtomicU64,
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

    fn preflight_key(&self, sequence: u64, process_index: usize, suffix: &[u8]) -> Vec<u8> {
        let mut key = b"ryft/compiled-artifact-preflight/v1/".to_vec();
        key.extend_from_slice(&self.runtime.launch_id);
        key.extend_from_slice(format!("/{sequence:016x}/{process_index:016x}/").as_bytes());
        key.extend_from_slice(suffix);
        key
    }

    fn failed(message: impl Into<String>) -> CompilationExchangeError {
        CompilationExchangeError::Failed { message: message.into() }
    }
}

impl CompilationArtifactExchange for DistributedCompilationArtifactExchange {
    fn process_index(&self) -> usize {
        self.process_index
    }

    fn process_count(&self) -> usize {
        self.process_count
    }

    fn preflight(&self, key: &[u8], timeout: Duration) -> Result<(), CompilationExchangeError> {
        if self.process_count == 0 || self.process_index >= self.process_count {
            return Err(CompilationExchangeError::Incompatible {
                message: "exchange process coordinates are invalid".to_string(),
            });
        }
        if key.len() > self.maximum_preflight_key_size {
            return Err(CompilationExchangeError::Incompatible {
                message: format!(
                    "persistent compilation key size {} exceeds configured preflight maximum {}",
                    key.len(),
                    self.maximum_preflight_key_size,
                ),
            });
        }

        let sequence = self.preflight_sequence.fetch_add(1, Ordering::Relaxed);
        self.runtime
            .key_value_store
            .put(&self.preflight_key(sequence, self.process_index, b"key"), key)
            .map_err(|error| Self::failed(error.to_string()))?;
        let manifest = CompilationPreflightManifest {
            schema_version: ARTIFACT_EXCHANGE_SCHEMA_VERSION,
            sequence,
            process_index: self.process_index as u64,
            process_count: self.process_count as u64,
            key_size: key.len() as u64,
            key_checksum: Sha256::digest(key).into(),
        };
        let manifest = serde_json::to_vec(&manifest)
            .map_err(|error| Self::failed(format!("failed to encode compilation preflight manifest: {error}")))?;
        self.runtime
            .key_value_store
            .put(&self.preflight_key(sequence, self.process_index, b"manifest"), manifest.as_slice())
            .map_err(|error| Self::failed(error.to_string()))?;

        let start = Instant::now();
        for process_index in 0..self.process_count {
            let remaining = timeout.checked_sub(start.elapsed()).ok_or(CompilationExchangeError::TimedOut)?;
            let manifest = self
                .runtime
                .key_value_store
                .get(&self.preflight_key(sequence, process_index, b"manifest"), remaining)
                .map_err(|error| match error {
                    ryft_pjrt::Error::NotFound { .. } | ryft_pjrt::Error::DeadlineExceeded { .. } => {
                        CompilationExchangeError::TimedOut
                    }
                    error => Self::failed(error.to_string()),
                })?;
            let manifest: CompilationPreflightManifest = serde_json::from_slice(manifest.as_slice())
                .map_err(|error| Self::failed(format!("failed to decode compilation preflight manifest: {error}")))?;
            if manifest.schema_version != ARTIFACT_EXCHANGE_SCHEMA_VERSION
                || manifest.sequence != sequence
                || manifest.process_index != process_index as u64
                || manifest.process_count != self.process_count as u64
            {
                return Err(CompilationExchangeError::Incompatible {
                    message: format!("process {process_index} reported incompatible compilation coordinates"),
                });
            }
            if manifest.key_size != key.len() as u64 || manifest.key_checksum != <[u8; 32]>::from(Sha256::digest(key)) {
                return Err(CompilationExchangeError::Incompatible {
                    message: format!("process {process_index} reported a different persistent compilation key"),
                });
            }
            let remaining = timeout.checked_sub(start.elapsed()).ok_or(CompilationExchangeError::TimedOut)?;
            let participant_key = self
                .runtime
                .key_value_store
                .get(&self.preflight_key(sequence, process_index, b"key"), remaining)
                .map_err(|error| match error {
                    ryft_pjrt::Error::NotFound { .. } | ryft_pjrt::Error::DeadlineExceeded { .. } => {
                        CompilationExchangeError::TimedOut
                    }
                    error => Self::failed(error.to_string()),
                })?;
            if participant_key.as_slice() != key {
                return Err(CompilationExchangeError::Incompatible {
                    message: format!("process {process_index} reported a different persistent compilation key"),
                });
            }
        }
        Ok(())
    }

    fn publish(&self, key: &[u8], artifact: &[u8]) -> Result<(), CompilationExchangeError> {
        if artifact.len() > self.maximum_artifact_size {
            return Err(Self::failed(format!(
                "compiled artifact size {} exceeds configured maximum {}",
                artifact.len(),
                self.maximum_artifact_size,
            )));
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

    fn publish_failure(&self, key: &[u8], message: &str) -> Result<(), CompilationExchangeError> {
        let manifest = CompilationArtifactManifest {
            schema_version: ARTIFACT_EXCHANGE_SCHEMA_VERSION,
            size: 0,
            chunk_count: 0,
            checksum: Sha256::digest(b"").into(),
            failure: Some(message.to_string()),
        };
        let manifest = serde_json::to_vec(&manifest)
            .map_err(|error| Self::failed(format!("failed to encode artifact failure: {error}")))?;
        self.runtime
            .key_value_store
            .put(&self.key(key, b"manifest"), manifest.as_slice())
            .map_err(|error| Self::failed(error.to_string()))
    }

    fn receive(&self, key: &[u8], timeout: Duration) -> Result<Option<Vec<u8>>, CompilationExchangeError> {
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
            let remaining = timeout.checked_sub(start.elapsed()).ok_or(CompilationExchangeError::TimedOut)?;
            let chunk =
                self.runtime.key_value_store.get(&self.chunk_key(key, index), remaining).map_err(
                    |error| match error {
                        ryft_pjrt::Error::NotFound { .. } | ryft_pjrt::Error::DeadlineExceeded { .. } => {
                            CompilationExchangeError::TimedOut
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
///   * a per-process [`DistributedRuntimeClient`] (wrapped in a [`DistributedKeyValueStore`]) that
///     speaks to the coordinator service over the configured address, and
///   * on the coordinator node only, the [`DistributedRuntimeService`] that hosts the
///     coordination state for every participant.
///
/// PJRT [`ryft_pjrt::Client`]s minted via [`Self::create_client`] borrow the handle's key-value
/// store, so the handle must outlive every such client.
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
    /// [`DistributedRuntimeClient`]. The call blocks (subject to the configured
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
        // panic-on-heartbeat-miss can install their own via [`Self::initialize_with_options`].
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
        // Coordinator (node 0) hosts the service so workers have something to dial into.
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
        Ok(Self { state: Arc::new(DistributedRuntimeState { key_value_store, launch_id, _service: service }) })
    }

    /// Returns the [`DistributedKeyValueStore`] backed by this runtime. Useful for callers that
    /// want to mint a [`ryft_pjrt::Client`] manually via
    /// [`ryft_pjrt::Plugin::client_with_key_value_store`].
    #[inline]
    pub fn key_value_store(&self) -> &DistributedKeyValueStore {
        &self.state.key_value_store
    }

    /// Creates a chunked, checksummed compilation-artifact exchange over this runtime's coordination store.
    ///
    /// The returned exchange keeps both the distributed client and coordinator service alive. A manifest is published
    /// only after every bounded chunk, so followers never accept a partially published artifact. Before compilation,
    /// every process participates in an ordered preflight round that compares the exact persistent compilation key and
    /// process count. The XLA persistent key includes platform and plugin versions, compiler identity, and logical
    /// topology, so those compatibility properties are checked before process zero compiles. Distributed compilation
    /// calls must occur in the same order on every process; launch-order disagreement reaches the bounded preflight
    /// timeout instead of waiting indefinitely.
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
            maximum_preflight_key_size: DEFAULT_MAX_PREFLIGHT_KEY_SIZE,
            preflight_sequence: AtomicU64::new(0),
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
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::process::{Command, Stdio};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::thread;
    use std::time::{Duration, Instant};

    use ryft_core::compilation::CompilationArtifactExchange;
    use ryft_pjrt::{ClientOptions, CpuClientOptions, KeyValueStore, load_cpu_plugin};
    use sha2::{Digest, Sha256};

    use super::{ARTIFACT_EXCHANGE_SCHEMA_VERSION, CompilationArtifactManifest};
    use super::{DistributedCompilationArtifactExchange, DistributedRuntime};

    const PROCESS_TEST_ROLE: &str = "RYFT_DISTRIBUTED_EXCHANGE_TEST_ROLE";
    const PROCESS_TEST_ADDRESS: &str = "RYFT_DISTRIBUTED_EXCHANGE_TEST_ADDRESS";
    const PROCESS_TEST_DIRECTORY: &str = "RYFT_DISTRIBUTED_EXCHANGE_TEST_DIRECTORY";
    const PROCESS_TEST_ACKNOWLEDGEMENT: &[u8] = b"ryft/test/artifact-consumed";

    /// Returns a `127.0.0.1` URL with an OS-assigned free port, or `None` if local-port binding
    /// is denied in this environment.
    fn loopback_address() -> Option<String> {
        let listener = TcpListener::bind("127.0.0.1:0").ok()?;
        Some(format!("127.0.0.1:{}", listener.local_addr().unwrap().port()))
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
            .create_client(&plugin, ClientOptions::CPU(CpuClientOptions { device_count: Some(1) }))
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
            maximum_preflight_key_size: 16,
            preflight_sequence: AtomicU64::new(0),
        };

        exchange.publish(b"program", b"abcdefgh").unwrap();
        assert_eq!(exchange.receive(b"program", Duration::from_secs(1)).unwrap(), Some(b"abcdefgh".to_vec()));
        exchange.publish_failure(b"failed-program", "backend compilation failed").unwrap();
        assert!(matches!(
            exchange.receive(b"failed-program", Duration::from_secs(1)),
            Err(ryft_core::compilation::CompilationExchangeError::Failed { message })
                if message.contains("backend compilation failed"),
        ));
        assert!(exchange.publish(b"too-large", &[0; 17]).is_err());
    }

    #[test]
    fn test_compilation_artifact_preflight_and_single_producer_path() {
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
                leader.preflight(b"complete-xla-persistent-key", Duration::from_secs(1)).unwrap();
                producer_calls.fetch_add(1, Ordering::Relaxed);
                leader.publish(b"complete-xla-persistent-key", b"serialized-executable").unwrap();
            });
            let follower = Arc::clone(&follower);
            let consumer = scope.spawn(move || {
                follower.preflight(b"complete-xla-persistent-key", Duration::from_secs(1)).unwrap();
                follower.receive(b"complete-xla-persistent-key", Duration::from_secs(1)).unwrap().unwrap()
            });

            producer.join().unwrap();
            assert_eq!(consumer.join().unwrap(), b"serialized-executable");
        });
        assert_eq!(producer_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_compilation_artifact_preflight_rejects_key_disagreement() {
        let Some(address) = loopback_address() else {
            return;
        };
        let plugin = load_cpu_plugin().unwrap();
        let runtime = DistributedRuntime::initialize(&plugin, &address, 1, 0).unwrap();
        let first = runtime.compilation_artifact_exchange(0, 2);
        let second = runtime.compilation_artifact_exchange(1, 2);

        thread::scope(|scope| {
            let first = scope.spawn(move || first.preflight(b"first-key", Duration::from_secs(1)));
            let second = scope.spawn(move || second.preflight(b"second-key", Duration::from_secs(1)));
            for result in [first.join().unwrap(), second.join().unwrap()] {
                assert!(matches!(result, Err(ryft_core::compilation::CompilationExchangeError::Incompatible { .. })));
            }
        });
    }

    #[test]
    fn test_compilation_artifact_preflight_rejects_process_count_disagreement() {
        let Some(address) = loopback_address() else {
            return;
        };
        let plugin = load_cpu_plugin().unwrap();
        let runtime = DistributedRuntime::initialize(&plugin, &address, 1, 0).unwrap();
        let first = runtime.compilation_artifact_exchange(0, 2);
        let second = runtime.compilation_artifact_exchange(1, 3);

        thread::scope(|scope| {
            let first = scope.spawn(move || first.preflight(b"key", Duration::from_secs(1)));
            let second = scope.spawn(move || second.preflight(b"key", Duration::from_secs(1)));
            for result in [first.join().unwrap(), second.join().unwrap()] {
                assert!(matches!(result, Err(ryft_core::compilation::CompilationExchangeError::Incompatible { .. })));
            }
        });
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
            let leader = scope.spawn(move || leader.preflight(b"key", Duration::from_secs(1)));
            let follower = scope.spawn(move || {
                follower.preflight(b"key", Duration::from_secs(1)).unwrap();
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
            maximum_preflight_key_size: 16,
            preflight_sequence: AtomicU64::new(0),
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
        exchange.preflight(b"multi-process-key", Duration::from_secs(5)).unwrap();

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
}
