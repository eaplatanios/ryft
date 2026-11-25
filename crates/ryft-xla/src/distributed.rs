//! Distributed runtime setup for `ryft-xla`.
//!
//! A distributed job consists of several processes that each own some devices and that cooperate on one program.
//! The processes find each other through a coordination service, which one of them hosts and all of them connect to,
//! and they exchange small records through the key-value store of that service (e.g., the rendezvous data of
//! cross-host transfers and collectives, compiled artifacts, and device inventories). [`DistributedRuntime`] is the
//! handle of one process on that service. It is the `ryft` analogue of `jax.distributed.initialize`.
//!
//! # Lifecycle
//!
//!   1. **Initialization:** Every process calls [`DistributedRuntime::initialize`] with the same coordinator address
//!      and process count and with its own process index. Process 0 is the coordinator, which hosts the service. The
//!      call returns once every process has connected. The coordinator then publishes a launch identity, which scopes
//!      the records of this launch, so a service that outlives a launch never serves stale records to the next one.
//!   2. **Sessions:** Each process builds its [`Session`](crate::Session) with
//!      [`SessionBuilder::distributed_runtime`](crate::SessionBuilder::distributed_runtime). The session then creates
//!      its PJRT [`Client`] through the key-value store of the runtime, with the process coordinates of the runtime in
//!      its client options, and it declares the devices of the other processes when the client does not list them.
//!      [`DistributedRuntime::create_client`] creates such a client without a session.
//!   3. **Shared Resources:** [`DistributedRuntime::compilation_artifact_exchange`] lets the processes share compiled
//!      artifacts, so that one process compiles each program and the others receive its result.
//!      [`DistributedRuntime::key_value_store`] serves other coordination protocols (e.g., the kernel coordination of
//!      [`crate::kernels`]).
//!   4. **Shutdown:** Dropping the last handle of a runtime disconnects its process and, on the coordinator, stops
//!      the service. A client borrows the key-value store of its runtime, so a session keeps a handle of its runtime
//!      for the rest of the process, as it keeps its client.
//!
//! The coordination service never overwrites or removes a record. Every protocol of this module therefore writes each
//! record once, and the records of a launch (including compiled artifacts) stay in the memory of the coordinator
//! until its service stops.
//!
//! # Example
//!
//! ```no_run
//! # use ryft_xla::{DistributedRuntime, Error, Platform, Session};
//! # fn build_session(process_count: usize, process_index: usize) -> Result<Session, Error> {
//! // Every process runs this with the same address and process count and with its own process index.
//! let plugin = Platform::Cuda.plugin()?;
//! let runtime = DistributedRuntime::initialize(&plugin, "10.0.0.1:24680", process_count, process_index)?;
//! Session::builder().platform(Platform::Cuda).distributed_runtime(runtime).build()
//! # }
//! ```

// TODO(eaplatanios): Review this module.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use ryft_core::{CompilationArtifactExchange, CompilationArtifactExchangeError, Device, DeviceId, ProcessIndex};
use ryft_pjrt::{
    Client, ClientOptions, DistributedKeyValueStore, DistributedRuntimeClientOptions, DistributedRuntimeService,
    DistributedRuntimeServiceOptions, KeyValueStore, Plugin,
};

use crate::{Error, FromPjrt};

/// Maximum number of artifact bytes that one key-value store record of a compilation artifact exchange holds.
const DEFAULT_ARTIFACT_CHUNK_SIZE: usize = 4 * 1024 * 1024;

/// Maximum size, in bytes, of a compilation artifact that an exchange publishes or receives.
const DEFAULT_MAXIMUM_ARTIFACT_SIZE: usize = 2 * 1024 * 1024 * 1024;

/// Maximum size, in bytes, of a persistent compilation key that compatibility validation accepts.
const DEFAULT_MAXIMUM_COMPATIBILITY_KEY_SIZE: usize = 64 * 1024 * 1024;

/// Duration for which [`DistributedRuntime::initialize`] keeps trying to connect to the coordination service, which
/// is the default of `jax.distributed.initialize`.
const DEFAULT_INITIALIZATION_TIMEOUT: Duration = Duration::from_secs(300);

/// Version of the compilation artifact exchange protocol, which every participant must share.
const ARTIFACT_EXCHANGE_SCHEMA_VERSION: u32 = 4;

/// Key-value store key under which the coordinator publishes the launch identity.
const LAUNCH_ID_KEY: &[u8] = b"ryft/distributed-launch/v1/id";

/// Handle of one process on the coordination service of a distributed `ryft-xla` job. Refer to the
/// [module documentation](self) for the lifecycle of a job.
///
/// The handle owns the [`DistributedRuntimeClient`](ryft_pjrt::DistributedRuntimeClient) of its process, wrapped in a
/// [`DistributedKeyValueStore`], and, on the coordinator only, the [`DistributedRuntimeService`] that hosts the
/// coordination state of every process. Clones share that state, which is released when the last of them is dropped.
///
/// A PJRT [`Client`] that [`Self::create_client`] returns borrows the key-value store of the handle, so the handle
/// must outlive it. [`SessionBuilder::distributed_runtime`](crate::SessionBuilder::distributed_runtime) takes care of
/// that for the client of a [`Session`](crate::Session), which lives for the rest of the process.
#[derive(Clone)]
pub struct DistributedRuntime {
    state: Arc<DistributedRuntimeState>,
}

impl DistributedRuntime {
    /// Initializes the [`DistributedRuntime`] of one process of a job, as `jax.distributed.initialize` does. Every
    /// process of the job must call this function with the same `coordinator_address` and `process_count` and with its
    /// own `process_index`. Process 0 is the coordinator, so it hosts the [`DistributedRuntimeService`] in addition to
    /// its own [`DistributedRuntimeClient`](ryft_pjrt::DistributedRuntimeClient). The call blocks until every process
    /// has connected.
    ///
    /// This function uses the default [`DistributedRuntimeServiceOptions`] and [`DistributedRuntimeClientOptions`],
    /// except that a process keeps trying to reach the coordinator for five minutes, as in JAX, and that a missed
    /// heartbeat does not terminate the process. Use [`Self::initialize_with_options`] for other timeouts.
    ///
    /// # Parameters
    ///
    ///   - `plugin`: PJRT [`Plugin`] that provides the service and the client (e.g., [`Platform::plugin`](
    ///     crate::Platform::plugin)).
    ///   - `coordinator_address`: TCP address that process 0 serves and every process connects to (e.g.,
    ///     `"10.0.0.1:24680"`). It must be reachable from every process.
    ///   - `process_count`: Number of processes of the job.
    ///   - `process_index`: Index of this process in `0..process_count`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::initialize_with_options`].
    pub fn initialize(
        plugin: &Plugin,
        coordinator_address: &str,
        process_count: usize,
        process_index: usize,
    ) -> Result<Self, Error> {
        let coordinate = |name: &str, value: usize| {
            u32::try_from(value).map_err(|_| {
                ryft_pjrt::Error::invalid_argument(format!("distributed runtime {name} {value} does not fit in `u32`"))
            })
        };
        let service_options = DistributedRuntimeServiceOptions {
            num_nodes: coordinate("process count", process_count)?,
            ..Default::default()
        };
        // The default `missed_heartbeat_callback` panics, which aborts a process during an orderly shutdown when its
        // polling request observes the service stopping before its own polling loop has stopped. Callers that want a
        // missed heartbeat to terminate the process install their own callback with `Self::initialize_with_options`.
        let client_options = DistributedRuntimeClientOptions {
            node_id: coordinate("process index", process_index)?,
            initialization_timeout: DEFAULT_INITIALIZATION_TIMEOUT,
            missed_heartbeat_callback: Some(Box::new(|_| {})),
            ..Default::default()
        };
        Self::initialize_with_options(plugin, coordinator_address, None, service_options, client_options)
    }

    /// Initializes the [`DistributedRuntime`] of one process as [`Self::initialize`] does, with explicit
    /// [`DistributedRuntimeServiceOptions`] and [`DistributedRuntimeClientOptions`] (e.g., for other heartbeat,
    /// request, or shutdown timeouts). `service_options.num_nodes` is the process count of the job and
    /// `client_options.node_id` is the index of this process. Only the coordinator uses the remaining service options.
    /// A process waits for the launch identity of the coordinator for `client_options.rpc_timeout`, which is also how
    /// long a [`Session`](crate::Session) waits for the device inventories of the other processes.
    ///
    /// # Parameters
    ///
    ///   - `plugin`: PJRT [`Plugin`] that provides the service and the client.
    ///   - `coordinator_address`: TCP address that every process connects to.
    ///   - `coordinator_bind_address`: TCP address that the coordinator serves, for a coordinator that cannot bind the
    ///     address that the other processes reach it at (e.g., behind address translation, or to serve every network
    ///     interface with `"[::]:24680"`, which is what `jax.distributed.initialize` binds unless it is told
    ///     otherwise). [`None`] serves `coordinator_address` itself. Only the coordinator uses it.
    ///   - `service_options`: Options of the coordination service.
    ///   - `client_options`: Options of the connection of this process to that service.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Pjrt`] with an invalid argument error if the process index is not smaller than the process
    /// count, and with the error of the service or of the client if the service cannot be started, the process cannot
    /// connect, or the launch identity does not arrive in time.
    pub fn initialize_with_options(
        plugin: &Plugin,
        coordinator_address: &str,
        coordinator_bind_address: Option<&str>,
        service_options: DistributedRuntimeServiceOptions,
        client_options: DistributedRuntimeClientOptions,
    ) -> Result<Self, Error> {
        let (process_index, process_count) = (client_options.node_id, service_options.num_nodes);
        if process_index >= process_count {
            return Err(ryft_pjrt::Error::invalid_argument(format!(
                "process index {process_index} is not smaller than the process count {process_count} of the \
                 distributed runtime",
            ))
            .into());
        }
        let request_timeout = client_options.rpc_timeout;
        // The coordinator hosts the service that every process, including the coordinator itself, connects to.
        let service = if process_index == 0 {
            let address = coordinator_bind_address.unwrap_or(coordinator_address);
            Some(plugin.distributed_runtime_service(address, service_options)?)
        } else {
            None
        };
        let client = plugin.distributed_runtime_client(coordinator_address, client_options)?;
        client.connect()?;
        let key_value_store = DistributedKeyValueStore::new(client);
        let launch_id = if process_index == 0 {
            let mut identity = Sha256::new();
            identity.update(coordinator_address.as_bytes());
            identity.update(std::process::id().to_le_bytes());
            identity.update(SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos().to_le_bytes());
            let launch_id: [u8; 32] = identity.finalize().into();
            key_value_store.put(LAUNCH_ID_KEY, launch_id.as_slice())?;
            launch_id
        } else {
            key_value_store
                .get(LAUNCH_ID_KEY, request_timeout)?
                .try_into()
                .map_err(|_| ryft_pjrt::Error::invalid_argument("distributed launch identity has the wrong length"))?
        };
        Ok(Self {
            state: Arc::new(DistributedRuntimeState {
                key_value_store,
                launch_id,
                kernel_sequence: AtomicU64::new(0),
                process_index: process_index as ProcessIndex,
                process_count: process_count as usize,
                request_timeout,
                is_shut_down: AtomicBool::new(false),
                service,
            }),
        })
    }

    /// Returns the index of this process among the processes of the job, as `jax.process_index()` does.
    #[inline]
    pub fn process_index(&self) -> ProcessIndex {
        self.state.process_index
    }

    /// Returns the number of processes of the job, as `jax.process_count()` does.
    #[inline]
    pub fn process_count(&self) -> usize {
        self.state.process_count
    }

    /// Returns the [`DistributedKeyValueStore`] of this runtime, which every process of the job shares, for
    /// coordination protocols of the caller. The store never overwrites or removes a record.
    #[inline]
    pub fn key_value_store(&self) -> &DistributedKeyValueStore {
        &self.state.key_value_store
    }

    /// Returns whether `other` is a handle of this runtime (i.e., a clone of this handle).
    #[inline]
    pub(crate) fn is_same_runtime(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.state, &other.state)
    }

    /// Disconnects this process from the coordination service and, on the coordinator, stops the service, as
    /// `jax.distributed.shutdown` does. Disconnecting waits until every process of the job disconnects, for the
    /// shutdown timeout of the client options. Later calls on any handle of this runtime do nothing. Dropping the last
    /// handle shuts the runtime down as well, so this function is only needed to shut it down earlier or to observe
    /// the error of doing so. Clients and exchanges of the runtime cannot use its key-value store afterwards.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Pjrt`] with the error of the coordination service if this process cannot disconnect cleanly
    /// (e.g., because another process did not disconnect in time). The coordinator still stops its service then.
    pub fn shutdown(&self) -> Result<(), Error> {
        if self.state.is_shut_down.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        let disconnected = self.state.key_value_store.client().shutdown();
        if let Some(service) = &self.state.service {
            service.shutdown();
        }
        Ok(disconnected?)
    }

    /// Reserves one ordered kernel coordinator namespace within the existing distributed launch, returning the launch
    /// identity, the reserved sequence number, the index of this process, and the number of processes.
    pub(crate) fn reserve_kernel_coordinator(&self) -> Result<([u8; 32], u64, u32, u32), Error> {
        let sequence = self
            .state
            .kernel_sequence
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |value| value.checked_add(1))
            .map_err(|_| ryft_pjrt::Error::invalid_argument("kernel coordinator sequence overflow"))?;
        // The coordinates were converted from `u32` values when the runtime was initialized.
        Ok((self.state.launch_id, sequence, self.state.process_index as u32, self.state.process_count as u32))
    }

    /// Creates a chunked, checksummed compilation artifact exchange over the key-value store of this runtime, for
    /// [`CompilationCache::with_artifact_exchange`](ryft_core::CompilationCache::with_artifact_exchange). The exchange
    /// takes its process index and process count from this runtime.
    ///
    /// The returned exchange keeps both the distributed client and coordinator service alive. A manifest is published
    /// only after every bounded chunk, so followers never accept a partially published artifact. Every record is
    /// addressed by the persistent compilation key, so processes may compile programs in different orders, skip
    /// programs that their in-memory caches already hold, or compile programs that no other process compiles (e.g.,
    /// eager operations that only process zero runs). The records stay in the memory of the coordinator for the rest
    /// of the launch.
    ///
    /// The leader of each compilation is the process that the compilation names (refer to
    /// [`CompilationCacheDomain::artifact_exchange_leader`](ryft_core::CompilationCacheDomain::artifact_exchange_leader)),
    /// which for a [`Session`](crate::Session) is the lowest index of the processes that own a device of the mesh of
    /// the program. Before the leader compiles a program, [`CompilationArtifactExchange::validate_compatibility`]
    /// announces the persistent compilation key and the process count without waiting for any other process. A
    /// follower that misses its in-memory cache waits for that announcement before it waits for the artifact, and
    /// rejects a process count that disagrees with its own. The XLA persistent key includes platform and plugin
    /// versions, compiler identity, and logical topology, so a follower whose compatibility properties differ from
    /// those of its leader never observes an announcement for its key. Its compatibility validation then times out,
    /// and the configured exchange policy decides whether it compiles locally.
    pub fn compilation_artifact_exchange(&self) -> Arc<dyn CompilationArtifactExchange> {
        Arc::new(DistributedCompilationArtifactExchange {
            runtime: Arc::clone(&self.state),
            process_index: self.state.process_index,
            process_count: self.state.process_count,
            chunk_size: DEFAULT_ARTIFACT_CHUNK_SIZE,
            maximum_artifact_size: DEFAULT_MAXIMUM_ARTIFACT_SIZE,
            maximum_compatibility_key_size: DEFAULT_MAXIMUM_COMPATIBILITY_KEY_SIZE,
        })
    }

    /// Creates a PJRT [`Client`] of `plugin` that uses the key-value store of this runtime, so that it takes part in
    /// the cross-host transfers and collectives of the job. The client borrows this handle, which must outlive it.
    /// `options` are completed with the process coordinates of this runtime as [`Self::client_options`] describes.
    /// A [`Session`](crate::Session) creates its client this way when it is built with
    /// [`SessionBuilder::distributed_runtime`](crate::SessionBuilder::distributed_runtime).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Pjrt`] with an invalid argument error if `options` name another process index or process
    /// count than this runtime has, and with the error of the plugin if the client cannot be created.
    pub fn create_client<'s>(&'s self, plugin: &Plugin, options: ClientOptions) -> Result<Client<'s>, Error> {
        Ok(plugin.client_with_key_value_store(self.client_options(options)?, &self.state.key_value_store)?)
    }

    /// Returns `options` with the process coordinates of this runtime, which are the `process_id` of
    /// [`ClientOptions::CPU`] and the `node_id` and `node_count` of [`ClientOptions::GPU`], where `options` leave them
    /// unset. A client that is created without them reports process 0 on every process. The unstructured
    /// [`ClientOptions::Other`] are returned as they are, because their keys belong to their plugin.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Pjrt`] with an invalid argument error if `options` set a coordinate to another value than this
    /// runtime has.
    pub(crate) fn client_options(&self, mut options: ClientOptions) -> Result<ClientOptions, Error> {
        let complete = |name: &str, option: &mut Option<usize>, value: usize| match *option {
            Some(existing) if existing != value => Err(ryft_pjrt::Error::invalid_argument(format!(
                "client option `{name}` is {existing}, but its distributed runtime has {value}",
            ))),
            _ => {
                *option = Some(value);
                Ok(())
            }
        };
        match &mut options {
            ClientOptions::CPU(options) => complete("process_id", &mut options.process_id, self.process_index())?,
            ClientOptions::GPU(options) => {
                complete("node_id", &mut options.node_id, self.process_index())?;
                complete("node_count", &mut options.node_count, self.process_count())?;
            }
            ClientOptions::Other(_) => {}
        }
        Ok(options)
    }

    /// Returns the devices of the other processes of the job that `client`, a client of this process that uses the
    /// key-value store of this runtime, does not list, which a [`Session`](crate::Session) of that client adds to
    /// its devices (refer to [`SessionBuilder::distributed_runtime`](crate::SessionBuilder::distributed_runtime)). A
    /// client that lists a device of every process (e.g., a CUDA client) needs none, and nothing is exchanged for it.
    ///
    /// Otherwise (e.g., for a CPU client, which lists only the devices of its own process), this process publishes
    /// its devices to the key-value store and reads those that each other process published. Every process of the
    /// job must therefore make this call, which a process does when it builds its session, and the call waits for the
    /// other processes for the request timeout of this runtime.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Pjrt`] with a failed precondition error if `client` does not use the key-value store of this
    /// runtime, if it reports another process index than this runtime has (i.e., its options did not name its
    /// process), or if this process already published other devices for this runtime, and with the error of the store
    /// if another process does not publish its devices in time.
    pub(crate) fn remote_devices(&self, client: &Client<'_>) -> Result<Vec<Device>, Error> {
        let (process_index, process_count) = (self.process_index(), self.process_count());
        if !client.key_value_store().is_some_and(|store| std::ptr::addr_eq(store, &self.state.key_value_store)) {
            return Err(ryft_pjrt::Error::failed_precondition(
                "client does not use the key-value store of its distributed runtime; create it with \
                 `DistributedRuntime::create_client`",
            )
            .into());
        }
        let client_process_index = client.process_index()?;
        if client_process_index != process_index {
            return Err(ryft_pjrt::Error::failed_precondition(format!(
                "client reports process index {client_process_index}, but its distributed runtime has process index \
                 {process_index}; name the process in the options of the client",
            ))
            .into());
        }
        let devices = client.devices()?.iter().map(Device::from_pjrt).collect::<Result<Vec<_>, _>>()?;
        let listed_processes = devices.iter().map(Device::process_index).collect::<BTreeSet<_>>();
        if (0..process_count).all(|process| listed_processes.contains(&process)) {
            return Ok(Vec::new());
        }

        let key = |process: ProcessIndex| {
            let mut key = b"ryft/session-devices/v1/".to_vec();
            key.extend_from_slice(&self.state.launch_id);
            key.extend_from_slice(format!("/{process}").as_bytes());
            key
        };
        let decode = |record: &[u8]| {
            serde_json::from_slice::<Vec<DeviceId>>(record).map_err(|error| {
                ryft_pjrt::Error::invalid_argument(format!("failed to decode a published device inventory: {error}"))
            })
        };
        let store = &self.state.key_value_store;
        let local_devices = devices
            .iter()
            .filter(|device| device.process_index() == process_index)
            .map(Device::id)
            .collect::<Vec<_>>();
        // The store never overwrites a record, so a later session of this process finds the record of the first one.
        let record = serde_json::to_vec(&local_devices).unwrap();
        match store.put(&key(process_index), &record) {
            Ok(()) => {}
            Err(ryft_pjrt::Error::AlreadyExists { .. }) => {
                let published = decode(&store.try_get(&key(process_index))?)?;
                if published != local_devices {
                    return Err(ryft_pjrt::Error::failed_precondition(format!(
                        "process {process_index} already published the devices {published:?} to its distributed \
                         runtime, but this client has the devices {local_devices:?}",
                    ))
                    .into());
                }
            }
            Err(error) => return Err(error.into()),
        }

        let deadline = Instant::now() + self.state.request_timeout;
        let mut remote_devices = Vec::new();
        for process in (0..process_count).filter(|process| !listed_processes.contains(process)) {
            let remaining = deadline.saturating_duration_since(Instant::now());
            let record = store.get(&key(process), remaining)?;
            remote_devices.extend(decode(&record)?.into_iter().map(|device_id| Device::new(device_id, process)));
        }
        Ok(remote_devices)
    }
}

/// Record that the leader publishes after all chunks of a compilation artifact, or in place of them on failure.
#[derive(Serialize, Deserialize)]
struct CompilationArtifactManifest {
    /// Exchange protocol version of the leader.
    schema_version: u32,

    /// Size of the complete artifact in bytes, which is zero for a failure.
    size: u64,

    /// Number of chunks that hold the artifact.
    chunk_count: u64,

    /// SHA-256 checksum of the complete artifact.
    checksum: [u8; 32],

    /// Terminal compilation failure message, if the leader failed to compile the program.
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

/// Shared state of a [`DistributedRuntime`], retained by every clone of the runtime and by its artifact exchanges.
struct DistributedRuntimeState {
    /// Key-value store backed by the distributed runtime client of this process.
    key_value_store: DistributedKeyValueStore,

    /// Launch identity that the coordinator generated and published during initialization.
    launch_id: [u8; 32],

    /// Next sequence number that [`DistributedRuntime::reserve_kernel_coordinator`] reserves.
    kernel_sequence: AtomicU64,

    /// Index of this process among the processes of the job.
    process_index: ProcessIndex,

    /// Number of processes of the job.
    process_count: usize,

    /// Duration for which this process waits for a record that another process publishes during setup (i.e., the
    /// launch identity and the device inventories), which is the request timeout of its client options.
    request_timeout: Duration,

    /// Whether [`DistributedRuntime::shutdown`] already ran.
    is_shut_down: AtomicBool,

    /// Coordination service, which only the coordinator hosts. It is declared last, so that it is dropped after the
    /// key-value store has disconnected this process from it.
    service: Option<DistributedRuntimeService>,
}

/// [`CompilationArtifactExchange`] that publishes chunked, checksummed artifacts through the key-value store of a
/// [`DistributedRuntime`].
struct DistributedCompilationArtifactExchange {
    /// Shared runtime state that owns the key-value store and keeps the client and service alive.
    runtime: Arc<DistributedRuntimeState>,

    /// Index of this process in the exchange.
    process_index: usize,

    /// Number of processes that participate in the exchange.
    process_count: usize,

    /// Maximum number of artifact bytes per key-value store record.
    chunk_size: usize,

    /// Maximum size, in bytes, of an artifact that this exchange publishes or receives.
    maximum_artifact_size: usize,

    /// Maximum size, in bytes, of a persistent compilation key that compatibility validation accepts.
    maximum_compatibility_key_size: usize,
}

impl DistributedCompilationArtifactExchange {
    /// Returns the key-value store key of the record named `suffix` for the persistent compilation key `key`, scoped
    /// to the launch identity and addressed by the SHA-256 digest of `key`.
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

    /// Returns the key-value store key of the chunk at `index` of the artifact with the SHA-256 `checksum` for the
    /// persistent compilation key `key`. Chunks are addressed by the checksum of their artifact, so the chunks of two
    /// serializations of one program never mix, and a publication that failed after storing some of its chunks can
    /// be repeated.
    fn chunk_key(&self, key: &[u8], checksum: &[u8; 32], index: usize) -> Vec<u8> {
        self.key(key, format!("chunk/{}/{index:016x}", hex::encode(checksum)).as_bytes())
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

    /// Constructs a [`CompilationArtifactExchangeError::Failed`] error with the provided message.
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

    fn validate_compatibility(
        &self,
        key: &[u8],
        leader: usize,
        timeout: Duration,
    ) -> Result<(), CompilationArtifactExchangeError> {
        // Address compatibility validation by the compilation key because each process validates only compilations
        // that miss its own in-memory cache. The leader of a key announces it before it compiles the program without
        // waiting for anyone, and each follower waits for the announcement of its own key. Every process that takes
        // part in a compilation names the same leader for its key, which need not be process zero (e.g., for a program
        // that runs only on devices of other processes).
        if self.process_count == 0 || self.process_index >= self.process_count || leader >= self.process_count {
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

        if self.process_index == leader {
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
                    "process {leader} uses compilation artifact exchange schema {}, but this process uses schema {}",
                    manifest.schema_version, ARTIFACT_EXCHANGE_SCHEMA_VERSION,
                ),
            });
        }
        if manifest.process_count != self.process_count as u64 {
            return Err(CompilationArtifactExchangeError::Incompatible {
                message: format!(
                    "process {leader} expects {} processes, but this process expects {}",
                    manifest.process_count, self.process_count,
                ),
            });
        }
        if get(b"preflight/key")?.as_slice() != key {
            return Err(CompilationArtifactExchangeError::Incompatible {
                message: format!(
                    "process {leader} announced a different persistent compilation key with the same digest"
                ),
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
        // A chunk that an earlier publication of the same artifact already stored is kept, so a publication that
        // failed before its manifest can be repeated.
        let checksum = Sha256::digest(artifact).into();
        for (index, chunk) in artifact.chunks(self.chunk_size).enumerate() {
            self.put_if_absent(&self.chunk_key(key, &checksum, index), chunk)?;
        }
        let manifest = CompilationArtifactManifest {
            schema_version: ARTIFACT_EXCHANGE_SCHEMA_VERSION,
            size: artifact.len() as u64,
            chunk_count: artifact.len().div_ceil(self.chunk_size) as u64,
            checksum,
            failure: None,
        };
        let manifest = serde_json::to_vec(&manifest)
            .map_err(|error| Self::failed(format!("failed to encode artifact manifest: {error}")))?;
        self.put_if_absent(&self.key(key, b"manifest"), manifest.as_slice())
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
            usize::try_from(manifest.size).map_err(|_| Self::failed("compiled artifact size overflows `usize`"))?;
        let chunk_count = usize::try_from(manifest.chunk_count)
            .map_err(|_| Self::failed("compiled artifact chunk count overflows `usize`"))?;
        if size > self.maximum_artifact_size || chunk_count != size.div_ceil(self.chunk_size) {
            return Err(Self::failed("compiled artifact manifest exceeds configured bounds"));
        }
        let mut artifact = Vec::with_capacity(size);
        for index in 0..chunk_count {
            let remaining = timeout.checked_sub(start.elapsed()).ok_or(CompilationArtifactExchangeError::TimedOut)?;
            let chunk_key = self.chunk_key(key, &manifest.checksum, index);
            let chunk = self.runtime.key_value_store.get(&chunk_key, remaining).map_err(|error| match error {
                ryft_pjrt::Error::NotFound { .. } | ryft_pjrt::Error::DeadlineExceeded { .. } => {
                    CompilationArtifactExchangeError::TimedOut
                }
                error => Self::failed(error.to_string()),
            })?;
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

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::future::Future;
    #[cfg(feature = "cuda-13")]
    use std::io::Write;
    use std::pin::pin;
    use std::process::Command;
    #[cfg(feature = "cuda-13")]
    use std::process::Stdio;
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};
    use std::time::{Duration, Instant};
    use std::{env, ptr, thread};

    #[cfg(feature = "cuda-13")]
    use indoc::{formatdoc, indoc};
    use pretty_assertions::assert_eq;
    use ryft_core::{
        Array as CoreArray, ArrayReference, ArrayType, BatchAxis, CompilationArtifactExchange,
        CompilationArtifactExchangeError, DataType, Device, DeviceMesh, DomainTracer, Layout, LogicalMesh, Memory,
        MeshAxis, MeshAxisType, ParallelReduce, Place, Placement, PlacementRequest, ProjectedValue, ReductionKind,
        ReferenceType, ShardMapOptions, Sharding, ShardingDimension, StridedLayout, Tile, TileDimension, TiledLayout,
        UnreducedKind, ValueProjection, batch, shard_map,
    };
    #[cfg(feature = "cuda-13")]
    use ryft_core::{
        CompilationArtifactExchangePolicy, CompilationCache, Dimension, DimensionBounds, DimensionVariable,
        ParallelBroadcast, ProgramError, Shape, TypeError, Typed,
    };
    #[cfg(feature = "cuda-13")]
    use ryft_pjrt::{
        BufferType, ExecutionDeviceInputs, ExecutionInput, GpuMemoryAllocator, GpuPlatform, Program as PjrtProgram,
        load_cuda_13_plugin,
    };
    use ryft_pjrt::{
        Client, ClientOptions, CpuClientOptions, DistributedRuntimeClientOptions, DistributedRuntimeServiceOptions,
        Error as PjrtError, GpuClientOptions, KeyValueStore, load_cpu_plugin,
    };
    use sha2::{Digest, Sha256};

    #[cfg(feature = "cuda-13")]
    use crate::arrays::host::DIFFERENT_VALUES_MESSAGE;
    use crate::arrays::host::ValueComparison;
    use crate::operations::sharding::shard_map::tests::doubling_reduced_reference_program;
    use crate::sessions::tests::DeclareRemoteDevices;
    #[cfg(feature = "cuda-13")]
    use crate::tests::test_compilation_options;
    use crate::tests::{loopback_address, values_from_bytes, values_to_bytes};
    use crate::{
        Array, ArrayError, Error, FromPjrt, Options, PlaceInput, PlacementOptions, Platform, Reference, Session,
        SessionError, Type, Value, compile_statefully,
    };
    #[cfg(feature = "cuda-13")]
    use crate::{ExecutionError, ToPjrt};

    use super::{
        ARTIFACT_EXCHANGE_SCHEMA_VERSION, CompilationArtifactManifest, CompilationCompatibilityManifest,
        DistributedCompilationArtifactExchange, DistributedRuntime,
    };

    const PROCESS_TEST_ROLE: &str = "RYFT_DISTRIBUTED_EXCHANGE_TEST_ROLE";
    const PROCESS_TEST_ADDRESS: &str = "RYFT_DISTRIBUTED_EXCHANGE_TEST_ADDRESS";
    const PROCESS_TEST_ACKNOWLEDGEMENT: &[u8] = b"ryft/test/artifact-consumed";
    #[cfg(feature = "cuda-13")]
    const MULTI_HOST_TEST_ROLE: &str = "RYFT_DISTRIBUTED_MULTI_HOST_TEST_ROLE";
    #[cfg(feature = "cuda-13")]
    const MULTI_HOST_TEST_ADDRESS: &str = "RYFT_DISTRIBUTED_MULTI_HOST_TEST_ADDRESS";

    /// Initializes the runtime of a job of one process, which is both its coordinator and its only participant.
    fn single_process_runtime() -> DistributedRuntime {
        DistributedRuntime::initialize(&load_cpu_plugin().unwrap(), &loopback_address(), 1, 0).unwrap()
    }

    /// Returns an exchange over `runtime` for a process with the provided coordinates, so that one runtime can stand
    /// for several processes, with bounds that small payloads reach (i.e., chunks of three bytes and artifacts and
    /// keys of at most 32 bytes).
    fn exchange(
        runtime: &DistributedRuntime,
        process_index: usize,
        process_count: usize,
    ) -> DistributedCompilationArtifactExchange {
        DistributedCompilationArtifactExchange {
            runtime: Arc::clone(&runtime.state),
            process_index,
            process_count,
            chunk_size: 3,
            maximum_artifact_size: 32,
            maximum_compatibility_key_size: 32,
        }
    }

    /// Returns the elements of the buffer of every shard of `array` that the current process addresses, in mesh order.
    fn read_addressable_f32s(array: &Array) -> Vec<Vec<f32>> {
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
    /// and for two processes that address one device each, as JAX's multi-controller model requires. Every process of
    /// a job must call this function, because each scenario executes collectives over both devices.
    fn check_shard_map_multi_host_scenarios(client: &'static Client<'static>) {
        /// Wakes the thread polling upload readiness when a PJRT event completes.
        struct TransferWaker(thread::Thread);

        impl Wake for TransferWaker {
            fn wake(self: Arc<Self>) {
                self.0.unpark();
            }
        }

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
        let session = Session::builder().client(client).build().unwrap();
        let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(manual_mesh.clone(), devices.clone()).unwrap();
        let sharded = Sharding::new(manual_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(manual_mesh.clone(), 1);

        // Each rank packs only its local tiles and uses the same global type and mesh as its peers.
        let input_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap();
        let tiles = [[1f32, 2.0], [3.0, 4.0]];
        let local_tiles = local_positions
            .iter()
            .map(|&position| (devices[position].id(), Arc::<[u8]>::from(values_to_bytes(&tiles[position]))))
            .collect::<Vec<_>>();
        let input = Array::from_host_shards(
            &session,
            input_type.clone(),
            mesh.clone(),
            None,
            local_tiles.iter().map(|(device, bytes)| (*device, bytes.as_ref())),
        )
        .unwrap();
        input.block_until_ready().unwrap();
        let expected_local_tiles = local_positions.iter().map(|&position| tiles[position].to_vec()).collect::<Vec<_>>();
        assert_eq!(read_addressable_f32s(&input), expected_local_tiles);

        // Dense host materialization of a sharded array needs every tile, so it succeeds only in a process that
        // addresses both devices and otherwise names a shard of the other process before issuing any copy, as JAX
        // rejects fetching arrays whose value spans non-addressable devices.
        if local_positions.len() == 2 {
            let materialized = input.copy_to_host_bytes().unwrap();
            assert_eq!(values_from_bytes::<f32>(materialized.as_slice()), vec![1.0, 2.0, 3.0, 4.0]);
        } else {
            let remote_position = 1 - local_positions[0];
            assert!(matches!(
                input.copy_to_host_bytes(),
                Err(Error::Array(error)) if matches!(
                    *error,
                    ArrayError::MissingShardBuffer { shard_index, device_id }
                        if shard_index == remote_position && device_id == devices[remote_position].id(),
                ),
            ));
        }

        // Move the host buffers into the second upload so PJRT can retain them until the transfer finishes.
        let asynchronous_input =
            Array::from_host_shards(&session, input_type, mesh.clone(), None, local_tiles).unwrap();
        let mut ready = pin!(asynchronous_input.ready());
        let waker = Waker::from(Arc::new(TransferWaker(thread::current())));
        let mut context = Context::from_waker(&waker);
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            match ready.as_mut().poll(&mut context) {
                Poll::Ready(result) => {
                    result.unwrap();
                    break;
                }
                Poll::Pending => {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    assert!(!remaining.is_zero(), "asynchronous host shard upload timed out");
                    thread::park_timeout(remaining);
                }
            }
        }
        assert_eq!(read_addressable_f32s(&asynchronous_input), expected_local_tiles);

        // An unchecked map returns each tile as its candidate for an output that is unreduced along `x`, and a checked
        // map completes it with an all-reduce of the matching kind. Both uploads feed a reduction over both devices.
        let complete = |input: Array, kind: UnreducedKind, reduction: ReductionKind| {
            let unreduced = replicated.clone().with_unreduced_axes(["x"]).unwrap().with_unreduced_kind(kind).unwrap();
            let candidates = shard_map(
                Ok,
                input,
                manual_mesh.clone(),
                sharded.clone(),
                unreduced.clone(),
                ShardMapOptions::new().with_check_manual_variation(false),
            )
            .unwrap();
            assert_eq!(candidates.sharding(), &unreduced);
            shard_map(
                |candidates: ProjectedValue<ArrayType, DomainTracer<Session>>| {
                    candidates.parallel_reduce(reduction, "x")
                },
                candidates,
                manual_mesh.clone(),
                unreduced,
                replicated.clone(),
                ShardMapOptions::new(),
            )
            .unwrap()
        };
        let sum = complete(input, UnreducedKind::Sum, ReductionKind::Sum);
        assert_eq!(read_addressable_f32s(&sum), vec![vec![4.0, 6.0]; local_positions.len()]);
        let maximum = complete(asynchronous_input.clone(), UnreducedKind::Max, ReductionKind::Max);
        assert_eq!(read_addressable_f32s(&maximum), vec![vec![3.0, 4.0]; local_positions.len()]);

        // A replicated array spans every process, but each process holds its whole value, so dense host reads succeed
        // in every process, as JAX reads a value that the current process holds in full.
        let materialized = sum.copy_to_host_bytes().unwrap();
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
        let compiled = staged.lower(&session).unwrap().compile(&session).unwrap();
        let items = Array::from_host_buffer(
            &session,
            items_type,
            explicit_mesh,
            None,
            values_to_bytes(&[1f32, 2., 3., 4., 5., 6.]),
        )
        .unwrap();
        let Value::Array(doubled) = compiled.executable().execute(&session, Value::Array(items)).unwrap() else {
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
        let compiled = compile_statefully::<_, Vec<Type>, Vec<Type>>(
            |inputs| {
                let context = inputs[0].context().clone();
                program.interpret_in_context(&context, inputs)?;
                Ok(Vec::new())
            },
            vec![Type::Reference(ReferenceType::new(referent_type.clone()))],
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let reference = ArrayReference::new(
            Array::from_host_buffer(&session, referent_type, mesh, None, values_to_bytes(&[1f32, 2.0]).as_slice())
                .unwrap(),
        );
        compiled
            .execute_statefully_and_wait(
                &session,
                vec![Value::Reference(Reference::new(reference.clone(), session.clone()))],
            )
            .unwrap();
        assert_eq!(read_addressable_f32s(&reference.read().unwrap()), vec![vec![2.0, 4.0]; local_positions.len()]);
        let materialized = reference.read().unwrap().copy_to_host_bytes().unwrap();
        assert_eq!(values_from_bytes::<f32>(materialized.as_slice()), vec![2.0, 4.0]);
    }

    /// Places arrays of `[1, 2, 3, 4]` across the two processes of a CUDA job with `place`, which calls
    /// [`Array::to_placement`] or [`Array::into_placement`], and prints a `probe:` line with whether the client exposes
    /// the PJRT cross-host transfers extension and then one per placement with the elements of the shards that this
    /// process addresses afterwards and the result of [`Array::block_until_ready`]. Every process of the job must call
    /// this function, because each placement transfers shards between the processes or executes a compiled reshard
    /// over the devices of both.
    #[cfg(feature = "cuda-13")]
    fn probe_cross_host_placements(
        client: &'static Client<'static>,
        place: impl Fn(Array, &Session, Placement) -> Result<Array, Error>,
    ) {
        let rank = client.process_index().unwrap();
        match client.cross_host_transfers_extension() {
            Ok(_) => println!("probe: rank {rank} cross_host_transfers_extension: available"),
            Err(error) => println!("probe: rank {rank} cross_host_transfers_extension: {error}"),
        }
        let devices = client
            .devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(devices.iter().map(Device::process_index).collect::<Vec<_>>(), vec![0, 1]);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = |devices: Vec<Device>| {
            let axis = MeshAxis::new("x", devices.len(), MeshAxisType::Explicit).unwrap();
            DeviceMesh::new(LogicalMesh::new(vec![axis]).unwrap(), devices).unwrap()
        };
        let replicated = |mesh: &DeviceMesh| Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let sharded = |mesh: &DeviceMesh| {
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap()
        };
        let array = |mesh: &DeviceMesh, sharding: Sharding, values: [f32; 4]| {
            let r#type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding).unwrap();
            Array::from_host_buffer(&session, r#type, mesh.clone(), None, values_to_bytes(&values)).unwrap()
        };
        let process_0_mesh = mesh(vec![devices[0]]);
        let forward_mesh = mesh(devices.clone());
        let reversed_mesh = mesh(vec![devices[1], devices[0]]);

        // This checks whole-shard copies, slicing a region before sending it, and joining local and received
        // regions on each destination. CUDA pairs sends with receives in issue order for each pair of devices.
        // The last copy exposes an unmatched send from an earlier placement by receiving its old bytes.
        let placements = [
            (
                "sharded_to_reversed_sharded",
                array(&forward_mesh, sharded(&forward_mesh), [9.0, 10.0, 11.0, 12.0]),
                Placement::new(reversed_mesh.clone(), sharded(&reversed_mesh)).unwrap(),
            ),
            (
                "replicated_to_replicated",
                array(&process_0_mesh, replicated(&process_0_mesh), [1.0, 2.0, 3.0, 4.0]),
                Placement::new(forward_mesh.clone(), replicated(&forward_mesh)).unwrap(),
            ),
            (
                "replicated_to_sharded",
                array(&process_0_mesh, replicated(&process_0_mesh), [5.0, 6.0, 7.0, 8.0]),
                Placement::new(forward_mesh.clone(), sharded(&forward_mesh)).unwrap(),
            ),
            (
                "sharded_to_reversed_replicated",
                array(&forward_mesh, sharded(&forward_mesh), [13.0, 14.0, 15.0, 16.0]),
                Placement::new(reversed_mesh.clone(), replicated(&reversed_mesh)).unwrap(),
            ),
            (
                "replicated_to_replicated_again",
                array(&process_0_mesh, replicated(&process_0_mesh), [17.0, 18.0, 19.0, 20.0]),
                Placement::new(forward_mesh.clone(), replicated(&forward_mesh)).unwrap(),
            ),
        ];
        for (name, source, target) in placements {
            match place(source, &session, target) {
                Ok(placed) => {
                    let ready = match placed.block_until_ready() {
                        Ok(()) => "ready".to_string(),
                        Err(error) => error.to_string().lines().next().unwrap_or("").to_string(),
                    };
                    println!("probe: rank {rank} {name}: {:?} {ready}", read_addressable_f32s(&placed));
                }
                Err(error) => {
                    println!("probe: rank {rank} {name}: {}", error.to_string().lines().next().unwrap_or(""));
                }
            }
        }
    }

    /// Places `source` on `target` with [`Array::to_placement`] and prints a `probe:` line labeled `name` with the
    /// memory kind and the elements of every shard that this process addresses afterwards and the result of
    /// [`Array::block_until_ready`], or with the first line of the error of the placement.
    #[cfg(feature = "cuda-13")]
    fn probe_placement(rank: usize, name: &str, session: &Session, source: &Array, target: Placement) {
        match source.to_placement(session, target) {
            Ok(placed) => {
                let ready = match placed.block_until_ready() {
                    Ok(()) => "ready".to_string(),
                    Err(error) => error.to_string().lines().next().unwrap_or("").to_string(),
                };
                let shards = placed
                    .addressable_shards()
                    .map(|shard| {
                        let kind = shard.buffer().unwrap().memory().unwrap().kind().unwrap().into_owned();
                        (kind, values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()))
                    })
                    .collect::<Vec<_>>();
                println!("probe: rank {rank} {name}: {shards:?} {ready}");
            }
            Err(error) => println!("probe: rank {rank} {name}: {}", error.to_string().lines().next().unwrap_or("")),
        }
    }

    /// Places `source` on `target` with [`Array::to_placement`] for the step `step` of a test that both ranks of a
    /// CUDA job run, waits for the result, and prints two `probe:` lines labeled with the rank and the step, one with
    /// the metadata of the result and one with the slice, the memory kind, and the row-major elements of every shard
    /// that this rank addresses. A failed placement, readiness check, or readback panics with the rank and the step,
    /// which the report of [`run_cuda_rank_pair`] shows for both ranks.
    #[cfg(feature = "cuda-13")]
    fn probe_placement_step(rank: usize, step: &str, session: &Session, source: &Array, target: Placement) {
        let placed = source
            .to_placement(session, target)
            .and_then(|placed| placed.block_until_ready().map(|()| placed))
            .unwrap_or_else(|error| panic!("rank {rank} failed to place the source {step}: {error}"));
        println!(
            "probe: rank {rank} {step} metadata: {} {:?} {:?} committed={}",
            placed.r#type(),
            placed.sharding(),
            placed.mesh().devices(),
            placed.is_committed(),
        );
        let shards = placed
            .addressable_shards()
            .map(|shard| {
                let kind = shard.buffer().unwrap().memory().unwrap().kind().unwrap().into_owned();
                let bytes = shard
                    .copy_to_host_bytes()
                    .unwrap_or_else(|error| panic!("rank {rank} failed to read a shard placed {step}: {error}"));
                (shard.slice().to_vec(), kind, values_from_bytes::<f32>(&bytes))
            })
            .collect::<Vec<_>>();
        println!("probe: rank {rank} {step} shards: {shards:?}");
    }

    /// Runs the test `helper`, given as a path within this test binary, as the two ranks of a CUDA job (refer to
    /// [`run_cuda_rank_pair`]), in which each rank prints the `probe:` lines of [`probe_placement_step`] for every step
    /// of `steps`. Asserts that both ranks print the same metadata for each step, whichever of them holds the shards
    /// of its result, and that the `shards` lines of both ranks, those of rank 0 first, are `expected_shards`.
    #[cfg(feature = "cuda-13")]
    fn check_cuda_rank_pair_placement_steps(helper: &str, steps: &[String], expected_shards: &str) {
        let outputs = run_cuda_rank_pair(helper);
        let probes = outputs
            .iter()
            .map(|output| {
                output
                    .lines()
                    .filter_map(|line| line.split_once("probe: ").map(|(_, probe)| probe))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for step in steps {
            let metadata = probes
                .iter()
                .enumerate()
                .map(|(rank, probes)| {
                    let prefix = format!("rank {rank} {step} metadata: ");
                    let line = probes.iter().find_map(|probe| probe.strip_prefix(prefix.as_str()));
                    line.unwrap_or_else(|| panic!("rank {rank} printed no metadata for the step `{step}`"))
                })
                .collect::<Vec<_>>();
            assert_eq!(metadata[0], metadata[1], "the ranks disagree on the metadata of the step `{step}`");
        }
        let shards = probes.concat().into_iter().filter(|probe| probe.contains(" shards: ")).collect::<Vec<_>>();
        assert_eq!(shards.join("\n"), expected_shards);
    }

    #[test]
    fn test_distributed_runtime_initialize() {
        // A job of one process is its own coordinator, so it returns as soon as its service observes it.
        let plugin = load_cpu_plugin().unwrap();
        let runtime = DistributedRuntime::initialize(&plugin, &loopback_address(), 1, 0).unwrap();
        assert_eq!((runtime.process_index(), runtime.process_count()), (0, 1));

        // Coordinates that the coordination service cannot represent are rejected before anything is started.
        assert!(matches!(
            DistributedRuntime::initialize(&plugin, "unused.invalid:1", usize::MAX, 0),
            Err(Error::Pjrt(PjrtError::InvalidArgument { message, .. }))
                if message == format!("distributed runtime process count {} does not fit in `u32`", usize::MAX),
        ));
        assert!(matches!(
            DistributedRuntime::initialize(&plugin, "unused.invalid:1", 1, usize::MAX),
            Err(Error::Pjrt(PjrtError::InvalidArgument { message, .. }))
                if message == format!("distributed runtime process index {} does not fit in `u32`", usize::MAX),
        ));
    }

    #[test]
    fn test_distributed_runtime_initialize_with_options() {
        // The coordinator serves the bind address, while the process connects to the address that it is reached at.
        let bind_address = loopback_address();
        let coordinator_address = format!("localhost:{}", bind_address.rsplit(':').next().unwrap());
        let runtime = DistributedRuntime::initialize_with_options(
            &load_cpu_plugin().unwrap(),
            &coordinator_address,
            Some(&bind_address),
            DistributedRuntimeServiceOptions { num_nodes: 1, ..Default::default() },
            DistributedRuntimeClientOptions {
                node_id: 0,
                missed_heartbeat_callback: Some(Box::new(|_| {})),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!((runtime.process_index(), runtime.process_count()), (0, 1));
        runtime.key_value_store().put(b"key", b"value").unwrap();
        assert_eq!(runtime.key_value_store().try_get(b"key").unwrap(), b"value");
    }

    #[test]
    fn test_distributed_runtime_initialize_with_options_rejects_process_index_outside_process_count() {
        let plugin = load_cpu_plugin().unwrap();
        for (process_index, process_count) in [(2, 2), (0, 0)] {
            assert!(matches!(
                DistributedRuntime::initialize_with_options(
                    &plugin,
                    "unused.invalid:1",
                    None,
                    DistributedRuntimeServiceOptions { num_nodes: process_count, ..Default::default() },
                    DistributedRuntimeClientOptions { node_id: process_index, ..Default::default() },
                ),
                Err(Error::Pjrt(PjrtError::InvalidArgument { message, .. }))
                    if message == format!(
                        "process index {process_index} is not smaller than the process count {process_count} of the \
                         distributed runtime",
                    ),
            ));
        }
    }

    #[test]
    fn test_distributed_runtime_process_index() {
        assert_eq!(single_process_runtime().process_index(), 0);
    }

    #[test]
    fn test_distributed_runtime_process_count() {
        assert_eq!(single_process_runtime().process_count(), 1);
    }

    #[test]
    fn test_distributed_runtime_key_value_store() {
        let runtime = single_process_runtime();
        let store = runtime.key_value_store();
        store.put(b"key", b"value").unwrap();
        assert_eq!(store.try_get(b"key").unwrap(), b"value");
        assert_eq!(store.get(b"key", Duration::from_secs(1)).unwrap(), b"value");

        // The store keeps the first record of a key and reports a key without one.
        assert!(matches!(store.put(b"key", b"other"), Err(PjrtError::AlreadyExists { .. })));
        assert_eq!(store.try_get(b"key").unwrap(), b"value");
        assert!(matches!(store.try_get(b"missing"), Err(PjrtError::NotFound { .. })));
    }

    #[test]
    fn test_distributed_runtime_is_same_runtime() {
        let runtime = single_process_runtime();
        assert!(runtime.is_same_runtime(&runtime.clone()));
        assert!(!runtime.is_same_runtime(&single_process_runtime()));
    }

    #[test]
    fn test_distributed_runtime_shutdown() {
        // Every process of the job has disconnected once its only process has, and later calls do nothing.
        let runtime = single_process_runtime();
        assert!(matches!(runtime.shutdown(), Ok(())));
        assert!(matches!(runtime.clone().shutdown(), Ok(())));
    }

    #[test]
    fn test_distributed_runtime_reserve_kernel_coordinator() {
        let runtime = single_process_runtime();
        let (launch_id, sequence, process_index, process_count) = runtime.reserve_kernel_coordinator().unwrap();
        assert_eq!((sequence, process_index, process_count), (0, 0, 1));
        assert_eq!(runtime.clone().reserve_kernel_coordinator().unwrap(), (launch_id, 1, 0, 1));
        assert_ne!(single_process_runtime().reserve_kernel_coordinator().unwrap().0, launch_id);
    }

    #[test]
    fn test_distributed_runtime_compilation_artifact_exchange() {
        let exchange = single_process_runtime().compilation_artifact_exchange();
        assert_eq!((exchange.process_index(), exchange.process_count()), (0, 1));
        assert_eq!(exchange.validate_compatibility(b"key", 0, Duration::from_secs(1)), Ok(()));
        assert_eq!(exchange.publish(b"key", b"artifact"), Ok(()));
        assert_eq!(exchange.receive(b"key", Duration::from_secs(1)), Ok(Some(b"artifact".to_vec())));
    }

    #[test]
    fn test_distributed_runtime_create_client() {
        // The client takes its process from the runtime and uses the key-value store of the runtime.
        let runtime = single_process_runtime();
        let options = ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() });
        let client = runtime.create_client(&load_cpu_plugin().unwrap(), options).unwrap();
        assert_eq!(client.process_index().unwrap(), 0);
        assert_eq!(client.addressable_devices().unwrap().len(), 1);
        assert!(client.key_value_store().is_some());
    }

    #[test]
    fn test_distributed_runtime_client_options() {
        let runtime = single_process_runtime();

        // Unset coordinates are completed, and coordinates that agree with the runtime are kept.
        for process_id in [None, Some(0)] {
            assert_eq!(
                runtime
                    .client_options(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), process_id }))
                    .ok(),
                Some(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), process_id: Some(0) })),
            );
        }
        assert_eq!(
            runtime.client_options(ClientOptions::GPU(GpuClientOptions::default())).ok(),
            Some(ClientOptions::GPU(GpuClientOptions { node_id: Some(0), node_count: Some(1), ..Default::default() })),
        );
        assert_eq!(runtime.client_options(ClientOptions::default()).ok(), Some(ClientOptions::default()));

        // Coordinates of another process or of a job of another size are rejected.
        let rejected = [
            (ClientOptions::CPU(CpuClientOptions { process_id: Some(1), ..Default::default() }), "process_id", 1, 0),
            (ClientOptions::GPU(GpuClientOptions { node_id: Some(1), ..Default::default() }), "node_id", 1, 0),
            (ClientOptions::GPU(GpuClientOptions { node_count: Some(2), ..Default::default() }), "node_count", 2, 1),
        ];
        for (options, name, option, expected) in rejected {
            assert!(matches!(
                runtime.client_options(options),
                Err(Error::Pjrt(PjrtError::InvalidArgument { message, .. }))
                    if message == format!(
                        "client option `{name}` is {option}, but its distributed runtime has {expected}",
                    ),
            ));
        }
    }

    #[test]
    fn test_distributed_runtime_remote_devices() {
        // A client that lists a device of every process leaves nothing to declare.
        let runtime = single_process_runtime();
        let plugin = load_cpu_plugin().unwrap();
        let options = ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() });
        let client = runtime.create_client(&plugin, options).unwrap();
        assert!(matches!(runtime.remote_devices(&client), Ok(devices) if devices.is_empty()));

        // A client that was created without the runtime takes no part in its job.
        let options = ClientOptions::CPU(CpuClientOptions { device_count: Some(1), process_id: Some(1) });
        let unrelated = plugin.client(options.clone()).unwrap();
        assert!(matches!(
            runtime.remote_devices(&unrelated),
            Err(Error::Pjrt(PjrtError::FailedPrecondition { message, .. }))
                if message == "client does not use the key-value store of its distributed runtime; create it with \
                               `DistributedRuntime::create_client`",
        ));

        // A client whose options name another process cannot stand for the process of the runtime.
        let other = plugin.client_with_key_value_store(options, runtime.key_value_store()).unwrap();
        assert!(matches!(
            runtime.remote_devices(&other),
            Err(Error::Pjrt(PjrtError::FailedPrecondition { message, .. }))
                if message == "client reports process index 1, but its distributed runtime has process index 0; \
                               name the process in the options of the client",
        ));
    }

    #[test]
    fn test_distributed_runtime_remote_devices_cpu_process_helper() {
        let Ok(role) = env::var(PROCESS_TEST_ROLE) else {
            return;
        };
        let rank = role.parse::<usize>().unwrap();
        let address = env::var(PROCESS_TEST_ADDRESS).unwrap();
        let plugin = load_cpu_plugin().unwrap();
        let runtime = DistributedRuntime::initialize(&plugin, &address, 2, rank).unwrap();
        let options = ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() });

        // A session of a CPU client, which lists only its own device, declares the device of the other process, so
        // both processes know the same two devices and can describe one array over them.
        let session = Session::builder()
            .platform(Platform::Cpu)
            .client_options(options.clone())
            .distributed_runtime(runtime.clone())
            .build()
            .unwrap();
        let devices = [Device::new(0, 0), Device::new(2048, 1)];
        assert_eq!(session.client().process_index().unwrap(), rank);
        assert_eq!(session.addressable_devices().unwrap(), [devices[rank]]);
        assert_eq!(session.devices().unwrap(), devices);
        assert_eq!(session.remote_devices(), [devices[1 - rank]]);
        assert!(session.distributed_runtime().unwrap().is_same_runtime(&runtime));
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices.to_vec()).unwrap();
        let sharding = Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let array = Array::from_host_buffer(&session, r#type, mesh, Some(sharding), values_to_bytes(&[1f32, 2.]));
        assert_eq!(read_addressable_f32s(&array.unwrap()), [[[1f32], [2.]][rank]]);

        // A later session of the process shares the client and finds the same devices, and the runtime reports the
        // device of the other process for that client.
        let later = Session::builder()
            .platform(Platform::Cpu)
            .client_options(options)
            .distributed_runtime(runtime.clone())
            .build()
            .unwrap();
        assert!(ptr::eq(later.client(), session.client()));
        assert_eq!(later.remote_devices(), [devices[1 - rank]]);
        assert!(matches!(runtime.remote_devices(session.client()), Ok(remote) if remote == [devices[1 - rank]]));

        // Both processes have finished before either disconnects from the coordination service.
        let finished = |rank: usize| format!("ryft/test/remote-devices/finished/{rank}");
        let store = runtime.key_value_store();
        store.put(finished(rank).as_bytes(), b"ready").unwrap();
        assert_eq!(store.get(finished(1 - rank).as_bytes(), Duration::from_secs(10)).unwrap(), b"ready");
    }

    #[test]
    fn test_distributed_runtime_remote_devices_exchanges_devices_of_two_cpu_processes() {
        let directory = tempfile::tempdir().unwrap();
        run_cpu_processes(
            &loopback_address(),
            directory.path(),
            2,
            "distributed::tests::test_distributed_runtime_remote_devices_cpu_process_helper",
        );
    }

    #[test]
    fn test_distributed_compilation_artifact_exchange_process_index() {
        assert_eq!(exchange(&single_process_runtime(), 1, 3).process_index(), 1);
    }

    #[test]
    fn test_distributed_compilation_artifact_exchange_process_count() {
        assert_eq!(exchange(&single_process_runtime(), 1, 3).process_count(), 3);
    }

    #[test]
    fn test_distributed_compilation_artifact_exchange_validate_compatibility() {
        // The leader announces its key without waiting for anyone, and a follower accepts that announcement.
        let runtime = single_process_runtime();
        let (leader, follower) = (exchange(&runtime, 0, 2), exchange(&runtime, 1, 2));
        assert_eq!(leader.validate_compatibility(b"key", 0, Duration::from_secs(1)), Ok(()));
        assert_eq!(follower.validate_compatibility(b"key", 0, Duration::from_secs(1)), Ok(()));

        // Processes 1 and 2 of three compile a program that runs only on their devices, so process 1 leads it and
        // process 0 takes no part.
        let (leader, follower) = (exchange(&runtime, 1, 3), exchange(&runtime, 2, 3));
        assert_eq!(leader.validate_compatibility(b"other-key", 1, Duration::from_secs(1)), Ok(()));
        assert_eq!(follower.validate_compatibility(b"other-key", 1, Duration::from_secs(1)), Ok(()));
    }

    #[test]
    fn test_distributed_compilation_artifact_exchange_validate_compatibility_accepts_repeated_announcements() {
        // The leader compiles programs that the follower skips (e.g., because its in-memory cache already holds them)
        // and compiles one program twice (e.g., after evicting it), and the follower validates in another order.
        let runtime = single_process_runtime();
        let (leader, follower) = (exchange(&runtime, 0, 2), exchange(&runtime, 1, 2));
        for key in [b"first-key".as_slice(), b"second-key", b"third-key", b"first-key"] {
            assert_eq!(leader.validate_compatibility(key, 0, Duration::from_secs(1)), Ok(()));
        }
        assert_eq!(follower.validate_compatibility(b"third-key", 0, Duration::from_secs(1)), Ok(()));
        assert_eq!(follower.validate_compatibility(b"first-key", 0, Duration::from_secs(1)), Ok(()));
    }

    #[test]
    fn test_distributed_compilation_artifact_exchange_validate_compatibility_rejects_invalid_coordinates() {
        let runtime = single_process_runtime();
        let invalid = Err(CompilationArtifactExchangeError::Incompatible {
            message: "exchange process coordinates are invalid".to_string(),
        });
        assert_eq!(exchange(&runtime, 0, 0).validate_compatibility(b"key", 0, Duration::from_secs(1)), invalid);
        assert_eq!(exchange(&runtime, 2, 2).validate_compatibility(b"key", 0, Duration::from_secs(1)), invalid);
        assert_eq!(exchange(&runtime, 1, 2).validate_compatibility(b"key", 2, Duration::from_secs(1)), invalid);
    }

    #[test]
    fn test_distributed_compilation_artifact_exchange_validate_compatibility_rejects_oversized_keys() {
        assert_eq!(
            exchange(&single_process_runtime(), 0, 2).validate_compatibility(&[0; 33], 0, Duration::from_secs(1)),
            Err(CompilationArtifactExchangeError::Incompatible {
                message: "persistent compilation key size 33 exceeds configured compatibility validation maximum 32"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_distributed_compilation_artifact_exchange_validate_compatibility_times_out_without_announcement() {
        // A follower whose key differs from that of its leader never observes an announcement for its key.
        let runtime = single_process_runtime();
        let (leader, follower) = (exchange(&runtime, 0, 2), exchange(&runtime, 1, 2));
        assert_eq!(leader.validate_compatibility(b"first-key", 0, Duration::from_secs(1)), Ok(()));
        assert_eq!(
            follower.validate_compatibility(b"second-key", 0, Duration::from_millis(20)),
            Err(CompilationArtifactExchangeError::TimedOut),
        );
    }

    #[test]
    fn test_distributed_compilation_artifact_exchange_validate_compatibility_rejects_process_count_disagreement() {
        let runtime = single_process_runtime();
        assert_eq!(exchange(&runtime, 0, 2).validate_compatibility(b"key", 0, Duration::from_secs(1)), Ok(()));
        assert_eq!(
            exchange(&runtime, 1, 3).validate_compatibility(b"key", 0, Duration::from_secs(1)),
            Err(CompilationArtifactExchangeError::Incompatible {
                message: "process 0 expects 2 processes, but this process expects 3".to_string(),
            }),
        );
    }

    #[test]
    fn test_distributed_compilation_artifact_exchange_validate_compatibility_rejects_other_schema_versions() {
        let runtime = single_process_runtime();
        let follower = exchange(&runtime, 1, 2);
        let manifest = CompilationCompatibilityManifest { schema_version: 1, process_count: 2 };
        let manifest = serde_json::to_vec(&manifest).unwrap();
        runtime.key_value_store().put(&follower.key(b"key", b"preflight/manifest"), &manifest).unwrap();
        assert_eq!(
            follower.validate_compatibility(b"key", 0, Duration::from_secs(1)),
            Err(CompilationArtifactExchangeError::Incompatible {
                message: format!(
                    "process 0 uses compilation artifact exchange schema 1, but this process uses schema \
                     {ARTIFACT_EXCHANGE_SCHEMA_VERSION}",
                ),
            }),
        );
    }

    #[test]
    fn test_distributed_compilation_artifact_exchange_publish() {
        // The artifact spans three chunks of at most three bytes.
        let runtime = single_process_runtime();
        let (leader, follower) = (exchange(&runtime, 0, 2), exchange(&runtime, 1, 2));
        assert_eq!(leader.publish(b"key", b"abcdefgh"), Ok(()));
        assert_eq!(follower.receive(b"key", Duration::from_secs(1)), Ok(Some(b"abcdefgh".to_vec())));

        // An artifact without bytes has a manifest and no chunks.
        assert_eq!(leader.publish(b"empty-key", b""), Ok(()));
        assert_eq!(follower.receive(b"empty-key", Duration::from_secs(1)), Ok(Some(Vec::new())));
    }

    #[test]
    fn test_distributed_compilation_artifact_exchange_publish_keeps_first_artifact() {
        // The leader compiles a program again after evicting it, and the first artifact stays authoritative.
        let runtime = single_process_runtime();
        let leader = exchange(&runtime, 0, 2);
        assert_eq!(leader.publish(b"key", b"first"), Ok(()));
        assert_eq!(leader.publish(b"key", b"recompiled"), Ok(()));
        assert_eq!(exchange(&runtime, 1, 2).receive(b"key", Duration::from_secs(1)), Ok(Some(b"first".to_vec())));
    }

    #[test]
    fn test_distributed_compilation_artifact_exchange_publish_completes_partial_publication() {
        // A publication stored its first chunk and failed before its manifest, and another serialization of the same
        // program left a chunk too. The store never overwrites a record, so a repeated publication must not store
        // that first chunk again, and its followers must not read the chunk of the other serialization.
        let runtime = single_process_runtime();
        let leader = exchange(&runtime, 0, 2);
        let store = runtime.key_value_store();
        store.put(&leader.chunk_key(b"key", &Sha256::digest(b"abcdef").into(), 0), b"abc").unwrap();
        store.put(&leader.chunk_key(b"key", &Sha256::digest(b"uvwxyz").into(), 0), b"uvw").unwrap();
        assert_eq!(leader.publish(b"key", b"abcdef"), Ok(()));
        assert_eq!(exchange(&runtime, 1, 2).receive(b"key", Duration::from_secs(1)), Ok(Some(b"abcdef".to_vec())));
    }

    #[test]
    fn test_distributed_compilation_artifact_exchange_publish_rejects_oversized_artifacts() {
        assert_eq!(
            exchange(&single_process_runtime(), 0, 2).publish(b"key", &[0; 33]),
            Err(CompilationArtifactExchangeError::Failed {
                message: "compiled artifact size 33 exceeds configured maximum 32".to_string(),
            }),
        );
    }

    #[test]
    fn test_distributed_compilation_artifact_exchange_publish_failure() {
        let runtime = single_process_runtime();
        let leader = exchange(&runtime, 0, 2);
        assert_eq!(leader.publish_failure(b"key", "backend compilation failed"), Ok(()));
        assert_eq!(
            exchange(&runtime, 1, 2).receive(b"key", Duration::from_secs(1)),
            Err(CompilationArtifactExchangeError::Failed {
                message: "compilation producer failed: backend compilation failed".to_string(),
            }),
        );
    }

    #[test]
    fn test_distributed_compilation_artifact_exchange_publish_failure_stays_authoritative() {
        // Neither a later failure nor a later artifact replaces a published failure.
        let runtime = single_process_runtime();
        let leader = exchange(&runtime, 0, 2);
        assert_eq!(leader.publish_failure(b"key", "backend compilation failed"), Ok(()));
        assert_eq!(leader.publish_failure(b"key", "backend compilation failed again"), Ok(()));
        assert_eq!(
            leader.publish(b"key", b"artifact"),
            Err(CompilationArtifactExchangeError::Failed {
                message: "a terminal failure was already published for this compilation key".to_string(),
            }),
        );
        assert_eq!(
            exchange(&runtime, 1, 2).receive(b"key", Duration::from_secs(1)),
            Err(CompilationArtifactExchangeError::Failed {
                message: "compilation producer failed: backend compilation failed".to_string(),
            }),
        );

        // A failure that follows an artifact leaves the artifact in place as well.
        assert_eq!(leader.publish(b"other-key", b"artifact"), Ok(()));
        assert_eq!(leader.publish_failure(b"other-key", "backend compilation failed"), Ok(()));
        assert_eq!(leader.receive(b"other-key", Duration::from_secs(1)), Ok(Some(b"artifact".to_vec())));
    }

    #[test]
    fn test_distributed_compilation_artifact_exchange_receive() {
        // A follower that starts waiting before the leader publishes receives the artifact once it is published.
        let runtime = single_process_runtime();
        let (leader, follower) = (exchange(&runtime, 0, 2), exchange(&runtime, 1, 2));
        thread::scope(|scope| {
            let received = scope.spawn(|| follower.receive(b"key", Duration::from_secs(5)));
            assert_eq!(leader.publish(b"key", b"abcdefgh"), Ok(()));
            assert_eq!(received.join().unwrap(), Ok(Some(b"abcdefgh".to_vec())));
        });
    }

    #[test]
    fn test_distributed_compilation_artifact_exchange_receive_reports_missing_artifacts() {
        // The leader announced its key and never published an artifact for it.
        let runtime = single_process_runtime();
        let (leader, follower) = (exchange(&runtime, 0, 2), exchange(&runtime, 1, 2));
        assert_eq!(leader.validate_compatibility(b"key", 0, Duration::from_secs(1)), Ok(()));
        assert_eq!(follower.receive(b"key", Duration::from_millis(20)), Ok(None));
    }

    #[test]
    fn test_distributed_compilation_artifact_exchange_receive_rejects_chunks_that_differ_from_their_manifest() {
        let runtime = single_process_runtime();
        let follower = exchange(&runtime, 1, 2);
        let store = runtime.key_value_store();
        let checksum: [u8; 32] = Sha256::digest(b"abcdef").into();
        let publish = |key: &[u8], chunks: [&[u8]; 2]| {
            for (index, chunk) in chunks.into_iter().enumerate() {
                store.put(&follower.chunk_key(key, &checksum, index), chunk).unwrap();
            }
            let manifest = CompilationArtifactManifest {
                schema_version: ARTIFACT_EXCHANGE_SCHEMA_VERSION,
                size: 6,
                chunk_count: 2,
                checksum,
                failure: None,
            };
            store.put(&follower.key(key, b"manifest"), &serde_json::to_vec(&manifest).unwrap()).unwrap();
        };
        let failed = |message: &str| Err(CompilationArtifactExchangeError::Failed { message: message.to_string() });

        publish(b"corrupt", [b"xbc", b"def"]);
        assert_eq!(follower.receive(b"corrupt", Duration::from_secs(1)), failed("compiled artifact checksum mismatch"));
        publish(b"reordered", [b"def", b"abc"]);
        assert_eq!(
            follower.receive(b"reordered", Duration::from_secs(1)),
            failed("compiled artifact checksum mismatch")
        );
        publish(b"long-chunk", [b"abcd", b"ef"]);
        assert_eq!(
            follower.receive(b"long-chunk", Duration::from_secs(1)),
            failed("compiled artifact chunk violates manifest bounds"),
        );
    }

    #[test]
    fn test_distributed_compilation_artifact_exchange_receive_rejects_manifests_outside_bounds() {
        let runtime = single_process_runtime();
        let follower = exchange(&runtime, 1, 2);
        let publish = |key: &[u8], schema_version: u32, size: u64, chunk_count: u64| {
            let manifest =
                CompilationArtifactManifest { schema_version, size, chunk_count, checksum: [0; 32], failure: None };
            let manifest = serde_json::to_vec(&manifest).unwrap();
            runtime.key_value_store().put(&follower.key(key, b"manifest"), &manifest).unwrap();
        };
        let failed = |message: &str| Err(CompilationArtifactExchangeError::Failed { message: message.to_string() });

        publish(b"other-schema", 1, 6, 2);
        assert_eq!(
            follower.receive(b"other-schema", Duration::from_secs(1)),
            failed("unsupported compiled artifact exchange schema 1"),
        );
        publish(b"oversized", ARTIFACT_EXCHANGE_SCHEMA_VERSION, 33, 11);
        assert_eq!(
            follower.receive(b"oversized", Duration::from_secs(1)),
            failed("compiled artifact manifest exceeds configured bounds"),
        );
        publish(b"other-chunk-count", ARTIFACT_EXCHANGE_SCHEMA_VERSION, 6, 3);
        assert_eq!(
            follower.receive(b"other-chunk-count", Duration::from_secs(1)),
            failed("compiled artifact manifest exceeds configured bounds"),
        );
    }

    #[test]
    fn test_distributed_compilation_artifact_exchange_cpu_process_helper() {
        let Ok(role) = env::var(PROCESS_TEST_ROLE) else {
            return;
        };
        let process_index = role.parse::<usize>().unwrap();
        let address = env::var(PROCESS_TEST_ADDRESS).unwrap();
        let plugin = load_cpu_plugin().unwrap();
        let runtime = DistributedRuntime::initialize(&plugin, &address, 2, process_index).unwrap();
        let exchange = runtime.compilation_artifact_exchange();
        assert_eq!((exchange.process_index(), exchange.process_count()), (process_index, 2));
        assert_eq!(exchange.validate_compatibility(b"multi-process-key", 0, Duration::from_secs(5)), Ok(()));

        // The leader keeps the coordination service alive until the follower has received the artifact.
        if process_index == 0 {
            assert_eq!(exchange.publish(b"multi-process-key", b"serialized-executable"), Ok(()));
            runtime.key_value_store().get(PROCESS_TEST_ACKNOWLEDGEMENT, Duration::from_secs(5)).unwrap();
        } else {
            assert_eq!(
                exchange.receive(b"multi-process-key", Duration::from_secs(5)),
                Ok(Some(b"serialized-executable".to_vec())),
            );
            runtime.key_value_store().put(PROCESS_TEST_ACKNOWLEDGEMENT, b"ready").unwrap();
        }
    }

    #[test]
    fn test_distributed_compilation_artifact_exchange_shares_artifacts_between_two_cpu_processes() {
        let directory = tempfile::tempdir().unwrap();
        run_cpu_processes(
            &loopback_address(),
            directory.path(),
            2,
            "distributed::tests::test_distributed_compilation_artifact_exchange_cpu_process_helper",
        );
    }

    fn run_cpu_processes(address: &str, directory: &std::path::Path, process_count: usize, helper: &str) {
        let executable = env::current_exe().unwrap();
        let mut children = (0..process_count)
            .map(|process_index| {
                Command::new(&executable)
                    .args(["--exact", helper, "--nocapture", "--test-threads=1"])
                    .env(PROCESS_TEST_ROLE, process_index.to_string())
                    .env(PROCESS_TEST_ADDRESS, address)
                    .stdout(File::create(directory.join(format!("rank-{process_index}.stdout"))).unwrap())
                    .stderr(File::create(directory.join(format!("rank-{process_index}.stderr"))).unwrap())
                    .spawn()
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let deadline = Instant::now() + Duration::from_secs(30);
        let mut statuses = vec![None; process_count];
        while statuses.iter().any(Option::is_none) && Instant::now() < deadline {
            for (status, child) in statuses.iter_mut().zip(&mut children) {
                if status.is_none() {
                    *status = child.try_wait().unwrap();
                }
            }
            if statuses.iter().flatten().any(|status| !status.success()) {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        for (status, child) in statuses.iter().zip(&mut children) {
            if status.is_none() {
                let _ = child.kill();
                child.wait().unwrap();
            }
        }
        if statuses.iter().any(|status| !status.is_some_and(|status| status.success())) {
            let report = statuses
                .into_iter()
                .enumerate()
                .map(|(rank, status)| {
                    let stdout = std::fs::read_to_string(directory.join(format!("rank-{rank}.stdout"))).unwrap();
                    let stderr = std::fs::read_to_string(directory.join(format!("rank-{rank}.stderr"))).unwrap();
                    format!("rank {rank} exited with {status:?}\nstdout:\n{stdout}\nstderr:\n{stderr}")
                })
                .collect::<Vec<_>>()
                .join("\n");
            panic!("distributed process test `{helper}` failed:\n{report}");
        }
        for rank in 0..process_count {
            let stdout = std::fs::read_to_string(directory.join(format!("rank-{rank}.stdout"))).unwrap();
            assert!(
                stdout.contains(&format!("test {helper} ... "))
                    && stdout.contains("test result: ok. 1 passed; 0 failed; 0 ignored;"),
                "rank {rank} did not pass `{helper}`:\n{stdout}"
            );
        }
    }

    #[test]
    fn test_placement_comparison_cpu_process_helper() {
        let Ok(role) = env::var(PROCESS_TEST_ROLE) else {
            return;
        };
        let rank = role.parse::<usize>().unwrap();
        let address = env::var(PROCESS_TEST_ADDRESS).unwrap();
        let plugin = load_cpu_plugin().unwrap();
        let runtime = Box::leak(Box::new(DistributedRuntime::initialize(&plugin, &address, 2, rank).unwrap()));
        let options = ClientOptions::CPU(CpuClientOptions { device_count: Some(1), process_id: Some(rank) });
        let client: &'static Client<'static> = Box::leak(Box::new(runtime.create_client(&plugin, options).unwrap()));
        assert_eq!(client.devices().unwrap().len(), 1);
        let local = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let key = |rank| format!("ryft/test/placement/device/{rank}");
        let store = runtime.key_value_store();
        let timeout = Duration::from_secs(5);
        store.put(key(rank).as_bytes(), &local.id().to_le_bytes()).unwrap();
        let devices = [0, 1].map(|rank| {
            let bytes = store.get(key(rank).as_bytes(), timeout).unwrap();
            Device::new(usize::from_le_bytes(bytes.try_into().unwrap()), rank)
        });
        let new_session = || Session::builder().client(client).remote_devices([devices[1 - rank]]);
        let session = new_session().build().unwrap();
        assert_eq!(session.devices().unwrap(), devices);
        let mesh = session.mesh([MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        assert_eq!(mesh.devices(), devices);
        let logical_mesh = mesh.logical_mesh().clone();
        let target = Placement::new(mesh, Sharding::replicated(logical_mesh, 1)).unwrap();
        let r#type = ArrayType::new_static(DataType::U8, [2]);
        let host = || CoreArray::new(r#type.clone(), vec![rank as u8; 2]).unwrap();
        let check_rejected = |output: Result<Array, Error>| {
            assert!(matches!(output, Err(Error::Session(SessionError::PlacementComparisonUnavailable))));
        };

        // The declarations make both ranks reject even when one candidate can see every device.
        let comparison = ValueComparison::new(target.mesh().clone(), vec![local]);
        let candidate = (rank == 0).then_some(&comparison);
        let comparison = candidate.or(ValueComparison::of_session(&session).unwrap());
        check_rejected(session.place_with_comparison(host(), PlacementOptions::to(target.clone()), comparison));
        check_rejected(session.place(host(), PlacementOptions::to(target.clone())));
        let uploaded = Array::from_host_buffer(&session, r#type.clone(), None, None, [rank as u8; 2]).unwrap();
        let core_error = uploaded.place(&target, None).unwrap_err();
        assert_eq!(core_error.downcast_custom::<SessionError>(), Some(&SessionError::PlacementComparisonUnavailable));
        check_rejected(uploaded.to_placement(&session, target.clone()));
        check_rejected(uploaded.into_placement(&session, target.clone()));

        // Local targets and explicitly unchecked placements use no comparison.
        let local_placement = session.place(3u8, PlacementOptions::to(local)).unwrap();
        assert_eq!(local_placement.copy_to_host_bytes().unwrap(), [3]);
        let unchecked = new_session().value_comparison(false).build().unwrap();
        let placed = unchecked.place(host(), PlacementOptions::to(target)).unwrap();
        assert_eq!(placed.copy_to_host_bytes().unwrap(), [rank as u8; 2]);

        // CPU rejects the unsupported layout on the owner and nonowner before either can submit a cross-host leaf.
        let committed = unchecked.place(host(), PlacementOptions::to(devices[0])).unwrap();
        assert_eq!(committed.addressable_shards().count(), usize::from(rank == 0));
        let layouts = [
            (Layout::Strided(StridedLayout::new(vec![2])), "cpu PJRT does not support strided device layouts"),
            (
                Layout::Tiled(TiledLayout::new(vec![0], vec![Tile::new(vec![TileDimension::Sized(2)])])),
                "cpu PJRT does not support device layouts with tiles",
            ),
        ];
        for (layout, layout_error) in layouts {
            let upload = PlacementRequest::Device { device: devices[0], memory: None, layout: Some(layout.clone()) };
            let relayout = PlacementRequest::InPlace { memory: None, layout: Some(layout.clone()) };
            let transfer = PlacementRequest::from(devices[1]);
            let unsupported = |error: &Error| {
                matches!(error, Error::Pjrt(PjrtError::InvalidArgument { message, .. })
                    if message == layout_error)
            };
            let assert_leaf_error = |error: Error, expected_path: &str| {
                assert!(
                    matches!(&error, Error::Array(error) if matches!(error.as_ref(),
                        ArrayError::FailedParameterPlacement { source, path }
                            if path.to_string() == expected_path && unsupported(source)
                    )),
                    "{error:?}"
                );
            };
            let statistics = unchecked.array_statistics();
            let error = unchecked
                .place(
                    (PlaceInput::from(host()), PlaceInput::from(committed.clone())),
                    PlacementOptions::to((upload, transfer.clone())),
                )
                .unwrap_err();
            assert_leaf_error(error, "$.0");
            let error = unchecked
                .place(
                    (committed.clone(), committed.clone()),
                    PlacementOptions::to((relayout.clone(), transfer.clone())),
                )
                .unwrap_err();
            assert_leaf_error(error, "$.0");
            let error = unchecked
                .place((committed.clone(), committed.clone()), PlacementOptions::to((transfer, relayout.clone())))
                .unwrap_err();
            assert_leaf_error(error, "$.1");

            // Direct storage conversion and host constructors share the format check for empty upload lists too.
            let source_mesh = committed.mesh().clone();
            let bytes = [7u8, 9];
            let destination_type = r#type.clone().with_layout(layout.clone());
            let dense =
                Array::from_host_buffer(&unchecked, destination_type.clone(), source_mesh.clone(), None, &bytes);
            let shard = (rank == 0).then_some((devices[0].id(), &bytes));
            let sharded = Array::from_host_shards(&unchecked, destination_type, source_mesh.clone(), None, shard);
            let directly_placed = committed.to_placement(&unchecked, relayout.clone());
            let converted = committed.clone().place_storage(Memory::Device, Some(layout.clone()));
            for rejected in [dense, sharded, directly_placed, converted] {
                let error = rejected.unwrap_err();
                assert!(unsupported(&error), "{error:?}");
            }
            assert_eq!(unchecked.array_statistics().created_arrays, statistics.created_arrays);

            let unchanged = committed.to_placement(&unchecked, devices[0]).unwrap();
            assert_eq!(unchanged, committed);

            let zero_type = ArrayType::new_static(DataType::Zero, [2]);
            let zero = Array::from_host_buffer(&unchecked, zero_type.clone(), source_mesh.clone(), None, []).unwrap();
            let placed = unchecked.place(zero, PlacementOptions::to(relayout)).unwrap();
            let zero_type = zero_type.with_layout(layout.clone());
            let zero_shard = (rank == 0).then_some((devices[0].id(), &[] as &[u8]));
            let sharded =
                Array::from_host_shards(&unchecked, zero_type.clone(), source_mesh.clone(), None, zero_shard).unwrap();
            let dense = Array::from_host_buffer(&unchecked, zero_type, source_mesh, None, []).unwrap();
            for zero in [placed, sharded, dense] {
                assert_eq!(zero.layout(), Some(&layout));
                assert!(zero.addressable_shards().all(|shard| shard.buffer().is_none()));
                zero.block_until_ready().unwrap();
            }
        }

        let finished = |rank| format!("ryft/test/placement/finished/{rank}");
        store.put(finished(rank).as_bytes(), b"ready").unwrap();
        store.get(finished(1 - rank).as_bytes(), timeout).unwrap();
        runtime.shutdown().unwrap();
    }

    #[test]
    fn test_placement_comparison_on_two_cpu_processes() {
        let address = loopback_address();
        let directory = tempfile::tempdir().unwrap();
        run_cpu_processes(
            &address,
            directory.path(),
            2,
            "distributed::tests::test_placement_comparison_cpu_process_helper",
        );
    }

    #[test]
    fn test_cross_client_singleton_placement_cpu_process_helper() {
        let Ok(role) = env::var(PROCESS_TEST_ROLE) else {
            return;
        };
        let rank = role.parse::<usize>().unwrap();
        let address = env::var(PROCESS_TEST_ADDRESS).unwrap();
        let plugin = load_cpu_plugin().unwrap();
        let runtime: &'static DistributedRuntime =
            Box::leak(Box::new(DistributedRuntime::initialize(&plugin, &address, 2, rank).unwrap()));
        // One of the two clients names another process than the runtime has, which `DistributedRuntime::create_client`
        // rejects, so both are created from the key-value store of the runtime directly.
        let client = |process_id| {
            let options = ClientOptions::CPU(CpuClientOptions { device_count: Some(1), process_id: Some(process_id) });
            &*Box::leak(Box::new(plugin.client_with_key_value_store(options, runtime.key_value_store()).unwrap()))
        };
        let source_client = client(1 - rank);
        let target_client = client(rank);
        assert_eq!(source_client.process_index().unwrap(), 1 - rank);
        assert_eq!(target_client.process_index().unwrap(), rank);
        let store = runtime.key_value_store();
        let timeout = Duration::from_secs(5);
        let owner_device = |key: &[u8], client: &'static Client<'static>, process| {
            if rank == 0 {
                let addressable = client.addressable_devices().unwrap();
                store.put(key, &addressable[0].id().unwrap().to_le_bytes()).unwrap();
            }
            let id = usize::from_le_bytes(store.get(key, timeout).unwrap().try_into().unwrap());
            Device::new(id, process)
        };
        let source_device = owner_device(b"ryft/test/host-bridge/source", source_client, 1);
        let target_device = owner_device(b"ryft/test/host-bridge/target", target_client, 0);
        let session = |client, device| {
            Session::builder()
                .client(client)
                .remote_devices((rank == 1).then_some(device))
                .value_comparison(false)
                .build()
                .unwrap()
        };
        let source_session = session(source_client, source_device);
        let target_session = session(target_client, target_device);
        let source_mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![source_device]).unwrap();
        let r#type = ArrayType::new_static(DataType::U8, [1]);
        let source = Array::from_host_buffer(&source_session, r#type, source_mesh, None, [17u8]).unwrap();
        if rank == 0 {
            assert_eq!(source.copy_to_host_bytes().unwrap(), [17u8]);
        }
        let error = source.to_placement(&target_session, target_device).unwrap_err();
        assert!(matches!(
            error,
            Error::Pjrt(PjrtError::InvalidArgument { message, .. })
                if message.starts_with("cannot place an array of another PJRT client onto the devices of process 0"),
        ));

        let finished = |rank| format!("ryft/test/host-bridge/finished/{rank}");
        store.put(finished(rank).as_bytes(), b"ready").unwrap();
        store.get(finished(1 - rank).as_bytes(), timeout).unwrap();
        runtime.shutdown().unwrap();
    }

    #[test]
    fn test_cross_client_singleton_placement_on_two_cpu_processes() {
        let address = loopback_address();
        let directory = tempfile::tempdir().unwrap();
        run_cpu_processes(
            &address,
            directory.path(),
            2,
            "distributed::tests::test_cross_client_singleton_placement_cpu_process_helper",
        );
    }

    #[test]
    fn test_shard_map_multi_host_scenarios_on_one_process_with_two_cpu_devices() {
        // The single-process baseline of the multi-host scenarios, which reads every shard because this process
        // addresses both devices. The client is minted through a one-node runtime, as in a distributed job.
        let address = loopback_address();
        let plugin = load_cpu_plugin().unwrap();
        // Sessions need a client that lives for the process, and so does the runtime whose key-value store it borrows.
        let runtime: &'static DistributedRuntime =
            Box::leak(Box::new(DistributedRuntime::initialize(&plugin, &address, 1, 0).unwrap()));
        let client: &'static Client<'static> = Box::leak(Box::new(
            runtime
                .create_client(
                    &plugin,
                    ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }),
                )
                .unwrap(),
        ));
        check_shard_map_multi_host_scenarios(client);
    }

    /// Joins the two-process CUDA job that [`run_cuda_rank_pair`] launched this process into as rank `process_index`,
    /// runs `check` with a client whose two global devices are the ranks' shares of CUDA device 0, and then shuts the
    /// job down in order.
    #[cfg(feature = "cuda-13")]
    fn run_as_cuda_rank(process_index: usize, check: impl FnOnce(&'static Client<'static>)) {
        let address = env::var(MULTI_HOST_TEST_ADDRESS).unwrap();
        let timeout = Duration::from_secs(60);
        let plugin = load_cuda_13_plugin().unwrap();
        // Sessions need a client that lives for the rest of the process, and so does the runtime whose key-value store
        // the client borrows, so neither is ever dropped.
        let runtime: &'static DistributedRuntime = Box::leak(Box::new(
            DistributedRuntime::initialize_with_options(
                &plugin,
                &address,
                None,
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
            .unwrap(),
        ));

        // Every array and executable of `check` is dropped before the coordination client shuts down, while rank zero
        // keeps the coordination service alive.
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
            let client: &'static Client<'static> = Box::leak(Box::new(client));
            check(client);
        }
        let completed = |rank: usize| format!("ryft/test/cuda-ranks/completed/{rank}");
        runtime.key_value_store().put(completed(process_index).as_bytes(), b"ready").unwrap();
        for rank in 0..2 {
            assert_eq!(runtime.key_value_store().get(completed(rank).as_bytes(), timeout).unwrap(), b"ready");
        }
        runtime.shutdown().unwrap();
    }

    /// Runs the test `helper`, given as a path within this test binary, as the two ranks of a CUDA job that share CUDA
    /// device 0, asserts that both ranks pass, and returns their standard outputs. The test fails if the coordinator
    /// cannot bind a loopback port.
    ///
    /// The CPU plugin cannot host such jobs, because a CPU client that joins a distributed runtime sees only its own
    /// devices (e.g., process 1 sees just device 2048), so it has no cross-process topology for collectives. Instead,
    /// both ranks share CUDA device 0 through NCCL's multiple ranks per GPU, with CUDA MPS for concurrent progress,
    /// which mirrors the `cuda-distributed-collectives` differential suite that `python/README.md` describes.
    #[cfg(feature = "cuda-13")]
    fn run_cuda_rank_pair(helper: &str) -> [String; 2] {
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

        let address = loopback_address();
        let directory = tempfile::tempdir().unwrap();
        let mut environment = vec![
            ("NCCL_MULTI_RANK_GPU_ENABLE", "1".to_string()),
            ("NCCL_NVLS_ENABLE", "0".to_string()),
            ("NCCL_MAX_CTAS", "1".to_string()),
            ("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "45".to_string()),
            ("CUDA_VISIBLE_DEVICES", env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "0".to_string())),
        ];
        // A caller-owned MPS daemon is used as is. Otherwise, this test owns a private daemon for the two ranks.
        let _daemon = if env::var_os("CUDA_MPS_PIPE_DIRECTORY").is_some() {
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
        let executable = env::current_exe().unwrap();
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
        // A failed rank leaves its peer to be killed, so the report shows the outputs of both ranks, whichever of them
        // failed first.
        let outputs = [0, 1].map(|rank| {
            let stderr = std::fs::read_to_string(directory.path().join(format!("rank-{rank}.stderr"))).unwrap();
            let stdout = std::fs::read_to_string(directory.path().join(format!("rank-{rank}.stdout"))).unwrap();
            (statuses[rank].unwrap(), stdout, stderr)
        });
        let report = outputs
            .iter()
            .enumerate()
            .map(|(rank, (status, stdout, stderr))| {
                format!("rank {rank} exited with {status}\nstdout:\n{stdout}\nstderr:\n{stderr}")
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(outputs.iter().all(|(status, ..)| status.success()), "a rank of `{helper}` failed\n{report}");
        for (rank, (_, stdout, _)) in outputs.iter().enumerate() {
            assert!(
                stdout.contains(&format!("test {helper} ... "))
                    && stdout.contains("test result: ok. 1 passed; 0 failed; 0 ignored;"),
                "rank {rank} did not pass `{helper}`:\n{stdout}"
            );
        }
        outputs.map(|(_, stdout, _)| stdout)
    }

    // Worker of `test_shard_map_multi_host_scenarios_on_two_cuda_processes`, which returns immediately unless that
    // test launched it as one of its two ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_shard_map_multi_host_scenarios_cuda_process_helper() {
        let Ok(role) = env::var(MULTI_HOST_TEST_ROLE) else {
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

    // Worker of `test_gpu_only_collectives_on_two_cuda_processes`, which returns immediately unless that test launched
    // it as one of its two ranks. It compiles and executes each probe module with this rank's tile of `[1, 2, 3, 4]`
    // and prints one `probe:` line per module with its outputs or the first line of its compilation or execution
    // error.
    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_gpu_only_collectives_cuda_process_helper() {
        // JAX's CUDA lowerings of `shard_map` over a mesh `x = 2` whose input `f32[4]` is sharded along `x`, produced
        // by `.tasks/shard_map_jax_parity_gaps/m8_lower.py` with JAX 0.12.1.dev20261103 over jaxlib 0.12.0.1 and kept
        // verbatim. They cover `pbroadcast` (i.e., `collective_broadcast`), the asynchronous `*-start` and `*-done`
        // custom calls whose configuration is in the `async_collective_config` frontend attribute, and token-threaded
        // `send` and `recv`. The `send` and `recv` modules run last, because a mismatched point-to-point transfer would
        // hang both ranks.
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
        // `0 -> 1`, which is the module's only change, checks whether point-to-point transfers work at all.
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
        let Ok(role) = env::var(MULTI_HOST_TEST_ROLE) else {
            return;
        };
        run_as_cuda_rank(role.parse().unwrap(), |client| {
            let rank = client.process_index().unwrap();
            let device = client.addressable_devices().unwrap().remove(0);
            for (name, module) in modules {
                let compiled = client.compile(
                    &PjrtProgram::Mlir { bytecode: module.trim_start().as_bytes().to_vec() },
                    &test_compilation_options(client, Some(2)),
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
        // Pins what Ryft's pinned XLA supports of JAX's GPU-only collectives across two processes, as each rank
        // observes it. Following JAX's semantics, `pbroadcast` from source 1 gives both ranks `[3, 4]`, the swapping
        // `ppermute` gives `[3, 4]` and `[1, 2]`, the tiled all-gather gives both ranks `[1, 2, 3, 4]`, the tiled sum
        // scatter of the sum `[4, 6]` gives `[4]` and `[6]`, and the all-to-all gives `[1, 3]` and `[2, 4]`. XLA
        // rejects the swapping `send` and `recv`, whose pairs form a cycle, while the acyclic variant delivers `[1, 2]`
        // to rank 1 and leaves the received value of rank 0, which no pair targets, zero.
        let outputs = run_cuda_rank_pair("distributed::tests::test_gpu_only_collectives_cuda_process_helper");
        // libtest prints `test <name> ... ` without a line break before the helper runs, so the first probe line of
        // each rank follows that prefix.
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

    // Worker of `test_parallel_broadcast_on_two_cuda_processes`, which returns immediately unless that test launched
    // it as one of its two ranks. It stages a `shard_map` of `parallel_broadcast` over a mesh `x = 2` whose input
    // `f32[4]` is sharded along `x`, lowers it for this CUDA client through Ryft, executes it with this rank's tile of
    // `[1, 2, 3, 4]`, and prints a `probe:` line with this rank's output shard and the custom call targets of the
    // lowered module.
    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_parallel_broadcast_cuda_process_helper() {
        let Ok(role) = env::var(MULTI_HOST_TEST_ROLE) else {
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
            let session = Session::builder().client(client).build().unwrap();
            let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
            let mesh = DeviceMesh::new(manual_mesh.clone(), devices).unwrap();
            let sharded = Sharding::new(manual_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
            let input_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap();
            let input = Array::from_host_buffer(
                &session,
                input_type.clone(),
                mesh.clone(),
                None,
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
                    let lowered = staged.lower(&session).unwrap();
                    let targets = lowered
                        .lowered_program()
                        .stable_hlo()
                        .split("stablehlo.custom_call @")
                        .skip(1)
                        .map(|call| call.split('(').next().unwrap().trim_matches('"').to_string())
                        .collect::<Vec<_>>();
                    let compiled = lowered.compile(&session).unwrap();
                    let output = compiled.executable().execute(&session, Value::Array(input.clone())).unwrap();
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
        // with the values of JAX, as `test_gpu_only_collectives_on_two_cuda_processes` records them.
        let outputs = run_cuda_rank_pair("distributed::tests::test_parallel_broadcast_cuda_process_helper");
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

    // Worker of `test_array_to_placement_on_two_cuda_processes`, which returns immediately unless that test launched it
    // as one of its two ranks. Each placement also checks that the source array keeps its shards.
    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_array_to_placement_cuda_process_helper() {
        let Ok(role) = env::var(MULTI_HOST_TEST_ROLE) else {
            return;
        };
        run_as_cuda_rank(role.parse().unwrap(), |client| {
            probe_cross_host_placements(client, |source, session, target| {
                let source_shards = read_addressable_f32s(&source);
                let placed = source.to_placement(session, target)?;
                assert_eq!(read_addressable_f32s(&source), source_shards);
                Ok(placed)
            })
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_array_to_placement_on_two_cuda_processes() {
        let outputs = run_cuda_rank_pair("distributed::tests::test_array_to_placement_cuda_process_helper");
        let probes = outputs
            .iter()
            .flat_map(|output| output.lines().filter_map(|line| line.split_once("probe: ").map(|(_, probe)| probe)))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            probes,
            indoc! {"
                rank 0 cross_host_transfers_extension: available
                rank 0 sharded_to_reversed_sharded: [[11.0, 12.0]] ready
                rank 0 replicated_to_replicated: [[1.0, 2.0, 3.0, 4.0]] ready
                rank 0 replicated_to_sharded: [[5.0, 6.0]] ready
                rank 0 sharded_to_reversed_replicated: [[13.0, 14.0, 15.0, 16.0]] ready
                rank 0 replicated_to_replicated_again: [[17.0, 18.0, 19.0, 20.0]] ready
                rank 1 cross_host_transfers_extension: available
                rank 1 sharded_to_reversed_sharded: [[9.0, 10.0]] ready
                rank 1 replicated_to_replicated: [[1.0, 2.0, 3.0, 4.0]] ready
                rank 1 replicated_to_sharded: [[7.0, 8.0]] ready
                rank 1 sharded_to_reversed_replicated: [[13.0, 14.0, 15.0, 16.0]] ready
                rank 1 replicated_to_replicated_again: [[17.0, 18.0, 19.0, 20.0]] ready"},
        );
    }

    // Worker of `test_array_into_placement_on_two_cuda_processes`, which returns immediately unless that test launched
    // it as one of its two ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_array_into_placement_cuda_process_helper() {
        let Ok(role) = env::var(MULTI_HOST_TEST_ROLE) else {
            return;
        };
        run_as_cuda_rank(role.parse().unwrap(), |client| {
            probe_cross_host_placements(client, |source, session, target| source.into_placement(session, target))
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_array_into_placement_on_two_cuda_processes() {
        let outputs = run_cuda_rank_pair("distributed::tests::test_array_into_placement_cuda_process_helper");
        let probes = outputs
            .iter()
            .flat_map(|output| output.lines().filter_map(|line| line.split_once("probe: ").map(|(_, probe)| probe)))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            probes,
            indoc! {"
                rank 0 cross_host_transfers_extension: available
                rank 0 sharded_to_reversed_sharded: [[11.0, 12.0]] ready
                rank 0 replicated_to_replicated: [[1.0, 2.0, 3.0, 4.0]] ready
                rank 0 replicated_to_sharded: [[5.0, 6.0]] ready
                rank 0 sharded_to_reversed_replicated: [[13.0, 14.0, 15.0, 16.0]] ready
                rank 0 replicated_to_replicated_again: [[17.0, 18.0, 19.0, 20.0]] ready
                rank 1 cross_host_transfers_extension: available
                rank 1 sharded_to_reversed_sharded: [[9.0, 10.0]] ready
                rank 1 replicated_to_replicated: [[1.0, 2.0, 3.0, 4.0]] ready
                rank 1 replicated_to_sharded: [[7.0, 8.0]] ready
                rank 1 sharded_to_reversed_replicated: [[13.0, 14.0, 15.0, 16.0]] ready
                rank 1 replicated_to_replicated_again: [[17.0, 18.0, 19.0, 20.0]] ready"},
        );
    }

    /// Places on every rank of the job of `client` a host value with [`Session::place`] and an array that each rank
    /// uploaded without devices with [`Array::to_placement`] onto a split over the two global devices, with the same
    /// values on both ranks or, when `differ` is `true`, with values that differ between them, and then places an
    /// array of rank 0 replicated onto both devices. Prints one `probe:` line for each placement with the elements of
    /// the shards of this rank once its result is ready, `different values` when its readiness reports the failed
    /// comparison of the values of the ranks, or the first line of any other error.
    #[cfg(feature = "cuda-13")]
    fn probe_value_comparison(client: &'static Client<'static>, differ: bool) {
        let rank = client.process_index().unwrap();
        let devices = client
            .devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = |devices: Vec<Device>| {
            let axis = MeshAxis::new("x", devices.len(), MeshAxisType::Explicit).unwrap();
            DeviceMesh::new(LogicalMesh::new(vec![axis]).unwrap(), devices).unwrap()
        };
        let (process_0_mesh, forward_mesh) = (mesh(vec![devices[0]]), mesh(devices.clone()));
        let split = Sharding::new(forward_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]);
        let split = Placement::new(forward_mesh.clone(), split.unwrap()).unwrap();
        let replicated = |mesh: &DeviceMesh| Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let report = |name: &str, placed: Result<Array, Error>| {
            let outcome = match placed.and_then(|placed| placed.block_until_ready().map(|()| placed)) {
                Ok(placed) => format!("{:?}", read_addressable_f32s(&placed)),
                Err(Error::Pjrt(ryft_pjrt::Error::FailedPrecondition { message, .. }))
                    if message == DIFFERENT_VALUES_MESSAGE =>
                {
                    "different values".to_string()
                }
                Err(error) => error.to_string().lines().next().unwrap_or("").to_string(),
            };
            println!("probe: rank {rank} {name}: {outcome}");
        };

        let offset = if differ { 10. * rank as f32 } else { 0. };
        let values = values_to_bytes(&[1f32, 2., 3., 4.].map(|value| value + offset));
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let host = CoreArray::new(r#type.clone(), values.clone()).unwrap();
        report("host value", session.place(host, PlacementOptions::to(PlacementRequest::from(split.clone()))));
        let uploaded = Array::from_host_buffer(&session, r#type.clone(), None, None, values).unwrap();
        report("uploaded array", uploaded.to_placement(&session, split));

        // An ordinary cross-host placement afterwards completes only if both ranks are still in step.
        let r#type = r#type.with_sharding(replicated(&process_0_mesh)).unwrap();
        let values = values_to_bytes(&[17f32, 18., 19., 20.]);
        let source = Array::from_host_buffer(&session, r#type, process_0_mesh.clone(), None, values).unwrap();
        let target = Placement::new(forward_mesh.clone(), replicated(&forward_mesh)).unwrap();
        report("cross-host placement", source.to_placement(&session, target));
    }

    /// Runs the `helper` of a two-rank CUDA job (refer to [`run_cuda_rank_pair`]) that calls
    /// [`probe_value_comparison`], and returns the `probe:` lines of both ranks, those of rank 0 first.
    #[cfg(feature = "cuda-13")]
    fn run_value_comparison_rank_pair(helper: &str) -> String {
        let outputs = run_cuda_rank_pair(helper);
        let probes = outputs
            .iter()
            .flat_map(|output| output.lines().filter_map(|line| line.split_once("probe: ").map(|(_, probe)| probe)))
            .collect::<Vec<_>>();
        probes.join("\n")
    }

    // Worker of `test_session_place_compares_equal_values_on_two_cuda_processes`, which returns immediately unless that
    // test launched it as one of its two ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_session_place_compares_equal_values_cuda_process_helper() {
        let Ok(role) = env::var(MULTI_HOST_TEST_ROLE) else {
            return;
        };
        run_as_cuda_rank(role.parse().unwrap(), |client| probe_value_comparison(client, false));
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_session_place_compares_equal_values_on_two_cuda_processes() {
        // Equal values pass the comparison, and each rank holds its shard of the value.
        let helper = "distributed::tests::test_session_place_compares_equal_values_cuda_process_helper";
        let probes = run_value_comparison_rank_pair(helper);
        assert_eq!(
            probes,
            indoc! {"
                rank 0 host value: [[1.0, 2.0]]
                rank 0 uploaded array: [[1.0, 2.0]]
                rank 0 cross-host placement: [[17.0, 18.0, 19.0, 20.0]]
                rank 1 host value: [[3.0, 4.0]]
                rank 1 uploaded array: [[3.0, 4.0]]
                rank 1 cross-host placement: [[17.0, 18.0, 19.0, 20.0]]"},
        );
    }

    // Worker of `test_session_place_rejects_different_values_on_two_cuda_processes`, which returns immediately unless
    // that test launched it as one of its two ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_session_place_rejects_different_values_cuda_process_helper() {
        let Ok(role) = env::var(MULTI_HOST_TEST_ROLE) else {
            return;
        };
        run_as_cuda_rank(role.parse().unwrap(), |client| probe_value_comparison(client, true));
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_session_place_rejects_different_values_on_two_cuda_processes() {
        // Different values fail the results of both ranks alike without a hang, and the ranks stay in step.
        let helper = "distributed::tests::test_session_place_rejects_different_values_cuda_process_helper";
        let probes = run_value_comparison_rank_pair(helper);
        assert_eq!(
            probes,
            indoc! {"
                rank 0 host value: different values
                rank 0 uploaded array: different values
                rank 0 cross-host placement: [[17.0, 18.0, 19.0, 20.0]]
                rank 1 host value: different values
                rank 1 uploaded array: different values
                rank 1 cross-host placement: [[17.0, 18.0, 19.0, 20.0]]"},
        );
    }

    /// Places on every rank of the job of `client` a tree of two `bool[2]` host values with [`Session::place`] onto a
    /// split over the two global devices, whose first leaf on rank 0 holds a byte that does not encode a `bool`, and
    /// then places an array of rank 0 replicated onto both devices. Prints one `probe:` line for the tree, with
    /// `invalid value` when the call returns the error of that leaf and otherwise `different values` for each leaf
    /// whose readiness reports the failed comparison, and one for the cross-host placement, as
    /// [`probe_value_comparison`] does.
    #[cfg(feature = "cuda-13")]
    fn probe_value_comparison_with_invalid_leaf(client: &'static Client<'static>) {
        let rank = client.process_index().unwrap();
        let devices = client
            .devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let session = Session::builder().client(client).build().unwrap();
        let axis = |size: usize| MeshAxis::new("x", size, MeshAxisType::Explicit).unwrap();
        let forward_mesh = DeviceMesh::new(LogicalMesh::new(vec![axis(2)]).unwrap(), devices.clone()).unwrap();
        let process_0_mesh = DeviceMesh::new(LogicalMesh::new(vec![axis(1)]).unwrap(), vec![devices[0]]).unwrap();
        let split = Sharding::new(forward_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]);
        let split = Placement::new(forward_mesh.clone(), split.unwrap()).unwrap();
        let bools = ArrayType::new_static(DataType::Boolean, [2]);
        let mut first = CoreArray::new(bools.clone(), vec![1, 0]).unwrap();
        if rank == 0 {
            first.storage_bytes_mut()[0] = 2;
        }
        let tree = vec![first, CoreArray::new(bools, vec![0, 1]).unwrap()];
        let outcome = match session.place(tree, PlacementOptions::to(PlacementRequest::from(split))) {
            Ok(placed) => placed
                .iter()
                .map(|placed| match placed.block_until_ready() {
                    Ok(()) => "ready".to_string(),
                    Err(Error::Pjrt(ryft_pjrt::Error::FailedPrecondition { message, .. }))
                        if message == DIFFERENT_VALUES_MESSAGE =>
                    {
                        "different values".to_string()
                    }
                    Err(error) => error.to_string().lines().next().unwrap_or("").to_string(),
                })
                .collect::<Vec<_>>()
                .join(", "),
            Err(Error::Array(error))
                if matches!(error.as_ref(), ArrayError::FailedParameterPlacement { source, .. }
                    if source.to_string() == "array element 0 has invalid `bool` byte encoding [2]") =>
            {
                "invalid value".to_string()
            }
            Err(error) => error.to_string().lines().next().unwrap_or("").to_string(),
        };
        println!("probe: rank {rank} tree: {outcome}");

        // An ordinary cross-host placement afterwards completes only if both ranks are still in step.
        let replicated = |mesh: &DeviceMesh| Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let r#type = ArrayType::new_static(DataType::F32, [4]).with_sharding(replicated(&process_0_mesh)).unwrap();
        let values = values_to_bytes(&[17f32, 18., 19., 20.]);
        let source = Array::from_host_buffer(&session, r#type, process_0_mesh.clone(), None, values).unwrap();
        let target = Placement::new(forward_mesh.clone(), replicated(&forward_mesh)).unwrap();
        let placed = source.to_placement(&session, target);
        let placed = placed.and_then(|placed| placed.block_until_ready().map(|()| placed));
        match placed {
            Ok(placed) => println!("probe: rank {rank} cross-host placement: {:?}", read_addressable_f32s(&placed)),
            Err(error) => {
                let error = error.to_string();
                println!("probe: rank {rank} cross-host placement: {}", error.lines().next().unwrap_or(""));
            }
        }
    }

    // Worker of `test_session_place_reports_an_invalid_leaf_on_two_cuda_processes`, which returns immediately unless
    // that test launched it as one of its two ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_session_place_reports_an_invalid_leaf_cuda_process_helper() {
        let Ok(role) = env::var(MULTI_HOST_TEST_ROLE) else {
            return;
        };
        run_as_cuda_rank(role.parse().unwrap(), probe_value_comparison_with_invalid_leaf);
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_session_place_reports_an_invalid_leaf_on_two_cuda_processes() {
        // The rank whose first leaf is invalid returns its error after the one comparison of the call, the other rank
        // sees the mismatch on both of its results, and both stay in step.
        let helper = "distributed::tests::test_session_place_reports_an_invalid_leaf_cuda_process_helper";
        let probes = run_value_comparison_rank_pair(helper);
        assert_eq!(
            probes,
            indoc! {"
                rank 0 tree: invalid value
                rank 0 cross-host placement: [[17.0, 18.0, 19.0, 20.0]]
                rank 1 tree: different values, different values
                rank 1 cross-host placement: [[17.0, 18.0, 19.0, 20.0]]"},
        );
    }

    /// Places on every rank of the job of `client`, with one call of [`Session::place`], a host value split over the
    /// two global devices, which is a scalar on rank 0 (so that only rank 0 rejects it) and a vector on rank 1, and an
    /// array of rank 0 replicated onto both devices, which rank 1 can only receive from rank 0. Then places that array
    /// once more on its own. Prints one `probe:` line for the call, with `different values` when it returns the error
    /// of the failed comparison or otherwise the first line of its error or the elements of the shards of this rank,
    /// and one for the placement that follows it.
    #[cfg(feature = "cuda-13")]
    fn probe_call_admission_with_invalid_leaf(client: &'static Client<'static>) {
        let rank = client.process_index().unwrap();
        let devices = client
            .devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let session = Session::builder().client(client).build().unwrap();
        let axis = |size: usize| MeshAxis::new("x", size, MeshAxisType::Explicit).unwrap();
        let forward_mesh = DeviceMesh::new(LogicalMesh::new(vec![axis(2)]).unwrap(), devices.clone()).unwrap();
        let process_0_mesh = DeviceMesh::new(LogicalMesh::new(vec![axis(1)]).unwrap(), vec![devices[0]]).unwrap();
        let split = Sharding::new(forward_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]);
        let split = PlacementRequest::from(Placement::new(forward_mesh.clone(), split.unwrap()).unwrap());
        let replicated = Sharding::replicated(forward_mesh.logical_mesh().clone(), 1);
        let replicated = PlacementRequest::from(Placement::new(forward_mesh.clone(), replicated).unwrap());
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let values = values_to_bytes(&[17f32, 18., 19., 20.]);
        let source = Array::from_host_buffer(&session, r#type, process_0_mesh, None, values).unwrap();
        let host = match rank {
            0 => CoreArray::scalar(1f32).unwrap(),
            _ => CoreArray::vector(vec![1f32, 2.]).unwrap(),
        };
        let tree = vec![PlaceInput::Host(host), PlaceInput::Runtime(source.clone())];
        let options = PlacementOptions::to(vec![split, replicated.clone()]);
        let placed = session.place(tree, options).and_then(|placed| {
            placed.iter().try_for_each(Array::block_until_ready)?;
            Ok(placed)
        });
        let outcome = match placed {
            Ok(placed) => format!("{:?}", placed.iter().map(read_addressable_f32s).collect::<Vec<_>>()),
            Err(Error::Pjrt(ryft_pjrt::Error::FailedPrecondition { message, .. }))
                if message == DIFFERENT_VALUES_MESSAGE =>
            {
                "different values".to_string()
            }
            Err(error) => error.to_string().lines().next().unwrap_or("").to_string(),
        };
        println!("probe: rank {rank} tree: {outcome}");

        // An ordinary cross-host placement afterwards completes only if both ranks are still in step.
        let placed = source.to_placement(&session, replicated);
        match placed.and_then(|placed| placed.block_until_ready().map(|()| placed)) {
            Ok(placed) => println!("probe: rank {rank} cross-host placement: {:?}", read_addressable_f32s(&placed)),
            Err(error) => {
                let error = error.to_string();
                println!("probe: rank {rank} cross-host placement: {}", error.lines().next().unwrap_or(""));
            }
        }
    }

    // Worker of `test_session_place_admits_a_call_with_an_invalid_leaf_on_two_cuda_processes`, which returns
    // immediately unless that test launched it as one of its two ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_session_place_admits_a_call_with_an_invalid_leaf_cuda_process_helper() {
        let Ok(role) = env::var(MULTI_HOST_TEST_ROLE) else {
            return;
        };
        run_as_cuda_rank(role.parse().unwrap(), probe_call_admission_with_invalid_leaf);
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_session_place_admits_a_call_with_an_invalid_leaf_on_two_cuda_processes() {
        // The first leaf is invalid on rank 0 only, and the second leaf of the same call is a cross-host move that rank
        // 1 could only complete with rank 0. Both ranks return without placing either leaf, rank 0 with its own error
        // and rank 1 with the mismatch, and both stay in step for the placement that follows.
        let helper = "distributed::tests::test_session_place_admits_a_call_with_an_invalid_leaf_cuda_process_helper";
        let probes = run_value_comparison_rank_pair(helper);
        let leaf = "failed to place the parameter at `$[0]`";
        let error = "`place` target sharding rank (1) does not match the input rank (0)";
        assert_eq!(
            probes,
            formatdoc! {"
                rank 0 tree: {leaf}: {error}
                rank 0 cross-host placement: [[17.0, 18.0, 19.0, 20.0]]
                rank 1 tree: different values
                rank 1 cross-host placement: [[17.0, 18.0, 19.0, 20.0]]"},
        );
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_core_placement_validation_cuda_process_helper() {
        let Ok(role) = env::var(MULTI_HOST_TEST_ROLE) else {
            return;
        };
        run_as_cuda_rank(role.parse().unwrap(), |client| {
            let rank = client.process_index().unwrap();
            let session = Session::builder().client(client).build().unwrap();
            let mesh = session.mesh([MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
            let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
            let target = Placement::new(mesh, sharding).unwrap();
            let different_values = |output: Array| {
                assert!(matches!(
                    output.block_until_ready(),
                    Err(Error::Pjrt(PjrtError::FailedPrecondition { message, .. }))
                        if message == DIFFERENT_VALUES_MESSAGE,
                ));
            };

            // Both ranks reach concrete dispatch. The rank with an invalid leaf still takes part in admission.
            let shape = if rank == 0 { vec![] } else { vec![2] };
            let input = Array::from_host_buffer(
                &session,
                ArrayType::new_static(DataType::U8, shape),
                None,
                None,
                vec![1u8; 1 + rank],
            )
            .unwrap();
            assert!(input.is_locally_assigned());
            let before = session.next_value_comparison();
            let result = input.place(&target, None);
            if rank == 0 {
                assert_eq!(
                    result.unwrap_err(),
                    ProgramError::Type(TypeError::invalid(
                        "`place` target sharding rank (1) does not match the input rank (0)",
                    )),
                );
            } else {
                different_values(result.unwrap());
            }
            assert_eq!(session.next_value_comparison(), before + 2);

            let input =
                Array::from_host_buffer(&session, ArrayType::new_static(DataType::U8, [4]), None, None, [2u8; 4])
                    .unwrap();
            let before = session.next_value_comparison();
            let output = input.place(&target, None).unwrap();
            output.block_until_ready().unwrap();
            let shards = output.addressable_shards().map(|shard| shard.copy_to_host_bytes().unwrap());
            assert_eq!(shards.collect::<Vec<_>>(), [vec![2, 2]]);
            assert_eq!(session.next_value_comparison(), before + 2);
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_core_placement_validation_on_two_cuda_processes() {
        run_cuda_rank_pair("distributed::tests::test_core_placement_validation_cuda_process_helper");
    }

    // Worker of `test_array_to_placement_relayouts_cross_host_replica_on_two_cuda_processes`, which returns immediately
    // unless that test launched it as one of its two ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_array_to_placement_relayouts_cross_host_replica_cuda_process_helper() {
        let Ok(role) = env::var(MULTI_HOST_TEST_ROLE) else {
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
            let session = Session::builder().client(client).build().unwrap();
            let logical_mesh =
                |size| LogicalMesh::new(vec![MeshAxis::new("x", size, MeshAxisType::Explicit).unwrap()]).unwrap();
            let source_mesh = DeviceMesh::new(logical_mesh(1), vec![devices[0]]).unwrap();
            let destination_mesh = DeviceMesh::new(logical_mesh(2), devices.clone()).unwrap();
            let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));

            // A column-major replica of `[[1, 2, 3], [4, 5, 6]]` on the device of rank 0 is split by rows over both
            // ranks. Neither whole shards nor shard regions apply, so rank 0 sends its replica to rank 1 for a
            // compiled reshard on the destination mesh, and it must convert the replica to the row-major layout that
            // the receive of rank 1 allocates, or rank 1 would decode its row as `[5, 3, 6]`.
            let r#type = ArrayType::new_static(DataType::F32, [2, 3])
                .with_layout(column_major.clone())
                .with_sharding(Sharding::replicated(source_mesh.logical_mesh().clone(), 2))
                .unwrap();
            let values = values_to_bytes(&[1f32, 2., 3., 4., 5., 6.]);
            let source = Array::from_host_buffer(&session, r#type, source_mesh, None, values).unwrap();
            let rows = Sharding::new(
                destination_mesh.logical_mesh().clone(),
                vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
            )
            .unwrap();
            let placed = source.to_placement(&session, Placement::new(destination_mesh.clone(), rows.clone()).unwrap());
            let placed = placed.unwrap();
            placed.block_until_ready().unwrap();
            assert_eq!(placed.mesh(), &destination_mesh);
            assert_eq!(placed.sharding(), &rows);
            assert_eq!(placed.layout(), Some(&column_major));

            // Each rank prints the slice, the physical layout, and the dense row-major values of its shard, so that the
            // values of both ranks are compared together.
            let shards = placed
                .addressable_shards()
                .map(|shard| {
                    let layout = shard.buffer().unwrap().specification().unwrap().layout;
                    let column_major = layout == column_major.to_pjrt().unwrap();
                    (
                        shard.slice().to_vec(),
                        column_major,
                        values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()),
                    )
                })
                .collect::<Vec<_>>();
            println!("probe: rank {rank} relayouts_cross_host_replica: {shards:?}");
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_array_to_placement_relayouts_cross_host_replica_on_two_cuda_processes() {
        let outputs = run_cuda_rank_pair(
            "distributed::tests::test_array_to_placement_relayouts_cross_host_replica_cuda_process_helper",
        );
        let probes = outputs
            .iter()
            .flat_map(|output| output.lines().filter_map(|line| line.split_once("probe: ").map(|(_, probe)| probe)))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            probes,
            indoc! {"
                rank 0 relayouts_cross_host_replica: [([0..1, 0..3], true, [1.0, 2.0, 3.0])]
                rank 1 relayouts_cross_host_replica: [([1..2, 0..3], true, [4.0, 5.0, 6.0])]"},
        );
    }

    // Worker of `test_array_to_placement_agrees_on_strategy_on_two_cuda_processes`, which returns immediately unless
    // that test launched it as one of its two ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_array_to_placement_agrees_on_strategy_cuda_process_helper() {
        // Compilation artifact exchange that requires sharing every artifact with the other rank. A rank leads the
        // compilation of a program over its own devices, so it publishes the conversion of its own shard and never
        // waits for an artifact of the other rank. A placement that switched to a program over both ranks would wait.
        struct SharedCompilationExchange {
            process_index: usize,
        }

        impl CompilationArtifactExchange for SharedCompilationExchange {
            fn process_index(&self) -> usize {
                self.process_index
            }

            fn process_count(&self) -> usize {
                2
            }

            fn publish(&self, _: &[u8], _: &[u8]) -> Result<(), CompilationArtifactExchangeError> {
                Ok(())
            }

            fn receive(&self, _: &[u8], _: Duration) -> Result<Option<Vec<u8>>, CompilationArtifactExchangeError> {
                panic!("a placement of whole shards must not wait for a compiled program");
            }
        }

        let Ok(role) = env::var(MULTI_HOST_TEST_ROLE) else {
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
            let cache = CompilationCache::new().with_artifact_exchange(
                Arc::new(SharedCompilationExchange { process_index: rank }),
                CompilationArtifactExchangePolicy::RequireSharing { timeout: Duration::from_secs(60) },
            );
            let session = Session::builder().client(client).compilation_cache(cache).build().unwrap();
            let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
            let replicated = Sharding::replicated(logical_mesh.clone(), 2);
            let source_mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[0]]).unwrap();
            let destination_mesh = DeviceMesh::new(logical_mesh, vec![devices[1]]).unwrap();

            // Rank 0 holds `[[1, 2, 3], [4, 5, 6]]` in a column-major buffer, which the type does not declare, so only
            // rank 0 knows that its buffer is not in the row-major layout that the receive of rank 1 allocates. Rank 1
            // only receives. Both ranks choose whole shards from the shared metadata, and rank 0 converts its buffer
            // with a program over its own device, which it leads and publishes, rather than switching to a compiled
            // reshard that rank 1 would never join.
            let buffers = if rank == 0 {
                let device = client.addressable_devices().unwrap().remove(0);
                let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![])).to_pjrt().unwrap();
                let values = values_to_bytes(&[1f32, 2., 3., 4., 5., 6.]);
                let memory = device.default_memory().unwrap();
                let buffer =
                    client.buffer(values.as_slice(), BufferType::F32, [2u64, 3], None, memory, Some(column_major));
                let buffer = buffer.unwrap();
                assert_ne!(
                    buffer.specification().unwrap().layout,
                    Layout::Tiled(TiledLayout::new(vec![1, 0], vec![])).to_pjrt().unwrap(),
                );
                vec![buffer]
            } else {
                Vec::new()
            };
            let r#type = ArrayType::new_static(DataType::F32, [2, 3]).with_sharding(replicated.clone()).unwrap();
            let source = Array::from_addressable_buffers(&session, r#type, source_mesh, None, buffers).unwrap();
            let target = Placement::new(destination_mesh.clone(), replicated).unwrap();
            let placed = source.to_placement(&session, target).unwrap();
            placed.block_until_ready().unwrap();
            assert_eq!(placed.mesh(), &destination_mesh);
            assert_eq!(session.cached_program_count(), if rank == 0 { 1 } else { 0 });
            let shards =
                placed.addressable_shards().map(|shard| shard.copy_to_host_bytes().unwrap()).collect::<Vec<_>>();
            if rank == 0 {
                assert_eq!(shards.len(), 0);
            } else {
                assert_eq!(
                    shards.iter().map(|bytes| values_from_bytes::<f32>(bytes)).collect::<Vec<_>>(),
                    [vec![1., 2., 3., 4., 5., 6.]]
                );
            }
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_array_to_placement_agrees_on_strategy_on_two_cuda_processes() {
        run_cuda_rank_pair("distributed::tests::test_array_to_placement_agrees_on_strategy_cuda_process_helper");
    }

    // Worker of `test_array_to_placement_swaps_shards_into_memory_on_two_cuda_processes`, which returns immediately
    // unless that test launched it as one of its two ranks. It swaps the halves of `[1, 2, 3, 4]` between the ranks
    // into each memory and prints one `probe:` line per memory with the result.
    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_array_to_placement_swaps_shards_into_memory_cuda_process_helper() {
        let Ok(role) = env::var(MULTI_HOST_TEST_ROLE) else {
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
            let session = Session::builder().client(client).build().unwrap();
            let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
            let sharded = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
            let forward_mesh = DeviceMesh::new(logical_mesh.clone(), devices.clone()).unwrap();
            let reversed_mesh = DeviceMesh::new(logical_mesh, vec![devices[1], devices[0]]).unwrap();
            let r#type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap();
            let values = values_to_bytes(&[1f32, 2., 3., 4.]);
            let source = Array::from_host_buffer(&session, r#type, forward_mesh, None, values).unwrap();

            // Each rank sends its half to the other one and receives the other half into default device memory, which
            // it then copies to the requested memory. Each rank looks up that memory on its own device only after it
            // issued both of its transfers, so a memory that its device lacks fails that rank after the swap is paired,
            // and every later swap still pairs its transfers. A swap that left a transfer unmatched would make the next
            // swap between the same devices read stale halves. Both memories exist on CUDA devices, so this does not
            // exercise such a failure, and a rank that failed so would not reach the transfers of a later placement of
            // a tree.
            for memory in [Memory::Host { pinned: true }, Memory::Host { pinned: false }, Memory::Device] {
                let target = Placement::new(reversed_mesh.clone(), sharded.clone()).unwrap().with_memory(memory);
                match source.to_placement(&session, target) {
                    Ok(placed) => {
                        let ready = match placed.block_until_ready() {
                            Ok(()) => "ready".to_string(),
                            Err(error) => error.to_string().lines().next().unwrap_or("").to_string(),
                        };
                        let shards = placed
                            .addressable_shards()
                            .map(|shard| {
                                let kind = shard.buffer().unwrap().memory().unwrap().kind().unwrap().into_owned();
                                (kind, values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()))
                            })
                            .collect::<Vec<_>>();
                        println!("probe: rank {rank} {memory:?}: {shards:?} {ready}");
                    }
                    Err(error) => {
                        println!("probe: rank {rank} {memory:?}: {}", error.to_string().lines().next().unwrap_or(""));
                    }
                }
            }
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_array_to_placement_swaps_shards_into_memory_on_two_cuda_processes() {
        let outputs = run_cuda_rank_pair(
            "distributed::tests::test_array_to_placement_swaps_shards_into_memory_cuda_process_helper",
        );
        let probes = outputs
            .iter()
            .flat_map(|output| output.lines().filter_map(|line| line.split_once("probe: ").map(|(_, probe)| probe)))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            probes,
            indoc! {r#"
                rank 0 Host { pinned: true }: [("pinned_host", [3.0, 4.0])] ready
                rank 0 Host { pinned: false }: [("unpinned_host", [3.0, 4.0])] ready
                rank 0 Device: [("device", [3.0, 4.0])] ready
                rank 1 Host { pinned: true }: [("pinned_host", [1.0, 2.0])] ready
                rank 1 Host { pinned: false }: [("unpinned_host", [1.0, 2.0])] ready
                rank 1 Device: [("device", [1.0, 2.0])] ready"#},
        );
    }

    // Worker of `test_array_to_placement_sends_from_host_memory_on_two_cuda_processes`, which returns immediately
    // unless that test launched it as one of its two ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_array_to_placement_sends_from_host_memory_cuda_process_helper() {
        let Ok(role) = env::var(MULTI_HOST_TEST_ROLE) else {
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
            let session = Session::builder().client(client).build().unwrap();
            let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
            let replicated = Sharding::replicated(logical_mesh.clone(), 1);
            let source_mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[0]]).unwrap();
            let destination_mesh = DeviceMesh::new(logical_mesh, vec![devices[1]]).unwrap();

            // Rank 0 only sends `[1, 2, 3, 4]` from host memory and rank 1 only receives it, into the same memory of
            // its own device, because the destination keeps the memory of the source. The CUDA cross-host transfer
            // reads a device buffer, so rank 0 stages its host buffer in device memory before it sends it.
            for memory in [Memory::Host { pinned: true }, Memory::Host { pinned: false }] {
                let r#type = ArrayType::new_static(DataType::F32, [4])
                    .with_memory(memory)
                    .with_sharding(replicated.clone())
                    .unwrap();
                let values = values_to_bytes(&[1f32, 2., 3., 4.]);
                let source = Array::from_host_buffer(&session, r#type, source_mesh.clone(), None, values).unwrap();
                let target = Placement::new(destination_mesh.clone(), replicated.clone()).unwrap();
                probe_placement(rank, &format!("{memory:?}"), &session, &source, target);
            }
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_array_to_placement_sends_from_host_memory_on_two_cuda_processes() {
        let outputs = run_cuda_rank_pair(
            "distributed::tests::test_array_to_placement_sends_from_host_memory_cuda_process_helper",
        );
        let probes = outputs
            .iter()
            .flat_map(|output| output.lines().filter_map(|line| line.split_once("probe: ").map(|(_, probe)| probe)))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            probes,
            indoc! {r#"
                rank 0 Host { pinned: true }: [] ready
                rank 0 Host { pinned: false }: [] ready
                rank 1 Host { pinned: true }: [("pinned_host", [1.0, 2.0, 3.0, 4.0])] ready
                rank 1 Host { pinned: false }: [("unpinned_host", [1.0, 2.0, 3.0, 4.0])] ready"#},
        );
    }

    // Worker of `test_array_to_placement_sends_padded_bounded_dynamic_shard_on_two_cuda_processes`, which returns
    // immediately unless that test launched it as one of its two ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_array_to_placement_sends_padded_bounded_dynamic_shard_cuda_process_helper() {
        let Ok(role) = env::var(MULTI_HOST_TEST_ROLE) else {
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
            let session = Session::builder().client(client).build().unwrap();
            let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
            let replicated = Sharding::replicated(logical_mesh.clone(), 1);
            let source_mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[0]]).unwrap();
            let destination_mesh = DeviceMesh::new(logical_mesh, vec![devices[1]]).unwrap();

            // Rank 0 runs a bounded-dynamic identity whose result buffer is allocated for the bound of its dimension
            // with a trailer that records its extent, and rank 1 holds only the metadata of that result. The CUDA
            // plugin sends as many bytes as the allocation of the sender holds, while the receiver allocates a static
            // buffer of the shard, so rank 0 must send a static copy of exactly that size. This holds for an extent
            // below the bound, where the physical dimensions of the dynamic buffer differ from the shape of the shard,
            // and at the maximum extent, where they are equal but the allocation still holds the trailer. Each step
            // prints the number of received elements and whether they are `0, 1, 2, ...`, and a failure names the rank
            // and step.
            let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(1024)).unwrap());
            let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]));
            let identity: crate::CompiledFunction<ArrayType, ArrayType> =
                crate::compile_with_options(|input| input, dynamic_type, &session, Options::new(source_mesh.clone()))
                    .unwrap_or_else(|error| {
                        panic!("rank {rank} failed to compile the bounded-dynamic identity: {error}")
                    });
            for size in [2usize, 1023] {
                let step = format!("extent {size}");
                let r#type = ArrayType::new_static(DataType::F32, [size]);
                let values = (0..size).map(|value| value as f32).collect::<Vec<_>>();
                let source = if rank == 0 {
                    let input = Array::from_host_buffer(
                        &session,
                        r#type.clone(),
                        source_mesh.clone(),
                        Some(replicated.clone()),
                        values_to_bytes(&values),
                    )
                    .unwrap_or_else(|error| panic!("rank {rank} failed to upload the input of the {step}: {error}"));
                    let padded = identity.executable().execute(&session, input).unwrap_or_else(|error| {
                        panic!("rank {rank} failed to run the identity for the {step}: {error}")
                    });
                    let buffer = padded.shards()[0].buffer().unwrap();
                    println!(
                        "probe: rank {rank} {step} source: unpadded={:?} dynamic={:?} larger={}",
                        buffer.unpadded_dimensions().unwrap(),
                        buffer.dynamic_dimensions().unwrap(),
                        buffer.on_device_size_in_bytes().unwrap() > 4 * size,
                    );
                    padded
                } else {
                    let sharding = Some(replicated.clone());
                    Array::from_addressable_buffers(&session, r#type, source_mesh.clone(), sharding, Vec::new())
                        .unwrap_or_else(|error| {
                            panic!("rank {rank} failed to import the source of the {step}: {error}")
                        })
                };
                let target = Placement::new(destination_mesh.clone(), replicated.clone()).unwrap();
                let placed = source
                    .to_placement(&session, target)
                    .and_then(|placed| placed.block_until_ready().map(|()| placed))
                    .unwrap_or_else(|error| {
                        panic!("rank {rank} failed to send the dynamic shard of the {step}: {error}")
                    });
                let shards = placed
                    .addressable_shards()
                    .map(|shard| {
                        let buffer = shard.buffer().unwrap();
                        let bytes = shard.copy_to_host_bytes().unwrap_or_else(|error| {
                            panic!("rank {rank} failed to read the received shard of the {step}: {error}")
                        });
                        let received = values_from_bytes::<f32>(&bytes);
                        (
                            buffer.dimensions().unwrap().to_vec(),
                            buffer.on_device_size_in_bytes().unwrap(),
                            received.len(),
                            received == values,
                        )
                    })
                    .collect::<Vec<_>>();
                println!("probe: rank {rank} {step} placed: {shards:?}");
                session
                    .synchronize()
                    .unwrap_or_else(|error| panic!("rank {rank} failed to synchronize after the {step}: {error}"));
            }
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_array_to_placement_sends_padded_bounded_dynamic_shard_on_two_cuda_processes() {
        let outputs = run_cuda_rank_pair(
            "distributed::tests::test_array_to_placement_sends_padded_bounded_dynamic_shard_cuda_process_helper",
        );
        let probes = outputs
            .iter()
            .flat_map(|output| output.lines().filter_map(|line| line.split_once("probe: ").map(|(_, probe)| probe)))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            probes,
            indoc! {r#"
                rank 0 extent 2 source: unpadded=[2] dynamic=[0] larger=true
                rank 0 extent 2 placed: []
                rank 0 extent 1023 source: unpadded=[1023] dynamic=[0] larger=true
                rank 0 extent 1023 placed: []
                rank 1 extent 2 placed: [([2], 8, 2, true)]
                rank 1 extent 1023 placed: [([1023], 4092, 1023, true)]"#},
        );
    }

    // Worker of `test_array_to_placement_receives_into_memory_on_two_cuda_processes`, which returns immediately unless
    // that test launched it as one of its two ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_array_to_placement_receives_into_memory_cuda_process_helper() {
        let Ok(role) = env::var(MULTI_HOST_TEST_ROLE) else {
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
            let session = Session::builder().client(client).build().unwrap();
            let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
            let replicated = Sharding::replicated(logical_mesh.clone(), 1);
            let source_mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[0]]).unwrap();
            let destination_mesh = DeviceMesh::new(logical_mesh, vec![devices[1]]).unwrap();
            let r#type = ArrayType::new_static(DataType::F32, [4]).with_sharding(replicated.clone()).unwrap();
            let values = values_to_bytes(&[1f32, 2., 3., 4.]);
            let source = Array::from_host_buffer(&session, r#type, source_mesh, None, values).unwrap();

            // Rank 0 only sends and rank 1 only receives, into default device memory, and copies the received buffer to
            // the requested memory only after it issued its receive. A memory that only the receiver checks therefore
            // never leaves the sender waiting, and both ranks finish every placement. A placement that left a transfer
            // unmatched would make the next one between the same devices read stale values.
            for memory in [Memory::Host { pinned: true }, Memory::Host { pinned: false }, Memory::Device] {
                let target = Placement::new(destination_mesh.clone(), replicated.clone()).unwrap().with_memory(memory);
                probe_placement(rank, &format!("{memory:?}"), &session, &source, target);
            }
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_array_to_placement_receives_into_memory_on_two_cuda_processes() {
        let outputs =
            run_cuda_rank_pair("distributed::tests::test_array_to_placement_receives_into_memory_cuda_process_helper");
        let probes = outputs
            .iter()
            .flat_map(|output| output.lines().filter_map(|line| line.split_once("probe: ").map(|(_, probe)| probe)))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            probes,
            indoc! {r#"
                rank 0 Host { pinned: true }: [] ready
                rank 0 Host { pinned: false }: [] ready
                rank 0 Device: [] ready
                rank 1 Host { pinned: true }: [("pinned_host", [1.0, 2.0, 3.0, 4.0])] ready
                rank 1 Host { pinned: false }: [("unpinned_host", [1.0, 2.0, 3.0, 4.0])] ready
                rank 1 Device: [("device", [1.0, 2.0, 3.0, 4.0])] ready"#},
        );
    }

    // Worker of `test_array_to_placement_places_uncommitted_values_on_two_cuda_processes`, which returns immediately
    // unless that test launched it as one of its two ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_array_to_placement_places_uncommitted_values_cuda_process_helper() {
        let Ok(role) = env::var(MULTI_HOST_TEST_ROLE) else {
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
            let session = Session::builder().client(client).build().unwrap();
            let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
            let mesh = DeviceMesh::new(logical_mesh.clone(), devices.clone()).unwrap();
            let replicated = Sharding::replicated(logical_mesh.clone(), 1);
            let sharded = Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();

            // Each rank uploads `[1, 2, 3, 4]` without a destination, so its array lives uncommitted on its own device,
            // which the other rank does not know. Each rank places its own value on its own devices of the global mesh,
            // and neither plans a transfer. The last placement copies `[5, 6, 7, 8]` from rank 0 to rank 1, which would
            // receive the bytes of an earlier send that no receive matched.
            let r#type = ArrayType::new_static(DataType::F32, [4]);
            let values = values_to_bytes(&[1f32, 2., 3., 4.]);
            let source = Array::from_host_buffer(&session, r#type, None, None, values).unwrap();
            assert!(!source.is_committed());
            assert_eq!(source.mesh().devices(), [devices[rank]]);
            for (name, sharding) in [("replicated", &replicated), ("sharded", &sharded)] {
                let target = Placement::new(mesh.clone(), sharding.clone()).unwrap();
                probe_placement(rank, name, &session, &source, target);
            }
            let source_mesh = DeviceMesh::new(
                LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
                vec![devices[0]],
            )
            .unwrap();
            let r#type = ArrayType::new_static(DataType::F32, [4]);
            let values = values_to_bytes(&[5f32, 6., 7., 8.]);
            let committed = Array::from_host_buffer(&session, r#type, source_mesh, None, values).unwrap();
            let target = Placement::new(mesh, replicated).unwrap();
            probe_placement(rank, "committed", &session, &committed, target);
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_array_to_placement_places_uncommitted_values_on_two_cuda_processes() {
        let outputs = run_cuda_rank_pair(
            "distributed::tests::test_array_to_placement_places_uncommitted_values_cuda_process_helper",
        );
        let probes = outputs
            .iter()
            .flat_map(|output| output.lines().filter_map(|line| line.split_once("probe: ").map(|(_, probe)| probe)))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            probes,
            indoc! {r#"
                rank 0 replicated: [("device", [1.0, 2.0, 3.0, 4.0])] ready
                rank 0 sharded: [("device", [1.0, 2.0])] ready
                rank 0 committed: [("device", [5.0, 6.0, 7.0, 8.0])] ready
                rank 1 replicated: [("device", [1.0, 2.0, 3.0, 4.0])] ready
                rank 1 sharded: [("device", [3.0, 4.0])] ready
                rank 1 committed: [("device", [5.0, 6.0, 7.0, 8.0])] ready"#},
        );
    }

    // Worker of `test_array_to_placement_places_values_in_memory_on_two_cuda_processes`, which returns immediately
    // unless that test launched it as one of its two ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_array_to_placement_places_values_in_memory_cuda_process_helper() {
        let Ok(role) = env::var(MULTI_HOST_TEST_ROLE) else {
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
            let session = Session::builder().client(client).build().unwrap();
            let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
            let mesh = DeviceMesh::new(logical_mesh.clone(), devices.clone()).unwrap();
            let replicated = Sharding::replicated(logical_mesh.clone(), 1);
            let sharded = Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();

            // Each rank uploads `[1, 2, 3, 4]` without a destination and then places it into device memory, which
            // commits it to its own device. That device is still one that each rank chose on its own, so each rank
            // places its own value on its own devices of the global mesh, and neither plans a transfer that the other
            // does not match.
            let r#type = ArrayType::new_static(DataType::F32, [4]);
            let values = values_to_bytes(&[1f32, 2., 3., 4.]);
            let uploaded = Array::from_host_buffer(&session, r#type, None, None, values).unwrap();
            let source = uploaded.to_placement(&session, Memory::Device).unwrap();
            assert!(source.is_committed());
            assert_eq!(source.mesh().devices(), [devices[rank]]);
            for (name, sharding) in [("replicated", &replicated), ("sharded", &sharded)] {
                let target = Placement::new(mesh.clone(), sharding.clone()).unwrap();
                probe_placement(rank, name, &session, &source, target);
            }
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_array_to_placement_places_values_in_memory_on_two_cuda_processes() {
        let outputs = run_cuda_rank_pair(
            "distributed::tests::test_array_to_placement_places_values_in_memory_cuda_process_helper",
        );
        let probes = outputs
            .iter()
            .flat_map(|output| output.lines().filter_map(|line| line.split_once("probe: ").map(|(_, probe)| probe)))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            probes,
            indoc! {r#"
                rank 0 replicated: [("device", [1.0, 2.0, 3.0, 4.0])] ready
                rank 0 sharded: [("device", [1.0, 2.0])] ready
                rank 1 replicated: [("device", [1.0, 2.0, 3.0, 4.0])] ready
                rank 1 sharded: [("device", [3.0, 4.0])] ready"#},
        );
    }

    // Worker of `test_array_to_placement_agrees_on_mixed_layouts_on_two_cuda_processes`, which returns immediately
    // unless that test launched it as one of its two ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_array_to_placement_agrees_on_mixed_layouts_cuda_process_helper() {
        let Ok(role) = env::var(MULTI_HOST_TEST_ROLE) else {
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
            let session = Session::builder().client(client).build().unwrap();
            let logical_mesh = |size| LogicalMesh::new(vec![MeshAxis::new("x", size, MeshAxisType::Explicit).unwrap()]);
            let source_mesh = DeviceMesh::new(logical_mesh(1).unwrap(), vec![devices[0]]).unwrap();
            let mesh = DeviceMesh::new(logical_mesh(2).unwrap(), devices.clone()).unwrap();
            let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 2);
            let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
            let r#type = ArrayType::new_static(DataType::F32, [2, 3])
                .with_layout(column_major.clone())
                .with_sharding(Sharding::replicated(source_mesh.logical_mesh().clone(), 2))
                .unwrap();
            let values = values_to_bytes(&[1f32, 2., 3., 4., 5., 6.]);
            let source = Array::from_host_buffer(&session, r#type, source_mesh, None, values).unwrap();

            // Resharding the source onto both devices (i.e., `reshard_on_devices`) copies the column-major replica of
            // rank 0 to its own device and sends it to rank 1, which receives it in the default row-major layout. The
            // result has the declared layout of its input on both ranks, as a placement that changes only the devices
            // promises, so rank 1 converts the buffer that it received, and both ranks hold column-major buffers and
            // prepare the result for the compiled function below in the same way.
            let moved = source.clone().reshard_on_devices(&session, &mesh, &replicated, false).unwrap();
            moved.block_until_ready().unwrap();
            let shards = moved
                .addressable_shards()
                .map(|shard| {
                    let layout = shard.buffer().unwrap().specification().unwrap().layout;
                    (
                        layout == column_major.to_pjrt().unwrap(),
                        values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()),
                    )
                })
                .collect::<Vec<_>>();
            println!("probe: rank {rank} moved: {:?} {shards:?}", moved.layout());

            // A compiled function on both devices takes the moved array, whose buffers differ in layout between the
            // ranks, and both ranks prepare it in the same way. It rejects the source on both ranks before either moves
            // anything, because the source is committed to the device of rank 0 alone and a fixed device assignment
            // constrains committed public arguments.
            let input_type = ArrayType::new_static(DataType::F32, [2, 3]).with_sharding(replicated).unwrap();
            let compiled =
                crate::compile::<_, ArrayType, ArrayType>(|x| x.clone() + x, input_type, &session, mesh).unwrap();
            let rejected = compiled.executable().execute(&session, source).err().map(|error| error.to_string());
            let expected = Error::Execution(ExecutionError::CommittedInputDeviceMismatch {
                index: 0,
                actual: vec![devices[0].id()],
                expected: devices.iter().map(Device::id).collect(),
            });
            assert_eq!(rejected, Some(expected.to_string()));
            println!("probe: rank {rank} source: {}", rejected.unwrap());
            let output = compiled.executable().execute(&session, moved).unwrap();
            output.block_until_ready().unwrap();
            println!("probe: rank {rank} doubled moved: {:?}", read_addressable_f32s(&output));
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_array_to_placement_agrees_on_mixed_layouts_on_two_cuda_processes() {
        let outputs = run_cuda_rank_pair(
            "distributed::tests::test_array_to_placement_agrees_on_mixed_layouts_cuda_process_helper",
        );
        let probes = outputs
            .iter()
            .flat_map(|output| output.lines().filter_map(|line| line.split_once("probe: ").map(|(_, probe)| probe)))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            probes,
            indoc! {"
                rank 0 moved: Some(Tiled(TiledLayout { minor_to_major: [0, 1], tiles: [] })) [(true, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]
                rank 0 source: public argument #0 is committed to devices [0], but the executable requires [0, 1]
                rank 0 doubled moved: [[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]]
                rank 1 moved: Some(Tiled(TiledLayout { minor_to_major: [0, 1], tiles: [] })) [(true, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]
                rank 1 source: public argument #0 is committed to devices [0], but the executable requires [0, 1]
                rank 1 doubled moved: [[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]]"},
        );
    }

    // Worker of `test_array_to_placement_compiles_on_mesh_of_one_rank_on_two_cuda_processes`, which returns immediately
    // unless that test launched it as one of its two ranks. It places `[[1, 2], [3, 4]]`, which is split by rows over
    // the devices of both ranks in a column-major layout, replicated onto the device of each rank in turn, and prints
    // two `probe:` lines for each destination, with the metadata of the result and with its shards in row-major order.
    // A failed step panics with the rank and the step, which the report of the pair shows for both ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_array_to_placement_compiles_on_mesh_of_one_rank_cuda_process_helper() {
        let Ok(role) = env::var(MULTI_HOST_TEST_ROLE) else {
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
            let session = Session::builder().client(client).build().unwrap();
            let logical_mesh =
                |size| LogicalMesh::new(vec![MeshAxis::new("x", size, MeshAxisType::Explicit).unwrap()]).unwrap();
            let source_mesh = DeviceMesh::new(logical_mesh(2), devices.clone()).unwrap();
            let rows = Sharding::new(
                source_mesh.logical_mesh().clone(),
                vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
            )
            .unwrap();
            let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
            let r#type =
                ArrayType::new_static(DataType::F32, [2, 2]).with_layout(column_major).with_sharding(rows).unwrap();
            let values = values_to_bytes(&[1f32, 2., 3., 4.]);
            let source = Array::from_host_buffer(&session, r#type, source_mesh, None, values).unwrap();

            // The declared layout rules out shard regions, so each placement gathers the source on the devices of both
            // ranks, copies the gathered value to the destination device on the rank that owns it, and reshards it
            // there. Only that rank compiles and runs the final reshard, and the other rank returns the same metadata
            // without buffers. A rank that compiled for the device of its peer would fail, and a rank that skipped the
            // gather would leave its peer waiting in it.
            for owner in [0, 1] {
                let step = format!("onto rank {owner}");
                let mesh = DeviceMesh::new(logical_mesh(1), vec![devices[owner]]).unwrap();
                let target = Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 2));
                let placed = target
                    .map_err(Error::from)
                    .and_then(|target| source.to_placement(&session, target))
                    .and_then(|placed| placed.block_until_ready().map(|()| placed))
                    .unwrap_or_else(|error| panic!("rank {rank} failed to place the source {step}: {error}"));
                let shards = placed
                    .addressable_shards()
                    .map(|shard| values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()))
                    .collect::<Vec<_>>();
                println!(
                    "probe: rank {rank} {step} metadata: {} {:?} {:?} committed={}",
                    placed.r#type(),
                    placed.sharding(),
                    placed.mesh().devices(),
                    placed.is_committed(),
                );
                println!("probe: rank {rank} {step} shards: {shards:?}");
            }
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_array_to_placement_compiles_on_mesh_of_one_rank_on_two_cuda_processes() {
        let outputs = run_cuda_rank_pair(
            "distributed::tests::test_array_to_placement_compiles_on_mesh_of_one_rank_cuda_process_helper",
        );
        let probes = outputs
            .iter()
            .map(|output| {
                output
                    .lines()
                    .filter_map(|line| line.split_once("probe: ").map(|(_, probe)| probe))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // Both ranks return the same type, placement, devices, and commitment for each step, whichever of them owns
        // the destination.
        for step in ["onto rank 0", "onto rank 1"] {
            let metadata = probes
                .iter()
                .enumerate()
                .map(|(rank, probes)| {
                    let prefix = format!("rank {rank} {step} metadata: ");
                    let line = probes.iter().find_map(|probe| probe.strip_prefix(prefix.as_str()));
                    line.unwrap_or_else(|| panic!("rank {rank} printed no metadata for the step `{step}`"))
                })
                .collect::<Vec<_>>();
            assert_eq!(metadata[0], metadata[1], "the ranks disagree on the metadata of the step `{step}`");
        }
        let shards = probes.concat().into_iter().filter(|probe| probe.contains(" shards: ")).collect::<Vec<_>>();
        assert_eq!(
            shards.join("\n"),
            indoc! {"
                rank 0 onto rank 0 shards: [[1.0, 2.0, 3.0, 4.0]]
                rank 0 onto rank 1 shards: []
                rank 1 onto rank 0 shards: []
                rank 1 onto rank 1 shards: [[1.0, 2.0, 3.0, 4.0]]"},
        );
    }

    // Returns the steps of `test_array_to_placement_mixes_process_local_strategies_cuda_process_helper`.
    #[cfg(feature = "cuda-13")]
    fn mixed_strategy_steps() -> Vec<String> {
        let mut steps = Vec::new();
        for owner in [0, 1] {
            steps.push(format!("onto rank {owner}"));
            steps.push(format!("onto rank {owner} in pinned host memory"));
        }
        steps.extend(["replicated over both ranks", "split over both ranks", "committed"].map(String::from));
        steps
    }

    // Worker of `test_array_to_placement_mixes_process_local_strategies_on_two_cuda_processes`, which
    // returns immediately unless that test launched it as one of its two ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_array_to_placement_mixes_process_local_strategies_cuda_process_helper() {
        let Ok(role) = env::var(MULTI_HOST_TEST_ROLE) else {
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
            let session = Session::builder().client(client).build().unwrap();
            let logical_mesh =
                |size| LogicalMesh::new(vec![MeshAxis::new("x", size, MeshAxisType::Explicit).unwrap()]).unwrap();
            let steps = mixed_strategy_steps();
            let mut steps = steps.iter();

            // Each rank uploads `[1, 2, 3, 4]` without a destination, so its array lives on its own device, which the
            // other rank does not know. A destination on the device of one rank is placed by that rank with the
            // strategy of a shared source, which moves nothing between the ranks, and by the other rank with local
            // work only, which places nothing on its devices, and both ranks must still record the same metadata.
            let r#type = ArrayType::new_static(DataType::F32, [4]);
            let values = values_to_bytes(&[1f32, 2., 3., 4.]);
            let source = Array::from_host_buffer(&session, r#type, None, None, values).unwrap();
            assert!(
                source.is_locally_assigned(),
                "rank {rank} uploaded a value whose devices were not locally assigned"
            );
            assert_eq!(source.mesh().devices(), [devices[rank]]);
            for owner in [0, 1] {
                let mesh = DeviceMesh::new(logical_mesh(1), vec![devices[owner]]).unwrap();
                let target = Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 1));
                let target = target.unwrap();
                probe_placement_step(rank, steps.next().unwrap(), &session, &source, target.clone());
                let target = target.with_memory(Memory::Host { pinned: true });
                probe_placement_step(rank, steps.next().unwrap(), &session, &source, target);
            }

            // A destination over the devices of both ranks is placed by each rank with local work only.
            let mesh = DeviceMesh::new(logical_mesh(2), devices.clone()).unwrap();
            let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
            let split = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
            for sharding in [replicated.clone(), split] {
                let target = Placement::new(mesh.clone(), sharding).unwrap();
                probe_placement_step(rank, steps.next().unwrap(), &session, &source, target);
            }

            // A committed value of rank 0 then moves to rank 1, which would receive the bytes of an earlier send that
            // no receive matched.
            let source_mesh = DeviceMesh::new(logical_mesh(1), vec![devices[0]]).unwrap();
            let r#type = ArrayType::new_static(DataType::F32, [4]);
            let values = values_to_bytes(&[5f32, 6., 7., 8.]);
            let committed = Array::from_host_buffer(&session, r#type, source_mesh, None, values).unwrap();
            let target = Placement::new(mesh, replicated).unwrap();
            probe_placement_step(rank, steps.next().unwrap(), &session, &committed, target);
            assert!(steps.next().is_none());
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_array_to_placement_mixes_process_local_strategies_on_two_cuda_processes() {
        check_cuda_rank_pair_placement_steps(
            "distributed::tests::test_array_to_placement_mixes_process_local_strategies_cuda_process_helper",
            &mixed_strategy_steps(),
            indoc! {r#"
                rank 0 onto rank 0 shards: [([0..4], "device", [1.0, 2.0, 3.0, 4.0])]
                rank 0 onto rank 0 in pinned host memory shards: [([0..4], "pinned_host", [1.0, 2.0, 3.0, 4.0])]
                rank 0 onto rank 1 shards: []
                rank 0 onto rank 1 in pinned host memory shards: []
                rank 0 replicated over both ranks shards: [([0..4], "device", [1.0, 2.0, 3.0, 4.0])]
                rank 0 split over both ranks shards: [([0..2], "device", [1.0, 2.0])]
                rank 0 committed shards: [([0..4], "device", [5.0, 6.0, 7.0, 8.0])]
                rank 1 onto rank 0 shards: []
                rank 1 onto rank 0 in pinned host memory shards: []
                rank 1 onto rank 1 shards: [([0..4], "device", [1.0, 2.0, 3.0, 4.0])]
                rank 1 onto rank 1 in pinned host memory shards: [([0..4], "pinned_host", [1.0, 2.0, 3.0, 4.0])]
                rank 1 replicated over both ranks shards: [([0..4], "device", [1.0, 2.0, 3.0, 4.0])]
                rank 1 split over both ranks shards: [([2..4], "device", [3.0, 4.0])]
                rank 1 committed shards: [([0..4], "device", [5.0, 6.0, 7.0, 8.0])]"#},
        );
    }

    // Returns the steps of `test_array_to_placement_reshards_across_meshes_cuda_process_helper`.
    #[cfg(feature = "cuda-13")]
    fn across_mesh_steps() -> Vec<String> {
        [0, 1]
            .into_iter()
            .flat_map(|owner| ["forward", "reversed"].map(|mesh| format!("from rank {owner} onto the {mesh} mesh")))
            .collect()
    }

    // Worker of `test_array_to_placement_reshards_across_meshes_on_two_cuda_processes`, which returns immediately
    // unless that test launched it as one of its two ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_array_to_placement_reshards_across_meshes_cuda_process_helper() {
        let Ok(role) = env::var(MULTI_HOST_TEST_ROLE) else {
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
            let session = Session::builder().client(client).build().unwrap();
            let logical_mesh =
                |size| LogicalMesh::new(vec![MeshAxis::new("x", size, MeshAxisType::Explicit).unwrap()]).unwrap();
            let rows = |mesh: &DeviceMesh| {
                let dimensions = vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()];
                Sharding::new(mesh.logical_mesh().clone(), dimensions).unwrap()
            };
            let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
            let meshes = [
                DeviceMesh::new(logical_mesh(2), devices.clone()).unwrap(),
                DeviceMesh::new(logical_mesh(2), vec![devices[1], devices[0]]).unwrap(),
            ];
            let steps = across_mesh_steps();
            let mut steps = steps.iter();

            // `[[1, 2], [3, 4]]` is split by rows over the device of one rank alone, in a column-major layout. Its
            // shards differ from those of a split over both ranks, and the declared layout rules out shard regions, so
            // the owner gathers the value on its own device, keeps that replica for its own device of the destination
            // mesh and sends it to the other rank, which only receives it, and both ranks then run the final reshard
            // on the destination mesh. The agreement test of `crate::arrays::resharding` plans the same strategy for
            // the row `shared sharded with a declared layout -> sharded over both processes`.
            for owner in [0, 1] {
                let source_mesh = DeviceMesh::new(logical_mesh(1), vec![devices[owner]]).unwrap();
                let r#type = ArrayType::new_static(DataType::F32, [2, 2])
                    .with_layout(column_major.clone())
                    .with_sharding(rows(&source_mesh))
                    .unwrap();
                let values = values_to_bytes(&[1f32, 2., 3., 4.]);
                let source = Array::from_host_buffer(&session, r#type, source_mesh, None, values).unwrap();
                for mesh in &meshes {
                    let target = Placement::new(mesh.clone(), rows(mesh)).unwrap();
                    probe_placement_step(rank, steps.next().unwrap(), &session, &source, target);
                }
            }
            assert!(steps.next().is_none());
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_array_to_placement_reshards_across_meshes_on_two_cuda_processes() {
        check_cuda_rank_pair_placement_steps(
            "distributed::tests::test_array_to_placement_reshards_across_meshes_cuda_process_helper",
            &across_mesh_steps(),
            indoc! {r#"
                rank 0 from rank 0 onto the forward mesh shards: [([0..1, 0..2], "device", [1.0, 2.0])]
                rank 0 from rank 0 onto the reversed mesh shards: [([1..2, 0..2], "device", [3.0, 4.0])]
                rank 0 from rank 1 onto the forward mesh shards: [([0..1, 0..2], "device", [1.0, 2.0])]
                rank 0 from rank 1 onto the reversed mesh shards: [([1..2, 0..2], "device", [3.0, 4.0])]
                rank 1 from rank 0 onto the forward mesh shards: [([1..2, 0..2], "device", [3.0, 4.0])]
                rank 1 from rank 0 onto the reversed mesh shards: [([0..1, 0..2], "device", [1.0, 2.0])]
                rank 1 from rank 1 onto the forward mesh shards: [([1..2, 0..2], "device", [3.0, 4.0])]
                rank 1 from rank 1 onto the reversed mesh shards: [([0..1, 0..2], "device", [1.0, 2.0])]"#},
        );
    }

    // Returns the steps of `test_array_to_placement_receives_only_cuda_process_helper`.
    #[cfg(feature = "cuda-13")]
    fn receive_only_steps() -> Vec<String> {
        ["device", "pinned host", "unpinned host"].map(|memory| format!("into {memory} memory")).to_vec()
    }

    // Worker of `test_array_to_placement_receives_only_on_two_cuda_processes`, which returns immediately unless that
    // test launched it as one of its two ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_array_to_placement_receives_only_cuda_process_helper() {
        let Ok(role) = env::var(MULTI_HOST_TEST_ROLE) else {
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
            let session = Session::builder().client(client).build().unwrap();
            let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
            let replicated = Sharding::replicated(logical_mesh.clone(), 2);
            let source_mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[0]]).unwrap();
            let destination_mesh = DeviceMesh::new(logical_mesh, vec![devices[1]]).unwrap();
            let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
            let steps = receive_only_steps();

            // Rank 0 holds `[[1, 2], [3, 4]]` in a declared column-major layout and only sends it, converted to
            // the row-major layout that the receive allocates, and rank 1 only receives it. The receive records no
            // layout, because its layout differs from the declared one, so rank 1 then converts its buffer to the
            // declared layout and copies it to the requested memory, while rank 0, which holds no destination shard,
            // records the same metadata without any work.
            let r#type = ArrayType::new_static(DataType::F32, [2, 2])
                .with_layout(column_major)
                .with_sharding(replicated.clone())
                .unwrap();
            let values = values_to_bytes(&[1f32, 2., 3., 4.]);
            let source = Array::from_host_buffer(&session, r#type, source_mesh, None, values).unwrap();
            let memories = [Memory::Device, Memory::Host { pinned: true }, Memory::Host { pinned: false }];
            for (step, memory) in steps.iter().zip(memories) {
                let target = Placement::new(destination_mesh.clone(), replicated.clone()).unwrap();
                probe_placement_step(rank, step, &session, &source, target.with_memory(memory));
            }
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_array_to_placement_receives_only_on_two_cuda_processes() {
        check_cuda_rank_pair_placement_steps(
            "distributed::tests::test_array_to_placement_receives_only_cuda_process_helper",
            &receive_only_steps(),
            indoc! {r#"
                rank 0 into device memory shards: []
                rank 0 into pinned host memory shards: []
                rank 0 into unpinned host memory shards: []
                rank 1 into device memory shards: [([0..2, 0..2], "device", [1.0, 2.0, 3.0, 4.0])]
                rank 1 into pinned host memory shards: [([0..2, 0..2], "pinned_host", [1.0, 2.0, 3.0, 4.0])]
                rank 1 into unpinned host memory shards: [([0..2, 0..2], "unpinned_host", [1.0, 2.0, 3.0, 4.0])]"#},
        );
    }
}
