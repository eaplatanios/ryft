//! Distributed runtime setup for the Ryft XLA backend. A distributed job consists of several processes that each own
//! some devices and that cooperate on one program. The processes find each other through a coordination service, which
//! one of them hosts and all of them connect to, and they exchange small records through the key-value store of that
//! service (e.g., the rendezvous data of cross-host transfers and collectives, compiled artifacts, and device
//! inventories). [`DistributedRuntime`] is the handle of one process on that service.
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
//!   3. **Shared Resources:** [`DistributedRuntime::compilation_artifact_exchange`] lets the processes share
//!      compiled artifacts, so that one process compiles each program and the others receive its result.
//!      [`DistributedRuntime::key_value_store`] serves other coordination protocols (e.g., the
//!      kernel coordination of [`crate::kernels`]).
//!   4. **Shutdown:** Dropping the last handle of a runtime disconnects its process and, on the coordinator, stops
//!      the service. A client borrows the key-value store of its runtime, so a session keeps a handle of its runtime
//!      for the rest of the process, as it keeps its client.
//!
//! The coordination service never overwrites or removes a record. Every protocol of this module therefore writes each
//! record once, and the records of a launch (including compiled artifacts) stay in the memory of the coordinator until
//! its service stops.
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

use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
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

/// Duration for which [`DistributedRuntime::initialize`] keeps trying to connect to the coordination service.
const DEFAULT_INITIALIZATION_TIMEOUT: Duration = Duration::from_secs(300);

/// Version of the compilation artifact exchange protocol, which every participant must share.
const ARTIFACT_EXCHANGE_SCHEMA_VERSION: u32 = 1;

/// Key-value store key under which the coordinator publishes the launch identity.
const LAUNCH_ID_KEY: &[u8] = b"ryft/distributed-launch/v1/id";

/// Handle of one process on the coordination service of a distributed Ryft XLA job. Refer to
/// the [module documentation](self) for the lifecycle of a distributed job. The handle owns the
/// [`DistributedRuntimeClient`](ryft_pjrt::DistributedRuntimeClient) of its process, wrapped in a
/// [`DistributedKeyValueStore`], and, on the coordinator only, the [`DistributedRuntimeService`] that hosts the
/// coordination state of every process. Clones share that state, which is released when the last of them is dropped.
/// A PJRT [`Client`] that [`Self::create_client`] returns borrows the key-value store of the handle, so the handle
/// must outlive it. [`SessionBuilder::distributed_runtime`](crate::SessionBuilder::distributed_runtime) takes care
/// of that for the client of a [`Session`](crate::Session), which lives for the rest of the process.
#[derive(Clone)]
pub struct DistributedRuntime {
    state: Arc<DistributedRuntimeState>,
}

impl DistributedRuntime {
    /// Initializes the [`DistributedRuntime`] of one process of a distributed job. Every process of the job must call
    /// this function with the same `coordinator_address` and `process_count` and with its own `process_index`. Process
    /// 0 is the coordinator, so it hosts the [`DistributedRuntimeService`] in addition to its own
    /// [`DistributedRuntimeClient`](ryft_pjrt::DistributedRuntimeClient). The call blocks until
    /// every process has connected.
    ///
    /// This function uses the default [`DistributedRuntimeServiceOptions`] and [`DistributedRuntimeClientOptions`],
    /// except that a process keeps trying to reach the coordinator for five minutes, and that a missed heartbeat does
    /// not terminate the process. Use [`Self::initialize_with_options`] for other timeouts.
    ///
    /// # Parameters
    ///
    ///   - `plugin`: PJRT [`Plugin`] that provides the service and the client.
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
    /// request, or shutdown timeouts). `service_options.num_nodes` is the process count of the distributed job and
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
                key_prefix_counts: Mutex::new(HashMap::new()),
                process_index: process_index as ProcessIndex,
                process_count: process_count as usize,
                request_timeout,
                is_shut_down: AtomicBool::new(false),
                service,
            }),
        })
    }

    /// Returns the index of this process among the processes of the distributed job.
    #[inline]
    pub fn process_index(&self) -> ProcessIndex {
        self.state.process_index
    }

    /// Returns the number of processes of the distributed job.
    #[inline]
    pub fn process_count(&self) -> usize {
        self.state.process_count
    }

    /// Returns the [`DistributedKeyValueStore`] of this runtime, which every process of the distributed job shares,
    /// for coordination protocols of the caller. The store never overwrites or removes a record.
    #[inline]
    pub fn key_value_store(&self) -> &DistributedKeyValueStore {
        &self.state.key_value_store
    }

    /// Disconnects this process from the coordination service and, on the coordinator, stops the service. Disconnecting
    /// waits until every process of the job disconnects, for the shutdown timeout of the client options. Later calls on
    /// any handle of this runtime do nothing. Dropping the last handle shuts the runtime down as well, so this function
    /// is only needed to shut it down earlier or to observe the error of doing so. Clients and exchanges of the runtime
    /// cannot use its key-value store afterwards.
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

    /// Returns whether `other` is a handle of this runtime (i.e., a clone of this handle).
    #[inline]
    pub(crate) fn is_same_runtime(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.state, &other.state)
    }

    // TODO(eaplatanios): Review from here onwards.

    /// Reserves a key prefix in the key-value store of this runtime for one instance of the coordination protocol
    /// `protocol` (e.g., one coordinator of a protocol that keeps a round counter per coordinator). The prefix consists
    /// of `protocol`, the launch identity, and the number of prefixes that any handle of this runtime reserved for
    /// `protocol` before in this process, and so it differs from every prefix reserved before. The processes of a job
    /// agree on the prefix of an instance without communicating when each of them reserves the prefixes of `protocol`
    /// in the same order. Prefixes of other protocols do not affect that order.
    pub(crate) fn reserve_key_prefix(&self, protocol: &'static str) -> Vec<u8> {
        let mut counts = self.state.key_prefix_counts.lock().unwrap_or_else(PoisonError::into_inner);
        let count = counts.entry(protocol).or_insert(0u64);
        let mut prefix = protocol.as_bytes().to_vec();
        prefix.extend_from_slice(&self.state.launch_id);
        prefix.extend_from_slice(&count.to_le_bytes());
        *count += 1;
        prefix
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

/// Shared state of a [`DistributedRuntime`], retained by every clone of the runtime and by its artifact exchanges.
struct DistributedRuntimeState {
    /// Key-value store backed by the distributed runtime client of this process.
    key_value_store: DistributedKeyValueStore,

    /// Launch identity that the coordinator generated and published during initialization.
    launch_id: [u8; 32],

    /// Number of key prefixes that [`DistributedRuntime::reserve_key_prefix`] reserved so far for each protocol.
    key_prefix_counts: Mutex<HashMap<&'static str, u64>>,

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

#[cfg(test)]
pub(crate) mod tests {
    use std::fs::File;
    #[cfg(feature = "cuda-13")]
    use std::io::Write;
    #[cfg(feature = "cuda-13")]
    use std::process::Stdio;
    use std::process::{Child, Command};
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use std::{env, fs, ptr, thread};

    #[cfg(feature = "cuda-13")]
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        ArrayType, CompilationArtifactExchange, CompilationArtifactExchangeError, DataType, Device, DeviceMesh,
        LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension,
    };
    #[cfg(feature = "cuda-13")]
    use ryft_pjrt::{
        BufferType, Client, ExecutionDeviceInputs, ExecutionInput, GpuMemoryAllocator, GpuPlatform,
        Program as PjrtProgram, load_cuda_13_plugin,
    };
    use ryft_pjrt::{
        ClientOptions, CpuClientOptions, DistributedRuntimeClientOptions, DistributedRuntimeServiceOptions,
        Error as PjrtError, GpuClientOptions, KeyValueStore, load_cpu_plugin,
    };
    use sha2::{Digest, Sha256};

    #[cfg(feature = "cuda-13")]
    use crate::tests::test_compilation_options;
    use crate::tests::{loopback_address, values_from_bytes, values_to_bytes};
    use crate::{Array, Error, Platform, Session};

    use super::{
        ARTIFACT_EXCHANGE_SCHEMA_VERSION, CompilationArtifactManifest, CompilationCompatibilityManifest,
        DistributedCompilationArtifactExchange, DistributedRuntime,
    };

    /// Environment variable that carries the rank of a process that [`run_test_processes`] started.
    const PROCESS_TEST_ROLE: &str = "RYFT_DISTRIBUTED_TEST_ROLE";

    /// Environment variable that carries the coordinator address of a launch of [`run_test_processes`].
    const PROCESS_TEST_ADDRESS: &str = "RYFT_DISTRIBUTED_TEST_ADDRESS";

    /// Key under which the follower of the two-process exchange test acknowledges its artifact.
    const PROCESS_TEST_ACKNOWLEDGEMENT: &[u8] = b"ryft/test/artifact-consumed";

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

    /// Returns the rank of this process and the coordinator address of its launch when [`run_test_processes`] started
    /// this process to run a worker test, and [`None`] otherwise (e.g., when a worker test is run on its own), in which
    /// case the worker test returns immediately.
    pub(crate) fn test_process() -> Option<(usize, String)> {
        let rank = env::var(PROCESS_TEST_ROLE).ok()?.parse().unwrap();
        Some((rank, env::var(PROCESS_TEST_ADDRESS).unwrap()))
    }

    // TODO(eaplatanios): Split the workers that check many behaviors in one test, so that each multi-process test
    // checks one behavior of its owning item. The largest are `check_shard_map_multi_host_scenarios` in
    // `operations/sharding/shard_map.rs`, `test_session_place_cpu_process_helper` in `arrays/placement.rs`, the
    // step lists that `check_cuda_rank_pair_placement_steps` runs in `arrays/resharding.rs`, and the modes of
    // `test_distributed_kernel_process` in `kernels/distributed.rs`, whose launcher should also be replaced by
    // this one.
    /// Runs the worker test `helper`, given as its path within this test binary, in `process_count` processes that
    /// stand for the ranks of one job, with `environment` added to their environment, and returns their standard
    /// outputs in rank order. Each process finds its rank and a coordinator address on a free loopback port with
    /// [`test_process`]. Panics with the outputs of every process if a process fails, if a process did not run and
    /// pass exactly its worker test, or if the processes do not finish within `timeout`. A failed process ends the
    /// launch, because its peers would wait for it, and no process outlives this function.
    fn run_test_processes(
        helper: &str,
        process_count: usize,
        environment: &[(&str, String)],
        timeout: Duration,
    ) -> Vec<String> {
        /// Processes of one launch, which are killed when dropped.
        struct Processes(Vec<Child>);

        impl Drop for Processes {
            fn drop(&mut self) {
                for process in &mut self.0 {
                    // Panicking here could abort a test that is already unwinding, so a failure is only reported.
                    if let Err(error) = process.kill().and_then(|()| process.wait()) {
                        eprintln!("failed to stop a process of a distributed test: {error}");
                    }
                }
            }
        }

        let address = loopback_address();
        let directory = tempfile::tempdir().unwrap();
        let output_path = |rank: usize, stream: &str| directory.path().join(format!("rank-{rank}.{stream}"));
        // Each process writes its output to files rather than pipes, so verbose native logs cannot block it.
        let executable = env::current_exe().unwrap();
        let mut processes = Processes(Vec::with_capacity(process_count));
        for rank in 0..process_count {
            let process = Command::new(&executable)
                .args(["--exact", helper, "--ignored", "--nocapture", "--test-threads=1"])
                .envs(environment.iter().map(|(name, value)| (*name, value)))
                .env(PROCESS_TEST_ROLE, rank.to_string())
                .env(PROCESS_TEST_ADDRESS, &address)
                .stdout(File::create(output_path(rank, "stdout")).unwrap())
                .stderr(File::create(output_path(rank, "stderr")).unwrap())
                .spawn()
                .unwrap();
            processes.0.push(process);
        }
        let deadline = Instant::now() + timeout;
        let mut statuses = vec![None; process_count];
        while statuses.iter().any(Option::is_none) && Instant::now() < deadline {
            for (status, process) in statuses.iter_mut().zip(&mut processes.0) {
                if status.is_none() {
                    *status = process.try_wait().unwrap();
                }
            }
            if statuses.iter().flatten().any(|status| !status.success()) {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        drop(processes);

        // libtest prints `test <name> ... ` without a line break before a worker runs, so its output follows that
        // prefix.
        let outputs = (0..process_count)
            .map(|rank| {
                let stdout = fs::read_to_string(output_path(rank, "stdout")).unwrap();
                let stderr = fs::read_to_string(output_path(rank, "stderr")).unwrap();
                (stdout, stderr)
            })
            .collect::<Vec<_>>();
        let passed = statuses.iter().zip(&outputs).all(|(status, (stdout, _))| {
            status.is_some_and(|status| status.success())
                && stdout.contains(&format!("test {helper} ... "))
                && stdout.contains("test result: ok. 1 passed; 0 failed; 0 ignored;")
        });
        if !passed {
            let report = statuses
                .iter()
                .zip(&outputs)
                .enumerate()
                .map(|(rank, (status, (stdout, stderr)))| {
                    let status =
                        status.map_or("no status before it was stopped".to_string(), |status| status.to_string());
                    format!("rank {rank} ended with {status}\nstdout:\n{stdout}\nstderr:\n{stderr}")
                })
                .collect::<Vec<_>>()
                .join("\n");
            panic!("distributed test `{helper}` failed:\n{report}");
        }
        outputs.into_iter().map(|(stdout, _)| stdout).collect()
    }

    /// Runs the worker test `helper` in `process_count` CPU processes of one job and returns their standard outputs in
    /// rank order (refer to [`run_test_processes`]).
    pub(crate) fn run_cpu_processes(helper: &str, process_count: usize) -> Vec<String> {
        run_test_processes(helper, process_count, &[], Duration::from_secs(30))
    }

    /// Joins the two-process CUDA job that [`run_cuda_rank_pair`] launched this process into, runs `check` with a
    /// client whose two global devices are the ranks' shares of CUDA device 0, and then shuts the job down in order.
    /// Returns immediately in a process that no launcher started (refer to [`test_process`]).
    #[cfg(feature = "cuda-13")]
    pub(crate) fn run_as_cuda_rank(check: impl FnOnce(&'static Client<'static>)) {
        let Some((process_index, address)) = test_process() else {
            return;
        };
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
    /// device 0, asserts that both ranks pass, and returns their standard outputs (refer to [`run_test_processes`]).
    ///
    /// The CPU plugin cannot host such jobs, because a CPU client that joins a distributed runtime sees only its own
    /// devices (e.g., process 1 sees just device 2048), so it has no cross-process topology for collectives. Instead,
    /// both ranks share CUDA device 0 through NCCL's multiple ranks per GPU, with CUDA MPS for concurrent progress,
    /// which mirrors the `cuda-distributed-collectives` differential suite that `python/README.md` describes.
    #[cfg(feature = "cuda-13")]
    pub(crate) fn run_cuda_rank_pair(helper: &str) -> [String; 2] {
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
            fs::create_dir(&pipes).unwrap();
            fs::create_dir(&logs).unwrap();
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

        run_test_processes(helper, 2, &environment, Duration::from_secs(300)).try_into().unwrap()
    }

    /// Runs the worker test `helper` as the two ranks of a CUDA job (refer to [`run_cuda_rank_pair`]) and returns the
    /// lines that its ranks printed after `probe: `, those of rank 0 first.
    #[cfg(feature = "cuda-13")]
    pub(crate) fn run_cuda_rank_pair_probes(helper: &str) -> String {
        let outputs = run_cuda_rank_pair(helper);
        let probes = outputs
            .iter()
            .flat_map(|output| output.lines().filter_map(|line| line.split_once("probe: ").map(|(_, probe)| probe)))
            .collect::<Vec<_>>();
        probes.join("\n")
    }

    /// Returns the elements of the buffer of every shard of `array` that the current process addresses, in mesh order.
    pub(crate) fn read_addressable_f32s(array: &Array) -> Vec<Vec<f32>> {
        array
            .addressable_shards()
            .map(|shard| {
                let bytes = shard.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
                values_from_bytes::<f32>(bytes.as_slice())
            })
            .collect()
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
    fn test_distributed_runtime_reserve_key_prefix() {
        let runtime = single_process_runtime();
        let launch_id = runtime.state.launch_id;
        let prefix = |protocol: &str, count: u64| [protocol.as_bytes(), &launch_id, &count.to_le_bytes()].concat();
        assert_eq!(runtime.reserve_key_prefix("first/"), prefix("first/", 0));

        // Every handle of the runtime counts the prefixes of a protocol together, and each protocol is counted apart.
        assert_eq!(runtime.clone().reserve_key_prefix("first/"), prefix("first/", 1));
        assert_eq!(runtime.reserve_key_prefix("second/"), prefix("second/", 0));
        assert_eq!(runtime.reserve_key_prefix("first/"), prefix("first/", 2));

        // Another launch has another identity, and so its prefixes differ.
        assert_ne!(single_process_runtime().reserve_key_prefix("first/"), prefix("first/", 0));
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
    #[ignore = "entry point of a process of a multi-process test"]
    fn test_distributed_runtime_remote_devices_cpu_process_helper() {
        let Some((rank, address)) = test_process() else {
            return;
        };
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
        run_cpu_processes("distributed::tests::test_distributed_runtime_remote_devices_cpu_process_helper", 2);
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
        let manifest = CompilationCompatibilityManifest { schema_version: 2, process_count: 2 };
        let manifest = serde_json::to_vec(&manifest).unwrap();
        runtime.key_value_store().put(&follower.key(b"key", b"preflight/manifest"), &manifest).unwrap();
        assert_eq!(
            follower.validate_compatibility(b"key", 0, Duration::from_secs(1)),
            Err(CompilationArtifactExchangeError::Incompatible {
                message: format!(
                    "process 0 uses compilation artifact exchange schema 2, but this process uses schema \
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

        publish(b"other-schema", 2, 6, 2);
        assert_eq!(
            follower.receive(b"other-schema", Duration::from_secs(1)),
            failed("unsupported compiled artifact exchange schema 2"),
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
    #[ignore = "entry point of a process of a multi-process test"]
    fn test_distributed_compilation_artifact_exchange_cpu_process_helper() {
        let Some((process_index, address)) = test_process() else {
            return;
        };
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
        run_cpu_processes("distributed::tests::test_distributed_compilation_artifact_exchange_cpu_process_helper", 2);
    }

    // Worker of `test_gpu_only_collectives_on_two_cuda_processes`, which returns immediately unless that test launched
    // it as one of its two ranks. It compiles and executes each probe module with this rank's tile of `[1, 2, 3, 4]`
    // and prints one `probe:` line per module with its outputs or the first line of its compilation or execution
    // error.
    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "entry point of a process of a multi-process test"]
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
        run_as_cuda_rank(|client| {
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
}
