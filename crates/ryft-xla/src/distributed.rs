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
//! Drop order matters: keep the [`DistributedRuntime`] alive at least as long as the
//! [`ryft_pjrt::Client`] it produced, because the client borrows the runtime's key-value store.

use ryft_pjrt::{
    Client, ClientOptions, DistributedKeyValueStore, DistributedRuntimeClientOptions, DistributedRuntimeService,
    DistributedRuntimeServiceOptions, Plugin,
};

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
    /// Distributed key-value store backed by this process's [`DistributedRuntimeClient`]. PJRT
    /// clients minted via [`Self::create_client`] borrow this store for the duration of their
    /// lifetime.
    ///
    /// **Drop order**: This field comes first so the client is torn down before the
    /// coordinator service. Reversing the order causes the client's heartbeat-polling RPC to
    /// outlive the service and surface as a spurious "service vanished" error during normal
    /// cleanup.
    key_value_store: DistributedKeyValueStore,

    /// Coordinator service, owned by the elected coordinator node (node 0). Workers carry
    /// `None` here. The field is read only via its `Drop` impl, hence the leading underscore.
    _service: Option<DistributedRuntimeService>,
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
        Ok(Self { _service: service, key_value_store: DistributedKeyValueStore::new(client) })
    }

    /// Returns the [`DistributedKeyValueStore`] backed by this runtime. Useful for callers that
    /// want to mint a [`ryft_pjrt::Client`] manually via
    /// [`ryft_pjrt::Plugin::client_with_key_value_store`].
    #[inline]
    pub fn key_value_store(&self) -> &DistributedKeyValueStore {
        &self.key_value_store
    }

    /// Mints a PJRT [`Client`] bound to this runtime's key-value store. The returned client
    /// shares the runtime's lifetime: it remains valid as long as `self` is alive and is
    /// invalidated when the runtime is dropped.
    pub fn create_client<'s>(
        &'s self,
        plugin: &Plugin,
        options: ClientOptions,
    ) -> Result<Client<'s>, ryft_pjrt::Error> {
        plugin.client_with_key_value_store(options, &self.key_value_store)
    }
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;

    use ryft_pjrt::{ClientOptions, CpuClientOptions, load_cpu_plugin};

    use super::DistributedRuntime;

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
        use ryft_pjrt::KeyValueStore;
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
}
