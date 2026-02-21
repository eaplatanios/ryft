use std::ffi::{CStr, CString};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use ryft_xla_sys::distributed as ffi;

use crate::{Api, Error, Plugin, invoke_distributed_api_error_fn, invoke_distributed_api_void_fn};

/// Represents a key-value store that can be used by PJRT [`Client`](crate::Client)s for process-to-process coordination
/// in distributed environments. Note that the functions of this trait are expected to be thread-safe as they may be
/// called concurrently by [`Client`](crate::Client)s.
pub trait KeyValueStore {
    /// Stores the provided `value` under the provided `key`.
    fn put(&self, key: &[u8], value: &[u8]) -> Result<(), Error>;

    /// Retrieves the value for the provided `key`, potentially blocking until the value is available or the provided
    /// `timeout` expires. Returns [`Error::NotFound`] if the provided `key` does not exist in this [`KeyValueStore`].
    fn get(&self, key: &[u8], timeout: Duration) -> Result<Vec<u8>, Error>;

    /// Attempts to retrieve the value for the provided `key` without blocking. Returns [`Error::NotFound`] if the
    /// provided `key` does not exist in this [`KeyValueStore`].
    fn try_get(&self, key: &[u8]) -> Result<Vec<u8>, Error>;
}

/// [`KeyValueStore`] implementation backed by a [`DistributedRuntimeClient`]. This wrapper encodes binary keys and
/// values as hexadecimal strings before forwarding them to the underlying [`DistributedRuntimeClient`].
///
/// To use this [`KeyValueStore`], you must first instantiate a [`DistributedRuntimeService`] on some process and then
/// instantiate one [`DistributedRuntimeClient`] configured to _talk_ to that service for each client process that
/// participates in a distributed process.
pub struct DistributedKeyValueStore {
    /// [`DistributedRuntimeClient`] used to communicate with the underlying [`DistributedRuntimeService`].
    client: DistributedRuntimeClient,
}

impl DistributedKeyValueStore {
    /// Creates a new [`DistributedKeyValueStore`] backed by the provided [`DistributedRuntimeClient`].
    pub fn new(client: DistributedRuntimeClient) -> Self {
        Self { client }
    }

    /// Returns the underlying [`DistributedRuntimeClient`].
    pub fn client(&self) -> &DistributedRuntimeClient {
        &self.client
    }
}

impl KeyValueStore for DistributedKeyValueStore {
    fn put(&self, key: &[u8], value: &[u8]) -> Result<(), Error> {
        self.client.key_value_set(hex::encode(key), hex::encode(value))
    }

    fn get(&self, key: &[u8], timeout: Duration) -> Result<Vec<u8>, Error> {
        let encoded = self.client.blocking_key_value_get(hex::encode(key), timeout)?;
        hex::decode(encoded)
            .map_err(|error| Error::invalid_argument(format!("invalid hex-encoded key-value payload; {error}")))
    }

    fn try_get(&self, key: &[u8]) -> Result<Vec<u8>, Error> {
        let encoded = self.client.key_value_try_get(hex::encode(key))?;
        hex::decode(encoded)
            .map_err(|error| Error::invalid_argument(format!("invalid hex-encoded key-value payload; {error}")))
    }
}

/// Options used to configure a [`DistributedRuntimeService`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct DistributedRuntimeServiceOptions {
    /// Number of nodes participating in the distributed process.
    pub num_nodes: u32,

    /// Timeout used by the [`DistributedRuntimeService`] for heartbeat checks. Specifically, this represents how long
    /// the [`DistributedRuntimeService`] will wait to receive a heartbeat from a [`DistributedRuntimeClient`] before
    /// concluding that the client has vanished.
    pub heartbeat_timeout: Duration,

    /// Timeout used by the [`DistributedRuntimeService`] while waiting for all nodes to register. Specifically, this
    /// represents how long the service should wait for all clients to call [`DistributedRuntimeClient::connect`] before
    /// giving up.
    pub cluster_register_timeout: Duration,

    /// Timeout used by the [`DistributedRuntimeService`] when shutting down. Specifically, this represents how
    /// long to wait for all nodes to call [`DistributedRuntimeClient::shutdown`]. If the timeout expires, then
    /// [`DistributedRuntimeService::shutdown`] will report an error and return control.
    pub shutdown_timeout: Duration,
}

impl Default for DistributedRuntimeServiceOptions {
    fn default() -> Self {
        Self {
            num_nodes: 1,
            heartbeat_timeout: Duration::from_secs(100),
            cluster_register_timeout: Duration::from_mins(60),
            shutdown_timeout: Duration::from_mins(5),
        }
    }
}

/// Coordination service for distributed PJRT jobs. A [`DistributedRuntimeService`] must be created for each
/// distributed job (typically on a designated coordinator process), using [`Plugin::distributed_runtime_service`],
/// and all participating worker processes must create a [`DistributedRuntimeClient`] (each) that points to the same
/// service address, using [`Plugin::distributed_runtime_client`]. Each client must then call
/// [`DistributedRuntimeClient::connect`] to join the same coordination domain. Once clients are connected,
/// they can exchange rendezvous data through the [`KeyValueStore`] API by using the [`DistributedRuntimeClient`]
/// to create a [`DistributedKeyValueStore`].
pub struct DistributedRuntimeService {
    /// Handle that represents this [`DistributedRuntimeService`] in the PJRT C API.
    handle: *mut ffi::PJRT_Distributed_Runtime_Service,

    /// Boolean flag that tracks whether this [`DistributedRuntimeService`] has already been shut down.
    is_shut_down: AtomicBool,

    /// Underlying PJRT [`Api`].
    api: Api,
}

impl DistributedRuntimeService {
    /// Constructs a new [`DistributedRuntimeService`] from the provided
    /// [`PJRT_Distributed_Runtime_Service`](ffi::PJRT_Distributed_Runtime_Service) handle that came from a function
    /// in the PJRT C API.
    pub(crate) fn from_c_api(handle: *mut ffi::PJRT_Distributed_Runtime_Service, api: Api) -> Result<Self, Error> {
        if handle.is_null() {
            Err(Error::invalid_argument("the provided XLA distributed runtime service handle is a null pointer"))
        } else {
            Ok(Self { handle, api, is_shut_down: AtomicBool::new(false) })
        }
    }

    /// Returns the [`PJRT_Distributed_Runtime_Service`](ffi::PJRT_Distributed_Runtime_Service) that corresponds to this
    /// [`DistributedRuntimeService`] and which can be passed to functions in the PJRT C API.
    pub(crate) unsafe fn to_c_api(&self) -> *mut ffi::PJRT_Distributed_Runtime_Service {
        self.handle
    }

    /// Shuts down this [`DistributedRuntimeService`]. Note that it _is not_ required to call this function to shut down
    /// the service; it is only required to call it if you want to shut down the service before the process exits. That
    /// is because the [`DistributedRuntimeService`] will always shut down when it is dropped.
    pub fn shutdown(&self) {
        if !self.is_shut_down.load(Ordering::Acquire) {
            invoke_distributed_api_void_fn!(self.api, PJRT_Distributed_Runtime_Service_Shutdown, {
                service = self.to_c_api()
            });
        }
        self.is_shut_down.store(true, Ordering::Release);
    }
}

unsafe impl Send for DistributedRuntimeService {}
unsafe impl Sync for DistributedRuntimeService {}

impl Drop for DistributedRuntimeService {
    fn drop(&mut self) {
        self.shutdown();
        invoke_distributed_api_void_fn!(self.api, PJRT_Distributed_Runtime_Service_Destroy, {
            service = self.to_c_api()
        });
    }
}

/// Callback signature used by [`DistributedRuntimeClientOptions::missed_heartbeat_callback`]. The callback receives
/// an optional [`Error`] corresponding to the distributed runtime status reported by XLA.
pub type MissedHeartbeatCallback = Box<dyn FnMut(Option<Error>) + Send + 'static>;

/// Options used to configure a [`DistributedRuntimeClient`].
pub struct DistributedRuntimeClientOptions {
    /// Global unique ID of the [`DistributedRuntimeClient`] in the distributed process.
    pub node_id: u32,

    /// Timeout used by the [`DistributedRuntimeClient`] for RPC calls that do not have their own timeout specified.
    pub rpc_timeout: Duration,

    /// Timeout used by the [`DistributedRuntimeClient`] while connecting to the [`DistributedRuntimeService`].
    /// Specifically, this is the duration for which to keep re-trying [`DistributedRuntimeClient::connect`] while
    /// trying to connect to the [`DistributedRuntimeService`]. The client will keep trying to open the initial
    /// connection for this period, even if any individual [`DistributedRuntimeClient::connect`] RPC fails. Note that
    /// this may be zero, in which case [`DistributedRuntimeClient::connect`] will only be attempted once.
    pub initialization_timeout: Duration,

    /// Timeout used by the [`DistributedRuntimeClient`] when shutting down. Specifically, this represents how
    /// long to wait for all nodes to call [`DistributedRuntimeClient::shutdown`]. If the timeout expires, then
    /// [`DistributedRuntimeService::shutdown`] will report an error and return control.
    pub shutdown_timeout: Duration,

    /// Timeout used by the [`DistributedRuntimeClient`] for heartbeat checks. Specifically, this represents how long
    /// the [`DistributedRuntimeClient`] will wait to receive a heartbeat from the [`DistributedRuntimeService`] before
    /// concluding that the service has vanished.
    pub heartbeat_timeout: Duration,

    /// Boolean flag which controls whether a distributed job can continue even if the [`DistributedRuntimeClient`]
    /// built using these [`DistributedRuntimeClientOptions`] fails.
    pub recoverable: bool,

    /// Boolean flag which controls whether to use compression for the communication between the
    /// [`DistributedRuntimeClient`] built using these [`DistributedRuntimeClientOptions`] and its
    /// corresponding [`DistributedRuntimeService`].
    pub use_compression: bool,

    /// Optional callback invoked by the [`DistributedRuntimeClient`] when notification of a missing heartbeat is
    /// reported by the [`DistributedRuntimeService`], or we have not heard from the service recently. This is primarily
    /// exposed so that tests can override it to a callback that does not `panic!`.
    pub missed_heartbeat_callback: Option<MissedHeartbeatCallback>,
}

impl Default for DistributedRuntimeClientOptions {
    fn default() -> Self {
        Self {
            node_id: 0,
            rpc_timeout: Duration::from_secs(120),
            initialization_timeout: Duration::ZERO,
            shutdown_timeout: Duration::from_mins(5),
            heartbeat_timeout: Duration::from_secs(100),
            recoverable: false,
            use_compression: false,
            missed_heartbeat_callback: Some(Box::new(|error| {
                panic!(
                    "Terminating process because the Ryft distributed service detected fatal errors. This most likely \
                    indicates that another task died; see the other task logs for more details. Error: {}",
                    error.unwrap_or(Error::unknown("unknown")),
                );
            })),
        }
    }
}

/// Owned callback state that is passed through the distributed runtime `user_arg` to the C API wrapper
/// function of a [`MissedHeartbeatCallback`].
pub(crate) struct MissedHeartbeatCallbackState {
    /// [`MissedHeartbeatCallback`] guarded by a [`Mutex`] because the runtime may invoke it from arbitrary threads.
    callback: Mutex<MissedHeartbeatCallback>,

    /// Underlying PJRT [`Api`].
    api: Api,
}

/// Per-process [`DistributedRuntimeClient`] used to communicate with a [`DistributedRuntimeService`].
/// A [`DistributedRuntimeClient`] connects one process to a shared [`DistributedRuntimeService`], enabling
/// that process to participate in distributed coordination for a distributed job. Refer to the documentation of
/// [`DistributedRuntimeService`] for more information on how to use [`DistributedRuntimeClient`]s to construct
/// [`DistributedKeyValueStore`]s.
pub struct DistributedRuntimeClient {
    /// Handle that represents this [`DistributedRuntimeClient`] in the PJRT C API.
    handle: *mut ffi::PJRT_Distributed_Runtime_Client,

    /// Optional state for the missed-heartbeat callback that wraps the configured [`MissedHeartbeatCallback`]
    /// and is used to enable invoking the wrapped [`MissedHeartbeatCallback`] from within PJRT. This is stored
    /// in the [`DistributedRuntimeClient`] in order for it to stay alive for the duration of the lifetime of the
    /// [`DistributedRuntimeClient`]. If it was not stored here, then it would be dropped immediately after the
    /// call to [`Plugin::distributed_runtime_client`] that constructed this [`DistributedRuntimeClient`].
    #[allow(dead_code)]
    missed_heartbeat_callback_state: Option<Box<MissedHeartbeatCallbackState>>,

    /// Underlying PJRT [`Api`].
    api: Api,
}

impl DistributedRuntimeClient {
    /// Constructs a new [`DistributedRuntimeClient`] from the provided distributed runtime C API handle.
    pub(crate) fn from_c_api(
        handle: *mut ffi::PJRT_Distributed_Runtime_Client,
        missed_heartbeat_callback_state: Option<Box<MissedHeartbeatCallbackState>>,
        api: Api,
    ) -> Result<Self, Error> {
        if handle.is_null() {
            Err(Error::invalid_argument("the provided XLA distributed runtime client handle is a null pointer"))
        } else {
            Ok(Self { handle, missed_heartbeat_callback_state, api })
        }
    }

    /// Returns the underlying distributed runtime C API handle.
    pub(crate) unsafe fn to_c_api(&self) -> *mut ffi::PJRT_Distributed_Runtime_Client {
        self.handle
    }

    /// Connects this [`DistributedRuntimeClient`] to the [`DistributedRuntimeService`]. This function must always be
    /// called before trying to use any other functions of this [`DistributedRuntimeClient`] to properly initialize it.
    pub fn connect(&self) -> Result<(), Error> {
        invoke_distributed_api_error_fn!(self.api, PJRT_Distributed_Runtime_Client_Connect, {
            client = self.to_c_api()
        })
    }

    /// Stores the provided `value` under the provided `key` in the [`DistributedRuntimeService`].
    pub fn key_value_set<K: AsRef<str>, V: AsRef<str>>(&self, key: K, value: V) -> Result<(), Error> {
        let key = CString::new(key.as_ref()).map_err(|_| {
            Error::invalid_argument("XLA distributed runtime key-value key contains interior NUL bytes")
        })?;
        let value = CString::new(value.as_ref()).map_err(|_| {
            Error::invalid_argument("XLA distributed runtime key-value value contains interior NUL bytes")
        })?;
        invoke_distributed_api_error_fn!(
            self.api,
            PJRT_Distributed_Runtime_Client_Key_Value_Set,
            {
                client = self.to_c_api(),
                key = key.as_ptr(),
                value = value.as_ptr(),
            },
        )
    }

    /// Retrieves the value for the provided `key` from the [`DistributedRuntimeService`], potentially blocking until
    /// the value is available or the provided `timeout` expires. Returns [`Error::NotFound`] if the provided `key` does
    /// not exist in the [`DistributedRuntimeService`].
    pub fn blocking_key_value_get<K: AsRef<str>>(&self, key: K, timeout: Duration) -> Result<String, Error> {
        let key = CString::new(key.as_ref()).map_err(|_| {
            Error::invalid_argument("XLA distributed runtime key-value key contains interior NUL bytes")
        })?;
        invoke_distributed_api_error_fn!(
            self.api,
            PJRT_Distributed_Runtime_Client_Blocking_Key_Value_Get,
            {
                client = self.to_c_api(),
                key = key.as_ptr(),
                timeout = duration_to_seconds(timeout),
            },
            { value },
        )
        .map(|value| {
            if value.is_null() {
                String::new()
            } else {
                // The current distributed C API does not expose a value deleter callback for returned strings.
                unsafe { CStr::from_ptr(value) }.to_string_lossy().into_owned()
            }
        })
    }

    /// Attempts to retrieve the value for the provided `key` without blocking from the [`DistributedRuntimeService`].
    /// Returns [`Error::NotFound`] if the provided `key` does not exist in the [`DistributedRuntimeService`].
    pub fn key_value_try_get<K: AsRef<str>>(&self, key: K) -> Result<String, Error> {
        let key = CString::new(key.as_ref()).map_err(|_| {
            Error::invalid_argument("XLA distributed runtime key-value key contains interior NUL bytes")
        })?;
        invoke_distributed_api_error_fn!(
            self.api,
            PJRT_Distributed_Runtime_Client_Key_Value_Try_Get,
            {
                client = self.to_c_api(),
                key = key.as_ptr(),
            },
            { value },
        )
        .map(|value| {
            if value.is_null() {
                String::new()
            } else {
                // The current distributed C API does not expose a value deleter callback for returned strings.
                unsafe { CStr::from_ptr(value) }.to_string_lossy().into_owned()
            }
        })
    }

    /// Shuts down this [`DistributedRuntimeClient`]. Note that it _is not_ required to call this function to shut down
    /// the client; it is only required to call it if you want to shut down the client before the process exits. That
    /// is because the [`DistributedRuntimeClient`] will always shut down when it is dropped.
    pub fn shutdown(&self) -> Result<(), Error> {
        invoke_distributed_api_error_fn!(self.api, PJRT_Distributed_Runtime_Client_Shutdown, {
            client = self.to_c_api()
        })
    }
}

unsafe impl Send for DistributedRuntimeClient {}
unsafe impl Sync for DistributedRuntimeClient {}

impl Drop for DistributedRuntimeClient {
    fn drop(&mut self) {
        let _ = self.shutdown();
        invoke_distributed_api_void_fn!(self.api, PJRT_Distributed_Runtime_Client_Destroy, {
            client = self.to_c_api()
        });
    }
}

impl Plugin {
    /// Creates a new [`DistributedRuntimeService`] listening on `address`
    /// and using the provided [`DistributedRuntimeServiceOptions`].
    pub fn distributed_runtime_service<A: AsRef<str>>(
        &self,
        address: A,
        options: DistributedRuntimeServiceOptions,
    ) -> Result<DistributedRuntimeService, Error> {
        self.api().distributed_runtime_service(address, options)
    }

    /// Creates a new [`DistributedRuntimeClient`] communicating with a [`DistributedRuntimeService`] at `address`
    /// and using the provided [`DistributedRuntimeClientOptions`].
    pub fn distributed_runtime_client<A: AsRef<str>>(
        &self,
        address: A,
        options: DistributedRuntimeClientOptions,
    ) -> Result<DistributedRuntimeClient, Error> {
        self.api().distributed_runtime_client(address, options)
    }
}

impl Api {
    /// Creates a new [`DistributedRuntimeService`] listening on `address`
    /// and using the provided [`DistributedRuntimeServiceOptions`].
    pub(crate) fn distributed_runtime_service<A: AsRef<str>>(
        &self,
        address: A,
        options: DistributedRuntimeServiceOptions,
    ) -> Result<DistributedRuntimeService, Error> {
        let address = CString::new(address.as_ref()).map_err(|_| {
            Error::invalid_argument("XLA distributed runtime service address contains interior NUL bytes")
        })?;
        invoke_distributed_api_error_fn!(
            *self,
            PJRT_Distributed_Runtime_Service_New,
            {
                address = address.as_ptr(),
                num_nodes = options.num_nodes,
                heartbeat_timeout = duration_to_seconds(options.heartbeat_timeout),
                cluster_register_timeout = duration_to_seconds(options.cluster_register_timeout),
                shutdown_timeout = duration_to_seconds(options.shutdown_timeout),
            },
            { service },
        )
        .and_then(|handle| DistributedRuntimeService::from_c_api(handle, *self))
    }

    /// Creates a new [`DistributedRuntimeClient`] communicating with a [`DistributedRuntimeService`] at `address`
    /// and using the provided [`DistributedRuntimeClientOptions`].
    pub(crate) fn distributed_runtime_client<A: AsRef<str>>(
        &self,
        address: A,
        options: DistributedRuntimeClientOptions,
    ) -> Result<DistributedRuntimeClient, Error> {
        unsafe extern "C" fn missed_heartbeat_callback(error: *const ffi::PJRT_Error, user_arg: *mut std::ffi::c_void) {
            if !user_arg.is_null() {
                let state = unsafe { &*(user_arg as *mut MissedHeartbeatCallbackState) };
                let mut callback = state.callback.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                let error = unsafe { Error::from_c_api(error as *const crate::errors::ffi::PJRT_Error, state.api) }
                    .unwrap_or_else(|error| Some(error));
                callback(error);
            }
        }

        let address = CString::new(address.as_ref()).map_err(|_| {
            Error::invalid_argument("XLA distributed runtime client address contains interior NUL bytes")
        })?;

        let mut callback_state = options
            .missed_heartbeat_callback
            .map(|callback| Box::new(MissedHeartbeatCallbackState { callback: Mutex::new(callback), api: *self }));

        invoke_distributed_api_error_fn!(
            *self,
            PJRT_Distributed_Runtime_Client_New,
            {
                address = address.as_ptr(),
                node_id = options.node_id,
                rpc_timeout = duration_to_seconds(options.rpc_timeout),
                init_timeout = duration_to_seconds(options.initialization_timeout),
                shutdown_timeout = duration_to_seconds(options.shutdown_timeout),
                heartbeat_timeout = duration_to_seconds(options.heartbeat_timeout),
                missed_heartbeat_callback = missed_heartbeat_callback,
                missed_heartbeat_callback_user_arg = callback_state
                    .as_mut()
                    .map(|state| &mut **state as *mut MissedHeartbeatCallbackState as *mut std::ffi::c_void)
                    .unwrap_or(std::ptr::null_mut()),
                shutdown_on_destruction = true,
                recoverable = options.recoverable,
                use_compression = options.use_compression,
            },
            { client },
        )
        .and_then(|handle| DistributedRuntimeClient::from_c_api(handle, callback_state, *self))
    }
}

/// Converts the provided [`Duration`] to the number of seconds it corresponds to, rounding up any non-zero
/// subsecond part and saturating at [`u32::MAX`], so that it can be passed to C API functions.
fn duration_to_seconds(duration: Duration) -> u32 {
    duration.as_secs().saturating_add(u64::from(duration.subsec_nanos() > 0)).min(u32::MAX as u64) as u32
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    use crate::tests::test_cpu_plugin;
    use crate::{
        DistributedKeyValueStore, DistributedRuntimeClient, DistributedRuntimeClientOptions, DistributedRuntimeService,
        DistributedRuntimeServiceOptions, Error, KeyValueStore,
    };

    use super::duration_to_seconds;

    /// Returns a loopback address with an OS-assigned free ephemeral port for distributed-runtime tests. We bind
    /// to `127.0.0.1:0` so that the operating system selects an available port, which avoids flaky failures from
    /// hardcoded-port collisions across parallel tests, concurrent continuous integration jobs, or local processes.
    /// Returns `None` when the environment denies binding a local test port.
    fn test_loopback_address() -> Option<String> {
        let listener = match TcpListener::bind("127.0.0.1:0") {
            Ok(listener) => listener,
            Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => return None,
            Err(error) => panic!("failed to bind a local test port: {error}"),
        };
        Some(format!("127.0.0.1:{}", listener.local_addr().unwrap().port()))
    }

    #[test]
    fn test_distributed_key_value_store() {
        let plugin = test_cpu_plugin();
        let address = test_loopback_address().unwrap();
        let service_options = DistributedRuntimeServiceOptions::default();
        let _service = plugin.distributed_runtime_service(&address, service_options).unwrap();
        let client_options = DistributedRuntimeClientOptions::default();
        let client = plugin.distributed_runtime_client(&address, client_options).unwrap();
        let store = DistributedKeyValueStore::new(client);

        // Test using valid keys and values.
        let key = [0_u8, 1, 2, 3, 255];
        let value = [255_u8, 17, 0, 42, 12];
        assert!(store.put(&key, &value).is_ok());
        assert_eq!(store.get(&key, Duration::from_secs(5)).unwrap(), value);
        assert_eq!(store.try_get(&key).unwrap(), value);
        assert!(matches!(
            store.client().blocking_key_value_get("missing", Duration::from_millis(1)),
            Err(Error::NotFound { .. }) | Err(Error::DeadlineExceeded { .. }),
        ));
        assert!(matches!(store.try_get(b"missing"), Err(Error::NotFound { .. })));

        // Test using invalid keys and values.
        assert!(matches!(store.client().key_value_set("invalid\0key", "value"), Err(Error::InvalidArgument { .. })));
        assert!(matches!(store.client().key_value_set("key", "invalid\0value"), Err(Error::InvalidArgument { .. })));
        assert!(matches!(
            store.client().blocking_key_value_get("invalid\0key", Duration::from_secs(5)),
            Err(Error::InvalidArgument { .. }),
        ));
        assert!(matches!(store.client().key_value_try_get("invalid\0key"), Err(Error::InvalidArgument { .. })));
    }

    #[test]
    fn test_distributed_runtime_service() {
        let plugin = test_cpu_plugin();
        let address = test_loopback_address().unwrap();
        let service_options = DistributedRuntimeServiceOptions::default();

        // Test shutting down multiple times.
        let service = plugin.distributed_runtime_service(&address, service_options).unwrap();
        service.shutdown();
        service.shutdown();
        drop(service);

        // Test dropping without shutting down.
        let service = plugin.distributed_runtime_service(&address, service_options).unwrap();
        drop(service);

        // Test constructing invalid [`DistributedRuntimeService`]s.
        assert!(matches!(
            DistributedRuntimeService::from_c_api(std::ptr::null_mut(), plugin.api()),
            Err(Error::InvalidArgument { message, .. })
                if message == "the provided XLA distributed runtime service handle is a null pointer",
        ));
        assert!(matches!(
            plugin.distributed_runtime_service("in\0valid", DistributedRuntimeServiceOptions::default()),
            Err(Error::InvalidArgument { message, .. })
                if message == "XLA distributed runtime service address contains interior NUL bytes",
        ));
    }

    #[test]
    fn test_distributed_runtime_client() {
        let plugin = test_cpu_plugin();
        let address = test_loopback_address().unwrap();
        let service_options = DistributedRuntimeServiceOptions::default();
        let service = plugin.distributed_runtime_service(&address, service_options).unwrap();

        // Test shutting down multiple times.
        let client_options = DistributedRuntimeClientOptions::default();
        let client = plugin.distributed_runtime_client(&address, client_options).unwrap();
        assert!(client.connect().is_ok());
        assert!(client.connect().is_err());
        assert!(client.shutdown().is_ok());
        assert!(client.shutdown().is_ok());
        drop(client);

        // Test dropping without shutting down.
        let client_options = DistributedRuntimeClientOptions::default();
        let client = plugin.distributed_runtime_client(&address, client_options).unwrap();
        assert!(client.connect().is_ok());
        assert!(client.connect().is_err());
        drop(client);

        // Test dropping before connecting.
        let client_options = DistributedRuntimeClientOptions::default();
        let client = plugin.distributed_runtime_client(&address, client_options).unwrap();
        drop(client);

        // Test shutting down before connecting.
        let client_options = DistributedRuntimeClientOptions::default();
        let client = plugin.distributed_runtime_client(&address, client_options).unwrap();
        assert!(client.shutdown().is_ok());
        assert!(client.shutdown().is_ok());
        drop(client);

        // Test using a missed heartbeat callback.
        let callback_counter = Arc::new(AtomicUsize::new(0));
        let callback_invoked = Arc::new(AtomicBool::new(false));
        let callback_counter_clone = Arc::clone(&callback_counter);
        let callback_invoked_clone = Arc::clone(&callback_invoked);
        let client_options = DistributedRuntimeClientOptions {
            heartbeat_timeout: Duration::from_millis(1),
            missed_heartbeat_callback: Some(Box::new(move |_| {
                if callback_invoked_clone.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                    callback_counter_clone.fetch_add(1, Ordering::SeqCst);
                }
            })),
            ..DistributedRuntimeClientOptions::default()
        };
        let client = plugin.distributed_runtime_client(&address, client_options).unwrap();
        assert!(client.connect().is_ok());
        assert!(client.connect().is_err());
        drop(service);
        let start_time = Instant::now();
        while start_time.elapsed() < Duration::from_secs(1) {
            if callback_counter.load(Ordering::SeqCst) >= 1 {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let _ = client.shutdown();
        drop(client);
        assert_eq!(callback_counter.load(Ordering::SeqCst), 1);

        // Test constructing invalid [`DistributedRuntimeClient`]s.
        assert!(matches!(
            DistributedRuntimeClient::from_c_api(std::ptr::null_mut(), None, plugin.api()),
            Err(Error::InvalidArgument { message, .. })
                if message == "the provided XLA distributed runtime client handle is a null pointer",
        ));
        assert!(matches!(
            plugin.distributed_runtime_client("in\0valid", DistributedRuntimeClientOptions::default()),
            Err(Error::InvalidArgument { message, .. })
                if message == "XLA distributed runtime client address contains interior NUL bytes",
        ));
    }

    #[test]
    fn test_duration_to_seconds() {
        assert_eq!(duration_to_seconds(Duration::from_secs(0)), 0);
        assert_eq!(duration_to_seconds(Duration::from_millis(1)), 1);
        assert_eq!(duration_to_seconds(Duration::from_secs(1)), 1);
        assert_eq!(duration_to_seconds(Duration::from_secs(1) + Duration::from_nanos(1)), 2);
        assert_eq!(duration_to_seconds(Duration::from_secs(u64::from(u32::MAX))), u32::MAX);
        assert_eq!(duration_to_seconds(Duration::from_secs(u64::from(u32::MAX)) + Duration::from_secs(1)), u32::MAX);
    }
}
