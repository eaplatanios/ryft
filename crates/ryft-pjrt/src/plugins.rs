use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex, Once, OnceLock};

use libloading::Library;

use crate::{Api, Client, ClientOptions, Error, KeyValueStore, invoke_pjrt_api_error_fn};

/// Loaded PJRT [`Plugin`] that can be used via its [`Plugin::api`].
#[derive(Clone)]
pub struct Plugin {
    /// PJRT [`Api`] for this [`Plugin`].
    api: Api,

    /// Shared [`Once`] used to initialize this [`Plugin`] exactly once, even if it is cloned.
    initialization: Arc<Once>,
}

impl Plugin {
    /// Returns the initialized PJRT [`Api`] for this [`Plugin`].
    pub(crate) fn api(&self) -> Api {
        self.initialization.call_once(|| {
            use ffi::PJRT_Plugin_Initialize_Args;
            invoke_pjrt_api_error_fn!(self.api, PJRT_Plugin_Initialize).expect("PJRT plugin initialization failed");
        });
        self.api
    }

    /// Constructs a new PJRT [`Client`] using the provided (optional) platform-specific [`ClientOptions`].
    ///
    /// Note that the resulting [`Client`] will not have access to a [`KeyValueStore`] and thus will have no direct way
    /// to interact with other [`Client`]s. Refer to [`Plugin::client_with_key_value_store`] for more information.
    pub fn client(&self, options: ClientOptions) -> Result<Client<'static>, Error> {
        self.api().client(options)
    }

    /// Constructs a new PJRT [`Client`] using the provided (optional) platform-specific [`ClientOptions`] and
    /// [`KeyValueStore`]. The provided [`KeyValueStore`] must be accessible across multiple hosts and/or processes.
    /// Access to this [`KeyValueStore`] may be necessary to create certain kinds of multi-process or multi-host
    /// environments as it enables [`Client`]s (potentially on different machines) to communicate with each other.
    pub fn client_with_key_value_store<'s, Store: KeyValueStore>(
        &self,
        options: ClientOptions,
        key_value_store: &'s Store,
    ) -> Result<Client<'s>, Error> {
        self.api().client_with_key_value_store(options, key_value_store)
    }
}

/// Loaded shared [`Library`] that contains a PJRT [`Plugin`].
struct PluginLibrary {
    /// Shared [`Library`] that contains a PJRT [`Plugin`].
    library: Library,

    /// [`PathBuf`] pointing to the shared library file from which [`PluginLibrary::library`] was loaded.
    path: PathBuf,

    /// Cached loaded [`Plugin`] to avoid loading the same plugin multiple times.
    plugin: OnceLock<Result<Plugin, Error>>,
}

impl PluginLibrary {
    /// Loads the PJRT [`Plugin`] that is stored in this [`PluginLibrary`]. The underlying shared library must export
    /// a symbol called `GetPjrtApi` with a function signature matching `unsafe extern "C" fn() -> *const PJRT_Api`.
    /// This function loads that symbol at most once and then reuses it in future calls.
    fn load(&self) -> Result<Plugin, Error> {
        self.plugin
            .get_or_init(|| {
                let get_pjrt_api_function = unsafe {
                    self.library
                        .get::<unsafe extern "C" fn() -> *const crate::ffi::PJRT_Api>(b"GetPjrtApi")
                        .map_err(|error| Error::plugin_loading_error(format!("{:?}", self.path), error.to_string()))?
                };
                let api = unsafe { Api::from_c_api(get_pjrt_api_function()) }?;
                Ok(Plugin { api, initialization: Arc::new(Once::new()) })
            })
            .clone()
    }
}

/// Internal helper struct used for managing PJRT [`Plugin`]s that are loaded from shared libraries at runtime.
struct PluginManager {
    /// Thread-safe [`HashMap`] mapping [`PathBuf`]s pointing to PJRT [`Plugin`] shared libraries to the corresponding
    /// loaded [`PluginLibrary`] instances. Those instances can be used to load the underlying [`Plugin`]s.
    plugins: Arc<Mutex<HashMap<PathBuf, PluginLibrary>>>,
}

impl PluginManager {
    /// Loads a PJRT [`Plugin`] from the provided [`Path`] pointing to the shared library for that plugin.
    fn load_plugin(&self, library_path: &Path) -> Result<Plugin, Error> {
        let library_path = std::fs::canonicalize(library_path).unwrap_or_else(|_| library_path.to_path_buf());
        let mut plugins = self.plugins.lock().unwrap();
        match plugins.entry(library_path.clone()) {
            Entry::Occupied(entry) => entry.get().load(),
            Entry::Vacant(entry) => {
                let library = unsafe { Library::new(&library_path) }.map_err(|error| {
                    Error::plugin_loading_error(format!("{}", library_path.display()), format!("{:?}", error))
                })?;
                let plugin = PluginLibrary { library, path: library_path, plugin: OnceLock::new() };
                entry.insert(plugin).load()
            }
        }
    }
}

/// Static [`PluginManager`] that is used for loading and caching PJRT [`Plugin`]s.
static PLUGIN_MANAGER: LazyLock<PluginManager> =
    LazyLock::new(|| PluginManager { plugins: Arc::new(Mutex::new(HashMap::new())) });

/// Shared built-in CPU [`Plugin`]. The reason we need this static variable is that loading the built-in CPU
/// plugin does not go through a [`PluginManager`], and that is where [`Plugin`] caching happens for all other plugins.
static CPU_PLUGIN: LazyLock<Plugin> = LazyLock::new(|| {
    let api = unsafe { Api::from_c_api(ryft_xla_sys::bindings::GetPjrtApi() as *const crate::ffi::PJRT_Api) }
        .expect("failed to load the built-in PJRT CPU API");
    Plugin { api, initialization: Arc::new(Once::new()) }
});

/// Loads a PJRT [`Plugin`] from the provided [`Path`] pointing to the shared library for that plugin.
pub fn load_plugin(library_path: &Path) -> Result<Plugin, Error> {
    PLUGIN_MANAGER.load_plugin(library_path)
}

/// Loads the built-in CPU-only PJRT [`Plugin`] that is backed by [XLA](https://openxla.org/xla) from [`ryft_xla_sys`].
pub fn load_cpu_plugin() -> Result<Plugin, Error> {
    Ok(CPU_PLUGIN.clone())
}

/// Loads the [CUDA 12](https://docs.nvidia.com/cuda/) PJRT [`Plugin`] that is backed by [XLA](https://openxla.org/xla)
/// from [`ryft_xla_sys`].
#[cfg(feature = "cuda-12")]
pub fn load_cuda_12_plugin() -> Result<Plugin, Error> {
    load_plugin(&ryft_xla_sys::pjrt_cuda_12_plugin_path())
}

/// Loads the [CUDA 13](https://docs.nvidia.com/cuda/) PJRT [`Plugin`] that is backed by [XLA](https://openxla.org/xla)
/// from [`ryft_xla_sys`].
#[cfg(feature = "cuda-13")]
pub fn load_cuda_13_plugin() -> Result<Plugin, Error> {
    load_plugin(&ryft_xla_sys::pjrt_cuda_13_plugin_path())
}

/// Loads the [ROCm 7](https://rocm.docs.amd.com/) PJRT [`Plugin`] that is backed by [XLA](https://openxla.org/xla)
/// from [`ryft_xla_sys`].
#[cfg(feature = "rocm-7")]
pub fn load_rocm_7_plugin() -> Result<Plugin, Error> {
    load_plugin(&ryft_xla_sys::pjrt_rocm_7_plugin_path())
}

/// Loads the [TPU](https://cloud.google.com/tpu) PJRT [`Plugin`] that is backed by [XLA](https://openxla.org/xla)
/// from [`ryft_xla_sys`].
#[cfg(feature = "tpu")]
pub fn load_tpu_plugin() -> Result<Plugin, Error> {
    load_plugin(&ryft_xla_sys::pjrt_tpu_plugin_path())
}

/// Loads the [AWS Neuron](https://awsdocs-neuron.readthedocs-hosted.com/) PJRT [`Plugin`] that is backed by
/// [XLA](https://openxla.org/xla) from [`ryft_xla_sys`].
#[cfg(feature = "neuron")]
pub fn load_neuron_plugin() -> Result<Plugin, Error> {
    load_plugin(&ryft_xla_sys::pjrt_neuron_plugin_path())
}

/// Loads the [Metal](https://developer.apple.com/metal/jax/) PJRT [`Plugin`] that is backed by
/// [XLA](https://openxla.org/xla) from [`ryft_xla_sys`].
#[cfg(feature = "metal")]
pub fn load_metal_plugin() -> Result<Plugin, Error> {
    load_plugin(&ryft_xla_sys::pjrt_metal_plugin_path())
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use crate::errors::ffi::PJRT_Error;
    use crate::ffi::PJRT_Extension_Base;
    use crate::values::ffi::PJRT_NamedValue;

    #[repr(C)]
    pub struct PJRT_Plugin_Initialize_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
    }

    impl PJRT_Plugin_Initialize_Args {
        pub fn new() -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut() }
        }
    }

    pub type PJRT_Plugin_Initialize = unsafe extern "C" fn(args: *mut PJRT_Plugin_Initialize_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Plugin_Attributes_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub attributes: *const PJRT_NamedValue,
        pub num_attributes: usize,
    }

    impl PJRT_Plugin_Attributes_Args {
        pub fn new() -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                attributes: std::ptr::null_mut(),
                num_attributes: 0,
            }
        }
    }

    pub type PJRT_Plugin_Attributes = unsafe extern "C" fn(args: *mut PJRT_Plugin_Attributes_Args) -> *mut PJRT_Error;
}

#[cfg(test)]
mod tests {
    use crate::tests::test_for_each_platform;

    #[test]
    fn test_load_plugins() {
        // We simply need to verify that the plugin loads for each platform and so we pass an empty body to the macro.
        test_for_each_platform!(|_plugin, _client, _platform| {});
    }
}
