//! Platforms that sessions execute on, and the PJRT clients that `ryft-xla` creates for them.
//!
//! [`Platform`] names the platforms whose PJRT plugins `ryft-xla` can load, and a
//! [`SessionBuilder`](crate::SessionBuilder) selects one of them. A client that `ryft-xla` creates for a session lives
//! for the rest of the process, like the backend clients of JAX, and sessions that request the same plugin with equal
//! client options share it. Sharing matters most on accelerators, where a second client in the same process would
//! reserve device memory of its own. Programs that must destroy their client, or that need a client that `ryft-xla`
//! cannot create (e.g., one bound to a [`DistributedRuntime`](crate::DistributedRuntime)), create it themselves and
//! pass it to [`SessionBuilder::client`](crate::SessionBuilder::client).

// TODO(eaplatanios): Review this module.

use std::fmt::Display;
use std::sync::Mutex;

use ryft_pjrt::{
    Client, ClientOptions, CpuClientOptions, GpuClientOptions, GpuMemoryAllocator, GpuPlatform, Plugin, load_cpu_plugin,
};

use crate::Error;
use crate::domains::sessions::SessionError;

/// Platform that a [`Session`](crate::Session) executes on, which selects the PJRT plugin of the client that
/// `ryft-xla` creates for the session. Every platform other than [`Self::Cpu`] requires the Cargo feature of
/// `ryft-xla` that provides its plugin, and requesting a platform without its feature is reported when the session is
/// built, so code that selects a platform needs no conditional compilation.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Platform {
    /// First accelerator among [`Self::Cuda`], [`Self::Rocm`], [`Self::Tpu`], [`Self::Neuron`], [`Self::Metal`], and
    /// [`Self::Mps`], in that order, whose feature is enabled and whose client has at least one addressable device,
    /// and [`Self::Cpu`] otherwise. [`Target::platform_name`](crate::Target::platform_name) reports the selection.
    #[default]
    Auto,

    /// Built-in CPU platform, which is always available.
    Cpu,

    /// NVIDIA GPUs through CUDA, which requires the `cuda-13` or the `cuda-12` feature. CUDA 13 is preferred when both
    /// features are enabled.
    Cuda,

    /// AMD GPUs through ROCm, which requires the `rocm-10` feature.
    Rocm,

    /// Google TPUs, which require the `tpu` feature.
    Tpu,

    /// AWS Neuron accelerators, which require the `neuron` feature.
    Neuron,

    /// Apple GPUs through the Metal plugin, which requires the `metal` feature.
    Metal,

    /// Apple GPUs through the MPS plugin, which requires the `mps` feature.
    Mps,
}

/// Loads the plugin that `$load` names if the feature `$feature` of this crate is enabled, and otherwise reports that
/// `$platform` is unavailable without that feature.
macro_rules! load_platform_plugin {
    ($platform:expr, $feature:literal, $load:ident) => {{
        #[cfg(feature = $feature)]
        {
            Ok(ryft_pjrt::$load()?)
        }
        #[cfg(not(feature = $feature))]
        {
            Err(SessionError::UnavailablePlatform { platform: $platform, feature: $feature }.into())
        }
    }};
}

impl Platform {
    /// Accelerator platforms that [`Self::Auto`] considers, in order of preference.
    const ACCELERATORS: [Self; 6] = [Self::Cuda, Self::Rocm, Self::Tpu, Self::Neuron, Self::Metal, Self::Mps];

    /// Returns whether the feature of this crate that provides the plugin of this platform is enabled.
    /// [`Self::Auto`] and [`Self::Cpu`] are always available.
    pub fn is_available(self) -> bool {
        match self {
            Self::Auto | Self::Cpu => true,
            Self::Cuda => cfg!(any(feature = "cuda-13", feature = "cuda-12")),
            Self::Rocm => cfg!(feature = "rocm-10"),
            Self::Tpu => cfg!(feature = "tpu"),
            Self::Neuron => cfg!(feature = "neuron"),
            Self::Metal => cfg!(feature = "metal"),
            Self::Mps => cfg!(feature = "mps"),
        }
    }

    /// Loads the PJRT plugin of this platform. [`Self::Auto`] has no plugin of its own, and [`Self::client`] resolves
    /// it to one of the other platforms instead.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError::UnavailablePlatform`] if the feature that provides the plugin is not enabled, and
    /// [`Error::Pjrt`] if the plugin cannot be loaded.
    fn plugin(self) -> Result<Plugin, Error> {
        match self {
            Self::Auto | Self::Cpu => Ok(load_cpu_plugin()?),
            Self::Cuda => {
                #[cfg(feature = "cuda-13")]
                {
                    Ok(ryft_pjrt::load_cuda_13_plugin()?)
                }
                #[cfg(all(feature = "cuda-12", not(feature = "cuda-13")))]
                {
                    Ok(ryft_pjrt::load_cuda_12_plugin()?)
                }
                #[cfg(not(any(feature = "cuda-13", feature = "cuda-12")))]
                {
                    Err(SessionError::UnavailablePlatform { platform: self, feature: "cuda-13" }.into())
                }
            }
            Self::Rocm => load_platform_plugin!(self, "rocm-10", load_rocm_10_plugin),
            Self::Tpu => load_platform_plugin!(self, "tpu", load_tpu_plugin),
            Self::Neuron => load_platform_plugin!(self, "neuron", load_neuron_plugin),
            Self::Metal => load_platform_plugin!(self, "metal", load_metal_plugin),
            Self::Mps => load_platform_plugin!(self, "mps", load_mps_plugin),
        }
    }

    /// Returns the options of the client that `ryft-xla` creates for this platform when a session supplies none.
    /// These are the defaults of `ryft-pjrt` for the platform, except that a CUDA client allocates device memory on
    /// demand instead of reserving most of it up front. The client is shared by every session of the process and is
    /// never destroyed, and on devices whose memory is shared with the host an up-front reservation would also take
    /// that memory from the rest of the system.
    fn client_options(self) -> ClientOptions {
        match self {
            Self::Cpu => ClientOptions::CPU(CpuClientOptions::default()),
            Self::Cuda => ClientOptions::GPU(GpuClientOptions {
                platform: Some(GpuPlatform::CUDA),
                allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                ..Default::default()
            }),
            Self::Rocm => {
                ClientOptions::GPU(GpuClientOptions { platform: Some(GpuPlatform::ROCm), ..Default::default() })
            }
            Self::Auto | Self::Tpu | Self::Neuron | Self::Metal | Self::Mps => ClientOptions::default(),
        }
    }

    /// Returns the process-wide client of this platform for `options`, or for the default options of the platform if
    /// `options` is [`None`]. Refer to [`process_client`] for information on how such clients are shared.
    ///
    /// [`Self::Auto`] returns the client of the first accelerator, in order of preference, whose feature is enabled,
    /// whose plugin loads and initializes, and whose client has at least one addressable device, and the CPU client
    /// otherwise. Accelerators that fail any of these steps are skipped without reporting their failure.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError::UnavailablePlatform`] if the feature that provides the plugin of an explicitly
    /// requested platform is not enabled, [`SessionError::InvalidClientOptions`] if `options` are supplied for
    /// [`Self::Auto`], whose candidate platforms take different options, and [`Error::Pjrt`] if the plugin cannot be
    /// loaded or initialized or its client cannot be created.
    pub(super) fn client(self, options: Option<ClientOptions>) -> Result<&'static Client<'static>, Error> {
        if self != Self::Auto {
            return process_client(&self.plugin()?, options.unwrap_or_else(|| self.client_options()));
        }
        if options.is_some() {
            return Err(SessionError::InvalidClientOptions {
                reason: "client options require an explicit platform or plugin, because the platforms that \
                         `Platform::Auto` selects among take different options"
                    .to_string(),
            }
            .into());
        }
        for accelerator in Self::ACCELERATORS.into_iter().filter(|accelerator| accelerator.is_available()) {
            let client = accelerator.plugin().and_then(|plugin| process_client(&plugin, accelerator.client_options()));
            if let Ok(client) = client
                && client.addressable_devices().is_ok_and(|devices| !devices.is_empty())
            {
                return Ok(client);
            }
        }
        Self::Cpu.client(None)
    }
}

impl Display for Platform {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Rocm => "rocm",
            Self::Tpu => "tpu",
            Self::Neuron => "neuron",
            Self::Metal => "metal",
            Self::Mps => "mps",
        })
    }
}

/// Client that `ryft-xla` created for sessions and that lives for the rest of the process.
struct ProcessClient {
    /// Plugin that created the client.
    plugin: Plugin,

    /// Options that the client was created with.
    options: ClientOptions,

    /// Client, which is never destroyed.
    client: &'static Client<'static>,
}

/// Clients that `ryft-xla` created for sessions, in creation order. Clients are never removed.
static PROCESS_CLIENTS: Mutex<Vec<ProcessClient>> = Mutex::new(Vec::new());

/// Returns the process-wide client of `plugin` for `options`, creating it on first use. The client lives for the rest
/// of the process, and every later request for the same plugin with equal options returns it, so that sessions built
/// independently of each other share one client and its device resources. Requests are serialized, so concurrent
/// first requests for one configuration create a single client.
///
/// # Errors
///
/// Returns [`Error::Pjrt`] if the plugin cannot be initialized or the client cannot be created. Neither failure is
/// retained by this function, so a later request attempts to create the client again.
pub(super) fn process_client(plugin: &Plugin, options: ClientOptions) -> Result<&'static Client<'static>, Error> {
    let mut clients = PROCESS_CLIENTS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(existing) = clients.iter().find(|existing| existing.plugin == *plugin && existing.options == options) {
        return Ok(existing.client);
    }
    // Plugin initialization fails when the runtime of an accelerator is unavailable, which is reported here rather
    // than left to panic on the first use of the plugin.
    plugin.initialize()?;
    let client: &'static Client<'static> = Box::leak(Box::new(plugin.client(options.clone())?));
    clients.push(ProcessClient { plugin: plugin.clone(), options, client });
    Ok(client)
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_platform_is_available() {
        assert!(Platform::Auto.is_available());
        assert!(Platform::Cpu.is_available());
        assert_eq!(Platform::Cuda.is_available(), cfg!(any(feature = "cuda-13", feature = "cuda-12")));
        assert_eq!(Platform::Rocm.is_available(), cfg!(feature = "rocm-10"));
        assert_eq!(Platform::Tpu.is_available(), cfg!(feature = "tpu"));
    }

    #[test]
    fn test_platform_client() {
        // Requests for one platform with equal options share one client, while other options create another one.
        let default_client = Platform::Cpu.client(None).unwrap();
        assert!(default_client.platform_name().unwrap().eq_ignore_ascii_case("cpu"));
        assert!(std::ptr::eq(Platform::Cpu.client(None).unwrap(), default_client));
        let options = ClientOptions::CPU(CpuClientOptions { device_count: Some(3), ..Default::default() });
        let client = Platform::Cpu.client(Some(options.clone())).unwrap();
        assert_eq!(client.addressable_devices().unwrap().len(), 3);
        assert!(!std::ptr::eq(client, default_client));
        assert!(std::ptr::eq(Platform::Cpu.client(Some(options)).unwrap(), client));

        // Automatic selection always finds a client with a device, and rejects options that only fit one platform.
        assert!(!Platform::Auto.client(None).unwrap().addressable_devices().unwrap().is_empty());
        assert!(matches!(
            Platform::Auto.client(Some(ClientOptions::default())),
            Err(Error::Session(SessionError::InvalidClientOptions { .. })),
        ));
    }

    #[cfg(not(feature = "tpu"))]
    #[test]
    fn test_platform_client_rejects_platforms_without_their_feature() {
        let Err(error) = Platform::Tpu.client(None) else {
            panic!("a platform without its feature unexpectedly produced a client");
        };
        assert!(matches!(
            &error,
            Error::Session(SessionError::UnavailablePlatform { platform: Platform::Tpu, feature: "tpu" }),
        ));
        assert_eq!(
            error.to_string(),
            "platform `tpu` is unavailable because `ryft-xla` was built without its `tpu` feature",
        );
    }

    #[test]
    fn test_platform_display() {
        assert_eq!(Platform::Auto.to_string(), "auto");
        assert_eq!(Platform::Cpu.to_string(), "cpu");
        assert_eq!(Platform::Cuda.to_string(), "cuda");
        assert_eq!(Platform::Rocm.to_string(), "rocm");
        assert_eq!(Platform::Tpu.to_string(), "tpu");
        assert_eq!(Platform::Neuron.to_string(), "neuron");
        assert_eq!(Platform::Metal.to_string(), "metal");
        assert_eq!(Platform::Mps.to_string(), "mps");
    }

    #[test]
    fn test_process_client() {
        // Clients are keyed by their plugin and options, irrespective of the platform that requested them.
        let plugin = load_cpu_plugin().unwrap();
        let options = ClientOptions::CPU(CpuClientOptions { device_count: Some(5), ..Default::default() });
        let client = process_client(&plugin, options.clone()).unwrap();
        assert_eq!(client.addressable_devices().unwrap().len(), 5);
        assert!(std::ptr::eq(process_client(&plugin, options.clone()).unwrap(), client));
        assert!(std::ptr::eq(Platform::Cpu.client(Some(options)).unwrap(), client));
    }
}
