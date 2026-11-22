use std::fmt::Display;
use std::sync::Mutex;

use ryft_pjrt::{
    Client, ClientOptions, CpuClientOptions, GpuClientOptions, GpuMemoryAllocator, GpuPlatform, Plugin, load_cpu_plugin,
};

use crate::Error;
use crate::sessions::SessionError;

/// Platform that a [`Session`](crate::Session) executes on, which selects the PJRT plugin of the client that the Ryft
/// XLA backend creates for the session. Use [`SessionBuilder::platform`](crate::SessionBuilder::platform) to select a
/// specific platform. Each accelerator platform requires the Cargo feature that provides its plugin. Building the
/// session reports a missing feature, so code that selects a platform can do so without conditional compilation.
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

    /// Apple GPUs through the MPS plugin, which requires the `mps` feature. The plugin has a single device and does
    /// not support the Shardy dialect, so programs are compiled for it without sharding information and `shard_map`
    /// regions are rejected.
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
    const AUTO_PLATFORM_CANDIDATES: [Self; 6] =
        [Self::Cuda, Self::Rocm, Self::Tpu, Self::Neuron, Self::Metal, Self::Mps];

    /// Returns whether the feature of this crate that provides the plugin of this platform is enabled.
    /// [`Self::Auto`] and [`Self::Cpu`] are always available.
    #[inline]
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

    /// Returns the process-wide client of this platform for `options`, or for the default options of the platform if
    /// `options` is [`None`]. Calls for the same plugin and equal options share one client for the rest of the process.
    ///
    /// [`Self::Auto`] returns the client of the first accelerator, in order of preference, whose feature is enabled,
    /// whose plugin loads and initializes, and whose client has at least one addressable device, and the CPU client
    /// otherwise. Accelerators that fail any of these steps are skipped without reporting their failure.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError::UnavailablePlatform`] if the feature that provides the plugin of an explicitly
    /// requested platform is not enabled, [`SessionError::InvalidClientOptions`] if `options` are supplied for
    /// [`Self::Auto`], whose candidate platforms take different options, and [`Error::Pjrt`] if the plugin cannot
    /// be loaded or initialized or its client cannot be created.
    pub fn client(self, options: Option<ClientOptions>) -> Result<&'static Client<'static>, Error> {
        if self != Self::Auto {
            return client_for_plugin(&self.plugin()?, options.unwrap_or_else(|| self.client_options()));
        }

        if options.is_some() {
            return Err(SessionError::InvalidClientOptions {
                reason: "client options require an explicit platform or plugin, because the platforms that \
                         `Platform::Auto` selects among take different options"
                    .to_string(),
            }
            .into());
        }

        for accelerator in Self::AUTO_PLATFORM_CANDIDATES.into_iter().filter(|accelerator| accelerator.is_available()) {
            let client = accelerator.client(None);
            if let Ok(client) = client
                && client.addressable_devices().is_ok_and(|devices| !devices.is_empty())
            {
                return Ok(client);
            }
        }

        Self::Cpu.client(None)
    }

    /// Loads the PJRT [`Plugin`] that corresponds to this platform. [`Self::Auto`] has no plugin of its own,
    /// and, in that case, [`Self::client`] resolves it to one of the other platforms instead.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError::UnavailablePlatform`] if the feature that provides the plugin is not enabled,
    /// and [`Error::Pjrt`] if the plugin cannot be loaded.
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

    /// Returns the [`ClientOptions`] of the client that the Ryft XLA backend creates for this [`Platform`] when a
    /// [`Session`](crate::Session) supplies none. These are the defaults of `ryft-pjrt` for the platform, except that
    /// the allocator of a CUDA client grows on demand instead of reserving most of the device memory up front. The
    /// client is shared by every session of the process and is never destroyed, and on devices whose memory is shared
    /// with the host an up-front reservation would also take that memory from the rest of the system. The allocator is
    /// still the best-fit one, and not [`GpuMemoryAllocator::CudaAsync`], which asks the driver for every buffer. That
    /// costs about 100 microseconds per eager operation and made an eager forward pass of a 1B-parameter Transformer
    /// about five times slower.
    fn client_options(self) -> ClientOptions {
        match self {
            Self::Cpu => ClientOptions::CPU(CpuClientOptions::default()),
            Self::Cuda => ClientOptions::GPU(GpuClientOptions {
                platform: Some(GpuPlatform::CUDA),
                allocator: GpuMemoryAllocator::BFC { memory_fraction: 0.75, preallocate: false },
                ..Default::default()
            }),
            Self::Rocm => {
                ClientOptions::GPU(GpuClientOptions { platform: Some(GpuPlatform::ROCm), ..Default::default() })
            }
            Self::Auto | Self::Tpu | Self::Neuron | Self::Metal | Self::Mps => ClientOptions::default(),
        }
    }
}

impl Display for Platform {
    #[inline]
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

/// Client that the Ryft XLA backend created for sessions and that lives for the rest of the process.
struct ProcessClient {
    /// Plugin that created the client.
    plugin: Plugin,

    /// Options that the client was created with.
    options: ClientOptions,

    /// Client, which is never destroyed.
    client: &'static Client<'static>,
}

/// Clients that the Ryft XLA backend created for sessions, in creation order. Clients are never removed.
static PROCESS_CLIENTS: Mutex<Vec<ProcessClient>> = Mutex::new(Vec::new());

/// Returns the process-wide client of `plugin` for `options`, creating it on first use. The client lives for the rest
/// of the process, and every later request for the same plugin with equal options returns it, so that sessions built
/// independently of each other share one client and its device resources. On accelerators, sharing avoids separate
/// clients reserving device memory independently. Requests are serialized, so concurrent first requests for one
/// configuration create a single client.
///
/// # Errors
///
/// Returns [`Error::Pjrt`] if the plugin cannot be initialized or the client cannot be created. Neither failure is
/// retained by this function, so a later request attempts to create the client again.
pub(super) fn client_for_plugin(plugin: &Plugin, options: ClientOptions) -> Result<&'static Client<'static>, Error> {
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
    fn test_client_for_plugin() {
        // Clients are keyed by their plugin and options, irrespective of the platform that requested them.
        let plugin = load_cpu_plugin().unwrap();
        let options = ClientOptions::CPU(CpuClientOptions { device_count: Some(5), ..Default::default() });
        let client = client_for_plugin(&plugin, options.clone()).unwrap();
        assert_eq!(client.addressable_devices().unwrap().len(), 5);
        assert!(std::ptr::eq(client_for_plugin(&plugin, options.clone()).unwrap(), client));
        assert!(std::ptr::eq(Platform::Cpu.client(Some(options)).unwrap(), client));
    }
}
