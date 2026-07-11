#[cfg(any(
    feature = "cuda-12",
    feature = "cuda-13",
    feature = "rocm-7",
    feature = "tpu",
    feature = "neuron",
    feature = "metal",
    feature = "mps",
))]
use std::path::PathBuf;

#[cfg(feature = "cuda-12")]
pub fn pjrt_cuda_12_plugin_path() -> PathBuf {
    PathBuf::from(env!("RYFT_PJRT_PLUGIN_CUDA_12"))
}

#[cfg(feature = "cuda-13")]
pub fn pjrt_cuda_13_plugin_path() -> PathBuf {
    PathBuf::from(env!("RYFT_PJRT_PLUGIN_CUDA_13"))
}

#[cfg(feature = "rocm-7")]
pub fn pjrt_rocm_7_plugin_path() -> PathBuf {
    PathBuf::from(env!("RYFT_PJRT_PLUGIN_ROCM_7"))
}

#[cfg(feature = "tpu")]
pub fn pjrt_tpu_plugin_path() -> PathBuf {
    PathBuf::from(env!("RYFT_PJRT_PLUGIN_TPU"))
}

#[cfg(feature = "neuron")]
pub fn pjrt_neuron_plugin_path() -> PathBuf {
    PathBuf::from(env!("RYFT_PJRT_PLUGIN_NEURON"))
}

#[cfg(feature = "metal")]
pub fn pjrt_metal_plugin_path() -> PathBuf {
    PathBuf::from(env!("RYFT_PJRT_PLUGIN_METAL"))
}

#[cfg(feature = "mps")]
pub fn pjrt_mps_plugin_path() -> PathBuf {
    PathBuf::from(env!("RYFT_PJRT_PLUGIN_MPS"))
}

pub mod bindings;
pub mod distributed;
pub mod mlir;
pub mod profiler;
pub mod protos;
