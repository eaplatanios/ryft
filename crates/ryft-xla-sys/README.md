# Ryft Bindings for XLA

This library contains low-level Rust bindings for components of the [XLA compiler](https://openxla.org/xla)
that are used by the `ryft` framework. Specifically, it contains bindings for:

- [MLIR](https://mlir.llvm.org/), including the [StableHLO dialect](https://openxla.org/stablehlo) that is used by XLA.
  The `ryft-mlir` crate provides a higher-level safe API for interacting with the MLIR bindings.
- [PJRT](https://openxla.org/xla/pjrt), including support for a built-in CPU-only plugin and support for loading
  arbitrary PJRT plugins. The `ryft-pjrt` crate provides a higher-level safe API for interacting with the PJRT bindings.
- [Protobuf](https://protobuf.dev/) message definitions that are used by XLA and which are often used when interacting
  with PJRT plugins.
- Bindings for the distributed runtime service and client that is provided by XLA and which is used by
  [JAX](https://docs.jax.dev/en/latest/).

This is a `-sys` crate, meaning that it handles locating native artifacts (in this case also potentially donwloading
them, as explained further down) and wiring the linker, and it is not intended to be used directly. Instead, users are
advised to use the higher-level safe APIs provided by the `ryft-mlir` and `ryft-pjrt` crates.

## Dependencies

This crate depends on a static XLA library that it is linked to at built time. This library can be built from source
using Bazel (though that is not supported when using `cargo vendor` since it requires network access) or it can be
provided as a precompiled archive using the `RYFT_XLA_SYS_ARCHIVE` environment variable. Note that, by default this
crate will attempt to download a precompiled archive from GitHub releases of `ryft`, if one can be found for the target
platform and `RYFT_XLA_SYS_ARCHIVE` is not set.

Furthermore, this crate has optional features for loading PJRT plugins for various kinds of accelerators, each having
potentially different build time and runtime requirements:

- **`cuda-12`:** Enables support for loading the PJRT [CUDA 12](https://docs.nvidia.com/cuda/) plugin for leveraging
  CUDA-enabled GPUs by Nvidia. If this feature is enabled, similar to what happens with the static XLA library,
  the corresponding PJRT plugin can be built from source using Bazel (though that is not supported when using
  `cargo vendor` since it requires network access) or it can be provided as a precompiled archive using the
  `PJRT_PLUGIN_CUDA_12_LIB` environment variable. Note that, by default this crate will attempt to download
  a precompiled plugin from GitHub releases of `ryft`, if one can be found for the target platform and
  `PJRT_PLUGIN_CUDA_12_LIB` is not set. Note that this plugin has various runtime dependencies that are not included
  in the shared library provided by this feature, including but not limited to: `cublas`, `cudart`, `cudnn`, `cufft`,
  `cupti`, `cusolver`, `cusparse`, `nccl`, `nvjitlink`, `nvptxcompiler`, `nvrtc`, and `nvshmem`.

- **`cuda-13`:** Enables support for loading the PJRT [CUDA 13](https://docs.nvidia.com/cuda/) plugin for leveraging
  CUDA-enabled GPUs by Nvidia. If this feature is enabled, similar to what happens with the static XLA library,
  the corresponding PJRT plugin can be built from source using Bazel (though that is not supported when using
  `cargo vendor` since it requires network access) or it can be provided as a precompiled archive using the
  `PJRT_PLUGIN_CUDA_13_LIB` environment variable. Note that, by default this crate will attempt to download
  a precompiled plugin from GitHub releases of `ryft`, if one can be found for the target platform and
  `PJRT_PLUGIN_CUDA_13_LIB` is not set. Note that this plugin has various runtime dependencies that are not included
  in the shared library provided by this feature, including but not limited to: `cublas`, `cudart`, `cudnn`, `cufft`,
  `cupti`, `cusolver`, `cusparse`, `nccl`, `nvjitlink`, `nvptxcompiler`, `nvrtc`, and `nvshmem`.

- **`rocm-7`:** Enables support for loading the PJRT [ROCm 7](https://rocm.docs.amd.com/) plugin for leveraging
  ROCm-enabled GPUs by AMD. If this feature is enabled, similar to what happens with the static XLA library,
  the corresponding PJRT plugin can be built from source using Bazel (though that is not supported when using
  `cargo vendor` since it requires network access) or it can be provided as a precompiled archive using the
  `PJRT_PLUGIN_ROCM_7_LIB` environment variable. Note that, by default this crate will attempt to download
  a precompiled plugin from GitHub releases of `ryft`, if one can be found for the target platform and
  `PJRT_PLUGIN_ROCM_7_LIB` is not set. Note that this plugin has various ROCm runtime dependencies that are not included
  in the shared library provided by this feature.

- **`tpu`:** Enables support for loading the PJRT [TPU](https://cloud.google.com/tpu) plugin for leveraging
  TPUs by Google. If this feature is enabled, this crate will attempt to download a precompiled plugin provided
  by Google, if one can be found for the target platform and `PJRT_PLUGIN_TPU_LIB` is not set. Similar to the other
  feature flags, `PJRT_PLUGIN_TPU_LIB` can be used to provide a path to the precompiled plugin to avoid downloading it.
  Note that, in contrast to some of the other PJRT plugins, the TPU plugin is closed source and thus cannot be built
  from source.

- **`neuron`:** Enables support for loading the PJRT [AWS Neuron](https://awsdocs-neuron.readthedocs-hosted.com/) plugin
  for leveraging Inferentia and Trainium accelerators by Amazon. If this feature is enabled, this crate will attempt to
  download a precompiled plugin provided by Amazon, if one can be found for the target platform and
  `PJRT_PLUGIN_NEURON_LIB` is not set. Similar to the other feature flags, `PJRT_PLUGIN_NEURON_LIB` can be used to
  provide a path to the precompiled plugin to avoid downloading it. Note that, in contrast to some of the other PJRT
  plugins, the AWS Neuron plugin is closed source and thus cannot be built from source.

Note that, in cases where precompiled binaries are downloaded, the build script of this crate will verify their SHA-256
checksums and will make sure to cache them for future builds.

Note that for offline builds (e.g., when using `cargo vendor`), you must use the aforementioned environment variables to
provide precompiled binaries that you have already downloaded ahead of time.

### Precompiled Artifacts and Supported Platforms

Currently, precompiled binaries are only available for the following target platforms:

- **`ryft-xla-sys` Static Library:**
    - Linux `x86_64`
    - MacOS `aarch64`
    - Windows `x86_64`
- **PJRT Plugins for CUDA 12 & 13, ROCm 7, TPUs, and AWS Neuron:**
    - Linux `x86_64`

## Contributing

When upgrading the XLA commit that this crate depends on, you must always update the following functions in `build.rs`,
which control how the precompiled binaries are named, where they are downloaded from, and how their integrity is
verified:

- `BuildConfiguration::precompiled_artifact_name`
- `BuildConfiguration::precompiled_artifact_url_prefix`
- `BuildConfiguration::precompiled_artifact_checksum`

Furthermore, you can use the `generate-bindings` feature to regenerate the C API bindings from the XLA source code
using `bindgen` and copy them over to the `src/bindings.rs` file.
