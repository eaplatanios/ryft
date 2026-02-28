# **Ryft XLA-SYS:** Rust Bindings for XLA

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
  `cupti`, `cusolver`, `cusparse`, `nccl`, `nvjitlink`, `nvptxcompiler`, `nvrtc`, and `nvshmem`. For Ubuntu 24.04 on
  x86-64, you can install these dependencies using the following commands:

  ```bash
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt update -y
  sudo apt install cuda-toolkit-12-8 \
    libcublas-12-8 \
    cuda-cudart-12-8 \
    cuda-nvrtc-12-8 \
    libcufft-12-8 \
    libnvjitlink-12-8 \
    libcudnn9-cuda-12 \
    libcudnn9-dev-cuda-12
  ```

- **`cuda-13`:** Enables support for loading the PJRT [CUDA 13](https://docs.nvidia.com/cuda/) plugin for leveraging
  CUDA-enabled GPUs by Nvidia. If this feature is enabled, similar to what happens with the static XLA library,
  the corresponding PJRT plugin can be built from source using Bazel (though that is not supported when using
  `cargo vendor` since it requires network access) or it can be provided as a precompiled archive using the
  `PJRT_PLUGIN_CUDA_13_LIB` environment variable. Note that, by default this crate will attempt to download
  a precompiled plugin from GitHub releases of `ryft`, if one can be found for the target platform and
  `PJRT_PLUGIN_CUDA_13_LIB` is not set. Note that this plugin has various runtime dependencies that are not included
  in the shared library provided by this feature, including but not limited to: `cublas`, `cudart`, `cudnn`, `cufft`,
  `cupti`, `cusolver`, `cusparse`, `nccl`, `nvjitlink`, `nvptxcompiler`, `nvrtc`, and `nvshmem`. For Ubuntu 24.04 on
  x86-64, you can install these dependencies using the following commands:

  ```bash
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt update -y
  sudo apt install cuda-toolkit-13-0 \
    libcublas-13-0 \
    cuda-cudart-13-0 \
    cuda-nvrtc-13-0 \
    libcufft-13-0 \
    libnvjitlink-13-0 \
    libcudnn9-cuda-13 \
    libcudnn9-dev-cuda-13
  ```

- **`rocm-7`:** Enables support for loading the PJRT [ROCm 7](https://rocm.docs.amd.com/) plugin for leveraging
  ROCm-enabled GPUs by AMD. If this feature is enabled, similar to what happens with the static XLA library,
  the corresponding PJRT plugin can be built from source using Bazel (though that is not supported when using
  `cargo vendor` since it requires network access) or it can be provided as a precompiled archive using the
  `PJRT_PLUGIN_ROCM_7_LIB` environment variable. Note that, by default this crate will attempt to download
  a precompiled plugin from GitHub releases of `ryft`, if one can be found for the target platform and
  `PJRT_PLUGIN_ROCM_7_LIB` is not set. Note that this plugin has various ROCm runtime dependencies that are not
  included in the shared library provided by this feature. For Ubuntu, you can install these dependencies using
  the following commands:

  ```bash
  wget https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_7.2.70200-1_all.deb
  sudo apt install ./amdgpu-install_7.2.70200-1_all.deb
  sudo apt update -y
  sudo apt install rocm rocm-core rocm-device-libs rocm-hip-sdk rocprofiler-sdk
  ```

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
  plugins, the AWS Neuron plugin is closed source and thus cannot be built from source. Also note that this plugin has
  various AWS Neuron SDK runtime dependencies that are not included in the shared library provided by this feature.
  For Ubuntu, you can install these dependencies using the following commands:

  ```bash
  wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -
  echo "deb [signed-by=/usr/share/keyrings/neuron-keyring.gpg] https://apt.repos.neuron.amazonaws.com noble main" | sudo tee /etc/apt/sources.list.d/neuron.list
  sudo apt update -y
  sudo apt install -y \
    aws-neuronx-dkms \
    aws-neuronx-collectives \
    aws-neuronx-runtime-lib \
    aws-neuronx-tools
  ```

- **`metal`:** Enables support for loading the PJRT [Metal](https://developer.apple.com/metal/jax/) plugin for
  leveraging Apple Silicon accelerators. If this feature is enabled, this crate will attempt to download a precompiled
  plugin provided as part of [JAX Metal](https://pypi.org/project/jax-metal/), if one can be found for the target
  platform and `PJRT_PLUGIN_METAL_LIB` is not set. Similar to the other feature flags, `PJRT_PLUGIN_METAL_LIB` can be
  used to provide a path to the precompiled plugin to avoid downloading it. Note that, in contrast to some of the other
  PJRT plugins, the Metal plugin is closed source and thus cannot be built from source.

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
- **PJRT Plugin for Metal (JAX Metal):**
    - MacOS `aarch64`

## Contribution

When upgrading the OpenXLA commit used by this crate, treat it as a cross-crate change and follow this checklist:

1. Update all references to the old commit in this crate.
    - Update `XLA_COMMIT` and `XLA_SHA256` in `WORKSPACE`.
    - Update any `build.rs` references that still point to the old commit.
    - Update `JAX_COMMIT` and `JAX_SHA256` only when required by the selected OpenXLA commit.
2. Rebuild and publish precompiled `ryft-xla-sys` artifacts using `.github/workflows/build_ryft_xla_sys.yaml`.
    - Use the GitHub CLI to trigger and monitor the workflow (it may take a couple of hours).
    - After the workflow publishes release artifacts for the new `XLA_COMMIT`, update these functions in `build.rs`:
        - `BuildConfiguration::precompiled_artifact_name`
        - `BuildConfiguration::precompiled_artifact_url_prefix`
        - `BuildConfiguration::precompiled_artifact_checksum`
3. Compare the old and new `xla/pjrt/c/pjrt_c_api.h` headers:
    - Regenerate bindings with the `generate-bindings` feature and sync `src/bindings.rs`.
    - Update any affected `ffi` modules in `crates/ryft-pjrt/src`.
    - Optionally, you can use `git diff <old_commit> <new_commit> xla/pjrt/c` in a checkout of the OpenXLA code
      repository to better understand what changed in the PJRT C API headers.
4. Compare all PJRT extension headers referenced by `pjrt_c_api.h` between the old and new commits.
    - Update `src/bindings.rs` and any affected modules in `crates/ryft-pjrt/src/extensions`.
    - If a new extension was added upstream, add a new module in `crates/ryft-pjrt/src/extensions` and include
      corresponding documentation and tests following existing repository conventions.
5. Compare Protobuf messages referenced by `src/protos.rs` with the corresponding upstream `.proto` files and update
   any stale message definitions.
6. Update `crates/ryft-xla-sys/CHANGELOG.md` as needed.
7. Propagate the changes through `crates/ryft-pjrt`, make sure that tests pass, and update
   `crates/ryft-pjrt/CHANGELOG.md` as needed.
8. Check whether the OpenXLA upgrade changed LLVM; if it did, update `crates/ryft-mlir` as needed, and final update
   `crates/ryft-mlir/CHANGELOG.md` as needed.
9. Audit downstream crates in this repository and apply any compatibility fixes required by the new XLA revision,
   also updating their corresponding `CHANGELOG.md` files as needed.

#### License

<sup>
Licensed under either <a href="../../LICENSE-APACHE">Apache License, Version 2.0</a>
or <a href="../../LICENSE-MIT">MIT license</a> at your option.
</sup>

<br>

<sub>
Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this crate by you,
as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
</sub>
