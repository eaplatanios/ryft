//! Test-only prototypes for proving integration seams before they become supported Ryft APIs.

pub use ryft_pjrt::*;

#[cfg(test)]
mod jax;

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashMap;

    use crate::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};

    /// Returns single-device compilation options shared by the prototype seam probes.
    pub(crate) fn test_compilation_options() -> CompilationOptions {
        CompilationOptions {
            argument_layouts: Vec::new(),
            parameter_is_tupled_arguments: false,
            executable_build_options: Some(ExecutableCompilationOptions {
                device_ordinal: -1,
                replica_count: 1,
                partition_count: 1,
                ..Default::default()
            }),
            compile_portable_executable: false,
            profile_version: 0,
            serialized_multi_slice_configuration: Vec::new(),
            environment_option_overrides: HashMap::new(),
            target_config: None,
            allow_in_place_mlir_modification: false,
            matrix_unit_operand_precision: Precision::Default as i32,
        }
    }

    /// Platform identifier used by [`test_for_each_platform`] in prototype tests.
    #[allow(dead_code)]
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    pub(crate) enum TestPlatform {
        Cpu,
        Cuda12,
        Cuda13,
        Rocm7,
        Tpu,
        Neuron,
        Metal,
    }

    /// Executes a prototype test once for CPU and once for each enabled CUDA backend.
    macro_rules! test_for_each_platform {
        (|$plugin:ident, $client:ident, $platform:ident| $body:block) => {{
            {
                let $plugin = $crate::load_cpu_plugin().expect("failed to load the PJRT CPU plugin");
                let $client =
                    $plugin.client($crate::ClientOptions::CPU($crate::CpuClientOptions { device_count: Some(8) }));
                let $client = $client.expect("failed to create a PJRT CPU client");
                let $platform = $crate::tests::TestPlatform::Cpu;
                $body
            }

            #[cfg(feature = "cuda-12")]
            {
                let $plugin = $crate::load_cuda_12_plugin().expect("failed to load the PJRT CUDA 12 plugin");
                let $client = $plugin.client($crate::ClientOptions::GPU($crate::GpuClientOptions {
                    allocator: $crate::GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                    ..Default::default()
                }));
                let $client = $client.expect("failed to create a PJRT CUDA 12 client");
                let $platform = $crate::tests::TestPlatform::Cuda12;
                $body
            }

            #[cfg(feature = "cuda-13")]
            {
                let $plugin = $crate::load_cuda_13_plugin().expect("failed to load the PJRT CUDA 13 plugin");
                let $client = $plugin.client($crate::ClientOptions::GPU($crate::GpuClientOptions {
                    allocator: $crate::GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                    ..Default::default()
                }));
                let $client = $client.expect("failed to create a PJRT CUDA 13 client");
                let $platform = $crate::tests::TestPlatform::Cuda13;
                $body
            }
        }};
    }

    pub(crate) use test_for_each_platform;
}
