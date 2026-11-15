//! Direct Triton artifacts embedded through the existing concrete platform runtime.

use ryft_core::{CustomCallOperation, EffectClass, Typed};
use ryft_kernels::VerifiedKernel;
use ryft_triton::kernels::{Artifact, COMPILER_SCHEMA_VERSION, CompiledKernel, Target};
use sha2::{Digest, Sha256};

use crate::kernels::cuda::{CUDA_KERNEL_CUSTOM_CALL_TARGET, CudaKernelEmbedding, CudaKernelParameterBinding};
use crate::kernels::rocm::RocmKernelEmbedding;
use crate::kernels::staging::{KernelExecutionFacts, KernelTarget};
use crate::kernels::{KernelEmbeddingError, KernelOutputEmbedding, ROCM_KERNEL_CUSTOM_CALL_TARGET};

// TODO(eaplatanios): Review this module.

/// Validates direct Triton output against the logical boundary and delegates concrete artifact execution.
#[derive(Copy, Clone, Debug, Default)]
pub struct TritonEmbedding;

impl KernelOutputEmbedding<CompiledKernel> for TritonEmbedding {
    fn configuration_key(&self) -> Result<Vec<u8>, KernelEmbeddingError> {
        Ok(format!(
            "Triton embedding 1; compiler schema {COMPILER_SCHEMA_VERSION}; cuda envelope 2; \
             rocm envelope 1; static global pointers"
        )
        .into_bytes())
    }

    fn custom_call(
        &self,
        kernel: &VerifiedKernel<'_>,
        output: &CompiledKernel,
    ) -> Result<CustomCallOperation, KernelEmbeddingError> {
        let logical = kernel.definition().operation();
        if output.semantic_key() != kernel.definition().semantic_key()? || !logical.prefetch_types().is_empty() {
            return Err(KernelEmbeddingError::Invalid {
                message: "triton artifact differs from the specialized kernel definition".into(),
            });
        }
        let types = logical.parameters().iter().map(|parameter| parameter.r#type().into_owned()).collect::<Vec<_>>();
        if output.parameter_types() != types {
            return Err(KernelEmbeddingError::Invalid {
                message: "triton parameter types differ from the logical kernel boundary".into(),
            });
        }
        let operation = match output.artifact() {
            Artifact::Cuda(artifact) => CudaKernelEmbedding::from_parameters(
                CUDA_KERNEL_CUSTOM_CALL_TARGET.into(),
                (0..types.len()).map(CudaKernelParameterBinding::Array).collect(),
            )
            .with_row_major_layouts(true)
            .with_assertions(kernel.definition().body().effects().classes().contains(EffectClass::OrderedAssertion))
            .custom_call(kernel, artifact)?,
            Artifact::Rocm(artifact) => {
                RocmKernelEmbedding::new(ROCM_KERNEL_CUSTOM_CALL_TARGET.into()).custom_call(kernel, artifact)?
            }
        };
        Ok(operation
            .with_attribute("ryft.triton.schema", i64::from(COMPILER_SCHEMA_VERSION))
            .with_attribute("ryft.triton.configuration", hex::encode(Sha256::digest(output.configuration_key()))))
    }
}

impl KernelTarget for Target {
    fn admit_execution(&self, facts: &KernelExecutionFacts) -> Result<(), KernelEmbeddingError> {
        let capability = match self {
            Target::Cuda { major, minor } => {
                if !matches!((*major, *minor), (8, 0) | (12, 1))
                    || !facts.platform_name().eq_ignore_ascii_case("cuda")
                    || facts.platform_version() != "cuda 13020"
                    || facts.pjrt_version() != ryft_pjrt::VERSION
                    || !facts.has_ffi_extension()
                    || facts.devices().is_empty()
                {
                    return Err(KernelEmbeddingError::Invalid {
                        message: "triton requires a qualified CUDA target and the pinned CUDA 13.2 PJRT FFI platform"
                            .into(),
                    });
                }
                format!("{major}.{minor}")
            }
            Target::Rocm { architecture } => {
                // The pinned StreamExecutor client exposes the runtime major, minor, and patch versions as this decimal
                // encoding. `MakeComputeCapabilityAttributeString` returns `gfx_version()`, excluding `xnack` and
                // `sramecc` suffixes.
                // The concrete HIP launcher checks those ELF feature requirements against actual device properties.
                let version =
                    facts.platform_version().strip_prefix("rocm ").and_then(|value| value.parse::<u32>().ok());
                if !matches!(architecture.as_str(), "gfx908" | "gfx90a" | "gfx942")
                    || !facts.platform_name().eq_ignore_ascii_case("rocm")
                    || version.is_none_or(|version| !(71_300_000..71_400_000).contains(&version))
                    || facts.pjrt_version() != ryft_pjrt::VERSION
                    || !facts.has_ffi_extension()
                    || facts.devices().is_empty()
                {
                    return Err(KernelEmbeddingError::Invalid {
                        message: "triton requires a supported AMD target and the HIP 7.13 PJRT FFI platform".into(),
                    });
                }
                architecture.clone()
            }
        };
        if facts.devices().iter().any(|device| {
            device.attributes().get("compute_capability") != Some(&ryft_pjrt::Value::String(capability.clone()))
        }) {
            return Err(KernelEmbeddingError::Invalid {
                message: format!("triton requires exact device compute capability `{capability}`"),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use pretty_assertions::assert_eq;
    use ryft_core::{ArrayType, CustomCallAttribute, DataType};
    use ryft_kernels::{KernelCompiler, KernelSchedule};
    use ryft_triton::kernels::{Compiler, Options};

    use crate::kernels::staging::KernelDeviceFacts;

    use super::*;

    /// Exercises the native compiler and embedding contracts without requiring a device runtime.
    #[ryft_kernels::kernel(crate = "ryft_kernels")]
    fn vector(
        #[input(data_type = F32, rank = 1)] left: &ryft_core::Array,
        #[input(data_type = F32, rank = 1)] right: &ryft_core::Array,
        #[output(data_type = F32, shape = [left.shape()[0]])] output: &mut ryft_core::Array,
    ) {
        output.store(left.load() + right.load());
    }

    #[test]
    fn test_triton_embedding_configuration_key() {
        assert_eq!(
            String::from_utf8(TritonEmbedding.configuration_key().unwrap()).unwrap(),
            "Triton embedding 1; compiler schema 2; cuda envelope 2; \
             rocm envelope 1; static global pointers"
        );
    }

    #[test]
    fn test_triton_embedding_custom_call() {
        if std::env::var("RYFT_RUN_TRITON_COMPILER_TESTS").ok().as_deref() != Some("1") {
            return;
        }
        let compiler = Compiler::new();
        let r#type = ArrayType::new_static(DataType::F32, [256]);
        let definition = vector::definition(&r#type, &r#type).unwrap();
        let verified = VerifiedKernel::new(&definition, 1024).unwrap();
        let changed_type = ArrayType::new_static(DataType::F32, [257]);
        let changed = vector::definition(&changed_type, &changed_type).unwrap();
        let changed = VerifiedKernel::new(&changed, 1024).unwrap();
        for (target, expected_target) in [
            (Target::Cuda { major: 12, minor: 1 }, CUDA_KERNEL_CUSTOM_CALL_TARGET),
            (Target::Rocm { architecture: "gfx942".into() }, ROCM_KERNEL_CUSTOM_CALL_TARGET),
        ] {
            let compiled =
                compiler.compile(&verified, &target, &Options::default(), &KernelSchedule::default()).unwrap();
            let operation = TritonEmbedding.custom_call(&verified, &compiled).unwrap();
            assert_eq!(operation.target_name(), expected_target);
            assert_eq!(
                &operation.attributes()[operation.attributes().len() - 2..],
                &[
                    ("ryft.triton.schema".to_owned(), CustomCallAttribute::I64(i64::from(COMPILER_SCHEMA_VERSION))),
                    (
                        "ryft.triton.configuration".to_owned(),
                        CustomCallAttribute::String(hex::encode(Sha256::digest(compiled.configuration_key()))),
                    ),
                ],
            );
            match target {
                Target::Cuda { .. } => {
                    CudaKernelEmbedding::from_custom_call(&verified, &operation).unwrap();
                }
                Target::Rocm { .. } => {
                    RocmKernelEmbedding::from_custom_call(&verified, &operation).unwrap();
                }
            }
            assert!(matches!(TritonEmbedding.custom_call(&changed, &compiled),
                Err(KernelEmbeddingError::Invalid { message })
                    if message == "triton artifact differs from the specialized kernel definition"));
        }
    }

    #[test]
    fn test_target_admit_execution() {
        let facts = |has_ffi_extension: bool| {
            KernelExecutionFacts::new(
                "CUDA",
                "cuda 13020",
                ryft_pjrt::VERSION,
                has_ffi_extension,
                BTreeMap::new(),
                vec![KernelDeviceFacts::new(
                    "GB10",
                    BTreeMap::from([("compute_capability".into(), ryft_pjrt::Value::String("12.1".into()))]),
                )],
            )
        };
        let target = Target::Cuda { major: 12, minor: 1 };
        target.admit_execution(&facts(true)).unwrap();
        let result = Target::Cuda { major: 8, minor: 0 }.admit_execution(&facts(true));
        assert!(matches!(result, Err(KernelEmbeddingError::Invalid { message })
            if message == "triton requires exact device compute capability `8.0`"));
        assert!(matches!(target.admit_execution(&facts(false)), Err(KernelEmbeddingError::Invalid { message })
            if message == "triton requires a qualified CUDA target and the pinned CUDA 13.2 PJRT FFI platform"));
    }

    #[test]
    fn test_target_admit_execution_rocm() {
        let facts = |platform_version: &str, architecture: &str| {
            KernelExecutionFacts::new(
                "ROCM",
                platform_version,
                ryft_pjrt::VERSION,
                true,
                BTreeMap::new(),
                vec![KernelDeviceFacts::new(
                    "AMD Instinct MI300X",
                    BTreeMap::from([("compute_capability".into(), ryft_pjrt::Value::String(architecture.into()))]),
                )],
            )
        };
        let target = Target::Rocm { architecture: "gfx942".into() };
        target.admit_execution(&facts("rocm 71300000", "gfx942")).unwrap();
        let result = target.admit_execution(&facts("rocm 71300000", "gfx90a"));
        assert!(matches!(result, Err(KernelEmbeddingError::Invalid { message })
            if message == "triton requires exact device compute capability `gfx942`"));
        let result = target.admit_execution(&facts("rocm 71400000", "gfx90a"));
        assert!(matches!(result, Err(KernelEmbeddingError::Invalid { message })
            if message == "triton requires a supported AMD target and the HIP 7.13 PJRT FFI platform"));
    }
}
