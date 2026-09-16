//! Direct Triton artifacts embedded through the existing concrete platform runtime.

use ryft_core::kernels::VerifiedKernel;
use ryft_core::operations::custom_call::CustomCallOperation;
use ryft_core::{EffectClass, Typed};
use ryft_triton::kernels::{Artifact, COMPILER_SCHEMA_VERSION, CompiledKernel, Target};
use sha2::{Digest, Sha256};

use crate::kernels::cuda::{CUDA_KERNEL_CUSTOM_CALL_TARGET, CudaKernelEmbedding, CudaKernelParameterBinding};
use crate::kernels::rocm::RocmKernelEmbedding;
use crate::kernels::staging::{XlaKernelExecutionFacts, XlaKernelTarget};
use crate::kernels::{KernelEmbeddingError, KernelOutputEmbedding, ROCM_KERNEL_CUSTOM_CALL_TARGET};

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
            .with_attribute("ryft.triton.configuration", format!("{:x}", Sha256::digest(output.configuration_key()))))
    }
}

impl XlaKernelTarget for Target {
    fn admit_execution(&self, facts: &XlaKernelExecutionFacts) -> Result<(), KernelEmbeddingError> {
        let capability = match self {
            Target::Cuda { major, minor } => {
                if !matches!((*major, *minor), (8, 0) | (12, 1))
                    || !facts.platform_name.eq_ignore_ascii_case("cuda")
                    || facts.platform_version != "cuda 13020"
                    || facts.pjrt_version != ryft_pjrt::VERSION
                    || !facts.has_ffi_extension
                    || facts.devices.is_empty()
                {
                    return Err(KernelEmbeddingError::Invalid {
                        message: "triton requires a qualified CUDA target and the pinned CUDA 13.2 PJRT FFI platform"
                            .into(),
                    });
                }
                format!("{major}.{minor}")
            }
            Target::Rocm { architecture } => {
                // The pinned StreamExecutor client exposes runtime major/minor/patch as this decimal encoding;
                // MakeComputeCapabilityAttributeString returns gfx_version(), excluding xnack/sramecc suffixes.
                // The concrete HIP launcher checks those ELF feature requirements against actual device properties.
                let version = facts.platform_version.strip_prefix("rocm ").and_then(|value| value.parse::<u32>().ok());
                if !matches!(architecture.as_str(), "gfx908" | "gfx90a" | "gfx942")
                    || !facts.platform_name.eq_ignore_ascii_case("rocm")
                    || version.is_none_or(|version| !(71_300_000..71_400_000).contains(&version))
                    || facts.pjrt_version != ryft_pjrt::VERSION
                    || !facts.has_ffi_extension
                    || facts.devices.is_empty()
                {
                    return Err(KernelEmbeddingError::Invalid {
                        message: "triton requires a supported AMD target and the HIP 7.13 PJRT FFI platform".into(),
                    });
                }
                architecture.clone()
            }
        };
        if facts.devices.iter().any(|device| {
            device.attributes.get("compute_capability") != Some(&ryft_pjrt::Value::String(capability.clone()))
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

    use crate::kernels::staging::XlaKernelDeviceFacts;

    use super::*;

    /// Exercises the native compiler and embedding contracts without requiring a device runtime.
    #[ryft_core::kernels::kernel]
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
        use ryft_core::kernels::{KernelCompiler, KernelSchedule};
        use ryft_core::operations::custom_call::CustomCallAttribute;
        use ryft_core::{ArrayType, DataType};
        use ryft_triton::kernels::{Compiler, Options};

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
            assert!(operation.attributes().contains(&(
                "ryft.triton.schema".into(),
                CustomCallAttribute::I64(i64::from(COMPILER_SCHEMA_VERSION)),
            )));
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
        let mut facts = XlaKernelExecutionFacts {
            platform_name: "CUDA".into(),
            platform_version: "cuda 13020".into(),
            pjrt_version: ryft_pjrt::VERSION,
            has_ffi_extension: true,
            attributes: BTreeMap::new(),
            devices: vec![XlaKernelDeviceFacts {
                kind: "GB10".into(),
                attributes: BTreeMap::from([("compute_capability".into(), ryft_pjrt::Value::String("12.1".into()))]),
            }],
        };
        let target = Target::Cuda { major: 12, minor: 1 };
        assert!(target.admit_execution(&facts).is_ok());
        let result = Target::Cuda { major: 8, minor: 0 }.admit_execution(&facts);
        assert!(matches!(result, Err(KernelEmbeddingError::Invalid { message })
            if message == "triton requires exact device compute capability `8.0`"));
        facts.has_ffi_extension = false;
        assert!(matches!(target.admit_execution(&facts), Err(KernelEmbeddingError::Invalid { message })
            if message == "triton requires a qualified CUDA target and the pinned CUDA 13.2 PJRT FFI platform"));
    }

    #[test]
    fn test_target_admit_execution_rocm() {
        let mut facts = XlaKernelExecutionFacts {
            platform_name: "ROCM".into(),
            platform_version: "rocm 71300000".into(),
            pjrt_version: ryft_pjrt::VERSION,
            has_ffi_extension: true,
            attributes: BTreeMap::new(),
            devices: vec![XlaKernelDeviceFacts {
                kind: "AMD Instinct MI300X".into(),
                attributes: BTreeMap::from([("compute_capability".into(), ryft_pjrt::Value::String("gfx942".into()))]),
            }],
        };
        let target = Target::Rocm { architecture: "gfx942".into() };
        assert!(target.admit_execution(&facts).is_ok());
        facts.devices[0]
            .attributes
            .insert("compute_capability".into(), ryft_pjrt::Value::String("gfx90a".into()));
        assert!(matches!(target.admit_execution(&facts), Err(KernelEmbeddingError::Invalid { message })
            if message == "triton requires exact device compute capability `gfx942`"));
        facts.platform_version = "rocm 71400000".into();
        assert!(matches!(target.admit_execution(&facts), Err(KernelEmbeddingError::Invalid { message })
            if message == "triton requires a supported AMD target and the HIP 7.13 PJRT FFI platform"));
    }
}
