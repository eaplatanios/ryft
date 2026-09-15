//! cuTile output embedding through the shared CUDA artifact handler.
//!
//! The adapter owns source compilation, its manifest, and physical argument expansion. This integration validates
//! logical array types and actual execution devices, then delegates artifact persistence and launch to the existing
//! CUDA embedding. Neither loading an executable nor executing it requires the Python compiler.

use ryft_core::kernels::VerifiedKernel;
use ryft_core::operations::custom_call::CustomCallOperation;
use ryft_core::{EffectClass, Typed};
use ryft_cuda::kernels::cutile::{
    Argument, COMPILER_SCHEMA_VERSION, CUDA_TILE_VERSION, CompiledKernel, TILEIRAS_VERSION, Target,
};
use sha2::{Digest, Sha256};

use crate::kernels::cuda::{CUDA_KERNEL_CUSTOM_CALL_TARGET, CudaKernelEmbedding, CudaKernelParameterBinding};
use crate::kernels::staging::{XlaKernelExecutionFacts, XlaKernelTarget};
use crate::kernels::{KernelEmbeddingError, KernelOutputEmbedding};

/// Embeds a validated cuTile artifact using the existing session-owned CUDA launcher.
#[derive(Copy, Clone, Debug, Default)]
pub struct CuTileEmbedding;

impl KernelOutputEmbedding<CompiledKernel> for CuTileEmbedding {
    fn configuration_key(&self) -> Result<Vec<u8>, KernelEmbeddingError> {
        Ok(format!(
            "cuTile embedding 1; compiler schema {COMPILER_SCHEMA_VERSION}; cuda-tile {CUDA_TILE_VERSION}; \
             tileiras {TILEIRAS_VERSION}; cutile_python_v2; cuda envelope 2; static i32 shapes and strides"
        )
        .into_bytes())
    }

    fn custom_call(
        &self,
        kernel: &VerifiedKernel<'_>,
        output: &CompiledKernel,
    ) -> Result<CustomCallOperation, KernelEmbeddingError> {
        if output.semantic_key() != kernel.definition().semantic_key()? {
            return Err(KernelEmbeddingError::Invalid {
                message: "cuTile artifact belongs to a different kernel definition".to_owned(),
            });
        }
        let logical = kernel.definition().operation();
        if !logical.prefetch_types().is_empty() {
            return Err(KernelEmbeddingError::Invalid {
                message: "cuTile requires scalar prefetch specialization before embedding".to_owned(),
            });
        }
        let types = logical.parameters().iter().map(|parameter| parameter.r#type().into_owned()).collect::<Vec<_>>();
        if output.parameter_types() != types {
            return Err(KernelEmbeddingError::Invalid {
                message: "cuTile parameter types differ from the logical kernel boundary".to_owned(),
            });
        }
        let parameters = output
            .arguments()
            .iter()
            .map(|argument| match *argument {
                Argument::Array(index) => CudaKernelParameterBinding::Array(index),
                Argument::I32(value) => CudaKernelParameterBinding::I32(value),
            })
            .collect();
        let embedding = CudaKernelEmbedding::from_parameters(CUDA_KERNEL_CUSTOM_CALL_TARGET.to_owned(), parameters)
            .with_row_major_layouts(true)
            .with_assertions(kernel.definition().body().effects().classes().contains(EffectClass::OrderedAssertion));
        Ok(embedding
            .custom_call(kernel, output.artifact())?
            .with_attribute("ryft.cutile.schema", i64::from(COMPILER_SCHEMA_VERSION))
            .with_attribute("ryft.cutile.configuration", format!("{:x}", Sha256::digest(output.configuration_key()))))
    }
}

impl XlaKernelTarget for Target {
    fn admit_execution(&self, facts: &XlaKernelExecutionFacts) -> Result<(), KernelEmbeddingError> {
        if !facts.platform_name.eq_ignore_ascii_case("cuda")
            || facts.platform_version != "cuda 13020"
            || !facts.has_ffi_extension
            || facts.pjrt_version != ryft_pjrt::VERSION
            || facts.devices.is_empty()
        {
            return Err(KernelEmbeddingError::Invalid {
                message: "cuTile requires the pinned CUDA 13.2 PJRT platform and FFI ABI".to_owned(),
            });
        }
        let (major, minor) = self.compute_capability();
        let capability = format!("{major}.{minor}");
        for device in &facts.devices {
            if device.attributes.get("compute_capability") != Some(&ryft_pjrt::Value::String(capability.clone())) {
                return Err(KernelEmbeddingError::Invalid {
                    message: format!("cuTile requires exact device compute capability `{capability}`"),
                });
            }
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

    /// Exact qualified Spark execution facts, without opening a native client.
    fn facts() -> XlaKernelExecutionFacts {
        XlaKernelExecutionFacts {
            platform_name: "CUDA".to_owned(),
            platform_version: "cuda 13020".to_owned(),
            pjrt_version: ryft_pjrt::VERSION,
            has_ffi_extension: true,
            attributes: BTreeMap::new(),
            devices: vec![XlaKernelDeviceFacts {
                kind: "GB10".to_owned(),
                attributes: BTreeMap::from([(
                    "compute_capability".to_owned(),
                    ryft_pjrt::Value::String("12.1".to_owned()),
                )]),
            }],
        }
    }

    #[test]
    fn test_cu_tile_embedding_configuration_key() {
        assert_eq!(
            String::from_utf8(CuTileEmbedding.configuration_key().unwrap()).unwrap(),
            concat!(
                "cuTile embedding 1; compiler schema 1; cuda-tile 1.5.0; ",
                "tileiras 13.3.36; cutile_python_v2; cuda envelope 2; static i32 shapes and strides",
            )
        );
    }

    #[test]
    fn test_target_admit_execution() {
        let target = Target::new(12, 1).unwrap();
        assert!(target.admit_execution(&facts()).is_ok());
        assert!(matches!(
            Target::new(12, 0).unwrap().admit_execution(&facts()),
            Err(KernelEmbeddingError::Invalid { message })
                if message == "cuTile requires exact device compute capability `12.0`",
        ));
        let mut execution = facts();
        execution.platform_version = "cuda 12090".to_owned();
        assert!(matches!(target.admit_execution(&execution), Err(KernelEmbeddingError::Invalid { message })
            if message == "cuTile requires the pinned CUDA 13.2 PJRT platform and FFI ABI"));
        execution = facts();
        execution.devices.clear();
        assert!(matches!(target.admit_execution(&execution), Err(KernelEmbeddingError::Invalid { message })
            if message == "cuTile requires the pinned CUDA 13.2 PJRT platform and FFI ABI"));
    }
}
