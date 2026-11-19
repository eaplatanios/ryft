//! cuTile output embedding through the shared CUDA artifact handler.
//!
//! The adapter owns source compilation, its manifest, and physical argument expansion. This integration validates
//! logical array types and actual execution devices, then delegates artifact persistence and launch to the existing
//! CUDA embedding. Neither loading an executable nor executing it requires the Python compiler.

use ryft_core::{CustomCallOperation, EffectClass, Typed};
use ryft_cuda::kernels::cutile::{
    Argument, COMPILER_SCHEMA_VERSION, CUDA_TILE_VERSION, CompiledKernel, TILEIRAS_VERSION, Target,
};
use ryft_kernels::VerifiedKernel;
use sha2::{Digest, Sha256};

use crate::kernels::cuda::{CUDA_KERNEL_CUSTOM_CALL_TARGET, CudaKernelEmbedding, CudaKernelParameterBinding};
use crate::kernels::staging::{KernelExecutionFacts, KernelTarget};
use crate::kernels::{KernelEmbeddingError, KernelOutputEmbedding};

// TODO(eaplatanios): Review this module.

/// Embeds a validated cuTile artifact using the existing CUDA launcher owned by the session.
#[derive(Copy, Clone, Debug, Default)]
pub struct CuTileEmbedding;

impl KernelOutputEmbedding<CompiledKernel> for CuTileEmbedding {
    fn configuration_key(&self) -> Result<Vec<u8>, KernelEmbeddingError> {
        Ok(format!(
            "cuTile embedding 1; compiler schema {COMPILER_SCHEMA_VERSION}; cuda-tile {CUDA_TILE_VERSION}; \
             tileiras {TILEIRAS_VERSION}; cutile_python_v2; cuda envelope 1; static i32 shapes and strides"
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
            .with_attribute("ryft.cutile.configuration", hex::encode(Sha256::digest(output.configuration_key()))))
    }
}

impl KernelTarget for Target {
    fn admit_execution(&self, facts: &KernelExecutionFacts) -> Result<(), KernelEmbeddingError> {
        if !facts.platform_name().eq_ignore_ascii_case("cuda")
            || facts.platform_version() != "cuda 13020"
            || !facts.has_ffi_extension()
            || facts.pjrt_version() != ryft_pjrt::VERSION
            || facts.devices().is_empty()
        {
            return Err(KernelEmbeddingError::Invalid {
                message: "cuTile requires the pinned CUDA 13.2 PJRT platform and FFI ABI".to_owned(),
            });
        }
        let (major, minor) = self.compute_capability();
        let capability = format!("{major}.{minor}");
        for device in facts.devices() {
            if device.attributes().get("compute_capability") != Some(&ryft_pjrt::Value::String(capability.clone())) {
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
    #[cfg(feature = "cutile-compiler")]
    use std::path::PathBuf;

    use pretty_assertions::assert_eq;
    #[cfg(feature = "cutile-compiler")]
    use ryft_core::{DataType, Device, DeviceMesh, LogicalMesh};
    #[cfg(feature = "cutile-compiler")]
    use ryft_cuda::kernels::cutile::{Compiler, Error, NVCC_VERSION, NVVM_VERSION, Options};
    #[cfg(feature = "cutile-compiler")]
    use ryft_kernels::{KernelCompilationError, KernelCompiler, KernelSchedule};
    #[cfg(feature = "cutile-compiler")]
    use serde::Serialize;

    #[cfg(feature = "cutile-compiler")]
    use crate::FromPjrt;
    #[cfg(feature = "cutile-compiler")]
    use crate::kernels::aot::tests::{executable_bundle, refresh_checksum};
    #[cfg(feature = "cutile-compiler")]
    use crate::kernels::aot::{KernelAotBundle, KernelAotError};
    #[cfg(feature = "cutile-compiler")]
    use crate::kernels::staging::KernelCompilerBinding;
    use crate::kernels::staging::KernelDeviceFacts;
    #[cfg(feature = "cutile-compiler")]
    #[cfg(feature = "cutile-compiler")]
    use crate::tests::execution_client;

    use super::*;

    /// Execution facts with `platform_version` and `devices`, without opening a native client.
    fn facts_with(platform_version: &str, devices: Vec<KernelDeviceFacts>) -> KernelExecutionFacts {
        KernelExecutionFacts::new("CUDA", platform_version, ryft_pjrt::VERSION, true, BTreeMap::new(), devices)
    }

    /// Facts of the one device of the qualified Spark.
    fn devices() -> Vec<KernelDeviceFacts> {
        vec![KernelDeviceFacts::new(
            "GB10",
            BTreeMap::from([("compute_capability".to_owned(), ryft_pjrt::Value::String("12.1".to_owned()))]),
        )]
    }

    /// Exact qualified Spark execution facts, without opening a native client.
    fn facts() -> KernelExecutionFacts {
        facts_with("cuda 13020", devices())
    }

    #[test]
    fn test_cu_tile_embedding_configuration_key() {
        assert_eq!(
            String::from_utf8(CuTileEmbedding.configuration_key().unwrap()).unwrap(),
            concat!(
                "cuTile embedding 1; compiler schema 1; cuda-tile 1.5.0; ",
                "tileiras 13.3.36; cutile_python_v2; cuda envelope 1; static i32 shapes and strides",
            )
        );
    }

    #[cfg(feature = "cutile-compiler")]
    #[test]
    fn test_kernel_aot_bundle_load_cu_tile_historical_lowering() {
        /// Configuration record of earlier revisions, in their field order, which identifies the bundled exporter
        /// but not the Rust lowering that generated the cuTile source.
        #[derive(Serialize)]
        struct HistoricalConfiguration {
            schema: u32,
            cuda_tile: &'static str,
            tileiras: &'static str,
            nvcc: &'static str,
            nvvm: &'static str,
            target: Target,
            python: PathBuf,
            pipeline_stages: Option<usize>,
            buffering_depth: Option<usize>,
            maximum_scratch_bytes: Option<usize>,
            worker_sha256: String,
        }

        /// Stands in for the producer of a recorded configuration, by reporting that configuration as its key.
        struct RecordedProducer(Vec<u8>);

        impl KernelCompiler for RecordedProducer {
            type Target = Target;
            type Options = Options;
            type Output = CompiledKernel;
            type Error = Error;

            fn admit(
                &self,
                kernel: &VerifiedKernel<'_>,
                target: &Target,
                options: &Options,
                schedule: &KernelSchedule,
            ) -> Result<(), KernelCompilationError<Error>> {
                let _ = (kernel, target, options, schedule);
                panic!("configuration identity checks must not invoke compiler admission")
            }

            fn configuration_key(
                &self,
                target: &Target,
                options: &Options,
                schedule: &KernelSchedule,
            ) -> Result<Vec<u8>, KernelCompilationError<Error>> {
                let _ = (target, options, schedule);
                Ok(self.0.clone())
            }

            fn compile(
                &self,
                kernel: &VerifiedKernel<'_>,
                target: &Target,
                options: &Options,
                schedule: &KernelSchedule,
            ) -> Result<CompiledKernel, KernelCompilationError<Error>> {
                let _ = (kernel, target, options, schedule);
                panic!("configuration identity checks must not invoke the compiler")
            }
        }

        let client = execution_client();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![]).unwrap(),
            vec![Device::from_pjrt(client.addressable_devices().unwrap().remove(0)).unwrap()],
        )
        .unwrap();
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let target = Target::new(12, 1).unwrap();
        let python = PathBuf::from("/explicit/python");
        let compiler = Compiler::new(python.clone());
        let current = compiler.configuration_key(&target, &Options::default(), &KernelSchedule::default()).unwrap();
        let worker = serde_json::from_slice::<serde_json::Value>(&current).unwrap()["worker_sha256"].clone();

        // An earlier revision lowered an unsharded 32×32 `BF16_BF16_F32` dot to the default `ct.mma`, which the
        // current lowering rejects. It recorded this configuration for the same exporter, target, and schedule.
        let historical = serde_json::to_vec(&HistoricalConfiguration {
            schema: 1,
            cuda_tile: CUDA_TILE_VERSION,
            tileiras: TILEIRAS_VERSION,
            nvcc: NVCC_VERSION,
            nvvm: NVVM_VERSION,
            target: target.clone(),
            python,
            pipeline_stages: None,
            buffering_depth: None,
            maximum_scratch_bytes: None,
            worker_sha256: worker.as_str().unwrap().to_owned(),
        })
        .unwrap();

        // The current record differs from it only by its last field, which identifies the Rust lowering.
        let lowering = serde_json::from_slice::<serde_json::Value>(&current).unwrap()["lowering_sha256"].clone();
        let mut expected = historical[..historical.len() - 1].to_vec();
        expected.extend_from_slice(format!(",\"lowering_sha256\":{lowering}}}").as_bytes());
        assert_eq!(current, expected);
        let historical = KernelCompilerBinding::new(
            RecordedProducer(historical),
            target.clone(),
            Options::default(),
            KernelSchedule::default(),
            CuTileEmbedding,
            1,
        )
        .unwrap();
        let fresh = KernelCompilerBinding::new(
            compiler,
            target,
            Options::default(),
            KernelSchedule::default(),
            CuTileEmbedding,
            1,
        )
        .unwrap();

        // A bundle of the historical producer is well formed and loads with the historical binding.
        let bundle = executable_bundle(&session, &mesh, DataType::I32, false, &historical);
        let mut bytes = bundle.to_bytes().unwrap();
        KernelAotBundle::from_bytes(&bytes, 1).unwrap().load(&session, &historical, &mesh).unwrap();

        // The current binding rejects that bundle before it deserializes the executable, which is replaced here by
        // bytes that no executable decoder accepts. The executable is the last section of the envelope.
        let length = u64::from_le_bytes(bytes[24..32].try_into().unwrap()) as usize;
        bytes.truncate(bytes.len() - length);
        bytes.extend_from_slice(b"transport-only executable bytes");
        bytes[24..32].copy_from_slice(&(b"transport-only executable bytes".len() as u64).to_le_bytes());
        refresh_checksum(&mut bytes);
        assert!(matches!(
            KernelAotBundle::from_bytes(&bytes, 1).unwrap().load(&session, &fresh, &mesh),
            Err(KernelAotError::Invalid { message }) if message == "compiler binding configuration mismatch",
        ));

        // A bundle of the current producer loads with the current binding.
        let bundle = executable_bundle(&session, &mesh, DataType::I32, false, &fresh);
        KernelAotBundle::from_bytes(&bundle.to_bytes().unwrap(), 1)
            .unwrap()
            .load(&session, &fresh, &mesh)
            .unwrap();
    }

    #[test]
    fn test_target_admit_execution() {
        let target = Target::new(12, 1).unwrap();
        target.admit_execution(&facts()).unwrap();
        assert!(matches!(
            Target::new(12, 0).unwrap().admit_execution(&facts()),
            Err(KernelEmbeddingError::Invalid { message })
                if message == "cuTile requires exact device compute capability `12.0`",
        ));
        let execution = facts_with("cuda 12090", devices());
        assert!(matches!(target.admit_execution(&execution), Err(KernelEmbeddingError::Invalid { message })
            if message == "cuTile requires the pinned CUDA 13.2 PJRT platform and FFI ABI"));
        let execution = facts_with("cuda 13020", Vec::new());
        assert!(matches!(target.admit_execution(&execution), Err(KernelEmbeddingError::Invalid { message })
            if message == "cuTile requires the pinned CUDA 13.2 PJRT platform and FFI ABI"));
    }
}
