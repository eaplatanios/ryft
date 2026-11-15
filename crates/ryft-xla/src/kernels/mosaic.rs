//! Embedding of binary Mosaic GPU compiler inputs into the native runtime owned by PJRT.
//!
//! The native `mosaic_gpu_v2` handler owns compilation, resource serialization, and execution. This integration
//! fixes the host buffer ABI and dense row-major layouts, and it does not register a second launcher or cache.
//! Execution admission requires a pinned CUDA platform version. The native compilation provider checks the actual
//! PTX compiler and LLVM support before selecting their common PTX ISA. Native compilation and loading remain
//! responsible for driver compatibility because PJRT does not expose those provider or driver versions here.
//! The `10.1` target spelling is restricted to the pinned CUDA 12.9 runtime, and `11.0` requires the pinned CUDA 13.2
//! runtime and PTX 9.0. These target names are checked against actual device facts rather than silently normalized.

use ryft_core::{CustomCallOperation, EffectClass, Typed};
use ryft_kernels::{KernelExtension, KernelParameterAccess, VerifiedKernel};
use ryft_mosaic::kernels::gpu::{CompiledKernel, Target};
use ryft_xla_sys::mlir::dialects::mosaic::gpu::{
    MOSAIC_GPU_FFI_TARGET, MOSAIC_GPU_RESOURCE_SCHEMA_VERSION, MOSAIC_GPU_SERDE_VERSION,
};
use sha2::{Digest, Sha256};

use crate::kernels::staging::{KernelExecutionFacts, KernelTarget};
use crate::kernels::{KernelEmbeddingError, KernelOutputEmbedding, dense_memory_layouts};

// TODO(eaplatanios): Review this module.

/// Embeds checked Mosaic GPU source using the pinned native schemas for binary sources and resources.
#[derive(Copy, Clone, Debug, Default)]
pub struct MosaicGpuEmbedding;

impl<Extension: KernelExtension> KernelOutputEmbedding<CompiledKernel, Extension> for MosaicGpuEmbedding {
    fn configuration_key(&self) -> Result<Vec<u8>, KernelEmbeddingError> {
        Ok(format!(
            "mosaic GPU embedding 2; target {MOSAIC_GPU_FFI_TARGET}; source {MOSAIC_GPU_SERDE_VERSION}; \
             resources {MOSAIC_GPU_RESOURCE_SCHEMA_VERSION}; inputs then optional assertion token then outputs then \
             optional assertion token; row-major; no collectives; no custom barrier"
        )
        .into_bytes())
    }

    fn custom_call(
        &self,
        kernel: &VerifiedKernel<'_, Extension>,
        output: &CompiledKernel,
    ) -> Result<CustomCallOperation, KernelEmbeddingError> {
        let logical = kernel.definition().operation();
        if !logical.prefetch_types().is_empty() {
            return Err(KernelEmbeddingError::Invalid {
                message: "mosaic GPU requires scalar prefetch specialization before embedding".to_owned(),
            });
        }
        let inputs = logical
            .parameters()
            .iter()
            .filter(|parameter| parameter.access() != KernelParameterAccess::WriteOnly)
            .map(|parameter| parameter.r#type().into_owned())
            .collect::<Vec<_>>();
        let outputs = logical
            .parameters()
            .iter()
            .filter(|parameter| parameter.access() != KernelParameterAccess::ReadOnly)
            .map(|parameter| parameter.r#type().into_owned())
            .collect::<Vec<_>>();
        let arguments = inputs.iter().chain(&outputs).cloned().collect::<Vec<_>>();
        let mut input_index = 0;
        let has_assertion = kernel.definition().body().effects().classes().contains(EffectClass::OrderedAssertion);
        let mut output_index = inputs.len() + usize::from(has_assertion);
        let slots = logical
            .parameters()
            .iter()
            .map(|parameter| {
                let slot =
                    if parameter.access() == KernelParameterAccess::ReadOnly { input_index } else { output_index };
                if parameter.access() != KernelParameterAccess::WriteOnly {
                    input_index += 1;
                }
                if parameter.access() != KernelParameterAccess::ReadOnly {
                    output_index += 1;
                }
                slot
            })
            .collect::<Vec<_>>();
        if output.argument_types() != arguments || output.parameter_slots() != slots {
            return Err(KernelEmbeddingError::Invalid {
                message: "mosaic GPU host-buffer ABI differs from the logical kernel boundary".to_owned(),
            });
        }
        dense_memory_layouts(&inputs, &outputs, "mosaic GPU")?;
        if output.module().is_empty() || Sha256::digest(output.module()).as_slice() != output.hash() {
            return Err(KernelEmbeddingError::Invalid {
                message: "mosaic GPU binary source does not match its native cache hash".to_owned(),
            });
        }
        if output.shared_memory_bytes() > output.target().maximum_shared_memory_bytes() {
            return Err(KernelEmbeddingError::Invalid {
                message: "mosaic GPU shared-memory usage exceeds its admitted target".to_owned(),
            });
        }
        let mut operation = CustomCallOperation::new(MOSAIC_GPU_FFI_TARGET, outputs)
            .with_attribute("module", output.module())
            .with_attribute("kernel_hash", output.hash().as_slice())
            .with_attribute("uses_xla_collective_metadata", false)
            .with_attribute("use_custom_barrier", false);
        if has_assertion {
            operation = operation.with_effect_class(EffectClass::OrderedAssertion);
        }
        for (output, input) in logical.aliases() {
            operation = operation.with_input_output_alias(input, output)?;
        }
        Ok(operation)
    }
}

impl KernelTarget for Target {
    fn admit_execution(&self, facts: &KernelExecutionFacts) -> Result<(), KernelEmbeddingError> {
        if !facts.platform_name().eq_ignore_ascii_case("cuda")
            || !facts.has_ffi_extension()
            || facts.pjrt_version() != ryft_pjrt::VERSION
            || facts.devices().is_empty()
        {
            return Err(KernelEmbeddingError::Invalid {
                message: "mosaic GPU requires CUDA devices and the pinned PJRT FFI ABI".to_owned(),
            });
        }
        if !matches!(facts.platform_version(), "cuda 12090" | "cuda 13020") {
            return Err(KernelEmbeddingError::Invalid {
                message: "mosaic GPU requires pinned CUDA platform version `cuda 12090` or `cuda 13020`".to_owned(),
            });
        }
        // The `sm_110` target name requires PTX 9.0, but CUDA 12.9 only accepts PTX through 8.8.
        if self.compute_capability() == (11, 0) && facts.platform_version() == "cuda 12090" {
            return Err(KernelEmbeddingError::Invalid {
                message: "compute capability `11.0` requires PTX 9.0 and the pinned CUDA 13.2 runtime".to_owned(),
            });
        }
        // CUDA 13 renamed `sm_101` to `sm_110` and no longer accepts the former assembler target.
        if self.compute_capability() == (10, 1) && facts.platform_version() == "cuda 13020" {
            return Err(KernelEmbeddingError::Invalid {
                message: "compute capability `10.1` requires the pinned CUDA 12.9 runtime".to_owned(),
            });
        }
        if self.maximum_shared_memory_bytes() > 48 * 1024 {
            return Err(KernelEmbeddingError::Invalid {
                message: "mosaic GPU execution admission currently supports at most 48 KiB shared memory".to_owned(),
            });
        }
        let (major, minor) = self.compute_capability();
        let expected = format!("{major}.{minor}");
        for device in facts.devices() {
            if device.attributes().get("compute_capability") != Some(&ryft_pjrt::Value::String(expected.clone())) {
                return Err(KernelEmbeddingError::Invalid {
                    message: format!("mosaic GPU requires exact device compute capability `{expected}`"),
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
    use ryft_core::{
        ArrayIrOperation, ArrayType, Context, CustomCallAttribute, DataType, DimensionBounds,
        DimensionFromScalarOperation, DimensionVariable, ReferenceRead, ReferenceWrite,
    };
    use ryft_kernels::{
        Grid, KernelCallOperation, KernelDefinition, KernelParameterAccess, KernelSchedule, NoKernelExtension,
        VerifiedKernel, whole_array_parameter,
    };
    use ryft_mosaic::kernels::gpu::{Compiler, Options};

    use crate::kernels::staging::KernelDeviceFacts;

    use super::*;

    /// Deterministic device facts with the provided platform version, PJRT version, FFI extension availability, and
    /// compute capability of the one device, without loading a hardware plugin.
    fn facts_with(
        platform_version: &str,
        pjrt_version: ryft_pjrt::Version,
        has_ffi_extension: bool,
        compute_capability: &str,
    ) -> KernelExecutionFacts {
        KernelExecutionFacts::new(
            "CUDA",
            platform_version,
            pjrt_version,
            has_ffi_extension,
            BTreeMap::new(),
            vec![KernelDeviceFacts::new(
                "fixture GPU",
                BTreeMap::from([(
                    "compute_capability".to_owned(),
                    ryft_pjrt::Value::String(compute_capability.to_owned()),
                )]),
            )],
        )
    }

    /// Complete deterministic live device facts without loading a hardware plugin.
    fn facts() -> KernelExecutionFacts {
        facts_with("cuda 13020", ryft_pjrt::VERSION, true, "9.0")
    }

    #[test]
    fn test_mosaic_gpu_embedding_configuration_key() {
        let key = <MosaicGpuEmbedding as KernelOutputEmbedding<CompiledKernel>>::configuration_key(&MosaicGpuEmbedding)
            .unwrap();
        assert_eq!(
            key,
            format!(
                "mosaic GPU embedding 2; target {MOSAIC_GPU_FFI_TARGET}; source {MOSAIC_GPU_SERDE_VERSION}; \
             resources {MOSAIC_GPU_RESOURCE_SCHEMA_VERSION}; inputs then optional assertion token then outputs then \
             optional assertion token; row-major; no collectives; no custom barrier"
            )
            .into_bytes()
        );
    }

    #[test]
    fn test_mosaic_gpu_embedding_custom_call() {
        let definition = crate::kernels::tests::definition();
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let output = verified
            .compile(&Compiler, &Target::new(9, 0).unwrap(), &Options::default(), &KernelSchedule::default())
            .unwrap();
        let operation = MosaicGpuEmbedding.custom_call(&verified, &output).unwrap();
        assert_eq!(output.argument_types(), &[ArrayType::scalar(DataType::I32), ArrayType::scalar(DataType::I32)]);
        assert_eq!(output.parameter_slots(), &[1]);
        assert_eq!(operation.target_name(), MOSAIC_GPU_FFI_TARGET);
        assert_eq!(operation.output_types(), &[ArrayType::scalar(DataType::I32)]);
        assert_eq!(
            operation
                .input_output_aliases()
                .iter()
                .map(|alias| (alias.input_index(), alias.output_index()))
                .collect::<Vec<_>>(),
            vec![(0, 0)]
        );
        assert_eq!(
            operation.attributes(),
            &[
                ("module".to_owned(), CustomCallAttribute::Bytes(output.module().to_vec())),
                ("kernel_hash".to_owned(), CustomCallAttribute::Bytes(output.hash().to_vec())),
                ("uses_xla_collective_metadata".to_owned(), CustomCallAttribute::Boolean(false)),
                ("use_custom_barrier".to_owned(), CustomCallAttribute::Boolean(false)),
            ]
        );
    }

    #[test]
    fn test_mosaic_gpu_embedding_custom_call_preserves_assertions() {
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![whole_array_parameter(ArrayType::scalar(DataType::I64), KernelParameterAccess::ReadWrite).unwrap()],
        )
        .unwrap();
        let definition = KernelDefinition::<NoKernelExtension>::trace(call, |(references, _)| {
            let value = references[0].read()?;
            references[0].context().bind(
                ArrayIrOperation::DimensionFromScalar(DimensionFromScalarOperation::new(
                    DimensionVariable::new("checked", DimensionBounds::new(0, Some(10)).unwrap()).into(),
                )),
                vec![],
                &[value.clone()],
            )?;
            references[0].write(&value)?;
            Ok(())
        })
        .unwrap();
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let output = verified
            .compile(&Compiler, &Target::new(9, 0).unwrap(), &Options::default(), &KernelSchedule::default())
            .unwrap();
        assert_eq!(output.parameter_slots(), &[2]);
        assert_eq!(output.argument_types(), &[ArrayType::scalar(DataType::I64), ArrayType::scalar(DataType::I64)]);
        let compiled =
            crate::kernels::CompiledKernel::from_output(&verified, b"assertion fixture", &output, &MosaicGpuEmbedding)
                .unwrap();
        assert_eq!(compiled.custom_call().effect_class(), Some(EffectClass::OrderedAssertion));
        assert!(compiled.custom_call().has_side_effect());
    }

    #[test]
    fn test_mosaic_gpu_embedding_custom_call_rejects_different_boundary() {
        let definition = crate::kernels::tests::definition();
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![whole_array_parameter(ArrayType::scalar(DataType::F32), KernelParameterAccess::ReadWrite).unwrap()],
        )
        .unwrap();
        let other = KernelDefinition::<NoKernelExtension>::trace(call, |(references, _)| {
            references[0].write(&references[0].read()?)?;
            Ok(())
        })
        .unwrap();
        let output = VerifiedKernel::new(&other, 1)
            .unwrap()
            .compile(&Compiler, &Target::new(9, 0).unwrap(), &Options::default(), &KernelSchedule::default())
            .unwrap();
        assert!(matches!(
            MosaicGpuEmbedding.custom_call(&verified, &output),
            Err(KernelEmbeddingError::Invalid { message })
                if message == "mosaic GPU host-buffer ABI differs from the logical kernel boundary",
        ));
    }

    #[test]
    fn test_target_admit_execution() {
        let target = Target::new(9, 0).unwrap();
        target.admit_execution(&facts()).unwrap();
        let execution = facts_with("cuda 13020", ryft_pjrt::VERSION, true, "9.1");
        assert!(matches!(target.admit_execution(&execution), Err(KernelEmbeddingError::Invalid { message })
            if message == "mosaic GPU requires exact device compute capability `9.0`"));
        target.admit_execution(&facts_with("cuda 12090", ryft_pjrt::VERSION, true, "9.0")).unwrap();
        let execution = facts_with("cuda 13010", ryft_pjrt::VERSION, true, "9.0");
        assert!(matches!(target.admit_execution(&execution), Err(KernelEmbeddingError::Invalid { message })
            if message == "mosaic GPU requires pinned CUDA platform version `cuda 12090` or `cuda 13020`"));
        let execution = facts_with("cuda 13020", ryft_pjrt::VERSION, false, "9.0");
        assert!(matches!(target.admit_execution(&execution), Err(KernelEmbeddingError::Invalid { message })
            if message == "mosaic GPU requires CUDA devices and the pinned PJRT FFI ABI"));
        let later_version = ryft_pjrt::Version { minor: ryft_pjrt::VERSION.minor + 1, ..ryft_pjrt::VERSION };
        let execution = facts_with("cuda 13020", later_version, true, "9.0");
        assert!(matches!(target.admit_execution(&execution), Err(KernelEmbeddingError::Invalid { message })
            if message == "mosaic GPU requires CUDA devices and the pinned PJRT FFI ABI"));
        assert!(matches!(target.with_maximum_shared_memory_bytes(49 * 1024).unwrap().admit_execution(&facts()),
            Err(KernelEmbeddingError::Invalid { message })
                if message == "mosaic GPU execution admission currently supports at most 48 KiB shared memory"));
    }

    #[test]
    fn test_target_admit_execution_ptx_version() {
        let target = Target::new(11, 0).unwrap();
        target.admit_execution(&facts_with("cuda 13020", ryft_pjrt::VERSION, true, "11.0")).unwrap();
        let execution = facts_with("cuda 12090", ryft_pjrt::VERSION, true, "11.0");
        assert!(matches!(target.admit_execution(&execution), Err(KernelEmbeddingError::Invalid { message })
            if message == "compute capability `11.0` requires PTX 9.0 and the pinned CUDA 13.2 runtime"));
        let execution = facts_with("cuda 12090", ryft_pjrt::VERSION, true, "12.1");
        Target::new(12, 1).unwrap().admit_execution(&execution).unwrap();
        let legacy = Target::new(10, 1).unwrap();
        legacy.admit_execution(&facts_with("cuda 12090", ryft_pjrt::VERSION, true, "10.1")).unwrap();
        let execution = facts_with("cuda 13020", ryft_pjrt::VERSION, true, "10.1");
        assert!(matches!(legacy.admit_execution(&execution), Err(KernelEmbeddingError::Invalid { message })
            if message == "compute capability `10.1` requires the pinned CUDA 12.9 runtime"));
    }
}
