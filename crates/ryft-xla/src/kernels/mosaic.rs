//! Embedding of binary Mosaic GPU compiler inputs into the native PJRT-owned runtime.
//!
//! The native `mosaic_gpu_v2` handler owns compilation, resource serialization, and execution. This integration
//! fixes the host-buffer ABI and dense row-major layouts; it does not register a second launcher or cache.
//! Execution admission requires a pinned CUDA platform version. The native compilation provider checks the actual
//! PTX compiler and LLVM support before selecting their common PTX ISA; native compilation and loading remain
//! responsible for driver compatibility because PJRT does not expose those provider or driver versions here.

use ryft_core::kernels::{KernelParameterAccess, VerifiedKernel};
use ryft_core::operations::custom_call::CustomCallOperation;
use ryft_core::{ArrayType, DataType, EffectClass, Layout, Memory, Typed};
use ryft_mlir::dialects::stable_hlo::CustomCallMemoryLayouts;
use ryft_mosaic::kernels::gpu::{CompiledKernel, Target};
use ryft_xla_sys::mlir::dialects::mosaic::gpu::{
    MOSAIC_GPU_FFI_TARGET, MOSAIC_GPU_RESOURCE_SCHEMA_VERSION, MOSAIC_GPU_SERDE_VERSION,
};
use sha2::{Digest, Sha256};

use crate::kernels::{KernelEmbeddingError, KernelOutputEmbedding, XlaKernelExecutionFacts, XlaKernelTarget};

/// Embeds checked Mosaic GPU source using the pinned native binary-source and resource schemas.
#[derive(Copy, Clone, Debug, Default)]
pub struct MosaicGpuEmbedding;

impl KernelOutputEmbedding<CompiledKernel> for MosaicGpuEmbedding {
    fn configuration_key(&self) -> Result<Vec<u8>, KernelEmbeddingError> {
        Ok(format!(
            "mosaic GPU embedding 2; target {MOSAIC_GPU_FFI_TARGET}; source {MOSAIC_GPU_SERDE_VERSION}; \
             resources {MOSAIC_GPU_RESOURCE_SCHEMA_VERSION}; inputs then optional assertion token then outputs then optional assertion token; row-major; no collectives; no custom barrier"
        ).into_bytes())
    }

    fn custom_call(
        &self,
        kernel: &VerifiedKernel<'_>,
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
        memory_layouts(&inputs, &outputs)?;
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

impl XlaKernelTarget for Target {
    fn admit_execution(&self, facts: &XlaKernelExecutionFacts) -> Result<(), KernelEmbeddingError> {
        if !facts.platform_name.eq_ignore_ascii_case("cuda")
            || !facts.has_ffi_extension
            || facts.pjrt_version != ryft_pjrt::VERSION
            || facts.devices.is_empty()
        {
            return Err(KernelEmbeddingError::Invalid {
                message: "mosaic GPU requires CUDA devices and the pinned PJRT FFI ABI".to_owned(),
            });
        }
        if !matches!(facts.platform_version.as_str(), "cuda 12090" | "cuda 13020") {
            return Err(KernelEmbeddingError::Invalid {
                message: "mosaic GPU requires pinned CUDA platform version `cuda 12090` or `cuda 13020`".to_owned(),
            });
        }
        if self.maximum_shared_memory_bytes() > 48 * 1024 {
            return Err(KernelEmbeddingError::Invalid {
                message: "mosaic GPU execution admission currently supports at most 48 KiB shared memory".to_owned(),
            });
        }
        let (major, minor) = self.compute_capability();
        let expected = format!("{major}.{minor}");
        for device in &facts.devices {
            if device.attributes.get("compute_capability") != Some(&ryft_pjrt::Value::String(expected.clone())) {
                return Err(KernelEmbeddingError::Invalid {
                    message: format!("mosaic GPU requires exact device compute capability `{expected}`"),
                });
            }
        }
        Ok(())
    }
}

/// Validates the native memref ABI and fixes complete row-major operand/result layouts, including default types.
pub(crate) fn memory_layouts(
    inputs: &[ArrayType],
    outputs: &[ArrayType],
) -> Result<CustomCallMemoryLayouts, KernelEmbeddingError> {
    for r#type in inputs.iter().chain(outputs) {
        if r#type.memory() != Memory::Device
            || r#type.static_shape().is_none()
            || matches!(r#type.data_type(), DataType::Zero | DataType::Token)
        {
            return Err(KernelEmbeddingError::Invalid {
                message: "mosaic GPU requires static device array buffers".to_owned(),
            });
        }
        if let Some(layout) = r#type.layout() {
            let valid = match layout {
                Layout::Tiled(layout) => {
                    layout.tiles().is_empty() && layout.minor_to_major().iter().copied().eq((0..r#type.rank()).rev())
                }
                Layout::Strided(_) => false,
            };
            if !valid {
                return Err(KernelEmbeddingError::Invalid {
                    message: "mosaic GPU requires untiled dense row-major array layouts".to_owned(),
                });
            }
        }
    }
    Ok(CustomCallMemoryLayouts {
        operands: inputs.iter().map(|r#type| (0..r#type.rank()).rev().collect()).collect(),
        results: outputs.iter().map(|r#type| (0..r#type.rank()).rev().collect()).collect(),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use pretty_assertions::assert_eq;
    use ryft_core::TiledLayout;
    use ryft_core::kernels::{KernelSchedule, VerifiedKernel};
    use ryft_core::operations::custom_call::CustomCallAttribute;
    use ryft_mosaic::kernels::gpu::{Compiler, Options};

    use crate::kernels::XlaKernelDeviceFacts;

    use super::*;

    /// Complete deterministic live-device facts without loading a hardware plugin.
    fn facts() -> XlaKernelExecutionFacts {
        XlaKernelExecutionFacts {
            platform_name: "CUDA".to_owned(),
            platform_version: "cuda 13020".to_owned(),
            pjrt_version: ryft_pjrt::VERSION,
            has_ffi_extension: true,
            attributes: BTreeMap::new(),
            devices: vec![XlaKernelDeviceFacts {
                kind: "fixture GPU".to_owned(),
                attributes: BTreeMap::from([(
                    "compute_capability".to_owned(),
                    ryft_pjrt::Value::String("9.0".to_owned()),
                )]),
            }],
        }
    }

    #[test]
    fn test_mosaic_gpu_embedding_configuration_key() {
        assert_eq!(MosaicGpuEmbedding.configuration_key().unwrap(), format!(
            "mosaic GPU embedding 2; target {MOSAIC_GPU_FFI_TARGET}; source {MOSAIC_GPU_SERDE_VERSION}; \
             resources {MOSAIC_GPU_RESOURCE_SCHEMA_VERSION}; inputs then optional assertion token then outputs then optional assertion token; row-major; no collectives; no custom barrier"
        ).into_bytes());
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
        use ryft_core::kernels::{
            Grid, KernelCallOperation, KernelDefinition, KernelParameterAccess, whole_array_parameter,
        };
        use ryft_core::{
            ArrayIrOperation, Context, DimensionBounds, DimensionFromScalarOperation, DimensionVariable, ReferenceRead,
            ReferenceWrite,
        };
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![whole_array_parameter(ArrayType::scalar(DataType::I64), KernelParameterAccess::ReadWrite).unwrap()],
        )
        .unwrap();
        let definition = KernelDefinition::trace(call, |(references, _)| {
            let value = references[0].read()?;
            references[0].context().bind(
                ArrayIrOperation::DimensionFromScalar(DimensionFromScalarOperation::new(DimensionVariable::new(
                    "checked",
                    DimensionBounds::new(0, Some(10)).unwrap(),
                ))),
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
    fn test_target_admit_execution() {
        let target = Target::new(9, 0).unwrap();
        let mut execution = facts();
        target.admit_execution(&execution).unwrap();
        execution.devices[0]
            .attributes
            .insert("compute_capability".to_owned(), ryft_pjrt::Value::String("9.1".to_owned()));
        assert!(matches!(target.admit_execution(&execution), Err(KernelEmbeddingError::Invalid { message })
            if message == "mosaic GPU requires exact device compute capability `9.0`"));
        execution = facts();
        execution.platform_version = "cuda 12090".to_owned();
        target.admit_execution(&execution).unwrap();
        execution.platform_version = "cuda 13010".to_owned();
        assert!(matches!(target.admit_execution(&execution), Err(KernelEmbeddingError::Invalid { message })
            if message == "mosaic GPU requires pinned CUDA platform version `cuda 12090` or `cuda 13020`"));
        execution = facts();
        execution.has_ffi_extension = false;
        assert!(matches!(target.admit_execution(&execution), Err(KernelEmbeddingError::Invalid { message })
            if message == "mosaic GPU requires CUDA devices and the pinned PJRT FFI ABI"));
        execution = facts();
        execution.pjrt_version.minor += 1;
        assert!(matches!(target.admit_execution(&execution), Err(KernelEmbeddingError::Invalid { message })
            if message == "mosaic GPU requires CUDA devices and the pinned PJRT FFI ABI"));
        assert!(matches!(target.with_maximum_shared_memory_bytes(49 * 1024).unwrap().admit_execution(&facts()),
            Err(KernelEmbeddingError::Invalid { message })
                if message == "mosaic GPU execution admission currently supports at most 48 KiB shared memory"));
    }

    #[test]
    fn test_memory_layouts() {
        let matrix = ArrayType::new_static(DataType::F32, [2, 3]);
        let scalar = ArrayType::new_static(DataType::F32, []);
        let layouts = memory_layouts(&[matrix.clone(), scalar.clone()], &[matrix.clone()]).unwrap();
        assert_eq!(layouts.operands, vec![vec![1, 0], vec![]]);
        assert_eq!(layouts.results, vec![vec![1, 0]]);
        let explicit = matrix.clone().with_layout(Layout::Tiled(TiledLayout::new(vec![1, 0], vec![])));
        assert_eq!(memory_layouts(&[explicit], &[]).unwrap().operands, vec![vec![1, 0]]);
        let column_major = matrix.with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], vec![])));
        assert!(matches!(memory_layouts(&[column_major], &[]), Err(KernelEmbeddingError::Invalid { message })
            if message == "mosaic GPU requires untiled dense row-major array layouts"));
    }
}
