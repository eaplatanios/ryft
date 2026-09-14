//! CUDA pointer-ABI embedding and validated payload reconstruction.

#[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
use std::sync::OnceLock;

use ryft_core::kernels::{KernelParameterAccess, VerifiedKernel};
use ryft_core::operations::custom_call::{CustomCallAttribute, CustomCallOperation};
use ryft_core::{ArrayIrType, DataType, Memory, Operation, Typed};
#[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
use ryft_cuda::CudaKernelLauncher;
use ryft_cuda::{
    CudaArtifactFormat, CudaKernelAbi, CudaKernelArtifact, CudaKernelLaunchDimensions, CudaKernelParameterType,
};
#[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
use ryft_pjrt::extensions::ffi::{
    FfiAttribute, FfiCallFrame, FfiError, FfiExecutionStage, FfiHandler, FfiHandlerTraits, FfiInput, FfiOutput,
    FfiTypeInformation, XLA_FFI_CallFrame, XLA_FFI_Error, XLA_FFI_Handler,
};
use ryft_pjrt::extensions::ffi::{FfiTypeId, FfiUserData};
use ryft_pjrt::{Client, ExecutionContext};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::ToPjrt;
use crate::kernels::{KernelEmbeddingError, KernelOutputEmbedding};

/// Registered handler for versioned ready CUDA artifacts embedded in StableHLO.
pub const CUDA_KERNEL_CUSTOM_CALL_TARGET: &str = "ryft.kernel.cuda";

/// Native type identifier allocated by the same plugin registry as the CUDA handler.
#[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
static CUDA_RUNTIME_REGISTRATION: OnceLock<Result<FfiTypeId, ryft_pjrt::Error>> = OnceLock::new();

/// Custom-call buffer supplying one physical CUDA pointer argument.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CudaKernelBufferBinding {
    /// Read-only custom-call operand at the specified index.
    Input(usize),

    /// Writable custom-call result at the specified index, including read-write aliased results.
    Output(usize),
}

/// XLA embedding of a ready [`CudaKernelArtifact`] whose parameters are array pointers.
///
/// `parameter_order` maps each physical CUDA argument to a logical kernel parameter. The complete permutation is
/// required: read-only parameters use custom-call operands, writable parameters use custom-call results, and
/// read-write parameters also carry their canonical operand/result alias. This baseline rejects scalar-expanded
/// ABIs instead of inventing shape/stride arguments. A compiler-specific bridge can implement
/// [`KernelOutputEmbedding`] for a richer adapter-owned output without changing the CUDA artifact contract.
///
/// Select [`CUDA_KERNEL_CUSTOM_CALL_TARGET`] to use the session-owned native handler. The XLA domain initializes
/// its launcher during compilation and reload, supplies execution-context user data at submission, and retains
/// whole-execution fences through session cleanup. Other target names select explicitly registered handlers with
/// the same payload contract. Artifact construction and embedding alone do not establish executable-code safety.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaKernelEmbedding {
    /// Registered XLA FFI target that decodes this envelope.
    target_name: String,

    /// Physical argument index to logical parameter index.
    parameter_order: Vec<usize>,
}

impl CudaKernelEmbedding {
    /// Creates a pointer-ABI mapping for an explicitly selected registered target.
    pub fn new(target_name: String, parameter_order: Vec<usize>) -> Self {
        Self { target_name, parameter_order }
    }

    /// Reconstructs and validates a serialized CUDA artifact and its pointer mapping before any native loading.
    /// Unknown versions, corruption, invalid canonical artifact metadata, and disagreement with the verified
    /// logical signature or buffer mapping are rejected. This does not invoke a compiler or load native code.
    pub fn from_custom_call<Extension: Operation<Type = ArrayIrType>>(
        kernel: &VerifiedKernel<'_, Extension>,
        operation: &CustomCallOperation,
    ) -> Result<(Self, CudaKernelArtifact), KernelEmbeddingError> {
        let attribute = |name: &str| {
            let values = operation.attributes().iter().filter(|(key, _)| key == name).collect::<Vec<_>>();
            match values.as_slice() {
                [(_, CustomCallAttribute::String(value))] => Ok(value.as_str()),
                _ => Err(KernelEmbeddingError::Invalid {
                    message: format!("expected exactly one string attribute `{name}`"),
                }),
            }
        };
        let payload = attribute("ryft.cuda.payload")?;
        let digest = attribute("ryft.cuda.sha256")?;
        if format!("{:x}", Sha256::digest(payload.as_bytes())) != digest {
            return Err(KernelEmbeddingError::Invalid { message: "cuda payload digest mismatch".to_owned() });
        }
        let envelope: CudaEnvelope = serde_json::from_str(payload)?;
        if envelope.version != 1 {
            return Err(KernelEmbeddingError::Invalid {
                message: format!("unsupported CUDA embedding schema `{}`", envelope.version),
            });
        }
        if envelope.argument_bindings.len() != envelope.parameter_order.len() {
            return Err(KernelEmbeddingError::Invalid {
                message: "cuda buffer binding count differs from its physical ABI".to_owned(),
            });
        }
        for binding in &envelope.argument_bindings {
            let valid = match binding {
                CudaKernelBufferBinding::Input(index) => *index < envelope.parameter_order.len(),
                CudaKernelBufferBinding::Output(index) => *index < operation.output_types().len(),
            };
            if !valid {
                return Err(KernelEmbeddingError::Invalid {
                    message: "cuda payload references a missing custom-call buffer".to_owned(),
                });
            }
        }
        let embedding = Self::new(operation.target_name().to_owned(), envelope.parameter_order.clone());
        if embedding.argument_bindings(kernel)? != envelope.argument_bindings {
            return Err(KernelEmbeddingError::Invalid {
                message: "cuda buffer bindings differ from the logical kernel signature".to_owned(),
            });
        }
        let artifact = envelope.artifact()?;
        let expected = embedding.custom_call(kernel, &artifact)?;
        if operation.output_types() != expected.output_types()
            || operation.input_output_aliases() != expected.input_output_aliases()
            || expected.attributes().iter().any(|attribute| !operation.attributes().contains(attribute))
        {
            return Err(KernelEmbeddingError::Invalid {
                message: "cuda custom-call signature differs from its verified kernel".to_owned(),
            });
        }
        Ok((embedding, artifact))
    }

    /// Returns the physical argument to logical parameter permutation.
    pub fn parameter_order(&self) -> &[usize] {
        &self.parameter_order
    }

    /// Resolves each physical pointer to the existing custom-call input/result buffer slots.
    /// Writable arguments use results so input liveness and donation remain XLA's ordinary functional alias policy.
    pub fn argument_bindings<Extension: Operation<Type = ArrayIrType>>(
        &self,
        kernel: &VerifiedKernel<'_, Extension>,
    ) -> Result<Vec<CudaKernelBufferBinding>, KernelEmbeddingError> {
        let mut input = 0;
        let mut output = 0;
        let logical = kernel
            .definition()
            .operation()
            .parameters()
            .iter()
            .map(|parameter| match parameter.access() {
                KernelParameterAccess::ReadOnly => {
                    let binding = CudaKernelBufferBinding::Input(input);
                    input += 1;
                    binding
                }
                KernelParameterAccess::WriteOnly => {
                    let binding = CudaKernelBufferBinding::Output(output);
                    output += 1;
                    binding
                }
                KernelParameterAccess::ReadWrite => {
                    let binding = CudaKernelBufferBinding::Output(output);
                    input += 1;
                    output += 1;
                    binding
                }
            })
            .collect::<Vec<_>>();
        self.parameter_order
            .iter()
            .map(|&parameter| {
                logical.get(parameter).copied().ok_or_else(|| KernelEmbeddingError::Invalid {
                    message: format!("cuda mapping references missing logical parameter `{parameter}`"),
                })
            })
            .collect()
    }
}

impl<Extension: Operation<Type = ArrayIrType>> KernelOutputEmbedding<CudaKernelArtifact, Extension>
    for CudaKernelEmbedding
{
    fn configuration_key(&self) -> Result<Vec<u8>, KernelEmbeddingError> {
        Ok(serde_json::to_vec(&(1u32, &self.target_name, &self.parameter_order))?)
    }

    fn custom_call(
        &self,
        kernel: &VerifiedKernel<'_, Extension>,
        output: &CudaKernelArtifact,
    ) -> Result<CustomCallOperation, KernelEmbeddingError> {
        let logical = kernel.definition().operation();
        let mut sorted = self.parameter_order.clone();
        sorted.sort_unstable();
        if sorted != (0..logical.parameters().len()).collect::<Vec<_>>() {
            return Err(KernelEmbeddingError::Invalid {
                message: "cuda pointer mapping must name every logical parameter exactly once".to_owned(),
            });
        }
        if output.abi().parameters() != vec![CudaKernelParameterType::DevicePointer; self.parameter_order.len()] {
            return Err(KernelEmbeddingError::Invalid {
                message: "cuda artifact ABI does not match the declared pointer mapping".to_owned(),
            });
        }
        for parameter in logical.parameters() {
            if parameter.r#type().memory() != Memory::Device
                || matches!(parameter.r#type().data_type(), DataType::Zero | DataType::Token)
            {
                return Err(KernelEmbeddingError::Invalid {
                    message: "cuda pointer embedding requires stored device arrays".to_owned(),
                });
            }
            if parameter.r#type().static_shape().is_none() {
                return Err(KernelEmbeddingError::Invalid {
                    message: "cuda pointer embedding requires static logical array shapes".to_owned(),
                });
            }
        }
        let launch = output.launch_dimensions();
        let envelope = CudaEnvelope {
            version: 1,
            format: match output.format() {
                CudaArtifactFormat::Cubin => "cubin",
                CudaArtifactFormat::Ptx => "ptx",
            }
            .to_owned(),
            image: output.bytes().to_vec(),
            symbol: output.symbol().to_owned(),
            architecture: output.target_architecture().to_owned(),
            abi_schema: output.abi().schema().to_owned(),
            abi_version: output.abi().version(),
            grid: launch.grid(),
            block: launch.block(),
            shared_memory_bytes: launch.dynamic_shared_memory_bytes(),
            parameter_order: self.parameter_order.clone(),
            argument_bindings: self.argument_bindings(kernel)?,
            input_count: logical
                .parameters()
                .iter()
                .filter(|parameter| parameter.access() != KernelParameterAccess::WriteOnly)
                .count(),
            output_count: logical
                .parameters()
                .iter()
                .filter(|parameter| parameter.access() != KernelParameterAccess::ReadOnly)
                .count(),
            parameter_types: logical
                .parameters()
                .iter()
                .map(|parameter| {
                    let array_type = parameter.r#type();
                    (
                        array_type.data_type().to_pjrt().to_string(),
                        array_type.static_shape().unwrap().dimensions().to_vec(),
                    )
                })
                .collect(),
        };
        let payload = serde_json::to_string(&envelope)?;
        let mut operation = CustomCallOperation::new(
            self.target_name.clone(),
            logical
                .parameters()
                .iter()
                .filter(|parameter| parameter.access() != KernelParameterAccess::ReadOnly)
                .map(|parameter| parameter.r#type().into_owned())
                .collect(),
        )
        .with_attribute("ryft.cuda.sha256", format!("{:x}", Sha256::digest(payload.as_bytes())))
        .with_attribute("ryft.cuda.payload", payload);
        for (output, input) in logical.aliases() {
            operation = operation.with_input_output_alias(input, output)?;
        }
        Ok(operation)
    }
}

/// Versioned XLA persistence envelope; canonical CUDA objects are reconstructed through their own validators.
/// Fields serialize the existing artifact and launch metadata without introducing a second runtime representation.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CudaEnvelope {
    /// Envelope decoder version.
    version: u32,

    /// Canonical image format.
    format: String,

    /// Owned image bytes.
    image: Vec<u8>,

    /// Canonical entry symbol.
    symbol: String,

    /// Canonical target architecture.
    architecture: String,

    /// Canonical producer-neutral ABI schema.
    abi_schema: String,

    /// Canonical ABI version.
    abi_version: u32,

    /// Grid launch dimensions.
    grid: [u32; 3],

    /// Block launch dimensions.
    block: [u32; 3],

    /// Dynamic shared-memory byte count.
    shared_memory_bytes: u32,

    /// Physical pointer arguments mapped to logical kernel parameters.
    parameter_order: Vec<usize>,

    /// Physical pointers resolved to canonical custom-call buffer slots.
    argument_bindings: Vec<CudaKernelBufferBinding>,

    /// Exact logical element types and static shapes used to validate native FFI buffers.
    parameter_types: Vec<(String, Vec<usize>)>,

    /// Exact custom-call operand count, including entering read-write arrays.
    input_count: usize,

    /// Exact custom-call result count, including updated read-write arrays.
    output_count: usize,
}

impl CudaEnvelope {
    /// Reconstructs the canonical driver artifact after validating the versioned envelope.
    fn artifact(&self) -> Result<CudaKernelArtifact, KernelEmbeddingError> {
        if self.version != 1
            || self.argument_bindings.len() != self.parameter_order.len()
            || self.parameter_types.len() != self.parameter_order.len()
        {
            return Err(KernelEmbeddingError::Invalid {
                message: "invalid cuda embedding schema or parameter count".to_owned(),
            });
        }
        let mut order = self.parameter_order.clone();
        order.sort_unstable();
        if order != (0..order.len()).collect::<Vec<_>>() {
            return Err(KernelEmbeddingError::Invalid {
                message: "cuda parameter order is not a permutation".to_owned(),
            });
        }
        let format = match self.format.as_str() {
            "cubin" => CudaArtifactFormat::Cubin,
            "ptx" => CudaArtifactFormat::Ptx,
            _ => return Err(KernelEmbeddingError::Invalid { message: "unsupported cuda image format".to_owned() }),
        };
        let abi = CudaKernelAbi::new(
            self.abi_schema.clone(),
            self.abi_version,
            vec![CudaKernelParameterType::DevicePointer; self.parameter_order.len()],
        )?;
        let launch = CudaKernelLaunchDimensions::new(self.grid, self.block, self.shared_memory_bytes)?;
        Ok(CudaKernelArtifact::new(
            format,
            self.image.clone(),
            self.symbol.clone(),
            self.architecture.clone(),
            launch,
            abi,
        )?)
    }
}

/// Session-owned launcher borrowed by native execution contexts. The session waits its existing execution fences
/// before dropping this object, so cached modules are released while the externally borrowed PJRT client is alive.
pub(crate) struct CudaKernelRuntime {
    /// Existing driver owner, including its canonical bounded module cache and cleanup reporting.
    #[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
    launcher: CudaKernelLauncher,

    /// Plugin-owned FFI type used to recover this stable session allocation.
    type_id: FfiTypeId,
}

impl CudaKernelRuntime {
    /// Loads the driver and idempotently registers the shared handler for this CUDA plugin.
    pub(crate) fn new(client: &Client<'_>) -> Result<Self, KernelEmbeddingError> {
        #[cfg(not(any(feature = "cuda-12", feature = "cuda-13")))]
        {
            let _client = client;
            Err(KernelEmbeddingError::Invalid {
                message: "cuda artifact execution requires the `cuda-12` or `cuda-13` feature".to_owned(),
            })
        }
        #[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
        {
            let launcher = client.cuda_kernel_launcher()?;
            let type_id = CUDA_RUNTIME_REGISTRATION
                .get_or_init(|| {
                    let type_id = client.register_ffi_type(
                        "ryft.kernel.cuda.runtime",
                        FfiTypeId::UNKNOWN,
                        FfiTypeInformation::new(None),
                    )?;
                    client.register_ffi_handler(
                        CUDA_KERNEL_CUSTOM_CALL_TARGET,
                        "CUDA",
                        FfiHandler::from(cuda_kernel_handler as XLA_FFI_Handler),
                        FfiHandlerTraits::NONE,
                    )?;
                    Ok(type_id)
                })
                .clone()?;
            Ok(Self { launcher, type_id })
        }
    }

    /// Creates a native context borrowing this stable allocation until all submitted devices complete.
    ///
    /// # Safety
    /// The owner must keep this object and its PJRT client alive until whole-execution completion, including when
    /// outputs are dropped early. PJRT retains the native context, but its user-data insertion is non-owning.
    pub(crate) unsafe fn execution_context(&self, client: &Client<'_>) -> Result<ExecutionContext, ryft_pjrt::Error> {
        let context = client.execution_context()?;
        unsafe { context.add_ffi_user_data(FfiUserData::new(self.type_id, (self as *const Self).cast_mut().cast())) }?;
        Ok(context)
    }
}

/// Decodes one ready artifact and submits it on the invocation's current CUDA stream.
#[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
fn launch_cuda_kernel(frame: &FfiCallFrame<'_>) -> Result<(), FfiError> {
    let mut payload = None;
    let mut digest = None;
    for attribute in frame.attributes() {
        let (name, attribute) = attribute?;
        if name == "ryft.cuda.payload" || name == "ryft.cuda.sha256" {
            let FfiAttribute::String { string } = attribute else {
                return Err(FfiError::invalid_argument("cuda payload attributes must be strings"));
            };
            let slot = if name == "ryft.cuda.payload" { &mut payload } else { &mut digest };
            if slot.replace(string).is_some() {
                return Err(FfiError::invalid_argument("duplicate cuda payload attribute"));
            }
        }
    }
    let payload = payload.ok_or_else(|| FfiError::invalid_argument("missing cuda payload"))?;
    if digest != Some(format!("{:x}", Sha256::digest(payload.as_bytes())).as_str()) {
        return Err(FfiError::invalid_argument("cuda payload digest mismatch"));
    }
    let envelope: CudaEnvelope =
        serde_json::from_str(payload).map_err(|error| FfiError::invalid_argument(error.to_string()))?;
    let artifact = envelope.artifact().map_err(|error| FfiError::invalid_argument(error.to_string()))?;
    if frame.input_count() != envelope.input_count || frame.output_count() != envelope.output_count {
        return Err(FfiError::invalid_argument("cuda frame buffer counts differ from the declared signature"));
    }
    let context = frame.context()?;
    let type_id = CUDA_RUNTIME_REGISTRATION
        .get()
        .ok_or_else(|| FfiError::internal("cuda runtime is not registered"))?
        .as_ref()
        .map_err(|error| FfiError::internal(error.to_string()))?;
    let data = context.user_data(*type_id)?.data;
    if data.is_null() {
        return Err(FfiError::invalid_argument("missing session cuda runtime"));
    }
    // The only producer is CudaKernelRuntime::execution_context; the session retains this exact allocation and
    // waits every whole-program execution before destruction, including dropped output handles.
    let runtime = unsafe { &*data.cast::<CudaKernelRuntime>() };
    let mut buffers = Vec::with_capacity(envelope.argument_bindings.len());
    for (physical, binding) in envelope.argument_bindings.iter().enumerate() {
        let buffer = match *binding {
            CudaKernelBufferBinding::Input(index) => {
                let FfiInput::Buffer { buffer } = frame.input(index)?;
                buffer
            }
            CudaKernelBufferBinding::Output(index) => {
                let FfiOutput::Buffer { buffer } = frame.output(index)?;
                buffer
            }
        };
        let (element_type, dimensions) = &envelope.parameter_types[envelope.parameter_order[physical]];
        if buffer.element_type().to_string() != *element_type
            || buffer
                .dimensions()
                .iter()
                .map(|&value| usize::try_from(value))
                .collect::<Result<Vec<_>, _>>()
                .ok()
                .as_ref()
                != Some(dimensions)
        {
            return Err(FfiError::invalid_argument("cuda buffer differs from its declared element type or shape"));
        }
        buffers.push(buffer);
    }
    let arguments = buffers
        .iter()
        .map(|buffer| unsafe { buffer.cuda_kernel_argument() })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| FfiError::invalid_argument(error.to_string()))?;
    // XLA owns buffer readiness, functional aliasing and the invocation stream. CUDA owner checks context, device,
    // graph capture and canonical launch limits; session completion retains modules through all queued work.
    let launch =
        unsafe { context.cuda_kernel_launch(arguments) }.map_err(|error| FfiError::internal(error.to_string()))?;
    unsafe { runtime.launcher.launch(&artifact, &launch) }.map_err(|error| FfiError::internal(error.to_string()))
}

/// Typed XLA FFI entry point for the shared CUDA artifact launcher.
#[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
unsafe extern "C" fn cuda_kernel_handler(frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
    let frame = match unsafe { FfiCallFrame::from_c_api(frame) } {
        Ok(frame) => frame,
        Err(_) => return std::ptr::null_mut(),
    };
    if frame.register_metadata(FfiTypeId::UNKNOWN) || frame.stage() != FfiExecutionStage::Execution {
        return std::ptr::null_mut();
    }
    match launch_cuda_kernel(&frame) {
        Ok(()) => std::ptr::null_mut(),
        Err(error) => match frame.api() {
            Ok(api) => unsafe { error.to_c_api(api) },
            Err(_) => std::ptr::null_mut(),
        },
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::kernels::VerifiedKernel;
    use ryft_cuda::CudaScalarType;

    use crate::kernels::CompiledKernel;
    use crate::kernels::tests::definition;

    use super::*;

    /// Ready artifact fixture constructed solely from the platform runtime's public types.
    fn artifact(parameters: Vec<CudaKernelParameterType>) -> CudaKernelArtifact {
        CudaKernelArtifact::new(
            CudaArtifactFormat::Ptx,
            b".version 8.0\n.target sm_90\n.address_size 64\n.visible .entry identity(.param .u64 data) { ret; }\n"
                .to_vec(),
            "identity",
            "compute_90",
            CudaKernelLaunchDimensions::new([1, 1, 1], [1, 1, 1], 0).unwrap(),
            CudaKernelAbi::new("fixture.pointer", 1, parameters).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn test_cuda_kernel_embedding_new() {
        let embedding = CudaKernelEmbedding::new("ryft.test.ready".to_owned(), vec![0]);
        assert_eq!(embedding.parameter_order(), &[0]);
    }

    #[test]
    fn test_cuda_kernel_embedding_from_custom_call() {
        let definition = definition();
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let artifact = artifact(vec![CudaKernelParameterType::DevicePointer]);
        let embedding = CudaKernelEmbedding::new("ryft.test.ready".to_owned(), vec![0]);
        let compiled = CompiledKernel::from_output(&verified, b"ready configuration", &artifact, &embedding).unwrap();
        let (decoded_embedding, decoded) =
            CudaKernelEmbedding::from_custom_call(&verified, compiled.custom_call()).unwrap();
        assert_eq!(decoded_embedding, embedding);
        assert_eq!(embedding.argument_bindings(&verified).unwrap(), vec![CudaKernelBufferBinding::Output(0)]);
        assert_eq!(decoded.bytes(), artifact.bytes());
        assert_eq!(decoded.abi(), artifact.abi());
        assert_eq!(decoded.launch_dimensions(), artifact.launch_dimensions());
        assert_eq!(decoded.target_architecture(), artifact.target_architecture());
        assert_eq!(decoded.symbol(), artifact.symbol());
        let roundtrip =
            CompiledKernel::from_output(&verified, b"ready configuration", &decoded, &decoded_embedding).unwrap();
        assert_eq!(roundtrip.custom_call().to_string(), compiled.custom_call().to_string());
    }

    #[test]
    fn test_cuda_kernel_embedding_declares_runtime_without_external_effects() {
        use ryft_core::{ArrayType, DataType, Placeholder};

        use crate::experimental::lowering::lower_mlir_module_for_program;
        use crate::experimental::ops::{XlaConstant, XlaProgramBuilder};

        let definition = definition();
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let artifact = artifact(vec![CudaKernelParameterType::DevicePointer]);
        let embedding = CudaKernelEmbedding::new(CUDA_KERNEL_CUSTOM_CALL_TARGET.to_owned(), vec![0]);
        let compiled = CompiledKernel::from_output(&verified, b"identity", &artifact, &embedding).unwrap();
        let scalar = ArrayType::scalar(DataType::I32);
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(scalar.clone().into());
        let output = builder.add_instruction(compiled.custom_call().clone(), vec![], vec![input], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let lowered = lower_mlir_module_for_program(
            &program,
            &[],
            &vec![scalar.clone()],
            &vec![scalar],
            "main",
            None,
            None,
            Some("cuda"),
        )
        .unwrap();
        let (module, signature, assertions) = lowered.into_parts();
        assert!(module.contains("stablehlo.custom_call @ryft.kernel.cuda"));
        assert!(signature.requires_cuda_kernel_runtime());
        assert!(!signature.has_effects());
        assert!(!assertions);
    }

    #[test]
    fn test_cuda_kernel_embedding_from_custom_call_rejects_corruption() {
        let definition = definition();
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let operation = CustomCallOperation::new("ryft.test.ready", vec![])
            .with_attribute("ryft.cuda.payload", "{}")
            .with_attribute("ryft.cuda.sha256", "corrupted");
        assert!(matches!(
            CudaKernelEmbedding::from_custom_call(&verified, &operation),
            Err(KernelEmbeddingError::Invalid { message }) if message == "cuda payload digest mismatch",
        ));
    }

    #[test]
    fn test_cuda_kernel_embedding_from_custom_call_rejects_changed_buffer_metadata() {
        let definition = definition();
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let embedding = CudaKernelEmbedding::new(CUDA_KERNEL_CUSTOM_CALL_TARGET.to_owned(), vec![0]);
        let artifact = artifact(vec![CudaKernelParameterType::DevicePointer]);
        let original = embedding.custom_call(&verified, &artifact).unwrap();
        let payload = original
            .attributes()
            .iter()
            .find_map(|(name, value)| match (name.as_str(), value) {
                ("ryft.cuda.payload", CustomCallAttribute::String(payload)) => Some(payload),
                _ => None,
            })
            .unwrap();
        let mut envelope: CudaEnvelope = serde_json::from_str(payload).unwrap();
        envelope.parameter_types[0].1 = vec![4];
        let payload = serde_json::to_string(&envelope).unwrap();
        let changed = CustomCallOperation::new(original.target_name(), original.output_types().to_vec())
            .with_input_output_alias(0, 0)
            .unwrap()
            .with_attribute("ryft.cuda.sha256", format!("{:x}", Sha256::digest(payload.as_bytes())))
            .with_attribute("ryft.cuda.payload", payload);
        assert!(matches!(
            CudaKernelEmbedding::from_custom_call(&verified, &changed),
            Err(KernelEmbeddingError::Invalid { message })
                if message == "cuda custom-call signature differs from its verified kernel",
        ));
    }

    #[test]
    fn test_cuda_kernel_embedding_custom_call_rejects_host_memory() {
        use ryft_core::kernels::{KernelCallOperation, KernelDefinition, KernelParameter};
        use ryft_core::{ArrayType, ReferenceRead, ReferenceWrite};

        let template = definition();
        let parameter = KernelParameter::new(
            ArrayType::scalar(DataType::I32).with_memory(Memory::Host { pinned: true }),
            KernelParameterAccess::ReadWrite,
            template.operation().parameters()[0].mapping().clone(),
        )
        .unwrap();
        let operation = KernelCallOperation::new(template.operation().grid().clone(), vec![parameter]).unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
            references[0].write(&references[0].read()?)?;
            Ok(())
        })
        .unwrap();
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let embedding = CudaKernelEmbedding::new(CUDA_KERNEL_CUSTOM_CALL_TARGET.to_owned(), vec![0]);
        assert!(matches!(
            embedding.custom_call(&verified, &artifact(vec![CudaKernelParameterType::DevicePointer])),
            Err(KernelEmbeddingError::Invalid { message })
                if message == "cuda pointer embedding requires stored device arrays",
        ));
    }

    #[test]
    fn test_cuda_kernel_embedding_custom_call_rejects_physical_mapping() {
        let definition = definition();
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let pointer = artifact(vec![CudaKernelParameterType::DevicePointer]);
        let invalid = CudaKernelEmbedding::new("ryft.test.ready".to_owned(), vec![1]);
        assert!(matches!(invalid.custom_call(&verified, &pointer), Err(KernelEmbeddingError::Invalid { message })
            if message == "cuda pointer mapping must name every logical parameter exactly once"));
        let scalar = artifact(vec![CudaKernelParameterType::Scalar(CudaScalarType::I32)]);
        let embedding = CudaKernelEmbedding::new("ryft.test.ready".to_owned(), vec![0]);
        assert!(matches!(embedding.custom_call(&verified, &scalar), Err(KernelEmbeddingError::Invalid { message })
            if message == "cuda artifact ABI does not match the declared pointer mapping"));
    }
}
