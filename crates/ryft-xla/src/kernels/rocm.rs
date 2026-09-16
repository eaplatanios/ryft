//! Concrete HSACO embedding and session-owned HIP execution through XLA FFI.

use std::sync::{Arc, OnceLock};

use ryft_core::kernels::{GridExecution, KernelParameterAccess, VerifiedKernel};
use ryft_core::operations::custom_call::{CustomCallAttribute, CustomCallOperation};
use ryft_core::{ArrayIrType, DataType, Dimension, EffectClass, Memory, Operation, Typed};
use ryft_pjrt::extensions::ffi::{
    FfiAttribute, FfiCallFrame, FfiError, FfiExecutionStage, FfiHandler, FfiHandlerTraits, FfiInput, FfiOutput,
    FfiTypeId, FfiTypeInformation, FfiUserData, XLA_FFI_CallFrame, XLA_FFI_Error, XLA_FFI_Handler,
};
use ryft_pjrt::{Client, ExecutionContext};
use ryft_rocm::{
    RocmDevicePointer, RocmKernelArtifact, RocmKernelLaunch, RocmKernelLaunchDimensions, RocmKernelLauncher, RocmStream,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::ToPjrt;
use crate::kernels::{
    KernelEmbeddingError, KernelOutputEmbedding, ROCM_KERNEL_CUSTOM_CALL_TARGET, dense_memory_layouts,
};

/// Plugin-owned user-data type registered once with the ROCm handler.
static ROCM_RUNTIME_REGISTRATION: OnceLock<Result<FfiTypeId, ryft_pjrt::Error>> = OnceLock::new();

/// Custom-call buffer supplying one physical HIP pointer argument.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum RocmKernelBufferBinding {
    Input(usize),
    Output(usize),
}

/// Embeds a validated HSACO artifact with one pointer per canonical logical parameter.
///
/// Buffers use dense row-major layouts and canonical declaration order. Read-only parameters use operands and
/// write-only parameters use result buffers. Read-write aliases, scalar prefetch and device assertion transports
/// are rejected. The session owns native modules until every submitted whole-execution fence completes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RocmKernelEmbedding {
    /// Explicitly selected registered handler.
    target_name: String,
}

impl RocmKernelEmbedding {
    /// Selects an existing handler; the built-in handler is [`ROCM_KERNEL_CUSTOM_CALL_TARGET`].
    pub fn new(target_name: String) -> Self {
        Self { target_name }
    }

    /// Reconstructs an artifact and validates its redundant payload and logical signature before loading native code.
    /// The digest detects corruption; it does not authenticate executable code from an untrusted producer.
    pub fn from_custom_call<Extension: Operation<Type = ArrayIrType>>(
        kernel: &VerifiedKernel<'_, Extension>,
        operation: &CustomCallOperation,
    ) -> Result<(Self, RocmKernelArtifact), KernelEmbeddingError> {
        let attribute = |name: &str| {
            let values = operation.attributes().iter().filter(|(key, _)| key == name).collect::<Vec<_>>();
            match values.as_slice() {
                [(_, CustomCallAttribute::String(value))] => Ok(value.as_str()),
                _ => Err(KernelEmbeddingError::Invalid {
                    message: format!("expected exactly one string attribute `{name}`"),
                }),
            }
        };
        let payload = attribute("ryft.rocm.payload")?;
        if payload.len() > RocmEnvelope::MAXIMUM_PAYLOAD_BYTES
            || format!("{:x}", Sha256::digest(payload.as_bytes())) != attribute("ryft.rocm.sha256")?
        {
            return Err(KernelEmbeddingError::Invalid {
                message: "rocm payload exceeds its size limit or has an invalid digest".into(),
            });
        }
        let envelope: RocmEnvelope = serde_json::from_str(payload)?;
        let artifact = envelope.artifact()?;
        let embedding = Self::new(operation.target_name().to_owned());
        let expected = embedding.custom_call(kernel, &artifact)?;
        if operation.effect_class() != expected.effect_class()
            || operation.output_types() != expected.output_types()
            || operation.input_output_aliases() != expected.input_output_aliases()
            || expected.attributes().iter().any(|attribute| !operation.attributes().contains(attribute))
        {
            return Err(KernelEmbeddingError::Invalid {
                message: "rocm custom-call signature differs from its verified kernel".into(),
            });
        }
        Ok((embedding, artifact))
    }
}

impl<Extension: Operation<Type = ArrayIrType>> KernelOutputEmbedding<RocmKernelArtifact, Extension>
    for RocmKernelEmbedding
{
    fn configuration_key(&self) -> Result<Vec<u8>, KernelEmbeddingError> {
        Ok(serde_json::to_vec(&(1u32, &self.target_name))?)
    }

    fn custom_call(
        &self,
        kernel: &VerifiedKernel<'_, Extension>,
        artifact: &RocmKernelArtifact,
    ) -> Result<CustomCallOperation, KernelEmbeddingError> {
        let logical = kernel.definition().operation();
        let effects = kernel.definition().body().effects();
        if effects.has_explicit_ordered_state() {
            return Err(KernelEmbeddingError::UnsupportedEffect { effect: EffectClass::OrderedState });
        }
        for effect in effects.classes() {
            if effect != EffectClass::OrderedState {
                return Err(KernelEmbeddingError::UnsupportedEffect { effect });
            }
        }
        let mut parallel = 1usize;
        for dimension in logical.grid().dimensions() {
            let Dimension::Static(extent) = dimension.extent() else {
                return Err(KernelEmbeddingError::Invalid {
                    message: "rocm embedding requires a static parallel grid".into(),
                });
            };
            if dimension.execution() != GridExecution::Parallel {
                return Err(KernelEmbeddingError::Invalid {
                    message: "rocm embedding requires a static parallel grid".into(),
                });
            }
            parallel = parallel.checked_mul(*extent).filter(|count| *count <= i32::MAX as usize).ok_or_else(|| {
                KernelEmbeddingError::Invalid { message: "rocm grid exceeds the signed native launch range".into() }
            })?;
        }
        if artifact.launch_dimensions().grid() != [parallel.max(1) as u32, 1, 1] {
            return Err(KernelEmbeddingError::Invalid {
                message: "rocm artifact grid differs from the verified parallel grid".into(),
            });
        }

        if !logical.prefetch_types().is_empty() || !logical.aliases().is_empty() {
            return Err(KernelEmbeddingError::Invalid {
                message: "rocm embedding requires specialized prefetch and no read-write aliases".into(),
            });
        }
        if artifact.parameter_count() != logical.parameters().len() {
            return Err(KernelEmbeddingError::Invalid {
                message: "rocm artifact argument count differs from the logical signature".into(),
            });
        }
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut bindings = Vec::new();
        let mut parameter_types = Vec::new();
        for parameter in logical.parameters() {
            let r#type = parameter.r#type();
            if r#type.memory() != Memory::Device
                || r#type.data_type() != DataType::F32
                || r#type.static_shape().is_none()
            {
                return Err(KernelEmbeddingError::Invalid {
                    message: "rocm pointer embedding requires static F32 device arrays".into(),
                });
            }
            parameter_types
                .push((r#type.data_type().to_pjrt().to_string(), r#type.static_shape().unwrap().dimensions().to_vec()));
            match parameter.access() {
                KernelParameterAccess::ReadOnly => {
                    bindings.push(RocmKernelBufferBinding::Input(inputs.len()));
                    inputs.push(r#type.into_owned());
                }
                KernelParameterAccess::WriteOnly => {
                    bindings.push(RocmKernelBufferBinding::Output(outputs.len()));
                    outputs.push(r#type.into_owned());
                }
                KernelParameterAccess::ReadWrite => unreachable!(),
            }
        }
        dense_memory_layouts(&inputs, &outputs, "rocm kernel")?;
        let launch = artifact.launch_dimensions();
        let envelope = RocmEnvelope {
            version: 1,
            image: artifact.image().to_vec(),
            symbol: artifact.entry_name().to_owned(),
            architecture: artifact.target().to_owned(),
            grid: launch.grid(),
            block: launch.block(),
            shared_memory_bytes: launch.dynamic_shared_memory_bytes(),
            argument_bindings: bindings,
            parameter_types,
            input_count: inputs.len(),
            output_count: outputs.len(),
        };
        let payload = serde_json::to_string(&envelope)?;
        if payload.len() > RocmEnvelope::MAXIMUM_PAYLOAD_BYTES {
            return Err(KernelEmbeddingError::Invalid { message: "rocm payload exceeds its size limit".into() });
        }
        Ok(CustomCallOperation::new(self.target_name.clone(), outputs)
            .with_attribute("ryft.rocm.sha256", format!("{:x}", Sha256::digest(payload.as_bytes())))
            .with_attribute("ryft.rocm.payload", payload)
            .with_attribute("ryft.rocm.row_major", true))
    }
}

/// Persistent HIP artifact metadata, reconstructed through the canonical artifact validator.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RocmEnvelope {
    /// Supported payload version.
    version: u32,

    /// Complete HSACO bytes.
    image: Vec<u8>,

    /// Entry symbol.
    symbol: String,

    /// Exact AMD architecture.
    architecture: String,

    /// Physical launch grid.
    grid: [u32; 3],

    /// Physical block dimensions.
    block: [u32; 3],

    /// Dynamic shared memory required by the compiler.
    shared_memory_bytes: u32,

    /// Canonical logical parameter order mapped to FFI buffers.
    argument_bindings: Vec<RocmKernelBufferBinding>,

    /// Element type and static shape checked before native submission.
    parameter_types: Vec<(String, Vec<usize>)>,

    /// Exact operand count.
    input_count: usize,

    /// Exact result count.
    output_count: usize,
}

impl RocmEnvelope {
    /// Bounds JSON expansion before decoding a persisted executable payload.
    const MAXIMUM_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;

    /// Validates canonical argument ordering and reconstructs native metadata before loading.
    fn artifact(&self) -> Result<RocmKernelArtifact, KernelEmbeddingError> {
        if self.version != 1
            || self.argument_bindings.len() != self.parameter_types.len()
            || self.input_count.checked_add(self.output_count) != Some(self.parameter_types.len())
        {
            return Err(KernelEmbeddingError::Invalid {
                message: "invalid rocm embedding schema or parameter count".into(),
            });
        }
        let mut inputs = 0;
        let mut outputs = 0;
        for binding in &self.argument_bindings {
            let valid = match binding {
                RocmKernelBufferBinding::Input(index) => {
                    let valid = *index == inputs;
                    inputs += 1;
                    valid
                }
                RocmKernelBufferBinding::Output(index) => {
                    let valid = *index == outputs;
                    outputs += 1;
                    valid
                }
            };
            if !valid {
                return Err(KernelEmbeddingError::Invalid {
                    message: "rocm buffer bindings do not follow canonical declaration order".into(),
                });
            }
        }
        if inputs != self.input_count || outputs != self.output_count {
            return Err(KernelEmbeddingError::Invalid {
                message: "rocm buffer counts differ from the declared signature".into(),
            });
        }
        Ok(RocmKernelArtifact::new(
            Arc::from(self.image.clone()),
            self.symbol.clone(),
            self.architecture.clone(),
            self.parameter_types.len(),
            RocmKernelLaunchDimensions::new(self.grid, self.block, self.shared_memory_bytes)?,
        )?)
    }
}

/// Stable session allocation borrowed by XLA until all whole-execution fences complete.
pub(crate) struct RocmKernelRuntime {
    /// Concrete bounded HIP module owner.
    launcher: RocmKernelLauncher,

    /// Type registered in the selected plugin's FFI registry.
    type_id: FfiTypeId,
}

impl RocmKernelRuntime {
    /// Creates the HIP owner and idempotently registers the concrete ROCm handler.
    pub(crate) fn new(client: &Client<'_>) -> Result<Self, KernelEmbeddingError> {
        let launcher = RocmKernelLauncher::new()?;
        let type_id = ROCM_RUNTIME_REGISTRATION
            .get_or_init(|| {
                let type_id = client.register_ffi_type(
                    "ryft.kernel.rocm.runtime",
                    FfiTypeId::UNKNOWN,
                    FfiTypeInformation::new(None),
                )?;
                client.register_ffi_handler(
                    ROCM_KERNEL_CUSTOM_CALL_TARGET,
                    "ROCM",
                    FfiHandler::from(rocm_kernel_handler as XLA_FFI_Handler),
                    FfiHandlerTraits::NONE,
                )?;
                Ok(type_id)
            })
            .clone()?;
        Ok(Self { launcher, type_id })
    }

    /// Creates an execution context borrowing this session allocation.
    ///
    /// # Safety
    /// This allocation and its PJRT client must outlive every queued whole-program execution, even if outputs are
    /// dropped. User-data insertion borrows the pointer; the session's retained fences enforce the lifetime.
    pub(crate) unsafe fn execution_context(&self, client: &Client<'_>) -> Result<ExecutionContext, ryft_pjrt::Error> {
        let context = client.execution_context()?;
        unsafe { context.add_ffi_user_data(FfiUserData::new(self.type_id, (self as *const Self).cast_mut().cast())) }?;
        Ok(context)
    }
}

/// Decodes and validates the invocation before borrowing HIP buffer and stream handles.
fn launch_rocm_kernel(frame: &FfiCallFrame<'_>) -> Result<(), FfiError> {
    let mut payload = None;
    let mut digest = None;
    for attribute in frame.attributes() {
        let (name, attribute) = attribute?;
        if name == "ryft.rocm.payload" || name == "ryft.rocm.sha256" {
            let FfiAttribute::String { string } = attribute else {
                return Err(FfiError::invalid_argument("rocm payload attributes must be strings"));
            };
            let slot = if name == "ryft.rocm.payload" { &mut payload } else { &mut digest };
            if slot.replace(string).is_some() {
                return Err(FfiError::invalid_argument("duplicate rocm payload attribute"));
            }
        }
    }
    let payload = payload.ok_or_else(|| FfiError::invalid_argument("missing rocm payload"))?;
    if payload.len() > RocmEnvelope::MAXIMUM_PAYLOAD_BYTES
        || digest != Some(format!("{:x}", Sha256::digest(payload.as_bytes())).as_str())
    {
        return Err(FfiError::invalid_argument("rocm payload exceeds its size limit or has an invalid digest"));
    }
    let envelope: RocmEnvelope =
        serde_json::from_str(payload).map_err(|error| FfiError::invalid_argument(error.to_string()))?;
    let artifact = envelope.artifact().map_err(|error| FfiError::invalid_argument(error.to_string()))?;
    if frame.input_count() != envelope.input_count || frame.output_count() != envelope.output_count {
        return Err(FfiError::invalid_argument("rocm frame buffer counts differ from the declared signature"));
    }
    let context = frame.context()?;
    let type_id = ROCM_RUNTIME_REGISTRATION
        .get()
        .ok_or_else(|| FfiError::internal("rocm runtime is not registered"))?
        .as_ref()
        .map_err(|error| FfiError::internal(error.to_string()))?;
    let data = context.user_data(*type_id)?.data;
    if data.is_null() {
        return Err(FfiError::invalid_argument("missing session rocm runtime"));
    }
    // The only producer is execution_context. Its owning session drains all queued execution fences before drop.
    let runtime = unsafe { &*data.cast::<RocmKernelRuntime>() };
    let mut arguments = Vec::new();
    for (binding, (element_type, dimensions)) in envelope.argument_bindings.iter().zip(&envelope.parameter_types) {
        let buffer = match *binding {
            RocmKernelBufferBinding::Input(index) => {
                let FfiInput::Buffer { buffer } = frame.input(index)?;
                buffer
            }
            RocmKernelBufferBinding::Output(index) => {
                let FfiOutput::Buffer { buffer } = frame.output(index)?;
                buffer
            }
        };
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
            return Err(FfiError::invalid_argument("rocm buffer differs from its declared element type or shape"));
        }
        arguments.push(
            unsafe { RocmDevicePointer::from_raw(buffer.data()) }
                .map_err(|error| FfiError::invalid_argument(error.to_string()))?,
        );
    }
    // XLA supplies current-stream ordering and owns the primary-context buffers. The HIP owner validates capture,
    // device, artifact resources and module lifetime; the session retains its existing execution fence.
    let stream = unsafe { RocmStream::from_raw(context.stream()?) }
        .map_err(|error| FfiError::invalid_argument(error.to_string()))?;
    unsafe { runtime.launcher.launch(&artifact, &RocmKernelLaunch::new(stream, arguments)) }
        .map_err(|error| FfiError::internal(error.to_string()))
}

/// Concrete XLA FFI entry point for the HIP artifact owner.
unsafe extern "C" fn rocm_kernel_handler(frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
    let frame = match unsafe { FfiCallFrame::from_c_api(frame) } {
        Ok(frame) => frame,
        Err(_) => return std::ptr::null_mut(),
    };
    if frame.register_metadata(FfiTypeId::UNKNOWN) || frame.stage() != FfiExecutionStage::Execution {
        return std::ptr::null_mut();
    }
    match launch_rocm_kernel(&frame) {
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

    use ryft_core::{Array, ArrayType};

    use super::*;

    /// Uses the same public portable macro and pointer order as the native vector fixture.
    #[ryft_core::kernels::kernel]
    fn vector(
        #[input(data_type = F32, rank = 1)] left: &Array,
        #[input(data_type = F32, rank = 1)] right: &Array,
        #[output(data_type = F32, shape = [256])] output: &mut Array,
    ) {
        output.store(left.load() + right.load());
    }

    /// Shares the compiler-produced HSACO fixture owned and validated by the concrete ROCm crate.
    fn artifact() -> RocmKernelArtifact {
        RocmKernelArtifact::new(
            Arc::from(include_bytes!("../../../ryft-rocm/src/fixtures/vector-gfx942.hsaco").as_slice()),
            "ryft_kernel",
            "gfx942",
            3,
            RocmKernelLaunchDimensions::new([1; 3], [256, 1, 1], 0).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn test_rocm_kernel_embedding_new() {
        let embedding = RocmKernelEmbedding::new(ROCM_KERNEL_CUSTOM_CALL_TARGET.to_owned());
        assert_eq!(embedding.target_name, ROCM_KERNEL_CUSTOM_CALL_TARGET);
    }

    #[test]
    fn test_rocm_kernel_embedding_from_custom_call() {
        let r#type = ArrayType::new_static(DataType::F32, [256]);
        let definition = vector::definition(&r#type, &r#type).unwrap();
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let embedding = RocmKernelEmbedding::new(ROCM_KERNEL_CUSTOM_CALL_TARGET.to_owned());
        let artifact = artifact();
        let operation = embedding.custom_call(&verified, &artifact).unwrap();
        let (decoded, restored) = RocmKernelEmbedding::from_custom_call(&verified, &operation).unwrap();
        assert_eq!(decoded, embedding);
        assert_eq!(restored.image(), artifact.image());
        assert_eq!(restored.launch_dimensions(), artifact.launch_dimensions());
        assert_eq!(decoded.custom_call(&verified, &restored).unwrap().to_string(), operation.to_string());
        let mut corrupted = CustomCallOperation::new(operation.target_name(), operation.output_types().to_vec());
        for (name, value) in operation.attributes() {
            if name != "ryft.rocm.sha256" {
                corrupted = corrupted.with_attribute(name, value.clone());
            }
        }
        let corrupted = corrupted.with_attribute("ryft.rocm.sha256", "corrupted");
        assert!(matches!(RocmKernelEmbedding::from_custom_call(&verified, &corrupted),
            Err(KernelEmbeddingError::Invalid { message })
                if message == "rocm payload exceeds its size limit or has an invalid digest"));
    }

    #[test]
    fn test_rocm_kernel_embedding_configuration_key() {
        let embedding = RocmKernelEmbedding::new(ROCM_KERNEL_CUSTOM_CALL_TARGET.to_owned());
        assert_eq!(
            <RocmKernelEmbedding as KernelOutputEmbedding<RocmKernelArtifact>>::configuration_key(&embedding).unwrap(),
            br#"[1,"ryft.kernel.rocm"]"#
        );
    }

    #[test]
    fn test_rocm_kernel_embedding_custom_call() {
        let r#type = ArrayType::new_static(DataType::F32, [256]);
        let definition = vector::definition(&r#type, &r#type).unwrap();
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let operation = RocmKernelEmbedding::new(ROCM_KERNEL_CUSTOM_CALL_TARGET.to_owned())
            .custom_call(&verified, &artifact())
            .unwrap();
        assert_eq!(operation.target_name(), ROCM_KERNEL_CUSTOM_CALL_TARGET);
        assert_eq!(operation.output_types(), &[r#type]);
        assert!(operation.input_output_aliases().is_empty());
        assert_eq!(
            operation.attributes().iter().find(|(name, _)| name == "ryft.rocm.row_major"),
            Some(&("ryft.rocm.row_major".to_owned(), CustomCallAttribute::Boolean(true)))
        );
    }

    #[test]
    fn test_rocm_envelope_artifact_rejects_invalid_bindings() {
        let mut envelope = RocmEnvelope {
            version: 1,
            image: vec![],
            symbol: "ryft_kernel".into(),
            architecture: "gfx942".into(),
            grid: [1; 3],
            block: [256, 1, 1],
            shared_memory_bytes: 0,
            argument_bindings: vec![RocmKernelBufferBinding::Input(1)],
            parameter_types: vec![("F32".into(), vec![256])],
            input_count: 1,
            output_count: 0,
        };
        assert!(matches!(envelope.artifact(), Err(KernelEmbeddingError::Invalid { message })
            if message == "rocm buffer bindings do not follow canonical declaration order"));
        envelope.version = 2;
        assert!(matches!(envelope.artifact(), Err(KernelEmbeddingError::Invalid { message })
            if message == "invalid rocm embedding schema or parameter count"));
    }

    #[test]
    fn test_rocm_kernel_embedding_custom_call_rejects_changed_grid() {
        let r#type = ArrayType::new_static(DataType::F32, [256]);
        let definition = vector::definition(&r#type, &r#type).unwrap();
        let artifact = artifact()
            .with_launch_dimensions(RocmKernelLaunchDimensions::new([2, 1, 1], [256, 1, 1], 0).unwrap())
            .unwrap();
        assert!(matches!(RocmKernelEmbedding::new(ROCM_KERNEL_CUSTOM_CALL_TARGET.to_owned())
            .custom_call(&VerifiedKernel::new(&definition, 1).unwrap(), &artifact),
            Err(KernelEmbeddingError::Invalid { message })
                if message == "rocm artifact grid differs from the verified parallel grid"));
    }

    #[test]
    fn test_rocm_kernel_embedding_custom_call_rejects_effects() {
        use ryft_core::kernels::{KernelDefinition, KernelOperation};
        use ryft_core::{ArrayIrValue, ArrayOperation, Placeholder, ProgramBuilder};

        for effect in [EffectClass::OrderedIo, EffectClass::OrderedState, EffectClass::OrderedAssertion] {
            let logical = crate::kernels::tests::definition().operation().clone();
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            builder.add_input(logical.parameters()[0].body_type());
            builder
                .add_instruction(
                    ArrayOperation::<Array>::CustomCall(
                        CustomCallOperation::new("test.external", vec![]).with_effect_class(effect),
                    ),
                    vec![],
                    vec![],
                    None,
                )
                .unwrap();
            let body = builder.build(vec![], vec![Placeholder], vec![]).unwrap();
            let definition = KernelDefinition::new(logical, body).unwrap();
            let verified = VerifiedKernel::new(&definition, 1).unwrap();
            let result =
                RocmKernelEmbedding::new(ROCM_KERNEL_CUSTOM_CALL_TARGET.to_owned()).custom_call(&verified, &artifact());
            assert!(matches!(result,
                Err(KernelEmbeddingError::UnsupportedEffect { effect: actual }) if actual == effect));
        }
    }
}
