//! Typed embedding of compiler outputs into the existing XLA custom call and compilation lifecycle.
//!
//! Compiler outputs retain their adapter ownership. A [`KernelOutputEmbedding`], owned by XLA, validates and
//! translates one concrete output type. [`CompiledKernel`] retains the verified source until that selection is
//! complete and emits the canonical custom call into ordinary staging. Existing XLA lowering, executable caching,
//! persistence, and PJRT fences then own the program. This module does not introduce a second executable or module
//! cache.
//!
//! [`stage_kernel`] preserves the complete kernel call through batching, control flow, and rematerialization.
//! Differentiation requires an explicit [`stage_kernel_with_jvp`], [`stage_kernel_with_vjp`], or pure
//! [`stage_kernel_with_fallback`] contract, or retained custom derivative rules (i.e., [`stage_kernel_with_rules`]).
//! Specialize scalar-prefetched values with [`KernelDefinition::specialize_prefetch`] before XLA staging, because
//! transforms never read captured device buffers back to the host. Local per-shard compilation retains full sharding
//! metadata and requires every manual axis to be bound by an enclosing shard map with the same axis descriptor.
//! Automatic partitioning and collectives are rejected.
//!
//! [`KernelTuningRequest`] defines a finite, fingerprinted schedule search. Call [`KernelTuner::load`] or
//! [`KernelTuner::run`] explicitly, sharing a tuner per device and controlling external contention. Samples measure
//! host submission through actual completion, and any readback owned by the runner must be part of its declared
//! methodology.
//! Cancellation stops further submissions and awaits work already submitted. Measurements do not imply portable
//! performance rankings or timing of device work alone.
//!
//! [`KernelAotBundle`] exports checked portable source, [`KernelCompilationReport`], StableHLO, compatibility facts,
//! and the existing complete executable envelope. Import checks compatibility before native loading and returns a
//! [`LoadedKernel`] using the ordinary asynchronous execution path. Unsupported source codecs fail explicitly, and
//! recompilation after incompatibility is an explicit caller decision. [`DistributedKernel`] coordinates functional
//! calls over an existing distributed runtime using explicit host transfers and the canonical pending completion.
//! See [`distributed`] for ordering, cancellation, topology admission, and publication semantics.
//!
//! The optional `triton` feature adds direct typed TTIR compilation with `TritonEmbedding`. CUDA artifacts reuse
//! [`CudaKernelEmbedding`], and ROCm artifacts use a HIP runtime extension enabled by `rocm`. Both retain the same
//! staging, persistence, and completion path over the whole execution. Triton AOT loading without a compiler
//! reconstructs the binding with recorded compiler configuration, but it still requires a compatible runtime and
//! device. AMD hardware
//! execution remains unqualified, and unsupported portable operations fail before native compilation.

// TODO(eaplatanios): Review this module.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use ryft_core::{
    ArrayIrType, ArrayType, Context, CustomCallOperation, DataType, EffectClass, Layout, Memory, MeshAxis,
    MeshAxisType, Operation as CoreOperation, ProgramError, ShardingDimension, TypeError, Typed,
};
use ryft_kernels::{
    KERNEL_KEY_VERSION, KernelCompiler, KernelDefinition, KernelExtension as PortableKernelExtension, KernelSchedule,
    NoKernelExtension, VerifiedKernel,
};
use ryft_mlir::dialects::stable_hlo::CustomCallMemoryLayouts;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::RuntimeProvider;

mod aot;
mod cuda;
#[cfg(feature = "cutile")]
mod cutile;
pub mod distributed;
#[cfg(feature = "mosaic-gpu")]
mod mosaic;
#[cfg(feature = "rocm")]
mod rocm;
mod staging;
#[cfg(feature = "triton")]
mod triton;
mod tuning;

#[cfg(feature = "cutile")]
pub use cutile::CuTileEmbedding;
#[cfg(feature = "mosaic-gpu")]
pub use mosaic::MosaicGpuEmbedding;

#[cfg(feature = "triton")]
pub use triton::TritonEmbedding;

pub use aot::{KernelAotBundle, KernelAotError, KernelCompilationReport, LoadedKernel};
pub use distributed::{DistributedKernel, DistributedKernelError, DistributedKernelOptions};
pub use tuning::{
    KernelTuner, KernelTuningBudget, KernelTuningError, KernelTuningRequest, KernelTuningResult, KernelTuningRunner,
};

pub(crate) use staging::select_kernels;
pub use staging::{
    KernelCompilerBinding, KernelDeviceFacts, KernelExecutionFacts, KernelExtension, KernelOperation, KernelTarget,
    stage_kernel, stage_kernel_with_fallback, stage_kernel_with_jvp, stage_kernel_with_rules, stage_kernel_with_vjp,
};

#[cfg(feature = "rocm")]
pub use rocm::RocmKernelEmbedding;

pub use cuda::{
    CUDA_KERNEL_CUSTOM_CALL_TARGET, CudaKernelBufferBinding, CudaKernelEmbedding, CudaKernelParameterBinding,
};

// TODO(eaplatanios): Promote kernel embedding errors to dedicated `ryft_xla::Error` variants after
// this support leaves experimental status.
/// Represents errors produced while compiling a kernel or embedding it as an XLA custom call through
/// [`CompiledKernel`].
///
/// Converting to [`crate::Error`] preserves the concrete error in [`crate::Error::Custom`].
#[derive(Debug, Error)]
pub enum KernelEmbeddingError {
    /// PJRT registration or execution context setup failed.
    #[error(transparent)]
    Runtime(#[from] ryft_pjrt::Error),

    /// Canonical CUDA artifact validation failed.
    #[error(transparent)]
    Cuda(#[from] ryft_cuda::Error),

    /// Concrete HIP artifact validation or runtime initialization failed.
    #[cfg(feature = "rocm")]
    #[error(transparent)]
    Rocm(#[from] ryft_rocm::Error),

    /// Versioned payload serialization or parsing failed.
    #[error(transparent)]
    Payload(#[from] serde_json::Error),

    /// Adapter admission or compilation failed, retaining the concrete source chain.
    #[error("kernel compilation failed: {0}")]
    Compiler(#[source] Box<dyn std::error::Error + Send + Sync>),

    /// A canonical type or custom call alias check failed.
    #[error(transparent)]
    Type(#[from] TypeError),

    /// The output's signature, alias, payload, or execution contract is invalid.
    #[error("invalid kernel embedding: {message}")]
    Invalid {
        /// Exact contract mismatch.
        message: String,
    },

    /// This bridge cannot represent an externally observable effect of the kernel.
    #[error("kernel embedding does not support external effect `{effect}`")]
    UnsupportedEffect {
        /// Effect that requires a separate supported token and handler contract.
        effect: EffectClass,
    },
}

// TODO(eaplatanios): Review this module.

// Experimental kernel integrations retain their concrete error payloads in the common custom carrier.
impl From<KernelEmbeddingError> for crate::Error {
    fn from(error: KernelEmbeddingError) -> Self {
        Self::custom(error)
    }
}

impl From<ryft_kernels::KernelError> for crate::Error {
    fn from(error: ryft_kernels::KernelError) -> Self {
        Self::custom(error)
    }
}

/// Typed FFI target used by persisted ROCm kernel calls.
pub const ROCM_KERNEL_CUSTOM_CALL_TARGET: &str = "ryft.kernel.rocm";

/// Conversion, owned by XLA, of one compiler output owned by an adapter into a canonical custom call.
///
/// Implementations are trusted integration code, so they must validate the output's version, physical argument map,
/// required buffers, target requirements, and serialized payload before returning. An adapter output never needs to
/// depend on this trait, because the XLA integration implements the trait for its own bridge over the foreign output
/// type. Ready and deferred outputs use the same boundary, and a deferred bridge additionally validates its native
/// decoder and compiler input schema. Registration and target capability checks must precede native loading or
/// execution.
///
/// [`CompiledKernel`] independently validates the logical signature, operation-local aliases, and effects, adds
/// semantic and compiler identity, and discharges only the reference state that the definition proves local. A bridge
/// cannot request external reference state slots or alter the kernel's logical inputs and results. An embedding
/// implementing native checked assertions must return precisely [`EffectClass::OrderedAssertion`], preserving their
/// token chain, because a pure call cannot discharge or silently discard an observable assertion.
pub trait KernelOutputEmbedding<Output, Extension: CoreOperation<Type = ArrayIrType> = NoKernelExtension> {
    /// Returns deterministic bytes covering target names, physical mappings, payload schemas, and plugin ABI facts.
    fn configuration_key(&self) -> Result<Vec<u8>, KernelEmbeddingError>;

    /// Validates the typed output and builds its complete target call, including payload attributes.
    fn custom_call(
        &self,
        kernel: &VerifiedKernel<'_, Extension>,
        output: &Output,
    ) -> Result<CustomCallOperation, KernelEmbeddingError>;
}

/// Selected compiler output together with the immutable source definition and its canonical custom call embedding.
///
/// Selection is explicit, and no constructor falls back to another compiler. The retained definition prevents an opaque
/// payload from masquerading as an unverified kernel. Binding emits the selected custom call only after validating
/// the exact logical inputs. Its attributes participate in existing StableHLO-based compilation and persistence keys.
#[derive(Clone, Debug)]
pub struct CompiledKernel<Extension: CoreOperation<Type = ArrayIrType> = NoKernelExtension> {
    /// Immutable body and boundary that were verified before compilation and embedding.
    definition: KernelDefinition<Extension>,

    /// Canonical array custom call selected for this exact definition and compiler configuration.
    custom_call: CustomCallOperation,
}

impl<Extension> CompiledKernel<Extension>
where
    Extension: PortableKernelExtension,
{
    /// Admits and compiles the verified definition with one explicitly selected adapter, then embeds its typed output.
    /// Compiler configuration is recorded even when it does not alter the emitted native bytes.
    pub fn from_compiler<C, B>(
        kernel: &VerifiedKernel<'_, Extension>,
        compiler: &C,
        target: &C::Target,
        options: &C::Options,
        schedule: &KernelSchedule,
        embedding: &B,
    ) -> Result<Self, KernelEmbeddingError>
    where
        C: KernelCompiler<Extension, Error: 'static + Send + Sync>,
        B: KernelOutputEmbedding<C::Output, Extension>,
    {
        let output = kernel
            .compile(compiler, target, options, schedule)
            .map_err(|error| KernelEmbeddingError::Compiler(Box::new(error)))?;
        let configuration = compiler
            .configuration_key(target, options, schedule)
            .map_err(|error| KernelEmbeddingError::Compiler(Box::new(error)))?;
        Self::from_output(kernel, &configuration, &output, embedding)
    }

    /// Embeds an already available typed output, permitting use with only a runtime when its native format supports
    /// it. `configuration` must be the complete key emitted by the compiler that produced `output`, and an integration
    /// must validate persisted adapter, schema, and plugin compatibility before invoking this constructor on reloaded
    /// output. The embedding configuration is independently included in the resulting custom call identity.
    pub fn from_output<Output, B: KernelOutputEmbedding<Output, Extension>>(
        kernel: &VerifiedKernel<'_, Extension>,
        configuration: &[u8],
        output: &Output,
        embedding: &B,
    ) -> Result<Self, KernelEmbeddingError> {
        // The immutable verified definition owns every body reference input and forbids reference constants,
        // captures, consumption, and escaping references. Removing only this local OrderedState is semantic
        // reference discharge at the operation boundary. External effects need a separately supported token ABI.
        // The effects of the definition include the assertions of mappings that may fail, which the custom call
        // must preserve like assertions of the body.
        let effects = kernel.definition().effects();
        if effects.has_explicit_ordered_state() {
            return Err(KernelEmbeddingError::UnsupportedEffect { effect: EffectClass::OrderedState });
        }
        let mut required_effect = None;
        for effect in effects.classes() {
            match effect {
                EffectClass::OrderedState => (),
                EffectClass::OrderedAssertion => required_effect = Some(effect),
                _ => return Err(KernelEmbeddingError::UnsupportedEffect { effect }),
            }
        }
        let mut custom_call = embedding.custom_call(kernel, output)?;
        // An assertion is observable even when every array result is unused. A trusted embedding must implement
        // that exact effect with its native handler and preserve the canonical assertion token chain.
        if custom_call.effect_class() != required_effect {
            return Err(KernelEmbeddingError::UnsupportedEffect {
                effect: required_effect.or(custom_call.effect_class()).unwrap(),
            });
        }
        let logical = kernel.definition().operation();
        let expected = logical.output_types();
        let actual = custom_call.output_types().iter().cloned().map(ArrayIrType::Array).collect::<Vec<_>>();
        if actual != expected {
            return Err(KernelEmbeddingError::Invalid {
                message: "compiler output types differ from the logical kernel results".to_owned(),
            });
        }
        let aliases = custom_call
            .input_output_aliases()
            .iter()
            .map(|alias| (alias.output_index(), alias.input_index()))
            .collect::<Vec<_>>();
        if aliases != logical.aliases() {
            return Err(KernelEmbeddingError::Invalid {
                message: "compiler aliases differ from the logical kernel aliases".to_owned(),
            });
        }
        if custom_call.target_name().is_empty() || custom_call.target_name().contains('\0') {
            return Err(KernelEmbeddingError::Invalid {
                message: "custom-call target must be nonempty and contain no NUL".to_owned(),
            });
        }
        let mut names = BTreeSet::new();
        for (name, _) in custom_call.attributes() {
            if name.starts_with("ryft.kernel.") || !names.insert(name) {
                return Err(KernelEmbeddingError::Invalid {
                    message: format!("duplicate or reserved custom-call attribute `{name}`"),
                });
            }
        }
        let embedding_configuration = embedding.configuration_key()?;
        let mut configuration_hash = Sha256::new();
        for component in [configuration, embedding_configuration.as_slice()] {
            configuration_hash.update((component.len() as u64).to_le_bytes());
            configuration_hash.update(component);
        }
        custom_call = custom_call
            .with_attribute("ryft.kernel.schema", i64::from(KERNEL_KEY_VERSION))
            .with_attribute(
                "ryft.kernel.semantic",
                hex::encode(Sha256::digest(kernel.definition().semantic_key()?.as_bytes())),
            )
            .with_attribute("ryft.kernel.configuration", hex::encode(configuration_hash.finalize()));
        let provenance = kernel
            .definition()
            .body()
            .regions()
            .iter()
            .flat_map(|region| region.instructions())
            .filter(|instruction| !instruction.provenance().is_unknown())
            .map(|instruction| instruction.provenance().to_string())
            .filter(|origin| !origin.is_empty())
            .collect::<BTreeSet<_>>();
        if !provenance.is_empty() {
            custom_call = custom_call.with_attribute("ryft.kernel.provenance", serde_json::to_string(&provenance)?);
        }
        let input_types = logical
            .input_types()
            .iter()
            .map(|value| <&ArrayType>::try_from(value).cloned())
            .collect::<Result<Vec<_>, _>>()?;
        custom_call.infer_output_types(&input_types, &[])?;
        Ok(Self { definition: kernel.definition().clone(), custom_call })
    }

    /// Returns the immutable source retained across adapter selection.
    pub fn definition(&self) -> &KernelDefinition<Extension> {
        &self.definition
    }

    /// Returns the selected canonical custom call, including deterministic semantic and compiler configuration keys.
    pub fn custom_call(&self) -> &CustomCallOperation {
        &self.custom_call
    }

    /// Binds the selected call through ordinary staging or an existing compilation domain after checking the inputs
    /// against the kernel signature. The inputs may be placed, like those of the kernel call that this call replaces
    /// (refer to [`KernelCallOperation`](ryft_kernels::KernelCallOperation)), and the results have the types that the
    /// kernel call infers for them. External reference state arguments are absent, and all operands and outputs are
    /// ordinary array values.
    pub fn bind<C>(&self, context: &C, inputs: &[C::Value]) -> Result<Vec<C::Value>, ProgramError>
    where
        C: Context<Type = ArrayIrType, Operation: From<CustomCallOperation>>,
    {
        let actual = inputs.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
        let custom_call = placed_custom_call(self.definition.operation(), &self.custom_call, &actual)?;
        context.bind(custom_call, Vec::new(), inputs)
    }
}

/// Returns the physical `custom_call` that a compiler embedding selected for the kernel call `operation`, declared for
/// a call on inputs of `input_types`, which may be placed.
///
/// The result of a read-write parameter is the storage of its input, and a custom call requires an aliased result to
/// have exactly the type of the input that it aliases. A kernel call types that result accordingly, so the returned
/// call declares every aliased result with the type of its input. Every other result keeps its physical declaration,
/// from which a custom call infers the same type as the kernel call. The returned call is checked by inferring its
/// results on `input_types` and comparing them with those of the kernel call, so a selected call always has the
/// result types of the call that it replaces.
///
/// # Errors
///
/// Returns the [`TypeError`] of [`KernelCallOperation::output_types_for`](ryft_kernels::KernelCallOperation) for
/// inputs that the kernel call rejects, and a [`TypeError`] when the custom call infers other result types than the
/// kernel call.
pub(crate) fn placed_custom_call(
    operation: &ryft_kernels::KernelCallOperation,
    custom_call: &CustomCallOperation,
    input_types: &[ArrayIrType],
) -> Result<CustomCallOperation, TypeError> {
    let logical = operation.output_types_for(input_types)?;
    let mut output_types = custom_call.output_types().to_vec();
    for alias in custom_call.input_output_aliases() {
        if let (Some(output), Some(ArrayIrType::Array(r#type))) =
            (output_types.get_mut(alias.output_index()), logical.get(alias.output_index()))
        {
            // The physical layout of the declaration belongs to the embedding and is kept.
            *output = r#type.clone().with_layout(output.layout().cloned());
        }
    }
    let placed = if output_types == custom_call.output_types() {
        custom_call.clone()
    } else {
        let mut placed = CustomCallOperation::new(custom_call.target_name(), output_types)
            .with_input_layouts(custom_call.input_layouts().iter().cloned())
            .with_batching(custom_call.batching());
        for (name, value) in custom_call.attributes() {
            placed = placed.with_attribute(name.clone(), value.clone());
        }
        for alias in custom_call.input_output_aliases() {
            placed = placed.with_input_output_alias(alias.input_index(), alias.output_index())?;
        }
        match custom_call.effect_class() {
            Some(effect_class) => placed.with_effect_class(effect_class),
            None => placed,
        }
    };
    let array_types = |types: &[ArrayIrType]| {
        types
            .iter()
            .filter_map(|r#type| match r#type {
                ArrayIrType::Array(r#type) => Some(r#type.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    if CoreOperation::infer_output_types(&placed, &array_types(input_types), &[])? != array_types(&logical) {
        return Err(TypeError::invalid(
            "selected kernel custom call does not have the result types of the kernel call that it replaces",
        ));
    }
    Ok(placed)
}

/// Returns available native kernel providers without initializing their drivers.
pub(crate) fn runtime_providers() -> Vec<Arc<dyn RuntimeProvider>> {
    #[cfg(feature = "rocm")]
    {
        vec![cuda::runtime_provider(), rocm::runtime_provider()]
    }
    #[cfg(not(feature = "rocm"))]
    {
        vec![cuda::runtime_provider()]
    }
}

/// Validates the native memref ABI and fixes complete row-major operand and result layouts, including default types.
pub(crate) fn dense_memory_layouts(
    inputs: &[ArrayType],
    outputs: &[ArrayType],
    owner: &str,
) -> Result<CustomCallMemoryLayouts, KernelEmbeddingError> {
    for r#type in inputs.iter().chain(outputs) {
        if r#type.memory() != Memory::Device
            || r#type.static_shape().is_none()
            || matches!(r#type.data_type(), DataType::Zero | DataType::Token)
        {
            return Err(KernelEmbeddingError::Invalid {
                message: format!("{owner} requires static device array buffers"),
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
                    message: format!("{owner} requires untiled dense row-major array layouts"),
                });
            }
        }
    }
    Ok(CustomCallMemoryLayouts {
        operands: inputs.iter().map(|r#type| (0..r#type.rank()).rev().collect()).collect(),
        results: outputs.iter().map(|r#type| (0..r#type.rank()).rev().collect()).collect(),
    })
}

/// Requires the array inputs of a kernel call to be manual shards that are already local, because adapter selection
/// never synthesizes partitioning or a collective. The parameter types of a kernel are physical, so `inputs` are the
/// types of the values that the call receives. Every diagnostic names the input and what keeps it from being local.
pub(crate) fn validate_kernel_sharding(
    inputs: &[ArrayType],
    bound_axes: &BTreeMap<String, MeshAxis>,
) -> Result<(), KernelEmbeddingError> {
    let invalid = |message: String| KernelEmbeddingError::Invalid { message };
    for (index, input) in inputs.iter().enumerate() {
        let Some(sharding) = input.sharding() else { continue };
        if !sharding.unreduced_axes().is_empty() || !sharding.reduced_axes().is_empty() {
            return Err(invalid(format!(
                "kernel input {index} carries a pending reduction, which a kernel cannot complete"
            )));
        }
        let mut axes = sharding.varying_manual_axes().clone();
        for dimension in sharding.dimensions() {
            match dimension {
                ShardingDimension::Replicated => {}
                ShardingDimension::Sharded(names) => axes.extend(names.iter().cloned()),
                ShardingDimension::Unconstrained => {
                    return Err(invalid(format!(
                        "kernel input {index} has an unconstrained dimension placement, so it is not known to be a \
                         local shard"
                    )));
                }
            }
        }
        if let Some(axis) = axes.iter().find(|axis| {
            sharding.mesh().axis_type(axis) != Some(MeshAxisType::Manual)
                || sharding.mesh().axis_index(axis).map(|index| &sharding.mesh().axes()[index]) != bound_axes.get(*axis)
        }) {
            return Err(invalid(unbound_kernel_axis_message(index, axis)));
        }
    }
    Ok(())
}

/// Returns the diagnostic for kernel input `index` that is placed on or varies along the mesh axis `axis`, which no
/// enclosing manual region binds. One device cannot run a kernel on such an input, because it holds only a part of
/// it.
pub(crate) fn unbound_kernel_axis_message(index: usize, axis: &str) -> String {
    format!(
        "kernel input {index} is placed on or varies along mesh axis `{axis}`, which no enclosing `shard_map` binds, \
         so one device holds only a part of it. Replicate the input along `{axis}` before the call, or apply the \
         kernel inside a `shard_map` over `{axis}`"
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use std::convert::Infallible;
    use std::error::Error as StandardError;

    use indoc::{formatdoc, indoc};
    use pretty_assertions::assert_eq;
    use ryft_core::{
        Array as CoreArray, ArrayIrOperation, ArrayIrValue, ArrayOperation, Context, DataType, Dimension,
        DimensionBounds, DimensionSubOperation, DimensionType, DimensionValue, EffectClasses, LogicalMesh, MeshAxis,
        Placeholder, ProgramBuilder as CoreProgramBuilder, ReferenceRead, ReferenceWrite, Sharding, ShardingError,
        TiledLayout, Tracer as CoreTracer, TracingContext,
    };
    use ryft_kernels::{
        BlockMapping, BoundaryPolicy, Grid, GridDimension, GridExecution, KernelCallOperation, KernelCompilationError,
        KernelOperation as PortableKernelOperation, KernelParameter, KernelParameterAccess,
    };

    use crate::lowering::lower_mlir_module_for_program;
    use crate::{Constant, Operation, ProgramBuilder};

    use super::*;

    /// A scalar identity with a functional read-write alias and no external reference slot.
    pub(crate) fn definition() -> KernelDefinition {
        let mapping = BlockMapping::new(
            CoreProgramBuilder::<ArrayIrValue<CoreArray>, ArrayIrOperation<CoreArray>>::new()
                .build(vec![], vec![], vec![])
                .unwrap(),
            vec![],
            BoundaryPolicy::InBounds,
        )
        .unwrap();
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                KernelParameter::new(ArrayType::scalar(DataType::I32), KernelParameterAccess::ReadWrite, mapping)
                    .unwrap(),
            ],
        )
        .unwrap();
        KernelDefinition::trace(operation, |(references, _)| {
            references[0].write(&references[0].read()?)?;
            Ok(())
        })
        .unwrap()
    }

    /// An in-place copy over two grid points whose window starts at `1 - coordinate`. The coordinate type admits
    /// values up to 3, so the mapping requires a runtime assertion, although no point of the grid fails it.
    pub(crate) fn mapping_assertion_definition() -> KernelDefinition {
        let coordinate = DimensionType::new("coordinate", DimensionBounds::non_negative(Some(4)).unwrap());
        let mut mapping = CoreProgramBuilder::<ArrayIrValue<CoreArray>, ArrayIrOperation<CoreArray>>::new();
        let input = mapping.add_input(coordinate.clone().into());
        let one = DimensionValue::constant(1).unwrap();
        let start = mapping.add_constant(ArrayIrValue::Dimension(one.clone()));
        let operation = DimensionSubOperation::new(&one.r#type(), &coordinate).unwrap();
        let start = mapping.add_instruction(operation, vec![], vec![start, input], None).unwrap()[0];
        let mapping = BlockMapping::new(
            mapping.build(vec![start], vec![Placeholder], vec![Placeholder]).unwrap(),
            vec![1],
            BoundaryPolicy::InBounds,
        )
        .unwrap();
        let operation = KernelCallOperation::new(
            Grid::new(vec![GridDimension::new(Dimension::Static(2), GridExecution::Parallel)]).unwrap(),
            vec![
                KernelParameter::new(
                    ArrayType::new_static(DataType::I32, vec![2]),
                    KernelParameterAccess::ReadWrite,
                    mapping,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        KernelDefinition::trace(operation, |(references, _)| {
            references[0].write(&references[0].read()?)?;
            Ok(())
        })
        .unwrap()
    }

    /// A versioned deferred compiler input owned independently of any XLA representation.
    struct DeferredCopy {
        /// Exact decoder schema version.
        schema: u32,
    }

    /// XLA decoder integration for the fixture's explicit deferred schema.
    struct DeferredEmbedding;

    impl KernelOutputEmbedding<DeferredCopy> for DeferredEmbedding {
        fn configuration_key(&self) -> Result<Vec<u8>, KernelEmbeddingError> {
            Ok(b"ryft.test.deferred_copy.schema1".to_vec())
        }

        fn custom_call(
            &self,
            kernel: &VerifiedKernel<'_>,
            output: &DeferredCopy,
        ) -> Result<CustomCallOperation, KernelEmbeddingError> {
            if output.schema != 1 {
                return Err(KernelEmbeddingError::Invalid {
                    message: format!("unsupported deferred copy schema `{}`", output.schema),
                });
            }
            let logical = kernel.definition().operation();
            if logical.parameters().len() != 1 || logical.parameters()[0].access() != KernelParameterAccess::ReadWrite {
                return Err(KernelEmbeddingError::Invalid {
                    message: "deferred copy expects one read-write parameter".to_owned(),
                });
            }
            Ok(CustomCallOperation::new("ryft.test.deferred_copy", vec![logical.parameters()[0].r#type().into_owned()])
                .with_attribute("deferred.schema", i64::from(output.schema))
                .with_input_output_alias(0, 0)?)
        }
    }

    /// Compiler that returns a typed deferred payload, exercising ordinary admission and configuration identity.
    struct Compiler;

    impl KernelCompiler for Compiler {
        type Target = bool;
        type Options = u32;
        type Output = DeferredCopy;
        type Error = Infallible;

        fn admit(
            &self,
            kernel: &VerifiedKernel<'_>,
            target: &bool,
            options: &u32,
            schedule: &KernelSchedule,
        ) -> Result<(), KernelCompilationError<Infallible>> {
            let _ = (kernel, options, schedule);
            if *target {
                Ok(())
            } else {
                Err(KernelCompilationError::Unavailable { message: "fixture target unavailable".to_owned() })
            }
        }

        fn configuration_key(
            &self,
            target: &bool,
            options: &u32,
            schedule: &KernelSchedule,
        ) -> Result<Vec<u8>, KernelCompilationError<Infallible>> {
            let _ = (target, schedule);
            Ok(options.to_le_bytes().to_vec())
        }

        fn compile(
            &self,
            kernel: &VerifiedKernel<'_>,
            target: &bool,
            options: &u32,
            schedule: &KernelSchedule,
        ) -> Result<DeferredCopy, KernelCompilationError<Infallible>> {
            let _ = (kernel, target, options, schedule);
            Ok(DeferredCopy { schema: 1 })
        }
    }

    #[test]
    fn test_error_from_kernel_embedding_error() {
        let error =
            crate::Error::from(KernelEmbeddingError::Compiler(Box::new(std::io::Error::other("compiler failed"))));
        assert_eq!(error.to_string(), "kernel compilation failed: compiler failed");
        let kernel = error.downcast_custom::<KernelEmbeddingError>().unwrap();
        assert!(matches!(kernel, KernelEmbeddingError::Compiler(_)));
        assert_eq!(kernel.source().unwrap().to_string(), "compiler failed");
    }

    #[test]
    fn test_error_from_kernel_error() {
        let error = crate::Error::from(ryft_kernels::KernelError::ParameterRank { array_rank: 3, block_rank: 2 });
        assert_eq!(error.to_string(), "kernel parameter has array rank 3 but block rank 2");
        assert_eq!(
            error.downcast_custom::<ryft_kernels::KernelError>(),
            Some(&ryft_kernels::KernelError::ParameterRank { array_rank: 3, block_rank: 2 }),
        );
    }

    #[test]
    fn test_compiled_kernel_from_compiler() {
        let definition = definition();
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let schedule = KernelSchedule::default();
        let first =
            CompiledKernel::from_compiler(&verified, &Compiler, &true, &1, &schedule, &DeferredEmbedding).unwrap();
        let second =
            CompiledKernel::from_compiler(&verified, &Compiler, &true, &2, &schedule, &DeferredEmbedding).unwrap();
        assert_eq!(first.definition().semantic_key().unwrap(), second.definition().semantic_key().unwrap());
        assert_ne!(first.custom_call().to_string(), second.custom_call().to_string());
        assert_eq!(
            first.definition().body().effects().classes().into_iter().collect::<Vec<_>>(),
            vec![EffectClass::OrderedState]
        );
        assert!(first.custom_call().effects().classes().is_empty());
        assert!(matches!(
            CompiledKernel::from_compiler(&verified, &Compiler, &false, &1, &schedule, &DeferredEmbedding),
            Err(KernelEmbeddingError::Compiler(error))
                if error.to_string() == "kernel compiler is unavailable: fixture target unavailable",
        ));
    }

    #[test]
    fn test_compiled_kernel_from_output() {
        /// Changes only the embedding ABI identity, preserving the emitted payload and logical call.
        struct Embedding(&'static [u8]);

        impl KernelOutputEmbedding<DeferredCopy> for Embedding {
            fn configuration_key(&self) -> Result<Vec<u8>, KernelEmbeddingError> {
                Ok(self.0.to_vec())
            }

            fn custom_call(
                &self,
                kernel: &VerifiedKernel<'_>,
                output: &DeferredCopy,
            ) -> Result<CustomCallOperation, KernelEmbeddingError> {
                DeferredEmbedding.custom_call(kernel, output)
            }
        }

        let definition = definition();
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let output = DeferredCopy { schema: 1 };
        let first = CompiledKernel::from_output(&verified, b"a", &output, &Embedding(b"bc")).unwrap();
        let repeated = CompiledKernel::from_output(&verified, b"a", &output, &Embedding(b"bc")).unwrap();
        let changed = CompiledKernel::from_output(&verified, b"a", &output, &Embedding(b"bd")).unwrap();
        let repartitioned = CompiledKernel::from_output(&verified, b"ab", &output, &Embedding(b"c")).unwrap();
        assert_eq!(first.custom_call().to_string(), repeated.custom_call().to_string());
        assert_ne!(first.custom_call().to_string(), changed.custom_call().to_string());
        assert_ne!(first.custom_call().to_string(), repartitioned.custom_call().to_string());
        let compiled = CompiledKernel::from_compiler(
            &verified,
            &Compiler,
            &true,
            &1,
            &KernelSchedule::default(),
            &Embedding(b"bc"),
        )
        .unwrap();
        let embedded = CompiledKernel::from_output(&verified, &1u32.to_le_bytes(), &output, &Embedding(b"bc")).unwrap();
        assert_eq!(compiled.custom_call().to_string(), embedded.custom_call().to_string());
    }

    #[test]
    fn test_compiled_kernel_from_output_provenance() {
        let operation = definition().operation().clone();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
            references[0]
                .context()
                .invoke_with_provenance_scope(ryft_core::ProvenanceScope::new("copy_output"), || {
                    references[0].write(&references[0].read()?)
                })
        })
        .unwrap();
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let compiled =
            CompiledKernel::from_output(&verified, b"configuration", &DeferredCopy { schema: 1 }, &DeferredEmbedding)
                .unwrap();
        assert_eq!(
            compiled.custom_call().to_string(),
            indoc! {r#"
                custom_call [
                    target=ryft.test.deferred_copy,
                    deferred.schema=1,
                    ryft.kernel.schema=1,
                    ryft.kernel.semantic="c0aace572db2a32780947575541c582a8c0e012a86dccb7f08be2cffcc1a97b0",
                    ryft.kernel.configuration="9b504f8715e50a1265774108552c85c4822b04e39355818c680592419a1e0f7d",
                    ryft.kernel.provenance="[\"copy_output\"]",
                    input_output_alias=0->0,
                ]
            "#}
            .trim_end(),
        );
        let unknown = super::tests::definition();
        let unknown = VerifiedKernel::new(&unknown, 1).unwrap();
        let compiled =
            CompiledKernel::from_output(&unknown, b"configuration", &DeferredCopy { schema: 1 }, &DeferredEmbedding)
                .unwrap();
        assert_eq!(
            compiled.custom_call().to_string(),
            indoc! {r#"
                custom_call [
                    target=ryft.test.deferred_copy,
                    deferred.schema=1,
                    ryft.kernel.schema=1,
                    ryft.kernel.semantic="c0aace572db2a32780947575541c582a8c0e012a86dccb7f08be2cffcc1a97b0",
                    ryft.kernel.configuration="9b504f8715e50a1265774108552c85c4822b04e39355818c680592419a1e0f7d",
                    input_output_alias=0->0,
                ]
            "#}
            .trim_end(),
        );
    }

    #[test]
    fn test_compiled_kernel_from_output_rejects_deferred_schema() {
        let definition = definition();
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        assert!(matches!(
            CompiledKernel::from_output(&verified, b"configuration", &DeferredCopy { schema: 2 }, &DeferredEmbedding),
            Err(KernelEmbeddingError::Invalid { message }) if message == "unsupported deferred copy schema `2`",
        ));
    }

    #[test]
    fn test_compiled_kernel_from_output_rejects_changed_aliases_and_effects() {
        /// Supplies an intentionally mismatched typed embedding to test the independent logical checks.
        struct ChangedEmbedding(bool);

        impl KernelOutputEmbedding<()> for ChangedEmbedding {
            fn configuration_key(&self) -> Result<Vec<u8>, KernelEmbeddingError> {
                Ok(b"ryft.test.changed.schema1".to_vec())
            }

            fn custom_call(
                &self,
                kernel: &VerifiedKernel<'_>,
                output: &(),
            ) -> Result<CustomCallOperation, KernelEmbeddingError> {
                let _ = (kernel, output);
                let operation = CustomCallOperation::new("test.changed", vec![ArrayType::scalar(DataType::I32)]);
                if self.0 {
                    Ok(operation.with_input_output_alias(0, 0)?.with_effect_class(EffectClass::OrderedIo))
                } else {
                    Ok(operation)
                }
            }
        }
        let definition = definition();
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        assert!(matches!(
            CompiledKernel::from_output(&verified, b"config", &(), &ChangedEmbedding(false)),
            Err(KernelEmbeddingError::Invalid { message })
                if message == "compiler aliases differ from the logical kernel aliases",
        ));
        assert!(matches!(
            CompiledKernel::from_output(&verified, b"config", &(), &ChangedEmbedding(true)),
            Err(KernelEmbeddingError::UnsupportedEffect { effect: EffectClass::OrderedIo })
        ));
    }

    #[test]
    fn test_compiled_kernel_from_output_rejects_external_body_effects() {
        for effect in [EffectClass::OrderedIo, EffectClass::OrderedState, EffectClass::OrderedAssertion] {
            let logical = definition().operation().clone();
            let mut builder = CoreProgramBuilder::<ArrayIrValue<CoreArray>, PortableKernelOperation>::new();
            builder.add_input(logical.parameters()[0].body_type());
            builder
                .add_instruction(
                    ArrayOperation::<CoreArray>::CustomCall(
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
            assert!(matches!(
                CompiledKernel::from_output(&verified, b"config", &DeferredCopy { schema: 1 }, &DeferredEmbedding),
                Err(KernelEmbeddingError::UnsupportedEffect { effect: actual }) if actual == effect,
            ));
        }
    }

    #[test]
    fn test_compiled_kernel_from_output_mapping_assertions() {
        /// Implements the assertion of the mapping with a native handler and the canonical assertion token chain.
        struct AssertingEmbedding;

        impl KernelOutputEmbedding<DeferredCopy> for AssertingEmbedding {
            fn configuration_key(&self) -> Result<Vec<u8>, KernelEmbeddingError> {
                Ok(b"ryft.test.asserting_copy.schema1".to_vec())
            }

            fn custom_call(
                &self,
                kernel: &VerifiedKernel<'_>,
                output: &DeferredCopy,
            ) -> Result<CustomCallOperation, KernelEmbeddingError> {
                Ok(DeferredEmbedding.custom_call(kernel, output)?.with_effect_class(EffectClass::OrderedAssertion))
            }
        }

        let definition = mapping_assertion_definition();
        assert_eq!(definition.body().effects().classes(), EffectClasses::single(EffectClass::OrderedState));
        let assertion = EffectClasses::single(EffectClass::OrderedAssertion);
        assert_eq!(definition.effects().classes(), assertion.union(EffectClasses::single(EffectClass::OrderedState)));
        let verified = VerifiedKernel::new(&definition, 2).unwrap();
        let output = DeferredCopy { schema: 1 };

        // An embedding without the assertion would discard an observable failure of the mapping.
        assert!(matches!(
            CompiledKernel::from_output(&verified, b"configuration", &output, &DeferredEmbedding),
            Err(KernelEmbeddingError::UnsupportedEffect { effect: EffectClass::OrderedAssertion }),
        ));
        let compiled = CompiledKernel::from_output(&verified, b"configuration", &output, &AssertingEmbedding).unwrap();
        assert_eq!(compiled.custom_call().effects().classes(), assertion);

        // The emitted custom call is ordered through the assertion token chain.
        let vector = ArrayType::new_static(DataType::I32, vec![2]);
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(vector.clone().into());
        builder.add_instruction(compiled.custom_call().clone(), vec![], vec![input], None).unwrap();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(program.clone().into_simplified().unwrap().to_string(), program.to_string());
        let lowered = lower_mlir_module_for_program(
            &program,
            &[],
            &vec![vector.clone()],
            &vec![vector],
            "main",
            None,
            None,
            None,
        )
        .unwrap();
        let (module, _signature, requires_assertion_handler, _requirements) = lowered.into_parts();
        assert!(requires_assertion_handler);
        let semantic = hex::encode(Sha256::digest(definition.semantic_key().unwrap().as_bytes()));
        let configuration = "e0d25cb2744279b74773008d0f4c001169484be52de590f2f2ce90c699015e4a";
        assert_eq!(
            module,
            formatdoc! {r#"
            module {{
              func.func @main(%arg0: tensor<2xi32>) -> tensor<2xi32> {{
                %0 = stablehlo.after_all  : !stablehlo.token
                %1:2 = stablehlo.custom_call @ryft.test.deferred_copy(%arg0, %0) {{api_version = 4 : i32, backend_config = {{deferred.schema = 1 : i64, ryft.kernel.configuration = "{configuration}", ryft.kernel.schema = 1 : i64, ryft.kernel.semantic = "{semantic}"}}, has_side_effect = true, operand_layouts = [dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]}} : (tensor<2xi32>, !stablehlo.token) -> (tensor<2xi32>, !stablehlo.token)
                return %arg0 : tensor<2xi32>
              }}
            }}
        "#}
        );
    }

    #[test]
    fn test_compiled_kernel_from_output_rejects_invalid_custom_calls() {
        /// Returns a fixed custom call so that each test case isolates one boundary check.
        struct FixedEmbedding(CustomCallOperation);

        impl KernelOutputEmbedding<()> for FixedEmbedding {
            fn configuration_key(&self) -> Result<Vec<u8>, KernelEmbeddingError> {
                Ok(b"ryft.test.fixed.schema1".to_vec())
            }

            fn custom_call(
                &self,
                kernel: &VerifiedKernel<'_>,
                output: &(),
            ) -> Result<CustomCallOperation, KernelEmbeddingError> {
                let _ = (kernel, output);
                Ok(self.0.clone())
            }
        }

        let definition = definition();
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let scalar = ArrayType::scalar(DataType::I32);
        let changed_output = CustomCallOperation::new("test.fixed", vec![ArrayType::scalar(DataType::F32)]);
        assert!(matches!(
            CompiledKernel::from_output(&verified, b"config", &(), &FixedEmbedding(changed_output)),
            Err(KernelEmbeddingError::Invalid { message })
                if message == "compiler output types differ from the logical kernel results",
        ));
        let empty_target = CustomCallOperation::new("", vec![scalar.clone()]).with_input_output_alias(0, 0).unwrap();
        assert!(matches!(
            CompiledKernel::from_output(&verified, b"config", &(), &FixedEmbedding(empty_target)),
            Err(KernelEmbeddingError::Invalid { message })
                if message == "custom-call target must be nonempty and contain no NUL",
        ));
        let reserved_attribute = CustomCallOperation::new("test.fixed", vec![scalar])
            .with_input_output_alias(0, 0)
            .unwrap()
            .with_attribute("ryft.kernel.schema", 1i64);
        assert!(matches!(
            CompiledKernel::from_output(&verified, b"config", &(), &FixedEmbedding(reserved_attribute)),
            Err(KernelEmbeddingError::Invalid { message })
                if message == "duplicate or reserved custom-call attribute `ryft.kernel.schema`",
        ));
    }

    #[test]
    fn test_compiled_kernel_bind() {
        let definition = definition();
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let compiled =
            CompiledKernel::from_output(&verified, b"configuration", &DeferredCopy { schema: 1 }, &DeferredEmbedding)
                .unwrap();
        let (_, program) = TracingContext::<Constant, Operation>::trace(
            |inputs: Vec<CoreTracer<TracingContext<Constant, Operation>>>| compiled.bind(inputs[0].context(), &inputs),
            vec![ArrayIrType::Array(ArrayType::scalar(DataType::I32))],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {r#"
                lambda %0:i32[] .
                let %1:i32[] = custom_call [
                    target=ryft.test.deferred_copy,
                    deferred.schema=1,
                    ryft.kernel.schema=1,
                    ryft.kernel.semantic="c0aace572db2a32780947575541c582a8c0e012a86dccb7f08be2cffcc1a97b0",
                    ryft.kernel.configuration="9b504f8715e50a1265774108552c85c4822b04e39355818c680592419a1e0f7d",
                    input_output_alias=0->0,
                ] %0
                in (%1)
            "#}
            .trim_end(),
        );
        let error = TracingContext::<Constant, Operation>::trace(
            |inputs: Vec<CoreTracer<TracingContext<Constant, Operation>>>| compiled.bind(inputs[0].context(), &inputs),
            vec![ArrayIrType::Array(ArrayType::scalar(DataType::F32))],
        )
        .unwrap_err();
        assert_eq!(
            error,
            ProgramError::Type(TypeError::invalid("`kernel_call` input types do not match its logical signature")),
        );

        // A placed input binds too. The result aliases the input, so the custom call declares it with the type of the
        // input, whether that is replicated on a mesh or a local shard that varies along a manual axis, and the
        // traced program is well typed only if the call accepts its own alias.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("device", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let replicated = Sharding::replicated(mesh, 0);
        for sharding in [replicated.clone(), replicated.with_varying_manual_axes(["device"]).unwrap()] {
            let placed = ArrayIrType::Array(ArrayType::scalar(DataType::I32).with_sharding(sharding).unwrap());
            let (output_types, program) = TracingContext::<Constant, Operation>::trace(
                |inputs: Vec<CoreTracer<TracingContext<Constant, Operation>>>| {
                    compiled.bind(inputs[0].context(), &inputs)
                },
                vec![placed.clone()],
            )
            .unwrap();
            assert_eq!(output_types, vec![placed.clone()]);
            let Operation::CustomCall(custom_call) = program.instructions()[0].operation() else {
                panic!("expected a custom call");
            };
            let ArrayIrType::Array(placed) = placed else { unreachable!() };
            assert_eq!(custom_call.output_types(), std::slice::from_ref(&placed));
            assert_eq!(
                CoreOperation::infer_output_types(custom_call, std::slice::from_ref(&placed), &[]),
                Ok(vec![placed]),
            );
        }
    }

    #[test]
    fn test_compiled_kernel_custom_call_stable_hlo() {
        let definition = definition();
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let compiled = CompiledKernel::from_output(
            &verified,
            b"fixture configuration",
            &DeferredCopy { schema: 1 },
            &DeferredEmbedding,
        )
        .unwrap();
        let scalar = ArrayType::scalar(DataType::I32);
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(scalar.clone().into());
        let output = builder.add_instruction(compiled.custom_call().clone(), vec![], vec![input], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert!(program.reference_analysis(0).unwrap().roots().next().is_none());
        let lowered = lower_mlir_module_for_program(
            &program,
            &[],
            &vec![scalar.clone()],
            &vec![scalar],
            "main",
            None,
            None,
            None,
        )
        .unwrap();
        let (module, _signature, requires_assertion_handler, _requirements) = lowered.into_parts();
        assert!(!requires_assertion_handler);
        let semantic = hex::encode(Sha256::digest(definition.semantic_key().unwrap().as_bytes()));
        let configuration = "6dea220b6278c1cf02bdf72bf1a7969892b830605f8a1a1baea5af53bfcad12e";
        assert_eq!(
            module,
            formatdoc! {r#"
            module {{
              func.func @main(%arg0: tensor<i32>) -> tensor<i32> {{
                %0 = stablehlo.custom_call @ryft.test.deferred_copy(%arg0) {{api_version = 4 : i32, backend_config = {{deferred.schema = 1 : i64, ryft.kernel.configuration = "{configuration}", ryft.kernel.schema = 1 : i64, ryft.kernel.semantic = "{semantic}"}}, operand_layouts = [dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<> : tensor<0xindex>]}} : (tensor<i32>) -> tensor<i32>
                return %0 : tensor<i32>
              }}
            }}
        "#}
        );
    }

    #[test]
    fn test_dense_memory_layouts() {
        let matrix = ArrayType::new_static(DataType::F32, [2, 3]);
        let scalar = ArrayType::new_static(DataType::F32, []);
        let layouts =
            dense_memory_layouts(&[matrix.clone(), scalar.clone()], std::slice::from_ref(&matrix), "mosaic GPU")
                .unwrap();
        assert_eq!(layouts.operands, vec![vec![1, 0], vec![]]);
        assert_eq!(layouts.results, vec![vec![1, 0]]);
        let explicit = matrix.clone().with_layout(Layout::Tiled(TiledLayout::new(vec![1, 0], vec![])));
        assert_eq!(dense_memory_layouts(&[explicit], &[], "mosaic GPU").unwrap().operands, vec![vec![1, 0]]);
        let column_major = matrix.with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], vec![])));
        assert!(
            matches!(dense_memory_layouts(&[column_major], &[], "mosaic GPU"), Err(KernelEmbeddingError::Invalid { message })
            if message == "mosaic GPU requires untiled dense row-major array layouts")
        );
        assert!(matches!(
            dense_memory_layouts(&[scalar.with_memory(Memory::Host { pinned: false })], &[], "cuda kernel"),
            Err(KernelEmbeddingError::Invalid { message })
                if message == "cuda kernel requires static device array buffers",
        ));
    }

    #[test]
    fn test_validate_kernel_sharding() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("device", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::Sharded(vec!["device".to_owned()])])
            .unwrap()
            .with_varying_manual_axes(["device"])
            .unwrap();
        let local = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding).unwrap();
        let bound = BTreeMap::from([("device".to_owned(), mesh.axes()[0].clone())]);
        validate_kernel_sharding(std::slice::from_ref(&local), &bound).unwrap();
        assert!(matches!(
            validate_kernel_sharding(std::slice::from_ref(&local), &BTreeMap::new()),
            Err(KernelEmbeddingError::Invalid { message }) if message == unbound_kernel_axis_message(0, "device"),
        ));
        let mismatched =
            BTreeMap::from([("device".to_owned(), MeshAxis::new("device", 4, MeshAxisType::Manual).unwrap())]);
        assert!(matches!(
            validate_kernel_sharding(&[local], &mismatched),
            Err(KernelEmbeddingError::Invalid { message }) if message == unbound_kernel_axis_message(0, "device"),
        ));
        let replicated = ArrayType::new_static(DataType::F32, [8])
            .with_sharding(Sharding::replicated(mesh.clone(), 1))
            .unwrap();
        validate_kernel_sharding(&[replicated], &BTreeMap::new()).unwrap();
        let unconstrained = ArrayType::new_static(DataType::F32, [8])
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::Unconstrained]).unwrap())
            .unwrap();
        assert!(matches!(
            validate_kernel_sharding(&[unconstrained], &bound),
            Err(KernelEmbeddingError::Invalid { message })
                if message == "kernel input 0 has an unconstrained dimension placement, so it is not known to be a \
                               local shard",
        ));
        let pending = ArrayType::new_static(DataType::F32, [8])
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["device"]).unwrap())
            .unwrap();
        assert!(matches!(
            validate_kernel_sharding(&[pending], &bound),
            Err(KernelEmbeddingError::Invalid { message })
                if message == "kernel input 0 carries a pending reduction, which a kernel cannot complete",
        ));
        assert_eq!(
            unbound_kernel_axis_message(0, "device"),
            "kernel input 0 is placed on or varies along mesh axis `device`, which no enclosing `shard_map` binds, so \
             one device holds only a part of it. Replicate the input along `device` before the call, or apply the \
             kernel inside a `shard_map` over `device`",
        );
        // Types cannot record placement over auto mesh axes, so no kernel parameter is partitioned over one.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("device", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let automatic = Sharding::new(mesh, vec![ShardingDimension::Sharded(vec!["device".to_owned()])]).unwrap();
        assert!(matches!(
            ArrayType::new_static(DataType::F32, [8]).with_sharding(automatic),
            Err(ShardingError::AutoAxisInArrayType { .. }),
        ));
    }
}
