//! Typed compiler-output embedding into the existing XLA custom-call and compilation lifecycle.
//!
//! Compiler outputs retain their adapter ownership. An XLA-owned [`KernelOutputEmbedding`] validates and translates
//! one concrete output type; [`CompiledKernel`] retains the verified source until that selection is complete and
//! emits the canonical custom call into ordinary staging. Existing XLA lowering, executable caching, persistence,
//! and PJRT fences then own the program. This module does not introduce a second executable or module cache.
//!
//! [`stage_kernel`] preserves the complete kernel call through batching, control flow, and rematerialization.
//! Differentiation requires an explicit [`stage_kernel_with_jvp`], [`stage_kernel_with_vjp`], or pure
//! [`stage_kernel_with_fallback`] contract. Specialize scalar-prefetched values with
//! [`KernelDefinition::specialize_prefetch`] before XLA staging; transforms never read captured device buffers back
//! to the host. Local per-shard compilation retains full sharding metadata and requires every manual axis to be bound
//! by an enclosing shard map with the same axis descriptor. Automatic partitioning and collectives are rejected.
//!
//! [`KernelTuningRequest`] defines a finite, fingerprinted schedule search. Call [`KernelTuner::load`] or
//! [`KernelTuner::run`] explicitly, sharing a tuner per device and controlling external contention. Samples measure
//! host submission through actual completion; any runner-owned readback must be part of its declared methodology.
//! Cancellation stops further submissions and awaits work already submitted. Measurements do not imply portable
//! performance rankings or device-only timing.
//!
//! [`KernelAotBundle`] exports checked portable source, [`KernelCompilationReport`], StableHLO, compatibility facts,
//! and the existing complete executable envelope. Import checks compatibility before native loading and returns a
//! [`LoadedKernel`] using the ordinary asynchronous execution path. Unsupported source codecs fail explicitly;
//! recompilation after incompatibility is an explicit caller decision. [`DistributedKernel`] coordinates functional
//! calls over an existing distributed runtime using explicit host staging and the canonical pending completion.
//! See [`distributed`] for ordering, cancellation, topology admission and publication semantics.
//!
//! The optional `triton` feature adds direct typed-TTIR compilation with `TritonEmbedding`. CUDA artifacts reuse
//! [`CudaKernelEmbedding`]; ROCm artifacts use a separate HIP session owner enabled by `rocm`. Both retain the same
//! staging, persistence and whole-execution completion path. Compiler-free Triton AOT loading reconstructs the
//! binding with recorded compiler configuration; it still requires a compatible runtime and device. AMD hardware
//! execution remains unqualified, and unsupported portable operations fail before native compilation.

use std::collections::{BTreeMap, BTreeSet};

use ryft_core::kernels::{
    KERNEL_SCHEMA_VERSION, KernelCompiler, KernelDefinition, KernelExtension, KernelSchedule, NoKernelExtension,
    VerifiedKernel,
};
use ryft_core::operations::custom_call::CustomCallOperation;
use ryft_core::{
    ArrayIrType, ArrayType, Context, DataType, EffectClass, Layout, Memory, MeshAxis, MeshAxisType, Operation,
    ProgramError, ShardingDimension, TypeError, Typed,
};
use ryft_mlir::dialects::stable_hlo::CustomCallMemoryLayouts;
use sha2::{Digest, Sha256};
use thiserror::Error;

mod aot;
mod cuda;
#[cfg(feature = "cutile")]
mod cutile;
pub mod distributed;
#[cfg(feature = "mosaic-gpu")]
pub(crate) mod mosaic;
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
    XlaKernelCompilerBinding, XlaKernelDeviceFacts, XlaKernelExecutionFacts, XlaKernelExtension, XlaKernelOperation,
    XlaKernelTarget, stage_kernel, stage_kernel_with_fallback, stage_kernel_with_jvp, stage_kernel_with_vjp,
};

#[cfg(feature = "rocm")]
pub use rocm::RocmKernelEmbedding;
#[cfg(feature = "rocm")]
pub(crate) use rocm::RocmKernelRuntime;

/// Typed FFI target used by persisted ROCm kernel calls.
pub const ROCM_KERNEL_CUSTOM_CALL_TARGET: &str = "ryft.kernel.rocm";

pub(crate) use cuda::CudaKernelRuntime;
pub use cuda::{
    CUDA_KERNEL_CUSTOM_CALL_TARGET, CudaKernelBufferBinding, CudaKernelEmbedding, CudaKernelParameterBinding,
};

/// Invalid compiler output or unsupported XLA embedding contract.
#[derive(Debug, Error)]
pub enum KernelEmbeddingError {
    /// PJRT registration or execution-context setup failed.
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

    /// A canonical type or custom-call alias check failed.
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
        /// Effect that requires a separate supported token/handler contract.
        effect: EffectClass,
    },
}

/// XLA-owned conversion of one adapter-owned compiler output into a canonical custom call.
///
/// Implementations are trusted integration code: they must validate the output's version, physical argument map,
/// required buffers, target requirements, and serialized payload before returning. An adapter output never needs to
/// depend on this trait: XLA integration implements the trait for its own bridge over the foreign output type.
/// Ready and deferred outputs use the same boundary; a deferred bridge additionally validates its native decoder
/// and compiler-input schema. Registration and target capability checks must precede native loading or execution.
///
/// [`CompiledKernel`] independently validates the logical signature, operation-local aliases, and effects, adds
/// semantic/compiler identity, and discharges only the definition's proven-local reference state. A bridge cannot
/// request external reference-state slots or alter the kernel's logical inputs and results. An embedding implementing
/// native checked assertions must return precisely [`EffectClass::OrderedAssertion`], preserving their token chain;
/// a pure call cannot discharge or silently discard an observable assertion.
pub trait KernelOutputEmbedding<Output, Extension: Operation<Type = ArrayIrType> = NoKernelExtension> {
    /// Returns deterministic bytes covering target names, physical mappings, payload schemas, and plugin ABI facts.
    fn configuration_key(&self) -> Result<Vec<u8>, KernelEmbeddingError>;

    /// Validates the typed output and builds its complete target call, including payload attributes.
    fn custom_call(
        &self,
        kernel: &VerifiedKernel<'_, Extension>,
        output: &Output,
    ) -> Result<CustomCallOperation, KernelEmbeddingError>;
}

/// Selected compiler output together with the immutable source definition and its canonical custom-call embedding.
///
/// Selection is explicit; no constructor falls back to another compiler. The retained definition prevents an opaque
/// payload from masquerading as an unverified kernel. Binding emits the selected custom call only after validating
/// the exact logical inputs. Its attributes participate in existing StableHLO-based compilation and persistence keys.
#[derive(Clone, Debug)]
pub struct CompiledKernel<Extension: Operation<Type = ArrayIrType> = NoKernelExtension> {
    /// Immutable body and boundary that were verified before compilation and embedding.
    definition: KernelDefinition<Extension>,

    /// Canonical array custom call selected for this exact definition and compiler configuration.
    custom_call: CustomCallOperation,
}

impl<Extension> CompiledKernel<Extension>
where
    Extension: KernelExtension,
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

    /// Embeds an already available typed output, permitting runtime-only use when its native format supports it.
    /// `configuration` must be the complete key emitted by the compiler that produced `output`; an integration must
    /// validate persisted adapter/schema/plugin compatibility before invoking this constructor on reloaded output.
    /// The embedding configuration is independently included in the resulting custom-call identity.
    pub fn from_output<Output, B: KernelOutputEmbedding<Output, Extension>>(
        kernel: &VerifiedKernel<'_, Extension>,
        configuration: &[u8],
        output: &Output,
        embedding: &B,
    ) -> Result<Self, KernelEmbeddingError> {
        // The immutable verified definition owns every body reference input and forbids reference constants,
        // captures, consumption, and escaping references. Removing only this local OrderedState is semantic
        // reference discharge at the operation boundary. External effects need a separately supported token ABI.
        let effects = kernel.definition().body().effects();
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
            .with_attribute("ryft.kernel.schema", i64::from(KERNEL_SCHEMA_VERSION))
            .with_attribute(
                "ryft.kernel.semantic",
                format!("{:x}", Sha256::digest(kernel.definition().semantic_key()?.as_bytes())),
            )
            .with_attribute("ryft.kernel.configuration", format!("{:x}", configuration_hash.finalize()));
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

    /// Binds the selected call through ordinary staging or an existing compilation domain after exact input checking.
    /// External reference-state arguments are absent; all operands and outputs are ordinary array values.
    pub fn bind<C>(&self, context: &C, inputs: &[C::Value]) -> Result<Vec<C::Value>, ProgramError>
    where
        C: Context<Type = ArrayIrType, Operation: From<CustomCallOperation>>,
    {
        let actual = inputs.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
        if actual != self.definition.operation().input_types() {
            return Err(TypeError::invalid("compiled kernel input types do not match its logical signature").into());
        }
        context.bind(self.custom_call.clone(), Vec::new(), inputs)
    }
}

/// Validates the native memref ABI and fixes complete row-major operand/result layouts, including default types.
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

/// Requires already-local manual shards; adapter selection never synthesizes partitioning or a collective.
pub(crate) fn validate_kernel_sharding(
    parameters: &[ArrayType],
    bound_axes: &BTreeMap<String, MeshAxis>,
) -> Result<(), KernelEmbeddingError> {
    for (index, parameter) in parameters.iter().enumerate() {
        let Some(sharding) = parameter.sharding() else { continue };
        let invalid = || KernelEmbeddingError::Invalid {
            message: format!(
                "kernel parameter {index} requires a local shard with all partitioned axes bound by `shard_map`"
            ),
        };
        if !sharding.unreduced_axes().is_empty() || !sharding.reduced_axes().is_empty() {
            return Err(invalid());
        }
        let mut axes = sharding.varying_manual_axes().clone();
        for dimension in sharding.dimensions() {
            match dimension {
                ShardingDimension::Replicated => {}
                ShardingDimension::Sharded(names) => axes.extend(names.iter().cloned()),
                ShardingDimension::Unconstrained => return Err(invalid()),
            }
        }
        if axes.iter().any(|axis| {
            sharding.mesh().axis_type(axis) != Some(MeshAxisType::Manual)
                || sharding.mesh().axis_index(axis).map(|index| &sharding.mesh().axes()[index]) != bound_axes.get(axis)
        }) {
            return Err(invalid());
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use std::convert::Infallible;

    use indoc::formatdoc;
    use pretty_assertions::assert_eq;
    use ryft_core::kernels::{
        BlockMapping, BoundaryPolicy, Grid, KernelCallOperation, KernelCompilationError, KernelParameter,
        KernelParameterAccess,
    };
    use ryft_core::{
        Array, ArrayIrOperation, ArrayIrValue, Context, DataType, Placeholder, ProgramBuilder, ReferenceRead,
        ReferenceWrite,
    };

    use crate::experimental::lowering::lower_mlir_module_for_program;
    use crate::experimental::ops::{XlaConstant, XlaProgramBuilder};

    use super::*;

    /// A scalar identity with a functional read-write alias and no external reference slot.
    pub(crate) fn definition() -> KernelDefinition {
        let mapping = BlockMapping::new(
            ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new()
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
            _kernel: &VerifiedKernel<'_>,
            target: &bool,
            _options: &u32,
            _schedule: &KernelSchedule,
        ) -> Result<(), KernelCompilationError<Infallible>> {
            if *target {
                Ok(())
            } else {
                Err(KernelCompilationError::Unavailable { message: "fixture target unavailable".to_owned() })
            }
        }
        fn configuration_key(
            &self,
            _target: &bool,
            options: &u32,
            _schedule: &KernelSchedule,
        ) -> Result<Vec<u8>, KernelCompilationError<Infallible>> {
            Ok(options.to_le_bytes().to_vec())
        }
        fn compile(
            &self,
            _kernel: &VerifiedKernel<'_>,
            _target: &bool,
            _options: &u32,
            _schedule: &KernelSchedule,
        ) -> Result<DeferredCopy, KernelCompilationError<Infallible>> {
            Ok(DeferredCopy { schema: 1 })
        }
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
            Err(KernelEmbeddingError::Compiler(_))
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
        let metadata = compiled.custom_call().to_string();
        assert!(metadata.contains("ryft.kernel.provenance"));
        assert!(metadata.contains("copy_output"));
        assert!(metadata.contains("ryft.kernel.semantic"));
        let unknown = super::tests::definition();
        let unknown = VerifiedKernel::new(&unknown, 1).unwrap();
        let compiled =
            CompiledKernel::from_output(&unknown, b"configuration", &DeferredCopy { schema: 1 }, &DeferredEmbedding)
                .unwrap();
        assert!(!compiled.custom_call().to_string().contains("ryft.kernel.provenance"));
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
                _kernel: &VerifiedKernel<'_>,
                _output: &(),
            ) -> Result<CustomCallOperation, KernelEmbeddingError> {
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
        use ryft_core::ArrayOperation;
        use ryft_core::kernels::KernelOperation;

        for effect in [EffectClass::OrderedIo, EffectClass::OrderedState, EffectClass::OrderedAssertion] {
            let logical = definition().operation().clone();
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
            assert!(matches!(
                CompiledKernel::from_output(&verified, b"config", &DeferredCopy { schema: 1 }, &DeferredEmbedding),
                Err(KernelEmbeddingError::UnsupportedEffect { effect: actual }) if actual == effect,
            ));
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
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(scalar.clone().into());
        let output = builder.add_instruction(compiled.custom_call().clone(), vec![], vec![input], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
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
        let (module, _signature, requires_assertion_handler) = lowered.into_parts();
        assert!(!requires_assertion_handler);
        let semantic = format!("{:x}", Sha256::digest(definition.semantic_key().unwrap().as_bytes()));
        let configuration = "6dea220b6278c1cf02bdf72bf1a7969892b830605f8a1a1baea5af53bfcad12e";
        assert_eq!(
            module,
            formatdoc! {r#"
            module {{
              func.func @main(%arg0: tensor<i32>) -> tensor<i32> {{
                %0 = stablehlo.custom_call @ryft.test.deferred_copy(%arg0) {{api_version = 4 : i32, backend_config = {{deferred.schema = 1 : i64, ryft.kernel.configuration = "{configuration}", ryft.kernel.schema = 2 : i64, ryft.kernel.semantic = "{semantic}"}}, output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}} : (tensor<i32>) -> tensor<i32>
                return %0 : tensor<i32>
              }}
            }}
        "#}
        );
    }
    #[test]
    fn test_dense_memory_layouts() {
        use ryft_core::TiledLayout;

        let matrix = ArrayType::new_static(DataType::F32, [2, 3]);
        let scalar = ArrayType::new_static(DataType::F32, []);
        let layouts = dense_memory_layouts(&[matrix.clone(), scalar.clone()], &[matrix.clone()], "mosaic GPU").unwrap();
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
        use ryft_core::{LogicalMesh, MeshAxis, Sharding};

        let mesh = LogicalMesh::new(vec![MeshAxis::new("device", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::Sharded(vec!["device".to_owned()])])
            .unwrap()
            .with_varying_manual_axes(["device"])
            .unwrap();
        let local = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding).unwrap();
        let bound = BTreeMap::from([("device".to_owned(), mesh.axes()[0].clone())]);
        assert!(validate_kernel_sharding(&[local.clone()], &bound).is_ok());
        assert!(matches!(
            validate_kernel_sharding(&[local.clone()], &BTreeMap::new()),
            Err(KernelEmbeddingError::Invalid { message })
                if message == "kernel parameter 0 requires a local shard with all partitioned axes bound by `shard_map`",
        ));
        let mismatched =
            BTreeMap::from([("device".to_owned(), MeshAxis::new("device", 4, MeshAxisType::Manual).unwrap())]);
        assert!(matches!(
            validate_kernel_sharding(&[local], &mismatched),
            Err(KernelEmbeddingError::Invalid { message })
                if message == "kernel parameter 0 requires a local shard with all partitioned axes bound by `shard_map`",
        ));
        let replicated = ArrayType::new_static(DataType::F32, [8])
            .with_sharding(Sharding::replicated(mesh.clone(), 1))
            .unwrap();
        assert!(validate_kernel_sharding(&[replicated], &BTreeMap::new()).is_ok());
        let unconstrained = ArrayType::new_static(DataType::F32, [8])
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::Unconstrained]).unwrap())
            .unwrap();
        assert!(matches!(
            validate_kernel_sharding(&[unconstrained], &bound),
            Err(KernelEmbeddingError::Invalid { message })
                if message == "kernel parameter 0 requires a local shard with all partitioned axes bound by `shard_map`",
        ));
        let mesh = LogicalMesh::new(vec![MeshAxis::new("device", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let automatic = ArrayType::new_static(DataType::F32, [8])
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::Sharded(vec!["device".to_owned()])]).unwrap())
            .unwrap();
        assert!(matches!(
            validate_kernel_sharding(&[automatic], &bound),
            Err(KernelEmbeddingError::Invalid { message })
                if message == "kernel parameter 0 requires a local shard with all partitioned axes bound by `shard_map`",
        ));
    }
}
