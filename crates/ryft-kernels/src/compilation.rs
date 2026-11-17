//! The contract between verified kernel definitions and the compiler adapters that compile them.
//!
//! A compiler adapter (e.g., for Mosaic, Triton, or cuTile) implements [`KernelCompiler`] and owns its target,
//! options, errors, and output representations. This module defines only that contract. It neither loads artifacts
//! nor manages executables or caches, which belong to execution integrations and their existing compilation domain
//! lifecycle.
//!
//! # Verification
//!
//! An adapter compiles only a [`VerifiedKernel`], which [`VerifiedKernel::new`] constructs after
//! [`validate_kernel_initialization`] accepts its [`KernelDefinition`]. That check covers definite initialization,
//! reference accesses, window bounds, complete output coverage, and the disjointness of the mutable windows of
//! unordered programs. A [`VerifiedKernel`] borrows its definition immutably, so the verified body and call metadata
//! cannot change while an adapter holds it, and a semantic rewrite requires a new definition and a new verification.
//! It proves only these core facts. It does not prove that any compiler, target, or hardware supports the kernel.
//!
//! # Adapter Contract
//!
//! [`KernelCompiler::admit`] checks semantic support, target compatibility, the compiler installation, and the
//! legality of the options, and it must run before any artifact lookup or compilation. Its rejections distinguish
//! unsupported contracts, missing installations, and incompatible versions through the variants of
//! [`KernelCompilationError`]. [`KernelCompiler::configuration_key`] returns deterministic bytes that cover every
//! choice that depends on the compiler, target, options, or schedule. Execution integrations combine this key with
//! the [`semantic_key`](KernelDefinition::semantic_key) of the definition before reusing compiled work.
//! [`KernelCompiler::compile`] compiles an admitted kernel into an output that the adapter owns, and
//! [`VerifiedKernel::compile`] runs admission and compilation in that order without fallback to another compiler.
//!
//! # Schedules
//!
//! A [`KernelSchedule`] carries optional hints (i.e., pipeline stages, buffering depth, and a scratch budget) that may
//! influence performance and resource use but never kernel results. An adapter may ignore a performance hint or reject
//! an unavailable resource budget. A schedule cannot change execution agent membership, synchronization, atomics, or
//! numerical precision, which remain semantic operations of the body, and schedules take no part in the semantic key.

use std::num::NonZeroUsize;

use ryft_core::{ArrayIrType, Operation};
use thiserror::Error;

use crate::calls::KernelDefinition;
use crate::initialization::{KernelInitializationError, validate_kernel_initialization};
use crate::operations::{KernelExtension, NoKernelExtension};

/// Compiler admission and invocation failures, distinct from invalid core semantics.
#[derive(Debug, Error)]
pub enum KernelCompilationError<E: std::error::Error> {
    /// The selected adapter cannot implement a valid requested semantic contract.
    #[error("{owner} cannot implement `{operation}` contract `{requested}`: missing {capability}")]
    Unsupported {
        /// Adapter reporting the unsupported contract.
        owner: String,

        /// Canonical operation requiring the capability.
        operation: &'static str,

        /// Exact requested semantic behavior.
        requested: String,

        /// Capability absent from this compiler/target combination.
        capability: String,
    },

    /// The compiler installation or required tool is unavailable.
    #[error("kernel compiler is unavailable: {message}")]
    Unavailable {
        /// Installation detail suitable for a user diagnostic.
        message: String,
    },

    /// Installed compiler or target versions do not satisfy the adapter's contract.
    #[error("kernel compiler is incompatible: {message}")]
    Incompatible {
        /// Exact version or calling convention mismatch.
        message: String,
    },

    /// An admitted compilation failed, and the adapter retains its concrete error and source chain.
    #[error(transparent)]
    Compiler(E),
}

/// Optional compilation hints that preserve kernel results. Absence leaves the choice to the adapter.
///
/// These hints never define execution agent membership, synchronization, atomics, or numerical precision. Those
/// remain semantic operations and cannot be erased with a schedule. Adapters may ignore a performance hint or reject
/// an unavailable resource budget, but must not change kernel results. Layouts specific to a target belong to adapter
/// options. Portable [`Layout`](ryft_core::Layout) and [`Memory`](ryft_core::Memory) constraints use the existing
/// [`ArrayType`](ryft_core::ArrayType) metadata, including scratch types, so no parallel placement or layout
/// descriptor is introduced. Those type constraints retain their ordinary meaning when hints are erased.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct KernelSchedule {
    /// Requested number of overlapping pipeline stages.
    pipeline_stages: Option<NonZeroUsize>,

    /// Requested number of reusable buffers within a pipeline.
    buffering_depth: Option<NonZeroUsize>,

    /// Maximum scratch bytes available to the adapter, with zero explicitly forbidding scratch.
    maximum_scratch_bytes: Option<usize>,
}

impl KernelSchedule {
    /// Requests a nonzero pipeline stage count without changing the semantic definition.
    pub fn with_pipeline_stages(mut self, stages: NonZeroUsize) -> Self {
        self.pipeline_stages = Some(stages);
        self
    }

    /// Requests a nonzero buffering depth without changing the semantic definition.
    pub fn with_buffering_depth(mut self, depth: NonZeroUsize) -> Self {
        self.buffering_depth = Some(depth);
        self
    }

    /// Limits adapter scratch allocation. A budget of zero is valid and forbids scratch allocation.
    pub fn with_maximum_scratch_bytes(mut self, bytes: usize) -> Self {
        self.maximum_scratch_bytes = Some(bytes);
        self
    }

    /// Returns the optional pipeline stage count.
    pub fn pipeline_stages(&self) -> Option<NonZeroUsize> {
        self.pipeline_stages
    }

    /// Returns the optional buffering depth.
    pub fn buffering_depth(&self) -> Option<NonZeroUsize> {
        self.buffering_depth
    }

    /// Returns the optional scratch budget in bytes.
    pub fn maximum_scratch_bytes(&self) -> Option<usize> {
        self.maximum_scratch_bytes
    }
}

/// Borrowed immutable definition that passed the executable core verifier.
///
/// The borrow prevents replacing its body or call metadata while an adapter holds this proof. Semantic rewrites
/// require a new definition and a new validation. This is a core proof only, so compiler admission must still check
/// the exact target and options before artifact lookup or compilation. No hardware capability is implied.
#[derive(Debug)]
pub struct VerifiedKernel<'o, Extension: Operation<Type = ArrayIrType> = NoKernelExtension> {
    /// Definition validated by the constructor and immutably borrowed for the lifetime of this proof.
    definition: &'o KernelDefinition<Extension>,
}

impl<'o, Extension> VerifiedKernel<'o, Extension>
where
    Extension: KernelExtension,
{
    /// Verifies initialization, accesses, bounds, output coverage, and the disjointness of mutable windows before
    /// admission. The enumeration budget bounds verification work and is excluded from semantic and compiler
    /// identity.
    ///
    /// # Errors
    ///
    /// Returns the [`KernelInitializationError`] of [`validate_kernel_initialization`] when the definition fails
    /// verification.
    pub fn new(
        definition: &'o KernelDefinition<Extension>,
        maximum_programs: usize,
    ) -> Result<Self, KernelInitializationError> {
        validate_kernel_initialization(definition.body().entry_region_ref(), definition.operation(), maximum_programs)?;
        Ok(Self { definition })
    }
}

impl<Extension: Operation<Type = ArrayIrType>> VerifiedKernel<'_, Extension> {
    /// Returns the exact immutable definition covered by this verification.
    pub fn definition(&self) -> &KernelDefinition<Extension> {
        self.definition
    }

    /// Admits the requested target and options before invoking the selected compiler. This function performs no
    /// fallback, caching, native loading, or execution, because the execution integration owns those policies and
    /// lifetimes.
    ///
    /// # Errors
    ///
    /// Returns the [`KernelCompilationError`] of [`KernelCompiler::admit`] when the compiler rejects the kernel, and
    /// otherwise the error of [`KernelCompiler::compile`].
    pub fn compile<C: KernelCompiler<Extension>>(
        &self,
        compiler: &C,
        target: &C::Target,
        options: &C::Options,
        schedule: &KernelSchedule,
    ) -> Result<C::Output, KernelCompilationError<C::Error>> {
        compiler.admit(self, target, options, schedule)?;
        compiler.compile(self, target, options, schedule)
    }
}

/// Capability implemented by a concrete kernel compiler adapter.
///
/// Implementations are trusted compiler code. Admission checks the entire verified operation family, including exact
/// extension semantics, and must run before artifact lookup. Compilation never chooses another compiler implicitly.
/// Associated outputs may be ready platform artifacts or versioned deferred compiler input, and neither requires an
/// envelope that XLA owns. Runtime buffers, pointers, streams, and outer execution fences are outside this capability.
pub trait KernelCompiler<Extension: Operation<Type = ArrayIrType> = NoKernelExtension> {
    /// Normalized target facts that the adapter owns, including required architecture and compiler compatibility.
    type Target;

    /// Typed adapter options, independent of the portable semantic definition.
    type Options;

    /// Ready artifact or deferred compiler payload that the adapter owns, with its physical argument mapping.
    type Output;

    /// Concrete error of a tool, a compiler, or artifact validation.
    type Error: std::error::Error;

    /// Checks semantic support, target compatibility, installation, and option legality. Rejections must distinguish
    /// unsupported contracts, missing installations, incompatible versions, and failures through the error variants.
    fn admit(
        &self,
        kernel: &VerifiedKernel<'_, Extension>,
        target: &Self::Target,
        options: &Self::Options,
        schedule: &KernelSchedule,
    ) -> Result<(), KernelCompilationError<Self::Error>>;

    /// Returns deterministic bytes covering every choice that depends on the compiler, target, options, or schedule.
    /// Include the adapter and schema versions and the native compiler versions, even when defaults choose them
    /// implicitly. The execution integration combines this with the core semantic key and its own ABI and plugin
    /// facts.
    fn configuration_key(
        &self,
        target: &Self::Target,
        options: &Self::Options,
        schedule: &KernelSchedule,
    ) -> Result<Vec<u8>, KernelCompilationError<Self::Error>>;

    /// Compiles an admitted definition into an output that the adapter owns. Implementations must validate tool output
    /// and preserve the logical signature and the alias and effect contract in their physical mapping. Callers must
    /// invoke [`Self::admit`] first, and [`VerifiedKernel::compile`] provides that ordering for uncached compilation.
    fn compile(
        &self,
        kernel: &VerifiedKernel<'_, Extension>,
        target: &Self::Target,
        options: &Self::Options,
        schedule: &KernelSchedule,
    ) -> Result<Self::Output, KernelCompilationError<Self::Error>>;
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::convert::Infallible;

    use pretty_assertions::assert_eq;
    use ryft_core::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayType, DataType, ProgramBuilder, ReferenceRead, ReferenceWrite,
    };

    use crate::calls::{KernelCallOperation, KernelParameter};
    use crate::grids::Grid;
    use crate::mappings::{BlockMapping, BoundaryPolicy};
    use crate::validation::KernelParameterAccess;

    use super::*;

    /// Constructs a scalar copy through the same tracing API used by a portable authoring frontend.
    fn definition(initialize: bool) -> KernelDefinition {
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
                KernelParameter::new(
                    ArrayType::scalar(DataType::I32),
                    KernelParameterAccess::ReadOnly,
                    mapping.clone(),
                )
                .unwrap(),
                KernelParameter::new(ArrayType::scalar(DataType::I32), KernelParameterAccess::WriteOnly, mapping)
                    .unwrap(),
            ],
        )
        .unwrap();
        KernelDefinition::trace(operation, |(references, _coordinates)| {
            if initialize {
                references[1].write(&references[0].read()?)?;
            }
            Ok(())
        })
        .unwrap()
    }

    #[test]
    fn test_kernel_compilation_error() {
        let unsupported = KernelCompilationError::<Infallible>::Unsupported {
            owner: "test compiler".to_owned(),
            operation: "kernel_call",
            requested: "scalar arrays".to_owned(),
            capability: "array support".to_owned(),
        };
        assert_eq!(
            unsupported.to_string(),
            "test compiler cannot implement `kernel_call` contract `scalar arrays`: missing array support",
        );
        let unavailable = KernelCompilationError::<Infallible>::Unavailable { message: "tool missing".to_owned() };
        assert_eq!(unavailable.to_string(), "kernel compiler is unavailable: tool missing");
        let incompatible = KernelCompilationError::<Infallible>::Incompatible { message: "schema 2".to_owned() };
        assert_eq!(incompatible.to_string(), "kernel compiler is incompatible: schema 2");
        let failed = KernelCompilationError::Compiler(std::io::Error::other("tool exited with status 1"));
        assert_eq!(failed.to_string(), "tool exited with status 1");
    }

    #[test]
    fn test_kernel_schedule() {
        let default = KernelSchedule::default();
        let scheduled = default.clone().with_maximum_scratch_bytes(0);
        assert_ne!(default, scheduled);
        let keys = HashMap::from([(default, 0), (scheduled.clone(), 1)]);
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[&scheduled], 1);
    }

    #[test]
    fn test_kernel_schedule_with_pipeline_stages() {
        let schedule = KernelSchedule::default().with_pipeline_stages(NonZeroUsize::new(2).unwrap());
        assert_eq!(schedule.pipeline_stages(), NonZeroUsize::new(2));
        assert_eq!(schedule.buffering_depth(), None);
    }

    #[test]
    fn test_kernel_schedule_with_buffering_depth() {
        let schedule = KernelSchedule::default().with_buffering_depth(NonZeroUsize::new(3).unwrap());
        assert_eq!(schedule.buffering_depth(), NonZeroUsize::new(3));
        assert_eq!(schedule.pipeline_stages(), None);
    }

    #[test]
    fn test_kernel_schedule_with_maximum_scratch_bytes() {
        let schedule = KernelSchedule::default().with_maximum_scratch_bytes(0);
        assert_eq!(schedule.maximum_scratch_bytes(), Some(0));
        assert_eq!(schedule.with_maximum_scratch_bytes(64).maximum_scratch_bytes(), Some(64));
    }

    #[test]
    fn test_kernel_schedule_pipeline_stages() {
        assert_eq!(KernelSchedule::default().pipeline_stages(), None);
    }

    #[test]
    fn test_kernel_schedule_buffering_depth() {
        assert_eq!(KernelSchedule::default().buffering_depth(), None);
    }

    #[test]
    fn test_kernel_schedule_maximum_scratch_bytes() {
        assert_eq!(KernelSchedule::default().maximum_scratch_bytes(), None);
    }

    #[test]
    fn test_verified_kernel_new() {
        let valid = definition(true);
        assert_eq!(
            VerifiedKernel::new(&valid, 1).map(|verified| std::ptr::eq(verified.definition(), &valid)),
            Ok(true),
        );
        let invalid = definition(false);
        assert_eq!(
            VerifiedKernel::new(&invalid, 1).map(|verified| std::ptr::eq(verified.definition(), &invalid)),
            Err(KernelInitializationError::IncompleteBody { parameter: 1 }),
        );
    }

    #[test]
    fn test_verified_kernel_definition() {
        let definition = definition(true);
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        assert!(std::ptr::eq(verified.definition(), &definition));
    }

    #[test]
    fn test_verified_kernel_compile() {
        /// Records adapter transitions without importing any execution or native artifact crate.
        struct Compiler {
            /// Admission and compilation calls in their actual order.
            events: RefCell<Vec<&'static str>>,
        }

        impl KernelCompiler for Compiler {
            type Target = bool;
            type Options = u32;
            type Output = Vec<ArrayIrType>;
            type Error = Infallible;

            fn admit(
                &self,
                _kernel: &VerifiedKernel<'_>,
                target: &bool,
                _options: &u32,
                _schedule: &KernelSchedule,
            ) -> Result<(), KernelCompilationError<Infallible>> {
                self.events.borrow_mut().push("admit");
                if !target {
                    return Err(KernelCompilationError::Unsupported {
                        owner: "test compiler".to_owned(),
                        operation: "kernel_call",
                        requested: "scalar arrays".to_owned(),
                        capability: "array support".to_owned(),
                    });
                }
                Ok(())
            }

            fn configuration_key(
                &self,
                target: &bool,
                options: &u32,
                schedule: &KernelSchedule,
            ) -> Result<Vec<u8>, KernelCompilationError<Infallible>> {
                let key = format!("test compiler schema 1 target {target} options {options} schedule {schedule:?}");
                Ok(key.into_bytes())
            }

            fn compile(
                &self,
                kernel: &VerifiedKernel<'_>,
                _target: &bool,
                _options: &u32,
                _schedule: &KernelSchedule,
            ) -> Result<Vec<ArrayIrType>, KernelCompilationError<Infallible>> {
                self.events.borrow_mut().push("compile");
                Ok(kernel.definition().operation().output_types())
            }
        }

        let definition = definition(true);
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let compiler = Compiler { events: RefCell::new(vec![]) };
        assert_eq!(
            verified.compile(&compiler, &true, &1, &KernelSchedule::default()).unwrap(),
            vec![ArrayIrType::Array(ArrayType::scalar(DataType::I32))],
        );
        assert_eq!(*compiler.events.borrow(), vec!["admit", "compile"]);

        compiler.events.borrow_mut().clear();
        assert!(matches!(
            verified.compile(&compiler, &false, &1, &KernelSchedule::default()),
            Err(KernelCompilationError::Unsupported { owner, operation: "kernel_call", requested, capability })
                if owner == "test compiler" && requested == "scalar arrays" && capability == "array support",
        ));
        assert_eq!(*compiler.events.borrow(), vec!["admit"]);
    }
}
