//! Verified semantic inputs and the minimal compiler-adapter capability.
//!
//! Adapters own their target, options, errors, and output representations. This module neither loads artifacts nor
//! manages executables or caches. Execution integrations use the existing compilation-domain lifecycle and combine
//! the kernel semantic key with the adapter's complete configuration key before reusing compiled work.

use std::num::NonZeroUsize;

use thiserror::Error;

use crate::arrays::{ArrayIrType, ArrayReferenceView};
use crate::kernels::calls::KernelDefinition;
use crate::kernels::initialization::{KernelInitializationError, validate_kernel_initialization};
use crate::kernels::operations::NoKernelExtension;
use crate::programs::{Operation, ReferenceViewOperation};

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
        /// Exact version or calling-convention mismatch.
        message: String,
    },

    /// An admitted compilation failed; the adapter retains its concrete error and source chain.
    #[error(transparent)]
    Compiler(E),
}

/// Optional result-preserving compilation hints. Absence leaves the choice to the adapter.
///
/// These hints never define execution-agent membership, synchronization, atomics, or numerical precision. Those
/// remain semantic operations and cannot be erased with a schedule. Adapters may ignore a performance hint or reject
/// an unavailable resource budget, but must not change kernel results. Target-specific layouts belong to adapter
/// options. Portable [`Layout`](crate::arrays::Layout) and [`Memory`](crate::arrays::Memory) constraints use the
/// existing [`ArrayType`](crate::arrays::ArrayType) metadata, including scratch types; no parallel placement or
/// layout descriptor is introduced. Those type constraints retain their ordinary meaning when hints are erased.
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
    /// Returns the optional pipeline-stage count.
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

    /// Requests a nonzero pipeline-stage count without changing the semantic definition.
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
}

/// Borrowed immutable definition that passed the executable core verifier.
///
/// The borrow prevents replacing its body or call metadata while an adapter holds this proof. Semantic rewrites
/// require a new definition and a new validation. This is a core proof only: compiler admission must still check the
/// exact target and options before artifact lookup or compilation. No hardware capability is implied.
#[derive(Debug)]
pub struct VerifiedKernel<'k, Extension: Operation<Type = ArrayIrType> = NoKernelExtension> {
    /// Definition validated by the constructor and immutably borrowed for this proof's lifetime.
    definition: &'k KernelDefinition<Extension>,
}

impl<'k, Extension> VerifiedKernel<'k, Extension>
where
    Extension: ReferenceViewOperation<Type = ArrayIrType, View = ArrayReferenceView>,
{
    /// Verifies initialization, accesses, bounds, output coverage, and mutable-window disjointness before admission.
    /// The enumeration budget bounds verification work and is excluded from semantic and compiler identity.
    pub fn new(
        definition: &'k KernelDefinition<Extension>,
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

    /// Admits the requested target/options before invoking the selected compiler. This function performs no fallback,
    /// caching, native loading, or execution; the execution integration owns those policies and lifetimes.
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
/// Associated outputs may be ready platform artifacts or versioned deferred compiler input; neither requires an
/// XLA-owned envelope. Runtime buffers, pointers, streams, and outer execution fences are outside this capability.
pub trait KernelCompiler<Extension: Operation<Type = ArrayIrType> = NoKernelExtension> {
    /// Adapter-owned normalized target facts, including required architecture and compiler compatibility.
    type Target;

    /// Typed adapter options, independent of the portable semantic definition.
    type Options;

    /// Adapter-owned ready artifact or deferred compiler payload with its physical argument mapping.
    type Output;

    /// Concrete tool, compiler, or artifact-validation error.
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

    /// Returns deterministic bytes covering every compiler-, target-, option-, and schedule-dependent choice.
    /// Include adapter/schema versions and native compiler versions, even when defaults choose them implicitly.
    /// The execution integration combines this with the core semantic key and its own ABI/plugin facts.
    fn configuration_key(
        &self,
        target: &Self::Target,
        options: &Self::Options,
        schedule: &KernelSchedule,
    ) -> Result<Vec<u8>, KernelCompilationError<Self::Error>>;

    /// Compiles an admitted definition into an adapter-owned output. Implementations must validate tool output and
    /// preserve the logical signature and alias/effect contract in their physical mapping. Callers must invoke
    /// [`Self::admit`] first; [`VerifiedKernel::compile`] provides that ordering for uncached compilation.
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

    use crate::arrays::{Array, ArrayIrOperation, ArrayIrValue, ArrayType, DataType};
    use crate::kernels::calls::{KernelCallOperation, KernelParameter};
    use crate::kernels::grids::Grid;
    use crate::kernels::mappings::{BlockMapping, BoundaryPolicy};
    use crate::kernels::validation::KernelParameterAccess;
    use crate::operations::{ReferenceRead, ReferenceWrite};
    use crate::programs::ProgramBuilder;

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
            Ok(format!("test compiler schema 1 target {target} options {options} schedule {schedule:?}").into_bytes())
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

    #[test]
    fn test_kernel_compilation_error() {
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
    fn test_verified_kernel_new() {
        let valid = definition(true);
        assert!(VerifiedKernel::new(&valid, 1).is_ok());
        let invalid = definition(false);
        assert!(matches!(
            VerifiedKernel::new(&invalid, 1),
            Err(KernelInitializationError::IncompleteBody { parameter: 1 }),
        ));
    }

    #[test]
    fn test_verified_kernel_definition() {
        let definition = definition(true);
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        assert!(std::ptr::eq(verified.definition(), &definition));
    }

    #[test]
    fn test_verified_kernel_compile() {
        let definition = definition(true);
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let compiler = Compiler { events: RefCell::new(vec![]) };
        assert_eq!(
            verified.compile(&compiler, &true, &1, &KernelSchedule::default()).unwrap(),
            vec![ArrayIrType::Array(ArrayType::scalar(DataType::I32))],
        );
        assert_eq!(*compiler.events.borrow(), vec!["admit", "compile"]);

        compiler.events.borrow_mut().clear();
        let error = verified.compile(&compiler, &false, &1, &KernelSchedule::default()).unwrap_err();
        assert_eq!(
            error.to_string(),
            "test compiler cannot implement `kernel_call` contract `scalar arrays`: missing array support",
        );
        assert_eq!(*compiler.events.borrow(), vec!["admit"]);
    }

    #[test]
    fn test_kernel_compiler_configuration_key() {
        let compiler = Compiler { events: RefCell::new(vec![]) };
        let schedule = KernelSchedule::default();
        let key = compiler.configuration_key(&true, &1, &schedule).unwrap();
        assert_eq!(key, compiler.configuration_key(&true, &1, &schedule).unwrap());
        assert_ne!(key, compiler.configuration_key(&false, &1, &schedule).unwrap());
        assert_ne!(key, compiler.configuration_key(&true, &2, &schedule).unwrap());
        assert_ne!(key, compiler.configuration_key(&true, &1, &schedule.with_maximum_scratch_bytes(0)).unwrap(),);
        assert_eq!(*compiler.events.borrow(), Vec::<&'static str>::new());
    }
}
