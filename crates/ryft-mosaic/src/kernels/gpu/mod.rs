//! Mosaic GPU compilation for the Hopper-or-newer baseline.
//!
//! A logical program executes within one CUDA thread block. Tile values use cooperative, thread-strided work and
//! CTA-local storage; ordinary dot uses scalar multiply/add accumulation rather than target tensor-core instructions.
//! Uniform control flow preserves barrier participation. Admission rejects unsupported operations and memory layouts
//! before source compilation. The native Mosaic runtime owns subsequent PTX/cubin compilation and execution.

use ryft_core::kernels::{KernelCompilationError, KernelCompiler, KernelSchedule, VerifiedKernel};
use ryft_core::{ArrayType, ProgramError, TypeError};
use ryft_mlir::{Context, Module};
use sha2::{Digest, Sha256};
use thiserror::Error;

mod lowering;
pub mod synchronization;

/// GPU target, lowering and native source verification errors.
#[derive(Debug, Error)]
pub enum Error {
    /// A target or option violates its checked contract.
    #[error("invalid Mosaic GPU configuration: {message}")]
    Invalid {
        /// Specific configuration failure.
        message: String,
    },

    /// A valid portable operation has no implementation in the admitted baseline.
    #[error("mosaic GPU cannot lower `{operation}`: {reason}")]
    Unsupported {
        /// Canonical operation requiring support.
        operation: &'static str,

        /// Missing target or lowering contract.
        reason: String,
    },

    /// Required CTA-local storage exceeds the explicit target or schedule limit.
    #[error("mosaic GPU requires {required} shared-memory bytes, exceeding the limit {maximum}")]
    SharedMemory {
        /// Computed byte count, including alignment and temporary tile storage.
        required: usize,

        /// Smaller of the target capacity and requested scratch budget.
        maximum: usize,
    },

    /// Canonical type validation failed.
    #[error(transparent)]
    Type(#[from] TypeError),

    /// Canonical program inspection failed.
    #[error(transparent)]
    Program(#[from] ProgramError),

    /// Native MLIR construction or verification failed.
    #[error(transparent)]
    Mlir(#[from] ryft_mlir::Error),

    /// The emitted communication schedule is invalid.
    #[error("invalid Mosaic GPU synchronization: {message}")]
    Synchronization {
        /// Communication simulator's exact diagnostic.
        message: String,
    },
}

/// Exact GPU architecture and execution-agent contract.
///
/// Thread-block membership affects collective communication and is never a result-preserving schedule hint.
/// The execution integration must match this architecture and validate the declared resource capacity against every
/// selected device. Compiler payloads include the architecture so native hash caches cannot cross device targets.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Target {
    /// Exact CUDA compute capability.
    compute_capability: (u32, u32),

    /// Number of agents participating in every CTA-wide barrier.
    threads_per_block: u32,

    /// Admitted CTA-local memory capacity; execution facts must support this bound.
    maximum_shared_memory_bytes: usize,
}

impl Target {
    /// Creates an exact Hopper or Blackwell target with 32 threads and a conservative 48 KiB shared-memory limit.
    pub fn new(major: u32, minor: u32) -> Result<Self, Error> {
        if !matches!(major, 9 | 10 | 12) || minor > 9 {
            return Err(Error::Invalid {
                message: format!("unsupported compute capability `{major}.{minor}`; expected Hopper or Blackwell"),
            });
        }
        Ok(Self { compute_capability: (major, minor), threads_per_block: 32, maximum_shared_memory_bytes: 48 * 1024 })
    }

    /// Returns the exact admitted CUDA compute capability.
    pub fn compute_capability(&self) -> (u32, u32) {
        self.compute_capability
    }

    /// Returns the number of threads in each logical program's CUDA block.
    pub fn threads_per_block(&self) -> u32 {
        self.threads_per_block
    }

    /// Returns the declared per-block shared-memory capacity.
    pub fn maximum_shared_memory_bytes(&self) -> usize {
        self.maximum_shared_memory_bytes
    }

    /// Sets an explicit positive block membership no larger than CUDA's 1,024-thread limit.
    pub fn with_threads_per_block(mut self, threads: u32) -> Result<Self, Error> {
        if threads == 0 || threads > 1024 {
            return Err(Error::Invalid { message: "threads per block must be between 1 and 1024".to_owned() });
        }
        self.threads_per_block = threads;
        Ok(self)
    }

    /// Sets a capacity which the execution integration must validate against actual device facts.
    /// A zero capacity is valid and admits only kernels requiring no shared storage.
    pub fn with_maximum_shared_memory_bytes(mut self, bytes: usize) -> Result<Self, Error> {
        if bytes > i32::MAX as usize {
            return Err(Error::Invalid {
                message: "shared-memory capacity exceeds the native byte-count range".to_owned(),
            });
        }
        self.maximum_shared_memory_bytes = bytes;
        Ok(self)
    }
}

/// Source-compilation limits independent of numerical semantics and target agent membership.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Options {
    /// Maximum source work admitted before native IR construction: instructions plus individual literal elements.
    maximum_instructions: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self { maximum_instructions: 16_384 }
    }
}

impl Options {
    /// Returns the maximum source work across nested regions, counting instructions and individual literal elements.
    pub fn maximum_instructions(&self) -> usize {
        self.maximum_instructions
    }

    /// Limits source construction work. Each instruction and each array-literal element contributes one unit;
    /// dimension literals contribute one unit. Zero admits only bodies without instructions or literals.
    pub fn with_maximum_instructions(mut self, maximum: usize) -> Self {
        self.maximum_instructions = maximum;
        self
    }
}

/// Checked binary compiler input and the native host-buffer ABI derived from a portable kernel.
///
/// Native buffer order is every logical input followed by every logical result, retaining aliased duplicates.
/// A body with ordered assertions reserves a token slot after the inputs and another after the results. Tokens carry
/// no array storage and are omitted from [`Self::argument_types`], but are counted in [`Self::parameter_slots`].
/// Read-only parameters use their input slot; mutable parameters use their result slot. No runtime pointer, client,
/// executable, or XLA-owned envelope is stored here. The binary module is hashed exactly as consumed by Mosaic.
#[derive(Clone, Debug)]
pub struct CompiledKernel {
    /// Versioned binary Mosaic GPU source.
    module: Vec<u8>,

    /// SHA-256 of the exact module bytes.
    hash: [u8; 32],

    /// Exact compilation and collective-membership target.
    target: Target,

    /// Array input and result types in native order, excluding assertion tokens.
    argument_types: Vec<ArrayType>,

    /// Native slot selecting each portable body reference.
    parameter_slots: Vec<usize>,

    /// Actual CTA-local storage required by lowered values and scratch.
    shared_memory_bytes: usize,
}

impl CompiledKernel {
    /// Returns binary MLIR with the pinned Mosaic source serde version.
    pub fn module(&self) -> &[u8] {
        &self.module
    }

    /// Returns the exact native cache key for this binary compiler input.
    pub fn hash(&self) -> &[u8; 32] {
        &self.hash
    }

    /// Returns the immutable target required by this payload.
    pub fn target(&self) -> &Target {
        &self.target
    }

    /// Returns array inputs followed by results, retaining aliases and excluding assertion tokens.
    pub fn argument_types(&self) -> &[ArrayType] {
        &self.argument_types
    }

    /// Returns native buffer indices in logical body-reference order.
    pub fn parameter_slots(&self) -> &[usize] {
        &self.parameter_slots
    }

    /// Returns actual shared-memory usage, including temporary value storage.
    pub fn shared_memory_bytes(&self) -> usize {
        self.shared_memory_bytes
    }
}

/// Stateless compiler for the admitted portable Mosaic GPU baseline.
#[derive(Copy, Clone, Debug, Default)]
pub struct Compiler;

impl Compiler {
    /// Builds typed MLIR for inspection before the source serialization pass.
    pub fn module<'c, 't>(
        &self,
        context: &'c Context<'t>,
        kernel: &VerifiedKernel<'_>,
        target: &Target,
        options: &Options,
        schedule: &KernelSchedule,
    ) -> Result<Module<'c, 't>, Error> {
        lowering::module(context, kernel, target, options, schedule).map(|(module, _, _, _)| module)
    }
}

impl KernelCompiler for Compiler {
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
        lowering::validate(kernel, target, options, schedule)
            .map(|_| ())
            .map_err(KernelCompilationError::Compiler)
    }

    fn configuration_key(
        &self,
        target: &Target,
        options: &Options,
        schedule: &KernelSchedule,
    ) -> Result<Vec<u8>, KernelCompilationError<Error>> {
        Ok(format!(
            "mosaic gpu 1; xla {}; jax {}; serde {}; resource {}; {target:?}; {options:?}; {schedule:?}",
            ryft_xla_sys::XLA_COMMIT,
            ryft_xla_sys::JAX_COMMIT,
            ryft_xla_sys::mlir::dialects::mosaic::gpu::MOSAIC_GPU_SERDE_VERSION,
            ryft_xla_sys::mlir::dialects::mosaic::gpu::MOSAIC_GPU_RESOURCE_SCHEMA_VERSION,
        )
        .into_bytes())
    }

    fn compile(
        &self,
        kernel: &VerifiedKernel<'_>,
        target: &Target,
        options: &Options,
        schedule: &KernelSchedule,
    ) -> Result<CompiledKernel, KernelCompilationError<Error>> {
        let context = Context::new();
        let result = (|| {
            let (module, argument_types, parameter_slots, shared_memory_bytes) =
                lowering::module(&context, kernel, target, options, schedule)?;
            let bytes = lowering::serialize(&module)?;
            Ok(CompiledKernel {
                hash: Sha256::digest(&bytes).into(),
                module: bytes,
                target: target.clone(),
                argument_types,
                parameter_slots,
                shared_memory_bytes,
            })
        })();
        result.map_err(KernelCompilationError::Compiler)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;
    use ryft_core::kernels::{
        Grid, KernelCallOperation, KernelDefinition, KernelParameterAccess, whole_array_parameter,
    };
    use ryft_core::{
        ArrayIrOperation, ArrayOperation, Context as CoreContext, DataType, DimensionBounds,
        DimensionFromScalarOperation, DimensionVariable, ReferenceRead, ReferenceWrite, SqrtOperation,
    };
    use ryft_mlir::dialects::mosaic::gpu::mosaic_gpu_serde_version;
    use ryft_xla_sys::mlir::dialects::mosaic::gpu::MOSAIC_GPU_SERDE_VERSION;

    use super::*;

    /// Scalar read/write source exercising the actual functional alias and native duplicate-buffer contract.
    fn scalar_definition() -> KernelDefinition {
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![whole_array_parameter(ArrayType::scalar(DataType::I32), KernelParameterAccess::ReadWrite).unwrap()],
        )
        .unwrap();
        KernelDefinition::trace(operation, |(references, _)| {
            references[0].write(&references[0].read()?)?;
            Ok(())
        })
        .unwrap()
    }

    #[test]
    fn test_target_new() {
        let target = Target::new(9, 0).unwrap();
        assert_eq!(target.compute_capability(), (9, 0));
        assert_eq!(target.threads_per_block(), 32);
        assert_eq!(target.maximum_shared_memory_bytes(), 48 * 1024);
        assert_eq!(HashMap::from([(target.clone(), "hopper")]).get(&target), Some(&"hopper"));
        assert_ne!(target, Target::new(12, 1).unwrap());
        assert!(matches!(Target::new(8, 0), Err(Error::Invalid { message })
            if message == "unsupported compute capability `8.0`; expected Hopper or Blackwell"));
    }

    #[test]
    fn test_target_with_threads_per_block() {
        assert_eq!(Target::new(9, 0).unwrap().with_threads_per_block(1024).unwrap().threads_per_block(), 1024);
        assert!(matches!(Target::new(9, 0).unwrap().with_threads_per_block(0), Err(Error::Invalid { message })
            if message == "threads per block must be between 1 and 1024"));
        assert!(matches!(Target::new(9, 0).unwrap().with_threads_per_block(1025), Err(Error::Invalid { message })
            if message == "threads per block must be between 1 and 1024"));
    }

    #[test]
    fn test_target_with_maximum_shared_memory_bytes() {
        assert_eq!(
            Target::new(9, 0)
                .unwrap()
                .with_maximum_shared_memory_bytes(0)
                .unwrap()
                .maximum_shared_memory_bytes(),
            0
        );
        assert!(matches!(Target::new(9, 0).unwrap().with_maximum_shared_memory_bytes(i32::MAX as usize + 1),
            Err(Error::Invalid { message }) if message == "shared-memory capacity exceeds the native byte-count range"));
    }

    #[test]
    fn test_options_default() {
        assert_eq!(Options::default().maximum_instructions(), 16_384);
    }

    #[test]
    fn test_options_with_maximum_instructions() {
        assert_eq!(Options::default().with_maximum_instructions(0).maximum_instructions(), 0);
        assert_ne!(Options::default(), Options::default().with_maximum_instructions(1));
    }

    #[test]
    fn test_compiler_module() {
        use ryft_mlir::{Operation, WalkOrder, WalkResult};

        let definition = scalar_definition();
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let context = Context::new();
        let module = Compiler
            .module(&context, &verified, &Target::new(9, 0).unwrap(), &Options::default(), &KernelSchedule::default())
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(mosaic_gpu_serde_version(&module), Ok(None));
        let mut launches = Vec::new();
        module.as_operation().unwrap().walk(WalkOrder::PreOrder, |operation| {
            if operation.name().as_str() == Ok("gpu.launch") {
                launches.push(operation.region_count());
            }
            WalkResult::Advance
        });
        assert_eq!(launches, vec![1]);
    }

    #[test]
    fn test_compiler_admit() {
        let definition = scalar_definition();
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let target = Target::new(9, 0).unwrap();
        Compiler.admit(&verified, &target, &Options::default(), &KernelSchedule::default()).unwrap();
        assert!(matches!(
            Compiler.admit(
                &verified,
                &target,
                &Options::default(),
                &KernelSchedule::default().with_maximum_scratch_bytes(15)
            ),
            Err(KernelCompilationError::Compiler(Error::SharedMemory { required: 16, maximum: 15 }))
        ));
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![whole_array_parameter(ArrayType::scalar(DataType::F32), KernelParameterAccess::ReadOnly).unwrap()],
        )
        .unwrap();
        let unsupported = KernelDefinition::trace(operation, |(references, _)| {
            let value = references[0].read()?;
            value.context().bind(
                ArrayIrOperation::from(ArrayOperation::Sqrt(SqrtOperation::new())),
                vec![],
                &[value.clone()],
            )?;
            Ok(())
        })
        .unwrap();
        let verified = VerifiedKernel::new(&unsupported, 1).unwrap();
        assert!(matches!(Compiler.admit(&verified, &target, &Options::default(), &KernelSchedule::default()),
            Err(KernelCompilationError::Compiler(Error::Unsupported { operation: "sqrt", reason }))
                if reason == "operation or its metadata has no baseline scalar implementation"));
    }

    #[test]
    fn test_compiler_configuration_key() {
        let target = Target::new(9, 0).unwrap();
        let options = Options::default();
        let schedule = KernelSchedule::default();
        let key = Compiler.configuration_key(&target, &options, &schedule).unwrap();
        assert_eq!(key, Compiler.configuration_key(&target, &options, &schedule).unwrap());
        assert_ne!(key, Compiler.configuration_key(&Target::new(12, 1).unwrap(), &options, &schedule).unwrap());
        assert_ne!(key, Compiler.configuration_key(&target, &options.with_maximum_instructions(1), &schedule).unwrap());
    }

    #[test]
    fn test_compiler_compile() {
        let definition = scalar_definition();
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let target = Target::new(9, 0).unwrap();
        let output = Compiler.compile(&verified, &target, &Options::default(), &KernelSchedule::default()).unwrap();
        assert_eq!(output.argument_types(), &[ArrayType::scalar(DataType::I32), ArrayType::scalar(DataType::I32)]);
        assert_eq!(output.parameter_slots(), &[1]);
        assert_eq!(output.shared_memory_bytes(), 16);
        assert_eq!(output.target(), &target);
        assert_eq!(output.hash(), &<[u8; 32]>::from(Sha256::digest(output.module())));
        let context = Context::new();
        context.allow_unregistered_dialects();
        let module = context.parse_module_from_bytes(output.module()).unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(mosaic_gpu_serde_version(&module), Ok(Some(i64::from(MOSAIC_GPU_SERDE_VERSION))));
        let different = target.with_threads_per_block(64).unwrap();
        assert_eq!(output.target().threads_per_block(), 32);
        assert_ne!(output.target(), &different);
    }

    #[test]
    fn test_compiler_compile_assertion_slots() {
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![whole_array_parameter(ArrayType::scalar(DataType::I32), KernelParameterAccess::ReadWrite).unwrap()],
        )
        .unwrap();
        let definition = KernelDefinition::trace(operation, |(references, _)| {
            let value = references[0].read()?;
            value.context().bind(
                ArrayIrOperation::DimensionFromScalar(DimensionFromScalarOperation::new(DimensionVariable::new(
                    "extent",
                    DimensionBounds::new(0, Some(8)).unwrap(),
                ))),
                vec![],
                &[value.clone()],
            )?;
            references[0].write(&value)?;
            Ok(())
        })
        .unwrap();
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let output = Compiler
            .compile(&verified, &Target::new(9, 0).unwrap(), &Options::default(), &KernelSchedule::default())
            .unwrap();
        assert_eq!(output.argument_types(), &[ArrayType::scalar(DataType::I32), ArrayType::scalar(DataType::I32)]);
        assert_eq!(output.parameter_slots(), &[2]);
        assert_eq!(output.hash(), &<[u8; 32]>::from(Sha256::digest(output.module())));
    }
}
