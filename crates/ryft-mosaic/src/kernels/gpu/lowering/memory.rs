//! Native global-to-shared copy groups and their emitted CTA communication contract.

use ryft_core::kernels::memory::ASYNC_COPY_OPERATION_NAME;
use ryft_core::kernels::{KernelExtension, KernelOperation, KernelParameterAccess, VerifiedKernel};
use ryft_core::{
    ArrayAddressing, ArrayIrOperation, ArrayReferenceView, ArraySliceAxis, ArrayType, DataType, Operation, Typed,
    ValueId,
};
use ryft_mlir::dialects::{arith, llvm, memref, nvgpu, nvvm, scf};
use ryft_mlir::{Attribute, Block, DetachedBlock, TypeRef, UnknownLocationRef};

use crate::kernels::gpu::lowering::{KernelValue, Lowering, Reference, append};
use crate::kernels::gpu::synchronization::SynchronizationEvent;
use crate::kernels::gpu::{Error, Target};

/// One canonical completion token and the full global parameter from which its TMA descriptor is built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct TmaCopyPlan {
    /// Canonical asynchronous-copy result, also identifying its auxiliary shared barrier storage.
    pub(super) token: ValueId,

    /// Index in the kernel call's reference parameter list.
    pub(super) source_parameter: usize,

    /// Canonical scratch root receiving the tensor transfer; native TMA requires 128-byte base alignment.
    pub(super) destination: ValueId,
}

/// Native completion representation; neither form escapes into the portable value family.
#[derive(Copy, Clone)]
pub(super) enum CopyToken<'c, 't> {
    /// Per-thread cp.async group token.
    Group(KernelValue<'c, 't>),

    /// Shared transaction barrier owned by the canonical completion token.
    Tma { barrier: KernelValue<'c, 't>, owner: ValueId },
}

/// Plans whole-root TMA transfers before MLIR construction or host descriptor initialization.
pub(super) fn plan<Extension: KernelExtension>(
    kernel: &VerifiedKernel<'_, Extension>,
    target: &Target,
) -> Result<Vec<TmaCopyPlan>, Error> {
    if target.compute_capability().0 < 9 {
        return Err(tma_error("TMA requires compute capability 9.0 or newer"));
    }
    let body = kernel.definition().body();
    let entry = body.entry_region_ref();
    let mut copies = Vec::new();
    for (id, instruction) in entry.instructions_in_closure() {
        if !matches!(instruction.operation(), KernelOperation::AsyncCopy(_)) {
            continue;
        }
        if id.region() != entry.id() {
            return Err(tma_error("TMA copies require the entry region"));
        }
        let source_parameter = entry
            .input_ids()
            .iter()
            .position(|value| value == &instruction.inputs()[0])
            .ok_or_else(|| tma_error("TMA source must be a full global reference parameter"))?;
        let parameter = &kernel.definition().operation().parameters()[source_parameter];
        let parameter_type = parameter.r#type();
        let shape = parameter_type.static_shape().ok_or_else(|| tma_error("TMA shapes must be static"))?;
        if parameter.access() != KernelParameterAccess::ReadOnly
            || parameter.mapping().block_shape() != shape.dimensions()
            || !parameter.mapping().tiling_axes().is_some_and(|axes| axes.iter().all(Option::is_none))
        {
            return Err(tma_error("TMA source must be a full read-only global reference parameter"));
        }
        validate_tma_type(&parameter.r#type())?;
        let destination =
            entry.instructions().iter().find(|candidate| candidate.outputs().contains(&instruction.inputs()[1]));
        if !destination.is_some_and(|instruction| matches!(instruction.operation(), KernelOperation::Scratch(_))) {
            return Err(tma_error("TMA destination must be a full scratch allocation"));
        }
        copies.push(TmaCopyPlan {
            token: ValueId::new(entry.id(), instruction.outputs()[0]),
            source_parameter,
            destination: ValueId::new(entry.id(), instruction.inputs()[1]),
        });
    }
    Ok(copies)
}

/// Validates the exact whole-window, unswizzled native descriptor contract. Host initialization must not abort.
pub(super) fn validate_tma_type(r#type: &ArrayType) -> Result<(), Error> {
    super::validation::checked_array(r#type)?;
    let shape = r#type.static_shape().unwrap();
    if !(1..=5).contains(&shape.dimensions().len())
        || shape.dimensions().iter().any(|extent| !(1..=256).contains(extent))
    {
        return Err(tma_error("TMA whole-window shapes require rank 1 through 5 and extents 1 through 256"));
    }
    let bytes = match r#type.data_type() {
        DataType::F16 | DataType::BF16 => 2usize,
        DataType::I32 | DataType::U32 | DataType::F32 => 4usize,
        DataType::I64 | DataType::U64 | DataType::F64 => 8usize,
        _ => return Err(tma_error("TMA requires 2-, 4-, or 8-byte scalar elements")),
    };
    let mut stride = bytes;
    for extent in shape.dimensions().iter().rev() {
        stride = stride.checked_mul(*extent).ok_or_else(|| tma_error("TMA byte strides overflow"))?;
        if stride % 16 != 0 || stride >= (1usize << 40) {
            return Err(tma_error(
                "TMA dense byte strides and innermost window bytes must be multiples of 16 below 2^40",
            ));
        }
    }
    if stride > (1 << 20) - 1 {
        return Err(tma_error("TMA transfer exceeds the transaction barrier byte-count limit"));
    }
    Ok(())
}

/// Constructs a precise asynchronous-copy admission diagnostic.
fn tma_error(reason: &str) -> Error {
    Error::Unsupported { operation: ASYNC_COPY_OPERATION_NAME, reason: reason.to_owned() }
}

/// Rejects instructions whose cooperative lowering would rendezvous before pending copies have been completed.
/// Pure dimension arithmetic and view construction may occur between issue and wait; copying itself also remains
/// asynchronous. This conservative baseline deliberately excludes overlapping unrelated cooperative array work.
pub(super) fn validate_async_copies<'o, Extension: 'o + KernelExtension>(
    operations: impl IntoIterator<Item = &'o KernelOperation<Extension>>,
) -> Result<(), Error> {
    let mut pending = false;
    for operation in operations {
        match operation {
            KernelOperation::AsyncCopy(_) => pending = true,
            KernelOperation::Wait(_) => pending = false,
            KernelOperation::Scratch(_)
            | KernelOperation::Portable(
                ArrayIrOperation::Dimension(_)
                | ArrayIrOperation::DimensionSize(_)
                | ArrayIrOperation::ReferenceIndex(_)
                | ArrayIrOperation::ReferenceSlice(_)
                | ArrayIrOperation::ReferenceDynamicIndex(_),
            ) => {}
            _ if pending => {
                return Err(Error::Unsupported {
                    operation: operation.name(),
                    reason: "cooperative work between async copy and wait is outside the native baseline".to_owned(),
                });
            }
            _ => {}
        }
    }
    if pending {
        return Err(Error::Unsupported {
            operation: ASYNC_COPY_OPERATION_NAME,
            reason: "native async copies must be waited before region exit".to_owned(),
        });
    }
    Ok(())
}

impl<'c, 't> Lowering<'c, 't> {
    /// Issues one aligned 4- or 8-byte copy per element, partitioned by CUDA thread. The initial baseline requires
    /// whole, statically in-bounds roots with dense addressing, a global source, and a shared destination. Shared base
    /// alignment is established by the module's outlining attributes; native input allocations must satisfy their
    /// element alignment. No synchronous read/write replaces the asynchronous instructions.
    pub(super) fn async_copy(
        &mut self,
        block: &mut DetachedBlock<'c, 't>,
        source: &Reference<'c, 't>,
        destination: &Reference<'c, 't>,
        token: ValueId,
    ) -> Result<CopyToken<'c, 't>, Error> {
        if source.buffer.shared || !destination.buffer.shared {
            return Err(Error::Unsupported {
                operation: ASYNC_COPY_OPERATION_NAME,
                reason: "native async copies require a global source and a shared destination".to_owned(),
            });
        }
        if !matches!(
            source.buffer.r#type.data_type(),
            DataType::I32 | DataType::U32 | DataType::F32 | DataType::I64 | DataType::U64 | DataType::F64
        ) || source.buffer.r#type.data_type() != destination.buffer.r#type.data_type()
            || source.shape != destination.shape
        {
            return Err(Error::Unsupported {
                operation: ASYNC_COPY_OPERATION_NAME,
                reason: "native async copies require matching shapes and 4- or 8-byte elements".to_owned(),
            });
        }
        for reference in [source, destination] {
            let root_shape = reference.buffer.r#type.static_shape().unwrap();
            let full_view = ArrayReferenceView::Slice {
                axes: root_shape.dimensions().iter().map(|extent| ArraySliceAxis::new(0, *extent, 1)).collect(),
            };
            let expected_strides = (0..reference.shape.len())
                .map(|axis| reference.shape[axis + 1..].iter().product::<usize>())
                .collect::<Vec<_>>();
            if reference.shape != root_shape.dimensions()
                || reference.static_view.as_ref() != Some(&full_view)
                || reference.strides != expected_strides
                || reference.predicate.is_some()
            {
                return Err(Error::Unsupported {
                    operation: ASYNC_COPY_OPERATION_NAME,
                    reason: "native async copies require whole statically in-bounds dense reference roots".to_owned(),
                });
            }
        }
        if let Some(descriptor) = self.tma_descriptors.get(&token).copied() {
            return self.tma_copy(block, source, destination, token, descriptor);
        }
        let count = source.buffer.r#type.element_count()?.unwrap();
        let participants = self.synchronization.as_ref().unwrap()[0].participants().get();
        for thread in 0..participants {
            for index in (thread as usize..count).step_by(participants as usize) {
                let mut remaining = index;
                let mut axes = vec![ArraySliceAxis::new(0, 1, 1); source.shape.len()];
                for axis in (0..axes.len()).rev() {
                    axes[axis] = ArraySliceAxis::new(remaining % source.shape[axis], 1, 1);
                    remaining /= source.shape[axis];
                }
                let view = ArrayReferenceView::Slice { axes };
                self.record_synchronization(
                    None,
                    thread,
                    SynchronizationEvent::AsyncCopy {
                        source: (source.buffer.owner, view.clone()),
                        destination: (destination.buffer.owner, view),
                    },
                )?;
            }
            self.record_synchronization(None, thread, SynchronizationEvent::CommitGroup)?;
        }
        self.distributed(block, count, |lowering, body, index| {
            body.append_operation(nvgpu::device_async_copy(
                destination.buffer.value,
                &[index],
                source.buffer.value,
                &[index],
                1,
                None,
                false,
                lowering.location,
            )?)?;
            Ok(())
        })?;
        // Commit is outside the strided loop so every lane commits exactly one group, including lanes with no copies.
        // Its empty SSA operand list is intentional: PTX commit_group captures every preceding uncommitted cp.async.
        Ok(CopyToken::Group(append(block, nvgpu::device_async_create_group(&[], self.location)?)?))
    }

    /// Issues one whole-window tensor transfer with a leader-owned transaction barrier.
    fn tma_copy(
        &mut self,
        block: &mut DetachedBlock<'c, 't>,
        source: &Reference<'c, 't>,
        destination: &Reference<'c, 't>,
        owner: ValueId,
        descriptor: KernelValue<'c, 't>,
    ) -> Result<CopyToken<'c, 't>, Error> {
        validate_tma_type(&source.buffer.r#type)?;
        let bytes = ArrayAddressing::new(source.buffer.r#type.clone())?.storage_byte_len();
        let barrier = self.shared_pointer(block, self.storage[&owner].value)?;
        let destination_pointer = self.shared_pointer(block, destination.buffer.value)?;
        let destination_pointer = append(
            block,
            llvm::addrspacecast(destination_pointer, self.context.llvm_pointer_type(7)?, self.location)?,
        )?;
        self.record_synchronization(
            None,
            0,
            SynchronizationEvent::InitializeBarrier { barrier: owner, arrival_count: 1 },
        )?;
        self.tma_leader(block, |lowering, body| {
            let one = lowering.tma_i32(body, 1)?;
            body.append_operation(nvvm::mbarrier_init(&[barrier, one], &[], &[], false, lowering.location)?)?;
            body.append_operation(nvvm::fence_mbarrier_init(&[], &[], &[], false, lowering.location)?)?;
            Ok(())
        })?;
        self.barrier(block)?;
        self.record_synchronization(None, 0, SynchronizationEvent::ArriveExpectTransaction { barrier: owner, bytes })?;
        self.record_synchronization(
            None,
            0,
            SynchronizationEvent::TmaCopy {
                barrier: owner,
                source: (source.buffer.owner, source.static_view.clone().unwrap()),
                destination: (destination.buffer.owner, destination.static_view.clone().unwrap()),
            },
        )?;
        self.tma_leader(block, |lowering, body| {
            let bytes = lowering.tma_i32(body, bytes as i32)?;
            body.append_operation(nvvm::mbarrier_arrive_expect_tx(
                &[barrier, bytes],
                &[],
                &[],
                false,
                lowering.location,
            )?)?;
            let zero = lowering.tma_i32(body, 0)?;
            let mut operands = vec![destination_pointer, descriptor];
            operands.extend(std::iter::repeat_n(zero, source.shape.len()));
            operands.push(barrier);
            let segments =
                lowering.context.dense_i32_array_attribute(&[1, 1, source.shape.len() as i32, 1, 0, 0, 0, 0])?;
            body.append_operation(nvvm::cp_async_bulk_tensor_shared_cluster_global(
                &operands,
                &[],
                &[("operandSegmentSizes", segments.as_ref())],
                false,
                lowering.location,
            )?)?;
            Ok(())
        })?;
        Ok(CopyToken::Tma { barrier, owner })
    }

    /// Acquires only the selected TMA completion, publishes its writes, and retires its auxiliary barrier.
    fn wait_tma(
        &mut self,
        block: &mut DetachedBlock<'c, 't>,
        barrier: KernelValue<'c, 't>,
        owner: ValueId,
    ) -> Result<(), Error> {
        self.record_synchronization(None, 0, SynchronizationEvent::WaitBarrier { barrier: owner, generation: 0 })?;
        self.tma_leader(block, |lowering, body| {
            let phase = lowering.tma_i32(body, 0)?;
            let ticks = lowering.tma_i32(body, 10_000_000)?;
            body.append_operation(nvvm::mbarrier_try_wait_parity(
                &[barrier, phase, ticks],
                &[],
                &[],
                false,
                lowering.location,
            )?)?;
            Ok(())
        })?;
        self.barrier(block)?;
        self.record_synchronization(None, 0, SynchronizationEvent::InvalidateBarrier { barrier: owner })?;
        self.tma_leader(block, |lowering, body| {
            body.append_operation(nvvm::mbarrier_inval(&[barrier], &[], &[], false, lowering.location)?)?;
            Ok(())
        })?;
        self.barrier(block)
    }

    /// Converts a dense shared allocation's aligned base to an explicit address-space-three pointer.
    fn shared_pointer(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        value: KernelValue<'c, 't>,
    ) -> Result<KernelValue<'c, 't>, Error> {
        let address = append(block, memref::extract_aligned_pointer_as_index(value, self.location)?)?;
        let address =
            append(block, arith::index_cast(address, self.context.signless_integer_type(64), self.location)?)?;
        append(block, llvm::inttoptr(address, self.context.llvm_pointer_type(3)?, self.location)?)
    }

    /// Creates an integer operand for the native tensor/barrier ABI.
    fn tma_i32(&self, block: &mut DetachedBlock<'c, 't>, value: i32) -> Result<KernelValue<'c, 't>, Error> {
        append(
            block,
            arith::constant(
                self.context.integer_attribute(self.context.signless_integer_type(32), value as i64),
                self.location,
            )?,
        )
    }

    /// Restricts transaction-engine issue and lifecycle operations to the first CUDA thread.
    fn tma_leader<F>(&mut self, block: &mut DetachedBlock<'c, 't>, function: F) -> Result<(), Error>
    where
        F: FnOnce(&mut Self, &mut DetachedBlock<'c, 't>) -> Result<(), Error>,
    {
        let zero = self.index(block, 0)?;
        let predicate =
            append(block, arith::cmpi(self.thread, zero, arith::IntegerComparisonPredicate::Equal, self.location)?)?;
        let mut leader = self.context.block::<TypeRef<'c, 't>, UnknownLocationRef<'c, 't>>(&[]);
        function(self, &mut leader)?;
        leader.append_operation(scf::r#yield(&[], self.location)?)?;
        block.append_operation(scf::r#if(predicate, &[], leader.try_into()?, None, self.location)?)?;
        Ok(())
    }

    /// Completes all preceding committed groups in each issuing thread, then publishes those writes to the full CTA.
    pub(super) fn wait(&mut self, block: &mut DetachedBlock<'c, 't>, token: CopyToken<'c, 't>) -> Result<(), Error> {
        let token = match token {
            CopyToken::Group(token) => token,
            CopyToken::Tma { barrier, owner } => return self.wait_tma(block, barrier, owner),
        };
        for thread in 0..self.synchronization.as_ref().unwrap()[0].participants().get() {
            self.record_synchronization(None, thread, SynchronizationEvent::WaitGroup)?;
        }
        block.append_operation(nvgpu::device_async_wait(token, Some(0), self.location)?)?;
        self.barrier(block)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::num::NonZeroU32;

    use pretty_assertions::assert_eq;
    use ryft_core::kernels::{AsyncCopyOperation, WaitOperation};
    use ryft_core::{Array, ArrayOperation, ArrayType, AtomId, ConstantOperation, InstructionId, RegionId, ValueId};
    use ryft_mlir::{Block, Context, Module, Operation, Value, WalkOrder, WalkResult};

    use crate::kernels::gpu::Target;
    use crate::kernels::gpu::lowering::Buffer;
    use crate::kernels::gpu::lowering::module;
    use crate::kernels::gpu::synchronization::{CtaSynchronization, SynchronizationError};

    use super::*;

    /// Builds actual launch storage around an isolated copy/wait construction test.
    fn copy_module<'c, 't>(
        context: &'c Context<'t>,
        tma: bool,
        body: impl FnOnce(
            &mut Lowering<'c, 't>,
            &mut DetachedBlock<'c, 't>,
            &Reference<'c, 't>,
            &Reference<'c, 't>,
        ) -> Result<(), Error>,
    ) -> Result<Module<'c, 't>, Error> {
        let r#type = ArrayType::new_static(DataType::F32, [if tma { 128 } else { 3 }]);
        let mut shared_types = vec![r#type.clone()];
        if tma {
            shared_types.push(ArrayType::scalar(DataType::U64));
        }
        module::build(
            context,
            &Target::new(9, 0)?,
            "copy",
            &[(0, r#type.clone())],
            &shared_types,
            if tma { &[128, 16] } else { &[] },
            if tma { &[0] } else { &[] },
            [1, 1, 1],
            |block, globals, shared, descriptors| {
                let source_id = ValueId::new(RegionId::new(0), AtomId::new(0));
                let destination_id = ValueId::new(RegionId::new(0), AtomId::new(1));
                let token = ValueId::new(RegionId::new(0), AtomId::new(2));
                let synchronization = CtaSynchronization::new(
                    NonZeroU32::new(32).unwrap(),
                    HashMap::from([(source_id, r#type.clone()), (destination_id, r#type.clone())]),
                )
                .unwrap();
                let mut lowering = Lowering {
                    context,
                    location: context.unknown_location(),
                    thread: block.argument(3)?.as_ref(),
                    threads: block.argument(9)?.as_ref(),
                    storage: if tma {
                        HashMap::from([(
                            token,
                            Buffer {
                                owner: token,
                                shared: true,
                                value: shared[1],
                                r#type: ArrayType::scalar(DataType::U64),
                            },
                        )])
                    } else {
                        HashMap::new()
                    },
                    tma_descriptors: if tma { HashMap::from([(token, descriptors[0])]) } else { HashMap::new() },
                    cluster_rank: None,
                    synchronization: Some(vec![synchronization]),
                    current_instruction: Some(InstructionId::new(RegionId::new(0), 0)),
                    next_barrier: 0,
                    instruction_scratch: HashMap::new(),
                    mma_stages: 1,
                };
                let source = lowering.reference(
                    block,
                    Buffer { owner: source_id, shared: false, value: globals[0], r#type: r#type.clone() },
                )?;
                let destination = lowering.reference(
                    block,
                    Buffer { owner: destination_id, shared: true, value: shared[0], r#type: r#type.clone() },
                )?;
                body(&mut lowering, block, &source, &destination)
            },
        )
    }

    #[test]
    fn test_validate_async_copies() {
        let copy: KernelOperation = KernelOperation::AsyncCopy(AsyncCopyOperation);
        let wait = KernelOperation::Wait(WaitOperation);
        let compute = KernelOperation::Portable(ArrayIrOperation::Array(ArrayOperation::Constant(
            ConstantOperation::new(Array::vector(vec![1i32]).unwrap()),
        )));
        assert!(validate_async_copies([&compute, &copy, &wait, &compute]).is_ok());
        assert!(matches!(validate_async_copies([&copy, &compute, &wait]),
            Err(Error::Unsupported { operation: "constant", reason })
                if reason == "cooperative work between async copy and wait is outside the native baseline"));
        assert!(matches!(validate_async_copies([&copy]),
            Err(Error::Unsupported { operation: ASYNC_COPY_OPERATION_NAME, reason })
                if reason == "native async copies must be waited before region exit"));
    }

    #[test]
    fn test_validate_tma_type() {
        assert!(validate_tma_type(&ArrayType::new_static(DataType::F32, [128])).is_ok());
        assert!(validate_tma_type(&ArrayType::new_static(DataType::F32, [8, 16])).is_ok());
        assert!(validate_tma_type(&ArrayType::new_static(DataType::U64, [64])).is_ok());
        assert!(validate_tma_type(&ArrayType::new_static(DataType::F16, [128])).is_ok());
        assert!(validate_tma_type(&ArrayType::new_static(DataType::BF16, [8, 16])).is_ok());
    }

    #[test]
    fn test_validate_tma_type_rejects_partial_transaction() {
        assert!(matches!(validate_tma_type(&ArrayType::new_static(DataType::F32, [67])),
            Err(Error::Unsupported { operation: "async_copy", reason })
                if reason == "TMA dense byte strides and innermost window bytes must be multiples of 16 below 2^40"));
    }

    #[test]
    fn test_validate_tma_type_rejects_invalid_shape() {
        for shape in [vec![], vec![0], vec![260], vec![1, 1, 1, 1, 1, 4]] {
            assert!(matches!(validate_tma_type(&ArrayType::new_static(DataType::F32, shape)),
                Err(Error::Unsupported { operation: "async_copy", reason })
                    if reason == "TMA whole-window shapes require rank 1 through 5 and extents 1 through 256"));
        }
    }

    #[test]
    fn test_lowering_async_copy() {
        let context = Context::new();
        let module = copy_module(&context, false, |lowering, block, source, destination| {
            lowering.async_copy(block, source, destination, ValueId::new(RegionId::new(0), AtomId::new(2)))?;
            assert_eq!(
                lowering.synchronization.as_ref().unwrap()[0].simulate(),
                Err(SynchronizationError::PendingExit { thread: 0, copies: 1 })
            );
            Ok(())
        })
        .unwrap();
        assert!(module.verify().unwrap());
        let mut names = Vec::new();
        module.as_operation().unwrap().walk(WalkOrder::PreOrder, |operation| {
            if operation.name().as_str().unwrap().starts_with("nvgpu.") {
                names.push(operation.name().as_str().unwrap().to_owned());
            }
            WalkResult::Advance
        });
        assert_eq!(names, ["nvgpu.device_async_copy", "nvgpu.device_async_create_group"]);
    }

    #[test]
    fn test_lowering_async_copy_rejects_shared_source() {
        let context = Context::new();
        let result = copy_module(&context, false, |lowering, block, _, destination| {
            lowering
                .async_copy(block, destination, destination, ValueId::new(RegionId::new(0), AtomId::new(2)))
                .map(|_| ())
        });
        assert!(matches!(result, Err(Error::Unsupported { operation: ASYNC_COPY_OPERATION_NAME, reason })
            if reason == "native async copies require a global source and a shared destination"));
    }

    #[test]
    fn test_lowering_wait() {
        let context = Context::new();
        let module = copy_module(&context, false, |lowering, block, source, destination| {
            let token =
                lowering.async_copy(block, source, destination, ValueId::new(RegionId::new(0), AtomId::new(2)))?;
            lowering.current_instruction = Some(InstructionId::new(RegionId::new(0), 1));
            lowering.wait(block, token)?;
            assert_eq!(lowering.synchronization.as_ref().unwrap()[0].simulate().unwrap().len(), 99);
            Ok(())
        })
        .unwrap();
        assert!(module.verify().unwrap());
        let mut names = Vec::new();
        module.as_operation().unwrap().walk(WalkOrder::PreOrder, |operation| {
            if operation.name().as_str().unwrap().starts_with("nvgpu.")
                || operation.name().as_str() == Ok("gpu.barrier")
            {
                names.push(operation.name().as_str().unwrap().to_owned());
            }
            WalkResult::Advance
        });
        assert_eq!(
            names,
            ["nvgpu.device_async_copy", "nvgpu.device_async_create_group", "nvgpu.device_async_wait", "gpu.barrier"]
        );
    }
    #[test]
    fn test_lowering_wait_tma() {
        let context = Context::new();
        let module = copy_module(&context, true, |lowering, block, source, destination| {
            let token =
                lowering.async_copy(block, source, destination, ValueId::new(RegionId::new(0), AtomId::new(2)))?;
            lowering.current_instruction = Some(InstructionId::new(RegionId::new(0), 1));
            lowering.wait(block, token)?;
            assert!(lowering.synchronization.as_ref().unwrap()[0].simulate().is_ok());
            Ok(())
        })
        .unwrap();
        assert!(module.verify().unwrap());
        let mut names = Vec::new();
        module.as_operation().unwrap().walk(WalkOrder::PreOrder, |operation| {
            let name = operation.name().as_str().unwrap().to_owned();
            if name.starts_with("nvvm.") || name == "gpu.barrier" {
                names.push(name);
            }
            WalkResult::Advance
        });
        assert_eq!(
            names,
            [
                "nvvm.mbarrier.init",
                "nvvm.fence.mbarrier.init",
                "gpu.barrier",
                "nvvm.mbarrier.arrive.expect_tx",
                "nvvm.cp.async.bulk.tensor.shared.cluster.global",
                "nvvm.mbarrier.try_wait.parity",
                "gpu.barrier",
                "nvvm.mbarrier.inval",
                "gpu.barrier"
            ]
        );
        assert!(!module::serialize(&module).unwrap().is_empty());
    }
}
