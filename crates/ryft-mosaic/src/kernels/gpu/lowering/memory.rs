//! Native global-to-shared copy groups and their emitted CTA communication contract.

use ryft_core::kernels::KernelOperation;
use ryft_core::kernels::memory::ASYNC_COPY_OPERATION_NAME;
use ryft_core::{ArrayIrOperation, ArrayReferenceView, ArraySliceAxis, DataType, Operation};
use ryft_mlir::dialects::nvgpu;
use ryft_mlir::{Block, DetachedBlock};

use crate::kernels::gpu::Error;
use crate::kernels::gpu::lowering::{KernelValue, Lowering, Reference, append};
use crate::kernels::gpu::synchronization::SynchronizationEvent;

/// Rejects instructions whose cooperative lowering would rendezvous before pending copies have been completed.
/// Pure dimension arithmetic and view construction may occur between issue and wait; copying itself also remains
/// asynchronous. This conservative baseline deliberately excludes overlapping unrelated cooperative array work.
pub(super) fn validate_async_copies<'o>(
    operations: impl IntoIterator<Item = &'o KernelOperation>,
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
    ) -> Result<KernelValue<'c, 't>, Error> {
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
        let count = source.buffer.r#type.element_count()?.unwrap();
        let instruction = self.current_instruction.unwrap();
        let synchronization = self.synchronization.as_mut().unwrap();
        let participants = synchronization.participants().get();
        for thread in 0..participants {
            for index in (thread as usize..count).step_by(participants as usize) {
                let mut remaining = index;
                let mut axes = vec![ArraySliceAxis::new(0, 1, 1); source.shape.len()];
                for axis in (0..axes.len()).rev() {
                    axes[axis] = ArraySliceAxis::new(remaining % source.shape[axis], 1, 1);
                    remaining /= source.shape[axis];
                }
                let view = ArrayReferenceView::Slice { axes };
                synchronization
                    .record(
                        thread,
                        instruction,
                        SynchronizationEvent::AsyncCopy {
                            source: (source.buffer.owner, view.clone()),
                            destination: (destination.buffer.owner, view),
                        },
                    )
                    .map_err(|error| Error::Synchronization { message: error.to_string() })?;
            }
            synchronization
                .record(thread, instruction, SynchronizationEvent::CommitGroup)
                .map_err(|error| Error::Synchronization { message: error.to_string() })?;
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
        append(block, nvgpu::device_async_create_group(&[], self.location)?)
    }

    /// Completes all preceding committed groups in each issuing thread, then publishes those writes to the full CTA.
    pub(super) fn wait(&mut self, block: &mut DetachedBlock<'c, 't>, token: KernelValue<'c, 't>) -> Result<(), Error> {
        let instruction = self.current_instruction.unwrap();
        let synchronization = self.synchronization.as_mut().unwrap();
        for thread in 0..synchronization.participants().get() {
            synchronization
                .record(thread, instruction, SynchronizationEvent::WaitGroup)
                .map_err(|error| Error::Synchronization { message: error.to_string() })?;
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
        body: impl FnOnce(
            &mut Lowering<'c, 't>,
            &mut DetachedBlock<'c, 't>,
            &Reference<'c, 't>,
            &Reference<'c, 't>,
        ) -> Result<(), Error>,
    ) -> Result<Module<'c, 't>, Error> {
        let r#type = ArrayType::new_static(DataType::F32, [3]);
        module::build(
            context,
            &Target::new(9, 0)?,
            "copy",
            &[(0, r#type.clone())],
            &[r#type.clone()],
            [1, 1, 1],
            |block, globals, shared| {
                let source_id = ValueId::new(RegionId::new(0), AtomId::new(0));
                let destination_id = ValueId::new(RegionId::new(0), AtomId::new(1));
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
                    storage: HashMap::new(),
                    synchronization: Some(synchronization),
                    current_instruction: Some(InstructionId::new(RegionId::new(0), 0)),
                    next_barrier: 0,
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
        let copy = KernelOperation::AsyncCopy(AsyncCopyOperation);
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
    fn test_lowering_async_copy() {
        let context = Context::new();
        let module = copy_module(&context, |lowering, block, source, destination| {
            lowering.async_copy(block, source, destination)?;
            assert_eq!(
                lowering.synchronization.as_ref().unwrap().simulate(),
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
        let result = copy_module(&context, |lowering, block, _, destination| {
            lowering.async_copy(block, destination, destination).map(|_| ())
        });
        assert!(matches!(result, Err(Error::Unsupported { operation: ASYNC_COPY_OPERATION_NAME, reason })
            if reason == "native async copies require a global source and a shared destination"));
    }

    #[test]
    fn test_lowering_wait() {
        let context = Context::new();
        let module = copy_module(&context, |lowering, block, source, destination| {
            let token = lowering.async_copy(block, source, destination)?;
            lowering.current_instruction = Some(InstructionId::new(RegionId::new(0), 1));
            lowering.wait(block, token)?;
            assert_eq!(lowering.synchronization.as_ref().unwrap().simulate().unwrap().len(), 99);
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
}
