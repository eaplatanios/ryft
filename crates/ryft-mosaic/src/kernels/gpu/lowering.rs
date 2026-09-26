//! Cooperative lowering of canonical array values and reference effects into a CUDA thread block.

use std::collections::HashMap;
use std::num::NonZeroU32;

use ryft_core::kernels::{GridExecution, KernelExtension, KernelOperation, KernelSchedule, VerifiedKernel};
use ryft_core::{
    Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayReferenceTransform,
    ArrayReferenceTransformIndex, ArraySliceAxis, ArrayType, Atom, AtomId, ConstantOperation, DataType, InstructionId,
    Operation as CoreOperation, ReferenceAccessOperation, ReferenceTransform, RegionRef, Typed, ValueId,
    validated_reference_access_descriptors,
};
use ryft_mlir::dialects::{arith, gpu, llvm, memref, nvvm, scf};
use ryft_mlir::{
    Block, Context, DetachedBlock, DetachedOp, Module, Operation, Type, TypeRef, UnknownLocationRef, Value, ValueRef,
};

use crate::kernels::gpu::lowering::memory::CopyToken;
use crate::kernels::gpu::lowering::module::element_type;
use crate::kernels::gpu::lowering::tmem::{Tmem, TmemToken};
use crate::kernels::gpu::synchronization::{CtaSynchronization, SynchronizationEvent};
use crate::kernels::gpu::{Error, GpuOperation, Options, Target, TmemOperation};

mod arrays;
mod dimensions;
mod memory;
mod mma;
mod module;
mod nvfp4;
mod tmem;
mod validation;

pub(super) use module::serialize;

/// A value whose owning MLIR context outlives every region built by this lowering.
type KernelValue<'c, 't> = ValueRef<'c, 'c, 't>;

/// Physical flat storage paired with its canonical logical array type.
#[derive(Clone)]
struct Buffer<'c, 't> {
    /// Canonical owner shared by every alias of this physical allocation.
    owner: ValueId,

    /// Whether this buffer is an actual workgroup attribution rather than a global argument.
    shared: bool,

    /// Flat global or workgroup memref.
    value: KernelValue<'c, 't>,

    /// Logical shape, element type, and placement constraints.
    r#type: ArrayType,
}

/// An allocation-preserving logical view; validity is checked before every physical reference access.
#[derive(Clone)]
struct Reference<'c, 't> {
    /// Exact root-relative selection when established statically; dynamic windows carry no speculative proof.
    static_transform: Option<ArrayReferenceTransform>,

    /// Validity carried by indexed-away axes, including scalar views of masked windows.
    predicate: Option<KernelValue<'c, 't>>,

    /// Underlying allocation and canonical root type.
    buffer: Buffer<'c, 't>,

    /// Logical view shape.
    shape: Vec<usize>,

    /// Flat physical offset of the view origin.
    offset: KernelValue<'c, 't>,

    /// Physical strides, in elements.
    strides: Vec<usize>,

    /// Number of valid elements on each view axis.
    valid: Vec<KernelValue<'c, 't>>,
}

/// Canonical value members retain their distinct physical representations during construction.
#[derive(Clone)]
enum Lowered<'c, 't> {
    Array(Buffer<'c, 't>),
    Dimension(KernelValue<'c, 't>),
    Reference(Reference<'c, 't>),
    Token(CopyToken<'c, 't>),
    Tmem(Tmem<'c, 't>),
    TmemToken(TmemToken<'c, 't>),
}

impl<'c, 't> Lowered<'c, 't> {
    /// Extracts a verifier-established ordinary array input.
    fn array(&self) -> &Buffer<'c, 't> {
        let Self::Array(value) = self else { unreachable!() };
        value
    }

    /// Extracts a verifier-established dimension input.
    fn dimension(&self) -> KernelValue<'c, 't> {
        let Self::Dimension(value) = self else { unreachable!() };
        *value
    }

    /// Extracts a verifier-established reference input.
    fn reference(&self) -> &Reference<'c, 't> {
        let Self::Reference(value) = self else { unreachable!() };
        value
    }
}

/// Mutable construction state for one logical program's cooperative CUDA block.
struct Lowering<'c, 't> {
    /// Owner of every generated operation and type.
    context: &'c Context<'t>,

    /// Source location used for generated operations.
    location: UnknownLocationRef<'c, 't>,

    /// Current lane's flattened CUDA thread index.
    thread: KernelValue<'c, 't>,

    /// Declared participant count, represented as an index value.
    threads: KernelValue<'c, 't>,

    /// Local CTA rank in a two-block cluster; absent for ordinary single-CTA programs.
    cluster_rank: Option<KernelValue<'c, 't>>,

    /// Preallocated transport storage, keyed by the canonical source value identity.
    storage: HashMap<ValueId, Buffer<'c, 't>>,

    /// Native by-value TMA descriptor pointers, keyed by the canonical completion token.
    tma_descriptors: HashMap<ValueId, KernelValue<'c, 't>>,

    /// Native operand transports owned by their canonical matrix instruction.
    instruction_scratch: HashMap<InstructionId, Vec<KernelValue<'c, 't>>>,

    /// Maximum outstanding committed warpgroup groups.
    mma_stages: usize,

    /// Actual emitted copy groups and CTA rendezvous, absent only in isolated arithmetic construction tests.
    synchronization: Option<Vec<CtaSynchronization>>,

    /// Canonical source operation currently being lowered.
    current_instruction: Option<InstructionId>,

    /// Unique emitted barrier site within this CUDA block.
    next_barrier: usize,
}

impl<'c, 't> Lowering<'c, 't> {
    /// Emits an index constant after admission has checked native integer representability.
    fn index(&self, block: &mut DetachedBlock<'c, 't>, value: usize) -> Result<KernelValue<'c, 't>, Error> {
        append(
            block,
            arith::constant(self.context.integer_attribute(self.context.index_type(), value as i64), self.location)?,
        )
    }

    /// Loads one physical element. Callers must establish bounds before emitting this operation.
    fn load(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        buffer: &Buffer<'c, 't>,
        index: KernelValue<'c, 't>,
    ) -> Result<KernelValue<'c, 't>, Error> {
        append(
            block,
            memref::load(
                buffer.value,
                &[index],
                element_type(self.context, buffer.r#type.data_type())?,
                false,
                None,
                self.location,
            )?,
        )
    }

    /// Stores one physical element. Callers must establish unique lane ownership and bounds.
    fn store(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        buffer: &Buffer<'c, 't>,
        index: KernelValue<'c, 't>,
        value: KernelValue<'c, 't>,
    ) -> Result<(), Error> {
        if let Some(rank) = self.cluster_rank.filter(|_| !buffer.shared) {
            let zero = self.index(block, 0)?;
            let leader =
                append(block, arith::cmpi(rank, zero, arith::IntegerComparisonPredicate::Equal, self.location)?)?;
            let mut body = self.context.block(&[] as &[(TypeRef, UnknownLocationRef)]);
            body.append_operation(memref::store(value, buffer.value, &[index], false, None, self.location)?)?;
            body.append_operation(scf::r#yield(&[], self.location)?)?;
            block.append_operation(scf::r#if(leader, &[], body.try_into()?, None, self.location)?)?;
        } else {
            block.append_operation(memref::store(value, buffer.value, &[index], false, None, self.location)?)?;
        }
        Ok(())
    }

    /// Distributes independent flat elements across the declared participant set without changing result order.
    fn distributed<F>(&mut self, block: &mut DetachedBlock<'c, 't>, count: usize, function: F) -> Result<(), Error>
    where
        F: FnOnce(&mut Self, &mut DetachedBlock<'c, 't>, KernelValue<'c, 't>) -> Result<(), Error>,
    {
        let upper = self.index(block, count)?;
        let mut body = self.context.block(&[(self.context.index_type().as_ref(), self.location)]);
        let index = body.argument(0)?.as_ref();
        function(self, &mut body, index)?;
        body.append_operation(scf::r#yield(&[], self.location)?)?;
        block.append_operation(scf::r#for(
            self.thread,
            upper,
            self.threads,
            &[],
            false,
            body.try_into()?,
            self.location,
        )?)?;
        Ok(())
    }

    /// Publishes completed shared-memory writes before subsequent cross-thread reads.
    fn barrier(&mut self, block: &mut DetachedBlock<'c, 't>) -> Result<(), Error> {
        if let Some(plans) = &self.synchronization {
            let participants = plans[0].participants().get();
            for thread in 0..participants {
                self.record_synchronization(None, thread, SynchronizationEvent::Barrier { site: self.next_barrier })?;
            }
            self.next_barrier += 1;
        }
        block.append_operation(gpu::barrier(None, self.location)?)?;
        self.cluster_barrier(block)
    }

    /// Publishes completed CTA-local writes to the peer and keeps replicated program execution in lockstep.
    fn cluster_barrier(&mut self, block: &mut DetachedBlock<'c, 't>) -> Result<(), Error> {
        if self.cluster_rank.is_none() {
            return Ok(());
        }
        if let Some(plans) = &self.synchronization {
            let participants = plans[0].participants().get();
            for thread in 0..participants {
                self.record_synchronization(
                    None,
                    thread,
                    SynchronizationEvent::ClusterBarrier { site: self.next_barrier },
                )?;
            }
            self.next_barrier += 1;
        }
        block.append_operation(nvvm::cluster_arrive(&[], &[], &[], false, self.location)?)?;
        block.append_operation(nvvm::cluster_wait(&[], &[], &[], false, self.location)?)?;
        Ok(())
    }

    /// Copies the peer's completed rows into this CTA's replicated result, with cluster rendezvous around all reads.
    /// The native collective load has already written exactly this CTA's `rows_per_cta` contiguous rows.
    fn cluster_gather_rows(
        &mut self,
        block: &mut DetachedBlock<'c, 't>,
        buffer: &Buffer<'c, 't>,
        rows_per_cta: usize,
    ) -> Result<(), Error> {
        let dimensions = shape(&buffer.r#type)?;
        let participants = self.synchronization.as_ref().unwrap()[0].participants().get();
        if !buffer.shared
            || dimensions.len() != 2
            || dimensions[0] != 2 * rows_per_cta
            || rows_per_cta != participants as usize
            || self.cluster_rank.is_none()
        {
            return Err(Error::Invalid {
                message: "cluster row replication requires one shared row per declared thread in each cta".to_owned(),
            });
        }
        self.barrier(block)?;
        let columns = dimensions[1];
        for cta in 0..2 {
            for thread in 0..participants {
                let row = (1 - cta) * rows_per_cta + thread as usize;
                let transform = ArrayReferenceTransform::Slice {
                    axes: vec![ArraySliceAxis::new(row, 1, 1), ArraySliceAxis::new(0, columns, 1)],
                };
                self.record_synchronization(
                    Some(cta),
                    thread,
                    SynchronizationEvent::DistributedCopy {
                        source_block: (1 - cta) as u32,
                        source: (buffer.owner, transform.clone()),
                        destination: (buffer.owner, transform),
                    },
                )?;
            }
        }
        let rank = self.cluster_rank.unwrap();
        let one = self.index(block, 1)?;
        let peer = append(block, arith::subi(one, rank, self.location)?)?;
        let peer_i32 =
            append(block, arith::index_castui(peer, self.context.signless_integer_type(32), self.location)?)?;
        let row_count = self.index(block, rows_per_cta)?;
        let row_start = append(block, arith::muli(peer, row_count, self.location)?)?;
        let row_width = self.index(block, columns)?;
        let bytes = ryft_core::ArrayAddressing::new(buffer.r#type.clone())?.element_byte_width();
        let element_bytes = self.index(block, bytes)?;
        let base = append(block, memref::extract_aligned_pointer_as_index(buffer.value, self.location)?)?;
        self.distributed(block, rows_per_cta, |lowering, body, row| {
            let row = append(body, arith::addi(row_start, row, lowering.location)?)?;
            let row_offset = append(body, arith::muli(row, row_width, lowering.location)?)?;
            let zero = lowering.index(body, 0)?;
            let one = lowering.index(body, 1)?;
            let mut copy = lowering.context.block(&[(lowering.context.index_type().as_ref(), lowering.location)]);
            let column = copy.argument(0)?.as_ref();
            let index = append(&mut copy, arith::addi(row_offset, column, lowering.location)?)?;
            let offset = append(&mut copy, arith::muli(index, element_bytes, lowering.location)?)?;
            let address = append(&mut copy, arith::addi(base, offset, lowering.location)?)?;
            let address = append(
                &mut copy,
                arith::index_castui(address, lowering.context.signless_integer_type(64), lowering.location)?,
            )?;
            let address =
                append(&mut copy, llvm::inttoptr(address, lowering.context.llvm_pointer_type(3)?, lowering.location)?)?;
            let address = append(
                &mut copy,
                nvvm::mapa(
                    &[address, peer_i32],
                    &[lowering.context.llvm_pointer_type(7)?.as_ref()],
                    &[],
                    false,
                    lowering.location,
                )?,
            )?;
            let value = append(
                &mut copy,
                llvm::load(
                    address,
                    element_type(lowering.context, buffer.r#type.data_type())?,
                    Some(bytes as i64),
                    false,
                    lowering.location,
                )?,
            )?;
            lowering.store(&mut copy, buffer, index, value)?;
            copy.append_operation(scf::r#yield(&[], lowering.location)?)?;
            body.append_operation(scf::r#for(zero, row_width, one, &[], false, copy.try_into()?, lowering.location)?)?;
            Ok(())
        })?;
        self.barrier(block)
    }

    /// Records the actual emitted instruction in each participating CTA, or only in one explicitly elected CTA.
    fn record_synchronization(
        &mut self,
        block: Option<usize>,
        thread: u32,
        event: SynchronizationEvent,
    ) -> Result<(), Error> {
        if let (Some(plans), Some(instruction)) = (&mut self.synchronization, self.current_instruction) {
            for (index, plan) in plans.iter_mut().enumerate() {
                if block.is_none_or(|block| block == index) {
                    plan.record(thread, instruction, event.clone())
                        .map_err(|error| Error::Synchronization { message: error.to_string() })?;
                }
            }
        }
        Ok(())
    }
}

/// Appends a single-result operation while preserving context-owned value lifetimes.
fn append<'c, 't, O: DetachedOp<'c, 'c, 't>>(
    block: &mut DetachedBlock<'c, 't>,
    operation: O,
) -> Result<KernelValue<'c, 't>, Error> {
    Ok(block.append_operation(operation)?.result(0)?.as_ref())
}

/// Returns a fully specialized logical shape without inventing dynamic extents.
fn shape(r#type: &ArrayType) -> Result<Vec<usize>, Error> {
    r#type.static_shape().map(|shape| shape.dimensions().to_vec()).ok_or_else(|| Error::Unsupported {
        operation: "array type",
        reason: "dynamic shapes must be specialized before GPU compilation".to_owned(),
    })
}

/// Returns whether `axes` select all of a reference with the logical `shape`. Such identity slices are the only
/// transforms that native asynchronous copies accept, because they preserve the whole-root proof that those copies
/// require.
fn is_whole_slice(axes: &[ArraySliceAxis], shape: &[usize]) -> bool {
    axes.len() == shape.len()
        && axes
            .iter()
            .zip(shape)
            .all(|(axis, extent)| axis.start() == 0 && axis.size() == *extent && axis.stride() == 1)
}

/// Checks source support before any native module is constructed or an artifact cache is consulted.
pub(super) fn validate<Extension: KernelExtension + Into<GpuOperation>>(
    kernel: &VerifiedKernel<'_, Extension>,
    target: &Target,
    options: &Options,
    schedule: &KernelSchedule,
) -> Result<validation::Plan, Error> {
    let plan = validation::Plan::new(kernel, target, options, schedule)?;
    memory::validate_async_copies(
        kernel
            .definition()
            .body()
            .entry_region_ref()
            .instructions()
            .iter()
            .map(|instruction| instruction.operation()),
    )?;
    for (instruction_id, instruction) in kernel.definition().body().entry_region_ref().instructions_in_closure() {
        let supported = match instruction.operation() {
            KernelOperation::Portable(ArrayIrOperation::Assert(_))
            | KernelOperation::Portable(ArrayIrOperation::Array(ArrayOperation::Assert(_))) => {
                return Err(Error::Unsupported {
                    operation: "assert",
                    reason: "runtime assertions require native failure propagation".to_owned(),
                });
            }
            KernelOperation::Portable(ArrayIrOperation::Array(_)) => true,
            KernelOperation::Portable(ArrayIrOperation::Dimension(operation)) => {
                operation.effects().is_pure() && !matches!(operation, ryft_core::DimensionOperation::Pow(_))
            }
            KernelOperation::Portable(
                ArrayIrOperation::DimensionFromScalar(_)
                | ArrayIrOperation::DimensionToScalar(_)
                | ArrayIrOperation::ReferenceRead(_)
                | ArrayIrOperation::ReferenceFreeze(_)
                | ArrayIrOperation::ReferenceWrite(_)
                | ArrayIrOperation::ReferenceSwap(_),
            ) => true,
            KernelOperation::Portable(ArrayIrOperation::Condition(_)) => instruction.outputs().iter().all(|id| {
                matches!(
                    kernel.definition().body().regions().get(instruction_id.region()).unwrap().atoms()[id.index()]
                        .r#type()
                        .as_ref(),
                    ryft_core::ArrayIrType::Array(_)
                )
            }),
            KernelOperation::Portable(ArrayIrOperation::DimensionSize(operation)) => {
                operation.input_dimension().value().is_some()
            }
            KernelOperation::Portable(ArrayIrOperation::While(operation)) => {
                let body = kernel.definition().body().region(instruction.regions()[1])?;
                let condition = kernel.definition().body().region(instruction.regions()[0])?;
                let scalar_predicate = matches!(condition.output_types().as_slice(),
                    [ryft_core::ArrayIrType::Array(r#type)] if r#type.rank() == 0);
                let preserved_carries =
                    body.input_ids().iter().zip(body.output_ids()).all(|(input, output)| {
                        match body.atoms()[input.index()].r#type().as_ref() {
                            ryft_core::ArrayIrType::Array(_) => true,
                            ryft_core::ArrayIrType::Reference(_) => input == output,
                            ryft_core::ArrayIrType::Dimension(_) => input == output,
                        }
                    });
                operation.iteration_bound().is_some_and(|bound| bound <= i64::MAX as usize)
                    && scalar_predicate
                    && preserved_carries
            }
            KernelOperation::Scratch(_)
            | KernelOperation::TileLoad(_)
            | KernelOperation::MaskedLoad(_)
            | KernelOperation::MaskedStore(_)
            | KernelOperation::MaskedSwap(_) => true,
            KernelOperation::AsyncCopy(operation) => {
                // Native copies address whole roots. Reject partial transforms (views that select less than the whole
                // root) here, before any artifact lookup, instead of leaving them to fail during lowering; full
                // identity slices keep the whole root selected.
                let atoms = kernel.definition().body().regions().get(instruction_id.region()).unwrap().atoms();
                for (input, transforms) in [
                    (instruction.inputs()[0], operation.source_transforms()),
                    (instruction.inputs()[1], operation.destination_transforms()),
                ] {
                    let shape = match atoms[input.index()].r#type().as_ref() {
                        ArrayIrType::Reference(reference) => reference.referent().static_shape(),
                        _ => None,
                    };
                    let whole = transforms.iter().all(|transform| {
                        matches!(transform, ArrayReferenceTransform::Slice { axes }
                            if shape.as_ref().is_some_and(|shape| is_whole_slice(axes, shape.dimensions())))
                    });
                    if !whole {
                        return Err(Error::Unsupported {
                            operation: operation.name(),
                            reason: "native async copies require transforms that select whole references".to_owned(),
                        });
                    }
                }
                let entry = kernel.definition().body().entry_region_ref();
                let source = entry.input_ids().iter().position(|id| id == &instruction.inputs()[0]);
                let destination = entry
                    .instructions()
                    .iter()
                    .find(|candidate| candidate.outputs().contains(&instruction.inputs()[1]));
                instruction_id.region() == entry.id()
                    && source.is_some_and(|index| {
                        kernel.definition().operation().parameters().get(index).is_some_and(|parameter| {
                            parameter.mapping().tiling_axes().is_some_and(|axes| axes.iter().all(Option::is_none))
                                && parameter
                                    .r#type()
                                    .static_shape()
                                    .is_some_and(|shape| shape.dimensions() == parameter.mapping().block_shape())
                                && parameter.r#type().data_type() != DataType::Boolean
                        })
                    })
                    && destination
                        .is_some_and(|instruction| matches!(instruction.operation(), KernelOperation::Scratch(_)))
            }
            KernelOperation::Wait(_) => true,
            KernelOperation::Extension(extension) => {
                matches!(
                    extension.clone().into(),
                    GpuOperation::Wgmma
                        | GpuOperation::Nvfp4 { .. }
                        | GpuOperation::Nvfp4Sparse { .. }
                        | GpuOperation::Tmem(_)
                )
            }
            _ => false,
        };
        if !supported {
            return Err(Error::Unsupported {
                operation: instruction.operation().name(),
                reason: "operation has no baseline GPU lowering".to_owned(),
            });
        }
    }
    Ok(plan)
}

/// Constructs the checked source module and its physical argument mapping.
pub(super) fn module<'c, 't, Extension: KernelExtension + Into<GpuOperation>>(
    context: &'c Context<'t>,
    kernel: &VerifiedKernel<'_, Extension>,
    target: &Target,
    options: &Options,
    schedule: &KernelSchedule,
) -> Result<(Module<'c, 't>, Vec<ArrayType>, Vec<usize>, usize), Error> {
    let plan = validate(kernel, target, options, schedule)?;
    let call = kernel.definition().operation();
    let parallel_count = call
        .grid()
        .dimensions()
        .iter()
        .zip(&plan.grid_extents)
        .filter(|(dimension, _)| dimension.execution() == GridExecution::Parallel)
        .map(|(_, extent)| *extent)
        .product::<usize>();
    let empty = plan.grid_extents.contains(&0);
    let shared_types = plan
        .storage
        .iter()
        .map(|(_, r#type)| r#type.clone())
        .chain(plan.instruction_scratch.iter().flat_map(|(_, types)| types.iter().cloned()))
        .collect::<Vec<_>>();
    let shared_alignments = plan
        .storage
        .iter()
        .map(|(owner, _)| if plan.tma_copies.iter().any(|copy| copy.destination == *owner) { 128 } else { 16 })
        .chain(std::iter::repeat_n(256, plan.instruction_scratch.iter().map(|(_, types)| types.len()).sum()))
        .collect::<Vec<_>>();
    let has_assertions =
        kernel.definition().body().effects().classes().contains(ryft_core::EffectClass::OrderedAssertion);
    let input_count = call.input_types().len();
    let arguments = plan
        .argument_types
        .iter()
        .enumerate()
        .map(|(index, r#type)| (index + usize::from(has_assertions && index >= input_count), r#type.clone()))
        .collect::<Vec<_>>();
    let module = module::build(
        context,
        target,
        "ryft_kernel",
        &arguments,
        &shared_types,
        &shared_alignments,
        &plan.tma_copies.iter().map(|copy| plan.parameter_slots[copy.source_parameter]).collect::<Vec<_>>(),
        [parallel_count.max(1) * target.blocks_per_cluster() as usize, 1, 1],
        |block, globals, shared, descriptors| {
            if empty {
                return Ok(());
            }
            let location = context.unknown_location();
            let mut synchronization_storage = plan.storage.iter().cloned().collect::<HashMap<_, _>>();
            let region = kernel.definition().body().entry_region_ref();
            for (index, parameter) in call.parameters().iter().enumerate() {
                synchronization_storage
                    .insert(ValueId::new(region.id(), region.input_ids()[index]), parameter.r#type().into_owned());
            }
            for instruction in region.instructions() {
                if matches!(instruction.operation(), KernelOperation::Extension(extension)
                if matches!(extension.clone().into(), GpuOperation::Tmem(
                    TmemOperation::Allocate { .. } | TmemOperation::AllocateScales { .. }
                ))) {
                    let owner = instruction.outputs()[0];
                    let r#type = region.atoms()[owner.index()].r#type();
                    let ArrayIrType::Reference(reference) = r#type.as_ref() else { unreachable!() };
                    // The native address slot shares the source owner; simulation tracks its logical TMEM contents.
                    synchronization_storage.insert(ValueId::new(region.id(), owner), reference.referent().clone());
                }
            }
            let synchronization =
                CtaSynchronization::new(NonZeroU32::new(target.threads_per_block()).unwrap(), synchronization_storage)
                    .map_err(|error| Error::Synchronization { message: error.to_string() })?;
            let mut lowering = Lowering {
                context,
                location,
                thread: block.argument(3)?.as_ref(),
                threads: block.argument(9)?.as_ref(),
                cluster_rank: if target.blocks_per_cluster() == 2 {
                    let size = append(
                        block,
                        arith::constant(
                            context.integer_attribute(context.index_type(), i64::from(target.blocks_per_cluster())),
                            location,
                        )?,
                    )?;
                    Some(append(block, arith::remui(block.argument(0)?.as_ref(), size, location)?)?)
                } else {
                    None
                },
                tma_descriptors: plan
                    .tma_copies
                    .iter()
                    .zip(descriptors)
                    .map(|(copy, descriptor)| (copy.token, *descriptor))
                    .collect(),
                instruction_scratch: {
                    let mut offset = plan.storage.len();
                    plan.instruction_scratch
                        .iter()
                        .map(|(instruction, types)| {
                            let values = shared[offset..offset + types.len()].to_vec();
                            offset += types.len();
                            (*instruction, values)
                        })
                        .collect()
                },
                mma_stages: schedule.pipeline_stages().map(|stages| stages.get()).unwrap_or(1),
                synchronization: Some(vec![synchronization; target.blocks_per_cluster() as usize]),
                current_instruction: None,
                next_barrier: 0,
                storage: plan
                    .storage
                    .iter()
                    .zip(shared)
                    .map(|((id, r#type), value)| {
                        (*id, Buffer { owner: *id, shared: true, value: *value, r#type: r#type.clone() })
                    })
                    .collect(),
            };
            let mut remaining = block.argument(if target.blocks_per_cluster() == 2 { 12 } else { 0 })?.as_ref();
            let mut coordinates = vec![lowering.index(block, 0)?; plan.grid_extents.len()];
            for axis in (0..coordinates.len()).rev() {
                if call.grid().dimensions()[axis].execution() == GridExecution::Parallel {
                    let extent = lowering.index(block, plan.grid_extents[axis])?;
                    coordinates[axis] = append(block, arith::remui(remaining, extent, location)?)?;
                    remaining = append(block, arith::divui(remaining, extent, location)?)?;
                }
            }
            lowering.grid(block, kernel, &plan, globals, &mut coordinates, 0)?;
            if plan.uses_tmem {
                lowering.tmem_relinquish(block)?;
            }
            let synchronization = lowering.synchronization.as_ref().unwrap();
            if target.blocks_per_cluster() == 2 {
                synchronization[0]
                    .simulate_cluster(&synchronization[1])
                    .map_err(|error| Error::Synchronization { message: error.to_string() })?;
            } else {
                synchronization[0]
                    .simulate()
                    .map_err(|error| Error::Synchronization { message: error.to_string() })?;
            }
            Ok(())
        },
    )?;
    let parameter_slots = plan.parameter_slots.iter().map(|slot| arguments[*slot].0).collect();
    Ok((module, plan.argument_types, parameter_slots, plan.shared_memory_bytes))
}

impl<'c, 't> Lowering<'c, 't> {
    /// Nests sequential grid axes inside each parallel CTA while preserving the declared traversal order.
    fn grid<Extension: KernelExtension + Into<GpuOperation>>(
        &mut self,
        block: &mut DetachedBlock<'c, 't>,
        kernel: &VerifiedKernel<'_, Extension>,
        plan: &validation::Plan,
        globals: &[KernelValue<'c, 't>],
        coordinates: &mut [KernelValue<'c, 't>],
        axis: usize,
    ) -> Result<(), Error> {
        let call = kernel.definition().operation();
        if axis < coordinates.len() {
            if call.grid().dimensions()[axis].execution() == GridExecution::Parallel {
                return self.grid(block, kernel, plan, globals, coordinates, axis + 1);
            }
            let lower = self.index(block, 0)?;
            let upper = self.index(block, plan.grid_extents[axis])?;
            let step = self.index(block, 1)?;
            let mut body = self.context.block(&[(self.context.index_type().as_ref(), self.location)]);
            coordinates[axis] = body.argument(0)?.as_ref();
            self.grid(&mut body, kernel, plan, globals, coordinates, axis + 1)?;
            body.append_operation(scf::r#yield(&[], self.location)?)?;
            block.append_operation(scf::r#for(lower, upper, step, &[], false, body.try_into()?, self.location)?)?;
            return Ok(());
        }
        let mut inputs = Vec::new();
        for (parameter_index, (parameter, slot)) in call.parameters().iter().zip(&plan.parameter_slots).enumerate() {
            let starts = self.mapping(block, parameter.mapping(), coordinates)?;
            let root = self.reference(
                block,
                Buffer {
                    owner: ValueId::new(
                        kernel.definition().body().entry_region_ref().id(),
                        kernel.definition().body().input_ids()[parameter_index],
                    ),
                    shared: false,
                    value: globals[*slot],
                    r#type: parameter.r#type().into_owned(),
                },
            )?;
            let mut window = self.window(block, &root, &starts, parameter.mapping().block_shape())?;
            if parameter.mapping().block_shape() == root.shape.as_slice()
                && parameter.mapping().tiling_axes().is_some_and(|axes| axes.iter().all(Option::is_none))
            {
                window.static_transform = root.static_transform.clone();
            }
            inputs.push(Lowered::Reference(window));
        }
        inputs.extend(coordinates.iter().copied().map(Lowered::Dimension));
        self.region(block, kernel.definition().body().entry_region_ref(), &inputs)?;
        if self.cluster_rank.is_some() {
            self.barrier(block)?;
        }
        Ok(())
    }

    /// Creates a whole-allocation view with canonical row-major strides.
    fn reference(&self, block: &mut DetachedBlock<'c, 't>, buffer: Buffer<'c, 't>) -> Result<Reference<'c, 't>, Error> {
        let shape = shape(&buffer.r#type)?;
        let strides = (0..shape.len()).map(|axis| shape[axis + 1..].iter().product()).collect();
        let valid = shape.iter().map(|extent| self.index(block, *extent)).collect::<Result<Vec<_>, _>>()?;
        Ok(Reference {
            static_transform: Some(ArrayReferenceTransform::Slice {
                axes: shape.iter().map(|extent| ArraySliceAxis::new(0, *extent, 1)).collect(),
            }),
            buffer,
            shape,
            offset: self.index(block, 0)?,
            strides,
            valid,
            predicate: None,
        })
    }

    /// Composes a rank-preserving window and clips validity without ever forming a speculative memory access.
    fn window(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        source: &Reference<'c, 't>,
        starts: &[KernelValue<'c, 't>],
        shape: &[usize],
    ) -> Result<Reference<'c, 't>, Error> {
        let mut result = source.clone();
        result.static_transform = None;
        result.shape = shape.to_vec();
        for axis in 0..shape.len() {
            let stride = self.index(block, source.strides[axis])?;
            let offset = append(block, arith::muli(starts[axis], stride, self.location)?)?;
            result.offset = append(block, arith::addi(result.offset, offset, self.location)?)?;
            let clipped = append(block, arith::minui(starts[axis], source.valid[axis], self.location)?)?;
            let remaining = append(block, arith::subi(source.valid[axis], clipped, self.location)?)?;
            let extent = self.index(block, shape[axis])?;
            result.valid[axis] = append(block, arith::minui(remaining, extent, self.location)?)?;
        }
        Ok(result)
    }

    /// Computes a flat physical address and its validity predicate for one logical view element.
    fn address(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        reference: &Reference<'c, 't>,
        index: KernelValue<'c, 't>,
    ) -> Result<(KernelValue<'c, 't>, KernelValue<'c, 't>), Error> {
        let mut remaining = index;
        let mut offset = reference.offset;
        let mut valid = append(
            block,
            arith::constant(self.context.integer_attribute(self.context.signless_integer_type(1), 1), self.location)?,
        )?;
        if let Some(predicate) = reference.predicate {
            valid = append(block, arith::andi(valid, predicate, self.location)?)?;
        }
        for axis in (0..reference.shape.len()).rev() {
            let extent = self.index(block, reference.shape[axis].max(1))?;
            let coordinate = append(block, arith::remui(remaining, extent, self.location)?)?;
            remaining = append(block, arith::divui(remaining, extent, self.location)?)?;
            let in_bounds = append(
                block,
                arith::cmpi(
                    coordinate,
                    reference.valid[axis],
                    arith::IntegerComparisonPredicate::UnsignedLessThan,
                    self.location,
                )?,
            )?;
            valid = append(block, arith::andi(valid, in_bounds, self.location)?)?;
            let stride = self.index(block, reference.strides[axis])?;
            let contribution = append(block, arith::muli(coordinate, stride, self.location)?)?;
            offset = append(block, arith::addi(offset, contribution, self.location)?)?;
        }
        Ok((offset, valid))
    }

    /// Reads valid lanes and selects explicit padding in a control-flow branch for all invalid lanes.
    fn read(
        &mut self,
        block: &mut DetachedBlock<'c, 't>,
        reference: &Reference<'c, 't>,
        output: &Buffer<'c, 't>,
        other: Option<&Buffer<'c, 't>>,
        mask: Option<&Buffer<'c, 't>>,
    ) -> Result<(), Error> {
        self.distributed(block, output.r#type.element_count()?.unwrap(), |lowering, body, index| {
            let (offset, mut valid) = lowering.address(body, reference, index)?;
            if let Some(mask) = mask {
                let selected = lowering.load(body, mask, index)?;
                valid = append(body, arith::andi(valid, selected, lowering.location)?)?;
            }
            let mut selected = lowering.context.block::<TypeRef<'c, 't>, UnknownLocationRef<'c, 't>>(&[]);
            let value = lowering.load(&mut selected, &reference.buffer, offset)?;
            selected.append_operation(scf::r#yield(&[value], lowering.location)?)?;
            let mut unselected = lowering.context.block::<TypeRef<'c, 't>, UnknownLocationRef<'c, 't>>(&[]);
            let fallback = if let Some(other) = other {
                let other_index = if other.r#type.rank() == 0 { lowering.index(&mut unselected, 0)? } else { index };
                lowering.load(&mut unselected, other, other_index)?
            } else {
                lowering.literal(&mut unselected, output.r#type.data_type(), 0)?
            };
            unselected.append_operation(scf::r#yield(&[fallback], lowering.location)?)?;
            let value = append(
                body,
                scf::r#if(
                    valid,
                    &[element_type(lowering.context, output.r#type.data_type())?],
                    selected.try_into()?,
                    Some(unselected.try_into()?),
                    lowering.location,
                )?,
            )?;
            lowering.store(body, output, index, value)
        })?;
        self.barrier(block)
    }

    /// Publishes disjoint valid lanes of a reference write and makes shared destinations visible to the CTA.
    fn write(
        &mut self,
        block: &mut DetachedBlock<'c, 't>,
        reference: &Reference<'c, 't>,
        input: &Buffer<'c, 't>,
        mask: Option<&Buffer<'c, 't>>,
    ) -> Result<(), Error> {
        self.distributed(block, input.r#type.element_count()?.unwrap(), |lowering, body, index| {
            let (offset, mut valid) = lowering.address(body, reference, index)?;
            if let Some(mask) = mask {
                let selected = lowering.load(body, mask, index)?;
                valid = append(body, arith::andi(valid, selected, lowering.location)?)?;
            }
            let mut selected = lowering.context.block::<TypeRef<'c, 't>, UnknownLocationRef<'c, 't>>(&[]);
            let value = lowering.load(&mut selected, input, index)?;
            lowering.store(&mut selected, &reference.buffer, offset, value)?;
            selected.append_operation(scf::r#yield(&[], lowering.location)?)?;
            body.append_operation(scf::r#if(valid, &[], selected.try_into()?, None, lowering.location)?)?;
            Ok(())
        })?;
        self.barrier(block)
    }

    /// Copies a complete immutable value into an independently allocated destination before publishing it.
    fn copy(
        &mut self,
        block: &mut DetachedBlock<'c, 't>,
        source: &Buffer<'c, 't>,
        destination: &Buffer<'c, 't>,
    ) -> Result<(), Error> {
        if source.value == destination.value {
            return Ok(());
        }
        self.distributed(block, destination.r#type.element_count()?.unwrap(), |lowering, body, index| {
            let value = lowering.load(body, source, index)?;
            lowering.store(body, destination, index, value)
        })?;
        self.barrier(block)
    }

    /// Replays one canonical region, preserving its local identities and ordered reference effects.
    fn region<Extension: KernelExtension + Into<GpuOperation>>(
        &mut self,
        block: &mut DetachedBlock<'c, 't>,
        region: RegionRef<'_, ArrayIrValue<Array>, KernelOperation<Extension>>,
        inputs: &[Lowered<'c, 't>],
    ) -> Result<Vec<Lowered<'c, 't>>, Error> {
        let mut values = vec![None; region.atoms().len()];
        for (id, input) in region.input_ids().iter().zip(inputs) {
            values[id.index()] = Some(input.clone());
        }
        for (index, atom) in region.atoms().iter().enumerate() {
            if let Atom::Constant(value) = atom {
                values[index] = Some(match value {
                    ArrayIrValue::Dimension(value) => Lowered::Dimension(self.index(block, value.extent())?),
                    ArrayIrValue::Array(value) => {
                        let buffer = self.storage[&ValueId::new(region.id(), AtomId::new(index))].clone();
                        self.array(
                            block,
                            &ArrayOperation::Constant(ConstantOperation::new(value.clone())),
                            &[],
                            &buffer,
                        )?;
                        Lowered::Array(buffer)
                    }
                    ArrayIrValue::Reference(_) => {
                        return Err(Error::Unsupported {
                            operation: "constant",
                            reason: "captured references are not kernel arguments".to_owned(),
                        });
                    }
                });
            }
        }
        for (instruction_index, instruction) in region.instructions().iter().enumerate() {
            self.current_instruction = Some(InstructionId::new(region.id(), instruction_index));
            let mut inputs =
                instruction.inputs().iter().map(|id| values[id.index()].clone().unwrap()).collect::<Vec<_>>();
            // Resolve each access path before lowering the memory operation. Dynamic bindings are trailing inputs;
            // the operation itself then receives its original base inputs with the viewed reference substituted.
            let operation = instruction.operation();
            let mut has_reference_access = false;
            for (input_index, descriptor) in
                validated_reference_access_descriptors(operation, instruction.inputs().len())?.iter().enumerate()
            {
                if let Some(descriptor) = descriptor {
                    has_reference_access = true;
                    // Empty paths also describe references with specialized storage, such as tensor memory.
                    // Preserve their native representation instead of treating them as ordinary memory buffers.
                    if descriptor.transforms().is_empty() {
                        continue;
                    }
                    let mut reference = inputs[input_index].reference().clone();
                    let mut binding_index = descriptor.bindings().start;
                    for transform in descriptor.transforms() {
                        let count = transform.binding_count();
                        reference = self.apply_transform(
                            block,
                            &reference,
                            transform,
                            &inputs[binding_index..binding_index + count],
                        )?;
                        binding_index += count;
                    }
                    inputs[input_index] = Lowered::Reference(reference);
                }
            }
            if has_reference_access {
                inputs.truncate(operation.base_input_count());
            }
            let output_ids =
                instruction.outputs().iter().map(|atom| ValueId::new(region.id(), *atom)).collect::<Vec<_>>();
            let output_buffers = output_ids.iter().map(|id| self.storage.get(id).cloned()).collect::<Vec<_>>();
            let output = || output_buffers[0].as_ref().unwrap();
            let results = match instruction.operation() {
                KernelOperation::Portable(ArrayIrOperation::Assert(_))
                | KernelOperation::Portable(ArrayIrOperation::Array(ArrayOperation::Assert(_))) => {
                    return Err(Error::Unsupported {
                        operation: "assert",
                        reason: "runtime assertions require native failure propagation".to_owned(),
                    });
                }
                KernelOperation::Portable(ArrayIrOperation::Array(operation)) => {
                    let arrays = inputs.iter().map(|input| input.array().clone()).collect::<Vec<_>>();
                    if let Some(scratch) = self.instruction_scratch.get(&self.current_instruction.unwrap()).cloned() {
                        let scales = if let ArrayOperation::ScaledDot(operation) = operation {
                            [
                                operation.has_lhs_scale().then(|| &arrays[2]),
                                operation.has_rhs_scale().then(|| &arrays[2 + usize::from(operation.has_lhs_scale())]),
                            ]
                        } else {
                            [None, None]
                        };
                        self.wgmma(block, &arrays[..2], output(), &[scratch[0], scratch[1]], self.mma_stages, scales)?;
                    } else {
                        self.array(block, operation, &arrays, output())?;
                    }
                    vec![Lowered::Array(output().clone())]
                }
                KernelOperation::Extension(extension) => match extension.clone().into() {
                    GpuOperation::Tmem(operation) => match operation {
                        TmemOperation::Allocate { columns, .. } => vec![Lowered::Tmem(self.tmem_allocate(
                            block,
                            output_ids[0],
                            usize::from(columns),
                            output().value,
                            None,
                        )?)],
                        TmemOperation::AllocateScales { rows, blocks, data_type } => {
                            let scale_signs = if data_type == DataType::F8E4M3FN {
                                Some(Buffer {
                                    owner: output_ids[0],
                                    shared: true,
                                    value: self.instruction_scratch[&self.current_instruction.unwrap()][0],
                                    r#type: ArrayType::new_static(
                                        DataType::U8,
                                        [usize::from(rows), usize::from(blocks)],
                                    ),
                                })
                            } else {
                                None
                            };
                            vec![Lowered::Tmem(self.tmem_allocate(
                                block,
                                output_ids[0],
                                tmem::scale_columns(usize::from(rows), usize::from(blocks))?,
                                output().value,
                                scale_signs,
                            )?)]
                        }
                        TmemOperation::CopyScales => {
                            let Lowered::Tmem(destination) = &inputs[1] else { unreachable!() };
                            let scratch = self.instruction_scratch[&self.current_instruction.unwrap()].clone();
                            vec![Lowered::TmemToken(self.tmem_scale_copy(
                                block,
                                inputs[0].array(),
                                destination,
                                output_ids[0],
                                output().value,
                                scratch[0],
                            )?)]
                        }
                        TmemOperation::Mma { accumulate }
                        | TmemOperation::MmaBlockScaled { accumulate }
                        | TmemOperation::MmaNvfp4 { accumulate } => {
                            let scaled = !matches!(operation, TmemOperation::Mma { .. });
                            let Lowered::Tmem(destination) = &inputs[if scaled { 4 } else { 2 }] else {
                                unreachable!()
                            };
                            let scales = if scaled {
                                let (Lowered::Tmem(left), Lowered::Tmem(right)) = (&inputs[2], &inputs[3]) else {
                                    unreachable!()
                                };
                                Some([left.clone(), right.clone()])
                            } else {
                                None
                            };
                            let scratch = self.instruction_scratch[&self.current_instruction.unwrap()].clone();
                            vec![Lowered::TmemToken(self.tmem_mma(
                                block,
                                &[inputs[0].array().clone(), inputs[1].array().clone()],
                                destination,
                                output_ids[0],
                                output().value,
                                &[scratch[0], scratch[1]],
                                accumulate,
                                scales.as_ref(),
                            )?)]
                        }
                        TmemOperation::Commit => {
                            // Commit observes the same token again at Wait; update its canonical value slot.
                            let Some(Lowered::TmemToken(token)) = &mut values[instruction.inputs()[0].index()] else {
                                unreachable!()
                            };
                            self.tmem_commit(block, token)?;
                            Vec::new()
                        }
                        TmemOperation::Wait => {
                            let Lowered::TmemToken(token) = &inputs[0] else { unreachable!() };
                            self.tmem_wait(block, token.clone())?;
                            Vec::new()
                        }
                        TmemOperation::Load => {
                            let Lowered::Tmem(source) = &inputs[0] else { unreachable!() };
                            self.tmem_load(block, source, output())?;
                            vec![Lowered::Array(output().clone())]
                        }
                        TmemOperation::Release => {
                            let Lowered::Tmem(source) = &inputs[0] else { unreachable!() };
                            self.tmem_release(block, source.clone())?;
                            Vec::new()
                        }
                    },
                    GpuOperation::Wgmma => {
                        let arrays = inputs.iter().map(|input| input.array().clone()).collect::<Vec<_>>();
                        let scratch = self.instruction_scratch[&self.current_instruction.unwrap()].clone();
                        self.wgmma(block, &arrays, output(), &[scratch[0], scratch[1]], self.mma_stages, [None, None])?;
                        vec![Lowered::Array(output().clone())]
                    }
                    GpuOperation::Nvfp4 { tensor_scale } => {
                        let arrays = inputs.iter().map(|input| input.array().clone()).collect::<Vec<_>>();
                        self.nvfp4(block, &arrays, output(), tensor_scale)?;
                        vec![Lowered::Array(output().clone())]
                    }
                    GpuOperation::Nvfp4Sparse { tensor_scale } => {
                        let arrays = inputs.iter().map(|input| input.array().clone()).collect::<Vec<_>>();
                        self.nvfp4_sparse(block, &arrays, output(), tensor_scale)?;
                        vec![Lowered::Array(output().clone())]
                    }
                },
                KernelOperation::Portable(ArrayIrOperation::Dimension(operation)) => self
                    .dimension(block, operation, &inputs.iter().map(Lowered::dimension).collect::<Vec<_>>())?
                    .into_iter()
                    .map(Lowered::Dimension)
                    .collect(),
                KernelOperation::Portable(ArrayIrOperation::DimensionSize(operation)) => {
                    vec![Lowered::Dimension(self.index(block, operation.input_dimension().value().unwrap())?)]
                }
                KernelOperation::Portable(ArrayIrOperation::DimensionFromScalar(operation)) => {
                    let zero = self.index(block, 0)?;
                    let mut value = self.load(block, inputs[0].array(), zero)?;
                    let data_type = inputs[0].array().r#type.data_type();
                    let signed = matches!(data_type, DataType::I32 | DataType::I64);
                    if data_type == DataType::I32 {
                        value =
                            append(block, arith::extsi(value, self.context.signless_integer_type(64), self.location)?)?;
                    } else if data_type == DataType::U32 {
                        value =
                            append(block, arith::extui(value, self.context.signless_integer_type(64), self.location)?)?;
                    }
                    let bounds = operation.output_type().bounds();
                    let lower = self.literal(block, DataType::I64, bounds.lower() as u64)?;
                    let upper = self.literal(
                        block,
                        DataType::I64,
                        bounds.upper().map_or(i64::MAX as usize, |upper| upper - 1).min(i64::MAX as usize) as u64,
                    )?;
                    let below = append(
                        block,
                        arith::cmpi(
                            value,
                            lower,
                            if signed {
                                arith::IntegerComparisonPredicate::SignedLessThan
                            } else {
                                arith::IntegerComparisonPredicate::UnsignedLessThan
                            },
                            self.location,
                        )?,
                    )?;
                    let above = append(
                        block,
                        arith::cmpi(
                            value,
                            upper,
                            if signed {
                                arith::IntegerComparisonPredicate::SignedGreaterThan
                            } else {
                                arith::IntegerComparisonPredicate::UnsignedGreaterThan
                            },
                            self.location,
                        )?,
                    )?;
                    let invalid = append(block, arith::ori(below, above, self.location)?)?;
                    let mut failure = self.context.block::<TypeRef<'c, 't>, UnknownLocationRef<'c, 't>>(&[]);
                    failure.append_operation(llvm::intr_trap(self.location)?)?;
                    failure.append_operation(scf::r#yield(&[], self.location)?)?;
                    block.append_operation(scf::r#if(invalid, &[], failure.try_into()?, None, self.location)?)?;
                    vec![Lowered::Dimension(append(
                        block,
                        arith::index_cast(value, self.context.index_type(), self.location)?,
                    )?)]
                }
                KernelOperation::Portable(ArrayIrOperation::DimensionToScalar(_)) => {
                    let value = append(
                        block,
                        arith::index_cast(
                            inputs[0].dimension(),
                            self.context.signless_integer_type(64),
                            self.location,
                        )?,
                    )?;
                    self.distributed(block, 1, |lowering, body, index| lowering.store(body, output(), index, value))?;
                    self.barrier(block)?;
                    vec![Lowered::Array(output().clone())]
                }
                KernelOperation::Portable(
                    ArrayIrOperation::ReferenceRead(_) | ArrayIrOperation::ReferenceFreeze(_),
                ) => {
                    self.read(block, inputs[0].reference(), output(), None, None)?;
                    vec![Lowered::Array(output().clone())]
                }
                KernelOperation::Portable(ArrayIrOperation::ReferenceWrite(_)) => {
                    self.write(block, inputs[0].reference(), inputs[1].array(), None)?;
                    vec![]
                }
                KernelOperation::Portable(ArrayIrOperation::ReferenceSwap(_)) => {
                    self.read(block, inputs[0].reference(), output(), None, None)?;
                    self.write(block, inputs[0].reference(), inputs[1].array(), None)?;
                    vec![Lowered::Array(output().clone())]
                }
                KernelOperation::Scratch(_) => vec![Lowered::Reference(self.reference(block, output().clone())?)],
                KernelOperation::AsyncCopy(_) => {
                    vec![Lowered::Token(self.async_copy(
                        block,
                        inputs[0].reference(),
                        inputs[1].reference(),
                        output_ids[0],
                    )?)]
                }
                KernelOperation::Wait(_) => {
                    for input in &inputs {
                        let Lowered::Token(token) = input else { unreachable!() };
                        self.wait(block, *token)?;
                    }
                    vec![]
                }
                KernelOperation::TileLoad(operation) => {
                    let rank = operation.block_shape().len();
                    let starts = inputs[1..rank + 1].iter().map(Lowered::dimension).collect::<Vec<_>>();
                    let window = self.window(block, inputs[0].reference(), &starts, operation.block_shape())?;
                    self.read(block, &window, output(), Some(inputs[rank + 1].array()), None)?;
                    vec![Lowered::Array(output().clone())]
                }
                KernelOperation::MaskedLoad(_) => {
                    self.read(
                        block,
                        inputs[0].reference(),
                        output(),
                        Some(inputs[2].array()),
                        Some(inputs[1].array()),
                    )?;
                    vec![Lowered::Array(output().clone())]
                }
                KernelOperation::MaskedStore(_) => {
                    self.write(block, inputs[0].reference(), inputs[1].array(), Some(inputs[2].array()))?;
                    vec![]
                }
                KernelOperation::MaskedSwap(_) => {
                    self.read(
                        block,
                        inputs[0].reference(),
                        output(),
                        Some(inputs[3].array()),
                        Some(inputs[2].array()),
                    )?;
                    self.write(block, inputs[0].reference(), inputs[1].array(), Some(inputs[2].array()))?;
                    vec![Lowered::Array(output().clone())]
                }
                KernelOperation::Portable(ArrayIrOperation::Condition(_)) => {
                    let zero = self.index(block, 0)?;
                    let predicate = self.load(block, inputs[0].array(), zero)?;
                    let mut branches = Vec::new();
                    for id in instruction.regions() {
                        let mut branch = self.context.block::<TypeRef<'c, 't>, UnknownLocationRef<'c, 't>>(&[]);
                        let results = self.region(&mut branch, region.with_id(*id)?, &inputs[1..])?;
                        for (result, destination) in results.iter().zip(&output_buffers) {
                            self.copy(&mut branch, result.array(), destination.as_ref().unwrap())?;
                        }
                        branch.append_operation(scf::r#yield(&[], self.location)?)?;
                        branches.push(branch.try_into()?);
                    }
                    let otherwise = branches.pop().unwrap();
                    let selected = branches.pop().unwrap();
                    block.append_operation(scf::r#if(predicate, &[], selected, Some(otherwise), self.location)?)?;
                    output_buffers.iter().map(|buffer| Lowered::Array(buffer.as_ref().unwrap().clone())).collect()
                }
                KernelOperation::Portable(ArrayIrOperation::While(operation)) => {
                    // The canonical while ABI has no separate capture prefix: every input is a carried value.
                    let mut carries = inputs.clone();
                    for (carry, destination) in carries.iter_mut().zip(&output_buffers) {
                        if let Lowered::Array(source) = carry {
                            let destination = destination.as_ref().unwrap();
                            self.copy(block, source, destination)?;
                            *source = destination.clone();
                        }
                    }
                    let initial = self.index(block, 0)?;
                    let bound = self.index(block, operation.iteration_bound().unwrap())?;
                    let index_type = self.context.index_type().as_ref();
                    let mut condition = self.context.block(&[(index_type, self.location)]);
                    let counter = condition.argument(0)?.as_ref();
                    let within_bound = append(
                        &mut condition,
                        arith::cmpi(
                            counter,
                            bound,
                            arith::IntegerComparisonPredicate::UnsignedLessThan,
                            self.location,
                        )?,
                    )?;
                    // Reaching the semantic bound skips the source condition, including all of its reference effects.
                    let mut evaluate = self.context.block::<TypeRef<'c, 't>, UnknownLocationRef<'c, 't>>(&[]);
                    let predicate = self.region(&mut evaluate, region.with_id(instruction.regions()[0])?, &carries)?;
                    let zero = self.index(&mut evaluate, 0)?;
                    let predicate = self.load(&mut evaluate, predicate[0].array(), zero)?;
                    evaluate.append_operation(scf::r#yield(&[predicate], self.location)?)?;
                    let mut finished = self.context.block::<TypeRef<'c, 't>, UnknownLocationRef<'c, 't>>(&[]);
                    let boolean_type = self.context.signless_integer_type(1);
                    let no = append(
                        &mut finished,
                        arith::constant(self.context.integer_attribute(boolean_type, 0), self.location)?,
                    )?;
                    finished.append_operation(scf::r#yield(&[no], self.location)?)?;
                    let predicate = append(
                        &mut condition,
                        scf::r#if(
                            within_bound,
                            &[boolean_type.as_ref()],
                            evaluate.try_into()?,
                            Some(finished.try_into()?),
                            self.location,
                        )?,
                    )?;
                    condition.append_operation(scf::condition(predicate, &[counter], self.location)?)?;
                    let mut body = self.context.block(&[(index_type, self.location)]);
                    let counter = body.argument(0)?.as_ref();
                    let source_body = region.with_id(instruction.regions()[1])?;
                    let next = self.region(&mut body, source_body, &carries)?;
                    // Snapshot every next array into the already planned region-input storage before overwriting any
                    // carry. Sequential copies directly into carries would corrupt swaps and other alias permutations.
                    let mut snapshots = Vec::new();
                    for (next, input) in next.iter().zip(source_body.input_ids()) {
                        if let Lowered::Array(source) = next {
                            let snapshot = self.storage[&ValueId::new(source_body.id(), *input)].clone();
                            self.copy(&mut body, source, &snapshot)?;
                            snapshots.push(Some(snapshot));
                        } else {
                            snapshots.push(None);
                        }
                    }
                    for (snapshot, carry) in snapshots.iter().zip(&carries) {
                        if let Some(snapshot) = snapshot {
                            self.copy(&mut body, snapshot, carry.array())?;
                        }
                    }
                    let one = self.index(&mut body, 1)?;
                    let next_counter = append(&mut body, arith::addi(counter, one, self.location)?)?;
                    body.append_operation(scf::r#yield(&[next_counter], self.location)?)?;
                    block.append_operation(scf::r#while(
                        &[initial],
                        &[index_type],
                        condition.try_into()?,
                        body.try_into()?,
                        self.location,
                    )?)?;
                    carries
                }
                operation => {
                    return Err(Error::Unsupported {
                        operation: operation.name(),
                        reason: "operation has no baseline GPU lowering".to_owned(),
                    });
                }
            };
            for (id, result) in instruction.outputs().iter().zip(results) {
                values[id.index()] = Some(result);
            }
        }
        Ok(region.output_ids().iter().map(|id| values[id.index()].clone().unwrap()).collect())
    }

    /// Composes canonical static slices and clamped indices without changing allocation ownership.
    fn apply_transform(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        source: &Reference<'c, 't>,
        transform: &ArrayReferenceTransform,
        inputs: &[Lowered<'c, 't>],
    ) -> Result<Reference<'c, 't>, Error> {
        match transform {
            ArrayReferenceTransform::Slice { axes } => {
                // A full slice leaves the address and validity unchanged. Preserve the whole-root proof needed by
                // asynchronous copies instead of routing this identity transform through the general window
                // calculation.
                if is_whole_slice(axes, &source.shape) {
                    return Ok(source.clone());
                }
                let starts = axes.iter().map(|axis| self.index(block, axis.start())).collect::<Result<Vec<_>, _>>()?;
                let shape = axes.iter().map(|axis| axis.size()).collect::<Vec<_>>();
                self.window(block, source, &starts, &shape)
            }
            ArrayReferenceTransform::Index { axis, index } => {
                let index = match index {
                    ArrayReferenceTransformIndex::Static(index) => self.index(block, *index)?,
                    ArrayReferenceTransformIndex::Dynamic => {
                        let zero = self.index(block, 0)?;
                        let value = self.load(block, inputs[0].array(), zero)?;
                        let unsigned = matches!(inputs[0].array().r#type.data_type(), DataType::U32 | DataType::U64);
                        let value = if unsigned {
                            append(block, arith::index_castui(value, self.context.index_type(), self.location)?)?
                        } else {
                            append(block, arith::index_cast(value, self.context.index_type(), self.location)?)?
                        };
                        let nonnegative = if unsigned {
                            value
                        } else {
                            // Negative indices count from the end of the axis once before clamping, exactly as the
                            // dynamic slices that reference discharge stages for this transform do.
                            let extent = self.index(block, source.shape[*axis])?;
                            let wrapped = append(block, arith::addi(value, extent, self.location)?)?;
                            let negative = append(
                                block,
                                arith::cmpi(
                                    value,
                                    zero,
                                    arith::IntegerComparisonPredicate::SignedLessThan,
                                    self.location,
                                )?,
                            )?;
                            let value = append(block, arith::select(negative, wrapped, value, self.location)?)?;
                            append(block, arith::maxsi(value, zero, self.location)?)?
                        };
                        let maximum = self.index(block, source.shape[*axis].saturating_sub(1))?;
                        append(block, arith::minui(nonnegative, maximum, self.location)?)?
                    }
                };
                let mut starts = vec![self.index(block, 0)?; source.shape.len()];
                starts[*axis] = index;
                let mut shape = source.shape.clone();
                shape[*axis] = 1;
                let mut result = self.window(block, source, &starts, &shape)?;
                result.shape.remove(*axis);
                result.strides.remove(*axis);
                let selected_valid = result.valid.remove(*axis);
                let zero = self.index(block, 0)?;
                let selected = append(
                    block,
                    arith::cmpi(selected_valid, zero, arith::IntegerComparisonPredicate::NotEqual, self.location)?,
                )?;
                result.predicate = Some(if let Some(predicate) = result.predicate {
                    append(block, arith::andi(predicate, selected, self.location)?)?
                } else {
                    selected
                });
                Ok(result)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::kernels::{
        AsyncCopyOperation, Grid, GridDimension, KernelCallOperation, KernelDefinition, KernelParameterAccess,
        MaskedLoadOperation, ScratchOperation, WaitOperation, whole_array_parameter,
    };
    use ryft_core::{
        AddOperation, ArrayIrType, Context as CoreContext, Dimension, NestedTracingContext, ReferenceRead,
        ReferenceWrite, WhileOperation,
    };
    use ryft_mlir::dialects::mosaic::gpu::mosaic_gpu_serde_version;
    use ryft_mlir::{WalkOrder, WalkResult};
    use ryft_xla_sys::mlir::dialects::mosaic::gpu::MOSAIC_GPU_SERDE_VERSION;

    use super::*;

    /// Extracts a small structural snapshot of native control flow, independent of SSA printer numbering.
    fn control_snapshot(module: &Module<'_, '_>) -> Vec<String> {
        let mut names = Vec::new();
        module.as_operation().unwrap().walk(WalkOrder::PreOrder, |operation| {
            let name = operation.name().as_str().unwrap().to_owned();
            if matches!(
                name.as_str(),
                "gpu.launch" | "gpu.barrier" | "gpu.terminator" | "scf.for" | "scf.while" | "scf.if"
            ) {
                names.push(name);
            }
            WalkResult::Advance
        });
        names
    }

    /// Verifies and decodes the actual binary source consumed by the native runtime.
    fn assert_serializes(module: &Module<'_, '_>) {
        assert_eq!(module.verify(), Ok(true));
        let bytes = serialize(module).unwrap();
        let decoded = module.context().parse_module_from_bytes(&bytes).unwrap();
        assert_eq!(decoded.verify(), Ok(true));
        assert_eq!(mosaic_gpu_serde_version(&decoded), Ok(Some(i64::from(MOSAIC_GPU_SERDE_VERSION))));
    }

    #[test]
    fn test_module_scalar_reference_copy() {
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![whole_array_parameter(ArrayType::scalar(DataType::I32), KernelParameterAccess::ReadWrite).unwrap()],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
            references[0].write(&references[0].read()?)?;
            Ok(())
        })
        .unwrap();
        assert_eq!(definition.interpret(vec![Array::scalar(7i32).unwrap()], 1), Ok(vec![Array::scalar(7i32).unwrap()]));
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let context = Context::new();
        let (module, arguments, slots, shared) =
            module(&context, &verified, &Target::new(9, 0).unwrap(), &Options::default(), &KernelSchedule::default())
                .unwrap();
        assert_eq!(arguments, vec![ArrayType::scalar(DataType::I32), ArrayType::scalar(DataType::I32)]);
        assert_eq!(slots, vec![1]);
        assert_eq!(shared, 16);
        assert_eq!(
            control_snapshot(&module),
            vec![
                "gpu.launch",
                "scf.for",
                "scf.if",
                "gpu.barrier",
                "scf.for",
                "scf.if",
                "gpu.barrier",
                "gpu.terminator"
            ]
        );
        assert_serializes(&module);
    }

    #[test]
    fn test_module_async_copy_reference_transforms() {
        // Full identity slices keep the whole-root proof, so they are admitted and lower to the same native copy as a
        // copy without transforms.
        let r#type = ArrayType::new_static(DataType::F32, [3]);
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![whole_array_parameter(r#type.clone(), KernelParameterAccess::ReadOnly).unwrap()],
        )
        .unwrap();
        let transforms = vec![ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 3, 1)] }];
        let copy = AsyncCopyOperation::new()
            .with_source_transforms(transforms.clone())
            .with_destination_transforms(transforms);
        let scratch = ScratchOperation::new(r#type, 16).unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
            let context = references[0].context();
            let destination = context.bind(scratch.clone(), vec![], &[])?.remove(0);
            let token = context.bind(copy.clone(), vec![], &[references[0].clone(), destination])?.remove(0);
            context.bind(WaitOperation, vec![], &[token])?;
            Ok(())
        })
        .unwrap();
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let context = Context::new();
        let (module, _, _, _) =
            module(&context, &verified, &Target::new(9, 0).unwrap(), &Options::default(), &KernelSchedule::default())
                .unwrap();
        let mut copies = 0;
        module.as_operation().unwrap().walk(WalkOrder::PreOrder, |operation| {
            if operation.name().as_str() == Ok("nvgpu.device_async_copy") {
                copies += 1;
            }
            WalkResult::Advance
        });
        assert_eq!(copies, 1);
        assert_serializes(&module);
    }

    #[test]
    fn test_module_cluster_coordinates() {
        use ryft_mlir::{BlockArgumentRef, OperationResultRef};
        let grid = Grid::new(vec![GridDimension::new(Dimension::Static(4), GridExecution::Parallel)]).unwrap();
        let definition: KernelDefinition =
            KernelDefinition::trace(KernelCallOperation::new(grid, vec![]).unwrap(), |_| Ok(())).unwrap();
        let verified = VerifiedKernel::new(&definition, 4).unwrap();
        let context = Context::new();
        let (module, _, _, _) = module(
            &context,
            &verified,
            &Target::new(9, 0).unwrap().with_blocks_per_cluster(2).unwrap(),
            &Options::default(),
            &KernelSchedule::default(),
        )
        .unwrap();
        let mut divisions = Vec::new();
        module.as_operation().unwrap().walk(WalkOrder::PreOrder, |operation| {
            if operation.name().as_str().unwrap() == "arith.remui" {
                let numerator =
                    operation.operand_value(0).unwrap().cast::<BlockArgumentRef>().unwrap().argument_index();
                let denominator = operation
                    .operand_value(1)
                    .unwrap()
                    .cast::<OperationResultRef>()
                    .unwrap()
                    .operation()
                    .unwrap()
                    .attribute("value")
                    .unwrap()
                    .unwrap()
                    .to_string();
                divisions.push((numerator, denominator));
            }
            WalkResult::Advance
        });
        assert_eq!(divisions, vec![(0, "2 : index".to_owned()), (12, "4 : index".to_owned())]);
        assert_serializes(&module);
    }

    #[test]
    fn test_module_masked_vector_load() {
        let r#type = ArrayType::new_static(DataType::I32, [4]);
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(r#type.clone(), KernelParameterAccess::ReadOnly).unwrap(),
                whole_array_parameter(r#type, KernelParameterAccess::WriteOnly).unwrap(),
            ],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
            let context = references[0].context();
            let mask = context.lift(ArrayIrValue::Array(Array::vector(vec![true, false, true, false])?))?;
            let other = context.lift(ArrayIrValue::Array(Array::vector(vec![-1i32; 4])?))?;
            let value =
                context.bind(MaskedLoadOperation::new(), vec![], &[references[0].clone(), mask, other])?.remove(0);
            references[1].write(&value)?;
            Ok(())
        })
        .unwrap();
        assert_eq!(
            definition.interpret(vec![Array::vector(vec![3i32, 4, 5, 6]).unwrap()], 1),
            Ok(vec![Array::vector(vec![3i32, -1, 5, -1]).unwrap()])
        );
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let context = Context::new();
        let (module, _, slots, _) =
            module(&context, &verified, &Target::new(9, 0).unwrap(), &Options::default(), &KernelSchedule::default())
                .unwrap();
        assert_eq!(slots, vec![0, 1]);
        assert_serializes(&module);
    }

    #[test]
    fn test_module_while_semantic_bound() {
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![whole_array_parameter(ArrayType::scalar(DataType::I64), KernelParameterAccess::ReadWrite).unwrap()],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
            let context = references[0].context().clone();
            let input = references[0].read()?;
            let types = vec![ArrayIrType::Array(ArrayType::scalar(DataType::I64))];
            let (_, condition) = NestedTracingContext::trace(
                context.clone(),
                |inputs: Vec<_>| Ok(vec![inputs[0].context().lift(ArrayIrValue::Array(Array::scalar(true)?))?]),
                types.clone(),
            )?;
            let (_, body) = NestedTracingContext::trace(
                context.clone(),
                |inputs: Vec<_>| {
                    let context = inputs[0].context();
                    let one = context.lift(ArrayIrValue::Array(Array::scalar(1i64)?))?;
                    context.bind(
                        ArrayIrOperation::from(ArrayOperation::Add(AddOperation::new())),
                        vec![],
                        &[inputs[0].clone(), one],
                    )
                },
                types,
            )?;
            let output = context
                .bind(
                    ArrayIrOperation::While(WhileOperation::new().with_iteration_bound(3)?),
                    vec![condition, body],
                    &[input],
                )?
                .remove(0);
            references[0].write(&output)?;
            Ok(())
        })
        .unwrap();
        // The always-true condition is deliberately unable to terminate this loop; only the semantic bound does.
        assert_eq!(
            definition.interpret(vec![Array::scalar(7i64).unwrap()], 1),
            Ok(vec![Array::scalar(10i64).unwrap()])
        );
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let context = Context::new();
        let (module, _, _, _) =
            module(&context, &verified, &Target::new(9, 0).unwrap(), &Options::default(), &KernelSchedule::default())
                .unwrap();
        let mut loop_signature = Vec::new();
        module.as_operation().unwrap().walk(WalkOrder::PreOrder, |operation| {
            let name = operation.name().as_str().unwrap().to_owned();
            if matches!(name.as_str(), "scf.while" | "scf.condition") {
                loop_signature.push((name, operation.operand_count(), operation.result_count()));
            }
            WalkResult::Advance
        });
        assert_eq!(loop_signature, vec![("scf.while".to_owned(), 1, 1), ("scf.condition".to_owned(), 2, 0)]);
        assert_serializes(&module);
    }

    #[test]
    fn test_module_while_parallel_carries() {
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![whole_array_parameter(ArrayType::scalar(DataType::I64), KernelParameterAccess::ReadWrite).unwrap()],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
            let context = references[0].context();
            let first = references[0].read()?;
            let second = context.lift(ArrayIrValue::Array(Array::scalar(11i64)?))?;
            let outputs = ryft_core::kernels::for_loop(context, 0..1, vec![first, second], |_, inputs| {
                Ok(vec![inputs[1].clone(), inputs[0].clone()])
            })?;
            references[0].write(&outputs[0])?;
            Ok(())
        })
        .unwrap();
        assert_eq!(
            definition.interpret(vec![Array::scalar(7i64).unwrap()], 1),
            Ok(vec![Array::scalar(11i64).unwrap()])
        );
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let context = Context::new();
        let (module, _, _, _) =
            module(&context, &verified, &Target::new(9, 0).unwrap(), &Options::default(), &KernelSchedule::default())
                .unwrap();
        assert_serializes(&module);
    }

    #[test]
    fn test_validate_rejects_unsupported_array_operation() {
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![whole_array_parameter(ArrayType::scalar(DataType::F32), KernelParameterAccess::ReadOnly).unwrap()],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
            let value = references[0].read()?;
            value.context().bind(
                ArrayIrOperation::from(ArrayOperation::Sqrt(ryft_core::SqrtOperation::new())),
                vec![],
                &[value.clone()],
            )?;
            Ok(())
        })
        .unwrap();
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        assert!(
            matches!(validate(&verified, &Target::new(9, 0).unwrap(), &Options::default(), &KernelSchedule::default()),
            Err(Error::Unsupported { operation: "sqrt", reason })
                if reason == "operation or its metadata has no baseline scalar implementation")
        );
    }

    #[test]
    fn test_validate_rejects_assertions() {
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![whole_array_parameter(ArrayType::scalar(DataType::Boolean), KernelParameterAccess::ReadOnly).unwrap()],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
            let predicate = references[0].read()?;
            predicate.context().bind(
                ArrayIrOperation::Assert(ryft_core::AssertOperation::new("check")),
                vec![],
                &[predicate.clone()],
            )?;
            Ok(())
        })
        .unwrap();
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        assert!(
            matches!(validate(&verified, &Target::new(9, 0).unwrap(), &Options::default(), &KernelSchedule::default()),
            Err(Error::Unsupported { operation: "assert", reason }) if reason == "runtime assertions require native failure propagation")
        );
    }

    #[test]
    fn test_validate_rejects_partial_async_copy_transforms() {
        // Admission rejects a partial source or destination selection before any native module is constructed.
        // `test_module_async_copy_reference_transforms` covers the admitted full identity slices.
        for (source_extent, destination_extent) in [(3, 2), (2, 3)] {
            let operation = KernelCallOperation::new(
                Grid::new(vec![]).unwrap(),
                vec![
                    whole_array_parameter(
                        ArrayType::new_static(DataType::F32, [source_extent]),
                        KernelParameterAccess::ReadOnly,
                    )
                    .unwrap(),
                ],
            )
            .unwrap();
            let transforms = vec![ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 2, 1)] }];
            let copy = AsyncCopyOperation::new()
                .with_source_transforms(transforms.clone())
                .with_destination_transforms(transforms);
            let scratch =
                ScratchOperation::new(ArrayType::new_static(DataType::F32, [destination_extent]), 16).unwrap();
            let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
                let context = references[0].context();
                let destination = context.bind(scratch.clone(), vec![], &[])?.remove(0);
                let token = context.bind(copy.clone(), vec![], &[references[0].clone(), destination])?.remove(0);
                context.bind(WaitOperation, vec![], &[token])?;
                Ok(())
            })
            .unwrap();
            let verified = VerifiedKernel::new(&definition, 1).unwrap();
            assert!(matches!(
                validate(&verified, &Target::new(9, 0).unwrap(), &Options::default(), &KernelSchedule::default()),
                Err(Error::Unsupported { operation: "async_copy", reason })
                    if reason == "native async copies require transforms that select whole references",
            ));
        }
    }

    #[test]
    fn test_module_zero_and_sequential_grid() {
        for (extent, expected) in
            [(0, vec!["gpu.launch", "gpu.terminator"]), (2, vec!["gpu.launch", "scf.for", "gpu.terminator"])]
        {
            let grid =
                Grid::new(vec![GridDimension::new(Dimension::Static(extent), GridExecution::Sequential)]).unwrap();
            let definition: KernelDefinition =
                KernelDefinition::trace(KernelCallOperation::new(grid, vec![]).unwrap(), |_| Ok(())).unwrap();
            let verified = VerifiedKernel::new(&definition, 2).unwrap();
            let context = Context::new();
            let (module, arguments, slots, shared) = module(
                &context,
                &verified,
                &Target::new(9, 0).unwrap(),
                &Options::default(),
                &KernelSchedule::default(),
            )
            .unwrap();
            assert_eq!(arguments, Vec::<ArrayType>::new());
            assert_eq!(slots, Vec::<usize>::new());
            assert_eq!(shared, 0);
            assert_eq!(control_snapshot(&module), expected);
            assert_serializes(&module);
        }
    }
    #[test]
    fn test_lowering_apply_transform_dynamic_indices() {
        use ryft_core::{ArrayReferenceTransformIndex, RegionId};
        for (data_type, cast, signed) in [
            (DataType::I32, "arith.index_cast", true),
            (DataType::I64, "arith.index_cast", true),
            (DataType::U32, "arith.index_castui", false),
            (DataType::U64, "arith.index_castui", false),
        ] {
            let context = Context::new();
            let source_type = ArrayType::new_static(DataType::F32, [3]);
            let index_type = ArrayType::scalar(data_type);
            let module = module::build(
                &context,
                &Target::new(9, 0).unwrap(),
                "clamped_index",
                &[(0, source_type.clone()), (1, index_type.clone())],
                &[],
                &[],
                &[],
                [1, 1, 1],
                |block, globals, _, _| {
                    let lowering = Lowering {
                        context: &context,
                        location: context.unknown_location(),
                        thread: block.argument(3)?.as_ref(),
                        threads: block.argument(9)?.as_ref(),
                        storage: HashMap::new(),
                        tma_descriptors: HashMap::new(),
                        instruction_scratch: HashMap::new(),
                        mma_stages: 1,
                        cluster_rank: None,
                        synchronization: None,
                        current_instruction: None,
                        next_barrier: 0,
                    };
                    let source = lowering.reference(
                        block,
                        Buffer {
                            owner: ValueId::new(RegionId::new(0), AtomId::new(0)),
                            shared: false,
                            value: globals[0],
                            r#type: source_type.clone(),
                        },
                    )?;
                    let index = Lowered::Array(Buffer {
                        owner: ValueId::new(RegionId::new(0), AtomId::new(1)),
                        shared: false,
                        value: globals[1],
                        r#type: index_type.clone(),
                    });
                    let view = lowering.apply_transform(
                        block,
                        &source,
                        &ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
                        &[index],
                    )?;
                    assert_eq!(view.shape, Vec::<usize>::new());
                    assert!(view.predicate.is_some());
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(module.verify(), Ok(true));
            let mut clamps = Vec::new();
            module.as_operation().unwrap().walk(WalkOrder::PreOrder, |operation| {
                let name = operation.name();
                let name = name.as_str().unwrap();
                if matches!(
                    name,
                    "arith.index_cast"
                        | "arith.index_castui"
                        | "arith.cmpi"
                        | "arith.select"
                        | "arith.maxsi"
                        | "arith.minui"
                ) {
                    clamps.push(name.to_owned());
                }
                WalkResult::Advance
            });
            // Signed indices first count negative values from the end of the axis, then clamp to the valid range.
            let mut expected = vec![cast.to_owned()];
            if signed {
                expected.extend(["arith.cmpi".to_owned(), "arith.select".to_owned(), "arith.maxsi".to_owned()]);
            }
            expected.extend(vec!["arith.minui".to_owned(); 3]);
            // The window's validity predicate compares the clamped index against the axis extent.
            expected.push("arith.cmpi".to_owned());
            assert_eq!(clamps, expected);
        }
    }

    #[test]
    fn test_lowering_apply_transform_masked_scalar_preserves_predicate() {
        use ryft_core::{ArrayReferenceTransformIndex, RegionId};
        let context = Context::new();
        let source_type = ArrayType::new_static(DataType::F32, [3]);
        let module = module::build(
            &context,
            &Target::new(9, 0).unwrap(),
            "masked_scalar",
            &[(0, source_type.clone())],
            &[],
            &[],
            &[],
            [1, 1, 1],
            |block, globals, _, _| {
                let lowering = Lowering {
                    context: &context,
                    location: context.unknown_location(),
                    thread: block.argument(3)?.as_ref(),
                    threads: block.argument(9)?.as_ref(),
                    storage: HashMap::new(),
                    tma_descriptors: HashMap::new(),
                    instruction_scratch: HashMap::new(),
                    mma_stages: 1,
                    cluster_rank: None,
                    synchronization: None,
                    current_instruction: None,
                    next_barrier: 0,
                };
                let source = lowering.reference(
                    block,
                    Buffer {
                        owner: ValueId::new(RegionId::new(0), AtomId::new(0)),
                        shared: false,
                        value: globals[0],
                        r#type: source_type.clone(),
                    },
                )?;
                let zero = lowering.index(block, 0)?;
                let window = lowering.window(block, &source, &[zero], &[8])?;
                let scalar = lowering.apply_transform(
                    block,
                    &window,
                    &ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(7) },
                    &[],
                )?;
                assert_eq!(scalar.shape, Vec::<usize>::new());
                assert_eq!(scalar.valid.len(), 0);
                let predicate = scalar.predicate.unwrap();
                let (_, valid) = lowering.address(block, &scalar, zero)?;
                let guard = block.operations()?.last().unwrap()?;
                assert_eq!(guard.name().as_str(), Ok("arith.andi"));
                assert_eq!(guard.operand_value(1)?, predicate);
                assert_eq!(guard.result(0)?.as_ref(), valid);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(module.verify(), Ok(true));
    }

    #[test]
    fn test_module_condition_preserves_branches() {
        use ryft_mlir::Region;
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(ArrayType::scalar(DataType::Boolean), KernelParameterAccess::ReadOnly).unwrap(),
                whole_array_parameter(ArrayType::scalar(DataType::I64), KernelParameterAccess::ReadWrite).unwrap(),
            ],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
            let context = references[0].context().clone();
            let predicate = references[0].read()?;
            let value = references[1].read()?;
            let result = ryft_core::kernels::condition(
                &context,
                &predicate,
                vec![value],
                |inputs| {
                    inputs[0].context().bind(
                        ArrayIrOperation::from(ArrayOperation::Add(AddOperation::new())),
                        vec![],
                        &[inputs[0].clone(), inputs[0].clone()],
                    )
                },
                Ok,
            )?;
            references[1].write(&result[0])?;
            Ok(())
        })
        .unwrap();
        assert_eq!(
            definition.interpret(vec![Array::scalar(true).unwrap(), Array::scalar(7i64).unwrap()], 1),
            Ok(vec![Array::scalar(14i64).unwrap()])
        );
        assert_eq!(
            definition.interpret(vec![Array::scalar(false).unwrap(), Array::scalar(7i64).unwrap()], 1),
            Ok(vec![Array::scalar(7i64).unwrap()])
        );
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let context = Context::new();
        let (module, _, _, _) =
            module(&context, &verified, &Target::new(9, 0).unwrap(), &Options::default(), &KernelSchedule::default())
                .unwrap();
        assert_eq!(module.verify(), Ok(true));
        let mut branch_adds = Vec::new();
        module.as_operation().unwrap().walk(WalkOrder::PreOrder, |operation| {
            if operation.name().as_str() == Ok("scf.if") && operation.region_count() == 2 {
                let mut counts = Vec::new();
                for region in operation.regions() {
                    let mut count = 0;
                    for block in region.unwrap().blocks().unwrap() {
                        for child in block.unwrap().operations().unwrap() {
                            child.unwrap().walk(WalkOrder::PreOrder, |nested| {
                                if nested.name().as_str() == Ok("arith.addi") {
                                    count += 1;
                                }
                                WalkResult::Advance
                            });
                        }
                    }
                    counts.push(count);
                }
                if counts.iter().any(|count| *count > 0) {
                    branch_adds.push(counts);
                }
            }
            WalkResult::Advance
        });
        assert_eq!(branch_adds, vec![vec![1, 0]]);
        assert_serializes(&module);
    }

    #[test]
    fn test_module_dimension_from_scalar_bounds() {
        use ryft_core::{DimensionBounds, DimensionFromScalarOperation, DimensionToScalarOperation, DimensionVariable};
        for (data_type, extension, lower_predicate, upper_predicate) in [
            (
                DataType::I32,
                Some("arith.extsi"),
                arith::IntegerComparisonPredicate::SignedLessThan,
                arith::IntegerComparisonPredicate::SignedGreaterThan,
            ),
            (
                DataType::I64,
                None,
                arith::IntegerComparisonPredicate::SignedLessThan,
                arith::IntegerComparisonPredicate::SignedGreaterThan,
            ),
            (
                DataType::U32,
                Some("arith.extui"),
                arith::IntegerComparisonPredicate::UnsignedLessThan,
                arith::IntegerComparisonPredicate::UnsignedGreaterThan,
            ),
            (
                DataType::U64,
                None,
                arith::IntegerComparisonPredicate::UnsignedLessThan,
                arith::IntegerComparisonPredicate::UnsignedGreaterThan,
            ),
        ] {
            let call = KernelCallOperation::new(
                Grid::new(vec![]).unwrap(),
                vec![
                    whole_array_parameter(ArrayType::scalar(data_type), KernelParameterAccess::ReadOnly).unwrap(),
                    whole_array_parameter(ArrayType::scalar(DataType::I64), KernelParameterAccess::WriteOnly).unwrap(),
                ],
            )
            .unwrap();
            let definition: KernelDefinition = KernelDefinition::trace(call, |(references, _)| {
                let context = references[0].context();
                let source = references[0].read()?;
                let dimension = context.bind(
                    ArrayIrOperation::DimensionFromScalar(DimensionFromScalarOperation::new(DimensionVariable::new(
                        "extent",
                        DimensionBounds::new(2, Some(5)).unwrap(),
                    ))),
                    vec![],
                    &[source],
                )?;
                let scalar = context.bind(
                    ArrayIrOperation::DimensionToScalar(DimensionToScalarOperation),
                    vec![],
                    &dimension,
                )?;
                references[1].write(&scalar[0])?;
                Ok(())
            })
            .unwrap();
            let verified = VerifiedKernel::new(&definition, 1).unwrap();
            let context = Context::new();
            let (module, _, _, _) = module(
                &context,
                &verified,
                &Target::new(9, 0).unwrap(),
                &Options::default(),
                &KernelSchedule::default(),
            )
            .unwrap();
            assert_eq!(module.verify(), Ok(true));
            let mut literals = Vec::new();
            let mut comparisons = Vec::new();
            module.as_operation().unwrap().walk(WalkOrder::PreOrder, |operation| {
                if operation.name().as_str() == Ok("arith.constant")
                    && operation.result(0).unwrap().r#type().unwrap() == context.signless_integer_type(64).as_ref()
                {
                    literals.push((
                        operation.result(0).unwrap().as_ref(),
                        operation.integer_attribute("value").unwrap().signless_value(),
                    ));
                }
                if operation.name().as_str() == Ok("arith.cmpi")
                    && operation.operand_value(0).unwrap().r#type().unwrap()
                        == context.signless_integer_type(64).as_ref()
                {
                    let right = operation.operand_value(1).unwrap();
                    let bound = literals.iter().find(|(value, _)| *value == right).unwrap().1;
                    comparisons.push((operation.integer_attribute("predicate").unwrap().signless_value(), bound));
                }
                WalkResult::Advance
            });
            assert_eq!(comparisons, vec![(lower_predicate as i64, 2), (upper_predicate as i64, 4)]);
            let mut guarded = Vec::new();
            module.as_operation().unwrap().walk(WalkOrder::PreOrder, |operation| {
                if matches!(operation.name().as_str(), Ok("arith.extsi" | "arith.extui" | "llvm.intr.trap")) {
                    guarded.push(operation.name().as_str().unwrap().to_owned());
                }
                WalkResult::Advance
            });
            let mut expected = extension.into_iter().map(str::to_owned).collect::<Vec<_>>();
            expected.push("llvm.intr.trap".to_owned());
            assert_eq!(guarded, expected);
            assert_serializes(&module);
        }
    }
    #[test]
    fn test_module_while_preserves_dimension_captures() {
        for replace_coordinate in [false, true] {
            let grid = Grid::new(vec![GridDimension::new(Dimension::Static(2), GridExecution::Parallel)]).unwrap();
            let definition: KernelDefinition =
                KernelDefinition::trace(KernelCallOperation::new(grid, vec![]).unwrap(), |(_, coordinates)| {
                    let context = coordinates[0].context().clone();
                    let types = vec![coordinates[0].r#type().into_owned(); 2];
                    let (_, condition) = NestedTracingContext::trace(
                        context.clone(),
                        |inputs: Vec<_>| Ok(vec![inputs[0].context().lift(ArrayIrValue::Array(Array::scalar(true)?))?]),
                        types.clone(),
                    )?;
                    let (_, body) = NestedTracingContext::trace(
                        context.clone(),
                        |inputs: Vec<_>| {
                            if replace_coordinate { Ok(vec![inputs[1].clone(), inputs[0].clone()]) } else { Ok(inputs) }
                        },
                        types,
                    )?;
                    context.bind(
                        ArrayIrOperation::While(WhileOperation::new().with_iteration_bound(1)?),
                        vec![condition, body],
                        &[coordinates[0].clone(), coordinates[0].clone()],
                    )?;
                    Ok(())
                })
                .unwrap();
            let verified = VerifiedKernel::new(&definition, 2).unwrap();
            let context = Context::new();
            if replace_coordinate {
                assert!(matches!(
                    validate(&verified, &Target::new(9, 0).unwrap(), &Options::default(), &KernelSchedule::default()),
                    Err(Error::Unsupported { operation: "while", reason })
                        if reason == "operation has no baseline GPU lowering"
                ));
            } else {
                let (module, _, _, _) = module(
                    &context,
                    &verified,
                    &Target::new(9, 0).unwrap(),
                    &Options::default(),
                    &KernelSchedule::default(),
                )
                .unwrap();
                assert_serializes(&module);
            }
        }
    }

    #[test]
    fn test_module_while_preserves_reference_captures() {
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![whole_array_parameter(ArrayType::scalar(DataType::I32), KernelParameterAccess::ReadWrite).unwrap()],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
            let context = references[0].context().clone();
            let types = vec![references[0].r#type().into_owned()];
            let (_, condition) = NestedTracingContext::trace(
                context.clone(),
                |inputs: Vec<_>| Ok(vec![inputs[0].context().lift(ArrayIrValue::Array(Array::scalar(true)?))?]),
                types.clone(),
            )?;
            let (_, body) = NestedTracingContext::trace(context.clone(), |inputs: Vec<_>| Ok(inputs), types)?;
            context.bind(
                ArrayIrOperation::While(WhileOperation::new().with_iteration_bound(1)?),
                vec![condition, body],
                &references,
            )?;
            Ok(())
        })
        .unwrap();
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let context = Context::new();
        let (module, _, _, _) =
            module(&context, &verified, &Target::new(9, 0).unwrap(), &Options::default(), &KernelSchedule::default())
                .unwrap();
        assert_serializes(&module);
    }
}
