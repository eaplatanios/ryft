//! Checked native ABI and conservative CTA-local storage accounting before MLIR construction.

use std::collections::HashMap;

use ryft_core::kernels::{
    GridExecution, KernelExtension, KernelOperation, KernelParameterAccess, KernelSchedule, VerifiedKernel,
};
use ryft_core::{
    ArrayAddressing, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, AtomId, DataType,
    Dimension, InstructionId, Layout, Memory, Operation, RegionId, Typed, ValueId,
};

use crate::kernels::gpu::lowering::memory::{TmaCopyPlan, plan as plan_tma_copies};
use crate::kernels::gpu::{Error, GpuOperation, Mma, Options, Target, TmemOperation};

/// Complete immutable resource plan. Every private array receives distinct storage; lifetime reuse is not assumed.
pub(in crate::kernels::gpu) struct Plan {
    /// All private array values and scratch roots in canonical region/atom order.
    pub(super) storage: Vec<(ValueId, ArrayType)>,

    /// Native arguments in logical-input then logical-output order, retaining alias duplicates.
    pub(super) argument_types: Vec<ArrayType>,

    /// Native argument slot of each body reference parameter.
    pub(super) parameter_slots: Vec<usize>,

    /// Conservative aligned byte count for all private storage.
    pub(super) shared_memory_bytes: usize,

    /// Static logical grid dimensions in source order.
    pub(super) grid_extents: Vec<usize>,

    /// TMA descriptor and transaction-barrier owners in canonical instruction order.
    pub(super) tma_copies: Vec<TmaCopyPlan>,

    /// Aligned operand transport allocations for each native matrix or scale-transfer instruction.
    pub(super) instruction_scratch: Vec<(InstructionId, Vec<ArrayType>)>,

    /// Whether the final CTA epilogue must relinquish its tensor-memory allocation permit.
    pub(super) uses_tmem: bool,
}

impl Plan {
    /// Checks source work, native integer limits, ABI layout, and the complete private storage footprint.
    pub(super) fn new<Extension: KernelExtension + Into<GpuOperation>>(
        kernel: &VerifiedKernel<'_, Extension>,
        target: &Target,
        options: &Options,
        schedule: &KernelSchedule,
    ) -> Result<Self, Error> {
        if options.mma().is_some()
            || kernel.definition().body().entry_region_ref().instructions_in_closure().any(|(_, instruction)| {
                matches!(instruction.operation(), KernelOperation::Extension(extension)
                    if matches!(extension.clone().into(), GpuOperation::Wgmma))
            })
        {
            if target.compute_capability() != (9, 0) || target.threads_per_block() != 128 {
                return Err(Error::Invalid {
                    message: "wgmma requires compute capability 9.0 and exactly 128 threads per block".to_owned(),
                });
            }
            if !(1..=8).contains(&schedule.pipeline_stages().map(|stages| stages.get()).unwrap_or(1)) {
                return Err(Error::Invalid { message: "wgmma stages must be in [1, 8]".to_owned() });
            }
        }
        let logical = kernel.definition().operation();
        if !logical.prefetch_types().is_empty() {
            return Err(Error::Unsupported {
                operation: "kernel_call",
                reason: "scalar prefetch must be specialized".to_owned(),
            });
        }
        let grid_extents = logical
            .grid()
            .dimensions()
            .iter()
            .map(|dimension| match dimension.extent() {
                Dimension::Static(extent) => Ok(*extent),
                _ => Err(Error::Unsupported {
                    operation: "kernel_call",
                    reason: "grid dimensions must be static extents".to_owned(),
                }),
            })
            .collect::<Result<Vec<_>, _>>()?;
        grid_extents
            .iter()
            .zip(logical.grid().dimensions())
            .filter(|(_, dimension)| dimension.execution() == GridExecution::Parallel)
            .try_fold(target.blocks_per_cluster() as usize, |size, (extent, _)| size.checked_mul(*extent))
            .filter(|size| *size <= i32::MAX as usize)
            .ok_or_else(|| Error::Invalid {
                message: "flattened grid exceeds the native launch dimension range".to_owned(),
            })?;
        let inputs = logical
            .parameters()
            .iter()
            .filter(|parameter| parameter.access() != KernelParameterAccess::WriteOnly)
            .map(|parameter| parameter.r#type().into_owned())
            .collect::<Vec<_>>();
        let outputs = logical
            .parameters()
            .iter()
            .filter(|parameter| parameter.access() != KernelParameterAccess::ReadOnly)
            .map(|parameter| parameter.r#type().into_owned())
            .collect::<Vec<_>>();
        let argument_types = inputs.iter().chain(&outputs).cloned().collect::<Vec<_>>();
        if argument_types.len() > i32::MAX as usize {
            return Err(Error::Invalid {
                message: "native buffer argument count exceeds the signed index range".to_owned(),
            });
        }
        for r#type in &argument_types {
            checked_array(r#type)?;
        }
        let mut input = 0;
        let mut output = inputs.len();
        let parameter_slots = logical
            .parameters()
            .iter()
            .map(|parameter| {
                let slot = if parameter.access() == KernelParameterAccess::ReadOnly { input } else { output };
                if parameter.access() != KernelParameterAccess::WriteOnly {
                    input += 1;
                }
                if parameter.access() != KernelParameterAccess::ReadOnly {
                    output += 1;
                }
                slot
            })
            .collect();
        let body = kernel.definition().body();
        let mut work = 0usize;
        let mut storage = Vec::new();
        let mut instruction_scratch = Vec::new();
        let mut tmem_allocations = HashMap::<ValueId, usize>::new();
        let mut tmem_pending = HashMap::<ValueId, bool>::new();
        let mut uses_tmem = false;
        for (region_index, region) in body.regions().iter().enumerate() {
            work = checked_work(work, region.instructions().len(), options.maximum_instructions())?;
            for (atom_index, atom) in region.atoms().iter().enumerate() {
                if let Some(constant) = atom.as_constant() {
                    let count = match constant {
                        ArrayIrValue::Array(value) => {
                            ArrayAddressing::new(value.r#type().into_owned())?.element_count()
                        }
                        ArrayIrValue::Dimension(_) => 1,
                        ArrayIrValue::Reference(_) => 0,
                    };
                    work = checked_work(work, count, options.maximum_instructions())?;
                }
                if let ArrayIrType::Array(r#type) = atom.r#type().as_ref() {
                    checked_array(r#type)?;
                    storage.push((ValueId::new(RegionId::new(region_index), AtomId::new(atom_index)), r#type.clone()));
                }
            }
            for (instruction_index, instruction) in region.instructions().iter().enumerate() {
                if let KernelOperation::Portable(ArrayIrOperation::Array(operation)) = instruction.operation() {
                    if let ArrayOperation::Constant(operation) = operation {
                        work = checked_work(
                            work,
                            ArrayAddressing::new(operation.value().r#type().into_owned())?.element_count(),
                            options.maximum_instructions(),
                        )?;
                    }
                    if instruction.outputs().len() != 1 {
                        return Err(Error::Unsupported {
                            operation: operation.name(),
                            reason: "array operation requires exactly one result".to_owned(),
                        });
                    }
                    let inputs = instruction
                        .inputs()
                        .iter()
                        .map(|input| {
                            let r#type = region.atoms()[input.index()].r#type();
                            let ArrayIrType::Array(r#type) = r#type.as_ref() else { unreachable!() };
                            r#type.clone()
                        })
                        .collect::<Vec<_>>();
                    let output = region.atoms()[instruction.outputs()[0].index()].r#type();
                    let ArrayIrType::Array(output) = output.as_ref() else { unreachable!() };
                    let transport = match (options.mma(), operation) {
                        (Some(Mma::Wgmma), ArrayOperation::Dot(operation)) => {
                            Some(super::mma::requirements(operation, &inputs, output)?)
                        }
                        (Some(Mma::Wgmma), ArrayOperation::ScaledDot(operation)) => {
                            Some(super::mma::scaled_requirements(operation, &inputs, output)?)
                        }
                        _ => None,
                    };
                    if let Some(types) = transport {
                        instruction_scratch
                            .push((InstructionId::new(RegionId::new(region_index), instruction_index), types.to_vec()));
                    } else {
                        super::arrays::validate(operation, &inputs, output)?;
                    }
                }
                let is_tmem = matches!(instruction.operation(), KernelOperation::Extension(extension)
                    if matches!(extension.clone().into(), GpuOperation::Tmem(_)));
                if !is_tmem
                    && instruction.inputs().iter().any(|input| {
                        let owner = ValueId::new(RegionId::new(region_index), *input);
                        tmem_allocations.contains_key(&owner) || tmem_pending.contains_key(&owner)
                    })
                {
                    return Err(Error::Unsupported {
                        operation: instruction.operation().name(),
                        reason: "tensor-memory allocations and completion tokens require explicit TMEM operations"
                            .to_owned(),
                    });
                }
                if let KernelOperation::Extension(extension) = instruction.operation() {
                    match extension.clone().into() {
                        GpuOperation::Tmem(operation) => {
                            super::tmem::admit(target)?;
                            if RegionId::new(region_index) != body.entry_region_ref().id() {
                                return Err(Error::Unsupported {
                                    operation: operation.name(),
                                    reason: "tensor-memory operations must remain in the kernel entry region"
                                        .to_owned(),
                                });
                            }
                            uses_tmem = true;
                            let owner =
                                |index: usize| ValueId::new(RegionId::new(region_index), instruction.inputs()[index]);
                            match operation {
                                TmemOperation::Allocate { .. } | TmemOperation::AllocateScales { .. } => {
                                    let columns = match operation {
                                        TmemOperation::Allocate { rows, columns } => {
                                            if u32::from(rows) != 128 * target.blocks_per_cluster() {
                                                return Err(Error::Unsupported {
                                                    operation: operation.name(),
                                                    reason: "tensor-memory accumulator rows must equal 128 times the blocks per cluster".to_owned(),
                                                });
                                            }
                                            usize::from(columns)
                                        }
                                        TmemOperation::AllocateScales { rows, blocks, data_type } => {
                                            if data_type == DataType::F8E4M3FN {
                                                instruction_scratch.push((
                                                    InstructionId::new(RegionId::new(region_index), instruction_index),
                                                    vec![ArrayType::new_static(
                                                        DataType::U8,
                                                        [usize::from(rows), usize::from(blocks)],
                                                    )],
                                                ));
                                            }
                                            super::tmem::scale_columns(usize::from(rows), usize::from(blocks))?
                                        }
                                        _ => unreachable!(),
                                    };
                                    let result = ValueId::new(RegionId::new(region_index), instruction.outputs()[0]);
                                    if tmem_allocations.values().sum::<usize>() + columns > 512 {
                                        return Err(Error::Unsupported {
                                            operation: operation.name(),
                                            reason: "live tensor-memory allocations exceed 512 columns per CTA"
                                                .to_owned(),
                                        });
                                    }
                                    tmem_allocations.insert(result, columns);
                                    storage.push((result, ArrayType::scalar(DataType::U32)));
                                }
                                TmemOperation::Mma { .. }
                                | TmemOperation::MmaBlockScaled { .. }
                                | TmemOperation::MmaNvfp4 { .. } => {
                                    let scaled = !matches!(operation, TmemOperation::Mma { .. });
                                    if !tmem_allocations.contains_key(&owner(if scaled { 4 } else { 2 })) {
                                        return Err(Error::Unsupported {
                                            operation: operation.name(),
                                            reason: "matrix destination must be a live direct tensor-memory allocation"
                                                .to_owned(),
                                        });
                                    }
                                    if scaled
                                        && [2, 3].iter().any(|index| !tmem_allocations.contains_key(&owner(*index)))
                                    {
                                        return Err(Error::Unsupported {
                                            operation: operation.name(),
                                            reason: "matrix scales must be live direct tensor-memory allocations"
                                                .to_owned(),
                                        });
                                    }
                                    let result = ValueId::new(RegionId::new(region_index), instruction.outputs()[0]);
                                    let inputs = instruction.inputs()[..2]
                                        .iter()
                                        .map(|input| {
                                            let r#type = region.atoms()[input.index()].r#type();
                                            let ArrayIrType::Array(r#type) = r#type.as_ref() else { unreachable!() };
                                            r#type.clone()
                                        })
                                        .collect::<Vec<_>>();
                                    let types = super::tmem::requirements(&inputs)?;
                                    instruction_scratch.push((
                                        InstructionId::new(RegionId::new(region_index), instruction_index),
                                        types.to_vec(),
                                    ));
                                    storage.push((result, ArrayType::scalar(DataType::U64)));
                                    tmem_pending.insert(result, false);
                                }
                                TmemOperation::CopyScales => {
                                    if !tmem_allocations.contains_key(&owner(1)) {
                                        return Err(Error::Unsupported {
                                            operation: operation.name(),
                                            reason: "scale destination must be a live direct tensor-memory allocation"
                                                .to_owned(),
                                        });
                                    }
                                    let input_type = region.atoms()[instruction.inputs()[0].index()].r#type();
                                    let ArrayIrType::Array(input_type) = input_type.as_ref() else { unreachable!() };
                                    let transport = super::tmem::scale_requirements(input_type)?;
                                    instruction_scratch.push((
                                        InstructionId::new(RegionId::new(region_index), instruction_index),
                                        vec![transport],
                                    ));
                                    let result = ValueId::new(RegionId::new(region_index), instruction.outputs()[0]);
                                    storage.push((result, ArrayType::scalar(DataType::U64)));
                                    tmem_pending.insert(result, false);
                                }
                                TmemOperation::Commit => {
                                    if tmem_pending.get(&owner(0)) != Some(&false) {
                                        return Err(Error::Unsupported {
                                            operation: operation.name(),
                                            reason:
                                                "tensor-memory completion must be committed exactly once after issue"
                                                    .to_owned(),
                                        });
                                    }
                                    tmem_pending.insert(owner(0), true);
                                }
                                TmemOperation::Wait => {
                                    if tmem_pending.remove(&owner(0)) != Some(true) {
                                        return Err(Error::Unsupported {
                                            operation: operation.name(),
                                            reason: "tensor-memory completion must be committed before waiting"
                                                .to_owned(),
                                        });
                                    }
                                }
                                TmemOperation::Load | TmemOperation::Release => {
                                    if !tmem_allocations.contains_key(&owner(0)) {
                                        return Err(Error::Unsupported {
                                            operation: operation.name(),
                                            reason: "tensor-memory access requires a live direct allocation".to_owned(),
                                        });
                                    }
                                    if operation == TmemOperation::Release {
                                        tmem_allocations.remove(&owner(0));
                                    }
                                }
                            }
                        }
                        GpuOperation::Wgmma => {
                            let inputs = instruction
                                .inputs()
                                .iter()
                                .map(|input| {
                                    let r#type = region.atoms()[input.index()].r#type();
                                    let ArrayIrType::Array(r#type) = r#type.as_ref() else { unreachable!() };
                                    r#type.clone()
                                })
                                .collect::<Vec<_>>();
                            let output = region.atoms()[instruction.outputs()[0].index()].r#type();
                            let ArrayIrType::Array(output) = output.as_ref() else { unreachable!() };
                            let types = super::mma::requirements(
                                &ryft_core::DotOperation::matmul().with_accumulation_type(DataType::F32),
                                &inputs,
                                output,
                            )?;
                            instruction_scratch.push((
                                InstructionId::new(RegionId::new(region_index), instruction_index),
                                types.to_vec(),
                            ));
                        }
                        native @ (GpuOperation::Nvfp4 { .. } | GpuOperation::Nvfp4Sparse { .. }) => {
                            if !matches!(target.compute_capability(), (12, 0) | (12, 1))
                                || target.threads_per_block() != 32
                            {
                                return Err(Error::Unsupported {
                                    operation: native.name(),
                                    reason: "native block-scaled warp MMA requires compute capability 12.0 or 12.1 and exactly 32 threads per block".to_owned(),
                                });
                            }
                            let left = region.atoms()[instruction.inputs()[0].index()].r#type();
                            let output = region.atoms()[instruction.outputs()[0].index()].r#type();
                            let (ArrayIrType::Array(left), ArrayIrType::Array(output)) =
                                (left.as_ref(), output.as_ref())
                            else {
                                unreachable!()
                            };
                            let left = left.static_shape().unwrap();
                            let output = output.static_shape().unwrap();
                            if output.dimensions()[0] == 0
                                || output.dimensions()[0] % 16 != 0
                                || output.dimensions()[1] == 0
                                || output.dimensions()[1] % 8 != 0
                                || left.dimensions()[1] % 32 != 0
                            {
                                return Err(Error::Unsupported {
                                    operation: native.name(),
                                    reason: if matches!(native, GpuOperation::Nvfp4Sparse { .. }) {
                                        "native sparse NVFP4 requires positive M, N, and K divisible by 16, 8, and 128 respectively"
                                    } else {
                                        "native NVFP4 requires positive M, N, and K divisible by 16, 8, and 64 respectively"
                                    }.to_owned(),
                                });
                            }
                        }
                    }
                }
                if let KernelOperation::Scratch(scratch) = instruction.operation() {
                    if scratch.alignment() > 16 {
                        return Err(Error::Unsupported {
                            operation: scratch.name(),
                            reason: "scratch alignment above 16 bytes is not supported".to_owned(),
                        });
                    }
                    checked_array(scratch.referent())?;
                    storage.push((
                        ValueId::new(RegionId::new(region_index), instruction.outputs()[0]),
                        scratch.referent().clone(),
                    ));
                }
            }
        }
        if !tmem_allocations.is_empty() {
            return Err(Error::Unsupported {
                operation: "mosaic_gpu.tmem.release",
                reason: "every tensor-memory allocation must be explicitly released before kernel exit".to_owned(),
            });
        }
        let tma_copies = if options.tma() { plan_tma_copies(kernel, target)? } else { Vec::new() };
        if !tma_copies.is_empty() {
            // A flat memref carries two pointers, an offset, a size, and a stride. Reserve descriptor alignment
            // as well as its by-value payload before native outlining can construct the actual parameter layout.
            let argument_bytes = argument_types
                .len()
                .checked_mul(40)
                .and_then(|bytes| bytes.checked_add(63))
                .map(|bytes| bytes & !63)
                .and_then(|bytes| {
                    tma_copies.len().checked_mul(128).and_then(|descriptors| bytes.checked_add(descriptors))
                });
            if argument_bytes.is_none_or(|bytes| bytes > 4096) {
                return Err(Error::Invalid {
                    message: "tma kernel arguments exceed the conservative 4096-byte parameter limit".to_owned(),
                });
            }
            storage.extend(tma_copies.iter().map(|copy| (copy.token, ArrayType::scalar(DataType::U64))));
        }
        storage.sort_by_key(|(value, _)| *value);
        let maximum = schedule.maximum_scratch_bytes().unwrap_or(usize::MAX).min(target.maximum_shared_memory_bytes());
        let mut shared_memory_bytes = 0usize;
        for (owner, r#type) in &storage {
            let alignment = if tma_copies.iter().any(|copy| copy.destination == *owner) { 128 } else { 16 };
            let bytes = ArrayAddressing::new(r#type.clone())?.storage_byte_len();
            let aligned = bytes
                .checked_add(alignment - 1)
                .map(|bytes| bytes & !(alignment - 1))
                .ok_or(Error::SharedMemory { required: usize::MAX, maximum })?;
            shared_memory_bytes = shared_memory_bytes
                .checked_add(alignment - 16)
                .and_then(|bytes| bytes.checked_add(aligned))
                .ok_or(Error::SharedMemory { required: usize::MAX, maximum })?;
            if shared_memory_bytes > maximum {
                return Err(Error::SharedMemory { required: shared_memory_bytes, maximum });
            }
        }
        for (_, types) in &instruction_scratch {
            for r#type in types {
                let bytes = ArrayAddressing::new(r#type.clone())?.storage_byte_len();
                // Outlining may reorder shared globals; reserve the worst gap from a 16-byte-aligned predecessor.
                let aligned = bytes
                    .checked_add(255)
                    .map(|bytes| bytes & !255)
                    .ok_or(Error::SharedMemory { required: usize::MAX, maximum })?;
                shared_memory_bytes = shared_memory_bytes
                    .checked_add(256 - 16)
                    .and_then(|bytes| bytes.checked_add(aligned))
                    .ok_or(Error::SharedMemory { required: usize::MAX, maximum })?;
                if shared_memory_bytes > maximum {
                    return Err(Error::SharedMemory { required: shared_memory_bytes, maximum });
                }
            }
        }
        Ok(Self {
            storage,
            argument_types,
            parameter_slots,
            shared_memory_bytes,
            grid_extents,
            tma_copies,
            instruction_scratch,
            uses_tmem,
        })
    }
}

/// Accounts literal elements as source-construction work, including explicit constant operation payloads.
fn checked_work(work: usize, additional: usize, maximum: usize) -> Result<usize, Error> {
    work.checked_add(additional).filter(|count| *count <= maximum).ok_or_else(|| Error::Invalid {
        message: "kernel instruction and literal count exceeds the source compilation limit".to_owned(),
    })
}

/// Checks native layouts and representability without constructing MLIR.
pub(super) fn checked_array(r#type: &ArrayType) -> Result<(), Error> {
    let Some(shape) = r#type.static_shape() else {
        return Err(Error::Invalid { message: "mosaic GPU requires static array shapes".to_owned() });
    };
    if r#type.memory() != Memory::Device {
        return Err(Error::Invalid { message: "mosaic GPU requires device-resident arrays".to_owned() });
    }
    if r#type.sharding().is_some() {
        return Err(Error::Invalid { message: "mosaic GPU requires unsharded kernel-local array types".to_owned() });
    }
    if !matches!(
        r#type.data_type(),
        DataType::Boolean
            | DataType::U8
            | DataType::F8E4M3FN
            | DataType::F8E8M0FNU
            | DataType::I32
            | DataType::U32
            | DataType::I64
            | DataType::U64
            | DataType::F16
            | DataType::BF16
            | DataType::F32
            | DataType::F64
    ) {
        return Err(Error::Invalid {
            message: "mosaic GPU supports only boolean, u8, f8e4m3fn, f8e8m0fnu, i32, u32, i64, u64, f16, bf16, f32, and f64 array elements"
                .to_owned(),
        });
    }
    if shape.dimensions().iter().any(|extent| *extent > i32::MAX as usize) {
        return Err(Error::Invalid { message: "array extent exceeds the native signed index range".to_owned() });
    }
    if let Some(layout) = r#type.layout() {
        let valid = matches!(layout, Layout::Tiled(layout)
            if layout.tiles().is_empty()
                && layout.minor_to_major().iter().copied().eq((0..r#type.rank()).rev()));
        if !valid {
            return Err(Error::Invalid { message: "mosaic GPU requires untiled dense row-major arrays".to_owned() });
        }
    }
    let addressing = ArrayAddressing::new(r#type.clone())?;
    if addressing.element_count() > i32::MAX as usize || addressing.storage_byte_len() > i64::MAX as usize {
        return Err(Error::Invalid { message: "array storage exceeds native index limits".to_owned() });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::kernels::{
        AsyncCopyOperation, BlockMapping, BoundaryPolicy, Grid, GridDimension, KernelCallOperation, KernelDefinition,
        KernelParameter, ScratchOperation, WaitOperation, whole_array_parameter,
    };
    use ryft_core::{
        Array, ArrayIrOperation, ArrayIrValue, Context, ProgramBuilder, ReferenceRead, ReferenceWrite, TiledLayout,
    };

    use super::*;

    /// Scalar read/write source with one private array result and an aliased functional boundary.
    fn definition() -> KernelDefinition {
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

    /// Builds independent whole-root copies so descriptor limits can be tested without earlier alias failures.
    fn tma_definition(copies: usize) -> KernelDefinition {
        let r#type = ArrayType::new_static(DataType::F32, [64]);
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![whole_array_parameter(r#type.clone(), KernelParameterAccess::ReadOnly).unwrap()],
        )
        .unwrap();
        let scratch = ScratchOperation::new(r#type, 16).unwrap();
        KernelDefinition::trace(operation, |(references, _)| {
            let context = references[0].context();
            for _ in 0..copies {
                let destination = context.bind(scratch.clone(), vec![], &[])?.remove(0);
                let token =
                    context.bind(AsyncCopyOperation::new(), vec![], &[references[0].clone(), destination])?.remove(0);
                context.bind(WaitOperation, vec![], &[token])?;
            }
            Ok(())
        })
        .unwrap()
    }

    /// Builds explicit allocations so adapter ownership and live column limits are tested independently of MMA.
    fn tmem_definition(columns: &[u16], release: bool) -> KernelDefinition<GpuOperation> {
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![whole_array_parameter(ArrayType::scalar(DataType::F32), KernelParameterAccess::ReadOnly).unwrap()],
        )
        .unwrap();
        KernelDefinition::trace(operation, |(references, _)| {
            let context = references[0].context();
            let mut allocations = Vec::new();
            for &columns in columns {
                allocations.push(
                    context
                        .bind(
                            KernelOperation::Extension(GpuOperation::Tmem(TmemOperation::Allocate {
                                rows: 128,
                                columns,
                            })),
                            vec![],
                            &[],
                        )?
                        .remove(0),
                );
            }
            if release {
                for allocation in allocations {
                    context.bind(
                        KernelOperation::Extension(GpuOperation::Tmem(TmemOperation::Release)),
                        vec![],
                        &[allocation],
                    )?;
                }
            }
            Ok(())
        })
        .unwrap()
    }

    /// Builds one asynchronous scale transfer to exercise physical packing and the native completion contract.
    fn scale_definition(data_type: DataType, commit: bool) -> KernelDefinition<GpuOperation> {
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(ArrayType::new_static(data_type, [128, 1]), KernelParameterAccess::ReadOnly)
                    .unwrap(),
            ],
        )
        .unwrap();
        KernelDefinition::trace(operation, |(references, _)| {
            let context = references[0].context();
            let source = references[0].read()?;
            let destination = context
                .bind(
                    KernelOperation::Extension(GpuOperation::Tmem(TmemOperation::AllocateScales {
                        rows: 128,
                        blocks: 1,
                        data_type,
                    })),
                    vec![],
                    &[],
                )?
                .remove(0);
            let token = context
                .bind(
                    KernelOperation::Extension(GpuOperation::Tmem(TmemOperation::CopyScales)),
                    vec![],
                    &[source, destination.clone()],
                )?
                .remove(0);
            if commit {
                context.bind(
                    KernelOperation::Extension(GpuOperation::Tmem(TmemOperation::Commit)),
                    vec![],
                    &[token.clone()],
                )?;
            }
            context.bind(KernelOperation::Extension(GpuOperation::Tmem(TmemOperation::Wait)), vec![], &[token])?;
            context.bind(
                KernelOperation::Extension(GpuOperation::Tmem(TmemOperation::Release)),
                vec![],
                &[destination],
            )?;
            Ok(())
        })
        .unwrap()
    }

    #[test]
    fn test_plan_new() {
        let definition = definition();
        let kernel = VerifiedKernel::new(&definition, 1).unwrap();
        let plan =
            Plan::new(&kernel, &Target::new(9, 0).unwrap(), &Options::default(), &KernelSchedule::default()).unwrap();
        assert_eq!(plan.argument_types, vec![ArrayType::scalar(DataType::I32), ArrayType::scalar(DataType::I32)]);
        assert_eq!(plan.parameter_slots, vec![1]);
        assert_eq!(plan.grid_extents, Vec::<usize>::new());
        assert_eq!(plan.storage.len(), 1);
        assert_eq!(plan.shared_memory_bytes, 16);
        assert!(matches!(
            Plan::new(
                &kernel,
                &Target::new(9, 0).unwrap(),
                &Options::default(),
                &KernelSchedule::default().with_maximum_scratch_bytes(15)
            ),
            Err(Error::SharedMemory { required: 16, maximum: 15 })
        ));
        assert!(matches!(Plan::new(&kernel, &Target::new(9, 0).unwrap(),
            &Options::default().with_maximum_instructions(1), &KernelSchedule::default()),
            Err(Error::Invalid { message }) if message == "kernel instruction and literal count exceeds the source compilation limit"));
    }

    #[test]
    fn test_plan_new_tma_resources() {
        let definition = tma_definition(1);
        let kernel = VerifiedKernel::new(&definition, 1).unwrap();
        let options = Options::default().with_tma(true);
        let plan = Plan::new(&kernel, &Target::new(12, 1).unwrap(), &options, &KernelSchedule::default()).unwrap();
        assert_eq!(plan.tma_copies.len(), 1);
        assert_eq!(plan.shared_memory_bytes, 384);
        assert_eq!(plan.storage.len(), 2);
        assert_eq!(
            plan.storage.iter().find(|(owner, _)| *owner == plan.tma_copies[0].token).unwrap().1,
            ArrayType::scalar(DataType::U64),
        );
        assert!(matches!(
            Plan::new(
                &kernel,
                &Target::new(12, 1).unwrap(),
                &options,
                &KernelSchedule::default().with_maximum_scratch_bytes(383)
            ),
            Err(Error::SharedMemory { required: 384, maximum: 383 }),
        ));
    }

    #[test]
    fn test_plan_new_tma_parameter_limit() {
        let definition = tma_definition(32);
        let kernel = VerifiedKernel::new(&definition, 1).unwrap();
        assert!(matches!(
            Plan::new(&kernel, &Target::new(12, 1).unwrap(), &Options::default().with_tma(true),
                &KernelSchedule::default()),
            Err(Error::Invalid { message })
                if message == "tma kernel arguments exceed the conservative 4096-byte parameter limit",
        ));
    }

    #[test]
    fn test_plan_new_tmem_resources() {
        let definition = tmem_definition(&[256, 256], true);
        let kernel = VerifiedKernel::new(&definition, 1).unwrap();
        let target = Target::new(10, 0).unwrap().with_threads_per_block(128).unwrap();
        let plan = Plan::new(&kernel, &target, &Options::default(), &KernelSchedule::default()).unwrap();
        assert!(plan.uses_tmem);
        assert_eq!(plan.shared_memory_bytes, 32);
        assert_eq!(
            plan.storage.iter().map(|(_, value)| value.clone()).collect::<Vec<_>>(),
            vec![ArrayType::scalar(DataType::U32), ArrayType::scalar(DataType::U32)]
        );
        assert!(matches!(
            Plan::new(&kernel, &target, &Options::default(), &KernelSchedule::default().with_maximum_scratch_bytes(31)),
            Err(Error::SharedMemory { required: 32, maximum: 31 }),
        ));
        let definition = tmem_definition(&[512, 32], true);
        let kernel = VerifiedKernel::new(&definition, 1).unwrap();
        assert!(matches!(
            Plan::new(&kernel, &target, &Options::default(), &KernelSchedule::default()),
            Err(Error::Unsupported { operation: "mosaic_gpu.tmem.allocate", reason })
                if reason == "live tensor-memory allocations exceed 512 columns per CTA",
        ));
    }

    #[test]
    fn test_plan_new_tmem_cluster_membership() {
        let definition = tmem_definition(&[32], true);
        let kernel = VerifiedKernel::new(&definition, 1).unwrap();
        let target =
            Target::new(10, 0).unwrap().with_threads_per_block(128).unwrap().with_blocks_per_cluster(2).unwrap();
        assert!(matches!(
            Plan::new(&kernel, &target, &Options::default(), &KernelSchedule::default()),
            Err(Error::Unsupported { operation: "mosaic_gpu.tmem.allocate", reason })
                if reason == "tensor-memory accumulator rows must equal 128 times the blocks per cluster",
        ));
    }

    #[test]
    fn test_plan_new_tmem_scale_copy() {
        let definition = scale_definition(DataType::F8E8M0FNU, true);
        let kernel = VerifiedKernel::new(&definition, 1).unwrap();
        let target = Target::new(10, 0).unwrap().with_threads_per_block(128).unwrap();
        let plan = Plan::new(&kernel, &target, &Options::default(), &KernelSchedule::default()).unwrap();
        assert!(plan.uses_tmem);
        assert_eq!(plan.shared_memory_bytes, 912);
        assert_eq!(
            plan.instruction_scratch.iter().map(|(_, types)| types.clone()).collect::<Vec<_>>(),
            vec![vec![ArrayType::new_static(DataType::U8, [512])]]
        );
        let definition = scale_definition(DataType::F8E8M0FNU, false);
        let kernel = VerifiedKernel::new(&definition, 1).unwrap();
        assert!(matches!(
            Plan::new(&kernel, &target, &Options::default(), &KernelSchedule::default()),
            Err(Error::Unsupported { operation: "mosaic_gpu.tmem.wait", reason })
                if reason == "tensor-memory completion must be committed before waiting",
        ));
    }

    #[test]
    fn test_plan_new_tmem_signed_scale_storage() {
        let definition = scale_definition(DataType::F8E4M3FN, true);
        let kernel = VerifiedKernel::new(&definition, 1).unwrap();
        let target = Target::new(10, 0).unwrap().with_threads_per_block(128).unwrap();
        let plan = Plan::new(&kernel, &target, &Options::default(), &KernelSchedule::default()).unwrap();
        assert_eq!(plan.shared_memory_bytes, 1408);
        assert_eq!(
            plan.instruction_scratch.iter().map(|(_, types)| types.clone()).collect::<Vec<_>>(),
            vec![vec![ArrayType::new_static(DataType::U8, [128, 1])], vec![ArrayType::new_static(DataType::U8, [512])],],
        );
        assert!(matches!(
            Plan::new(
                &kernel,
                &target,
                &Options::default(),
                &KernelSchedule::default().with_maximum_scratch_bytes(1407),
            ),
            Err(Error::SharedMemory { required: 1408, maximum: 1407 }),
        ));
    }

    #[test]
    fn test_plan_new_tmem_requires_release() {
        let definition = tmem_definition(&[32], false);
        let kernel = VerifiedKernel::new(&definition, 1).unwrap();
        let target = Target::new(10, 0).unwrap().with_threads_per_block(128).unwrap();
        assert!(matches!(
            Plan::new(&kernel, &target, &Options::default(), &KernelSchedule::default()),
            Err(Error::Unsupported { operation: "mosaic_gpu.tmem.release", reason })
                if reason == "every tensor-memory allocation must be explicitly released before kernel exit",
        ));
    }

    #[test]
    fn test_plan_new_sequential_and_empty_grids() {
        let grid = Grid::new(vec![
            GridDimension::new(Dimension::Static(2), GridExecution::Sequential),
            GridDimension::new(Dimension::Static(0), GridExecution::Parallel),
        ])
        .unwrap();
        let definition: KernelDefinition =
            KernelDefinition::trace(KernelCallOperation::new(grid, vec![]).unwrap(), |_| Ok(())).unwrap();
        let kernel = VerifiedKernel::new(&definition, 1).unwrap();
        let plan =
            Plan::new(&kernel, &Target::new(9, 0).unwrap(), &Options::default(), &KernelSchedule::default()).unwrap();
        assert_eq!(plan.grid_extents, vec![2, 0]);
        assert_eq!(plan.shared_memory_bytes, 0);
    }

    #[test]
    fn test_plan_new_counts_each_literal_element() {
        for (operation_constant, required) in [(false, 5), (true, 6)] {
            let parameter = ryft_core::kernels::whole_array_parameter(
                ArrayType::new_static(DataType::I32, [4]),
                KernelParameterAccess::WriteOnly,
            )
            .unwrap();
            let definition: KernelDefinition = KernelDefinition::trace(
                KernelCallOperation::new(Grid::new(vec![]).unwrap(), vec![parameter]).unwrap(),
                |(references, _)| {
                    let context = references[0].context();
                    let constant = Array::vector(vec![1i32, 2, 3, 4])?;
                    let value = if operation_constant {
                        context
                            .bind(
                                ArrayIrOperation::from(ArrayOperation::Constant(ryft_core::ConstantOperation::new(
                                    constant,
                                ))),
                                vec![],
                                &[],
                            )?
                            .remove(0)
                    } else {
                        context.lift(ArrayIrValue::Array(constant))?
                    };
                    references[0].write(&value)?;
                    Ok(())
                },
            )
            .unwrap();
            let kernel = VerifiedKernel::new(&definition, 1).unwrap();
            let target = Target::new(9, 0).unwrap();
            let schedule = KernelSchedule::default();
            Plan::new(&kernel, &target, &Options::default().with_maximum_instructions(required), &schedule).unwrap();
            assert!(matches!(Plan::new(&kernel, &target,
                &Options::default().with_maximum_instructions(required - 1), &schedule),
                Err(Error::Invalid { message })
                    if message == "kernel instruction and literal count exceeds the source compilation limit"));
        }
    }

    #[test]
    fn test_checked_array() {
        let matrix = ArrayType::new_static(DataType::F32, [2, 3]);
        checked_array(&matrix).unwrap();
        let column_major = matrix.clone().with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], vec![])));
        assert!(matches!(checked_array(&column_major), Err(Error::Invalid { message })
            if message == "mosaic GPU requires untiled dense row-major arrays"));
        assert!(matches!(checked_array(&matrix.with_memory(Memory::Host { pinned: false })),
            Err(Error::Invalid { message }) if message == "mosaic GPU requires device-resident arrays"));
        assert!(matches!(checked_array(&ArrayType::new_static(DataType::F32, [i32::MAX as usize, 2])),
            Err(Error::Invalid { message }) if message == "array storage exceeds native index limits"));
    }
}
