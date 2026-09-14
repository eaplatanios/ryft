//! Checked native ABI and conservative CTA-local storage accounting before MLIR construction.

use ryft_core::kernels::{GridExecution, KernelOperation, KernelParameterAccess, KernelSchedule, VerifiedKernel};
use ryft_core::{
    ArrayAddressing, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, AtomId, DataType,
    Dimension, Layout, Memory, Operation, RegionId, Typed, ValueId,
};

use crate::kernels::gpu::{Error, Options, Target};

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
}

impl Plan {
    /// Checks source work, native integer limits, ABI layout, and the complete private storage footprint.
    pub(super) fn new(
        kernel: &VerifiedKernel<'_>,
        target: &Target,
        options: &Options,
        schedule: &KernelSchedule,
    ) -> Result<Self, Error> {
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
            .try_fold(1usize, |size, (extent, _)| size.checked_mul(*extent))
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
            for instruction in region.instructions() {
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
                    super::arrays::validate(operation, &inputs, output)?;
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
        storage.sort_by_key(|(value, _)| *value);
        let maximum = schedule.maximum_scratch_bytes().unwrap_or(usize::MAX).min(target.maximum_shared_memory_bytes());
        let mut shared_memory_bytes = 0usize;
        for (_, r#type) in &storage {
            let bytes = ArrayAddressing::new(r#type.clone())?.storage_byte_len();
            let aligned = bytes
                .checked_add(15)
                .map(|bytes| bytes & !15)
                .ok_or(Error::SharedMemory { required: usize::MAX, maximum })?;
            shared_memory_bytes = shared_memory_bytes
                .checked_add(aligned)
                .ok_or(Error::SharedMemory { required: usize::MAX, maximum })?;
            if shared_memory_bytes > maximum {
                return Err(Error::SharedMemory { required: shared_memory_bytes, maximum });
            }
        }
        Ok(Self { storage, argument_types, parameter_slots, shared_memory_bytes, grid_extents })
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
            | DataType::I32
            | DataType::U32
            | DataType::I64
            | DataType::U64
            | DataType::F32
            | DataType::F64
    ) {
        return Err(Error::Invalid {
            message: "mosaic GPU supports only boolean, i32, u32, i64, u64, f32, and f64 array elements".to_owned(),
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
        BlockMapping, BoundaryPolicy, Grid, GridDimension, KernelCallOperation, KernelDefinition, KernelParameter,
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
    fn test_plan_new_sequential_and_empty_grids() {
        let grid = Grid::new(vec![
            GridDimension::new(Dimension::Static(2), GridExecution::Sequential),
            GridDimension::new(Dimension::Static(0), GridExecution::Parallel),
        ])
        .unwrap();
        let definition = KernelDefinition::trace(KernelCallOperation::new(grid, vec![]).unwrap(), |_| Ok(())).unwrap();
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
            let definition = KernelDefinition::trace(
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
