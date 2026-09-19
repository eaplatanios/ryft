//! Lowering of bound-proven canonical dimension arithmetic and block mappings.

use ryft_core::kernels::BlockMapping;
use ryft_core::{ArrayIrOperation, ArrayIrValue, Atom, DimensionOperation, DimensionValue, Operation};
use ryft_mlir::DetachedBlock;
use ryft_mlir::dialects::arith;

use crate::kernels::gpu::Error;
use crate::kernels::gpu::lowering::{KernelValue, Lowering, append};

impl<'c, 't> Lowering<'c, 't> {
    /// Lowers arithmetic whose canonical type bounds prove that no runtime assertion is necessary. Potentially failing arithmetic are rejected until native device failure propagation is supported.
    pub(super) fn dimension(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        operation: &DimensionOperation<DimensionValue>,
        inputs: &[KernelValue<'c, 't>],
    ) -> Result<Vec<KernelValue<'c, 't>>, Error> {
        if !operation.effects().is_pure() {
            return Err(Error::Unsupported {
                operation: operation.name(),
                reason: "runtime dimension assertions require native failure propagation".to_owned(),
            });
        }
        let result = match operation {
            DimensionOperation::Constant(operation) => self.index(block, operation.value().extent())?,
            DimensionOperation::Add(_) => append(block, arith::addi(inputs[0], inputs[1], self.location)?)?,
            DimensionOperation::Sub(_) => append(block, arith::subi(inputs[0], inputs[1], self.location)?)?,
            DimensionOperation::SaturatingSub(_) => {
                let maximum = append(block, arith::maxui(inputs[0], inputs[1], self.location)?)?;
                append(block, arith::subi(maximum, inputs[1], self.location)?)?
            }
            DimensionOperation::Mul(_) => append(block, arith::muli(inputs[0], inputs[1], self.location)?)?,
            DimensionOperation::Div(_) => append(block, arith::divui(inputs[0], inputs[1], self.location)?)?,
            DimensionOperation::Rem(_) => append(block, arith::remui(inputs[0], inputs[1], self.location)?)?,
            DimensionOperation::Min(_) => append(block, arith::minui(inputs[0], inputs[1], self.location)?)?,
            DimensionOperation::Max(_) => append(block, arith::maxui(inputs[0], inputs[1], self.location)?)?,
            DimensionOperation::Pow(_) => {
                return Err(Error::Unsupported {
                    operation: operation.name(),
                    reason: "dimension operation is outside the native arithmetic baseline".to_owned(),
                });
            }
        };
        Ok(vec![result])
    }

    /// Lowers the canonical flat mapping program to start indices, preserving instruction order and constant values.
    /// The caller supplies one dimension binding per declared mapping input; scalar prefetch must be specialized first.
    pub(super) fn mapping(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        mapping: &BlockMapping,
        coordinates: &[KernelValue<'c, 't>],
    ) -> Result<Vec<KernelValue<'c, 't>>, Error> {
        let program = mapping.program();
        if program.input_ids().len() != coordinates.len() {
            return Err(Error::Invalid {
                message: "block mapping coordinate count does not match its signature".to_owned(),
            });
        }
        let mut values = vec![None; program.atoms().len()];
        for (position, atom) in program.atoms().iter().enumerate() {
            if let Atom::Constant(ArrayIrValue::Dimension(value)) = atom {
                values[position] = Some(self.index(block, value.extent())?);
            }
        }
        for (input, value) in program.input_ids().iter().zip(coordinates) {
            values[input.index()] = Some(*value);
        }
        for instruction in program.instructions() {
            let ArrayIrOperation::Dimension(operation) = instruction.operation() else {
                unreachable!("BlockMapping construction admits only dimension operations");
            };
            let inputs = instruction.inputs().iter().map(|input| values[input.index()].unwrap()).collect::<Vec<_>>();
            let outputs = self.dimension(block, operation, &inputs)?;
            for (output, value) in instruction.outputs().iter().zip(outputs) {
                values[output.index()] = Some(value);
            }
        }
        Ok(program.output_ids().iter().map(|output| values[output.index()].unwrap()).collect())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;
    use ryft_core::kernels::BoundaryPolicy;
    use ryft_core::{
        Array, ArrayIrType, ConstantOperation, DimensionAddOperation, DimensionBounds, DimensionSaturatingSubOperation,
        DimensionSubOperation, DimensionType, Placeholder, ProgramBuilder,
    };
    use ryft_mlir::{Block, Context, Operation as MlirOperation, Value};

    use super::*;

    #[test]
    fn test_lowering_dimension() {
        let context = Context::new();
        let location = context.unknown_location();
        let mut block = context.block(&[(context.index_type(), location); 2]);
        let left = block.argument(0).unwrap().as_ref();
        let right = block.argument(1).unwrap().as_ref();
        let lowering = Lowering {
            context: &context,
            location,
            thread: left,
            threads: right,
            storage: HashMap::new(),
            tma_descriptors: HashMap::new(),
            instruction_scratch: HashMap::new(),
            mma_stages: 1,
            cluster_rank: None,
            synchronization: None,
            current_instruction: None,
            next_barrier: 0,
        };
        let left_type = DimensionType::new("left", DimensionBounds::new(0, Some(8)).unwrap());
        let right_type = DimensionType::new("right", DimensionBounds::new(0, Some(4)).unwrap());
        let add = DimensionOperation::Add(DimensionAddOperation::new(&left_type, &right_type).unwrap());
        assert_eq!(lowering.dimension(&mut block, &add, &[left, right]).unwrap().len(), 1);
        let sub =
            DimensionOperation::SaturatingSub(DimensionSaturatingSubOperation::new(&left_type, &right_type).unwrap());
        assert_eq!(lowering.dimension(&mut block, &sub, &[left, right]).unwrap().len(), 1);
        let constant = DimensionOperation::Constant(ConstantOperation::new(DimensionValue::constant(3).unwrap()));
        assert_eq!(lowering.dimension(&mut block, &constant, &[]).unwrap().len(), 1);
        let names = block
            .operations()
            .unwrap()
            .map(|operation| operation.unwrap().name().as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        assert_eq!(names, ["arith.addi", "arith.maxui", "arith.subi", "arith.constant"]);
        let subtraction = DimensionOperation::Sub(DimensionSubOperation::new(&left_type, &right_type).unwrap());
        assert!(matches!(lowering.dimension(&mut block, &subtraction, &[left, right]),
            Err(Error::Unsupported { operation: "dimension_sub", reason })
                if reason == "runtime dimension assertions require native failure propagation"));
        assert_eq!(block.operations().unwrap().count(), 4);
    }

    #[test]
    fn test_lowering_mapping() {
        let context = Context::new();
        let location = context.unknown_location();
        let mut block = context.block(&[(context.index_type(), location)]);
        let coordinate = block.argument(0).unwrap().as_ref();
        let lowering = Lowering {
            context: &context,
            location,
            thread: coordinate,
            threads: coordinate,
            storage: HashMap::new(),
            tma_descriptors: HashMap::new(),
            instruction_scratch: HashMap::new(),
            mma_stages: 1,
            cluster_rank: None,
            synchronization: None,
            current_instruction: None,
            next_barrier: 0,
        };
        let r#type = DimensionType::new("coordinate", DimensionBounds::new(0, Some(4)).unwrap());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayIrType::Dimension(r#type));
        let zero = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(0).unwrap()));
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![input, zero],
                vec![Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let mapping = BlockMapping::new(program, vec![1, 1], BoundaryPolicy::InBounds).unwrap();
        let values = lowering.mapping(&mut block, &mapping, &[coordinate]).unwrap();
        assert_eq!(values.len(), 2);
        assert_eq!(values[0], coordinate);
        assert!(matches!(lowering.mapping(&mut block, &mapping, &[]), Err(Error::Invalid { message })
            if message == "block mapping coordinate count does not match its signature"));
    }
}
