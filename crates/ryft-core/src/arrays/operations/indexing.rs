//! Deferred eager gather and scatter kernels and their array-family integration tests.

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::arrays::addressing::ArrayAddressing;
use crate::arrays::arrays::Array;
use crate::arrays::elements::{ArrayElement, NumericArrayElement, i1, i2, i4, u1, u2, u4};
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::data::DataType;
use crate::contexts::EagerContext;
use crate::macros::dispatch_on_array_element_type;
use crate::operations::{
    Gather, GatherOperation, GatherScatterMode, Scatter, ScatterOperation, ScatterReductionKind, Zero,
};
use crate::programs::{ProgramError, Typed};

impl Array {
    /// Decodes the logical integer element at `index` as the signed representation used by reference indexing
    /// kernels. Unsigned `u64` values narrow with Rust's two's-complement `as i64` semantics. The type-level validation
    /// performed by every caller rules out non-integer element types and invalid indices.
    fn index_value(&self, addressing: &ArrayAddressing, index: &[usize]) -> i64 {
        let bytes = &self.storage_bytes()[addressing.byte_range_unchecked(index)];
        match self.r#type().data_type() {
            DataType::I1 => i64::from(i1::decode(bytes).value()),
            DataType::I2 => i64::from(i2::decode(bytes).value()),
            DataType::I4 => i64::from(i4::decode(bytes).value()),
            DataType::I8 => i64::from(i8::decode(bytes)),
            DataType::I16 => i64::from(i16::decode(bytes)),
            DataType::I32 => i64::from(i32::decode(bytes)),
            DataType::I64 => i64::decode(bytes),
            DataType::U1 => i64::from(u1::decode(bytes).value()),
            DataType::U2 => i64::from(u2::decode(bytes).value()),
            DataType::U4 => i64::from(u4::decode(bytes).value()),
            DataType::U8 => i64::from(u8::decode(bytes)),
            DataType::U16 => i64::from(u16::decode(bytes)),
            DataType::U32 => i64::from(u32::decode(bytes)),
            DataType::U64 => u64::decode(bytes) as i64,
            data_type => unreachable!("cannot use an array of element data type `{data_type}` as indices"),
        }
    }
}

impl Gather for Array {
    fn gather(&self, indices: &Self, operation: &GatherOperation) -> Result<Self, ProgramError> {
        let output_type = self.r#type().gather(indices.r#type().as_ref(), operation)?;
        let dimensions = operation.dimensions();
        let slice_sizes = operation.slice_sizes();
        let operand_shape = self.r#type().static_shape().unwrap();
        let indices_shape = indices.r#type().static_shape().unwrap();
        let operand_rank = operand_shape.rank();
        let indices_rank = indices_shape.rank();
        let output_rank = output_type.rank();
        let index_vector_dimension = indices_rank - 1;
        let index_vector_extent = indices_shape[index_vector_dimension];

        // Classify operand axes (window axes carry the slice; collapsed/batching do not) and output axes (offset
        // positions carry the window, the rest carry the indices' batch coordinates).
        let collapsed: BTreeSet<usize> = dimensions.collapsed_slice_dimensions().iter().copied().collect();
        let batching: BTreeSet<usize> = dimensions.operand_batching_dimensions().iter().copied().collect();
        let operand_window_axes: Vec<usize> =
            (0..operand_rank).filter(|axis| !collapsed.contains(axis) && !batching.contains(axis)).collect();
        let offset_positions: BTreeSet<usize> = dimensions.offset_dimensions().iter().copied().collect();
        let batch_output_positions: Vec<usize> =
            (0..output_rank).filter(|position| !offset_positions.contains(position)).collect();
        let indices_batch_axes: Vec<usize> = (0..indices_rank).filter(|axis| *axis != index_vector_dimension).collect();

        // Only `FillOrDrop` needs an out-of-bounds fill element. Construct it through the ordinary array capability so
        // this kernel does not assume an all-zero encoding; the other modes therefore also support element formats
        // such as F8E8M0FNU that cannot represent zero at all.
        let dropped_fill = if operation.mode() == GatherScatterMode::FillOrDrop {
            let value = EagerContext::<Array>::new().zero(&ArrayType::scalar(output_type.data_type()))?;
            let addressing = ArrayAddressing::new(value.r#type().into_owned())?;
            Some((value, addressing))
        } else {
            None
        };
        let input_addressing = ArrayAddressing::new(self.r#type().into_owned())?;
        let indices_addressing = ArrayAddressing::new(indices.r#type().into_owned())?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let mut bytes = vec![0; output_addressing.storage_byte_len()];
        let mut output_index = vec![0usize; output_rank];
        let mut indices_index = vec![0usize; indices_rank];
        let mut starts = vec![0i64; index_vector_extent];
        let mut operand_index = vec![0i64; operand_rank];
        let mut operand_storage_index = vec![0usize; operand_rank];
        for output_element in 0..output_addressing.element_count() {
            // Place the output's batch coordinates into the indices multi-index and read this query's start vector.
            indices_index.fill(0);
            for (position, &output_position) in batch_output_positions.iter().enumerate() {
                indices_index[indices_batch_axes[position]] = output_index[output_position];
            }
            for (component, start) in starts.iter_mut().enumerate() {
                indices_index[index_vector_dimension] = component;
                *start = indices.index_value(&indices_addressing, &indices_index);
            }
            // Assemble the operand multi-index: window offsets, then batching coordinates, then start offsets.
            operand_index.fill(0);
            for (window, &operand_axis) in operand_window_axes.iter().enumerate() {
                operand_index[operand_axis] = output_index[dimensions.offset_dimensions()[window]] as i64;
            }
            for (batch, &operand_axis) in dimensions.operand_batching_dimensions().iter().enumerate() {
                operand_index[operand_axis] =
                    indices_index[dimensions.start_indices_batching_dimensions()[batch]] as i64;
            }
            let mut dropped = false;
            for (component, &operand_axis) in dimensions.start_index_map().iter().enumerate() {
                let raw = starts[component];
                let maximum = (operand_shape[operand_axis] - slice_sizes[operand_axis]) as i64;
                match operation.mode() {
                    GatherScatterMode::FillOrDrop => {
                        if raw < 0 || raw > maximum {
                            dropped = true;
                        }
                        operand_index[operand_axis] += raw;
                    }
                    GatherScatterMode::PromiseInBounds | GatherScatterMode::Clip => {
                        operand_index[operand_axis] += raw.clamp(0, maximum)
                    }
                }
            }
            let source = if dropped {
                let (value, addressing) = dropped_fill.as_ref().unwrap();
                &value.storage_bytes()[addressing.byte_range_for_flat_index(0)]
            } else {
                for axis in 0..operand_rank {
                    operand_storage_index[axis] = operand_index[axis] as usize;
                }
                &self.storage_bytes()[input_addressing.byte_range_unchecked(&operand_storage_index)]
            };
            bytes[output_addressing.byte_range_for_flat_index(output_element)].copy_from_slice(source);
            output_addressing.advance_index(&mut output_index);
        }
        Ok(Self::new_unchecked(output_type, Arc::new(bytes)))
    }
}

impl Array {
    /// Applies one already-validated scatter using a byte-slice combiner, keeping index traversal independent of the
    /// selected element arithmetic. The combiner receives one mutable operand encoding and one update encoding.
    fn scatter_with_combiner(
        &self,
        indices: &Self,
        updates: &Self,
        output_type: ArrayType,
        operation: &ScatterOperation,
        combine: impl Fn(&mut [u8], &[u8]) -> Result<(), ProgramError>,
    ) -> Result<Self, ProgramError> {
        let dimensions = operation.dimensions();
        let operand_shape = self.r#type().static_shape().unwrap();
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let indices_shape = indices.r#type().static_shape().unwrap();
        let indices_addressing = ArrayAddressing::new(indices.r#type().into_owned())?;
        let updates_shape = updates.r#type().static_shape().unwrap();
        let updates_addressing = ArrayAddressing::new(updates.r#type().into_owned())?;
        let operand_rank = operand_shape.rank();
        let indices_rank = indices_shape.rank();
        let updates_rank = updates_shape.rank();
        let index_vector_dimension = indices_rank - 1;
        let index_vector_extent = indices_shape[index_vector_dimension];

        let inserted: BTreeSet<usize> = dimensions.inserted_window_dimensions().iter().copied().collect();
        let batching: BTreeSet<usize> = dimensions.operand_batching_dimensions().iter().copied().collect();
        let operand_window_axes: Vec<usize> =
            (0..operand_rank).filter(|axis| !inserted.contains(axis) && !batching.contains(axis)).collect();
        let update_window: BTreeSet<usize> = dimensions.update_window_dimensions().iter().copied().collect();
        let update_scatter_axes: Vec<usize> = (0..updates_rank).filter(|axis| !update_window.contains(axis)).collect();
        let indices_batch_axes: Vec<usize> = (0..indices_rank).filter(|axis| *axis != index_vector_dimension).collect();
        // Window size per operand axis (the update extent on window axes, 1 elsewhere), used to clamp the start so the
        // whole window stays in bounds.
        let mut operand_window_size = vec![1usize; operand_rank];
        for (window, &operand_axis) in operand_window_axes.iter().enumerate() {
            operand_window_size[operand_axis] = updates_shape[dimensions.update_window_dimensions()[window]];
        }

        let mut output = Self::new_unchecked(output_type, self.shared_storage().clone());
        let output_bytes = output.storage_bytes_mut();
        let mut update_index = vec![0usize; updates_rank];
        let mut indices_index = vec![0usize; indices_rank];
        let mut starts = vec![0i64; index_vector_extent];
        let mut operand_index = vec![0i64; operand_rank];
        let mut operand_storage_index = vec![0usize; operand_rank];
        for written in 0..updates_addressing.element_count() {
            indices_index.fill(0);
            for (position, &update_axis) in update_scatter_axes.iter().enumerate() {
                indices_index[indices_batch_axes[position]] = update_index[update_axis];
            }
            for (component, start) in starts.iter_mut().enumerate() {
                indices_index[index_vector_dimension] = component;
                *start = indices.index_value(&indices_addressing, &indices_index);
            }
            operand_index.fill(0);
            for (window, &operand_axis) in operand_window_axes.iter().enumerate() {
                operand_index[operand_axis] = update_index[dimensions.update_window_dimensions()[window]] as i64;
            }
            for (batch, &operand_axis) in dimensions.operand_batching_dimensions().iter().enumerate() {
                operand_index[operand_axis] =
                    indices_index[dimensions.scatter_indices_batching_dimensions()[batch]] as i64;
            }
            let mut dropped = false;
            for (component, &operand_axis) in dimensions.scatter_dimensions_to_operand_dimensions().iter().enumerate() {
                let raw = starts[component];
                let maximum = (operand_shape[operand_axis] - operand_window_size[operand_axis]) as i64;
                match operation.mode() {
                    GatherScatterMode::FillOrDrop => {
                        if raw < 0 || raw > maximum {
                            dropped = true;
                        }
                        operand_index[operand_axis] += raw;
                    }
                    GatherScatterMode::PromiseInBounds | GatherScatterMode::Clip => {
                        operand_index[operand_axis] += raw.clamp(0, maximum)
                    }
                }
            }
            if !dropped {
                for axis in 0..operand_rank {
                    operand_storage_index[axis] = operand_index[axis] as usize;
                }
                combine(
                    &mut output_bytes[output_addressing.byte_range_unchecked(&operand_storage_index)],
                    &updates.storage_bytes()[updates_addressing.byte_range_for_flat_index(written)],
                )?;
            }
            updates_addressing.advance_index(&mut update_index);
        }
        Ok(output)
    }
}

impl Scatter for Array {
    fn scatter(&self, indices: &Self, updates: &Self, operation: &ScatterOperation) -> Result<Self, ProgramError> {
        let output_type = self.r#type().scatter(indices.r#type().as_ref(), updates.r#type().as_ref(), operation)?;
        let data_type = output_type.data_type();
        if operation.kind() == ScatterReductionKind::Overwrite || data_type == DataType::Zero {
            return self.scatter_with_combiner(indices, updates, output_type, operation, |current, update| {
                current.copy_from_slice(update);
                Ok(())
            });
        }
        match operation.kind() {
            ScatterReductionKind::Add | ScatterReductionKind::Mul => {
                dispatch_on_array_element_type!(@numeric data_type, |Element| {
                    self.scatter_with_combiner(indices, updates, output_type, operation, |current, update| {
                        let current_value = Element::decode(current);
                        let update_value = Element::decode(update);
                        let result = if operation.kind() == ScatterReductionKind::Add {
                            <Element as NumericArrayElement>::add(current_value, update_value)?
                        } else {
                            <Element as NumericArrayElement>::mul(current_value, update_value)?
                        };
                        result.encode(current);
                        Ok(())
                    })
                })
            }
            ScatterReductionKind::Min | ScatterReductionKind::Max => {
                dispatch_on_array_element_type!(data_type, |Element| {
                    self.scatter_with_combiner(indices, updates, output_type, operation, |current, update| {
                        let current_value = Element::decode(current);
                        let update_value = Element::decode(update);
                        let result = if operation.kind() == ScatterReductionKind::Min {
                            ArrayElement::min(&current_value, &update_value)
                        } else {
                            ArrayElement::max(&current_value, &update_value)
                        };
                        result.encode(current);
                        Ok(())
                    })
                })
            }
            ScatterReductionKind::Overwrite => unreachable!("overwrite scatter returns before typed dispatch"),
        }
    }
}

#[cfg(test)]
mod tests {
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::dimensions::DimensionValue;
    use crate::arrays::ir::ArrayIrValue;
    use crate::arrays::operations::{ArrayIrOperation, ArrayOperation};
    use crate::arrays::types::dimensions::{Dimension, DimensionBounds, DimensionType, DimensionVariable, Shape};
    use crate::arrays::types::ir::ArrayIrType;
    use crate::arrays::types::layouts::{Layout, StridedLayout};
    use crate::differentiation::DifferentiableType;
    use crate::operations::{DynamicReshapeOperation, GatherDimensionNumbers, OneOperation, ScatterDimensionNumbers};
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;

    use super::*;

    #[test]
    fn test_array_ir_gather_differentiation() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(6)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent.clone())]));
        let indices_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(3), Dimension::Static(1)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.clone().into());
        let indices = builder.add_input(indices_type.clone().into());
        let operation = GatherOperation::new(GatherDimensionNumbers::new(Vec::new(), vec![0], vec![0]), vec![1]);
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Gather(operation)),
                Vec::new(),
                vec![input, indices],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();

        // A dynamically shaped operand reaches the mixed member rule, never the homogeneous array rule: the composite
        // rule intercepts it and delegates only fully static operands downward. That routing is what keeps the
        // homogeneous `gather` and `slice` transpose rules static-only, and it is observable in the residual
        // signature, because retaining a runtime extent as a first-class dimension is something the homogeneous rule
        // cannot express. The tangent boundary is therefore the operand tangent followed by the indices and that
        // extent.
        assert_eq!(linearization.residual_count(), 2);
        assert_eq!(
            linearization.tangent().input_types(),
            &[
                input_type.tangent().unwrap().into(),
                indices_type.clone().into(),
                ArrayIrType::Dimension(DimensionType::new(extent)),
            ],
        );
        assert!(linearization.tangent().to_string().contains("linear_call [residual_count=2]"));
        let indices = ArrayIrValue::Array(Array::from_elements::<i32>(indices_type, &[1, 1, 3]).unwrap());
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0]).unwrap()), indices])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayIrValue::Array(Array::vector(vec![20.0_f64, 20.0, 40.0]).unwrap()));
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap())];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![2.0_f64, 2.0, 4.0]).unwrap())]),
        );
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::vector(vec![2.0_f64, 3.0, 5.0]).unwrap())];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![0.0_f64, 5.0, 0.0, 5.0]).unwrap())]),
        );
    }

    #[test]
    fn test_array_ir_scatter_differentiation() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(4, Some(7)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let indices_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2), Dimension::Static(1)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let indices = builder.add_input(indices_type.clone().into());
        let updates = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)])).into());
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Scatter(ScatterOperation::new(
                    ScatterDimensionNumbers::new(Vec::new(), vec![0], vec![0]),
                    ScatterReductionKind::Add,
                ))),
                Vec::new(),
                vec![input, indices, updates],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();

        assert_eq!(linearization.residual_count(), 1);
        let indices = ArrayIrValue::Array(Array::from_elements::<i32>(indices_type, &[1, 3]).unwrap());
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap()),
                indices,
                ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0]).unwrap()),
            ])
            .unwrap();
        assert_eq!(primal_outputs[0], ArrayIrValue::Array(Array::vector(vec![1.0_f64, 12.0, 3.0, 24.0]).unwrap()));
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![
            ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap()),
            ArrayIrValue::Array(Array::vector(vec![5.0_f64, 6.0]).unwrap()),
        ];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 7.0, 3.0, 10.0]).unwrap())]),
        );
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0]).unwrap())];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![20.0_f64, 40.0]).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_array_ir_dynamic_scatter_disconnected_operand_tangent_uses_runtime_extent_residuals() {
        // Mixed scatter materializes a structurally zero operand tangent through the residual protocol, using the
        // operand primal as the runtime source for each symbolic extent omitted by its tangent type.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(4, Some(7)).unwrap());
        let extent_type = DimensionType::new(extent);
        let indices_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2), Dimension::Static(1)]));
        let updates_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let padded_extent = builder.add_input(extent_type.clone().into());
        let indices = builder.add_input(indices_type.clone().into());
        let updates = builder.add_input(updates_type.into());

        // The reshaped operand is a static nullary one constant, so its tangent is a structural zero of a static type
        // that no rule needs to materialize. Mixed reshape then carries that zero tangent into a structural zero of its
        // own output type with a symbolic extent, which is exactly the disconnected dynamic operand tangent the
        // scattered operand receives. Its primal carries the required runtime extent and the update tangent stays
        // live, so the rule must materialize a concrete operand tangent through the residual protocol before staging
        // tangent scatter.
        let ones = builder
            .add_instruction(
                ArrayOperation::One(OneOperation::new(ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![Dimension::Static(4)]),
                ))),
                Vec::new(),
                Vec::new(),
                None,
            )
            .unwrap()[0];
        let operand = builder
            .add_instruction(DynamicReshapeOperation::new(), Vec::new(), vec![ones, padded_extent], None)
            .unwrap()[0];
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Scatter(ScatterOperation::new(
                    ScatterDimensionNumbers::new(Vec::new(), vec![0], vec![0]),
                    ScatterReductionKind::Add,
                ))),
                Vec::new(),
                vec![operand, indices, updates],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.interpret(vec![
                ArrayIrValue::Dimension(DimensionValue::new(extent_type, 4).unwrap()),
                ArrayIrValue::Array(Array::from_elements::<i32>(indices_type, &[1, 3]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![10.0_f64, 20.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0]).unwrap()),
            ]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 11.0, 1.0, 21.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![0.0_f64, 1.0, 0.0, 2.0]).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_array_gather() {
        // Gather rows 2 and 0 of a 3x2 matrix.
        let operand = Array::matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let indices = Array::matrix(2, 1, vec![2i64, 0]).unwrap();
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![1], vec![0], vec![0]), vec![1, 2]);
        let gathered = operand.gather(&indices, &operation).unwrap();
        assert_eq!(gathered.r#type().into_owned(), ArrayType::new_static(DataType::F64, [2, 2]));
        assert_eq!(gathered.to_f64s(), vec![5.0, 6.0, 1.0, 2.0]);

        // In-bounds and clipping modes do not materialize an unused zero fill, so they work for formats that cannot
        // represent zero.
        let operand = Array::new(ArrayType::new_static(DataType::F8E8M0FNU, [2]), vec![0x7f, 0x80]).unwrap();
        let indices = Array::matrix(1, 1, vec![1i64]).unwrap();
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![], vec![0], vec![0]), vec![1]);
        assert_eq!(
            operand.gather(&indices, &operation),
            Array::new(ArrayType::new_static(DataType::F8E8M0FNU, [1]), vec![0x80])
        );

        // Gather reads both a reversed operand and reversed sub-byte indices through their physical addressing. An
        // out-of-bounds query in fill-or-drop mode writes the element type's zero encoding into the dense result.
        let operand_type =
            ArrayType::new_static(DataType::U16, [3]).with_layout(Layout::Strided(StridedLayout::new(vec![-2])));
        let operand = Array::from_elements(operand_type, &[10u16, 20, 30]).unwrap();
        let indices_type =
            ArrayType::new_static(DataType::I4, [3, 1]).with_layout(Layout::Strided(StridedLayout::new(vec![-1, 1])));
        let indices =
            Array::from_elements(indices_type, &[i4::new(2).unwrap(), i4::new(-1).unwrap(), i4::new(1).unwrap()])
                .unwrap();
        let operation = GatherOperation::new(GatherDimensionNumbers::new(vec![], vec![0], vec![0]), vec![1])
            .with_mode(GatherScatterMode::FillOrDrop);
        let gathered = operand.gather(&indices, &operation).unwrap();
        assert_eq!(gathered.elements::<u16>(), Ok(vec![30, 0, 20]));
        assert_eq!(gathered.storage_bytes(), [30, 0, 0, 0, 20, 0]);
    }

    #[test]
    fn test_array_scatter() {
        // Scatter-add updates 10 and 20 into elements 3 and 0 of a vector.
        let operand = Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let indices = Array::from_elements::<i64>(ArrayType::new_static(DataType::I64, [2, 1]), &[3, 0]).unwrap();
        let updates = Array::vector(vec![10.0, 20.0]).unwrap();
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), ScatterReductionKind::Add);
        let scattered = operand.scatter(&indices, &updates, &operation).unwrap();
        assert_eq!(scattered, Array::vector(vec![21.0, 2.0, 3.0, 14.0]).unwrap());

        // Scatter decodes sub-byte indices through their physical layout without materializing a scalar index vector.
        let indices_type =
            ArrayType::new_static(DataType::I4, [2, 1]).with_layout(Layout::Strided(StridedLayout::new(vec![-1, 1])));
        let indices = Array::from_elements(indices_type, &[i4::new(3).unwrap(), i4::new(0).unwrap()]).unwrap();
        assert_eq!(
            operand.scatter(&indices, &updates, &operation).unwrap(),
            Array::vector(vec![21.0, 2.0, 3.0, 14.0]).unwrap(),
        );

        // Operand and update payloads are decoded and written through their independent physical layouts.
        let operand_type =
            ArrayType::new_static(DataType::U16, [4]).with_layout(Layout::Strided(StridedLayout::new(vec![-2])));
        let operand = Array::from_elements(operand_type.clone(), &[1u16, 2, 3, 4]).unwrap();
        let updates_type =
            ArrayType::new_static(DataType::U16, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![-2])));
        let updates = Array::from_elements(updates_type, &[10u16, 20]).unwrap();
        assert_eq!(
            operand.scatter(&indices, &updates, &operation),
            Array::from_elements(operand_type, &[21u16, 2, 3, 14]),
        );

        // Sub-byte arithmetic wraps in the declared bit width, including repeated modular addition.
        let operand = Array::vector(vec![i4::new(7).unwrap(), i4::new(-8).unwrap()]).unwrap();
        let indices = Array::matrix(2, 1, vec![0i32, 1]).unwrap();
        let updates = Array::vector(vec![i4::new(2).unwrap(), i4::new(-3).unwrap()]).unwrap();
        assert_eq!(
            operand.scatter(&indices, &updates, &operation).unwrap().elements::<i4>(),
            Ok(vec![i4::new(-7).unwrap(), i4::new(5).unwrap()]),
        );

        // Overwrite moves encodings without requiring arithmetic identities, including for formats without zero.
        let operand = Array::new(ArrayType::new_static(DataType::F8E8M0FNU, [2]), vec![0x7f, 0x80]).unwrap();
        let updates = Array::new(ArrayType::new_static(DataType::F8E8M0FNU, [1]), vec![0x81]).unwrap();
        let indices = Array::matrix(1, 1, vec![0i32]).unwrap();
        let operation = ScatterOperation::new(
            ScatterDimensionNumbers::new(vec![], vec![0], vec![0]),
            ScatterReductionKind::Overwrite,
        );
        assert_eq!(
            operand.scatter(&indices, &updates, &operation),
            Array::new(ArrayType::new_static(DataType::F8E8M0FNU, [2]), vec![0x81, 0x80]),
        );

        // Extrema follow JAX for floating-point NaNs and signed zero and for lexicographically ordered complex values.
        let indices = Array::matrix(2, 1, vec![0i32, 1]).unwrap();
        let operand = Array::vector(vec![f32::NAN, -0.0]).unwrap();
        let updates = Array::vector(vec![1.0f32, 0.0]).unwrap();
        let maximum =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), ScatterReductionKind::Max);
        let minimum =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), ScatterReductionKind::Min);
        let maximum_values = operand.scatter(&indices, &updates, &maximum).unwrap().elements::<f32>().unwrap();
        assert!(maximum_values[0].is_nan());
        assert_eq!(maximum_values[1].to_bits(), 0.0f32.to_bits());
        let minimum_values = operand.scatter(&indices, &updates, &minimum).unwrap().elements::<f32>().unwrap();
        assert!(minimum_values[0].is_nan());
        assert_eq!(minimum_values[1].to_bits(), (-0.0f32).to_bits());

        let operand = Array::vector(vec![ComplexNumber::new(1.0f32, 9.0), ComplexNumber::new(2.0, -1.0)]).unwrap();
        let updates = Array::vector(vec![ComplexNumber::new(1.0f32, 10.0), ComplexNumber::new(1.0, 100.0)]).unwrap();
        assert_eq!(
            operand.scatter(&indices, &updates, &maximum).unwrap().elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(1.0, 10.0), ComplexNumber::new(2.0, -1.0)]),
        );
        assert_eq!(
            operand.scatter(&indices, &updates, &minimum).unwrap().elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(1.0, 9.0), ComplexNumber::new(1.0, 100.0)]),
        );
    }
}
